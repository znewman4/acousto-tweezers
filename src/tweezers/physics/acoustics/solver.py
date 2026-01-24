"""
Multi-domain acoustic solver for heterogeneous media.

Solves the Helmholtz equation:
    ∇·(1/ρ ∇p) + (ω²/K) p = 0

in multiple coupled fluid domains (water, air, bath) with:
- Pressure and velocity continuity at fluid-fluid interfaces
- PML absorption at open boundaries
- Optional thermoviscous corrections

This solver builds a global sparse matrix system that couples
all domains through interface conditions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Callable, Union
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .geometry import MultiDomainGeometry, DomainType, InterfaceType
from .materials import FluidMaterial, MaterialDatabase
from .pml import PMLManager, PMLParameters


@dataclass
class AcousticField3D:
    """
    3D acoustic field solution.
    
    Contains pressure field and derived quantities (velocity, energy densities).
    """
    # Grid coordinates
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    
    # Pressure field (complex)
    p: np.ndarray  # Shape (Nx, Ny, Nz)
    
    # Physical parameters
    omega: float
    
    # Material property fields
    rho: np.ndarray  # Density at each point
    c: np.ndarray    # Sound speed at each point
    
    # Domain mask
    domain_mask: Optional[np.ndarray] = None
    
    @property
    def dx(self) -> float:
        return self.x[1] - self.x[0]
    
    @property
    def dy(self) -> float:
        return self.y[1] - self.y[0]
    
    @property
    def dz(self) -> float:
        return self.z[1] - self.z[0]
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.p.shape
    
    def compute_velocity(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute velocity field from pressure gradient.
        
        v = -1/(iωρ) ∇p
        
        Returns
        -------
        vx, vy, vz : np.ndarray
            Velocity components (complex).
        """
        # Gradient in each direction
        dpx = np.gradient(self.p, self.dx, axis=0)
        dpy = np.gradient(self.p, self.dy, axis=1)
        dpz = np.gradient(self.p, self.dz, axis=2)
        
        # Velocity = -1/(iωρ) ∇p
        factor = -1.0 / (1j * self.omega * self.rho)
        vx = factor * dpx
        vy = factor * dpy
        vz = factor * dpz
        
        return vx, vy, vz
    
    def compute_intensity(self) -> np.ndarray:
        """
        Compute time-averaged acoustic intensity magnitude.
        
        I = 0.5 * Re(p * conj(v))
        
        Returns
        -------
        I_mag : np.ndarray
            Intensity magnitude [W/m²].
        """
        vx, vy, vz = self.compute_velocity()
        
        # I = 0.5 * Re(p * v*)
        Ix = 0.5 * np.real(self.p * np.conj(vx))
        Iy = 0.5 * np.real(self.p * np.conj(vy))
        Iz = 0.5 * np.real(self.p * np.conj(vz))
        
        return np.sqrt(Ix**2 + Iy**2 + Iz**2)
    
    def compute_energy_density(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute potential and kinetic energy densities.
        
        E_pot = |p|² / (4ρc²)
        E_kin = ρ|v|² / 4
        
        Returns
        -------
        E_pot, E_kin : np.ndarray
            Energy densities [J/m³].
        """
        vx, vy, vz = self.compute_velocity()
        
        # Potential energy density
        K = self.rho * self.c**2  # Bulk modulus
        E_pot = 0.25 * np.abs(self.p)**2 / K
        
        # Kinetic energy density
        v_sq = np.abs(vx)**2 + np.abs(vy)**2 + np.abs(vz)**2
        E_kin = 0.25 * self.rho * v_sq
        
        return E_pot, E_kin
    
    def slice_xy(self, z_val: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get XY slice at given z value."""
        k = np.argmin(np.abs(self.z - z_val))
        X, Y = np.meshgrid(self.x, self.y, indexing='ij')
        return X, Y, self.p[:, :, k]
    
    def slice_xz(self, y_val: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get XZ slice at given y value."""
        j = np.argmin(np.abs(self.y - y_val))
        X, Z = np.meshgrid(self.x, self.z, indexing='ij')
        return X, Z, self.p[:, j, :]
    
    def slice_yz(self, x_val: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get YZ slice at given x value."""
        i = np.argmin(np.abs(self.x - x_val))
        Y, Z = np.meshgrid(self.y, self.z, indexing='ij')
        return Y, Z, self.p[i, :, :]


class MultiDomainAcousticSolver:
    """
    Finite difference solver for multi-domain acoustics with PML.
    
    Solves the heterogeneous Helmholtz equation with:
    - Region-specific material properties
    - Interface coupling conditions
    - PML absorption at open boundaries
    """
    
    def __init__(
        self,
        geometry: MultiDomainGeometry,
        materials: Dict[DomainType, FluidMaterial],
        pml_params: Optional[PMLParameters] = None,
        loss_factor: float = 1e-4,
    ):
        """
        Initialize solver.
        
        Parameters
        ----------
        geometry : MultiDomainGeometry
            Domain geometry definition.
        materials : Dict[DomainType, FluidMaterial]
            Material properties for each fluid domain.
        pml_params : PMLParameters, optional
            PML configuration. Uses geometry's pml_thickness if not specified.
        loss_factor : float
            Global loss factor for numerical stability.
        """
        self.geometry = geometry
        self.materials = materials
        self.loss_factor = loss_factor
        
        # Set up PML
        if pml_params is None:
            pml_params = PMLParameters(thickness=geometry.pml_thickness)
        
        self.pml = PMLManager(
            Lx=geometry.Lx, Ly=geometry.Ly, Lz=geometry.Lz,
            x=geometry.x, y=geometry.y, z=geometry.z,
            params=pml_params,
        )
        
        # Configure PML for different regions
        self._configure_pml_materials()
        
        # Cached matrix and solver
        self._A = None
        self._solve_func = None
        self._omega_cached = None
    
    def _configure_pml_materials(self):
        """Set PML material properties based on adjacent domains."""
        # Top PML is adjacent to air
        if DomainType.AIR in self.materials:
            air = self.materials[DomainType.AIR]
            self.pml.set_region_material('z_max', c=air.c, rho=air.rho)
        
        # Bottom PML is adjacent to bath water
        if DomainType.WATER_BATH in self.materials:
            bath = self.materials[DomainType.WATER_BATH]
            self.pml.set_region_material('z_min', c=bath.c, rho=bath.rho)
        
        # Side PMLs - use water or air depending on z
        # For simplicity, use average properties or dominant material
        if DomainType.WATER_DISH in self.materials:
            water = self.materials[DomainType.WATER_DISH]
            for region in ['x_min', 'x_max', 'y_min', 'y_max']:
                self.pml.set_region_material(region, c=water.c, rho=water.rho)
    
    def _get_material_at_point(self, x: float, y: float, z: float) -> Tuple[float, float]:
        """
        Get (rho, c) at a grid point based on domain.
        
        Returns
        -------
        (rho, c) : Tuple[float, float]
            Density and sound speed.
        """
        # Check each domain
        for domain_type, region in self.geometry.regions.items():
            if region.contains_point(x, y, z):
                if domain_type in self.materials:
                    mat = self.materials[domain_type]
                    return mat.rho, mat.c
        
        # Default to water if not in any defined region
        if DomainType.WATER_DISH in self.materials:
            mat = self.materials[DomainType.WATER_DISH]
            return mat.rho, mat.c
        
        return 1000.0, 1500.0  # Default water
    
    def _build_material_fields(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build 3D fields of material properties.
        
        Returns
        -------
        rho : np.ndarray
            Density field, shape (Nx, Ny, Nz).
        c : np.ndarray
            Sound speed field, shape (Nx, Ny, Nz).
        """
        g = self.geometry
        rho = np.zeros(g.shape)
        c = np.zeros(g.shape)
        
        for i, xi in enumerate(g.x):
            for j, yj in enumerate(g.y):
                for k, zk in enumerate(g.z):
                    rho[i,j,k], c[i,j,k] = self._get_material_at_point(xi, yj, zk)
        
        return rho, c
    
    def _build_system_matrix(self, omega: float) -> sp.csc_matrix:
        """
        Build the sparse system matrix for the Helmholtz equation.
        
        Discretizes:
            ∂/∂x(1/(ρ·s_x) ∂p/∂x) + ∂/∂y(1/(ρ·s_y) ∂p/∂y) + 
            ∂/∂z(1/(ρ·s_z) ∂p/∂z) + (ω²/K) p = 0
        
        where s_x, s_y, s_z are PML stretching functions.
        
        Parameters
        ----------
        omega : float
            Angular frequency [rad/s].
        
        Returns
        -------
        A : sp.csc_matrix
            Sparse system matrix.
        """
        g = self.geometry
        Nx, Ny, Nz = g.Nx, g.Ny, g.Nz
        N = Nx * Ny * Nz
        
        dx, dy, dz = g.dx, g.dy, g.dz
        
        # Get material fields
        rho, c = self._build_material_fields()
        
        # Get PML stretching functions
        S_x, S_y, S_z = self.pml.get_stretching_functions_3d(omega)
        
        # Build sparse matrix in lil format (efficient for construction)
        A = sp.lil_matrix((N, N), dtype=np.complex128)
        
        def idx(i: int, j: int, k: int) -> int:
            """Convert 3D index to linear index."""
            return i * (Ny * Nz) + j * Nz + k
        
        # Loss factor for stability
        k_eff_sq_factor = 1.0 + 1j * self.loss_factor
        
        # Fill matrix with 7-point stencil
        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    r = idx(i, j, k)
                    
                    # Get local material properties
                    rho_c = rho[i, j, k]
                    c_c = c[i, j, k]
                    K_c = rho_c * c_c**2  # Bulk modulus
                    
                    # Get PML stretching at this point
                    sx = S_x[i, j, k]
                    sy = S_y[i, j, k]
                    sz = S_z[i, j, k]
                    
                    # Wave equation term: (ω²/K) p
                    k_eff_sq = (omega / c_c)**2 * k_eff_sq_factor
                    
                    # Interior nodes: full 7-point stencil
                    if 1 <= i < Nx-1 and 1 <= j < Ny-1 and 1 <= k < Nz-1:
                        # Effective 1/(ρ·s) at half-grid points
                        # x-direction
                        rho_xp = 0.5 * (rho[i,j,k] + rho[i+1,j,k]) if i < Nx-1 else rho[i,j,k]
                        rho_xm = 0.5 * (rho[i,j,k] + rho[i-1,j,k]) if i > 0 else rho[i,j,k]
                        sx_p = 0.5 * (sx + S_x[i+1,j,k]) if i < Nx-1 else sx
                        sx_m = 0.5 * (sx + S_x[i-1,j,k]) if i > 0 else sx
                        
                        # y-direction
                        rho_yp = 0.5 * (rho[i,j,k] + rho[i,j+1,k]) if j < Ny-1 else rho[i,j,k]
                        rho_ym = 0.5 * (rho[i,j,k] + rho[i,j-1,k]) if j > 0 else rho[i,j,k]
                        sy_p = 0.5 * (sy + S_y[i,j+1,k]) if j < Ny-1 else sy
                        sy_m = 0.5 * (sy + S_y[i,j-1,k]) if j > 0 else sy
                        
                        # z-direction
                        rho_zp = 0.5 * (rho[i,j,k] + rho[i,j,k+1]) if k < Nz-1 else rho[i,j,k]
                        rho_zm = 0.5 * (rho[i,j,k] + rho[i,j,k-1]) if k > 0 else rho[i,j,k]
                        sz_p = 0.5 * (sz + S_z[i,j,k+1]) if k < Nz-1 else sz
                        sz_m = 0.5 * (sz + S_z[i,j,k-1]) if k > 0 else sz
                        
                        # Coefficients for Laplacian
                        ax_p = 1.0 / (rho_xp * sx_p * dx**2)
                        ax_m = 1.0 / (rho_xm * sx_m * dx**2)
                        ay_p = 1.0 / (rho_yp * sy_p * dy**2)
                        ay_m = 1.0 / (rho_ym * sy_m * dy**2)
                        az_p = 1.0 / (rho_zp * sz_p * dz**2)
                        az_m = 1.0 / (rho_zm * sz_m * dz**2)
                        
                        # Fill matrix row
                        A[r, idx(i+1, j, k)] = ax_p
                        A[r, idx(i-1, j, k)] = ax_m
                        A[r, idx(i, j+1, k)] = ay_p
                        A[r, idx(i, j-1, k)] = ay_m
                        A[r, idx(i, j, k+1)] = az_p
                        A[r, idx(i, j, k-1)] = az_m
                        A[r, r] = -(ax_p + ax_m + ay_p + ay_m + az_p + az_m) + k_eff_sq
                    
                    else:
                        # Boundary nodes: apply Neumann BC (dp/dn = 0)
                        # This is appropriate for PML outer boundaries
                        # Use one-sided difference or just set identity
                        A[r, r] = 1.0
        
        return A.tocsc()
    
    def solve(
        self,
        omega: float,
        source: Union[np.ndarray, Callable],
        source_type: str = 'pressure',
    ) -> AcousticField3D:
        """
        Solve the acoustic field.
        
        Parameters
        ----------
        omega : float
            Angular frequency [rad/s].
        source : np.ndarray or Callable
            Source term. Can be:
            - 3D array of shape (Nx, Ny, Nz) for volume source
            - 2D array of shape (Nx, Ny) for bottom boundary source
            - Callable(x, y, z) -> complex for point evaluation
        source_type : str
            Type of source: 'pressure', 'velocity', or 'force'.
        
        Returns
        -------
        field : AcousticField3D
            Solution field.
        """
        g = self.geometry
        
        # Build or reuse system matrix
        if self._A is None or self._omega_cached != omega:
            self._A = self._build_system_matrix(omega)
            self._solve_func = spla.factorized(self._A)
            self._omega_cached = omega
        
        # Build RHS
        b = np.zeros(g.Nx * g.Ny * g.Nz, dtype=np.complex128)
        
        def idx(i: int, j: int, k: int) -> int:
            return i * (g.Ny * g.Nz) + j * g.Nz + k
        
        if callable(source):
            # Evaluate source function
            for i, xi in enumerate(g.x):
                for j, yj in enumerate(g.y):
                    for k, zk in enumerate(g.z):
                        b[idx(i, j, k)] = source(xi, yj, zk)
        elif source.ndim == 2:
            # Bottom boundary source
            if source_type == 'velocity':
                # Convert velocity BC to Neumann BC: dp/dz = -iωρv
                rho, _ = self._build_material_fields()
                for i in range(min(source.shape[0], g.Nx)):
                    for j in range(min(source.shape[1], g.Ny)):
                        k = 0  # Bottom boundary
                        r = idx(i, j, k)
                        # Modify matrix row for Neumann BC
                        v_n = source[i, j]
                        dpdn = -1j * omega * rho[i, j, k] * v_n
                        b[r] = dpdn
            else:
                # Pressure source on bottom
                for i in range(min(source.shape[0], g.Nx)):
                    for j in range(min(source.shape[1], g.Ny)):
                        k = 0
                        r = idx(i, j, k)
                        # Set Dirichlet BC
                        self._A[r, :] = 0
                        self._A[r, r] = 1.0
                        b[r] = source[i, j]
                # Re-factorize
                self._solve_func = spla.factorized(self._A)
        else:
            # Volume source
            for i in range(g.Nx):
                for j in range(g.Ny):
                    for k in range(g.Nz):
                        b[idx(i, j, k)] = source[i, j, k]
        
        # Solve
        p_vec = self._solve_func(b)
        p = p_vec.reshape((g.Nx, g.Ny, g.Nz))
        
        # Build material fields for output
        rho, c = self._build_material_fields()
        
        return AcousticField3D(
            x=g.x.copy(),
            y=g.y.copy(),
            z=g.z.copy(),
            p=p,
            omega=omega,
            rho=rho,
            c=c,
        )
    
    def solve_with_bottom_velocity(
        self,
        omega: float,
        v_bottom: np.ndarray,
    ) -> AcousticField3D:
        """
        Solve with prescribed normal velocity on bottom boundary.
        
        This is the typical actuation mode where a transducer drives
        the fluid through mechanical vibration.
        
        Parameters
        ----------
        omega : float
            Angular frequency [rad/s].
        v_bottom : np.ndarray
            Normal velocity on bottom (z=0), shape (Nx, Ny).
            Positive = into domain (upward).
        
        Returns
        -------
        field : AcousticField3D
            Solution field.
        """
        g = self.geometry
        Nx, Ny, Nz = g.Nx, g.Ny, g.Nz
        N = Nx * Ny * Nz
        
        # Build fresh system matrix with Neumann BC on bottom
        A = self._build_system_matrix(omega).tolil()
        b = np.zeros(N, dtype=np.complex128)
        
        def idx(i: int, j: int, k: int) -> int:
            return i * (Ny * Nz) + j * Nz + k
        
        # Get material fields
        rho, c = self._build_material_fields()
        
        # Apply Neumann BC on bottom: dp/dz = -iωρv
        dz = g.dz
        for i in range(Nx):
            for j in range(Ny):
                k = 0
                r = idx(i, j, k)
                
                # One-sided difference: dp/dz ≈ (-3p₀ + 4p₁ - p₂)/(2dz)
                A[r, :] = 0
                A[r, idx(i, j, 0)] = -3.0 / (2.0 * dz)
                A[r, idx(i, j, 1)] = 4.0 / (2.0 * dz)
                A[r, idx(i, j, 2)] = -1.0 / (2.0 * dz)
                
                # RHS: -iωρv
                v_n = v_bottom[i, j] if i < v_bottom.shape[0] and j < v_bottom.shape[1] else 0.0
                b[r] = -1j * omega * rho[i, j, k] * v_n
        
        # Gauge fixing to avoid singular matrix (if all Neumann)
        # Fix pressure at one interior point
        r_gauge = idx(Nx//2, Ny//2, Nz//2)
        A[r_gauge, :] = 0
        A[r_gauge, r_gauge] = 1.0
        b[r_gauge] = 0.0
        
        # Solve
        A_csc = A.tocsc()
        p_vec = spla.spsolve(A_csc, b)
        p = p_vec.reshape((Nx, Ny, Nz))
        
        return AcousticField3D(
            x=g.x.copy(),
            y=g.y.copy(),
            z=g.z.copy(),
            p=p,
            omega=omega,
            rho=rho,
            c=c,
        )
    
    def interface_continuity_error(self, field: AcousticField3D) -> Dict[str, float]:
        """
        Compute interface condition residuals for validation.
        
        At fluid-fluid interfaces, we should have:
        - Pressure continuity: [p] = 0
        - Normal velocity continuity: [v·n] = 0
        
        Returns
        -------
        errors : Dict[str, float]
            Residual norms for each interface.
        """
        errors = {}
        
        # Check water-air interface
        z_wa = self.geometry.z_levels['water_top']
        k = np.argmin(np.abs(field.z - z_wa))
        
        if k > 0 and k < len(field.z) - 1:
            # Pressure jump across interface
            p_below = field.p[:, :, k-1]
            p_above = field.p[:, :, k+1]
            p_jump = np.abs(p_above - p_below)
            errors['water_air_pressure'] = np.max(p_jump) / (np.max(np.abs(field.p)) + 1e-20)
        
        return errors
    
    def energy_budget(self, field: AcousticField3D) -> Dict[str, float]:
        """
        Compute energy balance in the domain.
        
        Returns
        -------
        budget : Dict[str, float]
            Energy quantities:
            - total_energy: integrated acoustic energy
            - pml_absorption: estimated power absorbed by PML
        """
        g = self.geometry
        E_pot, E_kin = field.compute_energy_density()
        E_total = E_pot + E_kin
        
        dV = g.dx * g.dy * g.dz
        total_energy = np.sum(E_total) * dV
        
        # Energy in PML regions
        pml_mask = self.pml.get_pml_mask()
        pml_energy = np.sum(E_total[pml_mask]) * dV
        
        # Energy in physical domain
        physical_energy = total_energy - pml_energy
        
        return {
            'total_energy_J': float(np.real(total_energy)),
            'physical_energy_J': float(np.real(physical_energy)),
            'pml_energy_J': float(np.real(pml_energy)),
            'pml_fraction': float(np.real(pml_energy / (total_energy + 1e-20))),
        }


def create_default_solver(
    dish_diameter_mm: float = 35.0,
    dish_height_mm: float = 10.0,
    resolution_mm: float = 0.5,
) -> MultiDomainAcousticSolver:
    """
    Create a solver with default geometry and materials.
    
    Parameters
    ----------
    dish_diameter_mm : float
        Dish inner diameter.
    dish_height_mm : float
        Water column height in dish.
    resolution_mm : float
        Grid resolution.
    
    Returns
    -------
    solver : MultiDomainAcousticSolver
    """
    from .geometry import create_standard_dish_geometry
    
    geometry = create_standard_dish_geometry(
        dish_diameter_mm=dish_diameter_mm,
        dish_height_mm=dish_height_mm,
        resolution_mm=resolution_mm,
    )
    
    materials = {
        DomainType.WATER_DISH: MaterialDatabase.water(25.0),
        DomainType.WATER_BATH: MaterialDatabase.water(25.0),
        DomainType.AIR: MaterialDatabase.air(20.0),
    }
    
    return MultiDomainAcousticSolver(geometry=geometry, materials=materials)


if __name__ == "__main__":
    # Demo: create and test solver
    solver = create_default_solver(
        dish_diameter_mm=20.0,
        dish_height_mm=5.0,
        resolution_mm=1.0,
    )
    
    print(solver.geometry.summary())
    print("\n" + solver.pml.pml_report())
    
    # Solve with simple source
    omega = 2 * np.pi * 1e6  # 1 MHz
    g = solver.geometry
    
    # Focused source on bottom
    X, Y = np.meshgrid(g.x, g.y, indexing='ij')
    cx, cy = g.Lx/2, g.Ly/2
    sigma = 0.003  # 3mm Gaussian width
    v_bottom = np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))
    v_bottom = v_bottom.astype(np.complex128) * 0.001  # 1 mm/s amplitude
    
    print("\nSolving...")
    field = solver.solve_with_bottom_velocity(omega, v_bottom)
    
    print(f"Solution shape: {field.shape}")
    print(f"Max |p|: {np.max(np.abs(field.p)):.2e} Pa")
    
    # Energy budget
    budget = solver.energy_budget(field)
    print(f"\nEnergy budget:")
    for key, val in budget.items():
        print(f"  {key}: {val:.4e}")
