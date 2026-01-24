"""
FEM acoustics solver for first-order pressure acoustics.

Implements the Helmholtz equation in the frequency domain:

    ∇·(1/ρ ∇p) + ω²/(ρc²) p = 0

in weak form:

    ∫_Ω (1/ρ) ∇φ·∇p dV - ∫_Ω (ω²/K) φ p dV + boundary terms = 0

where K = ρc² is the bulk modulus.

Fluid-fluid interface conditions (from MASTER BRIEF):
- Pressure continuity: p₁ = p₂
- Normal velocity continuity: (1/ρ₁) ∂p₁/∂n = (1/ρ₂) ∂p₂/∂n

These are automatically satisfied through the weak form when using
continuous pressure approximation and proper material property jumps.

References
----------
- Ihlenburg, F. (1998): Finite Element Analysis of Acoustic Scattering
- Kaltenbacher, M. (2015): Numerical Simulation of Mechatronic Sensors
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List, Callable
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

from .domains import DomainType, InterfaceType, Domain
from .geometry import (
    FEMMesh, 
    get_shape_functions_hex8, 
    get_shape_gradients_hex8,
    GAUSS_POINTS_HEX8,
    GAUSS_WEIGHTS_HEX8,
)
from .materials import FluidMaterial, MaterialDatabase
from .config import FEMConfig


@dataclass
class AcousticField:
    """
    Solution of the acoustic field.
    
    Contains pressure and derived quantities.
    
    The velocity field is computed from pressure:
        v = -1/(iωρ) ∇p
    
    Energy densities:
        Potential: Ep = |p|²/(4ρc²)
        Kinetic:   Ek = ρ|v|²/4
    """
    # Grid coordinates
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    
    # Pressure field (complex, nodal values)
    p: np.ndarray  # (num_nodes,) complex
    
    # Frequency
    omega: float
    
    # Material property fields (at nodes)
    rho: np.ndarray  # Density
    c: np.ndarray    # Sound speed
    
    # Mesh reference
    mesh: Optional[FEMMesh] = None
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        """Grid shape for structured mesh."""
        if self.mesh:
            return (self.mesh.nx, self.mesh.ny, self.mesh.nz)
        return (len(self.x), len(self.y), len(self.z))
    
    @property
    def p_grid(self) -> np.ndarray:
        """Pressure reshaped to 3D grid (for structured mesh)."""
        if self.mesh:
            return self.p.reshape((self.mesh.nz, self.mesh.ny, self.mesh.nx)).transpose((2, 1, 0))
        return self.p
    
    @property
    def p_amplitude(self) -> np.ndarray:
        """Pressure amplitude |p|."""
        return np.abs(self.p)
    
    @property
    def p_phase(self) -> np.ndarray:
        """Pressure phase angle."""
        return np.angle(self.p)
    
    def compute_velocity(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute velocity field from pressure gradient.
        
        v = -1/(iωρ) ∇p
        
        Returns
        -------
        vx, vy, vz : np.ndarray
            Velocity components (complex).
        """
        if self.mesh is None:
            raise ValueError("Mesh required for velocity computation")
        
        # For structured mesh, use finite differences
        dx, dy, dz = self.mesh.dx, self.mesh.dy, self.mesh.dz
        
        # Reshape pressure to 3D
        p_3d = self.p_grid
        nx, ny, nz = p_3d.shape
        
        # Central differences (second order)
        dpx = np.zeros_like(p_3d)
        dpy = np.zeros_like(p_3d)
        dpz = np.zeros_like(p_3d)
        
        # Interior points
        dpx[1:-1, :, :] = (p_3d[2:, :, :] - p_3d[:-2, :, :]) / (2 * dx)
        dpy[:, 1:-1, :] = (p_3d[:, 2:, :] - p_3d[:, :-2, :]) / (2 * dy)
        dpz[:, :, 1:-1] = (p_3d[:, :, 2:] - p_3d[:, :, :-2]) / (2 * dz)
        
        # Boundary (one-sided)
        dpx[0, :, :] = (p_3d[1, :, :] - p_3d[0, :, :]) / dx
        dpx[-1, :, :] = (p_3d[-1, :, :] - p_3d[-2, :, :]) / dx
        dpy[:, 0, :] = (p_3d[:, 1, :] - p_3d[:, 0, :]) / dy
        dpy[:, -1, :] = (p_3d[:, -1, :] - p_3d[:, -2, :]) / dy
        dpz[:, :, 0] = (p_3d[:, :, 1] - p_3d[:, :, 0]) / dz
        dpz[:, :, -1] = (p_3d[:, :, -1] - p_3d[:, :, -2]) / dz
        
        # Reshape density to 3D
        rho_3d = self.rho.reshape((nz, ny, nx)).transpose((2, 1, 0))
        
        # v = -∇p / (iωρ)
        factor = -1.0 / (1j * self.omega * rho_3d)
        vx = factor * dpx
        vy = factor * dpy
        vz = factor * dpz
        
        return vx, vy, vz
    
    def compute_energy_densities(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute time-averaged acoustic energy densities.
        
        Potential energy: Ep = ⟨p²⟩/(ρc²) = |p|²/(2ρc²)
        Kinetic energy:   Ek = ρ⟨v²⟩ = ρ|v|²/2
        
        Returns
        -------
        E_pot, E_kin : np.ndarray
            Energy densities [J/m³].
        """
        vx, vy, vz = self.compute_velocity()
        
        # Reshape to 3D
        p_3d = self.p_grid
        nx, ny, nz = p_3d.shape
        rho_3d = self.rho.reshape((nz, ny, nx)).transpose((2, 1, 0))
        c_3d = self.c.reshape((nz, ny, nx)).transpose((2, 1, 0))
        K_3d = rho_3d * c_3d**2
        
        # Time-averaged (factor of 1/4 for harmonic fields)
        E_pot = np.abs(p_3d)**2 / (4 * K_3d)
        E_kin = rho_3d * (np.abs(vx)**2 + np.abs(vy)**2 + np.abs(vz)**2) / 4
        
        return E_pot, E_kin
    
    def compute_intensity(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute time-averaged acoustic intensity.
        
        I = ⟨p·v⟩ = Re(p·v*)/2
        
        Returns
        -------
        Ix, Iy, Iz : np.ndarray
            Intensity components [W/m²].
        """
        vx, vy, vz = self.compute_velocity()
        p_3d = self.p_grid
        
        Ix = 0.5 * np.real(p_3d * np.conj(vx))
        Iy = 0.5 * np.real(p_3d * np.conj(vy))
        Iz = 0.5 * np.real(p_3d * np.conj(vz))
        
        return Ix, Iy, Iz


class FEMAcousticSolver:
    """
    Finite Element Method solver for first-order pressure acoustics.
    
    Solves the Helmholtz equation:
    
        ∇·(1/ρ ∇p) + ω²/K p = 0
    
    using the Galerkin weak form on hexahedral elements.
    
    Boundary conditions:
    - Dirichlet: p = p_D (prescribed pressure)
    - Neumann: (1/ρ) ∂p/∂n = v_n (prescribed normal velocity)
    - Robin: (1/ρ) ∂p/∂n + αp = g (impedance condition)
    
    Interface conditions at fluid-fluid boundaries are automatically
    satisfied by the continuous pressure approximation with jump in ρ.
    """
    
    def __init__(
        self,
        mesh: FEMMesh,
        materials: MaterialDatabase,
        config: FEMConfig,
    ):
        """
        Initialize acoustic solver.
        
        Parameters
        ----------
        mesh : FEMMesh
            Finite element mesh with domain assignments.
        materials : MaterialDatabase
            Material properties.
        config : FEMConfig
            Simulation configuration.
        """
        self.mesh = mesh
        self.materials = materials
        self.config = config
        
        self.omega = config.physics.omega
        self.frequency = config.physics.frequency
        
        # Build material property arrays
        self._build_material_fields()
        
        # System matrices (assembled on demand)
        self._K: Optional[sparse.csr_matrix] = None  # Stiffness
        self._M: Optional[sparse.csr_matrix] = None  # Mass
        self._A: Optional[sparse.csr_matrix] = None  # System matrix
        self._assembled = False
    
    def _build_material_fields(self):
        """Build nodal arrays of material properties."""
        n_nodes = self.mesh.num_nodes
        
        self.rho = np.zeros(n_nodes, dtype=np.float64)
        self.c = np.zeros(n_nodes, dtype=np.float64)
        self.K_bulk = np.zeros(n_nodes, dtype=np.complex128)  # Complex for loss
        
        # Get fluid materials
        water = self.materials.water
        air = self.materials.air
        
        # Assign properties based on domain
        for domain, info in self.mesh.domain_info.items():
            if domain.is_fluid or domain.is_pml:
                if domain.is_water_like or domain == DomainType.PML_WATER or domain == DomainType.PML_BATH:
                    mat = water
                else:
                    mat = air
                
                for node_id in info.node_ids:
                    self.rho[node_id] = mat.rho
                    self.c[node_id] = mat.c
                    self.K_bulk[node_id] = mat.K_complex
        
        # Handle nodes not in any fluid domain (set to water default)
        zero_mask = self.rho == 0
        if np.any(zero_mask):
            self.rho[zero_mask] = water.rho
            self.c[zero_mask] = water.c
            self.K_bulk[zero_mask] = water.K_complex
    
    def set_sources(self, sources: Optional[Dict[str, Any]] = None) -> None:
        """
        Set acoustic sources (e.g., transducers).
        
        Parameters
        ----------
        sources : dict, optional
            Source specification. Currently a placeholder for future expansion.
            Could contain transducer positions, amplitudes, etc.
        """
        self.sources = sources or {}
    
    def assemble_system(self) -> None:
        """
        Assemble global stiffness and mass matrices.
        
        Weak form:
            ∫_Ω (1/ρ) ∇φ·∇p dV = stiffness term → K
            ∫_Ω (1/K) φ p dV = mass term → M
        
        System matrix: A = K - ω²M
        """
        if self._assembled:
            return
        
        n_nodes = self.mesh.num_nodes
        
        # Triplet format for assembly
        rows = []
        cols = []
        K_vals = []
        M_vals = []
        
        # Jacobian for structured mesh (constant)
        J_scale = np.diag([self.mesh.dx / 2, self.mesh.dy / 2, self.mesh.dz / 2])
        J_inv = np.diag([2 / self.mesh.dx, 2 / self.mesh.dy, 2 / self.mesh.dz])
        det_J = (self.mesh.dx * self.mesh.dy * self.mesh.dz) / 8
        
        # Loop over elements
        for e in range(self.mesh.num_elements):
            # Skip solid elements
            domain = DomainType(self.mesh.element_domains[e])
            if domain.is_solid:
                continue
            
            node_ids = self.mesh.elements[e]
            
            # Element material properties (averaged at centroid)
            rho_e = np.mean(self.rho[node_ids])
            K_e = np.mean(self.K_bulk[node_ids])
            
            # Local stiffness and mass matrices (8×8)
            Ke = np.zeros((8, 8), dtype=np.complex128)
            Me = np.zeros((8, 8), dtype=np.complex128)
            
            # Gauss quadrature
            for g, (xi, w) in enumerate(zip(GAUSS_POINTS_HEX8, GAUSS_WEIGHTS_HEX8)):
                N = get_shape_functions_hex8(xi)
                dN_local = get_shape_gradients_hex8(xi)
                
                # Transform gradients to physical coordinates
                dN = dN_local @ J_inv
                
                # Stiffness: ∫ (1/ρ) ∇N·∇N dV
                Ke += (1.0 / rho_e) * (dN @ dN.T) * det_J * w
                
                # Mass: ∫ (1/K) N·N dV
                Me += (1.0 / K_e) * np.outer(N, N) * det_J * w
            
            # Assemble into global
            for i in range(8):
                for j in range(8):
                    rows.append(node_ids[i])
                    cols.append(node_ids[j])
                    K_vals.append(Ke[i, j])
                    M_vals.append(Me[i, j])
        
        # Build sparse matrices
        self._K = sparse.csr_matrix(
            (K_vals, (rows, cols)),
            shape=(n_nodes, n_nodes),
            dtype=np.complex128,
        )
        
        self._M = sparse.csr_matrix(
            (M_vals, (rows, cols)),
            shape=(n_nodes, n_nodes),
            dtype=np.complex128,
        )
        
        # System matrix: A = K - ω²M
        self._A = self._K - self.omega**2 * self._M
        
        self._assembled = True
    
    def apply_boundary_conditions(
        self,
        dirichlet_nodes: Optional[Dict[int, complex]] = None,
        neumann_data: Optional[Callable] = None,
        impedance_bc: Optional[Dict[str, float]] = None,
    ) -> Tuple[sparse.csr_matrix, np.ndarray]:
        """
        Apply boundary conditions to the system.
        
        Parameters
        ----------
        dirichlet_nodes : dict
            Node ID → prescribed pressure value.
        neumann_data : callable
            Function (x, y, z) → normal velocity for Neumann BC.
        impedance_bc : dict
            Parameters for impedance (Robin) BC on outer boundaries.
        
        Returns
        -------
        A_bc : sparse matrix
            System matrix with BCs applied.
        b : array
            Right-hand side vector.
        """
        if not self._assembled:
            self.assemble_system()
        
        A = self._A.tolil()
        b = np.zeros(self.mesh.num_nodes, dtype=np.complex128)
        
        # Apply Dirichlet BCs (strong form)
        if dirichlet_nodes:
            for node_id, p_val in dirichlet_nodes.items():
                A[node_id, :] = 0
                A[node_id, node_id] = 1.0
                b[node_id] = p_val
        
        # Apply Neumann BC (natural - enters RHS)
        # For prescribed normal velocity: ∂p/∂n = iωρ v_n
        if neumann_data:
            # Find boundary nodes and apply
            # This is simplified - full implementation would integrate over faces
            pass
        
        return A.tocsr(), b
    
    def solve(
        self,
        source_field: Optional[np.ndarray] = None,
        bottom_velocity: Optional[np.ndarray] = None,
    ) -> AcousticField:
        """
        Solve the acoustic problem.
        
        Parameters
        ----------
        source_field : np.ndarray, optional
            Volume source distribution (for testing).
        bottom_velocity : np.ndarray, optional
            Normal velocity on bottom boundary (z=z_min).
            Shape should match (nx, ny) grid.
        
        Returns
        -------
        field : AcousticField
            Solution containing pressure and derived quantities.
        """
        self.assemble_system()
        
        # Prepare boundary conditions
        dirichlet_nodes = {}
        
        # Apply actuation via bottom velocity (Neumann BC)
        if bottom_velocity is not None:
            # Convert velocity to pressure gradient
            # ∂p/∂n = -iωρ v_n on bottom (n = -z)
            b = self._apply_bottom_velocity_bc(bottom_velocity)
        else:
            b = np.zeros(self.mesh.num_nodes, dtype=np.complex128)
        
        # Get system with BCs
        A, b_bc = self.apply_boundary_conditions(dirichlet_nodes=dirichlet_nodes)
        b += b_bc
        
        # Add PML damping if enabled
        if self.config.enable_pml:
            A = self._add_pml_terms(A)
        
        # Solve linear system
        if self.config.solver.linear_solver == "direct":
            p = spla.spsolve(A, b)
        else:
            p, info = spla.gmres(
                A, b,
                rtol=self.config.solver.iterative_tol,
                maxiter=self.config.solver.iterative_maxiter,
            )
            if info != 0:
                import warnings
                warnings.warn(f"Iterative solver did not converge (info={info})")
        
        # Build result
        field = AcousticField(
            x=self.mesh.x,
            y=self.mesh.y,
            z=self.mesh.z,
            p=p,
            omega=self.omega,
            rho=self.rho,
            c=self.c,
            mesh=self.mesh,
        )
        
        return field
    
    def _apply_bottom_velocity_bc(
        self,
        v_bottom: np.ndarray,
    ) -> np.ndarray:
        """
        Apply normal velocity BC on bottom boundary.
        
        Neumann condition: (1/ρ) ∂p/∂n = v_n
        
        For bottom face (n = -ẑ), this becomes:
            -(1/ρ) ∂p/∂z = v_n
        
        In weak form, this adds to RHS:
            ∫_Γ v_n φ dA
        """
        b = np.zeros(self.mesh.num_nodes, dtype=np.complex128)
        
        # Bottom nodes are at z = z_min
        z_min = self.mesh.z[0]
        dz = self.mesh.dz
        
        # Find nodes on bottom face
        bottom_mask = np.abs(self.mesh.nodes[:, 2] - z_min) < dz / 10
        bottom_nodes = np.where(bottom_mask)[0]
        
        # Get velocity at each bottom node
        for node_id in bottom_nodes:
            x, y, z = self.mesh.nodes[node_id]
            
            # Find corresponding index in v_bottom array
            ix = int((x - self.mesh.x[0]) / self.mesh.dx + 0.5)
            iy = int((y - self.mesh.y[0]) / self.mesh.dy + 0.5)
            
            # Clamp to valid range
            ix = max(0, min(ix, v_bottom.shape[0] - 1))
            iy = max(0, min(iy, v_bottom.shape[1] - 1))
            
            v_n = v_bottom[ix, iy]
            
            # Contribution: -iωρ v_n × face area contribution
            # Simplified: assume uniform element sizes
            face_area = self.mesh.dx * self.mesh.dy / 4  # Corner contribution
            rho_node = self.rho[node_id]
            
            b[node_id] += -1j * self.omega * rho_node * v_n * face_area
        
        return b
    
    def _add_pml_terms(
        self,
        A: sparse.csr_matrix,
    ) -> sparse.csr_matrix:
        """
        Add PML (Perfectly Matched Layer) terms to system matrix.
        
        PML uses complex coordinate stretching:
            x̃ = x + (i/ω) ∫₀ˣ σ(x') dx'
        
        This modifies the Helmholtz operator to create artificial damping
        in the PML region that absorbs outgoing waves.
        """
        A = A.tolil()
        
        # PML parameters
        pml_thickness = self.config.geometry.pml_thickness
        pml_sigma_max = self.config.geometry.pml_max_sigma * self.omega
        pml_order = self.config.geometry.pml_stretch_order
        
        # Domain boundaries (where PML starts)
        x_pml_min = self.mesh.x[0] + pml_thickness
        x_pml_max = self.mesh.x[-1] - pml_thickness
        y_pml_min = self.mesh.y[0] + pml_thickness
        y_pml_max = self.mesh.y[-1] - pml_thickness
        z_pml_min = self.mesh.z[0] + pml_thickness
        z_pml_max = self.mesh.z[-1] - pml_thickness
        
        def sigma_profile(d: float, L: float) -> complex:
            """PML damping profile."""
            if d <= 0:
                return 0.0
            ratio = (d / L) ** pml_order
            return pml_sigma_max * ratio
        
        def stretch_factor(d: float, L: float) -> complex:
            """Complex coordinate stretching factor."""
            sigma = sigma_profile(d, L)
            return 1.0 + 1j * sigma / self.omega
        
        # Modify diagonal entries in PML region
        for node_id in range(self.mesh.num_nodes):
            x, y, z = self.mesh.nodes[node_id]
            
            # Distance into PML region
            dx_pml = max(x_pml_min - x, x - x_pml_max, 0)
            dy_pml = max(y_pml_min - y, y - y_pml_max, 0)
            dz_pml = max(z_pml_min - z, z - z_pml_max, 0)
            
            if dx_pml > 0 or dy_pml > 0 or dz_pml > 0:
                # Compute stretching factors
                sx = stretch_factor(dx_pml, pml_thickness)
                sy = stretch_factor(dy_pml, pml_thickness)
                sz = stretch_factor(dz_pml, pml_thickness)
                
                # Modify coefficients (simplified - full implementation
                # would modify element matrices during assembly)
                pml_factor = sx * sy * sz
                A[node_id, node_id] *= pml_factor
        
        return A.tocsr()


def solve_acoustics(
    config: FEMConfig,
    mesh: Optional[FEMMesh] = None,
    materials: Optional[MaterialDatabase] = None,
    actuation_pattern: Optional[np.ndarray] = None,
) -> AcousticField:
    """
    High-level interface to solve the acoustic problem.
    
    Parameters
    ----------
    config : FEMConfig
        Simulation configuration.
    mesh : FEMMesh, optional
        Pre-built mesh. Created if not provided.
    materials : MaterialDatabase, optional
        Material properties. Created if not provided.
    actuation_pattern : np.ndarray, optional
        2D array of actuation amplitudes on bottom.
    
    Returns
    -------
    field : AcousticField
        Solution containing pressure field.
    """
    from .geometry import create_petri_dish_mesh
    
    if mesh is None:
        if materials is None:
            materials = MaterialDatabase(temperature=config.physics.temperature)
        mesh = create_petri_dish_mesh(config, materials)
    
    if materials is None:
        materials = MaterialDatabase(temperature=config.physics.temperature)
    
    solver = FEMAcousticSolver(mesh, materials, config)
    
    # Create actuation pattern if not provided
    if actuation_pattern is None:
        nx, ny = mesh.nx, mesh.ny
        actuation_pattern = np.zeros((nx, ny), dtype=np.complex128)
        
        # Default: circular actuator at center
        X, Y = np.meshgrid(mesh.x, mesh.y, indexing='ij')
        R = config.geometry.dish_radius
        r2 = X**2 + Y**2
        actuator_mask = r2 < (0.8 * R)**2
        actuation_pattern[actuator_mask] = config.physics.actuation_amplitude
    
    field = solver.solve(bottom_velocity=actuation_pattern)
    
    return field
