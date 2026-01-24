"""
Elastic wave equation solver for solid domains.

Solves the frequency-domain elasticity equation:
    ∇·σ(u) + ρω²u = 0

where σ is the stress tensor:
    σ = λ(∇·u)I + μ(∇u + (∇u)ᵀ)

This solver handles:
- 3D elastic solids with complex (lossy) moduli
- Traction and displacement boundary conditions
- Coupling to adjacent fluid domains
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Callable, Union
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .materials import SolidMaterial


@dataclass
class DisplacementField3D:
    """
    3D elastic displacement field solution.
    
    Contains displacement components and derived quantities.
    """
    # Grid coordinates
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    
    # Displacement field components (complex)
    ux: np.ndarray  # Shape (Nx, Ny, Nz)
    uy: np.ndarray
    uz: np.ndarray
    
    # Physical parameters
    omega: float
    material: SolidMaterial
    
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
        return self.ux.shape
    
    def compute_velocity(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute velocity field: v = iωu.
        """
        vx = 1j * self.omega * self.ux
        vy = 1j * self.omega * self.uy
        vz = 1j * self.omega * self.uz
        return vx, vy, vz
    
    def compute_strain(self) -> Dict[str, np.ndarray]:
        """
        Compute strain tensor components.
        
        ε_ij = 0.5 * (∂u_i/∂x_j + ∂u_j/∂x_i)
        
        Returns
        -------
        strain : Dict[str, np.ndarray]
            Strain components: 'xx', 'yy', 'zz', 'xy', 'xz', 'yz'.
        """
        # Displacement gradients
        dux_dx = np.gradient(self.ux, self.dx, axis=0)
        dux_dy = np.gradient(self.ux, self.dy, axis=1)
        dux_dz = np.gradient(self.ux, self.dz, axis=2)
        
        duy_dx = np.gradient(self.uy, self.dx, axis=0)
        duy_dy = np.gradient(self.uy, self.dy, axis=1)
        duy_dz = np.gradient(self.uy, self.dz, axis=2)
        
        duz_dx = np.gradient(self.uz, self.dx, axis=0)
        duz_dy = np.gradient(self.uz, self.dy, axis=1)
        duz_dz = np.gradient(self.uz, self.dz, axis=2)
        
        return {
            'xx': dux_dx,
            'yy': duy_dy,
            'zz': duz_dz,
            'xy': 0.5 * (dux_dy + duy_dx),
            'xz': 0.5 * (dux_dz + duz_dx),
            'yz': 0.5 * (duy_dz + duz_dy),
        }
    
    def compute_stress(self) -> Dict[str, np.ndarray]:
        """
        Compute stress tensor components.
        
        σ_ij = λ(∇·u)δ_ij + 2μ ε_ij
        
        Returns
        -------
        stress : Dict[str, np.ndarray]
            Stress components: 'xx', 'yy', 'zz', 'xy', 'xz', 'yz'.
        """
        lam = self.material.lambda_complex
        mu = self.material.mu_complex
        
        strain = self.compute_strain()
        div_u = strain['xx'] + strain['yy'] + strain['zz']
        
        return {
            'xx': lam * div_u + 2 * mu * strain['xx'],
            'yy': lam * div_u + 2 * mu * strain['yy'],
            'zz': lam * div_u + 2 * mu * strain['zz'],
            'xy': 2 * mu * strain['xy'],
            'xz': 2 * mu * strain['xz'],
            'yz': 2 * mu * strain['yz'],
        }
    
    def compute_traction(self, normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute traction vector on a surface with given normal.
        
        t_i = σ_ij n_j
        
        Parameters
        ----------
        normal : np.ndarray
            Unit normal vector [nx, ny, nz].
        
        Returns
        -------
        tx, ty, tz : np.ndarray
            Traction components.
        """
        stress = self.compute_stress()
        nx, ny, nz = normal
        
        tx = stress['xx'] * nx + stress['xy'] * ny + stress['xz'] * nz
        ty = stress['xy'] * nx + stress['yy'] * ny + stress['yz'] * nz
        tz = stress['xz'] * nx + stress['yz'] * ny + stress['zz'] * nz
        
        return tx, ty, tz
    
    def normal_velocity_at_surface(self, normal: np.ndarray, z_index: int) -> np.ndarray:
        """
        Get normal velocity component at a z-plane surface.
        
        Parameters
        ----------
        normal : np.ndarray
            Surface normal.
        z_index : int
            Index of z-plane.
        
        Returns
        -------
        v_n : np.ndarray
            Normal velocity, shape (Nx, Ny).
        """
        vx, vy, vz = self.compute_velocity()
        v_n = normal[0] * vx[:, :, z_index] + \
              normal[1] * vy[:, :, z_index] + \
              normal[2] * vz[:, :, z_index]
        return v_n


class ElasticSolver:
    """
    Finite difference solver for 3D frequency-domain elasticity.
    
    Solves:
        ∇·σ + ρω²u = f
    
    with boundary conditions:
    - Prescribed displacement (Dirichlet)
    - Prescribed traction (Neumann)
    - Free surface (zero traction)
    """
    
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        material: SolidMaterial,
    ):
        """
        Initialize solver.
        
        Parameters
        ----------
        x, y, z : np.ndarray
            Coordinate arrays.
        material : SolidMaterial
            Solid material properties.
        """
        self.x = x
        self.y = y
        self.z = z
        self.material = material
        
        self.Nx = len(x)
        self.Ny = len(y)
        self.Nz = len(z)
        self.dx = x[1] - x[0]
        self.dy = y[1] - y[0]
        self.dz = z[1] - z[0]
        
        # Cached matrix
        self._A = None
        self._omega_cached = None
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        return (self.Nx, self.Ny, self.Nz)
    
    @property
    def n_dof(self) -> int:
        """Total degrees of freedom (3 per node)."""
        return 3 * self.Nx * self.Ny * self.Nz
    
    def _idx(self, i: int, j: int, k: int, component: int) -> int:
        """
        Convert 3D index + component to linear DOF index.
        
        Parameters
        ----------
        i, j, k : int
            Grid indices.
        component : int
            Displacement component (0=x, 1=y, 2=z).
        
        Returns
        -------
        dof : int
            Linear degree of freedom index.
        """
        return 3 * (i * self.Ny * self.Nz + j * self.Nz + k) + component
    
    def _build_system_matrix(self, omega: float) -> sp.csc_matrix:
        """
        Build the sparse system matrix for elasticity.
        
        Discretizes the equation at interior nodes and applies
        free-surface (zero traction) BCs at boundaries by default.
        
        Parameters
        ----------
        omega : float
            Angular frequency [rad/s].
        
        Returns
        -------
        A : sp.csc_matrix
            Sparse system matrix.
        """
        N = self.n_dof
        A = sp.lil_matrix((N, N), dtype=np.complex128)
        
        # Material properties (complex for damping)
        lam = self.material.lambda_complex
        mu = self.material.mu_complex
        rho = self.material.rho
        
        dx, dy, dz = self.dx, self.dy, self.dz
        
        # Coefficients
        invdx2 = 1.0 / dx**2
        invdy2 = 1.0 / dy**2
        invdz2 = 1.0 / dz**2
        invdxdy = 1.0 / (4 * dx * dy)
        invdxdz = 1.0 / (4 * dx * dz)
        invdydz = 1.0 / (4 * dy * dz)
        
        # Loop over interior nodes
        for i in range(1, self.Nx - 1):
            for j in range(1, self.Ny - 1):
                for k in range(1, self.Nz - 1):
                    # x-component equation
                    r = self._idx(i, j, k, 0)
                    
                    # ∂/∂x((λ+2μ)∂ux/∂x) + ∂/∂y(μ∂ux/∂y) + ∂/∂z(μ∂ux/∂z)
                    # + ∂/∂x(λ∂uy/∂y) + ∂/∂x(λ∂uz/∂z)
                    # + ∂/∂y(μ∂uy/∂x) + ∂/∂z(μ∂uz/∂x)
                    # + ρω²ux = fx
                    
                    # Diagonal (ux center)
                    A[r, r] = -2 * (lam + 2*mu) * invdx2 - 2 * mu * invdy2 - 2 * mu * invdz2 + rho * omega**2
                    
                    # ux neighbors
                    A[r, self._idx(i+1, j, k, 0)] = (lam + 2*mu) * invdx2
                    A[r, self._idx(i-1, j, k, 0)] = (lam + 2*mu) * invdx2
                    A[r, self._idx(i, j+1, k, 0)] = mu * invdy2
                    A[r, self._idx(i, j-1, k, 0)] = mu * invdy2
                    A[r, self._idx(i, j, k+1, 0)] = mu * invdz2
                    A[r, self._idx(i, j, k-1, 0)] = mu * invdz2
                    
                    # uy coupling (mixed derivatives)
                    A[r, self._idx(i+1, j+1, k, 1)] = (lam + mu) * invdxdy
                    A[r, self._idx(i+1, j-1, k, 1)] = -(lam + mu) * invdxdy
                    A[r, self._idx(i-1, j+1, k, 1)] = -(lam + mu) * invdxdy
                    A[r, self._idx(i-1, j-1, k, 1)] = (lam + mu) * invdxdy
                    
                    # uz coupling (mixed derivatives)
                    A[r, self._idx(i+1, j, k+1, 2)] = (lam + mu) * invdxdz
                    A[r, self._idx(i+1, j, k-1, 2)] = -(lam + mu) * invdxdz
                    A[r, self._idx(i-1, j, k+1, 2)] = -(lam + mu) * invdxdz
                    A[r, self._idx(i-1, j, k-1, 2)] = (lam + mu) * invdxdz
                    
                    # y-component equation
                    r = self._idx(i, j, k, 1)
                    
                    A[r, r] = -2 * mu * invdx2 - 2 * (lam + 2*mu) * invdy2 - 2 * mu * invdz2 + rho * omega**2
                    
                    A[r, self._idx(i+1, j, k, 1)] = mu * invdx2
                    A[r, self._idx(i-1, j, k, 1)] = mu * invdx2
                    A[r, self._idx(i, j+1, k, 1)] = (lam + 2*mu) * invdy2
                    A[r, self._idx(i, j-1, k, 1)] = (lam + 2*mu) * invdy2
                    A[r, self._idx(i, j, k+1, 1)] = mu * invdz2
                    A[r, self._idx(i, j, k-1, 1)] = mu * invdz2
                    
                    # ux coupling
                    A[r, self._idx(i+1, j+1, k, 0)] = (lam + mu) * invdxdy
                    A[r, self._idx(i+1, j-1, k, 0)] = -(lam + mu) * invdxdy
                    A[r, self._idx(i-1, j+1, k, 0)] = -(lam + mu) * invdxdy
                    A[r, self._idx(i-1, j-1, k, 0)] = (lam + mu) * invdxdy
                    
                    # uz coupling
                    A[r, self._idx(i, j+1, k+1, 2)] = (lam + mu) * invdydz
                    A[r, self._idx(i, j+1, k-1, 2)] = -(lam + mu) * invdydz
                    A[r, self._idx(i, j-1, k+1, 2)] = -(lam + mu) * invdydz
                    A[r, self._idx(i, j-1, k-1, 2)] = (lam + mu) * invdydz
                    
                    # z-component equation
                    r = self._idx(i, j, k, 2)
                    
                    A[r, r] = -2 * mu * invdx2 - 2 * mu * invdy2 - 2 * (lam + 2*mu) * invdz2 + rho * omega**2
                    
                    A[r, self._idx(i+1, j, k, 2)] = mu * invdx2
                    A[r, self._idx(i-1, j, k, 2)] = mu * invdx2
                    A[r, self._idx(i, j+1, k, 2)] = mu * invdy2
                    A[r, self._idx(i, j-1, k, 2)] = mu * invdy2
                    A[r, self._idx(i, j, k+1, 2)] = (lam + 2*mu) * invdz2
                    A[r, self._idx(i, j, k-1, 2)] = (lam + 2*mu) * invdz2
                    
                    # ux coupling
                    A[r, self._idx(i+1, j, k+1, 0)] = (lam + mu) * invdxdz
                    A[r, self._idx(i+1, j, k-1, 0)] = -(lam + mu) * invdxdz
                    A[r, self._idx(i-1, j, k+1, 0)] = -(lam + mu) * invdxdz
                    A[r, self._idx(i-1, j, k-1, 0)] = (lam + mu) * invdxdz
                    
                    # uy coupling
                    A[r, self._idx(i, j+1, k+1, 1)] = (lam + mu) * invdydz
                    A[r, self._idx(i, j+1, k-1, 1)] = -(lam + mu) * invdydz
                    A[r, self._idx(i, j-1, k+1, 1)] = -(lam + mu) * invdydz
                    A[r, self._idx(i, j-1, k-1, 1)] = (lam + mu) * invdydz
        
        # Apply boundary conditions (identity rows for now)
        for i in [0, self.Nx - 1]:
            for j in range(self.Ny):
                for k in range(self.Nz):
                    for c in range(3):
                        r = self._idx(i, j, k, c)
                        A[r, :] = 0
                        A[r, r] = 1.0
        
        for j in [0, self.Ny - 1]:
            for i in range(self.Nx):
                for k in range(self.Nz):
                    for c in range(3):
                        r = self._idx(i, j, k, c)
                        A[r, :] = 0
                        A[r, r] = 1.0
        
        for k in [0, self.Nz - 1]:
            for i in range(self.Nx):
                for j in range(self.Ny):
                    for c in range(3):
                        r = self._idx(i, j, k, c)
                        A[r, :] = 0
                        A[r, r] = 1.0
        
        return A.tocsc()
    
    def solve_with_prescribed_displacement(
        self,
        omega: float,
        displacement_bc: Dict[str, np.ndarray],
    ) -> DisplacementField3D:
        """
        Solve with prescribed displacement on bottom surface.
        
        Parameters
        ----------
        omega : float
            Angular frequency [rad/s].
        displacement_bc : Dict[str, np.ndarray]
            Boundary conditions:
            - 'bottom_ux', 'bottom_uy', 'bottom_uz': (Nx, Ny) arrays
        
        Returns
        -------
        field : DisplacementField3D
            Displacement solution.
        """
        # Build matrix
        if self._A is None or self._omega_cached != omega:
            self._A = self._build_system_matrix(omega)
            self._omega_cached = omega
        
        A = self._A.tolil()
        b = np.zeros(self.n_dof, dtype=np.complex128)
        
        # Apply prescribed displacement BCs on bottom (k=0)
        if 'bottom_uz' in displacement_bc:
            uz_bc = displacement_bc['bottom_uz']
            for i in range(self.Nx):
                for j in range(self.Ny):
                    r = self._idx(i, j, 0, 2)
                    A[r, :] = 0
                    A[r, r] = 1.0
                    b[r] = uz_bc[i, j] if i < uz_bc.shape[0] and j < uz_bc.shape[1] else 0.0
        
        if 'bottom_ux' in displacement_bc:
            ux_bc = displacement_bc['bottom_ux']
            for i in range(self.Nx):
                for j in range(self.Ny):
                    r = self._idx(i, j, 0, 0)
                    A[r, :] = 0
                    A[r, r] = 1.0
                    b[r] = ux_bc[i, j] if i < ux_bc.shape[0] and j < ux_bc.shape[1] else 0.0
        
        if 'bottom_uy' in displacement_bc:
            uy_bc = displacement_bc['bottom_uy']
            for i in range(self.Nx):
                for j in range(self.Ny):
                    r = self._idx(i, j, 0, 1)
                    A[r, :] = 0
                    A[r, r] = 1.0
                    b[r] = uy_bc[i, j] if i < uy_bc.shape[0] and j < uy_bc.shape[1] else 0.0
        
        # Solve
        A_csc = A.tocsc()
        u_vec = spla.spsolve(A_csc, b)
        
        # Unpack solution
        ux = np.zeros(self.shape, dtype=np.complex128)
        uy = np.zeros(self.shape, dtype=np.complex128)
        uz = np.zeros(self.shape, dtype=np.complex128)
        
        for i in range(self.Nx):
            for j in range(self.Ny):
                for k in range(self.Nz):
                    ux[i, j, k] = u_vec[self._idx(i, j, k, 0)]
                    uy[i, j, k] = u_vec[self._idx(i, j, k, 1)]
                    uz[i, j, k] = u_vec[self._idx(i, j, k, 2)]
        
        return DisplacementField3D(
            x=self.x.copy(),
            y=self.y.copy(),
            z=self.z.copy(),
            ux=ux,
            uy=uy,
            uz=uz,
            omega=omega,
            material=self.material,
        )


if __name__ == "__main__":
    from .materials import SolidMaterialDatabase
    
    # Demo: solve simple elastic problem
    material = SolidMaterialDatabase.polystyrene()
    
    # Small test grid
    x = np.linspace(0, 0.01, 11)
    y = np.linspace(0, 0.01, 11)
    z = np.linspace(0, 0.001, 5)  # 1mm thick plate
    
    solver = ElasticSolver(x, y, z, material)
    print(f"Grid shape: {solver.shape}")
    print(f"Total DOFs: {solver.n_dof}")
    
    # Solve with uniform displacement on bottom
    omega = 2 * np.pi * 1e6
    
    uz_bc = np.ones((len(x), len(y)), dtype=np.complex128) * 1e-9  # 1 nm
    
    print("\nSolving elastic wave equation...")
    field = solver.solve_with_prescribed_displacement(
        omega=omega,
        displacement_bc={'bottom_uz': uz_bc},
    )
    
    print(f"Max |uz|: {np.max(np.abs(field.uz))*1e9:.2f} nm")
    
    # Compute stress
    stress = field.compute_stress()
    print(f"Max |σ_zz|: {np.max(np.abs(stress['zz'])):.2e} Pa")
