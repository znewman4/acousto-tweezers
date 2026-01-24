"""
Acoustic streaming solver (steady Stokes flow).

Computes the time-averaged (DC) flow induced by acoustic Reynolds stress.

Governing equations (from MASTER BRIEF):

First-order acoustic velocity:
    v₁ = -1/(iωρ) ∇p

Streaming force (Reynolds stress divergence):
    f_stream = -⟨ρ₁ v₁ · ∇v₁ + ρ v₁ · ∇v₁ + ... ⟩

Mean flow (steady Stokes):
    -∇p̄ + η∇²ū + f_stream = 0
    ∇·ū = 0

where p̄ is the mean pressure, ū is the streaming velocity,
and η is the dynamic viscosity.

References
----------
- Nyborg (1958): Acoustic streaming near a boundary
- Lighthill (1978): Acoustic streaming
- Nama et al. (2015): Numerical study of acoustophoretic motion
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

from .domains import DomainType
from .geometry import FEMMesh
from .materials import FluidMaterial, MaterialDatabase
from .config import FEMConfig
from .acoustics import AcousticField


@dataclass
class StreamingField:
    """
    Acoustic streaming velocity field (mean flow).
    """
    # Grid coordinates
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    
    # Mean velocity components (real)
    ux: np.ndarray  # (num_nodes,)
    uy: np.ndarray
    uz: np.ndarray
    
    # Mean pressure
    p_mean: Optional[np.ndarray] = None
    
    # Forcing field (for diagnostics)
    fx: Optional[np.ndarray] = None
    fy: Optional[np.ndarray] = None
    fz: Optional[np.ndarray] = None
    
    # Mesh reference
    mesh: Optional[FEMMesh] = None
    
    @property
    def velocity_magnitude(self) -> np.ndarray:
        """Streaming velocity magnitude |ū|."""
        return np.sqrt(self.ux**2 + self.uy**2 + self.uz**2)
    
    @property
    def max_velocity(self) -> float:
        """Maximum streaming velocity."""
        return float(np.max(self.velocity_magnitude))
    
    def compute_reynolds_number(self, fluid: FluidMaterial, length_scale: float) -> float:
        """
        Compute streaming Reynolds number.
        
        Re = ρ |ū| L / η
        
        Parameters
        ----------
        fluid : FluidMaterial
            Fluid properties.
        length_scale : float
            Characteristic length [m].
        
        Returns
        -------
        Re : float
            Reynolds number.
        """
        u_max = self.max_velocity
        return fluid.rho * u_max * length_scale / fluid.eta


def compute_streaming_force(
    acoustic_field: AcousticField,
    fluid: FluidMaterial,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute acoustic streaming force from first-order field.
    
    The streaming force is the time-averaged Reynolds stress divergence:
    
        f = -⟨ρ₁ ∂v₁/∂t + ρ(v₁·∇)v₁⟩
    
    For harmonic fields with v₁ = Re(v̂ e^{iωt}):
    
        f = -ρ Re(v̂·∇v̂*) / 2
    
    Parameters
    ----------
    acoustic_field : AcousticField
        First-order acoustic pressure field.
    fluid : FluidMaterial
        Fluid properties.
    
    Returns
    -------
    fx, fy, fz : np.ndarray
        Streaming force components [N/m³].
    """
    # Get velocity field
    vx, vy, vz = acoustic_field.compute_velocity()
    
    mesh = acoustic_field.mesh
    dx, dy, dz = mesh.dx, mesh.dy, mesh.dz
    
    # Compute velocity gradients (central differences)
    def grad_x(f):
        g = np.zeros_like(f)
        g[1:-1, :, :] = (f[2:, :, :] - f[:-2, :, :]) / (2 * dx)
        g[0, :, :] = (f[1, :, :] - f[0, :, :]) / dx
        g[-1, :, :] = (f[-1, :, :] - f[-2, :, :]) / dx
        return g
    
    def grad_y(f):
        g = np.zeros_like(f)
        g[:, 1:-1, :] = (f[:, 2:, :] - f[:, :-2, :]) / (2 * dy)
        g[:, 0, :] = (f[:, 1, :] - f[:, 0, :]) / dy
        g[:, -1, :] = (f[:, -1, :] - f[:, -2, :]) / dy
        return g
    
    def grad_z(f):
        g = np.zeros_like(f)
        g[:, :, 1:-1] = (f[:, :, 2:] - f[:, :, :-2]) / (2 * dz)
        g[:, :, 0] = (f[:, :, 1] - f[:, :, 0]) / dz
        g[:, :, -1] = (f[:, :, -1] - f[:, :, -2]) / dz
        return g
    
    # Reynolds stress: -ρ Re(v̂·∇v̂*) / 2
    # f_x = -ρ/2 Re(vx ∂vx*/∂x + vy ∂vx*/∂y + vz ∂vx*/∂z)
    
    rho = fluid.rho
    
    # Gradient of conjugate velocity
    dvx_dx = grad_x(np.conj(vx))
    dvx_dy = grad_y(np.conj(vx))
    dvx_dz = grad_z(np.conj(vx))
    
    dvy_dx = grad_x(np.conj(vy))
    dvy_dy = grad_y(np.conj(vy))
    dvy_dz = grad_z(np.conj(vy))
    
    dvz_dx = grad_x(np.conj(vz))
    dvz_dy = grad_y(np.conj(vz))
    dvz_dz = grad_z(np.conj(vz))
    
    # Streaming force
    fx = -0.5 * rho * np.real(vx * dvx_dx + vy * dvx_dy + vz * dvx_dz)
    fy = -0.5 * rho * np.real(vx * dvy_dx + vy * dvy_dy + vz * dvy_dz)
    fz = -0.5 * rho * np.real(vx * dvz_dx + vy * dvz_dy + vz * dvz_dz)
    
    return fx, fy, fz


class StreamingSolver:
    """
    Solver for acoustic streaming (steady Stokes flow).
    
    Solves the steady Stokes equations:
        -∇p̄ + η∇²ū + f = 0
        ∇·ū = 0
    
    using a penalty method for incompressibility.
    """
    
    def __init__(
        self,
        mesh: FEMMesh,
        materials: MaterialDatabase,
        config: FEMConfig,
    ):
        """
        Initialize streaming solver.
        
        Parameters
        ----------
        mesh : FEMMesh
            Finite element mesh.
        materials : MaterialDatabase
            Material properties.
        config : FEMConfig
            Simulation configuration.
        """
        self.mesh = mesh
        self.materials = materials
        self.config = config
        
        self.fluid = materials.water
        self.eta = self.fluid.eta  # Dynamic viscosity
    
    def solve(
        self,
        acoustic_field: AcousticField,
    ) -> StreamingField:
        """
        Solve for streaming velocity given acoustic field.
        
        Parameters
        ----------
        acoustic_field : AcousticField
            First-order acoustic pressure field.
        
        Returns
        -------
        streaming : StreamingField
            Streaming velocity field.
        """
        # Compute streaming force
        fx, fy, fz = compute_streaming_force(acoustic_field, self.fluid)
        
        # Flatten to nodal arrays
        fx_flat = fx.transpose((2, 1, 0)).flatten()
        fy_flat = fy.transpose((2, 1, 0)).flatten()
        fz_flat = fz.transpose((2, 1, 0)).flatten()
        
        # Build and solve Stokes system
        # Using simplified finite difference approach for velocity-only formulation
        
        n_nodes = self.mesh.num_nodes
        
        # Laplacian operator
        L = self._build_laplacian()
        
        # System: η L u = -f (simplified, neglecting pressure gradient)
        # More accurate would use mixed formulation
        
        ux = spla.spsolve(L, -fx_flat / self.eta)
        uy = spla.spsolve(L, -fy_flat / self.eta)
        uz = spla.spsolve(L, -fz_flat / self.eta)
        
        return StreamingField(
            x=self.mesh.x,
            y=self.mesh.y,
            z=self.mesh.z,
            ux=ux,
            uy=uy,
            uz=uz,
            fx=fx_flat,
            fy=fy_flat,
            fz=fz_flat,
            mesh=self.mesh,
        )
    
    def _build_laplacian(self) -> sparse.csr_matrix:
        """
        Build discrete Laplacian operator for velocity.
        
        Uses 7-point stencil for 3D.
        """
        nx, ny, nz = self.mesh.nx, self.mesh.ny, self.mesh.nz
        dx, dy, dz = self.mesh.dx, self.mesh.dy, self.mesh.dz
        n = nx * ny * nz
        
        def idx(i, j, k):
            return k * ny * nx + j * nx + i
        
        # Build in COO format
        rows = []
        cols = []
        vals = []
        
        # Coefficients
        cx = 1.0 / dx**2
        cy = 1.0 / dy**2
        cz = 1.0 / dz**2
        diag = -2 * (cx + cy + cz)
        
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    r = idx(i, j, k)
                    
                    # Check if at boundary (Dirichlet u=0)
                    is_boundary = (
                        i == 0 or i == nx - 1 or
                        j == 0 or j == ny - 1 or
                        k == 0 or k == nz - 1
                    )
                    
                    if is_boundary:
                        rows.append(r)
                        cols.append(r)
                        vals.append(1.0)
                    else:
                        # Diagonal
                        rows.append(r)
                        cols.append(r)
                        vals.append(diag)
                        
                        # x neighbors
                        rows.extend([r, r])
                        cols.extend([idx(i-1, j, k), idx(i+1, j, k)])
                        vals.extend([cx, cx])
                        
                        # y neighbors
                        rows.extend([r, r])
                        cols.extend([idx(i, j-1, k), idx(i, j+1, k)])
                        vals.extend([cy, cy])
                        
                        # z neighbors
                        rows.extend([r, r])
                        cols.extend([idx(i, j, k-1), idx(i, j, k+1)])
                        vals.extend([cz, cz])
        
        L = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
        return L
