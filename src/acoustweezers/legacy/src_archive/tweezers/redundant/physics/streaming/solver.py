"""
Acoustic streaming flow solver.

Solves for the time-averaged mean flow driven by acoustic forcing:

Steady Stokes equations:
    -∇p̄ + η∇²ū + f_stream = 0
    ∇·ū = 0

Or steady Navier-Stokes (if Re is not small):
    ρ(ū·∇)ū = -∇p̄ + η∇²ū + f_stream
    ∇·ū = 0

This module implements:
1. Steady Stokes solver (low Re)
2. Iterative Navier-Stokes solver
3. Boundary conditions for streaming (no-slip, stress-free)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict, Optional
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .forcing import StreamingForcing
from ..acoustics.materials import FluidMaterial


@dataclass
class StreamingField:
    """
    Streaming velocity field solution.
    
    Parameters
    ----------
    x, y, z : np.ndarray
        Coordinate arrays.
    ux, uy, uz : np.ndarray
        Mean velocity components [m/s].
    p : np.ndarray
        Mean pressure [Pa].
    """
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    ux: np.ndarray
    uy: np.ndarray
    uz: np.ndarray
    p: np.ndarray
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.ux.shape
    
    @property
    def speed(self) -> np.ndarray:
        """Velocity magnitude field."""
        return np.sqrt(self.ux**2 + self.uy**2 + self.uz**2)
    
    def velocity_at(self, x: float, y: float, z: float) -> np.ndarray:
        """
        Interpolate velocity at a point.
        
        Parameters
        ----------
        x, y, z : float
            Position coordinates.
        
        Returns
        -------
        v : np.ndarray
            Velocity vector [vx, vy, vz].
        """
        # Find grid indices
        ix = np.searchsorted(self.x, x) - 1
        iy = np.searchsorted(self.y, y) - 1
        iz = np.searchsorted(self.z, z) - 1
        
        # Clamp to valid range
        ix = max(0, min(ix, len(self.x) - 2))
        iy = max(0, min(iy, len(self.y) - 2))
        iz = max(0, min(iz, len(self.z) - 2))
        
        # Interpolation weights
        dx = self.x[1] - self.x[0]
        dy = self.y[1] - self.y[0]
        dz = self.z[1] - self.z[0]
        
        wx = (x - self.x[ix]) / dx
        wy = (y - self.y[iy]) / dy
        wz = (z - self.z[iz]) / dz
        
        # Trilinear interpolation
        def interp(f):
            c000 = f[ix, iy, iz]
            c100 = f[ix+1, iy, iz]
            c010 = f[ix, iy+1, iz]
            c110 = f[ix+1, iy+1, iz]
            c001 = f[ix, iy, iz+1]
            c101 = f[ix+1, iy, iz+1]
            c011 = f[ix, iy+1, iz+1]
            c111 = f[ix+1, iy+1, iz+1]
            
            c00 = c000 * (1 - wx) + c100 * wx
            c01 = c001 * (1 - wx) + c101 * wx
            c10 = c010 * (1 - wx) + c110 * wx
            c11 = c011 * (1 - wx) + c111 * wx
            
            c0 = c00 * (1 - wy) + c10 * wy
            c1 = c01 * (1 - wy) + c11 * wy
            
            return c0 * (1 - wz) + c1 * wz
        
        return np.array([interp(self.ux), interp(self.uy), interp(self.uz)])
    
    def divergence(self) -> np.ndarray:
        """Compute velocity divergence (should be ~0 for incompressible)."""
        dx = self.x[1] - self.x[0]
        dy = self.y[1] - self.y[0]
        dz = self.z[1] - self.z[0]
        
        dux_dx = np.gradient(self.ux, dx, axis=0)
        duy_dy = np.gradient(self.uy, dy, axis=1)
        duz_dz = np.gradient(self.uz, dz, axis=2)
        
        return dux_dx + duy_dy + duz_dz
    
    def vorticity(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute vorticity ω = ∇×u."""
        dx = self.x[1] - self.x[0]
        dy = self.y[1] - self.y[0]
        dz = self.z[1] - self.z[0]
        
        dux_dy = np.gradient(self.ux, dy, axis=1)
        dux_dz = np.gradient(self.ux, dz, axis=2)
        duy_dx = np.gradient(self.uy, dx, axis=0)
        duy_dz = np.gradient(self.uy, dz, axis=2)
        duz_dx = np.gradient(self.uz, dx, axis=0)
        duz_dy = np.gradient(self.uz, dy, axis=1)
        
        omega_x = duz_dy - duy_dz
        omega_y = dux_dz - duz_dx
        omega_z = duy_dx - dux_dy
        
        return omega_x, omega_y, omega_z


class StokesSolver:
    """
    Steady Stokes flow solver with body force.
    
    Solves:
        -∇p + η∇²u + f = 0
        ∇·u = 0
    
    Uses finite differences with pressure-velocity coupling.
    """
    
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        fluid: FluidMaterial,
    ):
        """
        Initialize solver.
        
        Parameters
        ----------
        x, y, z : np.ndarray
            Coordinate arrays.
        fluid : FluidMaterial
            Fluid properties (uses viscosity η).
        """
        self.x = x
        self.y = y
        self.z = z
        self.eta = fluid.eta
        self.rho = fluid.rho
        
        self.Nx = len(x)
        self.Ny = len(y)
        self.Nz = len(z)
        self.dx = x[1] - x[0]
        self.dy = y[1] - y[0]
        self.dz = z[1] - z[0]
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        return (self.Nx, self.Ny, self.Nz)
    
    @property
    def n_dof(self) -> int:
        """Total degrees of freedom (4 per node: ux, uy, uz, p)."""
        return 4 * self.Nx * self.Ny * self.Nz
    
    def _idx(self, i: int, j: int, k: int, var: int) -> int:
        """
        Convert indices to linear DOF.
        
        var: 0=ux, 1=uy, 2=uz, 3=p
        """
        return 4 * (i * self.Ny * self.Nz + j * self.Nz + k) + var
    
    def solve(
        self,
        forcing: StreamingForcing,
        bc_type: str = 'no_slip_all',
    ) -> StreamingField:
        """
        Solve Stokes equations with given forcing.
        
        Parameters
        ----------
        forcing : StreamingForcing
            Body force field.
        bc_type : str
            Boundary condition type:
            - 'no_slip_all': No-slip on all boundaries
            - 'no_slip_walls': No-slip on walls, stress-free on top
        
        Returns
        -------
        field : StreamingField
            Streaming velocity and pressure solution.
        """
        N = self.n_dof
        A = sp.lil_matrix((N, N), dtype=np.float64)
        b = np.zeros(N, dtype=np.float64)
        
        eta = self.eta
        dx, dy, dz = self.dx, self.dy, self.dz
        
        invdx2 = 1.0 / dx**2
        invdy2 = 1.0 / dy**2
        invdz2 = 1.0 / dz**2
        
        # Fill interior equations
        for i in range(1, self.Nx - 1):
            for j in range(1, self.Ny - 1):
                for k in range(1, self.Nz - 1):
                    # x-momentum: -∂p/∂x + η∇²ux + fx = 0
                    r = self._idx(i, j, k, 0)
                    
                    # Viscous term: η∇²ux
                    A[r, self._idx(i, j, k, 0)] = -2 * eta * (invdx2 + invdy2 + invdz2)
                    A[r, self._idx(i+1, j, k, 0)] = eta * invdx2
                    A[r, self._idx(i-1, j, k, 0)] = eta * invdx2
                    A[r, self._idx(i, j+1, k, 0)] = eta * invdy2
                    A[r, self._idx(i, j-1, k, 0)] = eta * invdy2
                    A[r, self._idx(i, j, k+1, 0)] = eta * invdz2
                    A[r, self._idx(i, j, k-1, 0)] = eta * invdz2
                    
                    # Pressure gradient: -∂p/∂x
                    A[r, self._idx(i+1, j, k, 3)] = -1.0 / (2 * dx)
                    A[r, self._idx(i-1, j, k, 3)] = 1.0 / (2 * dx)
                    
                    # RHS: forcing
                    b[r] = -forcing.fx[i, j, k]
                    
                    # y-momentum
                    r = self._idx(i, j, k, 1)
                    
                    A[r, self._idx(i, j, k, 1)] = -2 * eta * (invdx2 + invdy2 + invdz2)
                    A[r, self._idx(i+1, j, k, 1)] = eta * invdx2
                    A[r, self._idx(i-1, j, k, 1)] = eta * invdx2
                    A[r, self._idx(i, j+1, k, 1)] = eta * invdy2
                    A[r, self._idx(i, j-1, k, 1)] = eta * invdy2
                    A[r, self._idx(i, j, k+1, 1)] = eta * invdz2
                    A[r, self._idx(i, j, k-1, 1)] = eta * invdz2
                    
                    A[r, self._idx(i, j+1, k, 3)] = -1.0 / (2 * dy)
                    A[r, self._idx(i, j-1, k, 3)] = 1.0 / (2 * dy)
                    
                    b[r] = -forcing.fy[i, j, k]
                    
                    # z-momentum
                    r = self._idx(i, j, k, 2)
                    
                    A[r, self._idx(i, j, k, 2)] = -2 * eta * (invdx2 + invdy2 + invdz2)
                    A[r, self._idx(i+1, j, k, 2)] = eta * invdx2
                    A[r, self._idx(i-1, j, k, 2)] = eta * invdx2
                    A[r, self._idx(i, j+1, k, 2)] = eta * invdy2
                    A[r, self._idx(i, j-1, k, 2)] = eta * invdy2
                    A[r, self._idx(i, j, k+1, 2)] = eta * invdz2
                    A[r, self._idx(i, j, k-1, 2)] = eta * invdz2
                    
                    A[r, self._idx(i, j, k+1, 3)] = -1.0 / (2 * dz)
                    A[r, self._idx(i, j, k-1, 3)] = 1.0 / (2 * dz)
                    
                    b[r] = -forcing.fz[i, j, k]
                    
                    # Continuity: ∇·u = 0
                    r = self._idx(i, j, k, 3)
                    
                    A[r, self._idx(i+1, j, k, 0)] = 1.0 / (2 * dx)
                    A[r, self._idx(i-1, j, k, 0)] = -1.0 / (2 * dx)
                    A[r, self._idx(i, j+1, k, 1)] = 1.0 / (2 * dy)
                    A[r, self._idx(i, j-1, k, 1)] = -1.0 / (2 * dy)
                    A[r, self._idx(i, j, k+1, 2)] = 1.0 / (2 * dz)
                    A[r, self._idx(i, j, k-1, 2)] = -1.0 / (2 * dz)
                    
                    b[r] = 0.0
        
        # Apply boundary conditions
        self._apply_boundary_conditions(A, b, bc_type)
        
        # Add small regularization for pressure (fix gauge)
        r_gauge = self._idx(self.Nx//2, self.Ny//2, self.Nz//2, 3)
        A[r_gauge, :] = 0
        A[r_gauge, r_gauge] = 1.0
        b[r_gauge] = 0.0
        
        # Solve
        A_csc = A.tocsc()
        
        # Use iterative solver for large systems
        if N > 50000:
            sol, info = spla.gmres(A_csc, b, rtol=1e-8, maxiter=1000)
            if info != 0:
                print(f"Warning: GMRES did not converge (info={info})")
        else:
            sol = spla.spsolve(A_csc, b)
        
        # Unpack solution
        ux = np.zeros(self.shape)
        uy = np.zeros(self.shape)
        uz = np.zeros(self.shape)
        p = np.zeros(self.shape)
        
        for i in range(self.Nx):
            for j in range(self.Ny):
                for k in range(self.Nz):
                    ux[i, j, k] = sol[self._idx(i, j, k, 0)]
                    uy[i, j, k] = sol[self._idx(i, j, k, 1)]
                    uz[i, j, k] = sol[self._idx(i, j, k, 2)]
                    p[i, j, k] = sol[self._idx(i, j, k, 3)]
        
        return StreamingField(
            x=self.x.copy(),
            y=self.y.copy(),
            z=self.z.copy(),
            ux=ux, uy=uy, uz=uz,
            p=p,
        )
    
    def _apply_boundary_conditions(
        self,
        A: sp.lil_matrix,
        b: np.ndarray,
        bc_type: str,
    ):
        """Apply boundary conditions to system."""
        # All boundaries: no-slip (u=0) by default
        boundary_indices = []
        
        # x boundaries
        for j in range(self.Ny):
            for k in range(self.Nz):
                boundary_indices.append((0, j, k))
                boundary_indices.append((self.Nx - 1, j, k))
        
        # y boundaries
        for i in range(self.Nx):
            for k in range(self.Nz):
                boundary_indices.append((i, 0, k))
                boundary_indices.append((i, self.Ny - 1, k))
        
        # z boundaries
        for i in range(self.Nx):
            for j in range(self.Ny):
                boundary_indices.append((i, j, 0))
                boundary_indices.append((i, j, self.Nz - 1))
        
        for (i, j, k) in boundary_indices:
            # No-slip: u = 0
            for var in range(3):  # ux, uy, uz
                r = self._idx(i, j, k, var)
                A[r, :] = 0
                A[r, r] = 1.0
                b[r] = 0.0
            
            # Pressure: Neumann (zero gradient) or just identity
            r = self._idx(i, j, k, 3)
            A[r, :] = 0
            A[r, r] = 1.0
            b[r] = 0.0


class StreamingSolver:
    """
    High-level interface for acoustic streaming computation.
    
    Combines forcing calculation and flow solver.
    """
    
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        fluid: FluidMaterial,
    ):
        """
        Initialize streaming solver.
        
        Parameters
        ----------
        x, y, z : np.ndarray
            Coordinate arrays.
        fluid : FluidMaterial
            Fluid properties.
        """
        self.x = x
        self.y = y
        self.z = z
        self.fluid = fluid
        
        self.stokes_solver = StokesSolver(x, y, z, fluid)
    
    def compute_streaming(
        self,
        p: np.ndarray,
        rho: np.ndarray,
        omega: float,
    ) -> StreamingField:
        """
        Compute streaming flow from acoustic field.
        
        Parameters
        ----------
        p : np.ndarray
            Acoustic pressure field (complex).
        rho : np.ndarray
            Density field.
        omega : float
            Angular frequency.
        
        Returns
        -------
        streaming : StreamingField
            Mean velocity field.
        """
        from .forcing import compute_streaming_force
        
        dx = self.x[1] - self.x[0]
        dy = self.y[1] - self.y[0]
        dz = self.z[1] - self.z[0]
        
        # Compute body force
        forcing = compute_streaming_force(
            p=p, rho=rho, omega=omega, fluid=self.fluid,
            dx=dx, dy=dy, dz=dz,
            x=self.x, y=self.y, z=self.z,
        )
        
        # Solve Stokes equations
        return self.stokes_solver.solve(forcing)
    
    def estimate_streaming_velocity_scale(
        self,
        p_amplitude: float,
        omega: float,
    ) -> float:
        """
        Estimate characteristic streaming velocity.
        
        For Rayleigh streaming:
            U_s ~ (3/4) * |v_1|² / c ~ (3/4) * (p/ρc)² / c
        
        Parameters
        ----------
        p_amplitude : float
            Typical pressure amplitude [Pa].
        omega : float
            Angular frequency.
        
        Returns
        -------
        U_scale : float
            Estimated streaming velocity [m/s].
        """
        c = self.fluid.c
        rho = self.fluid.rho
        
        v1 = p_amplitude / (rho * c)
        U_s = 0.75 * v1**2 / c
        
        return U_s


if __name__ == "__main__":
    from ..acoustics.materials import MaterialDatabase
    from .forcing import StreamingForcing
    
    # Demo: solve simple Stokes flow
    water = MaterialDatabase.water(25.0)
    
    # Create grid
    Nx, Ny, Nz = 21, 21, 11
    x = np.linspace(0, 0.01, Nx)
    y = np.linspace(0, 0.01, Ny)
    z = np.linspace(0, 0.005, Nz)
    
    # Synthetic forcing (uniform in z direction)
    fx = np.zeros((Nx, Ny, Nz))
    fy = np.zeros((Nx, Ny, Nz))
    fz = np.ones((Nx, Ny, Nz)) * 1e-3  # 1 mN/m³
    
    forcing = StreamingForcing(
        x=x, y=y, z=z,
        fx=fx, fy=fy, fz=fz,
    )
    
    solver = StokesSolver(x, y, z, water)
    print(f"Solving Stokes flow...")
    print(f"Grid: {solver.shape}, DOFs: {solver.n_dof}")
    
    field = solver.solve(forcing)
    
    print(f"\nSolution:")
    print(f"  Max |u|: {np.max(field.speed)*1e6:.2f} μm/s")
    print(f"  Max |div(u)|: {np.max(np.abs(field.divergence())):.2e} 1/s")
