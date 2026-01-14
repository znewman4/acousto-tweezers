#src/acousto/solvers/helmholtz_3d_simple.py
"""
Simple 3D finite-difference Helmholtz solver for acoustic radiation force validation.

Domain: (x, y, z) ∈ [0, Lx] × [0, Ly] × [0, Lz]
Boundary conditions:
  - Bottom (z=0): Neumann BC with prescribed normal velocity vb(x, y)
  - All other faces: Neumann (free/rigid)

Model: (∇² + k_eff²) p = 0

This is a coarse implementation for verification. Not optimized for speed.
Used to validate whether true 3D model produces physically meaningful forces.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Union

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


BoundarySpec = Union[complex, float, np.ndarray, Callable[[np.ndarray], np.ndarray]]


@dataclass(frozen=True)
class Field3D:
    """3D acoustic pressure field p(x, y, z)."""
    x: np.ndarray  # shape (Nx,)
    y: np.ndarray  # shape (Ny,)
    z: np.ndarray  # shape (Nz,)
    p: np.ndarray  # complex, shape (Nz, Ny, Nx) [z, y, x] = [z][y][x]
    omega: float
    c0: float
    rho0: float

    @property
    def k(self) -> float:
        return self.omega / self.c0


def build_helmholtz_3d_operator(
    *,
    Lx: float,
    Ly: float,
    Lz: float,
    Nx: int,
    Ny: int,
    Nz: int,
    f: float,
    c0: float,
    rho0: float,
    loss_eta: float = 1e-3,
) -> tuple[callable, np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    """Build 3D Helmholtz solver (returns factorized solver, grids, and dimensions).
    
    Returns
    -------
    (solve_func, x, y, z, Nx, Ny, Nz)
        where solve_func takes RHS of shape (Nz, Ny, Nx) and returns p of same shape.
    """
    x = np.linspace(0.0, Lx, Nx)
    y = np.linspace(0.0, Ly, Ny)
    z = np.linspace(0.0, Lz, Nz)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = z[1] - z[0]

    omega = 2.0 * np.pi * f
    k = omega / c0
    keff2 = k * k * (1.0 + 1j * loss_eta)

    N = Nx * Ny * Nz

    def idx(iz: int, iy: int, ix: int) -> int:
        """Row-major indexing: [iz][iy][ix]."""
        return iz * (Ny * Nx) + iy * Nx + ix

    # Build sparse matrix
    A = sp.lil_matrix((N, N), dtype=np.complex128)
    b0 = np.zeros(N, dtype=np.complex128)

    invdx2 = 1.0 / (dx * dx)
    invdy2 = 1.0 / (dy * dy)
    invdz2 = 1.0 / (dz * dz)
    lap_center = -2.0 * (invdx2 + invdy2 + invdz2) + keff2

    # Interior stencil
    for iz in range(1, Nz - 1):
        for iy in range(1, Ny - 1):
            for ix in range(1, Nx - 1):
                r = idx(iz, iy, ix)
                # Center
                A[r, idx(iz, iy, ix)] = lap_center
                # 6-point stencil neighbors
                A[r, idx(iz, iy, ix - 1)] = invdx2
                A[r, idx(iz, iy, ix + 1)] = invdx2
                A[r, idx(iz, iy - 1, ix)] = invdy2
                A[r, idx(iz, iy + 1, ix)] = invdy2
                A[r, idx(iz - 1, iy, ix)] = invdz2
                A[r, idx(iz + 1, iy, ix)] = invdz2
                b0[r] = 0.0

    # Boundary condition helper: Neumann 2nd-order one-sided
    def set_neumann_x_left(iy: int, iz: int, value: complex):
        r = idx(iz, iy, 0)
        A[r, :] = 0.0
        A[r, idx(iz, iy, 0)] = -3.0 / (2.0 * dx)
        A[r, idx(iz, iy, 1)] = 4.0 / (2.0 * dx)
        A[r, idx(iz, iy, 2)] = -1.0 / (2.0 * dx)
        b0[r] = value

    def set_neumann_x_right(iy: int, iz: int, value: complex):
        r = idx(iz, iy, Nx - 1)
        A[r, :] = 0.0
        A[r, idx(iz, iy, Nx - 1)] = 3.0 / (2.0 * dx)
        A[r, idx(iz, iy, Nx - 2)] = -4.0 / (2.0 * dx)
        A[r, idx(iz, iy, Nx - 3)] = 1.0 / (2.0 * dx)
        b0[r] = value

    def set_neumann_y_left(ix: int, iz: int, value: complex):
        r = idx(iz, 0, ix)
        A[r, :] = 0.0
        A[r, idx(iz, 0, ix)] = -3.0 / (2.0 * dy)
        A[r, idx(iz, 1, ix)] = 4.0 / (2.0 * dy)
        A[r, idx(iz, 2, ix)] = -1.0 / (2.0 * dy)
        b0[r] = value

    def set_neumann_y_right(ix: int, iz: int, value: complex):
        r = idx(iz, Ny - 1, ix)
        A[r, :] = 0.0
        A[r, idx(iz, Ny - 1, ix)] = 3.0 / (2.0 * dy)
        A[r, idx(iz, Ny - 2, ix)] = -4.0 / (2.0 * dy)
        A[r, idx(iz, Ny - 3, ix)] = 1.0 / (2.0 * dy)
        b0[r] = value

    def set_neumann_z_bottom(ix: int, iy: int, value: complex):
        r = idx(0, iy, ix)
        A[r, :] = 0.0
        A[r, idx(0, iy, ix)] = -3.0 / (2.0 * dz)
        A[r, idx(1, iy, ix)] = 4.0 / (2.0 * dz)
        A[r, idx(2, iy, ix)] = -1.0 / (2.0 * dz)
        b0[r] = value

    def set_neumann_z_top(ix: int, iy: int, value: complex):
        r = idx(Nz - 1, iy, ix)
        A[r, :] = 0.0
        A[r, idx(Nz - 1, iy, ix)] = 3.0 / (2.0 * dz)
        A[r, idx(Nz - 2, iy, ix)] = -4.0 / (2.0 * dz)
        A[r, idx(Nz - 3, iy, ix)] = 1.0 / (2.0 * dz)
        b0[r] = value

    # Apply all-Neumann boundary conditions
    # x boundaries
    for iz in range(1, Nz - 1):
        for iy in range(1, Ny - 1):
            set_neumann_x_left(iy, iz, 0.0)
            set_neumann_x_right(iy, iz, 0.0)

    # y boundaries
    for iz in range(1, Nz - 1):
        for ix in range(1, Nx - 1):
            set_neumann_y_left(ix, iz, 0.0)
            set_neumann_y_right(ix, iz, 0.0)

    # z boundaries (top: Neumann 0, bottom: will be updated dynamically)
    for iy in range(1, Ny - 1):
        for ix in range(1, Nx - 1):
            set_neumann_z_top(ix, iy, 0.0)
            # Bottom boundary is set by solve_for_bottom_vb

    # Handle corners and edges (simplified: tie to interior)
    # This is a rough treatment; production code would be more careful
    edges_and_corners = []
    for iz in [0, Nz - 1]:
        for iy in [0, Ny - 1]:
            for ix in range(Nx):
                edges_and_corners.append((iz, iy, ix))
        for iy in range(1, Ny - 1):
            for ix in [0, Nx - 1]:
                edges_and_corners.append((iz, iy, ix))
    for iz in range(1, Nz - 1):
        for iy in [0, Ny - 1]:
            for ix in [0, Nx - 1]:
                edges_and_corners.append((iz, iy, ix))

    for (iz, iy, ix) in edges_and_corners:
        r = idx(iz, iy, ix)
        if A[r, r] == 0.0:  # Not yet set
            A[r, :] = 0.0
            A[r, r] = 1.0
            b0[r] = 0.0

    # Gauge fixing (all-Neumann -> add constraint to fix phase)
    r_gauge = idx(1, 1, 1)
    A[r_gauge, :] = 0.0
    A[r_gauge, r_gauge] = 1.0
    b0[r_gauge] = 0.0

    # Factorize
    A_csc = A.tocsc()
    
    # Add small regularization to avoid singularity
    A_csc = A_csc + 1e-14 * sp.identity(N)
    
    try:
        solve_lu = spla.factorized(A_csc)
    except RuntimeError:
        # Fallback: use direct solver without factorization
        def solve_func(b_rhs: np.ndarray) -> np.ndarray:
            b_vec = b_rhs.ravel()
            p_vec = spla.spsolve(A_csc, b_vec)
            return p_vec.reshape(b_rhs.shape)
        return solve_func, x, y, z, Nx, Ny, Nz, A, b0, idx

    def solve_func(b_rhs: np.ndarray) -> np.ndarray:
        """Solve A p = b_rhs, return p in same shape as input."""
        b_vec = b_rhs.ravel()
        p_vec = solve_lu(b_vec)
        return p_vec.reshape(b_rhs.shape)

    return solve_func, x, y, z, Nx, Ny, Nz, A, b0, idx


def solve_helmholtz_3d_bottom_driven(
    *,
    Lx: float,
    Ly: float,
    Lz: float,
    Nx: int,
    Ny: int,
    Nz: int,
    f: float,
    c0: float,
    rho0: float,
    vb: Union[float, np.ndarray] = 0.0,
    loss_eta: float = 1e-3,
) -> Field3D:
    """Solve 3D Helmholtz with bottom-driven Neumann BC.
    
    Parameters
    ----------
    Lx, Ly, Lz : float
        Domain size (meters).
    Nx, Ny, Nz : int
        Grid points in each direction.
    f : float
        Frequency (Hz).
    c0 : float
        Sound speed (m/s).
    rho0 : float
        Density (kg/m³).
    vb : float or array of shape (Ny, Nx)
        Bottom normal velocity pattern vb(x, y) (m/s).
    loss_eta : float
        Loss parameter.
    
    Returns
    -------
    Field3D
    """
    solve_func, x, y, z, Nx, Ny, Nz, A, b0, idx = build_helmholtz_3d_operator(
        Lx=Lx, Ly=Ly, Lz=Lz, Nx=Nx, Ny=Ny, Nz=Nz,
        f=f, c0=c0, rho0=rho0, loss_eta=loss_eta
    )

    # Build RHS
    b = b0.copy()

    # Bottom boundary: ∂p/∂z = -i*omega*rho0*vb
    omega = 2.0 * np.pi * f
    dz = z[1] - z[0]

    if isinstance(vb, (int, float)):
        vb_xy = np.full((Ny, Nx), complex(vb), dtype=np.complex128)
    else:
        vb_xy = np.asarray(vb, dtype=np.complex128)

    dpz_bottom = (-1j * omega * rho0) * vb_xy

    for iy in range(1, Ny - 1):
        for ix in range(1, Nx - 1):
            r = idx(0, iy, ix)
            A[r, :] = 0.0
            A[r, idx(0, iy, ix)] = -3.0 / (2.0 * dz)
            A[r, idx(1, iy, ix)] = 4.0 / (2.0 * dz)
            A[r, idx(2, iy, ix)] = -1.0 / (2.0 * dz)
            b[r] = dpz_bottom[iy, ix]

    # Solve
    A_csc = A.tocsc()
    solve_new = spla.factorized(A_csc)
    p_vec = solve_new(b)
    p = p_vec.reshape((Nz, Ny, Nx))

    return Field3D(x=x, y=y, z=z, p=p, omega=omega, c0=c0, rho0=rho0)
