# src/acousto/solvers/fd_helmholtz_2d_forced_25d.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Union, Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .fd_helmholtz_2d import Field2D, BoundarySpec, _eval_boundary, BCType

@dataclass
class Helmholtz25DOperator:
    x: np.ndarray
    y: np.ndarray
    Nx: int
    Ny: int
    dx: float
    dy: float
    omega: float
    c0: float
    rho0: float
    coupling_alpha: float
    # reusable solver closure: p_vec = solve(b_vec)
    solve: Callable[[np.ndarray], np.ndarray]

    def solve_for_vb(self, vb_xy: np.ndarray) -> Field2D:
        # vb_xy shape (Ny, Nx), complex
        s_xy = self.coupling_alpha * (-1j * self.omega * self.rho0) * vb_xy
        b = np.zeros(self.Nx * self.Ny, dtype=np.complex128)

        def idx(j: int, i: int) -> int:
            return j * self.Nx + i

        # Fill interior RHS only (boundary rows are BC equations)
        for j in range(1, self.Ny - 1):
            for i in range(1, self.Nx - 1):
                b[idx(j, i)] = s_xy[j, i]

        p_vec = self.solve(b)
        p = p_vec.reshape((self.Ny, self.Nx))
        return Field2D(x=self.x, y=self.y, p=p, omega=self.omega, c0=self.c0, rho0=self.rho0)


SourceSpec = Union[
    complex,
    float,
    np.ndarray,
    Callable[[np.ndarray, np.ndarray], np.ndarray],
]


def _eval_source(
    source: SourceSpec,
    X: np.ndarray,
    Y: np.ndarray,
    expected_shape: tuple[int, int],
    name: str,
) -> np.ndarray:
    """
    Convert a source specification into a complex array of shape (Ny, Nx).

    source can be:
    - scalar (float/complex): constant source everywhere
    - array-like of shape (Ny, Nx)
    - callable: source(X, Y) -> array-like (Ny, Nx), where X,Y are meshgrids
    """
    if callable(source):
        vals = np.asarray(source(X, Y))
    else:
        vals = np.asarray(source)

    if vals.ndim == 0:
        return np.full(expected_shape, complex(vals), dtype=np.complex128)

    if vals.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}. Got {vals.shape}.")
    return vals.astype(np.complex128, copy=False)

def build_helmholtz_2d_forced_25d_operator(
    *,
    Lx: float,
    Ly: float,
    Nx: int,
    Ny: int,
    f: float,
    c0: float,
    rho0: float,
    left_type: BCType = "neumann",
    right_type: BCType = "neumann",
    bottom_type: BCType = "neumann",
    top_type: BCType = "neumann",
    left: BoundarySpec = 0.0 + 0.0j,
    right: BoundarySpec = 0.0 + 0.0j,
    bottom: BoundarySpec = 0.0 + 0.0j,
    top: BoundarySpec = 0.0 + 0.0j,
    kz: float = 0.0,
    coupling_alpha: float = 1.0,
    loss_eta: float = 1e-3,
) -> Helmholtz25DOperator:
    if Nx < 3 or Ny < 3:
        raise ValueError("Nx and Ny must be >= 3.")

    x = np.linspace(0.0, Lx, Nx)
    y = np.linspace(0.0, Ly, Ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    omega = 2.0 * np.pi * f
    k = omega / c0
    keff2 = (k * k - kz * kz) * (1.0 + 1j * loss_eta)

    gL = _eval_boundary(left, y, Ny, "left")
    gR = _eval_boundary(right, y, Ny, "right")
    gB = _eval_boundary(bottom, x, Nx, "bottom")
    gT = _eval_boundary(top, x, Nx, "top")

    N = Nx * Ny

    def idx(j: int, i: int) -> int:
        return j * Nx + i

    A = sp.lil_matrix((N, N), dtype=np.complex128)
    b0 = np.zeros(N, dtype=np.complex128)  # placeholder, not used after factorization

    invdx2 = 1.0 / (dx * dx)
    invdy2 = 1.0 / (dy * dy)

    # Interior rows: (∇² + keff2) p = s  -> A p = b
    for j in range(1, Ny - 1):
        for i in range(1, Nx - 1):
            r = idx(j, i)
            A[r, idx(j, i)] = (-2.0 * invdx2 - 2.0 * invdy2) + keff2
            A[r, idx(j, i - 1)] = invdx2
            A[r, idx(j, i + 1)] = invdx2
            A[r, idx(j - 1, i)] = invdy2
            A[r, idx(j + 1, i)] = invdy2
            b0[r] = 0.0 + 0.0j

    def set_dirichlet(r: int, node: int, value: complex):
        A[r, :] = 0.0
        A[r, node] = 1.0
        b0[r] = value

    def set_neumann_left(j: int, value: complex):
        r = idx(j, 0)
        A[r, :] = 0.0
        A[r, idx(j, 0)] = -3.0 / (2.0 * dx)
        A[r, idx(j, 1)] =  4.0 / (2.0 * dx)
        A[r, idx(j, 2)] = -1.0 / (2.0 * dx)
        b0[r] = value

    def set_neumann_right(j: int, value: complex):
        r = idx(j, Nx - 1)
        A[r, :] = 0.0
        A[r, idx(j, Nx - 1)] =  3.0 / (2.0 * dx)
        A[r, idx(j, Nx - 2)] = -4.0 / (2.0 * dx)
        A[r, idx(j, Nx - 3)] =  1.0 / (2.0 * dx)
        b0[r] = value

    def set_neumann_bottom(i: int, value: complex):
        r = idx(0, i)
        A[r, :] = 0.0
        A[r, idx(0, i)] = -3.0 / (2.0 * dy)
        A[r, idx(1, i)] =  4.0 / (2.0 * dy)
        A[r, idx(2, i)] = -1.0 / (2.0 * dy)
        b0[r] = value

    def set_neumann_top(i: int, value: complex):
        r = idx(Ny - 1, i)
        A[r, :] = 0.0
        A[r, idx(Ny - 1, i)] =  3.0 / (2.0 * dy)
        A[r, idx(Ny - 2, i)] = -4.0 / (2.0 * dy)
        A[r, idx(Ny - 3, i)] =  1.0 / (2.0 * dy)
        b0[r] = value

    for j in range(1, Ny - 1):
        if left_type == "dirichlet":
            set_dirichlet(idx(j, 0), idx(j, 0), gL[j])
        else:
            set_neumann_left(j, gL[j])

    for j in range(1, Ny - 1):
        if right_type == "dirichlet":
            set_dirichlet(idx(j, Nx - 1), idx(j, Nx - 1), gR[j])
        else:
            set_neumann_right(j, gR[j])

    for i in range(1, Nx - 1):
        if bottom_type == "dirichlet":
            set_dirichlet(idx(0, i), idx(0, i), gB[i])
        else:
            set_neumann_bottom(i, gB[i])

    for i in range(1, Nx - 1):
        if top_type == "dirichlet":
            set_dirichlet(idx(Ny - 1, i), idx(Ny - 1, i), gT[i])
        else:
            set_neumann_top(i, gT[i])

    def set_corner_tie_to_interior(j: int, i: int):
        r = idx(j, i)
        A[r, :] = 0.0
        A[r, idx(j, i)] = 1.0
        ji = 1 if j == 0 else (Ny - 2)
        ii = 1 if i == 0 else (Nx - 2)
        A[r, idx(ji, ii)] = -1.0
        b0[r] = 0.0 + 0.0j

    def corner_dirichlet_value(j: int, i: int) -> complex:
        if i == 0 and left_type == "dirichlet":
            return gL[j]
        if i == Nx - 1 and right_type == "dirichlet":
            return gR[j]
        if j == 0 and bottom_type == "dirichlet":
            return gB[i]
        if j == Ny - 1 and top_type == "dirichlet":
            return gT[i]
        return 0.0 + 0.0j

    corners = [(0, 0), (0, Nx - 1), (Ny - 1, 0), (Ny - 1, Nx - 1)]
    for (j, i) in corners:
        any_dir = (
            (i == 0 and left_type == "dirichlet")
            or (i == Nx - 1 and right_type == "dirichlet")
            or (j == 0 and bottom_type == "dirichlet")
            or (j == Ny - 1 and top_type == "dirichlet")
        )
        if any_dir:
            set_dirichlet(idx(j, i), idx(j, i), corner_dirichlet_value(j, i))
        else:
            set_corner_tie_to_interior(j, i)

    all_neumann = (
        left_type == "neumann"
        and right_type == "neumann"
        and bottom_type == "neumann"
        and top_type == "neumann"
    )
    if all_neumann:
        jg, ig = 1, 1
        rg = idx(jg, ig)
        A[rg, :] = 0.0
        A[rg, rg] = 1.0
        b0[rg] = 0.0 + 0.0j

    # Factorize once
    A_csc = A.tocsc()
    solve = spla.factorized(A_csc)

    return Helmholtz25DOperator(
        x=x, y=y, Nx=Nx, Ny=Ny, dx=dx, dy=dy,
        omega=omega, c0=c0, rho0=rho0,
        coupling_alpha=coupling_alpha,
        solve=solve,
    )



def solve_helmholtz_2d_forced_25d(
    *,
    Lx: float,
    Ly: float,
    Nx: int,
    Ny: int,
    f: float,
    c0: float,
    rho0: float,
    # side-wall BCs in x–y (usually rigid walls)
    left_type: BCType = "neumann",
    right_type: BCType = "neumann",
    bottom_type: BCType = "neumann",
    top_type: BCType = "neumann",
    left: BoundarySpec = 0.0 + 0.0j,
    right: BoundarySpec = 0.0 + 0.0j,
    bottom: BoundarySpec = 0.0 + 0.0j,
    top: BoundarySpec = 0.0 + 0.0j,
    # 2.5D controls
    kz: float = 0.0,          # vertical wavenumber (rad/m); kz=0 recovers 2D
    # bottom drive -> effective forcing
    vb: SourceSpec = 0.0,     # bottom normal velocity pattern vb(x,y) [m/s], or effective map
    coupling_alpha: float = 1.0,  # dimensionless coupling constant
    loss_eta: float = 1e-3,
) -> Field2D:
    """
    2.5D forced Helmholtz solver on an (x,y) rectangle, representing bottom-driven actuation.

    Model:
      (∇_xy^2 + k_eff^2) p(x,y) = s(x,y)

    where:
      k_eff^2 = (k^2 - kz^2) * (1 + i*loss_eta)
      s(x,y)  = coupling_alpha * (-i ω ρ0) * vb(x,y)

    Notes:
    - This does NOT add a z-dimension. It is a reduced model (2.5D).
    - Boundary conditions here are ONLY the lateral boundaries in the x–y plane.
      For a "dish", you typically use rigid walls => Neumann 0 on all 4 sides.
    """
    if Nx < 3 or Ny < 3:
        raise ValueError("Nx and Ny must be >= 3.")

    x = np.linspace(0.0, Lx, Nx)
    y = np.linspace(0.0, Ly, Ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    omega = 2.0 * np.pi * f
    k = omega / c0

    # 2.5D effective wavenumber
    keff2 = (k * k - kz * kz) * (1.0 + 1j * loss_eta)

    # Evaluate boundary specs as vectors on boundary nodes
    gL = _eval_boundary(left, y, Ny, "left")
    gR = _eval_boundary(right, y, Ny, "right")
    gB = _eval_boundary(bottom, x, Nx, "bottom")
    gT = _eval_boundary(top, x, Nx, "top")

    # Build source field s(x,y)
    X, Y = np.meshgrid(x, y)  # shapes (Ny, Nx)
    vb_xy = _eval_source(vb, X, Y, (Ny, Nx), "vb")
    s_xy = coupling_alpha * (-1j * omega * rho0) * vb_xy  # shape (Ny, Nx)

    N = Nx * Ny

    def idx(j: int, i: int) -> int:
        return j * Nx + i  # row-major

    A = sp.lil_matrix((N, N), dtype=np.complex128)
    b = np.zeros(N, dtype=np.complex128)

    invdx2 = 1.0 / (dx * dx)
    invdy2 = 1.0 / (dy * dy)

    # Interior stencil: (∇² + keff2) p = s
    for j in range(1, Ny - 1):
        for i in range(1, Nx - 1):
            r = idx(j, i)
            A[r, idx(j, i)] = (-2.0 * invdx2 - 2.0 * invdy2) + keff2
            A[r, idx(j, i - 1)] = invdx2
            A[r, idx(j, i + 1)] = invdx2
            A[r, idx(j - 1, i)] = invdy2
            A[r, idx(j + 1, i)] = invdy2
            b[r] = s_xy[j, i]  # forcing

    # Dirichlet helper
    def set_dirichlet(r: int, node: int, value: complex):
        A[r, :] = 0.0
        A[r, node] = 1.0
        b[r] = value

    # Neumann rows (2nd-order one-sided)
    def set_neumann_left(j: int, value: complex):
        r = idx(j, 0)
        A[r, :] = 0.0
        A[r, idx(j, 0)] = -3.0 / (2.0 * dx)
        A[r, idx(j, 1)] =  4.0 / (2.0 * dx)
        A[r, idx(j, 2)] = -1.0 / (2.0 * dx)
        b[r] = value

    def set_neumann_right(j: int, value: complex):
        r = idx(j, Nx - 1)
        A[r, :] = 0.0
        A[r, idx(j, Nx - 1)] =  3.0 / (2.0 * dx)
        A[r, idx(j, Nx - 2)] = -4.0 / (2.0 * dx)
        A[r, idx(j, Nx - 3)] =  1.0 / (2.0 * dx)
        b[r] = value

    def set_neumann_bottom(i: int, value: complex):
        r = idx(0, i)
        A[r, :] = 0.0
        A[r, idx(0, i)] = -3.0 / (2.0 * dy)
        A[r, idx(1, i)] =  4.0 / (2.0 * dy)
        A[r, idx(2, i)] = -1.0 / (2.0 * dy)
        b[r] = value

    def set_neumann_top(i: int, value: complex):
        r = idx(Ny - 1, i)
        A[r, :] = 0.0
        A[r, idx(Ny - 1, i)] =  3.0 / (2.0 * dy)
        A[r, idx(Ny - 2, i)] = -4.0 / (2.0 * dy)
        A[r, idx(Ny - 3, i)] =  1.0 / (2.0 * dy)
        b[r] = value

    # Left boundary
    for j in range(1, Ny - 1):
        if left_type == "dirichlet":
            set_dirichlet(idx(j, 0), idx(j, 0), gL[j])
        else:
            set_neumann_left(j, gL[j])

    # Right boundary
    for j in range(1, Ny - 1):
        if right_type == "dirichlet":
            set_dirichlet(idx(j, Nx - 1), idx(j, Nx - 1), gR[j])
        else:
            set_neumann_right(j, gR[j])

    # Bottom boundary (y=0 edge in the 2D plane)
    for i in range(1, Nx - 1):
        if bottom_type == "dirichlet":
            set_dirichlet(idx(0, i), idx(0, i), gB[i])
        else:
            set_neumann_bottom(i, gB[i])

    # Top boundary (y=Ly edge in the 2D plane)
    for i in range(1, Nx - 1):
        if top_type == "dirichlet":
            set_dirichlet(idx(Ny - 1, i), idx(Ny - 1, i), gT[i])
        else:
            set_neumann_top(i, gT[i])

    # Corners: same strategy as your mixed_bc (Dirichlet wins, else tie to interior)
    def set_corner_tie_to_interior(j: int, i: int):
        r = idx(j, i)
        A[r, :] = 0.0
        A[r, idx(j, i)] = 1.0
        ji = 1 if j == 0 else (Ny - 2)
        ii = 1 if i == 0 else (Nx - 2)
        A[r, idx(ji, ii)] = -1.0
        b[r] = 0.0 + 0.0j

    def corner_dirichlet_value(j: int, i: int) -> complex:
        if i == 0 and left_type == "dirichlet":
            return gL[j]
        if i == Nx - 1 and right_type == "dirichlet":
            return gR[j]
        if j == 0 and bottom_type == "dirichlet":
            return gB[i]
        if j == Ny - 1 and top_type == "dirichlet":
            return gT[i]
        return 0.0 + 0.0j

    corners = [(0, 0), (0, Nx - 1), (Ny - 1, 0), (Ny - 1, Nx - 1)]
    for (j, i) in corners:
        any_dir = (
            (i == 0 and left_type == "dirichlet")
            or (i == Nx - 1 and right_type == "dirichlet")
            or (j == 0 and bottom_type == "dirichlet")
            or (j == Ny - 1 and top_type == "dirichlet")
        )
        if any_dir:
            set_dirichlet(idx(j, i), idx(j, i), corner_dirichlet_value(j, i))
        else:
            set_corner_tie_to_interior(j, i)

    # Gauge fixing if all-Neumann (helps conditioning / repeatability)
    all_neumann = (
        left_type == "neumann"
        and right_type == "neumann"
        and bottom_type == "neumann"
        and top_type == "neumann"
    )
    if all_neumann:
        jg, ig = 1, 1
        rg = idx(jg, ig)
        A[rg, :] = 0.0
        A[rg, rg] = 1.0
        b[rg] = 0.0 + 0.0j

    p_vec = spla.spsolve(A.tocsr(), b)
    p = p_vec.reshape((Ny, Nx))

    return Field2D(x=x, y=y, p=p, omega=omega, c0=c0, rho0=rho0)
