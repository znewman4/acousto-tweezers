"""
2D finite-difference Helmholtz solver on a rectangular domain with Dirichlet BCs.

We solve (∇² + k²) p = 0 on (x,y) ∈ [0,Lx]×[0,Ly], using a 5-point stencil
and sparse linear algebra.

Dirichlet boundaries are prescribed on all four edges. Unknowns are the
interior nodes only, with boundary values incorporated into the RHS.

Conventions:
- x has Nx points, y has Ny points
- p is stored as shape (Ny, Nx) so p[j, i] = p(y_j, x_i)
"""
#fd_helmholtz_2d.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


BoundarySpec = Union[complex, float, np.ndarray, Callable[[np.ndarray], np.ndarray]]


@dataclass(frozen=True)
class Field2D:
    x: np.ndarray  # (Nx,)
    y: np.ndarray  # (Ny,)
    p: np.ndarray  # complex, (Ny, Nx)
    omega: float
    c0: float
    rho0: float

    @property
    def k(self) -> float:
        return self.omega / self.c0


def _eval_boundary(
    spec: BoundarySpec,
    coord: np.ndarray,
    expected_len: int,
    name: str,
) -> np.ndarray:
    """
    Convert a boundary specification into a complex vector of length expected_len.

    spec can be:
    - scalar (float/complex): constant boundary value
    - array-like of shape (expected_len,)
    - callable: spec(coord) -> array-like
    """
    if callable(spec):
        vals = np.asarray(spec(coord))
    else:
        vals = np.asarray(spec)

    if vals.ndim == 0:
        out = np.full(expected_len, complex(vals), dtype=np.complex128)
        return out

    if vals.shape != (expected_len,):
        raise ValueError(
            f"{name} must be a scalar, callable returning shape ({expected_len},), "
            f"or an array of shape ({expected_len},). Got {vals.shape}."
        )
    return vals.astype(np.complex128, copy=False)

from typing import Literal

BCType = Literal["dirichlet", "neumann"]


def solve_helmholtz_2d_mixed_bc(
    Lx: float,
    Ly: float,
    Nx: int,
    Ny: int,
    f: float,
    c0: float,
    rho0: float,
    # For each side: type + spec
    left_type: BCType = "neumann",
    right_type: BCType = "neumann",
    bottom_type: BCType = "neumann",
    top_type: BCType = "neumann",
    left: BoundarySpec = 0.0 + 0.0j,
    right: BoundarySpec = 0.0 + 0.0j,
    bottom: BoundarySpec = 0.0 + 0.0j,
    top: BoundarySpec = 0.0 + 0.0j,
    loss_eta: float = 1e-3,
) -> Field2D:
    """
    Solve 2D Helmholtz on a rectangular grid with mixed BCs:
      - Dirichlet: p = value
      - Neumann:   ∂p/∂n = value  (units: pressure per meter)

    Implementation:
      - Unknowns are ALL grid nodes (Ny*Nx).
      - Interior rows use 5-point Helmholtz stencil.
      - Boundary rows are replaced with BC equations (2nd-order one-sided).
    """
    if Nx < 3 or Ny < 3:
        raise ValueError("Nx and Ny must be >= 3.")

    x = np.linspace(0.0, Lx, Nx)
    y = np.linspace(0.0, Ly, Ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    omega = 2.0 * np.pi * f
    k = omega / c0
    k2 = (k**2) * (1.0 + 1j * loss_eta)

    # Evaluate boundary specs as vectors on the boundary nodes
    # left/right vary along y (Ny); bottom/top vary along x (Nx)
    gL = _eval_boundary(left, y, Ny, "left")
    gR = _eval_boundary(right, y, Ny, "right")
    gB = _eval_boundary(bottom, x, Nx, "bottom")
    gT = _eval_boundary(top, x, Nx, "top")

    N = Nx * Ny

    def idx(j: int, i: int) -> int:
        return j * Nx + i  # row-major: y then x

    A = sp.lil_matrix((N, N), dtype=np.complex128)
    b = np.zeros(N, dtype=np.complex128)

    # --- Interior: 5-point stencil for (∇² + k²)p = 0 ---
    invdx2 = 1.0 / (dx * dx)
    invdy2 = 1.0 / (dy * dy)

    for j in range(1, Ny - 1):
        for i in range(1, Nx - 1):
            r = idx(j, i)
            A[r, idx(j, i)] = (-2.0 * invdx2 - 2.0 * invdy2) + k2
            A[r, idx(j, i - 1)] = invdx2
            A[r, idx(j, i + 1)] = invdx2
            A[r, idx(j - 1, i)] = invdy2
            A[r, idx(j + 1, i)] = invdy2
            b[r] = 0.0

    # Helper to write Dirichlet row: p = value
    def set_dirichlet(r: int, node: int, value: complex):
        A[r, :] = 0.0
        A[r, node] = 1.0
        b[r] = value

    # Helper to write Neumann row using 2nd-order one-sided derivative
    # Left (x=0):  (-3 p0 + 4 p1 - p2)/(2 dx) = ∂p/∂x
    # Right(x=L): ( 3 pN - 4 pN-1 + pN-2)/(2 dx) = ∂p/∂x
    # Bottom(y=0): (-3 p0 + 4 p1 - p2)/(2 dy) = ∂p/∂y
    # Top(y=Ly):   ( 3 pN - 4 pN-1 + pN-2)/(2 dy) = ∂p/∂y
    def set_neumann_left(j: int, value: complex):
        r = idx(j, 0)
        A[r, :] = 0.0
        A[r, idx(j, 0)] = -3.0 / (2.0 * dx)
        A[r, idx(j, 1)] =  4.0 / (2.0 * dx)
        A[r, idx(j, 2)] = -1.0 / (2.0 * dx)
        b[r] = value  # ∂p/∂x at x=0

    def set_neumann_right(j: int, value: complex):
        r = idx(j, Nx - 1)
        A[r, :] = 0.0
        A[r, idx(j, Nx - 1)] =  3.0 / (2.0 * dx)
        A[r, idx(j, Nx - 2)] = -4.0 / (2.0 * dx)
        A[r, idx(j, Nx - 3)] =  1.0 / (2.0 * dx)
        b[r] = value  # ∂p/∂x at x=Lx

    def set_neumann_bottom(i: int, value: complex):
        r = idx(0, i)
        A[r, :] = 0.0
        A[r, idx(0, i)] = -3.0 / (2.0 * dy)
        A[r, idx(1, i)] =  4.0 / (2.0 * dy)
        A[r, idx(2, i)] = -1.0 / (2.0 * dy)
        b[r] = value  # ∂p/∂y at y=0

    def set_neumann_top(i: int, value: complex):
        r = idx(Ny - 1, i)
        A[r, :] = 0.0
        A[r, idx(Ny - 1, i)] =  3.0 / (2.0 * dy)
        A[r, idx(Ny - 2, i)] = -4.0 / (2.0 * dy)
        A[r, idx(Ny - 3, i)] =  1.0 / (2.0 * dy)
        b[r] = value  # ∂p/∂y at y=Ly

    # --- Left boundary (i=0), excluding corners (we'll set corners last) ---
    for j in range(1, Ny - 1):
        if left_type == "dirichlet":
            set_dirichlet(idx(j, 0), idx(j, 0), gL[j])
        else:
            # outward normal at left is -x, but we usually specify ∂p/∂x.
            # For rigid wall: value=0 works either way.
            set_neumann_left(j, gL[j])

    # --- Right boundary (i=Nx-1) ---
    for j in range(1, Ny - 1):
        if right_type == "dirichlet":
            set_dirichlet(idx(j, Nx - 1), idx(j, Nx - 1), gR[j])
        else:
            set_neumann_right(j, gR[j])

    # --- Bottom boundary (j=0) ---
    for i in range(1, Nx - 1):
        if bottom_type == "dirichlet":
            set_dirichlet(idx(0, i), idx(0, i), gB[i])
        else:
            set_neumann_bottom(i, gB[i])

    # --- Top boundary (j=Ny-1) ---
    for i in range(1, Nx - 1):
        if top_type == "dirichlet":
            set_dirichlet(idx(Ny - 1, i), idx(Ny - 1, i), gT[i])
        else:
            set_neumann_top(i, gT[i])

    # --- Corners ---
    # Corners touch two boundaries; enforcing two one-sided Neumann equations at a single node
    # is awkward in this row-replacement approach. Robust choice:
    #   - If any adjacent side is Dirichlet -> enforce that Dirichlet value.
    #   - Else (all-neumann corner) -> tie corner pressure to its adjacent *interior* neighbor.
    #
    # This is a numerical closure, not a physical BC; it prevents conflicting constraints and
    # keeps corners from dominating the solve. The physical Neumann behaviour is still enforced
    # on edge nodes adjacent to the corner.

    def set_corner_tie_to_interior(j: int, i: int):
        """Enforce p(j,i) = p(j_interior, i_interior) with the nearest interior neighbor."""
        r = idx(j, i)
        A[r, :] = 0.0
        A[r, idx(j, i)] = 1.0

        # choose nearest interior node (move one step inward in each direction)
        ji = 1 if j == 0 else (Ny - 2)
        ii = 1 if i == 0 else (Nx - 2)

        # tie to interior neighbor
        A[r, idx(ji, ii)] = -1.0
        b[r] = 0.0 + 0.0j

    def corner_dirichlet_value(j: int, i: int) -> complex:
        # Prefer Dirichlet if available from any adjacent edge
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

    # --- Gauge fixing for all-Neumann problems ---
    # Pure Neumann specifications can lead to ill-conditioning (and Laplace would be singular).
    # We pin one DOF to remove any remaining near-null modes and make solves repeatable.
    all_neumann = (
        left_type == "neumann"
        and right_type == "neumann"
        and bottom_type == "neumann"
        and top_type == "neumann"
    )
    if all_neumann:
        # Pin an interior node (more neutral than a corner)
        jg, ig = 1, 1
        rg = idx(jg, ig)
        A[rg, :] = 0.0
        A[rg, rg] = 1.0
        b[rg] = 0.0 + 0.0j


    # Solve
    p_vec = spla.spsolve(A.tocsr(), b)
    p = p_vec.reshape((Ny, Nx))

    return Field2D(x=x, y=y, p=p, omega=omega, c0=c0, rho0=rho0)

def solve_helmholtz_2d_neumann_velocity(
    *,
    Lx: float,
    Ly: float,
    Nx: int,
    Ny: int,
    f: float,
    c0: float,
    rho0: float,
    vn_left: BoundarySpec = 0.0 + 0.0j,
    vn_right: BoundarySpec = 0.0 + 0.0j,
    vn_bottom: BoundarySpec = 0.0 + 0.0j,
    vn_top: BoundarySpec = 0.0 + 0.0j,
    loss_eta: float = 1e-3,
) -> Field2D:
    """
    Solve Helmholtz with Neumann actuation specified as wall normal velocity vn (m/s).

    Uses: ∂p/∂n = - i ω ρ0 vn
    and converts outward-normal dpdn into the solver's internal (∂p/∂x or ∂p/∂y) conventions.
    """
    omega = 2.0 * np.pi * f

    def dpdn_from_vn(spec: BoundarySpec) -> BoundarySpec:
        if callable(spec):
            return lambda s: (-1j * omega * rho0) * np.asarray(spec(s))
        return (-1j * omega * rho0) * spec

    # Convert outward-normal ∂p/∂n to the derivative directions enforced in mixed_bc:
    # left boundary enforces ∂p/∂x, but outward normal is -x => ∂p/∂x = -∂p/∂n
    # bottom boundary enforces ∂p/∂y, but outward normal is -y => ∂p/∂y = -∂p/∂n
    def dpx_from_vn_left(spec: BoundarySpec) -> BoundarySpec:
        if callable(spec):
            return lambda s: (+1j * omega * rho0) * np.asarray(spec(s))
        return (+1j * omega * rho0) * spec

    def dpx_from_vn_right(spec: BoundarySpec) -> BoundarySpec:
        if callable(spec):
            return lambda s: (-1j * omega * rho0) * np.asarray(spec(s))
        return (-1j * omega * rho0) * spec

    def dpy_from_vn_bottom(spec: BoundarySpec) -> BoundarySpec:
        if callable(spec):
            return lambda s: (+1j * omega * rho0) * np.asarray(spec(s))
        return (+1j * omega * rho0) * spec

    def dpy_from_vn_top(spec: BoundarySpec) -> BoundarySpec:
        if callable(spec):
            return lambda s: (-1j * omega * rho0) * np.asarray(spec(s))
        return (-1j * omega * rho0) * spec

    left_dpx   = dpx_from_vn_left(vn_left)
    right_dpx  = dpx_from_vn_right(vn_right)
    bottom_dpy = dpy_from_vn_bottom(vn_bottom)
    top_dpy    = dpy_from_vn_top(vn_top)

    return solve_helmholtz_2d_mixed_bc(
        Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny,
        f=f, c0=c0, rho0=rho0,
        left_type="neumann",
        right_type="neumann",
        bottom_type="neumann",
        top_type="neumann",
        left=left_dpx,
        right=right_dpx,
        bottom=bottom_dpy,
        top=top_dpy,
        loss_eta=loss_eta,
    )






def solve_helmholtz_2d_dirichlet(
    Lx: float,
    Ly: float,
    Nx: int,
    Ny: int,
    f: float,
    c0: float,
    rho0: float,
    p_left: BoundarySpec = 1.0 + 0.0j,
    p_right: BoundarySpec = 0.0 + 0.0j,
    p_bottom: BoundarySpec = 0.0 + 0.0j,
    p_top: BoundarySpec = 0.0 + 0.0j,
    loss_eta: float = 1e-3,
) -> Field2D:
    """
    Solve 2D Helmholtz with Dirichlet boundary conditions.

    Parameters
    ----------
    Lx, Ly : float
        Domain size in meters.
    Nx, Ny : int
        Number of grid points in x and y (including boundaries).
        Must be >= 3.
    f : float
        Frequency (Hz).
    c0, rho0 : float
        Medium sound speed (m/s) and density (kg/m^3).
    p_left, p_right : BoundarySpec
        Dirichlet boundary value at x=0 and x=Lx. Either scalar, array of length Ny,
        or callable of y -> values.
    p_bottom, p_top : BoundarySpec
        Dirichlet boundary value at y=0 and y=Ly. Either scalar, array of length Nx,
        or callable of x -> values.
    loss_eta : float
        Small loss factor for conditioning: k^2 -> k^2 (1 + i*loss_eta)

    Returns
    -------
    Field2D
        Field object with x, y, and complex pressure p (Ny, Nx).
    """
    if Nx < 3 or Ny < 3:
        raise ValueError("Nx and Ny must be >= 3 (need at least one interior node).")

    x = np.linspace(0.0, Lx, Nx)
    y = np.linspace(0.0, Ly, Ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    omega = 2.0 * np.pi * f
    k = omega / c0
    k2 = (k**2) * (1.0 + 1j * loss_eta)

    # Interior sizes
    nx = Nx - 2
    ny = Ny - 2
    N = nx * ny

    # Build 1D second-derivative matrices for interior nodes
    ex = np.ones(nx, dtype=np.float64)
    ey = np.ones(ny, dtype=np.float64)

    Tx = sp.diags([ex, -2.0 * ex, ex], offsets=[-1, 0, 1], shape=(nx, nx), format="csr") / (dx**2)
    Ty = sp.diags([ey, -2.0 * ey, ey], offsets=[-1, 0, 1], shape=(ny, ny), format="csr") / (dy**2)

    Ix = sp.eye(nx, format="csr")
    Iy = sp.eye(ny, format="csr")

    # 2D Laplacian using Kronecker products
    L = sp.kron(Iy, Tx, format="csr") + sp.kron(Ty, Ix, format="csr")

    A = L + (k2 * sp.eye(N, format="csr"))

    # Evaluate boundaries
    pL = _eval_boundary(p_left, y, Ny, "p_left")
    pR = _eval_boundary(p_right, y, Ny, "p_right")
    pB = _eval_boundary(p_bottom, x, Nx, "p_bottom")
    pT = _eval_boundary(p_top, x, Nx, "p_top")

    # Assemble RHS from boundary contributions
    # Unknown vector ordering: u[j, i] with j=0..ny-1 (y interior), i=0..nx-1 (x interior)
    # Flattened index: idx = j*nx + i
    b = np.zeros(N, dtype=np.complex128)

    def idx(j: int, i: int) -> int:
        return j * nx + i

    # Loop boundaries only (O(N) with tiny constant). Clear + robust.
    for j in range(ny):
        jj = j + 1  # maps to full grid y-index
        for i in range(nx):
            ii = i + 1  # maps to full grid x-index
            r = idx(j, i)

            # Neighbor at x=0 boundary for i==0
            if i == 0:
                b[r] -= (pL[jj] / (dx**2))
            # Neighbor at x=Lx boundary for i==nx-1
            if i == nx - 1:
                b[r] -= (pR[jj] / (dx**2))

            # Neighbor at y=0 boundary for j==0
            if j == 0:
                b[r] -= (pB[ii] / (dy**2))
            # Neighbor at y=Ly boundary for j==ny-1
            if j == ny - 1:
                b[r] -= (pT[ii] / (dy**2))

    # Solve sparse system
    u = spla.spsolve(A, b)

    # Reconstruct full field including boundaries
    p = np.zeros((Ny, Nx), dtype=np.complex128)

    # Fill boundaries
    p[:, 0] = pL
    p[:, -1] = pR
    p[0, :] = pB
    p[-1, :] = pT

    # Fill interior
    p[1:-1, 1:-1] = u.reshape((ny, nx))

    return Field2D(x=x, y=y, p=p, omega=omega, c0=c0, rho0=rho0)
