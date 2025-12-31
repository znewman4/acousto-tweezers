# src/acousto/analysis/traps_2d.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Trap2D:
    x: float
    y: float
    U: float
    Fx: float
    Fy: float
    K: np.ndarray  # 2x2 stiffness matrix (Hessian of U)
    eigvals: np.ndarray  # (2,)
    eigvecs: np.ndarray  # (2,2)


def _hessian_from_U(U: np.ndarray, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Hessian components Uxx, Uxy, Uyy using finite differences.
    Uses numpy.gradient twice. Shape preserved.
    """
    dUdy, dUdx = np.gradient(U, dy, dx, edge_order=2)
    d2Udy2, d2Udydx = np.gradient(dUdy, dy, dx, edge_order=2)
    d2Udydx2, d2Udx2 = np.gradient(dUdx, dy, dx, edge_order=2)
    # d2Udydx and d2Udydx2 should be similar; take the average for symmetry
    Uxy = 0.5 * (d2Udydx + d2Udydx2)
    Uxx = d2Udx2
    Uyy = d2Udy2
    return Uxx, Uxy, Uyy


def find_traps_from_force(
    x: np.ndarray,
    y: np.ndarray,
    U: np.ndarray,
    Fx: np.ndarray,
    Fy: np.ndarray,
    *,
    max_traps: int = 10,
    force_rel_thresh: float = 0.03,
    border: int = 2,
) -> list[Trap2D]:
    """
    Heuristic trap finder: identify local minima of |F| and return traps with stiffness.

    - Finds candidates where |F| is locally minimal in a 3x3 neighborhood.
    - Filters by |F| <= force_rel_thresh * max(|F|) (relative threshold).
    - Computes Hessian of U and returns eigenpairs for stability info.

    Notes:
      U, Fx, Fy are shape (Ny, Nx) with y-axis first.
    """
    Ny, Nx = U.shape
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    Fmag = np.sqrt(Fx**2 + Fy**2)
    Fmax = float(np.max(Fmag))
    if Fmax == 0.0:
        return []

    thresh = force_rel_thresh * Fmax

    # Hessian for stiffness
    Uxx, Uxy, Uyy = _hessian_from_U(U, dx, dy)

    candidates: list[tuple[float, int, int]] = []

    # avoid edges where derivatives are less reliable
    j0 = border
    j1 = Ny - border
    i0 = border
    i1 = Nx - border

    for j in range(j0, j1):
        for i in range(i0, i1):
            val = Fmag[j, i]
            if val > thresh:
                continue
            # local minimum in 3x3
            nb = Fmag[j-1:j+2, i-1:i+2]
            if val <= np.min(nb):
                candidates.append((float(val), j, i))

    # sort by smallest |F|
    candidates.sort(key=lambda t: t[0])

    traps: list[Trap2D] = []
    used = np.zeros((Ny, Nx), dtype=bool)

    for _, j, i in candidates:
        if len(traps) >= max_traps:
            break
        # simple de-duplication: skip if near an already-chosen trap
        if used[j-2:j+3, i-2:i+3].any():
            continue
        used[j-2:j+3, i-2:i+3] = True

        K = np.array([[Uxx[j, i], Uxy[j, i]],
                      [Uxy[j, i], Uyy[j, i]]], dtype=float)
        eigvals, eigvecs = np.linalg.eigh(K)

        traps.append(
            Trap2D(
                x=float(x[i]),
                y=float(y[j]),
                U=float(U[j, i]),
                Fx=float(Fx[j, i]),
                Fy=float(Fy[j, i]),
                K=K,
                eigvals=eigvals,
                eigvecs=eigvecs,
            )
        )

    return traps
