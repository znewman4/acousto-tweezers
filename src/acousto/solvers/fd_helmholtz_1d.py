from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.sparse import diags, csc_matrix
from scipy.sparse.linalg import spsolve


@dataclass(frozen=True)
class Field1D:
    """Complex acoustic pressure field p(x) for a 1D domain."""
    x: np.ndarray          # shape (N,)
    p: np.ndarray          # complex, shape (N,)
    omega: float           # rad/s
    c0: float              # m/s
    rho0: float            # kg/m^3

    @property
    def k(self) -> float:
        return self.omega / self.c0


def solve_helmholtz_1d_dirichlet(
    *,
    L: float,
    N: int,
    f: float,
    c0: float,
    rho0: float,
    p_left: complex = 1.0 + 0.0j,
    p_right: complex = 0.0 + 0.0j,
    loss_eta: float = 1e-3,
) -> Field1D:
    """
    Solve 1D Helmholtz: p_xx + k^2 p = 0 on x∈[0,L]
    with Dirichlet BCs p(0)=p_left, p(L)=p_right.

    loss_eta adds small complex loss: k^2 -> k^2*(1 + i*loss_eta)
    to improve conditioning near resonances.

    This is a Phase-1 "bootstrap" solver: simple, fast, and analytically checkable.
    """
    if L <= 0:
        raise ValueError("L must be > 0")
    if N < 3:
        raise ValueError("N must be >= 3")
    if f <= 0:
        raise ValueError("f must be > 0")
    if c0 <= 0 or rho0 <= 0:
        raise ValueError("c0 and rho0 must be > 0")

    x = np.linspace(0.0, L, N)
    h = x[1] - x[0]

    omega = 2.0 * np.pi * f
    k = omega / c0
    k2 = (k**2) * (1.0 + 1j * loss_eta)

    # Unknowns are interior nodes i=1..N-2
    n = N - 2
    main = (-2.0 / h**2 + k2) * np.ones(n, dtype=np.complex128)
    off = (1.0 / h**2) * np.ones(n - 1, dtype=np.complex128)

    A = diags([off, main, off], offsets=[-1, 0, 1], format="csc")  # type: ignore
    b = np.zeros(n, dtype=np.complex128)

    # Dirichlet boundary contributions
    b[0] -= (1.0 / h**2) * p_left
    b[-1] -= (1.0 / h**2) * p_right

    p_int = spsolve(csc_matrix(A), b)

    p = np.zeros(N, dtype=np.complex128)
    p[0] = p_left
    p[-1] = p_right
    p[1:-1] = p_int

    return Field1D(x=x, p=p, omega=omega, c0=c0, rho0=rho0)


def analytic_standing_wave_dirichlet(
    x: np.ndarray, *, L: float, k: float, p_left: complex, p_right: complex
) -> np.ndarray:
    """
    Analytic solution of 1D Helmholtz with Dirichlet BCs:
      p(0)=p_left, p(L)=p_right
    General solution: p = A cos(kx) + B sin(kx)
    """
    # Handle near-zero k robustly (not expected in acoustics, but keep safe)
    if abs(k) < 1e-12:
        return p_left + (p_right - p_left) * (x / L)

    A = p_left
    B = (p_right - p_left * np.cos(k * L)) / np.sin(k * L)
    return A * np.cos(k * x) + B * np.sin(k * x)
