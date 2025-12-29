from __future__ import annotations
import numpy as np


def simulate_overdamped_2d(
    x: np.ndarray,
    y: np.ndarray,
    Fx: np.ndarray,
    Fy: np.ndarray,
    *,
    x0: float,
    y0: float,
    mu: float,
    a: float,
    dt: float,
    steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Overdamped dynamics: r_dot = F / (6π μ a)

    Force is sampled from the grid with nearest-neighbor lookup (fast + simple).
    """
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    Ny, Nx = Fx.shape

    gamma = 6.0 * np.pi * mu * a  # Stokes drag

    xs = np.empty(steps + 1)
    ys = np.empty(steps + 1)
    xs[0] = x0
    ys[0] = y0

    for n in range(steps):
        i = int(np.clip(round((xs[n] - x[0]) / dx), 0, Nx - 1))
        j = int(np.clip(round((ys[n] - y[0]) / dy), 0, Ny - 1))

        vx = Fx[j, i] / gamma
        vy = Fy[j, i] / gamma

        xs[n + 1] = np.clip(xs[n] + dt * vx, x[0], x[-1])
        ys[n + 1] = np.clip(ys[n] + dt * vy, y[0], y[-1])

    return xs, ys
