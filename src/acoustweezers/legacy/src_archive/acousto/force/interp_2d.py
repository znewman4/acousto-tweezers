from __future__ import annotations

import numpy as np


def bilinear_sample(x: np.ndarray, y: np.ndarray, F: np.ndarray, xq: float, yq: float) -> float:
    """Bilinearly sample a scalar grid ``F`` at a continuous query point.

    Conventions (matching the rest of this repo):
    - ``x`` has shape (Nx,), increasing
    - ``y`` has shape (Ny,), increasing
    - ``F`` has shape (Ny, Nx) where ``F[j, i]`` corresponds to (x[i], y[j])

    Notes
    -----
    We clamp queries to the grid bounds (no extrapolation). This makes
    optimisation/dynamics more stable near boundaries.
    """
    Nx = int(x.size)
    Ny = int(y.size)
    if Nx < 2 or Ny < 2:
        raise ValueError("Need at least a 2x2 grid for bilinear interpolation.")

    # Clamp to domain
    xq = float(np.clip(xq, float(x[0]), float(x[-1])))
    yq = float(np.clip(yq, float(y[0]), float(y[-1])))

    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    if dx <= 0.0 or dy <= 0.0:
        raise ValueError("x and y must be strictly increasing.")

    # Fractional indices in grid coordinates
    fx = (xq - float(x[0])) / dx
    fy = (yq - float(y[0])) / dy

    i0 = int(np.clip(np.floor(fx), 0, Nx - 2))
    j0 = int(np.clip(np.floor(fy), 0, Ny - 2))

    tx = float(fx - i0)
    ty = float(fy - j0)

    f00 = float(F[j0, i0])
    f10 = float(F[j0, i0 + 1])
    f01 = float(F[j0 + 1, i0])
    f11 = float(F[j0 + 1, i0 + 1])

    return (
        (1.0 - tx) * (1.0 - ty) * f00
        + tx * (1.0 - ty) * f10
        + (1.0 - tx) * ty * f01
        + tx * ty * f11
    )


def bilinear_sample_vec(
    x: np.ndarray,
    y: np.ndarray,
    Fx: np.ndarray,
    Fy: np.ndarray,
    xq: float,
    yq: float,
) -> tuple[float, float]:
    """Bilinearly sample a vector field (Fx, Fy) at query (xq, yq)."""
    return bilinear_sample(x, y, Fx, xq, yq), bilinear_sample(x, y, Fy, xq, yq)
