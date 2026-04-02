"""
Interpolation helpers for vortex entry experiment.

Exact copies of the inline helpers from vortex_entry_test.py.
No logic changes.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator


def make_interp(
    F: np.ndarray,
    xg: np.ndarray,
    yg: np.ndarray,
) -> RegularGridInterpolator:
    """Build a RegularGridInterpolator for a 2-D field F(yg, xg)."""
    return RegularGridInterpolator(
        (yg, xg), F, bounds_error=False, fill_value=0.0)


def eval_at(
    interp: RegularGridInterpolator,
    pos_xy: np.ndarray,
) -> np.ndarray:
    """Evaluate interpolator at pos_xy (N×2, columns are x,y)."""
    pts = np.column_stack([pos_xy[:, 1], pos_xy[:, 0]])
    return interp(pts)
