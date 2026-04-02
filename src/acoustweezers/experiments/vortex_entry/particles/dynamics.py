"""
Particle dynamics for vortex entry experiment.

Extracted verbatim from vortex_entry_test.py.  No physics changes.
The update step is the same overdamped integrator with clamped displacement.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from scripts.lib.particle_dynamics_utils import SCALE, DT_DEFAULT

from ..utils.interpolation import eval_at


def update_particles(
    pos_A: np.ndarray,
    pos_B: np.ndarray,
    iFx: RegularGridInterpolator,
    iFy: RegularGridInterpolator,
    max_step: float,
    xg: np.ndarray,
    yg: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Advance particles A and B by one dynamics step.

    Same logic as the inner dynamics loop in the original script:
    1. Evaluate force interpolators at current positions
    2. Overdamped step: pos += clip(SCALE * DT_DEFAULT * F, -max_step, max_step)
    3. Clamp positions to grid interior (2 cells from edge)

    Returns updated (pos_A, pos_B).
    """
    FA = np.array([
        float(eval_at(iFx, pos_A[None, :])[0]),
        float(eval_at(iFy, pos_A[None, :])[0])])
    FB = np.array([
        float(eval_at(iFx, pos_B[None, :])[0]),
        float(eval_at(iFy, pos_B[None, :])[0])])

    pos_A = pos_A + np.clip(
        SCALE * DT_DEFAULT * FA,
        -max_step, max_step)
    pos_B = pos_B + np.clip(
        SCALE * DT_DEFAULT * FB,
        -max_step, max_step)

    pos_A[0] = np.clip(pos_A[0], xg[2], xg[-3])
    pos_A[1] = np.clip(pos_A[1], yg[2], yg[-3])
    pos_B[0] = np.clip(pos_B[0], xg[2], xg[-3])
    pos_B[1] = np.clip(pos_B[1], yg[2], yg[-3])

    return pos_A, pos_B


def b_status(
    pos_B: np.ndarray,
    vortex_center: np.ndarray,
    r_barrier: float,
) -> str:
    """Classify B's position relative to the barrier ring."""
    d = float(np.linalg.norm(pos_B - vortex_center))
    if d > r_barrier * 1.05:
        return "OUTSIDE"
    if d < r_barrier * 0.95:
        return "INSIDE"
    return "ON"
