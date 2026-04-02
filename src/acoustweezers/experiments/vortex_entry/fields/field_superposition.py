"""
Field superposition for vortex entry experiment.

Extracted verbatim from vortex_entry_test.py.  No physics changes.
"""
from __future__ import annotations

import numpy as np

from scripts.lib.perturbation_vortex import VortexPerturbation


def total_pressure(
    p_sw: np.ndarray,
    vortex_gen: VortexPerturbation,
    psi: float,
    alpha: float,
    beta: float,
    center_xy: np.ndarray,
) -> np.ndarray:
    """Compute total pressure field: beta * p_sw + alpha * exp(i*psi) * p_vortex."""
    p_v = vortex_gen.get_field(center_xy)
    return beta * p_sw + alpha * np.exp(1j * psi) * p_v
