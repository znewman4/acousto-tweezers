"""
Inertial particle ensemble state for vortex entry diagnostics.

Holds positions and velocities for N particles as (N, 2) arrays.
Pure data — no physics, no I/O.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class ParticleEnsemble:
    """
    State of N inertial particles.

    Attributes
    ----------
    pos : np.ndarray, shape (N, 2), float64
        Particle positions [m].  pos[:, 0] = x, pos[:, 1] = y.
    vel : np.ndarray, shape (N, 2), float64
        Particle velocities [m/s].
    """
    pos: np.ndarray   # (N, 2)  [m]
    vel: np.ndarray   # (N, 2)  [m/s]

    @property
    def N(self) -> int:
        return self.pos.shape[0]

    def copy(self) -> "ParticleEnsemble":
        return ParticleEnsemble(
            pos=self.pos.copy(),
            vel=self.vel.copy(),
        )


def make_grid_ensemble(
    view_xlim: Tuple[float, float],
    view_ylim: Tuple[float, float],
    n_per_side: int,
) -> ParticleEnsemble:
    """
    Initialise a regular n_per_side × n_per_side grid of particles.

    Parameters
    ----------
    view_xlim : (xmin_mm, xmax_mm)
        Horizontal extent of the view window [mm].
    view_ylim : (ymin_mm, ymax_mm)
        Vertical extent of the view window [mm].
    n_per_side : int
        Number of grid points along each axis.  Total N = n_per_side².

    Returns
    -------
    ParticleEnsemble
        Initial positions covering the view window, zero velocities.
    """
    xs = np.linspace(view_xlim[0] * 1e-3, view_xlim[1] * 1e-3, n_per_side)
    ys = np.linspace(view_ylim[0] * 1e-3, view_ylim[1] * 1e-3, n_per_side)
    XX, YY = np.meshgrid(xs, ys)
    pos = np.column_stack([XX.ravel(), YY.ravel()]).astype(np.float64)
    vel = np.zeros_like(pos)
    return ParticleEnsemble(pos=pos, vel=vel)
