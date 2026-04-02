"""
Scheduling helpers for high-resolution vortex entry diagnostics.

Provides non-uniform path sampling (denser near the merge target)
and adaptive snapshot scheduling for GIF frame capture.
"""
from __future__ import annotations

from typing import Set, Tuple

import numpy as np


def dense_late_path(
    start: np.ndarray,
    end: np.ndarray,
    n_steps: int,
    late_frac: float = 0.3,
    density_boost: int = 3,
    path_mode: str = "straight",
    arc_height_mm: float = 0.0,
) -> np.ndarray:
    """
    Build a vortex-centre path with denser waypoints in the final
    ``late_frac`` of the journey (near the merge target B).

    The first ``(1 - late_frac)`` of the normalised parameter ``t``
    gets ``n_early`` uniformly spaced points; the remaining
    ``late_frac`` gets ``density_boost`` times as many points per
    unit ``t``.

    Parameters
    ----------
    start, end : (2,) arrays — path endpoints [m].
    n_steps : base step count (comparable to the original uniform path).
    late_frac : fraction of the [0,1] parameter range considered "late".
    density_boost : multiplier for point density in the late region.
    path_mode : ``"straight"`` | ``"arc_left"`` | ``"arc_right"``.
    arc_height_mm : Bézier sag for arc modes [mm].

    Returns
    -------
    path : (N, 2) array of waypoint positions [m], where
           N = n_early + n_late (generally > n_steps).
    """
    t_split = 1.0 - late_frac

    # Points per unit-t in early region
    pts_per_unit_early = n_steps / 1.0  # baseline density
    n_early = max(1, int(round(pts_per_unit_early * t_split)))
    n_late = max(1, int(round(pts_per_unit_early * late_frac * density_boost)))

    t_early = np.linspace(0.0, t_split, n_early, endpoint=False)
    t_late = np.linspace(t_split, 1.0, n_late, endpoint=True)
    t = np.concatenate([t_early, t_late])

    return _eval_path(start, end, t, path_mode, arc_height_mm)


def _eval_path(
    start: np.ndarray,
    end: np.ndarray,
    t: np.ndarray,
    path_mode: str,
    arc_height_mm: float,
) -> np.ndarray:
    """Evaluate straight or Bézier-arc path at arbitrary t values."""
    if path_mode == "straight" or arc_height_mm == 0.0:
        return start[None, :] + t[:, None] * (end - start)[None, :]

    d = end - start
    perp = np.array([-d[1], d[0]])
    perp = perp / np.linalg.norm(perp)
    sign = 1.0 if path_mode == "arc_left" else -1.0
    mid_ctrl = 0.5 * (start + end) + sign * arc_height_mm * 1e-3 * perp

    path = np.zeros((len(t), 2))
    for i, ti in enumerate(t):
        path[i] = ((1 - ti) ** 2 * start
                    + 2 * (1 - ti) * ti * mid_ctrl
                    + ti ** 2 * end)
    return path


def adaptive_snapshot_schedule(
    n_vortex_steps: int,
    n_dyn_steps: int,
    base_fps: int = 4,
    late_fps: int = 12,
    late_frac: float = 0.3,
) -> Set[Tuple[int, int]]:
    """
    Return a set of ``(v_step, dyn_i)`` pairs at which to capture a
    GIF frame.

    Early vortex steps (first ``1 - late_frac``) get ``base_fps``
    evenly-spaced frames per step; late steps get ``late_fps``.

    Parameters
    ----------
    n_vortex_steps : total vortex path waypoints.
    n_dyn_steps : dynamics sub-steps per vortex position.
    base_fps : frames per vortex step in early phase.
    late_fps : frames per vortex step in late phase.
    late_frac : fraction of vortex steps considered "late".

    Returns
    -------
    snaps : set of (v_step, dyn_i) tuples.
    """
    late_start = int(round(n_vortex_steps * (1 - late_frac)))
    snaps: Set[Tuple[int, int]] = set()

    for vs in range(n_vortex_steps):
        fps = late_fps if vs >= late_start else base_fps
        indices = np.linspace(0, n_dyn_steps - 1, fps, dtype=int)
        for di in indices:
            snaps.add((vs, int(di)))
    return snaps
