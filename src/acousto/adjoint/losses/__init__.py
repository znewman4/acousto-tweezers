# src/acousto/adjoint/losses/__init__.py
"""Loss functions for adjoint-based trajectory optimization."""

from .circle_track import (
    CircleLossConfig,
    circle_metrics,
    compute_step_loss,
    compute_trajectory_loss,
    compute_dL_dx_t,
    compute_dL_dx_tp1,
    compute_dL_du,
)

from .path_track import (
    Path,
    CirclePath,
    PolylinePath,
    PathLossConfig,
    path_metrics,
    compute_progress,
    compute_step_loss as path_compute_step_loss,
    compute_trajectory_loss as path_compute_trajectory_loss,
    compute_dL_dx_t as path_compute_dL_dx_t,
    compute_dL_dx_tp1 as path_compute_dL_dx_tp1,
    compute_dL_du as path_compute_dL_du,
)

__all__ = [
    # Circle tracking (legacy)
    'CircleLossConfig',
    'circle_metrics',
    'compute_step_loss',
    'compute_trajectory_loss',
    'compute_dL_dx_t',
    'compute_dL_dx_tp1',
    'compute_dL_du',
    # Path tracking (general)
    'Path',
    'CirclePath',
    'PolylinePath',
    'PathLossConfig',
    'path_metrics',
    'compute_progress',
    'path_compute_step_loss',
    'path_compute_trajectory_loss',
    'path_compute_dL_dx_t',
    'path_compute_dL_dx_tp1',
    'path_compute_dL_du',
]
