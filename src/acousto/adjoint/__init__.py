# src/acousto/adjoint/__init__.py
"""Adjoint gradient computations for acoustic field optimization."""

from .gradients import (
    TransducerParams,
    compute_dJdp_pressure_at_point,
    compute_dJdp_simple_real_pressure,
    compute_dJdp_complex_pressure,
    compute_dJdp_gorkov_potential,
    compute_dbdu_single_transducer,
    compute_dbdu_position,
    adjoint_gradient,
    adjoint_gradient_vectorized,
)

from .trajectory import (
    TrajectoryState,
    compute_dF_dx_fd,
    compute_dF_du_fd,
    compute_dU_dx_fd,
    forward_rollout,
    backward_pass,
    compute_trajectory_gradient,
    gradcheck_trajectory_scalar,
)

__all__ = [
    "TransducerParams",
    "compute_dJdp_pressure_at_point",
    "compute_dJdp_simple_real_pressure",
    "compute_dJdp_complex_pressure",
    "compute_dJdp_gorkov_potential",
    "compute_dbdu_single_transducer",
    "compute_dbdu_position",
    "adjoint_gradient",
    "adjoint_gradient_vectorized",
    # Trajectory adjoint
    "TrajectoryState",
    "compute_dF_dx_fd",
    "compute_dF_du_fd",
    "compute_dU_dx_fd",
    "forward_rollout",
    "backward_pass",
    "compute_trajectory_gradient",
    "gradcheck_trajectory_scalar",
]
