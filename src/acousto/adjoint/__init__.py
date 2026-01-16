# src/acousto/adjoint/__init__.py
"""Adjoint gradient computations for acoustic field optimization."""

from .gradients import (
    TransducerParams,
    compute_dJdp_pressure_at_point,
    compute_dJdp_simple_real_pressure,
    compute_dJdp_complex_pressure,
    compute_dbdu_single_transducer,
    compute_dbdu_position,
    adjoint_gradient,
    adjoint_gradient_vectorized,
)

__all__ = [
    "TransducerParams",
    "compute_dJdp_pressure_at_point",
    "compute_dJdp_simple_real_pressure",
    "compute_dJdp_complex_pressure",
    "compute_dbdu_single_transducer",
    "compute_dbdu_position",
    "adjoint_gradient",
    "adjoint_gradient_vectorized",
]
