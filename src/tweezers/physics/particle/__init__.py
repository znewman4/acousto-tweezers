"""
Particle dynamics with radiation force and streaming.

This module implements:
- 3D Gor'kov potential and radiation force
- Overdamped particle dynamics
- Combined radiation + streaming motion
- Trilinear/higher-order interpolation in 3D
"""
from .properties import (
    Particle3D,
    ParticleContrast,
    compute_contrast_factors,
    ParticleDatabase,
)
from .dynamics import (
    ParticleDynamics3D,
    ParticleState,
    ParticleTrajectory,
    StokesianDynamics,
    simulate_multiple_particles,
    compute_characteristic_velocity,
    make_boundary_event,
)
from .interpolation import (
    Grid3D,
    TrilinearInterpolator,
    VectorFieldInterpolator,
    GradientInterpolator,
    SplineInterpolator3D,
    compute_hessian,
)
from .gorkov import (
    GorkovPotential3D,
    estimate_max_radiation_force,
    compute_stiffness,
    find_potential_minima,
)

__all__ = [
    # Properties
    "Particle3D",
    "ParticleContrast",
    "compute_contrast_factors",
    "ParticleDatabase",
    # Dynamics
    "ParticleDynamics3D",
    "ParticleState",
    "ParticleTrajectory",
    "StokesianDynamics",
    "simulate_multiple_particles",
    "compute_characteristic_velocity",
    "make_boundary_event",
    # Interpolation
    "Grid3D",
    "TrilinearInterpolator",
    "VectorFieldInterpolator",
    "GradientInterpolator",
    "SplineInterpolator3D",
    "compute_hessian",
    # Gorkov
    "GorkovPotential3D",
    "estimate_max_radiation_force",
    "compute_stiffness",
    "find_potential_minima",
]