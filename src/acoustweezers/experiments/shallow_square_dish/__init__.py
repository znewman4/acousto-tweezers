"""
Shallow Square Dish Experiment Module.

Device-realistic 3D model:
- 5 cm × 5 cm × 5 mm shallow dish
- Bottom (z=0): Vortex lens actuation
- Side walls: Standing wave transducers
- Top (z=H): Free surface / air interface

Author: Acousto-Tweezers Project
Date: 2026-02-08
"""

from .config import ShallowDishConfig
from .solve_pressure import solve_all_pressure_cases
from .streaming import compute_streaming_velocity
from .particles import integrate_particle_trajectory
from .export import export_all_fields

__all__ = [
    "ShallowDishConfig",
    "solve_all_pressure_cases",
    "compute_streaming_velocity",
    "integrate_particle_trajectory",
    "export_all_fields",
]
