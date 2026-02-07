"""
AcoustWeezers: Acoustic Tweezers Simulation Package

A unified package for modeling acoustic tweezers with FEniCSx.
Replaces the previous acousto/tweezers split.
"""

__version__ = "0.1.0"

# Expose key modules for convenience
from . import core
from . import physics
from . import numerics
from . import viz
from . import experiments

__all__ = ['core', 'physics', 'numerics', 'viz', 'experiments']
