"""
Acoustic streaming solver.

This module implements:
- Time-averaged forcing from first-order velocity
- Steady Stokes solver for mean flow
- Boundary conditions for streaming (no-slip, stress-free)
"""
from .forcing import compute_streaming_force, StreamingForcing
from .solver import StreamingSolver, StokesSolver, StreamingField

__all__ = [
    "compute_streaming_force",
    "StreamingForcing",
    "StreamingSolver",
    "StokesSolver",
    "StreamingField",
]
