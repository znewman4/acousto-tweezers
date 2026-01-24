"""
Multi-domain acoustic solver with PML and thermoviscous effects.

This module implements:
- Linear acoustics in heterogeneous media (water, air, bath)
- Fluid-fluid interface coupling (pressure/velocity continuity)
- PML (Perfectly Matched Layer) for open boundaries
- Thermoviscous acoustics near walls
"""
from .materials import FluidMaterial, MaterialDatabase
from .geometry import MultiDomainGeometry, DomainType, InterfaceType
from .pml import PMLRegion, PMLProfile, PMLManager, PMLParameters
from .thermoviscous import ThermoviscousLayer, compute_boundary_layer_thickness
from .solver import MultiDomainAcousticSolver, AcousticField3D

__all__ = [
    "FluidMaterial",
    "MaterialDatabase",
    "MultiDomainGeometry",
    "DomainType",
    "InterfaceType",
    "PMLRegion",
    "PMLProfile",
    "PMLManager",
    "PMLParameters",
    "ThermoviscousLayer",
    "compute_boundary_layer_thickness",
    "MultiDomainAcousticSolver",
    "AcousticField3D",
]
