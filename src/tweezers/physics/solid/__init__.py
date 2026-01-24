"""
Solid mechanics solver for elastic dish plate and walls.

This module implements:
- Frequency-domain elasticity with viscoelastic damping
- Complex modulus for material loss
- Fluid-solid coupling interface conditions
"""
from .materials import SolidMaterial, ViscoelasticMaterial, SolidMaterialDatabase
from .solver import ElasticSolver, DisplacementField3D
from .coupling import FluidSolidCoupling, PlateTransmission

__all__ = [
    "SolidMaterial",
    "ViscoelasticMaterial",
    "SolidMaterialDatabase",
    "ElasticSolver",
    "DisplacementField3D",
    "FluidSolidCoupling",
    "PlateTransmission",
]
