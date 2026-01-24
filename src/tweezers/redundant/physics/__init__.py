"""
Multiphysics simulation package for acoustic tweezers.

Modules:
- acoustics: Multi-domain acoustic solver with PML and thermoviscous effects
- solid: Elastic wave equation for dish plate and walls
- streaming: Acoustic streaming (second-order mean flow)
- particle: Particle dynamics with radiation force and streaming
- solver: Unified multiphysics orchestrator
"""
from .acoustics import (
    MultiDomainGeometry,
    MultiDomainAcousticSolver,
    PMLRegion,
    DomainType,
    FluidMaterial,
    MaterialDatabase,
    PMLManager,
    AcousticField3D,
)
from .solid import (
    ElasticSolver,
    SolidMaterial,
    FluidSolidCoupling,
)
from .streaming import (
    StreamingSolver,
    StreamingField,
)
from .particle import (
    Particle3D,
    ParticleDatabase,
    ParticleDynamics3D,
    ParticleTrajectory,
    GorkovPotential3D,
)
from .solver import (
    MultiphysicsSolver,
    SimulationParameters,
    MultiphysicsResults,
    run_standard_simulation,
)

__all__ = [
    # Unified solver
    "MultiphysicsSolver",
    "SimulationParameters",
    "MultiphysicsResults",
    "run_standard_simulation",
    # Acoustics
    "MultiDomainGeometry",
    "MultiDomainAcousticSolver",
    "PMLRegion",
    "DomainType",
    "FluidMaterial",
    "MaterialDatabase",
    "PMLManager",
    "AcousticField3D",
    # Solid
    "ElasticSolver",
    "SolidMaterial",
    "FluidSolidCoupling",
    # Streaming
    "StreamingSolver",
    "StreamingField",
    # Particle
    "Particle3D",
    "ParticleDatabase",
    "ParticleDynamics3D",
    "ParticleTrajectory",
    "GorkovPotential3D",
]
