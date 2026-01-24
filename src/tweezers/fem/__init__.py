"""
FEM-based multiphysics acoustic tweezers simulator.

This package provides a research-grade, physically correct implementation
of acoustic trapping simulation using the Finite Element Method.

Architecture
------------
- config.py: Canonical configuration and physics level management
- domains.py: Domain tagging abstraction (no magic integers)
- geometry.py: FEM mesh and domain specification
- materials.py: Centralized material property database
- fem_core.py: Core FEM infrastructure (mesh, elements, assembly)
- acoustics.py: First-order acoustics (Helmholtz with weak form)
- solids.py: Frequency-domain elastodynamics with damping
- coupling.py: Fluid-solid interface conditions
- pml.py: Perfectly Matched Layer implementation
- thermoviscous.py: Boundary layer corrections
- streaming.py: Acoustic streaming (steady Stokes)
- particles.py: Gor'kov force and particle dynamics
- diagnostics.py: Automatic sanity checks and validation
- solver.py: Unified multiphysics solver

Physics Ladder
--------------
1. ACOUSTICS_ONLY: Helmholtz in water domain
2. ACOUSTICS_PML: + PML boundaries
3. FLUID_AIR_BATH: + air and bath domains
4. FLUID_SOLID: + elastic plate and walls
5. THERMOVISCOUS: + boundary layer effects
6. STREAMING: + mean flow from Reynolds stress
7. PARTICLES: + radiation force and trajectories

References
----------
- Settnes & Bruus (2012): Gor'kov potential theory
- Nama et al. (2015): Acoustic streaming in microchannels
- Dual & Möller (2012): Piezoelectric transducer modeling
"""

from .config import (
    FEMConfig,
    PhysicsLevel,
    GeometryConfig,
    PhysicsConfig,
    SolverConfig,
    OutputConfig,
)
from .domains import Domain, DomainType, Interface, InterfaceType
from .materials import (
    FluidMaterial,
    SolidMaterial,
    ParticleMaterial,
    MaterialDatabase,
)
from .geometry import FEMMesh, create_petri_dish_mesh
from .acoustics import AcousticField, FEMAcousticSolver
from .solids import DisplacementField, FEMSolidSolver
from .pml import PMLParameters, PMLMetrics, PMLHandler
from .thermoviscous import (
    ThermoviscousParameters,
    ThermoviscousCorrection,
    ThermoviscousSolver,
)
from .streaming import StreamingField, StreamingSolver
from .particles import (
    GorkovPotential,
    ParticleTrajectory,
    ParticleDynamics,
    compute_gorkov_potential,
)
from .solver import FEMMultiphysicsSolver, MultiphysicsResult, run_simulation
from .diagnostics import (
    Diagnostics,
    DiagnosticReport,
    DiagnosticResult,
    DiagnosticLevel,
)

__all__ = [
    # Configuration
    'FEMConfig',
    'PhysicsLevel',
    'GeometryConfig',
    'PhysicsConfig',
    'SolverConfig',
    'OutputConfig',
    # Domains
    'Domain',
    'DomainType',
    'Interface',
    'InterfaceType',
    # Materials
    'FluidMaterial',
    'SolidMaterial',
    'ParticleMaterial',
    'MaterialDatabase',
    # Geometry
    'FEMMesh',
    'create_petri_dish_mesh',
    # Acoustics
    'AcousticField',
    'FEMAcousticSolver',
    # Solids
    'DisplacementField',
    'FEMSolidSolver',
    # PML
    'PMLParameters',
    'PMLMetrics',
    'PMLHandler',
    # Thermoviscous
    'ThermoviscousParameters',
    'ThermoviscousCorrection',
    'ThermoviscousSolver',
    # Streaming
    'StreamingField',
    'StreamingSolver',
    # Particles
    'GorkovPotential',
    'ParticleTrajectory',
    'ParticleDynamics',
    'compute_gorkov_potential',
    # Solver
    'FEMMultiphysicsSolver',
    'MultiphysicsResult',
    'run_simulation',
    # Diagnostics
    'Diagnostics',
    'DiagnosticReport',
    'DiagnosticResult',
    'DiagnosticLevel',
]
