"""
FEniCSx-based multiphysics acoustic tweezers simulation.

This package provides a research-grade FEM implementation for simulating
acoustic tweezers in a Petri dish geometry using FEniCSx (dolfinx + PETSc).

Physics Ladder:
    Level 1: ACOUSTICS_ONLY    - Helmholtz equation in fluids
    Level 2: ACOUSTICS_PML     - + PML boundary conditions
    Level 3: FLUID_AIR_BATH    - + Air and bath fluid domains
    Level 4: FLUID_SOLID       - + Elastic solid coupling
    Level 5: THERMOVISCOUS     - + Boundary layer corrections
    Level 6: STREAMING         - + Acoustic streaming
    Level 7: PARTICLES         - + Radiation force and particles

All physics is implemented using UFL variational forms, assembled by
DOLFINx, and solved by PETSc. No homebrew FEM.

Entry Point:
    python -m tweezers.fenicsx.run_multiphysics

Or use the solver directly:
    from tweezers.fenicsx import FEMConfig, FEMMultiphysicsSolver
    
    config = FEMConfig.default()
    solver = FEMMultiphysicsSolver(config)
    result = solver.solve("results/my_run")

Author: Acousto-Tweezers Project
Date: January 2026
"""

from .config import FEMConfig, PhysicsLevel, GeometryConfig, PhysicsConfig
from .domains import Domain, Interface
from .materials import MaterialDatabase, FluidMaterial, SolidMaterial, ParticleMaterial
from .geometry import create_petri_dish_geometry, MeshInfo
from .acoustics import AcousticSolver, AcousticField
from .solids import SolidSolver, DisplacementField
from .coupling import CoupledSolver, CoupledField
from .pml import PMLHandler, PMLMetrics
from .thermoviscous import ThermoviscousSolver, ThermoviscousCorrection
from .streaming import StreamingSolver, StreamingField
from .particles import ParticleDynamics, GorkovPotential, ParticleTrajectory
from .solver import FEMMultiphysicsSolver, MultiphysicsResult, run_simulation
from .diagnostics import DiagnosticsReport, compute_diagnostics

__all__ = [
    # Configuration
    'FEMConfig',
    'PhysicsLevel',
    'GeometryConfig',
    'PhysicsConfig',
    
    # Domains
    'Domain',
    'Interface',
    
    # Materials
    'MaterialDatabase',
    'FluidMaterial',
    'SolidMaterial',
    'ParticleMaterial',
    
    # Geometry
    'create_petri_dish_geometry',
    'MeshInfo',
    
    # Solvers
    'AcousticSolver',
    'SolidSolver',
    'CoupledSolver',
    'StreamingSolver',
    'ParticleDynamics',
    'FEMMultiphysicsSolver',
    
    # Results
    'AcousticField',
    'DisplacementField',
    'CoupledField',
    'StreamingField',
    'GorkovPotential',
    'ParticleTrajectory',
    'MultiphysicsResult',
    
    # PML
    'PMLHandler',
    'PMLMetrics',
    
    # Thermoviscous
    'ThermoviscousSolver',
    'ThermoviscousCorrection',
    
    # Diagnostics
    'DiagnosticsReport',
    'compute_diagnostics',
    
    # Entry point
    'run_simulation',
]

__version__ = "2.0.0"
