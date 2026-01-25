"""
Canonical configuration for FEniCSx multiphysics simulation.

This module provides the SINGLE authoritative configuration object
controlling all simulation parameters. This is the ONLY configuration
system that should be used.

Requirements (from MASTER BRIEF):
- backend = "fem" (using FEniCSx, no alternatives)
- enable_* flags mapped to PhysicsLevel
- Configuration saved to config.json with every run
- CLI arguments map cleanly onto config fields

Author: Acousto-Tweezers Project
Date: January 2026
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import IntEnum
from pathlib import Path
from typing import Optional, Dict, Any, List


class PhysicsLevel(IntEnum):
    """
    Physics ladder defining simulation complexity.
    
    Each level REQUIRES all prerequisites from lower levels.
    The solver automatically asserts prerequisites are met.
    Running streaming without viscosity, or particles without streaming,
    MUST raise errors.
    
    Level | Name              | Description                           | Prerequisites
    ------|-------------------|---------------------------------------|---------------
    1     | ACOUSTICS_ONLY    | Helmholtz in fluids only              | None
    2     | ACOUSTICS_PML     | + PML boundary conditions             | Level 1
    3     | FLUID_AIR_BATH    | + Air and bath fluid domains          | Level 2
    4     | FLUID_SOLID       | + Elastic solid coupling              | Level 3
    5     | THERMOVISCOUS     | + Viscous/thermal boundary layers     | Level 4
    6     | STREAMING         | + Acoustic streaming (mean flow)      | Level 5
    7     | PARTICLES         | + Radiation force and particle dynamics | Level 6
    """
    ACOUSTICS_ONLY = 1    # Helmholtz in water domain only
    ACOUSTICS_PML = 2     # + PML boundaries
    FLUID_AIR_BATH = 3    # + Air and bath domains
    FLUID_SOLID = 4       # + Elastic solids (plate, walls)
    THERMOVISCOUS = 5     # + Viscous/thermal boundary layer corrections
    STREAMING = 6         # + Acoustic streaming (time-averaged flow)
    PARTICLES = 7         # + Radiation force and particle dynamics
    
    def __str__(self) -> str:
        return self.name
    
    def requires(self) -> List['PhysicsLevel']:
        """Return list of prerequisite physics levels."""
        return [PhysicsLevel(i) for i in range(1, self.value)]
    
    def includes(self, other: 'PhysicsLevel') -> bool:
        """Check if this level includes another level."""
        return self.value >= other.value
    
    def description(self) -> str:
        """Human-readable description of this level."""
        descriptions = {
            1: "First-order acoustics (Helmholtz) in water domain",
            2: "Acoustics with Perfectly Matched Layer boundaries",
            3: "Multi-fluid: water, air, and bath domains",
            4: "Fluid-solid coupling with elastic plate and walls",
            5: "Thermoviscous boundary layer corrections",
            6: "Acoustic streaming (time-averaged flow)",
            7: "Particle radiation force and trajectory integration",
        }
        return descriptions.get(self.value, "Unknown")
    
    @classmethod
    def from_string(cls, name: str) -> 'PhysicsLevel':
        """Create from string name."""
        return cls[name.upper()]


@dataclass
class GeometryConfig:
    """
    Geometry specification for the Petri dish acoustic tweezers setup.
    
    All dimensions in SI units (meters).
    
    Coordinate system:
    - Origin at center of dish bottom (inside surface)
    - z-axis points upward
    - x,y span the horizontal plane
    
    Domain schematic (from MASTER BRIEF):
    
                            ┌─────────────────────────────────┐
                            │           PML_TOP               │
         ┌──────────────────┼─────────────────────────────────┼──────────────────┐
         │                  │             AIR                 │                  │
         │    PML_LEFT      ├────────┬───────────────┬────────┤    PML_RIGHT     │
         │                  │  WALL  │     WATER     │  WALL  │                  │
         │                  │        │   (target)    │        │                  │
         │                  │        └───────────────┘        │                  │
         │                  │              PLATE              │                  │
         │                  ├─────────────────────────────────┤                  │
         │                  │              BATH               │                  │
         │                  │          (transducers)          │                  │
         └──────────────────┼─────────────────────────────────┼──────────────────┘
                            │          PML_BOTTOM             │
                            └─────────────────────────────────┘
    """
    # Dish dimensions
    dish_diameter: float = 35.0e-3       # 35 mm standard Petri dish
    dish_wall_thickness: float = 1.0e-3  # 1 mm walls
    dish_height: float = 10.0e-3         # 10 mm total height
    dish_bottom_thickness: float = 1.0e-3  # 1 mm bottom plate
    
    # Water fill level (inside dish)
    water_depth: float = 2.0e-3          # 2 mm water depth
    
    # External coupling bath
    bath_depth: float = 5.0e-3           # 5 mm bath below dish
    bath_lateral_extent: float = 10.0e-3 # 10 mm beyond dish edges
    
    # Air domain above water
    air_height: float = 8.0e-3           # 8 mm air column
    
    # Lens parameters (explicit acoustic lens)
    lens_enabled: bool = True
    lens_radius: float = 5.0e-3          # 5 mm lens radius
    lens_focal_length: float = 15.0e-3   # 15 mm focal length
    lens_thickness: float = 2.0e-3       # 2 mm lens thickness
    lens_material: str = "polystyrene"   # Lens material
    
    # PML parameters
    pml_thickness: float = 5.0e-3        # 5 mm PML region
    pml_stretch_order: int = 2           # Polynomial order for stretching
    pml_max_sigma: float = 1.0           # Maximum damping (normalized)
    
    # Mesh resolution
    elements_per_wavelength: float = 10.0  # Target mesh density
    min_element_size: float = 100.0e-6     # Minimum element size (100 μm)
    max_element_size: float = 500.0e-6     # Maximum element size (500 μm)
    
    @property
    def dish_inner_radius(self) -> float:
        """Inner radius of the dish."""
        return self.dish_diameter / 2 - self.dish_wall_thickness
    
    @property
    def dish_outer_radius(self) -> float:
        """Outer radius of the dish."""
        return self.dish_diameter / 2
    
    @property
    def total_width(self) -> float:
        """Total domain width including bath and PML."""
        return self.dish_diameter + 2 * (self.bath_lateral_extent + self.pml_thickness)
    
    @property
    def total_height(self) -> float:
        """Total domain height from bath bottom to air top (+ PML)."""
        return (self.pml_thickness + self.bath_depth + self.dish_bottom_thickness + 
                self.water_depth + self.air_height + self.pml_thickness)


@dataclass
class PhysicsConfig:
    """
    Physical parameters for the simulation.
    
    All values in SI units.
    """
    # Operating frequency
    frequency: float = 2.0e6             # 2 MHz
    
    # Temperature (affects material properties)
    temperature: float = 25.0            # 25°C
    
    # Actuation (MECHANICAL ONLY - no shortcuts)
    # Energy path: transducer → lens → bath → plate → dish water
    actuation_type: str = "displacement"  # "displacement" or "traction"
    actuation_amplitude: float = 1.0e-9   # 1 nm displacement amplitude
    actuation_phase_pattern: str = "uniform"  # "uniform", "focused"
    
    # Number of transducers (for array configurations)
    num_transducers: int = 1
    transducer_positions: Optional[List[List[float]]] = None
    transducer_phases: Optional[List[float]] = None
    
    # Particle properties (for radiation force)
    particle_radius: float = 5.0e-6      # 5 μm radius
    particle_density: float = 1050.0     # 1050 kg/m³ (polystyrene)
    particle_compressibility: float = 2.4e-10  # Pa⁻¹
    
    # Number of particles to track
    num_particles: int = 50
    
    # Time integration
    dt: float = 1.0e-5                   # 10 μs timestep
    t_max: float = 1.0                   # 1 s total simulation time
    
    @property
    def omega(self) -> float:
        """Angular frequency."""
        return 2.0 * 3.141592653589793 * self.frequency
    
    @property
    def wavelength_water(self) -> float:
        """Wavelength in water (approximate, for mesh sizing)."""
        c_water = 1480.0  # m/s at 25°C
        return c_water / self.frequency


@dataclass 
class SolverConfig:
    """
    Numerical solver parameters.
    """
    # Linear solver
    ksp_type: str = "preonly"    # PETSc KSP type
    pc_type: str = "lu"          # PETSc preconditioner type
    pc_factor_solver_type: str = "mumps"  # Direct solver
    
    # Tolerances
    rtol: float = 1e-10          # Relative tolerance
    atol: float = 1e-14          # Absolute tolerance
    max_iter: int = 1000         # Maximum iterations
    
    # Mesh quality
    check_mesh_quality: bool = True
    min_mesh_quality: float = 0.1  # Minimum element quality (0-1)
    
    # Output verbosity
    verbose: bool = True


@dataclass
class OutputConfig:
    """
    Output and diagnostics configuration.
    """
    # Output directory
    output_dir: str = "results/fem_multiphysics"
    
    # What to save
    save_mesh: bool = True
    save_fields: bool = True
    save_trajectories: bool = True
    save_animations: bool = True
    
    # Animation parameters
    animation_fps: int = 5
    animation_dpi: int = 100
    
    # Diagnostics
    compute_diagnostics: bool = True
    pml_reflection_target: float = 0.01  # < 1% target


@dataclass
class FEMConfig:
    """
    Complete FEM simulation configuration.
    
    This is the SINGLE authoritative configuration object.
    All modules must use this configuration.
    """
    # Physics level (controls what physics to include)
    physics_level: PhysicsLevel = PhysicsLevel.PARTICLES
    
    # Backend (always "fem" using FEniCSx)
    backend: str = "fem"
    
    # Sub-configurations
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    # Metadata
    name: str = "acousto_tweezers_simulation"
    description: str = "FEniCSx multiphysics acoustic tweezers simulation"
    
    def __post_init__(self):
        """Validate configuration."""
        if self.backend != "fem":
            raise ValueError(f"Only 'fem' backend supported, got: {self.backend}")
    
    @classmethod
    def default(cls) -> 'FEMConfig':
        """Create default configuration."""
        return cls()
    
    @classmethod
    def from_file(cls, path: str) -> 'FEMConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FEMConfig':
        """Create configuration from dictionary."""
        config = cls()
        
        if 'physics_level' in data:
            config.physics_level = PhysicsLevel.from_string(data['physics_level'])
        if 'backend' in data:
            config.backend = data['backend']
        if 'name' in data:
            config.name = data['name']
        if 'description' in data:
            config.description = data['description']
            
        # Load sub-configs
        if 'geometry' in data:
            for key, value in data['geometry'].items():
                if hasattr(config.geometry, key):
                    setattr(config.geometry, key, value)
        if 'physics' in data:
            for key, value in data['physics'].items():
                if hasattr(config.physics, key):
                    setattr(config.physics, key, value)
        if 'solver' in data:
            for key, value in data['solver'].items():
                if hasattr(config.solver, key):
                    setattr(config.solver, key, value)
        if 'output' in data:
            for key, value in data['output'].items():
                if hasattr(config.output, key):
                    setattr(config.output, key, value)
                    
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'physics_level': self.physics_level.name,
            'backend': self.backend,
            'name': self.name,
            'description': self.description,
            'geometry': asdict(self.geometry),
            'physics': {k: v for k, v in asdict(self.physics).items() 
                       if v is not None},
            'solver': asdict(self.solver),
            'output': asdict(self.output),
        }
    
    def save(self, path: str):
        """Save configuration to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def validate(self):
        """
        Validate configuration consistency.
        
        Raises ValueError if configuration is invalid.
        """
        # Physics level validation
        if not isinstance(self.physics_level, PhysicsLevel):
            raise ValueError(f"Invalid physics level: {self.physics_level}")
        
        # Geometry validation
        if self.geometry.dish_diameter <= 0:
            raise ValueError("Dish diameter must be positive")
        if self.geometry.water_depth <= 0:
            raise ValueError("Water depth must be positive")
        if self.geometry.water_depth > self.geometry.dish_height:
            raise ValueError("Water depth cannot exceed dish height")
        
        # Physics validation
        if self.physics.frequency <= 0:
            raise ValueError("Frequency must be positive")
        if self.physics.particle_radius <= 0:
            raise ValueError("Particle radius must be positive")
        
        # Mesh resolution check
        wavelength = self.physics.wavelength_water
        if self.geometry.max_element_size > wavelength / 5:
            import warnings
            warnings.warn(
                f"Maximum element size ({self.geometry.max_element_size*1e6:.1f} μm) "
                f"may be too coarse for wavelength ({wavelength*1e6:.1f} μm). "
                f"Consider reducing to < {wavelength/10*1e6:.1f} μm for 10 PPW."
            )
    
    def log_summary(self) -> str:
        """Generate summary string for logging."""
        lines = [
            "=" * 60,
            "FEM MULTIPHYSICS CONFIGURATION",
            "=" * 60,
            f"Physics Level: {self.physics_level.name} ({self.physics_level.description()})",
            f"Backend: {self.backend} (FEniCSx)",
            "",
            "GEOMETRY:",
            f"  Dish diameter: {self.geometry.dish_diameter*1e3:.1f} mm",
            f"  Water depth: {self.geometry.water_depth*1e3:.1f} mm",
            f"  Bath depth: {self.geometry.bath_depth*1e3:.1f} mm",
            f"  Air height: {self.geometry.air_height*1e3:.1f} mm",
            f"  PML thickness: {self.geometry.pml_thickness*1e3:.1f} mm",
            "",
            "PHYSICS:",
            f"  Frequency: {self.physics.frequency/1e6:.1f} MHz",
            f"  Wavelength (water): {self.physics.wavelength_water*1e6:.1f} μm",
            f"  Actuation: {self.physics.actuation_type}",
            f"  Amplitude: {self.physics.actuation_amplitude*1e9:.2f} nm",
            f"  Particle radius: {self.physics.particle_radius*1e6:.1f} μm",
            "",
            "SOLVER:",
            f"  KSP type: {self.solver.ksp_type}",
            f"  PC type: {self.solver.pc_type}",
            f"  Direct solver: {self.solver.pc_factor_solver_type}",
            "",
            "OUTPUT:",
            f"  Directory: {self.output.output_dir}",
            "=" * 60,
        ]
        return "\n".join(lines)
