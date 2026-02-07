"""
Canonical configuration for FEM multiphysics simulation.

This module provides the SINGLE authoritative configuration object
controlling all simulation parameters. No other configuration systems
should be used.

Requirements (from MASTER BRIEF):
- backend = "fem" (default) or "fd" (deprecated)
- enable_* flags for partial-physics runs
- PhysicsLevel enum for explicit physics ladder
- Configuration saved to config.json with every run
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import IntEnum, auto
from pathlib import Path
from typing import Optional, Dict, Any, List


class PhysicsLevel(IntEnum):
    """
    Physics ladder defining simulation complexity.
    
    Each level requires all prerequisites from lower levels.
    The solver automatically asserts prerequisites are met.
    
    Level | Description                    | Prerequisites
    ------|--------------------------------|---------------
    1     | ACOUSTICS_ONLY                 | None
    2     | SOLID_COUPLING                 | Level 1
    3     | PML                            | Level 2
    4     | THERMOVISCOUS                  | Level 3
    5     | STREAMING                      | Level 4
    6     | RADIATION_FORCE                | Level 5
    7     | PARTICLES                      | Level 6
    """
    ACOUSTICS_ONLY = 1    # Helmholtz in water domain only
    SOLID_COUPLING = 2    # + elastic solids
    PML = 3               # + PML boundary conditions
    THERMOVISCOUS = 4     # + viscous/thermal boundary layers
    STREAMING = 5         # + acoustic streaming (mean flow)
    RADIATION_FORCE = 6   # + Gor'kov potential
    PARTICLES = 7         # + radiation force and particle dynamics
    
    def __str__(self) -> str:
        return self.name
    
    def requires(self) -> List['PhysicsLevel']:
        """Return list of prerequisite physics levels."""
        return [PhysicsLevel(i) for i in range(1, self.value)]
    
    def description(self) -> str:
        """Human-readable description of this level."""
        descriptions = {
            1: "First-order acoustics (Helmholtz) in water domain",
            2: "Acoustics with elastic solid coupling",
            3: "Acoustics with Perfectly Matched Layer boundaries",
            4: "Thermoviscous boundary layer corrections",
            5: "Acoustic streaming (time-averaged flow)",
            6: "Gor'kov radiation force potential",
            7: "Particle radiation force and trajectory integration",
        }
        return descriptions.get(self.value, "Unknown")


@dataclass
class GeometryConfig:
    """
    Geometry specification for the Petri dish acoustic tweezers setup.
    
    All dimensions in SI units (meters).
    
    Coordinate system:
    - Origin at center of dish bottom (inside surface)
    - z-axis points upward
    - x,y span the horizontal plane
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
    
    # PML parameters
    pml_thickness: float = 5.0e-3        # 5 mm PML region
    pml_stretch_order: int = 2           # Polynomial order
    pml_max_sigma: float = 1.0           # Maximum damping (normalized)
    
    # Mesh resolution
    resolution: float = 0.0002           # Default resolution 0.2mm
    elements_per_wavelength: float = 10.0  # Target mesh density
    min_element_size: float = 50.0e-6      # Minimum element size (50 μm)
    max_element_size: float = 500.0e-6     # Maximum element size (500 μm)
    
    @property
    def dish_radius(self) -> float:
        """Inner radius of the dish."""
        return self.dish_diameter / 2
    
    @property
    def total_width(self) -> float:
        """Total domain width including bath and PML."""
        return self.dish_diameter + 2 * (self.bath_lateral_extent + self.pml_thickness)
    
    @property
    def total_height(self) -> float:
        """Total domain height from bath bottom to air top (+ PML)."""
        return (self.bath_depth + self.dish_bottom_thickness + 
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
    
    # Actuation
    actuation_type: str = "displacement"  # "displacement", "velocity", or "traction"
    actuation_amplitude: float = 1.0e-9   # 1 nm displacement amplitude
    actuation_region: str = "bath_bottom"  # Where actuation is applied
    source_amplitude: float = 1.0e5        # Source pressure amplitude in Pa
    
    # Particle properties (for radiation force)
    particle_radius: float = 5.0e-6      # 5 μm radius
    particle_density: float = 1050.0     # 1050 kg/m³ (polystyrene)
    particle_compressibility: float = 2.4e-10  # Pa⁻¹
    
    # Particle simulation
    num_particles: int = 10
    particle_sim_time: float = 0.1       # 100 ms
    particle_dt: float = 1.0e-4          # 100 μs timestep
    
    @property
    def omega(self) -> float:
        """Angular frequency [rad/s]."""
        return 2.0 * 3.141592653589793 * self.frequency
    
    @property
    def period(self) -> float:
        """Acoustic period [s]."""
        return 1.0 / self.frequency


@dataclass
class SolverConfig:
    """
    Numerical solver configuration.
    """
    # Linear solver
    linear_solver: str = "direct"        # "direct" or "iterative"
    iterative_tol: float = 1.0e-8        # Tolerance for iterative solvers
    iterative_maxiter: int = 5000        # Maximum iterations
    
    # Matrix assembly
    use_sparse: bool = True              # Use sparse matrices
    matrix_format: str = "csr"           # "csr", "csc", "coo"
    
    # Nonlinear iterations (for streaming)
    nonlinear_tol: float = 1.0e-6
    nonlinear_maxiter: int = 50
    
    # Parallelization
    n_threads: int = 1                   # Number of threads (1 = serial)


@dataclass
class OutputConfig:
    """
    Output and diagnostic configuration.
    """
    # Output directory
    output_dir: Optional[str] = None     # Auto-generated if None
    
    # What to save
    save_pressure_field: bool = True
    save_velocity_field: bool = True
    save_displacement_field: bool = True
    save_streaming_field: bool = True
    save_gorkov_potential: bool = True
    save_particle_trajectories: bool = True
    
    # Visualization
    create_animations: bool = True
    animation_fps: int = 30
    animation_duration: float = 5.0      # seconds
    
    # Diagnostics
    compute_energy_budget: bool = True
    compute_pml_reflection: bool = True
    run_sanity_checks: bool = True
    
    # Verbosity
    verbose: bool = True
    log_file: Optional[str] = None


@dataclass
class FEMConfig:
    """
    Master configuration object for FEM multiphysics simulation.
    
    This is the SINGLE authoritative configuration. All simulation
    parameters must be specified here.
    
    Example
    -------
    >>> config = FEMConfig(
    ...     physics_level=PhysicsLevel.PARTICLES,
    ...     geometry=GeometryConfig(dish_diameter=35e-3),
    ...     physics=PhysicsConfig(frequency=2e6),
    ... )
    >>> solver = FEMMultiphysicsSolver(config)
    >>> results = solver.solve()
    """
    # Backend selection
    backend: str = "fem"                 # "fem" (default) or "fd" (deprecated)
    
    # Physics level (determines which modules are active)
    physics_level: PhysicsLevel = PhysicsLevel.PARTICLES
    
    # Individual enable flags (for fine-grained control)
    enable_air: bool = True              # Include air domain
    enable_bath: bool = True             # Include external bath
    enable_solids: bool = True           # Include elastic solids
    enable_pml: bool = True              # Include PML boundaries
    enable_thermoviscous: bool = True    # Include boundary layer effects
    enable_streaming: bool = True        # Compute acoustic streaming
    enable_particles: bool = True        # Simulate particles
    
    # Sub-configurations
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    version: str = "1.0.0"
    description: str = ""
    
    def __post_init__(self):
        """Validate configuration and set derived values."""
        self._validate()
        self._set_output_dir()
    
    def _validate(self):
        """Validate configuration consistency."""
        if self.backend not in ("fem", "fd"):
            raise ValueError(f"backend must be 'fem' or 'fd', got '{self.backend}'")
        
        if self.backend == "fd":
            import warnings
            warnings.warn(
                "FD backend is deprecated. Use backend='fem' for production.",
                DeprecationWarning
            )
        
        # Check physics level consistency with enable flags  
        if self.physics_level >= PhysicsLevel.SOLID_COUPLING and not self.enable_solids:
            raise ValueError(
                f"PhysicsLevel {self.physics_level} requires enable_solids=True"
            )
        
        if self.physics_level >= PhysicsLevel.PML and not self.enable_pml:
            raise ValueError(
                f"PhysicsLevel {self.physics_level} requires enable_pml=True"
            )
    
    def _set_output_dir(self):
        """Set output directory if not specified."""
        if self.output.output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output.output_dir = f"results/fem_multiphysics/run_{timestamp}"
    
    def get_active_physics(self) -> List[str]:
        """Return list of active physics modules."""
        active = ["acoustics"]
        
        if self.enable_pml:
            active.append("pml")
        if self.enable_air or self.enable_bath:
            active.append("multi_fluid")
        if self.enable_solids:
            active.append("solids")
        if self.enable_thermoviscous:
            active.append("thermoviscous")
        if self.enable_streaming:
            active.append("streaming")
        if self.enable_particles:
            active.append("particles")
        
        return active
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        d = asdict(self)
        d['physics_level'] = self.physics_level.name
        return d
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'FEMConfig':
        """Create configuration from dictionary."""
        # Handle physics level enum
        if isinstance(d.get('physics_level'), str):
            d['physics_level'] = PhysicsLevel[d['physics_level']]
        
        # Handle nested configs
        if isinstance(d.get('geometry'), dict):
            d['geometry'] = GeometryConfig(**d['geometry'])
        if isinstance(d.get('physics'), dict):
            d['physics'] = PhysicsConfig(**d['physics'])
        if isinstance(d.get('solver'), dict):
            d['solver'] = SolverConfig(**d['solver'])
        if isinstance(d.get('output'), dict):
            d['output'] = OutputConfig(**d['output'])
        
        return cls(**d)
    
    def summary(self) -> str:
        """Return human-readable configuration summary."""
        lines = [
            "=" * 60,
            "FEM Multiphysics Configuration",
            "=" * 60,
            f"Backend:        {self.backend}",
            f"Physics Level:  {self.physics_level.name} ({self.physics_level.value})",
            f"                {self.physics_level.description()}",
            "",
            "Active Physics:",
        ]
        for p in self.get_active_physics():
            lines.append(f"  ✓ {p}")
        
        lines.extend([
            "",
            "Geometry:",
            f"  Dish diameter:    {self.geometry.dish_diameter*1e3:.1f} mm",
            f"  Water depth:      {self.geometry.water_depth*1e3:.1f} mm",
            f"  Air height:       {self.geometry.air_height*1e3:.1f} mm",
            f"  Bath depth:       {self.geometry.bath_depth*1e3:.1f} mm",
            f"  PML thickness:    {self.geometry.pml_thickness*1e3:.1f} mm",
            "",
            "Physics:",
            f"  Frequency:        {self.physics.frequency/1e6:.2f} MHz",
            f"  Temperature:      {self.physics.temperature:.1f} °C",
            f"  Actuation:        {self.physics.actuation_type}",
            f"  Amplitude:        {self.physics.actuation_amplitude*1e9:.2f} nm",
            "",
            "Solver:",
            f"  Linear solver:    {self.solver.linear_solver}",
            f"  Sparse matrices:  {self.solver.use_sparse}",
            "",
            f"Output: {self.output.output_dir}",
            "=" * 60,
        ])
        
        return "\n".join(lines)


def get_default_config() -> FEMConfig:
    """Return default configuration for full physics simulation."""
    return FEMConfig()


# Alias for compatibility
FEMConfig.default = staticmethod(get_default_config)


def save_config(config: FEMConfig, path: Path) -> None:
    """Save configuration to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2, default=str)


def load_config(path: Path) -> FEMConfig:
    """Load configuration from JSON file."""
    with open(path, 'r') as f:
        d = json.load(f)
    return FEMConfig.from_dict(d)
