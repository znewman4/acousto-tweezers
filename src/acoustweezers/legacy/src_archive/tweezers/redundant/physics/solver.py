"""
Unified multiphysics solver for 3D acoustic trapping.

Orchestrates:
- Multi-domain Helmholtz acoustics with PML
- Elastic solid mechanics
- Fluid-solid coupling
- Thermoviscous boundary layers
- Acoustic streaming
- Particle radiation force and dynamics
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any, List
from pathlib import Path
import numpy as np
import time

from .acoustics.geometry import MultiDomainGeometry, DomainType
from .acoustics.materials import FluidMaterial, MaterialDatabase
from .acoustics.pml import PMLManager, PMLParameters
from .acoustics.thermoviscous import ThermoviscousCorrector
from .acoustics.solver import MultiDomainAcousticSolver, AcousticField3D

from .solid.materials import SolidMaterial, SolidMaterialDatabase
from .solid.solver import ElasticSolver, DisplacementField3D
from .solid.coupling import FluidSolidCoupling, PlateTransmission

from .streaming.solver import StreamingSolver, StreamingField
from .streaming.forcing import StreamingForcing

from .particle.properties import Particle3D, ParticleDatabase
from .particle.interpolation import Grid3D
from .particle.gorkov import GorkovPotential3D
from .particle.dynamics import ParticleDynamics3D, ParticleTrajectory


@dataclass
class SimulationParameters:
    """
    Complete simulation parameters.
    
    Attributes
    ----------
    frequency : float
        Acoustic frequency [Hz].
    actuation_amplitude : float
        Transducer velocity amplitude [m/s].
    dish_radius : float
        Dish radius [m].
    water_depth : float
        Water depth in dish [m].
    air_height : float
        Air height above water [m].
    plate_thickness : float
        Dish plate thickness [m].
    wall_thickness : float
        Side wall thickness [m].
    grid_resolution : float
        Grid resolution [m].
    pml_thickness : int
        PML thickness in grid points.
    temperature : float
        Temperature [°C].
    """
    frequency: float = 2.0e6  # 2 MHz
    actuation_amplitude: float = 1e-6  # 1 μm/s
    dish_radius: float = 17.5e-3  # 35 mm diameter dish
    water_depth: float = 2.0e-3  # 2 mm water
    air_height: float = 5.0e-3  # 5 mm air
    plate_thickness: float = 1.0e-3  # 1 mm plate
    wall_thickness: float = 1.5e-3  # 1.5 mm wall
    grid_resolution: float = 50e-6  # 50 μm
    pml_thickness: int = 10
    temperature: float = 25.0


@dataclass
class MultiphysicsResults:
    """
    Complete multiphysics simulation results.
    
    Attributes
    ----------
    parameters : SimulationParameters
        Input parameters.
    geometry : MultiDomainGeometry
        Domain geometry.
    acoustic_field : AcousticField3D
        Acoustic pressure and velocity.
    displacement_field : DisplacementField3D, optional
        Solid displacement.
    streaming_field : StreamingField, optional
        Streaming velocity.
    gorkov_potential : np.ndarray, optional
        Gor'kov potential for default particle.
    particle_trajectories : list of ParticleTrajectory, optional
        Simulated particle paths.
    computation_times : dict
        Timing for each solver stage.
    energy_budget : dict
        Energy analysis (power in, dissipated, radiated).
    """
    parameters: SimulationParameters
    geometry: MultiDomainGeometry
    acoustic_field: AcousticField3D
    displacement_field: Optional[DisplacementField3D] = None
    streaming_field: Optional[StreamingField] = None
    gorkov_potential: Optional[np.ndarray] = None
    particle_trajectories: Optional[List[ParticleTrajectory]] = None
    computation_times: Dict[str, float] = field(default_factory=dict)
    energy_budget: Dict[str, float] = field(default_factory=dict)
    
    def save(self, path: Path) -> None:
        """Save results to NPZ file."""
        data = {
            'pressure_real': self.acoustic_field.p.real,
            'pressure_imag': self.acoustic_field.p.imag,
            'grid_x': self.geometry.grid_x,
            'grid_y': self.geometry.grid_y,
            'grid_z': self.geometry.grid_z,
            'frequency': self.parameters.frequency,
        }
        
        if self.gorkov_potential is not None:
            data['gorkov_potential'] = self.gorkov_potential
        
        if self.streaming_field is not None:
            data['streaming_vx'] = self.streaming_field.vx
            data['streaming_vy'] = self.streaming_field.vy
            data['streaming_vz'] = self.streaming_field.vz
        
        np.savez_compressed(path, **data)
    
    @classmethod
    def load(cls, path: Path) -> "MultiphysicsResults":
        """Load results from NPZ file."""
        data = np.load(path)
        # Minimal reconstruction - full reconstruction requires geometry
        raise NotImplementedError("Full reconstruction requires geometry recreation")


class MultiphysicsSolver:
    """
    Unified multiphysics acoustic trapping solver.
    
    Coordinates all physics modules in correct sequence:
    1. Acoustics: Multi-domain Helmholtz with PML
    2. Solid: Elastic wave propagation in dish
    3. Coupling: Interface conditions at fluid-solid boundaries
    4. Streaming: Time-averaged flow from acoustic forcing
    5. Particle: Radiation force and trajectory integration
    """
    
    def __init__(
        self,
        params: Optional[SimulationParameters] = None,
        verbose: bool = True,
    ):
        """
        Initialize solver.
        
        Parameters
        ----------
        params : SimulationParameters, optional
            Simulation parameters. Default: standard dish setup.
        verbose : bool
            Print progress messages.
        """
        self.params = params or SimulationParameters()
        self.verbose = verbose
        
        # Will be initialized on solve()
        self.geometry: Optional[MultiDomainGeometry] = None
        self.materials: Dict[str, Any] = {}
        self._grid: Optional[Grid3D] = None
        self.acoustic_solver: Optional[MultiDomainAcousticSolver] = None
        
    def _log(self, msg: str) -> None:
        """Print progress message if verbose."""
        if self.verbose:
            print(f"[MultiphysicsSolver] {msg}")
    
    def _setup_geometry(self) -> None:
        """Create multi-domain geometry."""
        from .acoustics.geometry import create_standard_dish_geometry
        
        self._log("Setting up geometry...")
        
        self.geometry = create_standard_dish_geometry(
            dish_diameter_mm=self.params.dish_radius * 2.0 * 1e3,  # radius -> diameter in mm
            dish_height_mm=self.params.water_depth * 1e3,           # m -> mm
            air_height_mm=self.params.air_height * 1e3,             # m -> mm
            plate_thickness_mm=self.params.plate_thickness * 1e3,   # m -> mm
            wall_thickness_mm=self.params.wall_thickness * 1e3,     # m -> mm
            resolution_mm=self.params.grid_resolution * 1e3,        # m -> mm
        )
        
        # Create Grid3D for particle module
        self._grid = Grid3D(
            x=self.geometry.x,
            y=self.geometry.y,
            z=self.geometry.z,
        )
        
        self._log(f"  Grid shape: {self.geometry.shape}")
        self._log(f"  Domains: {list(self.geometry.regions.keys())}")
    
    def _setup_materials(self) -> None:
        """Initialize material properties."""
        self._log("Setting up materials...")
        
        T = self.params.temperature
        
        self.materials = {
            'water': MaterialDatabase.water(T),
            'air': MaterialDatabase.air(T),
            'glass': SolidMaterialDatabase.glass_borosilicate(),
            'polystyrene': SolidMaterialDatabase.polystyrene(),
        }
        
        for name, mat in self.materials.items():
            if hasattr(mat, 'c'):
                self._log(f"  {name}: ρ={mat.rho:.1f} kg/m³, c={mat.c:.1f} m/s")
            elif hasattr(mat, 'c_L'):
                self._log(f"  {name}: ρ={mat.rho:.1f} kg/m³, c_L={mat.c_L:.1f} m/s")
    
    def _solve_acoustics(self) -> AcousticField3D:
        """Solve multi-domain Helmholtz equation."""
    def _solve_acoustics(self) -> AcousticField3D:
        """Solve multi-domain Helmholtz equation."""
        self._log("Solving acoustics...")
        
        t_start = time.time()
        
        # Setup PML
        pml_params = PMLParameters(
            thickness=self.params.pml_thickness,
            R0=1e-6,
        )
        
        # Map materials to domains
        domain_materials = {
            DomainType.WATER_DISH: self.materials['water'],
            DomainType.WATER_BATH: self.materials['water'],
            DomainType.AIR: self.materials['air'],
        }
        
        # Create solver
        solver = MultiDomainAcousticSolver(
            geometry=self.geometry,
            materials=domain_materials,
            pml_params=pml_params,
        )
        self.acoustic_solver = solver  # Store for later use
        
        # Frequency
        omega = 2.0 * np.pi * self.params.frequency
        
        # Create bottom velocity field (actuator)
        nx, ny, nz = self.geometry.shape
        bottom_velocity = np.zeros((nx, ny), dtype=complex)
        
        # Gaussian actuator pattern at dish center
        X, Y = np.meshgrid(self.geometry.x, self.geometry.y, indexing='ij')
        cx, cy = self.geometry.Lx / 2, self.geometry.Ly / 2
        actuator_radius = self.params.dish_radius * 0.8
        r2 = (X - cx)**2 + (Y - cy)**2
        actuator_mask = r2 < actuator_radius**2
        bottom_velocity[actuator_mask] = self.params.actuation_amplitude
        
        # Solve (omega first, then v_bottom)
        field = solver.solve_with_bottom_velocity(omega, bottom_velocity)
        
        t_elapsed = time.time() - t_start
        self._log(f"  Acoustics solved in {t_elapsed:.2f} s")
        self._log(f"  Max |p| = {np.max(np.abs(field.p)):.2e} Pa")
        
        return field
    
    def _solve_streaming(
        self,
        acoustic_field: AcousticField3D,
    ) -> StreamingField:
        """Compute acoustic streaming velocity."""
        self._log("Solving streaming...")
        
        t_start = time.time()
        
        water = self.materials['water']
        omega = 2.0 * np.pi * self.params.frequency
        
        streaming_solver = StreamingSolver(
            x=acoustic_field.x,
            y=acoustic_field.y,
            z=acoustic_field.z,
            fluid=water,
        )
        
        streaming = streaming_solver.compute_streaming(
            p=acoustic_field.p,
            rho=acoustic_field.rho,
            omega=omega,
        )
        
        t_elapsed = time.time() - t_start
        self._log(f"  Streaming solved in {t_elapsed:.2f} s")
        v_max = np.sqrt(streaming.ux**2 + streaming.uy**2 + streaming.uz**2).max()
        self._log(f"  Max |u_stream| = {v_max:.2e} m/s")
        
        return streaming
    
    def _compute_gorkov(
        self,
        acoustic_field: AcousticField3D,
        particle: Optional[Particle3D] = None,
    ) -> Tuple[np.ndarray, GorkovPotential3D]:
        """Compute Gor'kov potential for particle."""
        self._log("Computing Gor'kov potential...")
        
        t_start = time.time()
        
        if particle is None:
            particle = ParticleDatabase.polystyrene_bead(5.0)
        
        water = self.materials['water']
        omega = 2.0 * np.pi * self.params.frequency
        
        gorkov = GorkovPotential3D(
            grid=self._grid,
            pressure=acoustic_field.p,
            fluid=water,
            omega=omega,
        )
        
        U = gorkov.compute_potential(particle)
        
        t_elapsed = time.time() - t_start
        self._log(f"  Gor'kov computed in {t_elapsed:.2f} s")
        self._log(f"  Potential range: [{U.min():.2e}, {U.max():.2e}] J")
        
        return U, gorkov
    
    def _simulate_particles(
        self,
        gorkov: GorkovPotential3D,
        streaming: Optional[StreamingField],
        particle: Particle3D,
        initial_positions: np.ndarray,
        duration: float = 0.1,
    ) -> List[ParticleTrajectory]:
        """Simulate particle trajectories."""
        self._log("Simulating particle dynamics...")
        
        t_start = time.time()
        
        water = self.materials['water']
        
        streaming_vel = None
        if streaming is not None:
            streaming_vel = (streaming.ux, streaming.uy, streaming.uz)
        
        dynamics = ParticleDynamics3D(
            grid=self._grid,
            gorkov=gorkov,
            particle=particle,
            fluid=water,
            streaming_velocity=streaming_vel,
        )
        
        from .particle.dynamics import simulate_multiple_particles, make_boundary_event
        
        boundary_event = make_boundary_event(self._grid, margin=self.params.grid_resolution)
        
        trajectories = simulate_multiple_particles(
            dynamics,
            initial_positions,
            duration=duration,
            dt=1e-4,
            events=[boundary_event],
        )
        
        t_elapsed = time.time() - t_start
        self._log(f"  Dynamics computed in {t_elapsed:.2f} s")
        self._log(f"  Simulated {len(trajectories)} particles for {duration*1e3:.1f} ms")
        
        return trajectories
    
    def solve(
        self,
        solve_streaming: bool = True,
        compute_gorkov: bool = True,
        simulate_particles: bool = False,
        particle: Optional[Particle3D] = None,
        initial_positions: Optional[np.ndarray] = None,
        particle_duration: float = 0.1,
    ) -> MultiphysicsResults:
        """
        Run complete multiphysics simulation.
        
        Parameters
        ----------
        solve_streaming : bool
            Compute acoustic streaming.
        compute_gorkov : bool
            Compute Gor'kov potential.
        simulate_particles : bool
            Run particle dynamics.
        particle : Particle3D, optional
            Particle for Gor'kov/dynamics. Default: 5 μm polystyrene.
        initial_positions : np.ndarray, optional
            Initial particle positions for dynamics.
        particle_duration : float
            Simulation duration for particles [s].
        
        Returns
        -------
        results : MultiphysicsResults
            Complete simulation results.
        """
        self._log("=" * 60)
        self._log("Starting multiphysics simulation")
        self._log("=" * 60)
        
        computation_times = {}
        
        # Setup
        t0 = time.time()
        self._setup_geometry()
        self._setup_materials()
        computation_times['setup'] = time.time() - t0
        
        # Acoustics
        t0 = time.time()
        acoustic_field = self._solve_acoustics()
        computation_times['acoustics'] = time.time() - t0
        
        # Streaming (optional)
        streaming_field = None
        if solve_streaming:
            t0 = time.time()
            streaming_field = self._solve_streaming(acoustic_field)
            computation_times['streaming'] = time.time() - t0
        
        # Gor'kov (optional)
        gorkov_potential = None
        gorkov = None
        if compute_gorkov:
            if particle is None:
                particle = ParticleDatabase.polystyrene_bead(5.0)
            t0 = time.time()
            gorkov_potential, gorkov = self._compute_gorkov(acoustic_field, particle)
            computation_times['gorkov'] = time.time() - t0
        
        # Particle dynamics (optional)
        trajectories = None
        if simulate_particles and gorkov is not None:
            if initial_positions is None:
                # Default: random positions in water domain
                n_particles = 10
                initial_positions = self._generate_random_positions(n_particles)
            
            t0 = time.time()
            trajectories = self._simulate_particles(
                gorkov, streaming_field, particle,
                initial_positions, particle_duration
            )
            computation_times['dynamics'] = time.time() - t0
        
        # Energy budget analysis
        energy_budget = self.acoustic_solver.energy_budget(acoustic_field)
        
        total_time = sum(computation_times.values())
        self._log("=" * 60)
        self._log(f"Simulation complete. Total time: {total_time:.2f} s")
        for stage, t in computation_times.items():
            self._log(f"  {stage}: {t:.2f} s ({100*t/total_time:.1f}%)")
        self._log("=" * 60)
        
        return MultiphysicsResults(
            parameters=self.params,
            geometry=self.geometry,
            acoustic_field=acoustic_field,
            streaming_field=streaming_field,
            gorkov_potential=gorkov_potential,
            particle_trajectories=trajectories,
            computation_times=computation_times,
            energy_budget=energy_budget,
        )
    
    def _generate_random_positions(self, n: int) -> np.ndarray:
        """Generate random initial positions in water region."""
        positions = []
        
        # Water domain bounds (approximate)
        r_max = self.params.dish_radius * 0.8
        z_min = self.params.plate_thickness + self.params.grid_resolution
        z_max = self.params.plate_thickness + self.params.water_depth * 0.9
        
        for _ in range(n):
            # Random position in cylinder
            r = r_max * np.sqrt(np.random.random())
            theta = 2.0 * np.pi * np.random.random()
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            z = z_min + (z_max - z_min) * np.random.random()
            positions.append([x, y, z])
        
        return np.array(positions)


def run_standard_simulation(
    output_dir: Optional[Path] = None,
    **kwargs,
) -> MultiphysicsResults:
    """
    Run standard simulation with default parameters.
    
    Parameters
    ----------
    output_dir : Path, optional
        Directory for output files.
    **kwargs
        Override default SimulationParameters.
    
    Returns
    -------
    results : MultiphysicsResults
        Simulation results.
    """
    params = SimulationParameters(**kwargs)
    solver = MultiphysicsSolver(params)
    
    results = solver.solve(
        solve_streaming=True,
        compute_gorkov=True,
        simulate_particles=True,
    )
    
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        results.save(output_dir / "results.npz")
    
    return results
