"""
Unified FEM multiphysics solver.

Orchestrates all physics modules according to the physics ladder:

    PhysicsLevel.ACOUSTICS_ONLY    (1) - Helmholtz in fluid
    PhysicsLevel.SOLID_COUPLING    (2) - Add elastic solid
    PhysicsLevel.PML               (3) - Add PML boundaries
    PhysicsLevel.THERMOVISCOUS     (4) - Add boundary layer losses
    PhysicsLevel.STREAMING         (5) - Add acoustic streaming
    PhysicsLevel.RADIATION_FORCE   (6) - Compute Gor'kov force
    PhysicsLevel.PARTICLES         (7) - Full particle dynamics

The solver automatically enables dependencies when a higher level is requested.

Example usage:
    config = FEMConfig.default()
    config.physics.level = PhysicsLevel.PARTICLES
    
    solver = FEMMultiphysicsSolver(config)
    result = solver.solve()
    
    # Access results
    print(result.acoustic_field.max_pressure)
    print(result.gorkov.trap_depth)
    result.save('results/my_run')
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from pathlib import Path
import numpy as np
import json
from datetime import datetime

from .config import FEMConfig, PhysicsLevel
from .domains import DomainType, InterfaceType
from .materials import MaterialDatabase, FluidMaterial, SolidMaterial, ParticleMaterial
from .geometry import FEMMesh, create_petri_dish_mesh
from .acoustics import AcousticField, FEMAcousticSolver
from .solids import DisplacementField, FEMSolidSolver
from .pml import PMLHandler, PMLMetrics
from .thermoviscous import ThermoviscousSolver, ThermoviscousCorrection
from .streaming import StreamingField, StreamingSolver
from .particles import GorkovPotential, ParticleDynamics, ParticleTrajectory


@dataclass
class MultiphysicsResult:
    """
    Complete result from a multiphysics simulation.
    """
    # Configuration used
    config: FEMConfig
    
    # Mesh
    mesh: FEMMesh
    
    # Materials
    materials: Dict[str, Any] = field(default_factory=dict)
    
    # Fields (populated depending on physics level)
    acoustic_field: Optional[AcousticField] = None
    displacement_field: Optional[DisplacementField] = None
    pml_metrics: Optional[PMLMetrics] = None
    thermoviscous_correction: Optional[ThermoviscousCorrection] = None
    streaming_field: Optional[StreamingField] = None
    gorkov: Optional[GorkovPotential] = None
    trajectories: Optional[List[ParticleTrajectory]] = None
    
    # Diagnostics
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    
    # Timing
    timing: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def save(self, output_dir: str):
        """
        Save results to directory.
        
        Parameters
        ----------
        output_dir : str or Path
            Output directory path.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config_dict = {
            'physics_level': self.config.physics_level.name,
            'frequency_Hz': self.config.physics.frequency,
            'resolution_mm': self.config.geometry.resolution * 1e3,
            'timestamp': self.timestamp,
        }
        with open(output_dir / 'config.json', 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        # Save fields as numpy arrays
        if self.acoustic_field is not None:
            np.savez(
                output_dir / 'acoustic_field.npz',
                p=self.acoustic_field.p,
                x=self.acoustic_field.mesh.x,
                y=self.acoustic_field.mesh.y,
                z=self.acoustic_field.mesh.z,
            )
        
        if self.streaming_field is not None:
            np.savez(
                output_dir / 'streaming_field.npz',
                ux=self.streaming_field.ux,
                uy=self.streaming_field.uy,
                uz=self.streaming_field.uz,
            )
        
        if self.gorkov is not None:
            np.savez(
                output_dir / 'gorkov.npz',
                U=self.gorkov.U,
                Fx=self.gorkov.Fx,
                Fy=self.gorkov.Fy,
                Fz=self.gorkov.Fz,
            )
        
        if self.trajectories is not None:
            for i, traj in enumerate(self.trajectories):
                np.savez(
                    output_dir / f'trajectory_{i:03d}.npz',
                    t=traj.t,
                    x=traj.x,
                    y=traj.y,
                    z=traj.z,
                    x0=traj.x0,
                )
        
        # Save diagnostics
        with open(output_dir / 'diagnostics.json', 'w') as f:
            # Convert numpy types to native Python
            diag_serializable = {}
            for k, v in self.diagnostics.items():
                if isinstance(v, np.ndarray):
                    diag_serializable[k] = v.tolist()
                elif isinstance(v, (np.floating, np.integer)):
                    diag_serializable[k] = float(v)
                else:
                    diag_serializable[k] = v
            json.dump(diag_serializable, f, indent=2)
        
        # Save timing
        with open(output_dir / 'timing.json', 'w') as f:
            json.dump(self.timing, f, indent=2)
        
        print(f"Results saved to {output_dir}")


class FEMMultiphysicsSolver:
    """
    Unified FEM multiphysics solver.
    
    Orchestrates the solution of coupled physics problems for acoustic
    tweezers simulation according to the physics ladder.
    """
    
    def __init__(self, config: FEMConfig):
        """
        Initialize solver.
        
        Parameters
        ----------
        config : FEMConfig
            Complete simulation configuration.
        """
        self.config = config
        self._validate_config()
        
        # Initialize materials
        self.mat_db = MaterialDatabase()
        self._setup_materials()
        
        # Initialize mesh (lazy)
        self._mesh: Optional[FEMMesh] = None
        
        # Sub-solvers (created on demand)
        self._acoustic_solver: Optional[FEMAcousticSolver] = None
        self._solid_solver: Optional[FEMSolidSolver] = None
        self._pml_handler: Optional[PMLHandler] = None
        self._thermoviscous_solver: Optional[ThermoviscousSolver] = None
        self._streaming_solver: Optional[StreamingSolver] = None
        
    def _validate_config(self):
        """Validate configuration consistency."""
        # Check physics level is valid
        level = self.config.physics_level
        assert isinstance(level, PhysicsLevel), f"Invalid physics level: {level}"
        
        # Check resolution is reasonable
        c = 1500  # Approximate sound speed
        wavelength = c / self.config.physics.frequency
        h = self.config.geometry.resolution
        
        if h > wavelength / 6:
            import warnings
            warnings.warn(
                f"Resolution h={h*1e3:.2f}mm > λ/6={wavelength*1e3/6:.2f}mm. "
                "Recommend finer mesh for accuracy."
            )
    
    def _setup_materials(self):
        """Initialize materials from configuration."""
        cfg = self.config.physics
        
        # Fluid (water)
        self.fluid = self.mat_db.get_fluid('water')
        
        # Solid (petri dish)
        self.solid = self.mat_db.get_solid('polystyrene')
        
        # Particle (if needed)
        if self.config.physics_level >= PhysicsLevel.RADIATION_FORCE:
            self.particle = self.mat_db.get_particle('polystyrene_bead')
        else:
            self.particle = None
    
    @property
    def mesh(self) -> FEMMesh:
        """Get or create mesh."""
        if self._mesh is None:
            self._mesh = create_petri_dish_mesh(self.config)
        return self._mesh
    
    @property
    def physics_level(self) -> PhysicsLevel:
        """Current physics level."""
        return self.config.physics_level
    
    def solve(self) -> MultiphysicsResult:
        """
        Solve the multiphysics problem.
        
        Automatically solves all physics up to the configured level.
        
        Returns
        -------
        result : MultiphysicsResult
            Complete simulation result.
        """
        import time
        
        result = MultiphysicsResult(
            config=self.config,
            mesh=self.mesh,
            materials={
                'fluid': self.fluid,
                'solid': self.solid,
                'particle': self.particle,
            },
        )
        
        level = self.physics_level
        
        # Level 1: Acoustics
        if level >= PhysicsLevel.ACOUSTICS_ONLY:
            t0 = time.time()
            result.acoustic_field = self._solve_acoustics()
            result.timing['acoustics'] = time.time() - t0
            print(f"  Acoustics: {result.timing['acoustics']:.2f}s")
        
        # Level 2: Solid coupling
        if level >= PhysicsLevel.SOLID_COUPLING:
            t0 = time.time()
            result.displacement_field = self._solve_solids(result.acoustic_field)
            result.timing['solids'] = time.time() - t0
            print(f"  Solids: {result.timing['solids']:.2f}s")
        
        # Level 3: PML
        if level >= PhysicsLevel.PML:
            t0 = time.time()
            result.pml_metrics = self._compute_pml_metrics(result.acoustic_field)
            result.timing['pml'] = time.time() - t0
            print(f"  PML: {result.timing['pml']:.2f}s")
        
        # Level 4: Thermoviscous
        if level >= PhysicsLevel.THERMOVISCOUS:
            t0 = time.time()
            result.thermoviscous_correction = self._solve_thermoviscous()
            result.timing['thermoviscous'] = time.time() - t0
            print(f"  Thermoviscous: {result.timing['thermoviscous']:.2f}s")
        
        # Level 5: Streaming
        if level >= PhysicsLevel.STREAMING:
            t0 = time.time()
            result.streaming_field = self._solve_streaming(result.acoustic_field)
            result.timing['streaming'] = time.time() - t0
            print(f"  Streaming: {result.timing['streaming']:.2f}s")
        
        # Level 6: Radiation force
        if level >= PhysicsLevel.RADIATION_FORCE:
            t0 = time.time()
            result.gorkov = self._compute_radiation_force(result.acoustic_field)
            result.timing['radiation_force'] = time.time() - t0
            print(f"  Radiation force: {result.timing['radiation_force']:.2f}s")
        
        # Level 7: Particles
        if level >= PhysicsLevel.PARTICLES:
            t0 = time.time()
            result.trajectories = self._integrate_particles(
                result.gorkov, result.streaming_field
            )
            result.timing['particles'] = time.time() - t0
            print(f"  Particles: {result.timing['particles']:.2f}s")
        
        # Diagnostics
        result.diagnostics = self._compute_diagnostics(result)
        
        return result
    
    def _solve_acoustics(self) -> AcousticField:
        """Solve acoustic pressure field."""
        print("Solving acoustics (Helmholtz)...")
        
        if self._acoustic_solver is None:
            self._acoustic_solver = FEMAcousticSolver(
                mesh=self.mesh,
                materials=self.mat_db,
                config=self.config,
            )
        
        # Apply transducer sources
        sources = self._create_sources()
        self._acoustic_solver.set_sources(sources)
        
        return self._acoustic_solver.solve()
    
    def _solve_solids(self, acoustic_field: AcousticField) -> DisplacementField:
        """Solve solid mechanics (dish walls)."""
        print("Solving solid mechanics...")
        
        if self._solid_solver is None:
            self._solid_solver = FEMSolidSolver(
                mesh=self.mesh,
                materials=self.mat_db,
                config=self.config,
            )
        
        # Fluid-solid coupling: solve with acoustic field
        if acoustic_field is not None:
            return self._solid_solver.solve_coupled(acoustic_field)
        else:
            return DisplacementField(u=np.zeros(self.mesh.num_nodes, dtype=np.complex128))
    
    def _compute_pml_metrics(self, acoustic_field: AcousticField) -> PMLMetrics:
        """Compute PML performance metrics."""
        print("Computing PML metrics...")
        
        if self._pml_handler is None:
            from .pml import PMLParameters
            pml_params = PMLParameters()
            self._pml_handler = PMLHandler(
                mesh=self.mesh,
                params=pml_params,
                omega=self.config.physics.omega,
            )
        
        return self._pml_handler.compute_metrics(acoustic_field.p)
    
    def _solve_thermoviscous(self) -> ThermoviscousCorrection:
        """Compute thermoviscous boundary layer correction."""
        print("Computing thermoviscous correction...")
        
        if self._thermoviscous_solver is None:
            self._thermoviscous_solver = ThermoviscousSolver(
                mesh=self.mesh,
                fluid=self.fluid,
                frequency=self.config.physics.frequency,
            )
        
        return self._thermoviscous_solver.compute_correction()
    
    def _solve_streaming(self, acoustic_field: AcousticField) -> StreamingField:
        """Solve acoustic streaming."""
        print("Solving acoustic streaming...")
        
        if self._streaming_solver is None:
            self._streaming_solver = StreamingSolver(
                mesh=self.mesh,
                materials=self.mat_db,
                config=self.config,
            )
        
        return self._streaming_solver.solve(acoustic_field)
    
    def _compute_radiation_force(self, acoustic_field: AcousticField) -> GorkovPotential:
        """Compute Gor'kov radiation force."""
        print("Computing Gor'kov radiation force...")
        
        from .particles import compute_gorkov_potential
        return compute_gorkov_potential(
            acoustic_field=acoustic_field,
            fluid=self.fluid,
            particle=self.particle,
        )
    
    def _integrate_particles(
        self,
        gorkov: GorkovPotential,
        streaming: Optional[StreamingField],
    ) -> List[ParticleTrajectory]:
        """Integrate particle trajectories."""
        print("Integrating particle trajectories...")
        
        from .particles import ParticleDynamics, generate_random_initial_positions
        
        dynamics = ParticleDynamics(
            gorkov=gorkov,
            streaming=streaming,
            fluid=self.fluid,
            particle=self.particle,
        )
        
        # Generate initial positions
        n_particles = self.config.physics.num_particles
        initial_pos = generate_random_initial_positions(
            mesh=self.mesh,
            n_particles=n_particles,
            domain=DomainType.WATER,
            seed=42,
        )
        
        # Integrate
        trajectories = dynamics.integrate_ensemble(
            initial_positions=initial_pos,
            t_final=self.config.physics.particle_sim_time,
            dt=self.config.physics.particle_dt,
            method='rk4',
        )
        
        return trajectories
    
    def _create_sources(self) -> dict:
        """Create acoustic sources from transducer configuration."""
        # Default: single point source at bottom center
        # TODO: Load from config
        
        # Find nodes at bottom of water domain
        z_min = self.mesh.z[0]
        bottom_nodes = np.where(np.abs(self.mesh.nodes[:, 2] - z_min) < self.mesh.dz)[0]
        
        # Source magnitude (Pa)
        p0 = self.config.physics.source_amplitude
        
        sources = {
            'node_ids': bottom_nodes,
            'amplitudes': np.full(len(bottom_nodes), p0, dtype=complex),
        }
        
        return sources
    
    def _compute_diagnostics(self, result: MultiphysicsResult) -> Dict[str, Any]:
        """Compute diagnostic quantities."""
        diagnostics = {}
        
        # Wavelength and mesh quality
        wavelength = self.fluid.c / self.config.physics.frequency
        h = self.config.geometry.resolution
        diagnostics['wavelength_mm'] = wavelength * 1e3
        diagnostics['h_mm'] = h * 1e3
        diagnostics['nodes_per_wavelength'] = wavelength / h
        diagnostics['mesh_adequate'] = (wavelength / h) >= 6
        
        # Acoustic field statistics
        if result.acoustic_field is not None:
            p = result.acoustic_field.p
            diagnostics['p_max_Pa'] = float(np.max(np.abs(p)))
            diagnostics['p_mean_Pa'] = float(np.mean(np.abs(p)))
            diagnostics['p_rms_Pa'] = float(np.sqrt(np.mean(np.abs(p)**2)))
        
        # PML metrics
        if result.pml_metrics is not None:
            diagnostics['pml_reflection'] = result.pml_metrics.reflection_coefficient
            if hasattr(result.pml_metrics, 'meets_target'):
                # Check if it's a property or method
                if callable(result.pml_metrics.meets_target):
                    diagnostics['pml_meets_target'] = result.pml_metrics.meets_target()
                else:
                    diagnostics['pml_meets_target'] = result.pml_metrics.meets_target
        
        # Trap quality
        if result.gorkov is not None:
            diagnostics['trap_depth_J'] = result.gorkov.trap_depth
            diagnostics['trap_depth_kT'] = result.gorkov.trap_depth / (1.38e-23 * 300)
            
            traps = result.gorkov.find_trap_locations()
            diagnostics['n_traps'] = len(traps)
            if traps:
                diagnostics['trap_positions'] = [t.tolist() for t in traps]
        
        # Particle trapping
        if result.trajectories is not None:
            final_positions = [t.final_position for t in result.trajectories]
            diagnostics['n_particles'] = len(result.trajectories)
            
            # Check how many reached trap
            if result.gorkov is not None:
                traps = result.gorkov.find_trap_locations()
                if traps:
                    trap_center = traps[0]
                    distances = [np.linalg.norm(p - trap_center) for p in final_positions]
                    trapped = sum(1 for d in distances if d < wavelength/4)
                    diagnostics['n_trapped'] = trapped
                    diagnostics['trapping_efficiency'] = trapped / len(result.trajectories)
        
        return diagnostics


def run_simulation(
    config: Optional[FEMConfig] = None,
    output_dir: Optional[str] = None,
) -> MultiphysicsResult:
    """
    Run a complete multiphysics simulation.
    
    Parameters
    ----------
    config : FEMConfig, optional
        Configuration. If None, uses default.
    output_dir : str, optional
        Directory to save results.
    
    Returns
    -------
    result : MultiphysicsResult
        Simulation result.
    """
    if config is None:
        config = FEMConfig.default()
    
    print("=" * 60)
    print(f"FEM Multiphysics Solver")
    print(f"Physics level: {config.physics_level.name}")
    print(f"Frequency: {config.physics.frequency/1e6:.1f} MHz")
    print(f"Resolution: {config.geometry.resolution*1e3:.2f} mm")
    print("=" * 60)
    
    solver = FEMMultiphysicsSolver(config)
    result = solver.solve()
    
    # Print diagnostics
    print("\nDiagnostics:")
    for key, value in result.diagnostics.items():
        if isinstance(value, bool):
            print(f"  {key}: {'✓' if value else '✗'}")
        elif isinstance(value, float):
            print(f"  {key}: {value:.4g}")
        else:
            print(f"  {key}: {value}")
    
    # Save if requested
    if output_dir is not None:
        result.save(output_dir)
    
    return result
