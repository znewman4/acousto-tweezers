"""
Unified FEniCSx multiphysics solver for acoustic tweezers.

Orchestrates all physics modules according to the physics ladder:

    PhysicsLevel.ACOUSTICS_ONLY    (1) - Helmholtz in fluid
    PhysicsLevel.ACOUSTICS_PML     (2) - + PML boundaries
    PhysicsLevel.FLUID_AIR_BATH    (3) - + Air and bath domains
    PhysicsLevel.FLUID_SOLID       (4) - + Elastic solids
    PhysicsLevel.THERMOVISCOUS     (5) - + Boundary layer losses
    PhysicsLevel.STREAMING         (6) - + Acoustic streaming
    PhysicsLevel.PARTICLES         (7) - + Radiation force & particles

Each level INCLUDES all physics from levels below.
Running streaming without viscosity, or particles without streaming,
will raise errors.

IMPORTANT: This solver uses FEniCSx (dolfinx + PETSc) exclusively.
No homebrew FEM, no Python element loops, no manual sparse assembly.

Author: Acousto-Tweezers Project  
Date: January 2026
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from pathlib import Path
import numpy as np
import json
from datetime import datetime
import time

from mpi4py import MPI

from .config import FEMConfig, PhysicsLevel
from .domains import Domain, Interface
from .materials import MaterialDatabase
from .geometry import create_petri_dish_geometry, MeshInfo
from .acoustics import AcousticSolver, AcousticField
from .solids import SolidSolver, DisplacementField
from .coupling import CoupledSolver, CoupledField
from .pml import PMLHandler, PMLMetrics
from .thermoviscous import ThermoviscousSolver, ThermoviscousCorrection
from .streaming import StreamingSolver, StreamingField
from .particles import ParticleDynamics, GorkovPotential, ParticleTrajectory


@dataclass
class MultiphysicsResult:
    """
    Complete result from a multiphysics simulation.
    """
    # Configuration used
    config: FEMConfig
    
    # Mesh info
    mesh_info: MeshInfo
    
    # Fields (populated depending on physics level)
    acoustic_field: Optional[AcousticField] = None
    displacement_field: Optional[DisplacementField] = None
    coupled_field: Optional[CoupledField] = None
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
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    
    def save(self, output_dir: str):
        """
        Save results to directory.
        
        Parameters
        ----------
        output_dir : str or Path
            Output directory path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config_path = output_dir / "config.json"
        self.config.save(str(config_path))
        
        # Save diagnostics
        diag_dir = output_dir / "diagnostics"
        diag_dir.mkdir(exist_ok=True)
        
        # Sanity report
        with open(diag_dir / "sanity_report.txt", 'w') as f:
            f.write(self._generate_sanity_report())
        
        # Summary CSV
        with open(output_dir / "summary.csv", 'w') as f:
            f.write("metric,value,unit\n")
            for key, value in self.diagnostics.items():
                if isinstance(value, (int, float)):
                    f.write(f"{key},{value},\n")
        
        # Save timing
        with open(output_dir / "timing.json", 'w') as f:
            json.dump(self.timing, f, indent=2)
        
        # Save trajectories if available
        if self.trajectories is not None:
            from .particles import save_trajectories_csv
            save_trajectories_csv(self.trajectories, str(output_dir / "traj.csv"))
        
        # Save fields as numpy arrays
        if self.acoustic_field is not None:
            np.savez(
                output_dir / "acoustic_field.npz",
                p=self.acoustic_field.p,
                coords=self.acoustic_field.coords,
            )
        
        if self.gorkov is not None:
            np.savez(
                output_dir / "gorkov.npz",
                U=self.gorkov.U,
            )
        
        print(f"Results saved to {output_dir}")
    
    def _generate_sanity_report(self) -> str:
        """Generate physics sanity check report."""
        lines = [
            "=" * 60,
            "PHYSICS SANITY REPORT",
            f"Timestamp: {self.timestamp}",
            f"Physics Level: {self.config.physics_level.name}",
            "=" * 60,
            "",
        ]
        
        # Mesh quality
        lines.append("MESH QUALITY:")
        if 'wavelength' in self.diagnostics:
            wavelength = self.diagnostics['wavelength']
            h = self.diagnostics.get('mesh_size', 0)
            ppw = wavelength / h if h > 0 else 0
            status = "PASS" if ppw >= 10 else "WARN" if ppw >= 5 else "FAIL"
            lines.append(f"  [{status}] Points per wavelength: {ppw:.1f} (target: ≥10)")
        
        # Acoustic field
        if self.acoustic_field is not None:
            lines.append("")
            lines.append("ACOUSTIC FIELD:")
            p_max = self.acoustic_field.max_pressure
            p_mean = self.acoustic_field.mean_pressure
            
            status = "PASS" if p_max > 0 else "FAIL"
            lines.append(f"  [{status}] max|p|: {p_max:.2e} Pa")
            lines.append(f"  [INFO] mean|p|: {p_mean:.2e} Pa")
        
        # PML
        if self.pml_metrics is not None:
            lines.append("")
            lines.append("PML BOUNDARY:")
            R = self.pml_metrics.reflection_coefficient
            status = "PASS" if R < 0.01 else "WARN" if R < 0.05 else "FAIL"
            lines.append(f"  [{status}] Reflection: {R*100:.2f}% (target: <1%)")
        
        # Streaming
        if self.streaming_field is not None:
            lines.append("")
            lines.append("ACOUSTIC STREAMING:")
            u_max = self.streaming_field.max_velocity
            lines.append(f"  [INFO] max|u_stream|: {u_max:.2e} m/s")
            
            Re_s = self.diagnostics.get('streaming_reynolds', 0)
            status = "PASS" if Re_s < 1 else "WARN"
            lines.append(f"  [{status}] Re_streaming: {Re_s:.2f} (expect: <1)")
        
        # Particles
        if self.trajectories is not None:
            lines.append("")
            lines.append("PARTICLE DYNAMICS:")
            n_trapped = sum(1 for t in self.trajectories if t.is_trapped())
            lines.append(f"  [INFO] Particles tracked: {len(self.trajectories)}")
            lines.append(f"  [INFO] Particles trapped: {n_trapped}")
            
            mean_disp = np.mean([t.displacement for t in self.trajectories])
            lines.append(f"  [INFO] Mean displacement: {mean_disp*1e6:.2f} μm")
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)


class FEMMultiphysicsSolver:
    """
    Unified FEniCSx multiphysics solver.
    
    Orchestrates the solution of coupled physics problems for acoustic
    tweezers simulation according to the physics ladder.
    """
    
    def __init__(self, config: FEMConfig):
        """
        Initialize solver.
        
        Parameters
        ----------
        config : FEMConfig
            Complete simulation configuration
        """
        self.config = config
        self._validate_config()
        
        # Print configuration
        print(config.log_summary())
        
        # Initialize materials
        self.materials = MaterialDatabase(config.physics.temperature)
        print(self.materials.summary())
        
        # Mesh and tags (created on demand)
        self._mesh = None
        self._cell_tags = None
        self._facet_tags = None
        self._mesh_info = None
        
        # Sub-solvers (created on demand)
        self._acoustic_solver = None
        self._solid_solver = None
        self._coupled_solver = None
        self._pml_handler = None
        self._thermoviscous_solver = None
        self._streaming_solver = None
        self._particle_dynamics = None
        
    def _validate_config(self):
        """Validate configuration consistency."""
        level = self.config.physics_level
        
        if not isinstance(level, PhysicsLevel):
            raise ValueError(f"Invalid physics level: {level}")
        
        # Validate physics ladder requirements
        if level >= PhysicsLevel.STREAMING and level < PhysicsLevel.THERMOVISCOUS:
            raise ValueError(
                "Streaming (level 6) requires thermoviscous (level 5). "
                "Physics ladder violation: cannot skip prerequisites."
            )
        
        if level >= PhysicsLevel.PARTICLES and level < PhysicsLevel.STREAMING:
            raise ValueError(
                "Particles (level 7) requires streaming (level 6). "
                "Physics ladder violation: cannot skip prerequisites."
            )
        
        self.config.validate()
        
    def _create_mesh(self, output_dir: Optional[str] = None):
        """Create or load mesh."""
        if self._mesh is not None:
            return
        
        print("\nCreating mesh...")
        mesh_dir = output_dir if output_dir else None
        
        self._mesh, self._cell_tags, self._facet_tags, self._mesh_info = \
            create_petri_dish_geometry(self.config, mesh_dir, verbose=True)
        
    def solve(self, output_dir: Optional[str] = None) -> MultiphysicsResult:
        """
        Solve the multiphysics problem.
        
        Parameters
        ----------
        output_dir : str, optional
            Directory to save intermediate results
            
        Returns
        -------
        MultiphysicsResult
            Complete simulation results
        """
        timing = {}
        diagnostics = {}
        
        level = self.config.physics_level
        
        # Create mesh
        t0 = time.time()
        self._create_mesh(output_dir)
        timing['mesh_generation'] = time.time() - t0
        
        # Compute derived quantities for diagnostics
        omega = self.config.physics.omega
        c_water = self.materials.water.sound_speed
        wavelength = 2 * np.pi * c_water / omega
        diagnostics['wavelength'] = wavelength
        diagnostics['frequency'] = self.config.physics.frequency
        diagnostics['mesh_size'] = self._mesh_info.min_element_size
        diagnostics['ppw'] = wavelength / self._mesh_info.min_element_size
        
        # Initialize result
        result = MultiphysicsResult(
            config=self.config,
            mesh_info=self._mesh_info,
        )
        
        # =====================================================
        # LEVEL 1-3: ACOUSTICS (with PML and multi-fluid)
        # =====================================================
        if level >= PhysicsLevel.ACOUSTICS_ONLY:
            print("\nSolving acoustics...")
            t0 = time.time()
            
            if level >= PhysicsLevel.FLUID_SOLID:
                # Use coupled solver
                result = self._solve_coupled(result, diagnostics)
            else:
                # Fluid-only acoustic solve
                result = self._solve_acoustics(result, diagnostics)
            
            timing['acoustics'] = time.time() - t0
        
        # =====================================================
        # LEVEL 5: THERMOVISCOUS
        # =====================================================
        if level >= PhysicsLevel.THERMOVISCOUS:
            print("\nComputing thermoviscous corrections...")
            t0 = time.time()
            
            result = self._solve_thermoviscous(result, diagnostics)
            
            timing['thermoviscous'] = time.time() - t0
        
        # =====================================================
        # LEVEL 6: STREAMING
        # =====================================================
        if level >= PhysicsLevel.STREAMING:
            print("\nSolving acoustic streaming...")
            t0 = time.time()
            
            result = self._solve_streaming(result, diagnostics)
            
            timing['streaming'] = time.time() - t0
        
        # =====================================================
        # LEVEL 7: PARTICLES
        # =====================================================
        if level >= PhysicsLevel.PARTICLES:
            print("\nComputing particle dynamics...")
            t0 = time.time()
            
            result = self._solve_particles(result, diagnostics)
            
            timing['particles'] = time.time() - t0
        
        result.timing = timing
        result.diagnostics = diagnostics
        
        # Save if output directory specified
        if output_dir:
            result.save(output_dir)
        
        return result
    
    def _solve_acoustics(self, result: MultiphysicsResult, 
                         diagnostics: Dict) -> MultiphysicsResult:
        """Solve acoustic problem (fluid only)."""
        if self._acoustic_solver is None:
            self._acoustic_solver = AcousticSolver(
                self.config, self._mesh, self._cell_tags,
                self._facet_tags, self.materials
            )
        
        # Solve with actuation
        acoustic_field = self._acoustic_solver.solve_with_actuation(
            self.config.physics.actuation_amplitude,
            actuation_phase=0.0
        )
        
        result.acoustic_field = acoustic_field
        
        # Diagnostics
        diagnostics['p_max'] = acoustic_field.max_pressure
        diagnostics['p_mean'] = acoustic_field.mean_pressure
        diagnostics['p_rms'] = acoustic_field.rms_pressure
        
        # PML evaluation
        if self.config.physics_level >= PhysicsLevel.ACOUSTICS_PML:
            if self._pml_handler is None:
                self._pml_handler = PMLHandler(self.config, self._mesh)
            
            pml_metrics = self._pml_handler.evaluate_reflection(
                acoustic_field.p_function,
                self.config.physics.source_amplitude
            )
            result.pml_metrics = pml_metrics
            diagnostics['pml_reflection'] = pml_metrics.reflection_coefficient
        
        return result
    
    def _solve_coupled(self, result: MultiphysicsResult,
                       diagnostics: Dict) -> MultiphysicsResult:
        """Solve coupled fluid-solid problem."""
        if self._coupled_solver is None:
            self._coupled_solver = CoupledSolver(
                self.config, self._mesh, self._cell_tags,
                self._facet_tags, self.materials
            )
        
        coupled_field = self._coupled_solver.solve(
            self.config.physics.actuation_amplitude
        )
        
        result.coupled_field = coupled_field
        result.acoustic_field = coupled_field.acoustic_field
        result.displacement_field = coupled_field.displacement_field
        
        # Diagnostics
        diagnostics['p_max'] = coupled_field.acoustic_field.max_pressure
        diagnostics['u_max'] = coupled_field.displacement_field.max_displacement()
        
        if coupled_field.interface_residuals:
            for key, val in coupled_field.interface_residuals.items():
                diagnostics[f'interface_{key}'] = val
        
        return result
    
    def _solve_thermoviscous(self, result: MultiphysicsResult,
                             diagnostics: Dict) -> MultiphysicsResult:
        """Compute thermoviscous corrections."""
        if self._thermoviscous_solver is None:
            self._thermoviscous_solver = ThermoviscousSolver(
                self.config, self.materials
            )
        
        correction = self._thermoviscous_solver.compute_correction(
            self._mesh, self._facet_tags, 
            [Interface.WATER_PLATE.gmsh_tag, Interface.WATER_WALL.gmsh_tag]
        )
        
        result.thermoviscous_correction = correction
        
        diagnostics['delta_v'] = correction.delta_v
        diagnostics['delta_t'] = correction.delta_t
        diagnostics['thermoviscous_loss'] = correction.total_loss
        
        return result
    
    def _solve_streaming(self, result: MultiphysicsResult,
                         diagnostics: Dict) -> MultiphysicsResult:
        """Solve acoustic streaming."""
        if result.acoustic_field is None:
            raise ValueError("Acoustic field required for streaming")
        
        if self._streaming_solver is None:
            self._streaming_solver = StreamingSolver(
                self.config, self._mesh, self._cell_tags,
                self._facet_tags, self.materials
            )
        
        streaming_field = self._streaming_solver.solve(result.acoustic_field)
        
        result.streaming_field = streaming_field
        
        diagnostics['u_stream_max'] = streaming_field.max_velocity
        diagnostics['streaming_reynolds'] = streaming_field.reynolds_number(
            self.materials.water.kinematic_viscosity,
            self.config.physics.wavelength_water
        )
        
        return result
    
    def _solve_particles(self, result: MultiphysicsResult,
                         diagnostics: Dict) -> MultiphysicsResult:
        """Solve particle dynamics."""
        if result.acoustic_field is None:
            raise ValueError("Acoustic field required for particles")
        
        if self._particle_dynamics is None:
            self._particle_dynamics = ParticleDynamics(
                self.config, self._mesh, self.materials
            )
        
        # Compute Gor'kov potential
        gorkov = self._particle_dynamics.compute_gorkov_potential(
            result.acoustic_field
        )
        result.gorkov = gorkov
        
        diagnostics['gorkov_trap_depth'] = gorkov.trap_depth
        diagnostics['f1_contrast'] = gorkov.f1
        diagnostics['f2_contrast'] = gorkov.f2
        
        # Generate initial positions
        n_particles = self.config.physics.num_particles
        x0 = self._particle_dynamics.generate_initial_positions(n_particles)
        
        # Integrate trajectories
        trajectories = self._particle_dynamics.integrate_ensemble(
            gorkov,
            result.streaming_field,
            x0,
            self.config.physics.t_max,
            self.config.physics.dt
        )
        
        result.trajectories = trajectories
        
        # Particle diagnostics
        displacements = [t.displacement for t in trajectories]
        diagnostics['particle_mean_displacement'] = np.mean(displacements)
        diagnostics['particle_max_displacement'] = np.max(displacements)
        diagnostics['particles_trapped'] = sum(1 for t in trajectories if t.is_trapped())
        
        return result


def run_simulation(config: FEMConfig, output_dir: str) -> MultiphysicsResult:
    """
    Run a complete multiphysics simulation.
    
    This is the main entry point for running simulations.
    
    Parameters
    ----------
    config : FEMConfig
        Simulation configuration
    output_dir : str
        Output directory for results
        
    Returns
    -------
    MultiphysicsResult
        Simulation results
    """
    solver = FEMMultiphysicsSolver(config)
    result = solver.solve(output_dir)
    return result
