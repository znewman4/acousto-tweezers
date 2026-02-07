"""
Diagnostics and validation for FEniCSx acoustic simulation.

EXPANDED DIAGNOSTICS (from MASTER SPEC):
1. Mesh + Resolution diagnostics (mesh_report.txt)
   - wavelength λ
   - PPW in each domain
   - element size statistics by domain
   - cell/DOF counts
   
2. Solver diagnostics (solver_report.txt)
   - KSP/PC type
   - iteration count
   - final residual
   - convergence reason
   - timing
   
3. Field sanity diagnostics (acoustics_report.txt)
   - max/mean/rms |p|
   - ∫|p|² dx by domain
   - ∫|∇p|² dx by domain
   - boundary flux proxy
   
4. Coupling diagnostics (interface_residuals.txt)
   - interface pressure jump L2 norm
   - normal velocity mismatch L2 norm
   - traction residual L2 norm
   
5. PML diagnostics (pml_report.txt)
   - plane wave reflection metric
   - PASS/WARN/FAIL status
   
6. Actuation diagnostics (actuation_report.txt)
   - actuation region area
   - imposed amplitude
   - estimated injected power

Author: Acousto-Tweezers Project
Date: January 2026
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from pathlib import Path
import numpy as np
import json
from datetime import datetime

from .config import FEMConfig, PhysicsLevel
from .materials import MaterialDatabase


@dataclass
class MeshQualityMetrics:
    """Mesh quality diagnostics."""
    wavelength: float          # λ [m]
    min_element_size: float    # h_min [m]
    max_element_size: float    # h_max [m]
    mean_element_size: float   # h_mean [m]
    points_per_wavelength: float  # PPW = λ/h
    num_elements: int
    num_nodes: int
    num_dofs_pressure: int = 0
    num_dofs_displacement: int = 0
    # Per-domain stats
    domain_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Interface facet counts
    interface_counts: Dict[str, int] = field(default_factory=dict)
    
    @property
    def ppw_status(self) -> str:
        """Status based on PPW (recommend > 10)."""
        if self.points_per_wavelength >= 10:
            return "PASS"
        elif self.points_per_wavelength >= 5:
            return "WARN"
        else:
            return "FAIL"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'wavelength_m': self.wavelength,
            'min_element_size_m': self.min_element_size,
            'max_element_size_m': self.max_element_size,
            'mean_element_size_m': self.mean_element_size,
            'ppw': self.points_per_wavelength,
            'ppw_status': self.ppw_status,
            'num_elements': self.num_elements,
            'num_nodes': self.num_nodes,
            'num_dofs_pressure': self.num_dofs_pressure,
            'num_dofs_displacement': self.num_dofs_displacement,
            'domain_stats': self.domain_stats,
            'interface_counts': self.interface_counts,
        }
    
    def generate_mesh_report(self) -> str:
        """Generate mesh_report.txt content."""
        lines = [
            "=" * 60,
            "MESH REPORT",
            "=" * 60,
            "",
            "WAVELENGTH & RESOLUTION:",
            f"  Wavelength (λ): {self.wavelength*1e6:.1f} μm",
            f"  [{self.ppw_status}] Points per wavelength: {self.points_per_wavelength:.1f} (target ≥10)",
            "",
            "ELEMENT SIZE STATISTICS:",
            f"  Minimum h: {self.min_element_size*1e6:.1f} μm",
            f"  Maximum h: {self.max_element_size*1e6:.1f} μm", 
            f"  Mean h: {self.mean_element_size*1e6:.1f} μm",
            "",
            "CELL/DOF COUNTS:",
            f"  Number of cells: {self.num_elements}",
            f"  Number of nodes: {self.num_nodes}",
            f"  Pressure DOFs: {self.num_dofs_pressure}",
            f"  Displacement DOFs: {self.num_dofs_displacement}",
            "",
        ]
        
        if self.domain_stats:
            lines.append("PER-DOMAIN STATISTICS:")
            for domain, stats in self.domain_stats.items():
                lines.append(f"  {domain}:")
                # Handle both dict and int values
                if isinstance(stats, dict):
                    lines.append(f"    Cells: {stats.get('num_cells', 0)}")
                    lines.append(f"    h (min/mean/max): {stats.get('h_min', 0)*1e6:.1f} / {stats.get('h_mean', 0)*1e6:.1f} / {stats.get('h_max', 0)*1e6:.1f} μm")
                    lines.append(f"    PPW: {stats.get('ppw', 0):.1f}")
                else:
                    # stats is just the cell count
                    lines.append(f"    Cells: {stats}")
            lines.append("")
        
        if self.interface_counts:
            lines.append("INTERFACE FACET COUNTS:")
            for interface, count in self.interface_counts.items():
                lines.append(f"  {interface}: {count}")
            lines.append("")
        
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class SolverMetrics:
    """Solver performance diagnostics."""
    ksp_type: str
    pc_type: str
    num_iterations: int
    final_residual: float
    converged: bool
    convergence_reason: str
    assembly_time_s: float
    solve_time_s: float
    total_time_s: float
    
    @property
    def status(self) -> str:
        if self.converged:
            return "PASS"
        else:
            return "FAIL"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'ksp_type': self.ksp_type,
            'pc_type': self.pc_type,
            'num_iterations': self.num_iterations,
            'final_residual': self.final_residual,
            'converged': self.converged,
            'convergence_reason': self.convergence_reason,
            'assembly_time_s': self.assembly_time_s,
            'solve_time_s': self.solve_time_s,
            'total_time_s': self.total_time_s,
        }
    
    def generate_solver_report(self) -> str:
        """Generate solver_report.txt content."""
        lines = [
            "=" * 60,
            "SOLVER REPORT",
            "=" * 60,
            "",
            "SOLVER CONFIGURATION:",
            f"  KSP type: {self.ksp_type}",
            f"  PC type: {self.pc_type}",
            "",
            "CONVERGENCE:",
            f"  [{self.status}] Converged: {self.converged}",
            f"  Iterations: {self.num_iterations}",
            f"  Final residual: {self.final_residual:.2e}",
            f"  Reason: {self.convergence_reason}",
            "",
            "TIMING:",
            f"  Assembly time: {self.assembly_time_s:.2f} s",
            f"  Solve time: {self.solve_time_s:.2f} s",
            f"  Total time: {self.total_time_s:.2f} s",
            "",
            "=" * 60,
        ]
        return "\n".join(lines)


@dataclass
class AcousticFieldMetrics:
    """Acoustic field diagnostics."""
    p_max: float      # max|p| [Pa]
    p_mean: float     # mean|p| [Pa]
    p_rms: float      # rms|p| [Pa]
    p_min_real: float # min(Re(p)) [Pa]
    p_max_real: float # max(Re(p)) [Pa]
    # Energy-like quantities by domain
    energy_by_domain: Dict[str, float] = field(default_factory=dict)  # ∫|p|² dx
    gradient_energy_by_domain: Dict[str, float] = field(default_factory=dict)  # ∫|∇p|² dx
    # Boundary flux
    boundary_flux: Optional[float] = None  # ∫(1/ρ) ∂ₙp p̄ ds
    
    @property
    def dynamic_range(self) -> float:
        """Dynamic range of pressure field."""
        if self.p_mean > 0:
            return self.p_max / self.p_mean
        return 0.0
    
    @property
    def status(self) -> str:
        if self.p_max > 0:
            return "PASS"
        return "FAIL"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'p_max_Pa': self.p_max,
            'p_mean_Pa': self.p_mean,
            'p_rms_Pa': self.p_rms,
            'dynamic_range': self.dynamic_range,
            'energy_by_domain': self.energy_by_domain,
            'gradient_energy_by_domain': self.gradient_energy_by_domain,
            'boundary_flux': self.boundary_flux,
        }
    
    def generate_acoustics_report(self) -> str:
        """Generate acoustics_report.txt content."""
        lines = [
            "=" * 60,
            "ACOUSTICS REPORT",
            "=" * 60,
            "",
            "PRESSURE FIELD STATISTICS:",
            f"  [{self.status}] max|p|: {self.p_max:.2e} Pa",
            f"  mean|p|: {self.p_mean:.2e} Pa",
            f"  rms|p|: {self.p_rms:.2e} Pa",
            f"  Re(p) range: [{self.p_min_real:.2e}, {self.p_max_real:.2e}] Pa",
            f"  Dynamic range: {self.dynamic_range:.1f}x",
            "",
        ]
        
        if self.energy_by_domain:
            lines.append("ENERGY BY DOMAIN (∫|p|² dx):")
            for domain, energy in self.energy_by_domain.items():
                lines.append(f"  {domain}: {energy:.2e} Pa²·m³")
            lines.append("")
        
        if self.gradient_energy_by_domain:
            lines.append("GRADIENT ENERGY BY DOMAIN (∫|∇p|² dx):")
            for domain, energy in self.gradient_energy_by_domain.items():
                lines.append(f"  {domain}: {energy:.2e} Pa²/m")
            lines.append("")
        
        if self.boundary_flux is not None:
            lines.append("BOUNDARY FLUX:")
            lines.append(f"  ∫(1/ρ) ∂ₙp p̄ ds: {self.boundary_flux:.2e}")
            lines.append("")
        
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class InterfaceResiduals:
    """Fluid-solid interface coupling diagnostics."""
    pressure_jump_l2: float        # ||[p]||_L2 on interface
    velocity_mismatch_l2: float    # ||vf·n - vs·n||_L2
    traction_residual_l2: float    # ||σn + pn||_L2
    
    @property
    def status(self) -> str:
        # These should be small (relative to field magnitude)
        if self.pressure_jump_l2 < 1e-6 and self.velocity_mismatch_l2 < 1e-6:
            return "PASS"
        elif self.pressure_jump_l2 < 1e-3 and self.velocity_mismatch_l2 < 1e-3:
            return "WARN"
        else:
            return "FAIL"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'pressure_jump_l2': self.pressure_jump_l2,
            'velocity_mismatch_l2': self.velocity_mismatch_l2,
            'traction_residual_l2': self.traction_residual_l2,
            'status': self.status,
        }
    
    def generate_interface_report(self) -> str:
        """Generate interface_residuals.txt content."""
        lines = [
            "=" * 60,
            "INTERFACE RESIDUALS (Fluid-Solid Coupling)",
            "=" * 60,
            "",
            "COUPLING CONDITIONS:",
            "  Traction balance: σ(u)·n = -p·n",
            "  Velocity continuity: vf·n = vs·n",
            "",
            "RESIDUAL NORMS:",
            f"  [{self.status}] Pressure jump ||[p]||_L2: {self.pressure_jump_l2:.2e}",
            f"  Velocity mismatch ||vf·n - vs·n||_L2: {self.velocity_mismatch_l2:.2e}",
            f"  Traction residual ||σn + pn||_L2: {self.traction_residual_l2:.2e}",
            "",
            "NOTE: These should be small compared to field magnitudes.",
            "      Non-zero values may indicate discretization error.",
            "",
            "=" * 60,
        ]
        return "\n".join(lines)


@dataclass
class StreamingMetrics:
    """Streaming field diagnostics."""
    u_max: float           # max|u| [m/s]
    u_mean: float          # mean|u| [m/s]
    reynolds_number: float # Re = uL/ν
    shear_rate: float      # ∇u [1/s]
    
    @property
    def is_stokes_regime(self) -> bool:
        """Check if streaming is in Stokes regime (Re << 1)."""
        return self.reynolds_number < 1.0
    
    @property
    def status(self) -> str:
        if self.is_stokes_regime:
            return "PASS"
        return "WARN"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'u_max_m_s': self.u_max,
            'u_mean_m_s': self.u_mean,
            'reynolds_number': self.reynolds_number,
            'stokes_regime': self.is_stokes_regime,
        }


@dataclass
class ParticleMetrics:
    """Particle dynamics diagnostics."""
    num_particles: int
    mean_displacement: float  # [m]
    max_displacement: float   # [m]
    mean_velocity: float      # [m/s]
    num_trapped: int
    trapping_efficiency: float  # fraction trapped
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'num_particles': self.num_particles,
            'mean_displacement_m': self.mean_displacement,
            'max_displacement_m': self.max_displacement,
            'mean_velocity_m_s': self.mean_velocity,
            'num_trapped': self.num_trapped,
            'trapping_efficiency': self.trapping_efficiency,
        }


@dataclass
class PMLMetrics:
    """PML boundary diagnostics."""
    reflection_coefficient: float
    target_reflection: float
    decay_factor: float = 1.0
    field_decay_factor: float = 1.0
    
    @property
    def passed(self) -> bool:
        return self.reflection_coefficient < self.target_reflection
    
    @property
    def status(self) -> str:
        if self.passed:
            return "PASS"
        elif self.reflection_coefficient < 2 * self.target_reflection:
            return "WARN"
        else:
            return "FAIL"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'reflection_coefficient': self.reflection_coefficient,
            'target': self.target_reflection,
            'status': self.status,
            'decay_factor': self.decay_factor,
        }
    
    def generate_pml_report(self) -> str:
        """Generate pml_report.txt content."""
        lines = [
            "=" * 60,
            "PML REPORT",
            "=" * 60,
            "",
            "PML PERFORMANCE:",
            f"  [{self.status}] Reflection coefficient: {self.reflection_coefficient*100:.2f}%",
            f"  Target: <{self.target_reflection*100:.0f}%",
            f"  Field decay factor: {self.decay_factor:.1f}x",
            "",
            "STATUS: " + self.status,
            "",
            "=" * 60,
        ]
        return "\n".join(lines)


@dataclass
class ActuationMetrics:
    """Actuation diagnostics."""
    actuation_type: str
    actuation_region_area: float  # m²
    imposed_amplitude: float      # m (displacement) or Pa (traction)
    estimated_power: float        # W (approximate)
    num_transducers: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'actuation_type': self.actuation_type,
            'actuation_region_area_m2': self.actuation_region_area,
            'imposed_amplitude': self.imposed_amplitude,
            'estimated_power_W': self.estimated_power,
            'num_transducers': self.num_transducers,
        }
    
    def generate_actuation_report(self) -> str:
        """Generate actuation_report.txt content."""
        lines = [
            "=" * 60,
            "ACTUATION REPORT",
            "=" * 60,
            "",
            "ACTUATION CONFIGURATION:",
            f"  Type: {self.actuation_type}",
            f"  Number of transducers: {self.num_transducers}",
            "",
            "ACTUATION REGION:",
            f"  Area: {self.actuation_region_area*1e6:.2f} mm²",
            "",
            "IMPOSED VALUES:",
            f"  Amplitude: {self.imposed_amplitude*1e9:.2f} nm",
            "",
            "POWER ESTIMATE:",
            f"  Estimated injected power: {self.estimated_power*1e3:.2f} mW",
            "",
            "NOTE: Power estimate is approximate (P ≈ ρ c ω² A² S / 2)",
            "",
            "=" * 60,
        ]
        return "\n".join(lines)


@dataclass
class DiagnosticsReport:
    """Complete diagnostics report with all required files."""
    mesh_quality: Optional[MeshQualityMetrics] = None
    acoustic_field: Optional[AcousticFieldMetrics] = None
    streaming: Optional[StreamingMetrics] = None
    particles: Optional[ParticleMetrics] = None
    pml: Optional[PMLMetrics] = None
    solver: Optional[SolverMetrics] = None
    interface: Optional[InterfaceResiduals] = None
    actuation: Optional[ActuationMetrics] = None
    
    physics_level: PhysicsLevel = PhysicsLevel.ACOUSTICS_ONLY
    timestamp: str = ""
    
    def overall_status(self) -> str:
        """Determine overall status from all checks."""
        statuses = []
        
        if self.mesh_quality:
            statuses.append(self.mesh_quality.ppw_status)
        if self.pml:
            statuses.append(self.pml.status)
        if self.acoustic_field:
            statuses.append(self.acoustic_field.status)
        if self.solver:
            statuses.append(self.solver.status)
        if self.interface:
            statuses.append(self.interface.status)
        
        if "FAIL" in statuses:
            return "FAIL"
        elif "WARN" in statuses:
            return "WARN"
        elif not statuses:
            return "UNKNOWN"
        else:
            return "PASS"
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            'physics_level': self.physics_level.name,
            'timestamp': self.timestamp,
            'overall_status': self.overall_status(),
        }
        
        if self.mesh_quality:
            result['mesh_quality'] = self.mesh_quality.to_dict()
        if self.acoustic_field:
            result['acoustic_field'] = self.acoustic_field.to_dict()
        if self.streaming:
            result['streaming'] = self.streaming.to_dict()
        if self.particles:
            result['particles'] = self.particles.to_dict()
        if self.pml:
            result['pml'] = self.pml.to_dict()
        if self.solver:
            result['solver'] = self.solver.to_dict()
        if self.interface:
            result['interface'] = self.interface.to_dict()
        if self.actuation:
            result['actuation'] = self.actuation.to_dict()
        
        return result
    
    def save(self, output_dir: str):
        """
        Save all diagnostic files to the correct locations.
        
        Creates:
        - diagnostics/sanity_report.txt
        - diagnostics/mesh_report.txt  
        - diagnostics/solver_report.txt
        - diagnostics/acoustics_report.txt
        - diagnostics/interface_residuals.txt (if solids enabled)
        - diagnostics/pml_report.txt (if PML enabled)
        - diagnostics/actuation_report.txt
        - summary.csv
        """
        output_dir = Path(output_dir)
        diag_dir = output_dir / "diagnostics"
        diag_dir.mkdir(parents=True, exist_ok=True)
        
        # JSON summary
        with open(diag_dir / "diagnostics.json", 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        # Main sanity report
        with open(diag_dir / "sanity_report.txt", 'w') as f:
            f.write(self.generate_report())
        
        # Individual report files
        if self.mesh_quality:
            with open(diag_dir / "mesh_report.txt", 'w') as f:
                f.write(self.mesh_quality.generate_mesh_report())
        
        if self.solver:
            with open(diag_dir / "solver_report.txt", 'w') as f:
                f.write(self.solver.generate_solver_report())
        
        if self.acoustic_field:
            with open(diag_dir / "acoustics_report.txt", 'w') as f:
                f.write(self.acoustic_field.generate_acoustics_report())
        
        if self.interface:
            with open(diag_dir / "interface_residuals.txt", 'w') as f:
                f.write(self.interface.generate_interface_report())
        
        if self.pml:
            with open(diag_dir / "pml_report.txt", 'w') as f:
                f.write(self.pml.generate_pml_report())
        
        if self.actuation:
            with open(diag_dir / "actuation_report.txt", 'w') as f:
                f.write(self.actuation.generate_actuation_report())
        
        # Summary CSV for quick parsing
        with open(output_dir / "summary.csv", 'w') as f:
            f.write("metric,value,unit\n")
            self._write_csv_metrics(f)
    
    def _write_csv_metrics(self, f):
        """Write metrics to CSV file."""
        if self.mesh_quality:
            f.write(f"wavelength_m,{self.mesh_quality.wavelength},\n")
            f.write(f"ppw,{self.mesh_quality.points_per_wavelength},\n")
            f.write(f"num_elements,{self.mesh_quality.num_elements},\n")
            f.write(f"num_nodes,{self.mesh_quality.num_nodes},\n")
        
        if self.acoustic_field:
            f.write(f"p_max_Pa,{self.acoustic_field.p_max},\n")
            f.write(f"p_mean_Pa,{self.acoustic_field.p_mean},\n")
            f.write(f"p_rms_Pa,{self.acoustic_field.p_rms},\n")
        
        if self.solver:
            f.write(f"solve_time_s,{self.solver.solve_time_s},\n")
            f.write(f"num_iterations,{self.solver.num_iterations},\n")
        
        if self.streaming:
            f.write(f"u_stream_max_m_s,{self.streaming.u_max},\n")
            f.write(f"reynolds_streaming,{self.streaming.reynolds_number},\n")
        
        if self.particles:
            f.write(f"mean_displacement_m,{self.particles.mean_displacement},\n")
            f.write(f"trapping_efficiency,{self.particles.trapping_efficiency},\n")
        
        if self.pml:
            f.write(f"pml_reflection,{self.pml.reflection_coefficient},\n")
        
        if self.interface:
            f.write(f"pressure_jump_l2,{self.interface.pressure_jump_l2},\n")
            f.write(f"velocity_mismatch_l2,{self.interface.velocity_mismatch_l2},\n")
    
    def generate_report(self) -> str:
        """Generate human-readable sanity report."""
        lines = [
            "=" * 60,
            "PHYSICS SANITY REPORT",
            f"Physics Level: {self.physics_level.name}",
            f"Timestamp: {self.timestamp}",
            f"Overall Status: {self.overall_status()}",
            "=" * 60,
            "",
        ]
        
        # Mesh quality
        if self.mesh_quality:
            mq = self.mesh_quality
            lines.extend([
                "MESH QUALITY:",
                f"  Wavelength: {mq.wavelength*1e6:.1f} μm",
                f"  Element size: {mq.min_element_size*1e6:.1f} - {mq.max_element_size*1e6:.1f} μm",
                f"  [{mq.ppw_status}] Points per wavelength: {mq.points_per_wavelength:.1f} (target: ≥10)",
                f"  Elements: {mq.num_elements}, Nodes: {mq.num_nodes}",
                "",
            ])
        
        # Solver
        if self.solver:
            sv = self.solver
            lines.extend([
                "SOLVER:",
                f"  [{sv.status}] Converged: {sv.converged}",
                f"  Iterations: {sv.num_iterations}",
                f"  Total time: {sv.total_time_s:.2f} s",
                "",
            ])
        
        # Acoustic field
        if self.acoustic_field:
            af = self.acoustic_field
            lines.extend([
                "ACOUSTIC FIELD:",
                f"  [{af.status}] max|p|: {af.p_max:.2e} Pa",
                f"  [INFO] mean|p|: {af.p_mean:.2e} Pa",
                f"  [INFO] rms|p|: {af.p_rms:.2e} Pa",
                f"  [INFO] Dynamic range: {af.dynamic_range:.1f}x",
                "",
            ])
        
        # Interface residuals
        if self.interface:
            ir = self.interface
            lines.extend([
                "INTERFACE COUPLING:",
                f"  [{ir.status}] Pressure jump: {ir.pressure_jump_l2:.2e}",
                f"  Velocity mismatch: {ir.velocity_mismatch_l2:.2e}",
                f"  Traction residual: {ir.traction_residual_l2:.2e}",
                "",
            ])
        
        # PML
        if self.pml:
            pm = self.pml
            lines.extend([
                "PML BOUNDARY:",
                f"  [{pm.status}] Reflection: {pm.reflection_coefficient*100:.2f}% (target: <{pm.target_reflection*100:.0f}%)",
                f"  [INFO] Decay factor: {pm.decay_factor:.1f}x",
                "",
            ])
        
        # Actuation
        if self.actuation:
            ac = self.actuation
            lines.extend([
                "ACTUATION:",
                f"  Type: {ac.actuation_type}",
                f"  Amplitude: {ac.imposed_amplitude*1e9:.2f} nm",
                f"  Est. power: {ac.estimated_power*1e3:.2f} mW",
                "",
            ])
        
        # Streaming
        if self.streaming:
            st = self.streaming
            lines.extend([
                "ACOUSTIC STREAMING:",
                f"  [INFO] max|u|: {st.u_max:.2e} m/s",
                f"  [{st.status}] Re_streaming: {st.reynolds_number:.2e} (expect: <1)",
                "",
            ])
        
        # Particles
        if self.particles:
            pt = self.particles
            lines.extend([
                "PARTICLE DYNAMICS:",
                f"  [INFO] Particles: {pt.num_particles}",
                f"  [INFO] Mean displacement: {pt.mean_displacement*1e6:.2f} μm",
                f"  [INFO] Max displacement: {pt.max_displacement*1e6:.2f} μm",
                f"  [INFO] Trapped: {pt.num_trapped} ({pt.trapping_efficiency*100:.1f}%)",
                "",
            ])
        
        lines.append("=" * 60)
        
        return "\n".join(lines)


def _get_pressure_dofs(result: "MultiphysicsResult") -> int:
    """Extract pressure DOF count from result.
    
    Returns the number of DOFs in the pressure function space.
    Returns 0 if no acoustic field available (with warning).
    """
    if result.acoustic_field is None:
        return 0
    try:
        V = result.acoustic_field.p_function.function_space
        return V.dofmap.index_map.size_global
    except Exception:
        # Fallback to array length
        return len(result.acoustic_field.p)


def _get_displacement_dofs(result: "MultiphysicsResult") -> int:
    """Extract displacement DOF count from result.
    
    Returns the number of DOFs in the displacement function space.
    Returns 0 if no displacement field available.
    """
    if result.displacement_field is None:
        return 0
    try:
        V = result.displacement_field.u_function.function_space
        return V.dofmap.index_map.size_global
    except Exception:
        return 0


def compute_diagnostics(result: "MultiphysicsResult",
                        config: FEMConfig,
                        materials: MaterialDatabase) -> DiagnosticsReport:
    """
    Compute comprehensive diagnostics from simulation result.
    
    Parameters
    ----------
    result : MultiphysicsResult
        Simulation result
    config : FEMConfig
        Configuration
    materials : MaterialDatabase
        Material properties
        
    Returns
    -------
    DiagnosticsReport
        Complete diagnostics with all required files
    """
    report = DiagnosticsReport(
        physics_level=config.physics_level,
        timestamp=datetime.now().isoformat(),
    )
    
    # Mesh quality
    omega = config.physics.omega
    c_water = materials.water.sound_speed
    wavelength = 2 * np.pi * c_water / omega
    
    mesh_info = result.mesh_info
    h_min = mesh_info.min_element_size
    h_max = mesh_info.max_element_size
    h_mean = (h_min + h_max) / 2  # Approximate
    
    report.mesh_quality = MeshQualityMetrics(
        wavelength=wavelength,
        min_element_size=h_min,
        max_element_size=h_max,
        mean_element_size=h_mean,
        points_per_wavelength=wavelength / h_max,
        num_elements=mesh_info.num_cells,
        num_nodes=mesh_info.num_nodes,
        num_dofs_pressure=_get_pressure_dofs(result),
        num_dofs_displacement=_get_displacement_dofs(result),
        domain_stats=getattr(mesh_info, 'domain_counts', {}),
        interface_counts=getattr(mesh_info, 'interface_counts', {}),
    )
    
    # Solver metrics (from timing dict)
    if hasattr(result, 'timing') and result.timing:
        timing = result.timing
        report.solver = SolverMetrics(
            ksp_type=config.solver.ksp_type,
            pc_type=config.solver.pc_type,
            num_iterations=timing.get('iterations', 0),
            final_residual=timing.get('residual', 0.0),
            converged=timing.get('converged', True),
            convergence_reason=timing.get('convergence_reason', 'converged'),
            assembly_time_s=timing.get('assembly_time', 0.0),
            solve_time_s=timing.get('solve_time', 0.0),
            total_time_s=timing.get('total_time', 0.0),
        )
    
    # Acoustic field
    if result.acoustic_field is not None:
        af = result.acoustic_field
        p = af.p
        
        report.acoustic_field = AcousticFieldMetrics(
            p_max=np.max(np.abs(p)),
            p_mean=np.mean(np.abs(p)),
            p_rms=np.sqrt(np.mean(np.abs(p)**2)),
            p_min_real=np.min(np.real(p)),
            p_max_real=np.max(np.real(p)),
        )
    
    # Actuation metrics
    report.actuation = ActuationMetrics(
        actuation_type=config.physics.actuation_type,
        actuation_region_area=np.pi * (config.geometry.dish_inner_radius)**2,  # Approx
        imposed_amplitude=config.physics.actuation_amplitude,
        estimated_power=estimate_actuation_power(config, materials),
        num_transducers=config.physics.num_transducers,
    )
    
    # PML metrics
    if result.pml_metrics is not None:
        pm = result.pml_metrics
        report.pml = PMLMetrics(
            reflection_coefficient=pm.reflection_coefficient,
            target_reflection=config.output.pml_reflection_target,
            decay_factor=getattr(pm, 'field_decay_factor', 1.0),
        )
    
    # Interface residuals (only if solids enabled)
    if config.physics_level.value >= PhysicsLevel.FLUID_SOLID.value:
        if hasattr(result, 'coupled_field') and result.coupled_field is not None:
            cf = result.coupled_field
            if cf.interface_residuals:
                ir = cf.interface_residuals
                report.interface = InterfaceResiduals(
                    pressure_jump_l2=ir.get('pressure_jump', 0.0),
                    velocity_mismatch_l2=ir.get('velocity_mismatch', 0.0),
                    traction_residual_l2=ir.get('traction_residual', 0.0),
                )
        else:
            # Placeholder if coupling not yet computed
            report.interface = InterfaceResiduals(
                pressure_jump_l2=0.0,
                velocity_mismatch_l2=0.0,
                traction_residual_l2=0.0,
            )
    
    # Streaming
    if result.streaming_field is not None:
        sf = result.streaming_field
        nu = materials.water.kinematic_viscosity
        L = config.physics.wavelength_water
        
        report.streaming = StreamingMetrics(
            u_max=sf.max_velocity,
            u_mean=sf.mean_velocity,
            reynolds_number=sf.max_velocity * L / nu if nu > 0 else 0,
            shear_rate=0.0,  # Would need gradient computation
        )
    
    # Particles
    if result.trajectories is not None:
        trajs = result.trajectories
        n = len(trajs)
        if n > 0:
            displacements = [t.displacement for t in trajs]
            velocities = [t.path_length / t.t[-1] if len(t.t) > 0 and t.t[-1] > 0 else 0 for t in trajs]
            trapped = [t.is_trapped() for t in trajs]
            
            report.particles = ParticleMetrics(
                num_particles=n,
                mean_displacement=np.mean(displacements),
                max_displacement=np.max(displacements),
                mean_velocity=np.mean(velocities),
                num_trapped=sum(trapped),
                trapping_efficiency=sum(trapped) / n,
            )
    
    return report


def estimate_actuation_power(config: FEMConfig, materials: MaterialDatabase) -> float:
    """
    Estimate the injected acoustic power from actuation.
    
    Uses P ≈ ρ c ω² A² S / 2 for displacement actuation.
    
    Parameters
    ----------
    config : FEMConfig
        Configuration
    materials : MaterialDatabase
        Material properties
        
    Returns
    -------
    float
        Estimated power in Watts
    """
    rho = materials.water.density
    c = materials.water.sound_speed
    omega = config.physics.omega
    A = config.physics.actuation_amplitude
    S = np.pi * (config.geometry.dish_inner_radius)**2  # Approximate area
    
    # P = (1/2) * Z * v^2 * S = (1/2) * ρc * (ωA)^2 * S
    power = 0.5 * rho * c * (omega * A)**2 * S
    return power
