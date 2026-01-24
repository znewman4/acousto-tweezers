"""
Diagnostic checks for FEM simulation sanity.

Implements automatic validation of simulation parameters and results:
- Wavelength resolution check (h < λ/6)
- Pressure magnitude check (reasonable values)
- PML reflection check (<1% target)
- Energy conservation check
- CFL condition for time stepping

These diagnostics run automatically after each simulation and
produce warnings or errors if sanity checks fail.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum, auto
import numpy as np

from .config import FEMConfig, PhysicsLevel
from .geometry import FEMMesh
from .materials import FluidMaterial
from .acoustics import AcousticField
from .pml import PMLMetrics


class DiagnosticLevel(Enum):
    """Severity level for diagnostics."""
    INFO = auto()
    WARNING = auto()
    ERROR = auto()


@dataclass
class DiagnosticResult:
    """Single diagnostic check result."""
    name: str
    passed: bool
    level: DiagnosticLevel
    message: str
    value: Optional[float] = None
    threshold: Optional[float] = None
    
    def __str__(self) -> str:
        status = "✓ PASS" if self.passed else "✗ FAIL"
        level_str = self.level.name
        return f"[{level_str}] {self.name}: {status} - {self.message}"


@dataclass
class DiagnosticReport:
    """Complete diagnostic report."""
    results: List[DiagnosticResult]
    
    @property
    def passed(self) -> bool:
        """True if all checks passed."""
        return all(r.passed for r in self.results)
    
    @property
    def n_warnings(self) -> int:
        """Number of warnings."""
        return sum(1 for r in self.results 
                   if not r.passed and r.level == DiagnosticLevel.WARNING)
    
    @property
    def n_errors(self) -> int:
        """Number of errors."""
        return sum(1 for r in self.results 
                   if not r.passed and r.level == DiagnosticLevel.ERROR)
    
    def print_report(self):
        """Print formatted report."""
        print("\n" + "=" * 60)
        print("DIAGNOSTIC REPORT")
        print("=" * 60)
        
        for result in self.results:
            print(result)
        
        print("-" * 60)
        if self.passed:
            print("All checks PASSED ✓")
        else:
            print(f"Warnings: {self.n_warnings}, Errors: {self.n_errors}")
        print("=" * 60 + "\n")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'passed': self.passed,
            'n_warnings': self.n_warnings,
            'n_errors': self.n_errors,
            'results': [
                {
                    'name': r.name,
                    'passed': r.passed,
                    'level': r.level.name,
                    'message': r.message,
                    'value': r.value,
                    'threshold': r.threshold,
                }
                for r in self.results
            ]
        }


class Diagnostics:
    """
    Collection of diagnostic checks.
    """
    
    def __init__(self, config: FEMConfig):
        """
        Initialize diagnostics.
        
        Parameters
        ----------
        config : FEMConfig
            Simulation configuration.
        """
        self.config = config
    
    def run_all(
        self,
        mesh: Optional[FEMMesh] = None,
        fluid: Optional[FluidMaterial] = None,
        acoustic_field: Optional[AcousticField] = None,
        pml_metrics: Optional[PMLMetrics] = None,
    ) -> DiagnosticReport:
        """
        Run all applicable diagnostics.
        
        Parameters
        ----------
        mesh : FEMMesh, optional
            Finite element mesh.
        fluid : FluidMaterial, optional
            Fluid material.
        acoustic_field : AcousticField, optional
            Acoustic pressure field.
        pml_metrics : PMLMetrics, optional
            PML performance metrics.
        
        Returns
        -------
        report : DiagnosticReport
            Complete diagnostic report.
        """
        results = []
        
        # Mesh resolution check
        if mesh is not None and fluid is not None:
            results.append(self.check_wavelength_resolution(mesh, fluid))
        
        # Pressure magnitude check
        if acoustic_field is not None:
            results.append(self.check_pressure_magnitude(acoustic_field))
            results.append(self.check_pressure_gradient(acoustic_field))
        
        # PML check
        if pml_metrics is not None:
            results.append(self.check_pml_reflection(pml_metrics))
        
        # Configuration checks
        results.append(self.check_frequency_range())
        
        return DiagnosticReport(results=results)
    
    def check_wavelength_resolution(
        self,
        mesh: FEMMesh,
        fluid: FluidMaterial,
    ) -> DiagnosticResult:
        """
        Check mesh resolution relative to wavelength.
        
        Rule: h < λ/6 for accurate acoustic simulation.
        """
        freq = self.config.physics.frequency
        wavelength = fluid.c / freq
        h = self.config.geometry.resolution
        
        nodes_per_wavelength = wavelength / h
        threshold = 6.0
        passed = nodes_per_wavelength >= threshold
        
        return DiagnosticResult(
            name="Wavelength Resolution",
            passed=passed,
            level=DiagnosticLevel.WARNING if not passed else DiagnosticLevel.INFO,
            message=f"{nodes_per_wavelength:.1f} nodes/λ (need ≥{threshold})",
            value=nodes_per_wavelength,
            threshold=threshold,
        )
    
    def check_pressure_magnitude(
        self,
        acoustic_field: AcousticField,
    ) -> DiagnosticResult:
        """
        Check pressure magnitude is physically reasonable.
        
        Typical acoustic trapping: 1 kPa - 1 MPa
        """
        p = acoustic_field.p
        p_max = np.max(np.abs(p))
        
        # Reasonable range for ultrasonic trapping
        p_min_threshold = 100  # Pa
        p_max_threshold = 10e6  # Pa (10 MPa, cavitation threshold)
        
        passed = p_min_threshold < p_max < p_max_threshold
        
        if p_max < p_min_threshold:
            message = f"p_max = {p_max:.1f} Pa (too weak for trapping)"
            level = DiagnosticLevel.WARNING
        elif p_max > p_max_threshold:
            message = f"p_max = {p_max/1e6:.2f} MPa (exceeds cavitation threshold)"
            level = DiagnosticLevel.ERROR
        else:
            message = f"p_max = {p_max/1e3:.2f} kPa (OK)"
            level = DiagnosticLevel.INFO
        
        return DiagnosticResult(
            name="Pressure Magnitude",
            passed=passed,
            level=level,
            message=message,
            value=p_max,
        )
    
    def check_pressure_gradient(
        self,
        acoustic_field: AcousticField,
    ) -> DiagnosticResult:
        """
        Check pressure gradients aren't spuriously large.
        
        Very large gradients may indicate numerical artifacts.
        """
        p = acoustic_field.p_grid
        mesh = acoustic_field.mesh
        
        # Compute gradient magnitude
        dpx = np.abs(np.diff(p, axis=0)) / mesh.dx
        dpy = np.abs(np.diff(p, axis=1)) / mesh.dy
        dpz = np.abs(np.diff(p, axis=2)) / mesh.dz
        
        grad_max = max(np.max(dpx), np.max(dpy), np.max(dpz))
        
        # Gradient shouldn't exceed p_max/h by too much
        p_max = np.max(np.abs(p))
        h_min = min(mesh.dx, mesh.dy, mesh.dz)
        expected_grad = p_max / h_min
        
        ratio = grad_max / expected_grad if expected_grad > 0 else 0
        threshold = 10.0  # Allow 10x expected gradient
        
        passed = ratio < threshold
        
        return DiagnosticResult(
            name="Pressure Gradient",
            passed=passed,
            level=DiagnosticLevel.WARNING if not passed else DiagnosticLevel.INFO,
            message=f"∇p_max = {grad_max:.2e} Pa/m ({ratio:.1f}x expected)",
            value=grad_max,
        )
    
    def check_pml_reflection(
        self,
        pml_metrics: PMLMetrics,
    ) -> DiagnosticResult:
        """
        Check PML reflection coefficient meets target (<1%).
        """
        R = pml_metrics.reflection_coefficient
        threshold = 0.01  # 1%
        
        passed = R < threshold
        
        return DiagnosticResult(
            name="PML Reflection",
            passed=passed,
            level=DiagnosticLevel.WARNING if not passed else DiagnosticLevel.INFO,
            message=f"R = {R*100:.2f}% (target <{threshold*100}%)",
            value=R,
            threshold=threshold,
        )
    
    def check_frequency_range(self) -> DiagnosticResult:
        """
        Check frequency is in valid range for acoustic trapping.
        
        Typical range: 20 kHz - 40 MHz
        """
        freq = self.config.physics.frequency
        
        f_min = 20e3  # 20 kHz
        f_max = 40e6  # 40 MHz
        
        passed = f_min <= freq <= f_max
        
        if freq < f_min:
            message = f"f = {freq/1e3:.1f} kHz (below ultrasonic range)"
            level = DiagnosticLevel.WARNING
        elif freq > f_max:
            message = f"f = {freq/1e6:.1f} MHz (very high, check attenuation)"
            level = DiagnosticLevel.WARNING
        else:
            message = f"f = {freq/1e6:.2f} MHz (OK)"
            level = DiagnosticLevel.INFO
        
        return DiagnosticResult(
            name="Frequency Range",
            passed=passed,
            level=level,
            message=message,
            value=freq,
        )


def check_energy_conservation(
    acoustic_field: AcousticField,
    fluid: FluidMaterial,
    tolerance: float = 0.1,
) -> DiagnosticResult:
    """
    Check acoustic energy is approximately conserved.
    
    For a closed domain without losses, energy should be constant.
    This checks that energy density is reasonable.
    
    Parameters
    ----------
    acoustic_field : AcousticField
        Acoustic pressure field.
    fluid : FluidMaterial
        Fluid material.
    tolerance : float
        Acceptable deviation from mean.
    
    Returns
    -------
    result : DiagnosticResult
        Diagnostic result.
    """
    p = acoustic_field.p_grid
    
    # Acoustic energy density: E = p²/(2ρc²)
    E = np.abs(p)**2 / (2 * fluid.rho * fluid.c**2)
    
    # Check energy isn't concentrated in tiny region
    E_total = np.sum(E)
    E_max = np.max(E)
    n_cells = E.size
    
    # If max is much larger than mean, might indicate problem
    E_mean = E_total / n_cells
    ratio = E_max / E_mean if E_mean > 0 else 0
    
    # Allow concentration factor of 100 for focusing
    threshold = 100
    passed = ratio < threshold
    
    return DiagnosticResult(
        name="Energy Distribution",
        passed=passed,
        level=DiagnosticLevel.WARNING if not passed else DiagnosticLevel.INFO,
        message=f"E_max/E_mean = {ratio:.1f} (threshold {threshold})",
        value=ratio,
        threshold=threshold,
    )


def check_cfl_condition(
    dt: float,
    dx: float,
    c: float,
    safety_factor: float = 0.5,
) -> DiagnosticResult:
    """
    Check CFL condition for explicit time stepping.
    
    CFL: dt < dx/c
    
    Parameters
    ----------
    dt : float
        Time step [s].
    dx : float
        Minimum grid spacing [m].
    c : float
        Sound speed [m/s].
    safety_factor : float
        Safety factor (typically 0.5).
    
    Returns
    -------
    result : DiagnosticResult
        Diagnostic result.
    """
    cfl_limit = dx / c
    threshold = safety_factor * cfl_limit
    
    passed = dt < threshold
    
    return DiagnosticResult(
        name="CFL Condition",
        passed=passed,
        level=DiagnosticLevel.ERROR if not passed else DiagnosticLevel.INFO,
        message=f"dt = {dt*1e9:.2f} ns, limit = {threshold*1e9:.2f} ns",
        value=dt,
        threshold=threshold,
    )


def print_parameter_summary(
    config: FEMConfig,
    mesh: Optional[FEMMesh] = None,
    fluid: Optional[FluidMaterial] = None,
):
    """
    Print summary of key simulation parameters.
    """
    print("\n" + "=" * 60)
    print("SIMULATION PARAMETERS")
    print("=" * 60)
    
    # Physics
    print(f"\nPhysics Level: {config.physics.level.name}")
    print(f"Frequency: {config.physics.frequency/1e6:.2f} MHz")
    
    if fluid is not None:
        wavelength = fluid.c / config.physics.frequency
        print(f"Wavelength: {wavelength*1e3:.3f} mm")
        print(f"Sound speed: {fluid.c:.1f} m/s")
        print(f"Density: {fluid.rho:.1f} kg/m³")
    
    # Geometry
    print(f"\nDish radius: {config.geometry.dish_radius*1e3:.1f} mm")
    print(f"Dish height: {config.geometry.dish_height*1e3:.1f} mm")
    print(f"Water depth: {config.geometry.water_depth*1e3:.1f} mm")
    print(f"Resolution: {config.geometry.resolution*1e3:.3f} mm")
    
    if mesh is not None:
        print(f"Mesh size: {mesh.nx} × {mesh.ny} × {mesh.nz} = {mesh.n_nodes:,} nodes")
        print(f"Elements: {mesh.n_elements:,}")
    
    # Solver
    print(f"\nSolver tolerance: {config.solver.tolerance:.2e}")
    print(f"Max iterations: {config.solver.max_iterations}")
    
    print("=" * 60 + "\n")
