"""
Perfectly Matched Layer (PML) implementation for open-domain radiation.

PML creates artificial damping at domain boundaries by complex coordinate
stretching, absorbing outgoing waves without reflection.

Complex coordinate stretching:
    x̃ = x + (i/ω) ∫₀ˣ σ(x') dx'

This transforms the Helmholtz equation in the PML region such that
propagating waves become exponentially decaying.

Target performance (from MASTER BRIEF):
    < 1% reflection coefficient

PML parameter report and reflection diagnostic are mandatory outputs.

References
----------
- Berenger, J.P. (1994): A perfectly matched layer for the absorption 
  of electromagnetic waves
- Teixeira & Chew (1997): General closed-form PML constitutive tensors
- Bermúdez et al. (2007): An optimal PML for acoustic media
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from scipy import sparse

from .domains import DomainType
from .geometry import FEMMesh
from .config import GeometryConfig


@dataclass
class PMLParameters:
    """
    Parameters for PML configuration.
    
    Attributes
    ----------
    thickness : float
        PML thickness [m].
    sigma_max : float
        Maximum damping coefficient (normalized).
    order : int
        Polynomial order for damping profile (typically 2-3).
    """
    thickness: float = 5.0e-3  # 5 mm
    sigma_max: float = 1.0     # Normalized to ω
    order: int = 2             # Quadratic profile


@dataclass
class PMLMetrics:
    """
    PML performance metrics and diagnostics.
    
    Computed after solving to validate PML effectiveness.
    """
    # Estimated reflection coefficient
    reflection_coefficient: float
    
    # Maximum wave amplitude at PML outer boundary
    boundary_amplitude: float
    
    # Ratio of energy absorbed in PML to total energy
    absorption_ratio: float
    
    # PML parameters used
    thickness: float
    sigma_max: float
    order: int
    
    # Number of elements per PML thickness
    elements_per_thickness: float
    
    @property
    def meets_target(self) -> bool:
        """Check if reflection < 1% target."""
        return self.reflection_coefficient < 0.01
    
    def report(self) -> str:
        """Generate PML diagnostic report."""
        status = "✓ PASS" if self.meets_target else "✗ FAIL"
        
        lines = [
            "=" * 50,
            "PML Diagnostic Report",
            "=" * 50,
            f"Reflection coefficient:  {self.reflection_coefficient:.4f} ({self.reflection_coefficient*100:.2f}%)",
            f"Target:                  < 1%",
            f"Status:                  {status}",
            "",
            f"PML thickness:           {self.thickness*1e3:.2f} mm",
            f"Max sigma (normalized):  {self.sigma_max:.2f}",
            f"Profile order:           {self.order}",
            f"Elements per thickness:  {self.elements_per_thickness:.1f}",
            "",
            f"Boundary amplitude:      {self.boundary_amplitude:.2e}",
            f"Absorption ratio:        {self.absorption_ratio:.2f}",
            "=" * 50,
        ]
        
        return "\n".join(lines)


class PMLHandler:
    """
    Handler for PML implementation in FEM acoustics.
    
    Provides:
    1. Complex coordinate stretching factors
    2. Modified material properties in PML region
    3. Reflection diagnostics
    """
    
    def __init__(
        self,
        mesh: FEMMesh,
        params: PMLParameters,
        omega: float,
    ):
        """
        Initialize PML handler.
        
        Parameters
        ----------
        mesh : FEMMesh
            Finite element mesh.
        params : PMLParameters
            PML configuration.
        omega : float
            Angular frequency [rad/s].
        """
        self.mesh = mesh
        self.params = params
        self.omega = omega
        
        # Domain boundaries (where PML starts)
        L = params.thickness
        self.x_pml_start = (mesh.x[0] + L, mesh.x[-1] - L)
        self.y_pml_start = (mesh.y[0] + L, mesh.y[-1] - L)
        self.z_pml_start = (mesh.z[0] + L, mesh.z[-1] - L)
        
        # Precompute stretching factors at nodes
        self._compute_stretching_factors()
    
    def _compute_stretching_factors(self):
        """
        Compute complex coordinate stretching factors at all nodes.
        
        Stretching factor:
            s_x = 1 + σ_x(x) / (iω)
        
        where σ_x(x) = σ_max * (d/L)^n for distance d into PML.
        """
        n_nodes = self.mesh.num_nodes
        
        self.sx = np.ones(n_nodes, dtype=np.complex128)
        self.sy = np.ones(n_nodes, dtype=np.complex128)
        self.sz = np.ones(n_nodes, dtype=np.complex128)
        
        L = self.params.thickness
        sigma_max = self.params.sigma_max * self.omega  # Dimensional
        n = self.params.order
        
        for node_id in range(n_nodes):
            x, y, z = self.mesh.nodes[node_id]
            
            # Distance into PML (positive if in PML)
            dx_min = self.x_pml_start[0] - x  # Distance from x_min PML
            dx_max = x - self.x_pml_start[1]  # Distance into x_max PML
            dy_min = self.y_pml_start[0] - y
            dy_max = y - self.y_pml_start[1]
            dz_min = self.z_pml_start[0] - z
            dz_max = z - self.z_pml_start[1]
            
            # X direction
            if dx_min > 0:
                sigma = sigma_max * (dx_min / L) ** n
                self.sx[node_id] = 1 + sigma / (1j * self.omega)
            elif dx_max > 0:
                sigma = sigma_max * (dx_max / L) ** n
                self.sx[node_id] = 1 + sigma / (1j * self.omega)
            
            # Y direction
            if dy_min > 0:
                sigma = sigma_max * (dy_min / L) ** n
                self.sy[node_id] = 1 + sigma / (1j * self.omega)
            elif dy_max > 0:
                sigma = sigma_max * (dy_max / L) ** n
                self.sy[node_id] = 1 + sigma / (1j * self.omega)
            
            # Z direction
            if dz_min > 0:
                sigma = sigma_max * (dz_min / L) ** n
                self.sz[node_id] = 1 + sigma / (1j * self.omega)
            elif dz_max > 0:
                sigma = sigma_max * (dz_max / L) ** n
                self.sz[node_id] = 1 + sigma / (1j * self.omega)
    
    def get_stretching_at_node(self, node_id: int) -> Tuple[complex, complex, complex]:
        """Get stretching factors at a node."""
        return self.sx[node_id], self.sy[node_id], self.sz[node_id]
    
    def get_jacobian_factor(self, node_id: int) -> complex:
        """
        Get Jacobian determinant modification factor.
        
        In the PML, det(J̃) = det(J) * sx * sy * sz
        """
        return self.sx[node_id] * self.sy[node_id] * self.sz[node_id]
    
    def get_inverse_jacobian_factors(self, node_id: int) -> Tuple[complex, complex, complex]:
        """
        Get inverse stretching factors for gradient transformation.
        
        In PML: ∂/∂x̃ = (1/sx) ∂/∂x
        """
        return 1.0 / self.sx[node_id], 1.0 / self.sy[node_id], 1.0 / self.sz[node_id]
    
    def is_in_pml(self, node_id: int) -> bool:
        """Check if node is in PML region."""
        return (
            np.abs(self.sx[node_id] - 1.0) > 1e-10 or
            np.abs(self.sy[node_id] - 1.0) > 1e-10 or
            np.abs(self.sz[node_id] - 1.0) > 1e-10
        )
    
    def modify_element_matrices(
        self,
        Ke: np.ndarray,
        Me: np.ndarray,
        node_ids: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Modify element stiffness and mass matrices for PML.
        
        The weak form in PML becomes:
        
            ∫_Ω̃ (1/ρ) ∇̃φ·∇̃p dṼ - ∫_Ω̃ (ω²/K) φ p dṼ = 0
        
        where ∇̃ = [1/sx ∂/∂x, 1/sy ∂/∂y, 1/sz ∂/∂z]
        and dṼ = sx·sy·sz dV
        
        This is simplified by averaging stretching factors over element.
        """
        # Average stretching factors over element nodes
        sx_avg = np.mean([self.sx[n] for n in node_ids])
        sy_avg = np.mean([self.sy[n] for n in node_ids])
        sz_avg = np.mean([self.sz[n] for n in node_ids])
        
        det_factor = sx_avg * sy_avg * sz_avg
        
        # Stiffness modification
        # K_ij = ∫ (1/ρ)(∂Ni/∂x̃ ∂Nj/∂x̃ + ...) dṼ
        #      = ∫ (1/ρ)(1/sx² ∂Ni/∂x ∂Nj/∂x + ...) sx·sy·sz dV
        
        # For each direction, factor is (1/s²) * (s·s·s) = s_other_1 * s_other_2
        # Simplified: use average factor
        K_factor = det_factor / (sx_avg**2 + sy_avg**2 + sz_avg**2) * 3
        
        # Mass modification: M → M * det_factor
        M_factor = det_factor
        
        Ke_pml = Ke * K_factor
        Me_pml = Me * M_factor
        
        return Ke_pml, Me_pml
    
    def compute_reflection_coefficient(
        self,
        pressure: np.ndarray,
        reference_amplitude: float,
    ) -> float:
        """
        Estimate reflection coefficient from solved pressure field.
        
        R ≈ |p|_boundary / |p|_reference
        
        Parameters
        ----------
        pressure : np.ndarray
            Solved pressure field (nodal values).
        reference_amplitude : float
            Reference pressure amplitude inside domain.
        
        Returns
        -------
        R : float
            Estimated reflection coefficient.
        """
        # Find nodes at PML outer boundary
        tol = self.mesh.dx / 2
        x_min, x_max = self.mesh.x[0], self.mesh.x[-1]
        y_min, y_max = self.mesh.y[0], self.mesh.y[-1]
        z_min, z_max = self.mesh.z[0], self.mesh.z[-1]
        
        boundary_amplitudes = []
        
        for node_id in range(self.mesh.num_nodes):
            x, y, z = self.mesh.nodes[node_id]
            
            is_outer_boundary = (
                abs(x - x_min) < tol or abs(x - x_max) < tol or
                abs(y - y_min) < tol or abs(y - y_max) < tol or
                abs(z - z_min) < tol or abs(z - z_max) < tol
            )
            
            if is_outer_boundary:
                boundary_amplitudes.append(np.abs(pressure[node_id]))
        
        if not boundary_amplitudes or reference_amplitude < 1e-20:
            return 0.0
        
        max_boundary = np.max(boundary_amplitudes)
        return max_boundary / reference_amplitude
    
    def compute_metrics(
        self,
        pressure: np.ndarray,
    ) -> PMLMetrics:
        """
        Compute comprehensive PML diagnostics.
        
        Parameters
        ----------
        pressure : np.ndarray
            Solved pressure field (nodal values).
        
        Returns
        -------
        metrics : PMLMetrics
            PML performance metrics.
        """
        # Reference amplitude (max in non-PML region)
        reference_amplitude = 0.0
        pml_energy = 0.0
        total_energy = 0.0
        
        for node_id in range(self.mesh.num_nodes):
            p_amp = np.abs(pressure[node_id])
            energy = p_amp**2  # Proportional to energy
            total_energy += energy
            
            if self.is_in_pml(node_id):
                pml_energy += energy
            else:
                reference_amplitude = max(reference_amplitude, p_amp)
        
        # Reflection coefficient
        R = self.compute_reflection_coefficient(pressure, reference_amplitude)
        
        # Boundary amplitude
        boundary_amp = R * reference_amplitude
        
        # Absorption ratio
        absorption_ratio = 1.0 - (pml_energy / total_energy if total_energy > 0 else 0)
        
        # Elements per PML thickness
        h = min(self.mesh.dx, self.mesh.dy, self.mesh.dz)
        elements_per_thickness = self.params.thickness / h
        
        return PMLMetrics(
            reflection_coefficient=R,
            boundary_amplitude=boundary_amp,
            absorption_ratio=absorption_ratio,
            thickness=self.params.thickness,
            sigma_max=self.params.sigma_max,
            order=self.params.order,
            elements_per_thickness=elements_per_thickness,
        )


def optimal_pml_parameters(
    frequency: float,
    sound_speed: float,
    target_reflection: float = 0.01,
) -> PMLParameters:
    """
    Compute optimal PML parameters for given conditions.
    
    Rule of thumb:
    - Thickness ≈ 2-3 wavelengths
    - σ_max ≈ (n+1) / (2L) * log(1/R) for polynomial order n
    
    Parameters
    ----------
    frequency : float
        Operating frequency [Hz].
    sound_speed : float
        Sound speed in medium [m/s].
    target_reflection : float
        Target reflection coefficient.
    
    Returns
    -------
    params : PMLParameters
        Recommended PML parameters.
    """
    wavelength = sound_speed / frequency
    
    # Thickness: 2 wavelengths typically sufficient
    thickness = 2 * wavelength
    
    # Order 2 is common
    order = 2
    
    # Optimal sigma_max for target reflection
    # R ≈ exp(-2 σ_max L / (n+1))
    # σ_max = -(n+1)/(2L) * ln(R)
    sigma_max = -(order + 1) / (2 * thickness) * np.log(target_reflection)
    sigma_max_normalized = sigma_max / (2 * np.pi * frequency)  # Normalize by ω
    
    return PMLParameters(
        thickness=thickness,
        sigma_max=sigma_max_normalized,
        order=order,
    )
