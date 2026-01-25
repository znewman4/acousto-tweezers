"""
Thermoviscous acoustics module.

Implements boundary layer corrections for acoustic fields near solid walls.

Thermoviscous effects (from MASTER BRIEF):

    δ_v = √(2ν/ω)    (viscous boundary layer thickness)
    δ_t = √(2α/ω)    (thermal boundary layer thickness)

where:
    ν = η/ρ (kinematic viscosity)
    α = κ_th/(ρ c_p) (thermal diffusivity)

For typical water at 1 MHz:
    δ_v ≈ 0.56 μm
    δ_t ≈ 0.22 μm

The boundary layer causes acoustic energy dissipation. The correction
can be applied as:
    1. Modified boundary conditions (slip velocity, temperature jump)
    2. Effective domain loss via imaginary wavenumber component
    3. Full thermoviscous formulation (Navier-Stokes + heat equation)

This module implements approach (2) as a pragmatic compromise between
accuracy and computational cost.

References
----------
- Nyborg (1965): Acoustic streaming
- Baasch & Dual (2018): Acoustofluidic particle dynamics
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from .domains import DomainType
from .geometry import FEMMesh
from .materials import FluidMaterial
from .config import FEMConfig


@dataclass
class ThermoviscousParameters:
    """
    Parameters for thermoviscous corrections.
    """
    # Angular frequency [rad/s]
    omega: float
    
    # Fluid properties
    rho: float        # Density [kg/m³]
    eta: float        # Dynamic viscosity [Pa·s]
    c: float          # Sound speed [m/s]
    kappa: float      # Thermal conductivity [W/(m·K)]
    cp: float         # Specific heat [J/(kg·K)]
    beta: float       # Thermal expansion coefficient [1/K]
    
    @classmethod
    def from_fluid(cls, fluid: FluidMaterial, frequency: float) -> "ThermoviscousParameters":
        """Create from fluid material and frequency."""
        omega = 2 * np.pi * frequency
        return cls(
            omega=omega,
            rho=fluid.rho,
            eta=fluid.eta,
            c=fluid.c,
            kappa=fluid.kappa,
            cp=fluid.cp,
            beta=fluid.beta,
        )
    
    @property
    def nu(self) -> float:
        """Kinematic viscosity ν = η/ρ [m²/s]."""
        return self.eta / self.rho
    
    @property
    def alpha(self) -> float:
        """Thermal diffusivity α = κ/(ρ c_p) [m²/s]."""
        return self.kappa / (self.rho * self.cp)
    
    @property
    def delta_v(self) -> float:
        """Viscous boundary layer thickness δ_v = √(2ν/ω) [m]."""
        return np.sqrt(2 * self.nu / self.omega)
    
    @property
    def delta_t(self) -> float:
        """Thermal boundary layer thickness δ_t = √(2α/ω) [m]."""
        return np.sqrt(2 * self.alpha / self.omega)
    
    @property
    def Pr(self) -> float:
        """Prandtl number Pr = ν/α."""
        return self.nu / self.alpha
    
    @property
    def k(self) -> complex:
        """
        Effective wavenumber including thermoviscous losses.
        
        k = ω/c + i·(α_v + α_t)
        
        where α_v and α_t are the viscous and thermal absorption coefficients.
        """
        k0 = self.omega / self.c
        
        # Classical absorption (Stokes-Kirchhoff)
        # α = ω²/(2ρc³) [4η/3 + η_b + κ_th(γ-1)/cp]
        # Simplified: α ≈ ω² δ_v/(2c²) [1 + (γ-1)/√Pr]
        
        # For simplicity, use bulk viscosity approximation
        # η_b ≈ 3η for water (Litovitz approximation)
        gamma = 1.0  # Isothermal approximation for liquids
        
        alpha_v = self.omega**2 * (4/3 * self.eta) / (2 * self.rho * self.c**3)
        alpha_t = self.omega**2 * (self.kappa * (gamma - 1) / self.cp) / (2 * self.rho * self.c**3)
        
        return k0 + 1j * (alpha_v + alpha_t)
    
    @property
    def wavelength(self) -> float:
        """Acoustic wavelength λ = c/f = 2πc/ω [m]."""
        return 2 * np.pi * self.c / self.omega
    
    def penetration_depth(self) -> float:
        """Acoustic penetration depth where amplitude decays by 1/e [m]."""
        return 1.0 / np.imag(self.k)


@dataclass
class ThermoviscousCorrection:
    """
    Thermoviscous correction field.
    """
    # Grid coordinates
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    
    # Correction factor (complex, multiplicative)
    correction: np.ndarray  # (nx, ny, nz)
    
    # Boundary layer mask
    boundary_layer_mask: np.ndarray  # Boolean mask
    
    # Parameters used
    params: ThermoviscousParameters
    
    # Mesh reference
    mesh: Optional[FEMMesh] = None


def compute_distance_to_boundary(
    mesh: FEMMesh,
    solid_domains: list = None,
) -> np.ndarray:
    """
    Compute signed distance to nearest solid boundary.
    
    Parameters
    ----------
    mesh : FEMMesh
        Finite element mesh.
    solid_domains : list, optional
        List of DomainType values to consider as solid.
        Default: [PLATE, WALL]
    
    Returns
    -------
    distance : np.ndarray
        Distance field (nx, ny, nz).
    """
    if solid_domains is None:
        solid_domains = [DomainType.PLATE, DomainType.WALL]
    
    nx, ny, nz = mesh.nx, mesh.ny, mesh.nz
    X, Y, Z = np.meshgrid(mesh.x, mesh.y, mesh.z, indexing='ij')
    
    # Initialize with large value
    distance = np.full((nx, ny, nz), np.inf)
    
    # Get solid node coordinates
    solid_nodes = []
    for domain_type in solid_domains:
        if domain_type in mesh.domain_info:
            elem_ids = mesh.domain_info[domain_type].element_ids
            if len(elem_ids) > 0:
                node_ids = np.unique(mesh.elements[elem_ids].flatten())
                solid_nodes.extend(mesh.nodes[node_ids].tolist())
    
    if not solid_nodes:
        # No solid boundaries - return domain size
        Lx = mesh.x[-1] - mesh.x[0]
        Ly = mesh.y[-1] - mesh.y[0]
        Lz = mesh.z[-1] - mesh.z[0]
        return np.full((nx, ny, nz), min(Lx, Ly, Lz))
    
    solid_nodes = np.array(solid_nodes)
    
    # Compute distance for each grid point
    # (Brute force - for large meshes, use KD-tree)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                point = np.array([mesh.x[i], mesh.y[j], mesh.z[k]])
                dists = np.linalg.norm(solid_nodes - point, axis=1)
                distance[i, j, k] = np.min(dists)
    
    return distance


class ThermoviscousSolver:
    """
    Solver for thermoviscous corrections.
    
    Applies boundary layer effects to acoustic fields.
    """
    
    def __init__(
        self,
        mesh: FEMMesh,
        fluid: FluidMaterial,
        frequency: float,
    ):
        """
        Initialize solver.
        
        Parameters
        ----------
        mesh : FEMMesh
            Finite element mesh.
        fluid : FluidMaterial
            Fluid material properties.
        frequency : float
            Acoustic frequency [Hz].
        """
        self.mesh = mesh
        self.fluid = fluid
        self.frequency = frequency
        
        # Compute parameters
        self.params = ThermoviscousParameters.from_fluid(fluid, frequency)
        
        # Cache distance field
        self._distance: Optional[np.ndarray] = None
    
    @property
    def distance_field(self) -> np.ndarray:
        """Distance to boundary, computed lazily."""
        if self._distance is None:
            self._distance = compute_distance_to_boundary(self.mesh)
        return self._distance
    
    def compute_correction(
        self,
        n_boundary_layers: float = 5.0,
    ) -> ThermoviscousCorrection:
        """
        Compute thermoviscous correction field.
        
        The correction accounts for enhanced dissipation in the
        viscous and thermal boundary layers.
        
        Parameters
        ----------
        n_boundary_layers : float
            Number of boundary layer thicknesses to include.
        
        Returns
        -------
        correction : ThermoviscousCorrection
            Correction field.
        """
        delta_v = self.params.delta_v
        delta_t = self.params.delta_t
        
        # Use larger of the two
        delta = max(delta_v, delta_t)
        
        # Distance to boundary
        d = self.distance_field
        
        # Boundary layer mask
        boundary_layer_mask = d < n_boundary_layers * delta
        
        # Correction factor
        # In boundary layer: enhanced dissipation
        # exp(-d/δ) decay of boundary layer effects
        correction = np.ones_like(d, dtype=complex)
        
        # Add imaginary part for dissipation in boundary layer
        # The exact form depends on the geometry (flat plate, cylinder, etc.)
        # Here we use exponential decay as a simple model
        in_layer = d < n_boundary_layers * delta
        correction[in_layer] = 1.0 - 1j * np.exp(-d[in_layer] / delta)
        
        return ThermoviscousCorrection(
            x=self.mesh.x,
            y=self.mesh.y,
            z=self.mesh.z,
            correction=correction,
            boundary_layer_mask=boundary_layer_mask,
            params=self.params,
            mesh=self.mesh,
        )
    
    def effective_wavenumber(self, distance: float = np.inf) -> complex:
        """
        Get effective wavenumber accounting for thermoviscous losses.
        
        Parameters
        ----------
        distance : float
            Distance from boundary [m]. If within boundary layer,
            enhanced dissipation is applied.
        
        Returns
        -------
        k : complex
            Effective wavenumber.
        """
        k_bulk = self.params.k
        
        delta = max(self.params.delta_v, self.params.delta_t)
        
        if distance < 5 * delta:
            # Enhanced dissipation in boundary layer
            enhancement = np.exp(-distance / delta)
            k_imag_enhanced = np.imag(k_bulk) * (1 + enhancement)
            return np.real(k_bulk) + 1j * k_imag_enhanced
        
        return k_bulk
    
    def apply_correction(
        self,
        pressure: np.ndarray,
    ) -> np.ndarray:
        """
        Apply thermoviscous correction to pressure field.
        
        Parameters
        ----------
        pressure : np.ndarray
            Acoustic pressure field.
        
        Returns
        -------
        corrected : np.ndarray
            Corrected pressure field.
        """
        corr = self.compute_correction()
        return pressure * corr.correction
    
    def summary(self) -> dict:
        """
        Get summary of thermoviscous parameters.
        
        Returns
        -------
        info : dict
            Dictionary with key parameters.
        """
        p = self.params
        return {
            'frequency_Hz': self.frequency,
            'omega_rad_s': p.omega,
            'delta_v_um': p.delta_v * 1e6,
            'delta_t_um': p.delta_t * 1e6,
            'Prandtl_number': p.Pr,
            'penetration_depth_m': p.penetration_depth(),
            'wavelength_mm': p.wavelength * 1e3,
            'k_real': np.real(p.k),
            'k_imag': np.imag(p.k),
        }
