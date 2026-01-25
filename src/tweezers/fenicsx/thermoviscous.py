"""
Thermoviscous acoustics for FEniCSx.

Implements viscous and thermal boundary layer corrections per MASTER BRIEF:

    δ_v = √(2ν/ω)     (viscous boundary layer thickness)
    δ_t = √(2α/ω)     (thermal boundary layer thickness)

These boundary layers cause additional acoustic losses near solid walls.
The corrections are included when PhysicsLevel >= THERMOVISCOUS.

The losses arise from physics (viscosity, thermal conductivity) and are
NOT tuning constants.

Author: Acousto-Tweezers Project
Date: January 2026
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np

from dolfinx import fem, mesh as dmesh
from dolfinx.fem import Function

from .config import FEMConfig
from .materials import MaterialDatabase, FluidMaterial


@dataclass
class ThermoviscousCorrection:
    """
    Thermoviscous boundary layer correction data.
    """
    # Boundary layer thicknesses
    delta_v: float  # Viscous boundary layer [m]
    delta_t: float  # Thermal boundary layer [m]
    
    # Loss contributions
    viscous_loss: float   # Power loss from viscosity [W/m²]
    thermal_loss: float   # Power loss from thermal conduction [W/m²]
    
    # Wall area
    wall_area: float  # Total wall area [m²]
    
    @property
    def total_loss(self) -> float:
        """Total power loss [W]."""
        return (self.viscous_loss + self.thermal_loss) * self.wall_area
    
    @property
    def loss_ratio(self) -> float:
        """Ratio of viscous to thermal loss."""
        if self.thermal_loss > 0:
            return self.viscous_loss / self.thermal_loss
        return np.inf
    
    def summary(self) -> str:
        return (
            f"Thermoviscous Corrections:\n"
            f"  δ_v = {self.delta_v*1e6:.2f} μm\n"
            f"  δ_t = {self.delta_t*1e6:.2f} μm\n"
            f"  Viscous loss: {self.viscous_loss:.2e} W/m²\n"
            f"  Thermal loss: {self.thermal_loss:.2e} W/m²\n"
            f"  Total loss: {self.total_loss:.2e} W"
        )


class ThermoviscousSolver:
    """
    Handles thermoviscous boundary layer corrections.
    
    Two approaches are available:
    1. Effective boundary impedance modification
    2. Full thermoviscous acoustic equations (complex, more accurate)
    
    This implementation uses approach 1 for efficiency.
    """
    
    def __init__(self, config: FEMConfig, materials: MaterialDatabase):
        """
        Initialize thermoviscous solver.
        
        Parameters
        ----------
        config : FEMConfig
            Simulation configuration
        materials : MaterialDatabase
            Material property database
        """
        self.config = config
        self.materials = materials
        self.omega = config.physics.omega
        
        # Compute boundary layer thicknesses
        self._compute_boundary_layers()
        
    def _compute_boundary_layers(self):
        """Compute viscous and thermal boundary layer thicknesses."""
        water = self.materials.water
        air = self.materials.air
        
        omega = self.omega
        
        # Water boundary layers
        self.delta_v_water = water.viscous_boundary_layer(omega)
        self.delta_t_water = water.thermal_boundary_layer(omega)
        
        # Air boundary layers
        self.delta_v_air = air.viscous_boundary_layer(omega)
        self.delta_t_air = air.thermal_boundary_layer(omega)
        
        # Report
        print(f"Thermoviscous boundary layers at f = {self.config.physics.frequency/1e6:.1f} MHz:")
        print(f"  Water: δ_v = {self.delta_v_water*1e6:.2f} μm, δ_t = {self.delta_t_water*1e6:.2f} μm")
        print(f"  Air:   δ_v = {self.delta_v_air*1e6:.2f} μm, δ_t = {self.delta_t_air*1e6:.2f} μm")
        
    def compute_wall_impedance(self, fluid: FluidMaterial) -> complex:
        """
        Compute effective wall impedance including thermoviscous losses.
        
        The wall impedance with boundary layer losses is:
        
        Z_wall = Z_0 * (1 + (1-i)/2 * (k*δ_v + (γ-1)*k*δ_t))
        
        where Z_0 = ρc is the characteristic impedance.
        
        Parameters
        ----------
        fluid : FluidMaterial
            Fluid properties
            
        Returns
        -------
        complex
            Effective wall impedance [Pa·s/m]
        """
        omega = self.omega
        
        # Boundary layer thicknesses
        delta_v = fluid.viscous_boundary_layer(omega)
        delta_t = fluid.thermal_boundary_layer(omega)
        
        # Wavenumber
        k = omega / fluid.sound_speed
        
        # Specific heat ratio (approximate for water)
        gamma = fluid.specific_heat_cp / (fluid.specific_heat_cp - 
                fluid.thermal_expansion**2 * fluid.temperature_not_implemented() / 
                (fluid.density * fluid.compressibility_not_implemented()))
        # Simplified: use gamma ≈ 1.0 for water (nearly incompressible)
        gamma = 1.0
        
        # Base impedance
        Z_0 = fluid.acoustic_impedance()
        
        # Thermoviscous correction factor
        factor = 1.0 + (1 - 1j) / 2 * (k * delta_v + (gamma - 1) * k * delta_t)
        
        return Z_0 * factor
    
    def compute_loss_coefficient(self, fluid: FluidMaterial) -> complex:
        """
        Compute loss coefficient for weak form modification.
        
        Adds imaginary part to wavenumber: k → k - i α
        
        The attenuation coefficient is:
        α = (ω/2c) * (δ_v + (γ-1)*δ_t) / L_char
        
        where L_char is a characteristic length (e.g., pipe radius).
        
        Parameters
        ----------
        fluid : FluidMaterial
            Fluid properties
            
        Returns
        -------
        complex
            Complex wavenumber modification
        """
        omega = self.omega
        c = fluid.sound_speed
        
        delta_v = fluid.viscous_boundary_layer(omega)
        delta_t = fluid.thermal_boundary_layer(omega)
        
        # Simplified attenuation (per unit length)
        # This is approximate - full treatment requires surface integrals
        alpha = omega / (2 * c) * (delta_v + delta_t)
        
        return alpha
    
    def compute_correction(self, 
                           mesh: dmesh.Mesh,
                           facet_tags: dmesh.MeshTags,
                           wall_tags: list) -> ThermoviscousCorrection:
        """
        Compute thermoviscous correction for given walls.
        
        Parameters
        ----------
        mesh : Mesh
            The mesh
        facet_tags : MeshTags
            Facet tags
        wall_tags : list
            List of facet tags for walls
            
        Returns
        -------
        ThermoviscousCorrection
            Correction data
        """
        water = self.materials.water
        omega = self.omega
        
        delta_v = water.viscous_boundary_layer(omega)
        delta_t = water.thermal_boundary_layer(omega)
        
        # Compute wall area (sum of tagged facet areas)
        # This requires mesh facet computation
        # Simplified: estimate from geometry
        geo = self.config.geometry
        wall_area = 2 * np.pi * geo.dish_inner_radius * geo.water_depth
        wall_area += np.pi * geo.dish_inner_radius**2  # Bottom
        
        # Estimate losses (simplified)
        # Full computation would integrate over boundary
        rho = water.density
        c = water.sound_speed
        eta = water.dynamic_viscosity
        
        # Viscous loss per unit area: ~ η ω / (2 δ_v)
        viscous_loss = eta * omega / (2 * delta_v)
        
        # Thermal loss (usually smaller for water)
        kappa = water.thermal_conductivity
        thermal_loss = kappa * omega / (2 * c**2 * delta_t)
        
        return ThermoviscousCorrection(
            delta_v=delta_v,
            delta_t=delta_t,
            viscous_loss=viscous_loss,
            thermal_loss=thermal_loss,
            wall_area=wall_area,
        )
    
    def is_significant(self, fluid: FluidMaterial, 
                       length_scale: float) -> bool:
        """
        Check if thermoviscous effects are significant.
        
        Effects are significant when boundary layer thickness is
        comparable to smallest geometric feature or mesh size.
        
        Parameters
        ----------
        fluid : FluidMaterial
            Fluid properties
        length_scale : float
            Characteristic geometric length [m]
            
        Returns
        -------
        bool
            True if thermoviscous effects should be included
        """
        delta_v = fluid.viscous_boundary_layer(self.omega)
        
        # Significant if δ_v > 0.01 * L
        return delta_v > 0.01 * length_scale


def estimate_boundary_layer_resolution(fluid: FluidMaterial,
                                        omega: float,
                                        mesh_size: float) -> Dict[str, float]:
    """
    Estimate mesh resolution requirements for boundary layer resolution.
    
    Parameters
    ----------
    fluid : FluidMaterial
        Fluid properties
    omega : float
        Angular frequency
    mesh_size : float
        Current mesh element size
        
    Returns
    -------
    Dict[str, float]
        Resolution metrics
    """
    delta_v = fluid.viscous_boundary_layer(omega)
    delta_t = fluid.thermal_boundary_layer(omega)
    
    # Elements across boundary layer
    n_v = delta_v / mesh_size
    n_t = delta_t / mesh_size
    
    # Recommended: at least 3-5 elements across boundary layer
    recommended_size = min(delta_v, delta_t) / 5
    
    return {
        'delta_v': delta_v,
        'delta_t': delta_t,
        'mesh_size': mesh_size,
        'elements_in_viscous_bl': n_v,
        'elements_in_thermal_bl': n_t,
        'recommended_mesh_size': recommended_size,
        'bl_resolved': n_v >= 3 and n_t >= 3,
    }
