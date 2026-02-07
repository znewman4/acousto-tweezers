"""
Material property database for FEniCSx acoustic simulation.

Provides temperature-dependent material properties for all domains:
- Fluids: water, air
- Solids: polystyrene, glass, PDMS (for dish materials)

All properties in SI units.

References:
- Kaye & Laby Tables of Physical & Chemical Constants
- Pierce, A.D. (1989) Acoustics: An Introduction
- Rubber materials: standard viscoelastic data

Author: Acousto-Tweezers Project
Date: January 2026
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np

from .domains import Domain


@dataclass
class FluidMaterial:
    """
    Acoustic properties of a fluid material.
    
    For the Helmholtz equation:
        ∇·(1/ρ ∇p) + ω²/(ρc²) p = 0
        
    The bulk modulus is K = ρc².
    """
    name: str
    density: float           # ρ [kg/m³]
    sound_speed: float       # c [m/s]
    dynamic_viscosity: float # η [Pa·s] (for streaming/thermoviscous)
    thermal_conductivity: float  # κ [W/(m·K)]
    specific_heat_cp: float  # Cp [J/(kg·K)]
    thermal_expansion: float # β [1/K]
    
    @property
    def bulk_modulus(self) -> float:
        """Bulk modulus K = ρc² [Pa]."""
        return self.density * self.sound_speed**2
    
    @property
    def kinematic_viscosity(self) -> float:
        """Kinematic viscosity ν = η/ρ [m²/s]."""
        return self.dynamic_viscosity / self.density
    
    @property
    def thermal_diffusivity(self) -> float:
        """Thermal diffusivity α = κ/(ρ·Cp) [m²/s]."""
        return self.thermal_conductivity / (self.density * self.specific_heat_cp)
    
    @property
    def prandtl_number(self) -> float:
        """Prandtl number Pr = ν/α."""
        return self.kinematic_viscosity / self.thermal_diffusivity
    
    def viscous_boundary_layer(self, omega: float) -> float:
        """
        Viscous boundary layer thickness.
        
        δ_v = √(2ν/ω)
        
        Parameters
        ----------
        omega : float
            Angular frequency [rad/s]
            
        Returns
        -------
        float
            Boundary layer thickness [m]
        """
        return np.sqrt(2 * self.kinematic_viscosity / omega)
    
    def thermal_boundary_layer(self, omega: float) -> float:
        """
        Thermal boundary layer thickness.
        
        δ_t = √(2α/ω)
        
        Parameters
        ----------
        omega : float
            Angular frequency [rad/s]
            
        Returns
        -------
        float
            Boundary layer thickness [m]
        """
        return np.sqrt(2 * self.thermal_diffusivity / omega)
    
    def acoustic_impedance(self) -> float:
        """Acoustic impedance Z = ρc [Pa·s/m]."""
        return self.density * self.sound_speed


@dataclass
class SolidMaterial:
    """
    Elastic properties of a solid material.
    
    For frequency-domain elasticity:
        ∇·σ(u) + ρ_s ω² u = 0
        σ = λ(∇·u)I + 2μ ε(u)
        
    With viscoelastic damping:
        E → E(1 + iη)
    """
    name: str
    density: float          # ρ [kg/m³]
    youngs_modulus: float   # E [Pa]
    poissons_ratio: float   # ν [-]
    loss_factor: float      # η [-] (viscoelastic damping)
    
    @property
    def complex_youngs_modulus(self) -> complex:
        """Complex Young's modulus with loss factor E(1 + iη)."""
        return self.youngs_modulus * (1 + 1j * self.loss_factor)
    
    @property
    def lame_lambda(self) -> float:
        """First Lamé parameter λ = Eν/((1+ν)(1-2ν)) [Pa]."""
        E = self.youngs_modulus
        nu = self.poissons_ratio
        return E * nu / ((1 + nu) * (1 - 2 * nu))
    
    @property
    def lame_mu(self) -> float:
        """Second Lamé parameter (shear modulus) μ = E/(2(1+ν)) [Pa]."""
        E = self.youngs_modulus
        nu = self.poissons_ratio
        return E / (2 * (1 + nu))
    
    @property
    def complex_lame_lambda(self) -> complex:
        """Complex first Lamé parameter with loss factor."""
        E = self.complex_youngs_modulus
        nu = self.poissons_ratio
        return E * nu / ((1 + nu) * (1 - 2 * nu))
    
    @property
    def complex_lame_mu(self) -> complex:
        """Complex shear modulus with loss factor."""
        E = self.complex_youngs_modulus
        nu = self.poissons_ratio
        return E / (2 * (1 + nu))
    
    @property
    def bulk_modulus(self) -> float:
        """Bulk modulus K = E/(3(1-2ν)) [Pa]."""
        return self.youngs_modulus / (3 * (1 - 2 * self.poissons_ratio))
    
    @property
    def longitudinal_wave_speed(self) -> float:
        """Longitudinal (P-wave) speed c_L = √((λ+2μ)/ρ) [m/s]."""
        return np.sqrt((self.lame_lambda + 2 * self.lame_mu) / self.density)
    
    @property 
    def shear_wave_speed(self) -> float:
        """Shear (S-wave) speed c_S = √(μ/ρ) [m/s]."""
        return np.sqrt(self.lame_mu / self.density)
    
    def acoustic_impedance(self) -> float:
        """Longitudinal acoustic impedance Z = ρ·c_L [Pa·s/m]."""
        return self.density * self.longitudinal_wave_speed


@dataclass
class ParticleMaterial:
    """
    Properties of spherical particles for radiation force calculation.
    
    For Gor'kov potential:
        U = (4π/3)a³ [f₁·⟨p²⟩/(2ρc²) - f₂·(3ρ/4)·⟨v²⟩]
        
    Contrast factors:
        f₁ = 1 - κ_p/κ_f  (monopole)
        f₂ = 2(ρ_p - ρ_f)/(2ρ_p + ρ_f)  (dipole)
    """
    name: str
    radius: float           # a [m]
    density: float          # ρ_p [kg/m³]
    compressibility: float  # κ_p [1/Pa]
    
    def monopole_contrast(self, fluid: FluidMaterial) -> float:
        """
        Monopole contrast factor f₁.
        
        f₁ = 1 - κ_p/κ_f = 1 - ρ_f·c_f²·κ_p
        
        f₁ > 0: particle more rigid than fluid (moves to pressure nodes)
        """
        kappa_f = 1.0 / fluid.bulk_modulus
        return 1.0 - self.compressibility / kappa_f
    
    def dipole_contrast(self, fluid: FluidMaterial) -> float:
        """
        Dipole contrast factor f₂.
        
        f₂ = 2(ρ_p - ρ_f)/(2ρ_p + ρ_f)
        
        f₂ > 0: particle denser than fluid (moves to velocity antinodes)
        """
        rho_p = self.density
        rho_f = fluid.density
        return 2 * (rho_p - rho_f) / (2 * rho_p + rho_f)
    
    def stokes_mobility(self, fluid: FluidMaterial) -> float:
        """
        Stokes mobility μ = 1/(6πηa) [m/(N·s)].
        
        Velocity v = μF for force F on particle.
        """
        return 1.0 / (6 * np.pi * fluid.dynamic_viscosity * self.radius)
    
    @property
    def volume(self) -> float:
        """Particle volume [m³]."""
        return (4.0 / 3.0) * np.pi * self.radius**3
    
    @property
    def mass(self) -> float:
        """Particle mass [kg]."""
        return self.density * self.volume


class MaterialDatabase:
    """
    Database of material properties.
    
    Provides temperature-dependent properties for all materials
    used in the acoustic tweezers simulation.
    """
    
    def __init__(self, temperature: float = 25.0):
        """
        Initialize material database.
        
        Parameters
        ----------
        temperature : float
            Temperature in Celsius
        """
        self.temperature = temperature
        self._init_materials()
    
    def _init_materials(self):
        """Initialize material property tables."""
        T = self.temperature
        
        # ===== FLUIDS =====
        
        # Water (temperature-dependent properties)
        # Sound speed: Marczak (1997) polynomial fit
        c_water = (1.402385e3 + 5.038813 * T - 5.799136e-2 * T**2 +
                   3.287156e-4 * T**3 - 1.398845e-6 * T**4 + 2.787860e-9 * T**5)
        # Density: IAPWS-95
        rho_water = 999.84 + 0.0675 * T - 0.00925 * T**2
        # Dynamic viscosity (Vogel equation)
        eta_water = 2.414e-5 * 10**(247.8 / (T + 273.15 - 140))
        
        self._water = FluidMaterial(
            name="water",
            density=rho_water,
            sound_speed=c_water,
            dynamic_viscosity=eta_water,
            thermal_conductivity=0.598,      # W/(m·K) at 20°C
            specific_heat_cp=4182.0,         # J/(kg·K)
            thermal_expansion=2.1e-4,        # 1/K
        )
        
        # Air (ideal gas approximation)
        T_K = T + 273.15
        rho_air = 1.293 * 273.15 / T_K  # Ideal gas
        c_air = 331.3 * np.sqrt(T_K / 273.15)  # Approx
        eta_air = 1.81e-5 * (T_K / 293.15)**0.7  # Sutherland approx
        
        self._air = FluidMaterial(
            name="air",
            density=rho_air,
            sound_speed=c_air,
            dynamic_viscosity=eta_air,
            thermal_conductivity=0.026,      # W/(m·K)
            specific_heat_cp=1005.0,         # J/(kg·K)
            thermal_expansion=1.0 / T_K,     # Ideal gas
        )
        
        # ===== SOLIDS =====
        
        # Polystyrene (typical Petri dish material)
        self._polystyrene = SolidMaterial(
            name="polystyrene",
            density=1050.0,                  # kg/m³
            youngs_modulus=3.0e9,            # 3 GPa
            poissons_ratio=0.34,
            loss_factor=0.01,                # η = 1% typical for PS
        )
        
        # Glass (borosilicate - glass dish)
        self._glass = SolidMaterial(
            name="glass",
            density=2230.0,                  # kg/m³
            youngs_modulus=63.0e9,           # 63 GPa
            poissons_ratio=0.20,
            loss_factor=0.001,               # Very low loss
        )
        
        # PDMS (soft lithography material)
        self._pdms = SolidMaterial(
            name="pdms",
            density=970.0,                   # kg/m³
            youngs_modulus=2.0e6,            # 2 MPa (soft!)
            poissons_ratio=0.49,             # Nearly incompressible
            loss_factor=0.05,                # Higher damping
        )
        
        # Acrylic (PMMA) - alternative dish material
        self._acrylic = SolidMaterial(
            name="acrylic",
            density=1180.0,
            youngs_modulus=3.2e9,
            poissons_ratio=0.37,
            loss_factor=0.02,
        )
        
        # ===== PARTICLES =====
        
        # Polystyrene microspheres (common for acoustic tweezers)
        self._ps_particle = ParticleMaterial(
            name="polystyrene_particle",
            radius=5.0e-6,                   # Default 5 μm
            density=1050.0,
            compressibility=2.4e-10,         # 1/Pa
        )
        
        # Silica microspheres
        self._silica_particle = ParticleMaterial(
            name="silica_particle",
            radius=5.0e-6,
            density=2000.0,
            compressibility=2.7e-11,         # Much stiffer
        )
    
    @property
    def water(self) -> FluidMaterial:
        """Water properties."""
        return self._water
    
    @property
    def air(self) -> FluidMaterial:
        """Air properties."""
        return self._air
    
    @property
    def polystyrene(self) -> SolidMaterial:
        """Polystyrene (dish material) properties."""
        return self._polystyrene
    
    @property
    def glass(self) -> SolidMaterial:
        """Glass properties."""
        return self._glass
    
    @property
    def pdms(self) -> SolidMaterial:
        """PDMS properties."""
        return self._pdms
    
    @property
    def acrylic(self) -> SolidMaterial:
        """Acrylic (PMMA) properties."""
        return self._acrylic
    
    @property
    def ps_particle(self) -> ParticleMaterial:
        """Polystyrene particle properties."""
        return self._ps_particle
    
    @property
    def silica_particle(self) -> ParticleMaterial:
        """Silica particle properties."""
        return self._silica_particle
    
    def get_fluid(self, name: str) -> FluidMaterial:
        """Get fluid material by name."""
        fluids = {
            "water": self._water,
            "air": self._air,
        }
        if name not in fluids:
            raise ValueError(f"Unknown fluid: {name}. Available: {list(fluids.keys())}")
        return fluids[name]
    
    def get_solid(self, name: str) -> SolidMaterial:
        """Get solid material by name."""
        solids = {
            "polystyrene": self._polystyrene,
            "glass": self._glass,
            "pdms": self._pdms,
            "acrylic": self._acrylic,
        }
        if name not in solids:
            raise ValueError(f"Unknown solid: {name}. Available: {list(solids.keys())}")
        return solids[name]
    
    def get_particle(self, name: str) -> ParticleMaterial:
        """Get particle material by name."""
        particles = {
            "polystyrene": self._ps_particle,
            "polystyrene_particle": self._ps_particle,
            "silica": self._silica_particle,
            "silica_particle": self._silica_particle,
        }
        if name not in particles:
            raise ValueError(f"Unknown particle: {name}. Available: {list(particles.keys())}")
        return particles[name]
    
    def get_material_for_domain(self, domain: Domain) -> FluidMaterial | SolidMaterial:
        """
        Get material properties for a domain.
        
        Parameters
        ----------
        domain : Domain
            The domain enum value
            
        Returns
        -------
        FluidMaterial or SolidMaterial
            Material properties for the domain
        """
        material_map = {
            Domain.WATER: self._water,
            Domain.AIR: self._air,
            Domain.BATH: self._water,  # Bath is also water
            Domain.PLATE: self._polystyrene,
            Domain.WALL: self._polystyrene,
            Domain.LENS: self._polystyrene,
            Domain.PML_WATER: self._water,
            Domain.PML_AIR: self._air,
            Domain.PML_BATH: self._water,
            Domain.PML_TOP: self._air,
            Domain.PML_BOTTOM: self._water,
            Domain.PML_LEFT: self._water,
            Domain.PML_RIGHT: self._water,
            Domain.TRANSDUCER: self._polystyrene,
        }
        if domain not in material_map:
            raise ValueError(f"No material defined for domain: {domain}")
        return material_map[domain]
    
    def summary(self) -> str:
        """Generate summary of material properties."""
        lines = [
            f"Material Database (T = {self.temperature}°C)",
            "=" * 50,
            "",
            "FLUIDS:",
            f"  Water:  ρ = {self.water.density:.1f} kg/m³, c = {self.water.sound_speed:.1f} m/s",
            f"          Z = {self.water.acoustic_impedance()/1e6:.2f} MRayl",
            f"  Air:    ρ = {self.air.density:.3f} kg/m³, c = {self.air.sound_speed:.1f} m/s",
            f"          Z = {self.air.acoustic_impedance():.1f} Rayl",
            "",
            "SOLIDS:",
            f"  Polystyrene: ρ = {self.polystyrene.density:.0f} kg/m³, E = {self.polystyrene.youngs_modulus/1e9:.1f} GPa",
            f"               c_L = {self.polystyrene.longitudinal_wave_speed:.0f} m/s, η = {self.polystyrene.loss_factor:.3f}",
            f"  Glass:       ρ = {self.glass.density:.0f} kg/m³, E = {self.glass.youngs_modulus/1e9:.1f} GPa",
            f"               c_L = {self.glass.longitudinal_wave_speed:.0f} m/s, η = {self.glass.loss_factor:.4f}",
            "",
            "PARTICLES:",
            f"  PS sphere: a = {self.ps_particle.radius*1e6:.1f} μm, ρ = {self.ps_particle.density:.0f} kg/m³",
            f"             f₁ = {self.ps_particle.monopole_contrast(self.water):.3f}, "
            f"f₂ = {self.ps_particle.dipole_contrast(self.water):.3f}",
        ]
        return "\n".join(lines)
