"""
Centralized material property database for FEM simulation.

This module provides physically correct material properties for all
domains in the acoustic tweezers simulation.

Materials are organized by type:
- Fluids: water, air
- Solids: glass (borosilicate), polystyrene, PDMS
- Particles: polystyrene beads, silica

All properties are temperature-dependent where data is available.
Complex properties (with loss factors) are provided for lossy materials.

References
----------
- Kaye & Laby Tables of Physical Constants
- CRC Handbook of Chemistry and Physics
- Settnes & Bruus (2012) for acoustic contrast factors
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np


@dataclass
class FluidMaterial:
    """
    Acoustic properties of a fluid material.
    
    The Helmholtz equation in fluids:
    
        ∇·(1/ρ ∇p) + ω²/(ρc²) p = 0
    
    where K = ρc² is the bulk modulus.
    
    Attributes
    ----------
    name : str
        Material identifier.
    rho : float
        Density [kg/m³].
    c : float
        Speed of sound [m/s].
    eta : float
        Dynamic viscosity [Pa·s].
    kappa : float
        Thermal conductivity [W/(m·K)].
    cp : float
        Specific heat capacity at constant pressure [J/(kg·K)].
    beta : float
        Thermal expansion coefficient [1/K].
    loss_factor : float
        Acoustic loss factor (imaginary part of K).
    """
    name: str
    rho: float          # Density [kg/m³]
    c: float            # Sound speed [m/s]
    eta: float          # Dynamic viscosity [Pa·s]
    kappa: float        # Thermal conductivity [W/(m·K)]
    cp: float           # Heat capacity [J/(kg·K)]
    beta: float         # Thermal expansion [1/K]
    loss_factor: float = 0.0  # Acoustic attenuation
    
    @property
    def K(self) -> float:
        """Bulk modulus K = ρc² [Pa]."""
        return self.rho * self.c**2
    
    @property
    def K_complex(self) -> complex:
        """Complex bulk modulus with loss K(1 + i·η)."""
        return self.K * (1 + 1j * self.loss_factor)
    
    @property
    def nu(self) -> float:
        """Kinematic viscosity ν = η/ρ [m²/s]."""
        return self.eta / self.rho
    
    @property
    def alpha(self) -> float:
        """Thermal diffusivity α = κ/(ρ·cp) [m²/s]."""
        return self.kappa / (self.rho * self.cp)
    
    @property
    def Pr(self) -> float:
        """Prandtl number Pr = ν/α."""
        return self.nu / self.alpha
    
    def boundary_layer_thickness_viscous(self, omega: float) -> float:
        """
        Viscous boundary layer thickness.
        
        δᵥ = √(2ν/ω)
        
        Parameters
        ----------
        omega : float
            Angular frequency [rad/s].
        
        Returns
        -------
        delta_v : float
            Viscous boundary layer thickness [m].
        """
        return np.sqrt(2 * self.nu / omega)
    
    def boundary_layer_thickness_thermal(self, omega: float) -> float:
        """
        Thermal boundary layer thickness.
        
        δₜ = √(2α/ω)
        """
        return np.sqrt(2 * self.alpha / omega)
    
    def wavelength(self, frequency: float) -> float:
        """Acoustic wavelength λ = c/f [m]."""
        return self.c / frequency
    
    def wavenumber(self, frequency: float) -> float:
        """Acoustic wavenumber k = ω/c = 2πf/c [1/m]."""
        return 2 * np.pi * frequency / self.c
    
    def acoustic_impedance(self) -> float:
        """Acoustic impedance Z = ρc [kg/(m²·s)]."""
        return self.rho * self.c
    
    def compressibility(self) -> float:
        """Adiabatic compressibility κ = 1/K [1/Pa]."""
        return 1.0 / self.K


@dataclass
class SolidMaterial:
    """
    Elastic properties of a solid material.
    
    The frequency-domain elasticity equation:
    
        ∇·σ(u) + ρω²u = 0
    
    with viscoelastic damping E → E(1 + iη).
    
    Attributes
    ----------
    name : str
        Material identifier.
    rho : float
        Density [kg/m³].
    E : float
        Young's modulus [Pa].
    nu : float
        Poisson's ratio [-].
    loss_eta : float
        Loss factor for E (imaginary part).
    """
    name: str
    rho: float          # Density [kg/m³]
    E: float            # Young's modulus [Pa]
    nu: float           # Poisson's ratio [-]
    loss_eta: float = 0.0  # Damping loss factor
    
    @property
    def E_complex(self) -> complex:
        """Complex Young's modulus with damping."""
        return self.E * (1 + 1j * self.loss_eta)
    
    @property
    def G(self) -> float:
        """Shear modulus G = E/(2(1+ν)) [Pa]."""
        return self.E / (2 * (1 + self.nu))
    
    @property
    def G_complex(self) -> complex:
        """Complex shear modulus with damping."""
        return self.E_complex / (2 * (1 + self.nu))
    
    @property
    def lambda_lame(self) -> float:
        """First Lamé parameter λ = Eν/((1+ν)(1-2ν)) [Pa]."""
        return self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
    
    @property
    def lambda_lame_complex(self) -> complex:
        """Complex Lamé parameter with damping."""
        return self.E_complex * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
    
    @property
    def mu_lame(self) -> float:
        """Second Lamé parameter μ = G [Pa]."""
        return self.G
    
    @property
    def mu_lame_complex(self) -> complex:
        """Complex Lamé parameter with damping."""
        return self.G_complex
    
    @property
    def c_longitudinal(self) -> float:
        """Longitudinal (P-wave) speed cₗ = √((λ+2μ)/ρ) [m/s]."""
        return np.sqrt((self.lambda_lame + 2 * self.G) / self.rho)
    
    @property
    def c_shear(self) -> float:
        """Shear (S-wave) speed cₛ = √(μ/ρ) [m/s]."""
        return np.sqrt(self.G / self.rho)
    
    @property
    def c_plate(self) -> float:
        """Plate wave speed (thin plate approximation) [m/s]."""
        # Approximate: cp ≈ √(E/(ρ(1-ν²)))
        return np.sqrt(self.E / (self.rho * (1 - self.nu**2)))
    
    def acoustic_impedance(self) -> float:
        """Acoustic impedance Z = ρ·cₗ [kg/(m²·s)]."""
        return self.rho * self.c_longitudinal


@dataclass
class ParticleMaterial:
    """
    Properties of a spherical particle for radiation force calculation.
    
    The Gor'kov potential:
    
        U = V [f₁ (κ/3) ⟨p²⟩ - f₂ (ρ/2) ⟨v²⟩]
    
    where f₁ = 1 - κₚ/κ and f₂ = 2(ρₚ-ρ)/(2ρₚ+ρ).
    
    Attributes
    ----------
    name : str
        Material identifier.
    rho : float
        Density [kg/m³].
    kappa : float
        Compressibility [1/Pa].
    radius : float
        Particle radius [m].
    """
    name: str
    rho: float          # Density [kg/m³]
    kappa: float        # Compressibility [1/Pa]
    radius: float       # Particle radius [m]
    
    @property
    def volume(self) -> float:
        """Particle volume V = 4πr³/3 [m³]."""
        return (4/3) * np.pi * self.radius**3
    
    @property
    def mass(self) -> float:
        """Particle mass m = ρV [kg]."""
        return self.rho * self.volume
    
    def monopole_coefficient(self, fluid: FluidMaterial) -> float:
        """
        Monopole scattering coefficient f₁.
        
        f₁ = 1 - κₚ/κ = 1 - Kf/Kp
        
        where K is the bulk modulus.
        """
        return 1 - fluid.compressibility() / self.kappa
    
    def dipole_coefficient(self, fluid: FluidMaterial) -> float:
        """
        Dipole scattering coefficient f₂.
        
        f₂ = 2(ρₚ - ρf) / (2ρₚ + ρf)
        """
        return 2 * (self.rho - fluid.rho) / (2 * self.rho + fluid.rho)
    
    def acoustic_contrast_factor(self, fluid: FluidMaterial) -> float:
        """
        Acoustic contrast factor Φ.
        
        Φ = f₁/3 + f₂/2
        
        Positive Φ → particle moves to pressure node.
        Negative Φ → particle moves to pressure antinode.
        """
        f1 = self.monopole_coefficient(fluid)
        f2 = self.dipole_coefficient(fluid)
        return f1 / 3 + f2 / 2
    
    def mobility(self, fluid: FluidMaterial) -> float:
        """
        Stokes mobility μ = 1/(6πηa) [m/(N·s)].
        
        Relates force to velocity in overdamped limit: v = μF
        """
        return 1.0 / (6 * np.pi * fluid.eta * self.radius)
    
    def stokes_drag(self, fluid: FluidMaterial, velocity: float) -> float:
        """
        Stokes drag force Fd = 6πηav [N].
        """
        return 6 * np.pi * fluid.eta * self.radius * velocity


class MaterialDatabase:
    """
    Central database of material properties.
    
    All materials are accessed through this class to ensure consistency.
    
    Example
    -------
    >>> db = MaterialDatabase(temperature=25.0)
    >>> water = db.water
    >>> glass = db.borosilicate_glass
    """
    
    def __init__(self, temperature: float = 25.0):
        """
        Initialize database at given temperature.
        
        Parameters
        ----------
        temperature : float
            Temperature in Celsius.
        """
        self.temperature = temperature
        self._T = temperature  # Alias
    
    # =========================================================================
    # FLUID MATERIALS
    # =========================================================================
    
    @property
    def water(self) -> FluidMaterial:
        """
        Water properties at specified temperature.
        
        Sources:
        - Kaye & Laby Tables
        - Marczak (1997) for c(T) correlation
        """
        T = self._T
        
        # Density: ρ(T) polynomial fit [kg/m³]
        # Valid 0-100°C
        rho = (999.842594 + 6.793952e-2 * T 
               - 9.095290e-3 * T**2 + 1.001685e-4 * T**3
               - 1.120083e-6 * T**4 + 6.536332e-9 * T**5)
        
        # Sound speed: Marczak (1997) correlation [m/s]
        # Valid 0-95°C
        c = (1.402385e3 + 5.038813 * T - 5.799136e-2 * T**2
             + 3.287156e-4 * T**3 - 1.398845e-6 * T**4
             + 2.787860e-9 * T**5)
        
        # Dynamic viscosity [Pa·s]
        # Vogel equation
        eta = 2.414e-5 * 10**(247.8 / (T + 273.15 - 140))
        
        # Thermal conductivity [W/(m·K)]
        kappa = 0.6065 * (1 + 0.00175 * T)
        
        # Heat capacity [J/(kg·K)]
        cp = 4182.0 - 0.5 * T  # Approximate
        
        # Thermal expansion [1/K]
        beta = 2.1e-4 + 4e-6 * T
        
        return FluidMaterial(
            name="water",
            rho=rho,
            c=c,
            eta=eta,
            kappa=kappa,
            cp=cp,
            beta=beta,
            loss_factor=0.0,  # Negligible at MHz frequencies
        )
    
    @property
    def air(self) -> FluidMaterial:
        """
        Air properties at specified temperature (1 atm).
        
        Sources:
        - Ideal gas relations
        - Sutherland's formula for viscosity
        """
        T = self._T
        T_K = T + 273.15  # Kelvin
        
        # Density: ideal gas [kg/m³]
        # ρ = P/(R·T), P = 101325 Pa, R = 287 J/(kg·K)
        rho = 101325 / (287 * T_K)
        
        # Sound speed [m/s]
        # c = √(γRT), γ = 1.4, R = 287
        c = np.sqrt(1.4 * 287 * T_K)
        
        # Dynamic viscosity: Sutherland's formula [Pa·s]
        eta = 1.458e-6 * T_K**1.5 / (T_K + 110.4)
        
        # Thermal conductivity [W/(m·K)]
        kappa = 0.0241 * (T_K / 273)**0.84
        
        # Heat capacity [J/(kg·K)]
        cp = 1005.0  # Approximately constant
        
        # Thermal expansion [1/K]
        beta = 1 / T_K  # Ideal gas
        
        return FluidMaterial(
            name="air",
            rho=rho,
            c=c,
            eta=eta,
            kappa=kappa,
            cp=cp,
            beta=beta,
            loss_factor=0.0,
        )
    
    # =========================================================================
    # SOLID MATERIALS
    # =========================================================================
    
    @property
    def borosilicate_glass(self) -> SolidMaterial:
        """
        Borosilicate glass (Pyrex/Borofloat).
        
        Typical Petri dish material.
        """
        return SolidMaterial(
            name="borosilicate_glass",
            rho=2230.0,          # kg/m³
            E=63e9,              # Pa (63 GPa)
            nu=0.2,              # Poisson's ratio
            loss_eta=0.001,      # Low damping
        )
    
    @property
    def polystyrene(self) -> SolidMaterial:
        """
        Polystyrene (solid).
        
        Common dish material and particle material.
        """
        return SolidMaterial(
            name="polystyrene",
            rho=1050.0,          # kg/m³
            E=3.0e9,             # Pa (3 GPa)
            nu=0.34,
            loss_eta=0.003,
        )
    
    @property
    def pmma(self) -> SolidMaterial:
        """PMMA (acrylic, Plexiglas)."""
        return SolidMaterial(
            name="pmma",
            rho=1180.0,
            E=3.3e9,
            nu=0.37,
            loss_eta=0.002,
        )
    
    @property
    def pdms(self) -> SolidMaterial:
        """PDMS (Sylgard 184, common microfluidic material)."""
        return SolidMaterial(
            name="pdms",
            rho=970.0,
            E=2.0e6,             # Very soft (~2 MPa)
            nu=0.49,             # Nearly incompressible
            loss_eta=0.05,       # Significant damping
        )
    
    @property
    def steel(self) -> SolidMaterial:
        """Stainless steel 316."""
        return SolidMaterial(
            name="steel",
            rho=8000.0,
            E=200e9,
            nu=0.3,
            loss_eta=0.0001,
        )
    
    @property
    def aluminum(self) -> SolidMaterial:
        """Aluminum 6061-T6."""
        return SolidMaterial(
            name="aluminum",
            rho=2700.0,
            E=69e9,
            nu=0.33,
            loss_eta=0.0001,
        )
    
    # =========================================================================
    # PARTICLE MATERIALS
    # =========================================================================
    
    @property
    def polystyrene_bead(self) -> ParticleMaterial:
        """
        Polystyrene microsphere (default 5 μm radius).
        
        Common acoustic trapping target.
        """
        return ParticleMaterial(
            name="polystyrene_bead",
            rho=1050.0,
            kappa=2.4e-10,       # 1/Pa
            radius=5.0e-6,       # 5 μm
        )
    
    @property
    def silica_bead(self) -> ParticleMaterial:
        """Silica (glass) microsphere."""
        return ParticleMaterial(
            name="silica_bead",
            rho=2000.0,
            kappa=2.7e-11,       # Much stiffer
            radius=5.0e-6,
        )
    
    def polystyrene_bead_custom(self, radius: float) -> ParticleMaterial:
        """Create polystyrene bead with custom radius."""
        return ParticleMaterial(
            name=f"polystyrene_bead_{radius*1e6:.1f}um",
            rho=1050.0,
            kappa=2.4e-10,
            radius=radius,
        )
    
    # =========================================================================
    # LOOKUP METHODS
    # =========================================================================
    
    def get_fluid(self, name: str) -> FluidMaterial:
        """Get fluid material by name."""
        materials = {
            "water": self.water,
            "air": self.air,
        }
        if name not in materials:
            raise KeyError(f"Unknown fluid material: {name}")
        return materials[name]
    
    def get_solid(self, name: str) -> SolidMaterial:
        """Get solid material by name."""
        materials = {
            "borosilicate_glass": self.borosilicate_glass,
            "glass": self.borosilicate_glass,
            "polystyrene": self.polystyrene,
            "pmma": self.pmma,
            "pdms": self.pdms,
            "steel": self.steel,
            "aluminum": self.aluminum,
        }
        if name not in materials:
            raise KeyError(f"Unknown solid material: {name}")
        return materials[name]
    
    def get_particle(self, name: str) -> ParticleMaterial:
        """Get particle material by name."""
        materials = {
            "polystyrene": self.polystyrene_bead,
            "polystyrene_bead": self.polystyrene_bead,
            "silica": self.silica_bead,
            "silica_bead": self.silica_bead,
        }
        if name not in materials:
            raise KeyError(f"Unknown particle material: {name}")
        return materials[name]
    
    def summary(self) -> str:
        """Return summary of all materials."""
        lines = [
            f"Material Database (T = {self.temperature}°C)",
            "=" * 50,
            "",
            "FLUIDS:",
            f"  water: ρ={self.water.rho:.1f} kg/m³, c={self.water.c:.1f} m/s",
            f"  air:   ρ={self.air.rho:.3f} kg/m³, c={self.air.c:.1f} m/s",
            "",
            "SOLIDS:",
            f"  glass: ρ={self.borosilicate_glass.rho:.0f} kg/m³, "
            f"E={self.borosilicate_glass.E/1e9:.1f} GPa, "
            f"cL={self.borosilicate_glass.c_longitudinal:.0f} m/s",
            "",
            "PARTICLES:",
            f"  PS bead: ρ={self.polystyrene_bead.rho:.0f} kg/m³, "
            f"r={self.polystyrene_bead.radius*1e6:.1f} μm",
        ]
        return "\n".join(lines)
