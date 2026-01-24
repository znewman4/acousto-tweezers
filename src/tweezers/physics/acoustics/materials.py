"""
Material properties for acoustic simulations.

Defines fluid and solid material properties including:
- Density, sound speed, bulk modulus
- Viscosity and thermal properties for thermoviscous effects
- Complex moduli for lossy materials
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict
import numpy as np


@dataclass
class FluidMaterial:
    """
    Acoustic properties of a fluid medium.
    
    Parameters
    ----------
    name : str
        Material name for identification.
    rho : float
        Density [kg/m³].
    c : float
        Speed of sound [m/s].
    eta : float
        Dynamic viscosity [Pa·s].
    kappa_T : float, optional
        Isothermal compressibility [1/Pa]. If None, computed from rho and c.
    alpha : float, optional
        Thermal diffusivity [m²/s]. Required for thermoviscous effects.
    gamma : float
        Ratio of specific heats cp/cv.
    Pr : float
        Prandtl number (eta*cp)/(k_thermal).
    loss_factor : float
        Additional bulk loss factor for phenomenological damping.
    """
    name: str
    rho: float          # Density [kg/m³]
    c: float            # Sound speed [m/s]
    eta: float          # Dynamic viscosity [Pa·s]
    kappa_T: Optional[float] = None    # Isothermal compressibility [1/Pa]
    alpha: Optional[float] = None      # Thermal diffusivity [m²/s]
    gamma: float = 1.0                 # Ratio of specific heats
    Pr: float = 7.0                    # Prandtl number
    loss_factor: float = 0.0           # Bulk loss factor
    
    def __post_init__(self):
        if self.kappa_T is None:
            # Adiabatic compressibility (inviscid limit)
            self.kappa_T = 1.0 / (self.rho * self.c**2)
    
    @property
    def K(self) -> float:
        """Bulk modulus [Pa]."""
        return self.rho * self.c**2
    
    @property
    def Z(self) -> float:
        """Acoustic impedance [kg/(m²·s)]."""
        return self.rho * self.c
    
    def wavenumber(self, omega: float) -> complex:
        """
        Compute complex wavenumber including losses.
        
        k = ω/c * (1 + i*η/2) for small losses
        """
        k0 = omega / self.c
        if self.loss_factor > 0:
            return k0 * np.sqrt(1 + 1j * self.loss_factor)
        return complex(k0, 0.0)
    
    def viscous_boundary_layer(self, omega: float) -> float:
        """
        Viscous boundary layer thickness δv = sqrt(2ν/ω).
        """
        nu = self.eta / self.rho  # Kinematic viscosity
        return np.sqrt(2 * nu / omega)
    
    def thermal_boundary_layer(self, omega: float) -> float:
        """
        Thermal boundary layer thickness δt = sqrt(2α/ω).
        """
        if self.alpha is None:
            # Estimate from Prandtl number: α = ν/Pr
            nu = self.eta / self.rho
            alpha = nu / self.Pr
        else:
            alpha = self.alpha
        return np.sqrt(2 * alpha / omega)


@dataclass
class SolidMaterial:
    """
    Elastic properties of a solid material.
    
    Parameters
    ----------
    name : str
        Material name.
    rho : float
        Density [kg/m³].
    E : float
        Young's modulus [Pa].
    nu : float
        Poisson's ratio [-].
    loss_eta : float
        Loss factor η for viscoelastic damping. 
        Complex modulus: E_eff = E(1 + i*η)
    """
    name: str
    rho: float      # Density [kg/m³]
    E: float        # Young's modulus [Pa]
    nu: float       # Poisson's ratio [-]
    loss_eta: float = 0.0  # Loss factor for damping
    
    @property
    def E_complex(self) -> complex:
        """Complex Young's modulus with damping."""
        return self.E * (1 + 1j * self.loss_eta)
    
    @property
    def lambda_lame(self) -> float:
        """First Lamé parameter λ [Pa]."""
        return self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
    
    @property
    def mu_lame(self) -> float:
        """Second Lamé parameter (shear modulus) μ [Pa]."""
        return self.E / (2 * (1 + self.nu))
    
    @property
    def lambda_complex(self) -> complex:
        """Complex first Lamé parameter with damping."""
        E = self.E_complex
        return E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
    
    @property
    def mu_complex(self) -> complex:
        """Complex shear modulus with damping."""
        E = self.E_complex
        return E / (2 * (1 + self.nu))
    
    @property
    def c_longitudinal(self) -> float:
        """Longitudinal wave speed [m/s]."""
        lam = self.lambda_lame
        mu = self.mu_lame
        return np.sqrt((lam + 2 * mu) / self.rho)
    
    @property
    def c_shear(self) -> float:
        """Shear wave speed [m/s]."""
        return np.sqrt(self.mu_lame / self.rho)
    
    @property
    def c_plate(self) -> float:
        """Plate wave speed (approximate) [m/s]."""
        return np.sqrt(self.E / (self.rho * (1 - self.nu**2)))
    
    @property
    def Z_longitudinal(self) -> float:
        """Longitudinal acoustic impedance [kg/(m²·s)]."""
        return self.rho * self.c_longitudinal


class MaterialDatabase:
    """
    Database of common materials for acoustic simulations.
    
    Provides factory methods for:
    - Water (various temperatures)
    - Air (standard conditions)
    - Common plastics (polystyrene, PMMA, etc.)
    - Glass
    """
    
    @staticmethod
    def water(T_celsius: float = 25.0) -> FluidMaterial:
        """
        Water properties at given temperature.
        
        Uses empirical fits for T in [0, 100]°C.
        """
        T = T_celsius
        
        # Sound speed (m/s) - Del Grosso fit
        c = (1402.7 + 5.0 * T - 0.0559 * T**2 + 0.000221 * T**3)
        
        # Density (kg/m³) - simple fit
        rho = 1000.0 * (1 - (T - 4)**2 / 119000)
        
        # Dynamic viscosity (Pa·s)
        eta = 0.001002 * 10**((1.3272 * (20 - T) - 0.001053 * (T - 20)**2) / (T + 105))
        
        # Thermal diffusivity (m²/s)
        alpha = 1.43e-7  # Approximate at 25°C
        
        return FluidMaterial(
            name=f"water_{T_celsius}C",
            rho=rho,
            c=c,
            eta=eta,
            alpha=alpha,
            gamma=1.0,  # Approximately 1 for liquids
            Pr=7.0,
            loss_factor=1e-6,  # Very small bulk loss
        )
    
    @staticmethod
    def air(T_celsius: float = 20.0, P_atm: float = 1.0) -> FluidMaterial:
        """
        Air properties at given temperature and pressure.
        """
        T_K = T_celsius + 273.15
        
        # Sound speed (m/s) - ideal gas approximation
        gamma = 1.4
        R = 287.05  # Specific gas constant for air
        c = np.sqrt(gamma * R * T_K)
        
        # Density from ideal gas law
        P = P_atm * 101325  # Pa
        rho = P / (R * T_K)
        
        # Dynamic viscosity - Sutherland's law
        eta = 1.716e-5 * (T_K / 273.15)**1.5 * (273.15 + 110.4) / (T_K + 110.4)
        
        # Thermal diffusivity (m²/s)
        alpha = 2.2e-5  # Approximate at 20°C
        
        return FluidMaterial(
            name=f"air_{T_celsius}C",
            rho=rho,
            c=c,
            eta=eta,
            alpha=alpha,
            gamma=gamma,
            Pr=0.71,
            loss_factor=1e-4,  # Small bulk viscosity contribution
        )
    
    @staticmethod
    def polystyrene() -> SolidMaterial:
        """Polystyrene (typical petri dish material)."""
        return SolidMaterial(
            name="polystyrene",
            rho=1050.0,     # kg/m³
            E=3.0e9,        # Pa (3 GPa)
            nu=0.34,        # Poisson's ratio
            loss_eta=0.01,  # ~1% loss factor
        )
    
    @staticmethod
    def pmma() -> SolidMaterial:
        """PMMA (acrylic, Plexiglas)."""
        return SolidMaterial(
            name="pmma",
            rho=1180.0,     # kg/m³
            E=3.3e9,        # Pa
            nu=0.37,
            loss_eta=0.02,
        )
    
    @staticmethod
    def glass() -> SolidMaterial:
        """Borosilicate glass."""
        return SolidMaterial(
            name="glass",
            rho=2230.0,     # kg/m³
            E=63e9,         # Pa
            nu=0.20,
            loss_eta=0.001, # Very low loss
        )
    
    @staticmethod
    def soft_pdms() -> SolidMaterial:
        """Soft PDMS (common microfluidics material)."""
        return SolidMaterial(
            name="pdms_soft",
            rho=970.0,
            E=1.0e6,        # 1 MPa (very soft)
            nu=0.49,        # Nearly incompressible
            loss_eta=0.05,
        )


@dataclass
class ParticleMaterial:
    """
    Properties of a spherical particle for radiation force calculations.
    
    Parameters
    ----------
    name : str
        Material name.
    rho : float
        Density [kg/m³].
    c : float
        Sound speed in particle material [m/s].
    """
    name: str
    rho: float  # Density [kg/m³]
    c: float    # Sound speed [m/s]
    
    @property
    def kappa(self) -> float:
        """Compressibility [1/Pa]."""
        return 1.0 / (self.rho * self.c**2)
    
    def contrast_factors(self, fluid: FluidMaterial) -> tuple[float, float]:
        """
        Compute acoustic contrast factors f1 and f2 for Gor'kov potential.
        
        f1 = 1 - κp/κ0 (compressibility contrast)
        f2 = 2(ρp - ρ0)/(2ρp + ρ0) (density contrast)
        """
        kappa_0 = fluid.kappa_T
        kappa_p = self.kappa
        rho_0 = fluid.rho
        rho_p = self.rho
        
        f1 = 1.0 - kappa_p / kappa_0
        f2 = 2.0 * (rho_p - rho_0) / (2.0 * rho_p + rho_0)
        
        return f1, f2


class ParticleDatabase:
    """Common particle materials."""
    
    @staticmethod
    def polystyrene_bead() -> ParticleMaterial:
        """Polystyrene microsphere."""
        return ParticleMaterial(
            name="polystyrene_bead",
            rho=1050.0,
            c=2350.0,
        )
    
    @staticmethod
    def silica_bead() -> ParticleMaterial:
        """Silica (glass) microsphere."""
        return ParticleMaterial(
            name="silica_bead",
            rho=2200.0,
            c=5900.0,
        )
    
    @staticmethod
    def cell_typical() -> ParticleMaterial:
        """Typical biological cell (approximate)."""
        return ParticleMaterial(
            name="cell",
            rho=1050.0,
            c=1550.0,  # Slightly higher than water
        )
    
    @staticmethod
    def air_bubble() -> ParticleMaterial:
        """Air bubble in water."""
        return ParticleMaterial(
            name="air_bubble",
            rho=1.2,
            c=343.0,
        )


if __name__ == "__main__":
    # Demo: print material properties
    water = MaterialDatabase.water(25.0)
    air = MaterialDatabase.air(20.0)
    ps = MaterialDatabase.polystyrene()
    
    print("Water at 25°C:")
    print(f"  ρ = {water.rho:.1f} kg/m³")
    print(f"  c = {water.c:.1f} m/s")
    print(f"  Z = {water.Z:.2e} kg/(m²·s)")
    print(f"  η = {water.eta:.2e} Pa·s")
    print(f"  δv(1MHz) = {water.viscous_boundary_layer(2*np.pi*1e6)*1e6:.2f} μm")
    print(f"  δt(1MHz) = {water.thermal_boundary_layer(2*np.pi*1e6)*1e6:.2f} μm")
    
    print("\nAir at 20°C:")
    print(f"  ρ = {air.rho:.3f} kg/m³")
    print(f"  c = {air.c:.1f} m/s")
    print(f"  Z = {air.Z:.2f} kg/(m²·s)")
    
    print("\nPolystyrene:")
    print(f"  ρ = {ps.rho:.0f} kg/m³")
    print(f"  E = {ps.E/1e9:.1f} GPa")
    print(f"  c_L = {ps.c_longitudinal:.0f} m/s")
    print(f"  c_S = {ps.c_shear:.0f} m/s")
    
    # Contrast factors
    particle = ParticleDatabase.polystyrene_bead()
    f1, f2 = particle.contrast_factors(water)
    print(f"\nPolystyrene bead in water:")
    print(f"  f1 = {f1:.3f} (compressibility)")
    print(f"  f2 = {f2:.3f} (density)")
