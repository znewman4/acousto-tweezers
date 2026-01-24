"""
Solid material properties for elastic wave simulation.

Defines material properties for:
- Elastic solids (dish plate, walls)
- Viscoelastic damping via complex modulus
- Derived quantities (wave speeds, impedances)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class SolidMaterial:
    """
    Elastic solid material properties.
    
    Parameters
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
        Loss factor for viscoelastic damping.
        Complex modulus: E → E(1 + i·η)
    """
    name: str
    rho: float          # Density [kg/m³]
    E: float            # Young's modulus [Pa]
    nu: float           # Poisson's ratio [-]
    loss_eta: float = 0.0  # Loss factor
    
    @property
    def E_complex(self) -> complex:
        """Complex Young's modulus with damping."""
        return self.E * (1.0 + 1j * self.loss_eta)
    
    @property
    def lambda_lame(self) -> float:
        """First Lamé parameter λ [Pa] (real part)."""
        return self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
    
    @property
    def mu_lame(self) -> float:
        """Second Lamé parameter (shear modulus) μ [Pa] (real part)."""
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
    def bulk_modulus(self) -> float:
        """Bulk modulus K [Pa]."""
        return self.E / (3 * (1 - 2 * self.nu))
    
    @property
    def c_longitudinal(self) -> float:
        """Longitudinal (P-wave) speed [m/s]."""
        lam = self.lambda_lame
        mu = self.mu_lame
        return np.sqrt((lam + 2 * mu) / self.rho)
    
    @property
    def c_shear(self) -> float:
        """Shear (S-wave) speed [m/s]."""
        return np.sqrt(self.mu_lame / self.rho)
    
    @property
    def c_plate(self) -> float:
        """Plate wave speed (thin plate limit) [m/s]."""
        return np.sqrt(self.E / (self.rho * (1 - self.nu**2)))
    
    @property
    def Z_longitudinal(self) -> float:
        """Longitudinal acoustic impedance [kg/(m²·s)]."""
        return self.rho * self.c_longitudinal
    
    @property
    def Z_shear(self) -> float:
        """Shear acoustic impedance [kg/(m²·s)]."""
        return self.rho * self.c_shear
    
    def wavelength(self, freq_hz: float, wave_type: str = 'longitudinal') -> float:
        """
        Compute wavelength at given frequency.
        
        Parameters
        ----------
        freq_hz : float
            Frequency [Hz].
        wave_type : str
            'longitudinal', 'shear', or 'plate'.
        
        Returns
        -------
        wavelength : float
            Wavelength [m].
        """
        if wave_type == 'longitudinal':
            c = self.c_longitudinal
        elif wave_type == 'shear':
            c = self.c_shear
        elif wave_type == 'plate':
            c = self.c_plate
        else:
            raise ValueError(f"Unknown wave type: {wave_type}")
        
        return c / freq_hz
    
    def wavenumber(self, omega: float, wave_type: str = 'longitudinal') -> complex:
        """
        Compute complex wavenumber including damping.
        
        Parameters
        ----------
        omega : float
            Angular frequency [rad/s].
        wave_type : str
            'longitudinal' or 'shear'.
        
        Returns
        -------
        k : complex
            Complex wavenumber [1/m].
        """
        if wave_type == 'longitudinal':
            lam = self.lambda_complex
            mu = self.mu_complex
            c_sq = (lam + 2 * mu) / self.rho
        else:
            c_sq = self.mu_complex / self.rho
        
        return omega / np.sqrt(c_sq)


@dataclass
class ViscoelasticMaterial(SolidMaterial):
    """
    Viscoelastic solid with frequency-dependent modulus.
    
    Implements standard linear solid (SLS) model:
        E*(ω) = E_∞ + (E_0 - E_∞) / (1 + iωτ)
    
    Parameters
    ----------
    E_0 : float
        Relaxed (low-frequency) modulus [Pa].
    E_inf : float
        Unrelaxed (high-frequency) modulus [Pa].
    tau : float
        Relaxation time [s].
    """
    E_0: float = None      # Relaxed modulus
    E_inf: float = None    # Unrelaxed modulus  
    tau: float = 0.0       # Relaxation time
    
    def __post_init__(self):
        if self.E_0 is None:
            self.E_0 = self.E
        if self.E_inf is None:
            self.E_inf = self.E
    
    def E_frequency(self, omega: float) -> complex:
        """
        Compute frequency-dependent complex modulus.
        
        Parameters
        ----------
        omega : float
            Angular frequency [rad/s].
        
        Returns
        -------
        E_star : complex
            Complex modulus at this frequency.
        """
        if self.tau == 0:
            return self.E_complex
        
        return self.E_inf + (self.E_0 - self.E_inf) / (1.0 + 1j * omega * self.tau)
    
    def loss_angle(self, omega: float) -> float:
        """
        Compute loss angle δ = arctan(E''/E').
        
        Parameters
        ----------
        omega : float
            Angular frequency [rad/s].
        
        Returns
        -------
        delta : float
            Loss angle [radians].
        """
        E_star = self.E_frequency(omega)
        return np.arctan2(np.imag(E_star), np.real(E_star))


class SolidMaterialDatabase:
    """Database of common solid materials."""
    
    @staticmethod
    def polystyrene() -> SolidMaterial:
        """Polystyrene (typical petri dish)."""
        return SolidMaterial(
            name="polystyrene",
            rho=1050.0,
            E=3.0e9,
            nu=0.34,
            loss_eta=0.01,
        )
    
    @staticmethod
    def pmma() -> SolidMaterial:
        """PMMA (acrylic)."""
        return SolidMaterial(
            name="pmma",
            rho=1180.0,
            E=3.3e9,
            nu=0.37,
            loss_eta=0.02,
        )
    
    @staticmethod
    def glass_borosilicate() -> SolidMaterial:
        """Borosilicate glass."""
        return SolidMaterial(
            name="glass",
            rho=2230.0,
            E=63e9,
            nu=0.20,
            loss_eta=0.001,
        )
    
    @staticmethod
    def pdms_soft() -> SolidMaterial:
        """Soft PDMS."""
        return SolidMaterial(
            name="pdms",
            rho=970.0,
            E=1.0e6,
            nu=0.49,
            loss_eta=0.05,
        )
    
    @staticmethod
    def steel() -> SolidMaterial:
        """Steel."""
        return SolidMaterial(
            name="steel",
            rho=7800.0,
            E=200e9,
            nu=0.30,
            loss_eta=0.001,
        )
    
    @staticmethod
    def aluminum() -> SolidMaterial:
        """Aluminum."""
        return SolidMaterial(
            name="aluminum",
            rho=2700.0,
            E=70e9,
            nu=0.33,
            loss_eta=0.001,
        )


if __name__ == "__main__":
    # Demo: print material properties
    ps = SolidMaterialDatabase.polystyrene()
    
    print(f"Polystyrene properties:")
    print(f"  ρ = {ps.rho} kg/m³")
    print(f"  E = {ps.E/1e9:.1f} GPa")
    print(f"  ν = {ps.nu}")
    print(f"  c_L = {ps.c_longitudinal:.0f} m/s")
    print(f"  c_S = {ps.c_shear:.0f} m/s")
    print(f"  c_plate = {ps.c_plate:.0f} m/s")
    print(f"  Z_L = {ps.Z_longitudinal:.2e} kg/(m²·s)")
    
    # Wavelengths at 1 MHz
    f = 1e6
    print(f"\nAt {f/1e6:.0f} MHz:")
    print(f"  λ_L = {ps.wavelength(f, 'longitudinal')*1e3:.2f} mm")
    print(f"  λ_S = {ps.wavelength(f, 'shear')*1e3:.2f} mm")
