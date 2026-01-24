"""
Particle properties for acoustic manipulation.

Defines particle material properties and contrast factors for
radiation force calculations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np

from ..acoustics.materials import FluidMaterial


@dataclass
class Particle3D:
    """
    Spherical particle properties.
    
    Parameters
    ----------
    a : float
        Particle radius [m].
    rho : float
        Particle density [kg/m³].
    c : float
        Sound speed in particle material [m/s].
    name : str
        Particle identifier.
    """
    a: float           # Radius [m]
    rho: float         # Density [kg/m³]
    c: float           # Sound speed [m/s]
    name: str = "particle"
    
    @property
    def kappa(self) -> float:
        """Compressibility [1/Pa]."""
        return 1.0 / (self.rho * self.c**2)
    
    @property
    def volume(self) -> float:
        """Particle volume [m³]."""
        return (4.0 / 3.0) * np.pi * self.a**3
    
    @property
    def mass(self) -> float:
        """Particle mass [kg]."""
        return self.rho * self.volume


@dataclass
class ParticleContrast:
    """
    Acoustic contrast factors for particle in fluid.
    
    The Gor'kov potential is:
        U = V * (f1*<E_pot> - (3/2)*f2*<E_kin>)
    
    where f1 and f2 are the contrast factors.
    """
    f1: float  # Compressibility contrast
    f2: float  # Density contrast
    particle: Particle3D
    fluid: FluidMaterial
    
    @property
    def acoustic_contrast_factor(self) -> float:
        """
        Combined acoustic contrast factor Φ.
        
        For standing wave: Φ = f1/3 + f2/2
        Positive Φ: particle moves to pressure node
        Negative Φ: particle moves to pressure antinode
        """
        return self.f1 / 3.0 + self.f2 / 2.0
    
    @property
    def is_positive_contrast(self) -> bool:
        """Check if particle has positive contrast (moves to nodes)."""
        return self.acoustic_contrast_factor > 0


def compute_contrast_factors(
    particle: Particle3D,
    fluid: FluidMaterial,
) -> ParticleContrast:
    """
    Compute acoustic contrast factors for particle in fluid.
    
    f1 = 1 - κp/κ0  (compressibility contrast)
    f2 = 2(ρp - ρ0)/(2ρp + ρ0)  (density contrast)
    
    Parameters
    ----------
    particle : Particle3D
        Particle properties.
    fluid : FluidMaterial
        Fluid properties.
    
    Returns
    -------
    contrast : ParticleContrast
        Contrast factors and related quantities.
    """
    kappa_0 = fluid.kappa_T
    kappa_p = particle.kappa
    rho_0 = fluid.rho
    rho_p = particle.rho
    
    f1 = 1.0 - kappa_p / kappa_0
    f2 = 2.0 * (rho_p - rho_0) / (2.0 * rho_p + rho_0)
    
    return ParticleContrast(
        f1=f1, f2=f2,
        particle=particle,
        fluid=fluid,
    )


class ParticleDatabase:
    """Database of common particle types."""
    
    @staticmethod
    def polystyrene_bead(radius_um: float = 5.0) -> Particle3D:
        """Polystyrene microsphere."""
        return Particle3D(
            a=radius_um * 1e-6,
            rho=1050.0,
            c=2350.0,
            name="polystyrene",
        )
    
    @staticmethod
    def silica_bead(radius_um: float = 5.0) -> Particle3D:
        """Silica (glass) microsphere."""
        return Particle3D(
            a=radius_um * 1e-6,
            rho=2200.0,
            c=5900.0,
            name="silica",
        )
    
    @staticmethod
    def cell(radius_um: float = 5.0) -> Particle3D:
        """Typical biological cell."""
        return Particle3D(
            a=radius_um * 1e-6,
            rho=1050.0,
            c=1550.0,
            name="cell",
        )
    
    @staticmethod
    def air_bubble(radius_um: float = 10.0) -> Particle3D:
        """Air bubble in water."""
        return Particle3D(
            a=radius_um * 1e-6,
            rho=1.2,
            c=343.0,
            name="air_bubble",
        )
    
    @staticmethod
    def lipid_droplet(radius_um: float = 5.0) -> Particle3D:
        """Lipid droplet (lower density than water)."""
        return Particle3D(
            a=radius_um * 1e-6,
            rho=900.0,
            c=1400.0,
            name="lipid",
        )


if __name__ == "__main__":
    from ..acoustics.materials import MaterialDatabase
    
    # Demo: compute contrast factors
    water = MaterialDatabase.water(25.0)
    
    particles = [
        ParticleDatabase.polystyrene_bead(5.0),
        ParticleDatabase.silica_bead(5.0),
        ParticleDatabase.cell(5.0),
        ParticleDatabase.air_bubble(10.0),
        ParticleDatabase.lipid_droplet(5.0),
    ]
    
    print("Acoustic contrast factors in water at 25°C:")
    print("-" * 60)
    print(f"{'Particle':<15} {'f1':>8} {'f2':>8} {'Φ':>8} {'Moves to':>12}")
    print("-" * 60)
    
    for p in particles:
        contrast = compute_contrast_factors(p, water)
        destination = "nodes" if contrast.is_positive_contrast else "antinodes"
        print(f"{p.name:<15} {contrast.f1:>8.3f} {contrast.f2:>8.3f} "
              f"{contrast.acoustic_contrast_factor:>8.3f} {destination:>12}")
