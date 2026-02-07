from dataclasses import dataclass
import numpy as np

@dataclass
class ParticleProps:
    a_m: float  # particle radius (m)
    rho_p: float  # particle density (kg/m^3)
    kappa_p: float  # particle compressibility (1/Pa)

@dataclass
class FluidProps:
    rho0: float  # fluid density (kg/m^3)
    c0: float    # sound speed (m/s)
    eta: float   # dynamic viscosity (Pa·s)
    kappa0: float = None  # fluid compressibility (1/Pa, optional)

    def __post_init__(self):
        if self.kappa0 is None:
            self.kappa0 = 1.0 / (self.rho0 * self.c0 ** 2)

def stokes_mobility(eta, a):
    """Return Stokes mobility mu = 1/(6*pi*eta*a) [m/(N·s)]"""
    return 1.0 / (6 * np.pi * eta * a)

def contrast_factors(fluid: FluidProps, particle: ParticleProps):
    """Return (f1, f2) for Gor'kov potential using standard definitions."""
    kappa0 = fluid.kappa0
    kappa_p = particle.kappa_p
    rho0 = fluid.rho0
    rho_p = particle.rho_p
    f1 = 1 - kappa_p / kappa0
    f2 = 2 * (rho_p - rho0) / (2 * rho_p + rho0)
    return f1, f2
