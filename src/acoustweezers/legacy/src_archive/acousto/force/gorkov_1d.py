from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from acousto.solvers.fd_helmholtz_1d import Field1D


@dataclass(frozen=True)
class ParticleProps:
    """Material/size properties for a spherical particle."""
    a: float        # radius [m]
    rho_p: float    # particle density [kg/m^3]
    c_p: float      # particle speed of sound [m/s]


def _dpdx_central(x: np.ndarray, p: np.ndarray) -> np.ndarray:
    """1D derivative dp/dx using central differences (one-sided at ends)."""
    dpdx = np.empty_like(p, dtype=np.complex128)
    h = x[1] - x[0]

    # interior: central diff
    dpdx[1:-1] = (p[2:] - p[:-2]) / (2.0 * h)

    # ends: one-sided
    dpdx[0] = (p[1] - p[0]) / h
    dpdx[-1] = (p[-1] - p[-2]) / h
    return dpdx


def gorkov_potential_and_force_1d(
    field: Field1D,
    particle: ParticleProps,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute 1D Gor'kov radiation potential U(x) and force F(x) = -dU/dx.

    Returns:
      U: Gor'kov potential [J] (up to modelling assumptions)
      F: radiation force [N]
      v: complex acoustic particle velocity amplitude in fluid [m/s]
    """
    x = field.x
    p = field.p
    omega = field.omega
    rho0 = field.rho0
    c0 = field.c0

    # Fluid compressibility kappa0 = 1/(rho0*c0^2)
    kappa0 = 1.0 / (rho0 * c0**2)
    # Particle compressibility kappap = 1/(rhop*cp^2)
    kappap = 1.0 / (particle.rho_p * particle.c_p**2)

    # Contrast factors
    f1 = 1.0 - (kappap / kappa0)
    f2 = 2.0 * (particle.rho_p - rho0) / (2.0 * particle.rho_p + rho0)

    # Velocity amplitude v = (1/(i*omega*rho0)) * dp/dx
    dpdx = _dpdx_central(x, p)
    v = dpdx / (1j * omega * rho0)

    # Time-averaged energy densities for harmonic fields
    E_pot = 0.25 * (np.abs(p) ** 2) * kappa0                  # = |p|^2/(4 rho0 c0^2)
    E_kin = 0.25 * rho0 * (np.abs(v) ** 2)

    # Gor'kov potential U
    V = (4.0 / 3.0) * np.pi * (particle.a ** 3)               # particle volume
    U = V * (f1 * E_pot - 1.5 * f2 * E_kin)

    # Force F = -dU/dx
    dUdx = np.empty_like(U, dtype=np.float64)
    h = x[1] - x[0]
    dUdx[1:-1] = (U[2:] - U[:-2]) / (2.0 * h)
    dUdx[0] = (U[1] - U[0]) / h
    dUdx[-1] = (U[-1] - U[-2]) / h

    F = -dUdx
    return U, F, v
