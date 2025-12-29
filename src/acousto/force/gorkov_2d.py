# src/acousto/force/gorkov_2d.py
from __future__ import annotations

import numpy as np

from acousto.solvers.fd_helmholtz_2d import Field2D
from .gorkov_1d import ParticleProps


def gorkov_potential_and_force_2d(
    field: Field2D,
    particle: ParticleProps,
    return_velocity: bool = False,
):
    """
    2D Gor'kov radiation potential U(x,y) and force F = -∇U.

    Matches the 1D implementation exactly in definitions and prefactors:
      kappa0 = 1/(rho0*c0^2)
      kappap = 1/(rho_p*c_p^2)
      f1 = 1 - kappap/kappa0
      f2 = 2 (rho_p - rho0)/(2 rho_p + rho0)
      E_pot = 0.25 |p|^2 kappa0
      E_kin = 0.25 rho0 |v|^2
      U = V ( f1 E_pot - 1.5 f2 E_kin )
      F = -∇U

    Conventions:
      field.p has shape (Ny, Nx) with p[j,i] = p(y_j, x_i)
    """
    x = field.x
    y = field.y
    p = field.p
    omega = field.omega
    rho0 = field.rho0
    c0 = field.c0

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # Compressibilities
    kappa0 = 1.0 / (rho0 * c0**2)
    kappap = 1.0 / (particle.rho_p * particle.c_p**2)

    # Contrast factors (same as 1D)
    f1 = 1.0 - (kappap / kappa0)
    f2 = 2.0 * (particle.rho_p - rho0) / (2.0 * particle.rho_p + rho0)

    # Pressure gradients (np.gradient returns [d/dy, d/dx] for a 2D array)
    dpdy, dpdx = np.gradient(p, dy, dx, edge_order=2)

    # Velocity phasor components: v = (1/(i*omega*rho0)) ∇p
    vx = dpdx / (1j * omega * rho0)
    vy = dpdy / (1j * omega * rho0)

    # Time-averaged energy densities (same prefactors as 1D)
    E_pot = 0.25 * (np.abs(p) ** 2) * kappa0
    v2 = (np.abs(vx) ** 2 + np.abs(vy) ** 2)
    E_kin = 0.25 * rho0 * v2

    # Gor'kov potential
    V = (4.0 / 3.0) * np.pi * (particle.a ** 3)
    U = V * (f1 * E_pot - 1.5 * f2 * E_kin)

    # Force components: F = -∇U
    dUdy, dUdx = np.gradient(U, dy, dx, edge_order=2)
    Fx = -dUdx
    Fy = -dUdy

    if return_velocity:
        return U, Fx, Fy, vx, vy
    return U, Fx, Fy
