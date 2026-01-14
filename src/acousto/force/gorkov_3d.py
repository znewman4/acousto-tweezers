"""
3D Gor'kov radiation force computation.

Extends gorkov_2d to handle true 3D acoustic fields.
Formula remains the same, but energy densities computed in 3D.
"""
from __future__ import annotations

import numpy as np

from acousto.solvers.helmholtz_3d_simple import Field3D
from .gorkov_1d import ParticleProps


def gorkov_potential_and_force_3d(
    field: Field3D,
    particle: ParticleProps,
    return_velocity: bool = False,
):
    """3D Gor'kov radiation potential and force.
    
    Same formula as 2D, but computed over 3D domain.
    
    Formula:
      kappa0 = 1/(rho0*c0^2)
      kappap = 1/(rho_p*c_p^2)
      f1 = 1 - kappap/kappa0
      f2 = 2(rho_p - rho0)/(2*rho_p + rho0)
      E_pot = 0.25 |p|^2 kappa0
      E_kin = 0.25 rho0 |v|^2
      U = V (f1 E_pot - 1.5 f2 E_kin)
      F = -∇U
    
    Parameters
    ----------
    field : Field3D
        3D pressure field with shape (Nz, Ny, Nx).
    particle : ParticleProps
        Particle properties.
    return_velocity : bool
        If True, also return velocity field vx, vy, vz.
    
    Returns
    -------
    (U, Fx, Fy, Fz) or (U, Fx, Fy, Fz, vx, vy, vz) if return_velocity=True
    All arrays have shape (Nz, Ny, Nx).
    """
    x = field.x
    y = field.y
    z = field.z
    p = field.p
    omega = field.omega
    rho0 = field.rho0
    c0 = field.c0

    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = z[1] - z[0]

    # Compressibilities
    kappa0 = 1.0 / (rho0 * c0**2)
    kappap = 1.0 / (particle.rho_p * particle.c_p**2)

    # Contrast factors
    f1 = 1.0 - (kappap / kappa0)
    f2 = 2.0 * (particle.rho_p - rho0) / (2.0 * particle.rho_p + rho0)

    # Pressure gradients: np.gradient returns [∂/∂axis0, ∂/∂axis1, ∂/∂axis2]
    # For our (Nz, Ny, Nx) array, this is [∂/∂z, ∂/∂y, ∂/∂x]
    dpz, dpy, dpx = np.gradient(p, dz, dy, dx, edge_order=2)

    # Velocity phasor: v = (1/(i*omega*rho0)) ∇p
    vx = dpx / (1j * omega * rho0)
    vy = dpy / (1j * omega * rho0)
    vz = dpz / (1j * omega * rho0)

    # Time-averaged energy densities
    E_pot = 0.25 * (np.abs(p) ** 2) * kappa0
    v2 = (np.abs(vx) ** 2 + np.abs(vy) ** 2 + np.abs(vz) ** 2)
    E_kin = 0.25 * rho0 * v2

    # Gor'kov potential
    V = (4.0 / 3.0) * np.pi * (particle.a ** 3)
    U = V * (f1 * E_pot - 1.5 * f2 * E_kin)

    # Force components: F = -∇U
    dUz, dUy, dUx = np.gradient(U, dz, dy, dx, edge_order=2)
    Fx = -dUx
    Fy = -dUy
    Fz = -dUz

    if return_velocity:
        return U, Fx, Fy, Fz, vx, vy, vz
    return U, Fx, Fy, Fz
