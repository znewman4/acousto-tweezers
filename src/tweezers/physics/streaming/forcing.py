"""
Acoustic streaming forcing computation.

Calculates the time-averaged body force that drives acoustic streaming
from first-order acoustic fields.

The streaming force arises from the nonlinear Reynolds stress:
    f_stream = -ρ₀ <v₁·∇v₁> - <ρ₁v₁>·∇v₁

For irrotational first-order flow with thermoviscous losses, the dominant
contribution is from boundary layer effects (Rayleigh streaming) and
from gradients in acoustic intensity (Eckart streaming).

This module implements:
1. Eckart streaming forcing (bulk)
2. Rayleigh streaming forcing (near boundaries)
3. Combined streaming force field
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np

from ..acoustics.materials import FluidMaterial


@dataclass
class StreamingForcing:
    """
    Streaming body force field.
    
    Parameters
    ----------
    x, y, z : np.ndarray
        Coordinate arrays.
    fx, fy, fz : np.ndarray
        Force density components [N/m³].
    """
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    fx: np.ndarray
    fy: np.ndarray
    fz: np.ndarray
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.fx.shape
    
    @property
    def magnitude(self) -> np.ndarray:
        """Force magnitude field."""
        return np.sqrt(self.fx**2 + self.fy**2 + self.fz**2)
    
    def total_force(self, dV: float) -> np.ndarray:
        """Compute total integrated force vector."""
        return np.array([
            np.sum(self.fx) * dV,
            np.sum(self.fy) * dV,
            np.sum(self.fz) * dV,
        ])


def compute_acoustic_velocity(
    p: np.ndarray,
    rho: np.ndarray,
    omega: float,
    dx: float,
    dy: float,
    dz: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute first-order velocity from pressure field.
    
    v₁ = -1/(iωρ) ∇p
    
    Parameters
    ----------
    p : np.ndarray
        Complex pressure field, shape (Nx, Ny, Nz).
    rho : np.ndarray
        Density field.
    omega : float
        Angular frequency.
    dx, dy, dz : float
        Grid spacing.
    
    Returns
    -------
    vx, vy, vz : np.ndarray
        Velocity components (complex).
    """
    dpx = np.gradient(p, dx, axis=0)
    dpy = np.gradient(p, dy, axis=1)
    dpz = np.gradient(p, dz, axis=2)
    
    factor = -1.0 / (1j * omega * rho)
    return factor * dpx, factor * dpy, factor * dpz


def compute_eckart_streaming_force(
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    rho0: float,
    c0: float,
    alpha: float,
    dx: float,
    dy: float,
    dz: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Eckart streaming body force.
    
    For a lossy medium, the Eckart streaming force is:
        f_E = 2α/c₀ * I
    
    where α is the attenuation coefficient and I is the intensity vector.
    
    The intensity is:
        I = ½ Re(p v*)
    
    So:
        f_E = (α/c₀) Re(p v*)
    
    Parameters
    ----------
    vx, vy, vz : np.ndarray
        First-order velocity (complex).
    rho0 : float
        Mean density [kg/m³].
    c0 : float
        Sound speed [m/s].
    alpha : float
        Attenuation coefficient [Np/m].
    dx, dy, dz : float
        Grid spacing.
    
    Returns
    -------
    fx, fy, fz : np.ndarray
        Eckart streaming force density [N/m³].
    """
    # Compute |v|² for intensity
    v_sq = np.abs(vx)**2 + np.abs(vy)**2 + np.abs(vz)**2
    
    # Intensity magnitude
    I_mag = 0.5 * rho0 * c0 * v_sq
    
    # Intensity direction (from velocity)
    v_mag = np.sqrt(v_sq + 1e-30)
    
    # Time-averaged velocity direction
    # For plane wave, this is the propagation direction
    # For standing wave, this is more complex
    # Use gradient of |v|² as proxy for intensity direction
    
    dI_dx = np.gradient(I_mag, dx, axis=0)
    dI_dy = np.gradient(I_mag, dy, axis=1)
    dI_dz = np.gradient(I_mag, dz, axis=2)
    
    # Eckart force: f = 2α/c * I (in direction of propagation)
    # Use intensity gradient direction as approximation
    factor = 2 * alpha / c0
    
    fx = factor * dI_dx
    fy = factor * dI_dy
    fz = factor * dI_dz
    
    return fx.real, fy.real, fz.real


def compute_reynolds_stress_force(
    p: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    rho0: float,
    c0: float,
    dx: float,
    dy: float,
    dz: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute streaming force from Reynolds stress divergence.
    
    f_i = -∂<ρ₀ v_i v_j>/∂x_j - ∂<p' v_i>/∂x_j / c₀²
    
    For time-harmonic fields:
        <v_i v_j> = ½ Re(v_i v_j*)
    
    Parameters
    ----------
    p : np.ndarray
        Pressure field (complex).
    vx, vy, vz : np.ndarray
        Velocity components (complex).
    rho0 : float
        Mean density.
    c0 : float
        Sound speed.
    dx, dy, dz : float
        Grid spacing.
    
    Returns
    -------
    fx, fy, fz : np.ndarray
        Reynolds stress force [N/m³].
    """
    # Reynolds stress tensor components (time-averaged)
    # τ_ij = ρ₀ <v_i v_j> = (ρ₀/2) Re(v_i v_j*)
    
    tau_xx = 0.5 * rho0 * np.real(vx * np.conj(vx))
    tau_yy = 0.5 * rho0 * np.real(vy * np.conj(vy))
    tau_zz = 0.5 * rho0 * np.real(vz * np.conj(vz))
    tau_xy = 0.5 * rho0 * np.real(vx * np.conj(vy))
    tau_xz = 0.5 * rho0 * np.real(vx * np.conj(vz))
    tau_yz = 0.5 * rho0 * np.real(vy * np.conj(vz))
    
    # Force = -∇·τ
    fx = -(np.gradient(tau_xx, dx, axis=0) +
           np.gradient(tau_xy, dy, axis=1) +
           np.gradient(tau_xz, dz, axis=2))
    
    fy = -(np.gradient(tau_xy, dx, axis=0) +
           np.gradient(tau_yy, dy, axis=1) +
           np.gradient(tau_yz, dz, axis=2))
    
    fz = -(np.gradient(tau_xz, dx, axis=0) +
           np.gradient(tau_yz, dy, axis=1) +
           np.gradient(tau_zz, dz, axis=2))
    
    return fx, fy, fz


def compute_streaming_force(
    p: np.ndarray,
    rho: np.ndarray,
    omega: float,
    fluid: FluidMaterial,
    dx: float,
    dy: float,
    dz: float,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    include_eckart: bool = True,
    include_reynolds: bool = True,
) -> StreamingForcing:
    """
    Compute total acoustic streaming body force.
    
    Combines Eckart and Reynolds stress contributions.
    
    Parameters
    ----------
    p : np.ndarray
        Pressure field (complex).
    rho : np.ndarray
        Density field.
    omega : float
        Angular frequency.
    fluid : FluidMaterial
        Fluid properties.
    dx, dy, dz : float
        Grid spacing.
    x, y, z : np.ndarray
        Coordinate arrays.
    include_eckart : bool
        Include Eckart (attenuation-driven) streaming.
    include_reynolds : bool
        Include Reynolds stress streaming.
    
    Returns
    -------
    forcing : StreamingForcing
        Streaming body force field.
    """
    # Compute velocity field
    vx, vy, vz = compute_acoustic_velocity(p, rho, omega, dx, dy, dz)
    
    fx = np.zeros_like(p, dtype=float)
    fy = np.zeros_like(p, dtype=float)
    fz = np.zeros_like(p, dtype=float)
    
    if include_eckart:
        # Estimate attenuation from loss factor
        alpha = omega / fluid.c * fluid.loss_factor / 2
        
        fx_e, fy_e, fz_e = compute_eckart_streaming_force(
            vx, vy, vz, fluid.rho, fluid.c, alpha, dx, dy, dz
        )
        fx += fx_e
        fy += fy_e
        fz += fz_e
    
    if include_reynolds:
        fx_r, fy_r, fz_r = compute_reynolds_stress_force(
            p, vx, vy, vz, fluid.rho, fluid.c, dx, dy, dz
        )
        fx += fx_r
        fy += fy_r
        fz += fz_r
    
    return StreamingForcing(
        x=x, y=y, z=z,
        fx=fx, fy=fy, fz=fz
    )


def boundary_layer_streaming_velocity(
    v_tangent: complex,
    delta_v: float,
    y_wall: float,
    y: np.ndarray,
) -> np.ndarray:
    """
    Compute Rayleigh streaming velocity profile near a wall.
    
    The classical Rayleigh streaming velocity just outside the
    viscous boundary layer is:
        U_s = -(3/4) * (|v_1|²/c) * (1 - cos(2φ))
    
    where v_1 is the tangential first-order velocity and φ is its phase.
    
    Parameters
    ----------
    v_tangent : complex
        Tangential first-order velocity amplitude.
    delta_v : float
        Viscous boundary layer thickness.
    y_wall : float
        Wall y-position.
    y : np.ndarray
        y-coordinates.
    
    Returns
    -------
    u_streaming : np.ndarray
        Streaming velocity profile.
    """
    # Distance from wall
    d = np.abs(y - y_wall)
    
    # Inside boundary layer: streaming increases
    # Outside: streaming is constant (Rayleigh streaming)
    u_outer = -0.375 * np.abs(v_tangent)**2 / 343.0  # Approximate
    
    # Profile: rises from 0 at wall to u_outer at y ~ 3δ
    profile = 1 - np.exp(-d / (3 * delta_v))
    
    return u_outer * profile


if __name__ == "__main__":
    # Demo: compute streaming force from synthetic acoustic field
    from ..acoustics.materials import MaterialDatabase
    
    # Create test grid
    Nx, Ny, Nz = 41, 41, 21
    x = np.linspace(0, 0.02, Nx)
    y = np.linspace(0, 0.02, Ny)
    z = np.linspace(0, 0.01, Nz)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = z[1] - z[0]
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Simple standing wave pressure
    omega = 2 * np.pi * 1e6
    water = MaterialDatabase.water(25.0)
    k = omega / water.c
    
    p0 = 1e5  # 0.1 MPa amplitude
    p = p0 * np.cos(k * Z)
    p = p.astype(np.complex128)
    
    rho = np.full_like(p, water.rho, dtype=float)
    
    print("Computing streaming force...")
    forcing = compute_streaming_force(
        p=p, rho=rho, omega=omega, fluid=water,
        dx=dx, dy=dy, dz=dz,
        x=x, y=y, z=z,
    )
    
    print(f"Force field shape: {forcing.shape}")
    print(f"Max |f|: {np.max(forcing.magnitude):.2e} N/m³")
    
    # Total force
    dV = dx * dy * dz
    F_total = forcing.total_force(dV)
    print(f"Total force: ({F_total[0]:.2e}, {F_total[1]:.2e}, {F_total[2]:.2e}) N")
