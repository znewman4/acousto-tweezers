"""
Gor'kov potential and radiation force for 3D acoustic fields.

Implements the time-averaged acoustic radiation force on
small spherical particles in an acoustic field.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np

from .properties import Particle3D, ParticleContrast, compute_contrast_factors
from .interpolation import Grid3D, TrilinearInterpolator, GradientInterpolator
from ..acoustics.materials import FluidMaterial


@dataclass
class GorkovPotential3D:
    """
    3D Gor'kov acoustic radiation potential.
    
    The Gor'kov potential is:
        U = V_p * (f1/(4ρc²) * <p²> - 3*f2/(8ρ) * <v²>)
    
    where:
        V_p = particle volume
        f1, f2 = contrast factors
        <p²> = time-averaged pressure squared
        <v²> = time-averaged velocity squared magnitude
    
    The radiation force is:
        F = -∇U
    """
    
    def __init__(
        self,
        grid: Grid3D,
        pressure: np.ndarray,
        fluid: FluidMaterial,
        omega: float,
        velocity: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    ):
        """
        Initialize Gor'kov potential calculator.
        
        Parameters
        ----------
        grid : Grid3D
            Spatial grid.
        pressure : np.ndarray
            Complex pressure field, shape (nx, ny, nz).
        fluid : FluidMaterial
            Fluid properties.
        omega : float
            Angular frequency [rad/s].
        velocity : tuple of 3 arrays, optional
            Complex velocity components (vx, vy, vz).
            If None, computed from pressure gradient.
        """
        self.grid = grid
        self.pressure = pressure
        self.fluid = fluid
        self.omega = omega
        
        # Compute velocity if not provided
        if velocity is None:
            # v = -1/(iωρ) ∇p
            grad_p = np.gradient(pressure, grid.dx, grid.dy, grid.dz)
            factor = -1.0 / (1j * omega * fluid.rho)
            self.vx = factor * grad_p[0]
            self.vy = factor * grad_p[1]
            self.vz = factor * grad_p[2]
        else:
            self.vx, self.vy, self.vz = velocity
        
        # Compute time-averaged quantities
        # <p²> = |p|²/2
        self.p_squared = 0.5 * np.abs(pressure)**2
        
        # <v²> = |v|²/2
        self.v_squared = 0.5 * (
            np.abs(self.vx)**2 + np.abs(self.vy)**2 + np.abs(self.vz)**2
        )
    
    def compute_potential(
        self,
        particle: Particle3D,
    ) -> np.ndarray:
        """
        Compute Gor'kov potential field for given particle.
        
        Parameters
        ----------
        particle : Particle3D
            Particle properties.
        
        Returns
        -------
        U : np.ndarray
            Gor'kov potential field [J], shape (nx, ny, nz).
        """
        contrast = compute_contrast_factors(particle, self.fluid)
        
        rho = self.fluid.rho
        c = self.fluid.c
        V = particle.volume
        
        # U = V * (f1/(4ρc²) * <p²> - 3*f2/(8ρ) * <v²>)
        coef_p = contrast.f1 / (4.0 * rho * c**2)
        coef_v = 3.0 * contrast.f2 / (8.0 * rho)
        
        U = V * (coef_p * self.p_squared - coef_v * self.v_squared)
        
        return U
    
    def compute_force(
        self,
        particle: Particle3D,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute radiation force field F = -∇U.
        
        Parameters
        ----------
        particle : Particle3D
            Particle properties.
        
        Returns
        -------
        Fx, Fy, Fz : np.ndarray
            Force field components [N], each shape (nx, ny, nz).
        """
        U = self.compute_potential(particle)
        
        # F = -∇U
        grad_U = np.gradient(U, self.grid.dx, self.grid.dy, self.grid.dz)
        
        Fx = -grad_U[0]
        Fy = -grad_U[1]
        Fz = -grad_U[2]
        
        return Fx, Fy, Fz
    
    def force_at(
        self,
        particle: Particle3D,
        position: np.ndarray,
    ) -> np.ndarray:
        """
        Interpolate radiation force at a specific position.
        
        Parameters
        ----------
        particle : Particle3D
            Particle properties.
        position : np.ndarray
            Position (x, y, z) [m].
        
        Returns
        -------
        F : np.ndarray
            Force vector (Fx, Fy, Fz) [N].
        """
        Fx, Fy, Fz = self.compute_force(particle)
        
        interp_x = TrilinearInterpolator(self.grid, Fx)
        interp_y = TrilinearInterpolator(self.grid, Fy)
        interp_z = TrilinearInterpolator(self.grid, Fz)
        
        position = np.atleast_2d(position)
        return np.array([
            interp_x(position)[0],
            interp_y(position)[0],
            interp_z(position)[0],
        ])
    
    def potential_at(
        self,
        particle: Particle3D,
        position: np.ndarray,
    ) -> float:
        """
        Interpolate Gor'kov potential at a specific position.
        
        Parameters
        ----------
        particle : Particle3D
            Particle properties.
        position : np.ndarray
            Position (x, y, z) [m].
        
        Returns
        -------
        U : float
            Potential [J].
        """
        U = self.compute_potential(particle)
        interp = TrilinearInterpolator(self.grid, U)
        position = np.atleast_2d(position)
        return float(interp(position)[0])


def estimate_max_radiation_force(
    particle: Particle3D,
    fluid: FluidMaterial,
    frequency: float,
    pressure_amplitude: float,
) -> float:
    """
    Estimate maximum radiation force for standing wave.
    
    For a 1D standing wave p = p0*sin(kx), the maximum force is:
        F_max = 4π*a³*k * Φ * E_ac / 3
    
    where E_ac = p0²/(4ρc²) is the acoustic energy density.
    
    Parameters
    ----------
    particle : Particle3D
        Particle properties.
    fluid : FluidMaterial
        Fluid properties.
    frequency : float
        Acoustic frequency [Hz].
    pressure_amplitude : float
        Pressure amplitude [Pa].
    
    Returns
    -------
    F_max : float
        Maximum force magnitude [N].
    """
    contrast = compute_contrast_factors(particle, fluid)
    Phi = contrast.acoustic_contrast_factor
    
    k = 2.0 * np.pi * frequency / fluid.c
    E_ac = pressure_amplitude**2 / (4.0 * fluid.rho * fluid.c**2)
    
    F_max = 4.0 * np.pi * particle.a**3 * k * abs(Phi) * E_ac / 3.0
    
    return F_max


def compute_force_normalized(
    Fx: np.ndarray, Fy: np.ndarray, Fz: np.ndarray,
    particle: Particle3D,
    fluid: FluidMaterial,
    frequency: float,
    pressure_amplitude: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize force field by characteristic force scale.
    
    Parameters
    ----------
    Fx, Fy, Fz : np.ndarray
        Force components [N].
    particle : Particle3D
        Particle properties.
    fluid : FluidMaterial
        Fluid properties.
    frequency : float
        Frequency [Hz].
    pressure_amplitude : float
        Reference pressure amplitude [Pa].
    
    Returns
    -------
    Fx_norm, Fy_norm, Fz_norm : np.ndarray
        Normalized force components.
    """
    F_scale = estimate_max_radiation_force(
        particle, fluid, frequency, pressure_amplitude
    )
    
    if F_scale > 0:
        return Fx / F_scale, Fy / F_scale, Fz / F_scale
    else:
        return Fx, Fy, Fz


def compute_stiffness(
    gorkov: GorkovPotential3D,
    particle: Particle3D,
    position: np.ndarray,
    delta: float = 1e-6,
) -> np.ndarray:
    """
    Compute trap stiffness at a position.
    
    Stiffness κ_i = -∂F_i/∂x_i (diagonal elements only).
    
    Parameters
    ----------
    gorkov : GorkovPotential3D
        Gor'kov potential calculator.
    particle : Particle3D
        Particle properties.
    position : np.ndarray
        Position (x, y, z) [m].
    delta : float
        Finite difference step [m].
    
    Returns
    -------
    stiffness : np.ndarray
        Stiffness (κx, κy, κz) [N/m].
    """
    stiffness = np.zeros(3)
    
    for i in range(3):
        pos_plus = position.copy()
        pos_minus = position.copy()
        pos_plus[i] += delta
        pos_minus[i] -= delta
        
        F_plus = gorkov.force_at(particle, pos_plus)
        F_minus = gorkov.force_at(particle, pos_minus)
        
        stiffness[i] = -(F_plus[i] - F_minus[i]) / (2.0 * delta)
    
    return stiffness


def find_potential_minima(
    U: np.ndarray,
    grid: Grid3D,
    threshold_fraction: float = 0.1,
) -> np.ndarray:
    """
    Find local minima of Gor'kov potential (trap locations).
    
    Parameters
    ----------
    U : np.ndarray
        Gor'kov potential field.
    grid : Grid3D
        Spatial grid.
    threshold_fraction : float
        Fraction of potential range to consider as minima.
    
    Returns
    -------
    minima : np.ndarray
        Array of (x, y, z) positions of local minima.
    """
    from scipy.ndimage import minimum_filter
    
    # Find local minima using morphological filter
    filtered = minimum_filter(U.real, size=3, mode='constant', cval=np.inf)
    local_min = (U.real == filtered) & (U.real < filtered.max())
    
    # Apply threshold
    U_range = U.real.max() - U.real.min()
    threshold = U.real.min() + threshold_fraction * U_range
    local_min = local_min & (U.real < threshold)
    
    # Get indices
    indices = np.argwhere(local_min)
    
    # Convert to positions
    minima = np.zeros((len(indices), 3))
    for i, (ix, iy, iz) in enumerate(indices):
        minima[i] = [grid.x[ix], grid.y[iy], grid.z[iz]]
    
    return minima
