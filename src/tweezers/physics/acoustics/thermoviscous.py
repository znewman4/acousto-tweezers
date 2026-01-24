"""
Thermoviscous acoustics for near-wall boundary layer effects.

Implements linearized thermoviscous acoustics that captures:
- Viscous boundary layer dissipation
- Thermal boundary layer effects
- Phase shifts near solid boundaries

Theory:
-------
Near walls, the standard inviscid Helmholtz equation is inadequate because
viscous and thermal boundary layers have thicknesses:
    δ_v = √(2ν/ω)   (viscous)
    δ_t = √(2α/ω)   (thermal)

For typical water at 1 MHz:
    δ_v ≈ 0.56 μm
    δ_t ≈ 0.24 μm

When these are non-negligible relative to geometry or particle proximity,
thermoviscous effects must be modeled.

This module implements:
1. Boundary layer thickness calculation
2. Thermoviscous correction to pressure field near walls
3. Modified velocity field including viscous and thermal modes
4. Interface between thermoviscous and inviscid regions
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
from scipy import special

from .materials import FluidMaterial


def compute_boundary_layer_thickness(
    fluid: FluidMaterial,
    omega: float,
) -> Tuple[float, float]:
    """
    Compute viscous and thermal boundary layer thicknesses.
    
    Parameters
    ----------
    fluid : FluidMaterial
        Fluid properties.
    omega : float
        Angular frequency [rad/s].
    
    Returns
    -------
    delta_v : float
        Viscous boundary layer thickness [m].
    delta_t : float
        Thermal boundary layer thickness [m].
    """
    nu = fluid.eta / fluid.rho  # Kinematic viscosity
    delta_v = np.sqrt(2 * nu / omega)
    
    if fluid.alpha is not None:
        alpha = fluid.alpha
    else:
        # Estimate from Prandtl number
        alpha = nu / fluid.Pr
    
    delta_t = np.sqrt(2 * alpha / omega)
    
    return delta_v, delta_t


def boundary_layer_significance(
    fluid: FluidMaterial,
    omega: float,
    characteristic_length: float,
    threshold: float = 0.01,
) -> Tuple[bool, float]:
    """
    Determine if thermoviscous effects are significant.
    
    Parameters
    ----------
    fluid : FluidMaterial
        Fluid properties.
    omega : float
        Angular frequency [rad/s].
    characteristic_length : float
        Smallest geometric scale (gap, particle size, etc.) [m].
    threshold : float
        Ratio threshold above which thermoviscous effects are significant.
    
    Returns
    -------
    is_significant : bool
        Whether thermoviscous effects should be modeled.
    ratio : float
        Ratio of boundary layer to characteristic length.
    """
    delta_v, delta_t = compute_boundary_layer_thickness(fluid, omega)
    ratio = max(delta_v, delta_t) / characteristic_length
    return ratio > threshold, ratio


@dataclass
class ThermoviscousParameters:
    """
    Parameters for thermoviscous acoustic modeling.
    
    Parameters
    ----------
    fluid : FluidMaterial
        Base fluid properties.
    omega : float
        Angular frequency [rad/s].
    """
    fluid: FluidMaterial
    omega: float
    
    # Computed properties
    _delta_v: float = None
    _delta_t: float = None
    _k_v: complex = None  # Viscous wavenumber
    _k_t: complex = None  # Thermal wavenumber
    _k_a: complex = None  # Acoustic wavenumber
    
    def __post_init__(self):
        """Compute derived quantities."""
        self._compute_wavenumbers()
    
    def _compute_wavenumbers(self):
        """Compute acoustic, viscous, and thermal wavenumbers."""
        f = self.fluid
        omega = self.omega
        
        # Boundary layer thicknesses
        nu = f.eta / f.rho
        self._delta_v = np.sqrt(2 * nu / omega)
        
        if f.alpha is not None:
            alpha = f.alpha
        else:
            alpha = nu / f.Pr
        self._delta_t = np.sqrt(2 * alpha / omega)
        
        # Acoustic wavenumber (with small damping)
        self._k_a = omega / f.c * np.sqrt(1 + 1j * f.loss_factor)
        
        # Viscous wavenumber: k_v = (1+i)/δ_v
        self._k_v = (1 + 1j) / self._delta_v
        
        # Thermal wavenumber: k_t = (1+i)/δ_t
        self._k_t = (1 + 1j) / self._delta_t
    
    @property
    def delta_v(self) -> float:
        """Viscous boundary layer thickness [m]."""
        return self._delta_v
    
    @property
    def delta_t(self) -> float:
        """Thermal boundary layer thickness [m]."""
        return self._delta_t
    
    @property
    def k_acoustic(self) -> complex:
        """Acoustic wavenumber [1/m]."""
        return self._k_a
    
    @property
    def k_viscous(self) -> complex:
        """Viscous wavenumber [1/m]."""
        return self._k_v
    
    @property
    def k_thermal(self) -> complex:
        """Thermal wavenumber [1/m]."""
        return self._k_t


@dataclass 
class ThermoviscousLayer:
    """
    Represents a thin thermoviscous layer near a solid boundary.
    
    This layer provides corrections to the pressure and velocity fields
    to account for viscous and thermal effects in the acoustic boundary layer.
    """
    params: ThermoviscousParameters
    
    # Wall normal direction and position
    wall_normal: np.ndarray  # Unit normal pointing into fluid
    wall_position: np.ndarray  # Point on wall
    
    # Layer extent (typically a few δ_v)
    layer_thickness_factor: float = 5.0  # Extend to 5*max(δ_v, δ_t)
    
    @property
    def layer_thickness(self) -> float:
        """Effective thickness of thermoviscous layer [m]."""
        return self.layer_thickness_factor * max(self.params.delta_v, self.params.delta_t)
    
    def distance_from_wall(self, point: np.ndarray) -> float:
        """
        Compute perpendicular distance from wall.
        
        Parameters
        ----------
        point : np.ndarray
            Position vector [x, y, z].
        
        Returns
        -------
        d : float
            Distance from wall [m]. Positive into fluid.
        """
        return np.dot(point - self.wall_position, self.wall_normal)
    
    def is_in_layer(self, point: np.ndarray) -> bool:
        """Check if point is within thermoviscous layer."""
        d = self.distance_from_wall(point)
        return 0 < d < self.layer_thickness
    
    def viscous_decay_factor(self, d: float) -> complex:
        """
        Compute viscous mode amplitude decay with distance from wall.
        
        The viscous mode decays as exp(-k_v * d) where k_v = (1+i)/δ_v.
        
        Parameters
        ----------
        d : float
            Distance from wall [m].
        
        Returns
        -------
        factor : complex
            Decay factor for viscous mode.
        """
        return np.exp(-self.params.k_viscous * d)
    
    def thermal_decay_factor(self, d: float) -> complex:
        """
        Compute thermal mode amplitude decay with distance from wall.
        """
        return np.exp(-self.params.k_thermal * d)
    
    def velocity_correction(
        self,
        d: float,
        v_acoustic_tangent: complex,
        v_wall_tangent: complex = 0.0,
    ) -> complex:
        """
        Compute viscous correction to tangential velocity near wall.
        
        In the viscous boundary layer, the tangential velocity must transition
        from the acoustic velocity (far from wall) to the wall velocity.
        
        v_tangent(d) = v_acoustic - (v_acoustic - v_wall) * exp(-k_v * d)
        
        Parameters
        ----------
        d : float
            Distance from wall [m].
        v_acoustic_tangent : complex
            Tangential acoustic velocity phasor far from wall.
        v_wall_tangent : complex
            Wall tangential velocity (usually 0 for rigid wall).
        
        Returns
        -------
        v_corrected : complex
            Corrected tangential velocity.
        """
        decay = self.viscous_decay_factor(d)
        return v_acoustic_tangent - (v_acoustic_tangent - v_wall_tangent) * decay
    
    def pressure_correction(
        self,
        d: float,
        p_acoustic: complex,
        T_wall: complex = 0.0,
    ) -> complex:
        """
        Compute thermal correction to pressure near wall (if isothermal BC).
        
        For an isothermal wall, the temperature fluctuation must vanish,
        introducing a thermal mode that corrects the pressure.
        
        Parameters
        ----------
        d : float
            Distance from wall [m].
        p_acoustic : complex
            Acoustic pressure phasor.
        T_wall : complex
            Wall temperature fluctuation (0 for isothermal).
        
        Returns
        -------
        p_corrected : complex
            Corrected pressure including thermal mode.
        """
        # For isothermal wall, pressure correction is related to 
        # temperature fluctuation in the acoustic wave
        gamma = self.params.fluid.gamma
        if gamma == 1.0:
            # Incompressible limit, no thermal correction
            return p_acoustic
        
        # Thermal correction factor
        decay = self.thermal_decay_factor(d)
        # The correction is proportional to (γ-1)/γ * p * exp(-k_t*d)
        thermal_correction = -(gamma - 1) / gamma * p_acoustic * decay
        
        return p_acoustic + thermal_correction
    
    def dissipation_power(
        self,
        p_acoustic: complex,
        v_tangent: complex,
    ) -> float:
        """
        Compute time-averaged power dissipation per unit wall area.
        
        Integrates dissipation through the boundary layer.
        
        Parameters
        ----------
        p_acoustic : complex
            Acoustic pressure amplitude.
        v_tangent : complex
            Tangential acoustic velocity amplitude.
        
        Returns
        -------
        P_diss : float
            Dissipated power per unit area [W/m²].
        """
        f = self.params.fluid
        omega = self.params.omega
        delta_v = self.params.delta_v
        delta_t = self.params.delta_t
        
        # Viscous dissipation: (1/4) * η * |v_tangent|² / δ_v
        P_viscous = 0.25 * f.eta * np.abs(v_tangent)**2 / delta_v
        
        # Thermal dissipation (for non-isentropic)
        if f.gamma > 1.0:
            # Thermal dissipation: (γ-1)/(4γ) * (ω/c²) * α * |p|²
            alpha = delta_t**2 * omega / 2
            P_thermal = (f.gamma - 1) / (4 * f.gamma) * omega / f.c**2 * alpha * np.abs(p_acoustic)**2
        else:
            P_thermal = 0.0
        
        return P_viscous + P_thermal


class ThermoviscousCorrector:
    """
    Applies thermoviscous corrections to acoustic fields near boundaries.
    
    This class manages multiple ThermoviscousLayer objects and applies
    corrections to pressure and velocity fields throughout a domain.
    """
    
    def __init__(
        self,
        fluid: FluidMaterial,
        omega: float,
    ):
        """
        Initialize corrector.
        
        Parameters
        ----------
        fluid : FluidMaterial
            Fluid properties.
        omega : float
            Angular frequency [rad/s].
        """
        self.params = ThermoviscousParameters(fluid=fluid, omega=omega)
        self.layers: list[ThermoviscousLayer] = []
    
    def add_planar_wall(
        self,
        normal: np.ndarray,
        position: np.ndarray,
        extent: float = 5.0,
    ):
        """
        Add a planar wall for thermoviscous corrections.
        
        Parameters
        ----------
        normal : np.ndarray
            Unit normal vector pointing into fluid.
        position : np.ndarray
            A point on the wall.
        extent : float
            Layer thickness in multiples of boundary layer thickness.
        """
        normal = np.asarray(normal) / np.linalg.norm(normal)
        layer = ThermoviscousLayer(
            params=self.params,
            wall_normal=normal,
            wall_position=np.asarray(position),
            layer_thickness_factor=extent,
        )
        self.layers.append(layer)
    
    def add_cylindrical_wall(
        self,
        center: np.ndarray,
        radius: float,
        z_min: float,
        z_max: float,
        is_inner: bool = True,
        extent: float = 5.0,
    ):
        """
        Add a cylindrical wall (approximated by many planar facets).
        
        Parameters
        ----------
        center : np.ndarray
            Center of cylinder [x, y].
        radius : float
            Radius of cylinder.
        z_min, z_max : float
            Vertical extent of cylinder.
        is_inner : bool
            If True, fluid is inside (normal points inward).
            If False, fluid is outside (normal points outward).
        extent : float
            Layer thickness factor.
        """
        # For cylindrical walls, we'd need special handling
        # For now, approximate with discrete panels or use continuous formula
        # This is a placeholder for full implementation
        pass
    
    def correct_velocity_field(
        self,
        X: np.ndarray,
        Y: np.ndarray, 
        Z: np.ndarray,
        vx: np.ndarray,
        vy: np.ndarray,
        vz: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply thermoviscous corrections to a 3D velocity field.
        
        Parameters
        ----------
        X, Y, Z : np.ndarray
            Coordinate meshgrids, shape (Nx, Ny, Nz).
        vx, vy, vz : np.ndarray
            Velocity field components (complex), shape (Nx, Ny, Nz).
        
        Returns
        -------
        vx_corr, vy_corr, vz_corr : np.ndarray
            Corrected velocity fields.
        """
        vx_corr = vx.copy()
        vy_corr = vy.copy()
        vz_corr = vz.copy()
        
        shape = X.shape
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    point = np.array([X[i,j,k], Y[i,j,k], Z[i,j,k]])
                    v_local = np.array([vx[i,j,k], vy[i,j,k], vz[i,j,k]])
                    
                    # Apply correction from each layer
                    for layer in self.layers:
                        d = layer.distance_from_wall(point)
                        if 0 < d < layer.layer_thickness:
                            n = layer.wall_normal
                            # Decompose velocity into normal and tangent
                            v_normal = np.dot(v_local, n) * n
                            v_tangent = v_local - v_normal
                            v_tangent_mag = np.linalg.norm(v_tangent)
                            
                            if v_tangent_mag > 1e-20:
                                # Apply viscous correction to tangent
                                v_tangent_dir = v_tangent / v_tangent_mag
                                v_tangent_corr = layer.velocity_correction(
                                    d, v_tangent_mag, 0.0
                                )
                                v_local = v_normal + v_tangent_corr * v_tangent_dir
                    
                    vx_corr[i,j,k] = v_local[0]
                    vy_corr[i,j,k] = v_local[1]
                    vz_corr[i,j,k] = v_local[2]
        
        return vx_corr, vy_corr, vz_corr
    
    def compute_total_dissipation(
        self,
        p: np.ndarray,
        vx: np.ndarray,
        vy: np.ndarray,
        vz: np.ndarray,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        dx: float,
        dy: float,
        dz: float,
    ) -> float:
        """
        Compute total power dissipation in thermoviscous layers.
        
        Parameters
        ----------
        p : np.ndarray
            Pressure field (complex).
        vx, vy, vz : np.ndarray
            Velocity field components.
        X, Y, Z : np.ndarray
            Coordinate meshgrids.
        dx, dy, dz : float
            Grid spacing.
        
        Returns
        -------
        P_total : float
            Total dissipated power [W].
        """
        P_total = 0.0
        
        # This is an approximation - integrate dissipation density
        shape = X.shape
        dV = dx * dy * dz
        
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    point = np.array([X[i,j,k], Y[i,j,k], Z[i,j,k]])
                    v_local = np.array([vx[i,j,k], vy[i,j,k], vz[i,j,k]])
                    p_local = p[i,j,k]
                    
                    for layer in self.layers:
                        d = layer.distance_from_wall(point)
                        if 0 < d < layer.layer_thickness:
                            v_tangent = np.linalg.norm(v_local - np.dot(v_local, layer.wall_normal) * layer.wall_normal)
                            # Volumetric dissipation (approximate)
                            P_local = layer.dissipation_power(p_local, v_tangent) / layer.layer_thickness
                            P_total += P_local * dV
        
        return P_total
    
    def report(self) -> str:
        """Generate diagnostic report."""
        lines = [
            "=" * 50,
            "Thermoviscous Acoustics Report",
            "=" * 50,
            f"Frequency: {self.params.omega / (2*np.pi) / 1e6:.2f} MHz",
            f"Viscous BL thickness: δ_v = {self.params.delta_v * 1e6:.3f} μm",
            f"Thermal BL thickness: δ_t = {self.params.delta_t * 1e6:.3f} μm",
            f"Number of wall layers: {len(self.layers)}",
        ]
        
        for i, layer in enumerate(self.layers):
            lines.append(f"  Layer {i}: thickness = {layer.layer_thickness * 1e6:.1f} μm")
        
        lines.append("=" * 50)
        return "\n".join(lines)


if __name__ == "__main__":
    from .materials import MaterialDatabase
    
    # Demo: compute boundary layer thicknesses
    water = MaterialDatabase.water(25.0)
    omega = 2 * np.pi * 1e6  # 1 MHz
    
    delta_v, delta_t = compute_boundary_layer_thickness(water, omega)
    print("Thermoviscous boundary layer thicknesses at 1 MHz:")
    print(f"  δ_v = {delta_v * 1e6:.3f} μm")
    print(f"  δ_t = {delta_t * 1e6:.3f} μm")
    
    # Check significance for 10 μm particle
    significant, ratio = boundary_layer_significance(water, omega, 10e-6)
    print(f"\nFor 10 μm particle:")
    print(f"  δ/a = {ratio:.3f}")
    print(f"  Significant: {significant}")
    
    # Create corrector
    corrector = ThermoviscousCorrector(water, omega)
    corrector.add_planar_wall(
        normal=np.array([0, 0, 1]),
        position=np.array([0, 0, 0]),
    )
    print("\n" + corrector.report())
