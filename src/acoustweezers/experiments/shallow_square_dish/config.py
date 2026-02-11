"""
Configuration for Shallow Square Dish Experiment.

Device-aligned configuration:
- BOTTOM (z=0): Vortex lens actuation (moveable center)
- SIDE WALLS: Standing wave transducers (4 walls)
- TOP (z=H): Free surface / air interface

Author: Acousto-Tweezers Project
Date: 2026-02-08
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, Optional, List
import numpy as np


@dataclass
class ShallowDishConfig:
    """
    Configuration for device-aligned shallow dish simulation.
    
    Physical Setup:
    - Shallow rectangular dish (L × L × H)
    - Bottom transducer with vortex lens pattern
    - Side wall transducers for standing wave
    - Top is air interface (free surface)
    """
    
    # ==========================================================================
    # GEOMETRY
    # ==========================================================================
    L: float = 0.05          # Dish lateral size [m] (5 cm)
    H: float = 0.005         # Dish depth [m] (5 mm)
    
    # ==========================================================================
    # FREQUENCY
    # ==========================================================================
    frequency_hz: float = 500e3  # Operating frequency [Hz]
    
    # ==========================================================================
    # MATERIAL PROPERTIES (Water at 20°C)
    # ==========================================================================
    rho: float = 997.0       # Density [kg/m³]
    c: float = 1484.0        # Sound speed [m/s]
    mu: float = 1.002e-3     # Dynamic viscosity [Pa·s]
    
    # ==========================================================================
    # VORTEX ACTUATION (BOTTOM BOUNDARY)
    # ==========================================================================
    vortex_velocity_amplitude: float = 10e-6   # V_vtx [m/s]
    vortex_topological_charge: int = 1         # ℓ (azimuthal mode)
    vortex_aperture_radius: float = 0.004      # Aperture radius [m] (4 mm)
    vortex_apodization: str = "cosine_taper"   # Amplitude profile
    vortex_phase_offset: float = 0.0           # φ₀ [rad]
    
    # Vortex center can be moved (for path tracking)
    vortex_center_x: float = None  # [m], None = centered at L/2
    vortex_center_y: float = None  # [m], None = centered at L/2
    
    # ==========================================================================
    # STANDING WAVE ACTUATION (SIDE WALLS)
    # ==========================================================================
    standing_velocity_amplitude: float = 1e-6  # V_stand [m/s]
    standing_phase_pattern: str = "antiphase"  # "antiphase", "quadrature", "inphase"
    standing_axis: str = "x"                   # "x", "y", or "both"
    
    # Aperture (if not full wall)
    standing_full_wall: bool = True            # If False, use aperture mask
    standing_aperture_height: float = None     # [m], active region height
    standing_aperture_y_range: Tuple[float, float] = None  # (y_min, y_max)
    
    # ==========================================================================
    # BOUNDARY CONDITIONS
    # ==========================================================================
    # Top boundary (air interface)
    top_bc_type: str = "impedance"             # "impedance" or "dirichlet"
    top_impedance_factor: float = 0.001        # Z_top = factor * Z_water
    
    # Bottom disc: circular transducer patch on z=0 floor.
    # Inside disc = impedance Robin (Z_water) + vortex source when active.
    # Outside disc = rigid (natural Neumann, ∂p/∂n = 0).
    # Radius defaults to vortex_aperture_radius if None.
    bottom_disc_radius: float = None           # [m], None → vortex_aperture_radius
    
    # Side walls (x±, y±) are ALWAYS rigid when inactive (natural Neumann)
    # and pure Neumann source when active. NEVER impedance-matched.
    
    # ==========================================================================
    # MESH
    # ==========================================================================
    elements_per_wavelength: int = 6           # Mesh density
    min_elements_z: int = 10                   # Minimum elements in z
    
    # ==========================================================================
    # PARTICLE PROPERTIES (Polystyrene)
    # ==========================================================================
    particle_radius: float = 5e-6              # a [m] (5 μm)
    particle_density: float = 1050.0           # ρ_p [kg/m³]
    particle_compressibility: float = 2.4e-10  # κ_p [Pa⁻¹]
    
    # ==========================================================================
    # SIMULATION
    # ==========================================================================
    # Vortex path for trajectory simulation
    vortex_path_type: str = "fixed"            # "fixed", "line", "circle"
    vortex_path_n_steps: int = 20              # Steps along path
    vortex_path_endpoints: Tuple[Tuple[float, float], Tuple[float, float]] = None
    
    # Particle integration
    particle_dt: float = 1e-5                  # Time step [s]
    particle_t_max: float = 0.1                # Max integration time [s]
    
    # ==========================================================================
    # DERIVED PROPERTIES
    # ==========================================================================
    
    @property
    def omega(self) -> float:
        """Angular frequency [rad/s]."""
        return 2 * np.pi * self.frequency_hz
    
    @property
    def k(self) -> float:
        """Wavenumber [rad/m]."""
        return self.omega / self.c
    
    @property
    def wavelength(self) -> float:
        """Acoustic wavelength [m]."""
        return self.c / self.frequency_hz
    
    @property
    def Z_water(self) -> float:
        """Acoustic impedance of water [Pa·s/m]."""
        return self.rho * self.c
    
    @property
    def Z_top(self) -> float:
        """Effective impedance at top boundary [Pa·s/m]."""
        return self.top_impedance_factor * self.Z_water
    
    @property
    def bottom_disc_radius_effective(self) -> float:
        """Effective bottom disc radius [m]."""
        if self.bottom_disc_radius is not None:
            return self.bottom_disc_radius
        return self.vortex_aperture_radius

    @property
    def vortex_center(self) -> np.ndarray:
        """Vortex center coordinates [m]."""
        cx = self.vortex_center_x if self.vortex_center_x is not None else self.L / 2
        cy = self.vortex_center_y if self.vortex_center_y is not None else self.L / 2
        return np.array([cx, cy, 0.0])
    
    @property
    def mesh_nx(self) -> int:
        """Number of elements in x/y direction."""
        return max(20, int(self.L / self.wavelength * self.elements_per_wavelength))
    
    @property
    def mesh_nz(self) -> int:
        """Number of elements in z direction."""
        nz_from_wavelength = int(self.H / self.wavelength * self.elements_per_wavelength)
        return max(self.min_elements_z, nz_from_wavelength)
    
    @property
    def n_wavelengths_lateral(self) -> float:
        """Number of wavelengths fitting laterally."""
        return self.L / self.wavelength
    
    @property
    def n_wavelengths_depth(self) -> float:
        """Number of wavelengths fitting in depth."""
        return self.H / self.wavelength
    
    @property
    def fluid_bulk_modulus(self) -> float:
        """Bulk modulus of water [Pa]."""
        return self.rho * self.c**2
    
    @property
    def fluid_compressibility(self) -> float:
        """Compressibility of water [Pa⁻¹]."""
        return 1.0 / self.fluid_bulk_modulus
    
    @property
    def f1_monopole(self) -> float:
        """Monopole contrast factor."""
        return 1.0 - self.particle_compressibility / self.fluid_compressibility
    
    @property
    def f2_dipole(self) -> float:
        """Dipole contrast factor."""
        return 2.0 * (self.particle_density - self.rho) / (2*self.particle_density + self.rho)
    
    @property
    def particle_volume(self) -> float:
        """Particle volume [m³]."""
        return (4.0/3.0) * np.pi * self.particle_radius**3
    
    @property
    def stokes_mobility(self) -> float:
        """Stokes mobility μ = 1/(6πηa) [m/(N·s)]."""
        return 1.0 / (6 * np.pi * self.mu * self.particle_radius)
    
    def get_vortex_path(self) -> np.ndarray:
        """
        Get vortex center positions for path tracking.
        
        Returns
        -------
        np.ndarray
            Array of (x, y) positions, shape (n_steps, 2)
        """
        if self.vortex_path_type == "fixed":
            cx = self.vortex_center_x if self.vortex_center_x is not None else self.L/2
            cy = self.vortex_center_y if self.vortex_center_y is not None else self.L/2
            return np.array([[cx, cy]])
        
        elif self.vortex_path_type == "line":
            if self.vortex_path_endpoints is None:
                # Default: sweep along x from 0.2L to 0.8L
                start = (0.2 * self.L, self.L / 2)
                end = (0.8 * self.L, self.L / 2)
            else:
                start, end = self.vortex_path_endpoints
            
            x = np.linspace(start[0], end[0], self.vortex_path_n_steps)
            y = np.linspace(start[1], end[1], self.vortex_path_n_steps)
            return np.column_stack([x, y])
        
        elif self.vortex_path_type == "circle":
            # Circular path around center
            center = np.array([self.L/2, self.L/2])
            radius = 0.1 * self.L
            theta = np.linspace(0, 2*np.pi, self.vortex_path_n_steps, endpoint=False)
            x = center[0] + radius * np.cos(theta)
            y = center[1] + radius * np.sin(theta)
            return np.column_stack([x, y])
        
        else:
            raise ValueError(f"Unknown vortex_path_type: {self.vortex_path_type}")
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        d = {}
        for key in self.__dataclass_fields__:
            val = getattr(self, key)
            if val is None:
                d[key] = None
            elif isinstance(val, np.ndarray):
                d[key] = val.tolist()
            elif isinstance(val, tuple):
                d[key] = list(val)
            else:
                d[key] = val
        return d
    
    def describe(self) -> str:
        """Return human-readable description of configuration."""
        return f"""
Shallow Dish Configuration
===========================
Geometry:     {self.L*1e3:.1f} mm × {self.L*1e3:.1f} mm × {self.H*1e3:.1f} mm
Frequency:    {self.frequency_hz/1e3:.0f} kHz
Wavelength:   {self.wavelength*1e3:.2f} mm
Depth/λ:      {self.n_wavelengths_depth:.2f}

Vortex (bottom):
  Amplitude:  {self.vortex_velocity_amplitude*1e6:.1f} μm/s
  Charge ℓ:   {self.vortex_topological_charge}
  Aperture:   {self.vortex_aperture_radius*1e3:.1f} mm radius

Standing (sides):
  Amplitude:  {self.standing_velocity_amplitude*1e6:.1f} μm/s
  Pattern:    {self.standing_phase_pattern}
  Axis:       {self.standing_axis}

BCs:
  Top:        impedance Robin (Z/Z_water = {self.top_impedance_factor})
  Side walls: rigid (inactive) / Neumann source (active)
  Bottom:     disc R={self.bottom_disc_radius_effective*1e3:.1f} mm = impedance + vortex
              remainder = rigid

Mesh: {self.mesh_nx}×{self.mesh_nx}×{self.mesh_nz}

Particle:
  Radius:     {self.particle_radius*1e6:.1f} μm
  Density:    {self.particle_density:.0f} kg/m³
  f1 (mono):  {self.f1_monopole:.3f}
  f2 (dipo):  {self.f2_dipole:.3f}
"""


# Preset configurations
def get_default_config() -> ShallowDishConfig:
    """Get default device configuration."""
    return ShallowDishConfig()


def get_high_vortex_config() -> ShallowDishConfig:
    """Configuration with stronger vortex for visualization."""
    return ShallowDishConfig(
        vortex_velocity_amplitude=50e-6,
        standing_velocity_amplitude=1e-6,
    )


def get_path_tracking_config() -> ShallowDishConfig:
    """Configuration for vortex path tracking demo."""
    return ShallowDishConfig(
        vortex_path_type="line",
        vortex_path_n_steps=20,
        particle_t_max=0.2,
    )
