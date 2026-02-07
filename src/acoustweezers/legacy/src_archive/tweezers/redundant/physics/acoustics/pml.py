"""
Perfectly Matched Layer (PML) implementation for open-domain acoustics.

The PML absorbs outgoing waves without reflection by applying complex 
coordinate stretching. This implementation supports:
- Cartesian PML in x, y, z directions
- Configurable profile (polynomial, exponential)
- Frequency-dependent parameters
- Integration with the multi-domain Helmholtz solver

Theory:
-------
In PML regions, the coordinate x is mapped to:
    x̃ = x - (i/ω) ∫₀ˣ σ(x') dx'

For the Helmholtz equation, this results in modified operators:
    ∂/∂x → (1/s_x) ∂/∂x
    
where s_x = 1 + i*σ_x/ω is the complex stretching function.

The PML Helmholtz equation becomes:
    ∂/∂x(1/(ρ·s_x) ∂p/∂x) + ... + (ω²/K)p = 0
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Tuple, Optional, Callable
import numpy as np


class PMLProfile(Enum):
    """PML absorption profile types."""
    POLYNOMIAL = auto()    # σ(d) = σ_max * (d/L)^n
    EXPONENTIAL = auto()   # σ(d) = σ_max * (1 - exp(-α*d/L))
    COSINE = auto()        # Smooth cosine ramp
    

@dataclass
class PMLParameters:
    """
    Parameters defining PML behavior.
    
    Parameters
    ----------
    thickness : float
        PML thickness [m].
    sigma_max : float
        Maximum absorption coefficient [1/s].
        Typical: σ_max ≈ (n+1) * c * log(R₀) / (2 * L)
        where R₀ is the target reflection coefficient.
    n : int
        Polynomial order (for POLYNOMIAL profile).
    R0 : float
        Target reflection coefficient (for automatic σ_max calculation).
    profile : PMLProfile
        Type of absorption profile.
    """
    thickness: float = 0.005        # 5mm default
    sigma_max: Optional[float] = None  # If None, computed from R0
    n: int = 2                      # Polynomial order
    R0: float = 1e-6               # Target reflection
    profile: PMLProfile = PMLProfile.POLYNOMIAL
    
    def compute_sigma_max(self, c: float) -> float:
        """
        Compute optimal σ_max for given sound speed and target reflection.
        
        σ_max = (n+1) * c * ln(1/R₀) / (2*L)
        
        Parameters
        ----------
        c : float
            Sound speed in adjacent medium [m/s].
        
        Returns
        -------
        sigma_max : float
            Maximum absorption coefficient [1/s].
        """
        if self.sigma_max is not None:
            return self.sigma_max
        return (self.n + 1) * c * np.log(1.0 / self.R0) / (2 * self.thickness)


@dataclass
class PMLRegion:
    """
    A single PML region attached to a domain boundary.
    
    Handles computation of complex stretching functions for finite difference
    discretization of the PML Helmholtz equation.
    """
    # PML parameters
    params: PMLParameters
    
    # Direction and position
    direction: str          # 'x', 'y', or 'z'
    side: str              # 'min' or 'max' (which boundary)
    boundary_position: float  # Position of inner PML boundary
    
    # Material properties of adjacent fluid
    c: float = 1500.0       # Sound speed [m/s]
    rho: float = 1000.0     # Density [kg/m³]
    
    def __post_init__(self):
        """Compute derived quantities."""
        self.sigma_max = self.params.compute_sigma_max(self.c)
        
        # PML extent
        if self.side == 'min':
            self.outer_position = self.boundary_position - self.params.thickness
        else:
            self.outer_position = self.boundary_position + self.params.thickness
    
    def distance_into_pml(self, coord: float) -> float:
        """
        Compute distance from PML inner boundary into the PML.
        
        Returns 0 if outside PML, positive value inside.
        """
        if self.side == 'min':
            if coord >= self.boundary_position:
                return 0.0
            return self.boundary_position - coord
        else:
            if coord <= self.boundary_position:
                return 0.0
            return coord - self.boundary_position
    
    def is_in_pml(self, coord: float) -> bool:
        """Check if coordinate is within this PML region."""
        return self.distance_into_pml(coord) > 0
    
    def sigma(self, coord: float) -> float:
        """
        Compute absorption coefficient σ at given coordinate.
        
        Parameters
        ----------
        coord : float
            Coordinate in PML direction.
        
        Returns
        -------
        sigma : float
            Absorption coefficient [1/s].
        """
        d = self.distance_into_pml(coord)
        if d <= 0:
            return 0.0
        
        L = self.params.thickness
        d_norm = min(d / L, 1.0)  # Normalized distance [0, 1]
        
        if self.params.profile == PMLProfile.POLYNOMIAL:
            return self.sigma_max * d_norm ** self.params.n
        elif self.params.profile == PMLProfile.EXPONENTIAL:
            alpha = 3.0  # Exponential rate parameter
            return self.sigma_max * (1 - np.exp(-alpha * d_norm))
        elif self.params.profile == PMLProfile.COSINE:
            return self.sigma_max * 0.5 * (1 - np.cos(np.pi * d_norm))
        else:
            return self.sigma_max * d_norm ** self.params.n
    
    def stretching_function(self, coord: float, omega: float) -> complex:
        """
        Compute complex stretching function s = 1 + i*σ/ω.
        
        Parameters
        ----------
        coord : float
            Coordinate value.
        omega : float
            Angular frequency [rad/s].
        
        Returns
        -------
        s : complex
            Complex stretching function.
        """
        sig = self.sigma(coord)
        return 1.0 + 1j * sig / omega
    
    def stretching_array(self, coords: np.ndarray, omega: float) -> np.ndarray:
        """
        Compute stretching function for array of coordinates.
        """
        s = np.ones(len(coords), dtype=np.complex128)
        for i, coord in enumerate(coords):
            s[i] = self.stretching_function(coord, omega)
        return s


class PMLManager:
    """
    Manages all PML regions for a 3D domain.
    
    Creates and coordinates PML regions on all 6 faces of the domain.
    Provides methods to compute the effective material properties (ρ̃, K̃)
    at each grid point including PML stretching.
    """
    
    def __init__(
        self,
        Lx: float, Ly: float, Lz: float,
        x: np.ndarray, y: np.ndarray, z: np.ndarray,
        params: PMLParameters,
        c_default: float = 1500.0,
        rho_default: float = 1000.0,
    ):
        """
        Initialize PML manager.
        
        Parameters
        ----------
        Lx, Ly, Lz : float
            Domain extents [m].
        x, y, z : np.ndarray
            Coordinate arrays.
        params : PMLParameters
            PML configuration.
        c_default : float
            Default sound speed for PML (used if region-specific not set).
        rho_default : float
            Default density for PML.
        """
        self.Lx, self.Ly, self.Lz = Lx, Ly, Lz
        self.x, self.y, self.z = x, y, z
        self.params = params
        self.c_default = c_default
        self.rho_default = rho_default
        
        # Create PML regions for all 6 faces
        self.regions = {}
        self._create_regions()
    
    def _create_regions(self):
        """Create PML regions on all faces."""
        # X boundaries
        self.regions['x_min'] = PMLRegion(
            params=self.params,
            direction='x',
            side='min',
            boundary_position=self.params.thickness,
            c=self.c_default,
            rho=self.rho_default,
        )
        self.regions['x_max'] = PMLRegion(
            params=self.params,
            direction='x',
            side='max',
            boundary_position=self.Lx - self.params.thickness,
            c=self.c_default,
            rho=self.rho_default,
        )
        
        # Y boundaries
        self.regions['y_min'] = PMLRegion(
            params=self.params,
            direction='y',
            side='min',
            boundary_position=self.params.thickness,
            c=self.c_default,
            rho=self.rho_default,
        )
        self.regions['y_max'] = PMLRegion(
            params=self.params,
            direction='y',
            side='max',
            boundary_position=self.Ly - self.params.thickness,
            c=self.c_default,
            rho=self.rho_default,
        )
        
        # Z boundaries
        self.regions['z_min'] = PMLRegion(
            params=self.params,
            direction='z',
            side='min',
            boundary_position=self.params.thickness,
            c=self.c_default,
            rho=self.rho_default,
        )
        self.regions['z_max'] = PMLRegion(
            params=self.params,
            direction='z',
            side='max',
            boundary_position=self.Lz - self.params.thickness,
            c=self.c_default,
            rho=self.rho_default,
        )
    
    def set_region_material(self, region_name: str, c: float, rho: float):
        """Set material properties for a specific PML region."""
        if region_name in self.regions:
            self.regions[region_name].c = c
            self.regions[region_name].rho = rho
            self.regions[region_name].sigma_max = self.regions[region_name].params.compute_sigma_max(c)
    
    def get_stretching_functions_1d(self, omega: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute 1D stretching functions for each coordinate direction.
        
        Returns
        -------
        s_x, s_y, s_z : np.ndarray
            Complex stretching functions for each direction.
            Shape: (Nx,), (Ny,), (Nz,)
        """
        s_x = np.ones(len(self.x), dtype=np.complex128)
        s_y = np.ones(len(self.y), dtype=np.complex128)
        s_z = np.ones(len(self.z), dtype=np.complex128)
        
        # X direction
        for i, xi in enumerate(self.x):
            if self.regions['x_min'].is_in_pml(xi):
                s_x[i] = self.regions['x_min'].stretching_function(xi, omega)
            elif self.regions['x_max'].is_in_pml(xi):
                s_x[i] = self.regions['x_max'].stretching_function(xi, omega)
        
        # Y direction
        for j, yj in enumerate(self.y):
            if self.regions['y_min'].is_in_pml(yj):
                s_y[j] = self.regions['y_min'].stretching_function(yj, omega)
            elif self.regions['y_max'].is_in_pml(yj):
                s_y[j] = self.regions['y_max'].stretching_function(yj, omega)
        
        # Z direction
        for k, zk in enumerate(self.z):
            if self.regions['z_min'].is_in_pml(zk):
                s_z[k] = self.regions['z_min'].stretching_function(zk, omega)
            elif self.regions['z_max'].is_in_pml(zk):
                s_z[k] = self.regions['z_max'].stretching_function(zk, omega)
        
        return s_x, s_y, s_z
    
    def get_stretching_functions_3d(self, omega: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute 3D stretching function arrays.
        
        Returns full 3D arrays of stretching functions at each grid point.
        
        Returns
        -------
        S_x, S_y, S_z : np.ndarray
            Complex stretching functions. Shape: (Nx, Ny, Nz)
        """
        s_x, s_y, s_z = self.get_stretching_functions_1d(omega)
        
        Nx, Ny, Nz = len(self.x), len(self.y), len(self.z)
        S_x = np.broadcast_to(s_x[:, np.newaxis, np.newaxis], (Nx, Ny, Nz)).copy()
        S_y = np.broadcast_to(s_y[np.newaxis, :, np.newaxis], (Nx, Ny, Nz)).copy()
        S_z = np.broadcast_to(s_z[np.newaxis, np.newaxis, :], (Nx, Ny, Nz)).copy()
        
        return S_x, S_y, S_z
    
    def get_pml_mask(self) -> np.ndarray:
        """
        Get boolean mask indicating which grid points are in PML.
        
        Returns
        -------
        mask : np.ndarray of bool, shape (Nx, Ny, Nz)
        """
        Nx, Ny, Nz = len(self.x), len(self.y), len(self.z)
        mask = np.zeros((Nx, Ny, Nz), dtype=bool)
        
        for i, xi in enumerate(self.x):
            if self.regions['x_min'].is_in_pml(xi) or self.regions['x_max'].is_in_pml(xi):
                mask[i, :, :] = True
        
        for j, yj in enumerate(self.y):
            if self.regions['y_min'].is_in_pml(yj) or self.regions['y_max'].is_in_pml(yj):
                mask[:, j, :] = True
        
        for k, zk in enumerate(self.z):
            if self.regions['z_min'].is_in_pml(zk) or self.regions['z_max'].is_in_pml(zk):
                mask[:, :, k] = True
        
        return mask
    
    def get_physical_domain_bounds(self) -> Tuple[float, float, float, float, float, float]:
        """
        Get bounds of the physical (non-PML) domain.
        
        Returns
        -------
        (x_min, x_max, y_min, y_max, z_min, z_max) : Tuple[float, ...]
        """
        L = self.params.thickness
        return (L, self.Lx - L, L, self.Ly - L, L, self.Lz - L)
    
    def pml_report(self) -> str:
        """
        Generate a diagnostic report of PML configuration.
        """
        lines = [
            "=" * 50,
            "PML Configuration Report",
            "=" * 50,
            f"Profile: {self.params.profile.name}",
            f"Polynomial order: n = {self.params.n}",
            f"Thickness: L = {self.params.thickness*1e3:.2f} mm",
            f"Target reflection: R₀ = {self.params.R0:.2e}",
            "",
            "Per-region σ_max values:",
        ]
        
        for name, region in self.regions.items():
            lines.append(f"  {name}: σ_max = {region.sigma_max:.2e} 1/s")
        
        # Estimate reflection for each region
        lines.append("")
        lines.append("Estimated theoretical reflection:")
        for name, region in self.regions.items():
            # For polynomial profile: R ≈ exp(-2*σ_max*L/(n+1)/c)
            L = self.params.thickness
            n = self.params.n
            R = np.exp(-2 * region.sigma_max * L / (n + 1) / region.c)
            lines.append(f"  {name}: R ≈ {R:.2e}")
        
        # Physical domain bounds
        bounds = self.get_physical_domain_bounds()
        lines.append("")
        lines.append("Physical domain (excluding PML):")
        lines.append(f"  x: [{bounds[0]*1e3:.2f}, {bounds[1]*1e3:.2f}] mm")
        lines.append(f"  y: [{bounds[2]*1e3:.2f}, {bounds[3]*1e3:.2f}] mm")
        lines.append(f"  z: [{bounds[4]*1e3:.2f}, {bounds[5]*1e3:.2f}] mm")
        
        lines.append("=" * 50)
        return "\n".join(lines)


def compute_pml_reflection_test(
    pml: PMLManager,
    omega: float,
    direction: str = 'x',
    n_points: int = 1000,
) -> float:
    """
    Estimate PML reflection coefficient numerically by integrating absorption.
    
    For a plane wave propagating into PML, the amplitude reduction is:
        A(x) = A₀ * exp(-∫₀ˣ σ(x')/c dx')
    
    The reflection coefficient is approximately twice the one-way transmission
    through the PML (wave travels in and reflects back).
    
    Parameters
    ----------
    pml : PMLManager
        PML configuration.
    omega : float
        Angular frequency [rad/s].
    direction : str
        Direction to test ('x', 'y', or 'z').
    n_points : int
        Number of integration points.
    
    Returns
    -------
    R : float
        Estimated reflection coefficient.
    """
    L = pml.params.thickness
    
    if direction == 'x':
        region = pml.regions['x_max']
        coords = np.linspace(region.boundary_position, region.outer_position, n_points)
    elif direction == 'y':
        region = pml.regions['y_max']
        coords = np.linspace(region.boundary_position, region.outer_position, n_points)
    else:
        region = pml.regions['z_max']
        coords = np.linspace(region.boundary_position, region.outer_position, n_points)
    
    # Integrate σ/c
    dx = coords[1] - coords[0]
    integral = 0.0
    for coord in coords:
        integral += region.sigma(coord) / region.c * dx
    
    # Two-way transmission = reflection
    R = np.exp(-2 * integral)
    return R


if __name__ == "__main__":
    # Demo: create and test PML
    params = PMLParameters(
        thickness=0.005,  # 5mm
        R0=1e-6,
        n=2,
        profile=PMLProfile.POLYNOMIAL,
    )
    
    x = np.linspace(0, 0.05, 101)
    y = np.linspace(0, 0.05, 101)
    z = np.linspace(0, 0.03, 61)
    
    pml = PMLManager(
        Lx=0.05, Ly=0.05, Lz=0.03,
        x=x, y=y, z=z,
        params=params,
        c_default=1500.0,  # Water
        rho_default=1000.0,
    )
    
    # Set air properties for top PML
    pml.set_region_material('z_max', c=343.0, rho=1.2)
    
    print(pml.pml_report())
    
    # Test reflection
    omega = 2 * np.pi * 1e6  # 1 MHz
    R = compute_pml_reflection_test(pml, omega, 'x')
    print(f"\nNumerical reflection test (x): R = {R:.2e}")
    
    # Plot σ profile
    import matplotlib.pyplot as plt
    
    xi = np.linspace(0, 0.01, 200)
    sigma_vals = [pml.regions['x_min'].sigma(x) for x in xi]
    
    plt.figure(figsize=(8, 4))
    plt.plot(xi * 1e3, sigma_vals)
    plt.xlabel('x [mm]')
    plt.ylabel('σ [1/s]')
    plt.title('PML absorption profile')
    plt.axvline(params.thickness * 1e3, color='r', linestyle='--', label='PML boundary')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('/tmp/pml_profile.png', dpi=100)
    print("\nSaved PML profile plot to /tmp/pml_profile.png")
