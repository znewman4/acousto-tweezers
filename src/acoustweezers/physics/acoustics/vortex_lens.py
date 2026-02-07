"""
Acoustic Vortex Lens Boundary Actuation
========================================

Implements azimuthal phase winding for vortex beam generation:

    φ(θ) = ℓθ

where:
- θ is azimuthal angle relative to vortex axis
- ℓ is integer topological charge (±1, ±2, ...)

The vortex lens acts as a boundary field generator:
    p_b(x) = A(x) exp(iφ(x))

This creates a pressure null at the vortex core with characteristic
phase singularity and helical wavefronts.

Physics Notes
-------------
- Topological charge ℓ determines phase winding: 2π per rotation
- Amplitude can be uniform or radially apodized
- Superimposes linearly with other boundary fields
- No volumetric forcing - actuation is boundary-only

Author: Acousto-Tweezers Project
Date: February 2026
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Callable
import numpy as np

from dolfinx import fem
from dolfinx.fem import Function
import ufl


@dataclass
class VortexLensConfig:
    """Configuration for acoustic vortex lens boundary actuation.
    
    Parameters
    ----------
    topological_charge : int
        Topological charge ℓ determining phase winding φ(θ) = ℓθ
        - ℓ = +1: single counterclockwise vortex
        - ℓ = -1: single clockwise vortex
        - |ℓ| > 1: higher-order vortices
        
    center : tuple
        (x_center, y_center, z_center) coordinates of vortex axis
        Default: None (uses domain center)
        
    amplitude : float
        Base amplitude A₀ in Pa
        
    aperture_radius : float
        Finite aperture radius (m). Actuation is localized within this radius
        from the vortex center. Outside this radius, amplitude tapers to zero.
        Default: None (full-boundary actuation, legacy behavior)
        
    apodization : str
        Radial amplitude profile:
        - 'uniform': A(r) = A₀ within aperture
        - 'gaussian': A(r) = A₀ exp(-r²/w²)
        - 'cosine_taper': A(r) = A₀ cos²(πr/2R) for r < R, smooth to zero
        - 'bessel': A(r) ≈ A₀ J₀(kr) (approximate)
        
    apodization_width : float
        Characteristic width for apodization (m)
        For gaussian: w (1/e² radius)
        For cosine_taper: uses aperture_radius
        Ignored if apodization='uniform'
        
    axis : str
        Vortex axis direction: 'z', 'x', or 'y'
        Default: 'z' (θ measured in xy-plane)
    """
    topological_charge: int = 1
    center: Optional[Tuple[float, float, float]] = None
    amplitude: float = 1e6  # 1 MPa default
    aperture_radius: Optional[float] = None  # Finite aperture (m)
    apodization: str = 'cosine_taper'  # Default smooth taper
    apodization_width: Optional[float] = None
    axis: str = 'z'
    
    def __post_init__(self):
        """Validate configuration."""
        if self.topological_charge == 0:
            raise ValueError("Topological charge must be nonzero integer")
        valid_apodizations = ['uniform', 'gaussian', 'bessel', 'cosine_taper']
        if self.apodization not in valid_apodizations:
            raise ValueError(f"Unknown apodization: {self.apodization}")
        if self.apodization == 'gaussian' and self.apodization_width is None:
            raise ValueError(f"apodization_width required for gaussian")
        if self.axis not in ['x', 'y', 'z']:
            raise ValueError(f"axis must be 'x', 'y', or 'z', got {self.axis}")


def compute_azimuthal_phase(
    coords: np.ndarray,
    config: VortexLensConfig,
    center: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute azimuthal phase φ(θ) = ℓθ at given coordinates.
    
    Parameters
    ----------
    coords : np.ndarray
        Coordinate array, shape (N, 3) for [x, y, z]
    config : VortexLensConfig
        Vortex configuration
    center : np.ndarray, optional
        Vortex center [x_c, y_c, z_c]
        If None, uses config.center or computes from coords
        
    Returns
    -------
    np.ndarray
        Phase values φ at each point (radians)
    """
    if center is None:
        if config.center is not None:
            center = np.array(config.center)
        else:
            # Use centroid of coordinates
            center = np.mean(coords, axis=0)
    
    # Compute relative coordinates
    dx = coords[:, 0] - center[0]
    dy = coords[:, 1] - center[1]
    dz = coords[:, 2] - center[2]
    
    # Compute azimuthal angle based on axis
    if config.axis == 'z':
        # θ in xy-plane
        theta = np.arctan2(dy, dx)
    elif config.axis == 'x':
        # θ in yz-plane
        theta = np.arctan2(dz, dy)
    else:  # 'y'
        # θ in xz-plane
        theta = np.arctan2(dz, dx)
    
    # Apply topological charge
    phase = config.topological_charge * theta
    
    return phase


def compute_amplitude_profile(
    coords: np.ndarray,
    config: VortexLensConfig,
    center: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute radial amplitude profile A(r) with finite aperture taper.
    
    Parameters
    ----------
    coords : np.ndarray
        Coordinate array, shape (N, 3)
    config : VortexLensConfig
        Vortex configuration
    center : np.ndarray, optional
        Vortex center [x_c, y_c, z_c]
        
    Returns
    -------
    np.ndarray
        Amplitude values A at each point (includes aperture taper)
    """
    A0 = config.amplitude
    
    # Compute radial distance from axis
    if center is None:
        if config.center is not None:
            center = np.array(config.center)
        else:
            center = np.mean(coords, axis=0)
    
    dx = coords[:, 0] - center[0]
    dy = coords[:, 1] - center[1]
    dz = coords[:, 2] - center[2]
    
    if config.axis == 'z':
        r = np.sqrt(dx**2 + dy**2)
    elif config.axis == 'x':
        r = np.sqrt(dy**2 + dz**2)
    else:  # 'y'
        r = np.sqrt(dx**2 + dz**2)
    
    # Apply apodization profile
    if config.apodization == 'uniform':
        amplitude = np.full_like(r, A0)
        
    elif config.apodization == 'cosine_taper':
        # Smooth cosine taper: A(r) = A₀ cos²(πr/2R) for r < R
        R = config.aperture_radius if config.aperture_radius else np.max(r) * 2
        taper = np.where(r < R, np.cos(np.pi * r / (2 * R))**2, 0.0)
        amplitude = A0 * taper
        
    elif config.apodization == 'gaussian':
        w = config.apodization_width
        amplitude = A0 * np.exp(-r**2 / w**2)
    
    elif config.apodization == 'bessel':
        # Approximate J₀(kr) with first few terms
        # J₀(x) ≈ 1 - x²/4 + x⁴/64 for small x
        k = 2 * np.pi / config.apodization_width  # approximate wavenumber
        x = k * r
        # Use first 3 terms of series
        bessel = 1.0 - (x**2)/4.0 + (x**4)/64.0
        amplitude = A0 * np.abs(bessel)  # Avoid negatives
    else:
        amplitude = np.full_like(r, A0)
    
    # Apply finite aperture cutoff (if specified and not using cosine_taper)
    if config.aperture_radius is not None and config.apodization != 'cosine_taper':
        R = config.aperture_radius
        taper_width = R * 0.2  # 20% rolloff region
        taper = np.where(r < R, 1.0, 
                        np.where(r < R + taper_width,
                                0.5 * (1 + np.cos(np.pi * (r - R) / taper_width)),
                                0.0))
        amplitude = amplitude * taper
    
    return amplitude


def create_vortex_boundary_function(
    function_space: fem.FunctionSpace,
    facet_indices: np.ndarray,
    config: VortexLensConfig,
    center: Optional[np.ndarray] = None,
    verbose: bool = True
) -> Function:
    """
    Create a DOLFINx Function representing vortex boundary field.
    
    p_vortex(x) = A(x) exp(i φ(x))
    
    where φ(x) = ℓ θ(x) is the azimuthal phase.
    
    Parameters
    ----------
    function_space : dolfinx.fem.FunctionSpace
        Function space for pressure field
    facet_indices : np.ndarray
        Indices of boundary facets where vortex is applied
    config : VortexLensConfig
        Vortex configuration
    center : np.ndarray, optional
        Vortex center coordinates [x_c, y_c, z_c]
    verbose : bool
        Print diagnostic information
        
    Returns
    -------
    dolfinx.fem.Function
        Complex-valued boundary field with vortex structure
        
    Notes
    -----
    The function is defined on DOFs associated with the specified facets.
    For interior DOFs not on these facets, the value is set to zero.
    """
    from mpi4py import MPI
    
    mesh = function_space.mesh
    comm = mesh.comm
    rank = comm.rank
    
    # Create function
    p_vortex = Function(function_space)
    
    # Get DOF coordinates for the boundary facets
    # We need to identify which DOFs are on these facets
    from dolfinx.fem import locate_dofs_topological
    
    tdim = mesh.topology.dim
    fdim = tdim - 1
    
    # Locate DOFs on specified facets
    dofs = locate_dofs_topological(function_space, fdim, facet_indices)
    
    if len(dofs) == 0:
        if verbose and rank == 0:
            print("[VortexLens] Warning: No DOFs found on specified facets")
        return p_vortex
    
    # Get coordinates of these DOFs
    coords = function_space.tabulate_dof_coordinates()
    dof_coords = coords[dofs]
    
    # Compute center if not provided
    if center is None:
        if config.center is not None:
            center = np.array(config.center)
        else:
            # Use centroid of boundary DOFs
            center = np.mean(dof_coords, axis=0)
            # Broadcast to all ranks
            center = comm.bcast(center, root=0)
    
    # Compute phase and amplitude at DOF locations
    phase = compute_azimuthal_phase(dof_coords, config, center)
    amplitude = compute_amplitude_profile(dof_coords, config, center)
    
    # Complex boundary field: A(x) exp(iφ(x))
    p_boundary = amplitude * np.exp(1j * phase)
    
    # Set DOF values
    p_vortex.x.array[dofs] = p_boundary
    
    if verbose and rank == 0:
        print(f"[VortexLens] Created vortex boundary field:")
        print(f"  Topological charge: ℓ = {config.topological_charge}")
        print(f"  Vortex center: ({center[0]*1e3:.3f}, {center[1]*1e3:.3f}, {center[2]*1e3:.3f}) mm")
        print(f"  Axis: {config.axis}")
        print(f"  Apodization: {config.apodization}")
        print(f"  Amplitude: {config.amplitude:.3e} Pa")
        print(f"  DOFs affected: {len(dofs)}")
        print(f"  Phase range: [{np.min(phase):.2f}, {np.max(phase):.2f}] rad")
    
    return p_vortex


def apply_vortex_neumann_bc(
    function_space: fem.FunctionSpace,
    facet_tags,
    boundary_tag: int,
    config: VortexLensConfig,
    omega: float,
    rho: float,
    center: Optional[np.ndarray] = None,
    velocity_amplitude: Optional[float] = None,
    verbose: bool = True
) -> ufl.Form:
    """
    Create Neumann BC term for vortex boundary actuation.
    
    Boundary condition:
        ∂p/∂n = -iωρ v₀ A(x) exp(iφ(x))
        
    where v₀ is the velocity amplitude and A(x)exp(iφ(x)) is the
    vortex spatial pattern.
    
    Parameters
    ----------
    function_space : dolfinx.fem.FunctionSpace
        Function space for pressure
    facet_tags : dolfinx.mesh.MeshTags
        Facet markers
    boundary_tag : int
        Tag identifying vortex actuation boundary
    config : VortexLensConfig
        Vortex configuration
    omega : float
        Angular frequency (rad/s)
    rho : float
        Fluid density at boundary (kg/m³)
    center : np.ndarray, optional
        Vortex center
    velocity_amplitude : float, optional
        Normal velocity amplitude v₀ (m/s)
        If None, uses config.amplitude / (omega * rho)
    verbose : bool
        
    Returns
    -------
    ufl.Form
        Neumann BC contribution to weak form RHS
        
    Notes
    -----
    This returns the boundary integral term that goes into the RHS
    of the weak form. Use it as:
    
        L_vortex = apply_vortex_neumann_bc(...)
        L_total = L_other_terms + L_vortex
    """
    from mpi4py import MPI
    
    mesh = function_space.mesh
    comm = mesh.comm
    rank = comm.rank
    
    # Get facets with the specified tag
    tdim = mesh.topology.dim
    fdim = tdim - 1
    facet_indices = facet_tags.find(boundary_tag)
    
    if len(facet_indices) == 0:
        if verbose and rank == 0:
            print(f"[VortexLens] Warning: No facets found with tag {boundary_tag}")
        # Return zero form
        v = ufl.TestFunction(function_space)
        ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)
        return Constant(mesh, 0.0) * v * ds(boundary_tag)
    
    # Compute velocity amplitude
    if velocity_amplitude is None:
        # Derive from pressure amplitude: p = ρ c v for plane wave
        # Or use impedance relationship: v₀ = A / (ω ρ)
        velocity_amplitude = config.amplitude / (omega * rho)
    
    # Create vortex boundary function
    p_vortex_pattern = create_vortex_boundary_function(
        function_space, facet_indices, config, center, verbose=False
    )
    
    # Normalize to unit amplitude, then scale by velocity
    # The pattern already has config.amplitude baked in, so extract phase only
    coords = function_space.tabulate_dof_coordinates()
    phase_pattern = compute_azimuthal_phase(coords, config, center)
    amplitude_pattern = compute_amplitude_profile(coords, config, center)
    
    # Create a function with just the spatial pattern (normalized)
    pattern_func = Function(function_space)
    # Reconstruct with unit amplitude
    max_amp = np.max(amplitude_pattern) if np.max(amplitude_pattern) > 0 else 1.0
    normalized_pattern = (amplitude_pattern / max_amp) * np.exp(1j * phase_pattern)
    pattern_func.x.array[:] = normalized_pattern
    
    # Neumann BC term: ∫_Γ φ * (-iωρ v₀ pattern) dS
    v = ufl.TestFunction(function_space)
    ds_vortex = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)
    
    # Actuation term
    g_vortex = -1j * omega * rho * velocity_amplitude * pattern_func
    
    L_vortex = v * g_vortex * ds_vortex(boundary_tag)
    
    if verbose and rank == 0:
        print(f"[VortexLens] Applied Neumann BC on boundary tag {boundary_tag}")
        print(f"  Velocity amplitude: v₀ = {velocity_amplitude*1e3:.3f} mm/s")
        print(f"  Impedance: ωρv₀ = {omega*rho*velocity_amplitude:.3e} Pa")
    
    return L_vortex


# Convenience function for typical use case
def create_vortex_lens(
    ell: int,
    amplitude: float = 1e6,
    center: Optional[Tuple[float, float, float]] = None,
    axis: str = 'z',
    apodization: str = 'uniform'
) -> VortexLensConfig:
    """
    Convenience constructor for vortex lens configuration.
    
    Parameters
    ----------
    ell : int
        Topological charge
    amplitude : float
        Pressure amplitude (Pa)
    center : tuple, optional
        (x, y, z) center coordinates (m)
    axis : str
        Vortex axis: 'x', 'y', or 'z'
    apodization : str
        Amplitude profile: 'uniform', 'gaussian', 'bessel'
        
    Returns
    -------
    VortexLensConfig
    """
    return VortexLensConfig(
        topological_charge=ell,
        center=center,
        amplitude=amplitude,
        axis=axis,
        apodization=apodization
    )
