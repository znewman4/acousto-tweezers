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

Plastic Lens Extension (Day 2)
-------------------------------
A fabricable plastic lens that encodes both vortex AND focusing phases
via thickness variation:

    φ_target(x,y) = ℓ·θ + k·(√((x-xf)² + (y-yf)² + f²) - f)

    t(x,y) = t₀ + mod(φ_target, 2π) / (k_lens - k_water)

    φ_plastic = (k_lens - k_water) · (t(x,y) - t₀)  = mod(φ_target, 2π)

The velocity drive on the bottom disk boundary is:

    v_n(x,y) = V₀ · A(r) · exp(i · φ_plastic(x,y))

Physics Notes
-------------
- Topological charge ℓ determines phase winding: 2π per rotation
- Amplitude can be uniform or radially apodized
- Superimposes linearly with other boundary fields
- No volumetric forcing - actuation is boundary-only
- Plastic lens wraps phase to [0, 2π] ⇒ fabricable stepped structure
- Off-axis focus (xf, yf) ≠ (0, 0) biases particle translation

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


# =====================================================================
# PLASTIC LENS MODEL (Day 2)
# =====================================================================

@dataclass
class PlasticLensConfig:
    """
    Configuration for a fabricable plastic lens boundary drive.

    The lens encodes a combined vortex + focusing phase via thickness:

        φ_target(x,y) = ℓ·θ + k_water · (√((x-xf)² + (y-yf)² + f²) - f)

    where (xf, yf) is the focus offset from disk center
    and f is the focal length.

    The phase is wrapped to [0, 2π) and used directly:
        v_n(x,y) = V₀ · A(r) · exp(i · mod(φ_target, 2π))

    Parameters
    ----------
    topological_charge : int
        Vortex topological charge ℓ (typically 1).
    focal_length : float
        Focal length of the lens [m].  Infinity → no focusing term.
    focus_offset_x : float
        x-offset of focus from disk center [m].  Nonzero biases translation.
    focus_offset_y : float
        y-offset of focus from disk center [m].
    c_lens : float
        Speed of sound in lens plastic [m/s].  Used for k_lens = ω/c_lens.
        Only affects the physical thickness t(x,y); the boundary phase
        φ_plastic = mod(φ_target, 2π) is independent of c_lens.
    c_water : float
        Speed of sound in water [m/s].  k_water = ω/c_water.
    frequency_hz : float
        Operating frequency [Hz].
    aperture_radius : float
        Lens disk aperture radius [m].
    center : tuple or None
        (x_c, y_c) center of the lens disk on the bottom face.
    apodization : str
        Amplitude taper: 'cosine_taper', 'uniform', 'tukey'.
    apodization_strength : float
        Controls taper roll-off width as fraction of R.
        For cosine_taper: taper = 0.5*(1 + cos(π·r/(R·strength))) for r < R·strength.
        Default 1.0 means full cosine taper over entire radius.
    """
    topological_charge: int = 1
    focal_length: float = 10.0e-3     # 10 mm
    focus_offset_x: float = 0.2e-3    # slight off-axis
    focus_offset_y: float = 0.0
    c_lens: float = 2700.0            # polycarbonate / acrylic
    c_water: float = 1484.0
    frequency_hz: float = 2.0e6
    aperture_radius: float = 1.0e-3   # matches FarFieldConfig.disk_radius
    center: Optional[Tuple[float, float]] = None
    apodization: str = "cosine_taper"
    apodization_strength: float = 1.0

    @property
    def omega(self) -> float:
        return 2 * np.pi * self.frequency_hz

    @property
    def k_water(self) -> float:
        return self.omega / self.c_water

    @property
    def k_lens(self) -> float:
        return self.omega / self.c_lens

    @property
    def wavelength_water(self) -> float:
        return self.c_water / self.frequency_hz


def compute_plastic_lens_phase(
    x: np.ndarray,
    y: np.ndarray,
    lens_cfg: PlasticLensConfig,
    center_x: float = 0.0,
    center_y: float = 0.0,
) -> np.ndarray:
    """
    Compute the plastic lens phase φ_plastic at given (x, y) coordinates.

    φ_target = ℓ·θ + k_water · (√((x-xf)² + (y-yf)² + f²) - f)
    φ_plastic = mod(φ_target, 2π)
    
    Parameters
    ----------
    x, y : np.ndarray
        Coordinates of evaluation points.
    lens_cfg : PlasticLensConfig
        Lens configuration.
    center_x, center_y : float
        Override center position (disk center on bottom face).
        
    Returns
    -------
    phi_target : np.ndarray
        Unwrapped target phase (for diagnostics).
    phi_plastic : np.ndarray
        Wrapped phase mod(φ_target, 2π) — what the lens actually imprints.
    """
    if lens_cfg.center is not None:
        cx, cy = lens_cfg.center
    else:
        cx, cy = center_x, center_y

    dx = x - cx
    dy = y - cy
    r = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)

    ell = lens_cfg.topological_charge
    k_w = lens_cfg.k_water
    f = lens_cfg.focal_length

    # Vortex phase
    phi_vortex = ell * theta

    # Focusing phase (converging spherical wavefront)
    # Focus point is offset from center
    xf = lens_cfg.focus_offset_x
    yf = lens_cfg.focus_offset_y
    dx_f = x - (cx + xf)
    dy_f = y - (cy + yf)
    rho_f = np.sqrt(dx_f**2 + dy_f**2 + f**2)
    phi_focus = k_w * (rho_f - f)

    phi_target = phi_vortex + phi_focus

    # Wrap to [0, 2π)  —  this is what the physical lens can fabricate
    phi_plastic = np.mod(phi_target, 2 * np.pi)

    return phi_target, phi_plastic


def compute_plastic_lens_amplitude(
    x: np.ndarray,
    y: np.ndarray,
    lens_cfg: PlasticLensConfig,
    center_x: float = 0.0,
    center_y: float = 0.0,
) -> np.ndarray:
    """
    Compute amplitude apodization A(r) for the plastic lens.
    
    Returns
    -------
    amplitude : np.ndarray
        Values in [0, 1] — caller scales by V₀.
    """
    if lens_cfg.center is not None:
        cx, cy = lens_cfg.center
    else:
        cx, cy = center_x, center_y

    dx = x - cx
    dy = y - cy
    r = np.sqrt(dx**2 + dy**2)
    R = lens_cfg.aperture_radius
    s = lens_cfg.apodization_strength

    amp = np.zeros_like(r)

    if lens_cfg.apodization == "cosine_taper":
        # Full cosine taper: 0.5*(1 + cos(π r / R)) for r < R
        inside = r <= R
        amp[inside] = 0.5 * (1 + np.cos(np.pi * r[inside] / R))

    elif lens_cfg.apodization == "tukey":
        # Flat interior + cosine roll-off in outer fraction s of R
        inside = r <= R
        r_taper_start = R * (1 - s)
        flat = inside & (r <= r_taper_start)
        roll = inside & (r > r_taper_start)
        amp[flat] = 1.0
        if np.any(roll):
            xi = (r[roll] - r_taper_start) / (R - r_taper_start)
            amp[roll] = 0.5 * (1 + np.cos(np.pi * xi))

    elif lens_cfg.apodization == "uniform":
        amp[r <= R] = 1.0

    else:
        # Default to cosine taper
        inside = r <= R
        amp[inside] = 0.5 * (1 + np.cos(np.pi * r[inside] / R))

    return amp


def compute_plastic_lens_thickness(
    x: np.ndarray,
    y: np.ndarray,
    lens_cfg: PlasticLensConfig,
    t0: float = None,
    center_x: float = 0.0,
    center_y: float = 0.0,
    safety_margin: float = 0.2e-3,
) -> np.ndarray:
    """
    Compute physical lens thickness t(x,y) for fabrication.

    t(x,y) = t₀ + mod(φ_target, 2π) / (k_lens - k_water)

    Sign convention
    ---------------
    dk = k_lens - k_water.  For typical plastics (c_lens > c_water),
    dk < 0, so increasing phase ⇒ thinner lens.  To ensure t > 0
    everywhere, t₀ must satisfy  t₀ ≥ 2π/|dk| + safety_margin.

    When t0=None (default), the safe base thickness is computed
    automatically so that all thickness values are positive.

    Parameters
    ----------
    t0 : float or None
        Minimum (base) thickness [m].  None = auto-compute safe value.
    safety_margin : float
        Extra margin above the minimum required base [m].

    Returns
    -------
    thickness : np.ndarray
        Physical thickness at each point [m].
    """
    _, phi_plastic = compute_plastic_lens_phase(
        x, y, lens_cfg, center_x, center_y)
    dk = lens_cfg.k_lens - lens_cfg.k_water
    if abs(dk) < 1e-10:
        raise ValueError("k_lens ≈ k_water → infinite thickness; check c_lens")

    # Auto-compute safe base thickness
    min_required = 2 * np.pi / abs(dk) + safety_margin
    if t0 is None:
        t0 = min_required
    elif t0 < min_required:
        import warnings
        warnings.warn(
            f"t0={t0*1e3:.2f} mm < min required {min_required*1e3:.2f} mm; "
            f"some thicknesses will be negative. Using safe t0 instead.",
            stacklevel=2,
        )
        t0 = min_required

    thickness = t0 + phi_plastic / dk
    return thickness


def create_plastic_lens_drive(
    coords_x: np.ndarray,
    coords_y: np.ndarray,
    lens_cfg: PlasticLensConfig,
    center_x: float = 0.0,
    center_y: float = 0.0,
    verbose: bool = True,
) -> np.ndarray:
    """
    Build complex boundary field for plastic lens drive.

    Returns v_pattern = A(r) · exp(i · φ_plastic(x,y))
    with unit peak amplitude (caller scales by V₀ × (−iωρ)).
    
    Parameters
    ----------
    coords_x, coords_y : np.ndarray
        (x, y) coordinates of DOFs on the disk boundary.
    lens_cfg : PlasticLensConfig
    center_x, center_y : float
        Disk center position.
    verbose : bool
        
    Returns
    -------
    pattern : np.ndarray (complex128)
        Complex drive pattern A(r) exp(i φ_plastic).
    """
    phi_target, phi_plastic = compute_plastic_lens_phase(
        coords_x, coords_y, lens_cfg, center_x, center_y)
    amplitude = compute_plastic_lens_amplitude(
        coords_x, coords_y, lens_cfg, center_x, center_y)

    pattern = amplitude * np.exp(1j * phi_plastic)

    if verbose:
        n_active = int(np.sum(amplitude > 1e-10))
        print(f"  [PlasticLens] l={lens_cfg.topological_charge}  "
              f"f={lens_cfg.focal_length*1e3:.1f} mm  "
              f"offset=({lens_cfg.focus_offset_x*1e3:.2f}, "
              f"{lens_cfg.focus_offset_y*1e3:.2f}) mm")
        print(f"    k_water={lens_cfg.k_water:.1f}  k_lens={lens_cfg.k_lens:.1f}  "
              f"dk={lens_cfg.k_lens - lens_cfg.k_water:.1f} rad/m")
        print(f"    phi_target range: [{phi_target.min():.2f}, {phi_target.max():.2f}] rad")
        print(f"    phi_plastic range: [{phi_plastic.min():.2f}, {phi_plastic.max():.2f}] rad")
        print(f"    amplitude: min={amplitude.min():.4f}  max={amplitude.max():.4f}")
        print(f"    active DOFs: {n_active}/{len(coords_x)}")

    return pattern


# =====================================================================
# Lens Presets
# =====================================================================

def lens_preset_A(aperture_radius: float = 1.0e-3, **kw) -> PlasticLensConfig:
    """Preset A: pure vortex, weak focus (large f = 50 mm)."""
    defaults = dict(
        topological_charge=1, focal_length=50e-3,
        focus_offset_x=0.0, focus_offset_y=0.0,
        c_lens=2700.0, c_water=1484.0, frequency_hz=2e6,
        aperture_radius=aperture_radius, apodization="cosine_taper",
    )
    defaults.update(kw)
    return PlasticLensConfig(**defaults)


def lens_preset_B(aperture_radius: float = 1.0e-3, **kw) -> PlasticLensConfig:
    """Preset B: focused vortex (moderate f = 10 mm)."""
    defaults = dict(
        topological_charge=1, focal_length=10e-3,
        focus_offset_x=0.0, focus_offset_y=0.0,
        c_lens=2700.0, c_water=1484.0, frequency_hz=2e6,
        aperture_radius=aperture_radius, apodization="cosine_taper",
    )
    defaults.update(kw)
    return PlasticLensConfig(**defaults)


def lens_preset_C(aperture_radius: float = 1.0e-3, **kw) -> PlasticLensConfig:
    """Preset C: off-axis focused vortex (f = 10 mm, xf = 0.2 mm)."""
    defaults = dict(
        topological_charge=1, focal_length=10e-3,
        focus_offset_x=0.2e-3, focus_offset_y=0.0,
        c_lens=2700.0, c_water=1484.0, frequency_hz=2e6,
        aperture_radius=aperture_radius, apodization="cosine_taper",
    )
    defaults.update(kw)
    return PlasticLensConfig(**defaults)


LENS_PRESETS = {"A": lens_preset_A, "B": lens_preset_B, "C": lens_preset_C}


# =====================================================================
# AXICON LENS MODEL — Bessel-like vortex via conical wavefront
# =====================================================================

@dataclass
class AxiconLensConfig:
    """
    Configuration for an axicon (conical) vortex lens.

    The axicon generates a Bessel-like beam with a non-diffracting
    central core by impressing a conical phase:

        phi(r, theta) = ell * theta + k_r * r

    where:
        k_r = k0 * sin(alpha)
        alpha = axicon half-angle
        r = radial distance from optical axis

    Unlike a converging (spherical) lens, the axicon produces an
    extended focal region along the beam axis rather than a single
    focal point.  The resulting field approximates J_ell(k_r r) —
    a Bessel vortex beam.

    Parameters
    ----------
    topological_charge : int
        Vortex topological charge ell.
    axicon_angle_deg : float
        Axicon half-angle alpha [degrees].  Typical range 5–30 deg.
        Larger angle → tighter core, shorter depth of field.
    c_water : float
        Speed of sound in water [m/s].
    frequency_hz : float
        Operating frequency [Hz].
    aperture_radius : float
        Lens disk radius [m].
    center : tuple or None
        (x_c, y_c) center of the disk.
    apodization : str
        Amplitude taper: 'cosine_taper', 'uniform', 'gaussian'.
    apodization_strength : float
        Controls taper width parameter.
    """
    topological_charge: int = 1
    axicon_angle_deg: float = 15.0
    c_water: float = 1484.0
    frequency_hz: float = 2.0e6
    aperture_radius: float = 1.0e-3
    center: Optional[Tuple[float, float]] = None
    apodization: str = "cosine_taper"
    apodization_strength: float = 1.0

    @property
    def omega(self) -> float:
        return 2 * np.pi * self.frequency_hz

    @property
    def k_water(self) -> float:
        return self.omega / self.c_water

    @property
    def axicon_angle_rad(self) -> float:
        return np.deg2rad(self.axicon_angle_deg)

    @property
    def k_r(self) -> float:
        """Transverse wavenumber k_r = k0 sin(alpha)."""
        return self.k_water * np.sin(self.axicon_angle_rad)

    @property
    def wavelength_water(self) -> float:
        return self.c_water / self.frequency_hz


def compute_axicon_phase(
    x: np.ndarray,
    y: np.ndarray,
    axicon_cfg: AxiconLensConfig,
    center_x: float = 0.0,
    center_y: float = 0.0,
) -> np.ndarray:
    """
    Compute axicon vortex phase: phi = ell * theta + k_r * r.

    Parameters
    ----------
    x, y : np.ndarray
        Coordinates of evaluation points.
    axicon_cfg : AxiconLensConfig
    center_x, center_y : float
        Override center position.

    Returns
    -------
    phi : np.ndarray
        Phase values [rad].
    """
    if axicon_cfg.center is not None:
        cx, cy = axicon_cfg.center
    else:
        cx, cy = center_x, center_y

    dx = x - cx
    dy = y - cy
    r = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)

    phi = axicon_cfg.topological_charge * theta + axicon_cfg.k_r * r
    return phi


def compute_axicon_amplitude(
    x: np.ndarray,
    y: np.ndarray,
    axicon_cfg: AxiconLensConfig,
    center_x: float = 0.0,
    center_y: float = 0.0,
) -> np.ndarray:
    """
    Compute amplitude apodization for the axicon lens.

    Returns values in [0, 1].
    """
    if axicon_cfg.center is not None:
        cx, cy = axicon_cfg.center
    else:
        cx, cy = center_x, center_y

    dx = x - cx
    dy = y - cy
    r = np.sqrt(dx**2 + dy**2)
    R = axicon_cfg.aperture_radius

    amp = np.zeros_like(r)

    if axicon_cfg.apodization == "cosine_taper":
        inside = r <= R
        amp[inside] = 0.5 * (1 + np.cos(np.pi * r[inside] / R))
    elif axicon_cfg.apodization == "uniform":
        amp[r <= R] = 1.0
    elif axicon_cfg.apodization == "gaussian":
        sigma = R * axicon_cfg.apodization_strength * 0.5
        amp = np.exp(-r**2 / (2 * sigma**2))
        amp[r > R] = 0.0
    else:
        inside = r <= R
        amp[inside] = 0.5 * (1 + np.cos(np.pi * r[inside] / R))

    return amp


def create_axicon_lens_drive(
    coords_x: np.ndarray,
    coords_y: np.ndarray,
    axicon_cfg: AxiconLensConfig,
    center_x: float = 0.0,
    center_y: float = 0.0,
    verbose: bool = True,
) -> np.ndarray:
    """
    Build complex boundary field for axicon lens drive.

    Returns pattern = A(r) * exp(i * phi(r, theta))
    with unit peak amplitude (caller scales by V0 * (-i omega rho)).

    Parameters
    ----------
    coords_x, coords_y : np.ndarray
        (x, y) coordinates of DOFs on the disk boundary.
    axicon_cfg : AxiconLensConfig
    center_x, center_y : float
    verbose : bool

    Returns
    -------
    pattern : np.ndarray (complex128)
    """
    phi = compute_axicon_phase(coords_x, coords_y, axicon_cfg,
                               center_x, center_y)
    amp = compute_axicon_amplitude(coords_x, coords_y, axicon_cfg,
                                    center_x, center_y)
    pattern = amp * np.exp(1j * phi)

    if verbose:
        n_active = int(np.sum(amp > 1e-10))
        print(f"  [AxiconLens] l={axicon_cfg.topological_charge}  "
              f"alpha={axicon_cfg.axicon_angle_deg:.1f} deg  "
              f"k_r={axicon_cfg.k_r:.1f} rad/m")
        print(f"    phi range: [{phi.min():.2f}, {phi.max():.2f}] rad")
        print(f"    amplitude: min={amp.min():.4f}  max={amp.max():.4f}")
        print(f"    active DOFs: {n_active}/{len(coords_x)}")

    return pattern


# =====================================================================
# Export helpers
# =====================================================================

def export_lens_maps(
    lens_cfg: PlasticLensConfig,
    out_dir,
    N: int = 200,
    center_x: float = 0.0,
    center_y: float = 0.0,
):
    """
    Export phase, amplitude, thickness maps to CSV and NPY files.

    Writes to *out_dir*:
      - phase_map.npy, amplitude_map.npy, thickness_map.npy  (2-D arrays)
      - grid_x.npy, grid_y.npy  (1-D coordinate arrays)
      - lens_maps_summary.csv  (scalar summary)

    Returns dict with summary info.
    """
    from pathlib import Path
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    R = lens_cfg.aperture_radius
    xg = np.linspace(-1.2 * R + center_x, 1.2 * R + center_x, N)
    yg = np.linspace(-1.2 * R + center_y, 1.2 * R + center_y, N)
    XX, YY = np.meshgrid(xg, yg)
    xf, yf = XX.ravel(), YY.ravel()

    phi_target, phi_plastic = compute_plastic_lens_phase(
        xf, yf, lens_cfg, center_x, center_y)
    amp = compute_plastic_lens_amplitude(
        xf, yf, lens_cfg, center_x, center_y)
    thickness = compute_plastic_lens_thickness(
        xf, yf, lens_cfg, center_x=center_x, center_y=center_y)

    phi_2d = phi_plastic.reshape(N, N)
    amp_2d = amp.reshape(N, N)
    thick_2d = thickness.reshape(N, N)

    np.save(out_dir / "phase_map.npy", phi_2d)
    np.save(out_dir / "amplitude_map.npy", amp_2d)
    np.save(out_dir / "thickness_map.npy", thick_2d)
    np.save(out_dir / "grid_x.npy", xg)
    np.save(out_dir / "grid_y.npy", yg)

    summary = {
        "phi_plastic_min": float(phi_plastic.min()),
        "phi_plastic_max": float(phi_plastic.max()),
        "amplitude_min": float(amp.min()),
        "amplitude_max": float(amp.max()),
        "thickness_min_mm": float(thickness.min() * 1e3),
        "thickness_max_mm": float(thickness.max() * 1e3),
        "dk_rad_m": float(lens_cfg.k_lens - lens_cfg.k_water),
        "base_thickness_mm": float(thickness.min() * 1e3),
    }

    import csv
    with open(out_dir / "lens_maps_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in summary.items():
            w.writerow([k, f"{v:.6f}"])

    return summary
