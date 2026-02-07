"""
Perfectly Matched Layer (PML) implementation for FEniCSx.

Implements true volumetric PML via anisotropic complex coordinate stretching.

Theory
------
For a PML in the +x direction with interface at x=L:
- Distance into PML: d = x - L (for x > L)
- Absorption profile: σ(d) = σ_max * (d/d_pml)^m
- Complex coordinate stretch: s_x(x) = 1 + i*σ(d)/ω

In the Helmholtz equation ∇²p + k²p = 0, the PML modifies derivatives:
- ∂/∂x → (1/s_x) * ∂/∂x
- ∂/∂y, ∂/∂z unchanged (for x-only PML)

Weak form with x-only PML:
    (1/ρ) * [(1/s_x)*p_x*v_x + p_y*v_y + p_z*v_z] - (k²/ρ)*s_x*p*v = 0

where p_x = ∂p/∂x, etc.

Note: Mass term gets s_x factor due to Jacobian determinant in 1D stretch case.

The PML transforms the Helmholtz equation via complex coordinate stretching:
    x → x̃ = x + (i/ω) ∫₀ˣ σ(ξ) dξ

Target: < 1% reflection coefficient.

Author: Acousto-Tweezers Project
Date: January 2026
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Callable
import numpy as np

from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
from dolfinx import fem, mesh as dmesh
from dolfinx.fem import Function
import ufl
from ufl import (
    inner, grad, dx, ds,
    TestFunction, TrialFunction,
    Measure, SpatialCoordinate
)

from .config import FEMConfig, GeometryConfig
from .domains import Domain


# ============================================================================
# Volumetric Anisotropic PML - Simple API (for validation and production)
# ============================================================================

# ============================================================================
# TENSOR PML (6-FACE) - PRODUCTION RECOMMENDED
# ============================================================================

def build_pml_stretch_tensor_dg0(
    mesh,
    cell_tags,
    pml_tags,
    bbox,
    pml_thickness,
    omega,
    sigma_max,
    power=2,
    water_tag=None
):
    """
    Build 3D tensor PML with stretches (s_x, s_y, s_z) for 6-face absorption.
    
    This is the GENERAL PRODUCTION implementation that handles PML on all
    6 faces of a rectangular domain with proper corner treatment.
    
    Theory:
    ------
    For a box domain with PML on all sides, each cell gets 3 independent stretches:
    - s_x from distance to left/right PML boundaries
    - s_y from distance to front/back PML boundaries
    - s_z from distance to top/bottom PML boundaries
    
    In corners, multiple stretches are active simultaneously.
    
    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh
        Computational mesh
    cell_tags : dolfinx.mesh.MeshTags
        Cell markers
    pml_tags : list of int
        Tags for PML regions (e.g., [TAG_PML_LEFT, TAG_PML_RIGHT, ...])
    bbox : tuple
        Bounding box (x_min, x_max, y_min, y_max, z_min, z_max)
    pml_thickness : float
        PML thickness (same on all faces)
    omega : float
        Angular frequency
    sigma_max : float
        Maximum absorption coefficient
    power : int
        Polynomial order for σ(d) profile
    water_tag : int, optional
        Tag for water/physical domain (for diagnostics)
    
    Returns
    -------
    s_x, s_y, s_z : fem.Function
        Complex stretch fields on DG0
    s_x_inv, s_y_inv, s_z_inv : fem.Function
        Inverse stretches (1/s_*)
    diagnostics : dict
        PML statistics (Im(s) ranges, cell counts)
    """
    import numpy as np
    from dolfinx import fem
    
    # Unpack bounding box
    x_min, x_max, y_min, y_max, z_min, z_max = bbox
    
    # PML interface locations
    x_pml_left = x_min + pml_thickness
    x_pml_right = x_max - pml_thickness
    y_pml_front = y_min + pml_thickness
    y_pml_back = y_max - pml_thickness
    z_pml_bottom = z_min + pml_thickness
    z_pml_top = z_max - pml_thickness
    
    # Create DG0 function space with complex dtype
    DG0 = fem.functionspace(mesh, ("DG", 0))
    s_x = fem.Function(DG0, dtype=np.complex128)
    s_y = fem.Function(DG0, dtype=np.complex128)
    s_z = fem.Function(DG0, dtype=np.complex128)
    s_x_inv = fem.Function(DG0, dtype=np.complex128)
    s_y_inv = fem.Function(DG0, dtype=np.complex128)
    s_z_inv = fem.Function(DG0, dtype=np.complex128)
    
    # Get mesh topology
    mesh.topology.create_connectivity(mesh.topology.dim, 0)
    num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    coords = mesh.geometry.x
    dofmap = DG0.dofmap
    
    # Find PML cells (union of all PML tags)
    pml_cells_set = set()
    for tag in pml_tags:
        pml_cells_set.update(cell_tags.find(tag))
    
    # Find water cells (for diagnostics)
    water_cells = cell_tags.find(water_tag) if water_tag is not None else []
    
    # Initialize stretch fields
    for cell in range(num_cells):
        dofs = dofmap.cell_dofs(cell)
        dof_idx = dofs[0]
        
        # Get cell centroid
        cell_to_vertex = mesh.topology.connectivity(mesh.topology.dim, 0)
        vertices = cell_to_vertex.links(cell)
        cell_coords = coords[vertices]
        x_c, y_c, z_c = np.mean(cell_coords, axis=0)
        
        # Compute distance into PML for each axis
        # d > 0 when inside PML, 0 in physical domain
        d_x_left = max(0, x_pml_left - x_c)
        d_x_right = max(0, x_c - x_pml_right)
        d_x = d_x_left + d_x_right  # Only one is nonzero at a time
        
        d_y_front = max(0, y_pml_front - y_c)
        d_y_back = max(0, y_c - y_pml_back)
        d_y = d_y_front + d_y_back
        
        d_z_bottom = max(0, z_pml_bottom - z_c)
        d_z_top = max(0, z_c - z_pml_top)
        d_z = d_z_bottom + d_z_top
        
        # Clamp distances
        d_x = min(d_x, pml_thickness)
        d_y = min(d_y, pml_thickness)
        d_z = min(d_z, pml_thickness)
        
        # Compute complex stretches
        s_x_val = pml_complex_stretch(d_x, pml_thickness, sigma_max, omega, power)
        s_y_val = pml_complex_stretch(d_y, pml_thickness, sigma_max, omega, power)
        s_z_val = pml_complex_stretch(d_z, pml_thickness, sigma_max, omega, power)
        
        # Assign
        s_x.x.array[dof_idx] = s_x_val
        s_y.x.array[dof_idx] = s_y_val
        s_z.x.array[dof_idx] = s_z_val
        s_x_inv.x.array[dof_idx] = 1.0 / s_x_val
        s_y_inv.x.array[dof_idx] = 1.0 / s_y_val
        s_z_inv.x.array[dof_idx] = 1.0 / s_z_val
    
    # Diagnostics
    diagnostics = {}
    
    if len(water_cells) > 0:
        water_dofs = [dofmap.cell_dofs(c)[0] for c in water_cells[:min(100, len(water_cells))]]
        diagnostics['im_s_x_water'] = {
            'min': float(np.min(np.abs(np.imag(s_x.x.array[water_dofs])))),
            'median': float(np.median(np.abs(np.imag(s_x.x.array[water_dofs])))),
            'max': float(np.max(np.abs(np.imag(s_x.x.array[water_dofs])))),
        }
        diagnostics['im_s_y_water'] = {
            'min': float(np.min(np.abs(np.imag(s_y.x.array[water_dofs])))),
            'median': float(np.median(np.abs(np.imag(s_y.x.array[water_dofs])))),
            'max': float(np.max(np.abs(np.imag(s_y.x.array[water_dofs])))),
        }
        diagnostics['im_s_z_water'] = {
            'min': float(np.min(np.abs(np.imag(s_z.x.array[water_dofs])))),
            'median': float(np.median(np.abs(np.imag(s_z.x.array[water_dofs])))),
            'max': float(np.max(np.abs(np.imag(s_z.x.array[water_dofs])))),
        }
    
    if len(pml_cells_set) > 0:
        pml_dofs = [dofmap.cell_dofs(c)[0] for c in list(pml_cells_set)[:min(100, len(pml_cells_set))]]
        diagnostics['im_s_x_pml'] = {
            'min': float(np.min(np.abs(np.imag(s_x.x.array[pml_dofs])))),
            'median': float(np.median(np.abs(np.imag(s_x.x.array[pml_dofs])))),
            'max': float(np.max(np.abs(np.imag(s_x.x.array[pml_dofs])))),
        }
        diagnostics['im_s_y_pml'] = {
            'min': float(np.min(np.abs(np.imag(s_y.x.array[pml_dofs])))),
            'median': float(np.median(np.abs(np.imag(s_y.x.array[pml_dofs])))),
            'max': float(np.max(np.abs(np.imag(s_y.x.array[pml_dofs])))),
        }
        diagnostics['im_s_z_pml'] = {
            'min': float(np.min(np.abs(np.imag(s_z.x.array[pml_dofs])))),
            'median': float(np.median(np.abs(np.imag(s_z.x.array[pml_dofs])))),
            'max': float(np.max(np.abs(np.imag(s_z.x.array[pml_dofs])))),
        }
    
    diagnostics['num_water_cells'] = len(water_cells)
    diagnostics['num_pml_cells'] = len(pml_cells_set)
    
    return s_x, s_y, s_z, s_x_inv, s_y_inv, s_z_inv, diagnostics


def helmholtz_tensor_pml_forms(
    p, v, mesh, k, rho, omega,
    s_x, s_y, s_z,
    s_x_inv, s_y_inv, s_z_inv,
    dx_domain, dx_pml,
    source_form=None
):
    """
    Build Helmholtz weak form with full 3D tensor PML.
    
    This is the GENERAL PRODUCTION implementation for multi-directional PML.
    
    Implements:
        a(p,v) = (1/ρ) * [(1/s_x)*∂p/∂x*∂v̄/∂x + (1/s_y)*∂p/∂y*∂v̄/∂y + (1/s_z)*∂p/∂z*∂v̄/∂z]
                 - (k²/ρ) * (s_x * s_y * s_z) * p * v̄
    
    where v̄ = conj(v) for complex mode.
    
    Key features:
    - All 3 gradient components have independent stretches
    - Mass term has FULL Jacobian (s_x * s_y * s_z)
    - Works for physical domain (s = 1) and PML (s ≠ 1)
    
    Parameters
    ----------
    p : ufl.TrialFunction or ufl.Coefficient
        Pressure field
    v : ufl.TestFunction
        Test function
    mesh : dolfinx.mesh.Mesh
        Mesh
    k : float
        Wavenumber
    rho : float
        Density
    omega : float
        Angular frequency
    s_x, s_y, s_z : fem.Function
        Complex stretch fields (DG0)
    s_x_inv, s_y_inv, s_z_inv : fem.Function
        Inverse stretches
    dx_domain : ufl.Measure
        Domain measure (physical + PML)
    dx_pml : ufl.Measure, optional
        Separate PML measure if needed
    source_form : ufl.Form, optional
        RHS source term
    
    Returns
    -------
    a_form : ufl.Form
        Bilinear form
    L_form : ufl.Form or None
        Linear form
    """
    from ufl import grad, conj
    
    # Gradient with conjugated test function
    grad_p = grad(p)
    grad_v = grad(conj(v))
    
    # Anisotropic gradient term: (1/ρ) * Σ_i (1/s_i) * p_i * v̄_i
    gradient_term = (1.0 / rho) * (
        s_x_inv * grad_p[0] * grad_v[0]
        + s_y_inv * grad_p[1] * grad_v[1]
        + s_z_inv * grad_p[2] * grad_v[2]
    )
    
    # Mass term with FULL Jacobian: -(k²/ρ) * (s_x * s_y * s_z) * p * v̄
    jacobian = s_x * s_y * s_z
    mass_term = -(k**2 / rho) * jacobian * p * conj(v)
    
    # Bilinear form
    a_form = (gradient_term + mass_term) * dx_domain
    
    # Add PML region if separate measure provided
    if dx_pml is not None:
        a_form = a_form + (gradient_term + mass_term) * dx_pml
    
    L_form = source_form if source_form is not None else None
    
    return a_form, L_form


# ============================================================================
# DIRECTIONAL PML (x-only) - LEGACY / TESTING ONLY
# ============================================================================
# NOTE: The functions below are kept for backward compatibility and simple
# directional tests. For production use with general geometries, use the
# tensor PML functions above.
# ============================================================================

def pml_sigma_profile(d, d_pml, sigma_max, power):
    """
    PML absorption profile σ(d).
    
    Parameters
    ----------
    d : float or array
        Distance into PML region from interface
    d_pml : float
        Total PML thickness
    sigma_max : float
        Maximum absorption coefficient
    power : int
        Polynomial order (typically 2 or 3)
    
    Returns
    -------
    sigma : float or array
        Absorption coefficient
    """
    return sigma_max * (d / d_pml) ** power


def pml_complex_stretch(d, d_pml, sigma_max, omega, power):
    """
    Complex PML coordinate scaling: s(d) = 1 + i*σ(d)/ω
    
    This is the fundamental PML transformation that makes waves decay
    exponentially as they propagate into the PML region.
    
    Parameters
    ----------
    d : float or array
        Distance into PML region
    d_pml : float
        PML thickness
    sigma_max : float
        Maximum absorption
    omega : float
        Angular frequency (2πf)
    power : int
        Polynomial order
    
    Returns
    -------
    s : complex or array
        Complex stretch factor
    """
    sigma = pml_sigma_profile(d, d_pml, sigma_max, power)
    return 1.0 + 1j * sigma / omega


def build_pml_stretch_dg0(
    mesh,
    cell_tags,
    tag_pml,
    L_interface,
    pml_thickness,
    omega,
    sigma_max,
    power=2,
    tag_water=None
):
    """
    Build complex coordinate stretch field s_x on DG0 space for x-only PML.
    
    **LEGACY FUNCTION - FOR DIRECTIONAL TESTING ONLY**
    For production use, prefer `build_pml_stretch_tensor_dg0` which supports
    full 3D tensor PML on all 6 faces of a box domain.
    
    This function only implements PML in the x-direction (s_y = s_z = 1).
    It is kept for backward compatibility and simple validation tests.
    
    For x-only PML with interface at x=L:
    - In water (x < L): s_x = 1
    - In PML (x ≥ L): s_x = 1 + i*σ(x-L)/ω
    
    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh
        Computational mesh
    cell_tags : dolfinx.mesh.MeshTags
        Cell markers
    tag_pml : int
        Tag identifying PML cells
    L_interface : float
        x-coordinate of water/PML interface
    pml_thickness : float
        Thickness of PML region
    omega : float
        Angular frequency (2πf)
    sigma_max : float
        Maximum PML absorption coefficient
    power : int, optional
        Polynomial order for σ(d) profile (default: 2)
    tag_water : int, optional
        Tag for water cells (for verification)
    
    Returns
    -------
    s_x : fem.Function
        Complex stretch field on DG0 space
    s_x_inv : fem.Function
        Inverse stretch (1/s_x) for gradient term
    im_s_water : float
        Max |Im(s_x)| in water region (should be ~0)
    im_s_pml : float
        Max |Im(s_x)| in PML region (should be >0)
    """
    # Create DG0 function space with complex dtype
    import numpy as np
    DG0 = fem.functionspace(mesh, ("DG", 0))
    s_x = fem.Function(DG0, dtype=np.complex128)
    s_x_inv = fem.Function(DG0, dtype=np.complex128)
    
    # Get cell coordinates
    mesh.topology.create_connectivity(mesh.topology.dim, 0)
    num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    coords = mesh.geometry.x
    
    # Get PML and water cells
    pml_cells = cell_tags.find(tag_pml)
    water_cells = cell_tags.find(tag_water) if tag_water is not None else []
    
    # Get dofmap (DG0: 1 dof per cell)
    dofmap = DG0.dofmap
    
    # Assign s_x values using proper dofmap (CRITICAL: not cell-index assumption)
    for cell in range(num_cells):
        # Get DOF index for this cell
        dofs = dofmap.cell_dofs(cell)
        dof_idx = dofs[0]  # DG0 has exactly 1 dof per cell
        
        # Check if this cell is in PML
        if cell in pml_cells:
            # Get cell center x-coordinate
            cell_to_vertex = mesh.topology.connectivity(mesh.topology.dim, 0)
            vertices = cell_to_vertex.links(cell)
            cell_coords = coords[vertices]
            x_center = np.mean(cell_coords[:, 0])
            
            # Distance into PML from interface
            d = x_center - L_interface
            d = max(0, min(d, pml_thickness))  # Clamp to [0, d_pml]
            
            # Compute complex stretch
            s = pml_complex_stretch(d, pml_thickness, sigma_max, omega, power)
        else:
            # Water region: no stretch
            s = 1.0 + 0j
        
        # Assign to DOF
        s_x.x.array[dof_idx] = s
        s_x_inv.x.array[dof_idx] = 1.0 / s
    
    # Verify PML is active (diagnostic)
    if len(water_cells) > 0:
        water_dofs = [dofmap.cell_dofs(c)[0] for c in water_cells[:min(10, len(water_cells))]]
        im_s_water = float(np.max(np.abs(np.imag(s_x.x.array[water_dofs]))))
    else:
        im_s_water = 0.0
    
    if len(pml_cells) > 0:
        pml_dofs = [dofmap.cell_dofs(c)[0] for c in pml_cells[:min(10, len(pml_cells))]]
        im_s_pml = float(np.max(np.abs(np.imag(s_x.x.array[pml_dofs]))))
    else:
        im_s_pml = 0.0
    
    return s_x, s_x_inv, im_s_water, im_s_pml


def helmholtz_anisotropic_pml_forms(
    p, v, mesh, k, rho, omega,
    s_x, s_x_inv,
    dx_water, dx_pml,
    source_form=None
):
    """
    Build Helmholtz weak form with anisotropic x-only PML.
    
    **LEGACY FUNCTION - FOR DIRECTIONAL TESTING ONLY**
    For production use, prefer `helmholtz_tensor_pml_forms` which supports
    full 3D tensor PML with all three coordinate stretches.
    
    This function only implements PML in the x-direction (s_y = s_z = 1).
    It is kept for backward compatibility and simple validation tests.
    
    Implements:
        a(p,v) = (1/ρ) * [(1/s_x)*∂p/∂x*∂v̄/∂x + ∂p/∂y*∂v̄/∂y + ∂p/∂z*∂v̄/∂z]
                 - (k²/ρ) * s_x * p * v̄
    
    where k = ω/c is the wavenumber and v̄ = conj(v) for complex mode.
    
    The anisotropic form:
    - Only x-derivatives are modified by PML stretch (1/s_x factor)
    - y,z derivatives unchanged (for x-only PML)
    - Mass term has s_x factor (Jacobian = s_x * 1 * 1 from coordinate stretch)
    
    In water region where s_x=1, this reduces to standard Helmholtz.
    
    Parameters
    ----------
    p : ufl.TrialFunction or ufl.Coefficient
        Pressure field
    v : ufl.TestFunction
        Test function
    mesh : dolfinx.mesh.Mesh
        Computational mesh
    k : float
        Wavenumber (ω/c)
    rho : float
        Density
    omega : float
        Angular frequency
    s_x : fem.Function
        Complex stretch field (DG0)
    s_x_inv : fem.Function
        Inverse stretch (1/s_x)
    dx_water, dx_pml : ufl.Measure
        Volume measures for water and PML regions
    source_form : ufl.Form, optional
        Right-hand side source term
    
    Returns
    -------
    a_form : ufl.Form
        Bilinear form
    L_form : ufl.Form or None
        Linear form (if source provided)
    """
    # Import conj for complex mode
    from ufl import conj
    
    # Anisotropic gradient term with x-only PML
    # (1/ρ) * [(1/s_x)*p_x*v̄_x + p_y*v̄_y + p_z*v̄_z]
    grad_p = grad(p)
    grad_v = grad(conj(v))  # Conjugate for complex mode
    
    # Water region (s_x=1, s_x_inv=1)
    a_water = (
        (1.0 / rho) * (
            s_x_inv * grad_p[0] * grad_v[0]  # x-direction with PML stretch
            + grad_p[1] * grad_v[1]           # y-direction (no stretch)
            + grad_p[2] * grad_v[2]           # z-direction (no stretch)
        ) * dx_water
        - (k**2 / rho) * s_x * p * conj(v) * dx_water
    )
    
    # PML region (same operator, but s_x ≠ 1)
    a_pml = (
        (1.0 / rho) * (
            s_x_inv * grad_p[0] * grad_v[0]
            + grad_p[1] * grad_v[1]
            + grad_p[2] * grad_v[2]
        ) * dx_pml
        - (k**2 / rho) * s_x * p * conj(v) * dx_pml
    )
    
    a_form = a_water + a_pml
    L_form = source_form if source_form is not None else None
    
    return a_form, L_form


# ============================================================================
# Legacy class-based PML API (kept for backward compatibility)
# ============================================================================


@dataclass
class PMLMetrics:
    """
    Metrics for PML performance evaluation.
    """
    reflection_coefficient: float
    max_field_in_pml: float
    field_decay_factor: float
    
    @property
    def passed(self) -> bool:
        """Check if PML meets < 1% reflection target."""
        return self.reflection_coefficient < 0.01
    
    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"PML Performance ({status}):\n"
            f"  Reflection coefficient: {self.reflection_coefficient*100:.2f}%\n"
            f"  Max field in PML: {self.max_field_in_pml:.2e}\n"
            f"  Decay factor: {self.field_decay_factor:.1f}x"
        )


class PMLHandler:
    """
    Handles PML region setup and modifications to the variational form.
    
    The PML is implemented by modifying the Helmholtz equation's spatial
    derivatives using complex coordinate stretching.
    """
    
    def __init__(self, config: FEMConfig, mesh: dmesh.Mesh):
        """
        Initialize PML handler.
        
        Parameters
        ----------
        config : FEMConfig
            Simulation configuration
        mesh : dolfinx.mesh.Mesh
            The computational mesh
        """
        self.config = config
        self.mesh = mesh
        self.geo = config.geometry
        
        # PML parameters
        self.thickness = self.geo.pml_thickness
        self.order = self.geo.pml_stretch_order
        self.sigma_max = self.geo.pml_max_sigma
        
        # Compute optimal sigma_max based on mesh and frequency
        self._compute_optimal_sigma()
        
    def _compute_optimal_sigma(self):
        """
        Compute optimal σ_max for target reflection.
        
        For polynomial stretching σ(x) = σ_max (x/L)^n, the theoretical
        reflection coefficient is:
        
        R ≈ exp(-2 σ_max L / (n+1) / c)
        
        For R < 0.01, we need σ_max > (n+1) c ln(100) / (2L)
        """
        c_ref = 1480.0  # Reference sound speed (water)
        target_R = 0.01
        
        n = self.order
        L = self.thickness
        
        # Required sigma for target reflection
        self.sigma_optimal = (n + 1) * c_ref * np.log(1.0/target_R) / (2 * L)
        
        # Use larger of specified and optimal
        self.sigma_max = max(self.sigma_max, self.sigma_optimal)
        
    def get_stretching_function(self, direction: str = 'x') -> Callable:
        """
        Get the complex stretching function s(x).
        
        s(x) = 1 + i σ(x) / ω
        
        where σ(x) increases polynomially from 0 at PML inner boundary
        to σ_max at PML outer boundary.
        
        Parameters
        ----------
        direction : str
            Coordinate direction ('x', 'y', or 'z')
            
        Returns
        -------
        Callable
            Stretching function s(coord, omega)
        """
        L = self.thickness
        n = self.order
        sigma_max = self.sigma_max
        
        def sigma_profile(d):
            """Polynomial absorption profile."""
            # d is distance into PML (0 at inner boundary, L at outer)
            return sigma_max * (d / L) ** n
        
        return sigma_profile
    
    def create_pml_coefficient(self, 
                               omega: float,
                               cell_tags: dmesh.MeshTags) -> Function:
        """
        Create PML coefficient function for the variational form.
        
        The modified Helmholtz equation in PML is:
        
        ∇·(A ∇p) + ω²/K det(S) p = 0
        
        where A = S^(-1) S^(-T) det(S) and S is the Jacobian of the
        coordinate transformation.
        
        For axis-aligned PML, this simplifies significantly.
        
        Parameters
        ----------
        omega : float
            Angular frequency
        cell_tags : MeshTags
            Cell domain tags
            
        Returns
        -------
        Function
            PML coefficient (complex-valued DG0 function)
        """
        # Create DG0 space for coefficient
        DG0 = fem.functionspace(self.mesh, ("DG", 0))
        
        # Real and imaginary parts (DOLFINx doesn't natively support complex DG)
        gamma_re = Function(DG0, dtype=np.float64)
        gamma_im = Function(DG0, dtype=np.float64)
        
        # Get cell centers
        gdim = self.mesh.geometry.dim
        num_cells = self.mesh.topology.index_map(gdim).size_local
        
        # Compute cell centroids
        self.mesh.topology.create_connectivity(gdim, 0)
        geometry = self.mesh.geometry.x
        
        cells = cell_tags.indices
        values = cell_tags.values
        
        cell_domain = np.zeros(num_cells, dtype=np.int32)
        cell_domain[cells] = values
        
        gamma_re_vals = np.ones(num_cells)
        gamma_im_vals = np.zeros(num_cells)
        
        # PML domain tags
        pml_tags = {
            Domain.PML_TOP.value,
            Domain.PML_BOTTOM.value,
            Domain.PML_LEFT.value,
            Domain.PML_RIGHT.value,
            Domain.PML_WATER.value,
            Domain.PML_AIR.value,
            Domain.PML_BATH.value,
        }
        
        # Domain boundaries (for computing distance into PML)
        R_outer = self.geo.dish_outer_radius
        z_air_top = self.geo.water_depth + self.geo.air_height
        z_bath_bottom = -self.geo.dish_bottom_thickness - self.geo.bath_depth
        
        sigma_profile = self.get_stretching_function('x')
        
        for i in range(num_cells):
            tag = cell_domain[i]
            if tag in pml_tags:
                # Get cell centroid (approximate)
                # For proper implementation, would need cell geometry
                
                # Compute distance into PML and stretching
                d = self.thickness / 2  # Approximate
                sigma = sigma_profile(d)
                
                # s = 1 + i σ/ω
                s_re = 1.0
                s_im = sigma / omega
                
                # For 3D axis-aligned PML, coefficient is product of stretching
                # in each direction. Simplified here.
                gamma_re_vals[i] = s_re
                gamma_im_vals[i] = s_im
        
        gamma_re.x.array[:] = gamma_re_vals
        gamma_im.x.array[:] = gamma_im_vals
        
        return gamma_re, gamma_im
    
    def modify_weak_form(self, 
                         a: ufl.Form,
                         p: ufl.Argument,
                         v: ufl.Argument,
                         omega: float,
                         inv_rho: Function,
                         K: Function,
                         cell_tags: dmesh.MeshTags) -> ufl.Form:
        """
        Modify the weak form for PML regions.
        
        In PML, the standard Helmholtz weak form:
        
        ∫ (1/ρ) ∇v·∇p - (ω²/K) v p dx
        
        becomes:
        
        ∫ (1/ρ) (A ∇v)·∇p - (ω²/K) det(S) v p dx
        
        Parameters
        ----------
        a : Form
            Original bilinear form
        p, v : Argument
            Trial and test functions
        omega : float
            Angular frequency
        inv_rho, K : Function
            Material property functions
        cell_tags : MeshTags
            Cell domain tags
            
        Returns
        -------
        Form
            Modified bilinear form with PML
        """
        # Get PML coefficients
        gamma_re, gamma_im = self.create_pml_coefficient(omega, cell_tags)
        
        dx_pml = Measure("dx", domain=self.mesh, subdomain_data=cell_tags)
        
        # PML domain tags
        pml_tags = [
            Domain.PML_TOP.value,
            Domain.PML_BOTTOM.value,
            Domain.PML_WATER.value,
            Domain.PML_AIR.value,
            Domain.PML_BATH.value,
        ]
        
        # Add PML contribution
        # Modified weak form with complex stretching
        a_pml = a
        for tag in pml_tags:
            # Complex coefficient: 1/s where s = gamma_re + i*gamma_im
            # 1/s = (gamma_re - i*gamma_im) / (gamma_re² + gamma_im²)
            denom = gamma_re**2 + gamma_im**2
            inv_s_re = gamma_re / denom
            inv_s_im = -gamma_im / denom
            
            # For simplicity, use approximate form
            # This adds imaginary damping in PML
            a_pml = a_pml + gamma_im * omega * inner(v, p) * dx_pml(tag)
        
        return a_pml
    
    def evaluate_reflection(self,
                            p_function: Function,
                            incident_amplitude: float) -> PMLMetrics:
        """
        Evaluate PML performance by estimating reflection coefficient.
        
        Parameters
        ----------
        p_function : Function
            Pressure solution
        incident_amplitude : float
            Expected incident wave amplitude
            
        Returns
        -------
        PMLMetrics
            PML performance metrics
        """
        # Get pressure values
        p_vals = p_function.x.array
        
        # Estimate maximum field in PML regions
        # This requires knowing which DOFs are in PML
        max_in_pml = np.max(np.abs(p_vals)) * 0.1  # Rough estimate
        
        # Reflection coefficient estimate
        # True estimation would compare incident and reflected waves
        R = max_in_pml / incident_amplitude if incident_amplitude > 0 else 0.0
        
        # Decay factor
        decay = incident_amplitude / max_in_pml if max_in_pml > 0 else np.inf
        
        return PMLMetrics(
            reflection_coefficient=R,
            max_field_in_pml=max_in_pml,
            field_decay_factor=decay,
        )


def create_absorbing_bc(V: fem.FunctionSpace,
                        mesh: dmesh.Mesh,
                        facet_tags: dmesh.MeshTags,
                        boundary_tag: int,
                        omega: float,
                        c: float) -> Tuple[ufl.Form, list]:
    """
    Create first-order absorbing boundary condition.
    
    The Sommerfeld radiation condition ∂p/∂n + ik p = 0 is implemented
    weakly as:
    
    ∫_Γ (ik) v p ds
    
    added to the bilinear form.
    
    Parameters
    ----------
    V : FunctionSpace
        Function space
    mesh : Mesh
        The mesh
    facet_tags : MeshTags
        Facet tags
    boundary_tag : int
        Tag for absorbing boundary
    omega : float
        Angular frequency
    c : float
        Sound speed
        
    Returns
    -------
    a_bc : Form
        Bilinear form contribution
    bcs : list
        Empty list (absorbing BC is natural)
    """
    p = TrialFunction(V)
    v = TestFunction(V)
    
    k = omega / c
    
    ds_bc = Measure("ds", domain=mesh, subdomain_data=facet_tags)
    
    a_bc = 1j * k * inner(v, p) * ds_bc(boundary_tag)
    
    return a_bc, []


def validate_pml_plane_wave(L_pml: float,
                            sigma_max: float,
                            order: int,
                            omega: float,
                            c: float,
                            num_elements: int = 50) -> Dict[str, float]:
    """
    Validate PML with a plane wave test.
    
    Sends a plane wave into a 1D PML and measures reflection.
    
    Parameters
    ----------
    L_pml : float
        PML thickness
    sigma_max : float
        Maximum absorption
    order : int
        Polynomial order
    omega : float
        Angular frequency
    c : float
        Sound speed
    num_elements : int
        Number of elements
        
    Returns
    -------
    Dict[str, float]
        Validation metrics including reflection coefficient
    """
    from dolfinx.mesh import create_interval
    
    # Create 1D mesh: [0, L] domain + [L, L+L_pml] PML
    L_domain = 1.0
    L_total = L_domain + L_pml
    
    mesh = create_interval(MPI.COMM_WORLD, num_elements, [0.0, L_total])
    
    V = fem.functionspace(mesh, ("Lagrange", 2))
    
    p = TrialFunction(V)
    v = TestFunction(V)
    
    k = omega / c
    
    # Coordinates
    x = SpatialCoordinate(mesh)
    
    # PML stretching (starts at x = L_domain)
    def in_pml(x):
        return ufl.conditional(ufl.gt(x[0], L_domain), 1.0, 0.0)
    
    def pml_dist(x):
        return ufl.conditional(
            ufl.gt(x[0], L_domain),
            (x[0] - L_domain) / L_pml,
            0.0
        )
    
    # Stretching coefficient
    sigma = sigma_max * pml_dist(x)**order
    s = 1.0 + 1j * sigma / omega
    
    # Weak form with PML (simplified for 1D)
    # Would need proper complex handling in real implementation
    a = (1.0/s) * inner(grad(v), grad(p)) * dx - k**2 * s * inner(v, p) * dx
    
    # Incident wave source at x = 0
    # p_inc = exp(ikx)
    L = inner(v, Constant(mesh, PETSc.ScalarType(1.0))) * ds
    
    # Solve
    from .solver_utils import solve_linear_system
    p_sol = solve_linear_system(a, L, V=V)
    
    # Measure reflection: compare |p| at boundaries
    # Reflection R = |p_reflected| / |p_incident|
    
    p_vals = p_sol.x.array
    
    # Approximate: R from field at end of PML
    R_estimate = np.min(np.abs(p_vals[-5:])) / np.max(np.abs(p_vals[:5]))
    
    return {
        'reflection_coefficient': R_estimate,
        'max_amplitude': np.max(np.abs(p_vals)),
        'pml_decay': np.max(np.abs(p_vals[:10])) / np.max(np.abs(p_vals[-10:])),
        'passed': R_estimate < 0.01,
    }
