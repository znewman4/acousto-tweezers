"""
Helmholtz Pressure Solver for Shallow Square Dish.

Solves the 3D complex Helmholtz equation with device-aligned BCs:
- BOTTOM (z=0): Vortex lens actuation (Neumann with vortex pattern)
- SIDE WALLS: Standing wave actuation (Neumann, configurable phase)
- TOP (z=H): Air interface (low impedance Robin BC)

Author: Acousto-Tweezers Project
Date: 2026-02-08
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass

from mpi4py import MPI
import dolfinx
from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem
import ufl
from ufl import inner, grad, dx, TrialFunction, TestFunction, Measure

from .config import ShallowDishConfig


# =============================================================================
# BOUNDARY TAGS
# =============================================================================

TAG_BOTTOM_DISC = 1   # z = 0, inside disc radius (vortex transducer)
TAG_BOTTOM_RIGID = 7  # z = 0, outside disc radius (rigid floor)
TAG_TOP = 2           # z = H (air interface)
TAG_X0 = 3            # x = 0 (standing wave or rigid)
TAG_XL = 4            # x = L (standing wave or rigid)
TAG_Y0 = 5            # y = 0 (standing wave or rigid)
TAG_YL = 6            # y = L (standing wave or rigid)

# Legacy alias — scripts may still use TAG_BOTTOM to mean "entire bottom".
# After segmentation, TAG_BOTTOM_DISC + TAG_BOTTOM_RIGID together equal it.
TAG_BOTTOM = TAG_BOTTOM_DISC  # kept for backward compat in streaming.py (no-slip)


# =============================================================================
# MESH GENERATION
# =============================================================================

def create_mesh(cfg: ShallowDishConfig, verbose: bool = True) -> Tuple[mesh.Mesh, mesh.MeshTags, Dict]:
    """
    Create 3D box mesh with explicit boundary tagging.
    
    Parameters
    ----------
    cfg : ShallowDishConfig
        Configuration
    verbose : bool
        Print mesh info
        
    Returns
    -------
    domain : mesh.Mesh
        The computational mesh
    facet_tags : mesh.MeshTags
        Boundary facet markers
    tag_map : dict
        Tag number to description mapping
    """
    L = cfg.L
    H = cfg.H
    nx = cfg.mesh_nx
    nz = cfg.mesh_nz
    
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    if verbose and rank == 0:
        print(f"\n{'='*70}")
        print("MESH GENERATION")
        print(f"{'='*70}")
        print(f"  Domain: {L*1e3:.1f} mm × {L*1e3:.1f} mm × {H*1e3:.2f} mm")
        print(f"  Grid: {nx} × {nx} × {nz}")
        print(f"  Wavelength: {cfg.wavelength*1e3:.2f} mm")
        print(f"  Depth: {cfg.n_wavelengths_depth:.2f} wavelengths")
    
    # Create box mesh
    domain = mesh.create_box(
        comm,
        [[0.0, 0.0, 0.0], [L, L, H]],
        [nx, nx, nz],
        cell_type=mesh.CellType.tetrahedron,
    )
    
    fdim = domain.topology.dim - 1
    domain.topology.create_connectivity(fdim, domain.topology.dim)
    
    # Define boundary locators with tolerance
    tol = min(L, H) * 1e-6
    
    def top(x):    return np.isclose(x[2], H, atol=tol)
    def x0(x):     return np.isclose(x[0], 0.0, atol=tol)
    def xL(x):     return np.isclose(x[0], L, atol=tol)
    def y0(x):     return np.isclose(x[1], 0.0, atol=tol)
    def yL(x):     return np.isclose(x[1], L, atol=tol)
    
    # Bottom facet segmentation: disc vs rigid
    R_disc = cfg.bottom_disc_radius_effective
    cx = cfg.vortex_center_x if cfg.vortex_center_x is not None else L / 2
    cy = cfg.vortex_center_y if cfg.vortex_center_y is not None else L / 2
    
    def bottom_disc(x):
        on_bottom = np.isclose(x[2], 0.0, atol=tol)
        r2 = (x[0] - cx)**2 + (x[1] - cy)**2
        return on_bottom & (r2 <= R_disc**2 + tol)
    
    def bottom_rigid(x):
        on_bottom = np.isclose(x[2], 0.0, atol=tol)
        r2 = (x[0] - cx)**2 + (x[1] - cy)**2
        return on_bottom & (r2 > R_disc**2 + tol)
    
    # Tag each boundary  (disc first so it wins ties at r=R_disc)
    boundaries = [
        (TAG_BOTTOM_DISC, bottom_disc),
        (TAG_BOTTOM_RIGID, bottom_rigid),
        (TAG_TOP, top),
        (TAG_X0, x0),
        (TAG_XL, xL),
        (TAG_Y0, y0),
        (TAG_YL, yL),
    ]
    
    facet_indices = []
    facet_markers = []
    
    for tag, locator in boundaries:
        facets = mesh.locate_entities_boundary(domain, fdim, locator)
        facet_indices.append(facets)
        facet_markers.append(np.full_like(facets, tag))
    
    # Combine and sort
    all_indices = np.hstack(facet_indices).astype(np.int32)
    all_markers = np.hstack(facet_markers).astype(np.int32)
    
    # Handle duplicates (keep first assignment)
    unique_mask = np.unique(all_indices, return_index=True)[1]
    sorted_indices = all_indices[unique_mask]
    sorted_markers = all_markers[unique_mask]
    
    order = np.argsort(sorted_indices)
    facet_tags = mesh.meshtags(domain, fdim, sorted_indices[order], sorted_markers[order])
    
    tag_map = {
        TAG_BOTTOM_DISC: f"z=0 disc (R={R_disc*1e3:.1f}mm, transducer)",
        TAG_BOTTOM_RIGID: "z=0 rigid (floor outside disc)",
        TAG_TOP: "z=H (TOP, air interface)",
        TAG_X0: "x=0 (standing or rigid)",
        TAG_XL: "x=L (standing or rigid)",
        TAG_Y0: "y=0 (standing or rigid)",
        TAG_YL: "y=L (standing or rigid)",
    }
    
    if verbose and rank == 0:
        print(f"\n  Boundary tags:")
        for tag, name in tag_map.items():
            count = np.sum(facet_tags.values == tag)
            print(f"    {tag}: {name} ({count} facets)")
        
        n_cells = domain.topology.index_map(domain.topology.dim).size_global
        n_verts = domain.topology.index_map(0).size_global
        print(f"\n  Mesh: {n_cells} cells, {n_verts} vertices")
        print(f"{'='*70}\n")
    
    return domain, facet_tags, tag_map


# =============================================================================
# HELMHOLTZ SOLVER
# =============================================================================

@dataclass
class PressureSolution:
    """Container for pressure solution and metadata."""
    p_function: fem.Function
    mode: str
    cfg: ShallowDishConfig
    
    @property
    def p_values(self) -> np.ndarray:
        return self.p_function.x.array.copy()
    
    @property
    def p_mag(self) -> np.ndarray:
        """Magnitude |p| of complex pressure phasor."""
        return np.abs(self.p_values)
    
    @property
    def p_phase(self) -> np.ndarray:
        """Phase angle arg(p) of complex pressure phasor."""
        return np.angle(self.p_values)
    
    @property
    def max_pressure(self) -> float:
        return np.max(self.p_mag)
    
    @property
    def coords(self) -> np.ndarray:
        return self.p_function.function_space.tabulate_dof_coordinates()
    
    @property
    def is_complex(self) -> bool:
        """Check if pressure values are complex."""
        return np.issubdtype(self.p_values.dtype, np.complexfloating)


def solve_helmholtz(
    domain: mesh.Mesh,
    facet_tags: mesh.MeshTags,
    cfg: ShallowDishConfig,
    mode: str = "combined",
    vortex_center: Optional[np.ndarray] = None,
    verbose: bool = True,
    disc_robin: bool = True,
    petsc_options: Optional[dict] = None,
) -> PressureSolution:
    """
    Solve complex Helmholtz equation with device-aligned BCs.
    
    This solver uses COMPLEX PHASOR representation for pressure.
    Requires PETSc built with complex scalar support (acousto-complex environment).
    
    Physics:
        ∇²p + k²p = 0  (Helmholtz in frequency domain)
        
    Boundary conditions:
        - Neumann (velocity actuation): ∂p/∂n = -iωρ v_n(x)  [v_n into domain]
        - Robin (impedance): ∂p/∂n = (iωρ/Z)p  [absorbing]
    
    Parameters
    ----------
    domain : mesh.Mesh
        Computational mesh
    facet_tags : mesh.MeshTags
        Boundary markers
    cfg : ShallowDishConfig
        Configuration
    mode : str
        "standing", "vortex", or "combined"
    vortex_center : np.ndarray, optional
        Override vortex center (x, y)
    verbose : bool
        Print progress
    disc_robin : bool
        If True (default), add impedance Robin BC on the bottom disc.
        Set False to make the entire bottom rigid (investigation mode).
    petsc_options : dict, optional
        PETSc solver options. Default: GMRES + ILU.
        For direct solve use e.g. {"ksp_type": "preonly", "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps"}.
        
    Returns
    -------
    PressureSolution
        Solution container with complex pressure phasor
    """
    from petsc4py import PETSc
    
    # Verify complex scalar support
    if not np.issubdtype(PETSc.ScalarType, np.complexfloating):
        raise RuntimeError(
            "Complex Helmholtz solver requires PETSc with complex scalars!\n"
            "Use: micromamba run -n acousto-complex python <script>"
        )
    
    L = cfg.L
    H = cfg.H
    omega = cfg.omega
    k = cfg.k
    rho = cfg.rho
    Z = cfg.Z_water
    Z_top = cfg.Z_top
    
    # Function space (P2 for accuracy) - will use complex scalars automatically
    V = fem.functionspace(domain, ("Lagrange", 2))
    
    # Measures
    dss = Measure("ds", domain=domain, subdomain_data=facet_tags)
    
    # Trial and test functions
    u = TrialFunction(V)
    v = TestFunction(V)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"SOLVING HELMHOLTZ [{mode.upper()}] (COMPLEX PHASOR)")
        print(f"{'='*70}")
        print(f"  PETSc ScalarType: {PETSc.ScalarType}")
    
    # =========================================================================
    # BILINEAR FORM (LHS): ∫(∇u·∇v̄ - k²u·v̄) dx + Robin BC terms
    # UFL's inner() handles complex conjugation of v automatically
    #
    # Derivation (standard form, no 1/ρ scaling):
    #   Strong PDE: ∇²p + k²p = 0
    #   Multiply by v̄, IBP: ∫∇p·∇v̄ dx - k²∫p·v̄ dx = ∮(∂p/∂n)v̄ ds
    #   Euler: ∇p = iωρ v⃗  →  ∂p/∂n = iωρ vₙ
    #   Impedance: p = Z vₙ  →  ∂p/∂n = (iωρ/Z)p
    #   Move Robin from RHS to LHS:  a -= (iωρ/Z)∫u·v̄ ds
    #   → Robin coefficient α = -(iωρ/Z)
    #   Source BC: ∂p/∂n = -iωρ v_n   →  adds -iωρ∫v_n v̄ ds to RHS
    #
    # Verified by 1D impedance test: α = -ik gives |R|≈0 for Z=ρc.
    #
    # WALL ROLE ASSIGNMENT (petri-dish model):
    #   Side walls (x±, y±) : NEVER impedance-matched absorbers.
    #     - Active transducer → pure Neumann source in RHS, NO Robin term.
    #     - Inactive          → rigid reflector = natural Neumann (∂p/∂n=0).
    #   Bottom disc (r ≤ R)  : impedance Robin (Z_water) + vortex source.
    #   Bottom rigid (r > R) : rigid = natural Neumann.
    #   Top (z = H)          : low-impedance Robin (air interface).
    #
    # Mode logic:
    #   standing → x-walls (and y-walls if axis=both) active, disc inactive
    #   vortex   → disc active, all side walls rigid
    #   combined → both active
    # =========================================================================
    a = (inner(grad(u), grad(v)) - k**2 * inner(u, v)) * dx
    
    # --- TOP (z=H): Air interface (low impedance Robin BC) ---
    if cfg.top_bc_type == "impedance":
        alpha_top = -1j * omega * rho / Z_top
        a += alpha_top * inner(u, v) * dss(TAG_TOP)
    
    # --- BOTTOM DISC: Impedance Robin (Z_water) ---
    # The disc always has impedance Robin whether or not the vortex source
    # is active.  A physical transducer is an impedance-matched boundary.
    # When disc_robin=False (investigation mode), the disc is treated as rigid.
    if disc_robin:
        alpha_disc = -1j * omega * rho / Z
        a += alpha_disc * inner(u, v) * dss(TAG_BOTTOM_DISC)
    
    # --- BOTTOM RIGID (r > R_disc): no term (natural Neumann = rigid) ---
    # --- SIDE WALLS: no Robin term — rigid or pure Neumann source ---
    
    if verbose:
        print(f"  BCs:")
        print(f"    Top:         impedance Robin (Z_top/Z_water = {cfg.top_impedance_factor})")
        if disc_robin:
            print(f"    Bottom disc: impedance Robin (Z = Z_water)")
        else:
            print(f"    Bottom disc: RIGID (disc_robin=False, investigation mode)")
        print(f"    Bottom rest: rigid (natural Neumann)")
        print(f"    Side walls:  rigid (+ Neumann source when active)")
    
    # =========================================================================
    # LINEAR FORM (RHS): Neumann actuation sources
    # From IBP: ∮(∂p/∂n)v̄ ds with ∂p/∂n = iωρ vₙ (Euler equation)
    # For a transducer pushing INTO domain: vₙ = -V (opposite to outward normal)
    #   → ∂p/∂n = -iωρ V
    # RHS contribution: L(v) = -iωρ ∫ V v̄ ds
    # Side walls: pure Neumann (no impedance part to subtract).
    # Bottom disc: impedance part already in bilinear form; add source here.
    # =========================================================================
    L_terms = []
    
    # --- STANDING WAVE (side walls, pure Neumann source) ---
    if mode in ("standing", "combined"):
        V_stand = cfg.standing_velocity_amplitude
        # Complex Neumann: g = -iωρ V_n
        g_stand = -1j * omega * rho * V_stand
        
        if cfg.standing_phase_pattern == "antiphase":
            # x=0: phase 0, x=L: phase π (opposite sign)
            L_terms.append(inner(g_stand, v) * dss(TAG_X0))
            L_terms.append(inner(-g_stand, v) * dss(TAG_XL))
        elif cfg.standing_phase_pattern == "inphase":
            L_terms.append(inner(g_stand, v) * dss(TAG_X0))
            L_terms.append(inner(g_stand, v) * dss(TAG_XL))
        elif cfg.standing_phase_pattern == "quadrature":
            # x=0: phase 0, x=L: phase π/2
            L_terms.append(inner(g_stand, v) * dss(TAG_X0))
            L_terms.append(inner(1j * g_stand, v) * dss(TAG_XL))
        
        if cfg.standing_axis == "both":
            if cfg.standing_phase_pattern == "antiphase":
                L_terms.append(inner(g_stand, v) * dss(TAG_Y0))
                L_terms.append(inner(-g_stand, v) * dss(TAG_YL))
            elif cfg.standing_phase_pattern == "inphase":
                L_terms.append(inner(g_stand, v) * dss(TAG_Y0))
                L_terms.append(inner(g_stand, v) * dss(TAG_YL))
            elif cfg.standing_phase_pattern == "quadrature":
                L_terms.append(inner(g_stand, v) * dss(TAG_Y0))
                L_terms.append(inner(1j * g_stand, v) * dss(TAG_YL))
        
        if verbose:
            print(f"  Standing wave: V={V_stand*1e6:.1f} μm/s, "
                  f"pattern={cfg.standing_phase_pattern}, axis={cfg.standing_axis}")
    
    # --- VORTEX (Bottom disc only) ---
    if mode in ("vortex", "combined"):
        V_vtx = cfg.vortex_velocity_amplitude
        ell = cfg.vortex_topological_charge
        
        # Create complex vortex pattern with phase singularity
        # v_n(x,y) = A(r) * exp(i * ℓ * θ)
        g_vtx = _create_vortex_source(V, domain, facet_tags, cfg, vortex_center, verbose)
        
        # Scale by -iωρ V_vtx
        g_vtx.x.array[:] *= -1j * omega * rho * V_vtx
        
        # Integrate over disc subdomain only (not entire bottom)
        L_terms.append(inner(g_vtx, v) * dss(TAG_BOTTOM_DISC))
        
        if verbose:
            print(f"  Vortex: V={V_vtx*1e6:.1f} μm/s, ℓ={ell}")
    
    # Combine RHS
    if len(L_terms) == 0:
        raise ValueError(f"No actuation sources for mode='{mode}'")
    
    L_form = L_terms[0]
    for term in L_terms[1:]:
        L_form = L_form + term
    
    # =========================================================================
    # SOLVE (using complex linear algebra)
    # =========================================================================
    if petsc_options is None:
        petsc_options = {
            "ksp_type": "gmres",
            "ksp_rtol": 1e-8,
            "ksp_max_it": 3000,
            "pc_type": "ilu",
        }
    problem = LinearProblem(
        a, L_form, bcs=[],
        petsc_options=petsc_options,
    )
    
    p_solution = problem.solve()
    p_solution.name = f"pressure_{mode}"
    
    # Stats
    p_vals = p_solution.x.array[:]
    max_mag = np.max(np.abs(p_vals))
    
    if verbose:
        print(f"\n  Solved: max|p| = {max_mag:.2f} Pa, DOFs = {len(p_vals)}")
        print(f"  Solution dtype: {p_vals.dtype}")
        print(f"  Is complex: {np.issubdtype(p_vals.dtype, np.complexfloating)}")
        print(f"{'='*70}\n")
    
    return PressureSolution(
        p_function=p_solution,
        mode=mode,
        cfg=cfg,
    )


def _create_vortex_source(
    V: fem.FunctionSpace,
    domain: mesh.Mesh,
    facet_tags: mesh.MeshTags,
    cfg: ShallowDishConfig,
    center: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> fem.Function:
    """
    Create complex vortex pattern function for BOTTOM boundary actuation.
    
    Pattern: A(r) * exp(i * ℓ * θ + i * φ₀)
    
    where:
        - A(r) = amplitude profile (cosine taper, gaussian, or uniform)
        - θ = azimuthal angle around vortex center
        - ℓ = topological charge (vortex winding number)
        - φ₀ = initial phase offset
    
    Parameters
    ----------
    V : fem.FunctionSpace
        Scalar function space
    domain : mesh.Mesh
        Computational mesh
    facet_tags : mesh.MeshTags
        Boundary tags
    cfg : ShallowDishConfig
        Configuration
    center : np.ndarray, optional
        Override vortex center (x, y)
    verbose : bool
        Print info
        
    Returns
    -------
    fem.Function
        Complex vortex pattern (nonzero only on bottom DOFs)
    """
    L = cfg.L
    
    # Vortex center
    if center is not None:
        cx, cy = center[0], center[1]
    else:
        cx = cfg.vortex_center_x if cfg.vortex_center_x is not None else L/2
        cy = cfg.vortex_center_y if cfg.vortex_center_y is not None else L/2
    
    fdim = domain.topology.dim - 1
    
    # Get bottom DISC boundary DOFs (only the transducer patch, not the rigid floor)
    disc_facets = facet_tags.indices[facet_tags.values == TAG_BOTTOM_DISC]
    disc_dofs = fem.locate_dofs_topological(V, fdim, disc_facets)
    
    # DOF coordinates
    dof_coords = V.tabulate_dof_coordinates()
    disc_coords = dof_coords[disc_dofs]
    
    # Azimuthal angle θ = atan2(y - cy, x - cx)
    dx_arr = disc_coords[:, 0] - cx
    dy_arr = disc_coords[:, 1] - cy
    theta = np.arctan2(dy_arr, dx_arr)
    
    # Radial distance
    r = np.sqrt(dx_arr**2 + dy_arr**2)
    
    # Amplitude profile (within disc, all DOFs are ≤ R_aperture by construction
    # when disc_radius == aperture_radius, but we still apply the profile)
    R_aperture = cfg.vortex_aperture_radius
    amplitude = np.zeros_like(r)
    inside = r <= R_aperture
    
    if cfg.vortex_apodization == "cosine_taper":
        amplitude[inside] = 0.5 * (1 + np.cos(np.pi * r[inside] / R_aperture))
    elif cfg.vortex_apodization == "uniform":
        amplitude[inside] = 1.0
    elif cfg.vortex_apodization == "gaussian":
        sigma = R_aperture / 2
        amplitude = np.exp(-r**2 / (2*sigma**2))
    else:
        amplitude[inside] = 1.0
    
    # Complex vortex phase: exp(i*(ℓ*θ + φ₀))
    ell = cfg.vortex_topological_charge
    phi0 = cfg.vortex_phase_offset
    phase = ell * theta + phi0
    
    # Complex pattern (normalized to 1)
    pattern = amplitude * np.exp(1j * phase)
    
    # Create complex function
    vortex_func = fem.Function(V)
    vortex_func.x.array[:] = 0.0 + 0.0j
    vortex_func.x.array[disc_dofs] = pattern
    
    if verbose:
        n_active = np.sum(np.abs(pattern) > 1e-10)
        print(f"  Vortex pattern:")
        print(f"    Center: ({cx*1e3:.2f}, {cy*1e3:.2f}) mm")
        print(f"    Aperture: {R_aperture*1e3:.1f} mm")
        print(f"    Charge ℓ: {ell}")
        print(f"    Active DOFs: {n_active}/{len(disc_dofs)} (disc only)")
    
    return vortex_func


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def solve_all_pressure_cases(
    cfg: ShallowDishConfig,
    vortex_center: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> Dict[str, PressureSolution]:
    """
    Solve standing, vortex, and combined cases.
    
    Parameters
    ----------
    cfg : ShallowDishConfig
        Configuration
    vortex_center : np.ndarray, optional
        Override vortex center
    verbose : bool
        Print progress
        
    Returns
    -------
    dict
        {"standing": ..., "vortex": ..., "combined": ...}
    """
    # Create mesh once
    domain, facet_tags, tag_map = create_mesh(cfg, verbose=verbose)
    
    results = {}
    
    for mode in ["standing", "vortex", "combined"]:
        results[mode] = solve_helmholtz(
            domain, facet_tags, cfg,
            mode=mode,
            vortex_center=vortex_center,
            verbose=verbose,
        )
    
    return results


# =============================================================================
# PHASE WINDING DIAGNOSTIC
# =============================================================================

def compute_phase_winding(
    p_solution: PressureSolution,
    center_xy: Tuple[float, float],
    radius: float,
    z: float,
    n_samples: int = 200,
) -> float:
    """
    Compute phase winding number around a circular contour.
    
    For a vortex with topological charge ℓ, the phase of p(x) should
    wind by 2πℓ around a closed contour encircling the vortex axis.
    
    This function samples the complex pressure around a circle,
    computes the total phase change (unwrapped), and returns the
    winding number (total phase change / 2π).
    
    Parameters
    ----------
    p_solution : PressureSolution
        Complex pressure solution from solve_helmholtz
    center_xy : tuple
        (x, y) center of the contour circle
    radius : float
        Radius of the sampling circle (meters)
    z : float
        z-coordinate of the sampling plane (meters)
    n_samples : int
        Number of points to sample around the circle
        
    Returns
    -------
    float
        Winding number (should be close to topological charge ℓ)
    """
    from scipy.interpolate import NearestNDInterpolator
    
    p_func = p_solution.p_function
    p_vals = p_func.x.array
    coords = p_solution.coords
    
    # Create interpolator for complex pressure
    # Use nearest-neighbor for simplicity (mesh is fine enough)
    interp_real = NearestNDInterpolator(coords, np.real(p_vals))
    interp_imag = NearestNDInterpolator(coords, np.imag(p_vals))
    
    # Sample points on circle
    cx, cy = center_xy
    theta_samples = np.linspace(0, 2*np.pi, n_samples, endpoint=False)
    
    x_samples = cx + radius * np.cos(theta_samples)
    y_samples = cy + radius * np.sin(theta_samples)
    z_samples = np.full_like(x_samples, z)
    
    sample_pts = np.column_stack([x_samples, y_samples, z_samples])
    
    # Interpolate complex pressure
    p_real = interp_real(sample_pts)
    p_imag = interp_imag(sample_pts)
    p_samples = p_real + 1j * p_imag
    
    # Compute phase angles
    phase = np.angle(p_samples)
    
    # Unwrap phase
    phase_unwrapped = np.unwrap(phase)
    
    # Total phase change
    total_phase_change = phase_unwrapped[-1] - phase_unwrapped[0]
    
    # Account for the wrap-around (close the loop)
    # Add the phase jump from last point back to first
    delta_final = np.angle(p_samples[0]) - np.angle(p_samples[-1])
    delta_final_unwrapped = np.arctan2(np.sin(delta_final), np.cos(delta_final))
    total_phase_change += delta_final_unwrapped
    
    # Winding number
    winding_number = total_phase_change / (2 * np.pi)
    
    return winding_number
