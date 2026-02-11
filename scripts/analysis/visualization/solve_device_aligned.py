#!/usr/bin/env python3
"""
Device-Aligned Helmholtz Solver for ParaView Export
=====================================================

PHYSICAL CONFIGURATION (matches real device):
- BOTTOM (z=0): Vortex lens transducer (azimuthal phase pattern)
- SIDE WALLS (x=0, x=L): Standing wave transducers (anti-phase)
- TOP (z=L): Air interface (impedance BC, NOT driven)
- Other walls: Passive (impedance or rigid)

This creates a vortex beam propagating UPWARD through the volume,
interacting with the horizontal standing wave in the BULK.

Output: ParaView-compatible VTU files on Windows OneDrive.

Author: Acousto-Tweezers Project
Date: 2026-02-08
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
import numpy as np
from dataclasses import dataclass, asdict
from typing import Tuple, Dict, Optional

from mpi4py import MPI
import dolfinx
from dolfinx import fem, mesh, io
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile, VTXWriter
import ufl
from ufl import inner, grad, dx, ds, TrialFunction, TestFunction, Measure
import pyvista as pv

# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class DeviceConfig:
    """Physical configuration for device-aligned simulation."""
    # Domain
    dish_size_m: float = 0.02          # 20 mm cube
    frequency_hz: float = 500e3        # 500 kHz
    
    # Material (water)
    rho: float = 997.0                 # kg/m³
    c: float = 1484.0                  # m/s
    
    # Vortex (BOTTOM transducer)
    vortex_velocity_amplitude: float = 10e-6  # m/s (V_vtx)
    vortex_aperture_radius: float = 0.004     # 4 mm radius
    vortex_topological_charge: int = 1        # ℓ = 1
    vortex_phase_offset: float = 0.0          # φ₀
    
    # Standing wave (SIDE transducers on x-walls)
    standing_velocity_amplitude: float = 1e-6  # m/s (V_stand)
    standing_axis: str = "x"                   # "x" or "y"
    
    # Boundary conditions
    top_bc_type: str = "impedance"     # "impedance" or "dirichlet"
    top_impedance_factor: float = 0.01  # Z_top = factor * Z_water (low = more open)
    passive_wall_bc: str = "impedance"  # "impedance" or "rigid"
    
    # Mesh
    elements_per_wavelength: int = 6
    
    @property
    def omega(self) -> float:
        return 2 * np.pi * self.frequency_hz
    
    @property
    def k(self) -> float:
        return self.omega / self.c
    
    @property
    def wavelength(self) -> float:
        return self.c / self.frequency_hz
    
    @property
    def Z_water(self) -> float:
        return self.rho * self.c


# ==============================================================================
# MESH GENERATION WITH EXPLICIT BOUNDARY TAGGING
# ==============================================================================

# Boundary tags (explicit)
TAG_BOTTOM = 1  # z = 0 (vortex source)
TAG_TOP = 2     # z = L (air interface)
TAG_X0 = 3      # x = 0 (standing transducer +)
TAG_XL = 4      # x = L (standing transducer -)
TAG_Y0 = 5      # y = 0 (passive)
TAG_YL = 6      # y = L (passive)


def create_mesh_with_tags(cfg: DeviceConfig, verbose: bool = True) -> Tuple:
    """
    Create 3D box mesh with EXPLICIT boundary tagging.
    
    Returns: (domain, facet_tags, tag_map)
    """
    L = cfg.dish_size_m
    wavelength = cfg.wavelength
    nx = max(10, int(L / wavelength * cfg.elements_per_wavelength))
    
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    if verbose and rank == 0:
        print(f"\n{'='*70}")
        print("MESH GENERATION")
        print(f"{'='*70}")
        print(f"  Domain size: {L*1e3:.1f} mm × {L*1e3:.1f} mm × {L*1e3:.1f} mm")
        print(f"  Wavelength: {wavelength*1e3:.3f} mm")
        print(f"  Elements per wavelength: {cfg.elements_per_wavelength}")
        print(f"  Grid: {nx} × {nx} × {nx}")
    
    # Create box mesh
    domain = mesh.create_box(
        comm,
        [[0.0, 0.0, 0.0], [L, L, L]],
        [nx, nx, nx],
        cell_type=mesh.CellType.tetrahedron,
    )
    
    fdim = domain.topology.dim - 1
    domain.topology.create_connectivity(fdim, domain.topology.dim)
    
    # Define boundary locators with tolerance
    tol = L * 1e-6
    
    def bottom(x): return np.isclose(x[2], 0.0, atol=tol)
    def top(x):    return np.isclose(x[2], L, atol=tol)
    def x0(x):     return np.isclose(x[0], 0.0, atol=tol)
    def xL(x):     return np.isclose(x[0], L, atol=tol)
    def y0(x):     return np.isclose(x[1], 0.0, atol=tol)
    def yL(x):     return np.isclose(x[1], L, atol=tol)
    
    # Locate facets for each boundary
    boundaries = [
        (TAG_BOTTOM, bottom, "z=0 (bottom, vortex)"),
        (TAG_TOP, top, "z=L (top, air interface)"),
        (TAG_X0, x0, "x=0 (standing +)"),
        (TAG_XL, xL, "x=L (standing -)"),
        (TAG_Y0, y0, "y=0 (passive)"),
        (TAG_YL, yL, "y=L (passive)"),
    ]
    
    facet_indices = []
    facet_markers = []
    
    for tag, locator, name in boundaries:
        facets = mesh.locate_entities_boundary(domain, fdim, locator)
        facet_indices.append(facets)
        facet_markers.append(np.full_like(facets, tag))
    
    # Combine and sort
    all_indices = np.hstack(facet_indices).astype(np.int32)
    all_markers = np.hstack(facet_markers).astype(np.int32)
    
    # Remove duplicates (edges/corners may be tagged multiple times)
    unique_mask = np.unique(all_indices, return_index=True)[1]
    sorted_indices = all_indices[unique_mask]
    sorted_markers = all_markers[unique_mask]
    
    # Sort by index
    order = np.argsort(sorted_indices)
    facet_tags = mesh.meshtags(domain, fdim, sorted_indices[order], sorted_markers[order])
    
    # Create tag map for documentation
    tag_map = {
        TAG_BOTTOM: "z=0 (bottom, VORTEX SOURCE)",
        TAG_TOP: "z=L (top, AIR INTERFACE)",
        TAG_X0: "x=0 (standing wave +)",
        TAG_XL: "x=L (standing wave -)",
        TAG_Y0: "y=0 (passive wall)",
        TAG_YL: "y=L (passive wall)",
    }
    
    # Verify tagging by sampling
    if verbose and rank == 0:
        print(f"\n  BOUNDARY TAG VERIFICATION:")
        for tag, name in tag_map.items():
            count = np.sum(facet_tags.values == tag)
            print(f"    Tag {tag}: {name} → {count} facets")
    
    n_cells = domain.topology.index_map(domain.topology.dim).size_global
    n_vertices = domain.topology.index_map(0).size_global
    
    if verbose and rank == 0:
        print(f"\n  Mesh: {n_cells} cells, {n_vertices} vertices")
        print(f"{'='*70}\n")
    
    return domain, facet_tags, tag_map


def verify_boundary_tags(domain, facet_tags, cfg: DeviceConfig, verbose: bool = True):
    """
    Verify boundary tags by sampling facet midpoints.
    Raises AssertionError if tags don't match expected coordinates.
    """
    L = cfg.dish_size_m
    tol = L * 0.01  # 1% tolerance
    
    fdim = domain.topology.dim - 1
    
    # Get facet midpoints
    facet_midpoints = mesh.compute_midpoints(domain, fdim, facet_tags.indices)
    
    checks = [
        (TAG_BOTTOM, 2, 0.0, "z=0"),
        (TAG_TOP, 2, L, "z=L"),
        (TAG_X0, 0, 0.0, "x=0"),
        (TAG_XL, 0, L, "x=L"),
        (TAG_Y0, 1, 0.0, "y=0"),
        (TAG_YL, 1, L, "y=L"),
    ]
    
    if verbose:
        print("BOUNDARY TAG COORDINATE VERIFICATION:")
    
    for tag, axis, expected_val, name in checks:
        mask = facet_tags.values == tag
        if np.sum(mask) == 0:
            continue
        
        # Find indices in facet_tags.indices that have this tag value
        tag_facet_indices = facet_tags.indices[mask]
        
        # Get midpoints for these facets
        coords = mesh.compute_midpoints(domain, fdim, tag_facet_indices)
        
        mean_coord = np.mean(coords[:, axis])
        std_coord = np.std(coords[:, axis])
        
        passed = abs(mean_coord - expected_val) < tol and std_coord < tol
        status = "✓" if passed else "✗"
        
        if verbose:
            print(f"  {status} Tag {tag} ({name}): mean={mean_coord:.6f}, std={std_coord:.6f}, expected={expected_val:.6f}")
        
        if not passed:
            raise AssertionError(f"Boundary tag {tag} ({name}) failed verification: "
                               f"mean={mean_coord:.6f}, expected={expected_val:.6f}")
    
    if verbose:
        print("  All boundary tags verified ✓\n")


# ==============================================================================
# VORTEX PATTERN (BOTTOM BOUNDARY)
# ==============================================================================

def create_vortex_pattern(V: fem.FunctionSpace, domain, facet_tags, cfg: DeviceConfig, 
                          verbose: bool = True) -> fem.Function:
    """
    Create vortex pattern function for BOTTOM boundary actuation.
    
    v_n(x,y) = V_vtx * A(r) * exp(i*(ℓ*θ + φ₀))
    
    Returns: fem.Function with complex pattern set on bottom boundary DOFs only.
    """
    L = cfg.dish_size_m
    center = np.array([L/2, L/2, 0.0])  # Vortex center at BOTTOM
    
    fdim = domain.topology.dim - 1
    
    # Get bottom boundary facets
    bottom_facets = facet_tags.indices[facet_tags.values == TAG_BOTTOM]
    bottom_dofs = fem.locate_dofs_topological(V, fdim, bottom_facets)
    
    # Get DOF coordinates
    dof_coords = V.tabulate_dof_coordinates()
    bottom_coords = dof_coords[bottom_dofs]
    
    # Compute azimuthal angle θ = atan2(y - yc, x - xc)
    dx = bottom_coords[:, 0] - center[0]
    dy = bottom_coords[:, 1] - center[1]
    theta = np.arctan2(dy, dx)
    
    # Compute radial distance from center
    r = np.sqrt(dx**2 + dy**2)
    
    # Amplitude profile: cosine taper within aperture
    R_aperture = cfg.vortex_aperture_radius
    amplitude = np.zeros_like(r)
    inside = r <= R_aperture
    if np.any(inside):
        # Cosine taper: 1 at center, 0 at edge
        amplitude[inside] = 0.5 * (1 + np.cos(np.pi * r[inside] / R_aperture))
    
    # Vortex phase: exp(i*(ℓ*θ + φ₀))
    ell = cfg.vortex_topological_charge
    phi0 = cfg.vortex_phase_offset
    phase = ell * theta + phi0
    
    # Complex pattern
    pattern = amplitude * np.exp(1j * phase)
    
    # Create function (zero everywhere, pattern on bottom DOFs)
    vortex_func = fem.Function(V)
    vortex_func.x.array[:] = 0.0 + 0.0j
    vortex_func.x.array[bottom_dofs] = pattern
    
    if verbose:
        n_active = np.sum(np.abs(pattern) > 1e-10)
        print(f"VORTEX PATTERN (bottom boundary):")
        print(f"  Center: ({center[0]*1e3:.1f}, {center[1]*1e3:.1f}, {center[2]*1e3:.1f}) mm")
        print(f"  Aperture radius: {R_aperture*1e3:.1f} mm")
        print(f"  Topological charge ℓ: {ell}")
        print(f"  Active DOFs: {n_active} / {len(bottom_dofs)}")
        print(f"  Max amplitude: {np.max(np.abs(pattern)):.3f}")
    
    return vortex_func


# ==============================================================================
# HELMHOLTZ SOLVER
# ==============================================================================

def solve_helmholtz(domain, facet_tags, cfg: DeviceConfig, 
                    mode: str = "combined",
                    verbose: bool = True) -> fem.Function:
    """
    Solve Helmholtz equation with device-aligned boundary conditions.
    
    Modes:
        'standing': Side wall actuation only (no vortex)
        'vortex': Bottom vortex actuation only (no standing)
        'combined': Both standing and vortex simultaneously
    
    Boundary conditions:
        BOTTOM (z=0): Vortex actuation (if mode includes vortex)
        TOP (z=L): Impedance BC (air interface)
        X-walls: Standing wave actuation (if mode includes standing)
        Y-walls: Passive (impedance or rigid)
    """
    L = cfg.dish_size_m
    omega = cfg.omega
    k = cfg.k
    rho = cfg.rho
    Z = cfg.Z_water
    
    # Function space (P2 for accuracy)
    V = fem.functionspace(domain, ("Lagrange", 2))
    
    # Measures
    dxx = dx
    dss = Measure("ds", domain=domain, subdomain_data=facet_tags)
    
    # Trial and test functions (match generate_rich_data.py convention)
    u = TrialFunction(V)
    v = TestFunction(V)
    
    # =========================================================================
    # BILINEAR FORM (LHS) 
    # Pattern from generate_rich_data.py: inner(grad(u), grad(v)), inner(u, v)
    # =========================================================================
    # Bulk Helmholtz
    a = (inner(grad(u), grad(v)) / rho - k**2 * inner(u, v) / rho) * dxx
    
    # --- TOP BOUNDARY (z=L): Air interface ---
    if cfg.top_bc_type == "impedance":
        # Robin BC: ∂p/∂n = -iω/Z_top * p
        Z_top = cfg.top_impedance_factor * Z
        a += (-1j * omega / Z_top) * inner(u, v) * dss(TAG_TOP)
    # If Dirichlet (p=0), handled separately
    
    # --- PASSIVE WALLS (y-walls) ---
    if cfg.passive_wall_bc == "impedance":
        # Impedance BC on y-walls
        a += (-1j * omega / Z) * inner(u, v) * dss(TAG_Y0)
        a += (-1j * omega / Z) * inner(u, v) * dss(TAG_YL)
    # If rigid, Neumann ∂p/∂n=0 is natural BC (no term needed)
    
    # --- X-WALLS: Impedance even if driven (mixed BC) ---
    # We apply impedance + actuation source
    a += (-1j * omega / Z) * inner(u, v) * dss(TAG_X0)
    a += (-1j * omega / Z) * inner(u, v) * dss(TAG_XL)
    
    # --- BOTTOM (z=0): Impedance even for vortex ---
    a += (-1j * omega / Z) * inner(u, v) * dss(TAG_BOTTOM)
    
    # =========================================================================
    # LINEAR FORM (RHS) - Actuation sources
    # Pattern from generate_rich_data.py: g_s * inner(1.0, v), inner(g_v, v)
    # Build L_form from pieces instead of initializing to zero
    # =========================================================================
    L_terms = []  # Collect RHS terms
    
    # --- STANDING WAVE ACTUATION (x-walls) ---
    if mode in ("standing", "combined"):
        V_stand = cfg.standing_velocity_amplitude
        g_stand = -1j * omega * rho * V_stand
        
        # x=0: positive phase, x=L: negative phase (anti-phase)
        # Match generate_rich_data.py pattern: g_s * inner(1.0, v)
        L_terms.append(g_stand * inner(1.0, v) * dss(TAG_X0))
        L_terms.append((-g_stand) * inner(1.0, v) * dss(TAG_XL))
        
        if verbose:
            print(f"STANDING WAVE ACTUATION (x-walls):")
            print(f"  Velocity amplitude: {V_stand*1e6:.2f} μm/s")
            print(f"  |g| = |−iωρV|: {np.abs(g_stand):.2e}")
    
    # --- VORTEX ACTUATION (bottom) ---
    if mode in ("vortex", "combined"):
        V_vtx = cfg.vortex_velocity_amplitude
        
        # Create vortex pattern function
        vortex_pattern = create_vortex_pattern(V, domain, facet_tags, cfg, verbose=verbose)
        
        # g_vtx = -iωρ * V_vtx * pattern
        # Match generate_rich_data.py pattern: inner(g_v, v)
        g_vtx = fem.Function(V)
        g_vtx.x.array[:] = -1j * omega * rho * V_vtx * vortex_pattern.x.array[:]
        
        L_terms.append(inner(g_vtx, v) * dss(TAG_BOTTOM))
    
    # Combine all L_form terms
    if len(L_terms) == 0:
        raise ValueError(f"No actuation sources for mode='{mode}'")
    L_form = L_terms[0]
    for term in L_terms[1:]:
        L_form = L_form + term
    
    # =========================================================================
    # SOLVE
    # =========================================================================
    if verbose:
        print(f"\nSOLVING HELMHOLTZ [{mode.upper()}]...")
    
    problem = LinearProblem(
        a, L_form, bcs=[],
        petsc_options={
            "ksp_type": "gmres",
            "ksp_rtol": 1e-8,
            "ksp_max_it": 3000,
            "pc_type": "ilu",
        }
    )
    
    p_solution = problem.solve()
    p_solution.name = f"pressure_{mode}"
    
    # Statistics
    p_vals = p_solution.x.array[:]
    max_mag = np.max(np.abs(p_vals))
    
    if verbose:
        print(f"  Solved: max|p| = {max_mag:.1f} Pa")
        print(f"  DOFs: {len(p_vals)}")
    
    return p_solution


# ==============================================================================
# DIAGNOSTICS
# ==============================================================================

def compute_diagnostics(p_standing: fem.Function, 
                        p_vortex: fem.Function,
                        p_combined: fem.Function,
                        cfg: DeviceConfig,
                        verbose: bool = True) -> Dict:
    """
    Compute quantitative diagnostics to verify bulk propagation.
    """
    L = cfg.dish_size_m
    V = p_combined.function_space
    coords = V.tabulate_dof_coordinates()
    
    # Compute delta_p
    delta_p = p_combined.x.array[:] - p_standing.x.array[:]
    delta_p_mag = np.abs(delta_p)
    
    # Z-coordinates
    z = coords[:, 2]
    
    # Define slabs
    bottom_slab = z < 0.05 * L
    mid_slab = (z > 0.45 * L) & (z < 0.55 * L)
    top_slab = z > 0.95 * L
    
    # Compute max |Δp| in each slab
    max_bottom = np.max(delta_p_mag[bottom_slab]) if np.any(bottom_slab) else 0
    max_mid = np.max(delta_p_mag[mid_slab]) if np.any(mid_slab) else 0
    max_top = np.max(delta_p_mag[top_slab]) if np.any(top_slab) else 0
    
    # Bulk ratio: mid / bottom
    bulk_ratio = max_mid / max_bottom if max_bottom > 1e-10 else 0
    
    # Z-profile at vortex center
    center_x, center_y = L/2, L/2
    tol = L * 0.05
    center_mask = (np.abs(coords[:, 0] - center_x) < tol) & \
                  (np.abs(coords[:, 1] - center_y) < tol)
    
    if np.any(center_mask):
        z_center = z[center_mask]
        delta_center = delta_p_mag[center_mask]
        order = np.argsort(z_center)
        z_profile = z_center[order]
        delta_profile = delta_center[order]
    else:
        z_profile = np.array([0, L])
        delta_profile = np.array([0, 0])
    
    # Standing wave strength
    max_standing = np.max(np.abs(p_standing.x.array[:]))
    max_combined = np.max(np.abs(p_combined.x.array[:]))
    max_vortex = np.max(np.abs(p_vortex.x.array[:]))
    max_delta = np.max(delta_p_mag)
    
    diagnostics = {
        "max_delta_p_bottom": float(max_bottom),
        "max_delta_p_mid": float(max_mid),
        "max_delta_p_top": float(max_top),
        "bulk_ratio_mid_over_bottom": float(bulk_ratio),
        "max_standing": float(max_standing),
        "max_vortex": float(max_vortex),
        "max_combined": float(max_combined),
        "max_delta_p": float(max_delta),
        "z_profile_sample": {
            "z": z_profile[::max(1, len(z_profile)//10)].tolist(),
            "delta_p_mag": delta_profile[::max(1, len(delta_profile)//10)].tolist(),
        },
        "domain_size_m": float(L),
        "wavelength_m": float(cfg.wavelength),
    }
    
    if verbose:
        print(f"\n{'='*70}")
        print("DIAGNOSTICS: BULK PROPAGATION CHECK")
        print(f"{'='*70}")
        print(f"  max|Δp| at bottom (z<0.05L): {max_bottom:.1f} Pa")
        print(f"  max|Δp| at mid (0.45L<z<0.55L): {max_mid:.1f} Pa")
        print(f"  max|Δp| at top (z>0.95L): {max_top:.1f} Pa")
        print(f"  BULK RATIO (mid/bottom): {bulk_ratio:.4f}")
        
        if bulk_ratio < 1e-3:
            print(f"  ⚠️ WARNING: Vortex NOT propagating into bulk (ratio < 0.001)")
        elif bulk_ratio < 0.1:
            print(f"  ⚠️ Vortex partially propagates (ratio < 0.1)")
        else:
            print(f"  ✓ Vortex propagates into bulk (ratio ≥ 0.1)")
        
        print(f"\n  Field strengths:")
        print(f"    max|p_standing|: {max_standing:.1f} Pa")
        print(f"    max|p_vortex|:   {max_vortex:.1f} Pa")
        print(f"    max|p_combined|: {max_combined:.1f} Pa")
        print(f"    max|Δp|:         {max_delta:.1f} Pa")
        
        print(f"\n  Z-profile at vortex center (sampled):")
        n_samples = min(10, len(z_profile))
        indices = np.linspace(0, len(z_profile)-1, n_samples, dtype=int)
        for i in indices:
            print(f"    z={z_profile[i]*1e3:6.2f} mm: |Δp|={delta_profile[i]:.1f} Pa")
        
        print(f"{'='*70}\n")
    
    return diagnostics


# ==============================================================================
# PARAVIEW EXPORT
# ==============================================================================

def export_to_paraview(domain, facet_tags,
                       p_standing: fem.Function,
                       p_vortex: fem.Function,
                       p_combined: fem.Function,
                       cfg: DeviceConfig,
                       diagnostics: Dict,
                       output_dir: Path,
                       verbose: bool = True) -> None:
    """
    Export all fields to ParaView-compatible VTU format.
    """
    V = p_combined.function_space
    coords = V.tabulate_dof_coordinates()
    n_pts = coords.shape[0]
    
    # Compute all fields
    standing = p_standing.x.array[:]
    vortex = p_vortex.x.array[:]
    combined = p_combined.x.array[:]
    delta = combined - standing
    
    if verbose:
        print(f"\n{'='*70}")
        print("PARAVIEW EXPORT")
        print(f"{'='*70}")
        print(f"  Output directory: {output_dir}")
        print(f"  Points: {n_pts}")
    
    # Create output directories
    (output_dir / "mesh").mkdir(parents=True, exist_ok=True)
    (output_dir / "fields_standing").mkdir(parents=True, exist_ok=True)
    (output_dir / "fields_vortex").mkdir(parents=True, exist_ok=True)
    (output_dir / "fields_combined").mkdir(parents=True, exist_ok=True)
    (output_dir / "fields_delta").mkdir(parents=True, exist_ok=True)
    (output_dir / "meta").mkdir(parents=True, exist_ok=True)
    
    # Helper to create VTU from field values
    def export_field_vtu(field_values: np.ndarray, name_prefix: str, 
                         out_folder: Path, include_ref: bool = False):
        """Export complex field to VTU with real, imag, mag, phase."""
        
        # Create PyVista unstructured grid from mesh
        topology, cell_types, x = dolfinx.plot.vtk_mesh(V)
        grid = pv.UnstructuredGrid(topology, cell_types, x)
        
        # Add field components
        p_real = np.real(field_values).astype(np.float64)
        p_imag = np.imag(field_values).astype(np.float64)
        p_mag = np.abs(field_values).astype(np.float64)
        p_phase = np.angle(field_values).astype(np.float64)
        
        grid.point_data[f"{name_prefix}_real"] = p_real
        grid.point_data[f"{name_prefix}_imag"] = p_imag
        grid.point_data[f"{name_prefix}_mag"] = p_mag
        grid.point_data[f"{name_prefix}_phase"] = p_phase
        
        # Optionally include reference fields for comparison
        if include_ref:
            grid.point_data["standing_p_mag"] = np.abs(standing).astype(np.float64)
            grid.point_data["standing_p_phase"] = np.angle(standing).astype(np.float64)
        
        # Save
        out_file = out_folder / f"{name_prefix}_fields.vtu"
        grid.save(str(out_file))
        
        if verbose:
            print(f"  Exported: {out_file.name}")
            print(f"    {name_prefix}_mag range: [{np.min(p_mag):.1f}, {np.max(p_mag):.1f}] Pa")
            print(f"    {name_prefix}_phase range: [{np.min(p_phase):.3f}, {np.max(p_phase):.3f}] rad")
        
        return out_file
    
    # Export each case
    export_field_vtu(standing, "standing_p", output_dir / "fields_standing")
    export_field_vtu(vortex, "vortex_p", output_dir / "fields_vortex")
    export_field_vtu(combined, "combined_p", output_dir / "fields_combined", include_ref=True)
    
    # Export delta with all components
    delta_file = export_field_vtu(delta, "delta_p", output_dir / "fields_delta", include_ref=True)
    
    # Add combined fields to delta for convenience
    grid = pv.read(str(delta_file))
    grid.point_data["combined_p_mag"] = np.abs(combined).astype(np.float64)
    grid.point_data["combined_p_phase"] = np.angle(combined).astype(np.float64)
    grid.save(str(delta_file))
    
    # Export mesh (P1 for geometry)
    with XDMFFile(domain.comm, str(output_dir / "mesh" / "domain.xdmf"), "w") as xf:
        xf.write_mesh(domain)
    if verbose:
        print(f"  Exported: mesh/domain.xdmf")
    
    # Save diagnostics
    diagnostics_file = output_dir / "meta" / "diagnostics.json"
    with open(diagnostics_file, "w") as f:
        json.dump(diagnostics, f, indent=2)
    if verbose:
        print(f"  Exported: meta/diagnostics.json")
    
    # Save configuration
    config_file = output_dir / "meta" / "config.json"
    cfg_dict = asdict(cfg)
    cfg_dict["omega"] = cfg.omega
    cfg_dict["k"] = cfg.k
    cfg_dict["wavelength"] = cfg.wavelength
    cfg_dict["Z_water"] = cfg.Z_water
    with open(config_file, "w") as f:
        json.dump(cfg_dict, f, indent=2)
    if verbose:
        print(f"  Exported: meta/config.json")
    
    # Create README
    readme = output_dir / "README.txt"
    readme_content = f"""
DEVICE-ALIGNED VORTEX SIMULATION
================================

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PHYSICAL CONFIGURATION:
-----------------------
- BOTTOM (z=0): Vortex lens transducer (azimuthal phase, ℓ={cfg.vortex_topological_charge})
- SIDE WALLS (x=0, x=L): Standing wave transducers (anti-phase)
- TOP (z=L): Air interface ({cfg.top_bc_type} BC)
- Y-walls: Passive ({cfg.passive_wall_bc})

DOMAIN:
-------
- Size: {cfg.dish_size_m*1e3:.1f} mm × {cfg.dish_size_m*1e3:.1f} mm × {cfg.dish_size_m*1e3:.1f} mm
- Frequency: {cfg.frequency_hz/1e3:.0f} kHz
- Wavelength: {cfg.wavelength*1e3:.3f} mm

FIELDS (in VTU format):
-----------------------
1) fields_standing/standing_p_fields.vtu
   - Standing wave only (x-wall actuation)
   
2) fields_vortex/vortex_p_fields.vtu
   - Vortex only (bottom actuation)
   
3) fields_combined/combined_p_fields.vtu
   - Combined (standing + vortex)
   
4) fields_delta/delta_p_fields.vtu ← MAIN VISUALIZATION
   - Difference: combined - standing
   - delta_p_mag: for iso-surfaces (Contour filter)
   - delta_p_phase: for coloring (use twilight cyclic colormap)

PARAVIEW WORKFLOW:
------------------
1) Open: fields_delta/delta_p_fields.vtu
2) Apply Contour filter on delta_p_mag
   - Start with value ~100 Pa (adjust to see structure)
3) Color the contour by delta_p_phase
   - Use twilight or HSV colormap
   - Ensure range is [-π, π]
4) Take z-slices to verify bulk structure:
   - Slice at z={cfg.dish_size_m*0.5*1e3:.1f} mm (mid-height)
   - Should see non-zero delta_p_mag in the BULK

DIAGNOSTICS:
------------
max|Δp| at bottom: {diagnostics['max_delta_p_bottom']:.1f} Pa
max|Δp| at mid:    {diagnostics['max_delta_p_mid']:.1f} Pa
max|Δp| at top:    {diagnostics['max_delta_p_top']:.1f} Pa
BULK RATIO:        {diagnostics['bulk_ratio_mid_over_bottom']:.4f}

SUCCESS CRITERIA:
-----------------
✓ Contour extends into volume (not just surface)
✓ Phase shows azimuthal winding (rainbow around core)
✓ Z-slices show structure at mid-height
✓ Bulk ratio > 0.1 (vortex propagates into volume)
"""
    
    with open(readme, "w") as f:
        f.write(readme_content)
    
    if verbose:
        print(f"  Exported: README.txt")
        print(f"{'='*70}\n")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Device-aligned Helmholtz solver with ParaView export"
    )
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: Windows OneDrive)")
    parser.add_argument("--vortex_gain", type=float, default=10.0,
                        help="Vortex velocity amplitude multiplier")
    parser.add_argument("--standing_gain", type=float, default=1.0,
                        help="Standing wave velocity amplitude multiplier")
    parser.add_argument("--top_impedance_factor", type=float, default=0.01,
                        help="Top BC impedance factor (lower = more open)")
    parser.add_argument("--aperture_radius_mm", type=float, default=4.0,
                        help="Vortex aperture radius in mm")
    args = parser.parse_args()
    
    # Configure
    cfg = DeviceConfig(
        vortex_velocity_amplitude=1e-6 * args.vortex_gain,
        standing_velocity_amplitude=1e-6 * args.standing_gain,
        top_impedance_factor=args.top_impedance_factor,
        vortex_aperture_radius=args.aperture_radius_mm * 1e-3,
    )
    
    # Output directory
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        base_dir = Path(args.output_dir)
    else:
        # Default: Windows OneDrive
        base_dir = Path("/mnt/c/Users/zachn/OneDrive - University of Bristol/"
                       "Major Project Onedrive/Research/Vortex 3D visualisation")
    
    output_dir = base_dir / f"vortex_deviceAligned_{ts}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'#'*70}")
    print("# DEVICE-ALIGNED HELMHOLTZ SOLVER")
    print(f"# Output: {output_dir}")
    print(f"{'#'*70}\n")
    
    # Create mesh
    domain, facet_tags, tag_map = create_mesh_with_tags(cfg, verbose=True)
    
    # Verify boundary tags
    verify_boundary_tags(domain, facet_tags, cfg, verbose=True)
    
    # Print boundary tag summary
    print("BOUNDARY TAG MAPPING:")
    for tag, desc in tag_map.items():
        print(f"  {tag}: {desc}")
    print()
    
    # Solve all cases
    print(f"\n{'='*70}")
    print("SOLVING HELMHOLTZ EQUATION (3 cases)")
    print(f"{'='*70}\n")
    
    p_standing = solve_helmholtz(domain, facet_tags, cfg, mode="standing", verbose=True)
    p_vortex = solve_helmholtz(domain, facet_tags, cfg, mode="vortex", verbose=True)
    p_combined = solve_helmholtz(domain, facet_tags, cfg, mode="combined", verbose=True)
    
    # Compute diagnostics
    diagnostics = compute_diagnostics(p_standing, p_vortex, p_combined, cfg, verbose=True)
    
    # Export to ParaView
    export_to_paraview(
        domain, facet_tags,
        p_standing, p_vortex, p_combined,
        cfg, diagnostics, output_dir,
        verbose=True
    )
    
    # Final summary
    print(f"\n{'#'*70}")
    print("# EXPORT COMPLETE")
    print(f"{'#'*70}")
    print(f"\nOutput location (Windows):")
    print(f"  {str(output_dir).replace('/mnt/c/', 'C:/')}")
    print(f"\nTo visualize in ParaView:")
    print(f"  1) Open: fields_delta/delta_p_fields.vtu")
    print(f"  2) Contour on delta_p_mag")
    print(f"  3) Color by delta_p_phase (twilight colormap)")
    print(f"\nBulk ratio: {diagnostics['bulk_ratio_mid_over_bottom']:.4f}")
    
    if diagnostics['bulk_ratio_mid_over_bottom'] < 1e-3:
        print("\n⚠️ WARNING: Vortex not propagating into bulk!")
        print("   Consider: increase aperture, reduce top impedance, or check BC signs")
    
    print()


if __name__ == "__main__":
    main()
