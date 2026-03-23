#!/usr/bin/env python3
"""
FEM Standing-Wave Mesh Convergence Study Driver
================================================

Runs the FEM standing-wave solver at a specified EPL and physical domain size,
extracts all quantities needed for mesh convergence analysis, and saves a
self-contained .npz output plus a running convergence_summary.csv.

Usage
-----
    # Run EPL=5 first (this becomes the reference solution)
    python scripts/dev/fem_convergence_study.py --epl 5 --convergence-mode

    # Then run coarser meshes (error metrics computed automatically)
    for epl in 2 3 3.5 4 4.5; do
        python scripts/dev/fem_convergence_study.py --epl $epl --convergence-mode
    done

    # Full production domain at validated resolution
    python scripts/dev/fem_convergence_study.py --epl 5 --production-mode

    # MUMPS out-of-core for RAM-limited machines
    python scripts/dev/fem_convergence_study.py --epl 5 --convergence-mode \\
        --mumps-out-of-core --mumps-mem-mb 20000

    # Domain-size sensitivity (override default physical size)
    python scripts/dev/fem_convergence_study.py --epl 5 --convergence-mode \\
        --physical-size-mm 4.0

    # PML sensitivity (override default PML thickness)
    python scripts/dev/fem_convergence_study.py --epl 5 --convergence-mode \\
        --pml-n-wavelengths-xy 2.0

Key design decisions
--------------------
  DIRECT FEM EVALUATION — no IDW, no RBF, no scattered-DOF interpolation.
  At each Cartesian grid point (xi, yj, z*):
      locate containing cell via bb_tree / compute_colliding_cells
      call p_function.eval(point, cell)
  This is the only defensible approach for a P2 FEM solution.

  Gor'kov potential computed from finite differences on the sampled Cartesian
  grid. z-gradient neglected (valid at pressure antinode where dp/dz ~ 0).

  Reference solution: EPL=5 on same domain/mode. Auto-located by filename
  pattern. Run EPL=5 before coarser meshes to enable error metrics.

  All wavelength-derived quantities (PML thickness, z_star, grid bounds)
  are computed from cfg AFTER instantiation — never from top-level constants.

Outputs (results/mesh_convergence_study/)
-----------------------------------------
  conv_epl{N}_{mode}_phys{S}mm_{ts}/
      figures/
          pmag_gorkov_epl{N}_{mode}.png
          centreline_epl{N}_{mode}.png
      [XDMF/HDF5 checkpoint from solve_helmholtz export_fields]
  conv_epl{N}_{mode}_phys{S}mm_{ts}.npz   ← self-contained data file
  convergence_summary.csv                  ← one row per run, appended

Author: Acousto-Tweezers Project
Date:   March 2026
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage

# ── project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from acoustweezers.experiments.farfield_petri_cuboid.config import FarFieldConfig
from acoustweezers.experiments.farfield_petri_cuboid.presets import (
    CORRECTED_PRESET, PETSC_MUMPS,
)
from acoustweezers.experiments.farfield_petri_cuboid.solve_pressure import (
    solve_helmholtz,
)

# ── output root ───────────────────────────────────────────────────────────────
STUDY_DIR = PROJECT_ROOT / "results" / "mesh_convergence_study"

# ── particle properties for Gor'kov (polystyrene, mid-range 45–150 µm) ───────
#    Radius: midpoint of experimental range
#    All other quantities match water properties in FarFieldConfig defaults
PARTICLE_RADIUS_M  = 97.5e-6   # m     — midpoint of 45–150 µm
PARTICLE_DENSITY   = 1050.0    # kg/m³ — polystyrene
PARTICLE_C         = 2350.0    # m/s   — polystyrene longitudinal


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FEM standing-wave mesh convergence study driver",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── run mode ──────────────────────────────────────────────────────────────
    mode_grp = p.add_mutually_exclusive_group()
    mode_grp.add_argument(
        "--convergence-mode", action="store_true", default=False,
        help="Reduced domain for mesh convergence study")
    mode_grp.add_argument(
        "--production-mode", action="store_true", default=False,
        help="Full production domain (5.4 mm physical interior)")

    # ── mesh resolution ───────────────────────────────────────────────────────
    p.add_argument(
        "--epl", type=float, default=5.0,
        help="Elements per wavelength — accepts floats e.g. 3.5")

    # ── domain geometry ───────────────────────────────────────────────────────
    p.add_argument(
        "--physical-size-mm", type=float, default=None,
        help=("Physical dish interior [mm, square]. "
              "Defaults: 3.0 mm (convergence) / 5.4 mm (production). "
              "Override for domain-sensitivity study."))

    # ── Cartesian evaluation grid ─────────────────────────────────────────────
    p.add_argument(
        "--grid-n", type=int, default=None,
        help="Grid points per axis for Cartesian evaluation plane. "
             "Defaults: 300 (convergence) / 500 (production).")

    # ── MUMPS memory controls ─────────────────────────────────────────────────
    p.add_argument(
        "--mumps-out-of-core", action="store_true", default=False,
        help="Enable MUMPS out-of-core factorisation (icntl_22 = 1)")
    p.add_argument(
        "--mumps-mem-mb", type=int, default=None,
        help="MUMPS working memory cap per process [MB] (icntl_23). "
             "0 or omitted = unlimited.")

    # ── PML controls ──────────────────────────────────────────────────────────
    p.add_argument(
        "--pml-n-wavelengths-xy", type=float, default=None,
        help="Lateral PML thickness in wavelengths (default: from preset, 1.0)")
    p.add_argument(
        "--pml-n-wavelengths-z", type=float, default=None,
        help="Bottom PML thickness in wavelengths (default: from preset, 1.0)")
    p.add_argument(
        "--pml-sigma-max-factor", type=float, default=None,
        help="PML sigma_max = factor * omega (default: from preset, 5.0)")

    # ── reference solution ────────────────────────────────────────────────────
    p.add_argument(
        "--reference-npz", type=str, default=None,
        help="Explicit path to reference .npz for error metrics. "
             "If omitted, auto-searches for EPL=5 run with matching mode "
             "and physical size.")

    # ── misc ──────────────────────────────────────────────────────────────────
    p.add_argument(
        "--timestamp",
        default=datetime.now().strftime("%Y%m%d_%H%M%S"),
        help="Timestamp string for output directory naming")
    p.add_argument(
        "--no-verify", action="store_true", default=False,
        help="Skip XDMF reload verification (saves time in batch runs)")

    return p.parse_args()


# =============================================================================
# Config builder
# =============================================================================

def build_config(args: argparse.Namespace) -> tuple[FarFieldConfig, str, float]:
    """
    Build FarFieldConfig from CLI args.

    Returns
    -------
    cfg : FarFieldConfig
    mode_tag : str   — 'convergence' or 'production'
    phys_size_mm : float — physical interior size used
    """
    # ── mode tag ──────────────────────────────────────────────────────────────
    if args.production_mode:
        mode_tag = "production"
        default_phys_mm = 5.4
        default_grid_n  = 500
    else:
        mode_tag = "convergence"
        default_phys_mm = 3.0
        default_grid_n  = 300

    phys_mm = args.physical_size_mm if args.physical_size_mm is not None else default_phys_mm

    # ── compute Lx from physical interior + 2×PML ──────────────────────────
    # Build a minimal config just to get wavelength at 2.15 MHz
    _tmp = FarFieldConfig(frequency_hz=2.15e6, c=1484.0)
    lam   = _tmp.wavelength            # 1484 / 2.15e6  ≈ 6.902e-4 m

    pml_xy_nw = args.pml_n_wavelengths_xy if args.pml_n_wavelengths_xy is not None else 1.0
    pml_z_nw  = args.pml_n_wavelengths_z  if args.pml_n_wavelengths_z  is not None else 1.0
    pml_sigma = args.pml_sigma_max_factor if args.pml_sigma_max_factor is not None else 5.0

    t_pml = pml_xy_nw * lam             # PML each side
    Lx    = (phys_mm * 1e-3) + 2.0 * t_pml
    Ly    = Lx

    overrides = {
        **CORRECTED_PRESET,            # inherit BCs, solver, lens settings
        # updated frequency & geometry
        "frequency_hz"               : 2.15e6,
        "Lx"                         : Lx,
        "Ly"                         : Ly,
        "H_under"                    : 5e-3,   # water-bath depth — unchanged
        "H_top"                      : 2e-3,   # Petri water depth — unchanged
        # standing wave only
        "disk_velocity_amplitude"    : 0.0,
        # mesh resolution from CLI
        "elements_per_wavelength"    : args.epl,
        # PML — from CLI or defaults
        "pml_n_wavelengths_xy"       : pml_xy_nw,
        "pml_n_wavelengths_z"        : pml_z_nw,
        "pml_sigma_max_factor"       : pml_sigma,
    }

    cfg = FarFieldConfig(**overrides)
    return cfg, mode_tag, phys_mm


# =============================================================================
# PETSc option builder
# =============================================================================

def build_petsc_options(args: argparse.Namespace) -> dict:
    """
    Extend the base MUMPS preset with any out-of-core / memory options.
    """
    opts = dict(PETSC_MUMPS)   # shallow copy

    if args.mumps_out_of_core:
        # icntl_22 = 1 → MUMPS out-of-core factorisation
        opts["mat_mumps_icntl_22"] = "1"

    if args.mumps_mem_mb is not None:
        # icntl_23 → per-process memory cap in MB (0 = unlimited)
        opts["mat_mumps_icntl_23"] = str(args.mumps_mem_mb)

    return opts


# =============================================================================
# Direct FEM evaluation on Cartesian plane
# =============================================================================

def eval_fem_on_cartesian_plane(
    p_function,
    domain,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    z_star: float,
    verbose: bool = True,
) -> np.ndarray:
    """
    Evaluate the FEM pressure solution at every point of a fixed 2D Cartesian
    grid at z = z_star by DIRECT cell-based evaluation.

    Method
    ------
    For each (xi, yj, z*):
        1. bb_tree locates the containing mesh cell
        2. compute_colliding_cells confirms cell ownership
        3. p_function.eval(point, cell) returns the P2-interpolated value

    This is the only approach that samples the true FEM field without
    introducing a second uncontrolled numerical approximation.
    Points outside the mesh (PML region or above domain) are set to NaN.

    Parameters
    ----------
    p_function : dolfinx.fem.Function  — complex P2 pressure solution
    domain     : dolfinx.mesh.Mesh
    x_grid     : 1D float array [m]
    y_grid     : 1D float array [m]
    z_star     : float [m] — evaluation plane

    Returns
    -------
    p_cart : complex ndarray, shape (len(y_grid), len(x_grid))
    """
    from dolfinx.geometry import (
        bb_tree,
        compute_collisions_points,
        compute_colliding_cells,
    )

    Nx = len(x_grid)
    Ny = len(y_grid)
    xx, yy = np.meshgrid(x_grid, y_grid)   # (Ny, Nx)

    pts_flat = np.column_stack([
        xx.ravel(),
        yy.ravel(),
        np.full(xx.size, z_star, dtype=np.float64),
    ])   # (Ny*Nx, 3)

    if verbose:
        print(f"    Building bb_tree … ", end="", flush=True)

    t0 = time.time()
    tree     = bb_tree(domain, domain.topology.dim)
    cands    = compute_collisions_points(tree, pts_flat)
    cell_map = compute_colliding_cells(domain, cands, pts_flat)

    if verbose:
        print(f"{time.time()-t0:.1f}s")
        print(f"    Evaluating {len(pts_flat):,} points … ", end="", flush=True)

    t1 = time.time()
    p_flat  = np.full(len(pts_flat), np.nan, dtype=complex)
    n_found = 0

    for i in range(len(pts_flat)):
        links = cell_map.links(i)
        if len(links) > 0:
            val = p_function.eval(pts_flat[i], links[0])
            p_flat[i] = complex(val[0])
            n_found += 1

    p_cart = p_flat.reshape(Ny, Nx)
    n_nan  = int(np.sum(np.isnan(np.abs(p_cart))))

    if verbose:
        print(f"{time.time()-t1:.1f}s")
        print(f"    {n_found:,} points found, {n_nan:,} outside mesh (NaN)")
        print(f"    max|p| on grid: {np.nanmax(np.abs(p_cart)):.4f} Pa")

    return p_cart


# =============================================================================
# Gor'kov potential
# =============================================================================

def compute_gorkov_2d(
    p_cart: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    cfg: FarFieldConfig,
    particle_radius: float  = PARTICLE_RADIUS_M,
    particle_density: float = PARTICLE_DENSITY,
    particle_c: float       = PARTICLE_C,
) -> np.ndarray:
    """
    Compute 2D Gor'kov radiation potential on the Cartesian grid.

    Formulation
    -----------
    U = (4π a³ / 3) [ f1 / (2 ρ₀ c₀²) <p²>  −  3 f2 ρ₀ / 4 <v²> ]

    where
        f1 = 1 − κ_p / κ_f            (compressibility contrast)
        f2 = 2(ρ_p − ρ_f)/(2ρ_p + ρ_f) (density contrast)
        <p²> = |P|² / 2                (cycle-averaged pressure squared)
        <v²> = |∇P|² / (2 ρ₀² ω²)    (cycle-averaged velocity squared)

    Gradient approximation
    ----------------------
    ∇P is computed by finite differences on the fixed Cartesian grid
    (np.gradient, second-order central differences at interior points).
    The z-gradient is neglected — valid at the pressure antinode (z_star)
    where ∂p/∂z ≈ 0 by symmetry. This is stated explicitly in the report.

    Particle parameters (polystyrene, midpoint of 45–150 µm range):
        a   = 97.5 µm,  ρ_p = 1050 kg/m³,  c_p = 2350 m/s

    Local minima of U are stable acoustic trap locations.

    Returns
    -------
    U : float ndarray, shape (Ny, Nx) [J]
    """
    rho_f = cfg.rho          # fluid density
    c_f   = cfg.c            # fluid speed of sound
    omega = cfg.omega

    # Contrast factors
    kappa_f = 1.0 / (rho_f * c_f**2)
    kappa_p = 1.0 / (particle_density * particle_c**2)
    f1 = 1.0 - kappa_p / kappa_f
    f2 = 2.0 * (particle_density - rho_f) / (2.0 * particle_density + rho_f)

    # Time-averaged pressure squared: <p²> = |P|² / 2
    p2_avg = 0.5 * np.abs(p_cart)**2

    # Finite-difference gradient on the Cartesian grid
    dx = float(x_grid[1] - x_grid[0])
    dy = float(y_grid[1] - y_grid[0])
    dp_dx = np.gradient(p_cart, dx, axis=1)   # ∂p/∂x
    dp_dy = np.gradient(p_cart, dy, axis=0)   # ∂p/∂y
    # z-gradient neglected at pressure antinode

    # Time-averaged velocity squared: <v²> = |∇P|² / (2 ρ₀² ω²)
    grad_p_sq = np.abs(dp_dx)**2 + np.abs(dp_dy)**2
    v2_avg    = grad_p_sq / (2.0 * rho_f**2 * omega**2)

    # Gor'kov potential
    prefactor = (4.0 * np.pi * particle_radius**3) / 3.0
    U = prefactor * (
          (f1 / (2.0 * rho_f * c_f**2)) * p2_avg
        - (3.0 * f2 * rho_f / 4.0)      * v2_avg
    )

    return U.astype(np.float64)


# =============================================================================
# Trap detection
# =============================================================================

def detect_traps(
    U: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    roi_fraction: float = 0.5,
    min_depth_fraction: float = 0.05,
    min_separation_m: float = None,
    wavelength: float = None,
) -> dict:
    """
    Detect local minima of the Gor'kov potential in the central ROI.

    Improvements over naïve detection:
      1. Prominence / depth threshold — reject shallow minima that are likely
         grid noise. A minimum must be at least `min_depth_fraction` of the
         full ROI U-range below the ROI median.
      2. Minimum separation — reject duplicate detections closer than
         `min_separation_m` (default: λ/4 if wavelength provided).
      3. ROI is defined as the central roi_fraction × roi_fraction of the
         physical domain (default: central 50%).

    Returns
    -------
    dict with keys:
        trap_positions_m          : (N, 2) float — (x, y) of each trap [m]
        trap_depths               : (N,)   float — U value at each trap [J]
        n_traps                   : int
        centreline_min_x          : 1D float — x of centreline minima [m]
        centreline_min_spacing_m  : float — mean inter-trap spacing [m]
        first_centreline_min_x_m  : float — x of first centreline minimum [m]
        roi_mask                  : (Ny, Nx) bool
    """
    Ny, Nx = U.shape
    x_lo, x_hi = x_grid[0], x_grid[-1]
    y_lo, y_hi = y_grid[0], y_grid[-1]
    xc = 0.5 * (x_lo + x_hi)
    yc = 0.5 * (y_lo + y_hi)
    half_x = 0.5 * roi_fraction * (x_hi - x_lo)
    half_y = 0.5 * roi_fraction * (y_hi - y_lo)

    if min_separation_m is None and wavelength is not None:
        min_separation_m = wavelength * 0.25  # λ/4

    xx, yy = np.meshgrid(x_grid, y_grid)
    roi_mask = (
        (np.abs(xx - xc) <= half_x) &
        (np.abs(yy - yc) <= half_y)
    )

    # Restrict to ROI; set exterior to +inf so they don't appear as minima
    U_roi = np.where(roi_mask & ~np.isnan(U), U, np.inf)

    # Local minima: pixel equals minimum in 5×5 neighbourhood
    neighbourhood = np.ones((5, 5), dtype=bool)
    local_min_mask = (
        (U_roi == ndimage.minimum_filter(U_roi, footprint=neighbourhood))
        & roi_mask
        & ~np.isnan(U)
        & np.isfinite(U_roi)
    )

    min_idx   = np.argwhere(local_min_mask)   # (N, 2) — [row, col]
    trap_x    = x_grid[min_idx[:, 1]]
    trap_y    = y_grid[min_idx[:, 0]]
    trap_U    = U[min_idx[:, 0], min_idx[:, 1]]

    # ── Depth threshold: reject shallow minima ────────────────────────────────
    U_roi_valid = U[roi_mask & ~np.isnan(U)]
    if len(U_roi_valid) > 0 and len(trap_U) > 0:
        U_median = np.median(U_roi_valid)
        U_range  = np.ptp(U_roi_valid)
        depth_threshold = U_median - min_depth_fraction * U_range
        keep = trap_U < depth_threshold
        trap_x = trap_x[keep]
        trap_y = trap_y[keep]
        trap_U = trap_U[keep]

    # Sort deepest first (most negative U)
    order  = np.argsort(trap_U)
    trap_x = trap_x[order]
    trap_y = trap_y[order]
    trap_U = trap_U[order]

    # ── Minimum separation: suppress duplicates ──────────────────────────────
    if min_separation_m is not None and min_separation_m > 0 and len(trap_x) > 1:
        keep_idx = [0]  # always keep deepest
        for i in range(1, len(trap_x)):
            dists = np.sqrt(
                (trap_x[i] - trap_x[keep_idx])**2 +
                (trap_y[i] - trap_y[keep_idx])**2
            )
            if np.all(dists >= min_separation_m):
                keep_idx.append(i)
        trap_x = trap_x[keep_idx]
        trap_y = trap_y[keep_idx]
        trap_U = trap_U[keep_idx]

    # ── centreline analysis: slice at row closest to y_centre ─────────────────
    j_centre = int(np.argmin(np.abs(y_grid - yc)))
    U_cl     = U[j_centre, :]

    # Centreline depth threshold
    cl_valid = U_cl[np.isfinite(U_cl)]
    if len(cl_valid) > 0:
        cl_median = np.median(cl_valid)
        cl_range  = np.ptp(cl_valid)
        cl_depth_thresh = cl_median - min_depth_fraction * cl_range
    else:
        cl_depth_thresh = np.inf

    # 1D local minima on centreline
    cl_local_min = (
        U_cl == ndimage.minimum_filter1d(U_cl, size=5)
    )
    cl_x_all = x_grid[cl_local_min & np.isfinite(U_cl)]

    # Filter to ROI x-range
    cl_x = cl_x_all[(cl_x_all >= xc - half_x) & (cl_x_all <= xc + half_x)]

    # Apply depth filter to centreline minima
    if len(cl_x) > 0:
        cl_U_vals = np.array([U_cl[np.argmin(np.abs(x_grid - cx))] for cx in cl_x])
        cl_x = cl_x[cl_U_vals < cl_depth_thresh]

    cl_x = np.sort(cl_x)   # left to right

    spacing     = float(np.mean(np.diff(cl_x)))  if len(cl_x) > 1 else np.nan
    first_min_x = float(cl_x[0])                 if len(cl_x) > 0 else np.nan

    return {
        "trap_positions_m"         : np.column_stack([trap_x, trap_y])
                                     if len(trap_x) > 0 else np.zeros((0, 2)),
        "trap_depths"              : trap_U,
        "n_traps"                  : len(trap_x),
        "centreline_min_x"         : cl_x,
        "centreline_min_spacing_m" : spacing,
        "first_centreline_min_x_m" : first_min_x,
        "roi_mask"                 : roi_mask,
    }


# =============================================================================
# Trap matching to reference
# =============================================================================

def match_traps_to_reference(
    current_traps: np.ndarray,
    reference_traps: np.ndarray,
    max_match_distance_m: float,
) -> dict:
    """
    Match detected traps to reference traps using nearest-neighbour assignment.

    Each current trap is matched to the nearest unmatched reference trap.
    Matches with distance > max_match_distance_m are rejected.
    This is a greedy assignment sorted by distance (sufficient when traps
    are well-separated relative to the matching threshold, which they are
    at λ/2 spacing with a λ/4 threshold).

    Parameters
    ----------
    current_traps   : (M, 2) float — (x, y) positions of current traps [m]
    reference_traps : (N, 2) float — (x, y) positions of reference traps [m]
    max_match_distance_m : float — reject matches beyond this distance [m]

    Returns
    -------
    dict with keys:
        n_matched             : int
        n_unmatched_current   : int — current traps with no reference match
        n_unmatched_reference : int — reference traps with no current match
        matched_errors_m      : (n_matched,) float — position error per match
        mean_trap_error_m     : float — mean matched position error [m]
        max_trap_error_m      : float — max matched position error [m]
        matched_pairs         : (n_matched, 2, 2) — (current_pos, ref_pos) pairs
    """
    M = len(current_traps)
    N = len(reference_traps)

    if M == 0 or N == 0:
        return {
            "n_matched"             : 0,
            "n_unmatched_current"   : M,
            "n_unmatched_reference" : N,
            "matched_errors_m"      : np.array([]),
            "mean_trap_error_m"     : np.nan,
            "max_trap_error_m"      : np.nan,
            "matched_pairs"         : np.zeros((0, 2, 2)),
        }

    # Build distance matrix: (M, N)
    dx = current_traps[:, 0:1] - reference_traps[:, 0:1].T   # (M, N)
    dy = current_traps[:, 1:2] - reference_traps[:, 1:2].T
    dist = np.sqrt(dx**2 + dy**2)

    # Greedy nearest-neighbour assignment
    matched_cur = []
    matched_ref = []
    matched_err = []

    used_ref = set()
    # Create a flat list of (distance, cur_idx, ref_idx) sorted by distance
    candidates = []
    for i in range(M):
        for j in range(N):
            if dist[i, j] <= max_match_distance_m:
                candidates.append((dist[i, j], i, j))
    candidates.sort()

    used_cur = set()
    for d, ci, rj in candidates:
        if ci in used_cur or rj in used_ref:
            continue
        matched_cur.append(ci)
        matched_ref.append(rj)
        matched_err.append(d)
        used_cur.add(ci)
        used_ref.add(rj)

    matched_err = np.array(matched_err) if matched_err else np.array([])
    n_matched = len(matched_err)

    if n_matched > 0:
        pairs = np.stack([
            current_traps[matched_cur],
            reference_traps[matched_ref],
        ], axis=1)
    else:
        pairs = np.zeros((0, 2, 2))

    return {
        "n_matched"             : n_matched,
        "n_unmatched_current"   : M - n_matched,
        "n_unmatched_reference" : N - n_matched,
        "matched_errors_m"      : matched_err,
        "mean_trap_error_m"     : float(np.mean(matched_err)) if n_matched > 0 else np.nan,
        "max_trap_error_m"      : float(np.max(matched_err))  if n_matched > 0 else np.nan,
        "matched_pairs"         : pairs,
    }


# =============================================================================
# Error metrics vs reference
# =============================================================================

def compute_error_metrics(
    p_cart: np.ndarray,
    U_cart: np.ndarray,
    trap_info: dict,
    ref_data: dict,
    roi_mask: np.ndarray,
    wavelength: float = None,
) -> dict:
    """
    Compute convergence error metrics against the EPL=5 reference solution.

    Metrics (primary)
    -----------------
    1. ε_L2_full  — relative L2 pressure error over full physical plane
    2. ε_L2_ROI   — relative L2 pressure error over central 50% ROI
    3. ε_gorkov   — relative L2 Gor'kov error over central 50% ROI

    Metrics (secondary — trap-based)
    --------------------------------
    4. spacing_err_pct  — error in mean centreline trap spacing as %
    5. n_matched_traps  — number of 2D traps successfully matched to reference
    6. n_unmatched_traps — current traps that found no reference partner
    7. mean_trap_error_m — mean position error of matched traps [m]
    8. max_trap_error_m  — max position error of matched traps [m]

    The old first-centreline-minimum-x metric is retained for backward
    compatibility but demoted: it is unreliable when coarse meshes fail
    to resolve edge traps, causing the minimum ordering to shift.

    Phase alignment
    ---------------
    Before computing field errors, the current solution is phase-aligned to
    the reference by normalising both fields by the complex value at the
    domain centre point.
    """
    p_ref = (ref_data["p_cart_real"] + 1j * ref_data["p_cart_imag"]).astype(complex)
    U_ref = ref_data["gorkov_2d"].astype(float)

    if p_cart.shape != p_ref.shape:
        raise ValueError(
            f"Grid shape mismatch: current {p_cart.shape} vs reference {p_ref.shape}. "
            "Ensure both runs use the same --grid-n and --physical-size-mm.")

    # Valid (non-NaN) mask
    valid = ~(np.isnan(np.abs(p_cart)) | np.isnan(np.abs(p_ref)))

    # ── Phase alignment ───────────────────────────────────────────────────────
    Ny, Nx = p_cart.shape
    ic, jc = Nx // 2, Ny // 2
    ref_val_cur = p_cart[jc, ic]
    ref_val_sol = p_ref[jc, ic]
    if np.abs(ref_val_cur) > 1e-30 and np.abs(ref_val_sol) > 1e-30:
        phase_diff = np.angle(ref_val_cur) - np.angle(ref_val_sol)
        p_aligned  = p_cart * np.exp(-1j * phase_diff)
    else:
        p_aligned = p_cart

    # ── 1. ε L2 full plane ───────────────────────────────────────────────────
    diff_full = np.abs(p_aligned - p_ref)[valid]
    norm_full = np.abs(p_ref)[valid]
    eps_full  = float(np.linalg.norm(diff_full) / (np.linalg.norm(norm_full) + 1e-30))

    # ── 2. ε L2 ROI ──────────────────────────────────────────────────────────
    roi_valid = valid & roi_mask
    diff_roi  = np.abs(p_aligned - p_ref)[roi_valid]
    norm_roi  = np.abs(p_ref)[roi_valid]
    eps_roi   = float(np.linalg.norm(diff_roi) / (np.linalg.norm(norm_roi) + 1e-30))

    # ── 3. Gor'kov ε ROI ─────────────────────────────────────────────────────
    gorkov_valid = roi_valid & ~np.isnan(U_cart) & ~np.isnan(U_ref)
    if np.sum(gorkov_valid) > 0:
        dU   = np.abs(U_cart[gorkov_valid] - U_ref[gorkov_valid])
        nU   = np.abs(U_ref[gorkov_valid])
        eps_gorkov = float(np.linalg.norm(dU) / (np.linalg.norm(nU) + 1e-30))
    else:
        eps_gorkov = np.nan

    # ── 4. Trap spacing error (centreline) ───────────────────────────────────
    sp_cur = trap_info["centreline_min_spacing_m"]
    sp_ref_arr = ref_data.get("centreline_min_spacing_m", np.array([np.nan]))
    sp_ref = float(np.squeeze(sp_ref_arr))
    spacing_err_pct = (abs(sp_cur - sp_ref) / (abs(sp_ref) + 1e-30) * 100.0) if (
        np.isfinite(sp_cur) and np.isfinite(sp_ref)) else np.nan

    # ── 5. Matched trap position error (2D) ──────────────────────────────────
    ref_traps = ref_data.get("trap_positions_m", np.zeros((0, 2)))
    if ref_traps.ndim == 1 and len(ref_traps) == 0:
        ref_traps = np.zeros((0, 2))
    cur_traps = trap_info["trap_positions_m"]

    max_match_dist = (wavelength * 0.25) if (wavelength and wavelength > 0) else 2e-4
    match_result = match_traps_to_reference(cur_traps, ref_traps, max_match_dist)

    # ── 6. Legacy first-centreline-min metric (kept for compatibility) ────────
    x_cur = trap_info["first_centreline_min_x_m"]
    x_ref_arr = ref_data.get("first_centreline_min_x_m", np.array([np.nan]))
    x_ref = float(np.squeeze(x_ref_arr))
    trap_pos_err_m = abs(x_cur - x_ref) if (
        np.isfinite(x_cur) and np.isfinite(x_ref)) else np.nan
    ref_lam = float(np.squeeze(ref_data.get("wavelength", np.array([np.nan]))))
    trap_pos_err_pct = (trap_pos_err_m / (0.5 * ref_lam) * 100.0) if (
        np.isfinite(trap_pos_err_m) and ref_lam > 0.0) else np.nan

    return {
        "eps_L2_full"               : eps_full,
        "eps_L2_roi"                : eps_roi,
        "eps_gorkov_roi"            : eps_gorkov,
        # Legacy (demoted)
        "trap_pos_err_m"            : trap_pos_err_m,
        "trap_pos_err_pct_halfwav"  : trap_pos_err_pct,
        # Robust trap metrics
        "trap_spacing_err_pct"      : spacing_err_pct,
        "n_matched_traps"           : match_result["n_matched"],
        "n_unmatched_current"       : match_result["n_unmatched_current"],
        "n_unmatched_reference"     : match_result["n_unmatched_reference"],
        "mean_trap_error_m"         : match_result["mean_trap_error_m"],
        "max_trap_error_m"          : match_result["max_trap_error_m"],
    }


# =============================================================================
# Reference solution locator
# =============================================================================

def find_reference_npz(
    study_dir: Path,
    mode_tag: str,
    phys_size_mm: float,
    ref_epl: float = 5.0,
) -> Optional[Path]:
    """
    Search study_dir for the most recent EPL=5 .npz matching mode and size.
    Filename pattern: conv_epl5.0_{mode}_phys{size}mm_{timestamp}.npz
    Returns Path or None if not found.
    """
    pattern = f"conv_epl{ref_epl:.1f}_{mode_tag}_phys{phys_size_mm:.1f}mm_*.npz"
    matches = sorted(study_dir.glob(pattern))
    return matches[-1] if matches else None


# =============================================================================
# Figures
# =============================================================================

def make_figures(
    p_cart: np.ndarray,
    U_cart: np.ndarray,
    trap_info: dict,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    z_star: float,
    cfg: FarFieldConfig,
    epl: float,
    mode_tag: str,
    fig_dir: Path,
) -> None:
    """
    Generate and save:
      1. Side-by-side |p| and Gor'kov potential 2D colourplots
      2. Centreline |p| and Gor'kov 1D slices
    All at 300 dpi, consistent axis labels in mm.
    """
    xmm = x_grid * 1e3
    ymm = y_grid * 1e3
    xc_mm = 0.5 * (xmm[0] + xmm[-1])
    yc_mm = 0.5 * (ymm[0] + ymm[-1])

    # ── Figure 1: 2D field plots ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    im = ax.pcolormesh(xmm, ymm, np.abs(p_cart),
                       cmap="inferno", shading="auto")
    cb = plt.colorbar(im, ax=ax)
    cb.set_label("|p| (Pa)", fontsize=10)
    ax.set_xlabel("x (mm)", fontsize=10)
    ax.set_ylabel("y (mm)", fontsize=10)
    ax.set_title(f"|p| at z* = {z_star*1e3:.3f} mm\n"
                 f"EPL = {epl},  {mode_tag}", fontsize=10)
    ax.set_aspect("equal")

    ax = axes[1]
    vmin_U = np.nanpercentile(U_cart, 2)
    vmax_U = np.nanpercentile(U_cart, 98)
    im2 = ax.pcolormesh(xmm, ymm, U_cart,
                        cmap="RdBu_r", shading="auto",
                        vmin=vmin_U, vmax=vmax_U)
    cb2 = plt.colorbar(im2, ax=ax)
    cb2.set_label("U (J)", fontsize=10)
    if trap_info["n_traps"] > 0:
        tx = trap_info["trap_positions_m"][:, 0] * 1e3
        ty = trap_info["trap_positions_m"][:, 1] * 1e3
        ax.scatter(tx, ty, c="yellow", s=18, zorder=5,
                   label=f"{trap_info['n_traps']} traps")
        ax.legend(fontsize=8)
    ax.set_xlabel("x (mm)", fontsize=10)
    ax.set_ylabel("y (mm)", fontsize=10)
    ax.set_title(f"Gor'kov U at z* = {z_star*1e3:.3f} mm\n"
                 f"EPL = {epl},  {mode_tag}", fontsize=10)
    ax.set_aspect("equal")

    fig.tight_layout()
    p1 = fig_dir / f"pmag_gorkov_epl{epl:.1f}_{mode_tag}.png"
    fig.savefig(p1, dpi=300)
    plt.close(fig)
    print(f"      {p1.name}")

    # ── Figure 2: centreline slices ───────────────────────────────────────────
    j_centre = int(np.argmin(np.abs(y_grid - 0.5 * (y_grid[0] + y_grid[-1]))))

    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 4))

    ax = axes2[0]
    ax.plot(xmm, np.abs(p_cart[j_centre, :]), lw=1.2, color="steelblue")
    ax.set_xlabel("x (mm)", fontsize=10)
    ax.set_ylabel("|p| (Pa)", fontsize=10)
    ax.set_title(f"Centreline |p|  (y = {y_grid[j_centre]*1e3:.2f} mm)\n"
                 f"EPL = {epl}", fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = axes2[1]
    ax.plot(xmm, U_cart[j_centre, :], lw=1.2, color="coral")
    for xm in trap_info["centreline_min_x"]:
        ax.axvline(xm * 1e3, color="navy", lw=0.8, alpha=0.6, ls="--")
    ax.set_xlabel("x (mm)", fontsize=10)
    ax.set_ylabel("U (J)", fontsize=10)
    ax.set_title(f"Centreline Gor'kov U\n"
                 f"EPL = {epl}  "
                 f"(spacing = {trap_info['centreline_min_spacing_m']*1e3:.3f} mm, "
                 f"λ/2 = {cfg.wavelength*0.5*1e3:.3f} mm)", fontsize=10)
    ax.grid(True, alpha=0.3)

    fig2.tight_layout()
    p2 = fig_dir / f"centreline_epl{epl:.1f}_{mode_tag}.png"
    fig2.savefig(p2, dpi=300)
    plt.close(fig2)
    print(f"      {p2.name}")


# =============================================================================
# Convergence CSV
# =============================================================================

def update_convergence_csv(csv_path: Path, row: dict) -> None:
    """Append one row to convergence_summary.csv, writing header if new file."""
    fieldnames = [
        "run_id", "timestamp", "mode", "physical_size_mm",
        "requested_epl",
        "mesh_nx", "mesh_ny", "mesh_nz", "dofs",
        "mesh_time_s", "solve_time_s", "total_wall_time_s",
        "mumps_out_of_core", "mumps_mem_mb",
        "max_p_Pa", "z_star_mm", "wavelength_mm",
        "n_traps_roi",
        "first_centreline_min_x_mm", "centreline_min_spacing_mm",
        "eps_L2_full", "eps_L2_roi", "eps_gorkov_roi",
        "trap_spacing_err_pct",
        "n_matched_traps", "n_unmatched_current", "n_unmatched_reference",
        "mean_trap_error_m", "max_trap_error_m",
        "trap_pos_err_m", "trap_pos_err_pct_halfwav",
        "pml_n_wavelengths_xy", "pml_n_wavelengths_z", "pml_sigma_max_factor",
        "npz_path",
    ]
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    args = parse_args()
    TS  = args.timestamp
    EPL = args.epl

    # ── default to convergence mode if neither flag given ─────────────────────
    if not args.production_mode and not args.convergence_mode:
        args.convergence_mode = True

    # ── build config ──────────────────────────────────────────────────────────
    cfg, mode_tag, phys_size_mm = build_config(args)

    # ── all geometry from cfg — never top-level constants ─────────────────────
    z_star     = cfg.H_under + cfg.H_top / 2.0 + 0.25 * cfg.wavelength
    phys_x_min = cfg.t_pml_xy
    phys_x_max = cfg.Lx - cfg.t_pml_xy
    phys_y_min = cfg.t_pml_xy
    phys_y_max = cfg.Ly - cfg.t_pml_xy

    # ── Cartesian evaluation grid ─────────────────────────────────────────────
    grid_n = args.grid_n if args.grid_n is not None else (
        300 if args.convergence_mode else 500)
    x_grid = np.linspace(phys_x_min, phys_x_max, grid_n)
    y_grid = np.linspace(phys_y_min, phys_y_max, grid_n)

    # ── output directories ────────────────────────────────────────────────────
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    run_id  = f"conv_epl{EPL:.1f}_{mode_tag}_phys{phys_size_mm:.1f}mm_{TS}"
    run_dir = STUDY_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    # ── header ────────────────────────────────────────────────────────────────
    print("=" * 72)
    print(f"MESH CONVERGENCE STUDY — EPL = {EPL}  |  mode = {mode_tag}")
    print("=" * 72)
    print(f"  Timestamp          : {TS}")
    print(f"  Run ID             : {run_id}")
    print(f"  Physical interior  : {phys_size_mm:.3f} × {phys_size_mm:.3f} mm")
    print(f"  Total box          : {cfg.Lx*1e3:.3f} mm  "
          f"(PML = {cfg.t_pml_xy*1e3:.3f} mm each side)")
    print(f"  Frequency          : {cfg.frequency_hz/1e6:.3f} MHz")
    print(f"  Wavelength         : {cfg.wavelength*1e3:.4f} mm")
    print(f"  k                  : {cfg.k:.2f} rad/m")
    print(f"  z*                 : {z_star*1e3:.4f} mm")
    print(f"  Mesh (target)      : {cfg.mesh_nx} × {cfg.mesh_ny} × {cfg.mesh_nz}")
    print(f"  Cartesian grid     : {grid_n} × {grid_n}")
    print(f"  MUMPS out-of-core  : {args.mumps_out_of_core}")
    if args.mumps_mem_mb:
        print(f"  MUMPS mem cap      : {args.mumps_mem_mb} MB")
    print(f"  Output dir         : {run_dir}")
    print()

    # ── PETSc options ─────────────────────────────────────────────────────────
    petsc_opts = build_petsc_options(args)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1 — FEM solve
    # ─────────────────────────────────────────────────────────────────────────
    print("─" * 72)
    print("STEP 1  FEM solve")
    print("─" * 72)

    t_wall0 = time.time()
    sol = solve_helmholtz(
        cfg,
        verbose=True,
        petsc_options=petsc_opts,
        export_fields=True,
        export_dir=str(run_dir),
    )
    t_wall_total = time.time() - t_wall0

    print(f"\n  DOFs          : {sol.dofs:,}")
    print(f"  Mesh time     : {sol.mesh_time:.1f}s")
    print(f"  Solve time    : {sol.solver_time:.1f}s")
    print(f"  Total wall    : {t_wall_total:.1f}s")
    print(f"  max|p|        : {sol.max_pressure:.4f} Pa")
    print(f"  KSP reason    : {sol.ksp_converged_reason}")
    print()

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2 — Direct FEM evaluation on Cartesian plane
    # ─────────────────────────────────────────────────────────────────────────
    print("─" * 72)
    print("STEP 2  Direct FEM evaluation on fixed Cartesian plane at z*")
    print("─" * 72)

    p_cart = eval_fem_on_cartesian_plane(
        sol.p_function, sol.domain,
        x_grid, y_grid, z_star,
        verbose=True,
    )
    print()

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3 — Gor'kov potential
    # ─────────────────────────────────────────────────────────────────────────
    print("─" * 72)
    print("STEP 3  Gor'kov potential (FD gradients on Cartesian grid)")
    print("─" * 72)

    U_cart = compute_gorkov_2d(p_cart, x_grid, y_grid, cfg)
    U_phys = np.where(~np.isnan(p_cart), U_cart, np.nan)
    print(f"  U range (physical) : "
          f"[{np.nanmin(U_phys):.3e}, {np.nanmax(U_phys):.3e}] J")
    print()

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 4 — Trap detection
    # ─────────────────────────────────────────────────────────────────────────
    print("─" * 72)
    print("STEP 4  Trap detection in central 50% ROI")
    print("─" * 72)

    trap_info = detect_traps(U_cart, x_grid, y_grid, roi_fraction=0.5,
                             wavelength=cfg.wavelength)

    expected_spacing_mm = cfg.wavelength * 0.5 * 1e3
    print(f"  Traps in ROI         : {trap_info['n_traps']}")
    print(f"  First centreline min : "
          f"{trap_info['first_centreline_min_x_m']*1e3:.4f} mm")
    print(f"  Mean trap spacing    : "
          f"{trap_info['centreline_min_spacing_m']*1e3:.4f} mm  "
          f"(expected λ/2 = {expected_spacing_mm:.4f} mm)")
    print()

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 5 — Error metrics vs reference
    # ─────────────────────────────────────────────────────────────────────────
    print("─" * 72)
    print("STEP 5  Error metrics vs EPL=5 reference")
    print("─" * 72)

    nan_metrics = {
        "eps_L2_full"               : np.nan,
        "eps_L2_roi"                : np.nan,
        "eps_gorkov_roi"            : np.nan,
        "trap_pos_err_m"            : np.nan,
        "trap_pos_err_pct_halfwav"  : np.nan,
        "trap_spacing_err_pct"      : np.nan,
        "n_matched_traps"           : 0,
        "n_unmatched_current"       : 0,
        "n_unmatched_reference"     : 0,
        "mean_trap_error_m"         : np.nan,
        "max_trap_error_m"          : np.nan,
    }
    error_metrics = dict(nan_metrics)

    ref_path = None
    if args.reference_npz:
        ref_path = Path(args.reference_npz)
        if not ref_path.exists():
            print(f"  WARNING: --reference-npz path not found: {ref_path}")
            ref_path = None

    if ref_path is None and EPL < 5.0:
        ref_path = find_reference_npz(STUDY_DIR, mode_tag, phys_size_mm, ref_epl=5.0)

    if ref_path is not None and ref_path.exists():
        print(f"  Reference  : {ref_path.name}")
        ref_data     = dict(np.load(ref_path, allow_pickle=True))
        error_metrics = compute_error_metrics(
            p_cart, U_cart, trap_info, ref_data, trap_info["roi_mask"],
            wavelength=cfg.wavelength)
        print(f"  ε L2 full  : {error_metrics['eps_L2_full']:.4e}")
        print(f"  ε L2 ROI   : {error_metrics['eps_L2_roi']:.4e}")
        print(f"  ε Gor'kov  : {error_metrics['eps_gorkov_roi']:.4e}")
        print(f"  spacing Δ  : {error_metrics['trap_spacing_err_pct']:.3f}%")
        print(f"  Matched traps      : {error_metrics['n_matched_traps']}")
        print(f"  Unmatched (cur/ref): {error_metrics['n_unmatched_current']} / "
              f"{error_metrics['n_unmatched_reference']}")
        if np.isfinite(error_metrics['mean_trap_error_m']):
            print(f"  Mean trap err      : {error_metrics['mean_trap_error_m']*1e6:.2f} µm")
            print(f"  Max  trap err      : {error_metrics['max_trap_error_m']*1e6:.2f} µm")
    elif EPL >= 5.0:
        print("  This run is the reference (EPL≥5) — error metrics not applicable.")
    else:
        print("  No EPL=5 reference found. Run EPL=5 first to enable error metrics.")
        print("  (Searching for: "
              f"conv_epl5.0_{mode_tag}_phys{phys_size_mm:.1f}mm_*.npz)")
    print()

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 6 — Save .npz
    # ─────────────────────────────────────────────────────────────────────────
    print("─" * 72)
    print("STEP 6  Save .npz")
    print("─" * 72)

    npz_path = STUDY_DIR / f"{run_id}.npz"

    meta = {
        "run_id"                   : run_id,
        "timestamp"                : TS,
        "mode"                     : mode_tag,
        "physical_size_mm"         : phys_size_mm,
        "requested_epl"            : EPL,
        "frequency_hz"             : cfg.frequency_hz,
        "wavelength_m"             : cfg.wavelength,
        "z_star_m"                 : z_star,
        "Lx"                       : cfg.Lx,
        "Ly"                       : cfg.Ly,
        "H_under"                  : cfg.H_under,
        "H_top"                    : cfg.H_top,
        "dofs"                     : sol.dofs,
        "mesh_nx"                  : cfg.mesh_nx,
        "mesh_ny"                  : cfg.mesh_ny,
        "mesh_nz"                  : cfg.mesh_nz,
        "mesh_time_s"              : sol.mesh_time,
        "solve_time_s"             : sol.solver_time,
        "total_wall_time_s"        : t_wall_total,
        "max_pressure_Pa"          : sol.max_pressure,
        "mumps_out_of_core"        : int(args.mumps_out_of_core),
        "mumps_mem_mb"             : args.mumps_mem_mb or 0,
        "grid_n"                   : grid_n,
        "particle_radius_m"        : PARTICLE_RADIUS_M,
        "pml_n_wavelengths_xy"     : cfg.pml_n_wavelengths_xy,
        "pml_n_wavelengths_z"      : cfg.pml_n_wavelengths_z,
        "pml_sigma_max_factor"     : cfg.pml_sigma_max_factor,
    }

    np.savez_compressed(
        npz_path,
        # ── Full DOF scatter (kept for reload verification) ────────────────
        coords                     = sol.coords,
        p_real                     = np.real(sol.p_values),
        p_imag                     = np.imag(sol.p_values),
        # ── Cartesian grid (direct FEM evaluation) ─────────────────────────
        p_cart_real                = np.real(p_cart).astype(np.float64),
        p_cart_imag                = np.imag(p_cart).astype(np.float64),
        x_grid                     = x_grid,
        y_grid                     = y_grid,
        z_star                     = np.array([z_star]),
        wavelength                 = np.array([cfg.wavelength]),
        # ── Gor'kov ────────────────────────────────────────────────────────
        gorkov_2d                  = U_cart,
        # ── Trap metrics ───────────────────────────────────────────────────
        trap_positions_m           = trap_info["trap_positions_m"],
        trap_depths                = trap_info["trap_depths"],
        centreline_min_x           = trap_info["centreline_min_x"],
        first_centreline_min_x_m   = np.array([trap_info["first_centreline_min_x_m"]]),
        centreline_min_spacing_m   = np.array([trap_info["centreline_min_spacing_m"]]),
        # ── Error metrics ──────────────────────────────────────────────────
        eps_L2_full                = np.array([error_metrics["eps_L2_full"]]),
        eps_L2_roi                 = np.array([error_metrics["eps_L2_roi"]]),
        eps_gorkov_roi             = np.array([error_metrics["eps_gorkov_roi"]]),
        trap_pos_err_m             = np.array([error_metrics["trap_pos_err_m"]]),
        trap_pos_err_pct_halfwav   = np.array([error_metrics["trap_pos_err_pct_halfwav"]]),
        trap_spacing_err_pct       = np.array([error_metrics["trap_spacing_err_pct"]]),
        n_matched_traps            = np.array([error_metrics["n_matched_traps"]]),
        n_unmatched_current        = np.array([error_metrics["n_unmatched_current"]]),
        n_unmatched_reference      = np.array([error_metrics["n_unmatched_reference"]]),
        mean_trap_error_m          = np.array([error_metrics["mean_trap_error_m"]]),
        max_trap_error_m           = np.array([error_metrics["max_trap_error_m"]]),
        # ── Metadata ───────────────────────────────────────────────────────
        metadata                   = meta,
    )
    print(f"  Saved: {npz_path}")
    print()

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 7 — Convergence CSV
    # ─────────────────────────────────────────────────────────────────────────
    print("─" * 72)
    print("STEP 7  Convergence summary CSV")
    print("─" * 72)

    def _fmt(v, fmt=".6e"):
        return format(v, fmt) if np.isfinite(v) else ""

    csv_path = STUDY_DIR / "convergence_summary.csv"
    csv_row = {
        "run_id"                   : run_id,
        "timestamp"                : TS,
        "mode"                     : mode_tag,
        "physical_size_mm"         : phys_size_mm,
        "requested_epl"            : EPL,
        "mesh_nx"                  : cfg.mesh_nx,
        "mesh_ny"                  : cfg.mesh_ny,
        "mesh_nz"                  : cfg.mesh_nz,
        "dofs"                     : sol.dofs,
        "mesh_time_s"              : round(sol.mesh_time, 2),
        "solve_time_s"             : round(sol.solver_time, 2),
        "total_wall_time_s"        : round(t_wall_total, 2),
        "mumps_out_of_core"        : int(args.mumps_out_of_core),
        "mumps_mem_mb"             : args.mumps_mem_mb or 0,
        "max_p_Pa"                 : round(sol.max_pressure, 4),
        "z_star_mm"                : round(z_star * 1e3, 4),
        "wavelength_mm"            : round(cfg.wavelength * 1e3, 4),
        "n_traps_roi"              : trap_info["n_traps"],
        "first_centreline_min_x_mm": _fmt(trap_info["first_centreline_min_x_m"] * 1e3, ".4f"),
        "centreline_min_spacing_mm": _fmt(trap_info["centreline_min_spacing_m"] * 1e3, ".4f"),
        "eps_L2_full"              : _fmt(error_metrics["eps_L2_full"]),
        "eps_L2_roi"               : _fmt(error_metrics["eps_L2_roi"]),
        "eps_gorkov_roi"           : _fmt(error_metrics["eps_gorkov_roi"]),
        "trap_spacing_err_pct"     : _fmt(error_metrics["trap_spacing_err_pct"], ".4f"),
        "n_matched_traps"          : error_metrics["n_matched_traps"],
        "n_unmatched_current"      : error_metrics["n_unmatched_current"],
        "n_unmatched_reference"    : error_metrics["n_unmatched_reference"],
        "mean_trap_error_m"        : _fmt(error_metrics["mean_trap_error_m"]),
        "max_trap_error_m"         : _fmt(error_metrics["max_trap_error_m"]),
        "trap_pos_err_m"           : _fmt(error_metrics["trap_pos_err_m"]),
        "trap_pos_err_pct_halfwav" : _fmt(error_metrics["trap_pos_err_pct_halfwav"], ".4f"),
        "pml_n_wavelengths_xy"     : cfg.pml_n_wavelengths_xy,
        "pml_n_wavelengths_z"      : cfg.pml_n_wavelengths_z,
        "pml_sigma_max_factor"     : cfg.pml_sigma_max_factor,
        "npz_path"                 : str(npz_path),
    }
    update_convergence_csv(csv_path, csv_row)
    print(f"  CSV: {csv_path}")
    print()

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 8 — Figures
    # ─────────────────────────────────────────────────────────────────────────
    print("─" * 72)
    print("STEP 8  Figures")
    print("─" * 72)

    make_figures(
        p_cart, U_cart, trap_info,
        x_grid, y_grid, z_star,
        cfg, EPL, mode_tag, fig_dir,
    )
    print()

    # ─────────────────────────────────────────────────────────────────────────
    # Final summary
    # ─────────────────────────────────────────────────────────────────────────
    print("=" * 72)
    print("COMPLETE")
    print("=" * 72)
    print(f"  Run ID        : {run_id}")
    print(f"  EPL           : {EPL}")
    print(f"  DOFs          : {sol.dofs:,}")
    print(f"  Solve time    : {sol.solver_time:.1f}s")
    print(f"  Total wall    : {t_wall_total:.1f}s")
    print(f"  max|p|        : {sol.max_pressure:.4f} Pa")
    print(f"  Traps (ROI)   : {trap_info['n_traps']}")
    if np.isfinite(error_metrics["eps_L2_roi"]):
        print(f"  ε L2 ROI      : {error_metrics['eps_L2_roi']:.4e}")
        print(f"  spacing err   : {error_metrics['trap_spacing_err_pct']:.3f}%")
        print(f"  matched traps : {error_metrics['n_matched_traps']}")
        if np.isfinite(error_metrics['mean_trap_error_m']):
            print(f"  mean trap err : {error_metrics['mean_trap_error_m']*1e6:.2f} µm")
    print(f"  .npz          : {npz_path.name}")
    print(f"  CSV           : {csv_path.name}")
    print()

    # Release FEM objects (not needed downstream)
    del sol
    gc.collect()
    return 0


if __name__ == "__main__":
    sys.exit(main())