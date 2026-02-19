#!/usr/bin/env python3
"""
Phase 1 — Transducer Size & Dish Size Architecture Sweep.

D1.1  Piezo diameter sweep: how does piezo diameter affect vortex authority,
      locality, and trap control?
D1.2  Dish footprint sweep: can the approach scale from 10×10 to 30×30 mm?

Configurations:
  L10_D05 : L=10mm, D_piezo= 5mm (R_disc=2.5mm)
  L10_D10 : L=10mm, D_piezo=10mm (R_disc=5.0mm, borderline — disc ≈ bottom)
  L30_D05 : L=30mm, D_piezo= 5mm (R_disc=2.5mm)
  L30_D10 : L=30mm, D_piezo=10mm (R_disc=5.0mm)
  L30_D20 : L=30mm, D_piezo=20mm (R_disc=10.0mm)

For each configuration: Cases A (standing), B (vortex), C (combined).
Gor'kov potential, trap finding, authority/locality/selectivity metrics.
Amplitude sweep for combined: V₀×{1, 3, 6}.

Usage:
    micromamba run -n acousto-complex python scripts/experiments/phase1_sweep.py
"""
from __future__ import annotations

import sys, os, csv, json, time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, replace, field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from scipy.ndimage import minimum_filter, label

from mpi4py import MPI
from dolfinx import fem
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells

from acoustweezers.experiments.shallow_square_dish.config import ShallowDishConfig
from acoustweezers.experiments.shallow_square_dish.solve_pressure import (
    create_mesh,
    solve_helmholtz,
    TAG_BOTTOM_DISC, TAG_BOTTOM_RIGID,
    TAG_TOP, TAG_X0, TAG_XL, TAG_Y0, TAG_YL,
)
from acoustweezers.experiments.shallow_square_dish.particles import (
    compute_gorkov_potential,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import TwoSlopeNorm

# ── Globals ─────────────────────────────────────────────────────────
comm = MPI.COMM_WORLD
rank = comm.rank
ROOT = Path(__file__).resolve().parents[2]
NOW = datetime.now()
STAMP = NOW.strftime("%Y%m%d_%H%M")
OUTROOT = ROOT / "results" / f"phase1_sweep_{STAMP}"
N_PLANE = 201

MUMPS_OPTS = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}


def log(msg="", end="\n"):
    if rank == 0:
        print(msg, end=end, flush=True)


# ── Configuration Matrix ────────────────────────────────────────────

@dataclass
class SweepConfig:
    """One configuration in the sweep."""
    label: str
    L: float           # dish size [m]
    R_disc: float      # disc radius [m]
    D_piezo_mm: float  # label
    L_mm: float        # label
    elem_per_wl: int = 10
    note: str = ""


SWEEP_CONFIGS = [
    SweepConfig("L10_D05",  10e-3,  2.5e-3,  5, 10, 10, ""),
    SweepConfig("L10_D10",  10e-3,  5.0e-3, 10, 10, 10,
                "Borderline: disc covers nearly all bottom"),
    SweepConfig("L30_D05",  30e-3,  2.5e-3,  5, 30,  5, ""),
    SweepConfig("L30_D10",  30e-3,  5.0e-3, 10, 30,  5, ""),
    SweepConfig("L30_D20",  30e-3, 10.0e-3, 20, 30,  5, ""),
]


def make_shallow_config(sc: SweepConfig) -> ShallowDishConfig:
    """Create a ShallowDishConfig from a sweep config."""
    return ShallowDishConfig(
        L=sc.L, H=1e-3,
        frequency_hz=500e3,
        elements_per_wavelength=sc.elem_per_wl,
        min_elements_z=8,
        rho=997.0, c=1484.0, mu=1.002e-3,
        vortex_velocity_amplitude=10e-6,
        standing_velocity_amplitude=10e-6,
        vortex_topological_charge=1,
        vortex_aperture_radius=sc.R_disc,
        vortex_apodization="cosine_taper",
        vortex_phase_offset=0.0,
        standing_axis="both",
        standing_phase_pattern="antiphase",
        top_bc_type="impedance",
        top_impedance_factor=0.001,
        bottom_disc_radius=None,
        standing_full_wall=True,
        particle_radius=5e-6,
        particle_density=1050.0,
        particle_compressibility=2.4e-10,
    )


# ── Sampling Utilities ──────────────────────────────────────────────

def make_plane_grid(cfg, z_val, N):
    xs = np.linspace(0, cfg.L, N)
    ys = np.linspace(0, cfg.L, N)
    X, Y = np.meshgrid(xs, ys)
    return np.column_stack([X.ravel(), Y.ravel(), np.full(N*N, z_val)])


def sample_pressure(p_func, pts):
    domain = p_func.function_space.mesh
    tree = bb_tree(domain, domain.topology.dim)
    cands = compute_collisions_points(tree, pts)
    cells = compute_colliding_cells(domain, cands, pts)
    vals = np.full(len(pts), np.nan + 0j, dtype=np.complex128)
    for i in range(len(pts)):
        links = cells.links(i)
        if len(links) == 0:
            continue
        vals[i] = complex(p_func.eval(pts[i], links[0])[0])
    return vals


def sample_scalar(func, pts):
    """Sample a real scalar FE function."""
    domain = func.function_space.mesh
    tree = bb_tree(domain, domain.topology.dim)
    cands = compute_collisions_points(tree, pts)
    cells = compute_colliding_cells(domain, cands, pts)
    vals = np.full(len(pts), np.nan, dtype=np.float64)
    for i in range(len(pts)):
        links = cells.links(i)
        if len(links) == 0:
            continue
        vals[i] = float(func.eval(pts[i], links[0])[0])
    return vals


# ── Gor'kov on Sampled Grid (finite-difference approach) ───────────

def compute_gorkov_grid(p_2d, dx, dy, cfg):
    """
    Compute Gor'kov potential on a 2D sampled pressure grid.

    Uses finite differences for ∇p. Returns U and auxiliary fields.
    """
    omega = cfg.omega
    rho = cfg.rho
    c = cfg.c
    f1 = cfg.f1_monopole
    f2 = cfg.f2_dipole
    V_p = cfg.particle_volume
    K = rho * c**2

    # |p|² and time-averaged ⟨p²⟩ = |p|²/2
    p_sq = np.abs(p_2d)**2
    p2_avg = p_sq / 2.0

    # ∇p via central finite differences (complex gradient)
    dpx = np.gradient(p_2d, dx, axis=1)
    dpy = np.gradient(p_2d, dy, axis=0)
    grad_p_sq = np.abs(dpx)**2 + np.abs(dpy)**2

    # |v|² = |∇p|²/(ω²ρ²), time-averaged ⟨v²⟩ = |v|²/2
    v2_avg = grad_p_sq / (2.0 * omega**2 * rho**2)

    # Gor'kov potential
    U = V_p * (f1 * p2_avg / (2.0 * K) - f2 * (3.0 * rho / 4.0) * v2_avg)

    return U, p_sq, grad_p_sq


# ── Trap Finding ────────────────────────────────────────────────────

def find_traps_2d(U, min_size=3):
    """
    Find local minima of Gor'kov potential on 2D grid.

    Returns list of (row, col, depth) for each trap.
    Depth = U_barrier - U_min, where barrier is the max of U on a
    small ring around the minimum.
    """
    # Replace NaN with large value for filtering
    U_clean = np.copy(U)
    nan_mask = np.isnan(U_clean)
    U_clean[nan_mask] = np.nanmax(U) if np.any(~nan_mask) else 0

    # Local minimum: U == minimum_filter(U) within a window
    footprint_size = 2 * min_size + 1
    local_min_map = (U_clean == minimum_filter(U_clean, size=footprint_size))
    local_min_map &= ~nan_mask

    # Barrier: max U in a ring around each minimum
    ring_size = 2 * min_size + 1
    from scipy.ndimage import maximum_filter
    local_max_map = maximum_filter(U_clean, size=ring_size)

    rows, cols = np.where(local_min_map)
    traps = []
    for r, c in zip(rows, cols):
        U_min = U_clean[r, c]
        U_barrier = local_max_map[r, c]
        depth = U_barrier - U_min
        traps.append((int(r), int(c), float(U_min), float(depth)))

    return traps


# ── Metrics ─────────────────────────────────────────────────────────

@dataclass
class CaseMetrics:
    """Metrics for a single solve."""
    label: str
    mode: str
    max_p: float
    max_p_plane: float
    p_center: float
    n_traps: int
    mean_trap_depth: float
    std_trap_depth: float
    gorkov_min: float
    gorkov_max: float
    gorkov_range: float


@dataclass
class ComparisonMetrics:
    """Authority / locality / selectivity metrics comparing combined vs standing."""
    label: str
    V0_mult: float
    authority_disc: float       # avg |ΔU|/|U_stand| inside disc
    authority_p_disc: float     # avg |Δ|p||/|p_stand| inside disc
    locality_radius_mm: float   # radius where vortex effect drops below 10%
    selectivity_ratio: float    # depth nearest disc / median depth
    n_traps_combined: int
    n_traps_standing: int
    delta_n_traps: int


def compute_comparison_metrics(
    U_stand, U_comb, p_stand_2d, p_comb_2d, cfg, sc, V0_mult,
):
    """Compute authority, locality, selectivity between standing and combined."""
    N = U_stand.shape[0]
    L_mm = cfg.L * 1e3
    xs_mm = np.linspace(0, L_mm, N)
    ys_mm = np.linspace(0, L_mm, N)
    Xg, Yg = np.meshgrid(xs_mm, ys_mm)
    cx = L_mm / 2
    cy = L_mm / 2
    R_disc_mm = sc.R_disc * 1e3
    dist_from_center = np.sqrt((Xg - cx)**2 + (Yg - cy)**2)
    inside_disc = dist_from_center <= R_disc_mm

    # Authority (Gor'kov)
    dU = U_comb - U_stand
    U_stand_abs = np.abs(U_stand)
    with np.errstate(divide='ignore', invalid='ignore'):
        relative_dU = np.abs(dU) / np.where(U_stand_abs > 1e-30, U_stand_abs, np.nan)
    auth_disc = float(np.nanmean(relative_dU[inside_disc]))

    # Authority (pressure)
    dp = np.abs(p_comb_2d) - np.abs(p_stand_2d)
    p_stand_abs = np.abs(p_stand_2d)
    with np.errstate(divide='ignore', invalid='ignore'):
        relative_dp = np.abs(dp) / np.where(p_stand_abs > 0.01, p_stand_abs, np.nan)
    auth_p_disc = float(np.nanmean(relative_dp[inside_disc]))

    # Locality: radial profile of vortex effect
    radii_mm = np.linspace(0, L_mm / 2, 100)
    radial_effect = np.zeros(len(radii_mm))
    dr = radii_mm[1] - radii_mm[0]
    for i, r in enumerate(radii_mm):
        ring = (dist_from_center >= r) & (dist_from_center < r + dr)
        if np.sum(ring) > 0:
            vals = relative_dU[ring]
            radial_effect[i] = float(np.nanmean(vals)) if np.any(~np.isnan(vals)) else 0
    # Find radius where effect drops below 10%
    threshold = 0.10
    locality_r = L_mm / 2  # default: entire domain
    for i in range(len(radii_mm)):
        if radial_effect[i] < threshold and i > 2:
            locality_r = float(radii_mm[i])
            break

    # Trap finding
    traps_stand = find_traps_2d(U_stand)
    traps_comb = find_traps_2d(U_comb)

    # Selectivity: depth of nearest-to-center trap vs median
    selectivity = 0.0
    if len(traps_comb) > 0:
        depths_comb = np.array([t[3] for t in traps_comb])
        median_depth = float(np.median(depths_comb))
        # Find trap nearest to disc center
        center_r = N // 2
        center_c = N // 2
        dists = [np.sqrt((t[0] - center_r)**2 + (t[1] - center_c)**2) for t in traps_comb]
        nearest_idx = int(np.argmin(dists))
        nearest_depth = float(traps_comb[nearest_idx][3])
        if median_depth > 0:
            selectivity = nearest_depth / median_depth

    return ComparisonMetrics(
        label=f"{sc.label}_V0x{V0_mult}",
        V0_mult=V0_mult,
        authority_disc=auth_disc,
        authority_p_disc=auth_p_disc,
        locality_radius_mm=locality_r,
        selectivity_ratio=selectivity,
        n_traps_combined=len(traps_comb),
        n_traps_standing=len(traps_stand),
        delta_n_traps=len(traps_comb) - len(traps_stand),
    )


# ── Plotting ────────────────────────────────────────────────────────

def disc_circle(cfg, color="white", ls="--"):
    cx = cfg.L / 2 * 1e3
    cy = cfg.L / 2 * 1e3
    R = cfg.bottom_disc_radius_effective * 1e3
    return Circle((cx, cy), R, fill=False, edgecolor=color,
                  linewidth=1.2, linestyle=ls)


def plot_pressure_field(p_2d, cfg, title, path, cmap="jet", n_levels=20):
    """Plot |p| contourf."""
    N = p_2d.shape[0]
    xs = np.linspace(0, cfg.L*1e3, N)
    ys = np.linspace(0, cfg.L*1e3, N)
    X, Y = np.meshgrid(xs, ys)
    data = np.abs(p_2d)
    maxp = np.nanmax(data)

    fig, ax = plt.subplots(figsize=(7, 5.8))
    cf = ax.contourf(X, Y, data, levels=n_levels, cmap=cmap)
    ax.contour(X, Y, data, levels=n_levels, colors="k", linewidths=0.3, alpha=0.5)
    cb = fig.colorbar(cf, ax=ax, shrink=0.82, pad=0.03)
    cb.set_label("|p| (Pa)", fontsize=11)
    ax.add_patch(disc_circle(cfg))
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    ax.set_title(f"{title}\nmax|p| = {maxp:.2f} Pa", fontsize=11)
    ax.set_aspect("equal"); fig.tight_layout()
    fig.savefig(path, dpi=180); plt.close(fig)


def plot_gorkov_field(U_2d, cfg, title, path, n_levels=20):
    """Plot Gor'kov potential contourf (diverging)."""
    N = U_2d.shape[0]
    xs = np.linspace(0, cfg.L*1e3, N)
    ys = np.linspace(0, cfg.L*1e3, N)
    X, Y = np.meshgrid(xs, ys)
    vmax = max(abs(np.nanmin(U_2d)), abs(np.nanmax(U_2d)))
    if vmax < 1e-40:
        vmax = 1e-30

    fig, ax = plt.subplots(figsize=(7, 5.8))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cf = ax.contourf(X, Y, U_2d, levels=n_levels, cmap="RdBu_r", norm=norm)
    ax.contour(X, Y, U_2d, levels=n_levels, colors="k", linewidths=0.25, alpha=0.4)
    cb = fig.colorbar(cf, ax=ax, shrink=0.82, pad=0.03)
    cb.set_label("U (J)", fontsize=11)
    ax.add_patch(disc_circle(cfg, "black"))
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    rng = np.nanmax(U_2d) - np.nanmin(U_2d)
    ax.set_title(f"{title}\nΔU = {rng:.2e} J", fontsize=11)
    ax.set_aspect("equal"); fig.tight_layout()
    fig.savefig(path, dpi=180); plt.close(fig)


def plot_trap_map(U_2d, traps, cfg, title, path):
    """Plot trap locations on Gor'kov potential."""
    N = U_2d.shape[0]
    xs = np.linspace(0, cfg.L*1e3, N)
    ys = np.linspace(0, cfg.L*1e3, N)
    X, Y = np.meshgrid(xs, ys)
    dx_mm = xs[1] - xs[0]

    fig, ax = plt.subplots(figsize=(7, 5.8))
    cf = ax.contourf(X, Y, U_2d, levels=20, cmap="viridis")
    cb = fig.colorbar(cf, ax=ax, shrink=0.82, pad=0.03)
    cb.set_label("U (J)", fontsize=11)

    if len(traps) > 0:
        trap_xs = [t[1] * dx_mm for t in traps]
        trap_ys = [t[0] * dx_mm for t in traps]
        ax.scatter(trap_xs, trap_ys, c="red", s=18, marker="x",
                   linewidths=0.8, zorder=5)

    ax.add_patch(disc_circle(cfg, "white"))
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    ax.set_title(f"{title}\n{len(traps)} traps", fontsize=11)
    ax.set_aspect("equal"); fig.tight_layout()
    fig.savefig(path, dpi=180); plt.close(fig)


def plot_phase_field(p_2d, cfg, title, path):
    """Plot arg(p)."""
    N = p_2d.shape[0]
    ext = [0, cfg.L*1e3, 0, cfg.L*1e3]
    data = np.angle(p_2d)
    fig, ax = plt.subplots(figsize=(7, 5.8))
    im = ax.imshow(data, origin="lower", extent=ext, cmap="twilight_shifted",
                   vmin=-np.pi, vmax=np.pi, aspect="equal")
    cb = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.03)
    cb.set_label("arg(p) (rad)", fontsize=11)
    cb.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    cb.set_ticklabels(["-π", "-π/2", "0", "π/2", "π"])
    ax.add_patch(disc_circle(cfg, "black"))
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=180); plt.close(fig)


# ── Main Processing ─────────────────────────────────────────────────

def process_config(sc: SweepConfig):
    """
    Run Cases A, B, C + amplitude sweep for one configuration.
    Returns dict of metrics.
    """
    cfg = make_shallow_config(sc)
    cdir = OUTROOT / sc.label
    (cdir / "figs").mkdir(parents=True, exist_ok=True)
    (cdir / "csv").mkdir(parents=True, exist_ok=True)

    log(f"\n{'#'*70}")
    log(f"  CONFIG: {sc.label}  (L={sc.L_mm}mm, D_piezo={sc.D_piezo_mm}mm, "
        f"R_disc={sc.R_disc*1e3:.1f}mm)")
    if sc.note:
        log(f"  NOTE: {sc.note}")
    log(f"  Mesh: {cfg.mesh_nx}×{cfg.mesh_nx}×{cfg.mesh_nz}, "
        f"λ={cfg.wavelength*1e3:.2f}mm, {sc.elem_per_wl} elem/λ")
    log(f"{'#'*70}")

    # Check feasibility
    margin = cfg.L * 0.05  # 5% margin
    if sc.R_disc > cfg.L / 2 - margin:
        log(f"  WARNING: R_disc ({sc.R_disc*1e3:.1f}mm) ≈ L/2 "
            f"({cfg.L/2*1e3:.1f}mm) — disc covers most of bottom!")

    # Create mesh
    domain, facet_tags, _ = create_mesh(cfg, verbose=True)
    n_cells = domain.topology.index_map(domain.topology.dim).size_global

    # Sampling grids
    z_H2 = cfg.H / 2.0
    pts_H2 = make_plane_grid(cfg, z_H2, N_PLANE)
    dx_m = cfg.L / (N_PLANE - 1)
    dy_m = dx_m

    # Disc mask for metrics
    xs_mm = np.linspace(0, cfg.L*1e3, N_PLANE)
    ys_mm = np.linspace(0, cfg.L*1e3, N_PLANE)
    Xg, Yg = np.meshgrid(xs_mm, ys_mm)
    cx_mm = cfg.L / 2 * 1e3
    cy_mm = cfg.L / 2 * 1e3
    R_mm = cfg.bottom_disc_radius_effective * 1e3
    inside_disc = np.sqrt((Xg - cx_mm)**2 + (Yg - cy_mm)**2) <= R_mm
    mid = N_PLANE // 2

    results = {"label": sc.label, "L_mm": sc.L_mm, "D_piezo_mm": sc.D_piezo_mm,
               "R_disc_mm": sc.R_disc*1e3, "n_cells": n_cells,
               "mesh_nx": cfg.mesh_nx, "mesh_nz": cfg.mesh_nz}
    case_metrics = []
    comp_metrics = []
    solves = {}

    # ════════════════════════════════════
    # Case A — Standing
    # ════════════════════════════════════
    log(f"\n  ── Case A: Standing ──")
    p_sol_A = solve_helmholtz(domain, facet_tags, cfg, mode="standing",
                              disc_robin=False, verbose=True,
                              petsc_options=MUMPS_OPTS)
    p_A_H2 = sample_pressure(p_sol_A.p_function, pts_H2)
    p_A_2d = p_A_H2.reshape(N_PLANE, N_PLANE)
    U_A, _, _ = compute_gorkov_grid(p_A_2d, dx_m, dy_m, cfg)
    traps_A = find_traps_2d(U_A)

    cm_A = CaseMetrics(
        label=f"{sc.label}_A", mode="standing",
        max_p=p_sol_A.max_pressure,
        max_p_plane=float(np.nanmax(np.abs(p_A_H2))),
        p_center=float(np.abs(p_A_2d[mid, mid])),
        n_traps=len(traps_A),
        mean_trap_depth=float(np.mean([t[3] for t in traps_A])) if traps_A else 0,
        std_trap_depth=float(np.std([t[3] for t in traps_A])) if traps_A else 0,
        gorkov_min=float(np.nanmin(U_A)),
        gorkov_max=float(np.nanmax(U_A)),
        gorkov_range=float(np.nanmax(U_A) - np.nanmin(U_A)),
    )
    case_metrics.append(cm_A)
    solves["A"] = (p_A_2d, U_A, traps_A)
    log(f"    max|p|={cm_A.max_p:.2f} Pa, traps={cm_A.n_traps}, "
        f"ΔU={cm_A.gorkov_range:.2e} J")

    # Figures
    plot_pressure_field(p_A_2d, cfg, f"{sc.label} — Standing |p|",
                        cdir / "figs" / "A_standing_abs_p.png")
    plot_gorkov_field(U_A, cfg, f"{sc.label} — Standing Gor'kov",
                      cdir / "figs" / "A_standing_gorkov.png")
    plot_trap_map(U_A, traps_A, cfg, f"{sc.label} — Standing traps",
                  cdir / "figs" / "A_standing_traps.png")
    plot_phase_field(p_A_2d, cfg, f"{sc.label} — Standing arg(p)",
                     cdir / "figs" / "A_standing_phase.png")

    # ════════════════════════════════════
    # Case B — Vortex
    # ════════════════════════════════════
    log(f"\n  ── Case B: Vortex ──")
    p_sol_B = solve_helmholtz(domain, facet_tags, cfg, mode="vortex",
                              disc_robin=False, verbose=True,
                              petsc_options=MUMPS_OPTS)
    p_B_H2 = sample_pressure(p_sol_B.p_function, pts_H2)
    p_B_2d = p_B_H2.reshape(N_PLANE, N_PLANE)
    U_B, _, _ = compute_gorkov_grid(p_B_2d, dx_m, dy_m, cfg)
    traps_B = find_traps_2d(U_B)

    cm_B = CaseMetrics(
        label=f"{sc.label}_B", mode="vortex",
        max_p=p_sol_B.max_pressure,
        max_p_plane=float(np.nanmax(np.abs(p_B_H2))),
        p_center=float(np.abs(p_B_2d[mid, mid])),
        n_traps=len(traps_B),
        mean_trap_depth=float(np.mean([t[3] for t in traps_B])) if traps_B else 0,
        std_trap_depth=float(np.std([t[3] for t in traps_B])) if traps_B else 0,
        gorkov_min=float(np.nanmin(U_B)),
        gorkov_max=float(np.nanmax(U_B)),
        gorkov_range=float(np.nanmax(U_B) - np.nanmin(U_B)),
    )
    case_metrics.append(cm_B)
    solves["B"] = (p_B_2d, U_B, traps_B)
    log(f"    max|p|={cm_B.max_p:.2f} Pa, traps={cm_B.n_traps}, "
        f"ΔU={cm_B.gorkov_range:.2e} J")

    plot_pressure_field(p_B_2d, cfg, f"{sc.label} — Vortex |p|",
                        cdir / "figs" / "B_vortex_abs_p.png")
    plot_gorkov_field(U_B, cfg, f"{sc.label} — Vortex Gor'kov",
                      cdir / "figs" / "B_vortex_gorkov.png")
    plot_phase_field(p_B_2d, cfg, f"{sc.label} — Vortex arg(p)",
                     cdir / "figs" / "B_vortex_phase.png")

    # ════════════════════════════════════
    # Case C — Combined (amplitude sweep)
    # ════════════════════════════════════
    V0_mults = [1, 3, 6]
    for mult in V0_mults:
        tag = f"C_V0x{mult}"
        log(f"\n  ── Case C: Combined V₀×{mult} ──")
        cfg_m = replace(cfg, vortex_velocity_amplitude=mult * cfg.vortex_velocity_amplitude)
        p_sol_C = solve_helmholtz(domain, facet_tags, cfg_m, mode="combined",
                                  disc_robin=False, verbose=True,
                                  petsc_options=MUMPS_OPTS)
        p_C_H2 = sample_pressure(p_sol_C.p_function, pts_H2)
        p_C_2d = p_C_H2.reshape(N_PLANE, N_PLANE)
        U_C, _, _ = compute_gorkov_grid(p_C_2d, dx_m, dy_m, cfg_m)
        traps_C = find_traps_2d(U_C)

        cm_C = CaseMetrics(
            label=f"{sc.label}_{tag}", mode=f"combined_V0x{mult}",
            max_p=p_sol_C.max_pressure,
            max_p_plane=float(np.nanmax(np.abs(p_C_H2))),
            p_center=float(np.abs(p_C_2d[mid, mid])),
            n_traps=len(traps_C),
            mean_trap_depth=float(np.mean([t[3] for t in traps_C])) if traps_C else 0,
            std_trap_depth=float(np.std([t[3] for t in traps_C])) if traps_C else 0,
            gorkov_min=float(np.nanmin(U_C)),
            gorkov_max=float(np.nanmax(U_C)),
            gorkov_range=float(np.nanmax(U_C) - np.nanmin(U_C)),
        )
        case_metrics.append(cm_C)
        log(f"    max|p|={cm_C.max_p:.2f} Pa, traps={cm_C.n_traps}, "
            f"ΔU={cm_C.gorkov_range:.2e} J")

        # Comparison metrics
        comp = compute_comparison_metrics(
            U_A, U_C, p_A_2d, p_C_2d, cfg_m, sc, mult)
        comp_metrics.append(comp)
        log(f"    authority(disc)={comp.authority_disc:.3f}, "
            f"locality={comp.locality_radius_mm:.1f}mm, "
            f"selectivity={comp.selectivity_ratio:.3f}")

        # Figures
        plot_pressure_field(p_C_2d, cfg_m,
                            f"{sc.label} — Combined V₀×{mult} |p|",
                            cdir / "figs" / f"{tag}_abs_p.png")
        plot_gorkov_field(U_C, cfg_m,
                          f"{sc.label} — Combined V₀×{mult} Gor'kov",
                          cdir / "figs" / f"{tag}_gorkov.png")
        plot_trap_map(U_C, traps_C, cfg_m,
                      f"{sc.label} — Combined V₀×{mult} traps",
                      cdir / "figs" / f"{tag}_traps.png")
        plot_phase_field(p_C_2d, cfg_m,
                         f"{sc.label} — Combined V₀×{mult} arg(p)",
                         cdir / "figs" / f"{tag}_phase.png")

        if mult == 1:
            solves["C"] = (p_C_2d, U_C, traps_C)

    results["case_metrics"] = [cm.__dict__ for cm in case_metrics]
    results["comparison_metrics"] = [cm.__dict__ for cm in comp_metrics]

    # Save CSV summary
    with open(cdir / "csv" / "case_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=case_metrics[0].__dict__.keys())
        w.writeheader()
        for cm in case_metrics:
            w.writerow(cm.__dict__)
    with open(cdir / "csv" / "comparison_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=comp_metrics[0].__dict__.keys())
        w.writeheader()
        for cm in comp_metrics:
            w.writerow(cm.__dict__)

    return results, case_metrics, comp_metrics, solves, cfg


# ── Comparison Figures (Cross-Config) ───────────────────────────────

def generate_comparison_figures(all_results, all_case_metrics,
                                all_comp_metrics, all_cfgs):
    """Generate main comparison figures across all configurations."""
    compdir = OUTROOT / "comparison"
    compdir.mkdir(parents=True, exist_ok=True)

    # ── R3: Piezo size decision table ──
    log(f"\n{'='*70}")
    log("  COMPARISON TABLE")
    log(f"{'='*70}")

    header = (f"  {'Config':<12} {'max|p|_A':>10} {'max|p|_B':>10} "
              f"{'max|p|_C':>10} {'traps_A':>8} {'traps_C':>8} "
              f"{'authority':>10} {'locality':>10} {'selectiv':>10}")
    log(header)
    log(f"  {'-'*98}")

    summary_rows = []
    for res, cms, comps in zip(all_results, all_case_metrics, all_comp_metrics):
        label = res["label"]
        # Find A, B, C_V0x1 metrics
        cm_A = next(c for c in cms if c.mode == "standing")
        cm_B = next(c for c in cms if c.mode == "vortex")
        cm_C1 = next(c for c in cms if c.mode == "combined_V0x1")
        comp1 = next(c for c in comps if c.V0_mult == 1)

        row = {
            "config": label,
            "L_mm": res["L_mm"],
            "D_piezo_mm": res["D_piezo_mm"],
            "R_disc_mm": res["R_disc_mm"],
            "max_p_A": cm_A.max_p,
            "max_p_B": cm_B.max_p,
            "max_p_C": cm_C1.max_p,
            "traps_A": cm_A.n_traps,
            "traps_C": cm_C1.n_traps,
            "gorkov_range_A": cm_A.gorkov_range,
            "gorkov_range_B": cm_B.gorkov_range,
            "gorkov_range_C": cm_C1.gorkov_range,
            "authority_disc": comp1.authority_disc,
            "authority_p_disc": comp1.authority_p_disc,
            "locality_mm": comp1.locality_radius_mm,
            "selectivity": comp1.selectivity_ratio,
        }
        summary_rows.append(row)

        log(f"  {label:<12} {cm_A.max_p:>8.1f}Pa {cm_B.max_p:>8.1f}Pa "
            f"{cm_C1.max_p:>8.1f}Pa {cm_A.n_traps:>8} {cm_C1.n_traps:>8} "
            f"{comp1.authority_disc:>10.3f} "
            f"{comp1.locality_radius_mm:>8.1f}mm "
            f"{comp1.selectivity_ratio:>10.3f}")

    # Save decision table CSV
    with open(compdir / "decision_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        w.writeheader()
        for row in summary_rows:
            w.writerow(row)

    # ── Bar chart: authority vs config ──
    labels = [r["config"] for r in summary_rows]
    authorities = [r["authority_disc"] for r in summary_rows]
    localities = [r["locality_mm"] for r in summary_rows]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = np.arange(len(labels))
    w = 0.6

    axes[0].bar(x, authorities, w, color="steelblue")
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, rotation=30, ha="right")
    axes[0].set_ylabel("Authority (avg |ΔU/U| in disc)")
    axes[0].set_title("Vortex Authority at V₀×1")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x, localities, w, color="darkorange")
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, rotation=30, ha="right")
    axes[1].set_ylabel("Locality radius (mm)")
    axes[1].set_title("Vortex Locality (10% threshold)")
    axes[1].grid(axis="y", alpha=0.3)

    selectivities = [r["selectivity"] for r in summary_rows]
    axes[2].bar(x, selectivities, w, color="seagreen")
    axes[2].set_xticks(x); axes[2].set_xticklabels(labels, rotation=30, ha="right")
    axes[2].set_ylabel("Selectivity ratio")
    axes[2].set_title("Selectivity (disc-centre / median)")
    axes[2].grid(axis="y", alpha=0.3)

    fig.suptitle("Phase 1 — Piezo & Dish Comparison", fontsize=14)
    fig.tight_layout()
    fig.savefig(compdir / "authority_locality_selectivity.png", dpi=200)
    plt.close(fig)

    # ── Amplitude sweep curves (all configs) ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for res, comps in zip(all_results, all_comp_metrics):
        label = res["label"]
        mults = [c.V0_mult for c in comps]
        auths = [c.authority_disc for c in comps]
        locs = [c.locality_radius_mm for c in comps]
        sels = [c.selectivity_ratio for c in comps]
        axes[0].plot(mults, auths, "o-", label=label, linewidth=1.5)
        axes[1].plot(mults, locs, "s-", label=label, linewidth=1.5)
        axes[2].plot(mults, sels, "^-", label=label, linewidth=1.5)

    for ax, ylabel, title in zip(axes,
        ["Authority", "Locality (mm)", "Selectivity"],
        ["Authority vs V₀ multiplier", "Locality vs V₀ multiplier",
         "Selectivity vs V₀ multiplier"]):
        ax.set_xlabel("V₀ multiplier")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Phase 1 — Amplitude Sweep Across Configs", fontsize=14)
    fig.tight_layout()
    fig.savefig(compdir / "amplitude_sweep_curves.png", dpi=200)
    plt.close(fig)

    # ── Trap count & depth bar chart ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    traps_A = [next(c for c in cms if c.mode == "standing").n_traps
               for cms in all_case_metrics]
    traps_C = [next(c for c in cms if c.mode == "combined_V0x1").n_traps
               for cms in all_case_metrics]
    depth_A = [next(c for c in cms if c.mode == "standing").gorkov_range
               for cms in all_case_metrics]
    depth_C = [next(c for c in cms if c.mode == "combined_V0x1").gorkov_range
               for cms in all_case_metrics]

    x = np.arange(len(labels))
    axes[0].bar(x - 0.15, traps_A, 0.3, label="Standing", color="steelblue")
    axes[0].bar(x + 0.15, traps_C, 0.3, label="Combined V₀", color="coral")
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, rotation=30, ha="right")
    axes[0].set_ylabel("Trap count")
    axes[0].set_title("Trap Count: Standing vs Combined")
    axes[0].legend(); axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x - 0.15, depth_A, 0.3, label="Standing", color="steelblue")
    axes[1].bar(x + 0.15, depth_C, 0.3, label="Combined V₀", color="coral")
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, rotation=30, ha="right")
    axes[1].set_ylabel("Gor'kov range ΔU (J)")
    axes[1].set_title("Trap Depth (Gor'kov Range)")
    axes[1].legend(); axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(compdir / "trap_count_depth.png", dpi=200)
    plt.close(fig)

    # ── 10mm vs 30mm scaling comparison ──
    configs_10 = [r for r in summary_rows if r["L_mm"] == 10]
    configs_30 = [r for r in summary_rows if r["L_mm"] == 30]

    if configs_10 and configs_30:
        fig, ax = plt.subplots(figsize=(8, 5))
        for configs, color, marker in [(configs_10, "blue", "o"),
                                        (configs_30, "red", "s")]:
            ds = [c["D_piezo_mm"] for c in configs]
            auths = [c["authority_disc"] for c in configs]
            ax.plot(ds, auths, f"{marker}-", color=color,
                    label=f"L={configs[0]['L_mm']}mm", linewidth=2, markersize=8)
        ax.set_xlabel("Piezo diameter (mm)")
        ax.set_ylabel("Vortex authority at V₀")
        ax.set_title("Authority vs Piezo Diameter — Dish Size Comparison")
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(compdir / "scaling_authority_vs_diameter.png", dpi=200)
        plt.close(fig)

    return summary_rows


# ── Main ────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    OUTROOT.mkdir(parents=True, exist_ok=True)

    log(f"\n{'#'*70}")
    log(f"  PHASE 1 — TRANSDUCER SIZE & DISH SIZE SWEEP")
    log(f"  Output: {OUTROOT.relative_to(ROOT)}")
    log(f"  Configs: {len(SWEEP_CONFIGS)}")
    log(f"{'#'*70}")

    all_results = []
    all_case_metrics = []
    all_comp_metrics = []
    all_cfgs = []

    for sc in SWEEP_CONFIGS:
        try:
            res, cms, comps, solves, cfg = process_config(sc)
            all_results.append(res)
            all_case_metrics.append(cms)
            all_comp_metrics.append(comps)
            all_cfgs.append(cfg)
        except Exception as e:
            log(f"\n  *** FAILED: {sc.label}: {e} ***")
            import traceback
            traceback.print_exc()

    # ── Cross-config comparison ──
    if all_results:
        summary_rows = generate_comparison_figures(
            all_results, all_case_metrics, all_comp_metrics, all_cfgs)

        # Save full JSON
        with open(OUTROOT / "all_results.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    elapsed = time.time() - t0
    log(f"\n{'#'*70}")
    log(f"  PHASE 1 COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f} min)")
    log(f"  Output: {OUTROOT.relative_to(ROOT)}")
    log(f"{'#'*70}")


if __name__ == "__main__":
    main()
