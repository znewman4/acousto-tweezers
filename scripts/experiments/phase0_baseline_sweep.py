#!/usr/bin/env python3
"""
Phase 0 — Lock the 10×10 mm baseline + smaller vortex disc.

Goal: L=10 mm, H=1 mm testbed.  Sweep disc diameters D∈{2,3,4} mm
so the disc covers < ~15 % of bottom area, and verify the vortex
produces a controllable perturbation without global disruption.

For each preset (L10_D02, L10_D03, L10_D04) runs:
    Case A — Standing only
    Case B — Vortex only
    Case C — Combined (V₀×{1, 3, 6})

Outputs (per config):
    figs/  — |p|, arg(p), Gor'kov, trap maps
    csv/   — case_metrics.csv, comparison_metrics.csv

Outputs (cross-config):
    comparison/ — decision_table.csv, comparison charts

Usage:
    micromamba run -n acousto-complex python scripts/experiments/phase0_baseline_sweep.py
"""
from __future__ import annotations

import sys, os, csv, json, time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, replace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from scipy.ndimage import minimum_filter, maximum_filter

from mpi4py import MPI
from dolfinx import fem
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells

from acoustweezers.experiments.shallow_square_dish.config import (
    ShallowDishConfig, PHASE0_PRESETS,
)
from acoustweezers.experiments.shallow_square_dish.solve_pressure import (
    create_mesh,
    solve_helmholtz,
    TAG_BOTTOM_DISC, TAG_BOTTOM_RIGID,
    TAG_TOP, TAG_X0, TAG_XL, TAG_Y0, TAG_YL,
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
OUTROOT = ROOT / "results" / f"phase0_baseline_{STAMP}"
N_PLANE = 201

MUMPS_OPTS = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}


def log(msg="", end="\n"):
    if rank == 0:
        print(msg, end=end, flush=True)


# ── Sampling Utilities ──────────────────────────────────────────────

def make_plane_grid(cfg, z_val, N):
    xs = np.linspace(0, cfg.L, N)
    ys = np.linspace(0, cfg.L, N)
    X, Y = np.meshgrid(xs, ys)
    return np.column_stack([X.ravel(), Y.ravel(), np.full(N * N, z_val)])


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


# ── Gor'kov on Sampled Grid ────────────────────────────────────────

def compute_gorkov_grid(p_2d, dx, dy, cfg):
    omega = cfg.omega
    rho = cfg.rho
    c = cfg.c
    f1 = cfg.f1_monopole
    f2 = cfg.f2_dipole
    V_p = cfg.particle_volume
    K = rho * c ** 2

    p_sq = np.abs(p_2d) ** 2
    p2_avg = p_sq / 2.0

    dpx = np.gradient(p_2d, dx, axis=1)
    dpy = np.gradient(p_2d, dy, axis=0)
    grad_p_sq = np.abs(dpx) ** 2 + np.abs(dpy) ** 2

    v2_avg = grad_p_sq / (2.0 * omega ** 2 * rho ** 2)

    U = V_p * (f1 * p2_avg / (2.0 * K) - f2 * (3.0 * rho / 4.0) * v2_avg)
    return U, p_sq, grad_p_sq


# ── Trap Finding ────────────────────────────────────────────────────

def find_traps_2d(U, min_size=3):
    U_clean = np.copy(U)
    nan_mask = np.isnan(U_clean)
    U_clean[nan_mask] = np.nanmax(U) if np.any(~nan_mask) else 0

    footprint_size = 2 * min_size + 1
    local_min_map = U_clean == minimum_filter(U_clean, size=footprint_size)
    local_min_map &= ~nan_mask

    local_max_map = maximum_filter(U_clean, size=footprint_size)

    rows, cols = np.where(local_min_map)
    traps = []
    for r, c in zip(rows, cols):
        U_min = U_clean[r, c]
        U_barrier = local_max_map[r, c]
        depth = U_barrier - U_min
        traps.append((int(r), int(c), float(U_min), float(depth)))
    return traps


# ── Metrics Dataclasses ─────────────────────────────────────────────

@dataclass
class CaseMetrics:
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
    label: str
    V0_mult: float
    authority_disc: float
    authority_p_disc: float
    locality_radius_mm: float
    selectivity_ratio: float
    n_traps_combined: int
    n_traps_standing: int
    delta_n_traps: int


def compute_comparison_metrics(U_stand, U_comb, p_stand_2d, p_comb_2d,
                                cfg, preset_label, R_disc, V0_mult):
    N = U_stand.shape[0]
    L_mm = cfg.L * 1e3
    xs_mm = np.linspace(0, L_mm, N)
    ys_mm = np.linspace(0, L_mm, N)
    Xg, Yg = np.meshgrid(xs_mm, ys_mm)
    cx = L_mm / 2
    cy = L_mm / 2
    R_disc_mm = R_disc * 1e3
    dist_from_center = np.sqrt((Xg - cx) ** 2 + (Yg - cy) ** 2)
    inside_disc = dist_from_center <= R_disc_mm

    dU = U_comb - U_stand
    U_stand_abs = np.abs(U_stand)
    with np.errstate(divide="ignore", invalid="ignore"):
        relative_dU = np.abs(dU) / np.where(
            U_stand_abs > 1e-30, U_stand_abs, np.nan
        )
    auth_disc = float(np.nanmean(relative_dU[inside_disc]))

    dp = np.abs(p_comb_2d) - np.abs(p_stand_2d)
    p_stand_abs = np.abs(p_stand_2d)
    with np.errstate(divide="ignore", invalid="ignore"):
        relative_dp = np.abs(dp) / np.where(
            p_stand_abs > 0.01, p_stand_abs, np.nan
        )
    auth_p_disc = float(np.nanmean(relative_dp[inside_disc]))

    radii_mm = np.linspace(0, L_mm / 2, 100)
    radial_effect = np.zeros(len(radii_mm))
    dr = radii_mm[1] - radii_mm[0]
    for i, r in enumerate(radii_mm):
        ring = (dist_from_center >= r) & (dist_from_center < r + dr)
        if np.sum(ring) > 0:
            vals = relative_dU[ring]
            radial_effect[i] = (
                float(np.nanmean(vals)) if np.any(~np.isnan(vals)) else 0
            )
    locality_r = L_mm / 2
    for i in range(len(radii_mm)):
        if radial_effect[i] < 0.10 and i > 2:
            locality_r = float(radii_mm[i])
            break

    traps_stand = find_traps_2d(U_stand)
    traps_comb = find_traps_2d(U_comb)

    selectivity = 0.0
    if len(traps_comb) > 0:
        depths_comb = np.array([t[3] for t in traps_comb])
        median_depth = float(np.median(depths_comb))
        center_r = N // 2
        center_c = N // 2
        dists = [
            np.sqrt((t[0] - center_r) ** 2 + (t[1] - center_c) ** 2)
            for t in traps_comb
        ]
        nearest_idx = int(np.argmin(dists))
        nearest_depth = float(traps_comb[nearest_idx][3])
        if median_depth > 0:
            selectivity = nearest_depth / median_depth

    return ComparisonMetrics(
        label=f"{preset_label}_V0x{V0_mult}",
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
    return Circle(
        (cx, cy), R, fill=False, edgecolor=color, linewidth=1.2, linestyle=ls
    )


def plot_pressure_field(p_2d, cfg, title, path, cmap="jet", n_levels=20):
    N = p_2d.shape[0]
    xs = np.linspace(0, cfg.L * 1e3, N)
    ys = np.linspace(0, cfg.L * 1e3, N)
    X, Y = np.meshgrid(xs, ys)
    data = np.abs(p_2d)
    maxp = np.nanmax(data)

    fig, ax = plt.subplots(figsize=(7, 5.8))
    cf = ax.contourf(X, Y, data, levels=n_levels, cmap=cmap)
    ax.contour(X, Y, data, levels=n_levels, colors="k", linewidths=0.3, alpha=0.5)
    cb = fig.colorbar(cf, ax=ax, shrink=0.82, pad=0.03)
    cb.set_label("|p| (Pa)", fontsize=11)
    ax.add_patch(disc_circle(cfg))
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(f"{title}\nmax|p| = {maxp:.2f} Pa", fontsize=11)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_gorkov_field(U_2d, cfg, title, path, n_levels=20):
    N = U_2d.shape[0]
    xs = np.linspace(0, cfg.L * 1e3, N)
    ys = np.linspace(0, cfg.L * 1e3, N)
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
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    rng = np.nanmax(U_2d) - np.nanmin(U_2d)
    ax.set_title(f"{title}\nΔU = {rng:.2e} J", fontsize=11)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_trap_map(U_2d, traps, cfg, title, path):
    N = U_2d.shape[0]
    xs = np.linspace(0, cfg.L * 1e3, N)
    ys = np.linspace(0, cfg.L * 1e3, N)
    X, Y = np.meshgrid(xs, ys)
    dx_mm = xs[1] - xs[0]

    fig, ax = plt.subplots(figsize=(7, 5.8))
    cf = ax.contourf(X, Y, U_2d, levels=20, cmap="viridis")
    cb = fig.colorbar(cf, ax=ax, shrink=0.82, pad=0.03)
    cb.set_label("U (J)", fontsize=11)

    if len(traps) > 0:
        trap_xs = [t[1] * dx_mm for t in traps]
        trap_ys = [t[0] * dx_mm for t in traps]
        ax.scatter(
            trap_xs, trap_ys, c="red", s=18, marker="x", linewidths=0.8, zorder=5
        )

    ax.add_patch(disc_circle(cfg, "white"))
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(f"{title}\n{len(traps)} traps", fontsize=11)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_phase_field(p_2d, cfg, title, path):
    N = p_2d.shape[0]
    ext = [0, cfg.L * 1e3, 0, cfg.L * 1e3]
    data = np.angle(p_2d)
    fig, ax = plt.subplots(figsize=(7, 5.8))
    im = ax.imshow(
        data,
        origin="lower",
        extent=ext,
        cmap="twilight_shifted",
        vmin=-np.pi,
        vmax=np.pi,
        aspect="equal",
    )
    cb = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.03)
    cb.set_label("arg(p) (rad)", fontsize=11)
    cb.set_ticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    cb.set_ticklabels(["-π", "-π/2", "0", "π/2", "π"])
    ax.add_patch(disc_circle(cfg, "black"))
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


# ── Main Processing ─────────────────────────────────────────────────

def process_preset(preset_label: str, cfg: ShallowDishConfig):
    """Run Cases A, B, C + amplitude sweep for one preset."""
    R_disc = cfg.bottom_disc_radius_effective
    D_disc_mm = R_disc * 2e3
    bottom_area = cfg.L ** 2
    disc_area = np.pi * R_disc ** 2
    disc_coverage = disc_area / bottom_area * 100

    cdir = OUTROOT / preset_label
    (cdir / "figs").mkdir(parents=True, exist_ok=True)
    (cdir / "csv").mkdir(parents=True, exist_ok=True)

    log(f"\n{'#' * 70}")
    log(f"  PRESET: {preset_label}  (L={cfg.L*1e3:.0f}mm, "
        f"D_disc={D_disc_mm:.1f}mm, R_disc={R_disc*1e3:.2f}mm)")
    log(f"  Disc coverage: {disc_coverage:.1f}% of bottom area")
    log(f"  Mesh: {cfg.mesh_nx}×{cfg.mesh_nx}×{cfg.mesh_nz}, "
        f"λ={cfg.wavelength*1e3:.2f}mm")
    log(f"{'#' * 70}")

    # Create mesh
    domain, facet_tags, _ = create_mesh(cfg, verbose=True)
    n_cells = domain.topology.index_map(domain.topology.dim).size_global

    z_H2 = cfg.H / 2.0
    pts_H2 = make_plane_grid(cfg, z_H2, N_PLANE)
    dx_m = cfg.L / (N_PLANE - 1)
    dy_m = dx_m
    mid = N_PLANE // 2

    case_metrics = []
    comp_metrics = []

    # ── Case A: Standing ──
    log(f"\n  ── Case A: Standing ──")
    p_sol_A = solve_helmholtz(
        domain, facet_tags, cfg, mode="standing",
        disc_robin=False, verbose=True, petsc_options=MUMPS_OPTS,
    )
    p_A = sample_pressure(p_sol_A.p_function, pts_H2).reshape(N_PLANE, N_PLANE)
    U_A, _, _ = compute_gorkov_grid(p_A, dx_m, dy_m, cfg)
    traps_A = find_traps_2d(U_A)
    cm_A = CaseMetrics(
        label=f"{preset_label}_A", mode="standing",
        max_p=p_sol_A.max_pressure,
        max_p_plane=float(np.nanmax(np.abs(p_A))),
        p_center=float(np.abs(p_A[mid, mid])),
        n_traps=len(traps_A),
        mean_trap_depth=float(np.mean([t[3] for t in traps_A])) if traps_A else 0,
        std_trap_depth=float(np.std([t[3] for t in traps_A])) if traps_A else 0,
        gorkov_min=float(np.nanmin(U_A)),
        gorkov_max=float(np.nanmax(U_A)),
        gorkov_range=float(np.nanmax(U_A) - np.nanmin(U_A)),
    )
    case_metrics.append(cm_A)
    log(f"    max|p|={cm_A.max_p:.2f} Pa, traps={cm_A.n_traps}, "
        f"ΔU={cm_A.gorkov_range:.2e} J")

    plot_pressure_field(p_A, cfg, f"{preset_label} — Standing |p|",
                        cdir / "figs" / "A_standing_abs_p.png")
    plot_gorkov_field(U_A, cfg, f"{preset_label} — Standing Gor'kov",
                      cdir / "figs" / "A_standing_gorkov.png")
    plot_trap_map(U_A, traps_A, cfg, f"{preset_label} — Standing traps",
                  cdir / "figs" / "A_standing_traps.png")
    plot_phase_field(p_A, cfg, f"{preset_label} — Standing arg(p)",
                     cdir / "figs" / "A_standing_phase.png")

    # ── Case B: Vortex ──
    log(f"\n  ── Case B: Vortex ──")
    p_sol_B = solve_helmholtz(
        domain, facet_tags, cfg, mode="vortex",
        disc_robin=False, verbose=True, petsc_options=MUMPS_OPTS,
    )
    p_B = sample_pressure(p_sol_B.p_function, pts_H2).reshape(N_PLANE, N_PLANE)
    U_B, _, _ = compute_gorkov_grid(p_B, dx_m, dy_m, cfg)
    traps_B = find_traps_2d(U_B)
    cm_B = CaseMetrics(
        label=f"{preset_label}_B", mode="vortex",
        max_p=p_sol_B.max_pressure,
        max_p_plane=float(np.nanmax(np.abs(p_B))),
        p_center=float(np.abs(p_B[mid, mid])),
        n_traps=len(traps_B),
        mean_trap_depth=float(np.mean([t[3] for t in traps_B])) if traps_B else 0,
        std_trap_depth=float(np.std([t[3] for t in traps_B])) if traps_B else 0,
        gorkov_min=float(np.nanmin(U_B)),
        gorkov_max=float(np.nanmax(U_B)),
        gorkov_range=float(np.nanmax(U_B) - np.nanmin(U_B)),
    )
    case_metrics.append(cm_B)
    log(f"    max|p|={cm_B.max_p:.2f} Pa, traps={cm_B.n_traps}, "
        f"ΔU={cm_B.gorkov_range:.2e} J")

    plot_pressure_field(p_B, cfg, f"{preset_label} — Vortex |p|",
                        cdir / "figs" / "B_vortex_abs_p.png")
    plot_gorkov_field(U_B, cfg, f"{preset_label} — Vortex Gor'kov",
                      cdir / "figs" / "B_vortex_gorkov.png")
    plot_phase_field(p_B, cfg, f"{preset_label} — Vortex arg(p)",
                     cdir / "figs" / "B_vortex_phase.png")

    # ── Case C: Combined (amplitude sweep) ──
    V0_mults = [1, 3, 6]
    for mult in V0_mults:
        tag = f"C_V0x{mult}"
        log(f"\n  ── Case C: Combined V₀×{mult} ──")
        cfg_m = replace(cfg, vortex_velocity_amplitude=mult * cfg.vortex_velocity_amplitude)
        p_sol_C = solve_helmholtz(
            domain, facet_tags, cfg_m, mode="combined",
            disc_robin=False, verbose=True, petsc_options=MUMPS_OPTS,
        )
        p_C = sample_pressure(p_sol_C.p_function, pts_H2).reshape(N_PLANE, N_PLANE)
        U_C, _, _ = compute_gorkov_grid(p_C, dx_m, dy_m, cfg_m)
        traps_C = find_traps_2d(U_C)

        cm_C = CaseMetrics(
            label=f"{preset_label}_{tag}", mode=f"combined_V0x{mult}",
            max_p=p_sol_C.max_pressure,
            max_p_plane=float(np.nanmax(np.abs(p_C))),
            p_center=float(np.abs(p_C[mid, mid])),
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

        comp = compute_comparison_metrics(
            U_A, U_C, p_A, p_C, cfg_m, preset_label, R_disc, mult
        )
        comp_metrics.append(comp)
        log(f"    authority(disc)={comp.authority_disc:.3f}, "
            f"locality={comp.locality_radius_mm:.1f}mm, "
            f"selectivity={comp.selectivity_ratio:.3f}")

        plot_pressure_field(p_C, cfg_m, f"{preset_label} — Combined V₀×{mult} |p|",
                            cdir / "figs" / f"{tag}_abs_p.png")
        plot_gorkov_field(U_C, cfg_m, f"{preset_label} — Combined V₀×{mult} Gor'kov",
                          cdir / "figs" / f"{tag}_gorkov.png")
        plot_trap_map(U_C, traps_C, cfg_m,
                      f"{preset_label} — Combined V₀×{mult} traps",
                      cdir / "figs" / f"{tag}_traps.png")
        plot_phase_field(p_C, cfg_m, f"{preset_label} — Combined V₀×{mult} arg(p)",
                         cdir / "figs" / f"{tag}_phase.png")

    # Save per-config CSV
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

    # Config JSON
    cfg.to_dict()  # ensure serializable
    with open(cdir / "config.json", "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    return {
        "label": preset_label,
        "D_disc_mm": D_disc_mm,
        "R_disc_mm": R_disc * 1e3,
        "disc_coverage_pct": disc_coverage,
        "n_cells": n_cells,
    }, case_metrics, comp_metrics


# ── Cross-Config Comparison ─────────────────────────────────────────

def generate_comparison(all_info, all_cms, all_comps):
    compdir = OUTROOT / "comparison"
    compdir.mkdir(parents=True, exist_ok=True)

    log(f"\n{'=' * 70}")
    log("  PHASE 0 COMPARISON TABLE")
    log(f"{'=' * 70}")

    header = (f"  {'Config':<10} {'D_disc':>7} {'Cov%':>6} "
              f"{'|p|_A':>8} {'|p|_B':>8} {'traps_A':>8} {'traps_C':>8} "
              f"{'auth':>8} {'local':>8} {'selec':>8}")
    log(header)
    log(f"  {'-' * 88}")

    summary_rows = []
    for info, cms, comps in zip(all_info, all_cms, all_comps):
        label = info["label"]
        cm_A = next(c for c in cms if c.mode == "standing")
        cm_B = next(c for c in cms if c.mode == "vortex")
        cm_C1 = next(c for c in cms if c.mode == "combined_V0x1")
        comp1 = next(c for c in comps if c.V0_mult == 1)

        row = {
            "config": label,
            "D_disc_mm": info["D_disc_mm"],
            "disc_coverage_pct": info["disc_coverage_pct"],
            "max_p_A": cm_A.max_p,
            "max_p_B": cm_B.max_p,
            "max_p_C": cm_C1.max_p,
            "traps_A": cm_A.n_traps,
            "traps_B": cm_B.n_traps,
            "traps_C": cm_C1.n_traps,
            "gorkov_range_A": cm_A.gorkov_range,
            "gorkov_range_C": cm_C1.gorkov_range,
            "authority_disc": comp1.authority_disc,
            "authority_p_disc": comp1.authority_p_disc,
            "locality_mm": comp1.locality_radius_mm,
            "selectivity": comp1.selectivity_ratio,
            "delta_n_traps": comp1.delta_n_traps,
        }
        summary_rows.append(row)

        log(f"  {label:<10} {info['D_disc_mm']:>5.1f}mm {info['disc_coverage_pct']:>5.1f}% "
            f"{cm_A.max_p:>7.1f} {cm_B.max_p:>7.1f} "
            f"{cm_A.n_traps:>8} {cm_C1.n_traps:>8} "
            f"{comp1.authority_disc:>8.3f} {comp1.locality_radius_mm:>6.1f}mm "
            f"{comp1.selectivity_ratio:>8.3f}")

    # CSV
    with open(compdir / "decision_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        w.writeheader()
        for row in summary_rows:
            w.writerow(row)

    # ── Comparison bar charts ──
    labels = [r["config"] for r in summary_rows]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].bar(x, [r["authority_disc"] for r in summary_rows], 0.6, color="steelblue")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20)
    axes[0].set_ylabel("Authority (avg |ΔU/U| in disc)")
    axes[0].set_title("Vortex Authority at V₀×1")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x, [r["locality_mm"] for r in summary_rows], 0.6, color="darkorange")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20)
    axes[1].set_ylabel("Locality radius (mm)")
    axes[1].set_title("Vortex Locality (10% threshold)")
    axes[1].grid(axis="y", alpha=0.3)

    axes[2].bar(x, [r["selectivity"] for r in summary_rows], 0.6, color="seagreen")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=20)
    axes[2].set_ylabel("Selectivity ratio")
    axes[2].set_title("Selectivity (disc-centre / median)")
    axes[2].grid(axis="y", alpha=0.3)

    fig.suptitle("Phase 0 — Disc Diameter Comparison (L=10 mm)", fontsize=14)
    fig.tight_layout()
    fig.savefig(compdir / "authority_locality_selectivity.png", dpi=200)
    plt.close(fig)

    # ── Trap count bar chart ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    traps_A = [r["traps_A"] for r in summary_rows]
    traps_C = [r["traps_C"] for r in summary_rows]
    axes[0].bar(x - 0.15, traps_A, 0.3, label="Standing", color="steelblue")
    axes[0].bar(x + 0.15, traps_C, 0.3, label="Combined V₀×1", color="coral")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20)
    axes[0].set_ylabel("Trap count")
    axes[0].set_title("Trap Count: Standing vs Combined")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    gorkov_A = [r["gorkov_range_A"] for r in summary_rows]
    gorkov_C = [r["gorkov_range_C"] for r in summary_rows]
    axes[1].bar(x - 0.15, gorkov_A, 0.3, label="Standing", color="steelblue")
    axes[1].bar(x + 0.15, gorkov_C, 0.3, label="Combined V₀×1", color="coral")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20)
    axes[1].set_ylabel("Gor'kov range ΔU (J)")
    axes[1].set_title("Trap Depth (Gor'kov Range)")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(compdir / "trap_count_depth.png", dpi=200)
    plt.close(fig)

    # ── Amplitude sweep ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for info, comps in zip(all_info, all_comps):
        label = info["label"]
        mults = [c.V0_mult for c in comps]
        auths = [c.authority_disc for c in comps]
        locs = [c.locality_radius_mm for c in comps]
        sels = [c.selectivity_ratio for c in comps]
        axes[0].plot(mults, auths, "o-", label=label, linewidth=1.5)
        axes[1].plot(mults, locs, "s-", label=label, linewidth=1.5)
        axes[2].plot(mults, sels, "^-", label=label, linewidth=1.5)

    for ax, ylabel, title in zip(
        axes,
        ["Authority", "Locality (mm)", "Selectivity"],
        ["Authority vs V₀ multiplier", "Locality vs V₀ multiplier",
         "Selectivity vs V₀ multiplier"],
    ):
        ax.set_xlabel("V₀ multiplier")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Phase 0 — Amplitude Sweep (L=10 mm)", fontsize=14)
    fig.tight_layout()
    fig.savefig(compdir / "amplitude_sweep_curves.png", dpi=200)
    plt.close(fig)

    # ── Disc coverage acceptance check ──
    log(f"\n{'=' * 70}")
    log("  ACCEPTANCE CHECKS")
    log(f"{'=' * 70}")
    for row in summary_rows:
        cov = row["disc_coverage_pct"]
        ok = cov < 15
        log(f"  {row['config']}: disc covers {cov:.1f}% of bottom "
            f"{'[PASS < 15%]' if ok else '[FAIL >= 15%]'}")
    log(f"{'=' * 70}")

    return summary_rows


# ── Main ────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    OUTROOT.mkdir(parents=True, exist_ok=True)

    log(f"\n{'#' * 70}")
    log(f"  PHASE 0 — 10×10 mm BASELINE + DISC DIAMETER SWEEP")
    log(f"  Output: {OUTROOT.relative_to(ROOT)}")
    log(f"  Presets: {list(PHASE0_PRESETS.keys())}")
    log(f"{'#' * 70}")

    all_info = []
    all_cms = []
    all_comps = []

    for preset_label, cfg_factory in PHASE0_PRESETS.items():
        try:
            cfg = cfg_factory()
            info, cms, comps = process_preset(preset_label, cfg)
            all_info.append(info)
            all_cms.append(cms)
            all_comps.append(comps)
        except Exception as e:
            log(f"\n  *** FAILED: {preset_label}: {e} ***")
            import traceback
            traceback.print_exc()

    if all_info:
        summary = generate_comparison(all_info, all_cms, all_comps)
        # Full JSON
        with open(OUTROOT / "all_results.json", "w") as f:
            json.dump({
                "phase": "phase0_baseline",
                "presets": [i["label"] for i in all_info],
                "info": all_info,
                "summary": summary,
            }, f, indent=2, default=str)

    elapsed = time.time() - t0
    log(f"\n{'#' * 70}")
    log(f"  PHASE 0 COMPLETE — {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    log(f"  Output: {OUTROOT.relative_to(ROOT)}")
    log(f"{'#' * 70}")


if __name__ == "__main__":
    main()
