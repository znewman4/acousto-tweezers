#!/usr/bin/env python3
"""
Phase 2 — Wall + Bottom Impedance Sweep (Lossy Cavity).

Goal: Introduce damping on cavity walls to reduce global recirculation
and make results more physically plausible.

Sweeps side-wall relative impedance Z_rel ∈ {inf (rigid), 50, 10, 3, 1}
using the Phase 0 baseline L10_D03 (L=10 mm, D=3 mm) as reference.

For each Z_rel value runs Cases A (standing), B (vortex), C (combined)
and computes:
  - max|p|, trap count, Gor'kov range
  - authority, locality, selectivity
  - "locality metric" improvement vs rigid baseline

Usage:
    micromamba run -n acousto-complex python scripts/experiments/impedance_sweep.py
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
    ShallowDishConfig, get_L10_D03_config,
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
OUTROOT = ROOT / "results" / f"phase2_impedance_{STAMP}"
N_PLANE = 201

MUMPS_OPTS = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}

# Sweep values: Z_rel = Z_wall / (ρc).
# None → rigid (infinite impedance / zero admittance)
ZREL_VALUES = [None, 50, 10, 3, 1]
ZREL_LABELS = ["rigid", "Z50", "Z10", "Z03", "Z01"]


def log(msg="", end="\n"):
    if rank == 0:
        print(msg, end=end, flush=True)


# ── Sampling Utilities (identical to Phase 0) ──────────────────────

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


# ── Metrics ─────────────────────────────────────────────────────────

@dataclass
class CaseMetrics:
    zrel_label: str
    mode: str
    max_p: float
    max_p_plane: float
    n_traps: int
    mean_trap_depth: float
    gorkov_range: float


@dataclass
class ComparisonMetrics:
    zrel_label: str
    authority_disc: float
    locality_radius_mm: float
    selectivity_ratio: float
    n_traps_combined: int
    n_traps_standing: int
    delta_n_traps: int


def compute_comparison_metrics(U_stand, U_comb, p_stand_2d, p_comb_2d,
                                cfg, zrel_label, R_disc):
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
        zrel_label=zrel_label,
        authority_disc=auth_disc,
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


# ── Main Processing ─────────────────────────────────────────────────

def process_impedance(zrel, zrel_label, base_cfg):
    """Run A/B/C for one impedance value."""
    cfg = replace(base_cfg, wall_impedance_Zrel=zrel)

    cdir = OUTROOT / zrel_label
    (cdir / "figs").mkdir(parents=True, exist_ok=True)
    (cdir / "csv").mkdir(parents=True, exist_ok=True)

    R_disc = cfg.bottom_disc_radius_effective
    zrel_str = f"Z_rel={zrel}" if zrel is not None else "rigid (inf)"

    log(f"\n{'#' * 70}")
    log(f"  IMPEDANCE: {zrel_label}  ({zrel_str})")
    log(f"{'#' * 70}")

    domain, facet_tags, _ = create_mesh(cfg, verbose=True)

    z_H2 = cfg.H / 2.0
    pts_H2 = make_plane_grid(cfg, z_H2, N_PLANE)
    dx_m = cfg.L / (N_PLANE - 1)
    dy_m = dx_m
    mid = N_PLANE // 2

    case_metrics = []

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
        zrel_label=zrel_label, mode="standing",
        max_p=p_sol_A.max_pressure,
        max_p_plane=float(np.nanmax(np.abs(p_A))),
        n_traps=len(traps_A),
        mean_trap_depth=float(np.mean([t[3] for t in traps_A])) if traps_A else 0,
        gorkov_range=float(np.nanmax(U_A) - np.nanmin(U_A)),
    )
    case_metrics.append(cm_A)
    log(f"    max|p|={cm_A.max_p:.2f} Pa, traps={cm_A.n_traps}, "
        f"ΔU={cm_A.gorkov_range:.2e} J")

    plot_pressure_field(p_A, cfg, f"{zrel_label} — Standing |p|",
                        cdir / "figs" / "A_standing_abs_p.png")
    plot_gorkov_field(U_A, cfg, f"{zrel_label} — Standing Gor'kov",
                      cdir / "figs" / "A_standing_gorkov.png")

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
        zrel_label=zrel_label, mode="vortex",
        max_p=p_sol_B.max_pressure,
        max_p_plane=float(np.nanmax(np.abs(p_B))),
        n_traps=len(traps_B),
        mean_trap_depth=float(np.mean([t[3] for t in traps_B])) if traps_B else 0,
        gorkov_range=float(np.nanmax(U_B) - np.nanmin(U_B)),
    )
    case_metrics.append(cm_B)
    log(f"    max|p|={cm_B.max_p:.2f} Pa, traps={cm_B.n_traps}, "
        f"ΔU={cm_B.gorkov_range:.2e} J")

    plot_pressure_field(p_B, cfg, f"{zrel_label} — Vortex |p|",
                        cdir / "figs" / "B_vortex_abs_p.png")

    # ── Case C: Combined ──
    log(f"\n  ── Case C: Combined ──")
    p_sol_C = solve_helmholtz(
        domain, facet_tags, cfg, mode="combined",
        disc_robin=False, verbose=True, petsc_options=MUMPS_OPTS,
    )
    p_C = sample_pressure(p_sol_C.p_function, pts_H2).reshape(N_PLANE, N_PLANE)
    U_C, _, _ = compute_gorkov_grid(p_C, dx_m, dy_m, cfg)
    traps_C = find_traps_2d(U_C)
    cm_C = CaseMetrics(
        zrel_label=zrel_label, mode="combined",
        max_p=p_sol_C.max_pressure,
        max_p_plane=float(np.nanmax(np.abs(p_C))),
        n_traps=len(traps_C),
        mean_trap_depth=float(np.mean([t[3] for t in traps_C])) if traps_C else 0,
        gorkov_range=float(np.nanmax(U_C) - np.nanmin(U_C)),
    )
    case_metrics.append(cm_C)
    log(f"    max|p|={cm_C.max_p:.2f} Pa, traps={cm_C.n_traps}, "
        f"ΔU={cm_C.gorkov_range:.2e} J")

    plot_pressure_field(p_C, cfg, f"{zrel_label} — Combined |p|",
                        cdir / "figs" / "C_combined_abs_p.png")
    plot_gorkov_field(U_C, cfg, f"{zrel_label} — Combined Gor'kov",
                      cdir / "figs" / "C_combined_gorkov.png")

    # Comparison
    comp = compute_comparison_metrics(
        U_A, U_C, p_A, p_C, cfg, zrel_label, R_disc
    )
    log(f"    authority={comp.authority_disc:.3f}, "
        f"locality={comp.locality_radius_mm:.1f}mm, "
        f"selectivity={comp.selectivity_ratio:.3f}")

    # Save CSV
    with open(cdir / "csv" / "case_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=case_metrics[0].__dict__.keys())
        w.writeheader()
        for cm in case_metrics:
            w.writerow(cm.__dict__)

    with open(cdir / "config.json", "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    return case_metrics, comp


# ── Comparison Figures ──────────────────────────────────────────────

def generate_comparison(all_cms, all_comps):
    compdir = OUTROOT / "comparison"
    compdir.mkdir(parents=True, exist_ok=True)

    log(f"\n{'=' * 70}")
    log("  IMPEDANCE SWEEP COMPARISON TABLE")
    log(f"{'=' * 70}")

    header = (f"  {'Z_rel':<8} {'|p|_A':>8} {'|p|_B':>8} {'|p|_C':>8} "
              f"{'traps_A':>8} {'traps_C':>8} "
              f"{'ΔU_A':>10} {'ΔU_C':>10} "
              f"{'auth':>8} {'local':>8} {'selec':>8}")
    log(header)
    log(f"  {'-' * 100}")

    summary_rows = []
    for cms, comp in zip(all_cms, all_comps):
        cm_A = next(c for c in cms if c.mode == "standing")
        cm_B = next(c for c in cms if c.mode == "vortex")
        cm_C = next(c for c in cms if c.mode == "combined")

        row = {
            "zrel_label": comp.zrel_label,
            "max_p_A": cm_A.max_p,
            "max_p_B": cm_B.max_p,
            "max_p_C": cm_C.max_p,
            "traps_A": cm_A.n_traps,
            "traps_B": cm_B.n_traps,
            "traps_C": cm_C.n_traps,
            "gorkov_range_A": cm_A.gorkov_range,
            "gorkov_range_C": cm_C.gorkov_range,
            "authority_disc": comp.authority_disc,
            "locality_mm": comp.locality_radius_mm,
            "selectivity": comp.selectivity_ratio,
            "delta_n_traps": comp.delta_n_traps,
        }
        summary_rows.append(row)

        log(f"  {comp.zrel_label:<8} {cm_A.max_p:>7.1f} {cm_B.max_p:>7.1f} "
            f"{cm_C.max_p:>7.1f} "
            f"{cm_A.n_traps:>8} {cm_C.n_traps:>8} "
            f"{cm_A.gorkov_range:>10.2e} {cm_C.gorkov_range:>10.2e} "
            f"{comp.authority_disc:>8.3f} {comp.locality_radius_mm:>6.1f}mm "
            f"{comp.selectivity_ratio:>8.3f}")

    # Save CSV
    with open(compdir / "decision_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    # ── Comparison Plots ──
    labels = [r["zrel_label"] for r in summary_rows]
    x = np.arange(len(labels))

    # 1) |p| vs Z_rel
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    colors_mode = {"A": ("Standing", "steelblue"), "B": ("Vortex", "darkorange"), "C": ("Combined", "seagreen")}
    for key, (mode_name, clr) in colors_mode.items():
        k = f"max_p_{key}"
        vals = [r[k] for r in summary_rows]
        axes[0].plot(x, vals, "o-", color=clr, label=mode_name, linewidth=1.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20)
    axes[0].set_ylabel("max|p| (Pa)")
    axes[0].set_title("Peak Pressure vs Wall Impedance")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)

    # 2) Trap depth
    axes[1].plot(x, [r["gorkov_range_A"] for r in summary_rows],
                 "o-", color="steelblue", label="Standing", linewidth=1.5)
    axes[1].plot(x, [r["gorkov_range_C"] for r in summary_rows],
                 "s-", color="seagreen", label="Combined", linewidth=1.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20)
    axes[1].set_ylabel("Gor'kov range ΔU (J)")
    axes[1].set_title("Trap Depth vs Wall Impedance")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.3)

    # 3) Trap count
    axes[2].bar(x - 0.15, [r["traps_A"] for r in summary_rows],
                0.3, label="Standing", color="steelblue")
    axes[2].bar(x + 0.15, [r["traps_C"] for r in summary_rows],
                0.3, label="Combined", color="coral")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=20)
    axes[2].set_ylabel("Trap count")
    axes[2].set_title("Trap Count vs Wall Impedance")
    axes[2].legend(fontsize=8)
    axes[2].grid(axis="y", alpha=0.3)

    fig.suptitle("Phase 2 — Impedance Sweep (L10_D03)", fontsize=14)
    fig.tight_layout()
    fig.savefig(compdir / "impedance_sweep_metrics.png", dpi=200)
    plt.close(fig)

    # 4) Authority, Locality, Selectivity
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].plot(x, [r["authority_disc"] for r in summary_rows],
                 "o-", color="steelblue", linewidth=2)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20)
    axes[0].set_ylabel("Authority")
    axes[0].set_title("Vortex Authority vs Z_rel")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].plot(x, [r["locality_mm"] for r in summary_rows],
                 "s-", color="darkorange", linewidth=2)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20)
    axes[1].set_ylabel("Locality (mm)")
    axes[1].set_title("Locality Radius vs Z_rel")
    axes[1].grid(axis="y", alpha=0.3)

    axes[2].plot(x, [r["selectivity"] for r in summary_rows],
                 "^-", color="seagreen", linewidth=2)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=20)
    axes[2].set_ylabel("Selectivity")
    axes[2].set_title("Selectivity vs Z_rel")
    axes[2].grid(axis="y", alpha=0.3)

    fig.suptitle("Phase 2 — Authority / Locality / Selectivity", fontsize=14)
    fig.tight_layout()
    fig.savefig(compdir / "authority_locality_selectivity.png", dpi=200)
    plt.close(fig)

    # Locality improvement vs rigid
    if len(summary_rows) > 1:
        rigid_locality = summary_rows[0]["locality_mm"]
        improvements = []
        for r in summary_rows:
            imp = (rigid_locality - r["locality_mm"]) / rigid_locality * 100
            improvements.append(imp)
        log(f"\n  Locality improvement vs rigid:")
        for r, imp in zip(summary_rows, improvements):
            log(f"    {r['zrel_label']}: {imp:+.1f}%")

    return summary_rows


# ── Main ────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    OUTROOT.mkdir(parents=True, exist_ok=True)

    base_cfg = get_L10_D03_config()

    log(f"\n{'#' * 70}")
    log(f"  PHASE 2 — WALL IMPEDANCE SWEEP")
    log(f"  Baseline: L10_D03 (L=10mm, D=3mm)")
    log(f"  Z_rel sweep: {ZREL_LABELS}")
    log(f"  Output: {OUTROOT.relative_to(ROOT)}")
    log(f"{'#' * 70}")

    all_cms = []
    all_comps = []

    for zrel, zrel_label in zip(ZREL_VALUES, ZREL_LABELS):
        try:
            cms, comp = process_impedance(zrel, zrel_label, base_cfg)
            all_cms.append(cms)
            all_comps.append(comp)
        except Exception as e:
            log(f"\n  *** FAILED: {zrel_label}: {e} ***")
            import traceback
            traceback.print_exc()

    if all_cms:
        summary = generate_comparison(all_cms, all_comps)
        with open(OUTROOT / "all_results.json", "w") as f:
            json.dump({
                "phase": "phase2_impedance",
                "baseline": "L10_D03",
                "zrel_values": [str(z) for z in ZREL_VALUES],
                "summary": summary,
            }, f, indent=2, default=str)

    elapsed = time.time() - t0
    log(f"\n{'#' * 70}")
    log(f"  PHASE 2 COMPLETE — {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    log(f"  Output: {OUTROOT.relative_to(ROOT)}")
    log(f"{'#' * 70}")


if __name__ == "__main__":
    main()
