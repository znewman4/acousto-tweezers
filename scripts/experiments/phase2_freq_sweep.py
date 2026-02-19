#!/usr/bin/env python3
"""
Phase 2 — Frequency Sweeps & Resonance Mapping.

D2.1  Standing-wave frequency sweep: identify cavity resonances that produce
      useful trap lattices (500 kHz → 2 MHz for 10 mm; 500→800 kHz for 30 mm).
D2.2  Two-frequency vortex superposition: fix f_stand at best mode, sweep
      f_vortex independently, combine U_total = U_stand + U_vortex.

Uses the best configs from Phase 1:
  - 10 mm dish, D_piezo = 10 mm  (L10_D10)
  - 30 mm dish, D_piezo = 10 mm  (L30_D10)

All solves use disc_robin=False (Attempt 4 fix).

Usage:
    micromamba run -n acousto-complex python scripts/experiments/phase2_freq_sweep.py
"""
from __future__ import annotations

import sys, os, csv, json, time, gc
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, replace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from scipy.ndimage import minimum_filter, maximum_filter

from mpi4py import MPI
from dolfinx import fem
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells

from acoustweezers.experiments.shallow_square_dish.config import ShallowDishConfig
from acoustweezers.experiments.shallow_square_dish.solve_pressure import (
    create_mesh,
    solve_helmholtz,
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
OUTROOT = ROOT / "results" / f"phase2_freq_sweep_{STAMP}"
N_PLANE = 201       # sampling grid per side

MAX_NX = 50         # max mesh elements per lateral side (memory cap)
MIN_EPW = 3         # minimum acceptable elements per wavelength (P2)

MUMPS_OPTS = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}


def log(msg="", end="\n"):
    if rank == 0:
        print(msg, end=end, flush=True)


# ── Dish / frequency matrix ────────────────────────────────────────

@dataclass
class DishSpec:
    """One dish + piezo geometry."""
    tag: str
    L_mm: float
    D_piezo_mm: float
    R_disc_m: float
    freqs_khz: list          # frequency list [kHz]
    base_epw: int = 8        # target elements per wavelength (capped by MAX_NX)


DISHES = [
    DishSpec(
        tag="L10_D10",
        L_mm=10, D_piezo_mm=10, R_disc_m=5.0e-3,
        freqs_khz=[500, 600, 700, 800, 900, 1000, 1200, 1500, 2000],
        base_epw=8,
    ),
    DishSpec(
        tag="L30_D10",
        L_mm=30, D_piezo_mm=10, R_disc_m=5.0e-3,
        freqs_khz=[500, 600, 700, 800],
        base_epw=6,
    ),
]


def compute_epw(L_m: float, freq_hz: float, target_epw: int) -> int:
    """
    Return the actual elements_per_wavelength that keeps mesh_nx <= MAX_NX.

    With the config formula:  mesh_nx = max(20, int(L / wavelength * epw))
    """
    lam = 1484.0 / freq_hz
    ideal_nx = int(L_m / lam * target_epw)
    if ideal_nx <= MAX_NX:
        return target_epw
    # reverse-engineer epw to hit MAX_NX
    cap_epw = int(MAX_NX * lam / L_m)
    return max(MIN_EPW, cap_epw)


def make_config(dish: DishSpec, freq_hz: float) -> ShallowDishConfig:
    """Create ShallowDishConfig for a specific dish & frequency."""
    L_m = dish.L_mm * 1e-3
    epw = compute_epw(L_m, freq_hz, dish.base_epw)
    return ShallowDishConfig(
        L=L_m, H=1e-3,
        frequency_hz=freq_hz,
        elements_per_wavelength=epw,
        min_elements_z=8,
        rho=997.0, c=1484.0, mu=1.002e-3,
        vortex_velocity_amplitude=10e-6,
        standing_velocity_amplitude=10e-6,
        vortex_topological_charge=1,
        vortex_aperture_radius=dish.R_disc_m,
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


# ── Sampling & Gor'kov utilities (reused from Phase 1) ─────────────

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
    """Gor'kov potential from sampled 2D complex pressure via finite differences."""
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
    """Find local minima of Gor'kov potential on a 2D grid."""
    U_clean = np.copy(U)
    nan_mask = np.isnan(U_clean)
    U_clean[nan_mask] = np.nanmax(U) if np.any(~nan_mask) else 0
    fp = 2 * min_size + 1
    local_min = (U_clean == minimum_filter(U_clean, size=fp)) & ~nan_mask
    local_max_map = maximum_filter(U_clean, size=fp)
    rows, cols = np.where(local_min)
    traps = []
    for r, c_ in zip(rows, cols):
        depth = local_max_map[r, c_] - U_clean[r, c_]
        traps.append((int(r), int(c_), float(U_clean[r, c_]), float(depth)))
    return traps


# ── Plotting helpers ────────────────────────────────────────────────

def disc_circle(cfg, color="white", ls="--"):
    cx = cfg.L / 2 * 1e3
    cy = cfg.L / 2 * 1e3
    R = cfg.bottom_disc_radius_effective * 1e3
    return Circle((cx, cy), R, fill=False, edgecolor=color,
                  linewidth=1.2, linestyle=ls)


def plot_pressure(p_2d, cfg, title, path):
    N = p_2d.shape[0]
    xs = np.linspace(0, cfg.L * 1e3, N)
    X, Y = np.meshgrid(xs, xs)
    data = np.abs(p_2d)
    fig, ax = plt.subplots(figsize=(6, 5))
    cf = ax.contourf(X, Y, data, levels=20, cmap="jet")
    ax.contour(X, Y, data, levels=20, colors="k", linewidths=0.2, alpha=0.4)
    fig.colorbar(cf, ax=ax, shrink=0.82, pad=0.03, label="|p| (Pa)")
    ax.add_patch(disc_circle(cfg))
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal"); fig.tight_layout()
    fig.savefig(path, dpi=150); plt.close(fig)


def plot_gorkov(U_2d, cfg, title, path):
    N = U_2d.shape[0]
    xs = np.linspace(0, cfg.L * 1e3, N)
    X, Y = np.meshgrid(xs, xs)
    vmax = max(abs(np.nanmin(U_2d)), abs(np.nanmax(U_2d)))
    if vmax < 1e-40:
        vmax = 1e-30
    fig, ax = plt.subplots(figsize=(6, 5))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cf = ax.contourf(X, Y, U_2d, levels=20, cmap="RdBu_r", norm=norm)
    fig.colorbar(cf, ax=ax, shrink=0.82, pad=0.03, label="U (J)")
    ax.add_patch(disc_circle(cfg, "black"))
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal"); fig.tight_layout()
    fig.savefig(path, dpi=150); plt.close(fig)


def plot_traps(U_2d, traps, cfg, title, path):
    N = U_2d.shape[0]
    xs = np.linspace(0, cfg.L * 1e3, N)
    X, Y = np.meshgrid(xs, xs)
    dx_mm = xs[1] - xs[0]
    fig, ax = plt.subplots(figsize=(6, 5))
    cf = ax.contourf(X, Y, U_2d, levels=20, cmap="viridis")
    fig.colorbar(cf, ax=ax, shrink=0.82, pad=0.03, label="U (J)")
    if traps:
        tx = [t[1] * dx_mm for t in traps]
        ty = [t[0] * dx_mm for t in traps]
        ax.scatter(tx, ty, c="red", s=14, marker="x", linewidths=0.7, zorder=5)
    ax.add_patch(disc_circle(cfg, "white"))
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    ax.set_title(f"{title}  ({len(traps)} traps)", fontsize=10)
    ax.set_aspect("equal"); fig.tight_layout()
    fig.savefig(path, dpi=150); plt.close(fig)


def plot_phase(p_2d, cfg, title, path):
    N = p_2d.shape[0]
    ext = [0, cfg.L * 1e3, 0, cfg.L * 1e3]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(np.angle(p_2d), origin="lower", extent=ext,
                   cmap="twilight_shifted", vmin=-np.pi, vmax=np.pi, aspect="equal")
    cb = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.03, label="arg(p) (rad)")
    cb.set_ticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    cb.set_ticklabels(["-π", "-π/2", "0", "π/2", "π"])
    ax.add_patch(disc_circle(cfg, "black"))
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150); plt.close(fig)


# ── D2.1 — Standing-wave frequency sweep ───────────────────────────

@dataclass
class FreqResult:
    """Metrics for one standing-wave solve at a given frequency."""
    tag: str
    freq_khz: float
    wavelength_mm: float
    mesh_nx: int
    mesh_nz: int
    epw_actual: float      # actual elements per wavelength
    n_cells: int
    max_p: float           # global max|p|
    max_p_plane: float     # max|p| at z=H/2 sampling plane
    n_traps: int
    mean_trap_depth: float
    max_trap_depth: float
    gorkov_range: float    # max(U) - min(U)
    lattice_spacing_mm: float  # median inter-trap distance
    solve_time_s: float


def run_d21_sweep(dish: DishSpec) -> list[FreqResult]:
    """D2.1: Sweep standing-wave frequency for one dish geometry."""
    results = []
    ddir = OUTROOT / dish.tag / "D2.1_standing_sweep"
    (ddir / "figs").mkdir(parents=True, exist_ok=True)

    log(f"\n{'='*70}")
    log(f"  D2.1 — Standing-wave frequency sweep: {dish.tag}")
    log(f"  Dish {dish.L_mm}×{dish.L_mm}×1 mm, D_piezo={dish.D_piezo_mm} mm")
    log(f"  Frequencies: {dish.freqs_khz} kHz")
    log(f"{'='*70}")

    for f_khz in dish.freqs_khz:
        freq_hz = f_khz * 1e3
        cfg = make_config(dish, freq_hz)
        lam_mm = cfg.wavelength * 1e3
        epw_act = cfg.mesh_nx * lam_mm / dish.L_mm

        log(f"\n  ── f = {f_khz} kHz  (λ={lam_mm:.2f} mm, "
            f"mesh {cfg.mesh_nx}×{cfg.mesh_nx}×{cfg.mesh_nz}, "
            f"epw={epw_act:.1f}) ──")

        t0 = time.time()

        # Create mesh
        domain, facet_tags, _ = create_mesh(cfg, verbose=False)
        n_cells = domain.topology.index_map(domain.topology.dim).size_global

        # Solve standing-only
        p_sol = solve_helmholtz(domain, facet_tags, cfg, mode="standing",
                                disc_robin=False, verbose=False,
                                petsc_options=MUMPS_OPTS)

        # Sample on z = H/2 plane
        pts = make_plane_grid(cfg, cfg.H / 2.0, N_PLANE)
        dx_m = cfg.L / (N_PLANE - 1)
        p_vals = sample_pressure(p_sol.p_function, pts)
        p_2d = p_vals.reshape(N_PLANE, N_PLANE)

        # Gor'kov & traps
        U, _, _ = compute_gorkov_grid(p_2d, dx_m, dx_m, cfg)
        traps = find_traps_2d(U)

        solve_t = time.time() - t0
        depths = [t[3] for t in traps] if traps else [0]

        # Lattice spacing: median nearest-neighbour distance between traps
        spacing_mm = 0.0
        if len(traps) >= 2:
            tx = np.array([t[1] * dx_m * 1e3 for t in traps])
            ty = np.array([t[0] * dx_m * 1e3 for t in traps])
            from scipy.spatial import cKDTree
            tree = cKDTree(np.column_stack([tx, ty]))
            dists, _ = tree.query(np.column_stack([tx, ty]), k=2)
            spacing_mm = float(np.median(dists[:, 1]))

        fr = FreqResult(
            tag=dish.tag, freq_khz=f_khz, wavelength_mm=lam_mm,
            mesh_nx=cfg.mesh_nx, mesh_nz=cfg.mesh_nz , epw_actual=epw_act,
            n_cells=n_cells,
            max_p=p_sol.max_pressure,
            max_p_plane=float(np.nanmax(np.abs(p_2d))),
            n_traps=len(traps),
            mean_trap_depth=float(np.mean(depths)),
            max_trap_depth=float(np.max(depths)),
            gorkov_range=float(np.nanmax(U) - np.nanmin(U)),
            lattice_spacing_mm=spacing_mm,
            solve_time_s=solve_t,
        )
        results.append(fr)

        log(f"    max|p|={fr.max_p:.2f} Pa, plane max={fr.max_p_plane:.2f} Pa, "
            f"traps={fr.n_traps}, ΔU={fr.gorkov_range:.2e} J, "
            f"spacing={fr.lattice_spacing_mm:.2f} mm, {fr.solve_time_s:.1f}s")

        # Per-frequency figures
        ftag = f"f{f_khz:04d}kHz"
        plot_pressure(p_2d, cfg,
                      f"{dish.tag} Standing {f_khz} kHz — |p|  (max {fr.max_p_plane:.1f} Pa)",
                      ddir / "figs" / f"{ftag}_abs_p.png")
        plot_gorkov(U, cfg,
                    f"{dish.tag} Standing {f_khz} kHz — Gor'kov  (ΔU={fr.gorkov_range:.2e})",
                    ddir / "figs" / f"{ftag}_gorkov.png")
        plot_traps(U, traps, cfg,
                   f"{dish.tag} Standing {f_khz} kHz",
                   ddir / "figs" / f"{ftag}_traps.png")
        plot_phase(p_2d, cfg,
                   f"{dish.tag} Standing {f_khz} kHz — arg(p)",
                   ddir / "figs" / f"{ftag}_phase.png")

        # Free solve memory
        del domain, facet_tags, p_sol, p_vals, p_2d, U, traps
        gc.collect()

    # Save CSV
    with open(ddir / "freq_sweep_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(FreqResult.__dataclass_fields__.keys()))
        w.writeheader()
        for fr in results:
            w.writerow(fr.__dict__)

    return results


# ── D2.1 summary plots ─────────────────────────────────────────────

def plot_d21_summary(all_results: dict[str, list[FreqResult]]):
    """Generate comparison figures across dishes for D2.1."""
    sdir = OUTROOT / "D2.1_summary"
    sdir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))

    for tag, results in all_results.items():
        fs = [r.freq_khz for r in results]
        max_ps = [r.max_p_plane for r in results]
        n_traps = [r.n_traps for r in results]
        depths = [r.mean_trap_depth for r in results]
        max_depths = [r.max_trap_depth for r in results]
        ranges = [r.gorkov_range for r in results]
        spacings = [r.lattice_spacing_mm for r in results]

        axes[0, 0].plot(fs, max_ps, "o-", label=tag, linewidth=1.5, markersize=5)
        axes[0, 1].plot(fs, n_traps, "s-", label=tag, linewidth=1.5, markersize=5)
        axes[0, 2].plot(fs, depths, "^-", label=tag, linewidth=1.5, markersize=5)
        axes[1, 0].plot(fs, max_depths, "D-", label=tag, linewidth=1.5, markersize=5)
        axes[1, 1].plot(fs, ranges, "v-", label=tag, linewidth=1.5, markersize=5)
        axes[1, 2].plot(fs, spacings, "p-", label=tag, linewidth=1.5, markersize=5)

    titles = [
        "max|p| at z=H/2 (Pa)", "Trap count", "Mean trap depth (J)",
        "Max trap depth (J)", "Gor'kov range ΔU (J)", "Lattice spacing (mm)",
    ]
    ylabels = titles
    for ax, t, yl in zip(axes.flat, titles, ylabels):
        ax.set_xlabel("Frequency (kHz)")
        ax.set_ylabel(yl)
        ax.set_title(t)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig.suptitle("D2.1 — Standing-Wave Frequency Sweep", fontsize=14)
    fig.tight_layout()
    fig.savefig(sdir / "freq_sweep_summary.png", dpi=200)
    plt.close(fig)

    # Also plot λ/2 vs measured spacing
    fig, ax = plt.subplots(figsize=(7, 5))
    for tag, results in all_results.items():
        fs = [r.freq_khz for r in results]
        spacings = [r.lattice_spacing_mm for r in results]
        half_lam = [r.wavelength_mm / 2 for r in results]
        ax.plot(fs, spacings, "o-", label=f"{tag} measured", linewidth=1.5)
        ax.plot(fs, half_lam, "--", label=f"{tag} λ/2", linewidth=1, alpha=0.7)
    ax.set_xlabel("Frequency (kHz)"); ax.set_ylabel("Spacing (mm)")
    ax.set_title("Trap Lattice Spacing vs λ/2 Prediction")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(sdir / "spacing_vs_half_lambda.png", dpi=200)
    plt.close(fig)

    log(f"  D2.1 summary figures saved to {sdir.relative_to(ROOT)}")


# ── D2.2 — Two-frequency vortex superposition ──────────────────────

@dataclass
class TwoFreqResult:
    """Metrics for one two-frequency combination."""
    tag: str
    f_stand_khz: float
    f_vortex_khz: float
    max_p_stand: float
    max_p_vortex: float
    n_traps_stand: int
    n_traps_combined: int
    gorkov_range_stand: float
    gorkov_range_combined: float
    mean_depth_stand: float
    mean_depth_combined: float
    centre_trap_depth_combined: float
    barrier_reduction_pct: float     # how much the nearest-to-centre barrier shrinks
    solve_time_s: float


def run_d22_sweep(dish: DishSpec, d21_results: list[FreqResult]) -> list[TwoFreqResult]:
    """
    D2.2: Fix f_stand at the frequency that gave the deepest traps in D2.1,
    then sweep f_vortex and combine the Gor'kov potentials.

    Physical basis: if f_stand ≠ f_vortex the cross-terms time-average to zero,
    so U_total = U_stand(f_stand) + U_vortex(f_vortex).
    """
    ddir = OUTROOT / dish.tag / "D2.2_two_freq"
    (ddir / "figs").mkdir(parents=True, exist_ok=True)

    # Pick best f_stand (maximum gorkov_range)
    best = max(d21_results, key=lambda r: r.gorkov_range)
    f_stand_khz = best.freq_khz
    f_stand_hz = f_stand_khz * 1e3

    # Vortex frequencies to try
    vortex_freqs_khz = [f for f in dish.freqs_khz if f != f_stand_khz]
    # Also add f_stand itself (same-frequency combined, as a reference)
    vortex_freqs_khz = [f_stand_khz] + sorted(vortex_freqs_khz)

    log(f"\n{'='*70}")
    log(f"  D2.2 — Two-frequency sweep: {dish.tag}")
    log(f"  Best standing mode: f_stand = {f_stand_khz} kHz "
        f"(ΔU = {best.gorkov_range:.2e} J)")
    log(f"  Vortex freqs: {vortex_freqs_khz} kHz")
    log(f"{'='*70}")

    # Pre-solve standing at f_stand
    cfg_stand = make_config(dish, f_stand_hz)
    domain_s, ft_s, _ = create_mesh(cfg_stand, verbose=False)
    p_stand_sol = solve_helmholtz(domain_s, ft_s, cfg_stand, mode="standing",
                                  disc_robin=False, verbose=False,
                                  petsc_options=MUMPS_OPTS)
    pts_stand = make_plane_grid(cfg_stand, cfg_stand.H / 2.0, N_PLANE)
    dx_m = cfg_stand.L / (N_PLANE - 1)
    p_stand_vals = sample_pressure(p_stand_sol.p_function, pts_stand)
    p_stand_2d = p_stand_vals.reshape(N_PLANE, N_PLANE)
    U_stand, _, _ = compute_gorkov_grid(p_stand_2d, dx_m, dx_m, cfg_stand)
    traps_stand = find_traps_2d(U_stand)

    log(f"  Standing reference: max|p|={float(np.nanmax(np.abs(p_stand_2d))):.2f} Pa, "
        f"traps={len(traps_stand)}")

    del domain_s, ft_s, p_stand_sol
    gc.collect()

    results = []
    mid = N_PLANE // 2

    for fv_khz in vortex_freqs_khz:
        fv_hz = fv_khz * 1e3
        t0 = time.time()
        log(f"\n  ── f_vortex = {fv_khz} kHz ──")

        cfg_v = make_config(dish, fv_hz)
        domain_v, ft_v, _ = create_mesh(cfg_v, verbose=False)

        p_vortex_sol = solve_helmholtz(domain_v, ft_v, cfg_v, mode="vortex",
                                       disc_robin=False, verbose=False,
                                       petsc_options=MUMPS_OPTS)

        # Sample vortex on same grid size (same physical L)
        pts_v = make_plane_grid(cfg_v, cfg_v.H / 2.0, N_PLANE)
        p_vortex_vals = sample_pressure(p_vortex_sol.p_function, pts_v)
        p_vortex_2d = p_vortex_vals.reshape(N_PLANE, N_PLANE)

        # Gor'kov of vortex-only (at its own frequency)
        U_vortex, _, _ = compute_gorkov_grid(p_vortex_2d, dx_m, dx_m, cfg_v)

        # Combined potential: U_total = U_stand + U_vortex
        # (Valid when f_stand ≠ f_vortex => cross terms vanish;
        #  also computed for f_stand = f_vortex as reference, though
        #  in that case the coherent combined is more accurate — this
        #  gives a conservative estimate.)
        U_combined = U_stand + U_vortex

        traps_combined = find_traps_2d(U_combined)
        depths_c = [t[3] for t in traps_combined] if traps_combined else [0]
        depths_s = [t[3] for t in traps_stand] if traps_stand else [0]

        # Centre-trap depth in combined
        centre_depth = 0.0
        barrier_red = 0.0
        if traps_combined:
            dists_to_centre = [np.sqrt((t[0] - mid) ** 2 + (t[1] - mid) ** 2)
                               for t in traps_combined]
            nearest = int(np.argmin(dists_to_centre))
            centre_depth = traps_combined[nearest][3]
        if traps_stand:
            dists_s = [np.sqrt((t[0] - mid) ** 2 + (t[1] - mid) ** 2)
                       for t in traps_stand]
            nearest_s = int(np.argmin(dists_s))
            stand_centre_depth = traps_stand[nearest_s][3]
            if stand_centre_depth > 0:
                barrier_red = (1 - centre_depth / stand_centre_depth) * 100

        solve_t = time.time() - t0

        tfr = TwoFreqResult(
            tag=dish.tag,
            f_stand_khz=f_stand_khz,
            f_vortex_khz=fv_khz,
            max_p_stand=float(np.nanmax(np.abs(p_stand_2d))),
            max_p_vortex=float(np.nanmax(np.abs(p_vortex_2d))),
            n_traps_stand=len(traps_stand),
            n_traps_combined=len(traps_combined),
            gorkov_range_stand=float(np.nanmax(U_stand) - np.nanmin(U_stand)),
            gorkov_range_combined=float(np.nanmax(U_combined) - np.nanmin(U_combined)),
            mean_depth_stand=float(np.mean(depths_s)),
            mean_depth_combined=float(np.mean(depths_c)),
            centre_trap_depth_combined=centre_depth,
            barrier_reduction_pct=barrier_red,
            solve_time_s=solve_t,
        )
        results.append(tfr)

        log(f"    vortex max|p|={tfr.max_p_vortex:.2f} Pa, "
            f"comb traps={tfr.n_traps_combined}, "
            f"comb ΔU={tfr.gorkov_range_combined:.2e}, "
            f"barrier Δ={tfr.barrier_reduction_pct:+.1f}%, "
            f"{solve_t:.1f}s")

        # Figures
        ftag = f"fv{fv_khz:04d}kHz"
        plot_gorkov(U_combined, cfg_stand,
                    f"{dish.tag} U_comb  f_s={f_stand_khz}kHz + f_v={fv_khz}kHz",
                    ddir / "figs" / f"{ftag}_gorkov_combined.png")
        plot_traps(U_combined, traps_combined, cfg_stand,
                   f"{dish.tag} Traps  f_s={f_stand_khz}kHz + f_v={fv_khz}kHz",
                   ddir / "figs" / f"{ftag}_traps_combined.png")
        plot_gorkov(U_vortex, cfg_v,
                    f"{dish.tag} U_vortex f_v={fv_khz}kHz only",
                    ddir / "figs" / f"{ftag}_gorkov_vortex_only.png")

        del domain_v, ft_v, p_vortex_sol, p_vortex_vals, p_vortex_2d, U_vortex, U_combined
        gc.collect()

    # CSV
    with open(ddir / "two_freq_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(TwoFreqResult.__dataclass_fields__.keys()))
        w.writeheader()
        for r in results:
            w.writerow(r.__dict__)

    return results


# ── D2.2 summary plots ─────────────────────────────────────────────

def plot_d22_summary(all_results: dict[str, list[TwoFreqResult]]):
    """Summary figures for D2.2 two-frequency sweeps."""
    sdir = OUTROOT / "D2.2_summary"
    sdir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    for tag, results in all_results.items():
        fvs = [r.f_vortex_khz for r in results]
        n_trap_c = [r.n_traps_combined for r in results]
        dU_c = [r.gorkov_range_combined for r in results]
        barrier = [r.barrier_reduction_pct for r in results]
        f_stand = results[0].f_stand_khz

        axes[0].plot(fvs, n_trap_c, "o-", label=f"{tag} (f_s={f_stand}kHz)",
                     linewidth=1.5, markersize=5)
        axes[1].plot(fvs, dU_c, "s-", label=f"{tag} (f_s={f_stand}kHz)",
                     linewidth=1.5, markersize=5)
        axes[2].plot(fvs, barrier, "^-", label=f"{tag} (f_s={f_stand}kHz)",
                     linewidth=1.5, markersize=5)

    axes[0].set_ylabel("Trap count (combined)")
    axes[0].set_title("Traps in U_stand + U_vortex")
    axes[1].set_ylabel("ΔU combined (J)")
    axes[1].set_title("Gor'kov range of combined potential")
    axes[2].set_ylabel("Barrier change (%)")
    axes[2].set_title("Centre-trap barrier vs standing-only")
    axes[2].axhline(0, color="k", lw=0.8, ls="--")

    for ax in axes:
        ax.set_xlabel("f_vortex (kHz)")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig.suptitle("D2.2 — Two-Frequency Vortex Superposition", fontsize=14)
    fig.tight_layout()
    fig.savefig(sdir / "two_freq_summary.png", dpi=200)
    plt.close(fig)

    log(f"  D2.2 summary figures saved to {sdir.relative_to(ROOT)}")


# ── Grand summary JSON ──────────────────────────────────────────────

def save_grand_summary(d21: dict, d22: dict):
    """Save a combined JSON of all Phase 2 results."""
    payload = {
        "phase": "Phase 2 — Frequency Sweeps & Resonance Mapping",
        "timestamp": NOW.isoformat(),
        "D2.1": {},
        "D2.2": {},
    }
    for tag, results in d21.items():
        payload["D2.1"][tag] = [r.__dict__ for r in results]
    for tag, results in d22.items():
        payload["D2.2"][tag] = [r.__dict__ for r in results]

    with open(OUTROOT / "phase2_results.json", "w") as f:
        json.dump(payload, f, indent=2, default=str)
    log(f"  Grand summary: {OUTROOT.relative_to(ROOT)}/phase2_results.json")


# ── Main ────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    OUTROOT.mkdir(parents=True, exist_ok=True)

    log(f"\n{'#'*70}")
    log(f"  PHASE 2 — FREQUENCY SWEEPS & RESONANCE MAPPING")
    log(f"  Output: {OUTROOT.relative_to(ROOT)}")
    log(f"  Dishes: {[d.tag for d in DISHES]}")
    log(f"{'#'*70}")

    # ═════════════════════════════════════════════════════════════════
    # D2.1 — Standing-wave frequency sweep
    # ═════════════════════════════════════════════════════════════════
    d21_all: dict[str, list[FreqResult]] = {}
    for dish in DISHES:
        try:
            d21_all[dish.tag] = run_d21_sweep(dish)
        except Exception as e:
            log(f"\n  *** D2.1 FAILED for {dish.tag}: {e} ***")
            import traceback; traceback.print_exc()

    if d21_all:
        plot_d21_summary(d21_all)

    # ═════════════════════════════════════════════════════════════════
    # D2.2 — Two-frequency vortex superposition
    # ═════════════════════════════════════════════════════════════════
    d22_all: dict[str, list[TwoFreqResult]] = {}
    for dish in DISHES:
        if dish.tag not in d21_all:
            log(f"  Skipping D2.2 for {dish.tag} (D2.1 failed)")
            continue
        try:
            d22_all[dish.tag] = run_d22_sweep(dish, d21_all[dish.tag])
        except Exception as e:
            log(f"\n  *** D2.2 FAILED for {dish.tag}: {e} ***")
            import traceback; traceback.print_exc()

    if d22_all:
        plot_d22_summary(d22_all)

    # ═════════════════════════════════════════════════════════════════
    # Grand summary
    # ═════════════════════════════════════════════════════════════════
    save_grand_summary(d21_all, d22_all)

    elapsed = time.time() - t0
    log(f"\n{'#'*70}")
    log(f"  PHASE 2 COMPLETE — {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    log(f"  Output: {OUTROOT.relative_to(ROOT)}")
    log(f"{'#'*70}")

    # Symlink latest
    latest = ROOT / "results" / "phase2_latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(OUTROOT.name)
    log(f"  Symlink: results/phase2_latest -> {OUTROOT.name}")


if __name__ == "__main__":
    main()
