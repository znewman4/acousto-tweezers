#!/usr/bin/env python3
"""
Phase 2b — Resonance Investigation.

Addresses the questions:
1. Are the huge pressure/Gor'kov spikes (900 kHz / 10mm, 600 kHz / 30mm)
   real cavity resonances or mesh artifacts?
2. Do vortex modes have their own resonances?
3. How do standing and vortex Gor'kov strengths interact as BOTH change
   frequency — especially around resonances?
4. Why didn't the combined ΔU change in D2.2?

Study design:
  Part A — Fine standing-wave sweep around the known spikes,
           plus mesh convergence check at the resonant frequency.
  Part B — Vortex-only frequency sweep (same fine grid).
  Part C — Combined interaction map: sweep BOTH f_stand and f_vortex
           on a 2D grid, including off-resonance standing frequencies
           where the vortex contribution actually matters.
  Part D — Proper visualisation: log-scale, dual-axis, normalised
           relative-contribution plots.

Usage:
    micromamba run -n acousto-complex python scripts/experiments/phase2b_resonance_investigation.py
"""
from __future__ import annotations

import sys, os, csv, json, time, gc
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

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
from matplotlib.colors import TwoSlopeNorm, LogNorm

# ── Globals ─────────────────────────────────────────────────────────
comm = MPI.COMM_WORLD
rank = comm.rank
ROOT = Path(__file__).resolve().parents[2]
NOW = datetime.now()
STAMP = NOW.strftime("%Y%m%d_%H%M")
OUTROOT = ROOT / "results" / f"phase2b_resonance_{STAMP}"
N_PLANE = 201
MAX_NX = 50
MIN_EPW = 3

MUMPS_OPTS = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}


def log(msg="", end="\n"):
    if rank == 0:
        print(msg, end=end, flush=True)


# ── Helpers ─────────────────────────────────────────────────────────

def compute_epw(L_m, freq_hz, target_epw):
    lam = 1484.0 / freq_hz
    ideal_nx = int(L_m / lam * target_epw)
    if ideal_nx <= MAX_NX:
        return target_epw
    return max(MIN_EPW, int(MAX_NX * lam / L_m))


def make_config(L_m, R_disc_m, freq_hz, target_epw=8):
    epw = compute_epw(L_m, freq_hz, target_epw)
    return ShallowDishConfig(
        L=L_m, H=1e-3,
        frequency_hz=freq_hz,
        elements_per_wavelength=epw,
        min_elements_z=8,
        rho=997.0, c=1484.0, mu=1.002e-3,
        vortex_velocity_amplitude=10e-6,
        standing_velocity_amplitude=10e-6,
        vortex_topological_charge=1,
        vortex_aperture_radius=R_disc_m,
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
        if len(links):
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
    fp = 2 * min_size + 1
    local_min = (U_clean == minimum_filter(U_clean, size=fp)) & ~nan_mask
    local_max_map = maximum_filter(U_clean, size=fp)
    rows, cols = np.where(local_min)
    traps = []
    for r, c_ in zip(rows, cols):
        depth = local_max_map[r, c_] - U_clean[r, c_]
        traps.append((int(r), int(c_), float(U_clean[r, c_]), float(depth)))
    return traps


def quick_solve(L_m, R_disc_m, freq_hz, mode, target_epw=8):
    """Solve and return (max|p|_plane, ΔU_gorkov, mean_trap_depth, p_2d, U) at z=H/2."""
    cfg = make_config(L_m, R_disc_m, freq_hz, target_epw)
    domain, ft, _ = create_mesh(cfg, verbose=False)
    nc = domain.topology.index_map(domain.topology.dim).size_global
    p_sol = solve_helmholtz(domain, ft, cfg, mode=mode,
                            disc_robin=False, verbose=False,
                            petsc_options=MUMPS_OPTS)
    pts = make_plane_grid(cfg, cfg.H / 2.0, N_PLANE)
    dx_m = cfg.L / (N_PLANE - 1)
    p_vals = sample_pressure(p_sol.p_function, pts)
    p_2d = p_vals.reshape(N_PLANE, N_PLANE)
    U, _, _ = compute_gorkov_grid(p_2d, dx_m, dx_m, cfg)
    traps = find_traps_2d(U)
    depths = [t[3] for t in traps] if traps else [0]
    max_p = float(np.nanmax(np.abs(p_2d)))
    dU = float(np.nanmax(U) - np.nanmin(U))
    mean_d = float(np.mean(depths))
    max_d = float(np.max(depths))
    n_traps = len(traps)
    # Free FE objects
    del domain, ft, p_sol
    gc.collect()
    return max_p, dU, mean_d, max_d, n_traps, p_2d, U, cfg


# ── Part A: Fine standing-wave sweeps & mesh convergence ────────────

def part_a_fine_sweep(L_mm, R_disc_mm, freqs_khz, tag, target_epw=8):
    """Fine standing-wave frequency sweep."""
    L_m = L_mm * 1e-3
    R_m = R_disc_mm * 1e-3
    adir = OUTROOT / tag / "A_standing_fine"
    (adir / "figs").mkdir(parents=True, exist_ok=True)

    log(f"\n{'='*70}")
    log(f"  Part A — Fine standing sweep: {tag}")
    log(f"  {len(freqs_khz)} frequencies: {freqs_khz[0]}–{freqs_khz[-1]} kHz")
    log(f"{'='*70}")

    rows = []
    for fk in freqs_khz:
        t0 = time.time()
        cfg = make_config(L_m, R_m, fk * 1e3, target_epw)
        max_p, dU, mean_d, max_d, nt, p_2d, U, _ = quick_solve(
            L_m, R_m, fk * 1e3, "standing", target_epw)
        dt = time.time() - t0
        lam_mm = 1484.0 / (fk * 1e3) * 1e3
        log(f"    f={fk:7.0f} kHz  nx={cfg.mesh_nx:3d}  epw={cfg.mesh_nx*lam_mm/L_mm:.1f}"
            f"  max|p|={max_p:10.2f}  ΔU={dU:.3e}  traps={nt:4d}  {dt:.1f}s")
        rows.append(dict(freq_khz=fk, max_p=max_p, gorkov_range=dU,
                         mean_depth=mean_d, max_depth=max_d, n_traps=nt,
                         mesh_nx=cfg.mesh_nx, lam_mm=lam_mm,
                         epw_actual=cfg.mesh_nx * lam_mm / L_mm))
        del p_2d, U
        gc.collect()

    # CSV
    with open(adir / "fine_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    return rows


def part_a_mesh_convergence(L_mm, R_disc_mm, freq_khz, tag, epws=[3, 4, 5, 6, 8]):
    """Mesh convergence at a single resonant frequency."""
    L_m = L_mm * 1e-3
    R_m = R_disc_mm * 1e-3
    adir = OUTROOT / tag / "A_mesh_convergence"
    adir.mkdir(parents=True, exist_ok=True)

    log(f"\n  Mesh convergence at {freq_khz} kHz")
    rows = []
    for epw in epws:
        cfg = make_config(L_m, R_m, freq_khz * 1e3, epw)
        if cfg.mesh_nx > 55:  # hard mem limit
            log(f"    epw={epw}: nx={cfg.mesh_nx} — SKIP (too large)")
            continue
        t0 = time.time()
        max_p, dU, mean_d, max_d, nt, _, _, _ = quick_solve(
            L_m, R_m, freq_khz * 1e3, "standing", epw)
        dt = time.time() - t0
        log(f"    epw={epw}: nx={cfg.mesh_nx}  max|p|={max_p:.2f}  ΔU={dU:.3e}  {dt:.1f}s")
        rows.append(dict(epw=epw, mesh_nx=cfg.mesh_nx,
                         max_p=max_p, gorkov_range=dU, n_traps=nt,
                         solve_time=dt))

    with open(adir / "mesh_convergence.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    return rows


# ── Part B: Vortex-only frequency sweep ─────────────────────────────

def part_b_vortex_sweep(L_mm, R_disc_mm, freqs_khz, tag, target_epw=8):
    """Vortex-only sweep at the same fine frequencies."""
    L_m = L_mm * 1e-3
    R_m = R_disc_mm * 1e-3
    bdir = OUTROOT / tag / "B_vortex_fine"
    (bdir / "figs").mkdir(parents=True, exist_ok=True)

    log(f"\n{'='*70}")
    log(f"  Part B — Fine vortex sweep: {tag}")
    log(f"{'='*70}")

    rows = []
    for fk in freqs_khz:
        t0 = time.time()
        cfg = make_config(L_m, R_m, fk * 1e3, target_epw)
        max_p, dU, mean_d, max_d, nt, p_2d, U, _ = quick_solve(
            L_m, R_m, fk * 1e3, "vortex", target_epw)
        dt = time.time() - t0
        lam_mm = 1484.0 / (fk * 1e3) * 1e3
        log(f"    f={fk:7.0f} kHz  max|p|={max_p:10.2f}  ΔU={dU:.3e}  {dt:.1f}s")
        rows.append(dict(freq_khz=fk, max_p=max_p, gorkov_range=dU,
                         mean_depth=mean_d, max_depth=max_d, n_traps=nt,
                         mesh_nx=cfg.mesh_nx))
        del p_2d, U
        gc.collect()

    with open(bdir / "vortex_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    return rows


# ── Part C: 2D interaction map (f_stand × f_vortex) ────────────────

def part_c_interaction_map(L_mm, R_disc_mm, stand_freqs_khz, vortex_freqs_khz,
                           tag, target_epw=8):
    """
    Sweep both f_stand and f_vortex.  For each (f_s, f_v) pair,
    solve standing and vortex independently, combine U_total = U_s + U_v
    (valid when f_s ≠ f_v), and record metrics.

    This shows how the RELATIVE vortex contribution behaves across the
    parameter space.
    """
    L_m = L_mm * 1e-3
    R_m = R_disc_mm * 1e-3
    cdir = OUTROOT / tag / "C_interaction_map"
    (cdir / "figs").mkdir(parents=True, exist_ok=True)

    log(f"\n{'='*70}")
    log(f"  Part C — Interaction map: {tag}")
    log(f"  f_stand: {stand_freqs_khz}")
    log(f"  f_vortex: {vortex_freqs_khz}")
    log(f"  Total: {len(stand_freqs_khz) * len(vortex_freqs_khz)} combos")
    log(f"{'='*70}")

    # Pre-solve all standing modes
    stand_cache = {}
    for fs in stand_freqs_khz:
        t0 = time.time()
        max_p, dU, _, _, nt, p_2d, U, cfg = quick_solve(
            L_m, R_m, fs * 1e3, "standing", target_epw)
        stand_cache[fs] = dict(max_p=max_p, dU=dU, n_traps=nt,
                               p_2d=p_2d, U=U, cfg=cfg)
        log(f"    Standing f={fs} kHz: max|p|={max_p:.2f}, ΔU={dU:.3e}  "
            f"({time.time()-t0:.1f}s)")

    # Pre-solve all vortex modes
    vortex_cache = {}
    for fv in vortex_freqs_khz:
        t0 = time.time()
        max_p, dU, _, _, nt, p_2d, U, cfg = quick_solve(
            L_m, R_m, fv * 1e3, "vortex", target_epw)
        vortex_cache[fv] = dict(max_p=max_p, dU=dU, n_traps=nt,
                                p_2d=p_2d, U=U, cfg=cfg)
        log(f"    Vortex  f={fv} kHz: max|p|={max_p:.2f}, ΔU={dU:.3e}  "
            f"({time.time()-t0:.1f}s)")

    # Combine
    rows = []
    for fs in stand_freqs_khz:
        s = stand_cache[fs]
        for fv in vortex_freqs_khz:
            v = vortex_cache[fv]
            U_comb = s["U"] + v["U"]
            traps_c = find_traps_2d(U_comb)
            dU_comb = float(np.nanmax(U_comb) - np.nanmin(U_comb))
            depths_c = [t[3] for t in traps_c] if traps_c else [0]

            # Relative vortex contribution
            if s["dU"] > 0:
                relative_vortex = v["dU"] / s["dU"]
            else:
                relative_vortex = float("inf")

            # How much does combining change the Gor'kov range vs standing-only?
            if s["dU"] > 0:
                delta_range_pct = (dU_comb - s["dU"]) / s["dU"] * 100
            else:
                delta_range_pct = 0.0

            rows.append(dict(
                f_stand_khz=fs,
                f_vortex_khz=fv,
                max_p_stand=s["max_p"],
                max_p_vortex=v["max_p"],
                dU_stand=s["dU"],
                dU_vortex=v["dU"],
                dU_combined=dU_comb,
                relative_vortex_strength=relative_vortex,
                delta_range_pct=delta_range_pct,
                n_traps_stand=s["n_traps"],
                n_traps_combined=len(traps_c),
                mean_depth_combined=float(np.mean(depths_c)),
            ))

    with open(cdir / "interaction_map.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    # Clean up cached 2D arrays
    del stand_cache, vortex_cache
    gc.collect()

    return rows


# ── Part D: Proper visualisation ────────────────────────────────────

def part_d_plots(tag, a_rows, b_rows, c_rows, a_conv=None):
    """Generate all the investigation plots."""
    pdir = OUTROOT / tag / "plots"
    pdir.mkdir(parents=True, exist_ok=True)

    # ── D1: Standing + vortex ΔU on SAME log-scale axes ──
    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    fs_a = [r["freq_khz"] for r in a_rows]
    dU_a = [r["gorkov_range"] for r in a_rows]
    fs_b = [r["freq_khz"] for r in b_rows]
    dU_b = [r["gorkov_range"] for r in b_rows]
    maxp_a = [r["max_p"] for r in a_rows]
    maxp_b = [r["max_p"] for r in b_rows]

    ax1.semilogy(fs_a, dU_a, "o-", color="steelblue", linewidth=1.8,
                 markersize=5, label="Standing ΔU", zorder=5)
    ax1.semilogy(fs_b, dU_b, "s-", color="orangered", linewidth=1.8,
                 markersize=5, label="Vortex ΔU", zorder=5)
    ax1.set_xlabel("Frequency (kHz)", fontsize=12)
    ax1.set_ylabel("Gor'kov range ΔU (J)", fontsize=12)
    ax1.legend(loc="upper left", fontsize=10)
    ax1.grid(True, alpha=0.3, which="both")
    ax1.set_title(f"{tag} — Standing vs Vortex Trap Strength (log scale)", fontsize=12)

    # Secondary axis for max|p|
    ax2 = ax1.twinx()
    ax2.semilogy(fs_a, maxp_a, "^--", color="steelblue", linewidth=0.8,
                 markersize=4, alpha=0.4, label="Standing max|p|")
    ax2.semilogy(fs_b, maxp_b, "v--", color="orangered", linewidth=0.8,
                 markersize=4, alpha=0.4, label="Vortex max|p|")
    ax2.set_ylabel("max|p| at z=H/2 (Pa)", fontsize=11)
    ax2.legend(loc="upper right", fontsize=9, framealpha=0.7)

    fig.tight_layout()
    fig.savefig(pdir / "D1_standing_vs_vortex_log.png", dpi=200)
    plt.close(fig)

    # ── D2: Relative vortex strength ΔU_vortex / ΔU_standing ──
    fig, ax = plt.subplots(figsize=(10, 5))
    # Need matching frequencies
    common_fs = sorted(set(fs_a) & set(fs_b))
    dU_s_dict = {r["freq_khz"]: r["gorkov_range"] for r in a_rows}
    dU_v_dict = {r["freq_khz"]: r["gorkov_range"] for r in b_rows}
    ratios = [dU_v_dict[f] / dU_s_dict[f] if dU_s_dict[f] > 0 else 0
              for f in common_fs]
    ax.semilogy(common_fs, ratios, "D-", color="purple", linewidth=1.8, markersize=6)
    ax.axhline(1.0, color="k", ls="--", lw=0.8, alpha=0.5)
    ax.axhline(0.1, color="gray", ls=":", lw=0.8, alpha=0.5)
    ax.set_xlabel("Frequency (kHz)", fontsize=12)
    ax.set_ylabel("ΔU_vortex / ΔU_standing", fontsize=12)
    ax.set_title(f"{tag} — Relative Vortex Strength (ratio of Gor'kov ranges)", fontsize=12)
    ax.grid(True, alpha=0.3, which="both")
    # Annotate key insight
    min_ratio_f = common_fs[int(np.argmin(ratios))]
    min_ratio_v = min(ratios)
    ax.annotate(f"Min ratio: {min_ratio_v:.3f}\nat {min_ratio_f} kHz",
                xy=(min_ratio_f, min_ratio_v),
                xytext=(min_ratio_f + 30, min_ratio_v * 5),
                arrowprops=dict(arrowstyle="->", color="purple"),
                fontsize=9, color="purple")
    fig.tight_layout()
    fig.savefig(pdir / "D2_vortex_relative_strength.png", dpi=200)
    plt.close(fig)

    # ── D3: max|p| for both modes (linear scale, fine grid) ──
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 5))
    ax_l.plot(fs_a, maxp_a, "o-", color="steelblue", linewidth=1.5, label="Standing")
    ax_l.plot(fs_b, maxp_b, "s-", color="orangered", linewidth=1.5, label="Vortex")
    ax_l.set_xlabel("Frequency (kHz)"); ax_l.set_ylabel("max|p| (Pa)")
    ax_l.set_title(f"{tag} — max|p| (linear, full range)")
    ax_l.legend(); ax_l.grid(True, alpha=0.3)

    # Same but excluding the spike to see the fine structure
    # Clip to 95th percentile to see detail
    all_vals = maxp_a + maxp_b
    clip_val = np.percentile(all_vals, 90) * 1.5
    ax_r.plot(fs_a, np.minimum(maxp_a, clip_val), "o-", color="steelblue",
              linewidth=1.5, label="Standing")
    ax_r.plot(fs_b, np.minimum(maxp_b, clip_val), "s-", color="orangered",
              linewidth=1.5, label="Vortex")
    ax_r.set_xlabel("Frequency (kHz)"); ax_r.set_ylabel("max|p| (Pa)")
    ax_r.set_title(f"{tag} — max|p| (clipped at {clip_val:.0f} Pa to show detail)")
    ax_r.legend(); ax_r.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(pdir / "D3_max_p_linear_detail.png", dpi=200)
    plt.close(fig)

    # ── D4: ΔU for both modes (linear scale, fine grid) — clipped ──
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 5))
    ax_l.plot(fs_a, dU_a, "o-", color="steelblue", linewidth=1.5, label="Standing")
    ax_l.plot(fs_b, dU_b, "s-", color="orangered", linewidth=1.5, label="Vortex")
    ax_l.set_xlabel("Frequency (kHz)"); ax_l.set_ylabel("ΔU (J)")
    ax_l.set_title(f"{tag} — Gor'kov range (linear, full range)")
    ax_l.legend(); ax_l.grid(True, alpha=0.3)

    all_dU = dU_a + dU_b
    clip_dU = np.percentile(all_dU, 85) * 2.0
    ax_r.plot(fs_a, np.minimum(dU_a, clip_dU), "o-", color="steelblue",
              linewidth=1.5, label="Standing")
    ax_r.plot(fs_b, np.minimum(dU_b, clip_dU), "s-", color="orangered",
              linewidth=1.5, label="Vortex")
    ax_r.set_xlabel("Frequency (kHz)"); ax_r.set_ylabel("ΔU (J)")
    ax_r.set_title(f"{tag} — Gor'kov range (clipped to show detail)")
    ax_r.legend(); ax_r.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(pdir / "D4_gorkov_range_linear_detail.png", dpi=200)
    plt.close(fig)

    # ── D5: Interaction heatmap — delta_range_pct ──
    if c_rows:
        fs_stand = sorted(set(r["f_stand_khz"] for r in c_rows))
        fs_vortex = sorted(set(r["f_vortex_khz"] for r in c_rows))
        ns, nv = len(fs_stand), len(fs_vortex)

        # Build 2D grids
        delta_grid = np.full((nv, ns), np.nan)
        rel_grid = np.full((nv, ns), np.nan)
        dU_comb_grid = np.full((nv, ns), np.nan)
        for r in c_rows:
            si = fs_stand.index(r["f_stand_khz"])
            vi = fs_vortex.index(r["f_vortex_khz"])
            delta_grid[vi, si] = r["delta_range_pct"]
            rel_grid[vi, si] = r["relative_vortex_strength"]
            dU_comb_grid[vi, si] = r["dU_combined"]

        # D5a: How much the combined ΔU changes vs standing-only (%)
        fig, ax = plt.subplots(figsize=(10, 7))
        vmax = max(abs(np.nanmin(delta_grid)), abs(np.nanmax(delta_grid)))
        if vmax < 0.01:
            vmax = 1.0
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        im = ax.imshow(delta_grid, origin="lower", aspect="auto",
                       cmap="RdBu_r", norm=norm,
                       extent=[fs_stand[0], fs_stand[-1],
                               fs_vortex[0], fs_vortex[-1]])
        fig.colorbar(im, ax=ax, label="Δ(ΔU) vs standing-only (%)", shrink=0.85)
        ax.set_xlabel("f_standing (kHz)", fontsize=12)
        ax.set_ylabel("f_vortex (kHz)", fontsize=12)
        ax.set_title(f"{tag} — Change in Gor'kov range when adding vortex (%)", fontsize=12)
        # Add text annotations
        for r in c_rows:
            si = fs_stand.index(r["f_stand_khz"])
            vi = fs_vortex.index(r["f_vortex_khz"])
            ax.text(r["f_stand_khz"], r["f_vortex_khz"],
                    f"{r['delta_range_pct']:+.1f}%",
                    ha="center", va="center", fontsize=7,
                    color="white" if abs(r["delta_range_pct"]) > vmax * 0.5 else "black")
        fig.tight_layout()
        fig.savefig(pdir / "D5a_interaction_delta_pct.png", dpi=200)
        plt.close(fig)

        # D5b: Relative vortex strength map
        fig, ax = plt.subplots(figsize=(10, 7))
        im = ax.imshow(rel_grid, origin="lower", aspect="auto",
                       cmap="magma",
                       norm=LogNorm(vmin=max(np.nanmin(rel_grid), 1e-4),
                                    vmax=np.nanmax(rel_grid)),
                       extent=[fs_stand[0], fs_stand[-1],
                               fs_vortex[0], fs_vortex[-1]])
        fig.colorbar(im, ax=ax, label="ΔU_vortex / ΔU_standing", shrink=0.85)
        ax.set_xlabel("f_standing (kHz)", fontsize=12)
        ax.set_ylabel("f_vortex (kHz)", fontsize=12)
        ax.set_title(f"{tag} — Relative Vortex Strength (ΔU_v / ΔU_s)", fontsize=12)
        for r in c_rows:
            ax.text(r["f_stand_khz"], r["f_vortex_khz"],
                    f"{r['relative_vortex_strength']:.2f}",
                    ha="center", va="center", fontsize=7,
                    color="white")
        fig.tight_layout()
        fig.savefig(pdir / "D5b_relative_vortex_map.png", dpi=200)
        plt.close(fig)

        # D5c: Combined ΔU heatmap (log scale)
        fig, ax = plt.subplots(figsize=(10, 7))
        im = ax.imshow(dU_comb_grid, origin="lower", aspect="auto",
                       cmap="viridis",
                       norm=LogNorm(vmin=np.nanmin(dU_comb_grid),
                                    vmax=np.nanmax(dU_comb_grid)),
                       extent=[fs_stand[0], fs_stand[-1],
                               fs_vortex[0], fs_vortex[-1]])
        fig.colorbar(im, ax=ax, label="Combined ΔU (J)", shrink=0.85)
        ax.set_xlabel("f_standing (kHz)", fontsize=12)
        ax.set_ylabel("f_vortex (kHz)", fontsize=12)
        ax.set_title(f"{tag} — Combined Gor'kov Range ΔU (log scale)", fontsize=12)
        for r in c_rows:
            ax.text(r["f_stand_khz"], r["f_vortex_khz"],
                    f"{r['dU_combined']:.1e}",
                    ha="center", va="center", fontsize=6,
                    color="white")
        fig.tight_layout()
        fig.savefig(pdir / "D5c_combined_gorkov_map.png", dpi=200)
        plt.close(fig)

    # ── D6: Mesh convergence (if available) ──
    if a_conv:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        epws = [r["epw"] for r in a_conv]
        nxs = [r["mesh_nx"] for r in a_conv]
        maxps = [r["max_p"] for r in a_conv]
        dUs = [r["gorkov_range"] for r in a_conv]

        ax1.plot(nxs, maxps, "o-", color="steelblue", linewidth=1.5)
        for i, e in enumerate(epws):
            ax1.annotate(f"epw={e}", (nxs[i], maxps[i]),
                         textcoords="offset points", xytext=(5, 5), fontsize=8)
        ax1.set_xlabel("mesh nx")
        ax1.set_ylabel("max|p| (Pa)")
        ax1.set_title("Mesh convergence — max|p|")
        ax1.grid(True, alpha=0.3)

        ax2.semilogy(nxs, dUs, "s-", color="orangered", linewidth=1.5)
        for i, e in enumerate(epws):
            ax2.annotate(f"epw={e}", (nxs[i], dUs[i]),
                         textcoords="offset points", xytext=(5, 5), fontsize=8)
        ax2.set_xlabel("mesh nx")
        ax2.set_ylabel("ΔU (J)")
        ax2.set_title("Mesh convergence — Gor'kov range (log)")
        ax2.grid(True, alpha=0.3, which="both")

        fig.suptitle(f"{tag} — Mesh Convergence at Resonance", fontsize=13)
        fig.tight_layout()
        fig.savefig(pdir / "D6_mesh_convergence.png", dpi=200)
        plt.close(fig)

    # ── D7: Standing ΔU vs frequency — BOTH linear clipped AND log ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Linear, full range
    axes[0].plot(fs_a, dU_a, "o-", color="steelblue", linewidth=1.5)
    axes[0].set_ylabel("ΔU (J)"); axes[0].set_xlabel("f (kHz)")
    axes[0].set_title("Standing ΔU — Linear (full)")
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Log scale
    axes[1].semilogy(fs_a, dU_a, "o-", color="steelblue", linewidth=1.5)
    axes[1].set_ylabel("ΔU (J)"); axes[1].set_xlabel("f (kHz)")
    axes[1].set_title("Standing ΔU — Log scale")
    axes[1].grid(True, alpha=0.3, which="both")

    # Panel 3: Linear, excluding the spike (median ± 5x)
    median_dU = np.median(dU_a)
    clip_hi = median_dU * 8
    dU_clipped = [min(d, clip_hi) for d in dU_a]
    axes[2].plot(fs_a, dU_clipped, "o-", color="steelblue", linewidth=1.5)
    axes[2].set_ylabel("ΔU (J)"); axes[2].set_xlabel("f (kHz)")
    axes[2].set_title(f"Standing ΔU — Clipped at {clip_hi:.2e}")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(f"{tag} — Standing Gor'kov Range: Three Views", fontsize=13)
    fig.tight_layout()
    fig.savefig(pdir / "D7_standing_gorkov_three_views.png", dpi=200)
    plt.close(fig)

    log(f"  All plots saved to {pdir.relative_to(ROOT)}")


# ── Main ────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    OUTROOT.mkdir(parents=True, exist_ok=True)

    log(f"\n{'#'*70}")
    log(f"  PHASE 2b — RESONANCE INVESTIGATION")
    log(f"  Output: {OUTROOT.relative_to(ROOT)}")
    log(f"{'#'*70}")

    # ───────────────────────────────────────────────────────────────
    # 10 mm dish  (resonance suspected at ~900 kHz)
    # ───────────────────────────────────────────────────────────────
    tag10 = "L10_D10"
    fine10 = list(range(500, 1010, 20))  # 500, 520, …, 1000 (26 pts)

    a10 = part_a_fine_sweep(10, 5.0, fine10, tag10, target_epw=8)
    conv10 = part_a_mesh_convergence(10, 5.0, 900, tag10, epws=[3, 4, 5, 6, 8])
    b10 = part_b_vortex_sweep(10, 5.0, fine10, tag10, target_epw=8)

    # Interaction map: pick 5 standing freqs spanning off-resonance to on-resonance
    # and 5 vortex freqs
    stand_set_10 = [500, 700, 800, 880, 900]
    vortex_set_10 = [500, 700, 800, 880, 900]
    c10 = part_c_interaction_map(10, 5.0, stand_set_10, vortex_set_10,
                                 tag10, target_epw=8)
    part_d_plots(tag10, a10, b10, c10, conv10)

    # ───────────────────────────────────────────────────────────────
    # 30 mm dish  (resonance suspected at ~600 kHz)
    # ───────────────────────────────────────────────────────────────
    tag30 = "L30_D10"
    fine30 = list(range(500, 810, 20))  # 500, 520, …, 800 (16 pts)

    a30 = part_a_fine_sweep(30, 5.0, fine30, tag30, target_epw=6)
    conv30 = part_a_mesh_convergence(30, 5.0, 600, tag30, epws=[3, 4, 5])
    b30 = part_b_vortex_sweep(30, 5.0, fine30, tag30, target_epw=6)

    stand_set_30 = [500, 560, 580, 600, 700]
    vortex_set_30 = [500, 560, 580, 600, 700]
    c30 = part_c_interaction_map(30, 5.0, stand_set_30, vortex_set_30,
                                 tag30, target_epw=6)
    part_d_plots(tag30, a30, b30, c30, conv30)

    # ── Grand summary JSON ──
    payload = {
        "phase": "Phase 2b — Resonance Investigation",
        "timestamp": NOW.isoformat(),
        "L10_D10": {
            "fine_standing": a10,
            "mesh_convergence": conv10,
            "fine_vortex": b10,
            "interaction_map": c10,
        },
        "L30_D10": {
            "fine_standing": a30,
            "mesh_convergence": conv30,
            "fine_vortex": b30,
            "interaction_map": c30,
        },
    }
    with open(OUTROOT / "phase2b_results.json", "w") as f:
        json.dump(payload, f, indent=2, default=str)

    # Symlink
    latest = ROOT / "results" / "phase2b_latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(OUTROOT.name)

    elapsed = time.time() - t0
    log(f"\n{'#'*70}")
    log(f"  PHASE 2b COMPLETE — {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    log(f"  Output: {OUTROOT.relative_to(ROOT)}")
    log(f"{'#'*70}")


if __name__ == "__main__":
    main()
