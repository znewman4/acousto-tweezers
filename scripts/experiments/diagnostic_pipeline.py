#!/usr/bin/env python3
"""
Diagnostic Pipeline — Steps 1–6 for trustworthy propagation + interaction.

Step 1: Convergence proof — GMRES vs Direct (MUMPS) on coarse mesh
Step 2: Free-space vortex unit test (no standing, no petri slab, PML all sides)
Step 3: PML stress test — sweep thickness and σ strength
Step 4: Amplitude balancing — 3×3 ratio sweep for visible interaction
Step 5: Interaction metrics — Δ maps and selectivity indices
Step 6: Gor'kov proxy — trap depth, stiffness, size sweep

Usage:
    micromamba run -n acousto-complex python scripts/experiments/diagnostic_pipeline.py
"""
from __future__ import annotations

import gc
import json
import csv
import sys
import time
import traceback
import numpy as np
from dataclasses import replace, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from acoustweezers.experiments.farfield_petri_cuboid.config import FarFieldConfig
from acoustweezers.experiments.farfield_petri_cuboid.solve_pressure import (
    solve_helmholtz, PressureSolution,
)
from acoustweezers.experiments.farfield_petri_cuboid.post import (
    slice_xy, slice_xz, centerline_z, energy_physical_vs_pml,
)


# ═════════════════════════════════════════════════════════════════════
#  Shared helpers
# ═════════════════════════════════════════════════════════════════════

def _off_axis_centerline(sol, x_offset, nz=500):
    from scipy.interpolate import NearestNDInterpolator
    coords = sol.coords; pv = sol.p_values
    interp = NearestNDInterpolator(coords, pv)
    cfg = sol.cfg
    zg = np.linspace(0, cfg.H_total, nz)
    cx = cfg.disk_center_x + x_offset
    cy = cfg.disk_center_y
    pts = np.column_stack([np.full(nz, cx), np.full(nz, cy), zg])
    pc = interp(pts)
    return zg, np.abs(pc)


def _radial_profile(sol, z_val, nr=100):
    """Radial |p|(r) at height z_val, azimuthally averaged."""
    from scipy.interpolate import NearestNDInterpolator
    coords = sol.coords; pv = sol.p_values
    interp = NearestNDInterpolator(coords, pv)
    cfg = sol.cfg
    cx, cy = cfg.disk_center_x, cfg.disk_center_y
    r_max = min(cfg.Lx/2 - cfg.t_pml_xy, cfg.Ly/2 - cfg.t_pml_xy)
    rg = np.linspace(0, r_max, nr)
    n_theta = 36
    thetas = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
    pmag_avg = np.zeros(nr)
    for ir, r in enumerate(rg):
        pts = np.column_stack([
            cx + r*np.cos(thetas),
            cy + r*np.sin(thetas),
            np.full(n_theta, z_val)
        ])
        pmag_avg[ir] = float(np.mean(np.abs(interp(pts))))
    return rg, pmag_avg


def _roi_mask_xy(xg, yg, cfg):
    t_xy = cfg.t_pml_xy
    X, Y = np.meshgrid(xg, yg)
    return (X >= t_xy) & (X <= cfg.Lx - t_xy) & (Y >= t_xy) & (Y <= cfg.Ly - t_xy)


def _roi_metrics(sol, z_val, nx=200, ny=200):
    xg, yg, pmag, _ = slice_xy(sol, z_val, nx, ny)
    mask = _roi_mask_xy(xg, yg, sol.cfg)
    pm = pmag[mask]
    return {"mean_abs_p": float(np.mean(pm)), "max_abs_p": float(np.max(pm))}


def _section_banner(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


KSP_REASONS = {
    1: "CONVERGED_RTOL_NORMAL", 2: "CONVERGED_RTOL",
    3: "CONVERGED_ATOL", 9: "CONVERGED_ITERATING",
    -3: "DIVERGED_ITS", -4: "DIVERGED_DTOL",
    -5: "DIVERGED_BREAKDOWN", -9: "DIVERGED_NANORINF",
}


# ═════════════════════════════════════════════════════════════════════
#  Base config for the pipeline  (coarse mesh for speed)
# ═════════════════════════════════════════════════════════════════════

BASE_CFG = dict(
    Lx=6e-3, Ly=6e-3, H_under=3e-3, H_top=1e-3,
    frequency_hz=2.0e6, disk_radius=1.0e-3,
    disk_velocity_amplitude=10e-6, vortex_topological_charge=1,
    standing_velocity_amplitude=1e-6, standing_phase_pattern="antiphase",
    standing_axis="both", top_bc_type="impedance", top_impedance_Zrel=0.001,
    pml_n_wavelengths_xy=1.0, pml_n_wavelengths_z=1.0,
    pml_degree=2, pml_sigma_max_factor=5.0, pml_enabled=True,
    elements_per_wavelength=5,
    lens_drive="plastic", lens_l=1, lens_focal_length=10e-3,
    lens_focus_offset_x=0.2e-3, lens_focus_offset_y=0.0,
    lens_c_lens=2700.0, lens_apodization="cosine_taper", lens_apodization_strength=1.0,
)

# Coarse version (3 elem/λ) for fast diagnostic solves
COARSE_CFG = {**BASE_CFG, "elements_per_wavelength": 3}


# ═════════════════════════════════════════════════════════════════════
#  STEP 1 — Prove convergence: GMRES vs Direct
# ═════════════════════════════════════════════════════════════════════

def step1_convergence(out_dir: Path, report: list):
    _section_banner("STEP 1 — CONVERGENCE PROOF (GMRES vs Direct)")
    fig_dir = out_dir / "step1_convergence" / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # We test vortex_only (the case that diverged) on coarse mesh
    cfg_coarse = FarFieldConfig(**{**COARSE_CFG, "standing_velocity_amplitude": 0.0})
    print(f"  Coarse mesh: {cfg_coarse.mesh_nx}×{cfg_coarse.mesh_ny}×{cfg_coarse.mesh_nz}")
    print(f"  DOFs estimate: ~{cfg_coarse.mesh_nx*cfg_coarse.mesh_ny*cfg_coarse.mesh_nz*10}")

    results = {}

    # --- A) GMRES + ILU (what we used before) ---
    print("\n  [1a] GMRES + ILU …")
    gmres_opts = {
        "ksp_type": "gmres", "ksp_rtol": 1e-6, "ksp_max_it": 10000,
        "ksp_gmres_restart": 300, "pc_type": "ilu",
    }
    sol_gmres = solve_helmholtz(cfg_coarse, verbose=True, petsc_options=gmres_opts)
    results["gmres_ilu"] = _extract_solver_meta(sol_gmres)
    zc_gmres, pc_gmres = centerline_z(sol_gmres)
    del sol_gmres; gc.collect()

    # --- B) Direct solve (MUMPS LU) ---
    print("\n  [1b] Direct solve (MUMPS LU) …")
    direct_opts = {
        "ksp_type": "preonly", "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    sol_direct = solve_helmholtz(cfg_coarse, verbose=True, petsc_options=direct_opts)
    results["direct_mumps"] = _extract_solver_meta(sol_direct)
    zc_direct, pc_direct = centerline_z(sol_direct)

    # --- C) GMRES + LU (preconditioned with full LU — should converge in 1) ---
    print("\n  [1c] GMRES + LU preconditioner …")
    gmres_lu_opts = {
        "ksp_type": "gmres", "ksp_rtol": 1e-6, "ksp_max_it": 100,
        "pc_type": "lu", "pc_factor_mat_solver_type": "mumps",
    }
    sol_gmres_lu = solve_helmholtz(cfg_coarse, verbose=True, petsc_options=gmres_lu_opts)
    results["gmres_lu"] = _extract_solver_meta(sol_gmres_lu)
    zc_glu, pc_glu = centerline_z(sol_gmres_lu)
    del sol_gmres_lu; gc.collect()

    # Get XZ slice from direct solve for reference
    xg, zg, pmag_direct, _ = slice_xz(sol_direct, cfg_coarse.Ly / 2)
    del sol_direct; gc.collect()

    # --- Plot comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(zc_direct*1e3, pc_direct, 'k-', lw=2, label='Direct (MUMPS)')
    axes[0].plot(zc_gmres*1e3, pc_gmres, 'r--', lw=1.5, label='GMRES+ILU')
    axes[0].plot(zc_glu*1e3, pc_glu, 'b:', lw=1.5, label='GMRES+LU')
    axes[0].set_xlabel("z [mm]"); axes[0].set_ylabel("|p| [Pa]")
    axes[0].set_title("Step 1: Centerline |p| — Solver Comparison")
    axes[0].legend()

    im = axes[1].pcolormesh(xg*1e3, zg*1e3, pmag_direct, shading="auto", cmap="inferno")
    axes[1].set_title("|p| XZ — Direct (MUMPS) reference")
    axes[1].set_xlabel("x [mm]"); axes[1].set_ylabel("z [mm]")
    plt.colorbar(im, ax=axes[1], label="Pa")
    fig.tight_layout()
    fig.savefig(fig_dir / "solver_comparison.png", dpi=150)
    plt.close(fig)

    # --- Verdict ---
    dir_max = results["direct_mumps"]["max_abs_p"]
    gmr_max = results["gmres_ilu"]["max_abs_p"]
    rel_diff = abs(gmr_max - dir_max) / (dir_max + 1e-30)

    verdict = ("SOLVER PROBLEM: GMRES+ILU diverges while direct solve is clean."
               if results["gmres_ilu"]["ksp_converged_reason"] < 0 and
                  results["direct_mumps"]["ksp_converged_reason"] >= 0
               else "BC/PML/MESH PROBLEM" if results["direct_mumps"]["max_abs_p"] < 1e-10
               else f"Both solvers produce fields. Relative max|p| diff = {rel_diff:.2e}")

    print(f"\n  ── STEP 1 VERDICT: {verdict}")
    report.append(f"## Step 1 — Convergence\n")
    report.append(f"| Solver | max|p| [Pa] | KSP iters | Converged | |r| | Time [s] |")
    report.append(f"|--------|------------|-----------|-----------|-----|----------|")
    for name, m in results.items():
        reason = KSP_REASONS.get(m["ksp_converged_reason"], str(m["ksp_converged_reason"]))
        report.append(f"| {name} | {m['max_abs_p']:.4f} | {m['ksp_iters']} | "
                      f"{reason} | {m['ksp_residual_norm']:.2e} | {m['solver_time_s']:.1f} |")
    report.append(f"\n**Verdict:** {verdict}\n")
    report.append(f"**Fix applied:** Switching default solver to MUMPS direct for reliability.\n")

    return results


def _extract_solver_meta(sol):
    return {
        "max_abs_p": sol.max_pressure,
        "dofs": sol.dofs,
        "ksp_iters": sol.ksp_iterations,
        "ksp_converged_reason": sol.ksp_converged_reason,
        "ksp_residual_norm": sol.ksp_residual_norm,
        "solver_time_s": sol.solver_time,
    }


# ═════════════════════════════════════════════════════════════════════
#  STEP 2 — Free-space vortex unit test
# ═════════════════════════════════════════════════════════════════════

def step2_freespace_vortex(out_dir: Path, report: list):
    _section_banner("STEP 2 — FREE-SPACE VORTEX UNIT TEST")
    fig_dir = out_dir / "step2_freespace" / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Stripped config: no standing, impedance-matched top (Z_rel=1 = water),
    # PML on all sides. This is a "free water" simulation.
    cfg = FarFieldConfig(
        **{**COARSE_CFG,
           "standing_velocity_amplitude": 0.0,
           "top_bc_type": "impedance",
           "top_impedance_Zrel": 1.0,   # matched → no reflection from top
           }
    )
    print(cfg.describe())

    direct_opts = {
        "ksp_type": "preonly", "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    sol = solve_helmholtz(cfg, verbose=True, petsc_options=direct_opts)

    # --- XY slices at several z heights ---
    z_heights = {
        "z=0.5mm": 0.5e-3,
        "z=1.0mm": 1.0e-3,
        "z=2.0mm": 2.0e-3,
        "z=3.0mm": 3.0e-3,   # = H_under (petri base)
        "z=3.5mm": 3.5e-3,   # petri mid
    }

    # Magnitude slices
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    vmax = 0
    slice_data = {}
    for label, zv in z_heights.items():
        xg, yg, pmag, pphase = slice_xy(sol, zv)
        slice_data[label] = (xg, yg, pmag, pphase)
        vmax = max(vmax, pmag.max())

    for ax, (label, zv) in zip(axes.flat[:5], z_heights.items()):
        xg, yg, pmag, pphase = slice_data[label]
        im = ax.pcolormesh(xg*1e3, yg*1e3, pmag, shading="auto",
                           cmap="inferno", vmin=0, vmax=vmax)
        ax.set_title(f"|p| at {label}")
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
        ax.set_aspect("equal")
    plt.colorbar(im, ax=axes.flat[:5].tolist(), label="Pa", shrink=0.7)
    axes.flat[5].axis("off")
    fig.suptitle("Step 2: Free-space vortex — |p| XY slices", fontsize=13)
    fig.tight_layout()
    fig.savefig(fig_dir / "xy_magnitude_slices.png", dpi=150)
    plt.close(fig)

    # Phase slices
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, (label, zv) in zip(axes.flat[:5], z_heights.items()):
        xg, yg, pmag, pphase = slice_data[label]
        im = ax.pcolormesh(xg*1e3, yg*1e3, pphase, shading="auto",
                           cmap="twilight", vmin=-np.pi, vmax=np.pi)
        ax.set_title(f"arg(p) at {label}")
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
        ax.set_aspect("equal")
    plt.colorbar(im, ax=axes.flat[:5].tolist(), label="rad", shrink=0.7)
    axes.flat[5].axis("off")
    fig.suptitle("Step 2: Free-space vortex — phase XY slices (vortex signature)", fontsize=13)
    fig.tight_layout()
    fig.savefig(fig_dir / "xy_phase_slices.png", dpi=150)
    plt.close(fig)

    # --- Centerline + off-axis + radial profiles ---
    zc, pc_on = centerline_z(sol)
    zc_off, pc_off = _off_axis_centerline(sol, cfg.lens_focus_offset_x)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(zc*1e3, pc_on, 'k-', label='on-axis')
    axes[0].plot(zc_off*1e3, pc_off, 'b--', label=f'off-axis (+{cfg.lens_focus_offset_x*1e3:.2f}mm)')
    axes[0].set_xlabel("z [mm]"); axes[0].set_ylabel("|p| [Pa]")
    axes[0].set_title("Centerline |p| vs z")
    axes[0].legend()

    # Radial profiles at several z
    for label, zv in list(z_heights.items())[:4]:
        rg, p_rad = _radial_profile(sol, zv)
        axes[1].plot(rg*1e3, p_rad, label=label)
    axes[1].set_xlabel("r [mm]"); axes[1].set_ylabel("|p| [Pa]")
    axes[1].set_title("Radial profile |p|(r)")
    axes[1].legend(fontsize=7)

    # XZ mid-plane
    xg, zg, pmag_xz, _ = slice_xz(sol, cfg.Ly/2)
    im = axes[2].pcolormesh(xg*1e3, zg*1e3, pmag_xz, shading="auto", cmap="inferno")
    axes[2].set_title("|p| XZ mid-plane")
    axes[2].set_xlabel("x [mm]"); axes[2].set_ylabel("z [mm]")
    plt.colorbar(im, ax=axes[2], label="Pa")
    fig.suptitle("Step 2: Free-space vortex diagnostics", fontsize=13)
    fig.tight_layout()
    fig.savefig(fig_dir / "profiles_and_xz.png", dpi=150)
    plt.close(fig)

    # --- Verdict ---
    # Check for vortex signature: on-axis should have a null (vortex core is dark)
    on_axis_roi = pc_on[(zc > 0.5e-3) & (zc < 3e-3)]
    off_axis_roi = pc_off[(zc_off > 0.5e-3) & (zc_off < 3e-3)]
    on_max = float(np.max(on_axis_roi)) if len(on_axis_roi) > 0 else 0
    off_max = float(np.max(off_axis_roi)) if len(off_axis_roi) > 0 else 0
    null_ratio = on_max / (off_max + 1e-30)

    if null_ratio < 0.3:
        verdict = f"PASS — clear vortex null on axis (on/off ratio = {null_ratio:.3f})"
    elif null_ratio < 0.7:
        verdict = f"PARTIAL — weak vortex null (on/off ratio = {null_ratio:.3f})"
    else:
        verdict = f"FAIL — no vortex null visible (on/off ratio = {null_ratio:.3f})"

    print(f"\n  ── STEP 2 VERDICT: {verdict}")
    report.append(f"## Step 2 — Free-Space Vortex Unit Test\n")
    report.append(f"- On-axis max |p| in ROI: {on_max:.4f} Pa")
    report.append(f"- Off-axis max |p| in ROI: {off_max:.4f} Pa")
    report.append(f"- Null ratio (on/off): {null_ratio:.3f}")
    report.append(f"- max|p| global: {sol.max_pressure:.4f} Pa")
    report.append(f"- DOFs: {sol.dofs}, KSP iters: {sol.ksp_iterations}")
    report.append(f"\n**Verdict:** {verdict}\n")

    del sol; gc.collect()


# ═════════════════════════════════════════════════════════════════════
#  STEP 3 — PML stress test
# ═════════════════════════════════════════════════════════════════════

def step3_pml_stress(out_dir: Path, report: list):
    _section_banner("STEP 3 — PML STRESS TEST")
    fig_dir = out_dir / "step3_pml" / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)

    direct_opts = {
        "ksp_type": "preonly", "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }

    # Sweep PML thickness and σ strength
    pml_thicknesses = [0.5, 1.0, 2.0]   # in wavelengths
    sigma_factors   = [2.0, 5.0, 10.0]

    results = {}
    ref_cfg = FarFieldConfig(**COARSE_CFG)
    z_petri_mid = ref_cfg.H_under + ref_cfg.H_top / 2

    for t_lam in pml_thicknesses:
        for sf in sigma_factors:
            label = f"t={t_lam:.1f}λ_σ={sf:.0f}"
            print(f"\n  [{label}]")
            cfg = FarFieldConfig(**{
                **COARSE_CFG,
                "standing_velocity_amplitude": 0.0,  # vortex only for clean test
                "pml_n_wavelengths_xy": t_lam,
                "pml_n_wavelengths_z": t_lam,
                "pml_sigma_max_factor": sf,
            })
            sol = solve_helmholtz(cfg, verbose=True, petsc_options=direct_opts)
            roi = _roi_metrics(sol, z_petri_mid)
            en = energy_physical_vs_pml(sol)
            zc, pc = centerline_z(sol)

            # Boundary energy: |p|² near outer edges of physical domain
            coords = sol.coords
            pv = sol.p_values
            x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
            t_xy = cfg.t_pml_xy
            margin = 0.2e-3  # 0.2mm strip inside physical boundary
            near_bdy = (
                (x > t_xy) & (x < t_xy + margin) |
                (x > cfg.Lx - t_xy - margin) & (x < cfg.Lx - t_xy) |
                (y > t_xy) & (y < t_xy + margin) |
                (y > cfg.Ly - t_xy - margin) & (y < cfg.Ly - t_xy)
            )
            bdy_energy = float(np.mean(np.abs(pv[near_bdy])**2)) if np.any(near_bdy) else 0

            results[label] = {
                "t_lam": t_lam, "sf": sf,
                "roi_mean": roi["mean_abs_p"], "roi_max": roi["max_abs_p"],
                "max_p": sol.max_pressure,
                "energy_ratio": en["ratio"],
                "bdy_energy": bdy_energy,
                "centerline_max": float(np.max(pc)),
                "time_s": sol.solver_time,
            }
            del sol; gc.collect()

    # --- Summary table ---
    report.append(f"## Step 3 — PML Stress Test\n")
    report.append(f"| PML thick | σ factor | ROI mean|p| | ROI max|p| | Boundary E | Energy ratio | max|p| |")
    report.append(f"|-----------|----------|------------|------------|------------|--------------|--------|")
    for label, r in results.items():
        report.append(f"| {r['t_lam']:.1f}λ | {r['sf']:.0f} | {r['roi_mean']:.4f} | "
                      f"{r['roi_max']:.4f} | {r['bdy_energy']:.4f} | {r['energy_ratio']:.4f} | "
                      f"{r['max_p']:.2f} |")
    report.append("")

    # --- Plot: ROI metric convergence ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for sf in sigma_factors:
        ts = [r["t_lam"] for r in results.values() if r["sf"] == sf]
        roi_means = [r["roi_mean"] for r in results.values() if r["sf"] == sf]
        roi_maxes = [r["roi_max"] for r in results.values() if r["sf"] == sf]
        bdys = [r["bdy_energy"] for r in results.values() if r["sf"] == sf]
        axes[0].plot(ts, roi_means, 'o-', label=f"σ={sf:.0f}")
        axes[1].plot(ts, roi_maxes, 'o-', label=f"σ={sf:.0f}")
        axes[2].plot(ts, bdys, 'o-', label=f"σ={sf:.0f}")
    axes[0].set_title("ROI mean|p| vs PML thickness"); axes[0].set_xlabel("PML [λ]")
    axes[1].set_title("ROI max|p| vs PML thickness"); axes[1].set_xlabel("PML [λ]")
    axes[2].set_title("Boundary energy vs PML thickness"); axes[2].set_xlabel("PML [λ]")
    for ax in axes:
        ax.legend(); ax.grid(alpha=0.3)
    fig.suptitle("Step 3: PML convergence (vortex-only, direct solve)", fontsize=13)
    fig.tight_layout()
    fig.savefig(fig_dir / "pml_convergence.png", dpi=150)
    plt.close(fig)

    # Verdict
    # Check if ROI mean converges (ratio of min to max across all runs)
    all_roi = [r["roi_mean"] for r in results.values()]
    spread = (max(all_roi) - min(all_roi)) / (np.mean(all_roi) + 1e-30)
    if spread < 0.1:
        verdict = f"PASS — ROI mean |p| spread = {spread:.1%} across PML configs"
    elif spread < 0.3:
        verdict = f"PARTIAL — ROI spread = {spread:.1%}, some sensitivity remains"
    else:
        verdict = f"WARN — ROI spread = {spread:.1%}, PML not yet converged"
    report.append(f"\n**Verdict:** {verdict}\n")
    print(f"\n  ── STEP 3 VERDICT: {verdict}")

    return results


# ═════════════════════════════════════════════════════════════════════
#  STEP 4 — Amplitude balancing (3×3 sweep)
# ═════════════════════════════════════════════════════════════════════

def step4_amplitude_balance(out_dir: Path, report: list):
    _section_banner("STEP 4 — AMPLITUDE BALANCE SWEEP")
    fig_dir = out_dir / "step4_balance" / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)

    direct_opts = {
        "ksp_type": "preonly", "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }

    # 3×3 sweep: standing amplitude vs disk amplitude
    # Current: disk=10µm/s, standing=1µm/s → ratio 10:1
    # We want to find where they're comparable in petri slab
    disk_amps    = [10e-6, 5e-6, 1e-6]     # µm/s
    stand_amps   = [1e-6,  5e-6, 10e-6]    # µm/s

    ref_cfg = FarFieldConfig(**COARSE_CFG)
    z_petri_mid = ref_cfg.H_under + ref_cfg.H_top / 2

    # First, run vortex-only and standing-only for each amplitude as reference
    # This lets us compute difference maps
    case_data = {}  # key = (disk_amp, stand_amp, case_type) → sol metrics + slice
    all_slices = {}  # for plotting

    for da in disk_amps:
        for sa in stand_amps:
            tag = f"d{da*1e6:.0f}_s{sa*1e6:.0f}"
            # Run three sub-cases: standing, vortex, combined
            for case_name, overrides in [
                ("standing", {"disk_velocity_amplitude": 0.0, "standing_velocity_amplitude": sa}),
                ("vortex", {"disk_velocity_amplitude": da, "standing_velocity_amplitude": 0.0}),
                ("combined", {"disk_velocity_amplitude": da, "standing_velocity_amplitude": sa}),
            ]:
                key = f"{tag}_{case_name}"
                print(f"\n  [{key}]")
                cfg = FarFieldConfig(**{**COARSE_CFG, **overrides})
                sol = solve_helmholtz(cfg, verbose=True, petsc_options=direct_opts)
                roi = _roi_metrics(sol, z_petri_mid)

                # Grab XY slice at petri mid
                xg, yg, pmag, pphase = slice_xy(sol, z_petri_mid, nx=150, ny=150)
                all_slices[key] = {"xg": xg, "yg": yg, "pmag": pmag, "pphase": pphase}
                case_data[key] = {
                    "roi_mean": roi["mean_abs_p"],
                    "roi_max": roi["max_abs_p"],
                    "max_p": sol.max_pressure,
                }
                del sol; gc.collect()

    # --- Plot: 3×3 grid of XY |p| at petri mid (combined) ---
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    for i, da in enumerate(disk_amps):
        for j, sa in enumerate(stand_amps):
            key = f"d{da*1e6:.0f}_s{sa*1e6:.0f}_combined"
            sl = all_slices[key]
            vmax_local = sl["pmag"].max()
            im = axes[i, j].pcolormesh(sl["xg"]*1e3, sl["yg"]*1e3, sl["pmag"],
                                       shading="auto", cmap="inferno")
            axes[i, j].set_title(f"D={da*1e6:.0f} V={sa*1e6:.0f}\nmax={vmax_local:.2f}Pa",
                                 fontsize=9)
            axes[i, j].set_aspect("equal")
            plt.colorbar(im, ax=axes[i, j], shrink=0.7)
    fig.suptitle("Step 4: |p| XY petri mid — combined (rows=disk amp, cols=stand amp)", fontsize=12)
    fig.tight_layout()
    fig.savefig(fig_dir / "balance_grid_combined.png", dpi=150)
    plt.close(fig)

    # --- Difference maps for each (da, sa) ---
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    for i, da in enumerate(disk_amps):
        for j, sa in enumerate(stand_amps):
            tag = f"d{da*1e6:.0f}_s{sa*1e6:.0f}"
            p_comb = all_slices[f"{tag}_combined"]["pmag"]
            p_vort = all_slices[f"{tag}_vortex"]["pmag"]
            diff = p_comb - p_vort
            vlim = max(abs(diff.min()), abs(diff.max()), 1e-8)
            im = axes[i, j].pcolormesh(all_slices[f"{tag}_combined"]["xg"]*1e3,
                                       all_slices[f"{tag}_combined"]["yg"]*1e3,
                                       diff, shading="auto", cmap="RdBu_r",
                                       vmin=-vlim, vmax=vlim)
            axes[i, j].set_title(f"|p_comb|-|p_vort|\nD={da*1e6:.0f} V={sa*1e6:.0f}",
                                 fontsize=9)
            axes[i, j].set_aspect("equal")
            plt.colorbar(im, ax=axes[i, j], shrink=0.7)
    fig.suptitle("Step 4: Δ|p| = |p_combined| - |p_vortex| (interaction signal)", fontsize=12)
    fig.tight_layout()
    fig.savefig(fig_dir / "difference_comb_minus_vortex.png", dpi=150)
    plt.close(fig)

    # Difference: combined - standing
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    for i, da in enumerate(disk_amps):
        for j, sa in enumerate(stand_amps):
            tag = f"d{da*1e6:.0f}_s{sa*1e6:.0f}"
            p_comb = all_slices[f"{tag}_combined"]["pmag"]
            p_stand = all_slices[f"{tag}_standing"]["pmag"]
            diff = p_comb - p_stand
            vlim = max(abs(diff.min()), abs(diff.max()), 1e-8)
            im = axes[i, j].pcolormesh(all_slices[f"{tag}_combined"]["xg"]*1e3,
                                       all_slices[f"{tag}_combined"]["yg"]*1e3,
                                       diff, shading="auto", cmap="RdBu_r",
                                       vmin=-vlim, vmax=vlim)
            axes[i, j].set_title(f"|p_comb|-|p_stand|\nD={da*1e6:.0f} V={sa*1e6:.0f}",
                                 fontsize=9)
            axes[i, j].set_aspect("equal")
            plt.colorbar(im, ax=axes[i, j], shrink=0.7)
    fig.suptitle("Step 4: Δ|p| = |p_combined| - |p_standing| (vortex contribution)", fontsize=12)
    fig.tight_layout()
    fig.savefig(fig_dir / "difference_comb_minus_standing.png", dpi=150)
    plt.close(fig)

    # --- 1D cut at trap height through center ---
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    for i, da in enumerate(disk_amps):
        for j, sa in enumerate(stand_amps):
            tag = f"d{da*1e6:.0f}_s{sa*1e6:.0f}"
            xg = all_slices[f"{tag}_combined"]["xg"]
            # Extract 1D cut through y = Ly/2 (middle row of pmag)
            ny_mid = all_slices[f"{tag}_combined"]["pmag"].shape[0] // 2
            for cn, ls, c in [("standing", "--", "blue"), ("vortex", ":", "red"),
                               ("combined", "-", "black")]:
                key = f"{tag}_{cn}"
                axes[i, j].plot(xg*1e3, all_slices[key]["pmag"][ny_mid, :],
                                ls=ls, color=c, label=cn, lw=1.2)
            axes[i, j].set_title(f"D={da*1e6:.0f} V={sa*1e6:.0f}", fontsize=9)
            axes[i, j].legend(fontsize=6)
            axes[i, j].set_xlabel("x [mm]")
    fig.suptitle("Step 4: 1D cut |p|(x) at petri mid, y=center", fontsize=12)
    fig.tight_layout()
    fig.savefig(fig_dir / "1d_cuts_petri_mid.png", dpi=150)
    plt.close(fig)

    # --- Report ---
    report.append(f"## Step 4 — Amplitude Balance Sweep\n")
    report.append(f"| Disk [µm/s] | Stand [µm/s] | ROI mean (stand) | ROI mean (vortex) | ROI mean (comb) | Δ (comb-vort) |")
    report.append(f"|------------|-------------|-----------------|------------------|----------------|--------------|")
    best_delta = 0; best_tag = ""
    for da in disk_amps:
        for sa in stand_amps:
            tag = f"d{da*1e6:.0f}_s{sa*1e6:.0f}"
            ms = case_data[f"{tag}_standing"]["roi_mean"]
            mv = case_data[f"{tag}_vortex"]["roi_mean"]
            mc = case_data[f"{tag}_combined"]["roi_mean"]
            delta = abs(mc - mv)
            if delta > best_delta:
                best_delta = delta; best_tag = tag
            report.append(f"| {da*1e6:.0f} | {sa*1e6:.0f} | {ms:.4f} | {mv:.4f} | "
                          f"{mc:.4f} | {delta:.4f} |")
    report.append(f"\n**Best interaction signal:** {best_tag} (Δ mean|p| = {best_delta:.4f} Pa)\n")
    print(f"\n  ── STEP 4: Best interaction at {best_tag}, Δ = {best_delta:.4f}")

    return case_data, all_slices, best_tag


# ═════════════════════════════════════════════════════════════════════
#  STEP 5 — Interaction metrics
# ═════════════════════════════════════════════════════════════════════

def step5_interaction_metrics(out_dir: Path, report: list,
                               all_slices: dict, best_tag: str):
    _section_banner("STEP 5 — INTERACTION METRICS")
    fig_dir = out_dir / "step5_interaction" / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Use the best amplitude combo from step 4
    p_stand = all_slices[f"{best_tag}_standing"]["pmag"]
    p_vort  = all_slices[f"{best_tag}_vortex"]["pmag"]
    p_comb  = all_slices[f"{best_tag}_combined"]["pmag"]
    xg = all_slices[f"{best_tag}_combined"]["xg"]
    yg = all_slices[f"{best_tag}_combined"]["yg"]

    # --- Δ maps ---
    delta_v = p_comb - p_vort
    delta_s = p_comb - p_stand

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    # |p_combined|
    im0 = axes[0].pcolormesh(xg*1e3, yg*1e3, p_comb, shading="auto", cmap="inferno")
    axes[0].set_title(f"|p| combined ({best_tag})")
    plt.colorbar(im0, ax=axes[0], label="Pa")

    # Δ (comb - vort)
    vlim = max(abs(delta_v.min()), abs(delta_v.max()), 1e-8)
    im1 = axes[1].pcolormesh(xg*1e3, yg*1e3, delta_v, shading="auto",
                              cmap="RdBu_r", vmin=-vlim, vmax=vlim)
    axes[1].set_title("Δ = |p_comb| - |p_vort|")
    plt.colorbar(im1, ax=axes[1], label="Pa")

    # Δ (comb - stand)
    vlim = max(abs(delta_s.min()), abs(delta_s.max()), 1e-8)
    im2 = axes[2].pcolormesh(xg*1e3, yg*1e3, delta_s, shading="auto",
                              cmap="RdBu_r", vmin=-vlim, vmax=vlim)
    axes[2].set_title("Δ = |p_comb| - |p_stand|")
    plt.colorbar(im2, ax=axes[2], label="Pa")
    for ax in axes:
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]"); ax.set_aspect("equal")
    fig.suptitle(f"Step 5: Interaction evidence ({best_tag})", fontsize=13)
    fig.tight_layout()
    fig.savefig(fig_dir / "interaction_maps.png", dpi=150)
    plt.close(fig)

    # --- Selectivity / localisation index ---
    # Define: selectivity = std(Δ) / mean(|p_comb|) in ROI
    cfg = FarFieldConfig(**COARSE_CFG)
    mask_roi = _roi_mask_xy(xg, yg, cfg)

    delta_v_roi = delta_v[mask_roi]
    delta_s_roi = delta_s[mask_roi]
    p_comb_roi = p_comb[mask_roi]

    selectivity_v = float(np.std(delta_v_roi)) / (float(np.mean(p_comb_roi)) + 1e-30)
    selectivity_s = float(np.std(delta_s_roi)) / (float(np.mean(p_comb_roi)) + 1e-30)

    # Localisation: fraction of ROI where |Δ| > mean(|Δ|)
    loc_v = float(np.mean(np.abs(delta_v_roi) > np.mean(np.abs(delta_v_roi))))
    loc_s = float(np.mean(np.abs(delta_s_roi) > np.mean(np.abs(delta_s_roi))))

    # Nonlinearity check: |p_comb|² vs |p_vort|² + |p_stand|²
    # For linear system: p_comb = p_vort + p_stand, but |p_comb| ≠ |p_vort| + |p_stand|
    # So the meaningful comparison is complex: we don't have complex slices cached simply.
    # Instead report the power: mean(|Δ|) / mean(|p_comb|)
    interaction_strength = float(np.mean(np.abs(delta_v_roi))) / (float(np.mean(p_comb_roi)) + 1e-30)

    print(f"  Selectivity (Δ_vort): {selectivity_v:.4f}")
    print(f"  Selectivity (Δ_stand): {selectivity_s:.4f}")
    print(f"  Interaction strength: {interaction_strength:.4f}")

    report.append(f"## Step 5 — Interaction Metrics\n")
    report.append(f"Using best balance: **{best_tag}**\n")
    report.append(f"| Metric | Δ(comb-vort) | Δ(comb-stand) |")
    report.append(f"|--------|-------------|--------------|")
    report.append(f"| mean |Δ| in ROI | {float(np.mean(np.abs(delta_v_roi))):.4f} | {float(np.mean(np.abs(delta_s_roi))):.4f} |")
    report.append(f"| max |Δ| in ROI | {float(np.max(np.abs(delta_v_roi))):.4f} | {float(np.max(np.abs(delta_s_roi))):.4f} |")
    report.append(f"| std(Δ) / mean(|p|) | {selectivity_v:.4f} | {selectivity_s:.4f} |")
    report.append(f"| localisation fraction | {loc_v:.3f} | {loc_s:.3f} |")
    report.append(f"| interaction strength | {interaction_strength:.4f} | — |")
    report.append("")

    return selectivity_v, interaction_strength


# ═════════════════════════════════════════════════════════════════════
#  STEP 6 — Gor'kov proxy
# ═════════════════════════════════════════════════════════════════════

def step6_gorkov(out_dir: Path, report: list, best_tag: str):
    _section_banner("STEP 6 — GOR'KOV PROXY")
    fig_dir = out_dir / "step6_gorkov" / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)

    direct_opts = {
        "ksp_type": "preonly", "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }

    # Parse best amplitude from best_tag, e.g. "d5_s5"
    parts = best_tag.split("_")
    da = float(parts[0][1:]) * 1e-6
    sa = float(parts[1][1:]) * 1e-6

    cfg_combined = FarFieldConfig(**{
        **COARSE_CFG,
        "disk_velocity_amplitude": da,
        "standing_velocity_amplitude": sa,
    })

    sol = solve_helmholtz(cfg_combined, verbose=True, petsc_options=direct_opts)

    # Gor'kov potential for a compressible sphere in an inviscid fluid:
    #   U = (4/3)πa³ [ f₁/(4ρc²) <|p|²> - 3f₂ρ/(8) <|v|²> ]
    # where f₁ = 1 - κp/κf,  f₂ = 2(ρp-ρf)/(2ρp+ρf)
    # For polystyrene in water:
    #   ρp=1050, cp=2350, κp=1/(ρp*cp²) → f₁, f₂
    #   v = -∇p / (iωρ)  →  |v| = |∇p|/(ωρ)
    # Since we use P2 elements, we can compute ∇p at DOFs.

    cfg = sol.cfg
    rho_f = cfg.rho
    c_f = cfg.c
    omega_val = cfg.omega

    # Particle properties (polystyrene)
    rho_p = 1050.0
    c_p = 2350.0
    kappa_f = 1.0 / (rho_f * c_f**2)
    kappa_p = 1.0 / (rho_p * c_p**2)
    f1 = 1.0 - kappa_p / kappa_f
    f2 = 2.0 * (rho_p - rho_f) / (2.0 * rho_p + rho_f)

    # Compute Gor'kov on XY slice at petri mid via finite differences
    z_petri_mid = cfg.H_under + cfg.H_top / 2
    nx_g, ny_g = 150, 150
    xg, yg, pmag, pphase = slice_xy(sol, z_petri_mid, nx_g, ny_g)
    dx = xg[1] - xg[0]
    dy = yg[1] - yg[0]

    # Also get complex p via re/im interpolation
    from scipy.interpolate import NearestNDInterpolator
    coords = sol.coords; pv = sol.p_values
    interp_re = NearestNDInterpolator(coords, np.real(pv))
    interp_im = NearestNDInterpolator(coords, np.imag(pv))
    X, Y = np.meshgrid(xg, yg)
    Z = np.full_like(X, z_petri_mid)
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    pc_re = interp_re(pts).reshape(X.shape)
    pc_im = interp_im(pts).reshape(X.shape)
    pc = pc_re + 1j * pc_im

    # Compute |p|² and |∇p|² on the grid
    p2 = np.abs(pc)**2

    # Gradient via central differences
    dpx_re = np.gradient(pc_re, dx, axis=1)
    dpx_im = np.gradient(pc_im, dx, axis=1)
    dpy_re = np.gradient(pc_re, dy, axis=0)
    dpy_im = np.gradient(pc_im, dy, axis=0)
    # Also need dz — get from a slightly offset z
    dz = 0.05e-3  # 50 µm step
    pts_up = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, z_petri_mid + dz)])
    pts_dn = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, z_petri_mid - dz)])
    pz_up = (interp_re(pts_up) + 1j * interp_im(pts_up)).reshape(X.shape)
    pz_dn = (interp_re(pts_dn) + 1j * interp_im(pts_dn)).reshape(X.shape)
    dpz = (pz_up - pz_dn) / (2 * dz)

    grad_p_sq = (np.abs(dpx_re + 1j*dpx_im)**2 +
                 np.abs(dpy_re + 1j*dpy_im)**2 +
                 np.abs(dpz)**2)

    # |v|² = |∇p|² / (ω²ρ²)
    v2 = grad_p_sq / (omega_val**2 * rho_f**2)

    # Gor'kov potential for different particle sizes
    particle_radii = [5e-6, 10e-6, 25e-6, 50e-6]  # 10, 20, 50, 100 µm diameter

    report.append(f"## Step 6 — Gor'kov Proxy\n")
    report.append(f"Particle: polystyrene (ρ={rho_p}, c={c_p} m/s)")
    report.append(f"- f₁ = {f1:.4f},  f₂ = {f2:.4f}")
    report.append(f"- ka range: {cfg.k * particle_radii[0]:.4f} – {cfg.k * particle_radii[-1]:.4f}")
    report.append(f"\n| a [µm] | ka | Trap depth [J] | U_max [J] | U_min [J] | Norm stiffness |")
    report.append(f"|--------|-----|-----------------|-----------|-----------|----------------|")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for idx, a in enumerate(particle_radii):
        V_part = (4.0/3.0) * np.pi * a**3
        # U = V [ f1/(4ρc²) <|p|²> - 3f2ρ/(8) <|v|²> ]
        # Time-averaged: <|p|²> = |p|²/2, <|v|²> = |v|²/2
        U = V_part * (f1 / (4 * rho_f * c_f**2) * p2 / 2
                      - 3 * f2 * rho_f / 8 * v2 / 2)

        # Trap depth
        mask = _roi_mask_xy(xg, yg, cfg)
        U_roi = U[mask]
        U_min = float(np.min(U_roi))
        U_max = float(np.max(U_roi))
        trap_depth = U_max - U_min

        # Normalised stiffness: (U_max - U_min) / |U_mean|
        U_mean = float(np.mean(np.abs(U_roi)))
        norm_stiff = trap_depth / (U_mean + 1e-40)

        ka = cfg.k * a
        report.append(f"| {a*1e6:.0f} | {ka:.4f} | {trap_depth:.2e} | "
                      f"{U_max:.2e} | {U_min:.2e} | {norm_stiff:.2f} |")

        ax = axes.flat[idx]
        im = ax.pcolormesh(xg*1e3, yg*1e3, U, shading="auto", cmap="RdBu_r")
        ax.set_title(f"U_Gorkov  a={a*1e6:.0f}µm  ka={ka:.3f}\ndepth={trap_depth:.2e} J",
                     fontsize=9)
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(f"Step 6: Gor'kov potential — polystyrene in water ({best_tag})", fontsize=12)
    fig.tight_layout()
    fig.savefig(fig_dir / "gorkov_potential.png", dpi=150)
    plt.close(fig)
    report.append("")

    # --- Size sweep: trap depth vs particle radius ---
    radii_sweep = np.logspace(np.log10(5e-6), np.log10(50e-6), 20)
    depths = []
    for a in radii_sweep:
        V_part = (4.0/3.0) * np.pi * a**3
        U = V_part * (f1 / (4 * rho_f * c_f**2) * p2 / 2
                      - 3 * f2 * rho_f / 8 * v2 / 2)
        U_roi = U[mask]
        depths.append(float(np.max(U_roi) - np.min(U_roi)))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(radii_sweep*1e6, depths, 'ko-')
    ax.set_xlabel("Particle radius a [µm]")
    ax.set_ylabel("Trap depth [J]")
    ax.set_title("Gor'kov trap depth vs particle size")
    # Mark kBT line
    kBT = 4.11e-21  # at 20°C
    ax.axhline(kBT, color='red', ls='--', label=f'kBT = {kBT:.1e} J')
    ax.axhline(10*kBT, color='orange', ls=':', label=f'10 kBT')
    ax2 = ax.twiny()
    ka_vals = cfg.k * radii_sweep
    ax2.set_xscale('log')
    ax2.set_xlim(ka_vals[0], ka_vals[-1])
    ax2.set_xlabel("ka")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "trap_depth_vs_size.png", dpi=150)
    plt.close(fig)

    del sol; gc.collect()


# ═════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════

def main():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path("results") / f"diagnostic_{stamp}"
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*70}")
    print(f"  DIAGNOSTIC PIPELINE — Steps 1-6")
    print(f"  Output: {out_root}")
    print(f"{'#'*70}\n")

    report = []
    report.append(f"# Diagnostic Pipeline Report\n")
    report.append(f"**Date:** {datetime.now().isoformat()}\n")

    # ── Step 1: Convergence ──
    try:
        step1_results = step1_convergence(out_root, report)
    except Exception as e:
        report.append(f"## Step 1 — FAILED\n```\n{traceback.format_exc()}\n```\n")
        print(f"  STEP 1 FAILED: {e}")
        step1_results = {}

    # ── Step 2: Free-space vortex ──
    try:
        step2_freespace_vortex(out_root, report)
    except Exception as e:
        report.append(f"## Step 2 — FAILED\n```\n{traceback.format_exc()}\n```\n")
        print(f"  STEP 2 FAILED: {e}")

    # ── Step 3: PML stress test ──
    try:
        step3_results = step3_pml_stress(out_root, report)
    except Exception as e:
        report.append(f"## Step 3 — FAILED\n```\n{traceback.format_exc()}\n```\n")
        print(f"  STEP 3 FAILED: {e}")

    # ── Step 4: Amplitude balancing ──
    try:
        case_data, all_slices, best_tag = step4_amplitude_balance(out_root, report)
    except Exception as e:
        report.append(f"## Step 4 — FAILED\n```\n{traceback.format_exc()}\n```\n")
        print(f"  STEP 4 FAILED: {e}")
        all_slices = {}; best_tag = "d5_s5"

    # ── Step 5: Interaction metrics ──
    if all_slices:
        try:
            step5_interaction_metrics(out_root, report, all_slices, best_tag)
        except Exception as e:
            report.append(f"## Step 5 — FAILED\n```\n{traceback.format_exc()}\n```\n")
            print(f"  STEP 5 FAILED: {e}")
    else:
        report.append(f"## Step 5 — SKIPPED (Step 4 failed)\n")

    # ── Step 6: Gor'kov ──
    try:
        step6_gorkov(out_root, report, best_tag)
    except Exception as e:
        report.append(f"## Step 6 — FAILED\n```\n{traceback.format_exc()}\n```\n")
        print(f"  STEP 6 FAILED: {e}")

    # ── Write report ──
    report_path = out_root / "REPORT.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report))
    print(f"\n  Report written to {report_path}")

    # ── Symlink ──
    latest = Path("results") / "diagnostic_latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(out_root.name)
    print(f"  Symlink: {latest} → {out_root.name}")

    print(f"\n{'='*70}")
    print(f"  DIAGNOSTIC PIPELINE COMPLETE")
    print(f"  {out_root}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
