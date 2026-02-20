#!/usr/bin/env python3
"""
Diagnostics — Interaction (Standing + Vortex)
==============================================

Runs three cases: standing_only, vortex_only, combined.

Produces:
    - Matched-scale |p| maps (XY at petri mid-plane)
    - Delta(|p|^2) maps:  |p_comb|^2 - |p_stand|^2 - |p_vort|^2
    - Localisation metric  (fraction of ROI with |Delta| > 5% of max)
    - Trap shift proxy     (shift of |p| minimum relative to standing)
    - Barrier reduction proxy (change in barrier height)

Output saved to:
    ~/OneDrive - University of Bristol/Major Project Onedrive/
      Research/Vortex 3D visualisation/Diagnostics_Interaction/

Usage:
    micromamba run -n fenicsx python scripts/validation/diagnostics_interaction.py
"""
from __future__ import annotations

import json
import sys
import gc
import csv
import numpy as np
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from acoustweezers.experiments.farfield_petri_cuboid.config import FarFieldConfig
from acoustweezers.experiments.farfield_petri_cuboid.solve_pressure import solve_helmholtz
from acoustweezers.experiments.farfield_petri_cuboid.post import (
    slice_xy, slice_xz, centerline_z, energy_physical_vs_pml,
)

ONEDRIVE_BASE = Path.home() / (
    "OneDrive - University of Bristol/Major Project Onedrive/"
    "Research/Vortex 3D visualisation"
)
OUTPUT_DIR = ONEDRIVE_BASE / "Diagnostics_Interaction"

PETSC_OPTS = {
    "ksp_type": "preonly", "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "mat_mumps_icntl_14": "80",
    "mat_mumps_icntl_23": "4000",
}

BASE_CFG = dict(
    Lx=6e-3, Ly=6e-3, H_under=3e-3, H_top=1e-3,
    frequency_hz=2.0e6, disk_radius=1.0e-3,
    disk_velocity_amplitude=1e-6,
    vortex_topological_charge=1,
    standing_velocity_amplitude=10e-6,
    standing_phase_pattern="antiphase", standing_axis="both",
    top_bc_type="impedance", top_impedance_Zrel=0.001,
    pml_n_wavelengths_xy=1.0, pml_n_wavelengths_z=1.0,
    pml_degree=2, pml_sigma_max_factor=5.0, pml_enabled=True,
    elements_per_wavelength=3,
    lens_drive="plastic", lens_l=1,
    lens_focal_length=10e-3,
    lens_focus_offset_x=0.2e-3, lens_focus_offset_y=0.0,
    lens_c_lens=2700.0, lens_apodization="cosine_taper",
    lens_apodization_strength=1.0,
)

CASES = {
    "standing_only": dict(disk_velocity_amplitude=0.0),
    "vortex_only":   dict(standing_velocity_amplitude=0.0),
    "combined":      {},
}


def _roi_mask(xg, yg, cfg):
    t_xy = cfg.t_pml_xy
    X, Y = np.meshgrid(xg, yg)
    return (X >= t_xy) & (X <= cfg.Lx - t_xy) & (Y >= t_xy) & (Y <= cfg.Ly - t_xy)


def _find_trap_minimum(pmag, xg, yg, mask):
    """Find (x, y) of the global |p| minimum within the ROI."""
    pmag_masked = np.where(mask, pmag, np.inf)
    idx = np.unravel_index(np.argmin(pmag_masked), pmag_masked.shape)
    return yg[idx[0]], xg[idx[1]]


def _barrier_height(pmag, mask):
    """Proxy barrier = max(|p|) - min(|p|) in ROI."""
    vals = pmag[mask]
    return float(np.max(vals) - np.min(vals))


def main():
    print(f"\n{'#'*70}")
    print(f"  DIAGNOSTICS — INTERACTION")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'#'*70}\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ref_cfg = FarFieldConfig(**BASE_CFG)
    z_petri_mid = ref_cfg.H_under + ref_cfg.H_top / 2

    solutions = {}
    slices = {}
    metrics = {}

    # ── Solve each case ───────────────────────────────────────────────
    for case_name, overrides in CASES.items():
        case_params = {**BASE_CFG, **overrides}

        # Try 5 elem/lambda first, fall back to 3
        for epw in [5, 3]:
            case_params["elements_per_wavelength"] = epw
            cfg = FarFieldConfig(**case_params)
            print(f"\n  CASE: {case_name} ({epw} elem/lambda)")
            try:
                sol = solve_helmholtz(cfg, verbose=True, petsc_options=PETSC_OPTS)
                break
            except Exception as e:
                print(f"    Failed: {e}")
                gc.collect()
        else:
            raise RuntimeError(f"All resolutions failed for {case_name}")

        solutions[case_name] = sol
        xg, yg, pmag, pphase = slice_xy(sol, z_petri_mid)
        slices[case_name] = {"xg": xg, "yg": yg, "pmag": pmag, "pphase": pphase}

        mask = _roi_mask(xg, yg, cfg)
        trap_y, trap_x = _find_trap_minimum(pmag, xg, yg, mask)
        bh = _barrier_height(pmag, mask)

        metrics[case_name] = {
            "max_p_global": float(sol.max_pressure),
            "mean_p_roi": float(np.mean(pmag[mask])),
            "max_p_roi": float(np.max(pmag[mask])),
            "min_p_roi": float(np.min(pmag[mask])),
            "trap_x_mm": float(trap_x * 1e3),
            "trap_y_mm": float(trap_y * 1e3),
            "barrier_height": bh,
            "dofs": sol.dofs,
            "ksp_reason": sol.ksp_converged_reason,
        }

        del sol
        gc.collect()

    # ── Matched-scale |p| maps ────────────────────────────────────────
    vmax = max(s["pmag"].max() for s in slices.values())
    xg = slices["standing_only"]["xg"]
    yg = slices["standing_only"]["yg"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, cn in zip(axes, CASES):
        im = ax.pcolormesh(xg*1e3, yg*1e3, slices[cn]["pmag"],
                           cmap="inferno", shading="auto", vmin=0, vmax=vmax)
        ax.set_title(f"|p| — {cn}")
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
        ax.set_aspect("equal")
    plt.colorbar(im, ax=axes.tolist(), label="Pa", shrink=0.8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "matched_scale_pmag.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: matched_scale_pmag.png")

    # ── Delta(|p|^2) maps ─────────────────────────────────────────────
    p2_stand = slices["standing_only"]["pmag"]**2
    p2_vort  = slices["vortex_only"]["pmag"]**2
    p2_comb  = slices["combined"]["pmag"]**2

    # Interaction term: |p_comb|^2 - |p_stand|^2 - |p_vort|^2
    # For linear superposition, this should be zero; nonzero shows interaction
    delta_p2 = p2_comb - p2_stand - p2_vort

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    dmax = max(np.abs(delta_p2).max(), 1e-20)

    # |p_comb|^2 - |p_stand|^2
    d1 = p2_comb - p2_stand
    dmax1 = max(np.abs(d1).max(), 1e-20)
    im0 = axes[0].pcolormesh(xg*1e3, yg*1e3, d1, cmap="coolwarm",
                              shading="auto", vmin=-dmax1, vmax=dmax1)
    axes[0].set_title("Delta|p|^2 (comb - stand)")
    axes[0].set_aspect("equal")
    plt.colorbar(im0, ax=axes[0], label="Pa^2")

    # |p_comb|^2 - |p_vort|^2
    d2 = p2_comb - p2_vort
    dmax2 = max(np.abs(d2).max(), 1e-20)
    im1 = axes[1].pcolormesh(xg*1e3, yg*1e3, d2, cmap="coolwarm",
                              shading="auto", vmin=-dmax2, vmax=dmax2)
    axes[1].set_title("Delta|p|^2 (comb - vort)")
    axes[1].set_aspect("equal")
    plt.colorbar(im1, ax=axes[1], label="Pa^2")

    # Interaction: |p_comb|^2 - |p_stand|^2 - |p_vort|^2
    im2 = axes[2].pcolormesh(xg*1e3, yg*1e3, delta_p2, cmap="coolwarm",
                              shading="auto", vmin=-dmax, vmax=dmax)
    axes[2].set_title("Interaction: |p_c|^2 - |p_s|^2 - |p_v|^2")
    axes[2].set_aspect("equal")
    plt.colorbar(im2, ax=axes[2], label="Pa^2")

    for ax in axes:
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "delta_p2_maps.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: delta_p2_maps.png")

    # ── Localisation metric ───────────────────────────────────────────
    mask = _roi_mask(xg, yg, ref_cfg)
    d_comb_stand = np.abs(p2_comb - p2_stand)
    threshold = 0.05 * np.max(d_comb_stand[mask])
    localisation = float(np.sum(d_comb_stand[mask] > threshold) / max(np.sum(mask), 1))

    # ── Trap shift proxy ──────────────────────────────────────────────
    stand_trap_x = metrics["standing_only"]["trap_x_mm"]
    stand_trap_y = metrics["standing_only"]["trap_y_mm"]
    comb_trap_x = metrics["combined"]["trap_x_mm"]
    comb_trap_y = metrics["combined"]["trap_y_mm"]
    trap_shift_mm = np.sqrt((comb_trap_x - stand_trap_x)**2
                            + (comb_trap_y - stand_trap_y)**2)

    # ── Barrier reduction proxy ───────────────────────────────────────
    bh_stand = metrics["standing_only"]["barrier_height"]
    bh_comb = metrics["combined"]["barrier_height"]
    barrier_reduction = (bh_stand - bh_comb) / bh_stand if bh_stand > 0 else 0.0

    interaction_metrics = {
        "localisation_fraction": localisation,
        "trap_shift_mm": float(trap_shift_mm),
        "barrier_standing": bh_stand,
        "barrier_combined": bh_comb,
        "barrier_reduction_frac": float(barrier_reduction),
        "delta_p2_max": float(np.max(np.abs(delta_p2[mask]))),
        "delta_p2_mean": float(np.mean(np.abs(delta_p2[mask]))),
    }

    # ── Per-case CSV ──────────────────────────────────────────────────
    csv_path = OUTPUT_DIR / "case_metrics.csv"
    rows = [{"case": cn, **m} for cn, m in metrics.items()]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {csv_path.resolve()}")

    # ── REPORT.md ─────────────────────────────────────────────────────
    lines = [
        "# Diagnostics — Interaction", "",
        f"**Date:** {datetime.now().isoformat()}", "",
        "## Per-Case Metrics (petri mid-plane ROI)", "",
        "| Case | max|p| global | mean|p| ROI | barrier | trap (x,y) mm |",
        "|------|--------------|-------------|---------|---------------|",
    ]
    for cn in CASES:
        m = metrics[cn]
        lines.append(
            f"| {cn} | {m['max_p_global']:.2f} | {m['mean_p_roi']:.4f} | "
            f"{m['barrier_height']:.4f} | ({m['trap_x_mm']:.2f}, {m['trap_y_mm']:.2f}) |"
        )
    lines.extend([
        "", "## Interaction Metrics", "",
        f"- **Localisation fraction:** {localisation:.3f}",
        f"  (fraction of ROI area where |Delta|p|^2| > 5% of max)",
        f"- **Trap shift:** {trap_shift_mm:.3f} mm",
        f"  (distance between trap minimum in standing_only vs combined)",
        f"- **Barrier reduction:** {barrier_reduction*100:.1f}%",
        f"  (standing barrier {bh_stand:.4f} -> combined {bh_comb:.4f})",
        f"- **max |Delta|p|^2|:** {interaction_metrics['delta_p2_max']:.6e}",
        f"- **mean |Delta|p|^2|:** {interaction_metrics['delta_p2_mean']:.6e}",
        "", "## Files", "",
        "- `matched_scale_pmag.png` — |p| for all 3 cases, same colourscale",
        "- `delta_p2_maps.png` — Delta(|p|^2) maps",
        "- `case_metrics.csv` — per-case ROI metrics",
        "",
    ])
    (OUTPUT_DIR / "REPORT.md").write_text("\n".join(lines))
    print(f"  Saved: REPORT.md")

    # ── results.json ──────────────────────────────────────────────────
    results = {
        "date": datetime.now().isoformat(),
        "cases": metrics,
        "interaction": interaction_metrics,
    }
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"  INTERACTION DIAGNOSTICS COMPLETE")
    print(f"  Output: {OUTPUT_DIR.resolve()}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
