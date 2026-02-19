#!/usr/bin/env python3
"""
Far-field vortex + standing-wave demo at 2 MHz with PML.

Builds a tall cuboid domain (Lx × Ly × H_total) with:
  - bottom disk: plastic lens drive (focused vortex) or ideal vortex
  - petri-slab side walls: standing-wave Neumann patches
  - PML on sides + bottom (outside disk column)
  - top face: water–air impedance Robin BC

Runs TWO solves:
  1) PML enabled  (production)
  2) PML disabled (rigid walls) — for reflection-reduction comparison

Saves diagnostics under  results/farfield_vortex_standing_<timestamp>/

Usage:
    micromamba run -n acousto-complex python scripts/experiments/farfield_vortex_plus_standing.py
    micromamba run -n acousto-complex python scripts/experiments/farfield_vortex_plus_standing.py --ideal

Author: Acousto-Tweezers Project
Date: 2026-02-16
"""

from __future__ import annotations

import json
import sys
import time
import csv
import gc
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

# ── project imports ───────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from acoustweezers.experiments.farfield_petri_cuboid.config import FarFieldConfig
from acoustweezers.experiments.farfield_petri_cuboid.solve_pressure import solve_helmholtz
from acoustweezers.experiments.farfield_petri_cuboid.post import (
    plot_all_diagnostics,
    energy_physical_vs_pml,
    centerline_z,
    slice_xz,
)


def parse_args():
    p = argparse.ArgumentParser(description="Far-field PML demo")
    p.add_argument("--ideal", action="store_true",
                   help="Use ideal vortex drive instead of plastic lens")
    p.add_argument("--rtol", type=float, default=1e-4,
                   help="GMRES relative tolerance (default: 1e-4)")
    p.add_argument("--restart", type=int, default=200,
                   help="GMRES restart (default: 200)")
    p.add_argument("--maxit", type=int, default=5000,
                   help="GMRES max iterations (default: 5000)")
    p.add_argument("--fast", action="store_true",
                   help="Use fast (4 elem/λ) for qualitative runs")
    return p.parse_args()


def main():
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path("results") / f"farfield_vortex_standing_{stamp}"
    out_root.mkdir(parents=True, exist_ok=True)
    fig_dir = out_root / "figs"
    csv_dir = out_root / "csv"
    fig_dir.mkdir(exist_ok=True)
    csv_dir.mkdir(exist_ok=True)

    lens_mode = "ideal" if args.ideal else "plastic"
    epw = 4 if args.fast else 5

    print(f"\n{'#'*70}")
    print(f"  FAR-FIELD VORTEX + STANDING — 2 MHz PML DEMO")
    print(f"  Lens drive: {lens_mode}   elem/λ: {epw}")
    print(f"  Solver: GMRES({args.restart}), rtol={args.rtol}, maxit={args.maxit}")
    print(f"  Output: {out_root}")
    print(f"{'#'*70}\n")

    # ── config ────────────────────────────────────────────────────────
    cfg = FarFieldConfig(
        Lx=6e-3,
        Ly=6e-3,
        H_under=3e-3,
        H_top=1e-3,
        frequency_hz=2.0e6,
        disk_radius=1.0e-3,
        disk_velocity_amplitude=10e-6,
        vortex_topological_charge=1,
        standing_velocity_amplitude=1e-6,
        standing_phase_pattern="antiphase",
        standing_axis="both",
        top_bc_type="impedance",
        top_impedance_Zrel=0.001,
        pml_n_wavelengths_xy=1.0,
        pml_n_wavelengths_z=1.0,
        pml_degree=2,
        pml_sigma_max_factor=5.0,
        pml_enabled=True,
        elements_per_wavelength=epw,
        # Plastic lens parameters
        lens_drive=lens_mode,
        lens_l=1,
        lens_focal_length=10e-3,
        lens_focus_offset_x=0.2e-3,
        lens_focus_offset_y=0.0,
        lens_c_lens=2700.0,
        lens_apodization="cosine_taper",
        lens_apodization_strength=1.0,
    )

    petsc_opts = {
        "ksp_type": "gmres",
        "ksp_rtol": args.rtol,
        "ksp_max_it": args.maxit,
        "ksp_gmres_restart": args.restart,
        "pc_type": "ilu",
    }

    print(cfg.describe())
    config_dict = cfg.to_dict()
    config_dict["petsc_options"] = petsc_opts
    with open(out_root / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2, default=str)

    # ══════════════════════════════════════════════════════════════════
    #  SOLVE 1:  PML enabled
    # ══════════════════════════════════════════════════════════════════
    print("\n▶ SOLVE 1: PML ENABLED")
    sol_pml = solve_helmholtz(cfg, verbose=True, petsc_options=petsc_opts)

    z_levels, en_pml = plot_all_diagnostics(sol_pml, out_root, label="pml")
    print(f"  Energy partition (PML on):  physical={en_pml['physical']:.3e}  "
          f"pml={en_pml['pml']:.3e}  ratio={en_pml['ratio']:.4f}")

    # Cache data for comparison plots
    zc_pml, pc_pml = centerline_z(sol_pml)
    xg_p, zg_p, pm_pml_xz, _ = slice_xz(sol_pml, cfg.Ly / 2)
    pml_max_p = sol_pml.max_pressure
    pml_dofs = sol_pml.dofs
    pml_time = sol_pml.solver_time

    # Compute per-slice metrics
    pml_slice_metrics = {}
    for name, zv in z_levels.items():
        _, _, pm_tmp, _ = slice_xz(sol_pml, cfg.Ly / 2, nx=50, nz=50)
        zg_tmp = np.linspace(0, cfg.H_total, 50)
        iz = np.argmin(np.abs(zg_tmp - zv))
        pml_slice_metrics[name] = float(pm_tmp[iz, :].max())

    # Free memory
    del sol_pml; gc.collect()

    # ══════════════════════════════════════════════════════════════════
    #  SOLVE 2:  Rigid walls (PML off)
    # ══════════════════════════════════════════════════════════════════
    print("\n▶ SOLVE 2: RIGID WALLS (PML disabled)")
    from dataclasses import replace
    cfg_rigid = replace(cfg, pml_enabled=False)
    sol_rigid = solve_helmholtz(cfg_rigid, verbose=True, petsc_options=petsc_opts)

    z_levels_r, en_rigid = plot_all_diagnostics(sol_rigid, out_root, label="rigid")
    print(f"  Energy partition (rigid):  physical={en_rigid['physical']:.3e}  "
          f"pml={en_rigid['pml']:.3e}  ratio={en_rigid['ratio']:.4f}")

    # ══════════════════════════════════════════════════════════════════
    #  COMPARISON PLOTS
    # ══════════════════════════════════════════════════════════════════
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    zc_rig, pc_rig = centerline_z(sol_rigid)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(zc_pml * 1e3, pc_pml, "b-", label="PML enabled")
    ax.plot(zc_rig * 1e3, pc_rig, "r--", alpha=0.7, label="Rigid walls (no PML)")
    ax.axvline(cfg.t_pml_z * 1e3, color="gray", ls=":", lw=0.7, label="PML-z limit")
    ax.axvline(cfg.H_under * 1e3, color="cyan", ls=":", lw=0.8, label="petri base")
    ax.set_xlabel("z [mm]"); ax.set_ylabel("|p| [Pa]")
    ax.set_title(f"Centerline |p| — PML vs Rigid ({lens_mode} lens)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "comparison_centerline.png", dpi=150)
    plt.close(fig)

    xg_r, zg_r, pm_rig_xz, _ = slice_xz(sol_rigid, cfg_rigid.Ly / 2)
    vmax = max(pm_pml_xz.max(), pm_rig_xz.max())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    im0 = axes[0].pcolormesh(xg_p * 1e3, zg_p * 1e3, pm_pml_xz, shading="auto",
                              cmap="inferno", vmin=0, vmax=vmax)
    axes[0].set_title("|p| — PML enabled")
    axes[0].set_xlabel("x [mm]"); axes[0].set_ylabel("z [mm]")
    plt.colorbar(im0, ax=axes[0], label="Pa")
    im1 = axes[1].pcolormesh(xg_r * 1e3, zg_r * 1e3, pm_rig_xz, shading="auto",
                              cmap="inferno", vmin=0, vmax=vmax)
    axes[1].set_title("|p| — Rigid walls")
    axes[1].set_xlabel("x [mm]"); axes[1].set_ylabel("z [mm]")
    plt.colorbar(im1, ax=axes[1], label="Pa")
    fig.suptitle(f"xz mid-plane comparison ({lens_mode} lens)", fontsize=13)
    fig.tight_layout()
    fig.savefig(fig_dir / "comparison_xz.png", dpi=150)
    plt.close(fig)

    # ══════════════════════════════════════════════════════════════════
    #  CSV summary
    # ══════════════════════════════════════════════════════════════════
    rows = []
    r_pml = {
        "case": "pml", "lens": lens_mode,
        "DOFs": pml_dofs,
        "solver_time_s": f"{pml_time:.1f}",
        "max_p_Pa": f"{pml_max_p:.2f}",
        "sigma_max": f"{cfg.sigma_max:.2e}",
        "t_pml_xy_mm": f"{cfg.t_pml_xy*1e3:.3f}",
        "t_pml_z_mm": f"{cfg.t_pml_z*1e3:.3f}",
        "energy_physical": f"{en_pml['physical']:.4e}",
        "energy_pml": f"{en_pml['pml']:.4e}",
        "energy_ratio": f"{en_pml['ratio']:.5f}",
    }
    for name, val in pml_slice_metrics.items():
        r_pml[f"max_p_{name}_Pa"] = f"{val:.2f}"
    rows.append(r_pml)

    r_rig = {
        "case": "rigid", "lens": lens_mode,
        "DOFs": sol_rigid.dofs,
        "solver_time_s": f"{sol_rigid.solver_time:.1f}",
        "max_p_Pa": f"{sol_rigid.max_pressure:.2f}",
        "sigma_max": "0",
        "t_pml_xy_mm": "0",
        "t_pml_z_mm": "0",
        "energy_physical": f"{en_rigid['physical']:.4e}",
        "energy_pml": f"{en_rigid['pml']:.4e}",
        "energy_ratio": f"{en_rigid['ratio']:.5f}",
    }
    for name, zv in z_levels.items():
        _, _, pm_tmp, _ = slice_xz(sol_rigid, cfg.Ly / 2, nx=50, nz=50)
        zg_tmp = np.linspace(0, cfg.H_total, 50)
        iz = np.argmin(np.abs(zg_tmp - zv))
        r_rig[f"max_p_{name}_Pa"] = f"{pm_tmp[iz, :].max():.2f}"
    rows.append(r_rig)

    summary_path = csv_dir / "summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # ── final report ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"  Output folder:  {out_root}")
    print(f"  Lens drive:     {lens_mode}")
    print(f"  PML run:")
    print(f"    max|p| = {pml_max_p:.2f} Pa")
    print(f"    DOFs   = {pml_dofs}")
    print(f"    time   = {pml_time:.1f} s")
    print(f"  Rigid run:")
    print(f"    max|p| = {sol_rigid.max_pressure:.2f} Pa")
    print(f"    DOFs   = {sol_rigid.dofs}")
    print(f"    time   = {sol_rigid.solver_time:.1f} s")
    print(f"  Reflection reduction:")
    print(f"    Centerline max (PML):   {pc_pml.max():.2f} Pa")
    print(f"    Centerline max (rigid): {pc_rig.max():.2f} Pa")
    print(f"    Energy ratio PML/phys:  {en_pml['ratio']:.4f}")
    print(f"    Energy ratio rigid:     {en_rigid['ratio']:.4f}")
    print(f"{'='*70}\n")

    # symlink
    latest = Path("results") / "farfield_latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(out_root.name)
    print(f"  Symlink: {latest} → {out_root.name}")

    return out_root


if __name__ == "__main__":
    main()
