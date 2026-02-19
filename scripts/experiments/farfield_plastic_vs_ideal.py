#!/usr/bin/env python3
"""
E1: Plastic vs ideal lens comparison on the same mesh.

Runs two solves (plastic lens + ideal vortex) with identical mesh and
compares max|p|, centerline, and 2D slices.

Usage:
    micromamba run -n acousto-complex python scripts/experiments/farfield_plastic_vs_ideal.py

Outputs: results/farfield_plastic_vs_ideal_<timestamp>/
"""
from __future__ import annotations

import sys, csv, gc, json
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from dataclasses import replace
from acoustweezers.experiments.farfield_petri_cuboid.config import FarFieldConfig
from acoustweezers.experiments.farfield_petri_cuboid.solve_pressure import solve_helmholtz
from acoustweezers.experiments.farfield_petri_cuboid.post import (
    centerline_z, slice_xy, slice_xz,
    export_slice_xy, export_slice_xz,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT = Path("results") / f"farfield_plastic_vs_ideal_{stamp}"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "figs").mkdir(exist_ok=True)
(OUT / "slices").mkdir(exist_ok=True)

# Base config (plastic lens, 4 elem/λ for speed)
cfg_plastic = FarFieldConfig(
    Lx=6e-3, Ly=6e-3, H_under=3e-3, H_top=1e-3,
    frequency_hz=2.0e6, disk_radius=1.0e-3,
    disk_velocity_amplitude=10e-6, vortex_topological_charge=1,
    standing_velocity_amplitude=1e-6, standing_phase_pattern="antiphase",
    standing_axis="both", top_bc_type="impedance", top_impedance_Zrel=0.001,
    pml_n_wavelengths_xy=1.0, pml_n_wavelengths_z=1.0,
    pml_degree=2, pml_sigma_max_factor=5.0,
    pml_enabled=True, elements_per_wavelength=4,
    lens_drive="plastic", lens_l=1, lens_focal_length=10e-3,
    lens_focus_offset_x=0.2e-3, lens_focus_offset_y=0.0,
    lens_c_lens=2700.0, lens_apodization="cosine_taper",
)
cfg_ideal = replace(cfg_plastic, lens_drive="ideal")

print(f"\n{'='*70}")
print("E1: PLASTIC vs IDEAL LENS COMPARISON")
print(f"{'='*70}\n")

# Save config
with open(OUT / "config.json", "w") as f:
    json.dump(cfg_plastic.to_dict(), f, indent=2, default=str)

# ── Solve 1: Plastic ──
print("▶ Solve 1: PLASTIC lens")
sol_p = solve_helmholtz(cfg_plastic, verbose=True)
zc_p, pc_p = centerline_z(sol_p)
p_max = sol_p.max_pressure
p_cl = float(pc_p.max())
p_ksp = sol_p.ksp_iterations

# Slices
z_xy = cfg_plastic.H_under + 0.5 * cfg_plastic.H_top  # mid-petri
export_slice_xy(sol_p, z_xy, OUT / "slices" / "plastic_xy.npz")
export_slice_xz(sol_p, cfg_plastic.Ly / 2, OUT / "slices" / "plastic_xz.npz")
xg_p, yg_p, pm_xy_p, _ = slice_xy(sol_p, z_xy)
xg_xz, zg_xz, pm_xz_p, _ = slice_xz(sol_p, cfg_plastic.Ly / 2)

del sol_p; gc.collect()

# ── Solve 2: Ideal ──
print("▶ Solve 2: IDEAL vortex")
sol_i = solve_helmholtz(cfg_ideal, verbose=True)
zc_i, pc_i = centerline_z(sol_i)
i_max = sol_i.max_pressure
i_cl = float(pc_i.max())
i_ksp = sol_i.ksp_iterations

export_slice_xy(sol_i, z_xy, OUT / "slices" / "ideal_xy.npz")
export_slice_xz(sol_i, cfg_ideal.Ly / 2, OUT / "slices" / "ideal_xz.npz")
_, _, pm_xy_i, _ = slice_xy(sol_i, z_xy)
_, _, pm_xz_i, _ = slice_xz(sol_i, cfg_ideal.Ly / 2)

# ── Comparison ──
diff_max = abs(p_max - i_max) / max(p_max, i_max, 1e-30) * 100
diff_cl = abs(p_cl - i_cl) / max(p_cl, i_cl, 1e-30) * 100

print(f"\n{'='*70}")
print("PLASTIC vs IDEAL COMPARISON")
print(f"{'='*70}")
print(f"  max|p|:  plastic={p_max:.2f} Pa  ideal={i_max:.2f} Pa  diff={diff_max:.1f}%")
print(f"  cl_max:  plastic={p_cl:.2f} Pa   ideal={i_cl:.2f} Pa   diff={diff_cl:.1f}%")
print(f"  KSP iters: plastic={p_ksp}  ideal={i_ksp}")

# ── CSV ──
with open(OUT / "comparison.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["metric", "plastic", "ideal", "diff_%"])
    w.writerow(["max_p_Pa", f"{p_max:.4f}", f"{i_max:.4f}", f"{diff_max:.2f}"])
    w.writerow(["cl_max_Pa", f"{p_cl:.4f}", f"{i_cl:.4f}", f"{diff_cl:.2f}"])
    w.writerow(["ksp_iters", p_ksp, i_ksp, ""])

# ── Plots ──
# 1) Centerline
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(zc_p * 1e3, pc_p, "b-", lw=1.5, label=f"Plastic (max={p_cl:.2f})")
ax.plot(zc_i * 1e3, pc_i, "r--", lw=1.2, label=f"Ideal (max={i_cl:.2f})")
ax.axvline(cfg_plastic.H_under * 1e3, color="cyan", ls=":", lw=0.8, label="petri base")
ax.set_xlabel("z [mm]"); ax.set_ylabel("|p| [Pa]")
ax.set_title(f"Centerline |p| — plastic vs ideal (diff={diff_cl:.1f}%)")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "figs" / "centerline_comparison.png", dpi=150)
plt.close(fig)

# 2) Side-by-side XY slice
vmax_xy = max(pm_xy_p.max(), pm_xy_i.max())
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
im0 = axes[0].pcolormesh(xg_p * 1e3, yg_p * 1e3, pm_xy_p, shading="auto",
                           cmap="inferno", vmin=0, vmax=vmax_xy)
axes[0].set_title(f"|p| plastic — z={z_xy*1e3:.2f}mm")
axes[0].set_aspect("equal")
plt.colorbar(im0, ax=axes[0], label="Pa")
im1 = axes[1].pcolormesh(xg_p * 1e3, yg_p * 1e3, pm_xy_i, shading="auto",
                           cmap="inferno", vmin=0, vmax=vmax_xy)
axes[1].set_title(f"|p| ideal — z={z_xy*1e3:.2f}mm")
axes[1].set_aspect("equal")
plt.colorbar(im1, ax=axes[1], label="Pa")
for ax in axes:
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
fig.suptitle("E1: XY slice — plastic vs ideal", fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "figs" / "xy_comparison.png", dpi=150)
plt.close(fig)

# 3) Side-by-side XZ slice
vmax_xz = max(pm_xz_p.max(), pm_xz_i.max())
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
im0 = axes[0].pcolormesh(xg_xz * 1e3, zg_xz * 1e3, pm_xz_p, shading="auto",
                           cmap="inferno", vmin=0, vmax=vmax_xz)
axes[0].set_title("|p| plastic — xz midplane")
plt.colorbar(im0, ax=axes[0], label="Pa")
im1 = axes[1].pcolormesh(xg_xz * 1e3, zg_xz * 1e3, pm_xz_i, shading="auto",
                           cmap="inferno", vmin=0, vmax=vmax_xz)
axes[1].set_title("|p| ideal — xz midplane")
plt.colorbar(im1, ax=axes[1], label="Pa")
for ax in axes:
    ax.set_xlabel("x [mm]"); ax.set_ylabel("z [mm]")
    ax.axhline(cfg_plastic.H_under * 1e3, color="cyan", ls=":", lw=0.8)
fig.suptitle("E1: XZ slice — plastic vs ideal", fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "figs" / "xz_comparison.png", dpi=150)
plt.close(fig)

del sol_i; gc.collect()

# ── Symlink ──
latest = Path("results") / "farfield_plastic_vs_ideal_latest"
if latest.is_symlink() or latest.exists():
    latest.unlink()
latest.symlink_to(OUT.name)

print(f"\n  Output: {OUT}")
print(f"  Files:  comparison.csv  figs/centerline_comparison.png")
print(f"          figs/xy_comparison.png  figs/xz_comparison.png")
print(f"          slices/{{plastic,ideal}}_{{xy,xz}}.npz")
