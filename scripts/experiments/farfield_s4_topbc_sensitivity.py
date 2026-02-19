#!/usr/bin/env python3
"""
B1: Top BC sensitivity diagnostic — quantifies when top BC matters.

Sweep:
  - H_under ∈ {1, 2, 3} mm
  - pml_enabled ∈ {True, False}
  - top_bc_type ∈ {"impedance", "dirichlet"}

For each (H_under, pml_enabled), compares impedance vs dirichlet and
reports %diff in max|p| and centerline max.

Usage:
    micromamba run -n acousto-complex python scripts/experiments/farfield_s4_topbc_sensitivity.py

Outputs: results/farfield_topbc_sensitivity_<timestamp>/
"""
from __future__ import annotations

import sys, csv, gc, json, time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from dataclasses import replace
from acoustweezers.experiments.farfield_petri_cuboid.config import FarFieldConfig
from acoustweezers.experiments.farfield_petri_cuboid.solve_pressure import solve_helmholtz
from acoustweezers.experiments.farfield_petri_cuboid.post import centerline_z

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT = Path("results") / f"farfield_topbc_sensitivity_{stamp}"
OUT.mkdir(parents=True, exist_ok=True)

# Use 4 elem/λ for speed — qualitative sweep
base_cfg = FarFieldConfig(
    Lx=6e-3, Ly=6e-3, H_top=1e-3,
    frequency_hz=2.0e6, disk_radius=1.0e-3,
    disk_velocity_amplitude=10e-6, vortex_topological_charge=1,
    standing_velocity_amplitude=1e-6, standing_phase_pattern="antiphase",
    standing_axis="both", top_impedance_Zrel=0.001,
    pml_n_wavelengths_xy=1.0, pml_n_wavelengths_z=1.0,
    pml_degree=2, pml_sigma_max_factor=5.0,
    elements_per_wavelength=4, lens_drive="ideal",
)

H_under_values = [1e-3, 2e-3, 3e-3]
# PML-off cases diverge (5000 iters GMRES, see A2 operator check).
# Restrict to PML-on — the production configuration.
pml_values = [True]

print(f"\n{'='*70}")
print("B1: TOP BC SENSITIVITY SWEEP")
print(f"{'='*70}\n")

rows = []
results = {}  # (H_under, pml) -> {impedance: ..., dirichlet: ...}

for H_u in H_under_values:
    for pml_on in pml_values:
        for bc_type in ["impedance", "dirichlet"]:
            key = (H_u, pml_on)
            cfg = replace(base_cfg,
                          H_under=H_u, pml_enabled=pml_on,
                          top_bc_type=bc_type)
            label = f"H={H_u*1e3:.0f}mm pml={pml_on} bc={bc_type}"
            print(f"  ▶ {label}")

            sol = solve_helmholtz(cfg, verbose=False)
            zc, pc = centerline_z(sol, nz=200)
            max_p = sol.max_pressure
            cl_max = float(pc.max())
            ksp_its = sol.ksp_iterations

            if key not in results:
                results[key] = {}
            results[key][bc_type] = {
                "max_p": max_p, "cl_max": cl_max, "ksp_its": ksp_its,
                "zc": zc, "pc": pc,
            }

            rows.append({
                "H_under_mm": f"{H_u*1e3:.0f}",
                "pml_enabled": pml_on,
                "top_bc": bc_type,
                "max_p_Pa": f"{max_p:.4f}",
                "cl_max_Pa": f"{cl_max:.4f}",
                "ksp_iters": ksp_its,
                "ksp_reason": sol.ksp_converged_reason,
            })

            del sol; gc.collect()
            print(f"    max|p|={max_p:.2f}  cl_max={cl_max:.2f}  iters={ksp_its}")

# Compute %diff for each (H_under, pml) pair
diff_rows = []
for (H_u, pml_on), bcs in results.items():
    imp = bcs["impedance"]
    diri = bcs["dirichlet"]
    d_max = abs(imp["max_p"] - diri["max_p"]) / max(imp["max_p"], diri["max_p"], 1e-30) * 100
    d_cl = abs(imp["cl_max"] - diri["cl_max"]) / max(imp["cl_max"], diri["cl_max"], 1e-30) * 100
    diff_rows.append({
        "H_under_mm": f"{H_u*1e3:.0f}",
        "pml_enabled": pml_on,
        "diff_max_p_%": f"{d_max:.2f}",
        "diff_cl_max_%": f"{d_cl:.2f}",
        "imp_max_p": f"{imp['max_p']:.4f}",
        "dir_max_p": f"{diri['max_p']:.4f}",
    })
    print(f"  H={H_u*1e3:.0f}mm pml={pml_on}: "
          f"max|p| diff={d_max:.2f}%  cl_max diff={d_cl:.2f}%")

# ── CSV ──
with open(OUT / "topbc_sensitivity.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

with open(OUT / "topbc_diff_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=diff_rows[0].keys())
    w.writeheader()
    w.writerows(diff_rows)

# ── Plot: %diff vs H_under ──
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, pml_on, title in zip(axes, [True, False],
                               ["PML enabled", "PML disabled (rigid)"]):
    hs, ds_max, ds_cl = [], [], []
    for dr in diff_rows:
        if dr["pml_enabled"] == pml_on:
            hs.append(float(dr["H_under_mm"]))
            ds_max.append(float(dr["diff_max_p_%"]))
            ds_cl.append(float(dr["diff_cl_max_%"]))
    ax.plot(hs, ds_max, "bo-", label="%diff max|p|")
    ax.plot(hs, ds_cl, "rs--", label="%diff cl_max")
    ax.set_xlabel("H_under [mm]")
    ax.set_ylabel("Impedance vs Dirichlet %diff")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

fig.suptitle("B1: Top BC Sensitivity — when does the top BC matter?", fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "topbc_sensitivity.png", dpi=150)
plt.close(fig)

# ── Symlink ──
latest = Path("results") / "farfield_topbc_sensitivity_latest"
if latest.is_symlink() or latest.exists():
    latest.unlink()
latest.symlink_to(OUT.name)

print(f"\n  Output: {OUT}")
print(f"  Files:  topbc_sensitivity.csv  topbc_diff_summary.csv  topbc_sensitivity.png")
