#!/usr/bin/env python3
"""
A2: PML operator check — proves PML actually affects the assembled operator.

Solves PML-enabled vs PML-disabled (rigid walls) on the SAME mesh geometry
and compares:
  - max|p|, centerline max, energy partition
  - Acceptance: >10% difference in at least one scalar metric

Usage:
    micromamba run -n acousto-complex python scripts/experiments/farfield_pml_operator_check.py

Outputs:  results/audit_farfield_operator_<timestamp>/
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
from acoustweezers.experiments.farfield_petri_cuboid.post import (
    centerline_z, energy_physical_vs_pml, slice_xz,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT = Path("results") / f"audit_farfield_operator_{stamp}"
OUT.mkdir(parents=True, exist_ok=True)

# Use 4 elem/λ for speed (qualitative check — only need >10% difference)
cfg_pml = FarFieldConfig(
    Lx=6e-3, Ly=6e-3, H_under=3e-3, H_top=1e-3,
    frequency_hz=2.0e6, disk_radius=1.0e-3,
    disk_velocity_amplitude=10e-6, vortex_topological_charge=1,
    standing_velocity_amplitude=1e-6, standing_phase_pattern="antiphase",
    standing_axis="both", top_bc_type="impedance", top_impedance_Zrel=0.001,
    pml_n_wavelengths_xy=1.0, pml_n_wavelengths_z=1.0,
    pml_degree=2, pml_sigma_max_factor=5.0,
    pml_enabled=True, elements_per_wavelength=4,
    lens_drive="ideal",
)
cfg_rigid = replace(cfg_pml, pml_enabled=False)

print(f"\n{'='*70}")
print("A2: PML OPERATOR CHECK — PML vs Rigid")
print(f"{'='*70}")

# ── Solve 1: PML ──
print("\n▶ Solve 1: PML enabled")
sol_pml = solve_helmholtz(cfg_pml, verbose=True)
zc_pml, pc_pml = centerline_z(sol_pml)
en_pml = energy_physical_vs_pml(sol_pml)
pml_max = sol_pml.max_pressure
pml_cl = float(pc_pml.max())
pml_ksp = sol_pml.ksp_iterations
pml_reason = sol_pml.ksp_converged_reason

del sol_pml; gc.collect()

# ── Solve 2: Rigid ──
print("\n▶ Solve 2: Rigid walls (PML disabled)")
sol_rig = solve_helmholtz(cfg_rigid, verbose=True)
zc_rig, pc_rig = centerline_z(sol_rig)
en_rig = energy_physical_vs_pml(sol_rig)
rig_max = sol_rig.max_pressure
rig_cl = float(pc_rig.max())
rig_ksp = sol_rig.ksp_iterations
rig_reason = sol_rig.ksp_converged_reason

# ── Comparison ──
diff_max = abs(pml_max - rig_max) / max(pml_max, rig_max, 1e-30) * 100
diff_cl = abs(pml_cl - rig_cl) / max(pml_cl, rig_cl, 1e-30) * 100
pass_check = diff_max > 10 or diff_cl > 10

print(f"\n{'='*70}")
print("PML OPERATOR CHECK RESULTS")
print(f"{'='*70}")
print(f"  max|p|:  PML={pml_max:.2f} Pa  Rigid={rig_max:.2f} Pa  diff={diff_max:.1f}%")
print(f"  cl_max:  PML={pml_cl:.2f} Pa   Rigid={rig_cl:.2f} Pa   diff={diff_cl:.1f}%")
print(f"  Energy ratio:  PML={en_pml['ratio']:.4f}  Rigid={en_rig['ratio']:.6f}")
print(f"  KSP iters: PML={pml_ksp}  Rigid={rig_ksp}")
print(f"  ACCEPTANCE: {'PASS' if pass_check else 'FAIL'} (need >10% diff in at least one)")

# ── CSV ──
with open(OUT / "summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["case", "max_p_Pa", "cl_max_Pa", "energy_phys", "energy_pml",
                "energy_ratio", "ksp_iters", "ksp_reason"])
    w.writerow(["pml", f"{pml_max:.4f}", f"{pml_cl:.4f}",
                f"{en_pml['physical']:.4e}", f"{en_pml['pml']:.4e}",
                f"{en_pml['ratio']:.6f}", pml_ksp, pml_reason])
    w.writerow(["rigid", f"{rig_max:.4f}", f"{rig_cl:.4f}",
                f"{en_rig['physical']:.4e}", f"{en_rig['pml']:.4e}",
                f"{en_rig['ratio']:.6f}", rig_ksp, rig_reason])
    w.writerow(["diff_%", f"{diff_max:.1f}", f"{diff_cl:.1f}", "", "", "", "", ""])
    w.writerow(["acceptance", "PASS" if pass_check else "FAIL", "", "", "", "", "", ""])

# ── Plots ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Centerline
axes[0].plot(zc_pml * 1e3, pc_pml, "b-", lw=1.5, label="PML")
axes[0].plot(zc_rig * 1e3, pc_rig, "r--", lw=1.2, label="Rigid")
axes[0].axvline(cfg_pml.t_pml_z * 1e3, color="gray", ls=":", lw=0.7, label="PML-z top")
axes[0].axvline(cfg_pml.H_under * 1e3, color="cyan", ls=":", lw=0.8, label="petri base")
axes[0].set_xlabel("z [mm]"); axes[0].set_ylabel("|p| [Pa]")
axes[0].set_title(f"|p| centerline — diff={diff_cl:.1f}%")
axes[0].legend()

# Energy bars
labels = ["PML\nPhys", "PML\nPML", "Rigid\nPhys", "Rigid\nPML"]
vals = [en_pml["physical"], en_pml["pml"], en_rig["physical"], en_rig["pml"]]
colors = ["steelblue", "salmon", "steelblue", "salmon"]
axes[1].bar(labels, vals, color=colors)
axes[1].set_ylabel("Σ|p|² (DOF sum)")
axes[1].set_title("Energy partition")

fig.suptitle(f"A2: PML Operator Check — {'PASS' if pass_check else 'FAIL'}", fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "pml_operator_check.png", dpi=150)
plt.close(fig)

del sol_rig; gc.collect()

# ── Symlink ──
latest = Path("results") / "audit_farfield_operator_latest"
if latest.is_symlink() or latest.exists():
    latest.unlink()
latest.symlink_to(OUT.name)

print(f"\n  Output: {OUT}")
print(f"  Files:  summary.csv  pml_operator_check.png")
