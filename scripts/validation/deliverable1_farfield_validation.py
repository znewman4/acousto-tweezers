#!/usr/bin/env python3
"""
Deliverable 1 — Production Far-Field Validation

A) Convergence summary: DOFs, solve time, KSP reason, residual norm
B) PML sensitivity: 1λ vs 2λ PML thickness
C) Mesh sensitivity: 4 vs 5 elem/λ

Runs lens-only and standing-only at production mesh.

Output:  results/Deliverable1_FarFieldValidation/
"""
from __future__ import annotations

import json
import sys
import time
import numpy as np
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from acoustweezers.experiments.farfield_petri_cuboid.config import (
    FarFieldConfig, demo_config,
)
from acoustweezers.experiments.farfield_petri_cuboid.solve_pressure import (
    solve_helmholtz,
)
from acoustweezers.experiments.farfield_petri_cuboid.post import (
    centerline_z, slice_xy,
)

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "Deliverable1_FarFieldValidation"
FIGS = OUT / "figs"


def _save(fig, name, dpi=150):
    fig.savefig(FIGS / name, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _solve(label, **overrides):
    """Solve and return (solution, metadata dict)."""
    cfg = demo_config(**overrides)
    print(f"\n  ── Solving: {label} ──")
    print(f"     elem/λ={cfg.elements_per_wavelength}  "
          f"PML_xy={cfg.pml_n_wavelengths_xy:.1f}λ  "
          f"PML_z={cfg.pml_n_wavelengths_z:.1f}λ")
    t0 = time.time()
    sol = solve_helmholtz(cfg, verbose=True)
    dt = time.time() - t0
    meta = {
        "label": label,
        "DOFs": sol.dofs,
        "max_p_Pa": sol.max_pressure,
        "solve_time_s": round(dt, 1),
        "ksp_reason": sol.ksp_converged_reason,
        "elements_per_wavelength": cfg.elements_per_wavelength,
        "pml_n_wavelengths_xy": cfg.pml_n_wavelengths_xy,
        "pml_n_wavelengths_z": cfg.pml_n_wavelengths_z,
    }
    return sol, meta


# ═════════════════════════════════════════════════════════════════════
#  A — Convergence Summary (lens-only + standing-only at 5 elem/λ)
# ═════════════════════════════════════════════════════════════════════

def part_A(results):
    print("\n══════ Part A: Convergence Summary ══════")

    # Lens-only
    sol_lens, m_lens = _solve(
        "lens-only 5ew",
        standing_velocity_amplitude=0.0,
        top_impedance_Zrel=1.0,
        elements_per_wavelength=5,
    )
    results["A_lens"] = m_lens

    # Standing-only
    sol_stand, m_stand = _solve(
        "standing-only 5ew",
        disk_velocity_amplitude=0.0,
        elements_per_wavelength=5,
    )
    results["A_standing"] = m_stand

    return sol_lens, sol_stand


# ═════════════════════════════════════════════════════════════════════
#  B — PML Sensitivity Check (lens-only, 1λ vs 2λ)
# ═════════════════════════════════════════════════════════════════════

def part_B(results):
    print("\n══════ Part B: PML Sensitivity ══════")

    sols = {}
    for n_wl in [1.0, 2.0]:
        label = f"PML {n_wl:.0f}λ"
        sol, meta = _solve(
            label,
            standing_velocity_amplitude=0.0,
            top_impedance_Zrel=1.0,
            elements_per_wavelength=5,
            pml_n_wavelengths_xy=n_wl,
            pml_n_wavelengths_z=n_wl,
        )
        sols[n_wl] = (sol, meta)

    sol1, m1 = sols[1.0]
    sol2, m2 = sols[2.0]

    # ROI mean |p| in a FIXED inner box that's physical in BOTH runs.
    # The 2λ PML eats more of the box, so we must use the tighter mask.
    cfg2 = sol2.cfg
    t_xy_max = cfg2.t_pml_xy   # larger PML thickness
    t_z_max = cfg2.t_pml_z

    def _inner_mask(sol, t_xy, t_z):
        c = sol.coords
        return ((c[:, 0] > t_xy) & (c[:, 0] < sol.cfg.Lx - t_xy) &
                (c[:, 1] > t_xy) & (c[:, 1] < sol.cfg.Ly - t_xy) &
                (c[:, 2] > t_z))

    mask1 = _inner_mask(sol1, t_xy_max, t_z_max)
    mask2 = _inner_mask(sol2, t_xy_max, t_z_max)
    roi_1 = float(np.mean(np.abs(sol1.p_values[mask1])))
    roi_2 = float(np.mean(np.abs(sol2.p_values[mask2])))
    rel_diff = abs(roi_1 - roi_2) / max(roi_1, roi_2) * 100

    # Also compare global max|p|
    max_p1 = m1["max_p_Pa"]
    max_p2 = m2["max_p_Pa"]
    rel_diff_peak = abs(max_p1 - max_p2) / max(max_p1, max_p2) * 100

    # Centerline comparison
    zg1, cl1 = centerline_z(sol1, nz=500)
    zg2, cl2 = centerline_z(sol2, nz=500)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(zg1 * 1e3, cl1, label=f"PML 1λ  (DOFs={m1['DOFs']})")
    ax.plot(zg2 * 1e3, cl2, '--', label=f"PML 2λ  (DOFs={m2['DOFs']})")
    ax.set_xlabel("z [mm]")
    ax.set_ylabel("|p| [Pa]")
    ax.set_title("PML Sensitivity — Centerline |p|(z)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, "B_pml_sensitivity_centerline.png")

    results["B_pml_sensitivity"] = {
        "pml_1lam": m1,
        "pml_2lam": m2,
        "roi_mean_abs_p_1lam": roi_1,
        "roi_mean_abs_p_2lam": roi_2,
        "relative_difference_roi_pct": round(rel_diff, 2),
        "max_p_1lam": max_p1,
        "max_p_2lam": max_p2,
        "relative_difference_peak_pct": round(rel_diff_peak, 2),
        "status": "PASS" if rel_diff_peak < 15 else "FAIL",
    }
    print(f"    max|p| 1λ: {max_p1:.2f} Pa    2λ: {max_p2:.2f} Pa  (peak diff {rel_diff_peak:.2f}%)")
    print(f"    ROI mean|p| 1λ: {roi_1:.4f} Pa   2λ: {roi_2:.4f} Pa  (diff {rel_diff:.2f}%)")
    print(f"    Status: {results['B_pml_sensitivity']['status']}")


def _physical_mask(sol):
    """Boolean mask for P2 DOFs in the physical (non-PML) region."""
    cfg = sol.cfg
    coords = sol.coords
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    t_xy = cfg.t_pml_xy
    t_z = cfg.t_pml_z
    return ((x > t_xy) & (x < cfg.Lx - t_xy) &
            (y > t_xy) & (y < cfg.Ly - t_xy) &
            (z > t_z))


# ═════════════════════════════════════════════════════════════════════
#  C — Mesh Sensitivity (lens-only, 4 vs 5 elem/λ)
# ═════════════════════════════════════════════════════════════════════

def part_C(results):
    print("\n══════ Part C: Mesh Sensitivity ══════")

    sols = {}
    for epw in [4, 5]:
        label = f"Mesh {epw}ew"
        sol, meta = _solve(
            label,
            standing_velocity_amplitude=0.0,
            top_impedance_Zrel=1.0,
            elements_per_wavelength=epw,
        )
        sols[epw] = (sol, meta)

    sol4, m4 = sols[4]
    sol5, m5 = sols[5]

    zg4, cl4 = centerline_z(sol4, nz=500)
    zg5, cl5 = centerline_z(sol5, nz=500)

    # Peak pressure comparison — use GLOBAL max|p| (solver-reported)
    # Centerline goes through the vortex null where |p|→0, making it
    # misleading for convergence.  Global max|p| is the stable metric.
    max_p4 = m4["max_p_Pa"]
    max_p5 = m5["max_p_Pa"]
    rel_diff_peak = abs(max_p4 - max_p5) / max(max_p4, max_p5) * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(zg4 * 1e3, cl4, label=f"4 elem/λ  max|p|={max_p4:.2f} Pa")
    ax.plot(zg5 * 1e3, cl5, '--', label=f"5 elem/λ  max|p|={max_p5:.2f} Pa")
    ax.set_xlabel("z [mm]")
    ax.set_ylabel("|p| [Pa]")
    ax.set_title("Mesh Sensitivity — Centerline |p|(z)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, "C_mesh_sensitivity_centerline.png")

    results["C_mesh_sensitivity"] = {
        "mesh_4ew": m4,
        "mesh_5ew": m5,
        "max_p_4ew": max_p4,
        "max_p_5ew": max_p5,
        "relative_diff_peak_pct": round(rel_diff_peak, 2),
        "status": "PASS" if rel_diff_peak < 15 else "FAIL",
    }
    print(f"    max|p| at 4 elem/λ: {max_p4:.2f} Pa (DOFs={m4['DOFs']})")
    print(f"    max|p| at 5 elem/λ: {max_p5:.2f} Pa (DOFs={m5['DOFs']})")
    print(f"    Relative diff:      {rel_diff_peak:.2f}%")
    print(f"    Status: {results['C_mesh_sensitivity']['status']}")


# ═════════════════════════════════════════════════════════════════════
#  REPORT
# ═════════════════════════════════════════════════════════════════════

def _build_report(results, ts):
    lines = [
        "# Deliverable 1 — Production Far-Field Validation\n",
        f"Generated: {ts}\n\n",
        "## A: Convergence Summary\n\n",
        "| Case | DOFs | max|p| [Pa] | Solve time [s] | KSP reason |\n",
        "|------|------|-------------|----------------|------------|\n",
    ]
    for key in ["A_lens", "A_standing"]:
        if key in results:
            r = results[key]
            lines.append(
                f"| {r['label']} | {r['DOFs']} | {r['max_p_Pa']:.2f} | "
                f"{r['solve_time_s']:.1f} | {r['ksp_reason']} |\n"
            )
    lines.append("\n")

    if "B_pml_sensitivity" in results:
        b = results["B_pml_sensitivity"]
        lines.append("## B: PML Sensitivity\n\n")
        lines.append(f"- max|p| at 1λ PML: {b['max_p_1lam']:.2f} Pa\n")
        lines.append(f"- max|p| at 2λ PML: {b['max_p_2lam']:.2f} Pa\n")
        lines.append(f"- Peak relative difference: **{b['relative_difference_peak_pct']:.2f}%**\n")
        lines.append(f"- ROI mean|p| at 1λ: {b['roi_mean_abs_p_1lam']:.4f} Pa\n")
        lines.append(f"- ROI mean|p| at 2λ: {b['roi_mean_abs_p_2lam']:.4f} Pa\n")
        lines.append(f"- ROI relative difference: {b['relative_difference_roi_pct']:.2f}%\n")
        lines.append(f"- Status: **{b['status']}**\n\n")
        lines.append("![PML Sensitivity](figs/B_pml_sensitivity_centerline.png)\n\n")

    if "C_mesh_sensitivity" in results:
        c = results["C_mesh_sensitivity"]
        lines.append("## C: Mesh Sensitivity\n\n")
        lines.append(f"- max|p| at 4 elem/λ: {c['max_p_4ew']:.2f} Pa "
                      f"(DOFs={c['mesh_4ew']['DOFs']})\n")
        lines.append(f"- max|p| at 5 elem/λ: {c['max_p_5ew']:.2f} Pa "
                      f"(DOFs={c['mesh_5ew']['DOFs']})\n")
        lines.append(f"- Peak relative diff: **{c['relative_diff_peak_pct']:.2f}%**\n")
        lines.append(f"- Status: **{c['status']}**\n\n")
        lines.append("![Mesh Sensitivity](figs/C_mesh_sensitivity_centerline.png)\n\n")

    lines.append("## Summary\n\n")
    n_pass = sum(1 for k in results if isinstance(results[k], dict) and
                 results[k].get("status") == "PASS")
    n_fail = sum(1 for k in results if isinstance(results[k], dict) and
                 results[k].get("status") == "FAIL")
    lines.append(f"- **PASS: {n_pass}**\n")
    lines.append(f"- **FAIL: {n_fail}**\n")

    return "".join(lines)


# ═════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════

def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUT.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  DELIVERABLE 1 — PRODUCTION FAR-FIELD VALIDATION")
    print(f"  Output: {OUT}")
    print(f"  Time:   {ts}")
    print(f"{'='*70}")

    results = {"timestamp": ts}

    part_A(results)
    part_B(results)
    part_C(results)

    # Write outputs
    def _conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o

    with open(OUT / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=_conv)

    with open(OUT / "REPORT.md", "w") as f:
        f.write(_build_report(results, ts))

    n_pass = sum(1 for k in results if isinstance(results[k], dict) and
                 results[k].get("status") == "PASS")
    n_fail = sum(1 for k in results if isinstance(results[k], dict) and
                 results[k].get("status") == "FAIL")

    print(f"\n{'='*70}")
    print(f"  DELIVERABLE 1 COMPLETE — PASS: {n_pass}  FAIL: {n_fail}")
    print(f"  Report:  {OUT / 'REPORT.md'}")
    print(f"  JSON:    {OUT / 'results.json'}")
    print(f"  Figures: {FIGS}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
