#!/usr/bin/env python3
"""
STANDING-WAVE RESONANCE SWEEP
==============================

Fine H_top sweep to find vertical resonance for maximum standing wave
strength.  The key physics:

  - Top face: Robin BC with Z_air ≈ 412 Pa·s/m  (nearly pressure-release)
  - Bottom: Hard wall (Neumann)  → pressure antinode at z=0
  - Vertical resonance: H_total ≈ (2m-1)·λ/4  (quarter-wave cavity)

The default H_top=2.000mm gives H_total=5.000mm which is very close to
the m=14 mode at 5.0085mm (off by only 0.011λ).  This explains why the
baseline already has high Mz.

This script sweeps H_top finely (stepping through quarter-wave intervals)
to confirm the resonance peaks and find the absolute optimum.
"""

from __future__ import annotations
import sys, os, time, json, gc, csv
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

NTHREADS = str(min(os.cpu_count() or 4, 14))
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, NTHREADS)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = PROJECT_ROOT / "results" / f"resonance_sweep_{TIMESTAMP}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import NearestNDInterpolator

from acoustweezers.experiments.farfield_petri_cuboid.presets import (
    CORRECTED_PRESET, PETSC_MUMPS,
)
from acoustweezers.experiments.farfield_petri_cuboid.config import FarFieldConfig
from acoustweezers.experiments.farfield_petri_cuboid.solve_pressure import solve_helmholtz
from acoustweezers.experiments.farfield_petri_cuboid.post import energy_physical_vs_pml

DPI = 300
ELEM_PER_LAMBDA = 4
NLINE = 500

PETSC_OPTS = {
    **PETSC_MUMPS,
    "mat_mumps_icntl_14": "100",
    "mat_mumps_icntl_23": "0",
    "mat_mumps_icntl_28": "2",
    "mat_mumps_icntl_29": "2",
}


class LightSol:
    def __init__(self, sol):
        self.coords = sol.coords.copy()
        self.p_values = sol.p_values.copy()
        self.cfg = sol.cfg
        self.dofs = sol.dofs
        self.ksp_converged_reason = sol.ksp_converged_reason
        self.max_pressure = sol.max_pressure


def solve_standing_only(H_top_m, label=""):
    """Solve standing-only case with given H_top, return metrics dict."""
    overrides = {
        **CORRECTED_PRESET,
        "H_top": H_top_m,
        "standing_velocity_amplitude": 10e-6,
        "disk_velocity_amplitude": 0.0,
        "elements_per_wavelength": ELEM_PER_LAMBDA,
    }
    cfg = FarFieldConfig(**overrides)

    print(f"\n  [{label}]  H_top={H_top_m*1e3:.4f}mm  H_total={cfg.H_total*1e3:.4f}mm")

    t0 = time.time()
    sol = solve_helmholtz(cfg, verbose=False, petsc_options=PETSC_OPTS,
                          export_fields=False)
    dt = time.time() - t0

    energy = energy_physical_vs_pml(sol)

    # Compute modulation metrics on centerline z
    coords = sol.coords.copy()
    pv = sol.p_values.copy()

    # Centerline |p| along z
    ire = NearestNDInterpolator(coords, np.real(pv))
    iim = NearestNDInterpolator(coords, np.imag(pv))
    zg = np.linspace(0, cfg.H_total, NLINE)
    pts = np.column_stack([np.full(NLINE, cfg.Lx/2),
                           np.full(NLINE, cfg.Ly/2), zg])
    pz = ire(pts) + 1j * iim(pts)
    pmag_z = np.abs(pz)
    re_z = np.real(pz)

    mean_z = np.mean(pmag_z) + 1e-30
    Mz = (np.max(pmag_z) - np.min(pmag_z)) / mean_z

    # x-line at trap plane
    trap_z = cfg.H_under + cfg.H_top / 2
    t_xy = cfg.t_pml_xy
    xg = np.linspace(t_xy, cfg.Lx - t_xy, NLINE)
    pts_x = np.column_stack([xg, np.full(NLINE, cfg.Ly/2),
                             np.full(NLINE, trap_z)])
    px = ire(pts_x) + 1j * iim(pts_x)
    pmag_x = np.abs(px)
    mean_x = np.mean(pmag_x) + 1e-30
    Mx = (np.max(pmag_x) - np.min(pmag_x)) / mean_x

    # Nodal spacing from Re(p) zero crossings on centerline
    sign_changes = np.where(np.diff(np.sign(re_z)))[0]
    if len(sign_changes) >= 2:
        zc = 0.5 * (zg[sign_changes] + zg[sign_changes + 1])
        spacings = np.diff(zc)
        nodal_spacing_z = float(np.median(spacings))
    else:
        nodal_spacing_z = float("nan")

    max_p = float(np.max(np.abs(pv)))

    del sol; gc.collect()

    m = {
        "H_top_mm": H_top_m * 1e3,
        "H_total_mm": cfg.H_total * 1e3,
        "max_p_Pa": max_p,
        "mean_p_Pa": float(np.mean(np.abs(pv))),
        "Mz": round(Mz, 4),
        "Mx": round(Mx, 4),
        "nodal_spacing_z_mm": round(nodal_spacing_z * 1e3, 4) if not np.isnan(nodal_spacing_z) else None,
        "energy_phys_pml_ratio": energy.get("ratio", None),
        "solve_time_s": round(dt, 1),
        "DOFs": cfg.mesh_nx * cfg.mesh_ny * cfg.mesh_nz * 6,  # rough
    }
    print(f"    max|p|={max_p:.2f}  Mz={Mz:.4f}  Mx={Mx:.4f}  "
          f"nodal_z={nodal_spacing_z*1e3:.4f}mm  time={dt:.0f}s")
    return m


def main():
    lam = 1484.0 / 2e6   # 0.742 mm
    H_under = 3e-3

    print("=" * 72)
    print("STANDING-WAVE RESONANCE SWEEP")
    print(f"Output: {OUT_DIR}")
    print(f"λ = {lam*1e3:.4f} mm,  λ/4 = {lam/4*1e3:.4f} mm")
    print("=" * 72)

    # Quarter-wave resonances: H_total = (2m-1)*λ/4
    # For H_top near 2mm (m=13,14,15): H_total = 4.638, 5.009, 5.380 mm
    # Also sweep near the exact default (H_top=2mm, H_total=5.0mm)

    # Fine sweep: from H_top = 1.4 to 2.6 mm in steps of λ/16
    step = lam / 16  # = 0.04638 mm
    H_top_min = 1.4e-3
    H_top_max = 2.6e-3
    H_top_values = np.arange(H_top_min, H_top_max + step/2, step)

    # Also add the exact resonant H_top values
    for m in [13, 14, 15]:
        H_res = (2 * m - 1) * lam / 4
        H_top_res = H_res - H_under
        if H_top_min <= H_top_res <= H_top_max:
            H_top_values = np.append(H_top_values, H_top_res)

    # Add the exact default
    if H_top_min <= 2e-3 <= H_top_max:
        H_top_values = np.append(H_top_values, 2e-3)

    H_top_values = np.sort(np.unique(np.round(H_top_values, decimals=7)))
    print(f"\nSweeping {len(H_top_values)} H_top values from "
          f"{H_top_values[0]*1e3:.4f} to {H_top_values[-1]*1e3:.4f} mm")

    rows = []
    for i, H_top in enumerate(H_top_values):
        label = f"{i+1}/{len(H_top_values)}"
        m = solve_standing_only(H_top, label)
        rows.append(m)

    # Write CSV
    csv_path = OUT_DIR / "resonance_sweep.csv"
    keys = rows[0].keys()
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {csv_path}")

    # ── Plot: max|p| and Mz vs H_top ──
    H_vals = np.array([r["H_top_mm"] for r in rows])
    max_p = np.array([r["max_p_Pa"] for r in rows])
    Mz_vals = np.array([r["Mz"] for r in rows])
    Mx_vals = np.array([r["Mx"] for r in rows])

    # Mark quarter-wave resonances
    qw_Htop = []
    for m in range(10, 20):
        H_res = (2 * m - 1) * lam / 4
        h_top = (H_res - H_under) * 1e3
        if H_top_min * 1e3 <= h_top <= H_top_max * 1e3:
            qw_Htop.append((m, h_top))

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    ax = axes[0]
    ax.plot(H_vals, max_p, "ko-", ms=4, lw=1.2)
    ax.set_ylabel("max|p| [Pa]")
    ax.set_title("Standing Wave Resonance Sweep (H_top)")
    for m, h in qw_Htop:
        ax.axvline(h, color="red", ls=":", lw=0.7, alpha=0.6)
        ax.text(h, ax.get_ylim()[1]*0.92, f"m={m}", fontsize=7,
                ha="center", color="red")
    ax.axvline(2.0, color="blue", ls="--", lw=0.8, alpha=0.5, label="default H_top=2mm")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(H_vals, Mz_vals, "rs-", ms=4, lw=1.2, label="Mz (z-modulation)")
    ax.plot(H_vals, Mx_vals, "b^-", ms=3, lw=1.0, label="Mx (x-modulation)")
    ax.set_ylabel("Modulation depth")
    for m, h in qw_Htop:
        ax.axvline(h, color="red", ls=":", lw=0.7, alpha=0.6)
    ax.axvline(2.0, color="blue", ls="--", lw=0.8, alpha=0.5)
    ax.legend(fontsize=8)

    ax = axes[2]
    ns_vals = np.array([r["nodal_spacing_z_mm"] if r["nodal_spacing_z_mm"] is not None
                        else np.nan for r in rows])
    ax.plot(H_vals, ns_vals, "g+-", ms=4, lw=1.0, label="nodal spacing (z)")
    ax.axhline(lam / 2 * 1e3, color="gray", ls="--", lw=0.8, label=f"λ/2 = {lam/2*1e3:.4f} mm")
    ax.set_ylabel("Nodal spacing z [mm]")
    ax.set_xlabel("H_top [mm]")
    for m, h in qw_Htop:
        ax.axvline(h, color="red", ls=":", lw=0.7, alpha=0.6)
    ax.axvline(2.0, color="blue", ls="--", lw=0.8, alpha=0.5)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "resonance_sweep.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    # ── Find the best ──
    best_idx = np.argmax(max_p)
    best = rows[best_idx]
    print(f"\n{'='*72}")
    print(f"BEST: H_top = {best['H_top_mm']:.4f} mm  "
          f"(H_total = {best['H_total_mm']:.4f} mm)")
    print(f"  max|p| = {best['max_p_Pa']:.2f} Pa")
    print(f"  Mz = {best['Mz']:.4f},  Mx = {best['Mx']:.4f}")
    print(f"  nodal_z = {best['nodal_spacing_z_mm']} mm  (λ/2 = {lam/2*1e3:.4f} mm)")

    # Find best by Mz
    best_mz_idx = np.argmax(Mz_vals)
    best_mz = rows[best_mz_idx]
    print(f"\nBest Mz: H_top = {best_mz['H_top_mm']:.4f} mm  Mz = {best_mz['Mz']:.4f}")

    with open(OUT_DIR / "best_config.json", "w") as f:
        json.dump({"best_max_p": best, "best_Mz": rows[best_mz_idx]}, f, indent=2)

    print(f"\n{'='*72}")
    print(f"Output: {OUT_DIR}")
    print(f"{'='*72}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
