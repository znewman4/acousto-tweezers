#!/usr/bin/env python3
"""
Phase 2.3C — Lens performance sweep.

Sweeps focal length and aperture radius independently for each
lens family (ideal, axicon) and measures:

  - focal z location
  - peak |p|
  - focal spot radius (half-power)

Generates:
  - metric-vs-parameter plots
  - comparison CSV
  - summary JSON

Output → results/deliverables/lens/
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lib.fem_cache_utils import load_fem_cache, LAM, C_WATER, F_HZ
from scripts.lib.asm_utils import (
    make_grid_from_fem, make_lens_phase, propagate_asm, K0,
)

OUT = PROJECT_ROOT / "results" / "deliverables" / "lens"
OUT.mkdir(parents=True, exist_ok=True)

_mm = lambda v: v * 1e3

# ══════════════════════════════════════════════════════════════════
# Parameters
# ══════════════════════════════════════════════════════════════════
NX, NY = 300, 300               # slightly coarser for speed on sweeps
AXICON_ANGLE = 15.0

# Sweep 1: focal length (ideal lens), fixed aperture
SWEEP_F_VALUES = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]  # mm
SWEEP_F_R_AP = 2.5e-3                                     # fixed aperture

# Sweep 2: aperture radius (ideal lens), fixed focal length
SWEEP_R_VALUES = [1.0, 1.5, 2.0, 2.5]                    # mm
SWEEP_R_F = 5.0e-3                                        # fixed focal length

# XZ sampling for focal-z detection
Z_MIN, Z_MAX = 0.5e-3, 12.0e-3
NZ = 80


def _build_source(XX, YY, family, focal_length, r_ap, axicon_deg=15.0):
    """Uniform disk + lens phase → converging source."""
    cx = 0.5 * (XX.min() + XX.max())
    cy = 0.5 * (YY.min() + YY.max())
    r = np.sqrt((XX - cx)**2 + (YY - cy)**2)
    amp = np.ones_like(XX, dtype=complex)
    amp[r > r_ap] = 0.0

    phi = make_lens_phase(
        XX, YY,
        focal_length=focal_length,
        aperture_radius=r_ap,
        family=family,
        k=K0,
        charge=0,
        axicon_angle_deg=axicon_deg,
    )
    return amp * np.exp(-1j * phi)


def _measure(source, dx, dy, x, y, z_arr, iy_c, ix_c):
    """Compute focal z, peak |p|, and spot radius from XZ scan."""
    nz = len(z_arr)
    on_axis = np.zeros(nz)
    peak_per_z = np.zeros(nz)

    for iz, z_val in enumerate(z_arr):
        p = propagate_asm(source, dx, dy, wavelength=LAM, z=z_val)
        mag = np.abs(p)
        on_axis[iz] = mag[iy_c, ix_c]
        peak_per_z[iz] = mag.max()

    # Focal z = z of max on-axis value
    i_foc = int(np.argmax(on_axis))
    focal_z = float(z_arr[i_foc])

    # Re-propagate at focal z for metrics
    p_foc = propagate_asm(source, dx, dy, wavelength=LAM, z=focal_z)
    mag_foc = np.abs(p_foc)
    peak_p = float(mag_foc.max())

    # Half-power spot radius
    half = peak_p * 0.5
    cx = 0.5 * (x[0] + x[-1])
    cy = 0.5 * (y[0] + y[-1])
    XX, YY = np.meshgrid(x, y)
    r_grid = np.sqrt((XX - cx)**2 + (YY - cy)**2)
    mask = mag_foc >= half
    spot_r = float(r_grid[mask].max()) if mask.any() else float(r_grid.max())

    return focal_z, peak_p, spot_r


def main() -> None:
    print("=" * 60)
    print("Phase 2.3C — Lens performance sweep")
    print("=" * 60)

    cache = load_fem_cache()
    grid = make_grid_from_fem(cache, nx=NX, ny=NY)
    x, y = grid["x"], grid["y"]
    XX, YY = grid["XX"], grid["YY"]
    dx, dy = grid["dx"], grid["dy"]
    iy_c, ix_c = NY // 2, NX // 2

    z_arr = np.linspace(Z_MIN, Z_MAX, NZ)
    print(f"Grid: {NX}×{NY}  dx={dx*1e6:.1f} µm  NZ={NZ}")

    all_rows = []

    # ══════════════════════════════════════════════════════════════
    # Sweep 1 — focal length (ideal lens)
    # ══════════════════════════════════════════════════════════════
    print(f"\n── Sweep 1: focal length (ideal, R_ap={SWEEP_F_R_AP*1e3:.1f} mm) ──")
    s1 = []
    for f_mm in SWEEP_F_VALUES:
        f_m = f_mm * 1e-3
        src = _build_source(XX, YY, "ideal", f_m, SWEEP_F_R_AP)
        fz, pk, sr = _measure(src, dx, dy, x, y, z_arr, iy_c, ix_c)
        row = {"sweep": "focal_length", "family": "ideal",
               "f_mm": f_mm, "R_ap_mm": SWEEP_F_R_AP * 1e3,
               "focal_z_mm": round(fz * 1e3, 3),
               "peak_p": round(pk, 5),
               "spot_radius_mm": round(sr * 1e3, 4)}
        s1.append(row)
        all_rows.append(row)
        print(f"  f={f_mm:5.1f} mm → focal_z={fz*1e3:.2f} mm  "
              f"peak={pk:.4f}  spot_r={sr*1e3:.3f} mm")

    # Plot sweep 1
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    f_vals = [r["f_mm"] for r in s1]

    ax = axes[0]
    ax.plot(f_vals, [r["focal_z_mm"] for r in s1], "o-", color="tab:blue", lw=1.8)
    ax.plot(f_vals, f_vals, "k--", lw=0.8, alpha=0.4, label="f = z (geometric)")
    ax.set_xlabel("Focal length f [mm]")
    ax.set_ylabel("Focal z [mm]")
    ax.set_title("Focal z vs f")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(f_vals, [r["peak_p"] for r in s1], "s-", color="tab:red", lw=1.8)
    ax.set_xlabel("Focal length f [mm]")
    ax.set_ylabel("Peak |p| [a.u.]")
    ax.set_title("Peak pressure vs f")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(f_vals, [r["spot_radius_mm"] for r in s1], "^-",
            color="tab:green", lw=1.8)
    ax.set_xlabel("Focal length f [mm]")
    ax.set_ylabel("Spot radius [mm]")
    ax.set_title("Half-power spot radius vs f")
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Sweep 1: Ideal lens — focal length  "
        f"(R_ap = {SWEEP_F_R_AP*1e3:.1f} mm)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT / "sweep_focal_length.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("→ saved sweep_focal_length.png")

    # ══════════════════════════════════════════════════════════════
    # Sweep 2 — aperture radius (ideal lens)
    # ══════════════════════════════════════════════════════════════
    print(f"\n── Sweep 2: aperture radius (ideal, f={SWEEP_R_F*1e3:.1f} mm) ──")
    s2 = []
    for r_mm in SWEEP_R_VALUES:
        r_m = r_mm * 1e-3
        src = _build_source(XX, YY, "ideal", SWEEP_R_F, r_m)
        fz, pk, sr = _measure(src, dx, dy, x, y, z_arr, iy_c, ix_c)
        row = {"sweep": "aperture_radius", "family": "ideal",
               "f_mm": SWEEP_R_F * 1e3, "R_ap_mm": r_mm,
               "focal_z_mm": round(fz * 1e3, 3),
               "peak_p": round(pk, 5),
               "spot_radius_mm": round(sr * 1e3, 4)}
        s2.append(row)
        all_rows.append(row)
        print(f"  R_ap={r_mm:4.1f} mm → focal_z={fz*1e3:.2f} mm  "
              f"peak={pk:.4f}  spot_r={sr*1e3:.3f} mm")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    r_vals = [r["R_ap_mm"] for r in s2]

    ax = axes[0]
    ax.plot(r_vals, [r["focal_z_mm"] for r in s2], "o-", color="tab:blue", lw=1.8)
    ax.set_xlabel("Aperture radius [mm]")
    ax.set_ylabel("Focal z [mm]")
    ax.set_title("Focal z vs R_ap")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(r_vals, [r["peak_p"] for r in s2], "s-", color="tab:red", lw=1.8)
    ax.set_xlabel("Aperture radius [mm]")
    ax.set_ylabel("Peak |p| [a.u.]")
    ax.set_title("Peak pressure vs R_ap")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(r_vals, [r["spot_radius_mm"] for r in s2], "^-",
            color="tab:green", lw=1.8)
    ax.set_xlabel("Aperture radius [mm]")
    ax.set_ylabel("Spot radius [mm]")
    ax.set_title("Half-power spot radius vs R_ap")
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Sweep 2: Ideal lens — aperture radius  "
        f"(f = {SWEEP_R_F*1e3:.1f} mm)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT / "sweep_aperture_radius.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("→ saved sweep_aperture_radius.png")

    # ══════════════════════════════════════════════════════════════
    # Sweep 3 — axicon angle
    # ══════════════════════════════════════════════════════════════
    AXICON_ANGLES = [5.0, 10.0, 15.0, 20.0, 25.0]
    SWEEP_AX_R_AP = 2.5e-3
    print(f"\n── Sweep 3: axicon angle (R_ap={SWEEP_AX_R_AP*1e3:.1f} mm) ──")
    s3 = []
    for a_deg in AXICON_ANGLES:
        src = _build_source(XX, YY, "axicon", 5e-3, SWEEP_AX_R_AP,
                            axicon_deg=a_deg)
        fz, pk, sr = _measure(src, dx, dy, x, y, z_arr, iy_c, ix_c)
        row = {"sweep": "axicon_angle", "family": "axicon",
               "axicon_deg": a_deg, "R_ap_mm": SWEEP_AX_R_AP * 1e3,
               "focal_z_mm": round(fz * 1e3, 3),
               "peak_p": round(pk, 5),
               "spot_radius_mm": round(sr * 1e3, 4)}
        s3.append(row)
        all_rows.append(row)
        print(f"  α={a_deg:5.1f}° → focal_z={fz*1e3:.2f} mm  "
              f"peak={pk:.4f}  spot_r={sr*1e3:.3f} mm")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    a_vals = [r["axicon_deg"] for r in s3]

    ax = axes[0]
    ax.plot(a_vals, [r["focal_z_mm"] for r in s3], "o-", color="tab:blue", lw=1.8)
    ax.set_xlabel("Axicon half-angle α [°]")
    ax.set_ylabel("Focal z [mm]")
    ax.set_title("Focal z vs α")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(a_vals, [r["peak_p"] for r in s3], "s-", color="tab:red", lw=1.8)
    ax.set_xlabel("Axicon half-angle α [°]")
    ax.set_ylabel("Peak |p| [a.u.]")
    ax.set_title("Peak pressure vs α")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(a_vals, [r["spot_radius_mm"] for r in s3], "^-",
            color="tab:green", lw=1.8)
    ax.set_xlabel("Axicon half-angle α [°]")
    ax.set_ylabel("Spot radius [mm]")
    ax.set_title("Half-power spot radius vs α")
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Sweep 3: Axicon lens — half-angle  "
        f"(R_ap = {SWEEP_AX_R_AP*1e3:.1f} mm)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT / "sweep_axicon_angle.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("→ saved sweep_axicon_angle.png")

    # ══════════════════════════════════════════════════════════════
    # CSV + JSON
    # ══════════════════════════════════════════════════════════════
    csv_path = OUT / "sweep_results.csv"
    fieldnames = sorted({k for r in all_rows for k in r.keys()})
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n→ saved sweep_results.csv  ({len(all_rows)} rows)")

    with open(OUT / "sweep_summary.json", "w") as f:
        json.dump({
            "sweeps": {
                "focal_length": {
                    "family": "ideal",
                    "R_ap_mm": SWEEP_F_R_AP * 1e3,
                    "f_range_mm": SWEEP_F_VALUES,
                },
                "aperture_radius": {
                    "family": "ideal",
                    "f_mm": SWEEP_R_F * 1e3,
                    "R_ap_range_mm": SWEEP_R_VALUES,
                },
                "axicon_angle": {
                    "family": "axicon",
                    "R_ap_mm": SWEEP_AX_R_AP * 1e3,
                    "angle_range_deg": AXICON_ANGLES,
                },
            },
            "wavelength_mm": LAM * 1e3,
            "grid_n": NX,
            "nz_scan": NZ,
            "z_range_mm": [Z_MIN * 1e3, Z_MAX * 1e3],
            "results": all_rows,
        }, f, indent=2)
    print("→ saved sweep_summary.json")

    print(f"\n{'='*60}")
    print(f"Done. Outputs → {OUT.relative_to(PROJECT_ROOT)}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
