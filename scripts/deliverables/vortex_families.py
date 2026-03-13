#!/usr/bin/env python3
"""
Phase 2.2C — Vortex family comparison.

Compares three vortex+lens families on the same FEM-compatible grid:
  1. Ideal  — spherical focusing lens with vortex phase
  2. Plastic — wrapped (fabricable) lens with vortex phase
  3. Axicon — conical (Bessel-generating) lens with vortex phase

For each family:
  - XY slice near focus  (magnitude + phase)
  - XZ propagation view  (magnitude)
  - Quantitative metrics

Generates:
  - comparison panel figure
  - per-family detail figures
  - family_metrics.json (machine-readable)
  - family_metrics.csv  (human-readable)

Output  → results/deliverables/vortex/
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

# ── Project root & imports ─────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lib.fem_cache_utils import load_fem_cache, LAM, C_WATER, F_HZ
from scripts.lib.asm_utils import (
    make_grid_from_fem,
    make_vortex_field,
    make_lens_phase,
    propagate_asm,
    K0,
)

OUT = PROJECT_ROOT / "results" / "deliverables" / "vortex"
OUT.mkdir(parents=True, exist_ok=True)

_mm = lambda v: v * 1e3

# ══════════════════════════════════════════════════════════════════
# Parameters (shared across all families for fair comparison)
# ══════════════════════════════════════════════════════════════════
CHARGE = 1
WAIST = 2.0e-3           # vortex beam waist [m]
R_AP = 2.5e-3            # aperture radius [m]
FOCAL_LENGTH = 3.5e-3    # shorter f so max phase > 2π → wrapping matters
AXICON_ANGLE = 15.0      # axicon half-angle [deg]
NX, NY = 400, 400

# Propagation sweep for XZ
Z_MIN, Z_MAX = 0.5e-3, 7.0e-3
NZ_XZ = 80

# Focal-plane z for XY slice (= focal_length for ideal/plastic;
# axicon produces a self-similar "Bessel zone" so we use the same z)
Z_FOCUS = FOCAL_LENGTH

# Families to compare
FAMILIES = [
    {"name": "Ideal",  "family": "ideal",  "label": "Ideal (spherical focus)"},
    {"name": "Axicon", "family": "axicon", "label": "Axicon (Bessel-like)"},
    {"name": "Free",   "family": "free",   "label": "Free (no lens)"},
]


def _ring_radius(mag_2d, x_1d, y_1d):
    """Estimate ring radius as distance from centre to peak |p| pixel."""
    iy, ix = np.unravel_index(np.argmax(mag_2d), mag_2d.shape)
    cx = 0.5 * (x_1d[0] + x_1d[-1])
    cy = 0.5 * (y_1d[0] + y_1d[-1])
    return float(np.sqrt((x_1d[ix] - cx)**2 + (y_1d[iy] - cy)**2))


def _focal_z(xz_mag, z_arr):
    """Estimate focal z as the z-plane where the peak XZ magnitude is highest."""
    peak_per_z = xz_mag.max(axis=1)
    return float(z_arr[np.argmax(peak_per_z)])


def main() -> None:
    print("=" * 60)
    print("Phase 2.2C — Vortex family comparison")
    print("=" * 60)

    # ── Load FEM cache for grid ────────────────────────────────────
    cache = load_fem_cache()
    grid = make_grid_from_fem(cache, nx=NX, ny=NY)
    x, y = grid["x"], grid["y"]
    XX, YY = grid["XX"], grid["YY"]
    dx, dy = grid["dx"], grid["dy"]
    iy_centre = NY // 2

    print(f"Grid: {NX}×{NY}  dx={dx*1e6:.1f} µm  dy={dy*1e6:.1f} µm")
    print(f"Shared params: ℓ={CHARGE}, w={WAIST*1e3:.1f} mm, "
          f"R_ap={R_AP*1e3:.1f} mm, f={FOCAL_LENGTH*1e3:.1f} mm\n")

    # ── Common source vortex (amplitude only, no lens) ─────────────
    vortex = make_vortex_field(XX, YY, charge=CHARGE, waist=WAIST, k=K0,
                               aperture_radius=R_AP)

    z_xz = np.linspace(Z_MIN, Z_MAX, NZ_XZ)
    metrics_list = []
    family_xy_fields = {}   # {name: complex field at Z_FOCUS}
    family_xz_mags = {}     # {name: (NZ, NX) array}

    for fam in FAMILIES:
        name = fam["name"]
        print(f"── {fam['label']} ──")

        # Build source: apply lens if not free-space
        if fam["family"] == "free":
            source = vortex.copy()
        else:
            lens = make_lens_phase(
                XX, YY,
                focal_length=FOCAL_LENGTH,
                aperture_radius=R_AP,
                family=fam["family"],
                k=K0,
                charge=0,           # vortex charge already in vortex field
                axicon_angle_deg=AXICON_ANGLE,
            )
            source = vortex * np.exp(-1j * lens)  # converging sign

        # XY slice at focal plane
        p_focus = propagate_asm(source, dx, dy, wavelength=LAM, z=Z_FOCUS)
        family_xy_fields[name] = p_focus
        mag_focus = np.abs(p_focus)

        # XZ propagation
        xz_mag = np.zeros((NZ_XZ, NX))
        for iz, z_val in enumerate(z_xz):
            p = propagate_asm(source, dx, dy, wavelength=LAM, z=z_val)
            xz_mag[iz, :] = np.abs(p[iy_centre, :])
        family_xz_mags[name] = xz_mag

        # Metrics
        peak_p = float(mag_focus.max())
        ring_r = _ring_radius(mag_focus, x, y)
        foc_z = _focal_z(xz_mag, z_xz)

        m = {
            "family": name,
            "peak_p_au": round(peak_p, 5),
            "ring_radius_mm": round(ring_r * 1e3, 4),
            "focal_z_mm": round(foc_z * 1e3, 3),
        }
        metrics_list.append(m)
        print(f"  peak |p| = {peak_p:.5f}  ring_r = {ring_r*1e3:.3f} mm  "
              f"focal_z = {foc_z*1e3:.1f} mm")

    # ══════════════════════════════════════════════════════════════
    # Comparison panel: XY magnitude at focus (one column per family)
    # ══════════════════════════════════════════════════════════════
    n_fam = len(FAMILIES)
    vmax_xy = max(np.abs(family_xy_fields[f["name"]]).max() for f in FAMILIES)

    fig, axes = plt.subplots(2, n_fam, figsize=(5 * n_fam, 9))
    for j, fam in enumerate(FAMILIES):
        name = fam["name"]
        mag = np.abs(family_xy_fields[name])
        phase = np.angle(family_xy_fields[name])

        # Magnitude row
        ax = axes[0, j]
        im = ax.pcolormesh(_mm(x), _mm(y), mag, shading="auto",
                           cmap="inferno", vmin=0, vmax=vmax_xy)
        ax.set_title(f"{fam['label']}\n|p| at z = {Z_FOCUS*1e3:.1f} mm",
                     fontsize=10)
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, shrink=0.78)

        # Phase row
        ax = axes[1, j]
        im = ax.pcolormesh(_mm(x), _mm(y), phase, shading="auto",
                           cmap="hsv", vmin=-np.pi, vmax=np.pi)
        ax.set_title(f"Phase at z = {Z_FOCUS*1e3:.1f} mm", fontsize=10)
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, shrink=0.78, label="rad")

    fig.suptitle(
        f"Vortex family comparison — XY at focus  (ℓ={CHARGE})",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    fig.savefig(OUT / "vortex_family_xy_comparison.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print("\n→ saved vortex_family_xy_comparison.png")

    # ══════════════════════════════════════════════════════════════
    # Comparison panel: XZ magnitude (one row per family)
    # ══════════════════════════════════════════════════════════════
    vmax_xz = max(v.max() for v in family_xz_mags.values())

    fig, axes = plt.subplots(n_fam, 1, figsize=(10, 4 * n_fam), sharex=True)
    if n_fam == 1:
        axes = [axes]
    for j, fam in enumerate(FAMILIES):
        name = fam["name"]
        ax = axes[j]
        im = ax.pcolormesh(_mm(x), _mm(z_xz), family_xz_mags[name],
                           shading="auto", cmap="inferno", vmin=0,
                           vmax=vmax_xz)
        ax.set_ylabel("z [mm]", fontsize=11)
        ax.set_title(f"{fam['label']}", fontsize=11)
        plt.colorbar(im, ax=ax, shrink=0.85, label="|p| [a.u.]")
        ax.axhline(Z_FOCUS * 1e3, color="white", ls="--", lw=0.7, alpha=0.6)

    axes[-1].set_xlabel("x [mm]", fontsize=11)
    fig.suptitle(
        f"Vortex family comparison — XZ propagation  (ℓ={CHARGE})",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    fig.savefig(OUT / "vortex_family_xz_comparison.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print("→ saved vortex_family_xz_comparison.png")

    # ══════════════════════════════════════════════════════════════
    # Metrics output
    # ══════════════════════════════════════════════════════════════
    with open(OUT / "family_metrics.json", "w") as f:
        json.dump({
            "parameters": {
                "charge": CHARGE,
                "waist_mm": WAIST * 1e3,
                "aperture_radius_mm": R_AP * 1e3,
                "focal_length_mm": FOCAL_LENGTH * 1e3,
                "axicon_angle_deg": AXICON_ANGLE,
                "z_focus_mm": Z_FOCUS * 1e3,
                "wavelength_mm": LAM * 1e3,
            },
            "families": metrics_list,
        }, f, indent=2)
    print("→ saved family_metrics.json")

    with open(OUT / "family_metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metrics_list[0].keys())
        writer.writeheader()
        writer.writerows(metrics_list)
    print("→ saved family_metrics.csv")

    # ── Summary table ─────────────────────────────────────────────
    print(f"\n{'Family':<12} {'peak |p|':>10} {'ring_r [mm]':>12} {'focal_z [mm]':>13}")
    print("-" * 50)
    for m in metrics_list:
        print(f"{m['family']:<12} {m['peak_p_au']:>10.5f} "
              f"{m['ring_radius_mm']:>12.4f} {m['focal_z_mm']:>13.3f}")

    print(f"\n{'='*60}")
    print(f"Done. Outputs → {OUT.relative_to(PROJECT_ROOT)}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
