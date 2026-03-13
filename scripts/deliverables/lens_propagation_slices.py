#!/usr/bin/env python3
"""
Phase 2.3B — Lens propagation slices.

For each lens family (flat, ideal, plastic, axicon):
  - generates a uniform source + lens phase
  - propagates via ASM
  - produces XY slices near focus and XZ propagation views

Uses converging sign convention: source = A · exp(-i φ_lens).

Output → results/deliverables/lens/
"""
from __future__ import annotations

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
FOCAL_LENGTH = 5.0e-3
R_AP = 2.5e-3
AXICON_ANGLE = 15.0
NX, NY = 400, 400

# XZ sweep
Z_MIN, Z_MAX = 0.5e-3, 9.0e-3
NZ_XZ = 100

FAMILIES = [
    {"name": "Flat",    "family": None,      "label": "Flat (no lens)"},
    {"name": "Ideal",   "family": "ideal",   "label": "Ideal spherical"},
    {"name": "Plastic", "family": "plastic", "label": "Plastic (wrapped)"},
    {"name": "Axicon",  "family": "axicon",  "label": "Axicon"},
]

COLOURS = {"Flat": "gray", "Ideal": "tab:blue",
           "Plastic": "tab:orange", "Axicon": "tab:green"}


def _make_source(XX, YY, fam, dx, dy):
    """Build a uniform-amplitude disk source with the given lens phase."""
    r_from_centre = np.sqrt(
        (XX - 0.5 * (XX.min() + XX.max()))**2 +
        (YY - 0.5 * (YY.min() + YY.max()))**2
    )
    amplitude = np.ones_like(XX, dtype=np.float64)
    amplitude[r_from_centre > R_AP] = 0.0

    if fam["family"] is None:
        return amplitude.astype(complex)

    phi = make_lens_phase(
        XX, YY,
        focal_length=FOCAL_LENGTH,
        aperture_radius=R_AP,
        family=fam["family"],
        k=K0,
        charge=0,
        axicon_angle_deg=AXICON_ANGLE,
    )
    return amplitude * np.exp(-1j * phi)


def _focal_z(xz_mag, z_arr):
    """z where peak XZ magnitude is highest."""
    return float(z_arr[np.argmax(xz_mag.max(axis=1))])


def _spot_radius(mag_2d, x_1d, y_1d):
    """Half-power radius from centre: radius enclosing pixels > peak/2."""
    peak = mag_2d.max()
    cx = 0.5 * (x_1d[0] + x_1d[-1])
    cy = 0.5 * (y_1d[0] + y_1d[-1])
    XX, YY = np.meshgrid(x_1d, y_1d)
    r = np.sqrt((XX - cx)**2 + (YY - cy)**2)
    mask = mag_2d >= peak * 0.5
    if mask.any():
        return float(r[mask].max())
    return float(r.max())


def main() -> None:
    print("=" * 60)
    print("Phase 2.3B — Lens propagation slices")
    print("=" * 60)

    cache = load_fem_cache()
    grid = make_grid_from_fem(cache, nx=NX, ny=NY)
    x, y = grid["x"], grid["y"]
    XX, YY = grid["XX"], grid["YY"]
    dx, dy = grid["dx"], grid["dy"]
    iy_centre = NY // 2

    print(f"Grid: {NX}×{NY}  dx={dx*1e6:.1f} µm  "
          f"R_ap={R_AP*1e3:.1f} mm  f={FOCAL_LENGTH*1e3:.1f} mm\n")

    z_xz = np.linspace(Z_MIN, Z_MAX, NZ_XZ)

    # Storage for comparison
    family_focal_fields = {}
    family_xz_mags = {}
    family_focal_z = {}
    metrics = []

    for fam in FAMILIES:
        name = fam["name"]
        print(f"── {fam['label']} ──")

        source = _make_source(XX, YY, fam, dx, dy)

        # XZ propagation
        xz_mag = np.zeros((NZ_XZ, NX))
        for iz, z_val in enumerate(z_xz):
            p = propagate_asm(source, dx, dy, wavelength=LAM, z=z_val)
            xz_mag[iz, :] = np.abs(p[iy_centre, :])
        family_xz_mags[name] = xz_mag

        fz = _focal_z(xz_mag, z_xz)
        family_focal_z[name] = fz
        print(f"  focal z ≈ {fz*1e3:.2f} mm")

        # XY at focal z
        p_focus = propagate_asm(source, dx, dy, wavelength=LAM, z=fz)
        family_focal_fields[name] = p_focus
        mag = np.abs(p_focus)
        peak = float(mag.max())
        spot_r = _spot_radius(mag, x, y)
        print(f"  peak |p| = {peak:.4f}  spot_r (half-power) = {spot_r*1e3:.3f} mm")

        metrics.append({
            "family": name,
            "focal_z_mm": round(fz * 1e3, 3),
            "peak_p": round(peak, 5),
            "spot_radius_mm": round(spot_r * 1e3, 4),
        })

    # ══════════════════════════════════════════════════════════════
    # XY comparison panel (2 rows: magnitude, phase × 4 families)
    # ══════════════════════════════════════════════════════════════
    n_fam = len(FAMILIES)
    vmax_xy = max(np.abs(family_focal_fields[f["name"]]).max()
                  for f in FAMILIES)

    fig, axes = plt.subplots(2, n_fam, figsize=(5 * n_fam, 9))
    for j, fam in enumerate(FAMILIES):
        name = fam["name"]
        fz = family_focal_z[name]
        mag = np.abs(family_focal_fields[name])
        phase = np.angle(family_focal_fields[name])

        ax = axes[0, j]
        im = ax.pcolormesh(_mm(x), _mm(y), mag, shading="auto",
                           cmap="inferno", vmin=0, vmax=vmax_xy)
        ax.set_title(f"{fam['label']}\n|p| at z = {fz*1e3:.1f} mm",
                     fontsize=10)
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, shrink=0.78)

        ax = axes[1, j]
        im = ax.pcolormesh(_mm(x), _mm(y), phase, shading="auto",
                           cmap="twilight", vmin=-np.pi, vmax=np.pi)
        ax.set_title(f"Phase at z = {fz*1e3:.1f} mm", fontsize=10)
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, shrink=0.78, label="rad")

    fig.suptitle(
        "Lens family comparison — XY at focal plane  (uniform source, charge=0)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    fig.savefig(OUT / "lens_xy_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("\n→ saved lens_xy_comparison.png")

    # ══════════════════════════════════════════════════════════════
    # XZ comparison panel (stacked)
    # ══════════════════════════════════════════════════════════════
    vmax_xz = max(v.max() for v in family_xz_mags.values())

    fig, axes = plt.subplots(n_fam, 1, figsize=(10, 3.5 * n_fam), sharex=True)
    for j, fam in enumerate(FAMILIES):
        name = fam["name"]
        ax = axes[j]
        im = ax.pcolormesh(_mm(x), _mm(z_xz), family_xz_mags[name],
                           shading="auto", cmap="inferno", vmin=0,
                           vmax=vmax_xz)
        ax.set_ylabel("z [mm]", fontsize=11)
        ax.set_title(fam["label"], fontsize=11)
        plt.colorbar(im, ax=ax, shrink=0.85, label="|p| [a.u.]")
        fz = family_focal_z[name]
        ax.axhline(fz * 1e3, color="white", ls="--", lw=0.7, alpha=0.7)
    axes[-1].set_xlabel("x [mm]", fontsize=11)

    fig.suptitle(
        "Lens family comparison — XZ propagation  (uniform source, charge=0)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    fig.savefig(OUT / "lens_xz_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("→ saved lens_xz_comparison.png")

    # ── On-axis intensity vs z ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ix_centre = NX // 2
    for fam in FAMILIES:
        name = fam["name"]
        ax.plot(_mm(z_xz), family_xz_mags[name][:, ix_centre],
                color=COLOURS[name], lw=1.8, label=fam["label"])
    ax.set_xlabel("z [mm]", fontsize=12)
    ax.set_ylabel("|p| on axis [a.u.]", fontsize=12)
    ax.set_title("On-axis pressure vs propagation distance",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "lens_onaxis_vs_z.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("→ saved lens_onaxis_vs_z.png")

    # ── Metrics JSON ──────────────────────────────────────────────
    with open(OUT / "propagation_metrics.json", "w") as f:
        json.dump({
            "parameters": {
                "focal_length_mm": FOCAL_LENGTH * 1e3,
                "aperture_radius_mm": R_AP * 1e3,
                "axicon_angle_deg": AXICON_ANGLE,
                "wavelength_mm": LAM * 1e3,
                "z_range_mm": [Z_MIN * 1e3, Z_MAX * 1e3],
                "nz": NZ_XZ,
            },
            "families": metrics,
        }, f, indent=2)
    print("→ saved propagation_metrics.json")

    # ── Summary table ─────────────────────────────────────────────
    print(f"\n{'Family':<10} {'focal z [mm]':>12} {'peak |p|':>10} "
          f"{'spot r [mm]':>12}")
    print("-" * 48)
    for m in metrics:
        print(f"{m['family']:<10} {m['focal_z_mm']:>12.3f} "
              f"{m['peak_p']:>10.5f} {m['spot_radius_mm']:>12.4f}")

    print(f"\n{'='*60}")
    print(f"Done. Outputs → {OUT.relative_to(PROJECT_ROOT)}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
