#!/usr/bin/env python3
"""
Phase 2.2A/B — Vortex multi-z XY slices and XZ propagation views.

Generates:
  A) XY magnitude and phase slices at multiple z through propagation
  B) XZ propagation slices (central y=0 cut) showing hourglass / waist

Uses the reusable ASM pipeline from Phase 2.1 and the FEM-compatible grid.

Output  → results/deliverables/vortex/
"""
from __future__ import annotations

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

# ── Helper ─────────────────────────────────────────────────────────
def _mm(v):
    return v * 1e3


# ══════════════════════════════════════════════════════════════════
# Parameters
# ══════════════════════════════════════════════════════════════════
CHARGE = 1
WAIST = 2.0e-3           # 2 mm beam waist (fills aperture → efficient focusing)
R_AP = 2.5e-3            # 2.5 mm aperture radius
FOCAL_LENGTH = 5.0e-3    # 5 mm — focal length for ideal focusing lens
NX, NY = 400, 400        # grid resolution (matches Phase 2.1 test)

# Propagation z range: we image from 1 mm to 9 mm in steps for XZ,
# centred around the focal length to show hourglass structure.
Z_MIN = 1.0e-3
Z_MAX = 9.0e-3
NZ_XZ = 120              # number of z-planes for XZ view (fine sampling)

# Representative z-planes for XY slices (before / near / at / past focus)
# With these params, effective focus is ~3-4 mm (Gaussian beam + thin lens).
Z_SLICES_MM = [1.0, 2.0, 3.0, 3.5, 4.0, 5.0, 7.0]


def main() -> None:
    print("=" * 60)
    print("Phase 2.2 — Vortex XY slices & XZ propagation views")
    print("=" * 60)

    # ── Load FEM cache for grid ────────────────────────────────────
    cache = load_fem_cache()
    grid = make_grid_from_fem(cache, nx=NX, ny=NY)
    x, y = grid["x"], grid["y"]
    XX, YY = grid["XX"], grid["YY"]
    dx, dy = grid["dx"], grid["dy"]

    print(f"Grid: {NX}×{NY}  dx={dx*1e6:.1f} µm  dy={dy*1e6:.1f} µm")
    print(f"x ∈ [{x[0]*1e3:.3f}, {x[-1]*1e3:.3f}] mm")
    print(f"y ∈ [{y[0]*1e3:.3f}, {y[-1]*1e3:.3f}] mm")

    # ── Build source field: vortex + ideal focusing lens ───────────
    vortex = make_vortex_field(XX, YY, charge=CHARGE, waist=WAIST, k=K0,
                               aperture_radius=R_AP)
    lens = make_lens_phase(XX, YY, focal_length=FOCAL_LENGTH,
                           aperture_radius=R_AP, family="ideal", k=K0,
                           charge=0)
    source = vortex * np.exp(-1j * lens)  # negative sign → converging lens

    print(f"\nSource: charge={CHARGE}, waist={WAIST*1e3:.1f} mm, "
          f"R_ap={R_AP*1e3:.1f} mm, f={FOCAL_LENGTH*1e3:.1f} mm")
    print(f"|source|_max = {np.abs(source).max():.4f}")

    # ══════════════════════════════════════════════════════════════
    # A. Multi-z XY slices
    # ══════════════════════════════════════════════════════════════
    print(f"\n── A. Multi-z XY slices ({len(Z_SLICES_MM)} planes) ──")

    z_slices = [z * 1e-3 for z in Z_SLICES_MM]
    fields_at_z = {}
    for z_val in z_slices:
        p = propagate_asm(source, dx, dy, wavelength=LAM, z=z_val)
        fields_at_z[z_val] = p
        print(f"  z = {z_val*1e3:5.1f} mm  |p|_max = {np.abs(p).max():.4f}")

    # --- Multi-panel magnitude figure ---
    n_cols = 4
    n_rows = int(np.ceil(len(z_slices) / n_cols))
    fig_mag, axes_mag = plt.subplots(n_rows, n_cols,
                                     figsize=(4 * n_cols, 3.8 * n_rows))
    axes_mag = np.atleast_2d(axes_mag).ravel()

    # Common vmax for consistent colour scale across panels
    vmax_mag = max(np.abs(fields_at_z[z]).max() for z in z_slices) * 1.0

    for i, z_val in enumerate(z_slices):
        ax = axes_mag[i]
        mag = np.abs(fields_at_z[z_val])
        im = ax.pcolormesh(_mm(x), _mm(y), mag, shading="auto",
                           cmap="inferno", vmin=0, vmax=vmax_mag)
        ax.set_title(f"z = {z_val*1e3:.1f} mm", fontsize=11)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, shrink=0.78, label="|p| [a.u.]")

    # Hide unused axes
    for j in range(len(z_slices), len(axes_mag)):
        axes_mag[j].set_visible(False)

    fig_mag.suptitle(
        f"Vortex |p| — XY slices  (ℓ={CHARGE}, w={WAIST*1e3:.1f} mm, "
        f"f={FOCAL_LENGTH*1e3:.0f} mm)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig_mag.tight_layout()
    fig_mag.savefig(OUT / "vortex_xy_magnitude_multi.png", dpi=200,
                    bbox_inches="tight")
    plt.close(fig_mag)
    print("  → saved vortex_xy_magnitude_multi.png")

    # --- Multi-panel phase figure ---
    fig_ph, axes_ph = plt.subplots(n_rows, n_cols,
                                    figsize=(4 * n_cols, 3.8 * n_rows))
    axes_ph = np.atleast_2d(axes_ph).ravel()

    for i, z_val in enumerate(z_slices):
        ax = axes_ph[i]
        phase = np.angle(fields_at_z[z_val])
        im = ax.pcolormesh(_mm(x), _mm(y), phase, shading="auto",
                           cmap="hsv", vmin=-np.pi, vmax=np.pi)
        ax.set_title(f"z = {z_val*1e3:.1f} mm", fontsize=11)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, shrink=0.78, label="phase [rad]")

    for j in range(len(z_slices), len(axes_ph)):
        axes_ph[j].set_visible(False)

    fig_ph.suptitle(
        f"Vortex phase — XY slices  (ℓ={CHARGE}, w={WAIST*1e3:.1f} mm, "
        f"f={FOCAL_LENGTH*1e3:.0f} mm)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig_ph.tight_layout()
    fig_ph.savefig(OUT / "vortex_xy_phase_multi.png", dpi=200,
                   bbox_inches="tight")
    plt.close(fig_ph)
    print("  → saved vortex_xy_phase_multi.png")

    # --- Individual per-slice PNGs ---
    for z_val in z_slices:
        p_z = fields_at_z[z_val]
        tag = f"{z_val*1e3:.1f}".replace(".", "p")

        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4))
        im0 = ax0.pcolormesh(_mm(x), _mm(y), np.abs(p_z), shading="auto",
                             cmap="inferno", vmin=0, vmax=vmax_mag)
        ax0.set_title(f"|p| at z = {z_val*1e3:.1f} mm")
        ax0.set_xlabel("x [mm]"); ax0.set_ylabel("y [mm]")
        ax0.set_aspect("equal")
        plt.colorbar(im0, ax=ax0, shrink=0.8, label="|p| [a.u.]")

        im1 = ax1.pcolormesh(_mm(x), _mm(y), np.angle(p_z), shading="auto",
                             cmap="hsv", vmin=-np.pi, vmax=np.pi)
        ax1.set_title(f"Phase at z = {z_val*1e3:.1f} mm")
        ax1.set_xlabel("x [mm]"); ax1.set_ylabel("y [mm]")
        ax1.set_aspect("equal")
        plt.colorbar(im1, ax=ax1, shrink=0.8, label="rad")

        fig.suptitle(f"Vortex beam — z = {z_val*1e3:.1f} mm", fontsize=12,
                     fontweight="bold")
        fig.tight_layout()
        fig.savefig(OUT / f"vortex_xy_z{tag}mm.png", dpi=150,
                    bbox_inches="tight")
        plt.close(fig)

    print(f"  → saved {len(z_slices)} individual vortex_xy_z*.png files")

    # ══════════════════════════════════════════════════════════════
    # B. XZ propagation views
    # ══════════════════════════════════════════════════════════════
    print(f"\n── B. XZ propagation views ({NZ_XZ} z-planes) ──")

    z_xz = np.linspace(Z_MIN, Z_MAX, NZ_XZ)
    iy_centre = NY // 2  # central y row

    # Off-centre y index (for second XZ view)
    y_off = 0.5e-3
    iy_off = int(np.argmin(np.abs(y - (y[iy_centre] + y_off))))

    # Preallocate: rows=z, cols=x
    xz_mag = np.zeros((NZ_XZ, NX))
    xz_phase = np.zeros((NZ_XZ, NX))
    xz_off_mag = np.zeros((NZ_XZ, NX))

    for iz, z_val in enumerate(z_xz):
        p = propagate_asm(source, dx, dy, wavelength=LAM, z=z_val)
        xz_mag[iz, :] = np.abs(p[iy_centre, :])
        xz_phase[iz, :] = np.angle(p[iy_centre, :])
        xz_off_mag[iz, :] = np.abs(p[iy_off, :])
        if iz % 20 == 0:
            print(f"  z = {z_val*1e3:5.1f} mm  (plane {iz+1}/{NZ_XZ})")

    # --- Magnitude XZ figure ---
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.pcolormesh(_mm(x), _mm(z_xz), xz_mag, shading="auto",
                       cmap="inferno")
    ax.set_xlabel("x [mm]", fontsize=12)
    ax.set_ylabel("z [mm]", fontsize=12)
    ax.set_title(
        f"Vortex |p| — XZ plane (y = {y[iy_centre]*1e3:.2f} mm)  "
        f"ℓ={CHARGE}, f={FOCAL_LENGTH*1e3:.0f} mm",
        fontsize=13, fontweight="bold",
    )
    plt.colorbar(im, ax=ax, shrink=0.85, label="|p| [a.u.]")
    ax.axhline(FOCAL_LENGTH * 1e3, color="white", ls="--", lw=0.8,
               alpha=0.7, label=f"focal plane ({FOCAL_LENGTH*1e3:.0f} mm)")
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "vortex_xz_magnitude.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  → saved vortex_xz_magnitude.png")

    # --- Phase XZ figure ---
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.pcolormesh(_mm(x), _mm(z_xz), xz_phase, shading="auto",
                       cmap="hsv", vmin=-np.pi, vmax=np.pi)
    ax.set_xlabel("x [mm]", fontsize=12)
    ax.set_ylabel("z [mm]", fontsize=12)
    ax.set_title(
        f"Vortex phase — XZ plane (y = {y[iy_centre]*1e3:.2f} mm)  "
        f"ℓ={CHARGE}, f={FOCAL_LENGTH*1e3:.0f} mm",
        fontsize=13, fontweight="bold",
    )
    plt.colorbar(im, ax=ax, shrink=0.85, label="phase [rad]")
    ax.axhline(FOCAL_LENGTH * 1e3, color="white", ls="--", lw=0.8,
               alpha=0.7, label=f"focal plane ({FOCAL_LENGTH*1e3:.0f} mm)")
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "vortex_xz_phase.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  → saved vortex_xz_phase.png")

    # --- Off-centre XZ (y offset by ~0.5 mm) ---
    # (already computed in the main loop above)

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.pcolormesh(_mm(x), _mm(z_xz), xz_off_mag, shading="auto",
                       cmap="inferno")
    ax.set_xlabel("x [mm]", fontsize=12)
    ax.set_ylabel("z [mm]", fontsize=12)
    ax.set_title(
        f"Vortex |p| — XZ off-centre (y = {y[iy_off]*1e3:.2f} mm)  "
        f"ℓ={CHARGE}",
        fontsize=13, fontweight="bold",
    )
    plt.colorbar(im, ax=ax, shrink=0.85, label="|p| [a.u.]")
    fig.tight_layout()
    fig.savefig(OUT / "vortex_xz_offcentre.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  → saved vortex_xz_offcentre.png")

    # ── Save raw data ─────────────────────────────────────────────
    np.savez_compressed(
        OUT / "vortex_xz_data.npz",
        x_mm=_mm(x),
        z_mm=_mm(z_xz),
        xz_mag=xz_mag,
        xz_phase=xz_phase,
        y_centre_mm=y[iy_centre] * 1e3,
    )
    print("  → saved vortex_xz_data.npz")

    # ── Summary JSON ──────────────────────────────────────────────
    summary = {
        "charge": CHARGE,
        "waist_mm": WAIST * 1e3,
        "aperture_radius_mm": R_AP * 1e3,
        "focal_length_mm": FOCAL_LENGTH * 1e3,
        "wavelength_mm": LAM * 1e3,
        "frequency_hz": F_HZ,
        "grid_nx": NX,
        "grid_ny": NY,
        "dx_um": dx * 1e6,
        "dy_um": dy * 1e6,
        "z_slices_mm": Z_SLICES_MM,
        "z_xz_range_mm": [Z_MIN * 1e3, Z_MAX * 1e3],
        "nz_xz": NZ_XZ,
        "peak_magnitudes": {
            f"{z*1e3:.1f}": float(np.abs(fields_at_z[z]).max())
            for z in z_slices
        },
    }
    with open(OUT / "vortex_slices_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("  → saved vortex_slices_summary.json")

    print(f"\n{'='*60}")
    print(f"Done. Outputs → {OUT.relative_to(PROJECT_ROOT)}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
