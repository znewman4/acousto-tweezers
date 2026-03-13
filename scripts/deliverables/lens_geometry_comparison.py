#!/usr/bin/env python3
"""
Phase 2.3A — Lens geometry comparison.

Generates phase-map figures for each lens family:
  - Flat (no lens)
  - Ideal spherical
  - Plastic (wrapped / fabricable)
  - Axicon (Bessel-generating)

Each family is shown with charge=0 (pure focusing) on the FEM-compatible grid.

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
from scripts.lib.asm_utils import make_grid_from_fem, make_lens_phase, K0

OUT = PROJECT_ROOT / "results" / "deliverables" / "lens"
OUT.mkdir(parents=True, exist_ok=True)

_mm = lambda v: v * 1e3

# ══════════════════════════════════════════════════════════════════
# Parameters
# ══════════════════════════════════════════════════════════════════
FOCAL_LENGTH = 5.0e-3      # 5 mm
R_AP = 2.5e-3              # 2.5 mm aperture
AXICON_ANGLE = 15.0        # degrees
NX, NY = 400, 400

FAMILIES = [
    {"name": "Flat",    "family": None,      "label": "Flat (no lens)"},
    {"name": "Ideal",   "family": "ideal",   "label": f"Ideal  (f = {FOCAL_LENGTH*1e3:.0f} mm)"},
    {"name": "Plastic", "family": "plastic", "label": f"Plastic  (f = {FOCAL_LENGTH*1e3:.0f} mm, wrapped)"},
    {"name": "Axicon",  "family": "axicon",  "label": f"Axicon  (α = {AXICON_ANGLE}°)"},
]


def main() -> None:
    print("=" * 60)
    print("Phase 2.3A — Lens geometry comparison")
    print("=" * 60)

    cache = load_fem_cache()
    grid = make_grid_from_fem(cache, nx=NX, ny=NY)
    x, y = grid["x"], grid["y"]
    XX, YY = grid["XX"], grid["YY"]
    dx, dy = grid["dx"], grid["dy"]

    print(f"Grid: {NX}×{NY}  dx={dx*1e6:.1f} µm  dy={dy*1e6:.1f} µm")
    print(f"R_ap = {R_AP*1e3:.1f} mm  f = {FOCAL_LENGTH*1e3:.1f} mm  "
          f"λ = {LAM*1e3:.3f} mm\n")

    phases = {}
    for fam in FAMILIES:
        name = fam["name"]
        if fam["family"] is None:
            phases[name] = np.zeros_like(XX)
        else:
            phases[name] = make_lens_phase(
                XX, YY,
                focal_length=FOCAL_LENGTH,
                aperture_radius=R_AP,
                family=fam["family"],
                k=K0,
                charge=0,
                axicon_angle_deg=AXICON_ANGLE,
            )
        pmax = phases[name].max()
        print(f"  {name:<10} φ_max = {pmax:.2f} rad  "
              f"({pmax/(2*np.pi):.2f} × 2π)")

    # ── 4-panel comparison figure ──────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    axes = axes.ravel()

    for i, fam in enumerate(FAMILIES):
        name = fam["name"]
        ax = axes[i]
        phi = phases[name]
        if name == "Flat":
            im = ax.pcolormesh(_mm(x), _mm(y), phi, shading="auto",
                               cmap="twilight", vmin=0, vmax=2*np.pi)
        else:
            im = ax.pcolormesh(_mm(x), _mm(y), phi, shading="auto",
                               cmap="twilight")
        ax.set_title(fam["label"], fontsize=11, fontweight="bold")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, shrink=0.82, label="φ [rad]")

        # Draw aperture circle
        theta = np.linspace(0, 2 * np.pi, 200)
        cx = 0.5 * (x[0] + x[-1])
        cy = 0.5 * (y[0] + y[-1])
        ax.plot(cx * 1e3 + R_AP * 1e3 * np.cos(theta),
                cy * 1e3 + R_AP * 1e3 * np.sin(theta),
                "w--", lw=0.8, alpha=0.7)

    fig.suptitle(
        f"Lens phase profiles  (R_ap = {R_AP*1e3:.1f} mm, λ = {LAM*1e3:.3f} mm)",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    fig.savefig(OUT / "lens_phase_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("\n→ saved lens_phase_comparison.png")

    # ── Radial phase profiles (1-D) ───────────────────────────────
    cx_val = 0.5 * (x[0] + x[-1])
    iy_c = NY // 2
    r_line = x - cx_val

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colours = {"Flat": "gray", "Ideal": "tab:blue",
               "Plastic": "tab:orange", "Axicon": "tab:green"}
    for fam in FAMILIES:
        name = fam["name"]
        phi_row = phases[name][iy_c, :]
        ax.plot(_mm(r_line), phi_row, color=colours[name],
                lw=1.8, label=fam["label"])

    ax.set_xlabel("r  [mm]", fontsize=12)
    ax.set_ylabel("φ  [rad]", fontsize=12)
    ax.set_title("Radial lens phase profiles", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.axvline(R_AP * 1e3, color="k", ls=":", lw=0.8, alpha=0.5)
    ax.axvline(-R_AP * 1e3, color="k", ls=":", lw=0.8, alpha=0.5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "lens_radial_profiles.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("→ saved lens_radial_profiles.png")

    # ── Equivalent thickness map (plastic lens) ────────────────────
    # For a plastic lens, physical thickness ∝ φ / Δk where
    # Δk = k_lens - k_water.  We show normalised height φ / (2π).
    phi_plastic = phases["Plastic"]
    height_norm = phi_plastic / (2.0 * np.pi)  # in units of "wavelength wraps"

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.pcolormesh(_mm(x), _mm(y), height_norm, shading="auto",
                       cmap="viridis")
    ax.set_title("Plastic lens — normalised height  (φ / 2π)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax, shrink=0.82, label="phase wraps")
    fig.tight_layout()
    fig.savefig(OUT / "lens_plastic_height.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("→ saved lens_plastic_height.png")

    # ── Summary JSON ──────────────────────────────────────────────
    summary = {
        "focal_length_mm": FOCAL_LENGTH * 1e3,
        "aperture_radius_mm": R_AP * 1e3,
        "axicon_angle_deg": AXICON_ANGLE,
        "wavelength_mm": LAM * 1e3,
        "grid_nx": NX, "grid_ny": NY,
        "dx_um": dx * 1e6, "dy_um": dy * 1e6,
        "families": [f["name"] for f in FAMILIES],
        "max_phase_rad": {
            f["name"]: float(phases[f["name"]].max()) for f in FAMILIES
        },
    }
    with open(OUT / "geometry_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("→ saved geometry_summary.json")

    print(f"\n{'='*60}")
    print(f"Done. Outputs → {OUT.relative_to(PROJECT_ROOT)}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
