#!/usr/bin/env python3
"""
Phase 2.4b — Vortex beam comparison (all l = 1).

Compares three vortex beams at the SAME topological charge l = 1:

  1. LG(l=1)          — (r/w)^1 exp(-r²/w²) exp(iθ)
  2. Bessel(l=1)       — uniform disk × exp(-i k_r r) × exp(iθ)
  3. Bessel–Gauss(l=1) — Gaussian × exp(-i k_r r) × exp(iθ)

All beams carry the same phase singularity.

Output → results/deliverables/beams/
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

from scripts.lib.fem_cache_utils import load_fem_cache, LAM
from scripts.lib.asm_utils import (
    make_grid_from_fem,
    make_vortex_field,
    make_lens_phase,
    propagate_asm,
    K0,
)

OUT = PROJECT_ROOT / "results" / "deliverables" / "beams"
OUT.mkdir(parents=True, exist_ok=True)

_mm = lambda v: v * 1e3

# ══════════════════════════════════════════════════════════════════
# Parameters
# ══════════════════════════════════════════════════════════════════
CHARGE = 1
WAIST = 2.0e-3
R_AP = 2.5e-3
AXICON_ANGLE = 15.0
NX, NY = 400, 400

Z_MIN, Z_MAX = 0.5e-3, 9.0e-3
NZ_XZ = 120

BEAMS = [
    {"key": "lg",           "label": "LG (l=1)",           "colour": "tab:orange"},
    {"key": "bessel",       "label": "Bessel (l=1)",       "colour": "tab:green"},
    {"key": "bessel_gauss", "label": "Bessel–Gauss (l=1)", "colour": "tab:red"},
]


# ══════════════════════════════════════════════════════════════════
# Source construction
# ══════════════════════════════════════════════════════════════════

def _make_source(key, XX, YY):
    cx = 0.5 * (XX.min() + XX.max())
    cy = 0.5 * (YY.min() + YY.max())
    r = np.sqrt((XX - cx)**2 + (YY - cy)**2)
    theta = np.arctan2(YY - cy, XX - cx)
    aperture = (r <= R_AP).astype(float)

    if key == "lg":
        # LG_{0,1}: (r/w) exp(-r²/w²) exp(iθ), normalised
        rw = r / WAIST
        amp = rw * np.exp(-r**2 / WAIST**2) * aperture
        amp /= amp.max()
        return amp * np.exp(1j * CHARGE * theta)

    elif key == "bessel":
        # Higher-order Bessel: uniform disk + axicon + vortex charge
        phi_axicon = make_lens_phase(
            XX, YY, family="axicon", aperture_radius=R_AP,
            axicon_angle_deg=AXICON_ANGLE, charge=CHARGE,
        )
        return aperture * np.exp(-1j * phi_axicon)

    elif key == "bessel_gauss":
        # Higher-order Bessel–Gauss: Gaussian + axicon + vortex charge
        amp = np.exp(-r**2 / WAIST**2) * aperture
        phi_axicon = make_lens_phase(
            XX, YY, family="axicon", aperture_radius=R_AP,
            axicon_angle_deg=AXICON_ANGLE, charge=CHARGE,
        )
        return amp * np.exp(-1j * phi_axicon)

    raise ValueError(key)


def main() -> None:
    print("=" * 60)
    print("Phase 2.4b — Vortex Beam Comparison (all l = 1)")
    print("=" * 60)

    cache = load_fem_cache()
    grid = make_grid_from_fem(cache, nx=NX, ny=NY)
    x, y = grid["x"], grid["y"]
    XX, YY = grid["XX"], grid["YY"]
    dx, dy = grid["dx"], grid["dy"]
    iy_centre = NY // 2
    ix_centre = NX // 2

    print(f"Grid: {NX}×{NY}  dx={dx*1e6:.1f} µm")
    print(f"Charge l = {CHARGE}   Waist = {WAIST*1e3:.1f} mm")
    print(f"Axicon angle = {AXICON_ANGLE}°   R_ap = {R_AP*1e3:.1f} mm\n")

    z_xz = np.linspace(Z_MIN, Z_MAX, NZ_XZ)

    all_xz_mag = {}
    all_char_z = {}
    all_char_field = {}
    metrics = []

    for beam in BEAMS:
        key = beam["key"]
        label = beam["label"]
        print(f"── {label} ──")

        source = _make_source(key, XX, YY)

        # Verify on-axis source is ~zero (vortex singularity)
        on_axis_src = np.abs(source[iy_centre, ix_centre])
        print(f"  source on-axis |A| = {on_axis_src:.6f}  (should be ≈ 0)")

        # XZ propagation
        xz_mag = np.zeros((NZ_XZ, NX))
        onaxis = np.zeros(NZ_XZ)
        for iz, z_val in enumerate(z_xz):
            p = propagate_asm(source, dx, dy, wavelength=LAM, z=z_val)
            xz_mag[iz, :] = np.abs(p[iy_centre, :])
            onaxis[iz] = np.abs(p[iy_centre, ix_centre])

        all_xz_mag[key] = xz_mag

        # Characteristic z = where max-in-slice is highest
        char_z = float(z_xz[np.argmax(xz_mag.max(axis=1))])
        all_char_z[key] = char_z
        print(f"  char. z = {char_z*1e3:.2f} mm")

        # XY at characteristic z
        p_char = propagate_asm(source, dx, dy, wavelength=LAM, z=char_z)
        all_char_field[key] = p_char
        mag = np.abs(p_char)
        peak = float(mag.max())
        on_axis_prop = float(np.abs(p_char[iy_centre, ix_centre]))

        # Ring radius (radius of peak)
        radial = mag[iy_centre, ix_centre:]
        r_1d = np.abs(x[ix_centre:] - 0.5 * (x[0] + x[-1]))
        ring_r = float(r_1d[np.argmax(radial)]) if radial.max() > 0 else 0.0

        print(f"  peak |p| = {peak:.4f}  on-axis |p| = {on_axis_prop:.4f}")
        print(f"  ring radius = {ring_r*1e3:.3f} mm")

        metrics.append({
            "beam": key, "label": label,
            "charge": CHARGE,
            "char_z_mm": round(char_z * 1e3, 3),
            "peak_p": round(peak, 5),
            "onaxis_at_char_z": round(on_axis_prop, 5),
            "ring_radius_mm": round(ring_r * 1e3, 4),
        })

    # ══════════════════════════════════════════════════════════════
    # FIGURE: vortex_comparison.png — 3 rows × 3 cols
    #   row: beam type | col: XY mag, XY phase, XZ mag
    # ══════════════════════════════════════════════════════════════
    print("\n── Generating vortex_comparison.png ──")
    n = len(BEAMS)
    vmax_xy = max(np.abs(all_char_field[b["key"]]).max() for b in BEAMS)
    vmax_xz = max(all_xz_mag[b["key"]].max() for b in BEAMS)

    fig, axes = plt.subplots(n, 3, figsize=(15, 5 * n))
    ext_xy = [_mm(x[0]), _mm(x[-1]), _mm(y[0]), _mm(y[-1])]
    ext_xz = [_mm(x[0]), _mm(x[-1]), _mm(z_xz[0]), _mm(z_xz[-1])]

    for i, beam in enumerate(BEAMS):
        key = beam["key"]
        fld = all_char_field[key]
        mag = np.abs(fld)
        phase = np.angle(fld)
        z_mm = all_char_z[key] * 1e3

        # XY magnitude
        im0 = axes[i, 0].imshow(mag, extent=ext_xy, origin="lower",
                                 cmap="inferno", vmin=0, vmax=vmax_xy, aspect="equal")
        axes[i, 0].set_title(f"{beam['label']}  |p|  (z={z_mm:.1f} mm)", fontsize=11)
        if i == n - 1:
            axes[i, 0].set_xlabel("x [mm]")
        axes[i, 0].set_ylabel("y [mm]")

        # XY phase
        im1 = axes[i, 1].imshow(phase, extent=ext_xy, origin="lower",
                                 cmap="twilight", vmin=-np.pi, vmax=np.pi, aspect="equal")
        axes[i, 1].set_title(f"{beam['label']}  phase  (z={z_mm:.1f} mm)", fontsize=11)
        if i == n - 1:
            axes[i, 1].set_xlabel("x [mm]")

        # XZ magnitude
        im2 = axes[i, 2].imshow(all_xz_mag[key], extent=ext_xz, origin="lower",
                                 cmap="inferno", vmin=0, vmax=vmax_xz, aspect="auto")
        axes[i, 2].set_title(f"{beam['label']}  XZ", fontsize=11)
        axes[i, 2].axhline(z_mm, color="w", ls="--", lw=0.8, alpha=0.7)
        if i == n - 1:
            axes[i, 2].set_xlabel("x [mm]")
        axes[i, 2].set_ylabel("z [mm]")

    fig.suptitle("Vortex Beam Comparison — all l = 1", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / "vortex_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {OUT / 'vortex_comparison.png'}")

    # ══════════════════════════════════════════════════════════════
    # Save metrics
    # ══════════════════════════════════════════════════════════════
    summary = {
        "phase": "2.4b",
        "description": "Vortex beam comparison — all l = 1",
        "charge": CHARGE,
        "parameters": {
            "waist_mm": WAIST * 1e3,
            "aperture_radius_mm": R_AP * 1e3,
            "axicon_angle_deg": AXICON_ANGLE,
            "wavelength_mm": round(LAM * 1e3, 4),
        },
        "metrics": metrics,
    }
    (OUT / "vortex_comparison_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(f"  → {OUT / 'vortex_comparison_summary.json'}")

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
