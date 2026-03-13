#!/usr/bin/env python3
"""
Phase 2.4 — Beam Family Deliverables.

Compares four beam types via ASM propagation:

  1. Gaussian          — Gaussian envelope, no vortex
  2. Laguerre–Gaussian — Gaussian envelope + vortex (charge = 1)
  3. Bessel            — Uniform amplitude + axicon phase
  4. Bessel–Gauss      — Gaussian amplitude + axicon phase

For each beam generates:
  - XY slice near waist / characteristic z
  - XZ propagation view
  - Multi-z XY slices
  - On-axis intensity vs z

Also generates a comparison panel with all beams.

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

# ── Project root & imports ─────────────────────────────────────────
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
WAIST = 2.0e-3            # Gaussian waist [m]
R_AP = 2.5e-3             # hard aperture radius [m]
AXICON_ANGLE = 15.0       # axicon half-angle [deg]
NX, NY = 400, 400

# Propagation z-range
Z_MIN = 0.5e-3
Z_MAX = 9.0e-3
NZ_XZ = 120               # z-planes for XZ / on-axis views

# Representative z-planes for multi-z XY slices [mm]
Z_SLICES_MM = [1.0, 2.0, 3.0, 5.0, 7.0]

# Beam definitions
BEAMS = [
    {"key": "gaussian",    "label": "Gaussian",           "colour": "tab:blue"},
    {"key": "lg_vortex",   "label": "Laguerre–Gaussian",  "colour": "tab:orange"},
    {"key": "bessel",      "label": "Bessel",             "colour": "tab:green"},
    {"key": "bessel_gauss","label": "Bessel–Gauss",       "colour": "tab:red"},
]


# ══════════════════════════════════════════════════════════════════
# Source field construction
# ══════════════════════════════════════════════════════════════════

def _make_source(key, XX, YY):
    """Build the source-plane complex field for a given beam type."""
    cx = 0.5 * (XX.min() + XX.max())
    cy = 0.5 * (YY.min() + YY.max())
    r = np.sqrt((XX - cx)**2 + (YY - cy)**2)
    aperture = (r <= R_AP).astype(float)

    if key == "gaussian":
        # Pure Gaussian amplitude, no phase
        amp = np.exp(-r**2 / WAIST**2) * aperture
        return amp.astype(complex)

    elif key == "lg_vortex":
        # Laguerre–Gaussian vortex (charge = 1), no focusing lens
        field = make_vortex_field(
            XX, YY, charge=1, waist=WAIST, aperture_radius=R_AP,
        )
        return field

    elif key == "bessel":
        # Uniform amplitude + axicon phase → Bessel beam
        phi = make_lens_phase(
            XX, YY,
            family="axicon",
            aperture_radius=R_AP,
            axicon_angle_deg=AXICON_ANGLE,
            charge=0,
        )
        return aperture * np.exp(-1j * phi)

    elif key == "bessel_gauss":
        # Gaussian amplitude + axicon phase → Bessel–Gauss beam
        amp = np.exp(-r**2 / WAIST**2) * aperture
        phi = make_lens_phase(
            XX, YY,
            family="axicon",
            aperture_radius=R_AP,
            axicon_angle_deg=AXICON_ANGLE,
            charge=0,
        )
        return amp * np.exp(-1j * phi)

    else:
        raise ValueError(f"Unknown beam key: {key!r}")


# ══════════════════════════════════════════════════════════════════
# Helper metrics
# ══════════════════════════════════════════════════════════════════

def _peak_z(onaxis, z_arr):
    """z at which on-axis |p| is maximum."""
    return float(z_arr[np.argmax(onaxis)])


def _spot_radius(mag_2d, x_1d, y_1d):
    """Half-power radius: maximum r where |p| >= peak/2."""
    peak = mag_2d.max()
    if peak == 0:
        return 0.0
    cx = 0.5 * (x_1d[0] + x_1d[-1])
    cy = 0.5 * (y_1d[0] + y_1d[-1])
    XX, YY = np.meshgrid(x_1d, y_1d)
    r = np.sqrt((XX - cx)**2 + (YY - cy)**2)
    mask = mag_2d >= peak * 0.5
    if mask.any():
        return float(r[mask].max())
    return float(r.max())


def _ring_radius(mag_2d, x_1d, y_1d):
    """Radius of peak intensity (ring radius for vortex, 0 for Gaussian)."""
    cx = 0.5 * (x_1d[0] + x_1d[-1])
    cy = 0.5 * (y_1d[0] + y_1d[-1])
    XX, YY = np.meshgrid(x_1d, y_1d)
    r = np.sqrt((XX - cx)**2 + (YY - cy)**2)
    idx = np.unravel_index(np.argmax(mag_2d), mag_2d.shape)
    return float(r[idx])


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 60)
    print("Phase 2.4 — Beam Family Deliverables")
    print("=" * 60)

    # ── Grid from FEM cache ────────────────────────────────────────
    cache = load_fem_cache()
    grid = make_grid_from_fem(cache, nx=NX, ny=NY)
    x, y = grid["x"], grid["y"]
    XX, YY = grid["XX"], grid["YY"]
    dx, dy = grid["dx"], grid["dy"]
    iy_centre = NY // 2
    ix_centre = NX // 2

    print(f"Grid: {NX}×{NY}  dx={dx*1e6:.1f} µm  dy={dy*1e6:.1f} µm")
    print(f"λ = {LAM*1e3:.4f} mm   k₀ = {K0:.1f} rad/m")
    print(f"Waist = {WAIST*1e3:.1f} mm   R_ap = {R_AP*1e3:.1f} mm")
    print(f"Axicon angle = {AXICON_ANGLE}°\n")

    z_xz = np.linspace(Z_MIN, Z_MAX, NZ_XZ)
    z_slices = [z * 1e-3 for z in Z_SLICES_MM]

    # ── Storage ────────────────────────────────────────────────────
    all_xz_mag = {}       # (NZ, NX)  — central XZ cut
    all_onaxis = {}       # (NZ,)     — on-axis |p|
    all_slice_fields = {} # key → list of (z, p_xy)
    all_char_z = {}       # characteristic z per beam
    all_char_field = {}   # XY field at characteristic z
    metrics = []

    for beam in BEAMS:
        key = beam["key"]
        label = beam["label"]
        print(f"── {label} ──")

        source = _make_source(key, XX, YY)
        print(f"  source max |A| = {np.abs(source).max():.4f}")

        # Propagate through z for XZ and on-axis
        xz_mag = np.zeros((NZ_XZ, NX))
        onaxis = np.zeros(NZ_XZ)

        for iz, z_val in enumerate(z_xz):
            p = propagate_asm(source, dx, dy, wavelength=LAM, z=z_val)
            xz_mag[iz, :] = np.abs(p[iy_centre, :])
            onaxis[iz] = np.abs(p[iy_centre, ix_centre])

        all_xz_mag[key] = xz_mag
        all_onaxis[key] = onaxis

        # Characteristic z: peak on-axis for Gaussian/Bessel types,
        # peak XZ magnitude for vortex
        if key == "lg_vortex":
            char_z = float(z_xz[np.argmax(xz_mag.max(axis=1))])
        else:
            char_z = _peak_z(onaxis, z_xz)
        all_char_z[key] = char_z
        print(f"  characteristic z = {char_z*1e3:.2f} mm")

        # Propagate at characteristic z for XY
        p_char = propagate_asm(source, dx, dy, wavelength=LAM, z=char_z)
        all_char_field[key] = p_char
        mag = np.abs(p_char)
        peak = float(mag.max())
        spot = _spot_radius(mag, x, y)
        ring = _ring_radius(mag, x, y)
        print(f"  peak |p| = {peak:.4f}  spot_r = {spot*1e3:.3f} mm  ring_r = {ring*1e3:.3f} mm")

        # Multi-z slices
        slices = []
        for z_val in z_slices:
            p_z = propagate_asm(source, dx, dy, wavelength=LAM, z=z_val)
            slices.append((z_val, p_z))
        all_slice_fields[key] = slices

        metrics.append({
            "beam": key,
            "label": label,
            "char_z_mm": round(char_z * 1e3, 3),
            "peak_p": round(peak, 5),
            "spot_radius_mm": round(spot * 1e3, 4),
            "ring_radius_mm": round(ring * 1e3, 4),
        })

    # ══════════════════════════════════════════════════════════════
    # FIGURE 1: beam_xy.png — XY at characteristic z (mag + phase)
    # ══════════════════════════════════════════════════════════════
    print("\n── Generating beam_xy.png ──")
    n_beams = len(BEAMS)
    vmax_mag = max(np.abs(all_char_field[b["key"]]).max() for b in BEAMS)

    fig, axes = plt.subplots(2, n_beams, figsize=(5 * n_beams, 9))
    for j, beam in enumerate(BEAMS):
        key = beam["key"]
        fld = all_char_field[key]
        mag = np.abs(fld)
        phase = np.angle(fld)
        z_mm = all_char_z[key] * 1e3

        ext = [_mm(x[0]), _mm(x[-1]), _mm(y[0]), _mm(y[-1])]
        axes[0, j].imshow(mag, extent=ext, origin="lower", cmap="inferno",
                          vmin=0, vmax=vmax_mag, aspect="equal")
        axes[0, j].set_title(f"{beam['label']}\nz = {z_mm:.1f} mm", fontsize=11)
        if j == 0:
            axes[0, j].set_ylabel("|p|  (magnitude)")

        axes[1, j].imshow(phase, extent=ext, origin="lower", cmap="twilight",
                          vmin=-np.pi, vmax=np.pi, aspect="equal")
        if j == 0:
            axes[1, j].set_ylabel("∠p  (phase)")
        axes[1, j].set_xlabel("x [mm]")

    fig.suptitle("Beam Families — XY at Characteristic z", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "beam_xy.png", dpi=200)
    plt.close(fig)
    print(f"  → {OUT / 'beam_xy.png'}")

    # ══════════════════════════════════════════════════════════════
    # FIGURE 2: beam_xz.png — XZ propagation (central y=0 cut)
    # ══════════════════════════════════════════════════════════════
    print("── Generating beam_xz.png ──")
    vmax_xz = max(all_xz_mag[b["key"]].max() for b in BEAMS)

    fig, axes = plt.subplots(1, n_beams, figsize=(5 * n_beams, 4))
    for j, beam in enumerate(BEAMS):
        key = beam["key"]
        ext = [_mm(x[0]), _mm(x[-1]), _mm(z_xz[0]), _mm(z_xz[-1])]
        axes[j].imshow(all_xz_mag[key], extent=ext, origin="lower",
                       cmap="inferno", vmin=0, vmax=vmax_xz, aspect="auto")
        axes[j].set_title(beam["label"], fontsize=11)
        axes[j].set_xlabel("x [mm]")
        if j == 0:
            axes[j].set_ylabel("z [mm]")
        # Mark characteristic z
        axes[j].axhline(_mm(all_char_z[key]), color="w", ls="--",
                        lw=0.8, alpha=0.7)

    fig.suptitle("Beam Families — XZ Propagation (y = 0)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "beam_xz.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {OUT / 'beam_xz.png'}")

    # ══════════════════════════════════════════════════════════════
    # FIGURE 3: beam_multiz.png — multi-z XY slices per beam
    # ══════════════════════════════════════════════════════════════
    print("── Generating beam_multiz.png ──")
    n_z = len(z_slices)
    fig, axes = plt.subplots(n_beams, n_z, figsize=(4 * n_z, 4 * n_beams))
    for i, beam in enumerate(BEAMS):
        key = beam["key"]
        slices = all_slice_fields[key]
        local_vmax = max(np.abs(s[1]).max() for s in slices)
        for j, (z_val, p_z) in enumerate(slices):
            mag = np.abs(p_z)
            ext = [_mm(x[0]), _mm(x[-1]), _mm(y[0]), _mm(y[-1])]
            axes[i, j].imshow(mag, extent=ext, origin="lower", cmap="inferno",
                              vmin=0, vmax=local_vmax, aspect="equal")
            if i == 0:
                axes[i, j].set_title(f"z = {z_val*1e3:.0f} mm", fontsize=10)
            if j == 0:
                axes[i, j].set_ylabel(f"{beam['label']}\ny [mm]", fontsize=9)
            else:
                axes[i, j].set_yticklabels([])
            if i == n_beams - 1:
                axes[i, j].set_xlabel("x [mm]")
            else:
                axes[i, j].set_xticklabels([])

    fig.suptitle("Beam Families — Multi-z XY Magnitude", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / "beam_multiz.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {OUT / 'beam_multiz.png'}")

    # ══════════════════════════════════════════════════════════════
    # FIGURE 4: beam_comparison.png — overlay of on-axis |p| vs z
    #            + radial profiles at characteristic z
    # ══════════════════════════════════════════════════════════════
    print("── Generating beam_comparison.png ──")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # (a) On-axis intensity vs z
    for beam in BEAMS:
        key = beam["key"]
        if key == "lg_vortex":
            # Vortex has zero on-axis; show max-in-slice instead
            ax1.plot(_mm(z_xz), all_xz_mag[key].max(axis=1),
                     label=beam["label"] + " (max in slice)",
                     color=beam["colour"], lw=1.8)
        else:
            ax1.plot(_mm(z_xz), all_onaxis[key],
                     label=beam["label"],
                     color=beam["colour"], lw=1.8)
    ax1.set_xlabel("z [mm]")
    ax1.set_ylabel("|p|")
    ax1.set_title("On-axis / Peak Intensity vs z")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # (b) Radial profile at characteristic z
    r_1d = np.abs(x - 0.5 * (x[0] + x[-1]))  # x distance from centre
    for beam in BEAMS:
        key = beam["key"]
        fld = all_char_field[key]
        radial = np.abs(fld[iy_centre, :])
        peak = radial.max()
        if peak > 0:
            radial = radial / peak
        ax2.plot(_mm(r_1d), radial,
                 label=f"{beam['label']} (z={all_char_z[key]*1e3:.1f} mm)",
                 color=beam["colour"], lw=1.8)
    ax2.set_xlabel("r [mm]")
    ax2.set_ylabel("|p| / max")
    ax2.set_title("Radial Profile at Characteristic z")
    ax2.legend(fontsize=9)
    ax2.set_xlim(0, _mm(R_AP))
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Beam Family Comparison", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "beam_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {OUT / 'beam_comparison.png'}")

    # ══════════════════════════════════════════════════════════════
    # Save metrics JSON
    # ══════════════════════════════════════════════════════════════
    summary = {
        "phase": "2.4",
        "description": "Beam family deliverables",
        "parameters": {
            "waist_mm": WAIST * 1e3,
            "aperture_radius_mm": R_AP * 1e3,
            "axicon_angle_deg": AXICON_ANGLE,
            "grid_nx": NX,
            "grid_ny": NY,
            "z_min_mm": Z_MIN * 1e3,
            "z_max_mm": Z_MAX * 1e3,
            "nz_xz": NZ_XZ,
            "z_slices_mm": Z_SLICES_MM,
            "wavelength_mm": round(LAM * 1e3, 4),
        },
        "metrics": metrics,
    }
    (OUT / "beam_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  → {OUT / 'beam_summary.json'}")

    # ══════════════════════════════════════════════════════════════
    # Save XZ data for reuse
    # ══════════════════════════════════════════════════════════════
    np.savez_compressed(
        OUT / "beam_xz_data.npz",
        z_mm=_mm(z_xz),
        x_mm=_mm(x),
        **{f"xz_{b['key']}": all_xz_mag[b["key"]] for b in BEAMS},
        **{f"onaxis_{b['key']}": all_onaxis[b["key"]] for b in BEAMS},
    )
    print(f"  → {OUT / 'beam_xz_data.npz'}")

    print("\n" + "=" * 60)
    print("Phase 2.4 complete — all beam family deliverables saved.")
    print("=" * 60)


if __name__ == "__main__":
    main()
