#!/usr/bin/env python3
"""
Phase 3.1 — Calibrate ASM perturbations to the trap layer z*.

This script calibrates vortex-family ASM perturbations on the FEM particle-plane
XY grid and reports whether the ring radius at z* matches the local-overlay target:

  target ring radius = 0.50 mm  (diameter ~1.0 mm)

Outputs are written to:
  results/deliverables/overlay_local/
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

from scripts.lib.fem_cache_utils import load_fem_cache, LAM
from scripts.lib.asm_utils import (
    make_grid_from_fem,
    make_lens_phase,
    make_cshape_mask,
    propagate_asm,
)
from scripts.lib.overlay_utils import estimate_ring_radius

OUT = PROJECT_ROOT / "results" / "deliverables" / "overlay_local"
OUT.mkdir(parents=True, exist_ok=True)

_mm = lambda v: v * 1e3

# ── Calibration targets ────────────────────────────────────────────
TARGET_RING_RADIUS = 0.50e-3
RING_TOL = 0.10e-3
CHARGE = 1
R_AP = 2.5e-3

# Coarse calibration grid (faster than full 400 while still informative)
NX_CAL, NY_CAL = 260, 260
NX_FINAL, NY_FINAL = 400, 400
Z_MIN, Z_MAX = 1.0e-3, 9.0e-3
NZ_SCAN = 60

# Candidate parameters (Bessel-Gauss vortex)
ANGLE_VALUES_DEG = [18.0, 22.0, 26.0, 30.0, 34.0, 36.0, 38.0, 40.0]
WAIST_VALUES_MM = [2.2, 2.6, 3.0]


# ───────────────────────────────────────────────────────────────────
def build_bessel_gauss_vortex_source(
    XX: np.ndarray,
    YY: np.ndarray,
    waist: float,
    angle_deg: float,
    charge: int = 1,
) -> np.ndarray:
    """Gaussian envelope + axicon phase + vortex charge."""
    cx = 0.5 * (XX.min() + XX.max())
    cy = 0.5 * (YY.min() + YY.max())
    r = np.sqrt((XX - cx) ** 2 + (YY - cy) ** 2)

    amp = np.exp(-(r ** 2) / (waist ** 2))
    amp[r > R_AP] = 0.0

    phi = make_lens_phase(
        XX,
        YY,
        family="axicon",
        aperture_radius=R_AP,
        axicon_angle_deg=angle_deg,
        charge=charge,
    )
    return amp * np.exp(-1j * phi)


def characterize_source(
    source: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    dx: float,
    dy: float,
    z_star: float,
) -> dict:
    """Compute characteristic z and ring metrics at z*."""
    z_scan = np.linspace(Z_MIN, Z_MAX, NZ_SCAN)

    # Characteristic plane = z where max(|p|) in the XY slice is highest.
    max_per_z = np.zeros(NZ_SCAN)
    for iz, z in enumerate(z_scan):
        pz = propagate_asm(source, dx, dy, wavelength=LAM, z=z)
        max_per_z[iz] = float(np.max(np.abs(pz)))

    iz_char = int(np.argmax(max_per_z))
    z_char = float(z_scan[iz_char])

    p_star = propagate_asm(source, dx, dy, wavelength=LAM, z=z_star)
    mag_star = np.abs(p_star)

    # Use first-ring window (r <= 1.0 mm) for local-overlay targeting.
    ring = estimate_ring_radius(
        mag_star,
        x,
        y,
        r_min=0.10e-3,
        r_max=1.00e-3,
        n_bins=220,
    )
    ring_r = float(ring["ring_radius_m"])
    peak_star = float(np.max(mag_star))

    return {
        "z_char_m": z_char,
        "z_scan_m": z_scan,
        "max_per_z": max_per_z,
        "p_star": p_star,
        "mag_star": mag_star,
        "ring_radius_m": ring_r,
        "ring_profile_r_m": ring["r_m"],
        "ring_profile": ring["profile"],
        "peak_star": peak_star,
    }


def score_candidate(ring_r: float, z_char: float, z_star: float) -> float:
    """Lower is better: prioritize radius match, then mild z alignment."""
    ring_err = abs(ring_r - TARGET_RING_RADIUS)
    z_err = abs(z_char - z_star)
    return float(ring_err + 0.01 * z_err)


# ───────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 64)
    print("Phase 3.1 — ASM calibration to z*")
    print("=" * 64)

    cache = load_fem_cache()
    z_star = float(cache["z_star"])

    grid = make_grid_from_fem(cache, nx=NX_CAL, ny=NY_CAL)
    x, y = grid["x"], grid["y"]
    XX, YY = grid["XX"], grid["YY"]
    dx, dy = grid["dx"], grid["dy"]

    print(f"Grid: {NX_CAL}x{NY_CAL}  dx={dx*1e6:.1f} um")
    print(f"z* = {z_star*1e3:.3f} mm")
    print(
        f"Target vortex ring radius at z*: {TARGET_RING_RADIUS*1e3:.3f} mm "
        f"(+/- {RING_TOL*1e3:.3f} mm)"
    )

    results = []
    traces = []

    print("\nScanning Bessel-Gauss vortex candidates ...")
    for waist_mm in WAIST_VALUES_MM:
        waist = waist_mm * 1e-3
        for angle in ANGLE_VALUES_DEG:
            source = build_bessel_gauss_vortex_source(
                XX,
                YY,
                waist=waist,
                angle_deg=angle,
                charge=CHARGE,
            )
            m = characterize_source(source, x, y, dx, dy, z_star)
            score = score_candidate(m["ring_radius_m"], m["z_char_m"], z_star)

            row = {
                "family": "bessel_gauss_vortex_l1",
                "charge": CHARGE,
                "waist_mm": waist_mm,
                "axicon_angle_deg": angle,
                "z_char_mm": _mm(m["z_char_m"]),
                "z_star_mm": _mm(z_star),
                "ring_radius_zstar_mm": _mm(m["ring_radius_m"]),
                "ring_error_mm": _mm(abs(m["ring_radius_m"] - TARGET_RING_RADIUS)),
                "peak_abs_p_zstar": m["peak_star"],
                "score": score,
            }
            results.append(row)
            traces.append((waist_mm, angle, m))

            print(
                "  waist={:.2f} mm  angle={:.1f} deg  z_char={:.2f} mm  "
                "ring(z*)={:.3f} mm  peak(z*)={:.3f}".format(
                    waist_mm,
                    angle,
                    _mm(m["z_char_m"]),
                    _mm(m["ring_radius_m"]),
                    m["peak_star"],
                )
            )

    results = sorted(results, key=lambda r: r["score"])
    best = results[0]

    # Final verification on full-resolution overlay grid.
    grid_f = make_grid_from_fem(cache, nx=NX_FINAL, ny=NY_FINAL)
    x_f, y_f = grid_f["x"], grid_f["y"]
    XX_f, YY_f = grid_f["XX"], grid_f["YY"]
    dx_f, dy_f = grid_f["dx"], grid_f["dy"]
    source_best_f = build_bessel_gauss_vortex_source(
        XX_f,
        YY_f,
        waist=best["waist_mm"] * 1e-3,
        angle_deg=best["axicon_angle_deg"],
        charge=CHARGE,
    )
    m_final = characterize_source(source_best_f, x_f, y_f, dx_f, dy_f, z_star)

    best["z_char_mm_final_grid"] = _mm(m_final["z_char_m"])
    best["ring_radius_zstar_mm_final_grid"] = _mm(m_final["ring_radius_m"])
    best["peak_abs_p_zstar_final_grid"] = m_final["peak_star"]

    print("\nBest vortex candidate:")
    print(
        "  waist={:.2f} mm, angle={:.1f} deg, z_char={:.2f} mm, "
        "ring(z*)={:.3f} mm".format(
            best["waist_mm"],
            best["axicon_angle_deg"],
            best["z_char_mm_final_grid"],
            best["ring_radius_zstar_mm_final_grid"],
        )
    )

    within_tol = abs(best["ring_radius_zstar_mm_final_grid"] * 1e-3 - TARGET_RING_RADIUS) <= RING_TOL

    # Optional second perturbation family characterization (C-shape)
    print("\nCharacterizing C-shape reference field at z* ...")
    source_c = make_cshape_mask(
        XX,
        YY,
        radius=TARGET_RING_RADIUS,
        gap_angle=0.0,
        thickness=0.14e-3,
        charge=1,
        gap_width=0.40,
        beta=1.0,
    )
    cx = 0.5 * (XX.min() + XX.max())
    cy = 0.5 * (YY.min() + YY.max())
    rr = np.sqrt((XX - cx) ** 2 + (YY - cy) ** 2)
    source_c = source_c * (rr <= R_AP)

    c_metrics = characterize_source(source_c, x, y, dx, dy, z_star)

    # ── Save CSV table ─────────────────────────────────────────────
    csv_path = OUT / "overlay_vortex_calibration_candidates.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"Saved {csv_path}")

    # ── Diagnostic figure ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    # (1) Ring radius vs angle for each waist
    ax = axes[0]
    for waist_mm in WAIST_VALUES_MM:
        sub = [r for r in results if abs(r["waist_mm"] - waist_mm) < 1e-9]
        sub = sorted(sub, key=lambda r: r["axicon_angle_deg"])
        ax.plot(
            [r["axicon_angle_deg"] for r in sub],
            [r["ring_radius_zstar_mm"] for r in sub],
            marker="o",
            lw=1.5,
            label=f"w={waist_mm:.1f} mm",
        )
    ax.axhline(_mm(TARGET_RING_RADIUS), color="k", ls="--", lw=1.0, label="target")
    ax.axhspan(_mm(TARGET_RING_RADIUS - RING_TOL), _mm(TARGET_RING_RADIUS + RING_TOL),
               color="gray", alpha=0.15, label="tolerance")
    ax.set_xlabel("Axicon angle [deg]")
    ax.set_ylabel("Ring radius at z* [mm]")
    ax.set_title("Vortex ring-radius calibration")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # (2) Characteristic z for candidates
    ax = axes[1]
    for waist_mm in WAIST_VALUES_MM:
        sub = [r for r in results if abs(r["waist_mm"] - waist_mm) < 1e-9]
        sub = sorted(sub, key=lambda r: r["axicon_angle_deg"])
        ax.plot(
            [r["axicon_angle_deg"] for r in sub],
            [r["z_char_mm"] for r in sub],
            marker="o",
            lw=1.5,
            label=f"w={waist_mm:.1f} mm",
        )
    ax.axhline(_mm(z_star), color="k", ls="--", lw=1.0, label="z*")
    ax.set_xlabel("Axicon angle [deg]")
    ax.set_ylabel("Characteristic z [mm]")
    ax.set_title("Characteristic plane alignment")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # (3) Radial profile at z* for best candidate
    best_trace = None
    for w_mm, a_deg, m in traces:
        if abs(w_mm - best["waist_mm"]) < 1e-9 and abs(a_deg - best["axicon_angle_deg"]) < 1e-9:
            best_trace = m
            break

    ax = axes[2]
    if best_trace is not None:
        r_mm = _mm(best_trace["ring_profile_r_m"])
        prof = best_trace["ring_profile"]
        prof = prof / max(float(np.max(prof)), 1e-30)
        ax.plot(r_mm, prof, lw=1.8, color="tab:red")
    ax.axvline(_mm(TARGET_RING_RADIUS), color="k", ls="--", lw=1.0, label="target")
    ax.axvspan(_mm(TARGET_RING_RADIUS - RING_TOL), _mm(TARGET_RING_RADIUS + RING_TOL),
               color="gray", alpha=0.15)
    ax.set_xlim(0.0, 1.8)
    ax.set_xlabel("Radius [mm]")
    ax.set_ylabel("Normalized radial |p|")
    ax.set_title("Best candidate radial profile at z*")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle("Phase 3.1 Calibration Diagnostics", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "overlay_vortex_calibration.png", dpi=220)
    plt.close(fig)
    print(f"Saved {OUT / 'overlay_vortex_calibration.png'}")

    # ── Save summary JSON ──────────────────────────────────────────
    summary = {
        "phase": "3.1",
        "goal": "Calibrate ASM perturbations at trapping layer z*",
        "z_star_mm": _mm(z_star),
        "target_vortex_ring_radius_mm": _mm(TARGET_RING_RADIUS),
        "tolerance_mm": _mm(RING_TOL),
        "selected_vortex": best,
        "selected_within_tolerance": bool(within_tol),
        "cshape_reference": {
            "family": "cshape_l1",
            "z_char_mm": _mm(c_metrics["z_char_m"]),
            "ring_radius_zstar_mm": _mm(c_metrics["ring_radius_m"]),
            "peak_abs_p_zstar": c_metrics["peak_star"],
            "params": {
                "radius_mm": _mm(TARGET_RING_RADIUS),
                "thickness_mm": 0.14,
                "gap_angle_rad": 0.0,
                "gap_width_rad": 0.40,
                "beta": 1.0,
                "charge": 1,
            },
        },
        "n_candidates": len(results),
        "candidates_csv": str(csv_path.relative_to(PROJECT_ROOT)),
    }

    summary_path = OUT / "overlay_calibration_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved {summary_path}")

    print("\nCalibration complete.")


if __name__ == "__main__":
    main()
