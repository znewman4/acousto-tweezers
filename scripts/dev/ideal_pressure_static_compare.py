#!/usr/bin/env python3
"""
Static comparison figure: standing-wave + synthetic ideal pressure perturbation.

Purpose:
- Load cached standing-wave field (from existing transport study)
- Construct a smooth synthetic ideal pressure perturbation
- Visualize the combined static field without transport or GIF
- Save full-domain and ROI comparison views

Design of the ideal perturbation:
- Anchor component centred at B (holds destination in place)
- Source component near A (creates top of slope)
- Optional side-confinement components (keep minimal)
- Overlay onto standing-wave baseline for direct comparison

This is a development script for exploring perturbation geometry before integrating
into the GIF workflow. Does not modify canonical deliverable scripts.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lib.fem_cache_utils import (
    C_WATER,
    OMEGA,
    RHO0,
    default_particle_params,
    gorkov_grid_2d,
)

PPAR = default_particle_params()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUT_DIR = PROJECT_ROOT / "results" / "dev" / "ideal_pressure_static_compare"

VORTEX_NPZ = (
    PROJECT_ROOT / "results" / "deliverables" / "vortex_stage_transport"
    / "transport" / "transport_case_for_gif.npz"
)
OVERLAY_NPZ = (
    PROJECT_ROOT / "results" / "c_shape_lens_15mm_overlay_study_20260310_170620"
    / "npz" / "roi_fields.npz"
)

# ---------------------------------------------------------------------------
# Ideal Perturbation Design Parameters (tunable)
# ---------------------------------------------------------------------------
# Minimal, highly localized perturbation: anchor at B only
# Purpose: Create local A→B bias without globally disturbing the field

# Anchor component at B: local minimum (attractive sink)
# Use NEGATIVE pressure to reduce field magnitude and create attractive well
ANCHOR_RADIUS_M = 8.0e-5  # 80 µm - very local scale
ANCHOR_AMPLITUDE_PA = -150.0  # Pa - negative to deepen potential well at B

# Slight destabilization at A: very weak and local
# Use small positive pressure to slightly increase confinement difficulty
SOURCE_RADIUS_M = 8.0e-5  # 80 µm - matches anchor scale
SOURCE_AMPLITUDE_PA = 50.0  # Pa - weak, only slight effect

# All other components disabled for minimal, local effect
SIDE_CONF_ENABLED = False  # Disable lateral confinement
BLEND_LENGTH_M = 0.0  # No ramp blending

# ---------------------------------------------------------------------------
# Visualization styles (reuse from canonical script)
# ---------------------------------------------------------------------------
CMAP = "RdBu_r"
IMSHOW_INTERP = "bicubic"
GORKOV_CLIP_LO = 0.5
GORKOV_CLIP_HI = 99.5

COL_A = "#e74c3c"
COL_B = "#3498db"
COL_NEIGH = "#95a5a6"
COL_HOME = "#2ecc71"
COL_CTR = "#f39c12"


# ---------------------------------------------------------------------------
# Synthetic Perturbation Construction
# ---------------------------------------------------------------------------
def gaussian_blob(
    x: np.ndarray,
    y: np.ndarray,
    center_x: float,
    center_y: float,
    radius: float,
    amplitude: float,
) -> np.ndarray:
    """
    Create a smooth Gaussian pressure blob on a 2D grid.

    Parameters
    ----------
    x, y : 1D arrays
        Coordinate grids [m]
    center_x, center_y : float
        Blob centre position [m]
    radius : float
        Gaussian width (1/e decay distance) [m]
    amplitude : float
        Peak pressure [Pa]

    Returns
    -------
    p : (ny, nx) array
        Pressure field [Pa]
    """
    yy, xx = np.meshgrid(y, x, indexing="ij")
    r_sq = (xx - center_x) ** 2 + (yy - center_y) ** 2
    p = amplitude * np.exp(-r_sq / (2.0 * radius**2))
    return p


def build_ideal_perturbation(
    x: np.ndarray,
    y: np.ndarray,
    point_a: np.ndarray,
    point_b: np.ndarray,
) -> np.ndarray:
    """
    Build a minimal synthetic ideal pressure perturbation field.

    Two highly local Gaussian components:
    - Negative (anchoring) at B: creates attractive local minimum
    - Weak positive at A: slight local destabilization for asymmetry
    
    No ramp, no side confinement - keeps effect very local.

    Parameters
    ----------
    x, y : 1D arrays
        Coordinate grids [m]
    point_a : 1D array
        Position of particle A (top of slope) [m]
    point_b : 1D array
        Position of particle B (anchor/destination) [m]

    Returns
    -------
    p_ideal : (ny, nx) complex array
        Ideal pressure perturbation field

    """
    p_ideal = np.zeros((len(y), len(x)), dtype=float)

    # Component: Anchor at B (negative well creates attractive minimum)
    p_anchor = gaussian_blob(x, y, point_b[0], point_b[1], ANCHOR_RADIUS_M, ANCHOR_AMPLITUDE_PA)
    p_ideal += p_anchor

    # Component: Slight destabilization at A (weak positive perturbation)
    p_source = gaussian_blob(x, y, point_a[0], point_a[1], SOURCE_RADIUS_M, SOURCE_AMPLITUDE_PA)
    p_ideal += p_source

    return p_ideal.astype(complex)


# ---------------------------------------------------------------------------
# Rendering helpers (reuse from canonical scripts)
# ---------------------------------------------------------------------------
def _gorkov_fields(p_total: np.ndarray, dx: float, dy: float):
    return gorkov_grid_2d(
        p_total,
        dx,
        dy,
        OMEGA,
        RHO0,
        C_WATER,
        PPAR["a"],
        PPAR["f1"],
        PPAR["f2"],
    )


def _auto_limits(*arrays: np.ndarray) -> tuple[float, float]:
    combined = np.concatenate([a.ravel() for a in arrays])
    return (
        float(np.percentile(combined, GORKOV_CLIP_LO)),
        float(np.percentile(combined, GORKOV_CLIP_HI)),
    )


def _crop_indices(
    x_full: np.ndarray,
    y_full: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    ix = np.where((x_full >= x_min) & (x_full <= x_max))[0]
    iy = np.where((y_full >= y_min) & (y_full <= y_max))[0]
    if ix.size < 2 or iy.size < 2:
        raise RuntimeError("ROI crop indices are empty; check ROI/full-grid overlap")
    return ix, iy


def _draw_static_panel(
    ax,
    field2d: np.ndarray,
    vmin: float,
    vmax: float,
    extent: list[float],
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    traps_mm: np.ndarray,
    idx_a: int,
    idx_b: int,
    neigh_idx: np.ndarray,
    title: str,
) -> None:
    """
    Draw a single static field panel (no time evolution).

    Parameters
    ----------
    ax : matplotlib axis
        Axis to draw on
    field2d : (ny, nx) array
        Field to visualize (e.g., Gor'kov potential)
    vmin, vmax : float
        Color scale limits
    extent : [xmin, xmax, ymin, ymax]
        Spatial extent in [mm]
    x_mm, y_mm : 1D arrays
        Coordinate arrays in [mm]
    traps_mm : (n_traps, 2) array
        Trap positions in [mm]
    idx_a, idx_b : int
        Indices of particles A and B
    neigh_idx : array
        Indices of neighboring particles
    title : str
        Panel title
    """
    ax.imshow(
        field2d,
        origin="lower",
        extent=extent,
        cmap=CMAP,
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
        interpolation=IMSHOW_INTERP,
    )

    # Home positions (green x markers)
    ax.scatter(
        traps_mm[:, 0],
        traps_mm[:, 1],
        marker="x",
        c=COL_HOME,
        s=28,
        linewidths=0.8,
        zorder=4,
        label="home",
    )

    # Neighbors (gray circles)
    ax.scatter(traps_mm[neigh_idx, 0], traps_mm[neigh_idx, 1], c=COL_NEIGH, s=28, zorder=6)

    # Particles A and B (red and blue circles)
    ax.scatter([traps_mm[idx_b, 0]], [traps_mm[idx_b, 1]], c=COL_B, s=44, zorder=7, label="B (dest)")
    ax.scatter([traps_mm[idx_a, 0]], [traps_mm[idx_a, 1]], c=COL_A, s=44, zorder=7, label="A (src)")

    ax.set_xlim(x_mm[0], x_mm[-1])
    ax.set_ylim(y_mm[0], y_mm[-1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.legend(loc="upper right", fontsize=8)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("IDEAL PRESSURE STATIC COMPARISON")
    print("=" * 70)

    # Load cached fields
    print("\n1. Loading cached transport data...")
    ov = np.load(OVERLAY_NPZ)
    x_roi = ov["xg"].astype(float)
    y_roi = ov["yg"].astype(float)
    traps_m = ov["traps_m"].astype(float)
    idx_A = int(ov["idx_A"])
    idx_B = int(ov["idx_B"])

    vd = np.load(VORTEX_NPZ)
    x_full = vd["xg"].astype(float)
    y_full = vd["yg"].astype(float)
    p_sw_full = vd["p_sw"].astype(complex)

    neigh_idx = np.array(sorted(set(range(len(traps_m))) - {idx_A, idx_B}), dtype=int)

    A_xy = traps_m[idx_A]
    B_xy = traps_m[idx_B]

    print(f"  Trap A (source, idx={idx_A}): {A_xy * 1e3} mm")
    print(f"  Trap B (destination, idx={idx_B}): {B_xy * 1e3} mm")
    print(f"  Full domain: [{x_full[0] * 1e3:.3f}, {x_full[-1] * 1e3:.3f}] mm")
    print(f"  ROI bounds:  [{x_roi[0] * 1e3:.3f}, {x_roi[-1] * 1e3:.3f}] mm")
    print(f"  Distance A->B: {np.linalg.norm(B_xy - A_xy) * 1e3:.2f} mm")

    # Build synthetic ideal perturbation
    print("\n2. Building minimal synthetic ideal pressure perturbation...")
    print(f"  Design philosophy: highly local, no global field disturbance")
    print(f"  Anchor at B: {ANCHOR_RADIUS_M * 1e6:.0f} µm radius, {ANCHOR_AMPLITUDE_PA:.0f} Pa (negative sink)")
    print(f"  Destabilization at A: {SOURCE_RADIUS_M * 1e6:.0f} µm radius, {SOURCE_AMPLITUDE_PA:.0f} Pa (weak)")
    print(f"  Ramp component: disabled")
    print(f"  Side confinement: disabled")

    p_ideal_full = build_ideal_perturbation(x_full, y_full, A_xy, B_xy)
    print(f"  Ideal field peak: {np.max(np.abs(p_ideal_full)):.2f} Pa")

    # Compute Gor'kov potentials
    print("\n3. Computing Gor'kov potentials...")
    dx_full = float(x_full[1] - x_full[0])
    dy_full = float(y_full[1] - y_full[0])

    # Standing wave only
    U_sw_full, _, _ = _gorkov_fields(p_sw_full, dx_full, dy_full)

    # Standing wave + ideal perturbation
    p_combined = p_sw_full + p_ideal_full
    U_combined_full, _, _ = _gorkov_fields(p_combined, dx_full, dy_full)

    # Extract ROI regions
    ix_roi, iy_roi = _crop_indices(
        x_full,
        y_full,
        float(x_roi[0]),
        float(x_roi[-1]),
        float(y_roi[0]),
        float(y_roi[-1]),
    )
    x_roi_from_full = x_full[ix_roi]
    y_roi_from_full = y_full[iy_roi]

    U_sw_roi = U_sw_full[np.ix_(iy_roi, ix_roi)]
    U_combined_roi = U_combined_full[np.ix_(iy_roi, ix_roi)]

    # Rendering setup
    x_full_mm = x_full * 1e3
    y_full_mm = y_full * 1e3
    x_roi_mm = x_roi_from_full * 1e3
    y_roi_mm = y_roi_from_full * 1e3

    extent_full = [x_full_mm[0], x_full_mm[-1], y_full_mm[0], y_full_mm[-1]]
    extent_roi = [x_roi_mm[0], x_roi_mm[-1], y_roi_mm[0], y_roi_mm[-1]]

    traps_mm = traps_m * 1e3

    # Create comparison figure (2x2 layout like canonical c_shape_replica script)
    print("\n4. Rendering static comparison figure (2x2 layout)...")

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 12.0))
    (ax_sw_roi, ax_combined_roi), (ax_sw_full, ax_combined_full) = axes

    # Use consistent color scale across all panels
    vmin, vmax = _auto_limits(U_sw_full, U_sw_roi, U_combined_full, U_combined_roi)

    # Top-left: ROI standing wave only
    _draw_static_panel(
        ax_sw_roi,
        U_sw_roi,
        vmin,
        vmax,
        extent_roi,
        x_roi_mm,
        y_roi_mm,
        traps_mm,
        idx_A,
        idx_B,
        neigh_idx,
        "Standing Wave Only (ROI)",
    )

    # Top-right: ROI standing wave + ideal perturbation
    _draw_static_panel(
        ax_combined_roi,
        U_combined_roi,
        vmin,
        vmax,
        extent_roi,
        x_roi_mm,
        y_roi_mm,
        traps_mm,
        idx_A,
        idx_B,
        neigh_idx,
        "Standing Wave + Ideal Perturbation (ROI)",
    )

    # Bottom-left: Full domain standing wave only
    _draw_static_panel(
        ax_sw_full,
        U_sw_full,
        vmin,
        vmax,
        extent_full,
        x_full_mm,
        y_full_mm,
        traps_mm,
        idx_A,
        idx_B,
        neigh_idx,
        "Standing Wave Only (Full Domain)",
    )

    # Bottom-right: Full domain standing wave + ideal perturbation
    _draw_static_panel(
        ax_combined_full,
        U_combined_full,
        vmin,
        vmax,
        extent_full,
        x_full_mm,
        y_full_mm,
        traps_mm,
        idx_A,
        idx_B,
        neigh_idx,
        "Standing Wave + Ideal Perturbation (Full Domain)",
    )

    fig.suptitle(
        "Static Comparison: Ideal Local Pressure Perturbation\n"
        "Gor'kov Potential U_Gorkov [J]",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    comparison_png = OUT_DIR / "static_comparison_ideal_perturbation.png"
    fig.savefig(comparison_png, dpi=150, bbox_inches="tight")
    print(f"  Saved: {comparison_png.name}")
    plt.close(fig)

    # ---- Save field data and metadata ----
    print("\n5. Saving field data and metadata...")

    fields_npz = OUT_DIR / "static_fields.npz"
    np.savez(
        fields_npz,
        p_sw_full=p_sw_full,
        p_ideal_full=p_ideal_full,
        p_combined=p_combined,
        U_sw_full=U_sw_full,
        U_combined_full=U_combined_full,
        U_sw_roi=U_sw_roi,
        U_combined_roi=U_combined_roi,
        x_full=x_full,
        y_full=y_full,
        x_roi=x_roi_from_full,
        y_roi=y_roi_from_full,
        traps_m=traps_m,
        idx_A=np.array(idx_A),
        idx_B=np.array(idx_B),
    )
    print(f"  Saved: {fields_npz.name}")

    metadata = {
        "script": "scripts/dev/ideal_pressure_static_compare.py",
        "purpose": "Static 2x2 comparison of standing-wave baseline vs. minimal ideal pressure perturbation",
        "date": str(__import__("datetime").datetime.now().isoformat()),
        "layout": "2x2 panels: [ROI SW only, ROI SW+ideal] / [Full SW only, Full SW+ideal]",
        "particle_A": {
            "index": int(idx_A),
            "position_mm": [float(A_xy[0] * 1e3), float(A_xy[1] * 1e3)],
            "role": "source (slightly destabilized)",
        },
        "particle_B": {
            "index": int(idx_B),
            "position_mm": [float(B_xy[0] * 1e3), float(B_xy[1] * 1e3)],
            "role": "destination (anchor sink)",
        },
        "distance_AB_mm": float(np.linalg.norm(B_xy - A_xy) * 1e3),
        "ideal_perturbation_design": {
            "philosophy": "Minimal, highly local perturbation to avoid global field disturbance",
            "anchor_at_B": {
                "radius_um": float(ANCHOR_RADIUS_M * 1e6),
                "amplitude_Pa": float(ANCHOR_AMPLITUDE_PA),
                "sign": "negative",
                "purpose": "creates local attractive minimum (sink) by reducing field magnitude",
            },
            "destabilization_at_A": {
                "radius_um": float(SOURCE_RADIUS_M * 1e6),
                "amplitude_Pa": float(SOURCE_AMPLITUDE_PA),
                "sign": "positive",
                "purpose": "weak local destabilization for asymmetry",
            },
            "ramp_component": "disabled",
            "side_confinement": "disabled",
        },
        "field_peaks": {
            "standing_wave_Pa": float(np.max(np.abs(p_sw_full))),
            "ideal_perturbation_Pa": float(np.max(np.abs(p_ideal_full))),
            "combined_Pa": float(np.max(np.abs(p_combined))),
        },
        "gorkov_potential_ranges": {
            "standing_wave": {
                "min_J": float(np.min(U_sw_full)),
                "max_J": float(np.max(U_sw_full)),
                "mean_J": float(np.mean(U_sw_full)),
            },
            "combined": {
                "min_J": float(np.min(U_combined_full)),
                "max_J": float(np.max(U_combined_full)),
                "mean_J": float(np.mean(U_combined_full)),
            },
        },
        "outputs": {
            "comparison_png": "static_comparison_ideal_perturbation.png",
            "fields_npz": "static_fields.npz",
            "metadata_json": "static_metadata.json",
        },
    }

    metadata_json = OUT_DIR / "static_metadata.json"
    with open(metadata_json, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved: {metadata_json.name}")

    print("\n" + "=" * 70)
    print("SUCCESS")
    print("=" * 70)
    print(f"\nOutput directory: {OUT_DIR}")
    print(f"\nMain comparison figure:")
    print(f"  - {comparison_png.name}")
    print(f"\nField data:")
    print(f"  - {fields_npz.name}")
    print(f"\nMetadata:")
    print(f"  - {metadata_json.name}")
    print(f"\nDesign changes:")
    print(f"  • Anchor at B: {ANCHOR_RADIUS_M*1e6:.0f} µm radius, {ANCHOR_AMPLITUDE_PA:.0f} Pa (negative)")
    print(f"  • Destabilization at A: {SOURCE_RADIUS_M*1e6:.0f} µm radius, {SOURCE_AMPLITUDE_PA:.0f} Pa")
    print(f"  • Ramp disabled: no broad vertical plume")
    print(f"  • Side confinement disabled: effect purely local")
    print(f"\nExpected appearance:")
    print(f"  • Full domain: mostly unchanged away from A/B pair")
    print(f"  • ROI: subtle local A→B bias (not one large central hotspot)")
    print(f"  • Standing-wave peaks: red crests in consistent color scale")
    print()



if __name__ == "__main__":
    main()
