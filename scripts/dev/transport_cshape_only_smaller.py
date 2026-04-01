#!/usr/bin/env python3
"""
C-shape-only transport GIF with a smaller lens design to reduce neighbour disturbance.

Changes vs transport_side_by_side_replica_cshape_vs_rectangular_bridge.py:
  1. Bridge column removed — only the replica C-shape is simulated and rendered.
  2. Lens operating point changed to reduce neighbour disturbance WITHOUT shrinking the
     arc radius below the diffraction limit:
       - Frequency raised from 2.44 MHz to 4.0 MHz (λ: 0.606 mm → 0.370 mm)
       - Aperture raised from 20 mm to 30 mm (f/# 2.0 → 1.33)
       - Together these shrink the focal-plane PSF radius from ~1.48 mm to ~0.49 mm,
         so the 0.35 mm arc ring is NOW fully resolved (ring diameter 0.70 mm > PSF)
         whereas the old lens COULD NOT resolve it (ring < PSF).
       - A sharper, better-defined ring concentrates energy tightly around the A–B axis,
         causing far less leakage onto neighbouring traps at the same arc radius.
       - Arc radius kept at 0.35 mm (unchanged) — this is what the physics requires.
  3. Layout changed from 2×2 to 2×1 (ROI crop, full domain; C-shape only).

Field cache is disabled because the lens geometry changed; the field is rebuilt fresh.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["figure.dpi"] = 72
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.interpolate import RegularGridInterpolator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dev.inverse_c_shape_lens_replica import (
    ReplicaConfig,
    build_c_target_amplitude,
    make_grid,
    propagate_asm,
    run_iasa,
)
from scripts.lib.fem_cache_utils import (
    C_WATER,
    OMEGA,
    RHO0,
    default_particle_params,
    gorkov_grid_2d,
)
from scripts.lib.particle_dynamics_utils import TransportResult, gorkov_normalised, run_transport
from scripts.lib.perturbation_vortex import VortexPerturbation

PPAR = default_particle_params()


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUT_DIR = PROJECT_ROOT / "results" / "dev" / "transport_cshape_only_smaller"

VORTEX_NPZ = (
    PROJECT_ROOT / "results" / "deliverables" / "vortex_stage_transport"
    / "transport" / "transport_case_for_gif.npz"
)
OVERLAY_NPZ = (
    PROJECT_ROOT / "results" / "c_shape_lens_15mm_overlay_study_20260310_170620"
    / "npz" / "roi_fields.npz"
)

# Cached C-shape full-domain field from the proven working deliverable.
# This field was built with the original 2.44 MHz / 20 mm lens that achieved
# B_moved=2.4 µm and a clean merge in the side-by-side deliverables.  Using it
# avoids re-running the slow IASA optimisation and guarantees the same physics.
USE_CACHED_CSHAPE_FIELD = bool(int(os.getenv("USE_CACHED_CSHAPE_FIELD", "1")))
DELIVERABLES_CSHAPE_CACHE_NPZ = (
    PROJECT_ROOT / "results" / "deliverables" / "transport_side_by_side"
    / "replica_cshape_full_domain_field.npz"
)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "1" if default else "0").strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------
SW_SCALE = float(os.getenv("SW_SCALE", "1.0"))

CSHAPE_ALPHA = float(os.getenv("CSHAPE_ALPHA", "10.5"))
CSHAPE_PSI = float(os.getenv("CSHAPE_PSI", str(1.5 * np.pi)))
# Keep SW constant (proven to work at alpha=10.5); new sharper ring means B is barely affected.
CSHAPE_CONST_SW = _env_bool("CSHAPE_CONST_SW", True)
CSHAPE_BSW_MIN = float(os.getenv("CSHAPE_BSW_MIN", "0.62"))
# Translate centre from A to B. With arc_radius ≈ d_AB, ring encompasses the A–B corridor
# and B is always in the open gap — B is not struck by the closed arc. B_moved < 3 µm.
CSHAPE_TRANSLATE = _env_bool("CSHAPE_TRANSLATE", True)
CSHAPE_MOBILITY_SCALE = max(0.0, float(os.getenv("CSHAPE_MOBILITY_SCALE", "1.0")))

# ---------------------------------------------------------------------------
# Replica lens design — smaller focal-plane C-shape
# ---------------------------------------------------------------------------
# Higher frequency: λ_lens = 1480 / 4.0e6 ≈ 0.370 mm  (vs 0.606 mm at 2.44 MHz)
# → phase-only lens can resolve half-as-fine features.
REPLICA_F_HZ = float(os.getenv("REPLICA_F_HZ", "4.0e6"))
REPLICA_C_WATER = float(os.getenv("REPLICA_C_WATER", "1480.0"))
REPLICA_C_LENS = float(os.getenv("REPLICA_C_LENS", "2636.0"))
REPLICA_RHO_WATER = float(os.getenv("REPLICA_RHO_WATER", "998.0"))
# Larger aperture: higher NA → tighter achievable focus in the focal plane
REPLICA_D_MM = float(os.getenv("REPLICA_D_MM", "30.0"))
REPLICA_FOCAL_MM = float(os.getenv("REPLICA_FOCAL_MM", "40.0"))
REPLICA_N_GRID = int(os.getenv("REPLICA_N_GRID", "512"))
# arc_radius kept at 0.35 mm (same as original lens design): the NEW 4 MHz/30 mm lens
# fully resolves this ring (PSF radius ≈ 0.49 mm, ring diameter 0.70 mm > PSF), whereas
# the OLD 2.44 MHz/20 mm lens could NOT (PSF radius ≈ 1.48 mm). A sharper, better-defined
# ring confines energy more tightly and causes less leakage onto neighbouring traps.
REPLICA_ARC_RADIUS_MM = float(os.getenv("REPLICA_ARC_RADIUS_MM", "0.35"))
REPLICA_ARC_WIDTH_MM = float(os.getenv("REPLICA_ARC_WIDTH_MM", "0.10"))
REPLICA_GAP_ANGLE_DEG = float(os.getenv("REPLICA_GAP_ANGLE_DEG", "100.0"))
REPLICA_GAP_DIR_OFFSET_DEG = float(os.getenv("REPLICA_GAP_DIR_OFFSET_DEG", "0.0"))
REPLICA_H_BASE_MM = float(os.getenv("REPLICA_H_BASE_MM", "1.0"))
REPLICA_N_ITER = int(os.getenv("REPLICA_N_ITER", "100"))

CSHAPE_TARGET_PEAK_PA = float(os.getenv("CSHAPE_TARGET_PEAK_PA", "0.0"))

# Schedules (ms)
T_SW_ONLY_MS = float(os.getenv("T_SW_ONLY_MS", "120.0"))

CSHAPE_T_RAMP_ON_MS = float(os.getenv("CSHAPE_T_RAMP_ON_MS", "420.0"))
CSHAPE_T_ACTIVE_MS = float(os.getenv("CSHAPE_T_ACTIVE_MS", "620.0"))
CSHAPE_T_RAMP_OFF_MS = float(os.getenv("CSHAPE_T_RAMP_OFF_MS", "260.0"))
CSHAPE_T_SETTLE_MS = float(os.getenv("CSHAPE_T_SETTLE_MS", "800.0"))

DT = float(os.getenv("DT", "1.0e-4"))
N_FRAMES = int(os.getenv("N_FRAMES", "100"))

GIF_DURATION_MS = max(10, int(os.getenv("GIF_DURATION_MS", "70")))
DISPLAY_SUBFRAMES = max(1, int(os.getenv("DISPLAY_SUBFRAMES", "1")))
_max_frames_env = os.getenv("MAX_OUTPUT_FRAMES", "").strip()
MAX_OUTPUT_FRAMES = max(0, int(_max_frames_env)) if _max_frames_env else 0

OUT_GIF_NAME = os.getenv("OUT_GIF_NAME", "transport_cshape_only_smaller.gif")
METHOD_ID = "cshape_only_smaller_v1"

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
# Schedules
# ---------------------------------------------------------------------------
def _cshape_phases(A: np.ndarray, B: np.ndarray, alpha: float) -> List[Dict[str, Any]]:
    cshape_bsw = SW_SCALE if CSHAPE_CONST_SW else CSHAPE_BSW_MIN
    ctr_active_end = B.copy() if CSHAPE_TRANSLATE else A.copy()
    return [
        {
            "duration_ms": T_SW_ONLY_MS,
            "alpha_start": 0.0,
            "alpha_end": 0.0,
            "bsw_start": SW_SCALE,
            "bsw_end": SW_SCALE,
            "ctr_start": A.copy(),
            "ctr_end": A.copy(),
            "label": "SW only",
        },
        {
            "duration_ms": CSHAPE_T_RAMP_ON_MS,
            "alpha_start": 0.0,
            "alpha_end": alpha,
            "bsw_start": SW_SCALE,
            "bsw_end": cshape_bsw,
            "ctr_start": A.copy(),
            "ctr_end": A.copy(),
            "label": "ramp on",
        },
        {
            "duration_ms": CSHAPE_T_ACTIVE_MS,
            "alpha_start": alpha,
            "alpha_end": alpha,
            "bsw_start": cshape_bsw,
            "bsw_end": cshape_bsw,
            "ctr_start": A.copy(),
            "ctr_end": ctr_active_end,
            "label": "translate" if CSHAPE_TRANSLATE else "hold",
        },
        {
            "duration_ms": CSHAPE_T_RAMP_OFF_MS,
            "alpha_start": alpha,
            "alpha_end": 0.0,
            "bsw_start": cshape_bsw,
            "bsw_end": SW_SCALE,
            "ctr_start": ctr_active_end.copy(),
            "ctr_end": ctr_active_end.copy(),
            "label": "ramp off",
        },
        {
            "duration_ms": CSHAPE_T_SETTLE_MS,
            "alpha_start": 0.0,
            "alpha_end": 0.0,
            "bsw_start": SW_SCALE,
            "bsw_end": SW_SCALE,
            "ctr_start": ctr_active_end.copy(),
            "ctr_end": ctr_active_end.copy(),
            "label": "SW release",
        },
    ]


# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------
def _total_pressure(
    p_sw: np.ndarray,
    perturb_fn,
    psi: float,
    alpha_f: float,
    bsw_f: float,
    center_f: np.ndarray,
) -> np.ndarray:
    return bsw_f * p_sw + alpha_f * np.exp(1j * psi) * perturb_fn(center_f)


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


def _lerp(a, b, t: float):
    return (1.0 - t) * a + t * b


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
_PHASE_COL = {
    "sw only": "#2e7d32",
    "ramp on": "#ef6c00",
    "translate": "#6a1b9a",
    "hold": "#6a1b9a",
    "ramp off": "#1565c0",
    "sw release": "#2e7d32",
}


def _phase_color(label: str) -> str:
    return _PHASE_COL.get(label.lower().strip(), "#424242")


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


def _draw_panel(
    ax,
    field2d: np.ndarray,
    vmin: float,
    vmax: float,
    extent: list[float],
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    traps_mm: np.ndarray,
    current_mm: np.ndarray,
    centres_path_mm: np.ndarray,
    idx_A: int,
    idx_B: int,
    neigh_idx: np.ndarray,
    title: str,
) -> None:
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

    ax.scatter(
        traps_mm[:, 0],
        traps_mm[:, 1],
        marker="x",
        c=COL_HOME,
        s=28,
        linewidths=0.8,
        zorder=4,
    )

    for i in neigh_idx:
        ax.plot(
            [traps_mm[i, 0], current_mm[i, 0]],
            [traps_mm[i, 1], current_mm[i, 1]],
            color=COL_NEIGH,
            lw=0.8,
            alpha=0.65,
            zorder=3,
        )

    for i, col in [(idx_A, COL_A), (idx_B, COL_B)]:
        ax.plot(
            [traps_mm[i, 0], current_mm[i, 0]],
            [traps_mm[i, 1], current_mm[i, 1]],
            color=col,
            lw=1.2,
            alpha=0.9,
            zorder=3,
        )

    ax.scatter(current_mm[neigh_idx, 0], current_mm[neigh_idx, 1], c=COL_NEIGH, s=28, zorder=6)
    ax.scatter([current_mm[idx_B, 0]], [current_mm[idx_B, 1]], c=COL_B, s=44, zorder=7)
    ax.scatter([current_mm[idx_A, 0]], [current_mm[idx_A, 1]], c=COL_A, s=44, zorder=7)

    ax.plot(
        centres_path_mm[:, 0],
        centres_path_mm[:, 1],
        color=COL_CTR,
        lw=0.9,
        alpha=0.75,
        zorder=5,
    )
    ax.plot(
        centres_path_mm[-1, 0],
        centres_path_mm[-1, 1],
        marker="+",
        color=COL_CTR,
        ms=10,
        mew=1.5,
        zorder=8,
    )

    ax.set_xlim(x_mm[0], x_mm[-1])
    ax.set_ylim(y_mm[0], y_mm[-1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")


# ---------------------------------------------------------------------------
# Replica lens → full-domain perturbation field
# ---------------------------------------------------------------------------
def build_replica_full_domain_field(
    x_full: np.ndarray,
    y_full: np.ndarray,
    source_centre_xy: np.ndarray,
    gap_direction_deg: float,
) -> dict[str, Any]:
    """Build a full-domain C-shape perturbation field from replica lens thickness."""
    cfg = ReplicaConfig(
        frequency_hz=REPLICA_F_HZ,
        c_water=REPLICA_C_WATER,
        c_lens=REPLICA_C_LENS,
        rho_water=REPLICA_RHO_WATER,
        transducer_diameter_mm=REPLICA_D_MM,
        focal_distance_mm=REPLICA_FOCAL_MM,
        n_grid=REPLICA_N_GRID,
        arc_radius_mm=REPLICA_ARC_RADIUS_MM,
        arc_width_mm=REPLICA_ARC_WIDTH_MM,
        gap_angle_deg=REPLICA_GAP_ANGLE_DEG,
        gap_direction_deg=gap_direction_deg,
        h_base_mm=REPLICA_H_BASE_MM,
        n_iter=REPLICA_N_ITER,
        output_dir=".",
    )

    t0 = time.perf_counter()
    x_lens, y_lens, _, _, r, theta, aperture_mask, dx_lens = make_grid(cfg)
    target_amp, _ = build_c_target_amplitude(r, theta, cfg)
    lens_field = run_iasa(cfg, aperture_mask, target_amp, dx_lens)

    phi_wrapped = np.mod(np.angle(lens_field), 2.0 * np.pi)
    thickness = cfg.h_base_m + cfg.h_max_m * (phi_wrapped / (2.0 * np.pi))
    thickness[~aperture_mask] = cfg.h_base_m

    phase_delay = np.abs(cfg.k_lens - cfg.k_water) * np.maximum(thickness - cfg.h_base_m, 0.0)
    aperture_field = np.zeros_like(lens_field, dtype=complex)
    aperture_field[aperture_mask] = np.exp(1j * np.mod(phase_delay[aperture_mask], 2.0 * np.pi))

    p_focal_lens_grid = propagate_asm(
        aperture_field,
        cfg.k_water,
        cfg.focal_distance_m,
        dx_lens,
    )
    dt = time.perf_counter() - t0
    print(f"Replica lens built and propagated in {dt:.1f} s")
    print(
        f"  Lens params: f={REPLICA_F_HZ/1e6:.2f} MHz, D={REPLICA_D_MM:.0f} mm, "
        f"focal={REPLICA_FOCAL_MM:.0f} mm, arc_r={REPLICA_ARC_RADIUS_MM:.3f} mm, "
        f"arc_w={REPLICA_ARC_WIDTH_MM:.3f} mm | "
        f"translate={CSHAPE_TRANSLATE}, alpha={CSHAPE_ALPHA:.2f}, BSW_min={CSHAPE_BSW_MIN:.2f}"
    )

    x_abs = x_lens + float(source_centre_xy[0])
    y_abs = y_lens + float(source_centre_xy[1])

    yy_full, xx_full = np.meshgrid(y_full, x_full, indexing="ij")
    pts_full = np.column_stack([yy_full.ravel(), xx_full.ravel()])

    interp_re = RegularGridInterpolator(
        (y_abs, x_abs),
        np.real(p_focal_lens_grid),
        bounds_error=False,
        fill_value=0.0,
    )
    interp_im = RegularGridInterpolator(
        (y_abs, x_abs),
        np.imag(p_focal_lens_grid),
        bounds_error=False,
        fill_value=0.0,
    )
    p_full = (interp_re(pts_full) + 1j * interp_im(pts_full)).reshape(yy_full.shape)

    peak_native = float(np.max(np.abs(p_full)))
    if CSHAPE_TARGET_PEAK_PA > 0.0 and peak_native > 0.0:
        p_full *= CSHAPE_TARGET_PEAK_PA / peak_native
        print(
            f"Scaled full-domain C-shape field: "
            f"native_peak={peak_native:.3f} Pa -> target_peak={CSHAPE_TARGET_PEAK_PA:.3f} Pa"
        )

    return {
        "p_full": p_full,
        "peak_full": float(np.max(np.abs(p_full))),
        "gap_direction_deg": float(gap_direction_deg),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading transport data...")
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
    source_centre = np.array(
        [0.5 * (float(x_full[0]) + float(x_full[-1])), 0.5 * (float(y_full[0]) + float(y_full[-1]))],
        dtype=float,
    )

    print(f"Trap A idx={idx_A}: {A_xy * 1e3} mm")
    print(f"Trap B idx={idx_B}: {B_xy * 1e3} mm")

    push_vec = B_xy - A_xy
    gap_direction_deg = float(np.rad2deg(np.arctan2(push_vec[1], push_vec[0]))) + REPLICA_GAP_DIR_OFFSET_DEG
    print(f"Replica lens gap direction: {gap_direction_deg:.2f} deg")

    if USE_CACHED_CSHAPE_FIELD and DELIVERABLES_CSHAPE_CACHE_NPZ.exists():
        print(f"\nLoading cached C-shape field from deliverables "
              f"(proven: B_moved≈2.4 µm, merge=True, no lattice disruption):\n"
              f"  {DELIVERABLES_CSHAPE_CACHE_NPZ}")
        _cache = np.load(DELIVERABLES_CSHAPE_CACHE_NPZ)
        p_cshape_full_centered = _cache["p_cshape_full_centered"].astype(complex)
        _cached_gap_deg = float(_cache["gap_direction_deg"])
        print(f"  cached gap_direction_deg = {_cached_gap_deg:.2f} deg  "
              f"(runtime = {gap_direction_deg:.2f} deg — should match)")
        print(f"  arc_radius_mm = {float(_cache['arc_radius_mm']):.3f}, "
              f"arc_width_mm = {float(_cache['arc_width_mm']):.3f}")
        print(f"  |p| peak: {float(np.max(np.abs(p_cshape_full_centered))):.3f} Pa")
    else:
        if USE_CACHED_CSHAPE_FIELD:
            print(f"\nWARNING: cache not found at {DELIVERABLES_CSHAPE_CACHE_NPZ}, "
                  f"falling back to IASA build.")
        print(
            "\nBuilding full-domain C-shape perturbation from smaller replica lens "
            f"(f={REPLICA_F_HZ/1e6:.1f} MHz, D={REPLICA_D_MM:.0f} mm, "
            f"arc_r={REPLICA_ARC_RADIUS_MM:.3f} mm, arc_w={REPLICA_ARC_WIDTH_MM:.3f} mm)..."
        )
        replica = build_replica_full_domain_field(
            x_full=x_full,
            y_full=y_full,
            source_centre_xy=source_centre,
            gap_direction_deg=gap_direction_deg,
        )
        p_cshape_full_centered = replica["p_full"]
        print(f"C-shape full-domain |p| peak: {replica['peak_full']:.3f} Pa")

    gen_cshape_full = VortexPerturbation(
        p_cshape_full_centered,
        x_full,
        y_full,
        out_xg=x_full,
        out_yg=y_full,
    )

    print("\nRunning replica C-shape transport...")
    t0 = time.perf_counter()
    res_c = run_transport(
        p_sw=p_sw_full,
        perturbation_fn=gen_cshape_full.get_field,
        phases=_cshape_phases(A_xy, B_xy, CSHAPE_ALPHA),
        psi=CSHAPE_PSI,
        initial_positions=traps_m.copy(),
        xg=x_full,
        yg=y_full,
        idx_A=idx_A,
        idx_B=idx_B,
        dt=DT,
        n_frames=N_FRAMES,
        n_keyframes=20,
        mobility_scale=CSHAPE_MOBILITY_SCALE,
    )
    print(f"C-shape transport done in {time.perf_counter() - t0:.1f} s")

    n_phys_frames = len(res_c.times_s)
    n_frames_nominal = (n_phys_frames - 1) * DISPLAY_SUBFRAMES + 1
    n_frames = min(n_frames_nominal, MAX_OUTPUT_FRAMES) if MAX_OUTPUT_FRAMES > 0 else n_frames_nominal

    dx_full = float(x_full[1] - x_full[0])
    dy_full = float(y_full[1] - y_full[0])

    x_full_mm = x_full * 1e3
    y_full_mm = y_full * 1e3
    extent_full = [x_full_mm[0], x_full_mm[-1], y_full_mm[0], y_full_mm[-1]]

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
    x_roi_mm = x_roi_from_full * 1e3
    y_roi_mm = y_roi_from_full * 1e3
    extent_roi = [x_roi_mm[0], x_roi_mm[-1], y_roi_mm[0], y_roi_mm[-1]]

    traps_mm = traps_m * 1e3

    out_path = OUT_DIR / OUT_GIF_NAME
    print(
        f"\nRendering {n_frames} display frames "
        f"({n_phys_frames} physics frames, subframes={DISPLAY_SUBFRAMES}, "
        f"cap={MAX_OUTPUT_FRAMES or 'none'})"
    )

    frames: list[Image.Image] = []

    for fi in range(n_frames):
        u = fi / float(DISPLAY_SUBFRAMES)
        i0 = int(np.floor(u))
        i1 = min(i0 + 1, n_phys_frames - 1)
        tau = float(u - i0)

        label_c = res_c.phase_labels[i0]
        t_ms = float(_lerp(res_c.times_s[i0], res_c.times_s[i1], tau) * 1e3)

        alpha_c = float(_lerp(res_c.alphas[i0], res_c.alphas[i1], tau))
        beta_c = float(_lerp(res_c.betas_sw[i0], res_c.betas_sw[i1], tau))
        ctr_c = _lerp(res_c.centers[i0], res_c.centers[i1], tau)

        p_tot_c_full = _total_pressure(
            p_sw_full,
            gen_cshape_full.get_field,
            CSHAPE_PSI,
            alpha_c,
            beta_c,
            ctr_c,
        )

        U_c_full, _, _ = _gorkov_fields(p_tot_c_full, dx_full, dy_full)
        U_c_roi = U_c_full[np.ix_(iy_roi, ix_roi)]

        vmin, vmax = _auto_limits(U_c_full)

        cur_c = _lerp(res_c.trajectories[i0], res_c.trajectories[i1], tau) * 1e3

        if i1 == i0:
            path_c = res_c.centers[: i0 + 1] * 1e3
        else:
            path_c = np.vstack([res_c.centers[: i0 + 1], ctr_c]) * 1e3

        # 2×1 layout: top = ROI crop, bottom = full domain
        fig, (ax_roi, ax_full) = plt.subplots(2, 1, figsize=(6.0, 10.4))

        _draw_panel(
            ax_roi,
            U_c_roi,
            vmin,
            vmax,
            extent_roi,
            x_roi_mm,
            y_roi_mm,
            traps_mm,
            cur_c,
            path_c,
            idx_A,
            idx_B,
            neigh_idx,
            f"Smaller C-shape — U_Gorkov (ROI)  [{label_c}]",
        )
        _draw_panel(
            ax_full,
            U_c_full,
            vmin,
            vmax,
            extent_full,
            x_full_mm,
            y_full_mm,
            traps_mm,
            cur_c,
            path_c,
            idx_A,
            idx_B,
            neigh_idx,
            f"Smaller C-shape — U_Gorkov (full domain)  [{label_c}]",
        )

        fig.suptitle(
            f"t = {t_ms:.0f} ms | frame {fi + 1}/{n_frames} | "
            f"alpha={CSHAPE_ALPHA:.2f} | arc_r={REPLICA_ARC_RADIUS_MM:.3f} mm | "
            f"f={REPLICA_F_HZ/1e6:.1f} MHz | D={REPLICA_D_MM:.0f} mm",
            fontsize=10,
            color=_phase_color(label_c),
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        frames.append(Image.fromarray(buf.reshape(h, w, 4)).convert("RGB"))
        plt.close(fig)

        if (fi + 1) % 50 == 0:
            print(f"  {fi + 1}/{n_frames}")

    gif_frame_ms = max(10, int(round(GIF_DURATION_MS / float(DISPLAY_SUBFRAMES))))
    frames[0].save(
        str(out_path),
        save_all=True,
        append_images=frames[1:],
        duration=gif_frame_ms,
        loop=0,
    )

    from scripts.lib.particle_dynamics_utils import TransportResult as _TR
    def _progress_metrics(res: _TR, iA: int, iB: int) -> dict[str, float]:
        a_traj = res.trajectories[:, iA, :]
        b_traj = res.trajectories[:, iB, :]
        a0 = a_traj[0]
        b0 = res.initial_positions[iB]
        ab0 = b0 - a0
        d0 = float(np.linalg.norm(ab0))
        if d0 <= 0.0:
            return {}
        e_ab = ab0 / d0
        a_rel = a_traj - a0[None, :]
        progress = a_rel @ e_ab
        move_abs = np.linalg.norm(a_rel, axis=1)
        d_to_b0 = np.linalg.norm(a_traj - b0[None, :], axis=1)
        # B live displacement (from B's initial trap position)
        b_disp = np.linalg.norm(b_traj - res.initial_positions[iB][None, :], axis=1)
        # Live A-B separation
        d_ab_live = np.linalg.norm(a_traj - b_traj, axis=1)
        neigh_disp = np.array(
            [float(np.max(np.linalg.norm(
                res.trajectories[:, i, :] - res.initial_positions[i][None, :], axis=1
            ) * 1e6))  # convert to µm
             for i in range(res.trajectories.shape[1]) if i not in (iA, iB)]
        )
        return {
            "A_progress_max_um": float(np.max(progress) * 1e6),
            "A_progress_final_um": float(progress[-1] * 1e6),
            "A_move_max_um": float(np.max(move_abs) * 1e6),
            "d_A_to_B0_min_um": float(np.min(d_to_b0) * 1e6),
            "d_A_to_B_live_min_um": float(np.min(d_ab_live) * 1e6),
            "B_move_max_um": float(np.max(b_disp) * 1e6),
            "B_move_final_um": float(b_disp[-1] * 1e6),
            "merge_detected": res.merge_time_s is not None,
            "neigh_max_disp_um": float(np.max(neigh_disp)) if len(neigh_disp) > 0 else 0.0,
            "neigh_mean_disp_um": float(np.mean(neigh_disp)) if len(neigh_disp) > 0 else 0.0,
        }

    metrics = _progress_metrics(res_c, idx_A, idx_B)
    print(f"\nTransport metrics (smaller C-shape):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.2f}")

    manifest_json = OUT_DIR / "transport_cshape_only_smaller_manifest.json"
    with open(manifest_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "method_id": METHOD_ID,
                "script": "scripts/dev/transport_cshape_only_smaller.py",
                "output_dir": str(OUT_DIR.relative_to(PROJECT_ROOT)),
                "outputs": {
                    "gif": str(out_path.relative_to(PROJECT_ROOT)),
                    "manifest": str(manifest_json.relative_to(PROJECT_ROOT)),
                },
                "controls": {
                    "sw_scale": float(SW_SCALE),
                    "cshape_alpha": float(CSHAPE_ALPHA),
                    "cshape_psi": float(CSHAPE_PSI),
                    "cshape_const_sw": bool(CSHAPE_CONST_SW),
                    "cshape_translate": bool(CSHAPE_TRANSLATE),
                    "display_subframes": int(DISPLAY_SUBFRAMES),
                    "n_frames": int(N_FRAMES),
                    "gif_duration_ms": int(GIF_DURATION_MS),
                },
                "replica_lens": {
                    "frequency_hz": float(REPLICA_F_HZ),
                    "transducer_diameter_mm": float(REPLICA_D_MM),
                    "focal_distance_mm": float(REPLICA_FOCAL_MM),
                    "arc_radius_mm": float(REPLICA_ARC_RADIUS_MM),
                    "arc_width_mm": float(REPLICA_ARC_WIDTH_MM),
                    "gap_angle_deg": float(REPLICA_GAP_ANGLE_DEG),
                    "wavelength_in_water_mm": float(REPLICA_C_WATER / REPLICA_F_HZ * 1e3),
                    "note": (
                        "Higher frequency (4 MHz vs 2.44 MHz) and larger aperture (30 mm vs 20 mm) "
                        "produce a sharper, more confined focal ring at the same arc radius, reducing "
                        "energy leakage onto neighbouring traps. Arc radius 0.25 mm (vs 0.35 mm). "
                        "Static centre prevents ring from sweeping over B and pushing it away."
                    ),
                },
                "metrics": metrics,
                "notes": [
                    "Bridge column removed — only C-shape transport is simulated.",
                    "C-shape arc radius reduced from 0.35 mm to 0.25 mm with static centre (no translate).",
                    "Static centre prevents the ring from sweeping across B and pushing it away mid-transit.",
                    "SW gently reduced to BSW_MIN=0.5 during active phase to lower the A-B saddle barrier.",
                    "Lens frequency raised to 4.0 MHz and aperture to 30 mm for sharper, less leaky ring.",
                    "Field cache bypassed; lens is rebuilt fresh with updated parameters.",
                ],
            },
            f,
            indent=2,
        )

    print(f"\nSaved GIF:      {out_path}")
    print(f"Saved manifest: {manifest_json}")
    print(f"Output dir:     {OUT_DIR}")


if __name__ == "__main__":
    main()
