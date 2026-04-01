#!/usr/bin/env python3
"""
Side-by-side transport GIF: replica C-shape vs rectangular bridge method.

Template source:
  scripts/deliverables/transport_side_by_side_replica_cshape.py

Layout and method are intentionally matched to the replica script:
- 2x2 panels
  - left column: replica C-shape (ROI crop, full domain)
  - right column: rectangular bridge (ROI crop, full domain)
- transport integration through scripts/lib/particle_dynamics_utils.run_transport
- full-domain Gor'kov fields with ROI as strict crop

Constraint from user request:
- Do not modify the canonical replica script
- Bridge alpha is auto-scaled until A shows clear movement toward B,
  regardless of neighbour disturbance.
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
from scripts.dev.rectangular_corridor_bridge import build_corridor_bridge_field
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
OUT_DIR = PROJECT_ROOT / "results" / "dev" / "transport_replica_cshape_vs_rectangular_bridge"

VORTEX_NPZ = (
    PROJECT_ROOT / "results" / "deliverables" / "vortex_stage_transport"
    / "transport" / "transport_case_for_gif.npz"
)
OVERLAY_NPZ = (
    PROJECT_ROOT / "results" / "c_shape_lens_15mm_overlay_study_20260310_170620"
    / "npz" / "roi_fields.npz"
)
REPLICA_FIELD_CACHE_NPZ = (
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

# Left column: replica C-shape
CSHAPE_ALPHA = float(os.getenv("CSHAPE_ALPHA", "10.5"))
CSHAPE_PSI = float(os.getenv("CSHAPE_PSI", str(1.5 * np.pi)))
CSHAPE_CONST_SW = _env_bool("CSHAPE_CONST_SW", True)
CSHAPE_BSW_MIN = float(os.getenv("CSHAPE_BSW_MIN", "0.62"))
CSHAPE_TRANSLATE = _env_bool("CSHAPE_TRANSLATE", True)
CSHAPE_MOBILITY_SCALE = max(0.0, float(os.getenv("CSHAPE_MOBILITY_SCALE", "1.0")))

# Right column: rectangular corridor bridge
BRIDGE_PSI = float(os.getenv("BRIDGE_PSI", "0.0"))
AUTO_TUNE_BRIDGE_ALPHA = _env_bool("AUTO_TUNE_BRIDGE_ALPHA", True)
BRIDGE_ALPHA_INIT = float(os.getenv("BRIDGE_ALPHA_INIT", "0.10"))
BRIDGE_ALPHA_MULT = float(os.getenv("BRIDGE_ALPHA_MULT", "1.8"))
BRIDGE_ALPHA_STEPS = max(1, int(os.getenv("BRIDGE_ALPHA_STEPS", "16")))
BRIDGE_ALPHA_MAX = float(os.getenv("BRIDGE_ALPHA_MAX", "500.0"))
BRIDGE_BSW_PRE_CAPTURE = float(np.clip(float(os.getenv("BRIDGE_BSW_PRE_CAPTURE", "0.55")), 0.0, SW_SCALE))

# Stop rule: require visible A progress toward B (not neighbour-gated)
A_TO_B_PROGRESS_THRESHOLD_UM = float(os.getenv("A_TO_B_PROGRESS_THRESHOLD_UM", "30.0"))
A_TO_B_CAPTURE_RADIUS_UM = float(os.getenv("A_TO_B_CAPTURE_RADIUS_UM", "50.0"))
B_TRAP_MAX_DRIFT_UM = float(os.getenv("B_TRAP_MAX_DRIFT_UM", "30.0"))

# Bridge geometry and component amplitudes (matches deterministic corridor script defaults)
CORRIDOR_WIDTH_M = float(os.getenv("CORRIDOR_WIDTH_M", "3.0e-4"))
CORRIDOR_PAD_A_M = float(os.getenv("CORRIDOR_PAD_A_M", "8.0e-5"))
CORRIDOR_PAD_B_M = float(os.getenv("CORRIDOR_PAD_B_M", "1.0e-4"))
CORRIDOR_EDGE_S_M = float(os.getenv("CORRIDOR_EDGE_S_M", "3.5e-5"))
CORRIDOR_EDGE_N_M = float(os.getenv("CORRIDOR_EDGE_N_M", "3.0e-5"))

SOURCE_HOTSPOT_PA = float(
    os.getenv(
        "SOURCE_HOTSPOT_PA",
        os.getenv("SOURCE_WEAKENING_PA", os.getenv("TAPER_HIGH_AT_A_PA", "120.0")),
    )
)
DESTINATION_POCKET_PA = float(
    os.getenv("DESTINATION_POCKET_PA", os.getenv("TAPER_LOW_AT_B_PA", "0.0"))
)
DESTINATION_REST_TRAP_PA = float(os.getenv("DESTINATION_REST_TRAP_PA", "0.0"))
SOURCE_SIGMA_M = float(os.getenv("SOURCE_SIGMA_M", "1.0e-4"))
DESTINATION_SIGMA_M = float(os.getenv("DESTINATION_SIGMA_M", "1.0e-4"))
DESTINATION_REST_SIGMA_M = float(os.getenv("DESTINATION_REST_SIGMA_M", "8.0e-5"))
SOURCE_ABOVE_OFFSET_M = float(os.getenv("SOURCE_ABOVE_OFFSET_M", "9.0e-5"))

CORRIDOR_START_PA = float(os.getenv("CORRIDOR_START_PA", "120.0"))
CORRIDOR_END_PA = float(os.getenv("CORRIDOR_END_PA", "20.0"))
CORRIDOR_DECAY_POWER = float(os.getenv("CORRIDOR_DECAY_POWER", "1.25"))
CORRIDOR_TRANSVERSE_SIGMA_RATIO = float(os.getenv("CORRIDOR_TRANSVERSE_SIGMA_RATIO", "0.32"))
B_QUIET_RADIUS_M = float(os.getenv("B_QUIET_RADIUS_M", "6.0e-5"))

# Optional symmetric-bridge variant (kept opt-in so baseline behaviour is unchanged)
USE_SYMMETRIC_BRIDGE = _env_bool("USE_SYMMETRIC_BRIDGE", False)
# Extra restoring Gaussian pocket baked into the SW baseline at B (symmetric-bridge mode only)
SYMM_EXTRA_B_POCKET_PA = float(os.getenv("SYMM_EXTRA_B_POCKET_PA", "20.0"))
SYMM_EXTRA_B_POCKET_SIGMA_M = float(os.getenv("SYMM_EXTRA_B_POCKET_SIGMA_M", "1.25e-4"))

# Replica lens design (same model as inverse_c_shape_lens_replica.py)
REPLICA_F_HZ = float(os.getenv("REPLICA_F_HZ", "2.44e6"))
REPLICA_C_WATER = float(os.getenv("REPLICA_C_WATER", "1480.0"))
REPLICA_C_LENS = float(os.getenv("REPLICA_C_LENS", "2636.0"))
REPLICA_RHO_WATER = float(os.getenv("REPLICA_RHO_WATER", "998.0"))
REPLICA_D_MM = float(os.getenv("REPLICA_D_MM", "20.0"))
REPLICA_FOCAL_MM = float(os.getenv("REPLICA_FOCAL_MM", "40.0"))
REPLICA_N_GRID = int(os.getenv("REPLICA_N_GRID", "512"))
REPLICA_ARC_RADIUS_MM = float(os.getenv("REPLICA_ARC_RADIUS_MM", "0.35"))
REPLICA_ARC_WIDTH_MM = float(os.getenv("REPLICA_ARC_WIDTH_MM", "0.10"))
REPLICA_GAP_ANGLE_DEG = float(os.getenv("REPLICA_GAP_ANGLE_DEG", "100.0"))
REPLICA_GAP_DIR_OFFSET_DEG = float(os.getenv("REPLICA_GAP_DIR_OFFSET_DEG", "0.0"))
REPLICA_H_BASE_MM = float(os.getenv("REPLICA_H_BASE_MM", "1.0"))
REPLICA_N_ITER = int(os.getenv("REPLICA_N_ITER", "100"))

USE_REPLICA_FIELD_CACHE = _env_bool("USE_REPLICA_FIELD_CACHE", True)
CSHAPE_TARGET_PEAK_PA = float(os.getenv("CSHAPE_TARGET_PEAK_PA", "0.0"))

# Schedules (ms)
T_SW_ONLY_MS = float(os.getenv("T_SW_ONLY_MS", "120.0"))

CSHAPE_T_RAMP_ON_MS = float(os.getenv("CSHAPE_T_RAMP_ON_MS", "420.0"))
CSHAPE_T_ACTIVE_MS = float(os.getenv("CSHAPE_T_ACTIVE_MS", "620.0"))
CSHAPE_T_RAMP_OFF_MS = float(os.getenv("CSHAPE_T_RAMP_OFF_MS", "260.0"))
CSHAPE_T_SETTLE_MS = float(os.getenv("CSHAPE_T_SETTLE_MS", "800.0"))

BRIDGE_T_PRE_CAPTURE_RAMP_MS = float(os.getenv("BRIDGE_T_PRE_CAPTURE_RAMP_MS", "1200.0"))
BRIDGE_T_PRE_CAPTURE_HOLD_MS = float(os.getenv("BRIDGE_T_PRE_CAPTURE_HOLD_MS", "400.0"))
BRIDGE_T_POST_CAPTURE_RAMP_MS = float(os.getenv("BRIDGE_T_POST_CAPTURE_RAMP_MS", "260.0"))
BRIDGE_T_POST_CAPTURE_SETTLE_MS = float(os.getenv("BRIDGE_T_POST_CAPTURE_SETTLE_MS", "800.0"))
BRIDGE_POST_FRAMES = max(40, int(os.getenv("BRIDGE_POST_FRAMES", "60")))

DT = float(os.getenv("DT", "1.0e-4"))
N_FRAMES = int(os.getenv("N_FRAMES", "100"))

GIF_DURATION_MS = max(10, int(os.getenv("GIF_DURATION_MS", "70")))
DISPLAY_SUBFRAMES = max(1, int(os.getenv("DISPLAY_SUBFRAMES", "1")))
_max_frames_env = os.getenv("MAX_OUTPUT_FRAMES", "").strip()
MAX_OUTPUT_FRAMES = max(0, int(_max_frames_env)) if _max_frames_env else 0

OUT_GIF_NAME = os.getenv(
    "OUT_GIF_NAME",
    "transport_replica_cshape_vs_rectangular_bridge.gif",
)
METHOD_ID = "replica_cshape_vs_rectangular_bridge_v1"

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


def _bridge_pre_capture_phases(A: np.ndarray, alpha: float) -> List[Dict[str, Any]]:
    """
    Bridge schedule before capture: slowly ramp bridge while reducing SW.

    The perturbation template is static in space. We only adjust alpha/beta.
    """
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
            "duration_ms": BRIDGE_T_PRE_CAPTURE_RAMP_MS,
            "alpha_start": 0.0,
            "alpha_end": alpha,
            "bsw_start": SW_SCALE,
            "bsw_end": BRIDGE_BSW_PRE_CAPTURE,
            "ctr_start": A.copy(),
            "ctr_end": A.copy(),
            "label": "ramp on",
        },
        {
            "duration_ms": BRIDGE_T_PRE_CAPTURE_HOLD_MS,
            "alpha_start": alpha,
            "alpha_end": alpha,
            "bsw_start": BRIDGE_BSW_PRE_CAPTURE,
            "bsw_end": BRIDGE_BSW_PRE_CAPTURE,
            "ctr_start": A.copy(),
            "ctr_end": A.copy(),
            "label": "bridge hold",
        },
    ]


def _bridge_post_capture_phases(A: np.ndarray, alpha_start: float, bsw_start: float) -> List[Dict[str, Any]]:
    """
    Once capture is reached, ramp bridge down and bring SW back.
    """
    return [
        {
            "duration_ms": BRIDGE_T_POST_CAPTURE_RAMP_MS,
            "alpha_start": alpha_start,
            "alpha_end": 0.0,
            "bsw_start": bsw_start,
            "bsw_end": SW_SCALE,
            "ctr_start": A.copy(),
            "ctr_end": A.copy(),
            "label": "capture -> SW return",
        },
        {
            "duration_ms": BRIDGE_T_POST_CAPTURE_SETTLE_MS,
            "alpha_start": 0.0,
            "alpha_end": 0.0,
            "bsw_start": SW_SCALE,
            "bsw_end": SW_SCALE,
            "ctr_start": A.copy(),
            "ctr_end": A.copy(),
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


def _snap_to_sw_minima(
    traps_m: np.ndarray,
    p_sw: np.ndarray,
    xg: np.ndarray,
    yg: np.ndarray,
    n_steps: int = 2000,
    step_scale: float = 0.02,
) -> np.ndarray:
    dx = float(xg[1] - xg[0])
    dy = float(yg[1] - yg[0])
    _, fx_sw, fy_sw = gorkov_normalised(p_sw, dx, dy)
    i_fx = RegularGridInterpolator((yg, xg), fx_sw, bounds_error=False, fill_value=0.0)
    i_fy = RegularGridInterpolator((yg, xg), fy_sw, bounds_error=False, fill_value=0.0)

    pos = np.asarray(traps_m, dtype=float).copy()
    x_lo, x_hi = float(xg[2]), float(xg[-3])
    y_lo, y_hi = float(yg[2]), float(yg[-3])

    for _ in range(int(n_steps)):
        pts = np.column_stack([pos[:, 1], pos[:, 0]])
        fx = np.asarray(i_fx(pts), dtype=float)
        fy = np.asarray(i_fy(pts), dtype=float)

        fx_den = float(np.max(np.abs(fx))) + 1.0e-30
        fy_den = float(np.max(np.abs(fy))) + 1.0e-30

        pos[:, 0] = np.clip(pos[:, 0] + float(step_scale) * dx * fx / fx_den, x_lo, x_hi)
        pos[:, 1] = np.clip(pos[:, 1] + float(step_scale) * dy * fy / fy_den, y_lo, y_hi)

    return pos


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
_PHASE_COL = {
    "sw only": "#2e7d32",
    "ramp on": "#ef6c00",
    "translate": "#6a1b9a",
    "hold": "#6a1b9a",
    "bridge hold": "#6a1b9a",
    "capture -> sw return": "#1565c0",
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
# Replica lens -> full-domain perturbation field
# ---------------------------------------------------------------------------
def build_replica_full_domain_field(
    x_full: np.ndarray,
    y_full: np.ndarray,
    source_centre_xy: np.ndarray,
    gap_direction_deg: float,
) -> dict[str, Any]:
    """
    Build a full-domain C-shape perturbation field from replica lens thickness.
    """
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
    print(f"Replica lens built + propagated in {dt:.1f} s")

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
            "Scaled full-domain replica C-shape field: "
            f"native_peak={peak_native:.3f} Pa -> target_peak={CSHAPE_TARGET_PEAK_PA:.3f} Pa"
        )

    return {
        "p_full": p_full,
        "peak_full": float(np.max(np.abs(p_full))),
        "gap_direction_deg": float(gap_direction_deg),
    }


def load_or_build_replica_full_domain_field(
    x_full: np.ndarray,
    y_full: np.ndarray,
    source_centre_xy: np.ndarray,
    gap_direction_deg: float,
) -> dict[str, Any]:
    if USE_REPLICA_FIELD_CACHE and REPLICA_FIELD_CACHE_NPZ.exists():
        try:
            d = np.load(REPLICA_FIELD_CACHE_NPZ)
            x_cached = d["x_full"].astype(float)
            y_cached = d["y_full"].astype(float)
            if np.array_equal(x_cached, x_full) and np.array_equal(y_cached, y_full):
                p_cached = d["p_cshape_full_centered"].astype(complex)
                print(f"Loaded cached replica full-domain field: {REPLICA_FIELD_CACHE_NPZ}")
                return {
                    "p_full": p_cached,
                    "peak_full": float(np.max(np.abs(p_cached))),
                    "gap_direction_deg": float(d.get("gap_direction_deg", gap_direction_deg)),
                }
            print("Replica cache grid mismatch; rebuilding field.")
        except Exception as exc:
            print(f"Replica cache load failed ({exc}); rebuilding field.")

    return build_replica_full_domain_field(
        x_full=x_full,
        y_full=y_full,
        source_centre_xy=source_centre_xy,
        gap_direction_deg=gap_direction_deg,
    )


class _StaticPerturbation:
    """Adapter that ignores center and always returns the same field."""

    def __init__(self, field: np.ndarray):
        self._field = np.asarray(field, dtype=complex)

    def get_field(self, _center_xy: np.ndarray) -> np.ndarray:
        return self._field


def _first_capture_index(
    res: TransportResult,
    idx_A: int,
    idx_B: int,
    capture_radius_um: float,
) -> Optional[int]:
    b0 = res.initial_positions[idx_B]
    d_a_to_b0 = np.linalg.norm(
        res.trajectories[:, idx_A, :] - b0[None, :],
        axis=1,
    )
    hit_idx = np.where(d_a_to_b0 <= float(capture_radius_um) * 1.0e-6)[0]
    return int(hit_idx[0]) if hit_idx.size > 0 else None


def _stitch_transport_results(
    pre: TransportResult,
    post: TransportResult,
    cut_idx: int,
) -> TransportResult:
    if cut_idx < 0 or cut_idx >= len(pre.times_s):
        raise ValueError("cut_idx out of bounds for pre result")

    t_cut = float(pre.times_s[cut_idx])
    times = np.concatenate([pre.times_s[: cut_idx + 1], t_cut + post.times_s[1:]])
    traj = np.concatenate([pre.trajectories[: cut_idx + 1], post.trajectories[1:]], axis=0)
    alphas = np.concatenate([pre.alphas[: cut_idx + 1], post.alphas[1:]])
    betas = np.concatenate([pre.betas_sw[: cut_idx + 1], post.betas_sw[1:]])
    centers = np.concatenate([pre.centers[: cut_idx + 1], post.centers[1:]], axis=0)
    labels = pre.phase_labels[: cut_idx + 1] + post.phase_labels[1:]

    merge_time_s = pre.merge_time_s
    if merge_time_s is None and post.merge_time_s is not None:
        merge_time_s = t_cut + float(post.merge_time_s)

    return TransportResult(
        times_s=times,
        trajectories=traj,
        alphas=alphas,
        betas_sw=betas,
        centers=centers,
        phase_labels=labels,
        initial_positions=pre.initial_positions.copy(),
        merge_time_s=merge_time_s,
    )


# ---------------------------------------------------------------------------
# Bridge alpha tuning
# ---------------------------------------------------------------------------
def _a_to_b_progress_metrics(res, idx_A: int, idx_B: int) -> dict[str, float]:
    a_traj = res.trajectories[:, idx_A, :]
    b_traj = res.trajectories[:, idx_B, :]
    a0 = a_traj[0]
    b0 = res.initial_positions[idx_B]

    ab0 = b0 - a0
    d0 = float(np.linalg.norm(ab0))
    if d0 <= 0.0:
        raise RuntimeError("A and B coincide; cannot compute A->B progress metrics")
    e_ab = ab0 / d0

    a_rel = a_traj - a0[None, :]
    progress = a_rel @ e_ab
    move_abs = np.linalg.norm(a_rel, axis=1)
    d_to_b0 = np.linalg.norm(a_traj - b0[None, :], axis=1)
    d_to_b_live = np.linalg.norm(a_traj - b_traj, axis=1)
    # B settles toward SW equilibrium early; use a settled reference, not nominal B0.
    b_settled = b_traj[max(1, len(b_traj) // 10)]
    b_move_abs = np.linalg.norm(b_traj - b_settled[None, :], axis=1)

    return {
        "A_progress_max_um": float(np.max(progress) * 1e6),
        "A_progress_final_um": float(progress[-1] * 1e6),
        "A_move_max_um": float(np.max(move_abs) * 1e6),
        "A_move_final_um": float(move_abs[-1] * 1e6),
        "d_A_to_B0_initial_um": float(d_to_b0[0] * 1e6),
        "d_A_to_B0_min_um": float(np.min(d_to_b0) * 1e6),
        "d_A_to_B0_final_um": float(d_to_b0[-1] * 1e6),
        "d_A_to_B0_max_reduction_um": float((d_to_b0[0] - np.min(d_to_b0)) * 1e6),
        "d_A_to_B_live_min_um": float(np.min(d_to_b_live) * 1e6),
        "d_A_to_B_live_final_um": float(d_to_b_live[-1] * 1e6),
        "B_move_max_um": float(np.max(b_move_abs) * 1e6),
        "B_move_final_um": float(b_move_abs[-1] * 1e6),
    }


def _build_alpha_candidates(alpha_init: float) -> list[float]:
    vals = []
    a = max(alpha_init, 1.0e-9)
    for _ in range(BRIDGE_ALPHA_STEPS):
        vals.append(min(a, BRIDGE_ALPHA_MAX))
        a *= BRIDGE_ALPHA_MULT
    out = sorted(set(round(v, 6) for v in vals))
    return [float(v) for v in out]


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

    # Gap opening direction follows A->B with optional user offset.
    push_vec = B_xy - A_xy
    gap_direction_deg = float(np.rad2deg(np.arctan2(push_vec[1], push_vec[0]))) + REPLICA_GAP_DIR_OFFSET_DEG
    print(f"Replica lens gap direction: {gap_direction_deg:.2f} deg")

    print("\nBuilding/loading full-domain C-shape perturbation from replica lens thickness...")
    replica = load_or_build_replica_full_domain_field(
        x_full=x_full,
        y_full=y_full,
        source_centre_xy=source_centre,
        gap_direction_deg=gap_direction_deg,
    )
    p_cshape_full_centered = replica["p_full"]
    print(f"Replica full-domain |p| peak: {replica['peak_full']:.3f} Pa")

    _bridge_corridor_kwargs = dict(
        width_m=CORRIDOR_WIDTH_M,
        pad_a_m=CORRIDOR_PAD_A_M,
        pad_b_m=CORRIDOR_PAD_B_M,
        edge_s_m=CORRIDOR_EDGE_S_M,
        edge_n_m=CORRIDOR_EDGE_N_M,
        source_hotspot_pa=SOURCE_HOTSPOT_PA,
        destination_pocket_pa=DESTINATION_POCKET_PA,
        source_sigma_m=SOURCE_SIGMA_M,
        destination_sigma_m=DESTINATION_SIGMA_M,
        destination_rest_trap_pa=DESTINATION_REST_TRAP_PA,
        destination_rest_sigma_m=DESTINATION_REST_SIGMA_M,
        source_above_offset_m=SOURCE_ABOVE_OFFSET_M,
        corridor_start_pa=CORRIDOR_START_PA,
        corridor_end_pa=CORRIDOR_END_PA,
        corridor_decay_power=CORRIDOR_DECAY_POWER,
        corridor_transverse_sigma_ratio=CORRIDOR_TRANSVERSE_SIGMA_RATIO,
        b_quiet_radius_m=B_QUIET_RADIUS_M,
    )

    print("\nBuilding fixed rectangular bridge perturbation field...")
    p_bridge_template_full, p_bridge_b_anchor_full, bridge_frame, _, _, _ = build_corridor_bridge_field(
        x=x_full,
        y=y_full,
        point_a=A_xy,
        point_b=B_xy,
        neighbour_positions_m=traps_m[neigh_idx],
        **_bridge_corridor_kwargs,
    )

    if USE_SYMMETRIC_BRIDGE:
        # Build a second corridor from mirror(A) to mirror(B) at full strength,
        # then sum. This constructs a naturally y-axis-symmetric field without
        # halving either corridor's amplitude.
        mirror_A = np.array([-A_xy[0], A_xy[1]])
        mirror_B = np.array([-B_xy[0], B_xy[1]])
        mirror_neighbours = traps_m[neigh_idx].copy()
        mirror_neighbours[:, 0] *= -1.0
        p_bridge_template_mirror, p_bridge_b_anchor_mirror, _, _, _, _ = build_corridor_bridge_field(
            x=x_full,
            y=y_full,
            point_a=mirror_A,
            point_b=mirror_B,
            neighbour_positions_m=mirror_neighbours,
            **_bridge_corridor_kwargs,
        )
        p_bridge_template_full = p_bridge_template_full + p_bridge_template_mirror
        print("Built construct-symmetric bridge: two full-strength corridors (real + mirror)")

    print(
        "Bridge geometry: "
        f"width={CORRIDOR_WIDTH_M * 1e6:.1f} um, "
        f"AB={bridge_frame.d_ab * 1e6:.1f} um"
    )

    # Phase-align anchor(s) to the local standing-wave phase at their location.
    ix_b_anchor = int(np.argmin(np.abs(x_full - B_xy[0])))
    iy_b_anchor = int(np.argmin(np.abs(y_full - B_xy[1])))
    phase_b_sw = float(np.angle(p_sw_full[iy_b_anchor, ix_b_anchor]))
    p_bridge_b_anchor_full = p_bridge_b_anchor_full * np.exp(1j * phase_b_sw)

    if USE_SYMMETRIC_BRIDGE:
        ix_mb = int(np.argmin(np.abs(x_full - (-B_xy[0]))))
        iy_mb = int(np.argmin(np.abs(y_full - B_xy[1])))
        phase_mb_sw = float(np.angle(p_sw_full[iy_mb, ix_mb]))
        p_bridge_b_anchor_full = (
            p_bridge_b_anchor_full
            + p_bridge_b_anchor_mirror * np.exp(1j * phase_mb_sw)
        )

    # Destination anchor is baked into the SW baseline.
    p_sw_bridge_full = p_sw_full + p_bridge_b_anchor_full

    if USE_SYMMETRIC_BRIDGE and SYMM_EXTRA_B_POCKET_PA > 0.0:
        # Add a real Gaussian restoring pocket at B so the destination trap is
        # reinforced – matches the field shown in bridge_pressure_field_standalone_scaled.
        yy_full, xx_full = np.meshgrid(y_full, x_full, indexing="ij")
        s2 = max(float(SYMM_EXTRA_B_POCKET_SIGMA_M), 1.0e-12) ** 2
        pocket = SYMM_EXTRA_B_POCKET_PA * np.exp(
            -0.5 * ((xx_full - B_xy[0]) ** 2 + (yy_full - B_xy[1]) ** 2) / s2
        )
        p_sw_bridge_full = p_sw_bridge_full + pocket.astype(complex)
        print(
            f"Added extra B-pocket to SW baseline: "
            f"pa={SYMM_EXTRA_B_POCKET_PA:.1f} Pa, "
            f"sigma={SYMM_EXTRA_B_POCKET_SIGMA_M*1e6:.0f} um"
        )

    gen_cshape_full = VortexPerturbation(
        p_cshape_full_centered,
        x_full,
        y_full,
        out_xg=x_full,
        out_yg=y_full,
    )
    gen_bridge_full = _StaticPerturbation(p_bridge_template_full)

    print("\nRunning replica C-shape transport (left column)...")
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

    print("\nRunning bridge transport with slow ramp until capture, then SW return...")
    tune_log: list[dict[str, Any]] = []

    def run_bridge(alpha_val: float) -> tuple[TransportResult, dict[str, Any]]:
        res_pre = run_transport(
            p_sw=p_sw_bridge_full,
            perturbation_fn=gen_bridge_full.get_field,
            phases=_bridge_pre_capture_phases(A_xy, alpha_val),
            psi=BRIDGE_PSI,
            initial_positions=traps_m.copy(),
            xg=x_full,
            yg=y_full,
            idx_A=idx_A,
            idx_B=idx_B,
            dt=DT,
            n_frames=N_FRAMES,
            n_keyframes=12,
            mobility_scale=1.0,
        )

        capture_idx = _first_capture_index(
            res_pre,
            idx_A=idx_A,
            idx_B=idx_B,
            capture_radius_um=A_TO_B_CAPTURE_RADIUS_UM,
        )
        if capture_idx is None:
            transition_idx = len(res_pre.times_s) - 1
            capture_reached = False
        else:
            transition_idx = int(capture_idx)
            capture_reached = True

        transition_state = res_pre.trajectories[transition_idx].copy()
        transition_alpha = float(res_pre.alphas[transition_idx])
        transition_bsw = float(res_pre.betas_sw[transition_idx])

        res_post = run_transport(
            p_sw=p_sw_bridge_full,
            perturbation_fn=gen_bridge_full.get_field,
            phases=_bridge_post_capture_phases(
                transition_state[idx_A],
                alpha_start=transition_alpha,
                bsw_start=transition_bsw,
            ),
            psi=BRIDGE_PSI,
            initial_positions=transition_state,
            xg=x_full,
            yg=y_full,
            idx_A=idx_A,
            idx_B=idx_B,
            dt=DT,
            n_frames=BRIDGE_POST_FRAMES,
            n_keyframes=8,
            mobility_scale=1.0,
        )
        res_full = _stitch_transport_results(res_pre, res_post, transition_idx)

        return res_full, {
            "capture_reached": bool(capture_reached),
            "capture_idx": int(capture_idx) if capture_idx is not None else None,
            "capture_time_ms": (
                float(res_pre.times_s[capture_idx] * 1e3) if capture_idx is not None else None
            ),
            "transition_idx": int(transition_idx),
            "transition_time_ms": float(res_pre.times_s[transition_idx] * 1e3),
            "transition_alpha": float(transition_alpha),
            "transition_bsw": float(transition_bsw),
            "pre_frames": int(len(res_pre.times_s)),
            "post_frames": int(len(res_post.times_s)),
            "full_frames": int(len(res_full.times_s)),
        }

    selected_bridge_alpha = BRIDGE_ALPHA_INIT
    best_bridge_res = None
    best_bridge_meta: dict[str, Any] | None = None
    best_row: dict[str, Any] | None = None
    best_score = float("inf")
    target_hit = False

    if AUTO_TUNE_BRIDGE_ALPHA:
        alpha_candidates = _build_alpha_candidates(BRIDGE_ALPHA_INIT)
        print(f"Bridge alpha candidates: {alpha_candidates}")

        for alpha in alpha_candidates:
            t_alpha = time.perf_counter()
            res_b, bridge_meta = run_bridge(alpha)
            m = _a_to_b_progress_metrics(res_b, idx_A, idx_B)
            elapsed = time.perf_counter() - t_alpha

            row = {
                "alpha": float(alpha),
                "A_progress_max_um": m["A_progress_max_um"],
                "A_progress_final_um": m["A_progress_final_um"],
                "A_move_max_um": m["A_move_max_um"],
                "B_move_max_um": m["B_move_max_um"],
                "B_move_final_um": m["B_move_final_um"],
                "d_A_to_B0_min_um": m["d_A_to_B0_min_um"],
                "d_A_to_B0_max_reduction_um": m["d_A_to_B0_max_reduction_um"],
                "d_A_to_B0_final_um": m["d_A_to_B0_final_um"],
                "d_A_to_B_live_min_um": m["d_A_to_B_live_min_um"],
                "d_A_to_B_live_final_um": m["d_A_to_B_live_final_um"],
                "capture_reached": bool(bridge_meta["capture_reached"]),
                "capture_time_ms": bridge_meta["capture_time_ms"],
                "transition_time_ms": bridge_meta["transition_time_ms"],
                "transition_alpha": bridge_meta["transition_alpha"],
                "transition_bsw": bridge_meta["transition_bsw"],
            }
            capture_penalty = 0.0 if row["capture_reached"] else 5.0e3
            row_score = (
                capture_penalty
                + row["d_A_to_B0_min_um"]
                + 15.0 * max(0.0, row["B_move_max_um"] - B_TRAP_MAX_DRIFT_UM)
                + 0.2 * max(0.0, A_TO_B_PROGRESS_THRESHOLD_UM - row["A_progress_max_um"])
            )
            row["selection_score"] = float(row_score)
            tune_log.append(row)

            print(
                f"  alpha={alpha:.3f}: "
                f"A_progress_max={row['A_progress_max_um']:.2f} um, "
                f"A_to_B0_min={row['d_A_to_B0_min_um']:.2f} um, "
                f"A_to_B_live_min={row['d_A_to_B_live_min_um']:.2f} um, "
                f"B_move_max={row['B_move_max_um']:.2f} um, "
                f"capture={row['capture_reached']} "
                f"({elapsed:.1f} s)"
            )

            if row_score < best_score:
                best_score = row_score
                best_row = row
                selected_bridge_alpha = alpha
                best_bridge_res = res_b
                best_bridge_meta = bridge_meta

            progress_ok = row["A_progress_max_um"] >= A_TO_B_PROGRESS_THRESHOLD_UM
            reaches_b = row["d_A_to_B0_min_um"] <= A_TO_B_CAPTURE_RADIUS_UM
            b_stable = row["B_move_max_um"] < 30.0

            if progress_ok and reaches_b and b_stable:
                target_hit = True
                best_row = row
                selected_bridge_alpha = alpha
                best_bridge_res = res_b
                best_bridge_meta = bridge_meta
                print(
                    "  target met: "
                    f"A_to_B_live_min <= {A_TO_B_CAPTURE_RADIUS_UM:.1f} um and "
                    f"B_move_max < 6.0 um at alpha={alpha:.3f}"
                )
                break
    else:
        best_bridge_res, best_bridge_meta = run_bridge(BRIDGE_ALPHA_INIT)
        selected_bridge_alpha = BRIDGE_ALPHA_INIT
        m = _a_to_b_progress_metrics(best_bridge_res, idx_A, idx_B)
        row = {
            "alpha": float(selected_bridge_alpha),
            "A_progress_max_um": m["A_progress_max_um"],
            "A_progress_final_um": m["A_progress_final_um"],
            "A_move_max_um": m["A_move_max_um"],
            "B_move_max_um": m["B_move_max_um"],
            "B_move_final_um": m["B_move_final_um"],
            "d_A_to_B0_min_um": m["d_A_to_B0_min_um"],
            "d_A_to_B0_max_reduction_um": m["d_A_to_B0_max_reduction_um"],
            "d_A_to_B0_final_um": m["d_A_to_B0_final_um"],
            "d_A_to_B_live_min_um": m["d_A_to_B_live_min_um"],
            "d_A_to_B_live_final_um": m["d_A_to_B_live_final_um"],
            "capture_reached": bool(best_bridge_meta["capture_reached"]),
            "capture_time_ms": best_bridge_meta["capture_time_ms"],
            "transition_time_ms": best_bridge_meta["transition_time_ms"],
            "transition_alpha": best_bridge_meta["transition_alpha"],
            "transition_bsw": best_bridge_meta["transition_bsw"],
        }
        capture_penalty = 0.0 if row["capture_reached"] else 5.0e3
        row_score = (
            capture_penalty
            + row["d_A_to_B0_min_um"]
            + 15.0 * max(0.0, row["B_move_max_um"] - B_TRAP_MAX_DRIFT_UM)
            + 0.2 * max(0.0, A_TO_B_PROGRESS_THRESHOLD_UM - row["A_progress_max_um"])
        )
        row["selection_score"] = float(row_score)
        best_row = row
        best_score = row_score
        tune_log.append(row)

    if best_bridge_res is None:
        raise RuntimeError("Bridge simulation did not produce any result")
    if best_bridge_meta is None:
        raise RuntimeError("Bridge simulation metadata was not produced")

    res_b = best_bridge_res
    m_bridge = _a_to_b_progress_metrics(res_b, idx_A, idx_B)
    print(
        f"Selected bridge alpha={selected_bridge_alpha:.3f} | "
        f"A_progress_max={m_bridge['A_progress_max_um']:.2f} um | "
        f"A_to_B_live_min={m_bridge['d_A_to_B_live_min_um']:.2f} um | "
        f"B_move_max={m_bridge['B_move_max_um']:.2f} um | "
        f"capture={best_bridge_meta['capture_reached']}"
    )

    tune_json = OUT_DIR / "rectangular_bridge_alpha_tuning.json"
    with open(tune_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "auto_tune_bridge_alpha": bool(AUTO_TUNE_BRIDGE_ALPHA),
                "A_to_B_progress_threshold_um": float(A_TO_B_PROGRESS_THRESHOLD_UM),
                "A_to_B_capture_radius_um": float(A_TO_B_CAPTURE_RADIUS_UM),
                "B_trap_max_drift_um": float(B_TRAP_MAX_DRIFT_UM),
                "target_hit": bool(target_hit),
                "threshold_hit": bool(target_hit),
                "selected_bridge_alpha": float(selected_bridge_alpha),
                "bridge_psi": float(BRIDGE_PSI),
                "selected_meta": best_bridge_meta,
                "bridge_schedule_ms": {
                    "sw_only": float(T_SW_ONLY_MS),
                    "pre_capture_ramp": float(BRIDGE_T_PRE_CAPTURE_RAMP_MS),
                    "pre_capture_hold": float(BRIDGE_T_PRE_CAPTURE_HOLD_MS),
                    "post_capture_ramp": float(BRIDGE_T_POST_CAPTURE_RAMP_MS),
                    "post_capture_settle": float(BRIDGE_T_POST_CAPTURE_SETTLE_MS),
                },
                "bridge_bsw_pre_capture": float(BRIDGE_BSW_PRE_CAPTURE),
                "selected_row": best_row,
                "rows": tune_log,
            },
            f,
            indent=2,
        )
    print(f"Saved bridge alpha tuning log: {tune_json}")

    n_phys_frames = min(len(res_c.times_s), len(res_b.times_s))
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
        f"({n_phys_frames} physics frames, subframes={DISPLAY_SUBFRAMES}, cap={MAX_OUTPUT_FRAMES or 'none'})"
    )

    frames: list[Image.Image] = []

    for fi in range(n_frames):
        u = fi / float(DISPLAY_SUBFRAMES)
        i0 = int(np.floor(u))
        i1 = min(i0 + 1, n_phys_frames - 1)
        tau = float(u - i0)

        label_c = res_c.phase_labels[i0]
        label_b = res_b.phase_labels[i0]
        t_ms = float(_lerp(res_c.times_s[i0], res_c.times_s[i1], tau) * 1e3)

        alpha_c = float(_lerp(res_c.alphas[i0], res_c.alphas[i1], tau))
        beta_c = float(_lerp(res_c.betas_sw[i0], res_c.betas_sw[i1], tau))
        ctr_c = _lerp(res_c.centers[i0], res_c.centers[i1], tau)

        alpha_b = float(_lerp(res_b.alphas[i0], res_b.alphas[i1], tau))
        beta_b = float(_lerp(res_b.betas_sw[i0], res_b.betas_sw[i1], tau))
        ctr_b = _lerp(res_b.centers[i0], res_b.centers[i1], tau)

        p_tot_c_full = _total_pressure(
            p_sw_full,
            gen_cshape_full.get_field,
            CSHAPE_PSI,
            alpha_c,
            beta_c,
            ctr_c,
        )
        p_tot_b_full = _total_pressure(
            p_sw_bridge_full,
            gen_bridge_full.get_field,
            BRIDGE_PSI,
            alpha_b,
            beta_b,
            ctr_b,
        )

        U_c_full, _, _ = _gorkov_fields(p_tot_c_full, dx_full, dy_full)
        U_b_full, _, _ = _gorkov_fields(p_tot_b_full, dx_full, dy_full)

        U_c_roi = U_c_full[np.ix_(iy_roi, ix_roi)]
        U_b_roi = U_b_full[np.ix_(iy_roi, ix_roi)]

        vmin, vmax = _auto_limits(U_c_full, U_b_full)

        cur_c = _lerp(res_c.trajectories[i0], res_c.trajectories[i1], tau) * 1e3
        cur_b = _lerp(res_b.trajectories[i0], res_b.trajectories[i1], tau) * 1e3

        if i1 == i0:
            path_c = res_c.centers[: i0 + 1] * 1e3
            path_b = res_b.centers[: i0 + 1] * 1e3
        else:
            path_c = np.vstack([res_c.centers[: i0 + 1], ctr_c]) * 1e3
            path_b = np.vstack([res_b.centers[: i0 + 1], ctr_b]) * 1e3

        fig, axes = plt.subplots(2, 2, figsize=(11.2, 10.4))
        (ax_c_roi, ax_b_roi), (ax_c_full, ax_b_full) = axes

        _draw_panel(
            ax_c_roi,
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
            f"Replica C-shape - U_Gorkov (ROI crop of full)  ({label_c})",
        )
        _draw_panel(
            ax_b_roi,
            U_b_roi,
            vmin,
            vmax,
            extent_roi,
            x_roi_mm,
            y_roi_mm,
            traps_mm,
            cur_b,
            path_b,
            idx_A,
            idx_B,
            neigh_idx,
            f"Rectangular bridge - U_Gorkov (ROI crop of full)  ({label_b})",
        )
        _draw_panel(
            ax_c_full,
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
            f"Replica C-shape - U_Gorkov (full domain)  ({label_c})",
        )
        _draw_panel(
            ax_b_full,
            U_b_full,
            vmin,
            vmax,
            extent_full,
            x_full_mm,
            y_full_mm,
            traps_mm,
            cur_b,
            path_b,
            idx_A,
            idx_B,
            neigh_idx,
            f"Rectangular bridge - U_Gorkov (full domain)  ({label_b})",
        )

        fig.suptitle(
            f"t = {t_ms:.0f} ms | frame {fi + 1}/{n_frames} | "
            f"bridge_alpha={selected_bridge_alpha:.3f} | "
            f"A_to_B_live_min={m_bridge['d_A_to_B_live_min_um']:.1f} um | "
            f"capture={best_bridge_meta['capture_reached']}",
            fontsize=10,
            color=_phase_color(label_b),
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

    cshape_metrics = _a_to_b_progress_metrics(res_c, idx_A, idx_B)

    manifest_json = OUT_DIR / "replica_cshape_vs_rectangular_bridge_manifest.json"
    with open(manifest_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "method_id": METHOD_ID,
                "script": "scripts/dev/transport_side_by_side_replica_cshape_vs_rectangular_bridge.py",
                "output_dir": str(OUT_DIR.relative_to(PROJECT_ROOT)),
                "outputs": {
                    "gif": str(out_path.relative_to(PROJECT_ROOT)),
                    "bridge_alpha_tuning": str(tune_json.relative_to(PROJECT_ROOT)),
                },
                "controls": {
                    "sw_scale": float(SW_SCALE),
                    "cshape_alpha": float(CSHAPE_ALPHA),
                    "cshape_psi": float(CSHAPE_PSI),
                    "bridge_selected_alpha": float(selected_bridge_alpha),
                    "bridge_psi": float(BRIDGE_PSI),
                    "bridge_auto_tune_alpha": bool(AUTO_TUNE_BRIDGE_ALPHA),
                    "A_to_B_progress_threshold_um": float(A_TO_B_PROGRESS_THRESHOLD_UM),
                    "A_to_B_capture_radius_um": float(A_TO_B_CAPTURE_RADIUS_UM),
                    "B_trap_max_drift_um": float(B_TRAP_MAX_DRIFT_UM),
                    "bridge_t_pre_capture_ramp_ms": float(BRIDGE_T_PRE_CAPTURE_RAMP_MS),
                    "bridge_t_pre_capture_hold_ms": float(BRIDGE_T_PRE_CAPTURE_HOLD_MS),
                    "bridge_t_post_capture_ramp_ms": float(BRIDGE_T_POST_CAPTURE_RAMP_MS),
                    "bridge_t_post_capture_settle_ms": float(BRIDGE_T_POST_CAPTURE_SETTLE_MS),
                    "bridge_bsw_pre_capture": float(BRIDGE_BSW_PRE_CAPTURE),
                    "bridge_post_frames": int(BRIDGE_POST_FRAMES),
                    "display_subframes": int(DISPLAY_SUBFRAMES),
                    "n_frames": int(N_FRAMES),
                    "gif_duration_ms": int(GIF_DURATION_MS),
                },
                "bridge_geometry": {
                    "use_symmetric_bridge": bool(USE_SYMMETRIC_BRIDGE),
                    "symm_extra_b_pocket_pa": float(SYMM_EXTRA_B_POCKET_PA),
                    "symm_extra_b_pocket_sigma_m": float(SYMM_EXTRA_B_POCKET_SIGMA_M),
                    "corridor_width_m": float(CORRIDOR_WIDTH_M),
                    "corridor_pad_a_m": float(CORRIDOR_PAD_A_M),
                    "corridor_pad_b_m": float(CORRIDOR_PAD_B_M),
                    "corridor_edge_s_m": float(CORRIDOR_EDGE_S_M),
                    "corridor_edge_n_m": float(CORRIDOR_EDGE_N_M),
                    "source_hotspot_pa": float(SOURCE_HOTSPOT_PA),
                    "destination_pocket_pa": float(DESTINATION_POCKET_PA),
                    "destination_rest_trap_pa": float(DESTINATION_REST_TRAP_PA),
                    "source_sigma_m": float(SOURCE_SIGMA_M),
                    "destination_sigma_m": float(DESTINATION_SIGMA_M),
                    "destination_rest_sigma_m": float(DESTINATION_REST_SIGMA_M),
                    "source_above_offset_m": float(SOURCE_ABOVE_OFFSET_M),
                    "corridor_start_pa": float(CORRIDOR_START_PA),
                    "corridor_end_pa": float(CORRIDOR_END_PA),
                    "corridor_decay_power": float(CORRIDOR_DECAY_POWER),
                    "corridor_transverse_sigma_ratio": float(CORRIDOR_TRANSVERSE_SIGMA_RATIO),
                },
                "metrics": {
                    "cshape_A_progress_max_um": float(cshape_metrics["A_progress_max_um"]),
                    "cshape_A_progress_final_um": float(cshape_metrics["A_progress_final_um"]),
                    "bridge_A_progress_max_um": float(m_bridge["A_progress_max_um"]),
                    "bridge_A_progress_final_um": float(m_bridge["A_progress_final_um"]),
                    "bridge_d_A_to_B0_min_um": float(m_bridge["d_A_to_B0_min_um"]),
                    "bridge_d_A_to_B_live_min_um": float(m_bridge["d_A_to_B_live_min_um"]),
                    "bridge_d_A_to_B0_max_reduction_um": float(m_bridge["d_A_to_B0_max_reduction_um"]),
                    "bridge_B_move_max_um": float(m_bridge["B_move_max_um"]),
                    "bridge_target_hit": bool(target_hit),
                    "bridge_capture_reached": bool(best_bridge_meta["capture_reached"]),
                    "bridge_capture_time_ms": best_bridge_meta["capture_time_ms"],
                    "bridge_transition_alpha": float(best_bridge_meta["transition_alpha"]),
                    "bridge_transition_bsw": float(best_bridge_meta["transition_bsw"]),
                },
                "notes": [
                    "Left column uses replica C-shape full-domain field with ROI as strict crop.",
                    "Right column uses deterministic bridge with high-pressure hotspot above A and decaying A->B corridor forcing.",
                    "Bridge alpha is tuned with a slow pre-capture ramp while preserving partial SW anchoring.",
                    "B-end resting trap is reinforced at the destination to keep B from drifting toward A.",
                    "Once A reaches B, bridge strength ramps down while standing-wave strength ramps back up.",
                ],
            },
            f,
            indent=2,
        )

    print(f"\nSaved GIF: {out_path}")
    print(f"Saved bridge alpha tuning: {tune_json}")
    print(f"Saved manifest: {manifest_json}")
    print(f"Output dir: {OUT_DIR}")


if __name__ == "__main__":
    main()
