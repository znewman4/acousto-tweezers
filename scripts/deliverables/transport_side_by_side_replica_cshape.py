#!/usr/bin/env python3
"""
Side-by-side transport GIF: vortex vs replica C-shape lens (full-domain).

This version enforces a single physically consistent C-shape perturbation path:
1. Build the replica lens with the same IASA pipeline used in
   scripts/dev/inverse_c_shape_lens_replica.py.
2. Convert the resulting wrapped phase to a thickness profile.
3. Convert thickness back to a transmitted aperture phase field.
4. Propagate that aperture field with ASM over the full lens domain.
5. Resample to the full transport grid and use this ONE field everywhere.
6. Translate this real propagated C-shape field in-plane during transport.

The ROI panels are strict crops of the full-domain Gor'kov maps.
No separate ROI-only C-shape solve is used.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
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
from scripts.lib.particle_dynamics_utils import run_transport
from scripts.lib.perturbation_vortex import VortexPerturbation

PPAR = default_particle_params()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUT_DIR = PROJECT_ROOT / "results" / "deliverables" / "transport_side_by_side"

VORTEX_NPZ = (
    PROJECT_ROOT / "results" / "deliverables" / "vortex_stage_transport"
    / "transport" / "transport_case_for_gif.npz"
)
OVERLAY_NPZ = (
    PROJECT_ROOT / "results" / "c_shape_lens_15mm_overlay_study_20260310_170620"
    / "npz" / "roi_fields.npz"
)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "1" if default else "0").strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------
VORTEX_ALPHA = float(os.getenv("VORTEX_ALPHA", "2.10"))
VORTEX_PSI = float(os.getenv("VORTEX_PSI", str(0.5 * np.pi)))

SW_SCALE = float(os.getenv("SW_SCALE", "1.0"))
CSHAPE_CONST_SW = _env_bool("CSHAPE_CONST_SW", True)
CSHAPE_BSW_MIN = float(os.getenv("CSHAPE_BSW_MIN", "0.62"))
CSHAPE_PSI = float(os.getenv("CSHAPE_PSI", str(1.5 * np.pi)))

CSHAPE_TRANSLATE = _env_bool("CSHAPE_TRANSLATE", True)
CSHAPE_MOBILITY_SCALE = max(0.0, float(os.getenv("CSHAPE_MOBILITY_SCALE", "1.0")))

# Alpha tuning (until A moves)
AUTO_TUNE_ALPHA = _env_bool("AUTO_TUNE_ALPHA", True)
CSHAPE_ALPHA_INIT = float(os.getenv("CSHAPE_ALPHA", "10.5"))
ALPHA_MULT = float(os.getenv("ALPHA_MULT", "1.8"))
ALPHA_STEPS = max(1, int(os.getenv("ALPHA_STEPS", "7")))
ALPHA_MAX = float(os.getenv("ALPHA_MAX", "300.0"))
A_MOVE_THRESHOLD_UM = float(os.getenv("A_MOVE_THRESHOLD_UM", "30.0"))

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

# Optional normalisation of the propagated C-shape field peak
# 0 or negative => keep native amplitude
CSHAPE_TARGET_PEAK_PA = float(os.getenv("CSHAPE_TARGET_PEAK_PA", "0.0"))

# Schedules (ms)
T_SW_ONLY_MS = 120.0
T_RAMP_ON_MS = 220.0
T_ACTIVE_MS = 260.0
T_RAMP_OFF_MS = 220.0
T_SETTLE_MS = 800.0

CSHAPE_T_RAMP_ON_MS = float(os.getenv("CSHAPE_T_RAMP_ON_MS", "420.0"))
CSHAPE_T_ACTIVE_MS = float(os.getenv("CSHAPE_T_ACTIVE_MS", "620.0"))
CSHAPE_T_RAMP_OFF_MS = float(os.getenv("CSHAPE_T_RAMP_OFF_MS", "260.0"))
CSHAPE_T_SETTLE_MS = float(os.getenv("CSHAPE_T_SETTLE_MS", str(T_SETTLE_MS)))

DT = 1.0e-4
N_FRAMES = 320

GIF_DURATION_MS = max(10, int(os.getenv("GIF_DURATION_MS", "110")))
DISPLAY_SUBFRAMES = max(1, int(os.getenv("DISPLAY_SUBFRAMES", "3")))
_max_frames_env = os.getenv("MAX_OUTPUT_FRAMES", "").strip()
MAX_OUTPUT_FRAMES = max(0, int(_max_frames_env)) if _max_frames_env else 0

OUT_GIF_NAME = os.getenv(
    "OUT_GIF_NAME",
    "transport_vortex_vs_cshape_gorkov_replica_fullfield_translated.gif",
)
METHOD_ID = "replica_cshape_fullfield_translated_v1"

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
def _vortex_phases(A: np.ndarray, B: np.ndarray) -> List[Dict[str, Any]]:
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
            "duration_ms": T_RAMP_ON_MS,
            "alpha_start": 0.0,
            "alpha_end": VORTEX_ALPHA,
            "bsw_start": SW_SCALE,
            "bsw_end": SW_SCALE,
            "ctr_start": A.copy(),
            "ctr_end": A.copy(),
            "label": "ramp on",
        },
        {
            "duration_ms": T_ACTIVE_MS,
            "alpha_start": VORTEX_ALPHA,
            "alpha_end": VORTEX_ALPHA,
            "bsw_start": SW_SCALE,
            "bsw_end": SW_SCALE,
            "ctr_start": A.copy(),
            "ctr_end": B.copy(),
            "label": "translate",
        },
        {
            "duration_ms": T_RAMP_OFF_MS,
            "alpha_start": VORTEX_ALPHA,
            "alpha_end": 0.0,
            "bsw_start": SW_SCALE,
            "bsw_end": SW_SCALE,
            "ctr_start": B.copy(),
            "ctr_end": B.copy(),
            "label": "ramp off",
        },
        {
            "duration_ms": T_SETTLE_MS,
            "alpha_start": 0.0,
            "alpha_end": 0.0,
            "bsw_start": SW_SCALE,
            "bsw_end": SW_SCALE,
            "ctr_start": B.copy(),
            "ctr_end": B.copy(),
            "label": "SW release",
        },
    ]


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

    Returns dict with full-domain field and intermediate lens-profile data.
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
    x_lens, y_lens, xg_lens, yg_lens, r, theta, aperture_mask, dx_lens = make_grid(cfg)
    target_amp, _ = build_c_target_amplitude(r, theta, cfg)
    lens_field = run_iasa(cfg, aperture_mask, target_amp, dx_lens)

    # Thickness profile that defines the printable lens geometry.
    phi_wrapped = np.mod(np.angle(lens_field), 2.0 * np.pi)
    thickness = cfg.h_base_m + cfg.h_max_m * (phi_wrapped / (2.0 * np.pi))
    thickness[~aperture_mask] = cfg.h_base_m

    # Convert thickness back to transmitted phase using |dk| to match mapping.
    phase_delay = np.abs(cfg.k_lens - cfg.k_water) * np.maximum(thickness - cfg.h_base_m, 0.0)
    aperture_field = np.zeros_like(lens_field, dtype=complex)
    aperture_field[aperture_mask] = np.exp(1j * np.mod(phase_delay[aperture_mask], 2.0 * np.pi))

    # Full-domain propagation comes from this real lens aperture field.
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
        "cfg": cfg,
        "x_lens": x_lens,
        "y_lens": y_lens,
        "aperture_mask": aperture_mask,
        "phi_wrapped": phi_wrapped,
        "thickness": thickness,
        "aperture_field": aperture_field,
        "p_focal_lens_grid": p_focal_lens_grid,
        "p_full": p_full,
        "peak_full": float(np.max(np.abs(p_full))),
    }


# ---------------------------------------------------------------------------
# Alpha tuning
# ---------------------------------------------------------------------------
def _a_motion_metrics(res, idx_A: int, idx_B: int) -> dict[str, float]:
    a0 = res.trajectories[0, idx_A]
    a_traj = res.trajectories[:, idx_A, :]
    move = np.linalg.norm(a_traj - a0, axis=1)
    a_move_max_um = float(np.max(move) * 1e6)
    a_move_final_um = float(move[-1] * 1e6)
    d_ab_final_um = float(np.linalg.norm(res.trajectories[-1, idx_A] - res.trajectories[-1, idx_B]) * 1e6)
    return {
        "A_move_max_um": a_move_max_um,
        "A_move_final_um": a_move_final_um,
        "d_AB_final_um": d_ab_final_um,
    }


def _build_alpha_candidates(alpha_init: float) -> list[float]:
    vals = []
    a = max(alpha_init, 1e-9)
    for _ in range(ALPHA_STEPS):
        vals.append(min(a, ALPHA_MAX))
        a *= ALPHA_MULT
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
    p_vortex_centered = vd["p_vortex_centered"].astype(complex)

    neigh_idx = np.array(sorted(set(range(len(traps_m))) - {idx_A, idx_B}), dtype=int)

    A_xy = traps_m[idx_A]
    B_xy = traps_m[idx_B]
    source_centre = np.array(
        [0.5 * (float(x_full[0]) + float(x_full[-1])), 0.5 * (float(y_full[0]) + float(y_full[-1]))],
        dtype=float,
    )

    print(f"Trap A idx={idx_A}: {A_xy * 1e3} mm")
    print(f"Trap B idx={idx_B}: {B_xy * 1e3} mm")
    print(f"Full domain: [{x_full[0] * 1e3:.3f}, {x_full[-1] * 1e3:.3f}] mm")
    print(f"ROI bounds:  [{x_roi[0] * 1e3:.3f}, {x_roi[-1] * 1e3:.3f}] mm")

    # Gap opening direction set from A -> B push direction, with optional offset.
    push_vec = B_xy - A_xy
    gap_direction_deg = float(np.rad2deg(np.arctan2(push_vec[1], push_vec[0]))) + REPLICA_GAP_DIR_OFFSET_DEG
    print(f"Replica lens gap direction: {gap_direction_deg:.2f} deg")

    print("\nBuilding full-domain C-shape perturbation from replica lens thickness...")
    replica = build_replica_full_domain_field(
        x_full=x_full,
        y_full=y_full,
        source_centre_xy=source_centre,
        gap_direction_deg=gap_direction_deg,
    )
    p_cshape_full_centered = replica["p_full"]
    print(f"Replica full-domain |p| peak: {replica['peak_full']:.3f} Pa")

    field_npz = OUT_DIR / "replica_cshape_full_domain_field.npz"
    np.savez(
        field_npz,
        p_cshape_full_centered=p_cshape_full_centered,
        x_full=x_full,
        y_full=y_full,
        source_centre=source_centre,
        traps_m=traps_m,
        idx_A=np.array(idx_A),
        idx_B=np.array(idx_B),
        phase_wrapped=replica["phi_wrapped"],
        thickness=replica["thickness"],
        x_lens=replica["x_lens"],
        y_lens=replica["y_lens"],
        aperture_mask=replica["aperture_mask"],
        p_focal_lens_grid=replica["p_focal_lens_grid"],
        arc_radius_mm=np.array(REPLICA_ARC_RADIUS_MM),
        arc_width_mm=np.array(REPLICA_ARC_WIDTH_MM),
        gap_angle_deg=np.array(REPLICA_GAP_ANGLE_DEG),
        gap_direction_deg=np.array(gap_direction_deg),
        n_iter=np.array(REPLICA_N_ITER),
    )
    print(f"Saved full-domain replica field package: {field_npz}")

    # Perturbation generators (translation by center_xy)
    gen_vortex_full = VortexPerturbation(
        p_vortex_centered,
        x_full,
        y_full,
        out_xg=x_full,
        out_yg=y_full,
    )
    gen_cshape_full = VortexPerturbation(
        p_cshape_full_centered,
        x_full,
        y_full,
        out_xg=x_full,
        out_yg=y_full,
    )

    # Run vortex reference transport on full domain.
    print("\nRunning vortex transport on full domain...")
    t0 = time.perf_counter()
    res_v = run_transport(
        p_sw=p_sw_full,
        perturbation_fn=gen_vortex_full.get_field,
        phases=_vortex_phases(A_xy, B_xy),
        psi=VORTEX_PSI,
        initial_positions=traps_m.copy(),
        xg=x_full,
        yg=y_full,
        idx_A=idx_A,
        idx_B=idx_B,
        dt=DT,
        n_frames=N_FRAMES,
        n_keyframes=20,
    )
    print(f"Vortex transport done in {time.perf_counter() - t0:.1f} s")

    # Tune C-shape alpha until A moves.
    print("\nRunning translated C-shape transport (full-domain field)...")

    tune_log: list[dict[str, float]] = []

    def run_cshape(alpha_val: float):
        return run_transport(
            p_sw=p_sw_full,
            perturbation_fn=gen_cshape_full.get_field,
            phases=_cshape_phases(A_xy, B_xy, alpha_val),
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

    selected_alpha = CSHAPE_ALPHA_INIT
    best_res = None
    best_move = -1.0

    if AUTO_TUNE_ALPHA:
        alphas = _build_alpha_candidates(CSHAPE_ALPHA_INIT)
        print(f"Alpha candidates: {alphas}")

        for alpha in alphas:
            t_alpha = time.perf_counter()
            res = run_cshape(alpha)
            metrics = _a_motion_metrics(res, idx_A, idx_B)
            elapsed = time.perf_counter() - t_alpha

            row = {
                "alpha": float(alpha),
                "A_move_max_um": metrics["A_move_max_um"],
                "A_move_final_um": metrics["A_move_final_um"],
                "d_AB_final_um": metrics["d_AB_final_um"],
            }
            tune_log.append(row)

            print(
                f"  alpha={alpha:.3f}: "
                f"A_move_max={row['A_move_max_um']:.2f} um, "
                f"A_move_final={row['A_move_final_um']:.2f} um, "
                f"d_AB_final={row['d_AB_final_um']:.2f} um "
                f"({elapsed:.1f} s)"
            )

            if row["A_move_max_um"] > best_move:
                best_move = row["A_move_max_um"]
                best_res = res
                selected_alpha = alpha

            if row["A_move_max_um"] >= A_MOVE_THRESHOLD_UM:
                print(
                    f"  threshold met: A_move_max >= {A_MOVE_THRESHOLD_UM:.1f} um "
                    f"at alpha={alpha:.3f}"
                )
                best_res = res
                selected_alpha = alpha
                break
    else:
        best_res = run_cshape(CSHAPE_ALPHA_INIT)
        selected_alpha = CSHAPE_ALPHA_INIT
        metrics = _a_motion_metrics(best_res, idx_A, idx_B)
        tune_log.append(
            {
                "alpha": float(selected_alpha),
                "A_move_max_um": metrics["A_move_max_um"],
                "A_move_final_um": metrics["A_move_final_um"],
                "d_AB_final_um": metrics["d_AB_final_um"],
            }
        )

    if best_res is None:
        raise RuntimeError("C-shape simulation did not produce any result")

    res_c = best_res
    m_sel = _a_motion_metrics(res_c, idx_A, idx_B)
    print(
        f"Selected alpha={selected_alpha:.3f} | "
        f"A_move_max={m_sel['A_move_max_um']:.2f} um | "
        f"A_move_final={m_sel['A_move_final_um']:.2f} um | "
        f"d_AB_final={m_sel['d_AB_final_um']:.2f} um"
    )

    tune_json = OUT_DIR / "replica_cshape_alpha_tuning.json"
    with open(tune_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "auto_tune_alpha": bool(AUTO_TUNE_ALPHA),
                "a_move_threshold_um": float(A_MOVE_THRESHOLD_UM),
                "selected_alpha": float(selected_alpha),
                "cshape_translate": bool(CSHAPE_TRANSLATE),
                "cshape_mobility_scale": float(CSHAPE_MOBILITY_SCALE),
                "rows": tune_log,
            },
            f,
            indent=2,
        )
    print(f"Saved alpha tuning log: {tune_json}")

    # Frame counts
    n_phys_frames = min(len(res_v.times_s), len(res_c.times_s))
    n_frames_nominal = (n_phys_frames - 1) * DISPLAY_SUBFRAMES + 1
    n_frames = min(n_frames_nominal, MAX_OUTPUT_FRAMES) if MAX_OUTPUT_FRAMES > 0 else n_frames_nominal

    # Grid metrics
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

        label_v = res_v.phase_labels[i0]
        label_c = res_c.phase_labels[i0]
        t_ms = float(_lerp(res_v.times_s[i0], res_v.times_s[i1], tau) * 1e3)

        alpha_v = float(_lerp(res_v.alphas[i0], res_v.alphas[i1], tau))
        beta_v = float(_lerp(res_v.betas_sw[i0], res_v.betas_sw[i1], tau))
        ctr_v = _lerp(res_v.centers[i0], res_v.centers[i1], tau)

        alpha_c = float(_lerp(res_c.alphas[i0], res_c.alphas[i1], tau))
        beta_c = float(_lerp(res_c.betas_sw[i0], res_c.betas_sw[i1], tau))
        ctr_c = _lerp(res_c.centers[i0], res_c.centers[i1], tau)

        p_tot_v_full = _total_pressure(
            p_sw_full,
            gen_vortex_full.get_field,
            VORTEX_PSI,
            alpha_v,
            beta_v,
            ctr_v,
        )
        p_tot_c_full = _total_pressure(
            p_sw_full,
            gen_cshape_full.get_field,
            CSHAPE_PSI,
            alpha_c,
            beta_c,
            ctr_c,
        )

        U_v_full, _, _ = _gorkov_fields(p_tot_v_full, dx_full, dy_full)
        U_c_full, _, _ = _gorkov_fields(p_tot_c_full, dx_full, dy_full)

        # ROI panels are direct crops from full-domain fields.
        U_v_roi = U_v_full[np.ix_(iy_roi, ix_roi)]
        U_c_roi = U_c_full[np.ix_(iy_roi, ix_roi)]

        vmin, vmax = _auto_limits(U_v_full, U_c_full)

        cur_v = _lerp(res_v.trajectories[i0], res_v.trajectories[i1], tau) * 1e3
        cur_c = _lerp(res_c.trajectories[i0], res_c.trajectories[i1], tau) * 1e3

        if i1 == i0:
            path_v = res_v.centers[: i0 + 1] * 1e3
            path_c = res_c.centers[: i0 + 1] * 1e3
        else:
            path_v = np.vstack([res_v.centers[: i0 + 1], ctr_v]) * 1e3
            path_c = np.vstack([res_c.centers[: i0 + 1], ctr_c]) * 1e3

        fig, axes = plt.subplots(2, 2, figsize=(11.2, 10.4))
        (ax_v_roi, ax_c_roi), (ax_v_full, ax_c_full) = axes

        _draw_panel(
            ax_v_roi,
            U_v_roi,
            vmin,
            vmax,
            extent_roi,
            x_roi_mm,
            y_roi_mm,
            traps_mm,
            cur_v,
            path_v,
            idx_A,
            idx_B,
            neigh_idx,
            f"Vortex - U_Gorkov (ROI crop of full)  ({label_v})",
        )
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
            ax_v_full,
            U_v_full,
            vmin,
            vmax,
            extent_full,
            x_full_mm,
            y_full_mm,
            traps_mm,
            cur_v,
            path_v,
            idx_A,
            idx_B,
            neigh_idx,
            f"Vortex - U_Gorkov (full domain)  ({label_v})",
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

        fig.suptitle(
            f"t = {t_ms:.0f} ms | frame {fi + 1}/{n_frames} | "
            f"alpha={selected_alpha:.3f} | translated={CSHAPE_TRANSLATE}",
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

    manifest_json = OUT_DIR / "replica_cshape_method_manifest.json"
    with open(manifest_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "method_id": METHOD_ID,
                "script": "scripts/deliverables/transport_side_by_side_replica_cshape.py",
                "output_dir": str(OUT_DIR.relative_to(PROJECT_ROOT)),
                "outputs": {
                    "gif": str(out_path.relative_to(PROJECT_ROOT)),
                    "field_package": str(field_npz.relative_to(PROJECT_ROOT)),
                    "alpha_tuning": str(tune_json.relative_to(PROJECT_ROOT)),
                },
                "controls": {
                    "auto_tune_alpha": bool(AUTO_TUNE_ALPHA),
                    "selected_alpha": float(selected_alpha),
                    "a_move_threshold_um": float(A_MOVE_THRESHOLD_UM),
                    "cshape_translate": bool(CSHAPE_TRANSLATE),
                    "cshape_mobility_scale": float(CSHAPE_MOBILITY_SCALE),
                    "const_sw": bool(CSHAPE_CONST_SW),
                    "sw_scale": float(SW_SCALE),
                    "display_subframes": int(DISPLAY_SUBFRAMES),
                    "n_frames": int(N_FRAMES),
                    "gif_duration_ms": int(GIF_DURATION_MS),
                },
                "replica_lens": {
                    "frequency_hz": float(REPLICA_F_HZ),
                    "c_water": float(REPLICA_C_WATER),
                    "c_lens": float(REPLICA_C_LENS),
                    "transducer_diameter_mm": float(REPLICA_D_MM),
                    "focal_distance_mm": float(REPLICA_FOCAL_MM),
                    "n_grid": int(REPLICA_N_GRID),
                    "arc_radius_mm": float(REPLICA_ARC_RADIUS_MM),
                    "arc_width_mm": float(REPLICA_ARC_WIDTH_MM),
                    "gap_angle_deg": float(REPLICA_GAP_ANGLE_DEG),
                    "gap_direction_deg": float(gap_direction_deg),
                    "h_base_mm": float(REPLICA_H_BASE_MM),
                    "n_iter": int(REPLICA_N_ITER),
                },
                "metrics": {
                    "A_move_max_um": float(m_sel["A_move_max_um"]),
                    "A_move_final_um": float(m_sel["A_move_final_um"]),
                    "d_AB_final_um": float(m_sel["d_AB_final_um"]),
                },
                "invariants": [
                    "Full-domain replica C-shape field is generated from lens thickness/phase pipeline.",
                    "Top-row ROI views are crops from the same full-domain Gor'kov arrays.",
                    "No separate ROI-only C-shape solve is used in this run.",
                ],
            },
            f,
            indent=2,
        )

    print(f"\nSaved GIF: {out_path}")
    print(f"Saved method manifest: {manifest_json}")
    print(f"Output dir: {OUT_DIR}")


if __name__ == "__main__":
    main()
