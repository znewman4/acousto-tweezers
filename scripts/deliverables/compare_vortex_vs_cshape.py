#!/usr/bin/env python3
"""
Corrected unified vortex-vs-C-shape comparison workflow.

This script performs a correction/refinement pass of the existing unified
comparison, with these key properties:

1. Uses the single-pass overdamped transport engine.
2. Recalibrates vortex controls (alpha, psi) for THIS overdamped workflow.
3. Uses a physically meaningful SW-only release/settle phase.
4. Produces clearer, equal-aspect visuals with release interpretation.
5. Reports split metrics (transport, B stability, neighbour stability,
   release success), with scalar score only as secondary context.

Outputs
-------
results/deliverables/compare_vortex_vs_cshape/
    vortex/
        transport.gif
        release.gif
        trajectories.png
        metrics.json
    cshape/
        transport.gif
        release.gif
        trajectories.png
        metrics.json
    comparison/
        displacement_vs_time.png
        neighbour_disturbance.png
        success_metric_table.csv
        summary.json
        diagnostic_notes.md
        release_overlay_sw_wells.png
        vortex_recalibration.csv
        vortex_recalibration.json
"""
from __future__ import annotations

import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.interpolate import RegularGridInterpolator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lib.particle_dynamics_utils import (
    CAPTURE_RADIUS,
    NEIGHBOUR_TOL,
    SCALE,
    TransportResult,
    bilinear_forces_at_positions,
    compute_metrics,
    expand_schedule,
    gorkov_normalised,
    precompute_bilinear_basis,
    run_transport,
)
from scripts.lib.perturbation_cshape import CShapePerturbation
from scripts.lib.perturbation_vortex import VortexPerturbation

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
OUT_ROOT = PROJECT_ROOT / "results" / "deliverables" / "compare_vortex_vs_cshape"
OUT_V = OUT_ROOT / "vortex"
OUT_C = OUT_ROOT / "cshape"
OUT_CMP = OUT_ROOT / "comparison"

VORTEX_NPZ = (
    PROJECT_ROOT
    / "results"
    / "deliverables"
    / "vortex_stage_transport"
    / "transport"
    / "transport_case_for_gif.npz"
)
VORTEX_GEOM_JSON = (
    PROJECT_ROOT
    / "results"
    / "deliverables"
    / "vortex_stage_transport"
    / "calibration"
    / "selected_vortex_summary.json"
)
OVERLAY_NPZ = (
    PROJECT_ROOT
    / "results"
    / "c_shape_lens_15mm_overlay_study_20260310_170620"
    / "npz"
    / "roi_fields.npz"
)

# ------------------------------------------------------------------
# Controls and schedule
# ------------------------------------------------------------------
# C-shape control kept near known working region, slightly less aggressive SW dip.
CSHAPE_ALPHA = 4.50
CSHAPE_PSI = 1.5 * np.pi
CSHAPE_BSW_MIN = 0.42

# Shared schedule structure for both methods.
SW_SCALE = 1.0
T_SW_ONLY_START_MS = 120.0
T_RAMP_ON_MS = 220.0
T_ACTIVE_MS = 260.0
T_RAMP_OFF_MS = 220.0
T_SETTLE_MS = 800.0

DT = 1.0e-4
N_FRAMES = 320

# Calibration settings (overdamped engine, same schedule structure).
VORTEX_PSI_CANDIDATES = [0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi]
VORTEX_ALPHA_CANDIDATES = [0.6, 0.9, 1.2, 1.5, 1.8, 2.1]
CALIB_FRAMES = 160
CALIB_KEYFRAMES = 10

# GIF styling
GIF_DURATION_MS = 55
CMAP_PRESSURE = "viridis"
CMAP_GORKOV = "magma"

COL_A = "#e74c3c"
COL_B = "#3498db"
COL_NEIGH = "#95a5a6"
COL_TRAP = "#111111"
COL_CTR = "#f39c12"
COL_HOME = "#2ecc71"


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------
@dataclass
class SWForceInterpolators:
    fx: RegularGridInterpolator
    fy: RegularGridInterpolator


# ------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------
def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _mm(x: np.ndarray) -> np.ndarray:
    return np.asarray(x) * 1e3


def _um(x: np.ndarray) -> np.ndarray:
    return np.asarray(x) * 1e6


def _phase_lower(labels: List[str]) -> np.ndarray:
    return np.array([str(s).lower().strip() for s in labels])


def _nearest_trap_indices(points_xy: np.ndarray, traps_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    d = np.linalg.norm(points_xy[:, None, :] - traps_xy[None, :, :], axis=2)
    idx = np.argmin(d, axis=1)
    dmin = d[np.arange(len(points_xy)), idx]
    return idx.astype(int), dmin.astype(float)


# ------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------
def load_data() -> Dict[str, Any]:
    ov = np.load(OVERLAY_NPZ)
    xg = ov["xg"].astype(float)
    yg = ov["yg"].astype(float)
    p_sw = ov["p_sw"].astype(complex)
    p_lens = ov["p_lens_roi"].astype(complex)
    traps = ov["traps_m"].astype(float)
    idx_A = int(ov["idx_A"])
    idx_B = int(ov["idx_B"])

    vd = np.load(VORTEX_NPZ)
    xg_v = vd["xg"].astype(float)
    yg_v = vd["yg"].astype(float)
    p_v_full = vd["p_vortex_centered"].astype(complex)

    neighbours = np.array(sorted(set(range(len(traps))) - {idx_A, idx_B}), dtype=int)

    geom_meta: Dict[str, Any] = {}
    if VORTEX_GEOM_JSON.exists():
        try:
            geom_meta = json.loads(VORTEX_GEOM_JSON.read_text())
        except Exception:
            geom_meta = {}

    return {
        "xg": xg,
        "yg": yg,
        "p_sw": p_sw,
        "p_lens": p_lens,
        "traps_m": traps,
        "idx_A": idx_A,
        "idx_B": idx_B,
        "neighbour_idx": neighbours,
        "xg_v": xg_v,
        "yg_v": yg_v,
        "p_vortex_full": p_v_full,
        "vortex_geom_meta": geom_meta,
    }


# ------------------------------------------------------------------
# Schedules
# ------------------------------------------------------------------
def build_vortex_phases(A_xy: np.ndarray, B_xy: np.ndarray, alpha: float) -> List[Dict[str, Any]]:
    A = np.asarray(A_xy, dtype=float)
    B = np.asarray(B_xy, dtype=float)
    return [
        {
            "duration_ms": T_SW_ONLY_START_MS,
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
            "alpha_end": alpha,
            "bsw_start": SW_SCALE,
            "bsw_end": SW_SCALE,
            "ctr_start": A.copy(),
            "ctr_end": A.copy(),
            "label": "ramp on",
        },
        {
            "duration_ms": T_ACTIVE_MS,
            "alpha_start": alpha,
            "alpha_end": alpha,
            "bsw_start": SW_SCALE,
            "bsw_end": SW_SCALE,
            "ctr_start": A.copy(),
            "ctr_end": B.copy(),
            "label": "translate",
        },
        {
            "duration_ms": T_RAMP_OFF_MS,
            "alpha_start": alpha,
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


def build_cshape_phases(A_xy: np.ndarray, B_xy: np.ndarray) -> List[Dict[str, Any]]:
    A = np.asarray(A_xy, dtype=float)
    B = np.asarray(B_xy, dtype=float)
    mid = 0.5 * (A + B)
    return [
        {
            "duration_ms": T_SW_ONLY_START_MS,
            "alpha_start": 0.0,
            "alpha_end": 0.0,
            "bsw_start": SW_SCALE,
            "bsw_end": SW_SCALE,
            "ctr_start": mid.copy(),
            "ctr_end": mid.copy(),
            "label": "SW only",
        },
        {
            "duration_ms": T_RAMP_ON_MS,
            "alpha_start": 0.0,
            "alpha_end": CSHAPE_ALPHA,
            "bsw_start": SW_SCALE,
            "bsw_end": CSHAPE_BSW_MIN,
            "ctr_start": mid.copy(),
            "ctr_end": mid.copy(),
            "label": "ramp on",
        },
        {
            "duration_ms": T_ACTIVE_MS,
            "alpha_start": CSHAPE_ALPHA,
            "alpha_end": CSHAPE_ALPHA,
            "bsw_start": CSHAPE_BSW_MIN,
            "bsw_end": CSHAPE_BSW_MIN,
            "ctr_start": mid.copy(),
            "ctr_end": mid.copy(),
            "label": "hold",
        },
        {
            "duration_ms": T_RAMP_OFF_MS,
            "alpha_start": CSHAPE_ALPHA,
            "alpha_end": 0.0,
            "bsw_start": CSHAPE_BSW_MIN,
            "bsw_end": SW_SCALE,
            "ctr_start": mid.copy(),
            "ctr_end": mid.copy(),
            "label": "ramp off",
        },
        {
            "duration_ms": T_SETTLE_MS,
            "alpha_start": 0.0,
            "alpha_end": 0.0,
            "bsw_start": SW_SCALE,
            "bsw_end": SW_SCALE,
            "ctr_start": mid.copy(),
            "ctr_end": mid.copy(),
            "label": "SW release",
        },
    ]


# ------------------------------------------------------------------
# Forces and diagnostics
# ------------------------------------------------------------------
def make_sw_force_interpolators(p_sw: np.ndarray, xg: np.ndarray, yg: np.ndarray) -> SWForceInterpolators:
    dx = float(xg[1] - xg[0])
    dy = float(yg[1] - yg[0])
    _, fx_sw, fy_sw = gorkov_normalised(p_sw, dx, dy)
    i_fx = RegularGridInterpolator((yg, xg), fx_sw, bounds_error=False, fill_value=0.0)
    i_fy = RegularGridInterpolator((yg, xg), fy_sw, bounds_error=False, fill_value=0.0)
    return SWForceInterpolators(fx=i_fx, fy=i_fy)


def settle_force_stats(
    result: TransportResult,
    idx_A: int,
    idx_B: int,
    sw_interp: SWForceInterpolators,
) -> Dict[str, Any]:
    labels = _phase_lower(result.phase_labels)
    mask = labels == "sw release"
    ii = np.where(mask)[0]

    if len(ii) == 0:
        return {
            "n_frames": 0,
            "A_force_mag": {"min": 0.0, "mean": 0.0, "max": 0.0},
            "B_force_mag": {"min": 0.0, "mean": 0.0, "max": 0.0},
        }

    posA = result.trajectories[ii, idx_A, :]
    posB = result.trajectories[ii, idx_B, :]

    ptsA = np.column_stack([posA[:, 1], posA[:, 0]])
    ptsB = np.column_stack([posB[:, 1], posB[:, 0]])

    fAx = sw_interp.fx(ptsA)
    fAy = sw_interp.fy(ptsA)
    fBx = sw_interp.fx(ptsB)
    fBy = sw_interp.fy(ptsB)

    fA = np.sqrt(fAx**2 + fAy**2)
    fB = np.sqrt(fBx**2 + fBy**2)

    return {
        "n_frames": int(len(ii)),
        "A_force_mag": {
            "min": float(np.min(fA)),
            "mean": float(np.mean(fA)),
            "max": float(np.max(fA)),
        },
        "B_force_mag": {
            "min": float(np.min(fB)),
            "mean": float(np.mean(fB)),
            "max": float(np.max(fB)),
        },
    }


def schedule_stats(phases: List[Dict[str, Any]], dt: float) -> Dict[str, Any]:
    _, _, _, _, labels = expand_schedule(phases, dt)
    labels_l = _phase_lower(labels)
    n_total = len(labels)
    n_settle = int(np.sum(labels_l == "sw release"))
    return {
        "dt_ms": dt * 1e3,
        "n_steps_total": int(n_total),
        "n_steps_settle": int(n_settle),
        "settle_ms": float(n_settle * dt * 1e3),
        "timing_ms": {
            "sw_only_start": T_SW_ONLY_START_MS,
            "ramp_on": T_RAMP_ON_MS,
            "active": T_ACTIVE_MS,
            "ramp_off": T_RAMP_OFF_MS,
            "settle": T_SETTLE_MS,
        },
    }


def compute_vortex_force_context(
    p_sw: np.ndarray,
    xg: np.ndarray,
    yg: np.ndarray,
    A_xy: np.ndarray,
    phases_v: List[Dict[str, Any]],
    gen_vortex: VortexPerturbation,
    alpha: float,
    psi: float,
) -> Dict[str, Any]:
    dx = float(xg[1] - xg[0])
    dy = float(yg[1] - yg[0])

    times, _, _, centers, labels = expand_schedule(phases_v, DT)
    labels_l = _phase_lower(labels)
    idx_tr = np.where(labels_l == "translate")[0]

    if len(idx_tr) == 0:
        return {}

    samples = {
        "start": int(idx_tr[0]),
        "mid": int(idx_tr[len(idx_tr) // 2]),
        "end": int(idx_tr[-1]),
    }

    out: Dict[str, Any] = {}
    posA = np.asarray(A_xy, dtype=float)[None, :]

    for key, k in samples.items():
        ctr = centers[k]
        p_eff = np.exp(1j * psi) * gen_vortex.get_field(ctr)
        basis = precompute_bilinear_basis(p_sw, p_eff, dx, dy)

        fx_sw, fy_sw = bilinear_forces_at_positions(basis, SW_SCALE, 0.0, posA, xg, yg)
        fx_vo, fy_vo = bilinear_forces_at_positions(basis, 0.0, alpha, posA, xg, yg)

        F_sw = np.array([float(fx_sw[0]), float(fy_sw[0])])
        F_vo = np.array([float(fx_vo[0]), float(fy_vo[0])])

        n_sw = float(np.linalg.norm(F_sw))
        n_vo = float(np.linalg.norm(F_vo))

        out[key] = {
            "t_ms": float(times[k] * 1e3),
            "center_mm": [float(ctr[0] * 1e3), float(ctr[1] * 1e3)],
            "F_sw_mag": n_sw,
            "F_vortex_only_mag": n_vo,
            "ratio_vortex_to_sw": float(n_vo / max(n_sw, 1e-30)),
        }

    return out


# ------------------------------------------------------------------
# Metrics (split sections)
# ------------------------------------------------------------------
def build_split_metrics(
    method: str,
    result: TransportResult,
    phases: List[Dict[str, Any]],
    p_sw: np.ndarray,
    xg: np.ndarray,
    yg: np.ndarray,
    traps_m: np.ndarray,
    idx_A: int,
    idx_B: int,
    neighbour_idx: np.ndarray,
    controls: Dict[str, Any],
    sw_interp: SWForceInterpolators,
    vortex_force_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    A0 = traps_m[idx_A]
    B0 = traps_m[idx_B]
    Af = result.trajectories[-1, idx_A]
    Bf = result.trajectories[-1, idx_B]

    d_ab0 = float(np.linalg.norm(B0 - A0))
    e_ab = (B0 - A0) / max(d_ab0, 1e-30)

    A_moved = float(np.linalg.norm(Af - A0))
    B_moved = float(np.linalg.norm(Bf - B0))
    A_to_B_end = float(np.linalg.norm(Af - B0))
    d_AB_final = float(np.linalg.norm(Af - Bf))
    progress = float(np.dot(Af - A0, e_ab))

    neigh = np.asarray(neighbour_idx, dtype=int)
    neigh_disp = np.linalg.norm(result.trajectories[-1, neigh, :] - traps_m[neigh, :], axis=1)

    nearest_idx_all, nearest_dist_all = _nearest_trap_indices(result.trajectories[-1], traps_m)

    neigh_home_idx = neigh
    neigh_nearest = nearest_idx_all[neigh]
    neigh_home_recovered = neigh_nearest == neigh_home_idx

    settle = settle_force_stats(result, idx_A, idx_B, sw_interp)
    sched = schedule_stats(phases, DT)

    legacy = compute_metrics(result, idx_A, idx_B, neighbour_idx)

    capture_ok = bool(A_to_B_end < CAPTURE_RADIUS)
    b_stable = bool(B_moved < NEIGHBOUR_TOL)
    neigh_stable_mask = neigh_disp < NEIGHBOUR_TOL

    transport_section = {
        "A_capture_success": capture_ok,
        "A_to_B_target_um": round(A_to_B_end * 1e6, 3),
        "A_to_B_pair_end_um": round(d_AB_final * 1e6, 3),
        "A_moved_um": round(A_moved * 1e6, 3),
        "progress_toward_B_um": round(progress * 1e6, 3),
        "progress_fraction_of_AB": round(progress / max(d_ab0, 1e-30), 4),
    }

    b_section = {
        "B_stable": b_stable,
        "B_moved_um": round(B_moved * 1e6, 3),
        "B_home_distance_um": round(float(np.linalg.norm(Bf - B0)) * 1e6, 3),
        "B_final_nearest_trap_idx": int(nearest_idx_all[idx_B]),
        "B_final_nearest_trap_distance_um": round(float(nearest_dist_all[idx_B]) * 1e6, 3),
    }

    neigh_section = {
        "neighbour_count": int(len(neigh)),
        "stable_count": int(np.sum(neigh_stable_mask)),
        "stable_fraction": round(float(np.mean(neigh_stable_mask)) if len(neigh) > 0 else 1.0, 4),
        "all_neighbours_stable": bool(np.all(neigh_stable_mask)) if len(neigh) > 0 else True,
        "max_displacement_um": round(float(np.max(neigh_disp) * 1e6) if len(neigh) > 0 else 0.0, 3),
        "rms_displacement_um": round(float(np.sqrt(np.mean(neigh_disp**2)) * 1e6) if len(neigh) > 0 else 0.0, 3),
        "mean_displacement_um": round(float(np.mean(neigh_disp) * 1e6) if len(neigh) > 0 else 0.0, 3),
    }

    release_section = {
        "A_final_nearest_trap_idx": int(nearest_idx_all[idx_A]),
        "A_final_nearest_trap_distance_um": round(float(nearest_dist_all[idx_A]) * 1e6, 3),
        "A_in_target_trap_region": bool(nearest_idx_all[idx_A] == idx_B),
        "B_final_nearest_trap_idx": int(nearest_idx_all[idx_B]),
        "B_in_home_trap_region": bool(nearest_idx_all[idx_B] == idx_B),
        "neighbour_home_recovered_count": int(np.sum(neigh_home_recovered)),
        "neighbour_home_recovered_fraction": round(float(np.mean(neigh_home_recovered)) if len(neigh) > 0 else 1.0, 4),
    }

    # Secondary scalar context only (not primary conclusion).
    secondary_score = (
        transport_section["progress_fraction_of_AB"]
        - 0.5 * (b_section["B_moved_um"] / max(NEIGHBOUR_TOL * 1e6, 1e-30))
        - 0.5 * (neigh_section["rms_displacement_um"] / max(NEIGHBOUR_TOL * 1e6, 1e-30))
        + (0.4 if transport_section["A_capture_success"] else 0.0)
    )

    out: Dict[str, Any] = {
        "method": method,
        "controls": controls,
        "schedule": sched,
        "transport_effectiveness": transport_section,
        "B_stability": b_section,
        "neighbour_stability": neigh_section,
        "release_success": release_section,
        "settle_diagnostics": settle,
        "secondary_scalar_score": round(float(secondary_score), 4),
        "legacy_metrics": legacy,
    }

    if vortex_force_context is not None:
        out["vortex_force_context"] = vortex_force_context

    return out


# ------------------------------------------------------------------
# Vortex recalibration (overdamped)
# ------------------------------------------------------------------
def calibrate_vortex_controls(
    p_sw: np.ndarray,
    xg: np.ndarray,
    yg: np.ndarray,
    traps_m: np.ndarray,
    idx_A: int,
    idx_B: int,
    neighbour_idx: np.ndarray,
    gen_vortex: VortexPerturbation,
    sw_interp: SWForceInterpolators,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    A_xy = traps_m[idx_A]
    B_xy = traps_m[idx_B]

    rows: List[Dict[str, Any]] = []

    print("\nRecalibrating VORTEX controls for overdamped engine...")
    print(f"  Candidates: {len(VORTEX_ALPHA_CANDIDATES)} alpha × {len(VORTEX_PSI_CANDIDATES)} psi")

    for psi in VORTEX_PSI_CANDIDATES:
        for alpha in VORTEX_ALPHA_CANDIDATES:
            phases = build_vortex_phases(A_xy, B_xy, alpha=alpha)
            res = run_transport(
                p_sw=p_sw,
                perturbation_fn=gen_vortex.get_field,
                phases=phases,
                psi=psi,
                initial_positions=traps_m.copy(),
                xg=xg,
                yg=yg,
                idx_A=idx_A,
                idx_B=idx_B,
                dt=DT,
                n_frames=CALIB_FRAMES,
                n_keyframes=CALIB_KEYFRAMES,
            )

            force_ctx = compute_vortex_force_context(
                p_sw=p_sw,
                xg=xg,
                yg=yg,
                A_xy=A_xy,
                phases_v=phases,
                gen_vortex=gen_vortex,
                alpha=alpha,
                psi=psi,
            )
            ratio_mid = float(force_ctx.get("mid", {}).get("ratio_vortex_to_sw", 0.0))

            m = build_split_metrics(
                method="vortex",
                result=res,
                phases=phases,
                p_sw=p_sw,
                xg=xg,
                yg=yg,
                traps_m=traps_m,
                idx_A=idx_A,
                idx_B=idx_B,
                neighbour_idx=neighbour_idx,
                controls={"alpha": alpha, "psi_over_pi": psi / np.pi, "beta_sw_min": SW_SCALE},
                sw_interp=sw_interp,
                vortex_force_context=force_ctx,
            )

            row = {
                "alpha": alpha,
                "psi": psi,
                "psi_over_pi": psi / np.pi,
                "A_capture_success": m["transport_effectiveness"]["A_capture_success"],
                "progress_fraction_of_AB": m["transport_effectiveness"]["progress_fraction_of_AB"],
                "A_to_B_target_um": m["transport_effectiveness"]["A_to_B_target_um"],
                "B_stable": m["B_stability"]["B_stable"],
                "B_moved_um": m["B_stability"]["B_moved_um"],
                "neigh_all_stable": m["neighbour_stability"]["all_neighbours_stable"],
                "neigh_stable_fraction": m["neighbour_stability"]["stable_fraction"],
                "neigh_rms_um": m["neighbour_stability"]["rms_displacement_um"],
                "neigh_max_um": m["neighbour_stability"]["max_displacement_um"],
                "A_target_region": m["release_success"]["A_in_target_trap_region"],
                "B_home_region": m["release_success"]["B_in_home_trap_region"],
                "force_ratio_mid": ratio_mid,
                "secondary_scalar_score": m["secondary_scalar_score"],
            }
            rows.append(row)

            print(
                f"    alpha={alpha:.2f}, psi={psi/np.pi:.1f}pi | "
                f"progress={row['progress_fraction_of_AB']:.3f}, "
                f"B_stable={row['B_stable']}, neigh_stable={row['neigh_all_stable']}, "
                f"ratio_mid={ratio_mid:.3f}"
            )

    # Multi-criteria ranking (lexicographic; not single scalar objective).
    ranked = sorted(
        rows,
        key=lambda r: (
            int(r["A_capture_success"]),
            int(r["B_stable"]),
            int(r["neigh_all_stable"]),
            float(r["progress_fraction_of_AB"]),
            float(r["force_ratio_mid"]),
            -float(r["B_moved_um"]),
            -float(r["neigh_rms_um"]),
        ),
        reverse=True,
    )
    selected = ranked[0]

    sel = {
        "alpha": float(selected["alpha"]),
        "psi": float(selected["psi"]),
        "psi_over_pi": float(selected["psi_over_pi"]),
        "selection_basis": {
            "priority_order": [
                "A_capture_success",
                "B_stable",
                "neigh_all_stable",
                "progress_fraction_of_AB",
                "force_ratio_mid",
                "minimize_B_moved",
                "minimize_neigh_rms",
            ],
            "selected_row": selected,
        },
    }

    return sel, ranked


# ------------------------------------------------------------------
# Rendering
# ------------------------------------------------------------------
def _phase_color(label: str) -> str:
    m = {
        "sw only": "#2e7d32",
        "ramp on": "#ef6c00",
        "translate": "#6a1b9a",
        "hold": "#6a1b9a",
        "ramp off": "#1565c0",
        "sw release": "#2e7d32",
    }
    return m.get(label.lower().strip(), "#424242")


def _pressure_frame(
    p_sw: np.ndarray,
    perturb_fn,
    psi: float,
    alpha_f: float,
    bsw_f: float,
    center_f: np.ndarray,
) -> np.ndarray:
    p_pert = perturb_fn(center_f)
    p_total = bsw_f * p_sw + alpha_f * np.exp(1j * psi) * p_pert
    return np.abs(p_total)


def _plot_home_to_current(
    ax,
    homes_mm: np.ndarray,
    current_mm: np.ndarray,
    idxs: np.ndarray,
    color: str,
    lw: float,
    alpha: float,
) -> None:
    for i in idxs:
        x0, y0 = homes_mm[i]
        x1, y1 = current_mm[i]
        ax.plot([x0, x1], [y0, y1], color=color, lw=lw, alpha=alpha, zorder=3)


def _gif_legend_handles() -> List[Any]:
    return [
        mpatches.Patch(color="none", label="Markers:"),
        mlines.Line2D([], [], color=COL_A, marker="o", linestyle="None", label="A current"),
        mlines.Line2D([], [], color=COL_B, marker="o", linestyle="None", label="B current"),
        mlines.Line2D([], [], color=COL_NEIGH, marker="o", linestyle="None", label="Neighbour current"),
        mlines.Line2D([], [], color=COL_HOME, marker="x", linestyle="None", label="Home trap"),
        mlines.Line2D([], [], color=COL_CTR, marker="+", linestyle="-", label="Perturb centre/path"),
    ]


def render_transport_gif(
    result: TransportResult,
    p_sw: np.ndarray,
    perturb_fn,
    psi: float,
    xg: np.ndarray,
    yg: np.ndarray,
    traps_m: np.ndarray,
    idx_A: int,
    idx_B: int,
    neighbour_idx: np.ndarray,
    out_path: Path,
    title: str,
    duration_ms: int = GIF_DURATION_MS,
) -> None:
    n_frames = len(result.times_s)
    frames: List[Image.Image] = []

    sample_idx = np.linspace(0, n_frames - 1, min(12, n_frames), dtype=int)
    p99 = []
    for fi in sample_idx:
        p_abs = _pressure_frame(
            p_sw,
            perturb_fn,
            psi,
            float(result.alphas[fi]),
            float(result.betas_sw[fi]),
            result.centers[fi],
        )
        p99.append(float(np.percentile(p_abs, 99.0)))
    vmax = float(np.max(p99))

    x_mm = _mm(xg)
    y_mm = _mm(yg)
    extent = [x_mm[0], x_mm[-1], y_mm[0], y_mm[-1]]
    traps_mm = _mm(traps_m)
    traj_mm = _mm(result.trajectories)

    legend_handles = _gif_legend_handles()

    for fi in range(n_frames):
        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11.2, 5.4))

        p_abs = _pressure_frame(
            p_sw,
            perturb_fn,
            psi,
            float(result.alphas[fi]),
            float(result.betas_sw[fi]),
            result.centers[fi],
        )
        ax_l.imshow(
            p_abs,
            origin="lower",
            extent=extent,
            cmap=CMAP_PRESSURE,
            vmin=0.0,
            vmax=vmax,
            aspect="equal",
        )

        current = traj_mm[fi]
        centres_path = _mm(result.centers[: fi + 1])

        # Home markers and trap markers.
        ax_l.scatter(traps_mm[:, 0], traps_mm[:, 1], marker="x", c=COL_HOME, s=30, linewidths=0.9, zorder=4)
        ax_l.scatter(traps_mm[:, 0], traps_mm[:, 1], marker="o", facecolors="none", edgecolors="#222222", s=18, zorder=4)

        # Home-to-current vectors.
        _plot_home_to_current(ax_l, traps_mm, current, neighbour_idx, COL_NEIGH, lw=0.8, alpha=0.7)
        _plot_home_to_current(ax_l, traps_mm, current, np.array([idx_A]), COL_A, lw=1.2, alpha=0.95)
        _plot_home_to_current(ax_l, traps_mm, current, np.array([idx_B]), COL_B, lw=1.2, alpha=0.95)

        # Current particles.
        ax_l.scatter(current[neighbour_idx, 0], current[neighbour_idx, 1], c=COL_NEIGH, s=30, zorder=6)
        ax_l.scatter([current[idx_B, 0]], [current[idx_B, 1]], c=COL_B, s=44, zorder=7)
        ax_l.scatter([current[idx_A, 0]], [current[idx_A, 1]], c=COL_A, s=44, zorder=7)

        # Perturbation center and path.
        ax_l.plot(centres_path[:, 0], centres_path[:, 1], color=COL_CTR, lw=1.0, alpha=0.8, zorder=5)
        ax_l.plot(centres_path[-1, 0], centres_path[-1, 1], marker="+", color=COL_CTR, ms=10, mew=1.5, zorder=8)

        ax_l.set_title(f"|p_total| and traps ({result.phase_labels[fi]})", fontsize=10)
        ax_l.set_xlabel("x [mm]")
        ax_l.set_ylabel("y [mm]")
        ax_l.set_aspect("equal", adjustable="box")

        # Right panel: trajectory context.
        ax_r.set_facecolor("#11141b")
        trail = max(0, fi - 50)

        ax_r.scatter(traps_mm[:, 0], traps_mm[:, 1], marker="x", c=COL_HOME, s=28, linewidths=0.8, alpha=0.9, zorder=2)
        _plot_home_to_current(ax_r, traps_mm, current, neighbour_idx, COL_NEIGH, lw=0.9, alpha=0.55)
        _plot_home_to_current(ax_r, traps_mm, current, np.array([idx_A]), COL_A, lw=1.3, alpha=0.95)
        _plot_home_to_current(ax_r, traps_mm, current, np.array([idx_B]), COL_B, lw=1.3, alpha=0.95)

        for j in neighbour_idx:
            ax_r.plot(traj_mm[trail : fi + 1, j, 0], traj_mm[trail : fi + 1, j, 1], color=COL_NEIGH, lw=0.8, alpha=0.7)
        ax_r.plot(traj_mm[trail : fi + 1, idx_B, 0], traj_mm[trail : fi + 1, idx_B, 1], color=COL_B, lw=1.4)
        ax_r.plot(traj_mm[trail : fi + 1, idx_A, 0], traj_mm[trail : fi + 1, idx_A, 1], color=COL_A, lw=1.4)

        ax_r.scatter(current[neighbour_idx, 0], current[neighbour_idx, 1], c=COL_NEIGH, s=28, zorder=5)
        ax_r.scatter([current[idx_B, 0]], [current[idx_B, 1]], c=COL_B, s=46, zorder=6)
        ax_r.scatter([current[idx_A, 0]], [current[idx_A, 1]], c=COL_A, s=46, zorder=6)

        ax_r.plot(centres_path[:, 0], centres_path[:, 1], color=COL_CTR, lw=1.0, alpha=0.9)
        ax_r.plot(centres_path[-1, 0], centres_path[-1, 1], marker="+", color=COL_CTR, ms=10, mew=1.5)

        ax_r.set_xlim(x_mm[0], x_mm[-1])
        ax_r.set_ylim(y_mm[0], y_mm[-1])
        ax_r.set_aspect("equal", adjustable="box")
        ax_r.set_xlabel("x [mm]", color="white")
        ax_r.set_ylabel("y [mm]", color="white")
        ax_r.tick_params(colors="white", labelsize=8)
        ax_r.set_title(f"Particle map t={result.times_s[fi]*1e3:.0f} ms", color="white", fontsize=10)

        fig.suptitle(
            f"{title} | frame {fi+1}/{n_frames}",
            fontsize=11,
            color=_phase_color(result.phase_labels[fi]),
        )
        ax_r.legend(handles=legend_handles, fontsize=7, loc="upper right", framealpha=0.35)
        fig.tight_layout(rect=[0, 0, 1, 0.94])

        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        img = Image.fromarray(buf.reshape(h, w, 4)).convert("RGB")
        frames.append(img)
        plt.close(fig)

    frames[0].save(
        str(out_path),
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    print(f"  GIF saved -> {out_path.name} ({n_frames} frames)")


def render_release_gif(
    result: TransportResult,
    p_sw: np.ndarray,
    xg: np.ndarray,
    yg: np.ndarray,
    traps_m: np.ndarray,
    idx_A: int,
    idx_B: int,
    neighbour_idx: np.ndarray,
    out_path: Path,
    title: str,
    duration_ms: int = GIF_DURATION_MS,
) -> None:
    labels = _phase_lower(result.phase_labels)
    rel_idx = np.where((labels == "ramp off") | (labels == "sw release"))[0]
    if len(rel_idx) == 0:
        rel_idx = np.arange(len(result.times_s))

    dx = float(xg[1] - xg[0])
    dy = float(yg[1] - yg[0])
    U_sw, _, _ = gorkov_normalised(p_sw, dx, dy)

    x_mm = _mm(xg)
    y_mm = _mm(yg)
    traps_mm = _mm(traps_m)
    traj_mm = _mm(result.trajectories)

    levels = np.linspace(np.percentile(U_sw, 3.0), np.percentile(U_sw, 97.0), 32)
    legend_handles = _gif_legend_handles()

    frames: List[Image.Image] = []
    for local_i, fi in enumerate(rel_idx):
        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11.2, 5.4))

        current = traj_mm[fi]
        centres_path = _mm(result.centers[: fi + 1])

        cf = ax_l.contourf(x_mm, y_mm, U_sw, levels=levels, cmap=CMAP_GORKOV)
        if local_i == 0:
            fig.colorbar(cf, ax=ax_l, fraction=0.046, pad=0.02, label="SW Gor'kov U (norm)")

        ax_l.scatter(traps_mm[:, 0], traps_mm[:, 1], marker="x", c=COL_HOME, s=30, linewidths=0.9, zorder=4)
        _plot_home_to_current(ax_l, traps_mm, current, neighbour_idx, COL_NEIGH, lw=0.8, alpha=0.7)
        _plot_home_to_current(ax_l, traps_mm, current, np.array([idx_A]), COL_A, lw=1.2, alpha=0.95)
        _plot_home_to_current(ax_l, traps_mm, current, np.array([idx_B]), COL_B, lw=1.2, alpha=0.95)

        ax_l.scatter(current[neighbour_idx, 0], current[neighbour_idx, 1], c=COL_NEIGH, s=30, zorder=6)
        ax_l.scatter([current[idx_B, 0]], [current[idx_B, 1]], c=COL_B, s=44, zorder=7)
        ax_l.scatter([current[idx_A, 0]], [current[idx_A, 1]], c=COL_A, s=44, zorder=7)

        ax_l.plot(centres_path[:, 0], centres_path[:, 1], color=COL_CTR, lw=1.0, alpha=0.85, zorder=5)
        ax_l.plot(centres_path[-1, 0], centres_path[-1, 1], marker="+", color=COL_CTR, ms=10, mew=1.5, zorder=8)

        ax_l.set_title(f"Release on SW Gor'kov wells ({result.phase_labels[fi]})", fontsize=10)
        ax_l.set_xlabel("x [mm]")
        ax_l.set_ylabel("y [mm]")
        ax_l.set_aspect("equal", adjustable="box")

        # Right panel: local release trajectory.
        ax_r.set_facecolor("#11141b")
        trail = max(rel_idx[0], fi - 50)
        for j in neighbour_idx:
            ax_r.plot(traj_mm[trail : fi + 1, j, 0], traj_mm[trail : fi + 1, j, 1], color=COL_NEIGH, lw=0.8, alpha=0.7)
        ax_r.plot(traj_mm[trail : fi + 1, idx_B, 0], traj_mm[trail : fi + 1, idx_B, 1], color=COL_B, lw=1.4)
        ax_r.plot(traj_mm[trail : fi + 1, idx_A, 0], traj_mm[trail : fi + 1, idx_A, 1], color=COL_A, lw=1.4)

        ax_r.scatter(traps_mm[:, 0], traps_mm[:, 1], marker="x", c=COL_HOME, s=28, linewidths=0.8, alpha=0.9)
        ax_r.scatter(current[neighbour_idx, 0], current[neighbour_idx, 1], c=COL_NEIGH, s=28, zorder=5)
        ax_r.scatter([current[idx_B, 0]], [current[idx_B, 1]], c=COL_B, s=46, zorder=6)
        ax_r.scatter([current[idx_A, 0]], [current[idx_A, 1]], c=COL_A, s=46, zorder=6)

        ax_r.plot(centres_path[:, 0], centres_path[:, 1], color=COL_CTR, lw=1.0, alpha=0.9)
        ax_r.plot(centres_path[-1, 0], centres_path[-1, 1], marker="+", color=COL_CTR, ms=10, mew=1.5)

        ax_r.set_xlim(x_mm[0], x_mm[-1])
        ax_r.set_ylim(y_mm[0], y_mm[-1])
        ax_r.set_aspect("equal", adjustable="box")
        ax_r.set_xlabel("x [mm]", color="white")
        ax_r.set_ylabel("y [mm]", color="white")
        ax_r.tick_params(colors="white", labelsize=8)
        ax_r.set_title(f"Release trajectory t={result.times_s[fi]*1e3:.0f} ms", color="white", fontsize=10)

        fig.suptitle(
            f"{title} | release frame {local_i+1}/{len(rel_idx)}",
            fontsize=11,
            color=_phase_color(result.phase_labels[fi]),
        )
        ax_r.legend(handles=legend_handles, fontsize=7, loc="upper right", framealpha=0.35)
        fig.tight_layout(rect=[0, 0, 1, 0.94])

        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        img = Image.fromarray(buf.reshape(h, w, 4)).convert("RGB")
        frames.append(img)
        plt.close(fig)

    frames[0].save(
        str(out_path),
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    print(f"  GIF saved -> {out_path.name} ({len(rel_idx)} frames)")


# ------------------------------------------------------------------
# Static figures
# ------------------------------------------------------------------
def plot_trajectories(
    result: TransportResult,
    idx_A: int,
    idx_B: int,
    neighbour_idx: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    t_ms = result.times_s * 1e3
    pos0 = result.initial_positions
    traj = result.trajectories

    A_disp = np.linalg.norm(traj[:, idx_A, :] - pos0[idx_A], axis=1) * 1e6
    B_disp = np.linalg.norm(traj[:, idx_B, :] - pos0[idx_B], axis=1) * 1e6
    N_disp = np.linalg.norm(traj[:, neighbour_idx, :] - pos0[neighbour_idx], axis=2) * 1e6

    fig, axes = plt.subplots(3, 1, figsize=(8.8, 9.4), sharex=True)

    axes[0].plot(t_ms, A_disp, color=COL_A, lw=1.5, label="A displacement")
    axes[0].axhline(CAPTURE_RADIUS * 1e6, color="#777777", ls="--", lw=0.9, label="Capture radius")
    axes[0].set_ylabel("A disp [um]")
    axes[0].set_title("Particle A transport")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)

    axes[1].plot(t_ms, B_disp, color=COL_B, lw=1.5, label="B displacement")
    axes[1].axhline(NEIGHBOUR_TOL * 1e6, color="#777777", ls="--", lw=0.9, label="Stability tol")
    axes[1].set_ylabel("B disp [um]")
    axes[1].set_title("Particle B stability")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)

    for j in range(N_disp.shape[1]):
        axes[2].plot(t_ms, N_disp[:, j], color=COL_NEIGH, lw=0.8, alpha=0.65)
    axes[2].plot(t_ms, np.max(N_disp, axis=1), color="black", lw=1.2, label="Neighbour max")
    axes[2].plot(t_ms, np.sqrt(np.mean(N_disp**2, axis=1)), color="black", lw=1.0, ls="--", label="Neighbour RMS")
    axes[2].axhline(NEIGHBOUR_TOL * 1e6, color="#777777", ls="--", lw=0.9, label="Stability tol")
    axes[2].set_ylabel("Neighbour disp [um]")
    axes[2].set_xlabel("Time [ms]")
    axes[2].set_title("Neighbour disturbance")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.25)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(str(out_path), dpi=130)
    plt.close(fig)
    print(f"  Figure saved -> {out_path.name}")


def plot_displacement_comparison(
    res_v: TransportResult,
    res_c: TransportResult,
    idx_A: int,
    idx_B: int,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 4.8))

    for res, lbl, ls in [(res_v, "Vortex", "-"), (res_c, "C-shape", "--")]:
        t_ms = res.times_s * 1e3
        pos0 = res.initial_positions
        traj = res.trajectories
        A_d = np.linalg.norm(traj[:, idx_A, :] - pos0[idx_A], axis=1) * 1e6
        B_d = np.linalg.norm(traj[:, idx_B, :] - pos0[idx_B], axis=1) * 1e6

        ax.plot(t_ms, A_d, color=COL_A, ls=ls, lw=1.5, label=f"A {lbl}")
        ax.plot(t_ms, B_d, color=COL_B, ls=ls, lw=1.2, alpha=0.85, label=f"B {lbl}")

    ax.axhline(CAPTURE_RADIUS * 1e6, color="#555555", lw=0.9, ls=":", label="Capture radius")
    ax.axhline(NEIGHBOUR_TOL * 1e6, color="#888888", lw=0.9, ls=":", label="Stability tol")
    ax.set_title("A/B displacement comparison")
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Displacement from initial [um]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=3)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=130)
    plt.close(fig)
    print(f"  Figure saved -> {out_path.name}")


def plot_neighbour_comparison(
    res_v: TransportResult,
    res_c: TransportResult,
    neighbour_idx: np.ndarray,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=True)

    for ax, res, title in [(axes[0], res_v, "Vortex"), (axes[1], res_c, "C-shape")]:
        t_ms = res.times_s * 1e3
        pos0 = res.initial_positions
        traj = res.trajectories
        N = np.linalg.norm(traj[:, neighbour_idx, :] - pos0[neighbour_idx], axis=2) * 1e6

        for j in range(N.shape[1]):
            ax.plot(t_ms, N[:, j], color=COL_NEIGH, lw=0.8, alpha=0.6)
        ax.plot(t_ms, np.max(N, axis=1), color="black", lw=1.2, label="Max")
        ax.plot(t_ms, np.sqrt(np.mean(N**2, axis=1)), color="black", ls="--", lw=1.0, label="RMS")
        ax.axhline(NEIGHBOUR_TOL * 1e6, color="#aa0000", lw=0.9, ls="--", label="Stability tol")

        ax.set_title(f"{title} neighbour disturbance")
        ax.set_xlabel("Time [ms]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Displacement [um]")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=130)
    plt.close(fig)
    print(f"  Figure saved -> {out_path.name}")


def plot_release_overlay_sw_wells(
    p_sw: np.ndarray,
    xg: np.ndarray,
    yg: np.ndarray,
    traps_m: np.ndarray,
    idx_A: int,
    idx_B: int,
    res_v: TransportResult,
    res_c: TransportResult,
    out_path: Path,
) -> None:
    dx = float(xg[1] - xg[0])
    dy = float(yg[1] - yg[0])
    U_sw, _, _ = gorkov_normalised(p_sw, dx, dy)

    x_mm = _mm(xg)
    y_mm = _mm(yg)
    traps_mm = _mm(traps_m)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.9), sharex=True, sharey=True)
    levels = np.linspace(np.percentile(U_sw, 3.0), np.percentile(U_sw, 97.0), 34)

    for ax, res, title in [(axes[0], res_v, "Vortex release"), (axes[1], res_c, "C-shape release")]:
        labels = _phase_lower(res.phase_labels)
        i_rel = np.where(labels == "sw release")[0]
        if len(i_rel) == 0:
            i_rel = np.arange(len(res.times_s))

        i0 = int(i_rel[0])
        i1 = int(i_rel[len(i_rel) // 2])
        i2 = int(i_rel[-1])

        ax.contourf(x_mm, y_mm, U_sw, levels=levels, cmap=CMAP_GORKOV)
        ax.scatter(traps_mm[:, 0], traps_mm[:, 1], marker="x", c=COL_HOME, s=34, linewidths=0.9)

        for name, ii, cA, cB in [
            ("start", i0, "#ff5252", "#64b5f6"),
            ("mid", i1, "#ff8a80", "#90caf9"),
            ("end", i2, "#ffd1d1", "#dbeeff"),
        ]:
            pA = _mm(res.trajectories[ii, idx_A])
            pB = _mm(res.trajectories[ii, idx_B])
            ax.scatter([pA[0]], [pA[1]], c=cA, s=40)
            ax.scatter([pB[0]], [pB[1]], c=cB, s=40, marker="s")
            ax.text(pA[0] + 0.01, pA[1] - 0.02, f"A {name}", color=cA, fontsize=7)
            ax.text(pB[0] + 0.01, pB[1] - 0.02, f"B {name}", color=cB, fontsize=7)

        ax.set_title(title)
        ax.set_xlabel("x [mm]")
        ax.set_aspect("equal", adjustable="box")

    axes[0].set_ylabel("y [mm]")
    fig.suptitle("Release overlay on standing-wave Gor'kov wells", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"  Figure saved -> {out_path.name}")


# ------------------------------------------------------------------
# Export helpers
# ------------------------------------------------------------------
def save_calibration_outputs(rows: List[Dict[str, Any]], selected: Dict[str, Any]) -> None:
    cal_csv = OUT_CMP / "vortex_recalibration.csv"
    cal_json = OUT_CMP / "vortex_recalibration.json"

    keys = [
        "alpha",
        "psi_over_pi",
        "A_capture_success",
        "progress_fraction_of_AB",
        "A_to_B_target_um",
        "B_stable",
        "B_moved_um",
        "neigh_all_stable",
        "neigh_stable_fraction",
        "neigh_rms_um",
        "neigh_max_um",
        "A_target_region",
        "B_home_region",
        "force_ratio_mid",
        "secondary_scalar_score",
    ]

    with open(cal_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            rr = {k: r.get(k) for k in keys}
            w.writerow(rr)

    cal_json.write_text(json.dumps({"selected": selected, "ranked_candidates": rows}, indent=2))
    print(f"  Calibration table -> {cal_csv.name}")
    print(f"  Calibration details -> {cal_json.name}")


def save_metric_csv(metrics_v: Dict[str, Any], metrics_c: Dict[str, Any]) -> None:
    rows = []
    for m in [metrics_v, metrics_c]:
        rows.append(
            {
                "method": m["method"],
                "alpha": m["controls"].get("alpha"),
                "psi_over_pi": m["controls"].get("psi_over_pi"),
                "transport_A_capture_success": m["transport_effectiveness"]["A_capture_success"],
                "transport_progress_fraction_of_AB": m["transport_effectiveness"]["progress_fraction_of_AB"],
                "transport_A_to_B_target_um": m["transport_effectiveness"]["A_to_B_target_um"],
                "B_stable": m["B_stability"]["B_stable"],
                "B_moved_um": m["B_stability"]["B_moved_um"],
                "neigh_all_stable": m["neighbour_stability"]["all_neighbours_stable"],
                "neigh_stable_fraction": m["neighbour_stability"]["stable_fraction"],
                "neigh_max_um": m["neighbour_stability"]["max_displacement_um"],
                "neigh_rms_um": m["neighbour_stability"]["rms_displacement_um"],
                "release_A_final_nearest": m["release_success"]["A_final_nearest_trap_idx"],
                "release_B_final_nearest": m["release_success"]["B_final_nearest_trap_idx"],
                "release_neigh_home_recovered_fraction": m["release_success"]["neighbour_home_recovered_fraction"],
                "secondary_scalar_score": m["secondary_scalar_score"],
            }
        )

    out_csv = OUT_CMP / "success_metric_table.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  Comparison CSV -> {out_csv.name}")


def build_summary(
    metrics_v: Dict[str, Any],
    metrics_c: Dict[str, Any],
    vortex_geom_meta: Dict[str, Any],
) -> Dict[str, Any]:
    selected_geom = vortex_geom_meta.get("selected", {})
    feasibility = vortex_geom_meta.get("feasibility_statement", {})

    summary = {
        "comparison_intent": "fair overdamped comparison with split metrics",
        "no_primary_scalar_winner": True,
        "transport_effectiveness": {
            "vortex": metrics_v["transport_effectiveness"],
            "cshape": metrics_c["transport_effectiveness"],
        },
        "B_stability": {
            "vortex": metrics_v["B_stability"],
            "cshape": metrics_c["B_stability"],
        },
        "neighbour_stability": {
            "vortex": metrics_v["neighbour_stability"],
            "cshape": metrics_c["neighbour_stability"],
        },
        "release_success": {
            "vortex": metrics_v["release_success"],
            "cshape": metrics_c["release_success"],
        },
        "secondary_scalar_scores": {
            "vortex": metrics_v["secondary_scalar_score"],
            "cshape": metrics_c["secondary_scalar_score"],
            "note": "secondary context only",
        },
        "schedule": metrics_v["schedule"],
        "vortex_geometry_context": {
            "family": selected_geom.get("family"),
            "aperture_mm": selected_geom.get("aperture_mm"),
            "source_distance_mm": selected_geom.get("source_distance_mm"),
            "cone_deg": selected_geom.get("cone_deg"),
            "waist_mm": selected_geom.get("waist_mm"),
            "ring_radius_mm": selected_geom.get("ring_radius_mm"),
            "spans_too_many_traps": feasibility.get("spans_too_many_traps"),
            "statement": feasibility.get("statement"),
        },
    }
    return summary


def write_diagnostic_notes(
    summary: Dict[str, Any],
    metrics_v: Dict[str, Any],
    metrics_c: Dict[str, Any],
) -> None:
    lines: List[str] = []
    lines.append("# Diagnostic Notes (corrected unified workflow)")
    lines.append("")
    lines.append("## What changed")
    lines.append("- Single-pass overdamped transport is used (no duplicate integration pass).")
    lines.append("- Vortex alpha/psi were recalibrated under the overdamped engine and current objective.")
    lines.append("- Shared schedule now includes meaningful SW-only start and long SW-only settle.")
    lines.append("- Visuals use equal aspect, clearer labels, home markers, displacement vectors, and centre path.")
    lines.append("- Release interpretation includes SW Gor'kov-well overlays.")
    lines.append("- Metrics are split by transport/B/neighbour/release; scalar score is secondary only.")
    lines.append("")

    lines.append("## Schedule and integration")
    sch = summary["schedule"]
    lines.append(f"- dt: {sch['dt_ms']:.3f} ms")
    lines.append(f"- total steps: {sch['n_steps_total']}")
    lines.append(f"- settle: {sch['settle_ms']:.1f} ms ({sch['n_steps_settle']} steps)")
    lines.append("")

    lines.append("## Vortex recalibration result")
    vctrl = metrics_v["controls"]
    lines.append(f"- selected alpha: {vctrl.get('alpha')}")
    lines.append(f"- selected psi/pi: {vctrl.get('psi_over_pi')}")
    geom = summary["vortex_geometry_context"]
    lines.append(f"- family: {geom.get('family')}")
    lines.append(f"- aperture_mm: {geom.get('aperture_mm')}")
    lines.append(f"- source_distance_mm: {geom.get('source_distance_mm')}")
    lines.append(f"- cone_deg: {geom.get('cone_deg')}")
    lines.append(f"- waist_mm: {geom.get('waist_mm')}")
    lines.append(f"- ring_radius_mm: {geom.get('ring_radius_mm')}")
    lines.append(f"- spans_too_many_traps: {geom.get('spans_too_many_traps')}")
    lines.append("")

    lines.append("## Settle-force diagnostics")
    s_v = metrics_v["settle_diagnostics"]
    s_c = metrics_c["settle_diagnostics"]
    lines.append(f"- vortex A settle force mean: {s_v['A_force_mag']['mean']:.3e}")
    lines.append(f"- vortex B settle force mean: {s_v['B_force_mag']['mean']:.3e}")
    lines.append(f"- cshape A settle force mean: {s_c['A_force_mag']['mean']:.3e}")
    lines.append(f"- cshape B settle force mean: {s_c['B_force_mag']['mean']:.3e}")
    lines.append("")

    lines.append("## Interpretation")
    tv = metrics_v["transport_effectiveness"]
    tc = metrics_c["transport_effectiveness"]
    nv = metrics_v["neighbour_stability"]
    nc = metrics_c["neighbour_stability"]
    lines.append(f"- vortex progress fraction AB: {tv['progress_fraction_of_AB']}")
    lines.append(f"- cshape progress fraction AB: {tc['progress_fraction_of_AB']}")
    lines.append(f"- vortex neighbour RMS [um]: {nv['rms_displacement_um']}")
    lines.append(f"- cshape neighbour RMS [um]: {nc['rms_displacement_um']}")

    out_md = OUT_CMP / "diagnostic_notes.md"
    out_md.write_text("\n".join(lines) + "\n")
    print(f"  Notes -> {out_md.name}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main() -> None:
    _ensure_dir(OUT_ROOT)
    _ensure_dir(OUT_V)
    _ensure_dir(OUT_C)
    _ensure_dir(OUT_CMP)

    print("Loading data...")
    D = load_data()

    xg = D["xg"]
    yg = D["yg"]
    p_sw = D["p_sw"]
    p_lens = D["p_lens"]
    traps_m = D["traps_m"]
    idx_A = D["idx_A"]
    idx_B = D["idx_B"]
    neighbour_idx = D["neighbour_idx"]

    xg_v = D["xg_v"]
    yg_v = D["yg_v"]
    p_vortex_full = D["p_vortex_full"]
    vortex_geom_meta = D["vortex_geom_meta"]

    A_xy = traps_m[idx_A]
    B_xy = traps_m[idx_B]

    print(f"  Grid ROI: {len(xg)}x{len(yg)}, x=[{xg[0]*1e3:.3f},{xg[-1]*1e3:.3f}] mm, y=[{yg[0]*1e3:.3f},{yg[-1]*1e3:.3f}] mm")
    print(f"  Trap A idx={idx_A}: {_mm(A_xy)} mm")
    print(f"  Trap B idx={idx_B}: {_mm(B_xy)} mm")
    print(f"  AB distance: {np.linalg.norm(A_xy-B_xy)*1e6:.2f} um")

    sw_interp = make_sw_force_interpolators(p_sw, xg, yg)

    # Generators
    gen_vortex = VortexPerturbation(
        p_vortex_centered=p_vortex_full,
        xg=xg_v,
        yg=yg_v,
        out_xg=xg,
        out_yg=yg,
    )
    gen_cshape = CShapePerturbation(p_lens, xg, yg, window_sigma=None)

    # Recalibrate vortex controls on corrected overdamped engine.
    selected_vortex, calib_rows = calibrate_vortex_controls(
        p_sw=p_sw,
        xg=xg,
        yg=yg,
        traps_m=traps_m,
        idx_A=idx_A,
        idx_B=idx_B,
        neighbour_idx=neighbour_idx,
        gen_vortex=gen_vortex,
        sw_interp=sw_interp,
    )
    save_calibration_outputs(calib_rows, selected_vortex)

    vortex_alpha = float(selected_vortex["alpha"])
    vortex_psi = float(selected_vortex["psi"])

    phases_v = build_vortex_phases(A_xy, B_xy, alpha=vortex_alpha)
    phases_c = build_cshape_phases(A_xy, B_xy)

    print("\nRunning final VORTEX case...")
    t0 = time.time()
    res_v = run_transport(
        p_sw=p_sw,
        perturbation_fn=gen_vortex.get_field,
        phases=phases_v,
        psi=vortex_psi,
        initial_positions=traps_m.copy(),
        xg=xg,
        yg=yg,
        idx_A=idx_A,
        idx_B=idx_B,
        dt=DT,
        n_frames=N_FRAMES,
    )
    print(f"  done in {time.time()-t0:.1f}s, frames={len(res_v.times_s)}, merge_time={res_v.merge_time_s}")

    print("Running final C-SHAPE case...")
    t0 = time.time()
    res_c = run_transport(
        p_sw=p_sw,
        perturbation_fn=gen_cshape.get_field,
        phases=phases_c,
        psi=CSHAPE_PSI,
        initial_positions=traps_m.copy(),
        xg=xg,
        yg=yg,
        idx_A=idx_A,
        idx_B=idx_B,
        dt=DT,
        n_frames=N_FRAMES,
    )
    print(f"  done in {time.time()-t0:.1f}s, frames={len(res_c.times_s)}, merge_time={res_c.merge_time_s}")

    vortex_force_context = compute_vortex_force_context(
        p_sw=p_sw,
        xg=xg,
        yg=yg,
        A_xy=A_xy,
        phases_v=phases_v,
        gen_vortex=gen_vortex,
        alpha=vortex_alpha,
        psi=vortex_psi,
    )

    metrics_v = build_split_metrics(
        method="vortex",
        result=res_v,
        phases=phases_v,
        p_sw=p_sw,
        xg=xg,
        yg=yg,
        traps_m=traps_m,
        idx_A=idx_A,
        idx_B=idx_B,
        neighbour_idx=neighbour_idx,
        controls={
            "alpha": vortex_alpha,
            "psi_over_pi": vortex_psi / np.pi,
            "psi_rad": vortex_psi,
            "beta_sw_min": SW_SCALE,
        },
        sw_interp=sw_interp,
        vortex_force_context=vortex_force_context,
    )

    metrics_c = build_split_metrics(
        method="cshape",
        result=res_c,
        phases=phases_c,
        p_sw=p_sw,
        xg=xg,
        yg=yg,
        traps_m=traps_m,
        idx_A=idx_A,
        idx_B=idx_B,
        neighbour_idx=neighbour_idx,
        controls={
            "alpha": CSHAPE_ALPHA,
            "psi_over_pi": CSHAPE_PSI / np.pi,
            "psi_rad": CSHAPE_PSI,
            "beta_sw_min": CSHAPE_BSW_MIN,
        },
        sw_interp=sw_interp,
    )

    # Save metrics.
    (OUT_V / "metrics.json").write_text(json.dumps(metrics_v, indent=2))
    (OUT_C / "metrics.json").write_text(json.dumps(metrics_c, indent=2))
    print("\nSaved metrics.json for both methods")

    # GIFs.
    print("\nRendering transport GIFs...")
    render_transport_gif(
        result=res_v,
        p_sw=p_sw,
        perturb_fn=gen_vortex.get_field,
        psi=vortex_psi,
        xg=xg,
        yg=yg,
        traps_m=traps_m,
        idx_A=idx_A,
        idx_B=idx_B,
        neighbour_idx=neighbour_idx,
        out_path=OUT_V / "transport.gif",
        title=f"Vortex transport alpha={vortex_alpha:.3f}, psi={vortex_psi/np.pi:.2f}pi",
    )
    render_transport_gif(
        result=res_c,
        p_sw=p_sw,
        perturb_fn=gen_cshape.get_field,
        psi=CSHAPE_PSI,
        xg=xg,
        yg=yg,
        traps_m=traps_m,
        idx_A=idx_A,
        idx_B=idx_B,
        neighbour_idx=neighbour_idx,
        out_path=OUT_C / "transport.gif",
        title=f"C-shape transport alpha={CSHAPE_ALPHA:.3f}, beta_sw_min={CSHAPE_BSW_MIN:.2f}",
    )

    print("Rendering release GIFs...")
    render_release_gif(
        result=res_v,
        p_sw=p_sw,
        xg=xg,
        yg=yg,
        traps_m=traps_m,
        idx_A=idx_A,
        idx_B=idx_B,
        neighbour_idx=neighbour_idx,
        out_path=OUT_V / "release.gif",
        title="Vortex release",
    )
    render_release_gif(
        result=res_c,
        p_sw=p_sw,
        xg=xg,
        yg=yg,
        traps_m=traps_m,
        idx_A=idx_A,
        idx_B=idx_B,
        neighbour_idx=neighbour_idx,
        out_path=OUT_C / "release.gif",
        title="C-shape release",
    )

    # Figures.
    print("\nRendering static figures...")
    plot_trajectories(
        result=res_v,
        idx_A=idx_A,
        idx_B=idx_B,
        neighbour_idx=neighbour_idx,
        out_path=OUT_V / "trajectories.png",
        title=f"Vortex trajectories (alpha={vortex_alpha:.3f}, psi={vortex_psi/np.pi:.2f}pi)",
    )
    plot_trajectories(
        result=res_c,
        idx_A=idx_A,
        idx_B=idx_B,
        neighbour_idx=neighbour_idx,
        out_path=OUT_C / "trajectories.png",
        title=f"C-shape trajectories (alpha={CSHAPE_ALPHA:.3f}, beta_sw_min={CSHAPE_BSW_MIN:.2f})",
    )

    plot_displacement_comparison(res_v, res_c, idx_A, idx_B, OUT_CMP / "displacement_vs_time.png")
    plot_neighbour_comparison(res_v, res_c, neighbour_idx, OUT_CMP / "neighbour_disturbance.png")
    plot_release_overlay_sw_wells(
        p_sw=p_sw,
        xg=xg,
        yg=yg,
        traps_m=traps_m,
        idx_A=idx_A,
        idx_B=idx_B,
        res_v=res_v,
        res_c=res_c,
        out_path=OUT_CMP / "release_overlay_sw_wells.png",
    )

    # CSV + summary + notes.
    save_metric_csv(metrics_v, metrics_c)
    summary = build_summary(metrics_v, metrics_c, vortex_geom_meta)
    (OUT_CMP / "summary.json").write_text(json.dumps(summary, indent=2))
    write_diagnostic_notes(summary, metrics_v, metrics_c)

    # Console summary focused on split metrics.
    print("\n" + "=" * 66)
    print("UPDATED FAIR COMPARISON COMPLETE")
    print("=" * 66)

    for label, m in [("VORTEX", metrics_v), ("C-SHAPE", metrics_c)]:
        t = m["transport_effectiveness"]
        b = m["B_stability"]
        n = m["neighbour_stability"]
        r = m["release_success"]
        print(f"\n{label}")
        print(f"  Transport: capture={t['A_capture_success']}, progress={t['progress_fraction_of_AB']:.3f}, A_to_B_target={t['A_to_B_target_um']} um")
        print(f"  B stability: stable={b['B_stable']}, B_moved={b['B_moved_um']} um")
        print(f"  Neighbours: all_stable={n['all_neighbours_stable']}, stable_fraction={n['stable_fraction']}, max={n['max_displacement_um']} um, rms={n['rms_displacement_um']} um")
        print(f"  Release: A_final_trap={r['A_final_nearest_trap_idx']}, B_final_trap={r['B_final_nearest_trap_idx']}, neighbour_home_recovered={r['neighbour_home_recovered_fraction']}")

    print("\nSecondary scalar scores (context only):")
    print(f"  vortex: {metrics_v['secondary_scalar_score']}")
    print(f"  cshape: {metrics_c['secondary_scalar_score']}")

    if "vortex_force_context" in metrics_v:
        vf = metrics_v["vortex_force_context"]
        print("\nVortex force ratio |F_vortex_only|/|F_sw| on A:")
        for key in ["start", "mid", "end"]:
            if key in vf:
                print(f"  {key}: {vf[key]['ratio_vortex_to_sw']:.3f}")

    print(f"\nOutputs in: {OUT_ROOT}")
    print("=" * 66)


if __name__ == "__main__":
    main()
