#!/usr/bin/env python3
"""
Random sweep for local ideal bridge perturbation candidates.

This script reuses the existing static comparison logic from:
  scripts/dev/ideal_pressure_static_compare.py

Goal:
- Sample many local perturbation parameter combinations.
- Score each candidate for A->B movement potential while penalizing disturbance.
- Save the top candidates as 2x2 comparison PNGs for manual inspection.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dev import ideal_pressure_static_compare as ipc


# ---------------------------------------------------------------------------
# Sweep controls
# ---------------------------------------------------------------------------
NUM_CANDIDATES = int(os.getenv("SWEEP_NUM_CANDIDATES", "140"))
TOP_K = int(os.getenv("SWEEP_TOP_K", "10"))
RNG_SEED = int(os.getenv("SWEEP_SEED", "20260319"))

# Candidate ranges (requested practical search bounds)
ANCHOR_AMPLITUDE_RANGE_PA = (-250.0, -50.0)
SOURCE_AMPLITUDE_RANGE_PA = (0.0, 150.0)
ANCHOR_RADIUS_RANGE_M = (5.0e-5, 1.5e-4)
SOURCE_RADIUS_RANGE_M = (5.0e-5, 1.5e-4)

# Scoring weights
PENALTY_WEIGHT = float(os.getenv("SWEEP_PENALTY_WEIGHT", "1.0"))
NEIGHBOR_MEAN_ABS_WEIGHT = float(os.getenv("SWEEP_NEIGHBOR_MEAN_ABS_WEIGHT", "0.5"))

# Exclusion radius around A/B when computing ROI disturbance penalty
AB_EXCLUSION_RADIUS_M = float(os.getenv("SWEEP_AB_EXCLUSION_RADIUS_M", "2.0e-4"))

OUT_DIR = ipc.OUT_DIR / "sweep_candidates"


@dataclass
class SweepContext:
    x_full: np.ndarray
    y_full: np.ndarray
    p_sw_full: np.ndarray
    U_sw_full: np.ndarray
    x_roi: np.ndarray
    y_roi: np.ndarray
    U_sw_roi: np.ndarray
    ix_roi: np.ndarray
    iy_roi: np.ndarray
    traps_m: np.ndarray
    traps_mm: np.ndarray
    idx_A: int
    idx_B: int
    neigh_idx: np.ndarray
    A_xy: np.ndarray
    B_xy: np.ndarray
    ix_A: int
    iy_A: int
    ix_B: int
    iy_B: int
    neigh_grid_ix: np.ndarray
    neigh_grid_iy: np.ndarray
    roi_exclude_ab_mask: np.ndarray
    dx_full: float
    dy_full: float
    x_full_mm: np.ndarray
    y_full_mm: np.ndarray
    x_roi_mm: np.ndarray
    y_roi_mm: np.ndarray
    extent_full: list[float]
    extent_roi: list[float]


def _nearest_index(axis: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(axis - value)))


def _sample_candidate_params(rng: np.random.Generator) -> dict[str, float]:
    return {
        "anchor_amplitude_pa": float(rng.uniform(*ANCHOR_AMPLITUDE_RANGE_PA)),
        "source_amplitude_pa": float(rng.uniform(*SOURCE_AMPLITUDE_RANGE_PA)),
        "anchor_radius_m": float(rng.uniform(*ANCHOR_RADIUS_RANGE_M)),
        "source_radius_m": float(rng.uniform(*SOURCE_RADIUS_RANGE_M)),
    }


def _build_perturbation_with_params(
    x: np.ndarray,
    y: np.ndarray,
    point_a: np.ndarray,
    point_b: np.ndarray,
    params: dict[str, float],
) -> np.ndarray:
    """
    Reuse the existing build_ideal_perturbation logic by temporarily setting
    parameter globals in the imported module.
    """
    old_anchor_amp = ipc.ANCHOR_AMPLITUDE_PA
    old_source_amp = ipc.SOURCE_AMPLITUDE_PA
    old_anchor_rad = ipc.ANCHOR_RADIUS_M
    old_source_rad = ipc.SOURCE_RADIUS_M

    try:
        ipc.ANCHOR_AMPLITUDE_PA = float(params["anchor_amplitude_pa"])
        ipc.SOURCE_AMPLITUDE_PA = float(params["source_amplitude_pa"])
        ipc.ANCHOR_RADIUS_M = float(params["anchor_radius_m"])
        ipc.SOURCE_RADIUS_M = float(params["source_radius_m"])
        return ipc.build_ideal_perturbation(x, y, point_a, point_b)
    finally:
        ipc.ANCHOR_AMPLITUDE_PA = old_anchor_amp
        ipc.SOURCE_AMPLITUDE_PA = old_source_amp
        ipc.ANCHOR_RADIUS_M = old_anchor_rad
        ipc.SOURCE_RADIUS_M = old_source_rad


def _build_context() -> SweepContext:
    ov = np.load(ipc.OVERLAY_NPZ)
    x_roi_target = ov["xg"].astype(float)
    y_roi_target = ov["yg"].astype(float)
    traps_m = ov["traps_m"].astype(float)
    idx_A = int(ov["idx_A"])
    idx_B = int(ov["idx_B"])

    vd = np.load(ipc.VORTEX_NPZ)
    x_full = vd["xg"].astype(float)
    y_full = vd["yg"].astype(float)
    p_sw_full = vd["p_sw"].astype(complex)

    dx_full = float(x_full[1] - x_full[0])
    dy_full = float(y_full[1] - y_full[0])

    U_sw_full, _, _ = ipc._gorkov_fields(p_sw_full, dx_full, dy_full)

    ix_roi, iy_roi = ipc._crop_indices(
        x_full,
        y_full,
        float(x_roi_target[0]),
        float(x_roi_target[-1]),
        float(y_roi_target[0]),
        float(y_roi_target[-1]),
    )
    x_roi = x_full[ix_roi]
    y_roi = y_full[iy_roi]
    U_sw_roi = U_sw_full[np.ix_(iy_roi, ix_roi)]

    neigh_idx = np.array(sorted(set(range(len(traps_m))) - {idx_A, idx_B}), dtype=int)
    A_xy = traps_m[idx_A]
    B_xy = traps_m[idx_B]

    ix_A = _nearest_index(x_full, float(A_xy[0]))
    iy_A = _nearest_index(y_full, float(A_xy[1]))
    ix_B = _nearest_index(x_full, float(B_xy[0]))
    iy_B = _nearest_index(y_full, float(B_xy[1]))

    neigh_grid_ix = np.array([_nearest_index(x_full, float(traps_m[i, 0])) for i in neigh_idx], dtype=int)
    neigh_grid_iy = np.array([_nearest_index(y_full, float(traps_m[i, 1])) for i in neigh_idx], dtype=int)

    yy_roi, xx_roi = np.meshgrid(y_roi, x_roi, indexing="ij")
    dist2_A = (xx_roi - float(A_xy[0])) ** 2 + (yy_roi - float(A_xy[1])) ** 2
    dist2_B = (xx_roi - float(B_xy[0])) ** 2 + (yy_roi - float(B_xy[1])) ** 2
    roi_exclude_ab_mask = (dist2_A > AB_EXCLUSION_RADIUS_M**2) & (dist2_B > AB_EXCLUSION_RADIUS_M**2)

    x_full_mm = x_full * 1e3
    y_full_mm = y_full * 1e3
    x_roi_mm = x_roi * 1e3
    y_roi_mm = y_roi * 1e3
    extent_full = [x_full_mm[0], x_full_mm[-1], y_full_mm[0], y_full_mm[-1]]
    extent_roi = [x_roi_mm[0], x_roi_mm[-1], y_roi_mm[0], y_roi_mm[-1]]

    return SweepContext(
        x_full=x_full,
        y_full=y_full,
        p_sw_full=p_sw_full,
        U_sw_full=U_sw_full,
        x_roi=x_roi,
        y_roi=y_roi,
        U_sw_roi=U_sw_roi,
        ix_roi=ix_roi,
        iy_roi=iy_roi,
        traps_m=traps_m,
        traps_mm=traps_m * 1e3,
        idx_A=idx_A,
        idx_B=idx_B,
        neigh_idx=neigh_idx,
        A_xy=A_xy,
        B_xy=B_xy,
        ix_A=ix_A,
        iy_A=iy_A,
        ix_B=ix_B,
        iy_B=iy_B,
        neigh_grid_ix=neigh_grid_ix,
        neigh_grid_iy=neigh_grid_iy,
        roi_exclude_ab_mask=roi_exclude_ab_mask,
        dx_full=dx_full,
        dy_full=dy_full,
        x_full_mm=x_full_mm,
        y_full_mm=y_full_mm,
        x_roi_mm=x_roi_mm,
        y_roi_mm=y_roi_mm,
        extent_full=extent_full,
        extent_roi=extent_roi,
    )


def _evaluate_candidate(ctx: SweepContext, params: dict[str, float]) -> dict[str, Any]:
    p_ideal = _build_perturbation_with_params(ctx.x_full, ctx.y_full, ctx.A_xy, ctx.B_xy, params)
    p_combined = ctx.p_sw_full + p_ideal
    U_combined_full, _, _ = ipc._gorkov_fields(p_combined, ctx.dx_full, ctx.dy_full)
    U_combined_roi = U_combined_full[np.ix_(ctx.iy_roi, ctx.ix_roi)]

    U_A = float(U_combined_full[ctx.iy_A, ctx.ix_A])
    U_B = float(U_combined_full[ctx.iy_B, ctx.ix_B])
    score_ab = U_A - U_B

    delta_roi = U_combined_roi - ctx.U_sw_roi
    roi_vals = delta_roi[ctx.roi_exclude_ab_mask]
    penalty_roi_std = float(np.std(roi_vals)) if roi_vals.size > 0 else 0.0

    delta_full = U_combined_full - ctx.U_sw_full
    if ctx.neigh_idx.size > 0:
        neigh_delta = delta_full[ctx.neigh_grid_iy, ctx.neigh_grid_ix]
        penalty_neigh_mean_abs = float(np.mean(np.abs(neigh_delta)))
    else:
        penalty_neigh_mean_abs = 0.0

    penalty = penalty_roi_std + NEIGHBOR_MEAN_ABS_WEIGHT * penalty_neigh_mean_abs
    score_final = score_ab - PENALTY_WEIGHT * penalty

    return {
        "params": params,
        "score_ab": float(score_ab),
        "penalty_roi_std": float(penalty_roi_std),
        "penalty_neigh_mean_abs": float(penalty_neigh_mean_abs),
        "penalty": float(penalty),
        "score_final": float(score_final),
        "U_A": float(U_A),
        "U_B": float(U_B),
        "peak_ideal_pa": float(np.max(np.abs(p_ideal))),
        "peak_combined_pa": float(np.max(np.abs(p_combined))),
    }


def _render_candidate_figure(
    ctx: SweepContext,
    params: dict[str, float],
    score_row: dict[str, Any],
    rank: int,
    out_png: Path,
) -> None:
    p_ideal = _build_perturbation_with_params(ctx.x_full, ctx.y_full, ctx.A_xy, ctx.B_xy, params)
    U_combined_full, _, _ = ipc._gorkov_fields(ctx.p_sw_full + p_ideal, ctx.dx_full, ctx.dy_full)
    U_combined_roi = U_combined_full[np.ix_(ctx.iy_roi, ctx.ix_roi)]

    # Keep the same scaling strategy used in ideal_pressure_static_compare.py
    vmin, vmax = ipc._auto_limits(ctx.U_sw_full, ctx.U_sw_roi, U_combined_full, U_combined_roi)

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 12.0))
    (ax_sw_roi, ax_combined_roi), (ax_sw_full, ax_combined_full) = axes

    ipc._draw_static_panel(
        ax_sw_roi,
        ctx.U_sw_roi,
        vmin,
        vmax,
        ctx.extent_roi,
        ctx.x_roi_mm,
        ctx.y_roi_mm,
        ctx.traps_mm,
        ctx.idx_A,
        ctx.idx_B,
        ctx.neigh_idx,
        "Standing Wave Only (ROI)",
    )
    ipc._draw_static_panel(
        ax_combined_roi,
        U_combined_roi,
        vmin,
        vmax,
        ctx.extent_roi,
        ctx.x_roi_mm,
        ctx.y_roi_mm,
        ctx.traps_mm,
        ctx.idx_A,
        ctx.idx_B,
        ctx.neigh_idx,
        "Standing Wave + Ideal Perturbation (ROI)",
    )
    ipc._draw_static_panel(
        ax_sw_full,
        ctx.U_sw_full,
        vmin,
        vmax,
        ctx.extent_full,
        ctx.x_full_mm,
        ctx.y_full_mm,
        ctx.traps_mm,
        ctx.idx_A,
        ctx.idx_B,
        ctx.neigh_idx,
        "Standing Wave Only (Full Domain)",
    )
    ipc._draw_static_panel(
        ax_combined_full,
        U_combined_full,
        vmin,
        vmax,
        ctx.extent_full,
        ctx.x_full_mm,
        ctx.y_full_mm,
        ctx.traps_mm,
        ctx.idx_A,
        ctx.idx_B,
        ctx.neigh_idx,
        "Standing Wave + Ideal Perturbation (Full Domain)",
    )

    fig.suptitle(
        "Static Comparison: Ideal Local Pressure Perturbation\n"
        "Gor'kov Potential U_Gorkov [J]\n"
        f"candidate {rank:02d} | score_final={score_row['score_final']:.3e} | "
        f"A-B={score_row['score_ab']:.3e} | penalty={score_row['penalty']:.3e}",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if NUM_CANDIDATES < 1:
        raise ValueError("SWEEP_NUM_CANDIDATES must be >= 1")
    if TOP_K < 1:
        raise ValueError("SWEEP_TOP_K must be >= 1")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old_png in OUT_DIR.glob("candidate_*.png"):
        old_png.unlink()

    print("=" * 72)
    print("SWEEP IDEAL BRIDGE FIELDS")
    print("=" * 72)
    print(f"Candidates: {NUM_CANDIDATES}")
    print(f"Top K: {TOP_K}")
    print(f"Seed: {RNG_SEED}")
    print(f"Output dir: {OUT_DIR}")

    print("\n1) Building baseline context from cached fields...")
    ctx = _build_context()
    print(f"   A idx={ctx.idx_A}, B idx={ctx.idx_B}, neighbours={len(ctx.neigh_idx)}")

    print("\n2) Sweeping random candidates and scoring...")
    rng = np.random.default_rng(RNG_SEED)
    rows: list[dict[str, Any]] = []
    for k in range(NUM_CANDIDATES):
        params = _sample_candidate_params(rng)
        row = _evaluate_candidate(ctx, params)
        row["candidate_index"] = int(k)
        rows.append(row)

        if (k + 1) % 25 == 0 or (k + 1) == NUM_CANDIDATES:
            print(f"   evaluated {k + 1}/{NUM_CANDIDATES}")

    rows_sorted = sorted(rows, key=lambda r: r["score_final"], reverse=True)
    top_rows = rows_sorted[: min(TOP_K, len(rows_sorted))]

    print("\n3) Rendering top candidates in the same 2x2 format...")
    for rank, row in enumerate(top_rows, start=1):
        png_path = OUT_DIR / f"candidate_{rank:02d}.png"
        _render_candidate_figure(ctx, row["params"], row, rank, png_path)
        print(f"   saved {png_path.name}")

    out_json = OUT_DIR / "candidate_parameters.json"
    payload = {
        "script": "scripts/dev/sweep_ideal_bridge_fields.py",
        "source_logic": "scripts/dev/ideal_pressure_static_compare.py",
        "num_candidates": int(NUM_CANDIDATES),
        "top_k": int(len(top_rows)),
        "rng_seed": int(RNG_SEED),
        "ranges": {
            "anchor_amplitude_pa": list(ANCHOR_AMPLITUDE_RANGE_PA),
            "source_amplitude_pa": list(SOURCE_AMPLITUDE_RANGE_PA),
            "anchor_radius_m": list(ANCHOR_RADIUS_RANGE_M),
            "source_radius_m": list(SOURCE_RADIUS_RANGE_M),
        },
        "scoring": {
            "score": "(U_A - U_B) - penalty_weight * (std(delta_U_roi_excluding_A_B) + neighbor_mean_abs_weight * mean(abs(delta_U_at_neighbors)))",
            "penalty_weight": float(PENALTY_WEIGHT),
            "neighbor_mean_abs_weight": float(NEIGHBOR_MEAN_ABS_WEIGHT),
            "ab_exclusion_radius_m": float(AB_EXCLUSION_RADIUS_M),
        },
        "top_candidates": [
            {
                "rank": int(rank),
                "png": f"candidate_{rank:02d}.png",
                "candidate_index": int(row["candidate_index"]),
                "score_final": float(row["score_final"]),
                "score_ab": float(row["score_ab"]),
                "penalty": float(row["penalty"]),
                "penalty_roi_std": float(row["penalty_roi_std"]),
                "penalty_neigh_mean_abs": float(row["penalty_neigh_mean_abs"]),
                "U_A": float(row["U_A"]),
                "U_B": float(row["U_B"]),
                "peak_ideal_pa": float(row["peak_ideal_pa"]),
                "peak_combined_pa": float(row["peak_combined_pa"]),
                "params": {
                    "anchor_amplitude_pa": float(row["params"]["anchor_amplitude_pa"]),
                    "source_amplitude_pa": float(row["params"]["source_amplitude_pa"]),
                    "anchor_radius_m": float(row["params"]["anchor_radius_m"]),
                    "source_radius_m": float(row["params"]["source_radius_m"]),
                },
            }
            for rank, row in enumerate(top_rows, start=1)
        ],
        "all_candidates_summary": {
            "score_final_min": float(min(r["score_final"] for r in rows_sorted)),
            "score_final_max": float(max(r["score_final"] for r in rows_sorted)),
            "score_final_mean": float(np.mean([r["score_final"] for r in rows_sorted])),
        },
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\n4) Sweep complete")
    print(f"   Wrote: {out_json}")
    print(f"   PNGs: {len(top_rows)} candidate images in {OUT_DIR}")


if __name__ == "__main__":
    main()