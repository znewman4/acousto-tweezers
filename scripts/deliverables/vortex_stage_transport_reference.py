#!/usr/bin/env python3
"""
Part 1 - Local reference scenario for translated-vortex stage transport study.

Outputs:
  results/deliverables/vortex_stage_transport/reference/
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lib.fem_cache_utils import LAM, interpolate_slice, load_fem_cache
from scripts.lib.vortex_limit_utils import (
    compute_gorkov_fields,
    detect_stable_traps,
    ensure_dir,
    load_traps_from_json,
    save_json,
    select_local_scenario,
)

OUT = ensure_dir(PROJECT_ROOT / "results" / "deliverables" / "vortex_stage_transport" / "reference")
TRAP_JSON = PROJECT_ROOT / "results" / "deliverables" / "trap_map" / "trap_data.json"
N_GRID = 400
ROI_HALF = 1.20e-3


def _mm(v: float) -> float:
    return float(v * 1e3)


def _plot_full_domain(
    p_sw: np.ndarray,
    xg: np.ndarray,
    yg: np.ndarray,
    traps_m: np.ndarray,
    idx_a: int,
    idx_b: int,
    neigh_idx: np.ndarray,
) -> None:
    ext = [xg[0] * 1e3, xg[-1] * 1e3, yg[0] * 1e3, yg[-1] * 1e3]
    a0 = traps_m[idx_a]
    b0 = traps_m[idx_b]

    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    im = ax.imshow(np.abs(p_sw), origin="lower", extent=ext, cmap="viridis", aspect="equal")

    ax.scatter(traps_m[:, 0] * 1e3, traps_m[:, 1] * 1e3, s=14, c="w", alpha=0.35, label="all traps")
    if len(neigh_idx):
        nxy = traps_m[neigh_idx]
        ax.scatter(
            nxy[:, 0] * 1e3,
            nxy[:, 1] * 1e3,
            s=48,
            marker="s",
            facecolors="none",
            edgecolors="cyan",
            linewidths=1.1,
            label="local neighbours",
        )

    ax.scatter(a0[0] * 1e3, a0[1] * 1e3, s=110, c="tomato", edgecolors="k", linewidths=0.8, label="A")
    ax.scatter(b0[0] * 1e3, b0[1] * 1e3, s=110, c="deepskyblue", edgecolors="k", linewidths=0.8, label="B")
    ax.plot([a0[0] * 1e3, b0[0] * 1e3], [a0[1] * 1e3, b0[1] * 1e3], "w--", linewidth=1.2, alpha=0.9)

    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title("Standing-wave reference at z* (full domain)")
    ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.84, label="|p_sw| [Pa]")
    fig.tight_layout()
    fig.savefig(OUT / "standing_wave_reference.png", dpi=240)
    plt.close(fig)


def _plot_roi(
    p_sw: np.ndarray,
    xg: np.ndarray,
    yg: np.ndarray,
    traps_m: np.ndarray,
    idx_a: int,
    idx_b: int,
    neigh_idx: np.ndarray,
    midpoint: np.ndarray,
) -> None:
    ext = [xg[0] * 1e3, xg[-1] * 1e3, yg[0] * 1e3, yg[-1] * 1e3]
    a0 = traps_m[idx_a]
    b0 = traps_m[idx_b]

    fig, ax = plt.subplots(figsize=(6.6, 5.8))
    im = ax.imshow(np.abs(p_sw), origin="lower", extent=ext, cmap="viridis", aspect="equal")

    ax.scatter(traps_m[:, 0] * 1e3, traps_m[:, 1] * 1e3, s=10, c="w", alpha=0.20)
    if len(neigh_idx):
        nxy = traps_m[neigh_idx]
        ax.scatter(
            nxy[:, 0] * 1e3,
            nxy[:, 1] * 1e3,
            s=50,
            marker="s",
            facecolors="none",
            edgecolors="cyan",
            linewidths=1.1,
            label="local neighbours",
        )

    ax.scatter(a0[0] * 1e3, a0[1] * 1e3, s=120, c="tomato", edgecolors="k", linewidths=0.8, label="A")
    ax.scatter(b0[0] * 1e3, b0[1] * 1e3, s=120, c="deepskyblue", edgecolors="k", linewidths=0.8, label="B")
    ax.plot([a0[0] * 1e3, b0[0] * 1e3], [a0[1] * 1e3, b0[1] * 1e3], "w--", linewidth=1.3, alpha=0.9)

    ax.set_xlim(_mm(midpoint[0] - ROI_HALF), _mm(midpoint[0] + ROI_HALF))
    ax.set_ylim(_mm(midpoint[1] - ROI_HALF), _mm(midpoint[1] + ROI_HALF))
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title("Standing-wave reference around A/B (ROI)")
    ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.84, label="|p_sw| [Pa]")
    fig.tight_layout()
    fig.savefig(OUT / "standing_wave_reference_roi.png", dpi=240)
    plt.close(fig)


def main() -> None:
    print("=" * 72)
    print("Vortex stage transport - Part 1 reference scenario")
    print("=" * 72)

    cache = load_fem_cache()
    z_star = float(cache["z_star"])

    sl = interpolate_slice(cache, z=z_star, n_grid=N_GRID)
    p_sw = sl["p_grid"]
    xg = sl["xg"]
    yg = sl["yg"]
    dx = float(sl["dx"])
    dy = float(sl["dy"])

    U_sw, Fx_sw, Fy_sw = compute_gorkov_fields(p_sw, dx, dy)

    if TRAP_JSON.exists():
        traps_m = load_traps_from_json(TRAP_JSON)
        trap_source = str(TRAP_JSON.relative_to(PROJECT_ROOT))
    else:
        traps_m = detect_stable_traps(U_sw, xg, yg, expected_spacing=LAM / 2.0)
        trap_source = "detected_from_reference_grid"

    if len(traps_m) < 2:
        raise RuntimeError("Need at least two traps for A/B scenario")

    scenario = select_local_scenario(traps_m, xg, yg, lam=LAM)
    idx_a = int(scenario["idx_A"])
    idx_b = int(scenario["idx_B"])
    midpoint = np.asarray(scenario["midpoint"], dtype=float)
    neigh_idx = np.asarray(scenario["neighbor_idx"], dtype=int)

    _plot_full_domain(p_sw, xg, yg, traps_m, idx_a, idx_b, neigh_idx)
    _plot_roi(p_sw, xg, yg, traps_m, idx_a, idx_b, neigh_idx, midpoint)

    np.savez_compressed(
        OUT / "reference_fields.npz",
        xg=xg,
        yg=yg,
        p_sw=p_sw,
        U_sw=U_sw,
        Fx_sw=Fx_sw,
        Fy_sw=Fy_sw,
        traps_m=traps_m,
        idx_A=idx_a,
        idx_B=idx_b,
        neighbor_idx=neigh_idx,
        z_star=z_star,
        lambda_m=LAM,
    )

    A0 = traps_m[idx_a]
    B0 = traps_m[idx_b]
    summary = {
        "phase": "part1_reference",
        "cache_path": str(cache["cache_path"].relative_to(PROJECT_ROOT)),
        "trap_source": trap_source,
        "z_star_mm": _mm(z_star),
        "grid": {
            "n": N_GRID,
            "dx_um": dx * 1e6,
            "dy_um": dy * 1e6,
        },
        "n_traps": int(len(traps_m)),
        "pair": {
            "idx_A": idx_a,
            "idx_B": idx_b,
            "A_mm": [_mm(float(A0[0])), _mm(float(A0[1]))],
            "B_mm": [_mm(float(B0[0])), _mm(float(B0[1]))],
            "distance_mm": _mm(float(np.linalg.norm(B0 - A0))),
            "distance_over_lambda": float(np.linalg.norm(B0 - A0) / LAM),
        },
        "neighbors": {
            "count": int(len(neigh_idx)),
            "indices": [int(v) for v in neigh_idx],
        },
        "artifacts": {
            "reference_full": "results/deliverables/vortex_stage_transport/reference/standing_wave_reference.png",
            "reference_roi": "results/deliverables/vortex_stage_transport/reference/standing_wave_reference_roi.png",
            "reference_npz": "results/deliverables/vortex_stage_transport/reference/reference_fields.npz",
        },
    }
    save_json(OUT / "reference_summary.json", summary)

    print(f"Saved {OUT / 'standing_wave_reference.png'}")
    print(f"Saved {OUT / 'standing_wave_reference_roi.png'}")
    print(f"Saved {OUT / 'reference_summary.json'}")
    print("Part 1 complete.")


if __name__ == "__main__":
    main()
