#!/usr/bin/env python3
"""
Part 1 - Local reference scenario for symmetric-vortex limit study.

Outputs:
  results/deliverables/vortex_limit/reference/
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

from scripts.lib.fem_cache_utils import LAM, interpolate_slice, load_fem_cache
from scripts.lib.vortex_limit_utils import (
    compute_gorkov_fields,
    detect_stable_traps,
    ensure_dir,
    load_traps_from_json,
    save_json,
    select_local_scenario,
)

OUT = ensure_dir(PROJECT_ROOT / "results" / "deliverables" / "vortex_limit" / "reference")
TRAP_JSON = PROJECT_ROOT / "results" / "deliverables" / "trap_map" / "trap_data.json"

N_GRID = 400


def _mm(v: float) -> float:
    return float(v * 1e3)


def _plot_reference(
    p_sw: np.ndarray,
    U_sw: np.ndarray,
    xg: np.ndarray,
    yg: np.ndarray,
    traps_m: np.ndarray,
    idx_a: int,
    idx_b: int,
    neigh_idx: np.ndarray,
    midpoint: np.ndarray,
) -> None:
    x_mm = xg * 1e3
    y_mm = yg * 1e3
    ext = [x_mm[0], x_mm[-1], y_mm[0], y_mm[-1]]

    a_xy = traps_m[idx_a]
    b_xy = traps_m[idx_b]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    ax = axes[0]
    im = ax.imshow(np.abs(p_sw), origin="lower", extent=ext, cmap="viridis", aspect="equal")
    ax.scatter(traps_m[:, 0] * 1e3, traps_m[:, 1] * 1e3, s=14, c="w", alpha=0.45)
    ax.scatter(a_xy[0] * 1e3, a_xy[1] * 1e3, s=90, c="red", edgecolors="k", linewidths=0.8, label="A")
    ax.scatter(b_xy[0] * 1e3, b_xy[1] * 1e3, s=90, c="deepskyblue", edgecolors="k", linewidths=0.8, label="B")
    if len(neigh_idx):
        nxy = traps_m[neigh_idx]
        ax.scatter(nxy[:, 0] * 1e3, nxy[:, 1] * 1e3, s=45, marker="s", facecolors="none", edgecolors="cyan", linewidths=1.0, label="neighbors")
    ax.plot([a_xy[0] * 1e3, b_xy[0] * 1e3], [a_xy[1] * 1e3, b_xy[1] * 1e3], "w--", linewidth=1.0)
    ax.set_title("Standing wave magnitude at z*")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.82, label="|p| [Pa]")

    ax = axes[1]
    im = ax.imshow(U_sw, origin="lower", extent=ext, cmap="RdBu_r", aspect="equal")
    ax.scatter(a_xy[0] * 1e3, a_xy[1] * 1e3, s=90, c="red", edgecolors="k", linewidths=0.8)
    ax.scatter(b_xy[0] * 1e3, b_xy[1] * 1e3, s=90, c="deepskyblue", edgecolors="k", linewidths=0.8)
    ax.plot([a_xy[0] * 1e3, b_xy[0] * 1e3], [a_xy[1] * 1e3, b_xy[1] * 1e3], "k--", linewidth=1.0)
    ax.set_title("Gorkov potential at z*")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    fig.colorbar(im, ax=ax, shrink=0.82, label="U [J]")

    ax = axes[2]
    roi_half = 1.1e-3
    mx, my = midpoint
    ax.imshow(np.abs(p_sw), origin="lower", extent=ext, cmap="viridis", aspect="equal")
    ax.scatter(a_xy[0] * 1e3, a_xy[1] * 1e3, s=90, c="red", edgecolors="k", linewidths=0.8)
    ax.scatter(b_xy[0] * 1e3, b_xy[1] * 1e3, s=90, c="deepskyblue", edgecolors="k", linewidths=0.8)
    if len(neigh_idx):
        nxy = traps_m[neigh_idx]
        ax.scatter(nxy[:, 0] * 1e3, nxy[:, 1] * 1e3, s=45, marker="s", facecolors="none", edgecolors="cyan", linewidths=1.0)
    ax.set_xlim(_mm(mx - roi_half), _mm(mx + roi_half))
    ax.set_ylim(_mm(my - roi_half), _mm(my + roi_half))
    ax.set_title("ROI zoom around A/B")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")

    fig.suptitle("Vortex limit study reference scenario", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "reference_overview.png", dpi=220)
    plt.close(fig)


def main() -> None:
    print("=" * 68)
    print("Vortex limit study - Part 1 reference scenario")
    print("=" * 68)

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
        print(f"Loaded traps from {trap_source}: n={len(traps_m)}")
    else:
        traps_m = detect_stable_traps(U_sw, xg, yg, expected_spacing=LAM / 2.0)
        trap_source = "detected_from_reference_grid"
        print(f"Detected traps on reference grid: n={len(traps_m)}")

    if len(traps_m) < 2:
        raise RuntimeError("Need at least two traps for A/B scenario")

    scenario = select_local_scenario(traps_m, xg, yg, lam=LAM)
    idx_a = int(scenario["idx_A"])
    idx_b = int(scenario["idx_B"])
    a_xy = np.asarray(scenario["A_xy"], dtype=float)
    b_xy = np.asarray(scenario["B_xy"], dtype=float)
    midpoint = np.asarray(scenario["midpoint"], dtype=float)
    neigh_idx = np.asarray(scenario["neighbor_idx"], dtype=int)

    _plot_reference(p_sw, U_sw, xg, yg, traps_m, idx_a, idx_b, neigh_idx, midpoint)

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

    summary = {
        "phase": "part1_reference",
        "cache_path": str(cache["cache_path"].relative_to(PROJECT_ROOT)),
        "z_star_mm": _mm(z_star),
        "grid": {
            "n": N_GRID,
            "dx_um": dx * 1e6,
            "dy_um": dy * 1e6,
        },
        "trap_source": trap_source,
        "n_traps": int(len(traps_m)),
        "pair": {
            "idx_A": idx_a,
            "idx_B": idx_b,
            "A_mm": [_mm(a_xy[0]), _mm(a_xy[1])],
            "B_mm": [_mm(b_xy[0]), _mm(b_xy[1])],
            "distance_mm": _mm(np.linalg.norm(b_xy - a_xy)),
            "distance_over_lambda": float(np.linalg.norm(b_xy - a_xy) / LAM),
        },
        "neighbors": {
            "count": int(len(neigh_idx)),
            "indices": [int(v) for v in neigh_idx],
        },
        "artifacts": {
            "overview_png": "results/deliverables/vortex_limit/reference/reference_overview.png",
            "fields_npz": "results/deliverables/vortex_limit/reference/reference_fields.npz",
        },
    }
    save_json(OUT / "reference_summary.json", summary)

    print(f"Saved {OUT / 'reference_overview.png'}")
    print(f"Saved {OUT / 'reference_fields.npz'}")
    print(f"Saved {OUT / 'reference_summary.json'}")
    print("Part 1 complete.")


if __name__ == "__main__":
    main()
