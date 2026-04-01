#!/usr/bin/env python3
"""
MPC Vortex Parameter Sweep — evaluates MPC merge performance across
vortex waist and topological charge combinations.

Outputs:
    results/dev/mpc_vortex_sweep/<timestamp>/
        sweep_results.json        — full results per configuration
        sweep_summary.csv         — tabular summary
        sweep_heatmap.png         — merge-success vs parameters

Usage:
    python scripts/dev/mpc_vortex_param_sweep.py
    python scripts/dev/mpc_vortex_param_sweep.py --waists 0.2 0.3 0.4 --charges 0 1 2
    python scripts/dev/mpc_vortex_param_sweep.py --T 1500 --K 8 --n_iters 15
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.lib.asm_utils import make_vortex_field
from scripts.lib.fem_cache_utils import (
    C_WATER,
    F_HZ,
    OMEGA,
    RHO0,
    default_particle_params,
)
from scripts.lib.mpc_controller import (
    ForceEvaluator,
    MPCConfig,
    mpc_result_to_transport,
    run_mpc,
)
from scripts.lib.particle_dynamics_utils import (
    CAPTURE_RADIUS,
    LAM,
    TransportResult,
    compute_metrics,
)
from scripts.lib.perturbation_vortex import VortexPerturbation

# ── Data paths ────────────────────────────────────────────────────

VORTEX_NPZ = (
    PROJECT_ROOT / "results" / "deliverables" / "vortex_stage_transport"
    / "transport" / "transport_case_for_gif.npz"
)
OVERLAY_NPZ = (
    PROJECT_ROOT / "results" / "c_shape_lens_15mm_overlay_study_20260310_170620"
    / "npz" / "roi_fields.npz"
)

VORTEX_APERTURE = 0.8e-3


def run_single_case(
    p_sw: np.ndarray,
    xg: np.ndarray,
    yg: np.ndarray,
    traps_m: np.ndarray,
    idx_A: int,
    idx_B: int,
    neigh_idx: np.ndarray,
    charge: int,
    waist_m: float,
    cfg: MPCConfig,
) -> Dict[str, Any]:
    """Run MPC for one (charge, waist) setting and return metrics."""
    domain_centre = np.array([0.5 * (xg[0] + xg[-1]), 0.5 * (yg[0] + yg[-1])])
    XX, YY = np.meshgrid(xg, yg)
    p_vortex = make_vortex_field(
        XX, YY,
        charge=charge,
        waist=waist_m,
        center=tuple(domain_centre),
        aperture_radius=VORTEX_APERTURE,
    )
    vortex_gen = VortexPerturbation(p_vortex, xg, yg, out_xg=xg, out_yg=yg)
    feval = ForceEvaluator(p_sw, vortex_gen, xg, yg)

    u0 = np.array([0.0, traps_m[idx_A, 0], traps_m[idx_A, 1], 5.0, 1.0])
    u0 = np.clip(u0, cfg.u_lo, cfg.u_hi)

    t0 = time.perf_counter()
    result = run_mpc(
        feval=feval,
        x0=traps_m.copy(),
        u_init=u0,
        idx_A=idx_A,
        idx_B=idx_B,
        neigh_idx=neigh_idx,
        target_pos=traps_m.copy(),
        cfg=cfg,
        verbose=False,
    )
    elapsed = time.perf_counter() - t0

    transport = mpc_result_to_transport(result, traps_m, cfg.dt)
    metrics = compute_metrics(transport, idx_A, idx_B, neigh_idx)

    return {
        "charge": charge,
        "waist_mm": round(waist_m * 1e3, 3),
        "elapsed_s": round(elapsed, 2),
        "merge_time_s": result.merge_time_s,
        "n_replans": len(result.J_history),
        "final_J": result.J_history[-1] if result.J_history else None,
        **metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="MPC Vortex Parameter Sweep")
    parser.add_argument("--waists", type=float, nargs="+",
                        default=[0.20, 0.30, 0.35, 0.40, 0.50, 0.60],
                        help="Vortex waists to sweep [mm]")
    parser.add_argument("--charges", type=int, nargs="+",
                        default=[0, 1, 2],
                        help="Topological charges to sweep")
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--T", type=int, default=2000)
    parser.add_argument("--n_iters", type=int, default=15)
    parser.add_argument("--replan_every", type=int, default=5)
    parser.add_argument("--n_particles", type=int, default=5)
    args = parser.parse_args()

    # ── Load data ─────────────────────────────────────────────────
    print("Loading data...")
    ov = np.load(OVERLAY_NPZ)
    traps_m_all = ov["traps_m"].astype(float)
    idx_A_orig = int(ov["idx_A"])
    idx_B_orig = int(ov["idx_B"])

    vd = np.load(VORTEX_NPZ)
    xg = vd["xg"].astype(float)
    yg = vd["yg"].astype(float)
    p_sw = vd["p_sw"].astype(complex)

    # Select particles
    A_xy = traps_m_all[idx_A_orig]
    B_xy = traps_m_all[idx_B_orig]
    n_part = min(max(args.n_particles, 2), len(traps_m_all))

    if n_part >= len(traps_m_all):
        particle_indices = np.arange(len(traps_m_all))
    else:
        dists = np.linalg.norm(traps_m_all - 0.5 * (A_xy + B_xy), axis=1)
        order = np.argsort(dists)
        selected = {idx_A_orig, idx_B_orig}
        for idx in order:
            if len(selected) >= n_part:
                break
            selected.add(idx)
        particle_indices = np.array(sorted(selected))

    traps_m = traps_m_all[particle_indices]
    idx_A = int(np.where(particle_indices == idx_A_orig)[0][0])
    idx_B = int(np.where(particle_indices == idx_B_orig)[0][0])
    neigh_idx = np.array([i for i in range(len(traps_m)) if i not in (idx_A, idx_B)], dtype=int)

    midpoint = 0.5 * (traps_m[idx_A] + traps_m[idx_B])
    corridor_margin = 2.0 * LAM

    cfg = MPCConfig(
        K=args.K,
        T=args.T,
        n_iters=args.n_iters,
        replan_every=args.replan_every,
    )
    cfg.u_lo = np.array([
        -4.0 * np.pi,
        midpoint[0] - corridor_margin,
        midpoint[1] - corridor_margin,
        0.0, 0.5,
    ])
    cfg.u_hi = np.array([
        4.0 * np.pi,
        midpoint[0] + corridor_margin,
        midpoint[1] + corridor_margin,
        15.0, 1.0,
    ])

    # ── Run sweep ─────────────────────────────────────────────────
    n_total = len(args.waists) * len(args.charges)
    results: List[Dict[str, Any]] = []
    print(f"\nSweep: {len(args.waists)} waists × {len(args.charges)} charges = {n_total} cases")
    print(f"Particles: {len(traps_m)} (A={idx_A}, B={idx_B})")

    for ci, charge in enumerate(args.charges):
        for wi, waist_mm in enumerate(args.waists):
            case_num = ci * len(args.waists) + wi + 1
            print(f"\n[{case_num}/{n_total}] charge={charge}, waist={waist_mm:.2f} mm")

            try:
                res = run_single_case(
                    p_sw, xg, yg, traps_m, idx_A, idx_B, neigh_idx,
                    charge, waist_mm * 1e-3, cfg,
                )
                results.append(res)
                cls = res.get("classification", "?")
                d = res.get("d_AB_final_um", "?")
                print(f"  → {cls}, d_AB={d} µm, t={res['elapsed_s']:.1f}s")
            except Exception as e:
                print(f"  → FAILED: {e}")
                results.append({
                    "charge": charge,
                    "waist_mm": waist_mm,
                    "error": str(e),
                    "classification": "error",
                })

    # ── Output ────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "results" / "dev" / "mpc_vortex_sweep" / f"sweep_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "sweep_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # CSV summary
    if results:
        keys = results[0].keys()
        with open(out_dir / "sweep_summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(keys))
            w.writeheader()
            for r in results:
                w.writerow(r)

    # Heatmap
    charges_arr = sorted(set(r.get("charge", 0) for r in results))
    waists_arr = sorted(set(r.get("waist_mm", 0) for r in results))

    if len(charges_arr) > 1 and len(waists_arr) > 1:
        grid = np.full((len(charges_arr), len(waists_arr)), np.nan)
        for r in results:
            if "error" in r:
                continue
            ci_idx = charges_arr.index(r["charge"])
            wi_idx = waists_arr.index(r["waist_mm"])
            d = r.get("d_AB_final_um", np.nan)
            grid[ci_idx, wi_idx] = float(d) if d is not None else np.nan

        fig, ax = plt.subplots(figsize=(8, 4))
        im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis_r")
        ax.set_xticks(range(len(waists_arr)))
        ax.set_xticklabels([f"{w:.2f}" for w in waists_arr])
        ax.set_yticks(range(len(charges_arr)))
        ax.set_yticklabels([str(c) for c in charges_arr])
        ax.set_xlabel("Vortex waist [mm]")
        ax.set_ylabel("Topological charge ℓ")
        ax.set_title("MPC Merge — final d(A,B) [µm]")
        plt.colorbar(im, ax=ax, label="d(A,B) [µm]")

        # Annotate success
        for i in range(len(charges_arr)):
            for j in range(len(waists_arr)):
                val = grid[i, j]
                if not np.isnan(val):
                    color = "white" if val > np.nanmedian(grid) else "black"
                    ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                            fontsize=8, color=color)

        fig.tight_layout()
        fig.savefig(str(out_dir / "sweep_heatmap.png"), dpi=150)
        plt.close(fig)

    print(f"\nSweep complete. Results saved to: {out_dir}")

    # Summary
    successes = [r for r in results if r.get("classification") == "successful_merge"]
    print(f"Successes: {len(successes)}/{len(results)}")
    if successes:
        best = min(successes, key=lambda r: r.get("d_AB_final_um", float("inf")))
        print(f"Best: charge={best['charge']}, waist={best['waist_mm']} mm, "
              f"d_AB={best.get('d_AB_final_um', '?')} µm")


if __name__ == "__main__":
    main()
