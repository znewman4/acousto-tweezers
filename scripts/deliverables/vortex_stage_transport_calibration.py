#!/usr/bin/env python3
"""
Part 2 - Best-case symmetric vortex calibration for translated-vortex transport.

Outputs:
  results/deliverables/vortex_stage_transport/calibration/
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

from scripts.lib.vortex_limit_utils import (
    build_symmetric_vortex_source,
    compute_gorkov_fields,
    ensure_dir,
    evaluate_vortex_metrics,
    propagate_candidate,
    save_json,
)

ROOT_OUT = ensure_dir(PROJECT_ROOT / "results" / "deliverables" / "vortex_stage_transport")
REF_DIR = ROOT_OUT / "reference"
OUT = ensure_dir(ROOT_OUT / "calibration")

APERTURE_MM = [1.00, 1.30, 1.60, 2.00, 2.40]
SOURCE_DISTANCE_MM = [1.8, 2.4, 3.0, 4.0]

LG_WAIST_MM = [0.12, 0.18, 0.25, 0.35]
LG_FOCAL_MM = [1.8, 2.4, 3.0, 4.0]

BG_WAIST_MM = [0.12, 0.18, 0.25, 0.35]
BG_CONE_DEG = [10.0, 15.0, 20.0, 25.0]

AXICON_CONE_DEG = [10.0, 15.0, 20.0, 25.0]

FORCE_FLOOR_FRAC = 0.45


def _iter_candidates() -> list[dict]:
    rows: list[dict] = []

    for ap in APERTURE_MM:
        for src in SOURCE_DISTANCE_MM:
            for waist in LG_WAIST_MM:
                for focal in LG_FOCAL_MM:
                    rows.append(
                        {
                            "family": "lg",
                            "aperture_mm": ap,
                            "source_distance_mm": src,
                            "waist_mm": waist,
                            "cone_deg": float("nan"),
                            "focal_mm": focal,
                        }
                    )

    for ap in APERTURE_MM:
        for src in SOURCE_DISTANCE_MM:
            for waist in BG_WAIST_MM:
                for cone in BG_CONE_DEG:
                    rows.append(
                        {
                            "family": "bessel_gauss",
                            "aperture_mm": ap,
                            "source_distance_mm": src,
                            "waist_mm": waist,
                            "cone_deg": cone,
                            "focal_mm": float("nan"),
                        }
                    )

    for ap in APERTURE_MM:
        for src in SOURCE_DISTANCE_MM:
            for cone in AXICON_CONE_DEG:
                rows.append(
                    {
                        "family": "axicon",
                        "aperture_mm": ap,
                        "source_distance_mm": src,
                        "waist_mm": float("nan"),
                        "cone_deg": cone,
                        "focal_mm": float("nan"),
                    }
                )

    return rows


def _fmt(v: float) -> str:
    if np.isnan(v):
        return ""
    return f"{v:.6g}"


def _save_candidate_table(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "family",
        "aperture_mm",
        "source_distance_mm",
        "waist_mm",
        "cone_deg",
        "focal_mm",
        "ring_radius_mm",
        "peak_abs_p",
        "localization",
        "central_ratio",
        "ring_concentration",
        "peak_force",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(float(r[k])) if isinstance(r[k], float) else r[k] for k in fieldnames})


def _plot_sweep(rows: list[dict], selected: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.3))
    colors = {"lg": "tab:blue", "bessel_gauss": "tab:orange", "axicon": "tab:green"}

    ax = axes[0]
    for fam in ["lg", "bessel_gauss", "axicon"]:
        sub = [r for r in rows if r["family"] == fam]
        ax.scatter(
            [r["ring_radius_mm"] for r in sub],
            [r["peak_force"] for r in sub],
            s=26,
            color=colors[fam],
            alpha=0.72,
            label=fam,
        )
    ax.scatter(
        [selected["ring_radius_mm"]],
        [selected["peak_force"]],
        s=140,
        marker="*",
        color="red",
        edgecolors="k",
        linewidths=0.8,
        label="selected",
    )
    ax.set_xlabel("Ring radius at z* [mm]")
    ax.set_ylabel("Peak |F| from vortex-only Gor'kov [N]")
    ax.set_title("Ring radius vs force")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    for fam in ["lg", "bessel_gauss", "axicon"]:
        sub = [r for r in rows if r["family"] == fam]
        ax.scatter(
            [r["peak_abs_p"] for r in sub],
            [r["localization"] for r in sub],
            s=26,
            color=colors[fam],
            alpha=0.72,
            label=fam,
        )
    ax.scatter(
        [selected["peak_abs_p"]],
        [selected["localization"]],
        s=140,
        marker="*",
        color="red",
        edgecolors="k",
        linewidths=0.8,
        label="selected",
    )
    ax.set_xlabel("Peak |p| at z* [arb]")
    ax.set_ylabel("Localization metric")
    ax.set_title("Peak pressure vs localization")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle("Symmetric vortex candidate sweep", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "vortex_candidate_sweep.png", dpi=240)
    plt.close(fig)


def _local_spacing_metrics(traps_m: np.ndarray, idx_a: int, idx_b: int, neigh_idx: np.ndarray) -> dict:
    local_idx = np.unique(np.concatenate([[int(idx_a), int(idx_b)], np.asarray(neigh_idx, dtype=int)]))
    local_pts = traps_m[local_idx]

    if len(local_pts) < 2:
        return {
            "nearest_neighbor_spacing_mm": float("nan"),
            "median_neighbor_spacing_mm": float("nan"),
            "local_count": int(len(local_pts)),
        }

    d = np.linalg.norm(local_pts[:, None, :] - local_pts[None, :, :], axis=2)
    d[d <= 0.0] = np.inf
    nearest = np.min(d, axis=1)

    return {
        "nearest_neighbor_spacing_mm": float(np.min(nearest) * 1e3),
        "median_neighbor_spacing_mm": float(np.median(nearest) * 1e3),
        "local_count": int(len(local_pts)),
    }


def _select_smallest_local(rows: list[dict]) -> tuple[dict, dict]:
    if not rows:
        raise ValueError("No calibration rows")

    peak_force_max = max(float(r["peak_force"]) for r in rows)
    force_floor = FORCE_FLOOR_FRAC * peak_force_max

    finite = [r for r in rows if np.isfinite(float(r["ring_radius_m"]))]
    feasible = [r for r in finite if float(r["peak_force"]) >= force_floor]
    if not feasible:
        feasible = finite
    if not feasible:
        feasible = rows

    chosen = sorted(
        feasible,
        key=lambda r: (
            float(r["ring_radius_m"]),
            -float(r["peak_force"]),
            -float(r["localization"]),
        ),
    )[0]

    debug = {
        "peak_force_max": float(peak_force_max),
        "force_floor_fraction": float(FORCE_FLOOR_FRAC),
        "force_floor": float(force_floor),
        "n_total": int(len(rows)),
        "n_finite_ring": int(len(finite)),
        "n_feasible": int(len(feasible)),
    }
    return dict(chosen), debug


def main() -> None:
    print("=" * 72)
    print("Vortex stage transport - Part 2 calibration")
    print("=" * 72)

    ref_npz = REF_DIR / "reference_fields.npz"
    ref_sum = REF_DIR / "reference_summary.json"
    if not ref_npz.exists() or not ref_sum.exists():
        raise FileNotFoundError("Run vortex_stage_transport_reference.py first")

    ref = np.load(ref_npz)
    ref_summary = json.loads(ref_sum.read_text())

    xg = ref["xg"]
    yg = ref["yg"]
    p_sw = ref["p_sw"]
    traps_m = ref["traps_m"]
    idx_a = int(ref["idx_A"])
    idx_b = int(ref["idx_B"])
    neigh_idx = ref["neighbor_idx"].astype(int)
    dx = float(xg[1] - xg[0])
    dy = float(yg[1] - yg[0])

    XX, YY = np.meshgrid(xg, yg)
    candidates = _iter_candidates()
    print(f"Scanning {len(candidates)} vortex candidates...")

    rows: list[dict] = []

    for i, c in enumerate(candidates):
        src = build_symmetric_vortex_source(
            XX,
            YY,
            family=c["family"],
            aperture_radius=float(c["aperture_mm"]) * 1e-3,
            waist=None if np.isnan(c["waist_mm"]) else float(c["waist_mm"]) * 1e-3,
            cone_angle_deg=None if np.isnan(c["cone_deg"]) else float(c["cone_deg"]),
            focal_length=None if np.isnan(c["focal_mm"]) else float(c["focal_mm"]) * 1e-3,
            charge=1,
        )
        pz = propagate_candidate(src, dx, dy, source_distance=float(c["source_distance_mm"]) * 1e-3)

        vm = evaluate_vortex_metrics(pz, xg, yg)
        _, Fx, Fy = compute_gorkov_fields(pz, dx, dy)
        peak_force = float(np.max(np.sqrt(Fx * Fx + Fy * Fy)))

        row = {
            "candidate_id": i,
            **c,
            "ring_radius_mm": float(vm["ring_radius_m"]) * 1e3,
            "ring_radius_m": float(vm["ring_radius_m"]),
            "peak_abs_p": float(vm["peak_abs_p"]),
            "localization": float(vm["localization"]),
            "central_ratio": float(vm["central_ratio"]),
            "ring_concentration": float(vm["ring_concentration"]),
            "peak_force": peak_force,
        }
        rows.append(row)

        if (i + 1) % 25 == 0 or i == len(candidates) - 1:
            print(f"  done {i + 1}/{len(candidates)}")

    selected, select_debug = _select_smallest_local(rows)
    src_sel = build_symmetric_vortex_source(
        XX,
        YY,
        family=str(selected["family"]),
        aperture_radius=float(selected["aperture_mm"]) * 1e-3,
        waist=None if np.isnan(float(selected["waist_mm"])) else float(selected["waist_mm"]) * 1e-3,
        cone_angle_deg=None if np.isnan(float(selected["cone_deg"])) else float(selected["cone_deg"]),
        focal_length=None if np.isnan(float(selected["focal_mm"])) else float(selected["focal_mm"]) * 1e-3,
        charge=1,
    )
    p_sel = propagate_candidate(src_sel, dx, dy, source_distance=float(selected["source_distance_mm"]) * 1e-3)

    _save_candidate_table(rows, OUT / "candidate_metrics.csv")
    save_json(OUT / "candidate_metrics.json", {"rows": rows})
    _plot_sweep(rows, selected)

    np.savez_compressed(OUT / "selected_vortex_field.npz", xg=xg, yg=yg, p_vortex=p_sel)

    sw_peak = float(np.max(np.abs(p_sw)))
    vortex_peak = float(np.max(np.abs(p_sel)))
    ab_mm = float(ref_summary["pair"]["distance_mm"])
    spacing = _local_spacing_metrics(traps_m, idx_a, idx_b, neigh_idx)
    nn_mm = float(spacing["nearest_neighbor_spacing_mm"])
    lam_mm = float(ref["lambda_m"] * 1e3)
    ring_mm = float(selected["ring_radius_mm"])
    ring_diam_mm = 2.0 * ring_mm

    ring_over_nn = ring_mm / max(nn_mm, 1e-30)
    diam_over_nn = ring_diam_mm / max(nn_mm, 1e-30)
    spans_many = bool(diam_over_nn >= 1.5)

    if spans_many:
        nonlocal_statement = (
            "Minimum feasible symmetric vortex remains broad relative to local trap spacing "
            "and is expected to perturb multiple nearby traps."
        )
    else:
        nonlocal_statement = (
            "Minimum feasible symmetric vortex is locally compact enough to be competitive with "
            "single-trap targeting in this geometry."
        )

    summary = {
        "phase": "part2_calibration",
        "n_candidates": int(len(rows)),
        "selection_rule": "minimum ring radius subject to >=45% max peak force",
        "selection_debug": select_debug,
        "selected": {
            "family": selected["family"],
            "aperture_mm": float(selected["aperture_mm"]),
            "source_distance_mm": float(selected["source_distance_mm"]),
            "waist_mm": float(selected["waist_mm"]),
            "cone_deg": float(selected["cone_deg"]),
            "focal_mm": float(selected["focal_mm"]),
            "ring_radius_mm": ring_mm,
            "peak_abs_p": float(selected["peak_abs_p"]),
            "peak_force": float(selected["peak_force"]),
            "localization": float(selected["localization"]),
        },
        "footprint": {
            "ring_radius_mm": ring_mm,
            "ring_diameter_mm": ring_diam_mm,
            "A_B_spacing_mm": ab_mm,
            "nearest_neighbor_spacing_mm": nn_mm,
            "median_neighbor_spacing_mm": float(spacing["median_neighbor_spacing_mm"]),
            "lambda_mm": lam_mm,
            "ring_over_A_B": ring_mm / max(ab_mm, 1e-30),
            "ring_diameter_over_A_B": ring_diam_mm / max(ab_mm, 1e-30),
            "ring_over_nearest_neighbor": ring_over_nn,
            "ring_diameter_over_nearest_neighbor": diam_over_nn,
            "ring_over_lambda": ring_mm / max(lam_mm, 1e-30),
            "ring_diameter_over_lambda": ring_diam_mm / max(lam_mm, 1e-30),
        },
        "feasibility_statement": {
            "spans_too_many_traps": spans_many,
            "statement": nonlocal_statement,
        },
        "ring_context": {
            "A_B_spacing_mm": ab_mm,
            "lambda_mm": lam_mm,
            "ring_over_AB": ring_mm / max(ab_mm, 1e-30),
            "ring_over_lambda": ring_mm / max(lam_mm, 1e-30),
        },
        "pressure_scaling_context": {
            "sw_peak_abs_p": sw_peak,
            "selected_vortex_peak_abs_p": vortex_peak,
            "selected_over_sw_peak": vortex_peak / max(sw_peak, 1e-30),
        },
        "artifacts": {
            "sweep_png": "results/deliverables/vortex_stage_transport/calibration/vortex_candidate_sweep.png",
            "selected_field_npz": "results/deliverables/vortex_stage_transport/calibration/selected_vortex_field.npz",
            "selected_summary_json": "results/deliverables/vortex_stage_transport/calibration/selected_vortex_summary.json",
        },
    }
    save_json(OUT / "selected_vortex_summary.json", summary)

    print(
        f"Selected family={selected['family']} ap={selected['aperture_mm']:.2f} mm "
        f"source={selected['source_distance_mm']:.2f} mm ring={ring_mm:.3f} mm"
    )
    print(f"Nearest-neighbour spacing={nn_mm:.3f} mm | ring diameter={ring_diam_mm:.3f} mm")
    print(nonlocal_statement)
    print(f"Saved {OUT / 'selected_vortex_summary.json'}")
    print("Part 2 complete.")


if __name__ == "__main__":
    main()
