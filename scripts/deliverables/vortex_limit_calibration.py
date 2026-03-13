#!/usr/bin/env python3
"""
Part 2 - Best-case symmetric vortex calibration at z*.

Searches over aperture, source distance, beam family, and shape parameters,
then selects the smallest-radius vortex that still provides strong force.

Outputs:
  results/deliverables/vortex_limit/vortex_calibration/
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
    select_best_candidate,
)

ROOT_OUT = ensure_dir(PROJECT_ROOT / "results" / "deliverables" / "vortex_limit")
REF_DIR = ROOT_OUT / "reference"
OUT = ensure_dir(ROOT_OUT / "vortex_calibration")


APERTURE_MM = [1.8, 2.2, 2.6]
SOURCE_DISTANCE_MM = [3.0, 4.5, 6.0]

LG_WAIST_MM = [0.25, 0.40, 0.60]
LG_FOCAL_MM = [3.0, 4.5, 6.0]

BG_WAIST_MM = [0.25, 0.40, 0.60]
BG_CONE_DEG = [10.0, 15.0, 20.0]

AXICON_CONE_DEG = [10.0, 15.0, 20.0]


def _mm(v: float) -> float:
    return float(v * 1e3)


def _iter_candidates() -> list[dict]:
    rows: list[dict] = []

    for ap_mm in APERTURE_MM:
        for src_mm in SOURCE_DISTANCE_MM:
            for waist_mm in LG_WAIST_MM:
                for foc_mm in LG_FOCAL_MM:
                    rows.append(
                        {
                            "family": "lg",
                            "aperture_mm": ap_mm,
                            "source_distance_mm": src_mm,
                            "waist_mm": waist_mm,
                            "cone_deg": float("nan"),
                            "focal_mm": foc_mm,
                        }
                    )

    for ap_mm in APERTURE_MM:
        for src_mm in SOURCE_DISTANCE_MM:
            for waist_mm in BG_WAIST_MM:
                for cone_deg in BG_CONE_DEG:
                    rows.append(
                        {
                            "family": "bessel_gauss",
                            "aperture_mm": ap_mm,
                            "source_distance_mm": src_mm,
                            "waist_mm": waist_mm,
                            "cone_deg": cone_deg,
                            "focal_mm": float("nan"),
                        }
                    )

    for ap_mm in APERTURE_MM:
        for src_mm in SOURCE_DISTANCE_MM:
            for cone_deg in AXICON_CONE_DEG:
                rows.append(
                    {
                        "family": "axicon",
                        "aperture_mm": ap_mm,
                        "source_distance_mm": src_mm,
                        "waist_mm": float("nan"),
                        "cone_deg": cone_deg,
                        "focal_mm": float("nan"),
                    }
                )

    return rows


def _format_float(v: float) -> str:
    if np.isnan(v):
        return ""
    return f"{v:.6g}"


def _save_table(rows: list[dict], path: Path) -> None:
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
            w.writerow({k: _format_float(float(r[k])) if isinstance(r[k], float) else r[k] for k in fieldnames})


def _plot_calibration(rows: list[dict], selected: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    fam_color = {
        "lg": "tab:blue",
        "bessel_gauss": "tab:orange",
        "axicon": "tab:green",
    }

    ax = axes[0]
    for fam in ["lg", "bessel_gauss", "axicon"]:
        sub = [r for r in rows if r["family"] == fam]
        ax.scatter(
            [r["ring_radius_mm"] for r in sub],
            [r["peak_force"] for r in sub],
            s=28,
            alpha=0.75,
            color=fam_color[fam],
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
    ax.set_title("Radius vs force")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    for fam in ["lg", "bessel_gauss", "axicon"]:
        sub = [r for r in rows if r["family"] == fam]
        ax.scatter(
            [r["peak_abs_p"] for r in sub],
            [r["localization"] for r in sub],
            s=28,
            alpha=0.75,
            color=fam_color[fam],
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
    ax.set_xlabel("Peak |p| at z* [Pa, normalized units]")
    ax.set_ylabel("Localization metric")
    ax.set_title("Pressure peak vs localization")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle("Vortex calibration candidate map", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "candidate_map.png", dpi=220)
    plt.close(fig)


def _plot_selected_field(p_sel: np.ndarray, xg: np.ndarray, yg: np.ndarray, selected: dict) -> None:
    ext = [xg[0] * 1e3, xg[-1] * 1e3, yg[0] * 1e3, yg[-1] * 1e3]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    mag = np.abs(p_sel)

    im = axes[0].imshow(mag, origin="lower", extent=ext, cmap="inferno", aspect="equal")
    axes[0].set_title("Selected vortex |p| at z*")
    axes[0].set_xlabel("x [mm]")
    axes[0].set_ylabel("y [mm]")
    fig.colorbar(im, ax=axes[0], shrink=0.82)

    im = axes[1].imshow(np.angle(p_sel), origin="lower", extent=ext, cmap="twilight", vmin=-np.pi, vmax=np.pi, aspect="equal")
    axes[1].set_title("Selected vortex phase at z*")
    axes[1].set_xlabel("x [mm]")
    axes[1].set_ylabel("y [mm]")
    fig.colorbar(im, ax=axes[1], shrink=0.82)

    cx = 0.5 * (xg[0] + xg[-1])
    cy = 0.5 * (yg[0] + yg[-1])
    XX, YY = np.meshgrid(xg, yg)
    rr = np.sqrt((XX - cx) ** 2 + (YY - cy) ** 2)
    radial_r = np.linspace(0.0, 2.0e-3, 250)
    radial_v = []
    for r0 in radial_r:
        mask = np.abs(rr - r0) <= 0.04e-3
        if np.any(mask):
            radial_v.append(float(np.mean(mag[mask])))
        else:
            radial_v.append(np.nan)
    axes[2].plot(radial_r * 1e3, radial_v, "k-", linewidth=1.8)
    axes[2].axvline(float(selected["ring_radius_mm"]), color="tab:red", linestyle="--", linewidth=1.0)
    axes[2].set_xlabel("r [mm]")
    axes[2].set_ylabel("Azimuthal mean |p|")
    axes[2].set_title("Radial profile")
    axes[2].grid(True, alpha=0.3)

    txt = (
        f"family={selected['family']}\n"
        f"ap={selected['aperture_mm']:.2f} mm\n"
        f"source={selected['source_distance_mm']:.2f} mm\n"
        f"waist={selected['waist_mm']:.2f} mm\n"
        f"cone={selected['cone_deg']:.2f} deg\n"
        f"focal={selected['focal_mm']:.2f} mm"
    )
    axes[2].text(0.98, 0.98, txt, transform=axes[2].transAxes, va="top", ha="right", fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT / "selected_vortex.png", dpi=220)
    plt.close(fig)


def main() -> None:
    print("=" * 68)
    print("Vortex limit study - Part 2 calibration")
    print("=" * 68)

    ref_npz = REF_DIR / "reference_fields.npz"
    if not ref_npz.exists():
        raise FileNotFoundError("Run vortex_limit_reference.py first")

    ref = np.load(ref_npz)
    xg = ref["xg"]
    yg = ref["yg"]
    p_sw = ref["p_sw"]

    dx = float(xg[1] - xg[0])
    dy = float(yg[1] - yg[0])
    XX, YY = np.meshgrid(xg, yg)

    candidates = _iter_candidates()
    print(f"Scanning {len(candidates)} symmetric-vortex candidates...")

    rows: list[dict] = []
    best_field: np.ndarray | None = None
    fields_for_id: dict[int, np.ndarray] = {}

    for i, c in enumerate(candidates):
        source = build_symmetric_vortex_source(
            XX,
            YY,
            family=c["family"],
            aperture_radius=float(c["aperture_mm"]) * 1e-3,
            waist=None if np.isnan(c["waist_mm"]) else float(c["waist_mm"]) * 1e-3,
            cone_angle_deg=None if np.isnan(c["cone_deg"]) else float(c["cone_deg"]),
            focal_length=None if np.isnan(c["focal_mm"]) else float(c["focal_mm"]) * 1e-3,
            charge=1,
        )
        pz = propagate_candidate(
            source,
            dx,
            dy,
            source_distance=float(c["source_distance_mm"]) * 1e-3,
        )

        vm = evaluate_vortex_metrics(pz, xg, yg)
        _, Fx, Fy = compute_gorkov_fields(pz, dx, dy)
        peak_force = float(np.max(np.sqrt(Fx * Fx + Fy * Fy)))

        row = {
            "candidate_id": i,
            **c,
            "ring_radius_mm": float(vm["ring_radius_m"]) * 1e3,
            "peak_abs_p": float(vm["peak_abs_p"]),
            "localization": float(vm["localization"]),
            "central_ratio": float(vm["central_ratio"]),
            "ring_concentration": float(vm["ring_concentration"]),
            "peak_force": peak_force,
            "ring_radius_m": float(vm["ring_radius_m"]),
        }
        rows.append(row)
        fields_for_id[i] = pz

        if (i + 1) % 20 == 0 or i == len(candidates) - 1:
            print(f"  done {i + 1}/{len(candidates)}")

    selected = select_best_candidate(rows)
    sel_id = int(selected["candidate_id"])
    best_field = fields_for_id[sel_id]

    rows_sorted = sorted(rows, key=lambda r: (r["ring_radius_mm"], -r["peak_force"]))
    top_rows = rows_sorted[:15]

    _save_table(rows, OUT / "candidate_metrics.csv")
    save_json(OUT / "candidate_metrics.json", {"rows": rows})

    _plot_calibration(rows, selected)
    _plot_selected_field(best_field, xg, yg, selected)

    np.savez_compressed(
        OUT / "selected_vortex_field.npz",
        xg=xg,
        yg=yg,
        p_vortex=best_field,
    )

    sw_peak = float(np.max(np.abs(p_sw)))
    vortex_peak = float(np.max(np.abs(best_field)))

    summary = {
        "phase": "part2_calibration",
        "n_candidates": int(len(rows)),
        "selection_rule": "minimum ring radius subject to >=60% of max peak force",
        "selected": {
            "family": selected["family"],
            "aperture_mm": float(selected["aperture_mm"]),
            "source_distance_mm": float(selected["source_distance_mm"]),
            "waist_mm": float(selected["waist_mm"]),
            "cone_deg": float(selected["cone_deg"]),
            "focal_mm": float(selected["focal_mm"]),
            "ring_radius_mm": float(selected["ring_radius_mm"]),
            "peak_abs_p": float(selected["peak_abs_p"]),
            "peak_force": float(selected["peak_force"]),
            "localization": float(selected["localization"]),
        },
        "smallest_waist_tested_mm": min(LG_WAIST_MM + BG_WAIST_MM),
        "pressure_scaling_context": {
            "sw_peak_abs_p": sw_peak,
            "selected_vortex_peak_abs_p": vortex_peak,
            "selected_over_sw_peak": vortex_peak / max(sw_peak, 1e-30),
        },
        "top_candidates": [
            {
                "family": r["family"],
                "aperture_mm": float(r["aperture_mm"]),
                "source_distance_mm": float(r["source_distance_mm"]),
                "waist_mm": float(r["waist_mm"]),
                "cone_deg": float(r["cone_deg"]),
                "focal_mm": float(r["focal_mm"]),
                "ring_radius_mm": float(r["ring_radius_mm"]),
                "peak_force": float(r["peak_force"]),
            }
            for r in top_rows
        ],
        "artifacts": {
            "candidate_csv": "results/deliverables/vortex_limit/vortex_calibration/candidate_metrics.csv",
            "candidate_json": "results/deliverables/vortex_limit/vortex_calibration/candidate_metrics.json",
            "candidate_map_png": "results/deliverables/vortex_limit/vortex_calibration/candidate_map.png",
            "selected_png": "results/deliverables/vortex_limit/vortex_calibration/selected_vortex.png",
            "selected_field_npz": "results/deliverables/vortex_limit/vortex_calibration/selected_vortex_field.npz",
        },
    }
    save_json(OUT / "calibration_summary.json", summary)

    print(f"Selected family={selected['family']} ring={selected['ring_radius_mm']:.3f} mm")
    print(f"Saved {OUT / 'calibration_summary.json'}")
    print("Part 2 complete.")


if __name__ == "__main__":
    main()
