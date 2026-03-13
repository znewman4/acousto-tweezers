#!/usr/bin/env python3
"""
Parts 3-6 - Rebuilt translated-vortex transport with A->B objective.

Primary objective in this rebuild:
- find the minimum alpha that gets particle A into B's original trap location,
- translate vortex centre smoothly,
- then ramp vortex off slowly once A reaches B trap.

Required outputs (root study folder):
  start_end_standing_wave_comparison.png
  transport_full_domain_sequence.png
  transport_roi_sequence.png
  transport_metrics.csv
  transport_summary.json
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lib.vortex_limit_utils import ensure_dir, save_json
from scripts.lib.vortex_stage_transport_utils import (
    PHASE_RAMP_OFF,
    PHASE_RAMP_ON,
    PHASE_SW_RELEASE,
    PHASE_SW_START,
    PHASE_TRANSLATE,
    StagedTransportResult,
    build_staged_schedule,
    mm,
    phase_label,
    phase_representative_indices,
    run_staged_transport,
    shift_complex_field,
)

ROOT_OUT = ensure_dir(PROJECT_ROOT / "results" / "deliverables" / "vortex_stage_transport")
REF_DIR = ROOT_OUT / "reference"
CAL_DIR = ROOT_OUT / "calibration"
OUT = ROOT_OUT

CAPTURE_RADIUS = 0.18e-3
ROI_HALF = 1.20e-3

PSI_CAND = [0.0, 0.5 * np.pi, 1.0 * np.pi, 1.5 * np.pi]
ALPHA_COARSE = [round(v, 2) for v in np.arange(0.10, 1.61, 0.10)]
ALPHA_EXT = [round(v, 2) for v in np.arange(1.80, 2.81, 0.20)]

SEARCH_S_END = 1.0
SEARCH_N_SW_START = 2
SEARCH_N_RAMP_ON = 4
SEARCH_N_TRANSLATE = 18
SEARCH_N_RAMP_OFF = 3
SEARCH_N_SW_RELEASE = 3

PROBE_N_SW_START = 6
PROBE_N_RAMP_ON = 10
PROBE_N_TRANSLATE = 48
PROBE_N_RAMP_OFF = 4
PROBE_N_SW_RELEASE = 4

FINAL_N_SW_START = 8
FINAL_N_RAMP_ON = 14
FINAL_N_RAMP_OFF = 30
FINAL_N_SW_RELEASE = 30


def _field_abs(
    p_sw: np.ndarray,
    p_v: np.ndarray,
    xg: np.ndarray,
    yg: np.ndarray,
    alpha: float,
    psi: float,
    centre_xy: np.ndarray,
) -> np.ndarray:
    if float(alpha) <= 1e-14:
        return np.abs(p_sw)
    source_c = np.array([0.5 * (xg[0] + xg[-1]), 0.5 * (yg[0] + yg[-1])], dtype=float)
    p_v_shift = shift_complex_field(p_v, xg, yg, centre_xy, source_centre=source_c)
    p_total = p_sw + float(alpha) * np.exp(1j * float(psi)) * p_v_shift
    return np.abs(p_total)


def _phase_title(code: int) -> str:
    if int(code) == PHASE_SW_START:
        return "1) standing-wave-only start"
    if int(code) == PHASE_RAMP_ON:
        return "2) vortex ramp-on"
    if int(code) == PHASE_TRANSLATE:
        return "3) moving vortex centre"
    if int(code) == PHASE_RAMP_OFF:
        return "4) vortex ramp-off"
    if int(code) == PHASE_SW_RELEASE:
        return "5) standing-wave-only release"
    return phase_label(int(code))


def _draw_scene(
    ax: plt.Axes,
    bg_abs: np.ndarray,
    xg: np.ndarray,
    yg: np.ndarray,
    traps_m: np.ndarray,
    idx_a: int,
    idx_b: int,
    neigh_idx: np.ndarray,
    res: StagedTransportResult,
    frame_idx: int,
    phase_text: str,
    show_roi: bool,
    title: str,
) -> None:
    ext = [xg[0] * 1e3, xg[-1] * 1e3, yg[0] * 1e3, yg[-1] * 1e3]

    A0 = traps_m[idx_a]
    B0 = traps_m[idx_b]
    midpoint = 0.5 * (A0 + B0)

    ax.imshow(bg_abs, origin="lower", extent=ext, cmap="viridis", aspect="equal")
    ax.scatter(traps_m[:, 0] * 1e3, traps_m[:, 1] * 1e3, s=8, c="w", alpha=0.18)

    nn = traps_m[np.asarray(neigh_idx, dtype=int)]
    if len(nn):
        ax.scatter(
            nn[:, 0] * 1e3,
            nn[:, 1] * 1e3,
            s=32,
            marker="s",
            facecolors="none",
            edgecolors="cyan",
            linewidths=0.8,
            label="neighbour homes",
        )
        ax.scatter(
            res.N_xy[frame_idx, :, 0] * 1e3,
            res.N_xy[frame_idx, :, 1] * 1e3,
            s=16,
            c="white",
            edgecolors="0.3",
            linewidths=0.4,
        )

    ax.plot(res.schedule.centres_xy[:, 0] * 1e3, res.schedule.centres_xy[:, 1] * 1e3, "m--", linewidth=1.6)
    ax.scatter(
        res.schedule.centres_xy[frame_idx, 0] * 1e3,
        res.schedule.centres_xy[frame_idx, 1] * 1e3,
        s=72,
        c="magenta",
        marker="D",
        edgecolors="k",
        linewidths=0.6,
        label="vortex centre",
    )

    ax.scatter(A0[0] * 1e3, A0[1] * 1e3, s=108, c="tomato", edgecolors="k", linewidths=0.7, label="A home")
    ax.scatter(B0[0] * 1e3, B0[1] * 1e3, s=108, c="deepskyblue", edgecolors="k", linewidths=0.7, label="B home")
    ax.plot([A0[0] * 1e3, B0[0] * 1e3], [A0[1] * 1e3, B0[1] * 1e3], "w--", linewidth=1.0, alpha=0.9)

    ax.plot(res.A_xy[:, 0] * 1e3, res.A_xy[:, 1] * 1e3, color="tomato", linewidth=1.5, alpha=0.9)
    ax.plot(res.B_xy[:, 0] * 1e3, res.B_xy[:, 1] * 1e3, color="deepskyblue", linewidth=1.5, alpha=0.9)

    ax.scatter(
        res.A_xy[frame_idx, 0] * 1e3,
        res.A_xy[frame_idx, 1] * 1e3,
        s=92,
        c="gold",
        marker="o",
        edgecolors="k",
        linewidths=0.7,
        label="A current",
    )
    ax.scatter(
        res.B_xy[frame_idx, 0] * 1e3,
        res.B_xy[frame_idx, 1] * 1e3,
        s=92,
        c="orange",
        marker="s",
        edgecolors="k",
        linewidths=0.7,
        label="B current",
    )

    if show_roi:
        ax.set_xlim(mm(midpoint[0] - ROI_HALF), mm(midpoint[0] + ROI_HALF))
        ax.set_ylim(mm(midpoint[1] - ROI_HALF), mm(midpoint[1] + ROI_HALF))

    alpha_k = float(res.schedule.alpha[frame_idx])
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(f"{title}\n{phase_text} | alpha={alpha_k:.2f}", fontsize=9)


def _plot_start_end_comparison(
    res: StagedTransportResult,
    p_sw: np.ndarray,
    xg: np.ndarray,
    yg: np.ndarray,
    traps_m: np.ndarray,
    idx_a: int,
    idx_b: int,
    neigh_idx: np.ndarray,
) -> None:
    start_idx = int(np.where(res.schedule.phase_codes == PHASE_SW_START)[0][0])
    end_idx = len(res.schedule.phase_codes) - 1

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.8))

    _draw_scene(
        axes[0],
        np.abs(p_sw),
        xg,
        yg,
        traps_m,
        idx_a,
        idx_b,
        neigh_idx,
        res,
        frame_idx=start_idx,
        phase_text="1) standing-wave-only start",
        show_roi=True,
        title="Start state",
    )

    _draw_scene(
        axes[1],
        np.abs(p_sw),
        xg,
        yg,
        traps_m,
        idx_a,
        idx_b,
        neigh_idx,
        res,
        frame_idx=end_idx,
        phase_text="5) standing-wave-only release (end)",
        show_roi=True,
        title="End state",
    )

    axes[0].legend(loc="upper right", fontsize=7)
    fig.suptitle("Standing-wave-only start vs standing-wave-only end", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "start_end_standing_wave_comparison.png", dpi=240)
    plt.close(fig)


def _plot_sequence(
    res: StagedTransportResult,
    p_sw: np.ndarray,
    p_v: np.ndarray,
    xg: np.ndarray,
    yg: np.ndarray,
    traps_m: np.ndarray,
    idx_a: int,
    idx_b: int,
    neigh_idx: np.ndarray,
    out_name: str,
    show_roi: bool,
) -> None:
    sel = phase_representative_indices(res.schedule, per_phase=2)
    if len(sel) == 0:
        raise RuntimeError("No representative indices produced")

    n_cols = 5
    n_rows = int(math.ceil(len(sel) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18.8, 3.9 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for i, k in enumerate(sel):
        code = int(res.schedule.phase_codes[k])
        bg = _field_abs(
            p_sw,
            p_v,
            xg,
            yg,
            alpha=float(res.schedule.alpha[k]),
            psi=float(res.psi),
            centre_xy=res.schedule.centres_xy[k],
        )
        _draw_scene(
            axes[i],
            bg,
            xg,
            yg,
            traps_m,
            idx_a,
            idx_b,
            neigh_idx,
            res,
            frame_idx=int(k),
            phase_text=_phase_title(code),
            show_roi=show_roi,
            title=f"frame {i + 1}/{len(sel)}",
        )

    for j in range(len(sel), len(axes)):
        axes[j].axis("off")

    st = "ROI" if show_roi else "Full domain"
    fig.suptitle(f"Translated-vortex staged transport sequence ({st})", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / out_name, dpi=240)
    plt.close(fig)


def _hit_metrics(res: StagedTransportResult, B0: np.ndarray) -> dict[str, Any]:
    tr_idx = np.where(res.schedule.phase_codes == PHASE_TRANSLATE)[0]
    if len(tr_idx) == 0:
        return {
            "hit": False,
            "min_dist_m": float(np.linalg.norm(res.A_xy[-1] - B0)),
            "first_hit_translate_step": -1,
            "first_hit_s": float("nan"),
        }

    d = np.linalg.norm(res.A_xy[tr_idx] - B0[None, :], axis=1)
    min_dist_m = float(np.min(d))
    hit_local = np.where(d <= CAPTURE_RADIUS)[0]
    if len(hit_local) == 0:
        return {
            "hit": False,
            "min_dist_m": min_dist_m,
            "first_hit_translate_step": -1,
            "first_hit_s": float("nan"),
        }

    j = int(hit_local[0])
    g = int(tr_idx[j])
    return {
        "hit": True,
        "min_dist_m": min_dist_m,
        "first_hit_translate_step": int(j + 1),
        "first_hit_s": float(res.schedule.translate_s[g]),
    }


def _release_outcome_label(res: StagedTransportResult, idx_b: int) -> str:
    if bool(res.metrics["A_release_to_B"]):
        return "snap_into_B"
    if bool(res.metrics["A_release_to_A"]):
        return "fall_back_A"
    if res.A_release_status == "jump_other" and int(res.A_release_nearest_idx) != int(idx_b):
        return "jump_other_trap"
    if res.A_release_status == "jump_other" and int(res.A_release_nearest_idx) == int(idx_b):
        return "snap_into_B"
    return "untrapped_or_other"


def _to_row(
    mode: str,
    alpha_target: float,
    psi: float,
    s_end: float,
    res: StagedTransportResult,
    idx_b: int,
    B0: np.ndarray,
) -> dict[str, Any]:
    hit = _hit_metrics(res, B0)
    a_release_to_b = bool(res.metrics["A_release_to_B"])
    return {
        "mode": mode,
        "alpha_target": float(alpha_target),
        "psi": float(psi),
        "stage_s_end": float(s_end),
        "n_steps_total": int(len(res.schedule.phase_codes)),
        "n_steps_translate": int(np.sum(res.schedule.phase_codes == PHASE_TRANSLATE)),
        "min_A_dist_to_B_translate_mm": mm(float(hit["min_dist_m"])),
        "A_hits_B_during_translation": bool(hit["hit"]),
        "first_hit_translate_step": int(hit["first_hit_translate_step"]),
        "first_hit_stage_fraction": None if not bool(hit["hit"]) else float(hit["first_hit_s"]),
        "A_progress_translate_mm": float(res.metrics["A_progress_translate_mm"]),
        "A_dist_to_B_after_rampoff_mm": float(res.metrics["A_dist_to_B_after_rampoff_mm"]),
        "A_dist_to_B_release_end_mm": float(res.metrics["A_dist_to_B_release_end_mm"]),
        "A_release_status": str(res.A_release_status),
        "A_release_nearest_idx": int(res.A_release_nearest_idx),
        "A_release_to_B": bool(a_release_to_b),
        "A_release_to_A": bool(res.metrics["A_release_to_A"]),
        "A_post_release_outcome": _release_outcome_label(res, idx_b),
        "release_success": bool(a_release_to_b),
    }


def _write_metrics_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError("No rows to write")
    fieldnames = [
        "mode",
        "alpha_target",
        "psi",
        "stage_s_end",
        "n_steps_total",
        "n_steps_translate",
        "min_A_dist_to_B_translate_mm",
        "A_hits_B_during_translation",
        "first_hit_translate_step",
        "first_hit_stage_fraction",
        "A_progress_translate_mm",
        "A_dist_to_B_after_rampoff_mm",
        "A_dist_to_B_release_end_mm",
        "A_release_status",
        "A_release_nearest_idx",
        "A_release_to_B",
        "A_release_to_A",
        "A_post_release_outcome",
        "release_success",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fieldnames})


def _run_case(
    p_sw: np.ndarray,
    p_v: np.ndarray,
    xg: np.ndarray,
    yg: np.ndarray,
    traps_m: np.ndarray,
    idx_a: int,
    idx_b: int,
    neigh_idx: np.ndarray,
    alpha_target: float,
    psi: float,
    s_end: float,
    n_sw_start: int,
    n_ramp_on: int,
    n_translate: int,
    n_ramp_off: int,
    n_sw_release: int,
) -> StagedTransportResult:
    schedule = build_staged_schedule(
        traps_m[idx_a],
        traps_m[idx_b],
        s_end=float(s_end),
        alpha_target=float(alpha_target),
        n_sw_start=int(n_sw_start),
        n_ramp_on=int(n_ramp_on),
        n_translate=int(n_translate),
        n_ramp_off=int(n_ramp_off),
        n_sw_release=int(n_sw_release),
    )
    return run_staged_transport(
        p_sw=p_sw,
        p_v_centered=p_v,
        xg=xg,
        yg=yg,
        traps_m=traps_m,
        idx_a=idx_a,
        idx_b=idx_b,
        neighbor_idx=neigh_idx,
        schedule=schedule,
        psi=float(psi),
        capture_radius=CAPTURE_RADIUS,
    )


def _choose_selected_row(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], bool]:
    success = [r for r in rows if bool(r["A_hits_B_during_translation"])]
    if success:
        selected = sorted(
            success,
            key=lambda r: (
                float(r["alpha_target"]),
                float(r["min_A_dist_to_B_translate_mm"]),
            ),
        )[0]
        return selected, True

    selected = sorted(
        rows,
        key=lambda r: (
            float(r["min_A_dist_to_B_translate_mm"]),
            float(r["alpha_target"]),
        ),
    )[0]
    return selected, False


def _save_case_npz(path: Path, res: StagedTransportResult, row: dict[str, Any], data: dict[str, Any]) -> None:
    np.savez_compressed(
        path,
        xg=data["xg"],
        yg=data["yg"],
        p_sw=data["p_sw"],
        p_vortex_centered=data["p_v"],
        traps_m=data["traps_m"],
        idx_A=int(data["idx_a"]),
        idx_B=int(data["idx_b"]),
        neighbor_idx=data["neigh_idx"],
        psi=float(row["psi"]),
        alpha_target=float(row["alpha_target"]),
        stage_s_end=float(row["stage_s_end"]),
        phase_codes=res.schedule.phase_codes,
        alpha_schedule=res.schedule.alpha,
        centres_xy=res.schedule.centres_xy,
        translate_s=res.schedule.translate_s,
        A_xy=res.A_xy,
        B_xy=res.B_xy,
        N_xy=res.N_xy,
        A_release_status=str(res.A_release_status),
        B_release_status=str(res.B_release_status),
        A_release_nearest_idx=int(res.A_release_nearest_idx),
        B_release_nearest_idx=int(res.B_release_nearest_idx),
        release_success=bool(row["release_success"]),
        ring_radius_m=float(data.get("ring_radius_m", 0.0)),
    )


def main() -> None:
    print("=" * 72)
    print("Vortex stage transport - Parts 3-6 A->B minimum-alpha rebuild")
    print("=" * 72)

    ref_npz = REF_DIR / "reference_fields.npz"
    ref_sum = REF_DIR / "reference_summary.json"
    cal_npz = CAL_DIR / "selected_vortex_field.npz"
    cal_sum = CAL_DIR / "selected_vortex_summary.json"
    if not ref_npz.exists() or not ref_sum.exists():
        raise FileNotFoundError("Run vortex_stage_transport_reference.py first")
    if not cal_npz.exists() or not cal_sum.exists():
        raise FileNotFoundError("Run vortex_stage_transport_calibration.py first")

    ref = np.load(ref_npz)
    cal = np.load(cal_npz)
    ref_summary = json.loads(ref_sum.read_text())
    cal_summary = json.loads(cal_sum.read_text())

    xg = ref["xg"]
    yg = ref["yg"]
    p_sw = ref["p_sw"]
    traps_m = ref["traps_m"]
    idx_a = int(ref["idx_A"])
    idx_b = int(ref["idx_B"])
    neigh_idx = ref["neighbor_idx"].astype(int)

    A0 = traps_m[idx_a]
    B0 = traps_m[idx_b]

    ring_radius_m = float(cal_summary["selected"]["ring_radius_mm"]) * 1e-3
    dx_grid = float(xg[1] - xg[0])
    d_AB = float(np.linalg.norm(B0 - A0))

    p_v_raw = cal["p_vortex"]
    sw_peak = float(np.max(np.abs(p_sw)))
    v_peak = float(np.max(np.abs(p_v_raw)))
    p_v = p_v_raw * (sw_peak / max(v_peak, 1e-30))

    print("Searching minimum alpha that gets A into B trap during translation...")
    rows_search: list[dict[str, Any]] = []

    for psi in PSI_CAND:
        coarse_rows_psi: list[dict[str, Any]] = []
        for alpha in ALPHA_COARSE:
            res = _run_case(
                p_sw,
                p_v,
                xg,
                yg,
                traps_m,
                idx_a,
                idx_b,
                neigh_idx,
                alpha_target=float(alpha),
                psi=float(psi),
                s_end=SEARCH_S_END,
                n_sw_start=SEARCH_N_SW_START,
                n_ramp_on=SEARCH_N_RAMP_ON,
                n_translate=SEARCH_N_TRANSLATE,
                n_ramp_off=SEARCH_N_RAMP_OFF,
                n_sw_release=SEARCH_N_SW_RELEASE,
            )
            row = _to_row("search_coarse", float(alpha), float(psi), SEARCH_S_END, res, idx_b, B0)
            rows_search.append(row)
            coarse_rows_psi.append(row)

        coarse_hit = [r for r in coarse_rows_psi if bool(r["A_hits_B_during_translation"])]
        if coarse_hit:
            first_hit = sorted(coarse_hit, key=lambda r: float(r["alpha_target"]))[0]
            hi = float(first_hit["alpha_target"])
            low_candidates = [a for a in ALPHA_COARSE if float(a) < hi]
            low = float(low_candidates[-1]) if low_candidates else 0.0
            refine = np.linspace(low, hi, 7)
            refine = sorted({round(float(v), 4) for v in refine if float(v) > low + 1e-9})
            for alpha in refine:
                res = _run_case(
                    p_sw,
                    p_v,
                    xg,
                    yg,
                    traps_m,
                    idx_a,
                    idx_b,
                    neigh_idx,
                    alpha_target=float(alpha),
                    psi=float(psi),
                    s_end=SEARCH_S_END,
                    n_sw_start=SEARCH_N_SW_START,
                    n_ramp_on=SEARCH_N_RAMP_ON,
                    n_translate=SEARCH_N_TRANSLATE,
                    n_ramp_off=SEARCH_N_RAMP_OFF,
                    n_sw_release=SEARCH_N_SW_RELEASE,
                )
                rows_search.append(_to_row("search_refine", float(alpha), float(psi), SEARCH_S_END, res, idx_b, B0))
        else:
            for alpha in ALPHA_EXT:
                res = _run_case(
                    p_sw,
                    p_v,
                    xg,
                    yg,
                    traps_m,
                    idx_a,
                    idx_b,
                    neigh_idx,
                    alpha_target=float(alpha),
                    psi=float(psi),
                    s_end=SEARCH_S_END,
                    n_sw_start=SEARCH_N_SW_START,
                    n_ramp_on=SEARCH_N_RAMP_ON,
                    n_translate=SEARCH_N_TRANSLATE,
                    n_ramp_off=SEARCH_N_RAMP_OFF,
                    n_sw_release=SEARCH_N_SW_RELEASE,
                )
                rows_search.append(_to_row("search_extend", float(alpha), float(psi), SEARCH_S_END, res, idx_b, B0))

    selected_search, found_hit = _choose_selected_row(rows_search)
    alpha_opt = float(selected_search["alpha_target"])
    psi_opt = float(selected_search["psi"])

    print(
        f"Selected search control alpha={alpha_opt:.4f}, psi={psi_opt:.4f} rad, "
        f"hit={bool(selected_search['A_hits_B_during_translation'])}, "
        f"min_dist={float(selected_search['min_A_dist_to_B_translate_mm']):.4f} mm"
    )

    probe = _run_case(
        p_sw,
        p_v,
        xg,
        yg,
        traps_m,
        idx_a,
        idx_b,
        neigh_idx,
        alpha_target=alpha_opt,
        psi=psi_opt,
        s_end=1.0,
        n_sw_start=PROBE_N_SW_START,
        n_ramp_on=PROBE_N_RAMP_ON,
        n_translate=PROBE_N_TRANSLATE,
        n_ramp_off=PROBE_N_RAMP_OFF,
        n_sw_release=PROBE_N_SW_RELEASE,
    )
    probe_row = _to_row("probe", alpha_opt, psi_opt, 1.0, probe, idx_b, B0)

    if bool(probe_row["A_hits_B_during_translation"]) and probe_row["first_hit_stage_fraction"] is not None:
        s_end_final = max(0.02, float(probe_row["first_hit_stage_fraction"]))
    else:
        s_end_final = 1.0

    # Use grid-spacing to set step count so each step moves ~1 grid cell.
    # The search (SEARCH_N_TRANSLATE steps over s_end=1.0) worked because
    # each step moved ~1 grid cell; replicate that density here.
    n_translate_final = max(SEARCH_N_TRANSLATE, int(s_end_final * d_AB / dx_grid) + 3)
    print(f"Final translate steps: {n_translate_final} ({s_end_final * d_AB * 1e6 / n_translate_final:.1f} µm/step, grid={dx_grid*1e6:.1f} µm)")
    final = _run_case(
        p_sw,
        p_v,
        xg,
        yg,
        traps_m,
        idx_a,
        idx_b,
        neigh_idx,
        alpha_target=alpha_opt,
        psi=psi_opt,
        s_end=s_end_final,
        n_sw_start=FINAL_N_SW_START,
        n_ramp_on=FINAL_N_RAMP_ON,
        n_translate=n_translate_final,
        n_ramp_off=FINAL_N_RAMP_OFF,
        n_sw_release=FINAL_N_SW_RELEASE,
    )
    final_row = _to_row("selected_final", alpha_opt, psi_opt, s_end_final, final, idx_b, B0)

    _plot_start_end_comparison(final, p_sw, xg, yg, traps_m, idx_a, idx_b, neigh_idx)
    _plot_sequence(
        final,
        p_sw,
        p_v,
        xg,
        yg,
        traps_m,
        idx_a,
        idx_b,
        neigh_idx,
        out_name="transport_full_domain_sequence.png",
        show_roi=False,
    )
    _plot_sequence(
        final,
        p_sw,
        p_v,
        xg,
        yg,
        traps_m,
        idx_a,
        idx_b,
        neigh_idx,
        out_name="transport_roi_sequence.png",
        show_roi=True,
    )

    rows_for_csv = sorted(rows_search, key=lambda r: (float(r["alpha_target"]), float(r["psi"])))
    rows_for_csv.append(probe_row)
    rows_for_csv.append(final_row)
    _write_metrics_csv(rows_for_csv, OUT / "transport_metrics.csv")

    data = {
        "xg": xg,
        "yg": yg,
        "p_sw": p_sw,
        "p_v": p_v,
        "traps_m": traps_m,
        "idx_a": idx_a,
        "idx_b": idx_b,
        "neigh_idx": neigh_idx,
        "ring_radius_m": ring_radius_m,
    }
    _save_case_npz(OUT / "transport_case_for_gif.npz", final, final_row, data)
    _save_case_npz(OUT / "release_case_for_gif.npz", final, final_row, data)

    min_alpha_hit = None
    success_search = [r for r in rows_search if bool(r["A_hits_B_during_translation"])]
    if success_search:
        min_alpha_hit = float(min(float(r["alpha_target"]) for r in success_search))

    cal_span = bool(cal_summary.get("feasibility_statement", {}).get("spans_too_many_traps", False))

    summary = {
        "phase": "parts3_to_6_staged_translated_transport",
        "objective": "minimum_alpha_to_move_A_to_B_start",
        "objective_note": "B and neighbours are not part of the optimization criterion in this rebuild.",
        "model_mode": "adiabatic_quasi_static_stepwise",
        "model_note": "Each step uses current alpha(t), centre c(t), and descends from previous particle states.",
        "staged_phases": [
            "standing_wave_start",
            "ramp_on",
            "translation",
            "ramp_off",
            "standing_wave_release",
        ],
        "search_config": {
            "psi_candidates": [float(v) for v in PSI_CAND],
            "alpha_coarse": ALPHA_COARSE,
            "alpha_extend": ALPHA_EXT,
            "capture_radius_mm": mm(CAPTURE_RADIUS),
            "search_schedule": {
                "s_end": SEARCH_S_END,
                "n_sw_start": SEARCH_N_SW_START,
                "n_ramp_on": SEARCH_N_RAMP_ON,
                "n_translate": SEARCH_N_TRANSLATE,
                "n_ramp_off": SEARCH_N_RAMP_OFF,
                "n_sw_release": SEARCH_N_SW_RELEASE,
            },
        },
        "selected_control": {
            "alpha_target": alpha_opt,
            "psi": psi_opt,
            "search_found_hit": bool(found_hit),
            "minimum_alpha_with_hit": min_alpha_hit,
            "search_selected_row": selected_search,
            "probe_row": probe_row,
            "final_row": final_row,
            "final_schedule": {
                "s_end": s_end_final,
                "n_sw_start": FINAL_N_SW_START,
                "n_ramp_on": FINAL_N_RAMP_ON,
                "n_translate": n_translate_final,
                "n_ramp_off": FINAL_N_RAMP_OFF,
                "n_sw_release": FINAL_N_SW_RELEASE,
            },
        },
        "stage_sweep_rows": rows_search,
        "key_questions": {
            "smallest_feasible_vortex_spans_many_traps": cal_span,
            "minimum_alpha_for_A_to_reach_B": min_alpha_hit,
            "A_reached_B_during_translation": bool(final_row["A_hits_B_during_translation"]),
            "A_release_outcome": str(final_row["A_post_release_outcome"]),
            "A_remains_in_B_after_rampoff": bool(final_row["A_release_to_B"]),
            "does_this_support_moving_to_c_shape": bool(cal_span or not bool(final_row["A_release_to_B"])),
        },
        "artifacts": {
            "start_end_standing_wave_comparison_png": "results/deliverables/vortex_stage_transport/start_end_standing_wave_comparison.png",
            "transport_full_domain_sequence_png": "results/deliverables/vortex_stage_transport/transport_full_domain_sequence.png",
            "transport_roi_sequence_png": "results/deliverables/vortex_stage_transport/transport_roi_sequence.png",
            "transport_metrics_csv": "results/deliverables/vortex_stage_transport/transport_metrics.csv",
            "transport_summary_json": "results/deliverables/vortex_stage_transport/transport_summary.json",
            "transport_case_npz": "results/deliverables/vortex_stage_transport/transport_case_for_gif.npz",
            "release_case_npz": "results/deliverables/vortex_stage_transport/release_case_for_gif.npz",
        },
    }
    save_json(OUT / "transport_summary.json", summary)

    print(f"Saved {OUT / 'transport_metrics.csv'}")
    print(f"Saved {OUT / 'transport_summary.json'}")
    print("Parts 3-6 complete.")


if __name__ == "__main__":
    main()
