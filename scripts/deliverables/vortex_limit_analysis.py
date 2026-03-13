#!/usr/bin/env python3
"""
Parts 3-6 - Overlay, displacement, release, and push-limit analysis.

Outputs:
  results/deliverables/vortex_limit/overlay/
  results/deliverables/vortex_limit/limit_plots/
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lib.vortex_limit_utils import (
    bilinear_sample_vector,
    classify_release,
    compute_gorkov_fields,
    descend_potential,
    ensure_dir,
    estimate_restoring_scale,
    project_toward,
    save_json,
)

ROOT_OUT = ensure_dir(PROJECT_ROOT / "results" / "deliverables" / "vortex_limit")
REF_DIR = ROOT_OUT / "reference"
CAL_DIR = ROOT_OUT / "vortex_calibration"
OVERLAY_OUT = ensure_dir(ROOT_OUT / "overlay")
LIMIT_OUT = ensure_dir(ROOT_OUT / "limit_plots")

ALPHA_VALUES = [round(v, 3) for v in np.linspace(0.05, 1.20, 24)]
PSI_VALUES = [0.0, 0.5 * np.pi, 1.0 * np.pi, 1.5 * np.pi]
PSI_LABELS = {0.0: "0", 0.5 * np.pi: "pi/2", 1.0 * np.pi: "pi", 1.5 * np.pi: "3pi/2"}

ROI_HALF = 1.2e-3
CAPTURE_RADIUS = 0.18e-3


def _mm(v: float) -> float:
    return float(v * 1e3)


def _scalar(v: np.ndarray) -> float:
    return float(np.real(v))


def _make_combined(p_sw: np.ndarray, p_v: np.ndarray, alpha: float, psi: float) -> np.ndarray:
    return p_sw + alpha * np.exp(1j * psi) * p_v


def _to_csv(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "alpha",
        "psi",
        "A_push_toward_B",
        "B_force_delta_norm",
        "B_force_along_AB",
        "neighbor_rms_disturb",
        "neighbor_max_disturb",
        "A_disp_toward_B_mm",
        "A_disp_norm_mm",
        "B_disp_toward_B_mm",
        "B_disp_norm_mm",
        "AB_distance_mm",
        "A_release_status",
        "B_release_status",
        "release_success",
        "B_returns_original",
        "B_jumps_other",
        "B_untrapped",
        "stability_ratio_A",
        "stability_ratio_B",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            row = {k: r[k] for k in fieldnames}
            w.writerow(row)


def _plot_case(
    tag: str,
    case: dict,
    p_sw: np.ndarray,
    p_v: np.ndarray,
    xg: np.ndarray,
    yg: np.ndarray,
    traps_m: np.ndarray,
    idx_a: int,
    idx_b: int,
    neigh_idx: np.ndarray,
    midpoint: np.ndarray,
) -> None:
    alpha = float(case["alpha"])
    psi = float(case["psi"])
    p_comb = _make_combined(p_sw, p_v, alpha, psi)

    U_sw, _, _ = compute_gorkov_fields(p_sw, float(xg[1] - xg[0]), float(yg[1] - yg[0]))
    U_c, _, _ = compute_gorkov_fields(p_comb, float(xg[1] - xg[0]), float(yg[1] - yg[0]))

    ext = [xg[0] * 1e3, xg[-1] * 1e3, yg[0] * 1e3, yg[-1] * 1e3]

    a0 = traps_m[idx_a]
    b0 = traps_m[idx_b]
    a_eq = np.array([case["A_eq_x"], case["A_eq_y"]], dtype=float)
    b_eq = np.array([case["B_eq_x"], case["B_eq_y"]], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    im = axes[0].imshow(np.abs(p_sw), origin="lower", extent=ext, cmap="viridis", aspect="equal")
    axes[0].set_title("|p_sw| at z*")
    axes[0].set_xlabel("x [mm]")
    axes[0].set_ylabel("y [mm]")
    fig.colorbar(im, ax=axes[0], shrink=0.82)

    im = axes[1].imshow(np.abs(p_comb), origin="lower", extent=ext, cmap="viridis", aspect="equal")
    axes[1].set_title(f"|p_comb| alpha={alpha:.2f}, psi={PSI_LABELS.get(psi, f'{psi:.2f}')}")
    axes[1].set_xlabel("x [mm]")
    axes[1].set_ylabel("y [mm]")
    fig.colorbar(im, ax=axes[1], shrink=0.82)

    im = axes[2].imshow(np.abs(p_comb) - np.abs(p_sw), origin="lower", extent=ext, cmap="RdBu_r", aspect="equal")
    axes[2].set_title("Delta |p|")
    axes[2].set_xlabel("x [mm]")
    axes[2].set_ylabel("y [mm]")
    fig.colorbar(im, ax=axes[2], shrink=0.82)

    for ax in axes:
        ax.scatter(traps_m[:, 0] * 1e3, traps_m[:, 1] * 1e3, s=10, c="w", alpha=0.35)
        ax.scatter(a0[0] * 1e3, a0[1] * 1e3, s=70, c="red", edgecolors="k", linewidths=0.8)
        ax.scatter(b0[0] * 1e3, b0[1] * 1e3, s=70, c="deepskyblue", edgecolors="k", linewidths=0.8)
        ax.scatter(a_eq[0] * 1e3, a_eq[1] * 1e3, s=70, c="gold", marker="x")
        ax.scatter(b_eq[0] * 1e3, b_eq[1] * 1e3, s=70, c="orange", marker="x")
        if len(neigh_idx):
            nn = traps_m[neigh_idx]
            ax.scatter(nn[:, 0] * 1e3, nn[:, 1] * 1e3, s=35, marker="s", facecolors="none", edgecolors="cyan", linewidths=0.8)

    fig.suptitle(f"Full-domain overlay ({tag})", fontsize=13)
    fig.tight_layout()
    fig.savefig(OVERLAY_OUT / f"full_domain_overlay_{tag}.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(np.abs(p_comb), origin="lower", extent=ext, cmap="viridis", aspect="equal")
    ax.scatter(a0[0] * 1e3, a0[1] * 1e3, s=80, c="red", edgecolors="k", linewidths=0.8, label="A0")
    ax.scatter(b0[0] * 1e3, b0[1] * 1e3, s=80, c="deepskyblue", edgecolors="k", linewidths=0.8, label="B0")
    ax.scatter(a_eq[0] * 1e3, a_eq[1] * 1e3, s=80, c="gold", marker="x", label="A_eq")
    ax.scatter(b_eq[0] * 1e3, b_eq[1] * 1e3, s=80, c="orange", marker="x", label="B_eq")
    if len(neigh_idx):
        nn = traps_m[neigh_idx]
        ax.scatter(nn[:, 0] * 1e3, nn[:, 1] * 1e3, s=35, marker="s", facecolors="none", edgecolors="cyan", linewidths=0.8)
    ax.set_xlim(_mm(midpoint[0] - ROI_HALF), _mm(midpoint[0] + ROI_HALF))
    ax.set_ylim(_mm(midpoint[1] - ROI_HALF), _mm(midpoint[1] + ROI_HALF))
    ax.set_title(f"ROI overlay ({tag})")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.82, label="|p| [Pa]")
    fig.tight_layout()
    fig.savefig(OVERLAY_OUT / f"roi_overlay_{tag}.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    im = axes[0].imshow(U_sw, origin="lower", extent=ext, cmap="RdBu_r", aspect="equal")
    axes[0].set_title("U_sw")
    axes[0].set_xlim(_mm(midpoint[0] - ROI_HALF), _mm(midpoint[0] + ROI_HALF))
    axes[0].set_ylim(_mm(midpoint[1] - ROI_HALF), _mm(midpoint[1] + ROI_HALF))
    axes[0].set_xlabel("x [mm]")
    axes[0].set_ylabel("y [mm]")
    fig.colorbar(im, ax=axes[0], shrink=0.82)

    dU = U_c - U_sw
    vmax = float(np.max(np.abs(dU)))
    im = axes[1].imshow(dU, origin="lower", extent=ext, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
    axes[1].set_title(f"Delta U ({tag})")
    axes[1].set_xlim(_mm(midpoint[0] - ROI_HALF), _mm(midpoint[0] + ROI_HALF))
    axes[1].set_ylim(_mm(midpoint[1] - ROI_HALF), _mm(midpoint[1] + ROI_HALF))
    axes[1].set_xlabel("x [mm]")
    axes[1].set_ylabel("y [mm]")
    fig.colorbar(im, ax=axes[1], shrink=0.82)

    fig.tight_layout()
    fig.savefig(OVERLAY_OUT / f"gorkov_roi_{tag}.png", dpi=220)
    plt.close(fig)


def _plot_limit_curves(rows: list[dict], psi_opt: float, fail_alpha: float | None) -> None:
    sub = sorted([r for r in rows if abs(float(r["psi"]) - psi_opt) < 1e-12], key=lambda r: float(r["alpha"]))
    alpha = [float(r["alpha"]) for r in sub]
    a_disp = [float(r["A_disp_toward_B_mm"]) for r in sub]
    b_disp = [float(r["B_disp_norm_mm"]) for r in sub]
    neigh = [float(r["neighbor_rms_disturb"]) for r in sub]
    release = [1.0 if bool(r["release_success"]) else 0.0 for r in sub]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    ax.plot(alpha, b_disp, "o-", color="tab:red", linewidth=1.8)
    if fail_alpha is not None:
        ax.axvline(fail_alpha, color="k", linestyle="--", linewidth=1.0)
    ax.set_xlabel("alpha")
    ax.set_ylabel("B displacement [mm]")
    ax.set_title("B displacement vs vortex strength")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(alpha, a_disp, "o-", color="tab:blue", linewidth=1.8)
    if fail_alpha is not None:
        ax.axvline(fail_alpha, color="k", linestyle="--", linewidth=1.0)
    ax.set_xlabel("alpha")
    ax.set_ylabel("A displacement toward B [mm]")
    ax.set_title("A displacement vs vortex strength")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.step(alpha, release, where="mid", color="tab:green", linewidth=1.8)
    if fail_alpha is not None:
        ax.axvline(fail_alpha, color="k", linestyle="--", linewidth=1.0)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("alpha")
    ax.set_ylabel("release success (1/0)")
    ax.set_title("Release success vs vortex strength")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(alpha, neigh, "o-", color="tab:orange", linewidth=1.8)
    if fail_alpha is not None:
        ax.axvline(fail_alpha, color="k", linestyle="--", linewidth=1.0)
    ax.set_xlabel("alpha")
    ax.set_ylabel("Neighbor disturbance RMS [N]")
    ax.set_title("Neighbor disturbance vs vortex strength")
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Push-limit curves (psi={PSI_LABELS.get(psi_opt, f'{psi_opt:.2f}')})", fontsize=13)
    fig.tight_layout()
    fig.savefig(LIMIT_OUT / "push_limit_curves.png", dpi=220)
    plt.close(fig)


def main() -> None:
    print("=" * 68)
    print("Vortex limit study - Parts 3-6 analysis")
    print("=" * 68)

    ref_npz = REF_DIR / "reference_fields.npz"
    cal_npz = CAL_DIR / "selected_vortex_field.npz"
    cal_sum = CAL_DIR / "calibration_summary.json"

    if not ref_npz.exists():
        raise FileNotFoundError("Run vortex_limit_reference.py first")
    if not cal_npz.exists() or not cal_sum.exists():
        raise FileNotFoundError("Run vortex_limit_calibration.py first")

    ref = np.load(ref_npz)
    cal = np.load(cal_npz)

    xg = ref["xg"]
    yg = ref["yg"]
    p_sw = ref["p_sw"]
    U_sw = ref["U_sw"]
    Fx_sw = ref["Fx_sw"]
    Fy_sw = ref["Fy_sw"]

    traps_m = ref["traps_m"]
    idx_a = int(ref["idx_A"])
    idx_b = int(ref["idx_B"])
    neigh_idx = ref["neighbor_idx"].astype(int)

    p_v_raw = cal["p_vortex"]

    sw_peak = float(np.max(np.abs(p_sw)))
    v_peak = float(np.max(np.abs(p_v_raw)))
    p_v = p_v_raw * (sw_peak / max(v_peak, 1e-30))

    A0 = traps_m[idx_a]
    B0 = traps_m[idx_b]

    # Centre vortex at B0 (not domain centre) so A is pulled toward B0, not midpoint.
    # ASM propagation is laterally shift-invariant (paraxial), so rolling
    # the 2-D field by (B0 - domain_centre) is equivalent to recentring the source.
    # np.roll wrapping is negligible: offset ~ 12-15 px on a 400-px domain.
    _cx_dom = 0.5 * (float(xg[0]) + float(xg[-1]))
    _cy_dom = 0.5 * (float(yg[0]) + float(yg[-1]))
    _dx_g = float(xg[1] - xg[0])
    _dy_g = float(yg[1] - yg[0])
    _shift_ix = int(round((B0[0] - _cx_dom) / _dx_g))
    _shift_iy = int(round((B0[1] - _cy_dom) / _dy_g))
    p_v = np.roll(np.roll(p_v, _shift_ix, axis=1), _shift_iy, axis=0)
    print(f"  Vortex recentred at B0: shift=({_shift_ix}, {_shift_iy}) pixels")
    AB = B0 - A0
    d_ab = float(np.linalg.norm(AB))
    e_ab = AB / max(d_ab, 1e-30)
    midpoint = 0.5 * (A0 + B0)

    F_A_sw = bilinear_sample_vector(Fx_sw, Fy_sw, xg, yg, A0)
    F_B_sw = bilinear_sample_vector(Fx_sw, Fy_sw, xg, yg, B0)

    restore_A = estimate_restoring_scale(Fx_sw, Fy_sw, xg, yg, A0)
    restore_B = estimate_restoring_scale(Fx_sw, Fy_sw, xg, yg, B0)

    rows: list[dict] = []
    print(f"Evaluating {len(ALPHA_VALUES) * len(PSI_VALUES)} overlay cases...")

    for psi in PSI_VALUES:
        for alpha in ALPHA_VALUES:
            p_comb = _make_combined(p_sw, p_v, alpha, psi)
            U_c, Fx_c, Fy_c = compute_gorkov_fields(p_comb, float(xg[1] - xg[0]), float(yg[1] - yg[0]))

            F_A = bilinear_sample_vector(Fx_c, Fy_c, xg, yg, A0)
            F_B = bilinear_sample_vector(Fx_c, Fy_c, xg, yg, B0)

            dF_A = F_A - F_A_sw
            dF_B = F_B - F_B_sw

            A_push = project_toward(dF_A, e_ab)
            B_along = project_toward(dF_B, e_ab)
            B_dist = float(np.linalg.norm(dF_B))

            neigh_dist = []
            for i in neigh_idx:
                Fi = bilinear_sample_vector(Fx_c, Fy_c, xg, yg, traps_m[int(i)])
                Fi_sw = bilinear_sample_vector(Fx_sw, Fy_sw, xg, yg, traps_m[int(i)])
                neigh_dist.append(float(np.linalg.norm(Fi - Fi_sw)))
            neigh_arr = np.array(neigh_dist, dtype=float) if len(neigh_dist) else np.zeros(1, dtype=float)
            neigh_rms = float(np.sqrt(np.mean(neigh_arr * neigh_arr)))
            neigh_max = float(np.max(neigh_arr))

            eq_A = descend_potential(U_c, xg, yg, A0)
            eq_B = descend_potential(U_c, xg, yg, B0)

            Aeq = np.asarray(eq_A["xy"], dtype=float)
            Beq = np.asarray(eq_B["xy"], dtype=float)

            A_disp = Aeq - A0
            B_disp = Beq - B0

            A_disp_proj = project_toward(A_disp, e_ab)
            B_disp_proj = project_toward(B_disp, e_ab)

            rel_A = descend_potential(U_sw, xg, yg, Aeq)
            rel_B = descend_potential(U_sw, xg, yg, Beq)
            Arel = np.asarray(rel_A["xy"], dtype=float)
            Brel = np.asarray(rel_B["xy"], dtype=float)

            cls_A = classify_release(Arel, traps_m, idx_a, CAPTURE_RADIUS)
            cls_B = classify_release(Brel, traps_m, idx_b, CAPTURE_RADIUS)

            b_returns = cls_B["status"] == "return_original"
            b_jump = cls_B["status"] == "jump_other"
            b_untrapped = cls_B["status"] == "untrapped"

            release_success = (cls_A["status"] == "return_original") and b_returns

            # Transport criterion: A must reach B0 AND stay at B0 after release.
            A_at_B_eq = float(np.linalg.norm(Aeq - B0)) < CAPTURE_RADIUS
            A_stays_at_B = (
                cls_A["status"] == "jump_other"
                and int(cls_A["nearest_idx"]) == int(idx_b)
            )
            transport_success = bool(A_at_B_eq and A_stays_at_B)

            F_sw_Aeq = np.linalg.norm(bilinear_sample_vector(Fx_sw, Fy_sw, xg, yg, Aeq))
            F_sw_Beq = np.linalg.norm(bilinear_sample_vector(Fx_sw, Fy_sw, xg, yg, Beq))

            row = {
                "alpha": float(alpha),
                "psi": float(psi),
                "A_push_toward_B": float(A_push),
                "B_force_delta_norm": float(B_dist),
                "B_force_along_AB": float(B_along),
                "neighbor_rms_disturb": float(neigh_rms),
                "neighbor_max_disturb": float(neigh_max),
                "A_disp_toward_B_mm": _mm(A_disp_proj),
                "A_disp_norm_mm": _mm(float(np.linalg.norm(A_disp))),
                "B_disp_toward_B_mm": _mm(B_disp_proj),
                "B_disp_norm_mm": _mm(float(np.linalg.norm(B_disp))),
                "AB_distance_mm": _mm(float(np.linalg.norm(Beq - Aeq))),
                "A_release_status": cls_A["status"],
                "B_release_status": cls_B["status"],
                "release_success": bool(release_success),
                "B_returns_original": bool(b_returns),
                "B_jumps_other": bool(b_jump),
                "B_untrapped": bool(b_untrapped),
                "stability_ratio_A": float(F_sw_Aeq / max(restore_A, 1e-30)),
                "stability_ratio_B": float(F_sw_Beq / max(restore_B, 1e-30)),
                "A_eq_x": float(Aeq[0]),
                "A_eq_y": float(Aeq[1]),
                "B_eq_x": float(Beq[0]),
                "B_eq_y": float(Beq[1]),
                "A_rel_x": float(Arel[0]),
                "A_rel_y": float(Arel[1]),
                "B_rel_x": float(Brel[0]),
                "B_rel_y": float(Brel[1]),
                "A_at_B_eq": bool(A_at_B_eq),
                "transport_success": bool(transport_success),
            }
            rows.append(row)

        print(f"  finished psi={PSI_LABELS.get(psi, f'{psi:.2f}')}")

    _to_csv(rows, LIMIT_OUT / "transport_limit_metrics.csv")
    save_json(LIMIT_OUT / "transport_limit_metrics.json", {"rows": rows})

    # Transport criterion: success when A reaches B0 AND stays there after vortex off.
    # Find psi that achieves transport at the lowest alpha (most efficient).
    success_rows_all = [r for r in rows if bool(r["transport_success"])]
    if success_rows_all:
        psi_to_min_alpha: dict[float, float] = {}
        for r in success_rows_all:
            p = float(r["psi"])
            psi_to_min_alpha[p] = min(psi_to_min_alpha.get(p, 1e9), float(r["alpha"]))
        psi_opt = min(psi_to_min_alpha, key=lambda p: psi_to_min_alpha[p])
    else:
        psi_opt = float(max(rows, key=lambda r: float(r["A_disp_toward_B_mm"]))["psi"])

    psi_rows = sorted([r for r in rows if abs(float(r["psi"]) - psi_opt) < 1e-12], key=lambda r: float(r["alpha"]))

    # For GIF: first_success = smallest alpha where transport works (A captured at B)
    #          last_fail     = largest alpha that still fails (A snaps back)
    psi_success = [r for r in psi_rows if bool(r["transport_success"])]
    psi_fail    = [r for r in psi_rows if not bool(r["transport_success"])]
    first_success = sorted(psi_success, key=lambda r: float(r["alpha"]))[0] if psi_success else None
    last_fail     = sorted(psi_fail,    key=lambda r: float(r["alpha"]))[-1] if psi_fail    else None

    # For the failure GIF, prefer the last failure BELOW the transport threshold
    # (shows "vortex too weak" cleanly, without non-monotonic high-alpha artefacts).
    if first_success is not None:
        _thresh = float(first_success["alpha"])
        _pf_below = [r for r in psi_fail if float(r["alpha"]) < _thresh]
        last_fail = (sorted(_pf_below, key=lambda r: float(r["alpha"]))[-1]
                     if _pf_below else last_fail)

    max_safe  = first_success  # reuse _plot_case for success overlay
    first_fail = last_fail     # reuse _plot_case for failure overlay

    safe_alpha_max = float(first_success["alpha"]) if first_success is not None else None
    fail_alpha     = float(last_fail["alpha"])     if last_fail     is not None else None

    psi_safe = psi_success  # used below for gif tracking

    fail_alpha = float(first_fail["alpha"]) if first_fail is not None else None
    _plot_limit_curves(rows, psi_opt, fail_alpha)

    if max_safe is not None:
        _plot_case("safe", max_safe, p_sw, p_v, xg, yg, traps_m, idx_a, idx_b, neigh_idx, midpoint)
    if first_fail is not None:
        _plot_case("failure", first_fail, p_sw, p_v, xg, yg, traps_m, idx_a, idx_b, neigh_idx, midpoint)

    psi_track = sorted([r for r in rows if abs(float(r["psi"]) - psi_opt) < 1e-12], key=lambda r: float(r["alpha"]))
    np.savez_compressed(
        LIMIT_OUT / "gif_inputs.npz",
        xg=xg,
        yg=yg,
        p_sw=p_sw,
        p_vortex_scaled=p_v,
        traps_m=traps_m,
        idx_A=idx_a,
        idx_B=idx_b,
        neighbor_idx=neigh_idx,
        alpha=np.array([float(r["alpha"]) for r in psi_track], dtype=float),
        A_eq_xy=np.array([[float(r["A_eq_x"]), float(r["A_eq_y"])] for r in psi_track], dtype=float),
        B_eq_xy=np.array([[float(r["B_eq_x"]), float(r["B_eq_y"])] for r in psi_track], dtype=float),
        release_success=np.array([1 if bool(r["release_success"]) else 0 for r in psi_track], dtype=int),
        psi_opt=float(psi_opt),
        safe_alpha=(np.nan if safe_alpha_max is None else float(safe_alpha_max)),
        fail_alpha=(np.nan if fail_alpha is None else float(fail_alpha)),
    )

    summary = {
        "phase": "parts3_to_6",
        "safety_definition": "transport success: A displaced to B0 in combined field AND stays at B0 after vortex off",
        "scaling": {
            "vortex_scaled_to_sw_peak": True,
            "sw_peak_abs_p": sw_peak,
            "vortex_raw_peak_abs_p": v_peak,
        },
        "sweep": {
            "alpha_values": ALPHA_VALUES,
            "psi_values": [float(v) for v in PSI_VALUES],
            "n_cases": len(rows),
        },
        "pair": {
            "idx_A": idx_a,
            "idx_B": idx_b,
            "distance_mm": _mm(d_ab),
        },
        "max_safe_case": None if max_safe is None else {
            "alpha": float(max_safe["alpha"]),
            "psi": float(max_safe["psi"]),
            "A_disp_toward_B_mm": float(max_safe["A_disp_toward_B_mm"]),
            "B_disp_norm_mm": float(max_safe["B_disp_norm_mm"]),
            "AB_distance_mm": float(max_safe["AB_distance_mm"]),
            "B_release_status": str(max_safe["B_release_status"]),
            "transport_success": bool(max_safe["transport_success"]),
        },
        "failure_case": None if first_fail is None else {
            "alpha": float(first_fail["alpha"]),
            "psi": float(first_fail["psi"]),
            "A_disp_toward_B_mm": float(first_fail["A_disp_toward_B_mm"]),
            "B_disp_norm_mm": float(first_fail["B_disp_norm_mm"]),
            "B_release_status": str(first_fail["B_release_status"]),
            "B_returns_original": bool(first_fail["B_returns_original"]),
            "B_jumps_other": bool(first_fail["B_jumps_other"]),
            "B_untrapped": bool(first_fail["B_untrapped"]),
        },
        "psi_opt": float(psi_opt),
        "transport_threshold_alpha": None if safe_alpha_max is None else float(safe_alpha_max),
        "pretransport_fail_alpha": None if fail_alpha is None else float(fail_alpha),
        "artifacts": {
            "metrics_csv": "results/deliverables/vortex_limit/limit_plots/transport_limit_metrics.csv",
            "metrics_json": "results/deliverables/vortex_limit/limit_plots/transport_limit_metrics.json",
            "curves_png": "results/deliverables/vortex_limit/limit_plots/push_limit_curves.png",
            "gif_inputs_npz": "results/deliverables/vortex_limit/limit_plots/gif_inputs.npz",
            "overlay_dir": "results/deliverables/vortex_limit/overlay",
        },
    }
    save_json(LIMIT_OUT / "analysis_summary.json", summary)

    print(f"Saved {LIMIT_OUT / 'analysis_summary.json'}")
    print("Parts 3-6 complete.")


if __name__ == "__main__":
    main()
