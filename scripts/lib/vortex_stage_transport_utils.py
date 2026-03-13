"""Utilities for staged translated symmetric-vortex transport studies.

The staged model is explicitly:
1) standing-wave-only start,
2) vortex ramp-on,
3) translated vortex centre,
4) vortex ramp-off,
5) standing-wave-only release.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.ndimage import shift as nd_shift

from scripts.lib.vortex_limit_utils import classify_release, compute_gorkov_fields, descend_potential

PHASE_SW_START = 1
PHASE_RAMP_ON = 2
PHASE_TRANSLATE = 3
PHASE_RAMP_OFF = 4
PHASE_SW_RELEASE = 5


def phase_label(code: int) -> str:
    return {
        PHASE_SW_START: "standing_wave_start",
        PHASE_RAMP_ON: "ramp_on",
        PHASE_TRANSLATE: "translation",
        PHASE_RAMP_OFF: "ramp_off",
        PHASE_SW_RELEASE: "standing_wave_release",
    }.get(int(code), "unknown")


@dataclass
class StageSchedule:
    phase_codes: np.ndarray
    alpha: np.ndarray
    centres_xy: np.ndarray
    translate_s: np.ndarray


@dataclass
class StagedTransportResult:
    schedule: StageSchedule
    psi: float
    alpha_target: float
    s_end: float
    A_xy: np.ndarray
    B_xy: np.ndarray
    N_xy: np.ndarray
    A_status: List[str]
    B_status: List[str]
    N_status: List[List[str]]
    A_release_status: str
    B_release_status: str
    N_release_status: List[str]
    A_release_nearest_idx: int
    B_release_nearest_idx: int
    N_release_nearest_idx: List[int]
    metrics: Dict[str, float | int | bool]


def mm(v: float) -> float:
    return float(v * 1e3)


def shift_complex_field(
    field_xy: np.ndarray,
    xg: np.ndarray,
    yg: np.ndarray,
    target_centre: np.ndarray,
    source_centre: np.ndarray | None = None,
    order: int = 1,
    mode: str = "constant",
) -> np.ndarray:
    """Shift a complex field so its centre moves from source_centre to target_centre."""
    dx = float(xg[1] - xg[0])
    dy = float(yg[1] - yg[0])

    if source_centre is None:
        source_centre = np.array([0.5 * (xg[0] + xg[-1]), 0.5 * (yg[0] + yg[-1])], dtype=float)

    dxy = np.asarray(target_centre, dtype=float) - np.asarray(source_centre, dtype=float)
    shift_x_pix = float(dxy[0] / dx)
    shift_y_pix = float(dxy[1] / dy)

    re = nd_shift(np.real(field_xy), shift=(shift_y_pix, shift_x_pix), order=order, mode=mode, cval=0.0)
    im = nd_shift(np.imag(field_xy), shift=(shift_y_pix, shift_x_pix), order=order, mode=mode, cval=0.0)
    return re + 1j * im


def classify_home(
    final_xy: np.ndarray,
    traps_m: np.ndarray,
    idx_home: int,
    capture_radius: float,
) -> Dict[str, Any]:
    cls = classify_release(final_xy, traps_m, idx_home, capture_radius)
    if cls["status"] == "return_original":
        state = "home"
    else:
        state = str(cls["status"])
    return {
        "state": state,
        "nearest_idx": int(cls["nearest_idx"]),
        "nearest_dist": float(cls["nearest_dist"]),
    }


def _cosine_ramp(n: int, start: float, end: float) -> np.ndarray:
    n = int(max(n, 1))
    if n == 1:
        return np.array([float(end)], dtype=float)
    u = np.linspace(0.0, 1.0, n)
    w = 0.5 * (1.0 - np.cos(np.pi * u))
    return (1.0 - w) * float(start) + w * float(end)


def build_staged_schedule(
    A0: np.ndarray,
    B0: np.ndarray,
    s_end: float,
    alpha_target: float,
    n_sw_start: int,
    n_ramp_on: int,
    n_translate: int,
    n_ramp_off: int,
    n_sw_release: int,
) -> StageSchedule:
    """Build explicit five-phase transport schedule arrays."""
    A0 = np.asarray(A0, dtype=float)
    B0 = np.asarray(B0, dtype=float)
    s_end = float(max(0.0, s_end))
    alpha_target = float(max(0.0, alpha_target))

    phase_codes: List[int] = []
    alpha_vals: List[float] = []
    centres: List[np.ndarray] = []
    s_vals: List[float] = []

    end_centre = A0 + s_end * (B0 - A0)

    for _ in range(max(1, int(n_sw_start))):
        phase_codes.append(PHASE_SW_START)
        alpha_vals.append(0.0)
        centres.append(A0.copy())
        s_vals.append(0.0)

    ramp_on = _cosine_ramp(max(1, int(n_ramp_on)) + 1, 0.0, alpha_target)[1:]
    for a in ramp_on:
        phase_codes.append(PHASE_RAMP_ON)
        alpha_vals.append(float(a))
        centres.append(A0.copy())
        s_vals.append(0.0)

    u_trans = np.linspace(0.0, 1.0, max(1, int(n_translate)) + 1)[1:]
    for u in u_trans:
        sval = float(s_end * float(u))
        phase_codes.append(PHASE_TRANSLATE)
        alpha_vals.append(alpha_target)
        centres.append(A0 + sval * (B0 - A0))
        s_vals.append(sval)

    ramp_off = _cosine_ramp(max(1, int(n_ramp_off)) + 1, alpha_target, 0.0)[1:]
    for a in ramp_off:
        phase_codes.append(PHASE_RAMP_OFF)
        alpha_vals.append(float(a))
        centres.append(end_centre.copy())
        s_vals.append(s_end)

    for _ in range(max(1, int(n_sw_release))):
        phase_codes.append(PHASE_SW_RELEASE)
        alpha_vals.append(0.0)
        centres.append(end_centre.copy())
        s_vals.append(s_end)

    return StageSchedule(
        phase_codes=np.asarray(phase_codes, dtype=int),
        alpha=np.asarray(alpha_vals, dtype=float),
        centres_xy=np.asarray(centres, dtype=float),
        translate_s=np.asarray(s_vals, dtype=float),
    )


def _displacement_from_home(path_xy: np.ndarray, home_xy: np.ndarray) -> np.ndarray:
    return np.linalg.norm(path_xy - home_xy[None, :], axis=1)


def run_staged_transport(
    p_sw: np.ndarray,
    p_v_centered: np.ndarray,
    xg: np.ndarray,
    yg: np.ndarray,
    traps_m: np.ndarray,
    idx_a: int,
    idx_b: int,
    neighbor_idx: np.ndarray,
    schedule: StageSchedule,
    psi: float,
    capture_radius: float,
    b_disturb_threshold: float | None = None,
) -> StagedTransportResult:
    """Run explicit five-phase staged transport with adiabatic quasi-static updates."""
    idx_a = int(idx_a)
    idx_b = int(idx_b)
    nidx = np.asarray(neighbor_idx, dtype=int)

    A0 = np.asarray(traps_m[idx_a], dtype=float)
    B0 = np.asarray(traps_m[idx_b], dtype=float)
    N0 = np.asarray(traps_m[nidx], dtype=float)

    n_steps = int(len(schedule.phase_codes))
    A_xy = np.zeros((n_steps, 2), dtype=float)
    B_xy = np.zeros((n_steps, 2), dtype=float)
    N_xy = np.zeros((n_steps, len(nidx), 2), dtype=float)

    A_status: List[str] = []
    B_status: List[str] = []
    N_status: List[List[str]] = []

    A_prev = A0.copy()
    B_prev = B0.copy()
    N_prev = N0.copy()

    dx = float(xg[1] - xg[0])
    dy = float(yg[1] - yg[0])
    source_centre = np.array([0.5 * (xg[0] + xg[-1]), 0.5 * (yg[0] + yg[-1])], dtype=float)

    psi = float(psi)
    ei_psi = np.exp(1j * psi)

    for k in range(n_steps):
        alpha_k = float(schedule.alpha[k])
        centre_k = np.asarray(schedule.centres_xy[k], dtype=float)

        if alpha_k <= 1e-14:
            p_total = p_sw
        else:
            p_v_shift = shift_complex_field(
                p_v_centered,
                xg,
                yg,
                target_centre=centre_k,
                source_centre=source_centre,
            )
            p_total = p_sw + alpha_k * ei_psi * p_v_shift

        U, _, _ = compute_gorkov_fields(p_total, dx, dy)

        A_now = np.asarray(descend_potential(U, xg, yg, A_prev)["xy"], dtype=float)
        B_now = np.asarray(descend_potential(U, xg, yg, B_prev)["xy"], dtype=float)

        N_now = np.zeros_like(N_prev)
        n_states: List[str] = []
        for j in range(len(nidx)):
            N_now[j] = np.asarray(descend_potential(U, xg, yg, N_prev[j])["xy"], dtype=float)
            n_cls = classify_home(N_now[j], traps_m, int(nidx[j]), capture_radius)
            n_states.append(str(n_cls["state"]))

        A_cls = classify_home(A_now, traps_m, idx_a, capture_radius)
        B_cls = classify_home(B_now, traps_m, idx_b, capture_radius)

        A_xy[k] = A_now
        B_xy[k] = B_now
        N_xy[k] = N_now
        A_status.append(str(A_cls["state"]))
        B_status.append(str(B_cls["state"]))
        N_status.append(n_states)

        A_prev = A_now
        B_prev = B_now
        N_prev = N_now

    A_rel = classify_release(A_xy[-1], traps_m, idx_a, capture_radius)
    B_rel = classify_release(B_xy[-1], traps_m, idx_b, capture_radius)

    N_release_status: List[str] = []
    N_release_nearest_idx: List[int] = []
    for j, idx_home in enumerate(nidx):
        n_rel = classify_release(N_xy[-1, j], traps_m, int(idx_home), capture_radius)
        N_release_status.append(str(n_rel["status"]))
        N_release_nearest_idx.append(int(n_rel["nearest_idx"]))

    phase = schedule.phase_codes
    active_mask = np.isin(phase, [PHASE_RAMP_ON, PHASE_TRANSLATE, PHASE_RAMP_OFF])
    translate_mask = phase == PHASE_TRANSLATE

    AB = B0 - A0
    d_ab = float(np.linalg.norm(AB))
    e_ab = AB / max(d_ab, 1e-30)

    idx_active_end = int(np.where(active_mask)[0][-1]) if np.any(active_mask) else n_steps - 1
    idx_translate_end = int(np.where(translate_mask)[0][-1]) if np.any(translate_mask) else idx_active_end

    A_progress_translate = float(np.dot(A_xy[idx_translate_end] - A0, e_ab))
    A_progress_active_end = float(np.dot(A_xy[idx_active_end] - A0, e_ab))

    A_lag = np.linalg.norm(A_xy[translate_mask] - schedule.centres_xy[translate_mask], axis=1) if np.any(translate_mask) else np.zeros(1)
    A_mean_lag = float(np.mean(A_lag))
    A_max_lag = float(np.max(A_lag))

    A_disp = _displacement_from_home(A_xy, A0)
    B_disp = _displacement_from_home(B_xy, B0)
    A_max_disp_active = float(np.max(A_disp[active_mask])) if np.any(active_mask) else float(np.max(A_disp))
    B_max_disp_active = float(np.max(B_disp[active_mask])) if np.any(active_mask) else float(np.max(B_disp))

    if len(nidx) > 0:
        N_disp = np.linalg.norm(N_xy - N0[None, :, :], axis=2)
        neigh_max_active = float(np.max(N_disp[active_mask])) if np.any(active_mask) else float(np.max(N_disp))
        neigh_rms_active = float(np.sqrt(np.mean(N_disp[active_mask] ** 2))) if np.any(active_mask) else float(np.sqrt(np.mean(N_disp ** 2)))
    else:
        neigh_max_active = 0.0
        neigh_rms_active = 0.0

    A_release_to_B = bool(A_rel["status"] == "jump_other" and int(A_rel["nearest_idx"]) == idx_b)
    A_release_to_A = bool(A_rel["status"] == "return_original")
    B_release_home = bool(B_rel["status"] == "return_original")
    neigh_home_frac_release = float(sum(st == "return_original" for st in N_release_status) / max(len(N_release_status), 1))

    if b_disturb_threshold is None:
        b_disturb_threshold = float(capture_radius)
    b_disturb_threshold = float(b_disturb_threshold)

    B_nonhome_active = bool(any(B_status[k] != "home" for k in np.where(active_mask)[0])) if np.any(active_mask) else False
    B_too_disturbed = bool(B_nonhome_active or B_max_disp_active > b_disturb_threshold or not B_release_home)

    metrics: Dict[str, float | int | bool] = {
        "stage_distance_mm": mm(float(schedule.translate_s[idx_active_end]) * d_ab),
        "A_progress_translate_mm": mm(A_progress_translate),
        "A_progress_active_end_mm": mm(A_progress_active_end),
        "A_mean_lag_to_centre_translate_mm": mm(A_mean_lag),
        "A_max_lag_to_centre_translate_mm": mm(A_max_lag),
        "A_dist_to_B_after_rampoff_mm": mm(float(np.linalg.norm(A_xy[idx_active_end] - B0))),
        "A_dist_to_B_release_end_mm": mm(float(np.linalg.norm(A_xy[-1] - B0))),
        "A_max_disp_active_mm": mm(A_max_disp_active),
        "B_max_disp_active_mm": mm(B_max_disp_active),
        "neighbor_max_disp_active_mm": mm(neigh_max_active),
        "neighbor_rms_disp_active_mm": mm(neigh_rms_active),
        "B_nonhome_active": bool(B_nonhome_active),
        "B_too_disturbed": bool(B_too_disturbed),
        "A_release_to_B": bool(A_release_to_B),
        "A_release_to_A": bool(A_release_to_A),
        "B_release_home": bool(B_release_home),
        "B_release_jump_other": bool(B_rel["status"] == "jump_other"),
        "B_release_untrapped": bool(B_rel["status"] == "untrapped"),
        "neighbor_home_fraction_release": float(neigh_home_frac_release),
        "neighbor_nonhome_count_release": int(sum(st != "return_original" for st in N_release_status)),
    }

    return StagedTransportResult(
        schedule=schedule,
        psi=psi,
        alpha_target=float(np.max(schedule.alpha)),
        s_end=float(np.max(schedule.translate_s)),
        A_xy=A_xy,
        B_xy=B_xy,
        N_xy=N_xy,
        A_status=A_status,
        B_status=B_status,
        N_status=N_status,
        A_release_status=str(A_rel["status"]),
        B_release_status=str(B_rel["status"]),
        N_release_status=N_release_status,
        A_release_nearest_idx=int(A_rel["nearest_idx"]),
        B_release_nearest_idx=int(B_rel["nearest_idx"]),
        N_release_nearest_idx=N_release_nearest_idx,
        metrics=metrics,
    )


def phase_representative_indices(schedule: StageSchedule, per_phase: int = 2) -> np.ndarray:
    """Pick evenly spaced frame indices within each phase for summary plots."""
    per_phase = max(int(per_phase), 1)
    idx: List[int] = []
    for code in [PHASE_SW_START, PHASE_RAMP_ON, PHASE_TRANSLATE, PHASE_RAMP_OFF, PHASE_SW_RELEASE]:
        where = np.where(schedule.phase_codes == code)[0]
        if len(where) == 0:
            continue
        if len(where) <= per_phase:
            idx.extend(int(v) for v in where)
        else:
            pick = np.linspace(where[0], where[-1], per_phase).astype(int)
            idx.extend(int(v) for v in pick)
    if not idx:
        return np.zeros(0, dtype=int)
    return np.unique(np.asarray(idx, dtype=int))
