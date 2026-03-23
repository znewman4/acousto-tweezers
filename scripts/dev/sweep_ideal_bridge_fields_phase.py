#!/usr/bin/env python3
"""
Phase-aware bridge proposition sweep for ideal perturbation design.

This script replaces random two-blob perturbations with explicit bridge-shape
propositions intended to promote A->B transfer while keeping the other
neighbouring traps stable.

Panel format for each rendered proposition:
    left column  : ideal perturbation only
    right column : standing wave + ideal perturbation

Field combination remains phase-aware:
    p_total = p_sw + alpha * exp(1j * psi) * p_pert
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
from scipy.interpolate import RegularGridInterpolator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dev import ideal_pressure_static_compare as ipc


# ---------------------------------------------------------------------------
# Sweep controls
# ---------------------------------------------------------------------------
TOP_K = int(os.getenv("SWEEP_PHASE_TOP_K", "8"))
PSI_SAMPLES = int(os.getenv("SWEEP_PHASE_PSI_SAMPLES", "12"))


def _parse_float_grid(name: str, default: str) -> np.ndarray:
    raw = os.getenv(name, default)
    vals: list[float] = []
    for chunk in raw.split(","):
        text = chunk.strip()
        if text:
            vals.append(float(text))
    if not vals:
        raise ValueError(f"{name} must define at least one float")
    return np.array(vals, dtype=float)


ALPHA_GRID = _parse_float_grid(
    "SWEEP_PHASE_ALPHA_GRID",
    "0.05,0.10,0.20,0.35,0.50,0.75,1.00,1.50,2.00",
)

# Visibility gates: select candidates that are visibly changed but not dominated.
VISIBILITY_MIN = float(os.getenv("SWEEP_PHASE_VISIBILITY_MIN", "0.06"))
VISIBILITY_MAX = float(os.getenv("SWEEP_PHASE_VISIBILITY_MAX", "0.45"))
VISIBILITY_TARGET = float(
    os.getenv(
        "SWEEP_PHASE_VISIBILITY_TARGET",
        f"{0.5 * (VISIBILITY_MIN + VISIBILITY_MAX):.6f}",
    )
)

# Scoring weights
PENALTY_WEIGHT = float(os.getenv("SWEEP_PHASE_PENALTY_WEIGHT", "1.0"))
NEIGHBOR_MEAN_ABS_WEIGHT = float(os.getenv("SWEEP_PHASE_NEIGHBOR_MEAN_ABS_WEIGHT", "0.5"))
BRIDGE_WEIGHT = float(os.getenv("SWEEP_PHASE_BRIDGE_WEIGHT", "1.0"))
AB_WEIGHT = float(os.getenv("SWEEP_PHASE_AB_WEIGHT", "0.75"))

# Exclusion radius around A/B when computing ROI disturbance penalty
AB_EXCLUSION_RADIUS_M = float(os.getenv("SWEEP_PHASE_AB_EXCLUSION_RADIUS_M", "2.0e-4"))
BRIDGE_METRIC_NPTS = int(os.getenv("SWEEP_PHASE_BRIDGE_METRIC_NPTS", "240"))

# Optional notch attenuation around neighbouring traps to keep the perturbation
# focused on the A/B pair.
NEIGHBOR_NOTCH_RADIUS_M = float(os.getenv("SWEEP_PHASE_NEIGHBOR_NOTCH_RADIUS_M", "1.2e-4"))
NEIGHBOR_NOTCH_STRENGTH = float(os.getenv("SWEEP_PHASE_NEIGHBOR_NOTCH_STRENGTH", "0.85"))

OUT_DIR = ipc.OUT_DIR / "sweep_candidates_phase_bridge_proposals"


@dataclass(frozen=True)
class BridgeProposal:
    name: str
    title: str
    description: str
    params: dict[str, float]


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
    bridge_B_sw: float


def _nearest_index(axis: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(axis - value)))


def _compute_bridge_metric(
    U: np.ndarray,
    xg: np.ndarray,
    yg: np.ndarray,
    point_a: np.ndarray,
    point_b: np.ndarray,
) -> tuple[float, float, float, float, np.ndarray, np.ndarray]:
    frac = np.linspace(0.0, 1.0, BRIDGE_METRIC_NPTS)
    x_line = float(point_a[0]) + frac * float(point_b[0] - point_a[0])
    y_line = float(point_a[1]) + frac * float(point_b[1] - point_a[1])

    rgi = RegularGridInterpolator(
        (yg, xg),
        U,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    pts = np.column_stack([y_line, x_line])
    U_line = rgi(pts)

    valid = U_line[np.isfinite(U_line)]
    if valid.size == 0:
        return float("inf"), float("nan"), float("nan"), float("nan"), frac, U_line

    U_a = float(U_line[0]) if np.isfinite(U_line[0]) else float(valid[0])
    U_b = float(U_line[-1]) if np.isfinite(U_line[-1]) else float(valid[-1])
    U_saddle = float(np.nanmax(valid))
    B = U_saddle - max(U_a, U_b)
    return float(B), U_a, U_b, U_saddle, frac, U_line


def _gaussian_2d(
    xx: np.ndarray,
    yy: np.ndarray,
    center_x: float,
    center_y: float,
    sigma_m: float,
    amplitude_pa: float,
) -> np.ndarray:
    sigma = max(float(sigma_m), 1.0e-12)
    r2 = (xx - float(center_x)) ** 2 + (yy - float(center_y)) ** 2
    return float(amplitude_pa) * np.exp(-0.5 * r2 / (sigma**2))


def _smooth_window(s: np.ndarray, start: float, end: float, edge: float) -> np.ndarray:
    edge_safe = max(float(edge), 1.0e-6)
    return 0.5 * (
        np.tanh((s - float(start)) / edge_safe)
        - np.tanh((s - float(end)) / edge_safe)
    )


def _line_coordinates(
    xx: np.ndarray,
    yy: np.ndarray,
    point_a: np.ndarray,
    point_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    ab = np.asarray(point_b, dtype=float) - np.asarray(point_a, dtype=float)
    dist = float(np.linalg.norm(ab))
    if dist <= 0.0:
        raise RuntimeError("A and B positions are identical; bridge line is undefined")

    ex, ey = float(ab[0] / dist), float(ab[1] / dist)
    nx, ny = -ey, ex

    rx = xx - float(point_a[0])
    ry = yy - float(point_a[1])

    s_m = rx * ex + ry * ey
    s = s_m / dist
    t_m = rx * nx + ry * ny
    s_clip = np.clip(s, 0.0, 1.0)
    return s, s_clip, t_m, dist


def _build_tapered_channel(
    xx: np.ndarray,
    yy: np.ndarray,
    point_a: np.ndarray,
    point_b: np.ndarray,
    p: dict[str, float],
) -> np.ndarray:
    s, s_clip, t_m, _ = _line_coordinates(xx, yy, point_a, point_b)
    gate = _smooth_window(s, -0.08, 1.08, 0.06)

    width = p["corridor_w_a_m"] * (1.0 - s_clip) + p["corridor_w_b_m"] * s_clip
    width = np.maximum(width, 1.0e-12)
    gain = p["corridor_gain_min"] + (1.0 - p["corridor_gain_min"]) * s_clip
    corridor = p["corridor_amp_pa"] * gate * gain * np.exp(-0.5 * (t_m / width) ** 2)

    rails_gate = _smooth_window(s, 0.02, 1.00, 0.05)
    rails = p["rail_amp_pa"] * rails_gate * (
        np.exp(-0.5 * ((t_m - p["rail_offset_m"]) / p["rail_w_m"]) ** 2)
        + np.exp(-0.5 * ((t_m + p["rail_offset_m"]) / p["rail_w_m"]) ** 2)
    )

    out = np.zeros_like(xx, dtype=float)
    out += _gaussian_2d(xx, yy, point_b[0], point_b[1], p["anchor_rad_m"], p["anchor_amp_pa"])
    out += _gaussian_2d(xx, yy, point_a[0], point_a[1], p["source_rad_m"], p["source_amp_pa"])
    out += corridor + rails
    return out


def _build_hourglass_guarded(
    xx: np.ndarray,
    yy: np.ndarray,
    point_a: np.ndarray,
    point_b: np.ndarray,
    p: dict[str, float],
) -> np.ndarray:
    s, s_clip, t_m, _ = _line_coordinates(xx, yy, point_a, point_b)
    gate = _smooth_window(s, -0.05, 1.05, 0.05)

    width = p["waist_w_m"] + (p["end_w_m"] - p["waist_w_m"]) * np.abs(2.0 * s_clip - 1.0) ** p["waist_power"]
    width = np.maximum(width, 1.0e-12)

    mid_focus = np.exp(-0.5 * ((s_clip - 0.55) / p["mid_sigma_s"]) ** 2)
    corridor = p["corridor_amp_pa"] * gate * (0.55 + 0.45 * mid_focus) * np.exp(-0.5 * (t_m / width) ** 2)

    rails = p["rail_amp_pa"] * mid_focus * (
        np.exp(-0.5 * ((t_m - p["rail_offset_m"]) / p["rail_w_m"]) ** 2)
        + np.exp(-0.5 * ((t_m + p["rail_offset_m"]) / p["rail_w_m"]) ** 2)
    )

    out = np.zeros_like(xx, dtype=float)
    out += _gaussian_2d(xx, yy, point_b[0], point_b[1], p["anchor_rad_m"], p["anchor_amp_pa"])
    out += _gaussian_2d(xx, yy, point_a[0], point_a[1], p["source_rad_m"], p["source_amp_pa"])
    out += corridor + rails
    return out


def _build_stepped_wells(
    xx: np.ndarray,
    yy: np.ndarray,
    point_a: np.ndarray,
    point_b: np.ndarray,
    p: dict[str, float],
) -> np.ndarray:
    s, s_clip, t_m, _ = _line_coordinates(xx, yy, point_a, point_b)
    gate = _smooth_window(s, -0.04, 1.05, 0.05)

    out = np.zeros_like(xx, dtype=float)
    out += _gaussian_2d(xx, yy, point_b[0], point_b[1], p["anchor_rad_m"], p["anchor_amp_pa"])
    out += _gaussian_2d(xx, yy, point_a[0], point_a[1], p["source_rad_m"], p["source_amp_pa"])

    for step_s, step_amp in (
        (p["step1_s"], p["step1_amp_pa"]),
        (p["step2_s"], p["step2_amp_pa"]),
        (p["step3_s"], p["step3_amp_pa"]),
    ):
        cx = float(point_a[0]) + float(step_s) * float(point_b[0] - point_a[0])
        cy = float(point_a[1]) + float(step_s) * float(point_b[1] - point_a[1])
        out += _gaussian_2d(xx, yy, cx, cy, p["step_rad_m"], step_amp)

    ramp = p["corridor_amp_pa"] * gate * (0.30 + 0.70 * s_clip) * np.exp(-0.5 * (t_m / p["corridor_w_m"]) ** 2)
    halo = p["halo_amp_pa"] * np.exp(-0.5 * ((s_clip - p["halo_s0"]) / p["halo_s_sigma"]) ** 2) * np.exp(
        -0.5 * (t_m / p["halo_w_m"]) ** 2
    )
    out += ramp + halo
    return out


def _build_s_bend_corridor(
    xx: np.ndarray,
    yy: np.ndarray,
    point_a: np.ndarray,
    point_b: np.ndarray,
    p: dict[str, float],
) -> np.ndarray:
    s, s_clip, t_m, _ = _line_coordinates(xx, yy, point_a, point_b)
    gate = _smooth_window(s, -0.06, 1.04, 0.06)

    bend = p["bend_amp_m"] * np.sin(np.pi * s_clip)
    t_curved = t_m - bend

    corridor = p["corridor_amp_pa"] * gate * np.exp(-0.5 * (t_curved / p["corridor_w_m"]) ** 2)
    mid_sink = p["mid_amp_pa"] * np.exp(-0.5 * ((s_clip - p["mid_s0"]) / p["mid_s_sigma"]) ** 2) * np.exp(
        -0.5 * (t_curved / p["mid_w_m"]) ** 2
    )
    rails_gate = _smooth_window(s, 0.08, 0.95, 0.06)
    rails = p["rail_amp_pa"] * rails_gate * (
        np.exp(-0.5 * ((t_m - p["rail_offset_m"]) / p["rail_w_m"]) ** 2)
        + np.exp(-0.5 * ((t_m + p["rail_offset_m"]) / p["rail_w_m"]) ** 2)
    )

    out = np.zeros_like(xx, dtype=float)
    out += _gaussian_2d(xx, yy, point_b[0], point_b[1], p["anchor_rad_m"], p["anchor_amp_pa"])
    out += _gaussian_2d(xx, yy, point_a[0], point_a[1], p["source_rad_m"], p["source_amp_pa"])
    out += corridor + mid_sink + rails
    return out


def _get_bridge_proposals() -> list[BridgeProposal]:
    return [
        BridgeProposal(
            name="tapered_channel",
            title="Tapered Channel",
            description="Narrowing A->B corridor with side rails and a strong B sink.",
            params={
                "anchor_amp_pa": -230.0,
                "anchor_rad_m": 8.0e-5,
                "source_amp_pa": 90.0,
                "source_rad_m": 7.0e-5,
                "corridor_amp_pa": -135.0,
                "corridor_w_a_m": 1.25e-4,
                "corridor_w_b_m": 6.0e-5,
                "corridor_gain_min": 0.35,
                "rail_amp_pa": 65.0,
                "rail_offset_m": 2.2e-4,
                "rail_w_m": 7.0e-5,
            },
        ),
        BridgeProposal(
            name="hourglass_guarded",
            title="Hourglass Guarded",
            description="Hourglass throat between A/B plus symmetric guard rails near the midpoint.",
            params={
                "anchor_amp_pa": -250.0,
                "anchor_rad_m": 8.5e-5,
                "source_amp_pa": 55.0,
                "source_rad_m": 6.0e-5,
                "corridor_amp_pa": -150.0,
                "waist_w_m": 5.8e-5,
                "end_w_m": 1.35e-4,
                "waist_power": 1.4,
                "mid_sigma_s": 0.30,
                "rail_amp_pa": 85.0,
                "rail_offset_m": 1.9e-4,
                "rail_w_m": 5.5e-5,
            },
        ),
        BridgeProposal(
            name="stepped_wells",
            title="Stepped Wells",
            description="Three local sinks from A to B to form a staged descent while limiting spread.",
            params={
                "anchor_amp_pa": -220.0,
                "anchor_rad_m": 8.0e-5,
                "source_amp_pa": 80.0,
                "source_rad_m": 6.5e-5,
                "step1_s": 0.34,
                "step2_s": 0.58,
                "step3_s": 0.82,
                "step1_amp_pa": -95.0,
                "step2_amp_pa": -125.0,
                "step3_amp_pa": -160.0,
                "step_rad_m": 6.8e-5,
                "corridor_amp_pa": -72.0,
                "corridor_w_m": 9.0e-5,
                "halo_amp_pa": 62.0,
                "halo_s0": 0.58,
                "halo_s_sigma": 0.34,
                "halo_w_m": 2.4e-4,
            },
        ),
        BridgeProposal(
            name="s_bend_corridor",
            title="S-Bend Corridor",
            description="Slightly curved transfer lane to reduce coupling into surrounding neighbours.",
            params={
                "anchor_amp_pa": -210.0,
                "anchor_rad_m": 8.0e-5,
                "source_amp_pa": 95.0,
                "source_rad_m": 7.0e-5,
                "corridor_amp_pa": -130.0,
                "corridor_w_m": 7.2e-5,
                "bend_amp_m": 8.0e-5,
                "mid_amp_pa": -70.0,
                "mid_s0": 0.66,
                "mid_s_sigma": 0.24,
                "mid_w_m": 8.0e-5,
                "rail_amp_pa": 55.0,
                "rail_offset_m": 2.3e-4,
                "rail_w_m": 7.0e-5,
            },
        ),
    ]


def _apply_neighbor_notches(
    p_field: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    ctx: SweepContext,
) -> np.ndarray:
    if ctx.neigh_idx.size == 0 or NEIGHBOR_NOTCH_STRENGTH <= 0.0:
        return p_field

    r0 = max(NEIGHBOR_NOTCH_RADIUS_M, 1.0e-12)
    attenuation = np.ones_like(p_field, dtype=float)
    for i in ctx.neigh_idx:
        cx = float(ctx.traps_m[i, 0])
        cy = float(ctx.traps_m[i, 1])
        r2 = (xx - cx) ** 2 + (yy - cy) ** 2
        attenuation *= 1.0 - NEIGHBOR_NOTCH_STRENGTH * np.exp(-0.5 * r2 / (r0**2))

    attenuation = np.clip(attenuation, 0.05, 1.0)
    return p_field * attenuation


def _build_proposal_perturbation(ctx: SweepContext, proposal: BridgeProposal) -> np.ndarray:
    yy, xx = np.meshgrid(ctx.y_full, ctx.x_full, indexing="ij")

    if proposal.name == "tapered_channel":
        p_field = _build_tapered_channel(xx, yy, ctx.A_xy, ctx.B_xy, proposal.params)
    elif proposal.name == "hourglass_guarded":
        p_field = _build_hourglass_guarded(xx, yy, ctx.A_xy, ctx.B_xy, proposal.params)
    elif proposal.name == "stepped_wells":
        p_field = _build_stepped_wells(xx, yy, ctx.A_xy, ctx.B_xy, proposal.params)
    elif proposal.name == "s_bend_corridor":
        p_field = _build_s_bend_corridor(xx, yy, ctx.A_xy, ctx.B_xy, proposal.params)
    else:
        raise ValueError(f"Unknown proposal: {proposal.name}")

    p_field = _apply_neighbor_notches(p_field, xx, yy, ctx)
    return p_field.astype(complex)


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

    bridge_B_sw, _, _, _, _, _ = _compute_bridge_metric(U_sw_full, x_full, y_full, A_xy, B_xy)

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
        bridge_B_sw=float(bridge_B_sw),
    )


def _evaluate_candidate(
    ctx: SweepContext,
    proposal: BridgeProposal,
    p_pert: np.ndarray,
    alpha: float,
    psi: float,
) -> dict[str, Any]:
    phase_factor = float(alpha) * np.exp(1j * float(psi))
    p_ideal = phase_factor * p_pert
    p_total = ctx.p_sw_full + p_ideal

    U_total_full, _, _ = ipc._gorkov_fields(p_total, ctx.dx_full, ctx.dy_full)
    U_total_roi = U_total_full[np.ix_(ctx.iy_roi, ctx.ix_roi)]

    U_A = float(U_total_full[ctx.iy_A, ctx.ix_A])
    U_B = float(U_total_full[ctx.iy_B, ctx.ix_B])
    score_ab = U_A - U_B

    bridge_B, bridge_U_A, bridge_U_B, bridge_U_saddle, _, _ = _compute_bridge_metric(
        U_total_full,
        ctx.x_full,
        ctx.y_full,
        ctx.A_xy,
        ctx.B_xy,
    )
    bridge_gain = float(ctx.bridge_B_sw - bridge_B)

    delta_roi = U_total_roi - ctx.U_sw_roi
    roi_vals = delta_roi[ctx.roi_exclude_ab_mask]
    penalty_roi_std = float(np.std(roi_vals)) if roi_vals.size > 0 else 0.0

    delta_full = U_total_full - ctx.U_sw_full
    if ctx.neigh_idx.size > 0:
        neigh_delta = delta_full[ctx.neigh_grid_iy, ctx.neigh_grid_ix]
        penalty_neigh_mean_abs = float(np.mean(np.abs(neigh_delta)))
        penalty_neigh_max_abs = float(np.max(np.abs(neigh_delta)))
    else:
        penalty_neigh_mean_abs = 0.0
        penalty_neigh_max_abs = 0.0

    penalty = penalty_roi_std + NEIGHBOR_MEAN_ABS_WEIGHT * penalty_neigh_mean_abs
    score_final = BRIDGE_WEIGHT * bridge_gain + AB_WEIGHT * score_ab - PENALTY_WEIGHT * penalty

    vis_num = float(np.percentile(np.abs(delta_roi), 95)) if delta_roi.size > 0 else 0.0
    vis_den = float(np.percentile(np.abs(ctx.U_sw_roi), 95)) if ctx.U_sw_roi.size > 0 else 1.0
    visibility_ratio = vis_num / max(vis_den, 1.0e-30)
    visibility_ok = bool(VISIBILITY_MIN <= visibility_ratio <= VISIBILITY_MAX)

    return {
        "proposal_name": proposal.name,
        "proposal_title": proposal.title,
        "score_ab": float(score_ab),
        "bridge_B": float(bridge_B),
        "bridge_B_sw": float(ctx.bridge_B_sw),
        "bridge_gain": float(bridge_gain),
        "bridge_U_A": float(bridge_U_A),
        "bridge_U_B": float(bridge_U_B),
        "bridge_U_saddle": float(bridge_U_saddle),
        "penalty_roi_std": float(penalty_roi_std),
        "penalty_neigh_mean_abs": float(penalty_neigh_mean_abs),
        "penalty_neigh_max_abs": float(penalty_neigh_max_abs),
        "penalty": float(penalty),
        "score_final": float(score_final),
        "alpha": float(alpha),
        "psi": float(psi),
        "U_A": float(U_A),
        "U_B": float(U_B),
        "visibility_ratio": float(visibility_ratio),
        "visibility_ok": visibility_ok,
        "peak_pert_unscaled_pa": float(np.max(np.abs(p_pert))),
        "peak_pert_pa": float(np.max(np.abs(p_ideal))),
        "peak_total_pa": float(np.max(np.abs(p_total))),
    }


def _select_best_candidate_for_proposal(
    ctx: SweepContext,
    proposal: BridgeProposal,
    p_pert: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    psi_values = np.linspace(0.0, 2.0 * np.pi, PSI_SAMPLES, endpoint=False)

    rows: list[dict[str, Any]] = []
    for alpha in ALPHA_GRID:
        for psi in psi_values:
            rows.append(_evaluate_candidate(ctx, proposal, p_pert, float(alpha), float(psi)))

    vis_rows = [r for r in rows if r["visibility_ok"]]
    if vis_rows:
        best = max(vis_rows, key=lambda r: r["score_final"])
        selection_mode = "visibility_constrained"
    else:
        best = max(
            rows,
            key=lambda r: (-abs(r["visibility_ratio"] - VISIBILITY_TARGET), r["score_final"]),
        )
        selection_mode = "visibility_fallback"

    best_out = dict(best)
    best_out["selection_mode"] = selection_mode
    best_out["evaluated_count"] = int(len(rows))
    return best_out, rows


def _render_candidate_figure(
    ctx: SweepContext,
    proposal: BridgeProposal,
    p_pert: np.ndarray,
    score_row: dict[str, Any],
    rank: int,
    out_png: Path,
) -> None:
    phase_factor = float(score_row["alpha"]) * np.exp(1j * float(score_row["psi"]))
    p_ideal = phase_factor * p_pert
    p_total = ctx.p_sw_full + p_ideal

    U_ideal_full, _, _ = ipc._gorkov_fields(p_ideal, ctx.dx_full, ctx.dy_full)
    U_ideal_roi = U_ideal_full[np.ix_(ctx.iy_roi, ctx.ix_roi)]

    U_total_full, _, _ = ipc._gorkov_fields(p_total, ctx.dx_full, ctx.dy_full)
    U_total_roi = U_total_full[np.ix_(ctx.iy_roi, ctx.ix_roi)]

    # Separate scales: left column highlights ideal perturbation geometry, right
    # column keeps standing-wave comparison context.
    vmin_ideal, vmax_ideal = ipc._auto_limits(U_ideal_full, U_ideal_roi)
    vmin_comb, vmax_comb = ipc._auto_limits(ctx.U_sw_full, ctx.U_sw_roi, U_total_full, U_total_roi)

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 12.0))
    (ax_ideal_roi, ax_total_roi), (ax_ideal_full, ax_total_full) = axes

    ipc._draw_static_panel(
        ax_ideal_roi,
        U_ideal_roi,
        vmin_ideal,
        vmax_ideal,
        ctx.extent_roi,
        ctx.x_roi_mm,
        ctx.y_roi_mm,
        ctx.traps_mm,
        ctx.idx_A,
        ctx.idx_B,
        ctx.neigh_idx,
        "Ideal Perturbation Only (ROI)",
    )
    ipc._draw_static_panel(
        ax_total_roi,
        U_total_roi,
        vmin_comb,
        vmax_comb,
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
        ax_ideal_full,
        U_ideal_full,
        vmin_ideal,
        vmax_ideal,
        ctx.extent_full,
        ctx.x_full_mm,
        ctx.y_full_mm,
        ctx.traps_mm,
        ctx.idx_A,
        ctx.idx_B,
        ctx.neigh_idx,
        "Ideal Perturbation Only (Full Domain)",
    )
    ipc._draw_static_panel(
        ax_total_full,
        U_total_full,
        vmin_comb,
        vmax_comb,
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
        f"Bridge Proposition {rank:02d}: {proposal.title}\n"
        "Left: ideal perturbation only | Right: standing wave + ideal perturbation\n"
        f"score={score_row['score_final']:.3e} | bridge_gain={score_row['bridge_gain']:.3e} | "
        f"B_sw={score_row['bridge_B_sw']:.3e} -> B={score_row['bridge_B']:.3e} | "
        f"A-B={score_row['score_ab']:.3e} | penalty={score_row['penalty']:.3e} | "
        f"alpha={score_row['alpha']:.3f} | psi={score_row['psi']:.3f} rad | "
        f"visibility={score_row['visibility_ratio']:.3f}",
        fontsize=10.5,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if TOP_K < 1:
        raise ValueError("SWEEP_PHASE_TOP_K must be >= 1")
    if PSI_SAMPLES < 2:
        raise ValueError("SWEEP_PHASE_PSI_SAMPLES must be >= 2")
    if ALPHA_GRID.size < 1:
        raise ValueError("SWEEP_PHASE_ALPHA_GRID must contain at least one value")
    if VISIBILITY_MIN >= VISIBILITY_MAX:
        raise ValueError("SWEEP_PHASE_VISIBILITY_MIN must be < SWEEP_PHASE_VISIBILITY_MAX")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old_png in OUT_DIR.glob("proposal_*.png"):
        old_png.unlink()

    print("=" * 76)
    print("SWEEP IDEAL BRIDGE FIELDS (PHASE-AWARE PROPOSITIONS)")
    print("=" * 76)
    print(f"Propositions: {len(_get_bridge_proposals())}")
    print(f"Top K: {TOP_K}")
    print(f"Alpha grid: {ALPHA_GRID.tolist()}")
    print(f"Psi samples: {PSI_SAMPLES}")
    print(f"Visibility window: [{VISIBILITY_MIN:.3f}, {VISIBILITY_MAX:.3f}]")
    print(f"Output dir: {OUT_DIR}")
    print("Using: p_total = p_sw + alpha * exp(1j * psi) * p_pert")

    print("\n1) Building baseline context from cached fields...")
    ctx = _build_context()
    print(f"   A idx={ctx.idx_A}, B idx={ctx.idx_B}, neighbours={len(ctx.neigh_idx)}")
    print(f"   Baseline bridge barrier B_sw={ctx.bridge_B_sw:.3e} J")

    proposals = _get_bridge_proposals()
    proposal_by_name = {p.name: p for p in proposals}

    print("\n2) Evaluating bridge-shape propositions over alpha/psi grid...")
    selected_rows: list[dict[str, Any]] = []
    search_summaries: dict[str, Any] = {}
    perturbations: dict[str, np.ndarray] = {}
    for k, proposal in enumerate(proposals, start=1):
        p_pert = _build_proposal_perturbation(ctx, proposal)
        perturbations[proposal.name] = p_pert

        best_row, all_rows = _select_best_candidate_for_proposal(ctx, proposal, p_pert)
        selected_rows.append(best_row)

        score_vals = np.array([r["score_final"] for r in all_rows], dtype=float)
        vis_vals = np.array([r["visibility_ratio"] for r in all_rows], dtype=float)
        bridge_vals = np.array([r["bridge_B"] for r in all_rows], dtype=float)
        search_summaries[proposal.name] = {
            "proposal_title": proposal.title,
            "evaluated": int(len(all_rows)),
            "score_final_min": float(np.min(score_vals)),
            "score_final_max": float(np.max(score_vals)),
            "visibility_min": float(np.min(vis_vals)),
            "visibility_max": float(np.max(vis_vals)),
            "bridge_B_min": float(np.min(bridge_vals)),
            "bridge_B_max": float(np.max(bridge_vals)),
            "best_selection_mode": best_row["selection_mode"],
            "best_alpha": float(best_row["alpha"]),
            "best_psi": float(best_row["psi"]),
            "best_score": float(best_row["score_final"]),
            "best_visibility": float(best_row["visibility_ratio"]),
            "best_bridge_B": float(best_row["bridge_B"]),
        }

        print(
            f"   [{k}/{len(proposals)}] {proposal.name}: "
            f"score={best_row['score_final']:.3e}, "
            f"bridge_B={best_row['bridge_B']:.3e}, "
            f"vis={best_row['visibility_ratio']:.3f}, "
            f"mode={best_row['selection_mode']}"
        )

    rows_sorted = sorted(selected_rows, key=lambda r: r["score_final"], reverse=True)
    top_rows = rows_sorted[: min(TOP_K, len(rows_sorted))]

    print("\n3) Rendering selected propositions in the requested 2x2 layout...")
    for rank, row in enumerate(top_rows, start=1):
        proposal = proposal_by_name[row["proposal_name"]]
        p_pert = perturbations[proposal.name]
        png_path = OUT_DIR / f"proposal_{rank:02d}_{proposal.name}.png"
        _render_candidate_figure(ctx, proposal, p_pert, row, rank, png_path)
        row["rank"] = int(rank)
        row["png"] = png_path.name
        row["proposal_description"] = proposal.description
        row["proposal_params"] = proposal.params
        print(f"   saved {png_path.name}")

    out_json = OUT_DIR / "proposal_parameters.json"
    payload = {
        "script": "scripts/dev/sweep_ideal_bridge_fields_phase.py",
        "mode": "bridge_shape_propositions",
        "source_logic": "scripts/dev/ideal_pressure_static_compare.py",
        "num_propositions": int(len(proposals)),
        "top_k": int(len(top_rows)),
        "proposal_names": [p.name for p in proposals],
        "search_grid": {
            "alpha": ALPHA_GRID.tolist(),
            "psi_samples": int(PSI_SAMPLES),
        },
        "visibility_window": [float(VISIBILITY_MIN), float(VISIBILITY_MAX)],
        "scoring": {
            "score": "bridge_weight*(B_sw-B) + ab_weight*(U_A-U_B) - penalty_weight*(std(delta_U_roi_excluding_A_B) + neighbor_mean_abs_weight*mean(abs(delta_U_at_neighbors)))",
            "penalty_weight": float(PENALTY_WEIGHT),
            "neighbor_mean_abs_weight": float(NEIGHBOR_MEAN_ABS_WEIGHT),
            "bridge_weight": float(BRIDGE_WEIGHT),
            "ab_weight": float(AB_WEIGHT),
            "ab_exclusion_radius_m": float(AB_EXCLUSION_RADIUS_M),
            "bridge_metric_npts": int(BRIDGE_METRIC_NPTS),
            "field_combination": "p_total = p_sw + alpha * exp(1j * psi) * p_pert",
        },
        "neighbour_protection": {
            "enabled": bool(NEIGHBOR_NOTCH_STRENGTH > 0.0),
            "notch_radius_m": float(NEIGHBOR_NOTCH_RADIUS_M),
            "notch_strength": float(NEIGHBOR_NOTCH_STRENGTH),
        },
        "baseline": {
            "bridge_B_sw": float(ctx.bridge_B_sw),
            "idx_A": int(ctx.idx_A),
            "idx_B": int(ctx.idx_B),
            "neighbour_count": int(len(ctx.neigh_idx)),
        },
        "search_summaries": search_summaries,
        "ranked_propositions": [
            {
                "rank": int(row["rank"]),
                "proposal_name": row["proposal_name"],
                "proposal_title": row["proposal_title"],
                "proposal_description": row["proposal_description"],
                "proposal_params": row["proposal_params"],
                "png": row["png"],
                "score_final": float(row["score_final"]),
                "score_ab": float(row["score_ab"]),
                "bridge_B": float(row["bridge_B"]),
                "bridge_B_sw": float(row["bridge_B_sw"]),
                "bridge_gain": float(row["bridge_gain"]),
                "penalty": float(row["penalty"]),
                "alpha": float(row["alpha"]),
                "psi": float(row["psi"]),
                "penalty_roi_std": float(row["penalty_roi_std"]),
                "penalty_neigh_mean_abs": float(row["penalty_neigh_mean_abs"]),
                "penalty_neigh_max_abs": float(row["penalty_neigh_max_abs"]),
                "U_A": float(row["U_A"]),
                "U_B": float(row["U_B"]),
                "visibility_ratio": float(row["visibility_ratio"]),
                "visibility_ok": bool(row["visibility_ok"]),
                "selection_mode": row["selection_mode"],
                "peak_pert_pa": float(row["peak_pert_pa"]),
                "peak_pert_unscaled_pa": float(row["peak_pert_unscaled_pa"]),
                "peak_total_pa": float(row["peak_total_pa"]),
            }
            for row in top_rows
        ],
        "all_proposals_summary": {
            "score_final_min": float(min(r["score_final"] for r in rows_sorted)),
            "score_final_max": float(max(r["score_final"] for r in rows_sorted)),
            "score_final_mean": float(np.mean([r["score_final"] for r in rows_sorted])),
            "bridge_B_min": float(min(r["bridge_B"] for r in rows_sorted)),
            "bridge_B_max": float(max(r["bridge_B"] for r in rows_sorted)),
            "visibility_min": float(min(r["visibility_ratio"] for r in rows_sorted)),
            "visibility_max": float(max(r["visibility_ratio"] for r in rows_sorted)),
        },
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\n4) Sweep complete")
    print(f"   Wrote: {out_json}")
    print(f"   PNGs: {len(top_rows)} proposal images in {OUT_DIR}")


if __name__ == "__main__":
    main()