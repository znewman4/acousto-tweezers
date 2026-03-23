#!/usr/bin/env python3
"""
Construct one explicit rectangular corridor bridge perturbation between A and B.

This script is intentionally deterministic and interpretable:
- no optimisation
- no random sweep
- no IASA

It builds a pair-relative synthetic perturbation field shaped as a soft
rectangular corridor aligned with the A->B segment.

The perturbation structure is intentionally three-part (topology focused):
1) a high-pressure source hotspot above A,
2) a confined corridor forcing profile that decreases from A to B,
3) a destination pocket near B to preserve/deepen the trap.

Superposition:
    p_total = p_sw + alpha * exp(1j * psi) * p_pert
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lib.fem_cache_utils import (
    C_WATER,
    OMEGA,
    RHO0,
    default_particle_params,
    gorkov_grid_2d,
)

PPAR = default_particle_params()


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUT_DIR = PROJECT_ROOT / "results" / "dev" / "rectangular_corridor_bridge"

VORTEX_NPZ = (
    PROJECT_ROOT / "results" / "deliverables" / "vortex_stage_transport"
    / "transport" / "transport_case_for_gif.npz"
)
OVERLAY_NPZ = (
    PROJECT_ROOT / "results" / "c_shape_lens_15mm_overlay_study_20260310_170620"
    / "npz" / "roi_fields.npz"
)


# ---------------------------------------------------------------------------
# Manual, fixed perturbation controls (no sweep)
# ---------------------------------------------------------------------------
ALPHA = 0.10
PSI = 1.5 * np.pi

CORRIDOR_WIDTH_M = 3.0e-4
CORRIDOR_PAD_A_M = 8.0e-5
CORRIDOR_PAD_B_M = 1.0e-4
CORRIDOR_EDGE_S_M = 3.5e-5
CORRIDOR_EDGE_N_M = 3.0e-5

SOURCE_HOTSPOT_PA = 120.0
DESTINATION_POCKET_PA = 80.0
SOURCE_SIGMA_M = 1.0e-4
DESTINATION_SIGMA_M = 1.0e-4
DESTINATION_REST_TRAP_PA = 120.0
DESTINATION_REST_SIGMA_M = 8.0e-5

SOURCE_ABOVE_OFFSET_M = 9.0e-5

CORRIDOR_START_PA = 120.0
CORRIDOR_END_PA = 20.0
CORRIDOR_DECAY_POWER = 1.25
CORRIDOR_TRANSVERSE_SIGMA_RATIO = 0.32

LINE_SAMPLES = 260


# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------
CMAP = "RdBu_r"
IMSHOW_INTERP = "bicubic"

COL_A = "#e74c3c"
COL_B = "#3498db"
COL_NEIGH = "#95a5a6"
COL_HOME = "#2ecc71"


@dataclass(frozen=True)
class ABFrame:
    a_xy: np.ndarray
    b_xy: np.ndarray
    e_s: np.ndarray
    e_n: np.ndarray
    d_ab: float


def _gorkov_fields(p_total: np.ndarray, dx: float, dy: float):
    return gorkov_grid_2d(
        p_total,
        dx,
        dy,
        OMEGA,
        RHO0,
        C_WATER,
        PPAR["a"],
        PPAR["f1"],
        PPAR["f2"],
    )


def _crop_indices(
    x_full: np.ndarray,
    y_full: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    ix = np.where((x_full >= x_min) & (x_full <= x_max))[0]
    iy = np.where((y_full >= y_min) & (y_full <= y_max))[0]
    if ix.size < 2 or iy.size < 2:
        raise RuntimeError("ROI crop indices are empty; check ROI/full-grid overlap")
    return ix, iy


def _nearest_index(axis: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(axis - value)))


def _smooth_box_window(coord: np.ndarray, lo: float, hi: float, edge: float) -> np.ndarray:
    edge_safe = max(float(edge), 1.0e-12)
    return 0.5 * (
        np.tanh((coord - float(lo)) / edge_safe)
        - np.tanh((coord - float(hi)) / edge_safe)
    )


def _gaussian_2d(
    xx: np.ndarray,
    yy: np.ndarray,
    cx: float,
    cy: float,
    sigma_m: float,
    amplitude_pa: float,
) -> np.ndarray:
    sigma = max(float(sigma_m), 1.0e-12)
    r2 = (xx - float(cx)) ** 2 + (yy - float(cy)) ** 2
    return float(amplitude_pa) * np.exp(-0.5 * r2 / (sigma**2))


def make_local_ab_frame(point_a: np.ndarray, point_b: np.ndarray) -> ABFrame:
    a_xy = np.asarray(point_a, dtype=float)
    b_xy = np.asarray(point_b, dtype=float)
    ab_vec = b_xy - a_xy
    d_ab = float(np.linalg.norm(ab_vec))
    if d_ab <= 0.0:
        raise RuntimeError("A and B are identical; cannot define local A->B frame")

    e_s = ab_vec / d_ab
    e_n = np.array([-e_s[1], e_s[0]], dtype=float)
    return ABFrame(a_xy=a_xy, b_xy=b_xy, e_s=e_s, e_n=e_n, d_ab=d_ab)


def make_soft_rectangular_corridor(
    s_m: np.ndarray,
    n_m: np.ndarray,
    ab_length_m: float,
    width_m: float,
    pad_a_m: float,
    pad_b_m: float,
    edge_s_m: float,
    edge_n_m: float,
) -> np.ndarray:
    s_lo = -float(pad_a_m)
    s_hi = float(ab_length_m + pad_b_m)
    w_s = _smooth_box_window(s_m, s_lo, s_hi, edge_s_m)
    w_n = _smooth_box_window(np.abs(n_m), 0.0, 0.5 * float(width_m), edge_n_m)
    return np.clip(w_s * w_n, 0.0, 1.0)


def make_longitudinal_taper(
    s_m: np.ndarray,
    ab_length_m: float,
    high_at_a_pa: float,
    low_at_b_pa: float,
) -> np.ndarray:
    s_hat = np.clip(s_m / max(float(ab_length_m), 1.0e-12), 0.0, 1.0)
    return float(high_at_a_pa) + (float(low_at_b_pa) - float(high_at_a_pa)) * s_hat


def make_decaying_corridor_profile(
    s_m: np.ndarray,
    n_m: np.ndarray,
    ab_length_m: float,
    width_m: float,
    start_pa: float,
    end_pa: float,
    decay_power: float,
    transverse_sigma_ratio: float,
) -> np.ndarray:
    """
    Corridor forcing profile that monotonically decays from A to B and is
    transversely confined to the interior lane.
    """
    s_hat = np.clip(s_m / max(float(ab_length_m), 1.0e-12), 0.0, 1.0)
    p = max(float(decay_power), 0.25)

    high = float(start_pa)
    low = float(end_pa)
    longitudinal = low + (high - low) * np.power(1.0 - s_hat, p)

    sigma_n = max(float(transverse_sigma_ratio) * float(width_m), 1.0e-12)
    transverse = np.exp(-0.5 * (n_m / sigma_n) ** 2)
    return longitudinal * transverse


def build_corridor_bridge_field(
    x: np.ndarray,
    y: np.ndarray,
    point_a: np.ndarray,
    point_b: np.ndarray,
    width_m: float,
    pad_a_m: float,
    pad_b_m: float,
    edge_s_m: float,
    edge_n_m: float,
    source_hotspot_pa: float,
    destination_pocket_pa: float,
    source_sigma_m: float = 1.0e-4,
    destination_sigma_m: float = 1.0e-4,
    destination_rest_trap_pa: float = 0.0,
    destination_rest_sigma_m: float = 1.0e-4,
    source_above_offset_m: float = 9.0e-5,
    corridor_start_pa: float = 120.0,
    corridor_end_pa: float = 20.0,
    corridor_decay_power: float = 1.25,
    corridor_transverse_sigma_ratio: float = 0.32,
    b_quiet_radius_m: float = 1.4e-4,
    b_quiet_edge_m: float = 3.5e-5,
    neighbour_positions_m: Optional[np.ndarray] = None,
    neighbour_quiet_radius_m: float = 1.4e-4,
    neighbour_quiet_edge_m: float = 3.5e-5,
) -> tuple[np.ndarray, np.ndarray, ABFrame, np.ndarray, np.ndarray, np.ndarray]:
    frame = make_local_ab_frame(point_a, point_b)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    rx = xx - frame.a_xy[0]
    ry = yy - frame.a_xy[1]

    s_m = rx * frame.e_s[0] + ry * frame.e_s[1]
    n_m = rx * frame.e_n[0] + ry * frame.e_n[1]

    corridor_mask = make_soft_rectangular_corridor(
        s_m=s_m,
        n_m=n_m,
        ab_length_m=frame.d_ab,
        width_m=width_m,
        pad_a_m=pad_a_m,
        pad_b_m=pad_b_m,
        edge_s_m=edge_s_m,
        edge_n_m=edge_n_m,
    )

    corridor_profile = make_decaying_corridor_profile(
        s_m=s_m,
        n_m=n_m,
        ab_length_m=frame.d_ab,
        width_m=width_m,
        start_pa=corridor_start_pa,
        end_pa=corridor_end_pa,
        decay_power=corridor_decay_power,
        transverse_sigma_ratio=corridor_transverse_sigma_ratio,
    )
    corridor_pa = corridor_mask * corridor_profile

    # Suppress corridor forcing in a small neighbourhood of B to reduce leakage
    # that can unpin B before A reaches the destination.
    d_b = np.hypot(xx - frame.b_xy[0], yy - frame.b_xy[1])
    b_quiet = 0.5 * (1.0 + np.tanh((d_b - float(b_quiet_radius_m)) / max(float(b_quiet_edge_m), 1.0e-12)))
    corridor_pa *= b_quiet

    # Suppress corridor near all neighbouring particles so their SW traps are
    # not perturbed during A→B transport.
    if neighbour_positions_m is not None and len(neighbour_positions_m) > 0:
        for n_pos in np.asarray(neighbour_positions_m, dtype=float):
            d_n = np.hypot(xx - float(n_pos[0]), yy - float(n_pos[1]))
            n_quiet = 0.5 * (1.0 + np.tanh(
                (d_n - float(neighbour_quiet_radius_m)) / max(float(neighbour_quiet_edge_m), 1.0e-12)
            ))
            corridor_pa *= n_quiet

    # Choose the A-normal direction with positive global y component so
    # "above A" is geometrically consistent with the plot orientation.
    above_dir = frame.e_n if frame.e_n[1] >= 0.0 else -frame.e_n
    source_center = frame.a_xy + float(source_above_offset_m) * above_dir

    source_hotspot = _gaussian_2d(
        xx,
        yy,
        source_center[0],
        source_center[1],
        source_sigma_m,
        float(source_hotspot_pa),
    )
    destination_pocket = _gaussian_2d(
        xx,
        yy,
        frame.b_xy[0],
        frame.b_xy[1],
        destination_sigma_m,
        float(destination_pocket_pa),
    )
    destination_rest = _gaussian_2d(
        xx,
        yy,
        frame.b_xy[0],
        frame.b_xy[1],
        destination_rest_sigma_m,
        float(destination_rest_trap_pa),
    )

    # Verify that B-end destination pocket amplitude exceeds corridor field value at B.
    # B is at index closest to frame.b_xy in the grid.
    b_x_idx = _nearest_index(x, frame.b_xy[0])
    b_y_idx = _nearest_index(y, frame.b_xy[1])
    corridor_at_b = float(corridor_pa[b_y_idx, b_x_idx])
    destination_at_b_peak = float(destination_pocket_pa + destination_rest_trap_pa)
    
    if destination_at_b_peak < corridor_at_b + 10.0:
        print(
            f"WARNING: B-end destination restoring force may be insufficient. "
            f"corridor_pa[B]={corridor_at_b:.1f} Pa, "
            f"destination_pocket_pa+rest_trap_pa={destination_at_b_peak:.1f} Pa. "
            f"Consider increasing destination_pocket_pa or destination_rest_trap_pa."
        )
    else:
        print(
            f"B-end destination verified: "
            f"corridor_pa[B]={corridor_at_b:.1f} Pa << "
            f"destination_pocket+rest={destination_at_b_peak:.1f} Pa (OK)"
        )

    # Original structure: destination pocket is alpha-scaled with the bridge;
    # only the destination rest trap stays as fixed anchor.
    p_bridge_b_anchor = destination_rest.astype(complex)
    p_pert = (source_hotspot + corridor_pa + destination_pocket).astype(complex)
    return p_pert, p_bridge_b_anchor, frame, corridor_mask, corridor_pa, s_m


def sample_line_profile(
    field2d: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    start_xy: np.ndarray,
    end_xy: np.ndarray,
    n_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    frac = np.linspace(0.0, 1.0, int(n_samples))
    x_line = float(start_xy[0]) + frac * float(end_xy[0] - start_xy[0])
    y_line = float(start_xy[1]) + frac * float(end_xy[1] - start_xy[1])
    rgi = RegularGridInterpolator(
        (y, x),
        field2d,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    vals = rgi(np.column_stack([y_line, x_line]))
    return frac, vals


def _draw_roi_panel(
    ax,
    field2d: np.ndarray,
    x_roi_mm: np.ndarray,
    y_roi_mm: np.ndarray,
    traps_mm: np.ndarray,
    idx_a: int,
    idx_b: int,
    neigh_idx: np.ndarray,
    title: str,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    extent_roi = [x_roi_mm[0], x_roi_mm[-1], y_roi_mm[0], y_roi_mm[-1]]
    im = ax.imshow(
        field2d,
        origin="lower",
        extent=extent_roi,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation=IMSHOW_INTERP,
        aspect="equal",
    )

    ax.scatter(traps_mm[:, 0], traps_mm[:, 1], marker="x", c=COL_HOME, s=28, linewidths=0.8, zorder=4)
    if neigh_idx.size > 0:
        ax.scatter(traps_mm[neigh_idx, 0], traps_mm[neigh_idx, 1], c=COL_NEIGH, s=30, zorder=6)
    ax.scatter([traps_mm[idx_b, 0]], [traps_mm[idx_b, 1]], c=COL_B, s=48, zorder=7)
    ax.scatter([traps_mm[idx_a, 0]], [traps_mm[idx_a, 1]], c=COL_A, s=48, zorder=7)

    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_xlim(x_roi_mm[0], x_roi_mm[-1])
    ax.set_ylim(y_roi_mm[0], y_roi_mm[-1])
    ax.set_aspect("equal", adjustable="box")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("RECTANGULAR CORRIDOR BRIDGE (DETERMINISTIC)")
    print("=" * 78)

    print("\n1) Loading cached standing-wave field and selected A/B pair...")
    ov = np.load(OVERLAY_NPZ)
    x_roi_target = ov["xg"].astype(float)
    y_roi_target = ov["yg"].astype(float)
    traps_m = ov["traps_m"].astype(float)
    idx_a = int(ov["idx_A"])
    idx_b = int(ov["idx_B"])

    vd = np.load(VORTEX_NPZ)
    x_full = vd["xg"].astype(float)
    y_full = vd["yg"].astype(float)
    p_sw_full = vd["p_sw"].astype(complex)

    ix_roi, iy_roi = _crop_indices(
        x_full,
        y_full,
        float(x_roi_target[0]),
        float(x_roi_target[-1]),
        float(y_roi_target[0]),
        float(y_roi_target[-1]),
    )
    x_roi = x_full[ix_roi]
    y_roi = y_full[iy_roi]

    a_xy = traps_m[idx_a]
    b_xy = traps_m[idx_b]
    neigh_idx = np.array(sorted(set(range(len(traps_m))) - {idx_a, idx_b}), dtype=int)

    print(f"   A idx={idx_a}, pos={[float(a_xy[0] * 1e3), float(a_xy[1] * 1e3)]} mm")
    print(f"   B idx={idx_b}, pos={[float(b_xy[0] * 1e3), float(b_xy[1] * 1e3)]} mm")
    print(f"   neighbour count={len(neigh_idx)}")

    print("\n2) Building explicit topology-oriented bridge perturbation in A/B local frame...")
    p_pert_full, p_bridge_b_anchor_full, frame, corridor_mask_full, corridor_pa_full, s_m_full = build_corridor_bridge_field(
        x=x_full,
        y=y_full,
        point_a=a_xy,
        point_b=b_xy,
        width_m=CORRIDOR_WIDTH_M,
        pad_a_m=CORRIDOR_PAD_A_M,
        pad_b_m=CORRIDOR_PAD_B_M,
        edge_s_m=CORRIDOR_EDGE_S_M,
        edge_n_m=CORRIDOR_EDGE_N_M,
        source_hotspot_pa=SOURCE_HOTSPOT_PA,
        destination_pocket_pa=DESTINATION_POCKET_PA,
        source_sigma_m=SOURCE_SIGMA_M,
        destination_sigma_m=DESTINATION_SIGMA_M,
        destination_rest_trap_pa=DESTINATION_REST_TRAP_PA,
        destination_rest_sigma_m=DESTINATION_REST_SIGMA_M,
        source_above_offset_m=SOURCE_ABOVE_OFFSET_M,
        corridor_start_pa=CORRIDOR_START_PA,
        corridor_end_pa=CORRIDOR_END_PA,
        corridor_decay_power=CORRIDOR_DECAY_POWER,
        corridor_transverse_sigma_ratio=CORRIDOR_TRANSVERSE_SIGMA_RATIO,
    )
    p_pert_scaled_full = ALPHA * np.exp(1j * PSI) * p_pert_full
    p_total_full = p_sw_full + p_pert_scaled_full + p_bridge_b_anchor_full

    corridor_length_m = float(frame.d_ab + CORRIDOR_PAD_A_M + CORRIDOR_PAD_B_M)
    print(f"   corridor width = {CORRIDOR_WIDTH_M * 1e6:.1f} um")
    print(f"   corridor length = {corridor_length_m * 1e6:.1f} um")
    print(f"   source hotspot above A = +{abs(SOURCE_HOTSPOT_PA):.1f} Pa")
    print(f"   source hotspot offset above A = {SOURCE_ABOVE_OFFSET_M * 1e6:.1f} um")
    print(f"   destination pocket near B = {DESTINATION_POCKET_PA:+.1f} Pa")
    print(f"   destination rest trap at B = {DESTINATION_REST_TRAP_PA:+.1f} Pa")
    print(
        "   corridor forcing profile: "
        f"start={CORRIDOR_START_PA:.1f} Pa, end={CORRIDOR_END_PA:.1f} Pa, "
        f"decay_power={CORRIDOR_DECAY_POWER:.2f}"
    )

    print("\n3) Computing Gor'kov fields and diagnostics...")
    dx_full = float(x_full[1] - x_full[0])
    dy_full = float(y_full[1] - y_full[0])

    U_base_full, _, _ = _gorkov_fields(p_sw_full, dx_full, dy_full)
    U_pert_full, _, _ = _gorkov_fields(p_pert_scaled_full, dx_full, dy_full)
    U_comb_full, _, _ = _gorkov_fields(p_total_full, dx_full, dy_full)
    U_delta_full = U_comb_full - U_base_full

    U_base_roi = U_base_full[np.ix_(iy_roi, ix_roi)]
    U_pert_roi = U_pert_full[np.ix_(iy_roi, ix_roi)]
    U_comb_roi = U_comb_full[np.ix_(iy_roi, ix_roi)]
    U_delta_roi = U_delta_full[np.ix_(iy_roi, ix_roi)]

    corridor_mask_roi = corridor_mask_full[np.ix_(iy_roi, ix_roi)]
    corridor_pa_roi = corridor_pa_full[np.ix_(iy_roi, ix_roi)]

    ix_a = _nearest_index(x_full, float(a_xy[0]))
    iy_a = _nearest_index(y_full, float(a_xy[1]))
    ix_b = _nearest_index(x_full, float(b_xy[0]))
    iy_b = _nearest_index(y_full, float(b_xy[1]))

    u_a_base = float(U_base_full[iy_a, ix_a])
    u_b_base = float(U_base_full[iy_b, ix_b])
    u_a_comb = float(U_comb_full[iy_a, ix_a])
    u_b_comb = float(U_comb_full[iy_b, ix_b])

    frac_line, u_line_base = sample_line_profile(U_base_full, x_full, y_full, a_xy, b_xy, LINE_SAMPLES)
    _, u_line_comb = sample_line_profile(U_comb_full, x_full, y_full, a_xy, b_xy, LINE_SAMPLES)
    u_line_delta = u_line_comb - u_line_base
    dist_line_um = frac_line * frame.d_ab * 1e6

    neighbour_delta: list[dict[str, Any]] = []
    for i in neigh_idx:
        nx_i = _nearest_index(x_full, float(traps_m[i, 0]))
        ny_i = _nearest_index(y_full, float(traps_m[i, 1]))
        neighbour_delta.append(
            {
                "trap_index": int(i),
                "x_mm": float(traps_m[i, 0] * 1e3),
                "y_mm": float(traps_m[i, 1] * 1e3),
                "delta_U_J": float(U_delta_full[ny_i, nx_i]),
            }
        )

    print("\n4) Rendering required diagnostics figures...")
    x_roi_mm = x_roi * 1e3
    y_roi_mm = y_roi * 1e3
    traps_mm = traps_m * 1e3

    vmin_base = float(np.percentile(np.concatenate([U_base_roi.ravel(), U_comb_roi.ravel()]), 1.0))
    vmax_base = float(np.percentile(np.concatenate([U_base_roi.ravel(), U_comb_roi.ravel()]), 99.0))

    uv_pert = float(np.percentile(np.abs(U_pert_roi), 99.0))
    uv_delta = float(np.percentile(np.abs(U_delta_roi), 99.0))

    fig, axes = plt.subplots(2, 3, figsize=(16.8, 10.2))
    ax1, ax2, ax3 = axes[0]
    ax4, ax5, ax6 = axes[1]

    _draw_roi_panel(
        ax1,
        U_base_roi,
        x_roi_mm,
        y_roi_mm,
        traps_mm,
        idx_a,
        idx_b,
        neigh_idx,
        "ROI Baseline Gor'kov U",
        CMAP,
        vmin=vmin_base,
        vmax=vmax_base,
    )
    _draw_roi_panel(
        ax2,
        U_pert_roi,
        x_roi_mm,
        y_roi_mm,
        traps_mm,
        idx_a,
        idx_b,
        neigh_idx,
        "ROI Perturbation-only Gor'kov U",
        CMAP,
        vmin=-uv_pert,
        vmax=uv_pert,
    )
    _draw_roi_panel(
        ax3,
        U_comb_roi,
        x_roi_mm,
        y_roi_mm,
        traps_mm,
        idx_a,
        idx_b,
        neigh_idx,
        "ROI Combined Gor'kov U",
        CMAP,
        vmin=vmin_base,
        vmax=vmax_base,
    )
    _draw_roi_panel(
        ax4,
        U_delta_roi,
        x_roi_mm,
        y_roi_mm,
        traps_mm,
        idx_a,
        idx_b,
        neigh_idx,
        "ROI Delta-U (Combined - Baseline)",
        CMAP,
        vmin=-uv_delta,
        vmax=uv_delta,
    )

    ax5.plot(dist_line_um, u_line_base, lw=2.0, color="#222222", label="baseline")
    ax5.plot(dist_line_um, u_line_comb, lw=2.0, color="#c0392b", label="combined")
    ax5.plot(dist_line_um, u_line_delta, lw=1.4, ls="--", color="#2980b9", label="delta")
    ax5.axvline(0.0, color=COL_A, ls=":", lw=1.0)
    ax5.axvline(float(frame.d_ab * 1e6), color=COL_B, ls=":", lw=1.0)
    ax5.set_title("U Along A->B Segment")
    ax5.set_xlabel("distance from A [um]")
    ax5.set_ylabel("U [J]")
    ax5.grid(alpha=0.3)
    ax5.legend(loc="best", fontsize=9)

    extent_roi = [x_roi_mm[0], x_roi_mm[-1], y_roi_mm[0], y_roi_mm[-1]]
    im6 = ax6.imshow(
        corridor_pa_roi,
        origin="lower",
        extent=extent_roi,
        cmap=CMAP,
        interpolation=IMSHOW_INTERP,
        aspect="equal",
    )
    ax6.scatter(traps_mm[:, 0], traps_mm[:, 1], marker="x", c=COL_HOME, s=28, linewidths=0.8, zorder=4)
    if neigh_idx.size > 0:
        ax6.scatter(traps_mm[neigh_idx, 0], traps_mm[neigh_idx, 1], c=COL_NEIGH, s=30, zorder=6)
    ax6.scatter([traps_mm[idx_b, 0]], [traps_mm[idx_b, 1]], c=COL_B, s=48, zorder=7)
    ax6.scatter([traps_mm[idx_a, 0]], [traps_mm[idx_a, 1]], c=COL_A, s=48, zorder=7)
    ax6.set_title("A-to-B Decaying Corridor Forcing Template")
    ax6.set_xlabel("x [mm]")
    ax6.set_ylabel("y [mm]")
    ax6.set_xlim(x_roi_mm[0], x_roi_mm[-1])
    ax6.set_ylim(y_roi_mm[0], y_roi_mm[-1])
    ax6.set_aspect("equal", adjustable="box")
    plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.02)

    fig.suptitle(
        "Rectangular Corridor Bridge Diagnostics\n"
        f"alpha={ALPHA:.3f}, psi={PSI:.3f} rad, width={CORRIDOR_WIDTH_M*1e6:.1f} um, "
        f"length={corridor_length_m*1e6:.1f} um",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    diagnostics_png = OUT_DIR / "rectangular_corridor_bridge_diagnostics.png"
    fig.savefig(diagnostics_png, dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig_lp, ax_lp = plt.subplots(figsize=(9.2, 5.2))
    ax_lp.plot(dist_line_um, u_line_base, lw=2.2, color="#222222", label="baseline")
    ax_lp.plot(dist_line_um, u_line_comb, lw=2.2, color="#c0392b", label="combined")
    ax_lp.plot(dist_line_um, u_line_delta, lw=1.5, ls="--", color="#2980b9", label="delta")
    ax_lp.axvline(0.0, color=COL_A, ls=":", lw=1.0)
    ax_lp.axvline(float(frame.d_ab * 1e6), color=COL_B, ls=":", lw=1.0)
    ax_lp.set_title("Rectangular Corridor Bridge: U Line Profile Along A->B")
    ax_lp.set_xlabel("distance from A [um]")
    ax_lp.set_ylabel("U [J]")
    ax_lp.grid(alpha=0.3)
    ax_lp.legend(loc="best")
    fig_lp.tight_layout()
    line_profile_png = OUT_DIR / "rectangular_corridor_bridge_line_profile.png"
    fig_lp.savefig(line_profile_png, dpi=180, bbox_inches="tight")
    plt.close(fig_lp)

    print("\n5) Saving JSON metrics...")
    metrics = {
        "script": "scripts/dev/rectangular_corridor_bridge.py",
        "purpose": "One explicit soft rectangular A->B bridge with source hotspot above A, decaying corridor forcing, and B pocket",
        "pair": {
            "idx_A": int(idx_a),
            "idx_B": int(idx_b),
            "A_mm": [float(a_xy[0] * 1e3), float(a_xy[1] * 1e3)],
            "B_mm": [float(b_xy[0] * 1e3), float(b_xy[1] * 1e3)],
            "distance_AB_um": float(frame.d_ab * 1e6),
        },
        "superposition": {
            "alpha": float(ALPHA),
            "psi_rad": float(PSI),
            "formula": "p_total = p_sw + alpha * exp(1j * psi) * p_pert",
        },
        "corridor_geometry": {
            "corridor_width_m": float(CORRIDOR_WIDTH_M),
            "corridor_width_um": float(CORRIDOR_WIDTH_M * 1e6),
            "corridor_length_m": float(corridor_length_m),
            "corridor_length_um": float(corridor_length_m * 1e6),
            "pad_a_m": float(CORRIDOR_PAD_A_M),
            "pad_b_m": float(CORRIDOR_PAD_B_M),
            "edge_s_m": float(CORRIDOR_EDGE_S_M),
            "edge_n_m": float(CORRIDOR_EDGE_N_M),
            "source_hotspot_pa": float(SOURCE_HOTSPOT_PA),
            "destination_pocket_pa": float(DESTINATION_POCKET_PA),
            "destination_rest_trap_pa": float(DESTINATION_REST_TRAP_PA),
            "source_sigma_m": float(SOURCE_SIGMA_M),
            "destination_sigma_m": float(DESTINATION_SIGMA_M),
            "destination_rest_sigma_m": float(DESTINATION_REST_SIGMA_M),
            "source_above_offset_m": float(SOURCE_ABOVE_OFFSET_M),
            "corridor_start_pa": float(CORRIDOR_START_PA),
            "corridor_end_pa": float(CORRIDOR_END_PA),
            "corridor_decay_power": float(CORRIDOR_DECAY_POWER),
            "corridor_transverse_sigma_ratio": float(CORRIDOR_TRANSVERSE_SIGMA_RATIO),
            "topology_goal": "high_pressure_above_A_then_decay_to_B_with_B_preserved",
            "ab_unit_vector": [float(frame.e_s[0]), float(frame.e_s[1])],
            "normal_unit_vector": [float(frame.e_n[0]), float(frame.e_n[1])],
        },
        "u_values": {
            "U_A_baseline_J": float(u_a_base),
            "U_A_combined_J": float(u_a_comb),
            "U_B_baseline_J": float(u_b_base),
            "U_B_combined_J": float(u_b_comb),
            "A_minus_B_baseline_J": float(u_a_base - u_b_base),
            "A_minus_B_combined_J": float(u_a_comb - u_b_comb),
        },
        "line_profile": {
            "fraction_along_AB": frac_line.tolist(),
            "distance_from_A_um": dist_line_um.tolist(),
            "U_baseline_J": u_line_base.tolist(),
            "U_combined_J": u_line_comb.tolist(),
            "delta_U_J": u_line_delta.tolist(),
        },
        "delta_U_at_neighbours": neighbour_delta,
        "outputs": {
            "diagnostics_png": "rectangular_corridor_bridge_diagnostics.png",
            "line_profile_png": "rectangular_corridor_bridge_line_profile.png",
            "metrics_json": "rectangular_corridor_bridge_metrics.json",
        },
    }

    metrics_json = OUT_DIR / "rectangular_corridor_bridge_metrics.json"
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 78)
    print("DONE")
    print("=" * 78)
    print(f"Saved: {diagnostics_png}")
    print(f"Saved: {line_profile_png}")
    print(f"Saved: {metrics_json}")


if __name__ == "__main__":
    main()
