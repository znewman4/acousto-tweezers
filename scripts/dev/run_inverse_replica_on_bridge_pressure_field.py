#!/usr/bin/env python3
"""
Run the replica IASA lens design pipeline using the produced bridge pressure field
as the target amplitude map.

This reuses the core iterative solver from:
  scripts/dev/inverse_c_shape_lens_replica.py
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.interpolate import RegularGridInterpolator

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import sys
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dev.inverse_c_shape_lens_replica import (
    ReplicaConfig,
    export_stl,
    make_grid,
    propagate_asm,
    run_iasa,
    run_iasa_roi,
)


BRIDGE_FIELDS_NPZ = (
    PROJECT_ROOT / "results" / "dev" / "bridge_pressure_field_standalone_scaled" / "bridge_pressure_fields.npz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ROI-constrained IASA on bridge pressure field target")
    parser.add_argument("--input-npz", type=Path, default=BRIDGE_FIELDS_NPZ)
    parser.add_argument("--field-key", type=str, default="p_bridge_effective_full")
    parser.add_argument("--n-iter", type=int, default=100)
    parser.add_argument("--focal-distance-mm", type=float, default=None,
                        help="Fixed focal distance. If omitted, sweep --focal-range-mm to find best.")
    parser.add_argument("--focal-range-mm", type=float, nargs=2, default=[5.0, 150.0],
                        metavar=("MIN", "MAX"),
                        help="Focal distance sweep range in mm (default: 5 150).")
    parser.add_argument("--focal-steps", type=int, default=30,
                        help="Number of focal distances to evaluate in log-spaced sweep (default: 30).")
    parser.add_argument("--focal-sweep-iters", type=int, default=30,
                        help="IASA iterations per focal-distance probe during sweep (default: 30).")
    parser.add_argument("--frequency-hz", type=float, default=2.15e6)
    parser.add_argument("--transducer-diameter-mm", type=float, default=20.0,
                        help="Lens aperture diameter in mm.")
    parser.add_argument("--n-grid", type=int, default=400,
                        help="IASA grid size (NxN).")
    parser.add_argument("--c-water", type=float, default=1480.0)
    parser.add_argument("--c-lens", type=float, default=2636.0)
    parser.add_argument("--h-base-mm", type=float, default=1.0)
    parser.add_argument("--source-pressure-pa", type=float, default=0.05e6)
    parser.add_argument("--clip-percentile", type=float, default=99.5)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--roi-pad-px", type=int, default=5,
                        help="Padding in pixels around nonzero target region for ROI mask (default: 5).")
    parser.add_argument("--outside-suppression", type=float, default=None,
                        help="Soft suppression factor [0,1] for outside-ROI amplitude each iteration. "
                             "0 = force zeros (classic IASA), 1 = free evolution. "
                             "If omitted, sweeps [0.0, 0.3, 0.5, 0.7, 0.85, 0.95, 1.0] and picks best.")
    parser.add_argument(
        "--stl-grid-stride",
        type=int,
        default=4,
        help="Downsample factor for STL export mesh (>=1). Higher is faster/coarser.",
    )
    parser.add_argument("--skip-stl", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--method",
        choices=["iasa", "backprop"],
        default="iasa",
        help="Lens design method: 'iasa' = ROI-constrained iterative IASA (default); "
             "'backprop' = raw complex backprop + phase-only projection (single-step).",
    )
    parser.add_argument(
        "--bridge-scale",
        type=float,
        default=0.0,
        help="Coordinate scale factor for the bridge field before IASA-grid interpolation. "
             "0.0 (default) = auto-fill: expand so the bridge domain fills the full aperture radius, "
             "making the bridge much larger on the lens grid. 1.0 = original physical scale.",
    )
    parser.add_argument(
        "--raw-norm",
        action="store_true",
        help="Use raw-amplitude normalisation: clip to [0,1] at the 99.5th percentile with no floor "
             "subtraction and no gamma compression, so the IASA target shape exactly matches the "
             "raw bridge pressure amplitude.",
    )
    parser.add_argument("--target-mode",
        choices=["bridge", "bridge_wide", "bridge_blur", "double_well", "double_well_bias", "tilt"],
        default="bridge")

    parser.add_argument("--bridge-width-scale", type=float, default=1.0)

    parser.add_argument("--tilt-strength", type=float, default=0.0)

    parser.add_argument("--z-blur-sigma", type=float, default=0.0)

    parser.add_argument("--aperture-mm-list", type=float, nargs="*")

    parser.add_argument("--frequency-list", type=float, nargs="*")

    parser.add_argument("--batch", action="store_true")

    parser.add_argument(
        "--selection-mode",
        choices=["bridge", "transport"],
        default="transport",
        help="Winner metric for sweep selection. 'bridge' keeps old morphology score; "
             "'transport' uses Gor'kov transport score A->B with neighbour penalty.",
    )

    parser.add_argument("--particle-radius-mm", type=float, default=0.05)
    parser.add_argument("--particle-density", type=float, default=1050.0)
    parser.add_argument("--particle-sound-speed", type=float, default=2350.0)
    parser.add_argument("--water-density", type=float, default=998.0)

    parser.add_argument(
        "--neighbour-count",
        type=int,
        default=6,
        help="How many nearest non-(A,B) traps to include in neighbour disturbance penalty."
    )

    parser.add_argument(
        "--ab-bias-strength",
        type=float,
        default=0.0,
        help="Directional ramp bias along A\u2192B axis applied to target amplitude before IASA. "
             "0 = off. E.g. 0.5 means B-side weighted 1.5x, A-side 0.5x. Clipped positive, renormalised.",
    )

    return parser.parse_args()


def _normalise_target(raw_amp: np.ndarray, clip_pct: float, gamma: float) -> np.ndarray:
    a = np.asarray(raw_amp, dtype=float)
    a = np.maximum(a, 0.0)
    a_floor = float(np.percentile(a, 1.0))
    a = np.maximum(a - a_floor, 0.0)
    a_clip = float(np.percentile(a, clip_pct))
    if a_clip <= 0.0:
        return np.zeros_like(a)
    a = np.clip(a / a_clip, 0.0, 1.0)
    g = max(float(gamma), 1.0e-6)
    return np.power(a, g)


def _save_map(data: np.ndarray, title: str, cbar: str, cmap: str, out: Path, vmin=None, vmax=None) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(data, origin="lower", cmap=cmap, aspect="equal", vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label=cbar)
    ax.set_title(title)
    ax.set_xlabel("grid x")
    ax.set_ylabel("grid y")
    fig.tight_layout()
    fig.savefig(out, dpi=190, bbox_inches="tight")
    plt.close(fig)


def _build_bridge_masks(
    roi_mask: np.ndarray,
    dx: float,
    a_px: tuple,
    b_px: tuple,
    n_centreline: int = 30,
) -> dict:
    """Pre-compute fixed spatial masks for bridge morphology scoring.

    All masks are boolean arrays matching the shape of `roi_mask`.
    Computed once at setup; passed to _compute_bridge_morphology each probe.
    """
    n = roi_mask.shape[0]
    a_r, a_c = a_px
    b_r, b_c = b_px

    # Row range (A has higher y → higher row index; B is lower)
    row_min = min(a_r, b_r)
    row_max = max(a_r, b_r)
    mid_r   = (row_min + row_max) // 2

    # Centreline sample coords from A → B (scipy map_coordinates order: [rows, cols])
    t = np.linspace(0.0, 1.0, n_centreline)
    cl_rows = a_r + t * (b_r - a_r)
    cl_cols = a_c + t * (b_c - a_c)

    # Upper half (A-side, rows [mid_r, row_max]) and lower half (B-side, rows [row_min, mid_r])
    upper_mask = roi_mask.copy()
    upper_mask[:mid_r, :]     = False   # keep rows >= mid_r
    upper_mask[row_max + 1:, :] = False
    lower_mask = roi_mask.copy()
    lower_mask[mid_r:, :]   = False     # keep rows < mid_r
    lower_mask[:row_min, :] = False

    # Estimate corridor half-width from x-extent of ROI at midpoint row
    row_slice = roi_mask[mid_r, :]
    if np.any(row_slice):
        cols_on = np.where(row_slice)[0]
        corridor_half_px = max(1, (int(cols_on[-1]) - int(cols_on[0])) // 2)
    else:
        corridor_half_px = max(1, int(round(1.5e-4 / dx)))  # 150 µm fallback

    # Side-band masks: same row range as bridge, just outside corridor
    gap_px  = max(1, corridor_half_px // 3)
    band_px = max(2, corridor_half_px)
    side_mask = np.zeros(roi_mask.shape, dtype=bool)
    for c0, c1 in [
        (a_c - corridor_half_px - gap_px - band_px,  a_c - corridor_half_px - gap_px),
        (a_c + corridor_half_px + gap_px,             a_c + corridor_half_px + gap_px + band_px),
    ]:
        c0c = max(0, c0); c1c = min(n - 1, c1)
        if c0c < c1c:
            side_mask[row_min:row_max + 1, c0c:c1c + 1] = True

    return dict(
        upper_mask=upper_mask, lower_mask=lower_mask, side_mask=side_mask,
        cl_rows=cl_rows, cl_cols=cl_cols,
        row_min=row_min, row_max=row_max, mid_r=mid_r,
        a_r=a_r, a_c=a_c, b_r=b_r, b_c=b_c,
        corridor_half_px=corridor_half_px,
    )


def _compute_bridge_morphology(
    amp: np.ndarray,
    roi_mask: np.ndarray,
    bm: dict,
    compactness_top_k_frac: float = 0.05,
) -> dict:
    """Compute five bridge-morphology sub-scores and a combined bridge_score.

    Sub-scores (all 0–1 unless noted):
      1. corridor_continuity  – min/max along A→B centreline (want HIGH)
      2. lower_balance_score  – energy ratio lower-half / upper-half (want ~1)
      3. anisotropy_score     – 1 - 1/eigval_ratio (want HIGH: elongated along bridge)
      4. side_leakage         – energy in side-bands / ROI energy (want LOW)
      5. compactness_top5pct  – fraction of ROI energy in top-5% pixels (want LOW)

    bridge_score = w1*continuity + w2*lower_balance + w3*anisotropy
                  - w4*side_leakage - w5*compactness
    """
    from scipy.ndimage import map_coordinates

    # ── 1. Corridor continuity ──────────────────────────────────────────
    cl_amp  = map_coordinates(amp, [bm["cl_rows"], bm["cl_cols"]],
                              order=1, mode="constant", cval=0.0)
    cl_max  = float(np.max(cl_amp)) + 1e-12
    continuity_score   = float(np.min(cl_amp)) / cl_max
    frac_elevated      = float(np.mean(cl_amp > 0.30 * cl_max))

    # ── 2. Lower-half energy balance ────────────────────────────────────
    e_upper = float(np.mean(amp[bm["upper_mask"]] ** 2)) if np.any(bm["upper_mask"]) else 0.0
    e_lower = float(np.mean(amp[bm["lower_mask"]] ** 2)) if np.any(bm["lower_mask"]) else 0.0
    lower_balance       = e_lower / (e_upper + 1e-12)
    lower_balance_score = float(min(lower_balance, 1.0))

    # ── 3. Anisotropy (amplitude²-weighted covariance in ROI) ───────────
    roi_pts = np.argwhere(roi_mask)
    if len(roi_pts) >= 10:
        rr, cc = roi_pts[:, 0], roi_pts[:, 1]
        w = amp[rr, cc] ** 2
        w_sum = w.sum() + 1e-12
        mr, mc = (w * rr).sum() / w_sum, (w * cc).sum() / w_sum
        dr, dc = rr - mr, cc - mc
        cov = np.array([[(w * dr**2).sum() / w_sum, (w * dr * dc).sum() / w_sum],
                        [(w * dr * dc).sum() / w_sum, (w * dc**2).sum() / w_sum]])
        ev = np.linalg.eigvalsh(cov)
        lam1 = max(float(ev[-1]), 0.0)
        lam2 = max(float(ev[0]),  0.0)
        axis_ratio      = lam1 / (lam2 + 1e-12)
        # 0 for circular, → 1 as elongation → ∞
        anisotropy_score = float(np.clip(1.0 - 1.0 / max(axis_ratio, 1.0), 0.0, 1.0))
    else:
        axis_ratio, anisotropy_score = 1.0, 0.0

    # ── 4. Side leakage ─────────────────────────────────────────────────
    roi_e  = float(np.mean(amp[roi_mask] ** 2)) if np.any(roi_mask) else 1e-12
    side_e = float(np.mean(amp[bm["side_mask"]] ** 2)) if np.any(bm["side_mask"]) else 0.0
    side_leakage = float(np.clip(side_e / (roi_e + 1e-12), 0.0, 1.0))

    # ── 5. Compactness / anti-focus penalty ─────────────────────────────
    roi_vals = amp[roi_mask]
    n_top    = max(1, int(round(compactness_top_k_frac * len(roi_vals))))
    roi_sq   = roi_vals ** 2
    top_k_e  = float(np.sum(np.partition(roi_sq, -n_top)[-n_top:]))
    total_e  = float(np.sum(roi_sq)) + 1e-12
    compactness = top_k_e / total_e

    # ── Combined bridge score ────────────────────────────────────────────
    w1, w2, w3, w4, w5 = 1.0, 0.5, 0.8, 0.4, 0.5
    bridge_score = (
        w1 * continuity_score
        + w2 * lower_balance_score
        + w3 * anisotropy_score
        - w4 * side_leakage
        - w5 * compactness
    )

    return dict(
        corridor_continuity=continuity_score,
        centreline_elevated_frac=frac_elevated,
        centreline_min_amp=float(np.min(cl_amp)),
        centreline_max_amp=float(cl_max),
        lower_half_energy=e_lower,
        upper_half_energy=e_upper,
        lower_upper_ratio=lower_balance,
        lower_balance_score=lower_balance_score,
        axis_eigenvalue_ratio=axis_ratio,
        anisotropy_score=anisotropy_score,
        side_leakage=side_leakage,
        compactness_top5pct=compactness,
        bridge_score=bridge_score,
        weights=dict(w_continuity=w1, w_lower_balance=w2, w_anisotropy=w3,
                     w_side_leakage=w4, w_compactness=w5),
    )


def widen_bridge(target, scale):

    if scale == 1.0:
        return target

    sigma = max(0.5, scale)

    return gaussian_filter(target, sigma)


def blur_target(target, sigma):

    if sigma <= 0:
        return target

    return gaussian_filter(target, sigma)


def add_tilt(target, xg, yg, strength):

    if strength == 0:
        return target

    tilt = (yg - yg.mean()) * strength

    tilt = tilt / np.max(np.abs(tilt))

    return np.clip(target + tilt, 0, None)


def make_double_well(target, a_px, b_px, width=10):

    out = np.zeros_like(target)

    ar, ac = a_px
    br, bc = b_px

    for r in range(out.shape[0]):
        for c in range(out.shape[1]):

            da = (r-ar)**2 + (c-ac)**2
            db = (r-br)**2 + (c-bc)**2

            out[r,c] = np.exp(-da/(2*width**2)) + 1.3*np.exp(-db/(2*width**2))

    return out / out.max()


def make_double_well_bias(a_px, b_px, n_grid, dx, ab_bias_strength=0.5, width_m=150e-6):
    """Two Gaussian spots at A and B; B amplitude = 1 + ab_bias_strength, A = 1.

    Width is set by width_m (physical metres), converted to pixels via dx.
    The result is a clean float32 array, normalised to [0, 1].
    """
    width_px = max(width_m / dx, 2.0)
    rr, cc = np.mgrid[0:n_grid, 0:n_grid].astype(float)
    ar, ac = a_px
    br, bc = b_px
    da2 = (rr - ar) ** 2 + (cc - ac) ** 2
    db2 = (rr - br) ** 2 + (cc - bc) ** 2
    amp_b = max(1.0 + float(ab_bias_strength), 0.0)
    out = np.exp(-da2 / (2.0 * width_px ** 2)) + amp_b * np.exp(-db2 / (2.0 * width_px ** 2))
    peak = float(np.max(out))
    return out / peak if peak > 0.0 else out


def run_batch(args):

    modes = [
        "bridge",
        "bridge_wide",
        "bridge_blur",
        "double_well",
        "tilt",
    ]

    apertures = args.aperture_mm_list or [20, 30, 40]

    freqs = args.frequency_list or [2.0e6, 2.15e6, 2.3e6]

    for m in modes:
        for ap in apertures:
            for f in freqs:

                args.target_mode = m
                args.transducer_diameter_mm = ap
                args.frequency_hz = f

                print("RUN:", m, ap, f)

                main_single(args)


def _compute_velocity_from_pressure(
    p: np.ndarray,
    dx: float,
    omega: float,
    rho_w: float,
):
    # np.gradient returns [d/dy, d/dx]
    dp_dy, dp_dx = np.gradient(p, dx, dx)
    vx = -(1.0 / (1j * omega * rho_w)) * dp_dx
    vy = -(1.0 / (1j * omega * rho_w)) * dp_dy
    return vx, vy


def _compute_gorkov_fields(
    p: np.ndarray,
    dx: float,
    frequency_hz: float,
    c_water: float,
    rho_w: float,
    particle_radius_m: float,
    rho_p: float,
    c_p: float,
):
    omega = 2.0 * np.pi * frequency_hz
    vx, vy = _compute_velocity_from_pressure(p, dx, omega, rho_w)
    v2 = np.abs(vx) ** 2 + np.abs(vy) ** 2
    p2 = np.abs(p) ** 2

    kappa_w = 1.0 / (rho_w * c_water**2)
    kappa_p = 1.0 / (rho_p * c_p**2)

    f1 = 1.0 - (kappa_p / kappa_w)
    f2 = 2.0 * (rho_p - rho_w) / (2.0 * rho_p + rho_w)

    Vp = (4.0 / 3.0) * np.pi * particle_radius_m**3

    U = Vp * (
        (f1 * p2) / (4.0 * rho_w * c_water**2)
        - (3.0 * f2 * rho_w * v2) / 8.0
    )

    dU_dy, dU_dx = np.gradient(U, dx, dx)
    Fx = -dU_dx
    Fy = -dU_dy

    return {
        "U": U,
        "Fx": Fx,
        "Fy": Fy,
        "vx": vx,
        "vy": vy,
    }


def _sample_field(field: np.ndarray, row: float, col: float) -> float:
    return float(map_coordinates(field, [[row], [col]], order=1, mode="nearest")[0])


def _sample_complex_field(field: np.ndarray, row: float, col: float) -> complex:
    re = map_coordinates(field.real, [[row], [col]], order=1, mode="nearest")[0]
    im = map_coordinates(field.imag, [[row], [col]], order=1, mode="nearest")[0]
    return complex(re, im)


def _build_trap_pixel_list(
    traps_raw: np.ndarray,
    idx_a_t: int,
    idx_b_t: int,
    x_center: float,
    y_center: float,
    bridge_scale: float,
    dx: float,
    n_grid: int,
):
    n_half = n_grid // 2

    def _to_px(pos_c):
        r = int(round(pos_c[1] / dx + n_half))
        c = int(round(pos_c[0] / dx + n_half))
        return (max(0, min(n_grid - 1, r)), max(0, min(n_grid - 1, c)))

    px_list = []
    for i in range(traps_raw.shape[0]):
        pos_c = (traps_raw[i][:2] - np.array([x_center, y_center])) * bridge_scale
        px = _to_px(pos_c)
        px_list.append(px)

    return px_list


def _compute_transport_metrics(
    p_field: np.ndarray,
    dx: float,
    frequency_hz: float,
    c_water: float,
    rho_w: float,
    particle_radius_m: float,
    rho_p: float,
    c_p: float,
    a_px: tuple,
    b_px: tuple,
    neighbour_px_list: list,
):
    g = _compute_gorkov_fields(
        p=p_field,
        dx=dx,
        frequency_hz=frequency_hz,
        c_water=c_water,
        rho_w=rho_w,
        particle_radius_m=particle_radius_m,
        rho_p=rho_p,
        c_p=c_p,
    )

    U = g["U"]
    Fx = g["Fx"]
    Fy = g["Fy"]

    a_r, a_c = a_px
    b_r, b_c = b_px

    # Unit vector from A -> B in row/col convention
    dr = float(b_r - a_r)
    dc = float(b_c - a_c)
    norm_ab = max(np.hypot(dr, dc), 1e-12)
    er = dr / norm_ab
    ec = dc / norm_ab

    # Force at A projected onto A->B direction
    FxA = _sample_field(Fx, a_r, a_c)
    FyA = _sample_field(Fy, a_r, a_c)
    Fproj_A = FyA * er + FxA * ec
    Fmag_A = np.hypot(FxA, FyA)
    force_alignment = Fproj_A / (Fmag_A + 1e-12)

    UA = _sample_field(U, a_r, a_c)
    UB = _sample_field(U, b_r, b_c)

    deltaU = UA - UB
    deltaU_norm = deltaU / (abs(UA) + abs(UB) + 1e-18)

    # Barrier along centreline
    t = np.linspace(0.0, 1.0, 50)
    cl_rows = a_r + t * (b_r - a_r)
    cl_cols = a_c + t * (b_c - a_c)
    U_line = map_coordinates(U, [cl_rows, cl_cols], order=1, mode="nearest")
    U_barrier = float(np.max(U_line) - max(UA, UB))
    U_scale = max(abs(UA - UB), 1e-18)
    barrier_penalty = max(U_barrier, 0.0) / U_scale

    # Neighbour penalty
    neigh_force_vals = []
    for nr, nc in neighbour_px_list:
        FxN = _sample_field(Fx, nr, nc)
        FyN = _sample_field(Fy, nr, nc)
        neigh_force_vals.append(np.hypot(FxN, FyN))

    mean_neigh_force = float(np.mean(neigh_force_vals)) if len(neigh_force_vals) else 0.0
    neighbour_penalty = mean_neigh_force / (Fmag_A + 1e-12)

    # Final transport score — high is good
    transport_score = (
        1.5 * force_alignment
        + 1.2 * deltaU_norm
        - 0.8 * barrier_penalty
        - 0.8 * neighbour_penalty
    )

    return {
        "U": U,
        "Fx": Fx,
        "Fy": Fy,
        "UA": float(UA),
        "UB": float(UB),
        "deltaU": float(deltaU),
        "deltaU_norm": float(deltaU_norm),
        "Fproj_A": float(Fproj_A),
        "Fmag_A": float(Fmag_A),
        "force_alignment": float(force_alignment),
        "barrier_penalty": float(barrier_penalty),
        "mean_neighbour_force": float(mean_neigh_force),
        "neighbour_penalty": float(neighbour_penalty),
        "transport_score": float(transport_score),
    }


def main_single(args) -> None:

    def _log(msg: str) -> None:
        print(msg, flush=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or (PROJECT_ROOT / "results" / "dev" / f"inverse_bridge_pressure_lens_replica_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    _log(f"[start] output_dir={out_dir}")

    # ── Load bridge pressure field ──────────────────────────────────────
    d = np.load(args.input_npz)
    _log(f"[load] input_npz={args.input_npz}")
    if args.field_key not in d.files:
        raise KeyError(f"Field key '{args.field_key}' not found in {args.input_npz}. Available: {list(d.files)}")

    p_field_full = d[args.field_key].astype(complex)
    x_target = d["x_full"].astype(float)
    y_target = d["y_full"].astype(float)
    target_raw_amp_full = np.abs(p_field_full)
    _log(f"[load] full domain: {p_field_full.shape}, "
         f"x=[{x_target[0]*1e3:.3f}, {x_target[-1]*1e3:.3f}] mm, "
         f"y=[{y_target[0]*1e3:.3f}, {y_target[-1]*1e3:.3f}] mm")

    # ── Build IASA config (focal_distance_mm placeholder — may be overwritten by sweep) ──
    focal_mm_initial = args.focal_distance_mm if args.focal_distance_mm is not None else 40.0
    cfg = ReplicaConfig(
        frequency_hz=float(args.frequency_hz),
        c_water=float(args.c_water),
        c_lens=float(args.c_lens),
        transducer_diameter_mm=float(args.transducer_diameter_mm),
        focal_distance_mm=focal_mm_initial,
        n_grid=int(args.n_grid),
        h_base_mm=float(args.h_base_mm),
        n_iter=int(args.n_iter),
        source_pressure_pa=float(args.source_pressure_pa),
        output_dir=str(out_dir),
    )

    timings: dict[str, float] = {}
    t0_all = time.perf_counter()

    # ── Grid creation ───────────────────────────────────────────────────
    t0 = time.perf_counter()
    _, _, xg, yg, _, _, aperture_mask, dx = make_grid(cfg)
    timings["grid_creation"] = time.perf_counter() - t0
    _log(f"[stage] grid_creation done in {timings['grid_creation']:.3f}s  "
         f"(dx={dx*1e6:.1f} µm/px, lens={cfg.transducer_diameter_mm:.1f} mm)")

    # ── Interpolate bridge field onto IASA grid (centred at origin) ────
    t0 = time.perf_counter()
    x_center = 0.5 * (float(x_target[0]) + float(x_target[-1]))
    y_center = 0.5 * (float(y_target[0]) + float(y_target[-1]))
    x_target_c = x_target - x_center
    y_target_c = y_target - y_center
    # ── Bridge coordinate scaling (makes bridge much larger on the IASA grid) ──
    if args.bridge_scale == 0.0:
        _field_half = max(float(np.abs(x_target_c).max()), float(np.abs(y_target_c).max()))
        bridge_scale = (cfg.transducer_diameter_mm * 0.5e-3) / max(_field_half, 1e-12)
    else:
        bridge_scale = float(args.bridge_scale)
    x_target_c = x_target_c * bridge_scale
    y_target_c = y_target_c * bridge_scale
    _log(f"[bridge-scale] scale={bridge_scale:.4f}x  "
         f"bridge domain now: ±{float(np.abs(x_target_c).max())*1e3:.2f} mm × "
         f"±{float(np.abs(y_target_c).max())*1e3:.2f} mm  (fills aperture radius="
         f"{cfg.transducer_diameter_mm*0.5:.1f} mm)")
    interp_fn = RegularGridInterpolator(
        (y_target_c, x_target_c),
        target_raw_amp_full,
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )
    pts = np.column_stack([yg.ravel(), xg.ravel()])
    target_raw_amp = interp_fn(pts).reshape(cfg.n_grid, cfg.n_grid)
    field_span_x = (x_target_c[-1] - x_target_c[0]) * 1e3
    field_span_y = (y_target_c[-1] - y_target_c[0]) * 1e3
    _log(f"[load] interpolated onto {cfg.n_grid}x{cfg.n_grid} grid. "
         f"Bridge spans {field_span_x:.2f}x{field_span_y:.2f} mm (scaled) within {cfg.transducer_diameter_mm:.1f} mm aperture. "
         f"Centred at ({x_center*1e3:.3f}, {y_center*1e3:.3f}) mm")
    timings["target_interpolation"] = time.perf_counter() - t0

    # ── Normalise target amplitude ─────────────────────────────────────
    t0 = time.perf_counter()
    if args.raw_norm:
        _a = np.maximum(target_raw_amp, 0.0)
        _a_nonzero = _a[_a > 0]
        _a_clip = float(np.percentile(_a_nonzero, 99.5)) if _a_nonzero.size > 0 else 1.0
        target_amp = np.clip(_a / _a_clip, 0.0, 1.0) * aperture_mask.astype(float)
        _log(f"[stage] raw-norm: clip at {_a_clip:.3f} Pa (99.5th pct of nonzero) — shape matches raw amplitude exactly")
    else:
        target_amp = _normalise_target(target_raw_amp, clip_pct=float(args.clip_percentile), gamma=float(args.gamma))
        target_amp = target_amp * aperture_mask.astype(float)

    if args.target_mode in ("double_well", "double_well_bias"):
        if "traps_m" not in d.files or "idx_a" not in d.files or "idx_b" not in d.files:
            raise KeyError("NPZ must contain 'traps_m', 'idx_a', 'idx_b' for bridge morphology scoring.")
        n_half_mode = cfg.n_grid // 2
        traps_raw_mode = d["traps_m"].astype(float)
        idx_a_t_mode = int(d["idx_a"])
        idx_b_t_mode = int(d["idx_b"])
        a_pos_c_mode = (traps_raw_mode[idx_a_t_mode][:2] - np.array([x_center, y_center])) * bridge_scale
        b_pos_c_mode = (traps_raw_mode[idx_b_t_mode][:2] - np.array([x_center, y_center])) * bridge_scale
        b_pos_c_mode[0] = a_pos_c_mode[0]

        def _to_px_mode(pos_c):
            r = int(round(pos_c[1] / dx + n_half_mode))
            c = int(round(pos_c[0] / dx + n_half_mode))
            return (max(0, min(cfg.n_grid - 1, r)), max(0, min(cfg.n_grid - 1, c)))

        a_px = _to_px_mode(a_pos_c_mode)
        b_px = _to_px_mode(b_pos_c_mode)

    if args.target_mode == "bridge_wide":
        target_amp = widen_bridge(target_amp, args.bridge_width_scale)

    elif args.target_mode == "bridge_blur":
        target_amp = blur_target(target_amp, args.z_blur_sigma)

    elif args.target_mode == "tilt":
        target_amp = add_tilt(target_amp, xg, yg, args.tilt_strength)

    elif args.target_mode == "double_well":
        target_amp = make_double_well(target_amp, a_px, b_px)

    elif args.target_mode == "double_well_bias":
        target_amp = make_double_well_bias(
            a_px, b_px,
            n_grid=cfg.n_grid,
            dx=dx,
            ab_bias_strength=float(args.ab_bias_strength),
        ) * aperture_mask.astype(float)
        _log(
            f"[double_well_bias] A_px={a_px}  B_px={b_px}  "
            f"A_amp=1.00  B_amp={1.0 + float(args.ab_bias_strength):.3f}  "
            f"width_px={max(150e-6 / dx, 2.0):.1f} px"
        )

    timings["target_normalisation"] = time.perf_counter() - t0

    # ── Build ROI mask (nonzero target + padding) ──────────────────────
    nonzero_mask = target_amp > 1e-6
    if args.roi_pad_px > 0:
        from scipy.ndimage import binary_dilation
        struct = np.ones((2 * args.roi_pad_px + 1, 2 * args.roi_pad_px + 1), dtype=bool)
        roi_mask = binary_dilation(nonzero_mask, structure=struct) & aperture_mask
    else:
        roi_mask = nonzero_mask & aperture_mask
    roi_frac = float(np.sum(roi_mask)) / float(np.sum(aperture_mask))
    _log(f"[stage] ROI mask: {np.sum(roi_mask)} px ({roi_frac*100:.1f}% of aperture), pad={args.roi_pad_px} px")

    # ── Bridge pixel coords on IASA grid ──────────────────────────────
    if "traps_m" not in d.files or "idx_a" not in d.files or "idx_b" not in d.files:
        raise KeyError("NPZ must contain 'traps_m', 'idx_a', 'idx_b' for bridge morphology scoring.")
    n_half    = cfg.n_grid // 2
    traps_raw = d["traps_m"].astype(float)
    idx_a_t   = int(d["idx_a"])
    idx_b_t   = int(d["idx_b"])
    # Physical positions centred on the IASA grid (same scale as the bridge field coordinates)
    a_pos_c = (traps_raw[idx_a_t][:2] - np.array([x_center, y_center])) * bridge_scale
    b_pos_c = (traps_raw[idx_b_t][:2] - np.array([x_center, y_center])) * bridge_scale
    b_pos_c[0] = a_pos_c[0]           # standalone script forces same x
    def _to_px(pos_c):
        r = int(round(pos_c[1] / dx + n_half))
        c = int(round(pos_c[0] / dx + n_half))
        return (max(0, min(cfg.n_grid - 1, r)), max(0, min(cfg.n_grid - 1, c)))
    a_px = _to_px(a_pos_c)
    b_px = _to_px(b_pos_c)
    bridge_masks = _build_bridge_masks(roi_mask, dx, a_px, b_px)
    _log(f"[bridge] A_px={a_px}  B_px={b_px}  span={abs(a_px[0]-b_px[0])} px  "
         f"corridor_half={bridge_masks['corridor_half_px']} px")

    all_trap_px = _build_trap_pixel_list(
        traps_raw=traps_raw,
        idx_a_t=idx_a_t,
        idx_b_t=idx_b_t,
        x_center=x_center,
        y_center=y_center,
        bridge_scale=bridge_scale,
        dx=dx,
        n_grid=cfg.n_grid,
    )

    ab_mid = np.array([(a_px[0] + b_px[0]) * 0.5, (a_px[1] + b_px[1]) * 0.5])
    neigh_candidates = []
    for i, px in enumerate(all_trap_px):
        if i in (idx_a_t, idx_b_t):
            continue
        dmid = float(np.hypot(px[0] - ab_mid[0], px[1] - ab_mid[1]))
        neigh_candidates.append((dmid, px))

    neigh_candidates.sort(key=lambda z: z[0])
    neighbour_px_list = [px for _, px in neigh_candidates[:max(args.neighbour_count, 0)]]

    _log(f"[neighbours] using {len(neighbour_px_list)} neighbours for disturbance penalty: {neighbour_px_list}")

    # ── A→B directional bias on target amplitude ───────────────────────
    # Skip for double_well_bias: bias is already encoded in the Gaussian amplitudes.
    if float(args.ab_bias_strength) > 0.0 and args.target_mode != "double_well_bias":
        a_r, a_c = a_px
        b_r, b_c = b_px
        dr = float(b_r - a_r)
        dc = float(b_c - a_c)
        L = max(np.hypot(dr, dc), 1e-12)
        ear, eac = dr / L, dc / L
        rr_g, cc_g = np.mgrid[0:cfg.n_grid, 0:cfg.n_grid]
        # Projection onto A→B axis; t_norm = -1 at A, +1 at B
        t_proj = (rr_g - a_r) * ear + (cc_g - a_c) * eac
        t_norm = (t_proj - L / 2.0) / (L / 2.0)
        ramp = np.maximum(1.0 + float(args.ab_bias_strength) * t_norm, 0.0)
        target_amp = target_amp * ramp
        ta_max = float(np.max(target_amp))
        if ta_max > 0.0:
            target_amp = target_amp / ta_max
        _log(
            f"[ab-bias] strength={args.ab_bias_strength:.3f}  "
            f"ramp=[{float(np.min(ramp)):.3f}, {float(np.max(ramp)):.3f}]  "
            f"A-weight={1.0 - float(args.ab_bias_strength):.3f}  "
            f"B-weight={1.0 + float(args.ab_bias_strength):.3f}"
        )

    # ── Backprop path: raw complex backprop + phase-only projection ────
    if args.method == "backprop":
        t0_bp = time.perf_counter()
        _log("[backprop] interpolating complex field onto IASA grid...")
        interp_re = RegularGridInterpolator(
            (y_target_c, x_target_c), p_field_full.real,
            method="linear", bounds_error=False, fill_value=0.0)
        interp_im = RegularGridInterpolator(
            (y_target_c, x_target_c), p_field_full.imag,
            method="linear", bounds_error=False, fill_value=0.0)
        target_complex = (
            interp_re(pts) + 1j * interp_im(pts)
        ).reshape(cfg.n_grid, cfg.n_grid) * aperture_mask
        timings["backprop_complex_interp"] = time.perf_counter() - t0_bp

        # Choose focal-distance candidates
        if args.focal_distance_mm is not None:
            bp_focal_candidates = [args.focal_distance_mm * 1e-3]
        else:
            bp_focal_candidates = np.geomspace(
                args.focal_range_mm[0] * 1e-3,
                args.focal_range_mm[1] * 1e-3,
                args.focal_steps,
            ).tolist()
        _log(f"[backprop] scanning {len(bp_focal_candidates)} focal distances...")

        best_focal_m = bp_focal_candidates[0]
        best_bridge_bp = -np.inf
        best_lens_bp = None
        sweep_results = []
        t0_sw = time.perf_counter()
        for fz in bp_focal_candidates:
            lens_back = propagate_asm(target_complex, cfg.k_water, -float(fz), dx)
            lens_ph = np.where(aperture_mask, np.exp(1j * np.angle(lens_back)), 0.0 + 0j)
            p_fwd = propagate_asm(lens_ph, cfg.k_water, float(fz), dx)
            bm_bp = _compute_bridge_morphology(np.abs(p_fwd), roi_mask, bridge_masks)
            sweep_results.append({"focal_mm": float(fz * 1e3), "bridge_score": bm_bp["bridge_score"]})
            tag = ""
            if bm_bp["bridge_score"] > best_bridge_bp:
                best_bridge_bp = bm_bp["bridge_score"]
                best_focal_m = float(fz)
                best_lens_bp = lens_ph
                tag = "  *** best"
            _log(f"  [backprop] f={fz*1e3:.2f} mm  bridge_score={bm_bp['bridge_score']:.4f}{tag}")
        timings["backprop_sweep"] = time.perf_counter() - t0_sw
        _log(f"[backprop] best: f={best_focal_m*1e3:.2f} mm  bridge_score={best_bridge_bp:.4f}")

        lens_field = best_lens_bp
        best_suppression = None   # not applicable
        timings["iasa_total"] = 0.0
        timings["iasa_per_iteration"] = 0.0

    # ── Helper: evaluate a probe run (IASA path) ───────────────────────
    def _eval_probe(fz_m: float, sup: float, iters: int) -> dict:
        """Run a quick IASA probe and return energy-based metrics."""
        lens_p = run_iasa_roi(
            cfg, aperture_mask, target_amp, roi_mask, dx,
            n_iter=iters, focal_distance_m=fz_m, outside_suppression=sup,
        )
        p_p = propagate_asm(lens_p, cfg.k_water, fz_m, dx)
        amp_p = np.abs(p_p)
        roi_e = float(np.mean(amp_p[roi_mask] ** 2))
        out_mask = aperture_mask & ~roi_mask
        out_e = float(np.mean(amp_p[out_mask] ** 2)) if np.any(out_mask) else 1e-30
        eratio = roi_e / (out_e + 1e-30)
        # Also compute ROI correlation for reporting
        ta_r = target_amp[roi_mask].ravel()
        ra_r = amp_p[roi_mask].ravel()
        if ra_r.max() > 0:
            ra_r_n = ra_r / (np.percentile(ra_r, 99.5) + 1e-12)
        else:
            ra_r_n = ra_r
        corr = float(np.corrcoef(ta_r, ra_r_n)[0, 1]) if np.std(ta_r) > 0 and np.std(ra_r_n) > 0 else 0.0
        bm_p = _compute_bridge_morphology(amp_p, roi_mask, bridge_masks)
        tm_p = _compute_transport_metrics(
            p_field=p_p,
            dx=dx,
            frequency_hz=cfg.frequency_hz,
            c_water=cfg.c_water,
            rho_w=float(args.water_density),
            particle_radius_m=float(args.particle_radius_mm) * 1e-3,
            rho_p=float(args.particle_density),
            c_p=float(args.particle_sound_speed),
            a_px=a_px,
            b_px=b_px,
            neighbour_px_list=neighbour_px_list,
        )
        return {
            "energy_ratio": eratio,
            "roi_corr": corr,
            "bridge_score": bm_p["bridge_score"],
            "transport_score": tm_p["transport_score"],
            "force_alignment": tm_p["force_alignment"],
            "deltaU_norm": tm_p["deltaU_norm"],
            "barrier_penalty": tm_p["barrier_penalty"],
            "neighbour_penalty": tm_p["neighbour_penalty"],
        }

    # ── Joint focal-distance + suppression sweep (IASA path) ─────────
    if args.method == "iasa":
        sweep_results = []
        best_focal_m = cfg.focal_distance_m
        best_suppression = 0.0 if args.outside_suppression is None else float(args.outside_suppression)

    if args.method == "iasa" and (args.focal_distance_mm is None or args.outside_suppression is None):
        t0 = time.perf_counter()
        # Focal candidates
        if args.focal_distance_mm is None:
            f_min_m = args.focal_range_mm[0] * 1e-3
            f_max_m = args.focal_range_mm[1] * 1e-3
            focal_candidates = np.geomspace(f_min_m, f_max_m, args.focal_steps).tolist()
        else:
            focal_candidates = [cfg.focal_distance_m]

        # Suppression candidates
        if args.outside_suppression is None:
            sup_candidates = [0.0, 0.3, 0.5, 0.7, 0.85, 0.95, 1.0]
        else:
            sup_candidates = [float(args.outside_suppression)]

        n_combos = len(focal_candidates) * len(sup_candidates)
        _log(f"[sweep] scanning {len(focal_candidates)} focal × {len(sup_candidates)} suppression = {n_combos} combos "
             f"with {args.focal_sweep_iters} iters each...")

        best_bridge = -np.inf
        best_eratio = -1.0
        combo_idx = 0
        winner_label = "bridge_score"

        for fz in focal_candidates:
            for sup in sup_candidates:
                combo_idx += 1
                m = _eval_probe(float(fz), sup, args.focal_sweep_iters)

                if args.selection_mode == "transport":
                    winner_metric = m["transport_score"]
                    winner_label = "transport_score"
                else:
                    winner_metric = m["bridge_score"]
                    winner_label = "bridge_score"

                sweep_results.append({
                    "focal_mm": float(fz * 1e3),
                    "suppression": sup,
                    "energy_ratio": m["energy_ratio"],
                    "roi_corr": m["roi_corr"],
                    "bridge_score": m["bridge_score"],
                    "transport_score": m["transport_score"],
                    "force_alignment": m["force_alignment"],
                    "deltaU_norm": m["deltaU_norm"],
                    "barrier_penalty": m["barrier_penalty"],
                    "neighbour_penalty": m["neighbour_penalty"],
                })

                tag = ""
                if winner_metric > best_bridge:
                    best_bridge = winner_metric
                    best_eratio = m["energy_ratio"]
                    best_focal_m = float(fz)
                    best_suppression = sup
                    tag = "  *** best"

                _log(
                    f"  [{combo_idx}/{n_combos}] f={fz*1e3:7.2f} mm  sup={sup:.2f}  "
                    f"roi_corr={m['roi_corr']:.4f}  "
                    f"bridge_score={m['bridge_score']:.4f}  "
                    f"transport_score={m['transport_score']:.4f}  "
                    f"force_align={m['force_alignment']:.3f}  "
                    f"deltaU_norm={m['deltaU_norm']:.3f}  "
                    f"neigh_pen={m['neighbour_penalty']:.3f}{tag}"
                )

        timings["parameter_sweep"] = time.perf_counter() - t0
        _log(
            f"[sweep] best: f={best_focal_m*1e3:.2f} mm, sup={best_suppression:.2f}, "
            f"{winner_label}={best_bridge:.4f}  (energy_ratio={best_eratio:.3f})"
        )
    elif args.method == "iasa":
        _log(f"[config] fixed focal={cfg.focal_distance_mm:.2f} mm, sup={best_suppression:.2f}")

    # Rebuild config with optimal focal distance
    cfg = ReplicaConfig(
        frequency_hz=cfg.frequency_hz,
        c_water=cfg.c_water,
        c_lens=cfg.c_lens,
        transducer_diameter_mm=cfg.transducer_diameter_mm,
        focal_distance_mm=best_focal_m * 1e3,
        n_grid=cfg.n_grid,
        h_base_mm=cfg.h_base_mm,
        n_iter=cfg.n_iter,
        source_pressure_pa=cfg.source_pressure_pa,
        output_dir=str(out_dir),
    )

    sup_str = f"{best_suppression:.2f}" if best_suppression is not None else "N/A"
    method_label = args.method if args.method == "iasa" else "raw complex backprop + phase-only projection"
    _log(
        f"[config] n_grid={cfg.n_grid}, n_iter={cfg.n_iter}, "
        f"transducer_diameter_mm={cfg.transducer_diameter_mm:.1f}, "
        f"frequency_hz={cfg.frequency_hz:.0f}, "
        f"focal_distance_mm={cfg.focal_distance_mm:.2f}, "
        f"outside_suppression={sup_str}, "
        f"method={method_label}, "
        f"skip_stl={args.skip_stl}, stl_grid_stride={max(int(args.stl_grid_stride), 1)}"
    )

    # ── Full ROI-constrained IASA (iasa path only) ─────────────────────
    if args.method == "iasa":
        t0 = time.perf_counter()
        _log(f"[stage] run_iasa_roi starting ({cfg.n_iter} iters, f={cfg.focal_distance_mm:.2f} mm, "
             f"sup={best_suppression:.2f})...")
        lens_field = run_iasa_roi(cfg, aperture_mask, target_amp, roi_mask, dx,
                                  outside_suppression=best_suppression)
        timings["iasa_total"] = time.perf_counter() - t0
        timings["iasa_per_iteration"] = timings["iasa_total"] / max(cfg.n_iter, 1)
        _log(f"[stage] run_iasa_roi done in {timings['iasa_total']:.3f}s "
             f"({timings['iasa_per_iteration']:.4f}s/iter)")

    # ── Reconstruction ─────────────────────────────────────────────────
    t0 = time.perf_counter()
    p_recon = propagate_asm(lens_field, cfg.k_water, cfg.focal_distance_m, dx)
    scale = cfg.source_pressure_pa / (np.sqrt(np.mean(np.abs(lens_field[aperture_mask]) ** 2)) + 1e-12)
    recon_amp = np.abs(p_recon) * scale
    # Normalise using ROI peak so the bridge region is visible, not washed out
    roi_p995 = float(np.percentile(recon_amp[roi_mask], 99.5)) if np.any(roi_mask) else 1.0
    recon_amp_n = recon_amp / (roi_p995 + 1e-12)
    recon_amp_n = np.clip(recon_amp_n, 0.0, 1.0)
    timings["reconstruction"] = time.perf_counter() - t0
    _log(f"[stage] reconstruction done in {timings['reconstruction']:.3f}s")

    # ── Thickness mapping ──────────────────────────────────────────────
    phi_wrapped = np.mod(np.angle(lens_field), 2.0 * np.pi)
    thickness = cfg.h_base_m + cfg.h_max_m * (phi_wrapped / (2.0 * np.pi))
    thickness[~aperture_mask] = cfg.h_base_m

    # ── Metrics ────────────────────────────────────────────────────────
    # Correlation inside full aperture (against processed target_amp)
    ta_full = target_amp[aperture_mask].ravel()
    ra_full = recon_amp_n[aperture_mask].ravel()
    corr_full = float(np.corrcoef(ta_full, ra_full)[0, 1]) if np.std(ta_full) > 0 and np.std(ra_full) > 0 else float("nan")

    # Correlation inside ROI only (against processed target_amp)
    ta_roi = target_amp[roi_mask].ravel()
    ra_roi = recon_amp_n[roi_mask].ravel()
    corr_roi = float(np.corrcoef(ta_roi, ra_roi)[0, 1]) if np.std(ta_roi) > 0 and np.std(ra_roi) > 0 else float("nan")

    # Correlation against RAW amplitude (key metric for backprop path)
    raw_p995 = float(np.percentile(target_raw_amp[roi_mask], 99.5)) if np.any(roi_mask) else 1.0
    ta_roi_raw = (target_raw_amp[roi_mask] / (raw_p995 + 1e-12)).ravel()
    corr_roi_raw = float(np.corrcoef(ta_roi_raw, ra_roi)[0, 1]) if np.std(ta_roi_raw) > 0 and np.std(ra_roi) > 0 else float("nan")
    ta_full_raw = (target_raw_amp[aperture_mask] / (raw_p995 + 1e-12)).ravel()
    corr_full_raw = float(np.corrcoef(ta_full_raw, ra_full)[0, 1]) if np.std(ta_full_raw) > 0 and np.std(ra_full) > 0 else float("nan")

    # RMSE inside ROI (normalised amplitudes)
    rmse_roi = float(np.sqrt(np.mean((ta_roi - ra_roi) ** 2)))

    # Energy concentration: mean squared amplitude in ROI vs outside
    roi_energy = float(np.mean(recon_amp[roi_mask] ** 2))
    outside_roi_in_aperture = aperture_mask & ~roi_mask
    outside_energy = float(np.mean(recon_amp[outside_roi_in_aperture] ** 2)) if np.any(outside_roi_in_aperture) else 0.0
    energy_ratio = roi_energy / (outside_energy + 1e-30)

    _log(f"[metrics] corr_roi={corr_roi:.4f}  corr_roi_raw={corr_roi_raw:.4f}  "
         f"corr_full={corr_full:.4f}  rmse_roi={rmse_roi:.4f}  energy_ratio={energy_ratio:.2f}")

    # ── Bridge morphology metrics (full run) ───────────────────────────
    bridge_morph = _compute_bridge_morphology(recon_amp, roi_mask, bridge_masks)
    _log(
        f"[bridge_score] {bridge_morph['bridge_score']:.4f}  "
        f"continuity={bridge_morph['corridor_continuity']:.3f}  "
        f"lower_bal={bridge_morph['lower_balance_score']:.3f}  "
        f"aniso={bridge_morph['anisotropy_score']:.3f}  "
        f"leak={bridge_morph['side_leakage']:.3f}  "
        f"compact={bridge_morph['compactness_top5pct']:.3f}"
    )

    transport_metrics = _compute_transport_metrics(
        p_field=p_recon * scale,
        dx=dx,
        frequency_hz=cfg.frequency_hz,
        c_water=cfg.c_water,
        rho_w=float(args.water_density),
        particle_radius_m=float(args.particle_radius_mm) * 1e-3,
        rho_p=float(args.particle_density),
        c_p=float(args.particle_sound_speed),
        a_px=a_px,
        b_px=b_px,
        neighbour_px_list=neighbour_px_list,
    )
    _log(
        f"[transport] score={transport_metrics['transport_score']:.4f}  "
        f"force_align={transport_metrics['force_alignment']:.3f}  "
        f"deltaU_norm={transport_metrics['deltaU_norm']:.3f}  "
        f"barrier_pen={transport_metrics['barrier_penalty']:.3f}  "
        f"neigh_pen={transport_metrics['neighbour_penalty']:.3f}"
    )

    # ── Figures ────────────────────────────────────────────────────────
    _log("[stage] saving figures...")
    _save_map(target_raw_amp, f"Raw bridge target amplitude ({args.field_key})", "|p| [Pa]", "inferno",
              out_dir / "target_raw_bridge_amplitude.png")
    _save_map(target_amp, "Normalised target amplitude (ROI-constrained IASA)", "a.u.", "inferno",
              out_dir / "target_amplitude_for_iasa.png", vmin=0.0, vmax=1.0)
    _save_map(phi_wrapped, "Lens wrapped phase", "phase [rad]", "twilight",
              out_dir / "lens_phase_wrapped.png", vmin=0.0, vmax=2.0 * np.pi)
    _save_map(recon_amp * 1e-3, "Reconstructed pressure at focus", "|p| [kPa]", "inferno",
              out_dir / "reconstructed_pressure_at_focus.png")
    _save_map(thickness * 1e3, "Lens thickness map", "thickness [mm]", "viridis",
              out_dir / "lens_thickness_map.png")

    # Target vs Reconstruction — show raw-amplitude target for backprop, normalised for IASA
    target_for_display = target_raw_amp / (raw_p995 + 1e-12) if args.method == "backprop" else target_amp
    target_label = "Target (raw |p|, normalised)" if args.method == "backprop" else "Target (processed, normalised)"
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))
    im0 = axes[0].imshow(np.clip(target_for_display, 0.0, 1.0), origin="lower", cmap="inferno", vmin=0.0, vmax=1.0)
    axes[0].set_title(target_label)
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.03)
    im1 = axes[1].imshow(recon_amp_n, origin="lower", cmap="inferno", vmin=0.0, vmax=1.0)
    axes[1].set_title("Reconstruction (normalised)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.03)
    fig.suptitle(
        f"Target vs Reconstruction [{args.method}]  "
        f"ROI corr_raw={corr_roi_raw:.4f}  corr_proc={corr_roi:.4f}  full={corr_full:.4f}"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_dir / "target_vs_reconstruction.png", dpi=190, bbox_inches="tight")
    plt.close(fig)

    # Sweep plot (if sweep was done)
    if sweep_results:
        if args.method == "backprop":
            # Backprop sweep: single panel — bridge_score vs focal distance
            fig_sw, ax_sw_bp = plt.subplots(figsize=(7, 4.5))
            fs_bp = [r["focal_mm"] for r in sweep_results]
            bs_bp = [r["bridge_score"] for r in sweep_results]
            ax_sw_bp.plot(fs_bp, bs_bp, "o-", markersize=5, color="steelblue")
            ax_sw_bp.axvline(cfg.focal_distance_mm, color="red", ls="--", alpha=0.8, label="chosen")
            ax_sw_bp.set_xlabel("Focal distance [mm]")
            ax_sw_bp.set_ylabel("Bridge score (morphology)")
            ax_sw_bp.set_title("Backprop focal sweep — bridge_score  ← winner metric")
            ax_sw_bp.set_xscale("log")
            ax_sw_bp.legend(fontsize=8)
            ax_sw_bp.grid(True, alpha=0.3)
            fig_sw.suptitle(f"Backprop sweep — best f={cfg.focal_distance_mm:.1f} mm")
            fig_sw.tight_layout()
            fig_sw.savefig(out_dir / "parameter_sweep.png", dpi=190, bbox_inches="tight")
            plt.close(fig_sw)
        else:
            # IASA sweep: 3-panel
            fig_sw, axes_sw = plt.subplots(1, 3, figsize=(18, 4.5))
            sup_vals = sorted(set(r["suppression"] for r in sweep_results))
            for sv in sup_vals:
                pts_s = [r for r in sweep_results if r["suppression"] == sv]
                fs = [r["focal_mm"] for r in pts_s]
                er = [r["energy_ratio"] for r in pts_s]
                cr = [r["roi_corr"] for r in pts_s]
                bs = [r.get("bridge_score", float("nan")) for r in pts_s]
                axes_sw[0].plot(fs, er, "o-", markersize=3, label=f"sup={sv:.2f}")
                axes_sw[1].plot(fs, cr, "o-", markersize=3, label=f"sup={sv:.2f}")
                axes_sw[2].plot(fs, bs, "o-", markersize=3, label=f"sup={sv:.2f}")
            for ax_sw in axes_sw:
                ax_sw.axvline(cfg.focal_distance_mm, color="red", ls="--", alpha=0.7, label="chosen")
                ax_sw.set_xlabel("Focal distance [mm]")
                ax_sw.set_xscale("log")
                ax_sw.legend(fontsize=7)
                ax_sw.grid(True, alpha=0.3)
            axes_sw[0].set_ylabel("Energy ratio (ROI/outside)")
            axes_sw[0].set_title("Energy concentration")
            axes_sw[1].set_ylabel("ROI correlation")
            axes_sw[1].set_title("ROI shape fidelity")
            axes_sw[2].set_ylabel("Bridge score (morphology)")
            axes_sw[2].set_title("Bridge morphology score  ← winner metric")
            fig_sw.suptitle(
                f"Parameter sweep — best: f={cfg.focal_distance_mm:.1f} mm, sup={best_suppression:.2f}"
                f"  (selected by bridge_score)"
            )
            fig_sw.tight_layout(rect=[0, 0, 1, 0.93])
            fig_sw.savefig(out_dir / "parameter_sweep.png", dpi=190, bbox_inches="tight")
            plt.close(fig_sw)

    # Bridge morphology debug figure (masks + centreline overlay)
    fig_bm, axes_bm = plt.subplots(1, 2, figsize=(11.0, 4.8))
    # Left: reconstruction + centreline samples + A/B markers
    ax_bm0 = axes_bm[0]
    ax_bm0.imshow(recon_amp_n, origin="lower", cmap="inferno", vmin=0, vmax=1)
    ax_bm0.plot(bridge_masks["cl_cols"], bridge_masks["cl_rows"],
                "c.-", markersize=4, linewidth=1, label="A→B centreline")
    ax_bm0.plot(a_px[1], a_px[0], "g^", markersize=9, label=f"A {a_px}")
    ax_bm0.plot(b_px[1], b_px[0], "rv", markersize=9, label=f"B {b_px}")
    ax_bm0.legend(fontsize=7, loc="upper right")
    ax_bm0.set_title(
        f"Recon + centreline\ncontinuity={bridge_morph['corridor_continuity']:.3f}"
        f"  elev_frac={bridge_morph['centreline_elevated_frac']:.3f}"
    )
    # Right: region masks overlaid on grayscale amplitude
    ax_bm1 = axes_bm[1]
    ax_bm1.imshow(recon_amp_n, origin="lower", cmap="gray", vmin=0, vmax=1, alpha=0.75)
    overlay = np.zeros((*recon_amp_n.shape, 4), dtype=float)
    overlay[bridge_masks["upper_mask"]] = [0.2, 0.4, 1.0, 0.45]  # blue  – A-side
    overlay[bridge_masks["lower_mask"]] = [1.0, 0.3, 0.3, 0.45]  # red   – B-side
    overlay[bridge_masks["side_mask"]]  = [1.0, 1.0, 0.0, 0.45]  # yellow – side-bands
    ax_bm1.imshow(overlay, origin="lower")
    ax_bm1.set_title(
        f"Masks: blue=upper(A), red=lower(B), yellow=side-band\n"
        f"lower_bal={bridge_morph['lower_balance_score']:.3f}  "
        f"aniso={bridge_morph['anisotropy_score']:.3f}  "
        f"leak={bridge_morph['side_leakage']:.3f}  "
        f"compact={bridge_morph['compactness_top5pct']:.3f}"
    )
    fig_bm.suptitle(f"Bridge morphology  bridge_score={bridge_morph['bridge_score']:.4f}")
    fig_bm.tight_layout(rect=[0, 0, 1, 0.93])
    fig_bm.savefig(out_dir / "bridge_morphology_masks.png", dpi=190, bbox_inches="tight")
    plt.close(fig_bm)

    # Gor'kov potential map
    _save_map(
        transport_metrics["U"],
        "Gor'kov potential",
        "U [J]",
        "RdBu_r",
        out_dir / "gorkov_potential.png",
    )

    # Force quiver on top of potential
    fig_f, ax_f = plt.subplots(figsize=(6.4, 5.4))
    Uplot = transport_metrics["U"]
    imf = ax_f.imshow(Uplot, origin="lower", cmap="RdBu_r")
    plt.colorbar(imf, ax=ax_f, label="U [J]")

    step = max(cfg.n_grid // 25, 1)
    rr = np.arange(0, cfg.n_grid, step)
    cc = np.arange(0, cfg.n_grid, step)
    CC, RR = np.meshgrid(cc, rr)
    Fx_s = transport_metrics["Fx"][RR, CC]
    Fy_s = transport_metrics["Fy"][RR, CC]
    ax_f.quiver(CC, RR, Fx_s, Fy_s, color="k", alpha=0.6, scale=None)

    ax_f.plot(a_px[1], a_px[0], "g^", markersize=9, label="A")
    ax_f.plot(b_px[1], b_px[0], "rv", markersize=9, label="B")
    for i, (nr, nc) in enumerate(neighbour_px_list):
        ax_f.plot(nc, nr, "wo", markersize=5)
    ax_f.legend(fontsize=8)
    ax_f.set_title(
        f"Transport field\nscore={transport_metrics['transport_score']:.3f}  "
        f"align={transport_metrics['force_alignment']:.3f}"
    )
    fig_f.tight_layout()
    fig_f.savefig(out_dir / "gorkov_force_field.png", dpi=190, bbox_inches="tight")
    plt.close(fig_f)

    _log("[stage] figures saved")

    # ── STL export ─────────────────────────────────────────────────────
    stl_status = "skipped"
    stl_error = None
    if not args.skip_stl:
        try:
            _log("[stage] STL export starting...")
            _, _, open_edges = export_stl(
                cfg=cfg,
                lens_field=lens_field,
                aperture_mask=aperture_mask,
                xg=xg,
                yg=yg,
                save_path=out_dir / "bridge_inverse_hologram_lens.stl",
                timings=timings,
                grid_stride=max(int(args.stl_grid_stride), 1),
            )
            stl_status = f"ok (open_edges={open_edges})"
            _log(f"[stage] STL export done: {stl_status}")
        except Exception as exc:
            stl_status = "failed"
            stl_error = str(exc)
            _log(f"[stage] STL export failed: {stl_error}")
    else:
        _log("[stage] STL export skipped (--skip-stl)")

    # ── Save NPZ ──────────────────────────────────────────────────────
    np.savez_compressed(
        out_dir / "bridge_inverse_replica_fields.npz",
        target_raw_amp=target_raw_amp,
        target_amp=target_amp,
        roi_mask=roi_mask,
        lens_field=lens_field,
        lens_phase_wrapped=phi_wrapped,
        thickness=thickness,
        recon_amp=recon_amp,
        recon_amp_n=recon_amp_n,
        aperture_mask=aperture_mask,
    )

    # ── Manifest ──────────────────────────────────────────────────────
    timings["total"] = time.perf_counter() - t0_all
    manifest = {
        "script": "scripts/dev/run_inverse_replica_on_bridge_pressure_field.py",
        "input_npz": str(args.input_npz),
        "field_key": args.field_key,
        "method": ("ROI-constrained IASA (partial-support Gerchberg-Saxton)"
                   if args.method == "iasa" else
                   "raw complex backprop + phase-only projection"),
        "enforcement": ("amplitude-only inside ROI; soft-suppressed outside ROI"
                        if args.method == "iasa" else
                        "complex ASM backprop to lens plane; aperture-masked phase-only projection"),
        "config": {
            "n_grid": cfg.n_grid,
            "transducer_diameter_mm": cfg.transducer_diameter_mm,
            "focal_distance_mm": cfg.focal_distance_mm,
            "frequency_hz": cfg.frequency_hz,
            "n_iter": cfg.n_iter,
            "h_base_mm": cfg.h_base_mm,
            "outside_suppression": best_suppression if best_suppression is not None else "N/A",
            "clip_percentile": float(args.clip_percentile),
            "gamma": float(args.gamma),
            "roi_pad_px": args.roi_pad_px,
            "stl_grid_stride": max(int(args.stl_grid_stride), 1),
            "target_field_extent_mm": [field_span_x, field_span_y],
            "target_centred_at_mm": [float(x_center * 1e3), float(y_center * 1e3)],
            "selection_mode": args.selection_mode,
            "particle_radius_mm": float(args.particle_radius_mm),
            "particle_density": float(args.particle_density),
            "particle_sound_speed": float(args.particle_sound_speed),
            "water_density": float(args.water_density),
            "neighbour_count": int(args.neighbour_count),
            "ab_bias_strength": float(args.ab_bias_strength),
        },
        "metrics": {
            "roi_correlation_vs_processed": corr_roi,
            "roi_correlation_vs_raw": corr_roi_raw,
            "full_aperture_correlation_vs_processed": corr_full,
            "full_aperture_correlation_vs_raw": corr_full_raw,
            "roi_correlation": corr_roi,
            "full_aperture_correlation": corr_full,
            "roi_rmse_normalised": rmse_roi,
            "roi_energy_ratio": energy_ratio,
            "roi_fraction_of_aperture": roi_frac,
            "target_amp_nonzero_frac": float(np.mean(target_amp[aperture_mask] > 0.0)),
            "thickness_min_mm": float(np.min(thickness[aperture_mask]) * 1e3),
            "thickness_max_mm": float(np.max(thickness[aperture_mask]) * 1e3),
            "h_max_mm": float(cfg.h_max_m * 1e3),
        },
        "bridge_morphology": {
            k: (float(v) if isinstance(v, (float, int, np.floating, np.integer)) else v)
            for k, v in bridge_morph.items()
        },
        "transport_metrics": {
            k: (float(v) if isinstance(v, (float, int, np.floating, np.integer)) else v)
            for k, v in transport_metrics.items()
            if not isinstance(v, np.ndarray)
        },
        "stl": {
            "status": stl_status,
            "error": stl_error,
        },
        "timings_s": timings,
        "outputs": {
            "target_raw_bridge_amplitude_png": "target_raw_bridge_amplitude.png",
            "target_amplitude_for_iasa_png": "target_amplitude_for_iasa.png",
            "lens_phase_wrapped_png": "lens_phase_wrapped.png",
            "reconstructed_pressure_at_focus_png": "reconstructed_pressure_at_focus.png",
            "lens_thickness_map_png": "lens_thickness_map.png",
            "target_vs_reconstruction_png": "target_vs_reconstruction.png",
            "bridge_morphology_masks_png": "bridge_morphology_masks.png",
            "gorkov_potential_png": "gorkov_potential.png",
            "gorkov_force_field_png": "gorkov_force_field.png",
            "fields_npz": "bridge_inverse_replica_fields.npz",
            "stl": "bridge_inverse_hologram_lens.stl" if stl_status.startswith("ok") else None,
        },
    }
    if sweep_results:
        manifest["parameter_sweep"] = sweep_results
        manifest["outputs"]["parameter_sweep_png"] = "parameter_sweep.png"

    with open(out_dir / "bridge_inverse_replica_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    _log("[done] manifest written")
    print(f"Output dir: {out_dir}")
    print(f"Method: {args.method}")
    print(f"ROI correlation (vs raw):       {corr_roi_raw:.4f}")
    print(f"ROI correlation (vs processed): {corr_roi:.4f}")
    print(f"Full aperture correlation:      {corr_full:.4f}")
    print(f"Energy ratio (ROI/outside): {energy_ratio:.3f}")
    print(f"Focal distance: {cfg.focal_distance_mm:.2f} mm")
    print(f"Outside suppression: {sup_str}")
    print(f"Thickness range [mm]: {np.min(thickness[aperture_mask])*1e3:.4f} .. {np.max(thickness[aperture_mask])*1e3:.4f}")
    print(f"STL: {stl_status}")
    print("")
    print(f"Bridge score: {bridge_morph['bridge_score']:.4f}")
    w = bridge_morph['weights']
    print(f"  +{w['w_continuity']:.1f} * corridor_continuity   = {bridge_morph['corridor_continuity']:.4f}  "
          f"(centreline min/max ratio; frac_elevated={bridge_morph['centreline_elevated_frac']:.3f})")
    print(f"  +{w['w_lower_balance']:.1f} * lower_balance_score = {bridge_morph['lower_balance_score']:.4f}  "
          f"(B-side / A-side mean energy; raw_ratio={bridge_morph['lower_upper_ratio']:.3f})")
    print(f"  +{w['w_anisotropy']:.1f} * anisotropy_score      = {bridge_morph['anisotropy_score']:.4f}  "
          f"(1-1/eigval_ratio; eigratio={bridge_morph['axis_eigenvalue_ratio']:.2f})")
    print(f"  -{w['w_side_leakage']:.1f} * side_leakage         = {bridge_morph['side_leakage']:.4f}  "
          f"(side-band / ROI mean energy)")
    print(f"  -{w['w_compactness']:.1f} * compactness_top5%     = {bridge_morph['compactness_top5pct']:.4f}  "
          f"(top-5% pixel energy / total ROI energy)")
    print("")
    print(f"Transport score: {transport_metrics['transport_score']:.4f}")
    print(f"  force_alignment(A->B): {transport_metrics['force_alignment']:.4f}")
    print(f"  deltaU_norm (A above B good): {transport_metrics['deltaU_norm']:.4f}")
    print(f"  barrier_penalty: {transport_metrics['barrier_penalty']:.4f}")
    print(f"  neighbour_penalty: {transport_metrics['neighbour_penalty']:.4f}")


def main():

    args = parse_args()

    if args.batch:
        run_batch(args)
        return

    main_single(args)


if __name__ == "__main__":
    main()
