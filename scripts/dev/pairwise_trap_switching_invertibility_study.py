#!/usr/bin/env python3
"""
Linked pairwise trap-switching + lens-invertibility study.

Purpose
-------
Search for compact perturbation fields that, when superposed with the standing
wave lattice, can selectively:
  1) release particle A,
  2) bias motion toward trap B,
  3) preserve B as the destination,
  4) minimise neighbour disturbance,
while also surviving phase-only lens inversion under relaxed hardware choices.

This script is intentionally a FRAMEWORK / study harness.
It does not modify the canonical pipeline. Instead, it reuses functions from
existing scripts where possible and keeps all new logic isolated.

Expected workflow
-----------------
1. Load standing-wave / ROI / trap data.
2. Build candidate perturbation targets directly in the ROI.
3. Superpose candidate with the standing wave and score the IDEAL behaviour.
4. Invert the candidate through the existing phase-only lens pipeline.
5. Reconstruct the propagated lens field and re-superpose with the standing wave.
6. Score the RECONSTRUCTED behaviour.
7. Rank candidates by combined ideal performance + reconstructed performance
   - invertibility loss.

Notes
-----
- The perturbation-only and target-family code here is deliberately compact and
  interpretable.
- The lens inversion stage is designed to call existing repo functions rather
  than re-implement the whole inversion stack.
- You should wire this into the repo by replacing the marked TODO blocks with
  exact imports / paths / helper calls that exist in your environment.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import map_coordinates as _map_coord

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import sys
sys.path.insert(0, str(PROJECT_ROOT))

# Existing lens / propagation primitives
from scripts.dev.inverse_c_shape_lens_replica import (
    ReplicaConfig,
    make_grid,
    propagate_asm,
    run_iasa_roi,
)

# Transport-metric function from the bridge inverse script.
from scripts.dev.run_inverse_replica_on_bridge_pressure_field import (
    _compute_transport_metrics,
)


# -----------------------------------------------------------------------------
# Defaults / paths
# -----------------------------------------------------------------------------
VORTEX_NPZ = (
    PROJECT_ROOT / "results" / "deliverables" / "vortex_stage_transport"
    / "transport" / "transport_case_for_gif.npz"
)
OVERLAY_NPZ = (
    PROJECT_ROOT / "results" / "c_shape_lens_15mm_overlay_study_20260310_170620"
    / "npz" / "roi_fields.npz"
)


# -----------------------------------------------------------------------------
# Study configuration containers
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class HardwareConfig:
    frequency_hz: float
    transducer_diameter_mm: float
    focal_distance_mm: float
    n_grid: int = 400
    c_water: float = 1480.0
    c_lens: float = 2636.0
    h_base_mm: float = 1.0
    n_iter: int = 60
    source_pressure_pa: float = 0.05e6
    outside_suppression: float = 0.85
    # Particle / medium parameters (used by Gorkov transport metric)
    rho_water: float = 998.0
    particle_radius_m: float = 50e-6
    rho_particle: float = 1050.0
    c_particle: float = 2350.0
    # Standing-wave frequency for Gorkov evaluation (must match the SW field,
    # NOT the lens hardware frequency which is a separate sweep variable).
    sw_frequency_hz: float = 2.0e6


@dataclass(frozen=True)
class CandidateConfig:
    family: str
    alpha: float
    psi: float
    a_release_pa: float
    b_pocket_pa: float
    lane_pa: float
    lane_width_m: float
    lane_sigma_ratio: float
    a_sigma_m: float
    b_sigma_m: float
    quiet_radius_m: float
    quiet_edge_m: float
    # Barrier-cancellation parameters (used only by 'barrier_cancel' family)
    barrier_cancel_xy: tuple[float, float] = (0.0, 0.0)
    barrier_cancel_amp: float = 0.0
    barrier_cancel_sigma_m: float = 5.0e-5


@dataclass
class CandidateResult:
    candidate: dict[str, Any]
    hardware: dict[str, Any]
    ideal_metrics: dict[str, Any]
    recon_metrics: dict[str, Any]
    invertibility_loss: dict[str, Any]
    scores: dict[str, float]
    outputs: dict[str, str]


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pairwise trap-switching + invertibility study")
    p.add_argument("--vortex-npz", type=Path, default=VORTEX_NPZ)
    p.add_argument("--overlay-npz", type=Path, default=OVERLAY_NPZ)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--max-candidates", type=int, default=0,
                   help="Optional cap for debugging. 0 = all combinations.")
    p.add_argument("--top-k", type=int, default=12)
    p.add_argument("--save-top-figures", type=int, default=6)
    p.add_argument("--coarse-only", action="store_true")
    p.add_argument(
        "--centreline-only", action="store_true",
        help="Run only the centreline_trough family at focused psi=π/2, "
             "alpha=[0.10,0.20] parameters. Implies --coarse-only.",
    )
    p.add_argument(
        "--barrier-cancel-only", action="store_true",
        help="Run only the barrier_cancel family: psi tuned to the SW phase at "
             "the barrier peak, alpha=[0.10,0.20]. Implies --coarse-only.",
    )
    p.add_argument(
        "--barrier-cancel-refine", action="store_true",
        help="Tight local refinement sweep around the best barrier_cancel result: "
             "alpha=0.10 only, psi centred at psi_cancel-0.2 rad (empirical winner), "
             "bc_amp/sigma/b_pocket/a_sigma swept narrowly. Implies --coarse-only.",
    )
    return p.parse_args()


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def _nearest_index(axis: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(axis - value)))


def _gaussian_2d(xx: np.ndarray, yy: np.ndarray, x0: float, y0: float, sigma_m: float, amp: float) -> np.ndarray:
    s2 = max(float(sigma_m), 1e-12) ** 2
    rr2 = (xx - x0) ** 2 + (yy - y0) ** 2
    return float(amp) * np.exp(-0.5 * rr2 / s2)


def _ab_frame(a_xy: np.ndarray, b_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    d = np.asarray(b_xy, float) - np.asarray(a_xy, float)
    L = float(np.linalg.norm(d))
    if L <= 0.0:
        raise RuntimeError("A and B are identical")
    e_s = d / L
    e_n = np.array([-e_s[1], e_s[0]], dtype=float)
    return e_s, e_n, L


def _smooth_box_window(coord: np.ndarray, lo: float, hi: float, edge: float) -> np.ndarray:
    edge_safe = max(float(edge), 1.0e-12)
    return 0.5 * (
        np.tanh((coord - float(lo)) / edge_safe)
        - np.tanh((coord - float(hi)) / edge_safe)
    )


def _lane_mask(xx: np.ndarray, yy: np.ndarray, a_xy: np.ndarray, b_xy: np.ndarray, width_m: float, edge_m: float = 3.5e-5) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    e_s, e_n, L = _ab_frame(a_xy, b_xy)
    rx = xx - float(a_xy[0])
    ry = yy - float(a_xy[1])
    s_m = rx * e_s[0] + ry * e_s[1]
    n_m = rx * e_n[0] + ry * e_n[1]
    ws = _smooth_box_window(s_m, 0.0, L, edge_m)
    wn = _smooth_box_window(np.abs(n_m), 0.0, 0.5 * float(width_m), edge_m)
    return np.clip(ws * wn, 0.0, 1.0), s_m, n_m, L


def _save_map(data: np.ndarray, title: str, cbar: str, cmap: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(data, origin="lower", cmap=cmap, aspect="equal")
    plt.colorbar(im, ax=ax, label=cbar)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Data loading and resampling
# -----------------------------------------------------------------------------
def load_base_fields(vortex_npz: Path, overlay_npz: Path) -> dict[str, Any]:
    vd = np.load(vortex_npz)
    ov = np.load(overlay_npz)

    x_full = vd["xg"].astype(float)
    y_full = vd["yg"].astype(float)
    p_sw_full = vd["p_sw"].astype(complex)
    sw_peak_pa = float(np.abs(p_sw_full).max())

    traps_m = ov["traps_m"].astype(float)
    idx_a = int(ov["idx_A"])
    idx_b = int(ov["idx_B"])

    print(f"  SW peak |p| = {sw_peak_pa:.2f} Pa  (perturbation alphas will be fractions of this)")
    return {
        "x_full": x_full,
        "y_full": y_full,
        "p_sw_full": p_sw_full,
        "sw_peak_pa": sw_peak_pa,
        "traps_m": traps_m,
        "idx_a": idx_a,
        "idx_b": idx_b,
    }


def resample_full_field_to_lens_grid(
    p_field_full: np.ndarray,
    x_full: np.ndarray,
    y_full: np.ndarray,
    xg: np.ndarray,
    yg: np.ndarray,
) -> np.ndarray:
    interp_re = RegularGridInterpolator((y_full, x_full), p_field_full.real, bounds_error=False, fill_value=0.0)
    interp_im = RegularGridInterpolator((y_full, x_full), p_field_full.imag, bounds_error=False, fill_value=0.0)
    pts = np.column_stack([yg.ravel(), xg.ravel()])
    return (interp_re(pts) + 1j * interp_im(pts)).reshape(xg.shape)


def _find_barrier_peak(
    p_sw: np.ndarray,
    x_full: np.ndarray,
    y_full: np.ndarray,
    a_xy: np.ndarray,
    b_xy: np.ndarray,
    n_samples: int = 200,
) -> tuple[tuple[float, float], float]:
    """Find the Gorkov-potential maximum along the A→B centreline.

    Returns ((x_bar, y_bar), psi_cancel) where psi_cancel is the perturbation
    phase that destructively interferes with p_sw at the barrier peak:
        psi_cancel = angle(p_sw(barrier)) + pi

    Uses the same particle / medium parameters as HardwareConfig defaults.
    """
    omega = 2.0 * np.pi * 2.0e6
    rho_w, c_w = 998.0, 1480.0
    rho_p, c_p, r_p = 1050.0, 2350.0, 50e-6
    Vp = (4.0 / 3.0) * np.pi * r_p ** 3
    f1 = 1.0 - (1.0 / (rho_p * c_p ** 2)) / (1.0 / (rho_w * c_w ** 2))
    f2 = 2.0 * (rho_p - rho_w) / (2.0 * rho_p + rho_w)
    dx = float(x_full[1] - x_full[0])
    dp_dy, dp_dx = np.gradient(p_sw, dx, dx)
    vx = -(1.0 / (1j * omega * rho_w)) * dp_dx
    vy = -(1.0 / (1j * omega * rho_w)) * dp_dy
    U = Vp * (
        f1 * np.abs(p_sw) ** 2 / (4.0 * rho_w * c_w ** 2)
        - 3.0 * f2 * rho_w * (np.abs(vx) ** 2 + np.abs(vy) ** 2) / 8.0
    )
    # Sample U and p_sw along the A→B centreline
    t = np.linspace(0.0, 1.0, n_samples)
    x_line = a_xy[0] + t * (b_xy[0] - a_xy[0])
    y_line = a_xy[1] + t * (b_xy[1] - a_xy[1])
    x0 = float(x_full[0])
    y0 = float(y_full[0])
    row_line = (y_line - y0) / dx
    col_line = (x_line - x0) / dx
    U_line    = _map_coord(U,       [row_line, col_line], order=1, mode="nearest")
    p_re_line = _map_coord(p_sw.real, [row_line, col_line], order=1, mode="nearest")
    p_im_line = _map_coord(p_sw.imag, [row_line, col_line], order=1, mode="nearest")
    idx_max = int(np.argmax(U_line))
    x_bar = float(x_line[idx_max])
    y_bar = float(y_line[idx_max])
    p_at_bar = complex(float(p_re_line[idx_max]), float(p_im_line[idx_max]))
    psi_cancel = float((np.angle(p_at_bar) + np.pi) % (2.0 * np.pi))
    print(
        f"  Barrier peak: ({x_bar*1e3:.3f}, {y_bar*1e3:.3f}) mm  "
        f"|p_sw|={abs(p_at_bar):.2f} Pa  "
        f"U={U_line[idx_max]:.3e} J  "
        f"psi_cancel={np.degrees(psi_cancel):.1f}°"
    )
    return (x_bar, y_bar), psi_cancel


# -----------------------------------------------------------------------------
# Candidate perturbation families
# -----------------------------------------------------------------------------
def build_candidate_roi_field(
    family: str,
    xx: np.ndarray,
    yy: np.ndarray,
    a_xy: np.ndarray,
    b_xy: np.ndarray,
    cfg: CandidateConfig,
    neighbour_positions: np.ndarray | None = None,
) -> np.ndarray:
    """
    Build an interpretable perturbation field directly in the image plane.
    Convention here: positive scalar template is used as pressure-amplitude-like
    target content. Keep this simple for the study harness.
    """
    out = np.zeros_like(xx, dtype=float)

    # A-release component (compact local perturbation around A)
    out += _gaussian_2d(xx, yy, a_xy[0], a_xy[1], cfg.a_sigma_m, cfg.a_release_pa)

    # B-capture / preserving pocket
    out += _gaussian_2d(xx, yy, b_xy[0], b_xy[1], cfg.b_sigma_m, cfg.b_pocket_pa)

    lane_mask, s_m, n_m, L = _lane_mask(xx, yy, a_xy, b_xy, cfg.lane_width_m)

    if family == "a_release_b_pocket":
        pass

    elif family == "a_release_lane_b_pocket":
        s_hat = np.clip(s_m / max(L, 1e-12), 0.0, 1.0)
        longitudinal = (1.0 - s_hat)
        transverse = np.exp(-0.5 * (n_m / max(cfg.lane_sigma_ratio * cfg.lane_width_m, 1e-12)) ** 2)
        out += cfg.lane_pa * lane_mask * longitudinal * transverse

    elif family == "pair_switch_dipole":
        # Slightly more compact / antisymmetric pair field.
        out += cfg.lane_pa * lane_mask * np.tanh((0.5 * L - s_m) / max(0.15 * L, 1e-12))

    elif family == "centreline_trough":
        # Goal: weaken trap A, suppress the Gorkov barrier along A→B, keep B
        # as destination, and add zero new amplitude on the direct centreline.
        #
        # A-blob and B-blob are already added above.  This branch adds narrow
        # lateral guide walls at ±(lane_width_m / 2.5) from the centreline,
        # peaking at the A-B midpoint (parabolic longitudinal envelope).
        # The direct s-axis is left empty so the template never amplifies
        # whatever standing-wave barrier sits there.
        if cfg.lane_pa > 0.0:
            guide_offset = max(cfg.lane_width_m / 2.5, 3.5e-5)
            guide_sigma  = max(cfg.lane_width_m * 0.15, 2.0e-5)
            # Two Gaussian ridges symmetrically flanking the centreline
            lateral_profile = (
                np.exp(-0.5 * ((n_m - guide_offset) / guide_sigma) ** 2)
                + np.exp(-0.5 * ((n_m + guide_offset) / guide_sigma) ** 2)
            )
            # Parabolic longitudinal taper: zero at A (s̃=0) and B (s̃=1),
            # maximum at midpoint — avoids swamping the A/B blobs.
            s_hat = np.clip(s_m / max(L, 1e-12), 0.0, 1.0)
            longit = np.clip(4.0 * s_hat * (1.0 - s_hat), 0.0, 1.0)
            out += cfg.lane_pa * lane_mask * lateral_profile * longit

    elif family == "barrier_cancel":
        # Compact Gaussian placed at the pre-computed Gorkov barrier peak.
        # The perturbation phase (psi) is set to destructively interfere with
        # the standing-wave pressure at that point: psi ≈ angle(p_sw(x_bar)) + π.
        # The A→B centreline is otherwise left empty (no lane fill).
        if cfg.barrier_cancel_amp > 0.0:
            bx, by = cfg.barrier_cancel_xy
            out += _gaussian_2d(xx, yy, bx, by,
                                cfg.barrier_cancel_sigma_m, cfg.barrier_cancel_amp)

    else:
        raise ValueError(f"Unknown family: {family}")

    # Quieting near neighbours to favour selectivity.
    if neighbour_positions is not None and len(neighbour_positions) > 0:
        for p in np.asarray(neighbour_positions, float):
            rr = np.hypot(xx - p[0], yy - p[1])
            quiet = 0.5 * (1.0 + np.tanh((rr - cfg.quiet_radius_m) / max(cfg.quiet_edge_m, 1e-12)))
            out *= quiet

    out = np.clip(out, 0.0, None)
    peak = float(np.max(out))
    return out / peak if peak > 0.0 else out


# -----------------------------------------------------------------------------
# Transport scoring
# -----------------------------------------------------------------------------
def compute_transport_metrics_wrapper(
    p_total: np.ndarray,
    dx: float,
    hardware: HardwareConfig,
    a_px: tuple[int, int],
    b_px: tuple[int, int],
    neighbour_px_list: list[tuple[int, int]],
) -> dict[str, Any]:
    """Wrapper around _compute_transport_metrics that also corrects the
    normalised sub-metrics for SW-field scale.

    The canonical function uses eps=1e-18 (J) and eps=1e-12 (N) to guard
    against division by zero. These overflow SW-field Gorkov quantities
    (~1e-20 J, ~1e-16 N) and render deltaU_norm, barrier_penalty,
    force_alignment, and neighbour_penalty all near-zero. The lens / bridge
    study is fine because its lens pressures are ~50 kPa → energies ~1e-14 J.
    Here we recompute the four normalised sub-metrics with eps=1e-40 so they
    reflect the true physical contrast.
    """
    m = _compute_transport_metrics(
        p_field=p_total,
        dx=dx,
        frequency_hz=hardware.sw_frequency_hz,   # SW frequency, NOT lens frequency
        c_water=hardware.c_water,
        rho_w=hardware.rho_water,
        particle_radius_m=hardware.particle_radius_m,
        rho_p=hardware.rho_particle,
        c_p=hardware.c_particle,
        a_px=a_px,
        b_px=b_px,
        neighbour_px_list=neighbour_px_list,
    )

    # ---- recompute normalised metrics with physics-appropriate epsilon ----
    def _s(field: np.ndarray, r: int, c: int) -> float:
        return float(_map_coord(field, [[r], [c]], order=1, mode="nearest")[0])

    U  = m["U"]
    Fx = m["Fx"]
    Fy = m["Fy"]
    a_r, a_c = a_px
    b_r, b_c = b_px

    UA  = _s(U, a_r, a_c)
    UB  = _s(U, b_r, b_c)
    FxA = _s(Fx, a_r, a_c)
    FyA = _s(Fy, a_r, a_c)
    Fmag_A = float(np.hypot(FxA, FyA))

    # Unit vector A→B (row/col convention)
    dr = float(b_r - a_r)
    dc = float(b_c - a_c)
    norm_ab = max(float(np.hypot(dr, dc)), 1e-12)
    er = dr / norm_ab
    ec = dc / norm_ab

    # force_alignment: projection of F(A) onto A→B, normalised by |F(A)|
    Fproj_A = FyA * er + FxA * ec
    force_alignment = Fproj_A / (Fmag_A + 1e-40)

    # deltaU_norm: normalised by actual energy scale
    deltaU = UA - UB
    eps_U = max(abs(UA) + abs(UB), 1e-40)
    deltaU_norm = deltaU / eps_U

    # barrier_penalty: barrier height relative to |UA - UB|
    t = np.linspace(0.0, 1.0, 50)
    cl_r = a_r + t * (b_r - a_r)
    cl_c = a_c + t * (b_c - a_c)
    U_line = _map_coord(U, [cl_r, cl_c], order=1, mode="nearest")
    U_barrier = float(np.max(U_line) - max(UA, UB))
    U_scale = max(abs(UA - UB), 1e-40)
    barrier_penalty = max(U_barrier, 0.0) / U_scale

    # neighbour_penalty: mean neighbour force relative to |F(A)|
    neigh_forces = [
        float(np.hypot(_s(Fx, nr, nc), _s(Fy, nr, nc)))
        for nr, nc in neighbour_px_list
    ]
    mean_neigh_force = float(np.mean(neigh_forces)) if neigh_forces else 0.0
    neighbour_penalty = mean_neigh_force / (Fmag_A + 1e-40)

    transport_score = (
        1.5 * force_alignment
        + 1.2 * deltaU_norm
        - 0.8 * barrier_penalty
        - 0.8 * neighbour_penalty
    )

    m.update({
        "UA": UA,
        "UB": UB,
        "force_alignment": float(force_alignment),
        "deltaU_norm": float(deltaU_norm),
        "barrier_penalty": float(barrier_penalty),
        "neighbour_penalty": float(neighbour_penalty),
        "transport_score": float(transport_score),
    })
    return m


def _strip_nonserializable(metrics: dict[str, Any]) -> dict[str, Any]:
    """Remove numpy array entries from a metrics dict before JSON serialization."""
    return {k: v for k, v in metrics.items() if not isinstance(v, np.ndarray)}


def compute_pairwise_switch_score(metrics: dict[str, Any]) -> dict[str, float]:
    """
    Study-specific scalar score.
    High is good.

    This is stricter than generic transport_score because the true goal is:
    release A, preserve/capture at B, minimise neighbour disturbance.
    """
    force_align = float(metrics.get("force_alignment", 0.0))
    deltaU_norm = float(metrics.get("deltaU_norm", 0.0))
    barrier = float(metrics.get("barrier_penalty", 0.0))
    neigh = float(metrics.get("neighbour_penalty", 0.0))

    # Soft heuristics. Tune after first runs.
    score = (
        1.6 * force_align
        + 1.3 * deltaU_norm
        - 0.9 * barrier
        - 1.1 * neigh
    )
    return {"pair_switch_score": float(score)}


def compute_invertibility_loss(ideal_metrics: dict[str, Any], recon_metrics: dict[str, Any]) -> dict[str, float]:
    def _abs_drop(key: str) -> float:
        return float(abs(float(ideal_metrics.get(key, 0.0)) - float(recon_metrics.get(key, 0.0))))

    loss = {
        "transport_score_drop": _abs_drop("transport_score"),
        "force_alignment_drop": _abs_drop("force_alignment"),
        "deltaU_norm_drop": _abs_drop("deltaU_norm"),
        "barrier_penalty_change": _abs_drop("barrier_penalty"),
        "neighbour_penalty_change": _abs_drop("neighbour_penalty"),
    }
    loss["invertibility_loss"] = float(
        1.0 * loss["transport_score_drop"]
        + 0.8 * loss["force_alignment_drop"]
        + 0.8 * loss["deltaU_norm_drop"]
        + 0.6 * loss["barrier_penalty_change"]
        + 0.8 * loss["neighbour_penalty_change"]
    )
    return loss


# -----------------------------------------------------------------------------
# Lens inversion + reconstruction
# -----------------------------------------------------------------------------
def invert_candidate_field(
    target_amp: np.ndarray,
    x_sw: np.ndarray,
    y_sw: np.ndarray,
    hardware: HardwareConfig,
    output_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Build the lens grid, run ROI-constrained IASA, reconstruct the propagated field.

    The target_amp is defined on the standing-wave grid (x_sw, y_sw).  It is
    resampled onto the centred lens grid before IASA, and the propagated output
    is resampled back to the standing-wave grid so it can be directly superposed
    with p_sw_full.

    Returns (lens_field, p_recon_on_sw, x_lens, y_lens, xg, yg,
             aperture_mask, roi_mask, target_on_lens, dx).
    """
    cfg = ReplicaConfig(
        frequency_hz=hardware.frequency_hz,
        c_water=hardware.c_water,
        c_lens=hardware.c_lens,
        transducer_diameter_mm=hardware.transducer_diameter_mm,
        focal_distance_mm=hardware.focal_distance_mm,
        n_grid=hardware.n_grid,
        h_base_mm=hardware.h_base_mm,
        n_iter=hardware.n_iter,
        source_pressure_pa=hardware.source_pressure_pa,
        output_dir=str(output_dir),
    )

    # make_grid returns (x_1d, y_1d, xg_2d, yg_2d, r, theta, aperture_mask, dx)
    # xg/yg are centred at 0, ranging ±dia/2.
    x_lens, y_lens, xg, yg, _, _, aperture_mask, dx = make_grid(cfg)

    # Coordinate origin of the standing-wave grid (physical, metres)
    x_center = 0.5 * (float(x_sw[0]) + float(x_sw[-1]))
    y_center = 0.5 * (float(y_sw[0]) + float(y_sw[-1]))

    # ── Resample target_amp (SW grid) → lens grid ─────────────────────────────
    # SW grid uses (row=y, col=x) indexing for RegularGridInterpolator.
    interp_target = RegularGridInterpolator(
        (y_sw, x_sw), target_amp.real, bounds_error=False, fill_value=0.0
    )
    # Each lens-grid point (xl, yl) corresponds to SW coordinate (xl+x_center, yl+y_center)
    pts_lens = np.column_stack([
        (yg + y_center).ravel(),
        (xg + x_center).ravel(),
    ])
    target_on_lens = np.clip(interp_target(pts_lens).reshape(xg.shape), 0.0, None)

    # ── ROI-constrained IASA on the lens grid ─────────────────────────────────
    roi_mask = (target_on_lens > 1e-6) & aperture_mask
    lens_field = run_iasa_roi(
        cfg,
        aperture_mask,
        target_on_lens * aperture_mask.astype(float),
        roi_mask,
        dx,
        outside_suppression=hardware.outside_suppression,
    )

    # ── ASM propagation to focal plane (lens grid coordinates) ────────────────
    p_recon_on_lens = propagate_asm(lens_field, cfg.k_water, cfg.focal_distance_m, dx)

    # ── Resample p_recon from lens grid → SW grid ─────────────────────────────
    # Lens coord (xl, yl) maps to SW coord (xl+x_center, yl+y_center)
    x_lens_shifted = x_lens + x_center
    y_lens_shifted = y_lens + y_center
    interp_re = RegularGridInterpolator(
        (y_lens_shifted, x_lens_shifted),
        p_recon_on_lens.real, bounds_error=False, fill_value=0.0
    )
    interp_im = RegularGridInterpolator(
        (y_lens_shifted, x_lens_shifted),
        p_recon_on_lens.imag, bounds_error=False, fill_value=0.0
    )
    xx_sw, yy_sw = np.meshgrid(x_sw, y_sw)
    pts_sw = np.column_stack([yy_sw.ravel(), xx_sw.ravel()])
    p_recon_on_sw = (
        interp_re(pts_sw) + 1j * interp_im(pts_sw)
    ).reshape(xx_sw.shape)

    return lens_field, p_recon_on_sw, x_lens, y_lens, xg, yg, aperture_mask, roi_mask, target_on_lens, dx


# -----------------------------------------------------------------------------
# Candidate generation
# -----------------------------------------------------------------------------
def generate_candidate_space(
    coarse_only: bool,
    centreline_only: bool = False,
    barrier_cancel_only: bool = False,
    barrier_cancel_refine: bool = False,
    barrier_xy: tuple[float, float] = (0.0, 0.0),
    psi_cancel: float = 0.0,
) -> tuple[list[CandidateConfig], list[HardwareConfig]]:
    # Alpha is a *fraction of SW peak pressure* — 0.05 means the perturbation
    # peak equals 5% of the standing-wave field peak.  Physically meaningful
    # range: 5% (barely visible) → 40% (strong perturbation, matching the
    # 10.7 Pa lens vs 46 Pa SW from the c-shape overlay study).
    if coarse_only:
        families = ["a_release_b_pocket", "a_release_lane_b_pocket"]
        alphas = [0.05, 0.10, 0.20, 0.40]   # fractions of SW peak
        psis = [0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi]
        a_release_pas = [40.0, 80.0]
        b_pocket_pas = [40.0, 80.0]
        lane_pas = [20.0, 60.0]
        widths = [2.0e-4, 3.2e-4]
        n_iters = 40
    else:
        families = ["a_release_b_pocket", "a_release_lane_b_pocket", "pair_switch_dipole"]
        alphas = [0.05, 0.10, 0.20, 0.40]   # fractions of SW peak
        psis = [0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi]
        a_release_pas = [30.0, 60.0, 100.0]
        b_pocket_pas = [30.0, 60.0, 100.0]
        lane_pas = [0.0, 20.0, 60.0]
        widths = [1.8e-4, 2.5e-4, 3.5e-4]
        n_iters = 60

    candidates: list[CandidateConfig] = []
    for fam in families:
        for alpha in alphas:
            for psi in psis:
                for a_pa in a_release_pas:
                    for b_pa in b_pocket_pas:
                        for lane_pa in lane_pas:
                            for w in widths:
                                if fam == "a_release_b_pocket" and lane_pa != lane_pas[0]:
                                    continue
                                candidates.append(CandidateConfig(
                                    family=fam,
                                    alpha=float(alpha),
                                    psi=float(psi),
                                    a_release_pa=float(a_pa),
                                    b_pocket_pa=float(b_pa),
                                    lane_pa=float(lane_pa),
                                    lane_width_m=float(w),
                                    lane_sigma_ratio=0.32,
                                    a_sigma_m=1.0e-4,
                                    b_sigma_m=1.0e-4,
                                    quiet_radius_m=1.4e-4,
                                    quiet_edge_m=3.5e-5,
                                ))

    # ── Centreline-trough targeted sweep ────────────────────────────────────
    # Focused parameter set: psi fixed at π/2, alpha=[0.10, 0.20].
    # Varies A-blob width, relative A/B amplitudes, and lateral guide strength.
    # When lane_pa=0 the family degenerates to pure A+B blobs with no lane fill
    # (a useful ablation baseline). Width is irrelevant when lane_pa=0 so only
    # one width is generated in that case to avoid identical duplicates.
    ct_n_iters = n_iters  # inherit coarse/full n_iter setting
    for alpha in [0.10, 0.20]:
        for a_sigma in [70e-6, 100e-6]:
            for a_pa in [40.0, 80.0]:
                for b_pa in [20.0, 40.0]:
                    for lane_pa in [0.0, 30.0, 60.0]:
                        guide_widths = [150e-6] if lane_pa == 0.0 else [150e-6, 250e-6]
                        for w in guide_widths:
                            candidates.append(CandidateConfig(
                                family="centreline_trough",
                                alpha=float(alpha),
                                psi=0.5 * np.pi,       # fixed at π/2
                                a_release_pa=float(a_pa),
                                b_pocket_pa=float(b_pa),
                                lane_pa=float(lane_pa),
                                lane_width_m=float(w),
                                lane_sigma_ratio=0.32,
                                a_sigma_m=float(a_sigma),
                                b_sigma_m=1.0e-4,
                                quiet_radius_m=1.4e-4,
                                quiet_edge_m=3.5e-5,
                            ))

    # ── Barrier-cancel targeted sweep ───────────────────────────────────
    # Only generated when barrier_cancel_only=True.  psi is centred on the
    # phase that destructively interferes with p_sw at the barrier peak;
    # ±0.2 rad offsets probe the sensitivity.  Sweeps A-blob width/amplitude,
    # barrier-blob amplitude, and barrier-blob width.  b_pa kept low (20)
    # so B is a light anchor and not a competing dominant blob.
    if barrier_cancel_only:
        for alpha in [0.10, 0.20]:
            for a_sigma in [70e-6, 100e-6]:
                for a_pa in [40.0, 80.0]:
                    for bc_amp in [40.0, 80.0, 120.0]:
                        for bc_sigma in [40e-6, 70e-6]:
                            for psi_off in [-0.2, 0.0, 0.2]:
                                candidates.append(CandidateConfig(
                                    family="barrier_cancel",
                                    alpha=float(alpha),
                                    psi=float((psi_cancel + psi_off) % (2.0 * np.pi)),
                                    a_release_pa=float(a_pa),
                                    b_pocket_pa=20.0,
                                    lane_pa=0.0,
                                    lane_width_m=150e-6,
                                    lane_sigma_ratio=0.32,
                                    a_sigma_m=float(a_sigma),
                                    b_sigma_m=1.0e-4,
                                    quiet_radius_m=1.4e-4,
                                    quiet_edge_m=3.5e-5,
                                    barrier_cancel_xy=barrier_xy,
                                    barrier_cancel_amp=float(bc_amp),
                                    barrier_cancel_sigma_m=float(bc_sigma),
                                ))

    # All hardware configs share sw_frequency_hz=2.0e6 (the field's actual
    # frequency) for Gorkov scoring.  frequency_hz sweeps only the LENS design.
    hardware = [
        HardwareConfig(frequency_hz=2.15e6, transducer_diameter_mm=20.0, focal_distance_mm=20.0, n_iter=n_iters, sw_frequency_hz=2.0e6),
        HardwareConfig(frequency_hz=2.15e6, transducer_diameter_mm=30.0, focal_distance_mm=20.0, n_iter=n_iters, sw_frequency_hz=2.0e6),
        HardwareConfig(frequency_hz=3.00e6, transducer_diameter_mm=30.0, focal_distance_mm=15.0, n_iter=n_iters, sw_frequency_hz=2.0e6),
    ]
    # ── Barrier-cancel refinement sweep ──────────────────────────────
    # Tight local search anchored on the empirical winner from barrier_cancel_v1:
    #   alpha=0.10, psi=psi_cancel-0.2 rad, bc_amp=120, bc_sigma=70µm, a_pa=40,
    #   a_sigma=70µm, b_pocket=20.
    # Tight: psi (3) × bc_amp (3) × bc_sigma (2) × b_pocket_pa (2) × a_sigma (1)
    # = 3×3×2×2×1 = 36 candidates × 2 hardware = 72 cases (~4 min).
    if barrier_cancel_refine:
        psi_centre = (psi_cancel - 0.2) % (2.0 * np.pi)  # empirical optimum
        for psi_off in [-0.1, 0.0, 0.1]:
            psi_val = (psi_centre + psi_off) % (2.0 * np.pi)
            for bc_amp in [100.0, 120.0, 150.0]:
                for bc_sigma in [55e-6, 70e-6]:
                    for b_pa in [20.0, 40.0]:
                        for a_sig in [70e-6]:
                            candidates.append(CandidateConfig(
                                family="barrier_cancel",
                                alpha=0.10,
                                psi=float(psi_val),
                                a_release_pa=40.0,
                                b_pocket_pa=float(b_pa),
                                lane_pa=0.0,
                                lane_width_m=150e-6,
                                lane_sigma_ratio=0.32,
                                a_sigma_m=float(a_sig),
                                b_sigma_m=1.0e-4,
                                quiet_radius_m=1.4e-4,
                                quiet_edge_m=3.5e-5,
                                barrier_cancel_xy=barrier_xy,
                                barrier_cancel_amp=float(bc_amp),
                                barrier_cancel_sigma_m=float(bc_sigma),
                            ))

    # All hardware kept for barrier_cancel_refine but limited to 2 configs to
    # save time (drop 3 GHz since all 3 gave identical scores in earlier runs).
    hardware_refine = [
        HardwareConfig(frequency_hz=2.15e6, transducer_diameter_mm=20.0, focal_distance_mm=20.0, n_iter=n_iters, sw_frequency_hz=2.0e6),
        HardwareConfig(frequency_hz=2.15e6, transducer_diameter_mm=30.0, focal_distance_mm=20.0, n_iter=n_iters, sw_frequency_hz=2.0e6),
    ]

    if centreline_only:
        candidates = [c for c in candidates if c.family == "centreline_trough"]
    elif barrier_cancel_only:
        candidates = [c for c in candidates if c.family == "barrier_cancel"]
    elif barrier_cancel_refine:
        candidates = [c for c in candidates if c.family == "barrier_cancel"]
        hardware = hardware_refine
    return candidates, hardware


# -----------------------------------------------------------------------------
# Main study loop
# -----------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or (PROJECT_ROOT / "results" / "dev" / f"pairwise_trap_switching_invertibility_study_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    base = load_base_fields(args.vortex_npz, args.overlay_npz)
    x_full = base["x_full"]
    y_full = base["y_full"]
    p_sw_full = base["p_sw_full"]
    sw_peak_pa = base["sw_peak_pa"]
    traps_m = base["traps_m"]
    idx_a = base["idx_a"]
    idx_b = base["idx_b"]
    a_xy = traps_m[idx_a][:2]
    b_xy = traps_m[idx_b][:2]
    neighbour_positions = traps_m[np.array(sorted(set(range(len(traps_m))) - {idx_a, idx_b}), dtype=int), :2]

    # Pre-compute barrier peak location and optimal cancellation phase from
    # the loaded SW field. Fast (~0.5 s); always run so all family types work.
    print("  Computing Gorkov barrier peak along A\u2192B centreline...")
    barrier_xy, psi_cancel = _find_barrier_peak(p_sw_full, x_full, y_full, a_xy, b_xy)

    candidates, hardware_list = generate_candidate_space(
        coarse_only=args.coarse_only or args.centreline_only or args.barrier_cancel_only or args.barrier_cancel_refine,
        centreline_only=args.centreline_only,
        barrier_cancel_only=args.barrier_cancel_only,
        barrier_cancel_refine=args.barrier_cancel_refine,
        barrier_xy=barrier_xy,
        psi_cancel=psi_cancel,
    )
    if args.max_candidates > 0:
        candidates = candidates[: args.max_candidates]

    results: list[CandidateResult] = []
    summary_rows: list[dict[str, Any]] = []
    xx_full, yy_full = np.meshgrid(x_full, y_full, indexing="xy")

    t0 = time.perf_counter()
    run_idx = 0
    total = len(candidates) * len(hardware_list)

    for cand in candidates:
        # Build candidate perturbation directly on the standing-wave grid.
        # build_candidate_roi_field returns a normalised [0,1] template.
        # Scale by alpha * sw_peak_pa so alpha is a physical fraction of
        # the standing-wave field peak (e.g. alpha=0.10 → 10% of SW peak).
        _pert_template = build_candidate_roi_field(
            family=cand.family,
            xx=xx_full,
            yy=yy_full,
            a_xy=a_xy,
            b_xy=b_xy,
            cfg=cand,
            neighbour_positions=neighbour_positions,
        ).astype(complex)
        p_pert_target = (cand.alpha * sw_peak_pa) * np.exp(1j * cand.psi) * _pert_template

        for hw in hardware_list:
            run_idx += 1
            case_name = f"case_{run_idx:05d}_{cand.family}_a{cand.alpha:.2f}_f{hw.frequency_hz/1e6:.2f}MHz_D{hw.transducer_diameter_mm:.0f}mm"
            case_dir = out_dir / case_name
            case_dir.mkdir(parents=True, exist_ok=True)
            print(f"[{run_idx}/{total}] {case_name}", flush=True)

            # Ideal superposed field on the standing-wave grid.
            p_total_ideal = p_sw_full + p_pert_target

            # Map A/B/neighbour positions to current grid pixels.
            ia = _nearest_index(x_full, float(a_xy[0]))
            ja = _nearest_index(y_full, float(a_xy[1]))
            ib = _nearest_index(x_full, float(b_xy[0]))
            jb = _nearest_index(y_full, float(b_xy[1]))
            a_px = (ja, ia)
            b_px = (jb, ib)
            neighbour_px_list = [(_nearest_index(y_full, float(p[1])), _nearest_index(x_full, float(p[0]))) for p in neighbour_positions]

            try:
                ideal_metrics = compute_transport_metrics_wrapper(
                    p_total_ideal,
                    dx=float(x_full[1] - x_full[0]),
                    hardware=hw,
                    a_px=a_px,
                    b_px=b_px,
                    neighbour_px_list=neighbour_px_list,
                )
                ideal_scores = compute_pairwise_switch_score(ideal_metrics)
            except Exception as exc:
                ideal_metrics = {"error": str(exc)}
                ideal_scores = {"pair_switch_score": -1e9}

            # Lens inversion stage — target is resampled SW→lens→SW inside invert_candidate_field.
            recon_metrics: dict[str, Any]
            invertibility_loss: dict[str, Any]
            total_scores: dict[str, float]
            outputs = {}

            try:
                lens_field, p_recon_pert, x_lens, y_lens, xg, yg, aperture_mask, roi_mask, target_on_lens, dx_lens = invert_candidate_field(
                    target_amp=np.abs(p_pert_target),
                    x_sw=x_full,
                    y_sw=y_full,
                    hardware=hw,
                    output_dir=case_dir,
                )
                # p_recon_pert is now on the SW grid — superpose directly.
                p_total_recon = p_sw_full + p_recon_pert
                recon_metrics = compute_transport_metrics_wrapper(
                    p_total_recon,
                    dx=float(x_full[1] - x_full[0]),
                    hardware=hw,
                    a_px=a_px,
                    b_px=b_px,
                    neighbour_px_list=neighbour_px_list,
                )
                recon_scores = compute_pairwise_switch_score(recon_metrics)
                invertibility_loss = compute_invertibility_loss(ideal_metrics, recon_metrics)

                combined = (
                    0.7 * float(ideal_scores["pair_switch_score"])
                    + 1.0 * float(recon_scores["pair_switch_score"])
                    - 0.9 * float(invertibility_loss["invertibility_loss"])
                )
                total_scores = {
                    "ideal_pair_switch_score": float(ideal_scores["pair_switch_score"]),
                    "recon_pair_switch_score": float(recon_scores["pair_switch_score"]),
                    "combined_score": float(combined),
                }

                _save_map(np.abs(p_pert_target), "Target perturbation amplitude", "a.u.", "inferno", case_dir / "target_perturbation.png")
                _save_map(np.abs(p_recon_pert), "Reconstructed perturbation amplitude", "a.u.", "inferno", case_dir / "reconstructed_perturbation.png")
                outputs = {
                    "target_perturbation_png": "target_perturbation.png",
                    "reconstructed_perturbation_png": "reconstructed_perturbation.png",
                }
            except Exception as exc:
                recon_metrics = {"error": str(exc)}
                invertibility_loss = {"invertibility_loss": 1e9, "error": str(exc)}
                total_scores = {
                    "ideal_pair_switch_score": float(ideal_scores["pair_switch_score"]),
                    "recon_pair_switch_score": -1e9,
                    "combined_score": -1e9,
                }

            record = CandidateResult(
                candidate=asdict(cand),
                hardware=asdict(hw),
                ideal_metrics=_strip_nonserializable(ideal_metrics),
                recon_metrics=_strip_nonserializable(recon_metrics),
                invertibility_loss=invertibility_loss,
                scores=total_scores,
                outputs=outputs,
            )
            results.append(record)
            summary_rows.append({
                "case": case_name,
                "family": cand.family,
                "freq_MHz": hw.frequency_hz / 1e6,
                "diameter_mm": hw.transducer_diameter_mm,
                "focal_mm": hw.focal_distance_mm,
                **total_scores,
            })

            with open(case_dir / "result.json", "w", encoding="utf-8") as f:
                json.dump(asdict(record), f, indent=2)

    # Sort and save summary.
    results_sorted = sorted(results, key=lambda r: r.scores.get("combined_score", -1e9), reverse=True)
    summary_sorted = sorted(summary_rows, key=lambda r: r.get("combined_score", -1e9), reverse=True)

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_sorted, f, indent=2)

    # Markdown leaderboard.
    md = [
        "# Pairwise Trap-Switching + Invertibility Study",
        "",
        f"Total cases: {len(summary_sorted)}",
        f"Runtime [s]: {time.perf_counter() - t0:.2f}",
        "",
        "| Rank | Case | Family | MHz | D [mm] | f [mm] | Ideal | Recon | Combined |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for i, row in enumerate(summary_sorted[: max(args.top_k, 1)], start=1):
        md.append(
            f"| {i} | {row['case']} | {row['family']} | {row['freq_MHz']:.2f} | {row['diameter_mm']:.1f} | {row['focal_mm']:.1f} | "
            f"{row['ideal_pair_switch_score']:.3f} | {row['recon_pair_switch_score']:.3f} | {row['combined_score']:.3f} |"
        )
    (out_dir / "README.md").write_text("\n".join(md), encoding="utf-8")

    print(f"Done. Results: {out_dir}")


if __name__ == "__main__":
    main()
