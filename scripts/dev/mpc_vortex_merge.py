#!/usr/bin/env python3
"""
MPC Vortex Merge — closed-loop controller for merging particle A into B's trap.

Uses an analytic LG vortex perturbation with 5 control DOFs per timestep:
    (ψ, x_v, y_v, α, β)
and a receding-horizon MPC with discrete adjoint gradients.

Outputs:
    results/dev/mpc_vortex_merge/<run_dir>/
        mpc_vortex_merge.gif      — animated Gor'kov potential + trajectories
        manifest.json             — full config + metrics
        controls.csv              — per-step control history
        trajectory.csv            — per-step particle positions
        convergence.png           — MPC inner-loop convergence

Usage:
    python scripts/dev/mpc_vortex_merge.py
    python scripts/dev/mpc_vortex_merge.py --K 15 --T 2000 --n_iters 25
    python scripts/dev/mpc_vortex_merge.py --gradcheck   # verify gradients only
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["figure.dpi"] = 72
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.interpolate import RegularGridInterpolator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.lib.fem_cache_utils import (
    C_WATER,
    F_HZ,
    OMEGA,
    RHO0,
    default_particle_params,
    gorkov_grid_2d,
)
from scripts.lib.mpc_controller import (
    I_ALPHA,
    I_BETA,
    I_PSI,
    I_XV,
    I_YV,
    N_CTRL,
    ForceEvaluator,
    MPCConfig,
    MPCResult,
    mpc_result_to_transport,
    run_mpc,
)
from scripts.lib.mpc_gorkov_basis import build_basis_for_vortex_position
from scripts.lib.particle_dynamics_utils import (
    CAPTURE_RADIUS,
    LAM,
    SCALE,
    TRAP_SP,
    TransportResult,
    compute_metrics,
    gorkov_normalised,
)
from scripts.lib.overlay_utils import estimate_ring_radius
from scripts.lib.asm_utils import make_vortex_field, propagate_asm, LAM as ASM_LAM, K0 as ASM_K0
from scripts.lib.perturbation_vortex import VortexPerturbation

PPAR = default_particle_params()

# ── Data paths ────────────────────────────────────────────────────

FIELD_NPZ = (
    PROJECT_ROOT / "results" / "deliverables" / "vortex_stage_transport"
    / "transport" / "transport_case_for_gif.npz"
)

# ── Rendering constants ──────────────────────────────────────────

CMAP = "RdBu_r"
COL_A = "#e74c3c"
COL_B = "#3498db"
COL_NEIGH = "#95a5a6"
COL_HOME = "#2ecc71"
COL_CTR = "#f39c12"
COL_TRAP = "#95a5a6"
COL_BARRIER = "#e67e22"
COL_HOME_A = "#e74c3c"
COL_HOME_B = "#3498db"
GORKOV_CLIP_LO = 0.5
GORKOV_CLIP_HI = 99.5
PARTICLE_RADIUS_MM = 0.045
VIEW_MARGIN_MM = 0.5
BETA_FIXED = 1.0             # beta used for GIF display (SW always visible)

# Settling-phase defaults
T_SETTLE_DEFAULT = 3000          # 3000 steps × 0.1ms = 300ms for α→0 ramp + SW relaxation


# ── Vortex design defaults (match vortex_entry_test) ────────────

VORTEX_CHARGE = 1
VORTEX_WAIST = 0.15e-3           # m  (LG fallback only)
VORTEX_APERTURE = 0.8e-3         # m  (LG fallback only)

# Focused Bessel vortex defaults (primary model, same as vortex_entry_test)
BESSEL_APERTURE_MM_DEFAULT = 3.5   # mm — aperture radius
BESSEL_PROP_DIST_MM_DEFAULT = 3.0  # mm — ASM propagation distance
FOCUS_MM_DEFAULT = 3.0             # mm — lens focal length

# ── Phase-sweep carry defaults (match vortex_entry_test) ─────────

N_PSI_DEFAULT = 24
N_VORTEX_STEPS_DEFAULT = 20
N_DYN_STEPS_DEFAULT = 150
N_BARRIER_SAMPLES = 50
W_BARRIER_DEFAULT = 1.0
W_PULL_DEFAULT = 1.0
W_LATERAL_DEFAULT = 0.5
W_RETAIN_DEFAULT = 1.0
MAX_STEP_DEFAULT = 2e-6
N_RING_THETA = 72
OPENING_SECTOR_DEG = 60.0
OPENING_THRESHOLD = 0.05
GEOM_MARGIN = 1.5              # d(B, vortex_start) > GEOM_MARGIN * r_barrier
N_INTRO_FRAMES = 15
FRAMES_PER_VSTEP = 6
N_NEIGH_RADIUS_DEFAULT = 3     # select neighbours within this many trap spacings of A-B midpoint

# ── Remote particle / open-loop defaults ─────────────────────────

MPC_ACTIVATION_RADIUS_LAM = 1.0

# ── GIF settings ─────────────────────────────────────────────────

N_GIF_FRAMES = 200
GIF_DURATION_MS = 80


# ════════════════════════════════════════════════════════════════
# Physics helpers
# ════════════════════════════════════════════════════════════════

def total_pressure(
    p_sw: np.ndarray,
    vortex_gen: VortexPerturbation,
    psi: float,
    alpha: float,
    beta: float,
    center_xy: np.ndarray,
) -> np.ndarray:
    """Compute p_total = β·p_sw + α·exp(iψ)·p_v(center)."""
    p_v = vortex_gen.get_field(center_xy)
    return beta * p_sw + alpha * np.exp(1j * psi) * p_v


def gorkov_from_pressure(p: np.ndarray, dx: float, dy: float):
    """Gor'kov potential and force from total pressure."""
    return gorkov_grid_2d(
        p, dx, dy, OMEGA, RHO0, C_WATER,
        PPAR["a"], PPAR["f1"], PPAR["f2"],
    )


# ════════════════════════════════════════════════════════════════
# Data loading, trap selection, geometry validation (same as vortex_entry_test)
# ════════════════════════════════════════════════════════════════

def load_data(
    aperture_m: float,
    prop_dist_m: float,
    focus_f_m: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float,
           np.ndarray, "VortexPerturbation", float]:
    """
    Load full-domain standing wave + trap lattice from FIELD_NPZ,
    build focused vortex, calibrate barrier radius.

    Returns (p_sw, xg, yg, dx, dy, traps_m, vortex_gen, r_barrier).
    """
    print("Loading full-domain standing-wave and trap lattice...")
    fd = np.load(FIELD_NPZ)
    xg      = fd["xg"].astype(float)
    yg      = fd["yg"].astype(float)
    p_sw    = fd["p_sw"].astype(complex)
    traps_m = fd["traps_m"].astype(float)
    p_sw_peak = float(np.max(np.abs(p_sw)))

    dx = float(xg[1] - xg[0])
    dy = float(yg[1] - yg[0])

    print(f"  Grid: {len(xg)}x{len(yg)}, "
          f"domain [{xg[0]*1e3:.2f}, {xg[-1]*1e3:.2f}] x "
          f"[{yg[0]*1e3:.2f}, {yg[-1]*1e3:.2f}] mm, "
          f"dx={dx*1e6:.1f} um")
    print(f"  p_sw max amplitude: {p_sw_peak:.4f}")
    print(f"  Trap lattice: {len(traps_m)} traps")

    grid_cx = 0.5 * (float(xg[0]) + float(xg[-1]))
    grid_cy = 0.5 * (float(yg[0]) + float(yg[-1]))
    XX, YY = np.meshgrid(xg, yg)

    print(f"  Generating focused vortex (R={aperture_m*1e3:.2f} mm, "
          f"f={focus_f_m*1e3:.2f} mm, z={prop_dist_m*1e3:.2f} mm)...")
    p_source = _make_focused_vortex_source(
        XX, YY,
        charge=VORTEX_CHARGE,
        aperture_radius=aperture_m,
        k=ASM_K0,
        focal_length=focus_f_m,
        center=(grid_cx, grid_cy),
        apodization="cosine_taper",
    )
    p_vortex_raw = propagate_asm(p_source, dx, dy, wavelength=ASM_LAM, z=prop_dist_m)

    p_vortex_peak = float(np.max(np.abs(p_vortex_raw)))
    if p_vortex_peak > 0:
        p_vortex_raw = p_vortex_raw * (p_sw_peak / p_vortex_peak)

    mag = np.abs(p_vortex_raw)
    center_xy = np.array([grid_cx, grid_cy])
    ring_info = estimate_ring_radius(
        mag, xg, yg, center=center_xy,
        r_min=0.05e-3, r_max=1.0e-3, n_bins=300,
    )
    r_barrier = float(ring_info["ring_radius_m"])
    ring_peak  = float(ring_info["ring_peak"])

    r_theory = 0.42 * ASM_LAM * focus_f_m / (2.0 * aperture_m)
    print(f"\n  --- Focused Vortex Calibration ---")
    print(f"  Aperture radius:      {aperture_m*1e3:.2f} mm")
    print(f"  Focal length:         {focus_f_m*1e3:.2f} mm")
    print(f"  Propagation distance: {prop_dist_m*1e3:.2f} mm  "
          f"({'at focus' if abs(prop_dist_m - focus_f_m) < 0.05e-3 else 'off-focus'})")
    print(f"  Pressure ring radius: {r_barrier*1e6:.1f} um ({r_barrier/LAM:.2f} lam)")
    print(f"  Ring peak amplitude:  {ring_peak:.4f}")
    print(f"  Theory (focal-plane): {r_theory*1e6:.1f} um  [0.42*lam*f/(2R)]")

    vortex_gen = VortexPerturbation(p_vortex_raw, xg, yg, out_xg=xg, out_yg=yg)
    return p_sw, xg, yg, dx, dy, traps_m, vortex_gen, r_barrier


def select_trap_pair(
    traps_m: np.ndarray,
    r_barrier: float,
    xg: np.ndarray,
    yg: np.ndarray,
    idx_B_override: Optional[int] = None,
    min_sep_factor: float = GEOM_MARGIN,
) -> Tuple[int, int]:
    """
    Choose B (central trap) and A (farthest trap satisfying geometry).
    Identical to vortex_entry_test.select_trap_pair.
    """
    grid_cx = 0.5 * (float(xg[0]) + float(xg[-1]))
    grid_cy = 0.5 * (float(yg[0]) + float(yg[-1]))
    center = np.array([grid_cx, grid_cy])

    if idx_B_override is not None:
        idx_B = idx_B_override
    else:
        dists_to_center = np.linalg.norm(traps_m - center, axis=1)
        idx_B = int(np.argmin(dists_to_center))

    B_xy = traps_m[idx_B]
    threshold = min_sep_factor * r_barrier
    dists_to_B = np.linalg.norm(traps_m - B_xy, axis=1)
    valid_mask = dists_to_B > threshold
    valid_mask[idx_B] = False

    print(f"\n  --- Trap Pair Selection ---")
    print(f"  B: trap [{idx_B}] at ({B_xy[0]*1e3:.3f}, {B_xy[1]*1e3:.3f}) mm")
    print(f"  Geometry constraint: d(A, B) > {min_sep_factor:.1f} * "
          f"{r_barrier*1e6:.1f} um = {threshold*1e6:.1f} um")
    print(f"  Candidates: {int(valid_mask.sum())} of {len(traps_m)} traps pass constraint")

    if not np.any(valid_mask):
        print(f"\n  FATAL: No trap satisfies d(A, B) > {threshold*1e6:.1f} um.")
        sys.exit(1)

    dists_valid = dists_to_B.copy()
    dists_valid[~valid_mask] = -np.inf
    idx_A = int(np.argmax(dists_valid))
    A_xy = traps_m[idx_A]
    d_AB = float(dists_to_B[idx_A])

    print(f"  A: trap [{idx_A}] at ({A_xy[0]*1e3:.3f}, {A_xy[1]*1e3:.3f}) mm  (farthest valid)")
    print(f"  d(A, B) = {d_AB*1e6:.1f} um = {d_AB/LAM:.2f} lam")
    print(f"  d(A, B) / r_barrier = {d_AB/r_barrier:.2f}  (>{min_sep_factor:.1f}  OK)")
    return idx_A, idx_B


def validate_geometry(
    traps_m: np.ndarray,
    idx_A: int,
    idx_B: int,
    r_barrier: float,
) -> str:
    """Check B starts outside the barrier ring. Returns status string."""
    d_B_vc = float(np.linalg.norm(traps_m[idx_B] - traps_m[idx_A]))
    ratio = d_B_vc / r_barrier if r_barrier > 0 else float("inf")

    if ratio > 1.1:
        status = "OUTSIDE"
    elif ratio > 0.9:
        status = "ON"
    else:
        status = "INSIDE"

    print(f"\n  --- Geometry Validation ---")
    print(f"  d(B, vortex_start) = {d_B_vc*1e6:.1f} um")
    print(f"  r_barrier          = {r_barrier*1e6:.1f} um")
    print(f"  ratio              = {ratio:.2f}")
    print(f"  B status:          {status}")
    if status != "OUTSIDE":
        print(f"  WARNING: B is NOT safely outside the barrier ring!")
    else:
        print(f"  OK: B starts {(d_B_vc - r_barrier)*1e6:.1f} um outside the barrier.")
    return status


# ════════════════════════════════════════════════════════════════
# Focused vortex source (same as vortex_entry_test)
# ════════════════════════════════════════════════════════════════

def _make_focused_vortex_source(
    XX: np.ndarray,
    YY: np.ndarray,
    charge: int,
    aperture_radius: float,
    k: float,
    focal_length: float,
    center: Tuple[float, float],
    apodization: str = "cosine_taper",
    apod_width: float = 0.15,
) -> np.ndarray:
    """Phase-only focused vortex: spiral phase + converging thin-lens phase."""
    cx, cy = center
    rx = XX - cx
    ry = YY - cy
    r = np.sqrt(rx ** 2 + ry ** 2)
    theta = np.arctan2(ry, rx)

    amp = np.ones_like(r)
    amp[r > aperture_radius] = 0.0

    if apodization == "cosine_taper" and apod_width > 0:
        r0 = aperture_radius * (1.0 - apod_width)
        mask = (r > r0) & (r <= aperture_radius)
        amp[mask] = 0.5 * (1.0 + np.cos(
            np.pi * (r[mask] - r0) / (aperture_radius * apod_width)
        ))

    a_max = float(amp.max())
    if a_max > 0:
        amp = amp / a_max

    phase = charge * theta - k * r ** 2 / (2.0 * focal_length)
    return amp * np.exp(1j * phase)


# ════════════════════════════════════════════════════════════════
# Phase-sweep carry helpers (same as vortex_entry_test)
# ════════════════════════════════════════════════════════════════

def _make_interp(F: np.ndarray, xg: np.ndarray, yg: np.ndarray) -> RegularGridInterpolator:
    return RegularGridInterpolator((yg, xg), F, bounds_error=False, fill_value=0.0)


def _eval_at(interp: RegularGridInterpolator, pos_xy: np.ndarray) -> np.ndarray:
    pts = np.column_stack([pos_xy[:, 1], pos_xy[:, 0]])
    return interp(pts)


def ring_barrier_test(
    iU: RegularGridInterpolator,
    vortex_center: np.ndarray,
    pos_B: np.ndarray,
    r_barrier: float,
    n_theta: int = N_RING_THETA,
    sector_half_deg: float = OPENING_SECTOR_DEG,
    opening_threshold: float = OPENING_THRESHOLD,
) -> dict:
    """Sample Gorkov around barrier ring; detect directional opening vs basin merging."""
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    ring_pts = np.zeros((n_theta, 2))
    ring_pts[:, 0] = vortex_center[0] + r_barrier * np.cos(theta)
    ring_pts[:, 1] = vortex_center[1] + r_barrier * np.sin(theta)
    U_ring = _eval_at(iU, ring_pts)

    dxy = pos_B - vortex_center
    theta_B = float(np.arctan2(dxy[1], dxy[0]))
    sector_half_rad = np.deg2rad(sector_half_deg)
    ang_diff = np.abs(np.angle(np.exp(1j * (theta - theta_B))))
    in_sector = ang_diff <= sector_half_rad

    U_B_sector = float(np.mean(U_ring[in_sector])) if np.any(in_sector) else float("nan")
    U_other = float(np.mean(U_ring[~in_sector])) if np.any(~in_sector) else float("nan")
    denom = abs(U_other) if U_other != 0.0 else 1.0
    delta_B = float((U_other - U_B_sector) / denom)

    if delta_B > opening_threshold:
        mechanism = "true_opening"
    elif delta_B > 0.0:
        mechanism = "weak_asymmetry"
    else:
        mechanism = "basin_merging"

    return {
        "theta": theta, "U_ring": U_ring, "theta_B": theta_B,
        "U_B_sector": U_B_sector, "U_other": U_other,
        "delta_B": delta_B, "mechanism": mechanism,
    }


def phase_sweep_carry(
    p_sw: np.ndarray,
    vortex_gen: "VortexPerturbation",
    xg: np.ndarray,
    yg: np.ndarray,
    dx: float,
    dy: float,
    psi_values: np.ndarray,
    alpha: float,
    beta: float,
    vortex_center: np.ndarray,
    pos_A: np.ndarray,
    pos_B: np.ndarray,
    r_barrier: float,
    w_barrier: float = W_BARRIER_DEFAULT,
    w_pull: float = W_PULL_DEFAULT,
    w_lateral: float = W_LATERAL_DEFAULT,
    w_retain: float = W_RETAIN_DEFAULT,
    opening_threshold: float = OPENING_THRESHOLD,
) -> dict:
    """
    Phase sweep matching vortex_entry_test: select ψ that retains A and
    minimises barrier score for B. Returns best-psi dict including interpolators.
    """
    e_B = vortex_center - pos_B
    norm_B = np.linalg.norm(e_B)
    e_B = e_B / norm_B if norm_B > 0 else np.array([0.0, 1.0])

    e_A = vortex_center - pos_A
    norm_A = np.linalg.norm(e_A)
    e_A = e_A / norm_A if norm_A > 0 else np.array([0.0, 1.0])

    t_samples = np.linspace(0.05, 0.95, N_BARRIER_SAMPLES)
    line_pts = pos_B[None, :] + t_samples[:, None] * (vortex_center - pos_B)[None, :]

    results = []
    for psi in psi_values:
        p_v = vortex_gen.get_field(vortex_center)
        p_tot = beta * p_sw + alpha * np.exp(1j * psi) * p_v
        U_n, Fx_n, Fy_n = gorkov_normalised(p_tot, dx, dy)

        iU = _make_interp(U_n, xg, yg)
        iFx = _make_interp(Fx_n, xg, yg)
        iFy = _make_interp(Fy_n, xg, yg)

        FA = np.array([float(_eval_at(iFx, pos_A[None, :])[0]),
                       float(_eval_at(iFy, pos_A[None, :])[0])])
        F_A_in = float(np.dot(FA, e_A))

        FB = np.array([float(_eval_at(iFx, pos_B[None, :])[0]),
                       float(_eval_at(iFy, pos_B[None, :])[0])])
        F_in_B = float(np.dot(FB, e_B))
        F_tang_B = FB - F_in_B * e_B
        F_perp_B = float(np.linalg.norm(F_tang_B))

        U_path = _eval_at(iU, line_pts)
        U_B = float(_eval_at(iU, pos_B[None, :])[0])
        DeltaU = float(np.max(U_path) - U_B)

        score = (w_barrier * DeltaU
                 - w_pull * F_in_B
                 + w_lateral * F_perp_B
                 - w_retain * F_A_in)

        results.append({
            "psi": psi, "F_A_in": F_A_in, "F_in_B": F_in_B,
            "F_perp_B": F_perp_B, "DeltaU": DeltaU, "score": score,
            "iU": iU, "iFx": iFx, "iFy": iFy,
        })

    best = min(results, key=lambda r: r["score"])
    best["mechanism"] = ring_barrier_test(best["iU"], vortex_center, pos_B, r_barrier,
                                          opening_threshold=opening_threshold)
    return best


def _ol_vortex_path(start: np.ndarray, end: np.ndarray, n_steps: int) -> np.ndarray:
    """Linear vortex path from start to end in n_steps."""
    t = np.linspace(0, 1, n_steps)
    return start[None, :] + t[:, None] * (end - start)[None, :]


def _b_status(pos_B: np.ndarray, vortex_center: np.ndarray, r_barrier: float) -> str:
    d = float(np.linalg.norm(pos_B - vortex_center))
    if d > r_barrier * 1.05:
        return "OUTSIDE"
    if d < r_barrier * 0.95:
        return "INSIDE"
    return "ON"


def _compute_view(
    traps_m: np.ndarray,
    idx_A: int,
    idx_B: int,
    vortex_path: np.ndarray,
    margin_mm: float = VIEW_MARGIN_MM,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Fixed view covering A, B, vortex path, and nearby cluster traps."""
    key_pts = np.vstack([
        traps_m[idx_A][None, :],
        traps_m[idx_B][None, :],
        vortex_path,
    ]) * 1e3  # mm
    mid = 0.5 * (traps_m[idx_A] + traps_m[idx_B])
    d_to_mid = np.linalg.norm(traps_m - mid, axis=1)
    nearby = d_to_mid < 1.5e-3
    if np.any(nearby):
        key_pts = np.vstack([key_pts, traps_m[nearby] * 1e3])
    view_xlim = (float(key_pts[:, 0].min()) - margin_mm,
                 float(key_pts[:, 0].max()) + margin_mm)
    view_ylim = (float(key_pts[:, 1].min()) - margin_mm,
                 float(key_pts[:, 1].max()) + margin_mm)
    return view_xlim, view_ylim


# ════════════════════════════════════════════════════════════════
# Gradient check
# ════════════════════════════════════════════════════════════════

def gradient_check(
    feval: ForceEvaluator,
    u0: np.ndarray,
    pos: np.ndarray,
    idx_A: int,
    idx_B: int,
    neigh_idx: np.ndarray,
    target_pos: np.ndarray,
    cfg: MPCConfig,
) -> bool:
    """
    Verify analytic MPC gradients match centered finite differences.

    Returns True if all relative errors < 5%.
    """
    from scripts.lib.mpc_controller import (
        compute_mpc_gradients,
        compute_total_cost,
        rollout,
    )

    controls = [u0.copy() for _ in range(cfg.K)]
    state = rollout(feval, controls, pos, cfg.dt)
    grads = compute_mpc_gradients(
        feval, state, idx_A, idx_B, neigh_idx, target_pos, cfg,
    )

    print("\n  Gradient check (analytic vs centered FD):")
    names = ["psi", "x_v", "y_v", "alpha", "beta"]
    eps_vals = [1e-4, 5e-6, 5e-6, 1e-4, 1e-4]
    all_ok = True

    # Check gradient for step t=0
    for dim in range(N_CTRL):
        eps = eps_vals[dim]
        controls_p = [u.copy() for u in controls]
        controls_m = [u.copy() for u in controls]
        controls_p[0][dim] += eps
        controls_m[0][dim] -= eps

        J_p = compute_total_cost(
            feval, controls_p, pos, idx_A, idx_B, neigh_idx, target_pos, cfg,
        )
        J_m = compute_total_cost(
            feval, controls_m, pos, idx_A, idx_B, neigh_idx, target_pos, cfg,
        )
        fd_grad = (J_p - J_m) / (2.0 * eps)
        analytic = grads[0][dim]
        rel_err = abs(analytic - fd_grad) / max(abs(fd_grad), 1e-30)
        ok = rel_err < 0.05 or abs(fd_grad) < 1e-20
        if not ok:
            all_ok = False
        print(f"    {names[dim]:>6s}: analytic={analytic:+.6e}  fd={fd_grad:+.6e}  "
              f"rel_err={rel_err:.4f}  {'OK' if ok else 'FAIL'}")

    return all_ok


# ════════════════════════════════════════════════════════════════
# GIF rendering
# ════════════════════════════════════════════════════════════════

def _gif_frame(
    ax,
    p_sw: np.ndarray,
    vortex_gen,
    xg: np.ndarray,
    yg: np.ndarray,
    traps_m: np.ndarray,
    idx_A: int,
    idx_B: int,
    neigh_idx: np.ndarray,
    r_barrier: float,
    view_xlim: Tuple[float, float],
    view_ylim: Tuple[float, float],
    extent,
    dx: float,
    dy: float,
    r_barrier_mm: float,
    traps_mm: np.ndarray,
    psi: float,
    alpha: float,
    vc_m: np.ndarray,
    pos_A_m: np.ndarray,
    pos_B_m: np.ndarray,
    neigh_pos_m: Optional[np.ndarray],
    trail_mm: Optional[np.ndarray],
    label: str,
    d_AB_um: float,
    b_stat: str,
    fi: int,
    n_frames_total: int,
) -> None:
    """Draw one GIF frame onto ax. Shared between carry and MPC phases."""
    p_total = total_pressure(p_sw, vortex_gen, psi, alpha, BETA_FIXED, vc_m)
    U, _, _ = gorkov_from_pressure(p_total, dx, dy)

    ix_lo = int(np.searchsorted(xg, (view_xlim[0] - 0.1) * 1e-3))
    ix_hi = int(np.searchsorted(xg, (view_xlim[1] + 0.1) * 1e-3))
    iy_lo = int(np.searchsorted(yg, (view_ylim[0] - 0.1) * 1e-3))
    iy_hi = int(np.searchsorted(yg, (view_ylim[1] + 0.1) * 1e-3))
    U_roi = U[iy_lo:iy_hi, ix_lo:ix_hi]
    lo = float(np.percentile(U_roi, GORKOV_CLIP_LO))
    hi = float(np.percentile(U_roi, GORKOV_CLIP_HI))

    ax.imshow(U, origin="lower", extent=extent, cmap=CMAP,
              vmin=lo, vmax=hi, aspect="equal", interpolation="bicubic")

    vc_mm = vc_m * 1e3
    pos_A_mm = pos_A_m * 1e3
    pos_B_mm = pos_B_m * 1e3

    # Trap markers (only those inside view)
    in_view = (
        (traps_mm[:, 0] >= view_xlim[0]) & (traps_mm[:, 0] <= view_xlim[1])
        & (traps_mm[:, 1] >= view_ylim[0]) & (traps_mm[:, 1] <= view_ylim[1])
    )
    if np.any(in_view):
        ax.scatter(traps_mm[in_view, 0], traps_mm[in_view, 1],
                   marker="x", c=COL_TRAP, s=28, linewidths=0.8,
                   zorder=4, alpha=0.5)

    ax.annotate("A-home", traps_mm[idx_A], fontsize=6, color=COL_HOME_A,
                alpha=0.8, ha="left", va="bottom",
                xytext=(3, 3), textcoords="offset points")
    ax.annotate("B-home", traps_mm[idx_B], fontsize=6, color=COL_HOME_B,
                alpha=0.8, ha="left", va="bottom",
                xytext=(3, 3), textcoords="offset points")

    if trail_mm is not None and len(trail_mm) > 1:
        ax.plot(trail_mm[:, 0], trail_mm[:, 1], color=COL_CTR,
                lw=0.9, alpha=0.75, zorder=5)
    ax.plot(vc_mm[0], vc_mm[1], marker="+", color=COL_CTR,
            ms=12, mew=2.0, zorder=8)

    ax.add_patch(mpatches.Circle(
        (vc_mm[0], vc_mm[1]), r_barrier_mm,
        fill=False, edgecolor=COL_BARRIER, linestyle="--",
        linewidth=1.0, alpha=0.7, zorder=6))

    if neigh_pos_m is not None:
        for ni_pos in neigh_pos_m:
            ax.add_patch(mpatches.Circle(
                (ni_pos[0] * 1e3, ni_pos[1] * 1e3), PARTICLE_RADIUS_MM,
                facecolor=COL_NEIGH, edgecolor="white", linewidth=0.6,
                zorder=7, alpha=0.7))

    ax.add_patch(mpatches.Circle(
        (pos_A_mm[0], pos_A_mm[1]), PARTICLE_RADIUS_MM,
        facecolor=COL_A, edgecolor="white", linewidth=0.6,
        zorder=8, alpha=0.9))
    ax.add_patch(mpatches.Circle(
        (pos_B_mm[0], pos_B_mm[1]), PARTICLE_RADIUS_MM,
        facecolor=COL_B, edgecolor="white", linewidth=0.6,
        zorder=8, alpha=0.9))

    ax.scatter([], [], c=COL_A, s=30, label="A (vortex)")
    ax.scatter([], [], c=COL_B, s=30, label="B (SW trap)")

    ax.set_xlim(*view_xlim)
    ax.set_ylim(*view_ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(
        f"[{label}] psi={psi:.2f} | alpha={alpha:.1f} | "
        f"d(A,B)={d_AB_um:.0f} um | B: {b_stat}\n"
        f"frame {fi + 1}/{n_frames_total}",
        fontsize=9,
    )
    ax.legend(loc="upper right", fontsize=7, framealpha=0.7)


def _fig_to_pil(fig) -> "Image.Image":
    fig.tight_layout()
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    img = Image.fromarray(buf.reshape(h, w, 4)).convert("RGB")
    plt.close(fig)
    return img


def render_gif(
    result: MPCResult,
    p_sw: np.ndarray,
    vortex_gen: "VortexPerturbation",
    xg: np.ndarray,
    yg: np.ndarray,
    traps_m: np.ndarray,
    idx_A: int,
    idx_B: int,
    neigh_idx: np.ndarray,
    out_path: Path,
    r_barrier: float,
    view_xlim: Tuple[float, float],
    view_ylim: Tuple[float, float],
    n_frames: int = N_GIF_FRAMES,
    duration_ms: int = GIF_DURATION_MS,
    open_loop_steps: int = 0,
    T_mpc: int = 0,
    carry_frames: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Render transport GIF matching vortex_entry_test visual style.

    If carry_frames is provided, those frames are rendered first (identical
    rendering to vortex_entry_test), then MPC+settle frames are uniformly
    subsampled from result[open_loop_steps:].
    """
    dx = float(xg[1] - xg[0])
    dy = float(yg[1] - yg[0])
    x_mm = xg * 1e3
    y_mm = yg * 1e3
    extent = [x_mm[0], x_mm[-1], y_mm[0], y_mm[-1]]
    traps_mm = traps_m * 1e3
    r_barrier_mm = r_barrier * 1e3

    # ── Part 1: carry frames (script-1 style, pre-captured dicts) ──
    frames: List[Image.Image] = []
    n_carry_frames = len(carry_frames) if carry_frames else 0

    if carry_frames:
        print(f"  Rendering {n_carry_frames} carry GIF frames...")
        for fi, fdat in enumerate(carry_frames):
            vc_m = fdat["vortex_center"]
            trail = fdat.get("vortex_trail", [])
            trail_mm = np.array(trail) * 1e3 if len(trail) > 1 else None
            fig, ax = plt.subplots(1, 1, figsize=(6.0, 6.0))
            _gif_frame(
                ax, p_sw, vortex_gen, xg, yg, traps_m, idx_A, idx_B, neigh_idx,
                r_barrier, view_xlim, view_ylim, extent, dx, dy, r_barrier_mm, traps_mm,
                psi=float(fdat["psi"]),
                alpha=float(fdat["alpha"]),
                vc_m=vc_m,
                pos_A_m=fdat["pos_A"],
                pos_B_m=fdat["pos_B"],
                neigh_pos_m=None,   # carry phase: only A and B shown (matches script 1)
                trail_mm=trail_mm,
                label=fdat.get("label", "CARRY"),
                d_AB_um=float(fdat["d_AB"]) * 1e6,
                b_stat=fdat.get("B_status", ""),
                fi=fi,
                n_frames_total=n_carry_frames,
            )
            frames.append(_fig_to_pil(fig))
            if (fi + 1) % 50 == 0:
                print(f"    {fi + 1}/{n_carry_frames}")

    # ── Part 2: MPC + settle frames, subsampled from result ────────
    T_mpc_settle = len(result.applied_controls) - open_loop_steps
    n_mpc_frames = max(0, n_frames - n_carry_frames)
    frame_every = max(1, T_mpc_settle // n_mpc_frames) if n_mpc_frames > 0 else 1
    mpc_indices = list(range(open_loop_steps, len(result.applied_controls), frame_every))
    if mpc_indices and mpc_indices[-1] != len(result.applied_controls) - 1:
        mpc_indices.append(len(result.applied_controls) - 1)

    n_total_frames = n_carry_frames + len(mpc_indices)
    print(f"  Rendering {len(mpc_indices)} MPC/settle GIF frames...")

    for mfi, step in enumerate(mpc_indices):
        fi = n_carry_frames + mfi
        u = result.applied_controls[step]
        psi, xv, yv, alpha, beta = u
        pos_all = result.positions[step]
        vc_m = np.array([xv, yv])

        # Full trail from control history
        trail_mm = np.array(
            [[c[I_XV], c[I_YV]] for c in result.applied_controls[:step + 1]]
        ) * 1e3

        d_AB_um = float(np.linalg.norm(pos_all[idx_A] - pos_all[idx_B]) * 1e6)
        b_stat = _b_status(pos_all[idx_B], vc_m, r_barrier)
        if step < open_loop_steps + T_mpc:
            phase_label = "MPC"
        else:
            phase_label = "SETTLE"

        fig, ax = plt.subplots(1, 1, figsize=(6.0, 6.0))
        _gif_frame(
            ax, p_sw, vortex_gen, xg, yg, traps_m, idx_A, idx_B, neigh_idx,
            r_barrier, view_xlim, view_ylim, extent, dx, dy, r_barrier_mm, traps_mm,
            psi=psi,
            alpha=alpha,
            vc_m=vc_m,
            pos_A_m=pos_all[idx_A],
            pos_B_m=pos_all[idx_B],
            neigh_pos_m=pos_all[neigh_idx] if len(neigh_idx) > 0 else None,
            trail_mm=trail_mm,
            label=phase_label,
            d_AB_um=d_AB_um,
            b_stat=b_stat,
            fi=fi,
            n_frames_total=n_total_frames,
        )
        frames.append(_fig_to_pil(fig))
        if (mfi + 1) % 50 == 0:
            print(f"    {mfi + 1}/{len(mpc_indices)}")

    if not frames:
        print("  WARNING: no frames to save")
        return

    frames[0].save(
        str(out_path), save_all=True, append_images=frames[1:],
        duration=duration_ms, loop=0,
    )
    print(f"  Saved GIF ({len(frames)} frames): {out_path}")


# ════════════════════════════════════════════════════════════════
# Convergence plot
# ════════════════════════════════════════════════════════════════

def plot_convergence(result: MPCResult, out_path: Path) -> None:
    """Plot MPC outer-loop cost and inner convergence."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Outer loop
    ax = axes[0]
    ax.plot(result.J_history, ".-")
    ax.set_xlabel("Replan step")
    ax.set_ylabel("Horizon cost J")
    ax.set_title("MPC outer-loop cost")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # Sample inner loops
    ax = axes[1]
    n_show = min(8, len(result.inner_J_histories))
    step = max(1, len(result.inner_J_histories) // n_show)
    for i in range(0, len(result.inner_J_histories), step):
        hist = result.inner_J_histories[i]
        if len(hist) > 1:
            ax.plot(hist, label=f"replan {i}")
    ax.set_xlabel("Inner iteration")
    ax.set_ylabel("J")
    ax.set_title("Inner-loop convergence (sampled)")
    ax.set_yscale("log")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"  Saved convergence plot: {out_path}")


# ════════════════════════════════════════════════════════════════
# Control & trajectory CSV
# ════════════════════════════════════════════════════════════════

def save_controls_csv(result: MPCResult, dt: float, path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "t_ms", "psi", "x_v", "y_v", "alpha", "beta"])
        for i, u in enumerate(result.applied_controls):
            w.writerow([i, i * dt * 1e3, *u])


def save_trajectory_csv(result: MPCResult, idx_A: int, idx_B: int, dt: float, path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "t_ms", "xA_mm", "yA_mm", "xB_mm", "yB_mm", "d_AB_um"])
        for i, pos in enumerate(result.positions):
            xA, yA = pos[idx_A] * 1e3
            xB, yB = pos[idx_B] * 1e3
            d = float(np.linalg.norm(pos[idx_A] - pos[idx_B]) * 1e6)
            w.writerow([i, i * dt * 1e3, xA, yA, xB, yB, d])


# ════════════════════════════════════════════════════════════════
# Settling phase — ramp α→0, let standing wave recapture
# ════════════════════════════════════════════════════════════════

def run_settling_phase(
    feval: ForceEvaluator,
    result: MPCResult,
    cfg: MPCConfig,
    idx_A: int,
    idx_B: int,
    T_settle: int = T_SETTLE_DEFAULT,
) -> MPCResult:
    """
    Append a settling phase to an MPC result.

    After the MPC transport phase:
      1. Rapidly ramp α → 0 over 50 steps (5 ms) to remove vortex influence.
      2. Hold α = 0, β = 1.0 for the remaining steps, letting the standing
         wave recapture all particles into their nearest trap nodes.

    If A has been brought within λ/4 of B's node, both settle into
    the same node → merge.

    The function mutates *result* in-place (appending to lists) and returns it.
    """
    step_scale = SCALE * cfg.dt

    pos = result.positions[-1].copy()
    u_final = result.applied_controls[-1].copy()
    alpha0 = u_final[I_ALPHA]
    beta0 = u_final[I_BETA]

    N_RAMP = min(50, T_settle)  # ramp α → 0 in 50 steps (5ms)

    print(f"\n  Settling phase: {T_settle} steps "
          f"(ramp {N_RAMP} + relax {T_settle - N_RAMP}), "
          f"α {alpha0:.3f}→0, β {beta0:.3f}→1.0")

    for k in range(T_settle):
        u = u_final.copy()

        if k < N_RAMP:
            # Quick ramp: α → 0, β → 1.0
            frac = (k + 1) / N_RAMP
            u[I_ALPHA] = alpha0 * (1.0 - frac)
            u[I_BETA] = beta0 + (1.0 - beta0) * frac
        else:
            # Pure SW relaxation
            u[I_ALPHA] = 0.0
            u[I_BETA] = 1.0

        u = np.clip(u, cfg.u_lo, cfg.u_hi)

        result.applied_controls.append(u.copy())
        Fx, Fy = feval.forces(u, pos)
        result.forces.append((Fx.copy(), Fy.copy()))
        pos = pos.copy()
        pos[:, 0] += step_scale * Fx
        pos[:, 1] += step_scale * Fy
        result.positions.append(pos.copy())

        # Check merge
        if result.merge_time_s is None:
            d_AB = float(np.linalg.norm(pos[idx_A] - pos[idx_B]))
            if d_AB < CAPTURE_RADIUS:
                result.merge_time_s = (cfg.T + k + 1) * cfg.dt

    d_final = float(np.linalg.norm(pos[idx_A] - pos[idx_B]) * 1e6)
    print(f"  Settle done. d(A,B) = {d_final:.1f} µm, "
          f"α_final = {result.applied_controls[-1][I_ALPHA]:.4f}")
    return result


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="MPC Vortex Merge")
    parser.add_argument("--K", type=int, default=10, help="MPC horizon")
    parser.add_argument("--T", type=int, default=3000, help="Total steps")
    parser.add_argument("--n_iters", type=int, default=10, help="Inner optimiser iterations")
    parser.add_argument("--replan_every", type=int, default=5, help="Replan frequency")
    parser.add_argument("--w_merge", type=float, default=1e12)
    parser.add_argument("--w_B_stable", type=float, default=1e12)
    parser.add_argument("--w_neigh", type=float, default=5e11)
    parser.add_argument("--w_smooth", type=float, default=1e8)
    parser.add_argument("--waist", type=float, default=VORTEX_WAIST * 1e3, help="Vortex waist [mm] (LG only)")
    parser.add_argument("--charge", type=int, default=VORTEX_CHARGE)
    parser.add_argument("--alpha_max", type=float, default=3.0, help="Max vortex amplitude")
    parser.add_argument("--alpha_init", type=float, default=3.0, help="Initial vortex amplitude (MPC phase)")
    parser.add_argument("--n_gif_frames", type=int, default=N_GIF_FRAMES)
    parser.add_argument("--gradcheck", action="store_true", help="Run gradient check only")
    parser.add_argument("--n_particles", type=int, default=9, help="Number of particles (2-9); default all")
    parser.add_argument("--T_settle", type=int, default=T_SETTLE_DEFAULT,
                        help="Settling-phase steps (α→0 ramp-down after MPC)")
    # ── Vortex beam (focused Bessel, same as vortex_entry_test) ──
    parser.add_argument("--vortex_source", type=str, default="bessel",
                        choices=["lg", "bessel"],
                        help="Vortex field model: 'bessel' (focused Bessel, default) or 'lg'")
    parser.add_argument("--bessel_aperture", type=float, default=BESSEL_APERTURE_MM_DEFAULT,
                        help="Lens aperture radius [mm] (default %(default)s)")
    parser.add_argument("--bessel_prop_dist", type=float, default=BESSEL_PROP_DIST_MM_DEFAULT,
                        help="ASM propagation distance [mm] (default %(default)s)")
    parser.add_argument("--focus_mm", type=float, default=FOCUS_MM_DEFAULT,
                        help="Lens focal length [mm] (default %(default)s)")
    # ── Phase-sweep carry (Phase I, same as vortex_entry_test) ───
    parser.add_argument("--n_psi", type=int, default=N_PSI_DEFAULT,
                        help="Number of phase-sweep values (default %(default)s)")
    parser.add_argument("--n_vortex_steps", type=int, default=N_VORTEX_STEPS_DEFAULT,
                        help="Vortex path steps in Phase I carry (default %(default)s)")
    parser.add_argument("--n_dyn_steps", type=int, default=N_DYN_STEPS_DEFAULT,
                        help="Dynamics steps per vortex position in Phase I (default %(default)s)")
    parser.add_argument("--w_barrier", type=float, default=W_BARRIER_DEFAULT)
    parser.add_argument("--w_pull", type=float, default=W_PULL_DEFAULT)
    parser.add_argument("--w_lateral", type=float, default=W_LATERAL_DEFAULT)
    parser.add_argument("--w_retain", type=float, default=W_RETAIN_DEFAULT,
                        help="Weight on -F_A_in in phase sweep score (default %(default)s)")
    parser.add_argument("--max_step", type=float, default=MAX_STEP_DEFAULT,
                        help="Max particle displacement per dynamics step [m]")
    # ── Trap selection ───────────────────────────────────────────
    parser.add_argument("--idx_B", type=int, default=None,
                        help="Override B trap index (default: nearest to grid centre)")
    parser.add_argument("--min_sep_factor", type=float, default=GEOM_MARGIN,
                        help="A must satisfy d(A,B) > factor * r_barrier (default %(default)s)")
    parser.add_argument("--n_neigh_radius", type=float, default=N_NEIGH_RADIUS_DEFAULT,
                        help="MPC neighbours: traps within this many trap-spacings of A-B midpoint")
    # ── MPC activation ───────────────────────────────────────────
    parser.add_argument("--mpc_activation_radius", type=float, default=MPC_ACTIVATION_RADIUS_LAM,
                        help="Distance from B at which MPC activates [λ]")
    parser.add_argument("--ol_alpha", type=float, default=3.0,
                        help="Vortex amplitude α during Phase I carry")
    parser.add_argument("--ol_beta", type=float, default=BETA_FIXED,
                        help="Standing-wave amplitude β during Phase I carry")
    # ── MPC per-step rate limits ──────────────────────────────────
    parser.add_argument("--vxy_rate", type=float, default=0.5,
                        help="MPC vortex position rate limit [µm/step]")
    parser.add_argument("--psi_rate", type=float, default=2.0 * np.pi,
                        help="MPC ψ rate limit [rad/step] (default 2π = free)")
    args = parser.parse_args()

    # ── Load data (same source as vortex_entry_test) ─────────────
    bessel_R = args.bessel_aperture * 1e-3
    bessel_z = args.bessel_prop_dist * 1e-3
    focus_f  = args.focus_mm * 1e-3
    (p_sw, xg, yg, dx, dy,
     traps_m_all, vortex_gen, r_barrier) = load_data(bessel_R, bessel_z, focus_f)
    vortex_family = "focused_bessel_asm" if args.vortex_source == "bessel" else "LG"

    # ── Select A/B trap pair (automatic, same as vortex_entry_test) ──
    idx_A, idx_B = select_trap_pair(
        traps_m_all, r_barrier, xg, yg,
        idx_B_override=args.idx_B,
        min_sep_factor=args.min_sep_factor,
    )
    validate_geometry(traps_m_all, idx_A, idx_B, r_barrier)

    A_remote_xy = traps_m_all[idx_A].copy()
    B_xy        = traps_m_all[idx_B].copy()

    # Target positions for MPC: A merges into B's trap, others stay home.
    traps_m = traps_m_all.copy()
    traps_m[idx_A] = B_xy.copy()

    # Init positions: all particles at their home traps.
    init_pos = traps_m_all.copy()

    # Neighbours for MPC: only traps within n_neigh_radius trap-spacings of
    # the A-B midpoint (avoids giving MPC 195 particles to track).
    mid_AB  = 0.5 * (A_remote_xy + B_xy)
    neigh_radius = args.n_neigh_radius * TRAP_SP
    dists_to_mid = np.linalg.norm(traps_m_all - mid_AB, axis=1)
    neigh_idx = np.array(
        [i for i in range(len(traps_m_all))
         if i not in (idx_A, idx_B) and dists_to_mid[i] < neigh_radius],
        dtype=int,
    )

    d_AB_init = np.linalg.norm(A_remote_xy - B_xy)
    print(f"\nParticles: A={idx_A}, B={idx_B}, neighbours={len(neigh_idx)} "
          f"(within {args.n_neigh_radius:.1f} trap spacings of midpoint)")
    print(f"  A (start):  {A_remote_xy*1e3} mm")
    print(f"  B (target): {B_xy*1e3} mm")
    print(f"  d(A,B) = {d_AB_init*1e6:.1f} µm = {d_AB_init/LAM:.1f} λ")

    # ── Configure MPC ─────────────────────────────────────────────
    corridor_margin = 2.0 * TRAP_SP
    all_positions = np.vstack([A_remote_xy[np.newaxis, :], traps_m])

    cfg = MPCConfig(
        K=args.K,
        T=args.T,
        n_iters=args.n_iters,
        replan_every=args.replan_every,
        w_merge=args.w_merge,
        w_B_stable=args.w_B_stable,
        w_neigh=args.w_neigh,
        w_smooth=args.w_smooth,
        w_beta_high=1e4,
    )
    # Per-DOF smoothness: ψ effectively free, xy/α/β smooth
    cfg.w_smooth_vec = np.array([
        1.0,                   # ψ: minimal penalty (effectively free)
        args.w_smooth,         # x_v: full smoothness
        args.w_smooth,         # y_v: full smoothness
        args.w_smooth * 0.1,   # α: moderate smoothness
        args.w_smooth * 0.1,   # β: moderate smoothness
    ])

    cfg.u_lo = np.array([
        0.0,
        float(np.min(all_positions[:, 0])) - corridor_margin,
        float(np.min(all_positions[:, 1])) - corridor_margin,
        0.0,
        0.2,                          # β lower bound (must allow open-loop β)
    ])
    cfg.u_hi = np.array([
        2.0 * np.pi,
        float(np.max(all_positions[:, 0])) + corridor_margin,
        float(np.max(all_positions[:, 1])) + corridor_margin,
        args.alpha_max,
        1.0,
    ])

    # Per-step rate limits:
    #   ψ:   large (effectively free — optimizer can jump anywhere in [0, 2π])
    #   x,y: slow (smooth vortex translation, default 0.5 µm/step = 5 mm/s)
    #   α:   ±0.005/step (ramp 0→2 over 400 steps = 40ms)
    #   β:   ±0.005/step
    cfg.du_max = np.array([
        args.psi_rate,              # ψ: free (2π/step by default)
        args.vxy_rate * 1e-6,       # x_v:  µm/step → m/step
        args.vxy_rate * 1e-6,       # y_v
        0.005,                      # α
        0.005,                      # β
    ])

    feval = ForceEvaluator(p_sw, vortex_gen, xg, yg)

    # ── Gradient check ────────────────────────────────────────────
    if args.gradcheck:
        u0_check = np.array([0.0, A_remote_xy[0], A_remote_xy[1], args.alpha_init, 0.9])
        u0_check = np.clip(u0_check, cfg.u_lo, cfg.u_hi)
        ok = gradient_check(
            feval, u0_check, init_pos.copy(), idx_A, idx_B, neigh_idx,
            traps_m.copy(), cfg,
        )
        print(f"\n  Gradient check {'PASSED' if ok else 'FAILED'}")
        return

    # ════════════════════════════════════════════════════════════════
    #  Phase I — Phase-sweep carry: move vortex (with A inside) toward B
    #  At each vortex step, sweep ψ to retain A and minimise barrier score.
    #  Identical method to vortex_entry_test; MPC takes over near B.
    # ════════════════════════════════════════════════════════════════
    mpc_activation_dist = args.mpc_activation_radius * LAM

    # Build discrete vortex path from A's trap to mpc_activation_dist before B
    transport_vec = traps_m[idx_B] - A_remote_xy
    transport_dist = float(np.linalg.norm(transport_vec))
    transport_dir = transport_vec / max(transport_dist, 1e-12)
    ol_travel_dist = max(0.0, transport_dist - mpc_activation_dist)
    ol_end = A_remote_xy + transport_dir * ol_travel_dist
    vortex_path_ol = _ol_vortex_path(A_remote_xy, ol_end, args.n_vortex_steps)

    # Fixed view: computed once from vortex path + cluster traps
    view_xlim, view_ylim = _compute_view(traps_m, idx_A, idx_B, vortex_path_ol)
    print(f"  Render view: x=[{view_xlim[0]:.2f}, {view_xlim[1]:.2f}] "
          f"y=[{view_ylim[0]:.2f}, {view_ylim[1]:.2f}] mm")

    psi_values = np.linspace(0, 2 * np.pi, args.n_psi, endpoint=False)
    step_scale = SCALE * cfg.dt
    pos = init_pos.copy()
    ol_positions = [pos.copy()]
    ol_controls = []
    ol_forces = []

    # Frame snapshot indices within each dyn sub-loop (matches vortex_entry_test)
    snap_indices = set(
        np.linspace(0, args.n_dyn_steps - 1, FRAMES_PER_VSTEP, dtype=int).tolist()
    )
    carry_frames: List[Dict[str, Any]] = []
    vortex_trail: List[np.ndarray] = []

    # ── Intro frames: standing wave only (α=0), before vortex activates ──
    intro_vc = vortex_path_ol[0].copy()
    b_stat_intro = _b_status(pos[idx_B], intro_vc, r_barrier)
    for _ in range(N_INTRO_FRAMES):
        carry_frames.append({
            "pos_A": pos[idx_A].copy(),
            "pos_B": pos[idx_B].copy(),
            "vortex_center": intro_vc.copy(),
            "vortex_trail": [intro_vc.copy()],
            "psi": 0.0, "alpha": 0.0,
            "d_AB": float(np.linalg.norm(pos[idx_A] - pos[idx_B])),
            "label": "INTRO", "B_status": b_stat_intro,
        })

    print(f"\n  Phase I — Phase-sweep carry ({args.n_vortex_steps} vortex steps × "
          f"{args.n_dyn_steps} dyn steps):")
    print(f"    Travel: {ol_travel_dist*1e3:.2f} mm, MPC activates at "
          f"{mpc_activation_dist*1e3:.2f} mm from B")

    for v_step in range(args.n_vortex_steps):
        vortex_center = vortex_path_ol[v_step].copy()
        vortex_trail.append(vortex_center.copy())

        best = phase_sweep_carry(
            p_sw, vortex_gen, xg, yg, dx, dy,
            psi_values, args.ol_alpha, args.ol_beta,
            vortex_center, pos[idx_A].copy(), pos[idx_B].copy(), r_barrier,
            w_barrier=args.w_barrier, w_pull=args.w_pull, w_lateral=args.w_lateral,
            w_retain=args.w_retain,
        )
        psi_best = float(best["psi"])
        mech = best["mechanism"]["mechanism"]
        ret_s = "OK" if best["F_A_in"] > 0 else "FALLBACK"
        print(f"    v_step {v_step+1:02d}/{args.n_vortex_steps} | "
              f"psi={psi_best:.2f} | {mech} | A-retain: {ret_s}")

        u_ol = np.array([psi_best, vortex_center[0], vortex_center[1],
                         args.ol_alpha, args.ol_beta])
        u_ol = np.clip(u_ol, cfg.u_lo, cfg.u_hi)

        for dyn_i in range(args.n_dyn_steps):
            Fx, Fy = feval.forces(u_ol, pos)
            ol_controls.append(u_ol.copy())
            ol_forces.append((Fx.copy(), Fy.copy()))
            pos = pos.copy()
            pos[:, 0] += np.clip(step_scale * Fx, -args.max_step, args.max_step)
            pos[:, 1] += np.clip(step_scale * Fy, -args.max_step, args.max_step)
            pos[:, 0] = np.clip(pos[:, 0], xg[2], xg[-3])
            pos[:, 1] = np.clip(pos[:, 1], yg[2], yg[-3])
            ol_positions.append(pos.copy())

            if dyn_i in snap_indices:
                b_stat = _b_status(pos[idx_B], vortex_center, r_barrier)
                carry_frames.append({
                    "pos_A": pos[idx_A].copy(),
                    "pos_B": pos[idx_B].copy(),
                    "vortex_center": vortex_center.copy(),
                    "vortex_trail": [v.copy() for v in vortex_trail],
                    "psi": psi_best, "alpha": args.ol_alpha,
                    "d_AB": float(np.linalg.norm(pos[idx_A] - pos[idx_B])),
                    "label": "CARRY", "B_status": b_stat,
                })

    open_loop_steps = len(ol_controls)
    open_loop_time = open_loop_steps * cfg.dt

    vortex_handoff = ol_controls[-1][[I_XV, I_YV]] if open_loop_steps > 0 else A_remote_xy.copy()

    d_AB_post_ol = float(np.linalg.norm(pos[idx_A] - pos[idx_B]) * 1e6)
    print(f"    Completed: {open_loop_steps} steps ({open_loop_time*1e3:.1f} ms)")
    print(f"    After carry: d(A,B) = {d_AB_post_ol:.1f} µm")

    # ════════════════════════════════════════════════════════════════
    #  Phase II — MPC-controlled approach
    # ════════════════════════════════════════════════════════════════
    u0_mpc = np.array([
        0.0,                         # ψ: start at 0, optimizer is free to explore
        vortex_handoff[0],
        vortex_handoff[1],
        args.alpha_init,
        1.0,                         # β = 1.0 during MPC (SW visible)
    ])
    u0_mpc = np.clip(u0_mpc, cfg.u_lo, cfg.u_hi)

    print(f"\n  Phase II — MPC approach (T={cfg.T}, K={cfg.K})...")
    mpc_result = run_mpc(
        feval=feval,
        x0=pos.copy(),
        u_init=u0_mpc,
        idx_A=idx_A,
        idx_B=idx_B,
        neigh_idx=neigh_idx,
        target_pos=traps_m.copy(),
        cfg=cfg,
        verbose=True,
    )

    # ════════════════════════════════════════════════════════════════
    #  Phase III — Settling (α→0 ramp, SW recapture)
    # ════════════════════════════════════════════════════════════════
    if args.T_settle > 0:
        mpc_result = run_settling_phase(
            feval, mpc_result, cfg, idx_A, idx_B, T_settle=args.T_settle,
        )

    # ── Merge open-loop + MPC results into a single MPCResult ─────
    all_positions = ol_positions[:-1] + mpc_result.positions  # avoid duplicate handoff frame
    all_controls = ol_controls + mpc_result.applied_controls
    all_forces = ol_forces + mpc_result.forces
    # Offset merge time to account for open-loop duration
    ol_duration = open_loop_steps * cfg.dt
    merge_time = None
    if mpc_result.merge_time_s is not None:
        merge_time = mpc_result.merge_time_s + ol_duration
    else:
        # Check if merge happened during open-loop
        for k in range(len(ol_positions)):
            d = float(np.linalg.norm(ol_positions[k][idx_A] - ol_positions[k][idx_B]))
            if d < CAPTURE_RADIUS:
                merge_time = k * cfg.dt
                break

    combined_result = MPCResult(
        positions=all_positions,
        applied_controls=all_controls,
        forces=all_forces,
        J_history=mpc_result.J_history,
        inner_J_histories=mpc_result.inner_J_histories,
        elapsed_s=mpc_result.elapsed_s,
        merge_time_s=merge_time,
    )

    # ── Metrics ───────────────────────────────────────────────────
    transport = mpc_result_to_transport(combined_result, traps_m, cfg.dt)
    metrics = compute_metrics(transport, idx_A, idx_B, neigh_idx)
    metrics["open_loop_steps"] = open_loop_steps
    metrics["open_loop_duration_ms"] = round(ol_duration * 1e3, 1)
    print(f"\nTransport metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # ── Output directory ──────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "results" / "dev" / "mpc_vortex_merge" / f"run_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Save outputs ──────────────────────────────────────────────
    save_controls_csv(combined_result, cfg.dt, out_dir / "controls.csv")
    save_trajectory_csv(combined_result, idx_A, idx_B, cfg.dt, out_dir / "trajectory.csv")
    plot_convergence(mpc_result, out_dir / "convergence.png")

    render_gif(
        combined_result, p_sw, vortex_gen, xg, yg, traps_m,
        idx_A, idx_B, neigh_idx,
        out_dir / "mpc_vortex_merge.gif",
        r_barrier=r_barrier,
        view_xlim=view_xlim,
        view_ylim=view_ylim,
        n_frames=args.n_gif_frames,
        open_loop_steps=open_loop_steps,
        T_mpc=cfg.T,
        carry_frames=carry_frames,
    )

    # ── Manifest ──────────────────────────────────────────────────
    manifest = {
        "method_id": "mpc_vortex_merge_v2",
        "script": "scripts/dev/mpc_vortex_merge.py",
        "timestamp": timestamp,
        "output_dir": str(out_dir.relative_to(PROJECT_ROOT)),
        "mpc_config": {
            "K": cfg.K,
            "T": cfg.T,
            "dt": cfg.dt,
            "replan_every": cfg.replan_every,
            "n_iters": cfg.n_iters,
            "w_merge": cfg.w_merge,
            "w_B_stable": cfg.w_B_stable,
            "w_neigh": cfg.w_neigh,
            "w_smooth": cfg.w_smooth,
            "w_smooth_vec": cfg.w_smooth_vec.tolist() if cfg.w_smooth_vec is not None else None,
            "w_beta_high": cfg.w_beta_high,
            "beta_terminal": cfg.beta_terminal,
            "u_lo": cfg.u_lo.tolist(),
            "u_hi": cfg.u_hi.tolist(),
            "du_max": cfg.du_max.tolist() if cfg.du_max is not None else None,
            "T_settle": args.T_settle,
        },
        "vortex_design": {
            "charge": args.charge,
            "family": vortex_family,
            "bessel_aperture_mm": args.bessel_aperture if args.vortex_source == "bessel" else None,
            "bessel_prop_dist_mm": args.bessel_prop_dist if args.vortex_source == "bessel" else None,
            "focus_mm": args.focus_mm if args.vortex_source == "bessel" else None,
            "waist_mm": args.waist if args.vortex_source == "lg" else None,
        },
        "scenario": {
            "data_source": "FIELD_NPZ",
            "mpc_activation_radius_lam": args.mpc_activation_radius,
            "carry_n_vortex_steps": args.n_vortex_steps,
            "carry_n_dyn_steps": args.n_dyn_steps,
            "carry_n_psi": args.n_psi,
            "carry_w_retain": args.w_retain,
            "ol_alpha": args.ol_alpha,
            "ol_beta": args.ol_beta,
            "carry_steps": open_loop_steps,
            "carry_duration_ms": round(ol_duration * 1e3, 1),
            "r_barrier_um": round(r_barrier * 1e6, 1),
            "vxy_rate_um_per_step": args.vxy_rate,
            "psi_rate_rad_per_step": args.psi_rate,
        },
        "particles": {
            "n_total": len(traps_m_all),
            "idx_A": idx_A,
            "idx_B": idx_B,
            "n_neighbours": len(neigh_idx),
            "d_AB_initial_um": round(d_AB_init * 1e6, 2),
        },
        "metrics": metrics,
        "performance": {
            "elapsed_s": round(combined_result.elapsed_s, 1),
            "n_replans": len(combined_result.J_history),
        },
    }

    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    print(f"\nAll outputs saved to: {out_dir}")
    print(f"Classification: {metrics.get('classification', 'unknown')}")


if __name__ == "__main__":
    main()
