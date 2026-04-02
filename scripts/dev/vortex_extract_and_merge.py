#!/usr/bin/env python3
"""
Vortex Extract and Merge — three-stage open-loop manipulation.

Story
-----
  Stage 1 — CLEAR (one neighbour at a time)
    B sits at the centre trap surrounded by its N nearest neighbours.
    The vortex ring captures each neighbour and carries it radially
    outward away from B.  B remains in its SW trap throughout.  Phase
    is swept to maximise the inward restoring force on the active
    neighbour (pure-retention objective, no pull target).

  Stage 2 — CARRY
    The vortex moves to particle A (top-left of the lattice), captures
    it inside the ring, and carries it toward B.  Uses the validated
    phase-sweep + soft-retain objective:
        score = w_barrier*ΔU − w_pull*F_in_B + w_lateral*F_perp_B − w_retain*F_A_in
    Canonical working config (2026-04-02): α=2.0, w_retain=1.0.

  Stage 3 — MERGE  (built into the final steps of CARRY)
    Phase-tuning opens the ring toward B; B slips inside alongside A.

Outputs
-------
    results/dev/vortex_extract_and_merge/<timestamp>/
        vortex_extract_and_merge.gif
        stage_log.csv
        manifest.json

Usage
-----
    python scripts/dev/vortex_extract_and_merge.py
    python scripts/dev/vortex_extract_and_merge.py --n_neigh 6
    python scripts/dev/vortex_extract_and_merge.py --n_clear_steps 20 --alpha_clear 1.5
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    C_WATER, F_HZ, OMEGA, RHO0,
    default_particle_params, gorkov_grid_2d,
)
from scripts.lib.particle_dynamics_utils import (
    SCALE, DT_DEFAULT, LAM,
    gorkov_normalised,
)
from scripts.lib.asm_utils import (
    propagate_asm,
    LAM as ASM_LAM, K0 as ASM_K0,
)
from scripts.lib.overlay_utils import estimate_ring_radius
from scripts.lib.perturbation_vortex import VortexPerturbation

FIELD_NPZ = (
    PROJECT_ROOT / "results" / "deliverables" / "vortex_stage_transport"
    / "transport" / "transport_case_for_gif.npz"
)

PPAR = default_particle_params()

# ── Vortex beam (canonical working values) ────────────────────────
VORTEX_CHARGE               = 1
BESSEL_APERTURE_MM_DEFAULT  = 3.5
BESSEL_PROP_DIST_MM_DEFAULT = 3.0
FOCUS_MM_DEFAULT            = 3.0

# ── Stage 1 — clearing ────────────────────────────────────────────
N_NEIGH_CLEAR_DEFAULT = 4     # nearest neighbours around B to extract
N_CLEAR_STEPS_DEFAULT = 15    # vortex path steps per neighbour
CLEAR_PUSH_FACTOR     = 3.0   # vortex endpoint = pos_ni + factor*(pos_ni − pos_B)
ALPHA_CLEAR_DEFAULT   = 2.0
W_RETAIN_CLEAR_DEFAULT = 2.0  # strong single-objective retention

# ── Stage 2/3 — carry + merge ─────────────────────────────────────
# Canonical working config (2026-04-02): α=2.0, w_retain=1.0
N_CARRY_STEPS_DEFAULT = 25
ALPHA_CARRY_DEFAULT   = 2.0
W_RETAIN_CARRY_DEFAULT = 1.0

# ── Shared dynamics ───────────────────────────────────────────────
BETA_FIXED           = 1.0
N_PSI_DEFAULT        = 24
N_DYN_STEPS_DEFAULT  = 150
MAX_STEP_DEFAULT     = 2e-6
W_BARRIER_DEFAULT    = 1.0
W_PULL_DEFAULT       = 1.0
W_LATERAL_DEFAULT    = 0.5
N_BARRIER_SAMPLES    = 50
OPENING_THRESHOLD    = 0.05
N_RING_THETA         = 72
OPENING_SECTOR_DEG   = 60.0

# ── Settling ──────────────────────────────────────────────────────
N_SETTLE_VSTEPS   = 20     # vortex-off steps after carry/merge
N_SETTLE_DYN      = 150    # dynamics sub-steps per settle step

# ── GIF rendering ─────────────────────────────────────────────────
N_INTRO_FRAMES    = 12
FRAMES_PER_VSTEP  = 5
GIF_DURATION_MS   = 80
CMAP              = "RdBu_r"
COL_A             = "#e74c3c"   # red   — particle A
COL_B             = "#3498db"   # blue  — particle B
COL_NEIGH_CROWD   = "#8e44ad"   # purple — neighbours not yet cleared
COL_NEIGH_ACTIVE  = "#f39c12"   # orange — being extracted right now
COL_NEIGH_SETTLED = "#7f8c8d"   # grey  — already extracted
COL_CTR           = "#f1c40f"   # yellow — vortex centre
COL_BARRIER       = "#e67e22"   # orange ring
COL_TRAP          = "#95a5a6"   # grey trap markers
PARTICLE_RADIUS_MM = 0.045
VIEW_MARGIN_MM    = 0.8


# ══════════════════════════════════════════════════════════════════
# Focused vortex source  (verbatim from vortex_entry_test.py)
# ══════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════
# Data loading & calibration  (verbatim from vortex_entry_test.py)
# ══════════════════════════════════════════════════════════════════

def load_data(
    aperture_m: float,
    prop_dist_m: float,
    focus_f_m: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float,
           np.ndarray, VortexPerturbation, float]:
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
    p_vortex_raw = propagate_asm(p_source, dx, dy, wavelength=ASM_LAM,
                                  z=prop_dist_m)
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
    r_theory = 0.42 * ASM_LAM * focus_f_m / (2.0 * aperture_m)
    print(f"\n  --- Focused Vortex Calibration ---")
    print(f"  r_barrier = {r_barrier*1e6:.1f} um ({r_barrier/LAM:.2f} lam)  "
          f"[theory: {r_theory*1e6:.1f} um]")

    vortex_gen = VortexPerturbation(p_vortex_raw, xg, yg,
                                     out_xg=xg, out_yg=yg)
    return p_sw, xg, yg, dx, dy, traps_m, vortex_gen, r_barrier


# ══════════════════════════════════════════════════════════════════
# Particle selection  (new)
# ══════════════════════════════════════════════════════════════════

def select_particles(
    traps_m: np.ndarray,
    xg: np.ndarray,
    yg: np.ndarray,
    n_neigh: int,
) -> Tuple[int, int, List[int]]:
    """
    Select:
      B  — nearest trap to grid centre
      A  — top-left corner of lattice (maximise y − x)
      neighbours — n_neigh nearest traps to B (excluding A and B)
    """
    grid_cx = 0.5 * (float(xg[0]) + float(xg[-1]))
    grid_cy = 0.5 * (float(yg[0]) + float(yg[-1]))
    center = np.array([grid_cx, grid_cy])

    dists_to_center = np.linalg.norm(traps_m - center, axis=1)
    idx_B = int(np.argmin(dists_to_center))

    tl_score = traps_m[:, 1] - traps_m[:, 0]
    idx_A = int(np.argmax(tl_score))

    dists_to_B = np.linalg.norm(traps_m - traps_m[idx_B], axis=1)
    excluded = np.zeros(len(traps_m), dtype=bool)
    excluded[idx_B] = True
    excluded[idx_A] = True
    dists_masked = dists_to_B.copy()
    dists_masked[excluded] = np.inf
    idx_neigh = [int(i) for i in np.argsort(dists_masked)[:n_neigh]]

    B = traps_m[idx_B]
    A = traps_m[idx_A]
    print(f"\n  --- Particle Selection ---")
    print(f"  B (centre):   trap {idx_B:3d}  "
          f"({B[0]*1e3:.3f}, {B[1]*1e3:.3f}) mm")
    print(f"  A (top-left): trap {idx_A:3d}  "
          f"({A[0]*1e3:.3f}, {A[1]*1e3:.3f}) mm")
    print(f"  d(A,B) = {np.linalg.norm(A-B)*1e3:.2f} mm "
          f"= {np.linalg.norm(A-B)/LAM:.1f} lam")
    print(f"  Neighbours to clear ({n_neigh}):")
    for ci, ni in enumerate(idx_neigh):
        d = np.linalg.norm(traps_m[ni] - B)
        print(f"    [{ci}] trap {ni:3d}  "
              f"({traps_m[ni,0]*1e3:.3f}, {traps_m[ni,1]*1e3:.3f}) mm  "
              f"d={d*1e6:.0f} um = {d/LAM:.2f} lam")

    return idx_A, idx_B, idx_neigh


# ══════════════════════════════════════════════════════════════════
# Path builders  (new)
# ══════════════════════════════════════════════════════════════════

def build_extraction_path(
    pos_ni: np.ndarray,
    pos_B: np.ndarray,
    n_steps: int,
    push_factor: float,
) -> np.ndarray:
    """
    Straight path for extraction vortex.
    - Starts at pos_ni (vortex centred on the neighbour to be cleared)
    - Ends at pos_ni + push_factor * (pos_ni − pos_B)
      i.e. push_factor trap-spacings beyond pos_ni in the outward direction

    With push_factor=3.0 and trap_spacing≈371 µm the endpoint is
    ~1.5 mm from B (well beyond the ring).
    """
    direction = pos_ni - pos_B
    end = pos_ni + push_factor * direction
    t = np.linspace(0.0, 1.0, n_steps)
    return pos_ni[None, :] + t[:, None] * (end - pos_ni)[None, :]


def build_carry_path(
    start: np.ndarray,
    end: np.ndarray,
    n_steps: int,
) -> np.ndarray:
    """Straight vortex path from start to end."""
    t = np.linspace(0.0, 1.0, n_steps)
    return start[None, :] + t[:, None] * (end - start)[None, :]


# ══════════════════════════════════════════════════════════════════
# Field helpers  (verbatim from vortex_entry_test.py)
# ══════════════════════════════════════════════════════════════════

def total_pressure(p_sw, vortex_gen, psi, alpha, beta, center_xy):
    p_v = vortex_gen.get_field(center_xy)
    return beta * p_sw + alpha * np.exp(1j * psi) * p_v


def _make_interp(F, xg, yg):
    return RegularGridInterpolator(
        (yg, xg), F, bounds_error=False, fill_value=0.0)


def _eval_at(interp, pos_xy: np.ndarray) -> np.ndarray:
    pts = np.column_stack([pos_xy[:, 1], pos_xy[:, 0]])
    return interp(pts)


def _b_status(pos_B, vortex_center, r_barrier):
    d = float(np.linalg.norm(pos_B - vortex_center))
    if d > r_barrier * 1.05:
        return "OUTSIDE"
    if d < r_barrier * 0.95:
        return "INSIDE"
    return "ON"


# ══════════════════════════════════════════════════════════════════
# Ring barrier test  (verbatim from vortex_entry_test.py)
# ══════════════════════════════════════════════════════════════════

def ring_barrier_test(
    iU: RegularGridInterpolator,
    vortex_center: np.ndarray,
    pos_B: np.ndarray,
    r_barrier: float,
    n_theta: int = N_RING_THETA,
    sector_half_deg: float = OPENING_SECTOR_DEG,
    opening_threshold: float = OPENING_THRESHOLD,
) -> Dict:
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

    U_B_sector = (float(np.mean(U_ring[in_sector]))
                  if np.any(in_sector) else float("nan"))
    U_other = (float(np.mean(U_ring[~in_sector]))
               if np.any(~in_sector) else float("nan"))

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
        "U_ring_min": float(np.min(U_ring)),
        "U_ring_max": float(np.max(U_ring)),
    }


# ══════════════════════════════════════════════════════════════════
# Phase sweep — clearing  (new)
# Objective: maximise inward restoring force on the active neighbour.
# No B-pull, no barrier-line calculation.
# ══════════════════════════════════════════════════════════════════

def phase_sweep_retain(
    p_sw,
    vortex_gen,
    xg: np.ndarray,
    yg: np.ndarray,
    dx: float,
    dy: float,
    psi_values: np.ndarray,
    alpha: float,
    vortex_center: np.ndarray,
    pos_carry: np.ndarray,
    w_retain: float = 2.0,
) -> Dict:
    """
    Find ψ that maximises the restoring force pulling pos_carry toward
    the vortex centre: score = −w_retain · F_carry_in.
    Returns best dict with keys: psi, F_carry_in, score, iFx, iFy.
    """
    e_carry = vortex_center - pos_carry
    norm = np.linalg.norm(e_carry)
    e_carry = e_carry / norm if norm > 1e-12 else np.array([1.0, 0.0])

    results = []
    for psi in psi_values:
        p_tot = total_pressure(p_sw, vortex_gen, psi, alpha, BETA_FIXED,
                               vortex_center)
        _, Fx_n, Fy_n = gorkov_normalised(p_tot, dx, dy)
        iFx = _make_interp(Fx_n, xg, yg)
        iFy = _make_interp(Fy_n, xg, yg)
        F = np.array([float(_eval_at(iFx, pos_carry[None, :])[0]),
                      float(_eval_at(iFy, pos_carry[None, :])[0])])
        F_carry_in = float(np.dot(F, e_carry))
        results.append({
            "psi": psi,
            "F_carry_in": F_carry_in,
            "score": -w_retain * F_carry_in,
            "iFx": iFx,
            "iFy": iFy,
        })
    return min(results, key=lambda r: r["score"])


# ══════════════════════════════════════════════════════════════════
# Phase sweep — carry/merge  (verbatim from vortex_entry_test.py)
# Canonical config: α=2.0, w_retain=1.0  (2026-04-02)
# score = w_barrier*ΔU − w_pull*F_in_B + w_lateral*F_perp_B − w_retain*F_A_in
# ══════════════════════════════════════════════════════════════════

def phase_sweep(
    p_sw, vortex_gen, xg, yg, dx, dy,
    psi_values, alpha, beta,
    vortex_center, pos_A, pos_B, r_barrier,
    w_barrier=1.0, w_pull=1.0, w_lateral=0.5, w_retain=1.0,
    opening_threshold=OPENING_THRESHOLD,
) -> Dict:
    e_B = vortex_center - pos_B
    norm_B = np.linalg.norm(e_B)
    e_B = e_B / norm_B if norm_B > 0 else np.array([0.0, 1.0])

    e_A = vortex_center - pos_A
    norm_A = np.linalg.norm(e_A)
    e_A = e_A / norm_A if norm_A > 0 else np.array([0.0, 1.0])

    t_samples = np.linspace(0.05, 0.95, N_BARRIER_SAMPLES)
    line_pts = (pos_B[None, :]
                + t_samples[:, None] * (vortex_center - pos_B)[None, :])

    results = []
    for psi in psi_values:
        p_tot = total_pressure(p_sw, vortex_gen, psi, alpha, beta,
                               vortex_center)
        U_n, Fx_n, Fy_n = gorkov_normalised(p_tot, dx, dy)

        iU  = _make_interp(U_n,  xg, yg)
        iFx = _make_interp(Fx_n, xg, yg)
        iFy = _make_interp(Fy_n, xg, yg)

        FA = np.array([float(_eval_at(iFx, pos_A[None, :])[0]),
                       float(_eval_at(iFy, pos_A[None, :])[0])])
        F_A_in = float(np.dot(FA, e_A))

        FB = np.array([float(_eval_at(iFx, pos_B[None, :])[0]),
                       float(_eval_at(iFy, pos_B[None, :])[0])])
        F_in_B   = float(np.dot(FB, e_B))
        F_tang_B = FB - F_in_B * e_B
        F_perp_B = float(np.linalg.norm(F_tang_B))

        U_path = _eval_at(iU, line_pts)
        U_B    = float(_eval_at(iU, pos_B[None, :])[0])
        DeltaU = float(np.max(U_path) - U_B)

        score = (w_barrier * DeltaU
                 - w_pull * F_in_B
                 + w_lateral * F_perp_B
                 - w_retain * F_A_in)

        results.append({
            "psi": psi, "F_A_in": F_A_in, "F_in_B": F_in_B,
            "F_perp_B": F_perp_B, "DeltaU": DeltaU, "score": score,
            "iFx": iFx, "iFy": iFy, "iU": iU,
        })

    best = min(results, key=lambda r: r["score"])
    best["mechanism"] = ring_barrier_test(
        best["iU"], vortex_center, pos_B, r_barrier,
        opening_threshold=opening_threshold)
    return best


# ══════════════════════════════════════════════════════════════════
# View computation  (new)
# ══════════════════════════════════════════════════════════════════

def _compute_full_view(
    traps_m: np.ndarray,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Single bounding box covering the entire trap lattice + margin."""
    pts_mm = traps_m * 1e3
    return (
        (float(pts_mm[:, 0].min()) - VIEW_MARGIN_MM,
         float(pts_mm[:, 0].max()) + VIEW_MARGIN_MM),
        (float(pts_mm[:, 1].min()) - VIEW_MARGIN_MM,
         float(pts_mm[:, 1].max()) + VIEW_MARGIN_MM),
    )


# ══════════════════════════════════════════════════════════════════
# GIF rendering  (new — multi-particle, single fixed view)
# ══════════════════════════════════════════════════════════════════

def _render_frame(
    ax,
    fd: Dict,
    xg: np.ndarray,
    yg: np.ndarray,
    dx: float,
    dy: float,
    p_sw: np.ndarray,
    vortex_gen,
    traps_m: np.ndarray,
    idx_A: int,
    idx_B: int,
    idx_neigh: List[int],
    r_barrier_mm: float,
    extent: List[float],
    view_xlim: Tuple[float, float],
    view_ylim: Tuple[float, float],
    frame_idx: int,
    n_total: int,
) -> None:
    stage        = fd["stage"]
    psi          = fd["psi"]
    alpha        = fd["alpha"]
    pos_A_mm     = fd["pos_A"] * 1e3
    pos_B_mm     = fd["pos_B"] * 1e3
    pos_neigh_mm = fd["pos_neigh"] * 1e3      # [n, 2] in mm
    vc_mm        = fd["vortex_center"] * 1e3
    trail_mm     = np.array(fd["vortex_trail"]) * 1e3
    cleared_mask = fd["cleared_mask"]          # list[bool]
    active_ci    = fd.get("active_neigh_ci")   # int or None
    d_AB_um      = fd["d_AB"] * 1e6
    b_status     = fd.get("B_status", "")

    # ── Gorkov background ─────────────────────────────────────────
    p_tot = total_pressure(p_sw, vortex_gen, psi, alpha, BETA_FIXED,
                           fd["vortex_center"])
    U_disp, _, _ = gorkov_grid_2d(
        p_tot, dx, dy, OMEGA, RHO0, C_WATER,
        PPAR["a"], PPAR["f1"], PPAR["f2"],
    )
    ix_lo = max(0, int(np.searchsorted(xg, (view_xlim[0] - 0.1) * 1e-3)))
    ix_hi = min(len(xg), int(np.searchsorted(xg, (view_xlim[1] + 0.1) * 1e-3)))
    iy_lo = max(0, int(np.searchsorted(yg, (view_ylim[0] - 0.1) * 1e-3)))
    iy_hi = min(len(yg), int(np.searchsorted(yg, (view_ylim[1] + 0.1) * 1e-3)))
    U_roi = U_disp[iy_lo:iy_hi, ix_lo:ix_hi]
    lo = float(np.percentile(U_roi, 0.5))
    hi = float(np.percentile(U_roi, 99.5))
    ax.imshow(U_disp, origin="lower", extent=extent, cmap=CMAP,
              vmin=lo, vmax=hi, aspect="equal", interpolation="bicubic")

    # ── Trap markers (in-view only) ───────────────────────────────
    traps_mm = traps_m * 1e3
    in_view = (
        (traps_mm[:, 0] >= view_xlim[0]) & (traps_mm[:, 0] <= view_xlim[1])
        & (traps_mm[:, 1] >= view_ylim[0]) & (traps_mm[:, 1] <= view_ylim[1])
    )
    if np.any(in_view):
        ax.scatter(traps_mm[in_view, 0], traps_mm[in_view, 1],
                   marker="x", c=COL_TRAP, s=28, linewidths=0.8,
                   zorder=4, alpha=0.5)

    # ── Vortex ring + trail (suppressed when alpha=0) ─────────────
    if alpha > 0:
        if len(trail_mm) > 1:
            ax.plot(trail_mm[:, 0], trail_mm[:, 1], "-",
                    color=COL_CTR, lw=0.8, alpha=0.6, zorder=5)
        ax.plot(vc_mm[0], vc_mm[1], "+",
                color=COL_CTR, ms=12, mew=2.0, zorder=9)
        ax.add_patch(mpatches.Circle(
            (vc_mm[0], vc_mm[1]), r_barrier_mm,
            fill=False, edgecolor=COL_BARRIER, linestyle="--",
            linewidth=1.2, alpha=0.8, zorder=6))

    # ── Neighbour particles (colour-coded) ────────────────────────
    for ci, ni in enumerate(idx_neigh):
        if cleared_mask[ci]:
            col = COL_NEIGH_SETTLED
        elif active_ci is not None and ci == active_ci:
            col = COL_NEIGH_ACTIVE
        else:
            col = COL_NEIGH_CROWD
        ax.add_patch(mpatches.Circle(
            (pos_neigh_mm[ci, 0], pos_neigh_mm[ci, 1]),
            PARTICLE_RADIUS_MM, facecolor=col, edgecolor="white",
            linewidth=0.6, zorder=8, alpha=0.9))

    # ── A and B ───────────────────────────────────────────────────
    ax.add_patch(mpatches.Circle(
        (pos_A_mm[0], pos_A_mm[1]), PARTICLE_RADIUS_MM,
        facecolor=COL_A, edgecolor="white", linewidth=0.8, zorder=10))
    ax.add_patch(mpatches.Circle(
        (pos_B_mm[0], pos_B_mm[1]), PARTICLE_RADIUS_MM,
        facecolor=COL_B, edgecolor="white", linewidth=0.8, zorder=10))

    ax.annotate("A", (pos_A_mm[0], pos_A_mm[1]), fontsize=7, color=COL_A,
                ha="right", va="bottom",
                xytext=(-4, 3), textcoords="offset points", zorder=11)
    ax.annotate("B", (pos_B_mm[0], pos_B_mm[1]), fontsize=7, color=COL_B,
                ha="left", va="bottom",
                xytext=(3, 3), textcoords="offset points", zorder=11)

    # ── Legend ────────────────────────────────────────────────────
    handles = [
        mpatches.Patch(color=COL_A,            label="A (target)"),
        mpatches.Patch(color=COL_B,            label="B (merge)"),
        mpatches.Patch(color=COL_NEIGH_CROWD,  label="crowd"),
        mpatches.Patch(color=COL_NEIGH_ACTIVE, label="extracting"),
        mpatches.Patch(color=COL_NEIGH_SETTLED, label="cleared"),
    ]
    ax.legend(handles=handles, loc="upper right",
              fontsize=7, framealpha=0.75)

    # ── Axes / title ──────────────────────────────────────────────
    ax.set_xlim(*view_xlim)
    ax.set_ylim(*view_ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    b_str = f" | B:{b_status}" if b_status else ""
    ax.set_title(
        f"[{stage}]  α={alpha:.1f}  ψ={psi:.2f}  "
        f"d(A,B)={d_AB_um:.0f} µm{b_str}\n"
        f"frame {frame_idx + 1}/{n_total}",
        fontsize=9)


def render_gif(
    frames_data: List[Dict],
    xg: np.ndarray,
    yg: np.ndarray,
    dx: float,
    dy: float,
    p_sw: np.ndarray,
    vortex_gen,
    traps_m: np.ndarray,
    idx_A: int,
    idx_B: int,
    idx_neigh: List[int],
    r_barrier_mm: float,
    view_xlim: Tuple[float, float],
    view_ylim: Tuple[float, float],
    out_path: Path,
    duration_ms: int = GIF_DURATION_MS,
) -> None:
    x_mm   = xg * 1e3
    y_mm   = yg * 1e3
    extent = [x_mm[0], x_mm[-1], y_mm[0], y_mm[-1]]
    n_total = len(frames_data)
    frames: List[Image.Image] = []
    print(f"  Rendering {n_total} GIF frames...")

    for fi, fd in enumerate(frames_data):
        fig, ax = plt.subplots(1, 1, figsize=(6.2, 6.2))
        _render_frame(
            ax, fd, xg, yg, dx, dy,
            p_sw, vortex_gen, traps_m,
            idx_A, idx_B, idx_neigh,
            r_barrier_mm, extent, view_xlim, view_ylim, fi, n_total,
        )
        fig.tight_layout()
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        frames.append(Image.fromarray(buf.reshape(h, w, 4)).convert("RGB"))
        plt.close(fig)

    frames[0].save(
        str(out_path), save_all=True, append_images=frames[1:],
        duration=duration_ms, loop=0)
    print(f"  Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════
# Main experiment
# ══════════════════════════════════════════════════════════════════

def run_experiment(args) -> None:
    t0_total = time.time()

    aperture_m  = args.bessel_aperture_mm * 1e-3
    prop_dist_m = args.bessel_prop_dist_mm * 1e-3
    focus_f_m   = args.focus_mm * 1e-3

    # ── Load & calibrate ──────────────────────────────────────────
    (p_sw, xg, yg, dx, dy, traps_m,
     vortex_gen, r_barrier) = load_data(aperture_m, prop_dist_m, focus_f_m)
    r_barrier_mm = r_barrier * 1e3

    # ── Select particles ──────────────────────────────────────────
    idx_A, idx_B, idx_neigh = select_particles(
        traps_m, xg, yg, args.n_neigh)

    # ── View ──────────────────────────────────────────────────────
    view_xlim, view_ylim = _compute_full_view(traps_m)
    print(f"\n  View: x={view_xlim}, y={view_ylim} mm")

    # ── Output directory ──────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (PROJECT_ROOT / "results" / "dev"
               / "vortex_extract_and_merge" / timestamp)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {out_dir}")

    psi_values = np.linspace(0, 2 * np.pi, args.n_psi, endpoint=False)
    snap_indices = set(
        np.linspace(0, args.n_dyn_steps - 1, FRAMES_PER_VSTEP,
                    dtype=int).tolist()
    )

    # ── Live particle state ───────────────────────────────────────
    pos_A     = traps_m[idx_A].copy()
    pos_B     = traps_m[idx_B].copy()
    pos_neigh = traps_m[idx_neigh].copy()   # [n_neigh, 2]
    cleared   = [False] * args.n_neigh

    frames_data: List[Dict] = []
    csv_rows:    List[List] = []

    # ── Intro frames (α=0, standing wave only) ────────────────────
    intro_vc = traps_m[idx_B].copy()
    for _ in range(N_INTRO_FRAMES):
        frames_data.append({
            "stage": "INTRO",
            "psi": 0.0, "alpha": 0.0,
            "pos_A": pos_A.copy(),
            "pos_B": pos_B.copy(),
            "pos_neigh": pos_neigh.copy(),
            "vortex_center": intro_vc.copy(),
            "vortex_trail": [intro_vc.copy()],
            "cleared_mask": list(cleared),
            "active_neigh_ci": None,
            "d_AB": float(np.linalg.norm(pos_A - pos_B)),
            "B_status": "",
        })

    # ══ Stage 1: CLEAR ════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  STAGE 1 — CLEAR ({args.n_neigh} neighbours, "
          f"α={args.alpha_clear:.1f}, w_retain={args.w_retain_clear:.1f})")
    print(f"{'='*60}")

    for ci in range(args.n_neigh):
        ni = idx_neigh[ci]
        extr_path = build_extraction_path(
            pos_neigh[ci], pos_B, args.n_clear_steps, args.clear_push_factor)
        vortex_trail: List[np.ndarray] = []

        print(f"\n  Clearing neighbour {ci + 1}/{args.n_neigh} "
              f"(trap {ni}):")

        for v_step, vc in enumerate(extr_path):
            vortex_trail.append(vc.copy())
            t_sw = time.time()

            best = phase_sweep_retain(
                p_sw, vortex_gen, xg, yg, dx, dy,
                psi_values, args.alpha_clear, vc,
                pos_neigh[ci], w_retain=args.w_retain_clear)

            psi_best   = float(best["psi"])
            F_carry_in = float(best["F_carry_in"])
            iFx, iFy   = best["iFx"], best["iFy"]
            dt_sw      = time.time() - t_sw

            d_ni_vc = float(np.linalg.norm(pos_neigh[ci] - vc))
            d_ni_B  = float(np.linalg.norm(pos_neigh[ci] - pos_B))
            in_ring = d_ni_vc < r_barrier

            print(f"    v{v_step + 1:02d}/{args.n_clear_steps} | "
                  f"psi={psi_best:.2f} | F_in={F_carry_in:+.3e} | "
                  f"Ni:{'IN ' if in_ring else 'OUT'} | "
                  f"d(Ni,B)={d_ni_B*1e6:.0f} um | {dt_sw*1e3:.0f}ms")

            # ── Dynamics: update ALL particles ─────────────────────
            for dyn_i in range(args.n_dyn_steps):
                # A
                F_a = np.array([
                    float(_eval_at(iFx, pos_A[None, :])[0]),
                    float(_eval_at(iFy, pos_A[None, :])[0])])
                pos_A = pos_A + np.clip(
                    SCALE * DT_DEFAULT * F_a, -args.max_step, args.max_step)
                pos_A[0] = np.clip(pos_A[0], xg[2], xg[-3])
                pos_A[1] = np.clip(pos_A[1], yg[2], yg[-3])

                # B
                F_b = np.array([
                    float(_eval_at(iFx, pos_B[None, :])[0]),
                    float(_eval_at(iFy, pos_B[None, :])[0])])
                pos_B = pos_B + np.clip(
                    SCALE * DT_DEFAULT * F_b, -args.max_step, args.max_step)
                pos_B[0] = np.clip(pos_B[0], xg[2], xg[-3])
                pos_B[1] = np.clip(pos_B[1], yg[2], yg[-3])

                # All neighbours
                for nj in range(len(idx_neigh)):
                    F_nj = np.array([
                        float(_eval_at(iFx, pos_neigh[nj:nj + 1])[0]),
                        float(_eval_at(iFy, pos_neigh[nj:nj + 1])[0])])
                    pos_neigh[nj] = pos_neigh[nj] + np.clip(
                        SCALE * DT_DEFAULT * F_nj, -args.max_step, args.max_step)
                    pos_neigh[nj, 0] = np.clip(pos_neigh[nj, 0], xg[2], xg[-3])
                    pos_neigh[nj, 1] = np.clip(pos_neigh[nj, 1], yg[2], yg[-3])

                if dyn_i in snap_indices:
                    b_stat = _b_status(pos_B, vc, r_barrier)
                    frames_data.append({
                        "stage": f"CLEAR{ci + 1}",
                        "psi": psi_best, "alpha": args.alpha_clear,
                        "pos_A": pos_A.copy(),
                        "pos_B": pos_B.copy(),
                        "pos_neigh": pos_neigh.copy(),
                        "vortex_center": vc.copy(),
                        "vortex_trail": [v.copy() for v in vortex_trail],
                        "cleared_mask": [j < ci for j in range(args.n_neigh)],
                        "active_neigh_ci": ci,
                        "d_AB": float(np.linalg.norm(pos_A - pos_B)),
                        "B_status": b_stat,
                    })

            csv_rows.append([f"CLEAR{ci + 1}", v_step + 1, psi_best,
                             args.alpha_clear, "", "",
                             False, F_carry_in, ""])

        cleared[ci] = True
        d_final = float(np.linalg.norm(pos_neigh[ci] - pos_B))
        print(f"  ✓ Neighbour {ci + 1} cleared — "
              f"d(Ni,B) = {d_final*1e3:.3f} mm = {d_final/LAM:.1f} lam")

        # 3 pause frames with vortex off between neighbours
        psi_last = float(extr_path[-1][0]) if len(extr_path) > 0 else 0.0
        for _ in range(3):
            frames_data.append({
                "stage": f"CLEAR{ci + 1}",
                "psi": 0.0, "alpha": 0.0,
                "pos_A": pos_A.copy(),
                "pos_B": pos_B.copy(),
                "pos_neigh": pos_neigh.copy(),
                "vortex_center": extr_path[-1].copy(),
                "vortex_trail": [],
                "cleared_mask": [j <= ci for j in range(args.n_neigh)],
                "active_neigh_ci": None,
                "d_AB": float(np.linalg.norm(pos_A - pos_B)),
                "B_status": "",
            })

    # ══ Stage 2+3: CARRY + MERGE ══════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  STAGE 2/3 — CARRY A → B + MERGE")
    print(f"  α={args.alpha_carry:.1f}  w_retain={args.w_retain_carry:.1f}  "
          f"n_carry_steps={args.n_carry_steps}")
    print(f"{'='*60}")

    # Carry path: straight from A's current position to B's current position
    carry_path = build_carry_path(
        pos_A.copy(), pos_B.copy(), args.n_carry_steps)
    vortex_trail = []

    for v_step, vc in enumerate(carry_path):
        vortex_trail.append(vc.copy())
        t_sw = time.time()

        best = phase_sweep(
            p_sw, vortex_gen, xg, yg, dx, dy,
            psi_values, args.alpha_carry, BETA_FIXED,
            vc, pos_A, pos_B, r_barrier,
            w_barrier=args.w_barrier,
            w_pull=args.w_pull,
            w_lateral=args.w_lateral,
            w_retain=args.w_retain_carry,
        )
        dt_sw    = time.time() - t_sw
        psi_best = float(best["psi"])
        F_A_in   = float(best["F_A_in"])
        F_in_B   = float(best["F_in_B"])
        mech_info = best["mechanism"]
        mech_str  = mech_info["mechanism"]
        iFx, iFy  = best["iFx"], best["iFy"]

        d_A_vc   = float(np.linalg.norm(pos_A - vc))
        A_in     = d_A_vc < r_barrier
        b_stat   = _b_status(pos_B, vc, r_barrier)
        stage_lbl = "MERGE" if b_stat == "INSIDE" else "CARRY"

        print(f"  [{stage_lbl}] v{v_step + 1:02d}/{args.n_carry_steps} | "
              f"α={args.alpha_carry:.1f} | psi={psi_best:.2f} | "
              f"F_A_in={F_A_in:+.3e} | F_in_B={F_in_B:+.3e} | "
              f"A:{'IN ' if A_in else 'OUT'} | B:{b_stat} | "
              f"{mech_str[:5]} | {dt_sw*1e3:.0f}ms")

        # ── Dynamics: update ALL particles ─────────────────────────
        for dyn_i in range(args.n_dyn_steps):
            F_A = np.array([float(_eval_at(iFx, pos_A[None, :])[0]),
                            float(_eval_at(iFy, pos_A[None, :])[0])])
            F_B = np.array([float(_eval_at(iFx, pos_B[None, :])[0]),
                            float(_eval_at(iFy, pos_B[None, :])[0])])

            pos_A = pos_A + np.clip(
                SCALE * DT_DEFAULT * F_A, -args.max_step, args.max_step)
            pos_B = pos_B + np.clip(
                SCALE * DT_DEFAULT * F_B, -args.max_step, args.max_step)

            pos_A[0] = np.clip(pos_A[0], xg[2], xg[-3])
            pos_A[1] = np.clip(pos_A[1], yg[2], yg[-3])
            pos_B[0] = np.clip(pos_B[0], xg[2], xg[-3])
            pos_B[1] = np.clip(pos_B[1], yg[2], yg[-3])

            for nj in range(len(idx_neigh)):
                F_nj = np.array([
                    float(_eval_at(iFx, pos_neigh[nj:nj + 1])[0]),
                    float(_eval_at(iFy, pos_neigh[nj:nj + 1])[0])])
                pos_neigh[nj] = pos_neigh[nj] + np.clip(
                    SCALE * DT_DEFAULT * F_nj, -args.max_step, args.max_step)
                pos_neigh[nj, 0] = np.clip(pos_neigh[nj, 0], xg[2], xg[-3])
                pos_neigh[nj, 1] = np.clip(pos_neigh[nj, 1], yg[2], yg[-3])

            if dyn_i in snap_indices:
                b_s   = _b_status(pos_B, vc, r_barrier)
                sl    = "MERGE" if b_s == "INSIDE" else "CARRY"
                frames_data.append({
                    "stage": sl,
                    "psi": psi_best, "alpha": args.alpha_carry,
                    "pos_A": pos_A.copy(),
                    "pos_B": pos_B.copy(),
                    "pos_neigh": pos_neigh.copy(),
                    "vortex_center": vc.copy(),
                    "vortex_trail": [v.copy() for v in vortex_trail],
                    "cleared_mask": [True] * args.n_neigh,
                    "active_neigh_ci": None,
                    "d_AB": float(np.linalg.norm(pos_A - pos_B)),
                    "B_status": b_s,
                })

        d_AB = float(np.linalg.norm(pos_A - pos_B))
        csv_rows.append([stage_lbl, v_step + 1, psi_best,
                         args.alpha_carry, d_AB * 1e6, b_stat,
                         A_in, F_A_in, F_in_B])

    # ══ Stage 4: SETTLE (pure standing wave, vortex off) ═════════
    print(f"\n{'='*60}")
    print(f"  STAGE 4 — SETTLE ({N_SETTLE_VSTEPS} steps, α=0)")
    print(f"{'='*60}")

    # Compute standing-wave-only forces once
    p_sw_only = total_pressure(p_sw, vortex_gen, 0.0, 0.0, BETA_FIXED,
                               pos_B)  # alpha=0 → pure SW
    _, Fx_settle, Fy_settle = gorkov_normalised(p_sw_only, dx, dy)
    iFx_settle = _make_interp(Fx_settle, xg, yg)
    iFy_settle = _make_interp(Fy_settle, xg, yg)

    settle_snap = set(
        np.linspace(0, N_SETTLE_DYN - 1, FRAMES_PER_VSTEP,
                    dtype=int).tolist())

    last_vc = carry_path[-1].copy() if len(carry_path) > 0 else pos_B.copy()

    for s_step in range(N_SETTLE_VSTEPS):
        for dyn_i in range(N_SETTLE_DYN):
            F_A = np.array([float(_eval_at(iFx_settle, pos_A[None, :])[0]),
                            float(_eval_at(iFy_settle, pos_A[None, :])[0])])
            F_B = np.array([float(_eval_at(iFx_settle, pos_B[None, :])[0]),
                            float(_eval_at(iFy_settle, pos_B[None, :])[0])])
            pos_A = pos_A + np.clip(
                SCALE * DT_DEFAULT * F_A, -args.max_step, args.max_step)
            pos_B = pos_B + np.clip(
                SCALE * DT_DEFAULT * F_B, -args.max_step, args.max_step)
            pos_A[0] = np.clip(pos_A[0], xg[2], xg[-3])
            pos_A[1] = np.clip(pos_A[1], yg[2], yg[-3])
            pos_B[0] = np.clip(pos_B[0], xg[2], xg[-3])
            pos_B[1] = np.clip(pos_B[1], yg[2], yg[-3])

            for nj in range(len(idx_neigh)):
                F_nj = np.array([
                    float(_eval_at(iFx_settle, pos_neigh[nj:nj + 1])[0]),
                    float(_eval_at(iFy_settle, pos_neigh[nj:nj + 1])[0])])
                pos_neigh[nj] = pos_neigh[nj] + np.clip(
                    SCALE * DT_DEFAULT * F_nj, -args.max_step, args.max_step)
                pos_neigh[nj, 0] = np.clip(pos_neigh[nj, 0], xg[2], xg[-3])
                pos_neigh[nj, 1] = np.clip(pos_neigh[nj, 1], yg[2], yg[-3])

            if dyn_i in settle_snap:
                frames_data.append({
                    "stage": "SETTLE",
                    "psi": 0.0, "alpha": 0.0,
                    "pos_A": pos_A.copy(),
                    "pos_B": pos_B.copy(),
                    "pos_neigh": pos_neigh.copy(),
                    "vortex_center": last_vc.copy(),
                    "vortex_trail": [],
                    "cleared_mask": [True] * args.n_neigh,
                    "active_neigh_ci": None,
                    "d_AB": float(np.linalg.norm(pos_A - pos_B)),
                    "B_status": "",
                })

        d_AB = float(np.linalg.norm(pos_A - pos_B))
        if (s_step + 1) % 5 == 0 or s_step == N_SETTLE_VSTEPS - 1:
            print(f"  settle {s_step + 1:3d}/{N_SETTLE_VSTEPS} | "
                  f"d(A,B) = {d_AB*1e6:.0f} µm")

    print(f"  Final d(A,B) = {float(np.linalg.norm(pos_A - pos_B))*1e6:.0f} µm")

    # ══ Outputs ═══════════════════════════════════════════════════
    gif_path = out_dir / "vortex_extract_and_merge.gif"
    render_gif(
        frames_data, xg, yg, dx, dy, p_sw, vortex_gen,
        traps_m, idx_A, idx_B, idx_neigh, r_barrier_mm,
        view_xlim, view_ylim, gif_path)

    csv_path = out_dir / "stage_log.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage", "v_step", "psi", "alpha",
                    "d_AB_um", "B_status", "A_in_ring", "F_A_in", "F_in_B"])
        w.writerows(csv_rows)
    print(f"Saved CSV: {csv_path}")

    elapsed = time.time() - t0_total
    manifest = {
        "timestamp":         timestamp,
        "elapsed_s":         round(elapsed, 2),
        "idx_A":             int(idx_A),
        "idx_B":             int(idx_B),
        "idx_neigh":         [int(i) for i in idx_neigh],
        "r_barrier_um":      round(r_barrier * 1e6, 1),
        "n_neigh":           args.n_neigh,
        "n_clear_steps":     args.n_clear_steps,
        "clear_push_factor": args.clear_push_factor,
        "alpha_clear":       args.alpha_clear,
        "w_retain_clear":    args.w_retain_clear,
        "n_carry_steps":     args.n_carry_steps,
        "alpha_carry":       args.alpha_carry,
        "w_retain_carry":    args.w_retain_carry,
        "w_barrier":         args.w_barrier,
        "w_pull":            args.w_pull,
        "w_lateral":         args.w_lateral,
        "n_psi":             args.n_psi,
        "n_dyn_steps":       args.n_dyn_steps,
        "max_step_m":        args.max_step,
        "aperture_mm":       args.bessel_aperture_mm,
        "prop_dist_mm":      args.bessel_prop_dist_mm,
        "focus_mm":          args.focus_mm,
        "beta_fixed":        BETA_FIXED,
        "n_intro_frames":    N_INTRO_FRAMES,
        "frames_per_vstep":  FRAMES_PER_VSTEP,
        "n_traps_total":     len(traps_m),
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone in {elapsed:.1f}s — results in {out_dir}")


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Vortex Extract and Merge — 3-stage GIF demo")

    g = parser.add_argument_group("vortex beam")
    g.add_argument("--bessel_aperture_mm",  type=float,
                   default=BESSEL_APERTURE_MM_DEFAULT)
    g.add_argument("--bessel_prop_dist_mm", type=float,
                   default=BESSEL_PROP_DIST_MM_DEFAULT)
    g.add_argument("--focus_mm",            type=float,
                   default=FOCUS_MM_DEFAULT)

    g = parser.add_argument_group("Stage 1 — clearing")
    g.add_argument("--n_neigh", type=int, default=N_NEIGH_CLEAR_DEFAULT,
                   help="Number of nearest neighbours of B to extract "
                        "(default %(default)s)")
    g.add_argument("--n_clear_steps", type=int, default=N_CLEAR_STEPS_DEFAULT,
                   help="Vortex path steps per neighbour extraction "
                        "(default %(default)s)")
    g.add_argument("--clear_push_factor", type=float, default=CLEAR_PUSH_FACTOR,
                   help="Push endpoint to FACTOR×d(Ni,B) beyond Ni "
                        "(default %(default)s)")
    g.add_argument("--alpha_clear", type=float, default=ALPHA_CLEAR_DEFAULT,
                   help="Vortex amplitude during clearing (default %(default)s)")
    g.add_argument("--w_retain_clear", type=float, default=W_RETAIN_CLEAR_DEFAULT,
                   help="Retention weight during clearing (default %(default)s)")

    g = parser.add_argument_group("Stage 2/3 — carry/merge")
    g.add_argument("--n_carry_steps", type=int, default=N_CARRY_STEPS_DEFAULT,
                   help="Vortex path steps for A→B carry (default %(default)s)")
    g.add_argument("--alpha_carry", type=float, default=ALPHA_CARRY_DEFAULT,
                   help="Vortex amplitude during carry/merge — "
                        "canonical: 2.0 (default %(default)s)")
    g.add_argument("--w_retain_carry", type=float, default=W_RETAIN_CARRY_DEFAULT,
                   help="A-retention weight during carry — "
                        "canonical: 1.0 (default %(default)s)")

    g = parser.add_argument_group("score weights (carry/merge)")
    g.add_argument("--w_barrier", type=float, default=W_BARRIER_DEFAULT)
    g.add_argument("--w_pull",    type=float, default=W_PULL_DEFAULT)
    g.add_argument("--w_lateral", type=float, default=W_LATERAL_DEFAULT)

    g = parser.add_argument_group("dynamics")
    g.add_argument("--n_psi",       type=int,   default=N_PSI_DEFAULT)
    g.add_argument("--n_dyn_steps", type=int,   default=N_DYN_STEPS_DEFAULT)
    g.add_argument("--max_step",    type=float, default=MAX_STEP_DEFAULT,
                   help="Max displacement per dynamics step [m] "
                        "(default %(default)s)")

    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
