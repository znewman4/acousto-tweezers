#!/usr/bin/env python3
"""
Vortex Entry Test — fast, reduced experiment (no MPC) to test whether
phase-tuning can open a vortex ring so particle B can enter the core
without being repelled, while particle A remains trapped inside.

Sweeps phase psi at each vortex position along a straight or curved path
from A's trap toward B's trap. Selects psi that:
  - passes the A-retention constraint (F_A_in > 0)
  - minimises score = w1*DeltaU_entry - w2*F_in + w3*F_perp

Includes:
  - Automatic A-trap selection: searches the full 197-trap lattice for a
    pair satisfying d(B, vortex_start) > 1.2 * r_barrier
  - Vortex calibration: measures barrier ring radius from radial profile
  - Geometry validation: ensures B starts outside the barrier ring
  - Mechanism test: samples U around the barrier ring to distinguish
    true phase-created opening from global basin merging

Outputs:
    results/dev/vortex_entry_test/<timestamp>/
        vortex_entry_alpha_<X>.gif   — animated Gorkov potential
        phase_sweep_results.csv      — per-step metrics
        manifest.json                — config

Usage:
    python scripts/dev/vortex_entry_test.py
    python scripts/dev/vortex_entry_test.py --alphas 2.0 3.0 --n_psi 36
    python scripts/dev/vortex_entry_test.py --focus_mm 2.0 --bessel_prop_dist_mm 2.0
    python scripts/dev/vortex_entry_test.py --bessel_aperture_mm 4.0 --focus_mm 3.0 --bessel_prop_dist_mm 3.0
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

# ── Project path ──────────────────────────────────────────────────
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

# ── Data path (full-domain, 197 traps) ───────────────────────────
FIELD_NPZ = (
    PROJECT_ROOT / "results" / "deliverables" / "vortex_stage_transport"
    / "transport" / "transport_case_for_gif.npz"
)

# ── Vortex design defaults ────────────────────────────────────────
VORTEX_CHARGE               = 1
BESSEL_APERTURE_MM_DEFAULT  = 3.5   # aperture radius [mm]
BESSEL_PROP_DIST_MM_DEFAULT = 3.0   # ASM propagation distance [mm]
FOCUS_MM_DEFAULT            = 3.0   # lens focal length [mm]; set = prop_dist for focal-plane eval

# ── Experiment defaults ───────────────────────────────────────────
N_PSI_DEFAULT        = 24
ALPHAS_DEFAULT       = [1.0, 2.0, 3.0]
BETA_FIXED           = 1.0
N_VORTEX_STEPS       = 20
N_DYN_STEPS_PER_POS  = 150
N_BARRIER_SAMPLES    = 50
W_BARRIER_DEFAULT    = 1.0
W_PULL_DEFAULT       = 1.0
W_LATERAL_DEFAULT    = 0.5
MAX_STEP_DEFAULT     = 2e-6
N_INTRO_FRAMES       = 15
FRAMES_PER_VSTEP     = 6

# Mechanism test
N_RING_THETA         = 72
OPENING_SECTOR_DEG   = 60.0

# Geometry constraint
GEOM_MARGIN          = 1.5   # d(B, vortex_start) > GEOM_MARGIN * r_barrier

# Mechanism test threshold
OPENING_THRESHOLD    = 0.05  # delta_B must exceed this fraction for "true_opening"

# ── Rendering ─────────────────────────────────────────────────────
CMAP               = "RdBu_r"
COL_A              = "#e74c3c"
COL_B              = "#3498db"
COL_CTR            = "#f39c12"
COL_BARRIER        = "#e67e22"
COL_TRAP           = "#95a5a6"
COL_HOME_A         = "#e74c3c"
COL_HOME_B         = "#3498db"
GIF_DURATION_MS    = 80
PARTICLE_RADIUS_MM = 0.045
VIEW_MARGIN_MM     = 0.5

PPAR = default_particle_params()


# ══════════════════════════════════════════════════════════════════
# Focused vortex source generator
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
    """
    Phase-only focused vortex source field.

    Aperture-plane phase:
        phi(r, theta) = charge * theta  -  k * r^2 / (2 * focal_length)

    The first term is the spiral phase (topological charge).
    The second term is a converging thin-lens phase.  Together they
    produce a tighter, more localised vortex at the focal plane
    compared to free-propagation spiral-phase alone.

    The amplitude is uniform inside the aperture (phase-only model),
    directly interpretable as a passive holographic acoustic lens
    that imprints combined spiral + converging-lens phase onto an
    incident plane wave.

    Ring radius at focal plane (first-order vortex, uniform aperture):
        r_ring ≈ 0.42 * lambda * focal_length / (2 * aperture_radius)
    """
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
# Data loading & calibration
# ══════════════════════════════════════════════════════════════════

def load_data(
    aperture_m: float,
    prop_dist_m: float,
    focus_f_m: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float,
           np.ndarray, VortexPerturbation, float]:
    """
    Load full-domain standing wave, full trap lattice, build focused
    vortex, and calibrate barrier radius.

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

    # ── Generate Bessel vortex ────────────────────────────────────
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

    # ── Calibrate: measure barrier ring radius ────────────────────
    mag = np.abs(p_vortex_raw)
    center_xy = np.array([grid_cx, grid_cy])
    ring_info = estimate_ring_radius(
        mag, xg, yg, center=center_xy,
        r_min=0.05e-3, r_max=1.0e-3, n_bins=300,
    )
    r_barrier = float(ring_info["ring_radius_m"])
    ring_peak = float(ring_info["ring_peak"])

    r_theory = 0.42 * ASM_LAM * focus_f_m / (2.0 * aperture_m)
    print(f"\n  --- Focused Vortex Calibration ---")
    print(f"  Aperture radius:      {aperture_m*1e3:.2f} mm")
    print(f"  Focal length:         {focus_f_m*1e3:.2f} mm")
    print(f"  Propagation distance: {prop_dist_m*1e3:.2f} mm  "
          f"({'at focus' if abs(prop_dist_m - focus_f_m) < 0.05e-3 else 'off-focus'})")
    print(f"  Pressure ring radius: {r_barrier*1e6:.1f} um "
          f"({r_barrier/LAM:.2f} lam)")
    print(f"  Ring peak amplitude:  {ring_peak:.4f}")
    print(f"  Theory (focal-plane): {r_theory*1e6:.1f} um  "
          f"[0.42*lam*f/(2R)]")

    vortex_gen = VortexPerturbation(p_vortex_raw, xg, yg,
                                     out_xg=xg, out_yg=yg)
    return p_sw, xg, yg, dx, dy, traps_m, vortex_gen, r_barrier


# ══════════════════════════════════════════════════════════════════
# Trap pair selection
# ══════════════════════════════════════════════════════════════════

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

    The vortex starts centred on A, so the constraint is:
        d(B, A) > min_sep_factor * r_barrier

    Among all valid A candidates, the farthest from B is picked to
    maximise initial clearance — B is then well outside the barrier,
    not marginally so.

    Raises SystemExit if no valid A trap exists.
    """
    grid_cx = 0.5 * (float(xg[0]) + float(xg[-1]))
    grid_cy = 0.5 * (float(yg[0]) + float(yg[-1]))
    center = np.array([grid_cx, grid_cy])

    # ── Pick B: nearest trap to grid centre ───────────────────────
    if idx_B_override is not None:
        idx_B = idx_B_override
    else:
        dists_to_center = np.linalg.norm(traps_m - center, axis=1)
        idx_B = int(np.argmin(dists_to_center))

    B_xy = traps_m[idx_B]
    threshold = min_sep_factor * r_barrier

    # ── Search for valid A ────────────────────────────────────────
    dists_to_B = np.linalg.norm(traps_m - B_xy, axis=1)
    valid_mask = dists_to_B > threshold
    valid_mask[idx_B] = False

    print(f"\n  --- Trap Pair Selection ---")
    print(f"  B: trap [{idx_B}] at "
          f"({B_xy[0]*1e3:.3f}, {B_xy[1]*1e3:.3f}) mm")
    print(f"  Geometry constraint: d(A, B) > {min_sep_factor:.1f} * "
          f"{r_barrier*1e6:.1f} um = {threshold*1e6:.1f} um")
    print(f"  Candidates: {int(valid_mask.sum())} of "
          f"{len(traps_m)} traps pass constraint")

    if not np.any(valid_mask):
        print(f"\n  FATAL: No trap satisfies d(A, B) > {threshold*1e6:.1f} um.")
        print(f"  Closest trap to B is [{int(np.argsort(dists_to_B)[1])}] "
              f"at {sorted(dists_to_B)[1]*1e6:.1f} um.")
        print(f"  Options:")
        print(f"    - Decrease --focus_mm (tighter ring → smaller r_barrier)")
        print(f"    - Increase --bessel_aperture_mm (larger NA → tighter ring)")
        print(f"    - Decrease --min_sep_factor (relax geometry constraint)")
        sys.exit(1)

    # Among valid, pick farthest from B (maximises initial clearance)
    dists_valid = dists_to_B.copy()
    dists_valid[~valid_mask] = -np.inf
    idx_A = int(np.argmax(dists_valid))
    A_xy = traps_m[idx_A]
    d_AB = float(dists_to_B[idx_A])

    print(f"  A: trap [{idx_A}] at "
          f"({A_xy[0]*1e3:.3f}, {A_xy[1]*1e3:.3f}) mm  (farthest valid)")
    print(f"  d(A, B) = {d_AB*1e6:.1f} um = {d_AB/LAM:.2f} lam")
    print(f"  d(A, B) / r_barrier = {d_AB/r_barrier:.2f}  (>{min_sep_factor:.1f}  OK)")

    return idx_A, idx_B


# ══════════════════════════════════════════════════════════════════
# Geometry validation
# ══════════════════════════════════════════════════════════════════

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
        print(f"  OK: B starts {(d_B_vc - r_barrier)*1e6:.1f} um "
              f"outside the barrier.")
    return status


# ══════════════════════════════════════════════════════════════════
# Path generation
# ══════════════════════════════════════════════════════════════════

def build_vortex_path(
    start: np.ndarray,
    end: np.ndarray,
    n_steps: int,
    path_mode: str = "straight",
    arc_height_mm: float = 0.0,
) -> np.ndarray:
    """Build the vortex centre path (straight or Bezier arc)."""
    t = np.linspace(0, 1, n_steps)
    if path_mode == "straight" or arc_height_mm == 0.0:
        return start[None, :] + t[:, None] * (end - start)[None, :]

    d = end - start
    perp = np.array([-d[1], d[0]])
    perp = perp / np.linalg.norm(perp)
    sign = 1.0 if path_mode == "arc_left" else -1.0
    mid_ctrl = 0.5 * (start + end) + sign * arc_height_mm * 1e-3 * perp

    path = np.zeros((n_steps, 2))
    for i, ti in enumerate(t):
        path[i] = ((1 - ti) ** 2 * start
                    + 2 * (1 - ti) * ti * mid_ctrl
                    + ti ** 2 * end)
    return path


# ══════════════════════════════════════════════════════════════════
# Field helpers
# ══════════════════════════════════════════════════════════════════

def total_pressure(p_sw, vortex_gen, psi, alpha, beta, center_xy):
    p_v = vortex_gen.get_field(center_xy)
    return beta * p_sw + alpha * np.exp(1j * psi) * p_v


def _make_interp(F, xg, yg):
    return RegularGridInterpolator(
        (yg, xg), F, bounds_error=False, fill_value=0.0)


def _eval_at(interp, pos_xy):
    pts = np.column_stack([pos_xy[:, 1], pos_xy[:, 0]])
    return interp(pts)


# ══════════════════════════════════════════════════════════════════
# Mechanism test: ring barrier profile
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
    """
    Sample Gorkov potential around the barrier ring and test for
    a directional opening vs global basin merging.
    """
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
# Phase sweep
# ══════════════════════════════════════════════════════════════════

def phase_sweep(
    p_sw, vortex_gen, xg, yg, dx, dy,
    psi_values, alpha, beta,
    vortex_center, pos_A, pos_B, r_barrier,
    w_barrier=1.0, w_pull=1.0, w_lateral=0.5,
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

        score = w_barrier * DeltaU - w_pull * F_in_B + w_lateral * F_perp_B

        results.append({
            "psi": psi, "F_A_in": F_A_in, "F_in_B": F_in_B,
            "F_perp_B": F_perp_B, "DeltaU": DeltaU, "score": score,
            "iFx": iFx, "iFy": iFy, "iU": iU,
        })

    valid = [r for r in results if r["F_A_in"] > 0]
    best = (min(valid, key=lambda r: r["score"]) if valid
            else max(results, key=lambda r: r["F_A_in"]))

    best["mechanism"] = ring_barrier_test(
        best["iU"], vortex_center, pos_B, r_barrier,
        opening_threshold=opening_threshold)
    return best


# ══════════════════════════════════════════════════════════════════
# GIF rendering
# ══════════════════════════════════════════════════════════════════

def _compute_view(
    traps_m: np.ndarray,
    idx_A: int,
    idx_B: int,
    vortex_path: np.ndarray,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Compute a fixed view that covers A, B, their path, and any
    neighbouring traps within a modest radius -- decoupled from the
    full simulation domain.
    """
    # Key points that must be visible
    key_pts = np.vstack([
        traps_m[idx_A][None, :],
        traps_m[idx_B][None, :],
        vortex_path,
    ]) * 1e3  # mm

    # Also include any traps within 1.5mm of the A-B midpoint
    mid = 0.5 * (traps_m[idx_A] + traps_m[idx_B])
    d_to_mid = np.linalg.norm(traps_m - mid, axis=1)
    nearby_mask = d_to_mid < 1.5e-3
    if np.any(nearby_mask):
        key_pts = np.vstack([key_pts, traps_m[nearby_mask] * 1e3])

    view_xlim = (float(key_pts[:, 0].min()) - VIEW_MARGIN_MM,
                 float(key_pts[:, 0].max()) + VIEW_MARGIN_MM)
    view_ylim = (float(key_pts[:, 1].min()) - VIEW_MARGIN_MM,
                 float(key_pts[:, 1].max()) + VIEW_MARGIN_MM)
    return view_xlim, view_ylim


def _render_frame(
    ax, fd, xg, yg, dx, dy,
    p_sw, vortex_gen, traps_m, idx_A, idx_B,
    r_barrier_mm, extent, view_xlim, view_ylim,
    frame_idx, n_frames_total,
):
    pos_A_mm = fd["pos_A"] * 1e3
    pos_B_mm = fd["pos_B"] * 1e3
    vc_mm    = fd["vortex_center"] * 1e3
    psi      = fd["psi"]
    alpha    = fd["alpha"]
    d_AB_um  = fd["d_AB"] * 1e6
    trail_mm = np.array(fd["vortex_trail"]) * 1e3
    label    = fd.get("label", "SWEEP")
    b_status = fd.get("B_status", "")

    p_tot = total_pressure(p_sw, vortex_gen, psi, alpha, BETA_FIXED,
                           fd["vortex_center"])
    U_disp, _, _ = gorkov_grid_2d(
        p_tot, dx, dy, OMEGA, RHO0, C_WATER,
        PPAR["a"], PPAR["f1"], PPAR["f2"],
    )

    ix_lo = max(0, int(np.searchsorted(xg, (view_xlim[0] - 0.1) * 1e-3)))
    ix_hi = min(len(xg),
                int(np.searchsorted(xg, (view_xlim[1] + 0.1) * 1e-3)))
    iy_lo = max(0, int(np.searchsorted(yg, (view_ylim[0] - 0.1) * 1e-3)))
    iy_hi = min(len(yg),
                int(np.searchsorted(yg, (view_ylim[1] + 0.1) * 1e-3)))
    U_roi = U_disp[iy_lo:iy_hi, ix_lo:ix_hi]
    lo = float(np.percentile(U_roi, 0.5))
    hi = float(np.percentile(U_roi, 99.5))

    ax.imshow(
        U_disp, origin="lower", extent=extent, cmap=CMAP,
        vmin=lo, vmax=hi, aspect="equal", interpolation="bicubic",
    )

    # ── Trap markers (only those inside the view) ────────────────
    traps_mm = traps_m * 1e3
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

    # ── Vortex trail ─────────────────────────────────────────────
    if len(trail_mm) > 1:
        ax.plot(trail_mm[:, 0], trail_mm[:, 1], color=COL_CTR,
                lw=0.9, alpha=0.75, zorder=5)
    ax.plot(vc_mm[0], vc_mm[1], marker="+", color=COL_CTR,
            ms=12, mew=2.0, zorder=8)

    # ── Barrier ring ─────────────────────────────────────────────
    ax.add_patch(mpatches.Circle(
        (vc_mm[0], vc_mm[1]), r_barrier_mm,
        fill=False, edgecolor=COL_BARRIER, linestyle="--",
        linewidth=1.0, alpha=0.7, zorder=6))

    # ── Particles ────────────────────────────────────────────────
    ax.add_patch(mpatches.Circle(
        (pos_A_mm[0], pos_A_mm[1]), PARTICLE_RADIUS_MM,
        facecolor=COL_A, edgecolor="white", linewidth=0.6,
        zorder=7, alpha=0.9))
    ax.add_patch(mpatches.Circle(
        (pos_B_mm[0], pos_B_mm[1]), PARTICLE_RADIUS_MM,
        facecolor=COL_B, edgecolor="white", linewidth=0.6,
        zorder=7, alpha=0.9))
    ax.scatter([], [], c=COL_A, s=30, label="A (vortex)")
    ax.scatter([], [], c=COL_B, s=30, label="B (SW trap)")

    # ── Title ────────────────────────────────────────────────────
    ax.set_xlim(*view_xlim)
    ax.set_ylim(*view_ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    status_str = f" | B: {b_status}" if b_status else ""
    ax.set_title(
        f"[{label}] psi={psi:.2f} | alpha={alpha:.1f} | "
        f"d(A,B)={d_AB_um:.0f} um{status_str}\n"
        f"frame {frame_idx + 1}/{n_frames_total}",
        fontsize=9)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.7)


def render_gif(
    frames_data, xg, yg, dx, dy,
    p_sw, vortex_gen, traps_m,
    idx_A, idx_B, r_barrier_mm,
    view_xlim, view_ylim,
    out_path, duration_ms=GIF_DURATION_MS,
):
    x_mm = xg * 1e3
    y_mm = yg * 1e3
    extent = [x_mm[0], x_mm[-1], y_mm[0], y_mm[-1]]

    n_total = len(frames_data)
    frames: List[Image.Image] = []
    print(f"  Rendering {n_total} GIF frames...")

    for fi, fdat in enumerate(frames_data):
        fig, ax = plt.subplots(1, 1, figsize=(6.0, 6.0))
        _render_frame(
            ax, fdat, xg, yg, dx, dy, p_sw, vortex_gen,
            traps_m, idx_A, idx_B, r_barrier_mm,
            extent, view_xlim, view_ylim, fi, n_total,
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
# B-status helper
# ══════════════════════════════════════════════════════════════════

def _b_status(pos_B, vortex_center, r_barrier):
    d = float(np.linalg.norm(pos_B - vortex_center))
    if d > r_barrier * 1.05:
        return "OUTSIDE"
    if d < r_barrier * 0.95:
        return "INSIDE"
    return "ON"


# ══════════════════════════════════════════════════════════════════
# Main experiment
# ══════════════════════════════════════════════════════════════════

def run_experiment(args) -> None:
    t0_total = time.time()

    aperture_m  = args.bessel_aperture_mm * 1e-3
    prop_dist_m = args.bessel_prop_dist_mm * 1e-3
    focus_f_m   = args.focus_mm * 1e-3

    # ── Load, calibrate ───────────────────────────────────────────
    (p_sw, xg, yg, dx, dy, traps_m,
     vortex_gen, r_barrier) = load_data(aperture_m, prop_dist_m, focus_f_m)

    r_barrier_mm = r_barrier * 1e3

    # ── Select A/B pair satisfying geometry constraint ────────────
    idx_A, idx_B = select_trap_pair(
        traps_m, r_barrier, xg, yg,
        idx_B_override=args.idx_B,
        min_sep_factor=args.min_sep_factor,
    )

    # ── Validate ──────────────────────────────────────────────────
    geom_status = validate_geometry(traps_m, idx_A, idx_B, r_barrier)

    vortex_start = traps_m[idx_A].copy()
    vortex_end   = traps_m[idx_B].copy()

    # ── Build path ────────────────────────────────────────────────
    psi_values = np.linspace(0, 2 * np.pi, args.n_psi, endpoint=False)
    vortex_path = build_vortex_path(
        vortex_start, vortex_end, args.n_vortex_steps,
        path_mode=args.path_mode,
        arc_height_mm=args.arc_height_mm,
    )
    print(f"  Path mode: {args.path_mode}"
          + (f" (arc height {args.arc_height_mm:.2f} mm)"
             if args.path_mode != "straight" else ""))

    # ── Compute render view (decoupled from sim domain) ───────────
    view_xlim, view_ylim = _compute_view(
        traps_m, idx_A, idx_B, vortex_path)
    print(f"  Render view: x=[{view_xlim[0]:.2f}, {view_xlim[1]:.2f}] "
          f"y=[{view_ylim[0]:.2f}, {view_ylim[1]:.2f}] mm")

    # ── Output directory ──────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (PROJECT_ROOT / "results" / "dev"
               / "vortex_entry_test" / timestamp)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {out_dir}")

    snap_indices = set(
        np.linspace(0, args.n_dyn_steps - 1, FRAMES_PER_VSTEP,
                    dtype=int).tolist()
    )

    all_csv_rows: List[List] = []

    for alpha in args.alphas:
        print(f"\n{'='*60}")
        print(f"  alpha = {alpha:.1f}")
        print(f"{'='*60}")

        pos_A = traps_m[idx_A].copy()
        pos_B = traps_m[idx_B].copy()
        frames_data: List[Dict] = []
        vortex_trail: List[np.ndarray] = []

        # ── Intro frames (alpha=0, standing wave only) ────────────
        intro_vc = vortex_start.copy()
        b_stat_intro = _b_status(pos_B, intro_vc, r_barrier)
        for _ in range(N_INTRO_FRAMES):
            frames_data.append({
                "pos_A": traps_m[idx_A].copy(),
                "pos_B": traps_m[idx_B].copy(),
                "vortex_center": intro_vc.copy(),
                "vortex_trail": [intro_vc.copy()],
                "psi": 0.0, "alpha": 0.0,
                "d_AB": float(np.linalg.norm(
                    traps_m[idx_A] - traps_m[idx_B])),
                "label": "INTRO", "B_status": b_stat_intro,
            })

        # ── Main loop ─────────────────────────────────────────────
        for v_step in range(args.n_vortex_steps):
            vortex_center = vortex_path[v_step].copy()
            vortex_trail.append(vortex_center.copy())

            t_sweep = time.time()
            best = phase_sweep(
                p_sw, vortex_gen, xg, yg, dx, dy,
                psi_values, alpha, BETA_FIXED,
                vortex_center, pos_A, pos_B, r_barrier,
                w_barrier=args.w_barrier,
                w_pull=args.w_pull,
                w_lateral=args.w_lateral,
                opening_threshold=args.opening_threshold,
            )
            psi_best = float(best["psi"])
            DeltaU   = float(best["DeltaU"])
            F_in_B   = float(best["F_in_B"])
            F_perp_B = float(best["F_perp_B"])
            F_A_in   = float(best["F_A_in"])
            score    = float(best["score"])
            iFx, iFy = best["iFx"], best["iFy"]
            mech     = best["mechanism"]
            dt_sweep = time.time() - t_sweep

            ret_s = "OK" if F_A_in > 0 else "FALLBACK"
            opn_s = mech["mechanism"].upper()
            print(f"  v_step {v_step+1:02d}/{args.n_vortex_steps} | "
                  f"psi={psi_best:.2f} | dU={DeltaU:.3e} | "
                  f"F_in_B={F_in_B:.3e} | F_A_in={F_A_in:.3e} "
                  f"({ret_s}) | {opn_s} | {dt_sweep*1e3:.0f}ms")

            # ── Dynamics ──────────────────────────────────────────
            for dyn_i in range(args.n_dyn_steps):
                FA = np.array([
                    float(_eval_at(iFx, pos_A[None, :])[0]),
                    float(_eval_at(iFy, pos_A[None, :])[0])])
                FB = np.array([
                    float(_eval_at(iFx, pos_B[None, :])[0]),
                    float(_eval_at(iFy, pos_B[None, :])[0])])

                pos_A = pos_A + np.clip(
                    SCALE * DT_DEFAULT * FA,
                    -args.max_step, args.max_step)
                pos_B = pos_B + np.clip(
                    SCALE * DT_DEFAULT * FB,
                    -args.max_step, args.max_step)

                pos_A[0] = np.clip(pos_A[0], xg[2], xg[-3])
                pos_A[1] = np.clip(pos_A[1], yg[2], yg[-3])
                pos_B[0] = np.clip(pos_B[0], xg[2], xg[-3])
                pos_B[1] = np.clip(pos_B[1], yg[2], yg[-3])

                if dyn_i in snap_indices:
                    b_stat = _b_status(pos_B, vortex_center, r_barrier)
                    frames_data.append({
                        "pos_A": pos_A.copy(),
                        "pos_B": pos_B.copy(),
                        "vortex_center": vortex_center.copy(),
                        "vortex_trail": [v.copy() for v in vortex_trail],
                        "psi": psi_best, "alpha": alpha,
                        "d_AB": float(np.linalg.norm(pos_A - pos_B)),
                        "label": "SWEEP",
                        "B_status": b_stat,
                    })

            d_AB = float(np.linalg.norm(pos_A - pos_B))
            d_A_vc = float(np.linalg.norm(pos_A - vortex_center))
            d_B_vc = float(np.linalg.norm(pos_B - vortex_center))
            delta_B = d_B_vc - r_barrier
            A_trapped = bool(d_A_vc < r_barrier)
            b_stat_end = _b_status(pos_B, vortex_center, r_barrier)

            all_csv_rows.append([
                alpha, v_step,
                vortex_center[0] * 1e3, vortex_center[1] * 1e3,
                psi_best, DeltaU, F_in_B, F_perp_B, F_A_in, score,
                pos_A[0] * 1e3, pos_A[1] * 1e3,
                pos_B[0] * 1e3, pos_B[1] * 1e3,
                d_AB * 1e6, delta_B * 1e6, b_stat_end,
                A_trapped, mech["mechanism"],
                mech["U_B_sector"], mech["U_other"],
            ])

        # ── GIF ───────────────────────────────────────────────────
        gif_path = out_dir / f"vortex_entry_alpha_{alpha:.1f}.gif"
        render_gif(frames_data, xg, yg, dx, dy, p_sw, vortex_gen,
                   traps_m, idx_A, idx_B, r_barrier_mm,
                   view_xlim, view_ylim, gif_path)

        # ── Summary ───────────────────────────────────────────────
        final = frames_data[-1]
        d_B_core = float(np.linalg.norm(
            final["pos_B"] - final["vortex_center"]))
        d_A_core = float(np.linalg.norm(
            final["pos_A"] - final["vortex_center"]))
        n_openings = sum(
            1 for r in all_csv_rows
            if r[0] == alpha and r[18] == "true_opening")
        print(f"\n  alpha={alpha:.1f} summary:")
        print(f"    d(A, vortex) = {d_A_core*1e6:.1f} um  "
              f"({'TRAPPED' if d_A_core < r_barrier else 'drifted'})")
        print(f"    d(B, vortex) = {d_B_core*1e6:.1f} um  "
              f"(delta_B = {(d_B_core - r_barrier)*1e6:.1f} um) "
              f"({final.get('B_status', '?')})")
        print(f"    True openings: {n_openings}/{args.n_vortex_steps}")

    # ── CSV ───────────────────────────────────────────────────────
    csv_path = out_dir / "phase_sweep_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "alpha", "v_step", "vortex_x_mm", "vortex_y_mm",
            "psi_best", "DeltaU_entry", "F_in_B", "F_perp_B", "F_A_in",
            "score", "A_x_mm", "A_y_mm", "B_x_mm", "B_y_mm",
            "d_AB_um", "delta_B_um", "B_status",
            "A_trapped", "mechanism", "U_B_sector", "U_other",
        ])
        writer.writerows(all_csv_rows)
    print(f"\nSaved CSV: {csv_path}")

    # ── Manifest ──────────────────────────────────────────────────
    elapsed = time.time() - t0_total
    manifest = {
        "timestamp": timestamp,
        "elapsed_s": round(elapsed, 2),
        "alphas": args.alphas,
        "beta_fixed": BETA_FIXED,
        "n_psi": args.n_psi,
        "n_vortex_steps": args.n_vortex_steps,
        "n_dyn_steps": args.n_dyn_steps,
        "max_step_m": args.max_step,
        "w_barrier": args.w_barrier,
        "w_pull": args.w_pull,
        "w_lateral": args.w_lateral,
        "path_mode": args.path_mode,
        "arc_height_mm": args.arc_height_mm,
        "vortex_charge": VORTEX_CHARGE,
        "aperture_mm": args.bessel_aperture_mm,
        "prop_dist_mm": args.bessel_prop_dist_mm,
        "focus_mm": args.focus_mm,
        "r_barrier_um": round(r_barrier * 1e6, 1),
        "initial_geometry": geom_status,
        "idx_A": idx_A,
        "idx_B": idx_B,
        "n_traps_total": len(traps_m),
        "n_intro_frames": N_INTRO_FRAMES,
        "frames_per_vstep": FRAMES_PER_VSTEP,
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone in {elapsed:.1f}s — results in {out_dir}")


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Vortex entry test — phase sweep without MPC")

    g = parser.add_argument_group("vortex beam")
    g.add_argument("--bessel_aperture_mm", type=float,
                   default=BESSEL_APERTURE_MM_DEFAULT,
                   help="Lens aperture radius [mm] — sets numerical aperture "
                        "(default %(default)s)")
    g.add_argument("--bessel_prop_dist_mm", type=float,
                   default=BESSEL_PROP_DIST_MM_DEFAULT,
                   help="ASM propagation distance [mm] — set equal to "
                        "--focus_mm to evaluate at the focal plane "
                        "(default %(default)s)")
    g.add_argument("--focus_mm", type=float,
                   default=FOCUS_MM_DEFAULT,
                   help="Lens focal length [mm] — controls ring tightness; "
                        "shorter f → smaller ring. Use with "
                        "--bessel_prop_dist_mm equal to this value for "
                        "focal-plane evaluation. (default %(default)s)")

    g = parser.add_argument_group("trap selection")
    g.add_argument("--idx_B", type=int, default=None,
                   help="Override B trap index (default: nearest to "
                        "grid centre)")
    g.add_argument("--min_sep_factor", type=float, default=GEOM_MARGIN,
                   help="A must satisfy d(A,B) > factor * r_barrier "
                        "(default %(default)s)")
    g.add_argument("--opening_threshold", type=float, default=OPENING_THRESHOLD,
                   help="delta_B fraction required for 'true_opening' "
                        "classification (default %(default)s)")

    g = parser.add_argument_group("sweep")
    g.add_argument("--n_psi", type=int, default=N_PSI_DEFAULT)
    g.add_argument("--alphas", nargs="+", type=float,
                   default=ALPHAS_DEFAULT)
    g.add_argument("--n_vortex_steps", type=int, default=N_VORTEX_STEPS)
    g.add_argument("--n_dyn_steps", type=int, default=N_DYN_STEPS_PER_POS)
    g.add_argument("--max_step", type=float, default=MAX_STEP_DEFAULT,
                   help="Max displacement per dyn step [m]")

    g = parser.add_argument_group("score weights")
    g.add_argument("--w_barrier", type=float, default=W_BARRIER_DEFAULT)
    g.add_argument("--w_pull", type=float, default=W_PULL_DEFAULT)
    g.add_argument("--w_lateral", type=float, default=W_LATERAL_DEFAULT)

    g = parser.add_argument_group("path")
    g.add_argument("--path_mode", type=str, default="straight",
                   choices=["straight", "arc_left", "arc_right"])
    g.add_argument("--arc_height_mm", type=float, default=0.15)

    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
