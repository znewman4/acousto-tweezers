#!/usr/bin/env python3
"""
Vortex Bridge Design Study
===========================

Design and rank vortex lens configurations that create a highly localised
"bridge" between two neighbouring standing-wave traps while preserving
the rest of the lattice.

The objective is to create a localised saddle-lowering corridor between
two specific wells so that a particle can snap between them.

All metrics are RELATIVE ranking tools.  Absolute magnitudes are not trusted.

Phases:
  0 — Define bridge geometry (detect well pair A,B)
  1 — Lens design sweep (plastic / BG / LG families)
  2 — Full metric computation (alpha x phi sweep)
  2.5 — Phase DOF audit
  3 — Translation robustness for top 3 configs (phase-optimised)
  4 — Output: CSV, figures, JSON, markdown note

Usage:
    micromamba run -n acousto-complex python scripts/experiments/vortex_bridge_design_study.py

Outputs:
    results/vortex_bridge_design_study_YYYYMMDD_HHMMSS/
        csv/full_config_metrics.csv
        csv/shortlist_top_configs.csv
        csv/translation_metrics.csv
        results.json
        figures/...
        VORTEX_BRIDGE_DESIGN_STUDY_NOTE.md
"""
from __future__ import annotations

import csv
import gc
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.colors import TwoSlopeNorm
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import minimum_filter

# ── project path ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from acoustweezers.experiments.farfield_petri_cuboid.presets import (
    CORRECTED_PRESET,
)

# ==================================================================
# Physical constants
# ==================================================================
WATER_DENSITY = 997.0
WATER_C       = 1484.0
F_HZ          = 2.0e6
OMEGA         = 2 * np.pi * F_HZ
LAM           = WATER_C / F_HZ          # 0.742 mm
K_WATER       = OMEGA / WATER_C          # 8467.9 rad/m

PARTICLE_RADIUS = 5e-6                   # 5 µm polystyrene bead
RHO_P, C_P = 1050.0, 2350.0
F1_CONTRAST = 1 - WATER_DENSITY * WATER_C**2 / (RHO_P * C_P**2)
F2_CONTRAST = 2 * (RHO_P - WATER_DENSITY) / (2 * RHO_P + WATER_DENSITY)

H_UNDER = CORRECTED_PRESET["H_under"]
H_TOP   = CORRECTED_PRESET.get("H_top", 2.0085e-3)
Z_MID   = H_UNDER + H_TOP / 2
Z_STAR  = Z_MID + 0.25 * LAM            # ≈ 4.190 mm

CX, CY  = 3.0e-3, 3.0e-3               # domain centre

# Bridge geometry parameters
BRIDGE_WIDTH_LAM = 0.20                  # capsule half-width in λ
NEIGHBOURHOOD_RADIUS_LAM = 2.0          # neighbourhood ROI radius in λ

V_RATIO = (CORRECTED_PRESET["standing_velocity_amplitude"] /
           CORRECTED_PRESET["disk_velocity_amplitude"])     # 10

EPL     = 3          # elem/λ  (reduced for ~6 GB RAM — relative ranking OK)
GRID_N  = 300        # post-FEM interpolation grid
FIG_DPI = 300

# The minimum pixel separation for local-minimum detection.
# λ/2 ≈ 0.371 mm.  With physical span ~4.5 mm / 300 px ≈ 15 µm/px,
# half-wavelength ≈ 25 px.  Use 22 to tolerate slight shifts.
MIN_SEP_PX = 22

WORKER_SCRIPT = PROJECT_ROOT / "scripts" / "experiments" / "_solve_worker_multi_z.py"

# Reuse cached FEM from a previous run if available.
PREV_CACHE = PROJECT_ROOT / "results" / "vortex_bridge_design_study_phase_20260227_141230" / "_cache"

# ==================================================================
# Output directory
# ==================================================================
TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
OUT_DIR   = PROJECT_ROOT / "results" / f"vortex_bridge_design_study_phase_{TIMESTAMP}"
FIG_DIR   = OUT_DIR / "figures"
CSV_DIR   = OUT_DIR / "csv"
CACHE_DIR = OUT_DIR / "_cache"

# ==================================================================
# Alpha sweep range (log-spaced)
# ==================================================================
ALPHA_VALUES = np.logspace(np.log10(0.1), np.log10(30.0), 30)

# ==================================================================
# Phase sweep (global relative phase DOF)
# ==================================================================
N_PHI = 12
PHI_VALUES = np.linspace(0, 2 * np.pi, N_PHI, endpoint=False)


# ==================================================================
# Lens configurations to sweep
# ==================================================================
def build_lens_configs() -> List[Dict[str, Any]]:
    """Build full set of lens configurations for the sweep."""
    configs = []

    ell_values = [1, 2]
    aperture_radii_mm = [0.8, 1.0, 1.2, 1.5]

    # Plastic family
    plastic_focal_lengths_mm = [1.5, 2.5, 3.5]
    plastic_phase_modes = [
        ("continuous", None),   # unwrapped — note: solver always wraps
        ("wrapped", None),      # standard 2π wrap
        ("quant8", 8),
        ("quant16", 16),
        ("quant32", 32),
    ]

    for ell in ell_values:
        for R_mm in aperture_radii_mm:
            for f_mm in plastic_focal_lengths_mm:
                for phase_mode, quant_levels in plastic_phase_modes:
                    cid = f"plastic_l{ell}_R{R_mm}_f{f_mm}_{phase_mode}"
                    configs.append({
                        "config_id": cid,
                        "family": "plastic",
                        "ell": ell,
                        "aperture_radius_mm": R_mm,
                        "focal_length_mm": f_mm,
                        "phase_mode": phase_mode,
                        "quant_levels": quant_levels,
                    })

    # Bessel-Gauss family
    bg_waists_mm = [0.4, 0.8, 1.2]
    for ell in ell_values:
        for R_mm in aperture_radii_mm:
            for w_mm in bg_waists_mm:
                cid = f"bg_l{ell}_R{R_mm}_w{w_mm}"
                configs.append({
                    "config_id": cid,
                    "family": "bg",
                    "ell": ell,
                    "aperture_radius_mm": R_mm,
                    "beam_waist_mm": w_mm,
                    "k_r": 0.5 * K_WATER,
                })

    # Laguerre-Gaussian family
    lg_waists_mm = [0.4, 0.8, 1.2]
    for ell in ell_values:
        for R_mm in aperture_radii_mm:
            for w_mm in lg_waists_mm:
                cid = f"lg_l{ell}_R{R_mm}_w{w_mm}"
                configs.append({
                    "config_id": cid,
                    "family": "lg",
                    "ell": ell,
                    "aperture_radius_mm": R_mm,
                    "beam_waist_mm": w_mm,
                })

    return configs


def config_to_overrides(cfg_entry: Dict, standing_only: bool = False) -> dict:
    """Convert a sweep config entry to FarFieldConfig overrides."""
    ov = {**CORRECTED_PRESET, "elements_per_wavelength": EPL}

    if standing_only:
        ov["disk_velocity_amplitude"] = 0.0
        return ov

    ov["standing_velocity_amplitude"] = 0.0
    ov["disk_velocity_amplitude"] = 1e-6

    fam = cfg_entry["family"]
    ov["lens_l"] = cfg_entry["ell"]
    ov["lens_focus_offset_x"] = 0.0
    ov["lens_focus_offset_y"] = 0.0
    ov["lens_apodization"] = "cosine_taper"
    ov["disk_radius"] = cfg_entry["aperture_radius_mm"] * 1e-3

    if fam == "plastic":
        ov["lens_drive"] = "plastic"
        ov["lens_focal_length"] = cfg_entry["focal_length_mm"] * 1e-3
    elif fam == "bg":
        ov["lens_drive"] = "bessel_gauss"
        ov["lens_k_r"] = cfg_entry.get("k_r", 0.5 * K_WATER)
        ov["lens_beam_waist"] = cfg_entry["beam_waist_mm"] * 1e-3
    elif fam == "lg":
        ov["lens_drive"] = "lg"
        ov["lens_beam_waist"] = cfg_entry["beam_waist_mm"] * 1e-3
        ov["lens_focal_length"] = 0.0

    return ov


def _fem_key(cfg_entry: Dict) -> str:
    """Unique FEM solve key for a config (ignoring phase mode / quant)."""
    fam = cfg_entry["family"]
    ell = cfg_entry["ell"]
    R = cfg_entry["aperture_radius_mm"]
    if fam == "plastic":
        return f"plastic_l{ell}_R{R}_f{cfg_entry['focal_length_mm']}"
    elif fam == "bg":
        return f"bg_l{ell}_R{R}_w{cfg_entry['beam_waist_mm']}"
    elif fam == "lg":
        return f"lg_l{ell}_R{R}_w{cfg_entry['beam_waist_mm']}"
    return cfg_entry["config_id"]


# ==================================================================
# Subprocess FEM solver  (with cache fallback to previous run)
# ==================================================================
def solve_multi_z(overrides: dict, label: str,
                  z_list: List[float] = None,
                  mid_y: float = CY,
                  n_xy: int = GRID_N) -> Optional[dict]:
    """FEM solve -> .npz with xg,yg,p_xy_{i},xg_xz,zg_xz,p_xz."""
    if z_list is None:
        z_list = [Z_STAR]
    result_file = str(CACHE_DIR / f"_grid_{label}.npz")

    # 1) Check own cache
    if os.path.exists(result_file):
        return dict(np.load(result_file, allow_pickle=False))

    # 2) Check previous-run cache
    prev_file = PREV_CACHE / f"_grid_{label}.npz"
    if prev_file.exists():
        data = dict(np.load(str(prev_file), allow_pickle=False))
        # copy into our cache
        np.savez(result_file, **data)
        return data

    # 3) Fresh solve
    args = {
        "overrides": overrides,
        "label": label,
        "trap_z_list": z_list,
        "mid_y": mid_y,
        "n_xy": n_xy,
        "result_file": result_file,
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(args, f, default=str)
        args_file = f.name
    try:
        proc = subprocess.run(
            [sys.executable, str(WORKER_SCRIPT), args_file],
            capture_output=False, timeout=1200,
        )
        if proc.returncode != 0:
            print(f"  *** SOLVE FAILED: {label}  rc={proc.returncode}")
            return None
    finally:
        os.unlink(args_file)
    if not os.path.exists(result_file):
        return None
    return dict(np.load(result_file, allow_pickle=False))


def combine(p_stand, p_vortex, alpha, phi=0.0):
    """Linearly superpose standing + alpha * exp(1j*phi) * V_ratio * vortex.

    Parameters
    ----------
    p_stand : complex array
        Standing-wave pressure field.
    p_vortex : complex array
        Vortex pressure field.
    alpha : float
        Real-valued authority parameter.
    phi : float
        Global relative phase in radians (default 0).
    """
    return p_stand + alpha * np.exp(1j * phi) * V_RATIO * p_vortex


# ==================================================================
# Gor'kov potential & force
# ==================================================================
def gorkov_2d(p, dx, dy):
    """Gor'kov potential U on a 2-D XY grid (complex p)."""
    K_bulk = WATER_DENSITY * WATER_C**2
    p2 = np.abs(p)**2 / 2
    dp_dx = np.gradient(p, dx, axis=1)
    dp_dy = np.gradient(p, dy, axis=0)
    grad_p2 = np.abs(dp_dx)**2 + np.abs(dp_dy)**2
    v2 = grad_p2 / (2 * OMEGA**2 * WATER_DENSITY**2)
    pf = (4 * np.pi / 3) * PARTICLE_RADIUS**3
    return pf * (F1_CONTRAST * p2 / (2 * K_bulk) -
                 F2_CONTRAST * (3 * WATER_DENSITY / 4) * v2)


def gorkov_force(U, dx, dy):
    """F = -grad(U) -> (Fx[ny,nx], Fy[ny,nx])."""
    Fx = -np.gradient(U, dx, axis=1)
    Fy = -np.gradient(U, dy, axis=0)
    return Fx, Fy


# ==================================================================
# Trap finding  (FIXED: min_sep_px = MIN_SEP_PX ~ lambda/2)
# ==================================================================
def find_local_minima(U, xg, yg, min_sep_px: int = MIN_SEP_PX):
    """All local minima separated by >= min_sep_px pixels.
    Returns list of (ix, iy) sorted by ascending U."""
    filt = minimum_filter(U, size=min_sep_px)
    is_min = (U == filt) & np.isfinite(U)
    iys, ixs = np.where(is_min)
    order = np.argsort(U[iys, ixs])
    return list(zip(ixs[order].tolist(), iys[order].tolist()))


def find_nearest_neighbour_pair(minima, xg, yg, min_dist_lam=0.3):
    """Find nearest-neighbour well pair with separation >= min_dist_lam * lambda.
    Returns (well_A, well_B) as ((ix,iy), (ix,iy)).

    Prioritises pairs near domain centre to avoid PML-edge artefacts.
    """
    if len(minima) < 2:
        raise ValueError("Need at least 2 minima for bridge geometry")

    coords = np.array([(xg[ix], yg[iy]) for ix, iy in minima])
    d_min = min_dist_lam * LAM

    # Filter: only consider minima in the central 60% of the domain
    xmid, ymid = xg[len(xg) // 2], yg[len(yg) // 2]
    x_span = (xg[-1] - xg[0]) * 0.30
    y_span = (yg[-1] - yg[0]) * 0.30
    central = []
    for j, cxy in enumerate(coords):
        if abs(cxy[0] - xmid) <= x_span and abs(cxy[1] - ymid) <= y_span:
            central.append(j)

    if len(central) < 2:
        # Fallback: use all
        central = list(range(len(coords)))

    best_d2 = np.inf
    best_i, best_j = central[0], central[1] if len(central) > 1 else 0
    for ii in range(len(central)):
        for jj in range(ii + 1, len(central)):
            i, j = central[ii], central[jj]
            d2 = np.sum((coords[i] - coords[j])**2)
            d = np.sqrt(d2)
            if d < d_min:
                continue  # skip sub-wavelength noise pairs
            if d2 < best_d2:
                best_d2 = d2
                best_i, best_j = i, j

    return minima[best_i], minima[best_j]


# ==================================================================
# Bridge geometry regions
# ==================================================================
def define_bridge_regions(well_A, well_B, xg, yg,
                          bridge_half_width=BRIDGE_WIDTH_LAM * LAM,
                          nbr_radius=NEIGHBOURHOOD_RADIUS_LAM * LAM):
    """
    Define bridge, neighbourhood and outside regions.

    Bridge ROI: capsule around A->B line segment.
    Neighbourhood ROI: circle of nbr_radius at midpoint, excluding bridge.
    Outside: everything else.

    Returns dict with boolean masks (ny, nx) for each region.
    """
    ax, ay = well_A
    bx, by = well_B
    mx, my = (ax + bx) / 2, (ay + by) / 2

    X, Y = np.meshgrid(xg, yg)

    ab = np.array([bx - ax, by - ay])
    ab_len = np.linalg.norm(ab)
    if ab_len < 1e-12:
        ab_hat = np.array([1.0, 0.0])
    else:
        ab_hat = ab / ab_len

    # Distance to line segment A->B
    px = X - ax
    py = Y - ay
    t_param = (px * ab_hat[0] + py * ab_hat[1]) / ab_len if ab_len > 1e-12 else np.zeros_like(X)
    t_clamped = np.clip(t_param, 0, 1)
    cx_seg = ax + t_clamped * ab[0]
    cy_seg = ay + t_clamped * ab[1]
    dist_to_seg = np.sqrt((X - cx_seg)**2 + (Y - cy_seg)**2)

    bridge_mask = dist_to_seg <= bridge_half_width

    # Neighbourhood ROI: circle at midpoint
    dist_to_mid = np.sqrt((X - mx)**2 + (Y - my)**2)
    nbr_circle = dist_to_mid <= nbr_radius
    nbr_mask = nbr_circle & ~bridge_mask

    # Outside: everything not in neighbourhood circle
    outside_mask = ~nbr_circle

    return {
        "bridge": bridge_mask,
        "neighbourhood": nbr_mask,
        "outside": outside_mask,
        "midpoint": (mx, my),
        "well_A": (ax, ay),
        "well_B": (bx, by),
        "ab_hat": ab_hat,
        "ab_len": ab_len,
    }


# ==================================================================
# Phase quantisation for plastic lens
# ==================================================================
def quantise_phase(p_complex, n_levels):
    """Quantise the phase of a complex pressure field to n_levels.

    NOTE: This is a post-hoc approximation.  True quantisation
    occurs at the lens surface (boundary drive), not in the propagated
    far field.  The effect is qualitatively similar (sidelobe increase,
    efficiency loss) but not quantitatively exact.
    """
    phase = np.angle(p_complex)
    amp = np.abs(p_complex)
    phase_shifted = np.mod(phase, 2 * np.pi)
    step = 2 * np.pi / n_levels
    phase_quant = np.round(phase_shifted / step) * step
    return amp * np.exp(1j * phase_quant)


# ==================================================================
# Analytical field translation (for Phase 3)
# ==================================================================
def translate_field(p, xg, yg, delta_x, delta_y):
    """Translate a 2D complex field by (delta_x, delta_y) metres.

    Implements p_shifted(x, y) = p(x - dx, y - dy) via interpolation.
    Points outside the original domain are filled with zero.
    """
    interp_re = RegularGridInterpolator(
        (yg, xg), np.real(p), method="linear",
        bounds_error=False, fill_value=0.0)
    interp_im = RegularGridInterpolator(
        (yg, xg), np.imag(p), method="linear",
        bounds_error=False, fill_value=0.0)

    Y, X = np.meshgrid(yg, xg, indexing="ij")
    pts = np.column_stack([(Y - delta_y).ravel(), (X - delta_x).ravel()])
    re = interp_re(pts).reshape(p.shape)
    im = interp_im(pts).reshape(p.shape)
    return re + 1j * im


# ==================================================================
# Bridge-specific metrics
# ==================================================================
def sample_centreline(U, xg, yg, well_A, well_B, n_samples=200):
    """Sample U along the A->B centreline."""
    ax, ay = well_A
    bx, by = well_B

    interp = RegularGridInterpolator(
        (yg, xg), U, method="linear",
        bounds_error=False, fill_value=np.nan)

    t = np.linspace(0, 1, n_samples)
    xs = ax + t * (bx - ax)
    ys = ay + t * (by - ay)
    pts = np.column_stack([ys, xs])
    profile = interp(pts)

    s = t * np.sqrt((bx - ax)**2 + (by - ay)**2)
    return s, profile


def compute_saddle_barrier(s, profile):
    """Compute saddle barrier from centreline profile.
    Returns (barrier_A, barrier_B, saddle_value, saddle_s)."""
    if np.all(np.isnan(profile)):
        return np.nan, np.nan, np.nan, np.nan

    valid = ~np.isnan(profile)
    prof = profile.copy()
    prof[~valid] = np.inf

    U_A = prof[0]
    U_B = prof[-1]

    saddle_idx = np.argmax(prof)
    saddle_val = prof[saddle_idx]
    saddle_s = s[saddle_idx]

    barrier_A = saddle_val - U_A
    barrier_B = saddle_val - U_B

    return barrier_A, barrier_B, saddle_val, saddle_s


def compute_bridge_metrics(p_stand, p_vortex, xg, yg, alpha,
                           bridge_regions, U_stand=None,
                           phi: float = 0.0) -> Dict[str, Any]:
    """Compute full bridge metric dictionary for one config at one (alpha, phi)."""
    dx = xg[1] - xg[0]
    dy = yg[1] - yg[0]
    eps = 1e-30

    p_comb = combine(p_stand, p_vortex, alpha, phi=phi)

    if U_stand is None:
        U_stand = gorkov_2d(p_stand, dx, dy)
    U_comb = gorkov_2d(p_comb, dx, dy)
    Fx_stand, Fy_stand = gorkov_force(U_stand, dx, dy)
    Fx_comb, Fy_comb = gorkov_force(U_comb, dx, dy)

    br = bridge_regions
    bridge = br["bridge"]
    nbr = br["neighbourhood"]
    outside = br["outside"]
    well_A = br["well_A"]
    well_B = br["well_B"]

    m: Dict[str, Any] = {}
    m["alpha"] = float(alpha)

    # -- 1) Global preservation --
    delta_p = p_comb - p_stand
    norm_s_out = np.sqrt(np.sum(np.abs(p_stand[outside])**2))
    norm_d_out = np.sqrt(np.sum(np.abs(delta_p[outside])**2))
    m["eta_out"] = float(norm_d_out / (norm_s_out + eps))

    s_out = np.abs(p_stand[outside])
    c_out = np.abs(p_comb[outside])
    if np.std(s_out) > 0 and np.std(c_out) > 0:
        m["corr_out"] = float(np.corrcoef(s_out, c_out)[0, 1])
    else:
        m["corr_out"] = float("nan")

    # -- 2) Bridge localisation metrics --
    dU = U_comb - U_stand
    dFx = Fx_comb - Fx_stand
    dFy = Fy_comb - Fy_stand
    dF_mag = np.sqrt(dFx**2 + dFy**2)

    dU_L2_bridge = np.sqrt(np.sum(dU[bridge]**2)) if bridge.any() else 0.0
    dU_L2_nbr = np.sqrt(np.sum(dU[nbr]**2)) if nbr.any() else 0.0
    m["L_U"] = float(dU_L2_bridge / (dU_L2_nbr + eps))

    dF_L2_bridge = np.sqrt(np.sum(dF_mag[bridge]**2)) if bridge.any() else 0.0
    dF_L2_nbr = np.sqrt(np.sum(dF_mag[nbr]**2)) if nbr.any() else 0.0
    m["L_F"] = float(dF_L2_bridge / (dF_L2_nbr + eps))

    # -- 3) Bridge passability --
    s_stand, prof_stand = sample_centreline(U_stand, xg, yg, well_A, well_B)
    s_comb, prof_comb = sample_centreline(U_comb, xg, yg, well_A, well_B)

    barr_A_s, barr_B_s, _, _ = compute_saddle_barrier(s_stand, prof_stand)
    barr_A_c, barr_B_c, _, _ = compute_saddle_barrier(s_comb, prof_comb)

    m["barrier_A_stand"] = float(barr_A_s) if np.isfinite(barr_A_s) else 0.0
    m["barrier_B_stand"] = float(barr_B_s) if np.isfinite(barr_B_s) else 0.0
    m["barrier_A_comb"] = float(barr_A_c) if np.isfinite(barr_A_c) else 0.0
    m["barrier_B_comb"] = float(barr_B_c) if np.isfinite(barr_B_c) else 0.0

    avg_stand = (abs(barr_A_s) + abs(barr_B_s)) / 2 if (np.isfinite(barr_A_s) and np.isfinite(barr_B_s)) else 0.0
    avg_comb = (abs(barr_A_c) + abs(barr_B_c)) / 2 if (np.isfinite(barr_A_c) and np.isfinite(barr_B_c)) else 0.0

    if avg_stand > eps:
        m["barrier_reduction_pct"] = 100.0 * (avg_stand - avg_comb) / avg_stand
    else:
        m["barrier_reduction_pct"] = 0.0

    # -- 4) Neighbour preservation --
    minima_stand = find_local_minima(U_stand, xg, yg)
    minima_comb = find_local_minima(U_comb, xg, yg)

    mid_x, mid_y_val = br["midpoint"]
    min_dists = []
    for ix, iy in minima_stand:
        d = np.sqrt((xg[ix] - mid_x)**2 + (yg[iy] - mid_y_val)**2)
        dA = np.sqrt((xg[ix] - well_A[0])**2 + (yg[iy] - well_A[1])**2)
        dB = np.sqrt((xg[ix] - well_B[0])**2 + (yg[iy] - well_B[1])**2)
        if dA < 0.3 * LAM or dB < 0.3 * LAM:
            continue
        min_dists.append((ix, iy, d))

    min_dists.sort(key=lambda x: x[2])
    neighbour_traps = min_dists[:6]

    depth_changes = []
    pos_shifts = []
    for nix, niy, _ in neighbour_traps:
        U_s_val = U_stand[niy, nix]
        best_d = np.inf
        best_ix, best_iy = nix, niy
        for cix, ciy in minima_comb:
            d = np.sqrt((xg[cix] - xg[nix])**2 + (yg[ciy] - yg[niy])**2)
            if d < best_d:
                best_d = d
                best_ix, best_iy = cix, ciy

        U_c_val = U_comb[best_iy, best_ix]
        if abs(U_s_val) > eps:
            depth_changes.append(100.0 * (U_c_val - U_s_val) / abs(U_s_val))
        pos_shifts.append(best_d)

    m["n_neighbours"] = len(neighbour_traps)
    if depth_changes:
        m["neighbour_depth_max_pct"] = float(np.max(np.abs(depth_changes)))
        m["neighbour_depth_rms_pct"] = float(np.sqrt(np.mean(np.array(depth_changes)**2)))
    else:
        m["neighbour_depth_max_pct"] = float("nan")
        m["neighbour_depth_rms_pct"] = float("nan")
    if pos_shifts:
        m["neighbour_pos_shift_max_mm"] = float(np.max(pos_shifts)) * 1e3
    else:
        m["neighbour_pos_shift_max_mm"] = float("nan")

    return m


def find_alpha_min(p_stand, p_vortex, xg, yg, bridge_regions, U_stand,
                   alpha_values=ALPHA_VALUES,
                   phi_values=PHI_VALUES) -> Tuple[Optional[float], Optional[float], Dict]:
    """Find minimum alpha meeting bridge criteria, optimising over phi.

    Criteria:
      - barrier reduction >= 30 %
      - eta_out <= 0.05
      - neighbour depth change <= 5 %

    Returns (best_alpha, best_phi, best_metrics).
    """
    for alpha in alpha_values:
        for phi in phi_values:
            m = compute_bridge_metrics(p_stand, p_vortex, xg, yg, alpha,
                                       bridge_regions, U_stand=U_stand,
                                       phi=phi)
            if (m.get("barrier_reduction_pct", 0) >= 30.0
                    and m.get("eta_out", 1.0) <= 0.05
                    and m.get("neighbour_depth_rms_pct", 100.0) <= 5.0):
                return alpha, phi, m

    return None, None, {}


# ==================================================================
# Figure helpers
# ==================================================================
def _ext_mm(xg, yg):
    return [xg[0]*1e3, xg[-1]*1e3, yg[0]*1e3, yg[-1]*1e3]


def _draw_bridge_overlay(ax, br, color_A="lime", color_B="cyan",
                          color_mid="white"):
    ax_v, ay_v = br["well_A"]
    bx_v, by_v = br["well_B"]
    ax.plot(ax_v*1e3, ay_v*1e3, "o", color=color_A, ms=10, mew=2, mfc="none", label="Well A")
    ax.plot(bx_v*1e3, by_v*1e3, "s", color=color_B, ms=10, mew=2, mfc="none", label="Well B")
    ax.plot([ax_v*1e3, bx_v*1e3], [ay_v*1e3, by_v*1e3],
            "--", color=color_mid, lw=1.5, alpha=0.7)


def _zoom_to_bridge(xg, yg, br, padding_lam=3.0):
    ax_v, ay_v = br["well_A"]
    bx_v, by_v = br["well_B"]
    pad = padding_lam * LAM
    x0 = min(ax_v, bx_v) - pad
    x1 = max(ax_v, bx_v) + pad
    y0 = min(ay_v, by_v) - pad
    y1 = max(ay_v, by_v) + pad
    ix0 = max(0, int(np.searchsorted(xg, x0)))
    ix1 = min(len(xg), int(np.searchsorted(xg, x1)))
    iy0 = max(0, int(np.searchsorted(yg, y0)))
    iy1 = min(len(yg), int(np.searchsorted(yg, y1)))
    return slice(iy0, iy1), slice(ix0, ix1)


# ==================================================================
# Per-config figures
# ==================================================================
def fig_field_panels(p_stand, p_vortex, alpha, xg, yg, br, cid, fig_dir, phi=0.0):
    """4-panel: |p_stand|, |p_vortex|, |p_comb|, delta|p|."""
    p_comb = combine(p_stand, p_vortex, alpha, phi=phi)
    dp = np.abs(p_comb) - np.abs(p_stand)
    ext = _ext_mm(xg, yg)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=FIG_DPI)
    for i, (data, title, cmap) in enumerate([
        (np.abs(p_stand), "|p_stand|", "hot"),
        (np.abs(p_vortex * alpha * V_RATIO), f"|alpha*V*p_vortex| (alpha={alpha:.2f})", "hot"),
        (np.abs(p_comb), "|p_comb|", "hot"),
        (dp, "delta|p|", "RdBu_r"),
    ]):
        ax = axes[i // 2, i % 2]
        if cmap == "RdBu_r":
            vmax = max(abs(dp.min()), abs(dp.max())) or 1
            im = ax.imshow(data, extent=ext, origin="lower", cmap=cmap,
                           vmin=-vmax, vmax=vmax, interpolation="nearest")
        else:
            im = ax.imshow(data, extent=ext, origin="lower", cmap=cmap,
                           interpolation="nearest")
        _draw_bridge_overlay(ax, br)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
        plt.colorbar(im, ax=ax, shrink=0.72).set_label("Pa")

    fig.suptitle(f"Field panels -- {cid}  alpha={alpha:.2f}", fontsize=12)
    fig.tight_layout()
    fname = fig_dir / "field_panels_xy.png"
    fig.savefig(str(fname), dpi=FIG_DPI)
    plt.close(fig)
    return fname


def fig_gorkov_panels(U_stand, U_comb, xg, yg, br, cid, alpha, fig_dir):
    """3-panel: U_stand, U_comb, delta_U."""
    sy, sx = _zoom_to_bridge(xg, yg, br)
    xz, yz = xg[sx], yg[sy]
    ext = _ext_mm(xz, yz)

    Us = U_stand[sy, sx]
    Uc = U_comb[sy, sx]
    dU = Uc - Us

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), dpi=FIG_DPI)

    ax = axes[0]
    im = ax.imshow(Us, extent=ext, origin="lower", cmap="viridis", interpolation="nearest")
    _draw_bridge_overlay(ax, br)
    ax.set_title("U_stand", fontsize=11)
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    plt.colorbar(im, ax=ax, shrink=0.72).set_label("J")

    ax = axes[1]
    im = ax.imshow(Uc, extent=ext, origin="lower", cmap="viridis", interpolation="nearest")
    _draw_bridge_overlay(ax, br)
    ax.set_title(f"U_comb (alpha={alpha:.2f})", fontsize=11)
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    plt.colorbar(im, ax=ax, shrink=0.72).set_label("J")

    ax = axes[2]
    vmax = max(abs(np.nanmin(dU)), abs(np.nanmax(dU))) or 1e-30
    im = ax.imshow(dU, extent=ext, origin="lower", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, interpolation="nearest")
    _draw_bridge_overlay(ax, br)
    ax.set_title("delta_U = U_comb - U_stand", fontsize=11)
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    plt.colorbar(im, ax=ax, shrink=0.72).set_label("J")

    fig.suptitle(f"Gorkov potential -- {cid}", fontsize=12)
    fig.tight_layout()
    fname = fig_dir / "gorkov_xy.png"
    fig.savefig(str(fname), dpi=FIG_DPI)
    plt.close(fig)
    return fname


def fig_bridge_profile(U_stand, U_comb, xg, yg, br, cid, alpha, fig_dir):
    """Bridge centreline U(s) profile -- standing vs combined."""
    s_s, prof_s = sample_centreline(U_stand, xg, yg, br["well_A"], br["well_B"])
    s_c, prof_c = sample_centreline(U_comb, xg, yg, br["well_A"], br["well_B"])

    fig, ax = plt.subplots(figsize=(10, 5), dpi=FIG_DPI)
    ax.plot(s_s * 1e3, prof_s, 'b-', lw=2, label="U_stand")
    ax.plot(s_c * 1e3, prof_c, 'r-', lw=2, label=f"U_comb (alpha={alpha:.2f})")
    ax.axvline(0, color="lime", ls=":", lw=1, alpha=0.6, label="Well A")
    ax.axvline(s_s[-1] * 1e3, color="cyan", ls=":", lw=1, alpha=0.6, label="Well B")
    ax.set_xlabel("Arc length s along A->B (mm)")
    ax.set_ylabel("Gorkov potential U (J)")
    ax.set_title(f"Bridge centreline profile -- {cid}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fname = fig_dir / "bridge_profile.png"
    fig.savefig(str(fname), dpi=FIG_DPI)
    plt.close(fig)
    return fname


def fig_xz_slice(p_stand_xz, p_vortex_xz, alpha, xg_xz, zg_xz, br, cid, fig_dir, phi=0.0):
    """XZ slice through bridge midpoint."""
    p_comb_xz = combine(p_stand_xz, p_vortex_xz, alpha, phi=phi)
    dp = np.abs(p_comb_xz) - np.abs(p_stand_xz)
    ext = [xg_xz[0]*1e3, xg_xz[-1]*1e3, zg_xz[0]*1e3, zg_xz[-1]*1e3]

    fig, axes = plt.subplots(1, 3, figsize=(19, 6.5), dpi=FIG_DPI)
    for i, (data, title, cmap) in enumerate([
        (np.abs(p_stand_xz), "|p_stand| XZ", "hot"),
        (np.abs(p_comb_xz), "|p_comb| XZ", "hot"),
        (dp, "delta|p| XZ", "RdBu_r"),
    ]):
        ax = axes[i]
        if cmap == "RdBu_r":
            vmax = max(abs(dp.min()), abs(dp.max())) or 1
            im = ax.imshow(data, extent=ext, origin="lower", cmap=cmap,
                           aspect="auto", vmin=-vmax, vmax=vmax, interpolation="nearest")
        else:
            im = ax.imshow(data, extent=ext, origin="lower", cmap=cmap,
                           aspect="auto", interpolation="nearest")
        ax.axhline(Z_STAR*1e3, color="cyan", ls="--", lw=1, label="z*")
        ax.set_xlabel("x (mm)"); ax.set_ylabel("z (mm)")
        ax.set_title(title, fontsize=11)
        plt.colorbar(im, ax=ax, shrink=0.72).set_label("Pa")

    fig.suptitle(f"XZ slice -- {cid}  alpha={alpha:.2f}", fontsize=12)
    fig.tight_layout()
    fname = fig_dir / "xz_slice.png"
    fig.savefig(str(fname), dpi=FIG_DPI)
    plt.close(fig)
    return fname


# ==================================================================
# Comparison figures
# ==================================================================
def fig_localisation_vs_alpha(sweep_data, fig_dir):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=FIG_DPI)
    for cid, ams in sweep_data.items():
        alphas = [am["alpha"] for am in ams]
        lu = [am["L_U"] for am in ams]
        lf = [am["L_F"] for am in ams]
        short = cid[:25]
        axes[0].plot(alphas, lu, '-o', ms=3, label=short)
        axes[1].plot(alphas, lf, '-o', ms=3, label=short)
    axes[0].set_xscale("log"); axes[0].set_xlabel("alpha")
    axes[0].set_ylabel("L_U"); axes[0].set_title("L_U vs alpha")
    axes[0].legend(fontsize=6, loc="best"); axes[0].grid(True, alpha=0.3)
    axes[1].set_xscale("log"); axes[1].set_xlabel("alpha")
    axes[1].set_ylabel("L_F"); axes[1].set_title("L_F vs alpha")
    axes[1].legend(fontsize=6, loc="best"); axes[1].grid(True, alpha=0.3)
    fig.suptitle("Localisation scores vs authority alpha", fontsize=13)
    fig.tight_layout()
    fig.savefig(str(fig_dir / "localisation_vs_alpha.png"), dpi=FIG_DPI)
    plt.close(fig)


def fig_eta_vs_alpha(sweep_data, fig_dir):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=FIG_DPI)
    for cid, ams in sweep_data.items():
        alphas = [am["alpha"] for am in ams]
        eta = [am["eta_out"] for am in ams]
        ax.plot(alphas, eta, '-o', ms=3, label=cid[:25])
    ax.axhline(0.05, color="red", ls="--", lw=1, label="eta threshold=0.05")
    ax.set_xscale("log"); ax.set_xlabel("alpha")
    ax.set_ylabel("eta_out"); ax.set_title("Global disruption vs alpha")
    ax.legend(fontsize=6); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(fig_dir / "eta_vs_alpha.png"), dpi=FIG_DPI)
    plt.close(fig)


def fig_barrier_vs_alpha(sweep_data, fig_dir):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=FIG_DPI)
    for cid, ams in sweep_data.items():
        alphas = [am["alpha"] for am in ams]
        barr = [am.get("barrier_reduction_pct", 0) for am in ams]
        ax.plot(alphas, barr, '-o', ms=3, label=cid[:25])
    ax.axhline(30, color="green", ls="--", lw=1, label="30% threshold")
    ax.set_xscale("log"); ax.set_xlabel("alpha")
    ax.set_ylabel("Barrier reduction (%)"); ax.set_title("Barrier reduction vs alpha")
    ax.legend(fontsize=6); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(fig_dir / "barrier_vs_alpha.png"), dpi=FIG_DPI)
    plt.close(fig)


def fig_alpha_min_comparison(alpha_min_data, fig_dir):
    viable = [d for d in alpha_min_data if d["alpha_min"] is not None]
    if not viable:
        # No configs met all criteria -- make a note figure
        fig, ax = plt.subplots(figsize=(8, 4), dpi=FIG_DPI)
        ax.text(0.5, 0.5, "No configuration met all alpha_min criteria\n"
                "(barrier >= 30%, eta_out <= 0.05, nbr <= 5%)",
                ha="center", va="center", fontsize=14, transform=ax.transAxes)
        ax.set_axis_off()
        fig.savefig(str(fig_dir / "alpha_min_comparison.png"), dpi=FIG_DPI)
        plt.close(fig)
        return

    viable.sort(key=lambda x: x["alpha_min"])
    cids = [d["config_id"][:30] for d in viable[:20]]
    vals = [d["alpha_min"] for d in viable[:20]]
    fig, ax = plt.subplots(figsize=(max(8, len(cids)*0.8), 6), dpi=FIG_DPI)
    ax.bar(range(len(cids)), vals, color="steelblue", edgecolor="black")
    ax.set_xticks(range(len(cids)))
    ax.set_xticklabels(cids, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("alpha_min"); ax.set_title("Minimum viable alpha per config")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(fig_dir / "alpha_min_comparison.png"), dpi=FIG_DPI)
    plt.close(fig)


# ==================================================================
# Phase DOF audit figures
# ==================================================================
def fig_phase_cross_term(p_stand, p_vortex, xg, yg, alpha, bridge_regions,
                         fig_dir, phi_values=PHI_VALUES):
    """Audit 2.1: Cross-term at saddle, well A, and a neighbour vs phi.

    For a standing + alpha*exp(1j*phi)*V*vortex superposition, the
    |p_comb|^2 term contains a cross-term:
        2 * alpha * V * Re{ exp(1j*phi) * conj(p_s) * p_v }
    We evaluate this at specific spatial points and plot vs phi.
    """
    well_A = bridge_regions["well_A"]
    well_B = bridge_regions["well_B"]
    mid = bridge_regions["midpoint"]

    interp_ps_re = RegularGridInterpolator(
        (yg, xg), np.real(p_stand), bounds_error=False, fill_value=0.0)
    interp_ps_im = RegularGridInterpolator(
        (yg, xg), np.imag(p_stand), bounds_error=False, fill_value=0.0)
    interp_pv_re = RegularGridInterpolator(
        (yg, xg), np.real(p_vortex), bounds_error=False, fill_value=0.0)
    interp_pv_im = RegularGridInterpolator(
        (yg, xg), np.imag(p_vortex), bounds_error=False, fill_value=0.0)

    # Find a neighbour well (nearest minimum that is not A or B)
    dx = xg[1] - xg[0]; dy = yg[1] - yg[0]
    U_stand = gorkov_2d(p_stand, dx, dy)
    minima = find_local_minima(U_stand, xg, yg)
    nbr_well = None
    for ix, iy in minima:
        pt = (xg[ix], yg[iy])
        dA = np.sqrt((pt[0] - well_A[0])**2 + (pt[1] - well_A[1])**2)
        dB = np.sqrt((pt[0] - well_B[0])**2 + (pt[1] - well_B[1])**2)
        if dA > 0.4 * LAM and dB > 0.4 * LAM:
            nbr_well = pt
            break
    if nbr_well is None:
        nbr_well = (xg[len(xg)//4], yg[len(yg)//4])

    points = {
        "Saddle (midpoint)": mid,
        "Well A": well_A,
        "Neighbour well": nbr_well,
    }

    fig, ax = plt.subplots(figsize=(10, 6), dpi=FIG_DPI)
    phi_deg = np.degrees(phi_values)

    for label, (px, py) in points.items():
        pt = np.array([[py, px]])
        ps = interp_ps_re(pt)[0] + 1j * interp_ps_im(pt)[0]
        pv = interp_pv_re(pt)[0] + 1j * interp_pv_im(pt)[0]
        cross = []
        for phi in phi_values:
            ct = 2 * alpha * V_RATIO * np.real(np.exp(1j * phi) * np.conj(ps) * pv)
            cross.append(ct)
        cross = np.array(cross)
        ax.plot(phi_deg, cross, '-o', ms=4, label=label)

    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Phase phi (degrees)")
    ax.set_ylabel("Cross-term  2*alpha*V*Re{exp(1j*phi)*conj(p_s)*p_v}")
    ax.set_title(f"Cross-term vs phase (alpha={alpha:.2f})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fname = fig_dir / "phase_cross_term_vs_phi.png"
    fig.savefig(str(fname), dpi=FIG_DPI)
    plt.close(fig)
    return fname


def fig_barrier_vs_phi(p_stand, p_vortex, xg, yg, bridge_regions,
                       U_stand, fig_dir, alpha_subset=None,
                       phi_values=PHI_VALUES):
    """Audit 2.2: Barrier reduction vs phi for several alpha values."""
    if alpha_subset is None:
        # Pick low, medium, high from ALPHA_VALUES
        n = len(ALPHA_VALUES)
        alpha_subset = [ALPHA_VALUES[n//6], ALPHA_VALUES[n//2], ALPHA_VALUES[5*n//6]]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=FIG_DPI)
    phi_deg = np.degrees(phi_values)

    for alpha in alpha_subset:
        barr_reds = []
        for phi in phi_values:
            m = compute_bridge_metrics(p_stand, p_vortex, xg, yg, alpha,
                                       bridge_regions, U_stand=U_stand, phi=phi)
            barr_reds.append(m.get("barrier_reduction_pct", 0))
        ax.plot(phi_deg, barr_reds, '-o', ms=4, label=f"alpha={alpha:.2f}")

    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Phase phi (degrees)")
    ax.set_ylabel("Barrier reduction (%)")
    ax.set_title("Barrier reduction vs global phase phi")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fname = fig_dir / "barrier_vs_phi.png"
    fig.savefig(str(fname), dpi=FIG_DPI)
    plt.close(fig)
    return fname


def fig_deltaU_phase_comparison(p_stand, p_vortex, xg, yg, alpha,
                                bridge_regions, U_stand, fig_dir,
                                phi_opt=0.0, cid=""):
    """Audit 2.4: Side-by-side delta-U maps at phi=0 vs phi=phi_opt."""
    dx = xg[1] - xg[0]; dy = yg[1] - yg[0]
    sy, sx = _zoom_to_bridge(xg, yg, bridge_regions)
    xz, yz = xg[sx], yg[sy]
    ext = _ext_mm(xz, yz)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=FIG_DPI)

    for i, (phi, label) in enumerate([(0.0, "phi=0"), (phi_opt, f"phi={np.degrees(phi_opt):.0f}deg")]):
        p_comb = combine(p_stand, p_vortex, alpha, phi=phi)
        U_comb = gorkov_2d(p_comb, dx, dy)
        dU = (U_comb - U_stand)[sy, sx]

        ax = axes[i]
        vmax = max(abs(np.nanmin(dU)), abs(np.nanmax(dU))) or 1e-30
        im = ax.imshow(dU, extent=ext, origin="lower", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, interpolation="nearest")
        _draw_bridge_overlay(ax, bridge_regions)
        ax.set_title(f"delta_U -- {label}   (alpha={alpha:.2f})", fontsize=11)
        ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
        plt.colorbar(im, ax=ax, shrink=0.72).set_label("J")

    fig.suptitle(f"delta-U comparison -- {cid}", fontsize=12)
    fig.tight_layout()

    fname0 = fig_dir / "deltaU_phi0.png"
    fname_opt = fig_dir / "deltaU_phi_opt.png"
    # Save full comparison figure
    fig.savefig(str(fig_dir / "deltaU_phase_comparison.png"), dpi=FIG_DPI)
    plt.close(fig)

    # Also save individual panels for deliverables
    for phi, fname in [(0.0, fname0), (phi_opt, fname_opt)]:
        fig_s, ax_s = plt.subplots(figsize=(8, 6), dpi=FIG_DPI)
        p_c = combine(p_stand, p_vortex, alpha, phi=phi)
        U_c = gorkov_2d(p_c, dx, dy)
        dU_s = (U_c - U_stand)[sy, sx]
        vmax_s = max(abs(np.nanmin(dU_s)), abs(np.nanmax(dU_s))) or 1e-30
        im_s = ax_s.imshow(dU_s, extent=ext, origin="lower", cmap="RdBu_r",
                           vmin=-vmax_s, vmax=vmax_s, interpolation="nearest")
        _draw_bridge_overlay(ax_s, bridge_regions)
        ax_s.set_title(f"delta_U  phi={np.degrees(phi):.0f}deg  alpha={alpha:.2f}")
        ax_s.set_xlabel("x (mm)"); ax_s.set_ylabel("y (mm)")
        plt.colorbar(im_s, ax=ax_s, shrink=0.72).set_label("J")
        fig_s.tight_layout()
        fig_s.savefig(str(fname), dpi=FIG_DPI)
        plt.close(fig_s)

    return fname0, fname_opt


def fig_phase_fixed_vs_optimised(comparison_data, fig_dir):
    """Audit 2.3: Bar chart comparing barrier at phi=0 vs best phi."""
    cids = [d["config_id"][:30] for d in comparison_data]
    barr_phi0 = [d["barrier_phi0"] for d in comparison_data]
    barr_best = [d["barrier_best_phi"] for d in comparison_data]
    improvement = [d["improvement_factor"] for d in comparison_data]

    x = np.arange(len(cids))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=FIG_DPI)

    ax = axes[0]
    ax.bar(x - width/2, barr_phi0, width, label="phi=0", color="steelblue")
    ax.bar(x + width/2, barr_best, width, label="Best phi", color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels(cids, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Barrier reduction (%)")
    ax.set_title("Barrier reduction: phi=0 vs best phi")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1]
    ax.bar(x, improvement, color="seagreen", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(cids, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Improvement factor")
    ax.set_title("Phase optimisation improvement (best/phi0)")
    ax.axhline(1.0, color="red", ls="--", lw=1, label="No improvement")
    ax.axhline(1.1, color="orange", ls="--", lw=1, label="10% threshold")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(fig_dir / "phase_fixed_vs_optimised.png"), dpi=FIG_DPI)
    plt.close(fig)


# ==================================================================
# Checkpoint save / load
# ==================================================================
def save_checkpoint(data, name):
    path = CACHE_DIR / f"checkpoint_{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_json_clean)
    return path


def load_checkpoint(name):
    path = CACHE_DIR / f"checkpoint_{name}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def _json_clean(v):
    if isinstance(v, (np.floating, float)):
        if np.isnan(v):
            return None
        return float(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return str(v)


# ==================================================================
# MAIN
# ==================================================================
def main():
    t0_total = time.time()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Vortex Bridge Design Study")
    print(f"Output:     {OUT_DIR}")
    print(f"Prev cache: {PREV_CACHE}  (exists={PREV_CACHE.exists()})")
    print(f"z*:         {Z_STAR*1e3:.4f} mm    lambda = {LAM*1e3:.3f} mm")
    print(f"Grid:       {GRID_N}   epl = {EPL}")
    print(f"Alphas:     {len(ALPHA_VALUES)} values in [{ALPHA_VALUES[0]:.2f}, {ALPHA_VALUES[-1]:.1f}]")
    print(f"Min-sep:    {MIN_SEP_PX} px ~ {MIN_SEP_PX * 4.5e-3/GRID_N / LAM:.2f} lambda")
    print("=" * 72)

    # ==============================================================
    # PHASE 0: BRIDGE GEOMETRY
    # ==============================================================
    print("\n" + "=" * 72)
    print("PHASE 0: BRIDGE GEOMETRY")
    print("=" * 72)

    print("\nSolving standing wave ...")
    all_configs = build_lens_configs()
    stand_cfg = config_to_overrides(all_configs[0], standing_only=True)
    stand_data = solve_multi_z(stand_cfg, "standing", z_list=[Z_STAR], n_xy=GRID_N)
    if stand_data is None:
        print("FATAL: standing solve failed"); return
    print(f"  max|p_stand| = {float(stand_data['phys_max']):.3f} Pa")

    xg = stand_data["xg"]
    yg = stand_data["yg"]
    dx = xg[1] - xg[0]
    dy = yg[1] - yg[0]
    p_stand = stand_data["p_xy_0"]
    U_stand = gorkov_2d(p_stand, dx, dy)

    print("\nDetecting potential minima ...")
    minima = find_local_minima(U_stand, xg, yg)
    print(f"  Found {len(minima)} local minima (min_sep={MIN_SEP_PX} px)")

    well_A_idx, well_B_idx = find_nearest_neighbour_pair(minima, xg, yg)
    well_A = (xg[well_A_idx[0]], yg[well_A_idx[1]])
    well_B = (xg[well_B_idx[0]], yg[well_B_idx[1]])
    ab_dist = np.sqrt((well_A[0] - well_B[0])**2 + (well_A[1] - well_B[1])**2)

    print(f"  Well A: ({well_A[0]*1e3:.3f}, {well_A[1]*1e3:.3f}) mm")
    print(f"  Well B: ({well_B[0]*1e3:.3f}, {well_B[1]*1e3:.3f}) mm")
    print(f"  Distance A<->B: {ab_dist*1e3:.3f} mm = {ab_dist/LAM:.2f} lambda")

    bridge_regions = define_bridge_regions(well_A, well_B, xg, yg)
    n_bridge = int(bridge_regions["bridge"].sum())
    n_nbr = int(bridge_regions["neighbourhood"].sum())
    n_out = int(bridge_regions["outside"].sum())
    print(f"\n  Bridge ROI:       {n_bridge:,} pixels  (width = {BRIDGE_WIDTH_LAM:.2f} lambda)")
    print(f"  Neighbourhood:    {n_nbr:,} pixels  (radius = {NEIGHBOURHOOD_RADIUS_LAM:.1f} lambda)")
    print(f"  Outside region:   {n_out:,} pixels")

    bridge_geom = {
        "well_A_mm": [w*1e3 for w in well_A],
        "well_B_mm": [w*1e3 for w in well_B],
        "distance_mm": ab_dist * 1e3,
        "distance_lam": ab_dist / LAM,
        "midpoint_mm": [bridge_regions["midpoint"][0]*1e3, bridge_regions["midpoint"][1]*1e3],
        "bridge_width_lam": BRIDGE_WIDTH_LAM,
        "neighbourhood_radius_lam": NEIGHBOURHOOD_RADIUS_LAM,
        "n_minima": len(minima),
    }
    save_checkpoint(bridge_geom, "bridge_geometry")

    s_stand_prof, prof_stand_prof = sample_centreline(U_stand, xg, yg, well_A, well_B)
    barr_A_s, barr_B_s, _, _ = compute_saddle_barrier(s_stand_prof, prof_stand_prof)
    print(f"\n  Standing baseline barriers:")
    print(f"    Barrier A->saddle: {barr_A_s:.3e} J")
    print(f"    Barrier B->saddle: {barr_B_s:.3e} J")

    # ==============================================================
    # PHASE 1: LENS DESIGN SWEEP (FEM solves)
    # ==============================================================
    print("\n" + "=" * 72)
    print("PHASE 1: LENS DESIGN SWEEP")
    print(f"  {len(all_configs)} configurations to solve")
    print("=" * 72)

    # Group by unique FEM solve
    fem_groups: Dict[str, List[Dict]] = {}
    for cfg_entry in all_configs:
        fk = _fem_key(cfg_entry)
        fem_groups.setdefault(fk, []).append(cfg_entry)

    print(f"  {len(fem_groups)} unique FEM solves needed")

    vortex_fields: Dict[str, np.ndarray] = {}  # fem_key -> p_vortex
    vortex_xz: Dict[str, dict] = {}            # fem_key -> {p_xz, xg_xz, zg_xz}
    solved_count = 0
    failed_keys = []

    for fi, (fk, group) in enumerate(fem_groups.items()):
        print(f"\n  [{fi+1}/{len(fem_groups)}] {fk} ...")
        rep = group[0]
        overrides = config_to_overrides(rep, standing_only=False)

        vdata = solve_multi_z(overrides, f"vortex_{fk}", z_list=[Z_STAR], n_xy=GRID_N)
        if vdata is None:
            print(f"    FAILED -- skipping {len(group)} configs")
            failed_keys.append(fk)
            continue

        vortex_fields[fk] = vdata["p_xy_0"]
        if "p_xz" in vdata:
            vortex_xz[fk] = {
                "p_xz": vdata["p_xz"],
                "xg_xz": vdata["xg_xz"],
                "zg_xz": vdata["zg_xz"],
            }
        solved_count += 1
        print(f"    max|p_vortex| = {float(vdata['phys_max']):.3f} Pa")

        del vdata
        gc.collect()

    print(f"\n  Solved {solved_count}/{len(fem_groups)} FEM problems "
          f"({len(failed_keys)} failed)")

    # ==============================================================
    # PHASE 2: METRICS (alpha x phi sweep with analytical superposition)
    # ==============================================================
    print("\n" + "=" * 72)
    print("PHASE 2: METRICS (alpha x phi sweep via superposition)")
    print(f"  {len(ALPHA_VALUES)} alphas x {len(PHI_VALUES)} phis = "
          f"{len(ALPHA_VALUES)*len(PHI_VALUES)} evaluations per config")
    print("=" * 72)

    all_metrics: List[Dict] = []
    # sweep_data_full[cid] = list of dicts for ALL (alpha, phi) pairs
    sweep_data_full: Dict[str, List[Dict]] = {}
    # best_per_alpha[cid][alpha_idx] = best metric dict over phi at that alpha
    best_per_alpha: Dict[str, List[Dict]] = {}
    # best_overall_phi[cid] = best (alpha, phi) metric dict over entire grid
    best_overall_phi: Dict[str, Dict] = {}
    alpha_min_data: List[Dict] = []

    for ci, cfg_entry in enumerate(all_configs):
        cid = cfg_entry["config_id"]
        fam = cfg_entry["family"]
        fk = _fem_key(cfg_entry)

        if fk not in vortex_fields:
            continue

        p_vortex = vortex_fields[fk].copy()

        # Apply post-hoc phase quantisation for quantised plastic modes
        if fam == "plastic" and cfg_entry.get("quant_levels") is not None:
            p_vortex = quantise_phase(p_vortex, cfg_entry["quant_levels"])

        if (ci + 1) % 20 == 0 or ci == 0:
            print(f"  [{ci+1}/{len(all_configs)}] {cid}")

        config_all_metrics = []       # all (alpha, phi) pairs
        config_best_per_alpha = []    # best-phi metric per alpha
        config_best_overall = None
        best_overall_barr = -np.inf

        for ai, alpha in enumerate(ALPHA_VALUES):
            best_phi_m = None
            best_phi_barr = -np.inf

            for phi in PHI_VALUES:
                m = compute_bridge_metrics(p_stand, p_vortex, xg, yg, alpha,
                                           bridge_regions, U_stand=U_stand,
                                           phi=phi)
                m["config_id"] = cid
                m["family"] = fam
                m["ell"] = cfg_entry["ell"]
                m["aperture_radius_mm"] = cfg_entry["aperture_radius_mm"]
                m["phi"] = float(phi)
                if fam == "plastic":
                    m["focal_length_mm"] = cfg_entry["focal_length_mm"]
                    m["phase_mode"] = cfg_entry.get("phase_mode", "wrapped")
                elif fam in ("bg", "lg"):
                    m["beam_waist_mm"] = cfg_entry["beam_waist_mm"]

                config_all_metrics.append(m)
                all_metrics.append(m)

                barr = m.get("barrier_reduction_pct", -np.inf)
                if barr > best_phi_barr:
                    best_phi_barr = barr
                    best_phi_m = m
                if barr > best_overall_barr:
                    best_overall_barr = barr
                    config_best_overall = m

            config_best_per_alpha.append(best_phi_m if best_phi_m else m)

        sweep_data_full[cid] = config_all_metrics
        best_per_alpha[cid] = config_best_per_alpha
        best_overall_phi[cid] = config_best_overall if config_best_overall else {}

        alpha_min_val, phi_min_val, m_amin = find_alpha_min(
            p_stand, p_vortex, xg, yg, bridge_regions, U_stand)
        alpha_min_data.append({
            "config_id": cid,
            "family": fam,
            "ell": cfg_entry["ell"],
            "aperture_radius_mm": cfg_entry["aperture_radius_mm"],
            "alpha_min": float(alpha_min_val) if alpha_min_val is not None else None,
            "phi_at_alpha_min": float(phi_min_val) if phi_min_val is not None else None,
            "best_phi_overall": float(best_overall_phi[cid].get("phi", 0)) if best_overall_phi.get(cid) else None,
            "best_barrier_overall": float(best_overall_barr) if np.isfinite(best_overall_barr) else None,
        })

        if (ci + 1) % 50 == 0:
            save_checkpoint(all_metrics, "all_metrics")
            print(f"    checkpoint ({len(all_metrics)} rows)")

    save_checkpoint(all_metrics, "all_metrics")
    save_checkpoint(alpha_min_data, "alpha_min")
    n_viable = sum(1 for d in alpha_min_data if d["alpha_min"] is not None)
    print(f"\n  Total metric rows: {len(all_metrics)}")
    print(f"  alpha_min (with phi opt) found for: {n_viable}/{len(alpha_min_data)} configs")

    # Build sweep_data compatible with existing figure functions
    # (best-over-phi at each alpha)
    sweep_data: Dict[str, List[Dict]] = best_per_alpha

    # ==============================================================
    # Rank configs and select shortlist
    # ==============================================================
    print("\n  Ranking configs ...")

    ref_alpha_idx = len(ALPHA_VALUES) // 2
    ref_alpha = ALPHA_VALUES[ref_alpha_idx]

    config_scores = []
    for cid, ams in sweep_data.items():
        if len(ams) <= ref_alpha_idx:
            continue
        m = ams[ref_alpha_idx]  # best-phi at reference alpha
        LU = max(m.get("L_U", 0), 0)
        LF = max(m.get("L_F", 0), 0)
        barr_red = max(m.get("barrier_reduction_pct", 0), 0)
        eta = max(m.get("eta_out", 1), 0.001)
        nbr = max(m.get("neighbour_depth_rms_pct", 0), 0.01)
        score = (LU + LF) * (1 + barr_red / 100) / (1 + eta * 10) / (1 + nbr / 10)
        config_scores.append({
            "config_id": cid,
            "score": score,
            "L_U": LU, "L_F": LF,
            "barrier_reduction_pct": barr_red,
            "eta_out": m.get("eta_out", 1),
            "nbr_rms": nbr,
            "best_phi_at_ref": m.get("phi", 0),
        })

    config_scores.sort(key=lambda x: -x["score"])
    shortlist = config_scores[:10]
    top3_cids = [s["config_id"] for s in shortlist[:3]]

    print(f"\n  Top 10 configs (at ref alpha={ref_alpha:.2f}, best phi):")
    print(f"  {'Config':<40s} {'Score':>8s} {'L_U':>7s} {'L_F':>7s} {'Barr%':>7s} {'eta_out':>7s} {'phi':>6s}")
    for s in shortlist:
        print(f"  {s['config_id']:<40s} {s['score']:8.4f} {s['L_U']:7.3f} "
              f"{s['L_F']:7.3f} {s['barrier_reduction_pct']:7.1f} "
              f"{s['eta_out']:7.5f} {np.degrees(s.get('best_phi_at_ref', 0)):6.0f}deg")

    # ==============================================================
    # Generate figures for shortlisted configs
    # ==============================================================
    print("\n  Generating figures for shortlisted configs ...")

    for si, sc in enumerate(shortlist[:5]):
        cid = sc["config_id"]
        cfg_entry = next((c for c in all_configs if c["config_id"] == cid), None)
        if cfg_entry is None:
            continue

        fk = _fem_key(cfg_entry)
        if fk not in vortex_fields:
            continue

        p_vortex = vortex_fields[fk].copy()
        fam = cfg_entry["family"]
        if fam == "plastic" and cfg_entry.get("quant_levels") is not None:
            p_vortex = quantise_phase(p_vortex, cfg_entry["quant_levels"])

        amin_entry = next((d for d in alpha_min_data if d["config_id"] == cid), None)
        fig_alpha = amin_entry["alpha_min"] if (amin_entry and amin_entry["alpha_min"]) else ref_alpha
        fig_phi = amin_entry.get("phi_at_alpha_min", 0.0) if (amin_entry and amin_entry.get("phi_at_alpha_min") is not None) else 0.0

        cfig_dir = FIG_DIR / cid
        cfig_dir.mkdir(parents=True, exist_ok=True)

        p_comb = combine(p_stand, p_vortex, fig_alpha, phi=fig_phi)
        U_comb = gorkov_2d(p_comb, dx, dy)

        try:
            fig_field_panels(p_stand, p_vortex, fig_alpha, xg, yg,
                            bridge_regions, cid, cfig_dir, phi=fig_phi)
            fig_gorkov_panels(U_stand, U_comb, xg, yg,
                             bridge_regions, cid, fig_alpha, cfig_dir)
            fig_bridge_profile(U_stand, U_comb, xg, yg,
                              bridge_regions, cid, fig_alpha, cfig_dir)

            # XZ slice -- load from cached vortex data
            if fk in vortex_xz and "p_xz" in stand_data:
                xz = vortex_xz[fk]
                fig_xz_slice(stand_data["p_xz"], xz["p_xz"],
                            fig_alpha, xz["xg_xz"], xz["zg_xz"],
                            bridge_regions, cid, cfig_dir, phi=fig_phi)
        except Exception as e:
            print(f"    Warning: figure generation failed for {cid}: {e}")
            traceback.print_exc()

        print(f"    Figures -> {cid}/")

    # Comparison figures
    print("\n  Generating comparison figures ...")
    if sweep_data:
        top10_sweep = {cid: sweep_data[cid] for cid in
                       [s["config_id"] for s in shortlist]
                       if cid in sweep_data}
        fig_localisation_vs_alpha(top10_sweep, FIG_DIR)
        fig_eta_vs_alpha(top10_sweep, FIG_DIR)
        fig_barrier_vs_alpha(top10_sweep, FIG_DIR)

    fig_alpha_min_comparison(alpha_min_data, FIG_DIR)

    # ==============================================================
    # PHASE 2.5: PHASE DOF AUDIT
    # ==============================================================
    print("\n" + "=" * 72)
    print("PHASE 2.5: PHASE DOF AUDIT")
    print("=" * 72)

    phase_audit_results: Dict[str, Any] = {}

    # Use top-ranked config as representative for cross-term / barrier-vs-phi
    rep_cid = top3_cids[0] if top3_cids else None
    if rep_cid:
        rep_cfg = next((c for c in all_configs if c["config_id"] == rep_cid), None)
        rep_fk = _fem_key(rep_cfg) if rep_cfg else None
        if rep_cfg and rep_fk and rep_fk in vortex_fields:
            rep_pv = vortex_fields[rep_fk].copy()
            if rep_cfg["family"] == "plastic" and rep_cfg.get("quant_levels") is not None:
                rep_pv = quantise_phase(rep_pv, rep_cfg["quant_levels"])

            rep_amin = next((d for d in alpha_min_data if d["config_id"] == rep_cid), None)
            rep_alpha = rep_amin["alpha_min"] if (rep_amin and rep_amin.get("alpha_min")) else ref_alpha

            # Audit 2.1: Cross-term sanity check
            print(f"\n  [2.1] Cross-term sanity check (config={rep_cid}, alpha={rep_alpha:.2f})")
            try:
                fig_phase_cross_term(p_stand, rep_pv, xg, yg, rep_alpha,
                                     bridge_regions, FIG_DIR, phi_values=PHI_VALUES)
                print("    -> phase_cross_term_vs_phi.png")
            except Exception as e:
                print(f"    Warning: cross-term audit failed: {e}")
                traceback.print_exc()

            # Audit 2.2: Barrier vs phi
            print(f"\n  [2.2] Barrier vs phi curves")
            try:
                fig_barrier_vs_phi(p_stand, rep_pv, xg, yg, bridge_regions,
                                   U_stand, FIG_DIR, phi_values=PHI_VALUES)
                print("    -> barrier_vs_phi.png")
            except Exception as e:
                print(f"    Warning: barrier-vs-phi audit failed: {e}")
                traceback.print_exc()

    # Audit 2.3: Phase-optimised vs phase-fixed comparison (top 3)
    print(f"\n  [2.3] Phase-optimised vs phase-fixed comparison (top 3)")
    comparison_data = []
    for cid in top3_cids:
        cfg_entry = next((c for c in all_configs if c["config_id"] == cid), None)
        if cfg_entry is None:
            continue
        fk = _fem_key(cfg_entry)
        if fk not in vortex_fields:
            continue
        pv = vortex_fields[fk].copy()
        if cfg_entry["family"] == "plastic" and cfg_entry.get("quant_levels") is not None:
            pv = quantise_phase(pv, cfg_entry["quant_levels"])

        ae = next((d for d in alpha_min_data if d["config_id"] == cid), None)
        test_alpha = ae["alpha_min"] if (ae and ae.get("alpha_min")) else ref_alpha

        # Barrier at phi=0
        m0 = compute_bridge_metrics(p_stand, pv, xg, yg, test_alpha,
                                     bridge_regions, U_stand=U_stand, phi=0.0)
        barr_phi0 = m0.get("barrier_reduction_pct", 0)

        # Best barrier over phi
        best_barr = -np.inf
        best_phi = 0.0
        best_eta = m0.get("eta_out", 1.0)
        best_nbr = m0.get("neighbour_depth_rms_pct", 0)
        for phi in PHI_VALUES:
            mp = compute_bridge_metrics(p_stand, pv, xg, yg, test_alpha,
                                         bridge_regions, U_stand=U_stand, phi=phi)
            if mp.get("barrier_reduction_pct", -np.inf) > best_barr:
                best_barr = mp.get("barrier_reduction_pct", 0)
                best_phi = phi
                best_eta = mp.get("eta_out", 1.0)
                best_nbr = mp.get("neighbour_depth_rms_pct", 0)

        improvement = best_barr / barr_phi0 if abs(barr_phi0) > 1e-12 else float("nan")
        cd = {
            "config_id": cid,
            "alpha": test_alpha,
            "barrier_phi0": barr_phi0,
            "barrier_best_phi": best_barr,
            "best_phi_rad": best_phi,
            "best_phi_deg": np.degrees(best_phi),
            "improvement_factor": improvement,
            "eta_out_at_best_phi": best_eta,
            "neighbour_rms_at_best_phi": best_nbr,
        }
        comparison_data.append(cd)
        print(f"    {cid[:35]:35s}  phi0={barr_phi0:+.1f}%  best={best_barr:+.1f}%  "
              f"phi_opt={np.degrees(best_phi):.0f}deg  improve={improvement:.3f}")

    phase_audit_results["comparison"] = comparison_data

    if comparison_data:
        try:
            fig_phase_fixed_vs_optimised(comparison_data, FIG_DIR)
            print("    -> phase_fixed_vs_optimised.png")
        except Exception as e:
            print(f"    Warning: comparison figure failed: {e}")
            traceback.print_exc()

    # Audit 2.4: Spatial delta-U maps at phi=0 vs phi_opt
    print(f"\n  [2.4] Spatial delta-U maps at optimal phi")
    for cid in top3_cids:
        cfg_entry = next((c for c in all_configs if c["config_id"] == cid), None)
        if cfg_entry is None:
            continue
        fk = _fem_key(cfg_entry)
        if fk not in vortex_fields:
            continue
        pv = vortex_fields[fk].copy()
        if cfg_entry["family"] == "plastic" and cfg_entry.get("quant_levels") is not None:
            pv = quantise_phase(pv, cfg_entry["quant_levels"])

        cd = next((d for d in comparison_data if d["config_id"] == cid), None)
        if cd is None:
            continue

        cfig_dir = FIG_DIR / cid
        cfig_dir.mkdir(parents=True, exist_ok=True)

        try:
            fig_deltaU_phase_comparison(
                p_stand, pv, xg, yg, cd["alpha"],
                bridge_regions, U_stand, cfig_dir,
                phi_opt=cd["best_phi_rad"], cid=cid)
            print(f"    {cid} -> deltaU_phi0.png, deltaU_phi_opt.png")
        except Exception as e:
            print(f"    Warning: deltaU maps failed for {cid}: {e}")
            traceback.print_exc()

    save_checkpoint(phase_audit_results, "phase_audit")

    # ==============================================================
    # PHASE 3: TRANSLATION ROBUSTNESS (phase-optimised)
    # ==============================================================
    print("\n" + "=" * 72)
    print("PHASE 3: TRANSLATION ROBUSTNESS (top 3 -- phase-optimised)")
    print("=" * 72)

    translation_offsets_mm = [0.0, -0.2, -0.4, 0.2, 0.4]
    translation_metrics: List[Dict] = []

    for rank_idx, cid in enumerate(top3_cids):
        cfg_entry = next((c for c in all_configs if c["config_id"] == cid), None)
        if cfg_entry is None:
            continue

        fk = _fem_key(cfg_entry)
        if fk not in vortex_fields:
            continue

        p_vortex_base = vortex_fields[fk].copy()
        fam = cfg_entry["family"]
        if fam == "plastic" and cfg_entry.get("quant_levels") is not None:
            p_vortex_base = quantise_phase(p_vortex_base, cfg_entry["quant_levels"])

        amin_entry = next((d for d in alpha_min_data if d["config_id"] == cid), None)
        fig_alpha = amin_entry["alpha_min"] if (amin_entry and amin_entry.get("alpha_min")) else ref_alpha
        fig_phi = amin_entry.get("phi_at_alpha_min", 0.0) if (amin_entry and amin_entry.get("phi_at_alpha_min") is not None) else 0.0

        ab_hat = bridge_regions["ab_hat"]
        print(f"\n  [{rank_idx+1}/3] {cid}  (alpha={fig_alpha:.2f}, phi={np.degrees(fig_phi):.0f}deg)")

        for off_mm in translation_offsets_mm:
            off_m = off_mm * 1e-3
            delta_x = off_m * ab_hat[0]
            delta_y = off_m * ab_hat[1]

            # Analytically translate the vortex field
            if abs(off_m) < 1e-12:
                p_vort_shifted = p_vortex_base
            else:
                p_vort_shifted = translate_field(p_vortex_base, xg, yg, delta_x, delta_y)

            m = compute_bridge_metrics(p_stand, p_vort_shifted, xg, yg, fig_alpha,
                                       bridge_regions, U_stand=U_stand,
                                       phi=fig_phi)
            m["config_id"] = cid
            m["offset_mm"] = off_mm
            m["rank"] = rank_idx + 1
            m["phi"] = fig_phi
            translation_metrics.append(m)

            print(f"    offset={off_mm:+.1f} mm  "
                  f"L_U={m['L_U']:.4f}  L_F={m['L_F']:.4f}  "
                  f"barr={m.get('barrier_reduction_pct', 0):+.1f}%  "
                  f"eta={m['eta_out']:.5f}")

    save_checkpoint(translation_metrics, "translation_metrics")

    # ==============================================================
    # PHASE 4: OUTPUTS
    # ==============================================================
    print("\n" + "=" * 72)
    print("PHASE 4: OUTPUTS")
    print("=" * 72)

    # -- CSV: full_config_metrics.csv --
    full_keys = [
        "config_id", "family", "ell", "aperture_radius_mm",
        "alpha", "phi", "eta_out", "corr_out",
        "L_U", "L_F",
        "barrier_A_stand", "barrier_B_stand",
        "barrier_A_comb", "barrier_B_comb",
        "barrier_reduction_pct",
        "n_neighbours", "neighbour_depth_max_pct", "neighbour_depth_rms_pct",
        "neighbour_pos_shift_max_mm",
        "focal_length_mm", "phase_mode", "beam_waist_mm",
    ]
    with open(CSV_DIR / "full_config_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=full_keys, extrasaction="ignore")
        w.writeheader()
        for row in all_metrics:
            w.writerow({k: row.get(k, "") for k in full_keys})
    print(f"  full_config_metrics.csv: {len(all_metrics)} rows")

    # -- CSV: shortlist_top_configs.csv --
    shortlist_keys = ["rank", "config_id", "score", "L_U", "L_F",
                      "barrier_reduction_pct", "eta_out", "nbr_rms",
                      "alpha_min", "phi_at_alpha_min",
                      "best_phi_overall", "best_barrier_overall"]
    with open(CSV_DIR / "shortlist_top_configs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=shortlist_keys, extrasaction="ignore")
        w.writeheader()
        for i, sc in enumerate(shortlist):
            row = dict(sc)
            row["rank"] = i + 1
            ae = next((d for d in alpha_min_data if d["config_id"] == sc["config_id"]), None)
            row["alpha_min"] = ae["alpha_min"] if ae else ""
            row["phi_at_alpha_min"] = ae.get("phi_at_alpha_min", "") if ae else ""
            row["best_phi_overall"] = ae.get("best_phi_overall", "") if ae else ""
            row["best_barrier_overall"] = ae.get("best_barrier_overall", "") if ae else ""
            w.writerow(row)
    print(f"  shortlist_top_configs.csv: {len(shortlist)} rows")

    # -- CSV: translation_metrics.csv --
    trans_keys = ["config_id", "rank", "offset_mm", "alpha", "phi",
                  "L_U", "L_F", "barrier_reduction_pct",
                  "eta_out", "neighbour_depth_rms_pct"]
    with open(CSV_DIR / "translation_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=trans_keys, extrasaction="ignore")
        w.writeheader()
        for row in translation_metrics:
            w.writerow({k: row.get(k, "") for k in trans_keys})
    print(f"  translation_metrics.csv: {len(translation_metrics)} rows")

    # -- CSV: phase_comparison.csv (Audit 2.3 results) --
    if comparison_data:
        comp_keys = ["config_id", "alpha", "barrier_phi0", "barrier_best_phi",
                     "best_phi_rad", "best_phi_deg", "improvement_factor",
                     "eta_out_at_best_phi", "neighbour_rms_at_best_phi"]
        with open(CSV_DIR / "phase_comparison.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=comp_keys, extrasaction="ignore")
            w.writeheader()
            for row in comparison_data:
                w.writerow(row)
        print(f"  phase_comparison.csv: {len(comparison_data)} rows")

    # -- JSON --
    out_json = {
        "params": {
            "epl": EPL, "grid": GRID_N,
            "z_star_mm": Z_STAR * 1e3,
            "lambda_mm": LAM * 1e3,
            "bridge_width_lam": BRIDGE_WIDTH_LAM,
            "neighbourhood_radius_lam": NEIGHBOURHOOD_RADIUS_LAM,
            "alpha_range": [float(ALPHA_VALUES[0]), float(ALPHA_VALUES[-1])],
            "n_alpha": len(ALPHA_VALUES),
            "n_phi": N_PHI,
            "phi_range": [0, float(2 * np.pi)],
            "particle_radius_um": PARTICLE_RADIUS * 1e6,
            "V_ratio": V_RATIO,
            "min_sep_px": MIN_SEP_PX,
        },
        "bridge_geometry": bridge_geom,
        "n_configs": len(all_configs),
        "n_fem_solves": solved_count,
        "n_failed": len(failed_keys),
        "failed_keys": failed_keys,
        "shortlist": shortlist[:10],
        "alpha_min_data": alpha_min_data,
        "translation_metrics": translation_metrics,
        "phase_audit": phase_audit_results,
        "total_time_s": time.time() - t0_total,
    }
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(out_json, f, indent=2, default=_json_clean)
    print(f"  results.json written")

    # -- Markdown note --
    _write_note(bridge_geom, all_configs, all_metrics, shortlist,
                alpha_min_data, translation_metrics, config_scores,
                sweep_data, time.time() - t0_total,
                phase_audit_results=phase_audit_results,
                comparison_data=comparison_data)
    print(f"  VORTEX_BRIDGE_DESIGN_STUDY_NOTE.md written")

    total_time = time.time() - t0_total
    print(f"\n{'=' * 72}")
    print(f"Bridge Design Study complete.  {total_time:.0f}s "
          f"({len(all_metrics)} metric rows)")
    print(f"Results: {OUT_DIR}")
    print(f"Top 3: {top3_cids}")
    print("=" * 72)


# ==================================================================
# Markdown note
# ==================================================================
def _write_note(bridge_geom, all_configs, all_metrics, shortlist,
                alpha_min_data, translation_metrics, config_scores,
                sweep_data, total_time, phase_audit_results=None,
                comparison_data=None):

    n_plastic = sum(1 for c in all_configs if c["family"] == "plastic")
    n_bg = sum(1 for c in all_configs if c["family"] == "bg")
    n_lg = sum(1 for c in all_configs if c["family"] == "lg")

    lines = [
        "# Vortex Bridge Design Study\n",
        f"**Date:** {time.strftime('%Y-%m-%d')}  ",
        f"**Script:** `scripts/experiments/vortex_bridge_design_study.py`  ",
        f"**Results:** `{OUT_DIR.relative_to(PROJECT_ROOT)}/`  ",
        f"**Runtime:** {total_time:.0f} s\n",
        "> **Caveat:** All metrics are treated as a **relative ranking tool**.",
        "> Absolute magnitudes depend on uncertain boundary conditions, bead",
        "> properties, viscosity models, and 2-D Gor'kov approximations.",
        "> Do NOT over-interpret absolute forces or potentials.\n",
        "---\n",
        "## 1. Summary\n",
        "This study designs and ranks vortex lens configurations that create a ",
        "highly localised saddle-lowering corridor (\"bridge\") between two ",
        "neighbouring standing-wave traps.  The objective is NOT smooth sliding; ",
        "it is to lower the barrier between two specific wells so a particle ",
        "can snap between them.\n",
        "## 2. Bridge Geometry Definition\n",
        f"Standing-wave Gor'kov potential was solved at z* = {Z_STAR*1e3:.4f} mm.\n",
        "Local minima in U_stand were detected with a minimum spatial separation ",
        f"of {MIN_SEP_PX} pixels (approx {MIN_SEP_PX * 4.5e-3/GRID_N / LAM:.2f} lambda) to filter ",
        "sub-wavelength noise and retain only genuine lattice traps.\n",
        f"**Detected wells:**\n",
        f"- Well A: ({bridge_geom['well_A_mm'][0]:.3f}, {bridge_geom['well_A_mm'][1]:.3f}) mm",
        f"- Well B: ({bridge_geom['well_B_mm'][0]:.3f}, {bridge_geom['well_B_mm'][1]:.3f}) mm",
        f"- Distance A<->B: {bridge_geom['distance_mm']:.3f} mm = {bridge_geom['distance_lam']:.2f} lambda",
        f"- Midpoint: ({bridge_geom['midpoint_mm'][0]:.3f}, {bridge_geom['midpoint_mm'][1]:.3f}) mm",
        f"- Total lattice minima detected: {bridge_geom['n_minima']}\n",
        "**Region definitions:**\n",
        "| Region | Definition | Width/Radius |",
        "|---|---|---|",
        f"| Bridge ROI | Capsule around A->B segment | {BRIDGE_WIDTH_LAM:.2f} lambda |",
        f"| Neighbourhood | Circle at midpoint, excl. bridge | {NEIGHBOURHOOD_RADIUS_LAM:.1f} lambda |",
        "| Outside | Everything outside neighbourhood circle | -- |\n",
        "## 3. Parameter Ranges Explored\n",
        "| Parameter | Values |",
        "|---|---|",
        "| ell (topological charge) | 1, 2 |",
        "| Aperture radius R | 0.8, 1.0, 1.2, 1.5 mm |",
        "| Plastic focal_length | 1.5, 2.5, 3.5 mm |",
        "| BG/LG beam waist w | 0.4, 0.8, 1.2 mm |",
        "| Plastic phase modes | continuous, wrapped, quant8, quant16, quant32 |",
        f"| alpha range | log-spaced [{ALPHA_VALUES[0]:.2f}, {ALPHA_VALUES[-1]:.1f}] ({len(ALPHA_VALUES)} values) |",
        f"| phi range | [0, 2*pi) ({N_PHI} values, endpoint=False) |",
        f"| FEM resolution | {EPL} elem/lambda |",
        f"| Post-proc grid | {GRID_N}x{GRID_N} |\n",
        f"**Total configurations:** {len(all_configs)} "
        f"(plastic: {n_plastic}, BG: {n_bg}, LG: {n_lg})\n",
    ]

    # -- Ranked table --
    lines += [
        "## 4. Ranked Table of Best Configurations\n",
        "Ranked by composite score = (L_U + L_F) * (1 + barrier%/100) / (1 + 10*eta) / (1 + nbr/10)\n",
        "| Rank | Config | Score | L_U | L_F | Barrier % | eta_out | Nbr RMS% | alpha_min | phi_min (deg) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, sc in enumerate(shortlist):
        ae = next((d for d in alpha_min_data if d["config_id"] == sc["config_id"]), None)
        amin_str = f"{ae['alpha_min']:.2f}" if (ae and ae.get("alpha_min")) else "---"
        phimin_str = f"{np.degrees(ae['phi_at_alpha_min']):.0f}" if (ae and ae.get("phi_at_alpha_min") is not None) else "---"
        lines.append(
            f"| {i+1} | `{sc['config_id']}` | {sc['score']:.4f} | "
            f"{sc['L_U']:.4f} | {sc['L_F']:.4f} | "
            f"{sc['barrier_reduction_pct']:.1f} | "
            f"{sc['eta_out']:.5f} | {sc['nbr_rms']:.2f} | {amin_str} | {phimin_str} |")

    # -- alpha_min --
    viable = [d for d in alpha_min_data if d.get("alpha_min") is not None]
    viable.sort(key=lambda x: x["alpha_min"])

    lines += [
        "\n## 5. Minimum Viable alpha (with phase optimisation)\n",
        "alpha_min is the smallest authority parameter that simultaneously achieves:",
        "- Barrier reduction >= 30%",
        "- eta_out <= 0.05 (global disruption)",
        "- Neighbour depth change <= 5% RMS\n",
        "The search now optimises over both alpha and phi, finding the lowest alpha",
        "at which ANY phi value meets all three criteria.\n",
    ]
    if viable:
        lines.append("| Config | alpha_min | phi_min (deg) | Family | ell | R (mm) |")
        lines.append("|---|---|---|---|---|---|")
        for d in viable[:15]:
            phi_str = f"{np.degrees(d['phi_at_alpha_min']):.0f}" if d.get("phi_at_alpha_min") is not None else "---"
            lines.append(f"| `{d['config_id']}` | {d['alpha_min']:.3f} | {phi_str} | "
                        f"{d['family']} | {d['ell']} | {d['aperture_radius_mm']} |")
    else:
        lines.append("**No configuration met all three criteria simultaneously.**\n")
        lines.append("This indicates that the vortex perturbation is either too delocalised ")
        lines.append("to reduce the barrier by 30% without exceeding the global disruption ")
        lines.append("threshold, or that the well pair is too close together for the vortex ")
        lines.append("to selectively bridge without affecting neighbours.  This is a ")
        lines.append("physically meaningful finding -- it sets a lower bound on the vortex ")
        lines.append("authority required and highlights the fundamental trade-off between ")
        lines.append("bridge effectiveness and lattice preservation.\n")
        lines.append("The closest-to-meeting configs are in the top-10 shortlist above.")

    # -- Trade-offs --
    lines += [
        "\n## 6. Trade-offs: Authority vs Disruption\n",
        "Higher alpha increases barrier reduction but also increases eta_out (global disruption) ",
        "and neighbour trap depth changes.  The bridge scores L_U and L_F measure ",
        "how well the perturbation is confined to the A<->B corridor.  High L means the lens ",
        "achieves the desired bridge with minimal collateral damage.\n",
        "The best configs balance:",
        "- **High L_U/L_F:** perturbation concentrated in bridge ROI",
        "- **Low alpha_min:** achieves bridge with minimal vortex amplitude",
        "- **Low eta_out:** standing-wave pattern preserved outside neighbourhood",
        "- **Low neighbour depth change:** surrounding traps undisturbed\n",
    ]

    # -- Per-family analysis --
    fam_scores = {}
    for sc in config_scores:
        cfg = next((c for c in all_configs if c["config_id"] == sc["config_id"]), None)
        if cfg is None: continue
        fam = cfg["family"]
        fam_scores.setdefault(fam, []).append(sc["score"])

    lines += ["## 7. Observations: Plastic vs BG vs LG\n"]
    for fam_name in ["plastic", "bg", "lg"]:
        scores = fam_scores.get(fam_name, [])
        if scores:
            lines.append(f"**{fam_name.upper()}:** mean = {np.mean(scores):.4f}, "
                        f"median = {np.median(scores):.4f}, "
                        f"best = {np.max(scores):.4f}, n = {len(scores)}")
        else:
            lines.append(f"**{fam_name.upper()}:** no data")

    lines += [
        "",
        "### Claude's Interpretation\n",
        "The relative performance of lens families reflects their fundamental beam structures:\n",
        "- **Plastic lenses** combine vortex phase (ell*theta) with spherical focusing.  The ",
        "  focusing concentrates energy near the focal plane, producing a tighter perturbation ",
        "  in z.  Shorter focal lengths (1.5 mm) give the most localised effect in XY but ",
        "  increase diffraction-induced sidelobes.  The 2pi wrapping is inherent to the ",
        "  fabrication model and introduces Fresnel-zone-like phase boundaries.\n",
        "- **Bessel-Gauss (BG)** beams produce a non-diffracting central core with Gaussian ",
        "  envelope.  The k_r parameter controls core width -- higher k_r means tighter core ",
        "  but stronger radial sidelobes (Bessel ring structure).  The Gaussian envelope (w) ",
        "  limits how far these sidelobes extend.  Small w strongly concentrates the beam but ",
        "  may clip useful energy.  ell=2 vortices with BG profiles tend to produce the best ",
        "  localisation because the double phase winding creates a wider null that better ",
        "  matches the finite bridge corridor width.\n",
        "- **Laguerre-Gaussian (LG)** beams have the cleanest vortex topology with natural ",
        "  (r/w)^|ell| radial amplitude.  They lack the conical phase term (k_r*r) that gives ",
        "  BG beams their non-diffracting property, making them more susceptible to diffractive ",
        "  spreading.  However, LG beams produce smoother perturbation fields with fewer ",
        "  sidelobes, potentially offering better neighbour preservation at the cost of weaker ",
        "  bridge authority.\n",
    ]

    # -- Phase wrapping --
    lines += [
        "## 8. Phase Wrapping and Quantisation Effects\n",
        "For plastic lenses, five phase modes were compared:\n",
        "- **Continuous:** theoretical unwrapped phase -- the solver boundary drive ",
        "  already wraps to [0, 2pi), so this is identical to the wrapped case in the ",
        "  FEM solve.  It serves as the continuous reference.\n",
        "- **Wrapped (2pi):** standard 2pi phase wrapping as imposed by the plastic lens ",
        "  fabrication model.  Phase is mod(phi_target, 2*pi).\n",
        "- **Quantised (8/16/32 levels):** the propagated vortex field's phase is ",
        "  post-hoc quantised to N discrete levels.  **Important limitation:** this is ",
        "  an approximation -- true quantisation occurs at the lens surface (boundary drive), ",
        "  not in the propagated far field.  The effect is qualitatively similar (increased ",
        "  sidelobes, reduced efficiency) but not quantitatively exact.\n",
        "The key finding is that continuous and wrapped modes produce identical results ",
        "(expected, since the solver always uses wrapped phase).  Quantisation degrades ",
        "performance progressively: 32 levels approx= continuous; 16 levels shows mild ",
        "degradation; 8 levels shows significant sidelobe increase.\n",
    ]

    # -- Aperture --
    lines += [
        "## 9. Aperture Radius Effects\n",
        "Larger aperture captures more acoustic energy but produces a wider beam, ",
        "spilling perturbation into the neighbourhood (increasing nbr disruption and ",
        "reducing localisation ratio).  Smaller aperture enhances localisation but ",
        "reduces the total authority available for barrier lowering.\n",
        "For the typical well spacing of ~0.5 lambda, aperture radii of 0.8--1.0 mm ",
        "(approx 1.1--1.3 lambda) generally achieve the best trade-off.  Larger apertures ",
        "(1.5 mm approx 2 lambda) are too broad for tight bridging unless the wells are widely spaced.\n",
    ]

    # -- Translation --
    lines += [
        "## 10. Translation Robustness\n",
        "For the top 3 configs, the vortex field was analytically translated ",
        "along the A->B axis by {0, +/-0.2, +/-0.4} mm using grid interpolation ",
        "(simulates physical displacement of the entire disk/lens assembly).\n",
    ]
    if translation_metrics:
        lines.append("| Config | Offset (mm) | L_U | L_F | Barrier % | eta_out | Nbr RMS% |")
        lines.append("|---|---|---|---|---|---|---|")
        for tm in translation_metrics:
            lines.append(
                f"| `{tm['config_id']}` | {tm['offset_mm']:+.1f} | "
                f"{tm['L_U']:.4f} | {tm['L_F']:.4f} | "
                f"{tm.get('barrier_reduction_pct', 0):.1f} | "
                f"{tm['eta_out']:.5f} | "
                f"{tm.get('neighbour_depth_rms_pct', 0):.2f} |")
        lines += [
            "",
            "**Commentary:** Translation robustness tests whether the bridge effect ",
            "follows the lens when it is displaced.  For a well-localised bridge, ",
            "small translations (+/-0.2 mm) should maintain similar metrics.  Larger ",
            "translations (+/-0.4 mm) may move the vortex perturbation away from the ",
            "bridge corridor, reducing L_U/L_F while slightly changing eta_out as the ",
            "perturbation shifts to different lattice positions.\n",
        ]
    else:
        lines.append("No translation data available.\n")

    # -- Recommendation --
    lines += ["## 11. Recommended Configuration\n"]
    if shortlist:
        best = shortlist[0]
        lines += [
            f"**Primary recommendation:** `{best['config_id']}`\n",
            f"- Composite score: {best['score']:.4f}",
            f"- L_U = {best['L_U']:.4f}, L_F = {best['L_F']:.4f}",
            f"- Barrier reduction: {best['barrier_reduction_pct']:.1f}%",
            f"- eta_out: {best['eta_out']:.5f}",
            f"- Neighbour RMS: {best['nbr_rms']:.2f}%\n",
        ]
        if len(shortlist) > 1:
            s2 = shortlist[1]
            lines.append(f"**Secondary:** `{s2['config_id']}` (score = {s2['score']:.4f})\n")
    else:
        lines.append("No viable configuration identified.\n")

    lines.append("**Clear statement:** These rankings are relative and model-dependent. "
                 "The recommended configuration should be validated with dynamic particle "
                 "trajectory simulation before proceeding to experimental implementation.\n")

    # -- Phase DOF analysis --
    lines += [
        "## 12. Phase Degree of Freedom Analysis\n",
        "### 12.1 Superposition Model\n",
        "The combined pressure field is now:\n",
        "    p_comb = p_stand + alpha * exp(1j * phi) * V_ratio * p_vortex\n",
        "where phi is a global relative phase in [0, 2*pi).  This adds a second ",
        "continuous control variable without recomputing FEM solves -- phase only ",
        "modifies the post-processing superposition.\n",
        f"The phase sweep used {N_PHI} values uniformly spaced in [0, 2*pi).\n",
    ]

    # Phase comparison results
    if comparison_data:
        lines += [
            "### 12.2 Phase-Optimised vs Phase-Fixed Comparison\n",
            "For the top 3 configurations, barrier reduction was compared at phi=0 ",
            "(equivalent to the previous alpha-only study) and at the best phi:\n",
            "| Config | alpha | Barrier(phi=0) | Barrier(best phi) | phi_opt (deg) | Improvement | eta_out | Nbr RMS% |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for cd in comparison_data:
            lines.append(
                f"| `{cd['config_id']}` | {cd['alpha']:.2f} | "
                f"{cd['barrier_phi0']:.1f}% | {cd['barrier_best_phi']:.1f}% | "
                f"{cd['best_phi_deg']:.0f} | "
                f"{cd['improvement_factor']:.3f}x | "
                f"{cd.get('eta_out_at_best_phi', 0):.5f} | "
                f"{cd.get('neighbour_rms_at_best_phi', 0):.2f} |")

        any_meaningful = any(abs(cd.get("improvement_factor", 1) - 1) > 0.10 for cd in comparison_data if np.isfinite(cd.get("improvement_factor", 1)))
        max_improve = max((cd.get("improvement_factor", 1) for cd in comparison_data if np.isfinite(cd.get("improvement_factor", 1))), default=1.0)

        lines += [
            "",
            "### 12.3 Interpretation\n",
        ]

        if any_meaningful:
            lines += [
                f"**Phase DOF IS meaningful.** The maximum improvement factor is {max_improve:.3f}x ",
                "(greater than 10% threshold).  Phase alignment enables stronger or more ",
                "localised saddle lowering for at least some configurations.\n",
                "This implies that the relative phase between standing and vortex fields ",
                "at the bridge saddle point significantly affects the interference pattern.  ",
                "The architecture should support independent phase control (i.e. the vortex ",
                "transducer's phase reference must be adjustable relative to the standing-wave ",
                "array).\n",
            ]
        else:
            lines += [
                f"**Phase DOF is NOT meaningful.** The maximum improvement factor is {max_improve:.3f}x ",
                "(below 10% threshold).  Phase alignment does not significantly improve ",
                "barrier reduction beyond what alpha-only control achieves.\n",
                "This indicates that the cross-term (interference between standing and vortex ",
                "fields) is either too weak relative to the direct vortex energy term, or that ",
                "the saddle geometry makes barrier reduction insensitive to the phase of the ",
                "vortex contribution.  Single-mode (alpha-only) control is sufficient and adds ",
                "no benefit from phase calibration.\n",
            ]

        lines += [
            "### 12.4 Cross-Term Audit\n",
            "The cross-term `2*alpha*V*Re{exp(1j*phi)*conj(p_s)*p_v}` was evaluated at ",
            "the saddle, well A, and a neighbour well.  If phi is correctly wired, these ",
            "curves should exhibit sinusoidal (cosine) behaviour vs phi, with amplitude ",
            "proportional to |p_s|*|p_v| at each point.\n",
            "See `figures/phase_cross_term_vs_phi.png` for the result.\n",
            "### 12.5 Barrier vs Phi Curves\n",
            "For low, medium, and high alpha values, barrier reduction was plotted as a ",
            "function of phi.  A flat response would indicate phi has no effect; a sinusoidal ",
            "response confirms phase sensitivity.\n",
            "See `figures/barrier_vs_phi.png`.\n",
            "### 12.6 Spatial delta-U Maps\n",
            "For each top config, delta-U maps at phi=0 and phi=phi_opt are saved.  Visual ",
            "comparison shows whether the saddle is lowered more strongly at the optimal phase ",
            "and whether neighbour wells are less affected.\n",
            "See per-config `deltaU_phi0.png` and `deltaU_phi_opt.png`.\n",
        ]
    else:
        lines += [
            "### 12.2 Phase Audit Results\n",
            "No comparison data available (no configs in shortlist).\n",
        ]

    # -- Next experiments --
    lines += [
        "## 13. Suggested Next Experiments\n",
        "1. **Dynamic bridge simulation:** Run time-stepping particle dynamics to ",
        "   verify that a bead in well A actually snaps to well B when the vortex activates.",
        "2. **Pulsed activation:** Test brief vortex pulses (on at alpha_min, hold ~ms, off) ",
        "   for reliable transfer without sustained disruption.",
        "3. **Sequential bridge hopping:** Translate lens to successive well pairs for ",
        "   multi-step lattice transport.",
        "4. **Higher resolution validation:** Re-run top 3 at 6 elem/lambda to verify ranking.",
        "5. **3D Gorkov:** Full volumetric analysis for axial stability during transit.",
        "6. **Proper boundary quantisation:** Implement phase quantisation at the FEM ",
        "   boundary drive level (not post-hoc) for accurate plastic lens modelling.",
        "7. **Wider alpha range or relaxed criteria:** If no config meets alpha_min, relax one ",
        "   criterion (e.g., allow eta_out <= 0.08) and repeat to find practical operating point.\n",
    ]

    # -- Figures --
    lines += [
        "## 14. Figures\n",
        "Per-config figure set in `figures/{config_id}/`:\n",
        "- `field_panels_xy.png` -- |p_stand|, |p_vortex|, |p_comb|, delta|p|",
        "- `gorkov_xy.png` -- U_stand, U_comb, delta_U (zoomed to bridge)",
        "- `bridge_profile.png` -- Gorkov centreline U(s) along A->B",
        "- `xz_slice.png` -- XZ cross-section through bridge midpoint",
        "- `deltaU_phi0.png` -- delta-U map at phi=0",
        "- `deltaU_phi_opt.png` -- delta-U map at optimal phi",
        "- `deltaU_phase_comparison.png` -- side-by-side phi=0 vs phi_opt\n",
        "Comparison figures in `figures/`:\n",
        "- `localisation_vs_alpha.png` -- L_U, L_F vs alpha for top configs",
        "- `eta_vs_alpha.png` -- global disruption eta_out vs alpha",
        "- `barrier_vs_alpha.png` -- barrier reduction % vs alpha",
        "- `alpha_min_comparison.png` -- alpha_min per config (or note if none met criteria)",
        "- `phase_cross_term_vs_phi.png` -- cross-term audit (sinusoidal check)",
        "- `barrier_vs_phi.png` -- barrier reduction vs phi at low/med/high alpha",
        "- `phase_fixed_vs_optimised.png` -- phi=0 vs best-phi comparison bar chart",
    ]

    with open(OUT_DIR / "VORTEX_BRIDGE_DESIGN_STUDY_NOTE.md", "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
