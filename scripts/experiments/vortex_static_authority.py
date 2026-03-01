#!/usr/bin/env python3
"""
Vortex Static Particle-Authority Comparison
============================================

Compare candidate vortex configs on:
  1. Global standing-wave preservation  (field disruption outside ROI)
  2. Local Gor'kov authority            (potential/force changes inside ROI)
  3. Steerability proxy                 (force-direction consistency)
  4. Neighbour-trap disruption          (depth changes at nearby minima)

All results are treated as a **relative ranking tool** — the model is
approximate, so absolute magnitudes should NOT be over-interpreted.

Usage:
    micromamba run -n acousto-complex python scripts/experiments/vortex_static_authority.py
    micromamba run -n acousto-complex python scripts/experiments/vortex_static_authority.py --add-optional

Outputs:
    results/vortex_static_authority_YYYYMMDD_HHMMSS/
        csv/config_summary.csv
        csv/roi_profiles.csv
        results.json
        figures/{config_id}/...
        VORTEX_STATIC_AUTHORITY_NOTE.md
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
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
Z_PLANES = [Z_STAR - 0.25 * LAM, Z_STAR, Z_STAR + 0.25 * LAM]
Z_LABELS = ["z_minus", "z_star", "z_plus"]

CX, CY  = 3.0e-3, 3.0e-3
ROI_R   = 1.5 * LAM

ALPHA   = 0.20
V_RATIO = (CORRECTED_PRESET["standing_velocity_amplitude"] /
           CORRECTED_PRESET["disk_velocity_amplitude"])     # 10

EPL     = 4          # elem/λ  (5 OOM on 8 GB)
GRID_N  = 400        # post-FEM interpolation grid

PHASE1_DIR    = PROJECT_ROOT / "results" / "vortex_lens_sweep_20260225_152950"
WORKER_SCRIPT = PROJECT_ROOT / "scripts" / "experiments" / "_solve_worker_multi_z.py"

# ==================================================================
# Output directory
# ==================================================================
TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
OUT_DIR   = PROJECT_ROOT / "results" / f"vortex_static_authority_{TIMESTAMP}"
FIG_DIR   = OUT_DIR / "figures"
CSV_DIR   = OUT_DIR / "csv"
FIG_DPI   = 300

# Desired steerability direction
E_HAT = np.array([1.0, 0.0])   # +x


# ==================================================================
# MANDATORY / OPTIONAL config ids
# ==================================================================
MANDATORY_IDS = [
    "bg_l1_w0.8_kr4234_cos",
    "plastic_l1_f2.5_off0.2_cos",
    "bg_l3_w0.4_kr4234_cos",
]
OPTIONAL_IDS = [
    "plastic_l1_f1.5_off0.2_cos",
    "bg_l2_w0.4_kr4234_cos",
]


# ==================================================================
# Config helpers
# ==================================================================

def load_phase1_configs(ids: List[str]) -> List[Dict[str, Any]]:
    """Load Phase-1 result rows for specific config_ids."""
    with open(PHASE1_DIR / "results.json") as f:
        ranked = json.load(f)["ranked_list"]

    lookup = {r["config_id"]: r for r in ranked if "config_id" in r}
    out = []
    for cid in ids:
        if cid not in lookup:
            print(f"  WARNING: {cid} not found in Phase-1 results")
            continue
        out.append(lookup[cid])
    return out


def config_to_overrides(r: Dict, standing_only: bool = False) -> dict:
    """Phase-1 result → FarFieldConfig overrides (epl=EPL, grid=GRID_N)."""
    cfg = {**CORRECTED_PRESET, "elements_per_wavelength": EPL}

    if standing_only:
        cfg["disk_velocity_amplitude"] = 0.0
        return cfg

    cfg["standing_velocity_amplitude"] = 0.0
    cfg["disk_velocity_amplitude"] = 1e-6

    fam = r["family"]
    cfg["lens_l"] = r["ell"]
    cfg["lens_focus_offset_x"] = r.get("offset_x_mm", 0.0) * 1e-3
    cfg["lens_focus_offset_y"] = r.get("offset_y_mm", 0.0) * 1e-3
    cfg["lens_apodization"] = r.get("apod", "cosine_taper")

    if fam == "lg":
        cfg["lens_drive"] = "lg"
        cfg["lens_beam_waist"] = r.get("beam_waist_mm", 0.6) * 1e-3
        fl = r.get("focal_length_mm")
        cfg["lens_focal_length"] = (fl * 1e-3) if fl else 0.0
    elif fam == "bg":
        cfg["lens_drive"] = "bessel_gauss"
        cfg["lens_k_r"] = r.get("k_r", 0.5 * K_WATER)
        cfg["lens_beam_waist"] = r.get("beam_waist_mm", 0.6) * 1e-3
    elif fam == "bessel":
        cfg["lens_drive"] = "bessel"
        cfg["lens_k_r"] = r.get("k_r", K_WATER)
    elif fam == "plastic":
        cfg["lens_drive"] = "plastic"
        cfg["lens_focal_length"] = r.get("focal_length_mm", 2.0) * 1e-3
    return cfg


# ==================================================================
# Subprocess FEM solver
# ==================================================================

def solve_multi_z(overrides: dict, label: str,
                  z_list: List[float] = Z_PLANES,
                  mid_y: float = CY,
                  n_xy: int = GRID_N) -> Optional[dict]:
    """FEM solve → .npz with xg,yg, p_xy_{i}, xg_xz,zg_xz,p_xz."""
    result_file = str(OUT_DIR / f"_grid_{label}.npz")
    if os.path.exists(result_file):
        return dict(np.load(result_file, allow_pickle=False))

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
    return dict(np.load(result_file, allow_pickle=False))


def combine(p_stand, p_vortex, alpha=ALPHA):
    return p_stand + alpha * V_RATIO * p_vortex


# ==================================================================
# Gor'kov potential & force
# ==================================================================

def gorkov_2d(p, dx, dy):
    """Gor'kov potential U on a 2-D XY grid (complex p)."""
    K = WATER_DENSITY * WATER_C**2
    p2 = np.abs(p)**2 / 2
    dp_dx = np.gradient(p, dx, axis=1)
    dp_dy = np.gradient(p, dy, axis=0)
    grad_p2 = np.abs(dp_dx)**2 + np.abs(dp_dy)**2
    v2 = grad_p2 / (2 * OMEGA**2 * WATER_DENSITY**2)
    pf = (4 * np.pi / 3) * PARTICLE_RADIUS**3
    return pf * (F1_CONTRAST * p2 / (2 * K) -
                 F2_CONTRAST * (3 * WATER_DENSITY / 4) * v2)


def gorkov_force(U, dx, dy):
    """F = -∇U  →  (Fx[ny,nx], Fy[ny,nx])."""
    Fx = -np.gradient(U, dx, axis=1)
    Fy = -np.gradient(U, dy, axis=0)
    return Fx, Fy


# ==================================================================
# Trap finding
# ==================================================================

def find_local_minima(U, xg, yg, min_sep_px: int = 15):
    """All local minima separated by ≥ min_sep_px pixels.
    Returns list of (ix, iy) sorted by ascending U."""
    filt = minimum_filter(U, size=min_sep_px)
    is_min = (U == filt) & np.isfinite(U)
    iys, ixs = np.where(is_min)
    order = np.argsort(U[iys, ixs])
    return list(zip(ixs[order].tolist(), iys[order].tolist()))


def find_target_trap(minima, xg, yg, target_x=CX, target_y=CY):
    """Return (ix, iy) of minimum nearest to target."""
    dists = [(xg[ix] - target_x)**2 + (yg[iy] - target_y)**2
             for ix, iy in minima]
    return minima[int(np.argmin(dists))]


def find_neighbors(minima, trap_ix, trap_iy, xg, yg,
                   n: int = 8, min_dist: float = 0.3 * LAM):
    """N nearest local minima to (trap_ix, trap_iy), excluding itself."""
    tx, ty = xg[trap_ix], yg[trap_iy]
    dists = [np.sqrt((xg[ix] - tx)**2 + (yg[iy] - ty)**2)
             for ix, iy in minima]
    order = np.argsort(dists)
    nb = []
    for i in order:
        if dists[i] > min_dist:
            nb.append(minima[i])
            if len(nb) >= n:
                break
    return nb


# ==================================================================
# Barrier depth  (radial escape search)
# ==================================================================

def compute_barrier(U, xg, yg, min_ix, min_iy,
                    n_angles: int = 72, n_radial: int = 120,
                    max_r: float = 2.5 * LAM):
    """
    Barrier depth = min over radial directions of
    (first local max along ray − U_min).
    """
    U_min_val = U[min_iy, min_ix]
    x0, y0 = xg[min_ix], yg[min_iy]

    interp = RegularGridInterpolator(
        (yg, xg), U, method="linear",
        bounds_error=False, fill_value=np.nan)

    barriers = []
    for theta in np.linspace(0, 2 * np.pi, n_angles, endpoint=False):
        r_vals = np.linspace(0, max_r, n_radial)
        xs = x0 + r_vals * np.cos(theta)
        ys = y0 + r_vals * np.sin(theta)
        pts = np.column_stack([ys, xs])
        profile = interp(pts)

        # find first local maximum beyond the minimum
        found = False
        for i in range(2, len(profile) - 1):
            if np.isnan(profile[i]):
                break
            if profile[i] >= profile[i - 1] and profile[i] > profile[i + 1]:
                barriers.append(float(profile[i] - U_min_val))
                found = True
                break
        if not found:
            valid = profile[~np.isnan(profile)]
            if len(valid) > 1:
                barriers.append(float(valid[-1] - U_min_val))

    if not barriers:
        return 0.0
    return float(np.min(barriers))   # weakest escape direction


# ==================================================================
# ROI helpers
# ==================================================================

def make_roi(xg, yg, cx, cy, r):
    X, Y = np.meshgrid(xg, yg)
    return (X - cx)**2 + (Y - cy)**2 <= r**2


def zoom_slice(xg, yg, cx, cy, half_w):
    """Return (sy, sx) index slices for a zoom window."""
    ix0 = max(int(np.searchsorted(xg, cx - half_w)) - 1, 0)
    ix1 = min(int(np.searchsorted(xg, cx + half_w)) + 1, len(xg))
    iy0 = max(int(np.searchsorted(yg, cy - half_w)) - 1, 0)
    iy1 = min(int(np.searchsorted(yg, cy + half_w)) + 1, len(yg))
    return slice(iy0, iy1), slice(ix0, ix1)


# ==================================================================
# Metrics computation
# ==================================================================

def compute_all_metrics(p_stand, p_vortex, xg, yg,
                        trap_ix, trap_iy,
                        U_stand, U_comb,
                        Fx_stand, Fy_stand,
                        Fx_comb, Fy_comb,
                        minima_stand, minima_comb) -> Dict[str, Any]:
    """Compute the full metric dictionary for one config at one z-plane."""

    dx = xg[1] - xg[0]
    dy = yg[1] - yg[0]
    tx, ty = xg[trap_ix], yg[trap_iy]

    roi  = make_roi(xg, yg, tx, ty, ROI_R)
    out_ = ~roi

    p_comb = combine(p_stand, p_vortex)
    m: Dict[str, Any] = {}

    # ── 1) Global preservation ────────────────────────────────────
    delta_p = p_comb - p_stand
    norm_s = np.sqrt(np.sum(np.abs(p_stand[out_])**2))
    norm_d = np.sqrt(np.sum(np.abs(delta_p[out_])**2))
    m["eta_out"] = float(norm_d / norm_s) if norm_s > 0 else float("nan")

    s_out = np.abs(p_stand[out_])
    c_out = np.abs(p_comb[out_])
    if np.std(s_out) > 0 and np.std(c_out) > 0:
        m["corr_out"] = float(np.corrcoef(s_out, c_out)[0, 1])
    else:
        m["corr_out"] = float("nan")

    # ── 2) Local Gor'kov authority ────────────────────────────────
    m["trap_x_mm"] = tx * 1e3
    m["trap_y_mm"] = ty * 1e3

    m["U_min_stand"] = float(U_stand[trap_iy, trap_ix])
    # find combined-field minimum closest to original trap
    roi_small = make_roi(xg, yg, tx, ty, 0.5 * LAM)
    U_comb_masked = np.where(roi_small, U_comb, np.inf)
    iy_c, ix_c = np.unravel_index(np.argmin(U_comb_masked), U_comb.shape)
    m["U_min_comb"] = float(U_comb[iy_c, ix_c])
    m["delta_U_min"] = float(m["U_min_comb"] - m["U_min_stand"])

    barrier_s = compute_barrier(U_stand, xg, yg, trap_ix, trap_iy)
    barrier_c = compute_barrier(U_comb, xg, yg, ix_c, iy_c)
    m["barrier_stand"] = barrier_s
    m["barrier_comb"]  = barrier_c
    if barrier_s != 0:
        m["barrier_reduction_pct"] = 100.0 * (barrier_s - barrier_c) / abs(barrier_s)
    else:
        m["barrier_reduction_pct"] = float("nan")

    # force metrics (combined field)
    Fmag_comb = np.sqrt(Fx_comb**2 + Fy_comb**2)
    mean_F_roi = float(np.mean(Fmag_comb[roi]))
    mean_F_out = float(np.mean(Fmag_comb[out_]))
    m["mean_F_roi"] = mean_F_roi
    m["mean_F_out"] = mean_F_out
    m["localization_ratio"] = mean_F_roi / mean_F_out if mean_F_out > 0 else float("nan")
    m["peak_F_roi"] = float(Fmag_comb[roi].max()) if roi.any() else 0.0

    # delta-force localization (vortex-induced change — most discriminating)
    dFx = Fx_comb - Fx_stand
    dFy = Fy_comb - Fy_stand
    dFmag = np.sqrt(dFx**2 + dFy**2)
    mean_dF_roi = float(np.mean(dFmag[roi]))
    mean_dF_out = float(np.mean(dFmag[out_]))
    m["mean_dF_roi"] = mean_dF_roi
    m["mean_dF_out"] = mean_dF_out
    m["dF_localization"] = mean_dF_roi / mean_dF_out if mean_dF_out > 0 else float("nan")
    m["peak_dF_roi"] = float(dFmag[roi].max()) if roi.any() else 0.0

    # ΔU ratio: fraction of standing barrier perturbed by vortex
    m["delta_U_ratio"] = abs(m["delta_U_min"]) / barrier_s if barrier_s > 0 else float("nan")

    # ── 3) Steerability proxy (ΔF basis) ──────────────────────────
    # dFx, dFy already computed above
    dF_par  = dFx * E_HAT[0] + dFy * E_HAT[1]     # F · ê
    dF_perp = -dFx * E_HAT[1] + dFy * E_HAT[0]    # F · ê_perp

    dF_par_roi = dF_par[roi]
    m["bias"]  = float(np.mean(dF_par_roi))
    m["sign_consistency"] = float(np.mean(dF_par_roi > 0))

    mean_par  = float(np.mean(np.abs(dF_par[roi])))
    mean_perp = float(np.mean(np.abs(dF_perp[roi])))
    m["anisotropy"] = mean_par / mean_perp if mean_perp > 0 else float("nan")

    # overall ΔF direction in ROI
    m["dF_mean_x"] = float(np.mean(dFx[roi]))
    m["dF_mean_y"] = float(np.mean(dFy[roi]))
    m["dF_mean_mag"] = float(np.sqrt(m["dF_mean_x"]**2 + m["dF_mean_y"]**2))
    m["dF_mean_angle_deg"] = float(np.degrees(np.arctan2(m["dF_mean_y"],
                                                          m["dF_mean_x"])))

    # ── 4) Neighbour-trap disruption ──────────────────────────────
    # Only count neighbors with meaningful barriers (≥ 10% of target trap)
    barr_thresh = 0.10 * barrier_s if barrier_s > 0 else 0
    nb_stand = find_neighbors(minima_stand, trap_ix, trap_iy, xg, yg, n=8)
    depth_changes = []
    for nix, niy in nb_stand:
        d_s = compute_barrier(U_stand, xg, yg, nix, niy)
        if d_s < barr_thresh:
            continue  # skip edge / shallow traps
        # find nearest combined minimum
        roi_nb = make_roi(xg, yg, xg[nix], yg[niy], 0.5 * LAM)
        U_c_nb = np.where(roi_nb, U_comb, np.inf)
        if not np.all(np.isinf(U_c_nb)):
            iy_n, ix_n = np.unravel_index(np.argmin(U_c_nb), U_comb.shape)
            d_c = compute_barrier(U_comb, xg, yg, ix_n, iy_n)
        else:
            d_c = 0.0
        depth_changes.append(100.0 * (d_s - d_c) / abs(d_s))

    m["n_neighbors"] = len(depth_changes)
    if depth_changes:
        m["neighbor_depth_max_pct"] = float(np.max(np.abs(depth_changes)))
        m["neighbor_depth_rms_pct"] = float(np.sqrt(np.mean(np.array(depth_changes)**2)))
    else:
        m["neighbor_depth_max_pct"] = float("nan")
        m["neighbor_depth_rms_pct"] = float("nan")

    # ── additional reference values ───────────────────────────────
    m["max_p_stand"] = float(np.abs(p_stand).max())
    m["max_p_vortex"] = float(np.abs(p_vortex).max())
    m["peak_vortex_roi"] = float(np.abs(p_vortex[roi]).max()) if roi.any() else 0.0

    return m


# ==================================================================
# Radial profiles for CSV export
# ==================================================================

def radial_profile(field, xg, yg, cx, cy, n_bins=60, max_r=3.0 * LAM):
    """Azimuthally-averaged radial profile around (cx,cy)."""
    X, Y = np.meshgrid(xg, yg)
    R = np.sqrt((X - cx)**2 + (Y - cy)**2)
    r_edges = np.linspace(0, max_r, n_bins + 1)
    r_mid = 0.5 * (r_edges[:-1] + r_edges[1:])
    vals = np.full(n_bins, np.nan)
    for i in range(n_bins):
        mask = (R >= r_edges[i]) & (R < r_edges[i + 1])
        if mask.any():
            vals[i] = np.mean(field[mask])
    return r_mid, vals


# ==================================================================
# Figure helpers
# ==================================================================

def _ext_mm(xg, yg):
    return [xg[0]*1e3, xg[-1]*1e3, yg[0]*1e3, yg[-1]*1e3]


def _roi_circle(ax, cx, cy, r, **kw):
    ax.add_patch(Circle((cx*1e3, cy*1e3), r*1e3,
                         fill=False, ec=kw.get("ec", "cyan"),
                         lw=kw.get("lw", 1.2), ls="--"))


def _mark_trap(ax, tx, ty, **kw):
    ax.plot(tx*1e3, ty*1e3,
            kw.get("marker", "+"), color=kw.get("color", "white"),
            ms=kw.get("ms", 14), mew=kw.get("mew", 2.5))


# ------------------------------------------------------------------
# 1) Field disruption — XY
# ------------------------------------------------------------------
def fig_disruption_xy(p_stand, p_vortex, xg, yg, tx, ty, cid, fig_dir):
    p_comb = combine(p_stand, p_vortex)
    dp = np.abs(p_comb) - np.abs(p_stand)
    ext = _ext_mm(xg, yg)

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.8), dpi=FIG_DPI)

    for i, (data, title, cmap) in enumerate([
        (np.abs(p_stand),  "|p_stand|",  "hot"),
        (np.abs(p_comb),   "|p_comb|",   "hot"),
        (dp,               "Δ|p|",       "RdBu_r"),
    ]):
        ax = axes[i]
        if cmap == "RdBu_r":
            vmax = max(abs(dp.min()), abs(dp.max())) or 1
            im = ax.imshow(data, extent=ext, origin="lower", cmap=cmap,
                           vmin=-vmax, vmax=vmax, interpolation="nearest")
        else:
            im = ax.imshow(data, extent=ext, origin="lower", cmap=cmap,
                           interpolation="nearest")
        _roi_circle(ax, tx, ty, ROI_R)
        _mark_trap(ax, tx, ty)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
        plt.colorbar(im, ax=ax, shrink=0.72).set_label("Pa")

    fig.suptitle(f"Field disruption — {cid}   α={ALPHA}", fontsize=12)
    fig.tight_layout()
    fname = fig_dir / "disruption_xy.png"
    fig.savefig(str(fname), dpi=FIG_DPI)
    plt.close(fig)
    return fname


# ------------------------------------------------------------------
# 2) Field disruption — XZ
# ------------------------------------------------------------------
def fig_disruption_xz(p_stand_xz, p_vortex_xz, xg_xz, zg_xz,
                      tx, cid, fig_dir):
    p_comb_xz = combine(p_stand_xz, p_vortex_xz)
    dp = np.abs(p_comb_xz) - np.abs(p_stand_xz)
    ext = [xg_xz[0]*1e3, xg_xz[-1]*1e3, zg_xz[0]*1e3, zg_xz[-1]*1e3]

    fig, axes = plt.subplots(1, 3, figsize=(19, 6.5), dpi=FIG_DPI)
    for i, (data, title, cmap) in enumerate([
        (np.abs(p_stand_xz),  "|p_stand| XZ", "hot"),
        (np.abs(p_comb_xz),   "|p_comb| XZ",  "hot"),
        (dp,                   "Δ|p| XZ",      "RdBu_r"),
    ]):
        ax = axes[i]
        if cmap == "RdBu_r":
            vmax = max(abs(dp.min()), abs(dp.max())) or 1
            im = ax.imshow(data, extent=ext, origin="lower", cmap=cmap,
                           aspect="auto", vmin=-vmax, vmax=vmax,
                           interpolation="nearest")
        else:
            im = ax.imshow(data, extent=ext, origin="lower", cmap=cmap,
                           aspect="auto", interpolation="nearest")
        ax.axhline(Z_STAR*1e3, color="cyan", ls="--", lw=1, label="z*")
        ax.set_xlabel("x (mm)"); ax.set_ylabel("z (mm)")
        ax.set_title(title, fontsize=11)
        plt.colorbar(im, ax=ax, shrink=0.72).set_label("Pa")

    fig.suptitle(f"XZ disruption (y = {CY*1e3:.1f} mm) — {cid}", fontsize=12)
    fig.tight_layout()
    fname = fig_dir / "disruption_xz.png"
    fig.savefig(str(fname), dpi=FIG_DPI)
    plt.close(fig)
    return fname


# ------------------------------------------------------------------
# 3) Gor'kov authority — XY (zoomed into ROI neighbourhood)
# ------------------------------------------------------------------
def fig_gorkov_xy(U_stand, U_comb, Fx_comb, Fy_comb,
                  xg, yg, tx, ty, cid, fig_dir):
    hw = 3.0 * LAM
    sy, sx = zoom_slice(xg, yg, tx, ty, hw)
    xz, yz = xg[sx], yg[sy]
    ext = _ext_mm(xz, yz)

    Us = U_stand[sy, sx]
    Uc = U_comb[sy, sx]
    dU = Uc - Us

    Fx_z = Fx_comb[sy, sx]
    Fy_z = Fy_comb[sy, sx]
    Fmag = np.sqrt(Fx_z**2 + Fy_z**2)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=FIG_DPI)

    # U_stand
    ax = axes[0, 0]
    im = ax.imshow(Us, extent=ext, origin="lower", cmap="viridis",
                   interpolation="nearest")
    _roi_circle(ax, tx, ty, ROI_R, ec="white")
    _mark_trap(ax, tx, ty, color="red")
    ax.set_title("U_stand  (Gor'kov)", fontsize=11)
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    plt.colorbar(im, ax=ax, shrink=0.72).set_label("J")

    # U_comb
    ax = axes[0, 1]
    im = ax.imshow(Uc, extent=ext, origin="lower", cmap="viridis",
                   interpolation="nearest")
    _roi_circle(ax, tx, ty, ROI_R, ec="white")
    _mark_trap(ax, tx, ty, color="red")
    ax.set_title(f"U_comb  (α={ALPHA})", fontsize=11)
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    plt.colorbar(im, ax=ax, shrink=0.72).set_label("J")

    # ΔU
    ax = axes[1, 0]
    vmax = max(abs(np.nanmin(dU)), abs(np.nanmax(dU))) or 1e-30
    im = ax.imshow(dU, extent=ext, origin="lower", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, interpolation="nearest")
    _roi_circle(ax, tx, ty, ROI_R, ec="lime")
    _mark_trap(ax, tx, ty, color="black")
    ax.set_title("ΔU = U_comb − U_stand", fontsize=11)
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    plt.colorbar(im, ax=ax, shrink=0.72).set_label("J")

    # |F| with quiver
    ax = axes[1, 1]
    im = ax.imshow(Fmag, extent=ext, origin="lower", cmap="magma",
                   interpolation="nearest")
    _roi_circle(ax, tx, ty, ROI_R, ec="cyan")
    _mark_trap(ax, tx, ty, color="white")
    # downsample quiver
    step = max(1, len(xz) // 25)
    Xq, Yq = np.meshgrid(xz[::step], yz[::step])
    ax.quiver(Xq*1e3, Yq*1e3,
              Fx_z[::step, ::step], Fy_z[::step, ::step],
              color="white", alpha=0.7, scale_units="xy",
              angles="xy", headwidth=4, headlength=5,
              scale=Fmag.max() / (hw * 1e3 * 0.6) if Fmag.max() > 0 else 1)
    ax.set_title("|F_comb| + direction", fontsize=11)
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    plt.colorbar(im, ax=ax, shrink=0.72).set_label("N")

    fig.suptitle(f"Gor'kov authority — {cid}", fontsize=13)
    fig.tight_layout()
    fname = fig_dir / "gorkov_xy.png"
    fig.savefig(str(fname), dpi=FIG_DPI)
    plt.close(fig)
    return fname


# ------------------------------------------------------------------
# 4) Steerability (ΔF-based)  — XY zoomed
# ------------------------------------------------------------------
def fig_steerability(Fx_stand, Fy_stand, Fx_comb, Fy_comb,
                     xg, yg, tx, ty, cid, fig_dir):
    dFx = Fx_comb - Fx_stand
    dFy = Fy_comb - Fy_stand
    dF_par = dFx * E_HAT[0] + dFy * E_HAT[1]
    dF_mag = np.sqrt(dFx**2 + dFy**2)

    hw = 3.0 * LAM
    sy, sx = zoom_slice(xg, yg, tx, ty, hw)
    xz, yz = xg[sx], yg[sy]
    ext = _ext_mm(xz, yz)

    fig, axes = plt.subplots(1, 3, figsize=(19, 6), dpi=FIG_DPI)

    # ΔF quiver over |ΔF|
    ax = axes[0]
    im = ax.imshow(dF_mag[sy, sx], extent=ext, origin="lower",
                   cmap="magma", interpolation="nearest")
    step = max(1, len(xz) // 22)
    Xq, Yq = np.meshgrid(xz[::step], yz[::step])
    dFx_z = dFx[sy, sx]
    dFy_z = dFy[sy, sx]
    sc = dF_mag[sy, sx].max() / (hw * 1e3 * 0.5) if dF_mag[sy, sx].max() > 0 else 1
    ax.quiver(Xq*1e3, Yq*1e3,
              dFx_z[::step, ::step], dFy_z[::step, ::step],
              color="cyan", alpha=0.75, scale_units="xy", angles="xy",
              headwidth=4, headlength=5, scale=sc)
    _roi_circle(ax, tx, ty, ROI_R, ec="white")
    _mark_trap(ax, tx, ty, color="lime")
    ax.set_title("|ΔF| + direction", fontsize=11)
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    plt.colorbar(im, ax=ax, shrink=0.72).set_label("N")

    # F_parallel heatmap (signed)
    ax = axes[1]
    par_z = dF_par[sy, sx]
    vmax = max(abs(np.nanmin(par_z)), abs(np.nanmax(par_z))) or 1e-30
    im = ax.imshow(par_z, extent=ext, origin="lower", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, interpolation="nearest")
    _roi_circle(ax, tx, ty, ROI_R, ec="lime")
    _mark_trap(ax, tx, ty, color="black")
    ax.set_title("ΔF·ê  (ê = +x)", fontsize=11)
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    plt.colorbar(im, ax=ax, shrink=0.72).set_label("N")

    # sign map
    ax = axes[2]
    sign_map = (par_z > 0).astype(float)
    im = ax.imshow(sign_map, extent=ext, origin="lower",
                   cmap="RdYlGn", vmin=0, vmax=1, interpolation="nearest")
    _roi_circle(ax, tx, ty, ROI_R, ec="magenta")
    _mark_trap(ax, tx, ty, color="black")
    ax.set_title("sign(ΔF·ê)  green=+x", fontsize=11)
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    plt.colorbar(im, ax=ax, shrink=0.72)

    fig.suptitle(f"Steerability proxy  (ê = +x) — {cid}", fontsize=12)
    fig.tight_layout()
    fname = fig_dir / "steerability_xy.png"
    fig.savefig(str(fname), dpi=FIG_DPI)
    plt.close(fig)
    return fname


# ------------------------------------------------------------------
# 5) Comparison bar chart (all configs)
# ------------------------------------------------------------------
def fig_comparison(all_metrics: List[Dict], fig_dir):
    """Multi-panel bar chart comparing configs at z*."""
    zs = [m for m in all_metrics if m["z_label"] == "z_star"]
    if not zs:
        return None
    cids = [m["config_id"] for m in zs]
    n = len(cids)
    x = np.arange(n)

    keys_panels = [
        ("eta_out",              "η_out (≤ 0.02)",                    "steelblue"),
        ("barrier_reduction_pct","Barrier reduction (%)",              "salmon"),
        ("dF_localization",      "ΔF localization |ΔF|_roi/|ΔF|_out", "seagreen"),
        ("delta_U_ratio",        "ΔU ratio |ΔU_min|/barrier",         "goldenrod"),
        ("sign_consistency",     "Sign consistency (+x)",              "mediumpurple"),
        ("neighbor_depth_rms_pct","Neighbor depth ΔRMS (%)",           "orange"),
    ]

    fig, axes = plt.subplots(1, len(keys_panels), figsize=(26, 5.5), dpi=FIG_DPI)
    short = [c.replace("_cos", "").replace("off0.2_", "") for c in cids]

    for i, (key, ylabel, color) in enumerate(keys_panels):
        ax = axes[i]
        vals = [m.get(key, 0) for m in zs]
        ax.bar(x, vals, color=color, edgecolor="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(short, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(axis="y", labelsize=8)
        ax.set_title(ylabel.split("(")[0].strip(), fontsize=10)

    fig.suptitle("Config comparison at z*  (relative ranking — not absolute truth)",
                 fontsize=12)
    fig.tight_layout()
    fname = fig_dir / "comparison_metrics.png"
    fig.savefig(str(fname), dpi=FIG_DPI)
    plt.close(fig)
    return fname


# ------------------------------------------------------------------
# 6) Comparison ΔU side-by-side
# ------------------------------------------------------------------
def fig_comparison_dU(results_cache, xg, yg, fig_dir):
    """Side-by-side ΔU for all configs (zoomed to ROI)."""
    zstar_items = [(cid, d) for cid, d in results_cache.items()
                   if "z_star" in d]
    if not zstar_items:
        return None

    n = len(zstar_items)
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 5.5), dpi=FIG_DPI)
    if n == 1:
        axes = [axes]

    for i, (cid, data) in enumerate(zstar_items):
        d = data["z_star"]
        dU = d["U_comb"] - d["U_stand"]
        tx, ty = d["trap_x"], d["trap_y"]
        hw = 3.0 * LAM
        sy, sx = zoom_slice(xg, yg, tx, ty, hw)
        xz, yz = xg[sx], yg[sy]
        ext = _ext_mm(xz, yz)
        dU_z = dU[sy, sx]
        vmax = max(abs(np.nanmin(dU_z)), abs(np.nanmax(dU_z))) or 1e-30

        ax = axes[i]
        im = ax.imshow(dU_z, extent=ext, origin="lower", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, interpolation="nearest")
        _roi_circle(ax, tx, ty, ROI_R, ec="lime")
        _mark_trap(ax, tx, ty, color="black")
        short = cid.replace("_cos", "").replace("off0.2_", "")
        ax.set_title(short, fontsize=10)
        ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
        plt.colorbar(im, ax=ax, shrink=0.72).set_label("J")

    fig.suptitle("ΔU = U_comb − U_stand  (zoomed ROI, z*)", fontsize=12)
    fig.tight_layout()
    fname = fig_dir / "comparison_dU.png"
    fig.savefig(str(fname), dpi=FIG_DPI)
    plt.close(fig)
    return fname


# ==================================================================
#  MAIN
# ==================================================================

def main():
    parser = argparse.ArgumentParser(description="Vortex static authority comparison")
    parser.add_argument("--add-optional", action="store_true",
                        help="Include optional configs D,E")
    args = parser.parse_args()

    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    ids = list(MANDATORY_IDS)
    if args.add_optional:
        ids.extend(OPTIONAL_IDS)

    print("=" * 72)
    print("Vortex Static Particle-Authority Comparison")
    print(f"Output : {OUT_DIR}")
    print(f"Configs: {ids}")
    print(f"z*     : {Z_STAR*1e3:.4f} mm    λ = {LAM*1e3:.3f} mm")
    print(f"Grid   : {GRID_N}   epl = {EPL}")
    print("=" * 72)

    # ── Load Phase-1 config params ────────────────────────────────
    configs = load_phase1_configs(ids)
    if not configs:
        print("No configs found!"); return
    print(f"\n  Loaded {len(configs)} config(s) from Phase-1 results.\n")

    # ── Solve standing once ───────────────────────────────────────
    print("Solving standing wave ...")
    stand_cfg = config_to_overrides(configs[0], standing_only=True)
    stand_data = solve_multi_z(stand_cfg, "standing")
    if stand_data is None:
        print("  FATAL: standing solve failed"); return
    print(f"  max|p_stand| = {float(stand_data['phys_max']):.3f} Pa  "
          f"t = {float(stand_data['solve_time']):.1f}s\n")

    xg = stand_data["xg"]
    yg = stand_data["yg"]
    dx = xg[1] - xg[0]
    dy = yg[1] - yg[0]

    # ── Per-z-plane standing Gor'kov + trap finding ───────────────
    stand_cache: Dict[str, dict] = {}    # z_label → {p, U, Fx, Fy, minima, trap}
    for zi, (zp, zlab) in enumerate(zip(Z_PLANES, Z_LABELS)):
        p_s = stand_data[f"p_xy_{zi}"]
        U_s = gorkov_2d(p_s, dx, dy)
        Fx_s, Fy_s = gorkov_force(U_s, dx, dy)
        minima = find_local_minima(U_s, xg, yg, min_sep_px=15)
        trap_ix, trap_iy = find_target_trap(minima, xg, yg)
        stand_cache[zlab] = dict(
            p=p_s, U=U_s, Fx=Fx_s, Fy=Fy_s,
            minima=minima,
            trap_ix=trap_ix, trap_iy=trap_iy,
            trap_x=xg[trap_ix], trap_y=yg[trap_iy])
        print(f"  z={zlab:8s}  trap @ ({xg[trap_ix]*1e3:.3f}, {yg[trap_iy]*1e3:.3f}) mm  "
              f"  U_min = {U_s[trap_iy, trap_ix]:.3e} J  "
              f"  {len(minima)} minima found")

    # ── Per vortex config ─────────────────────────────────────────
    all_metrics: List[Dict[str, Any]] = []
    results_cache: Dict[str, Dict[str, dict]] = {}   # cid → {z_label → arrays}

    for ci, r in enumerate(configs):
        cid = r["config_id"]
        print(f"\n{'─'*72}")
        print(f"  [{ci+1}/{len(configs)}]  {cid}")
        print(f"{'─'*72}")

        cfg_dir = FIG_DIR / cid
        cfg_dir.mkdir(parents=True, exist_ok=True)

        # Solve vortex
        vorides = config_to_overrides(r, standing_only=False)
        vdata = solve_multi_z(vorides, f"vortex_{cid}")
        if vdata is None:
            print(f"  SKIP: solve failed")
            continue
        print(f"  max|p_vortex| = {float(vdata['phys_max']):.3f} Pa  "
              f"t = {float(vdata['solve_time']):.1f}s")

        results_cache[cid] = {}

        for zi, (zp, zlab) in enumerate(zip(Z_PLANES, Z_LABELS)):
            p_vortex = vdata[f"p_xy_{zi}"]
            sc = stand_cache[zlab]
            p_stand  = sc["p"]
            U_stand  = sc["U"]
            Fx_stand = sc["Fx"]
            Fy_stand = sc["Fy"]
            trap_ix  = sc["trap_ix"]
            trap_iy  = sc["trap_iy"]
            tx, ty   = sc["trap_x"], sc["trap_y"]

            # Combined field Gor'kov
            p_comb = combine(p_stand, p_vortex)
            U_comb = gorkov_2d(p_comb, dx, dy)
            Fx_comb, Fy_comb = gorkov_force(U_comb, dx, dy)
            minima_comb = find_local_minima(U_comb, xg, yg, min_sep_px=15)

            # Store for comparison figures
            results_cache[cid][zlab] = dict(
                U_stand=U_stand, U_comb=U_comb,
                Fx_stand=Fx_stand, Fy_stand=Fy_stand,
                Fx_comb=Fx_comb, Fy_comb=Fy_comb,
                trap_x=tx, trap_y=ty)

            # Metrics
            m = compute_all_metrics(
                p_stand, p_vortex, xg, yg,
                trap_ix, trap_iy,
                U_stand, U_comb,
                Fx_stand, Fy_stand,
                Fx_comb, Fy_comb,
                sc["minima"], minima_comb)

            m["config_id"] = cid
            m["family"]    = r["family"]
            m["ell"]       = r["ell"]
            m["z_label"]   = zlab
            m["z_mm"]      = zp * 1e3
            m["alpha"]     = ALPHA
            all_metrics.append(m)

            tag = "***" if zlab == "z_star" else "   "
            print(f"  {tag} z={zlab:8s}  η={m['eta_out']:.5f}  "
                  f"barrier_Δ={m['barrier_reduction_pct']:+.1f}%  "
                  f"ΔF_loc={m['dF_localization']:.3f}  "
                  f"ΔU_rat={m['delta_U_ratio']:.4f}  "
                  f"sign={m['sign_consistency']:.2f}  "
                  f"nbrRMS={m['neighbor_depth_rms_pct']:.2f}%")

            # ── Figures (z* only to keep PNG count sane) ──────────
            if zlab == "z_star":
                fig_disruption_xy(p_stand, p_vortex, xg, yg, tx, ty, cid, cfg_dir)
                fig_gorkov_xy(U_stand, U_comb, Fx_comb, Fy_comb,
                              xg, yg, tx, ty, cid, cfg_dir)
                fig_steerability(Fx_stand, Fy_stand, Fx_comb, Fy_comb,
                                 xg, yg, tx, ty, cid, cfg_dir)

        # XZ disruption (one per config)
        fig_disruption_xz(stand_data["p_xz"], vdata["p_xz"],
                          vdata["xg_xz"], vdata["zg_xz"],
                          stand_cache["z_star"]["trap_x"],
                          cid, cfg_dir)

        print(f"  Figures → {cfg_dir.name}/")
        del vdata
        gc.collect()

    # ── Comparison figures ────────────────────────────────────────
    fig_comparison(all_metrics, FIG_DIR)
    fig_comparison_dU(results_cache, xg, yg, FIG_DIR)
    print("\n  Comparison figures → figures/")

    # ==============================================================
    # CSV: config_summary.csv
    # ==============================================================
    csv_keys = [
        "config_id", "family", "ell", "z_label", "z_mm", "alpha",
        "eta_out", "corr_out",
        "trap_x_mm", "trap_y_mm",
        "U_min_stand", "U_min_comb", "delta_U_min", "delta_U_ratio",
        "barrier_stand", "barrier_comb", "barrier_reduction_pct",
        "mean_F_roi", "mean_F_out", "localization_ratio", "peak_F_roi",
        "mean_dF_roi", "mean_dF_out", "dF_localization", "peak_dF_roi",
        "bias", "sign_consistency", "anisotropy",
        "dF_mean_x", "dF_mean_y", "dF_mean_mag", "dF_mean_angle_deg",
        "n_neighbors", "neighbor_depth_max_pct", "neighbor_depth_rms_pct",
        "max_p_stand", "max_p_vortex", "peak_vortex_roi",
    ]
    with open(CSV_DIR / "config_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_keys, extrasaction="ignore")
        w.writeheader()
        for row in all_metrics:
            w.writerow(row)
    print(f"\n  config_summary.csv : {len(all_metrics)} rows")

    # ==============================================================
    # CSV: roi_profiles.csv
    # ==============================================================
    profile_rows = []
    for m in all_metrics:
        if m["z_label"] != "z_star":
            continue
        cid = m["config_id"]
        tx, ty = m["trap_x_mm"] * 1e-3, m["trap_y_mm"] * 1e-3
        sc = stand_cache["z_star"]

        # find combined U from cache
        rc = results_cache.get(cid, {}).get("z_star")
        if rc is None:
            continue

        for field_name, field_data in [
            ("U_stand", sc["U"]),
            ("U_comb",  rc["U_comb"]),
            ("Fmag_stand", np.sqrt(sc["Fx"]**2 + sc["Fy"]**2)),
            ("Fmag_comb",  np.sqrt(rc["Fx_comb"]**2 + rc["Fy_comb"]**2)),
        ]:
            r_mid, vals = radial_profile(field_data, xg, yg, tx, ty)
            for ri in range(len(r_mid)):
                profile_rows.append({
                    "config_id": cid,
                    "field": field_name,
                    "r_mm": r_mid[ri] * 1e3,
                    "r_over_lam": r_mid[ri] / LAM,
                    "value": vals[ri],
                })

    prof_keys = ["config_id", "field", "r_mm", "r_over_lam", "value"]
    with open(CSV_DIR / "roi_profiles.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=prof_keys)
        w.writeheader()
        w.writerows(profile_rows)
    print(f"  roi_profiles.csv   : {len(profile_rows)} rows")

    # ==============================================================
    # JSON
    # ==============================================================
    def _clean(v):
        if isinstance(v, (np.floating, float)):
            return None if (isinstance(v, float) and v != v) else float(v)
        if isinstance(v, (np.integer, int)):
            return int(v)
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        return v

    out_json = {
        "params": {
            "alpha": ALPHA,
            "epl": EPL,
            "grid": GRID_N,
            "z_planes_mm": [z * 1e3 for z in Z_PLANES],
            "roi_r_mm": ROI_R * 1e3,
            "particle_radius_um": PARTICLE_RADIUS * 1e6,
            "e_hat": E_HAT.tolist(),
        },
        "configs": [r["config_id"] for r in configs],
        "metrics": [{k: _clean(v) for k, v in m.items()} for m in all_metrics],
        "total_time_s": time.time() - t0,
    }
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(out_json, f, indent=2, default=str)

    # ==============================================================
    # Ranked table
    # ==============================================================
    zs = [m for m in all_metrics if m["z_label"] == "z_star"]
    print(f"\n{'='*72}")
    print("RANKED TABLE at z*  (metrics are RELATIVE — do NOT over-interpret magnitudes)")
    print(f"{'='*72}")
    header = (f"{'config_id':36s} {'η_out':>7s} {'corr':>7s} "
              f"{'barr%':>7s} {'ΔF_loc':>7s} {'ΔU_rat':>7s} "
              f"{'sign%':>6s} {'nbrRMS%':>8s}")
    print(header)
    print("-" * len(header))
    for m in sorted(zs, key=lambda m: -m["dF_localization"]):
        print(f"{m['config_id']:36s} "
              f"{m['eta_out']:7.5f} {m['corr_out']:7.5f} "
              f"{m['barrier_reduction_pct']:+7.1f} "
              f"{m['dF_localization']:7.3f} "
              f"{m['delta_U_ratio']:7.4f} "
              f"{m['sign_consistency']:6.2f} "
              f"{m['neighbor_depth_rms_pct']:8.2f}")
    print()

    # ==============================================================
    # Markdown note
    # ==============================================================
    total = time.time() - t0
    _write_note(all_metrics, configs, total)
    print(f"Wrote VORTEX_STATIC_AUTHORITY_NOTE.md")

    print(f"\n{'='*72}")
    print(f"Static authority comparison complete.  {total:.0f}s  "
          f"({len(all_metrics)} metric rows)")
    print(f"Results: {OUT_DIR}")
    print("=" * 72)


# ==================================================================
# Markdown note
# ==================================================================

def _write_note(all_metrics, configs, total_time):
    zs = [m for m in all_metrics if m["z_label"] == "z_star"]
    zp = [m for m in all_metrics if m["z_label"] == "z_plus"]
    zm = [m for m in all_metrics if m["z_label"] == "z_minus"]

    lines = [
        "# Vortex Static Particle-Authority Comparison\n",
        f"**Date:** {time.strftime('%Y-%m-%d')}  ",
        f"**Script:** `scripts/experiments/vortex_static_authority.py`  ",
        f"**Results:** `{OUT_DIR.relative_to(PROJECT_ROOT)}/`  ",
        f"**Runtime:** {total_time:.0f} s\n",
        "> **Caveat:** All metrics are treated as a **relative ranking tool**.",
        "> Absolute magnitudes depend on uncertain boundary conditions, bead",
        "> properties, viscosity models, and 2-D Gor'kov approximations.",
        "> Do NOT over-interpret absolute forces or potentials.\n",
        "---\n",
        "## 1. Objective\n",
        "Compare candidate vortex lens configurations on global preservation, ",
        "local Gor'kov authority, steerability, and neighbor-trap disruption ",
        "to select 1–2 configs for dynamic transport simulation.\n",
        "## 2. Setup\n",
        "| Parameter | Value |",
        "|---|---|",
        f"| Configs | {len(configs)} |",
        f"| epl | {EPL} |",
        f"| Grid | {GRID_N} |",
        f"| z* | {Z_STAR*1e3:.4f} mm |",
        f"| α | {ALPHA} |",
        f"| Particle | a = {PARTICLE_RADIUS*1e6:.0f} µm polystyrene |",
        f"| Steer direction ê | +x |",
        f"| V_ratio | {V_RATIO:.0f} |",
        "\n## 3. Configs tested\n",
        "| Label | config_id | Family | ℓ |",
        "|---|---|---|---|",
    ]
    labels = "ABCDE"
    for i, r in enumerate(configs):
        lines.append(f"| {labels[i]} | `{r['config_id']}` | {r['family']} | {r['ell']} |")

    # ── z* results table ──────────────────────────────────────────
    lines += [
        "\n## 4. Results at z*\n",
        "| config_id | η_out | corr_out | barrier Δ% | ΔF loc | "
        "ΔU ratio | sign% | nbr RMS% |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for m in sorted(zs, key=lambda m: -m["dF_localization"]):
        lines.append(
            f"| `{m['config_id']}` "
            f"| {m['eta_out']:.5f} "
            f"| {m['corr_out']:.5f} "
            f"| {m['barrier_reduction_pct']:+.1f} "
            f"| {m['dF_localization']:.3f} "
            f"| {m['delta_U_ratio']:.4f} "
            f"| {m['sign_consistency']:.2f} "
            f"| {m['neighbor_depth_rms_pct']:.2f} |")

    # ── z-plane robustness ────────────────────────────────────────
    lines += [
        "\n## 5. Z-plane robustness\n",
        "| config_id | z | η_out | barrier Δ% | ΔF loc | ΔU ratio | sign% |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in all_metrics:
        lines.append(
            f"| `{m['config_id']}` "
            f"| {m['z_label']} "
            f"| {m['eta_out']:.5f} "
            f"| {m['barrier_reduction_pct']:+.1f} "
            f"| {m['dF_localization']:.3f} "
            f"| {m['delta_U_ratio']:.4f} "
            f"| {m['sign_consistency']:.2f} |")

    # ── Verdict ───────────────────────────────────────────────────
    best_loc = max(zs, key=lambda m: m["dF_localization"])
    best_du  = max(zs, key=lambda m: m["delta_U_ratio"])
    best_steer = max(zs, key=lambda m: m["sign_consistency"])
    least_disrupt = min(zs, key=lambda m: m["eta_out"])
    least_nbr = min(zs, key=lambda m: m.get("neighbor_depth_rms_pct", 999))

    lines += [
        "\n## 6. Verdict\n",
        f"- **Best ΔF localization (|ΔF|_roi / |ΔF|_out):** `{best_loc['config_id']}`",
        f"- **Best ΔU ratio (|ΔU_min| / barrier_stand):** `{best_du['config_id']}`",
        f"- **Best steerability (sign consistency):** `{best_steer['config_id']}`",
        f"- **Least field disruption:** `{least_disrupt['config_id']}`",
        f"- **Least neighbor disruption:** `{least_nbr['config_id']}`",
        "",
        "### Recommendations for dynamic transport\n",
    ]

    # attempt automated recommendation
    # score = dF_localization * (1 + delta_U_ratio) / (1 + nbr_rms/10)
    scored = []
    for m in zs:
        sc = (m["dF_localization"] * (1 + m["delta_U_ratio"])
              / (1 + m.get("neighbor_depth_rms_pct", 0) / 10))
        scored.append((m["config_id"], sc))
    scored.sort(key=lambda x: -x[1])

    lines.append("Composite ranking (ΔF_loc × (1+ΔU_ratio) / (1 + nbr_rms/10)):\n")
    for i, (cid, sc) in enumerate(scored):
        lines.append(f"  {i+1}. `{cid}`  score = {sc:.4f}")

    lines += [
        "",
        f"**Primary recommendation:** `{scored[0][0]}`  ",
        f"**Secondary:** `{scored[1][0]}` (if available)\n" if len(scored) > 1 else "",
        "These are relative rankings only. Dynamic transport simulation ",
        "will be the definitive test.\n",
        "## 7. Figures\n",
        "Per-config figure set in `figures/{config_id}/`:\n",
        "- `disruption_xy.png` — |p_stand|, |p_comb|, Δ|p|",
        "- `disruption_xz.png` — same for XZ midplane",
        "- `gorkov_xy.png` — U_stand, U_comb, ΔU, |F| quiver (zoomed ROI)",
        "- `steerability_xy.png` — ΔF direction, F_parallel, sign map",
        "",
        "Comparison figures in `figures/`:\n",
        "- `comparison_metrics.png` — bar chart of key metrics",
        "- `comparison_dU.png` — ΔU side-by-side for all configs",
    ]

    with open(OUT_DIR / "VORTEX_STATIC_AUTHORITY_NOTE.md", "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
