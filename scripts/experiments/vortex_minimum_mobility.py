#!/usr/bin/env python3
"""
Vortex Minimum-Mobility Investigation
======================================

Sweep vortex x-offset × α to determine the LOCKING → SLIDING transition.

The combined Gor'kov minimum can either:
  A) Stay LOCKED to the standing-wave lattice node, or
  B) SLIDE with the vortex core as it is translated.

This experiment sweeps offset (0–0.8 mm) and α (0.2–10) to find the
authority threshold α_crit where mobility_ratio > 0.8.

Linear superposition: p_comb = p_stand + α · V_ratio · p_vortex
(exact for the linear Helmholtz equation — equivalent to a full
combined FEM solve with both sources active).

Only 1 standing + N_configs × N_offsets vortex FEM solves are needed;
the α sweep is purely analytical.

Usage:
    micromamba run -n acousto-complex python scripts/experiments/vortex_minimum_mobility.py

Outputs:
    results/vortex_minimum_mobility_YYYYMMDD_HHMMSS/
        csv/minimum_tracking.csv
        results.json
        figures/offset_tracking_{config}.png
        figures/U_landscape_{config}.png
        figures/mobility_summary.png
        VORTEX_MINIMUM_MOBILITY_NOTE.md
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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import TwoSlopeNorm, Normalize
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

CX, CY  = 3.0e-3, 3.0e-3               # disk / domain centre
ROI_R   = 1.5 * LAM

V_RATIO = (CORRECTED_PRESET["standing_velocity_amplitude"] /
           CORRECTED_PRESET["disk_velocity_amplitude"])     # 10

EPL     = 4          # elem/λ  (5 OOM on 8 GB)
GRID_N  = 400        # post-FEM interpolation grid

PHASE1_DIR    = PROJECT_ROOT / "results" / "vortex_lens_sweep_20260225_152950"
WORKER_SCRIPT = PROJECT_ROOT / "scripts" / "experiments" / "_solve_worker_multi_z.py"

# ==================================================================
# Sweep parameters
# ==================================================================
CONFIG_IDS = [
    "plastic_l1_f2.5_off0.2_cos",
    "bg_l1_w0.8_kr4234_cos",
]

ALPHAS = [0.2, 0.5, 1.0, 2.0, 5.0, 10.0]

OFFSETS_MM = np.round(np.arange(0.0, 0.85, 0.1), 2).tolist()
# [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

# Minimum tracking
SEARCH_RADIUS = 0.5 * LAM   # search for tracked-min within this of previous
MIN_SEP_PX    = 5            # local-minima detection granularity

# Figures: representative subsets for landscape panels
ALPHAS_SHOW   = [0.2, 1.0, 2.0, 10.0]
OFFSETS_SHOW  = [0.0, 0.3, 0.6, 0.8]       # mm

MOBILITY_THRESHOLD = 0.8    # define α_crit as first α with mobility > this

# ==================================================================
# Output directory
# ==================================================================
TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
OUT_DIR   = PROJECT_ROOT / "results" / f"vortex_minimum_mobility_{TIMESTAMP}"
FIG_DIR   = OUT_DIR / "figures"
CSV_DIR   = OUT_DIR / "csv"
FIG_DPI   = 300

# Stable FEM cache — survives across re-runs (expensive to recompute)
CACHE_DIR = PROJECT_ROOT / "results" / "_mobility_fem_cache"


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


def config_to_overrides(r: Dict, standing_only: bool = False,
                        offset_x_m: float = 0.0) -> dict:
    """Phase-1 result → FarFieldConfig overrides.

    For vortex solves, ``offset_x_m`` overrides lens_focus_offset_x
    (absolute offset from disk centre).
    """
    cfg = {**CORRECTED_PRESET, "elements_per_wavelength": EPL}

    if standing_only:
        cfg["disk_velocity_amplitude"] = 0.0
        return cfg

    # vortex-only: disable standing, set unit disk amplitude
    cfg["standing_velocity_amplitude"] = 0.0
    cfg["disk_velocity_amplitude"] = 1e-6

    fam = r["family"]
    cfg["lens_l"] = r["ell"]
    cfg["lens_focus_offset_x"] = offset_x_m      # SWEEP PARAMETER
    cfg["lens_focus_offset_y"] = 0.0
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

def solve_single_z(overrides: dict, label: str,
                   z_val: float = Z_STAR,
                   mid_y: float = CY,
                   n_xy: int = GRID_N) -> Optional[dict]:
    """FEM solve → .npz with p_xy_0 at z_val (and XZ slice).

    Caches in CACHE_DIR (stable across re-runs) for fast iteration.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    result_file = str(CACHE_DIR / f"_grid_{label}.npz")
    if os.path.exists(result_file):
        d = dict(np.load(result_file, allow_pickle=False))
        return d

    args = {
        "overrides": overrides,
        "label": label,
        "trap_z_list": [z_val],
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


# ==================================================================
# Trap finding & barrier
# ==================================================================

def find_local_minima(U, xg, yg, min_sep_px: int = MIN_SEP_PX):
    """All local minima separated by ≥ min_sep_px pixels, sorted by ascending U."""
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


def compute_barrier(U, xg, yg, min_ix, min_iy,
                    n_angles: int = 72, n_radial: int = 120,
                    max_r: float = 2.5 * LAM):
    """Barrier depth = min over radial directions of
    (first local max along ray − U_min)."""
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
    return float(np.min(barriers))


# ==================================================================
# Minimum tracking & diagnostics
# ==================================================================

def track_minimum(U, xg, yg, prev_x, prev_y,
                  search_radius=SEARCH_RADIUS):
    """Find deepest local minimum within *search_radius* of (prev_x, prev_y).

    Returns (ix, iy) or (None, None) if no candidate found.
    """
    minima = find_local_minima(U, xg, yg)
    candidates = []
    for ix, iy in minima:
        dist = np.sqrt((xg[ix] - prev_x)**2 + (yg[iy] - prev_y)**2)
        if dist <= search_radius:
            candidates.append((ix, iy, U[iy, ix], dist))
    if not candidates:
        return None, None
    # deepest (lowest U value) within search radius
    candidates.sort(key=lambda c: c[2])
    return candidates[0][0], candidates[0][1]


def curvature_at(U, xg, yg, ix, iy):
    """∂²U/∂x² and ∂²U/∂y² at (ix, iy) via central differences."""
    dx = xg[1] - xg[0]
    dy = yg[1] - yg[0]
    if ix < 1 or ix >= len(xg) - 1 or iy < 1 or iy >= len(yg) - 1:
        return np.nan, np.nan
    kxx = float((U[iy, ix + 1] - 2 * U[iy, ix] + U[iy, ix - 1]) / dx**2)
    kyy = float((U[iy + 1, ix] - 2 * U[iy, ix] + U[iy - 1, ix]) / dy**2)
    return kxx, kyy


def detect_vortex_core(p_vortex, xg, yg,
                       expected_x, expected_y,
                       search_r=1.5 * LAM):
    """Find vortex core (min |p|) near expected position."""
    X, Y = np.meshgrid(xg, yg)
    mask = (X - expected_x)**2 + (Y - expected_y)**2 <= search_r**2
    amp = np.abs(p_vortex)
    amp_masked = np.where(mask, amp, np.inf)
    iy, ix = np.unravel_index(np.argmin(amp_masked), amp_masked.shape)
    return float(xg[ix]), float(yg[iy])


def compute_eta_out(p_stand, p_comb, xg, yg,
                    roi_cx=CX, roi_cy=CY, roi_r=ROI_R):
    """Relative L2 disruption of |p| outside ROI."""
    X, Y = np.meshgrid(xg, yg)
    outside = (X - roi_cx)**2 + (Y - roi_cy)**2 > roi_r**2
    diff = np.abs(p_comb[outside]) - np.abs(p_stand[outside])
    ref = np.abs(p_stand[outside])
    n_diff = np.sqrt(np.sum(diff**2))
    n_ref = np.sqrt(np.sum(ref**2))
    if n_ref == 0:
        return np.inf
    return float(n_diff / n_ref)


def compute_corr_out(p_stand, p_comb, xg, yg,
                     roi_cx=CX, roi_cy=CY, roi_r=ROI_R):
    """Pearson correlation of |p| outside ROI."""
    X, Y = np.meshgrid(xg, yg)
    outside = (X - roi_cx)**2 + (Y - roi_cy)**2 > roi_r**2
    a = np.abs(p_stand[outside])
    b = np.abs(p_comb[outside])
    az = a - a.mean()
    bz = b - b.mean()
    denom = np.sqrt(np.sum(az**2) * np.sum(bz**2))
    if denom == 0:
        return 1.0
    return float(np.sum(az * bz) / denom)


# ==================================================================
# Zoom-slice helpers
# ==================================================================

def zoom_slice(xg, yg, cx, cy, hw_x, hw_y):
    """Return (sy, sx) index slices for a zoom window."""
    ix0 = max(int(np.searchsorted(xg, cx - hw_x)) - 1, 0)
    ix1 = min(int(np.searchsorted(xg, cx + hw_x)) + 1, len(xg))
    iy0 = max(int(np.searchsorted(yg, cy - hw_y)) - 1, 0)
    iy1 = min(int(np.searchsorted(yg, cy + hw_y)) + 1, len(yg))
    return slice(iy0, iy1), slice(ix0, ix1)


# ==================================================================
# FIGURES
# ==================================================================

def fig_offset_tracking(cid, tracking_data, stand_node_x_mm,
                        vortex_cores_mm):
    """min_x vs offset for each α — THE key diagnostic plot."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9),
                                    height_ratios=[3, 1],
                                    sharex=True)

    cmap = plt.cm.viridis
    norm = Normalize(vmin=np.log10(min(ALPHAS)),
                     vmax=np.log10(max(ALPHAS)))
    colors = {a: cmap(norm(np.log10(a))) for a in ALPHAS}

    for alpha in ALPHAS:
        key = (cid, alpha)
        if key not in tracking_data:
            continue
        recs = tracking_data[key]
        offs = [r["offset_commanded_mm"] for r in recs]
        mins = [r["min_x_mm"] for r in recs]
        ax1.plot(offs, mins, "o-", color=colors[alpha], lw=1.8,
                 ms=5, label=f"α = {alpha}")

    # Identity line: perfect tracking
    # Standing node is at stand_node_x_mm.  Vortex core at CX*1e3 + offset.
    # Perfect tracking: min_x tracks the vortex core.
    off_arr = np.array(OFFSETS_MM)
    core_x_line = CX * 1e3 + off_arr
    ax1.plot(off_arr, core_x_line, "k--", lw=1, alpha=0.5,
             label="perfect tracking")
    ax1.axhline(stand_node_x_mm, color="gray", ls=":", lw=1,
                label=f"standing node ({stand_node_x_mm:.3f})")

    ax1.set_ylabel("Tracked minimum x (mm)")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.set_title(f"Minimum tracking — {cid}", fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Bottom: barrier depth vs offset per alpha
    for alpha in ALPHAS:
        key = (cid, alpha)
        if key not in tracking_data:
            continue
        recs = tracking_data[key]
        offs = [r["offset_commanded_mm"] for r in recs]
        barr = [r["barrier_depth"] for r in recs]
        ax2.plot(offs, barr, "o-", color=colors[alpha], lw=1.2, ms=4)

    ax2.set_xlabel("Commanded offset (mm)")
    ax2.set_ylabel("Barrier depth (J)")
    ax2.ticklabel_format(axis="y", style="scientific", scilimits=(-23, -22))
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = FIG_DIR / f"offset_tracking_{cid}.png"
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"    Figure → {out.name}")


def fig_U_landscape(cid, U_comb_cache, xg, yg, U_stand,
                    tracking_data, stand_node_x, stand_node_y,
                    vortex_cores):
    """Grid of zoomed U_comb panels: rows = α, cols = offset."""
    alphas_show = [a for a in ALPHAS_SHOW if a in ALPHAS]
    offsets_show = [o for o in OFFSETS_SHOW if o in OFFSETS_MM]
    nr, nc = len(alphas_show), len(offsets_show)
    if nr == 0 or nc == 0:
        return

    # Zoom window: encompasses standing node and vortex core at max offset
    max_off_m = max(OFFSETS_MM) * 1e-3
    zoom_cx = (stand_node_x + CX + max_off_m) / 2
    zoom_cy = stand_node_y
    hw_x = max_off_m / 2 + 1.8 * LAM
    hw_y = 1.2 * LAM
    sy, sx = zoom_slice(xg, yg, zoom_cx, zoom_cy, hw_x, hw_y)
    xz = xg[sx]
    yz = yg[sy]
    extent = [xz[0] * 1e3, xz[-1] * 1e3, yz[0] * 1e3, yz[-1] * 1e3]

    fig, axes = plt.subplots(nr, nc, figsize=(5 * nc, 4 * nr),
                              squeeze=False)

    for ri, alpha in enumerate(alphas_show):
        # Compute Umin/Umax for this row across shown offsets
        U_vals = []
        for off_mm in offsets_show:
            key = (cid, alpha, off_mm)
            if key in U_comb_cache:
                U_vals.append(U_comb_cache[key][sy, sx])
        if not U_vals:
            continue
        all_U = np.concatenate([u.ravel() for u in U_vals])
        vmin, vmax = np.nanpercentile(all_U, [2, 98])

        for ci, off_mm in enumerate(offsets_show):
            ax = axes[ri, ci]
            key = (cid, alpha, off_mm)
            if key not in U_comb_cache:
                ax.set_visible(False)
                continue

            U_z = U_comb_cache[key][sy, sx]
            im = ax.imshow(U_z, extent=extent, origin="lower",
                           aspect="equal", cmap="inferno",
                           vmin=vmin, vmax=vmax)

            # Markers
            snx_mm, sny_mm = stand_node_x * 1e3, stand_node_y * 1e3
            ax.plot(snx_mm, sny_mm, "^", color="white", ms=8,
                    mec="black", mew=0.8, zorder=5)

            # Tracked minimum
            trk_key = (cid, alpha)
            if trk_key in tracking_data:
                recs = tracking_data[trk_key]
                for rec in recs:
                    if abs(rec["offset_commanded_mm"] - off_mm) < 0.01:
                        ax.plot(rec["min_x_mm"], rec["min_y_mm"],
                                "o", color="white", ms=8,
                                mec="black", mew=0.8, zorder=5)
                        break

            # Vortex core
            vc_key = (cid, off_mm)
            if vc_key in vortex_cores:
                vcx, vcy = vortex_cores[vc_key]
                ax.plot(vcx * 1e3, vcy * 1e3, "x", color="yellow",
                        ms=9, mew=2, zorder=5)

            if ri == 0:
                ax.set_title(f"offset = {off_mm} mm", fontsize=10)
            if ci == 0:
                ax.set_ylabel(f"α = {alpha}\ny (mm)", fontsize=10)
            else:
                ax.set_yticklabels([])
            if ri == nr - 1:
                ax.set_xlabel("x (mm)")
            else:
                ax.set_xticklabels([])

        # One colorbar per row
        cb = fig.colorbar(im, ax=axes[ri, :].tolist(), shrink=0.85,
                          pad=0.02, label="U (J)")
        cb.formatter.set_powerlimits((-23, -22))
        cb.update_ticks()

    fig.suptitle(f"U_comb landscape — {cid}\n"
                 f"△ = standing node   ○ = tracked min   × = vortex core",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    out = FIG_DIR / f"U_landscape_{cid}.png"
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"    Figure → {out.name}")


def fig_mobility_summary(mobility_data, eta_data, configs):
    """Mobility ratio vs α and η_out vs α for both configs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    markers = ["o", "s"]
    colors = ["#1f77b4", "#d62728"]

    for i, cfg_r in enumerate(configs):
        cid = cfg_r["config_id"]
        if cid not in mobility_data:
            continue

        alphas = sorted(mobility_data[cid].keys())
        mob = [mobility_data[cid][a] for a in alphas]
        ax1.plot(alphas, mob, f"{markers[i]}-", color=colors[i],
                 lw=2, ms=7, label=cid[:25])

    ax1.axhline(MOBILITY_THRESHOLD, color="gray", ls="--", lw=1,
                label=f"threshold = {MOBILITY_THRESHOLD}")
    ax1.set_xlabel("α (vortex authority)")
    ax1.set_ylabel("Mobility ratio")
    ax1.set_xscale("log")
    ax1.set_ylim(-0.1, 1.3)
    ax1.legend(fontsize=8)
    ax1.set_title("Minimum mobility vs vortex authority")
    ax1.grid(True, alpha=0.3)

    for i, cfg_r in enumerate(configs):
        cid = cfg_r["config_id"]
        if cid not in eta_data:
            continue

        alphas = sorted(eta_data[cid].keys())
        etas = [eta_data[cid][a] for a in alphas]
        ax2.plot(alphas, etas, f"{markers[i]}-", color=colors[i],
                 lw=2, ms=7, label=cid[:25])

    ax2.set_xlabel("α (vortex authority)")
    ax2.set_ylabel("η_out (field disruption)")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.legend(fontsize=8)
    ax2.set_title("Far-field preservation vs authority")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = FIG_DIR / "mobility_summary.png"
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"    Figure → {out.name}")


def fig_pressure_panels(cid, p_stand, xg, yg,
                        vortex_cache, alpha_show, offsets_show_mm,
                        vortex_cores, stand_node_x, stand_node_y):
    """Pressure landscape panels for a selected α: rows = offset, cols = field."""
    offsets_sel = [o for o in offsets_show_mm if (cid, o) in vortex_cache]
    if not offsets_sel:
        return

    nrows = len(offsets_sel)
    ncols = 4  # |p_stand|, |p_vortex|, |p_comb|, Δ|p|
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                              squeeze=False)

    # Zoom window
    max_off_m = max(OFFSETS_MM) * 1e-3
    zoom_cx = (stand_node_x + CX + max_off_m) / 2
    zoom_cy = stand_node_y
    hw_x = max_off_m / 2 + 1.8 * LAM
    hw_y = 1.2 * LAM
    sy, sx = zoom_slice(xg, yg, zoom_cx, zoom_cy, hw_x, hw_y)
    xz = xg[sx]
    yz = yg[sy]
    extent = [xz[0] * 1e3, xz[-1] * 1e3, yz[0] * 1e3, yz[-1] * 1e3]

    p_stand_abs_z = np.abs(p_stand[sy, sx])

    for ri, off_mm in enumerate(offsets_sel):
        p_v = vortex_cache[(cid, off_mm)]
        p_comb = p_stand + alpha_show * V_RATIO * p_v
        p_v_abs_z = np.abs(p_v[sy, sx])
        p_comb_abs_z = np.abs(p_comb[sy, sx])
        dp_z = p_comb_abs_z - p_stand_abs_z

        vmax_p = max(p_stand_abs_z.max(), p_comb_abs_z.max())

        # |p_stand|
        ax = axes[ri, 0]
        ax.imshow(p_stand_abs_z, extent=extent, origin="lower",
                  aspect="equal", cmap="magma", vmin=0, vmax=vmax_p)
        ax.set_title("|p_stand|" if ri == 0 else "")
        ax.set_ylabel(f"off={off_mm}mm\ny (mm)")

        # |p_vortex|
        ax = axes[ri, 1]
        ax.imshow(p_v_abs_z, extent=extent, origin="lower",
                  aspect="equal", cmap="magma")
        ax.set_title("|p_vortex|" if ri == 0 else "")

        # |p_comb|
        ax = axes[ri, 2]
        ax.imshow(p_comb_abs_z, extent=extent, origin="lower",
                  aspect="equal", cmap="magma", vmin=0, vmax=vmax_p)
        ax.set_title(f"|p_comb| α={alpha_show}" if ri == 0 else "")

        # Δ|p|
        ax = axes[ri, 3]
        vlim = max(abs(dp_z.min()), abs(dp_z.max()))
        if vlim > 0:
            ax.imshow(dp_z, extent=extent, origin="lower",
                      aspect="equal", cmap="RdBu_r",
                      norm=TwoSlopeNorm(vcenter=0, vmin=-vlim, vmax=vlim))
        ax.set_title("Δ|p|" if ri == 0 else "")

        # Mark vortex core
        vc_key = (cid, off_mm)
        if vc_key in vortex_cores:
            vcx, vcy = vortex_cores[vc_key]
            for c in range(ncols):
                axes[ri, c].plot(vcx * 1e3, vcy * 1e3, "x", color="yellow",
                                 ms=7, mew=1.5, zorder=5)

    fig.suptitle(f"Pressure fields — {cid}  α = {alpha_show}", fontsize=12)
    plt.tight_layout()
    out = FIG_DIR / f"pressure_panels_{cid}_a{alpha_show}.png"
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"    Figure → {out.name}")


# ==================================================================
# Note writer
# ==================================================================

def _write_note(configs, all_rows, mobility_data, eta_data,
                alpha_crits, stand_node_x_mm, stand_node_y_mm,
                barrier_stand, total_time):
    lines = [
        "# Vortex Minimum-Mobility Investigation\n",
        f"**Date:** {time.strftime('%Y-%m-%d')}  ",
        f"**Script:** `scripts/experiments/vortex_minimum_mobility.py`  ",
        f"**Results:** `{OUT_DIR.relative_to(PROJECT_ROOT)}/`  ",
        f"**Runtime:** {total_time:.0f} s\n",
        "> **Caveat:** All metrics are **relative**. Absolute force/potential",
        "> magnitudes depend on uncertain BCs, bead properties, and 2-D",
        "> Gor'kov approximations. Do NOT over-interpret absolute values.\n",
        "---\n",
        "## 1. Objective\n",
        "Determine whether the vortex beam creates a movable Gor'kov minimum ",
        "that can **dominate** a standing-wave trap and translate with the ",
        "vortex core offset. Identify the authority threshold α_crit where ",
        "the combined minimum transitions from **locked** (immobile at ",
        "standing node) to **sliding** (tracking the vortex core).\n",
        "## 2. Setup\n",
        "| Parameter | Value |",
        "|---|---|",
        f"| Configs | {len(configs)} |",
        f"| epl | {EPL} |",
        f"| Grid | {GRID_N} |",
        f"| z* | {Z_STAR*1e3:.4f} mm |",
        f"| V_ratio | {V_RATIO:.0f} |",
        f"| α values | {ALPHAS} |",
        f"| Offsets (mm) | {OFFSETS_MM} |",
        f"| Search radius | {SEARCH_RADIUS*1e3:.3f} mm ({SEARCH_RADIUS/LAM:.2f} λ) |",
        f"| Particle | a = {PARTICLE_RADIUS*1e6:.0f} µm polystyrene |",
        f"| Standing node | ({stand_node_x_mm:.3f}, {stand_node_y_mm:.3f}) mm |",
        f"| Barrier (standing) | {barrier_stand:.3e} J |\n",
        "## 3. Configs tested\n",
        "| Label | config_id | Family | ℓ |",
        "|---|---|---|---|",
    ]
    labels = "ABCDE"
    for i, r in enumerate(configs):
        lines.append(f"| {labels[i]} | `{r['config_id']}` | "
                     f"{r['family']} | {r['ell']} |")

    # ── Mobility table ────────────────────────────────────────────
    lines += [
        "\n## 4. Mobility ratio vs α\n",
        "mobility_ratio = slope of (min_displacement_x vs commanded_offset)\n",
        "| config_id | " + " | ".join(f"α={a}" for a in ALPHAS) + " |",
        "|---" * (1 + len(ALPHAS)) + "|",
    ]
    for cfg_r in configs:
        cid = cfg_r["config_id"]
        vals = []
        for a in ALPHAS:
            m = mobility_data.get(cid, {}).get(a, np.nan)
            vals.append(f"{m:.3f}")
        lines.append(f"| `{cid}` | " + " | ".join(vals) + " |")

    # ── α_crit ────────────────────────────────────────────────────
    lines += [
        "\n## 5. Authority threshold (α_crit)\n",
        f"Threshold criterion: mobility_ratio > {MOBILITY_THRESHOLD}\n",
    ]
    for cfg_r in configs:
        cid = cfg_r["config_id"]
        ac = alpha_crits.get(cid)
        if ac is not None:
            lines.append(f"- **`{cid}`:** α_crit ≈ {ac}")
        else:
            lines.append(f"- **`{cid}`:** α_crit NOT REACHED in sweep")

    # ── η_out tradeoff ────────────────────────────────────────────
    lines += [
        "\n## 6. Preservation tradeoff (η_out vs α)\n",
        "| config_id | " + " | ".join(f"α={a}" for a in ALPHAS) + " |",
        "|---" * (1 + len(ALPHAS)) + "|",
    ]
    for cfg_r in configs:
        cid = cfg_r["config_id"]
        vals = []
        for a in ALPHAS:
            e = eta_data.get(cid, {}).get(a, np.nan)
            vals.append(f"{e:.4f}")
        lines.append(f"| `{cid}` | " + " | ".join(vals) + " |")

    # ── Interpretation ────────────────────────────────────────────
    lines += [
        "\n## 7. Interpretation\n",
        "### Regime classification\n",
        "- **Locked regime** (mobility ≈ 0): The standing-wave lattice traps ",
        "  dominate the Gor'kov landscape. The vortex perturbation is too weak ",
        "  to shift the combined minimum away from the standing node.\n",
        "- **Sliding regime** (mobility ≈ 1): The vortex-induced well ",
        "  dominates at least locally. The combined minimum tracks the vortex ",
        "  core position with near-unity fidelity.\n",
        "- **Transition region**: Partial coupling — the minimum shifts but ",
        "  lags behind the vortex core. May exhibit stick-slip behaviour.\n",
        "### Physical mechanism\n",
        "The vortex beam creates a Gor'kov minimum near its core (for dense ",
        "particles, the gradient term ∝ |∇p|² at the vortex ring provides ",
        "a trapping contribution). At low α, this perturbation sits atop a ",
        "much deeper standing-wave well and cannot dislodge the minimum. At ",
        "high α, the vortex well deepens until it dominates — analogous to a ",
        "depinning transition.\n",
        "The cost of high authority is far-field disruption (rising η_out). ",
        "The optimal operating point balances mobility against preservation.\n",
    ]

    # ── Verdict ───────────────────────────────────────────────────
    lines += [
        "## 8. Verdict\n",
    ]
    for cfg_r in configs:
        cid = cfg_r["config_id"]
        ac = alpha_crits.get(cid)
        if ac is not None:
            eta_at_crit = eta_data.get(cid, {}).get(ac, np.nan)
            lines.append(f"- **`{cid}`:** α_crit ≈ {ac}, "
                         f"η_out(α_crit) = {eta_at_crit:.4f}")
        else:
            lines.append(f"- **`{cid}`:** sliding regime NOT reached")

    lines += [
        "",
        "**Recommendation:** Use the config with the lowest α_crit that ",
        "maintains acceptable η_out for dynamic transport simulation.\n",
        "## 9. Figures\n",
        "- `offset_tracking_{config}.png` — tracked minimum x vs offset "
        "(one line per α)",
        "- `U_landscape_{config}.png` — zoomed Gor'kov landscape panels "
        "(rows = α, cols = offset)",
        "- `mobility_summary.png` — mobility ratio and η_out vs α",
        "- `pressure_panels_{config}_a{α}.png` — pressure field panels "
        "at selected authority levels",
    ]

    with open(OUT_DIR / "VORTEX_MINIMUM_MOBILITY_NOTE.md", "w") as f:
        f.write("\n".join(lines))
    print("  Wrote VORTEX_MINIMUM_MOBILITY_NOTE.md")


# ==================================================================
# CSV writer
# ==================================================================

def write_csv(all_rows):
    if not all_rows:
        return
    keys = list(all_rows[0].keys())
    path = CSV_DIR / "minimum_tracking.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(all_rows)
    print(f"  minimum_tracking.csv : {len(all_rows)} rows")


# ==================================================================
# MAIN SWEEP
# ==================================================================

def run_sweep():
    t0 = time.time()

    # ── Create output dirs ────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load configs ──────────────────────────────────────────────
    configs = load_phase1_configs(CONFIG_IDS)
    if not configs:
        print("FATAL: no configs loaded")
        return

    print("=" * 72)
    print("  Vortex Minimum-Mobility Investigation")
    print(f"  Output : {OUT_DIR}")
    print(f"  Configs: {[c['config_id'] for c in configs]}")
    print(f"  Alphas : {ALPHAS}")
    print(f"  Offsets: {OFFSETS_MM} mm")
    print(f"  z*     : {Z_STAR*1e3:.4f} mm    λ = {LAM*1e3:.4f} mm")
    print(f"  Grid   : {GRID_N}   epl = {EPL}")
    print(f"  FEM solves needed: 1 standing + {len(configs)*len(OFFSETS_MM)} vortex")
    print("=" * 72)

    # ── 1. Solve standing wave ────────────────────────────────────
    print("\nSolving standing wave ...")
    ov_stand = config_to_overrides(configs[0], standing_only=True)
    stand_data = solve_single_z(ov_stand, "standing")
    if stand_data is None:
        print("FATAL: standing solve failed")
        return

    xg = stand_data["xg"]
    yg = stand_data["yg"]
    p_stand = stand_data["p_xy_0"]
    dx = float(xg[1] - xg[0])
    dy = float(yg[1] - yg[0])

    print(f"  max|p_stand| = {np.abs(p_stand).max():.3f} Pa   t = {float(stand_data['solve_time']):.1f}s")

    # ── Standing-wave reference trap ──────────────────────────────
    U_stand = gorkov_2d(p_stand, dx, dy)
    minima_stand = find_local_minima(U_stand, xg, yg, min_sep_px=15)
    trap_ix, trap_iy = find_target_trap(minima_stand, xg, yg, CX, CY)
    stand_node_x = float(xg[trap_ix])
    stand_node_y = float(yg[trap_iy])
    barrier_stand = compute_barrier(U_stand, xg, yg, trap_ix, trap_iy)
    U_stand_min = float(U_stand[trap_iy, trap_ix])

    print(f"  Standing trap @ ({stand_node_x*1e3:.3f}, {stand_node_y*1e3:.3f}) mm")
    print(f"  U_min = {U_stand_min:.3e} J    barrier = {barrier_stand:.3e} J")
    print(f"  {len(minima_stand)} minima found")

    # ── 2. Solve vortex for each (config, offset) ────────────────
    vortex_cache: Dict[Tuple[str, float], np.ndarray] = {}
    vortex_cores: Dict[Tuple[str, float], Tuple[float, float]] = {}

    for cfg_r in configs:
        cid = cfg_r["config_id"]
        print(f"\n{'─' * 72}")
        print(f"  Solving vortex offsets: {cid}")
        print(f"{'─' * 72}")

        for off_mm in OFFSETS_MM:
            off_m = off_mm * 1e-3
            label = f"vortex_{cid}_off{off_mm:.1f}mm".replace(".", "p")
            ov = config_to_overrides(cfg_r, standing_only=False,
                                     offset_x_m=off_m)
            data = solve_single_z(ov, label)
            if data is None:
                print(f"  *** FAILED: offset={off_mm} mm")
                continue

            p_v = data["p_xy_0"]
            vortex_cache[(cid, off_mm)] = p_v

            # Detect vortex core from field
            expected_x = CX + off_m
            core_x, core_y = detect_vortex_core(p_v, xg, yg,
                                                 expected_x, CY)
            vortex_cores[(cid, off_mm)] = (core_x, core_y)

            print(f"    off={off_mm:.1f}mm  max|p_v|={np.abs(p_v).max():.3f} Pa"
                  f"  core=({core_x*1e3:.2f},{core_y*1e3:.2f}) mm")

        gc.collect()

    # ── 3. Analytical α sweep ─────────────────────────────────────
    print("\n" + "=" * 72)
    print("  Analytical α sweep (linear superposition)")
    print("=" * 72)

    all_rows: List[Dict[str, Any]] = []
    tracking_data: Dict[Tuple[str, float], List[Dict]] = {}
    U_comb_cache: Dict[Tuple[str, float, float], np.ndarray] = {}

    # Basin competition interpolator on U_stand
    interp_U_stand = RegularGridInterpolator(
        (yg, xg), U_stand, method="linear",
        bounds_error=False, fill_value=np.nan)

    for cfg_r in configs:
        cid = cfg_r["config_id"]
        print(f"\n  Config: {cid}")

        for alpha in ALPHAS:
            track_records: List[Dict] = []
            # Reset tracker to standing node at offset = 0
            prev_x, prev_y = stand_node_x, stand_node_y

            for off_mm in OFFSETS_MM:
                if (cid, off_mm) not in vortex_cache:
                    continue

                p_v = vortex_cache[(cid, off_mm)]
                p_comb = p_stand + alpha * V_RATIO * p_v
                U_comb = gorkov_2d(p_comb, dx, dy)

                # Cache U_comb for landscape figures (only selected combos)
                if alpha in ALPHAS_SHOW and off_mm in OFFSETS_SHOW:
                    U_comb_cache[(cid, alpha, off_mm)] = U_comb

                # Track minimum
                tix, tiy = track_minimum(U_comb, xg, yg, prev_x, prev_y)
                lost = (tix is None)

                if not lost:
                    min_x = float(xg[tix])
                    min_y = float(yg[tiy])
                    U_min = float(U_comb[tiy, tix])
                    kxx, kyy = curvature_at(U_comb, xg, yg, tix, tiy)
                    barrier = compute_barrier(U_comb, xg, yg, tix, tiy)
                    prev_x, prev_y = min_x, min_y
                else:
                    min_x, min_y = prev_x, prev_y
                    U_min, kxx, kyy, barrier = np.nan, np.nan, np.nan, np.nan

                # Vortex core position
                core_x, core_y = vortex_cores.get(
                    (cid, off_mm), (CX + off_mm * 1e-3, CY))

                # Basin competition: U_comb at vortex core vs standing node
                interp_U_comb = RegularGridInterpolator(
                    (yg, xg), U_comb, method="linear",
                    bounds_error=False, fill_value=np.nan)
                U_at_core = float(interp_U_comb([[core_y, core_x]])[0])
                U_at_node = float(interp_U_comb([[stand_node_y, stand_node_x]])[0])

                # Far-field preservation (only for reference, dominated by α)
                eta = compute_eta_out(p_stand, p_comb, xg, yg)
                corr = compute_corr_out(p_stand, p_comb, xg, yg)

                row = {
                    "config_id": cid,
                    "alpha": alpha,
                    "offset_commanded_mm": off_mm,
                    "min_x_mm": min_x * 1e3,
                    "min_y_mm": min_y * 1e3,
                    "min_displacement_x_mm": (min_x - stand_node_x) * 1e3,
                    "tracking_error_mm": abs(
                        (min_x - stand_node_x) * 1e3 - off_mm),
                    "U_min": U_min,
                    "barrier_depth": barrier,
                    "barrier_ratio": barrier / barrier_stand if barrier_stand > 0 else np.nan,
                    "curvature_xx": kxx,
                    "curvature_yy": kyy,
                    "core_x_mm": core_x * 1e3,
                    "core_y_mm": core_y * 1e3,
                    "U_at_core": U_at_core,
                    "U_at_node": U_at_node,
                    "core_deeper": bool(U_at_core < U_at_node),
                    "eta_out": eta,
                    "corr_out": corr,
                    "lost_tracking": lost,
                }
                track_records.append(row)
                all_rows.append(row)

            tracking_data[(cid, alpha)] = track_records

            # Summary line
            disps = [r["min_displacement_x_mm"] for r in track_records
                     if not r["lost_tracking"]]
            if disps:
                max_disp = max(abs(d) for d in disps)
            else:
                max_disp = 0.0
            eta_last = track_records[-1]["eta_out"] if track_records else np.nan
            print(f"    α={alpha:5.1f}  max|disp|={max_disp:.3f}mm"
                  f"  η_out={eta_last:.5f}")

    # ── 4. Compute mobility ratios ────────────────────────────────
    print("\n" + "=" * 72)
    print("  Computing mobility ratios")
    print("=" * 72)

    mobility_data: Dict[str, Dict[float, float]] = {}
    eta_data: Dict[str, Dict[float, float]] = {}
    alpha_crits: Dict[str, Optional[float]] = {}

    for cfg_r in configs:
        cid = cfg_r["config_id"]
        mob_by_alpha: Dict[float, float] = {}
        eta_by_alpha: Dict[float, float] = {}

        for alpha in ALPHAS:
            key = (cid, alpha)
            recs = tracking_data.get(key, [])
            if len(recs) < 2:
                mob_by_alpha[alpha] = 0.0
                eta_by_alpha[alpha] = np.nan
                continue

            offs = np.array([r["offset_commanded_mm"] for r in recs
                             if not r["lost_tracking"]])
            disps = np.array([r["min_displacement_x_mm"] for r in recs
                              if not r["lost_tracking"]])

            if len(offs) >= 2:
                coeffs = np.polyfit(offs, disps, 1)
                mob_by_alpha[alpha] = float(coeffs[0])
            else:
                mob_by_alpha[alpha] = 0.0

            # η_out: average over offsets
            etas = [r["eta_out"] for r in recs if np.isfinite(r["eta_out"])]
            eta_by_alpha[alpha] = float(np.mean(etas)) if etas else np.nan

        mobility_data[cid] = mob_by_alpha
        eta_data[cid] = eta_by_alpha

        # Find α_crit
        a_crit = None
        for a in sorted(ALPHAS):
            if mob_by_alpha.get(a, 0) >= MOBILITY_THRESHOLD:
                a_crit = a
                break
        alpha_crits[cid] = a_crit

        print(f"  {cid}:")
        for a in ALPHAS:
            mob = mob_by_alpha.get(a, 0)
            eta = eta_by_alpha.get(a, np.nan)
            tag = " <<<" if a == a_crit else ""
            print(f"    α={a:5.1f}  mobility={mob:+.3f}  η_out={eta:.5f}{tag}")
        if a_crit:
            print(f"    → α_crit = {a_crit}")
        else:
            print(f"    → α_crit NOT REACHED")

    # ── 5. Generate figures ───────────────────────────────────────
    print("\n  Generating figures ...")

    for cfg_r in configs:
        cid = cfg_r["config_id"]
        fig_offset_tracking(cid, tracking_data,
                            stand_node_x * 1e3,
                            vortex_cores)
        fig_U_landscape(cid, U_comb_cache, xg, yg, U_stand,
                        tracking_data, stand_node_x, stand_node_y,
                        vortex_cores)

        # Pressure panels at two α levels: 1.0 and highest showing sliding
        for alpha_p in [1.0, 10.0]:
            fig_pressure_panels(
                cid, p_stand, xg, yg,
                vortex_cache, alpha_p,
                [0.0, 0.3, 0.6, 0.8],
                vortex_cores, stand_node_x, stand_node_y)

    fig_mobility_summary(mobility_data, eta_data, configs)

    # ── 6. Write CSV, JSON, note ──────────────────────────────────
    write_csv(all_rows)

    results_json = {
        "params": {
            "alphas": ALPHAS,
            "offsets_mm": OFFSETS_MM,
            "epl": EPL,
            "grid": GRID_N,
            "z_star_mm": Z_STAR * 1e3,
            "search_radius_mm": SEARCH_RADIUS * 1e3,
            "V_ratio": V_RATIO,
            "mobility_threshold": MOBILITY_THRESHOLD,
        },
        "configs": [c["config_id"] for c in configs],
        "standing_node_mm": [stand_node_x * 1e3, stand_node_y * 1e3],
        "barrier_stand_J": barrier_stand,
        "mobility": {
            cid: {str(a): mobility_data[cid][a] for a in ALPHAS}
            for cid in mobility_data
        },
        "alpha_crit": {cid: alpha_crits.get(cid) for cid in alpha_crits},
        "eta_out": {
            cid: {str(a): eta_data[cid][a] for a in ALPHAS}
            for cid in eta_data
        },
        "tracking": all_rows,
        "total_time_s": time.time() - t0,
    }
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(results_json, f, indent=2, default=str)

    _write_note(configs, all_rows, mobility_data, eta_data,
                alpha_crits, stand_node_x * 1e3, stand_node_y * 1e3,
                barrier_stand, time.time() - t0)

    total = time.time() - t0
    print(f"\n{'=' * 72}")
    print(f"  Minimum-mobility investigation complete.  {total:.0f}s  "
          f"({len(all_rows)} tracking rows)")
    print(f"  Results: {OUT_DIR}")
    print(f"{'=' * 72}")


def main():
    run_sweep()


if __name__ == "__main__":
    main()
