#!/usr/bin/env python3
"""
Vortex Convergence Spot-Check  (Phase 2 gate)
==============================================

Reads Phase-1 sweep results, selects top propagating-only finalists,
and runs each at multiple mesh / grid resolutions to confirm ranking
stability, vortex authority, and produce high-resolution figures.

Usage:
    micromamba run -n acousto-complex python scripts/experiments/vortex_convergence_spotcheck.py
    micromamba run -n acousto-complex python scripts/experiments/vortex_convergence_spotcheck.py --n_finalists 3

Outputs:
    results/vortex_convergence_spotcheck_YYYYMMDD_HHMMSS/
        csv/spotcheck_metrics.csv
        csv/spotcheck_summary.csv
        results.json
        figures/{config_id}/...
        VORTEX_CONVERGENCE_SPOTCHECK_NOTE.md
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
from scipy.interpolate import RegularGridInterpolator

# ── Project path ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from acoustweezers.experiments.farfield_petri_cuboid.presets import (
    CORRECTED_PRESET,
)

# =====================================================================
# Constants  (mirrored from vortex_lens_sweep.py)
# =====================================================================
WATER_DENSITY = 997.0
WATER_C       = 1484.0
F_HZ          = 2.0e6
OMEGA         = 2 * np.pi * F_HZ
LAM           = WATER_C / F_HZ           # 0.742 mm
K_WATER       = OMEGA / WATER_C           # 8467.9 rad/m
R_DISK        = 1.0e-3

PARTICLE_RADIUS = 5e-6
RHO_P, C_P = 1050.0, 2350.0
F1_CONTRAST = 1 - WATER_DENSITY * WATER_C**2 / (RHO_P * C_P**2)
F2_CONTRAST = 2 * (RHO_P - WATER_DENSITY) / (2 * RHO_P + WATER_DENSITY)

H_UNDER = CORRECTED_PRESET["H_under"]
H_TOP   = CORRECTED_PRESET.get("H_top", 2.0085e-3)
Z_MID   = H_UNDER + H_TOP / 2
Z_STAR  = Z_MID + 0.25 * LAM                              # 4.190 mm
Z_PLANES = [Z_STAR - 0.25 * LAM, Z_STAR, Z_STAR + 0.25 * LAM]
Z_LABELS = ["z_minus", "z_star", "z_plus"]

CX, CY = 3.0e-3, 3.0e-3
ROI_RADIUS = 1.5 * LAM

ALPHA_DEFAULT = 0.20
V_RATIO = CORRECTED_PRESET["standing_velocity_amplitude"] / \
          CORRECTED_PRESET["disk_velocity_amplitude"]       # = 10

# Phase-1 thresholds
ETA_THR  = 0.02
CORR_THR = 0.995
W1, W2, W3, W4, W5 = 0.20, 0.15, 0.25, 0.25, 0.15

# Authority floor
AUTHORITY_E_ROI  = 0.05
AUTHORITY_PEAK_PA = 1.0

# Convergence grid
EPL_VALUES = [4, 5]       # elem_per_lambda; 6 may OOM
GRID_VALUES = [200, 400]  # post-processing grid

PHASE1_DIR = PROJECT_ROOT / "results" / "vortex_lens_sweep_20260225_152950"

WORKER_SCRIPT = PROJECT_ROOT / "scripts" / "experiments" / "_solve_worker_multi_z.py"

# =====================================================================
# Output directory
# =====================================================================
TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
OUT_DIR   = PROJECT_ROOT / "results" / f"vortex_convergence_spotcheck_{TIMESTAMP}"
FIG_DIR   = OUT_DIR / "figures"
CSV_DIR   = OUT_DIR / "csv"

# High-res figure defaults
FIG_DPI   = 300
FIG_LARGE = (14, 11)
FIG_WIDE  = (18, 8)


# =====================================================================
# Finalist selection
# =====================================================================

def select_finalists(n: int = 5) -> List[Dict[str, Any]]:
    """Read Phase-1 results.json, return top-n propagating-only configs."""
    with open(PHASE1_DIR / "results.json") as f:
        data = json.load(f)

    ranked = data["ranked_list"]
    finalists = []
    for r in ranked:
        cid = r.get("config_id", "")
        j   = r.get("J", -1)
        if j <= 0:
            continue
        # Hard-exclude evanescent
        if "kr12702" in cid or "kr8468" in cid:
            continue
        finalists.append(r)
        if len(finalists) >= n:
            break

    print(f"  Selected {len(finalists)} finalists (propagating only):")
    for i, r in enumerate(finalists):
        print(f"    {i+1}. {r['config_id']}  J={r['J']:.4f}  "
              f"η={r['eta_out']:.5f}  E_roi={r['E_roi']:.3f}")
    return finalists


# =====================================================================
# Config builder (from finalist metadata)
# =====================================================================

def finalist_to_overrides(r: Dict, epl: int, standing_only: bool = False) -> dict:
    """Convert a Phase-1 result row back to FarFieldConfig overrides."""
    cfg = {**CORRECTED_PRESET, "elements_per_wavelength": epl}

    if standing_only:
        cfg["disk_velocity_amplitude"] = 0.0
        return cfg

    # Vortex-only
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
        fl = r.get("focal_length_mm", 2.0)
        cfg["lens_focal_length"] = fl * 1e-3
    return cfg


# =====================================================================
# Subprocess solver (multi-z worker)
# =====================================================================

def solve_multi_z(overrides: dict, label: str,
                  z_list: List[float],
                  mid_y: float = CY,
                  n_xy: int = 200) -> Optional[dict]:
    """
    Solve via subprocess, return dict with:
        xg, yg, p_xy_0..2, xg_xz, zg_xz, p_xz, phys_max, solve_time
    Caches .npz in OUT_DIR.
    """
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
            print(f"  *** SOLVE FAILED: {label}  rc={proc.returncode}",
                  flush=True)
            return None
    finally:
        os.unlink(args_file)

    return dict(np.load(result_file, allow_pickle=False))


# =====================================================================
# Physics helpers  (same as sweep)
# =====================================================================

def gorkov_2d(p_grid, dx, dy):
    K = WATER_DENSITY * WATER_C**2
    p2 = np.abs(p_grid)**2 / 2
    dp_dx = np.gradient(p_grid, dx, axis=1)
    dp_dy = np.gradient(p_grid, dy, axis=0)
    grad_p2 = np.abs(dp_dx)**2 + np.abs(dp_dy)**2
    v2 = grad_p2 / (2 * OMEGA**2 * WATER_DENSITY**2)
    pf = (4 * np.pi / 3) * PARTICLE_RADIUS**3
    U = pf * (F1_CONTRAST * p2 / (2 * K) -
              F2_CONTRAST * (3 * WATER_DENSITY / 4) * v2)
    return U


def make_roi(xg, yg):
    X, Y = np.meshgrid(xg, yg)
    return (X - CX)**2 + (Y - CY)**2 <= ROI_RADIUS**2


def make_annulus(xg, yg, r_in, r_out):
    X, Y = np.meshgrid(xg, yg)
    d2 = (X - CX)**2 + (Y - CY)**2
    return (d2 >= r_in**2) & (d2 <= r_out**2)


def combine(p_stand, p_vortex, alpha=ALPHA_DEFAULT):
    return p_stand + alpha * V_RATIO * p_vortex


def winding_number(p_grid, xg, yg, radius, n_angles=360):
    phase = np.angle(p_grid)
    interp = RegularGridInterpolator((yg, xg), phase,
                                     method='nearest',
                                     bounds_error=False, fill_value=0.0)
    theta = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    pts = np.column_stack([CY + radius * np.sin(theta),
                           CX + radius * np.cos(theta)])
    phi = interp(pts)
    dphi = np.diff(phi)
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
    return float(np.sum(dphi) / (2 * np.pi))


# =====================================================================
# Metrics   (streamlined from sweep, with authority floor)
# =====================================================================

def compute_metrics(p_stand, p_vortex, xg, yg, alpha=ALPHA_DEFAULT) -> Dict[str, Any]:
    dx = xg[1] - xg[0]
    dy = yg[1] - yg[0]
    roi = make_roi(xg, yg)
    out = ~roi

    p_comb = combine(p_stand, p_vortex, alpha)

    m: Dict[str, Any] = {}

    # Non-disruption
    delta = p_comb - p_stand
    norm_s = np.sqrt(np.sum(np.abs(p_stand[out])**2))
    norm_d = np.sqrt(np.sum(np.abs(delta[out])**2))
    m["eta_out"]  = float(norm_d / norm_s) if norm_s > 0 else float('nan')
    s_out = np.abs(p_stand[out])
    c_out = np.abs(p_comb[out])
    if np.std(s_out) > 0 and np.std(c_out) > 0:
        m["corr_out"] = float(np.corrcoef(s_out, c_out)[0, 1])
    else:
        m["corr_out"] = float('nan')

    # Vortex quality
    v_energy = np.abs(p_vortex)**2
    total_v  = np.sum(v_energy)
    m["E_roi"]       = float(np.sum(v_energy[roi]) / total_v) if total_v > 0 else 0.0
    p_v_mag          = np.abs(p_vortex)
    m["peak_roi_Pa"] = float(p_v_mag[roi].max()) if roi.any() else 0.0
    m["core_min_Pa"] = float(p_v_mag[roi].min()) if roi.any() else 0.0

    # Core ratio
    annulus = make_annulus(xg, yg, 1.5 * LAM, 3.0 * LAM)
    peak_ann = float(p_v_mag[annulus].max()) if annulus.any() else 1.0
    m["core_ratio"] = float(m["peak_roi_Pa"] / peak_ann) if peak_ann > 0 else float('inf')

    # Winding
    w_vals = []
    for rm in [1.0, 1.5, 2.0]:
        w = winding_number(p_vortex, xg, yg, rm * LAM)
        m[f"w_{rm:.1f}lam"] = round(w, 3)
        w_vals.append(w)
    m["winding_std"] = float(np.std([abs(w) for w in w_vals]))

    # Authority floor
    m["authority_pass"] = bool(m["E_roi"] >= AUTHORITY_E_ROI or
                               m["peak_roi_Pa"] >= AUTHORITY_PEAK_PA)

    # Composite J
    if m["eta_out"] > ETA_THR or m["corr_out"] < CORR_THR:
        m["pass_hard"] = False
        m["J"] = -1.0
    elif not m["authority_pass"]:
        m["pass_hard"] = True
        m["J"] = -1.0   # authority floor fail
    else:
        m["pass_hard"] = True
        ws = min(m["winding_std"], 1.0)
        cr = min(m["core_ratio"], 10.0) / 10.0
        m["J"] = (W1 * (1 - m["eta_out"] / ETA_THR) +
                  W2 * m["corr_out"] +
                  W3 * m["E_roi"] +
                  W4 * cr +
                  W5 * (1 - ws))

    # Standing max for reference
    m["max_p_stand_Pa"] = float(np.abs(p_stand).max())
    m["max_p_vortex_Pa"] = float(p_v_mag.max())

    return m


# =====================================================================
# High-resolution figure generators
# =====================================================================

def _extent_mm(xg, yg):
    return [xg[0]*1e3, xg[-1]*1e3, yg[0]*1e3, yg[-1]*1e3]


def _add_roi_circle(ax, **kw):
    ax.add_patch(Circle((CX*1e3, CY*1e3), ROI_RADIUS*1e3,
                         fill=False, ec=kw.get("ec", "cyan"),
                         lw=kw.get("lw", 1.2), ls="--"))


def fig_vortex_xy(p_vortex, xg, yg, config_id, epl, grid_n, fig_dir):
    """XY: |p_vortex| and phase at z*."""
    ext = _extent_mm(xg, yg)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=FIG_DPI)

    # |p_vortex|
    ax = axes[0]
    im = ax.imshow(np.abs(p_vortex), extent=ext, origin='lower',
                   cmap='inferno', interpolation='nearest')
    _add_roi_circle(ax)
    ax.set_title(f"|p_vortex| at z*", fontsize=10)
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    cb = plt.colorbar(im, ax=ax, shrink=0.75)
    cb.set_label("Pa")

    # Mark core (min |p| in ROI)
    roi = make_roi(xg, yg)
    pmag = np.abs(p_vortex).copy()
    pmag[~roi] = np.inf
    iy, ix = np.unravel_index(pmag.argmin(), pmag.shape)
    ax.plot(xg[ix]*1e3, yg[iy]*1e3, 'w+', ms=12, mew=2.5)

    # Centroid marker
    v_e = np.abs(p_vortex)**2
    X, Y = np.meshgrid(xg, yg)
    if np.sum(v_e[roi]) > 0:
        cx_c = np.sum(X[roi]*v_e[roi]) / np.sum(v_e[roi])
        cy_c = np.sum(Y[roi]*v_e[roi]) / np.sum(v_e[roi])
        ax.plot(cx_c*1e3, cy_c*1e3, 'cx', ms=10, mew=2)

    # phase
    ax = axes[1]
    im = ax.imshow(np.angle(p_vortex), extent=ext, origin='lower',
                   cmap='twilight', vmin=-np.pi, vmax=np.pi,
                   interpolation='nearest')
    _add_roi_circle(ax)
    ax.plot(xg[ix]*1e3, yg[iy]*1e3, 'w+', ms=12, mew=2.5)
    ax.set_title("arg(p_vortex) at z*", fontsize=10)
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    cb = plt.colorbar(im, ax=ax, shrink=0.75)
    cb.set_label("rad")

    fig.suptitle(f"{config_id}  epl={epl}  grid={grid_n}", fontsize=11)
    fig.tight_layout()
    fname = fig_dir / f"vortex_xy_zstar_epl{epl}_grid{grid_n}.png"
    fig.savefig(str(fname), dpi=FIG_DPI)
    plt.close(fig)
    return fname


def fig_vortex_xz(p_xz, xg_xz, zg_xz, config_id, epl, grid_n, fig_dir):
    """XZ: |p_vortex| with z* line."""
    ext = [xg_xz[0]*1e3, xg_xz[-1]*1e3, zg_xz[0]*1e3, zg_xz[-1]*1e3]
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=FIG_DPI)
    im = ax.imshow(np.abs(p_xz), extent=ext, origin='lower',
                   cmap='inferno', aspect='auto', interpolation='nearest')
    ax.axhline(y=Z_STAR*1e3, color='cyan', ls='--', lw=1.2, label=f"z*={Z_STAR*1e3:.2f} mm")
    ax.axhline(y=H_UNDER*1e3, color='white', ls=':', lw=0.8, alpha=0.5, label="petri bottom")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("z (mm)")
    ax.set_title(f"|p_vortex| XZ (y=Ly/2)   {config_id}  epl={epl}  grid={grid_n}", fontsize=10)
    cb = plt.colorbar(im, ax=ax, shrink=0.75)
    cb.set_label("Pa")
    ax.legend(fontsize=8, loc='upper right')
    fig.tight_layout()
    fname = fig_dir / f"vortex_xz_yMid_epl{epl}_grid{grid_n}.png"
    fig.savefig(str(fname), dpi=FIG_DPI)
    plt.close(fig)
    return fname


def fig_comb_xy(p_stand, p_vortex, xg, yg, config_id, epl, grid_n, fig_dir):
    """XY: |p_stand|, |p_comb|, Δ|p|."""
    p_comb = combine(p_stand, p_vortex)
    dp = np.abs(p_comb) - np.abs(p_stand)
    ext = _extent_mm(xg, yg)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=FIG_DPI)

    # |p_stand|
    ax = axes[0]
    im = ax.imshow(np.abs(p_stand), extent=ext, origin='lower', cmap='hot',
                   interpolation='nearest')
    _add_roi_circle(ax, ec='cyan')
    ax.set_title("|p_stand|  (z*)", fontsize=10)
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    plt.colorbar(im, ax=ax, shrink=0.7).set_label("Pa")

    # |p_comb|
    ax = axes[1]
    im = ax.imshow(np.abs(p_comb), extent=ext, origin='lower', cmap='hot',
                   interpolation='nearest')
    _add_roi_circle(ax, ec='cyan')
    ax.set_title(f"|p_comb|  α={ALPHA_DEFAULT}", fontsize=10)
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    plt.colorbar(im, ax=ax, shrink=0.7).set_label("Pa")

    # Δ|p|
    ax = axes[2]
    vmax = max(abs(dp.min()), abs(dp.max())) or 1
    im = ax.imshow(dp, extent=ext, origin='lower', cmap='RdBu_r',
                   vmin=-vmax, vmax=vmax, interpolation='nearest')
    _add_roi_circle(ax, ec='lime', lw=2)
    ax.set_title("Δ|p| = |p_comb| − |p_stand|", fontsize=10)
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    plt.colorbar(im, ax=ax, shrink=0.7).set_label("Pa")

    fig.suptitle(f"{config_id}   epl={epl}  grid={grid_n}  α={ALPHA_DEFAULT}",
                 fontsize=11)
    fig.tight_layout()
    fname = fig_dir / f"comb_xy_zstar_epl{epl}_grid{grid_n}.png"
    fig.savefig(str(fname), dpi=FIG_DPI)
    plt.close(fig)
    return fname


def fig_comb_xz(p_stand_xz, p_vortex_xz, xg_xz, zg_xz,
                config_id, epl, grid_n, fig_dir):
    """XZ: |p_stand|, |p_comb|, Δ|p| side-by-side."""
    p_comb_xz = combine(p_stand_xz, p_vortex_xz)
    dp = np.abs(p_comb_xz) - np.abs(p_stand_xz)
    ext = [xg_xz[0]*1e3, xg_xz[-1]*1e3, zg_xz[0]*1e3, zg_xz[-1]*1e3]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=FIG_DPI)

    for i, (data, title, cmap) in enumerate([
        (np.abs(p_stand_xz),  "|p_stand| XZ",   "hot"),
        (np.abs(p_comb_xz),   "|p_comb| XZ",    "hot"),
        (dp,                   "Δ|p| XZ",        "RdBu_r"),
    ]):
        ax = axes[i]
        if cmap == "RdBu_r":
            vmax = max(abs(dp.min()), abs(dp.max())) or 1
            im = ax.imshow(data, extent=ext, origin='lower', cmap=cmap,
                           aspect='auto', vmin=-vmax, vmax=vmax,
                           interpolation='nearest')
        else:
            im = ax.imshow(data, extent=ext, origin='lower', cmap=cmap,
                           aspect='auto', interpolation='nearest')
        ax.axhline(y=Z_STAR*1e3, color='cyan', ls='--', lw=1, label="z*")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x (mm)"); ax.set_ylabel("z (mm)")
        cb = plt.colorbar(im, ax=ax, shrink=0.7)
        cb.set_label("Pa")

    fig.suptitle(f"{config_id}  XZ (y=Ly/2)  epl={epl}  grid={grid_n}  α={ALPHA_DEFAULT}",
                 fontsize=11)
    fig.tight_layout()
    fname = fig_dir / f"comb_xz_yMid_epl{epl}_grid{grid_n}.png"
    fig.savefig(str(fname), dpi=FIG_DPI)
    plt.close(fig)
    return fname


def fig_resolution_compare(data_lo, data_hi, xg_lo, yg_lo, xg_hi, yg_hi,
                           config_id, epl, fig_dir):
    """Side-by-side grid 200 vs 400: Δ|p| and |p_vortex|."""
    fig, axes = plt.subplots(2, 2, figsize=FIG_LARGE, dpi=FIG_DPI)

    for col, (xg, yg, dset, gn) in enumerate([
        (xg_lo, yg_lo, data_lo, 200),
        (xg_hi, yg_hi, data_hi, 400),
    ]):
        ext = _extent_mm(xg, yg)
        pv = dset["p_vortex"]
        ps = dset["p_stand"]
        dp = np.abs(combine(ps, pv)) - np.abs(ps)

        # Row 0: |p_vortex|
        ax = axes[0, col]
        im = ax.imshow(np.abs(pv), extent=ext, origin='lower', cmap='inferno',
                       interpolation='nearest')
        _add_roi_circle(ax)
        ax.set_title(f"|p_vortex|  grid={gn}", fontsize=10)
        ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
        plt.colorbar(im, ax=ax, shrink=0.65)

        # Row 1: Δ|p|
        ax = axes[1, col]
        vmax = max(abs(dp.min()), abs(dp.max())) or 1
        im = ax.imshow(dp, extent=ext, origin='lower', cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax, interpolation='nearest')
        _add_roi_circle(ax, ec='lime', lw=2)
        ax.set_title(f"Δ|p|  grid={gn}", fontsize=10)
        ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
        plt.colorbar(im, ax=ax, shrink=0.65)

    fig.suptitle(f"Resolution comparison: {config_id}  epl={epl}  "
                 f"grid 200 vs 400", fontsize=11)
    fig.tight_layout()
    fname = fig_dir / f"resolution_compare_xy_epl{epl}_grid200_vs_400.png"
    fig.savefig(str(fname), dpi=FIG_DPI)
    plt.close(fig)
    return fname


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Vortex convergence spot-check")
    parser.add_argument("--n_finalists", type=int, default=5)
    args = parser.parse_args()

    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Vortex Convergence Spot-Check")
    print(f"Output: {OUT_DIR}")
    print(f"z* = {Z_STAR*1e3:.4f} mm   z-planes: "
          + ", ".join(f"{z*1e3:.3f}" for z in Z_PLANES) + " mm")
    print(f"epl = {EPL_VALUES}   grids = {GRID_VALUES}")
    print("=" * 72)

    # ── Select finalists ──────────────────────────────────────────
    finalists = select_finalists(args.n_finalists)
    if not finalists:
        print("No finalists found!"); return

    # ── Main loop: epl × grid × finalist ─────────────────────────
    all_rows = []              # CSV rows
    config_best = {}           # config_id → list of (epl, grid_n, z_label, metrics)

    for epl in EPL_VALUES:
        print(f"\n{'='*72}")
        print(f"  elem_per_lambda = {epl}")
        print(f"{'='*72}")

        for grid_n in GRID_VALUES:
            tag = f"epl{epl}_g{grid_n}"
            print(f"\n  --- grid = {grid_n} ---")

            # Solve standing ONCE per (epl, grid_n)
            stand_label = f"standing_{tag}"
            stand_cfg = {**CORRECTED_PRESET,
                         "elements_per_wavelength": epl,
                         "disk_velocity_amplitude": 0.0}
            print(f"  Solving standing ({stand_label}) ...")
            stand_data = solve_multi_z(stand_cfg, stand_label,
                                       Z_PLANES, CY, grid_n)
            if stand_data is None:
                print(f"  FATAL: standing solve failed at epl={epl} grid={grid_n}")
                continue
            print(f"    Standing: max|p|={float(stand_data['phys_max']):.3f} Pa  "
                  f"t={float(stand_data['solve_time']):.1f}s")

            xg_s = stand_data["xg"]
            yg_s = stand_data["yg"]

            # ── Per finalist ──────────────────────────────────────
            for fi, r in enumerate(finalists):
                cid = r["config_id"]
                print(f"\n    [{fi+1}/{len(finalists)}] {cid}  "
                      f"(epl={epl}, grid={grid_n})")

                cfg_dir = FIG_DIR / cid
                cfg_dir.mkdir(parents=True, exist_ok=True)

                # Solve vortex
                vlabel = f"vortex_{cid}_{tag}"
                vorides = finalist_to_overrides(r, epl, standing_only=False)
                vdata = solve_multi_z(vorides, vlabel, Z_PLANES, CY, grid_n)
                if vdata is None:
                    print(f"      SKIP: solve failed")
                    continue
                print(f"      max|p_v|={float(vdata['phys_max']):.3f} Pa  "
                      f"t={float(vdata['solve_time']):.1f}s")

                # ── Process each z-plane ──────────────────────────
                for zi, (zp, zlab) in enumerate(zip(Z_PLANES, Z_LABELS)):
                    p_stand  = stand_data[f"p_xy_{zi}"]
                    p_vortex = vdata[f"p_xy_{zi}"]

                    m = compute_metrics(p_stand, p_vortex,
                                        xg_s, yg_s, ALPHA_DEFAULT)
                    m["config_id"]  = cid
                    m["family"]     = r["family"]
                    m["ell"]        = r["ell"]
                    m["epl"]        = epl
                    m["grid_n"]     = grid_n
                    m["z_plane"]    = zlab
                    m["z_mm"]       = zp * 1e3
                    m["alpha"]      = ALPHA_DEFAULT

                    all_rows.append(m)

                    key = (cid, epl, grid_n, zlab)
                    config_best.setdefault(cid, []).append(
                        (epl, grid_n, zlab, m))

                # ── Figures (z* only, i.e. zi=1) ─────────────────
                p_s_star = stand_data["p_xy_1"]
                p_v_star = vdata["p_xy_1"]

                fig_vortex_xy(p_v_star, xg_s, yg_s, cid, epl, grid_n, cfg_dir)
                fig_comb_xy(p_s_star, p_v_star, xg_s, yg_s, cid, epl, grid_n, cfg_dir)

                # XZ figures
                p_s_xz = stand_data["p_xz"]
                p_v_xz = vdata["p_xz"]
                xg_xz  = vdata["xg_xz"]
                zg_xz  = vdata["zg_xz"]
                fig_vortex_xz(p_v_xz, xg_xz, zg_xz, cid, epl, grid_n, cfg_dir)
                fig_comb_xz(p_s_xz, p_v_xz, xg_xz, zg_xz, cid, epl, grid_n, cfg_dir)

                print(f"      Figures saved → {cfg_dir.name}/")

                # ── Resolution comparison (when grid_n=400, pair with 200)
                if grid_n == 400:
                    # Try to load 200 data
                    tag_lo = f"epl{epl}_g200"
                    f_s_lo = OUT_DIR / f"_grid_standing_{tag_lo}.npz"
                    f_v_lo = OUT_DIR / f"_grid_vortex_{cid}_{tag_lo}.npz"
                    if f_s_lo.exists() and f_v_lo.exists():
                        sd_lo = dict(np.load(str(f_s_lo), allow_pickle=False))
                        vd_lo = dict(np.load(str(f_v_lo), allow_pickle=False))
                        data_lo = {"p_vortex": vd_lo["p_xy_1"],
                                   "p_stand": sd_lo["p_xy_1"]}
                        data_hi = {"p_vortex": p_v_star,
                                   "p_stand": p_s_star}
                        fig_resolution_compare(
                            data_lo, data_hi,
                            sd_lo["xg"], sd_lo["yg"],
                            xg_s, yg_s,
                            cid, epl, cfg_dir)
                        print(f"      Resolution compare saved")
                        del sd_lo, vd_lo, data_lo, data_hi

                # Free memory
                del vdata
                gc.collect()

            # Free standing data for this grid
            del stand_data
            gc.collect()

    # =====================================================================
    # CSV outputs
    # =====================================================================
    print(f"\n{'='*72}")
    print("Writing CSVs ...")
    print("=" * 72)

    csv_keys = [
        "config_id", "family", "ell", "epl", "grid_n", "z_plane", "z_mm",
        "alpha", "eta_out", "corr_out", "E_roi", "peak_roi_Pa", "core_min_Pa",
        "core_ratio", "w_1.0lam", "w_1.5lam", "w_2.0lam", "winding_std",
        "authority_pass", "pass_hard", "J",
        "max_p_stand_Pa", "max_p_vortex_Pa",
    ]
    with open(CSV_DIR / "spotcheck_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_keys, extrasaction='ignore')
        w.writeheader()
        for row in all_rows:
            w.writerow(row)
    print(f"  spotcheck_metrics.csv: {len(all_rows)} rows")

    # ── Summary CSV: stability per config ─────────────────────────
    summary_rows = []
    for cid in dict.fromkeys(r_["config_id"] for r_ in all_rows):
        rows_cid = [r_ for r_ in all_rows if r_["config_id"] == cid]
        # Filter to z_star only for ranking stability
        zs_rows = [r_ for r_ in rows_cid if r_["z_plane"] == "z_star"]

        if not zs_rows:
            continue

        j_vals = [r_["J"] for r_ in zs_rows]
        eta_vals = [r_["eta_out"] for r_ in zs_rows if not np.isnan(r_["eta_out"])]
        eroi_vals = [r_["E_roi"] for r_ in zs_rows]

        s = {
            "config_id": cid,
            "family": zs_rows[0]["family"],
            "ell": zs_rows[0]["ell"],
            "n_runs": len(zs_rows),
            "J_min": min(j_vals),
            "J_max": max(j_vals),
            "J_range": max(j_vals) - min(j_vals),
            "J_mean": np.mean(j_vals),
            "eta_min": min(eta_vals) if eta_vals else float('nan'),
            "eta_max": max(eta_vals) if eta_vals else float('nan'),
            "eta_range": (max(eta_vals)-min(eta_vals)) if eta_vals else float('nan'),
            "E_roi_min": min(eroi_vals),
            "E_roi_max": max(eroi_vals),
            "E_roi_range": max(eroi_vals) - min(eroi_vals),
            "all_pass": all(r_["pass_hard"] for r_ in zs_rows),
            "all_authority": all(r_["authority_pass"] for r_ in zs_rows),
        }

        # Rank stability: is J ordering preserved across resolutions?
        # Just record whether J is positive everywhere
        s["stable"] = bool(s["J_min"] > 0 and s["J_range"] < 0.15)
        summary_rows.append(s)

    sum_keys = list(summary_rows[0].keys()) if summary_rows else []
    with open(CSV_DIR / "spotcheck_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sum_keys, extrasaction='ignore')
        w.writeheader()
        w.writerows(summary_rows)
    print(f"  spotcheck_summary.csv: {len(summary_rows)} rows")

    # =====================================================================
    # JSON
    # =====================================================================
    def clean(v):
        if isinstance(v, (np.floating, float)):
            return None if np.isnan(v) else float(v)
        if isinstance(v, (np.integer, int)):
            return int(v)
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        return v

    verdicts = {}
    for s in summary_rows:
        verdicts[s["config_id"]] = {
            "stable": s["stable"],
            "all_pass": s["all_pass"],
            "all_authority": s["all_authority"],
            "J_range": clean(s["J_range"]),
            "J_mean": clean(s["J_mean"]),
            "eta_range": clean(s["eta_range"]),
        }

    output = {
        "params": {
            "epl_values": EPL_VALUES,
            "grid_values": GRID_VALUES,
            "z_planes_mm": [z*1e3 for z in Z_PLANES],
            "alpha": ALPHA_DEFAULT,
            "eta_threshold": ETA_THR,
            "corr_threshold": CORR_THR,
            "authority_E_roi": AUTHORITY_E_ROI,
            "authority_peak_Pa": AUTHORITY_PEAK_PA,
        },
        "finalists": [{k: clean(v) for k, v in r.items()} for r in finalists],
        "verdicts": verdicts,
        "summary": [{k: clean(v) for k, v in s.items()} for s in summary_rows],
        "total_time_s": time.time() - t0,
    }
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    # =====================================================================
    # Markdown note
    # =====================================================================
    stable_ids = [s["config_id"] for s in summary_rows if s["stable"]]
    unstable_ids = [s["config_id"] for s in summary_rows if not s["stable"]]
    no_auth = [s["config_id"] for s in summary_rows if not s["all_authority"]]

    note_lines = [
        "# Vortex Convergence Spot-Check — Phase 2 Gate\n",
        f"**Date:** {time.strftime('%Y-%m-%d')}  ",
        f"**Script:** `scripts/experiments/vortex_convergence_spotcheck.py`  ",
        f"**Results:** `{OUT_DIR.relative_to(PROJECT_ROOT)}/`  ",
        f"**Runtime:** {time.time()-t0:.0f} s\n",
        "---\n",
        "## 1. Objective\n",
        "Confirm that top propagating-beam finalists from the Phase-1 sweep are ",
        "stable under mesh and grid refinement, have real vortex authority at z*,",
        " and produce high-resolution figures for internal report.\n",
        "## 2. Setup\n",
        f"| Parameter | Value |",
        f"|---|---|",
        f"| Finalists | {len(finalists)} (propagating only, k_r ≤ 0.5k) |",
        f"| elem/λ | {EPL_VALUES} |",
        f"| Grid | {GRID_VALUES} |",
        f"| z-planes | z* − 0.25λ, z*, z* + 0.25λ ({', '.join(f'{z*1e3:.3f}' for z in Z_PLANES)} mm) |",
        f"| α | {ALPHA_DEFAULT} |",
        f"| Authority floor | E_roi ≥ {AUTHORITY_E_ROI} OR peak|p_v| ≥ {AUTHORITY_PEAK_PA} Pa |",
        "",
        "## 3. Finalists\n",
        "| # | config_id | Family | ℓ | Phase-1 J |",
        "|---|---|---|---|---|",
    ]
    for i, r in enumerate(finalists):
        note_lines.append(
            f"| {i+1} | `{r['config_id']}` | {r['family']} | {r['ell']} "
            f"| {r['J']:.4f} |")

    note_lines += [
        "",
        "## 4. Stability Results (z* plane)\n",
        "| config_id | J range | J mean | η range | Stable | Authority |",
        "|---|---|---|---|---|---|",
    ]
    for s in summary_rows:
        note_lines.append(
            f"| `{s['config_id']}` | {s['J_range']:.4f} | {s['J_mean']:.4f} "
            f"| {s['eta_range']:.5f} | {'YES' if s['stable'] else '**NO**'} "
            f"| {'YES' if s['all_authority'] else '**NO**'} |")

    note_lines += [
        "",
        "## 5. Verdict\n",
        f"**Stable winners:** {', '.join(f'`{c}`' for c in stable_ids) or 'NONE'}  ",
        f"**Unstable:** {', '.join(f'`{c}`' for c in unstable_ids) or 'none'}  ",
        f"**Authority-fail:** {', '.join(f'`{c}`' for c in no_auth) or 'none'}  \n",
    ]

    if stable_ids:
        note_lines.append("**GREEN LIGHT** — stable propagating configs confirmed for Phase 2.\n")
    else:
        note_lines.append("**YELLOW** — no fully stable configs; review J_range and figures.\n")

    note_lines += [
        "## 6. Figures\n",
        "Per-finalist figure set in `figures/{config_id}/`:\n",
        "- `vortex_xy_zstar_epl{N}_grid{M}.png` — |p_vortex| + phase",
        "- `vortex_xz_yMid_epl{N}_grid{M}.png` — |p_vortex| XZ midplane",
        "- `comb_xy_zstar_epl{N}_grid{M}.png` — |p_stand|, |p_comb|, Δ|p|",
        "- `comb_xz_yMid_epl{N}_grid{M}.png` — XZ combined comparison",
        "- `resolution_compare_xy_epl{N}_grid200_vs_400.png` — grid refinement",
    ]

    with open(OUT_DIR / "VORTEX_CONVERGENCE_SPOTCHECK_NOTE.md", "w") as f:
        f.write("\n".join(note_lines))

    total = time.time() - t0
    print(f"\n{'='*72}")
    print(f"Convergence spot-check complete.  {total:.0f}s  "
          f"({len(all_rows)} metric rows)")
    print(f"Stable: {stable_ids}")
    print(f"Results: {OUT_DIR}")
    print("=" * 72)


if __name__ == "__main__":
    main()
