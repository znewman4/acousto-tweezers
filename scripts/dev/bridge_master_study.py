#!/usr/bin/env python3
"""
Bridge Master Study — Large Parameter Space Exploration
========================================================

Expanded sweep over the full interference parameter space:
  α   ∈ {0.05, 0.10, …, 0.50}              10 values
  φ₀  ∈ 16 values in [0, 2π)
  x₀  ∈ {−0.5, −0.4, …, +0.5} × trap_sp   11 values
  y₀  ∈ {−0.5, −0.4, …, +0.5} × trap_sp   11 values

Optional extended sweeps (--extended):
  ℓ   ∈ {1, 2}
  apod ∈ {uniform, cosine_taper}
  focal perturbations around 4.2 mm

Field combination:
  p_total = p_stand + α · exp(i φ₀) · p̂_vort(x−x₀, y−y₀)

Bridge metric:
  B = U_saddle − max(U_min1, U_min2)
  B ≤ 0 ⟹ bridge formed

Outputs → results/bridge_master_study_TIMESTAMP/

Usage:
    python scripts/dev/bridge_master_study.py
    python scripts/dev/bridge_master_study.py --quick       # reduced grid
    python scripts/dev/bridge_master_study.py --extended    # + ℓ,apod sweep
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree
from scipy.ndimage import maximum_filter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── project root ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from acoustweezers.physics.acoustics.angular_spectrum import propagate_pressure_asm


# ═══════════════════════════════════════════════════════════════════
# Physical constants  (DO NOT CHANGE)
# ═══════════════════════════════════════════════════════════════════
C_WATER = 1484.0
F_HZ    = 2.0e6
K_WATER = 2.0 * np.pi * F_HZ / C_WATER
LAM     = C_WATER / F_HZ
TRAP_SP = LAM / 2.0
OMEGA   = 2.0 * np.pi * F_HZ
RHO0    = 997.0

LX = LY   = 6.0e-3
H_UNDER   = 3.0e-3
H_TOP     = 2.0085e-3
CX = CY   = LX / 2.0
Z_STAR    = H_UNDER + H_TOP / 2.0 + 0.25 * LAM

RHO_P   = 1050.0
C_P     = 2350.0
A_P     = 5.0e-6
V_P     = (4.0 / 3.0) * np.pi * A_P**3
KAPPA0  = 1.0 / (RHO0 * C_WATER**2)
KAPPA_P = 1.0 / (RHO_P * C_P**2)
F1      = 1.0 - KAPPA_P / KAPPA0
F2      = 2.0 * (RHO_P - RHO0) / (2.0 * RHO_P + RHO0)

ASM_N   = 512
ASM_PAD = 2
Z0_OFFSET = 0.0

ROI_HALF = 1.1 * LAM
NGRID    = 400
DZ_GRAD  = LAM / 15.0
K_IDW    = 16
BRIDGE_NPTS = 200
STANDING_CACHE_DIR = PROJECT_ROOT / "results" / "fem_standing_wave_cache"


# ═══════════════════════════════════════════════════════════════════
# Reusable helper functions
# ═══════════════════════════════════════════════════════════════════

def build_converging_drive(ell, focal, aperture, cx, cy,
                           asm_domain, asm_n, k, apod="uniform"):
    x0 = cx - asm_domain / 2.0
    y0 = cy - asm_domain / 2.0
    xg = np.linspace(x0, x0 + asm_domain, asm_n, endpoint=False)
    yg = np.linspace(y0, y0 + asm_domain, asm_n, endpoint=False)
    XX, YY = np.meshgrid(xg, yg)
    rx, ry = XX - cx, YY - cy
    r = np.sqrt(rx**2 + ry**2)
    theta = np.arctan2(ry, rx)
    phi = ell * theta - k * (np.sqrt(r**2 + focal**2) - focal)
    r_norm = r / aperture
    if apod == "uniform":
        A = np.where(r_norm <= 1.0, 1.0, 0.0)
    else:
        A = np.where(r_norm <= 1.0,
                     0.5 * (1.0 + np.cos(np.pi * r_norm)), 0.0)
    return xg, yg, A * np.exp(1j * phi)


def sample_idw(tree, p, pts, k=16, power=2.0, eps=1e-12):
    if k == 1:
        d, i = tree.query(pts, k=1)
        return p[i]
    d, i = tree.query(pts, k=k)
    w = 1.0 / (d**power + eps)
    w /= w.sum(axis=1, keepdims=True)
    return (p[i] * w).sum(axis=1)


def load_fem_cache(path: Path):
    d = np.load(path, allow_pickle=True)
    keys = list(d.keys())
    coords = d["coords"]
    if "p_real" in keys and "p_imag" in keys:
        p = d["p_real"].astype(np.float64) + 1j * d["p_imag"].astype(np.float64)
    elif "p" in keys:
        p = d["p"].astype(np.complex128)
    else:
        raise KeyError(f"Cannot find pressure (keys: {keys})")
    return coords, p


def find_latest_cache(d: Path) -> Path:
    fs = sorted(d.glob("*.npz"), key=lambda p: p.stat().st_mtime)
    if not fs:
        raise FileNotFoundError(f"No .npz in {d}")
    return fs[-1]


class Log:
    def __init__(self, fp=None):
        self._fp = fp
    def __call__(self, msg=""):
        print(msg)
        if self._fp:
            self._fp.write(msg + "\n")
            self._fp.flush()


def interp_asm_to_grid(p_full, xg_src, yg_src, XX_out, YY_out):
    pts = np.column_stack([YY_out.ravel(), XX_out.ravel()])
    ire = RegularGridInterpolator(
        (yg_src, xg_src), np.real(p_full),
        bounds_error=False, fill_value=0.0)
    iim = RegularGridInterpolator(
        (yg_src, xg_src), np.imag(p_full),
        bounds_error=False, fill_value=0.0)
    return (ire(pts) + 1j * iim(pts)).reshape(XX_out.shape)


def gorkov_potential(p_abs2, grad_p_mag2):
    E_pot = 0.25 * p_abs2 * KAPPA0
    E_kin = grad_p_mag2 / (4.0 * OMEGA**2 * RHO0)
    return V_P * (F1 * E_pot - 1.5 * F2 * E_kin)


def grad_p_mag2_xy(p, p_zp, p_zm, dx, dy, dz):
    dpy, dpx = np.gradient(p, dy, dx)
    dpz = (p_zp - p_zm) / (2.0 * dz)
    return np.abs(dpx)**2 + np.abs(dpy)**2 + np.abs(dpz)**2


def find_trap_minima(U, min_sep_pix=10, threshold_frac=0.3):
    neg_U = -U
    local_max = (neg_U == maximum_filter(neg_U, size=min_sep_pix))
    local_max &= (neg_U > threshold_frac * neg_U.max())
    iy, ix = np.where(local_max)
    return iy, ix


def find_adjacent_pair(U, xg, yg):
    iy, ix = find_trap_minima(U)
    if len(iy) < 2:
        return None
    n = len(iy)
    pairs = []
    for a in range(n):
        for b in range(a + 1, n):
            dy_m = yg[iy[a]] - yg[iy[b]]
            dx_m = xg[ix[a]] - xg[ix[b]]
            dist = np.sqrt(dx_m**2 + dy_m**2)
            angle = abs(np.arctan2(abs(dy_m), abs(dx_m)))
            if angle > np.pi / 6 and angle < np.pi / 3:
                continue
            pairs.append((dist, a, b))
    if not pairs:
        return None
    pairs.sort(key=lambda t: abs(t[0] - TRAP_SP))
    _, a, b = pairs[0]
    return (iy[a], ix[a]), (iy[b], ix[b])


def compute_bridge_metric(U, xg, yg, trap1, trap2, n_pts=BRIDGE_NPTS):
    iy1, ix1 = trap1
    iy2, ix2 = trap2
    x1, y1 = xg[ix1], yg[iy1]
    x2, y2 = xg[ix2], yg[iy2]
    frac = np.linspace(0, 1, n_pts)
    xline = x1 + frac * (x2 - x1)
    yline = y1 + frac * (y2 - y1)
    rgi = RegularGridInterpolator((yg, xg), U, method="linear",
                                  bounds_error=False, fill_value=np.nan)
    pts = np.column_stack([yline, xline])
    U_line = rgi(pts)
    U_min1, U_min2 = float(U_line[0]), float(U_line[-1])
    U_saddle = float(np.nanmax(U_line))
    B = U_saddle - max(U_min1, U_min2)
    return B, U_line, frac, U_min1, U_min2, U_saddle


# ═══════════════════════════════════════════════════════════════════
# Plotting functions (reusable, tight ROI)
# ═══════════════════════════════════════════════════════════════════

def _tight_imshow(ax, data, extent, **kwargs):
    """imshow with tight axis limits matching extent."""
    im = ax.imshow(data, extent=extent, origin="lower", aspect="equal", **kwargs)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    return im


def _trap_grid(ax, cx, cy, trap_sp, n=4):
    for off in np.arange(-n, n + 1) * trap_sp:
        ax.axhline((cy + off) * 1e3, color="white", lw=0.3, alpha=0.25)
        ax.axvline((cx + off) * 1e3, color="white", lw=0.3, alpha=0.25)


def plot_bridge_metric_heatmaps(results, alphas, phi_values, shift_fracs,
                                trap_depth, B_baseline, ext, fig_path):
    """B vs φ₀ (top) and B vs (x₀,y₀) (bottom) for each α."""
    n_a = len(alphas)
    fig, axes = plt.subplots(2, n_a, figsize=(min(4.0 * n_a, 36), 8),
                             squeeze=False)

    for ia, alpha in enumerate(alphas):
        recs_a = [r for r in results if r["alpha"] == alpha]

        # Top: B vs φ₀
        ax = axes[0, ia]
        B_phi = np.full(len(phi_values), np.inf)
        for ip, phi in enumerate(phi_values):
            sub = [r for r in recs_a if abs(r["phi0"] - phi) < 0.01]
            if sub:
                B_phi[ip] = min(r["B"] for r in sub)
        ax.plot(np.degrees(phi_values), B_phi / trap_depth * 100,
                "o-", ms=4, lw=1.2, color="darkred")
        ax.axhline(0, color="green", ls="--", lw=1, alpha=0.7)
        ax.axhline(B_baseline / trap_depth * 100, color="blue", ls=":", lw=1)
        ax.set_xlabel("φ₀ (°)")
        ax.set_ylabel("B / depth %")
        ax.set_title(f"α = {alpha:.2f}", fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Bottom: B vs (x₀, y₀)
        ax = axes[1, ia]
        ns = len(shift_fracs)
        B_grid = np.full((ns, ns), np.inf)
        for iix in range(ns):
            for iiy in range(ns):
                sub = [r for r in recs_a
                       if abs(r["x0_frac"] - shift_fracs[iix]) < 0.001
                       and abs(r["y0_frac"] - shift_fracs[iiy]) < 0.001]
                if sub:
                    B_grid[iiy, iix] = min(r["B"] for r in sub)
        B_rel = B_grid / trap_depth * 100
        bmax = max(abs(np.nanmin(B_rel)), abs(np.nanmax(B_rel)), 1)
        im = ax.imshow(B_rel,
                       extent=[shift_fracs[0], shift_fracs[-1],
                               shift_fracs[0], shift_fracs[-1]],
                       origin="lower", aspect="equal",
                       cmap="RdYlGn_r", vmin=-bmax, vmax=bmax)
        ax.set_xlabel("x₀ / sp")
        ax.set_ylabel("y₀ / sp")
        plt.colorbar(im, ax=ax, shrink=0.8, label="B/depth %")

    fig.suptitle(f"Bridge Metric Heatmaps  │  baseline B = "
                 f"{B_baseline/trap_depth*100:.1f}%",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(fig_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_best_bridge_gorkov(U_s, best_U, U_line_base, best_U_line, frac,
                            sep, B_baseline, best_B, best_cfg,
                            trap1, trap2, xg, yg, ext, trap_depth, fig_path):
    """3-panel: baseline U, best U, barrier profile."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    vU = [min(U_s.min(), best_U.min()), max(U_s.max(), best_U.max())]

    ax = axes[0]
    _tight_imshow(ax, U_s, ext, cmap="RdBu_r", vmin=vU[0], vmax=vU[1])
    _trap_grid(ax, CX, CY, TRAP_SP)
    ax.plot([xg[trap1[1]]*1e3, xg[trap2[1]]*1e3],
            [yg[trap1[0]]*1e3, yg[trap2[0]]*1e3], "k-o", ms=5, lw=2, zorder=5)
    ax.set_title(f"Baseline U\nB = {B_baseline:.2e} J", fontsize=10)
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")

    ax = axes[1]
    im = _tight_imshow(ax, best_U, ext, cmap="RdBu_r", vmin=vU[0], vmax=vU[1])
    _trap_grid(ax, CX, CY, TRAP_SP)
    ax.plot([xg[trap1[1]]*1e3, xg[trap2[1]]*1e3],
            [yg[trap1[0]]*1e3, yg[trap2[0]]*1e3], "k-o", ms=5, lw=2, zorder=5)
    ax.set_title(f"Best: α={best_cfg['alpha']:.2f}  "
                 f"φ₀={np.degrees(best_cfg['phi0']):.0f}°\n"
                 f"B = {best_B:.2e} J", fontsize=9)
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    plt.colorbar(im, ax=axes[:2], shrink=0.8, label="U (J)")

    ax = axes[2]
    dist = frac * sep * 1e3
    ax.plot(dist, U_line_base, "b-", lw=2, label="Baseline")
    ax.plot(dist, best_U_line, "r-", lw=2, label="Best")
    ax.axhline(0, color="gray", ls=":", lw=0.5)
    ax.set_xlabel("Distance (mm)"); ax.set_ylabel("U (J)")
    ax.set_title("Barrier profile")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    fig.suptitle(f"Best Bridge  │  depth = {trap_depth:.2e} J",
                 fontsize=12, fontweight="bold")
    fig.savefig(fig_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_pressure_examples(p_s, best_p_total, pv_best, best_cfg, ext, fig_path):
    """2×2: |p_stand|, |α·exp(iφ)·p_v|, |p_total|, phase(p_total)."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    abs_s = np.abs(p_s)
    vm = abs_s.max()

    ax = axes[0, 0]
    im = _tight_imshow(ax, abs_s, ext, cmap="inferno", vmin=0, vmax=vm)
    _trap_grid(ax, CX, CY, TRAP_SP)
    ax.set_title("|p_standing|"); plt.colorbar(im, ax=ax, shrink=0.8, label="Pa")

    ax = axes[0, 1]
    abs_v = np.abs(best_cfg["alpha"] * np.exp(1j * best_cfg["phi0"]) * pv_best)
    im = _tight_imshow(ax, abs_v, ext, cmap="viridis", vmin=0, vmax=abs_v.max())
    _trap_grid(ax, CX, CY, TRAP_SP)
    ax.set_title(f"|α·exp(iφ₀)·p_vort|  α={best_cfg['alpha']:.2f}")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Pa")

    ax = axes[1, 0]
    im = _tight_imshow(ax, np.abs(best_p_total), ext, cmap="inferno", vmin=0, vmax=vm)
    _trap_grid(ax, CX, CY, TRAP_SP)
    ax.set_title("|p_total| best"); plt.colorbar(im, ax=ax, shrink=0.8, label="Pa")

    ax = axes[1, 1]
    ph = np.angle(best_p_total)
    im = _tight_imshow(ax, ph, ext, cmap="hsv", vmin=-np.pi, vmax=np.pi)
    _trap_grid(ax, CX, CY, TRAP_SP)
    ax.set_title("phase(p_total)"); plt.colorbar(im, ax=ax, shrink=0.8, label="rad")

    for ax in axes.flat:
        ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    fig.suptitle("Pressure Field Examples", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(fig_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_phase_interference(p_s, pv_n, alpha, phi_values, xg, yg, ext,
                            trap_depth, trap1, trap2, fig_path):
    """Show interference pattern for several phase offsets at fixed α."""
    n = min(len(phi_values), 8)
    indices = np.linspace(0, len(phi_values) - 1, n, dtype=int)
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 6.5), squeeze=False)

    dx = float(xg[1] - xg[0])
    dy = float(yg[1] - yg[0])

    for j, ip in enumerate(indices):
        phi = phi_values[ip]
        p_t = p_s + alpha * np.exp(1j * phi) * pv_n

        ax = axes[0, j]
        im = _tight_imshow(ax, np.abs(p_t), ext, cmap="inferno",
                           vmin=0, vmax=np.abs(p_s).max())
        _trap_grid(ax, CX, CY, TRAP_SP)
        ax.set_title(f"φ₀={np.degrees(phi):.0f}°", fontsize=9)

        ax = axes[1, j]
        # Quick Gorkov (no z-gradient — just potential term)
        U_approx = gorkov_potential(np.abs(p_t)**2,
                                    np.zeros_like(np.abs(p_t)**2))
        im = _tight_imshow(ax, U_approx, ext, cmap="RdBu_r")
        _trap_grid(ax, CX, CY, TRAP_SP)

    for ax in axes.flat:
        ax.set_xlabel("x (mm)", fontsize=7)
        ax.tick_params(labelsize=6)

    fig.suptitle(f"Phase Interference  α={alpha:.2f}  │  "
                 f"top: |p_total|  bottom: U_approx",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(fig_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_offset_geometry(p_s, pv_fields, alpha, phi_best, shift_fracs,
                         xg, yg, ext, fig_path):
    """3×3 grid of offsets showing |p_total| for the best phase."""
    # Select subset for display: corners + centre + edge midpoints
    ns = len(shift_fracs)
    idx_show = [0, ns // 2, ns - 1]  # 3 values
    fig, axes = plt.subplots(3, 3, figsize=(11, 11), squeeze=False)
    vm = np.abs(p_s).max()

    for ir, iiy in enumerate(idx_show):
        for ic, iix in enumerate(idx_show):
            ax = axes[ir, ic]
            pv, _, _ = pv_fields[(iix, iiy)]
            p_t = p_s + alpha * np.exp(1j * phi_best) * pv
            im = _tight_imshow(ax, np.abs(p_t), ext, cmap="inferno",
                               vmin=0, vmax=vm)
            _trap_grid(ax, CX, CY, TRAP_SP)
            ax.set_title(f"x₀={shift_fracs[iix]:+.1f}  "
                         f"y₀={shift_fracs[iiy]:+.1f}", fontsize=9)
            ax.set_xlabel("x (mm)", fontsize=7)
            ax.set_ylabel("y (mm)", fontsize=7)
            ax.tick_params(labelsize=6)

    fig.suptitle(f"Offset Geometry  α={alpha:.2f}  φ₀={np.degrees(phi_best):.0f}°",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(fig_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_bridge_profiles(U_s, results_sorted, xg, yg, trap1, trap2, sep,
                         trap_depth, B_baseline, fig_path, n_show=5):
    """Overlay bridge profiles for the top-N configurations."""
    fig, ax = plt.subplots(figsize=(9, 5))
    frac = np.linspace(0, 1, BRIDGE_NPTS)
    dist = frac * sep * 1e3

    _, U_base, _, _, _, _ = compute_bridge_metric(U_s, xg, yg, trap1, trap2)
    ax.plot(dist, U_base, "k-", lw=2.5, label="Baseline", zorder=10)

    colors = plt.cm.plasma(np.linspace(0.1, 0.9, n_show))
    for i, r in enumerate(results_sorted[:n_show]):
        ax.plot(dist, r["_U_line"], "-", lw=1.5, color=colors[i],
                label=f"#{i+1} α={r['alpha']:.2f} φ={np.degrees(r['phi0']):.0f}°")

    ax.axhline(0, color="gray", ls=":", lw=0.5)
    ax.set_xlabel("Distance along bridge (mm)")
    ax.set_ylabel("U (J)")
    ax.set_title(f"Bridge Profiles — Top {n_show} Configs\n"
                 f"Baseline B = {B_baseline:.2e} J "
                 f"({B_baseline/trap_depth*100:.1f}% depth)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=str, default=None)
    ap.add_argument("--timestamp", type=str,
                    default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    ap.add_argument("--quick", action="store_true",
                    help="Reduced grid for testing")
    ap.add_argument("--extended", action="store_true",
                    help="Also sweep ℓ and apodization")
    args = ap.parse_args()

    t_GLOBAL = time.time()
    TS = args.timestamp

    cache_path = (Path(args.cache).resolve() if args.cache
                  else find_latest_cache(STANDING_CACHE_DIR))

    OUT = PROJECT_ROOT / "results" / f"bridge_master_study_{TS}"
    FIG = OUT / "figures"; FIG.mkdir(parents=True, exist_ok=True)
    DAT = OUT / "data";    DAT.mkdir(parents=True, exist_ok=True)

    log_fp = open(OUT / "console_log.txt", "w")
    log = Log(log_fp)

    # ── Parameter grids ───────────────────────────────────────────
    if args.quick:
        ALPHAS = [0.10, 0.20, 0.35, 0.50]
        N_PHASE = 8
        SHIFT_FRACS = np.array([-0.4, -0.2, 0.0, 0.2, 0.4])
    else:
        ALPHAS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
        N_PHASE = 16
        SHIFT_FRACS = np.array([-0.5, -0.4, -0.3, -0.2, -0.1,
                                 0.0, 0.1, 0.2, 0.3, 0.4, 0.5])

    PHI_VALUES = np.linspace(0, 2 * np.pi, N_PHASE, endpoint=False)
    X0_OFFSETS = SHIFT_FRACS * TRAP_SP
    Y0_OFFSETS = SHIFT_FRACS * TRAP_SP

    # Lens configurations
    if args.extended:
        LENS_CONFIGS = [
            {"ell": 1, "focal": 4.2e-3, "R": 7.0e-3, "apod": "uniform",
             "label": "ℓ1_f4.2_uni"},
            {"ell": 2, "focal": 4.2e-3, "R": 7.0e-3, "apod": "uniform",
             "label": "ℓ2_f4.2_uni"},
            {"ell": 1, "focal": 4.2e-3, "R": 7.0e-3, "apod": "cosine_taper",
             "label": "ℓ1_f4.2_cos"},
            {"ell": 1, "focal": 4.0e-3, "R": 7.0e-3, "apod": "uniform",
             "label": "ℓ1_f4.0_uni"},
            {"ell": 1, "focal": 4.4e-3, "R": 7.0e-3, "apod": "uniform",
             "label": "ℓ1_f4.4_uni"},
        ]
    else:
        LENS_CONFIGS = [
            {"ell": 1, "focal": 4.2e-3, "R": 7.0e-3, "apod": "uniform",
             "label": "ℓ1_f4.2_uni"},
        ]

    n_per_lens = len(ALPHAS) * N_PHASE * len(X0_OFFSETS) * len(Y0_OFFSETS)
    n_total = n_per_lens * len(LENS_CONFIGS)

    log("=" * 72)
    log("Bridge Master Study — Large Parameter Space")
    log("=" * 72)
    log(f"Date  : {datetime.now().isoformat()}")
    log(f"Cache : {cache_path}")
    log(f"Output: {OUT}")
    log(f"Mode  : {'quick' if args.quick else 'full'}"
        f"{'  +extended' if args.extended else ''}")
    log(f"λ = {LAM*1e3:.4f} mm   trap λ/2 = {TRAP_SP*1e3:.4f} mm")
    log(f"z* = {Z_STAR*1e3:.4f} mm")
    log(f"Sweep: {len(ALPHAS)} α × {N_PHASE} φ₀ × "
        f"{len(SHIFT_FRACS)}² offsets × {len(LENS_CONFIGS)} lens(es)")
    log(f"Total configurations: {n_total}")
    log()

    # ==============================================================
    # STEP 1 — Load FEM + baseline Gor'kov
    # ==============================================================
    log("=" * 72)
    log("STEP 1: Load FEM standing wave + baseline Gor'kov")
    log("=" * 72)

    t0 = time.time()
    coords, p_fem = load_fem_cache(cache_path)
    tree = cKDTree(coords)
    log(f"  FEM: {coords.shape[0]:,} DOFs ({time.time()-t0:.2f}s)")

    xg = np.linspace(CX - ROI_HALF, CX + ROI_HALF, NGRID)
    yg = np.linspace(CY - ROI_HALF, CY + ROI_HALF, NGRID)
    dx = float(xg[1] - xg[0])
    dy = float(yg[1] - yg[0])
    XX, YY = np.meshgrid(xg, yg)
    ext_roi = [xg[0] * 1e3, xg[-1] * 1e3, yg[0] * 1e3, yg[-1] * 1e3]

    log(f"  ROI: {2*ROI_HALF*1e3:.3f}mm  grid {NGRID}×{NGRID}  "
        f"dx={dx*1e6:.1f}µm")

    def _fem_xy(z_val):
        pts = np.column_stack([XX.ravel(), YY.ravel(),
                               np.full(XX.size, z_val)])
        return sample_idw(tree, p_fem, pts, k=K_IDW).reshape(XX.shape)

    t0 = time.time()
    p_s    = _fem_xy(Z_STAR)
    p_s_zp = _fem_xy(Z_STAR + DZ_GRAD)
    p_s_zm = _fem_xy(Z_STAR - DZ_GRAD)
    log(f"  FEM sampling: {time.time()-t0:.2f}s  max|p_s|={np.abs(p_s).max():.3f}")

    gp2_s = grad_p_mag2_xy(p_s, p_s_zp, p_s_zm, dx, dy, DZ_GRAD)
    U_s = gorkov_potential(np.abs(p_s)**2, gp2_s)
    trap_depth = float(U_s.max() - U_s.min())
    stand_peak = float(np.abs(p_s).max())

    pair = find_adjacent_pair(U_s, xg, yg)
    if pair is None:
        log("  FATAL: No adjacent trap pair found")
        log_fp.close()
        return
    trap1, trap2 = pair
    sep = np.sqrt((xg[trap1[1]] - xg[trap2[1]])**2 +
                  (yg[trap1[0]] - yg[trap2[0]])**2)
    B_baseline, U_line_base, frac_base, _, _, _ = compute_bridge_metric(
        U_s, xg, yg, trap1, trap2)

    log(f"  Trap depth = {trap_depth:.3e} J")
    log(f"  Trap pair sep = {sep*1e3:.3f} mm ({sep/TRAP_SP:.2f}×λ/2)")
    log(f"  Baseline B = {B_baseline:.4e} J ({B_baseline/trap_depth*100:.2f}%)")
    log()

    # ==============================================================
    # STEP 2 — Sweep over all lens configs
    # ==============================================================
    all_results = []
    global_best_B = np.inf
    global_best = None
    global_best_U = None
    global_best_U_line = None
    global_best_p_total = None
    global_best_pv = None
    # Store vort fields for best lens for plots
    best_lens_vort_fields = None
    best_lens_label = None

    for ilens, lcfg in enumerate(LENS_CONFIGS):
        log("=" * 72)
        log(f"LENS {ilens+1}/{len(LENS_CONFIGS)}: {lcfg['label']}")
        log(f"  ℓ={lcfg['ell']}  f={lcfg['focal']*1e3:.1f}mm  "
            f"R={lcfg['R']*1e3:.1f}mm  apod={lcfg['apod']}")
        log("=" * 72)

        # Build ASM drive for this lens
        asm_domain = max(LX, 2 * lcfg["R"] + 4 * LAM)
        xg_d, yg_d, D_drive = build_converging_drive(
            lcfg["ell"], lcfg["focal"], lcfg["R"], CX, CY,
            asm_domain, ASM_N, K_WATER, apod=lcfg["apod"])
        dx_d = float(xg_d[1] - xg_d[0])

        # Propagate ASM at z* (3 planes)
        t0 = time.time()
        z_prop = Z_STAR - Z0_OFFSET
        p_asm = propagate_pressure_asm(D_drive, dx_d, dx_d, K_WATER,
                                        z_prop, pad_factor=ASM_PAD,
                                        include_evanescent=True)
        p_asm_zp = propagate_pressure_asm(D_drive, dx_d, dx_d, K_WATER,
                                           z_prop + DZ_GRAD,
                                           pad_factor=ASM_PAD,
                                           include_evanescent=True)
        p_asm_zm = propagate_pressure_asm(D_drive, dx_d, dx_d, K_WATER,
                                           z_prop - DZ_GRAD,
                                           pad_factor=ASM_PAD,
                                           include_evanescent=True)
        log(f"  ASM propagation: {time.time()-t0:.2f}s")

        # Normalise
        p_v_ctr = interp_asm_to_grid(p_asm, xg_d, yg_d, XX, YY)
        vort_peak = float(np.abs(p_v_ctr).max()) + 1e-30
        nfac = stand_peak / vort_peak
        log(f"  nfac = {nfac:.3f}")

        # Pre-compute shifted vortex fields
        t0 = time.time()
        vort_fields = {}
        for iix, x0 in enumerate(X0_OFFSETS):
            for iiy, y0 in enumerate(Y0_OFFSETS):
                XX_sh, YY_sh = XX - x0, YY - y0
                pv = interp_asm_to_grid(p_asm, xg_d, yg_d, XX_sh, YY_sh) * nfac
                pv_zp = interp_asm_to_grid(p_asm_zp, xg_d, yg_d, XX_sh, YY_sh) * nfac
                pv_zm = interp_asm_to_grid(p_asm_zm, xg_d, yg_d, XX_sh, YY_sh) * nfac
                vort_fields[(iix, iiy)] = (pv, pv_zp, pv_zm)
        log(f"  Offset sampling: {time.time()-t0:.2f}s "
            f"({len(X0_OFFSETS)*len(Y0_OFFSETS)} offsets)")

        # Sweep
        t0_sweep = time.time()
        count = 0
        lens_best_B = np.inf

        for alpha in ALPHAS:
            for phi0 in PHI_VALUES:
                pf = np.exp(1j * phi0)
                for iix in range(len(X0_OFFSETS)):
                    for iiy in range(len(Y0_OFFSETS)):
                        pv, pv_zp, pv_zm = vort_fields[(iix, iiy)]
                        p_t    = p_s    + alpha * pf * pv
                        p_t_zp = p_s_zp + alpha * pf * pv_zp
                        p_t_zm = p_s_zm + alpha * pf * pv_zm

                        gp2 = grad_p_mag2_xy(p_t, p_t_zp, p_t_zm,
                                             dx, dy, DZ_GRAD)
                        U_c = gorkov_potential(np.abs(p_t)**2, gp2)
                        B, U_line, frac, Um1, Um2, Usad = \
                            compute_bridge_metric(U_c, xg, yg, trap1, trap2)

                        rec = {
                            "lens": lcfg["label"],
                            "ell": lcfg["ell"],
                            "focal_mm": lcfg["focal"] * 1e3,
                            "apod": lcfg["apod"],
                            "alpha": alpha,
                            "phi0": float(phi0),
                            "x0_frac": float(SHIFT_FRACS[iix]),
                            "y0_frac": float(SHIFT_FRACS[iiy]),
                            "B": float(B),
                            "B_rel": float(B / (trap_depth + 1e-30)),
                        }
                        all_results.append(rec)

                        if B < global_best_B:
                            global_best_B = B
                            global_best = rec.copy()
                            global_best["U_min1"] = float(Um1)
                            global_best["U_min2"] = float(Um2)
                            global_best["U_saddle"] = float(Usad)
                            global_best_U = U_c.copy()
                            global_best_U_line = U_line.copy()
                            global_best_p_total = p_t.copy()
                            global_best_pv = pv.copy()
                            best_lens_vort_fields = vort_fields
                            best_lens_label = lcfg["label"]

                        if B < lens_best_B:
                            lens_best_B = B

                        count += 1
                        if count % 500 == 0:
                            el = time.time() - t0_sweep
                            log(f"    [{count:6d}/{n_per_lens}]  "
                                f"lens_best={lens_best_B:.3e}  "
                                f"global_best={global_best_B:.3e}  "
                                f"[{el:.1f}s]")

        dt = time.time() - t0_sweep
        log(f"  Lens sweep: {dt:.1f}s  best B = {lens_best_B:.4e} J "
            f"({lens_best_B/trap_depth*100:+.2f}%)")
        log()

    total_time_sweep = time.time() - t_GLOBAL
    log(f"All sweeps complete: {total_time_sweep:.1f}s  "
        f"total configs = {len(all_results)}")
    log(f"★ GLOBAL BEST B = {global_best_B:.4e} J "
        f"({global_best_B/trap_depth*100:+.2f}%)")
    if global_best:
        log(f"  lens={global_best['lens']}  α={global_best['alpha']:.2f}  "
            f"φ₀={np.degrees(global_best['phi0']):.0f}°  "
            f"x₀={global_best['x0_frac']:+.1f}  y₀={global_best['y0_frac']:+.1f}")
    reduction = (1 - global_best_B / B_baseline) * 100 if B_baseline > 0 else 0
    bridge_formed = global_best_B <= 0
    log(f"  Reduction: {reduction:.1f}%   Bridge formed: {bridge_formed}")
    log()

    # ==============================================================
    # FIGURES
    # ==============================================================
    log("=" * 72)
    log("FIGURES")
    log("=" * 72)

    # 1. bridge_metric_heatmaps.png (use primary lens results)
    primary_results = [r for r in all_results
                       if r["lens"] == best_lens_label]
    plot_bridge_metric_heatmaps(
        primary_results, ALPHAS, PHI_VALUES, SHIFT_FRACS,
        trap_depth, B_baseline, ext_roi,
        FIG / "bridge_metric_heatmaps.png")
    log("  Saved bridge_metric_heatmaps.png")

    # 2. best_bridge_gorkov.png
    plot_best_bridge_gorkov(
        U_s, global_best_U, U_line_base, global_best_U_line, frac_base,
        sep, B_baseline, global_best_B, global_best,
        trap1, trap2, xg, yg, ext_roi, trap_depth,
        FIG / "best_bridge_gorkov.png")
    log("  Saved best_bridge_gorkov.png")

    # 3. pressure_field_examples.png
    plot_pressure_examples(
        p_s, global_best_p_total, global_best_pv, global_best, ext_roi,
        FIG / "pressure_field_examples.png")
    log("  Saved pressure_field_examples.png")

    # 4. phase_interference_examples.png
    # Use centre-offset vortex for phase sweep display
    pv_ctr, _, _ = best_lens_vort_fields[(len(SHIFT_FRACS) // 2,
                                           len(SHIFT_FRACS) // 2)]
    plot_phase_interference(
        p_s, pv_ctr, global_best["alpha"], PHI_VALUES, xg, yg, ext_roi,
        trap_depth, trap1, trap2,
        FIG / "phase_interference_examples.png")
    log("  Saved phase_interference_examples.png")

    # 5. offset_geometry_examples.png
    plot_offset_geometry(
        p_s, best_lens_vort_fields, global_best["alpha"],
        global_best["phi0"], SHIFT_FRACS, xg, yg, ext_roi,
        FIG / "offset_geometry_examples.png")
    log("  Saved offset_geometry_examples.png")

    # 6. bridge_profile_lines.png — top 5 configs + baseline
    # Need to recompute U_line for top configs that weren't stored
    all_results.sort(key=lambda r: r["B"])
    top10 = all_results[:10]

    # Recompute bridge profiles for top configs (from best lens data)
    top_with_lines = []
    for rec in top10:
        # Find matching vort field
        iix = int(np.argmin(np.abs(SHIFT_FRACS - rec["x0_frac"])))
        iiy = int(np.argmin(np.abs(SHIFT_FRACS - rec["y0_frac"])))
        if best_lens_vort_fields and rec["lens"] == best_lens_label:
            pv, pv_zp, pv_zm = best_lens_vort_fields[(iix, iiy)]
            pf = np.exp(1j * rec["phi0"])
            p_t = p_s + rec["alpha"] * pf * pv
            p_t_zp = p_s_zp + rec["alpha"] * pf * pv_zp
            p_t_zm = p_s_zm + rec["alpha"] * pf * pv_zm
            gp2 = grad_p_mag2_xy(p_t, p_t_zp, p_t_zm, dx, dy, DZ_GRAD)
            U_c = gorkov_potential(np.abs(p_t)**2, gp2)
            _, U_line, _, _, _, _ = compute_bridge_metric(
                U_c, xg, yg, trap1, trap2)
            r2 = rec.copy()
            r2["_U_line"] = U_line
            top_with_lines.append(r2)

    if top_with_lines:
        plot_bridge_profiles(
            U_s, top_with_lines, xg, yg, trap1, trap2, sep,
            trap_depth, B_baseline,
            FIG / "bridge_profile_lines.png",
            n_show=min(5, len(top_with_lines)))
        log("  Saved bridge_profile_lines.png")
    log()

    # ==============================================================
    # STATS + CSV + REPORT
    # ==============================================================
    log("=" * 72)
    log("STEP 3: Stats, CSV, REPORT")
    log("=" * 72)

    stats = {
        "lambda_mm": float(LAM * 1e3),
        "trap_spacing_mm": float(TRAP_SP * 1e3),
        "z_star_mm": round(float(Z_STAR * 1e3), 4),
        "roi_half_mm": round(float(ROI_HALF * 1e3), 4),
        "grid_size": NGRID,
        "mode": "quick" if args.quick else ("extended" if args.extended else "full"),
        "sweep": {
            "n_alpha": len(ALPHAS), "alphas": ALPHAS,
            "n_phase": N_PHASE,
            "n_offsets": len(SHIFT_FRACS),
            "shift_fracs": SHIFT_FRACS.tolist(),
            "n_lens": len(LENS_CONFIGS),
            "lens_labels": [lc["label"] for lc in LENS_CONFIGS],
            "total_configs": len(all_results),
        },
        "baseline": {
            "trap_depth_J": float(trap_depth),
            "B_J": float(B_baseline),
            "B_rel": float(B_baseline / trap_depth),
            "trap1_mm": [round(xg[trap1[1]]*1e3, 3),
                         round(yg[trap1[0]]*1e3, 3)],
            "trap2_mm": [round(xg[trap2[1]]*1e3, 3),
                         round(yg[trap2[0]]*1e3, 3)],
            "separation_mm": round(float(sep * 1e3), 4),
        },
        "best": global_best,
        "bridge_formed": bool(bridge_formed),
        "barrier_reduction_pct": round(float(reduction), 2),
        "top10": [{k: v for k, v in r.items() if not k.startswith("_")}
                  for r in top10],
    }
    with open(DAT / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    log("  Saved stats.json")

    # CSV
    csv_path = DAT / "parameter_table.csv"
    fields_csv = ["lens", "ell", "focal_mm", "apod",
                  "alpha", "phi0", "x0_frac", "y0_frac", "B", "B_rel"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields_csv, extrasaction="ignore")
        w.writeheader()
        for r in all_results:
            w.writerow(r)
    log(f"  Saved parameter_table.csv ({len(all_results)} rows)")

    # Fields
    np.savez_compressed(DAT / "fields.npz",
                        xg=xg, yg=yg, U_s=U_s,
                        best_U=global_best_U,
                        U_line_base=U_line_base,
                        best_U_line=global_best_U_line,
                        frac=frac_base)
    log("  Saved fields.npz")

    # ── REPORT.md ─────────────────────────────────────────────────
    elapsed = time.time() - t_GLOBAL
    log(f"\nTotal runtime: {elapsed:.1f}s")

    rp = []
    rp.append("# Bridge Master Study — Large Parameter Space Exploration\n")
    rp.append(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    rp.append(f"**Runtime**: {elapsed:.1f}s")
    rp.append(f"**Mode**: {'quick' if args.quick else 'full'}"
              f"{'  +extended' if args.extended else ''}")
    rp.append(f"**Cache**: `{cache_path.name}` ({coords.shape[0]:,} DOFs)\n")

    rp.append("## Configuration\n")
    rp.append(f"- **ROI**: ±{ROI_HALF*1e3:.3f} mm ({2*ROI_HALF/LAM:.1f}λ)")
    rp.append(f"- **z*** = {Z_STAR*1e3:.4f} mm")
    rp.append(f"- **λ** = {LAM*1e3:.4f} mm, trap spacing = {TRAP_SP*1e3:.4f} mm")
    rp.append(f"- **Field combination**: "
              "`p_total = p_stand + α·exp(iφ₀)·p̂_vort(x−x₀, y−y₀)`\n")

    rp.append("## Sweep Parameters\n")
    rp.append(f"- **α** ∈ {{{', '.join(f'{a:.2f}' for a in ALPHAS)}}}")
    rp.append(f"- **φ₀** ∈ {N_PHASE} values in [0, 2π)")
    rp.append(f"- **Offsets**: {len(SHIFT_FRACS)}×{len(SHIFT_FRACS)} grid, "
              f"fracs = {SHIFT_FRACS.tolist()}")
    rp.append(f"- **Lens configs**: {len(LENS_CONFIGS)}")
    for lc in LENS_CONFIGS:
        rp.append(f"  - {lc['label']}: ℓ={lc['ell']}, f={lc['focal']*1e3:.1f}mm, "
                  f"R={lc['R']*1e3:.1f}mm, {lc['apod']}")
    rp.append(f"- **Total configurations**: {len(all_results)}\n")

    rp.append("## Bridge Metric\n")
    rp.append(f"- Trap 1: ({xg[trap1[1]]*1e3:.3f}, {yg[trap1[0]]*1e3:.3f}) mm")
    rp.append(f"- Trap 2: ({xg[trap2[1]]*1e3:.3f}, {yg[trap2[0]]*1e3:.3f}) mm")
    rp.append(f"- Separation: {sep*1e3:.3f} mm ({sep/TRAP_SP:.2f} × λ/2)")
    rp.append(f"- B = U_saddle − max(U_min1, U_min2)")
    rp.append(f"- B ≤ 0 → bridge formed\n")

    rp.append("## Results\n")
    rp.append(f"**Baseline** B = {B_baseline:.3e} J "
              f"({B_baseline/trap_depth*100:.2f}% of depth)\n")
    if global_best:
        rp.append(f"**Best** B = {global_best_B:.3e} J "
                  f"({global_best_B/trap_depth*100:+.2f}% of depth)")
        rp.append(f"- Lens: {global_best['lens']}")
        rp.append(f"- α = {global_best['alpha']:.2f}")
        rp.append(f"- φ₀ = {np.degrees(global_best['phi0']):.1f}°")
        rp.append(f"- x₀ = {global_best['x0_frac']:+.1f} × sp")
        rp.append(f"- y₀ = {global_best['y0_frac']:+.1f} × sp\n")
    rp.append(f"**Barrier reduction**: {reduction:.1f}%")
    rp.append(f"**Bridge formed** (B ≤ 0): "
              f"{'**YES** ✓' if bridge_formed else 'NO'}\n")

    rp.append("### Top 10 Configurations\n")
    rp.append("| # | Lens | α | φ₀ (°) | x₀/sp | y₀/sp | B (J) | B/depth % |")
    rp.append("|---|------|---|--------|-------|-------|-------|-----------|")
    for i, r in enumerate(top10):
        rp.append(f"| {i+1} | {r.get('lens','-')} | {r['alpha']:.2f} | "
                  f"{np.degrees(r['phi0']):.0f} | {r['x0_frac']:+.1f} | "
                  f"{r['y0_frac']:+.1f} | {r['B']:.2e} | "
                  f"{r['B_rel']*100:+.1f} |")

    # Parameter sensitivity analysis
    rp.append("\n## Parameter Sensitivity\n")

    # α effect
    rp.append("### Effect of α (amplitude scaling)\n")
    rp.append("| α | Best B (J) | B/depth % |")
    rp.append("|---|-----------|-----------|")
    for alpha in ALPHAS:
        sub = [r for r in all_results if r["alpha"] == alpha]
        if sub:
            b_min = min(r["B"] for r in sub)
            rp.append(f"| {alpha:.2f} | {b_min:.2e} | "
                      f"{b_min/trap_depth*100:+.1f} |")

    # Phase effect
    rp.append("\n### Role of phase offset φ₀\n")
    rp.append("The relative phase between standing and vortex fields "
              "controls constructive/destructive interference. ")
    phi_best_idx = int(np.argmin([
        min((r["B"] for r in all_results
             if abs(r["phi0"] - phi) < 0.01), default=np.inf)
        for phi in PHI_VALUES]))
    rp.append(f"Best phase: φ₀ = {np.degrees(PHI_VALUES[phi_best_idx]):.0f}°\n")

    # Position effect
    rp.append("### Role of vortex position (x₀, y₀)\n")
    rp.append("Lateral offsets position the vortex ring between traps. "
              "Non-zero offsets can align the vortex intensity ring with "
              "the saddle point between adjacent traps.\n")

    rp.append("## Figures\n")
    rp.append("![bridge_metric_heatmaps](figures/bridge_metric_heatmaps.png)\n")
    rp.append("![best_bridge_gorkov](figures/best_bridge_gorkov.png)\n")
    rp.append("![pressure_field_examples](figures/pressure_field_examples.png)\n")
    rp.append("![phase_interference_examples]"
              "(figures/phase_interference_examples.png)\n")
    rp.append("![offset_geometry_examples]"
              "(figures/offset_geometry_examples.png)\n")
    rp.append("![bridge_profile_lines](figures/bridge_profile_lines.png)\n")

    rp.append("## Discussion\n")
    if bridge_formed:
        rp.append("**Bridge formation was achieved.** The Gor'kov barrier between "
                  "adjacent traps was reduced to zero or below, creating a "
                  "continuous potential channel. This demonstrates that "
                  "linear superposition of standing and vortex fields with "
                  "appropriate phase and position control can enable "
                  "particle transport between traps.\n")
    elif reduction > 50:
        rp.append(f"Substantial barrier reduction ({reduction:.0f}%) was achieved. "
                  "The phase offset and lateral positioning significantly modify "
                  "the interference pattern. The barrier was not fully eliminated "
                  "but is substantially weakened at higher α values. "
                  "Physical feasibility depends on the achievable vortex "
                  "amplitude relative to the standing wave.\n")
    else:
        rp.append(f"The barrier was reduced by {reduction:.0f}%. "
                  "The vortex perturbation modifies trap depths but does not "
                  "bridge adjacent traps at the tested α values. "
                  f"The vortex waist diameter (~1.67 × λ/2) spreads the "
                  "perturbation across multiple traps rather than concentrating "
                  "it between a single pair. Smaller waist, higher ℓ, or "
                  "stronger vortex amplitudes may be needed.\n")

    with open(OUT / "REPORT.md", "w") as f:
        f.write("\n".join(rp) + "\n")
    log("  Saved REPORT.md")

    log("\n" + "=" * 72)
    log("DONE")
    log("=" * 72)
    log_fp.close()


if __name__ == "__main__":
    main()
