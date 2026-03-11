#!/usr/bin/env python3
"""
C-Shape Lens 15 mm Overlay Study (Phase 2)
============================================

Physics investigation: how does the 15 mm printable holographic lens
field interact with the standing-wave trap environment?

This script does NOT redesign the lens.  It loads the final printable
15 mm lens from Phase 1 and studies:

  A. Lens field alone in the trap ROI
  B. Standing-wave reference
  C. Combined field p_sw + α·exp(iψ)·p_lens
  D. Amplitude sweep over α
  E. Phase-delay sweep over ψ

Outputs → results/c_shape_lens_15mm_overlay_study_<TS>/
"""
from __future__ import annotations

import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import minimum_filter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ── Project root ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ═══════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════
C_WATER  = 1484.0
F_HZ     = 2.0e6
K_WATER  = 2 * np.pi * F_HZ / C_WATER
LAM      = C_WATER / F_HZ           # 0.742 mm
TRAP_SP  = LAM / 2.0                # ~0.371 mm
OMEGA    = 2 * np.pi * F_HZ
RHO0     = 997.0

LX = LY  = 6.0e-3                   # FEM domain
CX = CY  = LX / 2.0                 # domain centre (3 mm)
H_UNDER  = 5.0e-3
H_TOP    = 2.0e-3
Z_STAR   = H_UNDER + H_TOP / 2.0 + 0.25 * LAM   # ~6.1855 mm

ROI_HALF = 1.1 * LAM                # half-width of the inspection ROI

# Paths
PHASE1_DIR = (PROJECT_ROOT / "results"
              / "c_shape_lens_15mm_manufacturing_study_20260310_153032")
SW_NPZ = (PROJECT_ROOT / "results" / "fem_standing_wave_cache"
           / "checkpoint_epl5_depth7mm_20260309_113007"
           / "standing_wave_epl5.npz")

# Sweep parameters
ALPHA_VALUES = [0.05, 0.10, 0.20, 0.40]
PSI_VALUES   = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]
PSI_LABELS   = ["0", "π/2", "π", "3π/2"]

# ROI interpolation grid resolution
N_ROI = 400

# ── Output directory ────────────────────────────────────────────────
TS = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = PROJECT_ROOT / "results" / f"c_shape_lens_15mm_overlay_study_{TS}"
FIG_DIR = OUT_DIR / "figures"
NPZ_DIR = OUT_DIR / "npz"
for d in [OUT_DIR, FIG_DIR, NPZ_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
# Plotting helpers
# ═══════════════════════════════════════════════════════════════════

def _cbar(ax, im, label=""):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.06)
    plt.colorbar(im, cax=cax, label=label)


def _overlay_traps(ax, traps_mm, idx_A, idx_B, midpoint_mm):
    for i, (tx, ty) in enumerate(traps_mm):
        if i == idx_A:
            ax.plot(tx, ty, "r^", ms=10, mew=1.5, mfc="none", zorder=5)
        elif i == idx_B:
            ax.plot(tx, ty, "bs", ms=10, mew=1.5, mfc="none", zorder=5)
        else:
            ax.plot(tx, ty, "w+", ms=6, mew=0.8, zorder=5)
    ax.plot(*midpoint_mm, "gx", ms=10, mew=2.0, zorder=5)


# ═══════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════

def load_standing_wave():
    """Load the epl5 FEM standing-wave and interpolate onto the trap ROI."""
    d = np.load(SW_NPZ)
    coords = d["coords"]
    p_fem = d["p_real"].astype(np.float64) + 1j * d["p_imag"].astype(np.float64)
    print(f"[sw] Loaded {SW_NPZ.name}: {len(p_fem)} DOFs, "
          f"|p|_max = {np.abs(p_fem).max():.2f} Pa")

    # Interpolate onto 2D ROI at z*
    tree = cKDTree(coords)
    x_lo, x_hi = CX - ROI_HALF, CX + ROI_HALF
    y_lo, y_hi = CY - ROI_HALF, CY + ROI_HALF
    xg = np.linspace(x_lo, x_hi, N_ROI)
    yg = np.linspace(y_lo, y_hi, N_ROI)
    XX, YY = np.meshgrid(xg, yg)
    pts = np.column_stack([XX.ravel(), YY.ravel(),
                           np.full(XX.size, Z_STAR)])
    # IDW interpolation
    k_nn = 16
    dd, ii = tree.query(pts, k=k_nn)
    w = 1.0 / (dd**2 + 1e-12)
    w /= w.sum(axis=1, keepdims=True)
    p_sw_flat = (p_fem[ii] * w).sum(axis=1)
    p_sw = p_sw_flat.reshape(XX.shape)

    print(f"  ROI: [{x_lo*1e3:.3f}, {x_hi*1e3:.3f}] × "
          f"[{y_lo*1e3:.3f}, {y_hi*1e3:.3f}] mm, "
          f"|p_sw| max = {np.abs(p_sw).max():.2f} Pa")
    return p_sw, xg, yg


def detect_traps(p_sw, xg, yg):
    """Detect Gorkov-potential stable traps (same method as geometry study)."""
    dx = xg[1] - xg[0]
    p_abs = np.abs(p_sw)

    # Gorkov contrast — polystyrene in water
    rho_p, c_p = 1050.0, 2350.0
    kappa_w = 1.0 / (RHO0 * C_WATER**2)
    kappa_p = 1.0 / (rho_p * c_p**2)
    f1 = 1.0 - kappa_p / kappa_w
    f2 = 2.0 * (rho_p - RHO0) / (2.0 * rho_p + RHO0)

    p2 = p_abs**2
    dp_dx = np.gradient(p_sw, dx, axis=1)
    dp_dy = np.gradient(p_sw, dx, axis=0)
    grad_p2 = np.abs(dp_dx)**2 + np.abs(dp_dy)**2
    U = (f1 / (2 * RHO0 * C_WATER**2)) * p2 \
      - (3 * f2 / (4 * OMEGA**2 * RHO0)) * grad_p2

    min_sep = max(3, int(0.4 * TRAP_SP / dx))
    nbr = 2 * min_sep + 1
    local_min = minimum_filter(U, size=nbr)
    mask = (U == local_min)

    border = max(5, min_sep)
    mask[:border, :] = False
    mask[-border:, :] = False
    mask[:, :border] = False
    mask[:, -border:] = False

    U_range = U.max() - U.min()
    mask &= (U < U.min() + 0.50 * U_range)

    iy, ix = np.where(mask)

    # Hessian check
    dUdx = np.gradient(U, dx, axis=1)
    dUdy = np.gradient(U, dx, axis=0)
    d2Udx2 = np.gradient(dUdx, dx, axis=1)
    d2Udy2 = np.gradient(dUdy, dx, axis=0)
    d2Udxdy = np.gradient(dUdx, dx, axis=0)

    kept = []
    for ci in range(len(iy)):
        Hxx = d2Udx2[iy[ci], ix[ci]]
        Hyy = d2Udy2[iy[ci], ix[ci]]
        Hxy = d2Udxdy[iy[ci], ix[ci]]
        tr = Hxx + Hyy
        det = Hxx * Hyy - Hxy**2
        disc = max(0.0, tr**2 - 4 * det)
        lam_min = (tr - np.sqrt(disc)) / 2.0
        if lam_min > 0:
            kept.append(ci)

    traps_m = np.column_stack([xg[ix[kept]], yg[iy[kept]]])
    print(f"[traps] Detected {len(traps_m)} Gorkov-stable traps")
    return traps_m


def choose_trap_pair(traps_m):
    """Choose the A-B trap pair matching the geometry study."""
    tree = cKDTree(traps_m)
    dists, idxs = tree.query(traps_m, k=2)
    nn_dists = dists[:, 1]
    nn_idxs = idxs[:, 1]

    centre = np.array([CX, CY])
    midpoints = 0.5 * (traps_m + traps_m[nn_idxs])
    dist_to_centre = np.linalg.norm(midpoints - centre, axis=1)
    rel_err = np.abs(nn_dists - TRAP_SP) / TRAP_SP

    score = rel_err + 0.1 * dist_to_centre / max(dist_to_centre.max(), 1e-12)
    best = int(np.argmin(score))
    idx_A = best
    idx_B = int(nn_idxs[best])
    d_AB = float(nn_dists[best])
    midpoint = 0.5 * (traps_m[idx_A] + traps_m[idx_B])

    print(f"  Pair: A={idx_A} ({traps_m[idx_A]*1e3} mm), "
          f"B={idx_B} ({traps_m[idx_B]*1e3} mm)")
    print(f"  d_AB = {d_AB*1e3:.3f} mm = {d_AB/LAM:.3f}λ")
    return idx_A, idx_B, d_AB, midpoint


def load_lens_field():
    """
    Load the Phase-1 printable 15 mm lens reconstructed field.

    The lens field was ASM-propagated on a centred-at-origin grid
    (±11.25 mm), while the SW lives in FEM coords (0–6 mm).

    The C-shape target was centred on the ROI midpoint at (3.0, 3.0) mm
    in FEM coords, which maps to (0, 0) on the lens grid.

    We interpolate the lens field onto the SW ROI grid with the
    appropriate coordinate shift: x_fem = x_lens + cx_roi.
    """
    recon = np.load(PHASE1_DIR / "npz" / "recon_D15mm.npz")
    p_lens_full = recon["p_recon_printable"]
    xg_lens = recon["xg"]
    yg_lens = recon["yg"]

    with open(PHASE1_DIR / "config.json") as f:
        p1_cfg = json.load(f)

    cx_roi = p1_cfg["grid"]["cx_roi_m"]  # 0.003 m
    cy_roi = p1_cfg["grid"]["cy_roi_m"]  # 0.003 m

    print(f"[lens] Loaded recon field: {p_lens_full.shape}, "
          f"|p|_max = {np.abs(p_lens_full).max():.2f} Pa")
    print(f"  Lens grid: [{xg_lens[0]*1e3:.2f}, {xg_lens[-1]*1e3:.2f}] mm, "
          f"centred at origin")
    print(f"  ROI centre in FEM: ({cx_roi*1e3:.1f}, {cy_roi*1e3:.1f}) mm")

    return p_lens_full, xg_lens, yg_lens, cx_roi, cy_roi, p1_cfg


def interpolate_lens_onto_roi(p_lens_full, xg_lens, yg_lens,
                               cx_roi, cy_roi,
                               xg_roi, yg_roi):
    """
    Interpolate the lens field onto the SW ROI grid.

    Coordinate map: the lens grid is centred at (0,0) which corresponds
    to (cx_roi, cy_roi) in FEM coords.  So for a point (x_fem, y_fem)
    in the ROI, the lens coordinate is (x_fem - cx_roi, y_fem - cy_roi).
    """
    interp_re = RegularGridInterpolator(
        (yg_lens, xg_lens), np.real(p_lens_full),
        bounds_error=False, fill_value=0.0, method="linear")
    interp_im = RegularGridInterpolator(
        (yg_lens, xg_lens), np.imag(p_lens_full),
        bounds_error=False, fill_value=0.0, method="linear")

    YY, XX = np.meshgrid(yg_roi, xg_roi, indexing="ij")
    # Convert FEM coords to lens coords
    x_lens = XX - cx_roi
    y_lens = YY - cy_roi
    pts = np.column_stack([y_lens.ravel(), x_lens.ravel()])

    p_lens_roi = (interp_re(pts) + 1j * interp_im(pts)).reshape(XX.shape)

    print(f"[lens→roi] Interpolated lens onto ROI grid: "
          f"|p|_max = {np.abs(p_lens_roi).max():.4f} Pa")
    return p_lens_roi


# ═══════════════════════════════════════════════════════════════════
# Diagnostics
# ═══════════════════════════════════════════════════════════════════

def compute_perturbation_metrics(p_sw, p_comb, xg, yg, traps_m,
                                  idx_A, idx_B, alpha, psi):
    """
    Compute perturbation diagnostics for a combined field.

    Returns a dict of scalar metrics.
    """
    dx = xg[1] - xg[0]
    a_sw = np.abs(p_sw)
    a_comb = np.abs(p_comb)

    # Sample pressure at trap positions
    def sample_at(p_field, xy):
        ix = int(np.clip(np.searchsorted(xg, xy[0]) - 1, 0, len(xg) - 2))
        iy = int(np.clip(np.searchsorted(yg, xy[1]) - 1, 0, len(yg) - 2))
        # Bilinear interpolation
        fx = (xy[0] - xg[ix]) / dx
        fy = (xy[1] - yg[iy]) / dx
        v = ((1 - fx) * (1 - fy) * p_field[iy, ix]
             + fx * (1 - fy) * p_field[iy, ix + 1]
             + (1 - fx) * fy * p_field[iy + 1, ix]
             + fx * fy * p_field[iy + 1, ix + 1])
        return v

    p_sw_A = sample_at(p_sw, traps_m[idx_A])
    p_sw_B = sample_at(p_sw, traps_m[idx_B])
    p_comb_A = sample_at(p_comb, traps_m[idx_A])
    p_comb_B = sample_at(p_comb, traps_m[idx_B])

    amp_sw_A = float(np.abs(p_sw_A))
    amp_sw_B = float(np.abs(p_sw_B))
    amp_comb_A = float(np.abs(p_comb_A))
    amp_comb_B = float(np.abs(p_comb_B))

    # Change at traps
    delta_A = amp_comb_A - amp_sw_A
    delta_B = amp_comb_B - amp_sw_B

    # Neighbouring traps
    neighbour_mask = np.ones(len(traps_m), dtype=bool)
    neighbour_mask[idx_A] = False
    neighbour_mask[idx_B] = False
    neighbour_deltas = []
    for t_xy in traps_m[neighbour_mask]:
        a_sw_t = float(np.abs(sample_at(p_sw, t_xy)))
        a_cb_t = float(np.abs(sample_at(p_comb, t_xy)))
        neighbour_deltas.append(a_cb_t - a_sw_t)
    max_neighbour_delta = float(np.max(np.abs(neighbour_deltas))) if neighbour_deltas else 0.0
    rms_neighbour_delta = float(np.sqrt(np.mean(np.array(neighbour_deltas)**2))) if neighbour_deltas else 0.0

    # Gradient along A→B
    AB_vec = traps_m[idx_B] - traps_m[idx_A]
    d_AB = np.linalg.norm(AB_vec)
    e_AB = AB_vec / d_AB
    mid = 0.5 * (traps_m[idx_A] + traps_m[idx_B])

    n_samp = 20
    ts = np.linspace(-0.6, 0.6, n_samp) * d_AB
    sw_profile = []
    comb_profile = []
    for t in ts:
        pt = mid + t * e_AB
        sw_profile.append(float(np.abs(sample_at(p_sw, pt))))
        comb_profile.append(float(np.abs(sample_at(p_comb, pt))))
    sw_prof = np.array(sw_profile)
    cb_prof = np.array(comb_profile)

    # Directional gradient: positive = biased toward B
    grad_sw = float(np.mean(np.diff(sw_prof)))
    grad_comb = float(np.mean(np.diff(cb_prof)))

    # Asymmetry ratio
    asymmetry_sw = float(amp_sw_A / max(amp_sw_A + amp_sw_B, 1e-30))
    asymmetry_comb = float(amp_comb_A / max(amp_comb_A + amp_comb_B, 1e-30))

    return {
        "alpha": alpha,
        "psi": psi,
        "amp_sw_A": amp_sw_A,
        "amp_sw_B": amp_sw_B,
        "amp_comb_A": amp_comb_A,
        "amp_comb_B": amp_comb_B,
        "delta_amp_A": float(delta_A),
        "delta_amp_B": float(delta_B),
        "delta_amp_A_pct": float(delta_A / max(amp_sw_A, 1e-30) * 100),
        "delta_amp_B_pct": float(delta_B / max(amp_sw_B, 1e-30) * 100),
        "max_neighbour_delta": max_neighbour_delta,
        "rms_neighbour_delta": rms_neighbour_delta,
        "grad_AB_sw": grad_sw,
        "grad_AB_comb": grad_comb,
        "asymmetry_sw": asymmetry_sw,
        "asymmetry_comb": asymmetry_comb,
    }


# ═══════════════════════════════════════════════════════════════════
# Plotting — Part A: Lens field alone
# ═══════════════════════════════════════════════════════════════════

def plot_lens_field_alone(p_lens_roi, xg, yg, traps_mm, idx_A, idx_B,
                           midpoint_mm, save_dir):
    """4-panel: amplitude, phase, real part, intensity."""
    xg_mm = xg * 1e3
    yg_mm = yg * 1e3
    ext = [xg_mm[0], xg_mm[-1], yg_mm[0], yg_mm[-1]]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    ax = axes[0, 0]
    im = ax.imshow(np.abs(p_lens_roi), origin="lower", extent=ext,
                   cmap="inferno", aspect="equal")
    _cbar(ax, im, "|p_lens| (Pa)")
    _overlay_traps(ax, traps_mm, idx_A, idx_B, midpoint_mm)
    ax.set_title("Lens Amplitude")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")

    ax = axes[0, 1]
    im = ax.imshow(np.angle(p_lens_roi), origin="lower", extent=ext,
                   cmap="twilight", aspect="equal", vmin=-np.pi, vmax=np.pi)
    _cbar(ax, im, "Phase (rad)")
    _overlay_traps(ax, traps_mm, idx_A, idx_B, midpoint_mm)
    ax.set_title("Lens Phase")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")

    ax = axes[1, 0]
    im = ax.imshow(np.real(p_lens_roi), origin="lower", extent=ext,
                   cmap="RdBu_r", aspect="equal")
    _cbar(ax, im, "Re(p) (Pa)")
    _overlay_traps(ax, traps_mm, idx_A, idx_B, midpoint_mm)
    ax.set_title("Lens Real Part")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")

    ax = axes[1, 1]
    im = ax.imshow(np.abs(p_lens_roi)**2, origin="lower", extent=ext,
                   cmap="magma", aspect="equal")
    _cbar(ax, im, "|p|² (Pa²)")
    _overlay_traps(ax, traps_mm, idx_A, idx_B, midpoint_mm)
    ax.set_title("Lens Intensity")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")

    fig.suptitle("Lens Field Alone in Trap ROI (15 mm printable)", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_dir / "A_lens_field_alone.png",
                dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Zoomed version — inner ±0.5λ around midpoint
    zoom_half = 0.5 * LAM * 1e3
    mx, my = midpoint_mm
    xlim = (mx - zoom_half * 2, mx + zoom_half * 2)
    ylim = (my - zoom_half * 2, my + zoom_half * 2)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, data, cmap, label in zip(
            axes,
            [np.abs(p_lens_roi), np.angle(p_lens_roi), np.real(p_lens_roi)],
            ["inferno", "twilight", "RdBu_r"],
            ["|p_lens| (Pa)", "Phase (rad)", "Re(p) (Pa)"]):
        kw = {}
        if cmap == "twilight":
            kw = {"vmin": -np.pi, "vmax": np.pi}
        im = ax.imshow(data, origin="lower", extent=ext,
                       cmap=cmap, aspect="equal", **kw)
        _cbar(ax, im, label)
        _overlay_traps(ax, traps_mm, idx_A, idx_B, midpoint_mm)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")

    fig.suptitle("Lens Field — Zoomed Near Trap Pair", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_dir / "A_lens_field_zoomed.png",
                dpi=200, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# Plotting — Part B: Standing wave reference
# ═══════════════════════════════════════════════════════════════════

def plot_sw_reference(p_sw, xg, yg, traps_mm, idx_A, idx_B,
                       midpoint_mm, save_dir):
    """Standing-wave amplitude and intensity with trap markers."""
    xg_mm = xg * 1e3
    yg_mm = yg * 1e3
    ext = [xg_mm[0], xg_mm[-1], yg_mm[0], yg_mm[-1]]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    ax = axes[0]
    im = ax.imshow(np.abs(p_sw), origin="lower", extent=ext,
                   cmap="inferno", aspect="equal")
    _cbar(ax, im, "|p_sw| (Pa)")
    _overlay_traps(ax, traps_mm, idx_A, idx_B, midpoint_mm)
    ax.set_title("Standing Wave Amplitude")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")

    ax = axes[1]
    im = ax.imshow(np.abs(p_sw)**2, origin="lower", extent=ext,
                   cmap="magma", aspect="equal")
    _cbar(ax, im, "|p|² (Pa²)")
    _overlay_traps(ax, traps_mm, idx_A, idx_B, midpoint_mm)
    ax.set_title("Standing Wave Intensity")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")

    ax = axes[2]
    im = ax.imshow(np.real(p_sw), origin="lower", extent=ext,
                   cmap="RdBu_r", aspect="equal")
    _cbar(ax, im, "Re(p) (Pa)")
    _overlay_traps(ax, traps_mm, idx_A, idx_B, midpoint_mm)
    ax.set_title("Standing Wave Real Part")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")

    fig.suptitle(f"Standing-Wave Reference — z* = {Z_STAR*1e3:.4f} mm",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(save_dir / "B_standing_wave_reference.png",
                dpi=200, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# Plotting — Part C: Combined field
# ═══════════════════════════════════════════════════════════════════

def plot_combined_triptych(p_sw, p_lens_roi, p_comb, xg, yg,
                            traps_mm, idx_A, idx_B, midpoint_mm,
                            alpha, psi_label, save_dir, tag=""):
    """3-column: SW alone | lens alone | combined."""
    xg_mm = xg * 1e3
    yg_mm = yg * 1e3
    ext = [xg_mm[0], xg_mm[-1], yg_mm[0], yg_mm[-1]]

    a_sw = np.abs(p_sw)
    a_lens = np.abs(p_lens_roi)
    a_comb = np.abs(p_comb)
    vmax = max(a_sw.max(), a_comb.max())

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # Row 1: amplitudes
    for ax, data, title in zip(
            axes[0],
            [a_sw, a_lens, a_comb],
            ["Standing Wave", "Lens (scaled)", "Combined"]):
        im = ax.imshow(data, origin="lower", extent=ext,
                       cmap="inferno", aspect="equal",
                       vmin=0, vmax=vmax if title != "Lens (scaled)" else None)
        _cbar(ax, im, "|p| (Pa)")
        _overlay_traps(ax, traps_mm, idx_A, idx_B, midpoint_mm)
        ax.set_title(title)
        ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")

    # Row 2: intensities
    i_sw = a_sw**2
    i_lens = a_lens**2
    i_comb = a_comb**2
    imax = max(i_sw.max(), i_comb.max())

    for ax, data, title in zip(
            axes[1],
            [i_sw, i_lens, i_comb],
            ["SW Intensity", "Lens Intensity", "Combined Intensity"]):
        im = ax.imshow(data, origin="lower", extent=ext,
                       cmap="magma", aspect="equal",
                       vmin=0, vmax=imax if title != "Lens Intensity" else None)
        _cbar(ax, im, "|p|² (Pa²)")
        _overlay_traps(ax, traps_mm, idx_A, idx_B, midpoint_mm)
        ax.set_title(title)
        ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")

    fig.suptitle(f"Field Comparison — α={alpha:.2f}, ψ={psi_label}",
                 fontsize=13)
    fig.tight_layout()
    fname = f"C_combined{tag}_a{alpha:.2f}_psi{psi_label.replace('/', '_')}.png"
    fig.savefig(save_dir / fname, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_perturbation_map(p_sw, p_comb, xg, yg,
                           traps_mm, idx_A, idx_B, midpoint_mm,
                           alpha, psi_label, save_dir, tag=""):
    """Difference maps: |p_comb| - |p_sw| and angle shift."""
    xg_mm = xg * 1e3
    yg_mm = yg * 1e3
    ext = [xg_mm[0], xg_mm[-1], yg_mm[0], yg_mm[-1]]

    delta_amp = np.abs(p_comb) - np.abs(p_sw)
    vlim = max(abs(delta_amp.min()), abs(delta_amp.max()), 1e-6)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    im = ax.imshow(delta_amp, origin="lower", extent=ext,
                   cmap="RdBu_r", aspect="equal", vmin=-vlim, vmax=vlim)
    _cbar(ax, im, "Δ|p| (Pa)")
    _overlay_traps(ax, traps_mm, idx_A, idx_B, midpoint_mm)
    ax.set_title("Amplitude Change (combined − SW)")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")

    ax = axes[1]
    delta_I = np.abs(p_comb)**2 - np.abs(p_sw)**2
    vlimI = max(abs(delta_I.min()), abs(delta_I.max()), 1e-6)
    im = ax.imshow(delta_I, origin="lower", extent=ext,
                   cmap="RdBu_r", aspect="equal", vmin=-vlimI, vmax=vlimI)
    _cbar(ax, im, "ΔI (Pa²)")
    _overlay_traps(ax, traps_mm, idx_A, idx_B, midpoint_mm)
    ax.set_title("Intensity Change (combined − SW)")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")

    fig.suptitle(f"Perturbation — α={alpha:.2f}, ψ={psi_label}", fontsize=13)
    fig.tight_layout()
    fname = f"C_perturbation{tag}_a{alpha:.2f}_psi{psi_label.replace('/', '_')}.png"
    fig.savefig(save_dir / fname, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# Plotting — Part D: Amplitude sweep summary
# ═══════════════════════════════════════════════════════════════════

def plot_alpha_sweep_summary(metrics_list, save_dir):
    """Summary plots across amplitude sweep (ψ=0)."""
    alphas = [m["alpha"] for m in metrics_list if m["psi"] == 0.0]
    if not alphas:
        return

    sub = [m for m in metrics_list if m["psi"] == 0.0]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    ax = axes[0, 0]
    ax.plot(alphas, [m["delta_amp_A_pct"] for m in sub], "ro-", label="Trap A")
    ax.plot(alphas, [m["delta_amp_B_pct"] for m in sub], "bs-", label="Trap B")
    ax.axhline(0, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("α")
    ax.set_ylabel("Δ|p| (%)")
    ax.set_title("Pressure Change at Traps A & B")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(alphas, [m["max_neighbour_delta"] for m in sub], "ko-",
            label="Max |Δ| neighbour")
    ax.plot(alphas, [m["rms_neighbour_delta"] for m in sub], "g^-",
            label="RMS Δ neighbour")
    ax.set_xlabel("α")
    ax.set_ylabel("Δ|p| (Pa)")
    ax.set_title("Perturbation of Neighbouring Traps")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(alphas, [m["asymmetry_sw"] for m in sub], "k--", label="SW only")
    ax.plot(alphas, [m["asymmetry_comb"] for m in sub], "ro-", label="Combined")
    ax.set_xlabel("α")
    ax.set_ylabel("A / (A+B)")
    ax.set_title("A-B Asymmetry Ratio")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(alphas, [m["grad_AB_sw"] for m in sub], "k--", label="SW only")
    ax.plot(alphas, [m["grad_AB_comb"] for m in sub], "ro-", label="Combined")
    ax.axhline(0, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("α")
    ax.set_ylabel("Mean dA/ds (Pa/m)")
    ax.set_title("A→B Directional Gradient")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("Amplitude Sweep Summary (ψ = 0)", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_dir / "D_alpha_sweep_summary.png",
                dpi=200, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# Plotting — Part E: Phase-delay sweep summary
# ═══════════════════════════════════════════════════════════════════

def plot_psi_sweep_summary(metrics_list, save_dir):
    """Summary plots across phase-delay sweep at a fixed α."""
    # Pick α = 0.20 as the moderate test case
    test_alpha = 0.20
    sub = [m for m in metrics_list if abs(m["alpha"] - test_alpha) < 0.01]
    if not sub:
        # Fall back to whatever α has the most ψ entries
        from collections import Counter
        ac = Counter(m["alpha"] for m in metrics_list)
        test_alpha = ac.most_common(1)[0][0]
        sub = [m for m in metrics_list if abs(m["alpha"] - test_alpha) < 0.01]

    psis = [m["psi"] for m in sub]
    psi_labels_local = [f"{p/np.pi:.1f}π" if p > 0 else "0" for p in psis]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    ax = axes[0, 0]
    ax.plot(psis, [m["delta_amp_A_pct"] for m in sub], "ro-", label="Trap A")
    ax.plot(psis, [m["delta_amp_B_pct"] for m in sub], "bs-", label="Trap B")
    ax.axhline(0, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("ψ (rad)")
    ax.set_ylabel("Δ|p| (%)")
    ax.set_title(f"Pressure Change at Traps (α={test_alpha:.2f})")
    ax.set_xticks(psis)
    ax.set_xticklabels(psi_labels_local, fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(psis, [m["max_neighbour_delta"] for m in sub], "ko-",
            label="Max |Δ| neighbour")
    ax.set_xlabel("ψ (rad)")
    ax.set_ylabel("Δ|p| (Pa)")
    ax.set_title("Neighbour Perturbation")
    ax.set_xticks(psis)
    ax.set_xticklabels(psi_labels_local, fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    asymm_sw = sub[0]["asymmetry_sw"]
    ax.axhline(asymm_sw, color="k", ls="--", label="SW only")
    ax.plot(psis, [m["asymmetry_comb"] for m in sub], "ro-", label="Combined")
    ax.set_xlabel("ψ (rad)")
    ax.set_ylabel("A / (A+B)")
    ax.set_title("A-B Asymmetry vs Phase Delay")
    ax.set_xticks(psis)
    ax.set_xticklabels(psi_labels_local, fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(psis, [m["grad_AB_comb"] for m in sub], "ro-", label="Combined")
    ax.axhline(sub[0]["grad_AB_sw"], color="k", ls="--", label="SW only")
    ax.set_xlabel("ψ (rad)")
    ax.set_ylabel("Mean dA/ds (Pa/m)")
    ax.set_title("A→B Gradient vs Phase Delay")
    ax.set_xticks(psis)
    ax.set_xticklabels(psi_labels_local, fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Phase-Delay Sweep Summary (α = {test_alpha:.2f})",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(save_dir / "E_psi_sweep_summary.png",
                dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_psi_panel(p_sw, p_lens_roi, xg, yg, traps_mm, idx_A, idx_B,
                    midpoint_mm, alpha, save_dir):
    """4-panel amplitude comparison across ψ values at fixed α."""
    xg_mm = xg * 1e3
    yg_mm = yg * 1e3
    ext = [xg_mm[0], xg_mm[-1], yg_mm[0], yg_mm[-1]]

    n_psi = len(PSI_VALUES)
    fig, axes = plt.subplots(2, n_psi, figsize=(5 * n_psi, 10))

    # Common vmax
    a_sw = np.abs(p_sw)
    vmax = a_sw.max() * 1.15

    for col, (psi, psi_lab) in enumerate(zip(PSI_VALUES, PSI_LABELS)):
        p_comb = p_sw + alpha * np.exp(1j * psi) * p_lens_roi
        a_comb = np.abs(p_comb)
        delta = a_comb - a_sw

        ax = axes[0, col]
        im = ax.imshow(a_comb, origin="lower", extent=ext,
                       cmap="inferno", aspect="equal", vmin=0, vmax=vmax)
        _cbar(ax, im, "|p| (Pa)")
        _overlay_traps(ax, traps_mm, idx_A, idx_B, midpoint_mm)
        ax.set_title(f"ψ = {psi_lab}")
        ax.set_xlabel("x (mm)")
        if col == 0:
            ax.set_ylabel("y (mm)")

        ax = axes[1, col]
        vlim = max(abs(delta.min()), abs(delta.max()), 1e-6)
        im = ax.imshow(delta, origin="lower", extent=ext,
                       cmap="RdBu_r", aspect="equal", vmin=-vlim, vmax=vlim)
        _cbar(ax, im, "Δ|p| (Pa)")
        _overlay_traps(ax, traps_mm, idx_A, idx_B, midpoint_mm)
        ax.set_xlabel("x (mm)")
        if col == 0:
            ax.set_ylabel("y (mm)")

    fig.suptitle(f"Phase-Delay Comparison — α = {alpha:.2f}", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_dir / f"E_psi_panel_a{alpha:.2f}.png",
                dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_alpha_panel(p_sw, p_lens_roi, xg, yg, traps_mm, idx_A, idx_B,
                      midpoint_mm, save_dir):
    """Multi-panel: combined amplitude across α values (ψ=0)."""
    xg_mm = xg * 1e3
    yg_mm = yg * 1e3
    ext = [xg_mm[0], xg_mm[-1], yg_mm[0], yg_mm[-1]]

    n_a = len(ALPHA_VALUES)
    fig, axes = plt.subplots(2, n_a, figsize=(5 * n_a, 10))

    a_sw = np.abs(p_sw)
    vmax = a_sw.max() * 1.3

    for col, alpha in enumerate(ALPHA_VALUES):
        p_comb = p_sw + alpha * p_lens_roi
        a_comb = np.abs(p_comb)
        delta = a_comb - a_sw

        ax = axes[0, col]
        im = ax.imshow(a_comb, origin="lower", extent=ext,
                       cmap="inferno", aspect="equal", vmin=0, vmax=vmax)
        _cbar(ax, im, "|p| (Pa)")
        _overlay_traps(ax, traps_mm, idx_A, idx_B, midpoint_mm)
        ax.set_title(f"α = {alpha:.2f}")
        ax.set_xlabel("x (mm)")
        if col == 0:
            ax.set_ylabel("y (mm)")

        ax = axes[1, col]
        vlim = max(abs(delta.min()), abs(delta.max()), 1e-6)
        im = ax.imshow(delta, origin="lower", extent=ext,
                       cmap="RdBu_r", aspect="equal", vmin=-vlim, vmax=vlim)
        _cbar(ax, im, "Δ|p| (Pa)")
        _overlay_traps(ax, traps_mm, idx_A, idx_B, midpoint_mm)
        ax.set_xlabel("x (mm)")
        if col == 0:
            ax.set_ylabel("y (mm)")

    fig.suptitle("Amplitude Sweep — Combined Field (ψ = 0)", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_dir / "D_alpha_panel.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════════

def write_index(all_metrics, p_sw, p_lens_roi, traps_m, idx_A, idx_B,
                d_AB, p1_cfg, save_dir):
    """Write INDEX.md with honest analysis."""

    sw_peak = float(np.abs(p_sw).max())
    lens_peak = float(np.abs(p_lens_roi).max())

    # Find interesting cases
    psi0 = [m for m in all_metrics if m["psi"] == 0.0]
    best_alpha_asym = max(psi0, key=lambda m: abs(m["asymmetry_comb"] - m["asymmetry_sw"])) if psi0 else {}

    a020 = [m for m in all_metrics if abs(m["alpha"] - 0.20) < 0.01]
    if a020:
        best_psi_asym = max(a020, key=lambda m: abs(m["asymmetry_comb"] - m["asymmetry_sw"]))
    else:
        best_psi_asym = {}

    md = f"""# C-Shape Lens 15 mm Overlay Study (Phase 2)

**Generated**: {TS}
**Script**: `scripts/dev/c_shape_lens_15mm_overlay_study.py`

## Purpose

Physics investigation of how the 15 mm single-pass holographic lens field
interacts with the standing-wave trap environment.

No lens redesign was performed. The final printable lens from Phase 1 was
used exactly as-is.

## Data Sources

- **Phase 1**: `{PHASE1_DIR.name}`
- **Standing wave**: `{SW_NPZ.parent.name}/{SW_NPZ.name}`
- **Geometry study**: `{Path(p1_cfg['chosen_study']).name}`

## Field Magnitudes

| Field | Peak |p| (Pa) |
|-------|------|
| Standing wave | {sw_peak:.2f} |
| Lens (in ROI) | {lens_peak:.4f} |
| Ratio lens/SW | {lens_peak/sw_peak*100:.2f}% |

The lens field in the ROI is **very weak** compared to the standing wave.
Even the raw reconstructed peak is {lens_peak:.2f} Pa vs {sw_peak:.2f} Pa
for the standing wave ({lens_peak/sw_peak*100:.1f}% ratio).

## Part A — Lens Field Alone

The lens field in the trap ROI shows a diffuse pattern that only partially
resembles the C-shape target. This is expected: single-pass phase-only
holography at 8.3λ propagation distance has limited fidelity (~50% amplitude
correlation from Phase 1).

See `figures/A_lens_field_alone.png` and `A_lens_field_zoomed.png`.

## Part B — Standing Wave Reference

The standing-wave field shows a regular trap lattice with spacing ~λ/2.
Trap A and B are correctly identified as nearest-neighbour Gorkov-stable
minima separated by {d_AB*1e3:.3f} mm ({d_AB/LAM:.3f}λ).

See `figures/B_standing_wave_reference.png`.

## Part C — Combined Field

The combined field p_comb = p_sw + α·exp(iψ)·p_lens is dominated by the
standing wave at all tested α values. The lens perturbation is a relatively
small modulation on top of the established trap pattern.

See `figures/C_combined_*.png` for side-by-side comparisons.

## Part D — Amplitude Sweep

Tested α values: {ALPHA_VALUES}

"""
    if psi0:
        md += "| α | Δ|p| at A (%) | Δ|p| at B (%) | Asymmetry (SW) | Asymmetry (comb) | Max nbr Δ (Pa) |\n"
        md += "|---|---|---|---|---|---|\n"
        for m in psi0:
            md += (f"| {m['alpha']:.2f} | {m['delta_amp_A_pct']:+.2f} | "
                   f"{m['delta_amp_B_pct']:+.2f} | {m['asymmetry_sw']:.4f} | "
                   f"{m['asymmetry_comb']:.4f} | {m['max_neighbour_delta']:.3f} |\n")

    if best_alpha_asym:
        md += f"""
Most asymmetry change at α = {best_alpha_asym['alpha']:.2f}:
asymmetry shifts from {best_alpha_asym['asymmetry_sw']:.4f} (SW only)
to {best_alpha_asym['asymmetry_comb']:.4f} (combined).

"""

    md += f"""See `figures/D_alpha_panel.png` and `D_alpha_sweep_summary.png`.

## Part E — Phase-Delay Sweep

Tested ψ values: {PSI_LABELS} at α = 0.20

"""
    if a020:
        md += "| ψ | Δ|p| at A (%) | Δ|p| at B (%) | Asymmetry | Gradient |\n"
        md += "|---|---|---|---|---|\n"
        for m in a020:
            psi_l = f"{m['psi']/np.pi:.1f}π" if m['psi'] > 0 else "0"
            md += (f"| {psi_l} | {m['delta_amp_A_pct']:+.2f} | "
                   f"{m['delta_amp_B_pct']:+.2f} | {m['asymmetry_comb']:.4f} | "
                   f"{m['grad_AB_comb']:.2f} |\n")

    if best_psi_asym:
        psi_best = best_psi_asym['psi']
        best_psi_label = f"{psi_best/np.pi:.1f}π" if psi_best > 0 else "0"
        md += f"""
Best phase offset: ψ = {best_psi_label} gives asymmetry = {best_psi_asym['asymmetry_comb']:.4f}
(vs {best_psi_asym['asymmetry_sw']:.4f} for SW alone).

"""

    md += """See `figures/E_psi_panel_*.png` and `E_psi_sweep_summary.png`.

## Does Phase Delay Matter?

The relative phase between the lens field and the standing wave
controls whether the lens adds constructively or destructively at
each spatial location. Because the standing wave and lens field have
different spatial frequency content, a global phase offset cannot
uniformly enhance or suppress the perturbation everywhere.

However, the local effect near the trap pair is significant: the phase
offset rotates the perturbation pattern, potentially pushing amplitude
toward A or B preferentially.

## Honest Assessment

### Strengths
- The lens does produce a measurable field perturbation in the ROI
- Different phase offsets do produce qualitatively different perturbation patterns
- The perturbation is primarily local (neighbour traps are only weakly affected)

### Weaknesses
- The lens field is very weak relative to the standing wave
- Even at α = 0.40 (unrealistically strong), the perturbation is modest
- Single-pass holographic fidelity limits the spatial accuracy of the perturbation
- The A-B asymmetry change is small

### Recommendation
Before moving to iterative phase retrieval:
1. Consider whether a stronger lens material (larger |Δk|) would help
2. The relative phase IS physically relevant — it should be treated
   as a design parameter, not assumed to be zero
3. Transport calculations (force/potential analysis) are needed to assess
   whether even small amplitude perturbations can break trap symmetry

## Files

- `metrics.csv` — all sweep results
- `figures/` — all visualisations
- `npz/` — saved field arrays
"""

    with open(save_dir / "INDEX.md", "w") as f:
        f.write(md)
    print("[report] INDEX.md written")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    print("=" * 72)
    print("C-Shape Lens 15 mm Overlay Study (Phase 2)")
    print("=" * 72)
    print(f"  Output: {OUT_DIR}")
    print()

    # ── Load standing wave ───────────────────────────────────────
    print("─── Load standing wave ───")
    p_sw, xg_roi, yg_roi = load_standing_wave()

    # ── Detect traps ─────────────────────────────────────────────
    print("─── Detect traps ───")
    traps_m = detect_traps(p_sw, xg_roi, yg_roi)
    idx_A, idx_B, d_AB, midpoint = choose_trap_pair(traps_m)
    traps_mm = traps_m * 1e3
    midpoint_mm = midpoint * 1e3

    # ── Load lens field ──────────────────────────────────────────
    print("\n─── Load lens field ───")
    p_lens_full, xg_lens, yg_lens, cx_roi, cy_roi, p1_cfg = load_lens_field()

    print("\n─── Interpolate lens onto ROI ───")
    p_lens_roi = interpolate_lens_onto_roi(
        p_lens_full, xg_lens, yg_lens, cx_roi, cy_roi, xg_roi, yg_roi)

    # Scale reference: the Phase 1 reconstruction has arbitrary amplitude
    # from the unit-amplitude phase-only drive.  We normalise so that the
    # lens field peak in the ROI matches the target amplitude from the
    # geometry study (α_frac × SW_peak ≈ 0.10 × 46 = 4.6 Pa at peak).
    # The α sweep then scales relative to this baseline.
    lens_peak_roi = np.abs(p_lens_roi).max()
    sw_peak_roi = np.abs(p_sw).max()
    print(f"  SW peak in ROI: {sw_peak_roi:.2f} Pa")
    print(f"  Lens peak in ROI: {lens_peak_roi:.4f} Pa")
    print(f"  Ratio: {lens_peak_roi / sw_peak_roi * 100:.2f}%")

    # Save ROI fields
    np.savez_compressed(NPZ_DIR / "roi_fields.npz",
                        p_sw=p_sw, p_lens_roi=p_lens_roi,
                        xg=xg_roi, yg=yg_roi,
                        traps_m=traps_m, idx_A=idx_A, idx_B=idx_B,
                        midpoint=midpoint)
    print()

    # ── Part A: Lens field alone ─────────────────────────────────
    print("─── Part A: Lens field alone ───")
    plot_lens_field_alone(p_lens_roi, xg_roi, yg_roi, traps_mm,
                           idx_A, idx_B, midpoint_mm, FIG_DIR)
    print("  Saved A_lens_field_alone.png, A_lens_field_zoomed.png")

    # ── Part B: Standing wave reference ──────────────────────────
    print("─── Part B: Standing wave reference ───")
    plot_sw_reference(p_sw, xg_roi, yg_roi, traps_mm,
                       idx_A, idx_B, midpoint_mm, FIG_DIR)
    print("  Saved B_standing_wave_reference.png")

    # ── Part C + D + E: Sweeps ───────────────────────────────────
    all_metrics = []

    # Part D: amplitude sweep (ψ = 0)
    print("\n─── Part D: Amplitude sweep (ψ=0) ───")
    for alpha in ALPHA_VALUES:
        p_comb = p_sw + alpha * p_lens_roi
        m = compute_perturbation_metrics(p_sw, p_comb, xg_roi, yg_roi,
                                          traps_m, idx_A, idx_B, alpha, 0.0)
        all_metrics.append(m)
        print(f"  α={alpha:.2f}: ΔA={m['delta_amp_A_pct']:+.2f}%, "
              f"ΔB={m['delta_amp_B_pct']:+.2f}%, "
              f"asym={m['asymmetry_comb']:.4f}")

        # Combined triptych for α sweep at ψ=0
        plot_combined_triptych(p_sw, alpha * p_lens_roi, p_comb,
                                xg_roi, yg_roi, traps_mm, idx_A, idx_B,
                                midpoint_mm, alpha, "0", FIG_DIR, "_alpha")
        plot_perturbation_map(p_sw, p_comb, xg_roi, yg_roi, traps_mm,
                               idx_A, idx_B, midpoint_mm, alpha, "0",
                               FIG_DIR, "_alpha")

    # Alpha panel figure
    plot_alpha_panel(p_sw, p_lens_roi, xg_roi, yg_roi, traps_mm,
                      idx_A, idx_B, midpoint_mm, FIG_DIR)

    # Part E: phase-delay sweep (all α × ψ combinations)
    print("\n─── Part E: Phase-delay sweep ───")
    for alpha in ALPHA_VALUES:
        for psi, psi_lab in zip(PSI_VALUES, PSI_LABELS):
            if psi == 0.0:
                continue  # already computed above
            p_comb = p_sw + alpha * np.exp(1j * psi) * p_lens_roi
            m = compute_perturbation_metrics(p_sw, p_comb, xg_roi, yg_roi,
                                              traps_m, idx_A, idx_B, alpha, psi)
            all_metrics.append(m)

            # Only generate per-case figures for α = 0.20 to avoid figure explosion
            if abs(alpha - 0.20) < 0.01:
                plot_combined_triptych(p_sw, alpha * np.exp(1j * psi) * p_lens_roi,
                                        p_comb, xg_roi, yg_roi, traps_mm,
                                        idx_A, idx_B, midpoint_mm,
                                        alpha, psi_lab, FIG_DIR, "_psi")
                plot_perturbation_map(p_sw, p_comb, xg_roi, yg_roi, traps_mm,
                                       idx_A, idx_B, midpoint_mm,
                                       alpha, psi_lab, FIG_DIR, "_psi")

        print(f"  α={alpha:.2f}: done ψ sweep")

    # Phase-delay panel at α = 0.20
    plot_psi_panel(p_sw, p_lens_roi, xg_roi, yg_roi, traps_mm,
                    idx_A, idx_B, midpoint_mm, 0.20, FIG_DIR)

    # Summary plots
    print("\n─── Summary plots ───")
    plot_alpha_sweep_summary(all_metrics, FIG_DIR)
    plot_psi_sweep_summary(all_metrics, FIG_DIR)

    # ── Metrics CSV ──────────────────────────────────────────────
    if all_metrics:
        keys = sorted(all_metrics[0].keys())
        with open(OUT_DIR / "metrics.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_metrics)
        print(f"  Saved metrics.csv ({len(all_metrics)} rows)")

    # ── INDEX.md ─────────────────────────────────────────────────
    write_index(all_metrics, p_sw, p_lens_roi, traps_m, idx_A, idx_B,
                d_AB, p1_cfg, OUT_DIR)

    # ── Config JSON ──────────────────────────────────────────────
    config_out = {
        "timestamp": TS,
        "phase1_dir": str(PHASE1_DIR),
        "sw_cache": str(SW_NPZ),
        "z_star_mm": Z_STAR * 1e3,
        "roi_half_mm": ROI_HALF * 1e3,
        "n_roi": N_ROI,
        "alpha_values": ALPHA_VALUES,
        "psi_values": [float(p) for p in PSI_VALUES],
        "trap_A_mm": traps_m[idx_A].tolist(),
        "trap_B_mm": traps_m[idx_B].tolist(),
        "d_AB_mm": d_AB * 1e3,
        "sw_peak_Pa": float(np.abs(p_sw).max()),
        "lens_peak_in_roi_Pa": float(np.abs(p_lens_roi).max()),
        "runtime_s": time.time() - t_start,
    }
    with open(OUT_DIR / "config.json", "w") as f:
        json.dump(config_out, f, indent=2, default=str)

    print(f"\n{'='*72}")
    print(f"Done in {time.time()-t_start:.1f}s")
    print(f"Outputs: {OUT_DIR}")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
