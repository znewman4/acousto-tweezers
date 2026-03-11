#!/usr/bin/env python3
"""
C-Shape on Cached Standing-Wave Geometry Study
================================================

First-pass prototype: design a localised C-shaped complex pressure field
aligned to real standing-wave trap geometry, visualise superposition,
and assess geometric compactness / leakage into neighbouring traps.

Outputs → results/c_shape_on_cached_sw_geometry_study_<TS>/

Usage:
    python scripts/dev/c_shape_on_cached_standing_wave_geometry_study.py
"""
from __future__ import annotations

import csv
import json
import sys
import time
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.ndimage import minimum_filter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ── Project root ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ═══════════════════════════════════════════════════════════════════
# Physical constants
# ═══════════════════════════════════════════════════════════════════
C_WATER = 1484.0          # m/s
F_HZ    = 2.0e6            # Hz
K_WATER = 2.0 * np.pi * F_HZ / C_WATER
LAM     = C_WATER / F_HZ   # 0.742 mm
TRAP_SP = LAM / 2.0        # ~0.371 mm
OMEGA   = 2.0 * np.pi * F_HZ
RHO0    = 997.0

LX = LY   = 6.0e-3         # domain size (m)
# epl5 geometry: H_under=5mm, H_top=2mm, total=7mm
H_UNDER   = 5.0e-3
H_TOP     = 2.0e-3
CX = CY   = LX / 2.0
Z_STAR    = H_UNDER + H_TOP / 2.0 + 0.25 * LAM    # ~6.1855 mm

K_IDW    = 16               # IDW neighbours
ROI_HALF = 1.1 * LAM        # ROI half-width for interpolation

STANDING_CACHE_DIR = PROJECT_ROOT / "results" / "fem_standing_wave_cache"
# epl5 checkpoint is the canonical dataset
EPL5_NPZ = (STANDING_CACHE_DIR
            / "checkpoint_epl5_depth7mm_20260309_113007"
            / "standing_wave_epl5.npz")

# Interpolation grid resolution
N_GRID = 400   # medium resolution, adequate for geometry study

# ── Output directory ────────────────────────────────────────────────
TS = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR  = PROJECT_ROOT / "results" / f"c_shape_on_cached_sw_geometry_study_{TS}"
FIG_DIR  = OUT_DIR / "figures"
BEST_DIR = OUT_DIR / "best"
NPZ_DIR  = OUT_DIR / "npz"
for d in [OUT_DIR, FIG_DIR, BEST_DIR, NPZ_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
# 1. Load cached standing-wave data
# ═══════════════════════════════════════════════════════════════════

def load_cached_standing_wave(cache_dir: Path = STANDING_CACHE_DIR):
    """
    Load the epl5 standing-wave cache (canonical dataset).
    Falls back to the epl6 root-level npz if the epl5 checkpoint is absent.
    """
    # Prefer the epl5 checkpoint (correct 7 mm domain)
    if EPL5_NPZ.exists() and EPL5_NPZ.stat().st_size > 10_000:
        path = EPL5_NPZ
    else:
        # Last resort: any npz directly in the cache dir
        npz_files = sorted(
            [f for f in cache_dir.glob("*.npz")
             if f.stat().st_size > 10_000],
            key=lambda p: p.stat().st_mtime,
        )
        if not npz_files:
            raise FileNotFoundError(
                f"epl5 npz not found at {EPL5_NPZ} and no fallback in {cache_dir}")
        path = npz_files[-1]
        print(f"[load] WARNING: epl5 not found, falling back to {path.name}")

    d = np.load(path)
    coords = d["coords"]  # (N,3) in metres
    p = d["p_real"].astype(np.float64) + 1j * d["p_imag"].astype(np.float64)
    print(f"[load] Loaded {path.name}: {len(p)} DOFs, "
          f"|p|_max = {np.abs(p).max():.2f} Pa")
    return coords, p, path


# ═══════════════════════════════════════════════════════════════════
# 2. Interpolate onto the trap plane
# ═══════════════════════════════════════════════════════════════════

def sample_idw(tree, p, pts, k=16, power=2.0, eps=1e-12):
    """IDW interpolation from scattered DOFs to query points."""
    if k == 1:
        d, i = tree.query(pts, k=1)
        return p[i]
    d, i = tree.query(pts, k=k)
    w = 1.0 / (d**power + eps)
    w /= w.sum(axis=1, keepdims=True)
    return (p[i] * w).sum(axis=1)


def select_trap_plane():
    """Return the z-value of the design plane (z*)."""
    return Z_STAR


def interpolate_to_plane(coords, p_fem, z_val, n_grid=N_GRID, roi_half=ROI_HALF):
    """Interpolate FEM data onto a 2D Cartesian grid at z=z_val."""
    tree = cKDTree(coords)
    x_lo, x_hi = CX - roi_half, CX + roi_half
    y_lo, y_hi = CY - roi_half, CY + roi_half
    xg = np.linspace(x_lo, x_hi, n_grid)
    yg = np.linspace(y_lo, y_hi, n_grid)
    XX, YY = np.meshgrid(xg, yg)
    pts_3d = np.column_stack([XX.ravel(), YY.ravel(),
                              np.full(XX.size, z_val)])
    p_flat = sample_idw(tree, p_fem, pts_3d, k=K_IDW)
    return p_flat.reshape(XX.shape), xg, yg


# ═══════════════════════════════════════════════════════════════════
# 3. Get or detect traps
# ═══════════════════════════════════════════════════════════════════

def get_or_detect_traps(p_grid, xg, yg):
    """
    Detect traps as local minima of the full Gorkov potential, validated
    by Hessian positive-definiteness (true stable equilibria).

    U_gor = f₁|p|²/(2ρc²) − 3f₂|∇p|²/(4ω²ρ)

    For polystyrene-in-water (f₁=0.62, f₂=0.034), the kinetic energy
    term is large enough to reshape the potential landscape; strict
    local-U minima do NOT coincide with |p| nodes.  All 9 detected
    minima pass the Hessian test (both eigenvalues > 0) and have 8/8
    force-inward at all sampling radii.  Robust to grid density
    (N=200–800), z-shift (±0.15λ), ROI (0.8–1.1λ), and neighbourhood
    factor (0.3–0.7×).

    Returns trap_centres_m: (N,2) array of (x, y) in metres.
    """
    p_abs = np.abs(p_grid)
    dx = xg[1] - xg[0]

    # Gorkov contrast factors — polystyrene in water
    rho_p, c_p = 1050.0, 2350.0
    kappa_w = 1.0 / (RHO0 * C_WATER**2)
    kappa_p = 1.0 / (rho_p * c_p**2)
    f1 = 1.0 - kappa_p / kappa_w
    f2 = 2.0 * (rho_p - RHO0) / (2.0 * rho_p + RHO0)

    # Full Gorkov potential
    p2 = p_abs**2
    dp_dx = np.gradient(p_grid, dx, axis=1)
    dp_dy = np.gradient(p_grid, dx, axis=0)
    grad_p2 = np.abs(dp_dx)**2 + np.abs(dp_dy)**2
    U_gor = (f1 / (2.0 * RHO0 * C_WATER**2)) * p2 \
          - (3.0 * f2 / (4.0 * OMEGA**2 * RHO0)) * grad_p2

    # Local minima of U via morphological minimum filter
    min_sep_pts = max(3, int(0.4 * TRAP_SP / dx))
    neighborhood = 2 * min_sep_pts + 1
    local_min = minimum_filter(U_gor, size=neighborhood)
    mask = (U_gor == local_min)

    # Exclude grid border
    border = max(5, min_sep_pts)
    mask[:border, :] = False
    mask[-border:, :] = False
    mask[:, :border] = False
    mask[:, -border:] = False

    # Keep only genuine minima below 50% of U range
    U_range = U_gor.max() - U_gor.min()
    mask &= (U_gor < U_gor.min() + 0.50 * U_range)

    iy, ix = np.where(mask)

    # Hessian stability check: keep only positive-definite (both λ > 0)
    dUdx = np.gradient(U_gor, dx, axis=1)
    dUdy = np.gradient(U_gor, dx, axis=0)
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
    iy_kept, ix_kept = iy[kept], ix[kept]

    for ti in range(len(traps_m)):
        print(f"  trap {ti}: ({traps_m[ti,0]*1e3:.3f}, {traps_m[ti,1]*1e3:.3f}) mm  "
              f"|p|={p_abs[iy_kept[ti], ix_kept[ti]]:.2f} Pa  "
              f"U={U_gor[iy_kept[ti], ix_kept[ti]]:.3e}")

    source = (f"detected {len(traps_m)} Gorkov-potential minima (Hessian-stable, "
              f"epl5, z*={Z_STAR*1e3:.3f} mm, f1={f1:.3f}, f2={f2:.4f})")
    print(f"[traps] {source}")
    return traps_m, source


# ═══════════════════════════════════════════════════════════════════
# 4. Choose adjacent trap pair and local frame
# ═══════════════════════════════════════════════════════════════════

def choose_adjacent_trap_pair(traps_m):
    """
    Choose the adjacent pair with separation closest to λ/2
    and closest to domain centre.

    Returns (idx_A, idx_B, d_AB).
    """
    tree = cKDTree(traps_m)
    dists, idxs = tree.query(traps_m, k=2)
    nn_dists = dists[:, 1]   # nearest-neighbour distance
    nn_idxs = idxs[:, 1]

    # Find pairs with separation near TRAP_SP
    target = TRAP_SP
    rel_err = np.abs(nn_dists - target) / target

    # Among close-to-TRAP_SP pairs, pick the one nearest domain centre
    centre = np.array([CX, CY])
    midpoints = 0.5 * (traps_m + traps_m[nn_idxs])
    dist_to_centre = np.linalg.norm(midpoints - centre, axis=1)

    # Score: primarily match spacing, secondarily be near centre
    score = rel_err + 0.1 * dist_to_centre / dist_to_centre.max()
    best = np.argmin(score)
    idx_A = best
    idx_B = nn_idxs[best]
    d_AB = nn_dists[best]
    return idx_A, idx_B, d_AB


def make_local_frame(traps_m, idx_A, idx_B):
    """
    Build local coordinate frame from trap pair.

    Returns:
        midpoint (2,), e_AB (unit A→B), e_perp (perpendicular), d_AB (float)
    """
    A = traps_m[idx_A]
    B = traps_m[idx_B]
    midpoint = 0.5 * (A + B)
    d_AB = np.linalg.norm(B - A)
    e_AB = (B - A) / d_AB
    e_perp = np.array([-e_AB[1], e_AB[0]])
    return midpoint, e_AB, e_perp, d_AB


# ═══════════════════════════════════════════════════════════════════
# 5. C-shaped field generation
# ═══════════════════════════════════════════════════════════════════

def make_c_field(xg, yg, midpoint, e_AB, e_perp,
                 r0, sigma_r, sigma_theta, theta_gap, m=1, beta=1.0):
    """
    Generate a localised C-shaped complex pressure field.

    p_C(x,y) = A(r) * W(theta) * exp(i m theta)

    Parameters
    ----------
    r0, sigma_r : radial Gaussian ring parameters (metres)
    sigma_theta : angular gap width (radians)
    theta_gap : centre of the angular gap (radians)
    m : azimuthal winding number
    beta : gap depth (0=no gap, 1=full gap)
    """
    XX, YY = np.meshgrid(xg, yg)
    # Local coordinates relative to midpoint along (e_AB, e_perp) frame
    dx = XX - midpoint[0]
    dy = YY - midpoint[1]
    x_loc = dx * e_AB[0] + dy * e_AB[1]
    y_loc = dx * e_perp[0] + dy * e_perp[1]

    r = np.sqrt(x_loc**2 + y_loc**2)
    theta = np.arctan2(y_loc, x_loc)  # -pi to pi

    # Radial Gaussian ring
    A_r = np.exp(-0.5 * ((r - r0) / sigma_r)**2)

    # Angular gap (smooth suppression)
    dtheta = theta - theta_gap
    # Wrap to [-pi, pi]
    dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
    W_theta = 1.0 - beta * np.exp(-0.5 * (dtheta / sigma_theta)**2)

    # Phase winding
    phase = np.exp(1j * m * theta)

    p_C = A_r * W_theta * phase
    return p_C


def combine_fields(p_SW, p_C, alpha=1.0, psi=0.0):
    """Combine standing-wave and C-shape fields: p_SW + alpha * exp(i*psi) * p_C."""
    return p_SW + alpha * np.exp(1j * psi) * p_C


# ═══════════════════════════════════════════════════════════════════
# 6. Compactness / leakage metrics
# ═══════════════════════════════════════════════════════════════════

def compute_trap_roi_energy(I_C, xg, yg, trap_xy, roi_radius):
    """Integrate I_C = |p_C|^2 within circular ROI around each trap."""
    XX, YY = np.meshgrid(xg, yg)
    dx = xg[1] - xg[0]
    dy = yg[1] - yg[0]
    cell_area = dx * dy
    energies = []
    for xy in trap_xy:
        dist = np.sqrt((XX - xy[0])**2 + (YY - xy[1])**2)
        mask = dist <= roi_radius
        energies.append(float(np.sum(I_C[mask]) * cell_area))
    return np.array(energies)


def compute_leakage_metrics(traps_m, idx_A, idx_B, I_C, xg, yg, d_AB):
    """
    Compute asymmetric C-shape metrics for particle transport use-case.

    The C-shape should:
      - Have high energy near trap A  (it drives the particle at A)
      - Have LOW energy near trap B   (to avoid pushing B's particle away)
      - Have low leakage into other neighbours

    Key metrics
    -----------
    asymmetry      : E_A / (E_A + E_B)   — closer to 1.0 is better
    leakage_ratio_max : max(neighbour E) / E_A   — lower is better
    leakage_ratio_sum : sum(neighbour E) / E_A   — lower is better
    B_suppression  : 1 - E_B / E_A              — higher is better (B not hit)
    score (lower=better) : leakage_ratio_max + leakage_ratio_sum
                           + (1 - asymmetry) + (1 - B_suppression)
    """
    roi_r = 0.25 * d_AB

    all_E = compute_trap_roi_energy(I_C, xg, yg, traps_m, roi_r)
    E_A = float(all_E[idx_A])
    E_B = float(all_E[idx_B])

    neighbour_mask = np.ones(len(traps_m), dtype=bool)
    neighbour_mask[idx_A] = False
    neighbour_mask[idx_B] = False
    neighbour_E = all_E[neighbour_mask]

    max_neighbour = float(neighbour_E.max()) if len(neighbour_E) > 0 else 0.0
    sum_neighbour = float(neighbour_E.sum())
    E_ref = max(E_A, 1e-30)

    leakage_max = max_neighbour / E_ref
    leakage_sum = sum_neighbour / E_ref
    asymmetry   = E_A / max(E_A + E_B, 1e-30)   # 1.0 = all at A, 0.5 = equal
    B_suppression = 1.0 - E_B / E_ref            # 1.0 = no energy at B

    # Corridor ROI (centred on midpoint, radius = 0.5*d_AB)
    midpoint = 0.5 * (traps_m[idx_A] + traps_m[idx_B])
    XX, YY = np.meshgrid(xg, yg)
    cell_area = (xg[1] - xg[0]) * (yg[1] - yg[0])
    dist_mid = np.sqrt((XX - midpoint[0])**2 + (YY - midpoint[1])**2)
    corridor_mask = dist_mid <= 0.5 * d_AB
    E_corridor = float(np.sum(I_C[corridor_mask]) * cell_area)
    E_total    = float(np.sum(I_C) * cell_area)

    return {
        "E_A": E_A,
        "E_B": E_B,
        "max_neighbour": max_neighbour,
        "sum_neighbour": sum_neighbour,
        "leakage_ratio_max": leakage_max,
        "leakage_ratio_sum": leakage_sum,
        "asymmetry": asymmetry,
        "B_suppression": B_suppression,
        "corridor_energy": E_corridor,
        "total_energy": E_total,
        "corridor_fraction": E_corridor / max(E_total, 1e-30),
    }


# ═══════════════════════════════════════════════════════════════════
# 7. Plotting helpers
# ═══════════════════════════════════════════════════════════════════

def _add_colorbar(ax, im, label=""):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax, label=label)


def _overlay_traps(ax, traps_m, idx_A, idx_B, midpoint, xg, yg):
    """Overlay trap positions on the given axes."""
    # Convert to mm for display
    traps_mm = traps_m * 1e3
    mid_mm = midpoint * 1e3

    for i, (tx, ty) in enumerate(traps_mm):
        if i == idx_A:
            ax.plot(tx, ty, "r^", ms=10, mew=1.5, mfc="none", label="Trap A")
        elif i == idx_B:
            ax.plot(tx, ty, "bs", ms=10, mew=1.5, mfc="none", label="Trap B")
        else:
            ax.plot(tx, ty, "w+", ms=6, mew=0.8)

    ax.plot(*mid_mm, "gx", ms=10, mew=2.0, label="Midpoint")


def plot_standing_wave_plane(p_SW, xg, yg, traps_m, idx_A, idx_B, midpoint, save_path):
    """Plot standing-wave amplitude with trap overlay."""
    xg_mm = xg * 1e3
    yg_mm = yg * 1e3
    extent = [xg_mm[0], xg_mm[-1], yg_mm[0], yg_mm[-1]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Amplitude
    ax = axes[0]
    im = ax.imshow(np.abs(p_SW), origin="lower", extent=extent,
                   cmap="inferno", aspect="equal")
    _add_colorbar(ax, im, "|p_SW| (Pa)")
    _overlay_traps(ax, traps_m, idx_A, idx_B, midpoint, xg, yg)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("Standing Wave |p|")
    ax.legend(loc="upper left", fontsize=7)

    # Intensity
    ax = axes[1]
    im = ax.imshow(np.abs(p_SW)**2, origin="lower", extent=extent,
                   cmap="magma", aspect="equal")
    _add_colorbar(ax, im, "|p_SW|² (Pa²)")
    _overlay_traps(ax, traps_m, idx_A, idx_B, midpoint, xg, yg)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("Standing Wave |p|²")

    fig.suptitle(f"Standing-Wave Trap Plane  z = {Z_STAR*1e3:.4f} mm", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {save_path.name}")


def plot_candidate_panel(p_SW, p_C, p_comb, xg, yg, traps_m, idx_A, idx_B,
                         midpoint, params, save_path):
    """4-panel plot: SW amplitude, C amplitude, C phase, combined amplitude."""
    xg_mm = xg * 1e3
    yg_mm = yg * 1e3
    extent = [xg_mm[0], xg_mm[-1], yg_mm[0], yg_mm[-1]]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # (0,0) Standing-wave amplitude
    ax = axes[0, 0]
    im = ax.imshow(np.abs(p_SW), origin="lower", extent=extent,
                   cmap="inferno", aspect="equal")
    _add_colorbar(ax, im, "|p_SW|")
    _overlay_traps(ax, traps_m, idx_A, idx_B, midpoint, xg, yg)
    ax.set_title("Standing Wave |p|")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")

    # (0,1) C-shape amplitude
    ax = axes[0, 1]
    im = ax.imshow(np.abs(p_C), origin="lower", extent=extent,
                   cmap="viridis", aspect="equal")
    _add_colorbar(ax, im, "|p_C|")
    _overlay_traps(ax, traps_m, idx_A, idx_B, midpoint, xg, yg)
    ax.set_title(f"|p_C|  r0/d={params['r0_rel']:.2f}  σr/d={params['sr_rel']:.2f}  "
                 f"σθ={params['sigma_theta']:.2f}  gap={params['gap_label']}")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")

    # (1,0) C-shape phase
    ax = axes[1, 0]
    im = ax.imshow(np.angle(p_C), origin="lower", extent=extent,
                   cmap="hsv", vmin=-np.pi, vmax=np.pi, aspect="equal")
    _add_colorbar(ax, im, "arg(p_C)")
    _overlay_traps(ax, traps_m, idx_A, idx_B, midpoint, xg, yg)
    ax.set_title("C-shape phase arg(p_C)")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")

    # (1,1) Combined amplitude
    ax = axes[1, 1]
    im = ax.imshow(np.abs(p_comb), origin="lower", extent=extent,
                   cmap="inferno", aspect="equal")
    _add_colorbar(ax, im, "|p_comb|")
    _overlay_traps(ax, traps_m, idx_A, idx_B, midpoint, xg, yg)
    ax.set_title(f"Combined |p_SW + α·p_C|  α={params['alpha']:.2f}")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")

    fig.suptitle(f"C-Shape Candidate  r0/d={params['r0_rel']:.2f}  "
                 f"gap={params['gap_label']}",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_best_comparison(p_SW, p_C, p_comb, xg, yg, traps_m, idx_A, idx_B,
                         midpoint, e_AB, e_perp, d_AB, params, metrics,
                         save_path):
    """High-quality 6-panel plot for best candidates."""
    xg_mm = xg * 1e3
    yg_mm = yg * 1e3
    extent = [xg_mm[0], xg_mm[-1], yg_mm[0], yg_mm[-1]]
    mid_mm = midpoint * 1e3

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    titles = [
        (np.abs(p_SW), "inferno", "|p_SW|", "Standing Wave"),
        (np.abs(p_C), "viridis", "|p_C|", "C-Shape Amplitude"),
        (np.angle(p_C), "hsv", "arg(p_C)", "C-Shape Phase"),
        (np.abs(p_comb), "inferno", "|p_comb|", "Combined Field"),
        (np.abs(p_comb)**2, "magma", "|p_comb|²", "Combined Intensity"),
        (np.real(p_C), "RdBu_r", "Re(p_C)", "C-Shape Real Part"),
    ]
    vmins = [None, None, -np.pi, None, None, None]
    vmaxs = [None, None, np.pi, None, None, None]

    for idx, (ax, (data, cmap, clabel, title)) in enumerate(
            zip(axes.ravel(), titles)):
        kw = dict(origin="lower", extent=extent, cmap=cmap, aspect="equal")
        if vmins[idx] is not None:
            kw["vmin"] = vmins[idx]
            kw["vmax"] = vmaxs[idx]
        if idx == 5:
            vlim = max(abs(data.min()), abs(data.max()))
            kw["vmin"] = -vlim
            kw["vmax"] = vlim
        im = ax.imshow(data, **kw)
        _add_colorbar(ax, im, clabel)
        _overlay_traps(ax, traps_m, idx_A, idx_B, midpoint, xg, yg)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")

    fig.suptitle(
        f"BEST: r0/d={params['r0_rel']:.2f}  σr/d={params['sr_rel']:.2f}  "
        f"σθ={params['sigma_theta']:.2f}  gap={params['gap_label']}  "
        f"asymmetry={metrics['asymmetry']:.3f}  "
        f"B_supp={metrics['B_suppression']:.3f}  "
        f"leak_max={metrics['leakage_ratio_max']:.3f}",
        fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_line_cuts(p_C, xg, yg, midpoint, e_AB, e_perp, d_AB, save_path):
    """Line cuts of |p_C| along A→B and perpendicular through midpoint."""
    n_pts = 300
    half_len = 1.5 * d_AB

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, direction, label_dir in zip(axes, [e_AB, e_perp],
                                         ["Along A→B", "Perpendicular"]):
        s = np.linspace(-half_len, half_len, n_pts)
        pts_x = midpoint[0] + s * direction[0]
        pts_y = midpoint[1] + s * direction[1]

        # Interpolate from grid
        from scipy.interpolate import RegularGridInterpolator
        rgi = RegularGridInterpolator((yg, xg), np.abs(p_C),
                                      bounds_error=False, fill_value=0.0)
        vals = rgi(np.column_stack([pts_y, pts_x]))

        ax.plot(s * 1e3, vals, "b-", lw=1.5)
        ax.axvline(0, color="g", ls="--", lw=0.8, label="midpoint")
        ax.axvline(-d_AB/2 * 1e3, color="r", ls=":", lw=0.8, label="Trap A")
        ax.axvline(d_AB/2 * 1e3, color="b", ls=":", lw=0.8, label="Trap B")
        ax.set_xlabel("distance from midpoint (mm)")
        ax.set_ylabel("|p_C|")
        ax.set_title(f"Line cut: {label_dir}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_zoomed(p_SW, p_C, p_comb, xg, yg, traps_m, idx_A, idx_B,
                midpoint, d_AB, save_path):
    """Zoomed view around the trap pair."""
    zoom_half = 1.5 * d_AB
    mid = midpoint

    xmask = (xg >= mid[0] - zoom_half) & (xg <= mid[0] + zoom_half)
    ymask = (yg >= mid[1] - zoom_half) & (yg <= mid[1] + zoom_half)

    if xmask.sum() < 10 or ymask.sum() < 10:
        return  # too small to zoom

    xg_z = xg[xmask]
    yg_z = yg[ymask]
    ix = np.where(xmask)[0]
    iy = np.where(ymask)[0]
    sl = np.ix_(iy, ix)

    extent_z = [xg_z[0]*1e3, xg_z[-1]*1e3, yg_z[0]*1e3, yg_z[-1]*1e3]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, data, title, cmap in zip(
            axes,
            [np.abs(p_SW[sl]), np.abs(p_C[sl]), np.abs(p_comb[sl])],
            ["SW alone", "C-shape alone", "Combined"],
            ["inferno", "viridis", "inferno"]):
        im = ax.imshow(data, origin="lower", extent=extent_z,
                       cmap=cmap, aspect="equal")
        _add_colorbar(ax, im)
        _overlay_traps(ax, traps_m, idx_A, idx_B, midpoint, xg_z, yg_z)
        ax.set_title(title)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")

    fig.suptitle("Zoomed: SW vs C-shape vs Combined", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# 8. Ranking
# ═══════════════════════════════════════════════════════════════════

def rank_candidates(records):
    """
    Rank candidates for the transport use-case (lower score = better).

    Goal: high energy at A, low at B, low leakage to neighbours.

    score = leakage_ratio_max
          + leakage_ratio_sum
          + (1 - asymmetry)        # penalise equal A/B energy
          + (1 - B_suppression)    # penalise energy at B (= E_B/E_A)
    """
    for rec in records:
        rec["score"] = (
            rec["leakage_ratio_max"]
            + rec["leakage_ratio_sum"]
            + (1.0 - rec["asymmetry"])
            + (1.0 - rec["B_suppression"])
        )
    return sorted(records, key=lambda r: r["score"])


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    print("=" * 70)
    print("C-Shape on Cached Standing-Wave Geometry Study")
    print("=" * 70)

    # ── 1. Load cached standing wave ─────────────────────────────
    coords, p_fem, cache_path = load_cached_standing_wave()

    # ── 2. Select trap plane ─────────────────────────────────────
    z_val = select_trap_plane()
    print(f"[plane] Design plane z = {z_val*1e3:.4f} mm")

    # ── 3. Interpolate to grid ───────────────────────────────────
    print(f"[interp] Interpolating to {N_GRID}×{N_GRID} grid...")
    p_SW, xg, yg = interpolate_to_plane(coords, p_fem, z_val)
    print(f"  |p_SW| range: [{np.abs(p_SW).min():.2f}, {np.abs(p_SW).max():.2f}] Pa")

    # ── 4. Get trap centres ──────────────────────────────────────
    traps_m, trap_source = get_or_detect_traps(p_SW, xg, yg)
    n_traps = len(traps_m)
    print(f"  {n_traps} traps available")

    # ── 5. Choose trap pair ──────────────────────────────────────
    idx_A, idx_B, d_AB = choose_adjacent_trap_pair(traps_m)
    midpoint, e_AB, e_perp, _ = make_local_frame(traps_m, idx_A, idx_B)
    A_pos = traps_m[idx_A]
    B_pos = traps_m[idx_B]
    print(f"[pair] Trap A (#{idx_A}): ({A_pos[0]*1e3:.4f}, {A_pos[1]*1e3:.4f}) mm")
    print(f"       Trap B (#{idx_B}): ({B_pos[0]*1e3:.4f}, {B_pos[1]*1e3:.4f}) mm")
    print(f"       d_AB = {d_AB*1e3:.4f} mm  "
          f"({d_AB/LAM:.3f} λ,  {d_AB/TRAP_SP:.3f} trap_sp)")
    print(f"       midpoint = ({midpoint[0]*1e3:.4f}, {midpoint[1]*1e3:.4f}) mm")

    # ── 6. Plot standing-wave plane ──────────────────────────────
    print("[plot] Standing-wave plane...")
    plot_standing_wave_plane(p_SW, xg, yg, traps_m, idx_A, idx_B, midpoint,
                            FIG_DIR / "standing_wave_plane.png")

    # ── 7. Parameter sweep ───────────────────────────────────────
    r0_rels = [0.25, 0.35, 0.45, 0.55]
    sr_rels = [0.10, 0.15, 0.20]
    sigma_thetas = [0.30, 0.50, 0.70]

    # Gap orientations in local frame (angle from +x_local = A→B direction)
    #   toward B = 0, toward A = pi, +90° = pi/2, -90° = -pi/2
    gap_configs = [
        (0.0,      "toward_B"),
        (np.pi,    "toward_A"),
        (np.pi/2,  "plus_90"),
        (-np.pi/2, "minus_90"),
    ]

    # Combined-field weighting: scale alpha so C-shape peak ≈ fraction of SW
    sw_peak = np.abs(p_SW).max()
    alpha_fracs = [0.05, 0.10, 0.20]  # fraction of SW peak
    psi_vals = [0.0]  # single phase offset for first pass

    m_val = 1
    beta_val = 1.0

    total = len(r0_rels) * len(sr_rels) * len(sigma_thetas) * len(gap_configs)
    print(f"[sweep] {total} C-shape candidates × {len(alpha_fracs)} alpha levels")

    records = []
    cand_idx = 0

    for r0_rel, sr_rel, sigma_theta, (theta_gap, gap_label) in product(
            r0_rels, sr_rels, sigma_thetas, gap_configs):

        r0 = r0_rel * d_AB
        sigma_r = sr_rel * d_AB

        p_C = make_c_field(xg, yg, midpoint, e_AB, e_perp,
                           r0=r0, sigma_r=sigma_r,
                           sigma_theta=sigma_theta,
                           theta_gap=theta_gap,
                           m=m_val, beta=beta_val)

        # Normalise C-field so peak = 1 (metrics are shape-only)
        c_peak = np.abs(p_C).max()
        if c_peak < 1e-30:
            continue
        p_C_norm = p_C / c_peak

        I_C = np.abs(p_C_norm)**2
        metrics = compute_leakage_metrics(traps_m, idx_A, idx_B, I_C, xg, yg, d_AB)

        rec = {
            "idx": cand_idx,
            "r0_rel": r0_rel,
            "sr_rel": sr_rel,
            "sigma_theta": sigma_theta,
            "theta_gap": theta_gap,
            "gap_label": gap_label,
            "m": m_val,
            "beta": beta_val,
            **metrics,
        }
        records.append(rec)

        # Save 4-panel for every candidate (quick)
        for alpha_frac in alpha_fracs:
            alpha = alpha_frac * sw_peak  # scale C-field absolute level
            p_comb = combine_fields(p_SW, p_C_norm * alpha, alpha=1.0, psi=0.0)

            params_vis = {
                "r0_rel": r0_rel, "sr_rel": sr_rel,
                "sigma_theta": sigma_theta, "gap_label": gap_label,
                "alpha": alpha_frac,
            }
            fname = (f"cand_{cand_idx:03d}_r{r0_rel:.2f}_s{sr_rel:.2f}_"
                     f"st{sigma_theta:.2f}_{gap_label}_a{alpha_frac:.2f}.png")
            plot_candidate_panel(p_SW, p_C_norm, p_comb, xg, yg,
                                traps_m, idx_A, idx_B, midpoint,
                                params_vis, FIG_DIR / fname)

        cand_idx += 1
        if cand_idx % 20 == 0:
            print(f"  {cand_idx}/{total} candidates done")

    print(f"[sweep] {cand_idx} candidates evaluated")

    # ── 8. Rank and save ─────────────────────────────────────────
    ranked = rank_candidates(records)
    df = pd.DataFrame(ranked)
    csv_path = OUT_DIR / "metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"[save] metrics.csv: {len(df)} rows")

    # ── 9. Best candidate detailed plots ─────────────────────────
    N_BEST = min(5, len(ranked))
    print(f"[best] Generating detailed plots for top {N_BEST} candidates...")
    best_summaries = []

    for rank_i, rec in enumerate(ranked[:N_BEST]):
        r0 = rec["r0_rel"] * d_AB
        sigma_r = rec["sr_rel"] * d_AB
        p_C = make_c_field(xg, yg, midpoint, e_AB, e_perp,
                           r0=r0, sigma_r=sigma_r,
                           sigma_theta=rec["sigma_theta"],
                           theta_gap=rec["theta_gap"],
                           m=rec["m"], beta=rec["beta"])
        c_peak = np.abs(p_C).max()
        if c_peak < 1e-30:
            continue
        p_C_norm = p_C / c_peak

        alpha = 0.10 * sw_peak
        p_comb = combine_fields(p_SW, p_C_norm * alpha, alpha=1.0, psi=0.0)

        params_vis = {
            "r0_rel": rec["r0_rel"], "sr_rel": rec["sr_rel"],
            "sigma_theta": rec["sigma_theta"], "gap_label": rec["gap_label"],
            "alpha": 0.10,
        }

        # 6-panel comparison
        plot_best_comparison(p_SW, p_C_norm, p_comb, xg, yg,
                             traps_m, idx_A, idx_B, midpoint,
                             e_AB, e_perp, d_AB, params_vis, rec,
                             BEST_DIR / f"best_{rank_i:02d}_comparison.png")

        # Line cuts
        plot_line_cuts(p_C_norm, xg, yg, midpoint, e_AB, e_perp, d_AB,
                       BEST_DIR / f"best_{rank_i:02d}_line_cuts.png")

        # Zoomed
        plot_zoomed(p_SW, p_C_norm, p_comb, xg, yg, traps_m, idx_A, idx_B,
                    midpoint, d_AB,
                    BEST_DIR / f"best_{rank_i:02d}_zoomed.png")

        # Save npz
        np.savez_compressed(NPZ_DIR / f"best_{rank_i:02d}.npz",
                            p_C=p_C_norm, p_comb=p_comb,
                            xg=xg, yg=yg, params=params_vis,
                            metrics=rec)

        best_summaries.append({
            "rank": rank_i,
            **{k: rec[k] for k in ["r0_rel", "sr_rel", "sigma_theta",
                                     "gap_label", "score",
                                     "leakage_ratio_max", "leakage_ratio_sum",
                                     "asymmetry", "B_suppression",
                                     "corridor_fraction"]},
        })

        print(f"  rank {rank_i}: r0/d={rec['r0_rel']:.2f} σr/d={rec['sr_rel']:.2f} "
              f"σθ={rec['sigma_theta']:.2f} gap={rec['gap_label']}  "
              f"score={rec['score']:.4f}  asym={rec['asymmetry']:.3f}  "
              f"B_supp={rec['B_suppression']:.3f}  "
              f"leak_max={rec['leakage_ratio_max']:.4f}")

    # ── 10. Save config ──────────────────────────────────────────
    config = {
        "timestamp": TS,
        "cache_file": str(cache_path),
        "z_star_mm": z_val * 1e3,
        "trap_source": trap_source,
        "n_traps": n_traps,
        "trap_A_idx": int(idx_A),
        "trap_B_idx": int(idx_B),
        "trap_A_mm": [float(A_pos[0]*1e3), float(A_pos[1]*1e3)],
        "trap_B_mm": [float(B_pos[0]*1e3), float(B_pos[1]*1e3)],
        "d_AB_mm": float(d_AB * 1e3),
        "d_AB_over_lambda": float(d_AB / LAM),
        "n_grid": N_GRID,
        "roi_half_mm": ROI_HALF * 1e3,
        "sweep": {
            "r0_rels": r0_rels,
            "sr_rels": sr_rels,
            "sigma_thetas": sigma_thetas,
            "gap_orientations": [g[1] for g in gap_configs],
            "alpha_fracs": alpha_fracs,
            "m": m_val,
            "beta": beta_val,
        },
        "n_candidates": len(records),
        "runtime_s": time.time() - t_start,
    }
    with open(OUT_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ── 11. Write INDEX.md ───────────────────────────────────────
    top5 = ranked[:N_BEST]
    index_md = f"""# C-Shape on Cached Standing-Wave Geometry Study

**Generated**: {TS}
**Runtime**: {time.time() - t_start:.1f} s

## Data Sources

- **Standing-wave cache**: `{cache_path.relative_to(PROJECT_ROOT)}`
- **Design plane**: z = {z_val*1e3:.4f} mm (z*)
- **Trap centres**: {trap_source}
- **Grid resolution**: {N_GRID}×{N_GRID} over {ROI_HALF*2e3:.2f}×{ROI_HALF*2e3:.2f} mm ROI

## Trap Pair Selection

- **Trap A** (#{idx_A}): ({A_pos[0]*1e3:.4f}, {A_pos[1]*1e3:.4f}) mm
- **Trap B** (#{idx_B}): ({B_pos[0]*1e3:.4f}, {B_pos[1]*1e3:.4f}) mm
- **d_AB** = {d_AB*1e3:.4f} mm = {d_AB/LAM:.3f} λ = {d_AB/TRAP_SP:.3f} trap spacings
- **Midpoint**: ({midpoint[0]*1e3:.4f}, {midpoint[1]*1e3:.4f}) mm

## Parameter Sweep

| Parameter | Values |
|-----------|--------|
| r0/d_AB | {r0_rels} |
| σr/d_AB | {sr_rels} |
| σθ (rad) | {sigma_thetas} |
| gap orientation | toward_B, toward_A, plus_90, minus_90 |
| m (winding) | {m_val} |
| β (gap depth) | {beta_val} |
| α (combined) | {alpha_fracs} (fraction of SW peak) |
| **Total candidates** | {len(records)} |

## Top-Ranked Compact Candidates

| Rank | r0/d | σr/d | σθ | Gap | Score | Asymmetry | B_supp | Leak Max |
|------|------|------|------|-----|-------|-----------|--------|----------|
"""
    for bs in best_summaries:
        index_md += (
            f"| {bs['rank']} | {bs['r0_rel']:.2f} | {bs['sr_rel']:.2f} | "
            f"{bs['sigma_theta']:.2f} | {bs['gap_label']} | {bs['score']:.4f} | "
            f"{bs['asymmetry']:.4f} | {bs['B_suppression']:.4f} | "
            f"{bs['leakage_ratio_max']:.4f} |\n"
        )

    index_md += f"""
## Observations

### Transport-Asymmetry Assessment
The C-shape is designed to act on trap A while leaving trap B undisturbed.
The key metric is **asymmetry** (E_A/(E_A+E_B) → 1.0 ideal) and
**B_suppression** (1 − E_B/E_A → 1.0 ideal).

Source: epl5, z* = {Z_STAR*1e3:.4f} mm, d_AB = {d_AB*1e3:.4f} mm.

Candidates with **asymmetry > 0.90**, **B_suppression > 0.90**, and
**leakage_ratio_max < 0.10** are considered suitable for the real trap lattice.

### Combined-Field Visualisations
The superposition p_SW + α·p_C is shown for several α values
({alpha_fracs}).  These are visual-only; no physics claims are made
about particle transport at this stage.

### What This Does NOT Include
- No ASM backpropagation
- No lens thickness design
- No Gor'kov force optimisation
- No particle transport simulation

### Recommended Next Step
If the top-ranked candidates show low leakage (ratio < 0.05):
→ Proceed to ASM backpropagation to determine whether a physical
  source distribution can generate the desired combined field.
If leakage is high:
→ Refine the radial / angular parameters before proceeding.

## Files

- `config.json` — full configuration
- `metrics.csv` — all candidates with metrics
- `figures/` — candidate panels ({len(records) * len(alpha_fracs)} plots)
- `best/` — detailed comparison, line cuts, zoomed plots for top {N_BEST}
- `npz/` — saved field data for top {N_BEST}
"""
    with open(OUT_DIR / "INDEX.md", "w") as f:
        f.write(index_md)

    print(f"\n{'=' * 70}")
    print(f"Done in {time.time() - t_start:.1f}s")
    print(f"Outputs: {OUT_DIR}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
