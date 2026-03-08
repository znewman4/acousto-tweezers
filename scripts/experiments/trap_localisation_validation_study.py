#!/usr/bin/env python3
"""
Trap Localisation Validation Study
====================================

Rigorous validation of trap detection: verifies that detected traps correspond
to real stable minima of the Gor'kov potential, not guessed lattice centres
or visual blob centres.

Produces diagnostics to determine exactly what is wrong with the current
grid-only, threshold-filtered detection used in bridge studies.

Outputs → results/trap_localisation_validation_YYYYMMDD_HHMMSS/

Usage:
    python scripts/experiments/trap_localisation_validation_study.py
"""
from __future__ import annotations

import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import minimum_filter, maximum_filter, label
from scipy.optimize import minimize
from scipy.spatial import cKDTree

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# ── Project root ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ═══════════════════════════════════════════════════════════════════
# Physical constants (MUST match bridge_master_study.py)
# ═══════════════════════════════════════════════════════════════════
C_WATER = 1484.0
F_HZ    = 2.0e6
K_WATER = 2.0 * np.pi * F_HZ / C_WATER
LAM     = C_WATER / F_HZ          # 0.742 mm
TRAP_SP = LAM / 2.0               # ~0.371 mm
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

DZ_GRAD  = LAM / 15.0
K_IDW    = 16

ROI_HALF = 1.1 * LAM   # default ROI half-width
STANDING_CACHE_DIR = PROJECT_ROOT / "results" / "fem_standing_wave_cache"

# ═══════════════════════════════════════════════════════════════════
# Resolution configurations for convergence study
# ═══════════════════════════════════════════════════════════════════
RESOLUTION_CONFIGS = {
    "coarse":  200,
    "medium":  400,
    "fine":    800,
}

# ═══════════════════════════════════════════════════════════════════
# Utility functions
# ═══════════════════════════════════════════════════════════════════

def log(msg="", fp=None):
    print(msg)
    if fp:
        fp.write(msg + "\n")
        fp.flush()


def load_fem_cache(path: Path):
    """Load cached FEM pressure solution."""
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


def sample_idw(tree, p, pts, k=16, power=2.0, eps=1e-12):
    """Inverse-distance-weighted interpolation from FEM DOFs to query points."""
    if k == 1:
        d, i = tree.query(pts, k=1)
        return p[i]
    d, i = tree.query(pts, k=k)
    w = 1.0 / (d**power + eps)
    w /= w.sum(axis=1, keepdims=True)
    return (p[i] * w).sum(axis=1)


def gorkov_potential_from_fields(p_abs2, grad_p_mag2):
    """Compute Gor'kov potential from |p|^2 and |grad p|^2."""
    E_pot = 0.25 * p_abs2 * KAPPA0
    E_kin = grad_p_mag2 / (4.0 * OMEGA**2 * RHO0)
    return V_P * (F1 * E_pot - 1.5 * F2 * E_kin)


def gorkov_and_force_on_grid(p_grid, dx, dy, p_zp=None, p_zm=None, dz=None):
    """
    Compute Gor'kov potential and radiation force on a 2D grid.

    Returns U, Fx, Fy, grad_p_mag2
    """
    p_abs2 = np.abs(p_grid)**2

    # Proper gradient: v = -(1/(i w rho)) grad(p)
    dpy, dpx = np.gradient(p_grid, dy, dx)
    grad_p_mag2 = np.abs(dpx)**2 + np.abs(dpy)**2

    if p_zp is not None and p_zm is not None and dz is not None:
        dpz = (p_zp - p_zm) / (2.0 * dz)
        grad_p_mag2 += np.abs(dpz)**2

    U = gorkov_potential_from_fields(p_abs2, grad_p_mag2)
    Fx = -np.gradient(U, dx, axis=1)
    Fy = -np.gradient(U, dy, axis=0)
    return U, Fx, Fy, grad_p_mag2


def interpolate_fem_to_grid(tree, p_fem, xg, yg, z_val, coords_3d=None):
    """Interpolate FEM pressure onto a 2D Cartesian grid at z=z_val."""
    XX, YY = np.meshgrid(xg, yg)
    pts_3d = np.column_stack([XX.ravel(), YY.ravel(),
                              np.full(XX.size, z_val)])
    p_flat = sample_idw(tree, p_fem, pts_3d, k=K_IDW)
    return p_flat.reshape(XX.shape), XX, YY


# ═══════════════════════════════════════════════════════════════════
# OLD trap detection (reproduced exactly from bridge_master_study.py)
# ═══════════════════════════════════════════════════════════════════

def find_trap_minima_OLD(U, min_sep_pix=10, threshold_frac=0.3):
    """Old method: local maxima of -U with threshold filter."""
    neg_U = -U
    local_max = (neg_U == maximum_filter(neg_U, size=min_sep_pix))
    local_max &= (neg_U > threshold_frac * neg_U.max())
    iy, ix = np.where(local_max)
    return iy, ix


# ═══════════════════════════════════════════════════════════════════
# NEW trap detection and validation pipeline
# ═══════════════════════════════════════════════════════════════════

def detect_candidates(U, xg, yg, min_sep_physical=None):
    """
    Detect all local minima of U using morphological minimum filter.

    min_sep_physical: minimum separation in metres. If None, defaults to
        0.4 * TRAP_SP so that separate traps at λ/2 spacing are resolved
        but sub-λ/2 ripples are suppressed.
    """
    dx = xg[1] - xg[0]
    if min_sep_physical is None:
        min_sep_physical = 0.4 * TRAP_SP
    min_distance_pts = max(3, int(min_sep_physical / dx))
    neighborhood = 2 * min_distance_pts + 1
    local_min = minimum_filter(U, size=neighborhood)
    minima_mask = (U == local_min)
    # Exclude boundaries (2-pixel border to avoid edge artefacts)
    minima_mask[:2, :] = False
    minima_mask[-2:, :] = False
    minima_mask[:, :2] = False
    minima_mask[:, -2:] = False
    iy, ix = np.where(minima_mask)
    return iy, ix


def refine_subgrid(U, xg, yg, iy0, ix0, half_win=3):
    """
    Refine a grid-detected minimum to sub-grid accuracy via local
    quadratic (paraboloid) fit.

    Fits U(x,y) ≈ a + b·x' + c·y' + d·x'² + e·y'² + f·x'·y'
    to a (2*half_win+1)² neighbourhood, then finds the minimum of
    the fitted paraboloid.

    Returns (x_refined, y_refined, U_fitted_min, fit_residual, eigenvalues_of_hessian).
    Returns None if fit fails or refined point is outside the window.
    """
    ny, nx = U.shape
    # Extract local window
    iy_lo = max(iy0 - half_win, 0)
    iy_hi = min(iy0 + half_win + 1, ny)
    ix_lo = max(ix0 - half_win, 0)
    ix_hi = min(ix0 + half_win + 1, nx)

    x_local = xg[ix_lo:ix_hi]
    y_local = yg[iy_lo:iy_hi]
    U_local = U[iy_lo:iy_hi, ix_lo:ix_hi]

    if U_local.size < 6:
        return None  # not enough points

    x_cen = xg[ix0]
    y_cen = yg[iy0]

    XL, YL = np.meshgrid(x_local - x_cen, y_local - y_cen)
    xl = XL.ravel()
    yl = YL.ravel()
    ul = U_local.ravel()

    # Fit: U = a + b*x + c*y + d*x^2 + e*y^2 + f*x*y
    A = np.column_stack([np.ones_like(xl), xl, yl, xl**2, yl**2, xl * yl])
    try:
        coeffs, residuals, rank, sv = np.linalg.lstsq(A, ul, rcond=None)
    except np.linalg.LinAlgError:
        return None

    a, b, c, d, e, f = coeffs

    # Hessian of the paraboloid: H = [[2d, f], [f, 2e]]
    H = np.array([[2 * d, f], [f, 2 * e]])
    eig_vals = np.linalg.eigvalsh(H)

    # For a true minimum, both eigenvalues must be positive
    if eig_vals[0] <= 0:
        # Not a minimum in fitted model — return info anyway for diagnostics
        grad = np.array([b, c])
        # best-effort minimum from gradient = 0 if Hessian is non-singular
        try:
            shift = -np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            return None
        x_ref = x_cen + shift[0]
        y_ref = y_cen + shift[1]
        # Check if refined point is within the local window
        dx_half = (x_local[-1] - x_local[0]) / 2
        dy_half = (y_local[-1] - y_local[0]) / 2
        if abs(shift[0]) > dx_half or abs(shift[1]) > dy_half:
            return None  # shifted outside window
        U_fit = float(coeffs @ np.array([1, shift[0], shift[1],
                                          shift[0]**2, shift[1]**2,
                                          shift[0] * shift[1]]))
        fit_res = float(np.sqrt(np.mean((A @ coeffs - ul)**2)))
        return (x_ref, y_ref, U_fit, fit_res, eig_vals, False)

    # Minimum of paraboloid: solve grad = 0 → [b + 2d*x + f*y, c + f*x + 2e*y] = 0
    grad = np.array([b, c])
    try:
        shift = -np.linalg.solve(H, grad)
    except np.linalg.LinAlgError:
        return None

    x_ref = x_cen + shift[0]
    y_ref = y_cen + shift[1]

    # Check if refined point is within the local window
    dx_half = (x_local[-1] - x_local[0]) / 2
    dy_half = (y_local[-1] - y_local[0]) / 2
    if abs(shift[0]) > dx_half or abs(shift[1]) > dy_half:
        return None  # shifted outside window — unreliable

    U_fit = float(coeffs @ np.array([1, shift[0], shift[1],
                                      shift[0]**2, shift[1]**2,
                                      shift[0] * shift[1]]))
    fit_res = float(np.sqrt(np.mean((A @ coeffs - ul)**2)))
    return (x_ref, y_ref, U_fit, fit_res, eig_vals, True)


def compute_local_depth(U, xg, yg, iy, ix, search_radius_pts=15):
    """
    Estimate local trap depth (prominence) as the difference between
    the minimum U value at (iy, ix) and the minimum of the surrounding
    annular ring.
    """
    ny, nx = U.shape
    U_min = U[iy, ix]

    iy_lo = max(iy - search_radius_pts, 0)
    iy_hi = min(iy + search_radius_pts + 1, ny)
    ix_lo = max(ix - search_radius_pts, 0)
    ix_hi = min(ix + search_radius_pts + 1, nx)

    U_patch = U[iy_lo:iy_hi, ix_lo:ix_hi]
    YP, XP = np.meshgrid(
        np.arange(iy_lo, iy_hi) - iy,
        np.arange(ix_lo, ix_hi) - ix,
        indexing="ij"
    )
    r = np.sqrt(XP**2 + YP**2)

    # Ring between 0.6*search_radius and search_radius
    ring_mask = (r >= 0.6 * search_radius_pts) & (r <= search_radius_pts)
    if ring_mask.sum() == 0:
        return 0.0

    U_ring_max = U_patch[ring_mask].max()
    depth = U_ring_max - U_min
    return float(depth)


def find_pressure_minima(p_abs, xg, yg, min_distance_pts=5, n_max=50):
    """Find local minima of |p| for comparison."""
    neighborhood = 2 * min_distance_pts + 1
    local_min = minimum_filter(p_abs, size=neighborhood)
    mask = (p_abs == local_min)
    mask[:2, :] = False
    mask[-2:, :] = False
    mask[:, :2] = False
    mask[:, -2:] = False
    iy, ix = np.where(mask)
    vals = p_abs[iy, ix]
    order = np.argsort(vals)
    n = min(n_max, len(order))
    return iy[order[:n]], ix[order[:n]]


def compute_gradient_at_point(U, xg, yg, x, y):
    """Interpolate U-gradient at an arbitrary (x,y) point."""
    Fy_grid, Fx_grid = np.gradient(U, yg[1] - yg[0], xg[1] - xg[0])
    rgi_fx = RegularGridInterpolator((yg, xg), -Fx_grid,
                                     bounds_error=False, fill_value=np.nan)
    rgi_fy = RegularGridInterpolator((yg, xg), -Fy_grid,
                                     bounds_error=False, fill_value=np.nan)
    pt = np.array([[y, x]])
    return float(rgi_fx(pt)), float(rgi_fy(pt))


# ═══════════════════════════════════════════════════════════════════
# Main validation pipeline
# ═══════════════════════════════════════════════════════════════════

def run_validation(ngrid, cache_path, z_val, roi_half, label_tag="medium"):
    """
    Run full trap-localisation validation pipeline at a given resolution.

    Returns a dict with all results for this resolution.
    """
    t0 = time.time()
    coords, p_fem = load_fem_cache(cache_path)
    tree = cKDTree(coords)

    # Build ROI grid
    xg = np.linspace(CX - roi_half, CX + roi_half, ngrid)
    yg = np.linspace(CY - roi_half, CY + roi_half, ngrid)
    dx = xg[1] - xg[0]
    dy = yg[1] - yg[0]

    # Interpolate FEM pressure at z*, z*+dz, z*-dz
    p_grid, XX, YY = interpolate_fem_to_grid(tree, p_fem, xg, yg, z_val)
    p_zp, _, _ = interpolate_fem_to_grid(tree, p_fem, xg, yg, z_val + DZ_GRAD)
    p_zm, _, _ = interpolate_fem_to_grid(tree, p_fem, xg, yg, z_val - DZ_GRAD)

    # Compute Gor'kov and force
    U, Fx, Fy, grad_p_mag2 = gorkov_and_force_on_grid(
        p_grid, dx, dy, p_zp=p_zp, p_zm=p_zm, dz=DZ_GRAD
    )
    F_mag = np.sqrt(Fx**2 + Fy**2)
    p_abs = np.abs(p_grid)
    p_phase = np.angle(p_grid)

    extent_mm = np.array([xg[0], xg[-1], yg[0], yg[-1]]) * 1e3

    # ── OLD METHOD ──
    iy_old, ix_old = find_trap_minima_OLD(U)
    old_traps = []
    for j in range(len(iy_old)):
        old_traps.append({
            "x_mm": float(xg[ix_old[j]] * 1e3),
            "y_mm": float(yg[iy_old[j]] * 1e3),
            "ix": int(ix_old[j]),
            "iy": int(iy_old[j]),
            "U": float(U[iy_old[j], ix_old[j]]),
        })

    # ── NEW CANDIDATE DETECTION ──
    iy_cand, ix_cand = detect_candidates(U, xg, yg)

    # ── PRESSURE MINIMA (for comparison) ──
    iy_pmin, ix_pmin = find_pressure_minima(p_abs, xg, yg)
    pressure_minima = []
    for j in range(len(iy_pmin)):
        pressure_minima.append({
            "x_mm": float(xg[ix_pmin[j]] * 1e3),
            "y_mm": float(yg[iy_pmin[j]] * 1e3),
            "p_abs": float(p_abs[iy_pmin[j], ix_pmin[j]]),
        })

    # ── REFINE AND CLASSIFY EACH CANDIDATE ──
    all_candidates = []
    accepted = []
    rejected = []

    for j in range(len(iy_cand)):
        iy_j, ix_j = int(iy_cand[j]), int(ix_cand[j])
        x_grid = float(xg[ix_j])
        y_grid = float(yg[iy_j])
        U_val = float(U[iy_j, ix_j])
        F_val = float(F_mag[iy_j, ix_j])

        # Compute local depth (prominence)
        depth = compute_local_depth(U, xg, yg, iy_j, ix_j,
                                    search_radius_pts=max(8, int(TRAP_SP / dx * 0.6)))

        # Sub-grid refinement
        ref_result = refine_subgrid(U, xg, yg, iy_j, ix_j, half_win=4)

        if ref_result is not None:
            x_ref, y_ref, U_ref, fit_res, eig_vals, is_true_min = ref_result
        else:
            x_ref, y_ref = x_grid, y_grid
            U_ref = U_val
            fit_res = np.nan
            eig_vals = np.array([np.nan, np.nan])
            is_true_min = False

        # Hessian from finite differences on grid (complementary check)
        if 2 <= iy_j < U.shape[0] - 2 and 2 <= ix_j < U.shape[1] - 2:
            Hxx = (U[iy_j, ix_j + 1] - 2 * U[iy_j, ix_j] + U[iy_j, ix_j - 1]) / dx**2
            Hyy = (U[iy_j + 1, ix_j] - 2 * U[iy_j, ix_j] + U[iy_j - 1, ix_j]) / dy**2
            Hxy = (U[iy_j + 1, ix_j + 1] - U[iy_j + 1, ix_j - 1]
                   - U[iy_j - 1, ix_j + 1] + U[iy_j - 1, ix_j - 1]) / (4 * dx * dy)
            H_fd = np.array([[Hxx, Hxy], [Hxy, Hyy]])
            eig_fd = np.linalg.eigvalsh(H_fd)
        else:
            Hxx = Hyy = Hxy = np.nan
            H_fd = np.array([[np.nan, np.nan], [np.nan, np.nan]])
            eig_fd = np.array([np.nan, np.nan])

        # Gradient norm at grid point
        grad_norm = float(F_val)  # F = -grad(U), so |F| = |grad(U)|

        # Shift from grid to refined
        shift_x = x_ref - x_grid
        shift_y = y_ref - y_grid
        shift_mm = np.sqrt(shift_x**2 + shift_y**2) * 1e3

        cand = {
            "idx": j,
            "ix": ix_j,
            "iy": iy_j,
            "x_grid_mm": float(x_grid * 1e3),
            "y_grid_mm": float(y_grid * 1e3),
            "x_ref_mm": float(x_ref * 1e3),
            "y_ref_mm": float(y_ref * 1e3),
            "shift_mm": float(shift_mm),
            "shift_frac_lambda": float(np.sqrt(shift_x**2 + shift_y**2) / LAM),
            "U_grid": float(U_val),
            "U_ref": float(U_ref),
            "depth": float(depth),
            "depth_frac": float(depth / abs(U_val)) if U_val != 0 else 0.0,
            "grad_norm": float(grad_norm),
            "Hxx": float(Hxx),
            "Hyy": float(Hyy),
            "Hxy": float(Hxy),
            "eig_fd_0": float(eig_fd[0]),
            "eig_fd_1": float(eig_fd[1]),
            "eig_fit_0": float(eig_vals[0]),
            "eig_fit_1": float(eig_vals[1]),
            "fit_residual": float(fit_res) if not np.isnan(fit_res) else None,
            "is_true_min_from_fit": bool(is_true_min),
        }

        # ── Classification ──
        rejection_reasons = []

        # 1. Not a true minimum from quadratic fit
        if not is_true_min and ref_result is not None:
            rejection_reasons.append("not_true_minimum_in_fit")

        # 2. Negative FD Hessian eigenvalue (not positive-definite)
        if not np.isnan(eig_fd[0]) and eig_fd[0] <= 0:
            rejection_reasons.append("non_positive_hessian_fd")

        # 3. Too shallow (depth < 1% of deepest candidate)
        # (deferred — computed after all candidates collected)

        # 4. Gradient too large (should be near zero at a minimum)
        # Force magnitude as fraction of max force in ROI
        rel_force = grad_norm / (F_mag.max() + 1e-30)
        if rel_force > 0.1:
            rejection_reasons.append(f"force_residual_high_{rel_force:.3f}")

        # 5. Boundary proximity
        border_pts = 5
        if (ix_j < border_pts or ix_j >= ngrid - border_pts or
                iy_j < border_pts or iy_j >= ngrid - border_pts):
            rejection_reasons.append("boundary_proximity")

        cand["rejection_reasons"] = rejection_reasons
        cand["accepted"] = len(rejection_reasons) == 0
        all_candidates.append(cand)

    # Deferred depth filter: reject candidates shallower than 1% of max depth
    if all_candidates:
        max_depth = max(c["depth"] for c in all_candidates)
        for c in all_candidates:
            if max_depth > 0 and c["depth"] < 0.01 * max_depth:
                c["rejection_reasons"].append(
                    f"too_shallow_{c['depth']/max_depth:.4f}")
                c["accepted"] = False

    # Deduplicate: if two accepted candidates are within 0.3 * trap_sp, keep deeper
    accepted_list = [c for c in all_candidates if c["accepted"]]
    accepted_list.sort(key=lambda c: c["U_ref"])
    keep_mask = np.ones(len(accepted_list), dtype=bool)
    dedup_dist_mm = 0.3 * TRAP_SP * 1e3
    for i in range(len(accepted_list)):
        if not keep_mask[i]:
            continue
        for k in range(i + 1, len(accepted_list)):
            if not keep_mask[k]:
                continue
            dist = np.sqrt(
                (accepted_list[i]["x_ref_mm"] - accepted_list[k]["x_ref_mm"])**2 +
                (accepted_list[i]["y_ref_mm"] - accepted_list[k]["y_ref_mm"])**2
            )
            if dist < dedup_dist_mm:
                keep_mask[k] = False
                all_candidates[accepted_list[k]["idx"]]["rejection_reasons"].append(
                    "duplicate_nearby")
                all_candidates[accepted_list[k]["idx"]]["accepted"] = False

    # Final accepted/rejected
    accepted = [c for c in all_candidates if c["accepted"]]
    rejected = [c for c in all_candidates if not c["accepted"]]

    # ── Nearest-neighbour distances for accepted traps ──
    if len(accepted) >= 2:
        pos_acc = np.array([[c["x_ref_mm"], c["y_ref_mm"]] for c in accepted])
        nn_tree = cKDTree(pos_acc)
        dd, ii = nn_tree.query(pos_acc, k=2)
        for j, c in enumerate(accepted):
            c["nn_distance_mm"] = float(dd[j, 1])
            c["nn_distance_frac_trap_sp"] = float(dd[j, 1] / (TRAP_SP * 1e3))
    else:
        for c in accepted:
            c["nn_distance_mm"] = np.nan
            c["nn_distance_frac_trap_sp"] = np.nan

    # ── Comparison: old vs new ──
    comparison = []
    for ot in old_traps:
        best_dist = np.inf
        best_new = None
        for nt in accepted:
            dist = np.sqrt((ot["x_mm"] - nt["x_ref_mm"])**2 +
                           (ot["y_mm"] - nt["y_ref_mm"])**2)
            if dist < best_dist:
                best_dist = dist
                best_new = nt
        comparison.append({
            "old_x_mm": ot["x_mm"],
            "old_y_mm": ot["y_mm"],
            "old_U": ot["U"],
            "new_x_mm": best_new["x_ref_mm"] if best_new else None,
            "new_y_mm": best_new["y_ref_mm"] if best_new else None,
            "new_U": best_new["U_ref"] if best_new else None,
            "offset_mm": float(best_dist) if best_new else None,
            "offset_frac_lambda": float(best_dist / (LAM * 1e3)) if best_new else None,
            "offset_frac_trap_sp": float(best_dist / (TRAP_SP * 1e3)) if best_new else None,
        })

    # ── Comparison: pressure minimum vs potential minimum ──
    p_vs_u_comparison = []
    for ac in accepted:
        # Find nearest pressure minimum
        best_dist_p = np.inf
        best_pm = None
        for pm in pressure_minima:
            dist = np.sqrt((ac["x_ref_mm"] - pm["x_mm"])**2 +
                           (ac["y_ref_mm"] - pm["y_mm"])**2)
            if dist < best_dist_p:
                best_dist_p = dist
                best_pm = pm
        p_vs_u_comparison.append({
            "trap_x_mm": ac["x_ref_mm"],
            "trap_y_mm": ac["y_ref_mm"],
            "trap_U": ac["U_ref"],
            "nearest_pmin_x_mm": best_pm["x_mm"] if best_pm else None,
            "nearest_pmin_y_mm": best_pm["y_mm"] if best_pm else None,
            "pmin_offset_mm": float(best_dist_p) if best_pm else None,
            "pmin_offset_frac_lambda": float(best_dist_p / (LAM * 1e3)) if best_pm else None,
        })

    elapsed = time.time() - t0

    return {
        "label": label_tag,
        "ngrid": ngrid,
        "dx_um": dx * 1e6,
        "dy_um": dy * 1e6,
        "pts_per_lambda": LAM / dx,
        "elapsed_s": elapsed,
        "xg": xg,
        "yg": yg,
        "XX": XX,
        "YY": YY,
        "p_grid": p_grid,
        "p_abs": p_abs,
        "p_phase": p_phase,
        "U": U,
        "Fx": Fx,
        "Fy": Fy,
        "F_mag": F_mag,
        "extent_mm": extent_mm,
        "old_traps": old_traps,
        "all_candidates": all_candidates,
        "accepted": accepted,
        "rejected": rejected,
        "pressure_minima": pressure_minima,
        "comparison_old_new": comparison,
        "comparison_p_vs_u": p_vs_u_comparison,
    }


# ═══════════════════════════════════════════════════════════════════
# Z-plane sensitivity
# ═══════════════════════════════════════════════════════════════════

def run_zplane_sensitivity(cache_path, ngrid, roi_half, z_offsets_frac_lam):
    """Run trap detection at several z-planes near z*."""
    results = []
    for dz_frac in z_offsets_frac_lam:
        z_val = Z_STAR + dz_frac * LAM
        r = run_validation(ngrid, cache_path, z_val, roi_half,
                           label_tag=f"z_offset_{dz_frac:+.2f}lam")
        trap_positions = [(c["x_ref_mm"], c["y_ref_mm"]) for c in r["accepted"]]
        results.append({
            "dz_frac_lam": dz_frac,
            "z_mm": z_val * 1e3,
            "n_accepted": len(r["accepted"]),
            "trap_positions": trap_positions,
        })
    return results


# ═══════════════════════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════════════════════

def plot_field_overlays(result, fig_dir, prefix="baseline"):
    """Generate field overlay diagnostic figures (Section 5A)."""
    ext = result["extent_mm"]
    U = result["U"]
    p_abs = result["p_abs"]
    p_phase = result["p_phase"]
    F_mag = result["F_mag"]
    Fx = result["Fx"]
    Fy = result["Fy"]
    xg = result["xg"]
    yg = result["yg"]
    old_traps = result["old_traps"]
    accepted = result["accepted"]
    all_candidates = result["all_candidates"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # A1: |p| heatmap
    ax = axes[0, 0]
    im = ax.imshow(p_abs, extent=ext, origin="lower", cmap="inferno")
    plt.colorbar(im, ax=ax, label="|p| [Pa]")
    ax.set_title("|p| (pressure magnitude)")
    _overlay_traps(ax, old_traps, accepted)

    # A2: phase
    ax = axes[0, 1]
    im = ax.imshow(p_phase, extent=ext, origin="lower", cmap="twilight",
                   vmin=-np.pi, vmax=np.pi)
    plt.colorbar(im, ax=ax, label="phase [rad]")
    ax.set_title("Pressure phase")
    _overlay_traps(ax, old_traps, accepted)

    # A3: Gor'kov potential
    ax = axes[0, 2]
    im = ax.imshow(U, extent=ext, origin="lower", cmap="viridis")
    plt.colorbar(im, ax=ax, label="U [J]")
    ax.set_title("Gor'kov potential U")
    _overlay_traps(ax, old_traps, accepted)

    # A4: Force magnitude
    ax = axes[1, 0]
    im = ax.imshow(F_mag, extent=ext, origin="lower", cmap="hot")
    plt.colorbar(im, ax=ax, label="|F| [N]")
    ax.set_title("Force magnitude |F|")
    _overlay_traps(ax, old_traps, accepted)

    # A5: Force quiver over potential
    ax = axes[1, 1]
    im = ax.imshow(U, extent=ext, origin="lower", cmap="viridis", alpha=0.7)
    plt.colorbar(im, ax=ax, label="U [J]")
    # Subsample quiver
    skip = max(1, result["ngrid"] // 30)
    xq = xg[::skip] * 1e3
    yq = yg[::skip] * 1e3
    Fxq = Fx[::skip, ::skip]
    Fyq = Fy[::skip, ::skip]
    F_scale = max(np.abs(Fxq).max(), np.abs(Fyq).max(), 1e-30)
    ax.quiver(xq, yq, Fxq / F_scale, Fyq / F_scale, color="white", alpha=0.7,
              scale=30, scale_units="width")
    ax.set_title("Force vectors over U")
    _overlay_traps(ax, old_traps, accepted)

    # A6: All candidate minima
    ax = axes[1, 2]
    im = ax.imshow(U, extent=ext, origin="lower", cmap="viridis")
    plt.colorbar(im, ax=ax, label="U [J]")
    # All candidates
    for c in all_candidates:
        color = "lime" if c["accepted"] else "red"
        marker = "o" if c["accepted"] else "x"
        ax.plot(c["x_grid_mm"], c["y_grid_mm"], marker, color=color,
                ms=5, alpha=0.7)
    ax.set_title("All candidates (green=accept, red=reject)")
    _overlay_traps(ax, old_traps, accepted)

    for ax in axes.ravel():
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")

    fig.suptitle(f"Field Overlays — {prefix} (N={result['ngrid']})", fontsize=14)
    fig.tight_layout()
    fig.savefig(fig_dir / f"{prefix}_field_overlays.png", dpi=150)
    plt.close(fig)


def _overlay_traps(ax, old_traps, accepted):
    """Overlay old and new trap markers."""
    for ot in old_traps:
        ax.plot(ot["x_mm"], ot["y_mm"], "s", color="cyan", ms=7,
                mew=1.5, mfc="none", label="old" if ot is old_traps[0] else "")
    for nt in accepted:
        ax.plot(nt["x_ref_mm"], nt["y_ref_mm"], "*", color="yellow", ms=10,
                mew=0.5, label="new" if nt is accepted[0] else "")


def plot_local_zoom(result, fig_dir, prefix="baseline", n_traps=4):
    """Generate zoomed panels for representative traps (Section 5B)."""
    accepted = result["accepted"]
    if not accepted:
        return

    U = result["U"]
    p_abs = result["p_abs"]
    Fx = result["Fx"]
    Fy = result["Fy"]
    xg = result["xg"]
    yg = result["yg"]
    dx = xg[1] - xg[0]
    old_traps = result["old_traps"]

    n_show = min(n_traps, len(accepted))
    fig, axes = plt.subplots(n_show, 4, figsize=(20, 5 * n_show))
    if n_show == 1:
        axes = axes[np.newaxis, :]

    zoom_half_mm = TRAP_SP * 1e3 * 1.2  # zoom to ~1.2 trap spacings

    for row, trap in enumerate(accepted[:n_show]):
        x_c = trap["x_ref_mm"]
        y_c = trap["y_ref_mm"]

        # Pixel window
        ix_c = np.argmin(np.abs(xg * 1e3 - x_c))
        iy_c = np.argmin(np.abs(yg * 1e3 - y_c))
        half_pix = int(zoom_half_mm / (dx * 1e3)) + 1

        ix_lo = max(ix_c - half_pix, 0)
        ix_hi = min(ix_c + half_pix + 1, len(xg))
        iy_lo = max(iy_c - half_pix, 0)
        iy_hi = min(iy_c + half_pix + 1, len(yg))

        ext_z = [xg[ix_lo] * 1e3, xg[ix_hi - 1] * 1e3,
                 yg[iy_lo] * 1e3, yg[iy_hi - 1] * 1e3]

        U_z = U[iy_lo:iy_hi, ix_lo:ix_hi]
        p_z = p_abs[iy_lo:iy_hi, ix_lo:ix_hi]
        Fx_z = Fx[iy_lo:iy_hi, ix_lo:ix_hi]
        Fy_z = Fy[iy_lo:iy_hi, ix_lo:ix_hi]

        # B1: |p| zoomed
        ax = axes[row, 0]
        im = ax.imshow(p_z, extent=ext_z, origin="lower", cmap="inferno")
        plt.colorbar(im, ax=ax, label="|p|")
        ax.set_title(f"Trap #{row + 1}: |p|")
        ax.plot(trap["x_grid_mm"], trap["y_grid_mm"], "s", color="cyan",
                ms=8, mfc="none", mew=1.5)
        ax.plot(x_c, y_c, "*", color="yellow", ms=12, mew=0.5)

        # B2: U zoomed with contours
        ax = axes[row, 1]
        im = ax.imshow(U_z, extent=ext_z, origin="lower", cmap="viridis")
        xz_mm = xg[ix_lo:ix_hi] * 1e3
        yz_mm = yg[iy_lo:iy_hi] * 1e3
        ax.contour(xz_mm, yz_mm, U_z, levels=8, colors="white",
                   linewidths=0.5, alpha=0.6)
        plt.colorbar(im, ax=ax, label="U [J]")
        ax.set_title(f"Trap #{row + 1}: U + contours")
        ax.plot(trap["x_grid_mm"], trap["y_grid_mm"], "s", color="cyan",
                ms=8, mfc="none", mew=1.5)
        ax.plot(x_c, y_c, "*", color="yellow", ms=12, mew=0.5)
        # Show old trap if nearby
        for ot in old_traps:
            if (abs(ot["x_mm"] - x_c) < zoom_half_mm and
                    abs(ot["y_mm"] - y_c) < zoom_half_mm):
                ax.plot(ot["x_mm"], ot["y_mm"], "D", color="red",
                        ms=8, mfc="none", mew=1.5)

        # B3: Force vectors zoomed
        ax = axes[row, 2]
        im = ax.imshow(U_z, extent=ext_z, origin="lower", cmap="viridis",
                       alpha=0.5)
        skip = max(1, (ix_hi - ix_lo) // 15)
        xq = xg[ix_lo:ix_hi:skip] * 1e3
        yq = yg[iy_lo:iy_hi:skip] * 1e3
        Fxs = Fx_z[::skip, ::skip]
        Fys = Fy_z[::skip, ::skip]
        F_sc = max(np.abs(Fxs).max(), np.abs(Fys).max(), 1e-30)
        ax.quiver(xq, yq, Fxs / F_sc, Fys / F_sc,
                  color="white", scale=30, scale_units="width")
        plt.colorbar(im, ax=ax, label="U [J]")
        ax.set_title(f"Trap #{row + 1}: force vectors")
        ax.plot(x_c, y_c, "*", color="yellow", ms=12, mew=0.5)

        # B4: Diagnostic text
        ax = axes[row, 3]
        ax.axis("off")
        info = (
            f"Grid pos: ({trap['x_grid_mm']:.4f}, {trap['y_grid_mm']:.4f}) mm\n"
            f"Refined:  ({trap['x_ref_mm']:.4f}, {trap['y_ref_mm']:.4f}) mm\n"
            f"Shift:    {trap['shift_mm']:.4f} mm "
            f"({trap['shift_frac_lambda']:.4f} λ)\n"
            f"U_grid:   {trap['U_grid']:.3e} J\n"
            f"U_ref:    {trap['U_ref']:.3e} J\n"
            f"Depth:    {trap['depth']:.3e} J\n"
            f"|F| at grid: {trap['grad_norm']:.3e} N\n"
            f"Hessian eig (FD): [{trap['eig_fd_0']:.2e}, {trap['eig_fd_1']:.2e}]\n"
            f"Hessian eig (fit): [{trap['eig_fit_0']:.2e}, {trap['eig_fit_1']:.2e}]\n"
            f"NN dist:  {trap.get('nn_distance_mm', float('nan')):.4f} mm "
            f"({trap.get('nn_distance_frac_trap_sp', float('nan')):.3f} × λ/2)\n"
            f"True min: {trap['is_true_min_from_fit']}"
        )
        ax.text(0.05, 0.95, info, transform=ax.transAxes, fontsize=9,
                verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    fig.suptitle(f"Local Zoom Diagnostics — {prefix}", fontsize=14)
    fig.tight_layout()
    fig.savefig(fig_dir / f"{prefix}_local_zoom.png", dpi=150)
    plt.close(fig)


def plot_convergence(convergence_data, fig_dir):
    """Plot trap position shifts between resolutions."""
    if len(convergence_data) < 2:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    labels = [d["label"] for d in convergence_data]
    n_traps = [len(d["accepted"]) for d in convergence_data]
    ngrids = [d["ngrid"] for d in convergence_data]

    # Trap count vs resolution
    ax = axes[0]
    ax.bar(labels, n_traps, color="steelblue")
    ax.set_ylabel("Number of accepted traps")
    ax.set_xlabel("Resolution")
    ax.set_title("Trap count vs grid resolution")
    for i, (lab, nt, ng) in enumerate(zip(labels, n_traps, ngrids)):
        ax.text(i, nt + 0.3, f"N={ng}\n{nt} traps", ha="center", fontsize=9)

    # Position shifts between resolutions
    ax = axes[1]
    # Compare with finest resolution
    finest = convergence_data[-1]
    if len(finest["accepted"]) == 0:
        ax.text(0.5, 0.5, "No accepted traps in finest resolution",
                transform=ax.transAxes, ha="center")
    else:
        pos_fine = np.array([[c["x_ref_mm"], c["y_ref_mm"]]
                             for c in finest["accepted"]])
        for cd in convergence_data[:-1]:
            if len(cd["accepted"]) == 0:
                continue
            pos_c = np.array([[c["x_ref_mm"], c["y_ref_mm"]]
                              for c in cd["accepted"]])
            # Match each coarse trap to nearest fine trap
            tree_fine = cKDTree(pos_fine)
            dd, _ = tree_fine.query(pos_c)
            ax.hist(dd, bins=20, alpha=0.6, label=f"{cd['label']} → fine")

        ax.set_xlabel("Position shift to fine [mm]")
        ax.set_ylabel("Count")
        ax.set_title("Trap position shift vs finest resolution")
        ax.legend()
        ax.axvline(TRAP_SP * 1e3 * 0.01, color="red", ls="--", lw=0.8,
                   label="1% of λ/2")

    fig.tight_layout()
    fig.savefig(fig_dir / "convergence_study.png", dpi=150)
    plt.close(fig)


def plot_symmetry_nn(result, fig_dir, prefix="baseline"):
    """Symmetry and nearest-neighbour diagnostic (Section 5F)."""
    accepted = result["accepted"]
    if len(accepted) < 2:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    nn_dists = [c["nn_distance_mm"] for c in accepted if not np.isnan(c.get("nn_distance_mm", np.nan))]
    trap_sp_mm = TRAP_SP * 1e3

    # F1: NN distance histogram
    ax = axes[0]
    ax.hist(nn_dists, bins=20, color="steelblue", edgecolor="black")
    ax.axvline(trap_sp_mm, color="red", ls="--", lw=2, label=f"λ/2 = {trap_sp_mm:.3f} mm")
    ax.set_xlabel("NN distance [mm]")
    ax.set_ylabel("Count")
    ax.set_title("Nearest-neighbour distances")
    ax.legend()

    # F2: Trap positions with expected grid
    ax = axes[1]
    ext = result["extent_mm"]
    ax.imshow(result["U"], extent=ext, origin="lower", cmap="viridis", alpha=0.5)
    for c in accepted:
        ax.plot(c["x_ref_mm"], c["y_ref_mm"], "*", color="yellow", ms=10)
    # Expected grid lines at λ/2 spacing from centre
    cx_mm, cy_mm = CX * 1e3, CY * 1e3
    for n in range(-10, 11):
        ax.axhline(cy_mm + n * trap_sp_mm, color="white", lw=0.3, alpha=0.3)
        ax.axvline(cx_mm + n * trap_sp_mm, color="white", lw=0.3, alpha=0.3)
    ax.set_title("Trap positions vs expected λ/2 grid")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")

    # F3: Deviation from nearest lattice point
    deviations = []
    for c in accepted:
        # Nearest expected lattice point
        nx = round((c["x_ref_mm"] - cx_mm) / trap_sp_mm)
        ny = round((c["y_ref_mm"] - cy_mm) / trap_sp_mm)
        x_exp = cx_mm + nx * trap_sp_mm
        y_exp = cy_mm + ny * trap_sp_mm
        dev = np.sqrt((c["x_ref_mm"] - x_exp)**2 + (c["y_ref_mm"] - y_exp)**2)
        deviations.append(dev)

    ax = axes[2]
    ax.hist(deviations, bins=20, color="coral", edgecolor="black")
    ax.set_xlabel("Deviation from nearest lattice point [mm]")
    ax.set_ylabel("Count")
    ax.set_title("Lattice registration error")
    ax.axvline(0, color="green", ls="--", lw=1)
    mean_dev = np.mean(deviations)
    ax.axvline(mean_dev, color="blue", ls="--", lw=1,
               label=f"mean = {mean_dev:.4f} mm")
    ax.legend()

    fig.suptitle(f"Symmetry / Lattice Diagnostics — {prefix}", fontsize=14)
    fig.tight_layout()
    fig.savefig(fig_dir / f"{prefix}_symmetry_nn.png", dpi=150)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# CSV Writers
# ═══════════════════════════════════════════════════════════════════

def write_csv(filepath, rows, fieldnames):
    with open(filepath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            # Flatten rejection_reasons list to string
            row = dict(r)
            if "rejection_reasons" in row:
                row["rejection_reasons"] = "; ".join(row["rejection_reasons"])
            w.writerow(row)


CANDIDATE_FIELDS = [
    "idx", "ix", "iy",
    "x_grid_mm", "y_grid_mm", "x_ref_mm", "y_ref_mm",
    "shift_mm", "shift_frac_lambda",
    "U_grid", "U_ref", "depth", "depth_frac",
    "grad_norm",
    "Hxx", "Hyy", "Hxy",
    "eig_fd_0", "eig_fd_1", "eig_fit_0", "eig_fit_1",
    "fit_residual", "is_true_min_from_fit",
    "nn_distance_mm", "nn_distance_frac_trap_sp",
    "accepted", "rejection_reasons",
]

COMPARISON_FIELDS = [
    "old_x_mm", "old_y_mm", "old_U",
    "new_x_mm", "new_y_mm", "new_U",
    "offset_mm", "offset_frac_lambda", "offset_frac_trap_sp",
]

P_VS_U_FIELDS = [
    "trap_x_mm", "trap_y_mm", "trap_U",
    "nearest_pmin_x_mm", "nearest_pmin_y_mm",
    "pmin_offset_mm", "pmin_offset_frac_lambda",
]


# ═══════════════════════════════════════════════════════════════════
# Report generation
# ═══════════════════════════════════════════════════════════════════

def generate_report(out_dir, results_medium, convergence_results,
                    zplane_results, config):
    """Generate the full markdown report."""
    r = results_medium  # primary result
    lines = []

    def w(s=""):
        lines.append(s)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    w("# Trap Localisation Validation Study")
    w()
    w(f"**Date**: {ts}")
    w(f"**Runtime**: {sum(c['elapsed_s'] for c in convergence_results):.1f}s (all resolutions)")
    w()
    w("## 1. Objective")
    w()
    w("Verify that detected traps correspond to **real stable minima** of the")
    w("Gor'kov potential, not artefacts of grid discretisation or threshold filtering.")
    w("Produce enough diagnostics to determine exactly what is wrong with the")
    w("current trap-finding method used in bridge studies.")
    w()
    w("## 2. Why Bridge Study is Paused")
    w()
    w("The bridge master study reported trap positions that do not visually sit at")
    w("the centres of the relevant low-pressure / potential features. The inferred")
    w("bridge metrics are therefore physically unreliable. This validation study")
    w("must be completed before any bridge optimisation resumes.")
    w()
    w("## 3. Current Trap-Finding Method — Audit")
    w()
    w("### Location in codebase")
    w()
    w("- `scripts/dev/bridge_master_study.py` → `find_trap_minima()`")
    w("- `scripts/dev/bridge_phase_offset_study.py` → same function")
    w("- `scripts/analysis/diagnostics_utils.py` → `find_gorkov_minima_2d()`")
    w()
    w("### What the current method does")
    w()
    w("```python")
    w("def find_trap_minima(U, min_sep_pix=10, threshold_frac=0.3):")
    w('    neg_U = -U')
    w('    local_max = (neg_U == maximum_filter(neg_U, size=min_sep_pix))')
    w('    local_max &= (neg_U > threshold_frac * neg_U.max())')
    w('    iy, ix = np.where(local_max)')
    w('    return iy, ix')
    w("```")
    w()
    w("### Issues identified")
    w()
    w("1. **Grid-only detection**: Trap positions are locked to grid nodes.")
    w("   No sub-grid refinement is performed. On a 400×400 grid over a 2.2λ ROI,")
    w(f"   grid spacing ~ {r['dx_um']:.1f} µm, which is {r['dx_um'] / (LAM * 1e6) * 100:.1f}%"
      " of λ. For λ/2 trap spacing the grid error can be a meaningful")
    w("   fraction of the trap separation.")
    w("2. **Threshold filter (`neg_U > 0.3 * neg_U.max()`)**: This hard threshold")
    w("   rejects any minimum shallower than 30% of the deepest minimum. This can")
    w("   discard physically real but weakly-confined traps, or include artefactual")
    w("   minima whose depth just happens to exceed 30%.")
    w("3. **No curvature check**: The method does not verify that candidates are true")
    w("   minima (positive-definite Hessian). A saddle point could pass if it is")
    w("   a local minimum along one axis.")
    w("4. **No force verification**: The gradient of U (= radiation force) at the")
    w("   detected grid-node position is not checked. A true trap must have |F| ≈ 0.")
    w("5. **No depth / prominence estimation**: The 30% threshold is relative to the")
    w("   global deepest minimum, not a local prominence measure.")
    w("6. **No sub-grid saddle refinement**: The bridge metric samples U along a")
    w("   200-point line between grid-locked trap positions. Both the trap positions")
    w("   and saddle are resolution-limited.")
    w()
    w("### Trap basis")
    w()
    w("Traps are detected from the **Gor'kov potential U**, which is correct in")
    w("principle. The Gor'kov formula uses the full velocity gradient (not the")
    w("plane-wave approximation), which was fixed in Feb 2026.")
    w()
    w("The fundamental issue is **not** using pressure minima instead of potential")
    w("minima — the code does use U. The issues are discretisation, thresholding,")
    w("and lack of validation/refinement.")
    w()
    w("## 4. New Formal Trap Definition")
    w()
    w("A **trap** is an accepted local minimum of the Gor'kov potential U, satisfying:")
    w()
    w("1. Sub-grid-refined position via local quadratic (paraboloid) fit")
    w("2. Positive-definite Hessian at the refined position (both eigenvalues > 0)")
    w("3. Near-zero force residual (|F|/|F|_max < 0.1)")
    w("4. Non-trivial local depth (prominence > 1% of maximum depth)")
    w("5. Not a boundary artefact (> 5 pixels from ROI edge)")
    w("6. Not a duplicate of a deeper nearby minimum")
    w()
    w("## 5. Baseline Field Setup")
    w()
    w(f"- **FEM cache**: `{config['cache_path']}`")
    w(f"- **DOFs**: {config['n_dofs']}")
    w(f"- **z***: {Z_STAR * 1e3:.4f} mm")
    w(f"- **ROI**: ±{ROI_HALF * 1e3:.3f} mm ({ROI_HALF / LAM:.1f}λ) centred at"
      f" ({CX * 1e3:.1f}, {CY * 1e3:.1f}) mm")
    w(f"- **λ**: {LAM * 1e3:.4f} mm, trap spacing λ/2 = {TRAP_SP * 1e3:.4f} mm")
    w(f"- **3D gradient**: z-gradient from ±{DZ_GRAD * 1e6:.1f} µm offset planes")
    w()
    w("## 6. Resolution / Convergence Study")
    w()
    w("| Resolution | N_grid | dx [µm] | pts/λ | # accepted | Runtime [s] |")
    w("|------------|--------|---------|-------|------------|-------------|")
    for cr in convergence_results:
        w(f"| {cr['label']} | {cr['ngrid']} | {cr['dx_um']:.1f} | "
          f"{cr['pts_per_lambda']:.1f} | {len(cr['accepted'])} | {cr['elapsed_s']:.1f} |")
    w()

    # Cross-resolution position shifts
    if len(convergence_results) >= 2:
        finest = convergence_results[-1]
        if len(finest["accepted"]) > 0:
            pos_fine = np.array([[c["x_ref_mm"], c["y_ref_mm"]]
                                 for c in finest["accepted"]])
            tree_fine = cKDTree(pos_fine)
            w("### Position shifts relative to finest resolution")
            w()
            w("| From | Matched traps | Mean shift [µm] | Max shift [µm] | Mean shift / (λ/2) |")
            w("|------|---------------|-----------------|----------------|-------------------|")
            for cr in convergence_results[:-1]:
                if len(cr["accepted"]) == 0:
                    continue
                pos_c = np.array([[c["x_ref_mm"], c["y_ref_mm"]]
                                  for c in cr["accepted"]])
                dd, _ = tree_fine.query(pos_c)
                # Only count matched within half a trap spacing
                matched = dd < TRAP_SP * 1e3 * 0.5
                if matched.sum() > 0:
                    dm = dd[matched]
                    w(f"| {cr['label']} | {matched.sum()} | "
                      f"{dm.mean() * 1e3:.2f} | {dm.max() * 1e3:.2f} | "
                      f"{dm.mean() / (TRAP_SP * 1e3):.4f} |")
            w()

    w("## 7. Primary Results (Medium Resolution)")
    w()
    w(f"- Grid: {r['ngrid']}×{r['ngrid']}, dx = {r['dx_um']:.1f} µm")
    w(f"- Total candidates detected: {len(r['all_candidates'])}")
    w(f"- **Accepted traps: {len(r['accepted'])}**")
    w(f"- Rejected candidates: {len(r['rejected'])}")
    w()

    # Accepted traps table
    if r['accepted']:
        w("### Accepted Traps")
        w()
        w("| # | x [mm] | y [mm] | U [J] | Depth [J] | Shift [µm] | "
          "NN dist [mm] | eig_min | eig_max |")
        w("|---|--------|--------|-------|-----------|------------|"
          "-------------|---------|---------|")
        for i, c in enumerate(r['accepted']):
            w(f"| {i + 1} | {c['x_ref_mm']:.4f} | {c['y_ref_mm']:.4f} | "
              f"{c['U_ref']:.3e} | {c['depth']:.3e} | "
              f"{c['shift_mm'] * 1e3:.1f} | "
              f"{c.get('nn_distance_mm', float('nan')):.4f} | "
              f"{c['eig_fd_0']:.2e} | {c['eig_fd_1']:.2e} |")
        w()

    # Rejection reasons summary
    if r['rejected']:
        w("### Rejection Reasons Summary")
        w()
        reason_counts = {}
        for c in r['rejected']:
            for reason in c['rejection_reasons']:
                key = reason.split("_")[0] if reason.startswith("force_residual") or reason.startswith("too_shallow") else reason
                reason_counts[key] = reason_counts.get(key, 0) + 1
        for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
            w(f"- **{reason}**: {count}")
        w()

    w("## 8. Old vs New Trap Comparison")
    w()
    w(f"Old method detected: {len(r['old_traps'])} traps")
    w(f"New method accepted: {len(r['accepted'])} traps")
    w()

    if r['comparison_old_new']:
        w("| Old x [mm] | Old y [mm] | New x [mm] | New y [mm] | Offset [mm] | Offset/λ | Offset/(λ/2) |")
        w("|------------|------------|------------|------------|-------------|----------|-------------|")
        for c in r['comparison_old_new']:
            if c['new_x_mm'] is not None:
                w(f"| {c['old_x_mm']:.4f} | {c['old_y_mm']:.4f} | "
                  f"{c['new_x_mm']:.4f} | {c['new_y_mm']:.4f} | "
                  f"{c['offset_mm']:.4f} | {c['offset_frac_lambda']:.4f} | "
                  f"{c['offset_frac_trap_sp']:.4f} |")
            else:
                w(f"| {c['old_x_mm']:.4f} | {c['old_y_mm']:.4f} | "
                  "— | — | — | — | — |")
        w()

    w("## 9. Pressure Minimum vs Potential Minimum Comparison")
    w()
    if r['comparison_p_vs_u']:
        w("| Trap x [mm] | Trap y [mm] | Nearest |p|-min x [mm] | Nearest |p|-min y [mm] | Offset [mm] | Offset/λ |")
        w("|-------------|-------------|------------------------|------------------------|-------------|----------|")
        for c in r['comparison_p_vs_u']:
            if c['nearest_pmin_x_mm'] is not None:
                w(f"| {c['trap_x_mm']:.4f} | {c['trap_y_mm']:.4f} | "
                  f"{c['nearest_pmin_x_mm']:.4f} | {c['nearest_pmin_y_mm']:.4f} | "
                  f"{c['pmin_offset_mm']:.4f} | {c['pmin_offset_frac_lambda']:.4f} |")
        w()
        offsets_pu = [c['pmin_offset_mm'] for c in r['comparison_p_vs_u']
                      if c['pmin_offset_mm'] is not None]
        if offsets_pu:
            w(f"**Mean |p|-min to U-min offset**: {np.mean(offsets_pu):.4f} mm "
              f"({np.mean(offsets_pu) / (LAM * 1e3):.4f} λ)")
            w()

    # Z-plane sensitivity
    if zplane_results:
        w("## 10. Z-Plane Sensitivity")
        w()
        w("| δz / λ | z [mm] | # accepted |")
        w("|--------|--------|------------|")
        for zr in zplane_results:
            w(f"| {zr['dz_frac_lam']:+.2f} | {zr['z_mm']:.4f} | {zr['n_accepted']} |")
        w()

    w("## 11. Symmetry / Lattice Diagnostics")
    w()
    if r['accepted']:
        nn_dists = [c["nn_distance_mm"] for c in r['accepted']
                    if not np.isnan(c.get("nn_distance_mm", np.nan))]
        if nn_dists:
            w(f"- Mean NN distance: {np.mean(nn_dists):.4f} mm "
              f"(expected λ/2 = {TRAP_SP * 1e3:.4f} mm)")
            w(f"- Std NN distance: {np.std(nn_dists):.4f} mm")
            w(f"- Min NN distance: {np.min(nn_dists):.4f} mm")
            w(f"- Max NN distance: {np.max(nn_dists):.4f} mm")
            w()

        # Lattice deviation
        cx_mm, cy_mm = CX * 1e3, CY * 1e3
        deviations = []
        for c in r['accepted']:
            nx = round((c["x_ref_mm"] - cx_mm) / (TRAP_SP * 1e3))
            ny = round((c["y_ref_mm"] - cy_mm) / (TRAP_SP * 1e3))
            x_exp = cx_mm + nx * (TRAP_SP * 1e3)
            y_exp = cy_mm + ny * (TRAP_SP * 1e3)
            dev = np.sqrt((c["x_ref_mm"] - x_exp)**2 + (c["y_ref_mm"] - y_exp)**2)
            deviations.append(dev)
        w(f"- Mean deviation from nearest λ/2 lattice point: {np.mean(deviations):.4f} mm "
          f"({np.mean(deviations) / (TRAP_SP * 1e3) * 100:.2f}% of λ/2)")
        w(f"- Max deviation: {np.max(deviations):.4f} mm")
        w()

    w("## 12. Key Findings")
    w()

    # Automated analysis
    if r['accepted']:
        mean_shift = np.mean([c['shift_mm'] * 1e3 for c in r['accepted']])
        max_shift = np.max([c['shift_mm'] * 1e3 for c in r['accepted']])
        w(f"1. **Sub-grid refinement shifts**: Mean = {mean_shift:.1f} µm, "
          f"Max = {max_shift:.1f} µm")

        if r['comparison_old_new']:
            offsets = [c['offset_mm'] for c in r['comparison_old_new']
                       if c['offset_mm'] is not None]
            if offsets:
                w(f"2. **Old→New offset**: Mean = {np.mean(offsets) * 1e3:.1f} µm "
                  f"({np.mean(offsets) / (LAM * 1e3) * 100:.2f}% of λ), "
                  f"Max = {np.max(offsets) * 1e3:.1f} µm")

        if r['comparison_p_vs_u']:
            p_offsets = [c['pmin_offset_mm'] for c in r['comparison_p_vs_u']
                         if c['pmin_offset_mm'] is not None]
            if p_offsets:
                w(f"3. **|p|-minimum to U-minimum offset**: Mean = {np.mean(p_offsets) * 1e3:.1f} µm "
                  f"({np.mean(p_offsets) / (LAM * 1e3) * 100:.2f}% of λ)")

        all_true_min = sum(1 for c in r['accepted'] if c['is_true_min_from_fit'])
        w(f"4. **True minima from quadratic fit**: {all_true_min}/{len(r['accepted'])} accepted traps")

        # Check if old method confuses pressure and potential minima
        w(f"5. **Old method trap count**: {len(r['old_traps'])} "
          f"(new: {len(r['accepted'])})")
    w()

    w("## 13. Recommended Next Steps")
    w()
    w("1. Review the diagnostic figures and CSVs in this folder")
    w("2. If old↔new offsets are significant (> 5% of λ/2), the bridge metric is unreliable")
    w("3. Adopt the new validated trap positions before resuming any bridge study")
    w("4. Consider increasing grid resolution if convergence study shows instability")
    w("5. Update `find_trap_minima()` in bridge scripts with sub-grid refinement")
    w()
    w("## Figures")
    w()
    w("![Field overlays](figures/baseline_field_overlays.png)")
    w()
    w("![Local zoom](figures/baseline_local_zoom.png)")
    w()
    w("![Convergence](figures/convergence_study.png)")
    w()
    w("![Symmetry/NN](figures/baseline_symmetry_nn.png)")
    w()
    w("---")
    w(f"*Generated by trap_localisation_validation_study.py on {ts}*")

    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines))
    return report_path


# ═══════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "results" / f"trap_localisation_validation_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    log_fp = open(out_dir / "run.log", "w")

    log(f"Trap Localisation Validation Study — {timestamp}", log_fp)
    log(f"Output: {out_dir}", log_fp)

    # Find FEM cache
    cache_path = find_latest_cache(STANDING_CACHE_DIR)
    coords, p_fem = load_fem_cache(cache_path)
    n_dofs = len(p_fem)
    log(f"FEM cache: {cache_path.name}  ({n_dofs} DOFs)", log_fp)
    log(f"z* = {Z_STAR * 1e3:.4f} mm", log_fp)
    log(f"λ = {LAM * 1e3:.4f} mm, λ/2 = {TRAP_SP * 1e3:.4f} mm", log_fp)

    config = {
        "timestamp": timestamp,
        "cache_path": str(cache_path),
        "n_dofs": n_dofs,
        "z_star_mm": Z_STAR * 1e3,
        "lambda_mm": LAM * 1e3,
        "trap_sp_mm": TRAP_SP * 1e3,
        "roi_half_mm": ROI_HALF * 1e3,
        "dz_grad_um": DZ_GRAD * 1e6,
        "resolutions": RESOLUTION_CONFIGS,
        "f1": F1,
        "f2": F2,
        "particle_radius_um": A_P * 1e6,
    }

    # Save config
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ══ Resolution convergence study ══
    log("\n═══ Resolution convergence study ═══", log_fp)
    convergence_results = []
    results_by_label = {}

    for label, ngrid in RESOLUTION_CONFIGS.items():
        log(f"\n  Running {label} (N={ngrid})...", log_fp)
        r = run_validation(ngrid, cache_path, Z_STAR, ROI_HALF, label_tag=label)
        log(f"    {len(r['accepted'])} accepted, {len(r['rejected'])} rejected "
            f"({r['elapsed_s']:.1f}s)", log_fp)
        convergence_results.append(r)
        results_by_label[label] = r

    # Primary result is medium
    primary = results_by_label["medium"]

    # ══ Z-plane sensitivity ══
    log("\n═══ Z-plane sensitivity ═══", log_fp)
    z_offsets = [-0.10, -0.05, 0.0, 0.05, 0.10]
    zplane_results = run_zplane_sensitivity(
        cache_path, 400, ROI_HALF, z_offsets
    )
    for zr in zplane_results:
        log(f"  z offset {zr['dz_frac_lam']:+.2f}λ: {zr['n_accepted']} traps", log_fp)

    # ══ Write CSVs ══
    log("\n═══ Writing CSVs ═══", log_fp)
    write_csv(out_dir / "traps_all_candidates.csv",
              primary["all_candidates"], CANDIDATE_FIELDS)
    write_csv(out_dir / "traps_accepted.csv",
              primary["accepted"], CANDIDATE_FIELDS)
    write_csv(out_dir / "traps_rejected.csv",
              primary["rejected"], CANDIDATE_FIELDS)
    write_csv(out_dir / "comparison_old_vs_new.csv",
              primary["comparison_old_new"], COMPARISON_FIELDS)
    write_csv(out_dir / "comparison_pressure_vs_potential.csv",
              primary["comparison_p_vs_u"], P_VS_U_FIELDS)

    # Convergence summary CSV
    conv_rows = []
    for cr in convergence_results:
        row = {
            "label": cr["label"],
            "ngrid": cr["ngrid"],
            "dx_um": cr["dx_um"],
            "pts_per_lambda": cr["pts_per_lambda"],
            "n_accepted": len(cr["accepted"]),
            "n_rejected": len(cr["rejected"]),
            "elapsed_s": cr["elapsed_s"],
        }
        conv_rows.append(row)
    with open(out_dir / "convergence_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(conv_rows[0].keys()))
        w.writeheader()
        w.writerows(conv_rows)

    # ══ Generate figures ══
    log("\n═══ Generating figures ═══", log_fp)
    plot_field_overlays(primary, fig_dir)
    plot_local_zoom(primary, fig_dir)
    plot_convergence(convergence_results, fig_dir)
    plot_symmetry_nn(primary, fig_dir)

    # ══ Save field data for inspection ══
    log("\n═══ Saving field cache ═══", log_fp)
    np.savez_compressed(
        out_dir / "baseline_fields.npz",
        xg=primary["xg"],
        yg=primary["yg"],
        p_real=np.real(primary["p_grid"]),
        p_imag=np.imag(primary["p_grid"]),
        U=primary["U"],
        Fx=primary["Fx"],
        Fy=primary["Fy"],
    )

    # ══ Report ══
    log("\n═══ Generating report ═══", log_fp)
    report_path = generate_report(
        out_dir, primary, convergence_results, zplane_results, config
    )
    log(f"\nReport: {report_path}", log_fp)

    total_time = time.time() - t_start
    log(f"\nTotal runtime: {total_time:.1f}s", log_fp)
    log(f"Output directory: {out_dir}", log_fp)

    log_fp.close()
    print(f"\n{'='*60}")
    print(f"DONE — see {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
