#!/usr/bin/env python3
"""
Trap Localisation Debug Study — Standing Wave Only
====================================================

Systematic diagnosis of WHY the trap detection pipeline produces unstable
trap counts across resolution (7/15/17 at 200/400/800) and bizarre
z-plane sensitivity (0 traps at +0.05λ, 13–15 at adjacent offsets).

Standing-wave-only. No vortex, no bridge, no combined-field work.

Outputs → results/trap_localisation_debug_standing_YYYYMMDD_HHMMSS/

Usage:
    python scripts/experiments/trap_localisation_debug_standing.py
"""
from __future__ import annotations

import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.ndimage import minimum_filter, maximum_filter
from scipy.spatial import cKDTree

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Project root ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ═══════════════════════════════════════════════════════════════════
# Physical constants (identical to bridge_master_study.py)
# ═══════════════════════════════════════════════════════════════════
C_WATER = 1484.0
F_HZ    = 2.0e6
K_WATER = 2.0 * np.pi * F_HZ / C_WATER
LAM     = C_WATER / F_HZ            # 0.742 mm
TRAP_SP = LAM / 2.0                  # ~0.371 mm
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

DZ_GRAD  = LAM / 15.0   # ~49.5 µm
K_IDW    = 16

ROI_HALF = 1.1 * LAM
STANDING_CACHE_DIR = PROJECT_ROOT / "results" / "fem_standing_wave_cache"


# ═══════════════════════════════════════════════════════════════════
# Logging
# ═══════════════════════════════════════════════════════════════════
_LOG_FP = None

def log(msg=""):
    print(msg)
    if _LOG_FP:
        _LOG_FP.write(msg + "\n")
        _LOG_FP.flush()


# ═══════════════════════════════════════════════════════════════════
# FEM I/O and interpolation
# ═══════════════════════════════════════════════════════════════════

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


def sample_idw(tree, p, pts, k=16, power=2.0, eps=1e-12):
    if k == 1:
        d, i = tree.query(pts, k=1)
        return p[i]
    d, i = tree.query(pts, k=k)
    w = 1.0 / (d**power + eps)
    w /= w.sum(axis=1, keepdims=True)
    return (p[i] * w).sum(axis=1)


def interpolate_fem_to_grid(tree, p_fem, xg, yg, z_val):
    XX, YY = np.meshgrid(xg, yg)
    pts_3d = np.column_stack([XX.ravel(), YY.ravel(),
                              np.full(XX.size, z_val)])
    p_flat = sample_idw(tree, p_fem, pts_3d, k=K_IDW)
    return p_flat.reshape(XX.shape)


# ═══════════════════════════════════════════════════════════════════
# Gor'kov computation
# ═══════════════════════════════════════════════════════════════════

def gorkov_potential_from_fields(p_abs2, grad_p_mag2):
    E_pot = 0.25 * p_abs2 * KAPPA0
    E_kin = grad_p_mag2 / (4.0 * OMEGA**2 * RHO0)
    return V_P * (F1 * E_pot - 1.5 * F2 * E_kin)


def gorkov_and_force_on_grid(p_grid, dx, dy, p_zp=None, p_zm=None, dz=None):
    p_abs2 = np.abs(p_grid)**2
    dpy, dpx = np.gradient(p_grid, dy, dx)
    grad_p_mag2 = np.abs(dpx)**2 + np.abs(dpy)**2

    if p_zp is not None and p_zm is not None and dz is not None:
        dpz = (p_zp - p_zm) / (2.0 * dz)
        grad_p_mag2 += np.abs(dpz)**2

    U = gorkov_potential_from_fields(p_abs2, grad_p_mag2)
    Fx = -np.gradient(U, dx, axis=1)
    Fy = -np.gradient(U, dy, axis=0)
    return U, Fx, Fy, grad_p_mag2


# ═══════════════════════════════════════════════════════════════════
# Candidate detection (with stage-by-stage logging)
# ═══════════════════════════════════════════════════════════════════

def detect_candidates(U, xg, yg, min_sep_physical=None):
    """Morphological minimum-filter detection."""
    dx = xg[1] - xg[0]
    if min_sep_physical is None:
        min_sep_physical = 0.4 * TRAP_SP
    min_distance_pts = max(3, int(min_sep_physical / dx))
    neighborhood = 2 * min_distance_pts + 1
    local_min = minimum_filter(U, size=neighborhood)
    minima_mask = (U == local_min)
    # Exclude 2-pixel border
    minima_mask[:2, :] = False
    minima_mask[-2:, :] = False
    minima_mask[:, :2] = False
    minima_mask[:, -2:] = False
    iy, ix = np.where(minima_mask)
    return iy, ix, {"min_distance_pts": min_distance_pts,
                     "neighborhood": neighborhood}


def refine_subgrid(U, xg, yg, iy0, ix0, half_win=4):
    """Local quadratic (paraboloid) fit for sub-grid refinement."""
    ny, nx = U.shape
    iy_lo = max(iy0 - half_win, 0)
    iy_hi = min(iy0 + half_win + 1, ny)
    ix_lo = max(ix0 - half_win, 0)
    ix_hi = min(ix0 + half_win + 1, nx)

    x_local = xg[ix_lo:ix_hi]
    y_local = yg[iy_lo:iy_hi]
    U_local = U[iy_lo:iy_hi, ix_lo:ix_hi]

    if U_local.size < 6:
        return None

    x_cen = xg[ix0]
    y_cen = yg[iy0]

    XL, YL = np.meshgrid(x_local - x_cen, y_local - y_cen)
    xl = XL.ravel()
    yl = YL.ravel()
    ul = U_local.ravel()

    A = np.column_stack([np.ones_like(xl), xl, yl, xl**2, yl**2, xl * yl])
    try:
        coeffs, _, rank, sv = np.linalg.lstsq(A, ul, rcond=None)
    except np.linalg.LinAlgError:
        return None

    a, b, c, d, e, f = coeffs
    H = np.array([[2 * d, f], [f, 2 * e]])
    eig_vals = np.linalg.eigvalsh(H)

    is_true_min = bool(eig_vals[0] > 0)

    grad = np.array([b, c])
    try:
        shift = -np.linalg.solve(H, grad)
    except np.linalg.LinAlgError:
        return None

    x_ref = x_cen + shift[0]
    y_ref = y_cen + shift[1]

    dx_half = (x_local[-1] - x_local[0]) / 2
    dy_half = (y_local[-1] - y_local[0]) / 2
    if abs(shift[0]) > dx_half or abs(shift[1]) > dy_half:
        return None

    U_fit = float(coeffs @ np.array([1, shift[0], shift[1],
                                      shift[0]**2, shift[1]**2,
                                      shift[0] * shift[1]]))
    fit_res = float(np.sqrt(np.mean((A @ coeffs - ul)**2)))
    return (x_ref, y_ref, U_fit, fit_res, eig_vals, is_true_min)


def compute_local_depth(U, xg, yg, iy, ix, search_radius_pts=15):
    ny, nx = U.shape
    U_min = U[iy, ix]
    iy_lo = max(iy - search_radius_pts, 0)
    iy_hi = min(iy + search_radius_pts + 1, ny)
    ix_lo = max(ix - search_radius_pts, 0)
    ix_hi = min(ix + search_radius_pts + 1, nx)
    U_patch = U[iy_lo:iy_hi, ix_lo:ix_hi]
    YP, XP = np.meshgrid(
        np.arange(iy_lo, iy_hi) - iy,
        np.arange(ix_lo, ix_hi) - ix, indexing="ij")
    r = np.sqrt(XP**2 + YP**2)
    ring_mask = (r >= 0.6 * search_radius_pts) & (r <= search_radius_pts)
    if ring_mask.sum() == 0:
        return 0.0
    U_ring_max = U_patch[ring_mask].max()
    return float(U_ring_max - U_min)


# ═══════════════════════════════════════════════════════════════════
# Full pipeline with stage-by-stage attrition reporting
# ═══════════════════════════════════════════════════════════════════

def run_pipeline_with_attrition(tree, p_fem, xg, yg, z_val, label="",
                                 force_thresholds=(0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0)):
    """
    Run the full trap detection pipeline, logging every filtering stage.

    Returns dict with:
      - fields (U, Fx, Fy, etc.)
      - stage-by-stage attrition table
      - candidate details at every stage
      - per-threshold trap counts for force criterion sweep
    """
    dx = xg[1] - xg[0]
    dy = yg[1] - yg[0]
    ngrid = len(xg)

    # ── Interpolate ──
    p_grid = interpolate_fem_to_grid(tree, p_fem, xg, yg, z_val)
    p_zp = interpolate_fem_to_grid(tree, p_fem, xg, yg, z_val + DZ_GRAD)
    p_zm = interpolate_fem_to_grid(tree, p_fem, xg, yg, z_val - DZ_GRAD)

    # ── Gor'kov and force ──
    U, Fx, Fy, grad_p_mag2 = gorkov_and_force_on_grid(
        p_grid, dx, dy, p_zp=p_zp, p_zm=p_zm, dz=DZ_GRAD)
    F_mag = np.sqrt(Fx**2 + Fy**2)
    p_abs = np.abs(p_grid)

    # Also compute U without z-gradient for comparison
    U_no_z, _, _, _ = gorkov_and_force_on_grid(p_grid, dx, dy)

    attrition = []  # list of (stage_name, n_remaining, n_dropped, details)

    # ── Stage 0: Raw morphological candidate detection ──
    iy_cand, ix_cand, detect_info = detect_candidates(U, xg, yg)
    n_raw = len(iy_cand)
    attrition.append(("0_morphological_detection", n_raw, 0,
                       f"neighborhood={detect_info['neighborhood']}px, "
                       f"min_dist={detect_info['min_distance_pts']}px"))

    # Build full candidate table
    candidates = []
    for j in range(n_raw):
        iy_j, ix_j = int(iy_cand[j]), int(ix_cand[j])
        x_g = float(xg[ix_j])
        y_g = float(yg[iy_j])
        U_val = float(U[iy_j, ix_j])
        F_val = float(F_mag[iy_j, ix_j])
        U_no_z_val = float(U_no_z[iy_j, ix_j])

        depth = compute_local_depth(U, xg, yg, iy_j, ix_j,
                                    search_radius_pts=max(8, int(TRAP_SP / dx * 0.6)))

        ref = refine_subgrid(U, xg, yg, iy_j, ix_j, half_win=4)
        if ref is not None:
            x_ref, y_ref, U_ref, fit_res, eig_vals, is_true_min = ref
        else:
            x_ref, y_ref = x_g, y_g
            U_ref = U_val
            fit_res = np.nan
            eig_vals = np.array([np.nan, np.nan])
            is_true_min = False

        # FD Hessian
        if 2 <= iy_j < U.shape[0] - 2 and 2 <= ix_j < U.shape[1] - 2:
            Hxx = (U[iy_j, ix_j + 1] - 2 * U[iy_j, ix_j] + U[iy_j, ix_j - 1]) / dx**2
            Hyy = (U[iy_j + 1, ix_j] - 2 * U[iy_j, ix_j] + U[iy_j - 1, ix_j]) / dy**2
            Hxy = (U[iy_j + 1, ix_j + 1] - U[iy_j + 1, ix_j - 1]
                   - U[iy_j - 1, ix_j + 1] + U[iy_j - 1, ix_j - 1]) / (4 * dx * dy)
            eig_fd = np.linalg.eigvalsh(np.array([[Hxx, Hxy], [Hxy, Hyy]]))
        else:
            Hxx = Hyy = Hxy = np.nan
            eig_fd = np.array([np.nan, np.nan])

        # Wide-stencil FD Hessian (±2 pixels) for comparison
        if 4 <= iy_j < U.shape[0] - 4 and 4 <= ix_j < U.shape[1] - 4:
            h = 2 * dx
            Hxx_w = (U[iy_j, ix_j + 2] - 2 * U[iy_j, ix_j] + U[iy_j, ix_j - 2]) / h**2
            Hyy_w = (U[iy_j + 2, ix_j] - 2 * U[iy_j, ix_j] + U[iy_j - 2, ix_j]) / h**2
            Hxy_w = (U[iy_j + 2, ix_j + 2] - U[iy_j + 2, ix_j - 2]
                     - U[iy_j - 2, ix_j + 2] + U[iy_j - 2, ix_j - 2]) / (4 * h**2)
            eig_fd_w = np.linalg.eigvalsh(np.array([[Hxx_w, Hxy_w], [Hxy_w, Hyy_w]]))
        else:
            eig_fd_w = np.array([np.nan, np.nan])

        shift = np.sqrt((x_ref - x_g)**2 + (y_ref - y_g)**2)
        force_frac = F_val / (F_mag.max() + 1e-30)

        candidates.append({
            "idx": j,
            "ix": ix_j, "iy": iy_j,
            "x_grid_mm": x_g * 1e3, "y_grid_mm": y_g * 1e3,
            "x_ref_mm": x_ref * 1e3, "y_ref_mm": y_ref * 1e3,
            "shift_mm": shift * 1e3,
            "U_grid": U_val, "U_ref": U_ref,
            "U_no_z": U_no_z_val,
            "depth": depth,
            "grad_norm": F_val,
            "force_frac": force_frac,
            "eig_fd_0": float(eig_fd[0]), "eig_fd_1": float(eig_fd[1]),
            "eig_fd_wide_0": float(eig_fd_w[0]), "eig_fd_wide_1": float(eig_fd_w[1]),
            "eig_fit_0": float(eig_vals[0]), "eig_fit_1": float(eig_vals[1]),
            "fit_residual": float(fit_res) if not np.isnan(fit_res) else None,
            "is_true_min_fit": is_true_min,
            "refine_ok": ref is not None,
            # Tracking which stages remove this candidate
            "removed_at": None,
            "removal_reason": None,
        })

    # ── Stage 1: Boundary proximity (5 pixels) ──
    border_pts = 5
    alive = list(range(len(candidates)))
    new_alive = []
    n_before = len(alive)
    for j in alive:
        c = candidates[j]
        if (c["ix"] < border_pts or c["ix"] >= ngrid - border_pts or
                c["iy"] < border_pts or c["iy"] >= ngrid - border_pts):
            c["removed_at"] = "1_boundary"
            c["removal_reason"] = "boundary_proximity"
        else:
            new_alive.append(j)
    alive = new_alive
    attrition.append(("1_boundary_filter", len(alive), n_before - len(alive), ""))

    # ── Stage 2: Hessian check (FD) ──
    n_before = len(alive)
    new_alive = []
    for j in alive:
        c = candidates[j]
        if not np.isnan(c["eig_fd_0"]) and c["eig_fd_0"] <= 0:
            c["removed_at"] = "2_hessian_fd"
            c["removal_reason"] = f"eig_fd_0={c['eig_fd_0']:.3e}"
        else:
            new_alive.append(j)
    alive = new_alive
    attrition.append(("2_hessian_fd", len(alive), n_before - len(alive), ""))

    # ── Stage 3: Quadratic fit true-minimum check ──
    n_before = len(alive)
    new_alive = []
    for j in alive:
        c = candidates[j]
        if c["refine_ok"] and not c["is_true_min_fit"]:
            c["removed_at"] = "3_fit_not_minimum"
            c["removal_reason"] = f"eig_fit_0={c['eig_fit_0']:.3e}"
        else:
            new_alive.append(j)
    alive = new_alive
    attrition.append(("3_quadratic_fit_min", len(alive), n_before - len(alive), ""))

    # ── Stage 4: Force criterion sweep (report for ALL thresholds, apply at 0.10) ──
    force_sweep = {}
    for thr in force_thresholds:
        n_pass = sum(1 for j in alive if candidates[j]["force_frac"] <= thr)
        force_sweep[thr] = n_pass

    n_before = len(alive)
    new_alive = []
    for j in alive:
        c = candidates[j]
        if c["force_frac"] > 0.10:
            c["removed_at"] = "4_force_residual"
            c["removal_reason"] = f"force_frac={c['force_frac']:.4f}"
        else:
            new_alive.append(j)
    alive = new_alive
    attrition.append(("4_force_0.10", len(alive), n_before - len(alive),
                       f"sweep: {force_sweep}"))

    # ── Stage 5: Depth filter (1% of max depth) ──
    n_before = len(alive)
    if alive:
        max_depth = max(candidates[j]["depth"] for j in alive)
    else:
        max_depth = 0
    new_alive = []
    for j in alive:
        c = candidates[j]
        if max_depth > 0 and c["depth"] < 0.01 * max_depth:
            c["removed_at"] = "5_too_shallow"
            c["removal_reason"] = f"depth_frac={c['depth'] / max_depth:.4f}"
        else:
            new_alive.append(j)
    alive = new_alive
    attrition.append(("5_depth_filter", len(alive), n_before - len(alive), ""))

    # ── Stage 6: Deduplication (0.3 * trap_sp) ──
    n_before = len(alive)
    alive_sorted = sorted(alive, key=lambda j: candidates[j]["U_ref"])
    dedup_dist = 0.3 * TRAP_SP * 1e3  # mm
    keep = set()
    for j in alive_sorted:
        c = candidates[j]
        too_close = False
        for k in keep:
            ck = candidates[k]
            dist = np.sqrt((c["x_ref_mm"] - ck["x_ref_mm"])**2 +
                           (c["y_ref_mm"] - ck["y_ref_mm"])**2)
            if dist < dedup_dist:
                too_close = True
                break
        if too_close:
            c["removed_at"] = "6_dedup"
            c["removal_reason"] = f"dist={dist:.4f}mm"
        else:
            keep.add(j)
    alive = list(keep)
    attrition.append(("6_deduplication", len(alive), n_before - len(alive), ""))

    accepted = [candidates[j] for j in alive]
    rejected = [c for c in candidates if c["removed_at"] is not None]

    # NN distances
    if len(accepted) >= 2:
        pos = np.array([[c["x_ref_mm"], c["y_ref_mm"]] for c in accepted])
        nn_tree = cKDTree(pos)
        dd, ii = nn_tree.query(pos, k=2)
        for j, c in enumerate(accepted):
            c["nn_dist_mm"] = float(dd[j, 1])
    else:
        for c in accepted:
            c["nn_dist_mm"] = np.nan

    return {
        "label": label,
        "ngrid": ngrid,
        "z_val": z_val,
        "dx_um": dx * 1e6,
        "xg": xg, "yg": yg,
        "p_grid": p_grid, "p_abs": p_abs,
        "U": U, "U_no_z": U_no_z,
        "Fx": Fx, "Fy": Fy, "F_mag": F_mag,
        "candidates": candidates,
        "accepted": accepted,
        "rejected": rejected,
        "attrition": attrition,
        "force_sweep": force_sweep,
    }


# ═══════════════════════════════════════════════════════════════════
# Test 1: Z-plane dense sweep
# ═══════════════════════════════════════════════════════════════════

def run_zplane_dense_sweep(tree, p_fem, ngrid, roi_half,
                            dz_fracs=None):
    """
    Dense z-plane sweep from -0.12λ to +0.12λ in 0.01λ steps.
    For each z-plane, run full pipeline and report attrition.
    """
    if dz_fracs is None:
        dz_fracs = np.arange(-0.12, 0.121, 0.01)

    xg = np.linspace(CX - roi_half, CX + roi_half, ngrid)
    yg = np.linspace(CY - roi_half, CY + roi_half, ngrid)

    results = []
    for dz_frac in dz_fracs:
        z_val = Z_STAR + dz_frac * LAM
        r = run_pipeline_with_attrition(
            tree, p_fem, xg, yg, z_val,
            label=f"z={dz_frac:+.2f}λ")
        results.append({
            "dz_frac": float(dz_frac),
            "z_mm": z_val * 1e3,
            "n_raw": r["attrition"][0][1],       # morphological
            "n_after_boundary": r["attrition"][1][1],
            "n_after_hessian": r["attrition"][2][1],
            "n_after_fit": r["attrition"][3][1],
            "n_after_force": r["attrition"][4][1],
            "n_after_depth": r["attrition"][5][1],
            "n_accepted": r["attrition"][6][1],   # after dedup
            "force_sweep": r["force_sweep"],
            "positions": [(c["x_ref_mm"], c["y_ref_mm"]) for c in r["accepted"]],
            "U_range": (float(r["U"].min()), float(r["U"].max())),
            "F_max": float(r["F_mag"].max()),
            "attrition": r["attrition"],
        })
        log(f"  z={dz_frac:+.3f}λ: raw={results[-1]['n_raw']}, "
            f"→boundary={results[-1]['n_after_boundary']}, "
            f"→hessian={results[-1]['n_after_hessian']}, "
            f"→fit={results[-1]['n_after_fit']}, "
            f"→force={results[-1]['n_after_force']}, "
            f"→depth={results[-1]['n_after_depth']}, "
            f"→dedup={results[-1]['n_accepted']}")
    return results


# ═══════════════════════════════════════════════════════════════════
# Test 2: Resolution convergence with robust matching
# ═══════════════════════════════════════════════════════════════════

def run_resolution_convergence(tree, p_fem, z_val, roi_half,
                                ngrids=(100, 150, 200, 300, 400, 600, 800)):
    """Run pipeline at many resolutions, cross-match traps."""
    results = []
    for ngrid in ngrids:
        xg = np.linspace(CX - roi_half, CX + roi_half, ngrid)
        yg = np.linspace(CY - roi_half, CY + roi_half, ngrid)
        r = run_pipeline_with_attrition(tree, p_fem, xg, yg, z_val,
                                         label=f"N={ngrid}")
        results.append(r)
        log(f"  N={ngrid}: dx={r['dx_um']:.1f}µm, "
            f"raw={r['attrition'][0][1]}, accepted={len(r['accepted'])}")
    return results


# ═══════════════════════════════════════════════════════════════════
# Test 3: DZ_GRAD sensitivity
# ═══════════════════════════════════════════════════════════════════

def run_dz_grad_sensitivity(tree, p_fem, ngrid, roi_half, z_val):
    """
    Test how DZ_GRAD (z-gradient offset) affects trap detection.
    Sweep DZ_GRAD from λ/50 to λ/5.
    """
    xg = np.linspace(CX - roi_half, CX + roi_half, ngrid)
    yg = np.linspace(CY - roi_half, CY + roi_half, ngrid)
    dx = xg[1] - xg[0]
    dy = yg[1] - yg[0]

    p_grid = interpolate_fem_to_grid(tree, p_fem, xg, yg, z_val)
    dz_fracs = [1/50, 1/30, 1/20, 1/15, 1/10, 1/7, 1/5]

    results = []
    for dz_frac in dz_fracs:
        dz = dz_frac * LAM
        p_zp = interpolate_fem_to_grid(tree, p_fem, xg, yg, z_val + dz)
        p_zm = interpolate_fem_to_grid(tree, p_fem, xg, yg, z_val - dz)
        U, Fx, Fy, gpm2 = gorkov_and_force_on_grid(p_grid, dx, dy,
                                                     p_zp=p_zp, p_zm=p_zm, dz=dz)
        F_mag = np.sqrt(Fx**2 + Fy**2)

        # Also compute U without z-gradient
        U_no_z, _, _, _ = gorkov_and_force_on_grid(p_grid, dx, dy)

        # Detect and count
        iy_c, ix_c, _ = detect_candidates(U, xg, yg)

        # Quick classification (same stages as pipeline)
        n_raw = len(iy_c)
        n_hessian_pass = 0
        n_force_pass = 0
        for k in range(n_raw):
            iy_k, ix_k = int(iy_c[k]), int(ix_c[k])
            if 2 <= iy_k < U.shape[0] - 2 and 2 <= ix_k < U.shape[1] - 2:
                Hxx = (U[iy_k, ix_k+1] - 2*U[iy_k, ix_k] + U[iy_k, ix_k-1]) / dx**2
                Hyy = (U[iy_k+1, ix_k] - 2*U[iy_k, ix_k] + U[iy_k-1, ix_k]) / dy**2
                Hxy = (U[iy_k+1, ix_k+1] - U[iy_k+1, ix_k-1]
                       - U[iy_k-1, ix_k+1] + U[iy_k-1, ix_k-1]) / (4*dx*dy)
                eigs = np.linalg.eigvalsh(np.array([[Hxx, Hxy], [Hxy, Hyy]]))
                if eigs[0] > 0:
                    n_hessian_pass += 1
                    frac = F_mag[iy_k, ix_k] / (F_mag.max() + 1e-30)
                    if frac <= 0.10:
                        n_force_pass += 1

        # Correlation between U and U_no_z
        corr = np.corrcoef(U.ravel(), U_no_z.ravel())[0, 1]
        diff_rms = np.sqrt(np.mean((U - U_no_z)**2))

        results.append({
            "dz_frac_lam": dz_frac,
            "dz_um": dz * 1e6,
            "n_raw": n_raw,
            "n_hessian_pass": n_hessian_pass,
            "n_force_pass": n_force_pass,
            "U_min": float(U.min()),
            "U_max": float(U.max()),
            "F_max": float(F_mag.max()),
            "corr_with_no_z": corr,
            "diff_rms_U": diff_rms,
        })
        log(f"  dz={dz_frac:.3f}λ ({dz*1e6:.1f}µm): raw={n_raw}, "
            f"hessian_ok={n_hessian_pass}, force_ok={n_force_pass}, "
            f"corr_no_z={corr:.6f}")

    return results


# ═══════════════════════════════════════════════════════════════════
# Test 4: Force criterion analysis
# ═══════════════════════════════════════════════════════════════════

def run_force_criterion_analysis(result):
    """
    Analyse the force criterion in detail: histogram, bimodal check,
    refined-position force, alternative normalisations.
    """
    candidates = result["candidates"]
    U = result["U"]
    xg = result["xg"]
    yg = result["yg"]
    dx = xg[1] - xg[0]
    dy = yg[1] - yg[0]
    Fx = result["Fx"]
    Fy = result["Fy"]
    F_mag = result["F_mag"]

    analysis = {
        "n_total": len(candidates),
        "force_fracs": [],
        "force_norms": [],
        "labels": [],        # "accepted" or removal stage
        "refined_force_fracs": [],
    }

    from scipy.interpolate import RegularGridInterpolator

    rgi_Fx = RegularGridInterpolator((yg, xg), Fx,
                                     bounds_error=False, fill_value=np.nan)
    rgi_Fy = RegularGridInterpolator((yg, xg), Fy,
                                     bounds_error=False, fill_value=np.nan)

    for c in candidates:
        analysis["force_fracs"].append(c["force_frac"])
        analysis["force_norms"].append(c["grad_norm"])
        lab = c["removed_at"] if c["removed_at"] else "accepted"
        analysis["labels"].append(lab)

        # Force at REFINED position
        x_r = c["x_ref_mm"] / 1e3
        y_r = c["y_ref_mm"] / 1e3
        fx_ref = float(rgi_Fx(np.array([[y_r, x_r]]))[0])
        fy_ref = float(rgi_Fy(np.array([[y_r, x_r]]))[0])
        f_ref = np.sqrt(fx_ref**2 + fy_ref**2)
        analysis["refined_force_fracs"].append(f_ref / (F_mag.max() + 1e-30))

    return analysis


# ═══════════════════════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════════════════════

def plot_zplane_attrition(zplane_results, fig_dir):
    """Figure 1: Z-plane sweep — trap count by stage."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    dz = [r["dz_frac"] for r in zplane_results]

    # A1: Accepted traps vs z-offset
    ax = axes[0, 0]
    ax.plot(dz, [r["n_accepted"] for r in zplane_results], "o-", lw=2,
            label="Accepted (final)")
    ax.plot(dz, [r["n_raw"] for r in zplane_results], "s--", alpha=0.5,
            label="Raw candidates")
    ax.plot(dz, [r["n_after_force"] for r in zplane_results], "^--", alpha=0.5,
            label="After force filter")
    ax.set_xlabel("z offset / λ")
    ax.set_ylabel("Trap count")
    ax.set_title("Trap count vs z-plane offset")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # A2: Stage-by-stage attrition
    ax = axes[0, 1]
    stages = ["n_raw", "n_after_boundary", "n_after_hessian",
              "n_after_fit", "n_after_force", "n_after_depth", "n_accepted"]
    stage_labels = ["Raw", "Boundary", "Hessian", "Fit", "Force",
                     "Depth", "Dedup"]
    for i, (key, slab) in enumerate(zip(stages, stage_labels)):
        vals = [r[key] for r in zplane_results]
        ax.plot(dz, vals, "o-", label=slab, alpha=0.7)
    ax.set_xlabel("z offset / λ")
    ax.set_ylabel("Count remaining")
    ax.set_title("Stage-by-stage attrition vs z-plane")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # A3: Force threshold sweep
    ax = axes[1, 0]
    thresholds = sorted(zplane_results[0]["force_sweep"].keys())
    for thr in thresholds:
        vals = [r["force_sweep"].get(thr, 0) for r in zplane_results]
        ax.plot(dz, vals, "o-", label=f"thr={thr:.2f}", alpha=0.7)
    ax.set_xlabel("z offset / λ")
    ax.set_ylabel("# passing force filter")
    ax.set_title("Force threshold sweep vs z-plane")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # A4: Field properties vs z
    ax = axes[1, 1]
    ax2 = ax.twinx()
    ax.plot(dz, [r["U_range"][0] for r in zplane_results], "b-o",
            label="U_min", ms=3)
    ax.plot(dz, [r["U_range"][1] for r in zplane_results], "r-o",
            label="U_max", ms=3)
    ax2.plot(dz, [r["F_max"] for r in zplane_results], "g-^",
             label="F_max", ms=3)
    ax.set_xlabel("z offset / λ")
    ax.set_ylabel("U [J]")
    ax2.set_ylabel("F_max [N]")
    ax.set_title("Field properties vs z-plane")
    ax.legend(loc="upper left", fontsize=7)
    ax2.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Z-Plane Dense Sweep Diagnostics", fontsize=14)
    fig.tight_layout()
    fig.savefig(fig_dir / "zplane_dense_sweep.png", dpi=150)
    plt.close(fig)


def plot_resolution_convergence(res_results, fig_dir):
    """Figure 2: Resolution convergence — attrition by stage."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    ngrids = [r["ngrid"] for r in res_results]
    # dx values
    dxs = [r["dx_um"] for r in res_results]

    # B1: Accepted count vs resolution
    ax = axes[0, 0]
    n_acc = [len(r["accepted"]) for r in res_results]
    ax.plot(ngrids, n_acc, "o-", lw=2, color="steelblue")
    for i, (ng, na) in enumerate(zip(ngrids, n_acc)):
        ax.annotate(f"{na}", (ng, na), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
    ax.set_xlabel("Grid points per side")
    ax.set_ylabel("Accepted traps")
    ax.set_title("Converged trap count vs resolution")
    ax.grid(True, alpha=0.3)

    # B2: Stage-by-stage
    ax = axes[0, 1]
    stage_names = ["Raw", "Boundary", "Hessian", "Fit", "Force",
                    "Depth", "Dedup"]
    for si in range(len(stage_names)):
        vals = [r["attrition"][si][1] for r in res_results]
        ax.plot(ngrids, vals, "o-", label=stage_names[si], alpha=0.7)
    ax.set_xlabel("Grid points per side")
    ax.set_ylabel("Count remaining")
    ax.set_title("Stage-by-stage attrition vs resolution")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # B3: Cross-resolution matching
    ax = axes[1, 0]
    if len(res_results) >= 2:
        finest = res_results[-1]
        if len(finest["accepted"]) > 0:
            pos_fine = np.array([[c["x_ref_mm"], c["y_ref_mm"]]
                                 for c in finest["accepted"]])
            tree_fine = cKDTree(pos_fine)
            for r in res_results[:-1]:
                if len(r["accepted"]) == 0:
                    continue
                pos_r = np.array([[c["x_ref_mm"], c["y_ref_mm"]]
                                  for c in r["accepted"]])
                dd, _ = tree_fine.query(pos_r)
                matched = dd < TRAP_SP * 1e3 * 0.5
                ax.bar(r["ngrid"], matched.sum(), width=20, alpha=0.6,
                       label=f"N={r['ngrid']}")
                # annotate mean shift of matched
                if matched.sum() > 0:
                    ax.annotate(f"mean shift\n{dd[matched].mean()*1e3:.1f}µm",
                                (r["ngrid"], matched.sum()),
                                textcoords="offset points",
                                xytext=(0, 8), ha="center", fontsize=7)
            ax.axhline(len(finest["accepted"]), color="red", ls="--",
                       label=f"Fine ({len(finest['accepted'])} traps)")
            ax.set_xlabel("Grid resolution")
            ax.set_ylabel("# matched to finest")
            ax.set_title("Cross-resolution matching")
            ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # B4: Position shift histogram (all resolutions vs finest)
    ax = axes[1, 1]
    if len(res_results) >= 2:
        finest = res_results[-1]
        if len(finest["accepted"]) > 0:
            pos_fine = np.array([[c["x_ref_mm"], c["y_ref_mm"]]
                                 for c in finest["accepted"]])
            tree_fine = cKDTree(pos_fine)
            for r in res_results[:-1]:
                if len(r["accepted"]) == 0:
                    continue
                pos_r = np.array([[c["x_ref_mm"], c["y_ref_mm"]]
                                  for c in r["accepted"]])
                dd, _ = tree_fine.query(pos_r)
                ax.hist(dd * 1e3, bins=20, alpha=0.5,
                        label=f"N={r['ngrid']} (n={len(pos_r)})")
            ax.set_xlabel("Position shift to finest [µm]")
            ax.set_ylabel("Count")
            ax.set_title("Position shifts vs finest resolution")
            ax.axvline(TRAP_SP * 1e3 * 0.01 * 1e3, color="red", ls="--",
                       lw=0.8, label="1% λ/2")
            ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Resolution Convergence Diagnostics", fontsize=14)
    fig.tight_layout()
    fig.savefig(fig_dir / "resolution_convergence.png", dpi=150)
    plt.close(fig)


def plot_force_criterion(force_analysis, fig_dir):
    """Figure 3: Force criterion analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    fracs = np.array(force_analysis["force_fracs"])
    labels = force_analysis["labels"]
    ref_fracs = np.array(force_analysis["refined_force_fracs"])

    # C1: force_frac histogram, coloured by acceptance
    ax = axes[0, 0]
    acc_mask = np.array([l == "accepted" for l in labels])
    bins = np.linspace(0, max(fracs.max(), 0.6), 40)
    ax.hist(fracs[acc_mask], bins=bins, alpha=0.7, color="green",
            label=f"Accepted ({acc_mask.sum()})")
    ax.hist(fracs[~acc_mask], bins=bins, alpha=0.7, color="red",
            label=f"Rejected ({(~acc_mask).sum()})")
    ax.axvline(0.10, color="blue", ls="--", lw=2, label="threshold=0.10")
    ax.set_xlabel("Force fraction (|F|/|F|_max)")
    ax.set_ylabel("Count")
    ax.set_title("Force fraction at grid point")
    ax.legend(fontsize=8)

    # C2: force_frac at REFINED position
    ax = axes[0, 1]
    ax.hist(ref_fracs[acc_mask], bins=bins, alpha=0.7, color="green",
            label=f"Accepted ({acc_mask.sum()})")
    ax.hist(ref_fracs[~acc_mask], bins=bins, alpha=0.7, color="red",
            label=f"Rejected ({(~acc_mask).sum()})")
    ax.axvline(0.10, color="blue", ls="--", lw=2, label="threshold=0.10")
    ax.set_xlabel("Force fraction at refined position")
    ax.set_ylabel("Count")
    ax.set_title("Force at refined position (interpolated)")
    ax.legend(fontsize=8)

    # C3: grid-point vs refined-point force
    ax = axes[1, 0]
    ax.scatter(fracs[acc_mask], ref_fracs[acc_mask], c="green", alpha=0.7,
               label="Accepted", s=30)
    ax.scatter(fracs[~acc_mask], ref_fracs[~acc_mask], c="red", alpha=0.7,
               label="Rejected", s=30)
    ax.plot([0, 0.6], [0, 0.6], "k--", lw=0.5)
    ax.axvline(0.10, color="blue", ls="--", lw=0.8)
    ax.axhline(0.10, color="blue", ls="--", lw=0.8)
    ax.set_xlabel("Force frac at grid point")
    ax.set_ylabel("Force frac at refined point")
    ax.set_title("Grid vs refined force fraction")
    ax.legend(fontsize=8)

    # C4: Rejection reason breakdown bar chart
    ax = axes[1, 1]
    reason_counts = {}
    for l in labels:
        if l != "accepted":
            reason_counts[l] = reason_counts.get(l, 0) + 1
    if reason_counts:
        bars = sorted(reason_counts.items(), key=lambda x: -x[1])
        ax.barh([b[0] for b in bars], [b[1] for b in bars], color="coral")
        ax.set_xlabel("Count")
        ax.set_title("Rejection reasons")
    else:
        ax.text(0.5, 0.5, "No rejections", transform=ax.transAxes,
                ha="center")

    fig.suptitle("Force Criterion Analysis", fontsize=14)
    fig.tight_layout()
    fig.savefig(fig_dir / "force_criterion_analysis.png", dpi=150)
    plt.close(fig)


def plot_candidate_map(result, fig_dir, prefix="baseline"):
    """Figure 4: All candidates on U field, colour-coded by rejection stage."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    ext = np.array([result["xg"][0], result["xg"][-1],
                     result["yg"][0], result["yg"][-1]]) * 1e3

    # Left: U with all candidates
    ax = axes[0]
    im = ax.imshow(result["U"], extent=ext, origin="lower", cmap="viridis")
    plt.colorbar(im, ax=ax, label="U [J]")

    stage_colors = {
        "accepted": "lime",
        "1_boundary": "cyan",
        "2_hessian_fd": "orange",
        "3_fit_not_minimum": "magenta",
        "4_force_residual": "red",
        "5_too_shallow": "yellow",
        "6_dedup": "white",
    }
    stage_markers = {
        "accepted": "*",
        "1_boundary": "v",
        "2_hessian_fd": "D",
        "3_fit_not_minimum": "^",
        "4_force_residual": "x",
        "5_too_shallow": "s",
        "6_dedup": "+",
    }

    for c in result["candidates"]:
        lab = c["removed_at"] if c["removed_at"] else "accepted"
        color = stage_colors.get(lab, "gray")
        marker = stage_markers.get(lab, "o")
        ms = 10 if lab == "accepted" else 6
        ax.plot(c["x_grid_mm"], c["y_grid_mm"], marker, color=color,
                ms=ms, alpha=0.8, mew=1.2)

    # Legend
    for lab, col in stage_colors.items():
        ax.plot([], [], stage_markers.get(lab, "o"), color=col,
                ms=8, label=lab, mew=1.2)
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title(f"All candidates by rejection stage ({prefix})")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")

    # Right: U with only accepted, showing refined positions and arrows
    ax = axes[1]
    im = ax.imshow(result["U"], extent=ext, origin="lower", cmap="viridis")
    plt.colorbar(im, ax=ax, label="U [J]")
    for c in result["accepted"]:
        ax.plot(c["x_grid_mm"], c["y_grid_mm"], "s", color="cyan",
                ms=7, mfc="none", mew=1.5)
        ax.plot(c["x_ref_mm"], c["y_ref_mm"], "*", color="yellow",
                ms=12, mew=0.5)
        ax.annotate("", xy=(c["x_ref_mm"], c["y_ref_mm"]),
                     xytext=(c["x_grid_mm"], c["y_grid_mm"]),
                     arrowprops=dict(arrowstyle="->", color="white", lw=0.8))
    ax.set_title(f"Accepted traps: grid→refined ({prefix})")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")

    fig.suptitle(f"Candidate Map — {prefix}", fontsize=14)
    fig.tight_layout()
    fig.savefig(fig_dir / f"candidate_map_{prefix}.png", dpi=150)
    plt.close(fig)


def plot_dz_grad_sensitivity(dz_results, fig_dir):
    """Figure 5: DZ_GRAD sensitivity."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    dz_um = [r["dz_um"] for r in dz_results]

    ax = axes[0]
    ax.plot(dz_um, [r["n_raw"] for r in dz_results], "o-", label="Raw")
    ax.plot(dz_um, [r["n_hessian_pass"] for r in dz_results], "s-",
            label="Hessian OK")
    ax.plot(dz_um, [r["n_force_pass"] for r in dz_results], "^-",
            label="Force OK")
    ax.set_xlabel("DZ_GRAD [µm]")
    ax.set_ylabel("Count")
    ax.set_title("Trap count vs z-gradient offset")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(dz_um, [r["corr_with_no_z"] for r in dz_results], "o-", color="blue")
    ax.set_xlabel("DZ_GRAD [µm]")
    ax.set_ylabel("Correlation with U(no z-grad)")
    ax.set_title("U correlation vs z-gradient offset")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(dz_um, [r["F_max"] for r in dz_results], "o-", color="red")
    ax.set_xlabel("DZ_GRAD [µm]")
    ax.set_ylabel("F_max [N]")
    ax.set_title("Maximum force vs z-gradient offset")
    ax.grid(True, alpha=0.3)

    fig.suptitle("DZ_GRAD Sensitivity", fontsize=14)
    fig.tight_layout()
    fig.savefig(fig_dir / "dz_grad_sensitivity.png", dpi=150)
    plt.close(fig)


def plot_zplane_anomaly_detail(tree, p_fem, fig_dir, ngrid=400, roi_half=None):
    """
    Figure 6: Detailed comparison of z-planes near the anomaly (+0.05λ).
    Show the U field, candidates, and force field for -0.01, 0.0, +0.04,
    +0.05, +0.06, +0.10λ offsets.
    """
    if roi_half is None:
        roi_half = ROI_HALF
    xg = np.linspace(CX - roi_half, CX + roi_half, ngrid)
    yg = np.linspace(CY - roi_half, CY + roi_half, ngrid)

    offsets = [-0.01, 0.0, 0.04, 0.05, 0.06, 0.10]
    fig, axes = plt.subplots(2, 6, figsize=(30, 10))

    for col, dz_frac in enumerate(offsets):
        z_val = Z_STAR + dz_frac * LAM
        r = run_pipeline_with_attrition(tree, p_fem, xg, yg, z_val,
                                         label=f"z={dz_frac:+.2f}λ")
        ext = np.array([xg[0], xg[-1], yg[0], yg[-1]]) * 1e3

        # Top row: U with candidates
        ax = axes[0, col]
        im = ax.imshow(r["U"], extent=ext, origin="lower", cmap="viridis")
        for c in r["accepted"]:
            ax.plot(c["x_ref_mm"], c["y_ref_mm"], "*", color="yellow",
                    ms=10, mew=0.5)
        for c in r["rejected"]:
            ax.plot(c["x_grid_mm"], c["y_grid_mm"], "x", color="red",
                    ms=5, alpha=0.6)
        ax.set_title(f"δz={dz_frac:+.2f}λ\n{len(r['accepted'])} accepted",
                     fontsize=9)
        if col == 0:
            ax.set_ylabel("U + traps")

        # Bottom row: F_mag
        ax = axes[1, col]
        im = ax.imshow(r["F_mag"], extent=ext, origin="lower", cmap="hot")
        for c in r["accepted"]:
            ax.plot(c["x_ref_mm"], c["y_ref_mm"], "*", color="cyan",
                    ms=10, mew=0.5)
        if col == 0:
            ax.set_ylabel("|F| + traps")

    fig.suptitle("Z-Plane Anomaly Detail (near +0.05λ)", fontsize=14)
    fig.tight_layout()
    fig.savefig(fig_dir / "zplane_anomaly_detail.png", dpi=150)
    plt.close(fig)


def plot_u_comparison_with_without_z(result, fig_dir, prefix="baseline"):
    """Figure 7: Compare U with and without z-gradient term."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ext = np.array([result["xg"][0], result["xg"][-1],
                     result["yg"][0], result["yg"][-1]]) * 1e3

    ax = axes[0]
    im = ax.imshow(result["U"], extent=ext, origin="lower", cmap="viridis")
    plt.colorbar(im, ax=ax, label="U [J]")
    ax.set_title("U (with z-gradient)")

    ax = axes[1]
    im = ax.imshow(result["U_no_z"], extent=ext, origin="lower", cmap="viridis")
    plt.colorbar(im, ax=ax, label="U [J]")
    ax.set_title("U (no z-gradient)")

    diff = result["U"] - result["U_no_z"]
    ax = axes[2]
    vmax = max(abs(diff.min()), abs(diff.max()))
    if vmax < 1e-30:
        vmax = 1e-30
    im = ax.imshow(diff, extent=ext, origin="lower", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, label="ΔU [J]")
    rms = np.sqrt(np.mean(diff**2))
    ax.set_title(f"U - U_no_z  (RMS={rms:.2e})")

    for ax in axes:
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")

    fig.suptitle(f"Z-Gradient Impact on Gor'kov Potential ({prefix})", fontsize=14)
    fig.tight_layout()
    fig.savefig(fig_dir / f"u_with_vs_without_z_{prefix}.png", dpi=150)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# Report generation
# ═══════════════════════════════════════════════════════════════════

def generate_report(out_dir, baseline, zplane_results, res_results,
                    dz_results, force_analysis, config):
    lines = []
    def w(s=""):
        lines.append(s)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    w("# Trap Localisation Debug Study — Standing Wave Only")
    w()
    w(f"**Date**: {ts}")
    w()
    w("## 1. Purpose")
    w()
    w("Diagnose WHY the trap detection pipeline produces unstable counts")
    w("(7/15/17 at 200/400/800 grid) and bizarre z-plane sensitivity")
    w("(0 traps at +0.05λ). Standing-wave-only — no vortex, no bridge analysis.")
    w()

    # ── Section 2: Baseline attrition ──
    w("## 2. Baseline Attrition (Medium, N=400)")
    w()
    w("| Stage | Remaining | Dropped | Notes |")
    w("|-------|-----------|---------|-------|")
    for stage_name, n_remain, n_drop, notes in baseline["attrition"]:
        w(f"| {stage_name} | {n_remain} | {n_drop} | {notes} |")
    w()
    w(f"**Final accepted**: {len(baseline['accepted'])}")
    w(f"**Force sweep (at z*)**: {baseline['force_sweep']}")
    w()

    # ── Section 3: Z-plane debug ──
    w("## 3. Z-Plane Dense Sweep")
    w()
    w("| δz/λ | z [mm] | Raw | Boundary | Hessian | Fit | Force | Depth | Final |")
    w("|------|--------|-----|----------|---------|-----|-------|-------|-------|")
    for r in zplane_results:
        w(f"| {r['dz_frac']:+.2f} | {r['z_mm']:.4f} | "
          f"{r['n_raw']} | {r['n_after_boundary']} | {r['n_after_hessian']} | "
          f"{r['n_after_fit']} | {r['n_after_force']} | {r['n_after_depth']} | "
          f"{r['n_accepted']} |")
    w()

    # Identify anomaly
    for r in zplane_results:
        if r["n_accepted"] == 0 and abs(r["dz_frac"]) < 0.15:
            w(f"**ANOMALY at δz={r['dz_frac']:+.2f}λ**: 0 accepted traps!")
            # Find which stage killed everything
            stages = [
                ("raw", r["n_raw"]),
                ("boundary", r["n_after_boundary"]),
                ("hessian", r["n_after_hessian"]),
                ("fit", r["n_after_fit"]),
                ("force", r["n_after_force"]),
                ("depth", r["n_after_depth"]),
                ("dedup", r["n_accepted"]),
            ]
            for i in range(1, len(stages)):
                if stages[i][1] < stages[i-1][1]:
                    w(f"  - {stages[i][0]} dropped "
                      f"{stages[i-1][1] - stages[i][1]} candidates")
            w()

    # Z-stability analysis
    w("### Z-plane stability analysis")
    w()
    counts = [r["n_accepted"] for r in zplane_results]
    dz_vals = [r["dz_frac"] for r in zplane_results]
    w(f"- Range of accepted counts: {min(counts)} – {max(counts)}")
    w(f"- Std dev of counts: {np.std(counts):.2f}")
    stable_region = [(d, c) for d, c in zip(dz_vals, counts)
                     if c >= max(counts) * 0.8]
    if stable_region:
        w(f"- Stable region (≥80% of max): "
          f"δz/λ ∈ [{min(d for d,c in stable_region):+.2f}, "
          f"{max(d for d,c in stable_region):+.2f}]")
    w()

    # ── Section 4: Resolution convergence ──
    w("## 4. Resolution Convergence")
    w()
    w("| N | dx [µm] | Raw | Accepted | Stages... |")
    w("|---|---------|-----|----------|-----------|")
    for r in res_results:
        stages_str = " → ".join(str(a[1]) for a in r["attrition"])
        w(f"| {r['ngrid']} | {r['dx_um']:.1f} | "
          f"{r['attrition'][0][1]} | {len(r['accepted'])} | {stages_str} |")
    w()

    # Cross-resolution matching
    if len(res_results) >= 2:
        finest = res_results[-1]
        if len(finest["accepted"]) > 0:
            pos_fine = np.array([[c["x_ref_mm"], c["y_ref_mm"]]
                                 for c in finest["accepted"]])
            tree_fine = cKDTree(pos_fine)
            w("### Cross-resolution matching (vs finest)")
            w()
            w("| N | Matched | Mean shift [µm] | Max shift [µm] |")
            w("|---|---------|-----------------|----------------|")
            for r in res_results[:-1]:
                if len(r["accepted"]) == 0:
                    w(f"| {r['ngrid']} | 0 | — | — |")
                    continue
                pos_r = np.array([[c["x_ref_mm"], c["y_ref_mm"]]
                                  for c in r["accepted"]])
                dd, _ = tree_fine.query(pos_r)
                matched = dd < TRAP_SP * 1e3 * 0.5
                if matched.sum() > 0:
                    w(f"| {r['ngrid']} | {matched.sum()} | "
                      f"{dd[matched].mean()*1e3:.1f} | {dd[matched].max()*1e3:.1f} |")
                else:
                    w(f"| {r['ngrid']} | 0 | — | — |")
            w()

    # Convergence stability
    n_acc_list = [len(r["accepted"]) for r in res_results]
    w(f"- Trap counts: {n_acc_list}")
    if len(n_acc_list) >= 3:
        # Check if converged in last 3
        last3 = n_acc_list[-3:]
        if max(last3) - min(last3) <= 1:
            w(f"- **CONVERGED**: last 3 resolutions give {last3}")
        else:
            w(f"- **NOT CONVERGED**: last 3 resolutions give {last3}")
    w()

    # ── Section 5: DZ_GRAD sensitivity ──
    w("## 5. DZ_GRAD Sensitivity")
    w()
    w("| DZ_GRAD/λ | DZ [µm] | Raw | Hessian OK | Force OK | corr(U, U_no_z) |")
    w("|-----------|---------|-----|------------|----------|----------------|")
    for r in dz_results:
        w(f"| {r['dz_frac_lam']:.3f} | {r['dz_um']:.1f} | {r['n_raw']} | "
          f"{r['n_hessian_pass']} | {r['n_force_pass']} | {r['corr_with_no_z']:.6f} |")
    w()

    # ── Section 6: Force criterion ──
    w("## 6. Force Criterion Analysis")
    w()
    fracs = np.array(force_analysis["force_fracs"])
    labels = force_analysis["labels"]
    acc_mask = np.array([l == "accepted" for l in labels])

    if acc_mask.sum() > 0:
        acc_fracs = fracs[acc_mask]
        w(f"- Accepted force fracs: min={acc_fracs.min():.4f}, "
          f"max={acc_fracs.max():.4f}, mean={acc_fracs.mean():.4f}")
    if (~acc_mask).sum() > 0:
        rej_fracs = fracs[~acc_mask]
        w(f"- Rejected force fracs: min={rej_fracs.min():.4f}, "
          f"max={rej_fracs.max():.4f}, mean={rej_fracs.mean():.4f}")

    # Gap analysis
    if acc_mask.sum() > 0 and (~acc_mask).sum() > 0:
        gap = rej_fracs.min() - acc_fracs.max()
        w(f"- **Gap between accepted and rejected**: {gap:.4f}")
        if gap > 0.05:
            w(f"  → Clean separation — threshold 0.10 is in the gap")
        else:
            w(f"  → Overlapping distributions — threshold may be splitting a "
              f"single population")

    ref_fracs = np.array(force_analysis["refined_force_fracs"])
    if acc_mask.sum() > 0:
        w(f"- Refined-position force fracs (accepted): "
          f"mean={ref_fracs[acc_mask].mean():.4f}")
    if (~acc_mask).sum() > 0:
        w(f"- Refined-position force fracs (rejected): "
          f"mean={ref_fracs[~acc_mask].mean():.4f}")
    w()

    # ── Section 7: Diagnosis and recommendations ──
    w("## 7. Diagnosis and Recommendations")
    w()
    w("### 7a. Root cause of resolution instability")
    w()
    if len(res_results) >= 2:
        # Count how many traps appear only at high resolution
        finest_n = len(res_results[-1]["accepted"])
        coarsest_n = len(res_results[0]["accepted"])
        w(f"Coarsest ({res_results[0]['ngrid']}): {coarsest_n} traps")
        w(f"Finest ({res_results[-1]['ngrid']}): {finest_n} traps")
        if finest_n > coarsest_n:
            w(f"→ {finest_n - coarsest_n} traps only resolve at higher resolution")
            w("  This suggests the coarse grid cannot detect some real minima")
            w("  because the minimum_filter footprint at coarse resolution")
            w("  (in physical units) is larger, merging nearby features.")
        w()

    w("### 7b. Root cause of z-plane anomaly")
    w()
    anomaly_z = [r for r in zplane_results if r["n_accepted"] == 0]
    if anomaly_z:
        for az in anomaly_z:
            w(f"At δz={az['dz_frac']:+.2f}λ, n_raw={az['n_raw']}, "
              f"n_after_force={az['n_after_force']}")
            if az["n_raw"] == 0:
                w("  → No candidates detected by morphological filter!")
                w("  → This means U has no local minima at this z-plane,")
                w("    or all are at the boundary.")
            elif az["n_after_force"] == 0 and az["n_raw"] > 0:
                w("  → Candidates exist but ALL fail the force criterion.")
                w("  → Check if this z-plane has a fundamentally different")
                w("    field topology (e.g., node plane).")
    else:
        w("No z-plane anomaly detected in this sweep.")
    w()

    w("### 7c. Recommended pipeline changes")
    w()
    w("1. If force criterion creates a clean bimodal separation, keep at 0.10")
    w("2. If coarse grids miss traps, set minimum resolution ≥ ptarget pts/λ")
    w("3. If z-anomaly is a real topological transition, document it as a")
    w("   physical feature rather than a bug")
    w("4. Consider evaluating force at REFINED position instead of grid point")
    w()

    w("## Figures")
    w()
    w("![Z-plane sweep](figures/zplane_dense_sweep.png)")
    w("![Z-plane anomaly detail](figures/zplane_anomaly_detail.png)")
    w("![Resolution convergence](figures/resolution_convergence.png)")
    w("![Force criterion](figures/force_criterion_analysis.png)")
    w("![Candidate map](figures/candidate_map_baseline.png)")
    w("![DZ_GRAD sensitivity](figures/dz_grad_sensitivity.png)")
    w("![U with/without z-grad](figures/u_with_vs_without_z_baseline.png)")
    w()
    w("---")
    w(f"*Generated by trap_localisation_debug_standing.py on {ts}*")

    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines))
    return report_path


# ═══════════════════════════════════════════════════════════════════
# CSV writers
# ═══════════════════════════════════════════════════════════════════

CANDIDATE_FIELDS = [
    "idx", "ix", "iy",
    "x_grid_mm", "y_grid_mm", "x_ref_mm", "y_ref_mm",
    "shift_mm", "U_grid", "U_ref", "U_no_z",
    "depth", "grad_norm", "force_frac",
    "eig_fd_0", "eig_fd_1", "eig_fd_wide_0", "eig_fd_wide_1",
    "eig_fit_0", "eig_fit_1", "fit_residual",
    "is_true_min_fit", "refine_ok",
    "nn_dist_mm",
    "removed_at", "removal_reason",
]


def write_csv(filepath, rows, fieldnames):
    with open(filepath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            row = {k: r.get(k, "") for k in fieldnames}
            w.writerow(row)


# ═══════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════

def main():
    global _LOG_FP

    t_start = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "results" / f"trap_localisation_debug_standing_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    _LOG_FP = open(out_dir / "run.log", "w")

    log(f"Trap Localisation Debug Study — Standing Wave Only — {timestamp}")
    log(f"Output: {out_dir}")

    # Load FEM cache
    cache_path = find_latest_cache(STANDING_CACHE_DIR)
    coords, p_fem = load_fem_cache(cache_path)
    tree = cKDTree(coords)
    n_dofs = len(p_fem)
    log(f"FEM cache: {cache_path.name}  ({n_dofs} DOFs)")
    log(f"z* = {Z_STAR * 1e3:.4f} mm")
    log(f"λ = {LAM * 1e3:.4f} mm, λ/2 = {TRAP_SP * 1e3:.4f} mm")
    log(f"DZ_GRAD = {DZ_GRAD * 1e6:.1f} µm = λ/{LAM/DZ_GRAD:.0f}")

    config = {
        "timestamp": timestamp,
        "cache_path": str(cache_path),
        "n_dofs": n_dofs,
        "z_star_mm": Z_STAR * 1e3,
        "lambda_mm": LAM * 1e3,
        "trap_sp_mm": TRAP_SP * 1e3,
        "roi_half_mm": ROI_HALF * 1e3,
        "dz_grad_um": DZ_GRAD * 1e6,
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ══════════════════════════════════════════════════════════════
    # Test 1: Baseline attrition (medium, N=400)
    # ══════════════════════════════════════════════════════════════
    log("\n═══ Test 1: Baseline attrition (N=400) ═══")
    xg_400 = np.linspace(CX - ROI_HALF, CX + ROI_HALF, 400)
    yg_400 = np.linspace(CY - ROI_HALF, CY + ROI_HALF, 400)
    baseline = run_pipeline_with_attrition(tree, p_fem, xg_400, yg_400,
                                            Z_STAR, label="baseline_N400")
    log(f"Baseline: {len(baseline['accepted'])} accepted, "
        f"{len(baseline['rejected'])} rejected")
    for stage_name, n_remain, n_drop, notes in baseline["attrition"]:
        log(f"  {stage_name}: {n_remain} remain ({n_drop} dropped)  {notes}")

    # Write CSVs
    write_csv(out_dir / "baseline_all_candidates.csv",
              baseline["candidates"], CANDIDATE_FIELDS)
    write_csv(out_dir / "baseline_accepted.csv",
              baseline["accepted"], CANDIDATE_FIELDS)

    # ══════════════════════════════════════════════════════════════
    # Test 2: Z-plane dense sweep
    # ══════════════════════════════════════════════════════════════
    log("\n═══ Test 2: Z-plane dense sweep ═══")
    zplane_results = run_zplane_dense_sweep(tree, p_fem, 400, ROI_HALF)

    # Write z-plane CSV
    zp_fields = ["dz_frac", "z_mm", "n_raw", "n_after_boundary",
                  "n_after_hessian", "n_after_fit", "n_after_force",
                  "n_after_depth", "n_accepted", "F_max"]
    with open(out_dir / "zplane_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=zp_fields, extrasaction="ignore")
        w.writeheader()
        for r in zplane_results:
            w.writerow({k: r.get(k, "") for k in zp_fields})

    # ══════════════════════════════════════════════════════════════
    # Test 3: Resolution convergence
    # ══════════════════════════════════════════════════════════════
    log("\n═══ Test 3: Resolution convergence ═══")
    res_results = run_resolution_convergence(
        tree, p_fem, Z_STAR, ROI_HALF,
        ngrids=(100, 150, 200, 300, 400, 600, 800))

    # Write convergence CSV
    with open(out_dir / "resolution_convergence.csv", "w", newline="") as f:
        fields = ["ngrid", "dx_um", "n_raw", "n_accepted"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in res_results:
            w.writerow({"ngrid": r["ngrid"], "dx_um": r["dx_um"],
                         "n_raw": r["attrition"][0][1],
                         "n_accepted": len(r["accepted"])})

    # ══════════════════════════════════════════════════════════════
    # Test 4: DZ_GRAD sensitivity
    # ══════════════════════════════════════════════════════════════
    log("\n═══ Test 4: DZ_GRAD sensitivity ═══")
    dz_results = run_dz_grad_sensitivity(tree, p_fem, 400, ROI_HALF, Z_STAR)

    # ══════════════════════════════════════════════════════════════
    # Test 5: Force criterion analysis
    # ══════════════════════════════════════════════════════════════
    log("\n═══ Test 5: Force criterion analysis ═══")
    force_analysis = run_force_criterion_analysis(baseline)
    fracs = np.array(force_analysis["force_fracs"])
    labels_arr = force_analysis["labels"]
    acc_mask = np.array([l == "accepted" for l in labels_arr])
    log(f"  Force fracs — accepted: [{fracs[acc_mask].min():.4f}, "
        f"{fracs[acc_mask].max():.4f}]" if acc_mask.sum() > 0 else "  No accepted")
    if (~acc_mask).sum() > 0:
        log(f"  Force fracs — rejected: [{fracs[~acc_mask].min():.4f}, "
            f"{fracs[~acc_mask].max():.4f}]")
    ref_fracs = np.array(force_analysis["refined_force_fracs"])
    if acc_mask.sum() > 0:
        log(f"  Refined force fracs — accepted: [{ref_fracs[acc_mask].min():.4f}, "
            f"{ref_fracs[acc_mask].max():.4f}]")

    # ══════════════════════════════════════════════════════════════
    # Generate figures
    # ══════════════════════════════════════════════════════════════
    log("\n═══ Generating figures ═══")
    plot_zplane_attrition(zplane_results, fig_dir)
    log("  zplane_dense_sweep.png")

    plot_resolution_convergence(res_results, fig_dir)
    log("  resolution_convergence.png")

    plot_force_criterion(force_analysis, fig_dir)
    log("  force_criterion_analysis.png")

    plot_candidate_map(baseline, fig_dir, prefix="baseline")
    log("  candidate_map_baseline.png")

    plot_dz_grad_sensitivity(dz_results, fig_dir)
    log("  dz_grad_sensitivity.png")

    plot_u_comparison_with_without_z(baseline, fig_dir, prefix="baseline")
    log("  u_with_vs_without_z_baseline.png")

    plot_zplane_anomaly_detail(tree, p_fem, fig_dir)
    log("  zplane_anomaly_detail.png")

    # ══════════════════════════════════════════════════════════════
    # Generate report
    # ══════════════════════════════════════════════════════════════
    log("\n═══ Generating report ═══")
    report_path = generate_report(out_dir, baseline, zplane_results,
                                   res_results, dz_results, force_analysis, config)
    log(f"Report: {report_path}")

    total_time = time.time() - t_start
    log(f"\nTotal runtime: {total_time:.1f}s")
    log(f"Output directory: {out_dir}")

    _LOG_FP.close()
    print(f"\n{'='*60}")
    print(f"DONE — see {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
