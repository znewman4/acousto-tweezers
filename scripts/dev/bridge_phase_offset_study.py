#!/usr/bin/env python3
"""
Bridge Formation Study — Phase & Lateral-Offset Sweep
======================================================

Investigates whether interference between a focused vortex beam and the
FEM standing-wave field can reduce the Gor'kov barrier between two adjacent
traps, forming a "bridge".

New degrees of freedom (vs previous α-only sweep):
  • φ0  — relative phase:  p_total = p_stand + α·exp(iφ0)·p_vort
  • (x0, y0) — lateral shift of vortex centre relative to trap lattice

Sweep grid:
  α   ∈ {0.05, 0.10, 0.15, 0.20}
  φ0  ∈ 8 values in [0, 2π)
  x0  ∈ {−0.2, −0.1, 0, +0.1, +0.2} × trap_spacing
  y0  ∈ {−0.2, −0.1, 0, +0.1, +0.2} × trap_spacing

Bridge metric:
  Pick two adjacent (non-diagonal) trap minima.
  Sample U along connecting line.
  B = U_saddle − max(U_min1, U_min2)
  B ≤ 0  ⟹  bridge formed.

Outputs → results/bridge_phase_offset_study_YYYYMMDD_HHMMSS/

Usage:
    python scripts/dev/bridge_phase_offset_study.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree
from scipy.ndimage import maximum_filter, label

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# ── project root ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from acoustweezers.physics.acoustics.angular_spectrum import propagate_pressure_asm

# ═══════════════════════════════════════════════════════════════════
# Physical constants
# ═══════════════════════════════════════════════════════════════════
C_WATER = 1484.0        # m/s
F_HZ    = 2.0e6
K_WATER = 2.0 * np.pi * F_HZ / C_WATER
LAM     = C_WATER / F_HZ   # ≈ 0.742 mm
TRAP_SP = LAM / 2.0         # ≈ 0.371 mm
OMEGA   = 2.0 * np.pi * F_HZ
RHO0    = 997.0             # kg/m³ water

# FEM domain geometry
LX = LY   = 6.0e-3
H_UNDER   = 3.0e-3
H_TOP     = 2.0085e-3
CX = CY   = LX / 2.0
Z_STAR    = H_UNDER + H_TOP / 2.0 + 0.25 * LAM  # ≈ 4.190 mm

# Particle (polystyrene in water)
RHO_P   = 1050.0
C_P     = 2350.0
A_P     = 5.0e-6       # radius
V_P     = (4.0 / 3.0) * np.pi * A_P**3
KAPPA0  = 1.0 / (RHO0 * C_WATER**2)
KAPPA_P = 1.0 / (RHO_P * C_P**2)
F1      = 1.0 - KAPPA_P / KAPPA0
F2      = 2.0 * (RHO_P - RHO0) / (2.0 * RHO_P + RHO0)

# ASM
ASM_N   = 512
ASM_PAD = 2

# ── Fixed lens config ────────────────────────────────────────────
ELL_FIXED   = 1
FOCAL_FIXED = 4.2e-3
R_FIXED     = 7.0e-3
APOD_FIXED  = "uniform"
Z0_OFFSET   = 0.0  # from previous calibration

# ── Small ROI: ~2λ ───────────────────────────────────────────────
ROI_HALF = 1.1 * LAM
NGRID    = 400

# Gradient step
DZ_GRAD = LAM / 15.0

# ── Sweep parameters ─────────────────────────────────────────────
ALPHAS      = [0.05, 0.10, 0.15, 0.20]
N_PHASE     = 8
PHI0_VALUES = np.linspace(0, 2 * np.pi, N_PHASE, endpoint=False)

# Lateral offsets in units of trap spacing
SHIFT_FRACS = np.array([-0.2, -0.1, 0.0, 0.1, 0.2])
X0_OFFSETS  = SHIFT_FRACS * TRAP_SP
Y0_OFFSETS  = SHIFT_FRACS * TRAP_SP

# Number of sample points along the bridge line
BRIDGE_NPTS = 200

STANDING_CACHE_DIR = PROJECT_ROOT / "results" / "fem_standing_wave_cache"

K_IDW = 16

# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def build_converging_drive(ell, focal, aperture, cx, cy,
                           asm_domain, asm_n, k, apod="uniform"):
    """φ = ℓ·θ − k·(√(r²+f²) − f)  [CONVERGING]"""
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
    meta = {}
    for mk in keys:
        if mk not in {"coords", "p_real", "p_imag", "p"}:
            v = d[mk]
            meta[mk] = v.item() if isinstance(v, np.ndarray) and v.ndim == 0 else v
    return coords, p, meta


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
    """Find Gor'kov minima (trap centres).

    U is a 2D Gor'kov potential.  Traps sit at local *minima* of U.
    We detect them as local maxima of −U.
    """
    neg_U = -U
    local_max = (neg_U == maximum_filter(neg_U, size=min_sep_pix))
    local_max &= (neg_U > threshold_frac * neg_U.max())
    iy, ix = np.where(local_max)
    return iy, ix


def find_adjacent_pair(U, xg, yg):
    """Return (iy1,ix1), (iy2,ix2) for the best pair of adjacent traps.

    Best = pair along x-axis near ROI centre with spacing closest to λ/2.
    """
    iy, ix = find_trap_minima(U)
    if len(iy) < 2:
        return None

    # Build pairs sorted by distance
    n = len(iy)
    pairs = []
    for a in range(n):
        for b in range(a + 1, n):
            dy_m = yg[iy[a]] - yg[iy[b]]
            dx_m = xg[ix[a]] - xg[ix[b]]
            dist = np.sqrt(dx_m**2 + dy_m**2)
            # Only consider near-horizontal or near-vertical (non-diagonal)
            angle = abs(np.arctan2(abs(dy_m), abs(dx_m)))
            if angle > np.pi / 6 and angle < np.pi / 3:
                continue  # skip diagonal
            pairs.append((dist, a, b))

    if not pairs:
        return None

    # Prefer pair closest to λ/2 spacing and closest to centre
    pairs.sort(key=lambda t: abs(t[0] - TRAP_SP))
    _, a, b = pairs[0]
    return (iy[a], ix[a]), (iy[b], ix[b])


def compute_bridge_metric(U, xg, yg, trap1, trap2, n_pts=BRIDGE_NPTS):
    """Compute bridge barrier B along the line connecting two trap minima.

    Returns
    -------
    B : float
        Saddle height above deeper minimum.
        B > 0 → separate traps, B ≤ 0 → bridge.
    U_line : ndarray
        Gor'kov potential along the connecting line.
    frac : ndarray
        Fractional distance along line [0, 1].
    U_min1, U_min2, U_saddle : float
    """
    iy1, ix1 = trap1
    iy2, ix2 = trap2
    x1, y1 = xg[ix1], yg[iy1]
    x2, y2 = xg[ix2], yg[iy2]

    frac = np.linspace(0, 1, n_pts)
    xline = x1 + frac * (x2 - x1)
    yline = y1 + frac * (y2 - y1)

    # Interpolate U along line
    rgi = RegularGridInterpolator((yg, xg), U, method="linear",
                                  bounds_error=False, fill_value=np.nan)
    pts = np.column_stack([yline, xline])
    U_line = rgi(pts)

    U_min1 = float(U_line[0])
    U_min2 = float(U_line[-1])
    U_saddle = float(np.nanmax(U_line))

    B = U_saddle - max(U_min1, U_min2)
    return B, U_line, frac, U_min1, U_min2, U_saddle


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=str, default=None)
    ap.add_argument("--timestamp", type=str,
                    default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    args = ap.parse_args()

    t_GLOBAL = time.time()
    TS = args.timestamp

    cache_path = (Path(args.cache).resolve() if args.cache
                  else find_latest_cache(STANDING_CACHE_DIR))

    OUT = PROJECT_ROOT / "results" / f"bridge_phase_offset_study_{TS}"
    FIG = OUT / "figures"; FIG.mkdir(parents=True, exist_ok=True)
    DAT = OUT / "data";    DAT.mkdir(parents=True, exist_ok=True)

    log_fp = open(OUT / "console_log.txt", "w")
    log = Log(log_fp)

    NA_FIXED = R_FIXED / np.sqrt(R_FIXED**2 + FOCAL_FIXED**2)

    log("=" * 72)
    log("Bridge Formation Study — Phase & Lateral-Offset Sweep")
    log("=" * 72)
    log(f"Date  : {datetime.now().isoformat()}")
    log(f"Cache : {cache_path}")
    log(f"Output: {OUT}")
    log(f"λ = {LAM*1e3:.4f} mm   k = {K_WATER:.1f} rad/m   "
        f"trap λ/2 = {TRAP_SP*1e3:.4f} mm")
    log(f"z* = {Z_STAR*1e3:.4f} mm   z0_offset = {Z0_OFFSET*1e3:.3f} mm")
    log(f"Lens: ℓ={ELL_FIXED}  f={FOCAL_FIXED*1e3:.1f}mm  "
        f"R={R_FIXED*1e3:.1f}mm  {APOD_FIXED}  NA={NA_FIXED:.4f}")
    log(f"Sweep: α={ALPHAS}  φ0={N_PHASE} values  "
        f"x0={len(X0_OFFSETS)} × y0={len(Y0_OFFSETS)} offsets")
    n_configs = len(ALPHAS) * N_PHASE * len(X0_OFFSETS) * len(Y0_OFFSETS)
    log(f"Total configurations: {n_configs}")
    log()

    # ==============================================================
    # STEP 1 — Load FEM cache + build ASM drive + compute baseline
    # ==============================================================
    log("=" * 72)
    log("STEP 1: Load data & compute baseline standing-wave Gor'kov")
    log("=" * 72)

    # Load FEM
    t0 = time.time()
    coords, p_fem, meta = load_fem_cache(cache_path)
    tree = cKDTree(coords)
    log(f"  FEM: {coords.shape[0]:,} DOFs ({time.time()-t0:.2f}s)")

    # Small ROI grid
    xg = np.linspace(CX - ROI_HALF, CX + ROI_HALF, NGRID)
    yg = np.linspace(CY - ROI_HALF, CY + ROI_HALF, NGRID)
    dx = float(xg[1] - xg[0])
    dy = float(yg[1] - yg[0])
    XX, YY = np.meshgrid(xg, yg)

    # ROI extent for plotting (mm) — TIGHT to actual data bounds
    ext_roi = [xg[0] * 1e3, xg[-1] * 1e3,
               yg[0] * 1e3, yg[-1] * 1e3]

    log(f"  ROI: {2*ROI_HALF*1e3:.3f}mm × {2*ROI_HALF*1e3:.3f}mm  "
        f"({2*ROI_HALF/LAM:.1f}λ)  grid {NGRID}×{NGRID}")
    log(f"  dx = {dx*1e6:.1f} µm  ({LAM/dx:.0f} pts/λ)")
    log()

    # FEM IDW sampling
    def _fem_xy(z_val):
        pts = np.column_stack([XX.ravel(), YY.ravel(),
                               np.full(XX.size, z_val)])
        return sample_idw(tree, p_fem, pts, k=K_IDW).reshape(XX.shape)

    # Sample standing wave (3 z-planes)
    log("  Sampling FEM standing wave (z*, z*±dz)...")
    t0 = time.time()
    p_s    = _fem_xy(Z_STAR)
    p_s_zp = _fem_xy(Z_STAR + DZ_GRAD)
    p_s_zm = _fem_xy(Z_STAR - DZ_GRAD)
    log(f"    Done: {time.time()-t0:.2f}s  max|p_s| = {np.abs(p_s).max():.3f} Pa")

    # Standing Gor'kov baseline
    gp2_s = grad_p_mag2_xy(p_s, p_s_zp, p_s_zm, dx, dy, DZ_GRAD)
    U_s   = gorkov_potential(np.abs(p_s)**2, gp2_s)
    trap_depth = float(U_s.max() - U_s.min())
    log(f"  Standing Gorkov: depth = {trap_depth:.3e} J")

    # Find baseline adjacent pair
    pair = find_adjacent_pair(U_s, xg, yg)
    if pair is None:
        log("  *** FATAL: Cannot find adjacent trap pair in baseline ***")
        log_fp.close()
        return
    trap1, trap2 = pair
    log(f"  Baseline trap pair: ({trap1[1]},{trap1[0]}) – ({trap2[1]},{trap2[0]})")
    log(f"    ({xg[trap1[1]]*1e3:.3f}, {yg[trap1[0]]*1e3:.3f}) mm  –  "
        f"({xg[trap2[1]]*1e3:.3f}, {yg[trap2[0]]*1e3:.3f}) mm")
    sep = np.sqrt((xg[trap1[1]] - xg[trap2[1]])**2 +
                  (yg[trap1[0]] - yg[trap2[0]])**2)
    log(f"    Separation: {sep*1e3:.3f} mm  ({sep/TRAP_SP:.2f} × λ/2)")

    B_baseline, U_line_base, frac_base, _, _, _ = compute_bridge_metric(
        U_s, xg, yg, trap1, trap2)
    log(f"  Baseline barrier B = {B_baseline:.4e} J  "
        f"({B_baseline/trap_depth*100:.2f}% of trap depth)")
    log()

    # Build ASM drive
    asm_domain = max(LX, 2 * R_FIXED + 4 * LAM)
    xg_d, yg_d, D_drive = build_converging_drive(
        ELL_FIXED, FOCAL_FIXED, R_FIXED, CX, CY,
        asm_domain, ASM_N, K_WATER, apod=APOD_FIXED)
    dx_d = float(xg_d[1] - xg_d[0])

    # ==============================================================
    # STEP 2 — Pre-compute vortex fields for each lateral offset
    # ==============================================================
    log("=" * 72)
    log("STEP 2: Pre-compute vortex fields for each lateral offset")
    log("=" * 72)

    # For each (x0, y0) offset we propagate the ASM vortex at z*
    # and z*±dz, then shift the interpolation grid.
    # Lateral shift: instead of shifting the drive, we shift the
    # interpolation points → p_vort(x−x0, y−y0) = sample at (x−x0, y−y0).

    vort_fields = {}  # key: (ix0, iy0) → (p_v, p_v_zp, p_v_zm)

    # Propagate once at each z (no offset in propagation — offset in sampling)
    t0 = time.time()
    z_prop_star = Z_STAR - Z0_OFFSET
    p_asm_full       = propagate_pressure_asm(D_drive, dx_d, dx_d, K_WATER,
                                              z_prop_star, pad_factor=ASM_PAD,
                                              include_evanescent=True)
    p_asm_full_zp    = propagate_pressure_asm(D_drive, dx_d, dx_d, K_WATER,
                                              z_prop_star + DZ_GRAD,
                                              pad_factor=ASM_PAD,
                                              include_evanescent=True)
    p_asm_full_zm    = propagate_pressure_asm(D_drive, dx_d, dx_d, K_WATER,
                                              z_prop_star - DZ_GRAD,
                                              pad_factor=ASM_PAD,
                                              include_evanescent=True)
    log(f"  ASM propagation (3 z-planes): {time.time()-t0:.2f}s")

    # Normalise vortex to standing peak
    p_v_centre = interp_asm_to_grid(p_asm_full, xg_d, yg_d, XX, YY)
    stand_peak = float(np.abs(p_s).max())
    vort_peak  = float(np.abs(p_v_centre).max()) + 1e-30
    nfac = stand_peak / vort_peak
    log(f"  nfac = {nfac:.3f}  (stand_peak={stand_peak:.3f} Pa, "
        f"vort_peak={vort_peak:.6f})")

    # Sample shifted vortex fields for each (x0, y0) offset
    t0 = time.time()
    n_offsets = len(X0_OFFSETS) * len(Y0_OFFSETS)
    log(f"  Sampling {n_offsets} lateral offsets...")

    for iix, x0 in enumerate(X0_OFFSETS):
        for iiy, y0 in enumerate(Y0_OFFSETS):
            # Shift: to get p_vort(x−x0, y−y0) we sample the ASM field
            # at (x−x0, y−y0) in the original coordinate system
            XX_shift = XX - x0
            YY_shift = YY - y0
            pv    = interp_asm_to_grid(p_asm_full, xg_d, yg_d,
                                       XX_shift, YY_shift) * nfac
            pv_zp = interp_asm_to_grid(p_asm_full_zp, xg_d, yg_d,
                                       XX_shift, YY_shift) * nfac
            pv_zm = interp_asm_to_grid(p_asm_full_zm, xg_d, yg_d,
                                       XX_shift, YY_shift) * nfac
            vort_fields[(iix, iiy)] = (pv, pv_zp, pv_zm)

    log(f"  Done: {time.time()-t0:.2f}s for {n_offsets} offsets × 3 planes")
    log()

    # ==============================================================
    # STEP 3 — Sweep: α × φ0 × (x0, y0)
    # ==============================================================
    log("=" * 72)
    log("STEP 3: Parameter sweep")
    log("=" * 72)

    results = []  # list of dicts
    total = len(ALPHAS) * N_PHASE * len(X0_OFFSETS) * len(Y0_OFFSETS)
    count = 0
    t0_sweep = time.time()

    best_B = np.inf
    best_cfg = None
    best_U = None
    best_U_line = None
    best_frac = None
    best_p_total = None

    for ia, alpha in enumerate(ALPHAS):
        for ip, phi0 in enumerate(PHI0_VALUES):
            phase_factor = np.exp(1j * phi0)
            for iix, x0 in enumerate(X0_OFFSETS):
                for iiy, y0 in enumerate(Y0_OFFSETS):
                    pv, pv_zp, pv_zm = vort_fields[(iix, iiy)]

                    p_t    = p_s    + alpha * phase_factor * pv
                    p_t_zp = p_s_zp + alpha * phase_factor * pv_zp
                    p_t_zm = p_s_zm + alpha * phase_factor * pv_zm

                    gp2 = grad_p_mag2_xy(p_t, p_t_zp, p_t_zm, dx, dy, DZ_GRAD)
                    U_c = gorkov_potential(np.abs(p_t)**2, gp2)

                    # Bridge metric on same trap pair
                    B, U_line, frac, Um1, Um2, Usad = compute_bridge_metric(
                        U_c, xg, yg, trap1, trap2)

                    rec = {
                        "alpha": alpha,
                        "phi0": float(phi0),
                        "x0_frac": float(SHIFT_FRACS[iix]),
                        "y0_frac": float(SHIFT_FRACS[iiy]),
                        "x0_mm": float(x0 * 1e3),
                        "y0_mm": float(y0 * 1e3),
                        "B": float(B),
                        "B_rel": float(B / (trap_depth + 1e-30)),
                        "U_min1": float(Um1),
                        "U_min2": float(Um2),
                        "U_saddle": float(Usad),
                    }
                    results.append(rec)

                    if B < best_B:
                        best_B = B
                        best_cfg = rec.copy()
                        best_U = U_c.copy()
                        best_U_line = U_line.copy()
                        best_frac = frac.copy()
                        best_p_total = p_t.copy()

                    count += 1
                    if count % 200 == 0 or count == total:
                        elapsed = time.time() - t0_sweep
                        log(f"  [{count:5d}/{total}]  "
                            f"best B = {best_B:.4e} J  "
                            f"({best_B/trap_depth*100:+.2f}% depth)  "
                            f"[{elapsed:.1f}s]")

    dt_sweep = time.time() - t0_sweep
    log(f"\n  Sweep complete: {dt_sweep:.1f}s for {total} configs")
    log(f"  ★ BEST B = {best_B:.4e} J  ({best_B/trap_depth*100:+.2f}% of depth)")
    log(f"    α={best_cfg['alpha']:.2f}  φ0={best_cfg['phi0']:.3f} rad  "
        f"x0={best_cfg['x0_frac']:+.1f}·sp  y0={best_cfg['y0_frac']:+.1f}·sp")
    log(f"  Baseline B = {B_baseline:.4e} J  ({B_baseline/trap_depth*100:.2f}%)")
    reduction = (1.0 - best_B / B_baseline) * 100 if B_baseline > 0 else 0
    log(f"  Barrier reduction: {reduction:.1f}%")
    bridge_formed = best_B <= 0
    log(f"  Bridge formed (B ≤ 0): {bridge_formed}")
    log()

    # ==============================================================
    # FIGURES
    # ==============================================================
    log("=" * 72)
    log("FIGURES")
    log("=" * 72)

    # ── Helper: tight axis limits ─────────────────────────────────
    def _setup_ax(ax):
        ax.set_xlim(ext_roi[0], ext_roi[1])
        ax.set_ylim(ext_roi[2], ext_roi[3])

    def _trap_grid(ax):
        for off in np.arange(-4, 5) * TRAP_SP:
            ax.axhline((CY + off) * 1e3, color="white", lw=0.3, alpha=0.25)
            ax.axvline((CX + off) * 1e3, color="white", lw=0.3, alpha=0.25)

    # ── Figure 1: gorkov_best_bridge.png ──────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: baseline U
    ax = axes[0]
    vmin_U = min(U_s.min(), best_U.min())
    vmax_U = max(U_s.max(), best_U.max())
    im = ax.imshow(U_s, extent=ext_roi, origin="lower", aspect="equal",
                   cmap="RdBu_r", vmin=vmin_U, vmax=vmax_U)
    _setup_ax(ax)
    _trap_grid(ax)
    # Mark trap pair
    ax.plot([xg[trap1[1]]*1e3, xg[trap2[1]]*1e3],
            [yg[trap1[0]]*1e3, yg[trap2[0]]*1e3],
            "k-o", ms=6, lw=2, zorder=5)
    ax.set_title(f"Baseline U_Gorkov\nB = {B_baseline:.2e} J", fontsize=10)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    plt.colorbar(im, ax=ax, shrink=0.8, label="J")

    # Panel B: best-bridge U
    ax = axes[1]
    im = ax.imshow(best_U, extent=ext_roi, origin="lower", aspect="equal",
                   cmap="RdBu_r", vmin=vmin_U, vmax=vmax_U)
    _setup_ax(ax)
    _trap_grid(ax)
    ax.plot([xg[trap1[1]]*1e3, xg[trap2[1]]*1e3],
            [yg[trap1[0]]*1e3, yg[trap2[0]]*1e3],
            "k-o", ms=6, lw=2, zorder=5)
    ax.set_title(f"Best Bridge U\nα={best_cfg['alpha']:.2f}  "
                 f"φ₀={best_cfg['phi0']:.2f} rad\n"
                 f"x₀={best_cfg['x0_frac']:+.1f}·sp  "
                 f"y₀={best_cfg['y0_frac']:+.1f}·sp\n"
                 f"B = {best_B:.2e} J", fontsize=9)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    plt.colorbar(im, ax=ax, shrink=0.8, label="J")

    # Panel C: U along bridge line
    ax = axes[2]
    dist_mm = frac_base * sep * 1e3
    ax.plot(dist_mm, U_line_base, "b-", lw=2, label="Baseline")
    ax.plot(dist_mm, best_U_line, "r-", lw=2,
            label=f"Best (α={best_cfg['alpha']:.2f})")
    ax.axhline(max(best_cfg["U_min1"], best_cfg["U_min2"]),
               color="gray", ls=":", lw=1, alpha=0.5)
    ax.set_xlabel("Distance along bridge (mm)")
    ax.set_ylabel("U_Gorkov (J)")
    ax.set_title(f"Barrier profile\nB_base={B_baseline:.2e}  "
                 f"B_best={best_B:.2e}", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Best Bridge Configuration  │  z* = {Z_STAR*1e3:.3f} mm  │  "
        f"trap depth = {trap_depth:.2e} J",
        fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIG / "gorkov_best_bridge.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    log("  Saved gorkov_best_bridge.png")

    # ── Figure 2: bridge_metric_heatmap.png ───────────────────────
    # For each α, make a 2D heatmap: phase vs spatial offset.
    # Collapse spatial to best (x0,y0) for each (α, φ0).
    # Also: heatmap of B vs (x0, y0) for best φ0 at each α.

    n_alpha = len(ALPHAS)
    fig, axes = plt.subplots(2, n_alpha, figsize=(4.5 * n_alpha, 8))
    if n_alpha == 1:
        axes = axes[:, np.newaxis]

    for ia, alpha in enumerate(ALPHAS):
        # Filter results for this α
        recs_a = [r for r in results if r["alpha"] == alpha]

        # ── Top row: B vs φ0 (best over spatial offsets) ──────────
        ax = axes[0, ia]
        B_vs_phi = np.full(N_PHASE, np.inf)
        for ip in range(N_PHASE):
            recs_phi = [r for r in recs_a
                        if abs(r["phi0"] - PHI0_VALUES[ip]) < 0.01]
            if recs_phi:
                B_vs_phi[ip] = min(r["B"] for r in recs_phi)

        ax.plot(np.degrees(PHI0_VALUES), B_vs_phi / trap_depth * 100,
                "o-", ms=5, lw=1.5, color="darkred")
        ax.axhline(0, color="green", ls="--", lw=1, alpha=0.7,
                   label="B=0 (bridge)")
        ax.axhline(B_baseline / trap_depth * 100, color="blue",
                   ls=":", lw=1, label="baseline")
        ax.set_xlabel("φ₀ (deg)")
        ax.set_ylabel("B / depth (%)")
        ax.set_title(f"α = {alpha:.2f}", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)
        if ia == 0:
            ax.legend(fontsize=7)

        # ── Bottom row: B vs (x0, y0) (best over φ0) ─────────────
        ax = axes[1, ia]
        B_grid = np.full((len(Y0_OFFSETS), len(X0_OFFSETS)), np.inf)
        for iix in range(len(X0_OFFSETS)):
            for iiy in range(len(Y0_OFFSETS)):
                recs_xy = [r for r in recs_a
                           if abs(r["x0_frac"] - SHIFT_FRACS[iix]) < 0.001
                           and abs(r["y0_frac"] - SHIFT_FRACS[iiy]) < 0.001]
                if recs_xy:
                    B_grid[iiy, iix] = min(r["B"] for r in recs_xy)

        B_grid_rel = B_grid / trap_depth * 100
        bmax = max(abs(np.nanmin(B_grid_rel)), abs(np.nanmax(B_grid_rel)), 1)
        im = ax.imshow(B_grid_rel,
                       extent=[SHIFT_FRACS[0], SHIFT_FRACS[-1],
                               SHIFT_FRACS[0], SHIFT_FRACS[-1]],
                       origin="lower", aspect="equal",
                       cmap="RdYlGn_r", vmin=-bmax, vmax=bmax)
        ax.set_xlabel("x₀ / trap_sp")
        ax.set_ylabel("y₀ / trap_sp")
        ax.set_title(f"B vs offset  α={alpha:.2f}", fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.8, label="B/depth %")

    fig.suptitle(
        f"Bridge Metric Heatmaps  │  "
        f"baseline B = {B_baseline/trap_depth*100:.1f}% depth",
        fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIG / "bridge_metric_heatmap.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    log("  Saved bridge_metric_heatmap.png")

    # ── Figure 3: example_xy_fields.png ───────────────────────────
    # Show baseline + best config: |p| and U side by side
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))

    # Get best vortex field for display
    bc = best_cfg
    best_iix = list(SHIFT_FRACS).index(bc["x0_frac"])
    best_iiy = list(SHIFT_FRACS).index(bc["y0_frac"])
    pv_best, _, _ = vort_fields[(best_iix, best_iiy)]
    best_phase = np.exp(1j * bc["phi0"])

    # Top row: baseline
    # (0,0) |p_stand|
    ax = axes[0, 0]
    abs_s = np.abs(p_s)
    vmax_p = abs_s.max()
    im = ax.imshow(abs_s, extent=ext_roi, origin="lower", aspect="equal",
                   cmap="inferno", vmin=0, vmax=vmax_p)
    _setup_ax(ax); _trap_grid(ax)
    ax.set_title("|p_standing|", fontsize=10, fontweight="bold")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Pa")

    # (0,1) U standing
    ax = axes[0, 1]
    im = ax.imshow(U_s, extent=ext_roi, origin="lower", aspect="equal",
                   cmap="RdBu_r", vmin=vmin_U, vmax=vmax_U)
    _setup_ax(ax); _trap_grid(ax)
    ax.set_title("U standing", fontsize=10, fontweight="bold")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    plt.colorbar(im, ax=ax, shrink=0.8, label="J")

    # (0,2) |α·exp(iφ)·p_vort| best
    ax = axes[0, 2]
    abs_v = np.abs(bc["alpha"] * best_phase * pv_best)
    im = ax.imshow(abs_v, extent=ext_roi, origin="lower", aspect="equal",
                   cmap="viridis", vmin=0, vmax=abs_v.max())
    _setup_ax(ax); _trap_grid(ax)
    ax.set_title(f"|α·exp(iφ₀)·p_vort|\nα={bc['alpha']:.2f}  "
                 f"φ₀={bc['phi0']:.2f}rad", fontsize=9, fontweight="bold")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Pa")

    # (0,3) |p_total| best
    ax = axes[0, 3]
    abs_t = np.abs(best_p_total)
    im = ax.imshow(abs_t, extent=ext_roi, origin="lower", aspect="equal",
                   cmap="inferno", vmin=0, vmax=vmax_p)
    _setup_ax(ax); _trap_grid(ax)
    ax.set_title("|p_total| best", fontsize=10, fontweight="bold")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Pa")

    # Bottom row
    # (1,0) ΔU = U_combined − U_standing
    ax = axes[1, 0]
    DU_best = best_U - U_s
    du_lim = max(np.abs(DU_best).max(), 1e-30)
    im = ax.imshow(DU_best, extent=ext_roi, origin="lower", aspect="equal",
                   cmap="RdBu_r", vmin=-du_lim, vmax=du_lim)
    _setup_ax(ax); _trap_grid(ax)
    ax.set_title("ΔU = U_comb − U_stand", fontsize=10, fontweight="bold")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    plt.colorbar(im, ax=ax, shrink=0.8, label="J")

    # (1,1) U combined best
    ax = axes[1, 1]
    im = ax.imshow(best_U, extent=ext_roi, origin="lower", aspect="equal",
                   cmap="RdBu_r", vmin=vmin_U, vmax=vmax_U)
    _setup_ax(ax); _trap_grid(ax)
    ax.plot([xg[trap1[1]]*1e3, xg[trap2[1]]*1e3],
            [yg[trap1[0]]*1e3, yg[trap2[0]]*1e3],
            "k-o", ms=6, lw=2, zorder=5)
    ax.set_title("U combined (best)", fontsize=10, fontweight="bold")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    plt.colorbar(im, ax=ax, shrink=0.8, label="J")

    # (1,2) Bridge profile
    ax = axes[1, 2]
    dist_mm = frac_base * sep * 1e3
    ax.plot(dist_mm, U_line_base, "b-", lw=2, label="Baseline")
    ax.plot(dist_mm, best_U_line, "r-", lw=2, label="Best combined")
    ax.axhline(0, color="gray", ls=":", lw=0.5)
    ax.set_xlabel("Distance (mm)")
    ax.set_ylabel("U (J)")
    ax.set_title("Bridge profile", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,3) Phase sweep summary: B vs φ0 for best α
    ax = axes[1, 3]
    best_alpha = best_cfg["alpha"]
    recs_best_a = [r for r in results if r["alpha"] == best_alpha]
    B_vs_phi_best = np.full(N_PHASE, np.inf)
    for ip in range(N_PHASE):
        recs_phi = [r for r in recs_best_a
                    if abs(r["phi0"] - PHI0_VALUES[ip]) < 0.01]
        if recs_phi:
            B_vs_phi_best[ip] = min(r["B"] for r in recs_phi)
    ax.plot(np.degrees(PHI0_VALUES), B_vs_phi_best / trap_depth * 100,
            "o-", ms=5, lw=1.5, color="darkred")
    ax.axhline(0, color="green", ls="--", lw=1, label="B=0")
    ax.axhline(B_baseline / trap_depth * 100, color="blue", ls=":", lw=1,
               label="baseline")
    ax.set_xlabel("φ₀ (deg)")
    ax.set_ylabel("B / depth (%)")
    ax.set_title(f"B vs φ₀  (α={best_alpha:.2f}, best offset)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Example Fields  │  Best Config  │  "
        f"B = {best_B:.2e} J ({best_B/trap_depth*100:+.1f}% depth)",
        fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG / "example_xy_fields.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    log("  Saved example_xy_fields.png")
    log()

    # ==============================================================
    # STEP 4 — Stats & REPORT
    # ==============================================================
    log("=" * 72)
    log("STEP 4: Stats & REPORT")
    log("=" * 72)

    # Sort results by B
    results.sort(key=lambda r: r["B"])
    top10 = results[:10]

    stats = {
        "lambda_mm": float(LAM * 1e3),
        "trap_spacing_mm": float(TRAP_SP * 1e3),
        "z_star_mm": round(float(Z_STAR * 1e3), 4),
        "roi_half_mm": round(float(ROI_HALF * 1e3), 4),
        "roi_width_lambda": round(float(2 * ROI_HALF / LAM), 2),
        "grid_size": NGRID,
        "gorkov_formula": "U = V_p*(f1*E_pot - 1.5*f2*E_kin)",
        "lens": {
            "ell": ELL_FIXED, "R_mm": R_FIXED * 1e3,
            "f_mm": FOCAL_FIXED * 1e3, "apodization": APOD_FIXED,
            "NA": round(float(NA_FIXED), 4),
        },
        "sweep": {
            "n_alpha": len(ALPHAS), "alphas": ALPHAS,
            "n_phase": N_PHASE,
            "n_x0": len(X0_OFFSETS), "n_y0": len(Y0_OFFSETS),
            "shift_fracs": SHIFT_FRACS.tolist(),
            "total_configs": int(total),
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
        "best": best_cfg,
        "bridge_formed": bool(bridge_formed),
        "barrier_reduction_pct": round(float(reduction), 2),
        "top10": top10,
        "nfac_vortex": round(float(nfac), 4),
    }

    with open(DAT / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    log("  Saved stats.json")

    # Save full sweep results
    with open(DAT / "sweep_results.json", "w") as f:
        json.dump(results, f, indent=1)
    log("  Saved sweep_results.json")

    np.savez_compressed(DAT / "fields.npz",
                        xg=xg, yg=yg,
                        U_s=U_s, best_U=best_U,
                        U_line_base=U_line_base,
                        best_U_line=best_U_line,
                        frac=frac_base)
    log("  Saved fields.npz")

    # ── REPORT.md ─────────────────────────────────────────────────
    elapsed = time.time() - t_GLOBAL
    log(f"\nTotal runtime: {elapsed:.1f}s")

    rp = []
    rp.append("# Bridge Formation Study — Phase & Lateral-Offset Sweep\n")
    rp.append(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    rp.append(f"**Runtime**: {elapsed:.1f}s")
    rp.append(f"**Cache**: `{cache_path.name}` "
              f"({coords.shape[0]:,} DOFs)\n")

    rp.append("## Configuration\n")
    rp.append(f"- **ROI**: ±{ROI_HALF*1e3:.3f} mm ({2*ROI_HALF/LAM:.1f}λ), "
              f"{NGRID}×{NGRID} grid")
    rp.append(f"- **Lens**: ℓ={ELL_FIXED}, f={FOCAL_FIXED*1e3:.1f} mm, "
              f"R={R_FIXED*1e3:.1f} mm, {APOD_FIXED}, NA={NA_FIXED:.4f}")
    rp.append(f"- **z*** = {Z_STAR*1e3:.4f} mm")
    rp.append(f"- **λ** = {LAM*1e3:.4f} mm, trap spacing = {TRAP_SP*1e3:.4f} mm")
    rp.append(f"- **Gorkov**: U = V_p·(f₁·E_pot − 1.5·f₂·E_kin)")
    rp.append(f"- **nfac** = {nfac:.3f}\n")

    rp.append("## Sweep Parameters\n")
    rp.append(f"- **α** ∈ {{{', '.join(f'{a:.2f}' for a in ALPHAS)}}}")
    rp.append(f"- **φ₀** ∈ {N_PHASE} values in [0, 2π)")
    rp.append(f"- **Lateral offsets** (x₀, y₀): "
              f"{len(X0_OFFSETS)}×{len(Y0_OFFSETS)} grid, "
              f"fracs = {SHIFT_FRACS.tolist()}")
    rp.append(f"- **Total configs**: {total}\n")
    rp.append(f"Field combination: `p_total = p_stand + α·exp(iφ₀)·p̂_vort`\n")

    rp.append("## Bridge Metric\n")
    rp.append("Two adjacent traps selected near ROI centre.")
    rp.append(f"- Trap 1: ({xg[trap1[1]]*1e3:.3f}, {yg[trap1[0]]*1e3:.3f}) mm")
    rp.append(f"- Trap 2: ({xg[trap2[1]]*1e3:.3f}, {yg[trap2[0]]*1e3:.3f}) mm")
    rp.append(f"- Separation: {sep*1e3:.3f} mm ({sep/TRAP_SP:.2f} × λ/2)\n")
    rp.append("B = U_saddle − max(U_min1, U_min2)")
    rp.append("- B > 0 → separate traps")
    rp.append("- B ≤ 0 → bridge (connected basin)\n")

    rp.append("## Results\n")
    rp.append(f"**Baseline** barrier: B = {B_baseline:.3e} J "
              f"({B_baseline/trap_depth*100:.2f}% of depth)\n")
    rp.append(f"**Best** barrier: B = {best_B:.3e} J "
              f"({best_B/trap_depth*100:+.2f}% of depth)")
    rp.append(f"- α = {best_cfg['alpha']:.2f}")
    rp.append(f"- φ₀ = {best_cfg['phi0']:.3f} rad "
              f"({np.degrees(best_cfg['phi0']):.1f}°)")
    rp.append(f"- x₀ = {best_cfg['x0_frac']:+.1f} × trap_sp "
              f"({best_cfg['x0_mm']:+.4f} mm)")
    rp.append(f"- y₀ = {best_cfg['y0_frac']:+.1f} × trap_sp "
              f"({best_cfg['y0_mm']:+.4f} mm)\n")
    rp.append(f"**Barrier reduction**: {reduction:.1f}%")
    rp.append(f"**Bridge formed** (B ≤ 0): {'YES ✓' if bridge_formed else 'NO'}\n")

    rp.append("### Top 10 configurations\n")
    rp.append("| Rank | α | φ₀ (deg) | x₀/sp | y₀/sp | B (J) | B/depth % |")
    rp.append("|------|---|----------|-------|-------|-------|-----------|")
    for i, r in enumerate(top10):
        rp.append(f"| {i+1} | {r['alpha']:.2f} | "
                  f"{np.degrees(r['phi0']):.0f} | "
                  f"{r['x0_frac']:+.1f} | {r['y0_frac']:+.1f} | "
                  f"{r['B']:.2e} | {r['B_rel']*100:+.1f} |")

    rp.append("\n## Figures\n")
    rp.append("![gorkov_best_bridge](figures/gorkov_best_bridge.png)\n")
    rp.append("![bridge_metric_heatmap](figures/bridge_metric_heatmap.png)\n")
    rp.append("![example_xy_fields](figures/example_xy_fields.png)\n")

    rp.append("## Interpretation\n")
    if bridge_formed:
        rp.append("A true bridge (B ≤ 0) was achieved. The inter-trap barrier "
                  "has been eliminated for the best configuration, creating a "
                  "continuous Gor'kov potential channel between adjacent traps.")
    elif reduction > 50:
        rp.append(f"Significant barrier reduction ({reduction:.0f}%) was achieved "
                  "but a true bridge was not formed. The phase and lateral offset "
                  "substantially modify the interference pattern. Further increases "
                  "in α or finer offset grids may reach B ≤ 0.")
    else:
        rp.append(f"Modest barrier reduction ({reduction:.0f}%). "
                  "The vortex perturbation modifies trap depths but "
                  "does not bridge adjacent traps at these α values. "
                  "The vortex waist diameter ({:.2f} × λ/2) may be too large "
                  "to produce localised inter-trap effects. Smaller waist or "
                  "higher α values may be needed.".format(
                      float(0.618 / (TRAP_SP * 1e3)) * 1e3))

    with open(OUT / "REPORT.md", "w") as f:
        f.write("\n".join(rp) + "\n")
    log("  Saved REPORT.md")

    log("\n" + "=" * 72)
    log("DONE")
    log("=" * 72)
    log_fp.close()


if __name__ == "__main__":
    main()
