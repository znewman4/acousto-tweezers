#!/usr/bin/env python3
"""
Quick ASM Vortex Overlay with Cached FEM Standing Wave
=======================================================

KDTree-based IDW sampling of the FEM cache — avoids the y-slab projection
artefact entirely by querying the unstructured mesh at exact 3D plane points.

Usage:
    python scripts/dev/quick_asm_vortex_overlay_with_fem_standing.py
    python scripts/dev/quick_asm_vortex_overlay_with_fem_standing.py --cache path/to.npz
    python scripts/dev/quick_asm_vortex_overlay_with_fem_standing.py --ell 1 --aperture 3e-3

Outputs → results/quick_overlay_YYYYMMDD_HHMMSS/
"""

import argparse
import json
import time
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ── project imports ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from acoustweezers.physics.acoustics.angular_spectrum import propagate_pressure_asm

# ═══════════════════════════════════════════════════════════════════
# PHYSICAL CONSTANTS (CORRECTED_PRESET defaults, overridable via cache)
# ═══════════════════════════════════════════════════════════════════
C_WATER   = 1484.0
RHO_WATER = 997.0
F_HZ      = 2.0e6
OMEGA     = 2.0 * np.pi * F_HZ
K_WATER   = OMEGA / C_WATER
LAM       = C_WATER / F_HZ           # ≈ 0.742 mm

LX        = 6.0e-3
LY        = 6.0e-3
H_UNDER   = 3.0e-3
H_TOP     = 2.0085e-3
H_TOTAL   = H_UNDER + H_TOP

# PML thickness (1 λ each side)
T_PML_XY  = 1.0 * LAM
T_PML_Z   = 1.0 * LAM

TRAP_SPACING = LAM / 2.0
CX, CY    = LX / 2, LY / 2
Z_STAR    = H_UNDER + H_TOP / 2.0 + 0.25 * LAM   # standing-wave observation plane

# ── Default vortex lens parameters ────────────────────────────────
DEFAULT_ELL        = 2
DEFAULT_APERTURE_R = 5.0e-3
DEFAULT_FOCAL      = 4.0e-3

# ── Grid parameters ───────────────────────────────────────────────
NGRID_XY   = 500        # XY slice resolution per axis
NGRID_XZ_X = 500        # XZ slice: x-axis points
NGRID_XZ_Z = 300        # XZ slice: z-axis points
ASM_N      = 512        # ASM FFT grid
ASM_PAD    = 2          # zero-padding factor

# ── Mixing ratios ────────────────────────────────────────────────
ALPHAS = [0.05, 0.1, 0.2]

STANDING_CACHE_DIR = PROJECT_ROOT / "results" / "fem_standing_wave_cache"


# ═══════════════════════════════════════════════════════════════════
# KDTree IDW SAMPLER
# ═══════════════════════════════════════════════════════════════════

def sample_complex_field_idw(tree, coords, p, query_pts, k=16, power=2.0, eps=1e-12):
    """
    KNN inverse-distance weighting for complex p at query points.

    Parameters
    ----------
    tree : cKDTree built from coords
    coords : (N, 3) source coordinates (unused except for reference)
    p : (N,) complex128 source field values
    query_pts : (M, 3) evaluation points
    k : int, number of nearest neighbours
    power : float, IDW exponent (2.0 = standard)
    eps : float, regularisation to avoid 1/0

    Returns
    -------
    (M,) complex128 interpolated values
    """
    if k == 1:
        dists, inds = tree.query(query_pts, k=1)
        return p[inds]

    dists, inds = tree.query(query_pts, k=k)       # (M, k)
    weights = 1.0 / (dists**power + eps)            # (M, k)
    weights /= weights.sum(axis=1, keepdims=True)   # normalize
    p_nn = p[inds]                                   # (M, k) complex
    return (p_nn * weights).sum(axis=1)              # (M,) complex


# ═══════════════════════════════════════════════════════════════════
# CACHE LOADER
# ═══════════════════════════════════════════════════════════════════

def _find_latest_cache(cache_dir: Path) -> Path:
    npz_files = sorted(cache_dir.glob("*.npz"), key=lambda p: p.stat().st_mtime)
    if not npz_files:
        raise FileNotFoundError(f"No .npz files in {cache_dir}")
    return npz_files[-1]


def load_fem_cache(path: Path):
    """Load cached FEM field.  Returns (coords, p_complex, meta)."""
    d = np.load(path, allow_pickle=True)
    keys = list(d.keys())

    # Robust key detection
    if "coords" in keys:
        coords = d["coords"]
    else:
        raise KeyError(f"Cannot find coordinates in NPZ (keys: {keys})")

    if "p_real" in keys and "p_imag" in keys:
        p = d["p_real"].astype(np.float64) + 1j * d["p_imag"].astype(np.float64)
    elif "p" in keys:
        p = d["p"].astype(np.complex128)
    else:
        raise KeyError(f"Cannot find pressure data in NPZ (keys: {keys})")

    meta = {}
    skip = {"coords", "p_real", "p_imag", "p"}
    for k in keys:
        if k not in skip:
            val = d[k]
            if isinstance(val, np.ndarray) and val.ndim == 0:
                val = val.item()
            meta[k] = val

    return coords, p, meta


# ═══════════════════════════════════════════════════════════════════
# LOGGING HELPER
# ═══════════════════════════════════════════════════════════════════

class Logger:
    def __init__(self, fp=None):
        self._fp = fp
    def __call__(self, msg=""):
        print(msg)
        if self._fp:
            self._fp.write(msg + "\n")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Quick ASM vortex overlay with cached FEM standing wave")
    parser.add_argument("--cache", type=str, default=None,
                        help="Path to FEM cache .npz (default: auto-detect latest)")
    parser.add_argument("--ell", type=int, default=DEFAULT_ELL,
                        help=f"Topological charge (default: {DEFAULT_ELL})")
    parser.add_argument("--aperture", type=float, default=DEFAULT_APERTURE_R,
                        help=f"Aperture radius [m] (default: {DEFAULT_APERTURE_R})")
    parser.add_argument("--focal", type=float, default=DEFAULT_FOCAL,
                        help=f"Focal length [m] (default: {DEFAULT_FOCAL})")
    parser.add_argument("--k-idw", type=int, default=16,
                        help="KNN neighbours for IDW sampling (default: 16)")
    parser.add_argument("--timestamp", type=str,
                        default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()

    t_START = time.time()

    # ── Locate cache ──────────────────────────────────────────────
    if args.cache:
        cache_path = Path(args.cache).resolve()
    else:
        cache_path = _find_latest_cache(STANDING_CACHE_DIR)
    if not cache_path.exists():
        print(f"ERROR: Cache not found: {cache_path}")
        sys.exit(1)

    # ── Output directory ──────────────────────────────────────────
    out_dir = PROJECT_ROOT / "results" / f"quick_overlay_{args.timestamp}"
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "data").mkdir(parents=True, exist_ok=True)

    log_fp = open(out_dir / "console_log.txt", "w")
    log = Logger(log_fp)

    log("=" * 72)
    log("Quick ASM Vortex Overlay with Cached FEM Standing Wave")
    log("=" * 72)
    log(f"Date       : {datetime.now().isoformat()}")
    log(f"Cache      : {cache_path}")
    log(f"Output     : {out_dir}")
    log(f"Lens       : ℓ={args.ell}, R={args.aperture*1e3:.2f} mm, "
        f"f={args.focal*1e3:.2f} mm")
    log(f"IDW k      : {args.k_idw}")
    log()

    # ══════════════════════════════════════════════════════════════
    # STEP 1: Load FEM cache + build KDTree
    # ══════════════════════════════════════════════════════════════
    log("─" * 72)
    log("STEP 1: Load FEM cache + build KDTree")
    log("─" * 72)

    t0 = time.time()
    coords, p_fem, meta = load_fem_cache(cache_path)
    t_load = time.time() - t0
    log(f"  Loaded: {coords.shape[0]} DOFs in {t_load:.2f}s")
    log(f"  NPZ keys (metadata): {list(meta.keys())}")

    # Extract or assume physical constants from metadata
    lam = float(meta.get("wavelength", LAM))
    freq = float(meta.get("frequency_hz", F_HZ))
    c_w = float(meta.get("c_water", C_WATER))
    k_w = 2 * np.pi * freq / c_w
    trap_sp = lam / 2.0

    assumptions = []
    if "wavelength" not in meta:
        assumptions.append(f"λ assumed {lam*1e3:.4f} mm (2 MHz, c=1484)")
    if "frequency_hz" not in meta:
        assumptions.append(f"f assumed {freq/1e6:.1f} MHz")

    log(f"  λ = {lam*1e3:.4f} mm, f = {freq/1e6:.1f} MHz, k = {k_w:.2f} rad/m")
    log(f"  max|p| = {np.abs(p_fem).max():.4f} Pa")

    # Domain extents
    x_lo, x_hi = coords[:, 0].min(), coords[:, 0].max()
    y_lo, y_hi = coords[:, 1].min(), coords[:, 1].max()
    z_lo, z_hi = coords[:, 2].min(), coords[:, 2].max()
    cx = (x_lo + x_hi) / 2.0
    cy = (y_lo + y_hi) / 2.0
    log(f"  Domain: x=[{x_lo*1e3:.3f}, {x_hi*1e3:.3f}], "
        f"y=[{y_lo*1e3:.3f}, {y_hi*1e3:.3f}], z=[{z_lo*1e3:.3f}, {z_hi*1e3:.3f}] mm")

    # Build KDTree
    t0 = time.time()
    tree = cKDTree(coords)
    t_tree = time.time() - t0
    log(f"  KDTree built in {t_tree:.2f}s")

    # Quick sampling quality check
    rng = np.random.default_rng(42)
    probe = coords[rng.choice(len(coords), min(5000, len(coords)), replace=False)]
    dd_probe, _ = tree.query(probe, k=2)
    median_nn = np.median(dd_probe[:, 1])
    log(f"  Median NN distance: {median_nn*1e3:.4f} mm (λ/{lam/median_nn:.1f})")
    if median_nn > lam / 3:
        log(f"  ⚠ WARNING: mesh is coarse relative to λ — sampling may smooth peaks")
    log()

    # ══════════════════════════════════════════════════════════════
    # STEP 2: Define ROI + structured grids
    # ══════════════════════════════════════════════════════════════
    log("─" * 72)
    log("STEP 2: Define ROI + structured grids")
    log("─" * 72)

    # ROI: ±1.1λ around center → see ~3 trap peaks each direction
    roi_half = 1.5 * trap_sp + 0.35 * lam   # ≈ 1.1λ
    roi_x = (cx - roi_half, cx + roi_half)
    roi_y = (cy - roi_half, cy + roi_half)

    # z-planes for XY slices
    z_star = H_UNDER + H_TOP / 2.0 + 0.25 * lam   # standing-wave observation
    z_waist = args.focal                             # nominal focal (ASM waist)

    # XZ z-range: avoid PML, cover physical domain
    z_phys_lo = T_PML_Z + 0.1 * lam
    z_phys_hi = H_TOTAL - 0.1 * lam

    log(f"  ROI XY: x=[{roi_x[0]*1e3:.4f}, {roi_x[1]*1e3:.4f}] mm  "
        f"({2*roi_half/lam:.2f}λ wide)")
    log(f"  z* (standing obs): {z_star*1e3:.4f} mm")
    log(f"  z_waist (focal)  : {z_waist*1e3:.4f} mm")
    log(f"  XZ z-range       : [{z_phys_lo*1e3:.4f}, {z_phys_hi*1e3:.4f}] mm")

    # XY grids
    xg_xy = np.linspace(roi_x[0], roi_x[1], NGRID_XY)
    yg_xy = np.linspace(roi_y[0], roi_y[1], NGRID_XY)
    XX_xy, YY_xy = np.meshgrid(xg_xy, yg_xy)

    # XZ grid
    xg_xz = np.linspace(roi_x[0], roi_x[1], NGRID_XZ_X)
    zg_xz = np.linspace(z_phys_lo, z_phys_hi, NGRID_XZ_Z)
    XX_xz, ZZ_xz = np.meshgrid(xg_xz, zg_xz)

    log(f"  XY grid: {NGRID_XY}×{NGRID_XY} = {NGRID_XY**2} pts")
    log(f"  XZ grid: {NGRID_XZ_X}×{NGRID_XZ_Z} = {NGRID_XZ_X * NGRID_XZ_Z} pts")
    log()

    # ══════════════════════════════════════════════════════════════
    # STEP 3: Sample FEM standing wave onto structured grids
    # ══════════════════════════════════════════════════════════════
    log("─" * 72)
    log("STEP 3: Sample FEM standing onto grids (KDTree IDW)")
    log("─" * 72)

    k_idw = args.k_idw

    # XY at z_star
    pts_xy_zstar = np.column_stack([
        XX_xy.ravel(), YY_xy.ravel(), np.full(XX_xy.size, z_star)])
    t0 = time.time()
    p_stand_xy_zstar = sample_complex_field_idw(
        tree, coords, p_fem, pts_xy_zstar, k=k_idw).reshape(XX_xy.shape)
    t_s1 = time.time() - t0
    log(f"  XY at z*   : {t_s1:.2f}s, max|p|={np.abs(p_stand_xy_zstar).max():.4f}")

    # XY at z_waist
    pts_xy_zwaist = np.column_stack([
        XX_xy.ravel(), YY_xy.ravel(), np.full(XX_xy.size, z_waist)])
    t0 = time.time()
    p_stand_xy_zwaist = sample_complex_field_idw(
        tree, coords, p_fem, pts_xy_zwaist, k=k_idw).reshape(XX_xy.shape)
    t_s2 = time.time() - t0
    log(f"  XY at waist: {t_s2:.2f}s, max|p|={np.abs(p_stand_xy_zwaist).max():.4f}")

    # XZ at y=CY
    pts_xz = np.column_stack([
        XX_xz.ravel(), np.full(XX_xz.size, cy), ZZ_xz.ravel()])
    t0 = time.time()
    p_stand_xz = sample_complex_field_idw(
        tree, coords, p_fem, pts_xz, k=k_idw).reshape(XX_xz.shape)
    t_s3 = time.time() - t0
    log(f"  XZ at y=CY : {t_s3:.2f}s, max|p|={np.abs(p_stand_xz).max():.4f}")

    # Sampling quality: median KNN distance on each grid
    for label, pts in [("XY-z*", pts_xy_zstar), ("XZ", pts_xz)]:
        dd_q, _ = tree.query(pts[:min(5000, len(pts))], k=1)
        med_d = np.median(dd_q)
        log(f"  {label} median NN distance: {med_d*1e3:.4f} mm (λ/{lam/med_d:.1f})")
    log()

    # ══════════════════════════════════════════════════════════════
    # STEP 4: Generate ASM vortex
    # ══════════════════════════════════════════════════════════════
    log("─" * 72)
    log("STEP 4: Generate ASM vortex field")
    log("─" * 72)

    # Source plane — build converging vortex drive directly
    # (library uses POSITIVE sign = diverging; we need NEGATIVE = converging,
    #  matching the proven hourglass-proof convention:
    #     φ = ℓ·θ  −  k·(√(r²+f²) − f)       ← converging spherical wavefront)
    asm_domain = max(LX, 2 * args.aperture + 4 * lam)
    x0_asm = cx - asm_domain / 2.0
    y0_asm = cy - asm_domain / 2.0
    xg_asm = np.linspace(x0_asm, x0_asm + asm_domain, ASM_N, endpoint=False)
    yg_asm = np.linspace(y0_asm, y0_asm + asm_domain, ASM_N, endpoint=False)
    dx_asm = float(xg_asm[1] - xg_asm[0])
    dy_asm = float(yg_asm[1] - yg_asm[0])
    XX_asm, YY_asm = np.meshgrid(xg_asm, yg_asm)

    t0 = time.time()
    rx = XX_asm - cx
    ry = YY_asm - cy
    r_asm = np.sqrt(rx**2 + ry**2)
    theta_asm = np.arctan2(ry, rx)

    phi_vortex = args.ell * theta_asm
    # NEGATIVE sign → converging spherical wavefront (matches hourglass proof)
    phi_focus = -k_w * (np.sqrt(r_asm**2 + args.focal**2) - args.focal)
    phi_target = phi_vortex + phi_focus

    # Cosine-taper apodization within aperture
    r_norm = r_asm / args.aperture
    A_asm = np.where(r_norm <= 1.0, 0.5 * (1.0 + np.cos(np.pi * r_norm)), 0.0)
    D_asm = A_asm * np.exp(1j * phi_target)
    t_lens = time.time() - t0

    nf_lens = args.aperture**2 / (lam * args.focal)
    log(f"  Converging vortex drive built in {t_lens:.3f}s")
    log(f"  Sign convention: φ_focus = −k(√(r²+f²)−f)  [CONVERGING]")
    log(f"  ℓ={args.ell}, R={args.aperture*1e3:.1f}mm, "
        f"f={args.focal*1e3:.1f}mm, N_F={nf_lens:.2f}")
    log(f"  ASM domain: {asm_domain*1e3:.2f} mm, grid {ASM_N}×{ASM_N}, "
        f"dx={dx_asm*1e3:.4f} mm")

    # ── XY vortex at z_star ───────────────────────────────────────
    t0 = time.time()
    p_vortex_full_zstar = propagate_pressure_asm(
        D_asm, dx_asm, dy_asm, k_w, z_star,
        pad_factor=ASM_PAD, include_evanescent=True)
    # Interpolate from ASM grid onto ROI grid
    from scipy.interpolate import RegularGridInterpolator
    ire = RegularGridInterpolator((yg_asm, xg_asm), np.real(p_vortex_full_zstar),
                                   bounds_error=False, fill_value=0.0)
    iim = RegularGridInterpolator((yg_asm, xg_asm), np.imag(p_vortex_full_zstar),
                                   bounds_error=False, fill_value=0.0)
    pts_yx = np.column_stack([YY_xy.ravel(), XX_xy.ravel()])
    p_vortex_xy_zstar = (ire(pts_yx) + 1j * iim(pts_yx)).reshape(XX_xy.shape)
    t_v1 = time.time() - t0
    log(f"  Vortex XY at z*   : {t_v1:.2f}s, max|p|={np.abs(p_vortex_xy_zstar).max():.6f}")

    # ── XY vortex at z_waist ──────────────────────────────────────
    t0 = time.time()
    p_vortex_full_zwaist = propagate_pressure_asm(
        D_asm, dx_asm, dy_asm, k_w, z_waist,
        pad_factor=ASM_PAD, include_evanescent=True)
    ire2 = RegularGridInterpolator((yg_asm, xg_asm), np.real(p_vortex_full_zwaist),
                                    bounds_error=False, fill_value=0.0)
    iim2 = RegularGridInterpolator((yg_asm, xg_asm), np.imag(p_vortex_full_zwaist),
                                    bounds_error=False, fill_value=0.0)
    p_vortex_xy_zwaist = (ire2(pts_yx) + 1j * iim2(pts_yx)).reshape(XX_xy.shape)
    t_v2 = time.time() - t0
    log(f"  Vortex XY at waist: {t_v2:.2f}s, max|p|={np.abs(p_vortex_xy_zwaist).max():.6f}")

    # ── XZ vortex (propagate to each z-plane, slice y=CY) ────────
    log(f"  Propagating vortex to {NGRID_XZ_Z} z-planes for XZ...")
    t0 = time.time()
    p_vortex_xz = np.zeros((NGRID_XZ_Z, NGRID_XZ_X), dtype=complex)
    iy_cen = np.argmin(np.abs(yg_asm - cy))
    for iz, zz in enumerate(zg_xz):
        p_z = propagate_pressure_asm(
            D_asm, dx_asm, dy_asm, k_w, zz,
            pad_factor=ASM_PAD, include_evanescent=True)
        p_line = p_z[iy_cen, :]
        p_vortex_xz[iz, :] = np.interp(xg_xz, xg_asm, np.real(p_line)) + \
                              1j * np.interp(xg_xz, xg_asm, np.imag(p_line))
    t_vxz = time.time() - t0
    log(f"  Vortex XZ: {t_vxz:.1f}s, max|p|={np.abs(p_vortex_xz).max():.6f}")
    log()

    # ══════════════════════════════════════════════════════════════
    # STEP 5: Normalize vortex + overlay
    # ══════════════════════════════════════════════════════════════
    log("─" * 72)
    log("STEP 5: Overlay — p_total = p_stand + α·p̂_vortex")
    log("─" * 72)

    stand_peak = max(np.abs(p_stand_xy_zstar).max(),
                     np.abs(p_stand_xy_zwaist).max())
    vortex_peak_xy = max(np.abs(p_vortex_xy_zstar).max(),
                         np.abs(p_vortex_xy_zwaist).max())
    vortex_peak_xz = np.abs(p_vortex_xz).max()

    log(f"  Standing peak: {stand_peak:.4f} Pa")
    log(f"  Vortex peak (XY): {vortex_peak_xy:.6f}")
    log(f"  Vortex peak (XZ): {vortex_peak_xz:.6f}")

    # Normalize vortex to unit peak relative to standing
    def _norm(pv, scale):
        return pv / (np.abs(pv).max() + 1e-30) * scale

    pv_norm_xy_zstar  = _norm(p_vortex_xy_zstar, stand_peak)
    pv_norm_xy_zwaist = _norm(p_vortex_xy_zwaist, stand_peak)
    pv_norm_xz        = _norm(p_vortex_xz, stand_peak)

    # ── Perturbation analysis per alpha ──────────────────────────
    perturbation_results = {}
    for alpha in ALPHAS:
        p_tot_zstar = p_stand_xy_zstar + alpha * pv_norm_xy_zstar
        shift = np.abs(p_tot_zstar) - np.abs(p_stand_xy_zstar)
        max_shift_frac = np.abs(shift).max() / (stand_peak + 1e-30)
        perturbation_results[alpha] = {
            "max_shift_frac": float(max_shift_frac),
            "max_combined": float(np.abs(p_tot_zstar).max()),
        }
        log(f"  α={alpha}: max |Δ|p||/peak = {max_shift_frac:.4f} "
            f"({max_shift_frac*100:.1f}%), combined max = "
            f"{perturbation_results[alpha]['max_combined']:.4f}")
    log()

    # ══════════════════════════════════════════════════════════════
    # STEP 6: Figures
    # ══════════════════════════════════════════════════════════════
    log("─" * 72)
    log("STEP 6: Generate figures")
    log("─" * 72)

    mm_fmt = FuncFormatter(lambda x, _: f"{x:.2f}")
    ext_xy = [roi_x[0]*1e3, roi_x[1]*1e3, roi_y[0]*1e3, roi_y[1]*1e3]
    ext_xz = [roi_x[0]*1e3, roi_x[1]*1e3, z_phys_lo*1e3, z_phys_hi*1e3]

    def _annotate_traps_xy(ax):
        """Draw λ/2 grid centred on ROI."""
        for offset in np.arange(-5, 6) * trap_sp:
            ax.axhline((cy + offset)*1e3, color="white", lw=0.3, alpha=0.3)
            ax.axvline((cx + offset)*1e3, color="white", lw=0.3, alpha=0.3)
        ax.plot(cx*1e3, cy*1e3, "+", color="cyan", ms=8, mew=1.5)

    def _annotate_xz(ax):
        ax.axhline(z_star*1e3, color="cyan", lw=0.6, ls="--", alpha=0.6)
        ax.axhline(z_waist*1e3, color="lime", lw=0.6, ls=":", alpha=0.6)
        ax.plot(cx*1e3, z_star*1e3, "+", color="cyan", ms=8, mew=1.5)

    # ── Helper: 4-panel figure (standing / vortex / combined / diff)
    def plot_4panel(p_s, p_v_norm, alpha, extent, label, annotate_fn,
                    xlabel="x [mm]", ylabel="y [mm]"):
        p_t = p_s + alpha * p_v_norm
        p_d = np.abs(p_t) - np.abs(p_s)
        vmax_s = np.abs(p_s).max()
        vmax_v = np.abs(alpha * p_v_norm).max()
        vmax_t = max(vmax_s, np.abs(p_t).max())

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        kw = dict(extent=extent, origin="lower", aspect="auto")

        ax = axes[0, 0]
        im = ax.imshow(np.abs(p_s), cmap="inferno", vmin=0, vmax=vmax_s, **kw)
        ax.set_title("|p_stand|"); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        annotate_fn(ax); plt.colorbar(im, ax=ax, shrink=0.8, label="Pa")

        ax = axes[0, 1]
        im = ax.imshow(np.abs(alpha * p_v_norm), cmap="viridis",
                        vmin=0, vmax=vmax_v, **kw)
        ax.set_title(f"|α·p̂_vortex| (α={alpha})")
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        annotate_fn(ax); plt.colorbar(im, ax=ax, shrink=0.8, label="Pa")

        ax = axes[1, 0]
        im = ax.imshow(np.abs(p_t), cmap="inferno", vmin=0, vmax=vmax_t, **kw)
        ax.set_title("|p_combined|"); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        annotate_fn(ax); plt.colorbar(im, ax=ax, shrink=0.8, label="Pa")

        ax = axes[1, 1]
        dmax = max(abs(p_d.min()), abs(p_d.max())) or 1.0
        im = ax.imshow(p_d, cmap="RdBu_r", vmin=-dmax, vmax=dmax, **kw)
        ax.set_title("|p_combined| − |p_stand|")
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        annotate_fn(ax); plt.colorbar(im, ax=ax, shrink=0.8, label="ΔPa")

        fig.suptitle(f"{label}  (α={alpha},  ℓ={args.ell}, R={args.aperture*1e3:.1f}mm)",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()
        return fig

    # Generate XY figures at z_star
    for alpha in ALPHAS:
        fig = plot_4panel(p_stand_xy_zstar, pv_norm_xy_zstar, alpha,
                          ext_xy, f"XY at z*={z_star*1e3:.3f}mm",
                          _annotate_traps_xy)
        fname = f"XY_zstar_alpha{alpha}.png"
        fig.savefig(out_dir / "figures" / fname, dpi=150)
        plt.close(fig)
        log(f"  Saved {fname}")

    # XY at z_waist
    for alpha in ALPHAS:
        fig = plot_4panel(p_stand_xy_zwaist, pv_norm_xy_zwaist, alpha,
                          ext_xy, f"XY at z_waist={z_waist*1e3:.3f}mm",
                          _annotate_traps_xy)
        fname = f"XY_zwaist_alpha{alpha}.png"
        fig.savefig(out_dir / "figures" / fname, dpi=150)
        plt.close(fig)
        log(f"  Saved {fname}")

    # XZ figures
    for alpha in ALPHAS:
        fig = plot_4panel(p_stand_xz, pv_norm_xz, alpha,
                          ext_xz, f"XZ at y=CY={cy*1e3:.3f}mm",
                          _annotate_xz, xlabel="x [mm]", ylabel="z [mm]")
        fname = f"XZ_alpha{alpha}.png"
        fig.savefig(out_dir / "figures" / fname, dpi=150)
        plt.close(fig)
        log(f"  Saved {fname}")

    # ── Combined summary figure (α=0.1 only) ─────────────────────
    alpha_summary = 0.1
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    p_t_zs = p_stand_xy_zstar + alpha_summary * pv_norm_xy_zstar
    p_t_zw = p_stand_xy_zwaist + alpha_summary * pv_norm_xy_zwaist
    p_t_xz = p_stand_xz + alpha_summary * pv_norm_xz

    slices = [
        (axes[0, 0], np.abs(p_stand_xy_zstar), ext_xy, f"|p_stand| XY z*", "inferno"),
        (axes[0, 1], np.abs(p_t_zs), ext_xy, f"|p_combined| XY z* (α={alpha_summary})", "inferno"),
        (axes[0, 2], np.abs(p_t_zs) - np.abs(p_stand_xy_zstar), ext_xy, "Δ|p| XY z*", "RdBu_r"),
        (axes[1, 0], np.abs(p_stand_xz), ext_xz, "|p_stand| XZ", "inferno"),
        (axes[1, 1], np.abs(p_t_xz), ext_xz, f"|p_combined| XZ (α={alpha_summary})", "inferno"),
        (axes[1, 2], np.abs(p_t_xz) - np.abs(p_stand_xz), ext_xz, "Δ|p| XZ", "RdBu_r"),
    ]
    for ax, data, ext, title, cmap in slices:
        if cmap == "RdBu_r":
            dmax = max(abs(data.min()), abs(data.max())) or 1.0
            im = ax.imshow(data, extent=ext, origin="lower", aspect="auto",
                           cmap=cmap, vmin=-dmax, vmax=dmax)
        else:
            im = ax.imshow(data, extent=ext, origin="lower", aspect="auto",
                           cmap=cmap, vmin=0)
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.8)

    # Annotate
    for ax in axes[0, :]:
        _annotate_traps_xy(ax)
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    for ax in axes[1, :]:
        _annotate_xz(ax)
        ax.set_xlabel("x [mm]"); ax.set_ylabel("z [mm]")

    fig.suptitle(f"Summary — ℓ={args.ell}, R={args.aperture*1e3:.1f}mm, "
                 f"f={args.focal*1e3:.1f}mm, α={alpha_summary}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "summary_overview.png", dpi=150)
    plt.close(fig)
    log(f"  Saved summary_overview.png")
    log()

    # ══════════════════════════════════════════════════════════════
    # STEP 7: Save data + sanity checks
    # ══════════════════════════════════════════════════════════════
    log("─" * 72)
    log("STEP 7: Save data + sanity checks")
    log("─" * 72)

    # Save field data
    np.savez_compressed(out_dir / "data" / "fields.npz",
                        xg_xy=xg_xy, yg_xy=yg_xy,
                        xg_xz=xg_xz, zg_xz=zg_xz,
                        p_stand_xy_zstar=p_stand_xy_zstar,
                        p_stand_xy_zwaist=p_stand_xy_zwaist,
                        p_stand_xz=p_stand_xz,
                        p_vortex_xy_zstar=p_vortex_xy_zstar,
                        p_vortex_xy_zwaist=p_vortex_xy_zwaist,
                        p_vortex_xz=p_vortex_xz)
    log(f"  Saved fields.npz")

    # Quick FFT check on XY standing wave
    dx_xy = xg_xy[1] - xg_xy[0]
    p_mag = np.abs(p_stand_xy_zstar)
    fft2 = np.fft.fft2(p_mag - p_mag.mean())
    fft_mag = np.abs(np.fft.fftshift(fft2))
    kx_full = np.fft.fftshift(np.fft.fftfreq(len(xg_xy), d=dx_xy)) * 2 * np.pi
    center = len(kx_full) // 2
    kx_pos = kx_full[center + 1:]
    power_kx = (fft_mag**2).sum(axis=0)[center + 1:]
    peak_kx = kx_pos[np.argmax(power_kx)]
    lam_fft = 2 * np.pi / (peak_kx + 1e-30)
    fft_err = abs(lam_fft - lam) / lam * 100
    # FFT bin resolution: Δk = 2π/L → Δλ ≈ λ²/L.  With ROI ~2.2λ, expect ~34% bin
    roi_width = roi_x[1] - roi_x[0]
    fft_bin_resolution_pct = (lam / roi_width) * 100  # ~45% for 2.2λ ROI
    log(f"  FFT dominant λ_x = {lam_fft*1e3:.4f} mm (error {fft_err:.1f}% from λ)")
    log(f"  FFT bin resolution: ~{fft_bin_resolution_pct:.0f}% of λ "
        f"(ROI={roi_width/lam:.1f}λ — small ROI limits spectral resolution)")

    # Sanity stats
    sanity = {
        "cache_file": str(cache_path.name),
        "cache_dofs": int(coords.shape[0]),
        "lambda_mm": float(lam * 1e3),
        "trap_spacing_mm": float(trap_sp * 1e3),
        "k_idw": int(k_idw),
        "median_nn_dist_mm": float(median_nn * 1e3),
        "median_nn_dist_over_lambda": float(median_nn / lam),
        "standing_max_zstar": float(np.abs(p_stand_xy_zstar).max()),
        "standing_median_zstar": float(np.median(np.abs(p_stand_xy_zstar))),
        "standing_max_xz": float(np.abs(p_stand_xz).max()),
        "vortex_max_xy_zstar": float(np.abs(p_vortex_xy_zstar).max()),
        "vortex_max_xz": float(np.abs(p_vortex_xz).max()),
        "vortex_standing_ratio_at_peak": float(
            np.abs(p_vortex_xy_zstar).max() / (np.abs(p_stand_xy_zstar).max() + 1e-30)),
        "fft_dominant_lambda_mm": float(lam_fft * 1e3),
        "fft_lambda_error_pct": float(fft_err),
        "lens_ell": int(args.ell),
        "lens_R_mm": float(args.aperture * 1e3),
        "lens_f_mm": float(args.focal * 1e3),
        "alphas": ALPHAS,
        "perturbation": {str(a): v for a, v in perturbation_results.items()},
        "z_star_mm": float(z_star * 1e3),
        "z_waist_mm": float(z_waist * 1e3),
        "roi_half_mm": float(roi_half * 1e3),
        "assumptions": assumptions,
    }

    with open(out_dir / "data" / "sanity_stats.json", "w") as f:
        json.dump(sanity, f, indent=2, default=str)
    log(f"  Saved sanity_stats.json")

    # Warnings
    warnings = []
    if median_nn / lam > 0.33:
        warnings.append(f"Mesh coarse: median NN = {median_nn/lam:.2f}λ (want <λ/3)")
    if fft_err > max(15, fft_bin_resolution_pct * 1.5):
        warnings.append(f"FFT λ mismatch: {fft_err:.1f}% error (beyond bin resolution)")
    if stand_peak < 1.0:
        warnings.append(f"Standing peak suspiciously low: {stand_peak:.4f} Pa")

    for w in warnings:
        log(f"  ⚠ {w}")
    if not warnings:
        log("  ✓ No warnings — all sanity checks clean")
    log()

    # ══════════════════════════════════════════════════════════════
    # STEP 8: Write REPORT.md
    # ══════════════════════════════════════════════════════════════
    elapsed = time.time() - t_START
    log(f"Total runtime: {elapsed:.1f}s")

    report_lines = []
    report_lines.append("# Quick ASM Vortex Overlay Report")
    report_lines.append(f"\n**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"**Runtime**: {elapsed:.1f}s")
    report_lines.append(f"**Cache**: `{cache_path.name}` ({coords.shape[0]:,} DOFs)")
    report_lines.append("")

    report_lines.append("## Method")
    report_lines.append("")
    report_lines.append("**KDTree KNN Inverse-Distance Weighting (IDW)**")
    report_lines.append("")
    report_lines.append(f"- K = {k_idw} nearest neighbours, power = 2.0")
    report_lines.append(f"- Query points placed at exact 3D plane coordinates "
                        f"(no y-slab projection)")
    report_lines.append(f"- Median nearest-neighbour distance: "
                        f"{median_nn*1e3:.4f} mm (λ/{lam/median_nn:.1f})")
    report_lines.append("")
    report_lines.append("### Why this avoids the previous artefact")
    report_lines.append("")
    report_lines.append("The previous XZ slice used a **2D slab projection**: "
                        "DOFs within ±3·h_elem of y=CY were collected, their "
                        "y-coordinate discarded, and a 2D Delaunay triangulation "
                        "built in (x,z). This mixed DOFs at different y-positions "
                        "(with very different standing-wave phases) into the same "
                        "2D plane, causing a checkerboard/blocky texture.")
    report_lines.append("")
    report_lines.append("This method instead queries the 3D KDTree at the exact "
                        "point (x, y=CY, z) for each pixel. The IDW weighting "
                        "naturally favours DOFs closest to the query point in "
                        "all three dimensions, so off-plane DOFs contribute "
                        "proportionally to their actual 3D distance — no "
                        "projection artefact is possible.")
    report_lines.append("")

    report_lines.append("## Parameters")
    report_lines.append("")
    report_lines.append("| Parameter | Value |")
    report_lines.append("|-----------|-------|")
    report_lines.append(f"| λ | {lam*1e3:.4f} mm |")
    report_lines.append(f"| f | {freq/1e6:.1f} MHz |")
    report_lines.append(f"| Trap spacing (λ/2) | {trap_sp*1e3:.4f} mm |")
    report_lines.append(f"| ROI half-width | {roi_half*1e3:.4f} mm ({roi_half/lam:.2f}λ) |")
    report_lines.append(f"| z* (standing plane) | {z_star*1e3:.4f} mm |")
    report_lines.append(f"| z_waist (focal) | {z_waist*1e3:.4f} mm |")
    report_lines.append(f"| Lens ℓ | {args.ell} |")
    report_lines.append(f"| Lens R | {args.aperture*1e3:.2f} mm |")
    report_lines.append(f"| Lens f | {args.focal*1e3:.2f} mm |")
    report_lines.append(f"| Grid XY | {NGRID_XY}×{NGRID_XY} |")
    report_lines.append(f"| Grid XZ | {NGRID_XZ_X}×{NGRID_XZ_Z} |")
    report_lines.append(f"| IDW k | {k_idw} |")
    report_lines.append(f"| α values | {ALPHAS} |")
    if assumptions:
        for a in assumptions:
            report_lines.append(f"| Assumption | {a} |")

    report_lines.append("")
    report_lines.append("## Key Metrics")
    report_lines.append("")
    report_lines.append("| Metric | Value |")
    report_lines.append("|--------|-------|")
    report_lines.append(f"| Standing max (z*) | {sanity['standing_max_zstar']:.4f} Pa |")
    report_lines.append(f"| Standing max (XZ) | {sanity['standing_max_xz']:.4f} Pa |")
    report_lines.append(f"| Vortex max (z*) | {sanity['vortex_max_xy_zstar']:.6f} |")
    report_lines.append(f"| Vortex/Standing ratio | "
                        f"{sanity['vortex_standing_ratio_at_peak']:.6f} |")
    report_lines.append(f"| FFT dominant λ | {sanity['fft_dominant_lambda_mm']:.4f} mm "
                        f"({sanity['fft_lambda_error_pct']:.1f}% error) |")
    for alpha in ALPHAS:
        pr = perturbation_results[alpha]
        report_lines.append(f"| α={alpha} max shift | "
                            f"{pr['max_shift_frac']*100:.1f}% |")

    report_lines.append("")
    if warnings:
        report_lines.append("## Warnings")
        for w in warnings:
            report_lines.append(f"- ⚠ {w}")
        report_lines.append("")

    report_lines.append("## Verdict")
    report_lines.append("")
    if not warnings:
        report_lines.append("**Standing cache looks consistent; overlay pipeline "
                            "ready for fast ASM lens iteration.**")
    else:
        report_lines.append("Standing cache usable but with noted warnings above. "
                            "Check mesh resolution if artefacts persist.")
    report_lines.append("")

    report_lines.append("## Figures")
    report_lines.append("")
    for fig_path in sorted((out_dir / "figures").glob("*.png")):
        report_lines.append(f"- [{fig_path.name}](figures/{fig_path.name})")

    report_path = out_dir / "REPORT.md"
    report_path.write_text("\n".join(report_lines))
    log(f"Report: {report_path}")
    log()
    log("=" * 72)
    log("DONE")
    log("=" * 72)

    log_fp.close()


if __name__ == "__main__":
    main()
