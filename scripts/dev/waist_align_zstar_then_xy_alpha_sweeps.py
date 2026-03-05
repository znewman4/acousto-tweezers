#!/usr/bin/env python3
"""
Small-ROI Waist-Align (z0-offset) + XY α Sweeps
=================================================

Steps:
  1. Fix lens config to proven best (ℓ=1, f=4.2mm, R=7.0mm, uniform).
     Sweep a single scalar z0_offset ∈ ±1 mm to shift the vortex waist
     onto z* in the dish coordinate frame.
  2. Hourglass verification gate: ring radius & intensity vs z.
  3. Pressure XY α sweep on small ROI (3×3 traps).
  4. Gor'kov XY α sweep on small ROI.
  5. Stats + REPORT.md.

All analysis uses a small ROI of ~2λ ≈ 1.5 mm showing 3×3 trap peaks.
No lens-family sweeps, no large ROIs, no XZ figures, no 3D exports.

Outputs → results/waist_align_zstar_then_xy_alpha_sweeps_YYYYMMDD_HHMMSS/

Usage:
    python scripts/dev/waist_align_zstar_then_xy_alpha_sweeps.py
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
from scipy.ndimage import maximum_filter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
H_TOTAL   = H_UNDER + H_TOP
T_PML     = 1.0 * LAM
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

# ── Fixed lens config (proven winner from previous run) ──────────
ELL_FIXED   = 1
FOCAL_FIXED = 4.2e-3        # m
R_FIXED     = 7.0e-3        # m
APOD_FIXED  = "uniform"

# ── Small ROI: ~2λ → shows 3×3 trap peaks ────────────────────────
ROI_HALF = 1.1 * LAM        # ~0.816 mm → ROI width ≈ 1.63 mm ≈ 2.2λ
NGRID    = 400               # 400×400 → ~245 pts/λ (>>20 pts/λ)

# α sweep
ALPHAS = [0.02, 0.05, 0.1, 0.2]

# Gradient step
DZ_GRAD = LAM / 15.0        # ≈ 0.049 mm

# z0_offset sweep range
Z0_OFFSETS = np.linspace(-1.0e-3, 1.0e-3, 41)

STANDING_CACHE_DIR = PROJECT_ROOT / "results" / "fem_standing_wave_cache"

# ═══════════════════════════════════════════════════════════════════
# Helpers (reused from waist_fix_overlay_alpha_sweep.py)
# ═══════════════════════════════════════════════════════════════════

def build_converging_drive(ell, focal, aperture, cx, cy,
                           asm_domain, asm_n, k, apod="cosine_taper"):
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


def radial_profile(I_2d, xg, yg, cx, cy, r_max=None, n_bins=300):
    XX, YY = np.meshgrid(xg, yg)
    R = np.sqrt((XX - cx)**2 + (YY - cy)**2)
    if r_max is None:
        r_max = 0.5 * min(xg[-1] - xg[0], yg[-1] - yg[0])
    r_bins = np.linspace(0, r_max, n_bins + 1)
    r_mid  = 0.5 * (r_bins[:-1] + r_bins[1:])
    I_mean = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (R >= r_bins[i]) & (R < r_bins[i + 1])
        if np.any(mask):
            I_mean[i] = np.mean(I_2d[mask])
    return r_mid, I_mean


def find_ring(I_2d, xg, yg, cx, cy, r_max, r_min=0.02e-3, n_bins=300):
    r_mid, I_prof = radial_profile(I_2d, xg, yg, cx, cy,
                                   r_max=r_max, n_bins=n_bins)
    mask = r_mid >= r_min
    if not np.any(mask):
        return 0.0, 0.0
    idx = np.where(mask)[0]
    ipk = idx[np.argmax(I_prof[idx])]
    return r_mid[ipk], I_prof[ipk]


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

    OUT = PROJECT_ROOT / "results" / f"waist_align_zstar_then_xy_alpha_sweeps_{TS}"
    FIG = OUT / "figures"; FIG.mkdir(parents=True, exist_ok=True)
    DAT = OUT / "data";    DAT.mkdir(parents=True, exist_ok=True)

    log_fp = open(OUT / "console_log.txt", "w")
    log = Log(log_fp)

    log("=" * 72)
    log("Small-ROI Waist-Align (z0-offset) + XY α Sweeps")
    log("=" * 72)
    log(f"Date  : {datetime.now().isoformat()}")
    log(f"Cache : {cache_path}")
    log(f"Output: {OUT}")
    log(f"λ = {LAM*1e3:.4f} mm   k = {K_WATER:.1f} rad/m   "
        f"trap λ/2 = {TRAP_SP*1e3:.4f} mm")
    log(f"Particle: polystyrene  a = {A_P*1e6:.0f} µm  "
        f"ρ_p = {RHO_P:.0f}  c_p = {C_P:.0f}")
    log(f"Contrast: f₁ = {F1:.4f}   f₂ = {F2:.4f}")
    log(f"z* = {Z_STAR*1e3:.4f} mm")
    log(f"Fixed lens: ℓ={ELL_FIXED}  f={FOCAL_FIXED*1e3:.1f}mm  "
        f"R={R_FIXED*1e3:.1f}mm  apod={APOD_FIXED}")
    NA_FIXED = R_FIXED / np.sqrt(R_FIXED**2 + FOCAL_FIXED**2)
    log(f"NA = {NA_FIXED:.4f}")
    log(f"Small ROI: ±{ROI_HALF*1e3:.3f} mm "
        f"({2*ROI_HALF/LAM:.1f}λ)  grid {NGRID}×{NGRID}")
    log()

    # ── Build ASM drive (once) ────────────────────────────────────
    asm_domain = max(LX, 2 * R_FIXED + 4 * LAM)
    xg_d, yg_d, D_drive = build_converging_drive(
        ELL_FIXED, FOCAL_FIXED, R_FIXED, CX, CY,
        asm_domain, ASM_N, K_WATER, apod=APOD_FIXED)
    dx_d = float(xg_d[1] - xg_d[0])

    # ── Calibration grids (moderate — used for z0 sweep) ─────────
    cal_half  = 3.0e-3
    cal_n_xy  = 200
    xg_cal = np.linspace(CX - cal_half, CX + cal_half, cal_n_xy)
    yg_cal = np.linspace(CY - cal_half, CY + cal_half, cal_n_xy)
    YYcal, XXcal = np.meshgrid(yg_cal, xg_cal, indexing="ij")

    zmin_cal, zmax_cal = 1.0e-3, 6.0e-3
    nz_cal = 50
    z_cal  = np.linspace(zmin_cal, zmax_cal, nz_cal)

    # ==============================================================
    # STEP 2 — z0_offset calibration sweep
    # ==============================================================
    log("=" * 72)
    log("STEP 2: z0_offset calibration sweep")
    log("=" * 72)
    log(f"  Sweep: {len(Z0_OFFSETS)} offsets in "
        f"[{Z0_OFFSETS[0]*1e3:.2f}, {Z0_OFFSETS[-1]*1e3:.2f}] mm")
    log()

    t0_cal = time.time()
    z_w_per_offset = np.zeros(len(Z0_OFFSETS))
    z_err_per_offset = np.zeros(len(Z0_OFFSETS))

    for io, z0off in enumerate(Z0_OFFSETS):
        ring_I_z = np.zeros(nz_cal)
        for iz, z_dish in enumerate(z_cal):
            z_prop = z_dish - z0off
            if z_prop <= 0:
                continue
            p_prop = propagate_pressure_asm(
                D_drive, dx_d, dx_d, K_WATER, z_prop, pad_factor=ASM_PAD)
            p_out = interp_asm_to_grid(p_prop, xg_d, yg_d, XXcal, YYcal)
            I_out = np.abs(p_out)**2
            _, Ip = find_ring(I_out, xg_cal, yg_cal, CX, CY,
                              r_max=cal_half * 0.9, r_min=0.01e-3)
            ring_I_z[iz] = Ip

        iz_w = np.argmax(ring_I_z)
        z_w_per_offset[io] = z_cal[iz_w]
        z_err_per_offset[io] = abs(z_cal[iz_w] - Z_STAR)

        if io % 10 == 0 or io == len(Z0_OFFSETS) - 1:
            log(f"    [{io+1:3d}/{len(Z0_OFFSETS)}] "
                f"z0={z0off*1e3:+6.3f}mm  z_w={z_cal[iz_w]*1e3:.3f}mm  "
                f"err={z_err_per_offset[io]*1e3:.3f}mm")

    ibest = np.argmin(z_err_per_offset)
    Z0_BEST = float(Z0_OFFSETS[ibest])
    z_w_best = z_w_per_offset[ibest]
    dt_cal = time.time() - t0_cal

    log(f"\n  Calibration: {dt_cal:.1f}s for {len(Z0_OFFSETS)} offsets")
    log(f"  ★ BEST z0_offset = {Z0_BEST*1e3:+.4f} mm")
    log(f"    z_w(dish) = {z_w_best*1e3:.3f} mm   "
        f"(z* = {Z_STAR*1e3:.3f}, err = {z_err_per_offset[ibest]*1e3:.3f} mm)")
    log()

    # ── Figure: z0_offset calibration ─────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(Z0_OFFSETS * 1e3, z_err_per_offset * 1e3, "b.-", lw=1.5, ms=5)
    ax.axvline(Z0_BEST * 1e3, color="red", ls="--", lw=1.5,
               label=f"best: {Z0_BEST*1e3:+.3f} mm")
    ax.set_xlabel("z0_offset (mm)")
    ax.set_ylabel("|z_w − z*| (mm)")
    ax.set_title("z0-offset calibration sweep\n"
                 f"ℓ={ELL_FIXED}  f={FOCAL_FIXED*1e3:.1f}mm  "
                 f"R={R_FIXED*1e3:.1f}mm  {APOD_FIXED}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "z0_offset_calibration.png", dpi=160)
    plt.close(fig)
    log("  Saved z0_offset_calibration.png")
    log()

    # ==============================================================
    # STEP 3 — Hourglass verification gate
    # ==============================================================
    log("=" * 72)
    log("STEP 3: Hourglass verification gate")
    log("=" * 72)

    nz_fine = 120
    z_fine  = np.linspace(zmin_cal, zmax_cal, nz_fine)

    # Use the small ROI grid for ring finding (centred on beam axis)
    ver_half = ROI_HALF * 2.5   # slightly wider for ring finding
    ver_n    = 250
    xg_ver = np.linspace(CX - ver_half, CX + ver_half, ver_n)
    yg_ver = np.linspace(CY - ver_half, CY + ver_half, ver_n)
    YYver, XXver = np.meshgrid(yg_ver, xg_ver, indexing="ij")

    ring_r_fine = np.zeros(nz_fine)
    ring_I_fine = np.zeros(nz_fine)

    log(f"  Fine scan: {nz_fine} z-planes, {ver_n}² grid, "
        f"z0_offset = {Z0_BEST*1e3:+.3f} mm")
    t0 = time.time()
    for iz, z_dish in enumerate(z_fine):
        z_prop = z_dish - Z0_BEST
        if z_prop <= 0:
            continue
        p_prop = propagate_pressure_asm(D_drive, dx_d, dx_d, K_WATER, z_prop,
                                        pad_factor=ASM_PAD)
        p_out = interp_asm_to_grid(p_prop, xg_d, yg_d, XXver, YYver)
        I_out = np.abs(p_out)**2
        rr, Ip = find_ring(I_out, xg_ver, yg_ver, CX, CY,
                           r_max=ver_half * 0.9, r_min=0.01e-3)
        ring_r_fine[iz] = rr
        ring_I_fine[iz] = Ip
    log(f"  Done in {time.time()-t0:.1f}s")

    iz_w = np.argmax(ring_I_fine)
    z_w  = z_fine[iz_w]
    r_w  = ring_r_fine[iz_w]
    D_w  = 2 * r_w

    # Also find index of minimum radius (informational, not a hard gate)
    valid = ring_r_fine > 0
    if valid.sum() > 5:
        iz_rmin = np.argmin(np.where(valid, ring_r_fine, 1e10))
    else:
        iz_rmin = iz_w

    log(f"  Waist: z_w = {z_w*1e3:.3f} mm  "
        f"D_w = {D_w*1e3:.4f} mm  D_w/(λ/2) = {D_w/TRAP_SP:.3f}")
    log(f"  Ring-radius min at z = {z_fine[iz_rmin]*1e3:.3f} mm  "
        f"(Intensity peak at z = {z_w*1e3:.3f} mm)")

    # ── Hourglass criteria ────────────────────────────────────────
    # (1) Intensity peak not at scan boundary
    hg_edge_ok = (iz_w > 2) and (iz_w < nz_fine - 3)
    # (2) Ring radius expands going away from waist in both directions
    hg_expand  = (ring_r_fine[0] > r_w * 0.95 if valid[0] else True) and \
                 (ring_r_fine[-1] > r_w * 0.95 if valid[-1] else True)
    # (3) Local contraction: radius in a ±10-step window around iz_w is
    #     smaller on average than in the outer quarters of the scan
    w = min(10, nz_fine // 6)
    inner_lo = max(0, iz_w - w)
    inner_hi = min(nz_fine, iz_w + w + 1)
    outer_lo = ring_r_fine[:nz_fine // 4]
    outer_hi = ring_r_fine[3 * nz_fine // 4:]
    r_inner = ring_r_fine[inner_lo:inner_hi]
    r_inner_valid = r_inner[r_inner > 0]
    r_outer_valid = np.concatenate([outer_lo[outer_lo > 0],
                                    outer_hi[outer_hi > 0]])
    if len(r_inner_valid) > 0 and len(r_outer_valid) > 0:
        hg_contract = float(np.mean(r_inner_valid)) < float(np.mean(r_outer_valid))
    else:
        hg_contract = False

    hourglass_ok = hg_edge_ok and hg_expand and hg_contract

    coloc_dz = abs(z_fine[iz_w] - z_fine[iz_rmin]) * 1e3
    log(f"  Hourglass checks:")
    log(f"    waist not at edge:    {hg_edge_ok}")
    log(f"    radius expands:       {hg_expand}")
    log(f"    local contraction:    {hg_contract}")
    log(f"    (info) I/r coloc Δz:  {coloc_dz:.3f} mm")
    log(f"  → HOURGLASS {'PASS ✓' if hourglass_ok else 'FAIL ✗'}")

    # Bridge-scale check
    bridge_lo = 0.3 * TRAP_SP
    bridge_hi = 0.7 * TRAP_SP
    bridge_met = bridge_lo <= D_w <= bridge_hi
    log(f"  Bridge-scale target: D_w ∈ [{bridge_lo*1e3:.3f}, "
        f"{bridge_hi*1e3:.3f}] mm  →  "
        f"{'MET' if bridge_met else f'NOT MET (D_w/{TRAP_SP*1e3:.3f} = {D_w/TRAP_SP:.2f})'}")
    log()

    # ── Figure: ring radius + intensity vs z ──────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    z_mm = z_fine * 1e3

    ax1.plot(z_mm, ring_I_fine / (ring_I_fine.max() + 1e-30), "b-", lw=2)
    ax1.axvline(z_w * 1e3, color="red", ls="--", lw=1.5,
                label=f"z_w = {z_w*1e3:.2f} mm")
    ax1.axvline(Z_STAR * 1e3, color="lime", ls=":", lw=1.5,
                label=f"z* = {Z_STAR*1e3:.2f} mm")
    ax1.set_ylabel("Normalised ring-peak intensity")
    ax1.set_title(f"Hourglass verification  "
                  f"ℓ={ELL_FIXED}  f={FOCAL_FIXED*1e3:.1f}mm  "
                  f"R={R_FIXED*1e3:.1f}mm  {APOD_FIXED}  "
                  f"z0_off={Z0_BEST*1e3:+.3f}mm")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.plot(z_mm, ring_r_fine * 1e3, "b-", lw=2)
    ax2.axhline(r_w * 1e3, color="red", ls=":", lw=1, alpha=0.5,
                label=f"r_w = {r_w*1e3:.3f} mm")
    ax2.axvline(z_w * 1e3, color="red", ls="--", lw=1.5)
    ax2.axvline(Z_STAR * 1e3, color="lime", ls=":", lw=1.5)
    ax2.set_xlabel("z_dish (mm)")
    ax2.set_ylabel("Ring radius (mm)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG / "ring_radius_and_intensity_vs_z.png", dpi=160)
    plt.close(fig)
    log("  Saved ring_radius_and_intensity_vs_z.png")

    # ── Symmetry check ────────────────────────────────────────────
    dz_sym = min(0.5e-3, (z_w - zmin_cal) * 0.3, (zmax_cal - z_w) * 0.3)
    sym_planes = {"lo": z_w - dz_sym, "waist": z_w, "hi": z_w + dz_sym}
    sym_half = ROI_HALF * 2
    sym_n = 200
    xg_sym = np.linspace(CX - sym_half, CX + sym_half, sym_n)
    yg_sym = np.linspace(CY - sym_half, CY + sym_half, sym_n)
    YYsym, XXsym = np.meshgrid(yg_sym, xg_sym, indexing="ij")

    I_sym, r_sym = {}, {}
    for tag, z_dish in sym_planes.items():
        z_prop = z_dish - Z0_BEST
        if z_prop <= 0:
            I_sym[tag] = np.zeros((sym_n, sym_n))
            r_sym[tag] = 0.0
            continue
        p_prop = propagate_pressure_asm(D_drive, dx_d, dx_d, K_WATER, z_prop,
                                        pad_factor=ASM_PAD)
        p_out = interp_asm_to_grid(p_prop, xg_d, yg_d, XXsym, YYsym)
        I_sym[tag] = np.abs(p_out)**2
        rr, _ = find_ring(I_sym[tag], xg_sym, yg_sym, CX, CY,
                          r_max=sym_half * 0.9)
        r_sym[tag] = rr

    ext_sym = [(xg_sym[0]-CX)*1e3, (xg_sym[-1]-CX)*1e3,
               (yg_sym[0]-CY)*1e3, (yg_sym[-1]-CY)*1e3]
    vmax_sym = max(v.max() for v in I_sym.values()) or 1.0

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, (tag, z_dish) in zip(axes, sym_planes.items()):
        im = ax.imshow(I_sym[tag], extent=ext_sym, origin="lower",
                       cmap="inferno", vmin=0, vmax=vmax_sym, aspect="equal")
        c = plt.Circle((0, 0), r_sym[tag]*1e3, fill=False,
                        ec="cyan", lw=1.5, ls="--", alpha=0.8)
        ax.add_patch(c)
        ax.set_title(f"z_dish={z_dish*1e3:.2f} mm ({tag})\n"
                     f"r_ring={r_sym[tag]*1e3:.3f} mm", fontsize=10)
        ax.set_xlabel("x − CX (mm)"); ax.set_ylabel("y − CY (mm)")
        plt.colorbar(im, ax=ax, shrink=0.8, label="|p|²")

    sym_ratio = r_sym["lo"] / (r_sym["hi"] + 1e-30)
    sym_ok = 0.65 < sym_ratio < 1.55
    fig.suptitle(f"Symmetry check  Δz=±{dz_sym*1e3:.2f}mm  "
                 f"ratio={sym_ratio:.3f}  "
                 f"{'OK' if sym_ok else 'ASYM'}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "symmetry_check.png", dpi=160)
    plt.close(fig)
    log(f"  Symmetry: r_lo/r_hi = {sym_ratio:.3f}  "
        f"{'OK' if sym_ok else 'ASYMMETRIC'}")
    log()

    # ── Gate decision ─────────────────────────────────────────────
    gate_pass = hourglass_ok
    if not gate_pass:
        log("  *** HOURGLASS GATE FAILED — skipping α sweeps ***")
        _write_report_failure(OUT, log, Z0_BEST, z_w, D_w, r_w,
                              sym_ratio, sym_ok, hourglass_ok, cache_path,
                              time.time() - t_GLOBAL)
        log_fp.close()
        return

    log("  HOURGLASS GATE PASSED → proceeding to α sweeps")
    log()

    # ==============================================================
    # STEP 4 — FEM + ASM overlay on small ROI
    # ==============================================================
    log("=" * 72)
    log("STEP 4: FEM + ASM overlay on small ROI at z*")
    log("=" * 72)

    # ── Load FEM cache ────────────────────────────────────────────
    t0 = time.time()
    coords, p_fem, meta = load_fem_cache(cache_path)
    tree = cKDTree(coords)
    K_IDW = 16
    log(f"  FEM: {coords.shape[0]:,} DOFs ({time.time()-t0:.2f}s)")

    # ── Small ROI grid ────────────────────────────────────────────
    xg = np.linspace(CX - ROI_HALF, CX + ROI_HALF, NGRID)
    yg = np.linspace(CY - ROI_HALF, CY + ROI_HALF, NGRID)
    dx = float(xg[1] - xg[0])
    dy = float(yg[1] - yg[0])
    XX, YY = np.meshgrid(xg, yg)
    ext_roi = [(CX - ROI_HALF) * 1e3, (CX + ROI_HALF) * 1e3,
               (CY - ROI_HALF) * 1e3, (CY + ROI_HALF) * 1e3]

    log(f"  ROI: {2*ROI_HALF*1e3:.3f}mm × {2*ROI_HALF*1e3:.3f}mm  "
        f"({2*ROI_HALF/LAM:.1f}λ)  grid {NGRID}×{NGRID}")
    log(f"  dx = {dx*1e6:.1f} µm  ({LAM/dx:.0f} pts/λ)")
    log()

    # ── Helper: FEM IDW on small ROI ──────────────────────────────
    def _fem_xy(z_val):
        pts = np.column_stack([XX.ravel(), YY.ravel(),
                               np.full(XX.size, z_val)])
        return sample_idw(tree, p_fem, pts, k=K_IDW).reshape(XX.shape)

    # ── Helper: ASM vortex on small ROI at dish z ─────────────────
    def _asm_xy(z_dish):
        z_prop = z_dish - Z0_BEST
        pf = propagate_pressure_asm(D_drive, dx_d, dx_d, K_WATER, z_prop,
                                    pad_factor=ASM_PAD, include_evanescent=True)
        return interp_asm_to_grid(pf, xg_d, yg_d, XX, YY)

    # ── Sample FEM: 3 z-planes for gradient ───────────────────────
    log("  FEM sampling (z*, z*±dz)...")
    t0 = time.time()
    p_s    = _fem_xy(Z_STAR)
    p_s_zp = _fem_xy(Z_STAR + DZ_GRAD)
    p_s_zm = _fem_xy(Z_STAR - DZ_GRAD)
    log(f"    3 planes: {time.time()-t0:.2f}s  max|p_s| = {np.abs(p_s).max():.3f} Pa")

    # ── Sample ASM vortex: 3 z-planes ─────────────────────────────
    log("  ASM vortex sampling (z*, z*±dz)...")
    t0 = time.time()
    p_v    = _asm_xy(Z_STAR)
    p_v_zp = _asm_xy(Z_STAR + DZ_GRAD)
    p_v_zm = _asm_xy(Z_STAR - DZ_GRAD)
    log(f"    3 planes: {time.time()-t0:.2f}s  max|p_v| = {np.abs(p_v).max():.6f}")
    log()

    # ── Normalise vortex to standing peak ─────────────────────────
    stand_peak = float(np.abs(p_s).max())
    vort_peak  = float(np.abs(p_v).max()) + 1e-30
    nfac = stand_peak / vort_peak
    log(f"  Standing peak: {stand_peak:.3f} Pa")
    log(f"  Vortex peak @ z*: {vort_peak:.6f}  → nfac = {nfac:.3f}")
    log()

    pv_n    = p_v    * nfac
    pv_n_zp = p_v_zp * nfac
    pv_n_zm = p_v_zm * nfac

    # ── Standing Gor'kov ──────────────────────────────────────────
    log("  Computing Gor'kov potentials...")
    gp2_s = grad_p_mag2_xy(p_s, p_s_zp, p_s_zm, dx, dy, DZ_GRAD)
    U_s   = gorkov_potential(np.abs(p_s)**2, gp2_s)

    trap_depth = float(U_s.max() - U_s.min())
    log(f"  Standing Gorkov: min={U_s.min():.3e}  max={U_s.max():.3e}  "
        f"depth={trap_depth:.3e} J")

    # ── Combined for each α ───────────────────────────────────────
    U_comb = {}
    DU     = {}
    perturb = {}
    for a in ALPHAS:
        p_t    = p_s    + a * pv_n
        p_t_zp = p_s_zp + a * pv_n_zp
        p_t_zm = p_s_zm + a * pv_n_zm
        gp2 = grad_p_mag2_xy(p_t, p_t_zp, p_t_zm, dx, dy, DZ_GRAD)
        U_comb[a] = gorkov_potential(np.abs(p_t)**2, gp2)
        DU[a] = U_comb[a] - U_s
        mx = float(np.abs(DU[a]).max() / (trap_depth + 1e-30))
        perturb[a] = mx
        log(f"    α={a:.2f}: max|ΔU|/depth = {mx*100:.2f}%")
    log()

    # ==============================================================
    # FIGURES
    # ==============================================================
    log("=" * 72)
    log("FIGURES")
    log("=" * 72)

    kw = dict(extent=ext_roi, origin="lower", aspect="equal")

    def _trap_grid(ax):
        for off in np.arange(-4, 5) * TRAP_SP:
            ax.axhline((CY + off) * 1e3, color="white", lw=0.3, alpha=0.25)
            ax.axvline((CX + off) * 1e3, color="white", lw=0.3, alpha=0.25)

    # ── Figure A: Pressure XY sweep ──────────────────────────────
    ncol = len(ALPHAS) + 1  # standing + 4 α values
    nrow = 4
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.3 * nrow))

    abs_s = np.abs(p_s)
    vmax_p = abs_s.max()

    row_labels = ["|p_stand|", "|α·p̂_vort|", "|p_total|", "Rel. perturbation"]

    for col in range(ncol):
        if col == 0:
            # Standing column
            a_val = None
            p_total = p_s
            p_vort_scaled = np.zeros_like(p_s)
        else:
            a_val = ALPHAS[col - 1]
            p_vort_scaled = a_val * pv_n
            p_total = p_s + p_vort_scaled

        abs_total = np.abs(p_total)
        abs_vort  = np.abs(p_vort_scaled)
        rel_pert  = (abs_total - abs_s) / (abs_s + 1e-30)

        # Row 0: |p_stand| (same for all columns)
        ax = axes[0, col]
        im = ax.imshow(abs_s, cmap="inferno", vmin=0, vmax=vmax_p, **kw)
        _trap_grid(ax)
        if col == 0:
            ax.set_title("FEM Standing", fontsize=10, fontweight="bold")
        else:
            ax.set_title(f"α = {a_val}", fontsize=10, fontweight="bold")
        if col == 0:
            ax.set_ylabel(row_labels[0], fontsize=9, fontweight="bold")
        plt.colorbar(im, ax=ax, shrink=0.7, label="Pa")

        # Row 1: |α·p_v|
        ax = axes[1, col]
        if col == 0:
            im = ax.imshow(np.zeros_like(abs_s), cmap="viridis",
                           vmin=0, vmax=1, **kw)
            ax.set_title("(vortex ref)", fontsize=9)
        else:
            vm_v = max(abs_vort.max(), 1e-30)
            im = ax.imshow(abs_vort, cmap="viridis", vmin=0, vmax=vm_v, **kw)
        _trap_grid(ax)
        if col == 0:
            ax.set_ylabel(row_labels[1], fontsize=9, fontweight="bold")
        plt.colorbar(im, ax=ax, shrink=0.7, label="Pa")

        # Row 2: |p_total|
        ax = axes[2, col]
        im = ax.imshow(abs_total, cmap="inferno", vmin=0, vmax=vmax_p, **kw)
        _trap_grid(ax)
        if col == 0:
            ax.set_ylabel(row_labels[2], fontsize=9, fontweight="bold")
        plt.colorbar(im, ax=ax, shrink=0.7, label="Pa")

        # Row 3: relative perturbation
        ax = axes[3, col]
        if col == 0:
            im = ax.imshow(np.zeros_like(abs_s), cmap="RdBu_r",
                           vmin=-0.5, vmax=0.5, **kw)
        else:
            pert_max = max(np.abs(rel_pert).max(), 0.01)
            im = ax.imshow(rel_pert, cmap="RdBu_r",
                           vmin=-pert_max, vmax=pert_max, **kw)
        _trap_grid(ax)
        if col == 0:
            ax.set_ylabel(row_labels[3], fontsize=9, fontweight="bold")
        plt.colorbar(im, ax=ax, shrink=0.7, label="rel.")

    for ax in axes.flat:
        ax.set_xlabel("x (mm)", fontsize=7)
        ax.tick_params(labelsize=6)
    for ax in axes[:, 0]:
        ax.set_ylabel(ax.get_ylabel(), fontsize=9)

    fig.suptitle(
        f"Pressure XY sweep at z* = {Z_STAR*1e3:.3f} mm  │  "
        f"3×3 trap ROI ({2*ROI_HALF*1e3:.2f} mm)  │  "
        f"z0_off = {Z0_BEST*1e3:+.3f} mm",
        fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIG / "pressure_xy_sweep_zstar.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    log("  Saved pressure_xy_sweep_zstar.png")

    # ── Figure B: Gor'kov XY sweep ────────────────────────────────
    ncol_g = len(ALPHAS) + 1
    fig, axes = plt.subplots(2, ncol_g, figsize=(3.8 * ncol_g, 7.5))

    vmin_U = min(U_s.min(), min(U_comb[a].min() for a in ALPHAS))
    vmax_U = max(U_s.max(), max(U_comb[a].max() for a in ALPHAS))
    du_max = max(np.abs(DU[a]).max() for a in ALPHAS)
    du_lim = max(du_max, 1e-30)

    # Row 0, col 0: U standing
    ax = axes[0, 0]
    im = ax.imshow(U_s, cmap="RdBu_r", vmin=vmin_U, vmax=vmax_U, **kw)
    ax.set_title("FEM Standing\nU_Gorkov", fontsize=10, fontweight="bold",
                 color="navy")
    _trap_grid(ax)
    plt.colorbar(im, ax=ax, shrink=0.7, label="J")

    # Row 1, col 0: blank (vortex-only Gorkov for reference)
    ax = axes[1, 0]
    gp2_v = grad_p_mag2_xy(pv_n, pv_n_zp, pv_n_zm, dx, dy, DZ_GRAD)
    U_v = gorkov_potential(np.abs(pv_n)**2, gp2_v)
    im = ax.imshow(U_v, cmap="viridis", **kw)
    ax.set_title("Vortex-only\nU_Gorkov (ref)", fontsize=10,
                 fontweight="bold", color="darkgreen")
    _trap_grid(ax)
    plt.colorbar(im, ax=ax, shrink=0.7, label="J")

    for j, a in enumerate(ALPHAS):
        col = j + 1

        # Row 0: U combined
        ax = axes[0, col]
        im = ax.imshow(U_comb[a], cmap="RdBu_r",
                       vmin=vmin_U, vmax=vmax_U, **kw)
        ax.set_title(f"COMBINED α={a}\nU(p_s+α·p_v)", fontsize=10,
                     fontweight="bold", color="darkred")
        _trap_grid(ax)
        plt.colorbar(im, ax=ax, shrink=0.7, label="J")

        # Row 1: ΔU (shared diverging colorbar)
        ax = axes[1, col]
        im = ax.imshow(DU[a], cmap="RdBu_r",
                       vmin=-du_lim, vmax=du_lim, **kw)
        ax.set_title(f"ΔU  α={a}\n"
                     f"max|ΔU|/depth = {perturb[a]*100:.1f}%",
                     fontsize=9, fontweight="bold", color="purple")
        _trap_grid(ax)
        plt.colorbar(im, ax=ax, shrink=0.7, label="J")

    for ax in axes.flat:
        ax.set_xlabel("x (mm)", fontsize=7)
        ax.tick_params(labelsize=6)

    fig.suptitle(
        f"Gor'kov XY sweep at z* = {Z_STAR*1e3:.3f} mm  │  "
        f"3×3 trap ROI  │  depth = {trap_depth:.2e} J  │  "
        f"z0_off = {Z0_BEST*1e3:+.3f} mm",
        fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIG / "gorkov_xy_sweep_zstar.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    log("  Saved gorkov_xy_sweep_zstar.png")
    log()

    # ==============================================================
    # STEP 5 — Stats + trap count
    # ==============================================================
    log("=" * 72)
    log("STEP 5: Stats & REPORT")
    log("=" * 72)

    trap_count_info = {}
    for a in ALPHAS:
        du_rel = np.abs(DU[a]) / (trap_depth + 1e-30)
        U_neg = -U_s
        lm = ((U_neg == maximum_filter(U_neg, size=15)) &
              (U_neg > 0.3 * U_neg.max()))
        n_peaks = int(lm.sum())
        n_affected = int((lm & (du_rel > 0.10)).sum())
        trap_count_info[str(a)] = {
            "n_traps": n_peaks, "n_affected_10pct": n_affected}
        log(f"  α={a:.2f}: {n_peaks} traps in ROI, "
            f"{n_affected} with >10% depth perturbation")

    J11_PRIME = 1.841
    D_diff = 2 * J11_PRIME * ELL_FIXED / (K_WATER * NA_FIXED)
    D_diff_ratio = D_diff / TRAP_SP

    stats = {
        "lambda_mm": float(LAM * 1e3),
        "trap_spacing_mm": float(TRAP_SP * 1e3),
        "z_star_mm": round(float(Z_STAR * 1e3), 4),
        "roi_half_mm": round(float(ROI_HALF * 1e3), 4),
        "roi_width_lambda": round(float(2 * ROI_HALF / LAM), 2),
        "grid_size": NGRID,
        "gorkov_formula": "U = V_p*(f1*E_pot - 1.5*f2*E_kin)",
        "particle": {
            "material": "polystyrene",
            "rho_p": RHO_P, "c_p": C_P,
            "a_um": A_P * 1e6,
            "f1": round(F1, 5), "f2": round(F2, 5),
        },
        "lens": {
            "ell": ELL_FIXED, "R_mm": R_FIXED * 1e3,
            "f_mm": FOCAL_FIXED * 1e3, "apodization": APOD_FIXED,
            "NA": round(float(NA_FIXED), 4),
            "sign_convention": "phi = ell*theta - k*(sqrt(r^2+f^2) - f) [CONVERGING]",
        },
        "z0_offset_mm": round(float(Z0_BEST * 1e3), 4),
        "waist": {
            "z_w_mm": round(float(z_w * 1e3), 4),
            "z_err_mm": round(float(abs(z_w - Z_STAR) * 1e3), 4),
            "D_w_mm": round(float(D_w * 1e3), 4),
            "D_w_over_half_lam": round(float(D_w / TRAP_SP), 4),
            "bridge_scale_met": bool(bridge_met),
            "diffraction_limit_D_mm": round(float(D_diff * 1e3), 4),
            "diffraction_limit_ratio": round(float(D_diff_ratio), 4),
        },
        "hourglass": {
            "pass": bool(hourglass_ok),
            "edge_ok": bool(hg_edge_ok),
            "expands": bool(hg_expand),
            "local_contraction": bool(hg_contract),
            "coloc_dz_mm": round(coloc_dz, 4),
        },
        "symmetry": {
            "r_lo_mm": round(float(r_sym["lo"] * 1e3), 4),
            "r_hi_mm": round(float(r_sym["hi"] * 1e3), 4),
            "ratio": round(float(sym_ratio), 4),
            "ok": bool(sym_ok),
        },
        "gorkov_standing": {
            "min_J": float(U_s.min()),
            "max_J": float(U_s.max()),
            "trap_depth_J": float(trap_depth),
        },
        "perturbation": {
            str(a): {"max_rel_depth": perturb[a]} for a in ALPHAS
        },
        "trap_count": trap_count_info,
        "nfac_vortex": round(float(nfac), 4),
        "dz_grad_mm": round(float(DZ_GRAD * 1e3), 4),
        "idw_k": K_IDW,
    }

    with open(DAT / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    log("  Saved stats.json")

    np.savez_compressed(DAT / "fields.npz",
                        xg=xg, yg=yg,
                        U_s=U_s, p_s=p_s, pv_n=pv_n,
                        z_w=z_w, z_star=Z_STAR, z0_offset=Z0_BEST)
    log("  Saved fields.npz")

    # ── REPORT.md ─────────────────────────────────────────────────
    elapsed = time.time() - t_GLOBAL
    log(f"\nTotal runtime: {elapsed:.1f}s")

    rp = _build_report(
        elapsed, cache_path, coords, Z0_BEST, z_w, D_w, r_w,
        sym_ratio, sym_ok, hourglass_ok, bridge_met,
        NA_FIXED, D_diff, D_diff_ratio, trap_depth,
        perturb, trap_count_info, ALPHAS)
    (OUT / "REPORT.md").write_text("\n".join(rp))
    log("  Saved REPORT.md")
    log()
    log("=" * 72)
    log("DONE")
    log("=" * 72)
    log_fp.close()


# ═══════════════════════════════════════════════════════════════════
# Report builders
# ═══════════════════════════════════════════════════════════════════

def _build_report(elapsed, cache_path, coords, z0_best, z_w, D_w, r_w,
                  sym_ratio, sym_ok, hourglass_ok, bridge_met,
                  NA_fixed, D_diff, D_diff_ratio, trap_depth,
                  perturb, trap_count_info, alphas):
    rp = []
    rp.append("# Small-ROI Waist-Align + XY α Sweeps")
    rp.append(f"\n**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    rp.append(f"**Runtime**: {elapsed:.1f}s")
    rp.append(f"**Cache**: `{cache_path.name}` ({coords.shape[0]:,} DOFs)")
    rp.append("")

    rp.append("## 0. Configuration")
    rp.append("")
    rp.append(f"- **Small ROI**: ±{ROI_HALF*1e3:.3f} mm "
              f"({2*ROI_HALF/LAM:.1f}λ), {NGRID}×{NGRID} grid")
    rp.append(f"- **Lens** (fixed): ℓ={ELL_FIXED}, "
              f"f={FOCAL_FIXED*1e3:.1f} mm, R={R_FIXED*1e3:.1f} mm, "
              f"apod={APOD_FIXED}, NA={NA_fixed:.4f}")
    rp.append(f"- **z*** = {Z_STAR*1e3:.4f} mm")
    rp.append(f"- **λ** = {LAM*1e3:.4f} mm, trap spacing λ/2 = "
              f"{TRAP_SP*1e3:.4f} mm")
    rp.append("")

    rp.append("## 1. z0-offset calibration")
    rp.append("")
    rp.append(f"Swept z0_offset ∈ [{Z0_OFFSETS[0]*1e3:.1f}, "
              f"{Z0_OFFSETS[-1]*1e3:.1f}] mm ({len(Z0_OFFSETS)} values).")
    rp.append(f"The offset shifts the ASM propagation reference: "
              f"`p_vort(x,y,z*) ← p_asm(x,y, z* − z0_offset)`.")
    rp.append("")
    rp.append(f"**Chosen z0_offset = {z0_best*1e3:+.4f} mm**")
    rp.append(f"- z_w(dish) = {z_w*1e3:.3f} mm  "
              f"(z* = {Z_STAR*1e3:.3f} mm, err = "
              f"{abs(z_w - Z_STAR)*1e3:.3f} mm)")
    rp.append("")
    rp.append("![z0_offset_calibration](figures/z0_offset_calibration.png)")
    rp.append("")

    rp.append("## 2. Hourglass verification")
    rp.append("")
    rp.append(f"- **Hourglass**: {'PASS ✓' if hourglass_ok else 'FAIL ✗'}")
    rp.append(f"- **Waist**: z_w = {z_w*1e3:.3f} mm, "
              f"D_w = {D_w*1e3:.4f} mm = "
              f"{D_w/TRAP_SP:.3f} × (λ/2)")
    rp.append(f"- **Diffraction limit**: D_min = {D_diff*1e3:.4f} mm = "
              f"{D_diff_ratio:.3f} × (λ/2)")
    rp.append(f"- **Bridge-scale target** [0.3, 0.7]×(λ/2): "
              f"{'MET ✓' if bridge_met else 'NOT MET'} "
              f"(D_w/(λ/2) = {D_w/TRAP_SP:.3f})")
    rp.append(f"- **Symmetry**: r_lo/r_hi = {sym_ratio:.3f} "
              f"({'OK' if sym_ok else 'asymmetric'})")
    rp.append("")
    rp.append("![ring_radius_and_intensity_vs_z]"
              "(figures/ring_radius_and_intensity_vs_z.png)")
    rp.append("")
    rp.append("![symmetry_check](figures/symmetry_check.png)")
    rp.append("")

    if not hourglass_ok:
        rp.append("**Hourglass gate FAILED — α sweeps not produced.**")
        return rp

    rp.append("## 3. Pressure XY sweep at z*")
    rp.append("")
    rp.append("![pressure_xy_sweep_zstar]"
              "(figures/pressure_xy_sweep_zstar.png)")
    rp.append("")
    rp.append("Rows: |p_stand|, |α·p̂_vort|, |p_total|, relative perturbation. "
              "Columns: standing (reference), then α = 0.02, 0.05, 0.10, 0.20.")
    rp.append("")

    rp.append("## 4. Gor'kov XY sweep at z*")
    rp.append("")
    rp.append(f"Particle: polystyrene a={A_P*1e6:.0f} µm, "
              f"f₁={F1:.4f}, f₂={F2:.4f}")
    rp.append(f"Gradient step dz = {DZ_GRAD*1e3:.4f} mm (λ/{LAM/DZ_GRAD:.0f})")
    rp.append(f"Standing trap depth = {trap_depth:.3e} J")
    rp.append("")
    rp.append("| α | max |ΔU|/depth | traps >10% | traps in ROI |")
    rp.append("|---|---|---|---|")
    for a in alphas:
        n_aff = trap_count_info[str(a)]["n_affected_10pct"]
        n_tot = trap_count_info[str(a)]["n_traps"]
        rp.append(f"| {a:.2f} | {perturb[a]*100:.1f}% | {n_aff} | {n_tot} |")
    rp.append("")
    rp.append("![gorkov_xy_sweep_zstar](figures/gorkov_xy_sweep_zstar.png)")
    rp.append("")
    rp.append("Top row: Gor'kov potential U (standing and combined). "
              "Bottom row: vortex-only U (reference), then ΔU = U_combined − "
              "U_standing on a **shared diverging colorbar**.")
    rp.append("")

    rp.append("## 5. Interpretation")
    rp.append("")
    a_ref = 0.1
    rp.append(f"At α={a_ref}, maximum Gor'kov perturbation is "
              f"{perturb[a_ref]*100:.1f}% of trap depth. ")
    n_aff = trap_count_info[str(a_ref)]["n_affected_10pct"]
    n_tot = trap_count_info[str(a_ref)]["n_traps"]
    rp.append(f"{n_aff} of {n_tot} trap sites in the 3×3 ROI exceed "
              f"10% depth perturbation.")
    rp.append("")
    if D_w / TRAP_SP < 1.0:
        rp.append("The vortex waist is sub-trap-spacing, suggesting it can "
                   "create a localised perturbation (potential bridge) between "
                   "adjacent traps without significantly disturbing the wider "
                   "trap lattice.")
    else:
        rp.append(f"The vortex waist diameter ({D_w/TRAP_SP:.2f} × λ/2) "
                  f"exceeds one trap spacing. The perturbation extends "
                  f"over multiple traps. Smaller NA or higher ℓ would be "
                  f"needed to approach bridge-scale.")
    rp.append("")

    return rp


def _write_report_failure(out, log, z0_best, z_w, D_w, r_w,
                          sym_ratio, sym_ok, hourglass_ok, cache_path,
                          elapsed):
    """Write a failure REPORT.md when hourglass gate doesn't pass."""
    rp = []
    rp.append("# Small-ROI Waist-Align + XY α Sweeps — GATE FAILED")
    rp.append(f"\n**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    rp.append(f"**Runtime**: {elapsed:.1f}s")
    rp.append(f"**Cache**: `{cache_path.name}`")
    rp.append("")
    rp.append("## Failure reason")
    rp.append("")
    rp.append("The hourglass verification gate did not pass. "
              "α sweeps were NOT produced.")
    rp.append("")
    rp.append(f"- z0_offset = {z0_best*1e3:+.4f} mm")
    rp.append(f"- z_w = {z_w*1e3:.3f} mm  "
              f"(z* = {Z_STAR*1e3:.3f} mm)")
    rp.append(f"- D_w = {D_w*1e3:.4f} mm = "
              f"{D_w/TRAP_SP:.3f} × (λ/2)")
    rp.append(f"- Symmetry ratio = {sym_ratio:.3f} "
              f"({'OK' if sym_ok else 'asymmetric'})")
    rp.append(f"- Hourglass pass = {hourglass_ok}")
    rp.append("")
    rp.append("![ring_radius_and_intensity_vs_z]"
              "(figures/ring_radius_and_intensity_vs_z.png)")
    rp.append("")
    rp.append("![symmetry_check](figures/symmetry_check.png)")
    (out / "REPORT.md").write_text("\n".join(rp))
    log("  Saved REPORT.md (failure)")


if __name__ == "__main__":
    main()
