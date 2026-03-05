#!/usr/bin/env python3
"""
Gor'kov-based Waist-Fix + Vortex Overlay + α Sweep
====================================================

Steps:
  1. Calibration sweep with focal targeted at z* so the vortex waist
     lands at the FEM observation plane.
  2. Hourglass verification (ring radius vs z + symmetry).
  3. Overlay ASM vortex on FEM standing wave via KDTree IDW.
     Compute full 3-D gradient → Gor'kov potential U(x,y).
     Consolidated α-sweep panels.
  4. Stats + REPORT.md.

Outputs → results/waist_fix_overlay_alpha_sweep_YYYYMMDD_HHMMSS/

Usage:
    python scripts/dev/waist_fix_overlay_alpha_sweep.py
    python scripts/dev/waist_fix_overlay_alpha_sweep.py --cache path/to.npz
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
C_WATER   = 1484.0          # m/s
F_HZ      = 2.0e6
K_WATER   = 2.0 * np.pi * F_HZ / C_WATER
LAM       = C_WATER / F_HZ  # ≈ 0.742 mm
TRAP_SP   = LAM / 2.0       # ≈ 0.371 mm
OMEGA     = 2.0 * np.pi * F_HZ
RHO0      = 997.0           # kg/m³  water

# FEM domain geometry
LX = LY   = 6.0e-3
H_UNDER   = 3.0e-3
H_TOP     = 2.0085e-3
H_TOTAL   = H_UNDER + H_TOP
T_PML     = 1.0 * LAM
CX = CY   = LX / 2.0
Z_STAR    = H_UNDER + H_TOP / 2.0 + 0.25 * LAM   # ≈ 4.190 mm

# Particle properties (polystyrene in water)
RHO_P  = 1050.0   # kg/m³
C_P    = 2350.0    # m/s
A_P    = 5.0e-6    # m  (radius — 10 µm diameter)
V_P    = (4.0 / 3.0) * np.pi * A_P**3

KAPPA0  = 1.0 / (RHO0 * C_WATER**2)
KAPPA_P = 1.0 / (RHO_P * C_P**2)
F1      = 1.0 - KAPPA_P / KAPPA0
F2      = 2.0 * (RHO_P - RHO0) / (2.0 * RHO_P + RHO0)

# ASM grid
ASM_N   = 512
ASM_PAD = 2

# Output grids
NGRID_XY   = 500
NGRID_XZ_X = 500
NGRID_XZ_Z = 300

# α sweep values
ALPHAS = [0.02, 0.05, 0.1, 0.2]

# Gradient finite-difference step (for out-of-plane derivative)
DZ_GRAD = LAM / 15.0   # ≈ 0.049 mm

# ROI — large enough to see many traps clearly
ROI_HALF = 2.0e-3  # 2 mm half-width → 4 mm domain

STANDING_CACHE_DIR = PROJECT_ROOT / "results" / "fem_standing_wave_cache"

# ═══════════════════════════════════════════════════════════════════
# Converging vortex drive builder
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
        A = np.where(r_norm <= 1.0, 0.5 * (1.0 + np.cos(np.pi * r_norm)), 0.0)
    return xg, yg, A * np.exp(1j * phi)


# ═══════════════════════════════════════════════════════════════════
# Ring-finding helpers
# ═══════════════════════════════════════════════════════════════════

def radial_profile(I_2d, xg, yg, cx, cy, r_max=None, n_bins=300):
    XX, YY = np.meshgrid(xg, yg)
    R = np.sqrt((XX - cx)**2 + (YY - cy)**2)
    if r_max is None:
        r_max = 0.5 * min(xg[-1] - xg[0], yg[-1] - yg[0])
    r_bins = np.linspace(0, r_max, n_bins + 1)
    r_mid = 0.5 * (r_bins[:-1] + r_bins[1:])
    I_mean = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (R >= r_bins[i]) & (R < r_bins[i + 1])
        if np.any(mask):
            I_mean[i] = np.mean(I_2d[mask])
    return r_mid, I_mean


def find_ring(I_2d, xg, yg, cx, cy, r_max, r_min=0.02e-3, n_bins=300):
    r_mid, I_prof = radial_profile(I_2d, xg, yg, cx, cy, r_max=r_max, n_bins=n_bins)
    mask = r_mid >= r_min
    if not np.any(mask):
        return 0.0, 0.0
    idx = np.where(mask)[0]
    ipk = idx[np.argmax(I_prof[idx])]
    return r_mid[ipk], I_prof[ipk]


# ═══════════════════════════════════════════════════════════════════
# KDTree IDW
# ═══════════════════════════════════════════════════════════════════

def sample_idw(tree, p, pts, k=16, power=2.0, eps=1e-12):
    if k == 1:
        d, i = tree.query(pts, k=1)
        return p[i]
    d, i = tree.query(pts, k=k)
    w = 1.0 / (d**power + eps)
    w /= w.sum(axis=1, keepdims=True)
    return (p[i] * w).sum(axis=1)


# ═══════════════════════════════════════════════════════════════════
# Cache loader
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
    meta = {}
    for k in keys:
        if k not in {"coords", "p_real", "p_imag", "p"}:
            v = d[k]
            meta[k] = v.item() if isinstance(v, np.ndarray) and v.ndim == 0 else v
    return coords, p, meta


def find_latest_cache(d: Path) -> Path:
    fs = sorted(d.glob("*.npz"), key=lambda p: p.stat().st_mtime)
    if not fs:
        raise FileNotFoundError(f"No .npz in {d}")
    return fs[-1]


# ═══════════════════════════════════════════════════════════════════
# Logger
# ═══════════════════════════════════════════════════════════════════

class Log:
    def __init__(self, fp=None):
        self._fp = fp
    def __call__(self, msg=""):
        print(msg)
        if self._fp:
            self._fp.write(msg + "\n")
            self._fp.flush()


# ═══════════════════════════════════════════════════════════════════
# ASM interpolation
# ═══════════════════════════════════════════════════════════════════

def interp_asm_to_grid(p_full, xg_src, yg_src, XX_out, YY_out):
    pts = np.column_stack([YY_out.ravel(), XX_out.ravel()])
    ire = RegularGridInterpolator(
        (yg_src, xg_src), np.real(p_full),
        bounds_error=False, fill_value=0.0)
    iim = RegularGridInterpolator(
        (yg_src, xg_src), np.imag(p_full),
        bounds_error=False, fill_value=0.0)
    return (ire(pts) + 1j * iim(pts)).reshape(XX_out.shape)


# ═══════════════════════════════════════════════════════════════════
# Gor'kov potential from complex pressure + gradient
# ═══════════════════════════════════════════════════════════════════

def gorkov_potential(p_abs2, grad_p_mag2):
    """
    U = V_p * (f1·E_pot − 1.5·f2·E_kin)
    E_pot = |p|²·κ₀ / 4
    E_kin = |∇p|² / (4·ω²·ρ₀)
    """
    E_pot = 0.25 * p_abs2 * KAPPA0
    E_kin = grad_p_mag2 / (4.0 * OMEGA**2 * RHO0)
    return V_P * (F1 * E_pot - 1.5 * F2 * E_kin)


def grad_p_mag2_xy(p, p_zp, p_zm, dx, dy, dz):
    """
    |∇p|² on an XY grid given planes at z, z±dz.
    p shape: (ny, nx).
    """
    dpy, dpx = np.gradient(p, dy, dx)
    dpz = (p_zp - p_zm) / (2.0 * dz)
    return np.abs(dpx)**2 + np.abs(dpy)**2 + np.abs(dpz)**2


def grad_p_mag2_xz(p_xz, p_yp, p_ym, dx, dz, dy):
    """
    |∇p|² on an XZ grid given offset planes at y±dy.
    p_xz shape: (nz, nx).
    """
    dpz, dpx = np.gradient(p_xz, dz, dx)
    dpy = (p_yp - p_ym) / (2.0 * dy)
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

    OUT = PROJECT_ROOT / "results" / f"waist_fix_overlay_alpha_sweep_{TS}"
    FIG = OUT / "figures"; FIG.mkdir(parents=True, exist_ok=True)
    DAT = OUT / "data";    DAT.mkdir(parents=True, exist_ok=True)

    log_fp = open(OUT / "console_log.txt", "w")
    log = Log(log_fp)

    log("=" * 72)
    log("Gor'kov Waist-Fix + Vortex Overlay + α Sweep")
    log("=" * 72)
    log(f"Date  : {datetime.now().isoformat()}")
    log(f"Cache : {cache_path}")
    log(f"Output: {OUT}")
    log(f"λ = {LAM*1e3:.4f} mm   k = {K_WATER:.1f} rad/m   "
        f"trap λ/2 = {TRAP_SP*1e3:.4f} mm")
    log(f"Particle: polystyrene  a = {A_P*1e6:.0f} µm  "
        f"ρ_p = {RHO_P:.0f}  c_p = {C_P:.0f}")
    log(f"Contrast: f₁ = {F1:.4f}   f₂ = {F2:.4f}")
    log(f"z* = {Z_STAR*1e3:.4f} mm  (target waist plane)")
    log()

    # ==============================================================
    # STEP 1 — Calibration sweep targeting z_w ≈ z*
    # ==============================================================
    log("=" * 72)
    log("STEP 1: Calibration sweep (waist → z*)")
    log("=" * 72)

    ells   = [1, 2]
    # Focal lengths centred near z* ≈ 4.19 mm
    focals = [3.5e-3, 3.8e-3, 4.0e-3, 4.2e-3, 4.5e-3, 5.0e-3]
    R_vals = [3.5e-3, 5.0e-3, 7.0e-3]
    apods  = ["cosine_taper", "uniform"]

    J11_PRIME = 1.841

    cal_half = 3.0e-3
    cal_n_xy = 300
    xg_cal = np.linspace(CX - cal_half, CX + cal_half, cal_n_xy)
    yg_cal = np.linspace(CY - cal_half, CY + cal_half, cal_n_xy)

    zmin_cal, zmax_cal = 1.0e-3, 6.0e-3
    nz_cal = 50
    z_cal  = np.linspace(zmin_cal, zmax_cal, nz_cal)

    asm_domain_cal = 22.0e-3

    rows = []
    best = None

    n_cand = len(ells) * len(focals) * len(R_vals) * len(apods)
    log(f"  ℓ ∈ {ells}")
    log(f"  f ∈ [{', '.join(f'{f*1e3:.1f}' for f in focals)}] mm")
    log(f"  R ∈ [{', '.join(f'{r*1e3:.1f}' for r in R_vals)}] mm")
    log(f"  apod ∈ {apods}")
    log(f"  Total candidates: {n_cand}")
    log()

    t0_cal = time.time()
    for ell in ells:
        for focal in focals:
            for R_apt in R_vals:
                for apod in apods:
                    NA = R_apt / np.sqrt(R_apt**2 + focal**2)
                    NF = R_apt**2 / (LAM * focal)
                    r_theory = J11_PRIME * ell / (K_WATER * NA)

                    label = (f"ℓ={ell} f={focal*1e3:.1f} "
                             f"R={R_apt*1e3:.1f} {apod[:4]}")
                    xg_d, yg_d, D_cal = build_converging_drive(
                        ell, focal, R_apt, CX, CY,
                        asm_domain_cal, ASM_N, K_WATER, apod=apod)
                    dx_d = float(xg_d[1] - xg_d[0])

                    ring_r = np.zeros(nz_cal)
                    ring_I = np.zeros(nz_cal)

                    for iz, zz in enumerate(z_cal):
                        p_prop = propagate_pressure_asm(
                            D_cal, dx_d, dx_d, K_WATER, zz,
                            pad_factor=ASM_PAD)
                        ire = RegularGridInterpolator(
                            (yg_d, xg_d), np.real(p_prop),
                            bounds_error=False, fill_value=0.0)
                        iim = RegularGridInterpolator(
                            (yg_d, xg_d), np.imag(p_prop),
                            bounds_error=False, fill_value=0.0)
                        YYo, XXo = np.meshgrid(yg_cal, xg_cal, indexing="ij")
                        pts_yx = np.column_stack([YYo.ravel(), XXo.ravel()])
                        p_out = (ire(pts_yx) + 1j * iim(pts_yx)).reshape(
                            cal_n_xy, cal_n_xy)
                        I_out = np.abs(p_out)**2
                        rr, Ip = find_ring(I_out, xg_cal, yg_cal, CX, CY,
                                           r_max=cal_half * 0.9, r_min=0.01e-3)
                        ring_r[iz] = rr
                        ring_I[iz] = Ip

                    iz_w = np.argmax(ring_I)
                    z_w  = z_cal[iz_w]
                    r_w  = ring_r[iz_w]
                    D_w  = 2 * r_w

                    hourglass_ok = ((iz_w > 0) and (iz_w < nz_cal - 1) and
                                    (ring_r[0] > r_w * 0.95) and
                                    (ring_r[-1] > r_w * 0.95))

                    z_err = abs(z_w - Z_STAR)

                    row = {
                        "ell": ell, "f_mm": round(focal * 1e3, 2),
                        "R_mm": round(R_apt * 1e3, 1), "apod": apod,
                        "NA": round(NA, 4), "NF": round(NF, 2),
                        "z_w_mm": round(z_w * 1e3, 3),
                        "z_err_mm": round(z_err * 1e3, 3),
                        "r_w_mm": round(r_w * 1e3, 4),
                        "D_w_mm": round(D_w * 1e3, 4),
                        "D_w_over_half_lam": round(D_w / TRAP_SP, 4),
                        "D_theory_mm": round(2 * r_theory * 1e3, 4),
                        "hourglass": hourglass_ok,
                    }
                    rows.append(row)

                    flag = "✓" if hourglass_ok else "✗"
                    log(f"  {flag} {label:35s} NA={NA:.3f}  "
                        f"z_w={z_w*1e3:.2f}  z_err={z_err*1e3:.2f}  "
                        f"D_w={D_w*1e3:.3f}mm  "
                        f"D_w/(λ/2)={D_w/TRAP_SP:.2f}  hg={hourglass_ok}")

                    if hourglass_ok:
                        # Pick candidate closest to z*, then smallest D_w
                        if best is None:
                            better = True
                        else:
                            cur_z_err = abs(best["z_w"] - Z_STAR)
                            if z_err < cur_z_err - 0.2e-3:
                                better = True
                            elif z_err < cur_z_err + 0.2e-3 and D_w < best["D_w"]:
                                better = True
                            else:
                                better = False
                        if better:
                            best = dict(ell=ell, focal=focal, R=R_apt,
                                        apod=apod, D_w=D_w, r_w=r_w,
                                        z_w=z_w, NF=NF, NA=NA,
                                        ring_r=ring_r.copy(),
                                        ring_I=ring_I.copy())

    dt_cal = time.time() - t0_cal
    log(f"\n  Calibration: {dt_cal:.1f}s for {len(rows)} candidates")

    csv_path = DAT / "waist_calibration.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    log(f"  Saved {csv_path.name}")

    if best is None:
        log("  ✗ No hourglass found — selecting smallest z_err candidate.")
        rows_sorted = sorted(rows, key=lambda r: (r["z_err_mm"], r["D_w_mm"]))
        fb = rows_sorted[0]
        best = dict(ell=fb["ell"], focal=fb["f_mm"]*1e-3,
                    R=fb["R_mm"]*1e-3, apod=fb["apod"],
                    D_w=fb["D_w_mm"]*1e-3, r_w=fb["r_w_mm"]*1e-3,
                    z_w=fb["z_w_mm"]*1e-3, NF=fb["NF"], NA=fb["NA"],
                    ring_r=np.zeros(nz_cal), ring_I=np.zeros(nz_cal))

    NA_best = best["NA"]
    D_diff = 2 * J11_PRIME * best["ell"] / (K_WATER * NA_best)
    D_diff_ratio = D_diff / TRAP_SP

    log(f"\n  ★ SELECTED: ℓ={best['ell']}, f={best['focal']*1e3:.1f}mm, "
        f"R={best['R']*1e3:.1f}mm, apod={best['apod']}")
    log(f"    z_w = {best['z_w']*1e3:.3f} mm   (z* = {Z_STAR*1e3:.3f} mm, "
        f"err = {abs(best['z_w']-Z_STAR)*1e3:.3f} mm)")
    log(f"    D_w = {best['D_w']*1e3:.4f} mm  =  "
        f"{best['D_w']/TRAP_SP:.3f} × (λ/2)")
    log(f"    NA  = {NA_best:.4f}   NF = {best['NF']:.2f}")
    log(f"    Diffraction limit: D_min = {D_diff*1e3:.4f} mm  "
        f"= {D_diff_ratio:.3f} × (λ/2)")
    log()

    # ==============================================================
    # STEP 2 — Hourglass verification (fine z-scan)
    # ==============================================================
    log("=" * 72)
    log("STEP 2: Hourglass verification")
    log("=" * 72)

    ELL_B   = best["ell"]
    FOCAL_B = best["focal"]
    R_B     = best["R"]
    APOD_B  = best["apod"]

    asm_domain = max(LX, 2 * R_B + 4 * LAM)
    xg_d, yg_d, D_best = build_converging_drive(
        ELL_B, FOCAL_B, R_B, CX, CY, asm_domain, ASM_N, K_WATER, apod=APOD_B)
    dx_d = float(xg_d[1] - xg_d[0])

    nz_fine = 120
    z_fine = np.linspace(zmin_cal, zmax_cal, nz_fine)

    ver_half = min(R_B * 1.5, asm_domain / 2 - 1e-3)
    ver_n = 250
    xg_ver = np.linspace(CX - ver_half, CX + ver_half, ver_n)
    yg_ver = np.linspace(CY - ver_half, CY + ver_half, ver_n)

    ring_r_fine = np.zeros(nz_fine)
    ring_I_fine = np.zeros(nz_fine)

    log(f"  Fine scan: {nz_fine} z, {ver_n}² output")
    t0 = time.time()
    for iz, zz in enumerate(z_fine):
        p_prop = propagate_pressure_asm(D_best, dx_d, dx_d, K_WATER, zz,
                                        pad_factor=ASM_PAD)
        p_out = interp_asm_to_grid(p_prop, xg_d, yg_d,
                                   *np.meshgrid(xg_ver, yg_ver))
        rr, Ip = find_ring(np.abs(p_out)**2, xg_ver, yg_ver, CX, CY,
                            r_max=ver_half * 0.9)
        ring_r_fine[iz] = rr
        ring_I_fine[iz] = Ip
    log(f"  Done in {time.time()-t0:.1f}s")

    iz_w = np.argmax(ring_I_fine)
    z_w  = z_fine[iz_w]
    r_w  = ring_r_fine[iz_w]
    D_w  = 2 * r_w
    log(f"  Waist: z_w={z_w*1e3:.3f} mm  D_w={D_w*1e3:.4f} mm  "
        f"D_w/(λ/2)={D_w/TRAP_SP:.3f}")

    # A) Ring-radius vs z
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    z_mm = z_fine * 1e3
    ax1.plot(z_mm, ring_I_fine / ring_I_fine.max(), "b-", lw=2)
    ax1.axvline(z_w * 1e3, color="red", ls="--", lw=1.5,
                label=f"waist z={z_w*1e3:.2f}")
    ax1.axvline(Z_STAR * 1e3, color="lime", ls=":", lw=1.5,
                label=f"z*={Z_STAR*1e3:.2f}")
    ax1.set_ylabel("Norm. ring-peak intensity")
    ax1.set_title(f"ℓ={ELL_B}  R={R_B*1e3:.1f}mm  f={FOCAL_B*1e3:.1f}mm  "
                  f"{APOD_B}  NA={NA_best:.3f}")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(z_mm, ring_r_fine * 1e3, "b-", lw=2)
    ax2.axhline(r_w * 1e3, color="red", ls=":", lw=1, alpha=0.5,
                label=f"r_w={r_w*1e3:.3f}")
    ax2.axvline(z_w * 1e3, color="red", ls="--", lw=1.5)
    ax2.axvline(Z_STAR * 1e3, color="lime", ls=":", lw=1.5)
    ax2.set_xlabel("z (mm)"); ax2.set_ylabel("Ring radius (mm)")
    ax2.legend(); ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "ring_radius_vs_z.png", dpi=160)
    plt.close(fig)
    log("  Saved ring_radius_vs_z.png")

    # B) Symmetry check
    dz_sym = min(0.8e-3, (z_w - zmin_cal) * 0.4, (zmax_cal - z_w) * 0.4)
    sym_planes = {"lo": z_w - dz_sym, "waist": z_w, "hi": z_w + dz_sym}
    sym_half = min(2.0e-3, ver_half)
    sym_n = 200
    xg_sym = np.linspace(CX - sym_half, CX + sym_half, sym_n)
    yg_sym = np.linspace(CY - sym_half, CY + sym_half, sym_n)
    I_sym, r_sym = {}, {}
    for tag, zz in sym_planes.items():
        p_prop = propagate_pressure_asm(D_best, dx_d, dx_d, K_WATER, zz,
                                        pad_factor=ASM_PAD)
        p_out = interp_asm_to_grid(p_prop, xg_d, yg_d,
                                   *np.meshgrid(xg_sym, yg_sym))
        I_sym[tag] = np.abs(p_out)**2
        rr, _ = find_ring(I_sym[tag], xg_sym, yg_sym, CX, CY,
                          r_max=sym_half * 0.9)
        r_sym[tag] = rr

    ext_sym = [(xg_sym[0]-CX)*1e3, (xg_sym[-1]-CX)*1e3,
               (yg_sym[0]-CY)*1e3, (yg_sym[-1]-CY)*1e3]
    vmax_sym = max(v.max() for v in I_sym.values())
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (tag, zz) in zip(axes, sym_planes.items()):
        im = ax.imshow(I_sym[tag], extent=ext_sym, origin="lower",
                       cmap="inferno", vmin=0, vmax=vmax_sym, aspect="equal")
        c = plt.Circle((0, 0), r_sym[tag]*1e3, fill=False, ec="cyan",
                        lw=1.5, ls="--", alpha=0.8)
        ax.add_patch(c)
        ax.set_title(f"z={zz*1e3:.2f} ({tag})\nr={r_sym[tag]*1e3:.3f} mm",
                     fontsize=10)
        ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
        plt.colorbar(im, ax=ax, shrink=0.8, label="|p|²")
    fig.suptitle(f"Symmetry check  Δz=±{dz_sym*1e3:.2f}mm  "
                 f"ℓ={ELL_B}  R={R_B*1e3:.1f}  f={FOCAL_B*1e3:.1f}  {APOD_B}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "symmetry_check.png", dpi=160)
    plt.close(fig)
    sym_ok = 0.65 < r_sym["lo"] / (r_sym["hi"] + 1e-30) < 1.55
    log(f"  Symmetry: ratio={r_sym['lo']/(r_sym['hi']+1e-30):.3f}  "
        f"{'OK' if sym_ok else 'ASYM'}")
    log()

    # ==============================================================
    # STEP 3 — Overlay + Gor'kov potential
    # ==============================================================
    log("=" * 72)
    log("STEP 3: FEM + ASM overlay → Gor'kov potential")
    log("=" * 72)

    # ── Load FEM cache ────────────────────────────────────────────
    t0 = time.time()
    coords, p_fem, meta = load_fem_cache(cache_path)
    log(f"  FEM: {coords.shape[0]:,} DOFs  ({time.time()-t0:.2f}s)")

    tree = cKDTree(coords)
    K_IDW = 16

    probe = coords[np.random.default_rng(42).choice(
        len(coords), min(5000, len(coords)), replace=False)]
    dd, _ = tree.query(probe, k=2)
    med_nn = np.median(dd[:, 1])
    log(f"  Median NN: {med_nn*1e3:.4f} mm (λ/{LAM/med_nn:.1f})")

    # ── ROI grids ─────────────────────────────────────────────────
    roi_x = (CX - ROI_HALF, CX + ROI_HALF)
    roi_y = (CY - ROI_HALF, CY + ROI_HALF)

    xg_xy = np.linspace(roi_x[0], roi_x[1], NGRID_XY)
    yg_xy = np.linspace(roi_y[0], roi_y[1], NGRID_XY)
    dx_xy = float(xg_xy[1] - xg_xy[0])
    dy_xy = float(yg_xy[1] - yg_xy[0])
    XX_xy, YY_xy = np.meshgrid(xg_xy, yg_xy)

    z_phys_lo = T_PML + 0.1 * LAM
    z_phys_hi = H_TOTAL - 0.1 * LAM
    xg_xz = np.linspace(roi_x[0], roi_x[1], NGRID_XZ_X)
    zg_xz = np.linspace(z_phys_lo, z_phys_hi, NGRID_XZ_Z)
    dx_xz = float(xg_xz[1] - xg_xz[0])
    dz_xz = float(zg_xz[1] - zg_xz[0])
    XX_xz, ZZ_xz = np.meshgrid(xg_xz, zg_xz)

    ext_xy = [roi_x[0]*1e3, roi_x[1]*1e3, roi_y[0]*1e3, roi_y[1]*1e3]
    ext_xz = [roi_x[0]*1e3, roi_x[1]*1e3, z_phys_lo*1e3, z_phys_hi*1e3]

    log(f"  ROI: {2*ROI_HALF*1e3:.1f}mm × {2*ROI_HALF*1e3:.1f}mm  "
        f"({2*ROI_HALF/LAM:.1f}λ)")
    log(f"  z*     = {Z_STAR*1e3:.4f} mm")
    log(f"  z_w    = {z_w*1e3:.4f} mm")
    log(f"  dz_grad = {DZ_GRAD*1e3:.4f} mm (λ/{LAM/DZ_GRAD:.0f})")
    log()

    # ── Helper: FEM IDW on XY grid at given z ────────────────────
    def _fem_xy(z_val):
        pts = np.column_stack([XX_xy.ravel(), YY_xy.ravel(),
                               np.full(XX_xy.size, z_val)])
        return sample_idw(tree, p_fem, pts, k=K_IDW).reshape(XX_xy.shape)

    def _fem_xz(y_val):
        pts = np.column_stack([XX_xz.ravel(), np.full(XX_xz.size, y_val),
                               ZZ_xz.ravel()])
        return sample_idw(tree, p_fem, pts, k=K_IDW).reshape(XX_xz.shape)

    # ── Sample FEM standing wave: 3 z-offsets for gradient ───────
    log("  FEM sampling (XY at z*, z*±dz)...")
    t0 = time.time()
    p_s_zs   = _fem_xy(Z_STAR)
    p_s_zs_p = _fem_xy(Z_STAR + DZ_GRAD)
    p_s_zs_m = _fem_xy(Z_STAR - DZ_GRAD)
    log(f"    3 planes: {time.time()-t0:.2f}s  max|p_s|={np.abs(p_s_zs).max():.3f}")

    log("  FEM sampling (XZ at y=CY, CY±dy)...")
    t0 = time.time()
    p_s_xz   = _fem_xz(CY)
    p_s_xz_p = _fem_xz(CY + DZ_GRAD)
    p_s_xz_m = _fem_xz(CY - DZ_GRAD)
    log(f"    3 planes: {time.time()-t0:.2f}s  max|p_s|={np.abs(p_s_xz).max():.3f}")
    log()

    # ── ASM vortex fields ─────────────────────────────────────────
    log("  ASM vortex propagation...")

    xg_v, yg_v, D_v = build_converging_drive(
        ELL_B, FOCAL_B, R_B, CX, CY, asm_domain, ASM_N, K_WATER, apod=APOD_B)
    dx_v = float(xg_v[1] - xg_v[0])

    def _asm_xy(z_val):
        pf = propagate_pressure_asm(D_v, dx_v, dx_v, K_WATER, z_val,
                                    pad_factor=ASM_PAD, include_evanescent=True)
        return interp_asm_to_grid(pf, xg_v, yg_v, XX_xy, YY_xy)

    t0 = time.time()
    p_v_zs   = _asm_xy(Z_STAR)
    p_v_zs_p = _asm_xy(Z_STAR + DZ_GRAD)
    p_v_zs_m = _asm_xy(Z_STAR - DZ_GRAD)
    log(f"    XY z* ± dz: {time.time()-t0:.2f}s  max|p_v|={np.abs(p_v_zs).max():.6f}")

    # XZ vortex: extract 3 y-rows per z (y=CY, CY±dy)
    log(f"    XZ vortex ({NGRID_XZ_Z} z-planes, 3 y-offsets)...")
    t0 = time.time()
    iy_cen = np.argmin(np.abs(yg_v - CY))
    iy_lo  = np.argmin(np.abs(yg_v - (CY - DZ_GRAD)))
    iy_hi  = np.argmin(np.abs(yg_v - (CY + DZ_GRAD)))
    dy_vort_actual = float(yg_v[iy_hi] - yg_v[iy_lo]) / 2.0

    p_v_xz   = np.zeros((NGRID_XZ_Z, NGRID_XZ_X), dtype=complex)
    p_v_xz_p = np.zeros_like(p_v_xz)
    p_v_xz_m = np.zeros_like(p_v_xz)

    for iz, zz in enumerate(zg_xz):
        pz = propagate_pressure_asm(D_v, dx_v, dx_v, K_WATER, zz,
                                    pad_factor=ASM_PAD, include_evanescent=True)
        for arr, iy in [(p_v_xz, iy_cen), (p_v_xz_m, iy_lo), (p_v_xz_p, iy_hi)]:
            row = pz[iy, :]
            arr[iz, :] = np.interp(xg_xz, xg_v, np.real(row)) + \
                         1j * np.interp(xg_xz, xg_v, np.imag(row))
    log(f"    XZ vortex: {time.time()-t0:.1f}s  "
        f"max|p_v|={np.abs(p_v_xz).max():.6f}")
    log()

    # ── Normalise vortex to standing peak ────────────────────────
    stand_peak = np.abs(p_s_zs).max()
    vort_peak  = np.abs(p_v_zs).max() + 1e-30
    nfac = stand_peak / vort_peak
    log(f"  Standing peak: {stand_peak:.3f} Pa")
    log(f"  Vortex peak @ z*: {vort_peak:.6f}  → norm factor {nfac:.3f}")

    def _norm(a):
        return a * nfac

    pv_n_zs   = _norm(p_v_zs)
    pv_n_zs_p = _norm(p_v_zs_p)
    pv_n_zs_m = _norm(p_v_zs_m)
    pv_n_xz   = _norm(p_v_xz)
    pv_n_xz_p = _norm(p_v_xz_p)
    pv_n_xz_m = _norm(p_v_xz_m)

    # ── Compute standing-wave gradients ──────────────────────────
    log("  Computing gradients & Gor'kov potentials...")

    gp2_s_zs = grad_p_mag2_xy(p_s_zs, p_s_zs_p, p_s_zs_m,
                                dx_xy, dy_xy, DZ_GRAD)
    U_s_zs = gorkov_potential(np.abs(p_s_zs)**2, gp2_s_zs)

    gp2_s_xz = grad_p_mag2_xz(p_s_xz, p_s_xz_p, p_s_xz_m,
                                dx_xz, dz_xz, DZ_GRAD)
    U_s_xz = gorkov_potential(np.abs(p_s_xz)**2, gp2_s_xz)

    # Vortex-only Gorkov (for reference panel)
    gp2_v_zs = grad_p_mag2_xy(pv_n_zs, pv_n_zs_p, pv_n_zs_m,
                                dx_xy, dy_xy, DZ_GRAD)
    U_v_zs = gorkov_potential(np.abs(pv_n_zs)**2, gp2_v_zs)

    # Combined Gorkov for each α
    U_comb_zs = {}
    DU_zs     = {}
    for a in ALPHAS:
        p_t   = p_s_zs   + a * pv_n_zs
        p_t_p = p_s_zs_p + a * pv_n_zs_p
        p_t_m = p_s_zs_m + a * pv_n_zs_m
        gp2 = grad_p_mag2_xy(p_t, p_t_p, p_t_m, dx_xy, dy_xy, DZ_GRAD)
        U_comb_zs[a] = gorkov_potential(np.abs(p_t)**2, gp2)
        DU_zs[a] = U_comb_zs[a] - U_s_zs

    # XZ combined
    gp2_v_xz = grad_p_mag2_xz(pv_n_xz, pv_n_xz_p, pv_n_xz_m,
                                dx_xz, dz_xz,
                                max(dy_vort_actual, DZ_GRAD))
    U_v_xz = gorkov_potential(np.abs(pv_n_xz)**2, gp2_v_xz)

    U_comb_xz = {}
    for a in ALPHAS:
        p_t   = p_s_xz   + a * pv_n_xz
        p_t_p = p_s_xz_p + a * pv_n_xz_p
        p_t_m = p_s_xz_m + a * pv_n_xz_m
        gp2 = grad_p_mag2_xz(p_t, p_t_p, p_t_m,
                               dx_xz, dz_xz,
                               max(dy_vort_actual, DZ_GRAD))
        U_comb_xz[a] = gorkov_potential(np.abs(p_t)**2, gp2)

    trap_depth = float(U_s_zs.max() - U_s_zs.min())
    log(f"  Gorkov standing: min={U_s_zs.min():.3e}  max={U_s_zs.max():.3e}  "
        f"depth={trap_depth:.3e} J")

    # Perturbation stats
    perturb = {}
    for a in ALPHAS:
        mx = float(np.abs(DU_zs[a]).max() / (trap_depth + 1e-30))
        perturb[a] = {"max_rel_depth": mx}
        log(f"  α={a:.2f}: max |ΔU|/depth = {mx*100:.2f}%")
    log()

    # ==============================================================
    # FIGURES
    # ==============================================================
    log("=" * 72)
    log("FIGURES")
    log("=" * 72)

    def _trap_grid(ax, cen_x=CX, cen_y=CY):
        for off in np.arange(-12, 13) * TRAP_SP:
            ax.axhline((cen_y + off) * 1e3, color="white", lw=0.3, alpha=0.2)
            ax.axvline((cen_x + off) * 1e3, color="white", lw=0.3, alpha=0.2)

    def _xz_lines(ax):
        ax.axhline(Z_STAR * 1e3, color="cyan", lw=0.8, ls="--", alpha=0.7,
                    label="z*")
        ax.axhline(z_w * 1e3, color="lime", lw=0.8, ls=":", alpha=0.7,
                    label="z_w")

    # ── Figure A: XY Gor'kov α-sweep at z* ──────────────────────
    na = len(ALPHAS)
    ncol = na + 1
    fig, axes = plt.subplots(2, ncol, figsize=(3.8 * ncol, 7.5))
    kw = dict(extent=ext_xy, origin="lower", aspect="equal")

    # Row 0, col 0: U standing
    ax = axes[0, 0]
    vmin_U = min(U_s_zs.min(), min(U_comb_zs[a].min() for a in ALPHAS))
    vmax_U = max(U_s_zs.max(), max(U_comb_zs[a].max() for a in ALPHAS))
    im = ax.imshow(U_s_zs, cmap="RdBu_r", vmin=vmin_U, vmax=vmax_U, **kw)
    ax.set_title("FEM Standing\nU_Gorkov", fontsize=10, fontweight="bold",
                 color="navy")
    _trap_grid(ax)
    plt.colorbar(im, ax=ax, shrink=0.7, label="J")

    # Row 1, col 0: U vortex only
    ax = axes[1, 0]
    im = ax.imshow(U_v_zs, cmap="viridis", **kw)
    ax.set_title("ASM Vortex\nU_Gorkov (×1 norm)", fontsize=10,
                 fontweight="bold", color="darkgreen")
    _trap_grid(ax)
    plt.colorbar(im, ax=ax, shrink=0.7, label="J")

    # Common ΔU scale across all α
    du_max = max(np.abs(DU_zs[a]).max() for a in ALPHAS)
    du_lim = max(du_max, 1e-30)

    for j, a in enumerate(ALPHAS):
        col = j + 1

        # Row 0: combined Gorkov
        ax = axes[0, col]
        im = ax.imshow(U_comb_zs[a], cmap="RdBu_r",
                       vmin=vmin_U, vmax=vmax_U, **kw)
        ax.set_title(f"COMBINED  α={a}\nU(p_s + α·p_v)", fontsize=10,
                     fontweight="bold", color="darkred")
        _trap_grid(ax)
        plt.colorbar(im, ax=ax, shrink=0.7, label="J")

        # Row 1: ΔU
        ax = axes[1, col]
        im = ax.imshow(DU_zs[a], cmap="RdBu_r",
                       vmin=-du_lim, vmax=du_lim, **kw)
        ax.set_title(f"ΔU  α={a}\nmax|ΔU|/depth={perturb[a]['max_rel_depth']*100:.1f}%",
                     fontsize=9, fontweight="bold", color="purple")
        _trap_grid(ax)
        plt.colorbar(im, ax=ax, shrink=0.7, label="J")

    for ax in axes.flat:
        ax.set_xlabel("x (mm)", fontsize=8)
        ax.set_ylabel("y (mm)", fontsize=8)
        ax.tick_params(labelsize=7)

    fig.suptitle(
        f"Gor'kov XY at z* = {Z_STAR*1e3:.3f} mm  │  "
        f"ℓ={ELL_B}  R={R_B*1e3:.1f}mm  f={FOCAL_B*1e3:.1f}mm  "
        f"D_w={D_w*1e3:.3f}mm  {APOD_B}",
        fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIG / "XY_gorkov_sweep_zstar.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    log("  Saved XY_gorkov_sweep_zstar.png")

    # ── Figure B: XZ Gor'kov sweep ───────────────────────────────
    xz_alphas = [a for a in ALPHAS if a >= 0.1]
    ncol_xz = 2 + len(xz_alphas)
    fig, axes = plt.subplots(1, ncol_xz, figsize=(5.0 * ncol_xz, 5.5))
    kw_xz = dict(extent=ext_xz, origin="lower", aspect="auto")

    vmin_xz = min(U_s_xz.min(), min(U_comb_xz[a].min() for a in xz_alphas))
    vmax_xz = max(U_s_xz.max(), max(U_comb_xz[a].max() for a in xz_alphas))

    ax = axes[0]
    im = ax.imshow(U_s_xz, cmap="RdBu_r", vmin=vmin_xz, vmax=vmax_xz, **kw_xz)
    ax.set_title("FEM Standing\nU_Gorkov", fontsize=11, fontweight="bold",
                 color="navy")
    _xz_lines(ax); ax.legend(fontsize=7, loc="upper right")
    plt.colorbar(im, ax=ax, shrink=0.8, label="J")

    ax = axes[1]
    im = ax.imshow(U_v_xz, cmap="viridis", **kw_xz)
    ax.set_title("ASM Vortex\nU_Gorkov", fontsize=11, fontweight="bold",
                 color="darkgreen")
    _xz_lines(ax); ax.legend(fontsize=7, loc="upper right")
    plt.colorbar(im, ax=ax, shrink=0.8, label="J")

    for j, a in enumerate(xz_alphas):
        ax = axes[2 + j]
        im = ax.imshow(U_comb_xz[a], cmap="RdBu_r",
                       vmin=vmin_xz, vmax=vmax_xz, **kw_xz)
        ax.set_title(f"COMBINED  α={a}\nU(p_s + α·p_v)", fontsize=11,
                     fontweight="bold", color="darkred")
        _xz_lines(ax); ax.legend(fontsize=7, loc="upper right")
        plt.colorbar(im, ax=ax, shrink=0.8, label="J")

    for ax in axes:
        ax.set_xlabel("x (mm)"); ax.set_ylabel("z (mm)")

    fig.suptitle(
        f"Gor'kov XZ (y=CY)  │  ℓ={ELL_B}  R={R_B*1e3:.1f}mm  "
        f"f={FOCAL_B*1e3:.1f}mm  {APOD_B}",
        fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG / "XZ_gorkov_sweep.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    log("  Saved XZ_gorkov_sweep.png")

    # ── Figure C: Pressure overlay (compact) ─────────────────────
    # Quick 2×2 for reference: |p_stand|, |p_vort|, |p_comb α=0.1|, |p_comb α=0.2|
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    abs_s = np.abs(p_s_zs)
    vmax_p = abs_s.max()
    kw_p = dict(extent=ext_xy, origin="lower", aspect="equal", cmap="inferno",
                vmin=0, vmax=vmax_p)
    for ax_i, (ax, label, data) in enumerate(zip(axes, [
            "FEM Standing |p|", "ASM Vortex |p̂| (norm)",
            "Combined α=0.1", "Combined α=0.2"],
            [abs_s, np.abs(pv_n_zs),
             np.abs(p_s_zs + 0.1 * pv_n_zs),
             np.abs(p_s_zs + 0.2 * pv_n_zs)])):
        if ax_i == 1:
            im = ax.imshow(data, extent=ext_xy, origin="lower", aspect="equal",
                           cmap="viridis", vmin=0, vmax=data.max())
        else:
            im = ax.imshow(data, **kw_p)
        ax.set_title(label, fontsize=10, fontweight="bold")
        _trap_grid(ax)
        ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
        plt.colorbar(im, ax=ax, shrink=0.7, label="Pa")
    fig.suptitle(f"|p| overview at z*={Z_STAR*1e3:.3f} mm", fontsize=12,
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "pressure_overview_zstar.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    log("  Saved pressure_overview_zstar.png")
    log()

    # ==============================================================
    # STEP 4 — Stats + trap count
    # ==============================================================
    log("=" * 72)
    log("STEP 4: Stats & trap estimate")
    log("=" * 72)

    from scipy.ndimage import maximum_filter
    trap_count_info = {}
    for a in ALPHAS:
        du_rel = np.abs(DU_zs[a]) / (trap_depth + 1e-30)
        # Find trap nodes (Gorkov potential minima = standing wave nodes)
        U_neg = -U_s_zs
        lm = ((U_neg == maximum_filter(U_neg, size=15)) &
              (U_neg > 0.3 * U_neg.max()))
        n_peaks = int(lm.sum())
        n_affected = int((lm & (du_rel > 0.10)).sum())
        trap_count_info[str(a)] = {
            "n_traps": n_peaks, "n_affected_10pct": n_affected}
        log(f"  α={a:.2f}: {n_peaks} traps, "
            f"{n_affected} with >10% depth perturbation")

    stats = {
        "lambda_mm": float(LAM * 1e3),
        "trap_spacing_mm": float(TRAP_SP * 1e3),
        "z_star_mm": round(float(Z_STAR * 1e3), 4),
        "gorkov_formula": "U = V_p*(f1*E_pot - 1.5*f2*E_kin)",
        "particle": {
            "material": "polystyrene",
            "rho_p": RHO_P, "c_p": C_P,
            "a_um": A_P * 1e6,
            "f1": round(F1, 5), "f2": round(F2, 5),
        },
        "best_vortex": {
            "ell": int(ELL_B), "R_mm": float(R_B * 1e3),
            "f_mm": float(FOCAL_B * 1e3), "apodization": APOD_B,
            "NA": round(float(NA_best), 4), "NF": round(float(best["NF"]), 3),
            "sign_convention": "phi = ell*theta - k*(sqrt(r^2+f^2) - f) [CONVERGING]",
        },
        "waist": {
            "z_w_mm": round(float(z_w * 1e3), 4),
            "z_err_mm": round(float(abs(z_w - Z_STAR) * 1e3), 4),
            "D_w_mm": round(float(D_w * 1e3), 4),
            "D_w_over_half_lam": round(float(D_w / TRAP_SP), 4),
            "diffraction_limit_D_mm": round(float(D_diff * 1e3), 4),
            "diffraction_limit_ratio": round(float(D_diff_ratio), 4),
        },
        "symmetry": {
            "r_lo_mm": round(float(r_sym["lo"] * 1e3), 4),
            "r_hi_mm": round(float(r_sym["hi"] * 1e3), 4),
            "ratio": round(float(r_sym["lo"] / (r_sym["hi"] + 1e-30)), 4),
            "ok": bool(sym_ok),
        },
        "gorkov_standing": {
            "min_J": float(U_s_zs.min()),
            "max_J": float(U_s_zs.max()),
            "trap_depth_J": float(trap_depth),
        },
        "perturbation": {str(a): v for a, v in perturb.items()},
        "trap_count": trap_count_info,
        "idw_k": K_IDW,
        "dz_grad_mm": round(float(DZ_GRAD * 1e3), 4),
        "roi_half_mm": round(float(ROI_HALF * 1e3), 4),
    }

    with open(DAT / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    log(f"  Saved stats.json")

    np.savez_compressed(DAT / "fields.npz",
                        xg_xy=xg_xy, yg_xy=yg_xy,
                        xg_xz=xg_xz, zg_xz=zg_xz,
                        U_s_zs=U_s_zs, U_v_zs=U_v_zs,
                        U_s_xz=U_s_xz, U_v_xz=U_v_xz,
                        p_s_zs=p_s_zs, pv_n_zs=pv_n_zs,
                        z_w=z_w, z_star=Z_STAR)
    log(f"  Saved fields.npz")
    log()

    # ==============================================================
    # REPORT.md
    # ==============================================================
    elapsed = time.time() - t_GLOBAL
    log(f"Total runtime: {elapsed:.1f}s")

    rp = []
    rp.append("# Gor'kov Waist-Fix + Vortex Overlay + α Sweep")
    rp.append(f"\n**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    rp.append(f"**Runtime**: {elapsed:.1f}s")
    rp.append(f"**Cache**: `{cache_path.name}` ({coords.shape[0]:,} DOFs)")
    rp.append("")

    rp.append("## 0. What changed from previous run")
    rp.append("")
    rp.append("1. **Gor'kov potential** instead of raw |p|: computes full 3D "
              "gradient via central differences (dz = λ/15) to get the "
              "velocity term. Formula: "
              "U = V_p·(f₁·E_pot − 1.5·f₂·E_kin).")
    rp.append("2. **Focal tuned so waist lands at z*** — swept f ∈ "
              f"[{focals[0]*1e3:.1f}, {focals[-1]*1e3:.1f}] mm "
              f"around z*={Z_STAR*1e3:.2f} mm.")
    rp.append("3. **Expanded ROI** to {:.1f} mm × {:.1f} mm "
              "(~{:.0f} trap spacings) — eliminates wasted whitespace."
              .format(2*ROI_HALF*1e3, 2*ROI_HALF*1e3, 2*ROI_HALF/TRAP_SP))
    rp.append("4. **ΔU perturbation** scaled to a common colorbar across all α, "
              "normalised by trap depth — no more white-box issue.")
    rp.append("")

    rp.append("## 1. Waist calibration (targeted at z*)")
    rp.append("")
    rp.append("| ℓ | f (mm) | R (mm) | apod | NA | z_w (mm) | z_err (mm) | "
              "D_w (mm) | D_w/(λ/2) | hg |")
    rp.append("|" + "|".join(["---"] * 10) + "|")
    rows_sorted = sorted(rows, key=lambda r: (r["z_err_mm"], r["D_w_mm"]))
    for row in rows_sorted[:15]:
        rp.append(
            f"| {row['ell']} | {row['f_mm']:.1f} | {row['R_mm']:.1f} | "
            f"{row['apod'][:4]} | {row['NA']:.3f} | "
            f"{row['z_w_mm']:.3f} | {row['z_err_mm']:.3f} | "
            f"{row['D_w_mm']:.4f} | {row['D_w_over_half_lam']:.3f} | "
            f"{'✓' if row['hourglass'] else '✗'} |")
    if len(rows) > 15:
        rp.append(f"| ... | | | | | | | | | ({len(rows)-15} more) |")
    rp.append("")

    rp.append(f"**Selected**: ℓ={ELL_B}, f={FOCAL_B*1e3:.1f} mm, "
              f"R={R_B*1e3:.1f} mm, apod={APOD_B}")
    rp.append(f"- z_w = {z_w*1e3:.3f} mm   (z* = {Z_STAR*1e3:.3f} mm, "
              f"err = {abs(z_w - Z_STAR)*1e3:.3f} mm)")
    rp.append(f"- **D_w = {D_w*1e3:.4f} mm = {D_w/TRAP_SP:.3f} × (λ/2)**")
    rp.append(f"- NA = {NA_best:.4f}")
    rp.append(f"- Diffraction limit: D_min = {D_diff*1e3:.4f} mm "
              f"= {D_diff_ratio:.3f} × (λ/2)")
    rp.append("")

    rp.append("## 2. Hourglass verification")
    rp.append("")
    rp.append("![ring_radius_vs_z](figures/ring_radius_vs_z.png)")
    rp.append("")
    rp.append("![symmetry_check](figures/symmetry_check.png)")
    rp.append("")
    rp.append(f"Symmetry ratio = "
              f"{r_sym['lo']/(r_sym['hi']+1e-30):.3f} "
              f"({'OK' if sym_ok else 'asymmetric'})")
    rp.append("")

    rp.append("## 3. Gor'kov potential overlay")
    rp.append("")
    rp.append(f"Particle: polystyrene a={A_P*1e6:.0f} µm, "
              f"f₁={F1:.4f}, f₂={F2:.4f}")
    rp.append(f"Gradient step dz = {DZ_GRAD*1e3:.4f} mm (λ/{LAM/DZ_GRAD:.0f})")
    rp.append(f"Standing trap depth = {trap_depth:.3e} J")
    rp.append("")

    rp.append("### α sweep results")
    rp.append("")
    rp.append("| α | max |ΔU|/depth | traps >10% | traps total |")
    rp.append("|---|-----------------|------------|-------------|")
    for a in ALPHAS:
        rp.append(f"| {a:.2f} | "
                  f"{perturb[a]['max_rel_depth']*100:.1f}% | "
                  f"{trap_count_info[str(a)]['n_affected_10pct']} | "
                  f"{trap_count_info[str(a)]['n_traps']} |")
    rp.append("")

    rp.append("### XY Gor'kov at z*")
    rp.append("")
    rp.append("![XY_gorkov_sweep_zstar](figures/XY_gorkov_sweep_zstar.png)")
    rp.append("")
    rp.append("Top row: Gor'kov potential (standing, then combined at each α). "
              "Bottom row: vortex-only reference, then ΔU = U_combined − U_standing "
              "for each α on a **shared diverging colorbar** so the sweep "
              "progression is visible left→right.")
    rp.append("")

    rp.append("### XZ Gor'kov")
    rp.append("")
    rp.append("![XZ_gorkov_sweep](figures/XZ_gorkov_sweep.png)")
    rp.append("")

    rp.append("### Pressure overview")
    rp.append("")
    rp.append("![pressure_overview](figures/pressure_overview_zstar.png)")
    rp.append("")

    rp.append("## 4. Conclusion")
    rp.append("")
    rp.append(f"Vortex waist placed at z_w = {z_w*1e3:.3f} mm "
              f"(z* = {Z_STAR*1e3:.3f} mm, err = "
              f"{abs(z_w - Z_STAR)*1e3:.3f} mm).")
    rp.append(f"D_w = {D_w*1e3:.3f} mm = {D_w/TRAP_SP:.2f} × (λ/2). "
              f"Diffraction limit at NA={NA_best:.3f} is "
              f"{D_diff_ratio:.2f} × (λ/2).")
    rp.append("")
    if trap_depth > 0:
        a_ref = 0.1
        n_aff = trap_count_info[str(a_ref)]["n_affected_10pct"]
        n_tot = trap_count_info[str(a_ref)]["n_traps"]
        rp.append(f"At α={a_ref}, {n_aff} of {n_tot} trap sites exceed 10% "
                  f"Gor'kov depth perturbation. The vortex is concentrated "
                  f"near the beam axis so most traps are unperturbed.")
    rp.append("")

    (OUT / "REPORT.md").write_text("\n".join(rp))
    log("  Saved REPORT.md")
    log()
    log("=" * 72)
    log("DONE")
    log("=" * 72)
    log_fp.close()


if __name__ == "__main__":
    main()
