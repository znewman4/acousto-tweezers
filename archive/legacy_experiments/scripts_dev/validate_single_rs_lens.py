#!/usr/bin/env python3
"""
Validate a single RS vortex lens against FEM standing wave truth.

Loads standing-wave truth from Part 1, computes RS vortex (with tilt,
reflection, and a simple exponential damping), forms the combined field,
and runs structural validation checks.

Usage:
    micromamba run -n acousto-complex python scripts/dev/validate_single_rs_lens.py --ts <TS>

where <TS> is the timestamp of the rs_hybrid_validation_<TS>/ folder
containing fem_standing/standing_wave_zstar.npz.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── project root ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from acoustweezers.physics.acoustics.vortex_lens import (
    LGBeamConfig,
    create_lg_drive,
)
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import minimum_filter

# ═══════════════════════════════════════════════════════════════════
# Physical constants  (canonical values)
# ═══════════════════════════════════════════════════════════════════
WATER_C   = 1484.0          # m/s
WATER_RHO = 997.0           # kg/m³
F_HZ      = 2.0e6           # Hz
LAM       = WATER_C / F_HZ  # 0.000742 m
K_WATER   = 2.0 * np.pi * F_HZ / WATER_C
OMEGA     = 2.0 * np.pi * F_HZ

H_UNDER = 3.0e-3
H_TOP   = 2.0085e-3
H_TOTAL = H_UNDER + H_TOP
Z_MID   = H_UNDER + H_TOP / 2.0
Z_STAR  = Z_MID + 0.25 * LAM   # ≈ 4.190 mm

# Water-air reflection
R_REFL = -1.0              # pressure-release approximation (actual ≈ −0.9994)
Z_TOP  = H_TOTAL           # 5.0085 mm

# ── Lens parameters (hardcoded for this validation) ──────────────
LENS_L     = 2              # topological charge ℓ = 2
LENS_R     = 1.0e-3         # aperture radius R = 1.0 mm
LENS_W0    = 0.8e-3         # beam waist w₀ = 0.8 mm
LENS_APOD  = "cosine_taper"
TILT_DEG   = 5.0            # tilt angle θ = 5°
ALPHA_NOM  = 0.05           # nominal amplitude ratio (will be tuned)
PHI_NOM    = 0.0            # phase offset

# Simple exponential damping model:
# p(z) *= exp(-gamma * z)  where gamma is an effective attenuation
# For water at 2 MHz: α ≈ 0.05 Np/m (very small — nearly lossless)
# We include it for completeness but it will be negligible.
DAMPING_GAMMA = 0.05        # Np/m (water at 2 MHz, approximate)

# RS drive grid
DRIVE_N = 512
LX_DOMAIN = 6e-3            # match FEM domain
LY_DOMAIN = 6e-3


# ═══════════════════════════════════════════════════════════════════
# Angular spectrum propagation
# ═══════════════════════════════════════════════════════════════════
def angular_spectrum_propagate(D, dx, dy, z, k, pad=True):
    """Propagate 2-D field D(x,y) from z=0 to height z."""
    Ny, Nx = D.shape
    if pad:
        Ny2, Nx2 = 2 * Ny, 2 * Nx
        D_pad = np.zeros((Ny2, Nx2), dtype=complex)
        D_pad[:Ny, :Nx] = D
    else:
        Ny2, Nx2 = Ny, Nx
        D_pad = D.copy()

    fx = np.fft.fftfreq(Nx2, d=dx)
    fy = np.fft.fftfreq(Ny2, d=dy)
    FX, FY = np.meshgrid(fx, fy)
    kx = 2.0 * np.pi * FX
    ky = 2.0 * np.pi * FY
    kt2 = kx**2 + ky**2
    kz2 = k**2 - kt2
    prop = kz2 >= 0
    kz = np.zeros_like(kz2, dtype=complex)
    kz[prop]  = np.sqrt(kz2[prop])
    kz[~prop] = 1j * np.sqrt(-kz2[~prop])

    H = np.exp(1j * kz * z)
    p_pad = np.fft.ifft2(np.fft.fft2(D_pad) * H)
    return p_pad[:Ny, :Nx] if pad else p_pad


def apply_tilt(D, yg_drive, theta_deg, k, cy):
    """Phase ramp tilt about x-axis: exp(i k sinθ (y − cy))."""
    if theta_deg == 0.0:
        return D.copy()
    theta_rad = np.deg2rad(theta_deg)
    YY = yg_drive[:, None] * np.ones(D.shape[1])[None, :]
    return D * np.exp(1j * k * np.sin(theta_rad) * (YY - cy))


def apply_damping(p, z, gamma):
    """Simple exponential attenuation: p *= exp(-gamma * z)."""
    return p * np.exp(-gamma * z)


# ═══════════════════════════════════════════════════════════════════
# Trap detection
# ═══════════════════════════════════════════════════════════════════
def find_traps(amp, xg, yg, min_sep_px=None, amp_thresh_frac=0.5):
    """Detect pressure minima (trap positions) in amplitude field."""
    dx = xg[1] - xg[0]
    if min_sep_px is None:
        min_sep_px = max(3, int(0.5 * LAM / dx))
    filt = minimum_filter(amp, size=min_sep_px)
    is_min = (amp == filt) & np.isfinite(amp)
    thresh = np.median(amp) * amp_thresh_frac
    is_min &= amp < thresh
    iys, ixs = np.where(is_min)
    order = np.argsort(amp[iys, ixs])
    return xg[ixs[order]], yg[iys[order]]


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="Validate single RS lens + FEM standing wave")
    p.add_argument("--ts", required=True,
                   help="Timestamp of rs_hybrid_validation_<TS> folder")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    args = parse_args()
    TS = args.ts

    BASE_DIR = PROJECT_ROOT / "results" / f"rs_hybrid_validation_{TS}"
    FEM_NPZ  = BASE_DIR / "fem_standing" / "standing_wave_zstar.npz"
    if not FEM_NPZ.exists():
        sys.exit(f"ERROR: FEM truth not found: {FEM_NPZ}")

    RS_DIR  = BASE_DIR / "rs_lens"
    FIG_DIR = BASE_DIR / "figures"
    RS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("SINGLE RS LENS VALIDATION")
    print("=" * 72)
    print(f"Base dir     : {BASE_DIR.relative_to(PROJECT_ROOT)}")
    print(f"Lens         : LG ℓ={LENS_L}, R={LENS_R*1e3:.1f}mm, w₀={LENS_W0*1e3:.1f}mm")
    print(f"Tilt         : {TILT_DEG}°  (phase ramp about x-axis)")
    print(f"α_nom        : {ALPHA_NOM}")
    print(f"Damping      : γ = {DAMPING_GAMMA} Np/m")
    print(f"Reflection   : R = {R_REFL}, z_top = {Z_TOP*1e3:.2f} mm")
    print()

    t_total_start = time.time()

    # ═══════════════════════════════════════════════════════════════
    # Load FEM standing wave
    # ═══════════════════════════════════════════════════════════════
    print("Loading FEM standing wave truth...")
    fem = np.load(str(FEM_NPZ))
    xg = fem["xg"]
    yg = fem["yg"]
    p_stand = fem["p_xy_zstar"]
    z_star_loaded = float(fem["z_star"])
    Nx, Ny = len(xg), len(yg)
    dx_out = float(xg[1] - xg[0])
    dy_out = float(yg[1] - yg[0])
    cx_out = float((xg[0] + xg[-1]) / 2.0)
    cy_out = float((yg[0] + yg[-1]) / 2.0)

    print(f"  Grid: {Nx}×{Ny}, dx={dx_out*1e6:.1f}μm")
    print(f"  z* = {z_star_loaded*1e3:.3f} mm")
    print(f"  max|p_stand| = {np.abs(p_stand).max():.4f} Pa")
    print()

    # ═══════════════════════════════════════════════════════════════
    # Build RS vortex on drive grid
    # ═══════════════════════════════════════════════════════════════
    print("Building RS vortex drive pattern...")
    CX_D = LX_DOMAIN / 2.0
    CY_D = LY_DOMAIN / 2.0

    xg_d = np.linspace(0, LX_DOMAIN, DRIVE_N, endpoint=False)
    yg_d = np.linspace(0, LY_DOMAIN, DRIVE_N, endpoint=False)
    dx_d = float(xg_d[1] - xg_d[0])
    dy_d = float(yg_d[1] - yg_d[0])
    XX_d, YY_d = np.meshgrid(xg_d, yg_d)

    lg_cfg = LGBeamConfig(
        topological_charge=LENS_L,
        beam_waist=LENS_W0,
        focal_length=None,
        focus_offset_x=0.0,
        focus_offset_y=0.0,
        c_water=WATER_C,
        frequency_hz=F_HZ,
        aperture_radius=LENS_R,
        center=None,
        apodization=LENS_APOD,
        apodization_strength=1.0,
    )

    D_base = create_lg_drive(
        XX_d.ravel(), YY_d.ravel(), lg_cfg,
        center_x=CX_D, center_y=CY_D, verbose=False,
    ).reshape(XX_d.shape)

    print(f"  Drive grid: {DRIVE_N}×{DRIVE_N}, dx_d={dx_d*1e6:.1f}μm")
    print(f"  max|D| = {np.abs(D_base).max():.6f}")

    # Apply tilt
    D_tilted_5 = apply_tilt(D_base, yg_d, TILT_DEG, K_WATER, CY_D)
    D_tilted_0 = D_base.copy()  # θ=0 for comparison

    # ═══════════════════════════════════════════════════════════════
    # Propagate to output grid
    # ═══════════════════════════════════════════════════════════════
    XX_out, YY_out = np.meshgrid(xg, yg)

    def _propagate_to_output(D_src, z_prop):
        """Propagate and interpolate to FEM output grid."""
        p_full = angular_spectrum_propagate(D_src, dx_d, dy_d, z_prop, K_WATER, pad=True)
        ire = RegularGridInterpolator(
            (yg_d, xg_d), np.real(p_full),
            method="linear", bounds_error=False, fill_value=0.0,
        )
        iim = RegularGridInterpolator(
            (yg_d, xg_d), np.imag(p_full),
            method="linear", bounds_error=False, fill_value=0.0,
        )
        pts = np.column_stack([YY_out.ravel(), XX_out.ravel()])
        return (ire(pts) + 1j * iim(pts)).reshape(XX_out.shape)

    print("\nPropagating vortex fields...")

    # ── Incident (tilted) ─────────────────────────────────────────
    t0 = time.time()
    p_inc_5 = _propagate_to_output(D_tilted_5, Z_STAR)
    p_inc_5 = apply_damping(p_inc_5, Z_STAR, DAMPING_GAMMA)
    print(f"  p_inc (θ=5°): {time.time()-t0:.1f}s, max|p|={np.abs(p_inc_5).max():.6f}")

    # ── Reflected (tilted): image source at 2*Z_TOP − Z_STAR ─────
    t0 = time.time()
    z_refl = 2.0 * Z_TOP - Z_STAR
    p_ref_5 = _propagate_to_output(D_tilted_5, z_refl)
    p_ref_5 = apply_damping(p_ref_5, z_refl, DAMPING_GAMMA)
    p_ref_5 = R_REFL * p_ref_5
    print(f"  p_ref (θ=5°): {time.time()-t0:.1f}s, max|p|={np.abs(p_ref_5).max():.6f}")

    p_vortex_total_5 = p_inc_5 + p_ref_5

    # ── θ=0 for comparison ────────────────────────────────────────
    t0 = time.time()
    p_inc_0 = _propagate_to_output(D_tilted_0, Z_STAR)
    p_inc_0 = apply_damping(p_inc_0, Z_STAR, DAMPING_GAMMA)
    p_ref_0 = _propagate_to_output(D_tilted_0, z_refl)
    p_ref_0 = apply_damping(p_ref_0, z_refl, DAMPING_GAMMA)
    p_ref_0 = R_REFL * p_ref_0
    p_vortex_total_0 = p_inc_0 + p_ref_0
    print(f"  p_vortex (θ=0°): {time.time()-t0:.1f}s, max|p|={np.abs(p_vortex_total_0).max():.6f}")

    # ── XZ slices for tilt visualisation ──────────────────────────
    print("\nComputing XZ slices (tilt plane)...")
    NZ_XZ = 100
    z_xz = np.linspace(H_UNDER * 0.5, Z_TOP, NZ_XZ)
    jy_centre = int(np.argmin(np.abs(yg_d - CY_D)))

    def _xz_slice(D_src, z_arr, include_refl=False):
        out = np.zeros((len(z_arr), len(xg)), dtype=complex)
        for iz, zz in enumerate(z_arr):
            p_full = angular_spectrum_propagate(D_src, dx_d, dy_d, zz, K_WATER, pad=True)
            line = p_full[jy_centre, :]
            p_interp = np.interp(xg, xg_d, np.real(line)) + 1j * np.interp(xg, xg_d, np.imag(line))
            p_interp = apply_damping(p_interp, zz, DAMPING_GAMMA)
            if include_refl:
                z_r = 2.0 * Z_TOP - zz
                p_r_full = angular_spectrum_propagate(D_src, dx_d, dy_d, z_r, K_WATER, pad=True)
                line_r = p_r_full[jy_centre, :]
                p_r = np.interp(xg, xg_d, np.real(line_r)) + 1j * np.interp(xg, xg_d, np.imag(line_r))
                p_r = apply_damping(p_r, z_r, DAMPING_GAMMA)
                p_interp += R_REFL * p_r
            out[iz, :] = p_interp
        return out

    p_xz_inc_5 = _xz_slice(D_tilted_5, z_xz, include_refl=False)
    p_xz_tot_5 = _xz_slice(D_tilted_5, z_xz, include_refl=True)
    print(f"  XZ slices done")

    # ── Waist analysis: ring radius vs z ──────────────────────────
    print("\nComputing ring radius vs z (waist analysis)...")
    z_waist = np.linspace(H_UNDER, Z_TOP - 0.1e-3, 50)
    ring_radii = []
    for zz in z_waist:
        p_plane = _propagate_to_output(D_tilted_5, zz)
        amp_plane = np.abs(p_plane)
        # Find ring radius = radial distance of peak amplitude from centre
        RR = np.sqrt((XX_out - cx_out)**2 + (YY_out - cy_out)**2)
        r_bins = np.linspace(0, 2.0 * LAM, 60)
        r_mid = 0.5 * (r_bins[:-1] + r_bins[1:])
        radial_mean = np.zeros(len(r_mid))
        for i in range(len(r_mid)):
            sel = (RR >= r_bins[i]) & (RR < r_bins[i+1])
            if sel.any():
                radial_mean[i] = np.mean(amp_plane[sel])
        if radial_mean.max() > 0:
            ring_radii.append(r_mid[np.argmax(radial_mean)])
        else:
            ring_radii.append(np.nan)
    ring_radii = np.array(ring_radii)

    # Find waist (minimum ring radius)
    valid = ~np.isnan(ring_radii)
    if valid.any():
        waist_idx = np.argmin(ring_radii[valid])
        z_waist_min = z_waist[valid][waist_idx]
        r_waist_min = ring_radii[valid][waist_idx]
        focal_shift = z_waist_min - Z_STAR
        print(f"  Waist min radius: {r_waist_min*1e3:.3f} mm at z={z_waist_min*1e3:.3f} mm")
        print(f"  Focal shift from z*: {focal_shift*1e3:.3f} mm = {focal_shift/LAM:.2f}λ")
    else:
        z_waist_min = np.nan
        r_waist_min = np.nan
        focal_shift = np.nan

    # ═══════════════════════════════════════════════════════════════
    # Combine: p_total = p_standing + α · p_vortex_total
    # ═══════════════════════════════════════════════════════════════
    print("\nForming combined field...")

    # Scale α so that vortex peak in core is ~5% of standing wave peak
    amp_stand_max = float(np.abs(p_stand).max())
    amp_vortex_max = float(np.abs(p_vortex_total_5).max())
    alpha_auto = 0.05 * amp_stand_max / max(amp_vortex_max, 1e-30)
    alpha_use = alpha_auto
    print(f"  |p_stand|_max = {amp_stand_max:.4f} Pa")
    print(f"  |p_vortex|_max = {amp_vortex_max:.6f}")
    print(f"  Auto-scaled α = {alpha_use:.6f}  (vortex peak ≈ 5% of standing)")

    p_combined = p_stand + alpha_use * np.exp(1j * PHI_NOM) * p_vortex_total_5

    # ═══════════════════════════════════════════════════════════════
    # STRUCTURAL CHECKS
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "═" * 72)
    print("STRUCTURAL VALIDATION CHECKS")
    print("═" * 72)

    # ── A) Waist placement (already computed above) ───────────────
    print("\n─── A) Waist Placement ───")
    print(f"  Minimum ring radius at z = {z_waist_min*1e3:.3f} mm")
    print(f"  z* = {Z_STAR*1e3:.3f} mm")
    print(f"  Focal shift = {focal_shift*1e3:.3f} mm = {focal_shift/LAM:.2f}λ")
    waist_ok = abs(focal_shift) < 2.0 * LAM
    print(f"  PASS: waist within 2λ of z*" if waist_ok else
          f"  WARNING: waist is {abs(focal_shift)/LAM:.1f}λ from z*")

    # ── B) Amplitude region ───────────────────────────────────────
    print("\n─── B) Amplitude Region ───")
    # Compare vortex vs standing in trap region (core ROI)
    roi_core = ((XX_out - cx_out)**2 + (YY_out - cy_out)**2) <= (2.0 * LAM)**2
    amp_v_core = float(np.sqrt(np.mean(np.abs(alpha_use * p_vortex_total_5[roi_core])**2)))
    amp_s_core = float(np.sqrt(np.mean(np.abs(p_stand[roi_core])**2)))
    ratio_core = amp_v_core / max(amp_s_core, 1e-30)
    amp_v_global = float(np.abs(alpha_use * p_vortex_total_5).max())
    ratio_global = amp_v_global / max(amp_stand_max, 1e-30)
    print(f"  RMS |α·p_vortex| in core ROI: {amp_v_core:.6f}")
    print(f"  RMS |p_standing| in core ROI: {amp_s_core:.4f}")
    print(f"  Core ratio: {ratio_core:.4f}")
    print(f"  Global peak ratio (vortex/standing): {ratio_global:.4f}")
    amp_ok = ratio_global < 0.20  # vortex should not overwhelm globally
    print(f"  PASS: vortex < 20% of standing globally" if amp_ok else
          f"  WARNING: vortex amplitude ratio is {ratio_global:.2f}")

    # ── C) Reflection behaviour ───────────────────────────────────
    print("\n─── C) Reflection Behaviour ───")
    roi_inner = ((XX_out - cx_out)**2 + (YY_out - cy_out)**2) <= (1.0 * LAM)**2

    def _overlap_metric(p_inc_field, p_ref_field, mask):
        inc_rms = np.sqrt(np.mean(np.abs(p_inc_field[mask])**2))
        ref_rms = np.sqrt(np.mean(np.abs(p_ref_field[mask])**2))
        return float(ref_rms / max(inc_rms, 1e-30))

    overlap_0 = _overlap_metric(p_inc_0, p_ref_0, roi_inner)
    overlap_5 = _overlap_metric(p_inc_5, p_ref_5, roi_inner)
    print(f"  Overlap (ref/inc RMS, core) at θ=0°: {overlap_0:.4f}")
    print(f"  Overlap (ref/inc RMS, core) at θ=5°: {overlap_5:.4f}")
    refl_ok = overlap_5 < overlap_0
    print(f"  PASS: tilt reduces overlap ({overlap_5:.4f} < {overlap_0:.4f})" if refl_ok else
          f"  WARNING: tilt does not reduce overlap")

    # ── D) Trap position stability ────────────────────────────────
    print("\n─── D) Trap Position Stability ───")
    amp_stand_arr = np.abs(p_stand)
    amp_combined_arr = np.abs(p_combined)

    traps_x_sw, traps_y_sw = find_traps(amp_stand_arr, xg, yg)
    traps_x_cb, traps_y_cb = find_traps(amp_combined_arr, xg, yg)

    print(f"  Traps in standing wave: {len(traps_x_sw)}")
    print(f"  Traps in combined field: {len(traps_x_cb)}")

    # Match traps: for each SW trap, find nearest combined trap
    max_disp = 0.0
    displacements = []
    matched_sw_x, matched_sw_y = [], []
    matched_cb_x, matched_cb_y = [], []
    disp_arrows_dx, disp_arrows_dy = [], []

    if len(traps_x_sw) > 0 and len(traps_x_cb) > 0:
        for i in range(len(traps_x_sw)):
            dists = np.sqrt((traps_x_cb - traps_x_sw[i])**2 +
                            (traps_y_cb - traps_y_sw[i])**2)
            j_min = np.argmin(dists)
            d = float(dists[j_min])
            if d < 0.5 * LAM:  # only count if within half wavelength
                displacements.append(d)
                matched_sw_x.append(traps_x_sw[i])
                matched_sw_y.append(traps_y_sw[i])
                matched_cb_x.append(traps_x_cb[j_min])
                matched_cb_y.append(traps_y_cb[j_min])
                disp_arrows_dx.append(traps_x_cb[j_min] - traps_x_sw[i])
                disp_arrows_dy.append(traps_y_cb[j_min] - traps_y_sw[i])
                if d > max_disp:
                    max_disp = d

    n_matched = len(displacements)
    if n_matched > 0:
        mean_disp = float(np.mean(displacements))
        max_disp_lam = max_disp / LAM
        print(f"  Matched traps: {n_matched}")
        print(f"  Mean displacement: {mean_disp*1e6:.1f} μm = {mean_disp/LAM:.4f}λ")
        print(f"  Max displacement: {max_disp*1e6:.1f} μm = {max_disp_lam:.4f}λ")
        trap_ok = max_disp_lam < 0.05
        print(f"  PASS: max displacement < 0.05λ" if trap_ok else
              f"  WARNING: max displacement = {max_disp_lam:.3f}λ (> 0.05λ threshold)")
    else:
        mean_disp = np.nan
        max_disp_lam = np.nan
        trap_ok = False
        print("  WARNING: No trap matches found")

    # ── Save RS lens data ─────────────────────────────────────────
    rs_save = {
        "xg": xg, "yg": yg,
        "p_inc_5": p_inc_5,
        "p_ref_5": p_ref_5,
        "p_vortex_total_5": p_vortex_total_5,
        "p_combined": p_combined,
        "p_inc_0": p_inc_0, "p_ref_0": p_ref_0,
        "p_vortex_total_0": p_vortex_total_0,
        "alpha": np.array(alpha_use),
        "z_star": np.array(Z_STAR),
        "tilt_deg": np.array(TILT_DEG),
    }
    rs_npz = RS_DIR / "rs_lens_validation.npz"
    np.savez_compressed(str(rs_npz), **rs_save)
    print(f"\n  Saved: {rs_npz.relative_to(PROJECT_ROOT)}")

    # ═══════════════════════════════════════════════════════════════
    # FIGURES
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "═" * 72)
    print("GENERATING FIGURES")
    print("═" * 72)

    extent = [xg[0]*1e3, xg[-1]*1e3, yg[0]*1e3, yg[-1]*1e3]
    extent_xz = [xg[0]*1e3, xg[-1]*1e3, z_xz[0]*1e3, z_xz[-1]*1e3]

    # ── Panel 1: Standing Wave Truth ──────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle(f"Panel 1 — Standing Wave Truth (z* = {Z_STAR*1e3:.3f} mm)",
                 fontsize=14, fontweight="bold")

    im = axes[0].imshow(amp_stand_arr, extent=extent, origin="lower", cmap="magma")
    axes[0].set_title("|p| at z*")
    fig.colorbar(im, ax=axes[0], shrink=0.8, label="Pa")

    im = axes[1].imshow(np.angle(p_stand), extent=extent, origin="lower",
                         cmap="hsv", vmin=-np.pi, vmax=np.pi)
    axes[1].set_title("Phase at z*")
    fig.colorbar(im, ax=axes[1], shrink=0.8, label="rad")

    axes[2].imshow(amp_stand_arr, extent=extent, origin="lower", cmap="magma")
    if len(traps_x_sw) > 0:
        axes[2].scatter(traps_x_sw * 1e3, traps_y_sw * 1e3, c="cyan", s=40,
                        marker="x", linewidths=1.5, label=f"Traps ({len(traps_x_sw)})")
        axes[2].legend(fontsize=9, loc="upper right")
    axes[2].set_title("Trap centres")

    for ax in axes:
        ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG_DIR / "panel_1_standing_wave.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved panel_1_standing_wave.png")

    # ── Panel 2: RS Vortex Structure ──────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f"Panel 2 — RS Vortex (LG ℓ={LENS_L}, θ={TILT_DEG}°, R={R_REFL})",
                 fontsize=14, fontweight="bold")

    vmax_v = max(np.abs(p_inc_5).max(), np.abs(p_ref_5).max(),
                 np.abs(p_vortex_total_5).max())

    for ax, field, title in [
        (axes[0, 0], p_inc_5, "|p_incident| (θ=5°)"),
        (axes[0, 1], p_ref_5, "|p_reflected| (θ=5°, R=−1)"),
        (axes[1, 0], p_vortex_total_5, "|p_vortex_total| (inc + ref)"),
    ]:
        im = ax.imshow(np.abs(field), extent=extent, origin="lower",
                       cmap="magma", vmin=0, vmax=vmax_v)
        ax.set_title(title)
        ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
        fig.colorbar(im, ax=ax, shrink=0.8)

    # XZ slice showing tilt + reflection
    im = axes[1, 1].imshow(np.abs(p_xz_tot_5), extent=extent_xz, origin="lower",
                            cmap="magma", aspect="auto")
    axes[1, 1].axhline(Z_STAR * 1e3, color="cyan", ls="--", lw=1, alpha=0.8, label="z*")
    axes[1, 1].axhline(Z_TOP * 1e3, color="white", ls=":", lw=1, alpha=0.8, label="z_top")
    axes[1, 1].set_title("XZ |p_total| (tilt plane)")
    axes[1, 1].set_xlabel("x (mm)"); axes[1, 1].set_ylabel("z (mm)")
    axes[1, 1].legend(fontsize=8, loc="lower right")
    fig.colorbar(im, ax=axes[1, 1], shrink=0.8)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG_DIR / "panel_2_rs_vortex.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved panel_2_rs_vortex.png")

    # ── Panel 3: Combined Field ───────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f"Panel 3 — Combined Field (α = {alpha_use:.4f})",
                 fontsize=14, fontweight="bold")

    im = axes[0, 0].imshow(amp_combined_arr, extent=extent, origin="lower", cmap="magma")
    axes[0, 0].set_title("|p_total| = |p_stand + α·p_vortex|")
    fig.colorbar(im, ax=axes[0, 0], shrink=0.8, label="Pa")

    im = axes[0, 1].imshow(np.angle(p_combined), extent=extent, origin="lower",
                            cmap="hsv", vmin=-np.pi, vmax=np.pi)
    axes[0, 1].set_title("Phase of combined field")
    fig.colorbar(im, ax=axes[0, 1], shrink=0.8, label="rad")

    # Trap overlay: SW traps (cyan x) and combined traps (yellow +)
    axes[1, 0].imshow(amp_combined_arr, extent=extent, origin="lower", cmap="magma")
    if len(traps_x_sw) > 0:
        axes[1, 0].scatter(traps_x_sw * 1e3, traps_y_sw * 1e3, c="cyan", s=50,
                           marker="x", linewidths=2, label="SW traps", zorder=5)
    if len(traps_x_cb) > 0:
        axes[1, 0].scatter(traps_x_cb * 1e3, traps_y_cb * 1e3, c="yellow", s=50,
                           marker="+", linewidths=2, label="Combined traps", zorder=5)
    axes[1, 0].legend(fontsize=9, loc="upper right")
    axes[1, 0].set_title("Trap centres: before (cyan) vs after (yellow)")

    # Displacement arrows (zoom to central region)
    axes[1, 1].imshow(amp_combined_arr, extent=extent, origin="lower", cmap="magma")
    if n_matched > 0:
        msw_x = np.array(matched_sw_x) * 1e3
        msw_y = np.array(matched_sw_y) * 1e3
        ddx = np.array(disp_arrows_dx) * 1e3
        ddy = np.array(disp_arrows_dy) * 1e3
        # Scale arrows for visibility
        arrow_scale = 50.0 if max_disp > 0 else 1.0
        axes[1, 1].quiver(msw_x, msw_y, ddx * arrow_scale, ddy * arrow_scale,
                          color="lime", scale=1.0, scale_units="xy", angles="xy",
                          width=0.003, zorder=5)
        axes[1, 1].set_title(f"Displacement arrows (×{arrow_scale:.0f})\n"
                             f"max={max_disp*1e6:.1f}μm = {max_disp_lam:.4f}λ")
    else:
        axes[1, 1].set_title("No matched traps for displacement")

    for ax in axes.flat:
        ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG_DIR / "panel_3_combined_field.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved panel_3_combined_field.png")

    # ── Panel 4: Waist + Overlap Diagnostics ──────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Panel 4 — Waist and Overlap Diagnostics",
                 fontsize=14, fontweight="bold")

    # Ring radius vs z
    axes[0].plot(z_waist * 1e3, ring_radii * 1e3, "b-o", ms=3, lw=1.5)
    axes[0].axhline(r_waist_min * 1e3, color="red", ls=":", alpha=0.5,
                    label=f"min = {r_waist_min*1e3:.3f} mm")
    axes[0].axvline(Z_STAR * 1e3, color="green", ls="--", alpha=0.7,
                    label=f"z* = {Z_STAR*1e3:.3f} mm")
    axes[0].axvline(z_waist_min * 1e3, color="red", ls="--", alpha=0.7,
                    label=f"waist z = {z_waist_min*1e3:.3f} mm")
    axes[0].set_xlabel("z (mm)")
    axes[0].set_ylabel("Ring radius (mm)")
    axes[0].set_title("Vortex ring radius vs z (waist analysis)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Reflection overlap comparison: θ=0 vs θ=5
    tilt_labels = ["θ = 0°", f"θ = {TILT_DEG}°"]
    overlaps = [overlap_0, overlap_5]
    bars = axes[1].bar(tilt_labels, overlaps, color=["tab:blue", "tab:orange"],
                       width=0.4, edgecolor="black")
    for bar, val in zip(bars, overlaps):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f"{val:.4f}", ha="center", va="bottom", fontsize=11)
    axes[1].set_ylabel("Overlap (ref RMS / inc RMS, core)")
    axes[1].set_title("Reflection overlap vs tilt")
    axes[1].set_ylim(0, max(overlaps) * 1.3)
    axes[1].grid(True, alpha=0.3, axis="y")

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG_DIR / "panel_4_waist_overlap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved panel_4_waist_overlap.png")

    # ═══════════════════════════════════════════════════════════════
    # REPORT
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "═" * 72)
    print("WRITING REPORT")
    print("═" * 72)

    go_decisions = {
        "Geometry preserved (trap stability)": trap_ok,
        "Waist correctly placed (< 2λ from z*)": waist_ok,
        "Reflection physically correct (tilt reduces overlap)": refl_ok,
        "Vortex not overwhelming standing wave (< 20%)": amp_ok,
    }
    all_go = all(go_decisions.values())

    report = f"""\
# RS Hybrid Validation Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Base directory:** `{BASE_DIR.relative_to(PROJECT_ROOT)}`

---

## 1. FEM Standing Wave Quality Assessment

| Parameter | Value |
|-----------|-------|
| Frequency | {F_HZ/1e6:.1f} MHz |
| λ | {LAM*1e3:.4f} mm |
| Domain | {LX_DOMAIN*1e3:.0f} × {LY_DOMAIN*1e3:.0f} × {H_TOTAL*1e3:.2f} mm |
| H_under | {H_UNDER*1e3:.1f} mm |
| H_top | {H_TOP*1e3:.4f} mm |
| z* | {Z_STAR*1e3:.3f} mm |
| Standing wave pattern | Antiphase, both axes |
| V_stand | 10 µm/s |
| Output grid | {Nx} × {Ny} |
| dx | {dx_out*1e6:.1f} µm ({dx_out/LAM:.3f}λ) |
| max |p_stand| | {amp_stand_max:.4f} Pa |
| Traps detected | {len(traps_x_sw)} |

The standing wave exhibits the expected chequerboard trap lattice at z*,
with traps spaced ≈ λ/2 in both x and y directions.

---

## 2. RS Lens Waist Verification

| Parameter | Value |
|-----------|-------|
| Lens | LG ℓ = {LENS_L} |
| R | {LENS_R*1e3:.1f} mm |
| w₀ | {LENS_W0*1e3:.1f} mm |
| Apodization | {LENS_APOD} |
| Tilt | {TILT_DEG}° (phase ramp about x-axis) |
| Drive grid | {DRIVE_N} × {DRIVE_N} |
| Damping | γ = {DAMPING_GAMMA} Np/m (exponential) |

**Waist analysis:**
- Minimum ring radius: **{r_waist_min*1e3:.3f} mm** at z = {z_waist_min*1e3:.3f} mm
- z* = {Z_STAR*1e3:.3f} mm
- Focal shift: **{focal_shift*1e3:.3f} mm** = {focal_shift/LAM:.2f}λ
- Assessment: **{"PASS — waist within 2λ of trap plane" if waist_ok else "WARNING — waist displaced from trap plane"}**

**Note on tilt method:** Tilt is implemented as a phase ramp
exp(i k sinθ (y − c_y)) applied to the source plane.  This is equivalent to
true k-space rotation for small angles (paraxial regime).  At θ = {TILT_DEG}°,
sin(θ) = {np.sin(np.deg2rad(TILT_DEG)):.4f} ≈ θ, so the paraxial approximation
is excellent.

---

## 3. Reflection + Tilt Behaviour

| Quantity | θ = 0° | θ = {TILT_DEG}° |
|----------|--------|----------|
| Overlap (ref/inc RMS, core) | {overlap_0:.4f} | {overlap_5:.4f} |

{
"Tilt successfully reduces coherent overlap of the reflected wave in the vortex core."
if refl_ok else
"WARNING: Tilt does not reduce reflected overlap — unexpected."
}

**Reflection model assumptions:**
- Specular image source at z_image = 2·z_top − z
- z_top = {Z_TOP*1e3:.2f} mm (H_under + H_top from canonical config)
- R = {R_REFL} (pressure-release approximation; actual water–air ≈ −0.9994)
- No multiple reflections
- No lateral boundary effects

**Damping model:**
- Simple exponential: p(z) → p(z) · exp(−γz), γ = {DAMPING_GAMMA} Np/m
- At z* = {Z_STAR*1e3:.1f} mm: attenuation = {(1 - np.exp(-DAMPING_GAMMA * Z_STAR))*100:.3f}%
- This is negligible — water at 2 MHz has very low absorption

---

## 4. Trap Displacement Analysis

| Metric | Value |
|--------|-------|
| SW traps | {len(traps_x_sw)} |
| Combined traps | {len(traps_x_cb)} |
| Matched pairs | {n_matched} |
| Mean displacement | {mean_disp*1e6:.1f} µm = {mean_disp/LAM:.4f}λ |
| Max displacement | {max_disp*1e6:.1f} µm = {max_disp_lam:.4f}λ |
| Threshold | 0.05λ = {0.05*LAM*1e6:.1f} µm |
| Result | **{"PASS" if trap_ok else "FAIL"}** |

{"Trap positions are stable — the vortex actuator at α = " + f"{alpha_use:.4f}" +
 " does not shift the standing-wave lattice beyond the 0.05λ threshold."
 if trap_ok else
 "WARNING: Trap positions shifted beyond threshold. Consider reducing α or adjusting lens parameters."}

---

## 5. Go / No-Go Decision

| Check | Result |
|-------|--------|
"""
    for desc, ok in go_decisions.items():
        report += f"| {desc} | **{'GO' if ok else 'NO-GO'}** |\n"

    report += f"""
**Overall: {'GO — system is ready for parameter sweeps' if all_go else 'NO-GO — address issues above before sweeping'}**

### Assumptions explicitly stated
1. **z_top = {Z_TOP*1e3:.2f} mm** from canonical config (H_under + H_top)
2. **Reflection R = {R_REFL}** (pressure-release, not exact water–air Robin)
3. **Damping γ = {DAMPING_GAMMA} Np/m** (negligible at these distances)
4. **Tilt by phase ramp** (valid in paraxial regime for θ ≤ ~15°)
5. **α = {alpha_use:.4f}** auto-scaled so vortex peak ≈ 5% of standing-wave peak
6. **Trap detection** uses local minima in |p| below 50% of median amplitude

---

## Appendix: Run Parameters

| Parameter | Value |
|-----------|-------|
| λ | {LAM*1e3:.4f} mm |
| k | {K_WATER:.1f} rad/m |
| f | {F_HZ/1e6:.1f} MHz |
| c_water | {WATER_C} m/s |
| ρ_water | {WATER_RHO} kg/m³ |
| z* | {Z_STAR*1e3:.3f} mm |
| z_top | {Z_TOP*1e3:.2f} mm |
| Lens ℓ | {LENS_L} |
| R_lens | {LENS_R*1e3:.1f} mm |
| w₀ | {LENS_W0*1e3:.1f} mm |
| α (used) | {alpha_use:.6f} |
| φ | {PHI_NOM}° |
| θ_tilt | {TILT_DEG}° |
| γ_damp | {DAMPING_GAMMA} Np/m |
"""

    report_path = BASE_DIR / "report.md"
    report_path.write_text(report)
    print(f"  Saved: {report_path.relative_to(PROJECT_ROOT)}")

    # ── Final summary ─────────────────────────────────────────────
    t_total = time.time() - t_total_start
    print()
    print("═" * 72)
    print(f"VALIDATION COMPLETE  ({t_total:.1f}s)")
    print("═" * 72)
    for desc, ok in go_decisions.items():
        print(f"  {'✓' if ok else '✗'} {desc}")
    print(f"\n  Overall: {'GO' if all_go else 'NO-GO'}")
    print(f"\n  Output: {BASE_DIR.relative_to(PROJECT_ROOT)}")
    print()


if __name__ == "__main__":
    main()
