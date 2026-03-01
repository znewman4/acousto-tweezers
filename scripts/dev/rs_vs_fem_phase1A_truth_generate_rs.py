#!/usr/bin/env python3
"""
RS vs FEM Phase 1A — Generate RS truth (angular-spectrum, free-space)
=====================================================================

Builds the LG ℓ=2 disk-plane drive D(x,y) using the repo's ``create_lg_drive``
and propagates to each z-plane via angular-spectrum propagation in **unbounded
water** (RS_free only — no reflections, matching the homogeneous-water FEM truth).

Key fix vs Phase 1: uses ``apodization='cosine_taper'`` to exactly match
what the FEM solver applies (Phase 1 used ``'none'`` for LG).

Outputs
-------
results/rs_vs_fem_phase1A_truth_<TIMESTAMP>/
  rs_truth/lg_l2_R1.0_w0.8.npz  — xg, yg, z_list, p_xy_0..4 (complex)

Usage
-----
  micromamba run -n acousto-complex python \\
      scripts/dev/rs_vs_fem_phase1A_truth_generate_rs.py --timestamp $TS
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

# ── paths ─────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from acoustweezers.experiments.farfield_petri_cuboid.presets import (
    CORRECTED_PRESET,
)
from acoustweezers.physics.acoustics.vortex_lens import (
    LGBeamConfig,
    create_lg_drive,
)

# ── Physical constants ────────────────────────────────────────────
WATER_RHO = 997.0
WATER_C   = 1484.0
F_HZ      = 2.0e6
LAM       = WATER_C / F_HZ
K_WATER   = 2 * np.pi * F_HZ / WATER_C

# Operating plane (matches Phase 1 / 1A FEM)
_H_UNDER_ORIG = CORRECTED_PRESET["H_under"]
_H_TOP_ORIG   = CORRECTED_PRESET.get("H_top", 2.0085e-3)
_Z_MID_ORIG   = _H_UNDER_ORIG + _H_TOP_ORIG / 2
Z_STAR        = _Z_MID_ORIG + 0.25 * LAM

Z_PLANES = [1.0e-3, 1.5e-3, 2.0e-3, 2.5e-3, Z_STAR]

CONFIG_ID = "lg_l2_R1.0_w0.8"

# Drive grid resolution (power of 2 for efficient FFT)
DRIVE_N = 512


# ==================================================================
# Angular-spectrum propagation (with zero-padded FFT)
# ==================================================================
def angular_spectrum_propagate(
    D: np.ndarray,
    dx: float,
    dy: float,
    z: float,
    k: float,
    pad: bool = True,
) -> np.ndarray:
    """
    Propagate complex field D(x,y) from z=0 to height z via angular-spectrum.

    Parameters
    ----------
    D : (Ny, Nx) complex — source field
    dx, dy : grid spacing [m]
    z : propagation distance [m]
    k : wavenumber in medium [rad/m]
    pad : if True, zero-pad to 2× in each dimension to reduce wraparound

    Returns
    -------
    p : (Ny, Nx) complex — propagated field (same shape as D)
    """
    Ny, Nx = D.shape

    if pad:
        Ny2, Nx2 = 2 * Ny, 2 * Nx
        D_pad = np.zeros((Ny2, Nx2), dtype=complex)
        D_pad[:Ny, :Nx] = D
    else:
        Ny2, Nx2 = Ny, Nx
        D_pad = D

    fx = np.fft.fftfreq(Nx2, d=dx)
    fy = np.fft.fftfreq(Ny2, d=dy)
    FX, FY = np.meshgrid(fx, fy)

    kx = 2 * np.pi * FX
    ky = 2 * np.pi * FY
    kt2 = kx ** 2 + ky ** 2

    kz2 = k ** 2 - kt2
    propagating = kz2 >= 0
    kz = np.zeros_like(kz2, dtype=complex)
    kz[propagating] = np.sqrt(kz2[propagating])
    kz[~propagating] = 1j * np.sqrt(-kz2[~propagating])

    H = np.exp(1j * kz * z)

    D_fft = np.fft.fft2(D_pad)
    p_pad = np.fft.ifft2(D_fft * H)

    if pad:
        return p_pad[:Ny, :Nx]
    return p_pad


# ==================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Phase 1A — RS truth (free-space angular spectrum)"
    )
    p.add_argument(
        "--timestamp",
        default=datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    p.add_argument("--lx_mm", type=float, default=5.0,
                   help="Domain Lx in mm (must match FEM)")
    p.add_argument("--ly_mm", type=float, default=5.0,
                   help="Domain Ly in mm (must match FEM)")
    p.add_argument("--pml_mm", type=float, default=None,
                   help="PML thickness in mm (default: 1 λ)")
    p.add_argument("--grid_n", type=int, default=200,
                   help="Output grid resolution (must match FEM)")
    p.add_argument("--drive_n", type=int, default=DRIVE_N,
                   help="Source-plane grid resolution (default: 512)")
    return p.parse_args()


def main():
    args = parse_args()
    TS = args.timestamp

    OUT_DIR = PROJECT_ROOT / "results" / f"rs_vs_fem_phase1A_truth_{TS}"
    RS_DIR  = OUT_DIR / "rs_truth"
    RS_DIR.mkdir(parents=True, exist_ok=True)

    Lx      = args.lx_mm * 1e-3
    Ly      = args.ly_mm * 1e-3
    GRID_N  = args.grid_n
    DN      = args.drive_n
    pml_t   = (args.pml_mm * 1e-3) if args.pml_mm else 1.0 * LAM

    CX, CY = Lx / 2, Ly / 2

    print("=" * 70)
    print("RS vs FEM Phase 1A — RS truth (angular-spectrum, free-space)")
    print("=" * 70)
    print(f"Output     : {OUT_DIR}")
    print(f"Config     : {CONFIG_ID}")
    print(f"Domain     : {Lx*1e3:.1f} × {Ly*1e3:.1f} mm")
    print(f"Drive grid : {DN}×{DN}   Output grid: {GRID_N}×{GRID_N}")
    print(f"PML (FEM)  : {pml_t*1e3:.3f} mm")
    print(f"λ = {LAM*1e3:.4f} mm   k = {K_WATER:.1f} rad/m")
    z_str = ", ".join(f"{z*1e3:.2f}" for z in Z_PLANES)
    print(f"Z-planes   : [{z_str}] mm")
    print()

    # ── Build LG drive on fine grid covering full domain ──────────
    # Use full domain (0..Lx, 0..Ly) for natural zero-padding outside disk
    xg_drive = np.linspace(0, Lx, DN, endpoint=False)
    yg_drive = np.linspace(0, Ly, DN, endpoint=False)
    dx = xg_drive[1] - xg_drive[0]
    dy = yg_drive[1] - yg_drive[0]

    XX, YY = np.meshgrid(xg_drive, yg_drive)
    coords_x = XX.ravel()
    coords_y = YY.ravel()

    # LG ℓ=2 with cosine_taper (matches FEM solver exactly)
    lg_cfg = LGBeamConfig(
        topological_charge=2,
        beam_waist=0.8e-3,
        focal_length=None,
        focus_offset_x=0.0,
        focus_offset_y=0.0,
        c_water=WATER_C,
        frequency_hz=F_HZ,
        aperture_radius=1.0e-3,
        center=None,
        apodization="cosine_taper",   # matches FEM cfg.lens_apodization
        apodization_strength=1.0,
    )

    pattern = create_lg_drive(
        coords_x, coords_y, lg_cfg,
        center_x=CX, center_y=CY, verbose=True,
    )
    D = pattern.reshape(XX.shape)
    print(f"  Drive max |D| = {np.abs(D).max():.4f}")
    print(f"  Drive non-zero fraction: "
          f"{np.sum(np.abs(D)>1e-10)/D.size:.3f}")

    # ── Output grid matching FEM physical region ──────────────────
    xg_out = np.linspace(pml_t, Lx - pml_t, GRID_N)
    yg_out = np.linspace(pml_t, Ly - pml_t, GRID_N)

    # ── Propagate to each z-plane ─────────────────────────────────
    save_dict = {
        "xg": xg_out,
        "yg": yg_out,
        "z_list": np.array(Z_PLANES),
    }

    from scipy.interpolate import RegularGridInterpolator

    t0 = time.time()
    for zi, z in enumerate(Z_PLANES):
        t1 = time.time()
        p_full = angular_spectrum_propagate(D, dx, dy, z, K_WATER, pad=True)
        t_prop = time.time() - t1

        # Interpolate from drive grid to output grid
        interp_re = RegularGridInterpolator(
            (yg_drive, xg_drive), np.real(p_full),
            method="linear", bounds_error=False, fill_value=0.0,
        )
        interp_im = RegularGridInterpolator(
            (yg_drive, xg_drive), np.imag(p_full),
            method="linear", bounds_error=False, fill_value=0.0,
        )
        XX_out, YY_out = np.meshgrid(xg_out, yg_out)
        pts = np.column_stack([YY_out.ravel(), XX_out.ravel()])
        p_out = (interp_re(pts) + 1j * interp_im(pts)).reshape(XX_out.shape)

        save_dict[f"p_xy_{zi}"] = p_out
        amp = np.abs(p_out).max()
        print(f"  z={z*1e3:5.2f} mm  max|p|={amp:.6f}  ({t_prop:.2f}s)")

    total = time.time() - t0

    # ── Save ──────────────────────────────────────────────────────
    out_npz = RS_DIR / f"{CONFIG_ID}.npz"
    np.savez_compressed(str(out_npz), **save_dict)
    print(f"\n  NPZ: {out_npz.relative_to(PROJECT_ROOT)}")

    print(f"\n{'=' * 70}")
    print(f"RS truth generation complete in {total:.1f}s")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
