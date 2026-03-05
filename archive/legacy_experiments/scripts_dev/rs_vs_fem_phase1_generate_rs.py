#!/usr/bin/env python3
"""
RS vs FEM Phase 1 — Generate Rayleigh–Sommerfeld / angular-spectrum vortex fields
==================================================================================

For each of the 6 selected actuator configs, builds the disk-plane complex
drive D(x,y) using the repo's lens functions, then propagates to each z-plane
via angular-spectrum propagation in water.

Two RS variants:
  A) RS_free   — no reflections (incident field in unbounded water)
  B) RS_reflect — first-order water/air reflection via image method

Outputs
-------
results/rs_vs_fem_phase1_YYYYMMDD_HHMMSS/rs_free/{config_id}.npz
results/rs_vs_fem_phase1_YYYYMMDD_HHMMSS/rs_reflect/{config_id}.npz
  Keys: xg, yg, z_list, p_xy_0 … p_xy_4  (complex128 each)

Usage
-----
    python scripts/dev/rs_vs_fem_phase1_generate_rs.py

    (does NOT need acousto-complex env — pure NumPy/SciPy)
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# ── paths ─────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from acoustweezers.experiments.farfield_petri_cuboid.presets import CORRECTED_PRESET
from acoustweezers.physics.acoustics.vortex_lens import (
    BGBeamConfig,
    LGBeamConfig,
    PlasticLensConfig,
    create_bg_drive,
    create_lg_drive,
    create_plastic_lens_drive,
)

# ── Shared output directory ──────────────────────────────────────
_TS_ENV = os.environ.get("RS_VS_FEM_PHASE1_TS")
TIMESTAMP = _TS_ENV if _TS_ENV else datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = PROJECT_ROOT / "results" / f"rs_vs_fem_phase1_{TIMESTAMP}"
RS_FREE_DIR = OUT_DIR / "rs_free"
RS_REFLECT_DIR = OUT_DIR / "rs_reflect"

# ── Physical constants ────────────────────────────────────────────
WATER_RHO = 997.0
WATER_C   = 1484.0
F_HZ      = 2.0e6
OMEGA     = 2 * np.pi * F_HZ
LAM       = WATER_C / F_HZ                       # 0.742 mm
K_WATER   = OMEGA / WATER_C                      # 8467.9 rad/m

H_UNDER   = CORRECTED_PRESET["H_under"]          # 3.0e-3
H_TOP     = CORRECTED_PRESET.get("H_top", 2.0085e-3)
H_TOTAL   = H_UNDER + H_TOP                      # 5.0085e-3
Z_MID     = H_UNDER + H_TOP / 2
Z_STAR    = Z_MID + 0.25 * LAM                   # ≈ 4.190 mm

CX, CY    = 3.0e-3, 3.0e-3                       # domain center
LX, LY    = 6.0e-3, 6.0e-3                       # domain size

# PML thickness (to match FEM physical grid extent)
PML_XY    = CORRECTED_PRESET.get("pml_n_wavelengths_xy", 1.0) * LAM
T_PML_XY  = PML_XY  # from preset

# ── Grid settings ─────────────────────────────────────────────────
GRID_N    = 200         # match FEM post-grid
# Drive source grid — much finer for accurate angular-spectrum
DRIVE_N   = 512         # source plane grid resolution

# Top-boundary Robin impedance for reflection coefficient
RHO_AIR   = 1.2
C_AIR     = 343.0
Z_AIR     = RHO_AIR * C_AIR                      # 411.6  Pa·s/m
Z_WATER   = WATER_RHO * WATER_C                  # 1.481e6 Pa·s/m

# Reflection coefficient at water/air interface (normal incidence)
# R = (Z_air - Z_water) / (Z_air + Z_water)  ≈ -1
R_COEFF   = (Z_AIR - Z_WATER) / (Z_AIR + Z_WATER)  # ≈ -0.9994

# ── Z-planes ─────────────────────────────────────────────────────
Z_PLANES = [1.0e-3, 1.5e-3, 2.0e-3, 2.5e-3, Z_STAR]

# ==================================================================
# 6 selected actuator configs (same as FEM script)
# ==================================================================
SELECTED_CONFIGS: List[Dict[str, Any]] = [
    {
        "config_id": "bg_l2_R1.0_w0.8",
        "family": "bg",
        "ell": 2,
        "aperture_radius_mm": 1.0,
        "beam_waist_mm": 0.8,
        "k_r": 0.5 * K_WATER,
    },
    {
        "config_id": "bg_l2_R1.0_w0.4",
        "family": "bg",
        "ell": 2,
        "aperture_radius_mm": 1.0,
        "beam_waist_mm": 0.4,
        "k_r": 0.5 * K_WATER,
    },
    {
        "config_id": "lg_l2_R1.0_w0.8",
        "family": "lg",
        "ell": 2,
        "aperture_radius_mm": 1.0,
        "beam_waist_mm": 0.8,
    },
    {
        "config_id": "lg_l2_R1.5_w1.2",
        "family": "lg",
        "ell": 2,
        "aperture_radius_mm": 1.5,
        "beam_waist_mm": 1.2,
    },
    {
        "config_id": "plastic_l2_R0.8_f1.5_wrapped",
        "family": "plastic",
        "ell": 2,
        "aperture_radius_mm": 0.8,
        "focal_length_mm": 1.5,
        "phase_mode": "wrapped",
    },
    {
        "config_id": "plastic_l2_R0.8_f3.5_wrapped",
        "family": "plastic",
        "ell": 2,
        "aperture_radius_mm": 0.8,
        "focal_length_mm": 3.5,
        "phase_mode": "wrapped",
    },
]


# ==================================================================
# Angular-spectrum propagation
# ==================================================================
def angular_spectrum_propagate(
    D: np.ndarray,
    dx: float,
    dy: float,
    z: float,
    k: float,
) -> np.ndarray:
    """
    Propagate complex field D(x,y) from z=0 to height z using
    angular-spectrum method.

    Parameters
    ----------
    D : (Ny, Nx) complex array — source field
    dx, dy : grid spacing [m]
    z : propagation distance [m] (positive = upward)
    k : total wavenumber in medium [rad/m]

    Returns
    -------
    p : (Ny, Nx) complex array — propagated field
    """
    Ny, Nx = D.shape
    # Spatial frequencies
    fx = np.fft.fftfreq(Nx, d=dx)   # cycles/m
    fy = np.fft.fftfreq(Ny, d=dy)
    FX, FY = np.meshgrid(fx, fy)

    # Transverse wavenumber squared
    kx = 2 * np.pi * FX
    ky = 2 * np.pi * FY
    kt2 = kx**2 + ky**2

    # Axial wavenumber kz (propagating + evanescent)
    kz2 = k**2 - kt2
    propagating = kz2 >= 0
    kz = np.zeros_like(kz2)
    kz[propagating] = np.sqrt(kz2[propagating])
    # Evanescent waves: decay exponentially
    kz[~propagating] = 1j * np.sqrt(-kz2[~propagating])

    # Transfer function
    H = np.exp(1j * kz * z)

    # Propagate
    D_fft = np.fft.fft2(D)
    p = np.fft.ifft2(D_fft * H)

    return p


def build_drive_on_grid(
    cfg: Dict,
    xg: np.ndarray,
    yg: np.ndarray,
    center_x: float = CX,
    center_y: float = CY,
) -> np.ndarray:
    """
    Build the disk-plane complex drive D(x,y) on a regular grid using
    the repo's lens functions.

    The lens functions expect 1-D coordinate arrays of DOF positions.
    We flatten the meshgrid, call the function, and reshape.

    Returns
    -------
    D : (Ny, Nx) complex array
    """
    XX, YY = np.meshgrid(xg, yg)
    coords_x = XX.ravel()
    coords_y = YY.ravel()

    fam = cfg["family"]
    ell = cfg["ell"]
    R = cfg["aperture_radius_mm"] * 1e-3

    if fam == "bg":
        bg_cfg = BGBeamConfig(
            topological_charge=ell,
            k_r=cfg.get("k_r", 0.5 * K_WATER),
            beam_waist=cfg["beam_waist_mm"] * 1e-3,
            c_water=WATER_C,
            frequency_hz=F_HZ,
            aperture_radius=R,
            apodization="cosine_taper",
        )
        pattern = create_bg_drive(coords_x, coords_y, bg_cfg,
                                  center_x=center_x, center_y=center_y,
                                  verbose=False)

    elif fam == "lg":
        lg_cfg = LGBeamConfig(
            topological_charge=ell,
            beam_waist=cfg["beam_waist_mm"] * 1e-3,
            focal_length=None,
            focus_offset_x=0.0,
            focus_offset_y=0.0,
            c_water=WATER_C,
            frequency_hz=F_HZ,
            aperture_radius=R,
            apodization="none",  # LG envelope already tapers
        )
        pattern = create_lg_drive(coords_x, coords_y, lg_cfg,
                                  center_x=center_x, center_y=center_y,
                                  verbose=False)

    elif fam == "plastic":
        pl_cfg = PlasticLensConfig(
            topological_charge=ell,
            focal_length=cfg["focal_length_mm"] * 1e-3,
            focus_offset_x=0.0,
            focus_offset_y=0.0,
            c_lens=2700.0,
            c_water=WATER_C,
            frequency_hz=F_HZ,
            aperture_radius=R,
            apodization="cosine_taper",
        )
        pattern = create_plastic_lens_drive(coords_x, coords_y, pl_cfg,
                                            center_x=center_x,
                                            center_y=center_y,
                                            verbose=False)
    else:
        raise ValueError(f"Unknown family: {fam}")

    D = pattern.reshape(XX.shape)
    return D


def propagate_config(
    cfg: Dict,
    z_planes: List[float],
    include_reflection: bool = False,
) -> Dict:
    """
    Build disk drive and propagate to all z-planes.

    The disk source sits at z=0 (bottom of domain).
    We propagate the source field to each z-plane.

    If include_reflection=True, add image source at z=2*H_TOTAL (top boundary
    reflected) with reflection coefficient R_COEFF.

    Returns dict ready to save as NPZ.
    """
    # Build drive on a fine grid covering the full domain extent
    # (matching FEM physical extent: from t_pml_xy to Lx-t_pml_xy)
    xg_drive = np.linspace(T_PML_XY, LX - T_PML_XY, DRIVE_N)
    yg_drive = np.linspace(T_PML_XY, LY - T_PML_XY, DRIVE_N)
    dx = xg_drive[1] - xg_drive[0]
    dy = yg_drive[1] - yg_drive[0]

    D = build_drive_on_grid(cfg, xg_drive, yg_drive)

    # The FEM solver applies: v_n = V₀ · pattern  →  p-BC via Neumann
    # For RS comparison we need the source field amplitude, not the velocity.
    # The FEM converts velocity BC to pressure via impedance.
    # For angular-spectrum propagation, the disk acts as a planar source
    # of pressure p₀ = ρ·c·v₀·pattern (Rayleigh piston approximation).
    # Since we compare shapes (not absolute amplitudes), we just propagate
    # the pattern directly. The comparison script will compensate via best-fit scale.

    # Also build the output grid matching FEM (GRID_N points in physical region)
    xg_out = np.linspace(T_PML_XY, LX - T_PML_XY, GRID_N)
    yg_out = np.linspace(T_PML_XY, LY - T_PML_XY, GRID_N)

    save_dict = {
        "xg": xg_out,
        "yg": yg_out,
        "z_list": np.array(z_planes),
    }

    for zi, z in enumerate(z_planes):
        # Direct propagation (source at z=0)
        p_direct = angular_spectrum_propagate(D, dx, dy, z, K_WATER)

        if include_reflection:
            # Image source: reflected at z = H_TOTAL (top boundary)
            # Image is at z_img = 2*H_TOTAL, propagating downward
            # At observation plane z, the distance from image is 2*H_TOTAL - z
            z_img_dist = 2 * H_TOTAL - z
            p_image = angular_spectrum_propagate(D, dx, dy, z_img_dist, K_WATER)
            p_total = p_direct + R_COEFF * p_image
        else:
            p_total = p_direct

        # Interpolate from drive grid to output grid
        if DRIVE_N != GRID_N:
            from scipy.interpolate import RegularGridInterpolator
            interp_re = RegularGridInterpolator(
                (yg_drive, xg_drive), np.real(p_total),
                method="linear", bounds_error=False, fill_value=0.0)
            interp_im = RegularGridInterpolator(
                (yg_drive, xg_drive), np.imag(p_total),
                method="linear", bounds_error=False, fill_value=0.0)
            XX_out, YY_out = np.meshgrid(xg_out, yg_out)
            pts = np.column_stack([YY_out.ravel(), XX_out.ravel()])
            p_out = (interp_re(pts) + 1j * interp_im(pts)).reshape(XX_out.shape)
        else:
            p_out = p_total

        save_dict[f"p_xy_{zi}"] = p_out

    return save_dict


def main():
    print("=" * 70)
    print("RS vs FEM Phase 1 — Angular-spectrum vortex propagation")
    print("=" * 70)
    print(f"Output : {OUT_DIR}")
    print(f"Configs: {len(SELECTED_CONFIGS)}")
    print(f"Z-planes: {len(Z_PLANES)}  ({', '.join(f'{z*1e3:.2f}' for z in Z_PLANES)} mm)")
    print(f"Drive grid: {DRIVE_N}×{DRIVE_N}  Output grid: {GRID_N}×{GRID_N}")
    print(f"λ={LAM*1e3:.3f} mm  k={K_WATER:.1f} rad/m")
    print(f"Z_STAR={Z_STAR*1e3:.4f} mm  H_total={H_TOTAL*1e3:.4f} mm")
    print(f"R_coeff={R_COEFF:.6f}")
    print()

    RS_FREE_DIR.mkdir(parents=True, exist_ok=True)
    RS_REFLECT_DIR.mkdir(parents=True, exist_ok=True)

    t0_all = time.time()
    results_info = []

    for i, cfg in enumerate(SELECTED_CONFIGS, 1):
        cid = cfg["config_id"]
        fam = cfg["family"]
        print(f"\n[{i}/{len(SELECTED_CONFIGS)}] {cid}  ({fam})")

        # RS_free
        t0 = time.time()
        data_free = propagate_config(cfg, Z_PLANES, include_reflection=False)
        t_free = time.time() - t0
        out_free = RS_FREE_DIR / f"{cid}.npz"
        np.savez_compressed(str(out_free), **data_free)
        print(f"  RS_free    : {t_free:.2f}s  → {out_free.relative_to(PROJECT_ROOT)}")

        # RS_reflect
        t0 = time.time()
        data_reflect = propagate_config(cfg, Z_PLANES, include_reflection=True)
        t_reflect = time.time() - t0
        out_reflect = RS_REFLECT_DIR / f"{cid}.npz"
        np.savez_compressed(str(out_reflect), **data_reflect)
        print(f"  RS_reflect : {t_reflect:.2f}s  → {out_reflect.relative_to(PROJECT_ROOT)}")

        results_info.append({
            "config_id": cid,
            "time_free_s": t_free,
            "time_reflect_s": t_reflect,
        })

    total_time = time.time() - t0_all

    # Write manifest
    manifest = OUT_DIR / "rs_manifest.json"
    import json as _json
    with open(manifest, "w") as f:
        _json.dump({
            "timestamp": TIMESTAMP,
            "drive_grid_n": DRIVE_N,
            "output_grid_n": GRID_N,
            "z_planes_mm": [z * 1e3 for z in Z_PLANES],
            "lambda_mm": LAM * 1e3,
            "z_star_mm": Z_STAR * 1e3,
            "h_total_mm": H_TOTAL * 1e3,
            "R_coeff": R_COEFF,
            "configs": results_info,
            "total_time_s": total_time,
        }, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"RS propagation complete in {total_time:.1f}s")
    print(f"Results: {OUT_DIR}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
