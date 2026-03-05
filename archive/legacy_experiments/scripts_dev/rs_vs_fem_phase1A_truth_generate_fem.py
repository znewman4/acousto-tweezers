#!/usr/bin/env python3
"""
RS vs FEM Phase 1A — Generate high-res FEM truth in homogeneous water
======================================================================

Runs ONE high-resolution FEM solve for LG ℓ=2 R=1.0 mm w=0.8 mm in a
reduced domain configured as **homogeneous water**:

  - H_under is tall enough to encompass all z-planes (no slab evaluation)
  - H_top is negligibly small (0.1 mm)
  - Top Robin BC uses matched impedance (ρ_air = ρ_water, c_air = c_water)
    → zero reflection coefficient
  - Standing-wave amplitude = 0 → no side-wall excitation in slab
  - Full lateral PML active throughout water bath

This removes slab physics, top reflection, and (with large enough domain)
minimises cavity contamination, giving a clean FEM reference for RS
angular-spectrum validation.

Outputs
-------
results/rs_vs_fem_phase1A_truth_<TIMESTAMP>/
  fem_truth/lg_l2_R1.0_w0.8.npz   — xg, yg, z_list, p_xy_0..4 (complex)
  fem_truth/manifest.json

Usage
-----
  export TS=$(date +%Y%m%d_%H%M%S)
  micromamba run -n acousto-complex python \\
      scripts/dev/rs_vs_fem_phase1A_truth_generate_fem.py --timestamp $TS
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np

# ── paths ─────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

WORKER_SCRIPT = (
    PROJECT_ROOT / "scripts" / "experiments" / "_solve_worker_multi_z.py"
)

from acoustweezers.experiments.farfield_petri_cuboid.presets import (
    CORRECTED_PRESET,
)

# ── Physical constants ────────────────────────────────────────────
WATER_RHO = 997.0
WATER_C   = 1484.0
F_HZ      = 2.0e6
LAM       = WATER_C / F_HZ                        # 0.742 mm
K_WATER   = 2 * np.pi * F_HZ / WATER_C

# Operating plane (matches Phase 1 exactly)
_H_UNDER_ORIG = CORRECTED_PRESET["H_under"]       # 3.0e-3
_H_TOP_ORIG   = CORRECTED_PRESET.get("H_top", 2.0085e-3)
_Z_MID_ORIG   = _H_UNDER_ORIG + _H_TOP_ORIG / 2
Z_STAR        = _Z_MID_ORIG + 0.25 * LAM          # ≈ 4.190 mm

# Same z-planes as Phase 1
Z_PLANES = [1.0e-3, 1.5e-3, 2.0e-3, 2.5e-3, Z_STAR]

CONFIG_ID = "lg_l2_R1.0_w0.8"


# ==================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Phase 1A — FEM truth in homogeneous water (LG ℓ=2)"
    )
    p.add_argument(
        "--timestamp",
        default=datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    p.add_argument("--epl", type=int, default=4,
                   help="Elements per wavelength (default: 4)")
    p.add_argument("--lx_mm", type=float, default=5.0,
                   help="Domain Lx in mm (default: 5.0)")
    p.add_argument("--ly_mm", type=float, default=5.0,
                   help="Domain Ly in mm (default: 5.0)")
    p.add_argument("--hz_mm", type=float, default=5.0,
                   help="Water bath height H_under in mm (default: 5.0)")
    p.add_argument("--pml_mm", type=float, default=None,
                   help="PML thickness in mm (default: 1 λ = 0.742)")
    p.add_argument("--grid_n", type=int, default=200,
                   help="Post-processing interpolation grid (default: 200)")
    p.add_argument("--core_roi_lam", type=float, default=2.0,
                   help="Core ROI radius in λ (default: 2.0)")
    return p.parse_args()


def main():
    args = parse_args()
    TS = args.timestamp

    OUT_DIR   = PROJECT_ROOT / "results" / f"rs_vs_fem_phase1A_truth_{TS}"
    FEM_DIR   = OUT_DIR / "fem_truth"
    CACHE_DIR = OUT_DIR / "_cache"

    FEM_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    Lx       = args.lx_mm * 1e-3
    Ly       = args.ly_mm * 1e-3
    H_under  = args.hz_mm * 1e-3
    H_top    = 0.1e-3                   # negligible slab
    EPL      = args.epl
    GRID_N   = args.grid_n
    pml_n    = (args.pml_mm * 1e-3 / LAM) if args.pml_mm else 1.0

    CX, CY = Lx / 2, Ly / 2

    print("=" * 70)
    print("RS vs FEM Phase 1A — FEM truth (homogeneous water)")
    print("=" * 70)
    print(f"Output   : {OUT_DIR}")
    print(f"Config   : {CONFIG_ID}")
    print(f"Domain   : {Lx*1e3:.1f} × {Ly*1e3:.1f} × {(H_under+H_top)*1e3:.1f} mm")
    print(f"H_under  : {H_under*1e3:.1f} mm   H_top: {H_top*1e3:.2f} mm")
    print(f"EPL      : {EPL}   Grid: {GRID_N}")
    print(f"PML      : {pml_n:.2f} λ = {pml_n*LAM*1e3:.3f} mm")
    print(f"λ        : {LAM*1e3:.4f} mm   k = {K_WATER:.1f} rad/m")
    print(f"Z_STAR   : {Z_STAR*1e3:.4f} mm  ({Z_STAR/LAM:.2f} λ)")
    z_str = ", ".join(f"{z*1e3:.2f}" for z in Z_PLANES)
    print(f"Z-planes : [{z_str}] mm")
    print()

    # Verify z-planes are inside the water bath
    for z in Z_PLANES:
        if z >= H_under:
            print(f"  WARNING: z={z*1e3:.2f} mm >= H_under={H_under*1e3:.1f} mm")
            print(f"  Increase --hz_mm to at least {z*1e3 + 1.0:.1f}")

    # ── Build overrides for homogeneous water ─────────────────────
    overrides = {
        **CORRECTED_PRESET,
        # Domain
        "Lx": Lx,
        "Ly": Ly,
        "H_under": H_under,
        "H_top": H_top,
        # Resolution
        "elements_per_wavelength": EPL,
        # LG ℓ=2 lens
        "lens_drive": "lg",
        "lens_l": 2,
        "lens_beam_waist": 0.8e-3,
        "lens_focal_length": 0.0,             # no focusing
        "lens_apodization": "cosine_taper",
        "lens_apodization_strength": 1.0,
        "lens_focus_offset_x": 0.0,
        "lens_focus_offset_y": 0.0,
        "disk_radius": 1.0e-3,
        "disk_velocity_amplitude": 1e-6,
        # No standing wave
        "standing_velocity_amplitude": 0.0,
        # Matched impedance top → R = 0 → no reflection
        "rho_air": WATER_RHO,
        "c_air": WATER_C,
        # PML
        "pml_n_wavelengths_xy": pml_n,
        "pml_n_wavelengths_z": pml_n,
    }

    # ── Solve via existing worker ─────────────────────────────────
    label     = f"truth_{CONFIG_ID}_epl{EPL}"
    npz_cache = str(CACHE_DIR / f"_grid_{label}.npz")

    solve_args = {
        "overrides": overrides,
        "label": label,
        "trap_z_list": Z_PLANES,
        "mid_y": CY,
        "n_xy": GRID_N,
        "result_file": npz_cache,
    }

    if os.path.exists(npz_cache):
        print(f"  Using cached solve: {npz_cache}")
    else:
        print(f"  Solving (EPL={EPL}, timeout=1200s) …")
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(solve_args, f, default=str)
            args_file = f.name
        try:
            t0 = time.time()
            proc = subprocess.run(
                [sys.executable, str(WORKER_SCRIPT), args_file],
                capture_output=False,
                timeout=1200,
            )
            elapsed = time.time() - t0
            if proc.returncode != 0:
                print(f"  *** SOLVE FAILED  rc={proc.returncode}  ({elapsed:.0f}s)")
                sys.exit(1)
            print(f"  Solve OK in {elapsed:.0f}s")
        finally:
            os.unlink(args_file)

    if not os.path.exists(npz_cache):
        print("  *** No NPZ produced — aborting.")
        sys.exit(1)

    # ── Re-package into clean output NPZ ──────────────────────────
    data = dict(np.load(npz_cache, allow_pickle=False))
    out_npz = FEM_DIR / f"{CONFIG_ID}.npz"

    save_dict = {
        "xg": data["xg"],
        "yg": data["yg"],
        "z_list": np.array(Z_PLANES),
    }
    n_ok = 0
    for zi in range(len(Z_PLANES)):
        key = f"p_xy_{zi}"
        if key in data:
            save_dict[key] = data[key]
            amp = np.abs(data[key]).max()
            print(f"  z={Z_PLANES[zi]*1e3:5.2f} mm  max|p|={amp:.4f}")
            n_ok += 1
        else:
            print(f"  z={Z_PLANES[zi]*1e3:5.2f} mm  MISSING")

    # Include XZ slice if available
    for k in ("xg_xz", "zg_xz", "p_xz"):
        if k in data:
            save_dict[k] = data[k]

    np.savez_compressed(str(out_npz), **save_dict)
    print(f"\n  NPZ: {out_npz.relative_to(PROJECT_ROOT)}  ({n_ok}/{len(Z_PLANES)} planes)")

    # ── Manifest ──────────────────────────────────────────────────
    t_pml = pml_n * LAM
    manifest = {
        "config_id": CONFIG_ID,
        "epl": EPL,
        "grid_n": GRID_N,
        "Lx_mm": Lx * 1e3,
        "Ly_mm": Ly * 1e3,
        "H_under_mm": H_under * 1e3,
        "H_top_mm": H_top * 1e3,
        "H_total_mm": (H_under + H_top) * 1e3,
        "pml_mm": t_pml * 1e3,
        "physical_x_mm": [(t_pml) * 1e3, (Lx - t_pml) * 1e3],
        "physical_y_mm": [(t_pml) * 1e3, (Ly - t_pml) * 1e3],
        "center_mm": [Lx / 2 * 1e3, Ly / 2 * 1e3],
        "disk_radius_mm": 1.0,
        "beam_waist_mm": 0.8,
        "ell": 2,
        "apodization": "cosine_taper",
        "lambda_mm": LAM * 1e3,
        "z_planes_mm": [z * 1e3 for z in Z_PLANES],
        "z_star_mm": Z_STAR * 1e3,
        "top_bc": "matched_impedance (rho_air=997, c_air=1484, R=0)",
        "n_z_planes_ok": n_ok,
        "solver_time_s": data.get("solve_time", np.array(0.0)).item()
        if "solve_time" in data
        else None,
    }
    mf = FEM_DIR / "manifest.json"
    with open(mf, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Manifest: {mf.relative_to(PROJECT_ROOT)}")

    print(f"\n{'=' * 70}")
    print(f"FEM truth generation complete.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
