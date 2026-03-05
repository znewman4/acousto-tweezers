#!/usr/bin/env python3
"""
RS vs FEM Phase 1 — Generate FEM vortex-only fields
=====================================================

For each of the 6 selected actuator configs, runs a vortex-only FEM solve
(standing amplitude = 0) and extracts complex pressure on XY grids at
5 z-planes.

Outputs
-------
results/rs_vs_fem_phase1_YYYYMMDD_HHMMSS/fem/{config_id}.npz
  Keys: xg, yg, z_list, p_xy_0 … p_xy_4  (complex128 each)
  Also: xg_xz, zg_xz, p_xz  (XZ slice through y=cy)

Usage
-----
    micromamba run -n acousto-complex python scripts/dev/rs_vs_fem_phase1_generate_fem.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# ── paths ─────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

WORKER_SCRIPT = PROJECT_ROOT / "scripts" / "experiments" / "_solve_worker_multi_z.py"

from acoustweezers.experiments.farfield_petri_cuboid.presets import CORRECTED_PRESET

# ── Shared output directory (created by whichever script runs first) ──
# Use environment variable if set (so all three scripts share a directory)
_TS_ENV = os.environ.get("RS_VS_FEM_PHASE1_TS")
TIMESTAMP = _TS_ENV if _TS_ENV else datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = PROJECT_ROOT / "results" / f"rs_vs_fem_phase1_{TIMESTAMP}"
FEM_DIR = OUT_DIR / "fem"
CACHE_DIR = OUT_DIR / "_cache"

# Previous caches to avoid re-solving
PREV_CACHES = [
    PROJECT_ROOT / "results" / "vortex_bridge_design_study_phase_20260227_141230" / "_cache",
]

# ── Physical constants ────────────────────────────────────────────
WATER_C  = 1484.0
F_HZ     = 2.0e6
LAM      = WATER_C / F_HZ                       # 0.742 mm
K_WATER  = 2 * np.pi * F_HZ / WATER_C           # 8467.9 rad/m

H_UNDER  = CORRECTED_PRESET["H_under"]           # 3.0e-3
H_TOP    = CORRECTED_PRESET.get("H_top", 2.0085e-3)
Z_MID    = H_UNDER + H_TOP / 2
Z_STAR   = Z_MID + 0.25 * LAM                   # ≈ 4.190 mm

CX, CY   = 3.0e-3, 3.0e-3                       # domain center (6 mm box)

# ── Grid / solve settings ────────────────────────────────────────
EPL      = 3            # baseline — EPL=4 needs smaller domain
GRID_N   = 200          # post-FEM interpolation grid

# ── Z-planes ─────────────────────────────────────────────────────
Z_PLANES = [1.0e-3, 1.5e-3, 2.0e-3, 2.5e-3, Z_STAR]

# ==================================================================
# 6 selected actuator configs
# ==================================================================
SELECTED_CONFIGS: List[Dict[str, Any]] = [
    # 1) BG moderate
    {
        "config_id": "bg_l2_R1.0_w0.8",
        "family": "bg",
        "ell": 2,
        "aperture_radius_mm": 1.0,
        "beam_waist_mm": 0.8,
        "k_r": 0.5 * K_WATER,
    },
    # 2) BG narrow/extreme
    {
        "config_id": "bg_l2_R1.0_w0.4",
        "family": "bg",
        "ell": 2,
        "aperture_radius_mm": 1.0,
        "beam_waist_mm": 0.4,
        "k_r": 0.5 * K_WATER,
    },
    # 3) LG moderate
    {
        "config_id": "lg_l2_R1.0_w0.8",
        "family": "lg",
        "ell": 2,
        "aperture_radius_mm": 1.0,
        "beam_waist_mm": 0.8,
    },
    # 4) LG wide
    {
        "config_id": "lg_l2_R1.5_w1.2",
        "family": "lg",
        "ell": 2,
        "aperture_radius_mm": 1.5,
        "beam_waist_mm": 1.2,
    },
    # 5) Plastic short f (wrapped is the standard mode)
    {
        "config_id": "plastic_l2_R0.8_f1.5_wrapped",
        "family": "plastic",
        "ell": 2,
        "aperture_radius_mm": 0.8,
        "focal_length_mm": 1.5,
        "phase_mode": "wrapped",
    },
    # 6) Plastic long f
    {
        "config_id": "plastic_l2_R0.8_f3.5_wrapped",
        "family": "plastic",
        "ell": 2,
        "aperture_radius_mm": 0.8,
        "focal_length_mm": 3.5,
        "phase_mode": "wrapped",
    },
]


def config_to_overrides(cfg: Dict) -> dict:
    """Convert selected config entry to FarFieldConfig overrides (vortex-only)."""
    ov = {**CORRECTED_PRESET, "elements_per_wavelength": EPL}
    # Vortex-only: turn off standing wave
    ov["standing_velocity_amplitude"] = 0.0
    ov["disk_velocity_amplitude"] = 1e-6

    fam = cfg["family"]
    ov["lens_l"] = cfg["ell"]
    ov["lens_focus_offset_x"] = 0.0
    ov["lens_focus_offset_y"] = 0.0
    ov["lens_apodization"] = "cosine_taper"
    ov["disk_radius"] = cfg["aperture_radius_mm"] * 1e-3

    if fam == "plastic":
        ov["lens_drive"] = "plastic"
        ov["lens_focal_length"] = cfg["focal_length_mm"] * 1e-3
    elif fam == "bg":
        ov["lens_drive"] = "bessel_gauss"
        ov["lens_k_r"] = cfg.get("k_r", 0.5 * K_WATER)
        ov["lens_beam_waist"] = cfg["beam_waist_mm"] * 1e-3
    elif fam == "lg":
        ov["lens_drive"] = "lg"
        ov["lens_beam_waist"] = cfg["beam_waist_mm"] * 1e-3
        ov["lens_focal_length"] = 0.0

    return ov


def _fem_solve_key(cfg: Dict) -> str:
    """Unique FEM cache key — includes '5z' suffix to differentiate from bridge study."""
    fam = cfg["family"]
    ell = cfg["ell"]
    R = cfg["aperture_radius_mm"]
    if fam == "plastic":
        return f"vortex_plastic_l{ell}_R{R}_f{cfg['focal_length_mm']}_5z"
    elif fam == "bg":
        return f"vortex_bg_l{ell}_R{R}_w{cfg['beam_waist_mm']}_5z"
    elif fam == "lg":
        return f"vortex_lg_l{ell}_R{R}_w{cfg['beam_waist_mm']}_5z"
    return f"vortex_{cfg['config_id']}_5z"


def solve_multi_z(overrides: dict, label: str,
                  z_list: List[float],
                  mid_y: float = CY,
                  n_xy: int = GRID_N) -> Optional[dict]:
    """Subprocess FEM solve -> .npz with xg,yg,p_xy_{i},xg_xz,zg_xz,p_xz."""
    result_file = str(CACHE_DIR / f"_grid_{label}.npz")

    # Check own cache
    if os.path.exists(result_file):
        return dict(np.load(result_file, allow_pickle=False))

    # Check previous-run caches
    for pc in PREV_CACHES:
        prev_file = pc / f"_grid_{label}.npz"
        if prev_file.exists():
            data = dict(np.load(str(prev_file), allow_pickle=False))
            np.savez(result_file, **data)
            return data

    # Fresh solve
    args = {
        "overrides": overrides,
        "label": label,
        "trap_z_list": z_list,
        "mid_y": mid_y,
        "n_xy": n_xy,
        "result_file": result_file,
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(args, f, default=str)
        args_file = f.name
    try:
        proc = subprocess.run(
            [sys.executable, str(WORKER_SCRIPT), args_file],
            capture_output=False, timeout=1200,
        )
        if proc.returncode != 0:
            print(f"  *** SOLVE FAILED: {label}  rc={proc.returncode}")
            return None
    finally:
        os.unlink(args_file)
    if not os.path.exists(result_file):
        return None
    return dict(np.load(result_file, allow_pickle=False))


def main():
    print("=" * 70)
    print("RS vs FEM Phase 1 — FEM vortex-only solves")
    print("=" * 70)
    print(f"Output : {OUT_DIR}")
    print(f"Configs: {len(SELECTED_CONFIGS)}")
    print(f"Z-planes: {len(Z_PLANES)}  ({', '.join(f'{z*1e3:.2f}' for z in Z_PLANES)} mm)")
    print(f"EPL={EPL}  Grid={GRID_N}  λ={LAM*1e3:.3f} mm")
    print(f"Z_STAR={Z_STAR*1e3:.4f} mm  ({Z_STAR/LAM:.2f} λ)")
    print()

    FEM_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    t0_all = time.time()
    results_info = []

    for i, cfg in enumerate(SELECTED_CONFIGS, 1):
        cid = cfg["config_id"]
        fam = cfg["family"]
        ov = config_to_overrides(cfg)
        label = _fem_solve_key(cfg)

        print(f"\n[{i}/{len(SELECTED_CONFIGS)}] {cid}")
        print(f"  family={fam}  ell={cfg['ell']}  R={cfg['aperture_radius_mm']} mm")
        t0 = time.time()

        data = solve_multi_z(ov, label, z_list=Z_PLANES, mid_y=CY, n_xy=GRID_N)

        elapsed = time.time() - t0
        if data is None:
            print(f"  *** FAILED after {elapsed:.1f}s")
            results_info.append({"config_id": cid, "status": "FAIL"})
            continue

        # Re-package into clean NPZ for the comparison script
        out_npz = FEM_DIR / f"{cid}.npz"
        save_dict = {
            "xg": data["xg"],
            "yg": data["yg"],
            "z_list": np.array(Z_PLANES),
        }
        for zi in range(len(Z_PLANES)):
            key = f"p_xy_{zi}"
            if key in data:
                save_dict[key] = data[key]
            else:
                print(f"  WARNING: missing {key}")

        # Also include XZ slice if available
        for k in ("xg_xz", "zg_xz", "p_xz"):
            if k in data:
                save_dict[k] = data[k]

        np.savez_compressed(str(out_npz), **save_dict)
        n_ok = sum(1 for zi in range(len(Z_PLANES)) if f"p_xy_{zi}" in data)
        print(f"  OK: {n_ok}/{len(Z_PLANES)} planes  ({elapsed:.1f}s)")
        print(f"  Saved: {out_npz.relative_to(PROJECT_ROOT)}")
        results_info.append({
            "config_id": cid, "status": "OK",
            "n_planes": n_ok, "time_s": elapsed,
        })

    total_time = time.time() - t0_all

    # Write a manifest
    manifest = FEM_DIR / "manifest.json"
    import json as _json
    with open(manifest, "w") as f:
        _json.dump({
            "timestamp": TIMESTAMP,
            "epl": EPL,
            "grid_n": GRID_N,
            "z_planes_mm": [z * 1e3 for z in Z_PLANES],
            "lambda_mm": LAM * 1e3,
            "z_star_mm": Z_STAR * 1e3,
            "configs": results_info,
            "total_time_s": total_time,
        }, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"FEM generation complete in {total_time:.0f}s")
    print(f"Results: {FEM_DIR}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
