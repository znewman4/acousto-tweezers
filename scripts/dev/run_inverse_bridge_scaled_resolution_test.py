#!/usr/bin/env python3
"""
Resolution-limit test for the bridge inverse lens pipeline.

Part 1: Scale the bridge target by a given factor (default 2x) and run the
        same inverse IASA pipeline.
Part 2: Compute and report resolution estimates for the current system and
        the frequency required to resolve the real (unscaled) bridge.

This script does NOT modify the inverse algorithm or export pipeline.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BRIDGE_FIELDS_NPZ = (
    PROJECT_ROOT / "results" / "dev" / "bridge_pressure_field_standalone_scaled"
    / "bridge_pressure_fields.npz"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bridge resolution-limit test")
    p.add_argument("--scale-factor", type=float, default=2.0,
                   help="Factor by which to scale bridge geometry (default: 2.0).")
    p.add_argument("--input-npz", type=Path, default=BRIDGE_FIELDS_NPZ)
    p.add_argument("--focal-distance-mm", type=float, default=None,
                   help="If set, skip focal sweep and use this value.")
    p.add_argument("--focal-steps", type=int, default=15)
    p.add_argument("--stl-grid-stride", type=int, default=8)
    p.add_argument("--n-iter", type=int, default=100)
    p.add_argument("--frequency-hz", type=float, default=2.15e6)
    p.add_argument("--transducer-diameter-mm", type=float, default=20.0)
    p.add_argument("--c-water", type=float, default=1480.0)
    p.add_argument("--skip-stl", action="store_true")
    return p.parse_args()


def generate_scaled_bridge(input_npz: Path, scale: float, out_dir: Path) -> Path:
    """Load bridge NPZ and produce a scaled version.

    Scaling is done by multiplying the coordinate arrays by `scale`.
    The pressure field array stays the same — its *physical footprint* grows
    by the scale factor, so all spatial features (corridor width, A-B separation)
    appear `scale`x larger when the inverse pipeline maps them onto the lens grid.
    """
    d = np.load(input_npz)
    x_full = d["x_full"].astype(float)
    y_full = d["y_full"].astype(float)

    # Centre, scale, re-centre to same midpoint
    x_mid = 0.5 * (x_full[0] + x_full[-1])
    y_mid = 0.5 * (y_full[0] + y_full[-1])
    x_scaled = (x_full - x_mid) * scale + x_mid
    y_scaled = (y_full - y_mid) * scale + y_mid

    # Also scale ROI coords if present
    save_dict = {}
    for key in d.files:
        if key in ("x_full", "y_full"):
            continue
        if key in ("x_roi", "y_roi"):
            arr = d[key].astype(float)
            mid = 0.5 * (arr[0] + arr[-1])
            # Use the *same* centre as the full grid for consistency
            ref_mid = x_mid if key == "x_roi" else y_mid
            save_dict[key] = (arr - ref_mid) * scale + ref_mid
        elif key == "traps_m":
            traps = d[key].astype(float)
            traps_scaled = traps.copy()
            traps_scaled[:, 0] = (traps[:, 0] - x_mid) * scale + x_mid
            traps_scaled[:, 1] = (traps[:, 1] - y_mid) * scale + y_mid
            save_dict[key] = traps_scaled
        else:
            save_dict[key] = d[key]

    save_dict["x_full"] = x_scaled
    save_dict["y_full"] = y_scaled

    out_npz = out_dir / f"bridge_pressure_fields_scaled{scale:.0f}x.npz"
    np.savez_compressed(out_npz, **save_dict)
    return out_npz


def compute_resolution_estimates(
    frequency_hz: float,
    c_water: float,
    transducer_diameter_mm: float,
    focal_distance_mm: float,
    bridge_separation_um: float,
    corridor_width_um: float,
    scale_factor: float,
) -> dict:
    lam = c_water / frequency_hz
    D = transducer_diameter_mm * 1e-3
    z = focal_distance_mm * 1e-3
    NA = (D / 2.0) / z
    delta = lam / (2.0 * NA)  # Rayleigh-like resolution

    # Feature sizes
    real_sep = bridge_separation_um * 1e-6
    real_width = corridor_width_um * 1e-6
    scaled_sep = real_sep * scale_factor
    scaled_width = real_width * scale_factor

    # Frequency needed to resolve real bridge at this NA
    # delta = lam / (2*NA) = c / (f * 2*NA) <= real_sep
    # => f >= c / (2 * NA * real_sep)
    f_needed_sep = c_water / (2.0 * NA * real_sep)
    f_needed_width = c_water / (2.0 * NA * real_width)

    # Can resolution resolve the scaled bridge?
    can_resolve_scaled_sep = delta <= scaled_sep
    can_resolve_scaled_width = delta <= scaled_width

    return {
        "wavelength_um": lam * 1e6,
        "NA": NA,
        "resolution_delta_um": delta * 1e6,
        "resolution_delta_mm": delta * 1e3,
        "real_bridge_separation_um": bridge_separation_um,
        "real_corridor_width_um": corridor_width_um,
        "scaled_bridge_separation_um": scaled_sep * 1e6,
        "scaled_corridor_width_um": scaled_width * 1e6,
        "can_resolve_scaled_separation": can_resolve_scaled_sep,
        "can_resolve_scaled_width": can_resolve_scaled_width,
        "ratio_delta_to_real_sep": delta / real_sep,
        "ratio_delta_to_scaled_sep": delta / scaled_sep,
        "frequency_needed_for_real_sep_MHz": f_needed_sep / 1e6,
        "frequency_needed_for_real_width_MHz": f_needed_width / 1e6,
    }


def main() -> None:
    args = parse_args()
    scale = float(args.scale_factor)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"scaled{scale:.0f}x"
    out_dir = PROJECT_ROOT / "results" / "dev" / f"inverse_bridge_pressure_lens_replica_{suffix}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[start] output_dir={out_dir}", flush=True)
    print(f"[start] scale_factor={scale}", flush=True)

    # ── Part 1: Generate scaled bridge NPZ ─────────────────────────────
    scaled_npz = generate_scaled_bridge(args.input_npz, scale, out_dir)
    print(f"[scaled] saved: {scaled_npz}", flush=True)

    # Print original vs scaled geometry
    d_orig = np.load(args.input_npz)
    d_scaled = np.load(scaled_npz)
    traps_orig = d_orig["traps_m"]
    traps_scaled = d_scaled["traps_m"]
    ia = int(d_orig["idx_a"]); ib = int(d_orig["idx_b"])
    a_orig = traps_orig[ia]
    b_orig_y = traps_orig[ib][1]
    # standalone forces b_xy = [a[0], b_orig[1]]
    sep_orig = abs(a_orig[1] - b_orig_y)
    sep_scaled = sep_orig * scale
    print(f"[geom] original A-B separation: {sep_orig*1e6:.1f} µm", flush=True)
    print(f"[geom] scaled A-B separation:   {sep_scaled*1e6:.1f} µm ({scale:.0f}x)", flush=True)
    print(f"[geom] original corridor width:  300 µm", flush=True)
    print(f"[geom] scaled corridor width:    {300*scale:.0f} µm ({scale:.0f}x)", flush=True)
    print(f"[geom] original field extent:    "
          f"[{d_orig['x_full'][0]*1e3:.2f}, {d_orig['x_full'][-1]*1e3:.2f}] mm", flush=True)
    print(f"[geom] scaled field extent:      "
          f"[{d_scaled['x_full'][0]*1e3:.2f}, {d_scaled['x_full'][-1]*1e3:.2f}] mm", flush=True)

    # ── Run existing inverse pipeline on scaled field ──────────────────
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "dev" / "run_inverse_replica_on_bridge_pressure_field.py"),
        "--input-npz", str(scaled_npz),
        "--stl-grid-stride", str(args.stl_grid_stride),
        "--focal-steps", str(args.focal_steps),
        "--n-iter", str(args.n_iter),
        "--frequency-hz", str(args.frequency_hz),
        "--transducer-diameter-mm", str(args.transducer_diameter_mm),
        "--output-dir", str(out_dir),
    ]
    if args.focal_distance_mm is not None:
        cmd += ["--focal-distance-mm", str(args.focal_distance_mm)]
    if args.skip_stl:
        cmd += ["--skip-stl"]

    print(f"\n[run] {' '.join(cmd)}\n", flush=True)
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print(f"[error] inverse pipeline exited with code {result.returncode}", flush=True)
        sys.exit(1)

    # ── Part 2: Resolution estimates ───────────────────────────────────
    # Read the manifest to get the chosen focal distance
    manifest_path = out_dir / "bridge_inverse_replica_manifest.json"
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    focal_mm = manifest["config"]["focal_distance_mm"]

    res = compute_resolution_estimates(
        frequency_hz=args.frequency_hz,
        c_water=args.c_water,
        transducer_diameter_mm=args.transducer_diameter_mm,
        focal_distance_mm=focal_mm,
        bridge_separation_um=sep_orig * 1e6,
        corridor_width_um=300.0,
        scale_factor=scale,
    )

    # ── Print resolution report ────────────────────────────────────────
    print("\n" + "=" * 64, flush=True)
    print("  RESOLUTION ANALYSIS", flush=True)
    print("=" * 64, flush=True)
    print(f"  Frequency:              {args.frequency_hz/1e6:.2f} MHz", flush=True)
    print(f"  Wavelength:             {res['wavelength_um']:.1f} µm", flush=True)
    print(f"  Transducer diameter:    {args.transducer_diameter_mm:.1f} mm", flush=True)
    print(f"  Focal distance (used):  {focal_mm:.2f} mm", flush=True)
    print(f"  NA:                     {res['NA']:.4f}", flush=True)
    print(f"  Resolution δ:           {res['resolution_delta_um']:.1f} µm "
          f"({res['resolution_delta_mm']:.3f} mm)", flush=True)
    print(f"", flush=True)
    print(f"  Real bridge A-B sep:    {res['real_bridge_separation_um']:.1f} µm", flush=True)
    print(f"  Real corridor width:    {res['real_corridor_width_um']:.1f} µm", flush=True)
    print(f"  δ / real separation:    {res['ratio_delta_to_real_sep']:.2f}x "
          f"({'BELOW LIMIT' if res['ratio_delta_to_real_sep'] > 1 else 'RESOLVABLE'})", flush=True)
    print(f"", flush=True)
    print(f"  Scaled ({scale:.0f}x) A-B sep:    {res['scaled_bridge_separation_um']:.1f} µm", flush=True)
    print(f"  Scaled ({scale:.0f}x) width:       {res['scaled_corridor_width_um']:.1f} µm", flush=True)
    print(f"  δ / scaled separation:  {res['ratio_delta_to_scaled_sep']:.2f}x "
          f"({'BELOW LIMIT' if res['ratio_delta_to_scaled_sep'] > 1 else 'RESOLVABLE'})", flush=True)
    print(f"", flush=True)
    print(f"  Freq needed for real sep:   {res['frequency_needed_for_real_sep_MHz']:.2f} MHz", flush=True)
    print(f"  Freq needed for real width: {res['frequency_needed_for_real_width_MHz']:.2f} MHz", flush=True)
    print("=" * 64, flush=True)

    # ── Append resolution info to manifest ─────────────────────────────
    manifest["bridge_scale_factor"] = scale
    manifest["bridge_original_separation_um"] = sep_orig * 1e6
    manifest["bridge_original_corridor_width_um"] = 300.0
    # Convert numpy types to native Python types for JSON serialization
    manifest["resolution_analysis"] = {
        k: (bool(v) if isinstance(v, (np.bool_,)) else
            float(v) if isinstance(v, (np.floating, np.integer)) else v)
        for k, v in res.items()
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[done] manifest updated with resolution analysis: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
