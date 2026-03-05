#!/usr/bin/env python3
"""
Hybrid FEM–ASM Bridge Formation Pipeline
=========================================

This script reproduces the full vortex–standing-wave bridge formation study.

It performs the complete investigation pipeline:
  1. Loads the pre-computed FEM standing-wave cache
  2. Generates ASM vortex fields with configurable lens parameters
  3. Runs a parameter sweep over (α, φ₀, x₀, y₀)
  4. Evaluates the Gor'kov bridge metric between adjacent traps
  5. Produces figures and a REPORT.md

The study investigates whether interference between a focused vortex beam
and the standing-wave lattice can reduce the Gor'kov potential barrier
between adjacent traps, forming a transport "bridge".

Usage:
    python scripts/run_bridge_pipeline.py              # standard sweep
    python scripts/run_bridge_pipeline.py --large      # expanded sweep
    python scripts/run_bridge_pipeline.py --quick      # reduced sweep for testing
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(
        description="Run the hybrid FEM–ASM bridge formation study")
    ap.add_argument(
        "--large", action="store_true",
        help="Run the large master parameter study (extended α, phase, offset grids)")
    ap.add_argument(
        "--quick", action="store_true",
        help="Run a quick reduced sweep for testing")
    ap.add_argument(
        "--cache", type=str, default=None,
        help="Path to FEM standing-wave cache .npz file")
    args = ap.parse_args()

    # Verify FEM cache exists
    cache_dir = PROJECT_ROOT / "results" / "fem_standing_wave_cache"
    if args.cache:
        cache_path = Path(args.cache).resolve()
    else:
        npzs = sorted(cache_dir.glob("*.npz"), key=lambda p: p.stat().st_mtime)
        if not npzs:
            print("ERROR: No FEM cache found in", cache_dir)
            print("Generate one first with the FEM solver.")
            sys.exit(1)
        cache_path = npzs[-1]

    if not cache_path.exists():
        print(f"ERROR: Cache file not found: {cache_path}")
        sys.exit(1)

    print(f"FEM cache: {cache_path}")
    print(f"  Size: {cache_path.stat().st_size / 1e6:.1f} MB")

    # Select which script to run
    if args.large:
        script = PROJECT_ROOT / "scripts" / "dev" / "bridge_master_study.py"
        label = "LARGE master parameter study"
    else:
        script = PROJECT_ROOT / "scripts" / "dev" / "bridge_phase_offset_study.py"
        label = "standard bridge phase-offset study"

    if not script.exists():
        print(f"ERROR: Script not found: {script}")
        sys.exit(1)

    cmd = [sys.executable, str(script), "--cache", str(cache_path)]
    if args.quick:
        cmd.append("--quick")

    print(f"\nRunning: {label}")
    print(f"Command: {' '.join(cmd)}\n")
    print("=" * 72)

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
