#!/usr/bin/env python3
"""
High-fidelity vortex perturbation hero render.

Generates a single, publication-quality 3D visualization of the
vortex–standing wave interaction using the perturbation field Δp.

Usage:
    python scripts/visualization/render_vortex_hero.py \
        --run_dir results/rich_A_20260208_102757 \
        --output hero_vortex_perturbation.png
"""

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parents[2] / 'src'))

from acousto.viz.loaders import load_rich
from acousto.viz.views import render_vortex_perturbation_hires


def main():
    parser = argparse.ArgumentParser(
        description='Render high-fidelity vortex perturbation hero image'
    )
    parser.add_argument('--run_dir', type=str, required=True,
                        help='Run directory with standing.npz, combined.npz')
    parser.add_argument('--output', type=str, default=None,
                        help='Output PNG path (default: {run_dir}/hero_vortex_perturbation.png)')
    parser.add_argument('--roi_size', type=float, default=0.010,
                        help='ROI cube edge in metres (default: 0.010 = 10 mm)')
    parser.add_argument('--percentile', type=int, default=97,
                        help='Iso-surface percentile of |Δp| (default: 97)')
    parser.add_argument('--resolution', type=int, nargs=2, default=[2400, 1400],
                        help='Resolution W H in pixels (default: 2400 1400)')
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Error: run_dir not found: {run_dir}")
        sys.exit(1)

    out_path = Path(args.output) if args.output else \
               run_dir / 'hero_vortex_perturbation.png'

    print(f"\n{'='*70}")
    print(f"VORTEX PERTURBATION — HIGH-FIDELITY HERO RENDER")
    print(f"{'='*70}")
    print(f"Run dir : {run_dir}")
    print(f"Output  : {out_path}")
    print(f"ROI size: {args.roi_size*1000:.1f} mm")
    print(f"Percentile: {args.percentile}%")
    print(f"Resolution: {args.resolution[0]}×{args.resolution[1]}")
    print(f"{'='*70}\n")

    # Load data
    print("Loading standing and combined fields...")
    standing = load_rich(run_dir, 'standing')
    combined = load_rich(run_dir, 'combined')
    print(f"  Standing: {standing.n_points} points")
    print(f"  Combined: {combined.n_points} points\n")

    # Render
    render_vortex_perturbation_hires(
        standing, combined,
        out_path,
        roi_center=None,
        roi_size=args.roi_size,
        percentile=args.percentile,
        resolution=tuple(args.resolution),
        title=f'Vortex Perturbation Δp (percentile {args.percentile}%)'
    )

    print(f"\n{'='*70}")
    print(f"DONE — {out_path}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
