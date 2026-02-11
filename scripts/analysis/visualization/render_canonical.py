#!/usr/bin/env python3
"""
Render all 4 canonical visualizations from a rich-data directory.

Usage:
    python scripts/visualization/render_canonical.py \\
        --run_dir results/rich_A_20260207_195247 \\
        --roi_size 0.008
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / 'src'))

from acousto.viz.batch import render_all


def main():
    parser = argparse.ArgumentParser(
        description='Render canonical acoustic-tweezers visualizations')
    parser.add_argument('--run_dir', required=True,
                        help='Directory with NPZ + XDMF from generate_rich_data.py')
    parser.add_argument('--roi_size', type=float, default=0.008,
                        help='ROI cube edge length in metres (default: 0.008 = 8 mm)')
    parser.add_argument('--roi_center', type=float, nargs=3, default=None,
                        help='ROI center [x, y, z] in metres (default: mesh centre)')
    args = parser.parse_args()

    roi_center = args.roi_center
    if roi_center is not None:
        import numpy as np
        roi_center = np.array(roi_center)

    render_all(args.run_dir, roi_center=roi_center, roi_size=args.roi_size)


if __name__ == '__main__':
    main()
