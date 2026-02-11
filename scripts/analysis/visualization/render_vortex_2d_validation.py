#!/usr/bin/env python3
"""
2D validation slice of vortex perturbation field.

Shows |Δp| magnitude and phase structure in a horizontal cross-section.
Useful for understanding the 3D geometry and validating the perturbation is local.
"""

import argparse
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

sys.path.append(str(Path(__file__).parents[2] / 'src'))

from acousto.viz.loaders import load_rich, clip_roi


def main():
    parser = argparse.ArgumentParser(
        description='Create 2D validation slice of vortex perturbation'
    )
    parser.add_argument('--run_dir', type=str, required=True)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--roi_size', type=float, default=0.010)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_path = Path(args.output) if args.output else \
               run_dir / 'hero_vortex_perturbation_2d_slice.png'

    print(f"\n{'='*70}")
    print(f"VORTEX PERTURBATION — 2D VALIDATION SLICE")
    print(f"{'='*70}")

    # Load data
    standing = load_rich(run_dir, 'standing')
    combined = load_rich(run_dir, 'combined')
    
    # Auto ROI center
    bounds = combined.bounds
    roi_center = np.array([
        (bounds[0] + bounds[1]) / 2,
        (bounds[2] + bounds[3]) / 2,
        (bounds[4] + bounds[5]) / 2,
    ])
    
    # Clip to ROI
    standing_clip = clip_roi(standing, roi_center, args.roi_size)
    combined_clip = clip_roi(combined, roi_center, args.roi_size)
    
    # Compute perturbation
    p_stand = standing_clip.point_data['p_real'] + 1j * standing_clip.point_data['p_imag']
    p_comb  = combined_clip.point_data['p_real'] + 1j * combined_clip.point_data['p_imag']
    n = min(len(p_stand), len(p_comb))
    
    delta_p = p_comb[:n] - p_stand[:n]
    
    # Extract horizontal slice
    z_slice = roi_center[2]
    origin = [roi_center[0], roi_center[1], z_slice]
    normal = [0, 0, 1]
    sliced = combined_clip.extract_points(np.arange(n)).slice(normal=normal, origin=origin)
    
    if sliced.n_points == 0:
        print("Warning: No points in slice")
        return
    
    x = sliced.points[:, 0] * 1000  # mm
    y = sliced.points[:, 1] * 1000
    
    # Resample perturbation onto slice
    combined_clip.point_data['delta_magnitude'] = np.abs(delta_p)
    combined_clip.point_data['delta_phase'] = np.angle(delta_p)
    sliced_resampled = sliced.sample(combined_clip)
    
    mag_delta = sliced_resampled.point_data['delta_magnitude']
    phase_delta = sliced_resampled.point_data['delta_phase']
    
    # Create figure with two panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=120)
    
    # Panel 1: Magnitude of perturbation
    scatter1 = ax1.scatter(x, y, c=mag_delta, cmap='plasma', s=30, alpha=0.8, edgecolors='none')
    ax1.set_xlabel('x (mm)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('y (mm)', fontsize=12, fontweight='bold')
    ax1.set_title('|Δp| — Magnitude of Vortex Perturbation', fontsize=13, fontweight='bold')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.2)
    cbar1 = plt.colorbar(scatter1, ax=ax1, label='|Δp| (Pa)')
    
    # Dish outline
    dish_center_mm = roi_center[:2] * 1000
    rect1 = plt.Rectangle(
        (dish_center_mm[0] - 10, dish_center_mm[1] - 10), 20, 20,
        fill=False, edgecolor='red', linewidth=2, linestyle='--'
    )
    ax1.add_patch(rect1)
    
    # Panel 2: Phase of perturbation
    scatter2 = ax2.scatter(x, y, c=phase_delta, cmap='twilight', s=30, alpha=0.8, 
                          vmin=-np.pi, vmax=np.pi, edgecolors='none')
    ax2.set_xlabel('x (mm)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('y (mm)', fontsize=12, fontweight='bold')
    ax2.set_title('arg(Δp) — Phase Structure (Vortex Helix)', fontsize=13, fontweight='bold')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.2)
    cbar2 = plt.colorbar(scatter2, ax=ax2, label='arg(Δp) (rad)')
    
    # Dish outline
    rect2 = plt.Rectangle(
        (dish_center_mm[0] - 10, dish_center_mm[1] - 10), 20, 20,
        fill=False, edgecolor='red', linewidth=2, linestyle='--'
    )
    ax2.add_patch(rect2)
    
    fig.suptitle('Vortex Perturbation Field — 2D Cross-Section (z = center)',
                fontsize=14, fontweight='bold', y=0.98)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=120, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Created: {out_path}")
    print(f"  Points in slice: {sliced.n_points}")
    print(f"  |Δp| range: [{np.min(mag_delta):.1f}, {np.max(mag_delta):.1f}] Pa")
    print(f"  arg(Δp) range: [{np.min(phase_delta):.2f}, {np.max(phase_delta):.2f}] rad")
    
    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    main()
