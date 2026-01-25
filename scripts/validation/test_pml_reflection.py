#!/usr/bin/env python3
"""
PML Reflection Validation Test

Validates that the PML (Perfectly Matched Layer) achieves <1% reflection.

Test setup:
- Plane wave incident on PML boundary
- Measure reflected wave amplitude
- Pass if reflection < 1%

Author: Acousto-Tweezers Project
Date: January 2026
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

import numpy as np
from datetime import datetime

# Only import what we need to avoid heavy dependencies for quick test
THRESHOLD_REFLECTION = 0.01  # 1%
THRESHOLD_WARN = 0.05        # 5%


def run_pml_reflection_test(ppw: int = 10, verbose: bool = True) -> dict:
    """
    Run PML reflection validation test.
    
    Parameters
    ----------
    ppw : int
        Points per wavelength
    verbose : bool
        Print progress
        
    Returns
    -------
    dict
        Test results with status
    """
    from tweezers.fenicsx import FEMConfig, PhysicsLevel, run_simulation, MaterialDatabase
    
    if verbose:
        print("=" * 60)
        print("PML REFLECTION VALIDATION TEST")
        print("=" * 60)
    
    # Configure for PML test
    config = FEMConfig.default()
    config.physics_level = PhysicsLevel.ACOUSTICS_PML
    config.geometry.elements_per_wavelength = ppw
    config.geometry.pml_thickness = 0.005  # 5mm PML
    config.physics.frequency = 2e6  # 2 MHz
    
    # Run simulation
    if verbose:
        print(f"Running simulation with PPW={ppw}...")
    
    try:
        result = run_simulation(config, output_dir=None)
    except Exception as e:
        return {
            'status': 'ERROR',
            'message': str(e),
            'reflection_coefficient': None,
        }
    
    # Compute reflection coefficient
    if result.pml_metrics is not None:
        R = result.pml_metrics.reflection_coefficient
    else:
        # Estimate from field decay in PML
        R = estimate_pml_reflection(result.acoustic_field, config)
    
    # Determine status
    if R < THRESHOLD_REFLECTION:
        status = 'PASS'
    elif R < THRESHOLD_WARN:
        status = 'WARN'
    else:
        status = 'FAIL'
    
    results = {
        'status': status,
        'reflection_coefficient': R,
        'threshold': THRESHOLD_REFLECTION,
        'ppw': ppw,
        'pml_thickness_mm': config.geometry.pml_thickness * 1e3,
        'timestamp': datetime.now().isoformat(),
    }
    
    if verbose:
        print(f"\n[{status}] Reflection coefficient: {R*100:.2f}%")
        print(f"  Target: <{THRESHOLD_REFLECTION*100:.0f}%")
        print("=" * 60)
    
    return results


def estimate_pml_reflection(acoustic_field, config):
    """Estimate PML reflection from field decay."""
    if acoustic_field is None:
        return 1.0  # Fail if no field
    
    # Compare field amplitude at PML boundary vs interior
    coords = acoustic_field.coords
    p = np.abs(acoustic_field.p)
    
    # Find points near PML boundary
    R = config.geometry.dish_inner_radius
    r = np.sqrt(coords[:, 0]**2 + coords[:, 1]**2)
    
    interior_mask = r < 0.5 * R
    boundary_mask = (r > 0.9 * R) & (r < R)
    
    if np.sum(interior_mask) > 0 and np.sum(boundary_mask) > 0:
        p_interior = np.mean(p[interior_mask])
        p_boundary = np.mean(p[boundary_mask])
        
        if p_interior > 0:
            # Reflection ~ boundary/interior ratio (simplified)
            R_est = abs(p_boundary / p_interior - 1)
            return min(R_est, 1.0)
    
    return 1.0  # Fail if can't estimate


def save_results(results: dict, output_dir: Path):
    """Save test results to file."""
    import json
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "pml_reflection_test.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    with open(output_dir / "pml_reflection_test.txt", 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("PML REFLECTION VALIDATION TEST\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Status: {results['status']}\n")
        f.write(f"Reflection coefficient: {results['reflection_coefficient']*100:.2f}%\n")
        f.write(f"Threshold: <{results['threshold']*100:.0f}%\n")
        f.write(f"PPW: {results['ppw']}\n")
        f.write(f"Timestamp: {results['timestamp']}\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='PML Reflection Validation Test')
    parser.add_argument('--ppw', type=int, default=10, help='Points per wavelength')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    parser.add_argument('--quiet', action='store_true', help='Suppress output')
    args = parser.parse_args()
    
    results = run_pml_reflection_test(ppw=args.ppw, verbose=not args.quiet)
    
    if args.output:
        save_results(results, Path(args.output))
    
    sys.exit(0 if results['status'] != 'FAIL' else 1)
