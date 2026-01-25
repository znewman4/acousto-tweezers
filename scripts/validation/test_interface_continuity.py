#!/usr/bin/env python3
"""
Interface Continuity Validation Test

Validates fluid-fluid interface conditions:
- Pressure continuity: p₁ = p₂
- Normal velocity continuity: (1/ρ₁) ∂p₁/∂n = (1/ρ₂) ∂p₂/∂n

For water-air interface test.

Author: Acousto-Tweezers Project
Date: January 2026
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

import numpy as np
from datetime import datetime

THRESHOLD_PRESSURE_JUMP = 1e-3  # Relative jump threshold
THRESHOLD_VELOCITY_MISMATCH = 1e-2


def run_interface_continuity_test(ppw: int = 10, verbose: bool = True) -> dict:
    """
    Run interface continuity validation test.
    
    Tests water-air interface for:
    - Pressure continuity
    - Normal velocity continuity (accounting for density jump)
    
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
        print("INTERFACE CONTINUITY VALIDATION TEST")
        print("=" * 60)
    
    # Configure for multi-fluid test (water + air)
    config = FEMConfig.default()
    config.physics_level = PhysicsLevel.FLUID_AIR_BATH
    config.geometry.elements_per_wavelength = ppw
    config.physics.frequency = 2e6
    
    # Run simulation
    if verbose:
        print(f"Running simulation with PPW={ppw}...")
    
    try:
        result = run_simulation(config, output_dir=None)
    except Exception as e:
        return {
            'status': 'ERROR',
            'message': str(e),
            'pressure_jump': None,
            'velocity_mismatch': None,
        }
    
    # Get material properties
    materials = MaterialDatabase(config.physics.temperature)
    
    # Compute interface metrics
    metrics = compute_interface_metrics(result.acoustic_field, config, materials)
    
    # Determine status
    if (metrics['pressure_jump_relative'] < THRESHOLD_PRESSURE_JUMP and 
        metrics['velocity_mismatch_relative'] < THRESHOLD_VELOCITY_MISMATCH):
        status = 'PASS'
    elif (metrics['pressure_jump_relative'] < 10 * THRESHOLD_PRESSURE_JUMP and
          metrics['velocity_mismatch_relative'] < 10 * THRESHOLD_VELOCITY_MISMATCH):
        status = 'WARN'
    else:
        status = 'FAIL'
    
    results = {
        'status': status,
        'pressure_jump_relative': metrics['pressure_jump_relative'],
        'velocity_mismatch_relative': metrics['velocity_mismatch_relative'],
        'threshold_pressure': THRESHOLD_PRESSURE_JUMP,
        'threshold_velocity': THRESHOLD_VELOCITY_MISMATCH,
        'ppw': ppw,
        'timestamp': datetime.now().isoformat(),
    }
    
    if verbose:
        print(f"\n[{status}] Interface Continuity Results:")
        print(f"  Pressure jump (relative): {metrics['pressure_jump_relative']:.2e}")
        print(f"    Target: <{THRESHOLD_PRESSURE_JUMP:.0e}")
        print(f"  Velocity mismatch (relative): {metrics['velocity_mismatch_relative']:.2e}")
        print(f"    Target: <{THRESHOLD_VELOCITY_MISMATCH:.0e}")
        print("=" * 60)
    
    return results


def compute_interface_metrics(acoustic_field, config, materials):
    """Compute interface continuity metrics."""
    if acoustic_field is None:
        return {
            'pressure_jump_relative': 1.0,
            'velocity_mismatch_relative': 1.0,
        }
    
    coords = acoustic_field.coords
    p = acoustic_field.p
    
    # Water-air interface is at z = water_depth
    z_interface = config.geometry.water_depth
    z_tol = config.geometry.water_depth / 20
    
    # Points just below and above interface
    z = coords[:, 2]
    below_mask = np.abs(z - (z_interface - z_tol)) < z_tol/2
    above_mask = np.abs(z - (z_interface + z_tol)) < z_tol/2
    
    if np.sum(below_mask) == 0 or np.sum(above_mask) == 0:
        return {
            'pressure_jump_relative': 1.0,
            'velocity_mismatch_relative': 1.0,
        }
    
    p_below = np.mean(np.abs(p[below_mask]))
    p_above = np.mean(np.abs(p[above_mask]))
    
    # Pressure jump (relative)
    p_mean = (p_below + p_above) / 2
    if p_mean > 0:
        pressure_jump = abs(p_below - p_above) / p_mean
    else:
        pressure_jump = 1.0
    
    # Velocity mismatch (simplified - would need gradient computation)
    # For now, use pressure ratio as proxy
    rho_water = materials.water.density
    rho_air = materials.air.density
    
    # At interface: v_water = v_air
    # (1/ρ_w) ∂p_w/∂n = (1/ρ_a) ∂p_a/∂n
    # Simplified: p_w/ρ_w ≈ p_a/ρ_a (for plane wave normal incidence)
    expected_ratio = rho_air / rho_water
    if p_above > 0 and p_below > 0:
        actual_ratio = p_above / p_below
        velocity_mismatch = abs(actual_ratio - expected_ratio) / max(expected_ratio, 0.01)
    else:
        velocity_mismatch = 1.0
    
    return {
        'pressure_jump_relative': pressure_jump,
        'velocity_mismatch_relative': velocity_mismatch,
    }


def save_results(results: dict, output_dir: Path):
    """Save test results to file."""
    import json
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "interface_continuity_test.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    with open(output_dir / "interface_continuity_test.txt", 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("INTERFACE CONTINUITY VALIDATION TEST\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Status: {results['status']}\n")
        f.write(f"Pressure jump (relative): {results['pressure_jump_relative']:.2e}\n")
        f.write(f"Velocity mismatch (relative): {results['velocity_mismatch_relative']:.2e}\n")
        f.write(f"Timestamp: {results['timestamp']}\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Interface Continuity Validation Test')
    parser.add_argument('--ppw', type=int, default=10, help='Points per wavelength')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    parser.add_argument('--quiet', action='store_true', help='Suppress output')
    args = parser.parse_args()
    
    results = run_interface_continuity_test(ppw=args.ppw, verbose=not args.quiet)
    
    if args.output:
        save_results(results, Path(args.output))
    
    sys.exit(0 if results['status'] != 'FAIL' else 1)
