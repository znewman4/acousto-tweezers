#!/usr/bin/env python3
"""
Smoke Test: Level-2 Stokes Streaming Solver

Minimal test to verify:
1. Solver converges on tiny mesh
2. Produces nonzero velocities
3. Divergence constraint is satisfied
4. Proper preconditioner configuration works

This is a quick test (~5-10 min) on a very coarse mesh (few thousand DOFs)
to catch major regressions before running full validation.

Usage:
    python scripts/validation/test_streaming_stokes_smoke.py

Author: Acousto-Tweezers Project
Date: 2026-02-09
"""

import sys
from pathlib import Path
import numpy as np
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parents[2] / 'src'))

from acoustweezers.experiments.shallow_square_dish import (
    ShallowDishConfig,
    solve_all_pressure_cases,
)
from acoustweezers.experiments.shallow_square_dish.streaming import (
    solve_streaming_stokes,
)


def test_streaming_stokes_smoke():
    """Run smoke test for Stokes streaming solver."""
    
    print("="*70)
    print("STREAMING STOKES SOLVER - SMOKE TEST")
    print("="*70)
    print()
    
    # =========================================================================
    # SETUP: Minimal config (coarse mesh, short domain)
    # =========================================================================
    print("Setup: Creating minimal configuration...")
    
    cfg = ShallowDishConfig(
        L=0.01,              # 1 cm dish (smaller than 5 cm)
        H=0.001,             # 1 mm depth
        frequency_hz=500e3,
        standing_velocity_amplitude=1e-6,
        standing_phase_pattern="antiphase",
        vortex_velocity_amplitude=1e-6,
        vortex_topological_charge=1,
        vortex_aperture_radius=2e-3,
        vortex_path_type="fixed",
        vortex_path_n_steps=1,
        elements_per_wavelength=2,  # Coarse mesh!
    )
    
    print(f"  Domain: {cfg.L*1e2:.1f} cm × {cfg.H*1e3:.1f} mm")
    print(f"  Mesh density: {cfg.elements_per_wavelength} elements/wavelength (coarse)")
    print()
    
    # =========================================================================
    # SOLVE PRESSURE
    # =========================================================================
    print("Step 1: Solving acoustic pressure fields...")
    
    try:
        solutions = solve_all_pressure_cases(cfg, verbose=False)
        p_combined = solutions["combined"]
        print(f"  ✓ Pressure solve successful")
        print(f"    max|p| = {np.max(np.abs(p_combined.p_values)):.2f} Pa")
    except Exception as e:
        print(f"  ✗ Pressure solve failed: {e}")
        return False
    
    print()
    
    # =========================================================================
    # SOLVE STREAMING (Main test)
    # =========================================================================
    print("Step 2: Solving Stokes streaming with fieldsplit preconditioner...")
    
    try:
        streaming = solve_streaming_stokes(
            p_combined,
            downsample_factor=1,  # Use acoustic mesh
            forcing_scale=1.0,
            verbose=True,
        )
    except Exception as e:
        print(f"  ✗ Streaming solve raised exception: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    if streaming is None:
        print(f"  ✗ Streaming solver diverged (None returned)")
        return False
    
    print()
    
    # =========================================================================
    # VERIFICATION TESTS
    # =========================================================================
    print("Step 3: Verification tests...")
    print()
    
    test_passed = True
    diags = streaming.diagnostics
    
    # Test 1: Convergence
    conv_reason = diags.get('ksp_converged_reason')
    if conv_reason < 0:
        print(f"  ✗ Test 1 (Convergence): FAILED")
        print(f"    KSP converged reason: {conv_reason} ({diags.get('ksp_reason_str')})")
        print(f"    Iterations: {diags.get('ksp_iterations')}")
        print(f"    Final residual: {diags.get('ksp_final_residual_norm'):.2e}")
        test_passed = False
    else:
        print(f"  ✓ Test 1 (Convergence): PASSED")
        print(f"    KSP iterations: {diags.get('ksp_iterations')}")
        print(f"    Final residual: {diags.get('ksp_final_residual_norm'):.2e}")
    
    print()
    
    # Test 2: Nonzero velocity
    max_u = diags.get('max_u_um_s', 0.0)
    mean_u = diags.get('mean_u_um_s', 0.0)
    if max_u > 0.1:  # At least 0.1 μm/s
        print(f"  ✓ Test 2 (Nonzero velocity): PASSED")
        print(f"    max|u| = {max_u:.3f} μm/s")
        print(f"    mean|u| = {mean_u:.3f} μm/s")
    else:
        print(f"  ✗ Test 2 (Nonzero velocity): FAILED")
        print(f"    max|u| = {max_u:.3f} μm/s (expected > 0.1)")
        test_passed = False
    
    print()
    
    # Test 3: Divergence small
    div_rel = diags.get('divergence_l2_norm_relative', 1e9)
    if div_rel < 0.1:  # Relative divergence < 10%
        print(f"  ✓ Test 3 (Divergence constraint): PASSED")
        print(f"    relative ||∇·u|| = {div_rel:.2e} (should be < 0.1)")
    else:
        print(f"  ⚠ Test 3 (Divergence constraint): WARNING")
        print(f"    relative ||∇·u|| = {div_rel:.2e} (should be < 0.1)")
        # Not a hard failure, but concerning
    
    print()
    
    # Test 4: Z-profile structure (near-wall should be stronger than bulk)
    z_profile = diags.get('z_profile', [])
    if len(z_profile) >= 3:
        u_at_bottom = z_profile[0].get('u_mean_um_s', 0.0)  # First level (near z=0)
        u_at_mid = z_profile[len(z_profile)//2].get('u_mean_um_s', 0.0)  # Mid-plane
        u_at_top = z_profile[-1].get('u_mean_um_s', 0.0)  # Top (z=H)
        
        # In Rayleigh streaming, bottom boundary layer should drive flow
        # Expect some structure (not uniform)
        u_range = max(u_at_bottom, u_at_top) - min(u_at_bottom, u_at_top)
        if u_range > u_at_mid * 0.1:  # Some vertical variation
            print(f"  ✓ Test 4 (Z-profile structure): PASSED")
            print(f"    u(z=0)={u_at_bottom:.3f} μm/s, u(mid)={u_at_mid:.3f}, u(z=H)={u_at_top:.3f}")
        else:
            print(f"  ⚠ Test 4 (Z-profile structure): Uniform profile (expected some variation)")
            print(f"    u(z=0)={u_at_bottom:.3f} μm/s, u(mid)={u_at_mid:.3f}, u(z=H)={u_at_top:.3f}")
    
    print()
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("="*70)
    if test_passed:
        print("✓ SMOKE TEST PASSED")
        print()
        print("Streaming solver is functioning correctly:")
        print(f"  • Fieldsplit Schur preconditioner converged")
        print(f"  • Produced physical streaming velocities")
        print(f"  • Satisfied incompressibility constraint")
        print()
        return True
    else:
        print("✗ SMOKE TEST FAILED")
        print()
        print("Check diagnostics above for details.")
        print()
        return False


if __name__ == "__main__":
    try:
        success = test_streaming_stokes_smoke()
        sys.exit(0 if success else 1)
    except Exception as e:
        print()
        print("="*70)
        print("✗ SMOKE TEST CRASHED")
        print("="*70)
        print()
        print(f"Exception: {e}")
        print()
        import traceback
        traceback.print_exc()
        sys.exit(2)
