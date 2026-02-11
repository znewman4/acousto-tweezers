#!/usr/bin/env python3
"""
Minimal test for Phase 2 - single step to verify functionality
"""

import sys
from pathlib import Path
scripts_dir = Path(__file__).parent
sys.path.insert(0, str(scripts_dir))

import numpy as np
from phase2_time_evolution import (
    Phase2Config,
    ScheduleType,
    get_schedule,
    ParticleTracker,
    create_square_dish_mesh,
    solve_helmholtz_wrapper,
    compute_gorkov_midplane,
    compute_force_on_grid,
    Phase1Config
)

print("Testing Phase 2 components...")

# Test 1: Schedule functions
print("\n1. Testing schedules...")
config = Phase2Config(T_total=1.0, n_steps=10)
schedule_func = get_schedule(ScheduleType.STEP_LR)
phases_t0 = schedule_func(0.0, 1.0)
phases_t05 = schedule_func(0.5, 1.0)
phases_t1 = schedule_func(1.0, 1.0)
print(f"   step_lr at t=0.0: {phases_t0}")
print(f"   step_lr at t=0.5: {phases_t05}")
print(f"   step_lr at t=1.0: {phases_t1}")
assert np.allclose(phases_t0, [0, 0, 0, 0])
assert np.allclose(phases_t1, [0, np.pi, 0, np.pi])
print("   ✓ Schedules working")

# Test 2: Particle tracker
print("\n2. Testing particle tracker...")
particles = ParticleTracker(config)
print(f"   Initial positions: {particles.positions.shape}")
print(f"   Position 0: {particles.positions[0] * 1e3} mm")
assert particles.positions.shape == (5, 2)
assert np.allclose(particles.positions[0], [config.L/2, config.L/2])
print("   ✓ Particle tracker working")

# Test 3: Mesh creation (this takes time)
print("\n3. Testing mesh creation...")
phase1_config = Phase1Config(
    Lx=config.L,
    Ly=config.L,
    Lz=config.H,
    frequency=config.frequency,
    elements_per_wavelength=config.elements_per_wavelength
)
try:
    domain, facet_tags, cell_tags = create_square_dish_mesh(phase1_config, verbose=False)
    print(f"   Mesh created successfully")
    print(f"   ✓ Mesh creation working")
    
    # Test 4: Solve Helmholtz (if mesh succeeded)
    print("\n4. Testing Helmholtz solver...")
    phases = np.array([0.0, 0.0, 0.0, 0.0])
    p_solution, diagnostics = solve_helmholtz_wrapper(domain, facet_tags, config, phases)
    print(f"   max|p| = {diagnostics['max_p']:.3e} Pa")
    print(f"   mean|p| = {diagnostics['mean_p']:.3e} Pa")
    print(f"   ✓ Helmholtz solver working")
    
    # Test 5: Gor'kov computation
    print("\n5. Testing Gor'kov computation...")
    x_coords, y_coords, U_grid = compute_gorkov_midplane(p_solution, domain, config)
    print(f"   Grid shape: {U_grid.shape}")
    print(f"   U range: [{np.min(U_grid):.3e}, {np.max(U_grid):.3e}] J")
    print(f"   ✓ Gor'kov computation working")
    
    # Test 6: Force computation
    print("\n6. Testing force computation...")
    Fx_grid, Fy_grid = compute_force_on_grid(U_grid, x_coords, y_coords)
    print(f"   Fx range: [{np.min(Fx_grid):.3e}, {np.max(Fx_grid):.3e}] N")
    print(f"   Fy range: [{np.min(Fy_grid):.3e}, {np.max(Fy_grid):.3e}] N")
    print(f"   ✓ Force computation working")
    
    # Test 7: Particle motion
    print("\n7. Testing particle motion...")
    initial_pos = particles.positions.copy()
    max_speed = particles.advance(Fx_grid, Fy_grid, x_coords, y_coords, config.dt_substep)
    displacement = np.linalg.norm(particles.positions - initial_pos, axis=1)
    print(f"   Max speed: {max_speed*1e3:.3f} mm/s")
    print(f"   Max displacement: {np.max(displacement)*1e6:.2f} µm")
    print(f"   ✓ Particle motion working")
    
    print("\n" + "="*70)
    print("ALL TESTS PASSED ✓")
    print("="*70)
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
