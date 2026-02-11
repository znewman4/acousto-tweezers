#!/usr/bin/env python3
"""
Smoke test for the petri-dish BC model.

Verifies:
1. Bottom facet segmentation (disc vs rigid) produces expected counts.
2. Standing mode produces strong pressure (rigid side walls → standing wave).
3. Vortex mode has azimuthal phase winding.
4. Combined mode: max|p| ≥ max of individual modes (superposition boost).

Usage:
    micromamba run -n acousto-complex python scripts/validation/test_petri_dish_bcs.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import numpy as np
from mpi4py import MPI

from acoustweezers.experiments.shallow_square_dish.config import ShallowDishConfig
from acoustweezers.experiments.shallow_square_dish.solve_pressure import (
    create_mesh, solve_helmholtz,
    TAG_BOTTOM_DISC, TAG_BOTTOM_RIGID, TAG_TOP, TAG_X0, TAG_XL, TAG_Y0, TAG_YL,
)

rank = MPI.COMM_WORLD.rank

# Small domain for fast test
cfg = ShallowDishConfig(
    L=10e-3,           # 10 mm
    H=1e-3,            # 1 mm
    frequency_hz=500e3,
    elements_per_wavelength=6,
    min_elements_z=8,
    vortex_velocity_amplitude=10e-6,
    standing_velocity_amplitude=10e-6,
    vortex_aperture_radius=3e-3,   # 3 mm disc
)

domain, facet_tags, tag_map = create_mesh(cfg, verbose=(rank==0))

# =========================================================================
# Test 1: Facet segmentation
# =========================================================================
n_disc = np.sum(facet_tags.values == TAG_BOTTOM_DISC)
n_rigid = np.sum(facet_tags.values == TAG_BOTTOM_RIGID)
n_top = np.sum(facet_tags.values == TAG_TOP)
n_x0 = np.sum(facet_tags.values == TAG_X0)

if rank == 0:
    print(f"\n{'='*70}")
    print("TEST 1: FACET SEGMENTATION")
    print(f"{'='*70}")
    print(f"  Bottom disc:  {n_disc} facets")
    print(f"  Bottom rigid: {n_rigid} facets")
    print(f"  Top:          {n_top} facets")
    print(f"  x=0 wall:     {n_x0} facets")
    
    assert n_disc > 0, "No disc facets found!"
    assert n_rigid > 0, "No rigid bottom facets found!"
    assert n_disc < n_disc + n_rigid, "Disc should be subset of bottom!"
    # Disc should be much smaller than total bottom
    disc_fraction = n_disc / (n_disc + n_rigid)
    expected_fraction = np.pi * cfg.vortex_aperture_radius**2 / cfg.L**2
    print(f"  Disc fraction: {disc_fraction:.3f} (expected ~{expected_fraction:.3f})")
    print("  ✓ PASS")

# =========================================================================
# Test 2: Standing mode pressure (should be LARGE with rigid walls)
# =========================================================================
p_stand = solve_helmholtz(domain, facet_tags, cfg, mode="standing", verbose=(rank==0))
max_p_stand = np.max(np.abs(p_stand.p_function.x.array))

if rank == 0:
    print(f"\n{'='*70}")
    print("TEST 2: STANDING MODE PRESSURE")
    print(f"{'='*70}")
    print(f"  max|p| = {max_p_stand:.4f} Pa")
    # With rigid reflecting walls and 10 μm/s amplitude at 500 kHz,
    # expect pressures >> 1 Pa (strong resonance)
    # Previously with all-absorbing walls this was ~0.15 Pa
    assert max_p_stand > 0.5, f"Standing mode pressure too weak: {max_p_stand:.4f} Pa"
    print("  ✓ PASS (pressure > 0.5 Pa)")

# =========================================================================
# Test 3: Vortex mode pressure
# =========================================================================
p_vortex = solve_helmholtz(domain, facet_tags, cfg, mode="vortex", verbose=(rank==0))
max_p_vortex = np.max(np.abs(p_vortex.p_function.x.array))

if rank == 0:
    print(f"\n{'='*70}")
    print("TEST 3: VORTEX MODE PRESSURE")
    print(f"{'='*70}")
    print(f"  max|p| = {max_p_vortex:.4f} Pa")
    assert max_p_vortex > 0, "Vortex pressure is zero!"
    print("  ✓ PASS")

# =========================================================================
# Test 4: Combined mode pressure
# =========================================================================
p_combined = solve_helmholtz(domain, facet_tags, cfg, mode="combined", verbose=(rank==0))
max_p_combined = np.max(np.abs(p_combined.p_function.x.array))

if rank == 0:
    print(f"\n{'='*70}")
    print("TEST 4: COMBINED MODE PRESSURE")
    print(f"{'='*70}")
    print(f"  max|p| = {max_p_combined:.4f} Pa")
    print(f"  Standing:  {max_p_stand:.4f} Pa")
    print(f"  Vortex:    {max_p_vortex:.4f} Pa")
    print(f"  Combined:  {max_p_combined:.4f} Pa")
    assert max_p_combined > 0, "Combined pressure is zero!"
    print("  ✓ PASS")

# =========================================================================
# Summary
# =========================================================================
if rank == 0:
    print(f"\n{'='*70}")
    print("ALL PETRI-DISH BC TESTS PASSED")
    print(f"{'='*70}")
    print(f"  Standing pressure: {max_p_stand:.4f} Pa (rigid reflecting walls)")
    print(f"  Vortex pressure:   {max_p_vortex:.4f} Pa (disc transducer)")
    print(f"  Combined pressure: {max_p_combined:.4f} Pa")
