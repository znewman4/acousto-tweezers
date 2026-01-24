#!/usr/bin/env python3
"""
Quick test of the multiphysics simulator.
"""
from tweezers.physics import MultiphysicsSolver, SimulationParameters

print("Initializing simulation parameters...")
params = SimulationParameters(
    frequency=2e6,           # 2 MHz
    dish_radius=17.5e-3,     # 17.5 mm
    water_depth=2.0e-3,      # 2 mm
    grid_resolution=200e-6,  # 200 μm (coarse for speed)
    temperature=25.0,
)

print(f"Creating solver with {params.grid_resolution*1e6:.0f} μm resolution...")
solver = MultiphysicsSolver(params, verbose=True)

print("\nRunning multiphysics simulation...")
results = solver.solve(
    solve_streaming=True,
    compute_gorkov=True,
    simulate_particles=True,
)

print("\n" + "="*60)
print("SIMULATION COMPLETE!")
print("="*60)

# Print summary
import numpy as np

p_max = np.max(np.abs(results.acoustic_field.pressure))
print(f"Max pressure:     {p_max:.2e} Pa ({20*np.log10(p_max/20e-6):.1f} dB SPL)")

if results.gorkov_potential is not None:
    U_range = results.gorkov_potential.real.max() - results.gorkov_potential.real.min()
    print(f"Gor'kov range:    {U_range:.2e} J")

if results.streaming_field is not None:
    v_max = np.sqrt(
        results.streaming_field.vx**2 +
        results.streaming_field.vy**2 +
        results.streaming_field.vz**2
    ).max()
    print(f"Max streaming:    {v_max*1e6:.2f} μm/s")

if results.particle_trajectories:
    total_distance = sum(t.distance_traveled for t in results.particle_trajectories)
    avg_distance = total_distance / len(results.particle_trajectories)
    print(f"Particles:        {len(results.particle_trajectories)}")
    print(f"Avg distance:     {avg_distance*1e6:.1f} μm")

total_time = sum(results.computation_times.values())
print(f"Total time:       {total_time:.2f} s")
print("="*60)
