#!/usr/bin/env python3
"""
Quick FEM multiphysics simulation demo.

Runs full physics ladder (Level 7: PARTICLES) on default 35mm Petri dish.
"""

from tweezers.fem import FEMConfig, PhysicsLevel, FEMMultiphysicsSolver
from pathlib import Path
import json

print("=" * 70)
print("FEM MULTIPHYSICS SIMULATION DEMO")
print("=" * 70)

# Configure simulation
config = FEMConfig.default()
config.physics_level = PhysicsLevel.PARTICLES  # Level 7: full physics
config.geometry.dish_diameter = 0.010  # Small: 10mm for demo speed
config.geometry.water_depth = 0.002  # 2mm water
config.geometry.max_element_size = 0.002  # 2mm elements (coarse)
config.geometry.min_element_size = 0.002

print(f"\n[CONFIG]")
print(f"  Physics Level:     {config.physics_level.name} (Level {config.physics_level.value})")
print(f"  Frequency:         {config.physics.frequency / 1e6:.1f} MHz")
print(f"  Temperature:       {config.physics.temperature:.1f}°C")
print(f"  Dish diameter:     {config.geometry.dish_diameter * 1e3:.1f} mm")
print(f"  Water depth:       {config.geometry.water_depth * 1e3:.1f} mm")
print(f"  Max element size:  {config.geometry.max_element_size * 1e6:.1f} μm")

# Create and run solver
print(f"\n[SOLVER]")
print(f"  Initializing solver...")
solver = FEMMultiphysicsSolver(config)

print(f"  Mesh created:")
print(f"    - Nodes: {solver.mesh.n_nodes}")
print(f"    - Elements: {solver.mesh.n_elements}")
if hasattr(solver.mesh, 'domain_info'):
    print(f"    - Domains: {len(solver.mesh.domain_info)} tagged regions")

print(f"\n  Running simulation...")
result = solver.solve()

print(f"\n[RESULTS]")
print(f"  Result type: {type(result).__name__}")
print(f"  Status: {result.success}")
print(f"  Message: {result.message}")

# Print result attributes
if hasattr(result, '__dict__'):
    print(f"\n  Available fields:")
    for key, value in result.__dict__.items():
        if key.startswith('_'):
            continue
        if value is None:
            print(f"    - {key}: None")
        elif hasattr(value, 'shape'):
            print(f"    - {key}: {type(value).__name__} {value.shape}")
        elif isinstance(value, (int, float)):
            print(f"    - {key}: {type(value).__name__} = {value}")
        else:
            print(f"    - {key}: {type(value).__name__}")

# Save config to results
output_dir = Path(result.output_dir) if result.output_dir else Path("results/fem_demo")
output_dir.mkdir(parents=True, exist_ok=True)

config_dict = {
    "physics_level": config.physics_level.name,
    "frequency_hz": config.physics.frequency,
    "temperature_c": config.physics.temperature,
    "geometry": {
        "dish_diameter_m": config.geometry.dish_diameter,
        "water_depth_m": config.geometry.water_depth,
        "max_element_size_m": config.geometry.max_element_size,
    },
    "mesh": {
        "n_nodes": solver.mesh.n_nodes,
        "n_elements": solver.mesh.n_elements,
    }
}

config_file = output_dir / "config.json"
with open(config_file, 'w') as f:
    json.dump(config_dict, f, indent=2)

print(f"\n[OUTPUT]")
print(f"  Results saved to: {output_dir}")
print(f"  Config saved to:  {config_file}")

print("\n" + "=" * 70)
print("SIMULATION COMPLETE")
print("=" * 70)
