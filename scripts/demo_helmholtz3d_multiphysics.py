#!/usr/bin/env python3
"""
demo_helmholtz3d_multiphysics.py

Complete multiphysics 3D acoustic trapping simulation demonstrating:
- Multi-domain Helmholtz equation with PML
- Explicit fluid/solid domains (water dish, air, plate)
- Acoustic streaming computation
- Gor'kov radiation force and particle dynamics

Usage:
    python demo_helmholtz3d_multiphysics.py [OPTIONS]

Options:
    --frequency FREQ    Acoustic frequency in MHz (default: 2.0)
    --resolution RES    Grid resolution in μm (default: 100)
    --dish-radius R     Dish radius in mm (default: 17.5)
    --water-depth D     Water depth in mm (default: 2.0)
    --n-particles N     Number of particles to simulate (default: 10)
    --duration T        Particle simulation duration in ms (default: 100)
    --no-streaming      Skip streaming computation
    --no-particles      Skip particle simulation
    --output-dir DIR    Output directory (default: auto-generated)
    --quick             Use coarse resolution for quick test
    --verbose           Print detailed progress

Example:
    # Full simulation with default parameters
    python demo_helmholtz3d_multiphysics.py

    # Quick test run
    python demo_helmholtz3d_multiphysics.py --quick

    # Custom parameters
    python demo_helmholtz3d_multiphysics.py --frequency 1.5 --resolution 75 --n-particles 20

Output structure:
    results/helmholtz3d_multiphysics/run_YYYYMMDD_HHMMSS/
    ├── results.npz          # Full simulation data
    ├── pressure_slices.png  # Pressure field visualization
    ├── gorkov_potential.png # Trap potential field
    ├── streaming_field.png  # Streaming velocity
    ├── trajectories_xy.png  # Particle paths (top view)
    ├── trajectories_xz.png  # Particle paths (side view)
    ├── anim_particles_xy.gif # Animated trajectories
    ├── energy_budget.png    # Energy analysis
    └── summary.png          # Comprehensive overview
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="3D Multiphysics Acoustic Trapping Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--frequency", "-f", type=float, default=2.0,
        help="Acoustic frequency in MHz (default: 2.0)"
    )
    parser.add_argument(
        "--resolution", "-r", type=float, default=100.0,
        help="Grid resolution in μm (default: 100)"
    )
    parser.add_argument(
        "--dish-radius", type=float, default=17.5,
        help="Dish radius in mm (default: 17.5)"
    )
    parser.add_argument(
        "--water-depth", type=float, default=2.0,
        help="Water depth in mm (default: 2.0)"
    )
    parser.add_argument(
        "--n-particles", "-n", type=int, default=10,
        help="Number of particles to simulate (default: 10)"
    )
    parser.add_argument(
        "--duration", "-t", type=float, default=100.0,
        help="Particle simulation duration in ms (default: 100)"
    )
    parser.add_argument(
        "--no-streaming", action="store_true",
        help="Skip streaming computation"
    )
    parser.add_argument(
        "--no-particles", action="store_true",
        help="Skip particle simulation"
    )
    parser.add_argument(
        "--output-dir", "-o", type=str, default=None,
        help="Output directory (default: auto-generated)"
    )
    parser.add_argument(
        "--quick", "-q", action="store_true",
        help="Use coarse resolution for quick test"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print detailed progress"
    )
    
    return parser.parse_args()


def main():
    """Run multiphysics simulation."""
    args = parse_args()
    
    # Import after parsing to show help quickly
    try:
        from tweezers.physics import (
            MultiphysicsSolver,
            SimulationParameters,
            ParticleDatabase,
        )
        from tweezers.physics.visualization import (
            MultiphysicsVisualizer,
            create_all_plots,
        )
    except ImportError as e:
        print(f"Error importing modules: {e}")
        print("Make sure tweezers package is installed: pip install -e .")
        sys.exit(1)
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("results/helmholtz3d_multiphysics") / f"run_{timestamp}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Adjust parameters for quick mode
    if args.quick:
        print("\n[QUICK MODE] Using coarse resolution for testing\n")
        resolution = 200.0  # μm
        n_particles = 5
        duration = 20.0  # ms
    else:
        resolution = args.resolution
        n_particles = args.n_particles
        duration = args.duration
    
    # Create simulation parameters
    params = SimulationParameters(
        frequency=args.frequency * 1e6,           # MHz -> Hz
        dish_radius=args.dish_radius * 1e-3,      # mm -> m
        water_depth=args.water_depth * 1e-3,      # mm -> m
        grid_resolution=resolution * 1e-6,        # μm -> m
        temperature=25.0,
    )
    
    print("=" * 60)
    print("MULTIPHYSICS ACOUSTIC TRAPPING SIMULATION")
    print("=" * 60)
    print(f"Frequency:      {params.frequency/1e6:.2f} MHz")
    print(f"Dish radius:    {params.dish_radius*1e3:.1f} mm")
    print(f"Water depth:    {params.water_depth*1e3:.1f} mm")
    print(f"Resolution:     {params.grid_resolution*1e6:.0f} μm")
    print(f"Particles:      {n_particles}")
    print(f"Duration:       {duration:.0f} ms")
    print("=" * 60)
    
    # Create solver
    solver = MultiphysicsSolver(params, verbose=True)
    
    # Generate random initial positions
    if not args.no_particles:
        np.random.seed(42)  # Reproducible results
        
        # Generate positions manually
        r_max = params.dish_radius * 0.8
        # Estimate z-bounds (plate_thickness not in params by default)
        plate_thickness = getattr(params, 'plate_thickness', 1.0e-3)
        z_min = plate_thickness + params.grid_resolution
        z_max = plate_thickness + params.water_depth * 0.9
        
        initial_positions = []
        for _ in range(n_particles):
            r = r_max * np.sqrt(np.random.random())
            theta = 2.0 * np.pi * np.random.random()
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            z = z_min + (z_max - z_min) * np.random.random()
            initial_positions.append([x, y, z])
        initial_positions = np.array(initial_positions)
    else:
        initial_positions = None
    
    # Run simulation
    particle = ParticleDatabase.polystyrene_bead(5.0)  # 5 μm polystyrene
    
    results = solver.solve(
        solve_streaming=not args.no_streaming,
        compute_gorkov=True,
        simulate_particles=not args.no_particles,
        particle=particle,
        initial_positions=initial_positions,
        particle_duration=duration * 1e-3,  # ms -> s
    )
    
    # Save results
    print("\nSaving results...")
    results.save(output_dir / "results.npz")
    
    # Save parameters
    with open(output_dir / "parameters.txt", "w") as f:
        f.write("Simulation Parameters\n")
        f.write("=" * 40 + "\n")
        f.write(f"Frequency: {params.frequency/1e6:.2f} MHz\n")
        f.write(f"Dish radius: {params.dish_radius*1e3:.1f} mm\n")
        f.write(f"Water depth: {params.water_depth*1e3:.1f} mm\n")
        f.write(f"Air height: {params.air_height*1e3:.1f} mm\n")
        f.write(f"Plate thickness: {params.plate_thickness*1e3:.1f} mm\n")
        f.write(f"Grid resolution: {params.grid_resolution*1e6:.0f} μm\n")
        f.write(f"Grid shape: {results.geometry.shape}\n")
        f.write(f"PML thickness: {params.pml_thickness} pts\n")
        f.write(f"Temperature: {params.temperature:.1f} °C\n")
        f.write("\nComputation Times\n")
        f.write("-" * 40 + "\n")
        for stage, t in results.computation_times.items():
            f.write(f"  {stage}: {t:.2f} s\n")
        f.write(f"  TOTAL: {sum(results.computation_times.values()):.2f} s\n")
    
    # Create visualizations
    print("\nCreating visualizations...")
    try:
        create_all_plots(results, output_dir)
    except Exception as e:
        print(f"Warning: Visualization failed: {e}")
        print("Results saved successfully, but plots could not be generated.")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)
    
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
        print(f"Avg distance:     {avg_distance*1e6:.1f} μm")
    
    total_time = sum(results.computation_times.values())
    print(f"Total time:       {total_time:.2f} s")
    print(f"Output:           {output_dir}")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
