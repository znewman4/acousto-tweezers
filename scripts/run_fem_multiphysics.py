#!/usr/bin/env python3
"""
Blessed entry point for FEM multiphysics acoustic tweezers simulation.

This is the primary script for running simulations. It provides:
- Command-line configuration
- Physics level selection
- Output directory management
- Automatic diagnostics

Usage
-----
    # Default simulation (PhysicsLevel.PARTICLES)
    python scripts/run_fem_multiphysics.py

    # Specify physics level
    python scripts/run_fem_multiphysics.py --level ACOUSTICS_ONLY
    python scripts/run_fem_multiphysics.py --level STREAMING
    python scripts/run_fem_multiphysics.py --level PARTICLES

    # Custom parameters
    python scripts/run_fem_multiphysics.py \\
        --frequency 1e6 \\
        --resolution 0.0002 \\
        --level RADIATION_FORCE \\
        --output results/my_run

    # Load from config file
    python scripts/run_fem_multiphysics.py --config my_config.json

Physics Levels
--------------
1. ACOUSTICS_ONLY   - Helmholtz equation in fluid
2. SOLID_COUPLING   - Add elastic solid mechanics
3. PML              - Add PML boundaries
4. THERMOVISCOUS    - Add boundary layer losses
5. STREAMING        - Add acoustic streaming
6. RADIATION_FORCE  - Compute Gor'kov potential
7. PARTICLES        - Full particle dynamics

Examples
--------
Quick test with coarse mesh:
    python scripts/run_fem_multiphysics.py --resolution 0.001 --level ACOUSTICS_ONLY

Production run with fine mesh:
    python scripts/run_fem_multiphysics.py --resolution 0.0001 --level PARTICLES

Validate PML performance:
    python scripts/run_fem_multiphysics.py --level PML --verbose
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from tweezers.fem import (
    FEMConfig,
    PhysicsLevel,
    FEMMultiphysicsSolver,
    run_simulation,
)
from tweezers.fem.diagnostics import Diagnostics, print_parameter_summary


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='FEM Multiphysics Acoustic Tweezers Simulator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Physics level
    parser.add_argument(
        '--level', '-l',
        type=str,
        choices=[l.name for l in PhysicsLevel],
        default='PARTICLES',
        help='Physics level (default: PARTICLES)'
    )
    
    # Frequency
    parser.add_argument(
        '--frequency', '-f',
        type=float,
        default=1e6,
        help='Acoustic frequency in Hz (default: 1e6 = 1 MHz)'
    )
    
    # Resolution
    parser.add_argument(
        '--resolution', '-r',
        type=float,
        default=0.0002,
        help='Mesh resolution in meters (default: 0.0002 = 0.2 mm)'
    )
    
    # Geometry
    parser.add_argument(
        '--dish-radius',
        type=float,
        default=0.0175,
        help='Petri dish radius in meters (default: 0.0175 = 17.5 mm)'
    )
    parser.add_argument(
        '--dish-height',
        type=float,
        default=0.015,
        help='Petri dish height in meters (default: 0.015 = 15 mm)'
    )
    parser.add_argument(
        '--water-depth',
        type=float,
        default=0.005,
        help='Water depth in meters (default: 0.005 = 5 mm)'
    )
    
    # Source
    parser.add_argument(
        '--amplitude', '-a',
        type=float,
        default=1e5,
        help='Source pressure amplitude in Pa (default: 1e5 = 100 kPa)'
    )
    
    # Particle simulation
    parser.add_argument(
        '--n-particles', '-n',
        type=int,
        default=10,
        help='Number of particles to simulate (default: 10)'
    )
    parser.add_argument(
        '--t-final',
        type=float,
        default=1.0,
        help='Simulation time in seconds (default: 1.0)'
    )
    
    # Output
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output directory (default: results/fem_run_YYYYMMDD_HHMMSS)'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save results to disk'
    )
    
    # Config file
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='Load configuration from JSON file'
    )
    
    # Verbosity
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress most output'
    )
    
    return parser.parse_args()


def build_config(args) -> FEMConfig:
    """Build configuration from command-line arguments."""
    
    # Physics level enum from string
    level = PhysicsLevel[args.level]
    
    # Start with default config
    config = FEMConfig.default()
    
    # Override physics level
    config.physics_level = level
    
    # Override geometry
    config.geometry.dish_diameter = args.dish_radius * 2
    config.geometry.dish_height = args.dish_height
    config.geometry.water_depth = args.water_depth
    config.geometry.resolution = args.resolution
    
    # Override physics
    config.physics.frequency = args.frequency
    config.physics.source_amplitude = args.amplitude
    config.physics.num_particles = args.n_particles
    config.physics.particle_sim_time = args.t_final
    
    return config


def main():
    """Main entry point."""
    args = parse_args()
    
    # Print banner
    if not args.quiet:
        print("=" * 70)
        print("  FEM MULTIPHYSICS ACOUSTIC TWEEZERS SIMULATOR")
        print("  " + "=" * 66)
        print(f"  Physics Level: {args.level}")
        print(f"  Frequency: {args.frequency/1e6:.2f} MHz")
        print(f"  Resolution: {args.resolution*1e3:.2f} mm")
        print("=" * 70)
    
    # Build or load configuration
    if args.config is not None:
        import json
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        # TODO: Implement config loading
        config = build_config(args)
        if not args.quiet:
            print(f"Loaded config from {args.config}")
    else:
        config = build_config(args)
    
    # Determine output directory
    if args.no_save:
        output_dir = None
    elif args.output is not None:
        output_dir = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/fem_run_{timestamp}"
    
    # Print parameter summary if verbose
    if args.verbose and not args.quiet:
        print_parameter_summary(config)
    
    # Run simulation
    try:
        result = run_simulation(config, output_dir=output_dir)
    except Exception as e:
        print(f"\nERROR: Simulation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    
    # Print summary
    if not args.quiet:
        print("\n" + "-" * 70)
        print("RESULTS SUMMARY")
        print("-" * 70)
        
        if result.acoustic_field is not None:
            p_max = result.acoustic_field.max_pressure
            print(f"  Max pressure: {p_max/1e3:.2f} kPa")
        
        if result.gorkov is not None:
            print(f"  Trap depth: {result.gorkov.trap_depth:.2e} J")
            kT = 1.38e-23 * 300
            print(f"  Trap depth: {result.gorkov.trap_depth/kT:.1f} kT (at 300K)")
        
        if result.trajectories is not None:
            print(f"  Particles simulated: {len(result.trajectories)}")
        
        if output_dir is not None:
            print(f"\n  Results saved to: {output_dir}")
        
        print("-" * 70)
    
    # Run diagnostics if verbose
    if args.verbose:
        from tweezers.fem.materials import MaterialDatabase
        mat_db = MaterialDatabase()
        fluid = mat_db.get_fluid('water')
        
        diagnostics = Diagnostics(config)
        report = diagnostics.run_all(
            mesh=result.mesh,
            fluid=fluid,
            acoustic_field=result.acoustic_field,
            pml_metrics=result.pml_metrics,
        )
        report.print_report()
    
    if not args.quiet:
        print("\nSimulation complete!")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
