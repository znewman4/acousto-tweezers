#!/usr/bin/env python3
"""
FEniCSx Multiphysics Acoustic Tweezers Simulation

================================================================================
THIS IS THE SINGLE BLESSED ENTRY POINT FOR RUNNING FEM SIMULATIONS
================================================================================

All FEM is implemented using FEniCSx (dolfinx + PETSc). 
All PDEs are defined in UFL, assembled by dolfinx, solved by PETSc.

QUICK START:
    python scripts/run_fem_multiphysics.py --level ACOUSTICS_ONLY --quick

REQUIRED OUTPUTS (per run):
    results/fem_multiphysics/run_YYYYMMDD_HHMMSS/
    ├── config.json
    ├── summary.csv  
    ├── diagnostics/
    │   ├── sanity_report.txt
    │   ├── mesh_report.txt
    │   ├── solver_report.txt
    │   ├── acoustics_report.txt
    │   ├── interface_residuals.txt (if FLUID_SOLID+)
    │   ├── pml_report.txt (if PML+)
    │   └── actuation_report.txt
    ├── mesh/
    ├── figures/
    │   ├── p_slice.png
    │   └── anim_U_contours.gif (3D runs)
    ├── fields/
    └── logs/
        └── run.log

Physics Levels (prerequisite ladder):
    1. ACOUSTICS_ONLY    - Helmholtz equation in water domain
    2. ACOUSTICS_PML     - + PML boundaries for absorption
    3. FLUID_AIR_BATH    - + Air and bath domains
    4. FLUID_SOLID       - + Elastic solid coupling (plate, walls)
    5. THERMOVISCOUS     - + Boundary layer corrections
    6. STREAMING         - + Acoustic streaming (Stokes flow)
    7. PARTICLES         - + Radiation force & particle dynamics

Author: Acousto-Tweezers Project (FEniCSx Implementation)
Date: January 2026
"""

import argparse
import sys
import logging
from datetime import datetime
from pathlib import Path
import time

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Import from the new FEniCSx-based implementation
from tweezers.fenicsx import (
    FEMConfig,
    PhysicsLevel,
    FEMMultiphysicsSolver,
    run_simulation,
    compute_diagnostics,
    MaterialDatabase,
)
from tweezers.core.io import make_run_dir, get_repo_root
from tweezers.core.logging import setup_logging, get_logger


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='FEniCSx Multiphysics Acoustic Tweezers Simulator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Physics Levels:
  ACOUSTICS_ONLY (1)   - Helmholtz equation in fluids
  ACOUSTICS_PML (2)    - + PML boundary conditions  
  FLUID_AIR_BATH (3)   - + Air and bath domains
  FLUID_SOLID (4)      - + Elastic solid coupling
  THERMOVISCOUS (5)    - + Boundary layer corrections
  STREAMING (6)        - + Acoustic streaming
  PARTICLES (7)        - + Radiation force & particles (default)
"""
    )
    
    # Physics level
    parser.add_argument(
        '--level', '-l',
        type=str,
        default='PARTICLES',
        help='Physics level: name (ACOUSTICS_ONLY, ..., PARTICLES) or number (1-7)'
    )
    
    # Frequency
    parser.add_argument(
        '--frequency', '-f',
        type=float,
        default=2e6,
        help='Acoustic frequency in Hz (default: 2e6 = 2 MHz)'
    )
    
    # Mesh resolution
    parser.add_argument(
        '--ppw',
        type=float,
        default=10.0,
        help='Elements per wavelength (default: 10)'
    )
    
    # Geometry
    parser.add_argument(
        '--dish-diameter',
        type=float,
        default=35e-3,
        help='Petri dish diameter in meters (default: 35 mm)'
    )
    parser.add_argument(
        '--water-depth',
        type=float,
        default=2e-3,
        help='Water depth in meters (default: 2 mm)'
    )
    parser.add_argument(
        '--wall-thickness',
        type=float,
        default=1e-3,
        help='Wall thickness in meters (default: 1 mm)'
    )
    
    # Source
    parser.add_argument(
        '--amplitude', '-a',
        type=float,
        default=1e-9,
        help='Actuation displacement amplitude in meters (default: 1 nm)'
    )
    
    # Particle simulation
    parser.add_argument(
        '--n-particles', '-n',
        type=int,
        default=50,
        help='Number of particles to simulate (default: 50)'
    )
    parser.add_argument(
        '--particle-radius',
        type=float,
        default=5e-6,
        help='Particle radius in meters (default: 5 um)'
    )
    parser.add_argument(
        '--t-final',
        type=float,
        default=1.0,
        help='Simulation time in seconds (default: 1.0)'
    )
    
    # Temperature
    parser.add_argument(
        '--temperature', '-T',
        type=float,
        default=25.0,
        help='Temperature in Celsius (default: 25.0)'
    )
    
    # Quick mode (reduced resolution for testing)
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode: use coarser mesh (ppw=4) for fast testing'
    )
    
    # Output
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output directory (default: results/fem_multiphysics/run_TIMESTAMP)'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save results to disk'
    )
    parser.add_argument(
        '--no-animations',
        action='store_true',
        help='Skip generating GIF animations'
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
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print configuration and exit without running'
    )
    
    return parser.parse_args()


def build_config(args) -> FEMConfig:
    """Build configuration from command-line arguments."""
    
    # Start with default config or load from file
    if args.config is not None:
        config = FEMConfig.from_file(args.config)
    else:
        config = FEMConfig.default()
    
    # Physics level (name or number)
    try:
        if args.level.isdigit():
            config.physics_level = PhysicsLevel(int(args.level))
        else:
            config.physics_level = PhysicsLevel[args.level.upper()]
    except (KeyError, ValueError):
        print(f"ERROR: Invalid physics level '{args.level}'")
        print("Valid levels: " + ", ".join(
            f"{l.name} ({l.value})" for l in PhysicsLevel
        ))
        sys.exit(1)
    
    # Quick mode: coarser mesh + smaller domain for fast testing
    if args.quick:
        args.ppw = 3  # Override PPW (very coarse)
        config.geometry.elements_per_wavelength = 3
        # Reduce domain size for memory
        args.dish_diameter = 10e-3  # 10mm instead of 35mm
        args.water_depth = 1e-3    # 1mm instead of 2mm
        config.geometry.dish_diameter = args.dish_diameter
        config.geometry.water_depth = args.water_depth
        config.geometry.bath_depth = 2e-3  # Smaller bath
        config.geometry.air_height = 3e-3  # Smaller air
        print("[QUICK MODE] Reduced mesh (PPW=3) + smaller domain (10mm dish)")
    
    # Geometry
    config.geometry.dish_diameter = args.dish_diameter
    config.geometry.water_depth = args.water_depth
    config.geometry.wall_thickness = args.wall_thickness
    config.geometry.elements_per_wavelength = args.ppw
    
    # Physics
    config.physics.frequency = args.frequency
    config.physics.actuation_amplitude = args.amplitude
    config.physics.temperature = args.temperature
    config.physics.particle_radius = args.particle_radius
    config.physics.num_particles = args.n_particles
    config.physics.t_max = args.t_final
    
    # Output
    config.output.save_animations = not args.no_animations
    config.solver.verbose = args.verbose
    
    return config


def main():
    """Main entry point."""
    args = parse_args()
    
    # Build configuration
    config = build_config(args)
    
    # Determine output directory using authoritative make_run_dir
    if args.no_save:
        output_dir = None
    elif args.output is not None:
        # Custom output - still enforce structure
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        for subdir in ["diagnostics", "mesh", "figures", "fields", "logs"]:
            (output_dir / subdir).mkdir(exist_ok=True)
    else:
        # Use authoritative run directory creation
        output_dir = make_run_dir(reference_file=__file__)
    
    if output_dir is not None:
        # Setup logging using core module
        logger = setup_logging(output_dir, verbose=not args.quiet, quiet=args.quiet)
    else:
        logger = get_logger("run_fem")
        if not args.quiet:
            logging.basicConfig(level=logging.INFO)
    
    # Print banner
    if not args.quiet:
        print("=" * 70)
        print("  FEniCSx MULTIPHYSICS ACOUSTIC TWEEZERS SIMULATOR")
        print("  " + "=" * 66)
        print(f"  Physics Level: {config.physics_level.name} ({config.physics_level.value})")
        print(f"  Frequency: {config.physics.frequency/1e6:.2f} MHz")
        print(f"  Elements/wavelength: {config.geometry.elements_per_wavelength:.1f}")
        print(f"  Dish diameter: {config.geometry.dish_diameter*1e3:.1f} mm")
        print(f"  Water depth: {config.geometry.water_depth*1e3:.1f} mm")
        if output_dir:
            print(f"  Output: {output_dir}")
        print("=" * 70)
    
    # Verbose config printout
    if args.verbose:
        print("\n" + config.log_summary())
    
    # Save configuration
    if output_dir is not None:
        config.save(str(output_dir / "config.json"))
    
    # Dry run - just print config and exit
    if args.dry_run:
        print("\n[DRY RUN] Configuration printed. Exiting without simulation.")
        return 0
    
    # Run simulation
    logger.info("Starting simulation...")
    start_time = time.time()
    
    try:
        result = run_simulation(config, str(output_dir) if output_dir else None)
        
        total_time = time.time() - start_time
        
        # Compute and save diagnostics
        if output_dir is not None:
            logger.info("Computing diagnostics...")
            materials = MaterialDatabase(config.physics.temperature)
            diagnostics = compute_diagnostics(result, config, materials)
            diagnostics.save(str(output_dir))
            
            if not args.quiet:
                print("\n" + diagnostics.generate_report())
            
            # Check status
            status = diagnostics.overall_status()
            if status == "FAIL":
                logger.warning("Diagnostics checks FAILED")
            elif status == "WARN":
                logger.warning("Diagnostics checks have WARNINGS")
        
        # Generate animations
        if not args.no_animations and output_dir is not None:
            logger.info("Generating visualizations...")
            generate_visualizations(result, output_dir, config)
        
        # Print summary
        if not args.quiet:
            print_result_summary(result, config)
        
        logger.info(f"Simulation complete!")
        if output_dir:
            logger.info(f"Results saved to: {output_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def print_result_summary(result, config):
    """Print summary of simulation results."""
    print("\n" + "-" * 70)
    print("RESULTS SUMMARY")
    print("-" * 70)
    
    if result.acoustic_field is not None:
        p_max = result.acoustic_field.max_pressure
        print(f"  Max pressure amplitude: {p_max:.2e} Pa ({p_max/1e3:.2f} kPa)")
    
    if result.gorkov is not None:
        print(f"  Gor'kov potential range: [{result.gorkov.U_min:.2e}, {result.gorkov.U_max:.2e}] J")
        kT = 1.38e-23 * (config.physics.temperature + 273.15)
        trap_depth = result.gorkov.U_max - result.gorkov.U_min
        print(f"  Trap depth: {trap_depth/kT:.1f} kT")
    
    if result.trajectories is not None:
        print(f"  Particles simulated: {len(result.trajectories)}")
    
    if result.streaming_field is not None:
        print(f"  Max streaming velocity: {result.streaming_field.max_velocity:.2e} m/s")
    
    print("-" * 70)


def generate_visualizations(result, output_dir: Path, config: FEMConfig):
    """Generate visualization plots and animations."""
    try:
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
    except ImportError:
        print("Matplotlib not available, skipping visualizations")
        return
    
    # Pressure field plot
    if result.acoustic_field is not None:
        try:
            p = result.acoustic_field.p
            coords = result.acoustic_field.coords
            
            # Save to figures/ subdirectory
            figures_dir = output_dir / "figures"
            figures_dir.mkdir(exist_ok=True)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Find z-slice at mid-water depth
            z_slice = config.geometry.water_depth / 2
            z_tol = config.geometry.water_depth / 10
            
            mask = np.abs(coords[:, 2] - z_slice) < z_tol
            if np.sum(mask) > 10:
                x_slice = coords[mask, 0]
                y_slice = coords[mask, 1]
                p_slice = np.abs(p[mask])
                
                scatter = ax.scatter(
                    x_slice * 1e3, y_slice * 1e3,
                    c=p_slice, cmap='viridis', s=2
                )
                ax.set_xlabel('x [mm]')
                ax.set_ylabel('y [mm]')
                ax.set_title('Pressure Amplitude |p|')
                ax.set_aspect('equal')
                plt.colorbar(scatter, label='|p| [Pa]')
                
                plt.savefig(figures_dir / "p_slice.png", dpi=150, bbox_inches='tight')
                plt.close()
                print(f"  Saved: {figures_dir / 'p_slice.png'}")
            
            # Generate 3D slice animated GIF (MANDATORY for 3D runs)
            if coords.shape[1] >= 3 and np.max(coords[:, 2]) > np.min(coords[:, 2]) + 1e-6:
                create_3d_slice_animation(result.acoustic_field, figures_dir, config)
                
        except Exception as e:
            print(f"  Could not generate pressure plot: {e}")
            import traceback
            traceback.print_exc()
    
    # Particle trajectories
    if result.trajectories is not None and len(result.trajectories) > 0:
        try:
            figures_dir = output_dir / "figures"
            figures_dir.mkdir(exist_ok=True)
            
            fig, ax = plt.subplots(figsize=(8, 8))
            
            # Draw dish boundary
            theta = np.linspace(0, 2*np.pi, 100)
            R = config.geometry.dish_inner_radius * 1e3
            ax.plot(R * np.cos(theta), R * np.sin(theta), 'k--', linewidth=1, label='Dish')
            
            # Plot trajectories
            for i, traj in enumerate(result.trajectories[:30]):
                ax.plot(traj.x * 1e3, traj.y * 1e3, 'b-', alpha=0.3, linewidth=0.5)
                ax.plot(traj.x[0] * 1e3, traj.y[0] * 1e3, 'go', markersize=3)
                ax.plot(traj.x[-1] * 1e3, traj.y[-1] * 1e3, 'ro', markersize=3)
            
            ax.set_xlabel('x [mm]')
            ax.set_ylabel('y [mm]')
            ax.set_title('Particle Trajectories (green=start, red=end)')
            ax.set_aspect('equal')
            ax.set_xlim(-R*1.1, R*1.1)
            ax.set_ylim(-R*1.1, R*1.1)
            
            plt.savefig(figures_dir / "particle_trajectories.png", dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {figures_dir / 'particle_trajectories.png'}")
            
            # Animated GIF
            if config.output.save_animations:
                create_particle_animation(result.trajectories, output_dir, config)
                
        except Exception as e:
            print(f"  Could not generate trajectory plot: {e}")
    
    # Gor'kov potential
    if result.gorkov is not None:
        try:
            create_gorkov_plot(result.gorkov, output_dir, config)
        except Exception as e:
            print(f"  Could not generate Gor'kov plot: {e}")


def create_particle_animation(trajectories, output_dir: Path, config: FEMConfig):
    """Create animated GIF of particle motion."""
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    trajs = trajectories[:20]  # Limit for clarity
    n_frames = min(50, len(trajs[0].t))
    frame_indices = np.linspace(0, len(trajs[0].t)-1, n_frames, dtype=int)
    
    theta = np.linspace(0, 2*np.pi, 100)
    R = config.geometry.dish_inner_radius * 1e3
    
    def update(frame_idx):
        ax.clear()
        ax.plot(R * np.cos(theta), R * np.sin(theta), 'k--', linewidth=1)
        
        idx = frame_indices[frame_idx]
        for traj in trajs:
            ax.plot(traj.x[:idx+1] * 1e3, traj.y[:idx+1] * 1e3,
                   'b-', alpha=0.3, linewidth=0.5)
            ax.plot(traj.x[idx] * 1e3, traj.y[idx] * 1e3, 'ro', markersize=4)
        
        ax.set_xlim(-R*1.1, R*1.1)
        ax.set_ylim(-R*1.1, R*1.1)
        ax.set_aspect('equal')
        ax.set_xlabel('x [mm]')
        ax.set_ylabel('y [mm]')
        ax.set_title(f'Particle Motion (t = {trajs[0].t[idx]:.3f} s)')
        return []
    
    anim = FuncAnimation(fig, update, frames=n_frames, interval=200)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    anim.save(figures_dir / "anim_particles.gif", writer='pillow', fps=5)
    plt.close()
    print(f"  Saved: {figures_dir / 'anim_particles.gif'}")


def create_3d_slice_animation(acoustic_field, output_dir: Path, config: FEMConfig):
    """
    Create animated GIF showing 3D pressure slices at different z-levels.
    
    This is a MANDATORY deliverable for 3D runs:
    - Choose 2-4 z-slices (z=0.25H, 0.5H, 0.75H)
    - Time-harmonic reconstruction: p(x,t) = Re(p(x) e^{-iωt})
    - Save to figures/anim_U_contours.gif
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from scipy.interpolate import griddata
    
    print("  Generating 3D slice animation...")
    
    p = acoustic_field.p
    coords = acoustic_field.coords
    omega = acoustic_field.omega
    
    # Determine z-levels (water depth range)
    z_min = np.min(coords[:, 2])
    z_max = np.max(coords[:, 2])
    h_water = config.geometry.water_depth
    
    # Use slices at 0.25, 0.5, 0.75 of water depth
    z_slices = [0.25 * h_water, 0.5 * h_water, 0.75 * h_water]
    z_tol = h_water / 20  # Tolerance for finding points
    
    # Number of time frames
    n_frames = 30
    T = 2 * np.pi / omega  # One period
    times = np.linspace(0, T, n_frames, endpoint=False)
    
    # Create regular grid for interpolation
    R = config.geometry.dish_inner_radius
    n_grid = 50
    x_grid = np.linspace(-R, R, n_grid)
    y_grid = np.linspace(-R, R, n_grid)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Mask outside dish
    disk_mask = X**2 + Y**2 <= R**2
    
    # Prepare data for each z-slice
    slice_data = []
    for z_level in z_slices:
        mask = np.abs(coords[:, 2] - z_level) < z_tol
        if np.sum(mask) > 10:
            x_pts = coords[mask, 0]
            y_pts = coords[mask, 1]
            p_pts = p[mask]
            
            # Interpolate to grid
            p_real = griddata((x_pts, y_pts), np.real(p_pts), (X, Y), method='linear', fill_value=0)
            p_imag = griddata((x_pts, y_pts), np.imag(p_pts), (X, Y), method='linear', fill_value=0)
            p_grid = p_real + 1j * p_imag
            
            slice_data.append({
                'z': z_level,
                'p_grid': p_grid,
            })
    
    if len(slice_data) == 0:
        print("    Warning: No valid z-slices found for animation")
        return
    
    # Create figure with subplots for each slice
    n_slices = len(slice_data)
    fig, axes = plt.subplots(1, n_slices, figsize=(5*n_slices, 4))
    if n_slices == 1:
        axes = [axes]
    
    # Find global colorbar limits
    p_max = max(np.max(np.abs(sd['p_grid'])) for sd in slice_data)
    
    def init():
        return []
    
    def update(frame_idx):
        t = times[frame_idx]
        phase = np.exp(-1j * omega * t)
        
        for i, (ax, sd) in enumerate(zip(axes, slice_data)):
            ax.clear()
            
            # Time-harmonic reconstruction: p(x,t) = Re(p(x) * e^{-iωt})
            p_t = np.real(sd['p_grid'] * phase)
            p_t[~disk_mask] = np.nan
            
            im = ax.imshow(
                p_t, extent=[-R*1e3, R*1e3, -R*1e3, R*1e3],
                origin='lower', cmap='RdBu_r',
                vmin=-p_max, vmax=p_max
            )
            
            ax.set_xlabel('x [mm]')
            ax.set_ylabel('y [mm]')
            ax.set_title(f'z = {sd["z"]*1e3:.1f} mm')
            ax.set_aspect('equal')
            
            # Draw dish boundary
            theta = np.linspace(0, 2*np.pi, 100)
            ax.plot(R*1e3*np.cos(theta), R*1e3*np.sin(theta), 'k-', linewidth=1)
        
        fig.suptitle(f'Pressure Field p(x,t) | t = {t*1e6:.1f} μs', fontsize=12)
        plt.tight_layout()
        return []
    
    anim = FuncAnimation(fig, update, init_func=init, frames=n_frames, interval=100, blit=False)
    
    gif_path = output_dir / "anim_U_contours.gif"
    anim.save(gif_path, writer='pillow', fps=config.output.animation_fps)
    plt.close()
    print(f"  Saved: {gif_path}")


def create_gorkov_plot(gorkov, output_dir: Path, config: FEMConfig):
    """Create Gor'kov potential visualization."""
    import numpy as np
    import matplotlib.pyplot as plt
    
    U = gorkov.U
    coords = gorkov.coords
    
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    z_slice = config.geometry.water_depth / 2
    z_tol = config.geometry.water_depth / 10
    
    mask = np.abs(coords[:, 2] - z_slice) < z_tol
    if np.sum(mask) > 10:
        x_slice = coords[mask, 0]
        y_slice = coords[mask, 1]
        U_slice = U[mask]
        
        scatter = ax.scatter(
            x_slice * 1e3, y_slice * 1e3,
            c=U_slice, cmap='RdBu_r', s=2
        )
        ax.set_xlabel('x [mm]')
        ax.set_ylabel('y [mm]')
        ax.set_title("Gor'kov Potential U")
        ax.set_aspect('equal')
        plt.colorbar(scatter, label='U [J]')
        
        plt.savefig(figures_dir / "gorkov_potential.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {figures_dir / 'gorkov_potential.png'}")


if __name__ == '__main__':
    sys.exit(main())
