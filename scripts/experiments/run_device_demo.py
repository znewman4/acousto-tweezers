#!/usr/bin/env python3
"""
Device Demo: Shallow Square Dish with Streaming and Particles

Complete workflow for device-realistic 3D acoustic tweezers simulation:
1. Solve standing wave (side wall transducers)
2. Solve vortex (bottom lens transducer)
3. Solve combined field
4. Compute acoustic streaming
5. Compute Gor'kov potential and radiation force
6. Integrate particle trajectories
7. Export all fields to ParaView-ready VTU

Usage:
    # With defaults
    python scripts/shallow_dish/run_device_demo.py
    
    # Custom parameters
    python scripts/shallow_dish/run_device_demo.py \\
      --L 0.05 --H 0.005 --freq 500e3 \\
      --standing_gain 1.0 \\
      --vortex_gain 1.0 --ell 1 --aperture_radius_mm 4 \\
      --out results/device_shallow_custom/

    # Path tracking (vortex moves)
    python scripts/shallow_dish/run_device_demo.py \\
      --vortex_path line --n_steps 20

Author: Acousto-Tweezers Project
Date: 2026-02-08
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parents[2] / 'src'))

from acoustweezers.experiments.shallow_square_dish import (
    ShallowDishConfig,
    solve_all_pressure_cases,
    compute_streaming_velocity,
    integrate_particle_trajectory,
    export_all_fields,
)
from acoustweezers.experiments.shallow_square_dish.solve_pressure import (
    create_mesh, solve_helmholtz,
)
from acoustweezers.experiments.shallow_square_dish.streaming import (
    solve_streaming,
)
from acoustweezers.experiments.shallow_square_dish.particles import (
    compute_gorkov_potential, ParticleDynamics, save_trajectories_csv,
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Shallow dish acoustic tweezers demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Geometry
    parser.add_argument("--L", type=float, default=0.05,
                        help="Dish lateral size [m]")
    parser.add_argument("--H", type=float, default=0.005,
                        help="Dish depth [m]")
    parser.add_argument("--freq", type=float, default=500e3,
                        help="Operating frequency [Hz]")
    
    # Standing wave
    parser.add_argument("--standing_gain", type=float, default=1.0,
                        help="Standing wave amplitude multiplier")
    parser.add_argument("--standing_pattern", type=str, default="antiphase",
                        choices=["antiphase", "quadrature", "inphase"],
                        help="Phase pattern for standing wave transducers")
    
    # Vortex
    parser.add_argument("--vortex_gain", type=float, default=10.0,
                        help="Vortex amplitude multiplier")
    parser.add_argument("--ell", type=int, default=1,
                        help="Vortex topological charge")
    parser.add_argument("--aperture_radius_mm", type=float, default=4.0,
                        help="Vortex aperture radius [mm]")
    
    # Vortex path
    parser.add_argument("--vortex_path", type=str, default="fixed",
                        choices=["fixed", "line", "circle"],
                        help="Vortex motion pattern")
    parser.add_argument("--n_steps", type=int, default=1,
                        help="Number of vortex path steps")
    
    # Particles
    parser.add_argument("--n_particles", type=int, default=5,
                        help="Number of particles to track")
    parser.add_argument("--t_max", type=float, default=0.1,
                        help="Particle integration time [s]")
    
    # Mesh
    parser.add_argument("--elements_per_wavelength", type=int, default=6,
                        help="Mesh density")
    
    # Streaming solver
    parser.add_argument("--streaming_model", type=str, default="stokes",
                        choices=["stokes", "penalty", "skip"],
                        help="Streaming solver model: stokes (Level-2, default), penalty (future), skip (no streaming)")
    parser.add_argument("--streaming_downsample", type=int, default=2,
                        choices=[1, 2, 3],
                        help="Mesh downsampling for streaming: 1=acoustic mesh, 2=coarse (~8x fewer cells in 3D)")
    parser.add_argument("--forcing_scale", type=float, default=1.0,
                        help="Reynolds stress forcing scale factor (for conditioning tests)")
    
    # Output
    parser.add_argument("--out", type=str, default=None,
                        help="Output directory (auto-generated if not specified)")
    
    # Options
    parser.add_argument("--skip_particles", action="store_true",
                        help="Skip particle integration")
    parser.add_argument("--quiet", action="store_true",
                        help="Reduce output verbosity")
    
    return parser.parse_args()


def main():
    """Run the complete device demo workflow."""
    args = parse_args()
    verbose = not args.quiet
    
    # Create output directory
    if args.out is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        repo_root = Path(__file__).parents[2]
        output_dir = repo_root / "results" / f"device_shallow_{timestamp}"
    else:
        output_dir = Path(args.out)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("ACOUSTIC TWEEZERS - SHALLOW DISH DEMO")
    print("="*70)
    print(f"Output: {output_dir}")
    print()
    
    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    cfg = ShallowDishConfig(
        L=args.L,
        H=args.H,
        frequency_hz=args.freq,
        
        # Standing wave
        standing_velocity_amplitude=1e-6 * args.standing_gain,
        standing_phase_pattern=args.standing_pattern,
        
        # Vortex
        vortex_velocity_amplitude=1e-6 * args.vortex_gain,
        vortex_topological_charge=args.ell,
        vortex_aperture_radius=args.aperture_radius_mm * 1e-3,
        
        # Path tracking
        vortex_path_type=args.vortex_path,
        vortex_path_n_steps=args.n_steps,
        
        # Particles
        particle_t_max=args.t_max,
        
        # Mesh
        elements_per_wavelength=args.elements_per_wavelength,
    )
    
    if verbose:
        print(cfg.describe())
    
    # =========================================================================
    # STEP 1: SOLVE PRESSURE FIELDS
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 1: SOLVING PRESSURE FIELDS")
    print("="*70)
    
    # For path tracking, we solve at each vortex position
    vortex_path = cfg.get_vortex_path()
    n_positions = len(vortex_path)
    
    if n_positions == 1:
        # Single position: solve all three cases
        solutions = solve_all_pressure_cases(cfg, verbose=verbose)
        p_combined = solutions["combined"]
    else:
        # Path tracking: solve standing once, then combined at each position
        domain, facet_tags, tag_map = create_mesh(cfg, verbose=verbose)
        
        # Standing is position-independent
        solutions = {}
        solutions["standing"] = solve_helmholtz(
            domain, facet_tags, cfg, mode="standing", verbose=verbose
        )
        
        # Combined at first position (for streaming/gorkov)
        vortex_center = vortex_path[0]
        solutions["combined"] = solve_helmholtz(
            domain, facet_tags, cfg, mode="combined",
            vortex_center=vortex_center, verbose=verbose
        )
        solutions["vortex"] = solve_helmholtz(
            domain, facet_tags, cfg, mode="vortex",
            vortex_center=vortex_center, verbose=verbose
        )
        p_combined = solutions["combined"]
    
    # =========================================================================
    # STEP 2: COMPUTE GOR'KOV POTENTIAL
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 2: COMPUTING GOR'KOV POTENTIAL")
    print("="*70)
    
    gorkov = compute_gorkov_potential(p_combined, verbose=verbose)
    
    # =========================================================================
    # STEP 3: COMPUTE STREAMING (optional, depends on --streaming_model)
    # =========================================================================
    streaming = None
    if args.streaming_model == "skip":
        if verbose:
            print("\n" + "="*70)
            print("STEP 3: STREAMING SKIPPED (--streaming_model skip)")
            print("="*70)
    else:
        print("\n" + "="*70)
        print(f"STEP 3: COMPUTING ACOUSTIC STREAMING ({args.streaming_model})")
        print("="*70)
        
        if args.streaming_model == "stokes":
            # Level-2: Full Stokes with fieldsplit Schur preconditioner
            try:
                from acoustweezers.experiments.shallow_square_dish.streaming import (
                    solve_streaming_stokes,
                )
                
                streaming = solve_streaming_stokes(
                    p_combined,
                    downsample_factor=args.streaming_downsample,
                    forcing_scale=args.forcing_scale,
                    verbose=verbose,
                )
                
                if streaming is None:
                    print(f"\n  WARNING: Stokes streaming solver diverged")
                    print(f"  Check diagnostics in output directory")
                else:
                    if verbose:
                        diags = streaming.diagnostics
                        print(f"\n  Streaming solver converged successfully")
                        print(f"  KSP iterations: {diags.get('ksp_iterations', '?')}")
                        print(f"  Final residual: {diags.get('ksp_final_residual_norm', '?'):.2e}")
                
            except Exception as e:
                print(f"\n  ERROR: Stokes streaming solve failed: {e}")
                print("  Traceback:")
                import traceback
                traceback.print_exc()
                print("  Continuing without streaming...")
                streaming = None
        
        elif args.streaming_model == "penalty":
            print(f"\n  INFO: penalty streaming model not yet implemented")
            print(f"  Falling back to skip")
            streaming = None
    
    # =========================================================================
    # STEP 4: PARTICLE TRAJECTORIES (optional)
    # =========================================================================
    trajectories = []
    if not args.skip_particles:
        print("\n" + "="*70)
        print("STEP 4: INTEGRATING PARTICLE TRAJECTORIES")
        print("="*70)
        
        # Generate initial positions
        L, H = cfg.L, cfg.H
        np.random.seed(42)  # Reproducible
        
        # Start particles in the bulk, away from boundaries
        initial_positions = []
        for i in range(args.n_particles):
            x0 = L * (0.3 + 0.4 * np.random.random())
            y0 = L * (0.3 + 0.4 * np.random.random())
            z0 = H * (0.2 + 0.3 * np.random.random())
            initial_positions.append([x0, y0, z0])
        initial_positions = np.array(initial_positions)
        
        dynamics = ParticleDynamics(gorkov, streaming, cfg)
        
        for i, x0 in enumerate(initial_positions):
            if verbose:
                print(f"\n  Particle {i+1}/{args.n_particles}")
                print(f"    Start: ({x0[0]*1e3:.2f}, {x0[1]*1e3:.2f}, {x0[2]*1e3:.2f}) mm")
            
            traj = dynamics.integrate(x0, method="rk2")
            trajectories.append(traj)
            
            if verbose:
                end = traj.final_position
                print(f"    End:   ({end[0]*1e3:.2f}, {end[1]*1e3:.2f}, {end[2]*1e3:.2f}) mm")
                print(f"    Displacement: {traj.displacement*1e3:.3f} mm")
    
    # =========================================================================
    # STEP 5: EXPORT ALL FIELDS
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 5: EXPORTING PARAVIEW DATA")
    print("="*70)
    
    exported = export_all_fields(
        output_dir=output_dir,
        cfg=cfg,
        solutions=solutions,
        streaming=streaming,
        gorkov=gorkov,
        trajectories=trajectories if len(trajectories) > 0 else None,
        verbose=verbose,
    )
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)
    print(f"\nOutput directory: {output_dir}")
    print("\nFiles created:")
    for name, path in exported.items():
        if path is not None:
            # Handle both Path objects and dicts (from gorkov export)
            if isinstance(path, dict):
                for subname, subpath in path.items():
                    print(f"  - {Path(subpath).name}")
            else:
                print(f"  - {Path(path).name}")
    
    print("\nNext steps:")
    print("  1. Open ParaView")
    print("  2. File → Open → combined_fields.vtu")
    print("  3. See PARAVIEW_README.md for visualization guide")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
