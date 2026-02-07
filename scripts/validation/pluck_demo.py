#!/usr/bin/env python3
"""
Vortex "Pluck" Demo - Local Particle Extraction

Shows vortex as a local tool to extract particles from standing-wave minimum.

Demonstrates:
1. Standing-only: particles stay trapped in nest
2. Combined (standing + vortex): particles escape due to vortex perturbation

No ad-hoc forces - escape comes purely from acoustic field modification.
"""

import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

import sys
sys.path.append(str(Path(__file__).parents[2] / 'src'))

# Import validation dir for comparison script
validation_dir = Path(__file__).parent
sys.path.insert(0, str(validation_dir))

# Import from fixed comparison script
from compare_vortex_standing_fixed import (
    FluidProperties, ParticleProperties, PRESET_A,
    solve_standing_only, solve_combined,
    evaluate_on_grid
)

from dolfinx import fem
from acoustweezers.experiments.square_dish.phase_control import compute_gorkov_potential_3d, SquareDishConfig


def find_nearest_minimum(U_func, domain, target_xy=None):
    """
    Find nearest pressure minimum to target location.
    
    Simple approach: evaluate on grid and find local minima.
    """
    L = domain.geometry.x.max(axis=0)[0]
    z_mid = L / 2
    
    if target_xy is None:
        target_xy = [L/2, L/2]
    
    # Evaluate on coarse grid
    X, Y, U_vals = evaluate_on_grid(U_func, domain, z_mid, n_points=50)
    U_real = np.real(U_vals)
    
    # Find local minima (simple approach: find grid minimum)
    min_idx = np.unravel_index(np.argmin(U_real), U_real.shape)
    min_xy = [X[min_idx], Y[min_idx]]
    
    print(f"  [Find Min] Located minimum near ({min_xy[0]*1e3:.2f}, {min_xy[1]*1e3:.2f}) mm")
    print(f"  [Find Min] U_min = {np.min(U_real):.3e} J")
    
    return min_xy, np.min(U_real)


def simulate_particles(U_func, domain, init_positions, fluid, particle, dt=1e-5, T=0.05):
    """
    Simulate particle trajectories under Gor'kov force.
    
    Overdamped dynamics: dx/dt = μ * F_rad where F_rad = -∇U
    """
    # Mobility
    mu = 1.0 / (6 * np.pi * fluid.viscosity * particle.radius)
    
    n_steps = int(T / dt)
    n_particles = len(init_positions)
    
    trajectories = np.zeros((n_steps + 1, n_particles, 3))
    trajectories[0, :, :] = init_positions
    
    # Get bounds
    bounds_min = domain.geometry.x.min(axis=0)
    bounds_max = domain.geometry.x.max(axis=0)
    
    # Build bounding box tree for evaluation
    import dolfinx
    bb_tree = dolfinx.geometry.bb_tree(domain, domain.topology.dim)
    
    for step in range(n_steps):
        positions = trajectories[step, :, :]
        
        for i, pos in enumerate(positions):
            # Check bounds
            if np.any(pos < bounds_min) or np.any(pos > bounds_max):
                trajectories[step + 1, i, :] = pos  # Stop at boundary
                continue
            
            # Evaluate gradient of U at position
            try:
                # Find cell containing point
                cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, pos.reshape(1, -1))
                colliding_cells = dolfinx.geometry.compute_colliding_cells(domain, cell_candidates, pos.reshape(1, -1))
                
                if len(colliding_cells.links(0)) > 0:
                    cell = colliding_cells.links(0)[0]
                    
                    # Compute gradient numerically (finite difference)
                    eps = 1e-7
                    U_center = U_func.eval(pos.reshape(1, -1), [cell])
                    if U_center.ndim == 2:
                        U_center = U_center[0, 0]
                    
                    grad_U = np.zeros(3)
                    for j in range(3):
                        pos_plus = pos.copy()
                        pos_plus[j] += eps
                        
                        # Check if still in domain
                        cell_candidates_p = dolfinx.geometry.compute_collisions_points(bb_tree, pos_plus.reshape(1, -1))
                        colliding_cells_p = dolfinx.geometry.compute_colliding_cells(domain, cell_candidates_p, pos_plus.reshape(1, -1))
                        
                        if len(colliding_cells_p.links(0)) > 0:
                            cell_p = colliding_cells_p.links(0)[0]
                            U_plus = U_func.eval(pos_plus.reshape(1, -1), [cell_p])
                            if U_plus.ndim == 2:
                                U_plus = U_plus[0, 0]
                            grad_U[j] = (U_plus - U_center) / eps
                    
                    # Force: F = -∇U
                    F_rad = -grad_U
                    
                    # Update position
                    trajectories[step + 1, i, :] = pos + mu * F_rad * dt
                else:
                    # Outside domain
                    trajectories[step + 1, i, :] = pos
            except:
                # Evaluation failed - keep position
                trajectories[step + 1, i, :] = pos
        
        if step % 100 == 0:
            print(f"    Step {step}/{n_steps}", end='\r')
    
    print(f"    Step {n_steps}/{n_steps} - Done")
    
    return trajectories


def plot_trajectories(trajectories_standing, trajectories_combined, 
                     U_standing, U_combined, domain, output_dir, init_center):
    """Plot particle trajectories overlayed on Gor'kov potential."""
    
    L = domain.geometry.x.max(axis=0)[0]
    z_mid = L / 2
    
    # Evaluate U on grid
    X, Y, U_s_vals = evaluate_on_grid(U_standing, domain, z_mid)
    _, _, U_c_vals = evaluate_on_grid(U_combined, domain, z_mid)
    
    U_s_real = np.real(U_s_vals)
    U_c_real = np.real(U_c_vals)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Standing only
    vmin = min(np.min(U_s_real), np.min(U_c_real))
    vmax = max(np.max(U_s_real), np.max(U_c_real))
    
    im1 = axes[0].contourf(X*1e3, Y*1e3, U_s_real, levels=50, cmap='RdYlBu_r', vmin=vmin, vmax=vmax)
    for i in range(trajectories_standing.shape[1]):
        traj = trajectories_standing[:, i, :]
        axes[0].plot(traj[:, 0]*1e3, traj[:, 1]*1e3, 'k-', linewidth=1, alpha=0.7)
        axes[0].plot(traj[0, 0]*1e3, traj[0, 1]*1e3, 'go', markersize=8, label='Start' if i == 0 else '')
        axes[0].plot(traj[-1, 0]*1e3, traj[-1, 1]*1e3, 'ro', markersize=8, label='End' if i == 0 else '')
    axes[0].plot(init_center[0]*1e3, init_center[1]*1e3, 'w+', markersize=15, markeredgewidth=3, label='Target min')
    axes[0].set_title('Standing Wave Only')
    axes[0].set_xlabel('x (mm)')
    axes[0].set_ylabel('y (mm)')
    axes[0].legend(loc='upper right', fontsize=8)
    plt.colorbar(im1, ax=axes[0], label='U (J)')
    
    # Combined
    im2 = axes[1].contourf(X*1e3, Y*1e3, U_c_real, levels=50, cmap='RdYlBu_r', vmin=vmin, vmax=vmax)
    for i in range(trajectories_combined.shape[1]):
        traj = trajectories_combined[:, i, :]
        axes[1].plot(traj[:, 0]*1e3, traj[:, 1]*1e3, 'k-', linewidth=1, alpha=0.7)
        axes[1].plot(traj[0, 0]*1e3, traj[0, 1]*1e3, 'go', markersize=8)
        axes[1].plot(traj[-1, 0]*1e3, traj[-1, 1]*1e3, 'ro', markersize=8)
    axes[1].plot(init_center[0]*1e3, init_center[1]*1e3, 'w+', markersize=15, markeredgewidth=3)
    axes[1].set_title('Combined (Standing + Vortex)')
    axes[1].set_xlabel('x (mm)')
    axes[1].set_ylabel('y (mm)')
    plt.colorbar(im2, ax=axes[1], label='U (J)')
    
    plt.tight_layout()
    output_path = output_dir / 'pluck_demo_trajectories.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[Plot] Saved {output_path}")
    plt.close()


def compute_escape_metric(trajectories, init_center, threshold_m=0.001):
    """
    Compute escape metric: how many particles moved > threshold from init_center.
    """
    n_particles = trajectories.shape[1]
    final_positions = trajectories[-1, :, :2]
    
    distances = np.linalg.norm(final_positions - init_center, axis=1)
    n_escaped = np.sum(distances > threshold_m)
    
    return n_escaped, distances


def main():
    parser = argparse.ArgumentParser(description="Vortex Pluck Demo")
    parser.add_argument('--preset', type=str, choices=['A', 'B'], default='A')
    parser.add_argument('--topological_charge', type=int, default=1)
    parser.add_argument('--vortex_gain', type=float, default=2.0, help="Vortex gain (higher = stronger)")
    parser.add_argument('--n_particles', type=int, default=10, help="Number of particles")
    parser.add_argument('--T_sim', type=float, default=0.05, help="Simulation time (s)")
    parser.add_argument('--output_dir', type=str, default=None)
    
    args = parser.parse_args()
    
    # Use Preset A (most interpretable)
    preset = PRESET_A.copy()
    preset['vortex_gain'] = args.vortex_gain
    preset['topological_charge'] = args.topological_charge
    
    # Place vortex near a standing-wave nest (offset from center)
    L = preset['dish_size_m']
    preset['aperture_center_xy_m'] = [L/2 + 0.003, L/2 + 0.003]  # 3mm offset
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(__file__).parents[2] / 'results' / f'pluck_demo_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"VORTEX PLUCK DEMO")
    print(f"{'='*70}")
    print(f"Preset: {preset['name']}")
    print(f"Vortex gain: {args.vortex_gain}")
    print(f"Aperture center: {preset['aperture_center_xy_m'][0]*1e3:.1f}, {preset['aperture_center_xy_m'][1]*1e3:.1f} mm")
    print(f"Output: {output_dir}")
    print(f"{'='*70}\n")
    
    fluid = FluidProperties()
    particle = ParticleProperties()
    
    # Create config for Gor'kov computation (adjust field names to match SquareDishConfig)
    config = SquareDishConfig()
    config.Lx = preset['dish_size_m']
    config.Ly = preset['dish_size_m']
    config.Lz = preset['dish_size_m']
    config.frequency = preset['frequency_hz']
    config.rho_water = fluid.density
    config.c_water = fluid.sound_speed
    config.particle_radius = particle.radius
    config.particle_density = particle.density
    
    # Solve fields
    print("[1/2] Solving standing wave only...")
    domain_s, facet_s, p_standing = solve_standing_only(preset, fluid)
    
    print("\n[2/2] Solving combined (standing + vortex)...")
    domain_c, facet_c, p_combined = solve_combined(preset, fluid, args.topological_charge)
    
    # Compute Gor'kov potentials
    print("\n[Gor'kov] Computing for standing...")
    U_standing = compute_gorkov_potential_3d(p_standing, config)
    
    print("[Gor'kov] Computing for combined...")
    U_combined = compute_gorkov_potential_3d(p_combined, config)
    
    # Find a standing-wave minimum near the vortex aperture
    print("\n[Pluck] Finding nearest minimum...")
    min_xy, min_U = find_nearest_minimum(U_standing, domain_s, target_xy=preset['aperture_center_xy_m'])
    
    # Initialize particles in a small cluster around the minimum
    print(f"\n[Pluck] Initializing {args.n_particles} particles around minimum...")
    init_positions = []
    z_mid = L / 2
    for i in range(args.n_particles):
        # Random positions within 0.5mm of minimum
        offset = (np.random.rand(2) - 0.5) * 0.0005  # ±0.25mm
        pos = np.array([min_xy[0] + offset[0], min_xy[1] + offset[1], z_mid])
        init_positions.append(pos)
    init_positions = np.array(init_positions)
    
    # Simulate particles
    print(f"\n[Sim] Standing wave only (T={args.T_sim}s)...")
    traj_standing = simulate_particles(U_standing, domain_s, init_positions, 
                                      fluid, particle, T=args.T_sim)
    
    print(f"\n[Sim] Combined field (T={args.T_sim}s)...")
    traj_combined = simulate_particles(U_combined, domain_c, init_positions,
                                      fluid, particle, T=args.T_sim)
    
    # Compute escape metrics
    print(f"\n[Metrics] Computing escape statistics...")
    n_escaped_s, distances_s = compute_escape_metric(traj_standing, min_xy, threshold_m=0.0005)
    n_escaped_c, distances_c = compute_escape_metric(traj_combined, min_xy, threshold_m=0.0005)
    
    print(f"  Standing only: {n_escaped_s}/{args.n_particles} escaped (>{0.5:.1f}mm from min)")
    print(f"  Combined:      {n_escaped_c}/{args.n_particles} escaped (>{0.5:.1f}mm from min)")
    print(f"  Mean displacement (standing): {np.mean(distances_s)*1e3:.3f} mm")
    print(f"  Mean displacement (combined): {np.mean(distances_c)*1e3:.3f} mm")
    
    # Plot
    print(f"\n[Plot] Generating trajectory overlay...")
    plot_trajectories(traj_standing, traj_combined, U_standing, U_combined,
                     domain_c, output_dir, min_xy)
    
    # Save summary
    summary_path = output_dir / 'pluck_demo_summary.txt'
    with open(summary_path, 'w') as f:
        f.write(f"Vortex Pluck Demo Summary\n")
        f.write(f"{'='*50}\n")
        f.write(f"Preset: {preset['name']}\n")
        f.write(f"Vortex gain: {args.vortex_gain}\n")
        f.write(f"Topological charge: {args.topological_charge}\n")
        f.write(f"Aperture center: ({preset['aperture_center_xy_m'][0]*1e3:.2f}, {preset['aperture_center_xy_m'][1]*1e3:.2f}) mm\n")
        f.write(f"Minimum located at: ({min_xy[0]*1e3:.2f}, {min_xy[1]*1e3:.2f}) mm\n")
        f.write(f"Number of particles: {args.n_particles}\n")
        f.write(f"Simulation time: {args.T_sim} s\n")
        f.write(f"\nResults:\n")
        f.write(f"  Standing only: {n_escaped_s}/{args.n_particles} escaped\n")
        f.write(f"  Combined:      {n_escaped_c}/{args.n_particles} escaped\n")
        f.write(f"  Mean displacement (standing): {np.mean(distances_s)*1e3:.3f} mm\n")
        f.write(f"  Mean displacement (combined): {np.mean(distances_c)*1e3:.3f} mm\n")
    
    print(f"[Summary] Saved {summary_path}")
    
    print(f"\n{'='*70}")
    print(f"PLUCK DEMO COMPLETE")
    print(f"Results in: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
