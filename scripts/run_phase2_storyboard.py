#!/usr/bin/env python3
"""
Phase 2 Storyboard Generator

Runs Phase 2 simulations and creates PNG storyboards with:
- Trajectory tails
- Particle labels
- Consistent colorbars
- Both Gor'kov U and pressure |p| plots
"""

import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import json
import csv
from dolfinx import mesh as dmesh
from dolfinx import fem
from dolfinx.fem import Function
from mpi4py import MPI
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from phase2_time_evolution import (
    Phase2Config, ScheduleType, create_fast_box_mesh,
    solve_helmholtz_wrapper, compute_gorkov_midplane
)


def plot_storyboard_frame(
    x_coords, y_coords, U_grid, p_grid, 
    particle_positions, particle_history,
    step, time, phases, field_diag,
    output_dir, schedule_name,
    U_vmin, U_vmax, p_vmin, p_vmax,
    axis_limits
):
    """Generate both Gor'kov and pressure plots for one timestep"""
    
    # Gor'kov plot
    fig, ax = plt.subplots(figsize=(10, 8))
    X, Y = np.meshgrid(x_coords * 1e3, y_coords * 1e3)
    
    im = ax.contourf(X, Y, U_grid, levels=50, cmap='viridis', vmin=U_vmin, vmax=U_vmax)
    cbar = plt.colorbar(im, ax=ax, label="Gor'kov Potential U (J)")
    
    # Draw trajectory tails
    if len(particle_history) > 1:
        tail_length = min(5, len(particle_history))
        for i in range(particle_positions.shape[0]):
            tail_x = []
            tail_y = []
            for hist in particle_history[-tail_length:]:
                tail_x.append(hist[i, 0] * 1e3)
                tail_y.append(hist[i, 1] * 1e3)
            ax.plot(tail_x, tail_y, 'cyan', alpha=0.6, linewidth=2, zorder=9)
    
    # Overlay particles with labels
    particle_xy = particle_positions * 1e3
    ax.scatter(particle_xy[:, 0], particle_xy[:, 1],
               c='red', s=150, marker='o', edgecolors='white', linewidths=2.5,
               zorder=10)
    
    for i, pos in enumerate(particle_xy):
        ax.text(pos[0], pos[1], f'P{i+1}', color='white', fontsize=9,
                ha='center', va='center', fontweight='bold', zorder=11)
    
    # Set fixed limits
    xmin, xmax, ymin, ymax = axis_limits
    ax.set_xlim(xmin * 1e3, xmax * 1e3)
    ax.set_ylim(ymin * 1e3, ymax * 1e3)
    ax.set_aspect('equal')
    
    # Title with full info
    title = (f"{schedule_name} | Step {step} | t = {time:.3f}s\\n"
             f"Phases: L={phases[0]:.3f}, R={phases[1]:.3f}, "
             f"F={phases[2]:.3f}, B={phases[3]:.3f}")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('x (mm)', fontsize=10)
    ax.set_ylabel('y (mm)', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"U_step_{step:03d}.png", dpi=200, bbox_inches='tight')
    plt.close()
    
    # Pressure magnitude plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.contourf(X, Y, p_grid, levels=50, cmap='plasma', vmin=p_vmin, vmax=p_vmax)
    cbar = plt.colorbar(im, ax=ax, label='Pressure Magnitude |p| (Pa)')
    
    # Draw trajectory tails
    if len(particle_history) > 1:
        tail_length = min(5, len(particle_history))
        for i in range(particle_positions.shape[0]):
            tail_x = []
            tail_y = []
            for hist in particle_history[-tail_length:]:
                tail_x.append(hist[i, 0] * 1e3)
                tail_y.append(hist[i, 1] * 1e3)
            ax.plot(tail_x, tail_y, 'cyan', alpha=0.6, linewidth=2, zorder=9)
    
    # Overlay particles with labels
    ax.scatter(particle_xy[:, 0], particle_xy[:, 1],
               c='red', s=150, marker='o', edgecolors='white', linewidths=2.5,
               zorder=10)
    
    for i, pos in enumerate(particle_xy):
        ax.text(pos[0], pos[1], f'P{i+1}', color='white', fontsize=9,
                ha='center', va='center', fontweight='bold', zorder=11)
    
    ax.set_xlim(xmin * 1e3, xmax * 1e3)
    ax.set_ylim(ymin * 1e3, ymax * 1e3)
    ax.set_aspect('equal')
    
    title = (f"{schedule_name} | Step {step} | t = {time:.3f}s\\n"
             f"max|p| = {field_diag['max_p']:.2e} Pa, "
             f"mean|p| = {field_diag['mean_p']:.2e} Pa")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('x (mm)', fontsize=10)
    ax.set_ylabel('y (mm)', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"P_step_{step:03d}.png", dpi=200, bbox_inches='tight')
    plt.close()


def run_storyboard_simulation(schedule_name, T_total, n_steps, save_every, epw=8):
    """Run simulation and generate storyboard"""
    
    print(f"\\n{'='*80}")
    print(f"STORYBOARD GENERATION: {schedule_name}")
    print(f"{'='*80}")
    
    # Create config
    config = Phase2Config(
        schedule_type=ScheduleType(schedule_name),
        T_total=T_total,
        n_steps=n_steps,
        save_every=save_every,
        elements_per_wavelength=epw
    )
    
    # Create output directory with storyboard subfolder
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(f"results/phase2_{schedule_name}/run_{timestamp}")
    run_dir.mkdir(parents=True, exist_ok=True)
    storyboard_dir = run_dir / "storyboard"
    storyboard_dir.mkdir(exist_ok=True)
    
    # Save config
    config_dict = {
        'schedule': schedule_name,
        'T_total': T_total,
        'n_steps': n_steps,
        'save_every': save_every,
        'elements_per_wavelength': epw,
        'timestamp': timestamp
    }
    with open(run_dir / "config.json", 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"Output: {run_dir}")
    print(f"Storyboard: {storyboard_dir}")
    
    # Create mesh
    print("\\n[Mesh] Creating domain...")
    domain, facet_tags = create_fast_box_mesh(config)
    print(f"  Elements: {config.nx}×{config.ny}×{config.nz}")
    
    # Initialize particles
    from phase2_time_evolution import ParticleTracker
    particles = ParticleTracker(config)
    particle_history = [particles.positions.copy()]
    
    # Get schedule function
    from phase2_time_evolution import schedule_step_lr, schedule_ramp_quadrature, schedule_sine_pushpull
    schedule_funcs = {
        'step_lr': schedule_step_lr,
        'ramp_quadrature': schedule_ramp_quadrature,
        'sine_pushpull': schedule_sine_pushpull,
    }
    schedule_func = schedule_funcs[schedule_name]
    
    # Storage for all frames
    all_U_grids = []
    all_p_grids = []
    all_positions = []
    all_times = []
    all_phases = []
    all_field_diags = []
    
    print(f"\\n{'='*70}")
    print("PHASE 1: SOLVING ALL TIMESTEPS")
    print(f"{'='*70}")
    
    # Solve all timesteps first
    from dolfinx.geometry import bb_tree, compute_collisions_points
    
    x_coords = None
    y_coords = None
    
    for step in range(n_steps + 1):
        t = step * config.dt_macro
        phases = schedule_func(t, config.T_total)
        
        print(f"\\n[Step {step}/{n_steps}] t = {t:.4f}s")
        print(f"  Phases: {phases}")
        
        # Solve Helmholtz
        print("  Solving Helmholtz...")
        p_solution, field_diag = solve_helmholtz_wrapper(domain, facet_tags, config, phases)
        print(f"  max|p| = {field_diag['max_p']:.3e} Pa")
        
        # Compute Gor'kov
        print("  Computing Gor'kov...")
        x_coords, y_coords, U_grid = compute_gorkov_midplane(p_solution, domain, config)
        
        # Compute pressure magnitude on same grid
        print("  Computing pressure magnitude...")
        tree = bb_tree(domain, domain.topology.dim)
        points = np.array([[x, y, config.H/2] for y in y_coords for x in x_coords])
        cell_candidates = compute_collisions_points(tree, points)
        
        p_vals = np.zeros(len(points), dtype=complex)
        for i in range(len(points)):
            if len(cell_candidates.links(i)) > 0:
                cell = cell_candidates.links(i)[0]
                p_vals[i] = p_solution.eval(points[i], cell)[0]
        
        p_mag_grid = np.abs(p_vals).reshape((len(y_coords), len(x_coords)))
        
        # Store
        all_U_grids.append(U_grid)
        all_p_grids.append(p_mag_grid)
        all_positions.append(particles.positions.copy())
        all_times.append(t)
        all_phases.append(phases)
        all_field_diags.append(field_diag)
        
        # Advance particles
        if step < n_steps:
            from phase2_time_evolution import compute_force_on_grid
            Fx_grid, Fy_grid = compute_force_on_grid(U_grid, x_coords, y_coords)
            
            for substep in range(config.n_substeps):
                particles.advance(Fx_grid, Fy_grid, x_coords, y_coords, config.dt_substep)
            
            particle_history.append(particles.positions.copy())
    
    # Compute global min/max for consistent colorbars
    print(f"\\n{'='*70}")
    print("PHASE 2: COMPUTING GLOBAL RANGES")
    print(f"{'='*70}")
    
    U_vmin = min(np.min(U) for U in all_U_grids)
    U_vmax = max(np.max(U) for U in all_U_grids)
    p_vmin = min(np.min(p) for p in all_p_grids)
    p_vmax = max(np.max(p) for p in all_p_grids)
    
    print(f"U range: [{U_vmin:.3e}, {U_vmax:.3e}] J")
    print(f"|p| range: [{p_vmin:.3e}, {p_vmax:.3e}] Pa")
    
    axis_limits = (0, config.L, 0, config.L)
    
    # Generate plots
    print(f"\\n{'='*70}")
    print("PHASE 3: GENERATING STORYBOARD FRAMES")
    print(f"{'='*70}")
    
    frames_generated = []
    
    for step in range(n_steps + 1):
        if step % save_every == 0:
            print(f"\\nGenerating frame for step {step}...")
            
            # Build particle history up to this point
            hist_slice = [all_positions[i] for i in range(step + 1)]
            
            plot_storyboard_frame(
                x_coords, y_coords,
                all_U_grids[step], all_p_grids[step],
                all_positions[step], hist_slice,
                step, all_times[step], all_phases[step], all_field_diags[step],
                storyboard_dir, schedule_name,
                U_vmin, U_vmax, p_vmin, p_vmax,
                axis_limits
            )
            
            frames_generated.append(step)
            print(f"  Saved U_step_{step:03d}.png and P_step_{step:03d}.png")
    
    # Create storyboard index
    print(f"\\nCreating storyboard index...")
    with open(storyboard_dir / "storyboard_index.md", 'w') as f:
        f.write(f"# Storyboard: {schedule_name}\\n\\n")
        f.write(f"**Run:** {timestamp}\\n")
        f.write(f"**Schedule:** {schedule_name}\\n")
        f.write(f"**Duration:** {T_total}s over {n_steps} steps\\n")
        f.write(f"**Mesh:** {epw} elements/wavelength\\n")
        f.write(f"**Frames:** {len(frames_generated)}\\n\\n")
        f.write(f"## Key Parameters\\n\\n")
        f.write(f"- Domain: {config.L*1e3:.1f} × {config.L*1e3:.1f} × {config.H*1e3:.1f} mm³\\n")
        f.write(f"- Frequency: {config.frequency*1e-6:.2f} MHz\\n")
        f.write(f"- Particles: {config.num_particles} × {config.particle_radius*1e6:.1f} µm\\n")
        f.write(f"- Gor'kov grid: {len(x_coords)} × {len(y_coords)}\\n\\n")
        f.write(f"## Frames\\n\\n")
        
        for step in frames_generated:
            f.write(f"### Step {step} (t = {all_times[step]:.3f}s)\\n\\n")
            f.write(f"**Phases:** L={all_phases[step][0]:.3f}, R={all_phases[step][1]:.3f}, ")
            f.write(f"F={all_phases[step][2]:.3f}, B={all_phases[step][3]:.3f}\\n\\n")
            f.write(f"![Gor'kov U](U_step_{step:03d}.png)\\n\\n")
            f.write(f"![Pressure |p|](P_step_{step:03d}.png)\\n\\n")
            f.write(f"---\\n\\n")
    
    print(f"\\n{'='*80}")
    print(f"STORYBOARD COMPLETE: {len(frames_generated)} frames generated")
    print(f"Location: {storyboard_dir}")
    print(f"{'='*80}")
    
    return run_dir, storyboard_dir


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Phase 2 storyboards")
    parser.add_argument('--schedule', type=str, required=True,
                        choices=['step_lr', 'ramp_quadrature', 'sine_pushpull'])
    parser.add_argument('--T_total', type=float, required=True)
    parser.add_argument('--n_steps', type=int, required=True)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--epw', type=int, default=8)
    
    args = parser.parse_args()
    
    run_dir, storyboard_dir = run_storyboard_simulation(
        args.schedule, args.T_total, args.n_steps, args.save_every, args.epw
    )
    
    print(f"\\nDone! Check {storyboard_dir}")
