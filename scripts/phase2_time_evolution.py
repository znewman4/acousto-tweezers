#!/usr/bin/env python3
"""
Phase 2: Time-Varying Phase Schedules with Overdamped Particle Motion

Implements quasi-static time evolution:
- At each macro time step: solve Helmholtz for current phases
- Compute Gor'kov force field on midplane
- Advance particle positions using overdamped Stokes dynamics
- Output per-step diagnostics (CSV/JSON) and visualizations (PNG)

Author: Acousto-Tweezers Project
Date: February 2026
"""

import sys
from pathlib import Path

# Add scripts directory to path for imports
scripts_dir = Path(__file__).parent
sys.path.insert(0, str(scripts_dir))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import csv
import argparse
from enum import Enum

# Import Phase 1 solver components
from square_dish_phase_control import (
    solve_helmholtz_square_dish,
    SquareDishConfig as Phase1Config,
    PhaseConfiguration
)

# FEniCSx imports
from mpi4py import MPI
from dolfinx import mesh as dmesh

# Try to import diagnostics utilities
try:
    from diagnostics_utils import find_gorkov_minima_2d
    DIAGNOSTICS_AVAILABLE = True
except ImportError:
    DIAGNOSTICS_AVAILABLE = False
    print("[Warning] diagnostics_utils not available - minima finding disabled")


# ============================================================================
# Fast Mesh Creation (replaces slow gmsh)
# ============================================================================

def create_fast_box_mesh(config: 'Phase2Config'):
    """
    Create box mesh using dolfinx built-in (much faster than gmsh)
    
    Returns
    -------
    domain : dolfinx.mesh.Mesh
    facet_tags : dolfinx.mesh.MeshTags
    """
    
    # Create basic box mesh
    domain = dmesh.create_box(
        MPI.COMM_WORLD,
        [[0.0, 0.0, 0.0], [config.L, config.L, config.H]],
        [config.nx, config.ny, config.nz],
        cell_type=dmesh.CellType.tetrahedron
    )
    
    # Tag boundaries
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    
    def left(x):
        return np.isclose(x[0], 0.0)
    def right(x):
        return np.isclose(x[0], config.L)
    def front(x):
        return np.isclose(x[1], 0.0)
    def back(x):
        return np.isclose(x[1], config.L)
    def bottom(x):
        return np.isclose(x[2], 0.0)
    def top(x):
        return np.isclose(x[2], config.H)
    
    # Locate boundary facets
    facet_markers = {
        1: left,
        2: right,
        3: front,
        4: back,
        5: bottom,
        6: top
    }
    
    facet_indices, facet_values = [], []
    for tag, locator in facet_markers.items():
        facets = dmesh.locate_entities_boundary(domain, fdim, locator)
        facet_indices.append(facets)
        facet_values.append(np.full_like(facets, tag))
    
    facet_indices = np.hstack(facet_indices).astype(np.int32)
    facet_values = np.hstack(facet_values).astype(np.int32)
    
    # Sort and create mesh tags
    sorted_idx = np.argsort(facet_indices)
    facet_tags = dmesh.meshtags(
        domain, fdim,
        facet_indices[sorted_idx],
        facet_values[sorted_idx]
    )
    
    return domain, facet_tags


# ============================================================================
# Phase Schedule System
# ============================================================================

class ScheduleType(Enum):
    """Available phase schedule types"""
    STEP_LR = "step_lr"
    RAMP_QUADRATURE = "ramp_quadrature"
    SINE_PUSHPULL = "sine_pushpull"


def schedule_step_lr(t, T_total):
    """
    Step schedule: switch from uniform to LR opposite at t=T/2
    
    t < T/2:  [0, 0, 0, 0]
    t ≥ T/2:  [0, π, 0, π]  (left-right opposite phase)
    """
    if t < T_total / 2:
        return np.array([0.0, 0.0, 0.0, 0.0])
    else:
        return np.array([0.0, np.pi, 0.0, np.pi])


def schedule_ramp_quadrature(t, T_total):
    """
    Ramp schedule: smoothly transition from uniform to quadrature
    
    φ(t) = α(t) * [0, π/2, π, 3π/2]
    where α(t) = t/T clipped to [0, 1]
    """
    alpha = np.clip(t / T_total, 0.0, 1.0)
    target = np.array([0.0, np.pi/2, np.pi, 3*np.pi/2])
    return alpha * target


def schedule_sine_pushpull(t, T_total):
    """
    Sinusoidal push-pull schedule: LR walls oscillate in anti-phase
    
    φ_L(t) = A * sin(2πt/T)
    φ_R(t) = -A * sin(2πt/T)
    φ_F(t) = 0
    φ_B(t) = 0
    
    where A = π/2
    """
    A = np.pi / 2
    phase_arg = 2 * np.pi * t / T_total
    phi_L = A * np.sin(phase_arg)
    phi_R = -A * np.sin(phase_arg)
    return np.array([phi_L, phi_R, 0.0, 0.0])


def get_schedule(schedule_type: ScheduleType):
    """Get schedule function by type"""
    schedules = {
        ScheduleType.STEP_LR: schedule_step_lr,
        ScheduleType.RAMP_QUADRATURE: schedule_ramp_quadrature,
        ScheduleType.SINE_PUSHPULL: schedule_sine_pushpull,
    }
    return schedules[schedule_type]


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Phase2Config:
    """Configuration for Phase 2 time evolution"""
    
    # Geometry (m)
    L: float = 2.0e-3          # Square dish side length
    H: float = 2.0e-3          # Height
    
    # Physics
    frequency: float = 2.0e6   # Hz
    rho_water: float = 997.0   # kg/m³
    c_water: float = 1497.0    # m/s
    v0: float = 1.0e-3         # m/s (actuation velocity)
    
    # Impedance boundaries
    rho_polystyrene: float = 1050.0  # kg/m³
    c_polystyrene: float = 2350.0    # m/s
    rho_air: float = 1.2             # kg/m³
    c_air: float = 343.0             # m/s
    
    # Mesh
    elements_per_wavelength: float = 12.0
    
    # Particles
    num_particles: int = 5
    particle_radius: float = 40.0e-6   # m
    particle_density: float = 1050.0   # kg/m³ (polystyrene)
    initial_offset: float = 0.25       # Fraction of L for cross pattern
    
    # Time evolution
    T_total: float = 1.0       # Total simulation time (s)
    n_steps: int = 20          # Number of macro time steps
    n_substeps: int = 10       # Substeps per macro step for particle integration
    save_every: int = 1        # Save plots every N steps
    
    # Schedule
    schedule_type: ScheduleType = ScheduleType.RAMP_QUADRATURE
    
    # Safety
    max_particle_speed: float = 10.0e-3  # m/s (clamp to prevent blowups)
    wall_margin: float = 50.0e-6         # m (minimum distance from walls)
    
    # Viscosity
    mu_water: float = 8.9e-4   # Pa·s
    
    def __post_init__(self):
        """Compute derived quantities"""
        self.omega = 2 * np.pi * self.frequency
        self.k_water = self.omega / self.c_water
        self.wavelength = self.c_water / self.frequency
        self.lambda_water = self.wavelength
        
        # Impedances
        self.Z_water = self.rho_water * self.c_water
        self.Z_bottom = self.rho_polystyrene * self.c_polystyrene
        self.Z_air = self.rho_air * self.c_air
        
        # Mesh size
        self.h_target = self.wavelength / self.elements_per_wavelength
        self.nx = int(np.ceil(self.L / self.h_target))
        self.ny = int(np.ceil(self.L / self.h_target))
        self.nz = int(np.ceil(self.H / self.h_target))
        
        # Time step
        self.dt_macro = self.T_total / self.n_steps
        self.dt_substep = self.dt_macro / self.n_substeps
        
        # Stokes mobility: v = μ * F where μ = 1/(6πηa)
        self.stokes_mobility = 1.0 / (6 * np.pi * self.mu_water * self.particle_radius)
        
        # Gor'kov contrast factors (polystyrene in water)
        self.f1 = 1 - (self.rho_water / self.particle_density)
        self.f2 = (self.rho_water * self.c_water**2) / (self.particle_density * self.c_polystyrene**2) - 1
        self.f2 = self.f2 / 3
        
        # Particle volume
        self.particle_volume = (4.0/3.0) * np.pi * self.particle_radius**3
        
        # Dimensionless size
        self.ka = self.k_water * self.particle_radius


# ============================================================================
# Helmholtz Solver (reused from Phase 1)
# ============================================================================

def solve_helmholtz_wrapper(mesh, facet_tags, config: 'Phase2Config', phases):
    """
    Wrapper around Phase 1 solver for Phase 2 time evolution
    
    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh
    facet_tags : dolfinx.mesh.MeshTags
    config : Phase2Config
    phases : array_like
        [phi_left, phi_right, phi_front, phi_back]
    
    Returns
    -------
    p_solution : dolfinx.fem.Function
        Complex pressure field
    diagnostics : dict
        Field scalars (max_p, mean_p, l2_p)
    """
    
    phi_left, phi_right, phi_front, phi_back = phases
    
    # Convert Phase2Config to Phase1Config format
    phase1_config = Phase1Config(
        Lx=config.L,
        Ly=config.L,
        Lz=config.H,
        frequency=config.frequency,
        rho_water=config.rho_water,
        c_water=config.c_water,
        v0_amplitude=config.v0,
        rho_polystyrene=config.rho_polystyrene,
        c_polystyrene=config.c_polystyrene,
        rho_air=config.rho_air,
        c_air=config.c_air,
        elements_per_wavelength=config.elements_per_wavelength,
        num_particles=config.num_particles,
        particle_radius=config.particle_radius,
        particle_density=config.particle_density
    )
    
    # Create PhaseConfiguration
    phase_config = PhaseConfiguration(
        name="current",
        phases=(phi_left, phi_right, phi_front, phi_back),
        description=f"t={phases.tolist()}"
    )
    
    # Call Phase 1 solver with correct argument order
    p_solution, diagnostics = solve_helmholtz_square_dish(
        phase1_config, mesh, facet_tags, phase_config, verbose=False
    )
    
    return p_solution, diagnostics


# ============================================================================
# Gor'kov Potential and Force
# ============================================================================

def compute_gorkov_midplane(p_solution, domain, config: Phase2Config, z_slice=None):
    """
    Compute Gor'kov potential U on midplane
    
    Returns
    -------
    x_coords, y_coords : 1D arrays
    U_grid : 2D array (ny, nx)
    """
    
    if z_slice is None:
        z_slice = config.H / 2
    
    # Create grid for evaluation (50×50 for better resolution)
    nx_eval = 50
    ny_eval = 50
    x_coords = np.linspace(0, config.L, nx_eval)
    y_coords = np.linspace(0, config.L, ny_eval)
    
    # Evaluate pressure on grid
    points = []
    for y in y_coords:
        for x in x_coords:
            points.append([x, y, z_slice])
    points = np.array(points)
    
    # Find cells for all points (dolfinx 0.9 API)
    from dolfinx.geometry import bb_tree, compute_collisions_points
    tree = bb_tree(domain, domain.topology.dim)
    cell_candidates = compute_collisions_points(tree, points)
    
    # Evaluate p at all points
    p_vals = np.zeros(len(points), dtype=complex)
    for i in range(len(points)):
        if len(cell_candidates.links(i)) > 0:
            cell = cell_candidates.links(i)[0]
            p_vals[i] = p_solution.eval(points[i], cell)[0]
    
    p_grid = np.abs(p_vals).reshape((ny_eval, nx_eval))
    
    # Compute Gor'kov potential
    # U = V[(f₁/2)|⟨p⟩|² - (3f₂/4)|⟨v⟩|²]
    # For standing wave: |⟨p⟩|² = |p|²/2, |⟨v⟩|² = |p|²/(2ρ²c²)
    # U ≈ V[f₁|p|²/4 - 3f₂|p|²/(8ρ²c²)]
    
    f1 = config.f1
    f2 = config.f2
    rho = config.rho_water
    c = config.c_water
    V = config.particle_volume
    
    p_mag_sq = np.abs(p_grid)**2
    
    # Gor'kov potential (simplified for standing wave)
    U_grid = V * (f1 * p_mag_sq / 4 - 3 * f2 * p_mag_sq / (8 * rho**2 * c**2))
    
    return x_coords, y_coords, U_grid


def compute_force_on_grid(U_grid, x_coords, y_coords):
    """
    Compute force F = -∇U on 2D grid
    
    Returns
    -------
    Fx_grid, Fy_grid : 2D arrays (ny, nx)
    """
    
    dx = x_coords[1] - x_coords[0]
    dy = y_coords[1] - y_coords[0]
    
    # Central differences
    Fx_grid = -np.gradient(U_grid, dx, axis=1)
    Fy_grid = -np.gradient(U_grid, dy, axis=0)
    
    return Fx_grid, Fy_grid


# ============================================================================
# Particle Dynamics
# ============================================================================

class ParticleTracker:
    """Track 5 particles with overdamped dynamics"""
    
    def __init__(self, config: Phase2Config):
        self.config = config
        self.n_particles = config.num_particles
        
        # Initialize positions (deterministic cross pattern)
        self.positions = self._initialize_positions()
        self.velocities = np.zeros((self.n_particles, 2))
        
        # History
        self.position_history = [self.positions.copy()]
        self.speed_clamp_count = 0
        self.wall_hit_count = 0
    
    def _initialize_positions(self):
        """Create cross pattern: center + 4 cardinal offsets"""
        L = self.config.L
        offset = self.config.initial_offset * L
        
        center = np.array([L/2, L/2])
        positions = [
            center,                           # Center
            center + np.array([offset, 0]),   # Right
            center + np.array([-offset, 0]),  # Left
            center + np.array([0, offset]),   # Top
            center + np.array([0, -offset]),  # Bottom
        ]
        return np.array(positions)
    
    def interpolate_force(self, position, Fx_grid, Fy_grid, x_coords, y_coords):
        """Bilinear interpolation of force at particle position"""
        x, y = position
        
        # Find grid indices
        ix = np.searchsorted(x_coords, x) - 1
        iy = np.searchsorted(y_coords, y) - 1
        
        # Clamp to grid bounds
        ix = np.clip(ix, 0, len(x_coords) - 2)
        iy = np.clip(iy, 0, len(y_coords) - 2)
        
        # Interpolation weights
        x0, x1 = x_coords[ix], x_coords[ix + 1]
        y0, y1 = y_coords[iy], y_coords[iy + 1]
        
        wx = (x - x0) / (x1 - x0) if x1 != x0 else 0.5
        wy = (y - y0) / (y1 - y0) if y1 != y0 else 0.5
        
        # Bilinear interpolation
        Fx = (
            Fx_grid[iy, ix] * (1 - wx) * (1 - wy) +
            Fx_grid[iy, ix + 1] * wx * (1 - wy) +
            Fx_grid[iy + 1, ix] * (1 - wx) * wy +
            Fx_grid[iy + 1, ix + 1] * wx * wy
        )
        
        Fy = (
            Fy_grid[iy, ix] * (1 - wx) * (1 - wy) +
            Fy_grid[iy, ix + 1] * wx * (1 - wy) +
            Fy_grid[iy + 1, ix] * (1 - wx) * wy +
            Fy_grid[iy + 1, ix + 1] * wx * wy
        )
        
        return np.array([Fx, Fy])
    
    def advance(self, Fx_grid, Fy_grid, x_coords, y_coords, dt):
        """
        Advance particles by dt using overdamped dynamics
        
        ẋ = μ * F where μ = 1/(6πηa)
        """
        
        max_speed_this_step = 0.0
        
        for i in range(self.n_particles):
            # Get force at current position
            force = self.interpolate_force(
                self.positions[i], Fx_grid, Fy_grid, x_coords, y_coords
            )
            
            # Overdamped velocity
            velocity = self.config.stokes_mobility * force
            speed = np.linalg.norm(velocity)
            
            # Clamp speed if needed
            if speed > self.config.max_particle_speed:
                velocity = velocity / speed * self.config.max_particle_speed
                speed = self.config.max_particle_speed
                self.speed_clamp_count += 1
            
            max_speed_this_step = max(max_speed_this_step, speed)
            
            # Update position
            self.positions[i] += velocity * dt
            self.velocities[i] = velocity
            
            # Enforce boundaries
            margin = self.config.wall_margin
            if self.positions[i, 0] < margin:
                self.positions[i, 0] = margin
                self.wall_hit_count += 1
            if self.positions[i, 0] > self.config.L - margin:
                self.positions[i, 0] = self.config.L - margin
                self.wall_hit_count += 1
            if self.positions[i, 1] < margin:
                self.positions[i, 1] = margin
                self.wall_hit_count += 1
            if self.positions[i, 1] > self.config.L - margin:
                self.positions[i, 1] = self.config.L - margin
                self.wall_hit_count += 1
        
        self.position_history.append(self.positions.copy())
        
        return max_speed_this_step


# ============================================================================
# Visualization
# ============================================================================

def plot_midplane_with_particles(
    x_coords, y_coords, field_grid, particles, 
    title, output_path, field_label="Field",
    particle_history=None, vmin=None, vmax=None, axis_limits=None
):
    """Plot 2D field with particle positions and trajectory tails overlaid
    
    Parameters
    ----------
    particle_history : list of np.ndarray, optional
        List of past particle positions (each is N×2 array)
        Most recent last. Used to draw trajectory tails.
    vmin, vmax : float, optional
        Fixed colorbar limits for consistency across frames
    axis_limits : tuple, optional
        (xmin, xmax, ymin, ymax) in meters
    """
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Convert to mm for display
    X, Y = np.meshgrid(x_coords * 1e3, y_coords * 1e3)
    
    # Plot field with fixed colorbar if specified
    if vmin is not None and vmax is not None:
        im = ax.contourf(X, Y, field_grid, levels=50, cmap='viridis', vmin=vmin, vmax=vmax)
    else:
        im = ax.contourf(X, Y, field_grid, levels=50, cmap='viridis')
    plt.colorbar(im, ax=ax, label=field_label)
    
    # Draw trajectory tails if history provided
    if particle_history is not None and len(particle_history) > 1:
        # Show last 5 positions as tails
        tail_length = min(5, len(particle_history))
        for i in range(particles.shape[0]):  # For each particle
            tail_x = []
            tail_y = []
            for hist in particle_history[-tail_length:]:
                tail_x.append(hist[i, 0] * 1e3)
                tail_y.append(hist[i, 1] * 1e3)
            ax.plot(tail_x, tail_y, 'c-', alpha=0.5, linewidth=1.5, zorder=9)
    
    # Overlay particles with labels
    particle_xy = particles * 1e3  # Convert to mm
    ax.scatter(particle_xy[:, 0], particle_xy[:, 1], 
               c='red', s=120, marker='o', edgecolors='white', linewidths=2,
               zorder=10)
    
    # Add particle labels
    for i, pos in enumerate(particle_xy):
        ax.text(pos[0], pos[1], f'P{i+1}', color='white', fontsize=8, 
                ha='center', va='center', fontweight='bold', zorder=11)
    
    # Set fixed axis limits if provided
    if axis_limits is not None:
        xmin, xmax, ymin, ymax = axis_limits
        ax.set_xlim(xmin * 1e3, xmax * 1e3)
        ax.set_ylim(ymin * 1e3, ymax * 1e3)
    
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_title(title, fontsize=10)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


# ============================================================================
# Main Simulation Loop
# ============================================================================

def run_phase2_simulation(config: Phase2Config, output_dir: Path):
    """
    Main Phase 2 simulation: time-varying phases with particle motion
    """
    
    print("=" * 70)
    print("PHASE 2: TIME-VARYING SCHEDULES + PARTICLE DYNAMICS")
    print("=" * 70)
    print(f"\nSchedule: {config.schedule_type.value}")
    print(f"Total time: {config.T_total} s")
    print(f"Macro steps: {config.n_steps} (dt = {config.dt_macro:.4f} s)")
    print(f"Substeps per macro: {config.n_substeps}")
    print(f"Particles: {config.num_particles} × {config.particle_radius*1e6:.1f} µm")
    print(f"Stokes mobility: {config.stokes_mobility:.2e} m/(N·s)")
    
    # Create mesh (using fast built-in method instead of slow gmsh)
    print("\n[Mesh] Creating domain (fast method)...")
    domain, facet_tags = create_fast_box_mesh(config)
    print(f"[Mesh] Elements: {config.nx}×{config.ny}×{config.nz}")
    print(f"[Mesh] DOFs: ~{config.nx * config.ny * config.nz * 10}")
    
    # Initialize particles
    particles = ParticleTracker(config)
    print(f"\n[Particles] Initial positions:")
    for i, pos in enumerate(particles.positions):
        print(f"  Particle {i+1}: ({pos[0]*1e3:.3f}, {pos[1]*1e3:.3f}) mm")
    
    # Get schedule function
    schedule_func = get_schedule(config.schedule_type)
    
    # Diagnostics storage
    diagnostics_list = []
    
    # CSV file for per-step data
    csv_path = output_dir / "time_evolution.csv"
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        'step', 'time', 
        'phi_left', 'phi_right', 'phi_front', 'phi_back',
        'max_p', 'mean_p', 'l2_p',
        'deepest_U', 'trap_depth',
        'max_particle_speed', 'speed_clamp_triggered',
        'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'x5', 'y5'
    ])
    
    print("\n" + "=" * 70)
    print("STARTING TIME EVOLUTION")
    print("=" * 70)
    
    # Main time loop
    for step in range(config.n_steps + 1):  # +1 to include t=T_total
        t = step * config.dt_macro
        
        # Get current phases
        phases = schedule_func(t, config.T_total)
        phi_left, phi_right, phi_front, phi_back = phases
        
        print(f"\n[Step {step}/{config.n_steps}] t = {t:.4f} s")
        print(f"  Phases: L={phi_left:.3f}, R={phi_right:.3f}, "
              f"F={phi_front:.3f}, B={phi_back:.3f}")
        
        # Solve Helmholtz for current phases
        print(f"  Solving Helmholtz...")
        p_solution, field_diag = solve_helmholtz_wrapper(domain, facet_tags, config, phases)
        print(f"  max|p| = {field_diag['max_p']:.3e} Pa")
        
        # Compute Gor'kov potential on midplane
        print(f"  Computing Gor'kov potential...")
        x_coords, y_coords, U_grid = compute_gorkov_midplane(p_solution, domain, config)
        
        # Gor'kov diagnostics
        deepest_U = np.min(U_grid)
        shallowest_U = np.max(U_grid)
        trap_depth = shallowest_U - deepest_U
        
        print(f"  Gor'kov: min = {deepest_U:.3e} J, trap depth = {trap_depth:.3e} J")
        
        # Compute force field
        Fx_grid, Fy_grid = compute_force_on_grid(U_grid, x_coords, y_coords)
        
        # Advance particles (with sub-stepping)
        if step < config.n_steps:  # Don't advance on final step
            print(f"  Advancing particles ({config.n_substeps} substeps)...")
            max_speed = 0.0
            speed_clamped = False
            
            for substep in range(config.n_substeps):
                step_max_speed = particles.advance(
                    Fx_grid, Fy_grid, x_coords, y_coords, config.dt_substep
                )
                max_speed = max(max_speed, step_max_speed)
                if particles.speed_clamp_count > 0:
                    speed_clamped = True
            
            print(f"  Max particle speed: {max_speed*1e3:.3f} mm/s")
            if speed_clamped:
                print(f"  [Warning] Speed clamp triggered!")
        else:
            max_speed = 0.0
            speed_clamped = False
        
        # Save diagnostics to CSV
        row = [
            step, t,
            phi_left, phi_right, phi_front, phi_back,
            field_diag['max_p'], field_diag['mean_p'], field_diag['l2_p'],
            deepest_U, trap_depth,
            max_speed, int(speed_clamped)
        ]
        # Add particle positions
        for pos in particles.positions:
            row.extend([pos[0], pos[1]])
        csv_writer.writerow(row)
        
        # Save to diagnostics list
        step_diag = {
            'step': step,
            'time': t,
            'phases': {
                'left': float(phi_left),
                'right': float(phi_right),
                'front': float(phi_front),
                'back': float(phi_back)
            },
            'field': field_diag,
            'gorkov': {
                'deepest_U': float(deepest_U),
                'trap_depth': float(trap_depth)
            },
            'particles': {
                'positions': particles.positions.tolist(),
                'max_speed': float(max_speed),
                'speed_clamped': speed_clamped
            }
        }
        diagnostics_list.append(step_diag)
        
        # Save plots if requested
        if step % config.save_every == 0:
            print(f"  Generating plots...")
            
            # Plot |p| with particles
            from dolfinx.geometry import bb_tree, compute_collisions_points
            p_mag_grid = np.zeros_like(U_grid)
            points = []
            for y in y_coords:
                for x in x_coords:
                    points.append([x, y, config.H/2])
            points = np.array(points)
            
            # Use bb_tree for cell lookup
            tree = bb_tree(domain, domain.topology.dim)
            cell_candidates = compute_collisions_points(tree, points)
            p_vals = np.zeros(len(points), dtype=complex)
            for i in range(len(points)):
                if len(cell_candidates.links(i)) > 0:
                    cell = cell_candidates.links(i)[0]
                    p_vals[i] = p_solution.eval(points[i], cell)[0]
            
            p_mag_grid = np.abs(p_vals).reshape((len(y_coords), len(x_coords)))
            
            plot_midplane_with_particles(
                x_coords, y_coords, p_mag_grid, particles.positions,
                f"Pressure |p| at t={t:.4f}s (step {step})",
                output_dir / f"pressure_step_{step:04d}.png",
                field_label="|p| (Pa)"
            )
            
            # Plot U with particles
            plot_midplane_with_particles(
                x_coords, y_coords, U_grid, particles.positions,
                f"Gor'kov U at t={t:.4f}s (step {step})",
                output_dir / f"gorkov_step_{step:04d}.png",
                field_label="U (J)"
            )
    
    csv_file.close()
    print(f"\n[Output] CSV saved: {csv_path}")
    
    # Save JSON diagnostics
    json_path = output_dir / "time_evolution.json"
    with open(json_path, 'w') as f:
        json.dump({
            'config': {k: str(v) if isinstance(v, (Path, ScheduleType)) else v 
                      for k, v in asdict(config).items()},
            'diagnostics': diagnostics_list,
            'particle_summary': {
                'total_wall_hits': particles.wall_hit_count,
                'total_speed_clamps': particles.speed_clamp_count
            }
        }, f, indent=2)
    print(f"[Output] JSON saved: {json_path}")
    
    print("\n" + "=" * 70)
    print("PHASE 2 COMPLETE")
    print("=" * 70)
    print(f"\nResults: {output_dir}")
    print(f"  {config.n_steps + 1} time steps")
    print(f"  {(config.n_steps // config.save_every) + 1} PNG frames")
    print(f"  Wall hits: {particles.wall_hit_count}")
    print(f"  Speed clamps: {particles.speed_clamp_count}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: Time-varying phase schedules with particle dynamics"
    )
    
    # Schedule
    parser.add_argument('--schedule', type=str, 
                       choices=['step_lr', 'ramp_quadrature', 'sine_pushpull'],
                       default='ramp_quadrature',
                       help='Phase schedule type')
    
    # Time evolution
    parser.add_argument('--T_total', type=float, default=1.0,
                       help='Total simulation time (seconds)')
    parser.add_argument('--n_steps', type=int, default=20,
                       help='Number of macro time steps')
    parser.add_argument('--n_substeps', type=int, default=10,
                       help='Particle integration substeps per macro step')
    parser.add_argument('--save_every', type=int, default=1,
                       help='Save plots every N steps')
    
    # Particles
    parser.add_argument('--particle_radius', type=float, default=40.0e-6,
                       help='Particle radius (m)')
    parser.add_argument('--initial_offset', type=float, default=0.25,
                       help='Initial cross pattern offset (fraction of L)')
    
    # Mesh
    parser.add_argument('--elements_per_wavelength', type=float, default=12.0,
                       help='Mesh resolution')
    
    args = parser.parse_args()
    
    # Create config
    config = Phase2Config(
        schedule_type=ScheduleType(args.schedule),
        T_total=args.T_total,
        n_steps=args.n_steps,
        n_substeps=args.n_substeps,
        save_every=args.save_every,
        particle_radius=args.particle_radius,
        initial_offset=args.initial_offset,
        elements_per_wavelength=args.elements_per_wavelength
    )
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results") / f"phase2_{args.schedule}" / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_path = output_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump({k: str(v) if isinstance(v, (Path, ScheduleType)) else v 
                  for k, v in asdict(config).items()}, f, indent=2)
    print(f"[Config] Saved to: {config_path}")
    
    # Run simulation
    run_phase2_simulation(config, output_dir)


if __name__ == "__main__":
    main()
