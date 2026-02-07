#!/usr/bin/env python3
"""
Optimized 3D Helmholtz solver + particle simulation with streaming GIF output.

Features:
  - Memory-efficient: uses float32/complex64 by default
  - Matrix caching: reuses system matrix across time steps
  - Streaming GIF rendering: writes frames one at a time
  - CLI args for all parameters
  - Memory diagnostics and reporting
"""

import argparse
import numpy as np
import sys
import os
from datetime import datetime
import csv

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from tweezers.grid.grid3d import Grid3D
from tweezers.control.field_interface_3d import Helmholtz3DSolver
from tweezers.physics.particle_props import ParticleProps, FluidProps, stokes_mobility, contrast_factors
from tweezers.actuation.lens_fields import lens_focus
from tweezers.actuation.bath_propagation import angular_spectrum_propagate
from tweezers.actuation.plate_transmission import apply_plate_transmission
from tweezers.diagnostics.memory import MemoryTracker, print_memory_banner, memory_checkpoint, array_summary
from tweezers.viz.render_slice_gif import render_trajectory_2d_slice


class DemoConfig:
    """Configuration for the demo run."""
    
    def __init__(self, args):
        # Grid parameters
        self.Lx = args.Lx
        self.Ly = args.Ly
        self.H = args.H
        self.dx = args.dx
        self.dy = args.dy
        self.dz = args.dz
        self.omega_hz = args.omega_hz
        
        # Simulation parameters
        self.n_steps = args.n_steps
        self.dt_s = args.dt_s
        self.render_stride = args.render_stride
        self.slice_z = args.slice_z
        self.gif_fps = args.gif_fps
        
        # Output control
        self.gif = args.gif
        self.save_png_frames = args.save_png_frames
        self.max_ram_mb = args.max_ram_mb
        
        # Solver control
        self.dtype_str = args.dtype
        self.solver_method = args.solver
        self.dtype = np.complex64 if args.dtype == 'single' else np.complex128
        self.dtype_real = np.float32 if args.dtype == 'single' else np.float64
        
        # Feature flags
        self.no_gorkov = args.no_gorkov
        self.no_particle = args.no_particle
        self.no_lens_pipeline = args.no_lens_pipeline
        self.time_varying = args.time_varying
        
        # Derived
        self.omega = 2 * np.pi * self.omega_hz
    
    def compute_grid_points(self):
        """Compute Nx, Ny, Nz from domain and spacing."""
        Nx = int(np.round(self.Lx / self.dx)) + 1
        Ny = int(np.round(self.Ly / self.dy)) + 1
        Nz = int(np.round(self.H / self.dz)) + 1
        return Nx, Ny, Nz


class DemoRunner:
    """Main demo execution class."""
    
    def __init__(self, config):
        self.config = config
        self.mem_tracker = MemoryTracker()
        self.setup_output_dir()
    
    def setup_output_dir(self):
        """Create timestamped output directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = os.path.join(os.path.dirname(__file__), '../results/helmholtz3d_demo')
        self.output_dir = os.path.join(base, f"run_{timestamp}")
        self.frames_dir = os.path.join(self.output_dir, 'frames')
        os.makedirs(self.frames_dir, exist_ok=True)
        print(f"[OUTPUT] Results will be saved to: {self.output_dir}")
    
    def run(self):
        """Execute the demo."""
        print("\n" + "="*80)
        print("3D HELMHOLTZ SOLVER + PARTICLE SIMULATION (MEMORY-OPTIMIZED)")
        print("="*80 + "\n")
        
        # Banner
        Nx, Ny, Nz = self.config.compute_grid_points()
        print_memory_banner(
            f"Grid {Nx}x{Ny}x{Nz}, {self.config.n_steps} steps, {self.config.dtype_str} precision",
            Nx, Ny, Nz, self.config.omega, self.config.dtype_str
        )
        
        # Create grid
        self.mem_tracker.checkpoint("Before grid creation")
        grid = Grid3D(self.config.Lx, self.config.Ly, self.config.H, 
                      self.config.dx, self.config.dy, self.config.dz)
        self.mem_tracker.checkpoint("After grid creation")
        
        # Create solver
        fluid_props = FluidProps(rho0=1000.0, c0=1500.0, eta=1e-3)
        solver = Helmholtz3DSolver(grid, self.config.omega, fluid_props)
        solver.op.dtype = self.config.dtype
        solver.op.solver_method = self.config.solver_method
        self.mem_tracker.checkpoint("After solver creation")
        
        # Example fluid and particle properties
        fluid = FluidProps(rho0=1000.0, c0=1500.0, eta=1e-3)
        particle = ParticleProps(a_m=1e-6, rho_p=1050.0, kappa_p=4.5e-10)
        mu = stokes_mobility(fluid.eta, particle.a_m)
        
        # Setup for time-varying simulation
        if self.config.time_varying:
            self.run_time_varying(grid, solver, fluid, particle, mu)
        else:
            self.run_static_demo(grid, solver, fluid, particle, mu)
        
        # Memory report
        self.mem_tracker.report()
        
        # Summary
        self.print_run_summary()
    
    def run_static_demo(self, grid, solver, fluid, particle, mu):
        """Run a simple static demo (no time evolution)."""
        print("\n[DEMO] Running static field demo (n_steps=1 effectively)...")
        
        X, Y = np.meshgrid(grid.x, grid.y, indexing='ij')
        x0 = (grid.x[0] + grid.x[-1]) / 2
        y0 = (grid.y[0] + grid.y[-1]) / 2
        sigma = 0.005
        
        # Single lens field
        p_lens = lens_focus(X, Y, x0, y0, f=0.01, sigma=sigma, k=solver.k)
        self.mem_tracker.checkpoint("After lens field generation")
        
        # Propagate and transmit
        dx, dy = grid.dx, grid.dy
        dz_bath = 0.002
        p_bath = angular_spectrum_propagate(p_lens, dx, dy, dz_bath, solver.k)
        self.mem_tracker.checkpoint("After bath propagation")
        
        p_bot = apply_plate_transmission(p_bath, dx, dy, solver.k, solver.k * 1500.0 / 2730.0,
                                         1000.0, 1180.0, 1500.0, 2730.0, 0.001)
        self.mem_tracker.checkpoint("After plate transmission")
        
        # Solve
        if not self.config.no_gorkov:
            field = solver.solve(p_bot)
            self.mem_tracker.checkpoint("After Helmholtz solve")
            
            U = field.compute_gorkov_potential(fluid, particle)
            Fx, Fy, Fz = field.compute_radiation_force()
            Fmag = np.sqrt(np.abs(Fx)**2 + np.abs(Fy)**2 + np.abs(Fz)**2)
            
            print(f"[FIELD] U range: [{np.min(U):.3e}, {np.max(U):.3e}]")
            print(f"[FIELD] |F| range: [{np.min(Fmag):.3e}, {np.max(Fmag):.3e}]")
            
            # Save static data
            if not self.config.no_particle:
                z_init = grid.z[1] + 0.1 * (grid.z[-1] - grid.z[1])
                pos = np.array([x0, y0, z_init], dtype=float)
                print(f"[PARTICLE] Initial position: {pos}")
            
            # Save a frame as PNG
            import matplotlib.pyplot as plt
            from tweezers.viz.render_slice_gif import render_slice_frame_to_array
            
            iz_slice = np.argmin(np.abs(grid.z - self.config.slice_z))
            F_slice = Fmag[:, :, iz_slice]
            frame = render_slice_frame_to_array(F_slice, grid.x, grid.y, (x0, y0), z_init,
                                               0.0, 0.0, 0.0, title_suffix="(static)")
            png_path = os.path.join(self.frames_dir, "static_frame.png")
            from PIL import Image
            Image.fromarray(frame).save(png_path)
            print(f"[OUTPUT] Saved frame PNG: {png_path}")
    
    def run_time_varying(self, grid, solver, fluid, particle, mu):
        """Run time-varying simulation with moving lens and streaming GIF."""
        print("\n[DEMO] Running time-varying simulation with moving lens...")
        
        # Moving lens path
        sigma = 0.005
        n_steps = self.config.n_steps
        x_path = np.linspace(grid.x[0]+sigma, grid.x[-1]-sigma, n_steps)
        y_path = np.full(n_steps, (grid.y[0] + grid.y[-1]) / 2)
        
        # Initial particle state
        z_init = grid.z[1] + 0.1 * (grid.z[-1] - grid.z[1])
        pos = np.array([x_path[0], y_path[0], z_init], dtype=self.config.dtype_real)
        
        # Storage for 2D slices (not full 3D)
        store_2d_slices = True
        iz_slice = np.argmin(np.abs(grid.z - self.config.slice_z))
        
        X, Y = np.meshgrid(grid.x, grid.y, indexing='ij')
        traj_data = {'t': [], 'x': [], 'y': [], 'z': [], 'Fx': [], 'Fy': [], 'Fz': [], 'U': []}
        Fmag_2d_frames = []
        
        dt = self.config.dt_s
        dx, dy = grid.dx, grid.dy
        
        print(f"  Simulating {n_steps} steps...")
        
        for i in range(n_steps):
            if i % max(1, n_steps//10) == 0:
                print(f"    Step {i}/{n_steps}")
            
            x0, y0 = x_path[i], y_path[i]
            
            # Lens field
            p_lens = lens_focus(X, Y, x0, y0, f=0.01, sigma=sigma, k=solver.k)
            
            # Propagate and transmit
            dz_bath = 0.002
            p_bath = angular_spectrum_propagate(p_lens, dx, dy, dz_bath, solver.k)
            p_bot = apply_plate_transmission(p_bath, dx, dy, solver.k, solver.k * 1500.0 / 2730.0,
                                             1000.0, 1180.0, 1500.0, 2730.0, 0.001)
            
            # Solve
            if not self.config.no_gorkov:
                field = solver.solve(p_bot)
                U = field.compute_gorkov_potential(fluid, particle)
                Fx, Fy, Fz = field.compute_radiation_force()
                Fmag = np.sqrt(np.abs(Fx)**2 + np.abs(Fy)**2 + np.abs(Fz)**2)
                
                # Store only 2D slice
                Fmag_2d_frames.append(Fmag[:, :, iz_slice].astype(self.config.dtype_real))
                
                # Particle update
                if not self.config.no_particle:
                    ix = np.searchsorted(grid.x, pos[0])
                    iy = np.searchsorted(grid.y, pos[1])
                    iz = np.searchsorted(grid.z, pos[2])
                    ix = np.clip(ix, 0, grid.Nx-1)
                    iy = np.clip(iy, 0, grid.Ny-1)
                    iz = np.clip(iz, 0, grid.Nz-1)
                    
                    Fp = np.array([Fx[ix, iy, iz], Fy[ix, iy, iz], Fz[ix, iy, iz]], dtype=self.config.dtype_real)
                    pos_new = pos + mu * Fp * dt
                    pos_new[0] = np.clip(pos_new[0], grid.x[0], grid.x[-1])
                    pos_new[1] = np.clip(pos_new[1], grid.y[0], grid.y[-1])
                    pos_new[2] = np.clip(pos_new[2], grid.z[0], grid.z[-1])
                    
                    traj_data['t'].append(i * dt)
                    traj_data['x'].append(pos[0])
                    traj_data['y'].append(pos[1])
                    traj_data['z'].append(pos[2])
                    traj_data['Fx'].append(Fp[0])
                    traj_data['Fy'].append(Fp[1])
                    traj_data['Fz'].append(Fp[2])
                    traj_data['U'].append(U[ix, iy, iz])
                    
                    pos = pos_new
        
        self.mem_tracker.checkpoint("After time-varying simulation")
        
        # Save trajectory CSV
        if self.config.no_particle:
            print("[DEMO] Particle simulation skipped (--no_particle)")
        else:
            traj_csv = os.path.join(self.output_dir, 'traj_moving_lens.csv')
            with open(traj_csv, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['t_s', 'x_m', 'y_m', 'z_m', 'Fx_N', 'Fy_N', 'Fz_N', 'U_J'])
                for j in range(len(traj_data['t'])):
                    w.writerow([traj_data['t'][j], traj_data['x'][j], traj_data['y'][j], 
                               traj_data['z'][j], traj_data['Fx'][j], traj_data['Fy'][j], 
                               traj_data['Fz'][j], traj_data['U'][j]])
            print(f"[OUTPUT] Saved trajectory: {traj_csv}")
        
        # Render GIF with streaming writer
        if self.config.gif and Fmag_2d_frames:
            print("[GIF] Rendering with streaming writer...")
            gif_path = os.path.join(self.output_dir, 'particle_slice.gif')
            render_trajectory_2d_slice(
                grid.x, grid.y, grid.z,
                traj_data, Fmag_2d_frames,
                self.config.slice_z, gif_path,
                downsample=self.config.render_stride,
                title_suffix="(moving lens)"
            )
            print(f"[OUTPUT] Saved GIF: {gif_path}")
        
        print(f"[SUMMARY] Rendered {len(Fmag_2d_frames)} force slices")
    
    def print_run_summary(self):
        """Print final run summary."""
        print("\n" + "="*80)
        print("[RUN COMPLETE]")
        print("="*80)
        print(f"Output directory: {self.output_dir}")
        print(f"Configuration:")
        print(f"  Grid: {self.config.compute_grid_points()}")
        print(f"  Frequency: {self.config.omega_hz/1e6:.1f} MHz")
        print(f"  Time steps: {self.config.n_steps}")
        print(f"  Dtype: {self.config.dtype_str}")
        print(f"  Solver: {self.config.solver_method}")
        print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Optimized 3D Helmholtz solver demo")
    
    # Grid parameters
    parser.add_argument('--Lx', type=float, default=0.03, help='Domain width x [m]')
    parser.add_argument('--Ly', type=float, default=0.03, help='Domain width y [m]')
    parser.add_argument('--H', type=float, default=0.01, help='Domain height z [m]')
    parser.add_argument('--dx', type=float, default=0.0003, help='Grid spacing x [m]')
    parser.add_argument('--dy', type=float, default=0.0003, help='Grid spacing y [m]')
    parser.add_argument('--dz', type=float, default=0.0003, help='Grid spacing z [m]')
    
    # Simulation parameters
    parser.add_argument('--omega_hz', type=float, default=1e6, help='Frequency [Hz]')
    parser.add_argument('--n_steps', type=int, default=50, help='Number of time steps')
    parser.add_argument('--dt_s', type=float, default=1e-3, help='Time step [s]')
    parser.add_argument('--render_stride', type=int, default=1, help='Render every N steps')
    parser.add_argument('--slice_z', type=float, default=None, help='z-slice for rendering [m]')
    parser.add_argument('--gif_fps', type=int, default=10, help='GIF frames per second')
    
    # Output control
    parser.add_argument('--gif', type=int, default=1, help='Generate GIF (0/1)')
    parser.add_argument('--save_png_frames', type=int, default=0, help='Save individual frames (0/1)')
    parser.add_argument('--max_ram_mb', type=int, default=2048, help='Max RAM budget [MB]')
    
    # Solver control
    parser.add_argument('--dtype', choices=['single', 'double'], default='single', 
                       help='Floating point precision')
    parser.add_argument('--solver', choices=['direct', 'gmres', 'bicgstab'], default='direct',
                       help='Linear solver method')
    
    # Feature flags
    parser.add_argument('--no_gorkov', action='store_true', help='Skip Gor\'kov potential')
    parser.add_argument('--no_particle', action='store_true', help='Skip particle simulation')
    parser.add_argument('--no_lens_pipeline', action='store_true', help='Skip lens pipeline')
    parser.add_argument('--time_varying', type=int, default=1, help='Time-varying (0/1)')
    
    args = parser.parse_args()
    
    # Use default slice_z if not specified
    if args.slice_z is None:
        args.slice_z = args.H / 2
    
    # Run demo
    config = DemoConfig(args)
    runner = DemoRunner(config)
    runner.run()


if __name__ == "__main__":
    main()
