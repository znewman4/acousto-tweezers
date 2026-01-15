#!/usr/bin/env python3
"""
Path following with Smooth MPC Controller.

Uses the new SmoothMPCController with anti-jitter mechanisms:
- Warm-start from previous best sequence
- Cross-entropy method (CEM) for candidate generation
- Jitter penalty (penalizes ΔΔu)
- Sign reversal penalty
- Optional reference control guidance

Run:
    python scripts/path_follow_smooth.py
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from typing import Optional
import shutil

from acousto.force import ParticleProps, gorkov_potential_and_force_2d
from acousto.analysis import find_trap_center

from tweezers.control import (
    DishDomain, MediumProps, EvaluatorConfig,
    Control2Pucks, BottomFootprint25DEvaluator,
    ControlState, ControlVector, ControlBounds, ControlRateLimits,
    # New smooth controller
    SmoothMPCConfig, SmoothMPCController, ControlHistory,
    plot_control_smoothness,
)

# =============================================================================
# Configuration
# =============================================================================
PATH_SCALE = 0.3       # Scale of path around centroid
T_STEPS = 300          # Number of simulation steps
RENDER_STRIDE = 2      # Render every Nth step for GIF
MAKE_GIF = True        # Whether to create animated GIF
COARSE_GRID = False    # Use coarser grid for faster testing


def make_polyline_path(points: list[tuple[float, float]], T: int) -> np.ndarray:
    """Create smooth path interpolating through waypoints."""
    points = np.array(points)
    n_segments = len(points) - 1
    
    # Compute segment lengths
    seg_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total_length = seg_lengths.sum()
    
    # Allocate timesteps proportional to segment length
    seg_times = (seg_lengths / total_length * T).astype(int)
    seg_times[-1] = T - seg_times[:-1].sum()  # Ensure total is T
    
    path = []
    for i in range(n_segments):
        t_seg = np.linspace(0, 1, seg_times[i], endpoint=(i == n_segments - 1))
        for ti in t_seg:
            px = points[i, 0] + ti * (points[i+1, 0] - points[i, 0])
            py = points[i, 1] + ti * (points[i+1, 1] - points[i, 1])
            path.append([px, py])
    
    return np.array(path)


def main() -> None:
    REPO = Path(__file__).resolve().parents[1]
    RESULTS = REPO / "results"
    frames_dir = RESULTS / "frames_path_follow_smooth"
    out_dir = RESULTS / "path_follow_smooth"
    
    # Clean and recreate
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PATH FOLLOWING WITH SMOOTH MPC CONTROLLER")
    print("=" * 70)

    # ===== Domain + Physics Setup =====
    if COARSE_GRID:
        domain = DishDomain(Lx=2e-3, Ly=2e-3, Nx=80, Ny=80)
    else:
        domain = DishDomain(Lx=2e-3, Ly=2e-3, Nx=120, Ny=120)
    
    medium = MediumProps(f=2e6, c0=1500.0, rho0=1000.0, loss_eta=1e-3, kz=0.0, coupling_alpha=1.0)
    particle = ParticleProps(a=5e-6, rho_p=1050.0, c_p=2350.0)

    # Evaluator config - tuned for reasonable motion
    alpha_g = 1e3  # Moderate scaling
    v_amp = 0.05   # Standard amplitude
    
    cfg = EvaluatorConfig(
        sigma_x=0.10e-3,
        sigma_y=0.15e-3,
        bottom_band=0.25e-3,
        dt=5e-3,
        viscosity=1e-3,
        border_penalty=1e6,
        smooth_u=0.0,
        alpha_g=alpha_g,
        max_step=0.05e-3,
        use_2d_forcing=True,
    )

    ev = BottomFootprint25DEvaluator(domain, medium, particle, cfg)
    
    print(f"\nPhysics config:")
    print(f"  alpha_g = {alpha_g:.0e}")
    print(f"  dt = {cfg.dt*1e3:.1f} ms")
    print(f"  v_amp = {v_amp}")
    print(f"  Grid: {domain.Nx}x{domain.Ny}")

    # ===== Smooth MPC Controller Setup =====
    bounds = ControlBounds(
        x_min=0.0, x_max=domain.Lx,
        y_min=0.0, y_max=cfg.bottom_band,
        v_min=0.0, v_max=0.2,
    )
    
    rate_limits = ControlRateLimits(
        dx_max=0.08e-3,
        dy_max=0.04e-3,
        dv_max=0.01,
        dphi_max=0.3,
    )
    
    # Smooth MPC config with anti-jitter
    mpc_cfg = SmoothMPCConfig(
        horizon=5,
        n_candidates=60,
        position_noise_init=0.03e-3,
        amplitude_noise_init=0.01,
        phase_noise_init=0.2,
        tracking_weight=1e6,
        effort_weight=0.0001,
        # Anti-jitter weights
        jitter_weight=5e3,          # Penalize ΔΔu
        sign_reversal_weight=2e3,   # Penalize direction changes
        reference_weight=0.05,      # Soft constraint toward reference
        # CEM parameters
        cem_iterations=2,
        noise_decay=0.7,
    )
    
    controller = SmoothMPCController(
        evaluator=ev,
        config=mpc_cfg,
        bounds=bounds,
        rate_limits=rate_limits,
    )
    
    print(f"\nSmooth MPC config:")
    print(f"  Horizon: {mpc_cfg.horizon}")
    print(f"  Candidates: {mpc_cfg.n_candidates}")
    print(f"  CEM iterations: {mpc_cfg.cem_iterations}")
    print(f"  Jitter weight: {mpc_cfg.jitter_weight:.0e}")
    print(f"  Sign reversal weight: {mpc_cfg.sign_reversal_weight:.0e}")

    # ===== Desired Path =====
    T = T_STEPS
    
    # Rectangle path
    raw_points = [
        (0.4e-3, 0.6e-3),
        (1.6e-3, 0.6e-3),
        (1.6e-3, 1.4e-3),
        (0.4e-3, 1.4e-3),
        (0.4e-3, 0.6e-3),
    ]
    
    # Scale around centroid
    pts_arr = np.array(raw_points)
    centroid = pts_arr.mean(axis=0)
    
    scaled_points = []
    for px, py in raw_points:
        sx = centroid[0] + PATH_SCALE * (px - centroid[0])
        sy = centroid[1] + PATH_SCALE * (py - centroid[1])
        scaled_points.append((sx, sy))
    
    path = make_polyline_path(points=scaled_points, T=T)
    
    print(f"\nPath configuration:")
    print(f"  PATH_SCALE: {PATH_SCALE}")
    print(f"  T (steps): {T}")
    print(f"  Centroid: ({centroid[0]*1e3:.3f}, {centroid[1]*1e3:.3f}) mm")

    # ===== Initial State =====
    state = ControlState(x=float(path[0, 0]), y=float(path[0, 1]))
    
    initial_control = ControlVector(
        xA=0.6e-3, yA=0.05e-3,
        xB=1.4e-3, yB=0.05e-3,
        vA=v_amp, vB=v_amp,
        phiA=0.0, phiB=np.pi,
        bounds=bounds,
        rate_limits=rate_limits,
    )
    control = initial_control

    # ===== Simulation Loop =====
    traj_xy_mm: list[tuple[float, float]] = [(state.x * 1e3, state.y * 1e3)]
    control_history_raw: list[np.ndarray] = [control.to_array()]
    trap_history: list[tuple[float, float]] = []
    error_history: list[float] = []
    frame_paths: list[Path] = []
    
    print(f"\nStarting simulation...")
    print("-" * 100)
    print(f"{'Step':>5} {'px_mm':>8} {'py_mm':>8} {'err_mm':>8} {'trap_x':>8} {'trap_y':>8} {'xA':>8} {'xB':>8}")
    print("-" * 100)

    for t in range(T - 1):
        # Target for this step
        target = ControlState(x=float(path[t + 1, 0]), y=float(path[t + 1, 1]))
        
        # Future targets for MPC horizon
        horizon_end = min(t + 1 + mpc_cfg.horizon, T)
        targets_horizon = [
            ControlState(x=float(path[i, 0]), y=float(path[i, 1]))
            for i in range(t + 1, horizon_end)
        ]
        
        # Compute reference control (transducers straddling target)
        ref_xA = max(0.1e-3, target.x - 0.5e-3)
        ref_xB = min(domain.Lx - 0.1e-3, target.x + 0.5e-3)
        reference_control = ControlVector(
            xA=ref_xA, yA=0.05e-3,
            xB=ref_xB, yB=0.05e-3,
            vA=v_amp, vB=v_amp,
            phiA=0.0, phiB=np.pi,
            bounds=bounds,
            rate_limits=rate_limits,
        )
        
        # Controller step
        new_control, new_state, info = controller.step(
            state=state,
            target=target,
            current_control=control,
            targets_horizon=targets_horizon,
            reference_control=reference_control,
        )
        
        # Update state
        control = new_control
        state = new_state
        traj_xy_mm.append((state.x * 1e3, state.y * 1e3))
        control_history_raw.append(control.to_array())
        
        # Find trap for diagnostics
        u2p = control.to_control2pucks()
        vb = ev.control_to_forcing_band_vb(u2p)
        field = ev.op.solve_for_bottom_vb(vb)
        U, Fx, Fy = gorkov_potential_and_force_2d(field, particle)
        
        trap_result = find_trap_center(
            field.x, field.y, U, Fx, Fy,
            particle_x=state.x, particle_y=state.y,
            search_radius=0.4e-3,
        )
        trap_history.append((trap_result.x * 1e3, trap_result.y * 1e3))
        
        tracking_err = state.distance_to(target) * 1e3
        error_history.append(tracking_err)
        
        # Print progress
        if t % 20 == 0 or t == T - 2:
            print(f"{t:5d} {state.x*1e3:8.4f} {state.y*1e3:8.4f} {tracking_err:8.4f} "
                  f"{trap_result.x*1e3:8.4f} {trap_result.y*1e3:8.4f} "
                  f"{control.xA*1e3:8.4f} {control.xB*1e3:8.4f}")
        
        # Render frame
        if MAKE_GIF and t % RENDER_STRIDE == 0:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # Left: Field and trajectory
            ax = axes[0]
            x_mm = field.x * 1e3
            y_mm = field.y * 1e3
            X, Y = np.meshgrid(x_mm, y_mm)
            
            ax.contourf(X, Y, U * 1e15, levels=30, cmap='viridis', alpha=0.8)
            
            # Desired path
            ax.plot(path[:, 0] * 1e3, path[:, 1] * 1e3, 'w--', lw=1.5, label='desired')
            
            # Actual trajectory
            traj = np.array(traj_xy_mm)
            ax.plot(traj[:, 0], traj[:, 1], 'c-', lw=2, label='actual')
            
            # Current position
            ax.scatter(state.x * 1e3, state.y * 1e3, s=150, c='red', marker='o', 
                       edgecolors='white', linewidths=2, zorder=10, label='particle')
            
            # Target
            ax.scatter(target.x * 1e3, target.y * 1e3, s=100, c='yellow', marker='*',
                       edgecolors='black', linewidths=1, zorder=9, label='target')
            
            # Trap
            if trap_result.is_stable:
                ax.scatter(trap_result.x * 1e3, trap_result.y * 1e3, s=80, c='lime',
                           marker='x', linewidths=2, zorder=8, label='trap')
            
            # Transducers
            ax.scatter(control.xA * 1e3, control.yA * 1e3, s=100, c='orange', 
                       marker='^', edgecolors='black', label='puck A')
            ax.scatter(control.xB * 1e3, control.yB * 1e3, s=100, c='blue',
                       marker='^', edgecolors='black', label='puck B')
            
            ax.set_xlim(0, domain.Lx * 1e3)
            ax.set_ylim(0, domain.Ly * 1e3)
            ax.set_xlabel('x (mm)')
            ax.set_ylabel('y (mm)')
            ax.set_title(f'Step {t}: Smooth MPC Control')
            ax.set_aspect('equal')
            ax.legend(loc='upper right', fontsize=8)
            
            # Right: Control history
            ax = axes[1]
            ctrl_arr = np.array(control_history_raw)
            steps = np.arange(len(ctrl_arr))
            
            ax.plot(steps, ctrl_arr[:, 0] * 1e3, 'orange', lw=2, label='xA')
            ax.plot(steps, ctrl_arr[:, 2] * 1e3, 'blue', lw=2, label='xB')
            ax.axvline(t, color='red', linestyle='--', alpha=0.5)
            
            ax.set_xlabel('Step')
            ax.set_ylabel('Transducer x (mm)')
            ax.set_title('Control Smoothness')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, T)
            
            plt.tight_layout()
            frame_path = frames_dir / f"frame_{t:04d}.png"
            plt.savefig(frame_path, dpi=100)
            plt.close()
            frame_paths.append(frame_path)

    print("-" * 100)
    
    # ===== Results Summary =====
    traj = np.array(traj_xy_mm)
    errors = np.array(error_history)
    
    print(f"\nResults Summary:")
    print(f"  Mean tracking error: {errors.mean():.4f} mm")
    print(f"  Max tracking error:  {errors.max():.4f} mm")
    print(f"  Final error:         {errors[-1]:.4f} mm")
    
    # Compute control smoothness metrics
    ctrl_arr = np.array(control_history_raw)
    delta_xA = np.diff(ctrl_arr[:, 0])
    delta_xB = np.diff(ctrl_arr[:, 2])
    
    jitter_xA = np.diff(delta_xA)  # ΔΔxA
    jitter_xB = np.diff(delta_xB)  # ΔΔxB
    
    print(f"\nControl Smoothness:")
    print(f"  Mean |ΔxA|: {np.mean(np.abs(delta_xA))*1e6:.2f} µm/step")
    print(f"  Mean |ΔxB|: {np.mean(np.abs(delta_xB))*1e6:.2f} µm/step")
    print(f"  Mean |ΔΔxA| (jitter): {np.mean(np.abs(jitter_xA))*1e6:.2f} µm/step²")
    print(f"  Mean |ΔΔxB| (jitter): {np.mean(np.abs(jitter_xB))*1e6:.2f} µm/step²")
    
    # Sign reversals
    sign_changes_xA = np.sum(np.diff(np.sign(delta_xA)) != 0)
    sign_changes_xB = np.sum(np.diff(np.sign(delta_xB)) != 0)
    print(f"  Sign reversals xA: {sign_changes_xA}")
    print(f"  Sign reversals xB: {sign_changes_xB}")
    
    # Save trajectory
    np.save(out_dir / "traj_xy_mm.npy", traj)
    np.save(out_dir / "desired_xy_mm.npy", path * 1e3)
    np.save(out_dir / "control_history.npy", ctrl_arr)
    np.save(out_dir / "errors.npy", errors)
    
    # Plot control smoothness
    plot_control_smoothness(control_history_raw, out_dir / "control_smoothness.png")
    
    # Create final summary plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Trajectory
    ax = axes[0, 0]
    ax.plot(path[:, 0] * 1e3, path[:, 1] * 1e3, 'k--', lw=2, label='desired')
    ax.plot(traj[:, 0], traj[:, 1], 'b-', lw=2, label='actual')
    ax.scatter(traj[0, 0], traj[0, 1], s=100, c='green', marker='o', label='start')
    ax.scatter(traj[-1, 0], traj[-1, 1], s=100, c='red', marker='s', label='end')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_title('Trajectory')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Tracking error
    ax = axes[0, 1]
    ax.plot(errors, 'b-', lw=1.5)
    ax.axhline(errors.mean(), color='r', linestyle='--', label=f'mean={errors.mean():.4f}')
    ax.set_xlabel('Step')
    ax.set_ylabel('Error (mm)')
    ax.set_title('Tracking Error')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Control positions
    ax = axes[1, 0]
    ax.plot(ctrl_arr[:, 0] * 1e3, 'orange', lw=1.5, label='xA')
    ax.plot(ctrl_arr[:, 2] * 1e3, 'blue', lw=1.5, label='xB')
    ax.set_xlabel('Step')
    ax.set_ylabel('Position (mm)')
    ax.set_title('Transducer Positions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Control deltas (smoothness)
    ax = axes[1, 1]
    ax.plot(delta_xA * 1e6, 'orange', lw=1, alpha=0.7, label='ΔxA')
    ax.plot(delta_xB * 1e6, 'blue', lw=1, alpha=0.7, label='ΔxB')
    ax.axhline(0, color='k', linestyle='-', lw=0.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Δx (µm/step)')
    ax.set_title('Control Rate (Smoothness)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / "summary.png", dpi=150)
    plt.close()
    print(f"\nSaved: {out_dir / 'summary.png'}")
    
    # Create GIF
    if MAKE_GIF and frame_paths:
        print(f"\nCreating GIF from {len(frame_paths)} frames...")
        images = [imageio.imread(str(p)) for p in frame_paths]
        gif_path = out_dir / "path_follow_smooth.gif"
        imageio.mimsave(str(gif_path), images, fps=15, loop=0)
        print(f"Saved: {gif_path}")
    
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
