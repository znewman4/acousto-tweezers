#!/usr/bin/env python3
"""
Large Circle Demo: 3-Puck Hierarchical Control with Full Visualization.

STAGE 5 FINAL DEMONSTRATION:
Creates a particle trajectory following a large circle (≥60% of domain size)
with all the required features:

- Uses 3 pucks for full 2D control authority
- Particle visibly moves and follows the target path
- Trap center follows target
- Transducers move smoothly (no oscillatory dithering)
- Stability maintained throughout

GIF overlays show:
- Particle position
- Trap center
- Target position
- Transducer positions (3 pucks)
- Stiffness metric

This demo integrates all stages:
- Reachability awareness (Stage 1)
- Macro actions (Stage 2)
- Surrogate prediction (Stage 3, optional)
- Hierarchical planning (Stage 4)

Usage:
    python scripts/demo_large_circle.py
    python scripts/demo_large_circle.py --fast  # Quick test
    python scripts/demo_large_circle.py --fast --record  # With flight recorder
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio.v2 as imageio
from typing import Optional
import shutil
import json

from acousto.force import ParticleProps, gorkov_potential_and_force_2d
from acousto.analysis import find_trap_center

from tweezers.control import (
    DishDomain, MediumProps, EvaluatorConfig,
    Control3Pucks, ControlVector3Pucks, ControlBounds3Pucks, ControlRateLimits3Pucks,
    Evaluator3Pucks,
)
from tweezers.viz.render_3d import normalize_gorkov_field, diagnose_field
from tweezers.diagnostics import FlightRecorder

# Import hierarchical controller
from scripts.hierarchical_controller import (
    HierarchicalController, HierarchicalControllerConfig,
    ReachabilityAwarePlanner, load_reachability_data,
)
from scripts.surrogate_model import GaussianProcessSurrogate


def make_circle_path(
    center_x: float,
    center_y: float,
    radius: float,
    n_points: int,
    n_loops: int = 1,
) -> np.ndarray:
    """Create smooth circular path."""
    theta = np.linspace(0, 2 * np.pi * n_loops, n_points)
    x = center_x + radius * np.cos(theta)
    y = center_y + radius * np.sin(theta)
    return np.column_stack([x, y])


def render_demo_frame(
    frame_path: Path,
    step: int,
    domain: DishDomain,
    field,
    U: np.ndarray,
    particle_x: float,
    particle_y: float,
    target_x: float,
    target_y: float,
    trap_x: float,
    trap_y: float,
    trap_stable: bool,
    ctrl: Control3Pucks,
    path: np.ndarray,
    traj_xy_mm: list[tuple[float, float]],
    stiffness: Optional[np.ndarray],
    control_mode: str,
    errors: list[float],
):
    """Render comprehensive frame with 2D view, 3D view, and diagnostics."""
    fig = plt.figure(figsize=(18, 8))
    
    # === Main 2D view ===
    ax1 = fig.add_subplot(1, 3, 1)
    
    x_mm = field.x * 1e3
    y_mm = field.y * 1e3
    X, Y = np.meshgrid(x_mm, y_mm)
    
    # Potential contours
    U_scaled = U * 1e15
    levels = 30
    cf = ax1.contourf(X, Y, U_scaled, levels=levels, cmap='viridis', alpha=0.8)
    
    # Desired path
    ax1.plot(path[:, 0] * 1e3, path[:, 1] * 1e3, 'w--', lw=2, label='desired path')
    
    # Actual trajectory
    if len(traj_xy_mm) > 1:
        traj = np.array(traj_xy_mm)
        # Gradient color from old (gray) to new (cyan)
        for i in range(len(traj) - 1):
            alpha = (i + 1) / len(traj)
            ax1.plot(traj[i:i+2, 0], traj[i:i+2, 1], 
                    color=(0, 1 - 0.5*alpha, 1), lw=2, alpha=0.5 + 0.5*alpha)
    
    # Current particle position
    ax1.scatter(particle_x * 1e3, particle_y * 1e3, s=250, c='red', marker='o',
                edgecolors='white', linewidths=3, zorder=10, label='particle')
    
    # Target
    ax1.scatter(target_x * 1e3, target_y * 1e3, s=150, c='yellow', marker='*',
                edgecolors='black', linewidths=1.5, zorder=9, label='target')
    
    # Trap center
    if trap_stable and np.isfinite(trap_x) and np.isfinite(trap_y):
        ax1.scatter(trap_x * 1e3, trap_y * 1e3, s=120, c='lime',
                    marker='x', linewidths=3, zorder=8, label='trap center')
    
    # Transducers (3 pucks)
    ax1.scatter(ctrl.xA * 1e3, ctrl.yA * 1e3, s=150, c='orange',
                marker='^', edgecolors='black', linewidths=1.5, label='puck A')
    ax1.scatter(ctrl.xB * 1e3, ctrl.yB * 1e3, s=150, c='blue',
                marker='^', edgecolors='black', linewidths=1.5, label='puck B')
    ax1.scatter(ctrl.xC * 1e3, ctrl.yC * 1e3, s=150, c='magenta',
                marker='^', edgecolors='black', linewidths=1.5, label='puck C')
    
    # Draw lines from pucks to trap
    for px, py in [(ctrl.xA, ctrl.yA), (ctrl.xB, ctrl.yB), (ctrl.xC, ctrl.yC)]:
        ax1.plot([px * 1e3, trap_x * 1e3], [py * 1e3, trap_y * 1e3],
                'w-', lw=0.5, alpha=0.3)
    
    ax1.set_xlim(0, domain.Lx * 1e3)
    ax1.set_ylim(0, domain.Ly * 1e3)
    ax1.set_xlabel('x (mm)')
    ax1.set_ylabel('y (mm)')
    
    stiff_str = ""
    if stiffness is not None:
        stiff_str = f" | κ={np.min(stiffness):.1e}"
    ax1.set_title(f'Step {step}: {control_mode}{stiff_str}')
    ax1.set_aspect('equal')
    ax1.legend(loc='upper right', fontsize=7, ncol=2)
    
    # === 3D view ===
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    
    U_vis, _ = normalize_gorkov_field(U, verbose=False)
    
    step_size = max(1, len(x_mm) // 50)
    X_sub = X[::step_size, ::step_size]
    Y_sub = Y[::step_size, ::step_size]
    U_sub = U_vis[::step_size, ::step_size]
    
    ax2.plot_surface(X_sub, Y_sub, U_sub, cmap='viridis', alpha=0.7,
                     rstride=1, cstride=1, linewidth=0, antialiased=True)
    
    # Particle on surface
    ix = int(np.clip((particle_x * 1e3 - x_mm[0]) / (x_mm[-1] - x_mm[0]) * (len(x_mm) - 1),
                     0, len(x_mm) - 1))
    iy = int(np.clip((particle_y * 1e3 - y_mm[0]) / (y_mm[-1] - y_mm[0]) * (len(y_mm) - 1),
                     0, len(y_mm) - 1))
    z_particle = U_vis[iy, ix]
    
    ax2.scatter([particle_x * 1e3], [particle_y * 1e3], [z_particle + 0.1],
                s=200, c='red', marker='o', edgecolors='white', linewidths=2, zorder=100)
    
    # Target
    ix_t = int(np.clip((target_x * 1e3 - x_mm[0]) / (x_mm[-1] - x_mm[0]) * (len(x_mm) - 1),
                       0, len(x_mm) - 1))
    iy_t = int(np.clip((target_y * 1e3 - y_mm[0]) / (y_mm[-1] - y_mm[0]) * (len(y_mm) - 1),
                       0, len(y_mm) - 1))
    z_target = U_vis[iy_t, ix_t]
    ax2.scatter([target_x * 1e3], [target_y * 1e3], [z_target + 0.1],
                s=100, c='yellow', marker='*', edgecolors='black', zorder=99)
    
    # Path on floor
    ax2.plot(path[:, 0] * 1e3, path[:, 1] * 1e3, np.zeros(len(path)),
             'w--', lw=1, alpha=0.5)
    
    ax2.set_xlabel('x (mm)')
    ax2.set_ylabel('y (mm)')
    ax2.set_zlabel('U (norm)')
    ax2.set_title('3D Potential Landscape')
    ax2.view_init(elev=30, azim=-60 + step * 0.5)
    
    # === Diagnostics panel ===
    ax3 = fig.add_subplot(1, 3, 3)
    
    if len(errors) > 0:
        ax3.plot(errors, 'b-', lw=1.5)
        ax3.axhline(np.mean(errors), color='r', linestyle='--', 
                   label=f'mean={np.mean(errors):.3f}')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Error (mm)')
        ax3.set_title('Tracking Error')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0, max(len(errors), 100))
        ax3.set_ylim(0, max(0.3, max(errors) * 1.1))
    else:
        ax3.text(0.5, 0.5, 'Collecting data...', ha='center', va='center',
                transform=ax3.transAxes)
    
    plt.tight_layout()
    plt.savefig(frame_path, dpi=100)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Fast mode (fewer steps)")
    parser.add_argument("--steps", type=int, default=400, help="Number of steps")
    parser.add_argument("--coarse", action="store_true", help="Coarse grid")
    parser.add_argument("--atlas", type=str, default="results/reachability_3puck")
    parser.add_argument("--surrogate", type=str, default="results/surrogate_model")
    parser.add_argument("--record", action="store_true", help="Enable flight recorder")
    parser.add_argument("--record_stride", type=int, default=1, help="Record every N steps")
    parser.add_argument("--save_fields_stride", type=int, default=20, help="Save fields every N steps")
    args = parser.parse_args()
    
    REPO = Path(__file__).resolve().parents[1]
    RESULTS = REPO / "results"
    frames_dir = RESULTS / "frames_demo_large_circle"
    out_dir = RESULTS / "demo_large_circle"
    
    # Create timestamped run directory for flight recorder
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_dir / f"run_{run_timestamp}"
    
    # Clean and recreate
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("LARGE CIRCLE DEMO: 3-PUCK HIERARCHICAL CONTROL")
    print("=" * 70)

    # ===== Domain + Physics =====
    if args.coarse or args.fast:
        domain = DishDomain(Lx=2e-3, Ly=2e-3, Nx=80, Ny=80)
    else:
        domain = DishDomain(Lx=2e-3, Ly=2e-3, Nx=120, Ny=120)
    
    medium = MediumProps(f=2e6, c0=1500.0, rho0=1000.0, loss_eta=1e-3, kz=0.0, coupling_alpha=1.0)
    particle = ParticleProps(a=5e-6, rho_p=1050.0, c_p=2350.0)
    
    cfg = EvaluatorConfig(
        sigma_x=0.10e-3,
        sigma_y=0.15e-3,
        bottom_band=0.25e-3,
        dt=5e-3,
        viscosity=1e-3,
        alpha_g=2e3,
        max_step=0.08e-3,
        use_2d_forcing=True,
    )
    
    ev = Evaluator3Pucks(domain, medium, particle, cfg)
    
    print(f"\nPhysics:")
    print(f"  Domain: {domain.Lx*1e3:.1f} x {domain.Ly*1e3:.1f} mm")
    print(f"  Grid: {domain.Nx} x {domain.Ny}")
    print(f"  alpha_g: {cfg.alpha_g:.0e}")
    print(f"  Using 3 pucks for full 2D control")

    # ===== Load reachability and surrogate =====
    planner = None
    surrogate = None
    
    atlas_path = Path(args.atlas)
    if atlas_path.exists() and (atlas_path / "trap_positions.npy").exists():
        print(f"\nLoading reachability atlas from {atlas_path}")
        planner = load_reachability_data(atlas_path)
        print(f"  Loaded {len(planner.trap_positions)} trap positions")
    else:
        print(f"\nNo reachability atlas found at {atlas_path}")
        print("  Run reachability_atlas_3puck.py first for better results")
    
    surrogate_path = Path(args.surrogate)
    if (surrogate_path / "gp_x.joblib").exists():
        print(f"Loading surrogate from {surrogate_path}")
        surrogate = GaussianProcessSurrogate()
        try:
            surrogate.load(surrogate_path)
        except Exception as e:
            print(f"  Warning: Could not load surrogate: {e}")
            surrogate = None

    # ===== Controller Setup =====
    bounds = ControlBounds3Pucks(
        x_min=0.0, x_max=domain.Lx,
        y_min=0.0, y_max=cfg.bottom_band,
        y_max_C=domain.Ly * 0.5,
        v_min=0.0, v_max=0.2,
    )
    
    rate_limits = ControlRateLimits3Pucks(
        dx_max=0.08e-3,
        dy_max=0.05e-3,
        dv_max=0.015,
        dphi_max=0.4,
    )
    
    hc_config = HierarchicalControllerConfig(
        segment_length=10,
        macro_magnitude=0.05e-3,
        macro_phase_step=0.15,
        mpc_candidates=40,
        macro_threshold=0.1e-3,
        control_smoothing=0.35,
    )
    
    controller = HierarchicalController(
        evaluator=ev,
        bounds=bounds,
        rate_limits=rate_limits,
        config=hc_config,
        planner=planner,
        surrogate=surrogate,
    )
    
    print(f"\nController:")
    print(f"  Macro magnitude: {hc_config.macro_magnitude*1e3:.2f} mm")
    print(f"  Macro threshold: {hc_config.macro_threshold*1e3:.2f} mm")
    print(f"  Smoothing: {hc_config.control_smoothing}")

    # ===== Path Setup: Large Circle =====
    T = 200 if args.fast else args.steps
    
    # Circle: 60% of domain diameter
    scale = 0.60
    radius = scale * min(domain.Lx, domain.Ly) / 2
    center_x = domain.Lx / 2
    center_y = domain.Ly * 0.55  # Slightly above center for visibility
    
    path = make_circle_path(center_x, center_y, radius, T)
    
    print(f"\nPath: Large Circle")
    print(f"  Center: ({center_x*1e3:.3f}, {center_y*1e3:.3f}) mm")
    print(f"  Radius: {radius*1e3:.3f} mm ({scale*100:.0f}% of domain)")
    print(f"  Steps: {T}")

    # Validate path reachability
    if planner is not None:
        frac, _ = planner.validate_path(path)
        print(f"  Estimated reachability: {frac*100:.1f}%")

    # ===== Initial State =====
    particle_x = float(path[0, 0])
    particle_y = float(path[0, 1])
    
    # Initial 3-puck configuration straddling the start
    ctrl = Control3Pucks(
        xA=0.4e-3, yA=0.03e-3, vA=0.08, phiA=0.0,
        xB=1.6e-3, yB=0.03e-3, vB=0.08, phiB=np.pi,
        xC=1.0e-3, yC=0.15e-3, vC=0.08, phiC=np.pi/2,
    )

    # ===== Simulation Loop =====
    traj_xy_mm: list[tuple[float, float]] = [(particle_x * 1e3, particle_y * 1e3)]
    control_history: list[np.ndarray] = []
    trap_history: list[tuple[float, float]] = []
    error_history: list[float] = []
    mode_history: list[str] = []
    
    # ===== Flight Recorder Setup =====
    recorder = FlightRecorder(
        out_dir=run_dir,
        enabled=args.record,
        stride=args.record_stride,
        save_fields_stride=args.save_fields_stride,
        save_fields_on_flat=True,
        verbose=True,
    )
    
    if args.record:
        print(f"\n[FlightRecorder] Recording enabled")
        print(f"  Output: {run_dir}")
        print(f"  Stride: every {args.record_stride} steps")
        print(f"  Fields: every {args.save_fields_stride} steps + flat frames")
    frame_paths: list[Path] = []
    
    render_stride = 3 if args.fast else 2
    
    print(f"\nStarting simulation...")
    print("-" * 100)
    print(f"{'Step':>5} {'px_mm':>8} {'py_mm':>8} {'err_mm':>8} {'trap_x':>8} {'trap_y':>8} "
          f"{'mode':>8} {'disp_um':>8}")
    print("-" * 100)

    for t in range(T - 1):
        target_x = float(path[t + 1, 0])
        target_y = float(path[t + 1, 1])
        
        # Controller step
        new_ctrl, new_x, new_y, info = controller.step(
            particle_x=particle_x,
            particle_y=particle_y,
            target_x=target_x,
            target_y=target_y,
            current_ctrl=ctrl,
        )
        
        # Update state
        ctrl = new_ctrl
        particle_x = new_x
        particle_y = new_y
        
        traj_xy_mm.append((particle_x * 1e3, particle_y * 1e3))
        
        ctrl_arr = np.array([
            ctrl.xA, ctrl.yA, ctrl.vA, ctrl.phiA,
            ctrl.xB, ctrl.yB, ctrl.vB, ctrl.phiB,
            ctrl.xC, ctrl.yC, ctrl.vC, ctrl.phiC,
        ])
        control_history.append(ctrl_arr)
        
        trap_x = info.get("trap_x", np.nan)
        trap_y = info.get("trap_y", np.nan)
        trap_history.append((trap_x * 1e3 if np.isfinite(trap_x) else np.nan,
                            trap_y * 1e3 if np.isfinite(trap_y) else np.nan))
        
        tracking_err = np.sqrt((particle_x - target_x)**2 + (particle_y - target_y)**2) * 1e3
        error_history.append(tracking_err)
        
        mode = info.get("mode", "?")
        mode_history.append(mode)
        
        displacement = info.get("displacement", 0.0)
        
        # ===== Flight Recorder: Record step =====
        if args.record:
            # Get metrics from evaluator (computed every step via return_metrics=True)
            eval_metrics = info.get("metrics", {})
            
            # ===== Trigger-based field dumps =====
            # Determine if we need to save field arrays on this step
            should_save_fields = False
            
            # Trigger 1: Stride-based (every N steps)
            if args.save_fields_stride > 0 and t % args.save_fields_stride == 0:
                should_save_fields = True
            
            # Trigger 2: Trap not found
            trap_found = eval_metrics.get("trap_found", False)
            trap_stable = eval_metrics.get("trap_stable", False)
            if not trap_found:
                # Check consecutive trap failures
                if not hasattr(args, '_trap_fail_count'):
                    args._trap_fail_count = 0
                args._trap_fail_count += 1
                if args._trap_fail_count >= 5:  # N=5 consecutive failures
                    should_save_fields = True
            else:
                args._trap_fail_count = 0
            
            # Trigger 3: Trap unstable when controller expects stable
            if not trap_stable and info.get("mode", "") == "mpc":
                should_save_fields = True
            
            # Trigger 4: NaN detected in fields
            if eval_metrics.get("U_nan_frac", 0) > 0 or eval_metrics.get("p_nan_frac", 0) > 0:
                should_save_fields = True
                info["render_nan_flag"] = True
            
            # Trigger 5: Very weak potential (U_ptp extremely small)
            U_ptp = eval_metrics.get("U_ptp", np.nan)
            if np.isfinite(U_ptp) and U_ptp < 1e-15:  # femtojoule scale
                should_save_fields = True
                info["render_flat_flag"] = True
            
            # Trigger 6: Force magnitude extremely small
            Fmag_max = eval_metrics.get("Fmag_max", np.nan)
            if np.isfinite(Fmag_max) and Fmag_max < 1e-15:
                should_save_fields = True
            
            # Build fields dict only if saving
            fields_dict = None
            if should_save_fields:
                vb_rec = ev.control_to_forcing_band_vb(ctrl)
                field_rec = ev.op.solve_for_bottom_vb(vb_rec)
                U_rec, Fx_rec, Fy_rec = gorkov_potential_and_force_2d(field_rec, particle)
                
                fields_dict = {
                    "p": field_rec.p,
                    "U": U_rec,
                    "Fx": Fx_rec,
                    "Fy": Fy_rec,
                    "vb": vb_rec,
                    "x": field_rec.x,
                    "y": field_rec.y,
                }
                
                # Run field diagnostics on problematic frames
                if info.get("render_flat_flag", False) or not trap_found:
                    diag = diagnose_field(U_rec, field_rec.p, vb_rec, ctrl, frame_id=f"step_{t}")
                    info["field_diagnosis"] = diag
                    if t % 25 == 0:
                        print(f"  [DIAG {t}] {diag.get('diagnosis', 'unknown')}")
            
            # Update info with trap status
            info["trap_found"] = trap_found
            
            recorder.record_step(
                step_idx=t,
                control=ctrl,
                particle_xy=(particle_x, particle_y),
                target_xy=(target_x, target_y),
                trap_xy=(trap_x, trap_y),
                info=info,
                fields=fields_dict,
                force_save_fields=should_save_fields,
            )
        
        # Print progress
        if t % 25 == 0 or t == T - 2:
            print(f"{t:5d} {particle_x*1e3:8.4f} {particle_y*1e3:8.4f} {tracking_err:8.4f} "
                  f"{trap_x*1e3:8.4f} {trap_y*1e3:8.4f} "
                  f"{mode:>8} {displacement*1e6:8.2f}")
        
        # Render frame
        if t % render_stride == 0:
            vb = ev.control_to_forcing_band_vb(ctrl)
            field = ev.op.solve_for_bottom_vb(vb)
            U, Fx, Fy = gorkov_potential_and_force_2d(field, particle)
            
            # Get stiffness
            trap_result = ev.find_trap(ctrl, particle_x, particle_y)
            stiffness = trap_result.stiffness_eigvals if trap_result.is_stable else None
            
            frame_path = frames_dir / f"frame_{t:04d}.png"
            render_demo_frame(
                frame_path=frame_path,
                step=t,
                domain=domain,
                field=field,
                U=U,
                particle_x=particle_x, particle_y=particle_y,
                target_x=target_x, target_y=target_y,
                trap_x=trap_x, trap_y=trap_y,
                trap_stable=info.get("trap_stable", False),
                ctrl=ctrl,
                path=path,
                traj_xy_mm=traj_xy_mm,
                stiffness=stiffness,
                control_mode=mode,
                errors=error_history,
            )
            frame_paths.append(frame_path)

    print("-" * 100)
    
    # ===== Results Summary =====
    traj = np.array(traj_xy_mm)
    errors = np.array(error_history)
    
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  Mean tracking error: {errors.mean():.4f} mm")
    print(f"  Max tracking error:  {errors.max():.4f} mm")
    print(f"  Final error:         {errors[-1]:.4f} mm")
    
    # Path coverage
    total_path_length = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)) * 1e3
    total_traj_length = np.sum(np.linalg.norm(np.diff(traj/1e3, axis=0), axis=1)) * 1e3
    print(f"\n  Desired path length: {total_path_length:.3f} mm")
    print(f"  Actual traj length:  {total_traj_length:.3f} mm")
    print(f"  Coverage ratio:      {total_traj_length/total_path_length*100:.1f}%")
    
    # Control smoothness
    ctrl_arr = np.array(control_history)
    delta_xA = np.diff(ctrl_arr[:, 0])
    delta_xB = np.diff(ctrl_arr[:, 4])
    delta_xC = np.diff(ctrl_arr[:, 8])
    
    jitter_xA = np.diff(delta_xA)
    jitter_xB = np.diff(delta_xB)
    jitter_xC = np.diff(delta_xC)
    
    print(f"\n  Control Smoothness:")
    print(f"    Mean |ΔxA|: {np.mean(np.abs(delta_xA))*1e6:.2f} µm/step")
    print(f"    Mean |ΔxB|: {np.mean(np.abs(delta_xB))*1e6:.2f} µm/step")
    print(f"    Mean |ΔxC|: {np.mean(np.abs(delta_xC))*1e6:.2f} µm/step")
    print(f"    Mean jitter |ΔΔx|: {np.mean(np.abs(jitter_xA))*1e6:.2f} µm/step²")
    
    # Mode distribution
    n_macro = sum(1 for m in mode_history if m == "macro")
    n_mpc = sum(1 for m in mode_history if m == "mpc")
    print(f"\n  Control Mode Distribution:")
    print(f"    Macro actions: {n_macro} ({n_macro/len(mode_history)*100:.1f}%)")
    print(f"    Local MPC:     {n_mpc} ({n_mpc/len(mode_history)*100:.1f}%)")
    
    # Save data
    np.save(out_dir / "traj_xy_mm.npy", traj)
    np.save(out_dir / "desired_xy_mm.npy", path * 1e3)
    np.save(out_dir / "control_history.npy", ctrl_arr)
    np.save(out_dir / "errors.npy", errors)
    
    # Save summary JSON
    summary = {
        "path_type": "circle",
        "radius_mm": float(radius * 1e3),
        "scale_percent": scale * 100,
        "n_steps": T,
        "mean_error_mm": float(errors.mean()),
        "max_error_mm": float(errors.max()),
        "path_length_mm": float(total_path_length),
        "traj_length_mm": float(total_traj_length),
        "coverage_percent": float(total_traj_length / total_path_length * 100),
        "control_jitter_um": float(np.mean(np.abs(jitter_xA)) * 1e6),
        "macro_steps": n_macro,
        "mpc_steps": n_mpc,
    }
    
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_dir / 'summary.json'}")
    
    # Create final summary plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Trajectory comparison
    ax = axes[0, 0]
    ax.plot(path[:, 0] * 1e3, path[:, 1] * 1e3, 'k--', lw=2, label='desired')
    ax.plot(traj[:, 0], traj[:, 1], 'b-', lw=2, label='actual')
    ax.scatter(traj[0, 0], traj[0, 1], s=100, c='green', marker='o', label='start', zorder=10)
    ax.scatter(traj[-1, 0], traj[-1, 1], s=100, c='red', marker='s', label='end', zorder=10)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_title(f'Large Circle Demo: {scale*100:.0f}% Domain Coverage')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Tracking error
    ax = axes[0, 1]
    ax.plot(errors, 'b-', lw=1.5)
    ax.axhline(errors.mean(), color='r', linestyle='--', label=f'mean={errors.mean():.4f} mm')
    ax.fill_between(range(len(errors)), 0, errors, alpha=0.2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Error (mm)')
    ax.set_title('Tracking Error Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Control positions
    ax = axes[1, 0]
    ax.plot(ctrl_arr[:, 0] * 1e3, 'orange', lw=1.5, label='xA')
    ax.plot(ctrl_arr[:, 4] * 1e3, 'blue', lw=1.5, label='xB')
    ax.plot(ctrl_arr[:, 8] * 1e3, 'magenta', lw=1.5, label='xC')
    ax.set_xlabel('Step')
    ax.set_ylabel('Position (mm)')
    ax.set_title('Transducer X Positions (Smooth Motion)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Control deltas (smoothness verification)
    ax = axes[1, 1]
    ax.plot(delta_xA * 1e6, 'orange', lw=1, alpha=0.7, label='ΔxA')
    ax.plot(delta_xB * 1e6, 'blue', lw=1, alpha=0.7, label='ΔxB')
    ax.plot(delta_xC * 1e6, 'magenta', lw=1, alpha=0.7, label='ΔxC')
    ax.axhline(0, color='k', linestyle='-', lw=0.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Δx (µm/step)')
    ax.set_title('Control Rate (Smoothness Verification)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / "summary_large_circle.png", dpi=150)
    plt.close()
    print(f"Saved: {out_dir / 'summary_large_circle.png'}")
    
    # Create GIF
    if frame_paths:
        print(f"\nCreating GIF from {len(frame_paths)} frames...")
        images = [imageio.imread(str(p)) for p in frame_paths]
        gif_path = out_dir / "demo_large_circle.gif"
        imageio.mimsave(str(gif_path), images, fps=15, loop=0)
        print(f"Saved: {gif_path}")
    
    # Finalize flight recorder
    recorder.finalize()
    
    print("\n" + "=" * 70)
    print("LARGE CIRCLE DEMO COMPLETE")
    print("=" * 70)
    print(f"\nKey files:")
    print(f"  GIF:     {out_dir / 'demo_large_circle.gif'}")
    print(f"  Summary: {out_dir / 'summary.json'}")
    print(f"  Plot:    {out_dir / 'summary_large_circle.png'}")
    if args.record:
        print(f"  Flight Recorder: {run_dir}")
        print(f"\nTo analyze diagnostics:")
        print(f"  python scripts/plot_control_diagnostics.py {run_dir}")


if __name__ == "__main__":
    main()
