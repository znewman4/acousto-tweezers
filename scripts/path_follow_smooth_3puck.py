#!/usr/bin/env python3
"""
Path following with 3-Puck Smooth MPC Controller.

FIXES from path_follow_smooth.py:
1. Uses 3 pucks instead of 2 for full 2D control authority
2. Ensures particle actually moves by proper force application
3. Includes 3D view alongside 2D view in GIF frames

Run:
    python scripts/path_follow_smooth_3puck.py
    python scripts/path_follow_smooth_3puck.py --circle --large
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio.v2 as imageio
from typing import Optional
import shutil

from acousto.force import ParticleProps, gorkov_potential_and_force_2d
from acousto.analysis import find_trap_center

from tweezers.control import (
    DishDomain, MediumProps, EvaluatorConfig,
    Control3Pucks, ControlVector3Pucks, ControlBounds3Pucks, ControlRateLimits3Pucks,
    Evaluator3Pucks, default_3puck_config, default_3puck_spread,
)
from tweezers.viz.render_3d import normalize_gorkov_field


# =============================================================================
# Configuration
# =============================================================================
PATH_SCALE = 0.6       # Scale of path (60% of domain for circle)
T_STEPS = 400          # Number of simulation steps
RENDER_STRIDE = 2      # Render every Nth step for GIF
MAKE_GIF = True        # Whether to create animated GIF
COARSE_GRID = False    # Use coarser grid for faster testing


def make_circle_path(
    center_x: float, center_y: float, 
    radius: float, 
    T: int,
    n_loops: int = 1,
) -> np.ndarray:
    """Create smooth circular path."""
    theta = np.linspace(0, 2 * np.pi * n_loops, T)
    x = center_x + radius * np.cos(theta)
    y = center_y + radius * np.sin(theta)
    return np.column_stack([x, y])


def make_polyline_path(points: list[tuple[float, float]], T: int) -> np.ndarray:
    """Create smooth path interpolating through waypoints."""
    points = np.array(points)
    n_segments = len(points) - 1
    
    # Compute segment lengths
    seg_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total_length = seg_lengths.sum()
    
    # Allocate timesteps proportional to segment length
    seg_times = (seg_lengths / total_length * T).astype(int)
    seg_times[-1] = T - seg_times[:-1].sum()
    
    path = []
    for i in range(n_segments):
        t_seg = np.linspace(0, 1, seg_times[i], endpoint=(i == n_segments - 1))
        for ti in t_seg:
            px = points[i, 0] + ti * (points[i+1, 0] - points[i, 0])
            py = points[i, 1] + ti * (points[i+1, 1] - points[i, 1])
            path.append([px, py])
    
    return np.array(path)


class SmoothMPC3PuckController:
    """
    Simplified MPC controller for 3-puck system.
    
    Key improvements:
    - Uses 3 pucks for full 2D authority
    - Macro-action guided search
    - Jitter penalty
    """
    
    def __init__(
        self,
        evaluator: Evaluator3Pucks,
        bounds: ControlBounds3Pucks,
        rate_limits: ControlRateLimits3Pucks,
        *,
        horizon: int = 4,
        n_candidates: int = 50,
        tracking_weight: float = 1e6,
        trap_weight: float = 2e6,
        jitter_weight: float = 1e4,
        seed: int = 42,
    ):
        self.ev = evaluator
        self.bounds = bounds
        self.rate_limits = rate_limits
        self.horizon = horizon
        self.n_candidates = n_candidates
        self.tracking_weight = tracking_weight
        self.trap_weight = trap_weight
        self.jitter_weight = jitter_weight
        self.rng = np.random.default_rng(seed)
        
        # History for jitter penalty
        self.control_history: list[np.ndarray] = []
        self.prev_best_sequence: list[ControlVector3Pucks] = []
    
    def _sample_candidates(
        self,
        base: ControlVector3Pucks,
        prev: Optional[ControlVector3Pucks],
        target_x: float,
        target_y: float,
        trap_x: float,
        trap_y: float,
    ) -> list[ControlVector3Pucks]:
        """Generate candidate controls guided by target direction."""
        candidates: list[ControlVector3Pucks] = [base]
        
        # Direction from trap to target
        dx = target_x - trap_x
        dy = target_y - trap_y
        dist = np.sqrt(dx**2 + dy**2)
        
        if dist > 1e-9:
            dx /= dist
            dy /= dist
        else:
            dx, dy = 0.0, 0.0
        
        base_arr = base.to_array()
        
        # Noise scales
        pos_noise = 0.02e-3
        amp_noise = 0.005
        phase_noise = 0.15
        
        for _ in range(self.n_candidates - 1):
            # Random perturbation with bias toward target
            noise = self.rng.normal(size=12)
            
            # Position noise for all 3 transducers
            # Bias: move centroid toward target
            noise[0] += dx * 0.5  # xA
            noise[4] += dx * 0.5  # xB
            noise[8] += dx * 0.5  # xC
            noise[1] += dy * 0.3  # yA (smaller y bias for bottom pucks)
            noise[5] += dy * 0.3  # yB
            noise[9] += dy * 0.5  # yC (C can move more in y)
            
            scales = np.array([
                pos_noise, pos_noise * 0.5, amp_noise, phase_noise,  # A
                pos_noise, pos_noise * 0.5, amp_noise, phase_noise,  # B
                pos_noise, pos_noise, amp_noise, phase_noise,        # C
            ])
            
            perturbed = base_arr + noise * scales
            ctrl = ControlVector3Pucks.from_array(perturbed, self.bounds, self.rate_limits)
            ctrl = ctrl.clamp_to_bounds()
            if prev is not None:
                ctrl = ctrl.apply_rate_limits(prev)
            candidates.append(ctrl)
        
        # Add warm-start from previous best sequence
        if len(self.prev_best_sequence) > 1:
            candidates.append(self.prev_best_sequence[1])
        
        return candidates
    
    def _compute_jitter_penalty(self, ctrl_arr: np.ndarray) -> float:
        """Compute jitter penalty from control history."""
        if len(self.control_history) < 2:
            return 0.0
        
        # ΔΔu = u - 2*u_prev + u_prev2
        u_prev = self.control_history[-1]
        u_prev2 = self.control_history[-2] if len(self.control_history) >= 2 else u_prev
        
        jitter = ctrl_arr - 2 * u_prev + u_prev2
        
        # Weight position dimensions more
        weights = np.array([
            1.0, 0.5, 0.1, 0.1,  # A: x, y, v, phi
            1.0, 0.5, 0.1, 0.1,  # B
            1.0, 1.0, 0.1, 0.1,  # C
        ])
        
        return self.jitter_weight * np.sum(weights * jitter**2)
    
    def step(
        self,
        particle_x: float,
        particle_y: float,
        target_x: float,
        target_y: float,
        current_ctrl: ControlVector3Pucks,
        *,
        targets_horizon: Optional[list[tuple[float, float]]] = None,
    ) -> tuple[ControlVector3Pucks, float, float, dict]:
        """
        One MPC step.
        
        Returns: (best_control, new_x, new_y, info)
        """
        # Find current trap
        u = current_ctrl.to_control3pucks()
        trap_result = self.ev.find_trap(u, particle_x, particle_y, search_radius=0.5e-3)
        trap_x = trap_result.x if trap_result.is_stable else particle_x
        trap_y = trap_result.y if trap_result.is_stable else particle_y
        
        prev_ctrl = None
        if len(self.control_history) > 0:
            prev_ctrl = ControlVector3Pucks.from_array(
                self.control_history[-1], self.bounds, self.rate_limits
            )
        
        # Generate candidates
        candidates = self._sample_candidates(
            current_ctrl, prev_ctrl, target_x, target_y, trap_x, trap_y
        )
        
        best_cost = float("inf")
        best_ctrl = current_ctrl
        best_next_x = particle_x
        best_next_y = particle_y
        best_info: dict = {}
        
        for ctrl in candidates:
            u3p = ctrl.to_control3pucks()
            
            # Simulate one step
            xp1, yp1, loss, info = self.ev.step(
                xp=particle_x, yp=particle_y,
                target_x=target_x, target_y=target_y,
                u=u3p,
            )
            
            # Find trap for new control
            trap_new = self.ev.find_trap(u3p, xp1, yp1, search_radius=0.5e-3)
            
            # Tracking cost
            dx = xp1 - target_x
            dy = yp1 - target_y
            tracking_cost = self.tracking_weight * (dx**2 + dy**2)
            
            # Trap steering cost
            trap_cost = 0.0
            if trap_new.is_stable:
                trap_dx = trap_new.x - target_x
                trap_dy = trap_new.y - target_y
                trap_cost = self.trap_weight * (trap_dx**2 + trap_dy**2)
            
            # Jitter cost
            jitter_cost = self._compute_jitter_penalty(ctrl.to_array())
            
            total_cost = tracking_cost + trap_cost + jitter_cost
            
            if total_cost < best_cost:
                best_cost = total_cost
                best_ctrl = ctrl
                best_next_x = xp1
                best_next_y = yp1
                best_info = {
                    "trap_x": trap_new.x if trap_new.is_stable else np.nan,
                    "trap_y": trap_new.y if trap_new.is_stable else np.nan,
                    "trap_stable": trap_new.is_stable,
                    "stiffness": trap_new.stiffness_eigvals if trap_new.is_stable else None,
                    "fx": info["fx"],
                    "fy": info["fy"],
                    "displacement": info["displacement"],
                }
        
        # Update history
        self.control_history.append(best_ctrl.to_array())
        if len(self.control_history) > 10:
            self.control_history.pop(0)
        
        return best_ctrl, best_next_x, best_next_y, best_info


def render_frame_2d_3d(
    frame_path: Path,
    step: int,
    domain: DishDomain,
    field,
    U: np.ndarray,
    Fx: np.ndarray,
    Fy: np.ndarray,
    particle_x: float,
    particle_y: float,
    target_x: float,
    target_y: float,
    trap_x: float,
    trap_y: float,
    trap_stable: bool,
    ctrl: ControlVector3Pucks,
    path: np.ndarray,
    traj_xy_mm: list[tuple[float, float]],
    stiffness: Optional[np.ndarray],
):
    """Render frame with both 2D and 3D views."""
    fig = plt.figure(figsize=(16, 6))
    
    # === 2D view (left) ===
    ax1 = fig.add_subplot(1, 2, 1)
    
    x_mm = field.x * 1e3
    y_mm = field.y * 1e3
    X, Y = np.meshgrid(x_mm, y_mm)
    
    # Potential contours
    U_scaled = U * 1e15  # pJ scale
    levels = 30
    cf = ax1.contourf(X, Y, U_scaled, levels=levels, cmap='viridis', alpha=0.8)
    
    # Desired path
    ax1.plot(path[:, 0] * 1e3, path[:, 1] * 1e3, 'w--', lw=1.5, label='desired')
    
    # Actual trajectory
    if len(traj_xy_mm) > 1:
        traj = np.array(traj_xy_mm)
        ax1.plot(traj[:, 0], traj[:, 1], 'c-', lw=2, label='actual')
    
    # Current position
    ax1.scatter(particle_x * 1e3, particle_y * 1e3, s=200, c='red', marker='o',
                edgecolors='white', linewidths=2, zorder=10, label='particle')
    
    # Target
    ax1.scatter(target_x * 1e3, target_y * 1e3, s=120, c='yellow', marker='*',
                edgecolors='black', linewidths=1, zorder=9, label='target')
    
    # Trap
    if trap_stable and np.isfinite(trap_x) and np.isfinite(trap_y):
        ax1.scatter(trap_x * 1e3, trap_y * 1e3, s=100, c='lime',
                    marker='x', linewidths=3, zorder=8, label='trap')
    
    # Transducers (3 pucks)
    ax1.scatter(ctrl.xA * 1e3, ctrl.yA * 1e3, s=120, c='orange',
                marker='^', edgecolors='black', linewidths=1, label='puck A')
    ax1.scatter(ctrl.xB * 1e3, ctrl.yB * 1e3, s=120, c='blue',
                marker='^', edgecolors='black', linewidths=1, label='puck B')
    ax1.scatter(ctrl.xC * 1e3, ctrl.yC * 1e3, s=120, c='magenta',
                marker='^', edgecolors='black', linewidths=1, label='puck C')
    
    ax1.set_xlim(0, domain.Lx * 1e3)
    ax1.set_ylim(0, domain.Ly * 1e3)
    ax1.set_xlabel('x (mm)')
    ax1.set_ylabel('y (mm)')
    
    stiff_str = ""
    if stiffness is not None:
        stiff_str = f" | stiff={np.min(stiffness):.2e}"
    ax1.set_title(f'Step {step}: 3-Puck Control{stiff_str}')
    ax1.set_aspect('equal')
    ax1.legend(loc='upper right', fontsize=7, ncol=2)
    
    # === 3D view (right) ===
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    
    # Normalize U for visualization
    U_vis, is_flat = normalize_gorkov_field(U, verbose=False)
    
    # Subsample for faster 3D rendering
    step_size = max(1, len(x_mm) // 60)
    X_sub = X[::step_size, ::step_size]
    Y_sub = Y[::step_size, ::step_size]
    U_sub = U_vis[::step_size, ::step_size]
    
    # 3D surface
    ax2.plot_surface(X_sub, Y_sub, U_sub, cmap='viridis', alpha=0.7,
                     rstride=1, cstride=1, linewidth=0, antialiased=True)
    
    # Particle position on surface
    # Find z value at particle position
    ix = int((particle_x * 1e3 - x_mm[0]) / (x_mm[-1] - x_mm[0]) * (len(x_mm) - 1))
    iy = int((particle_y * 1e3 - y_mm[0]) / (y_mm[-1] - y_mm[0]) * (len(y_mm) - 1))
    ix = np.clip(ix, 0, len(x_mm) - 1)
    iy = np.clip(iy, 0, len(y_mm) - 1)
    z_particle = U_vis[iy, ix]
    
    ax2.scatter([particle_x * 1e3], [particle_y * 1e3], [z_particle + 0.1],
                s=200, c='red', marker='o', edgecolors='white', linewidths=2, zorder=100)
    
    # Target
    ix_t = int((target_x * 1e3 - x_mm[0]) / (x_mm[-1] - x_mm[0]) * (len(x_mm) - 1))
    iy_t = int((target_y * 1e3 - y_mm[0]) / (y_mm[-1] - y_mm[0]) * (len(y_mm) - 1))
    ix_t = np.clip(ix_t, 0, len(x_mm) - 1)
    iy_t = np.clip(iy_t, 0, len(y_mm) - 1)
    z_target = U_vis[iy_t, ix_t]
    ax2.scatter([target_x * 1e3], [target_y * 1e3], [z_target + 0.1],
                s=120, c='yellow', marker='*', edgecolors='black', linewidths=1, zorder=99)
    
    # Draw desired path on base
    path_x = path[:, 0] * 1e3
    path_y = path[:, 1] * 1e3
    path_z = np.zeros_like(path_x)  # On the floor
    ax2.plot(path_x, path_y, path_z, 'w--', lw=1, alpha=0.5)
    
    ax2.set_xlabel('x (mm)')
    ax2.set_ylabel('y (mm)')
    ax2.set_zlabel('U (normalized)')
    ax2.set_title('3D Gor\'kov Potential Landscape')
    ax2.view_init(elev=35, azim=-60 + step * 0.3)  # Slowly rotate view
    
    plt.tight_layout()
    plt.savefig(frame_path, dpi=100)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--circle", action="store_true", help="Use circular path")
    parser.add_argument("--large", action="store_true", help="Use large path (60% domain)")
    parser.add_argument("--steps", type=int, default=T_STEPS, help="Number of steps")
    parser.add_argument("--coarse", action="store_true", help="Use coarse grid")
    args = parser.parse_args()
    
    REPO = Path(__file__).resolve().parents[1]
    RESULTS = REPO / "results"
    frames_dir = RESULTS / "frames_path_follow_3puck"
    out_dir = RESULTS / "path_follow_3puck"
    
    # Clean and recreate
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PATH FOLLOWING WITH 3-PUCK SMOOTH MPC CONTROLLER")
    print("=" * 70)

    # ===== Domain + Physics Setup =====
    if args.coarse:
        domain = DishDomain(Lx=2e-3, Ly=2e-3, Nx=80, Ny=80)
    else:
        domain = DishDomain(Lx=2e-3, Ly=2e-3, Nx=120, Ny=120)
    
    medium = MediumProps(f=2e6, c0=1500.0, rho0=1000.0, loss_eta=1e-3, kz=0.0, coupling_alpha=1.0)
    particle = ParticleProps(a=5e-6, rho_p=1050.0, c_p=2350.0)

    # Evaluator config - tuned for visible motion
    alpha_g = 2e3  # Strong force scaling for visible motion
    v_amp = 0.08   # Stronger amplitude
    
    cfg = EvaluatorConfig(
        sigma_x=0.10e-3,
        sigma_y=0.15e-3,
        bottom_band=0.25e-3,
        dt=5e-3,
        viscosity=1e-3,
        border_penalty=1e6,
        smooth_u=0.0,
        alpha_g=alpha_g,
        max_step=0.08e-3,  # Allow larger steps for visible motion
        use_2d_forcing=True,
    )

    ev = Evaluator3Pucks(domain, medium, particle, cfg)
    
    print(f"\nPhysics config:")
    print(f"  alpha_g = {alpha_g:.0e}")
    print(f"  dt = {cfg.dt*1e3:.1f} ms")
    print(f"  v_amp = {v_amp}")
    print(f"  max_step = {cfg.max_step*1e6:.1f} µm")
    print(f"  Grid: {domain.Nx}x{domain.Ny}")
    print(f"  Using 3 pucks for full 2D control")

    # ===== Controller Setup =====
    bounds = ControlBounds3Pucks(
        x_min=0.0, x_max=domain.Lx,
        y_min=0.0, y_max=cfg.bottom_band,
        y_max_C=domain.Ly * 0.5,  # C can go higher
        v_min=0.0, v_max=0.2,
    )
    
    rate_limits = ControlRateLimits3Pucks(
        dx_max=0.08e-3,
        dy_max=0.05e-3,
        dv_max=0.015,
        dphi_max=0.4,
    )
    
    controller = SmoothMPC3PuckController(
        evaluator=ev,
        bounds=bounds,
        rate_limits=rate_limits,
        horizon=4,
        n_candidates=60,
        tracking_weight=1e6,
        trap_weight=2e6,
        jitter_weight=5e3,
    )
    
    print(f"\nController config:")
    print(f"  Horizon: {controller.horizon}")
    print(f"  Candidates: {controller.n_candidates}")

    # ===== Desired Path =====
    T = args.steps
    
    if args.circle or args.large:
        # Large circle path (60% of domain)
        scale = 0.6 if args.large else 0.4
        radius = scale * min(domain.Lx, domain.Ly) / 2
        center_x = domain.Lx / 2
        center_y = domain.Ly * 0.55  # Slightly above center
        path = make_circle_path(center_x, center_y, radius, T)
        print(f"\nPath: Circle")
        print(f"  Center: ({center_x*1e3:.3f}, {center_y*1e3:.3f}) mm")
        print(f"  Radius: {radius*1e3:.3f} mm ({scale*100:.0f}% of domain)")
    else:
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
        scale = 0.6 if args.large else 0.4
        
        scaled_points = []
        for px, py in raw_points:
            sx = centroid[0] + scale * (px - centroid[0])
            sy = centroid[1] + scale * (py - centroid[1])
            scaled_points.append((sx, sy))
        
        path = make_polyline_path(points=scaled_points, T=T)
        print(f"\nPath: Rectangle")
        print(f"  Scale: {scale}")
        print(f"  Centroid: ({centroid[0]*1e3:.3f}, {centroid[1]*1e3:.3f}) mm")

    # ===== Initial State =====
    # Start particle at first path point
    particle_x = float(path[0, 0])
    particle_y = float(path[0, 1])
    
    # Initial 3-puck configuration straddling the starting position
    initial_ctrl = ControlVector3Pucks(
        xA=0.4e-3, yA=0.03e-3, vA=v_amp, phiA=0.0,
        xB=1.6e-3, yB=0.03e-3, vB=v_amp, phiB=np.pi,
        xC=1.0e-3, yC=0.15e-3, vC=v_amp, phiC=np.pi/2,  # C at center, higher y
        bounds=bounds,
        rate_limits=rate_limits,
    )
    control = initial_ctrl

    # ===== Simulation Loop =====
    traj_xy_mm: list[tuple[float, float]] = [(particle_x * 1e3, particle_y * 1e3)]
    control_history_raw: list[np.ndarray] = [control.to_array()]
    trap_history: list[tuple[float, float]] = []
    error_history: list[float] = []
    frame_paths: list[Path] = []
    
    print(f"\nStarting simulation with {T} steps...")
    print("-" * 110)
    print(f"{'Step':>5} {'px_mm':>8} {'py_mm':>8} {'err_mm':>8} {'trap_x':>8} {'trap_y':>8} "
          f"{'xA':>8} {'xB':>8} {'xC':>8} {'disp_um':>8}")
    print("-" * 110)

    for t in range(T - 1):
        # Target for this step
        target_x = float(path[t + 1, 0])
        target_y = float(path[t + 1, 1])
        
        # Controller step
        new_control, new_x, new_y, info = controller.step(
            particle_x=particle_x,
            particle_y=particle_y,
            target_x=target_x,
            target_y=target_y,
            current_ctrl=control,
        )
        
        # Update state
        control = new_control
        particle_x = new_x
        particle_y = new_y
        traj_xy_mm.append((particle_x * 1e3, particle_y * 1e3))
        control_history_raw.append(control.to_array())
        
        trap_x = info.get("trap_x", np.nan)
        trap_y = info.get("trap_y", np.nan)
        trap_history.append((trap_x * 1e3 if np.isfinite(trap_x) else np.nan,
                            trap_y * 1e3 if np.isfinite(trap_y) else np.nan))
        
        tracking_err = np.sqrt((particle_x - target_x)**2 + (particle_y - target_y)**2) * 1e3
        error_history.append(tracking_err)
        
        displacement = info.get("displacement", 0.0)
        
        # Print progress
        if t % 25 == 0 or t == T - 2:
            print(f"{t:5d} {particle_x*1e3:8.4f} {particle_y*1e3:8.4f} {tracking_err:8.4f} "
                  f"{trap_x*1e3:8.4f} {trap_y*1e3:8.4f} "
                  f"{control.xA*1e3:8.4f} {control.xB*1e3:8.4f} {control.xC*1e3:8.4f} "
                  f"{displacement*1e6:8.2f}")
        
        # Render frame
        if MAKE_GIF and t % RENDER_STRIDE == 0:
            # Get field for visualization
            u3p = control.to_control3pucks()
            vb = ev.control_to_forcing_band_vb(u3p)
            field = ev.op.solve_for_bottom_vb(vb)
            U, Fx, Fy = gorkov_potential_and_force_2d(field, particle)
            
            frame_path = frames_dir / f"frame_{t:04d}.png"
            render_frame_2d_3d(
                frame_path=frame_path,
                step=t,
                domain=domain,
                field=field,
                U=U, Fx=Fx, Fy=Fy,
                particle_x=particle_x, particle_y=particle_y,
                target_x=target_x, target_y=target_y,
                trap_x=trap_x, trap_y=trap_y,
                trap_stable=info.get("trap_stable", False),
                ctrl=control,
                path=path,
                traj_xy_mm=traj_xy_mm,
                stiffness=info.get("stiffness"),
            )
            frame_paths.append(frame_path)

    print("-" * 110)
    
    # ===== Results Summary =====
    traj = np.array(traj_xy_mm)
    errors = np.array(error_history)
    
    print(f"\nResults Summary:")
    print(f"  Mean tracking error: {errors.mean():.4f} mm")
    print(f"  Max tracking error:  {errors.max():.4f} mm")
    print(f"  Final error:         {errors[-1]:.4f} mm")
    
    # Path coverage
    total_path_length = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)) * 1e3
    total_traj_length = np.sum(np.linalg.norm(np.diff(traj/1e3, axis=0), axis=1)) * 1e3
    print(f"  Path length:  {total_path_length:.3f} mm")
    print(f"  Traj length:  {total_traj_length:.3f} mm")
    
    # Compute control smoothness metrics
    ctrl_arr = np.array(control_history_raw)
    delta_xA = np.diff(ctrl_arr[:, 0])
    delta_xB = np.diff(ctrl_arr[:, 4])
    delta_xC = np.diff(ctrl_arr[:, 8])
    
    jitter_xA = np.diff(delta_xA)
    jitter_xB = np.diff(delta_xB)
    jitter_xC = np.diff(delta_xC)
    
    print(f"\nControl Smoothness:")
    print(f"  Mean |ΔxA|: {np.mean(np.abs(delta_xA))*1e6:.2f} µm/step")
    print(f"  Mean |ΔxB|: {np.mean(np.abs(delta_xB))*1e6:.2f} µm/step")
    print(f"  Mean |ΔxC|: {np.mean(np.abs(delta_xC))*1e6:.2f} µm/step")
    print(f"  Mean |ΔΔxA| (jitter): {np.mean(np.abs(jitter_xA))*1e6:.2f} µm/step²")
    
    # Save data
    np.save(out_dir / "traj_xy_mm.npy", traj)
    np.save(out_dir / "desired_xy_mm.npy", path * 1e3)
    np.save(out_dir / "control_history.npy", ctrl_arr)
    np.save(out_dir / "errors.npy", errors)
    
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
    ax.set_title('Trajectory (3-Puck Control)')
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
    ax.plot(ctrl_arr[:, 4] * 1e3, 'blue', lw=1.5, label='xB')
    ax.plot(ctrl_arr[:, 8] * 1e3, 'magenta', lw=1.5, label='xC')
    ax.set_xlabel('Step')
    ax.set_ylabel('Position (mm)')
    ax.set_title('Transducer X Positions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Control deltas (smoothness)
    ax = axes[1, 1]
    ax.plot(delta_xA * 1e6, 'orange', lw=1, alpha=0.7, label='ΔxA')
    ax.plot(delta_xB * 1e6, 'blue', lw=1, alpha=0.7, label='ΔxB')
    ax.plot(delta_xC * 1e6, 'magenta', lw=1, alpha=0.7, label='ΔxC')
    ax.axhline(0, color='k', linestyle='-', lw=0.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Δx (µm/step)')
    ax.set_title('Control Rate (Smoothness)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / "summary_3puck.png", dpi=150)
    plt.close()
    print(f"\nSaved: {out_dir / 'summary_3puck.png'}")
    
    # Create GIF
    if MAKE_GIF and frame_paths:
        print(f"\nCreating GIF from {len(frame_paths)} frames...")
        images = [imageio.imread(str(p)) for p in frame_paths]
        gif_path = out_dir / "path_follow_3puck.gif"
        imageio.mimsave(str(gif_path), images, fps=15, loop=0)
        print(f"Saved: {gif_path}")
    
    print("\n" + "=" * 70)
    print("DONE - 3-Puck Path Following Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
