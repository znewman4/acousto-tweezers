#!/usr/bin/env python3
"""
STAGE F: Two Clean Demos for Large 2D Particle Manipulation

Demo 1: Slow Sweep in X (with Y response)
  - Target: x from 0.5*Lx → 0.7*Lx, allowing natural y response
  - Shows 2D control authority after Stage A upgrade

Demo 2: Large Circle
  - Circle radius ~ 0.3-0.4 * min(Lx, Ly)
  - Centre roughly middle of dish
  - Demonstrates full 2D trajectory following

Both demos include:
  - Trap centre tracking
  - Stiffness eigenvalue overlay
  - Step limiter flags
  - Tracking error metrics
  - GIF output
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
import argparse

from acousto.force import ParticleProps, bilinear_sample_vec
from acousto.analysis import find_traps_from_force, find_trap_center, TrapTracker

from tweezers.control import (
    DishDomain, MediumProps, EvaluatorConfig,
    Control2Pucks, BottomFootprint25DEvaluator,
    ControlState, ControlVector, ControlBounds, ControlRateLimits,
    ControllerConfig, ParticleController, SafetyConfig,
)

from tweezers.viz.render_3d import (
    Cylinder2D, classify_trap, normalize_gorkov_field,
)


# =============================================================================
# Demo Configuration
# =============================================================================

class DemoConfig:
    """Configuration for a demo."""
    name: str
    T_steps: int
    render_stride: int
    
    def __init__(self, name: str, T_steps: int = 300, render_stride: int = 1):
        self.name = name
        self.T_steps = T_steps
        self.render_stride = render_stride


def generate_sweep_path(
    Lx: float,
    Ly: float,
    T: int,
    x_start_frac: float = 0.3,
    x_end_frac: float = 0.7,
    y_center_frac: float = 0.5,
) -> np.ndarray:
    """
    Generate a slow sweep path in x direction.
    
    Returns path array of shape (T, 2) in meters.
    """
    x_start = x_start_frac * Lx
    x_end = x_end_frac * Lx
    y_center = y_center_frac * Ly
    
    x_path = np.linspace(x_start, x_end, T)
    y_path = np.full(T, y_center)
    
    return np.column_stack([x_path, y_path])


def generate_circle_path(
    Lx: float,
    Ly: float,
    T: int,
    radius_frac: float = 0.35,
    center_x_frac: float = 0.5,
    center_y_frac: float = 0.55,
    n_loops: float = 1.0,
) -> np.ndarray:
    """
    Generate a circular path.
    
    Returns path array of shape (T, 2) in meters.
    """
    min_dim = min(Lx, Ly)
    radius = radius_frac * min_dim
    center_x = center_x_frac * Lx
    center_y = center_y_frac * Ly
    
    # Ensure the circle fits within the domain with margin
    margin = 0.1 * min_dim
    max_radius = min(
        center_x - margin,
        Lx - center_x - margin,
        center_y - margin,
        Ly - center_y - margin,
    )
    radius = min(radius, max_radius)
    
    theta = np.linspace(0, 2 * np.pi * n_loops, T)
    x_path = center_x + radius * np.cos(theta)
    y_path = center_y + radius * np.sin(theta)
    
    return np.column_stack([x_path, y_path])


def compute_guided_control(
    target_x: float,
    target_y: float,
    current_control: ControlVector,
    domain_Lx: float,
    separation: float = 0.8e-3,
    y_fixed: float = 0.05e-3,  # Closer to boundary for better coupling
    margin: float = 0.1e-3,
) -> tuple[float, float, float, float]:
    """
    Compute transducer positions to straddle the target.
    
    With sigma_y=0.15mm and y_fixed=0.05mm, the boundary coupling is:
        exp(-0.05^2 / (2 * 0.15^2)) = exp(-0.056) ≈ 0.95
    
    Returns (xA, yA, xB, yB) in meters.
    """
    half_sep = separation / 2.0
    
    xA_ideal = target_x - half_sep
    xB_ideal = target_x + half_sep
    
    xA = float(np.clip(xA_ideal, margin, domain_Lx - margin))
    xB = float(np.clip(xB_ideal, margin, domain_Lx - margin))
    
    # Ensure minimum separation
    min_sep = 0.2e-3
    if xB - xA < min_sep:
        center = (xA + xB) / 2.0
        xA = center - min_sep / 2.0
        xB = center + min_sep / 2.0
        xA = float(np.clip(xA, margin, domain_Lx - margin))
        xB = float(np.clip(xB, margin, domain_Lx - margin))
    
    return xA, y_fixed, xB, y_fixed


def render_demo_frame(
    *,
    out_png: Path,
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    U: np.ndarray,
    particle_xy_mm: tuple[float, float],
    target_xy_mm: tuple[float, float],
    trap_centre_mm: tuple[float, float] | None,
    track_xy_mm: list[tuple[float, float]],
    target_path_mm: np.ndarray,
    cylinders: list,
    metrics: dict,
    step: int,
    total_steps: int,
) -> None:
    """
    Render a single demo frame with comprehensive overlays.
    """
    px_mm, py_mm = particle_xy_mm
    tx_mm, ty_mm = target_xy_mm
    
    # Prepare data
    X, Y = np.meshgrid(x_mm, y_mm)
    U_display = U * 1e15
    Uvis, _ = normalize_gorkov_field(U_display, verbose=False)
    
    # Create 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # ============ Panel 1: 2D Trajectory View ============
    ax1 = axes[0, 0]
    
    # Contour of U
    ax1.contourf(X, Y, Uvis, levels=20, cmap="viridis", alpha=0.7)
    ax1.contour(X, Y, Uvis, levels=12, colors="k", linewidths=0.3, alpha=0.3)
    
    # Draw full target path (light gray)
    ax1.plot(target_path_mm[:, 0], target_path_mm[:, 1], 
             color='gray', linewidth=1.5, linestyle='--', alpha=0.5, label='target path')
    
    # Draw trajectory history with gradient
    if track_xy_mm and len(track_xy_mm) >= 2:
        tx_hist = np.array([p[0] for p in track_xy_mm])
        ty_hist = np.array([p[1] for p in track_xy_mm])
        n_pts = len(tx_hist)
        for i in range(n_pts - 1):
            alpha_color = i / max(n_pts - 1, 1)
            color = (1 - alpha_color) * np.array([0.5, 0.5, 0.5]) + alpha_color * np.array([0, 1, 1])
            ax1.plot(tx_hist[i:i+2], ty_hist[i:i+2], linewidth=2.5, color=color, alpha=0.9)
    
    # Trap centre (cyan X)
    if trap_centre_mm is not None:
        ax1.scatter(trap_centre_mm[0], trap_centre_mm[1], s=120, marker="x", color="cyan",
                   linewidth=3, label="trap", zorder=18)
    
    # Target (yellow star)
    ax1.scatter(tx_mm, ty_mm, s=250, marker="*", color="yellow", edgecolors="black",
               linewidth=2, label="target", zorder=20)
    
    # Particle (red circle)
    ax1.scatter(px_mm, py_mm, s=180, marker="o", color="red", edgecolors="white",
               linewidth=2, label="particle", zorder=15)
    
    # Transducers
    for i, cyl in enumerate(cylinders):
        circle = plt.Circle((cyl.x_mm, cyl.y_mm), cyl.r_mm, fill=False,
                           edgecolor="yellow", linewidth=2, linestyle="--")
        ax1.add_patch(circle)
        ax1.annotate(f"T{i+1}", (cyl.x_mm, cyl.y_mm), color="yellow",
                    fontsize=9, ha="center", va="center")
    
    ax1.set_xlabel("x (mm)")
    ax1.set_ylabel("y (mm)")
    ax1.set_title(f"Step {step}/{total_steps}: Trajectory + Trap Centre", fontweight="bold")
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", fontsize=8)
    
    # ============ Panel 2: Metrics Text ============
    ax2 = axes[0, 1]
    ax2.axis("off")
    
    stiffness = metrics.get('stiffness_eigs', [0, 0])
    is_stable = all(s < 0 for s in stiffness) if stiffness is not None else False
    
    text = f"""Control Diagnostics
{'=' * 35}

Position (particle):
  ({px_mm:.4f}, {py_mm:.4f}) mm

Target:
  ({tx_mm:.4f}, {ty_mm:.4f}) mm

Trap Centre:
  ({metrics.get('trap_x_mm', 0):.4f}, {metrics.get('trap_y_mm', 0):.4f}) mm

Distances:
  particle → target: {metrics.get('err_mm', 0):.4f} mm
  trap → target:     {metrics.get('trap_to_target_mm', 0):.4f} mm
  particle → trap:   {metrics.get('particle_to_trap_mm', 0):.4f} mm

Force:
  |F| = {metrics.get('F_mag_N', 0):.3e} N
  direction: cos(θ) = {metrics.get('cos_to_target', 0):.3f}

Stiffness (eigenvalues):
  λ₁ = {stiffness[0]:.3e}
  λ₂ = {stiffness[1]:.3e}
  Stable: {'✓ YES' if is_stable else '✗ NO'}

Step Limiter:
  Limited: {'YES' if metrics.get('step_limited', False) else 'NO'}
  Scale: {metrics.get('step_scale', 1.0):.3f}
  Raw step: {metrics.get('raw_step_mm', 0):.4f} mm

Control:
  vA = {metrics.get('vA', 0) * 1e4:.2f} ×10⁻⁴ m/s
  vB = {metrics.get('vB', 0) * 1e4:.2f} ×10⁻⁴ m/s
  φA = {metrics.get('phiA', 0):.3f} rad
  φB = {metrics.get('phiB', 0):.3f} rad
"""
    
    ax2.text(0.05, 0.95, text, transform=ax2.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.9))
    ax2.set_title("Metrics", fontweight="bold")
    
    # ============ Panel 3: Tracking Error Time Series ============
    ax3 = axes[1, 0]
    
    err_history = metrics.get('err_history', [])
    trap_to_target_history = metrics.get('trap_to_target_history', [])
    particle_to_trap_history = metrics.get('particle_to_trap_history', [])
    
    if len(err_history) >= 2:
        t_steps = np.arange(len(err_history))
        ax3.plot(t_steps, err_history, 'r-', linewidth=2, label='p→target')
        ax3.plot(t_steps, trap_to_target_history, 'b--', linewidth=1.5, label='trap→target')
        ax3.plot(t_steps, particle_to_trap_history, 'g:', linewidth=1.5, label='p→trap')
        ax3.set_xlabel("Step")
        ax3.set_ylabel("Distance (mm)")
        ax3.legend(loc="upper right", fontsize=8)
        ax3.grid(True, alpha=0.3)
    
    ax3.set_title("Tracking Error Over Time", fontweight="bold")
    
    # ============ Panel 4: Stiffness Time Series ============
    ax4 = axes[1, 1]
    
    stiffness_history = metrics.get('stiffness_history', [])
    
    if len(stiffness_history) >= 2:
        t_steps = np.arange(len(stiffness_history))
        eig1 = [s[0] for s in stiffness_history]
        eig2 = [s[1] for s in stiffness_history]
        ax4.plot(t_steps, eig1, 'b-', linewidth=2, label='λ₁')
        ax4.plot(t_steps, eig2, 'r--', linewidth=2, label='λ₂')
        ax4.axhline(y=0, color='k', linestyle=':', alpha=0.5)
        ax4.set_xlabel("Step")
        ax4.set_ylabel("Eigenvalue")
        ax4.legend(loc="upper right", fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        # Mark unstable regions
        for i, (e1, e2) in enumerate(stiffness_history):
            if e1 > 0 or e2 > 0:
                ax4.axvspan(i - 0.5, i + 0.5, alpha=0.2, color='red')
    
    ax4.set_title("Trap Stiffness (Stability)", fontweight="bold")
    
    # Overall title
    fig.suptitle(f"Stage F Demo: {metrics.get('demo_name', 'Demo')}", fontsize=14, fontweight="bold")
    
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def run_demo(
    demo_type: str,
    output_dir: Path,
    T_steps: int = 300,
    render_stride: int = 2,
    use_mpc: bool = True,
) -> None:
    """
    Run a demo simulation and generate GIF.
    
    Parameters
    ----------
    demo_type : str
        Either "sweep" or "circle"
    output_dir : Path
        Output directory for frames and GIF
    T_steps : int
        Number of timesteps
    render_stride : int
        Render every N steps
    use_mpc : bool
        Use MPC mode
    """
    print("=" * 70)
    print(f"STAGE F DEMO: {demo_type.upper()}")
    print("=" * 70)
    
    # Create output directories
    frames_dir = output_dir / f"frames_{demo_type}"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    # ===== Domain + Physics Setup =====
    domain = DishDomain(Lx=2e-3, Ly=2e-3, Nx=120, Ny=120)
    medium = MediumProps(f=2e6, c0=1500.0, rho0=1000.0, loss_eta=1e-3, kz=0.0, coupling_alpha=1.0)
    particle = ParticleProps(a=5e-6, rho_p=1050.0, c_p=2350.0)
    
    # Evaluator config with Stage A 2D forcing enabled
    # sigma_y controls the y-coupling decay. Larger sigma_y = more forgiving for
    # transducers positioned away from y=0, but less y-control authority.
    # With sigma_y=0.15mm and transducers at y~0.05mm, coupling is ~95%.
    #
    # Force scaling: With v_amp=0.05 m/s, raw Gor'kov forces are strong enough.
    # alpha_g tunes the effective force. Target: ~10-50µm displacement per step.
    cfg = EvaluatorConfig(
        sigma_x=0.10e-3,
        sigma_y=0.15e-3,  # Stage A: 2D forcing (larger for better coupling)
        bottom_band=0.25e-3,
        dt=5e-3,
        viscosity=1e-3,
        border_penalty=1e6,
        smooth_u=0.0,
        alpha_g=1e3,  # Reduced from 1e6 to match higher transducer amplitudes
        max_step=0.05e-3,
        use_2d_forcing=True,  # Stage A: Enable 2D forcing
    )
    
    ev = BottomFootprint25DEvaluator(domain, medium, particle, cfg)
    
    print(f"Domain: {domain.Lx*1e3:.1f} x {domain.Ly*1e3:.1f} mm")
    print(f"Grid: {domain.Nx} x {domain.Ny}")
    print(f"2D Forcing: sigma_x={cfg.sigma_x*1e3:.2f}mm, sigma_y={cfg.sigma_y*1e3:.2f}mm")
    
    # ===== Generate path =====
    if demo_type == "sweep":
        path = generate_sweep_path(domain.Lx, domain.Ly, T_steps,
                                   x_start_frac=0.3, x_end_frac=0.7, y_center_frac=0.55)
        demo_name = "Slow Sweep (X direction)"
    elif demo_type == "circle":
        path = generate_circle_path(domain.Lx, domain.Ly, T_steps,
                                    radius_frac=0.30, center_x_frac=0.5, center_y_frac=0.55)
        demo_name = "Large Circle"
    else:
        raise ValueError(f"Unknown demo type: {demo_type}")
    
    path_mm = path * 1e3
    print(f"\nPath: {demo_name}")
    print(f"  Start: ({path[0, 0]*1e3:.3f}, {path[0, 1]*1e3:.3f}) mm")
    print(f"  End: ({path[-1, 0]*1e3:.3f}, {path[-1, 1]*1e3:.3f}) mm")
    
    # ===== Controller setup =====
    bounds = ControlBounds(
        x_min=0.0, x_max=domain.Lx,
        y_min=0.0, y_max=cfg.bottom_band,
        v_min=0.0, v_max=0.1,  # Up to 100 mm/s for realistic pressures
    )
    
    rate_limits = ControlRateLimits(
        dx_max=0.08e-3,
        dy_max=0.04e-3,
        dv_max=0.01,  # Allow amplitude changes up to 10 mm/s per step
        dphi_max=0.4,
    )
    
    controller_cfg = ControllerConfig(
        tracking_weight=1e6,
        effort_weight=0.001,
        stiffness_weight=0.0001,
        trap_weight=2e6,  # Stage D: Trap-steering
        particle_trap_weight=0.5e6,
        horizon=4,
        n_candidates=60,
        position_noise=0.04e-3,
        amplitude_noise=0.005,  # 5 mm/s noise for exploration at higher amplitudes
        phase_noise=0.25,
        dt=cfg.dt,
        viscosity=cfg.viscosity,
        particle_radius=particle.a,
    )
    
    safety_cfg = SafetyConfig(
        min_stiffness=-1e-8,
        min_transducer_separation=0.15e-3,
        boundary_margin=0.08e-3,
        reject_saddle_proximity=0.25e-3,
        max_control_magnitude=0.2,  # Increased for higher amplitude operation
    )
    
    controller = ParticleController(
        evaluator=ev,
        config=controller_cfg,
        safety_config=safety_cfg,
        bounds=bounds,
        rate_limits=rate_limits,
    )
    
    # ===== Initial state =====
    state = ControlState(x=float(path[0, 0]), y=float(path[0, 1]))
    
    # Transducer amplitude: Use realistic value for MHz acoustic tweezers
    # p = rho * c * v => v = p / (rho * c)
    # For ~0.1 MPa pressure: v = 0.1e6 / (1000 * 1500) ≈ 0.067 m/s
    v_amp = 0.05  # 50 mm/s - gives ~75 kPa pressure
    control = ControlVector(
        xA=0.3 * domain.Lx, yA=0.05e-3,  # Closer to boundary for better coupling
        xB=0.7 * domain.Lx, yB=0.05e-3,
        vA=v_amp, vB=v_amp,
        phiA=0.0, phiB=np.pi,
        bounds=bounds,
        rate_limits=rate_limits,
    )
    
    # TrapTracker for Stage C
    trap_tracker = TrapTracker()
    
    # ===== Simulation loop =====
    traj_xy_mm: list[tuple[float, float]] = [(state.x * 1e3, state.y * 1e3)]
    frame_paths: list[Path] = []
    
    # Metrics history
    err_history: list[float] = []
    trap_to_target_history: list[float] = []
    particle_to_trap_history: list[float] = []
    stiffness_history: list[list[float]] = []
    
    cyl_r_mm = (2.0 * cfg.sigma_x) * 1e3
    
    print(f"\nRunning simulation (T={T_steps}, MPC={use_mpc})...")
    print("-" * 100)
    print("Step  px_mm    py_mm    err_mm   trap→tgt  p→trap   cos_θ   lim  stiff_min")
    print("-" * 100)
    
    for t in range(T_steps - 1):
        target = ControlState(x=float(path[t + 1, 0]), y=float(path[t + 1, 1]))
        
        # Horizon targets for MPC
        horizon_end = min(t + 1 + controller_cfg.horizon, T_steps)
        targets_horizon = [
            ControlState(x=float(path[i, 0]), y=float(path[i, 1]))
            for i in range(t + 1, horizon_end)
        ]
        
        # Guided transducer geometry
        xA_g, yA_g, xB_g, yB_g = compute_guided_control(
            target.x, target.y, control, domain.Lx
        )
        guided_control = ControlVector(
            xA=xA_g, yA=yA_g, xB=xB_g, yB=yB_g,
            vA=control.vA, vB=control.vB,
            phiA=control.phiA, phiB=control.phiB,
            bounds=bounds, rate_limits=rate_limits,
        )
        
        # Controller step
        prev_state = state
        new_control, new_state, info = controller.step(
            state=state,
            target=target,
            current_control=guided_control,
            targets_horizon=targets_horizon if use_mpc else None,
            use_mpc=use_mpc,
        )
        
        control = new_control
        state = new_state
        traj_xy_mm.append((state.x * 1e3, state.y * 1e3))
        
        # Get field for visualization
        u2p = control.to_control2pucks()
        _, _, _, info_field, field, U, Fx, Fy = ev.step(
            xp=state.x, yp=state.y,
            target_x=target.x, target_y=target.y,
            u=u2p, u_prev=None, return_fields=True,
        )
        
        # Trap centre from info
        trap_xy = info.get("trap_xy", (state.x, state.y))
        trap_x, trap_y = trap_xy if trap_xy else (state.x, state.y)
        
        stiffness_eigs = info.get("trap_stiffness_eigs", np.array([0.0, 0.0]))
        if stiffness_eigs is None:
            stiffness_eigs = np.array([0.0, 0.0])
        
        # Metrics
        err_mm = state.distance_to(target) * 1e3
        trap_to_target_mm = np.sqrt((trap_x - target.x)**2 + (trap_y - target.y)**2) * 1e3
        particle_to_trap_mm = np.sqrt((state.x - trap_x)**2 + (state.y - trap_y)**2) * 1e3
        
        # Directional check
        d_desired = np.array([target.x - prev_state.x, target.y - prev_state.y])
        d_actual = np.array([state.x - prev_state.x, state.y - prev_state.y])
        d_desired_norm = np.linalg.norm(d_desired)
        d_actual_norm = np.linalg.norm(d_actual)
        cos_to_target = float(np.dot(d_desired, d_actual) / (d_desired_norm * d_actual_norm + 1e-12))
        
        err_history.append(err_mm)
        trap_to_target_history.append(trap_to_target_mm)
        particle_to_trap_history.append(particle_to_trap_mm)
        stiffness_history.append(list(stiffness_eigs))
        
        # Print progress
        step_limited = info.get("step_limited", False)
        min_eig = float(np.min(stiffness_eigs))
        
        # Print transducer positions every 10 steps
        if (t + 1) % 10 == 0 or t == 0:
            print(f"{t+1:04d}  p=({state.x*1e3:.2f},{state.y*1e3:.2f})  "
                  f"tgt=({target.x*1e3:.2f},{target.y*1e3:.2f})  "
                  f"trans: xA={control.xA*1e3:.2f} xB={control.xB*1e3:.2f}  "
                  f"trap→tgt={trap_to_target_mm:.2f}mm")
        else:
            print(f"{t+1:04d}  {state.x*1e3:7.4f}  {state.y*1e3:7.4f}  {err_mm:7.4f}  "
                  f"{trap_to_target_mm:7.4f}  {particle_to_trap_mm:7.4f}  {cos_to_target:6.3f}  "
                  f"{'Y' if step_limited else 'N':3s}  {min_eig:.2e}")
        
        # Render frame
        if (t + 1) % render_stride == 0:
            frame_path = frames_dir / f"frame_{t+1:05d}.png"
            
            cylinders = [
                Cylinder2D(x_mm=control.xA * 1e3, y_mm=control.yA * 1e3, r_mm=cyl_r_mm),
                Cylinder2D(x_mm=control.xB * 1e3, y_mm=control.yB * 1e3, r_mm=cyl_r_mm),
            ]
            
            metrics = {
                'demo_name': demo_name,
                'trap_x_mm': trap_x * 1e3,
                'trap_y_mm': trap_y * 1e3,
                'err_mm': err_mm,
                'trap_to_target_mm': trap_to_target_mm,
                'particle_to_trap_mm': particle_to_trap_mm,
                'cos_to_target': cos_to_target,
                'F_mag_N': info.get("fx", 0)**2 + info.get("fy", 0)**2,
                'stiffness_eigs': list(stiffness_eigs),
                'step_limited': step_limited,
                'step_scale': info.get("step_scale", 1.0),
                'raw_step_mm': info.get("raw_step_mm", 0.0),
                'vA': control.vA,
                'vB': control.vB,
                'phiA': control.phiA,
                'phiB': control.phiB,
                'err_history': err_history.copy(),
                'trap_to_target_history': trap_to_target_history.copy(),
                'particle_to_trap_history': particle_to_trap_history.copy(),
                'stiffness_history': stiffness_history.copy(),
            }
            
            render_demo_frame(
                out_png=frame_path,
                x_mm=field.x * 1e3,
                y_mm=field.y * 1e3,
                U=U,
                particle_xy_mm=(state.x * 1e3, state.y * 1e3),
                target_xy_mm=(target.x * 1e3, target.y * 1e3),
                trap_centre_mm=(trap_x * 1e3, trap_y * 1e3),
                track_xy_mm=traj_xy_mm.copy(),
                target_path_mm=path_mm,
                cylinders=cylinders,
                metrics=metrics,
                step=t + 1,
                total_steps=T_steps,
            )
            frame_paths.append(frame_path)
    
    print("-" * 100)
    
    # ===== Generate GIF =====
    gif_path = output_dir / f"demo_{demo_type}.gif"
    print(f"\nGenerating GIF: {gif_path}")
    
    images = [imageio.imread(str(fp)) for fp in frame_paths]
    imageio.mimsave(str(gif_path), images, duration=0.1)
    
    print(f"  {len(images)} frames -> {gif_path}")
    
    # ===== Summary =====
    print(f"\n{'=' * 70}")
    print(f"DEMO SUMMARY: {demo_name}")
    print(f"{'=' * 70}")
    print(f"  Final tracking error: {err_history[-1]:.4f} mm")
    print(f"  Mean tracking error: {np.mean(err_history):.4f} mm")
    print(f"  Max tracking error: {np.max(err_history):.4f} mm")
    print(f"  Mean trap→target: {np.mean(trap_to_target_history):.4f} mm")
    print(f"  Mean p→trap: {np.mean(particle_to_trap_history):.4f} mm")
    
    # Count unstable steps
    n_unstable = sum(1 for s in stiffness_history if s[0] > 0 or s[1] > 0)
    print(f"  Unstable steps: {n_unstable}/{len(stiffness_history)}")
    
    # Save trajectory
    traj_arr = np.array(traj_xy_mm)
    np.save(output_dir / f"traj_{demo_type}_mm.npy", traj_arr)
    np.save(output_dir / f"desired_{demo_type}_mm.npy", path_mm)
    
    print(f"\nOutputs saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Stage F Demo: Large 2D Trajectories")
    parser.add_argument("--demo", choices=["sweep", "circle", "both"], default="both",
                       help="Which demo to run")
    parser.add_argument("--steps", type=int, default=300, help="Number of timesteps")
    parser.add_argument("--stride", type=int, default=2, help="Render every N steps")
    args = parser.parse_args()
    
    REPO = Path(__file__).resolve().parents[1]
    RESULTS = REPO / "results" / "stage_f_demos"
    RESULTS.mkdir(parents=True, exist_ok=True)
    
    demos_to_run = ["sweep", "circle"] if args.demo == "both" else [args.demo]
    
    for demo in demos_to_run:
        run_demo(
            demo_type=demo,
            output_dir=RESULTS,
            T_steps=args.steps,
            render_stride=args.stride,
            use_mpc=True,
        )
        print("\n")


if __name__ == "__main__":
    main()
