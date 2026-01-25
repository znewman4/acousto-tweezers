#!/usr/bin/env python3
"""
4-Puck Greedy Surf Controller Demo: 4-Transducer System with Gating.

GREEDY SURFING WITH 4 TRANSDUCERS + ON/OFF GATING:
Extends the 3-puck surfing controller to the full 4-transducer system with:
- Transducer D for enhanced 2D control authority
- Per-transducer ON/OFF gating (TOGGLE_*_ON/OFF actions)
- Move-while-off actions (reposition silent transducers)

Key insight: This controller "surfs" on the acoustic radiation force field,
selecting macro actions that maximize force alignment with the desired direction.
Transducer D and gating provide additional degrees of freedom for navigating
challenging regions of the workspace.

Scoring function at each step:
    score = w_align * (F̂ · d̂)    # unit alignment
          + w_push  * (F · d̂)     # force magnitude in direction
          - w_switch * I[action != prev]  # switching penalty

Outputs:
    - results/4puck_demo_surf_greedy/run_YYYYMMDD_HHMMSS/
        - 4puck_demo_surf_greedy.gif
        - summary.png
        - steps.csv

Usage:
    python scripts/4puck_demo_surf_greedy.py --path circle
    python scripts/4puck_demo_surf_greedy.py --path circle --fast
    python scripts/4puck_demo_surf_greedy.py --path circle --action_subset standard
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import argparse
import csv
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from typing import Optional, List
import json
from collections import Counter

# Add project root to path
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from acousto.force import ParticleProps, gorkov_potential_and_force_2d, bilinear_sample_vec
from acousto.analysis import find_trap_center

from tweezers.control import (
    DishDomain, MediumProps, EvaluatorConfig,
    Control4Pucks, default_4puck_config,
)
from tweezers.control.evaluator_4pucks import Evaluator4Pucks

from scripts.macro_actions_4puck import (
    MacroAction4Puck,
    MacroActionType4Puck,
    apply_macro_action_4puck,
    get_standard_actions_4puck,
    get_3puck_compatible_actions_4puck,
    get_gating_actions_4puck,
    get_all_actions_4puck,
)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class GreedySurfConfig4Puck:
    """Configuration for 4-puck greedy surf controller."""
    # Scoring weights
    w_align: float = 1.0       # Weight for unit alignment (F̂ · d̂)
    w_push: float = 1e6        # Weight for force projection (F · d̂)
    w_switch: float = 0.05     # Penalty for switching action
    
    # Force threshold
    min_force_threshold: float = 1e-12  # N
    
    # Macro action parameters
    macro_magnitude: float = 0.05e-3   # Position step size (m)
    macro_phase_step: float = 0.15     # Phase step (rad)
    macro_amplitude_step: float = 0.01 # Amplitude step
    
    # Dynamics
    dt: float = 5e-3           # Integration timestep
    max_step: float = 0.08e-3  # Max particle displacement per step


# =============================================================================
# Step Log
# =============================================================================

@dataclass
class GreedySurfStep4Puck:
    """Log entry for one step of 4-puck greedy surf control."""
    step_idx: int
    particle_x: float
    particle_y: float
    target_x: float
    target_y: float
    tracking_error: float
    
    chosen_action: str
    action_switched: bool
    n_actions_evaluated: int
    
    Fp_x: float
    Fp_y: float
    Fp_mag: float
    Fp_hat_dot_d: float  # Unit alignment
    Fp_dot_d: float      # Force projection
    score: float
    
    trap_candidate_x: float = np.nan
    trap_candidate_y: float = np.nan
    trap_stable: bool = False
    stiff_min: float = np.nan
    
    solver_time_ms: float = 0.0
    target_idx: int = 0
    dist_to_target: float = 0.0
    target_advanced: bool = False
    
    # Circle metrics
    cross_track_error: float = 0.0
    tangential_alignment: float = 0.0
    
    # 4-puck specific
    gates_active: str = "ABCD"  # Which transducers are on


# =============================================================================
# Greedy Surf Controller for 4 Pucks
# =============================================================================

class GreedySurfController4Puck:
    """
    Greedy controller for 4-puck system with gating.
    
    At each step:
    1. Compute desired direction d̂ toward target
    2. For each candidate action:
       a. Clone control, apply action
       b. Solve PDE to get force field
       c. Sample force at particle position
       d. Compute score based on alignment + magnitude
    3. Pick action with highest score
    4. Apply chosen action and integrate particle
    """
    
    def __init__(
        self,
        evaluator: Evaluator4Pucks,
        config: GreedySurfConfig4Puck,
        action_set: List[MacroActionType4Puck] | None = None,
    ):
        self.ev = evaluator
        self.config = config
        
        # Default action set (standard 4-puck actions)
        if action_set is None:
            action_set = get_standard_actions_4puck()
        
        self.action_types = action_set
        self.prev_action: MacroActionType4Puck | None = None
        self.best_field: tuple | None = None
        
    def _make_action(self, action_type: MacroActionType4Puck) -> MacroAction4Puck:
        """Create MacroAction4Puck from type with current config parameters."""
        return MacroAction4Puck(
            action_type=action_type,
            magnitude=self.config.macro_magnitude,
            phase_step=self.config.macro_phase_step,
            amplitude_step=self.config.macro_amplitude_step,
        )
    
    def _compute_action_score(
        self,
        action_type: MacroActionType4Puck,
        Fp_x: float,
        Fp_y: float,
        d_hat_x: float,
        d_hat_y: float,
        w_align: float | None = None,
        w_push: float | None = None,
        w_switch: float | None = None,
    ) -> tuple[float, float, float, float]:
        """
        Compute score for an action given force and desired direction.
        
        Returns: (score, Fp_hat_dot_d, Fp_dot_d, Fp_mag)
        """
        if w_align is None:
            w_align = self.config.w_align
        if w_push is None:
            w_push = self.config.w_push
        if w_switch is None:
            w_switch = self.config.w_switch
            
        Fp_mag = np.sqrt(Fp_x**2 + Fp_y**2)
        eps = 1e-15
        
        # Unit force direction
        Fp_hat_x = Fp_x / (Fp_mag + eps)
        Fp_hat_y = Fp_y / (Fp_mag + eps)
        
        # Alignment scores
        Fp_hat_dot_d = Fp_hat_x * d_hat_x + Fp_hat_y * d_hat_y
        Fp_dot_d = Fp_x * d_hat_x + Fp_y * d_hat_y
        
        # Build score
        score = w_align * Fp_hat_dot_d + w_push * Fp_dot_d
        
        # Penalty for near-zero force
        if Fp_mag < self.config.min_force_threshold:
            score -= 0.5
        
        # Switching penalty
        if self.prev_action is not None and action_type != self.prev_action:
            score -= w_switch
        
        return score, Fp_hat_dot_d, Fp_dot_d, Fp_mag
    
    def step(
        self,
        particle_x: float,
        particle_y: float,
        target_x: float,
        target_y: float,
        ctrl: Control4Pucks,
        step_idx: int = 0,
        desired_direction: tuple[float, float] | None = None,
        weight_overrides: dict | None = None,
    ) -> tuple[Control4Pucks, float, float, GreedySurfStep4Puck]:
        """
        Execute one greedy surf control step.
        
        Returns:
            new_ctrl: Updated control configuration
            new_x, new_y: New particle position after integration
            log: GreedySurfStep4Puck with all diagnostic info
        """
        total_solver_time = 0.0
        
        # Get scoring weights
        w_align = self.config.w_align
        w_push = self.config.w_push
        w_switch = self.config.w_switch
        if weight_overrides:
            w_switch = weight_overrides.get('w_switch', w_switch)
            w_push = weight_overrides.get('w_push', w_push)
        
        # Desired direction
        if desired_direction is not None:
            d_hat_x, d_hat_y = desired_direction
            dist_to_target = np.sqrt((target_x - particle_x)**2 + (target_y - particle_y)**2)
        else:
            dx = target_x - particle_x
            dy = target_y - particle_y
            dist_to_target = np.sqrt(dx**2 + dy**2)
            eps = 1e-12
            d_hat_x = dx / (dist_to_target + eps)
            d_hat_y = dy / (dist_to_target + eps)
        
        # Evaluate all actions
        best_score = -np.inf
        best_action_type = MacroActionType4Puck.HOLD
        best_Fp = (0.0, 0.0)
        best_metrics = (0.0, 0.0, 0.0)
        best_field = None
        best_trap_info = (np.nan, np.nan, False, np.nan)
        best_ctrl = ctrl
        
        for action_type in self.action_types:
            # Clone and apply action
            action = self._make_action(action_type)
            u_cand = apply_macro_action_4puck(ctrl, action)
            u_cand = self.ev.clip_control(u_cand)
            
            # Solve PDE
            t0 = time.perf_counter()
            vb = self.ev.control_to_forcing_band_vb(u_cand)
            field = self.ev.op.solve_for_bottom_vb(vb)
            U, Fx, Fy = gorkov_potential_and_force_2d(field, self.ev.particle)
            
            # Apply alpha_g scaling
            Fx_scaled = Fx * self.ev.cfg.alpha_g
            Fy_scaled = Fy * self.ev.cfg.alpha_g
            
            total_solver_time += (time.perf_counter() - t0) * 1000.0
            
            # Sample force at particle position
            fx, fy = bilinear_sample_vec(field.x, field.y, Fx_scaled, Fy_scaled, particle_x, particle_y)
            
            # Compute score
            score, Fp_hat_dot_d, Fp_dot_d, Fp_mag = self._compute_action_score(
                action_type, fx, fy, d_hat_x, d_hat_y,
                w_align=w_align, w_push=w_push, w_switch=w_switch,
            )
            
            if score > best_score:
                best_score = score
                best_action_type = action_type
                best_Fp = (fx, fy)
                best_metrics = (Fp_hat_dot_d, Fp_dot_d, Fp_mag)
                best_field = (field, U, Fx_scaled, Fy_scaled)
                best_ctrl = u_cand
                
                # Get trap info
                trap = find_trap_center(
                    field.x, field.y, U, Fx, Fy,
                    particle_x=particle_x, particle_y=particle_y,
                    search_radius=0.4e-3,
                )
                best_trap_info = (trap.x, trap.y, trap.is_stable, trap.min_eigenvalue)
        
        # Apply best action
        self.prev_action = best_action_type
        self.best_field = best_field
        
        # Integrate particle motion
        Fp_x, Fp_y = best_Fp
        new_x, new_y = self._integrate_particle(
            particle_x, particle_y, Fp_x, Fp_y
        )
        
        # Build log
        action_switched = (self.prev_action is not None and 
                          best_action_type != self.prev_action)
        
        # Get active gates
        gates = ""
        if best_ctrl.gateA: gates += "A"
        if best_ctrl.gateB: gates += "B"
        if best_ctrl.gateC: gates += "C"
        if best_ctrl.gateD: gates += "D"
        
        log = GreedySurfStep4Puck(
            step_idx=step_idx,
            particle_x=new_x,
            particle_y=new_y,
            target_x=target_x,
            target_y=target_y,
            tracking_error=np.sqrt((new_x - target_x)**2 + (new_y - target_y)**2),
            chosen_action=best_action_type.name,
            action_switched=action_switched,
            n_actions_evaluated=len(self.action_types),
            Fp_x=Fp_x,
            Fp_y=Fp_y,
            Fp_mag=best_metrics[2],
            Fp_hat_dot_d=best_metrics[0],
            Fp_dot_d=best_metrics[1],
            score=best_score,
            trap_candidate_x=best_trap_info[0],
            trap_candidate_y=best_trap_info[1],
            trap_stable=best_trap_info[2],
            stiff_min=best_trap_info[3],
            solver_time_ms=total_solver_time,
            dist_to_target=dist_to_target,
            gates_active=gates,
        )
        
        return best_ctrl, new_x, new_y, log
    
    def _integrate_particle(
        self,
        x: float,
        y: float,
        Fx: float,
        Fy: float,
    ) -> tuple[float, float]:
        """Integrate particle position under force."""
        dt = self.config.dt
        max_step = self.config.max_step
        
        # Stokes drag: v = F / (6πηa)
        # For simplicity, we use a mobility coefficient
        # With viscosity=1e-3 Pa·s, a=5e-6 m: mobility ≈ 1e6 m/(N·s)
        viscosity = 1e-3
        a = self.ev.particle.a
        mobility = 1.0 / (6 * np.pi * viscosity * a)
        
        vx = Fx * mobility
        vy = Fy * mobility
        
        dx = vx * dt
        dy = vy * dt
        
        # Limit step size
        step_size = np.sqrt(dx**2 + dy**2)
        if step_size > max_step:
            scale = max_step / step_size
            dx *= scale
            dy *= scale
        
        new_x = x + dx
        new_y = y + dy
        
        # Clip to domain
        new_x = np.clip(new_x, 0.0, self.ev.domain.Lx)
        new_y = np.clip(new_y, 0.0, self.ev.domain.Ly)
        
        return float(new_x), float(new_y)


# =============================================================================
# Path Generators
# =============================================================================

def make_circle_path(cx: float, cy: float, radius: float, n_points: int) -> np.ndarray:
    """Generate circular path with CCW direction."""
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    x = cx + radius * np.cos(angles)
    y = cy + radius * np.sin(angles)
    return np.column_stack([x, y])


def make_straight_line_path(x0: float, y0: float, x1: float, y1: float, n_points: int) -> np.ndarray:
    """Generate straight line path."""
    x = np.linspace(x0, x1, n_points)
    y = np.linspace(y0, y1, n_points)
    return np.column_stack([x, y])


# =============================================================================
# Visualization
# =============================================================================

def plot_frame_4puck(
    field, U, Fx_scaled, Fy_scaled,
    domain, ctrl: Control4Pucks, particle_x: float, particle_y: float,
    target_x: float, target_y: float,
    step_idx: int, chosen_action: str, tracking_error: float,
    path: np.ndarray | None = None,
    score: float = 0.0,
    gates_active: str = "ABCD",
    contour_limits: tuple | None = None,
    trajectory_history: List[tuple[float, float]] | None = None,
) -> plt.Figure:
    """Create visualization frame for 4-puck demo."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Gor'kov potential contours
    if contour_limits is not None:
        vmin, vmax = contour_limits
    else:
        vmin, vmax = np.nanpercentile(U, [5, 95])
    
    if vmax - vmin > 1e-20:
        levels = np.linspace(vmin, vmax, 25)
        X_mm = field.x * 1e3
        Y_mm = field.y * 1e3
        cf = ax.contourf(X_mm, Y_mm, U.T, levels=levels, cmap='viridis', extend='both')
        plt.colorbar(cf, ax=ax, label='Gor\'kov U (J)')
    else:
        ax.imshow(U.T, origin='lower', extent=[0, domain.Lx*1e3, 0, domain.Ly*1e3],
                  aspect='auto', cmap='viridis')
    
    # Force quiver (subsampled)
    step = 5
    X_q = field.x[::step] * 1e3
    Y_q = field.y[::step] * 1e3
    XX, YY = np.meshgrid(X_q, Y_q, indexing='ij')
    Fx_q = Fx_scaled[::step, ::step]
    Fy_q = Fy_scaled[::step, ::step]
    ax.quiver(XX, YY, Fx_q, Fy_q, color='white', alpha=0.5, scale=1e-7)
    
    # Path
    if path is not None:
        ax.plot(path[:, 0] * 1e3, path[:, 1] * 1e3, 'w--', lw=2, alpha=0.7, label='path')
    
    # Particle trail (trajectory history)
    if trajectory_history is not None and len(trajectory_history) > 1:
        traj = np.array(trajectory_history)
        # Color gradient from old (faded) to new (bright)
        n_pts = len(traj)
        colors = plt.cm.Reds(np.linspace(0.3, 1.0, n_pts))
        for i in range(n_pts - 1):
            ax.plot([traj[i, 0], traj[i+1, 0]], [traj[i, 1], traj[i+1, 1]],
                    color=colors[i], lw=2, alpha=0.8)
        # Also draw small dots along trail
        ax.scatter(traj[:-1, 0], traj[:-1, 1], c='red', s=10, alpha=0.5, zorder=18)
    
    # Target (current waypoint)
    ax.scatter(target_x * 1e3, target_y * 1e3, s=200, c='yellow', marker='*',
               edgecolors='black', linewidths=1, zorder=20, label='target')
    
    # Particle
    ax.scatter(particle_x * 1e3, particle_y * 1e3, s=150, c='red', marker='o',
               edgecolors='white', linewidths=2, zorder=21, label='particle')
    
    # Transducers with gate visualization
    transducer_colors = {
        'A': ('blue', ctrl.xA, ctrl.yA, ctrl.gateA),
        'B': ('green', ctrl.xB, ctrl.yB, ctrl.gateB),
        'C': ('orange', ctrl.xC, ctrl.yC, ctrl.gateC),
        'D': ('purple', ctrl.xD, ctrl.yD, ctrl.gateD),
    }
    
    for name, (color, x, y, gate) in transducer_colors.items():
        marker = 'o' if gate else 'x'
        alpha = 1.0 if gate else 0.3
        ax.scatter(x * 1e3, y * 1e3, s=120, c=color, marker=marker,
                   edgecolors='white' if gate else 'gray', linewidths=1.5,
                   alpha=alpha, zorder=15, label=f'{name} ({"ON" if gate else "OFF"})')
    
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_xlim(0, domain.Lx * 1e3)
    ax.set_ylim(0, domain.Ly * 1e3)
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=8)
    
    title = (f"4-Puck Surf | Step {step_idx} | Action: {chosen_action}\n"
             f"Error: {tracking_error*1e6:.1f} µm | Score: {score:.3f} | Gates: {gates_active}")
    ax.set_title(title, fontsize=11)
    
    plt.tight_layout()
    return fig


# =============================================================================
# Contour Manager
# =============================================================================

class ContourManager:
    """Manages contour color scale for consistent visualization."""
    
    def __init__(self, mode: str = "warmup_fixed", warmup_frames: int = 10,
                 pct_lo: float = 5.0, pct_hi: float = 95.0, min_range: float = 1e-16):
        self.mode = mode
        self.warmup_frames = warmup_frames
        self.pct_lo = pct_lo
        self.pct_hi = pct_hi
        self.min_range = min_range
        
        self.warmup_values = []
        self.fixed_limits = None
    
    def update(self, U: np.ndarray, frame_idx: int) -> tuple[float, float] | None:
        """Update and return contour limits."""
        if self.mode == "per_frame":
            lo, hi = np.nanpercentile(U, [self.pct_lo, self.pct_hi])
            if hi - lo < self.min_range:
                return None
            return (lo, hi)
        
        elif self.mode == "warmup_fixed":
            if frame_idx < self.warmup_frames:
                self.warmup_values.append(U.flatten())
                lo, hi = np.nanpercentile(U, [self.pct_lo, self.pct_hi])
                if hi - lo < self.min_range:
                    return None
                return (lo, hi)
            else:
                if self.fixed_limits is None and len(self.warmup_values) > 0:
                    all_vals = np.concatenate(self.warmup_values)
                    lo, hi = np.nanpercentile(all_vals, [self.pct_lo, self.pct_hi])
                    if hi - lo < self.min_range:
                        self.fixed_limits = (None, None)
                    else:
                        self.fixed_limits = (lo, hi)
                return self.fixed_limits if self.fixed_limits[0] is not None else None
        
        elif self.mode == "fixed":
            return self.fixed_limits
        
        return None


# =============================================================================
# Circle Direction Helpers
# =============================================================================

def compute_circle_direction_tangent_plus_radial(
    particle_x: float, particle_y: float,
    center_x: float, center_y: float, radius: float,
    k_radial: float = 2.0,
) -> tuple[float, float, float]:
    """
    Compute desired direction for circle following.
    
    Returns (d_hat_x, d_hat_y, cross_track_error).
    """
    # Vector from center to particle
    dx = particle_x - center_x
    dy = particle_y - center_y
    r = np.sqrt(dx**2 + dy**2)
    
    eps = 1e-12
    # Unit radial vector (from center outward)
    r_hat_x = dx / (r + eps)
    r_hat_y = dy / (r + eps)
    
    # Tangent vector (CCW)
    t_hat_x = -r_hat_y
    t_hat_y = r_hat_x
    
    # Cross-track error (positive = outside circle)
    cross_track = r - radius
    
    # Combined direction: tangent + radial correction
    # d = t̂ - k_radial * (r - R) * r̂
    d_x = t_hat_x - k_radial * cross_track * r_hat_x / radius
    d_y = t_hat_y - k_radial * cross_track * r_hat_y / radius
    
    d_mag = np.sqrt(d_x**2 + d_y**2) + eps
    d_hat_x = d_x / d_mag
    d_hat_y = d_y / d_mag
    
    return d_hat_x, d_hat_y, cross_track


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="4-Puck Greedy Surf Controller Demo")
    parser.add_argument("--fast", action="store_true", help="Fast mode (100 steps)")
    parser.add_argument("--steps", type=int, default=400, help="Number of steps (default 400 for full circle)")
    parser.add_argument("--path", type=str, default="circle", choices=["line", "circle"],
                       help="Path type (default: circle)")
    parser.add_argument("--action_subset", type=str, default="standard",
                       choices=["standard", "3puck_compat", "full", "gating"],
                       help="Action subset: standard=21, 3puck_compat=20, full=53, gating=8")
    parser.add_argument("--coarse", action="store_true", help="Use coarse grid")
    parser.add_argument("--render_stride", type=int, default=None,
                       help="Render every N steps (default: 3 if --fast, else 2)")
    
    # Macro action step sizes
    parser.add_argument("--macro_step_pos_um", type=float, default=50.0,
                       help="Position step for macro actions (µm)")
    parser.add_argument("--macro_step_phase_rad", type=float, default=0.3,
                       help="Phase step for macro actions (radians)")
    
    # Target gating
    parser.add_argument("--target_hold_tol_um", type=float, default=120.0,
                       help="Target only advances when within this distance (µm)")
    
    # Circle direction
    parser.add_argument("--k_radial", type=float, default=2.0,
                       help="Radial correction gain for circle following")
    
    # Scoring weights
    parser.add_argument("--w_align", type=float, default=1.0)
    parser.add_argument("--w_push", type=float, default=1e6)
    parser.add_argument("--w_switch", type=float, default=0.05)
    
    args = parser.parse_args()
    
    REPO = Path(__file__).resolve().parents[1]
    RESULTS = REPO / "results"
    
    # Create timestamped output directory
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS / "4puck_demo_surf_greedy" / f"run_{run_timestamp}"
    frames_dir = out_dir / "frames"
    
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("4-PUCK GREEDY SURF CONTROLLER DEMO")
    print("=" * 70)
    print(f"Output: {out_dir}")
    
    # ===== Domain + Physics =====
    if args.coarse or args.fast:
        domain = DishDomain(Lx=2e-3, Ly=2e-3, Nx=80, Ny=80)
    else:
        domain = DishDomain(Lx=2e-3, Ly=2e-3, Nx=100, Ny=100)
    
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
    
    ev = Evaluator4Pucks(domain, medium, particle, cfg)
    
    print(f"\nPhysics:")
    print(f"  Domain: {domain.Lx*1e3:.1f} x {domain.Ly*1e3:.1f} mm")
    print(f"  Grid: {domain.Nx} x {domain.Ny}")
    print(f"  alpha_g: {cfg.alpha_g:.0e}")
    print(f"  Transducers: 4 (A, B, C, D) with ON/OFF gating")
    
    # ===== Macro action step sizes =====
    macro_step_pos = args.macro_step_pos_um * 1e-6
    macro_step_phase = args.macro_step_phase_rad
    
    # ===== Controller Setup =====
    surf_config = GreedySurfConfig4Puck(
        w_align=args.w_align,
        w_push=args.w_push,
        w_switch=args.w_switch,
        dt=cfg.dt,
        max_step=cfg.max_step,
        macro_magnitude=macro_step_pos,
        macro_phase_step=macro_step_phase,
    )
    
    # Action set based on subset choice
    if args.action_subset == "standard":
        action_set = get_standard_actions_4puck()
    elif args.action_subset == "3puck_compat":
        action_set = get_3puck_compatible_actions_4puck()
    elif args.action_subset == "gating":
        action_set = get_gating_actions_4puck()
    else:  # full
        action_set = get_all_actions_4puck()
    
    controller = GreedySurfController4Puck(ev, surf_config, action_set)
    
    print(f"\nController (greedy 4-puck):")
    print(f"  w_align: {surf_config.w_align}")
    print(f"  w_push: {surf_config.w_push:.1e}")
    print(f"  w_switch: {surf_config.w_switch}")
    print(f"  macro_step_pos: {surf_config.macro_magnitude*1e6:.0f} µm")
    print(f"  macro_step_phase: {surf_config.macro_phase_step:.2f} rad")
    print(f"  Actions ({args.action_subset}): {len(controller.action_types)}")
    for a in controller.action_types:
        print(f"    - {a.name}")
    
    # ===== Path Setup =====
    T = 100 if args.fast else args.steps
    
    if args.path == "line":
        start_x = 0.5e-3
        start_y = domain.Ly * 0.5
        end_x = 1.5e-3
        end_y = domain.Ly * 0.5
        path = make_straight_line_path(start_x, start_y, end_x, end_y, T)
        path_name = "straight_line"
        is_circle = False
        print(f"\nPath: Straight line")
        print(f"  From: ({start_x*1e3:.2f}, {start_y*1e3:.2f}) mm")
        print(f"  To:   ({end_x*1e3:.2f}, {end_y*1e3:.2f}) mm")
    else:
        radius = 0.4e-3
        center_x = domain.Lx / 2
        center_y = domain.Ly * 0.55
        path = make_circle_path(center_x, center_y, radius, T)
        path_name = "circle"
        is_circle = True
        print(f"\nPath: Circle")
        print(f"  Center: ({center_x*1e3:.2f}, {center_y*1e3:.2f}) mm")
        print(f"  Radius: {radius*1e3:.2f} mm")
    
    print(f"  Steps: {T}")
    
    # Target gating
    target_hold_tol = args.target_hold_tol_um * 1e-6
    print(f"  Target hold tolerance: {args.target_hold_tol_um:.0f} µm")
    
    # ===== Initial State =====
    particle_x = float(path[0, 0])
    particle_y = float(path[0, 1])
    
    # Initial 4-puck configuration
    # A and B at bottom straddling particle, C above, D opposite C
    ctrl = Control4Pucks(
        xA=particle_x - 0.4e-3, yA=0.03e-3, vA=0.08, phiA=0.0,
        xB=particle_x + 0.4e-3, yB=0.03e-3, vB=0.08, phiB=np.pi,
        xC=particle_x, yC=0.20e-3, vC=0.08, phiC=np.pi/2,
        xD=particle_x, yD=1.8e-3, vD=0.05, phiD=-np.pi/2,  # D at top
        gateA=True, gateB=True, gateC=True, gateD=True,
    )
    ctrl = ev.clip_control(ctrl)
    
    print(f"\nInitial transducers:")
    print(f"  A: ({ctrl.xA*1e3:.2f}, {ctrl.yA*1e3:.2f}) mm, gate={'ON' if ctrl.gateA else 'OFF'}")
    print(f"  B: ({ctrl.xB*1e3:.2f}, {ctrl.yB*1e3:.2f}) mm, gate={'ON' if ctrl.gateB else 'OFF'}")
    print(f"  C: ({ctrl.xC*1e3:.2f}, {ctrl.yC*1e3:.2f}) mm, gate={'ON' if ctrl.gateC else 'OFF'}")
    print(f"  D: ({ctrl.xD*1e3:.2f}, {ctrl.yD*1e3:.2f}) mm, gate={'ON' if ctrl.gateD else 'OFF'}")
    
    # ===== Contour Manager =====
    contour_mgr = ContourManager(mode="warmup_fixed", warmup_frames=10)
    
    # ===== Control Loop =====
    print(f"\n{'='*70}")
    print("RUNNING 4-PUCK SURF CONTROLLER...")
    print(f"{'='*70}")
    
    step_logs: List[GreedySurfStep4Puck] = []
    traj_xy_mm: List[tuple[float, float]] = [(particle_x * 1e3, particle_y * 1e3)]
    action_history: List[str] = []
    frame_paths: List[Path] = []
    
    target_idx = 0
    target_advance_count = 0
    
    render_stride = args.render_stride if args.render_stride else (3 if args.fast else 2)
    
    # Circle metrics
    if is_circle:
        initial_angle = np.arctan2(particle_y - center_y, particle_x - center_x)
        angle_history = [initial_angle]
    
    t_start = time.perf_counter()
    
    for t in range(T - 1):
        target_x = float(path[target_idx, 0])
        target_y = float(path[target_idx, 1])
        
        # Compute desired direction
        if is_circle:
            d_hat_x, d_hat_y, cross_track = compute_circle_direction_tangent_plus_radial(
                particle_x, particle_y, center_x, center_y, radius, args.k_radial
            )
            desired_direction = (d_hat_x, d_hat_y)
        else:
            desired_direction = None
            cross_track = 0.0
        
        # Step controller
        ctrl, particle_x, particle_y, log = controller.step(
            particle_x, particle_y, target_x, target_y, ctrl, step_idx=t,
            desired_direction=desired_direction,
        )
        
        # Add circle metrics
        if is_circle:
            log.cross_track_error = cross_track
            current_angle = np.arctan2(particle_y - center_y, particle_x - center_x)
            angle_history.append(current_angle)
        
        # Target gating
        dist = np.sqrt((particle_x - target_x)**2 + (particle_y - target_y)**2)
        if dist < target_hold_tol:
            next_idx = (target_idx + 1) % len(path) if is_circle else min(target_idx + 1, len(path) - 1)
            if next_idx != target_idx:
                target_idx = next_idx
                target_advance_count += 1
                log.target_advanced = True
        
        log.target_idx = target_idx
        
        step_logs.append(log)
        traj_xy_mm.append((particle_x * 1e3, particle_y * 1e3))
        action_history.append(log.chosen_action)
        
        # Render frame
        if t % render_stride == 0 and controller.best_field is not None:
            field, U, Fx_scaled, Fy_scaled = controller.best_field
            contour_limits = contour_mgr.update(U, len(frame_paths))
            
            # Build trajectory history in mm for visualization
            traj_for_viz = [(x, y) for x, y in traj_xy_mm]
            
            fig = plot_frame_4puck(
                field, U, Fx_scaled, Fy_scaled,
                domain, ctrl, particle_x, particle_y,
                target_x, target_y,
                t, log.chosen_action, log.tracking_error,
                path=path, score=log.score, gates_active=log.gates_active,
                contour_limits=contour_limits,
                trajectory_history=traj_for_viz,
            )
            
            frame_path = frames_dir / f"frame_{t:04d}.png"
            fig.savefig(frame_path, dpi=100)
            plt.close(fig)
            frame_paths.append(frame_path)
        
        # Progress
        if t % 20 == 0:
            elapsed = time.perf_counter() - t_start
            pct = 100 * (t + 1) / (T - 1)
            print(f"  Step {t+1:4d}/{T-1} ({pct:5.1f}%) | "
                  f"Action: {log.chosen_action:25s} | "
                  f"Error: {log.tracking_error*1e6:6.1f} µm | "
                  f"Gates: {log.gates_active} | "
                  f"Time: {elapsed:.1f}s")
    
    total_time = time.perf_counter() - t_start
    print(f"\nCompleted in {total_time:.1f}s ({total_time/(T-1)*1000:.1f} ms/step)")
    
    # ===== Analysis =====
    errors = np.array([log.tracking_error for log in step_logs])
    alignments = np.array([log.Fp_hat_dot_d for log in step_logs])
    pushes = np.array([log.Fp_dot_d for log in step_logs])
    
    switch_count = sum(1 for log in step_logs if log.action_switched)
    
    # Circle metrics
    if is_circle:
        angles = np.array(angle_history)
        diffs = np.diff(np.unwrap(angles))
        total_angle = np.sum(diffs)
        angle_progress_deg = np.degrees(total_angle)
        
        cross_errors = np.array([log.cross_track_error for log in step_logs])
        mean_cross_track_error_um = np.abs(cross_errors).mean() * 1e6
        
        tangential_alignments = np.array([log.tangential_alignment for log in step_logs])
        pct_positive_tangential = 100.0 * np.mean(tangential_alignments > 0) if tangential_alignments.any() else 0.0
    else:
        angle_progress_deg = 0.0
        mean_cross_track_error_um = 0.0
        pct_positive_tangential = 0.0
    
    # Progress metrics
    initial_pos = np.array([path[0, 0], path[0, 1]])
    final_pos = np.array([particle_x, particle_y])
    final_target = np.array([path[-1, 0], path[-1, 1]])
    
    if is_circle:
        total_arc_length = 2 * np.pi * radius
        net_progress = total_angle * radius
        path_length = total_arc_length
    else:
        path_direction = final_target - initial_pos
        path_length = np.linalg.norm(path_direction)
        if path_length > 0:
            net_progress = np.dot(final_pos - initial_pos, path_direction / path_length)
        else:
            net_progress = 0.0
    
    final_error = np.linalg.norm(final_pos - final_target)
    initial_error = np.linalg.norm(initial_pos - final_target)
    
    pct_positive_alignment = 100.0 * np.mean(alignments > 0)
    pct_positive_push = 100.0 * np.mean(pushes > 0)
    
    mean_solver_time = np.mean([log.solver_time_ms for log in step_logs])
    
    print(f"\n{'='*70}")
    print("4-PUCK GREEDY SURF CONTROLLER RESULTS")
    print(f"{'='*70}")
    
    print(f"\n  NET PROGRESS:")
    print(f"    Initial → final target distance: {initial_error*1e3:.3f} mm → {final_error*1e3:.3f} mm")
    if is_circle:
        print(f"    Angle progress: {angle_progress_deg:.1f}°")
        print(f"    Arc length: {net_progress*1e3:.3f} mm / {path_length*1e3:.3f} mm")
    else:
        print(f"    Net progress: {net_progress*1e3:.3f} mm / {path_length*1e3:.3f} mm")
    progress_frac = net_progress / path_length if path_length > 0 else 0.0
    print(f"    Progress fraction: {progress_frac*100:.1f}%")
    
    if is_circle:
        print(f"\n  CIRCLE METRICS:")
        print(f"    Mean cross-track error: {mean_cross_track_error_um:.1f} µm")
    
    print(f"\n  TRACKING:")
    print(f"    Mean tracking error: {errors.mean()*1e3:.3f} mm ({errors.mean()*1e6:.1f} µm)")
    print(f"    Max tracking error: {errors.max()*1e3:.3f} mm")
    print(f"    Final tracking error: {errors[-1]*1e3:.3f} mm")
    
    print(f"\n  DIRECTIONAL ALIGNMENT:")
    print(f"    Mean F̂·d̂ (unit alignment): {alignments.mean():.3f}")
    print(f"    % steps with F̂·d̂ > 0: {pct_positive_alignment:.1f}%")
    print(f"    % steps with F·d̂ > 0: {pct_positive_push:.1f}%")
    
    print(f"\n  CONTROL SMOOTHNESS:")
    print(f"    Action switches: {switch_count} ({switch_count/(T-1)*100:.1f}%)")
    
    print(f"\n  GATING USAGE:")
    gate_counts = Counter([log.gates_active for log in step_logs])
    for gates, count in gate_counts.most_common():
        print(f"    {gates}: {count} steps ({100*count/len(step_logs):.1f}%)")
    
    print(f"\n  PERFORMANCE:")
    print(f"    Average solver time: {mean_solver_time:.1f} ms/step")
    print(f"    Actions evaluated per step: {len(controller.action_types)}")
    
    # ===== Save outputs =====
    
    # Save steps.csv
    csv_path = out_dir / "steps.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        columns = [
            "step_idx", "particle_x", "particle_y", "target_x", "target_y",
            "tracking_error", "chosen_action", "action_switched",
            "Fp_x", "Fp_y", "Fp_mag", "Fp_hat_dot_d", "Fp_dot_d", "score",
            "trap_candidate_x", "trap_candidate_y", "trap_stable", "stiff_min",
            "solver_time_ms", "n_actions_evaluated",
            "target_idx", "dist_to_target", "target_advanced",
            "gates_active", "cross_track_error"
        ]
        writer.writerow(columns)
        
        for log in step_logs:
            row = [
                log.step_idx, log.particle_x, log.particle_y,
                log.target_x, log.target_y, log.tracking_error,
                log.chosen_action, log.action_switched,
                log.Fp_x, log.Fp_y, log.Fp_mag,
                log.Fp_hat_dot_d, log.Fp_dot_d, log.score,
                log.trap_candidate_x, log.trap_candidate_y,
                log.trap_stable, log.stiff_min,
                log.solver_time_ms, log.n_actions_evaluated,
                log.target_idx, log.dist_to_target, log.target_advanced,
                log.gates_active, log.cross_track_error
            ]
            writer.writerow(row)
    print(f"\nSaved: {csv_path}")
    
    # Save summary JSON
    summary = {
        "demo": "4puck_demo_surf_greedy",
        "path_type": path_name,
        "is_circle": is_circle,
        "n_steps": T,
        "n_actions": len(controller.action_types),
        "action_subset": args.action_subset,
        "transducers": 4,
        "weights": {
            "w_align": surf_config.w_align,
            "w_push": surf_config.w_push,
            "w_switch": surf_config.w_switch,
        },
        "results": {
            "initial_error_mm": float(initial_error * 1e3),
            "final_error_mm": float(final_error * 1e3),
            "net_progress_mm": float(net_progress * 1e3),
            "path_length_mm": float(path_length * 1e3),
            "progress_fraction": float(progress_frac),
            "mean_tracking_error_mm": float(errors.mean() * 1e3),
            "max_tracking_error_mm": float(errors.max() * 1e3),
            "mean_alignment": float(alignments.mean()),
            "pct_positive_alignment": float(pct_positive_alignment),
            "pct_positive_push": float(pct_positive_push),
            "action_switch_count": switch_count,
            "mean_solver_time_ms": float(mean_solver_time),
            "target_advance_count": target_advance_count,
            "final_target_idx": target_idx,
        },
    }
    
    if is_circle:
        summary["circle_metrics"] = {
            "angle_progress_deg": float(angle_progress_deg),
            "mean_cross_track_error_um": float(mean_cross_track_error_um),
            "k_radial": args.k_radial,
            "radius_mm": float(radius * 1e3),
            "center_x_mm": float(center_x * 1e3),
            "center_y_mm": float(center_y * 1e3),
        }
    
    # Gate usage stats
    summary["gate_usage"] = {gates: count for gates, count in gate_counts.items()}
    
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {out_dir / 'summary.json'}")
    
    # ===== Create summary plot =====
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Trajectory
    ax = axes[0, 0]
    ax.plot(path[:, 0] * 1e3, path[:, 1] * 1e3, 'k--', lw=2, label='desired')
    traj = np.array(traj_xy_mm)
    ax.plot(traj[:, 0], traj[:, 1], 'b-', lw=2, label='actual')
    ax.scatter(traj[0, 0], traj[0, 1], s=100, c='green', marker='o', label='start', zorder=10)
    ax.scatter(traj[-1, 0], traj[-1, 1], s=100, c='red', marker='s', label='end', zorder=10)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_title(f'4-Puck Greedy Surf: {path_name}')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Tracking error
    ax = axes[0, 1]
    ax.plot(errors * 1e3, 'b-', lw=1.5)
    ax.axhline(errors.mean() * 1e3, color='r', linestyle='--',
               label=f'mean={errors.mean()*1e3:.3f} mm')
    ax.set_xlabel('Step')
    ax.set_ylabel('Error (mm)')
    ax.set_title('Tracking Error')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Alignment
    ax = axes[1, 0]
    ax.plot(alignments, 'g-', lw=1, alpha=0.7)
    ax.axhline(0, color='k', linestyle='-', lw=0.5)
    ax.axhline(alignments.mean(), color='r', linestyle='--',
               label=f'mean={alignments.mean():.3f}')
    ax.fill_between(range(len(alignments)), 0, alignments,
                   where=alignments > 0, alpha=0.3, color='green')
    ax.fill_between(range(len(alignments)), 0, alignments,
                   where=alignments < 0, alpha=0.3, color='red')
    ax.set_xlabel('Step')
    ax.set_ylabel('F̂·d̂')
    ax.set_title(f'Force Alignment ({pct_positive_alignment:.1f}% positive)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.1, 1.1)
    
    # Action distribution
    ax = axes[1, 1]
    action_counts = Counter(action_history)
    actions = list(action_counts.keys())
    counts = [action_counts[a] for a in actions]
    y_pos = np.arange(len(actions))
    ax.barh(y_pos, counts, color='steelblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([a[:25] for a in actions], fontsize=8)
    ax.set_xlabel('Count')
    ax.set_title('Action Distribution')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(out_dir / "summary.png", dpi=150)
    plt.close()
    print(f"Saved: {out_dir / 'summary.png'}")
    
    # ===== Create GIF =====
    if frame_paths:
        print(f"\nCreating GIF from {len(frame_paths)} frames...")
        images = [imageio.imread(str(p)) for p in frame_paths]
        gif_path = out_dir / "4puck_demo_surf_greedy.gif"
        imageio.mimsave(str(gif_path), images, fps=10, loop=0)
        print(f"Saved: {gif_path}")
    
    print(f"\n{'='*70}")
    print("4-PUCK GREEDY SURF DEMO COMPLETE")
    print(f"{'='*70}")
    print(f"\nOutput directory: {out_dir}")
    print(f"\nKey files:")
    print(f"  GIF:     {out_dir / '4puck_demo_surf_greedy.gif'}")
    print(f"  Summary: {out_dir / 'summary.json'}")
    print(f"  Plot:    {out_dir / 'summary.png'}")
    print(f"  Steps:   {out_dir / 'steps.csv'}")


if __name__ == "__main__":
    main()
