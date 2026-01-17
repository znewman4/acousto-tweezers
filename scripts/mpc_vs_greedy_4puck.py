#!/usr/bin/env python3
"""
mpc_vs_greedy_4puck.py - Fair comparison of Greedy vs MPC using the full 4-puck macro action framework.

Both controllers use the SAME:
- Evaluator4Pucks physics stack
- Macro action set (toggle, widen, narrow, move, translate, etc.)
- Scoring function
- Circle path following task

The key difference:
- **Greedy**: 1-step lookahead, picks the best action NOW
- **MPC**: K-step lookahead, optimizes action sequences for future cost

Outputs (in results/mpc_vs_greedy_4puck/run_YYYYMMDD_HHMMSS/):
    - greedy/steps.csv, greedy/summary.json, greedy/gorkov.gif
    - mpc/steps.csv, mpc/summary.json, mpc/gorkov.gif  
    - comparison/compare_summary.json, compare_plot.png, side_by_side.gif

Usage:
    python scripts/mpc_vs_greedy_4puck.py --fast          # Quick test (100 steps)
    python scripts/mpc_vs_greedy_4puck.py --K 5 --T 300   # Full circle
    python scripts/mpc_vs_greedy_4puck.py --mpc_only      # Run only MPC
"""

from __future__ import annotations

import matplotlib
matplotlib.use('Agg')

import sys
import json
import csv
import argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Tuple, List, Dict, Any, Optional
from enum import Enum, auto
import time
import copy

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "scripts"))

# Import the same infrastructure as the working demo
from tweezers.control import (
    DishDomain, MediumProps, EvaluatorConfig,
    Control4Pucks, default_4puck_config,
)
from tweezers.control.evaluator_4pucks import Evaluator4Pucks
from acousto.force import ParticleProps, gorkov_potential_and_force_2d, bilinear_sample_vec

# Import the macro action framework from the demo
from macro_actions_4puck import (
    MacroActionType4Puck,
    MacroAction4Puck,
    apply_macro_action_4puck,
    get_standard_actions_4puck,
)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    """Configuration for MPC vs Greedy comparison."""
    # Domain (matches 4puck_demo)
    Lx: float = 2.0e-3
    Ly: float = 2.0e-3
    Nx: int = 80
    Ny: int = 80
    
    # Physics (matches 4puck_demo)
    f: float = 2.0e6
    c0: float = 1500.0
    rho0: float = 1000.0
    loss_eta: float = 1e-3
    kz: float = 0.0
    coupling_alpha: float = 1.0
    
    # Transducer params (matches 4puck_demo)
    sigma_x: float = 0.10e-3
    sigma_y: float = 0.15e-3
    bottom_band: float = 0.25e-3
    
    # Particle (matches 4puck_demo)
    particle_a: float = 5.0e-6
    particle_rho_p: float = 1050.0
    particle_c_p: float = 2350.0
    
    # Dynamics (matches 4puck_demo)
    dt: float = 5e-3
    viscosity: float = 1e-3
    alpha_g: float = 2e3
    max_step: float = 0.08e-3
    
    # Macro action parameters (matches 4puck_demo)
    macro_magnitude: float = 0.05e-3
    macro_phase_step: float = 0.15
    macro_amplitude_step: float = 0.01
    
    # Scoring weights (matches 4puck_demo)
    w_align: float = 1.0
    w_push: float = 1e6       # Demo uses 1e6, not 1e8!
    w_switch: float = 0.05
    min_force_threshold: float = 1e-12
    
    # MPC parameters
    K: int = 3                  # horizon length (action sequence length)
    T: int = 300                # total steps
    n_top_actions: int = 5      # beam width for MPC search
    mpc_discount: float = 0.95  # discount factor for future costs
    
    # Path geometry (circle) - match demo
    cx: float = 1.0e-3          # domain.Lx / 2
    cy: float = 1.1e-3          # domain.Ly * 0.55
    R: float = 0.4e-3
    ccw: bool = True
    n_waypoints: int = 400      # Dense waypoints (same as demo steps)
    waypoint_tol: float = 0.12e-3  # 120 µm advance threshold (same as demo)
    k_radial: float = 2.0       # radial correction gain for direction
    
    # Initial particle position - match demo (start at theta=0, i.e., right side)
    theta0: float = 0.0


def create_evaluator(cfg: Config) -> Evaluator4Pucks:
    """Create Evaluator4Pucks with same config as working demo."""
    domain = DishDomain(Lx=cfg.Lx, Ly=cfg.Ly, Nx=cfg.Nx, Ny=cfg.Ny)
    medium = MediumProps(
        f=cfg.f, c0=cfg.c0, rho0=cfg.rho0,
        loss_eta=cfg.loss_eta, kz=cfg.kz, coupling_alpha=cfg.coupling_alpha
    )
    particle = ParticleProps(a=cfg.particle_a, rho_p=cfg.particle_rho_p, c_p=cfg.particle_c_p)
    
    ev_cfg = EvaluatorConfig(
        sigma_x=cfg.sigma_x,
        sigma_y=cfg.sigma_y,
        bottom_band=cfg.bottom_band,
        dt=cfg.dt,
        viscosity=cfg.viscosity,
        alpha_g=cfg.alpha_g,
        max_step=cfg.max_step,
        use_2d_forcing=True,
    )
    
    ev = Evaluator4Pucks(domain, medium, particle, ev_cfg)
    return ev


def get_action_set() -> List[MacroActionType4Puck]:
    """Get the set of macro actions to consider (same as demo)."""
    # Use the standard action set from the demo
    return get_standard_actions_4puck()


# =============================================================================
# Path Following
# =============================================================================

def generate_circle_waypoints(cfg: Config) -> List[Tuple[float, float]]:
    """Generate waypoints around the circle."""
    waypoints = []
    for i in range(cfg.n_waypoints):
        if cfg.ccw:
            theta = cfg.theta0 + 2 * np.pi * i / cfg.n_waypoints
        else:
            theta = cfg.theta0 - 2 * np.pi * i / cfg.n_waypoints
        x = cfg.cx + cfg.R * np.cos(theta)
        y = cfg.cy + cfg.R * np.sin(theta)
        waypoints.append((x, y))
    return waypoints


def get_desired_direction(
    particle_x: float, particle_y: float,
    target_x: float, target_y: float,
    cfg: Config,
) -> Tuple[float, float]:
    """
    Get desired direction for path following using tangent + radial correction.
    
    This matches the demo's compute_circle_direction_tangent_plus_radial.
    d = t̂ - k_radial * (r - R) * r̂ / R
    """
    # Vector from center to particle
    dx = particle_x - cfg.cx
    dy = particle_y - cfg.cy
    r = np.sqrt(dx**2 + dy**2)
    eps = 1e-12
    
    # Unit radial vector (from center outward)
    r_hat_x = dx / (r + eps)
    r_hat_y = dy / (r + eps)
    
    # Tangent vector (CCW)
    t_hat_x = -r_hat_y
    t_hat_y = r_hat_x
    
    # Cross-track error (positive = outside circle)
    cross_track = r - cfg.R
    
    # Combined direction: tangent + radial correction
    # d = t̂ - k_radial * (r - R) * r̂ / R
    d_x = t_hat_x - cfg.k_radial * cross_track * r_hat_x / cfg.R
    d_y = t_hat_y - cfg.k_radial * cross_track * r_hat_y / cfg.R
    
    d_mag = np.sqrt(d_x**2 + d_y**2) + eps
    d_hat_x = d_x / d_mag
    d_hat_y = d_y / d_mag
    
    return d_hat_x, d_hat_y


def path_metrics(x: float, y: float, cfg: Config) -> Dict[str, float]:
    """Compute path tracking metrics for a position."""
    # Vector from center to particle
    dx = x - cfg.cx
    dy = y - cfg.cy
    r = np.sqrt(dx**2 + dy**2) + 1e-12
    
    # Radial unit vector (outward)
    r_hat_x = dx / r
    r_hat_y = dy / r
    
    # Tangent vector (CCW or CW)
    if cfg.ccw:
        t_hat_x = -r_hat_y
        t_hat_y = r_hat_x
    else:
        t_hat_x = r_hat_y
        t_hat_y = -r_hat_x
    
    # Lateral error (signed distance from circle)
    e_perp = r - cfg.R
    
    # Arc length position
    theta = np.arctan2(dy, dx)
    s = cfg.R * ((theta - cfg.theta0) % (2 * np.pi))
    
    return {
        'e_perp': e_perp,
        't_hat': (t_hat_x, t_hat_y),
        'n_hat': (r_hat_x, r_hat_y),
        's': s,
        'theta': theta,
    }


# =============================================================================
# Physics Helpers
# =============================================================================

def make_macro_action(action_type: MacroActionType4Puck, cfg: Config) -> MacroAction4Puck:
    """Create a macro action with config parameters."""
    return MacroAction4Puck(
        action_type=action_type,
        magnitude=cfg.macro_magnitude,
        phase_step=cfg.macro_phase_step,
        amplitude_step=cfg.macro_amplitude_step,
    )


def evaluate_action(
    ev: Evaluator4Pucks,
    ctrl: Control4Pucks,
    action_type: MacroActionType4Puck,
    particle_x: float, particle_y: float,
    cfg: Config,
) -> Tuple[Control4Pucks, float, float, np.ndarray]:
    """
    Evaluate a macro action: apply it, solve PDE, get force at particle.
    
    Returns: (new_ctrl, Fx, Fy, U_field)
    """
    action = make_macro_action(action_type, cfg)
    new_ctrl = apply_macro_action_4puck(ctrl, action)
    new_ctrl = ev.clip_control(new_ctrl)
    
    # Solve PDE
    vb = ev.control_to_forcing_band_vb(new_ctrl)
    field = ev.op.solve_for_bottom_vb(vb)
    U, Fx, Fy = gorkov_potential_and_force_2d(field, ev.particle)
    
    # Apply alpha_g scaling
    Fx_scaled = Fx * ev.cfg.alpha_g
    Fy_scaled = Fy * ev.cfg.alpha_g
    
    # Sample force at particle
    fx, fy = bilinear_sample_vec(field.x, field.y, Fx_scaled, Fy_scaled, particle_x, particle_y)
    
    return new_ctrl, float(fx), float(fy), U


def integrate_particle(
    x: float, y: float,
    Fx: float, Fy: float,
    cfg: Config,
) -> Tuple[float, float]:
    """Overdamped particle integration with step limiting."""
    gamma = 6.0 * np.pi * cfg.viscosity * cfg.particle_a
    
    dx = cfg.dt * Fx / gamma
    dy = cfg.dt * Fy / gamma
    step = np.sqrt(dx**2 + dy**2)
    
    if step > cfg.max_step and step > 0:
        scale = cfg.max_step / step
        dx *= scale
        dy *= scale
    
    x_new = np.clip(x + dx, 0, cfg.Lx)
    y_new = np.clip(y + dy, 0, cfg.Ly)
    return float(x_new), float(y_new)


def compute_action_score(
    Fx: float, Fy: float,
    d_hat_x: float, d_hat_y: float,
    action_type: MacroActionType4Puck,
    prev_action: Optional[MacroActionType4Puck],
    cfg: Config,
) -> Tuple[float, float, float, float]:
    """
    Compute score for an action (same as demo).
    
    Returns: (score, Fp_hat_dot_d, Fp_dot_d, Fp_mag)
    """
    Fp_mag = np.sqrt(Fx**2 + Fy**2)
    eps = 1e-15
    
    Fp_hat_x = Fx / (Fp_mag + eps)
    Fp_hat_y = Fy / (Fp_mag + eps)
    
    Fp_hat_dot_d = Fp_hat_x * d_hat_x + Fp_hat_y * d_hat_y
    Fp_dot_d = Fx * d_hat_x + Fy * d_hat_y
    
    score = cfg.w_align * Fp_hat_dot_d + cfg.w_push * Fp_dot_d
    
    if Fp_mag < cfg.min_force_threshold:
        score -= 0.5
    
    if prev_action is not None and action_type != prev_action:
        score -= cfg.w_switch
    
    return score, Fp_hat_dot_d, Fp_dot_d, Fp_mag


# =============================================================================
# Step Log
# =============================================================================

@dataclass
class StepLog:
    """Log for one step (compatible with demo format)."""
    step_idx: int
    particle_x: float
    particle_y: float
    target_x: float
    target_y: float
    tracking_error: float
    chosen_action: str
    action_switched: bool
    Fp_x: float
    Fp_y: float
    Fp_mag: float
    Fp_hat_dot_d: float
    Fp_dot_d: float
    score: float
    solver_time_ms: float
    n_actions_evaluated: int
    target_idx: int
    dist_to_target: float
    target_advanced: bool
    gates_active: str
    cross_track_error: float


# =============================================================================
# GREEDY CONTROLLER
# =============================================================================

def run_greedy(
    ev: Evaluator4Pucks,
    x0: float, y0: float,
    ctrl0: Control4Pucks,
    waypoints: List[Tuple[float, float]],
    cfg: Config,
    verbose: bool = True,
) -> Tuple[List[StepLog], List[np.ndarray], List[Control4Pucks]]:
    """
    Run greedy controller for T steps.
    
    Returns: (step_logs, U_fields, controls)
    """
    action_set = get_action_set()
    
    step_logs = []
    U_fields = []
    controls = [ctrl0]
    
    x, y = x0, y0
    ctrl = ctrl0
    target_idx = 0
    prev_action = None
    
    if verbose:
        print(f"\n   Running Greedy: T={cfg.T} steps")
        print(f"   Actions: {len(action_set)}")
        print(f"   Progress: ", end="", flush=True)
    
    start_time = time.time()
    
    for t in range(cfg.T):
        target_x, target_y = waypoints[target_idx]
        d_hat_x, d_hat_y = get_desired_direction(x, y, target_x, target_y, cfg)
        
        # Evaluate all actions
        best_score = -np.inf
        best_action_type = MacroActionType4Puck.HOLD
        best_ctrl = ctrl
        best_Fx, best_Fy = 0.0, 0.0
        best_U = None
        best_metrics = (0.0, 0.0, 0.0)
        
        t0_solve = time.time()
        n_evaluated = 0
        
        for action_type in action_set:
            new_ctrl, Fx, Fy, U = evaluate_action(ev, ctrl, action_type, x, y, cfg)
            n_evaluated += 1
            
            score, Fp_hat_dot_d, Fp_dot_d, Fp_mag = compute_action_score(
                Fx, Fy, d_hat_x, d_hat_y, action_type, prev_action, cfg
            )
            
            if score > best_score:
                best_score = score
                best_action_type = action_type
                best_ctrl = new_ctrl
                best_Fx, best_Fy = Fx, Fy
                best_U = U
                best_metrics = (Fp_hat_dot_d, Fp_dot_d, Fp_mag)
        
        solver_time_ms = (time.time() - t0_solve) * 1000
        
        # Apply best action
        action_switched = prev_action is not None and best_action_type != prev_action
        prev_action = best_action_type
        ctrl = best_ctrl
        
        # Integrate particle
        x_new, y_new = integrate_particle(x, y, best_Fx, best_Fy, cfg)
        
        # Check waypoint advancement
        dist_to_target = np.sqrt((x_new - target_x)**2 + (y_new - target_y)**2)
        target_advanced = False
        if dist_to_target < cfg.waypoint_tol and target_idx < len(waypoints) - 1:
            target_idx += 1
            target_advanced = True
        
        # Path metrics
        pm = path_metrics(x_new, y_new, cfg)
        
        # Gates string
        gates = ""
        if ctrl.gateA: gates += "A"
        if ctrl.gateB: gates += "B"
        if ctrl.gateC: gates += "C"
        if ctrl.gateD: gates += "D"
        
        # Log
        log = StepLog(
            step_idx=t,
            particle_x=x_new,
            particle_y=y_new,
            target_x=target_x,
            target_y=target_y,
            tracking_error=np.sqrt((x_new - target_x)**2 + (y_new - target_y)**2),
            chosen_action=best_action_type.name,
            action_switched=action_switched,
            Fp_x=best_Fx,
            Fp_y=best_Fy,
            Fp_mag=best_metrics[2],
            Fp_hat_dot_d=best_metrics[0],
            Fp_dot_d=best_metrics[1],
            score=best_score,
            solver_time_ms=solver_time_ms,
            n_actions_evaluated=n_evaluated,
            target_idx=target_idx,
            dist_to_target=dist_to_target,
            target_advanced=target_advanced,
            gates_active=gates,
            cross_track_error=pm['e_perp'],
        )
        
        step_logs.append(log)
        U_fields.append(best_U)
        controls.append(ctrl)
        
        x, y = x_new, y_new
        
        if verbose and (t + 1) % 50 == 0:
            print(f"{t+1}", end=" ", flush=True)
    
    elapsed = time.time() - start_time
    
    if verbose:
        print(f"\n   Completed in {elapsed:.1f}s ({elapsed/cfg.T*1000:.1f} ms/step)")
        print(f"   Final waypoint: {target_idx}/{len(waypoints)}")
    
    return step_logs, U_fields, controls


# =============================================================================
# MPC CONTROLLER
# =============================================================================

def mpc_rollout(
    ev: Evaluator4Pucks,
    ctrl: Control4Pucks,
    x: float, y: float,
    action_sequence: List[MacroActionType4Puck],
    target_x: float, target_y: float,
    prev_action: Optional[MacroActionType4Puck],
    cfg: Config,
) -> Tuple[float, List[Tuple[float, float]], List[Control4Pucks]]:
    """
    Roll out an action sequence and compute cumulative score.
    
    Returns: (total_discounted_score, positions, controls)
    """
    total_score = 0.0
    positions = [(x, y)]
    ctrls = [ctrl]
    
    current_x, current_y = x, y
    current_ctrl = ctrl
    current_prev_action = prev_action
    
    for k, action_type in enumerate(action_sequence):
        # Get desired direction (towards target for simplicity)
        d_hat_x, d_hat_y = get_desired_direction(current_x, current_y, target_x, target_y, cfg)
        
        # Evaluate action
        new_ctrl, Fx, Fy, U = evaluate_action(ev, current_ctrl, action_type, current_x, current_y, cfg)
        
        # Score
        score, _, _, _ = compute_action_score(
            Fx, Fy, d_hat_x, d_hat_y, action_type, current_prev_action, cfg
        )
        
        # Discounted sum
        total_score += (cfg.mpc_discount ** k) * score
        
        # Integrate
        new_x, new_y = integrate_particle(current_x, current_y, Fx, Fy, cfg)
        
        positions.append((new_x, new_y))
        ctrls.append(new_ctrl)
        
        current_x, current_y = new_x, new_y
        current_ctrl = new_ctrl
        current_prev_action = action_type
    
    return total_score, positions, ctrls


def mpc_beam_search(
    ev: Evaluator4Pucks,
    ctrl: Control4Pucks,
    x: float, y: float,
    target_x: float, target_y: float,
    prev_action: Optional[MacroActionType4Puck],
    cfg: Config,
    action_set: List[MacroActionType4Puck],
) -> Tuple[MacroActionType4Puck, float, int]:
    """
    MPC with beam search over K-step action sequences.
    
    Returns: (best_first_action, best_score, n_sequences_evaluated)
    """
    K = cfg.K
    beam_width = cfg.n_top_actions
    n_evaluated = 0
    
    # Initialize beam with single-action sequences
    beam = []  # List of (score, action_sequence)
    
    for action_type in action_set:
        total_score, positions, ctrls = mpc_rollout(
            ev, ctrl, x, y, [action_type], target_x, target_y, prev_action, cfg
        )
        n_evaluated += 1
        beam.append((total_score, [action_type], positions[-1], ctrls[-1]))
    
    # Sort and keep top beam_width
    beam.sort(key=lambda x: -x[0])
    beam = beam[:beam_width]
    
    # Expand beam for remaining horizon
    for depth in range(1, K):
        new_beam = []
        
        for parent_score, parent_seq, (px, py), parent_ctrl in beam:
            # Get the prev_action for this branch
            branch_prev_action = parent_seq[-1] if parent_seq else prev_action
            
            for action_type in action_set:
                full_seq = parent_seq + [action_type]
                total_score, positions, ctrls = mpc_rollout(
                    ev, ctrl, x, y, full_seq, target_x, target_y, prev_action, cfg
                )
                n_evaluated += len(action_set)  # Approximate count
                new_beam.append((total_score, full_seq, positions[-1], ctrls[-1]))
        
        # Sort and prune
        new_beam.sort(key=lambda x: -x[0])
        beam = new_beam[:beam_width]
    
    # Return best first action
    best_score, best_seq, _, _ = beam[0]
    return best_seq[0], best_score, n_evaluated


def run_mpc(
    ev: Evaluator4Pucks,
    x0: float, y0: float,
    ctrl0: Control4Pucks,
    waypoints: List[Tuple[float, float]],
    cfg: Config,
    verbose: bool = True,
) -> Tuple[List[StepLog], List[np.ndarray], List[Control4Pucks]]:
    """
    Run MPC controller for T steps.
    
    Returns: (step_logs, U_fields, controls)
    """
    action_set = get_action_set()
    
    step_logs = []
    U_fields = []
    controls = [ctrl0]
    
    x, y = x0, y0
    ctrl = ctrl0
    target_idx = 0
    prev_action = None
    
    if verbose:
        print(f"\n   Running MPC: T={cfg.T} steps, K={cfg.K} horizon")
        print(f"   Actions: {len(action_set)}, Beam width: {cfg.n_top_actions}")
        print(f"   Progress: ", end="", flush=True)
    
    start_time = time.time()
    
    for t in range(cfg.T):
        target_x, target_y = waypoints[target_idx]
        d_hat_x, d_hat_y = get_desired_direction(x, y, target_x, target_y, cfg)
        
        t0_solve = time.time()
        
        # MPC beam search
        best_action_type, best_mpc_score, n_evaluated = mpc_beam_search(
            ev, ctrl, x, y, target_x, target_y, prev_action, cfg, action_set
        )
        
        # Execute best action
        best_ctrl, best_Fx, best_Fy, best_U = evaluate_action(
            ev, ctrl, best_action_type, x, y, cfg
        )
        
        # Compute metrics for logging
        score, Fp_hat_dot_d, Fp_dot_d, Fp_mag = compute_action_score(
            best_Fx, best_Fy, d_hat_x, d_hat_y, best_action_type, prev_action, cfg
        )
        
        solver_time_ms = (time.time() - t0_solve) * 1000
        
        # Apply action
        action_switched = prev_action is not None and best_action_type != prev_action
        prev_action = best_action_type
        ctrl = best_ctrl
        
        # Integrate particle
        x_new, y_new = integrate_particle(x, y, best_Fx, best_Fy, cfg)
        
        # Check waypoint advancement
        dist_to_target = np.sqrt((x_new - target_x)**2 + (y_new - target_y)**2)
        target_advanced = False
        if dist_to_target < cfg.waypoint_tol and target_idx < len(waypoints) - 1:
            target_idx += 1
            target_advanced = True
        
        # Path metrics
        pm = path_metrics(x_new, y_new, cfg)
        
        # Gates string
        gates = ""
        if ctrl.gateA: gates += "A"
        if ctrl.gateB: gates += "B"
        if ctrl.gateC: gates += "C"
        if ctrl.gateD: gates += "D"
        
        # Log
        log = StepLog(
            step_idx=t,
            particle_x=x_new,
            particle_y=y_new,
            target_x=target_x,
            target_y=target_y,
            tracking_error=np.sqrt((x_new - target_x)**2 + (y_new - target_y)**2),
            chosen_action=best_action_type.name,
            action_switched=action_switched,
            Fp_x=best_Fx,
            Fp_y=best_Fy,
            Fp_mag=Fp_mag,
            Fp_hat_dot_d=Fp_hat_dot_d,
            Fp_dot_d=Fp_dot_d,
            score=score,
            solver_time_ms=solver_time_ms,
            n_actions_evaluated=n_evaluated,
            target_idx=target_idx,
            dist_to_target=dist_to_target,
            target_advanced=target_advanced,
            gates_active=gates,
            cross_track_error=pm['e_perp'],
        )
        
        step_logs.append(log)
        U_fields.append(best_U)
        controls.append(ctrl)
        
        x, y = x_new, y_new
        
        if verbose and (t + 1) % 50 == 0:
            print(f"{t+1}", end=" ", flush=True)
    
    elapsed = time.time() - start_time
    
    if verbose:
        print(f"\n   Completed in {elapsed:.1f}s ({elapsed/cfg.T*1000:.1f} ms/step)")
        print(f"   Final waypoint: {target_idx}/{len(waypoints)}")
    
    return step_logs, U_fields, controls


# =============================================================================
# Output Saving
# =============================================================================

def save_steps_csv(out_path: Path, logs: List[StepLog]):
    """Save step logs to CSV (same format as demo)."""
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header
        writer.writerow([
            'step_idx', 'particle_x', 'particle_y', 'target_x', 'target_y',
            'tracking_error', 'chosen_action', 'action_switched',
            'Fp_x', 'Fp_y', 'Fp_mag', 'Fp_hat_dot_d', 'Fp_dot_d', 'score',
            'solver_time_ms', 'n_actions_evaluated', 'target_idx',
            'dist_to_target', 'target_advanced', 'gates_active', 'cross_track_error'
        ])
        for log in logs:
            writer.writerow([
                log.step_idx, log.particle_x, log.particle_y,
                log.target_x, log.target_y, log.tracking_error,
                log.chosen_action, log.action_switched,
                log.Fp_x, log.Fp_y, log.Fp_mag,
                log.Fp_hat_dot_d, log.Fp_dot_d, log.score,
                log.solver_time_ms, log.n_actions_evaluated, log.target_idx,
                log.dist_to_target, log.target_advanced, log.gates_active,
                log.cross_track_error
            ])
    print(f"   Saved: {out_path}")


def compute_summary(logs: List[StepLog], cfg: Config, method: str) -> Dict[str, Any]:
    """Compute summary statistics."""
    if not logs:
        return {}
    
    # Extract arrays
    x_arr = np.array([log.particle_x for log in logs])
    y_arr = np.array([log.particle_y for log in logs])
    Fp_mag_arr = np.array([log.Fp_mag for log in logs])
    tracking_err_arr = np.array([log.tracking_error for log in logs])
    cross_track_arr = np.array([log.cross_track_error for log in logs])
    solver_time_arr = np.array([log.solver_time_ms for log in logs])
    
    # Total distance traveled
    dx = np.diff(x_arr)
    dy = np.diff(y_arr)
    if len(dx) > 0:
        total_distance = np.sum(np.sqrt(dx**2 + dy**2))
    else:
        total_distance = 0.0
    
    # Arc length progress (approximate)
    final_theta = np.arctan2(y_arr[-1] - cfg.cy, x_arr[-1] - cfg.cx)
    initial_theta = np.arctan2(y_arr[0] - cfg.cy, x_arr[0] - cfg.cx)
    delta_theta = (final_theta - initial_theta) % (2 * np.pi)
    if not cfg.ccw:
        delta_theta = (initial_theta - final_theta) % (2 * np.pi)
    arc_progress = cfg.R * delta_theta
    
    # Waypoint progress
    final_waypoint = logs[-1].target_idx
    
    # Action switches
    n_switches = sum(1 for log in logs if log.action_switched)
    
    return {
        'method': method,
        'n_steps': len(logs),
        'final_waypoint': final_waypoint,
        'n_waypoints': cfg.n_waypoints,
        'waypoint_completion': final_waypoint / cfg.n_waypoints,
        'total_distance_mm': float(total_distance * 1e3),
        'arc_progress_mm': float(arc_progress * 1e3),
        'circle_fraction': float(delta_theta / (2 * np.pi)),
        'mean_tracking_error_um': float(np.mean(tracking_err_arr) * 1e6),
        'max_tracking_error_um': float(np.max(tracking_err_arr) * 1e6),
        'mean_cross_track_error_um': float(np.mean(np.abs(cross_track_arr)) * 1e6),
        'max_cross_track_error_um': float(np.max(np.abs(cross_track_arr)) * 1e6),
        'mean_force_N': float(np.mean(Fp_mag_arr)),
        'max_force_N': float(np.max(Fp_mag_arr)),
        'mean_solver_time_ms': float(np.mean(solver_time_arr)),
        'total_runtime_s': float(np.sum(solver_time_arr) / 1000),
        'n_action_switches': n_switches,
        'switch_rate': float(n_switches / len(logs)),
    }


def save_summary_json(out_path: Path, summary: Dict[str, Any], cfg: Config):
    """Save summary to JSON."""
    data = {
        'config': {
            'T': cfg.T,
            'K': cfg.K,
            'n_top_actions': cfg.n_top_actions,
            'Nx': cfg.Nx,
            'Ny': cfg.Ny,
            'alpha_g': cfg.alpha_g,
            'dt': cfg.dt,
            'n_waypoints': cfg.n_waypoints,
            'circle_R_mm': cfg.R * 1e3,
        },
        'summary': summary,
    }
    with open(out_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"   Saved: {out_path}")


def create_gorkov_gif(
    out_path: Path,
    logs: List[StepLog],
    U_fields: List[np.ndarray],
    controls: List[Control4Pucks],
    ev: Evaluator4Pucks,
    cfg: Config,
    method: str,
    max_frames: int = 100,
    frame_duration: float = 0.1,
):
    """Create Gor'kov contour GIF.
    
    Args:
        out_path: Output path for GIF
        logs: Step logs
        U_fields: Gor'kov potential fields for each step
        controls: Control states for each step (for transducer visualization)
        ev: Evaluator
        cfg: Config
        method: Method name for title
        max_frames: Maximum number of frames in GIF
        frame_duration: Duration per frame in seconds (higher = slower)
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    import imageio.v2 as imageio
    
    # Subsample frames
    n_total = len(logs)
    if n_total > max_frames:
        indices = np.linspace(0, n_total - 1, max_frames, dtype=int)
    else:
        indices = np.arange(n_total)
    
    frames = []
    temp_dir = out_path.parent / f"_temp_{out_path.stem}"
    temp_dir.mkdir(exist_ok=True)
    
    x_mm = ev.op.x * 1e3
    y_mm = ev.op.y * 1e3
    X, Y = np.meshgrid(x_mm, y_mm)
    
    # Compute global U limits for consistent colorbar
    # Include a wider range to ensure contours are always visible
    all_U = np.concatenate([U_fields[i].flatten() for i in indices])
    U_lo_global, U_hi_global = np.nanpercentile(all_U, [2, 98])
    
    # Also compute per-frame local limits around particle to ensure visibility
    # We'll use the max range across all frames to ensure consistency
    local_ranges = []
    for t in indices:
        log = logs[t]
        U = U_fields[t]
        # Get values in a region around the particle (within 0.3mm)
        px_mm, py_mm = log.particle_x * 1e3, log.particle_y * 1e3
        mask = ((X - px_mm)**2 + (Y - py_mm)**2) < 0.3**2
        local_vals = U[mask]
        if len(local_vals) > 0:
            local_ranges.append((np.nanmin(local_vals), np.nanmax(local_vals)))
    
    if local_ranges:
        local_lo = min(r[0] for r in local_ranges)
        local_hi = max(r[1] for r in local_ranges)
        # Expand global range to include local extremes with some padding
        U_lo = min(U_lo_global, local_lo - 0.1 * abs(local_lo))
        U_hi = max(U_hi_global, local_hi + 0.1 * abs(local_hi))
    else:
        U_lo, U_hi = U_lo_global, U_hi_global
    
    # Ensure we have some range (avoid blank contours)
    if U_hi - U_lo < 1e-20:
        U_lo = U_lo - 1e-18
        U_hi = U_hi + 1e-18
    
    for frame_idx, t in enumerate(indices):
        fig, ax = plt.subplots(figsize=(8, 7))
        
        U = U_fields[t]
        log = logs[t]
        
        # Contours
        levels = np.linspace(U_lo, U_hi, 25)
        contourf = ax.contourf(X, Y, U, levels=levels, cmap="RdBu_r", alpha=0.85, extend='both')
        ax.contour(X, Y, U, levels=levels[::2], colors="k", linewidths=0.3, alpha=0.3)
        cbar = fig.colorbar(contourf, ax=ax)
        cbar.set_label("U (J)")
        
        # Path circle
        circle = Circle((cfg.cx * 1e3, cfg.cy * 1e3), cfg.R * 1e3,
                         fill=False, edgecolor='lime', linewidth=2, linestyle='--')
        ax.add_patch(circle)
        
        # Trail
        trail_indices = [i for i in indices if i <= t]
        if len(trail_indices) >= 2:
            tx = [logs[i].particle_x * 1e3 for i in trail_indices]
            ty = [logs[i].particle_y * 1e3 for i in trail_indices]
            for i in range(len(tx) - 1):
                alpha_val = 0.3 + 0.7 * i / max(len(tx) - 1, 1)
                color = (1-alpha_val) * np.array([0.5, 0.5, 0.5]) + alpha_val * np.array([0, 1, 1])
                ax.plot(tx[i:i+2], ty[i:i+2], linewidth=2, color=color, alpha=0.9)
        
        # Transducers (pucks) with gate visualization
        ctrl = controls[t]
        transducer_info = [
            ('A', 'blue', ctrl.xA, ctrl.yA, ctrl.gateA),
            ('B', 'green', ctrl.xB, ctrl.yB, ctrl.gateB),
            ('C', 'orange', ctrl.xC, ctrl.yC, ctrl.gateC),
            ('D', 'purple', ctrl.xD, ctrl.yD, ctrl.gateD),
        ]
        for name, puck_color, px, py, gate in transducer_info:
            marker = 'o' if gate else 'x'
            alpha_val = 1.0 if gate else 0.3
            ax.scatter(px * 1e3, py * 1e3, s=120, c=puck_color, marker=marker,
                       edgecolors='white' if gate else 'gray', linewidths=1.5,
                       alpha=alpha_val, zorder=90)
        
        # Current position (particle)
        ax.scatter(log.particle_x * 1e3, log.particle_y * 1e3, s=250, marker='o',
                   color='red', edgecolors='white', linewidth=2, zorder=100)
        
        # Target
        ax.scatter(log.target_x * 1e3, log.target_y * 1e3, s=100, marker='x',
                   color='yellow', linewidth=2, zorder=99)
        
        # Force arrow
        F_scale = 1e8  # Scale for visibility
        ax.arrow(log.particle_x * 1e3, log.particle_y * 1e3,
                 log.Fp_x * F_scale * 0.1, log.Fp_y * F_scale * 0.1,
                 head_width=0.03, head_length=0.01, fc='magenta', ec='black', zorder=100)
        
        ax.set_xlim(x_mm[0], x_mm[-1])
        ax.set_ylim(y_mm[0], y_mm[-1])
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)
        
        # Add legend for transducers
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='A'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='B'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=8, label='C'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=8, label='D'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
        
        title = f"{method}: Step {t}/{n_total-1}\n"
        title += f"|F|={log.Fp_mag:.2e}N, err={log.tracking_error*1e6:.1f}µm, WP={log.target_idx}, Gates={log.gates_active}"
        ax.set_title(title)
        
        fig.tight_layout()
        
        frame_path = temp_dir / f"frame_{frame_idx:04d}.png"
        fig.savefig(frame_path, dpi=80)
        plt.close(fig)
        
        frames.append(imageio.imread(frame_path))
    
    imageio.mimsave(out_path, frames, duration=frame_duration)
    
    for f in temp_dir.glob("*.png"):
        f.unlink()
    temp_dir.rmdir()
    
    print(f"   Saved: {out_path}")


def create_comparison_plot(
    out_path: Path,
    greedy_logs: List[StepLog],
    mpc_logs: List[StepLog],
    cfg: Config,
):
    """Create comparison plots."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Trajectory plot
    ax1 = axes[0, 0]
    greedy_x = [log.particle_x * 1e3 for log in greedy_logs]
    greedy_y = [log.particle_y * 1e3 for log in greedy_logs]
    mpc_x = [log.particle_x * 1e3 for log in mpc_logs]
    mpc_y = [log.particle_y * 1e3 for log in mpc_logs]
    
    ax1.plot(greedy_x, greedy_y, 'b-', linewidth=1.5, alpha=0.8, label='Greedy')
    ax1.plot(mpc_x, mpc_y, 'r-', linewidth=1.5, alpha=0.8, label='MPC')
    
    # Circle path
    theta = np.linspace(0, 2*np.pi, 100)
    cx_mm = cfg.cx * 1e3
    cy_mm = cfg.cy * 1e3
    R_mm = cfg.R * 1e3
    ax1.plot(cx_mm + R_mm * np.cos(theta), cy_mm + R_mm * np.sin(theta),
             'g--', linewidth=2, alpha=0.5, label='Target path')
    
    ax1.set_xlabel('x (mm)')
    ax1.set_ylabel('y (mm)')
    ax1.set_aspect('equal')
    ax1.legend()
    ax1.set_title('Trajectories')
    ax1.grid(True, alpha=0.3)
    
    # 2. Cross-track error over time
    ax2 = axes[0, 1]
    greedy_cte = [log.cross_track_error * 1e6 for log in greedy_logs]
    mpc_cte = [log.cross_track_error * 1e6 for log in mpc_logs]
    t_arr = np.arange(len(greedy_logs))
    
    ax2.plot(t_arr, greedy_cte, 'b-', linewidth=1, alpha=0.8, label=f'Greedy (mean={np.mean(np.abs(greedy_cte)):.1f}µm)')
    ax2.plot(t_arr[:len(mpc_cte)], mpc_cte, 'r-', linewidth=1, alpha=0.8, label=f'MPC (mean={np.mean(np.abs(mpc_cte)):.1f}µm)')
    ax2.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Cross-track Error (µm)')
    ax2.legend()
    ax2.set_title('Path Tracking Error')
    ax2.grid(True, alpha=0.3)
    
    # 3. Waypoint progress
    ax3 = axes[1, 0]
    greedy_wp = [log.target_idx for log in greedy_logs]
    mpc_wp = [log.target_idx for log in mpc_logs]
    
    ax3.plot(t_arr, greedy_wp, 'b-', linewidth=2, label=f'Greedy (final={greedy_wp[-1]})')
    ax3.plot(t_arr[:len(mpc_wp)], mpc_wp, 'r-', linewidth=2, label=f'MPC (final={mpc_wp[-1]})')
    ax3.axhline(cfg.n_waypoints, color='g', linestyle='--', alpha=0.5, label='Complete')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Waypoint Index')
    ax3.legend()
    ax3.set_title('Waypoint Progress')
    ax3.grid(True, alpha=0.3)
    
    # 4. Force magnitude
    ax4 = axes[1, 1]
    greedy_F = [log.Fp_mag for log in greedy_logs]
    mpc_F = [log.Fp_mag for log in mpc_logs]
    
    ax4.semilogy(t_arr, greedy_F, 'b-', linewidth=1, alpha=0.8, label=f'Greedy (mean={np.mean(greedy_F):.2e}N)')
    ax4.semilogy(t_arr[:len(mpc_F)], mpc_F, 'r-', linewidth=1, alpha=0.8, label=f'MPC (mean={np.mean(mpc_F):.2e}N)')
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Force Magnitude (N)')
    ax4.legend()
    ax4.set_title('Force Magnitude')
    ax4.grid(True, alpha=0.3)
    
    fig.suptitle('Greedy vs MPC: 4-Puck Macro Action Comparison', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    
    print(f"   Saved: {out_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="MPC vs Greedy 4-puck comparison")
    
    # Grid
    parser.add_argument("--Nx", type=int, default=80)
    parser.add_argument("--Ny", type=int, default=80)
    
    # MPC parameters
    parser.add_argument("--K", type=int, default=3, help="MPC horizon length")
    parser.add_argument("--T", type=int, default=300, help="Total steps")
    parser.add_argument("--beam_width", type=int, default=5, help="Beam search width")
    
    # Modes
    parser.add_argument("--fast", action="store_true", help="Quick test mode")
    parser.add_argument("--greedy_only", action="store_true", help="Run only greedy")
    parser.add_argument("--mpc_only", action="store_true", help="Run only MPC")
    
    # Output
    parser.add_argument("--output_dir", type=str, default=None)
    
    args = parser.parse_args()
    
    # Config
    cfg = Config(
        Nx=args.Nx,
        Ny=args.Ny,
        K=args.K,
        T=args.T,
        n_top_actions=args.beam_width,
    )
    
    if args.fast:
        cfg.T = 100
        cfg.K = 2
        cfg.Nx = 64
        cfg.Ny = 64
        cfg.n_top_actions = 3
    
    print("\n" + "="*70)
    print("  MPC vs Greedy: 4-Puck Macro Action Comparison")
    print("="*70)
    
    if args.fast:
        print("   [FAST MODE]")
    
    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = project_root / "results" / "mpc_vs_greedy_4puck" / f"run_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    (out_dir / "greedy").mkdir(exist_ok=True)
    (out_dir / "mpc").mkdir(exist_ok=True)
    (out_dir / "comparison").mkdir(exist_ok=True)
    
    print(f"\n   Output: {out_dir}")
    print(f"   Config: T={cfg.T}, K={cfg.K}, beam_width={cfg.n_top_actions}")
    print(f"   Grid: {cfg.Nx} x {cfg.Ny}")
    
    # Build evaluator
    print("\n   Building Evaluator4Pucks...")
    ev = create_evaluator(cfg)
    
    # Generate waypoints
    waypoints = generate_circle_waypoints(cfg)
    print(f"   Waypoints: {len(waypoints)} around circle")
    
    # Initial state
    x0 = cfg.cx + cfg.R * np.cos(cfg.theta0)
    y0 = cfg.cy + cfg.R * np.sin(cfg.theta0)
    
    # Initial control (same as demo - ALL gates True initially)
    Lx, Ly = cfg.Lx, cfg.Ly
    ctrl0 = Control4Pucks(
        xA=x0 - 0.4e-3, yA=0.03e-3, vA=0.08, phiA=0.0, gateA=True,
        xB=x0 + 0.4e-3, yB=0.03e-3, vB=0.08, phiB=np.pi, gateB=True,
        xC=x0, yC=0.20e-3, vC=0.08, phiC=np.pi/2, gateC=True,
        xD=x0, yD=1.8e-3, vD=0.05, phiD=-np.pi/2, gateD=True,  # D also on initially
    )
    ctrl0 = ev.clip_control(ctrl0)
    
    print(f"\n   Initial position: ({x0*1e3:.3f}, {y0*1e3:.3f}) mm")
    
    greedy_logs, greedy_U, greedy_ctrls = None, None, None
    mpc_logs, mpc_U, mpc_ctrls = None, None, None
    
    # Run Greedy
    if not args.mpc_only:
        print("\n" + "-"*50)
        print("  Running GREEDY Controller")
        print("-"*50)
        greedy_logs, greedy_U, greedy_ctrls = run_greedy(ev, x0, y0, ctrl0, waypoints, cfg)
        
        # Save greedy outputs
        print("\n   Saving Greedy outputs...")
        save_steps_csv(out_dir / "greedy" / "steps.csv", greedy_logs)
        greedy_summary = compute_summary(greedy_logs, cfg, "greedy")
        save_summary_json(out_dir / "greedy" / "summary.json", greedy_summary, cfg)
        
        print("   Creating Greedy GIF...")
        create_gorkov_gif(out_dir / "greedy" / "gorkov.gif", greedy_logs, greedy_U, greedy_ctrls,
                          ev, cfg, "GREEDY", max_frames=100, frame_duration=0.1)
    
    # Run MPC
    if not args.greedy_only:
        print("\n" + "-"*50)
        print("  Running MPC Controller")
        print("-"*50)
        mpc_logs, mpc_U, mpc_ctrls = run_mpc(ev, x0, y0, ctrl0, waypoints, cfg)
        
        # Save MPC outputs
        print("\n   Saving MPC outputs...")
        save_steps_csv(out_dir / "mpc" / "steps.csv", mpc_logs)
        mpc_summary = compute_summary(mpc_logs, cfg, "mpc")
        save_summary_json(out_dir / "mpc" / "summary.json", mpc_summary, cfg)
        
        print("   Creating MPC GIF...")
        # MPC uses 150 frames with slower duration so particle appears to move at similar speed to Greedy
        create_gorkov_gif(out_dir / "mpc" / "gorkov.gif", mpc_logs, mpc_U, mpc_ctrls,
                          ev, cfg, "MPC", max_frames=150, frame_duration=0.15)
    
    # Comparison
    if greedy_logs and mpc_logs:
        print("\n" + "-"*50)
        print("  Creating Comparison")
        print("-"*50)
        
        create_comparison_plot(out_dir / "comparison" / "compare_plot.png", greedy_logs, mpc_logs, cfg)
        
        # Summary comparison
        compare_summary = {
            'greedy': greedy_summary,
            'mpc': mpc_summary,
            'comparison': {
                'mpc_more_waypoints': mpc_summary['final_waypoint'] > greedy_summary['final_waypoint'],
                'mpc_lower_cross_track_error': mpc_summary['mean_cross_track_error_um'] < greedy_summary['mean_cross_track_error_um'],
                'mpc_fewer_switches': mpc_summary['n_action_switches'] < greedy_summary['n_action_switches'],
            }
        }
        
        with open(out_dir / "comparison" / "compare_summary.json", 'w') as f:
            json.dump(compare_summary, f, indent=2)
        print(f"   Saved: {out_dir / 'comparison' / 'compare_summary.json'}")
    
    print("\n" + "="*70)
    print("  COMPLETE")
    print("="*70)
    print(f"\n   All outputs saved to: {out_dir}")
    
    if greedy_logs:
        print(f"\n   GREEDY: {greedy_summary['final_waypoint']}/{cfg.n_waypoints} waypoints, "
              f"{greedy_summary['circle_fraction']*100:.1f}% circle, "
              f"CTE={greedy_summary['mean_cross_track_error_um']:.1f}µm")
    
    if mpc_logs:
        print(f"   MPC:    {mpc_summary['final_waypoint']}/{cfg.n_waypoints} waypoints, "
              f"{mpc_summary['circle_fraction']*100:.1f}% circle, "
              f"CTE={mpc_summary['mean_cross_track_error_um']:.1f}µm")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
