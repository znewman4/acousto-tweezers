#!/usr/bin/env python3
"""
path_tracking_comparison.py - Continuous Path-Tracking vs Waypoint-Chasing Comparison

This script implements and compares two fundamentally different control strategies:

1. **Waypoint-Chasing (Baseline)**:
   - Selects discrete target points on the circle
   - Controller pushes particle directly toward waypoint
   - Causes overshoot, oscillation, and action switching

2. **Continuous Path-Tracking (New)**:
   - Uses geometric path-following control law
   - Desired direction: d_des = v_∥ · t̂ - k_⊥ · e_⊥ · n̂
   - Decouples progress from accuracy
   - Should show smoother motion and fewer action switches

Both use the SAME:
- Evaluator4Pucks physics stack
- Macro action set
- Circle path geometry
- Initial conditions

Outputs (in results/path_tracking_comparison/run_YYYYMMDD_HHMMSS/):
    - waypoint_greedy/steps.csv, summary.json, gorkov.gif
    - waypoint_mpc/steps.csv, summary.json, gorkov.gif
    - pathtrack_greedy/steps.csv, summary.json, gorkov.gif
    - pathtrack_mpc/steps.csv, summary.json, gorkov.gif
    - comparison/compare_summary.json, trajectories.png
    - comparison/greedy_vs_greedy.gif (split-screen)
    - comparison/mpc_vs_mpc.gif (split-screen)
    - comparison/best_comparison.gif (waypoint_greedy vs pathtrack_mpc)

Usage:
    python scripts/path_tracking_comparison.py --fast           # Quick test
    python scripts/path_tracking_comparison.py --T 300 --K 3    # Full run
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
# Control Mode Enum
# =============================================================================

class ControlMode(Enum):
    """Control strategy mode."""
    WAYPOINT = "waypoint"      # Traditional waypoint chasing
    PATH_TRACK = "pathtrack"   # Continuous path-tracking


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    """Configuration for path-tracking comparison."""
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
    w_push: float = 1e6
    w_switch: float = 0.05
    min_force_threshold: float = 1e-12
    
    # MPC parameters
    K: int = 3                  # horizon length
    T: int = 300                # total steps
    n_top_actions: int = 5      # beam width
    mpc_discount: float = 0.95
    
    # Path geometry (circle)
    cx: float = 1.0e-3          # circle center x
    cy: float = 1.1e-3          # circle center y
    R: float = 0.4e-3           # circle radius
    ccw: bool = True            # counter-clockwise
    n_waypoints: int = 400      # for waypoint mode
    waypoint_tol: float = 0.12e-3  # 120 µm
    
    # === WAYPOINT MODE PARAMETERS ===
    # Traditional radial correction
    k_radial_waypoint: float = 2.0
    
    # === PATH-TRACKING MODE PARAMETERS ===
    # Continuous path-following control law:
    #   d_des = v_parallel * t_hat - k_perp * e_perp * n_hat
    v_parallel: float = 1.0     # Forward velocity gain (tangential)
    k_perp: float = 3.0         # Cross-track correction gain (lower = smoother, higher = tighter)
    
    # Initial particle position (start at theta=0, right side of circle)
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
    return get_standard_actions_4puck()


# =============================================================================
# Path Geometry Functions
# =============================================================================

def compute_circle_geometry(
    particle_x: float, particle_y: float,
    cfg: Config,
) -> Dict[str, Any]:
    """
    Compute path geometry for a circle.
    
    Given particle position, computes:
    - Closest point on circle
    - Radial error (e_perp): positive = outside circle
    - Unit normal (n_hat): radially outward from center
    - Unit tangent (t_hat): CCW direction
    - Arc position (theta, s)
    
    Returns dict with:
        'closest_x', 'closest_y': closest point on circle
        'e_perp': signed radial error (positive = outside)
        'n_hat': (n_hat_x, n_hat_y) unit normal (outward)
        't_hat': (t_hat_x, t_hat_y) unit tangent (CCW)
        'theta': angle from center
        's': arc length from theta0
    """
    eps = 1e-12
    
    # Vector from center to particle
    dx = particle_x - cfg.cx
    dy = particle_y - cfg.cy
    r = np.sqrt(dx**2 + dy**2)
    
    # Unit radial vector (from center outward = normal to path)
    n_hat_x = dx / (r + eps)
    n_hat_y = dy / (r + eps)
    
    # Unit tangent vector (CCW if ccw=True)
    if cfg.ccw:
        t_hat_x = -n_hat_y
        t_hat_y = n_hat_x
    else:
        t_hat_x = n_hat_y
        t_hat_y = -n_hat_x
    
    # Closest point on circle
    closest_x = cfg.cx + cfg.R * n_hat_x
    closest_y = cfg.cy + cfg.R * n_hat_y
    
    # Radial error (signed: positive = outside circle)
    e_perp = r - cfg.R
    
    # Angular position
    theta = np.arctan2(dy, dx)
    
    # Arc length from initial position
    delta_theta = (theta - cfg.theta0) if cfg.ccw else (cfg.theta0 - theta)
    delta_theta = delta_theta % (2 * np.pi)
    s = cfg.R * delta_theta
    
    return {
        'closest_x': closest_x,
        'closest_y': closest_y,
        'e_perp': e_perp,
        'n_hat': (n_hat_x, n_hat_y),
        't_hat': (t_hat_x, t_hat_y),
        'theta': theta,
        's': s,
        'r': r,
    }


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


# =============================================================================
# Desired Direction Functions (KEY DIFFERENCE)
# =============================================================================

def get_desired_direction_waypoint(
    particle_x: float, particle_y: float,
    target_x: float, target_y: float,
    cfg: Config,
) -> Tuple[float, float, Dict[str, float]]:
    """
    WAYPOINT MODE: Desired direction using tangent + radial correction to waypoint.
    
    This is the ORIGINAL approach that causes oscillation and action switching.
    The direction is fundamentally towards the waypoint, with some radial correction.
    
    d = t̂ - k_radial * (r - R) * n̂ / R
    
    Returns: (d_hat_x, d_hat_y, debug_info)
    """
    eps = 1e-12
    geom = compute_circle_geometry(particle_x, particle_y, cfg)
    
    t_hat_x, t_hat_y = geom['t_hat']
    n_hat_x, n_hat_y = geom['n_hat']
    e_perp = geom['e_perp']
    
    # Combined direction: tangent + radial correction
    # This still has the waypoint-chasing flavor through k_radial
    d_x = t_hat_x - cfg.k_radial_waypoint * e_perp * n_hat_x / cfg.R
    d_y = t_hat_y - cfg.k_radial_waypoint * e_perp * n_hat_y / cfg.R
    
    d_mag = np.sqrt(d_x**2 + d_y**2) + eps
    d_hat_x = d_x / d_mag
    d_hat_y = d_y / d_mag
    
    debug = {
        'e_perp': e_perp,
        't_hat_x': t_hat_x, 't_hat_y': t_hat_y,
        'n_hat_x': n_hat_x, 'n_hat_y': n_hat_y,
    }
    
    return d_hat_x, d_hat_y, debug


def get_desired_direction_pathtrack(
    particle_x: float, particle_y: float,
    cfg: Config,
) -> Tuple[float, float, Dict[str, float]]:
    """
    PATH-TRACKING MODE: Continuous geometric path-following control law.
    
    This is the NEW approach that should reduce oscillation and action switching.
    
    The desired direction is:
        d_des = v_∥ · t̂ - k_⊥ · e_⊥ · n̂
    
    Where:
    - v_∥ > 0 encourages forward motion along the circle (tangential)
    - k_⊥ > 0 pulls the particle back onto the path (radial correction)
    - e_⊥ is the signed cross-track error (positive = outside circle)
    - t̂ is the unit tangent (CCW)
    - n̂ is the unit normal (radially outward)
    
    This is a standard path-following control law (not waypoint pursuit).
    
    Returns: (d_hat_x, d_hat_y, debug_info)
    """
    eps = 1e-12
    geom = compute_circle_geometry(particle_x, particle_y, cfg)
    
    t_hat_x, t_hat_y = geom['t_hat']
    n_hat_x, n_hat_y = geom['n_hat']
    e_perp = geom['e_perp']
    
    # Continuous path-following control law:
    # d_des = v_parallel * t_hat - k_perp * e_perp * n_hat
    #
    # The negative sign on k_perp * e_perp * n_hat means:
    # - If particle is OUTSIDE circle (e_perp > 0), push INWARD (-n_hat direction)
    # - If particle is INSIDE circle (e_perp < 0), push OUTWARD (+n_hat direction)
    
    d_x = cfg.v_parallel * t_hat_x - cfg.k_perp * e_perp * n_hat_x
    d_y = cfg.v_parallel * t_hat_y - cfg.k_perp * e_perp * n_hat_y
    
    d_mag = np.sqrt(d_x**2 + d_y**2) + eps
    d_hat_x = d_x / d_mag
    d_hat_y = d_y / d_mag
    
    debug = {
        'e_perp': e_perp,
        't_hat_x': t_hat_x, 't_hat_y': t_hat_y,
        'n_hat_x': n_hat_x, 'n_hat_y': n_hat_y,
        'd_raw_x': d_x, 'd_raw_y': d_y,
        'd_mag': d_mag,
    }
    
    return d_hat_x, d_hat_y, debug


def get_desired_direction(
    particle_x: float, particle_y: float,
    target_x: float, target_y: float,
    cfg: Config,
    mode: ControlMode,
) -> Tuple[float, float, Dict[str, float]]:
    """
    Unified interface for desired direction based on control mode.
    """
    if mode == ControlMode.WAYPOINT:
        return get_desired_direction_waypoint(particle_x, particle_y, target_x, target_y, cfg)
    else:
        return get_desired_direction_pathtrack(particle_x, particle_y, cfg)


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
    Compute score for an action.
    
    score = w_align * (F̂ · d̂) + w_push * (F · d̂) - w_switch * switch_penalty
    
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
    """Log for one step."""
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
    arc_progress: float
    d_hat_x: float
    d_hat_y: float
    tangent_alignment: float  # F̂ · t̂ (how aligned is force with tangent)


# =============================================================================
# GREEDY CONTROLLER
# =============================================================================

def run_greedy(
    ev: Evaluator4Pucks,
    x0: float, y0: float,
    ctrl0: Control4Pucks,
    waypoints: List[Tuple[float, float]],
    cfg: Config,
    mode: ControlMode,
    verbose: bool = True,
) -> Tuple[List[StepLog], List[np.ndarray], List[Control4Pucks]]:
    """
    Run greedy controller for T steps.
    
    Args:
        mode: ControlMode.WAYPOINT or ControlMode.PATH_TRACK
    
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
    
    mode_str = "Path-Track" if mode == ControlMode.PATH_TRACK else "Waypoint"
    
    if verbose:
        print(f"\n   Running Greedy ({mode_str}): T={cfg.T} steps")
        print(f"   Actions: {len(action_set)}")
        print(f"   Progress: ", end="", flush=True)
    
    start_time = time.time()
    
    for t in range(cfg.T):
        target_x, target_y = waypoints[target_idx]
        
        # Get desired direction based on mode
        d_hat_x, d_hat_y, debug_info = get_desired_direction(x, y, target_x, target_y, cfg, mode)
        
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
        
        # Compute geometry for logging
        geom = compute_circle_geometry(x_new, y_new, cfg)
        
        # Check waypoint advancement
        dist_to_target = np.sqrt((x_new - target_x)**2 + (y_new - target_y)**2)
        target_advanced = False
        if dist_to_target < cfg.waypoint_tol and target_idx < len(waypoints) - 1:
            target_idx += 1
            target_advanced = True
        
        # Compute tangent alignment (F̂ · t̂)
        t_hat_x, t_hat_y = geom['t_hat']
        Fp_mag = best_metrics[2]
        if Fp_mag > 1e-15:
            tangent_alignment = (best_Fx * t_hat_x + best_Fy * t_hat_y) / Fp_mag
        else:
            tangent_alignment = 0.0
        
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
            cross_track_error=geom['e_perp'],
            arc_progress=geom['s'],
            d_hat_x=d_hat_x,
            d_hat_y=d_hat_y,
            tangent_alignment=tangent_alignment,
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
        # Final arc progress
        if step_logs:
            final_arc = step_logs[-1].arc_progress * 1e3
            print(f"   Arc progress: {final_arc:.3f} mm")
    
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
    mode: ControlMode,
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
        # Get desired direction based on mode
        d_hat_x, d_hat_y, _ = get_desired_direction(current_x, current_y, target_x, target_y, cfg, mode)
        
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
    mode: ControlMode,
) -> Tuple[MacroActionType4Puck, float, int]:
    """
    MPC with beam search over K-step action sequences.
    
    Returns: (best_first_action, best_score, n_sequences_evaluated)
    """
    K = cfg.K
    beam_width = cfg.n_top_actions
    n_evaluated = 0
    
    # Initialize beam with single-action sequences
    beam = []
    
    for action_type in action_set:
        total_score, positions, ctrls = mpc_rollout(
            ev, ctrl, x, y, [action_type], target_x, target_y, prev_action, cfg, mode
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
            branch_prev_action = parent_seq[-1] if parent_seq else prev_action
            
            for action_type in action_set:
                full_seq = parent_seq + [action_type]
                total_score, positions, ctrls = mpc_rollout(
                    ev, ctrl, x, y, full_seq, target_x, target_y, prev_action, cfg, mode
                )
                n_evaluated += len(action_set)
                new_beam.append((total_score, full_seq, positions[-1], ctrls[-1]))
        
        new_beam.sort(key=lambda x: -x[0])
        beam = new_beam[:beam_width]
    
    best_score, best_seq, _, _ = beam[0]
    return best_seq[0], best_score, n_evaluated


def run_mpc(
    ev: Evaluator4Pucks,
    x0: float, y0: float,
    ctrl0: Control4Pucks,
    waypoints: List[Tuple[float, float]],
    cfg: Config,
    mode: ControlMode,
    verbose: bool = True,
) -> Tuple[List[StepLog], List[np.ndarray], List[Control4Pucks]]:
    """
    Run MPC controller for T steps.
    """
    action_set = get_action_set()
    
    step_logs = []
    U_fields = []
    controls = [ctrl0]
    
    x, y = x0, y0
    ctrl = ctrl0
    target_idx = 0
    prev_action = None
    
    mode_str = "Path-Track" if mode == ControlMode.PATH_TRACK else "Waypoint"
    
    if verbose:
        print(f"\n   Running MPC ({mode_str}): T={cfg.T} steps, K={cfg.K} horizon")
        print(f"   Actions: {len(action_set)}, Beam width: {cfg.n_top_actions}")
        print(f"   Progress: ", end="", flush=True)
    
    start_time = time.time()
    
    for t in range(cfg.T):
        target_x, target_y = waypoints[target_idx]
        
        # Get desired direction for logging
        d_hat_x, d_hat_y, _ = get_desired_direction(x, y, target_x, target_y, cfg, mode)
        
        t0_solve = time.time()
        
        # MPC beam search
        best_action_type, best_mpc_score, n_evaluated = mpc_beam_search(
            ev, ctrl, x, y, target_x, target_y, prev_action, cfg, action_set, mode
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
        
        # Compute geometry for logging
        geom = compute_circle_geometry(x_new, y_new, cfg)
        
        # Check waypoint advancement
        dist_to_target = np.sqrt((x_new - target_x)**2 + (y_new - target_y)**2)
        target_advanced = False
        if dist_to_target < cfg.waypoint_tol and target_idx < len(waypoints) - 1:
            target_idx += 1
            target_advanced = True
        
        # Compute tangent alignment
        t_hat_x, t_hat_y = geom['t_hat']
        if Fp_mag > 1e-15:
            tangent_alignment = (best_Fx * t_hat_x + best_Fy * t_hat_y) / Fp_mag
        else:
            tangent_alignment = 0.0
        
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
            cross_track_error=geom['e_perp'],
            arc_progress=geom['s'],
            d_hat_x=d_hat_x,
            d_hat_y=d_hat_y,
            tangent_alignment=tangent_alignment,
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
        if step_logs:
            final_arc = step_logs[-1].arc_progress * 1e3
            print(f"   Arc progress: {final_arc:.3f} mm")
    
    return step_logs, U_fields, controls


# =============================================================================
# Output Saving
# =============================================================================

def save_steps_csv(out_path: Path, logs: List[StepLog]):
    """Save step logs to CSV."""
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header
        writer.writerow([
            'step_idx', 'particle_x', 'particle_y', 'target_x', 'target_y',
            'tracking_error', 'chosen_action', 'action_switched',
            'Fp_x', 'Fp_y', 'Fp_mag', 'Fp_hat_dot_d', 'Fp_dot_d', 'score',
            'solver_time_ms', 'n_actions_evaluated', 'target_idx',
            'dist_to_target', 'target_advanced', 'gates_active', 'cross_track_error',
            'arc_progress', 'd_hat_x', 'd_hat_y', 'tangent_alignment'
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
                log.cross_track_error, log.arc_progress, log.d_hat_x, log.d_hat_y,
                log.tangent_alignment
            ])
    print(f"   Saved: {out_path}")


def compute_summary(logs: List[StepLog], cfg: Config, method: str, mode: str) -> Dict[str, Any]:
    """Compute summary statistics."""
    if not logs:
        return {}
    
    # Extract arrays
    x_arr = np.array([log.particle_x for log in logs])
    y_arr = np.array([log.particle_y for log in logs])
    Fp_mag_arr = np.array([log.Fp_mag for log in logs])
    tracking_err_arr = np.array([log.tracking_error for log in logs])
    cross_track_arr = np.array([log.cross_track_error for log in logs])
    arc_progress_arr = np.array([log.arc_progress for log in logs])
    tangent_align_arr = np.array([log.tangent_alignment for log in logs])
    solver_time_arr = np.array([log.solver_time_ms for log in logs])
    
    # Total distance traveled
    dx = np.diff(x_arr)
    dy = np.diff(y_arr)
    if len(dx) > 0:
        total_distance = np.sum(np.sqrt(dx**2 + dy**2))
    else:
        total_distance = 0.0
    
    # Arc length progress (use logged value)
    arc_progress = arc_progress_arr[-1] if len(arc_progress_arr) > 0 else 0.0
    
    # Circle fraction (arc progress / circumference)
    circumference = 2 * np.pi * cfg.R
    circle_fraction = arc_progress / circumference
    
    # Waypoint progress
    final_waypoint = logs[-1].target_idx
    
    # Action switches
    n_switches = sum(1 for log in logs if log.action_switched)
    
    # Cross-track statistics
    abs_cross_track = np.abs(cross_track_arr)
    
    return {
        'method': method,
        'mode': mode,
        'n_steps': len(logs),
        'final_waypoint': final_waypoint,
        'n_waypoints': cfg.n_waypoints,
        'waypoint_completion': final_waypoint / cfg.n_waypoints,
        'total_distance_mm': float(total_distance * 1e3),
        'arc_progress_mm': float(arc_progress * 1e3),
        'circle_fraction': float(circle_fraction),
        'mean_cross_track_error_um': float(np.mean(abs_cross_track) * 1e6),
        'max_cross_track_error_um': float(np.max(abs_cross_track) * 1e6),
        'std_cross_track_error_um': float(np.std(cross_track_arr) * 1e6),
        'mean_tracking_error_um': float(np.mean(tracking_err_arr) * 1e6),
        'max_tracking_error_um': float(np.max(tracking_err_arr) * 1e6),
        'mean_force_N': float(np.mean(Fp_mag_arr)),
        'max_force_N': float(np.max(Fp_mag_arr)),
        'mean_tangent_alignment': float(np.mean(tangent_align_arr)),
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
            'v_parallel': cfg.v_parallel,
            'k_perp': cfg.k_perp,
            'k_radial_waypoint': cfg.k_radial_waypoint,
        },
        'summary': summary,
    }
    with open(out_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"   Saved: {out_path}")


# =============================================================================
# Visualization
# =============================================================================

def create_gorkov_gif(
    out_path: Path,
    logs: List[StepLog],
    U_fields: List[np.ndarray],
    controls: List[Control4Pucks],
    ev: Evaluator4Pucks,
    cfg: Config,
    title_prefix: str,
    max_frames: int = 100,
    frame_duration: float = 0.1,
    show_desired_direction: bool = True,
):
    """Create Gor'kov contour GIF with optional desired direction arrows."""
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
    
    # Compute global U limits
    all_U = np.concatenate([U_fields[i].flatten() for i in indices])
    U_lo, U_hi = np.nanpercentile(all_U, [2, 98])
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
        
        # Transducers
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
        
        # Force arrow (magenta)
        F_scale = 1e8
        ax.arrow(log.particle_x * 1e3, log.particle_y * 1e3,
                 log.Fp_x * F_scale * 0.1, log.Fp_y * F_scale * 0.1,
                 head_width=0.03, head_length=0.01, fc='magenta', ec='black', zorder=100)
        
        # Desired direction arrow (yellow, if enabled)
        if show_desired_direction:
            arrow_len = 0.15  # mm
            ax.arrow(log.particle_x * 1e3, log.particle_y * 1e3,
                     log.d_hat_x * arrow_len, log.d_hat_y * arrow_len,
                     head_width=0.025, head_length=0.015, fc='yellow', ec='black',
                     linewidth=1.5, zorder=99, alpha=0.9)
        
        ax.set_xlim(x_mm[0], x_mm[-1])
        ax.set_ylim(y_mm[0], y_mm[-1])
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)
        
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='A'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='B'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=8, label='C'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=8, label='D'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
        
        title = f"{title_prefix}: Step {t}/{n_total-1}\n"
        title += f"CTE={log.cross_track_error*1e6:.1f}µm, Arc={log.arc_progress*1e3:.2f}mm, Gates={log.gates_active}"
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


def create_split_screen_gif(
    out_path: Path,
    logs_left: List[StepLog],
    logs_right: List[StepLog],
    U_fields_left: List[np.ndarray],
    U_fields_right: List[np.ndarray],
    controls_left: List[Control4Pucks],
    controls_right: List[Control4Pucks],
    ev: Evaluator4Pucks,
    cfg: Config,
    title_left: str,
    title_right: str,
    max_frames: int = 100,
    frame_duration: float = 0.12,
):
    """Create side-by-side comparison GIF."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    import imageio.v2 as imageio
    
    n_total = min(len(logs_left), len(logs_right))
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
    
    # Compute global U limits across BOTH datasets for fair comparison
    all_U_left = np.concatenate([U_fields_left[i].flatten() for i in indices])
    all_U_right = np.concatenate([U_fields_right[i].flatten() for i in indices])
    all_U = np.concatenate([all_U_left, all_U_right])
    U_lo, U_hi = np.nanpercentile(all_U, [2, 98])
    if U_hi - U_lo < 1e-20:
        U_lo = U_lo - 1e-18
        U_hi = U_hi + 1e-18
    
    for frame_idx, t in enumerate(indices):
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6))
        
        for ax, logs, U_fields, controls, title in [
            (ax_left, logs_left, U_fields_left, controls_left, title_left),
            (ax_right, logs_right, U_fields_right, controls_right, title_right),
        ]:
            U = U_fields[t]
            log = logs[t]
            ctrl = controls[t]
            
            # Contours
            levels = np.linspace(U_lo, U_hi, 25)
            contourf = ax.contourf(X, Y, U, levels=levels, cmap="RdBu_r", alpha=0.85, extend='both')
            ax.contour(X, Y, U, levels=levels[::2], colors="k", linewidths=0.3, alpha=0.3)
            
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
            
            # Transducers
            transducer_info = [
                ('A', 'blue', ctrl.xA, ctrl.yA, ctrl.gateA),
                ('B', 'green', ctrl.xB, ctrl.yB, ctrl.gateB),
                ('C', 'orange', ctrl.xC, ctrl.yC, ctrl.gateC),
                ('D', 'purple', ctrl.xD, ctrl.yD, ctrl.gateD),
            ]
            for name, puck_color, px, py, gate in transducer_info:
                marker = 'o' if gate else 'x'
                alpha_val = 1.0 if gate else 0.3
                ax.scatter(px * 1e3, py * 1e3, s=100, c=puck_color, marker=marker,
                           edgecolors='white' if gate else 'gray', linewidths=1.5,
                           alpha=alpha_val, zorder=90)
            
            # Particle
            ax.scatter(log.particle_x * 1e3, log.particle_y * 1e3, s=200, marker='o',
                       color='red', edgecolors='white', linewidth=2, zorder=100)
            
            # Force arrow
            F_scale = 1e8
            ax.arrow(log.particle_x * 1e3, log.particle_y * 1e3,
                     log.Fp_x * F_scale * 0.1, log.Fp_y * F_scale * 0.1,
                     head_width=0.025, head_length=0.01, fc='magenta', ec='black', zorder=100)
            
            # Desired direction arrow
            arrow_len = 0.12
            ax.arrow(log.particle_x * 1e3, log.particle_y * 1e3,
                     log.d_hat_x * arrow_len, log.d_hat_y * arrow_len,
                     head_width=0.02, head_length=0.012, fc='yellow', ec='black',
                     linewidth=1.2, zorder=99, alpha=0.9)
            
            ax.set_xlim(x_mm[0], x_mm[-1])
            ax.set_ylim(y_mm[0], y_mm[-1])
            ax.set_xlabel('x (mm)')
            ax.set_ylabel('y (mm)')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.2)
            
            subtitle = f"CTE={log.cross_track_error*1e6:.1f}µm, Arc={log.arc_progress*1e3:.2f}mm"
            ax.set_title(f"{title}\n{subtitle}", fontsize=10)
        
        fig.suptitle(f"Step {t}/{n_total-1}", fontsize=12, fontweight='bold')
        fig.tight_layout()
        
        frame_path = temp_dir / f"frame_{frame_idx:04d}.png"
        fig.savefig(frame_path, dpi=90)
        plt.close(fig)
        
        frames.append(imageio.imread(frame_path))
    
    imageio.mimsave(out_path, frames, duration=frame_duration)
    
    for f in temp_dir.glob("*.png"):
        f.unlink()
    temp_dir.rmdir()
    
    print(f"   Saved: {out_path}")


def create_comparison_plots(
    out_path: Path,
    all_logs: Dict[str, List[StepLog]],
    all_summaries: Dict[str, Dict],
    cfg: Config,
):
    """Create comprehensive comparison plots."""
    import matplotlib.pyplot as plt
    
    fig = plt.figure(figsize=(16, 12))
    
    # Define colors and styles for each method
    styles = {
        'waypoint_greedy': {'color': 'blue', 'linestyle': '-', 'label': 'Waypoint Greedy'},
        'waypoint_mpc': {'color': 'cyan', 'linestyle': '-', 'label': 'Waypoint MPC'},
        'pathtrack_greedy': {'color': 'red', 'linestyle': '--', 'label': 'PathTrack Greedy'},
        'pathtrack_mpc': {'color': 'orange', 'linestyle': '--', 'label': 'PathTrack MPC'},
    }
    
    # 1. Trajectory plot
    ax1 = fig.add_subplot(2, 3, 1)
    theta = np.linspace(0, 2*np.pi, 100)
    ax1.plot(cfg.cx*1e3 + cfg.R*1e3*np.cos(theta), cfg.cy*1e3 + cfg.R*1e3*np.sin(theta),
             'g--', linewidth=2, alpha=0.5, label='Target path')
    
    for name, logs in all_logs.items():
        if logs:
            x = [log.particle_x * 1e3 for log in logs]
            y = [log.particle_y * 1e3 for log in logs]
            style = styles.get(name, {'color': 'gray', 'linestyle': '-', 'label': name})
            ax1.plot(x, y, color=style['color'], linestyle=style['linestyle'],
                     linewidth=1.5, alpha=0.8, label=style['label'])
    
    ax1.set_xlabel('x (mm)')
    ax1.set_ylabel('y (mm)')
    ax1.set_aspect('equal')
    ax1.legend(fontsize=8)
    ax1.set_title('Trajectories')
    ax1.grid(True, alpha=0.3)
    
    # 2. Cross-track error over time
    ax2 = fig.add_subplot(2, 3, 2)
    for name, logs in all_logs.items():
        if logs:
            cte = [log.cross_track_error * 1e6 for log in logs]
            t_arr = np.arange(len(logs))
            style = styles.get(name, {'color': 'gray', 'linestyle': '-', 'label': name})
            ax2.plot(t_arr, cte, color=style['color'], linestyle=style['linestyle'],
                     linewidth=1, alpha=0.8, label=f"{style['label']}")
    ax2.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Cross-track Error (µm)')
    ax2.legend(fontsize=7)
    ax2.set_title('Path Tracking Error')
    ax2.grid(True, alpha=0.3)
    
    # 3. Arc progress over time
    ax3 = fig.add_subplot(2, 3, 3)
    for name, logs in all_logs.items():
        if logs:
            arc = [log.arc_progress * 1e3 for log in logs]
            t_arr = np.arange(len(logs))
            style = styles.get(name, {'color': 'gray', 'linestyle': '-', 'label': name})
            ax3.plot(t_arr, arc, color=style['color'], linestyle=style['linestyle'],
                     linewidth=1.5, alpha=0.8, label=style['label'])
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Arc Progress (mm)')
    ax3.legend(fontsize=8)
    ax3.set_title('Progress Along Path')
    ax3.grid(True, alpha=0.3)
    
    # 4. Action switches (cumulative)
    ax4 = fig.add_subplot(2, 3, 4)
    for name, logs in all_logs.items():
        if logs:
            switches = np.cumsum([1 if log.action_switched else 0 for log in logs])
            t_arr = np.arange(len(logs))
            style = styles.get(name, {'color': 'gray', 'linestyle': '-', 'label': name})
            ax4.plot(t_arr, switches, color=style['color'], linestyle=style['linestyle'],
                     linewidth=1.5, alpha=0.8, label=style['label'])
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Cumulative Action Switches')
    ax4.legend(fontsize=8)
    ax4.set_title('Action Switching')
    ax4.grid(True, alpha=0.3)
    
    # 5. Tangent alignment
    ax5 = fig.add_subplot(2, 3, 5)
    for name, logs in all_logs.items():
        if logs:
            align = [log.tangent_alignment for log in logs]
            t_arr = np.arange(len(logs))
            style = styles.get(name, {'color': 'gray', 'linestyle': '-', 'label': name})
            # Moving average for smoothness
            window = 10
            if len(align) > window:
                align_smooth = np.convolve(align, np.ones(window)/window, mode='valid')
                t_smooth = t_arr[window//2:window//2+len(align_smooth)]
                ax5.plot(t_smooth, align_smooth, color=style['color'], linestyle=style['linestyle'],
                         linewidth=1.5, alpha=0.8, label=style['label'])
    ax5.axhline(1.0, color='g', linestyle='--', alpha=0.3, label='Perfect tangent')
    ax5.set_xlabel('Step')
    ax5.set_ylabel('Force-Tangent Alignment (F̂·t̂)')
    ax5.legend(fontsize=8)
    ax5.set_title('Force Alignment with Tangent')
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(-1.1, 1.1)
    
    # 6. Summary bar chart
    ax6 = fig.add_subplot(2, 3, 6)
    methods = list(all_summaries.keys())
    metrics_to_plot = ['mean_cross_track_error_um', 'n_action_switches', 'arc_progress_mm']
    x = np.arange(len(metrics_to_plot))
    width = 0.2
    
    for i, (name, summary) in enumerate(all_summaries.items()):
        if summary:
            values = [summary.get(m, 0) for m in metrics_to_plot]
            # Normalize for display
            values_norm = [
                values[0] / max(1, max(s.get('mean_cross_track_error_um', 1) for s in all_summaries.values() if s)),
                values[1] / max(1, max(s.get('n_action_switches', 1) for s in all_summaries.values() if s)),
                values[2] / max(0.001, max(s.get('arc_progress_mm', 0.001) for s in all_summaries.values() if s)),
            ]
            style = styles.get(name, {'color': 'gray', 'label': name})
            bars = ax6.bar(x + i*width, values_norm, width, label=style['label'], color=style['color'], alpha=0.8)
            # Add value labels
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax6.annotate(f'{val:.1f}' if val > 1 else f'{val:.2f}',
                            xy=(bar.get_x() + bar.get_width()/2, height),
                            xytext=(0, 2), textcoords="offset points",
                            ha='center', va='bottom', fontsize=6, rotation=45)
    
    ax6.set_ylabel('Normalized Value')
    ax6.set_title('Summary Comparison')
    ax6.set_xticks(x + width * 1.5)
    ax6.set_xticklabels(['Mean CTE (µm)', 'Switches', 'Arc (mm)'])
    ax6.legend(fontsize=7)
    ax6.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle('Path-Tracking vs Waypoint-Chasing: Comprehensive Comparison', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    
    print(f"   Saved: {out_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Path-tracking vs Waypoint comparison")
    
    # Grid
    parser.add_argument("--Nx", type=int, default=80)
    parser.add_argument("--Ny", type=int, default=80)
    
    # Control parameters
    parser.add_argument("--K", type=int, default=3, help="MPC horizon length")
    parser.add_argument("--T", type=int, default=300, help="Total steps")
    parser.add_argument("--beam_width", type=int, default=5, help="Beam search width")
    
    # Path-tracking parameters
    parser.add_argument("--v_parallel", type=float, default=1.0, help="Tangential velocity gain")
    parser.add_argument("--k_perp", type=float, default=5.0, help="Cross-track correction gain")
    
    # Modes
    parser.add_argument("--fast", action="store_true", help="Quick test mode")
    parser.add_argument("--skip_waypoint", action="store_true", help="Skip waypoint controllers")
    parser.add_argument("--skip_pathtrack", action="store_true", help="Skip path-track controllers")
    parser.add_argument("--greedy_only", action="store_true", help="Run only greedy (both modes)")
    parser.add_argument("--mpc_only", action="store_true", help="Run only MPC (both modes)")
    
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
        v_parallel=args.v_parallel,
        k_perp=args.k_perp,
    )
    
    if args.fast:
        cfg.T = 100
        cfg.K = 2
        cfg.Nx = 64
        cfg.Ny = 64
        cfg.n_top_actions = 3
    
    print("\n" + "="*70)
    print("  PATH-TRACKING vs WAYPOINT-CHASING COMPARISON")
    print("="*70)
    
    if args.fast:
        print("   [FAST MODE]")
    
    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = project_root / "results" / "path_tracking_comparison" / f"run_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n   Output: {out_dir}")
    print(f"   Config: T={cfg.T}, K={cfg.K}, beam_width={cfg.n_top_actions}")
    print(f"   Path-track params: v_∥={cfg.v_parallel}, k_⊥={cfg.k_perp}")
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
    
    # Initial control
    Lx, Ly = cfg.Lx, cfg.Ly
    ctrl0 = Control4Pucks(
        xA=x0 - 0.4e-3, yA=0.03e-3, vA=0.08, phiA=0.0, gateA=True,
        xB=x0 + 0.4e-3, yB=0.03e-3, vB=0.08, phiB=np.pi, gateB=True,
        xC=x0, yC=0.20e-3, vC=0.08, phiC=np.pi/2, gateC=True,
        xD=x0, yD=1.8e-3, vD=0.05, phiD=-np.pi/2, gateD=True,
    )
    ctrl0 = ev.clip_control(ctrl0)
    
    print(f"\n   Initial position: ({x0*1e3:.3f}, {y0*1e3:.3f}) mm")
    
    # Storage for all results
    all_logs = {}
    all_U_fields = {}
    all_controls = {}
    all_summaries = {}
    
    # Define what to run
    runs_to_do = []
    
    if not args.skip_waypoint:
        if not args.mpc_only:
            runs_to_do.append(('waypoint_greedy', ControlMode.WAYPOINT, 'greedy'))
        if not args.greedy_only:
            runs_to_do.append(('waypoint_mpc', ControlMode.WAYPOINT, 'mpc'))
    
    if not args.skip_pathtrack:
        if not args.mpc_only:
            runs_to_do.append(('pathtrack_greedy', ControlMode.PATH_TRACK, 'greedy'))
        if not args.greedy_only:
            runs_to_do.append(('pathtrack_mpc', ControlMode.PATH_TRACK, 'mpc'))
    
    # Run each controller
    for name, mode, controller_type in runs_to_do:
        print("\n" + "-"*50)
        mode_str = "Path-Track" if mode == ControlMode.PATH_TRACK else "Waypoint"
        ctrl_str = controller_type.upper()
        print(f"  Running {ctrl_str} ({mode_str})")
        print("-"*50)
        
        # Create output subdirectory
        sub_dir = out_dir / name
        sub_dir.mkdir(exist_ok=True)
        
        if controller_type == 'greedy':
            logs, U_fields, ctrls = run_greedy(ev, x0, y0, ctrl0, waypoints, cfg, mode)
        else:
            logs, U_fields, ctrls = run_mpc(ev, x0, y0, ctrl0, waypoints, cfg, mode)
        
        all_logs[name] = logs
        all_U_fields[name] = U_fields
        all_controls[name] = ctrls
        
        # Save outputs
        print(f"\n   Saving {name} outputs...")
        save_steps_csv(sub_dir / "steps.csv", logs)
        summary = compute_summary(logs, cfg, controller_type, mode.value)
        all_summaries[name] = summary
        save_summary_json(sub_dir / "summary.json", summary, cfg)
        
        # Create GIF
        title = f"{mode_str} {ctrl_str}"
        print(f"   Creating {name} GIF...")
        create_gorkov_gif(sub_dir / "gorkov.gif", logs, U_fields, ctrls, ev, cfg, title,
                          max_frames=100, frame_duration=0.1)
    
    # Create comparison outputs
    print("\n" + "-"*50)
    print("  Creating Comparison Outputs")
    print("-"*50)
    
    comp_dir = out_dir / "comparison"
    comp_dir.mkdir(exist_ok=True)
    
    # Save combined summary
    combined_summary = {
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
            'v_parallel': cfg.v_parallel,
            'k_perp': cfg.k_perp,
            'k_radial_waypoint': cfg.k_radial_waypoint,
        },
        'summaries': all_summaries,
    }
    
    with open(comp_dir / "compare_summary.json", 'w') as f:
        json.dump(combined_summary, f, indent=2)
    print(f"   Saved: {comp_dir / 'compare_summary.json'}")
    
    # Create comparison plots
    create_comparison_plots(comp_dir / "comparison_plots.png", all_logs, all_summaries, cfg)
    
    # Create split-screen GIFs
    print("\n   Creating split-screen comparison GIFs...")
    
    # Greedy vs Greedy (if both exist)
    if 'waypoint_greedy' in all_logs and 'pathtrack_greedy' in all_logs:
        create_split_screen_gif(
            comp_dir / "greedy_comparison.gif",
            all_logs['waypoint_greedy'], all_logs['pathtrack_greedy'],
            all_U_fields['waypoint_greedy'], all_U_fields['pathtrack_greedy'],
            all_controls['waypoint_greedy'], all_controls['pathtrack_greedy'],
            ev, cfg,
            "Waypoint Greedy", "Path-Track Greedy",
            max_frames=100
        )
    
    # MPC vs MPC (if both exist)
    if 'waypoint_mpc' in all_logs and 'pathtrack_mpc' in all_logs:
        create_split_screen_gif(
            comp_dir / "mpc_comparison.gif",
            all_logs['waypoint_mpc'], all_logs['pathtrack_mpc'],
            all_U_fields['waypoint_mpc'], all_U_fields['pathtrack_mpc'],
            all_controls['waypoint_mpc'], all_controls['pathtrack_mpc'],
            ev, cfg,
            "Waypoint MPC", "Path-Track MPC",
            max_frames=100
        )
    
    # Best comparison: Waypoint Greedy vs Path-Track MPC (if both exist)
    if 'waypoint_greedy' in all_logs and 'pathtrack_mpc' in all_logs:
        create_split_screen_gif(
            comp_dir / "best_comparison.gif",
            all_logs['waypoint_greedy'], all_logs['pathtrack_mpc'],
            all_U_fields['waypoint_greedy'], all_U_fields['pathtrack_mpc'],
            all_controls['waypoint_greedy'], all_controls['pathtrack_mpc'],
            ev, cfg,
            "Waypoint Greedy (Baseline)", "Path-Track MPC (Best)",
            max_frames=100
        )
    
    # Print summary table
    print("\n" + "="*70)
    print("  RESULTS SUMMARY")
    print("="*70)
    
    print("\n   {:<20} {:>12} {:>12} {:>12} {:>12}".format(
        "Method", "CTE (µm)", "Switches", "Arc (mm)", "Circle %"))
    print("   " + "-"*60)
    
    for name, summary in all_summaries.items():
        if summary:
            print("   {:<20} {:>12.1f} {:>12d} {:>12.2f} {:>11.1f}%".format(
                name,
                summary['mean_cross_track_error_um'],
                summary['n_action_switches'],
                summary['arc_progress_mm'],
                summary['circle_fraction'] * 100
            ))
    
    # Compute improvements
    if 'waypoint_greedy' in all_summaries and 'pathtrack_greedy' in all_summaries:
        wp_g = all_summaries['waypoint_greedy']
        pt_g = all_summaries['pathtrack_greedy']
        print("\n   Path-Track Greedy improvements over Waypoint Greedy:")
        if wp_g['mean_cross_track_error_um'] > 0:
            cte_improve = (wp_g['mean_cross_track_error_um'] - pt_g['mean_cross_track_error_um']) / wp_g['mean_cross_track_error_um'] * 100
            print(f"      CTE reduction: {cte_improve:+.1f}%")
        if wp_g['n_action_switches'] > 0:
            switch_improve = (wp_g['n_action_switches'] - pt_g['n_action_switches']) / wp_g['n_action_switches'] * 100
            print(f"      Switch reduction: {switch_improve:+.1f}%")
        if wp_g['arc_progress_mm'] > 0:
            arc_improve = (pt_g['arc_progress_mm'] - wp_g['arc_progress_mm']) / wp_g['arc_progress_mm'] * 100
            print(f"      Arc progress improvement: {arc_improve:+.1f}%")
    
    if 'waypoint_mpc' in all_summaries and 'pathtrack_mpc' in all_summaries:
        wp_m = all_summaries['waypoint_mpc']
        pt_m = all_summaries['pathtrack_mpc']
        print("\n   Path-Track MPC improvements over Waypoint MPC:")
        if wp_m['mean_cross_track_error_um'] > 0:
            cte_improve = (wp_m['mean_cross_track_error_um'] - pt_m['mean_cross_track_error_um']) / wp_m['mean_cross_track_error_um'] * 100
            print(f"      CTE reduction: {cte_improve:+.1f}%")
        if wp_m['n_action_switches'] > 0:
            switch_improve = (wp_m['n_action_switches'] - pt_m['n_action_switches']) / wp_m['n_action_switches'] * 100
            print(f"      Switch reduction: {switch_improve:+.1f}%")
        if wp_m['arc_progress_mm'] > 0:
            arc_improve = (pt_m['arc_progress_mm'] - wp_m['arc_progress_mm']) / wp_m['arc_progress_mm'] * 100
            print(f"      Arc progress improvement: {arc_improve:+.1f}%")
    
    print("\n" + "="*70)
    print("  COMPLETE")
    print("="*70)
    print(f"\n   All outputs saved to: {out_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
