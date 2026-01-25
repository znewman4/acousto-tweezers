#!/usr/bin/env python3
"""
optimized_mpc_comparison.py - MPC Computational Optimization (Option B)

Implements three key optimizations to make beam-search MPC computationally viable:

B1) MEMOIZATION / EVALUATION CACHING
    - Caches PDE solve results keyed by (control_signature, action_id)
    - Shared across all MPC rollouts
    - Logs cache hit/miss statistics

B2) PROGRESSIVE WIDENING (ACTION PRUNING)  
    - Ranks actions using cheap heuristics (no PDE solve)
    - Evaluates only top N_initial actions
    - Falls back to full expansion if needed

B3) TWO-STAGE SCORING (CHEAP → EXPENSIVE)
    - Stage 1: Approximate score using last force field (no PDE)
    - Stage 2: Full PDE solve only for top-K candidates

Target: MPC runtime within 2-5× greedy (currently ~27×)

Usage:
    python scripts/optimized_mpc_comparison.py --fast           # Quick test
    python scripts/optimized_mpc_comparison.py --T 300 --K 3    # Full run
    python scripts/optimized_mpc_comparison.py --ablation       # Test each optimization
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
from dataclasses import dataclass, field
from datetime import datetime
from typing import Tuple, List, Dict, Any, Optional, Set
from enum import Enum, auto
import time
import hashlib
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "scripts"))

# Import infrastructure
from tweezers.control import (
    DishDomain, MediumProps, EvaluatorConfig,
    Control4Pucks, default_4puck_config,
)
from tweezers.control.evaluator_4pucks import Evaluator4Pucks
from acousto.force import ParticleProps, gorkov_potential_and_force_2d, bilinear_sample_vec

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
    """Configuration for optimized MPC comparison."""
    # Domain
    Lx: float = 2.0e-3
    Ly: float = 2.0e-3
    Nx: int = 80
    Ny: int = 80
    
    # Physics
    f: float = 2.0e6
    c0: float = 1500.0
    rho0: float = 1000.0
    loss_eta: float = 1e-3
    kz: float = 0.0
    coupling_alpha: float = 1.0
    
    # Transducer params
    sigma_x: float = 0.10e-3
    sigma_y: float = 0.15e-3
    bottom_band: float = 0.25e-3
    
    # Particle
    particle_a: float = 5.0e-6
    particle_rho_p: float = 1050.0
    particle_c_p: float = 2350.0
    
    # Dynamics
    dt: float = 5e-3
    viscosity: float = 1e-3
    alpha_g: float = 2e3
    max_step: float = 0.08e-3
    
    # Macro action parameters
    macro_magnitude: float = 0.05e-3
    macro_phase_step: float = 0.15
    macro_amplitude_step: float = 0.01
    
    # Scoring weights
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
    cx: float = 1.0e-3
    cy: float = 1.1e-3
    R: float = 0.4e-3
    ccw: bool = True
    n_waypoints: int = 400
    waypoint_tol: float = 0.12e-3
    
    # Path-tracking parameters
    v_parallel: float = 1.0
    k_perp: float = 3.0
    
    theta0: float = 0.0
    
    # === OPTIMIZATION PARAMETERS (Option B) ===
    
    # B1) Caching
    enable_cache: bool = True
    cache_precision: int = 6      # Decimal places for control state hashing
    
    # B2) Progressive Widening / Action Pruning
    enable_pruning: bool = True
    n_initial_actions: int = 15   # Actions to evaluate with cheap heuristic (was 10)
    n_full_eval_actions: int = 10  # Actions to fully evaluate with PDE (was 6)
    
    # B3) Two-Stage Scoring
    enable_two_stage: bool = True
    stage1_candidates: int = 15   # Candidates to consider in stage 1 (was 12)
    stage2_candidates: int = 8    # Candidates to fully evaluate in stage 2 (was 5)


# =============================================================================
# B1) EVALUATION CACHE
# =============================================================================

class EvaluationCache:
    """
    Memoization cache for PDE evaluations.
    
    Keys: (control_signature, action_id)
    Values: (new_ctrl, Fx, Fy, U_field, score_components)
    
    This dramatically reduces redundant PDE solves in MPC beam search.
    """
    
    def __init__(self, precision: int = 6):
        self.precision = precision
        self.cache: Dict[str, Tuple] = {}
        self.hits = 0
        self.misses = 0
        self.total_queries = 0
    
    def _control_signature(self, ctrl: Control4Pucks, particle_x: float = None, particle_y: float = None) -> str:
        """Generate a hash signature for a control state and particle position."""
        # Round to precision to handle floating point noise
        def r(x): return round(x, self.precision)
        
        state = (
            r(ctrl.xA), r(ctrl.yA), r(ctrl.vA), r(ctrl.phiA), ctrl.gateA,
            r(ctrl.xB), r(ctrl.yB), r(ctrl.vB), r(ctrl.phiB), ctrl.gateB,
            r(ctrl.xC), r(ctrl.yC), r(ctrl.vC), r(ctrl.phiC), ctrl.gateC,
            r(ctrl.xD), r(ctrl.yD), r(ctrl.vD), r(ctrl.phiD), ctrl.gateD,
        )
        # Include particle position if provided (important for force sampling!)
        if particle_x is not None and particle_y is not None:
            state = state + (r(particle_x), r(particle_y))
        return hashlib.md5(str(state).encode()).hexdigest()[:16]
    
    def make_key(self, ctrl: Control4Pucks, action_type: MacroActionType4Puck,
                 particle_x: float = None, particle_y: float = None) -> str:
        """Create cache key from control state, action, and particle position."""
        ctrl_sig = self._control_signature(ctrl, particle_x, particle_y)
        return f"{ctrl_sig}_{action_type.name}"
    
    def get(self, ctrl: Control4Pucks, action_type: MacroActionType4Puck,
            particle_x: float = None, particle_y: float = None) -> Optional[Tuple]:
        """Get cached result if available."""
        self.total_queries += 1
        key = self.make_key(ctrl, action_type, particle_x, particle_y)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None
    
    def put(self, ctrl: Control4Pucks, action_type: MacroActionType4Puck, 
            result: Tuple, particle_x: float = None, particle_y: float = None):
        """Store result in cache."""
        key = self.make_key(ctrl, action_type, particle_x, particle_y)
        self.cache[key] = result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        hit_rate = self.hits / max(1, self.total_queries)
        return {
            'total_queries': self.total_queries,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'cache_size': len(self.cache),
        }
    
    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
        self.total_queries = 0


# =============================================================================
# B2) ACTION HEURISTICS (Cheap Ranking)
# =============================================================================

class ActionHeuristics:
    """
    Cheap heuristics for ranking actions without PDE solves.
    
    Used for progressive widening / action pruning in MPC.
    """
    
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.action_history: Dict[MacroActionType4Puck, List[float]] = defaultdict(list)
        self.last_force: Optional[Tuple[float, float]] = None
        self.last_good_actions: List[MacroActionType4Puck] = []
    
    def update_history(self, action: MacroActionType4Puck, score: float):
        """Update action performance history."""
        self.action_history[action].append(score)
        # Keep only recent history
        if len(self.action_history[action]) > 20:
            self.action_history[action] = self.action_history[action][-20:]
    
    def set_last_force(self, Fx: float, Fy: float):
        """Store last known force for heuristic estimation."""
        self.last_force = (Fx, Fy)
    
    def set_last_good_actions(self, actions: List[MacroActionType4Puck]):
        """Store recently good actions."""
        self.last_good_actions = actions[:5]
    
    def compute_geometric_alignment(
        self, 
        action_type: MacroActionType4Puck,
        d_hat: Tuple[float, float],
    ) -> float:
        """
        Estimate action quality based on geometric alignment.
        
        Maps action types to approximate movement directions and
        computes alignment with desired direction.
        """
        d_hat_x, d_hat_y = d_hat
        
        # Approximate movement direction for each action type
        action_directions = {
            MacroActionType4Puck.TRANSLATE_TRAP_X_POS: (1, 0),
            MacroActionType4Puck.TRANSLATE_TRAP_X_NEG: (-1, 0),
            MacroActionType4Puck.TRANSLATE_TRAP_Y_POS: (0, 1),
            MacroActionType4Puck.TRANSLATE_TRAP_Y_NEG: (0, -1),
            MacroActionType4Puck.MOVE_A_RIGHT: (0.3, 0),
            MacroActionType4Puck.MOVE_A_LEFT: (-0.3, 0),
            MacroActionType4Puck.MOVE_B_RIGHT: (0.3, 0),
            MacroActionType4Puck.MOVE_B_LEFT: (-0.3, 0),
            MacroActionType4Puck.MOVE_C_UP: (0, 0.3),
            MacroActionType4Puck.MOVE_C_DOWN: (0, -0.3),
            MacroActionType4Puck.MOVE_C_RIGHT: (0.3, 0),
            MacroActionType4Puck.MOVE_C_LEFT: (-0.3, 0),
            MacroActionType4Puck.MOVE_D_UP: (0, 0.3),
            MacroActionType4Puck.MOVE_D_DOWN: (0, -0.3),
            MacroActionType4Puck.MOVE_D_RIGHT: (0.3, 0),
            MacroActionType4Puck.MOVE_D_LEFT: (-0.3, 0),
            MacroActionType4Puck.ROTATE_INTERFERENCE_CW: (0.2, -0.2),
            MacroActionType4Puck.ROTATE_INTERFERENCE_CCW: (-0.2, 0.2),
            MacroActionType4Puck.HOLD: (0, 0),
        }
        
        if action_type in action_directions:
            ax, ay = action_directions[action_type]
            mag = np.sqrt(ax**2 + ay**2)
            if mag > 1e-6:
                alignment = (ax * d_hat_x + ay * d_hat_y) / mag
                return alignment
        
        # Default: small positive score for unknown actions
        return 0.1
    
    def compute_historical_score(self, action_type: MacroActionType4Puck) -> float:
        """Get historical performance score for an action."""
        history = self.action_history.get(action_type, [])
        if not history:
            return 0.0
        return np.mean(history[-5:])  # Recent average
    
    def compute_continuity_bonus(
        self, 
        action_type: MacroActionType4Puck,
        prev_action: Optional[MacroActionType4Puck],
    ) -> float:
        """Bonus for continuing with similar action (reduces switching)."""
        if prev_action is None:
            return 0.0
        if action_type == prev_action:
            return 0.2  # Bonus for same action
        # Bonus for actions in same category
        action_name = action_type.name
        prev_name = prev_action.name
        if action_name.split('_')[0] == prev_name.split('_')[0]:
            return 0.1
        return 0.0
    
    def rank_actions(
        self,
        action_set: List[MacroActionType4Puck],
        d_hat: Tuple[float, float],
        prev_action: Optional[MacroActionType4Puck],
    ) -> List[Tuple[MacroActionType4Puck, float]]:
        """
        Rank all actions using cheap heuristics.
        
        Returns list of (action, heuristic_score) sorted by score descending.
        """
        scored = []
        for action in action_set:
            score = 0.0
            
            # Geometric alignment (primary)
            score += 2.0 * self.compute_geometric_alignment(action, d_hat)
            
            # Historical performance
            score += 0.5 * self.compute_historical_score(action)
            
            # Continuity bonus
            score += self.compute_continuity_bonus(action, prev_action)
            
            # Bonus for recently good actions
            if action in self.last_good_actions:
                score += 0.3
            
            scored.append((action, score))
        
        # Sort by score descending
        scored.sort(key=lambda x: -x[1])
        return scored


# =============================================================================
# B3) TWO-STAGE SCORING
# =============================================================================

class TwoStageScorer:
    """
    Two-stage scoring: cheap approximation first, then full PDE solve.
    
    Stage 1: Use last known force field to approximate scores
    Stage 2: Full PDE solve only for top candidates
    """
    
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.last_U: Optional[np.ndarray] = None
        self.last_Fx: Optional[np.ndarray] = None
        self.last_Fy: Optional[np.ndarray] = None
        self.last_x: Optional[np.ndarray] = None
        self.last_y: Optional[np.ndarray] = None
    
    def store_field(self, U: np.ndarray, Fx: np.ndarray, Fy: np.ndarray,
                    x: np.ndarray, y: np.ndarray):
        """Store the last computed force field for reuse."""
        self.last_U = U
        self.last_Fx = Fx
        self.last_Fy = Fy
        self.last_x = x
        self.last_y = y
    
    def approximate_force(self, particle_x: float, particle_y: float) -> Tuple[float, float]:
        """
        Approximate force at particle position using last known field.
        
        This is MUCH cheaper than a full PDE solve.
        """
        if self.last_Fx is None or self.last_Fy is None:
            return 0.0, 0.0
        
        try:
            fx, fy = bilinear_sample_vec(
                self.last_x, self.last_y,
                self.last_Fx, self.last_Fy,
                particle_x, particle_y
            )
            return float(fx), float(fy)
        except:
            return 0.0, 0.0
    
    def stage1_score(
        self,
        particle_x: float, particle_y: float,
        d_hat_x: float, d_hat_y: float,
        action_type: MacroActionType4Puck,
        prev_action: Optional[MacroActionType4Puck],
    ) -> float:
        """
        Stage 1: Cheap approximate score using last force field.
        
        Does NOT solve PDE - uses cached force field with position offset.
        """
        # Use last known force as approximation
        Fx, Fy = self.approximate_force(particle_x, particle_y)
        
        Fp_mag = np.sqrt(Fx**2 + Fy**2)
        eps = 1e-15
        
        if Fp_mag > eps:
            Fp_hat_x = Fx / Fp_mag
            Fp_hat_y = Fy / Fp_mag
            Fp_hat_dot_d = Fp_hat_x * d_hat_x + Fp_hat_y * d_hat_y
        else:
            Fp_hat_dot_d = 0.0
        
        Fp_dot_d = Fx * d_hat_x + Fy * d_hat_y
        
        score = self.cfg.w_align * Fp_hat_dot_d + self.cfg.w_push * Fp_dot_d
        
        if prev_action is not None and action_type != prev_action:
            score -= self.cfg.w_switch
        
        return score


# =============================================================================
# Timing Statistics
# =============================================================================

@dataclass
class TimingStats:
    """Detailed timing statistics for MPC."""
    total_steps: int = 0
    total_time_ms: float = 0.0
    pde_solves: int = 0
    pde_time_ms: float = 0.0
    cache_lookups: int = 0
    heuristic_rankings: int = 0
    heuristic_time_ms: float = 0.0
    stage1_evals: int = 0
    stage1_time_ms: float = 0.0
    stage2_evals: int = 0
    stage2_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_steps': self.total_steps,
            'total_time_ms': self.total_time_ms,
            'mean_time_per_step_ms': self.total_time_ms / max(1, self.total_steps),
            'pde_solves': self.pde_solves,
            'pde_time_ms': self.pde_time_ms,
            'mean_pde_time_ms': self.pde_time_ms / max(1, self.pde_solves),
            'cache_lookups': self.cache_lookups,
            'heuristic_rankings': self.heuristic_rankings,
            'heuristic_time_ms': self.heuristic_time_ms,
            'stage1_evals': self.stage1_evals,
            'stage1_time_ms': self.stage1_time_ms,
            'stage2_evals': self.stage2_evals,
            'stage2_time_ms': self.stage2_time_ms,
        }


# =============================================================================
# Core Functions (from path_tracking_comparison.py)
# =============================================================================

def create_evaluator(cfg: Config) -> Evaluator4Pucks:
    """Create Evaluator4Pucks."""
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
    
    return Evaluator4Pucks(domain, medium, particle, ev_cfg)


def get_action_set() -> List[MacroActionType4Puck]:
    """Get the set of macro actions."""
    return get_standard_actions_4puck()


def compute_circle_geometry(particle_x: float, particle_y: float, cfg: Config) -> Dict[str, Any]:
    """Compute path geometry for a circle."""
    eps = 1e-12
    dx = particle_x - cfg.cx
    dy = particle_y - cfg.cy
    r = np.sqrt(dx**2 + dy**2)
    
    n_hat_x = dx / (r + eps)
    n_hat_y = dy / (r + eps)
    
    if cfg.ccw:
        t_hat_x = -n_hat_y
        t_hat_y = n_hat_x
    else:
        t_hat_x = n_hat_y
        t_hat_y = -n_hat_x
    
    e_perp = r - cfg.R
    theta = np.arctan2(dy, dx)
    delta_theta = (theta - cfg.theta0) if cfg.ccw else (cfg.theta0 - theta)
    delta_theta = delta_theta % (2 * np.pi)
    s = cfg.R * delta_theta
    
    return {
        'e_perp': e_perp,
        'n_hat': (n_hat_x, n_hat_y),
        't_hat': (t_hat_x, t_hat_y),
        'theta': theta,
        's': s,
        'r': r,
    }


def get_desired_direction_pathtrack(particle_x: float, particle_y: float, cfg: Config) -> Tuple[float, float]:
    """Path-tracking control law: d_des = v_∥ · t̂ - k_⊥ · e_⊥ · n̂"""
    eps = 1e-12
    geom = compute_circle_geometry(particle_x, particle_y, cfg)
    
    t_hat_x, t_hat_y = geom['t_hat']
    n_hat_x, n_hat_y = geom['n_hat']
    e_perp = geom['e_perp']
    
    d_x = cfg.v_parallel * t_hat_x - cfg.k_perp * e_perp * n_hat_x
    d_y = cfg.v_parallel * t_hat_y - cfg.k_perp * e_perp * n_hat_y
    
    d_mag = np.sqrt(d_x**2 + d_y**2) + eps
    return d_x / d_mag, d_y / d_mag


def generate_circle_waypoints(cfg: Config) -> List[Tuple[float, float]]:
    """Generate waypoints around the circle."""
    waypoints = []
    for i in range(cfg.n_waypoints):
        theta = cfg.theta0 + 2 * np.pi * i / cfg.n_waypoints
        if not cfg.ccw:
            theta = cfg.theta0 - 2 * np.pi * i / cfg.n_waypoints
        x = cfg.cx + cfg.R * np.cos(theta)
        y = cfg.cy + cfg.R * np.sin(theta)
        waypoints.append((x, y))
    return waypoints


def make_macro_action(action_type: MacroActionType4Puck, cfg: Config) -> MacroAction4Puck:
    """Create a macro action."""
    return MacroAction4Puck(
        action_type=action_type,
        magnitude=cfg.macro_magnitude,
        phase_step=cfg.macro_phase_step,
        amplitude_step=cfg.macro_amplitude_step,
    )


def evaluate_action_full(
    ev: Evaluator4Pucks,
    ctrl: Control4Pucks,
    action_type: MacroActionType4Puck,
    particle_x: float, particle_y: float,
    cfg: Config,
) -> Tuple[Control4Pucks, float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Full PDE evaluation of an action.
    
    Returns: (new_ctrl, Fx, Fy, U_field, Fx_field, Fy_field)
    """
    action = make_macro_action(action_type, cfg)
    new_ctrl = apply_macro_action_4puck(ctrl, action)
    new_ctrl = ev.clip_control(new_ctrl)
    
    vb = ev.control_to_forcing_band_vb(new_ctrl)
    field = ev.op.solve_for_bottom_vb(vb)
    U, Fx_field, Fy_field = gorkov_potential_and_force_2d(field, ev.particle)
    
    Fx_scaled = Fx_field * ev.cfg.alpha_g
    Fy_scaled = Fy_field * ev.cfg.alpha_g
    
    fx, fy = bilinear_sample_vec(field.x, field.y, Fx_scaled, Fy_scaled, particle_x, particle_y)
    
    return new_ctrl, float(fx), float(fy), U, Fx_scaled, Fy_scaled


def compute_action_score(
    Fx: float, Fy: float,
    d_hat_x: float, d_hat_y: float,
    action_type: MacroActionType4Puck,
    prev_action: Optional[MacroActionType4Puck],
    cfg: Config,
) -> Tuple[float, float, float, float]:
    """Compute score for an action."""
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


def integrate_particle(x: float, y: float, Fx: float, Fy: float, cfg: Config) -> Tuple[float, float]:
    """Overdamped particle integration."""
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


# =============================================================================
# Step Log
# =============================================================================

@dataclass
class StepLog:
    """Log for one step."""
    step_idx: int
    particle_x: float
    particle_y: float
    cross_track_error: float
    arc_progress: float
    chosen_action: str
    action_switched: bool
    Fp_x: float
    Fp_y: float
    Fp_mag: float
    score: float
    solver_time_ms: float
    pde_solves: int
    cache_hits: int
    target_idx: int


# =============================================================================
# GREEDY CONTROLLER (Baseline)
# =============================================================================

def run_greedy(
    ev: Evaluator4Pucks,
    x0: float, y0: float,
    ctrl0: Control4Pucks,
    waypoints: List[Tuple[float, float]],
    cfg: Config,
    verbose: bool = True,
) -> Tuple[List[StepLog], List[np.ndarray], List[Control4Pucks], TimingStats]:
    """Run path-tracking greedy controller."""
    action_set = get_action_set()
    
    step_logs = []
    U_fields = []
    controls = [ctrl0]
    stats = TimingStats()
    
    x, y = x0, y0
    ctrl = ctrl0
    target_idx = 0
    prev_action = None
    
    if verbose:
        print(f"\n   Running Path-Track Greedy: T={cfg.T} steps")
        print(f"   Actions: {len(action_set)}")
        print(f"   Progress: ", end="", flush=True)
    
    for t in range(cfg.T):
        d_hat_x, d_hat_y = get_desired_direction_pathtrack(x, y, cfg)
        
        best_score = -np.inf
        best_action_type = MacroActionType4Puck.HOLD
        best_ctrl = ctrl
        best_Fx, best_Fy = 0.0, 0.0
        best_U = None
        
        t0 = time.time()
        
        for action_type in action_set:
            new_ctrl, Fx, Fy, U, _, _ = evaluate_action_full(ev, ctrl, action_type, x, y, cfg)
            stats.pde_solves += 1
            
            score, _, _, _ = compute_action_score(Fx, Fy, d_hat_x, d_hat_y, action_type, prev_action, cfg)
            
            if score > best_score:
                best_score = score
                best_action_type = action_type
                best_ctrl = new_ctrl
                best_Fx, best_Fy = Fx, Fy
                best_U = U
        
        solver_time_ms = (time.time() - t0) * 1000
        stats.total_time_ms += solver_time_ms
        stats.total_steps += 1
        
        action_switched = prev_action is not None and best_action_type != prev_action
        prev_action = best_action_type
        ctrl = best_ctrl
        
        x_new, y_new = integrate_particle(x, y, best_Fx, best_Fy, cfg)
        
        geom = compute_circle_geometry(x_new, y_new, cfg)
        
        target_x, target_y = waypoints[target_idx]
        dist_to_target = np.sqrt((x_new - target_x)**2 + (y_new - target_y)**2)
        if dist_to_target < cfg.waypoint_tol and target_idx < len(waypoints) - 1:
            target_idx += 1
        
        gates = ""
        if ctrl.gateA: gates += "A"
        if ctrl.gateB: gates += "B"
        if ctrl.gateC: gates += "C"
        if ctrl.gateD: gates += "D"
        
        log = StepLog(
            step_idx=t,
            particle_x=x_new,
            particle_y=y_new,
            cross_track_error=geom['e_perp'],
            arc_progress=geom['s'],
            chosen_action=best_action_type.name,
            action_switched=action_switched,
            Fp_x=best_Fx,
            Fp_y=best_Fy,
            Fp_mag=np.sqrt(best_Fx**2 + best_Fy**2),
            score=best_score,
            solver_time_ms=solver_time_ms,
            pde_solves=len(action_set),
            cache_hits=0,
            target_idx=target_idx,
        )
        
        step_logs.append(log)
        U_fields.append(best_U)
        controls.append(ctrl)
        
        x, y = x_new, y_new
        
        if verbose and (t + 1) % 50 == 0:
            print(f"{t+1}", end=" ", flush=True)
    
    if verbose:
        print(f"\n   Completed in {stats.total_time_ms/1000:.1f}s")
        print(f"   PDE solves: {stats.pde_solves}")
    
    return step_logs, U_fields, controls, stats


# =============================================================================
# ORIGINAL MPC (No Optimizations)
# =============================================================================

def run_mpc_original(
    ev: Evaluator4Pucks,
    x0: float, y0: float,
    ctrl0: Control4Pucks,
    waypoints: List[Tuple[float, float]],
    cfg: Config,
    verbose: bool = True,
) -> Tuple[List[StepLog], List[np.ndarray], List[Control4Pucks], TimingStats]:
    """Run original MPC without optimizations (baseline for comparison)."""
    action_set = get_action_set()
    
    step_logs = []
    U_fields = []
    controls = [ctrl0]
    stats = TimingStats()
    
    x, y = x0, y0
    ctrl = ctrl0
    target_idx = 0
    prev_action = None
    
    if verbose:
        print(f"\n   Running Original MPC: T={cfg.T}, K={cfg.K}")
        print(f"   Actions: {len(action_set)}, Beam width: {cfg.n_top_actions}")
        print(f"   Progress: ", end="", flush=True)
    
    for t in range(cfg.T):
        d_hat_x, d_hat_y = get_desired_direction_pathtrack(x, y, cfg)
        
        t0 = time.time()
        
        # Original beam search (no caching, no pruning)
        K = cfg.K
        beam_width = cfg.n_top_actions
        
        # Initialize beam
        beam = []
        for action_type in action_set:
            new_ctrl, Fx, Fy, U, _, _ = evaluate_action_full(ev, ctrl, action_type, x, y, cfg)
            stats.pde_solves += 1
            
            score, _, _, _ = compute_action_score(Fx, Fy, d_hat_x, d_hat_y, action_type, prev_action, cfg)
            
            x_next, y_next = integrate_particle(x, y, Fx, Fy, cfg)
            beam.append((score, [action_type], x_next, y_next, new_ctrl, Fx, Fy, U))
        
        beam.sort(key=lambda b: -b[0])
        beam = beam[:beam_width]
        
        # Expand beam
        for depth in range(1, K):
            new_beam = []
            for parent_score, parent_seq, px, py, parent_ctrl, _, _, _ in beam:
                parent_d_hat_x, parent_d_hat_y = get_desired_direction_pathtrack(px, py, cfg)
                branch_prev = parent_seq[-1] if parent_seq else prev_action
                
                for action_type in action_set:
                    new_ctrl, Fx, Fy, U, _, _ = evaluate_action_full(ev, parent_ctrl, action_type, px, py, cfg)
                    stats.pde_solves += 1
                    
                    score, _, _, _ = compute_action_score(Fx, Fy, parent_d_hat_x, parent_d_hat_y, action_type, branch_prev, cfg)
                    
                    total_score = parent_score + (cfg.mpc_discount ** depth) * score
                    x_next, y_next = integrate_particle(px, py, Fx, Fy, cfg)
                    
                    new_beam.append((total_score, parent_seq + [action_type], x_next, y_next, new_ctrl, Fx, Fy, U))
            
            new_beam.sort(key=lambda b: -b[0])
            beam = new_beam[:beam_width]
        
        # Select best
        best_total_score, best_seq, _, _, _, _, _, _ = beam[0]
        best_action_type = best_seq[0]
        
        # Re-evaluate to get actual values
        best_ctrl, best_Fx, best_Fy, best_U, _, _ = evaluate_action_full(ev, ctrl, best_action_type, x, y, cfg)
        stats.pde_solves += 1
        best_score, _, _, Fp_mag = compute_action_score(best_Fx, best_Fy, d_hat_x, d_hat_y, best_action_type, prev_action, cfg)
        
        solver_time_ms = (time.time() - t0) * 1000
        stats.total_time_ms += solver_time_ms
        stats.total_steps += 1
        
        action_switched = prev_action is not None and best_action_type != prev_action
        prev_action = best_action_type
        ctrl = best_ctrl
        
        x_new, y_new = integrate_particle(x, y, best_Fx, best_Fy, cfg)
        
        geom = compute_circle_geometry(x_new, y_new, cfg)
        
        target_x, target_y = waypoints[target_idx]
        dist_to_target = np.sqrt((x_new - target_x)**2 + (y_new - target_y)**2)
        if dist_to_target < cfg.waypoint_tol and target_idx < len(waypoints) - 1:
            target_idx += 1
        
        log = StepLog(
            step_idx=t,
            particle_x=x_new,
            particle_y=y_new,
            cross_track_error=geom['e_perp'],
            arc_progress=geom['s'],
            chosen_action=best_action_type.name,
            action_switched=action_switched,
            Fp_x=best_Fx,
            Fp_y=best_Fy,
            Fp_mag=Fp_mag,
            score=best_score,
            solver_time_ms=solver_time_ms,
            pde_solves=stats.pde_solves,
            cache_hits=0,
            target_idx=target_idx,
        )
        
        step_logs.append(log)
        U_fields.append(best_U)
        controls.append(ctrl)
        
        x, y = x_new, y_new
        
        if verbose and (t + 1) % 50 == 0:
            print(f"{t+1}", end=" ", flush=True)
    
    if verbose:
        print(f"\n   Completed in {stats.total_time_ms/1000:.1f}s")
        print(f"   PDE solves: {stats.pde_solves}")
    
    return step_logs, U_fields, controls, stats


# =============================================================================
# OPTIMIZED MPC (Full Option B)
# =============================================================================

def run_mpc_optimized(
    ev: Evaluator4Pucks,
    x0: float, y0: float,
    ctrl0: Control4Pucks,
    waypoints: List[Tuple[float, float]],
    cfg: Config,
    verbose: bool = True,
    # Optimization flags
    use_cache: bool = True,
    use_pruning: bool = True,
    use_two_stage: bool = True,
) -> Tuple[List[StepLog], List[np.ndarray], List[Control4Pucks], TimingStats, Dict]:
    """
    Run optimized MPC with caching, pruning, and two-stage scoring.
    """
    action_set = get_action_set()
    
    step_logs = []
    U_fields = []
    controls = [ctrl0]
    stats = TimingStats()
    
    # Initialize optimization components
    cache = EvaluationCache(cfg.cache_precision) if use_cache else None
    heuristics = ActionHeuristics(cfg) if use_pruning else None
    two_stage = TwoStageScorer(cfg) if use_two_stage else None
    
    x, y = x0, y0
    ctrl = ctrl0
    target_idx = 0
    prev_action = None
    
    opt_str = []
    if use_cache: opt_str.append("Cache")
    if use_pruning: opt_str.append("Prune")
    if use_two_stage: opt_str.append("2Stage")
    opt_label = "+".join(opt_str) if opt_str else "None"
    
    if verbose:
        print(f"\n   Running Optimized MPC ({opt_label}): T={cfg.T}, K={cfg.K}")
        print(f"   Actions: {len(action_set)}, Beam width: {cfg.n_top_actions}")
        print(f"   Progress: ", end="", flush=True)
    
    for t in range(cfg.T):
        d_hat_x, d_hat_y = get_desired_direction_pathtrack(x, y, cfg)
        d_hat = (d_hat_x, d_hat_y)
        
        t0 = time.time()
        step_pde_solves = 0
        step_cache_hits = 0
        
        K = cfg.K
        beam_width = cfg.n_top_actions
        
        # =====================================================================
        # B2) ACTION PRUNING: Rank actions with cheap heuristics
        # =====================================================================
        if use_pruning and heuristics:
            t_heur = time.time()
            ranked_actions = heuristics.rank_actions(action_set, d_hat, prev_action)
            stats.heuristic_time_ms += (time.time() - t_heur) * 1000
            stats.heuristic_rankings += 1
            
            # Take only top N actions for initial evaluation
            actions_to_eval = [a for a, _ in ranked_actions[:cfg.n_initial_actions]]
        else:
            actions_to_eval = action_set
        
        # =====================================================================
        # B3) TWO-STAGE SCORING: Stage 1 (cheap approximation)
        # =====================================================================
        if use_two_stage and two_stage and two_stage.last_Fx is not None:
            t_s1 = time.time()
            stage1_scores = []
            
            for action_type in actions_to_eval:
                approx_score = two_stage.stage1_score(x, y, d_hat_x, d_hat_y, action_type, prev_action)
                stage1_scores.append((action_type, approx_score))
            
            stage1_scores.sort(key=lambda x: -x[1])
            actions_for_full_eval = [a for a, _ in stage1_scores[:cfg.stage2_candidates]]
            
            stats.stage1_time_ms += (time.time() - t_s1) * 1000
            stats.stage1_evals += len(actions_to_eval)
        else:
            actions_for_full_eval = actions_to_eval[:cfg.n_full_eval_actions]
        
        # =====================================================================
        # Initialize beam with full evaluation (with caching)
        # =====================================================================
        beam = []
        
        def evaluate_with_cache(ctrl_state, action, px, py):
            nonlocal step_pde_solves, step_cache_hits
            
            if use_cache and cache:
                # Include particle position in cache key
                cached = cache.get(ctrl_state, action, px, py)
                if cached is not None:
                    step_cache_hits += 1
                    stats.cache_lookups += 1
                    return cached
            
            # Full PDE solve
            t_pde = time.time()
            new_ctrl, Fx, Fy, U, Fx_field, Fy_field = evaluate_action_full(ev, ctrl_state, action, px, py, cfg)
            stats.pde_time_ms += (time.time() - t_pde) * 1000
            step_pde_solves += 1
            stats.pde_solves += 1
            
            result = (new_ctrl, Fx, Fy, U, Fx_field, Fy_field)
            
            if use_cache and cache:
                cache.put(ctrl_state, action, result, px, py)
            
            return result
        
        # Evaluate top actions for initial beam
        for action_type in actions_for_full_eval:
            new_ctrl, Fx, Fy, U, Fx_field, Fy_field = evaluate_with_cache(ctrl, action_type, x, y)
            
            score, _, _, _ = compute_action_score(Fx, Fy, d_hat_x, d_hat_y, action_type, prev_action, cfg)
            
            x_next, y_next = integrate_particle(x, y, Fx, Fy, cfg)
            beam.append((score, [action_type], x_next, y_next, new_ctrl, Fx, Fy, U, Fx_field, Fy_field))
            
            # Store field for two-stage scorer
            if use_two_stage and two_stage:
                two_stage.store_field(U, Fx_field, Fy_field, ev.op.x, ev.op.y)
        
        beam.sort(key=lambda b: -b[0])
        beam = beam[:beam_width]
        
        # =====================================================================
        # Expand beam for remaining horizon
        # =====================================================================
        for depth in range(1, K):
            new_beam = []
            
            for parent_score, parent_seq, px, py, parent_ctrl, _, _, _, parent_Fx_field, parent_Fy_field in beam:
                parent_d_hat_x, parent_d_hat_y = get_desired_direction_pathtrack(px, py, cfg)
                parent_d_hat = (parent_d_hat_x, parent_d_hat_y)
                branch_prev = parent_seq[-1] if parent_seq else prev_action
                
                # Pruning for expansion
                if use_pruning and heuristics:
                    ranked = heuristics.rank_actions(action_set, parent_d_hat, branch_prev)
                    expand_actions = [a for a, _ in ranked[:cfg.n_full_eval_actions]]
                else:
                    expand_actions = action_set[:cfg.n_full_eval_actions]
                
                for action_type in expand_actions:
                    new_ctrl, Fx, Fy, U, Fx_field, Fy_field = evaluate_with_cache(parent_ctrl, action_type, px, py)
                    
                    score, _, _, _ = compute_action_score(Fx, Fy, parent_d_hat_x, parent_d_hat_y, action_type, branch_prev, cfg)
                    
                    total_score = parent_score + (cfg.mpc_discount ** depth) * score
                    x_next, y_next = integrate_particle(px, py, Fx, Fy, cfg)
                    
                    new_beam.append((total_score, parent_seq + [action_type], x_next, y_next, new_ctrl, Fx, Fy, U, Fx_field, Fy_field))
            
            new_beam.sort(key=lambda b: -b[0])
            beam = new_beam[:beam_width]
        
        # =====================================================================
        # Select best action
        # =====================================================================
        best_total_score, best_seq, _, _, _, _, _, best_U, _, _ = beam[0]
        best_action_type = best_seq[0]
        
        # Get actual execution values
        best_ctrl, best_Fx, best_Fy, _, best_Fx_field, best_Fy_field = evaluate_with_cache(ctrl, best_action_type, x, y)
        best_score, _, _, Fp_mag = compute_action_score(best_Fx, best_Fy, d_hat_x, d_hat_y, best_action_type, prev_action, cfg)
        
        # Update heuristics
        if use_pruning and heuristics:
            heuristics.update_history(best_action_type, best_score)
            top_actions = [seq[0] for _, seq, _, _, _, _, _, _, _, _ in beam[:3]]
            heuristics.set_last_good_actions(top_actions)
        
        # Update two-stage scorer
        if use_two_stage and two_stage:
            two_stage.store_field(best_U, best_Fx_field, best_Fy_field, ev.op.x, ev.op.y)
        
        solver_time_ms = (time.time() - t0) * 1000
        stats.total_time_ms += solver_time_ms
        stats.total_steps += 1
        stats.stage2_evals += step_pde_solves
        
        action_switched = prev_action is not None and best_action_type != prev_action
        prev_action = best_action_type
        ctrl = best_ctrl
        
        x_new, y_new = integrate_particle(x, y, best_Fx, best_Fy, cfg)
        
        geom = compute_circle_geometry(x_new, y_new, cfg)
        
        target_x, target_y = waypoints[target_idx]
        dist_to_target = np.sqrt((x_new - target_x)**2 + (y_new - target_y)**2)
        if dist_to_target < cfg.waypoint_tol and target_idx < len(waypoints) - 1:
            target_idx += 1
        
        log = StepLog(
            step_idx=t,
            particle_x=x_new,
            particle_y=y_new,
            cross_track_error=geom['e_perp'],
            arc_progress=geom['s'],
            chosen_action=best_action_type.name,
            action_switched=action_switched,
            Fp_x=best_Fx,
            Fp_y=best_Fy,
            Fp_mag=Fp_mag,
            score=best_score,
            solver_time_ms=solver_time_ms,
            pde_solves=step_pde_solves,
            cache_hits=step_cache_hits,
            target_idx=target_idx,
        )
        
        step_logs.append(log)
        U_fields.append(best_U)
        controls.append(ctrl)
        
        x, y = x_new, y_new
        
        if verbose and (t + 1) % 50 == 0:
            print(f"{t+1}", end=" ", flush=True)
    
    cache_stats = cache.get_stats() if cache else {}
    
    if verbose:
        print(f"\n   Completed in {stats.total_time_ms/1000:.1f}s")
        print(f"   PDE solves: {stats.pde_solves}")
        if cache_stats:
            print(f"   Cache: {cache_stats['hits']} hits, {cache_stats['hit_rate']*100:.1f}% hit rate")
    
    return step_logs, U_fields, controls, stats, cache_stats


# =============================================================================
# Output Functions
# =============================================================================

def save_steps_csv(out_path: Path, logs: List[StepLog]):
    """Save step logs to CSV."""
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'step_idx', 'particle_x', 'particle_y', 'cross_track_error', 'arc_progress',
            'chosen_action', 'action_switched', 'Fp_x', 'Fp_y', 'Fp_mag', 'score',
            'solver_time_ms', 'pde_solves', 'cache_hits', 'target_idx'
        ])
        for log in logs:
            writer.writerow([
                log.step_idx, log.particle_x, log.particle_y,
                log.cross_track_error, log.arc_progress,
                log.chosen_action, log.action_switched,
                log.Fp_x, log.Fp_y, log.Fp_mag, log.score,
                log.solver_time_ms, log.pde_solves, log.cache_hits, log.target_idx
            ])
    print(f"   Saved: {out_path}")


def compute_summary(logs: List[StepLog], stats: TimingStats, cfg: Config, 
                    method: str, cache_stats: Dict = None) -> Dict[str, Any]:
    """Compute summary statistics."""
    if not logs:
        return {}
    
    cross_track_arr = np.array([log.cross_track_error for log in logs])
    arc_arr = np.array([log.arc_progress for log in logs])
    
    n_switches = sum(1 for log in logs if log.action_switched)
    
    circumference = 2 * np.pi * cfg.R
    circle_fraction = arc_arr[-1] / circumference if len(arc_arr) > 0 else 0.0
    
    summary = {
        'method': method,
        'n_steps': len(logs),
        'arc_progress_mm': float(arc_arr[-1] * 1e3) if len(arc_arr) > 0 else 0.0,
        'circle_fraction': float(circle_fraction),
        'mean_cross_track_error_um': float(np.mean(np.abs(cross_track_arr)) * 1e6),
        'max_cross_track_error_um': float(np.max(np.abs(cross_track_arr)) * 1e6),
        'std_cross_track_error_um': float(np.std(cross_track_arr) * 1e6),
        'n_action_switches': n_switches,
        'switch_rate': float(n_switches / len(logs)),
        'final_waypoint': logs[-1].target_idx,
        
        # Timing
        'total_time_s': float(stats.total_time_ms / 1000),
        'mean_time_per_step_ms': float(stats.total_time_ms / max(1, stats.total_steps)),
        'total_pde_solves': stats.pde_solves,
        'mean_pde_solves_per_step': float(stats.pde_solves / max(1, stats.total_steps)),
    }
    
    if cache_stats:
        summary['cache_hits'] = cache_stats.get('hits', 0)
        summary['cache_hit_rate'] = cache_stats.get('hit_rate', 0.0)
        summary['cache_size'] = cache_stats.get('cache_size', 0)
    
    return summary


def save_summary_json(out_path: Path, summary: Dict[str, Any], cfg: Config):
    """Save summary to JSON."""
    data = {
        'config': {
            'T': cfg.T,
            'K': cfg.K,
            'n_top_actions': cfg.n_top_actions,
            'Nx': cfg.Nx,
            'Ny': cfg.Ny,
            'n_initial_actions': cfg.n_initial_actions,
            'n_full_eval_actions': cfg.n_full_eval_actions,
            'stage2_candidates': cfg.stage2_candidates,
        },
        'summary': summary,
    }
    with open(out_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"   Saved: {out_path}")


def create_comparison_gif(
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
    
    all_U = np.concatenate([
        np.concatenate([U_fields_left[i].flatten() for i in indices]),
        np.concatenate([U_fields_right[i].flatten() for i in indices])
    ])
    U_lo, U_hi = np.nanpercentile(all_U, [2, 98])
    
    for frame_idx, t in enumerate(indices):
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6))
        
        for ax, logs, U_fields, controls, title in [
            (ax_left, logs_left, U_fields_left, controls_left, title_left),
            (ax_right, logs_right, U_fields_right, controls_right, title_right),
        ]:
            U = U_fields[t]
            log = logs[t]
            ctrl = controls[t]
            
            levels = np.linspace(U_lo, U_hi, 25)
            ax.contourf(X, Y, U, levels=levels, cmap="RdBu_r", alpha=0.85, extend='both')
            ax.contour(X, Y, U, levels=levels[::2], colors="k", linewidths=0.3, alpha=0.3)
            
            circle = Circle((cfg.cx * 1e3, cfg.cy * 1e3), cfg.R * 1e3,
                             fill=False, edgecolor='lime', linewidth=2, linestyle='--')
            ax.add_patch(circle)
            
            trail_indices = [i for i in indices if i <= t]
            if len(trail_indices) >= 2:
                tx = [logs[i].particle_x * 1e3 for i in trail_indices]
                ty = [logs[i].particle_y * 1e3 for i in trail_indices]
                ax.plot(tx, ty, 'c-', linewidth=2, alpha=0.7)
            
            ax.scatter(log.particle_x * 1e3, log.particle_y * 1e3, s=200, marker='o',
                       color='red', edgecolors='white', linewidth=2, zorder=100)
            
            ax.set_xlim(x_mm[0], x_mm[-1])
            ax.set_ylim(y_mm[0], y_mm[-1])
            ax.set_xlabel('x (mm)')
            ax.set_ylabel('y (mm)')
            ax.set_aspect('equal')
            
            subtitle = f"CTE={log.cross_track_error*1e6:.1f}µm, Arc={log.arc_progress*1e3:.2f}mm"
            ax.set_title(f"{title}\n{subtitle}", fontsize=10)
        
        fig.suptitle(f"Step {t}/{n_total-1}", fontsize=12, fontweight='bold')
        fig.tight_layout()
        
        frame_path = temp_dir / f"frame_{frame_idx:04d}.png"
        fig.savefig(frame_path, dpi=90)
        plt.close(fig)
        
        frames.append(imageio.imread(frame_path))
    
    imageio.mimsave(out_path, frames, duration=0.12)
    
    for f in temp_dir.glob("*.png"):
        f.unlink()
    temp_dir.rmdir()
    
    print(f"   Saved: {out_path}")


def create_timing_comparison_plot(out_path: Path, all_summaries: Dict[str, Dict], cfg: Config):
    """Create timing comparison plot."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    methods = list(all_summaries.keys())
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    # 1. Time per step
    ax1 = axes[0, 0]
    times = [all_summaries[m].get('mean_time_per_step_ms', 0) for m in methods]
    bars = ax1.bar(methods, times, color=colors[:len(methods)])
    ax1.set_ylabel('Mean Time per Step (ms)')
    ax1.set_title('Computational Cost')
    ax1.tick_params(axis='x', rotation=45)
    for bar, t in zip(bars, times):
        ax1.annotate(f'{t:.0f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=9)
    
    # 2. PDE solves
    ax2 = axes[0, 1]
    pde = [all_summaries[m].get('total_pde_solves', 0) for m in methods]
    bars = ax2.bar(methods, pde, color=colors[:len(methods)])
    ax2.set_ylabel('Total PDE Solves')
    ax2.set_title('PDE Evaluation Count')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Arc progress
    ax3 = axes[1, 0]
    arc = [all_summaries[m].get('arc_progress_mm', 0) for m in methods]
    bars = ax3.bar(methods, arc, color=colors[:len(methods)])
    ax3.set_ylabel('Arc Progress (mm)')
    ax3.set_title('Path Progress')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Cross-track error
    ax4 = axes[1, 1]
    cte = [all_summaries[m].get('mean_cross_track_error_um', 0) for m in methods]
    bars = ax4.bar(methods, cte, color=colors[:len(methods)])
    ax4.set_ylabel('Mean Cross-Track Error (µm)')
    ax4.set_title('Path Tracking Accuracy')
    ax4.tick_params(axis='x', rotation=45)
    
    fig.suptitle('MPC Optimization Comparison', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    
    print(f"   Saved: {out_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Optimized MPC comparison (Option B)")
    
    parser.add_argument("--Nx", type=int, default=80)
    parser.add_argument("--Ny", type=int, default=80)
    parser.add_argument("--K", type=int, default=3, help="MPC horizon")
    parser.add_argument("--T", type=int, default=300, help="Total steps")
    parser.add_argument("--beam_width", type=int, default=5, help="Beam width")
    
    # Optimization parameters
    parser.add_argument("--n_initial_actions", type=int, default=10)
    parser.add_argument("--n_full_eval_actions", type=int, default=6)
    parser.add_argument("--stage2_candidates", type=int, default=5)
    
    # Modes
    parser.add_argument("--fast", action="store_true", help="Quick test")
    parser.add_argument("--ablation", action="store_true", help="Run ablation study")
    parser.add_argument("--skip_original", action="store_true", help="Skip original MPC")
    
    parser.add_argument("--output_dir", type=str, default=None)
    
    args = parser.parse_args()
    
    cfg = Config(
        Nx=args.Nx,
        Ny=args.Ny,
        K=args.K,
        T=args.T,
        n_top_actions=args.beam_width,
        n_initial_actions=args.n_initial_actions,
        n_full_eval_actions=args.n_full_eval_actions,
        stage2_candidates=args.stage2_candidates,
    )
    
    if args.fast:
        cfg.T = 100
        cfg.K = 2
        cfg.Nx = 64
        cfg.Ny = 64
        cfg.n_top_actions = 3
    
    print("\n" + "="*70)
    print("  MPC COMPUTATIONAL OPTIMIZATION (Option B)")
    print("="*70)
    
    if args.fast:
        print("   [FAST MODE]")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = project_root / "results" / "optimized_mpc" / f"run_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n   Output: {out_dir}")
    print(f"   Config: T={cfg.T}, K={cfg.K}, beam_width={cfg.n_top_actions}")
    print(f"   Pruning: n_initial={cfg.n_initial_actions}, n_full={cfg.n_full_eval_actions}")
    
    print("\n   Building Evaluator4Pucks...")
    ev = create_evaluator(cfg)
    
    waypoints = generate_circle_waypoints(cfg)
    
    x0 = cfg.cx + cfg.R * np.cos(cfg.theta0)
    y0 = cfg.cy + cfg.R * np.sin(cfg.theta0)
    
    ctrl0 = Control4Pucks(
        xA=x0 - 0.4e-3, yA=0.03e-3, vA=0.08, phiA=0.0, gateA=True,
        xB=x0 + 0.4e-3, yB=0.03e-3, vB=0.08, phiB=np.pi, gateB=True,
        xC=x0, yC=0.20e-3, vC=0.08, phiC=np.pi/2, gateC=True,
        xD=x0, yD=1.8e-3, vD=0.05, phiD=-np.pi/2, gateD=True,
    )
    ctrl0 = ev.clip_control(ctrl0)
    
    print(f"\n   Initial position: ({x0*1e3:.3f}, {y0*1e3:.3f}) mm")
    
    all_logs = {}
    all_U_fields = {}
    all_controls = {}
    all_summaries = {}
    all_stats = {}
    
    # =========================================================================
    # Run Greedy (Baseline)
    # =========================================================================
    print("\n" + "-"*50)
    print("  Running Path-Track Greedy (Baseline)")
    print("-"*50)
    
    (out_dir / "greedy").mkdir(exist_ok=True)
    logs, U_fields, ctrls, stats = run_greedy(ev, x0, y0, ctrl0, waypoints, cfg)
    all_logs['greedy'] = logs
    all_U_fields['greedy'] = U_fields
    all_controls['greedy'] = ctrls
    all_stats['greedy'] = stats
    
    save_steps_csv(out_dir / "greedy" / "steps.csv", logs)
    summary = compute_summary(logs, stats, cfg, "greedy")
    all_summaries['greedy'] = summary
    save_summary_json(out_dir / "greedy" / "summary.json", summary, cfg)
    
    # =========================================================================
    # Run Original MPC (No optimizations)
    # =========================================================================
    if not args.skip_original:
        print("\n" + "-"*50)
        print("  Running Original MPC (No Optimizations)")
        print("-"*50)
        
        (out_dir / "mpc_original").mkdir(exist_ok=True)
        logs, U_fields, ctrls, stats = run_mpc_original(ev, x0, y0, ctrl0, waypoints, cfg)
        all_logs['mpc_original'] = logs
        all_U_fields['mpc_original'] = U_fields
        all_controls['mpc_original'] = ctrls
        all_stats['mpc_original'] = stats
        
        save_steps_csv(out_dir / "mpc_original" / "steps.csv", logs)
        summary = compute_summary(logs, stats, cfg, "mpc_original")
        all_summaries['mpc_original'] = summary
        save_summary_json(out_dir / "mpc_original" / "summary.json", summary, cfg)
    
    # =========================================================================
    # Ablation Study (if requested)
    # =========================================================================
    if args.ablation:
        ablation_configs = [
            ('mpc_cache_only', True, False, False),
            ('mpc_cache_prune', True, True, False),
        ]
        
        for name, use_cache, use_pruning, use_two_stage in ablation_configs:
            print("\n" + "-"*50)
            opt_str = []
            if use_cache: opt_str.append("Cache")
            if use_pruning: opt_str.append("Prune")
            if use_two_stage: opt_str.append("2Stage")
            print(f"  Running MPC ({'+'.join(opt_str) if opt_str else 'None'})")
            print("-"*50)
            
            (out_dir / name).mkdir(exist_ok=True)
            logs, U_fields, ctrls, stats, cache_stats = run_mpc_optimized(
                ev, x0, y0, ctrl0, waypoints, cfg,
                use_cache=use_cache, use_pruning=use_pruning, use_two_stage=use_two_stage
            )
            all_logs[name] = logs
            all_U_fields[name] = U_fields
            all_controls[name] = ctrls
            all_stats[name] = stats
            
            save_steps_csv(out_dir / name / "steps.csv", logs)
            summary = compute_summary(logs, stats, cfg, name, cache_stats)
            all_summaries[name] = summary
            save_summary_json(out_dir / name / "summary.json", summary, cfg)
    
    # =========================================================================
    # Run Full Optimized MPC
    # =========================================================================
    print("\n" + "-"*50)
    print("  Running Optimized MPC (Full Option B)")
    print("-"*50)
    
    (out_dir / "mpc_optimized").mkdir(exist_ok=True)
    logs, U_fields, ctrls, stats, cache_stats = run_mpc_optimized(
        ev, x0, y0, ctrl0, waypoints, cfg,
        use_cache=True, use_pruning=True, use_two_stage=True
    )
    all_logs['mpc_optimized'] = logs
    all_U_fields['mpc_optimized'] = U_fields
    all_controls['mpc_optimized'] = ctrls
    all_stats['mpc_optimized'] = stats
    
    save_steps_csv(out_dir / "mpc_optimized" / "steps.csv", logs)
    summary = compute_summary(logs, stats, cfg, "mpc_optimized", cache_stats)
    all_summaries['mpc_optimized'] = summary
    save_summary_json(out_dir / "mpc_optimized" / "summary.json", summary, cfg)
    
    # =========================================================================
    # Create Comparison Outputs
    # =========================================================================
    print("\n" + "-"*50)
    print("  Creating Comparison Outputs")
    print("-"*50)
    
    comp_dir = out_dir / "comparison"
    comp_dir.mkdir(exist_ok=True)
    
    # Save combined summary
    combined = {
        'config': {
            'T': cfg.T,
            'K': cfg.K,
            'n_top_actions': cfg.n_top_actions,
            'n_initial_actions': cfg.n_initial_actions,
            'n_full_eval_actions': cfg.n_full_eval_actions,
            'stage2_candidates': cfg.stage2_candidates,
        },
        'summaries': all_summaries,
    }
    
    with open(comp_dir / "compare_summary.json", 'w') as f:
        json.dump(combined, f, indent=2)
    print(f"   Saved: {comp_dir / 'compare_summary.json'}")
    
    # Timing comparison plot
    create_timing_comparison_plot(comp_dir / "timing_comparison.png", all_summaries, cfg)
    
    # Comparison GIF (Original vs Optimized)
    if 'mpc_original' in all_logs and 'mpc_optimized' in all_logs:
        print("\n   Creating comparison GIF...")
        create_comparison_gif(
            comp_dir / "mpc_comparison.gif",
            all_logs['mpc_original'], all_logs['mpc_optimized'],
            all_U_fields['mpc_original'], all_U_fields['mpc_optimized'],
            all_controls['mpc_original'], all_controls['mpc_optimized'],
            ev, cfg,
            "Original MPC", "Optimized MPC"
        )
    
    # =========================================================================
    # Print Summary
    # =========================================================================
    print("\n" + "="*70)
    print("  RESULTS SUMMARY")
    print("="*70)
    
    print("\n   {:<20} {:>10} {:>12} {:>10} {:>10}".format(
        "Method", "Time (s)", "PDE Solves", "Arc (mm)", "CTE (µm)"))
    print("   " + "-"*64)
    
    for name, summary in all_summaries.items():
        print("   {:<20} {:>10.1f} {:>12d} {:>10.2f} {:>10.1f}".format(
            name,
            summary['total_time_s'],
            summary['total_pde_solves'],
            summary['arc_progress_mm'],
            summary['mean_cross_track_error_um'],
        ))
    
    # Compute speedup
    if 'mpc_original' in all_summaries and 'mpc_optimized' in all_summaries:
        orig_time = all_summaries['mpc_original']['total_time_s']
        opt_time = all_summaries['mpc_optimized']['total_time_s']
        greedy_time = all_summaries['greedy']['total_time_s']
        
        speedup = orig_time / max(0.001, opt_time)
        vs_greedy = opt_time / max(0.001, greedy_time)
        
        print(f"\n   SPEEDUP: {speedup:.1f}× faster than original MPC")
        print(f"   Optimized MPC is {vs_greedy:.1f}× greedy time")
        
        orig_arc = all_summaries['mpc_original']['arc_progress_mm']
        opt_arc = all_summaries['mpc_optimized']['arc_progress_mm']
        if orig_arc > 0:
            arc_retention = opt_arc / orig_arc * 100
            print(f"   Arc progress retention: {arc_retention:.1f}%")
    
    print("\n" + "="*70)
    print("  COMPLETE")
    print("="*70)
    print(f"\n   All outputs saved to: {out_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
