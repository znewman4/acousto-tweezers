#!/usr/bin/env python3
"""
Greedy Surf Controller Demo: Truth-Model Baseline.

GREEDY SURFING BASELINE:
Moves the particle toward a target by selecting macro actions that maximize
surf force alignment with the desired direction.

This controller uses the full PDE solver as oracle - no GP/surrogate.
It works by evaluating ALL macro actions at each step and picking the one
that produces the best force alignment with the direction to target.

Key insight: This controller does NOT require trap_stable=True.
It "surfs" on the acoustic radiation force field, moving in the direction
of the force that best aligns with where we want to go.

Scoring function at each step:
    score = w_align * (F_hat · d_hat)    # unit alignment
          + w_push  * (F · d_hat)         # force magnitude in direction
          - w_switch * I[action != prev]  # switching penalty
          - w_step * action_step_norm     # action size penalty

Outputs:
    - results/demo_surf_greedy/run_YYYYMMDD_HHMMSS/
        - demo_surf_greedy.gif
        - summary.png
        - steps.csv

Usage:
    python scripts/demo_surf_greedy.py
    python scripts/demo_surf_greedy.py --fast
    python scripts/demo_surf_greedy.py --path circle
    python scripts/demo_surf_greedy.py --action_subset  # Use reduced action set
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
from typing import Optional
import shutil
import json
from scipy.interpolate import RegularGridInterpolator
from sklearn.ensemble import RandomForestRegressor

# Add project root to path for script imports
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from acousto.force import ParticleProps, gorkov_potential_and_force_2d, bilinear_sample_vec
from acousto.analysis import find_trap_center

from tweezers.control import (
    DishDomain, MediumProps, EvaluatorConfig,
    Control3Pucks,
)
from tweezers.control.evaluator_3pucks import Evaluator3Pucks

from scripts.macro_actions_3puck import (
    MacroAction3Puck,
    MacroActionType3Puck,
    apply_macro_action_3puck,
)


# =============================================================================
# Greedy Surf Controller
# =============================================================================

@dataclass
class GreedySurfConfig:
    """Configuration for greedy surf controller."""
    # Scoring weights
    w_align: float = 1.0       # Weight for unit alignment (F_hat · d_hat)
    w_push: float = 1e6        # Weight for force projection (F · d_hat) - scaled up for N-scale forces
    w_switch: float = 0.05     # Penalty for switching action
    w_step: float = 0.0        # Penalty for action step magnitude (disabled by default)
    
    # Force threshold - actions with force below this are heavily penalized
    min_force_threshold: float = 1e-12  # N
    
    # Macro action parameters
    macro_magnitude: float = 0.05e-3   # Position step size (meters)
    macro_phase_step: float = 0.15     # Phase step (radians)
    macro_amplitude_step: float = 0.01 # Amplitude step
    
    # Dynamics
    dt: float = 5e-3           # Integration timestep
    max_step: float = 0.08e-3  # Maximum particle displacement per step


@dataclass
class GreedySurfStep:
    """Log entry for one greedy surf step."""
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
    Fp_hat_dot_d: float  # Unit alignment
    Fp_dot_d: float      # Force projection
    
    score: float
    
    # Trap info (may be nan if no trap found)
    trap_candidate_x: float
    trap_candidate_y: float
    trap_stable: bool
    stiff_min: float
    
    solver_time_ms: float
    n_actions_evaluated: int
    
    # Target gating info
    target_idx: int = 0
    dist_to_target: float = 0.0
    target_advanced: bool = False
    
    # Circle-specific metrics
    cross_track_error: float = 0.0  # Distance from circle
    tangential_alignment: float = 0.0  # Alignment with tangent direction


class GreedySurfController:
    """
    Greedy controller that selects best macro action based on force alignment.
    
    At each step:
    1. Compute desired direction toward target: d_hat = (target - particle) / |...|
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
        evaluator: Evaluator3Pucks,
        config: GreedySurfConfig,
        action_set: list[MacroActionType3Puck] | None = None,
    ):
        self.ev = evaluator
        self.config = config
        
        # Default action set (core directional + interference control)
        if action_set is None:
            action_set = [
                MacroActionType3Puck.HOLD,
                MacroActionType3Puck.TRANSLATE_TRAP_X_POS,
                MacroActionType3Puck.TRANSLATE_TRAP_X_NEG,
                MacroActionType3Puck.TRANSLATE_TRAP_Y_POS,
                MacroActionType3Puck.TRANSLATE_TRAP_Y_NEG,
                MacroActionType3Puck.ROTATE_INTERFERENCE_CW,
                MacroActionType3Puck.ROTATE_INTERFERENCE_CCW,
                MacroActionType3Puck.MOVE_A_RIGHT,
                MacroActionType3Puck.MOVE_A_LEFT,
                MacroActionType3Puck.MOVE_B_RIGHT,
                MacroActionType3Puck.MOVE_B_LEFT,
                MacroActionType3Puck.PHASE_SHIFT_B_POS,
                MacroActionType3Puck.PHASE_SHIFT_B_NEG,
            ]
        
        self.action_types = action_set
        self.prev_action: MacroActionType3Puck | None = None
        # Store best field for visualization (set during step())
        self.best_field: tuple | None = None  # (field, U, Fx_scaled, Fy_scaled)
        
    def _make_action(self, action_type: MacroActionType3Puck) -> MacroAction3Puck:
        """Create MacroAction3Puck from type with current config parameters."""
        return MacroAction3Puck(
            action_type=action_type,
            magnitude=self.config.macro_magnitude,
            phase_step=self.config.macro_phase_step,
            amplitude_step=self.config.macro_amplitude_step,
        )
    
    def _compute_action_score(
        self,
        action_type: MacroActionType3Puck,
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
        # Use provided weights or fall back to config
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
        Fp_hat_dot_d = Fp_hat_x * d_hat_x + Fp_hat_y * d_hat_y  # Unit alignment [-1, 1]
        Fp_dot_d = Fp_x * d_hat_x + Fp_y * d_hat_y  # Force projection
        
        # Build score: combine alignment and push
        # The key insight: we want actions that produce BOTH good alignment AND significant force
        # Score = alignment_term + push_term
        # - alignment_term rewards pointing in the right direction
        # - push_term rewards actually producing force in that direction
        score = (
            w_align * Fp_hat_dot_d
            + w_push * Fp_dot_d  # w_push is large (1e6) to scale N-level forces
        )
        
        # Penalize actions with near-zero force (alignment is meaningless if force is tiny)
        if Fp_mag < self.config.min_force_threshold:
            score -= 0.5  # Penalty for producing essentially no force
        
        # Switching penalty
        if self.prev_action is not None and action_type != self.prev_action:
            score -= w_switch
        
        # Action step penalty (could use action magnitude here)
        # Currently disabled (w_step = 0)
        
        return score, Fp_hat_dot_d, Fp_dot_d, Fp_mag
    
    def step(
        self,
        particle_x: float,
        particle_y: float,
        target_x: float,
        target_y: float,
        ctrl: Control3Pucks,
        step_idx: int = 0,
        desired_direction: tuple[float, float] | None = None,  # Override d_hat if provided
        weight_overrides: dict | None = None,  # Temporary weight overrides for stuck escape
    ) -> tuple[Control3Pucks, float, float, GreedySurfStep]:
        """
        Execute one greedy surf control step.
        
        Args:
            desired_direction: If provided, use this (d_hat_x, d_hat_y) instead of computing from target.
            weight_overrides: Dict with optional 'w_switch', 'w_push' overrides for this step.
        
        Returns:
            new_ctrl: Updated control configuration
            new_x, new_y: New particle position after integration
            log: GreedySurfStep with all diagnostic info
        """
        total_solver_time = 0.0
        
        # Get scoring weights (with optional overrides)
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
        best_action_type = MacroActionType3Puck.HOLD
        best_Fp = (0.0, 0.0)
        best_metrics = (0.0, 0.0, 0.0)  # (Fp_hat_dot_d, Fp_dot_d, Fp_mag)
        best_field = None
        best_trap_info = (np.nan, np.nan, False, np.nan)  # (tx, ty, stable, stiff_min)
        
        for action_type in self.action_types:
            # Clone and apply action
            action = self._make_action(action_type)
            u_cand = apply_macro_action_3puck(ctrl, action)
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
            
            # Compute score (with possible weight overrides)
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
                
                # Get trap info for best action (for logging)
                trap = find_trap_center(
                    field.x, field.y, U, Fx, Fy,
                    particle_x=particle_x, particle_y=particle_y,
                    search_radius=0.4e-3,
                )
                stiff_min = np.nan
                if trap.stiffness_eigvals is not None:
                    stiff_min = float(np.min(trap.stiffness_eigvals))
                best_trap_info = (
                    trap.x if trap.x is not None else np.nan,
                    trap.y if trap.y is not None else np.nan,
                    trap.is_stable,
                    stiff_min,
                )
        
        # Store best field for visualization
        self.best_field = best_field

        # Apply best action
        action_switched = (self.prev_action is not None and best_action_type != self.prev_action)
        self.prev_action = best_action_type
        
        best_action = self._make_action(best_action_type)
        new_ctrl = apply_macro_action_3puck(ctrl, best_action)
        new_ctrl = self.ev.clip_control(new_ctrl)
        
        # Integrate particle using force from best action
        fx, fy = best_Fp
        a = float(self.ev.particle.a)
        gamma = 6.0 * np.pi * float(self.ev.cfg.viscosity) * a
        
        dx_raw = self.config.dt * (fx / gamma)
        dy_raw = self.config.dt * (fy / gamma)
        raw_disp = np.sqrt(dx_raw**2 + dy_raw**2)
        
        # Step limiting
        if raw_disp > self.config.max_step and raw_disp > 0:
            scale = self.config.max_step / raw_disp
            dx_raw *= scale
            dy_raw *= scale
        
        new_x = particle_x + dx_raw
        new_y = particle_y + dy_raw
        
        # Clip to domain
        new_x = float(np.clip(new_x, self.ev.op.x[0], self.ev.op.x[-1]))
        new_y = float(np.clip(new_y, self.ev.op.y[0], self.ev.op.y[-1]))
        
        # Compute tracking error
        tracking_error = np.sqrt((new_x - target_x)**2 + (new_y - target_y)**2)
        
        # Build log entry
        Fp_hat_dot_d, Fp_dot_d, Fp_mag = best_metrics
        trap_x, trap_y, trap_stable, stiff_min = best_trap_info
        
        log = GreedySurfStep(
            step_idx=step_idx,
            particle_x=float(new_x),
            particle_y=float(new_y),
            target_x=target_x,
            target_y=target_y,
            tracking_error=tracking_error,
            chosen_action=best_action_type.name,
            action_switched=action_switched,
            Fp_x=float(fx),
            Fp_y=float(fy),
            Fp_mag=Fp_mag,
            Fp_hat_dot_d=Fp_hat_dot_d,
            Fp_dot_d=Fp_dot_d,
            score=best_score,
            trap_candidate_x=trap_x,
            trap_candidate_y=trap_y,
            trap_stable=trap_stable,
            stiff_min=stiff_min,
            solver_time_ms=total_solver_time,
            n_actions_evaluated=len(self.action_types),
        )
        
        return new_ctrl, new_x, new_y, log


# =============================================================================
# Bayesian Surrogate for Action Selection
# =============================================================================

class BayesSurrogate:
    """
    Per-action surrogate models using Random Forest for UCB-based action selection.
    
    Features φ(s):
        - particle_x, particle_y
        - d_hat_x, d_hat_y (desired direction)
        - cross_track_error (for circles)
        - prev_action_id
    
    Each action has its own RF model trained on (φ(s), score) pairs.
    """
    
    def __init__(
        self,
        action_names: list[str],
        n_estimators: int = 20,
        min_samples_for_fit: int = 5,
        default_sigma: float = 1.0,
    ):
        self.action_names = action_names
        self.action_to_idx = {name: i for i, name in enumerate(action_names)}
        self.n_actions = len(action_names)
        self.n_estimators = n_estimators
        self.min_samples_for_fit = min_samples_for_fit
        self.default_sigma = default_sigma
        
        # Per-action datasets: X[action_name] -> list of feature vectors
        self.X: dict[str, list[np.ndarray]] = {name: [] for name in action_names}
        self.y: dict[str, list[float]] = {name: [] for name in action_names}
        
        # Per-action models
        self.models: dict[str, RandomForestRegressor | None] = {name: None for name in action_names}
        
    def get_features(
        self,
        particle_x: float,
        particle_y: float,
        d_hat_x: float,
        d_hat_y: float,
        cross_track_error: float,
        prev_action_id: int,
    ) -> np.ndarray:
        """Construct feature vector from state."""
        return np.array([
            particle_x * 1e3,  # Scale to mm for better numerics
            particle_y * 1e3,
            d_hat_x,
            d_hat_y,
            cross_track_error * 1e3,  # Scale to mm
            prev_action_id / max(1, self.n_actions - 1),  # Normalize to [0, 1]
        ])
    
    def add_observation(self, action_name: str, features: np.ndarray, score: float):
        """Add an observation to the dataset for a specific action."""
        self.X[action_name].append(features.copy())
        self.y[action_name].append(score)
        
        # Refit model if we have enough data
        if len(self.X[action_name]) >= self.min_samples_for_fit:
            X_arr = np.array(self.X[action_name])
            y_arr = np.array(self.y[action_name])
            
            rf = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=1,
            )
            rf.fit(X_arr, y_arr)
            self.models[action_name] = rf
    
    def predict(self, action_name: str, features: np.ndarray) -> tuple[float, float]:
        """
        Predict mean and std for a given action and state features.
        
        Returns: (mu, sigma)
        """
        model = self.models[action_name]
        
        if model is None:
            # No model yet - return default (explore)
            return 0.0, self.default_sigma
        
        # Get predictions from each tree in the forest
        X_query = features.reshape(1, -1)
        tree_preds = np.array([tree.predict(X_query)[0] for tree in model.estimators_])
        
        mu = float(np.mean(tree_preds))
        sigma = float(np.std(tree_preds))
        
        # Ensure minimum exploration
        sigma = max(sigma, 0.01)
        
        return mu, sigma
    
    def get_ucb_scores(
        self,
        features: np.ndarray,
        kappa: float = 1.0,
    ) -> dict[str, tuple[float, float, float]]:
        """
        Compute UCB scores for all actions.
        
        Returns: {action_name: (mu, sigma, ucb)}
        """
        result = {}
        for action_name in self.action_names:
            mu, sigma = self.predict(action_name, features)
            ucb = mu + kappa * sigma
            result[action_name] = (mu, sigma, ucb)
        return result
    
    def get_dataset_sizes(self) -> dict[str, int]:
        """Return the number of observations for each action."""
        return {name: len(self.X[name]) for name in self.action_names}


@dataclass
class BayesSurfStep(GreedySurfStep):
    """Extended log entry for Bayesian surf controller."""
    controller_mode: str = "bayes"
    n_actions_total: int = 0
    chosen_action_rank_ucb: int = 0  # Rank of chosen action by UCB (1-indexed, 1=highest UCB)
    chosen_action_mu: float = 0.0
    chosen_action_sigma: float = 0.0
    chosen_action_ucb: float = 0.0
    # Robustness mechanism tracking
    robustness_mode: str = "none"  # "none", "diversity", "rewarmup"
    consecutive_same_action: int = 0


class BayesGreedySurfController:
    """
    Bayesian controller that uses surrogate models to select a subset of actions to evaluate.
    
    At each step:
    1. During warmup (first N steps): evaluate ALL actions (like greedy)
    2. After warmup:
       a. Compute UCB scores for all actions using surrogate models
       b. Select top K actions by UCB (always including prev_action)
       c. Evaluate only those K actions with PDE solves
       d. Pick best action by true score among evaluated candidates
    3. Update surrogate models with new observations
    
    Robustness mechanisms:
    - Forced diversity: Cannot pick same action > max_repeat times unless score improves
    - Stuck-triggered re-warmup: Temporarily revert to greedy when progress stalls
    - High-authority inclusion: Always include at least one field-shaping action
    """
    
    # Define high-authority "field-shaping" actions that should always be considered
    HIGH_AUTHORITY_ACTIONS = {
        MacroActionType3Puck.ROTATE_INTERFERENCE_CW,
        MacroActionType3Puck.ROTATE_INTERFERENCE_CCW,
        MacroActionType3Puck.MOVE_B_LEFT,
        MacroActionType3Puck.MOVE_C_DOWN,
        MacroActionType3Puck.TRANSLATE_TRAP_X_POS,
    }
    
    def __init__(
        self,
        evaluator: Evaluator3Pucks,
        config: GreedySurfConfig,
        action_set: list[MacroActionType3Puck] | None = None,
        bayes_k: int = 3,
        bayes_kappa: float = 1.0,
        bayes_warmup_steps: int = 10,
        n_estimators: int = 20,
        max_repeat: int = 5,
        stuck_rewarmup_steps: int = 5,
    ):
        self.ev = evaluator
        self.config = config
        self.bayes_k = bayes_k
        self.bayes_kappa = bayes_kappa
        self.bayes_warmup_steps = bayes_warmup_steps
        self.max_repeat = max_repeat
        self.stuck_rewarmup_steps = stuck_rewarmup_steps
        
        # Default action set (same as greedy)
        if action_set is None:
            action_set = [
                MacroActionType3Puck.HOLD,
                MacroActionType3Puck.TRANSLATE_TRAP_X_POS,
                MacroActionType3Puck.TRANSLATE_TRAP_X_NEG,
                MacroActionType3Puck.TRANSLATE_TRAP_Y_POS,
                MacroActionType3Puck.TRANSLATE_TRAP_Y_NEG,
                MacroActionType3Puck.ROTATE_INTERFERENCE_CW,
                MacroActionType3Puck.ROTATE_INTERFERENCE_CCW,
                MacroActionType3Puck.MOVE_A_RIGHT,
                MacroActionType3Puck.MOVE_A_LEFT,
                MacroActionType3Puck.MOVE_B_RIGHT,
                MacroActionType3Puck.MOVE_B_LEFT,
                MacroActionType3Puck.PHASE_SHIFT_B_POS,
                MacroActionType3Puck.PHASE_SHIFT_B_NEG,
            ]
        
        self.action_types = action_set
        self.action_names = [a.name for a in action_set]
        self.prev_action: MacroActionType3Puck | None = None
        self.prev_action_id: int = 0
        self.step_count: int = 0
        
        # Robustness state
        self.consecutive_same_action: int = 0
        self.last_best_score: float = -np.inf
        self.rewarmup_steps_remaining: int = 0
        self.diversity_trigger_count: int = 0
        self.rewarmup_trigger_count: int = 0
        
        # Identify high-authority actions in the current action set
        self.high_authority_in_set = [
            a for a in action_set if a in self.HIGH_AUTHORITY_ACTIONS
        ]
        
        # Surrogate models
        self.surrogate = BayesSurrogate(
            action_names=self.action_names,
            n_estimators=n_estimators,
        )
        
        # Store best field for visualization
        self.best_field: tuple | None = None
        
    def _make_action(self, action_type: MacroActionType3Puck) -> MacroAction3Puck:
        """Create MacroAction3Puck from type with current config parameters."""
        return MacroAction3Puck(
            action_type=action_type,
            magnitude=self.config.macro_magnitude,
            phase_step=self.config.macro_phase_step,
            amplitude_step=self.config.macro_amplitude_step,
        )
    
    def _compute_action_score(
        self,
        action_type: MacroActionType3Puck,
        Fp_x: float,
        Fp_y: float,
        d_hat_x: float,
        d_hat_y: float,
        w_align: float | None = None,
        w_push: float | None = None,
        w_switch: float | None = None,
    ) -> tuple[float, float, float, float]:
        """Compute score for an action (same as greedy)."""
        if w_align is None:
            w_align = self.config.w_align
        if w_push is None:
            w_push = self.config.w_push
        if w_switch is None:
            w_switch = self.config.w_switch
            
        Fp_mag = np.sqrt(Fp_x**2 + Fp_y**2)
        eps = 1e-15
        
        Fp_hat_x = Fp_x / (Fp_mag + eps)
        Fp_hat_y = Fp_y / (Fp_mag + eps)
        
        Fp_hat_dot_d = Fp_hat_x * d_hat_x + Fp_hat_y * d_hat_y
        Fp_dot_d = Fp_x * d_hat_x + Fp_y * d_hat_y
        
        score = w_align * Fp_hat_dot_d + w_push * Fp_dot_d
        
        if Fp_mag < self.config.min_force_threshold:
            score -= 0.5
        
        if self.prev_action is not None and action_type != self.prev_action:
            score -= w_switch
        
        return score, Fp_hat_dot_d, Fp_dot_d, Fp_mag
    
    def _evaluate_action(
        self,
        action_type: MacroActionType3Puck,
        ctrl: Control3Pucks,
        particle_x: float,
        particle_y: float,
        d_hat_x: float,
        d_hat_y: float,
        w_align: float,
        w_push: float,
        w_switch: float,
    ) -> tuple[float, float, float, float, float, tuple]:
        """
        Evaluate a single action with PDE solve.
        
        Returns: (score, Fp_hat_dot_d, Fp_dot_d, Fp_mag, solver_time_ms, field_data)
        """
        action = self._make_action(action_type)
        u_cand = apply_macro_action_3puck(ctrl, action)
        u_cand = self.ev.clip_control(u_cand)
        
        t0 = time.perf_counter()
        vb = self.ev.control_to_forcing_band_vb(u_cand)
        field = self.ev.op.solve_for_bottom_vb(vb)
        U, Fx, Fy = gorkov_potential_and_force_2d(field, self.ev.particle)
        
        Fx_scaled = Fx * self.ev.cfg.alpha_g
        Fy_scaled = Fy * self.ev.cfg.alpha_g
        
        solver_time_ms = (time.perf_counter() - t0) * 1000.0
        
        fx, fy = bilinear_sample_vec(field.x, field.y, Fx_scaled, Fy_scaled, particle_x, particle_y)
        
        score, Fp_hat_dot_d, Fp_dot_d, Fp_mag = self._compute_action_score(
            action_type, fx, fy, d_hat_x, d_hat_y,
            w_align=w_align, w_push=w_push, w_switch=w_switch,
        )
        
        field_data = (field, U, Fx_scaled, Fy_scaled, fx, fy)
        
        return score, Fp_hat_dot_d, Fp_dot_d, Fp_mag, solver_time_ms, field_data
    
    def step(
        self,
        particle_x: float,
        particle_y: float,
        target_x: float,
        target_y: float,
        ctrl: Control3Pucks,
        step_idx: int = 0,
        desired_direction: tuple[float, float] | None = None,
        weight_overrides: dict | None = None,
        cross_track_error: float = 0.0,
    ) -> tuple[Control3Pucks, float, float, BayesSurfStep]:
        """Execute one Bayesian surf control step."""
        total_solver_time = 0.0
        self.step_count += 1
        
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
        
        # Build feature vector for surrogate
        features = self.surrogate.get_features(
            particle_x, particle_y,
            d_hat_x, d_hat_y,
            cross_track_error,
            self.prev_action_id,
        )
        
        # === Robustness: Check for stuck-triggered re-warmup ===
        in_rewarmup = (self.rewarmup_steps_remaining > 0)
        if in_rewarmup:
            self.rewarmup_steps_remaining -= 1
        
        # Determine which actions to evaluate
        in_warmup = (self.step_count <= self.bayes_warmup_steps)
        
        # Track whether robustness mechanism triggered this step
        robustness_mode = "none"
        
        if in_warmup or in_rewarmup:
            # Warmup or re-warmup: evaluate all actions
            actions_to_eval = list(self.action_types)
            if in_rewarmup:
                robustness_mode = "rewarmup"
        else:
            # Bayes mode: use UCB to select candidates
            ucb_scores = self.surrogate.get_ucb_scores(features, kappa=self.bayes_kappa)
            
            # Sort actions by UCB (descending)
            sorted_actions = sorted(
                self.action_types,
                key=lambda a: ucb_scores[a.name][2],  # UCB value
                reverse=True,
            )
            
            # Build candidate set: always include prev_action + top (K-1) by UCB
            actions_to_eval = []
            
            # Add previous action first (if we have one and not locked by diversity)
            # === Robustness: Forced diversity - don't include prev_action if repeated too many times ===
            include_prev = (
                self.prev_action is not None 
                and self.prev_action in self.action_types
                and self.consecutive_same_action < self.max_repeat
            )
            if include_prev:
                actions_to_eval.append(self.prev_action)
            elif self.consecutive_same_action >= self.max_repeat:
                robustness_mode = "diversity"
                self.diversity_trigger_count += 1
            
            # Add top UCB actions (excluding those already added)
            for action in sorted_actions:
                if action not in actions_to_eval:
                    actions_to_eval.append(action)
                if len(actions_to_eval) >= self.bayes_k:
                    break
            
            # === Robustness: Ensure at least one high-authority action is in candidate set ===
            has_high_authority = any(a in self.high_authority_in_set for a in actions_to_eval)
            if not has_high_authority and self.high_authority_in_set:
                # Add the top-UCB high-authority action
                for action in sorted_actions:
                    if action in self.high_authority_in_set and action not in actions_to_eval:
                        actions_to_eval.append(action)
                        break
        
        # Evaluate selected actions with PDE solves
        best_score = -np.inf
        best_action_type = MacroActionType3Puck.HOLD
        best_Fp = (0.0, 0.0)
        best_metrics = (0.0, 0.0, 0.0)
        best_field = None
        best_trap_info = (np.nan, np.nan, False, np.nan)
        
        evaluated_results: dict[str, tuple] = {}
        
        for action_type in actions_to_eval:
            score, Fp_hat_dot_d, Fp_dot_d, Fp_mag, solver_ms, field_data = self._evaluate_action(
                action_type, ctrl, particle_x, particle_y,
                d_hat_x, d_hat_y, w_align, w_push, w_switch,
            )
            
            total_solver_time += solver_ms
            field, U, Fx_scaled, Fy_scaled, fx, fy = field_data
            
            # Store result for surrogate update
            evaluated_results[action_type.name] = (score, fx, fy)
            
            # Update surrogate with this observation
            self.surrogate.add_observation(action_type.name, features, score)
            
            if score > best_score:
                best_score = score
                best_action_type = action_type
                best_Fp = (fx, fy)
                best_metrics = (Fp_hat_dot_d, Fp_dot_d, Fp_mag)
                best_field = (field, U, Fx_scaled, Fy_scaled)
                
                # Get trap info
                trap = find_trap_center(
                    field.x, field.y, U, Fx_scaled / self.ev.cfg.alpha_g, Fy_scaled / self.ev.cfg.alpha_g,
                    particle_x=particle_x, particle_y=particle_y,
                    search_radius=0.4e-3,
                )
                stiff_min = np.nan
                if trap.stiffness_eigvals is not None:
                    stiff_min = float(np.min(trap.stiffness_eigvals))
                best_trap_info = (
                    trap.x if trap.x is not None else np.nan,
                    trap.y if trap.y is not None else np.nan,
                    trap.is_stable,
                    stiff_min,
                )
        
        # Store best field for visualization
        self.best_field = best_field
        
        # Compute UCB rank of chosen action (for logging)
        if not in_warmup and not in_rewarmup:
            ucb_list = [(a.name, ucb_scores[a.name][2]) for a in self.action_types]
            ucb_list_sorted = sorted(ucb_list, key=lambda x: x[1], reverse=True)
            chosen_ucb_rank = next(
                (i + 1 for i, (name, _) in enumerate(ucb_list_sorted) if name == best_action_type.name),
                len(self.action_types)
            )
            chosen_mu, chosen_sigma, chosen_ucb = ucb_scores[best_action_type.name]
        else:
            chosen_ucb_rank = 0
            chosen_mu, chosen_sigma, chosen_ucb = 0.0, 0.0, 0.0
        
        # === Robustness: Update consecutive action counter ===
        action_switched = (self.prev_action is not None and best_action_type != self.prev_action)
        if action_switched:
            self.consecutive_same_action = 1
        else:
            self.consecutive_same_action += 1
        
        # === Robustness: Stuck detection - trigger re-warmup if score not improving ===
        score_improved = (best_score > self.last_best_score + 1e-10)
        if not score_improved and self.consecutive_same_action >= self.max_repeat:
            # Score is not improving and we're locked on same action -> trigger re-warmup
            self.rewarmup_steps_remaining = self.stuck_rewarmup_steps
            self.rewarmup_trigger_count += 1
        self.last_best_score = best_score
        
        # Apply best action
        self.prev_action = best_action_type
        self.prev_action_id = self.action_types.index(best_action_type)
        
        best_action = self._make_action(best_action_type)
        new_ctrl = apply_macro_action_3puck(ctrl, best_action)
        new_ctrl = self.ev.clip_control(new_ctrl)
        
        # Integrate particle
        fx, fy = best_Fp
        a = float(self.ev.particle.a)
        gamma = 6.0 * np.pi * float(self.ev.cfg.viscosity) * a
        
        dx_raw = self.config.dt * (fx / gamma)
        dy_raw = self.config.dt * (fy / gamma)
        raw_disp = np.sqrt(dx_raw**2 + dy_raw**2)
        
        if raw_disp > self.config.max_step and raw_disp > 0:
            scale = self.config.max_step / raw_disp
            dx_raw *= scale
            dy_raw *= scale
        
        new_x = particle_x + dx_raw
        new_y = particle_y + dy_raw
        
        new_x = float(np.clip(new_x, self.ev.op.x[0], self.ev.op.x[-1]))
        new_y = float(np.clip(new_y, self.ev.op.y[0], self.ev.op.y[-1]))
        
        tracking_error = np.sqrt((new_x - target_x)**2 + (new_y - target_y)**2)
        
        Fp_hat_dot_d, Fp_dot_d, Fp_mag = best_metrics
        trap_x, trap_y, trap_stable, stiff_min = best_trap_info
        
        log = BayesSurfStep(
            step_idx=step_idx,
            particle_x=float(new_x),
            particle_y=float(new_y),
            target_x=target_x,
            target_y=target_y,
            tracking_error=tracking_error,
            chosen_action=best_action_type.name,
            action_switched=action_switched,
            Fp_x=float(fx),
            Fp_y=float(fy),
            Fp_mag=Fp_mag,
            Fp_hat_dot_d=Fp_hat_dot_d,
            Fp_dot_d=Fp_dot_d,
            score=best_score,
            trap_candidate_x=trap_x,
            trap_candidate_y=trap_y,
            trap_stable=trap_stable,
            stiff_min=stiff_min,
            solver_time_ms=total_solver_time,
            n_actions_evaluated=len(actions_to_eval),
            # Bayes-specific fields
            controller_mode="rewarmup" if in_rewarmup else ("warmup" if in_warmup else "bayes"),
            n_actions_total=len(self.action_types),
            chosen_action_rank_ucb=chosen_ucb_rank,
            chosen_action_mu=chosen_mu,
            chosen_action_sigma=chosen_sigma,
            chosen_action_ucb=chosen_ucb,
            # Robustness tracking
            robustness_mode=robustness_mode,
            consecutive_same_action=self.consecutive_same_action,
        )
        
        return new_ctrl, new_x, new_y, log


# =============================================================================
# Path Generators
# =============================================================================

def make_straight_line_path(
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    n_points: int,
) -> np.ndarray:
    """Create straight line path."""
    t = np.linspace(0, 1, n_points)
    x = start_x + t * (end_x - start_x)
    y = start_y + t * (end_y - start_y)
    return np.column_stack([x, y])


def make_circle_path(
    center_x: float,
    center_y: float,
    radius: float,
    n_points: int,
    n_loops: int = 1,
) -> np.ndarray:
    """Create circular path."""
    theta = np.linspace(0, 2 * np.pi * n_loops, n_points)
    x = center_x + radius * np.cos(theta)
    y = center_y + radius * np.sin(theta)
    return np.column_stack([x, y])


def compute_circle_direction(
    particle_x: float,
    particle_y: float,
    center_x: float,
    center_y: float,
    radius: float,
    k_radial: float = 2.0,
    ccw: bool = True,
) -> tuple[float, float, float]:
    """
    Compute desired direction for circle following using tangent + radial correction.
    
    Args:
        particle_x, particle_y: Current particle position
        center_x, center_y: Circle center
        radius: Circle radius
        k_radial: Radial correction gain (higher = more aggressive return to circle)
        ccw: Counter-clockwise direction (True) or clockwise (False)
    
    Returns:
        (d_hat_x, d_hat_y, cross_track_error): Unit direction and distance from circle
    """
    # Radial vector from center to particle
    rx = particle_x - center_x
    ry = particle_y - center_y
    r_norm = np.sqrt(rx**2 + ry**2)
    eps = 1e-12
    
    # Closest point on circle
    r_hat_x = rx / (r_norm + eps)
    r_hat_y = ry / (r_norm + eps)
    p_star_x = center_x + radius * r_hat_x
    p_star_y = center_y + radius * r_hat_y
    
    # Tangent direction at p* (perpendicular to radial)
    # CCW: rotate r_hat by +90 degrees -> (-r_hat_y, r_hat_x)
    # CW: rotate r_hat by -90 degrees -> (r_hat_y, -r_hat_x)
    if ccw:
        t_hat_x = -r_hat_y
        t_hat_y = r_hat_x
    else:
        t_hat_x = r_hat_y
        t_hat_y = -r_hat_x
    
    # Radial error (points toward circle from particle)
    e_x = p_star_x - particle_x
    e_y = p_star_y - particle_y
    cross_track_error = np.sqrt(e_x**2 + e_y**2)
    
    # Combined direction: tangent + radial correction
    d_x = t_hat_x + k_radial * e_x
    d_y = t_hat_y + k_radial * e_y
    d_norm = np.sqrt(d_x**2 + d_y**2)
    d_hat_x = d_x / (d_norm + eps)
    d_hat_y = d_y / (d_norm + eps)
    
    return d_hat_x, d_hat_y, cross_track_error


def compute_angle_from_center(x: float, y: float, cx: float, cy: float) -> float:
    """Compute angle (radians) of point relative to circle center."""
    return np.arctan2(y - cy, x - cx)


# =============================================================================
# Contour Color Scale Manager
# =============================================================================

class ContourScaleManager:
    """Manages stable color scaling for Gor'kov potential contours."""
    
    def __init__(
        self,
        mode: str = "warmup_fixed",  # "warmup_fixed", "fixed", "per_frame"
        warmup_frames: int = 10,
        pct_lo: float = 5.0,
        pct_hi: float = 95.0,
        min_range: float = 1e-16,
        upsample: int = 1,
        verbose: bool = False,
    ):
        self.mode = mode
        self.warmup_frames = warmup_frames
        self.pct_lo = pct_lo
        self.pct_hi = pct_hi
        self.min_range = min_range
        self.upsample = upsample
        self.verbose = verbose
        
        # State for warmup_fixed mode
        self.warmup_U_samples: list[np.ndarray] = []
        self.frozen_vmin: float | None = None
        self.frozen_vmax: float | None = None
        self.last_good_vmin: float = 0.0
        self.last_good_vmax: float = 1.0
        self.frame_count = 0
    
    def get_vmin_vmax(self, U: np.ndarray) -> tuple[float, float]:
        """Get vmin, vmax for the given U field."""
        self.frame_count += 1
        
        # Clean the U array
        U_clean = np.nan_to_num(U, nan=0.0, posinf=0.0, neginf=0.0)
        
        if self.mode == "per_frame":
            # Compute percentiles fresh each frame
            vmin = float(np.percentile(U_clean, self.pct_lo))
            vmax = float(np.percentile(U_clean, self.pct_hi))
            return self._validate_range(vmin, vmax)
        
        elif self.mode == "fixed":
            # Use frozen values from the very first frame
            if self.frozen_vmin is None:
                vmin = float(np.percentile(U_clean, self.pct_lo))
                vmax = float(np.percentile(U_clean, self.pct_hi))
                vmin, vmax = self._validate_range(vmin, vmax)
                self.frozen_vmin = vmin
                self.frozen_vmax = vmax
            return self.frozen_vmin, self.frozen_vmax
        
        else:  # warmup_fixed
            if self.frozen_vmin is None:
                # Still in warmup phase
                self.warmup_U_samples.append(U_clean.flatten())
                
                if len(self.warmup_U_samples) >= self.warmup_frames:
                    # Compute percentiles over all warmup samples
                    all_U = np.concatenate(self.warmup_U_samples)
                    vmin = float(np.percentile(all_U, self.pct_lo))
                    vmax = float(np.percentile(all_U, self.pct_hi))
                    vmin, vmax = self._validate_range(vmin, vmax)
                    self.frozen_vmin = vmin
                    self.frozen_vmax = vmax
                    self.warmup_U_samples = []  # Free memory
                    return vmin, vmax
                else:
                    # During warmup, use per-frame scaling
                    vmin = float(np.percentile(U_clean, self.pct_lo))
                    vmax = float(np.percentile(U_clean, self.pct_hi))
                    return self._validate_range(vmin, vmax)
            else:
                return self.frozen_vmin, self.frozen_vmax
    
    def _validate_range(self, vmin: float, vmax: float) -> tuple[float, float]:
        """Ensure range is valid, use last good values if not."""
        # Check for invalid values
        if not (np.isfinite(vmin) and np.isfinite(vmax)):
            return self.last_good_vmin, self.last_good_vmax
        
        # Check for too-small range
        if vmax - vmin < self.min_range:
            return self.last_good_vmin, self.last_good_vmax
        
        # Valid range - save and return
        self.last_good_vmin = vmin
        self.last_good_vmax = vmax
        return vmin, vmax
    
    def upsample_field(self, x: np.ndarray, y: np.ndarray, U: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Upsample U field for smoother contour plotting."""
        if self.upsample <= 1:
            return x, y, U
        
        # Create interpolator
        interp = RegularGridInterpolator((x, y), U, method='linear', bounds_error=False, fill_value=None)
        
        # Create finer grid
        x_fine = np.linspace(x[0], x[-1], len(x) * self.upsample)
        y_fine = np.linspace(y[0], y[-1], len(y) * self.upsample)
        
        # Interpolate
        xx, yy = np.meshgrid(x_fine, y_fine, indexing='ij')
        U_fine = interp((xx, yy))
        
        return x_fine, y_fine, U_fine
    
    def log_frame(self, U: np.ndarray, vmin: float, vmax: float):
        """Log contour stats if verbose."""
        if not self.verbose:
            return
        
        U_clean = np.nan_to_num(U, nan=0.0, posinf=0.0, neginf=0.0)
        finite_frac = np.mean(np.isfinite(U))
        p5 = float(np.percentile(U_clean, 5))
        p95 = float(np.percentile(U_clean, 95))
        print(f"  [Contour] U_min={U_clean.min():.2e} U_max={U_clean.max():.2e} "
              f"p5={p5:.2e} p95={p95:.2e} range={p95-p5:.2e} "
              f"finite={finite_frac:.1%} vmin={vmin:.2e} vmax={vmax:.2e}")


# =============================================================================
# Rendering
# =============================================================================

def render_surf_frame(
    frame_path: Path,
    step: int,
    domain: DishDomain,
    particle_x: float,
    particle_y: float,
    target_x: float,
    target_y: float,
    ctrl: Control3Pucks,
    path: np.ndarray,
    traj_xy_mm: list[tuple[float, float]],
    log: GreedySurfStep,
    errors: list[float],
    field_data: tuple | None = None,  # (field, U, Fx, Fy) for contour plotting
    show_contours: bool = True,
    contour_mgr: ContourScaleManager | None = None,
):
    """Render frame for greedy surf demo."""
    fig = plt.figure(figsize=(16, 6))
    
    # === Main 2D view ===
    ax1 = fig.add_subplot(1, 3, 1)
    
    # Domain extent
    ax1.set_xlim(0, domain.Lx * 1e3)
    ax1.set_ylim(0, domain.Ly * 1e3)

    # Gor'kov potential contour (background)
    if show_contours and field_data is not None:
        field, U, Fx, Fy = field_data
        
        # Get color scale from manager
        if contour_mgr is not None:
            vmin, vmax = contour_mgr.get_vmin_vmax(U)
            x_plot, y_plot, U_plot = contour_mgr.upsample_field(field.x, field.y, U)
            contour_mgr.log_frame(U, vmin, vmax)
        else:
            # Fallback to per-frame
            U_clean = np.nan_to_num(U, nan=0.0)
            vmin = float(np.percentile(U_clean, 5))
            vmax = float(np.percentile(U_clean, 95))
            x_plot, y_plot, U_plot = field.x, field.y, U
        
        # Convert grid to mm for plotting
        X_mm = x_plot * 1e3
        Y_mm = y_plot * 1e3
        
        # Clip U values to [vmin, vmax] to avoid extreme outliers dominating the colormap
        # The Gor'kov potential has localized extreme minima (traps) that create visual artifacts
        U_clipped = np.clip(U_plot, vmin, vmax)
        
        # Contour fill with transparency
        cf = ax1.contourf(X_mm, Y_mm, U_clipped.T, levels=30, cmap='viridis',
                          vmin=vmin, vmax=vmax, alpha=0.6)
        # Add thin contour lines for clarity
        ax1.contour(X_mm, Y_mm, U_clipped.T, levels=15, colors='white',
                    linewidths=0.3, alpha=0.4)

    # Desired path
    ax1.plot(path[:, 0] * 1e3, path[:, 1] * 1e3, 'k--', lw=1.5, label='desired path')
    
    # Actual trajectory
    if len(traj_xy_mm) > 1:
        traj = np.array(traj_xy_mm)
        for i in range(len(traj) - 1):
            alpha = (i + 1) / len(traj)
            ax1.plot(traj[i:i+2, 0], traj[i:i+2, 1],
                    color=(0, 1 - 0.5*alpha, 1), lw=2, alpha=0.5 + 0.5*alpha)
    
    # Current particle
    ax1.scatter(particle_x * 1e3, particle_y * 1e3, s=200, c='red', marker='o',
                edgecolors='white', linewidths=2, zorder=10, label='particle')
    
    # Target
    ax1.scatter(target_x * 1e3, target_y * 1e3, s=120, c='yellow', marker='*',
                edgecolors='black', linewidths=1.5, zorder=9, label='target')
    
    # Force vector at particle
    scale = 2e6  # Scale force for visualization
    fx_vis = log.Fp_x * scale
    fy_vis = log.Fp_y * scale
    ax1.quiver(particle_x * 1e3, particle_y * 1e3, fx_vis * 1e3, fy_vis * 1e3,
               color='green', scale=1, scale_units='xy', width=0.008, zorder=8,
               label=f'force (align={log.Fp_hat_dot_d:.2f})')
    
    # Transducers
    ax1.scatter(ctrl.xA * 1e3, ctrl.yA * 1e3, s=100, c='orange',
                marker='^', edgecolors='black', label='puck A')
    ax1.scatter(ctrl.xB * 1e3, ctrl.yB * 1e3, s=100, c='blue',
                marker='^', edgecolors='black', label='puck B')
    ax1.scatter(ctrl.xC * 1e3, ctrl.yC * 1e3, s=100, c='magenta',
                marker='^', edgecolors='black', label='puck C')
    
    ax1.set_xlabel('x (mm)')
    ax1.set_ylabel('y (mm)')
    ax1.set_title(f'Step {step}: {log.chosen_action}\n'
                  f'F·d̂={log.Fp_dot_d:.2e}, F̂·d̂={log.Fp_hat_dot_d:.2f}')
    ax1.set_aspect('equal')
    ax1.legend(loc='upper right', fontsize=6, ncol=2)
    ax1.grid(True, alpha=0.3)
    
    # === Error plot ===
    ax2 = fig.add_subplot(1, 3, 2)
    if len(errors) > 0:
        ax2.plot(errors, 'b-', lw=1.5)
        ax2.axhline(np.mean(errors), color='r', linestyle='--',
                   label=f'mean={np.mean(errors)*1e3:.2f} µm')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Error (mm)')
        ax2.set_title('Tracking Error')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, max(len(errors) + 10, 50))
    else:
        ax2.text(0.5, 0.5, 'Collecting data...', ha='center', va='center',
                transform=ax2.transAxes)
    
    # === Alignment plot ===
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.text(0.05, 0.95, f"Step: {step}", transform=ax3.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace')
    ax3.text(0.05, 0.85, f"Action: {log.chosen_action}", transform=ax3.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    ax3.text(0.05, 0.75, f"Switched: {log.action_switched}", transform=ax3.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    ax3.text(0.05, 0.65, f"F̂·d̂ (align): {log.Fp_hat_dot_d:.3f}", transform=ax3.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    ax3.text(0.05, 0.55, f"F·d̂ (push): {log.Fp_dot_d:.2e}", transform=ax3.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    ax3.text(0.05, 0.45, f"|F|: {log.Fp_mag:.2e}", transform=ax3.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    ax3.text(0.05, 0.35, f"Score: {log.score:.3f}", transform=ax3.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    ax3.text(0.05, 0.25, f"Trap stable: {log.trap_stable}", transform=ax3.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    ax3.text(0.05, 0.15, f"Error: {log.tracking_error*1e3:.1f} µm", transform=ax3.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    ax3.set_title('Step Diagnostics')
    
    plt.tight_layout()
    plt.savefig(frame_path, dpi=100)
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Greedy Surf Controller Demo")
    parser.add_argument("--fast", action="store_true", help="Fast mode (fewer steps)")
    parser.add_argument("--steps", type=int, default=200, help="Number of steps")
    parser.add_argument("--path", type=str, default="line", choices=["line", "circle"],
                       help="Path type")
    parser.add_argument("--action_subset", type=str, default="none",
                       choices=["none", "small", "wide", "pruned", "pruned_progress"],
                       help="Action subset: none=full 13, small=8, wide=15, pruned=10, pruned_progress=12")
    parser.add_argument("--coarse", action="store_true", help="Use coarse grid")
    parser.add_argument("--render_stride", type=int, default=None,
                       help="Render every N steps (default: 3 if --fast, else 2)")

    # Macro action step sizes
    parser.add_argument("--macro_step_pos_um", type=float, default=50.0,
                       help="Position step for macro actions (µm)")
    parser.add_argument("--macro_step_angle_deg", type=float, default=5.0,
                       help="Angle step for macro actions (degrees) - NOT USED currently")
    parser.add_argument("--macro_step_phase_rad", type=float, default=0.3,
                       help="Phase step for macro actions (radians)")
    
    # Target gating
    parser.add_argument("--target_hold_tol_um", type=float, default=120.0,
                       help="Target only advances when particle is within this distance (µm)")

    # Visualization / Contour settings
    parser.add_argument("--gif_contours", type=lambda x: x.lower() in ('true', '1', 'yes'),
                       default=True, help="Overlay Gor'kov potential contours on GIF frames")
    parser.add_argument("--contour_scale", type=str, default="warmup_fixed",
                       choices=["warmup_fixed", "fixed", "per_frame"],
                       help="Contour color scale mode")
    parser.add_argument("--contour_warmup_frames", type=int, default=10,
                       help="Number of frames to use for warmup color scale")
    parser.add_argument("--contour_percentiles", type=str, default="5,95",
                       help="Percentiles for color scaling (comma-separated)")
    parser.add_argument("--contour_min_range", type=float, default=1e-16,
                       help="Minimum range for valid color scale")
    parser.add_argument("--contour_upsample", type=int, default=1,
                       help="Upsampling factor for contour plotting (1=off)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose debug logging for contours")
    
    # Circle direction mode
    parser.add_argument("--circle_dir_mode", type=str, default="tangent_plus_radial",
                       choices=["tangent_plus_radial", "waypoint"],
                       help="Direction mode for circle paths")
    parser.add_argument("--k_radial", type=float, default=2.0,
                       help="Radial correction gain for tangent_plus_radial mode")
    
    # Stuck detector
    parser.add_argument("--stuck_detect", type=lambda x: x.lower() in ('true', '1', 'yes'),
                       default=True, help="Enable stuck detector for circles")
    parser.add_argument("--stuck_window", type=int, default=10,
                       help="Window size for stuck detection")
    parser.add_argument("--stuck_threshold_deg", type=float, default=3.0,
                       help="Minimum angle progress in stuck_window to not be stuck")
    parser.add_argument("--stuck_w_switch_factor", type=float, default=0.1,
                       help="Multiply w_switch by this when stuck")
    parser.add_argument("--stuck_w_push_factor", type=float, default=2.0,
                       help="Multiply w_push by this when stuck")

    # Scoring weights
    parser.add_argument("--w_align", type=float, default=1.0)
    parser.add_argument("--w_push", type=float, default=1e6)
    parser.add_argument("--w_switch", type=float, default=0.05)
    
    # Controller selection
    parser.add_argument("--controller", type=str, default="greedy",
                       choices=["greedy", "bayes"],
                       help="Controller type: greedy (full PDE) or bayes (UCB surrogate)")
    
    # Bayesian controller parameters
    parser.add_argument("--bayes_k", type=int, default=3,
                       help="Number of actions to evaluate with PDE per step in Bayes mode")
    parser.add_argument("--bayes_kappa", type=float, default=1.0,
                       help="UCB exploration weight (kappa)")
    parser.add_argument("--bayes_warmup_steps", type=int, default=10,
                       help="Number of steps to use full greedy before switching to Bayes")
    parser.add_argument("--bayes_model", type=str, default="rf",
                       choices=["rf", "gp"],
                       help="Surrogate model type (rf=RandomForest, gp=GaussianProcess)")
    
    # Bayesian robustness parameters
    parser.add_argument("--bayes_max_repeat", type=int, default=5,
                       help="Max consecutive steps with same action before forcing diversity")
    parser.add_argument("--bayes_stuck_rewarmup_steps", type=int, default=5,
                       help="Steps of full evaluation when stuck detected")
    
    args = parser.parse_args()
    
    # Parse contour percentiles
    contour_pct = [float(x) for x in args.contour_percentiles.split(",")]
    contour_pct_lo, contour_pct_hi = contour_pct[0], contour_pct[1]
    
    REPO = Path(__file__).resolve().parents[1]
    RESULTS = REPO / "results"
    
    # Create timestamped output directory
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS / "demo_surf_greedy" / f"run_{run_timestamp}"
    frames_dir = out_dir / "frames"
    
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("GREEDY SURF CONTROLLER DEMO")
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
    
    ev = Evaluator3Pucks(domain, medium, particle, cfg)
    
    print(f"\nPhysics:")
    print(f"  Domain: {domain.Lx*1e3:.1f} x {domain.Ly*1e3:.1f} mm")
    print(f"  Grid: {domain.Nx} x {domain.Ny}")
    print(f"  alpha_g: {cfg.alpha_g:.0e}")
    
    # ===== Macro action step sizes =====
    macro_step_pos = args.macro_step_pos_um * 1e-6  # µm -> m
    macro_step_phase = args.macro_step_phase_rad    # radians
    # angle_deg is reserved for future use (e.g., rotating puck arrays)
    
    # ===== Controller Setup =====
    surf_config = GreedySurfConfig(
        w_align=args.w_align,
        w_push=args.w_push,
        w_switch=args.w_switch,
        dt=cfg.dt,
        max_step=cfg.max_step,
        macro_magnitude=macro_step_pos,
        macro_phase_step=macro_step_phase,
    )
    
    # Action set based on subset choice
    action_set = None
    if args.action_subset == "small":
        # Reduced core set (8 actions)
        action_set = [
            MacroActionType3Puck.HOLD,
            MacroActionType3Puck.TRANSLATE_TRAP_X_POS,
            MacroActionType3Puck.TRANSLATE_TRAP_X_NEG,
            MacroActionType3Puck.TRANSLATE_TRAP_Y_POS,
            MacroActionType3Puck.TRANSLATE_TRAP_Y_NEG,
            MacroActionType3Puck.ROTATE_INTERFERENCE_CW,
            MacroActionType3Puck.ROTATE_INTERFERENCE_CCW,
            MacroActionType3Puck.PHASE_SHIFT_B_POS,
        ]
    elif args.action_subset == "wide":
        # Expanded set with individual puck moves and C control (15 actions)
        action_set = [
            MacroActionType3Puck.HOLD,
            MacroActionType3Puck.TRANSLATE_TRAP_X_POS,
            MacroActionType3Puck.TRANSLATE_TRAP_X_NEG,
            MacroActionType3Puck.TRANSLATE_TRAP_Y_POS,
            MacroActionType3Puck.TRANSLATE_TRAP_Y_NEG,
            MacroActionType3Puck.ROTATE_INTERFERENCE_CW,
            MacroActionType3Puck.ROTATE_INTERFERENCE_CCW,
            MacroActionType3Puck.MOVE_A_RIGHT,
            MacroActionType3Puck.MOVE_A_LEFT,
            MacroActionType3Puck.MOVE_B_RIGHT,
            MacroActionType3Puck.MOVE_B_LEFT,
            MacroActionType3Puck.MOVE_C_UP,
            MacroActionType3Puck.MOVE_C_DOWN,
            MacroActionType3Puck.PHASE_SHIFT_B_POS,
            MacroActionType3Puck.PHASE_SHIFT_B_NEG,
        ]
    elif args.action_subset == "pruned":
        # High-authority action set based on control authority diagnostic (10 actions)
        # Removed: TRANSLATE_TRAP_Y_*, MOVE_B_RIGHT, MOVE_A_LEFT, PHASE_SHIFT_C_*
        # Kept only actions with max_delta_force_proj > 1e-10 N
        # This basis is well-conditioned: all actions have comparable authority
        action_set = [
            MacroActionType3Puck.HOLD,                     # Stability baseline
            MacroActionType3Puck.TRANSLATE_TRAP_X_POS,     # 4.93e-10 N (strong x-control)
            MacroActionType3Puck.TRANSLATE_TRAP_X_NEG,     # 3.54e-10 N (symmetric pair)
            MacroActionType3Puck.ROTATE_INTERFERENCE_CW,   # 6.28e-10 N (strongest rotation)
            MacroActionType3Puck.ROTATE_INTERFERENCE_CCW,  # 4.99e-10 N (symmetric pair)
            MacroActionType3Puck.MOVE_B_LEFT,              # 6.65e-10 N (strongest individual)
            MacroActionType3Puck.MOVE_C_DOWN,              # 5.81e-10 N (y-authority via C)
            MacroActionType3Puck.MOVE_C_UP,                # 3.12e-10 N (symmetric C pair)
            MacroActionType3Puck.PHASE_SHIFT_B_POS,        # 2.85e-10 N (phase control)
            MacroActionType3Puck.PHASE_SHIFT_B_NEG,        # 3.14e-10 N (symmetric pair)
        ]
    elif args.action_subset == "pruned_progress":
        # Extended high-authority set for circle trajectory progress (12 actions)
        # Adds TRANSLATE_TRAP_Y_* and NARROW for better tangential control
        # Based on 4-puck diagnostic: TOGGLE_B_OFF and MOVE_B_LEFT are highest authority
        # but toggling is too disruptive for continuous tracking.
        # This set balances tracking accuracy with tangential progress capability.
        action_set = [
            MacroActionType3Puck.HOLD,                     # Stability baseline
            MacroActionType3Puck.TRANSLATE_TRAP_X_POS,     # Strong x-control
            MacroActionType3Puck.TRANSLATE_TRAP_X_NEG,     # Symmetric pair
            MacroActionType3Puck.TRANSLATE_TRAP_Y_POS,     # Y-control for tangential progress
            MacroActionType3Puck.TRANSLATE_TRAP_Y_NEG,     # Symmetric pair
            MacroActionType3Puck.ROTATE_INTERFERENCE_CW,   # Strongest rotation
            MacroActionType3Puck.ROTATE_INTERFERENCE_CCW,  # Symmetric pair
            MacroActionType3Puck.MOVE_B_LEFT,              # Highest authority individual move
            MacroActionType3Puck.NARROW,                   # ~4.2e-10 N (helps with trap shaping)
            MacroActionType3Puck.MOVE_C_DOWN,              # Y-authority via C position
            MacroActionType3Puck.PHASE_SHIFT_B_POS,        # Phase control
            MacroActionType3Puck.PHASE_SHIFT_B_NEG,        # Symmetric pair
        ]
    # else action_set = None -> uses default 13-action set
    
    # Instantiate controller based on selection
    if args.controller == "bayes":
        controller = BayesGreedySurfController(
            ev, surf_config, action_set,
            bayes_k=args.bayes_k,
            bayes_kappa=args.bayes_kappa,
            bayes_warmup_steps=args.bayes_warmup_steps,
            max_repeat=args.bayes_max_repeat,
            stuck_rewarmup_steps=args.bayes_stuck_rewarmup_steps,
        )
        controller_mode = "bayes"
    else:
        controller = GreedySurfController(ev, surf_config, action_set)
        controller_mode = "greedy"
    
    print(f"\nController ({controller_mode}):")
    print(f"  w_align: {surf_config.w_align}")
    print(f"  w_push: {surf_config.w_push:.1e}")
    print(f"  w_switch: {surf_config.w_switch}")
    print(f"  macro_step_pos: {surf_config.macro_magnitude*1e6:.0f} µm")
    print(f"  macro_step_phase: {surf_config.macro_phase_step:.2f} rad")
    print(f"  Actions ({args.action_subset}): {len(controller.action_types)}")
    if args.controller == "bayes":
        print(f"  Bayes K: {args.bayes_k}")
        print(f"  Bayes kappa: {args.bayes_kappa}")
        print(f"  Bayes warmup: {args.bayes_warmup_steps} steps")
        print(f"  Bayes max_repeat: {args.bayes_max_repeat}")
        print(f"  Bayes stuck_rewarmup: {args.bayes_stuck_rewarmup_steps} steps")
    for a in controller.action_types:
        print(f"    - {a.name}")
    
    # ===== Path Setup =====
    T = 100 if args.fast else args.steps
    
    if args.path == "line":
        # Straight line across domain - start more centrally for better initial conditions
        start_x = 0.5e-3
        start_y = domain.Ly * 0.5
        end_x = 1.5e-3
        end_y = domain.Ly * 0.5
        path = make_straight_line_path(start_x, start_y, end_x, end_y, T)
        path_name = "straight_line"
        print(f"\nPath: Straight line")
        print(f"  From: ({start_x*1e3:.2f}, {start_y*1e3:.2f}) mm")
        print(f"  To:   ({end_x*1e3:.2f}, {end_y*1e3:.2f}) mm")
    else:
        # Circle
        radius = 0.4e-3
        center_x = domain.Lx / 2
        center_y = domain.Ly * 0.55
        path = make_circle_path(center_x, center_y, radius, T)
        path_name = "circle"
        print(f"\nPath: Circle")
        print(f"  Center: ({center_x*1e3:.2f}, {center_y*1e3:.2f}) mm")
        print(f"  Radius: {radius*1e3:.2f} mm")
    
    print(f"  Steps: {T}")
    
    # Target gating parameters
    target_hold_tol = args.target_hold_tol_um * 1e-6  # Convert µm to m
    is_closed_path_for_gating = (args.path == "circle")  # Circle wraps around
    print(f"  Target hold tolerance: {args.target_hold_tol_um:.0f} µm")
    
    # ===== Initial State =====
    particle_x = float(path[0, 0])
    particle_y = float(path[0, 1])
    
    # Initial 3-puck configuration - position transducers around the particle start
    # Puck A and B straddle the particle along x, C provides y-control from above
    ctrl = Control3Pucks(
        xA=particle_x - 0.4e-3, yA=0.03e-3, vA=0.08, phiA=0.0,
        xB=particle_x + 0.4e-3, yB=0.03e-3, vB=0.08, phiB=np.pi,
        xC=particle_x, yC=0.20e-3, vC=0.08, phiC=np.pi/2,
    )
    ctrl = ev.clip_control(ctrl)  # Ensure within bounds
    
    print(f"\nInitial transducers:")
    print(f"  A: ({ctrl.xA*1e3:.2f}, {ctrl.yA*1e3:.2f}) mm")
    print(f"  B: ({ctrl.xB*1e3:.2f}, {ctrl.yB*1e3:.2f}) mm")
    print(f"  C: ({ctrl.xC*1e3:.2f}, {ctrl.yC*1e3:.2f}) mm")
    
    # ===== Contour Manager =====
    contour_mgr = ContourScaleManager(
        mode=args.contour_scale,
        warmup_frames=args.contour_warmup_frames,
        pct_lo=contour_pct_lo,
        pct_hi=contour_pct_hi,
        min_range=args.contour_min_range,
        upsample=args.contour_upsample,
        verbose=args.verbose,
    )
    
    # ===== Circle-specific setup =====
    is_circle = (args.path == "circle")
    if is_circle:
        circle_center_x = center_x
        circle_center_y = center_y
        circle_radius = radius
        print(f"\nCircle direction mode: {args.circle_dir_mode}")
        print(f"  k_radial: {args.k_radial}")
    
    # ===== Simulation Loop =====
    traj_xy_mm: list[tuple[float, float]] = [(particle_x * 1e3, particle_y * 1e3)]
    error_history: list[float] = []
    step_logs: list[GreedySurfStep] = []
    frame_paths: list[Path] = []
    
    alignment_history: list[float] = []
    push_history: list[float] = []
    action_history: list[str] = []
    switch_count = 0
    
    # Circle-specific tracking
    angle_history: list[float] = []  # Unwrapped angle progress
    cross_track_history: list[float] = []  # Distance from circle
    tangential_alignment_history: list[float] = []  # Alignment with tangent
    
    if is_circle:
        initial_angle = compute_angle_from_center(particle_x, particle_y, circle_center_x, circle_center_y)
        angle_history.append(initial_angle)
    
    # Target gating state
    target_idx = 1  # Start targeting the second waypoint (first is starting position)
    target_advance_count = 0
    
    # Stuck detector state
    stuck_mode_steps_remaining = 0
    stuck_kick_count = 0
    
    print(f"\nStarting simulation...")
    print("-" * 110)
    print(f"{'Step':>5} {'px_mm':>8} {'py_mm':>8} {'err_µm':>8} {'tgt_idx':>7} {'adv':>3} {'action':>22} "
          f"{'F̂·d̂':>8} {'F·d̂':>10} {'stab':>4}")
    print("-" * 110)
    
    if args.render_stride is not None:
        render_stride = args.render_stride
    else:
        render_stride = 3 if args.fast else 2
    
    for t in range(T - 1):
        # Get current target from gated index (not t+1)
        target_x = float(path[target_idx, 0])
        target_y = float(path[target_idx, 1])
        
        # Compute distance to target BEFORE this step
        dist_to_target = np.sqrt((particle_x - target_x)**2 + (particle_y - target_y)**2)
        
        # Compute desired direction
        desired_direction = None
        cross_track_error = 0.0
        if is_circle and args.circle_dir_mode == "tangent_plus_radial":
            d_hat_x, d_hat_y, cross_track_error = compute_circle_direction(
                particle_x, particle_y,
                circle_center_x, circle_center_y, circle_radius,
                k_radial=args.k_radial, ccw=True
            )
            desired_direction = (d_hat_x, d_hat_y)
        
        # Stuck detector: check if we should apply weight overrides
        weight_overrides = None
        if stuck_mode_steps_remaining > 0:
            weight_overrides = {
                'w_switch': args.w_switch * args.stuck_w_switch_factor,
                'w_push': args.w_push * args.stuck_w_push_factor,
            }
            stuck_mode_steps_remaining -= 1
        elif args.stuck_detect and is_circle and t >= args.stuck_window:
            # Check angle progress over last stuck_window steps
            recent_angles = angle_history[-args.stuck_window:]
            if len(recent_angles) >= args.stuck_window:
                angle_progress = abs(recent_angles[-1] - recent_angles[0])
                angle_progress_deg = np.degrees(angle_progress)
                if angle_progress_deg < args.stuck_threshold_deg:
                    # We're stuck! Enable kick mode
                    stuck_mode_steps_remaining = args.stuck_window
                    stuck_kick_count += 1
                    if args.verbose:
                        print(f"  [Stuck] Detected at step {t}, angle progress {angle_progress_deg:.1f}° < {args.stuck_threshold_deg}°")
                    weight_overrides = {
                        'w_switch': args.w_switch * args.stuck_w_switch_factor,
                        'w_push': args.w_push * args.stuck_w_push_factor,
                    }
        
        # Controller step
        step_kwargs = dict(
            particle_x=particle_x,
            particle_y=particle_y,
            target_x=target_x,
            target_y=target_y,
            ctrl=ctrl,
            step_idx=t,
            desired_direction=desired_direction,
            weight_overrides=weight_overrides,
        )
        # Bayes controller needs cross_track_error for features
        if args.controller == "bayes":
            step_kwargs["cross_track_error"] = cross_track_error
        
        new_ctrl, new_x, new_y, log = controller.step(**step_kwargs)
        
        # Update state
        ctrl = new_ctrl
        particle_x = new_x
        particle_y = new_y
        
        # Track circle-specific metrics
        if is_circle:
            current_angle = compute_angle_from_center(particle_x, particle_y, circle_center_x, circle_center_y)
            # Unwrap angle
            if angle_history:
                prev_angle = angle_history[-1]
                delta = current_angle - (prev_angle % (2 * np.pi))
                # Handle wraparound
                if delta > np.pi:
                    delta -= 2 * np.pi
                elif delta < -np.pi:
                    delta += 2 * np.pi
                current_angle = prev_angle + delta
            angle_history.append(current_angle)
            cross_track_history.append(cross_track_error)
            
            # Compute tangential alignment (force dot tangent at current position)
            rx = particle_x - circle_center_x
            ry = particle_y - circle_center_y
            r_norm = np.sqrt(rx**2 + ry**2) + 1e-12
            t_hat_x = -ry / r_norm  # CCW tangent
            t_hat_y = rx / r_norm
            Fp_mag = log.Fp_mag + 1e-15
            tangential_align = (log.Fp_x * t_hat_x + log.Fp_y * t_hat_y) / Fp_mag
            tangential_alignment_history.append(tangential_align)
            log.cross_track_error = cross_track_error
            log.tangential_alignment = tangential_align
        
        # Target gating: check if we should advance target
        target_advanced = False
        
        if is_circle:
            # For circles: advance target based on angle
            # When particle's angle passes the target's angle, move target ahead
            particle_angle = compute_angle_from_center(
                particle_x, particle_y, circle_center_x, circle_center_y
            )
            target_angle = compute_angle_from_center(
                target_x, target_y, circle_center_x, circle_center_y
            )
            
            # Check if particle has passed target (accounting for wraparound)
            # We want target to be ahead of particle in CCW direction
            # Compute angle difference (target - particle), normalized to [-pi, pi]
            angle_diff = target_angle - particle_angle
            while angle_diff > np.pi:
                angle_diff -= 2 * np.pi
            while angle_diff < -np.pi:
                angle_diff += 2 * np.pi
            
            # If particle is within ~20° ahead of target OR has passed it, advance target
            # (angle_diff < small_positive means target is behind or barely ahead of particle)
            advance_threshold_rad = np.radians(15)  # Keep target at least 15° ahead
            if angle_diff < advance_threshold_rad:
                # Advance target to stay ahead of particle
                target_idx = (target_idx + 1) % len(path)
                target_advanced = True
                target_advance_count += 1
        else:
            # For lines: use distance-based advancement
            dist_after_step = np.sqrt((particle_x - target_x)**2 + (particle_y - target_y)**2)
            if dist_after_step <= target_hold_tol:
                target_idx = min(target_idx + 1, len(path) - 1)
                target_advanced = True
                target_advance_count += 1
        
        # Update log with target gating info
        log.target_idx = target_idx
        log.dist_to_target = dist_to_target
        log.target_advanced = target_advanced
        
        # Log
        traj_xy_mm.append((particle_x * 1e3, particle_y * 1e3))
        error_history.append(log.tracking_error)
        step_logs.append(log)
        alignment_history.append(log.Fp_hat_dot_d)
        push_history.append(log.Fp_dot_d)
        action_history.append(log.chosen_action)
        if log.action_switched:
            switch_count += 1
        
        # Print progress
        if t % 20 == 0 or t == T - 2:
            print(f"{t:5d} {particle_x*1e3:8.4f} {particle_y*1e3:8.4f} "
                  f"{log.tracking_error*1e6:8.1f} {target_idx:7d} {'Y' if target_advanced else 'N':>3} {log.chosen_action:>22} "
                  f"{log.Fp_hat_dot_d:8.3f} {log.Fp_dot_d:10.2e} "
                  f"{'Y' if log.trap_stable else 'N':>4}")
        
        # Render frame
        if t % render_stride == 0:
            frame_path = frames_dir / f"frame_{t:04d}.png"
            render_surf_frame(
                frame_path=frame_path,
                step=t,
                domain=domain,
                particle_x=particle_x,
                particle_y=particle_y,
                target_x=target_x,
                target_y=target_y,
                ctrl=ctrl,
                path=path,
                traj_xy_mm=traj_xy_mm,
                log=log,
                errors=error_history,
                field_data=controller.best_field,  # Chosen action's Gor'kov potential field
                show_contours=args.gif_contours,
                contour_mgr=contour_mgr,  # Stable color scaling
            )
            frame_paths.append(frame_path)
    
    print("-" * 100)
    
    # ===== Results Summary =====
    errors = np.array(error_history)
    alignments = np.array(alignment_history)
    pushes = np.array(push_history)
    
    # Compute net progress
    initial_pos = np.array([path[0, 0], path[0, 1]])
    final_pos = np.array([particle_x, particle_y])
    final_target = np.array([path[-1, 0], path[-1, 1]])
    path_direction = final_target - initial_pos
    path_length = np.linalg.norm(path_direction)
    
    # Circle-specific metrics
    angle_progress_deg = 0.0
    mean_cross_track_error_um = 0.0
    pct_positive_tangential = 0.0
    
    if is_circle and len(angle_history) > 1:
        angle_progress_deg = np.degrees(angle_history[-1] - angle_history[0])
        mean_cross_track_error_um = np.mean(cross_track_history) * 1e6
        pct_positive_tangential = 100.0 * np.mean(np.array(tangential_alignment_history) > 0)
    
    # Check if this is a closed path (start ~= end)
    is_closed_path = bool(path_length < 1e-6)
    
    if is_closed_path:
        # For closed paths, compute total arc length of desired path
        total_arc_length = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))
        # And actual trajectory arc length
        traj_array = np.array(traj_xy_mm) * 1e-3  # Convert back to meters
        actual_arc_length = np.sum(np.linalg.norm(np.diff(traj_array, axis=0), axis=1))
        # Net progress = how much of the arc was covered (approximation)
        net_progress = actual_arc_length
        path_length = total_arc_length
    else:
        # Open path - measure progress toward end
        path_direction_hat = path_direction / path_length
        net_displacement = final_pos - initial_pos
        net_progress = np.dot(net_displacement, path_direction_hat)
    
    # Final distance to target
    final_error = np.linalg.norm(final_pos - final_target)
    initial_error = np.linalg.norm(initial_pos - final_target)
    
    pct_positive_alignment = 100.0 * np.mean(alignments > 0)
    pct_positive_push = 100.0 * np.mean(pushes > 0)
    
    mean_solver_time = np.mean([log.solver_time_ms for log in step_logs])
    
    print(f"\n{'='*70}")
    print("GREEDY SURF CONTROLLER RESULTS")
    print(f"{'='*70}")
    
    print(f"\n  NET PROGRESS:")
    print(f"    Initial → final target distance: {initial_error*1e3:.3f} mm → {final_error*1e3:.3f} mm")
    if is_closed_path:
        print(f"    Arc length traveled: {net_progress*1e3:.3f} mm")
        print(f"    Desired arc length: {path_length*1e3:.3f} mm")
    else:
        print(f"    Net progress along path direction: {net_progress*1e3:.3f} mm")
        print(f"    Path length: {path_length*1e3:.3f} mm")
    progress_frac = net_progress / path_length if path_length > 0 else 0.0
    print(f"    Progress fraction: {progress_frac*100:.1f}%")
    
    # Circle-specific output
    if is_circle:
        print(f"\n  CIRCLE METRICS:")
        print(f"    Angle progress: {angle_progress_deg:.1f}°")
        print(f"    Mean cross-track error: {mean_cross_track_error_um:.1f} µm")
        print(f"    % positive tangential alignment: {pct_positive_tangential:.1f}%")
        print(f"    Stuck kicks: {stuck_kick_count}")
    
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
    
    print(f"\n  TARGET GATING:")
    print(f"    Target advances: {target_advance_count} ({target_advance_count/(T-1)*100:.1f}%)")
    print(f"    Final target_idx: {target_idx} / {len(path)-1}")
    
    print(f"\n  PERFORMANCE:")
    print(f"    Average solver time: {mean_solver_time:.1f} ms/step")
    if args.controller == "bayes":
        actions_evaluated_list = [log.n_actions_evaluated for log in step_logs]
        mean_actions = np.mean(actions_evaluated_list)
        print(f"    Actions evaluated per step: {mean_actions:.1f} / {len(controller.action_types)} (Bayes K={args.bayes_k})")
        print(f"    Speedup estimate: {len(controller.action_types) / mean_actions:.2f}x")
    else:
        print(f"    Actions evaluated per step: {len(controller.action_types)}")
    
    # Stability stats
    stable_count = sum(1 for log in step_logs if log.trap_stable)
    print(f"\n  TRAP STABILITY (info only - surfing doesn't require stable traps):")
    print(f"    Steps with stable trap: {stable_count}/{len(step_logs)} ({100*stable_count/len(step_logs):.1f}%)")
    
    # ===== Acceptance Tests =====
    print(f"\n{'='*70}")
    print("ACCEPTANCE TESTS")
    print(f"{'='*70}")
    
    test_net_progress = bool(net_progress > 0.1e-3)  # At least 100 µm progress
    test_alignment = bool(pct_positive_alignment > 60.0)  # >60% positive alignment
    
    print(f"  [{'PASS' if test_net_progress else 'FAIL'}] Net progress > 100 µm: {net_progress*1e6:.1f} µm")
    print(f"  [{'PASS' if test_alignment else 'FAIL'}] % positive alignment > 60%: {pct_positive_alignment:.1f}%")
    
    # ===== Save outputs =====
    
    # Compute Bayes-specific metrics
    if args.controller == "bayes":
        actions_evaluated_per_step = [log.n_actions_evaluated for log in step_logs]
        mean_actions_evaluated = float(np.mean(actions_evaluated_per_step))
        speedup_estimate = len(controller.action_types) / mean_actions_evaluated if mean_actions_evaluated > 0 else 1.0
    else:
        mean_actions_evaluated = float(len(controller.action_types))
        speedup_estimate = 1.0
    
    # Save steps.csv
    csv_path = out_dir / "steps.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        # Base columns
        columns = [
            "step_idx", "particle_x", "particle_y", "target_x", "target_y",
            "tracking_error", "chosen_action", "action_switched",
            "Fp_x", "Fp_y", "Fp_mag", "Fp_hat_dot_d", "Fp_dot_d", "score",
            "trap_candidate_x", "trap_candidate_y", "trap_stable", "stiff_min",
            "solver_time_ms", "n_actions_evaluated",
            "target_idx", "dist_to_target", "target_advanced"
        ]
        # Bayes-specific columns
        if args.controller == "bayes":
            columns.extend([
                "controller_mode", "n_actions_total",
                "chosen_action_rank_ucb", "chosen_action_mu",
                "chosen_action_sigma", "chosen_action_ucb"
            ])
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
                log.target_idx, log.dist_to_target, log.target_advanced
            ]
            if args.controller == "bayes":
                row.extend([
                    log.controller_mode, log.n_actions_total,
                    log.chosen_action_rank_ucb, log.chosen_action_mu,
                    log.chosen_action_sigma, log.chosen_action_ucb
                ])
            writer.writerow(row)
    print(f"\nSaved: {csv_path}")
    
    # Save summary JSON
    summary = {
        "path_type": path_name,
        "is_closed_path": is_closed_path,
        "n_steps": T,
        "n_actions": len(controller.action_types),
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
            "pct_trap_stable": float(100 * stable_count / len(step_logs)),
            "mean_solver_time_ms": float(mean_solver_time),
            "target_advance_count": target_advance_count,
            "final_target_idx": target_idx,
            "target_hold_tol_um": args.target_hold_tol_um,
        },
        "acceptance_tests": {
            "net_progress_pass": test_net_progress,
            "alignment_pass": test_alignment,
        }
    }
    
    # Add circle-specific metrics
    if is_circle:
        summary["circle_metrics"] = {
            "angle_progress_deg": float(angle_progress_deg),
            "mean_cross_track_error_um": float(mean_cross_track_error_um),
            "pct_positive_tangential_alignment": float(pct_positive_tangential),
            "stuck_kick_count": stuck_kick_count,
            "circle_dir_mode": args.circle_dir_mode,
            "k_radial": args.k_radial,
        }
    
    # Add Bayes-specific metrics
    if args.controller == "bayes":
        summary["bayes_metrics"] = {
            "controller": "bayes",
            "bayes_k": args.bayes_k,
            "bayes_kappa": args.bayes_kappa,
            "bayes_warmup_steps": args.bayes_warmup_steps,
            "mean_actions_evaluated_per_step": mean_actions_evaluated,
            "n_actions_total": len(controller.action_types),
            "speedup_estimate": speedup_estimate,
        }
    else:
        summary["bayes_metrics"] = {
            "controller": "greedy",
            "mean_actions_evaluated_per_step": mean_actions_evaluated,
            "n_actions_total": len(controller.action_types),
            "speedup_estimate": 1.0,
        }
    
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
    ax.set_title(f'Greedy Surf: {path_name}')
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
    
    # Alignment over time
    ax = axes[1, 0]
    ax.plot(alignments, 'g-', lw=1, alpha=0.7, label='F̂·d̂')
    ax.axhline(0, color='k', linestyle='-', lw=0.5)
    ax.axhline(alignments.mean(), color='r', linestyle='--',
               label=f'mean={alignments.mean():.3f}')
    ax.fill_between(range(len(alignments)), 0, alignments,
                   where=alignments > 0, alpha=0.3, color='green', label='positive')
    ax.fill_between(range(len(alignments)), 0, alignments,
                   where=alignments < 0, alpha=0.3, color='red', label='negative')
    ax.set_xlabel('Step')
    ax.set_ylabel('F̂·d̂ (unit alignment)')
    ax.set_title(f'Force Alignment ({pct_positive_alignment:.1f}% positive)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.1, 1.1)
    
    # Action distribution
    ax = axes[1, 1]
    from collections import Counter
    action_counts = Counter(action_history)
    actions = list(action_counts.keys())
    counts = [action_counts[a] for a in actions]
    y_pos = np.arange(len(actions))
    ax.barh(y_pos, counts, color='steelblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([a[:20] for a in actions], fontsize=8)
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
        gif_path = out_dir / "demo_surf_greedy.gif"
        imageio.mimsave(str(gif_path), images, fps=10, loop=0)
        print(f"Saved: {gif_path}")
    
    print(f"\n{'='*70}")
    print("GREEDY SURF DEMO COMPLETE")
    print(f"{'='*70}")
    print(f"\nOutput directory: {out_dir}")
    print(f"\nKey files:")
    print(f"  GIF:     {out_dir / 'demo_surf_greedy.gif'}")
    print(f"  Summary: {out_dir / 'summary.json'}")
    print(f"  Plot:    {out_dir / 'summary.png'}")
    print(f"  Steps:   {out_dir / 'steps.csv'}")


if __name__ == "__main__":
    main()
