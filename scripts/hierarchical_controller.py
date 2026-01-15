#!/usr/bin/env python3
"""
Reachability-Aware Path Planning for Acoustic Tweezers.

STAGE 4 IMPLEMENTATION:
Hierarchical control structure:
    Path Planner → Macro Actions → Local MPC → Physics

The planner:
1. Validates path feasibility against reachability atlas
2. Breaks path into reachable segments
3. Selects macro actions that move trap in correct direction
4. Uses surrogate to refine magnitude
5. Runs local MPC for refinement

This enables large-scale motion that was previously impossible
with random-shooting MPC alone.

Usage:
    python scripts/hierarchical_controller.py --demo-circle
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import numpy as np
import json

from acousto.force import ParticleProps, gorkov_potential_and_force_2d
from acousto.analysis import find_trap_center

from tweezers.control import (
    DishDomain, MediumProps, EvaluatorConfig,
    Control3Pucks, ControlVector3Pucks, ControlBounds3Pucks, ControlRateLimits3Pucks,
    Evaluator3Pucks, default_3puck_config,
)

from scripts.macro_actions_3puck import (
    MacroActionType3Puck, MacroAction3Puck, 
    apply_macro_action_3puck, measure_action_effect_3puck,
)
from scripts.surrogate_model import GaussianProcessSurrogate


@dataclass
class PathSegment:
    """A segment of the path with reachability info."""
    start_idx: int
    end_idx: int
    is_reachable: bool
    
    # Recommended macro actions for this segment
    macro_actions: list[MacroActionType3Puck] = field(default_factory=list)
    
    # Control adjustments needed
    control_adjustments: list[np.ndarray] = field(default_factory=list)


@dataclass
class HierarchicalControllerConfig:
    """Configuration for hierarchical controller."""
    # Path planning
    segment_length: int = 10  # Steps per segment
    lookahead: int = 5  # Steps to look ahead for macro selection
    
    # Macro action parameters
    macro_magnitude: float = 0.04e-3
    macro_phase_step: float = 0.12
    
    # MPC refinement
    mpc_candidates: int = 30
    mpc_horizon: int = 3
    
    # Switching thresholds
    macro_threshold: float = 0.1e-3  # Switch to MPC when closer than this
    uncertainty_threshold: float = 0.05e-3  # Use macro when surrogate confident
    
    # Smoothing
    control_smoothing: float = 0.3  # Blend factor for control updates
    
    # Fallback objective (Phase 1-D improvement)
    stiffness_threshold: float = -1e-6  # Min eigenvalue for "stable" trap
    use_force_fallback: bool = True  # Fall back to force-to-target when trap weak
    force_fallback_weight: float = 10.0  # Weight for force projection in cost


class ReachabilityAwarePlanner:
    """
    Path planner that uses reachability atlas to validate and plan paths.
    """
    
    def __init__(
        self,
        trap_positions: np.ndarray,  # (N, 2) trap positions from atlas
        control_configs: np.ndarray,  # (N, 12) corresponding controls
        tolerance: float = 0.1e-3,
    ):
        self.trap_positions = trap_positions
        self.control_configs = control_configs
        self.tolerance = tolerance
    
    def validate_path(self, path: np.ndarray) -> tuple[float, np.ndarray]:
        """
        Check what fraction of path is reachable.
        
        Returns (fraction, reachable_mask).
        """
        reachable = np.zeros(len(path), dtype=bool)
        
        for i, (px, py) in enumerate(path):
            distances = np.sqrt(
                (self.trap_positions[:, 0] - px)**2 +
                (self.trap_positions[:, 1] - py)**2
            )
            if np.min(distances) < self.tolerance:
                reachable[i] = True
        
        return np.mean(reachable), reachable
    
    def find_nearest_reachable(self, target_x: float, target_y: float) -> tuple[int, float]:
        """
        Find index of nearest reachable trap position.
        
        Returns (index, distance).
        """
        distances = np.sqrt(
            (self.trap_positions[:, 0] - target_x)**2 +
            (self.trap_positions[:, 1] - target_y)**2
        )
        idx = np.argmin(distances)
        return int(idx), float(distances[idx])
    
    def get_control_for_target(
        self,
        target_x: float,
        target_y: float,
    ) -> tuple[np.ndarray, float]:
        """
        Get recommended control configuration for reaching target.
        
        Returns (control_config, expected_distance).
        """
        idx, dist = self.find_nearest_reachable(target_x, target_y)
        return self.control_configs[idx], dist
    
    def plan_path_segments(
        self,
        path: np.ndarray,
        segment_length: int = 10,
    ) -> list[PathSegment]:
        """
        Break path into segments with reachability info.
        """
        segments = []
        n_points = len(path)
        
        i = 0
        while i < n_points:
            end = min(i + segment_length, n_points)
            
            # Check segment reachability
            segment_path = path[i:end]
            frac, mask = self.validate_path(segment_path)
            
            segment = PathSegment(
                start_idx=i,
                end_idx=end,
                is_reachable=(frac > 0.5),
            )
            
            # Determine direction and recommend macro actions
            if end > i:
                dx = path[end-1, 0] - path[i, 0]
                dy = path[end-1, 1] - path[i, 1]
                
                if abs(dx) > abs(dy):
                    if dx > 0:
                        segment.macro_actions.append(MacroActionType3Puck.TRANSLATE_TRAP_X_POS)
                    else:
                        segment.macro_actions.append(MacroActionType3Puck.TRANSLATE_TRAP_X_NEG)
                else:
                    if dy > 0:
                        segment.macro_actions.append(MacroActionType3Puck.TRANSLATE_TRAP_Y_POS)
                    else:
                        segment.macro_actions.append(MacroActionType3Puck.TRANSLATE_TRAP_Y_NEG)
            
            segments.append(segment)
            i = end
        
        return segments


class HierarchicalController:
    """
    Hierarchical controller combining:
    - Reachability-aware planning
    - Macro action selection
    - Surrogate-guided search
    - Local MPC refinement
    """
    
    def __init__(
        self,
        evaluator: Evaluator3Pucks,
        bounds: ControlBounds3Pucks,
        rate_limits: ControlRateLimits3Pucks,
        config: HierarchicalControllerConfig,
        *,
        planner: Optional[ReachabilityAwarePlanner] = None,
        surrogate: Optional[GaussianProcessSurrogate] = None,
    ):
        self.ev = evaluator
        self.bounds = bounds
        self.rate_limits = rate_limits
        self.cfg = config
        self.planner = planner
        self.surrogate = surrogate
        
        self.rng = np.random.default_rng(42)
        
        # State tracking
        self.current_segment_idx = 0
        self.steps_in_segment = 0
        
        # Control history for smoothing
        self.control_history: list[np.ndarray] = []
        self.prev_macro_action: Optional[MacroActionType3Puck] = None
    
    def _select_macro_action(
        self,
        current_ctrl: Control3Pucks,
        trap_x: float,
        trap_y: float,
        target_x: float,
        target_y: float,
    ) -> MacroActionType3Puck:
        """
        Select best macro action based on target direction and surrogate.
        """
        dx = target_x - trap_x
        dy = target_y - trap_y
        dist = np.sqrt(dx**2 + dy**2)
        
        if dist < 1e-9:
            return MacroActionType3Puck.HOLD
        
        # Normalize direction
        dx_norm = dx / dist
        dy_norm = dy / dist
        
        # Score each candidate action
        candidate_actions = [
            MacroActionType3Puck.TRANSLATE_TRAP_X_POS,
            MacroActionType3Puck.TRANSLATE_TRAP_X_NEG,
            MacroActionType3Puck.TRANSLATE_TRAP_Y_POS,
            MacroActionType3Puck.TRANSLATE_TRAP_Y_NEG,
            MacroActionType3Puck.MOVE_A_RIGHT,
            MacroActionType3Puck.MOVE_A_LEFT,
            MacroActionType3Puck.MOVE_B_RIGHT,
            MacroActionType3Puck.MOVE_B_LEFT,
            MacroActionType3Puck.MOVE_C_UP,
            MacroActionType3Puck.MOVE_C_DOWN,
            MacroActionType3Puck.PHASE_SHIFT_B_POS,
            MacroActionType3Puck.PHASE_SHIFT_B_NEG,
        ]
        
        best_action = MacroActionType3Puck.HOLD
        best_score = -np.inf
        
        for action_type in candidate_actions:
            action = MacroAction3Puck(
                action_type=action_type,
                magnitude=self.cfg.macro_magnitude,
                phase_step=self.cfg.macro_phase_step,
            )
            
            # Apply action to get new control
            u_new = apply_macro_action_3puck(current_ctrl, action)
            
            # If we have surrogate, use it to predict effect
            if self.surrogate is not None and self.surrogate.is_trained:
                ctrl_arr = np.array([
                    current_ctrl.xA, current_ctrl.yA, current_ctrl.vA, current_ctrl.phiA,
                    current_ctrl.xB, current_ctrl.yB, current_ctrl.vB, current_ctrl.phiB,
                    current_ctrl.xC, current_ctrl.yC, current_ctrl.vC, current_ctrl.phiC,
                ])
                delta_arr = np.array([
                    u_new.xA - current_ctrl.xA, u_new.yA - current_ctrl.yA,
                    u_new.vA - current_ctrl.vA, u_new.phiA - current_ctrl.phiA,
                    u_new.xB - current_ctrl.xB, u_new.yB - current_ctrl.yB,
                    u_new.vB - current_ctrl.vB, u_new.phiB - current_ctrl.phiB,
                    u_new.xC - current_ctrl.xC, u_new.yC - current_ctrl.yC,
                    u_new.vC - current_ctrl.vC, u_new.phiC - current_ctrl.phiC,
                ])
                
                pred_delta, uncertainty = self.surrogate.predict(ctrl_arr, delta_arr)
                
                # Score = dot product with target direction
                score = pred_delta[0] * dx_norm + pred_delta[1] * dy_norm
                
                # Penalize high uncertainty
                score -= np.mean(uncertainty) * 1000
            else:
                # Heuristic scoring based on action type
                if action_type == MacroActionType3Puck.TRANSLATE_TRAP_X_POS:
                    score = dx_norm * 0.8
                elif action_type == MacroActionType3Puck.TRANSLATE_TRAP_X_NEG:
                    score = -dx_norm * 0.8
                elif action_type == MacroActionType3Puck.TRANSLATE_TRAP_Y_POS:
                    score = dy_norm * 0.6
                elif action_type == MacroActionType3Puck.TRANSLATE_TRAP_Y_NEG:
                    score = -dy_norm * 0.6
                elif action_type == MacroActionType3Puck.MOVE_C_UP:
                    score = dy_norm * 0.4
                elif action_type == MacroActionType3Puck.MOVE_C_DOWN:
                    score = -dy_norm * 0.4
                else:
                    score = 0.0
            
            if score > best_score:
                best_score = score
                best_action = action_type
        
        return best_action
    
    def _apply_macro_action(
        self,
        current_ctrl: Control3Pucks,
        action_type: MacroActionType3Puck,
    ) -> Control3Pucks:
        """Apply macro action with smoothing."""
        action = MacroAction3Puck(
            action_type=action_type,
            magnitude=self.cfg.macro_magnitude,
            phase_step=self.cfg.macro_phase_step,
        )
        
        u_new = apply_macro_action_3puck(current_ctrl, action)
        u_new = self.ev.clip_control(u_new)
        
        return u_new
    
    def _local_mpc_refine(
        self,
        particle_x: float,
        particle_y: float,
        target_x: float,
        target_y: float,
        current_ctrl: Control3Pucks,
        trap_is_weak: bool = False,
    ) -> tuple[Control3Pucks, dict]:
        """
        Local MPC refinement around current control.
        
        Uses smaller noise and fewer candidates for fine-tuning.
        
        When trap_is_weak=True, uses force-to-target fallback objective
        instead of trap-based objective.
        
        Returns: (best_control, mpc_info)
        """
        best_ctrl = current_ctrl
        best_cost = float("inf")
        
        candidate_costs = []
        bounds_clipped = {"x": False, "y": False, "v": False}
        
        base_arr = np.array([
            current_ctrl.xA, current_ctrl.yA, current_ctrl.vA, current_ctrl.phiA,
            current_ctrl.xB, current_ctrl.yB, current_ctrl.vB, current_ctrl.phiB,
            current_ctrl.xC, current_ctrl.yC, current_ctrl.vC, current_ctrl.phiC,
        ])
        
        # Small perturbations
        noise_scale = np.array([
            0.01e-3, 0.005e-3, 0.002, 0.05,  # A
            0.01e-3, 0.005e-3, 0.002, 0.05,  # B
            0.01e-3, 0.01e-3, 0.002, 0.05,   # C
        ])
        
        # Direction to target
        dx_target = target_x - particle_x
        dy_target = target_y - particle_y
        dist_to_target = np.sqrt(dx_target**2 + dy_target**2)
        if dist_to_target > 1e-9:
            dir_x = dx_target / dist_to_target
            dir_y = dy_target / dist_to_target
        else:
            dir_x, dir_y = 0.0, 0.0
        
        for i in range(self.cfg.mpc_candidates):
            noise = self.rng.normal(size=12) * noise_scale
            perturbed = base_arr + noise
            
            u_test = Control3Pucks(
                xA=perturbed[0], yA=perturbed[1], vA=perturbed[2], phiA=perturbed[3],
                xB=perturbed[4], yB=perturbed[5], vB=perturbed[6], phiB=perturbed[7],
                xC=perturbed[8], yC=perturbed[9], vC=perturbed[10], phiC=perturbed[11],
            )
            u_clipped = self.ev.clip_control(u_test)
            
            # Track if bounds were clipped
            clipped_arr = np.array([
                u_clipped.xA, u_clipped.yA, u_clipped.vA, u_clipped.phiA,
                u_clipped.xB, u_clipped.yB, u_clipped.vB, u_clipped.phiB,
                u_clipped.xC, u_clipped.yC, u_clipped.vC, u_clipped.phiC,
            ])
            if np.any(np.abs(clipped_arr[[0,4,8]] - perturbed[[0,4,8]]) > 1e-9):
                bounds_clipped["x"] = True
            if np.any(np.abs(clipped_arr[[1,5,9]] - perturbed[[1,5,9]]) > 1e-9):
                bounds_clipped["y"] = True
            if np.any(np.abs(clipped_arr[[2,6,10]] - perturbed[[2,6,10]]) > 1e-9):
                bounds_clipped["v"] = True
            
            u_test = u_clipped
            
            # Evaluate
            xp1, yp1, loss, info = self.ev.step(
                xp=particle_x, yp=particle_y,
                target_x=target_x, target_y=target_y,
                u=u_test,
            )
            
            # Find trap
            trap = self.ev.find_trap(u_test, xp1, yp1, search_radius=0.4e-3)
            
            # Compute cost based on trap quality
            if trap_is_weak or not trap.is_stable or (
                trap.stiffness_eigvals is not None and 
                np.min(trap.stiffness_eigvals) > self.cfg.stiffness_threshold
            ):
                # FALLBACK OBJECTIVE: optimize force projection toward target
                # This is used when trap is weak/unstable
                if self.cfg.use_force_fallback:
                    # Get force at particle location
                    vb = self.ev.control_to_forcing_band_vb(u_test)
                    field = self.ev.op.solve_for_bottom_vb(vb)
                    U, Fx, Fy = gorkov_potential_and_force_2d(field, self.ev.particle)
                    
                    # Interpolate force at particle position
                    ix = int(np.clip((xp1 - field.x[0]) / (field.x[1] - field.x[0]), 0, len(field.x) - 1))
                    iy = int(np.clip((yp1 - field.y[0]) / (field.y[1] - field.y[0]), 0, len(field.y) - 1))
                    
                    force_x = Fx[iy, ix]
                    force_y = Fy[iy, ix]
                    
                    # Force magnitude
                    force_mag = np.sqrt(force_x**2 + force_y**2) + 1e-15
                    
                    # Projection toward target (maximize dot product)
                    force_projection = (force_x * dir_x + force_y * dir_y) / force_mag
                    
                    # Cost = negative projection (we want to maximize projection)
                    force_cost = -self.cfg.force_fallback_weight * force_projection
                    
                    # Also penalize particle-to-target distance
                    particle_cost = (xp1 - target_x)**2 + (yp1 - target_y)**2
                    
                    total_cost = particle_cost + force_cost
                else:
                    # Just use particle distance if fallback disabled
                    total_cost = (xp1 - target_x)**2 + (yp1 - target_y)**2
            else:
                # NORMAL OBJECTIVE: trap-based + particle-based
                particle_cost = (xp1 - target_x)**2 + (yp1 - target_y)**2
                trap_cost = (trap.x - target_x)**2 + (trap.y - target_y)**2
                total_cost = particle_cost + 2.0 * trap_cost
            
            candidate_costs.append(total_cost)
            
            if total_cost < best_cost:
                best_cost = total_cost
                best_ctrl = u_test
        
        mpc_info = {
            "candidate_costs": candidate_costs,
            "chosen_idx": int(np.argmin(candidate_costs)) if candidate_costs else -1,
            "bounds_clipped_x": bounds_clipped["x"],
            "bounds_clipped_y": bounds_clipped["y"],
            "bounds_clipped_v": bounds_clipped["v"],
            "used_force_fallback": trap_is_weak or not trap.is_stable,
        }
        
        return best_ctrl, mpc_info
    
    def _smooth_control(
        self,
        new_ctrl: Control3Pucks,
    ) -> Control3Pucks:
        """Apply temporal smoothing to control."""
        if len(self.control_history) == 0:
            return new_ctrl
        
        prev_arr = self.control_history[-1]
        new_arr = np.array([
            new_ctrl.xA, new_ctrl.yA, new_ctrl.vA, new_ctrl.phiA,
            new_ctrl.xB, new_ctrl.yB, new_ctrl.vB, new_ctrl.phiB,
            new_ctrl.xC, new_ctrl.yC, new_ctrl.vC, new_ctrl.phiC,
        ])
        
        # Blend
        alpha = self.cfg.control_smoothing
        smoothed = alpha * new_arr + (1 - alpha) * prev_arr
        
        return Control3Pucks(
            xA=smoothed[0], yA=smoothed[1], vA=smoothed[2], phiA=smoothed[3],
            xB=smoothed[4], yB=smoothed[5], vB=smoothed[6], phiB=smoothed[7],
            xC=smoothed[8], yC=smoothed[9], vC=smoothed[10], phiC=smoothed[11],
        )
    
    def step(
        self,
        particle_x: float,
        particle_y: float,
        target_x: float,
        target_y: float,
        current_ctrl: Control3Pucks,
    ) -> tuple[Control3Pucks, float, float, dict]:
        """
        One hierarchical control step.
        
        Returns: (new_control, new_x, new_y, info)
        """
        # Find current trap
        trap_result = self.ev.find_trap(current_ctrl, particle_x, particle_y)
        trap_x = trap_result.x if trap_result.is_stable else particle_x
        trap_y = trap_result.y if trap_result.is_stable else particle_y
        trap_is_weak = not trap_result.is_stable
        
        # Check stiffness threshold
        if trap_result.is_stable and trap_result.stiffness_eigvals is not None:
            stiff_min = np.min(trap_result.stiffness_eigvals)
            if stiff_min > self.cfg.stiffness_threshold:
                trap_is_weak = True  # Positive eigenvalue = unstable
        else:
            stiff_min = np.nan
        
        # Distance to target
        dist_to_target = np.sqrt((trap_x - target_x)**2 + (trap_y - target_y)**2)
        
        # Decide: macro action or local MPC
        info = {
            "mode": "macro",
            "trap_stable": trap_result.is_stable,
            "trap_weak": trap_is_weak,
            "stiffness": trap_result.stiffness_eigvals if trap_result.is_stable else None,
        }
        
        mpc_info = {}
        
        if dist_to_target > self.cfg.macro_threshold:
            # Far from target: use macro action
            action_type = self._select_macro_action(
                current_ctrl, trap_x, trap_y, target_x, target_y
            )
            new_ctrl = self._apply_macro_action(current_ctrl, action_type)
            info["macro_action"] = action_type.name
        else:
            # Close to target: use local MPC
            new_ctrl, mpc_info = self._local_mpc_refine(
                particle_x, particle_y, target_x, target_y, current_ctrl,
                trap_is_weak=trap_is_weak,
            )
            info["mode"] = "mpc"
            info.update(mpc_info)
        
        # Apply smoothing
        new_ctrl = self._smooth_control(new_ctrl)
        
        # Check if rate limited
        rate_limited = False
        if len(self.control_history) > 0:
            prev_arr = self.control_history[-1]
            new_arr = np.array([
                new_ctrl.xA, new_ctrl.yA, new_ctrl.vA, new_ctrl.phiA,
                new_ctrl.xB, new_ctrl.yB, new_ctrl.vB, new_ctrl.phiB,
                new_ctrl.xC, new_ctrl.yC, new_ctrl.vC, new_ctrl.phiC,
            ])
            delta = np.abs(new_arr - prev_arr)
            # Check against rate limits if defined
            if self.rate_limits is not None:
                if np.any(delta[[0,4,8]] >= self.rate_limits.dx_max * 0.99):
                    rate_limited = True
                if np.any(delta[[1,5,9]] >= self.rate_limits.dy_max * 0.99):
                    rate_limited = True
        
        info["rate_limited"] = rate_limited
        
        # Simulate step (with return_metrics=True, step_info will include field metrics)
        xp1, yp1, loss, step_info = self.ev.step(
            xp=particle_x, yp=particle_y,
            target_x=target_x, target_y=target_y,
            u=new_ctrl,
            return_metrics=True,  # Ensure we get scalar field metrics
        )
        
        # Check if max_step clipped the motion
        max_step_clipped = step_info.get("step_limited", False)
        info["max_step_clipped"] = max_step_clipped
        
        # Pass through metrics from evaluator (critical for flight recorder)
        if "metrics" in step_info:
            info["metrics"] = step_info["metrics"]
        
        # Update history
        ctrl_arr = np.array([
            new_ctrl.xA, new_ctrl.yA, new_ctrl.vA, new_ctrl.phiA,
            new_ctrl.xB, new_ctrl.yB, new_ctrl.vB, new_ctrl.phiB,
            new_ctrl.xC, new_ctrl.yC, new_ctrl.vC, new_ctrl.phiC,
        ])
        self.control_history.append(ctrl_arr)
        if len(self.control_history) > 10:
            self.control_history.pop(0)
        
        # Update surrogate online if available
        if self.surrogate is not None and len(self.control_history) >= 2:
            prev_ctrl_arr = self.control_history[-2]
            delta_arr = ctrl_arr - prev_ctrl_arr
            
            # Find new trap
            trap_new = self.ev.find_trap(new_ctrl, xp1, yp1)
            if trap_new.is_stable and trap_result.is_stable:
                delta_trap = np.array([trap_new.x - trap_x, trap_new.y - trap_y])
                self.surrogate.update_online(prev_ctrl_arr, delta_arr, delta_trap)
        
        info.update({
            "trap_x": trap_result.x if trap_result.is_stable else np.nan,
            "trap_y": trap_result.y if trap_result.is_stable else np.nan,
            "displacement": step_info.get("displacement", 0.0),
            "stiff_min": stiff_min,
        })
        
        return new_ctrl, xp1, yp1, info


def load_reachability_data(path: Path) -> ReachabilityAwarePlanner:
    """Load reachability atlas for planner."""
    trap_positions = np.load(path / "trap_positions.npy")
    control_configs = np.load(path / "control_configs.npy")
    
    return ReachabilityAwarePlanner(
        trap_positions=trap_positions,
        control_configs=control_configs,
        tolerance=0.1e-3,
    )


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-circle", action="store_true")
    parser.add_argument("--atlas", type=str, default="results/reachability_3puck")
    parser.add_argument("--surrogate", type=str, default="results/surrogate_model")
    args = parser.parse_args()
    
    print("=" * 60)
    print("HIERARCHICAL CONTROLLER")
    print("=" * 60)
    
    # Setup physics
    domain = DishDomain(Lx=2e-3, Ly=2e-3, Nx=100, Ny=100)
    medium = MediumProps(f=2e6, c0=1500.0, rho0=1000.0, loss_eta=1e-3, kz=0.0, coupling_alpha=1.0)
    particle = ParticleProps(a=5e-6, rho_p=1050.0, c_p=2350.0)
    
    cfg = EvaluatorConfig(
        sigma_x=0.10e-3, sigma_y=0.15e-3, bottom_band=0.25e-3,
        dt=5e-3, viscosity=1e-3, alpha_g=2e3, max_step=0.08e-3,
        use_2d_forcing=True,
    )
    
    ev = Evaluator3Pucks(domain, medium, particle, cfg)
    
    bounds = ControlBounds3Pucks(
        x_min=0.0, x_max=domain.Lx,
        y_min=0.0, y_max=cfg.bottom_band,
        y_max_C=domain.Ly * 0.5,
        v_min=0.0, v_max=0.2,
    )
    
    rate_limits = ControlRateLimits3Pucks(
        dx_max=0.08e-3, dy_max=0.05e-3, dv_max=0.015, dphi_max=0.4,
    )
    
    # Try to load planner and surrogate
    planner = None
    surrogate = None
    
    atlas_path = Path(args.atlas)
    if atlas_path.exists():
        print(f"Loading reachability atlas from {atlas_path}")
        planner = load_reachability_data(atlas_path)
        print(f"  Loaded {len(planner.trap_positions)} trap positions")
    
    surrogate_path = Path(args.surrogate)
    if (surrogate_path / "gp_x.joblib").exists():
        print(f"Loading surrogate from {surrogate_path}")
        surrogate = GaussianProcessSurrogate()
        surrogate.load(surrogate_path)
    
    # Create controller
    hc_config = HierarchicalControllerConfig(
        segment_length=10,
        macro_magnitude=0.04e-3,
        macro_phase_step=0.12,
        mpc_candidates=30,
        macro_threshold=0.08e-3,
        control_smoothing=0.3,
    )
    
    controller = HierarchicalController(
        evaluator=ev,
        bounds=bounds,
        rate_limits=rate_limits,
        config=hc_config,
        planner=planner,
        surrogate=surrogate,
    )
    
    if args.demo_circle:
        print("\nRunning circle demo with hierarchical control...")
        
        # Circle path (60% of domain)
        T = 200
        radius = 0.6 * min(domain.Lx, domain.Ly) / 2
        center_x = domain.Lx / 2
        center_y = domain.Ly * 0.55
        
        theta = np.linspace(0, 2 * np.pi, T)
        path = np.column_stack([
            center_x + radius * np.cos(theta),
            center_y + radius * np.sin(theta),
        ])
        
        print(f"Circle: center=({center_x*1e3:.2f}, {center_y*1e3:.2f}) mm, "
              f"radius={radius*1e3:.2f} mm")
        
        # Initial state
        particle_x = float(path[0, 0])
        particle_y = float(path[0, 1])
        
        ctrl = Control3Pucks(
            xA=0.4e-3, yA=0.03e-3, vA=0.08, phiA=0.0,
            xB=1.6e-3, yB=0.03e-3, vB=0.08, phiB=np.pi,
            xC=1.0e-3, yC=0.15e-3, vC=0.08, phiC=np.pi/2,
        )
        
        errors = []
        
        print(f"\n{'Step':>5} {'px':>8} {'py':>8} {'err':>8} {'mode':>8}")
        print("-" * 50)
        
        for t in range(T - 1):
            target_x = float(path[t + 1, 0])
            target_y = float(path[t + 1, 1])
            
            ctrl, particle_x, particle_y, info = controller.step(
                particle_x, particle_y, target_x, target_y, ctrl
            )
            
            err = np.sqrt((particle_x - target_x)**2 + (particle_y - target_y)**2)
            errors.append(err * 1e3)
            
            if t % 20 == 0:
                print(f"{t:5d} {particle_x*1e3:8.3f} {particle_y*1e3:8.3f} "
                      f"{err*1e3:8.4f} {info['mode']:>8}")
        
        print("-" * 50)
        print(f"\nMean error: {np.mean(errors):.4f} mm")
        print(f"Max error:  {np.max(errors):.4f} mm")
    
    else:
        print("Run with --demo-circle for demonstration")


if __name__ == "__main__":
    main()
