"""
Smooth MPC Controller with anti-jitter mechanisms.

FIXES IMPLEMENTED:
1. Warm-start MPC from previous best sequence
2. Control smoothing penalty (penalize ΔΔu jitter)
3. Cross-Entropy Method (CEM) for candidate generation
4. Transducer motion prior (penalize sign reversals)
5. Reference-guided search (constrain candidates around guided control)
6. Sequential rate limiting in rollouts

Author: Acoustic Tweezers Team
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Callable
import numpy as np

from .controller import (
    ControlVector, ControlState, ControlBounds, ControlRateLimits,
)
from .evaluator import Control2Pucks, BottomFootprint25DEvaluator


@dataclass
class SmoothMPCConfig:
    """Configuration for smooth MPC controller."""
    # Horizon and candidates
    horizon: int = 4
    n_candidates: int = 60
    n_elite: int = 10  # Top candidates for CEM
    cem_iterations: int = 2  # CEM refinement iterations
    
    # Cost weights
    tracking_weight: float = 1e6
    trap_weight: float = 2e6
    particle_trap_weight: float = 0.5e6
    effort_weight: float = 0.001
    
    # Anti-jitter weights (NEW)
    jitter_weight: float = 1e4  # Penalize ΔΔu (second derivative)
    sign_reversal_weight: float = 5e3  # Penalize direction reversals
    reference_weight: float = 0.1  # Soft constraint toward guided control
    
    # Noise scales for CEM
    position_noise_init: float = 0.08e-3
    amplitude_noise_init: float = 0.005
    phase_noise_init: float = 0.3
    noise_decay: float = 0.7  # Decay factor per CEM iteration
    
    # Physics
    dt: float = 5e-3
    viscosity: float = 1e-3
    particle_radius: float = 5e-6


@dataclass
class ControlHistory:
    """Tracks control history for jitter detection and warm-start."""
    controls: list[ControlVector] = field(default_factory=list)
    max_history: int = 10
    
    # Warm-start: previous best MPC sequence
    prev_best_sequence: list[ControlVector] = field(default_factory=list)
    
    def add(self, ctrl: ControlVector) -> None:
        self.controls.append(ctrl)
        if len(self.controls) > self.max_history:
            self.controls.pop(0)
    
    def get_prev(self, n: int = 1) -> Optional[ControlVector]:
        """Get control from n steps ago."""
        if len(self.controls) >= n:
            return self.controls[-n]
        return None
    
    def compute_delta(self) -> Optional[np.ndarray]:
        """Compute Δu = u(t) - u(t-1)."""
        if len(self.controls) >= 2:
            return self.controls[-1].to_array() - self.controls[-2].to_array()
        return None
    
    def compute_jitter(self) -> Optional[np.ndarray]:
        """Compute ΔΔu = Δu(t) - Δu(t-1) = u(t) - 2*u(t-1) + u(t-2)."""
        if len(self.controls) >= 3:
            u0 = self.controls[-3].to_array()
            u1 = self.controls[-2].to_array()
            u2 = self.controls[-1].to_array()
            return u2 - 2*u1 + u0
        return None
    
    def has_sign_reversal(self, proposed_delta: np.ndarray, dims: list[int]) -> np.ndarray:
        """Check if proposed Δu reverses sign from previous Δu for given dimensions."""
        prev_delta = self.compute_delta()
        if prev_delta is None:
            return np.zeros(len(dims), dtype=bool)
        
        reversals = np.zeros(len(dims), dtype=bool)
        for i, d in enumerate(dims):
            if prev_delta[d] * proposed_delta[d] < 0:  # Sign change
                reversals[i] = True
        return reversals


class CEMCandidateGenerator:
    """
    Cross-Entropy Method for candidate generation.
    
    Instead of pure Gaussian noise, CEM:
    1. Samples candidates from current distribution
    2. Evaluates all candidates
    3. Fits new distribution to elite (top-k) candidates
    4. Repeats for refinement
    """
    
    def __init__(
        self,
        config: SmoothMPCConfig,
        bounds: ControlBounds,
        rate_limits: ControlRateLimits,
        rng: np.random.Generator,
    ):
        self.cfg = config
        self.bounds = bounds
        self.rate_limits = rate_limits
        self.rng = rng
        
        # Distribution parameters (mean, std per dimension)
        self.n_dims = 8  # For 2-puck
        self.reset_distribution()
    
    def reset_distribution(self, center: Optional[np.ndarray] = None):
        """Reset to initial wide distribution."""
        self.mean = center if center is not None else np.zeros(self.n_dims)
        self.std = np.array([
            self.cfg.position_noise_init,  # xA
            self.cfg.position_noise_init,  # yA
            self.cfg.position_noise_init,  # xB
            self.cfg.position_noise_init,  # yB
            self.cfg.amplitude_noise_init,  # vA
            self.cfg.amplitude_noise_init,  # vB
            self.cfg.phase_noise_init,  # phiA
            self.cfg.phase_noise_init,  # phiB
        ])
    
    def sample_candidates(
        self,
        base_control: ControlVector,
        reference_control: Optional[ControlVector],
        prev_control: Optional[ControlVector],
        n: int,
    ) -> list[ControlVector]:
        """
        Sample n candidates around base_control.
        
        If reference_control is provided, bias sampling toward it.
        """
        base_arr = base_control.to_array()
        candidates: list[ControlVector] = []
        
        # Always include base control
        candidates.append(base_control)
        
        # Include reference control if provided
        if reference_control is not None:
            candidates.append(reference_control)
        
        # Include warm-start shifted sequence if available
        # (handled externally)
        
        # Sample remaining from Gaussian
        n_to_sample = n - len(candidates)
        
        # Blend center toward reference if provided
        center = base_arr.copy()
        if reference_control is not None:
            ref_arr = reference_control.to_array()
            blend = 0.3  # 30% toward reference
            center = (1 - blend) * base_arr + blend * ref_arr
        
        for _ in range(n_to_sample):
            noise = self.rng.normal(scale=self.std)
            perturbed = center + noise
            
            ctrl = ControlVector.from_array(perturbed, self.bounds, self.rate_limits)
            ctrl = ctrl.clamp_to_bounds()
            if prev_control is not None:
                ctrl = ctrl.apply_rate_limits(prev_control)
            candidates.append(ctrl)
        
        return candidates
    
    def update_distribution(self, elite_controls: list[ControlVector]):
        """Update distribution based on elite samples."""
        if len(elite_controls) < 2:
            return
        
        elite_arrays = np.array([c.to_array() for c in elite_controls])
        new_mean = np.mean(elite_arrays, axis=0)
        new_std = np.std(elite_arrays, axis=0) + 1e-8  # Prevent collapse
        
        # Smooth update
        alpha = 0.5
        self.mean = alpha * new_mean + (1 - alpha) * self.mean
        self.std = alpha * new_std + (1 - alpha) * self.std
        
        # Apply decay
        self.std *= self.cfg.noise_decay
        
        # Minimum std to prevent collapse
        min_std = np.array([1e-5, 1e-5, 1e-5, 1e-5, 1e-6, 1e-6, 0.01, 0.01])
        self.std = np.maximum(self.std, min_std)


def compute_jitter_cost(
    proposed_control: ControlVector,
    history: ControlHistory,
    config: SmoothMPCConfig,
) -> float:
    """
    Compute jitter penalty for proposed control.
    
    Penalizes:
    1. Large ΔΔu (second derivative of control)
    2. Sign reversals in position changes
    """
    cost = 0.0
    
    prev = history.get_prev(1)
    prev2 = history.get_prev(2)
    
    if prev is None:
        return 0.0
    
    proposed_arr = proposed_control.to_array()
    prev_arr = prev.to_array()
    
    # Jitter cost: ||u - 2*u_prev + u_prev2||^2
    if prev2 is not None:
        prev2_arr = prev2.to_array()
        jitter = proposed_arr - 2 * prev_arr + prev2_arr
        
        # Weight position dimensions more (indices 0-3)
        weights = np.array([1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1])
        cost += config.jitter_weight * np.sum(weights * jitter**2)
    
    # Sign reversal cost for position dimensions
    delta = proposed_arr - prev_arr
    prev_delta = history.compute_delta()
    
    if prev_delta is not None:
        position_dims = [0, 1, 2, 3]  # xA, yA, xB, yB
        for d in position_dims:
            if delta[d] * prev_delta[d] < 0:  # Sign reversal
                # Penalty proportional to magnitude of reversal
                reversal_magnitude = abs(delta[d]) + abs(prev_delta[d])
                cost += config.sign_reversal_weight * reversal_magnitude
    
    return cost


def compute_reference_cost(
    proposed_control: ControlVector,
    reference_control: Optional[ControlVector],
    config: SmoothMPCConfig,
) -> float:
    """
    Soft penalty for deviating from reference (guided) control.
    
    This encourages the controller to stay near the "ideal" geometry
    while still allowing optimization.
    """
    if reference_control is None:
        return 0.0
    
    proposed_arr = proposed_control.to_array()
    ref_arr = reference_control.to_array()
    
    diff = proposed_arr - ref_arr
    
    # Weight position dimensions more
    weights = np.array([1.0, 0.5, 1.0, 0.5, 0.01, 0.01, 0.01, 0.01])
    
    return config.reference_weight * np.sum(weights * diff**2)


class SmoothMPCController:
    """
    MPC Controller with anti-jitter mechanisms.
    
    Key improvements over basic random shooting:
    1. Warm-start from previous best sequence
    2. CEM-style candidate refinement
    3. Jitter penalty in cost function
    4. Reference-guided search
    5. Sequential rate limiting in rollouts
    """
    
    def __init__(
        self,
        evaluator,  # BottomFootprint25DEvaluator
        config: SmoothMPCConfig,
        bounds: ControlBounds,
        rate_limits: ControlRateLimits,
        seed: int = 42,
    ):
        self.evaluator = evaluator
        self.cfg = config
        self.bounds = bounds
        self.rate_limits = rate_limits
        self.rng = np.random.default_rng(seed)
        
        self.history = ControlHistory()
        self.candidate_generator = CEMCandidateGenerator(
            config, bounds, rate_limits, self.rng
        )
        
        # Logging
        self.step_logs: list[dict] = []
    
    def predict_motion(
        self,
        state: ControlState,
        control: ControlVector,
    ) -> tuple[ControlState, dict]:
        """Predict next state using evaluator."""
        u2p = control.to_control2pucks()
        xp1, yp1, _, info = self.evaluator.step(
            xp=state.x, yp=state.y,
            target_x=state.x, target_y=state.y,  # Target doesn't matter for physics
            u=u2p, u_prev=None,
        )
        return ControlState(x=xp1, y=yp1), info
    
    def evaluate_cost(
        self,
        predicted_state: ControlState,
        target: ControlState,
        control: ControlVector,
        eval_info: dict,
        reference_control: Optional[ControlVector] = None,
    ) -> float:
        """
        Compute cost for a predicted state.
        
        Includes tracking, trap-steering, effort, jitter, and reference costs.
        """
        # Tracking cost: particle to target
        dx = predicted_state.x - target.x
        dy = predicted_state.y - target.y
        tracking_cost = self.cfg.tracking_weight * (dx**2 + dy**2)
        
        # Trap-steering cost: trap centre to target
        trap_xy = eval_info.get("trap_xy")
        trap_cost = 0.0
        if trap_xy is not None:
            trap_x, trap_y = trap_xy
            trap_dx = trap_x - target.x
            trap_dy = trap_y - target.y
            trap_cost = self.cfg.trap_weight * (trap_dx**2 + trap_dy**2)
        
        # Particle-trap cost: keep particle near trap
        particle_trap_cost = 0.0
        if trap_xy is not None:
            trap_x, trap_y = trap_xy
            pt_dx = predicted_state.x - trap_x
            pt_dy = predicted_state.y - trap_y
            particle_trap_cost = self.cfg.particle_trap_weight * (pt_dx**2 + pt_dy**2)
        
        # Effort cost
        effort_cost = self.cfg.effort_weight * (control.vA**2 + control.vB**2)
        
        # Jitter cost (uses history)
        jitter_cost = compute_jitter_cost(control, self.history, self.cfg)
        
        # Reference cost
        ref_cost = compute_reference_cost(control, reference_control, self.cfg)
        
        total = tracking_cost + trap_cost + particle_trap_cost + effort_cost + jitter_cost + ref_cost
        
        return total
    
    def step(
        self,
        state: ControlState,
        target: ControlState,
        current_control: ControlVector,
        targets_horizon: Optional[list[ControlState]] = None,
        reference_control: Optional[ControlVector] = None,
    ) -> tuple[ControlVector, ControlState, dict]:
        """
        Perform one MPC step with anti-jitter mechanisms.
        
        Parameters
        ----------
        state : ControlState
            Current particle position
        target : ControlState
            Target position for this step
        current_control : ControlVector
            Current control (starting point for search)
        targets_horizon : list[ControlState], optional
            Future targets for MPC horizon
        reference_control : ControlVector, optional
            "Ideal" control geometry (e.g., guided straddle)
        
        Returns
        -------
        best_control, new_state, info
        """
        H = self.cfg.horizon
        if targets_horizon is None:
            targets_horizon = [target] * H
        H = min(H, len(targets_horizon))
        
        prev_control = self.history.get_prev(1)
        
        # Reset candidate generator for this step
        self.candidate_generator.reset_distribution(current_control.to_array())
        
        best_control = current_control
        best_cost = float("inf")
        best_state = state
        best_info = {}
        
        candidate_costs: list[tuple[float, ControlVector]] = []
        
        # CEM iterations
        for cem_iter in range(self.cfg.cem_iterations):
            # Generate candidates
            candidates = self.candidate_generator.sample_candidates(
                current_control, reference_control, prev_control,
                self.cfg.n_candidates,
            )
            
            # Add warm-start candidates from previous best sequence
            if cem_iter == 0 and len(self.history.prev_best_sequence) > 0:
                # Shift sequence: drop first, use rest
                for i, ctrl in enumerate(self.history.prev_best_sequence[1:H+1]):
                    if ctrl is not None:
                        candidates.append(ctrl)
            
            # Evaluate each candidate
            candidate_costs.clear()
            
            for ctrl in candidates:
                # Apply rate limits vs previous control
                if prev_control is not None:
                    ctrl = ctrl.apply_rate_limits(prev_control)
                ctrl = ctrl.clamp_to_bounds()
                
                # Simulate horizon
                total_cost = 0.0
                sim_state = state
                seq_prev_ctrl = prev_control
                is_feasible = True
                first_predicted = None
                first_info = {}
                
                # Build sequence for this candidate
                sequence = [ctrl]
                seq_ctrl = ctrl
                for h in range(1, H):
                    # For future steps, perturb from previous in sequence
                    noise = self.rng.normal(scale=self.candidate_generator.std * 0.5)
                    perturbed = seq_ctrl.to_array() + noise
                    next_ctrl = ControlVector.from_array(perturbed, self.bounds, self.rate_limits)
                    next_ctrl = next_ctrl.clamp_to_bounds()
                    if seq_prev_ctrl is not None:
                        next_ctrl = next_ctrl.apply_rate_limits(seq_prev_ctrl)
                    sequence.append(next_ctrl)
                    seq_prev_ctrl = seq_ctrl
                    seq_ctrl = next_ctrl
                
                # Evaluate sequence
                seq_prev = prev_control
                for h, seq_ctrl in enumerate(sequence):
                    pred_state, eval_info = self.predict_motion(sim_state, seq_ctrl)
                    
                    if h == 0:
                        first_predicted = pred_state
                        first_info = eval_info
                    
                    # Compute cost
                    step_cost = self.evaluate_cost(
                        pred_state, targets_horizon[min(h, len(targets_horizon)-1)],
                        seq_ctrl, eval_info, reference_control,
                    )
                    
                    # Discount future costs
                    discount = 0.9 ** h
                    total_cost += discount * step_cost
                    
                    sim_state = pred_state
                    seq_prev = seq_ctrl
                
                if first_predicted is not None:
                    candidate_costs.append((total_cost, ctrl))
                    
                    if total_cost < best_cost:
                        best_cost = total_cost
                        best_control = ctrl
                        best_state = first_predicted
                        best_info = first_info
                        best_info["best_sequence"] = sequence
            
            # CEM update: fit distribution to elite
            if cem_iter < self.cfg.cem_iterations - 1:
                candidate_costs.sort(key=lambda x: x[0])
                elite = [c for _, c in candidate_costs[:self.cfg.n_elite]]
                self.candidate_generator.update_distribution(elite)
        
        # Update history
        self.history.add(best_control)
        if "best_sequence" in best_info:
            self.history.prev_best_sequence = best_info["best_sequence"]
        
        # Logging
        log_entry = {
            "state": (state.x, state.y),
            "target": (target.x, target.y),
            "control": best_control.to_array().tolist(),
            "best_cost": best_cost,
            "n_candidates_evaluated": len(candidate_costs),
            "trap_xy": best_info.get("trap_xy"),
        }
        self.step_logs.append(log_entry)
        
        return best_control, best_state, best_info
    
    def get_control_trajectory(self) -> np.ndarray:
        """Get array of control values over time for plotting."""
        if not self.history.controls:
            return np.array([])
        return np.array([c.to_array() for c in self.history.controls])


def plot_control_smoothness(controller: SmoothMPCController, save_path: str):
    """
    Plot transducer positions over time to verify smoothness.
    
    Creates 4 subplots:
    - xA(t), xB(t)
    - yA(t), yB(t)  
    - Δx per step (should be smooth, not oscillating)
    - Phase difference over time
    """
    import matplotlib.pyplot as plt
    
    trajectory = controller.get_control_trajectory()
    if len(trajectory) < 2:
        print("Not enough data to plot")
        return
    
    t = np.arange(len(trajectory))
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # xA, xB over time
    ax = axes[0, 0]
    ax.plot(t, trajectory[:, 0] * 1e3, label='xA', linewidth=2)
    ax.plot(t, trajectory[:, 2] * 1e3, label='xB', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('x (mm)')
    ax.set_title('Transducer X positions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # yA, yB over time
    ax = axes[0, 1]
    ax.plot(t, trajectory[:, 1] * 1e3, label='yA', linewidth=2)
    ax.plot(t, trajectory[:, 3] * 1e3, label='yB', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('y (mm)')
    ax.set_title('Transducer Y positions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Delta x per step
    ax = axes[1, 0]
    if len(trajectory) > 1:
        delta_xA = np.diff(trajectory[:, 0]) * 1e6  # µm
        delta_xB = np.diff(trajectory[:, 2]) * 1e6
        ax.plot(t[1:], delta_xA, label='ΔxA', linewidth=2)
        ax.plot(t[1:], delta_xB, label='ΔxB', linewidth=2)
        ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Step')
    ax.set_ylabel('Δx (µm/step)')
    ax.set_title('Position changes (should not oscillate)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Phase difference
    ax = axes[1, 1]
    phase_diff = trajectory[:, 7] - trajectory[:, 6]  # phiB - phiA
    ax.plot(t, phase_diff, linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('φB - φA (rad)')
    ax.set_title('Phase difference')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")
