"""
Structured control layer for acoustic tweezers particle manipulation.

This module implements physics-consistent, interpretable control for steering
particles by varying transducer phase, amplitude, and position while maintaining
trap stability.

Key components:
- ControlState: Particle position state vector
- ControlVector: Full control parameterization with bounds and rate limits
- JacobianEstimator: Finite-difference estimation of stiffness and control effectiveness
- ParticleController: Main controller with one-step and MPC modes

Stage B/C/D/E upgrades:
- Trap-centre detection integrated into cost function
- TrapTracker for identity continuity
- Sequential rate limiting in MPC
- Near-saturation soft penalties
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Callable, Any
from enum import Enum
import numpy as np

from .evaluator import Control2Pucks, BottomFootprint25DEvaluator
from acousto.analysis import TrapTracker, TrapCenterResult


# =============================================================================
# Step 1: Formalised Control Interfaces
# =============================================================================

@dataclass
class ControlState:
    """State vector: particle position (x, y) in meters."""
    x: float
    y: float
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float64)
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> "ControlState":
        return cls(x=float(arr[0]), y=float(arr[1]))
    
    def distance_to(self, other: "ControlState") -> float:
        """Euclidean distance to another state."""
        return float(np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2))


@dataclass
class ControlBounds:
    """Bounds for control variables (min, max)."""
    x_min: float = 0.0
    x_max: float = 2e-3
    y_min: float = 0.0
    y_max: float = 0.25e-3   # Transducers confined to bottom band
    v_min: float = 0.0
    v_max: float = 2e-3       # Max velocity amplitude
    phi_min: float = -np.pi
    phi_max: float = np.pi


@dataclass
class ControlRateLimits:
    """Per-step rate limits for control variables."""
    dx_max: float = 0.1e-3    # Max transducer position change per step (m)
    dy_max: float = 0.05e-3
    dv_max: float = 2e-4      # Max amplitude change per step
    dphi_max: float = 0.5     # Max phase change per step (rad)


@dataclass
class ControlVector:
    """
    Full control parameterization for two-transducer system.
    
    Control vector u = [xA, yA, xB, yB, vA, vB, phiA, phiB]
    
    Supports:
    - Conversion to/from Control2Pucks
    - Bounds checking and clamping
    - Rate limiting between consecutive controls
    """
    xA: float
    yA: float
    xB: float
    yB: float
    vA: float
    vB: float
    phiA: float
    phiB: float
    
    bounds: ControlBounds = field(default_factory=ControlBounds)
    rate_limits: ControlRateLimits = field(default_factory=ControlRateLimits)
    
    @classmethod
    def from_control2pucks(
        cls,
        u: Control2Pucks,
        bounds: Optional[ControlBounds] = None,
        rate_limits: Optional[ControlRateLimits] = None,
    ) -> "ControlVector":
        """Create ControlVector from legacy Control2Pucks."""
        return cls(
            xA=u.xA, yA=u.yA, xB=u.xB, yB=u.yB,
            vA=u.vA, vB=u.vB, phiA=u.phiA, phiB=u.phiB,
            bounds=bounds or ControlBounds(),
            rate_limits=rate_limits or ControlRateLimits(),
        )
    
    def to_control2pucks(self) -> Control2Pucks:
        """Convert to legacy Control2Pucks for evaluator compatibility."""
        return Control2Pucks(
            xA=self.xA, yA=self.yA, xB=self.xB, yB=self.yB,
            vA=self.vA, vB=self.vB, phiA=self.phiA, phiB=self.phiB,
        )
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array [xA, yA, xB, yB, vA, vB, phiA, phiB]."""
        return np.array([
            self.xA, self.yA, self.xB, self.yB,
            self.vA, self.vB, self.phiA, self.phiB
        ], dtype=np.float64)
    
    @classmethod
    def from_array(
        cls,
        arr: np.ndarray,
        bounds: Optional[ControlBounds] = None,
        rate_limits: Optional[ControlRateLimits] = None,
    ) -> "ControlVector":
        """Create from numpy array."""
        return cls(
            xA=float(arr[0]), yA=float(arr[1]),
            xB=float(arr[2]), yB=float(arr[3]),
            vA=float(arr[4]), vB=float(arr[5]),
            phiA=float(arr[6]), phiB=float(arr[7]),
            bounds=bounds or ControlBounds(),
            rate_limits=rate_limits or ControlRateLimits(),
        )
    
    def clamp_to_bounds(self) -> "ControlVector":
        """Return new ControlVector clamped to bounds."""
        b = self.bounds
        return ControlVector(
            xA=float(np.clip(self.xA, b.x_min, b.x_max)),
            yA=float(np.clip(self.yA, b.y_min, b.y_max)),
            xB=float(np.clip(self.xB, b.x_min, b.x_max)),
            yB=float(np.clip(self.yB, b.y_min, b.y_max)),
            vA=float(np.clip(self.vA, b.v_min, b.v_max)),
            vB=float(np.clip(self.vB, b.v_min, b.v_max)),
            phiA=float(self._wrap_angle(self.phiA)),
            phiB=float(self._wrap_angle(self.phiB)),
            bounds=self.bounds,
            rate_limits=self.rate_limits,
        )
    
    def apply_rate_limits(self, u_prev: "ControlVector") -> "ControlVector":
        """Return new ControlVector with rate limits applied relative to u_prev."""
        r = self.rate_limits
        
        def clamp_delta(new_val: float, old_val: float, max_delta: float) -> float:
            delta = new_val - old_val
            return old_val + float(np.clip(delta, -max_delta, max_delta))
        
        def clamp_angle_delta(new_phi: float, old_phi: float, max_delta: float) -> float:
            # Handle wrap-around for phase
            delta = self._wrap_angle(new_phi - old_phi)
            return self._wrap_angle(old_phi + float(np.clip(delta, -max_delta, max_delta)))
        
        return ControlVector(
            xA=clamp_delta(self.xA, u_prev.xA, r.dx_max),
            yA=clamp_delta(self.yA, u_prev.yA, r.dy_max),
            xB=clamp_delta(self.xB, u_prev.xB, r.dx_max),
            yB=clamp_delta(self.yB, u_prev.yB, r.dy_max),
            vA=clamp_delta(self.vA, u_prev.vA, r.dv_max),
            vB=clamp_delta(self.vB, u_prev.vB, r.dv_max),
            phiA=clamp_angle_delta(self.phiA, u_prev.phiA, r.dphi_max),
            phiB=clamp_angle_delta(self.phiB, u_prev.phiB, r.dphi_max),
            bounds=self.bounds,
            rate_limits=self.rate_limits,
        )
    
    @staticmethod
    def _wrap_angle(a: float) -> float:
        """Wrap angle to [-π, π]."""
        return float((a + np.pi) % (2.0 * np.pi) - np.pi)
    
    @property
    def dim(self) -> int:
        """Dimension of control vector."""
        return 8


# =============================================================================
# Step 2: Local Control Effectiveness Estimation (Jacobians)
# =============================================================================

@dataclass
class JacobianInfo:
    """
    Jacobian information at a given state and control.
    
    Attributes:
        dF_dx: Trap stiffness matrix (2x2), ∂F/∂x
        dF_du: Control effectiveness matrix (2x8), ∂F/∂u
        stiffness_eigenvalues: Eigenvalues of dF_dx (indicates trap stability)
        is_stable: True if both eigenvalues are negative (stable trap)
        min_eigenvalue: Most negative eigenvalue (trap strength)
    """
    dF_dx: np.ndarray           # Shape (2, 2): [∂Fx/∂x, ∂Fx/∂y; ∂Fy/∂x, ∂Fy/∂y]
    dF_du: np.ndarray           # Shape (2, 8): [∂Fx/∂u; ∂Fy/∂u]
    stiffness_eigenvalues: np.ndarray
    is_stable: bool
    min_eigenvalue: float


class JacobianEstimator:
    """
    Finite-difference estimation of trap stiffness and control effectiveness.
    
    At each timestep, estimates:
    - ∂F/∂x: How force changes with particle position (trap stiffness)
    - ∂F/∂u: How force changes with control input (control effectiveness)
    """
    
    def __init__(
        self,
        evaluator: BottomFootprint25DEvaluator,
        eps_x: float = 1e-6,      # Spatial perturbation (m)
        eps_u: float = 1e-6,      # Control perturbation (various units)
    ):
        self.evaluator = evaluator
        self.eps_x = eps_x
        self.eps_u = eps_u
        
        # Cache for smoothing Jacobians over time
        self._dF_dx_cache: Optional[np.ndarray] = None
        self._dF_du_cache: Optional[np.ndarray] = None
        self._cache_alpha: float = 0.3  # Exponential smoothing factor
    
    def _eval_force(self, x: float, y: float, u: ControlVector) -> tuple[float, float]:
        """Evaluate force at (x, y) under control u."""
        u2p = u.to_control2pucks()
        _, _, _, info = self.evaluator.step(
            xp=x, yp=y,
            target_x=x, target_y=y,  # Target doesn't affect force computation
            u=u2p, u_prev=None,
        )
        return info["fx"], info["fy"]
    
    def estimate(
        self,
        state: ControlState,
        control: ControlVector,
        use_cache: bool = True,
    ) -> JacobianInfo:
        """
        Estimate Jacobians at current state and control using finite differences.
        
        Parameters
        ----------
        state : ControlState
            Current particle position.
        control : ControlVector
            Current control input.
        use_cache : bool
            If True, apply exponential smoothing to reduce noise.
        
        Returns
        -------
        JacobianInfo with stiffness matrix, control effectiveness, and stability info.
        """
        x, y = state.x, state.y
        eps_x = self.eps_x
        eps_u = self.eps_u
        
        # Base force
        fx0, fy0 = self._eval_force(x, y, control)
        
        # ===== ∂F/∂x (Stiffness matrix) =====
        # Perturb in x
        fx_px, fy_px = self._eval_force(x + eps_x, y, control)
        fx_mx, fy_mx = self._eval_force(x - eps_x, y, control)
        
        # Perturb in y
        fx_py, fy_py = self._eval_force(x, y + eps_x, control)
        fx_my, fy_my = self._eval_force(x, y - eps_x, control)
        
        # Central differences for stiffness
        dFx_dx = (fx_px - fx_mx) / (2 * eps_x)
        dFy_dx = (fy_px - fy_mx) / (2 * eps_x)
        dFx_dy = (fx_py - fx_my) / (2 * eps_x)
        dFy_dy = (fy_py - fy_my) / (2 * eps_x)
        
        dF_dx = np.array([
            [dFx_dx, dFx_dy],
            [dFy_dx, dFy_dy],
        ], dtype=np.float64)
        
        # ===== ∂F/∂u (Control effectiveness) =====
        # Control perturbation scales (adjusted for each control dimension)
        u_arr = control.to_array()
        eps_vec = np.array([
            eps_u,        # xA (position)
            eps_u,        # yA
            eps_u,        # xB
            eps_u,        # yB
            eps_u * 1e2,  # vA (amplitude, scale up)
            eps_u * 1e2,  # vB
            eps_u * 1e3,  # phiA (phase, scale up more)
            eps_u * 1e3,  # phiB
        ])
        
        dF_du = np.zeros((2, 8), dtype=np.float64)
        
        for i in range(8):
            u_plus = u_arr.copy()
            u_minus = u_arr.copy()
            u_plus[i] += eps_vec[i]
            u_minus[i] -= eps_vec[i]
            
            ctrl_plus = ControlVector.from_array(u_plus, control.bounds, control.rate_limits)
            ctrl_minus = ControlVector.from_array(u_minus, control.bounds, control.rate_limits)
            
            fx_p, fy_p = self._eval_force(x, y, ctrl_plus)
            fx_m, fy_m = self._eval_force(x, y, ctrl_minus)
            
            dF_du[0, i] = (fx_p - fx_m) / (2 * eps_vec[i])
            dF_du[1, i] = (fy_p - fy_m) / (2 * eps_vec[i])
        
        # ===== Apply smoothing if caching enabled =====
        if use_cache:
            if self._dF_dx_cache is not None:
                alpha = self._cache_alpha
                dF_dx = alpha * dF_dx + (1 - alpha) * self._dF_dx_cache
                dF_du = alpha * dF_du + (1 - alpha) * self._dF_du_cache
            self._dF_dx_cache = dF_dx.copy()
            self._dF_du_cache = dF_du.copy()
        
        # ===== Compute stability info =====
        eigenvalues = np.linalg.eigvalsh(dF_dx)
        is_stable = bool(np.all(eigenvalues < 0))
        min_eigenvalue = float(np.min(eigenvalues))
        
        return JacobianInfo(
            dF_dx=dF_dx,
            dF_du=dF_du,
            stiffness_eigenvalues=eigenvalues,
            is_stable=is_stable,
            min_eigenvalue=min_eigenvalue,
        )
    
    def reset_cache(self) -> None:
        """Clear Jacobian cache."""
        self._dF_dx_cache = None
        self._dF_du_cache = None


# =============================================================================
# Step 5: Trap-aware Safety Constraints
# =============================================================================

@dataclass
class SafetyConfig:
    """Configuration for trap-aware safety constraints."""
    min_stiffness: float = -1e-10       # Minimum acceptable trap stiffness (eigenvalue)
    min_transducer_separation: float = 0.2e-3  # Min distance between transducers (m)
    boundary_margin: float = 0.1e-3     # Keep particle this far from domain edges
    reject_saddle_proximity: float = 0.3e-3    # Reject if particle within this of saddle
    max_control_magnitude: float = 5e-4  # Maximum velocity amplitude


class SafetyChecker:
    """
    Evaluates safety constraints for control proposals.
    
    Refuses control updates that:
    - Destroy trap stability (stiffness too low)
    - Move particle too close to boundaries
    - Bring transducers too close together
    - Risk losing the particle to saddle points
    """
    
    def __init__(
        self,
        config: SafetyConfig,
        domain_bounds: tuple[float, float, float, float],  # (x_min, x_max, y_min, y_max)
    ):
        self.config = config
        self.x_min, self.x_max, self.y_min, self.y_max = domain_bounds
    
    def check(
        self,
        state: ControlState,
        control: ControlVector,
        jacobian: Optional[JacobianInfo] = None,
        saddle_points: Optional[list[tuple[float, float]]] = None,
    ) -> tuple[bool, list[str]]:
        """
        Check if state/control combination is safe.
        
        Returns
        -------
        (is_safe, violations) where violations is a list of constraint violation messages.
        """
        violations: list[str] = []
        cfg = self.config
        
        # Check trap stiffness
        if jacobian is not None:
            if jacobian.min_eigenvalue > cfg.min_stiffness:
                violations.append(
                    f"Trap too weak: min_eigenvalue={jacobian.min_eigenvalue:.3e} > {cfg.min_stiffness:.3e}"
                )
        
        # Check boundary margins
        margin = cfg.boundary_margin
        if state.x < self.x_min + margin:
            violations.append(f"Particle too close to left boundary: x={state.x*1e3:.3f}mm")
        if state.x > self.x_max - margin:
            violations.append(f"Particle too close to right boundary: x={state.x*1e3:.3f}mm")
        if state.y < self.y_min + margin:
            violations.append(f"Particle too close to bottom boundary: y={state.y*1e3:.3f}mm")
        if state.y > self.y_max - margin:
            violations.append(f"Particle too close to top boundary: y={state.y*1e3:.3f}mm")
        
        # Check transducer separation
        sep = np.sqrt((control.xA - control.xB)**2 + (control.yA - control.yB)**2)
        if sep < cfg.min_transducer_separation:
            violations.append(f"Transducers too close: separation={sep*1e3:.3f}mm")
        
        # Check amplitude bounds
        if control.vA > cfg.max_control_magnitude:
            violations.append(f"vA too high: {control.vA:.3e} > {cfg.max_control_magnitude:.3e}")
        if control.vB > cfg.max_control_magnitude:
            violations.append(f"vB too high: {control.vB:.3e} > {cfg.max_control_magnitude:.3e}")
        
        # Check saddle proximity
        if saddle_points is not None:
            for sx, sy in saddle_points:
                dist = np.sqrt((state.x - sx)**2 + (state.y - sy)**2)
                if dist < cfg.reject_saddle_proximity:
                    violations.append(f"Too close to saddle at ({sx*1e3:.2f}, {sy*1e3:.2f})mm")
        
        is_safe = len(violations) == 0
        return is_safe, violations


# =============================================================================
# Step 6: Diagnostics & Logging
# =============================================================================

@dataclass
class ControlLog:
    """Single timestep log entry."""
    step: int
    state: ControlState
    target: ControlState
    control: ControlVector
    predicted_state: ControlState
    actual_state: ControlState
    tracking_error: float
    prediction_error: float
    stiffness: float
    control_magnitude: float
    was_rejected: bool
    rejection_reasons: list[str]
    jacobian_info: Optional[JacobianInfo]


class ControlLogger:
    """
    Logs control performance for analysis.
    
    Tracks:
    - Predicted vs actual particle displacement
    - Trap stiffness over time
    - Control magnitude per dimension
    - Rejected control proposals
    """
    
    def __init__(self, max_history: int = 1000):
        self.history: list[ControlLog] = []
        self.max_history = max_history
        self._rejected_count = 0
        self._total_count = 0
    
    def log(
        self,
        step: int,
        state: ControlState,
        target: ControlState,
        control: ControlVector,
        predicted_state: ControlState,
        actual_state: ControlState,
        jacobian_info: Optional[JacobianInfo] = None,
        was_rejected: bool = False,
        rejection_reasons: Optional[list[str]] = None,
    ) -> None:
        """Log a control step."""
        tracking_error = actual_state.distance_to(target)
        prediction_error = actual_state.distance_to(predicted_state)
        stiffness = jacobian_info.min_eigenvalue if jacobian_info else 0.0
        
        # Control magnitude (norm of amplitude vector)
        control_mag = np.sqrt(control.vA**2 + control.vB**2)
        
        entry = ControlLog(
            step=step,
            state=state,
            target=target,
            control=control,
            predicted_state=predicted_state,
            actual_state=actual_state,
            tracking_error=tracking_error,
            prediction_error=prediction_error,
            stiffness=stiffness,
            control_magnitude=control_mag,
            was_rejected=was_rejected,
            rejection_reasons=rejection_reasons or [],
            jacobian_info=jacobian_info,
        )
        
        self.history.append(entry)
        self._total_count += 1
        if was_rejected:
            self._rejected_count += 1
        
        # Trim history if needed
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def get_summary(self) -> dict:
        """Get summary statistics."""
        if not self.history:
            return {}
        
        tracking_errors = [e.tracking_error for e in self.history]
        prediction_errors = [e.prediction_error for e in self.history]
        stiffnesses = [e.stiffness for e in self.history]
        control_mags = [e.control_magnitude for e in self.history]
        
        return {
            "n_steps": len(self.history),
            "tracking_error_mean": float(np.mean(tracking_errors)),
            "tracking_error_max": float(np.max(tracking_errors)),
            "prediction_error_mean": float(np.mean(prediction_errors)),
            "stiffness_mean": float(np.mean(stiffnesses)),
            "stiffness_min": float(np.min(stiffnesses)),
            "control_mag_mean": float(np.mean(control_mags)),
            "rejection_rate": self._rejected_count / max(1, self._total_count),
        }
    
    def get_arrays(self) -> dict[str, np.ndarray]:
        """Get history as numpy arrays for plotting."""
        if not self.history:
            return {}
        
        return {
            "steps": np.array([e.step for e in self.history]),
            "x": np.array([e.actual_state.x for e in self.history]),
            "y": np.array([e.actual_state.y for e in self.history]),
            "target_x": np.array([e.target.x for e in self.history]),
            "target_y": np.array([e.target.y for e in self.history]),
            "tracking_error": np.array([e.tracking_error for e in self.history]),
            "prediction_error": np.array([e.prediction_error for e in self.history]),
            "stiffness": np.array([e.stiffness for e in self.history]),
            "control_magnitude": np.array([e.control_magnitude for e in self.history]),
        }


# =============================================================================
# Step 3 & 4: One-Step Control Law and MPC Controller
# =============================================================================

@dataclass
class ControllerConfig:
    """Configuration for ParticleController."""
    # One-step controller weights
    tracking_weight: float = 1.0         # Weight on position tracking error
    effort_weight: float = 0.01          # Weight on control effort (||Δu||²)
    stiffness_weight: float = 0.1        # Weight on maintaining trap stiffness
    
    # Task 5: Trap-steering weights
    trap_weight: float = 2.0             # Weight on ||trap_centre - target||²
    particle_trap_weight: float = 0.5    # Weight on ||particle - trap_centre||²
    
    # MPC parameters
    horizon: int = 5                     # Prediction horizon (H)
    n_candidates: int = 100              # Number of random candidates to evaluate
    
    # Random shooting parameters
    position_noise: float = 0.05e-3      # Std dev for position perturbation
    amplitude_noise: float = 0.5e-4      # Std dev for amplitude perturbation
    phase_noise: float = 0.3             # Std dev for phase perturbation (rad)
    
    # Physical parameters
    dt: float = 5e-3                     # Time step (s)
    viscosity: float = 1e-3              # Fluid viscosity (Pa·s)
    particle_radius: float = 5e-6        # Particle radius (m)


class ParticleController:
    """
    Structured controller for acoustic tweezers particle manipulation.
    
    Implements:
    - One-step gradient-informed control
    - Short-horizon MPC with random shooting
    - Trap-aware safety constraints
    - Full diagnostics and logging
    
    Usage:
        controller = ParticleController(evaluator, config)
        control, info = controller.step(state, target, current_control)
    """
    
    def __init__(
        self,
        evaluator: BottomFootprint25DEvaluator,
        config: Optional[ControllerConfig] = None,
        safety_config: Optional[SafetyConfig] = None,
        bounds: Optional[ControlBounds] = None,
        rate_limits: Optional[ControlRateLimits] = None,
    ):
        self.evaluator = evaluator
        self.config = config or ControllerConfig()
        self.bounds = bounds or ControlBounds(
            x_max=evaluator.domain.Lx,
            y_max=min(0.25e-3, evaluator.domain.Ly),
        )
        self.rate_limits = rate_limits or ControlRateLimits()
        
        # Safety checker
        domain_bounds = (
            float(evaluator.op.x[0]),
            float(evaluator.op.x[-1]),
            float(evaluator.op.y[0]),
            float(evaluator.op.y[-1]),
        )
        self.safety = SafetyChecker(
            safety_config or SafetyConfig(),
            domain_bounds,
        )
        
        # Jacobian estimator
        self.jacobian_estimator = JacobianEstimator(evaluator)
        
        # Logger
        self.logger = ControlLogger()
        
        # Random generator
        self._rng = np.random.default_rng(42)
        
        # Previous control for rate limiting
        self._prev_control: Optional[ControlVector] = None
        self._step_count = 0
        
        # STAGE C: TrapTracker for identity continuity
        self.trap_tracker = TrapTracker(
            max_distance=0.2e-3,  # 0.2 mm max distance for matching
            stiffness_weight=0.1,
            lost_threshold=5,
        )
    
    def compute_drag_coefficient(self) -> float:
        """Compute Stokes drag coefficient γ = 6πηa."""
        return 6.0 * np.pi * self.config.viscosity * self.config.particle_radius
    
    def predict_motion(
        self,
        state: ControlState,
        control: ControlVector,
    ) -> tuple[ControlState, dict]:
        """
        Predict particle motion under given control.
        
        Uses overdamped dynamics: x_{t+1} = x_t + (dt/γ) * F(x_t, u)
        
        Returns
        -------
        (next_state, eval_info) - eval_info contains limiter diagnostics
        """
        u2p = control.to_control2pucks()
        xp1, yp1, _, info = self.evaluator.step(
            xp=state.x, yp=state.y,
            target_x=state.x, target_y=state.y,  # Target doesn't affect dynamics
            u=u2p, u_prev=None,
        )
        return ControlState(x=xp1, y=yp1), info
    
    def evaluate_cost(
        self,
        state: ControlState,
        target: ControlState,
        control: ControlVector,
        prev_control: Optional[ControlVector] = None,
        jacobian: Optional[JacobianInfo] = None,
        trap_centre: Optional[tuple[float, float]] = None,
    ) -> float:
        """
        Evaluate total cost for a state-control pair.
        
        Task 5: Primary objective is to steer the trap, not the particle directly.
        
        Cost = trap_weight * ||trap - target||²         (primary: move the well)
             + particle_trap_weight * ||particle - trap||² (secondary: keep in well)
             + effort_weight * ||Δu||² 
             + stiffness_weight * max(0, -min_eigenvalue)
             
        Falls back to tracking_weight * ||particle - target||² if no trap_centre.
        """
        cfg = self.config
        
        # Task 5: Trap-steering objective
        if trap_centre is not None:
            trap_x, trap_y = trap_centre
            trap_state = ControlState(x=trap_x, y=trap_y)
            
            # Primary: drive trap centre toward target
            trap_target_cost = cfg.trap_weight * trap_state.distance_to(target)**2
            
            # Secondary: keep particle inside trap well
            particle_trap_cost = cfg.particle_trap_weight * state.distance_to(trap_state)**2
            
            tracking_cost = trap_target_cost + particle_trap_cost
        else:
            # Fallback: original direct particle tracking
            tracking_cost = cfg.tracking_weight * state.distance_to(target)**2
        
        # Control effort (change from previous)
        effort_cost = 0.0
        if prev_control is not None:
            du = control.to_array() - prev_control.to_array()
            # Normalize by typical scales
            scales = np.array([1e-3, 1e-3, 1e-3, 1e-3, 1e-4, 1e-4, 1.0, 1.0])
            effort_cost = cfg.effort_weight * float(np.sum((du / scales)**2))
        
        # Stiffness penalty (penalize loss of trap stability)
        stiffness_cost = 0.0
        if jacobian is not None:
            # Penalize if eigenvalue approaches zero (losing trap)
            if jacobian.min_eigenvalue > -1e-12:
                stiffness_cost = cfg.stiffness_weight * (1e-12 - jacobian.min_eigenvalue)**2
        
        return tracking_cost + effort_cost + stiffness_cost
    
    def generate_candidates(
        self,
        base_control: ControlVector,
        n: int,
    ) -> list[ControlVector]:
        """Generate control candidates via Gaussian perturbation."""
        cfg = self.config
        candidates: list[ControlVector] = [base_control]  # Always include current
        
        base_arr = base_control.to_array()
        
        # Noise scales for each dimension
        noise_scales = np.array([
            cfg.position_noise,   # xA
            cfg.position_noise,   # yA
            cfg.position_noise,   # xB
            cfg.position_noise,   # yB
            cfg.amplitude_noise,  # vA
            cfg.amplitude_noise,  # vB
            cfg.phase_noise,      # phiA
            cfg.phase_noise,      # phiB
        ])
        
        for _ in range(n - 1):
            perturbed = base_arr + self._rng.normal(scale=noise_scales)
            ctrl = ControlVector.from_array(perturbed, self.bounds, self.rate_limits)
            ctrl = ctrl.clamp_to_bounds()
            if self._prev_control is not None:
                ctrl = ctrl.apply_rate_limits(self._prev_control)
            candidates.append(ctrl)
        
        return candidates
    
    def step_onestep(
        self,
        state: ControlState,
        target: ControlState,
        current_control: ControlVector,
    ) -> tuple[ControlVector, dict]:
        """
        One-step control law using random shooting with Jacobian bias.
        
        STAGE D: Uses trap_centre from evaluator for trap-steering objective.
        
        Returns
        -------
        (best_control, info_dict)
        """
        # Estimate Jacobians for current state
        jacobian = self.jacobian_estimator.estimate(state, current_control)
        
        # Generate candidates
        candidates = self.generate_candidates(current_control, self.config.n_candidates)
        
        best_control = current_control
        best_cost = float("inf")
        best_predicted = state
        rejected_candidates: list[tuple[ControlVector, list[str]]] = []
        
        best_eval_info = {}  # Store eval info from best candidate
        best_trap_xy = None
        
        for i, ctrl in enumerate(candidates):
            # Predict next state (now returns eval_info with trap_xy)
            predicted, eval_info = self.predict_motion(state, ctrl)
            
            # STAGE D: Extract trap centre from evaluator output
            trap_xy = eval_info.get("trap_xy", None)
            
            # Check safety
            pred_jacobian = self.jacobian_estimator.estimate(predicted, ctrl, use_cache=False)
            is_safe, violations = self.safety.check(predicted, ctrl, pred_jacobian)
            
            if not is_safe:
                rejected_candidates.append((ctrl, violations))
                continue
            
            # STAGE D: Evaluate cost with trap_centre
            cost = self.evaluate_cost(
                predicted, target, ctrl,
                prev_control=self._prev_control,
                jacobian=pred_jacobian,
                trap_centre=trap_xy,  # Pass trap centre for trap-steering
            )
            
            # Saturation penalty: penalize candidates that hit the step limiter
            # This prevents the controller from always choosing controls that saturate
            if eval_info.get("step_limited", False):
                step_scale = eval_info.get("step_scale", 1.0)
                # Penalty proportional to how much we had to scale down
                saturation_penalty = 0.1 * (1.0 - step_scale)**2
                cost += saturation_penalty
            
            # STAGE E: Near-saturation soft penalty
            step_scale = eval_info.get("step_scale", 1.0)
            if step_scale < 0.9:
                # Soft penalty when approaching saturation
                near_saturation_penalty = 0.05 * (0.9 - step_scale)**2
                cost += near_saturation_penalty
            
            if cost < best_cost:
                best_cost = cost
                best_control = ctrl
                best_predicted = predicted
                best_eval_info = eval_info
                best_trap_xy = trap_xy
        
        info = {
            "jacobian": jacobian,
            "best_cost": best_cost,
            "predicted_state": best_predicted,
            "n_rejected": len(rejected_candidates),
            "n_candidates": len(candidates),
            # Limiter diagnostics from best candidate
            "step_limited": best_eval_info.get("step_limited", False),
            "step_scale": best_eval_info.get("step_scale", 1.0),
            "raw_step_mm": best_eval_info.get("raw_step_mm", 0.0),
            "step_mm": best_eval_info.get("step_mm", 0.0),
            "fx": best_eval_info.get("fx", 0.0),
            "fy": best_eval_info.get("fy", 0.0),
            # STAGE D: Trap centre from best candidate
            "trap_xy": best_trap_xy,
            "trap_stiffness_eigs": best_eval_info.get("trap_stiffness_eigs", None),
            "trap_is_stable": best_eval_info.get("trap_is_stable", False),
        }
        
        return best_control, info
    
    def step_mpc(
        self,
        state: ControlState,
        targets: list[ControlState],
        current_control: ControlVector,
    ) -> tuple[ControlVector, dict]:
        """
        Model Predictive Control with receding horizon.
        
        STAGE D/E: Uses trap centres for cost evaluation and applies rate limits
        sequentially inside the rollout (not just from prev_control at start).
        
        Parameters
        ----------
        state : ControlState
            Current particle position.
        targets : list[ControlState]
            Target positions for the next H steps.
        current_control : ControlVector
            Current control input.
        
        Returns
        -------
        (best_first_control, info_dict) - Only first control is returned (receding horizon).
        """
        H = min(self.config.horizon, len(targets))
        n_candidates = self.config.n_candidates
        
        best_sequence: list[ControlVector] = []
        best_total_cost = float("inf")
        best_trajectory: list[ControlState] = []
        best_trap_xy = None
        
        # Generate candidate control sequences
        for _ in range(n_candidates):
            # Generate a sequence of controls
            sequence: list[ControlVector] = []
            ctrl = current_control
            # STAGE E: Track previous control within sequence for rate limiting
            seq_prev_ctrl = self._prev_control
            
            for h in range(H):
                # Perturb from previous in sequence
                candidates = self.generate_candidates(ctrl, 5)
                new_ctrl = self._rng.choice(candidates)  # type: ignore
                
                # STAGE E: Apply rate limits SEQUENTIALLY inside rollout
                if seq_prev_ctrl is not None:
                    new_ctrl = new_ctrl.apply_rate_limits(seq_prev_ctrl)
                new_ctrl = new_ctrl.clamp_to_bounds()
                
                sequence.append(new_ctrl)
                seq_prev_ctrl = new_ctrl  # Update for next step in sequence
                ctrl = new_ctrl
            
            # Simulate forward and compute total cost
            total_cost = 0.0
            trajectory: list[ControlState] = [state]
            sim_state = state
            prev_ctrl = self._prev_control
            is_feasible = True
            first_trap_xy = None
            
            for h, ctrl in enumerate(sequence):
                # Predict motion (returns trap_xy in eval_info)
                pred_state, eval_info = self.predict_motion(sim_state, ctrl)
                trajectory.append(pred_state)
                
                # STAGE D: Extract trap centre
                trap_xy = eval_info.get("trap_xy", None)
                if h == 0:
                    first_trap_xy = trap_xy
                
                # Safety check
                is_safe, _ = self.safety.check(pred_state, ctrl)
                if not is_safe:
                    is_feasible = False
                    break
                
                # Accumulate cost (discounted) - STAGE D: with trap centre
                discount = 0.9 ** h
                step_cost = self.evaluate_cost(
                    pred_state, targets[h], ctrl,
                    prev_control=prev_ctrl,
                    trap_centre=trap_xy,  # Pass trap centre
                )
                
                # Saturation penalty in MPC rollout
                if eval_info.get("step_limited", False):
                    step_scale = eval_info.get("step_scale", 1.0)
                    step_cost += 0.1 * (1.0 - step_scale)**2
                
                # STAGE E: Near-saturation soft penalty
                step_scale = eval_info.get("step_scale", 1.0)
                if step_scale < 0.9:
                    step_cost += 0.05 * (0.9 - step_scale)**2
                
                total_cost += discount * step_cost
                
                sim_state = pred_state
                prev_ctrl = ctrl
            
            if is_feasible and total_cost < best_total_cost:
                best_total_cost = total_cost
                best_sequence = sequence
                best_trajectory = trajectory
                best_trap_xy = first_trap_xy
        
        # Return first control in best sequence (receding horizon)
        if best_sequence:
            first_control = best_sequence[0]
        else:
            # Fallback to current control if no feasible sequence found
            first_control = current_control
            # Task 3: Ensure we ALWAYS have a predicted trajectory
            # Even if no feasible sequence, predict one step with current control
            fallback_pred, fallback_info = self.predict_motion(state, current_control)
            best_trajectory = [state, fallback_pred]
            best_trap_xy = fallback_info.get("trap_xy", None)
        
        # Task 3: Ensure predicted_trajectory is never empty
        if not best_trajectory or len(best_trajectory) < 2:
            fallback_pred, fallback_info = self.predict_motion(state, first_control)
            best_trajectory = [state, fallback_pred]
            if best_trap_xy is None:
                best_trap_xy = fallback_info.get("trap_xy", None)
        
        info = {
            "horizon": H,
            "best_cost": best_total_cost,
            "predicted_trajectory": best_trajectory,
            "sequence_length": len(best_sequence),
            # STAGE D: Trap centre from first step of best sequence
            "trap_xy": best_trap_xy,
        }
        
        return first_control, info
    
    def step(
        self,
        state: ControlState,
        target: ControlState,
        current_control: ControlVector,
        targets_horizon: Optional[list[ControlState]] = None,
        use_mpc: bool = False,
    ) -> tuple[ControlVector, ControlState, dict]:
        """
        Main control step interface.
        
        STAGE D: Returns trap_xy in info dict for visualization.
        
        Parameters
        ----------
        state : ControlState
            Current particle position.
        target : ControlState
            Target position for this step.
        current_control : ControlVector
            Current control input.
        targets_horizon : list[ControlState], optional
            Future targets for MPC mode.
        use_mpc : bool
            If True, use MPC; otherwise use one-step control.
        
        Returns
        -------
        (new_control, predicted_next_state, info_dict)
        """
        if use_mpc and targets_horizon:
            new_control, info = self.step_mpc(state, targets_horizon, current_control)
        else:
            new_control, info = self.step_onestep(state, target, current_control)
        
        # Track control change magnitude (for diagnostics, not printed)
        du = np.linalg.norm(new_control.to_array() - current_control.to_array())
        info["control_delta_norm"] = float(du)
        
        # Apply control and get actual next state
        predicted = info.get("predicted_state")
        if predicted is None:
            predicted, _ = self.predict_motion(state, new_control)
        actual, actual_info = self.predict_motion(state, new_control)
        
        # Merge actual eval info into info dict
        info["step_limited"] = actual_info.get("step_limited", False)
        info["step_scale"] = actual_info.get("step_scale", 1.0)
        info["raw_step_mm"] = actual_info.get("raw_step_mm", 0.0)
        info["step_mm"] = actual_info.get("step_mm", 0.0)
        info["fx"] = actual_info.get("fx", 0.0)
        info["fy"] = actual_info.get("fy", 0.0)
        
        # STAGE D: Ensure trap_xy is in info (from actual evaluation)
        if "trap_xy" not in info or info["trap_xy"] is None:
            info["trap_xy"] = actual_info.get("trap_xy", None)
        info["trap_stiffness_eigs"] = actual_info.get("trap_stiffness_eigs", None)
        info["trap_is_stable"] = actual_info.get("trap_is_stable", False)
        info["trap_min_eigenvalue"] = actual_info.get("trap_min_eigenvalue", 0.0)
        
        # STAGE C: Update TrapTracker if trap_xy is available
        trap_xy = info.get("trap_xy")
        if trap_xy is not None:
            # Create a minimal TrapCenterResult for the tracker
            trap_result = TrapCenterResult(
                x=trap_xy[0],
                y=trap_xy[1],
                stiffness_eigvals=info.get("trap_stiffness_eigs", np.array([0.0, 0.0])),
                is_stable=info.get("trap_is_stable", False),
                min_eigenvalue=info.get("trap_min_eigenvalue", 0.0),
                U_at_trap=0.0,  # Not available here
                distance_from_particle=0.0,
                method="controller",
            )
            tracked = self.trap_tracker.update(trap_result)
            info["trap_track_id"] = tracked.track_id
            info["trap_lost"] = tracked.lost
            info["trap_frames_tracked"] = tracked.frames_tracked
        
        # Log
        jacobian = info.get("jacobian")
        self.logger.log(
            step=self._step_count,
            state=state,
            target=target,
            control=new_control,
            predicted_state=predicted,
            actual_state=actual,
            jacobian_info=jacobian,
            was_rejected=info.get("n_rejected", 0) == info.get("n_candidates", 0),
        )
        
        # Update state
        self._prev_control = new_control
        self._step_count += 1
        
        info["actual_state"] = actual
        
        return new_control, actual, info
    
    def reset(self) -> None:
        """Reset controller state."""
        self._prev_control = None
        self._step_count = 0
        self.jacobian_estimator.reset_cache()
        self.logger = ControlLogger()
        self.trap_tracker.reset()  # STAGE C: Reset trap tracker


# =============================================================================
# Step 7: Visualization Hooks
# =============================================================================

@dataclass
class VisualizationData:
    """Data structure for visualization overlays."""
    current_state: ControlState
    target_state: ControlState
    predicted_trajectory: list[ControlState]
    trap_center: Optional[tuple[float, float]]
    stiffness_eigenvalues: Optional[np.ndarray]
    control_vector: ControlVector
    tracking_error: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "current_xy_mm": (self.current_state.x * 1e3, self.current_state.y * 1e3),
            "target_xy_mm": (self.target_state.x * 1e3, self.target_state.y * 1e3),
            "predicted_xy_mm": [(s.x * 1e3, s.y * 1e3) for s in self.predicted_trajectory],
            "trap_center_mm": (self.trap_center[0] * 1e3, self.trap_center[1] * 1e3) if self.trap_center else None,
            "stiffness": self.stiffness_eigenvalues.tolist() if self.stiffness_eigenvalues is not None else None,
            "transducer_A_mm": (self.control_vector.xA * 1e3, self.control_vector.yA * 1e3),
            "transducer_B_mm": (self.control_vector.xB * 1e3, self.control_vector.yB * 1e3),
            "tracking_error_mm": self.tracking_error * 1e3,
        }


def create_visualization_data(
    state: ControlState,
    target: ControlState,
    control: ControlVector,
    info: dict,
    trap_xy: Optional[tuple[float, float]] = None,
) -> VisualizationData:
    """Create visualization data from controller output."""
    predicted_traj = info.get("predicted_trajectory", [])
    if not predicted_traj and "predicted_state" in info:
        predicted_traj = [info["predicted_state"]]
    
    jacobian = info.get("jacobian")
    eigenvalues = jacobian.stiffness_eigenvalues if jacobian else None
    
    return VisualizationData(
        current_state=state,
        target_state=target,
        predicted_trajectory=predicted_traj,
        trap_center=trap_xy,
        stiffness_eigenvalues=eigenvalues,
        control_vector=control,
        tracking_error=state.distance_to(target),
    )
