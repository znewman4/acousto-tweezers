# src/acousto/adjoint/trajectory.py
"""
Discrete-time adjoint backpropagation through overdamped particle dynamics.

This module implements exact trajectory gradients for the K-step lookahead
objective:
    J = sum_{t=0}^{K-1} U(x_t; u_t) + β * U(x_K; u_{K-1})

The forward dynamics are:
    x_{t+1} = x_t + dt * μ * F(x_t, u_t)

where μ = 1/(6πηa) is the mobility (inverse Stokes drag).

The adjoint (co-state) recursion is:
    λ_K = β * ∇_x U(x_K; u_{K-1})  (terminal condition)
    λ_t = ∇_x U(x_t; u_t) + (∂x_{t+1}/∂x_t)^T λ_{t+1}

And the gradient w.r.t. control at step t is:
    ∇_{u_t} J = ∇_{u_t} U(x_t; u_t) + (∂x_{t+1}/∂u_t)^T λ_{t+1}

Key Jacobians:
    ∂x_{t+1}/∂x_t = I + dt * μ * ∂F/∂x_t
    ∂x_{t+1}/∂u_t = dt * μ * ∂F/∂u_t
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List, Callable, Optional
import numpy as np


@dataclass
class TrajectoryState:
    """Cached state from forward rollout for use in backward pass."""
    positions: List[Tuple[float, float]]  # x_t for t=0..K
    controls: List[Tuple[float, float]]   # (v_t, phi_t) for t=0..K-1
    forces: List[Tuple[float, float]]     # (Fx_t, Fy_t) for t=0..K-1
    U_values: List[float]                 # U(x_t; u_t) for t=0..K-1


def compute_dF_dx_fd(
    compute_force_fn: Callable[[float, float, float, float], Tuple[float, float, float, float]],
    v: float, phi: float, x: float, y: float,
    eps: float = 1e-7,
) -> np.ndarray:
    """
    Compute ∂F/∂x via centered finite differences.
    
    Parameters
    ----------
    compute_force_fn : callable
        Function (v, phi, x, y) -> (field, U, Fx, Fy)
    v, phi : float
        Control parameters.
    x, y : float
        Particle position.
    eps : float
        FD step size.
        
    Returns
    -------
    dF_dx : np.ndarray, shape (2, 2)
        Jacobian [[∂Fx/∂x, ∂Fx/∂y], [∂Fy/∂x, ∂Fy/∂y]]
    """
    # Baseline
    _, _, Fx0, Fy0 = compute_force_fn(v, phi, x, y)
    
    # Perturb x
    _, _, Fx_xp, Fy_xp = compute_force_fn(v, phi, x + eps, y)
    _, _, Fx_xm, Fy_xm = compute_force_fn(v, phi, x - eps, y)
    
    # Perturb y
    _, _, Fx_yp, Fy_yp = compute_force_fn(v, phi, x, y + eps)
    _, _, Fx_ym, Fy_ym = compute_force_fn(v, phi, x, y - eps)
    
    dFx_dx = (Fx_xp - Fx_xm) / (2 * eps)
    dFy_dx = (Fy_xp - Fy_xm) / (2 * eps)
    dFx_dy = (Fx_yp - Fx_ym) / (2 * eps)
    dFy_dy = (Fy_yp - Fy_ym) / (2 * eps)
    
    return np.array([[dFx_dx, dFx_dy],
                     [dFy_dx, dFy_dy]])


def compute_dF_du_fd(
    compute_force_fn: Callable[[float, float, float, float], Tuple[float, float, float, float]],
    v: float, phi: float, x: float, y: float,
    eps_v: float = 1e-5,
    eps_phi: float = 1e-5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute ∂F/∂v and ∂F/∂phi via centered finite differences.
    
    Parameters
    ----------
    compute_force_fn : callable
        Function (v, phi, x, y) -> (field, U, Fx, Fy)
    v, phi : float
        Control parameters.
    x, y : float
        Particle position.
    eps_v, eps_phi : float
        FD step sizes.
        
    Returns
    -------
    dF_dv : np.ndarray, shape (2,)
        [∂Fx/∂v, ∂Fy/∂v]
    dF_dphi : np.ndarray, shape (2,)
        [∂Fx/∂phi, ∂Fy/∂phi]
    """
    # Perturb v
    _, _, Fx_vp, Fy_vp = compute_force_fn(v + eps_v, phi, x, y)
    _, _, Fx_vm, Fy_vm = compute_force_fn(v - eps_v, phi, x, y)
    
    dF_dv = np.array([(Fx_vp - Fx_vm) / (2 * eps_v),
                      (Fy_vp - Fy_vm) / (2 * eps_v)])
    
    # Perturb phi
    _, _, Fx_pp, Fy_pp = compute_force_fn(v, phi + eps_phi, x, y)
    _, _, Fx_pm, Fy_pm = compute_force_fn(v, phi - eps_phi, x, y)
    
    dF_dphi = np.array([(Fx_pp - Fx_pm) / (2 * eps_phi),
                        (Fy_pp - Fy_pm) / (2 * eps_phi)])
    
    return dF_dv, dF_dphi


def compute_dU_dx_fd(
    compute_force_fn: Callable[[float, float, float, float], Tuple[float, float, float, float]],
    v: float, phi: float, x: float, y: float,
    eps: float = 1e-7,
) -> np.ndarray:
    """
    Compute ∇_x U = [∂U/∂x, ∂U/∂y] via centered finite differences.
    
    Note: For Gor'kov potential, F = -∇U, so ∇U = -F.
    However, we compute directly from U for consistency.
    
    Parameters
    ----------
    compute_force_fn : callable
        Function (v, phi, x, y) -> (field, U, Fx, Fy)
    v, phi : float
        Control parameters.
    x, y : float
        Particle position.
    eps : float
        FD step size.
        
    Returns
    -------
    grad_U : np.ndarray, shape (2,)
        [∂U/∂x, ∂U/∂y]
    """
    _, U_xp, _, _ = compute_force_fn(v, phi, x + eps, y)
    _, U_xm, _, _ = compute_force_fn(v, phi, x - eps, y)
    _, U_yp, _, _ = compute_force_fn(v, phi, x, y + eps)
    _, U_ym, _, _ = compute_force_fn(v, phi, x, y - eps)
    
    dU_dx = (U_xp - U_xm) / (2 * eps)
    dU_dy = (U_yp - U_ym) / (2 * eps)
    
    return np.array([dU_dx, dU_dy])


def forward_rollout(
    controls: List[Tuple[float, float]],
    x0: float, y0: float,
    compute_force_fn: Callable[[float, float, float, float], Tuple[float, float, float, float]],
    dt: float,
    mobility: float,
    x_bounds: Tuple[float, float],
    y_bounds: Tuple[float, float],
) -> TrajectoryState:
    """
    Execute forward rollout and cache states for backward pass.
    
    Parameters
    ----------
    controls : list of (v, phi) tuples
        Control sequence for t=0..K-1.
    x0, y0 : float
        Initial particle position.
    compute_force_fn : callable
        (v, phi, x, y) -> (field, U, Fx, Fy)
    dt : float
        Time step.
    mobility : float
        Particle mobility μ = 1/(6πηa).
    x_bounds, y_bounds : tuple
        Domain bounds (min, max) for clamping.
        
    Returns
    -------
    state : TrajectoryState
        Cached trajectory data.
    """
    positions = [(x0, y0)]
    forces = []
    U_values = []
    
    x, y = x0, y0
    for v, phi in controls:
        _, U, Fx, Fy = compute_force_fn(v, phi, x, y)
        U_values.append(U)
        forces.append((Fx, Fy))
        
        # Overdamped step with clamping
        x_new = np.clip(x + dt * mobility * Fx, x_bounds[0], x_bounds[1])
        y_new = np.clip(y + dt * mobility * Fy, y_bounds[0], y_bounds[1])
        x, y = x_new, y_new
        positions.append((x, y))
    
    return TrajectoryState(
        positions=positions,
        controls=list(controls),
        forces=forces,
        U_values=U_values,
    )


def backward_pass(
    state: TrajectoryState,
    compute_force_fn: Callable[[float, float, float, float], Tuple[float, float, float, float]],
    compute_dU_du_fn: Callable[[float, float, float, float], Tuple[float, float]],
    dt: float,
    mobility: float,
    beta_terminal: float = 0.0,
    eps_x: float = 1e-7,
    eps_v: float = 1e-5,
    eps_phi: float = 1e-5,
) -> Tuple[List[Tuple[float, float]], List[np.ndarray]]:
    """
    Compute gradients via discrete-time adjoint backpropagation.
    
    Parameters
    ----------
    state : TrajectoryState
        Cached forward rollout data.
    compute_force_fn : callable
        (v, phi, x, y) -> (field, U, Fx, Fy)
    compute_dU_du_fn : callable
        (v, phi, x, y) -> (dU_dv, dU_dphi) - uses adjoint for this!
    dt : float
        Time step.
    mobility : float
        Particle mobility.
    beta_terminal : float
        Weight on terminal state.
    eps_x, eps_v, eps_phi : float
        FD step sizes.
        
    Returns
    -------
    gradients : list of (grad_v, grad_phi) tuples
        Gradient ∇_{u_t} J for each step t=0..K-1.
    lambda_history : list of np.ndarray
        Co-state λ_t for debugging (optional).
    """
    K = len(state.controls)
    gradients = []
    lambda_history = []
    
    # Terminal condition: λ_K = β * ∇_x U(x_K; u_{K-1})
    x_K, y_K = state.positions[-1]
    v_Km1, phi_Km1 = state.controls[-1]
    
    if beta_terminal > 0:
        grad_U_xK = compute_dU_dx_fd(compute_force_fn, v_Km1, phi_Km1, x_K, y_K, eps_x)
        lambda_t = beta_terminal * grad_U_xK
    else:
        lambda_t = np.zeros(2)
    
    lambda_history.append(lambda_t.copy())
    
    # Backward recursion: t = K-1, K-2, ..., 0
    for t in reversed(range(K)):
        v_t, phi_t = state.controls[t]
        x_t, y_t = state.positions[t]
        
        # === Direct term: ∇_{u_t} U(x_t; u_t) ===
        # Use the adjoint-based gradient (exact for U w.r.t. u)
        dU_dv, dU_dphi = compute_dU_du_fn(v_t, phi_t, x_t, y_t)
        
        # === Dynamics term: (∂x_{t+1}/∂u_t)^T λ_{t+1} ===
        # ∂x_{t+1}/∂u_t = dt * μ * ∂F/∂u_t
        dF_dv, dF_dphi = compute_dF_du_fd(compute_force_fn, v_t, phi_t, x_t, y_t, eps_v, eps_phi)
        
        dx_du_v = dt * mobility * dF_dv      # shape (2,)
        dx_du_phi = dt * mobility * dF_dphi  # shape (2,)
        
        # Dynamics contribution to gradient
        dyn_term_v = np.dot(lambda_t, dx_du_v)
        dyn_term_phi = np.dot(lambda_t, dx_du_phi)
        
        # Total gradient at step t
        grad_v = dU_dv + dyn_term_v
        grad_phi = dU_dphi + dyn_term_phi
        
        gradients.append((grad_v, grad_phi))
        
        # === Update λ for next iteration (moving backward) ===
        # λ_{t} = ∇_x U(x_t; u_t) + (∂x_{t+1}/∂x_t)^T λ_{t+1}
        # where ∂x_{t+1}/∂x_t = I + dt * μ * ∂F/∂x_t
        
        grad_U_xt = compute_dU_dx_fd(compute_force_fn, v_t, phi_t, x_t, y_t, eps_x)
        dF_dx = compute_dF_dx_fd(compute_force_fn, v_t, phi_t, x_t, y_t, eps_x)
        
        # ∂x_{t+1}/∂x_t = I + dt * μ * ∂F/∂x
        dx_dx = np.eye(2) + dt * mobility * dF_dx
        
        # λ_{t-1} = ∇_x U + dx_dx^T @ λ_t  (λ_t here is actually λ_{t+1} from perspective of t)
        lambda_new = grad_U_xt + dx_dx.T @ lambda_t
        lambda_t = lambda_new
        
        lambda_history.append(lambda_t.copy())
    
    # Reverse to get t=0..K-1 order
    gradients.reverse()
    lambda_history.reverse()
    
    return gradients, lambda_history


def compute_trajectory_gradient(
    controls: List[Tuple[float, float]],
    x0: float, y0: float,
    compute_force_fn: Callable[[float, float, float, float], Tuple[float, float, float, float]],
    compute_dU_du_fn: Callable[[float, float, float, float], Tuple[float, float]],
    dt: float,
    mobility: float,
    x_bounds: Tuple[float, float],
    y_bounds: Tuple[float, float],
    beta_terminal: float = 0.0,
    eps_x: float = 1e-7,
    eps_v: float = 1e-5,
    eps_phi: float = 1e-5,
) -> Tuple[List[Tuple[float, float]], TrajectoryState]:
    """
    Compute trajectory gradients via discrete-time adjoint.
    
    This is the main entry point for gradient computation.
    
    Parameters
    ----------
    controls : list of (v, phi) tuples
        Control sequence for t=0..K-1.
    x0, y0 : float
        Initial particle position.
    compute_force_fn : callable
        (v, phi, x, y) -> (field, U, Fx, Fy)
    compute_dU_du_fn : callable
        (v, phi, x, y) -> (dU_dv, dU_dphi)
    dt : float
        Time step.
    mobility : float
        Particle mobility μ = 1/(6πηa).
    x_bounds, y_bounds : tuple
        Domain bounds.
    beta_terminal : float
        Weight on terminal cost.
    eps_x, eps_v, eps_phi : float
        FD step sizes.
        
    Returns
    -------
    gradients : list of (grad_v, grad_phi)
        Gradient ∇_{u_t} J for each step.
    state : TrajectoryState
        Cached trajectory data.
    """
    # Forward pass
    state = forward_rollout(
        controls, x0, y0, compute_force_fn, dt, mobility, x_bounds, y_bounds
    )
    
    # Backward pass
    gradients, _ = backward_pass(
        state, compute_force_fn, compute_dU_du_fn, dt, mobility,
        beta_terminal, eps_x, eps_v, eps_phi
    )
    
    return gradients, state


def gradcheck_trajectory_scalar(
    controls: List[Tuple[float, float]],
    x0: float, y0: float,
    compute_force_fn: Callable[[float, float, float, float], Tuple[float, float, float, float]],
    compute_dU_du_fn: Callable[[float, float, float, float], Tuple[float, float]],
    dt: float,
    mobility: float,
    x_bounds: Tuple[float, float],
    y_bounds: Tuple[float, float],
    beta_terminal: float = 0.0,
    t_check: int = 0,
    param: str = 'v',
    eps_fd: float = 1e-5,
) -> Tuple[float, float, float]:
    """
    Verify gradient w.r.t. one scalar control parameter against FD.
    
    Parameters
    ----------
    t_check : int
        Time step to check gradient for.
    param : str
        'v' or 'phi' - which parameter to check.
    eps_fd : float
        FD step size for verification.
        
    Returns
    -------
    grad_adjoint : float
        Gradient from adjoint method.
    grad_fd : float
        Gradient from finite differences.
    rel_error : float
        Relative error |adj - fd| / (|adj| + |fd| + eps).
    """
    # Compute adjoint gradient
    gradients, state = compute_trajectory_gradient(
        controls, x0, y0, compute_force_fn, compute_dU_du_fn,
        dt, mobility, x_bounds, y_bounds, beta_terminal
    )
    
    if param == 'v':
        grad_adjoint = gradients[t_check][0]
    else:
        grad_adjoint = gradients[t_check][1]
    
    # Helper to compute J given perturbed control
    def compute_J(ctrl_list):
        st = forward_rollout(ctrl_list, x0, y0, compute_force_fn, dt, mobility, x_bounds, y_bounds)
        J = sum(st.U_values)
        if beta_terminal > 0:
            x_K, y_K = st.positions[-1]
            v_last, phi_last = ctrl_list[-1]
            _, U_K, _, _ = compute_force_fn(v_last, phi_last, x_K, y_K)
            J += beta_terminal * U_K
        return J
    
    # Perturb the specified parameter
    controls_plus = [c for c in controls]
    controls_minus = [c for c in controls]
    
    v_t, phi_t = controls[t_check]
    if param == 'v':
        controls_plus[t_check] = (v_t + eps_fd, phi_t)
        controls_minus[t_check] = (v_t - eps_fd, phi_t)
    else:
        controls_plus[t_check] = (v_t, phi_t + eps_fd)
        controls_minus[t_check] = (v_t, phi_t - eps_fd)
    
    J_plus = compute_J(controls_plus)
    J_minus = compute_J(controls_minus)
    
    grad_fd = (J_plus - J_minus) / (2 * eps_fd)
    
    rel_error = abs(grad_adjoint - grad_fd) / (abs(grad_adjoint) + abs(grad_fd) + 1e-30)
    
    return grad_adjoint, grad_fd, rel_error
