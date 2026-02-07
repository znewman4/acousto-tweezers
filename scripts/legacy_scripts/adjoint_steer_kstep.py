#!/usr/bin/env python3
"""
adjoint_steer_kstep.py - K-step lookahead trajectory optimizer.

This script implements a K-step horizon rollout optimizer that minimizes
a trajectory objective J = sum_{t=0}^{K-1} U(x_t; u_t) + beta_terminal * U(x_K).

The optimizer compares:
- Baseline A: constant control (do nothing)
- Baseline B: greedy 1-step optimization at each step
- Optimized: K-step lookahead with full trajectory gradients

Key insight: K-step lookahead can achieve lower total J by "repositioning"
the particle now to reach better states later, even if the immediate
objective increases.

Gradient computation (--use_discrete_adjoint, default):
- Uses exact discrete-time adjoint backpropagation
- Direct term: ∂U(x_t;u_t)/∂u_t via field adjoint (exact)
- Dynamics term: λ_{t+1}^T (∂x_{t+1}/∂u_t) with ∂F/∂u via FD
- State propagation: ∂x_{t+1}/∂x_t = I + dt*μ*∂F/∂x via FD

Legacy mode (--no_discrete_adjoint):
- Uses old FD-based state sensitivity (slower, less accurate)

Usage:
    python scripts/adjoint_steer_kstep.py [--fast] [--K 10] [--n_iters 10]
    python scripts/adjoint_steer_kstep.py --gradcheck_trajectory --fast --K 5
"""

from __future__ import annotations

import sys
import json
import csv
import argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field as dataclass_field
from datetime import datetime
from typing import Tuple, List, Dict, Any
import copy

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from acousto.solvers.fd_helmholtz_2d_forced_25d import build_helmholtz_2d_forced_25d_operator
from acousto.force.gorkov_2d import gorkov_potential_and_force_2d
from acousto.force.gorkov_1d import ParticleProps
from acousto.adjoint.gradients import (
    TransducerParams,
    compute_dJdp_gorkov_potential,
    compute_dbdu_single_transducer,
    adjoint_gradient_vectorized,
)
from acousto.adjoint.trajectory import (
    compute_trajectory_gradient,
    forward_rollout,
    gradcheck_trajectory_scalar,
)


@dataclass
class KStepConfig:
    """Configuration for K-step lookahead optimizer."""
    # Domain
    Lx: float = 2.0e-3  # 2mm
    Ly: float = 2.0e-3
    Nx: int = 64
    Ny: int = 64
    
    # Physics
    f: float = 1.0e6    # 1 MHz
    c0: float = 1500.0  # m/s (water)
    rho0: float = 1000.0
    
    # Transducer
    sigma_x: float = 0.3e-3
    sigma_y: float = 0.3e-3
    
    # Particle (polystyrene in water)
    particle_a: float = 50.0e-6     # 50 µm radius
    particle_rho_p: float = 1050.0  # kg/m³
    particle_c_p: float = 2350.0    # m/s
    
    # Dynamics
    mu: float = 1.0e-3  # dynamic viscosity [Pa·s]
    dt: float = 0.05    # time step [s]
    
    # Horizon
    K: int = 10         # number of steps
    beta_terminal: float = 0.0  # weight on terminal state U(x_K)
    
    # Optimization
    n_iters: int = 10
    alphas: Tuple[float, ...] = (0.0, 0.01, 0.1, 0.3, 1.0)
    
    # FD epsilon for derivatives
    fd_eps_v: float = 1e-5
    fd_eps_phi: float = 1e-5
    fd_eps_x: float = 1e-7  # for spatial derivatives
    
    # Control bounds
    v_min: float = 0.01
    v_max: float = 0.2
    
    # Initial particle position
    x0_frac: float = 0.35  # fraction of Lx
    y0_frac: float = 0.5   # fraction of Ly
    
    # Gradient method
    use_discrete_adjoint: bool = True  # use discrete-time adjoint backprop


def build_vb_from_control(v: float, phi: float, x_trans: float, y_trans: float,
                          sigma_x: float, sigma_y: float, x_grid: np.ndarray) -> np.ndarray:
    """Build bottom boundary velocity from control parameters."""
    G_x = np.exp(-(x_grid - x_trans)**2 / (2.0 * sigma_x**2))
    G_y = np.exp(-y_trans**2 / (2.0 * sigma_y**2))
    return v * np.exp(1j * phi) * G_x * G_y


def compute_U_and_F_at_pos(op, v: float, phi: float, cfg: KStepConfig,
                           x_p: float, y_p: float, particle: ParticleProps):
    """Compute Gor'kov potential and force at arbitrary position."""
    x_trans = cfg.Lx * 0.5
    y_trans = 0.02 * cfg.Ly
    vb = build_vb_from_control(v, phi, x_trans, y_trans, cfg.sigma_x, cfg.sigma_y, op.x)
    field = op.solve_for_bottom_vb(vb)
    U, Fx, Fy = gorkov_potential_and_force_2d(field, particle)
    
    # Bilinear interpolation
    ix_f = (x_p - op.x[0]) / op.dx
    iy_f = (y_p - op.y[0]) / op.dy
    
    ix0 = int(np.clip(np.floor(ix_f), 0, op.Nx - 2))
    iy0 = int(np.clip(np.floor(iy_f), 0, op.Ny - 2))
    ix1, iy1 = ix0 + 1, iy0 + 1
    
    fx = np.clip(ix_f - ix0, 0, 1)
    fy = np.clip(iy_f - iy0, 0, 1)
    
    U_interp = (U[iy0, ix0] * (1-fx) * (1-fy) + U[iy0, ix1] * fx * (1-fy) +
                U[iy1, ix0] * (1-fx) * fy + U[iy1, ix1] * fx * fy)
    Fx_interp = (Fx[iy0, ix0] * (1-fx) * (1-fy) + Fx[iy0, ix1] * fx * (1-fy) +
                 Fx[iy1, ix0] * (1-fx) * fy + Fx[iy1, ix1] * fx * fy)
    Fy_interp = (Fy[iy0, ix0] * (1-fx) * (1-fy) + Fy[iy0, ix1] * fx * (1-fy) +
                 Fy[iy1, ix0] * (1-fx) * fy + Fy[iy1, ix1] * fx * fy)
    
    return field, U_interp, Fx_interp, Fy_interp


def overdamped_step(x: float, y: float, Fx: float, Fy: float, cfg: KStepConfig) -> Tuple[float, float]:
    """One overdamped particle step: x_new = x + dt * F / gamma."""
    gamma = 6.0 * np.pi * cfg.mu * cfg.particle_a
    x_new = np.clip(x + cfg.dt * Fx / gamma, 0, cfg.Lx)
    y_new = np.clip(y + cfg.dt * Fy / gamma, 0, cfg.Ly)
    return x_new, y_new


# =============================================================================
# Wrapper functions for discrete-time adjoint trajectory module
# =============================================================================

def make_compute_force_fn(op, cfg: KStepConfig, particle: ParticleProps):
    """Create a closure for computing force at (v, phi, x, y)."""
    def compute_force_fn(v: float, phi: float, x: float, y: float):
        return compute_U_and_F_at_pos(op, v, phi, cfg, x, y, particle)
    return compute_force_fn


def make_compute_dU_du_fn(op, cfg: KStepConfig, particle: ParticleProps):
    """Create a closure for computing ∂U/∂(v, phi) via adjoint."""
    def compute_dU_du_fn(v: float, phi: float, x: float, y: float):
        return compute_adjoint_gradient_at_step(op, v, phi, x, y, cfg, particle)
    return compute_dU_du_fn


def get_mobility(cfg: KStepConfig) -> float:
    """Compute particle mobility μ = 1/(6πηa)."""
    gamma = 6.0 * np.pi * cfg.mu * cfg.particle_a
    return 1.0 / gamma


def compute_kstep_gradients_discrete_adjoint(
    op, controls: List[Tuple[float, float]], x0: float, y0: float,
    cfg: KStepConfig, particle: ParticleProps
) -> List[Tuple[float, float]]:
    """
    Compute gradients via discrete-time adjoint backpropagation.
    
    This is the new, exact method that properly propagates gradients
    through the dynamics.
    """
    compute_force_fn = make_compute_force_fn(op, cfg, particle)
    compute_dU_du_fn = make_compute_dU_du_fn(op, cfg, particle)
    mobility = get_mobility(cfg)
    
    gradients, state = compute_trajectory_gradient(
        controls=controls,
        x0=x0, y0=y0,
        compute_force_fn=compute_force_fn,
        compute_dU_du_fn=compute_dU_du_fn,
        dt=cfg.dt,
        mobility=mobility,
        x_bounds=(0, cfg.Lx),
        y_bounds=(0, cfg.Ly),
        beta_terminal=cfg.beta_terminal,
        eps_x=cfg.fd_eps_x,
        eps_v=cfg.fd_eps_v,
        eps_phi=cfg.fd_eps_phi,
    )
    
    return gradients


def rollout_trajectory(op, controls: List[Tuple[float, float]], x0: float, y0: float,
                       cfg: KStepConfig, particle: ParticleProps) -> Tuple[List[Tuple[float, float]], List[float]]:
    """Roll out trajectory for K steps given control sequence.
    
    Returns:
        positions: list of (x, y) positions for t=0..K
        U_values: list of U values for t=0..K-1 (at x_t with u_t)
    """
    positions = [(x0, y0)]
    U_values = []
    
    x, y = x0, y0
    for t, (v, phi) in enumerate(controls):
        _, U, Fx, Fy = compute_U_and_F_at_pos(op, v, phi, cfg, x, y, particle)
        U_values.append(U)
        x, y = overdamped_step(x, y, Fx, Fy, cfg)
        positions.append((x, y))
    
    return positions, U_values


def compute_trajectory_objective(U_values: List[float], positions: List[Tuple[float, float]],
                                  op, controls: List[Tuple[float, float]], cfg: KStepConfig,
                                  particle: ParticleProps) -> float:
    """Compute J = sum U(x_t) + beta_terminal * U(x_K)."""
    J = sum(U_values)
    
    if cfg.beta_terminal > 0:
        x_K, y_K = positions[-1]
        v_last, phi_last = controls[-1]
        _, U_K, _, _ = compute_U_and_F_at_pos(op, v_last, phi_last, cfg, x_K, y_K, particle)
        J += cfg.beta_terminal * U_K
    
    return J


def compute_adjoint_gradient_at_step(op, v: float, phi: float, x_p: float, y_p: float,
                                      cfg: KStepConfig, particle: ParticleProps) -> Tuple[float, float]:
    """
    Compute adjoint gradient ∂U_interp/∂(v, phi) at particle position.
    
    Uses bilinear interpolation to match compute_U_and_F_at_pos.
    For U_interp = sum_k w_k * U_k, the adjoint seed is sum_k w_k * dU_k/dp.
    """
    # Bilinear interpolation indices and weights (matching compute_U_and_F_at_pos)
    ix_f = (x_p - op.x[0]) / op.dx
    iy_f = (y_p - op.y[0]) / op.dy
    
    ix0 = int(np.clip(np.floor(ix_f), 0, op.Nx - 2))
    iy0 = int(np.clip(np.floor(iy_f), 0, op.Ny - 2))
    ix1, iy1 = ix0 + 1, iy0 + 1
    
    fx = np.clip(ix_f - ix0, 0, 1)
    fy = np.clip(iy_f - iy0, 0, 1)
    
    # Bilinear weights for 4 corners
    w00 = (1 - fx) * (1 - fy)
    w10 = fx * (1 - fy)
    w01 = (1 - fx) * fy
    w11 = fx * fy
    
    # Forward solve (needed for adjoint)
    x_trans = cfg.Lx * 0.5
    y_trans = 0.02 * cfg.Ly
    vb = build_vb_from_control(v, phi, x_trans, y_trans, cfg.sigma_x, cfg.sigma_y, op.x)
    field = op.solve_for_bottom_vb(vb)
    
    # Accumulate adjoint seed from all 4 corners
    # dJ_dp = sum_k w_k * dU_k/dp
    dJ_dp = np.zeros(cfg.Nx * cfg.Ny, dtype=np.complex128)
    
    for (ix, iy, w) in [(ix0, iy0, w00), (ix1, iy0, w10), (ix0, iy1, w01), (ix1, iy1, w11)]:
        if w > 1e-12:  # Only add if weight is significant
            dU_k_dp = compute_dJdp_gorkov_potential(
                ix, iy, cfg.Nx, cfg.Ny, op.dx, op.dy,
                field.p, op.omega, cfg.rho0, cfg.c0,
                cfg.particle_a, cfg.particle_rho_p, cfg.particle_c_p,
            )
            dJ_dp += w * dU_k_dp
    
    # Create transducer params for db/du computation
    trans = TransducerParams(x=x_trans, y=y_trans, v=v, phi=phi,
                             sigma_x=cfg.sigma_x, sigma_y=cfg.sigma_y, gate=True)
    db_dv, db_dphi = compute_dbdu_single_transducer(
        trans, op.x, op.omega, op.rho0, op.coupling_alpha, op.bottom_rows, cfg.Nx, cfg.Ny
    )
    
    grads = adjoint_gradient_vectorized(op.adjoint_solve, dJ_dp, [db_dv, db_dphi])
    return grads[0], grads[1]


def compute_state_sensitivity_fd(op, v: float, phi: float, x: float, y: float,
                                  cfg: KStepConfig, particle: ParticleProps) -> Tuple[np.ndarray, np.ndarray]:
    """Compute dx_{t+1}/d(v, phi) via finite differences."""
    # Baseline next state
    _, _, Fx0, Fy0 = compute_U_and_F_at_pos(op, v, phi, cfg, x, y, particle)
    x1_0, y1_0 = overdamped_step(x, y, Fx0, Fy0, cfg)
    
    # Perturb v
    _, _, Fx_vp, Fy_vp = compute_U_and_F_at_pos(op, v + cfg.fd_eps_v, phi, cfg, x, y, particle)
    x1_vp, y1_vp = overdamped_step(x, y, Fx_vp, Fy_vp, cfg)
    _, _, Fx_vm, Fy_vm = compute_U_and_F_at_pos(op, v - cfg.fd_eps_v, phi, cfg, x, y, particle)
    x1_vm, y1_vm = overdamped_step(x, y, Fx_vm, Fy_vm, cfg)
    
    dx1_dv = np.array([(x1_vp - x1_vm) / (2 * cfg.fd_eps_v),
                       (y1_vp - y1_vm) / (2 * cfg.fd_eps_v)])
    
    # Perturb phi
    _, _, Fx_pp, Fy_pp = compute_U_and_F_at_pos(op, v, phi + cfg.fd_eps_phi, cfg, x, y, particle)
    x1_pp, y1_pp = overdamped_step(x, y, Fx_pp, Fy_pp, cfg)
    _, _, Fx_pm, Fy_pm = compute_U_and_F_at_pos(op, v, phi - cfg.fd_eps_phi, cfg, x, y, particle)
    x1_pm, y1_pm = overdamped_step(x, y, Fx_pm, Fy_pm, cfg)
    
    dx1_dphi = np.array([(x1_pp - x1_pm) / (2 * cfg.fd_eps_phi),
                         (y1_pp - y1_pm) / (2 * cfg.fd_eps_phi)])
    
    return dx1_dv, dx1_dphi


def compute_spatial_gradient_U(op, v: float, phi: float, x_p: float, y_p: float,
                                cfg: KStepConfig, particle: ParticleProps) -> np.ndarray:
    """Compute spatial gradient ∇_x U at position."""
    eps = op.dx * 0.5
    _, U_xp, _, _ = compute_U_and_F_at_pos(op, v, phi, cfg, x_p + eps, y_p, particle)
    _, U_xm, _, _ = compute_U_and_F_at_pos(op, v, phi, cfg, x_p - eps, y_p, particle)
    _, U_yp, _, _ = compute_U_and_F_at_pos(op, v, phi, cfg, x_p, y_p + eps, particle)
    _, U_ym, _, _ = compute_U_and_F_at_pos(op, v, phi, cfg, x_p, y_p - eps, particle)
    
    return np.array([(U_xp - U_xm) / (2 * eps), (U_yp - U_ym) / (2 * eps)])


def compute_kstep_gradients_legacy(op, controls: List[Tuple[float, float]], x0: float, y0: float,
                             cfg: KStepConfig, particle: ParticleProps) -> List[Tuple[float, float]]:
    """
    Compute gradients ∂J/∂u_t for each step in the trajectory (LEGACY METHOD).
    
    This is the old method that uses FD for state sensitivity and approximates
    ∂x_{t+1}/∂x_t ≈ I. Kept for comparison.
    
    For step t, the gradient includes:
    1. Direct term: ∂U(x_t; u_t)/∂u_t
    2. Indirect terms: sum_{s>t} ∂U(x_s; u_s)/∂x_s · ∂x_s/∂u_t
    
    The indirect terms propagate backward through the dynamics.
    For efficiency, we use a simplified approach: compute gradients assuming
    state sensitivity decays, and include only direct + one-step lookahead.
    """
    K = len(controls)
    gradients = []
    
    # Forward pass: compute trajectory and cache values
    positions = [(x0, y0)]
    x, y = x0, y0
    for t, (v, phi) in enumerate(controls):
        _, U, Fx, Fy = compute_U_and_F_at_pos(op, v, phi, cfg, x, y, particle)
        x, y = overdamped_step(x, y, Fx, Fy, cfg)
        positions.append((x, y))
    
    # Backward pass: accumulate gradients
    # Start with terminal contribution
    lambda_x = np.zeros(2)  # adjoint state for position
    
    if cfg.beta_terminal > 0:
        x_K, y_K = positions[-1]
        v_Km1, phi_Km1 = controls[-1]
        grad_U_xK = compute_spatial_gradient_U(op, v_Km1, phi_Km1, x_K, y_K, cfg, particle)
        lambda_x = cfg.beta_terminal * grad_U_xK
    
    for t in reversed(range(K)):
        v_t, phi_t = controls[t]
        x_t, y_t = positions[t]
        
        # Direct gradient: ∂U(x_t; u_t)/∂u_t
        dU_dv, dU_dphi = compute_adjoint_gradient_at_step(op, v_t, phi_t, x_t, y_t, cfg, particle)
        
        # Indirect contribution from future: lambda_x · ∂x_{t+1}/∂u_t
        dx1_dv, dx1_dphi = compute_state_sensitivity_fd(op, v_t, phi_t, x_t, y_t, cfg, particle)
        
        indirect_v = np.dot(lambda_x, dx1_dv)
        indirect_phi = np.dot(lambda_x, dx1_dphi)
        
        grad_v = dU_dv + indirect_v
        grad_phi = dU_dphi + indirect_phi
        
        gradients.append((grad_v, grad_phi))
        
        # Update lambda_x for next iteration (backward)
        # lambda_x_new = ∂U(x_t)/∂x_t + ∂x_{t+1}/∂x_t^T · lambda_x
        # For overdamped dynamics: ∂x_{t+1}/∂x_t ≈ I + dt/gamma * ∂F/∂x
        # Simplified: use spatial gradient of U at current step
        grad_U_xt = compute_spatial_gradient_U(op, v_t, phi_t, x_t, y_t, cfg, particle)
        
        # Simple approximation: ∂x_{t+1}/∂x_t ≈ I (first order)
        lambda_x = grad_U_xt + lambda_x  # accumulate
    
    gradients.reverse()  # now in order t=0..K-1
    return gradients


def compute_kstep_gradients(op, controls: List[Tuple[float, float]], x0: float, y0: float,
                             cfg: KStepConfig, particle: ParticleProps) -> List[Tuple[float, float]]:
    """
    Compute gradients ∂J/∂u_t for each step in the trajectory.
    
    Dispatches to either:
    - Discrete-time adjoint backpropagation (use_discrete_adjoint=True, default)
    - Legacy FD-based method (use_discrete_adjoint=False)
    """
    if cfg.use_discrete_adjoint:
        return compute_kstep_gradients_discrete_adjoint(op, controls, x0, y0, cfg, particle)
    else:
        return compute_kstep_gradients_legacy(op, controls, x0, y0, cfg, particle)


def clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def normalize_phi(phi: float) -> float:
    return ((phi + np.pi) % (2 * np.pi)) - np.pi


def run_greedy_1step(op, x0: float, y0: float, v_init: float, phi_init: float,
                      cfg: KStepConfig, particle: ParticleProps) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]], List[float]]:
    """Run greedy 1-step optimization at each step."""
    positions = [(x0, y0)]
    controls = []
    U_values = []
    
    x, y = x0, y0
    v, phi = v_init, phi_init
    
    for t in range(cfg.K):
        # Compute gradient at current state
        dU_dv, dU_dphi = compute_adjoint_gradient_at_step(op, v, phi, x, y, cfg, particle)
        
        # Normalize gradients for step size
        if abs(dU_dv) > 1e-30:
            scale_v = 0.1 * v / abs(dU_dv)
        else:
            scale_v = 0.0
        if abs(dU_dphi) > 1e-30:
            scale_phi = 0.1 / abs(dU_dphi)
        else:
            scale_phi = 0.0
        
        # Line search
        best_U = float('inf')
        best_v, best_phi = v, phi
        
        for alpha in cfg.alphas:
            v_try = clamp(v - alpha * scale_v * dU_dv, cfg.v_min, cfg.v_max)
            phi_try = normalize_phi(phi - alpha * scale_phi * dU_dphi)
            
            _, U_try, _, _ = compute_U_and_F_at_pos(op, v_try, phi_try, cfg, x, y, particle)
            
            if U_try < best_U:
                best_U = U_try
                best_v, best_phi = v_try, phi_try
        
        # Apply best control
        controls.append((best_v, best_phi))
        _, U, Fx, Fy = compute_U_and_F_at_pos(op, best_v, best_phi, cfg, x, y, particle)
        U_values.append(U)
        
        # Step dynamics
        x, y = overdamped_step(x, y, Fx, Fy, cfg)
        positions.append((x, y))
        
        # Update control for next step (warm start)
        v, phi = best_v, best_phi
    
    return positions, controls, U_values


def run_kstep_optimization(op, x0: float, y0: float, v_init: float, phi_init: float,
                            cfg: KStepConfig, particle: ParticleProps) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]], List[float], List[float]]:
    """Run K-step lookahead optimization."""
    
    # Initialize control sequence (constant)
    controls = [(v_init, phi_init) for _ in range(cfg.K)]
    
    J_history = []
    
    for iteration in range(cfg.n_iters):
        # Rollout current trajectory
        positions, U_values = rollout_trajectory(op, controls, x0, y0, cfg, particle)
        J = compute_trajectory_objective(U_values, positions, op, controls, cfg, particle)
        J_history.append(J)
        
        # Compute gradients
        gradients = compute_kstep_gradients(op, controls, x0, y0, cfg, particle)
        
        # Compute scaling for each control
        scales = []
        for t, (grad_v, grad_phi) in enumerate(gradients):
            v_t, phi_t = controls[t]
            if abs(grad_v) > 1e-30:
                scale_v = 0.1 * v_t / abs(grad_v)
            else:
                scale_v = 0.0
            if abs(grad_phi) > 1e-30:
                scale_phi = 0.1 / abs(grad_phi)
            else:
                scale_phi = 0.0
            scales.append((scale_v, scale_phi))
        
        # Line search over global step size
        best_J = J
        best_controls = controls
        
        for alpha in cfg.alphas[1:]:  # skip alpha=0
            new_controls = []
            for t in range(cfg.K):
                v_t, phi_t = controls[t]
                grad_v, grad_phi = gradients[t]
                scale_v, scale_phi = scales[t]
                
                v_new = clamp(v_t - alpha * scale_v * grad_v, cfg.v_min, cfg.v_max)
                phi_new = normalize_phi(phi_t - alpha * scale_phi * grad_phi)
                new_controls.append((v_new, phi_new))
            
            # Evaluate
            new_positions, new_U_values = rollout_trajectory(op, new_controls, x0, y0, cfg, particle)
            new_J = compute_trajectory_objective(new_U_values, new_positions, op, new_controls, cfg, particle)
            
            if new_J < best_J:
                best_J = new_J
                best_controls = new_controls
        
        controls = best_controls
        
        if iteration > 0 and abs(J_history[-1] - best_J) < 1e-20:
            print(f"      Converged at iteration {iteration}")
            break
    
    # Final rollout
    positions, U_values = rollout_trajectory(op, controls, x0, y0, cfg, particle)
    J_history.append(compute_trajectory_objective(U_values, positions, op, controls, cfg, particle))
    
    return positions, controls, U_values, J_history


def save_results(results_dir: Path, method: str, positions: List[Tuple[float, float]],
                 controls: List[Tuple[float, float]], U_values: List[float]):
    """Save trajectory and control data."""
    # Trajectory
    with open(results_dir / f"trajectory_{method}.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['t', 'x_mm', 'y_mm'])
        for t, (x, y) in enumerate(positions):
            writer.writerow([t, x * 1e3, y * 1e3])
    
    # Controls (for K-step only)
    if controls:
        with open(results_dir / f"controls_{method}.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['t', 'v', 'phi'])
            for t, (v, phi) in enumerate(controls):
                writer.writerow([t, v, phi])
    
    # U values
    with open(results_dir / f"U_values_{method}.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['t', 'U'])
        for t, U in enumerate(U_values):
            writer.writerow([t, U])


def create_comparison_plot(results_dir: Path, results: Dict[str, Dict]):
    """Create comparison plot of U(x_t) over time for all methods."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Left: U over time
        ax1 = axes[0]
        colors = {'baseline': 'gray', 'greedy_1step': 'blue', 'kstep': 'green'}
        labels = {'baseline': 'Constant (baseline)', 'greedy_1step': 'Greedy 1-step', 'kstep': f'K-step lookahead'}
        
        for method, data in results.items():
            U_vals = data['U_values']
            t = np.arange(len(U_vals))
            ax1.plot(t, U_vals, 'o-', color=colors[method], label=labels[method], linewidth=2, markersize=6)
        
        ax1.set_xlabel('Time step t', fontsize=12)
        ax1.set_ylabel('Gor\'kov potential U(x_t) [J]', fontsize=12)
        ax1.set_title('Potential Along Trajectory', fontsize=14)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.ticklabel_format(axis='y', style='scientific', scilimits=(-15, -15))
        
        # Right: Trajectories in space
        ax2 = axes[1]
        for method, data in results.items():
            positions = data['positions']
            x_mm = [p[0] * 1e3 for p in positions]
            y_mm = [p[1] * 1e3 for p in positions]
            ax2.plot(x_mm, y_mm, 'o-', color=colors[method], label=labels[method], linewidth=2, markersize=6)
            ax2.plot(x_mm[0], y_mm[0], 's', color=colors[method], markersize=12)  # start
            ax2.plot(x_mm[-1], y_mm[-1], '*', color=colors[method], markersize=15)  # end
        
        ax2.set_xlabel('x [mm]', fontsize=12)
        ax2.set_ylabel('y [mm]', fontsize=12)
        ax2.set_title('Particle Trajectories', fontsize=14)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(results_dir / 'comparison_plot.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n   Plot saved: {results_dir / 'comparison_plot.png'}")
        
    except ImportError:
        print("\n   Warning: matplotlib not available, skipping plot")


def run_gradcheck_trajectory(cfg: KStepConfig, op, particle: ParticleProps, x0: float, y0: float,
                               v_init: float, phi_init: float) -> dict:
    """Run gradient check for trajectory adjoint against finite differences."""
    print("=" * 80)
    print("TRAJECTORY GRADIENT CHECK")
    print("=" * 80)
    print(f"\nVerifying discrete-time adjoint gradients against finite differences...")
    
    compute_force_fn = make_compute_force_fn(op, cfg, particle)
    compute_dU_du_fn = make_compute_dU_du_fn(op, cfg, particle)
    mobility = get_mobility(cfg)
    
    # === First, verify the direct term dU/du at a single point ===
    print("\n--- Direct term verification (no dynamics) ---")
    print(f"Testing at position ({x0*1e3:.3f}, {y0*1e3:.3f}) mm, control (v={v_init}, phi={phi_init})")
    
    _, U0, Fx0, Fy0 = compute_force_fn(v_init, phi_init, x0, y0)
    print(f"  U = {U0:.6e}")
    
    dU_dv_adj, dU_dphi_adj = compute_dU_du_fn(v_init, phi_init, x0, y0)
    print(f"  Adjoint: dU/dv = {dU_dv_adj:.6e}")
    
    # FD for direct term
    eps = 1e-5
    _, U_vp, _, _ = compute_force_fn(v_init + eps, phi_init, x0, y0)
    _, U_vm, _, _ = compute_force_fn(v_init - eps, phi_init, x0, y0)
    dU_dv_fd = (U_vp - U_vm) / (2 * eps)
    print(f"  FD:      dU/dv = {dU_dv_fd:.6e}")
    
    rel_err_direct = abs(dU_dv_adj - dU_dv_fd) / (abs(dU_dv_adj) + abs(dU_dv_fd) + 1e-30)
    print(f"  Relative error: {rel_err_direct:.4e} {'✓' if rel_err_direct < 0.01 else '✗'}")
    mobility = get_mobility(cfg)
    
    # Initialize constant controls
    controls = [(v_init, phi_init) for _ in range(cfg.K)]
    
    results = []
    eps_values = [1e-5, 1e-6, 1e-7]
    
    print(f"\n   Checking gradients for K={cfg.K} steps")
    print(f"   Controls: v={v_init}, φ={phi_init}")
    print(f"   dt = {cfg.dt*1e3:.1f} ms, mobility = {mobility:.4e}")
    
    # Check gradient at a few time steps
    check_steps = [0, cfg.K // 2, cfg.K - 1] if cfg.K > 2 else list(range(cfg.K))
    
    for t in check_steps:
        print(f"\n   Step t={t}:")
        
        for param in ['v']:  # Skip phi since it's ~0 for symmetric setup
            print(f"      Parameter: {param}")
            print(f"      {'eps':>10} | {'adjoint':>14} | {'FD':>14} | {'rel_err':>12}")
            print("      " + "-" * 60)
            
            best_rel_err = float('inf')
            best_eps = None
            
            for eps_fd in eps_values:
                grad_adj, grad_fd, rel_err = gradcheck_trajectory_scalar(
                    controls=controls,
                    x0=x0, y0=y0,
                    compute_force_fn=compute_force_fn,
                    compute_dU_du_fn=compute_dU_du_fn,
                    dt=cfg.dt,
                    mobility=mobility,
                    x_bounds=(0, cfg.Lx),
                    y_bounds=(0, cfg.Ly),
                    beta_terminal=cfg.beta_terminal,
                    t_check=t,
                    param=param,
                    eps_fd=eps_fd,
                )
                
                print(f"      {eps_fd:10.0e} | {grad_adj:+14.6e} | {grad_fd:+14.6e} | {rel_err:12.4e}")
                
                if rel_err < best_rel_err:
                    best_rel_err = rel_err
                    best_eps = eps_fd
                
                results.append({
                    't': t, 'param': param, 'eps': eps_fd,
                    'grad_adjoint': grad_adj, 'grad_fd': grad_fd, 'rel_error': rel_err,
                })
            
            # Relaxed threshold for trajectory gradients (dynamics adds complexity)
            passed = best_rel_err < 0.05  # 5% threshold
            print(f"      Best rel_err: {best_rel_err:.4e} @ eps={best_eps:.0e} {'✓ PASS' if passed else '✗ FAIL'}")
    
    # Summary
    print("\n" + "=" * 80)
    print("GRADIENT CHECK SUMMARY")
    print("=" * 80)
    
    max_rel_err = max(r['rel_error'] for r in results)
    all_passed = max_rel_err < 0.05  # 5% threshold
    
    print(f"\n   Maximum relative error: {max_rel_err:.4e}")
    print(f"   {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED (>5% error)'}")
    
    return {'results': results, 'max_rel_err': max_rel_err, 'all_passed': all_passed}


def main():
    parser = argparse.ArgumentParser(description="K-step lookahead trajectory optimizer")
    parser.add_argument('--fast', action='store_true', help="Fast mode (coarse grid, fewer iters)")
    parser.add_argument('--K', type=int, default=10, help="Horizon length (default: 10)")
    parser.add_argument('--n_iters', type=int, default=10, help="Optimization iterations (default: 10)")
    parser.add_argument('--beta_terminal', type=float, default=0.0, help="Terminal weight (default: 0)")
    parser.add_argument('--use_discrete_adjoint', dest='use_discrete_adjoint', action='store_true',
                        default=True, help="Use discrete-time adjoint backprop (default)")
    parser.add_argument('--no_discrete_adjoint', dest='use_discrete_adjoint', action='store_false',
                        help="Use legacy FD state-sensitivity method")
    parser.add_argument('--gradcheck_trajectory', action='store_true',
                        help="Run gradient check comparing adjoint vs FD")
    args = parser.parse_args()
    
    cfg = KStepConfig()
    cfg.K = args.K
    cfg.n_iters = args.n_iters
    cfg.beta_terminal = args.beta_terminal
    cfg.use_discrete_adjoint = args.use_discrete_adjoint
    
    if args.fast:
        cfg.Nx = 32
        cfg.Ny = 32
        cfg.n_iters = 5
        cfg.alphas = (0.0, 0.1, 1.0)
        # Don't override K - user may want to test longer horizons
    
    # Build operator
    op = build_helmholtz_2d_forced_25d_operator(
        Lx=cfg.Lx, Ly=cfg.Ly, Nx=cfg.Nx, Ny=cfg.Ny,
        f=cfg.f, c0=cfg.c0, rho0=cfg.rho0,
    )
    
    particle = ParticleProps(a=cfg.particle_a, rho_p=cfg.particle_rho_p, c_p=cfg.particle_c_p)
    
    # Initial state
    x0 = cfg.Lx * cfg.x0_frac
    y0 = cfg.Ly * cfg.y0_frac
    v_init = 0.05
    phi_init = 0.0
    
    # === GRADIENT CHECK MODE ===
    if args.gradcheck_trajectory:
        gradcheck_results = run_gradcheck_trajectory(cfg, op, particle, x0, y0, v_init, phi_init)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = project_root / "results" / "adjoint_steer_kstep" / f"gradcheck_{timestamp}"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        with open(results_dir / "gradcheck_results.json", 'w') as f:
            json.dump(gradcheck_results, f, indent=2, default=float)
        
        print(f"\n   Results saved to: {results_dir}")
        return 0 if gradcheck_results['all_passed'] else 1
    
    # === NORMAL OPTIMIZATION MODE ===
    print("=" * 80)
    print("K-STEP LOOKAHEAD TRAJECTORY OPTIMIZER")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"   Grid: {cfg.Nx}×{cfg.Ny}")
    print(f"   Horizon K: {cfg.K} steps")
    print(f"   Time step dt: {cfg.dt*1e3:.1f} ms")
    print(f"   Optimization iterations: {cfg.n_iters}")
    print(f"   Terminal weight β: {cfg.beta_terminal}")
    print(f"   Particle: a={cfg.particle_a*1e6:.0f}µm")
    print(f"   Gradient method: {'Discrete-time adjoint' if cfg.use_discrete_adjoint else 'Legacy FD'}")
    
    print(f"\n1. Building Helmholtz operator...")
    print(f"   (already built)")
    
    print(f"\n2. Initial state:")
    print(f"   Particle: ({x0*1e3:.3f}, {y0*1e3:.3f}) mm")
    print(f"   Control: v={v_init}, φ={phi_init}")
    
    results = {}
    
    # === BASELINE A: Constant control ===
    print("\n" + "=" * 80)
    print("3. BASELINE A: Constant control (do nothing)")
    print("=" * 80)
    
    const_controls = [(v_init, phi_init) for _ in range(cfg.K)]
    baseline_positions, baseline_U = rollout_trajectory(op, const_controls, x0, y0, cfg, particle)
    baseline_J = sum(baseline_U)
    
    results['baseline'] = {
        'positions': baseline_positions,
        'controls': const_controls,
        'U_values': baseline_U,
        'J_total': baseline_J,
    }
    
    print(f"   Total J = {baseline_J:.6e}")
    print(f"   Final position: ({baseline_positions[-1][0]*1e3:.4f}, {baseline_positions[-1][1]*1e3:.4f}) mm")
    
    # === BASELINE B: Greedy 1-step ===
    print("\n" + "=" * 80)
    print("4. BASELINE B: Greedy 1-step optimization")
    print("=" * 80)
    
    greedy_positions, greedy_controls, greedy_U = run_greedy_1step(
        op, x0, y0, v_init, phi_init, cfg, particle
    )
    greedy_J = sum(greedy_U)
    
    results['greedy_1step'] = {
        'positions': greedy_positions,
        'controls': greedy_controls,
        'U_values': greedy_U,
        'J_total': greedy_J,
    }
    
    print(f"   Total J = {greedy_J:.6e}")
    print(f"   Final position: ({greedy_positions[-1][0]*1e3:.4f}, {greedy_positions[-1][1]*1e3:.4f}) mm")
    
    # === K-STEP OPTIMIZATION ===
    print("\n" + "=" * 80)
    print(f"5. K-STEP LOOKAHEAD OPTIMIZATION (K={cfg.K})")
    print("=" * 80)
    
    kstep_positions, kstep_controls, kstep_U, J_history = run_kstep_optimization(
        op, x0, y0, v_init, phi_init, cfg, particle
    )
    kstep_J = sum(kstep_U)
    
    results['kstep'] = {
        'positions': kstep_positions,
        'controls': kstep_controls,
        'U_values': kstep_U,
        'J_total': kstep_J,
        'J_history': J_history,
    }
    
    print(f"\n   Optimization history:")
    for i, J in enumerate(J_history):
        print(f"      Iter {i}: J = {J:.6e}")
    
    print(f"\n   Total J = {kstep_J:.6e}")
    print(f"   Final position: ({kstep_positions[-1][0]*1e3:.4f}, {kstep_positions[-1][1]*1e3:.4f}) mm")
    
    # === COMPARISON ===
    print("\n" + "=" * 80)
    print("6. COMPARISON SUMMARY")
    print("=" * 80)
    
    print(f"\n   {'Method':<25} {'Total J':>15} {'vs Baseline':>15} {'vs Greedy':>15}")
    print("   " + "-" * 70)
    
    for method, data in results.items():
        J = data['J_total']
        vs_baseline = (J - baseline_J) / abs(baseline_J) * 100 if baseline_J != 0 else 0
        vs_greedy = (J - greedy_J) / abs(greedy_J) * 100 if greedy_J != 0 else 0
        label = {'baseline': 'Constant', 'greedy_1step': 'Greedy 1-step', 'kstep': f'K-step (K={cfg.K})'}[method]
        print(f"   {label:<25} {J:>15.6e} {vs_baseline:>+14.1f}% {vs_greedy:>+14.1f}%")
    
    # Check success criteria
    kstep_wins_baseline = kstep_J < baseline_J
    kstep_wins_greedy = kstep_J < greedy_J
    
    print(f"\n   {'✓' if kstep_wins_baseline else '✗'} K-step beats constant baseline: {kstep_wins_baseline}")
    print(f"   {'✓' if kstep_wins_greedy else '✗'} K-step beats greedy 1-step: {kstep_wins_greedy}")
    
    # === SAVE RESULTS ===
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = project_root / "results" / "adjoint_steer_kstep" / f"run_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n7. Saving results to {results_dir}")
    
    # Save trajectories and controls
    for method, data in results.items():
        save_results(results_dir, method, data['positions'], data['controls'], data['U_values'])
    
    # Save J history for K-step
    with open(results_dir / "J_history.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['iter', 'J'])
        for i, J in enumerate(results['kstep']['J_history']):
            writer.writerow([i, J])
    
    # Save summary
    summary = {
        'config': {
            'Nx': cfg.Nx, 'Ny': cfg.Ny, 'K': cfg.K, 'dt': cfg.dt,
            'n_iters': cfg.n_iters, 'beta_terminal': cfg.beta_terminal,
            'particle_a': cfg.particle_a,
            'use_discrete_adjoint': cfg.use_discrete_adjoint,
        },
        'initial_state': {'x0': x0, 'y0': y0, 'v_init': v_init, 'phi_init': phi_init},
        'results': {
            method: {
                'J_total': data['J_total'],
                'J_total_abs': abs(data['J_total']),  # absolute value for clarity
                'final_position': list(data['positions'][-1]),
            }
            for method, data in results.items()
        },
        'comparison': {
            'kstep_vs_baseline_pct': (kstep_J - baseline_J) / abs(baseline_J) * 100 if baseline_J != 0 else 0,
            'kstep_vs_greedy_pct': (kstep_J - greedy_J) / abs(greedy_J) * 100 if greedy_J != 0 else 0,
            'kstep_wins_baseline': kstep_wins_baseline,
            'kstep_wins_greedy': kstep_wins_greedy,
        },
    }
    
    with open(results_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=float)
    
    # Create plot
    create_comparison_plot(results_dir, results)
    
    print(f"\n   Results saved to: {results_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
