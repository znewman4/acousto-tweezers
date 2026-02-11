#!/usr/bin/env python3
"""
adjoint_circle_track_kstep.py - Circle tracking controller using adjoint gradients.

This script implements a K-step horizon optimizer that tracks a circular path
while maintaining a strong acoustic trap.

Objective (per-step):
    L_t = w_r * e_r^2  - w_prog * (F_t · t_hat)  + w_U * U(x_t; u_t) + w_reg * ||u_t - u_ref||^2

where:
    - e_r = ||x_t - c|| - R  (radial error)
    - t_hat = tangent unit vector (CCW around circle)
    - F_t · t_hat = tangent progress proxy (since overdamped v ∝ F)

Full trajectory objective:
    J = Σ_{t=0}^{K-1} L_t + β_T * (w_rT * e_r(x_K)^2 + w_UT * U(x_K))

The script compares:
    1. Baseline: constant control (do nothing)
    2. Greedy 1-step: optimize L_t at each step
    3. K-step: full trajectory optimization via discrete adjoint

Note: Uses Option 1 (F · t_hat progress proxy) for simpler adjoint recursion.
The objective L_t depends only on (x_t, u_t), avoiding explicit x_{t+1} dependence.

Usage:
    python scripts/adjoint_circle_track_kstep.py --fast
    python scripts/adjoint_circle_track_kstep.py --K 10 --n_iters 20 --R 0.3e-3
"""

from __future__ import annotations

import sys
import json
import csv
import argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple, List, Dict, Any, Callable, Optional
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


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class CircleTrackConfig:
    """Configuration for circle tracking optimizer."""
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
    
    # Circle geometry (centered in domain by default)
    cx: float = 1.0e-3  # circle center x
    cy: float = 1.0e-3  # circle center y
    R: float = 0.4e-3   # circle radius (0.4 mm)
    
    # Objective weights
    w_r: float = 1.0e12      # radial error weight (scaled for ~1e-6 m errors)
    w_prog: float = 1.0e-6   # tangent progress weight (F is ~1e-12 N)
    w_U: float = 1.0         # Gor'kov potential weight (U is ~1e-18 J)
    w_reg: float = 0.0       # regularization weight
    
    # Terminal weights (relative to per-step)
    beta_terminal: float = 1.0
    w_rT: float = 1.0        # terminal radial weight multiplier
    w_UT: float = 0.5        # terminal U weight multiplier
    
    # Optimization
    n_iters: int = 10
    alphas: Tuple[float, ...] = (0.0, 0.01, 0.05, 0.1, 0.3, 1.0)
    
    # FD epsilon for derivatives
    fd_eps_v: float = 1e-5
    fd_eps_phi: float = 1e-5
    fd_eps_x: float = 1e-7  # for spatial derivatives
    
    # Control bounds
    v_min: float = 0.01
    v_max: float = 0.2
    v_ref: float = 0.05     # reference control for regularization
    phi_ref: float = 0.0
    
    # Initial particle position (on the circle, at angle theta0)
    theta0: float = 0.0  # starting angle on circle (radians, 0 = right)
    
    # Motion direction
    ccw: bool = True  # counter-clockwise motion
    
    # Domain penalty (soft constraint to stay inside)
    domain_penalty: float = 1e6  # penalty for leaving domain


# =============================================================================
# Circle geometry helpers
# =============================================================================

def compute_circle_metrics(x: float, y: float, cfg: CircleTrackConfig) -> Dict[str, float]:
    """
    Compute circle-related metrics for a position.
    
    Returns:
        r: distance from circle center
        radial_err: r - R (positive = outside circle)
        r_hat: radial unit vector (x, y) pointing outward
        t_hat: tangent unit vector (tx, ty) in CCW direction
        theta: angle from center (radians)
    """
    dx = x - cfg.cx
    dy = y - cfg.cy
    r = np.sqrt(dx**2 + dy**2)
    
    if r < 1e-12:
        # At center - arbitrary direction
        r_hat_x, r_hat_y = 1.0, 0.0
    else:
        r_hat_x, r_hat_y = dx / r, dy / r
    
    # Tangent: perpendicular to radial, CCW direction
    if cfg.ccw:
        t_hat_x, t_hat_y = -r_hat_y, r_hat_x
    else:
        t_hat_x, t_hat_y = r_hat_y, -r_hat_x
    
    radial_err = r - cfg.R
    theta = np.arctan2(dy, dx)
    
    return {
        'r': r,
        'radial_err': radial_err,
        'r_hat': (r_hat_x, r_hat_y),
        't_hat': (t_hat_x, t_hat_y),
        'theta': theta,
    }


# =============================================================================
# Forward model (same as kstep script)
# =============================================================================

def build_vb_from_control(v: float, phi: float, x_trans: float, y_trans: float,
                          sigma_x: float, sigma_y: float, x_grid: np.ndarray) -> np.ndarray:
    """Build bottom boundary velocity from control parameters."""
    G_x = np.exp(-(x_grid - x_trans)**2 / (2.0 * sigma_x**2))
    G_y = np.exp(-y_trans**2 / (2.0 * sigma_y**2))
    return v * np.exp(1j * phi) * G_x * G_y


def compute_U_and_F_at_pos(op, v: float, phi: float, cfg: CircleTrackConfig,
                           x_p: float, y_p: float, particle: ParticleProps):
    """Compute Gor'kov potential and force at arbitrary position using bilinear interpolation."""
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
    
    return field, U, U_interp, Fx_interp, Fy_interp


def overdamped_step(x: float, y: float, Fx: float, Fy: float, cfg: CircleTrackConfig) -> Tuple[float, float]:
    """One overdamped particle step: x_new = x + dt * F / gamma."""
    gamma = 6.0 * np.pi * cfg.mu * cfg.particle_a
    x_new = np.clip(x + cfg.dt * Fx / gamma, 0, cfg.Lx)
    y_new = np.clip(y + cfg.dt * Fy / gamma, 0, cfg.Ly)
    return x_new, y_new


def get_mobility(cfg: CircleTrackConfig) -> float:
    """Compute particle mobility μ = 1/(6πηa)."""
    gamma = 6.0 * np.pi * cfg.mu * cfg.particle_a
    return 1.0 / gamma


# =============================================================================
# Circle tracking objective
# =============================================================================

def compute_step_objective(
    x: float, y: float, 
    Fx: float, Fy: float,
    U: float,
    v: float, phi: float,
    cfg: CircleTrackConfig,
) -> Dict[str, float]:
    """
    Compute per-step objective L_t and its components.
    
    L_t = w_r * e_r^2 - w_prog * (F · t_hat) + w_U * U + w_reg * ||u - u_ref||^2
    
    Note: We MINIMIZE L_t, so:
    - radial error term is positive (penalize being off-circle)
    - progress term is NEGATIVE (reward moving in tangent direction)
    - U term: lower U is better (typical Gor'kov trap)
    """
    metrics = compute_circle_metrics(x, y, cfg)
    e_r = metrics['radial_err']
    t_hat_x, t_hat_y = metrics['t_hat']
    
    # Tangent progress: F · t_hat (positive = good, moving in right direction)
    tangent_progress = Fx * t_hat_x + Fy * t_hat_y
    
    # Regularization: ||u - u_ref||^2
    reg = (v - cfg.v_ref)**2 + (phi - cfg.phi_ref)**2
    
    # Compute components
    L_radial = cfg.w_r * e_r**2
    L_prog = -cfg.w_prog * tangent_progress  # negative because we minimize, want positive progress
    L_U = cfg.w_U * U
    L_reg = cfg.w_reg * reg
    
    L_total = L_radial + L_prog + L_U + L_reg
    
    return {
        'L_total': L_total,
        'L_radial': L_radial,
        'L_prog': L_prog,
        'L_U': L_U,
        'L_reg': L_reg,
        'radial_err': e_r,
        'tangent_progress': tangent_progress,
        'r': metrics['r'],
        'theta': metrics['theta'],
    }


def compute_trajectory_objective(
    positions: List[Tuple[float, float]],
    controls: List[Tuple[float, float]],
    forces: List[Tuple[float, float]],
    U_values: List[float],
    cfg: CircleTrackConfig,
    terminal_U: Optional[float] = None,
) -> Dict[str, float]:
    """
    Compute full trajectory objective J and its components.
    
    J = Σ_{t=0}^{K-1} L_t + β_T * (w_rT * e_r(x_K)^2 + w_UT * U(x_K))
    """
    K = len(controls)
    
    J_total = 0.0
    J_radial = 0.0
    J_prog = 0.0
    J_U = 0.0
    J_reg = 0.0
    
    for t in range(K):
        x, y = positions[t]
        v, phi = controls[t]
        Fx, Fy = forces[t]
        U = U_values[t]
        
        step_obj = compute_step_objective(x, y, Fx, Fy, U, v, phi, cfg)
        
        J_total += step_obj['L_total']
        J_radial += step_obj['L_radial']
        J_prog += step_obj['L_prog']
        J_U += step_obj['L_U']
        J_reg += step_obj['L_reg']
    
    # Terminal cost
    J_terminal = 0.0
    if cfg.beta_terminal > 0:
        x_K, y_K = positions[-1]
        metrics_K = compute_circle_metrics(x_K, y_K, cfg)
        e_r_K = metrics_K['radial_err']
        
        J_terminal_radial = cfg.w_rT * cfg.w_r * e_r_K**2
        J_terminal_U = cfg.w_UT * cfg.w_U * (terminal_U if terminal_U is not None else U_values[-1])
        J_terminal = cfg.beta_terminal * (J_terminal_radial + J_terminal_U)
        
        J_total += J_terminal
        J_radial += cfg.beta_terminal * J_terminal_radial
        J_U += cfg.beta_terminal * J_terminal_U
    
    return {
        'J_total': J_total,
        'J_radial': J_radial,
        'J_prog': J_prog,
        'J_U': J_U,
        'J_reg': J_reg,
        'J_terminal': J_terminal,
    }


# =============================================================================
# Trajectory rollout
# =============================================================================

@dataclass
class TrajectoryState:
    """Cached trajectory data."""
    positions: List[Tuple[float, float]]
    controls: List[Tuple[float, float]]
    forces: List[Tuple[float, float]]
    U_values: List[float]
    U_fields: List[np.ndarray]  # Full U field for visualization


def rollout_trajectory(
    op, 
    controls: List[Tuple[float, float]], 
    x0: float, y0: float,
    cfg: CircleTrackConfig, 
    particle: ParticleProps,
    store_fields: bool = False,
) -> TrajectoryState:
    """
    Roll out trajectory for K steps given control sequence.
    """
    positions = [(x0, y0)]
    forces = []
    U_values = []
    U_fields = []
    
    x, y = x0, y0
    for t, (v, phi) in enumerate(controls):
        field, U_full, U, Fx, Fy = compute_U_and_F_at_pos(op, v, phi, cfg, x, y, particle)
        U_values.append(U)
        forces.append((Fx, Fy))
        if store_fields:
            U_fields.append(U_full)
        x, y = overdamped_step(x, y, Fx, Fy, cfg)
        positions.append((x, y))
    
    return TrajectoryState(
        positions=positions,
        controls=list(controls),
        forces=forces,
        U_values=U_values,
        U_fields=U_fields,
    )


# =============================================================================
# Gradient computation (discrete adjoint for circle objective)
# =============================================================================

def compute_dL_dx(x: float, y: float, Fx: float, Fy: float, v: float, phi: float,
                  cfg: CircleTrackConfig, eps: float = 1e-7) -> np.ndarray:
    """
    Compute ∂L_t/∂x_t via finite differences.
    
    L_t = w_r * e_r^2 - w_prog * (F · t_hat) + w_U * U + w_reg * ||u - u_ref||^2
    
    Note: F and t_hat both depend on x, but F also depends on the field.
    For the F-dependence on x, we use the same FD approach as the existing code.
    """
    # We'll use FD for the full L_t w.r.t. x
    # Need to compute L at perturbed positions
    
    def L_at_pos(xp, yp, Fx_val, Fy_val):
        """Compute L at position, assuming F is given (computed elsewhere)."""
        metrics = compute_circle_metrics(xp, yp, cfg)
        e_r = metrics['radial_err']
        t_hat_x, t_hat_y = metrics['t_hat']
        tangent_progress = Fx_val * t_hat_x + Fy_val * t_hat_y
        
        # Note: U and reg don't depend on x directly in this formulation
        # (U depends on x through the field, but we handle that separately)
        return cfg.w_r * e_r**2 - cfg.w_prog * tangent_progress
    
    L_xp = L_at_pos(x + eps, y, Fx, Fy)
    L_xm = L_at_pos(x - eps, y, Fx, Fy)
    L_yp = L_at_pos(x, y + eps, Fx, Fy)
    L_ym = L_at_pos(x, y - eps, Fx, Fy)
    
    dL_dx = (L_xp - L_xm) / (2 * eps)
    dL_dy = (L_yp - L_ym) / (2 * eps)
    
    return np.array([dL_dx, dL_dy])


def compute_dL_dF(x: float, y: float, cfg: CircleTrackConfig) -> np.ndarray:
    """
    Compute ∂L_t/∂F_t.
    
    The only F-dependent term in L_t is: -w_prog * (F · t_hat)
    So ∂L/∂F = -w_prog * t_hat
    """
    metrics = compute_circle_metrics(x, y, cfg)
    t_hat_x, t_hat_y = metrics['t_hat']
    return np.array([-cfg.w_prog * t_hat_x, -cfg.w_prog * t_hat_y])


def compute_adjoint_dU_du(op, v: float, phi: float, x_p: float, y_p: float,
                           cfg: CircleTrackConfig, particle: ParticleProps) -> Tuple[float, float]:
    """
    Compute adjoint gradient ∂U_interp/∂(v, phi) at particle position.
    Uses bilinear interpolation weights matching forward pass.
    """
    ix_f = (x_p - op.x[0]) / op.dx
    iy_f = (y_p - op.y[0]) / op.dy
    
    ix0 = int(np.clip(np.floor(ix_f), 0, op.Nx - 2))
    iy0 = int(np.clip(np.floor(iy_f), 0, op.Ny - 2))
    ix1, iy1 = ix0 + 1, iy0 + 1
    
    fx = np.clip(ix_f - ix0, 0, 1)
    fy = np.clip(iy_f - iy0, 0, 1)
    
    w00 = (1 - fx) * (1 - fy)
    w10 = fx * (1 - fy)
    w01 = (1 - fx) * fy
    w11 = fx * fy
    
    x_trans = cfg.Lx * 0.5
    y_trans = 0.02 * cfg.Ly
    vb = build_vb_from_control(v, phi, x_trans, y_trans, cfg.sigma_x, cfg.sigma_y, op.x)
    field = op.solve_for_bottom_vb(vb)
    
    dJ_dp = np.zeros(cfg.Nx * cfg.Ny, dtype=np.complex128)
    
    for (ix, iy, w) in [(ix0, iy0, w00), (ix1, iy0, w10), (ix0, iy1, w01), (ix1, iy1, w11)]:
        if w > 1e-12:
            dU_k_dp = compute_dJdp_gorkov_potential(
                ix, iy, cfg.Nx, cfg.Ny, op.dx, op.dy,
                field.p, op.omega, cfg.rho0, cfg.c0,
                cfg.particle_a, cfg.particle_rho_p, cfg.particle_c_p,
            )
            dJ_dp += w * dU_k_dp
    
    trans = TransducerParams(x=x_trans, y=y_trans, v=v, phi=phi,
                             sigma_x=cfg.sigma_x, sigma_y=cfg.sigma_y, gate=True)
    db_dv, db_dphi = compute_dbdu_single_transducer(
        trans, op.x, op.omega, op.rho0, op.coupling_alpha, op.bottom_rows, cfg.Nx, cfg.Ny
    )
    
    grads = adjoint_gradient_vectorized(op.adjoint_solve, dJ_dp, [db_dv, db_dphi])
    return grads[0], grads[1]


def compute_dF_du_fd(op, v: float, phi: float, x: float, y: float,
                      cfg: CircleTrackConfig, particle: ParticleProps) -> Tuple[np.ndarray, np.ndarray]:
    """Compute ∂F/∂(v, phi) via finite differences."""
    # Perturb v
    _, _, _, Fx_vp, Fy_vp = compute_U_and_F_at_pos(op, v + cfg.fd_eps_v, phi, cfg, x, y, particle)
    _, _, _, Fx_vm, Fy_vm = compute_U_and_F_at_pos(op, v - cfg.fd_eps_v, phi, cfg, x, y, particle)
    
    dF_dv = np.array([(Fx_vp - Fx_vm) / (2 * cfg.fd_eps_v),
                      (Fy_vp - Fy_vm) / (2 * cfg.fd_eps_v)])
    
    # Perturb phi
    _, _, _, Fx_pp, Fy_pp = compute_U_and_F_at_pos(op, v, phi + cfg.fd_eps_phi, cfg, x, y, particle)
    _, _, _, Fx_pm, Fy_pm = compute_U_and_F_at_pos(op, v, phi - cfg.fd_eps_phi, cfg, x, y, particle)
    
    dF_dphi = np.array([(Fx_pp - Fx_pm) / (2 * cfg.fd_eps_phi),
                        (Fy_pp - Fy_pm) / (2 * cfg.fd_eps_phi)])
    
    return dF_dv, dF_dphi


def compute_dF_dx_fd(op, v: float, phi: float, x: float, y: float,
                      cfg: CircleTrackConfig, particle: ParticleProps) -> np.ndarray:
    """Compute ∂F/∂x via finite differences."""
    eps = cfg.fd_eps_x
    
    _, _, _, Fx_xp, Fy_xp = compute_U_and_F_at_pos(op, v, phi, cfg, x + eps, y, particle)
    _, _, _, Fx_xm, Fy_xm = compute_U_and_F_at_pos(op, v, phi, cfg, x - eps, y, particle)
    _, _, _, Fx_yp, Fy_yp = compute_U_and_F_at_pos(op, v, phi, cfg, x, y + eps, particle)
    _, _, _, Fx_ym, Fy_ym = compute_U_and_F_at_pos(op, v, phi, cfg, x, y - eps, particle)
    
    dFx_dx = (Fx_xp - Fx_xm) / (2 * eps)
    dFy_dx = (Fy_xp - Fy_xm) / (2 * eps)
    dFx_dy = (Fx_yp - Fx_ym) / (2 * eps)
    dFy_dy = (Fy_yp - Fy_ym) / (2 * eps)
    
    return np.array([[dFx_dx, dFx_dy],
                     [dFy_dx, dFy_dy]])


def compute_dU_dx_fd(op, v: float, phi: float, x: float, y: float,
                      cfg: CircleTrackConfig, particle: ParticleProps) -> np.ndarray:
    """Compute ∇_x U via finite differences."""
    eps = cfg.fd_eps_x
    _, _, U_xp, _, _ = compute_U_and_F_at_pos(op, v, phi, cfg, x + eps, y, particle)
    _, _, U_xm, _, _ = compute_U_and_F_at_pos(op, v, phi, cfg, x - eps, y, particle)
    _, _, U_yp, _, _ = compute_U_and_F_at_pos(op, v, phi, cfg, x, y + eps, particle)
    _, _, U_ym, _, _ = compute_U_and_F_at_pos(op, v, phi, cfg, x, y - eps, particle)
    
    return np.array([(U_xp - U_xm) / (2 * eps), (U_yp - U_ym) / (2 * eps)])


def compute_circle_trajectory_gradients(
    op,
    controls: List[Tuple[float, float]],
    x0: float, y0: float,
    cfg: CircleTrackConfig,
    particle: ParticleProps,
) -> Tuple[List[Tuple[float, float]], TrajectoryState]:
    """
    Compute gradients ∂J/∂u_t for circle tracking objective via discrete adjoint.
    
    The objective is:
        J = Σ_t L_t + β_T * L_terminal
    
    where L_t = w_r * e_r^2 - w_prog * (F·t_hat) + w_U * U + w_reg * ||u-u_ref||^2
    
    Gradient decomposition:
        ∂J/∂u_t = ∂L_t/∂u_t + (∂x_{t+1}/∂u_t)^T λ_{t+1}
    
    where:
        ∂L_t/∂u_t = w_U * ∂U/∂u + w_reg * 2*(u-u_ref) + (∂L/∂F) · (∂F/∂u)
        
    The adjoint λ_t satisfies:
        λ_K = β_T * ∂L_terminal/∂x_K
        λ_t = ∂L_t/∂x_t + (∂x_{t+1}/∂x_t)^T λ_{t+1}
    """
    K = len(controls)
    mobility = get_mobility(cfg)
    
    # Forward pass
    state = rollout_trajectory(op, controls, x0, y0, cfg, particle, store_fields=True)
    
    gradients = []
    
    # Terminal adjoint
    x_K, y_K = state.positions[-1]
    v_Km1, phi_Km1 = controls[-1]
    Fx_Km1, Fy_Km1 = state.forces[-1]
    
    if cfg.beta_terminal > 0:
        # ∂L_terminal/∂x_K = 2 * w_rT * w_r * e_r * ∂e_r/∂x + w_UT * w_U * ∂U/∂x
        metrics_K = compute_circle_metrics(x_K, y_K, cfg)
        e_r_K = metrics_K['radial_err']
        r_hat_x, r_hat_y = metrics_K['r_hat']
        
        # ∂e_r/∂x = r_hat (since e_r = r - R and ∂r/∂x = r_hat)
        d_er_dx = np.array([r_hat_x, r_hat_y])
        
        dL_terminal_dx_radial = 2 * cfg.w_rT * cfg.w_r * e_r_K * d_er_dx
        dL_terminal_dx_U = cfg.w_UT * cfg.w_U * compute_dU_dx_fd(op, v_Km1, phi_Km1, x_K, y_K, cfg, particle)
        
        lambda_t = cfg.beta_terminal * (dL_terminal_dx_radial + dL_terminal_dx_U)
    else:
        lambda_t = np.zeros(2)
    
    # Backward pass
    for t in reversed(range(K)):
        v_t, phi_t = controls[t]
        x_t, y_t = state.positions[t]
        Fx_t, Fy_t = state.forces[t]
        U_t = state.U_values[t]
        
        # === Direct gradient ∂L_t/∂u_t ===
        
        # 1) w_U * ∂U/∂u (via adjoint)
        dU_dv, dU_dphi = compute_adjoint_dU_du(op, v_t, phi_t, x_t, y_t, cfg, particle)
        
        # 2) w_reg * 2*(u - u_ref)
        dreg_dv = cfg.w_reg * 2 * (v_t - cfg.v_ref)
        dreg_dphi = cfg.w_reg * 2 * (phi_t - cfg.phi_ref)
        
        # 3) (∂L/∂F) · (∂F/∂u) for the progress term
        dL_dF = compute_dL_dF(x_t, y_t, cfg)  # shape (2,)
        dF_dv, dF_dphi = compute_dF_du_fd(op, v_t, phi_t, x_t, y_t, cfg, particle)  # each shape (2,)
        
        dLprog_dv = np.dot(dL_dF, dF_dv)
        dLprog_dphi = np.dot(dL_dF, dF_dphi)
        
        # Total direct gradient
        direct_v = cfg.w_U * dU_dv + dreg_dv + dLprog_dv
        direct_phi = cfg.w_U * dU_dphi + dreg_dphi + dLprog_dphi
        
        # === Dynamics term (∂x_{t+1}/∂u_t)^T λ_{t+1} ===
        # ∂x_{t+1}/∂u_t = dt * μ * ∂F/∂u
        dx_du_v = cfg.dt * mobility * dF_dv
        dx_du_phi = cfg.dt * mobility * dF_dphi
        
        dyn_v = np.dot(lambda_t, dx_du_v)
        dyn_phi = np.dot(lambda_t, dx_du_phi)
        
        grad_v = direct_v + dyn_v
        grad_phi = direct_phi + dyn_phi
        
        gradients.append((grad_v, grad_phi))
        
        # === Update λ for backward recursion ===
        # λ_{t-1} = ∂L_t/∂x_t + (∂x_{t+1}/∂x_t)^T λ_t
        
        # ∂L_t/∂x_t has three components:
        # 1) w_r * 2 * e_r * ∂e_r/∂x
        metrics_t = compute_circle_metrics(x_t, y_t, cfg)
        e_r_t = metrics_t['radial_err']
        r_hat_x, r_hat_y = metrics_t['r_hat']
        dL_radial_dx = cfg.w_r * 2 * e_r_t * np.array([r_hat_x, r_hat_y])
        
        # 2) -w_prog * ∂(F·t_hat)/∂x
        # This is complex because both F and t_hat depend on x
        # Use FD for simplicity
        dL_prog_dx = compute_dL_dx(x_t, y_t, Fx_t, Fy_t, v_t, phi_t, cfg, cfg.fd_eps_x)
        
        # 3) w_U * ∂U/∂x
        dU_dx = compute_dU_dx_fd(op, v_t, phi_t, x_t, y_t, cfg, particle)
        dL_U_dx = cfg.w_U * dU_dx
        
        dL_dx_total = dL_radial_dx + dL_prog_dx + dL_U_dx
        
        # ∂x_{t+1}/∂x_t = I + dt * μ * ∂F/∂x
        dF_dx = compute_dF_dx_fd(op, v_t, phi_t, x_t, y_t, cfg, particle)
        dx_dx = np.eye(2) + cfg.dt * mobility * dF_dx
        
        lambda_t = dL_dx_total + dx_dx.T @ lambda_t
    
    gradients.reverse()
    return gradients, state


# =============================================================================
# Optimization methods
# =============================================================================

def clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def normalize_phi(phi: float) -> float:
    return ((phi + np.pi) % (2 * np.pi)) - np.pi


def run_baseline(
    op, x0: float, y0: float, v_init: float, phi_init: float,
    cfg: CircleTrackConfig, particle: ParticleProps,
) -> Tuple[TrajectoryState, Dict[str, float]]:
    """Run baseline with constant controls."""
    controls = [(v_init, phi_init) for _ in range(cfg.K)]
    state = rollout_trajectory(op, controls, x0, y0, cfg, particle, store_fields=True)
    
    # Compute terminal U
    x_K, y_K = state.positions[-1]
    _, _, U_K, _, _ = compute_U_and_F_at_pos(op, v_init, phi_init, cfg, x_K, y_K, particle)
    
    obj = compute_trajectory_objective(
        state.positions, state.controls, state.forces, state.U_values, cfg, terminal_U=U_K
    )
    return state, obj


def run_greedy_1step(
    op, x0: float, y0: float, v_init: float, phi_init: float,
    cfg: CircleTrackConfig, particle: ParticleProps,
) -> Tuple[TrajectoryState, Dict[str, float]]:
    """Run greedy 1-step optimization at each step."""
    positions = [(x0, y0)]
    controls = []
    forces = []
    U_values = []
    U_fields = []
    
    x, y = x0, y0
    v, phi = v_init, phi_init
    
    for t in range(cfg.K):
        # Evaluate current objective and gradient for 1-step
        _, U_full, U, Fx, Fy = compute_U_and_F_at_pos(op, v, phi, cfg, x, y, particle)
        
        # Compute gradient for single-step L_t
        dU_dv, dU_dphi = compute_adjoint_dU_du(op, v, phi, x, y, cfg, particle)
        dL_dF = compute_dL_dF(x, y, cfg)
        dF_dv, dF_dphi = compute_dF_du_fd(op, v, phi, x, y, cfg, particle)
        
        grad_v = cfg.w_U * dU_dv + np.dot(dL_dF, dF_dv)
        grad_phi = cfg.w_U * dU_dphi + np.dot(dL_dF, dF_dphi)
        
        # Adaptive scaling
        if abs(grad_v) > 1e-30:
            scale_v = 0.1 * v / abs(grad_v)
        else:
            scale_v = 0.0
        if abs(grad_phi) > 1e-30:
            scale_phi = 0.1 / abs(grad_phi)
        else:
            scale_phi = 0.0
        
        # Line search over L_t
        best_L = float('inf')
        best_v, best_phi = v, phi
        
        for alpha in cfg.alphas:
            v_try = clamp(v - alpha * scale_v * grad_v, cfg.v_min, cfg.v_max)
            phi_try = normalize_phi(phi - alpha * scale_phi * grad_phi)
            
            _, _, U_try, Fx_try, Fy_try = compute_U_and_F_at_pos(op, v_try, phi_try, cfg, x, y, particle)
            step_obj = compute_step_objective(x, y, Fx_try, Fy_try, U_try, v_try, phi_try, cfg)
            
            if step_obj['L_total'] < best_L:
                best_L = step_obj['L_total']
                best_v, best_phi = v_try, phi_try
        
        # Apply best control
        controls.append((best_v, best_phi))
        _, U_full, U, Fx, Fy = compute_U_and_F_at_pos(op, best_v, best_phi, cfg, x, y, particle)
        U_values.append(U)
        forces.append((Fx, Fy))
        U_fields.append(U_full)
        
        # Step dynamics
        x, y = overdamped_step(x, y, Fx, Fy, cfg)
        positions.append((x, y))
        
        # Warm start
        v, phi = best_v, best_phi
    
    state = TrajectoryState(positions, controls, forces, U_values, U_fields)
    
    # Terminal U
    x_K, y_K = positions[-1]
    _, _, U_K, _, _ = compute_U_and_F_at_pos(op, v, phi, cfg, x_K, y_K, particle)
    
    obj = compute_trajectory_objective(positions, controls, forces, U_values, cfg, terminal_U=U_K)
    return state, obj


def run_kstep_optimization(
    op, x0: float, y0: float, v_init: float, phi_init: float,
    cfg: CircleTrackConfig, particle: ParticleProps,
) -> Tuple[TrajectoryState, Dict[str, float], List[Dict[str, float]]]:
    """Run K-step lookahead optimization."""
    controls = [(v_init, phi_init) for _ in range(cfg.K)]
    
    J_history = []
    
    for iteration in range(cfg.n_iters):
        # Compute gradients
        gradients, state = compute_circle_trajectory_gradients(op, controls, x0, y0, cfg, particle)
        
        # Compute terminal U
        x_K, y_K = state.positions[-1]
        v_last, phi_last = controls[-1]
        _, _, U_K, _, _ = compute_U_and_F_at_pos(op, v_last, phi_last, cfg, x_K, y_K, particle)
        
        obj = compute_trajectory_objective(
            state.positions, state.controls, state.forces, state.U_values, cfg, terminal_U=U_K
        )
        J_history.append(obj)
        
        # Compute scaling
        scales = []
        for t, (grad_v, grad_phi) in enumerate(gradients):
            v_t, phi_t = controls[t]
            scale_v = 0.1 * v_t / abs(grad_v) if abs(grad_v) > 1e-30 else 0.0
            scale_phi = 0.1 / abs(grad_phi) if abs(grad_phi) > 1e-30 else 0.0
            scales.append((scale_v, scale_phi))
        
        # Line search
        best_J = obj['J_total']
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
            new_state = rollout_trajectory(op, new_controls, x0, y0, cfg, particle)
            x_K, y_K = new_state.positions[-1]
            v_last, phi_last = new_controls[-1]
            _, _, U_K, _, _ = compute_U_and_F_at_pos(op, v_last, phi_last, cfg, x_K, y_K, particle)
            
            new_obj = compute_trajectory_objective(
                new_state.positions, new_controls, new_state.forces, new_state.U_values, cfg, terminal_U=U_K
            )
            
            if new_obj['J_total'] < best_J:
                best_J = new_obj['J_total']
                best_controls = new_controls
        
        controls = best_controls
        
        # Check convergence
        if iteration > 0 and abs(J_history[-1]['J_total'] - best_J) < 1e-25:
            print(f"      Converged at iteration {iteration}")
            break
    
    # Final rollout with field storage for visualization
    state = rollout_trajectory(op, controls, x0, y0, cfg, particle, store_fields=True)
    x_K, y_K = state.positions[-1]
    v_last, phi_last = controls[-1]
    _, _, U_K, _, _ = compute_U_and_F_at_pos(op, v_last, phi_last, cfg, x_K, y_K, particle)
    
    final_obj = compute_trajectory_objective(
        state.positions, state.controls, state.forces, state.U_values, cfg, terminal_U=U_K
    )
    J_history.append(final_obj)
    
    return state, final_obj, J_history


# =============================================================================
# Output / Visualization
# =============================================================================

def save_trajectory_csv(
    path: Path,
    state: TrajectoryState,
    cfg: CircleTrackConfig,
):
    """Save trajectory data with circle metrics."""
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['t', 'x_mm', 'y_mm', 'U', 'Fx', 'Fy', 'r_mm', 'theta_rad', 'radial_err_um', 'tangent_progress'])
        
        for t in range(len(state.controls)):
            x, y = state.positions[t]
            Fx, Fy = state.forces[t]
            U = state.U_values[t]
            metrics = compute_circle_metrics(x, y, cfg)
            
            writer.writerow([
                t,
                x * 1e3,
                y * 1e3,
                U,
                Fx,
                Fy,
                metrics['r'] * 1e3,
                metrics['theta'],
                metrics['radial_err'] * 1e6,  # µm
                metrics['r'] * (cfg.R if t == 0 else 0),  # placeholder
            ])
        
        # Final position
        x_K, y_K = state.positions[-1]
        metrics_K = compute_circle_metrics(x_K, y_K, cfg)
        writer.writerow([len(state.controls), x_K*1e3, y_K*1e3, '', '', '', 
                         metrics_K['r']*1e3, metrics_K['theta'], metrics_K['radial_err']*1e6, ''])


def save_controls_csv(path: Path, state: TrajectoryState, cfg: CircleTrackConfig):
    """Save control sequence."""
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['t', 'v', 'phi'])
        for t, (v, phi) in enumerate(state.controls):
            writer.writerow([t, v, phi])


def save_J_history_csv(path: Path, J_history: List[Dict[str, float]]):
    """Save optimization history."""
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['iter', 'J_total', 'J_radial', 'J_prog', 'J_U', 'J_reg', 'J_terminal'])
        for i, obj in enumerate(J_history):
            writer.writerow([
                i,
                obj['J_total'],
                obj['J_radial'],
                obj['J_prog'],
                obj['J_U'],
                obj['J_reg'],
                obj.get('J_terminal', 0),
            ])


def create_gorkov_contour_gif(
    out_path: Path,
    state: TrajectoryState,
    op,
    cfg: CircleTrackConfig,
    show_controls: bool = False,
    title_prefix: str = "",
):
    """Create GIF of Gorkov contour evolution with particle trail and circle overlay."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
        import imageio.v2 as imageio
    except ImportError:
        print("   Warning: matplotlib or imageio not available, skipping GIF")
        return
    
    frames = []
    temp_dir = out_path.parent / f"_temp_frames_{out_path.stem}"
    temp_dir.mkdir(exist_ok=True)
    
    x_mm = op.x * 1e3
    y_mm = op.y * 1e3
    X, Y = np.meshgrid(x_mm, y_mm)
    
    for t in range(len(state.controls)):
        fig, ax = plt.subplots(figsize=(8, 7))
        
        # Gorkov potential contour
        U = state.U_fields[t]
        U_mean = np.mean(U)
        U_std = np.std(U)
        
        if U_std > 1e-25:
            Uvis = (U - U_mean) / (U_std + 1e-30)
            levels = np.linspace(-3, 3, 25)
            contourf = ax.contourf(X, Y, Uvis, levels=levels, cmap="RdBu_r", alpha=0.85)
            ax.contour(X, Y, Uvis, levels=levels[::2], colors="k", linewidths=0.3, alpha=0.15)
            cbar = fig.colorbar(contourf, ax=ax, label=f"U (σ units)")
        else:
            ax.set_facecolor("#ffffcc")
        
        # Circle target
        circle = Circle((cfg.cx * 1e3, cfg.cy * 1e3), cfg.R * 1e3,
                         fill=False, edgecolor='lime', linewidth=2, linestyle='--', label='Target circle')
        ax.add_patch(circle)
        
        # Trail (history up to t)
        trail = state.positions[:t+1]
        if len(trail) >= 2:
            tx = [p[0] * 1e3 for p in trail]
            ty = [p[1] * 1e3 for p in trail]
            n_pts = len(tx)
            for i in range(n_pts - 1):
                alpha_val = 0.3 + 0.7 * i / max(n_pts - 1, 1)
                color = (1-alpha_val) * np.array([0.5, 0.5, 0.5]) + alpha_val * np.array([0, 1, 1])
                ax.plot(tx[i:i+2], ty[i:i+2], linewidth=2.5, color=color, alpha=0.95)
        
        # Current particle position
        x_t, y_t = state.positions[t]
        ax.scatter(x_t * 1e3, y_t * 1e3, s=300, marker='o', color='red', 
                   edgecolors='white', linewidth=3, zorder=100, label='Particle')
        
        # Tangent arrow
        metrics = compute_circle_metrics(x_t, y_t, cfg)
        t_hat_x, t_hat_y = metrics['t_hat']
        arrow_scale = 0.15  # mm
        ax.arrow(x_t * 1e3, y_t * 1e3, t_hat_x * arrow_scale, t_hat_y * arrow_scale,
                 head_width=0.05, head_length=0.02, fc='yellow', ec='black', linewidth=1, zorder=101)
        
        # Force direction arrow (scaled)
        Fx, Fy = state.forces[t]
        F_mag = np.sqrt(Fx**2 + Fy**2) + 1e-30
        Fx_norm, Fy_norm = Fx / F_mag, Fy / F_mag
        ax.arrow(x_t * 1e3, y_t * 1e3, Fx_norm * arrow_scale * 0.8, Fy_norm * arrow_scale * 0.8,
                 head_width=0.04, head_length=0.015, fc='magenta', ec='black', linewidth=1, zorder=100)
        
        ax.set_xlim(x_mm[0], x_mm[-1])
        ax.set_ylim(y_mm[0], y_mm[-1])
        ax.set_xlabel('x (mm)', fontsize=11)
        ax.set_ylabel('y (mm)', fontsize=11)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)
        
        title = f"{title_prefix}Step {t}/{len(state.controls)-1}"
        if show_controls:
            v, phi = state.controls[t]
            title += f"  |  v={v:.4f}, φ={phi:.3f}"
        radial_err = metrics['radial_err'] * 1e6
        title += f"\nRadial err: {radial_err:.1f} µm"
        ax.set_title(title, fontsize=12)
        
        ax.legend(loc='upper right', fontsize=8)
        
        frame_path = temp_dir / f"frame_{t:03d}.png"
        fig.tight_layout()
        fig.savefig(frame_path, dpi=100)
        plt.close(fig)
        
        frames.append(imageio.imread(frame_path))
    
    # Create GIF
    imageio.mimsave(out_path, frames, duration=0.5)
    
    # Cleanup temp files
    for f in temp_dir.glob("*.png"):
        f.unlink()
    temp_dir.rmdir()
    
    print(f"   Saved GIF: {out_path}")


def create_comparison_plot(
    out_path: Path,
    results: Dict[str, Tuple[TrajectoryState, Dict[str, float]]],
    cfg: CircleTrackConfig,
):
    """Create comparison plot: U(t), radial error, cumulative tangent progress."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("   Warning: matplotlib not available, skipping plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = {'baseline': 'gray', 'greedy_1step': 'blue', 'kstep': 'green'}
    labels = {'baseline': 'Constant (baseline)', 'greedy_1step': 'Greedy 1-step', 'kstep': 'K-step lookahead'}
    
    # Top-left: U over time
    ax1 = axes[0, 0]
    for method, (state, _) in results.items():
        t = np.arange(len(state.U_values))
        ax1.plot(t, state.U_values, 'o-', color=colors[method], label=labels[method], linewidth=2, markersize=6)
    ax1.set_xlabel('Time step t', fontsize=11)
    ax1.set_ylabel("Gor'kov potential U(x_t) [J]", fontsize=11)
    ax1.set_title("Potential Along Trajectory", fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.ticklabel_format(axis='y', style='scientific', scilimits=(-18, -18))
    
    # Top-right: Radial error over time
    ax2 = axes[0, 1]
    for method, (state, _) in results.items():
        radial_errs = []
        for t in range(len(state.controls)):
            x, y = state.positions[t]
            metrics = compute_circle_metrics(x, y, cfg)
            radial_errs.append(abs(metrics['radial_err']) * 1e6)  # µm
        t = np.arange(len(radial_errs))
        ax2.plot(t, radial_errs, 'o-', color=colors[method], label=labels[method], linewidth=2, markersize=6)
    ax2.set_xlabel('Time step t', fontsize=11)
    ax2.set_ylabel('Radial error |r - R| [µm]', fontsize=11)
    ax2.set_title('Radial Tracking Error', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Bottom-left: Cumulative tangent progress
    ax3 = axes[1, 0]
    for method, (state, _) in results.items():
        cum_progress = [0.0]
        total = 0.0
        for t in range(len(state.controls)):
            x, y = state.positions[t]
            Fx, Fy = state.forces[t]
            metrics = compute_circle_metrics(x, y, cfg)
            t_hat_x, t_hat_y = metrics['t_hat']
            progress = Fx * t_hat_x + Fy * t_hat_y
            total += progress
            cum_progress.append(total)
        t = np.arange(len(cum_progress))
        ax3.plot(t, cum_progress, 'o-', color=colors[method], label=labels[method], linewidth=2, markersize=6)
    ax3.set_xlabel('Time step t', fontsize=11)
    ax3.set_ylabel('Cumulative F·t_hat [N]', fontsize=11)
    ax3.set_title('Cumulative Tangent Progress', fontsize=12)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.ticklabel_format(axis='y', style='scientific', scilimits=(-12, -12))
    
    # Bottom-right: Trajectories in space with circle
    ax4 = axes[1, 1]
    
    # Draw circle
    theta_circle = np.linspace(0, 2*np.pi, 100)
    circle_x = cfg.cx * 1e3 + cfg.R * 1e3 * np.cos(theta_circle)
    circle_y = cfg.cy * 1e3 + cfg.R * 1e3 * np.sin(theta_circle)
    ax4.plot(circle_x, circle_y, 'g--', linewidth=2, label='Target circle')
    
    for method, (state, _) in results.items():
        x_mm = [p[0] * 1e3 for p in state.positions]
        y_mm = [p[1] * 1e3 for p in state.positions]
        ax4.plot(x_mm, y_mm, 'o-', color=colors[method], label=labels[method], linewidth=2, markersize=6)
        ax4.plot(x_mm[0], y_mm[0], 's', color=colors[method], markersize=12)  # start
        ax4.plot(x_mm[-1], y_mm[-1], '*', color=colors[method], markersize=15)  # end
    
    ax4.set_xlabel('x [mm]', fontsize=11)
    ax4.set_ylabel('y [mm]', fontsize=11)
    ax4.set_title('Particle Trajectories', fontsize=12)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   Saved comparison plot: {out_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Circle tracking controller using adjoint gradients")
    
    # Mode flags
    parser.add_argument('--fast', action='store_true', help="Fast mode (coarse grid, fewer iters)")
    
    # Horizon and optimization
    parser.add_argument('--K', type=int, default=10, help="Horizon length (default: 10)")
    parser.add_argument('--n_iters', type=int, default=10, help="Optimization iterations (default: 10)")
    parser.add_argument('--dt', type=float, default=0.05, help="Time step [s] (default: 0.05)")
    parser.add_argument('--mobility', type=float, default=None, help="Override mobility (default: compute from particle)")
    
    # Circle geometry
    parser.add_argument('--cx', type=float, default=1.0e-3, help="Circle center x [m] (default: 1e-3)")
    parser.add_argument('--cy', type=float, default=1.0e-3, help="Circle center y [m] (default: 1e-3)")
    parser.add_argument('--R', type=float, default=0.4e-3, help="Circle radius [m] (default: 0.4e-3)")
    
    # Objective weights
    parser.add_argument('--w_r', type=float, default=1.0e12, help="Radial error weight")
    parser.add_argument('--w_prog', type=float, default=1.0e-6, help="Tangent progress weight")
    parser.add_argument('--w_U', type=float, default=1.0, help="Gor'kov potential weight")
    parser.add_argument('--w_reg', type=float, default=0.0, help="Regularization weight")
    parser.add_argument('--beta_terminal', type=float, default=1.0, help="Terminal cost weight")
    
    args = parser.parse_args()
    
    # Build config
    cfg = CircleTrackConfig()
    cfg.K = args.K
    cfg.n_iters = args.n_iters
    cfg.dt = args.dt
    cfg.cx = args.cx
    cfg.cy = args.cy
    cfg.R = args.R
    cfg.w_r = args.w_r
    cfg.w_prog = args.w_prog
    cfg.w_U = args.w_U
    cfg.w_reg = args.w_reg
    cfg.beta_terminal = args.beta_terminal
    
    if args.fast:
        cfg.Nx = 32
        cfg.Ny = 32
        cfg.n_iters = 5
        cfg.alphas = (0.0, 0.1, 0.3, 1.0)
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = project_root / "results" / "adjoint_circle_track_kstep" / f"run_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("CIRCLE TRACKING CONTROLLER (K-STEP ADJOINT)")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"   Grid: {cfg.Nx}×{cfg.Ny}")
    print(f"   Horizon K: {cfg.K} steps")
    print(f"   Time step dt: {cfg.dt*1e3:.1f} ms")
    print(f"   Circle: center=({cfg.cx*1e3:.2f}, {cfg.cy*1e3:.2f}) mm, R={cfg.R*1e3:.2f} mm")
    print(f"   Weights: w_r={cfg.w_r:.2e}, w_prog={cfg.w_prog:.2e}, w_U={cfg.w_U:.2e}, w_reg={cfg.w_reg:.2e}")
    print(f"   Terminal: β={cfg.beta_terminal}")
    print(f"   Optimization iterations: {cfg.n_iters}")
    
    # Build operator
    print(f"\n1. Building Helmholtz operator...")
    op = build_helmholtz_2d_forced_25d_operator(
        Lx=cfg.Lx, Ly=cfg.Ly, Nx=cfg.Nx, Ny=cfg.Ny,
        f=cfg.f, c0=cfg.c0, rho0=cfg.rho0,
    )
    
    particle = ParticleProps(a=cfg.particle_a, rho_p=cfg.particle_rho_p, c_p=cfg.particle_c_p)
    
    # Initial state: on circle at theta0
    x0 = cfg.cx + cfg.R * np.cos(cfg.theta0)
    y0 = cfg.cy + cfg.R * np.sin(cfg.theta0)
    v_init = 0.05
    phi_init = 0.0
    
    print(f"\n2. Initial state:")
    print(f"   Particle: ({x0*1e3:.3f}, {y0*1e3:.3f}) mm (on circle at θ={np.degrees(cfg.theta0):.0f}°)")
    print(f"   Control: v={v_init}, φ={phi_init}")
    
    results = {}
    
    # === BASELINE ===
    print("\n" + "=" * 80)
    print("3. BASELINE: Constant control")
    print("=" * 80)
    
    baseline_state, baseline_obj = run_baseline(op, x0, y0, v_init, phi_init, cfg, particle)
    results['baseline'] = (baseline_state, baseline_obj)
    
    print(f"   J_total = {baseline_obj['J_total']:.6e}")
    print(f"   J_radial = {baseline_obj['J_radial']:.6e}")
    print(f"   J_prog = {baseline_obj['J_prog']:.6e}")
    print(f"   J_U = {baseline_obj['J_U']:.6e}")
    x_K, y_K = baseline_state.positions[-1]
    metrics_K = compute_circle_metrics(x_K, y_K, cfg)
    print(f"   Final radial error: {metrics_K['radial_err']*1e6:.2f} µm")
    
    # === GREEDY 1-STEP ===
    print("\n" + "=" * 80)
    print("4. GREEDY 1-STEP: Optimize L_t at each step")
    print("=" * 80)
    
    greedy_state, greedy_obj = run_greedy_1step(op, x0, y0, v_init, phi_init, cfg, particle)
    results['greedy_1step'] = (greedy_state, greedy_obj)
    
    print(f"   J_total = {greedy_obj['J_total']:.6e}")
    print(f"   J_radial = {greedy_obj['J_radial']:.6e}")
    print(f"   J_prog = {greedy_obj['J_prog']:.6e}")
    print(f"   J_U = {greedy_obj['J_U']:.6e}")
    x_K, y_K = greedy_state.positions[-1]
    metrics_K = compute_circle_metrics(x_K, y_K, cfg)
    print(f"   Final radial error: {metrics_K['radial_err']*1e6:.2f} µm")
    
    # === K-STEP OPTIMIZATION ===
    print("\n" + "=" * 80)
    print(f"5. K-STEP LOOKAHEAD OPTIMIZATION (K={cfg.K})")
    print("=" * 80)
    
    kstep_state, kstep_obj, J_history = run_kstep_optimization(op, x0, y0, v_init, phi_init, cfg, particle)
    results['kstep'] = (kstep_state, kstep_obj)
    
    print(f"\n   Optimization history:")
    for i, obj in enumerate(J_history):
        print(f"      Iter {i}: J = {obj['J_total']:.6e}")
    
    print(f"\n   J_total = {kstep_obj['J_total']:.6e}")
    print(f"   J_radial = {kstep_obj['J_radial']:.6e}")
    print(f"   J_prog = {kstep_obj['J_prog']:.6e}")
    print(f"   J_U = {kstep_obj['J_U']:.6e}")
    x_K, y_K = kstep_state.positions[-1]
    metrics_K = compute_circle_metrics(x_K, y_K, cfg)
    print(f"   Final radial error: {metrics_K['radial_err']*1e6:.2f} µm")
    
    # === COMPARISON ===
    print("\n" + "=" * 80)
    print("6. COMPARISON SUMMARY")
    print("=" * 80)
    
    print(f"\n   {'Method':<25} {'J_total':>15} {'J_radial':>15} {'J_prog':>15} {'J_U':>15}")
    print("   " + "-" * 85)
    
    for method, (_, obj) in results.items():
        label = {'baseline': 'Constant', 'greedy_1step': 'Greedy 1-step', 'kstep': f'K-step (K={cfg.K})'}[method]
        print(f"   {label:<25} {obj['J_total']:>15.4e} {obj['J_radial']:>15.4e} {obj['J_prog']:>15.4e} {obj['J_U']:>15.4e}")
    
    # === SAVE RESULTS ===
    print("\n" + "=" * 80)
    print("7. SAVING RESULTS")
    print("=" * 80)
    
    # Trajectories
    for method, (state, _) in results.items():
        save_trajectory_csv(results_dir / f"trajectory_{method}.csv", state, cfg)
    
    # Controls for K-step
    save_controls_csv(results_dir / "controls_kstep.csv", kstep_state, cfg)
    
    # J history
    save_J_history_csv(results_dir / "J_history.csv", J_history)
    
    # Summary JSON
    summary = {
        'config': {
            'Nx': cfg.Nx, 'Ny': cfg.Ny, 'K': cfg.K, 'dt': cfg.dt,
            'cx_mm': cfg.cx * 1e3, 'cy_mm': cfg.cy * 1e3, 'R_mm': cfg.R * 1e3,
            'w_r': cfg.w_r, 'w_prog': cfg.w_prog, 'w_U': cfg.w_U, 'w_reg': cfg.w_reg,
            'beta_terminal': cfg.beta_terminal,
            'n_iters': cfg.n_iters,
        },
        'initial_state': {
            'x0_mm': x0 * 1e3, 'y0_mm': y0 * 1e3,
            'v_init': v_init, 'phi_init': phi_init,
        },
        'results': {},
    }
    
    for method, (state, obj) in results.items():
        x_final, y_final = state.positions[-1]
        metrics_final = compute_circle_metrics(x_final, y_final, cfg)
        summary['results'][method] = {
            'J_total': float(obj['J_total']),
            'J_radial': float(obj['J_radial']),
            'J_prog': float(obj['J_prog']),
            'J_U': float(obj['J_U']),
            'J_reg': float(obj['J_reg']),
            'J_terminal': float(obj.get('J_terminal', 0)),
            'final_position_mm': [x_final * 1e3, y_final * 1e3],
            'final_radial_err_um': metrics_final['radial_err'] * 1e6,
            'final_theta_rad': metrics_final['theta'],
        }
    
    with open(results_dir / "circle_track_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"   Saved trajectory CSVs")
    print(f"   Saved controls_kstep.csv")
    print(f"   Saved J_history.csv")
    print(f"   Saved circle_track_summary.json")
    
    # === GIFs ===
    print("\n   Creating visualizations...")
    
    # GIF for K-step (main result)
    create_gorkov_contour_gif(
        results_dir / "gorkov_contour_path.gif",
        kstep_state, op, cfg,
        show_controls=False,
        title_prefix="K-step: ",
    )
    
    create_gorkov_contour_gif(
        results_dir / "gorkov_contour_controls.gif",
        kstep_state, op, cfg,
        show_controls=True,
        title_prefix="K-step: ",
    )
    
    # Comparison PNG
    create_comparison_plot(results_dir / "comparison_plot.png", results, cfg)
    
    print(f"\n   Results saved to: {results_dir}")
    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
