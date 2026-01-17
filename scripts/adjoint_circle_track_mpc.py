#!/usr/bin/env python3
"""
adjoint_circle_track_mpc.py - Rolling-horizon MPC for circle tracking.

This script implements Model Predictive Control for acoustic tweezers circle tracking:
1. At each real timestep t:
   - Solve a K-step optimization from current state x_t
   - Apply only the first control u_t = u*(0)
   - Step forward one timestep with real dynamics
   - Repeat

Key advantages over open-loop K-step:
- Replans every step (or every replan_every steps), handling drift and disturbances
- Handles model mismatch better
- More robust to nonlinearity
- True closed-loop control behavior

Objective (per-step):
    L_t = w_r * e_r^2 
          - w_prog * (F_t · t_hat)      [force mode]
          - w_prog * (v · t_hat)        [displacement mode]
          + w_U * U(x_t; u_t) 
          + w_reg * ||u_t - u_ref||^2
          + w_du * ||u_t - u_{t-1}||^2   [control smoothness]

Usage:
    python scripts/adjoint_circle_track_mpc.py --fast
    python scripts/adjoint_circle_track_mpc.py --K 5 --T 50 --w_du 1e-3
    python scripts/adjoint_circle_track_mpc.py --K 10 --T 100 --replan_every 1 --progress_mode force
"""

from __future__ import annotations

import sys
import json
import csv
import argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import Tuple, List, Dict, Any, Optional
import time

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
class MPCConfig:
    """Configuration for MPC circle tracking."""
    # Domain
    Lx: float = 2.0e-3
    Ly: float = 2.0e-3
    Nx: int = 64
    Ny: int = 64
    
    # Physics
    f: float = 1.0e6
    c0: float = 1500.0
    rho0: float = 1000.0
    
    # Transducer
    sigma_x: float = 0.3e-3
    sigma_y: float = 0.3e-3
    
    # Particle
    particle_a: float = 50.0e-6
    particle_rho_p: float = 1050.0
    particle_c_p: float = 2350.0
    
    # Dynamics
    mu: float = 1.0e-3
    dt: float = 0.05
    
    # MPC parameters
    K: int = 5              # horizon length
    T: int = 50             # total executed steps
    replan_every: int = 1   # replan frequency (1 = every step)
    n_iters: int = 8        # optimization iterations per MPC solve
    
    # Circle geometry
    cx: float = 1.0e-3
    cy: float = 1.0e-3
    R: float = 0.4e-3
    ccw: bool = True
    
    # Objective weights
    w_r: float = 1.0e12
    w_prog: float = 1.0e-6
    w_U: float = 1.0
    w_reg: float = 0.0
    w_du: float = 0.0       # control smoothness weight
    
    # Progress mode
    progress_mode: str = "force"  # "force" or "displacement"
    
    # Terminal weights
    beta_terminal: float = 1.0
    w_rT: float = 1.0
    w_UT: float = 0.5
    
    # Optimization
    alphas: Tuple[float, ...] = (0.0, 0.01, 0.05, 0.1, 0.3, 1.0)
    
    # FD epsilons
    fd_eps_v: float = 1e-5
    fd_eps_phi: float = 1e-5
    fd_eps_x: float = 1e-7
    
    # Control bounds
    v_min: float = 0.01
    v_max: float = 0.2
    v_ref: float = 0.05
    phi_ref: float = 0.0
    
    # Initial position
    theta0: float = 0.0


# =============================================================================
# Helpers (imported pattern from kstep)
# =============================================================================

def compute_circle_metrics(x: float, y: float, cfg: MPCConfig) -> Dict[str, float]:
    """Compute circle-related metrics."""
    dx = x - cfg.cx
    dy = y - cfg.cy
    r = np.sqrt(dx**2 + dy**2)
    
    if r < 1e-12:
        r_hat_x, r_hat_y = 1.0, 0.0
    else:
        r_hat_x, r_hat_y = dx / r, dy / r
    
    if cfg.ccw:
        t_hat_x, t_hat_y = -r_hat_y, r_hat_x
    else:
        t_hat_x, t_hat_y = r_hat_y, -r_hat_x
    
    return {
        'r': r,
        'radial_err': r - cfg.R,
        'r_hat': (r_hat_x, r_hat_y),
        't_hat': (t_hat_x, t_hat_y),
        'theta': np.arctan2(dy, dx),
    }


def build_vb_from_control(v: float, phi: float, x_trans: float, y_trans: float,
                          sigma_x: float, sigma_y: float, x_grid: np.ndarray) -> np.ndarray:
    """Build bottom boundary velocity from control parameters."""
    G_x = np.exp(-(x_grid - x_trans)**2 / (2.0 * sigma_x**2))
    G_y = np.exp(-y_trans**2 / (2.0 * sigma_y**2))
    return v * np.exp(1j * phi) * G_x * G_y


def compute_U_and_F_at_pos(op, v: float, phi: float, cfg: MPCConfig,
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
    
    return field, U, U_interp, Fx_interp, Fy_interp


def overdamped_step(x: float, y: float, Fx: float, Fy: float, cfg: MPCConfig) -> Tuple[float, float]:
    """One overdamped particle step."""
    gamma = 6.0 * np.pi * cfg.mu * cfg.particle_a
    x_new = np.clip(x + cfg.dt * Fx / gamma, 0, cfg.Lx)
    y_new = np.clip(y + cfg.dt * Fy / gamma, 0, cfg.Ly)
    return x_new, y_new


def get_mobility(cfg: MPCConfig) -> float:
    """Compute particle mobility."""
    gamma = 6.0 * np.pi * cfg.mu * cfg.particle_a
    return 1.0 / gamma


def clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def normalize_phi(phi: float) -> float:
    return ((phi + np.pi) % (2 * np.pi)) - np.pi


# =============================================================================
# Objective computation with smoothness term
# =============================================================================

def compute_step_objective(
    x: float, y: float,
    Fx: float, Fy: float,
    U: float,
    v: float, phi: float,
    v_prev: Optional[float], phi_prev: Optional[float],
    cfg: MPCConfig,
) -> Dict[str, float]:
    """
    Compute per-step objective L_t with control smoothness.
    """
    metrics = compute_circle_metrics(x, y, cfg)
    e_r = metrics['radial_err']
    t_hat_x, t_hat_y = metrics['t_hat']
    
    tangent_progress = Fx * t_hat_x + Fy * t_hat_y
    
    L_radial = cfg.w_r * e_r**2
    L_prog = -cfg.w_prog * tangent_progress
    L_U = cfg.w_U * U
    L_reg = cfg.w_reg * ((v - cfg.v_ref)**2 + (phi - cfg.phi_ref)**2)
    
    # Control smoothness term
    L_du = 0.0
    if cfg.w_du > 0 and v_prev is not None and phi_prev is not None:
        L_du = cfg.w_du * ((v - v_prev)**2 + (phi - phi_prev)**2)
    
    L_total = L_radial + L_prog + L_U + L_reg + L_du
    
    return {
        'L_total': L_total,
        'L_radial': L_radial,
        'L_prog': L_prog,
        'L_U': L_U,
        'L_reg': L_reg,
        'L_du': L_du,
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
    cfg: MPCConfig,
    u_prev: Optional[Tuple[float, float]] = None,
    terminal_U: Optional[float] = None,
) -> Dict[str, float]:
    """Compute full trajectory objective with smoothness."""
    K = len(controls)
    
    J_total, J_radial, J_prog, J_U, J_reg, J_du = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    for t in range(K):
        x, y = positions[t]
        v, phi = controls[t]
        Fx, Fy = forces[t]
        U = U_values[t]
        
        if t == 0:
            v_prev, phi_prev = (u_prev[0], u_prev[1]) if u_prev else (None, None)
        else:
            v_prev, phi_prev = controls[t - 1]
        
        step_obj = compute_step_objective(x, y, Fx, Fy, U, v, phi, v_prev, phi_prev, cfg)
        
        J_total += step_obj['L_total']
        J_radial += step_obj['L_radial']
        J_prog += step_obj['L_prog']
        J_U += step_obj['L_U']
        J_reg += step_obj['L_reg']
        J_du += step_obj['L_du']
    
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
    
    return {
        'J_total': J_total,
        'J_radial': J_radial,
        'J_prog': J_prog,
        'J_U': J_U,
        'J_reg': J_reg,
        'J_du': J_du,
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
    U_fields: List[np.ndarray]


def rollout_trajectory(
    op,
    controls: List[Tuple[float, float]],
    x0: float, y0: float,
    cfg: MPCConfig,
    particle: ParticleProps,
    store_fields: bool = False,
) -> TrajectoryState:
    """Roll out trajectory for K steps."""
    positions = [(x0, y0)]
    forces = []
    U_values = []
    U_fields = []
    
    x, y = x0, y0
    for v, phi in controls:
        field, U_full, U, Fx, Fy = compute_U_and_F_at_pos(op, v, phi, cfg, x, y, particle)
        U_values.append(U)
        forces.append((Fx, Fy))
        if store_fields:
            U_fields.append(U_full)
        x, y = overdamped_step(x, y, Fx, Fy, cfg)
        positions.append((x, y))
    
    return TrajectoryState(positions, list(controls), forces, U_values, U_fields)


# =============================================================================
# Gradient computation (discrete adjoint)
# =============================================================================

def compute_dL_dF(x: float, y: float, cfg: MPCConfig) -> np.ndarray:
    """∂L/∂F = -w_prog * t_hat."""
    metrics = compute_circle_metrics(x, y, cfg)
    t_hat_x, t_hat_y = metrics['t_hat']
    return np.array([-cfg.w_prog * t_hat_x, -cfg.w_prog * t_hat_y])


def compute_adjoint_dU_du(op, v: float, phi: float, x_p: float, y_p: float,
                           cfg: MPCConfig, particle: ParticleProps) -> Tuple[float, float]:
    """Compute adjoint gradient ∂U_interp/∂(v, phi)."""
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
                      cfg: MPCConfig, particle: ParticleProps) -> Tuple[np.ndarray, np.ndarray]:
    """Compute ∂F/∂(v, phi) via FD."""
    _, _, _, Fx_vp, Fy_vp = compute_U_and_F_at_pos(op, v + cfg.fd_eps_v, phi, cfg, x, y, particle)
    _, _, _, Fx_vm, Fy_vm = compute_U_and_F_at_pos(op, v - cfg.fd_eps_v, phi, cfg, x, y, particle)
    dF_dv = np.array([(Fx_vp - Fx_vm) / (2 * cfg.fd_eps_v),
                      (Fy_vp - Fy_vm) / (2 * cfg.fd_eps_v)])
    
    _, _, _, Fx_pp, Fy_pp = compute_U_and_F_at_pos(op, v, phi + cfg.fd_eps_phi, cfg, x, y, particle)
    _, _, _, Fx_pm, Fy_pm = compute_U_and_F_at_pos(op, v, phi - cfg.fd_eps_phi, cfg, x, y, particle)
    dF_dphi = np.array([(Fx_pp - Fx_pm) / (2 * cfg.fd_eps_phi),
                        (Fy_pp - Fy_pm) / (2 * cfg.fd_eps_phi)])
    
    return dF_dv, dF_dphi


def compute_dF_dx_fd(op, v: float, phi: float, x: float, y: float,
                      cfg: MPCConfig, particle: ParticleProps) -> np.ndarray:
    """Compute ∂F/∂x via FD."""
    eps = cfg.fd_eps_x
    _, _, _, Fx_xp, Fy_xp = compute_U_and_F_at_pos(op, v, phi, cfg, x + eps, y, particle)
    _, _, _, Fx_xm, Fy_xm = compute_U_and_F_at_pos(op, v, phi, cfg, x - eps, y, particle)
    _, _, _, Fx_yp, Fy_yp = compute_U_and_F_at_pos(op, v, phi, cfg, x, y + eps, particle)
    _, _, _, Fx_ym, Fy_ym = compute_U_and_F_at_pos(op, v, phi, cfg, x, y - eps, particle)
    
    return np.array([[(Fx_xp - Fx_xm) / (2 * eps), (Fx_yp - Fx_ym) / (2 * eps)],
                     [(Fy_xp - Fy_xm) / (2 * eps), (Fy_yp - Fy_ym) / (2 * eps)]])


def compute_dU_dx_fd(op, v: float, phi: float, x: float, y: float,
                      cfg: MPCConfig, particle: ParticleProps) -> np.ndarray:
    """Compute ∇_x U via FD."""
    eps = cfg.fd_eps_x
    _, _, U_xp, _, _ = compute_U_and_F_at_pos(op, v, phi, cfg, x + eps, y, particle)
    _, _, U_xm, _, _ = compute_U_and_F_at_pos(op, v, phi, cfg, x - eps, y, particle)
    _, _, U_yp, _, _ = compute_U_and_F_at_pos(op, v, phi, cfg, x, y + eps, particle)
    _, _, U_ym, _, _ = compute_U_and_F_at_pos(op, v, phi, cfg, x, y - eps, particle)
    return np.array([(U_xp - U_xm) / (2 * eps), (U_yp - U_ym) / (2 * eps)])


def compute_dL_dx_fd(x: float, y: float, Fx: float, Fy: float, cfg: MPCConfig, eps: float = 1e-7) -> np.ndarray:
    """Compute ∂L/∂x for radial + progress terms via FD."""
    def L_at_pos(xp, yp):
        metrics = compute_circle_metrics(xp, yp, cfg)
        e_r = metrics['radial_err']
        t_hat_x, t_hat_y = metrics['t_hat']
        tangent_progress = Fx * t_hat_x + Fy * t_hat_y
        return cfg.w_r * e_r**2 - cfg.w_prog * tangent_progress
    
    dL_dx = (L_at_pos(x + eps, y) - L_at_pos(x - eps, y)) / (2 * eps)
    dL_dy = (L_at_pos(x, y + eps) - L_at_pos(x, y - eps)) / (2 * eps)
    return np.array([dL_dx, dL_dy])


def compute_mpc_gradients(
    op,
    controls: List[Tuple[float, float]],
    x0: float, y0: float,
    cfg: MPCConfig,
    particle: ParticleProps,
    u_prev: Optional[Tuple[float, float]] = None,
) -> Tuple[List[Tuple[float, float]], TrajectoryState]:
    """
    Compute gradients ∂J/∂u_t for MPC horizon via discrete adjoint.
    
    Includes control smoothness term w_du * ||u_t - u_{t-1}||^2.
    """
    K = len(controls)
    mobility = get_mobility(cfg)
    
    # Forward pass
    state = rollout_trajectory(op, controls, x0, y0, cfg, particle, store_fields=True)
    
    gradients = []
    
    # Terminal adjoint
    x_K, y_K = state.positions[-1]
    v_Km1, phi_Km1 = controls[-1]
    
    if cfg.beta_terminal > 0:
        metrics_K = compute_circle_metrics(x_K, y_K, cfg)
        e_r_K = metrics_K['radial_err']
        r_hat_x, r_hat_y = metrics_K['r_hat']
        
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
        
        # === Direct gradient ∂L_t/∂u_t ===
        
        # U term
        dU_dv, dU_dphi = compute_adjoint_dU_du(op, v_t, phi_t, x_t, y_t, cfg, particle)
        
        # Regularization
        dreg_dv = cfg.w_reg * 2 * (v_t - cfg.v_ref)
        dreg_dphi = cfg.w_reg * 2 * (phi_t - cfg.phi_ref)
        
        # Progress term
        dL_dF = compute_dL_dF(x_t, y_t, cfg)
        dF_dv, dF_dphi = compute_dF_du_fd(op, v_t, phi_t, x_t, y_t, cfg, particle)
        dLprog_dv = np.dot(dL_dF, dF_dv)
        dLprog_dphi = np.dot(dL_dF, dF_dphi)
        
        # Smoothness term: w_du * 2 * (u_t - u_{t-1})
        dLdu_dv = 0.0
        dLdu_dphi = 0.0
        if cfg.w_du > 0:
            if t == 0:
                if u_prev is not None:
                    dLdu_dv = cfg.w_du * 2 * (v_t - u_prev[0])
                    dLdu_dphi = cfg.w_du * 2 * (phi_t - u_prev[1])
            else:
                v_tm1, phi_tm1 = controls[t - 1]
                dLdu_dv = cfg.w_du * 2 * (v_t - v_tm1)
                dLdu_dphi = cfg.w_du * 2 * (phi_t - phi_tm1)
        
        # Smoothness from L_{t+1}: -w_du * 2 * (u_{t+1} - u_t) contributes to u_t gradient
        dLdu_from_tp1_dv = 0.0
        dLdu_from_tp1_dphi = 0.0
        if cfg.w_du > 0 and t < K - 1:
            v_tp1, phi_tp1 = controls[t + 1]
            dLdu_from_tp1_dv = -cfg.w_du * 2 * (v_tp1 - v_t)
            dLdu_from_tp1_dphi = -cfg.w_du * 2 * (phi_tp1 - phi_t)
        
        direct_v = cfg.w_U * dU_dv + dreg_dv + dLprog_dv + dLdu_dv + dLdu_from_tp1_dv
        direct_phi = cfg.w_U * dU_dphi + dreg_dphi + dLprog_dphi + dLdu_dphi + dLdu_from_tp1_dphi
        
        # === Dynamics term ===
        dx_du_v = cfg.dt * mobility * dF_dv
        dx_du_phi = cfg.dt * mobility * dF_dphi
        
        dyn_v = np.dot(lambda_t, dx_du_v)
        dyn_phi = np.dot(lambda_t, dx_du_phi)
        
        grad_v = direct_v + dyn_v
        grad_phi = direct_phi + dyn_phi
        
        gradients.append((grad_v, grad_phi))
        
        # === Update λ ===
        metrics_t = compute_circle_metrics(x_t, y_t, cfg)
        e_r_t = metrics_t['radial_err']
        r_hat_x, r_hat_y = metrics_t['r_hat']
        
        dL_radial_dx = cfg.w_r * 2 * e_r_t * np.array([r_hat_x, r_hat_y])
        dL_prog_dx = compute_dL_dx_fd(x_t, y_t, Fx_t, Fy_t, cfg, cfg.fd_eps_x)
        dU_dx = compute_dU_dx_fd(op, v_t, phi_t, x_t, y_t, cfg, particle)
        dL_U_dx = cfg.w_U * dU_dx
        
        dL_dx_total = dL_radial_dx + dL_prog_dx + dL_U_dx
        
        dF_dx = compute_dF_dx_fd(op, v_t, phi_t, x_t, y_t, cfg, particle)
        dx_dx = np.eye(2) + cfg.dt * mobility * dF_dx
        
        lambda_t = dL_dx_total + dx_dx.T @ lambda_t
    
    gradients.reverse()
    return gradients, state


# =============================================================================
# MPC inner optimization (K-step solve)
# =============================================================================

def solve_mpc_horizon(
    op,
    x0: float, y0: float,
    u_init: List[Tuple[float, float]],
    cfg: MPCConfig,
    particle: ParticleProps,
    u_prev: Optional[Tuple[float, float]] = None,
    verbose: bool = False,
) -> Tuple[List[Tuple[float, float]], float, List[float]]:
    """
    Solve K-step MPC optimization from current state.
    
    Returns:
        controls: optimized control sequence
        J_final: final objective value
        J_history: optimization history
    """
    controls = list(u_init)
    J_history = []
    
    for iteration in range(cfg.n_iters):
        gradients, state = compute_mpc_gradients(op, controls, x0, y0, cfg, particle, u_prev)
        
        # Terminal U
        x_K, y_K = state.positions[-1]
        v_last, phi_last = controls[-1]
        _, _, U_K, _, _ = compute_U_and_F_at_pos(op, v_last, phi_last, cfg, x_K, y_K, particle)
        
        obj = compute_trajectory_objective(
            state.positions, state.controls, state.forces, state.U_values, cfg, u_prev, U_K
        )
        J_history.append(obj['J_total'])
        
        if verbose and iteration % 2 == 0:
            print(f"        iter {iteration}: J = {obj['J_total']:.4e}")
        
        # Scaling
        scales = []
        for t, (grad_v, grad_phi) in enumerate(gradients):
            v_t, _ = controls[t]
            scale_v = 0.1 * v_t / abs(grad_v) if abs(grad_v) > 1e-30 else 0.0
            scale_phi = 0.1 / abs(grad_phi) if abs(grad_phi) > 1e-30 else 0.0
            scales.append((scale_v, scale_phi))
        
        # Line search
        best_J = obj['J_total']
        best_controls = controls
        
        for alpha in cfg.alphas[1:]:
            new_controls = []
            for t in range(cfg.K):
                v_t, phi_t = controls[t]
                grad_v, grad_phi = gradients[t]
                scale_v, scale_phi = scales[t]
                
                v_new = clamp(v_t - alpha * scale_v * grad_v, cfg.v_min, cfg.v_max)
                phi_new = normalize_phi(phi_t - alpha * scale_phi * grad_phi)
                new_controls.append((v_new, phi_new))
            
            new_state = rollout_trajectory(op, new_controls, x0, y0, cfg, particle)
            x_K, y_K = new_state.positions[-1]
            v_last, phi_last = new_controls[-1]
            _, _, U_K, _, _ = compute_U_and_F_at_pos(op, v_last, phi_last, cfg, x_K, y_K, particle)
            
            new_obj = compute_trajectory_objective(
                new_state.positions, new_controls, new_state.forces, new_state.U_values, cfg, u_prev, U_K
            )
            
            if new_obj['J_total'] < best_J:
                best_J = new_obj['J_total']
                best_controls = new_controls
        
        controls = best_controls
        
        # Check convergence
        if iteration > 0 and abs(J_history[-1] - best_J) < 1e-25:
            break
    
    J_history.append(best_J)
    return controls, best_J, J_history


# =============================================================================
# MPC main loop
# =============================================================================

@dataclass
class MPCResult:
    """Results from MPC run."""
    positions: List[Tuple[float, float]]       # T+1 positions
    applied_controls: List[Tuple[float, float]] # T applied controls
    forces: List[Tuple[float, float]]           # T forces
    U_values: List[float]                       # T potentials
    U_fields: List[np.ndarray]                  # T potential fields
    inner_J_history: List[List[float]]          # T × n_iters inner optimization
    per_step_metrics: List[Dict[str, float]]    # T step metrics


def run_mpc(
    op,
    x0: float, y0: float,
    v_init: float, phi_init: float,
    cfg: MPCConfig,
    particle: ParticleProps,
    verbose: bool = True,
) -> MPCResult:
    """
    Run rolling-horizon MPC for T steps.
    """
    positions = [(x0, y0)]
    applied_controls = []
    forces = []
    U_values = []
    U_fields = []
    inner_J_history = []
    per_step_metrics = []
    
    x, y = x0, y0
    
    # Initialize control sequence (warm start)
    u_horizon = [(v_init, phi_init) for _ in range(cfg.K)]
    u_prev = None
    
    if verbose:
        print(f"\n   Running MPC: T={cfg.T} steps, K={cfg.K} horizon, replan every {cfg.replan_every} steps")
        print(f"   Progress: ", end="", flush=True)
    
    start_time = time.time()
    
    for t in range(cfg.T):
        # Replan if needed
        if t % cfg.replan_every == 0:
            u_horizon, J_opt, J_hist = solve_mpc_horizon(
                op, x, y, u_horizon, cfg, particle, u_prev, verbose=False
            )
            inner_J_history.append(J_hist)
        
        # Apply first control
        v_apply, phi_apply = u_horizon[0]
        applied_controls.append((v_apply, phi_apply))
        
        # Compute physics at current state
        field, U_full, U, Fx, Fy = compute_U_and_F_at_pos(op, v_apply, phi_apply, cfg, x, y, particle)
        U_values.append(U)
        forces.append((Fx, Fy))
        U_fields.append(U_full)
        
        # Compute step metrics
        v_prev_step = applied_controls[-2][0] if len(applied_controls) > 1 else None
        phi_prev_step = applied_controls[-2][1] if len(applied_controls) > 1 else None
        step_metrics = compute_step_objective(x, y, Fx, Fy, U, v_apply, phi_apply, v_prev_step, phi_prev_step, cfg)
        per_step_metrics.append(step_metrics)
        
        # Step dynamics
        x_new, y_new = overdamped_step(x, y, Fx, Fy, cfg)
        positions.append((x_new, y_new))
        
        # Shift horizon (warm start for next MPC solve)
        u_prev = (v_apply, phi_apply)
        u_horizon = u_horizon[1:] + [u_horizon[-1]]  # shift left, repeat last
        
        x, y = x_new, y_new
        
        if verbose and (t + 1) % 10 == 0:
            print(f"{t+1}", end=" ", flush=True)
    
    elapsed = time.time() - start_time
    
    if verbose:
        print(f"\n   Completed in {elapsed:.1f}s ({elapsed/cfg.T*1000:.1f} ms/step)")
    
    return MPCResult(
        positions=positions,
        applied_controls=applied_controls,
        forces=forces,
        U_values=U_values,
        U_fields=U_fields,
        inner_J_history=inner_J_history,
        per_step_metrics=per_step_metrics,
    )


# =============================================================================
# Visualization
# =============================================================================

def create_mpc_gif(
    out_path: Path,
    result: MPCResult,
    op,
    cfg: MPCConfig,
    title_prefix: str = "MPC: ",
):
    """Create GIF of MPC trajectory."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
        import imageio.v2 as imageio
    except ImportError:
        print("   Warning: matplotlib or imageio not available")
        return
    
    frames = []
    temp_dir = out_path.parent / f"_temp_{out_path.stem}"
    temp_dir.mkdir(exist_ok=True)
    
    x_mm = op.x * 1e3
    y_mm = op.y * 1e3
    X, Y = np.meshgrid(x_mm, y_mm)
    
    T = len(result.applied_controls)
    
    for t in range(T):
        fig, ax = plt.subplots(figsize=(8, 7))
        
        U = result.U_fields[t]
        U_mean, U_std = np.mean(U), np.std(U)
        
        if U_std > 1e-25:
            Uvis = (U - U_mean) / (U_std + 1e-30)
            levels = np.linspace(-3, 3, 25)
            contourf = ax.contourf(X, Y, Uvis, levels=levels, cmap="RdBu_r", alpha=0.85)
            ax.contour(X, Y, Uvis, levels=levels[::2], colors="k", linewidths=0.3, alpha=0.15)
            fig.colorbar(contourf, ax=ax, label="U (σ units)")
        
        # Circle target
        circle = Circle((cfg.cx * 1e3, cfg.cy * 1e3), cfg.R * 1e3,
                         fill=False, edgecolor='lime', linewidth=2, linestyle='--', label='Target')
        ax.add_patch(circle)
        
        # Trail
        trail = result.positions[:t+1]
        if len(trail) >= 2:
            tx = [p[0] * 1e3 for p in trail]
            ty = [p[1] * 1e3 for p in trail]
            for i in range(len(tx) - 1):
                alpha_val = 0.3 + 0.7 * i / max(len(tx) - 1, 1)
                color = (1-alpha_val) * np.array([0.5, 0.5, 0.5]) + alpha_val * np.array([0, 1, 1])
                ax.plot(tx[i:i+2], ty[i:i+2], linewidth=2.5, color=color, alpha=0.95)
        
        # Current position
        x_t, y_t = result.positions[t]
        ax.scatter(x_t * 1e3, y_t * 1e3, s=300, marker='o', color='red',
                   edgecolors='white', linewidth=3, zorder=100)
        
        # Arrows
        metrics = compute_circle_metrics(x_t, y_t, cfg)
        t_hat_x, t_hat_y = metrics['t_hat']
        arrow_scale = 0.15
        ax.arrow(x_t * 1e3, y_t * 1e3, t_hat_x * arrow_scale, t_hat_y * arrow_scale,
                 head_width=0.05, head_length=0.02, fc='yellow', ec='black', zorder=101)
        
        Fx, Fy = result.forces[t]
        F_mag = np.sqrt(Fx**2 + Fy**2) + 1e-30
        ax.arrow(x_t * 1e3, y_t * 1e3, Fx/F_mag * arrow_scale * 0.8, Fy/F_mag * arrow_scale * 0.8,
                 head_width=0.04, head_length=0.015, fc='magenta', ec='black', zorder=100)
        
        ax.set_xlim(x_mm[0], x_mm[-1])
        ax.set_ylim(y_mm[0], y_mm[-1])
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)
        
        v, phi = result.applied_controls[t]
        radial_err = metrics['radial_err'] * 1e6
        ax.set_title(f"{title_prefix}Step {t}/{T-1}  |  v={v:.3f}, φ={phi:.2f}\nRadial err: {radial_err:.1f} µm")
        
        frame_path = temp_dir / f"frame_{t:04d}.png"
        fig.tight_layout()
        fig.savefig(frame_path, dpi=100)
        plt.close(fig)
        
        frames.append(imageio.imread(frame_path))
    
    imageio.mimsave(out_path, frames, duration=0.3)
    
    for f in temp_dir.glob("*.png"):
        f.unlink()
    temp_dir.rmdir()
    
    print(f"   Saved: {out_path}")


def create_mpc_analysis_plot(out_path: Path, result: MPCResult, cfg: MPCConfig):
    """Create analysis plots for MPC run."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return
    
    T = len(result.applied_controls)
    t_arr = np.arange(T)
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # 1. Radial error over time
    ax1 = axes[0, 0]
    radial_errs = [m['radial_err'] * 1e6 for m in result.per_step_metrics]
    ax1.plot(t_arr, radial_errs, 'b-o', linewidth=2, markersize=3)
    ax1.axhline(0, color='green', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Time step')
    ax1.set_ylabel('Radial error [µm]')
    ax1.set_title('Circle Tracking Error')
    ax1.grid(True, alpha=0.3)
    
    # 2. Angular progress
    ax2 = axes[0, 1]
    thetas = [m['theta'] for m in result.per_step_metrics]
    theta_unwrap = np.unwrap(thetas)
    ax2.plot(t_arr, np.degrees(theta_unwrap), 'g-o', linewidth=2, markersize=3)
    ax2.set_xlabel('Time step')
    ax2.set_ylabel('Angle θ [°]')
    ax2.set_title('Angular Progress')
    ax2.grid(True, alpha=0.3)
    
    # 3. Gor'kov potential
    ax3 = axes[0, 2]
    ax3.plot(t_arr, result.U_values, 'r-o', linewidth=2, markersize=3)
    ax3.set_xlabel('Time step')
    ax3.set_ylabel('U [J]')
    ax3.set_title("Gor'kov Potential")
    ax3.ticklabel_format(axis='y', style='scientific', scilimits=(-18, -18))
    ax3.grid(True, alpha=0.3)
    
    # 4. Controls over time
    ax4 = axes[1, 0]
    v_arr = [c[0] for c in result.applied_controls]
    phi_arr = [c[1] for c in result.applied_controls]
    ax4.plot(t_arr, v_arr, 'b-', linewidth=2, label='v (amplitude)')
    ax4.set_xlabel('Time step')
    ax4.set_ylabel('v', color='blue')
    ax4.tick_params(axis='y', labelcolor='blue')
    ax4_twin = ax4.twinx()
    ax4_twin.plot(t_arr, phi_arr, 'r-', linewidth=2, label='φ (phase)')
    ax4_twin.set_ylabel('φ [rad]', color='red')
    ax4_twin.tick_params(axis='y', labelcolor='red')
    ax4.set_title('Control Inputs')
    ax4.grid(True, alpha=0.3)
    
    # 5. Tangent progress
    ax5 = axes[1, 1]
    prog = [m['tangent_progress'] for m in result.per_step_metrics]
    cum_prog = np.cumsum(prog)
    ax5.plot(t_arr, cum_prog, 'm-o', linewidth=2, markersize=3)
    ax5.set_xlabel('Time step')
    ax5.set_ylabel('Cumulative F·t_hat [N]')
    ax5.set_title('Cumulative Tangent Progress')
    ax5.ticklabel_format(axis='y', style='scientific', scilimits=(-12, -12))
    ax5.grid(True, alpha=0.3)
    
    # 6. Trajectory in space
    ax6 = axes[1, 2]
    theta_circle = np.linspace(0, 2*np.pi, 100)
    circle_x = cfg.cx * 1e3 + cfg.R * 1e3 * np.cos(theta_circle)
    circle_y = cfg.cy * 1e3 + cfg.R * 1e3 * np.sin(theta_circle)
    ax6.plot(circle_x, circle_y, 'g--', linewidth=2, label='Target')
    
    x_mm = [p[0] * 1e3 for p in result.positions]
    y_mm = [p[1] * 1e3 for p in result.positions]
    ax6.plot(x_mm, y_mm, 'b-', linewidth=2, label='MPC trajectory')
    ax6.scatter(x_mm[0], y_mm[0], s=150, marker='s', color='blue', zorder=10, label='Start')
    ax6.scatter(x_mm[-1], y_mm[-1], s=150, marker='*', color='red', zorder=10, label='End')
    ax6.set_xlabel('x [mm]')
    ax6.set_ylabel('y [mm]')
    ax6.set_title('Particle Trajectory')
    ax6.legend(fontsize=9)
    ax6.set_aspect('equal')
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle(f"MPC Circle Tracking: K={cfg.K}, T={cfg.T}, w_du={cfg.w_du:.1e}", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   Saved: {out_path}")


# =============================================================================
# Output
# =============================================================================

def save_mpc_results(result: MPCResult, cfg: MPCConfig, output_dir: Path):
    """Save all MPC results."""
    
    # 1. Applied controls
    with open(output_dir / "controls_mpc.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['t', 'v', 'phi'])
        for t, (v, phi) in enumerate(result.applied_controls):
            writer.writerow([t, v, phi])
    
    # 2. Trajectory
    with open(output_dir / "trajectory_mpc.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['t', 'x_mm', 'y_mm', 'U', 'Fx', 'Fy', 'radial_err_um', 'theta_rad', 'tangent_progress'])
        for t in range(len(result.applied_controls)):
            x, y = result.positions[t]
            m = result.per_step_metrics[t]
            Fx, Fy = result.forces[t]
            writer.writerow([
                t, x*1e3, y*1e3, result.U_values[t], Fx, Fy,
                m['radial_err']*1e6, m['theta'], m['tangent_progress']
            ])
        # Final position
        x_f, y_f = result.positions[-1]
        m_f = compute_circle_metrics(x_f, y_f, cfg)
        writer.writerow([len(result.applied_controls), x_f*1e3, y_f*1e3, '', '', '', m_f['radial_err']*1e6, m_f['theta'], ''])
    
    # 3. Step metrics
    with open(output_dir / "step_metrics_mpc.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['t', 'L_total', 'L_radial', 'L_prog', 'L_U', 'L_reg', 'L_du'])
        for t, m in enumerate(result.per_step_metrics):
            writer.writerow([t, m['L_total'], m['L_radial'], m['L_prog'], m['L_U'], m['L_reg'], m['L_du']])
    
    # 4. Inner optimization history (compressed)
    inner_summary = []
    for t, J_hist in enumerate(result.inner_J_history):
        inner_summary.append({
            't': t,
            'n_iters': len(J_hist),
            'J_init': J_hist[0] if J_hist else None,
            'J_final': J_hist[-1] if J_hist else None,
        })
    
    with open(output_dir / "inner_opt_summary.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['t', 'n_iters', 'J_init', 'J_final'])
        for s in inner_summary:
            writer.writerow([s['t'], s['n_iters'], s['J_init'], s['J_final']])
    
    # 5. Summary JSON
    x_final, y_final = result.positions[-1]
    metrics_final = compute_circle_metrics(x_final, y_final, cfg)
    
    thetas = [m['theta'] for m in result.per_step_metrics]
    theta_unwrap = np.unwrap(thetas)
    total_angle = theta_unwrap[-1] - theta_unwrap[0] if len(theta_unwrap) > 1 else 0.0
    
    summary = {
        'config': {
            'Nx': cfg.Nx, 'Ny': cfg.Ny,
            'K': cfg.K, 'T': cfg.T,
            'replan_every': cfg.replan_every,
            'n_iters': cfg.n_iters,
            'dt': cfg.dt,
            'cx_mm': cfg.cx * 1e3, 'cy_mm': cfg.cy * 1e3, 'R_mm': cfg.R * 1e3,
            'w_r': cfg.w_r, 'w_prog': cfg.w_prog, 'w_U': cfg.w_U, 'w_reg': cfg.w_reg, 'w_du': cfg.w_du,
            'progress_mode': cfg.progress_mode,
        },
        'results': {
            'final_position_mm': [x_final * 1e3, y_final * 1e3],
            'final_radial_err_um': metrics_final['radial_err'] * 1e6,
            'final_theta_rad': metrics_final['theta'],
            'total_angle_rad': float(total_angle),
            'total_angle_deg': float(np.degrees(total_angle)),
            'mean_radial_err_um': float(np.mean([m['radial_err']*1e6 for m in result.per_step_metrics])),
            'max_radial_err_um': float(np.max([abs(m['radial_err'])*1e6 for m in result.per_step_metrics])),
            'total_L': float(sum(m['L_total'] for m in result.per_step_metrics)),
        },
    }
    
    with open(output_dir / "mpc_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"   Saved CSV/JSON results to {output_dir}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Rolling-horizon MPC for circle tracking")
    
    parser.add_argument('--fast', action='store_true', help="Fast mode (smaller grid/horizon)")
    
    # Grid
    parser.add_argument('--Nx', type=int, default=64, help="Grid points X")
    parser.add_argument('--Ny', type=int, default=64, help="Grid points Y")
    
    # MPC parameters
    parser.add_argument('--K', type=int, default=5, help="Horizon length (default: 5)")
    parser.add_argument('--T', type=int, default=50, help="Total executed steps (default: 50)")
    parser.add_argument('--replan_every', type=int, default=1, help="Replan frequency (default: 1)")
    parser.add_argument('--n_iters', type=int, default=8, help="Optimization iterations per MPC solve")
    parser.add_argument('--dt', type=float, default=0.05, help="Time step [s]")
    
    # Objective weights
    parser.add_argument('--w_r', type=float, default=1.0e12, help="Radial error weight")
    parser.add_argument('--w_prog', type=float, default=1.0e-6, help="Tangent progress weight")
    parser.add_argument('--w_U', type=float, default=1.0, help="Gor'kov potential weight")
    parser.add_argument('--w_reg', type=float, default=0.0, help="Regularization weight")
    parser.add_argument('--w_du', type=float, default=0.0, help="Control smoothness weight")
    
    # Progress mode
    parser.add_argument('--progress_mode', type=str, default='force',
                        choices=['force', 'displacement'], help="Progress proxy mode")
    
    # Circle geometry
    parser.add_argument('--cx', type=float, default=1.0e-3, help="Circle center x [m]")
    parser.add_argument('--cy', type=float, default=1.0e-3, help="Circle center y [m]")
    parser.add_argument('--R', type=float, default=0.4e-3, help="Circle radius [m]")
    
    args = parser.parse_args()
    
    # Build config
    cfg = MPCConfig()
    cfg.Nx = args.Nx
    cfg.Ny = args.Ny
    cfg.K = args.K
    cfg.T = args.T
    cfg.replan_every = args.replan_every
    cfg.n_iters = args.n_iters
    cfg.dt = args.dt
    cfg.w_r = args.w_r
    cfg.w_prog = args.w_prog
    cfg.w_U = args.w_U
    cfg.w_reg = args.w_reg
    cfg.w_du = args.w_du
    cfg.progress_mode = args.progress_mode
    cfg.cx = args.cx
    cfg.cy = args.cy
    cfg.R = args.R
    
    if args.fast:
        cfg.Nx = 32
        cfg.Ny = 32
        cfg.K = 3
        cfg.T = 20
        cfg.n_iters = 5
        cfg.alphas = (0.0, 0.1, 0.3, 1.0)
    
    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = project_root / "results" / "adjoint_circle_track_mpc" / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("ROLLING-HORIZON MPC FOR CIRCLE TRACKING")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"   Grid: {cfg.Nx}×{cfg.Ny}")
    print(f"   Horizon K: {cfg.K} steps")
    print(f"   Total steps T: {cfg.T}")
    print(f"   Replan every: {cfg.replan_every} step(s)")
    print(f"   Opt iterations: {cfg.n_iters}")
    print(f"   Time step dt: {cfg.dt*1e3:.1f} ms")
    print(f"   Circle: center=({cfg.cx*1e3:.2f}, {cfg.cy*1e3:.2f}) mm, R={cfg.R*1e3:.2f} mm")
    print(f"   Progress mode: {cfg.progress_mode}")
    print(f"   Weights: w_r={cfg.w_r:.2e}, w_prog={cfg.w_prog:.2e}, w_U={cfg.w_U:.2e}")
    print(f"            w_reg={cfg.w_reg:.2e}, w_du={cfg.w_du:.2e}")
    
    # Build operator
    print(f"\n1. Building Helmholtz operator...")
    op = build_helmholtz_2d_forced_25d_operator(
        Lx=cfg.Lx, Ly=cfg.Ly, Nx=cfg.Nx, Ny=cfg.Ny,
        f=cfg.f, c0=cfg.c0, rho0=cfg.rho0,
    )
    
    particle = ParticleProps(a=cfg.particle_a, rho_p=cfg.particle_rho_p, c_p=cfg.particle_c_p)
    
    # Initial state
    x0 = cfg.cx + cfg.R * np.cos(cfg.theta0)
    y0 = cfg.cy + cfg.R * np.sin(cfg.theta0)
    v_init = 0.05
    phi_init = 0.0
    
    print(f"\n2. Initial state:")
    print(f"   Particle: ({x0*1e3:.3f}, {y0*1e3:.3f}) mm (on circle at θ=0°)")
    print(f"   Control: v={v_init}, φ={phi_init}")
    
    # Run MPC
    print("\n3. Running MPC...")
    result = run_mpc(op, x0, y0, v_init, phi_init, cfg, particle, verbose=True)
    
    # Results summary
    print("\n4. Results Summary:")
    x_final, y_final = result.positions[-1]
    metrics_final = compute_circle_metrics(x_final, y_final, cfg)
    
    radial_errs = [m['radial_err'] * 1e6 for m in result.per_step_metrics]
    thetas = [m['theta'] for m in result.per_step_metrics]
    theta_unwrap = np.unwrap(thetas)
    total_angle = theta_unwrap[-1] - theta_unwrap[0] if len(theta_unwrap) > 1 else 0.0
    
    print(f"   Final position: ({x_final*1e3:.3f}, {y_final*1e3:.3f}) mm")
    print(f"   Final radial error: {metrics_final['radial_err']*1e6:.2f} µm")
    print(f"   Mean radial error: {np.mean(radial_errs):.2f} µm")
    print(f"   Max radial error: {np.max(np.abs(radial_errs)):.2f} µm")
    print(f"   Total angular progress: {np.degrees(total_angle):.1f}°")
    
    # Save results
    print("\n5. Saving results...")
    save_mpc_results(result, cfg, output_dir)
    
    # Create visualizations
    print("\n6. Creating visualizations...")
    create_mpc_gif(output_dir / "mpc_trajectory.gif", result, op, cfg)
    create_mpc_analysis_plot(output_dir / "mpc_analysis.png", result, cfg)
    
    print(f"\n   All results saved to: {output_dir}")
    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
