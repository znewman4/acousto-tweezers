# src/acousto/adjoint/losses/circle_track.py
"""
Circle tracking loss functions for adjoint-based trajectory optimization.

Supports two progress modes:
- "force": Uses F · t_hat as progress proxy (simpler adjoint, L_t depends only on x_t, u_t)
- "displacement": Uses (x_{t+1} - x_t)/dt · t_hat (true motion, L_t depends on x_t AND x_{t+1})

Full per-step objective:
    L_t = w_r * e_r^2 
          - w_prog * progress_term
          + w_U * U(x_t; u_t)
          + w_reg * ||u_t - u_ref||^2
          + w_du * ||u_t - u_{t-1}||^2   (control smoothness)
          + w_bounds * bounds_penalty(u_t)
          + w_domain * domain_penalty(x_t)

For displacement mode, the adjoint recursion must include ∂L_t/∂x_{t+1}.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Dict, List, Optional, Literal


@dataclass
class CircleLossConfig:
    """Configuration for circle tracking loss."""
    # Circle geometry
    cx: float = 1.0e-3      # circle center x [m]
    cy: float = 1.0e-3      # circle center y [m]
    R: float = 0.4e-3       # circle radius [m]
    ccw: bool = True        # counter-clockwise motion
    
    # Primary objective weights
    w_r: float = 1.0e12     # radial error weight
    w_prog: float = 1.0e-6  # tangent progress weight
    w_U: float = 1.0        # Gor'kov potential weight
    w_reg: float = 0.0      # control magnitude regularization
    
    # Control smoothness (Option C)
    w_du: float = 0.0       # control rate-of-change penalty
    
    # Soft bounds penalty (Option C)
    w_bounds: float = 0.0   # penalty for controls near/exceeding bounds
    v_min: float = 0.01
    v_max: float = 0.2
    phi_min: float = -np.pi
    phi_max: float = np.pi
    bounds_margin: float = 0.01  # margin before bounds where penalty kicks in
    
    # Domain boundary penalty (Option C)
    w_domain: float = 0.0   # penalty for particle near domain boundary
    domain_margin: float = 0.1e-3  # margin from domain edge [m]
    Lx: float = 2.0e-3
    Ly: float = 2.0e-3
    
    # Reference control for regularization
    v_ref: float = 0.05
    phi_ref: float = 0.0
    
    # Terminal cost
    beta_terminal: float = 1.0
    w_rT: float = 1.0       # terminal radial weight multiplier
    w_UT: float = 0.5       # terminal U weight multiplier
    
    # Progress mode
    progress_mode: Literal["force", "displacement"] = "force"
    
    # Time step (needed for displacement mode)
    dt: float = 0.05


def circle_metrics(
    x: float, y: float,
    cx: float, cy: float,
    R: float,
    ccw: bool = True,
) -> Dict[str, float]:
    """
    Compute circle-related metrics for a position.
    
    Parameters
    ----------
    x, y : float
        Particle position [m]
    cx, cy : float
        Circle center [m]
    R : float
        Circle radius [m]
    ccw : bool
        Counter-clockwise motion direction
        
    Returns
    -------
    dict with:
        r: distance from circle center
        radial_err: r - R (positive = outside circle)
        r_hat: (rx, ry) radial unit vector pointing outward
        t_hat: (tx, ty) tangent unit vector in motion direction
        theta: angle from center (radians)
    """
    dx = x - cx
    dy = y - cy
    r = np.sqrt(dx**2 + dy**2)
    
    if r < 1e-12:
        r_hat_x, r_hat_y = 1.0, 0.0
    else:
        r_hat_x, r_hat_y = dx / r, dy / r
    
    # Tangent: perpendicular to radial, in motion direction
    if ccw:
        t_hat_x, t_hat_y = -r_hat_y, r_hat_x
    else:
        t_hat_x, t_hat_y = r_hat_y, -r_hat_x
    
    radial_err = r - R
    theta = np.arctan2(dy, dx)
    
    return {
        'r': r,
        'radial_err': radial_err,
        'r_hat': (r_hat_x, r_hat_y),
        't_hat': (t_hat_x, t_hat_y),
        'theta': theta,
    }


def _soft_bounds_penalty(val: float, lo: float, hi: float, margin: float) -> float:
    """
    Soft quadratic penalty for value approaching bounds.
    Returns 0 if val is well inside [lo+margin, hi-margin].
    """
    penalty = 0.0
    if val < lo + margin:
        penalty += (lo + margin - val)**2
    if val > hi - margin:
        penalty += (val - (hi - margin))**2
    return penalty


def _domain_penalty(x: float, y: float, Lx: float, Ly: float, margin: float) -> float:
    """
    Penalty for particle being near domain boundaries.
    """
    penalty = 0.0
    if x < margin:
        penalty += (margin - x)**2
    if x > Lx - margin:
        penalty += (x - (Lx - margin))**2
    if y < margin:
        penalty += (margin - y)**2
    if y > Ly - margin:
        penalty += (y - (Ly - margin))**2
    return penalty


def compute_step_loss(
    x_t: Tuple[float, float],
    x_tp1: Optional[Tuple[float, float]],
    u_t: Tuple[float, float],
    u_tm1: Optional[Tuple[float, float]],
    F_t: Tuple[float, float],
    U_t: float,
    cfg: CircleLossConfig,
) -> Dict[str, float]:
    """
    Compute per-step loss L_t and its components.
    
    Parameters
    ----------
    x_t : (x, y)
        Position at time t
    x_tp1 : (x, y) or None
        Position at time t+1 (required for displacement mode)
    u_t : (v, phi)
        Control at time t
    u_tm1 : (v, phi) or None
        Control at time t-1 (for smoothness penalty)
    F_t : (Fx, Fy)
        Force at time t
    U_t : float
        Gor'kov potential at time t
    cfg : CircleLossConfig
        Loss configuration
        
    Returns
    -------
    dict with loss components:
        L_total, L_radial, L_prog, L_U, L_reg, L_du, L_bounds, L_domain
        Also: radial_err, tangent_progress, r, theta
    """
    x, y = x_t
    v, phi = u_t
    Fx, Fy = F_t
    
    metrics = circle_metrics(x, y, cfg.cx, cfg.cy, cfg.R, cfg.ccw)
    e_r = metrics['radial_err']
    t_hat_x, t_hat_y = metrics['t_hat']
    
    # === Radial error ===
    L_radial = cfg.w_r * e_r**2
    
    # === Progress term ===
    if cfg.progress_mode == "force":
        # Force-based proxy: F · t_hat
        tangent_progress = Fx * t_hat_x + Fy * t_hat_y
        L_prog = -cfg.w_prog * tangent_progress
    else:
        # Displacement-based: (x_{t+1} - x_t)/dt · t_hat
        if x_tp1 is None:
            raise ValueError("x_tp1 required for displacement progress mode")
        x_tp1_val, y_tp1_val = x_tp1
        vel_x = (x_tp1_val - x) / cfg.dt
        vel_y = (y_tp1_val - y) / cfg.dt
        tangent_progress = vel_x * t_hat_x + vel_y * t_hat_y
        L_prog = -cfg.w_prog * tangent_progress
    
    # === Gor'kov potential ===
    L_U = cfg.w_U * U_t
    
    # === Control magnitude regularization ===
    L_reg = cfg.w_reg * ((v - cfg.v_ref)**2 + (phi - cfg.phi_ref)**2)
    
    # === Control smoothness (rate of change) ===
    L_du = 0.0
    if cfg.w_du > 0 and u_tm1 is not None:
        v_tm1, phi_tm1 = u_tm1
        L_du = cfg.w_du * ((v - v_tm1)**2 + (phi - phi_tm1)**2)
    
    # === Soft bounds penalty ===
    L_bounds = 0.0
    if cfg.w_bounds > 0:
        L_bounds += cfg.w_bounds * _soft_bounds_penalty(v, cfg.v_min, cfg.v_max, cfg.bounds_margin)
        L_bounds += cfg.w_bounds * _soft_bounds_penalty(phi, cfg.phi_min, cfg.phi_max, cfg.bounds_margin)
    
    # === Domain boundary penalty ===
    L_domain = 0.0
    if cfg.w_domain > 0:
        L_domain = cfg.w_domain * _domain_penalty(x, y, cfg.Lx, cfg.Ly, cfg.domain_margin)
    
    L_total = L_radial + L_prog + L_U + L_reg + L_du + L_bounds + L_domain
    
    return {
        'L_total': L_total,
        'L_radial': L_radial,
        'L_prog': L_prog,
        'L_U': L_U,
        'L_reg': L_reg,
        'L_du': L_du,
        'L_bounds': L_bounds,
        'L_domain': L_domain,
        'radial_err': e_r,
        'tangent_progress': tangent_progress,
        'r': metrics['r'],
        'theta': metrics['theta'],
    }


def compute_trajectory_loss(
    positions: List[Tuple[float, float]],
    controls: List[Tuple[float, float]],
    forces: List[Tuple[float, float]],
    U_values: List[float],
    cfg: CircleLossConfig,
    terminal_U: Optional[float] = None,
    u_init: Optional[Tuple[float, float]] = None,
) -> Dict[str, float]:
    """
    Compute full trajectory loss J and its components.
    
    J = Σ_{t=0}^{K-1} L_t + β_T * L_terminal
    
    Parameters
    ----------
    positions : list of (x, y)
        Positions for t=0..K (K+1 entries)
    controls : list of (v, phi)
        Controls for t=0..K-1 (K entries)
    forces : list of (Fx, Fy)
        Forces for t=0..K-1 (K entries)
    U_values : list of float
        Gor'kov potential for t=0..K-1 (K entries)
    cfg : CircleLossConfig
        Loss configuration
    terminal_U : float, optional
        U at terminal state (for terminal cost)
    u_init : (v, phi), optional
        Control before t=0 (for smoothness at t=0)
        
    Returns
    -------
    dict with:
        J_total, J_radial, J_prog, J_U, J_reg, J_du, J_bounds, J_domain, J_terminal
    """
    K = len(controls)
    
    J_total = 0.0
    J_radial = 0.0
    J_prog = 0.0
    J_U = 0.0
    J_reg = 0.0
    J_du = 0.0
    J_bounds = 0.0
    J_domain = 0.0
    
    for t in range(K):
        x_t = positions[t]
        x_tp1 = positions[t + 1] if cfg.progress_mode == "displacement" else None
        u_t = controls[t]
        u_tm1 = controls[t - 1] if t > 0 else u_init
        F_t = forces[t]
        U_t = U_values[t]
        
        step_loss = compute_step_loss(x_t, x_tp1, u_t, u_tm1, F_t, U_t, cfg)
        
        J_total += step_loss['L_total']
        J_radial += step_loss['L_radial']
        J_prog += step_loss['L_prog']
        J_U += step_loss['L_U']
        J_reg += step_loss['L_reg']
        J_du += step_loss['L_du']
        J_bounds += step_loss['L_bounds']
        J_domain += step_loss['L_domain']
    
    # Terminal cost
    J_terminal = 0.0
    if cfg.beta_terminal > 0:
        x_K, y_K = positions[-1]
        metrics_K = circle_metrics(x_K, y_K, cfg.cx, cfg.cy, cfg.R, cfg.ccw)
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
        'J_du': J_du,
        'J_bounds': J_bounds,
        'J_domain': J_domain,
        'J_terminal': J_terminal,
    }


# =============================================================================
# Gradient computations for adjoint recursion
# =============================================================================

def compute_dL_dx_t(
    x_t: Tuple[float, float],
    x_tp1: Optional[Tuple[float, float]],
    u_t: Tuple[float, float],
    F_t: Tuple[float, float],
    cfg: CircleLossConfig,
    dU_dx: np.ndarray,
    dF_dx: np.ndarray,
    eps: float = 1e-7,
) -> np.ndarray:
    """
    Compute ∂L_t/∂x_t for adjoint recursion.
    
    For force mode:
        ∂L/∂x_t = w_r * 2*e_r * r_hat + w_U * ∂U/∂x + ∂L_prog/∂x_t
        
    For displacement mode:
        ∂L/∂x_t includes additional term from (x_{t+1} - x_t)/dt · t_hat
    
    Parameters
    ----------
    x_t : (x, y)
        Position at time t
    x_tp1 : (x, y) or None
        Position at time t+1 (for displacement mode)
    u_t : (v, phi)
        Control at time t
    F_t : (Fx, Fy)
        Force at time t
    cfg : CircleLossConfig
        Loss configuration
    dU_dx : np.ndarray, shape (2,)
        Spatial gradient of U at x_t
    dF_dx : np.ndarray, shape (2, 2)
        Jacobian of F w.r.t. x at x_t: [[∂Fx/∂x, ∂Fx/∂y], [∂Fy/∂x, ∂Fy/∂y]]
    eps : float
        FD step for geometric derivatives
        
    Returns
    -------
    dL_dx : np.ndarray, shape (2,)
        [∂L/∂x, ∂L/∂y]
    """
    x, y = x_t
    Fx, Fy = F_t
    
    metrics = circle_metrics(x, y, cfg.cx, cfg.cy, cfg.R, cfg.ccw)
    e_r = metrics['radial_err']
    r_hat_x, r_hat_y = metrics['r_hat']
    t_hat_x, t_hat_y = metrics['t_hat']
    
    # === Radial error gradient: ∂(w_r * e_r^2)/∂x = 2 * w_r * e_r * r_hat ===
    dL_radial_dx = cfg.w_r * 2 * e_r * np.array([r_hat_x, r_hat_y])
    
    # === U gradient: w_U * ∂U/∂x ===
    dL_U_dx = cfg.w_U * dU_dx
    
    # === Progress gradient ===
    if cfg.progress_mode == "force":
        # L_prog = -w_prog * (F · t_hat)
        # ∂L_prog/∂x = -w_prog * (∂F/∂x · t_hat + F · ∂t_hat/∂x)
        
        # ∂(F · t_hat)/∂x via chain rule
        # = (∂Fx/∂x * t_hat_x + ∂Fy/∂x * t_hat_y, ∂Fx/∂y * t_hat_x + ∂Fy/∂y * t_hat_y)
        # + Fx * ∂t_hat_x/∂x + Fy * ∂t_hat_y/∂x, etc.
        
        # dF_dx contribution
        dFdot_dx_from_F = np.array([
            dF_dx[0, 0] * t_hat_x + dF_dx[1, 0] * t_hat_y,  # ∂(F·t)/∂x
            dF_dx[0, 1] * t_hat_x + dF_dx[1, 1] * t_hat_y,  # ∂(F·t)/∂y
        ])
        
        # t_hat derivative via FD (geometric only)
        def get_t_hat(xp, yp):
            m = circle_metrics(xp, yp, cfg.cx, cfg.cy, cfg.R, cfg.ccw)
            return np.array(m['t_hat'])
        
        t_hat_xp = get_t_hat(x + eps, y)
        t_hat_xm = get_t_hat(x - eps, y)
        t_hat_yp = get_t_hat(x, y + eps)
        t_hat_ym = get_t_hat(x, y - eps)
        
        dt_hat_dx = (t_hat_xp - t_hat_xm) / (2 * eps)  # shape (2,)
        dt_hat_dy = (t_hat_yp - t_hat_ym) / (2 * eps)  # shape (2,)
        
        F_vec = np.array([Fx, Fy])
        dFdot_dx_from_t = np.array([
            np.dot(F_vec, dt_hat_dx),
            np.dot(F_vec, dt_hat_dy),
        ])
        
        dL_prog_dx = -cfg.w_prog * (dFdot_dx_from_F + dFdot_dx_from_t)
        
    else:  # displacement mode
        # L_prog = -w_prog * ((x_{t+1} - x_t)/dt · t_hat(x_t))
        # ∂L_prog/∂x_t = -w_prog * ((-1/dt) * t_hat + (x_{t+1} - x_t)/dt · ∂t_hat/∂x_t)
        
        if x_tp1 is None:
            raise ValueError("x_tp1 required for displacement mode gradient")
        
        x_tp1_val, y_tp1_val = x_tp1
        vel = np.array([(x_tp1_val - x) / cfg.dt, (y_tp1_val - y) / cfg.dt])
        t_hat = np.array([t_hat_x, t_hat_y])
        
        # First term: ∂/∂x_t of (x_{t+1} - x_t)/dt = -1/dt * I
        # So contribution is: -w_prog * (-1/dt * t_hat) = w_prog/dt * t_hat
        dL_prog_dx_from_vel = (cfg.w_prog / cfg.dt) * t_hat
        
        # Second term: vel · ∂t_hat/∂x_t (same FD as above)
        def get_t_hat(xp, yp):
            m = circle_metrics(xp, yp, cfg.cx, cfg.cy, cfg.R, cfg.ccw)
            return np.array(m['t_hat'])
        
        t_hat_xp = get_t_hat(x + eps, y)
        t_hat_xm = get_t_hat(x - eps, y)
        t_hat_yp = get_t_hat(x, y + eps)
        t_hat_ym = get_t_hat(x, y - eps)
        
        dt_hat_dx = (t_hat_xp - t_hat_xm) / (2 * eps)
        dt_hat_dy = (t_hat_yp - t_hat_ym) / (2 * eps)
        
        dL_prog_dx_from_t = -cfg.w_prog * np.array([
            np.dot(vel, dt_hat_dx),
            np.dot(vel, dt_hat_dy),
        ])
        
        dL_prog_dx = dL_prog_dx_from_vel + dL_prog_dx_from_t
    
    # === Domain penalty gradient ===
    dL_domain_dx = np.zeros(2)
    if cfg.w_domain > 0:
        margin = cfg.domain_margin
        if x < margin:
            dL_domain_dx[0] += cfg.w_domain * 2 * (x - margin)
        if x > cfg.Lx - margin:
            dL_domain_dx[0] += cfg.w_domain * 2 * (x - (cfg.Lx - margin))
        if y < margin:
            dL_domain_dx[1] += cfg.w_domain * 2 * (y - margin)
        if y > cfg.Ly - margin:
            dL_domain_dx[1] += cfg.w_domain * 2 * (y - (cfg.Ly - margin))
    
    return dL_radial_dx + dL_U_dx + dL_prog_dx + dL_domain_dx


def compute_dL_dx_tp1(
    x_t: Tuple[float, float],
    x_tp1: Tuple[float, float],
    cfg: CircleLossConfig,
) -> np.ndarray:
    """
    Compute ∂L_t/∂x_{t+1} for displacement mode.
    
    Only non-zero in displacement mode where:
        L_prog = -w_prog * ((x_{t+1} - x_t)/dt · t_hat(x_t))
        ∂L_prog/∂x_{t+1} = -w_prog * (1/dt) * t_hat(x_t)
    
    Parameters
    ----------
    x_t : (x, y)
        Position at time t
    x_tp1 : (x, y)
        Position at time t+1
    cfg : CircleLossConfig
        Loss configuration
        
    Returns
    -------
    dL_dx_tp1 : np.ndarray, shape (2,)
        [∂L/∂x_{t+1}, ∂L/∂y_{t+1}]
    """
    if cfg.progress_mode != "displacement":
        return np.zeros(2)
    
    x, y = x_t
    metrics = circle_metrics(x, y, cfg.cx, cfg.cy, cfg.R, cfg.ccw)
    t_hat_x, t_hat_y = metrics['t_hat']
    
    # ∂L_prog/∂x_{t+1} = -w_prog * (1/dt) * t_hat
    return -cfg.w_prog / cfg.dt * np.array([t_hat_x, t_hat_y])


def compute_dL_du(
    u_t: Tuple[float, float],
    u_tm1: Optional[Tuple[float, float]],
    u_tp1: Optional[Tuple[float, float]],
    dU_du: Tuple[float, float],
    dF_du: Tuple[np.ndarray, np.ndarray],
    x_t: Tuple[float, float],
    cfg: CircleLossConfig,
) -> np.ndarray:
    """
    Compute ∂L_t/∂u_t (direct term, not including dynamics propagation).
    
    Parameters
    ----------
    u_t : (v, phi)
        Control at time t
    u_tm1 : (v, phi) or None
        Control at time t-1
    u_tp1 : (v, phi) or None
        Control at time t+1 (for smoothness term from L_{t+1})
    dU_du : (∂U/∂v, ∂U/∂phi)
        Adjoint gradient of U w.r.t. control
    dF_du : (dF_dv, dF_dphi)
        Finite difference gradients of F w.r.t. control, each shape (2,)
    x_t : (x, y)
        Position at time t (for progress term in force mode)
    cfg : CircleLossConfig
        Loss configuration
        
    Returns
    -------
    dL_du : np.ndarray, shape (2,)
        [∂L/∂v, ∂L/∂phi]
    """
    v, phi = u_t
    dU_dv, dU_dphi = dU_du
    dF_dv, dF_dphi = dF_du
    
    # === U term: w_U * ∂U/∂u ===
    dL_U_du = cfg.w_U * np.array([dU_dv, dU_dphi])
    
    # === Regularization: w_reg * 2 * (u - u_ref) ===
    dL_reg_du = cfg.w_reg * 2 * np.array([v - cfg.v_ref, phi - cfg.phi_ref])
    
    # === Smoothness: from L_t (forward) ===
    dL_du_t_from_smooth = np.zeros(2)
    if cfg.w_du > 0 and u_tm1 is not None:
        v_tm1, phi_tm1 = u_tm1
        dL_du_t_from_smooth = cfg.w_du * 2 * np.array([v - v_tm1, phi - phi_tm1])
    
    # === Smoothness: from L_{t+1} (backward) ===
    # L_{t+1} contains w_du * ||u_{t+1} - u_t||^2
    # ∂L_{t+1}/∂u_t = w_du * 2 * (u_t - u_{t+1}) = -w_du * 2 * (u_{t+1} - u_t)
    dL_du_tp1_from_smooth = np.zeros(2)
    if cfg.w_du > 0 and u_tp1 is not None:
        v_tp1, phi_tp1 = u_tp1
        dL_du_tp1_from_smooth = cfg.w_du * 2 * np.array([v - v_tp1, phi - phi_tp1])
    
    # === Progress term (force mode only - displacement mode has no direct u dependence) ===
    dL_prog_du = np.zeros(2)
    if cfg.progress_mode == "force":
        # L_prog = -w_prog * (F · t_hat)
        # ∂L_prog/∂u = -w_prog * (∂F/∂u · t_hat)
        x, y = x_t
        metrics = circle_metrics(x, y, cfg.cx, cfg.cy, cfg.R, cfg.ccw)
        t_hat_x, t_hat_y = metrics['t_hat']
        
        dFdot_dv = dF_dv[0] * t_hat_x + dF_dv[1] * t_hat_y
        dFdot_dphi = dF_dphi[0] * t_hat_x + dF_dphi[1] * t_hat_y
        
        dL_prog_du = -cfg.w_prog * np.array([dFdot_dv, dFdot_dphi])
    
    # === Bounds penalty ===
    dL_bounds_du = np.zeros(2)
    if cfg.w_bounds > 0:
        margin = cfg.bounds_margin
        if v < cfg.v_min + margin:
            dL_bounds_du[0] += cfg.w_bounds * 2 * (v - (cfg.v_min + margin))
        if v > cfg.v_max - margin:
            dL_bounds_du[0] += cfg.w_bounds * 2 * (v - (cfg.v_max - margin))
        if phi < cfg.phi_min + margin:
            dL_bounds_du[1] += cfg.w_bounds * 2 * (phi - (cfg.phi_min + margin))
        if phi > cfg.phi_max - margin:
            dL_bounds_du[1] += cfg.w_bounds * 2 * (phi - (cfg.phi_max - margin))
    
    return (dL_U_du + dL_reg_du + dL_du_t_from_smooth + dL_du_tp1_from_smooth + 
            dL_prog_du + dL_bounds_du)
