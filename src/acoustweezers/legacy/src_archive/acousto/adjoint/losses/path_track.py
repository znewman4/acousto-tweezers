# src/acousto/adjoint/losses/path_track.py
"""
General path tracking loss functions for adjoint-based trajectory optimization.

Supports multiple path types:
- Circle: center (cx, cy), radius R
- Polyline: list of waypoints
- (Future) Spline: cubic spline through waypoints

Progress control modes:
- "displacement": prog_t = (x_{t+1} - x_t)/dt · t_hat(x_t)  [true velocity]
- "force": prog_t = F(x_t, u_t) · t_hat(x_t)  [force proxy]

Per-step objective:
    L_t = w_perp * e_perp^2 
          + w_U * U(x_t; u_t)
          + w_reg * ||u_t - u_ref||^2
          + w_du * ||u_t - u_{t-1}||^2
          - w_prog * prog_t
          + w_sdot * (prog_t - v_ref)^2   [optional speed tracking]
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Dict, List, Optional, Literal, Union
from abc import ABC, abstractmethod


# =============================================================================
# Path Definitions
# =============================================================================

class Path(ABC):
    """Abstract base class for paths."""
    
    @abstractmethod
    def closest_point(self, x: float, y: float) -> Tuple[float, float]:
        """Find closest point on path to (x, y)."""
        pass
    
    @abstractmethod
    def tangent_at(self, s: float) -> Tuple[float, float]:
        """Get unit tangent vector at arclength parameter s."""
        pass
    
    @abstractmethod
    def normal_at(self, s: float) -> Tuple[float, float]:
        """Get unit normal vector at arclength parameter s (pointing left of tangent)."""
        pass
    
    @abstractmethod
    def arclength_at(self, x: float, y: float) -> float:
        """Get arclength parameter s for closest point to (x, y)."""
        pass
    
    @abstractmethod
    def total_length(self) -> float:
        """Total path length."""
        pass
    
    @abstractmethod
    def is_closed(self) -> bool:
        """Whether the path is closed (loops back)."""
        pass
    
    def metrics(self, x: float, y: float) -> Dict[str, any]:
        """
        Compute all path metrics for a point.
        
        Returns dict with:
            p_closest: (px, py) closest point on path
            t_hat: (tx, ty) unit tangent at closest point
            n_hat: (nx, ny) unit normal (perpendicular to tangent, pointing left)
            e_perp: signed lateral error (positive = left of path)
            s: arclength parameter at closest point
            s_unwrapped: unwrapped arclength (for progress tracking on closed paths)
        """
        px, py = self.closest_point(x, y)
        s = self.arclength_at(x, y)
        tx, ty = self.tangent_at(s)
        nx, ny = self.normal_at(s)
        
        # Signed lateral error: (x - p_closest) · n_hat
        dx, dy = x - px, y - py
        e_perp = dx * nx + dy * ny
        
        return {
            'p_closest': (px, py),
            't_hat': (tx, ty),
            'n_hat': (nx, ny),
            'e_perp': e_perp,
            's': s,
            's_unwrapped': s,  # subclasses can override for unwrapping
        }


@dataclass
class CirclePath(Path):
    """Circular path."""
    cx: float = 1.0e-3
    cy: float = 1.0e-3
    R: float = 0.4e-3
    ccw: bool = True  # counter-clockwise direction
    
    # For unwrapped progress tracking
    _last_theta: float = field(default=0.0, repr=False)
    _theta_offset: float = field(default=0.0, repr=False)
    
    def closest_point(self, x: float, y: float) -> Tuple[float, float]:
        dx, dy = x - self.cx, y - self.cy
        r = np.sqrt(dx**2 + dy**2)
        if r < 1e-12:
            return (self.cx + self.R, self.cy)
        return (self.cx + self.R * dx / r, self.cy + self.R * dy / r)
    
    def tangent_at(self, s: float) -> Tuple[float, float]:
        # s is angle in radians for circle
        if self.ccw:
            return (-np.sin(s), np.cos(s))
        else:
            return (np.sin(s), -np.cos(s))
    
    def normal_at(self, s: float) -> Tuple[float, float]:
        # Normal points outward (radial direction)
        # For CCW motion, normal is to the right of tangent (outward)
        return (np.cos(s), np.sin(s))
    
    def arclength_at(self, x: float, y: float) -> float:
        dx, dy = x - self.cx, y - self.cy
        return np.arctan2(dy, dx)
    
    def total_length(self) -> float:
        return 2 * np.pi * self.R
    
    def is_closed(self) -> bool:
        return True
    
    def metrics(self, x: float, y: float) -> Dict[str, any]:
        dx, dy = x - self.cx, y - self.cy
        r = np.sqrt(dx**2 + dy**2)
        theta = np.arctan2(dy, dx)
        
        if r < 1e-12:
            r_hat_x, r_hat_y = 1.0, 0.0
        else:
            r_hat_x, r_hat_y = dx / r, dy / r
        
        # Tangent perpendicular to radial
        if self.ccw:
            t_hat_x, t_hat_y = -r_hat_y, r_hat_x
        else:
            t_hat_x, t_hat_y = r_hat_y, -r_hat_x
        
        # Normal points outward (radial)
        n_hat_x, n_hat_y = r_hat_x, r_hat_y
        
        # Closest point on circle
        px = self.cx + self.R * r_hat_x
        py = self.cy + self.R * r_hat_y
        
        # Signed lateral error (positive = outside circle)
        e_perp = r - self.R
        
        # Arclength: s = R * theta (for CCW from theta=0)
        s = self.R * theta
        
        return {
            'p_closest': (px, py),
            't_hat': (t_hat_x, t_hat_y),
            'n_hat': (n_hat_x, n_hat_y),
            'e_perp': e_perp,
            's': s,
            'theta': theta,
            'r': r,
        }


@dataclass
class PolylinePath(Path):
    """Polyline path through waypoints."""
    waypoints: List[Tuple[float, float]] = field(default_factory=list)
    closed: bool = False
    
    def __post_init__(self):
        if len(self.waypoints) < 2:
            raise ValueError("Polyline needs at least 2 waypoints")
        self._compute_segments()
    
    def _compute_segments(self):
        """Precompute segment data."""
        self._segments = []
        self._cumulative_lengths = [0.0]
        
        pts = list(self.waypoints)
        if self.closed:
            pts.append(pts[0])
        
        for i in range(len(pts) - 1):
            p1 = pts[i]
            p2 = pts[i + 1]
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            length = np.sqrt(dx**2 + dy**2)
            if length < 1e-12:
                length = 1e-12
            self._segments.append({
                'p1': p1,
                'p2': p2,
                'dx': dx,
                'dy': dy,
                'length': length,
                't_hat': (dx / length, dy / length),
                'n_hat': (-dy / length, dx / length),  # left of tangent
            })
            self._cumulative_lengths.append(self._cumulative_lengths[-1] + length)
    
    def closest_point(self, x: float, y: float) -> Tuple[float, float]:
        best_dist = float('inf')
        best_point = self._segments[0]['p1']
        
        for seg in self._segments:
            px, py = self._closest_point_on_segment(x, y, seg)
            dist = (x - px)**2 + (y - py)**2
            if dist < best_dist:
                best_dist = dist
                best_point = (px, py)
        
        return best_point
    
    def _closest_point_on_segment(self, x: float, y: float, seg: dict) -> Tuple[float, float]:
        p1 = seg['p1']
        dx, dy = x - p1[0], y - p1[1]
        t = (dx * seg['dx'] + dy * seg['dy']) / (seg['length']**2)
        t = np.clip(t, 0, 1)
        return (p1[0] + t * seg['dx'], p1[1] + t * seg['dy'])
    
    def tangent_at(self, s: float) -> Tuple[float, float]:
        seg_idx = self._segment_at_arclength(s)
        return self._segments[seg_idx]['t_hat']
    
    def normal_at(self, s: float) -> Tuple[float, float]:
        seg_idx = self._segment_at_arclength(s)
        return self._segments[seg_idx]['n_hat']
    
    def _segment_at_arclength(self, s: float) -> int:
        for i, cum_len in enumerate(self._cumulative_lengths[1:], 0):
            if s <= cum_len:
                return i
        return len(self._segments) - 1
    
    def arclength_at(self, x: float, y: float) -> float:
        best_s = 0.0
        best_dist = float('inf')
        
        for i, seg in enumerate(self._segments):
            px, py = self._closest_point_on_segment(x, y, seg)
            dist = (x - px)**2 + (y - py)**2
            if dist < best_dist:
                best_dist = dist
                # Compute arclength along this segment
                t = np.sqrt((px - seg['p1'][0])**2 + (py - seg['p1'][1])**2)
                best_s = self._cumulative_lengths[i] + t
        
        return best_s
    
    def total_length(self) -> float:
        return self._cumulative_lengths[-1]
    
    def is_closed(self) -> bool:
        return self.closed


# =============================================================================
# Loss Configuration
# =============================================================================

@dataclass
class PathLossConfig:
    """Configuration for path tracking loss."""
    # Path (set one of these)
    path: Optional[Path] = None
    
    # Shortcut for circle path
    cx: float = 1.0e-3
    cy: float = 1.0e-3
    R: float = 0.4e-3
    ccw: bool = True
    
    # Primary weights
    w_perp: float = 1.0e12    # lateral error weight
    w_prog: float = 1.0e-6    # progress reward weight
    w_U: float = 1.0          # Gor'kov potential weight
    w_reg: float = 0.0        # control magnitude regularization
    
    # Speed tracking (alternative to pure progress reward)
    w_sdot: float = 0.0       # speed tracking weight
    v_ref: float = 0.0        # desired tangential speed [m/s]
    
    # Control smoothness
    w_du: float = 0.0         # ||u_t - u_{t-1}||^2
    
    # Soft bounds
    w_bounds: float = 0.0
    v_min: float = 0.01
    v_max: float = 0.2
    phi_min: float = -np.pi
    phi_max: float = np.pi
    bounds_margin: float = 0.01
    
    # Domain penalty
    w_domain: float = 0.0
    domain_margin: float = 0.1e-3
    Lx: float = 2.0e-3
    Ly: float = 2.0e-3
    
    # Reference control
    v_ref_ctrl: float = 0.05
    phi_ref: float = 0.0
    
    # Terminal cost
    beta_terminal: float = 1.0
    w_perp_T: float = 1.0     # terminal lateral error multiplier
    w_UT: float = 0.5         # terminal U multiplier
    
    # Progress mode
    progress_mode: Literal["force", "displacement"] = "displacement"
    
    # Time step
    dt: float = 0.05
    
    def get_path(self) -> Path:
        """Get or create path object."""
        if self.path is not None:
            return self.path
        return CirclePath(cx=self.cx, cy=self.cy, R=self.R, ccw=self.ccw)


# =============================================================================
# Metrics and Loss Computation
# =============================================================================

def path_metrics(x: float, y: float, path: Path) -> Dict[str, any]:
    """Get all path metrics for a position."""
    return path.metrics(x, y)


def compute_progress(
    x_t: Tuple[float, float],
    x_tp1: Optional[Tuple[float, float]],
    F_t: Tuple[float, float],
    path: Path,
    cfg: PathLossConfig,
) -> float:
    """
    Compute progress along path.
    
    For displacement mode: (x_{t+1} - x_t)/dt · t_hat(x_t)
    For force mode: F_t · t_hat(x_t)
    """
    metrics = path.metrics(x_t[0], x_t[1])
    t_hat_x, t_hat_y = metrics['t_hat']
    
    if cfg.progress_mode == "displacement":
        if x_tp1 is None:
            return 0.0
        vel_x = (x_tp1[0] - x_t[0]) / cfg.dt
        vel_y = (x_tp1[1] - x_t[1]) / cfg.dt
        return vel_x * t_hat_x + vel_y * t_hat_y
    else:  # force mode
        return F_t[0] * t_hat_x + F_t[1] * t_hat_y


def _soft_bounds_penalty(val: float, lo: float, hi: float, margin: float) -> float:
    """Soft quadratic penalty for value approaching bounds."""
    penalty = 0.0
    if val < lo + margin:
        penalty += (lo + margin - val)**2
    if val > hi - margin:
        penalty += (val - (hi - margin))**2
    return penalty


def _domain_penalty(x: float, y: float, Lx: float, Ly: float, margin: float) -> float:
    """Penalty for particle being near domain boundaries."""
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
    cfg: PathLossConfig,
) -> Dict[str, float]:
    """
    Compute per-step loss L_t and its components.
    
    L_t = w_perp * e_perp^2 
          + w_U * U
          + w_reg * ||u - u_ref||^2
          + w_du * ||u - u_{t-1}||^2
          - w_prog * prog_t
          + w_sdot * (prog_t - v_ref)^2
          + w_bounds * bounds_penalty
          + w_domain * domain_penalty
    """
    path = cfg.get_path()
    x, y = x_t
    v, phi = u_t
    
    metrics = path.metrics(x, y)
    e_perp = metrics['e_perp']
    t_hat = metrics['t_hat']
    
    # Progress
    prog = compute_progress(x_t, x_tp1, F_t, path, cfg)
    
    # === Loss components ===
    L_perp = cfg.w_perp * e_perp**2
    
    L_prog = -cfg.w_prog * prog
    
    L_sdot = 0.0
    if cfg.w_sdot > 0:
        L_sdot = cfg.w_sdot * (prog - cfg.v_ref)**2
    
    L_U = cfg.w_U * U_t
    
    L_reg = cfg.w_reg * ((v - cfg.v_ref_ctrl)**2 + (phi - cfg.phi_ref)**2)
    
    L_du = 0.0
    if cfg.w_du > 0 and u_tm1 is not None:
        v_tm1, phi_tm1 = u_tm1
        L_du = cfg.w_du * ((v - v_tm1)**2 + (phi - phi_tm1)**2)
    
    L_bounds = 0.0
    if cfg.w_bounds > 0:
        L_bounds += cfg.w_bounds * _soft_bounds_penalty(v, cfg.v_min, cfg.v_max, cfg.bounds_margin)
        L_bounds += cfg.w_bounds * _soft_bounds_penalty(phi, cfg.phi_min, cfg.phi_max, cfg.bounds_margin)
    
    L_domain = 0.0
    if cfg.w_domain > 0:
        L_domain = cfg.w_domain * _domain_penalty(x, y, cfg.Lx, cfg.Ly, cfg.domain_margin)
    
    L_total = L_perp + L_prog + L_sdot + L_U + L_reg + L_du + L_bounds + L_domain
    
    return {
        'L_total': L_total,
        'L_perp': L_perp,
        'L_prog': L_prog,
        'L_sdot': L_sdot,
        'L_U': L_U,
        'L_reg': L_reg,
        'L_du': L_du,
        'L_bounds': L_bounds,
        'L_domain': L_domain,
        'e_perp': e_perp,
        'progress': prog,
        's': metrics['s'],
        't_hat': t_hat,
    }


def compute_trajectory_loss(
    positions: List[Tuple[float, float]],
    controls: List[Tuple[float, float]],
    forces: List[Tuple[float, float]],
    U_values: List[float],
    cfg: PathLossConfig,
    terminal_U: Optional[float] = None,
    u_init: Optional[Tuple[float, float]] = None,
) -> Dict[str, float]:
    """
    Compute full trajectory loss J.
    
    J = Σ_{t=0}^{K-1} L_t + β_T * L_terminal
    """
    K = len(controls)
    path = cfg.get_path()
    
    J = {k: 0.0 for k in ['J_total', 'J_perp', 'J_prog', 'J_sdot', 'J_U', 
                           'J_reg', 'J_du', 'J_bounds', 'J_domain', 'J_terminal']}
    total_progress = 0.0
    
    for t in range(K):
        x_t = positions[t]
        x_tp1 = positions[t + 1] if cfg.progress_mode == "displacement" else None
        u_t = controls[t]
        u_tm1 = controls[t - 1] if t > 0 else u_init
        F_t = forces[t]
        U_t = U_values[t]
        
        step = compute_step_loss(x_t, x_tp1, u_t, u_tm1, F_t, U_t, cfg)
        
        J['J_total'] += step['L_total']
        J['J_perp'] += step['L_perp']
        J['J_prog'] += step['L_prog']
        J['J_sdot'] += step['L_sdot']
        J['J_U'] += step['L_U']
        J['J_reg'] += step['L_reg']
        J['J_du'] += step['L_du']
        J['J_bounds'] += step['L_bounds']
        J['J_domain'] += step['L_domain']
        total_progress += step['progress']
    
    # Terminal cost
    if cfg.beta_terminal > 0:
        x_K, y_K = positions[-1]
        metrics_K = path.metrics(x_K, y_K)
        e_perp_K = metrics_K['e_perp']
        
        J_terminal_perp = cfg.w_perp_T * cfg.w_perp * e_perp_K**2
        J_terminal_U = cfg.w_UT * cfg.w_U * (terminal_U if terminal_U is not None else U_values[-1])
        J['J_terminal'] = cfg.beta_terminal * (J_terminal_perp + J_terminal_U)
        
        J['J_total'] += J['J_terminal']
        J['J_perp'] += cfg.beta_terminal * J_terminal_perp
        J['J_U'] += cfg.beta_terminal * J_terminal_U
    
    J['total_progress'] = total_progress
    
    return J


# =============================================================================
# Gradient Helpers for Adjoint Recursion
# =============================================================================

def compute_dL_dx_t(
    x_t: Tuple[float, float],
    x_tp1: Optional[Tuple[float, float]],
    u_t: Tuple[float, float],
    F_t: Tuple[float, float],
    cfg: PathLossConfig,
    dU_dx: np.ndarray,
    dF_dx: np.ndarray,
    eps: float = 1e-7,
) -> np.ndarray:
    """
    Compute ∂L_t/∂x_t for adjoint recursion.
    
    Includes:
    - ∂(w_perp * e_perp^2)/∂x_t
    - ∂L_prog/∂x_t (depends on mode)
    - w_U * ∂U/∂x_t
    - domain penalty gradient
    """
    path = cfg.get_path()
    x, y = x_t
    
    # === Lateral error gradient ===
    # Use FD for path geometry derivatives
    m0 = path.metrics(x, y)
    m_xp = path.metrics(x + eps, y)
    m_xm = path.metrics(x - eps, y)
    m_yp = path.metrics(x, y + eps)
    m_ym = path.metrics(x, y - eps)
    
    de_perp_dx = (m_xp['e_perp'] - m_xm['e_perp']) / (2 * eps)
    de_perp_dy = (m_yp['e_perp'] - m_ym['e_perp']) / (2 * eps)
    
    dL_perp_dx = cfg.w_perp * 2 * m0['e_perp'] * np.array([de_perp_dx, de_perp_dy])
    
    # === U gradient ===
    dL_U_dx = cfg.w_U * dU_dx
    
    # === Progress gradient ===
    if cfg.progress_mode == "force":
        # L_prog = -w_prog * (F · t_hat)
        # ∂L_prog/∂x = -w_prog * (∂F/∂x · t_hat + F · ∂t_hat/∂x)
        t_hat = np.array(m0['t_hat'])
        
        # dF/dx contribution
        dFdot_dx_from_F = np.array([
            dF_dx[0, 0] * t_hat[0] + dF_dx[1, 0] * t_hat[1],
            dF_dx[0, 1] * t_hat[0] + dF_dx[1, 1] * t_hat[1],
        ])
        
        # t_hat derivative
        t_hat_xp = np.array(m_xp['t_hat'])
        t_hat_xm = np.array(m_xm['t_hat'])
        t_hat_yp = np.array(m_yp['t_hat'])
        t_hat_ym = np.array(m_ym['t_hat'])
        
        dt_hat_dx = (t_hat_xp - t_hat_xm) / (2 * eps)
        dt_hat_dy = (t_hat_yp - t_hat_ym) / (2 * eps)
        
        F_vec = np.array(F_t)
        dFdot_dx_from_t = np.array([
            np.dot(F_vec, dt_hat_dx),
            np.dot(F_vec, dt_hat_dy),
        ])
        
        dL_prog_dx = -cfg.w_prog * (dFdot_dx_from_F + dFdot_dx_from_t)
        
    else:  # displacement mode
        # L_prog = -w_prog * ((x_{t+1} - x_t)/dt · t_hat(x_t))
        if x_tp1 is None:
            dL_prog_dx = np.zeros(2)
        else:
            vel = np.array([(x_tp1[0] - x) / cfg.dt, (x_tp1[1] - y) / cfg.dt])
            t_hat = np.array(m0['t_hat'])
            
            # ∂/∂x_t of (x_{t+1} - x_t)/dt = -1/dt * I
            dL_prog_dx_from_vel = (cfg.w_prog / cfg.dt) * t_hat
            
            # vel · ∂t_hat/∂x_t
            t_hat_xp = np.array(m_xp['t_hat'])
            t_hat_xm = np.array(m_xm['t_hat'])
            t_hat_yp = np.array(m_yp['t_hat'])
            t_hat_ym = np.array(m_ym['t_hat'])
            
            dt_hat_dx = (t_hat_xp - t_hat_xm) / (2 * eps)
            dt_hat_dy = (t_hat_yp - t_hat_ym) / (2 * eps)
            
            dL_prog_dx_from_t = -cfg.w_prog * np.array([
                np.dot(vel, dt_hat_dx),
                np.dot(vel, dt_hat_dy),
            ])
            
            dL_prog_dx = dL_prog_dx_from_vel + dL_prog_dx_from_t
    
    # Speed tracking gradient
    dL_sdot_dx = np.zeros(2)
    if cfg.w_sdot > 0 and cfg.progress_mode == "displacement" and x_tp1 is not None:
        prog = compute_progress(x_t, x_tp1, F_t, path, cfg)
        # Similar to progress gradient but with 2 * w_sdot * (prog - v_ref) factor
        # Simplified: use FD
        prog_xp = compute_progress((x + eps, y), x_tp1, F_t, path, cfg)
        prog_xm = compute_progress((x - eps, y), x_tp1, F_t, path, cfg)
        prog_yp = compute_progress((x, y + eps), x_tp1, F_t, path, cfg)
        prog_ym = compute_progress((x, y - eps), x_tp1, F_t, path, cfg)
        
        dprog_dx = (prog_xp - prog_xm) / (2 * eps)
        dprog_dy = (prog_yp - prog_ym) / (2 * eps)
        
        dL_sdot_dx = cfg.w_sdot * 2 * (prog - cfg.v_ref) * np.array([dprog_dx, dprog_dy])
    
    # === Domain penalty ===
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
    
    return dL_perp_dx + dL_U_dx + dL_prog_dx + dL_sdot_dx + dL_domain_dx


def compute_dL_dx_tp1(
    x_t: Tuple[float, float],
    x_tp1: Tuple[float, float],
    cfg: PathLossConfig,
) -> np.ndarray:
    """
    Compute ∂L_t/∂x_{t+1} for displacement mode.
    
    Only non-zero when progress_mode == "displacement".
    """
    if cfg.progress_mode != "displacement":
        return np.zeros(2)
    
    path = cfg.get_path()
    x, y = x_t
    metrics = path.metrics(x, y)
    t_hat = np.array(metrics['t_hat'])
    
    # L_prog = -w_prog * ((x_{t+1} - x_t)/dt · t_hat)
    # ∂L_prog/∂x_{t+1} = -w_prog * (1/dt) * t_hat
    dL_prog = -cfg.w_prog / cfg.dt * t_hat
    
    # Speed tracking term
    dL_sdot = np.zeros(2)
    if cfg.w_sdot > 0:
        F_dummy = (0.0, 0.0)  # F not used in displacement mode progress
        prog = compute_progress(x_t, x_tp1, F_dummy, path, cfg)
        # ∂L_sdot/∂x_{t+1} = w_sdot * 2 * (prog - v_ref) * (1/dt) * t_hat
        dL_sdot = cfg.w_sdot * 2 * (prog - cfg.v_ref) / cfg.dt * t_hat
    
    return dL_prog + dL_sdot


def compute_dL_du(
    u_t: Tuple[float, float],
    u_tm1: Optional[Tuple[float, float]],
    u_tp1: Optional[Tuple[float, float]],
    dU_du: Tuple[float, float],
    dF_du: Tuple[np.ndarray, np.ndarray],
    x_t: Tuple[float, float],
    cfg: PathLossConfig,
) -> np.ndarray:
    """
    Compute ∂L_t/∂u_t (direct term).
    """
    path = cfg.get_path()
    v, phi = u_t
    dU_dv, dU_dphi = dU_du
    dF_dv, dF_dphi = dF_du
    
    # === U term ===
    dL_U_du = cfg.w_U * np.array([dU_dv, dU_dphi])
    
    # === Regularization ===
    dL_reg_du = cfg.w_reg * 2 * np.array([v - cfg.v_ref_ctrl, phi - cfg.phi_ref])
    
    # === Smoothness (forward) ===
    dL_du_t_from_smooth = np.zeros(2)
    if cfg.w_du > 0 and u_tm1 is not None:
        v_tm1, phi_tm1 = u_tm1
        dL_du_t_from_smooth = cfg.w_du * 2 * np.array([v - v_tm1, phi - phi_tm1])
    
    # === Smoothness (backward from L_{t+1}) ===
    dL_du_tp1_from_smooth = np.zeros(2)
    if cfg.w_du > 0 and u_tp1 is not None:
        v_tp1, phi_tp1 = u_tp1
        dL_du_tp1_from_smooth = cfg.w_du * 2 * np.array([v - v_tp1, phi - phi_tp1])
    
    # === Progress term (force mode only) ===
    dL_prog_du = np.zeros(2)
    if cfg.progress_mode == "force":
        metrics = path.metrics(x_t[0], x_t[1])
        t_hat = metrics['t_hat']
        
        dFdot_dv = dF_dv[0] * t_hat[0] + dF_dv[1] * t_hat[1]
        dFdot_dphi = dF_dphi[0] * t_hat[0] + dF_dphi[1] * t_hat[1]
        
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
