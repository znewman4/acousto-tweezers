#!/usr/bin/env python3
"""
adjoint_path_track_mpc_compare.py - Side-by-side comparison of Greedy vs MPC path tracking.

This script runs two controllers from identical initial conditions:
1. **Greedy controller**: Selects macro actions maximizing force alignment with path tangent
2. **Adjoint MPC**: Rolling-horizon optimization with discrete adjoint gradients

Both use the SAME Evaluator4Pucks infrastructure as the working 4puck_demo_surf_greedy.py,
ensuring forces and U fields match the demo regime where motion is visible.

Outputs (in results/adjoint_path_track_mpc_compare/run_YYYYMMDD_HHMMSS/):
    - trajectory_greedy.csv, trajectory_mpc.csv
    - controls_greedy.csv, controls_mpc.csv
    - summary_metrics.csv
    - compare_summary.json
    - gorkov_contour_compare.gif (split-screen)
    - gorkov_contour_compare_controls.gif (split-screen with control display)
    - compare_plot.png (stacked comparison plots)

Usage:
    python scripts/adjoint_path_track_mpc_compare.py --fast
    python scripts/adjoint_path_track_mpc_compare.py --K 5 --T 50 --w_prog 1e-6
    python scripts/adjoint_path_track_mpc_compare.py --preset simple_single
"""

from __future__ import annotations

import matplotlib
matplotlib.use('Agg')

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

# Import the SAME infrastructure as the working demo
from tweezers.control import (
    DishDomain, MediumProps, EvaluatorConfig,
    Control4Pucks, default_4puck_config,
)
from tweezers.control.evaluator_4pucks import Evaluator4Pucks
from acousto.force import ParticleProps, gorkov_potential_and_force_2d, bilinear_sample_vec
from acousto.adjoint.losses.path_track import (
    Path as TrackPath,
    CirclePath,
    PolylinePath,
    PathLossConfig,
    path_metrics,
    compute_progress,
    compute_step_loss as path_compute_step_loss,
)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class CompareConfig:
    """Configuration for MPC vs Greedy comparison."""
    # Domain (matches 4puck_demo)
    Lx: float = 2.0e-3
    Ly: float = 2.0e-3
    Nx: int = 80
    Ny: int = 80
    
    # Physics (matches 4puck_demo)
    f: float = 2.0e6
    c0: float = 1500.0
    rho0: float = 1000.0
    loss_eta: float = 1e-3
    kz: float = 0.0
    coupling_alpha: float = 1.0
    
    # Transducer params (matches 4puck_demo)
    sigma_x: float = 0.10e-3
    sigma_y: float = 0.15e-3
    bottom_band: float = 0.25e-3
    
    # Particle (matches 4puck_demo)
    particle_a: float = 5.0e-6
    particle_rho_p: float = 1050.0
    particle_c_p: float = 2350.0
    
    # Dynamics (matches 4puck_demo)
    dt: float = 5e-3            # Same as demo!
    viscosity: float = 1e-3
    alpha_g: float = 2e3        # Critical scaling factor!
    max_step: float = 0.08e-3
    
    # MPC parameters
    K: int = 5              # horizon length
    T: int = 50             # total executed steps
    n_iters: int = 8        # optimization iterations per MPC solve
    
    # Path geometry (circle default)
    path_type: str = "circle"
    cx: float = 1.0e-3
    cy: float = 1.1e-3      # Matches demo circle center
    R: float = 0.4e-3
    ccw: bool = True
    
    # Objective weights
    w_perp: float = 1.0e12    # lateral error weight
    w_prog: float = 1.0e-6    # progress reward
    w_U: float = 1.0          # Gor'kov potential
    w_reg: float = 0.0        # control regularization
    w_du: float = 0.0         # control smoothness
    
    # Progress mode
    progress_mode: str = "force"
    
    # Terminal weights
    beta_terminal: float = 1.0
    w_perp_T: float = 1.0
    w_UT: float = 0.5
    
    # Optimization
    alphas: Tuple[float, ...] = (0.0, 0.01, 0.05, 0.1, 0.3, 1.0)
    
    # FD epsilons
    fd_eps_v: float = 1e-3
    fd_eps_phi: float = 1e-3
    fd_eps_x: float = 1e-7
    
    # Control bounds - matching the demo amplitudes (v ~ 0.08)
    v_min: float = 0.02
    v_max: float = 0.15
    v_ref_ctrl: float = 0.08  # Demo uses 0.08
    phi_ref: float = 0.0
    
    # Greedy parameters
    greedy_n_candidates: int = 9
    greedy_v_range: Tuple[float, float] = (0.04, 0.12)  # Higher amplitudes
    greedy_phi_range: Tuple[float, float] = (-np.pi, np.pi)
    
    # Initial position (on circle)
    theta0: float = -np.pi / 2  # Start at bottom of circle
    
    def get_path(self) -> TrackPath:
        """Create path object."""
        if self.path_type == "circle":
            return CirclePath(cx=self.cx, cy=self.cy, R=self.R, ccw=self.ccw)
        else:
            raise NotImplementedError(f"Path type {self.path_type} not implemented yet")
    
    def get_loss_config(self) -> PathLossConfig:
        """Create PathLossConfig for loss computation."""
        return PathLossConfig(
            path=self.get_path(),
            cx=self.cx, cy=self.cy, R=self.R, ccw=self.ccw,
            w_perp=self.w_perp, w_prog=self.w_prog, w_U=self.w_U,
            w_reg=self.w_reg, w_du=self.w_du,
            v_ref_ctrl=self.v_ref_ctrl, phi_ref=self.phi_ref,
            beta_terminal=self.beta_terminal,
            w_perp_T=self.w_perp_T, w_UT=self.w_UT,
            progress_mode=self.progress_mode,
            dt=self.dt,
            Lx=self.Lx, Ly=self.Ly,
            v_min=self.v_min, v_max=self.v_max,
        )


def create_evaluator(cfg: CompareConfig) -> Tuple[Evaluator4Pucks, ParticleProps]:
    """Create Evaluator4Pucks with same config as working demo."""
    domain = DishDomain(Lx=cfg.Lx, Ly=cfg.Ly, Nx=cfg.Nx, Ny=cfg.Ny)
    medium = MediumProps(
        f=cfg.f, c0=cfg.c0, rho0=cfg.rho0,
        loss_eta=cfg.loss_eta, kz=cfg.kz, coupling_alpha=cfg.coupling_alpha
    )
    particle = ParticleProps(a=cfg.particle_a, rho_p=cfg.particle_rho_p, c_p=cfg.particle_c_p)
    
    ev_cfg = EvaluatorConfig(
        sigma_x=cfg.sigma_x,
        sigma_y=cfg.sigma_y,
        bottom_band=cfg.bottom_band,
        dt=cfg.dt,
        viscosity=cfg.viscosity,
        alpha_g=cfg.alpha_g,
        max_step=cfg.max_step,
        use_2d_forcing=True,
    )
    
    ev = Evaluator4Pucks(domain, medium, particle, ev_cfg)
    return ev, particle


# =============================================================================
# Physics Helpers (using Evaluator4Pucks)
# =============================================================================

def compute_U_and_F_at_pos(
    ev: Evaluator4Pucks,
    ctrl: Control4Pucks,
    x_p: float, y_p: float,
) -> Tuple[Any, np.ndarray, float, float, float]:
    """
    Compute Gor'kov potential and force at particle position using Evaluator4Pucks.
    
    Returns: (field, U_full, U_interp, Fx_interp, Fy_interp)
    """
    # Solve PDE
    vb = ev.control_to_forcing_band_vb(ctrl)
    field = ev.op.solve_for_bottom_vb(vb)
    
    # Compute Gor'kov potential and force
    U, Fx, Fy = gorkov_potential_and_force_2d(field, ev.particle)
    
    # Apply alpha_g scaling (critical!)
    Fx_scaled = Fx * ev.cfg.alpha_g
    Fy_scaled = Fy * ev.cfg.alpha_g
    
    # Bilinear interpolation at particle position
    fx, fy = bilinear_sample_vec(field.x, field.y, Fx_scaled, Fy_scaled, x_p, y_p)
    
    # Interpolate U
    ix_f = (x_p - ev.op.x[0]) / ev.op.dx
    iy_f = (y_p - ev.op.y[0]) / ev.op.dy
    
    ix0 = int(np.clip(np.floor(ix_f), 0, ev.op.Nx - 2))
    iy0 = int(np.clip(np.floor(iy_f), 0, ev.op.Ny - 2))
    ix1, iy1 = ix0 + 1, iy0 + 1
    
    fx_w = np.clip(ix_f - ix0, 0, 1)
    fy_w = np.clip(iy_f - iy0, 0, 1)
    
    U_interp = (U[iy0, ix0] * (1-fx_w) * (1-fy_w) + U[iy0, ix1] * fx_w * (1-fy_w) +
                U[iy1, ix0] * (1-fx_w) * fy_w + U[iy1, ix1] * fx_w * fy_w)
    
    return field, U, float(U_interp), float(fx), float(fy)


def overdamped_step(
    x: float, y: float,
    Fx: float, Fy: float,
    cfg: CompareConfig,
) -> Tuple[float, float]:
    """One overdamped particle step with step limiting."""
    gamma = 6.0 * np.pi * cfg.viscosity * cfg.particle_a
    
    dx_raw = cfg.dt * Fx / gamma
    dy_raw = cfg.dt * Fy / gamma
    raw_displacement = np.sqrt(dx_raw**2 + dy_raw**2)
    
    # Step limiting (same as evaluator)
    if raw_displacement > cfg.max_step and raw_displacement > 0:
        scale = cfg.max_step / raw_displacement
        dx_raw *= scale
        dy_raw *= scale
    
    x_new = np.clip(x + dx_raw, 0, cfg.Lx)
    y_new = np.clip(y + dy_raw, 0, cfg.Ly)
    return float(x_new), float(y_new)


def get_mobility(cfg: CompareConfig) -> float:
    """Compute particle mobility."""
    gamma = 6.0 * np.pi * cfg.viscosity * cfg.particle_a
    return 1.0 / gamma


def clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def normalize_phi(phi: float) -> float:
    return ((phi + np.pi) % (2 * np.pi)) - np.pi


# =============================================================================
# Control Helpers
# =============================================================================

def make_control_with_vB(v: float, phi: float, cfg: CompareConfig) -> Control4Pucks:
    """
    Create Control4Pucks with transducers C and D active (like the working demo).
    
    The demo typically uses C+D configuration which creates a stronger trap.
    We vary v (amplitude) and phi (phase of C relative to D) to control the trap.
    
    - v controls the overall amplitude (both C and D)
    - phi controls the phase difference between C and D, which shifts the trap
    """
    Lx, Ly = cfg.Lx, cfg.Ly
    
    # Base amplitude for C and D
    v_base = v
    
    # Phase difference between C and D controls trap position
    phi_C = phi
    phi_D = 0.0  # D is reference
    
    return Control4Pucks(
        # A: off
        xA=0.25 * Lx, yA=0.02 * Ly, vA=0.0, phiA=0.0, gateA=False,
        # B: off
        xB=0.75 * Lx, yB=0.02 * Ly, vB=0.0, phiB=0.0, gateB=False,
        # C: active - left-center of bottom
        xC=0.35 * Lx, yC=0.02 * Ly, vC=v_base, phiC=phi_C, gateC=True,
        # D: active - right-center of bottom  
        xD=0.65 * Lx, yD=0.02 * Ly, vD=v_base, phiD=phi_D, gateD=True,
    )


def make_control_multi_puck(v_arr: np.ndarray, phi_arr: np.ndarray, cfg: CompareConfig) -> Control4Pucks:
    """
    Create Control4Pucks with multiple transducers active.
    
    v_arr, phi_arr: arrays of length 4 for [A, B, C, D]
    """
    Lx, Ly = cfg.Lx, cfg.Ly
    return Control4Pucks(
        xA=0.25 * Lx, yA=0.02 * Ly, vA=v_arr[0], phiA=phi_arr[0], gateA=v_arr[0] > 0.001,
        xB=0.75 * Lx, yB=0.02 * Ly, vB=v_arr[1], phiB=phi_arr[1], gateB=v_arr[1] > 0.001,
        xC=0.50 * Lx, yC=0.02 * Ly, vC=v_arr[2], phiC=phi_arr[2], gateC=v_arr[2] > 0.001,
        xD=0.50 * Lx, yD=0.15 * Ly, vD=v_arr[3], phiD=phi_arr[3], gateD=v_arr[3] > 0.001,
    )


# =============================================================================
# Trajectory Storage
# =============================================================================

@dataclass
class TrajectoryResult:
    """Results from either greedy or MPC run."""
    method: str
    positions: List[Tuple[float, float]]
    applied_controls: List[Tuple[float, float]]  # (v, phi) for simplicity
    forces: List[Tuple[float, float]]
    U_values: List[float]
    U_fields: List[np.ndarray]
    Fx_fields: List[np.ndarray]
    Fy_fields: List[np.ndarray]
    per_step_metrics: List[Dict[str, float]]
    runtime_s: float


# =============================================================================
# GREEDY CONTROLLER
# =============================================================================

def greedy_score(
    Fx: float, Fy: float, U: float,
    t_hat_x: float, t_hat_y: float,
    e_perp: float,
    v: float, phi: float,
    v_prev: Optional[float], phi_prev: Optional[float],
    cfg: CompareConfig,
) -> float:
    """
    Compute greedy score for an action.
    Higher is better: maximize force alignment and progress, minimize lateral error.
    """
    F_mag = np.sqrt(Fx**2 + Fy**2) + 1e-15
    
    # Force projection along tangent (progress)
    F_dot_t = Fx * t_hat_x + Fy * t_hat_y
    
    # Score: reward progress, penalize error and potential
    score = (cfg.w_prog * F_dot_t 
             - cfg.w_perp * e_perp**2 
             - cfg.w_U * U)
    
    # Control smoothness penalty
    if cfg.w_du > 0 and v_prev is not None:
        score -= cfg.w_du * ((v - v_prev)**2 + (phi - phi_prev)**2)
    
    return score


def run_greedy(
    ev: Evaluator4Pucks,
    x0: float, y0: float,
    v_init: float, phi_init: float,
    cfg: CompareConfig,
    verbose: bool = True,
) -> TrajectoryResult:
    """Run greedy controller for T steps."""
    positions = [(x0, y0)]
    applied_controls = []
    forces = []
    U_values = []
    U_fields = []
    Fx_fields = []
    Fy_fields = []
    per_step_metrics = []
    
    path = cfg.get_path()
    loss_cfg = cfg.get_loss_config()
    
    x, y = x0, y0
    v_prev, phi_prev = v_init, phi_init
    
    if verbose:
        print(f"\n   Running Greedy: T={cfg.T} steps")
        print(f"   Progress: ", end="", flush=True)
    
    start_time = time.time()
    
    # Generate candidate controls grid
    n = cfg.greedy_n_candidates
    v_candidates = np.linspace(cfg.greedy_v_range[0], cfg.greedy_v_range[1], n)
    phi_candidates = np.linspace(cfg.greedy_phi_range[0], cfg.greedy_phi_range[1], n, endpoint=False)
    
    for t in range(cfg.T):
        metrics = path.metrics(x, y)
        t_hat_x, t_hat_y = metrics['t_hat']
        e_perp = metrics['e_perp']
        
        best_score = -np.inf
        best_v, best_phi = v_init, phi_init
        best_Fx, best_Fy = 0.0, 0.0
        best_U = 0.0
        best_U_full = None
        best_Fx_full = None
        best_Fy_full = None
        
        # Evaluate all candidates
        for v_cand in v_candidates:
            for phi_cand in phi_candidates:
                ctrl = make_control_with_vB(v_cand, phi_cand, cfg)
                field, U_full, U, Fx, Fy = compute_U_and_F_at_pos(ev, ctrl, x, y)
                
                score = greedy_score(
                    Fx, Fy, U,
                    t_hat_x, t_hat_y, e_perp,
                    v_cand, phi_cand,
                    v_prev, phi_prev,
                    cfg
                )
                
                if score > best_score:
                    best_score = score
                    best_v, best_phi = v_cand, phi_cand
                    best_Fx, best_Fy = Fx, Fy
                    best_U = U
                    best_U_full = U_full
                    # Get scaled force fields
                    _, Fx_raw, Fy_raw = gorkov_potential_and_force_2d(field, ev.particle)
                    best_Fx_full = Fx_raw * ev.cfg.alpha_g
                    best_Fy_full = Fy_raw * ev.cfg.alpha_g
        
        # Store results
        applied_controls.append((best_v, best_phi))
        forces.append((best_Fx, best_Fy))
        U_values.append(best_U)
        U_fields.append(best_U_full)
        Fx_fields.append(best_Fx_full)
        Fy_fields.append(best_Fy_full)
        
        # Compute step metrics
        step_loss = path_compute_step_loss(
            (x, y), None, (best_v, best_phi), 
            (v_prev, phi_prev) if t > 0 else None,
            (best_Fx, best_Fy), best_U, loss_cfg
        )
        per_step_metrics.append(step_loss)
        
        # Step dynamics
        x_new, y_new = overdamped_step(x, y, best_Fx, best_Fy, cfg)
        positions.append((x_new, y_new))
        
        v_prev, phi_prev = best_v, best_phi
        x, y = x_new, y_new
        
        if verbose and (t + 1) % 10 == 0:
            print(f"{t+1}", end=" ", flush=True)
    
    elapsed = time.time() - start_time
    
    if verbose:
        print(f"\n   Completed in {elapsed:.1f}s ({elapsed/cfg.T*1000:.1f} ms/step)")
    
    return TrajectoryResult(
        method="greedy",
        positions=positions,
        applied_controls=applied_controls,
        forces=forces,
        U_values=U_values,
        U_fields=U_fields,
        Fx_fields=Fx_fields,
        Fy_fields=Fy_fields,
        per_step_metrics=per_step_metrics,
        runtime_s=elapsed,
    )


# =============================================================================
# MPC CONTROLLER
# =============================================================================

def compute_mpc_step_objective(
    x: float, y: float,
    Fx: float, Fy: float,
    U: float,
    v: float, phi: float,
    v_prev: Optional[float], phi_prev: Optional[float],
    cfg: CompareConfig,
    path: TrackPath,
) -> Dict[str, float]:
    """Compute per-step objective for MPC."""
    metrics = path.metrics(x, y)
    e_perp = metrics['e_perp']
    t_hat_x, t_hat_y = metrics['t_hat']
    
    tangent_progress = Fx * t_hat_x + Fy * t_hat_y
    
    L_perp = cfg.w_perp * e_perp**2
    L_prog = -cfg.w_prog * tangent_progress
    L_U = cfg.w_U * U
    L_reg = cfg.w_reg * ((v - cfg.v_ref_ctrl)**2 + (phi - cfg.phi_ref)**2)
    
    L_du = 0.0
    if cfg.w_du > 0 and v_prev is not None and phi_prev is not None:
        L_du = cfg.w_du * ((v - v_prev)**2 + (phi - phi_prev)**2)
    
    L_total = L_perp + L_prog + L_U + L_reg + L_du
    
    return {
        'L_total': L_total,
        'L_perp': L_perp,
        'L_prog': L_prog,
        'L_U': L_U,
        'L_reg': L_reg,
        'L_du': L_du,
        'e_perp': e_perp,
        'progress': tangent_progress,
        's': metrics['s'],
        't_hat': (t_hat_x, t_hat_y),
    }


def compute_mpc_trajectory_objective(
    positions: List[Tuple[float, float]],
    controls: List[Tuple[float, float]],
    forces: List[Tuple[float, float]],
    U_values: List[float],
    cfg: CompareConfig,
    path: TrackPath,
    u_prev: Optional[Tuple[float, float]] = None,
    terminal_U: Optional[float] = None,
) -> Dict[str, float]:
    """Compute full trajectory objective for MPC."""
    K = len(controls)
    
    J_total, J_perp, J_prog, J_U, J_reg, J_du = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    for t in range(K):
        x, y = positions[t]
        v, phi = controls[t]
        Fx, Fy = forces[t]
        U = U_values[t]
        
        if t == 0:
            v_prev, phi_prev = (u_prev[0], u_prev[1]) if u_prev else (None, None)
        else:
            v_prev, phi_prev = controls[t - 1]
        
        step_obj = compute_mpc_step_objective(x, y, Fx, Fy, U, v, phi, v_prev, phi_prev, cfg, path)
        
        J_total += step_obj['L_total']
        J_perp += step_obj['L_perp']
        J_prog += step_obj['L_prog']
        J_U += step_obj['L_U']
        J_reg += step_obj['L_reg']
        J_du += step_obj['L_du']
    
    # Terminal cost
    J_terminal = 0.0
    if cfg.beta_terminal > 0:
        x_K, y_K = positions[-1]
        metrics_K = path.metrics(x_K, y_K)
        e_perp_K = metrics_K['e_perp']
        
        J_terminal_perp = cfg.w_perp_T * cfg.w_perp * e_perp_K**2
        J_terminal_U = cfg.w_UT * cfg.w_U * (terminal_U if terminal_U is not None else U_values[-1])
        J_terminal = cfg.beta_terminal * (J_terminal_perp + J_terminal_U)
        
        J_total += J_terminal
    
    return {
        'J_total': J_total,
        'J_perp': J_perp,
        'J_prog': J_prog,
        'J_U': J_U,
        'J_reg': J_reg,
        'J_du': J_du,
        'J_terminal': J_terminal,
    }


def rollout_trajectory(
    ev: Evaluator4Pucks,
    controls: List[Tuple[float, float]],
    x0: float, y0: float,
    cfg: CompareConfig,
):
    """Roll out trajectory for K steps."""
    positions = [(x0, y0)]
    forces = []
    U_values = []
    
    x, y = x0, y0
    for v, phi in controls:
        ctrl = make_control_with_vB(v, phi, cfg)
        _, _, U, Fx, Fy = compute_U_and_F_at_pos(ev, ctrl, x, y)
        U_values.append(U)
        forces.append((Fx, Fy))
        x, y = overdamped_step(x, y, Fx, Fy, cfg)
        positions.append((x, y))
    
    return positions, forces, U_values


def compute_dL_dF(x: float, y: float, path: TrackPath, w_prog: float) -> np.ndarray:
    """∂L/∂F = -w_prog * t_hat."""
    metrics = path.metrics(x, y)
    t_hat_x, t_hat_y = metrics['t_hat']
    return np.array([-w_prog * t_hat_x, -w_prog * t_hat_y])


def compute_dF_du_fd(
    ev: Evaluator4Pucks,
    v: float, phi: float,
    x: float, y: float,
    cfg: CompareConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute ∂F/∂(v, phi) via FD."""
    ctrl_vp = make_control_with_vB(v + cfg.fd_eps_v, phi, cfg)
    ctrl_vm = make_control_with_vB(v - cfg.fd_eps_v, phi, cfg)
    _, _, _, Fx_vp, Fy_vp = compute_U_and_F_at_pos(ev, ctrl_vp, x, y)
    _, _, _, Fx_vm, Fy_vm = compute_U_and_F_at_pos(ev, ctrl_vm, x, y)
    dF_dv = np.array([(Fx_vp - Fx_vm) / (2 * cfg.fd_eps_v),
                      (Fy_vp - Fy_vm) / (2 * cfg.fd_eps_v)])
    
    ctrl_pp = make_control_with_vB(v, phi + cfg.fd_eps_phi, cfg)
    ctrl_pm = make_control_with_vB(v, phi - cfg.fd_eps_phi, cfg)
    _, _, _, Fx_pp, Fy_pp = compute_U_and_F_at_pos(ev, ctrl_pp, x, y)
    _, _, _, Fx_pm, Fy_pm = compute_U_and_F_at_pos(ev, ctrl_pm, x, y)
    dF_dphi = np.array([(Fx_pp - Fx_pm) / (2 * cfg.fd_eps_phi),
                        (Fy_pp - Fy_pm) / (2 * cfg.fd_eps_phi)])
    
    return dF_dv, dF_dphi


def compute_dF_dx_fd(
    ev: Evaluator4Pucks,
    v: float, phi: float,
    x: float, y: float,
    cfg: CompareConfig,
) -> np.ndarray:
    """Compute ∂F/∂x via FD."""
    eps = cfg.fd_eps_x
    ctrl = make_control_with_vB(v, phi, cfg)
    _, _, _, Fx_xp, Fy_xp = compute_U_and_F_at_pos(ev, ctrl, x + eps, y)
    _, _, _, Fx_xm, Fy_xm = compute_U_and_F_at_pos(ev, ctrl, x - eps, y)
    _, _, _, Fx_yp, Fy_yp = compute_U_and_F_at_pos(ev, ctrl, x, y + eps)
    _, _, _, Fx_ym, Fy_ym = compute_U_and_F_at_pos(ev, ctrl, x, y - eps)
    
    return np.array([[(Fx_xp - Fx_xm) / (2 * eps), (Fx_yp - Fx_ym) / (2 * eps)],
                     [(Fy_xp - Fy_xm) / (2 * eps), (Fy_yp - Fy_ym) / (2 * eps)]])


def compute_dU_dx_fd(
    ev: Evaluator4Pucks,
    v: float, phi: float,
    x: float, y: float,
    cfg: CompareConfig,
) -> np.ndarray:
    """Compute ∇_x U via FD."""
    eps = cfg.fd_eps_x
    ctrl = make_control_with_vB(v, phi, cfg)
    _, _, U_xp, _, _ = compute_U_and_F_at_pos(ev, ctrl, x + eps, y)
    _, _, U_xm, _, _ = compute_U_and_F_at_pos(ev, ctrl, x - eps, y)
    _, _, U_yp, _, _ = compute_U_and_F_at_pos(ev, ctrl, x, y + eps)
    _, _, U_ym, _, _ = compute_U_and_F_at_pos(ev, ctrl, x, y - eps)
    return np.array([(U_xp - U_xm) / (2 * eps), (U_yp - U_ym) / (2 * eps)])


def compute_dU_du_fd(
    ev: Evaluator4Pucks,
    v: float, phi: float,
    x: float, y: float,
    cfg: CompareConfig,
) -> Tuple[float, float]:
    """Compute ∂U/∂(v, phi) via FD."""
    ctrl_vp = make_control_with_vB(v + cfg.fd_eps_v, phi, cfg)
    ctrl_vm = make_control_with_vB(v - cfg.fd_eps_v, phi, cfg)
    _, _, U_vp, _, _ = compute_U_and_F_at_pos(ev, ctrl_vp, x, y)
    _, _, U_vm, _, _ = compute_U_and_F_at_pos(ev, ctrl_vm, x, y)
    dU_dv = (U_vp - U_vm) / (2 * cfg.fd_eps_v)
    
    ctrl_pp = make_control_with_vB(v, phi + cfg.fd_eps_phi, cfg)
    ctrl_pm = make_control_with_vB(v, phi - cfg.fd_eps_phi, cfg)
    _, _, U_pp, _, _ = compute_U_and_F_at_pos(ev, ctrl_pp, x, y)
    _, _, U_pm, _, _ = compute_U_and_F_at_pos(ev, ctrl_pm, x, y)
    dU_dphi = (U_pp - U_pm) / (2 * cfg.fd_eps_phi)
    
    return dU_dv, dU_dphi


def compute_dL_dx_fd(x: float, y: float, Fx: float, Fy: float, 
                      path: TrackPath, cfg: CompareConfig, eps: float = 1e-7) -> np.ndarray:
    """Compute ∂L/∂x for path tracking via FD."""
    def L_at_pos(xp, yp):
        metrics = path.metrics(xp, yp)
        e_perp = metrics['e_perp']
        t_hat_x, t_hat_y = metrics['t_hat']
        tangent_progress = Fx * t_hat_x + Fy * t_hat_y
        return cfg.w_perp * e_perp**2 - cfg.w_prog * tangent_progress
    
    dL_dx = (L_at_pos(x + eps, y) - L_at_pos(x - eps, y)) / (2 * eps)
    dL_dy = (L_at_pos(x, y + eps) - L_at_pos(x, y - eps)) / (2 * eps)
    return np.array([dL_dx, dL_dy])


def compute_mpc_gradients(
    ev: Evaluator4Pucks,
    controls: List[Tuple[float, float]],
    x0: float, y0: float,
    cfg: CompareConfig,
    path: TrackPath,
    u_prev: Optional[Tuple[float, float]] = None,
) -> Tuple[List[Tuple[float, float]], Any]:
    """Compute gradients ∂J/∂u_t via discrete adjoint."""
    K = len(controls)
    mobility = get_mobility(cfg)
    
    # Forward pass
    positions, forces, U_values = rollout_trajectory(ev, controls, x0, y0, cfg)
    
    gradients = []
    
    # Terminal adjoint
    x_K, y_K = positions[-1]
    v_Km1, phi_Km1 = controls[-1]
    
    if cfg.beta_terminal > 0:
        metrics_K = path.metrics(x_K, y_K)
        e_perp_K = metrics_K['e_perp']
        n_hat_x, n_hat_y = metrics_K['n_hat']
        
        d_eperp_dx = np.array([n_hat_x, n_hat_y])
        dL_terminal_dx_perp = 2 * cfg.w_perp_T * cfg.w_perp * e_perp_K * d_eperp_dx
        dL_terminal_dx_U = cfg.w_UT * cfg.w_U * compute_dU_dx_fd(ev, v_Km1, phi_Km1, x_K, y_K, cfg)
        lambda_t = cfg.beta_terminal * (dL_terminal_dx_perp + dL_terminal_dx_U)
    else:
        lambda_t = np.zeros(2)
    
    # Backward pass
    for t in reversed(range(K)):
        v_t, phi_t = controls[t]
        x_t, y_t = positions[t]
        Fx_t, Fy_t = forces[t]
        
        # === Direct gradient ∂L_t/∂u_t ===
        
        # U term
        dU_dv, dU_dphi = compute_dU_du_fd(ev, v_t, phi_t, x_t, y_t, cfg)
        
        # Regularization
        dreg_dv = cfg.w_reg * 2 * (v_t - cfg.v_ref_ctrl)
        dreg_dphi = cfg.w_reg * 2 * (phi_t - cfg.phi_ref)
        
        # Progress term
        dL_dF = compute_dL_dF(x_t, y_t, path, cfg.w_prog)
        dF_dv, dF_dphi = compute_dF_du_fd(ev, v_t, phi_t, x_t, y_t, cfg)
        dLprog_dv = np.dot(dL_dF, dF_dv)
        dLprog_dphi = np.dot(dL_dF, dF_dphi)
        
        # Smoothness term
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
        
        # Smoothness from L_{t+1}
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
        dL_perp_dx = compute_dL_dx_fd(x_t, y_t, Fx_t, Fy_t, path, cfg, cfg.fd_eps_x)
        dU_dx = compute_dU_dx_fd(ev, v_t, phi_t, x_t, y_t, cfg)
        dL_U_dx = cfg.w_U * dU_dx
        
        dL_dx_total = dL_perp_dx + dL_U_dx
        
        dF_dx = compute_dF_dx_fd(ev, v_t, phi_t, x_t, y_t, cfg)
        dx_dx = np.eye(2) + cfg.dt * mobility * dF_dx
        
        lambda_t = dL_dx_total + dx_dx.T @ lambda_t
    
    gradients.reverse()
    return gradients, (positions, forces, U_values)


def solve_mpc_horizon(
    ev: Evaluator4Pucks,
    x0: float, y0: float,
    u_init: List[Tuple[float, float]],
    cfg: CompareConfig,
    path: TrackPath,
    u_prev: Optional[Tuple[float, float]] = None,
) -> Tuple[List[Tuple[float, float]], float, List[float]]:
    """Solve K-step MPC optimization from current state."""
    controls = list(u_init)
    J_history = []
    
    for iteration in range(cfg.n_iters):
        gradients, (positions, forces, U_values) = compute_mpc_gradients(
            ev, controls, x0, y0, cfg, path, u_prev
        )
        
        # Terminal U
        x_K, y_K = positions[-1]
        v_last, phi_last = controls[-1]
        ctrl_last = make_control_with_vB(v_last, phi_last, cfg)
        _, _, U_K, _, _ = compute_U_and_F_at_pos(ev, ctrl_last, x_K, y_K)
        
        obj = compute_mpc_trajectory_objective(
            positions, controls, forces, U_values, cfg, path, u_prev, U_K
        )
        J_history.append(obj['J_total'])
        
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
            
            new_positions, new_forces, new_U_values = rollout_trajectory(
                ev, new_controls, x0, y0, cfg
            )
            x_K, y_K = new_positions[-1]
            v_last, phi_last = new_controls[-1]
            ctrl_last = make_control_with_vB(v_last, phi_last, cfg)
            _, _, U_K, _, _ = compute_U_and_F_at_pos(ev, ctrl_last, x_K, y_K)
            
            new_obj = compute_mpc_trajectory_objective(
                new_positions, new_controls, new_forces, new_U_values, cfg, path, u_prev, U_K
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


def run_mpc(
    ev: Evaluator4Pucks,
    x0: float, y0: float,
    v_init: float, phi_init: float,
    cfg: CompareConfig,
    verbose: bool = True,
) -> TrajectoryResult:
    """Run rolling-horizon MPC for T steps."""
    path = cfg.get_path()
    loss_cfg = cfg.get_loss_config()
    
    positions = [(x0, y0)]
    applied_controls = []
    forces = []
    U_values = []
    U_fields = []
    Fx_fields = []
    Fy_fields = []
    per_step_metrics = []
    
    x, y = x0, y0
    u_horizon = [(v_init, phi_init) for _ in range(cfg.K)]
    u_prev = None
    
    if verbose:
        print(f"\n   Running MPC: T={cfg.T} steps, K={cfg.K} horizon")
        print(f"   Progress: ", end="", flush=True)
    
    start_time = time.time()
    
    for t in range(cfg.T):
        # Solve MPC horizon
        u_horizon, J_opt, J_hist = solve_mpc_horizon(
            ev, x, y, u_horizon, cfg, path, u_prev
        )
        
        # Apply first control
        v_apply, phi_apply = u_horizon[0]
        applied_controls.append((v_apply, phi_apply))
        
        # Compute physics
        ctrl = make_control_with_vB(v_apply, phi_apply, cfg)
        field, U_full, U, Fx, Fy = compute_U_and_F_at_pos(ev, ctrl, x, y)
        U_values.append(U)
        forces.append((Fx, Fy))
        U_fields.append(U_full)
        
        # Store force fields
        _, Fx_raw, Fy_raw = gorkov_potential_and_force_2d(field, ev.particle)
        Fx_fields.append(Fx_raw * ev.cfg.alpha_g)
        Fy_fields.append(Fy_raw * ev.cfg.alpha_g)
        
        # Compute step metrics
        v_prev_step = applied_controls[-2][0] if len(applied_controls) > 1 else None
        phi_prev_step = applied_controls[-2][1] if len(applied_controls) > 1 else None
        step_loss = path_compute_step_loss(
            (x, y), None, (v_apply, phi_apply),
            (v_prev_step, phi_prev_step) if t > 0 else None,
            (Fx, Fy), U, loss_cfg
        )
        per_step_metrics.append(step_loss)
        
        # Step dynamics
        x_new, y_new = overdamped_step(x, y, Fx, Fy, cfg)
        positions.append((x_new, y_new))
        
        # Warm start
        u_prev = (v_apply, phi_apply)
        u_horizon = u_horizon[1:] + [u_horizon[-1]]
        
        x, y = x_new, y_new
        
        if verbose and (t + 1) % 10 == 0:
            print(f"{t+1}", end=" ", flush=True)
    
    elapsed = time.time() - start_time
    
    if verbose:
        print(f"\n   Completed in {elapsed:.1f}s ({elapsed/cfg.T*1000:.1f} ms/step)")
    
    return TrajectoryResult(
        method="mpc",
        positions=positions,
        applied_controls=applied_controls,
        forces=forces,
        U_values=U_values,
        U_fields=U_fields,
        Fx_fields=Fx_fields,
        Fy_fields=Fy_fields,
        per_step_metrics=per_step_metrics,
        runtime_s=elapsed,
    )


# =============================================================================
# METRICS COMPUTATION
# =============================================================================

def compute_summary_metrics(result: TrajectoryResult, cfg: CompareConfig) -> Dict[str, float]:
    """Compute summary metrics for a trajectory."""
    path = cfg.get_path()
    
    e_perp_vals = [path.metrics(x, y)['e_perp'] for x, y in result.positions]
    
    cumulative_progress = 0.0
    for t in range(len(result.forces)):
        x, y = result.positions[t]
        Fx, Fy = result.forces[t]
        metrics = path.metrics(x, y)
        t_hat_x, t_hat_y = metrics['t_hat']
        cumulative_progress += Fx * t_hat_x + Fy * t_hat_y
    
    U_vals = result.U_values
    
    control_variation = 0.0
    for t in range(1, len(result.applied_controls)):
        v_t, phi_t = result.applied_controls[t]
        v_tm1, phi_tm1 = result.applied_controls[t - 1]
        control_variation += np.sqrt((v_t - v_tm1)**2 + (phi_t - phi_tm1)**2)
    
    # Total distance traveled
    total_distance = 0.0
    for t in range(1, len(result.positions)):
        x1, y1 = result.positions[t-1]
        x2, y2 = result.positions[t]
        total_distance += np.sqrt((x2-x1)**2 + (y2-y1)**2)
    
    return {
        'mean_e_perp': float(np.mean(np.abs(e_perp_vals))),
        'max_e_perp': float(np.max(np.abs(e_perp_vals))),
        'std_e_perp': float(np.std(e_perp_vals)),
        'cumulative_progress': float(cumulative_progress),
        'total_distance_mm': float(total_distance * 1e3),
        'mean_U': float(np.mean(U_vals)),
        'control_variation': float(control_variation),
        'runtime_s': float(result.runtime_s),
        'ms_per_step': float(result.runtime_s / len(result.forces) * 1000),
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_comparison_gif(
    out_path: Path,
    greedy_result: TrajectoryResult,
    mpc_result: TrajectoryResult,
    ev: Evaluator4Pucks,
    cfg: CompareConfig,
    show_controls: bool = False,
):
    """Create side-by-side comparison GIF."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    import imageio.v2 as imageio
    
    frames = []
    temp_dir = out_path.parent / f"_temp_{out_path.stem}"
    temp_dir.mkdir(exist_ok=True)
    
    x_mm = ev.op.x * 1e3
    y_mm = ev.op.y * 1e3
    X, Y = np.meshgrid(x_mm, y_mm)
    
    path = cfg.get_path()
    T = cfg.T
    
    for t in range(T):
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        for ax, result, title in [(axes[0], greedy_result, "GREEDY"), (axes[1], mpc_result, "MPC")]:
            U = result.U_fields[t]
            
            # Use actual U range (not normalized) for realistic contours
            U_min, U_max = np.min(U), np.max(U)
            if U_max - U_min > 1e-25:
                levels = np.linspace(U_min, U_max, 25)
                contourf = ax.contourf(X, Y, U, levels=levels, cmap="RdBu_r", alpha=0.85)
                ax.contour(X, Y, U, levels=levels[::2], colors="k", linewidths=0.3, alpha=0.3)
                cbar = fig.colorbar(contourf, ax=ax)
                cbar.set_label("U (J)")
            
            # Path
            circle = Circle((cfg.cx * 1e3, cfg.cy * 1e3), cfg.R * 1e3,
                             fill=False, edgecolor='lime', linewidth=2, linestyle='--')
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
            metrics = path.metrics(x_t, y_t)
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
            e_perp = metrics['e_perp'] * 1e6
            
            # Cumulative progress
            cum_prog = sum(
                result.forces[i][0] * path.metrics(result.positions[i][0], result.positions[i][1])['t_hat'][0] +
                result.forces[i][1] * path.metrics(result.positions[i][0], result.positions[i][1])['t_hat'][1]
                for i in range(t + 1)
            )
            
            title_str = f"{title}: Step {t}/{T-1}\n"
            title_str += f"|F|: {F_mag:.2e} N, e_perp: {e_perp:.1f} µm"
            if show_controls:
                title_str += f"\nv={v:.3f}, φ={phi:.2f}"
            ax.set_title(title_str)
        
        fig.suptitle(f"Greedy vs MPC Path Tracking - Step {t}", fontsize=14, fontweight='bold')
        fig.tight_layout()
        
        frame_path = temp_dir / f"frame_{t:04d}.png"
        fig.savefig(frame_path, dpi=100)
        plt.close(fig)
        
        frames.append(imageio.imread(frame_path))
    
    imageio.mimsave(out_path, frames, duration=0.15)
    
    for f in temp_dir.glob("*.png"):
        f.unlink()
    temp_dir.rmdir()
    
    print(f"   Saved: {out_path}")


def create_comparison_plot(
    out_path: Path,
    greedy_result: TrajectoryResult,
    mpc_result: TrajectoryResult,
    cfg: CompareConfig,
):
    """Create stacked comparison plot."""
    import matplotlib.pyplot as plt
    
    path = cfg.get_path()
    T = cfg.T
    t_arr = np.arange(T)
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    
    # 1. Lateral error over time
    ax1 = axes[0]
    greedy_e_perp = [path.metrics(x, y)['e_perp'] * 1e6 for x, y in greedy_result.positions[:-1]]
    mpc_e_perp = [path.metrics(x, y)['e_perp'] * 1e6 for x, y in mpc_result.positions[:-1]]
    
    ax1.plot(t_arr, greedy_e_perp, 'b-', linewidth=2, label=f'Greedy (mean: {np.mean(np.abs(greedy_e_perp)):.1f} µm)')
    ax1.plot(t_arr, mpc_e_perp, 'r-', linewidth=2, label=f'MPC (mean: {np.mean(np.abs(mpc_e_perp)):.1f} µm)')
    ax1.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax1.set_ylabel('Lateral Error (µm)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Path Tracking: Lateral Error')
    
    # 2. Cumulative progress over time
    ax2 = axes[1]
    
    def cumulative_progress(result):
        cum = [0.0]
        for t in range(len(result.forces)):
            x, y = result.positions[t]
            Fx, Fy = result.forces[t]
            metrics = path.metrics(x, y)
            t_hat = metrics['t_hat']
            cum.append(cum[-1] + Fx * t_hat[0] + Fy * t_hat[1])
        return cum[1:]
    
    greedy_prog = cumulative_progress(greedy_result)
    mpc_prog = cumulative_progress(mpc_result)
    
    ax2.plot(t_arr, greedy_prog, 'b-', linewidth=2, label=f'Greedy (total: {greedy_prog[-1]:.2e})')
    ax2.plot(t_arr, mpc_prog, 'r-', linewidth=2, label=f'MPC (total: {mpc_prog[-1]:.2e})')
    ax2.set_ylabel('Cumulative Progress')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Path Progress')
    
    # 3. Potential U over time
    ax3 = axes[2]
    ax3.plot(t_arr, greedy_result.U_values, 'b-', linewidth=2, label=f'Greedy (mean: {np.mean(greedy_result.U_values):.2e})')
    ax3.plot(t_arr, mpc_result.U_values, 'r-', linewidth=2, label=f'MPC (mean: {np.mean(mpc_result.U_values):.2e})')
    ax3.set_ylabel('Gor\'kov Potential U')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Trapping Potential')
    
    # 4. Force magnitude
    ax4 = axes[3]
    greedy_F_mag = [np.sqrt(fx**2 + fy**2) for fx, fy in greedy_result.forces]
    mpc_F_mag = [np.sqrt(fx**2 + fy**2) for fx, fy in mpc_result.forces]
    ax4.plot(t_arr, greedy_F_mag, 'b-', linewidth=2, label=f'Greedy (mean: {np.mean(greedy_F_mag):.2e} N)')
    ax4.plot(t_arr, mpc_F_mag, 'r-', linewidth=2, label=f'MPC (mean: {np.mean(mpc_F_mag):.2e} N)')
    ax4.set_ylabel('Force Magnitude (N)')
    ax4.set_xlabel('Step')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_title('Force Magnitude')
    
    fig.suptitle('Greedy vs MPC: Side-by-Side Comparison', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    
    print(f"   Saved: {out_path}")


# =============================================================================
# OUTPUT SAVING
# =============================================================================

def save_trajectory_csv(out_path: Path, result: TrajectoryResult, cfg: CompareConfig):
    """Save trajectory data to CSV."""
    path = cfg.get_path()
    
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'x', 'y', 'U', 'Fx', 'Fy', 'F_mag', 'e_perp', 'progress', 'cumulative_S'])
        
        cumulative_S = 0.0
        for t in range(len(result.forces)):
            x, y = result.positions[t]
            Fx, Fy = result.forces[t]
            U = result.U_values[t]
            F_mag = np.sqrt(Fx**2 + Fy**2)
            metrics = path.metrics(x, y)
            e_perp = metrics['e_perp']
            t_hat = metrics['t_hat']
            progress = Fx * t_hat[0] + Fy * t_hat[1]
            cumulative_S += progress
            
            writer.writerow([t, x, y, U, Fx, Fy, F_mag, e_perp, progress, cumulative_S])
    
    print(f"   Saved: {out_path}")


def save_controls_csv(out_path: Path, result: TrajectoryResult):
    """Save control data to CSV."""
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'v', 'phi'])
        
        for t, (v, phi) in enumerate(result.applied_controls):
            writer.writerow([t, v, phi])
    
    print(f"   Saved: {out_path}")


def save_summary_metrics(out_path: Path, greedy_metrics: Dict, mpc_metrics: Dict):
    """Save summary metrics comparison CSV."""
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'greedy', 'mpc', 'mpc_better'])
        
        for key in greedy_metrics:
            g_val = greedy_metrics[key]
            m_val = mpc_metrics[key]
            
            if key in ['mean_e_perp', 'max_e_perp', 'std_e_perp', 'mean_U', 'control_variation', 'runtime_s', 'ms_per_step']:
                better = m_val < g_val
            else:
                better = m_val > g_val
            
            writer.writerow([key, g_val, m_val, better])
    
    print(f"   Saved: {out_path}")


def save_comparison_json(out_path: Path, greedy_metrics: Dict, mpc_metrics: Dict, cfg: CompareConfig):
    """Save JSON summary."""
    def total_J(metrics: Dict) -> float:
        return (cfg.w_perp * metrics['mean_e_perp']**2 * cfg.T +
                cfg.w_U * metrics['mean_U'] * cfg.T -
                cfg.w_prog * metrics['cumulative_progress'])
    
    summary = {
        'config': {
            'T': cfg.T,
            'K': cfg.K,
            'n_iters': cfg.n_iters,
            'w_perp': cfg.w_perp,
            'w_prog': cfg.w_prog,
            'w_U': cfg.w_U,
            'w_du': cfg.w_du,
            'progress_mode': cfg.progress_mode,
            'Nx': cfg.Nx,
            'Ny': cfg.Ny,
            'alpha_g': cfg.alpha_g,
            'dt': cfg.dt,
        },
        'greedy': {
            'metrics': greedy_metrics,
            'J_approx': total_J(greedy_metrics),
        },
        'mpc': {
            'metrics': mpc_metrics,
            'J_approx': total_J(mpc_metrics),
        },
        'comparison': {
            'mpc_lower_error': mpc_metrics['mean_e_perp'] < greedy_metrics['mean_e_perp'],
            'mpc_higher_progress': mpc_metrics['cumulative_progress'] > greedy_metrics['cumulative_progress'],
            'mpc_lower_U': mpc_metrics['mean_U'] < greedy_metrics['mean_U'],
            'mpc_better_overall': total_J(mpc_metrics) < total_J(greedy_metrics),
        },
    }
    
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"   Saved: {out_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Compare Greedy vs MPC path tracking")
    
    # Preset
    parser.add_argument("--preset", type=str, default="4puck_demo",
                        choices=["4puck_demo", "simple_single"],
                        help="Physics preset (4puck_demo matches working demo)")
    
    # Grid
    parser.add_argument("--Nx", type=int, default=None)
    parser.add_argument("--Ny", type=int, default=None)
    
    # MPC parameters
    parser.add_argument("--K", type=int, default=5, help="MPC horizon length")
    parser.add_argument("--T", type=int, default=50, help="Total executed steps")
    parser.add_argument("--n_iters", type=int, default=8, help="Optimization iterations per MPC solve")
    
    # Weights
    parser.add_argument("--w_perp", type=float, default=1.0e12, help="Lateral error weight")
    parser.add_argument("--w_prog", type=float, default=1.0e-6, help="Progress reward weight")
    parser.add_argument("--w_U", type=float, default=1.0, help="Gor'kov potential weight")
    parser.add_argument("--w_du", type=float, default=0.0, help="Control smoothness weight")
    
    # Progress mode
    parser.add_argument("--progress_mode", type=str, default="force", choices=["force", "displacement"])
    
    # Fast mode
    parser.add_argument("--fast", action="store_true", help="Use smaller grid and fewer steps")
    
    # Output
    parser.add_argument("--output_dir", type=str, default=None)
    
    args = parser.parse_args()
    
    # Create config based on preset
    if args.preset == "4puck_demo":
        # Matches 4puck_demo_surf_greedy.py exactly
        cfg = CompareConfig(
            Nx=80 if not args.fast else 64,
            Ny=80 if not args.fast else 64,
            f=2.0e6,
            alpha_g=2e3,
            dt=5e-3,
            sigma_x=0.10e-3,
            sigma_y=0.15e-3,
            K=args.K if not args.fast else 3,
            T=args.T if not args.fast else 30,
            n_iters=args.n_iters if not args.fast else 6,
            w_perp=args.w_perp, w_prog=args.w_prog, w_U=args.w_U, w_du=args.w_du,
            progress_mode=args.progress_mode,
        )
    else:  # simple_single
        cfg = CompareConfig(
            Nx=64 if not args.fast else 32,
            Ny=64 if not args.fast else 32,
            f=1.0e6,
            alpha_g=1.0,  # No scaling
            dt=0.05,
            sigma_x=0.3e-3,
            sigma_y=0.3e-3,
            K=args.K if not args.fast else 3,
            T=args.T if not args.fast else 20,
            n_iters=args.n_iters if not args.fast else 6,
            w_perp=args.w_perp, w_prog=args.w_prog, w_U=args.w_U, w_du=args.w_du,
            progress_mode=args.progress_mode,
        )
    
    # Override Nx/Ny if specified
    if args.Nx is not None:
        cfg.Nx = args.Nx
    if args.Ny is not None:
        cfg.Ny = args.Ny
    
    print("\n" + "="*70)
    print("  Greedy vs MPC Path Tracking Comparison")
    print("="*70)
    
    if args.fast:
        print("   [FAST MODE]")
    
    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = project_root / "results" / "adjoint_path_track_mpc_compare" / f"run_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n   Output: {out_dir}")
    print(f"   Preset: {args.preset}")
    print(f"   Config: K={cfg.K}, T={cfg.T}, n_iters={cfg.n_iters}")
    print(f"   Physics: f={cfg.f/1e6:.1f} MHz, alpha_g={cfg.alpha_g:.0e}, dt={cfg.dt*1e3:.1f} ms")
    print(f"   Weights: w_perp={cfg.w_perp:.0e}, w_prog={cfg.w_prog:.0e}, w_U={cfg.w_U}, w_du={cfg.w_du}")
    
    # Build evaluator (same as working demo!)
    print("\n   Building Evaluator4Pucks (same as working demo)...")
    ev, particle = create_evaluator(cfg)
    
    print(f"   Grid: {cfg.Nx} x {cfg.Ny}")
    print(f"   Transducer sigma: ({cfg.sigma_x*1e6:.0f}, {cfg.sigma_y*1e6:.0f}) µm")
    
    # Initial position (on circle)
    x0 = cfg.cx + cfg.R * np.cos(cfg.theta0)
    y0 = cfg.cy + cfg.R * np.sin(cfg.theta0)
    v_init = cfg.v_ref_ctrl
    phi_init = cfg.phi_ref
    
    print(f"\n   Initial position: ({x0*1e3:.3f}, {y0*1e3:.3f}) mm")
    print(f"   Circle: center=({cfg.cx*1e3:.2f}, {cfg.cy*1e3:.2f}) mm, R={cfg.R*1e3:.2f} mm")
    
    # Run greedy
    print("\n" + "-"*50)
    print("  Running GREEDY Controller")
    print("-"*50)
    greedy_result = run_greedy(ev, x0, y0, v_init, phi_init, cfg)
    
    # Run MPC
    print("\n" + "-"*50)
    print("  Running MPC Controller")
    print("-"*50)
    mpc_result = run_mpc(ev, x0, y0, v_init, phi_init, cfg)
    
    # Compute metrics
    print("\n" + "-"*50)
    print("  Computing Metrics")
    print("-"*50)
    greedy_metrics = compute_summary_metrics(greedy_result, cfg)
    mpc_metrics = compute_summary_metrics(mpc_result, cfg)
    
    print(f"\n   GREEDY:")
    print(f"      Mean |e_perp|:    {greedy_metrics['mean_e_perp']*1e6:.2f} µm")
    print(f"      Max |e_perp|:     {greedy_metrics['max_e_perp']*1e6:.2f} µm")
    print(f"      Total distance:   {greedy_metrics['total_distance_mm']:.3f} mm")
    print(f"      Progress:         {greedy_metrics['cumulative_progress']:.3e}")
    print(f"      Mean U:           {greedy_metrics['mean_U']:.3e}")
    print(f"      Runtime:          {greedy_metrics['runtime_s']:.1f}s ({greedy_metrics['ms_per_step']:.1f} ms/step)")
    
    print(f"\n   MPC:")
    print(f"      Mean |e_perp|:    {mpc_metrics['mean_e_perp']*1e6:.2f} µm")
    print(f"      Max |e_perp|:     {mpc_metrics['max_e_perp']*1e6:.2f} µm")
    print(f"      Total distance:   {mpc_metrics['total_distance_mm']:.3f} mm")
    print(f"      Progress:         {mpc_metrics['cumulative_progress']:.3e}")
    print(f"      Mean U:           {mpc_metrics['mean_U']:.3e}")
    print(f"      Runtime:          {mpc_metrics['runtime_s']:.1f}s ({mpc_metrics['ms_per_step']:.1f} ms/step)")
    
    # Comparison
    print(f"\n   COMPARISON:")
    print(f"      MPC lower error:     {mpc_metrics['mean_e_perp'] < greedy_metrics['mean_e_perp']}")
    print(f"      MPC higher progress: {mpc_metrics['cumulative_progress'] > greedy_metrics['cumulative_progress']}")
    print(f"      MPC lower U:         {mpc_metrics['mean_U'] < greedy_metrics['mean_U']}")
    
    # Save outputs
    print("\n" + "-"*50)
    print("  Saving Outputs")
    print("-"*50)
    
    save_trajectory_csv(out_dir / "trajectory_greedy.csv", greedy_result, cfg)
    save_trajectory_csv(out_dir / "trajectory_mpc.csv", mpc_result, cfg)
    save_controls_csv(out_dir / "controls_greedy.csv", greedy_result)
    save_controls_csv(out_dir / "controls_mpc.csv", mpc_result)
    save_summary_metrics(out_dir / "summary_metrics.csv", greedy_metrics, mpc_metrics)
    save_comparison_json(out_dir / "compare_summary.json", greedy_metrics, mpc_metrics, cfg)
    
    # Create visualizations
    print("\n   Creating GIFs...")
    create_comparison_gif(out_dir / "gorkov_contour_compare.gif", greedy_result, mpc_result, ev, cfg, show_controls=False)
    create_comparison_gif(out_dir / "gorkov_contour_compare_controls.gif", greedy_result, mpc_result, ev, cfg, show_controls=True)
    
    print("\n   Creating comparison plot...")
    create_comparison_plot(out_dir / "compare_plot.png", greedy_result, mpc_result, cfg)
    
    print("\n" + "="*70)
    print("  COMPLETE")
    print("="*70)
    print(f"\n   All outputs saved to: {out_dir}")
    
    # Final verdict
    if mpc_metrics['mean_e_perp'] < greedy_metrics['mean_e_perp']:
        improvement = (1 - mpc_metrics['mean_e_perp'] / greedy_metrics['mean_e_perp']) * 100
        print(f"\n   ✓ MPC achieved {improvement:.1f}% lower mean tracking error than Greedy!")
    else:
        print(f"\n   Note: Greedy achieved lower tracking error in this run.")
    
    if greedy_metrics['total_distance_mm'] > 0.01 or mpc_metrics['total_distance_mm'] > 0.01:
        print(f"   ✓ Particle moved significantly (Greedy: {greedy_metrics['total_distance_mm']:.3f} mm, MPC: {mpc_metrics['total_distance_mm']:.3f} mm)")
    else:
        print(f"   ⚠ Warning: Particle barely moved - check physics settings")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
