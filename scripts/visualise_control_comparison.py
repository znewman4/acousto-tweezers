#!/usr/bin/env python3
"""
visualise_control_comparison.py - Visual comparison of control strategies.

Creates compelling GIF and PNG deliverables showing how:
  - Greedy 1-step control
  - Adjoint 1-step control  
  - Adjoint K-step lookahead control

differ in behavior. The visuals communicate:
  "Repositioning now helps later."

Deliverables:
  1. Side-by-side comparison GIF (3 panels, synchronized)
  2. Hero GIF for K-step controller only
  3. Static snapshot PNG at t=20
  4. Quantitative U vs time plot

Usage:
    python scripts/visualise_control_comparison.py
    python scripts/visualise_control_comparison.py --fast --steps 30
    python scripts/visualise_control_comparison.py --K 10 --steps 50
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime
import json

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
class VisConfig:
    """Configuration for visualization."""
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
    
    # Particle
    particle_a: float = 50.0e-6     # 50 µm radius
    particle_rho_p: float = 1050.0  # kg/m³
    particle_c_p: float = 2350.0    # m/s
    
    # Dynamics
    mu: float = 1.0e-3  # dynamic viscosity [Pa·s]
    dt: float = 0.05    # time step [s]
    
    # Control bounds
    v_min: float = 0.01
    v_max: float = 0.2
    
    # K-step horizon
    K: int = 10
    n_iters: int = 10
    alphas: Tuple[float, ...] = (0.0, 0.01, 0.1, 0.3, 1.0)
    
    # Simulation
    n_steps: int = 50
    
    # Initial state
    x0_frac: float = 0.35
    y0_frac: float = 0.5
    
    # Visualization
    trail_length: int = 50
    force_scale: float = 5e11  # Scale force arrows


# =============================================================================
# Physics helpers (reused from adjoint_steer_kstep.py)
# =============================================================================

def build_vb_from_control(v: float, phi: float, x_trans: float, y_trans: float,
                          sigma_x: float, sigma_y: float, x_grid: np.ndarray) -> np.ndarray:
    """Build bottom boundary velocity from control parameters."""
    G_x = np.exp(-(x_grid - x_trans)**2 / (2.0 * sigma_x**2))
    G_y = np.exp(-y_trans**2 / (2.0 * sigma_y**2))
    return v * np.exp(1j * phi) * G_x * G_y


def compute_field_U_F(op, v: float, phi: float, cfg: VisConfig, particle: ParticleProps):
    """Compute full field, potential, and force arrays."""
    x_trans = cfg.Lx * 0.5
    y_trans = 0.02 * cfg.Ly
    vb = build_vb_from_control(v, phi, x_trans, y_trans, cfg.sigma_x, cfg.sigma_y, op.x)
    field = op.solve_for_bottom_vb(vb)
    U, Fx, Fy = gorkov_potential_and_force_2d(field, particle)
    return field, U, Fx, Fy


def interp_at_pos(arr: np.ndarray, x_p: float, y_p: float, op, cfg: VisConfig) -> float:
    """Bilinear interpolation of array at position."""
    ix_f = (x_p - op.x[0]) / op.dx
    iy_f = (y_p - op.y[0]) / op.dy
    
    ix0 = int(np.clip(np.floor(ix_f), 0, op.Nx - 2))
    iy0 = int(np.clip(np.floor(iy_f), 0, op.Ny - 2))
    ix1, iy1 = ix0 + 1, iy0 + 1
    
    fx = np.clip(ix_f - ix0, 0, 1)
    fy = np.clip(iy_f - iy0, 0, 1)
    
    return (arr[iy0, ix0] * (1-fx) * (1-fy) + arr[iy0, ix1] * fx * (1-fy) +
            arr[iy1, ix0] * (1-fx) * fy + arr[iy1, ix1] * fx * fy)


def compute_U_and_F_at_pos(op, v: float, phi: float, cfg: VisConfig,
                           x_p: float, y_p: float, particle: ParticleProps):
    """Compute potential and force at position."""
    _, U, Fx, Fy = compute_field_U_F(op, v, phi, cfg, particle)
    U_interp = interp_at_pos(U, x_p, y_p, op, cfg)
    Fx_interp = interp_at_pos(Fx, x_p, y_p, op, cfg)
    Fy_interp = interp_at_pos(Fy, x_p, y_p, op, cfg)
    return U_interp, Fx_interp, Fy_interp


def overdamped_step(x: float, y: float, Fx: float, Fy: float, cfg: VisConfig) -> Tuple[float, float]:
    """One overdamped particle step."""
    gamma = 6.0 * np.pi * cfg.mu * cfg.particle_a
    x_new = np.clip(x + cfg.dt * Fx / gamma, 0, cfg.Lx)
    y_new = np.clip(y + cfg.dt * Fy / gamma, 0, cfg.Ly)
    return x_new, y_new


def compute_adjoint_gradient(op, v: float, phi: float, x_p: float, y_p: float,
                              cfg: VisConfig, particle: ParticleProps) -> Tuple[float, float]:
    """Compute ∂U/∂(v, phi) via adjoint with bilinear interpolation."""
    # Bilinear interpolation indices and weights
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
    
    # Forward solve
    x_trans = cfg.Lx * 0.5
    y_trans = 0.02 * cfg.Ly
    vb = build_vb_from_control(v, phi, x_trans, y_trans, cfg.sigma_x, cfg.sigma_y, op.x)
    field = op.solve_for_bottom_vb(vb)
    
    # Accumulate adjoint seed from all 4 corners
    dJ_dp = np.zeros(cfg.Nx * cfg.Ny, dtype=np.complex128)
    
    for (ix, iy, w) in [(ix0, iy0, w00), (ix1, iy0, w10), (ix0, iy1, w01), (ix1, iy1, w11)]:
        if w > 1e-12:
            dU_k_dp = compute_dJdp_gorkov_potential(
                ix, iy, cfg.Nx, cfg.Ny, op.dx, op.dy,
                field.p, op.omega, cfg.rho0, cfg.c0,
                cfg.particle_a, cfg.particle_rho_p, cfg.particle_c_p,
            )
            dJ_dp += w * dU_k_dp
    
    # Transducer params for db/du
    trans = TransducerParams(x=x_trans, y=y_trans, v=v, phi=phi,
                             sigma_x=cfg.sigma_x, sigma_y=cfg.sigma_y, gate=True)
    db_dv, db_dphi = compute_dbdu_single_transducer(
        trans, op.x, op.omega, op.rho0, op.coupling_alpha, op.bottom_rows, cfg.Nx, cfg.Ny
    )
    
    grads = adjoint_gradient_vectorized(op.adjoint_solve, dJ_dp, [db_dv, db_dphi])
    return grads[0], grads[1]


# =============================================================================
# Controllers
# =============================================================================

def clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def normalize_phi(phi: float) -> float:
    return ((phi + np.pi) % (2 * np.pi)) - np.pi


def run_greedy_controller(op, cfg: VisConfig, particle: ParticleProps,
                           x0: float, y0: float, v_init: float, phi_init: float,
                           n_steps: int) -> Dict[str, Any]:
    """
    Greedy 1-step: At each step, pick control that minimizes U(x_t; u_t).
    No gradient, just line search over alphas from current control.
    """
    positions = [(x0, y0)]
    controls = []
    U_values = []
    F_values = []
    U_fields = []
    
    x, y = x0, y0
    v, phi = v_init, phi_init
    
    for t in range(n_steps):
        # Line search over controls
        best_U = float('inf')
        best_v, best_phi = v, phi
        
        for dv in [-0.02, -0.01, 0, 0.01, 0.02]:
            for dphi in [-0.2, -0.1, 0, 0.1, 0.2]:
                v_try = clamp(v + dv, cfg.v_min, cfg.v_max)
                phi_try = normalize_phi(phi + dphi)
                U_try, _, _ = compute_U_and_F_at_pos(op, v_try, phi_try, cfg, x, y, particle)
                if U_try < best_U:
                    best_U = U_try
                    best_v, best_phi = v_try, phi_try
        
        v, phi = best_v, best_phi
        controls.append((v, phi))
        
        # Compute state for recording
        _, U_field, Fx_field, Fy_field = compute_field_U_F(op, v, phi, cfg, particle)
        U = interp_at_pos(U_field, x, y, op, cfg)
        Fx = interp_at_pos(Fx_field, x, y, op, cfg)
        Fy = interp_at_pos(Fy_field, x, y, op, cfg)
        
        U_values.append(U)
        F_values.append((Fx, Fy))
        U_fields.append(U_field.copy())
        
        # Step dynamics
        x, y = overdamped_step(x, y, Fx, Fy, cfg)
        positions.append((x, y))
    
    return {
        'positions': positions,
        'controls': controls,
        'U_values': U_values,
        'F_values': F_values,
        'U_fields': U_fields,
        'method': 'Greedy 1-step',
    }


def run_adjoint_1step_controller(op, cfg: VisConfig, particle: ParticleProps,
                                  x0: float, y0: float, v_init: float, phi_init: float,
                                  n_steps: int) -> Dict[str, Any]:
    """
    Adjoint 1-step: At each step, use gradient ∂U/∂u to update control.
    """
    positions = [(x0, y0)]
    controls = []
    U_values = []
    F_values = []
    U_fields = []
    
    x, y = x0, y0
    v, phi = v_init, phi_init
    
    for t in range(n_steps):
        # Compute adjoint gradient
        dU_dv, dU_dphi = compute_adjoint_gradient(op, v, phi, x, y, cfg, particle)
        
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
            U_try, _, _ = compute_U_and_F_at_pos(op, v_try, phi_try, cfg, x, y, particle)
            if U_try < best_U:
                best_U = U_try
                best_v, best_phi = v_try, phi_try
        
        v, phi = best_v, best_phi
        controls.append((v, phi))
        
        # Compute state for recording
        _, U_field, Fx_field, Fy_field = compute_field_U_F(op, v, phi, cfg, particle)
        U = interp_at_pos(U_field, x, y, op, cfg)
        Fx = interp_at_pos(Fx_field, x, y, op, cfg)
        Fy = interp_at_pos(Fy_field, x, y, op, cfg)
        
        U_values.append(U)
        F_values.append((Fx, Fy))
        U_fields.append(U_field.copy())
        
        # Step dynamics
        x, y = overdamped_step(x, y, Fx, Fy, cfg)
        positions.append((x, y))
    
    return {
        'positions': positions,
        'controls': controls,
        'U_values': U_values,
        'F_values': F_values,
        'U_fields': U_fields,
        'method': 'Adjoint 1-step',
    }


def run_kstep_controller(op, cfg: VisConfig, particle: ParticleProps,
                          x0: float, y0: float, v_init: float, phi_init: float,
                          n_steps: int) -> Dict[str, Any]:
    """
    K-step lookahead: Optimize control sequence over K-step horizon using
    discrete-time adjoint backpropagation.
    """
    from acousto.adjoint.trajectory import compute_trajectory_gradient
    
    positions = [(x0, y0)]
    controls = []
    U_values = []
    F_values = []
    U_fields = []
    
    x, y = x0, y0
    
    # Precompute mobility
    gamma = 6.0 * np.pi * cfg.mu * cfg.particle_a
    mobility = 1.0 / gamma
    
    # Wrapper functions for trajectory module
    def compute_force_fn(v, phi, px, py):
        _, U_field, Fx_field, Fy_field = compute_field_U_F(op, v, phi, cfg, particle)
        U = interp_at_pos(U_field, px, py, op, cfg)
        Fx = interp_at_pos(Fx_field, px, py, op, cfg)
        Fy = interp_at_pos(Fy_field, px, py, op, cfg)
        return None, U, Fx, Fy
    
    def compute_dU_du_fn(v, phi, px, py):
        return compute_adjoint_gradient(op, v, phi, px, py, cfg, particle)
    
    # Rolling horizon control
    control_buffer = [(v_init, phi_init)] * cfg.K
    
    for t in range(n_steps):
        # Optimize K-step horizon from current state
        horizon_controls = list(control_buffer)
        
        for opt_iter in range(cfg.n_iters):
            # Compute gradients via discrete adjoint
            gradients, state = compute_trajectory_gradient(
                controls=horizon_controls,
                x0=x, y0=y,
                compute_force_fn=compute_force_fn,
                compute_dU_du_fn=compute_dU_du_fn,
                dt=cfg.dt,
                mobility=mobility,
                x_bounds=(0, cfg.Lx),
                y_bounds=(0, cfg.Ly),
                beta_terminal=0.0,
            )
            
            # Line search
            best_J = float('inf')
            best_controls = horizon_controls
            
            for alpha in cfg.alphas:
                trial_controls = []
                for k, (v_k, phi_k) in enumerate(horizon_controls):
                    grad_v, grad_phi = gradients[k]
                    
                    if abs(grad_v) > 1e-30:
                        scale_v = 0.1 * v_k / abs(grad_v)
                    else:
                        scale_v = 0.0
                    if abs(grad_phi) > 1e-30:
                        scale_phi = 0.1 / abs(grad_phi)
                    else:
                        scale_phi = 0.0
                    
                    v_new = clamp(v_k - alpha * scale_v * grad_v, cfg.v_min, cfg.v_max)
                    phi_new = normalize_phi(phi_k - alpha * scale_phi * grad_phi)
                    trial_controls.append((v_new, phi_new))
                
                # Evaluate trajectory objective
                J = 0.0
                px, py = x, y
                for v_k, phi_k in trial_controls:
                    U_k, Fx_k, Fy_k = compute_U_and_F_at_pos(op, v_k, phi_k, cfg, px, py, particle)
                    J += U_k
                    px, py = overdamped_step(px, py, Fx_k, Fy_k, cfg)
                
                if J < best_J:
                    best_J = J
                    best_controls = trial_controls
            
            horizon_controls = best_controls
        
        # Apply first control from optimized sequence
        v, phi = horizon_controls[0]
        controls.append((v, phi))
        
        # Shift buffer for warm start
        control_buffer = horizon_controls[1:] + [horizon_controls[-1]]
        
        # Compute state for recording
        _, U_field, Fx_field, Fy_field = compute_field_U_F(op, v, phi, cfg, particle)
        U = interp_at_pos(U_field, x, y, op, cfg)
        Fx = interp_at_pos(Fx_field, x, y, op, cfg)
        Fy = interp_at_pos(Fy_field, x, y, op, cfg)
        
        U_values.append(U)
        F_values.append((Fx, Fy))
        U_fields.append(U_field.copy())
        
        # Step dynamics
        x, y = overdamped_step(x, y, Fx, Fy, cfg)
        positions.append((x, y))
    
    return {
        'positions': positions,
        'controls': controls,
        'U_values': U_values,
        'F_values': F_values,
        'U_fields': U_fields,
        'method': f'Adjoint K-step (K={cfg.K})',
    }


# =============================================================================
# Visualization Functions
# =============================================================================

def compute_global_U_limits(results_list: List[Dict]) -> Tuple[float, float]:
    """Compute global U field limits across all methods and timesteps."""
    all_U = []
    for res in results_list:
        for U_field in res['U_fields']:
            all_U.append(U_field)
    
    all_U = np.array(all_U)
    vmin = np.percentile(all_U, 2)
    vmax = np.percentile(all_U, 98)
    return vmin, vmax


def create_comparison_gif(results_list: List[Dict], cfg: VisConfig, op,
                           output_path: Path, fps: int = 10):
    """Create side-by-side comparison GIF with 3 panels."""
    n_frames = len(results_list[0]['U_fields'])
    
    # Global color limits
    vmin, vmax = compute_global_U_limits(results_list)
    
    # Grid coordinates in mm
    x_mm = op.x * 1e3
    y_mm = op.y * 1e3
    X, Y = np.meshgrid(x_mm, y_mm)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.subplots_adjust(left=0.05, right=0.92, bottom=0.1, top=0.9, wspace=0.15)
    
    # Colorbar axis
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    norm = Normalize(vmin=vmin * 1e15, vmax=vmax * 1e15)
    sm = ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Gor\'kov Potential U [fJ]', fontsize=10)
    
    def animate(frame_idx):
        for ax in axes:
            ax.clear()
        
        for i, (ax, res) in enumerate(zip(axes, results_list)):
            U_field = res['U_fields'][frame_idx]
            positions = res['positions']
            F_values = res['F_values']
            U_values = res['U_values']
            method = res['method']
            
            # Current position
            x_p, y_p = positions[frame_idx]
            Fx, Fy = F_values[frame_idx]
            U_current = U_values[frame_idx]
            
            # Contour plot
            cf = ax.contourf(X, Y, U_field * 1e15, levels=30, 
                            cmap='viridis', vmin=vmin * 1e15, vmax=vmax * 1e15)
            ax.contour(X, Y, U_field * 1e15, levels=15, colors='white', 
                      linewidths=0.3, alpha=0.4)
            
            # Trail (faded history)
            trail_start = max(0, frame_idx - cfg.trail_length)
            trail_positions = positions[trail_start:frame_idx + 1]
            
            if len(trail_positions) > 1:
                trail_x = [p[0] * 1e3 for p in trail_positions]
                trail_y = [p[1] * 1e3 for p in trail_positions]
                
                # Gradient color trail
                n_trail = len(trail_x)
                for j in range(n_trail - 1):
                    alpha = 0.2 + 0.8 * (j / n_trail)
                    ax.plot(trail_x[j:j+2], trail_y[j:j+2], 
                           'w-', lw=2, alpha=alpha)
            
            # Particle position
            ax.scatter(x_p * 1e3, y_p * 1e3, s=200, c='red', marker='o',
                      edgecolors='white', linewidths=2, zorder=10)
            
            # Force vector
            F_mag = np.sqrt(Fx**2 + Fy**2)
            if F_mag > 1e-20:
                scale = cfg.force_scale
                ax.arrow(x_p * 1e3, y_p * 1e3, 
                        Fx * scale * 1e3, Fy * scale * 1e3,
                        head_width=0.05, head_length=0.02, 
                        fc='yellow', ec='black', linewidth=1.5, zorder=11)
            
            # Annotations
            ax.text(0.02, 0.98, f'U = {U_current * 1e15:.2f} fJ', 
                   transform=ax.transAxes, fontsize=10, fontweight='bold',
                   verticalalignment='top', color='white',
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
            
            ax.set_xlim(0, cfg.Lx * 1e3)
            ax.set_ylim(0, cfg.Ly * 1e3)
            ax.set_xlabel('x [mm]', fontsize=10)
            if i == 0:
                ax.set_ylabel('y [mm]', fontsize=10)
            ax.set_title(f'{method}', fontsize=12, fontweight='bold')
            ax.set_aspect('equal')
        
        fig.suptitle(f'Step {frame_idx + 1} / {n_frames}', fontsize=14, fontweight='bold')
        return axes
    
    print(f"Creating comparison GIF with {n_frames} frames...")
    anim = FuncAnimation(fig, animate, frames=n_frames, interval=1000//fps, blit=False)
    
    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer)
    plt.close(fig)
    print(f"   Saved: {output_path}")


def create_hero_gif(result: Dict, cfg: VisConfig, op, output_path: Path, fps: int = 5):
    """Create standalone hero GIF for K-step controller."""
    n_frames = len(result['U_fields'])
    
    # Color limits from this result only
    all_U = np.array(result['U_fields'])
    vmin = np.percentile(all_U, 2)
    vmax = np.percentile(all_U, 98)
    
    # Grid coordinates
    x_mm = op.x * 1e3
    y_mm = op.y * 1e3
    X, Y = np.meshgrid(x_mm, y_mm)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.subplots_adjust(left=0.1, right=0.85, bottom=0.1, top=0.92)
    
    # Colorbar axis
    cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])
    norm = Normalize(vmin=vmin * 1e15, vmax=vmax * 1e15)
    sm = ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Gor\'kov Potential U [fJ]', fontsize=12)
    
    def animate(frame_idx):
        ax.clear()
        
        U_field = result['U_fields'][frame_idx]
        positions = result['positions']
        F_values = result['F_values']
        U_values = result['U_values']
        
        x_p, y_p = positions[frame_idx]
        Fx, Fy = F_values[frame_idx]
        U_current = U_values[frame_idx]
        
        # Contour plot
        ax.contourf(X, Y, U_field * 1e15, levels=40, 
                   cmap='viridis', vmin=vmin * 1e15, vmax=vmax * 1e15)
        ax.contour(X, Y, U_field * 1e15, levels=20, colors='white', 
                  linewidths=0.3, alpha=0.5)
        
        # Trail with time-based coloring (early=faint, late=bright)
        trail_positions = positions[:frame_idx + 1]
        
        if len(trail_positions) > 1:
            n_trail = len(trail_positions)
            
            # Create colormap for trail
            for j in range(n_trail - 1):
                t_frac = j / max(1, n_trail - 1)  # 0 to 1
                # Color from light blue (early) to bright cyan (late)
                color = (0, 0.5 + 0.5 * t_frac, 1)
                alpha = 0.3 + 0.7 * t_frac
                lw = 1 + 3 * t_frac
                
                ax.plot([trail_positions[j][0] * 1e3, trail_positions[j+1][0] * 1e3],
                       [trail_positions[j][1] * 1e3, trail_positions[j+1][1] * 1e3],
                       color=color, lw=lw, alpha=alpha)
        
        # Particle
        ax.scatter(x_p * 1e3, y_p * 1e3, s=400, c='red', marker='o',
                  edgecolors='white', linewidths=3, zorder=10)
        
        # Force vector
        F_mag = np.sqrt(Fx**2 + Fy**2)
        if F_mag > 1e-20:
            scale = cfg.force_scale
            ax.arrow(x_p * 1e3, y_p * 1e3, 
                    Fx * scale * 1e3, Fy * scale * 1e3,
                    head_width=0.08, head_length=0.03, 
                    fc='yellow', ec='black', linewidth=2, zorder=11)
        
        # Annotations
        ax.text(0.02, 0.98, f'U = {U_current * 1e15:.2f} fJ', 
               transform=ax.transAxes, fontsize=14, fontweight='bold',
               verticalalignment='top', color='white',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
        
        ax.text(0.02, 0.88, f'Planned horizon K = {cfg.K}', 
               transform=ax.transAxes, fontsize=11,
               verticalalignment='top', color='white',
               bbox=dict(boxstyle='round', facecolor='darkblue', alpha=0.7))
        
        ax.set_xlim(0, cfg.Lx * 1e3)
        ax.set_ylim(0, cfg.Ly * 1e3)
        ax.set_xlabel('x [mm]', fontsize=12)
        ax.set_ylabel('y [mm]', fontsize=12)
        ax.set_title(f'{result["method"]} — Step {frame_idx + 1} / {n_frames}', 
                    fontsize=14, fontweight='bold')
        ax.set_aspect('equal')
        
        return [ax]
    
    print(f"Creating hero GIF with {n_frames} frames...")
    anim = FuncAnimation(fig, animate, frames=n_frames, interval=1000//fps, blit=False)
    
    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer)
    plt.close(fig)
    print(f"   Saved: {output_path}")


def create_snapshot(results_list: List[Dict], cfg: VisConfig, op, 
                    output_path: Path, t_snapshot: int = 20):
    """Create static snapshot at timestep t."""
    if t_snapshot >= len(results_list[0]['U_fields']):
        t_snapshot = len(results_list[0]['U_fields']) - 1
    
    # Global color limits
    vmin, vmax = compute_global_U_limits(results_list)
    
    # Grid coordinates
    x_mm = op.x * 1e3
    y_mm = op.y * 1e3
    X, Y = np.meshgrid(x_mm, y_mm)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.subplots_adjust(left=0.06, right=0.92, bottom=0.12, top=0.88, wspace=0.18)
    
    for i, (ax, res) in enumerate(zip(axes, results_list)):
        U_field = res['U_fields'][t_snapshot]
        positions = res['positions']
        F_values = res['F_values']
        U_values = res['U_values']
        method = res['method']
        
        x_p, y_p = positions[t_snapshot]
        Fx, Fy = F_values[t_snapshot]
        U_current = U_values[t_snapshot]
        
        # Contour plot
        cf = ax.contourf(X, Y, U_field * 1e15, levels=30, 
                        cmap='viridis', vmin=vmin * 1e15, vmax=vmax * 1e15)
        ax.contour(X, Y, U_field * 1e15, levels=15, colors='white', 
                  linewidths=0.3, alpha=0.5)
        
        # Trail
        trail_positions = positions[:t_snapshot + 1]
        if len(trail_positions) > 1:
            trail_x = [p[0] * 1e3 for p in trail_positions]
            trail_y = [p[1] * 1e3 for p in trail_positions]
            ax.plot(trail_x, trail_y, 'w-', lw=2, alpha=0.6)
        
        # Particle
        ax.scatter(x_p * 1e3, y_p * 1e3, s=250, c='red', marker='o',
                  edgecolors='white', linewidths=2, zorder=10)
        
        # Force vector
        F_mag = np.sqrt(Fx**2 + Fy**2)
        if F_mag > 1e-20:
            scale = cfg.force_scale
            ax.arrow(x_p * 1e3, y_p * 1e3, 
                    Fx * scale * 1e3, Fy * scale * 1e3,
                    head_width=0.06, head_length=0.025, 
                    fc='yellow', ec='black', linewidth=1.5, zorder=11)
        
        # U annotation
        ax.text(0.03, 0.97, f'U = {U_current * 1e15:.2f} fJ', 
               transform=ax.transAxes, fontsize=11, fontweight='bold',
               verticalalignment='top', color='white',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
        
        ax.set_xlim(0, cfg.Lx * 1e3)
        ax.set_ylim(0, cfg.Ly * 1e3)
        ax.set_xlabel('x [mm]', fontsize=11)
        if i == 0:
            ax.set_ylabel('y [mm]', fontsize=11)
        ax.set_title(f'{method}', fontsize=12, fontweight='bold')
        ax.set_aspect('equal')
    
    # Shared colorbar
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.68])
    norm = Normalize(vmin=vmin * 1e15, vmax=vmax * 1e15)
    sm = ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Gor\'kov Potential U [fJ]', fontsize=10)
    
    fig.suptitle(f'Control Comparison at t = {t_snapshot}', fontsize=14, fontweight='bold')
    
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"   Saved: {output_path}")


def create_U_vs_time_plot(results_list: List[Dict], cfg: VisConfig, output_path: Path):
    """Create quantitative U vs time plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#d62728', '#2ca02c', '#1f77b4']  # Red, Green, Blue
    linestyles = ['-', '--', '-']
    linewidths = [2, 2, 3]
    
    for res, color, ls, lw in zip(results_list, colors, linestyles, linewidths):
        U_values = np.array(res['U_values']) * 1e15  # Convert to fJ
        t = np.arange(len(U_values))
        ax.plot(t, U_values, color=color, ls=ls, lw=lw, label=res['method'], marker='o', 
               markersize=4, markerfacecolor=color, markeredgecolor='white', markeredgewidth=0.5)
    
    ax.set_xlabel('Time step', fontsize=12)
    ax.set_ylabel('Gor\'kov Potential U [fJ]', fontsize=12)
    ax.set_title('Potential Minimization: Controller Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Annotate final values
    for res, color in zip(results_list, colors):
        U_final = res['U_values'][-1] * 1e15
        ax.axhline(y=U_final, color=color, linestyle=':', alpha=0.5, lw=1)
        ax.text(len(res['U_values']) - 1, U_final, f'  {U_final:.2f}', 
               fontsize=9, color=color, va='center')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"   Saved: {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Visual comparison of control strategies")
    parser.add_argument('--fast', action='store_true', help="Fast mode (coarse grid, fewer steps)")
    parser.add_argument('--K', type=int, default=10, help="K-step horizon (default: 10)")
    parser.add_argument('--steps', type=int, default=50, help="Number of simulation steps (default: 50)")
    args = parser.parse_args()
    
    cfg = VisConfig()
    cfg.K = args.K
    cfg.n_steps = args.steps
    
    if args.fast:
        cfg.Nx = 32
        cfg.Ny = 32
        cfg.n_steps = min(30, args.steps)
        cfg.n_iters = 5
    
    print("=" * 70)
    print("CONTROL STRATEGY VISUALIZATION")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"   Grid: {cfg.Nx}×{cfg.Ny}")
    print(f"   Steps: {cfg.n_steps}")
    print(f"   K-step horizon: {cfg.K}")
    print(f"   dt: {cfg.dt * 1e3:.1f} ms")
    
    # Build operator
    print("\n1. Building Helmholtz operator...")
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
    
    print(f"   Initial position: ({x0*1e3:.3f}, {y0*1e3:.3f}) mm")
    print(f"   Initial control: v={v_init}, φ={phi_init}")
    
    # Run all three controllers
    print("\n2. Running controllers...")
    
    print("   [1/3] Greedy 1-step...")
    result_greedy = run_greedy_controller(op, cfg, particle, x0, y0, v_init, phi_init, cfg.n_steps)
    print(f"         Final U: {result_greedy['U_values'][-1] * 1e15:.3f} fJ")
    
    print("   [2/3] Adjoint 1-step...")
    result_adj1 = run_adjoint_1step_controller(op, cfg, particle, x0, y0, v_init, phi_init, cfg.n_steps)
    print(f"         Final U: {result_adj1['U_values'][-1] * 1e15:.3f} fJ")
    
    print("   [3/3] Adjoint K-step...")
    result_kstep = run_kstep_controller(op, cfg, particle, x0, y0, v_init, phi_init, cfg.n_steps)
    print(f"         Final U: {result_kstep['U_values'][-1] * 1e15:.3f} fJ")
    
    results_list = [result_greedy, result_adj1, result_kstep]
    
    # Create output directory
    output_dir = project_root / "results" / "visual_comparisons"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate all visualizations
    print("\n3. Creating visualizations...")
    
    print("   [1/4] Side-by-side comparison GIF...")
    create_comparison_gif(results_list, cfg, op, 
                          output_dir / "gorkov_compare_greedy_vs_adjoint1_vs_kstep.gif",
                          fps=8)
    
    print("   [2/4] K-step hero GIF...")
    create_hero_gif(result_kstep, cfg, op,
                    output_dir / "gorkov_kstep_trajectory.gif",
                    fps=5)
    
    print("   [3/4] Static snapshot at t=20...")
    create_snapshot(results_list, cfg, op,
                    output_dir / "gorkov_snapshot_t20.png",
                    t_snapshot=20)
    
    print("   [4/4] U vs time plot...")
    create_U_vs_time_plot(results_list, cfg, output_dir / "U_vs_time.png")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    total_U = {res['method']: sum(res['U_values']) for res in results_list}
    baseline = total_U['Greedy 1-step']
    
    print("\n   Total cumulative U (lower is better):")
    for method, J in total_U.items():
        pct = 100 * (J - baseline) / abs(baseline) if baseline != 0 else 0
        sign = '+' if pct >= 0 else ''
        print(f"      {method:25s}: {J * 1e15:10.3f} fJ  ({sign}{pct:.1f}% vs greedy)")
    
    print(f"\n   Deliverables saved to: {output_dir}")
    print("   Files:")
    print("      - gorkov_compare_greedy_vs_adjoint1_vs_kstep.gif")
    print("      - gorkov_kstep_trajectory.gif")
    print("      - gorkov_snapshot_t20.png")
    print("      - U_vs_time.png")
    
    # Save metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'Nx': cfg.Nx, 'Ny': cfg.Ny,
            'n_steps': cfg.n_steps, 'K': cfg.K, 'dt': cfg.dt,
            'x0_mm': x0 * 1e3, 'y0_mm': y0 * 1e3,
        },
        'results': {
            res['method']: {
                'total_U_fJ': sum(res['U_values']) * 1e15,
                'final_U_fJ': res['U_values'][-1] * 1e15,
                'final_position_mm': [res['positions'][-1][0] * 1e3, res['positions'][-1][1] * 1e3],
            }
            for res in results_list
        }
    }
    
    with open(output_dir / "run_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
