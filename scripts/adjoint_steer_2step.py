#!/usr/bin/env python3
"""
adjoint_steer_2step.py - Two-step lookahead adjoint steering demonstration.

This script demonstrates why lookahead matters for control:
1. One-step: optimize U(x0; u0) - myopic objective
2. Two-step: optimize J_total = U(x0; u0) + β * U(x1; u1)
   where x1 = x0 + dt * μ * F(x0; u0) is the particle position after one physics step

The two-step objective captures:
- Immediate trapping (U at current position)
- Future state quality (U at predicted next position)

This demonstrates the value of "repositioning" - control choices that may not
be optimal for the current instant but set up better future states.

Key insight: When we care about t+1, the optimal u0 depends on how it affects
both the current field AND the dynamics that move the particle.

Usage:
    python scripts/adjoint_steer_2step.py [--fast]
"""

from __future__ import annotations

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple, List

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


@dataclass  
class TwoStepConfig:
    """Configuration for 2-step lookahead steering."""
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
    particle_a: float = 50.0e-6     # 50 µm radius (larger for bigger forces)
    particle_rho_p: float = 1050.0  # kg/m³
    particle_c_p: float = 2350.0    # m/s
    
    # Dynamics
    mu: float = 1.0e-3  # dynamic viscosity [Pa·s]
    dt: float = 0.1     # time step [s] (longer for meaningful motion)
    
    # 2-step weighting
    beta: float = 1.0   # weight on future term
    
    # Line-search
    alphas: Tuple[float, ...] = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
    
    # FD epsilon for dx1/dtheta approximation
    fd_eps_v: float = 1e-4
    fd_eps_phi: float = 1e-4
    
    # Control bounds
    v_min: float = 0.01
    v_max: float = 0.2


def build_vb_from_transducer(trans: TransducerParams, x: np.ndarray) -> np.ndarray:
    """Build bottom boundary velocity from single transducer."""
    if not trans.gate:
        return np.zeros(len(x), dtype=np.complex128)
    G_x = np.exp(-(x - trans.x)**2 / (2.0 * trans.sigma_x**2))
    G_y = np.exp(-(trans.y)**2 / (2.0 * trans.sigma_y**2))
    return trans.v * np.exp(1j * trans.phi) * G_x * G_y


def compute_U_and_F_at_pos(op, trans: TransducerParams, particle: ParticleProps, x_p: float, y_p: float):
    """Compute Gor'kov potential and force at arbitrary position (continuous)."""
    vb_x = build_vb_from_transducer(trans, op.x)
    field = op.solve_for_bottom_vb(vb_x)
    U, Fx, Fy = gorkov_potential_and_force_2d(field, particle)
    
    # Bilinear interpolation for continuous position
    ix_f = (x_p - op.x[0]) / op.dx
    iy_f = (y_p - op.y[0]) / op.dy
    
    ix0 = int(np.clip(np.floor(ix_f), 0, op.Nx - 2))
    iy0 = int(np.clip(np.floor(iy_f), 0, op.Ny - 2))
    ix1 = ix0 + 1
    iy1 = iy0 + 1
    
    fx = ix_f - ix0
    fy = iy_f - iy0
    fx = np.clip(fx, 0, 1)
    fy = np.clip(fy, 0, 1)
    
    # Bilinear interp
    U_interp = (U[iy0, ix0] * (1-fx) * (1-fy) + U[iy0, ix1] * fx * (1-fy) +
                U[iy1, ix0] * (1-fx) * fy + U[iy1, ix1] * fx * fy)
    Fx_interp = (Fx[iy0, ix0] * (1-fx) * (1-fy) + Fx[iy0, ix1] * fx * (1-fy) +
                 Fx[iy1, ix0] * (1-fx) * fy + Fx[iy1, ix1] * fx * fy)
    Fy_interp = (Fy[iy0, ix0] * (1-fx) * (1-fy) + Fy[iy0, ix1] * fx * (1-fy) +
                 Fy[iy1, ix0] * (1-fx) * fy + Fy[iy1, ix1] * fx * fy)
    
    return field, U_interp, Fx_interp, Fy_interp


def compute_U_at_grid(op, trans: TransducerParams, particle: ParticleProps, ix: int, iy: int):
    """Compute U at exact grid point (for adjoint gradient)."""
    vb_x = build_vb_from_transducer(trans, op.x)
    field = op.solve_for_bottom_vb(vb_x)
    U, Fx, Fy = gorkov_potential_and_force_2d(field, particle)
    return field, U[iy, ix], Fx[iy, ix], Fy[iy, ix]


def overdamped_step(x0: float, y0: float, Fx: float, Fy: float, cfg: TwoStepConfig) -> Tuple[float, float]:
    """One overdamped particle step: x1 = x0 + dt * F / gamma."""
    gamma = 6.0 * np.pi * cfg.mu * cfg.particle_a  # Stokes drag
    x1 = np.clip(x0 + cfg.dt * Fx / gamma, 0, cfg.Lx)
    y1 = np.clip(y0 + cfg.dt * Fy / gamma, 0, cfg.Ly)
    return x1, y1


def compute_adjoint_gradient_at_pos(
    op, trans: TransducerParams, particle: ParticleProps,
    x_p: float, y_p: float, cfg: TwoStepConfig
) -> Tuple[float, float, np.ndarray]:
    """Compute adjoint gradient ∂U/∂(v, phi) at particle position."""
    # Use nearest grid point for adjoint (required by current implementation)
    ix = int(np.clip(round((x_p - op.x[0]) / op.dx), 0, op.Nx - 1))
    iy = int(np.clip(round((y_p - op.y[0]) / op.dy), 0, op.Ny - 1))
    
    # Forward solve
    vb_x = build_vb_from_transducer(trans, op.x)
    field = op.solve_for_bottom_vb(vb_x)
    
    dJ_dp = compute_dJdp_gorkov_potential(
        ix, iy, cfg.Nx, cfg.Ny, op.dx, op.dy,
        field.p, op.omega, cfg.rho0, cfg.c0,
        cfg.particle_a, cfg.particle_rho_p, cfg.particle_c_p,
    )
    
    db_dv, db_dphi = compute_dbdu_single_transducer(
        trans, op.x, op.omega, op.rho0, op.coupling_alpha, op.bottom_rows, cfg.Nx, cfg.Ny
    )
    
    grads = adjoint_gradient_vectorized(op.adjoint_solve, dJ_dp, [db_dv, db_dphi])
    return grads[0], grads[1], field


def compute_dx1_dtheta_fd(
    op, trans0: TransducerParams, particle: ParticleProps,
    x0: float, y0: float, cfg: TwoStepConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute dx1/dv and dx1/dphi via finite differences.
    
    x1 = x0 + dt * F(x0; theta) / gamma
    dx1/dtheta = dt/gamma * dF/dtheta
    """
    # Baseline x1
    _, _, Fx0, Fy0 = compute_U_at_grid(op, trans0, particle,
        int(np.clip(round((x0 - op.x[0]) / op.dx), 0, op.Nx - 1)),
        int(np.clip(round((y0 - op.y[0]) / op.dy), 0, op.Ny - 1)))
    x1_0, y1_0 = overdamped_step(x0, y0, Fx0, Fy0, cfg)
    
    # Perturb v
    trans_v_plus = TransducerParams(
        x=trans0.x, y=trans0.y, v=trans0.v + cfg.fd_eps_v, phi=trans0.phi,
        sigma_x=trans0.sigma_x, sigma_y=trans0.sigma_y, gate=True
    )
    _, _, Fx_vp, Fy_vp = compute_U_at_grid(op, trans_v_plus, particle,
        int(np.clip(round((x0 - op.x[0]) / op.dx), 0, op.Nx - 1)),
        int(np.clip(round((y0 - op.y[0]) / op.dy), 0, op.Ny - 1)))
    x1_vp, y1_vp = overdamped_step(x0, y0, Fx_vp, Fy_vp, cfg)
    
    trans_v_minus = TransducerParams(
        x=trans0.x, y=trans0.y, v=trans0.v - cfg.fd_eps_v, phi=trans0.phi,
        sigma_x=trans0.sigma_x, sigma_y=trans0.sigma_y, gate=True
    )
    _, _, Fx_vm, Fy_vm = compute_U_at_grid(op, trans_v_minus, particle,
        int(np.clip(round((x0 - op.x[0]) / op.dx), 0, op.Nx - 1)),
        int(np.clip(round((y0 - op.y[0]) / op.dy), 0, op.Ny - 1)))
    x1_vm, y1_vm = overdamped_step(x0, y0, Fx_vm, Fy_vm, cfg)
    
    dx1_dv = np.array([(x1_vp - x1_vm) / (2 * cfg.fd_eps_v),
                       (y1_vp - y1_vm) / (2 * cfg.fd_eps_v)])
    
    # Perturb phi
    trans_phi_plus = TransducerParams(
        x=trans0.x, y=trans0.y, v=trans0.v, phi=trans0.phi + cfg.fd_eps_phi,
        sigma_x=trans0.sigma_x, sigma_y=trans0.sigma_y, gate=True
    )
    _, _, Fx_pp, Fy_pp = compute_U_at_grid(op, trans_phi_plus, particle,
        int(np.clip(round((x0 - op.x[0]) / op.dx), 0, op.Nx - 1)),
        int(np.clip(round((y0 - op.y[0]) / op.dy), 0, op.Ny - 1)))
    x1_pp, y1_pp = overdamped_step(x0, y0, Fx_pp, Fy_pp, cfg)
    
    trans_phi_minus = TransducerParams(
        x=trans0.x, y=trans0.y, v=trans0.v, phi=trans0.phi - cfg.fd_eps_phi,
        sigma_x=trans0.sigma_x, sigma_y=trans0.sigma_y, gate=True
    )
    _, _, Fx_pm, Fy_pm = compute_U_at_grid(op, trans_phi_minus, particle,
        int(np.clip(round((x0 - op.x[0]) / op.dx), 0, op.Nx - 1)),
        int(np.clip(round((y0 - op.y[0]) / op.dy), 0, op.Ny - 1)))
    x1_pm, y1_pm = overdamped_step(x0, y0, Fx_pm, Fy_pm, cfg)
    
    dx1_dphi = np.array([(x1_pp - x1_pm) / (2 * cfg.fd_eps_phi),
                         (y1_pp - y1_pm) / (2 * cfg.fd_eps_phi)])
    
    return dx1_dv, dx1_dphi


def compute_gradU_x(op, trans: TransducerParams, particle: ParticleProps, x_p: float, y_p: float, cfg: TwoStepConfig):
    """Compute spatial gradient of U: ∇_x U at particle position."""
    eps = op.dx * 0.5
    
    _, U_xp, _, _ = compute_U_and_F_at_pos(op, trans, particle, x_p + eps, y_p)
    _, U_xm, _, _ = compute_U_and_F_at_pos(op, trans, particle, x_p - eps, y_p)
    _, U_yp, _, _ = compute_U_and_F_at_pos(op, trans, particle, x_p, y_p + eps)
    _, U_ym, _, _ = compute_U_and_F_at_pos(op, trans, particle, x_p, y_p - eps)
    
    dU_dx = (U_xp - U_xm) / (2 * eps)
    dU_dy = (U_yp - U_ym) / (2 * eps)
    
    return np.array([dU_dx, dU_dy])


def clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def run_2step_comparison(cfg: TwoStepConfig, fast: bool = False) -> dict:
    """Compare 1-step vs 2-step lookahead steering."""
    
    if fast:
        cfg.Nx = 32
        cfg.Ny = 32
        cfg.alphas = (0.0, 0.1, 1.0, 3.0)
    
    print("=" * 75)
    print("TWO-STEP LOOKAHEAD ADJOINT STEERING")
    print("=" * 75)
    print(f"\nObjective: J_total = U(x0; u0) + β * U(x1; u1), β = {cfg.beta}")
    print("where x1 = x0 + dt * μ * F(x0; u0) (overdamped step)")
    
    # Build operator
    print("\n1. Building Helmholtz operator...")
    op = build_helmholtz_2d_forced_25d_operator(
        Lx=cfg.Lx, Ly=cfg.Ly, Nx=cfg.Nx, Ny=cfg.Ny,
        f=cfg.f, c0=cfg.c0, rho0=cfg.rho0,
    )
    
    particle = ParticleProps(a=cfg.particle_a, rho_p=cfg.particle_rho_p, c_p=cfg.particle_c_p)
    
    # Initial particle position (off-center to create interesting dynamics)
    x0 = cfg.Lx * 0.35   # off-center in x
    y0 = cfg.Ly * 0.5    # mid-height
    print(f"\n2. Initial particle position: ({x0*1e3:.3f}, {y0*1e3:.3f}) mm")
    
    # Baseline controls
    trans0 = TransducerParams(
        x=cfg.Lx * 0.5, y=0.02 * cfg.Ly, v=0.05, phi=0.0,
        sigma_x=cfg.sigma_x, sigma_y=cfg.sigma_y, gate=True
    )
    # For 2-step, u1 is initially same as u0
    trans1 = TransducerParams(
        x=trans0.x, y=trans0.y, v=trans0.v, phi=trans0.phi,
        sigma_x=trans0.sigma_x, sigma_y=trans0.sigma_y, gate=True
    )
    
    print(f"\n3. Baseline control: v={trans0.v}, φ={trans0.phi}")
    
    # === BASELINE STATE ===
    print("\n4. Computing baseline state...")
    _, U0_at_x0, Fx0, Fy0 = compute_U_and_F_at_pos(op, trans0, particle, x0, y0)
    x1, y1 = overdamped_step(x0, y0, Fx0, Fy0, cfg)
    _, U0_at_x1, _, _ = compute_U_and_F_at_pos(op, trans1, particle, x1, y1)
    
    J_1step_baseline = U0_at_x0
    J_2step_baseline = U0_at_x0 + cfg.beta * U0_at_x1
    
    print(f"   U(x0; u0) = {U0_at_x0:.6e} J")
    print(f"   F(x0; u0) = ({Fx0:.4e}, {Fy0:.4e}) N")
    print(f"   x1 = ({x1*1e3:.4f}, {y1*1e3:.4f}) mm  (after dt={cfg.dt*1e3:.1f}ms)")
    print(f"   U(x1; u1) = {U0_at_x1:.6e} J")
    print(f"   J_1step = {J_1step_baseline:.6e}")
    print(f"   J_2step = {J_2step_baseline:.6e}")
    
    # === 1-STEP GRADIENT (myopic) ===
    print("\n5. One-step gradient (myopic): ∂U(x0;u0)/∂θ0")
    dU0_dv, dU0_dphi, _ = compute_adjoint_gradient_at_pos(op, trans0, particle, x0, y0, cfg)
    print(f"   ∂U(x0)/∂v   = {dU0_dv:+.6e}")
    print(f"   ∂U(x0)/∂phi = {dU0_dphi:+.6e}")
    
    # === 2-STEP GRADIENT ===
    # J_total = U(x0; u0) + β * U(x1(u0); u1_fixed)
    # dJ/dθ0 = dU(x0)/dθ0 + β * ∇_x U(x1) · dx1/dθ0
    print("\n6. Two-step gradient: ∂J_total/∂θ0")
    print("   J_total = U(x0; u0) + β * U(x1; u1)")
    print("   ∂J/∂θ0 = ∂U(x0)/∂θ0 + β * ∇_x U(x1) · ∂x1/∂θ0")
    
    # Spatial gradient of U at x1 under u1
    gradU_x1 = compute_gradU_x(op, trans1, particle, x1, y1, cfg)
    print(f"   ∇_x U(x1) = ({gradU_x1[0]:+.4e}, {gradU_x1[1]:+.4e})")
    
    # dx1/dθ via FD (since we don't have adjoint through F yet)
    dx1_dv, dx1_dphi = compute_dx1_dtheta_fd(op, trans0, particle, x0, y0, cfg)
    print(f"   ∂x1/∂v   = ({dx1_dv[0]:+.4e}, {dx1_dv[1]:+.4e}) m/(m/s)")
    print(f"   ∂x1/∂phi = ({dx1_dphi[0]:+.4e}, {dx1_dphi[1]:+.4e}) m/rad")
    
    # Chain rule: β * ∇_x U(x1) · dx1/dθ
    lookahead_term_v = cfg.beta * np.dot(gradU_x1, dx1_dv)
    lookahead_term_phi = cfg.beta * np.dot(gradU_x1, dx1_dphi)
    
    dJ2_dv = dU0_dv + lookahead_term_v
    dJ2_dphi = dU0_dphi + lookahead_term_phi
    
    print(f"\n   Lookahead term (β*∇U·dx1/dθ):")
    print(f"     v:   {lookahead_term_v:+.6e}")
    print(f"     phi: {lookahead_term_phi:+.6e}")
    print(f"   Full 2-step gradient:")
    print(f"     ∂J_total/∂v   = {dJ2_dv:+.6e}")
    print(f"     ∂J_total/∂phi = {dJ2_dphi:+.6e}")
    
    # === LINE SEARCH: 1-STEP ===
    print("\n" + "=" * 75)
    print("LINE SEARCH COMPARISON")
    print("=" * 75)
    
    # Scaling for gradient steps
    if abs(dU0_dv) > 1e-30:
        scale_v_1 = 0.1 * trans0.v / abs(dU0_dv)
    else:
        scale_v_1 = 0.0
    if abs(dU0_dphi) > 1e-30:
        scale_phi_1 = 0.1 / abs(dU0_dphi)
    else:
        scale_phi_1 = 0.0
        
    if abs(dJ2_dv) > 1e-30:
        scale_v_2 = 0.1 * trans0.v / abs(dJ2_dv)
    else:
        scale_v_2 = 0.0
    if abs(dJ2_dphi) > 1e-30:
        scale_phi_2 = 0.1 / abs(dJ2_dphi)
    else:
        scale_phi_2 = 0.0
    
    print("\n1-STEP LINE SEARCH (minimizing U(x0)):")
    print(f"   {'α':>6} | {'v':>8} | {'phi':>8} | {'U(x0)':>12} | {'U(x1)':>12} | {'J_2step':>12}")
    print("   " + "-" * 70)
    
    results_1step = []
    for alpha in cfg.alphas:
        v_new = clamp(trans0.v - alpha * scale_v_1 * dU0_dv, cfg.v_min, cfg.v_max)
        phi_new = trans0.phi - alpha * scale_phi_1 * dU0_dphi
        phi_new = ((phi_new + np.pi) % (2*np.pi)) - np.pi
        
        trans_new = TransducerParams(
            x=trans0.x, y=trans0.y, v=v_new, phi=phi_new,
            sigma_x=trans0.sigma_x, sigma_y=trans0.sigma_y, gate=True
        )
        
        _, U_x0, Fx_new, Fy_new = compute_U_and_F_at_pos(op, trans_new, particle, x0, y0)
        x1_new, y1_new = overdamped_step(x0, y0, Fx_new, Fy_new, cfg)
        _, U_x1, _, _ = compute_U_and_F_at_pos(op, trans1, particle, x1_new, y1_new)
        J_2step = U_x0 + cfg.beta * U_x1
        
        results_1step.append({
            'alpha': alpha, 'v': v_new, 'phi': phi_new,
            'U_x0': U_x0, 'U_x1': U_x1, 'J_2step': J_2step,
            'x1': x1_new, 'y1': y1_new,
        })
        
        print(f"   {alpha:6.3f} | {v_new:8.5f} | {phi_new:+8.4f} | {U_x0:+12.4e} | {U_x1:+12.4e} | {J_2step:+12.4e}")
    
    # Best 1-step by U(x0)
    valid_1 = [r for r in results_1step if r['alpha'] > 0]
    best_1step = min(valid_1, key=lambda r: r['U_x0']) if valid_1 else results_1step[0]
    
    print("\n2-STEP LINE SEARCH (minimizing J_total = U(x0) + β*U(x1)):")
    print(f"   {'α':>6} | {'v':>8} | {'phi':>8} | {'U(x0)':>12} | {'U(x1)':>12} | {'J_total':>12}")
    print("   " + "-" * 70)
    
    results_2step = []
    for alpha in cfg.alphas:
        v_new = clamp(trans0.v - alpha * scale_v_2 * dJ2_dv, cfg.v_min, cfg.v_max)
        phi_new = trans0.phi - alpha * scale_phi_2 * dJ2_dphi
        phi_new = ((phi_new + np.pi) % (2*np.pi)) - np.pi
        
        trans_new = TransducerParams(
            x=trans0.x, y=trans0.y, v=v_new, phi=phi_new,
            sigma_x=trans0.sigma_x, sigma_y=trans0.sigma_y, gate=True
        )
        
        _, U_x0, Fx_new, Fy_new = compute_U_and_F_at_pos(op, trans_new, particle, x0, y0)
        x1_new, y1_new = overdamped_step(x0, y0, Fx_new, Fy_new, cfg)
        _, U_x1, _, _ = compute_U_and_F_at_pos(op, trans1, particle, x1_new, y1_new)
        J_total = U_x0 + cfg.beta * U_x1
        
        results_2step.append({
            'alpha': alpha, 'v': v_new, 'phi': phi_new,
            'U_x0': U_x0, 'U_x1': U_x1, 'J_total': J_total,
            'x1': x1_new, 'y1': y1_new,
        })
        
        print(f"   {alpha:6.3f} | {v_new:8.5f} | {phi_new:+8.4f} | {U_x0:+12.4e} | {U_x1:+12.4e} | {J_total:+12.4e}")
    
    # Best 2-step by J_total
    valid_2 = [r for r in results_2step if r['alpha'] > 0]
    best_2step = min(valid_2, key=lambda r: r['J_total']) if valid_2 else results_2step[0]
    
    # === SUMMARY ===
    print("\n" + "=" * 75)
    print("SUMMARY: 1-STEP vs 2-STEP")
    print("=" * 75)
    
    print(f"\nBaseline:")
    print(f"   U(x0) = {J_1step_baseline:.6e}")
    print(f"   U(x1) = {U0_at_x1:.6e}")
    print(f"   J_2step = {J_2step_baseline:.6e}")
    
    print(f"\n1-Step Optimum (min U(x0), α={best_1step['alpha']:.3f}):")
    print(f"   v={best_1step['v']:.5f}, φ={best_1step['phi']:+.4f}")
    print(f"   U(x0) = {best_1step['U_x0']:.6e} (ΔU = {best_1step['U_x0'] - J_1step_baseline:+.4e})")
    print(f"   U(x1) = {best_1step['U_x1']:.6e}")
    print(f"   J_2step = {best_1step['J_2step']:.6e}")
    
    print(f"\n2-Step Optimum (min J_total, α={best_2step['alpha']:.3f}):")
    print(f"   v={best_2step['v']:.5f}, φ={best_2step['phi']:+.4f}")
    print(f"   U(x0) = {best_2step['U_x0']:.6e}")
    print(f"   U(x1) = {best_2step['U_x1']:.6e}")
    print(f"   J_total = {best_2step['J_total']:.6e} (ΔJ = {best_2step['J_total'] - J_2step_baseline:+.4e})")
    
    # Check if objectives differ
    different_choice = (best_1step['alpha'] != best_2step['alpha'] or 
                       abs(best_1step['v'] - best_2step['v']) > 1e-6 or
                       abs(best_1step['phi'] - best_2step['phi']) > 1e-6)
    
    print(f"\n{'✓' if different_choice else '○'} 1-step and 2-step prefer {'DIFFERENT' if different_choice else 'SAME'} controls")
    
    if different_choice:
        print("\n   Key insight: The lookahead term ∇_x U(x1) · dx1/dθ causes the")
        print("   2-step objective to favor controls that position the particle")
        print("   better for the next step, not just minimize current U.")
    
    return {
        'config': {'Nx': cfg.Nx, 'Ny': cfg.Ny, 'beta': cfg.beta, 'dt': cfg.dt},
        'baseline': {
            'x0': [x0, y0], 'x1': [x1, y1],
            'U_x0': U0_at_x0, 'U_x1': U0_at_x1,
            'J_1step': J_1step_baseline, 'J_2step': J_2step_baseline,
        },
        'gradients': {
            '1step': {'dU_dv': dU0_dv, 'dU_dphi': dU0_dphi},
            '2step': {'dJ_dv': dJ2_dv, 'dJ_dphi': dJ2_dphi},
            'lookahead_v': lookahead_term_v, 'lookahead_phi': lookahead_term_phi,
        },
        'results_1step': results_1step,
        'results_2step': results_2step,
        'best_1step': best_1step,
        'best_2step': best_2step,
        'different_choice': different_choice,
    }


def main():
    parser = argparse.ArgumentParser(description="Two-step lookahead adjoint steering")
    parser.add_argument('--fast', action='store_true', help="Fast mode with reduced resolution")
    args = parser.parse_args()
    
    cfg = TwoStepConfig()
    results = run_2step_comparison(cfg, fast=args.fast)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = project_root / "results" / "adjoint_steer_2step" / f"run_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / "summary.json", 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    # CSV for line searches
    import csv
    for name, data in [('1step', results['results_1step']), ('2step', results['results_2step'])]:
        with open(results_dir / f"linesearch_{name}.csv", 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            writer.writeheader()
            writer.writerows(data)
    
    print(f"\nResults saved to {results_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
