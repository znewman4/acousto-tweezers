#!/usr/bin/env python3
"""
adjoint_steer_1step.py - One-step adjoint steering demonstration.

This script demonstrates that adjoint gradients actually improve the objective:
1. Start from a baseline control u0
2. Compute J0 = U(x_p; u0) via forward solve
3. Compute gradient g = ∂J/∂θ via adjoint
4. Line-search: try θ1 = θ0 - α * g_normed for several α values
5. Show that J1 < J0 for optimal α (since we minimize U for trapping)

Note: We MINIMIZE U to push particle toward trap (U minimum).
      So the update is θ1 = θ0 - α * g (gradient descent).

Usage:
    python scripts/adjoint_steer_1step.py [--fast]
"""

from __future__ import annotations

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
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
class SteerConfig:
    """Configuration for 1-step adjoint steering."""
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
    particle_a: float = 5.0e-6      # 5 µm radius
    particle_rho_p: float = 1050.0  # kg/m³
    particle_c_p: float = 2350.0    # m/s
    
    # Line-search α values (in normalized gradient units)
    alphas: Tuple[float, ...] = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
    
    # Control bounds
    v_min: float = 0.01
    v_max: float = 0.2
    phi_min: float = -np.pi
    phi_max: float = np.pi


def build_vb_from_transducer(trans: TransducerParams, x: np.ndarray) -> np.ndarray:
    """Build bottom boundary velocity from single transducer."""
    if not trans.gate:
        return np.zeros(len(x), dtype=np.complex128)
    G_x = np.exp(-(x - trans.x)**2 / (2.0 * trans.sigma_x**2))
    G_y = np.exp(-(trans.y)**2 / (2.0 * trans.sigma_y**2))
    return trans.v * np.exp(1j * trans.phi) * G_x * G_y


def compute_U_and_F(op, trans: TransducerParams, particle: ParticleProps, x_p: float, y_p: float):
    """Compute Gor'kov potential and force at particle position."""
    vb_x = build_vb_from_transducer(trans, op.x)
    field = op.solve_for_bottom_vb(vb_x)
    U, Fx, Fy = gorkov_potential_and_force_2d(field, particle)
    
    # Get grid indices for particle position
    ix = int(np.clip(round((x_p - op.x[0]) / op.dx), 0, op.Nx - 1))
    iy = int(np.clip(round((y_p - op.y[0]) / op.dy), 0, op.Ny - 1))
    
    return field, U[iy, ix], Fx[iy, ix], Fy[iy, ix], ix, iy


def compute_adjoint_gradient(
    op, trans: TransducerParams, particle: ParticleProps, 
    field, ix: int, iy: int, cfg: SteerConfig
) -> Tuple[float, float]:
    """Compute adjoint gradient ∂U/∂(v, phi) at particle position."""
    dJ_dp = compute_dJdp_gorkov_potential(
        ix, iy, cfg.Nx, cfg.Ny, op.dx, op.dy,
        field.p, op.omega, cfg.rho0, cfg.c0,
        cfg.particle_a, cfg.particle_rho_p, cfg.particle_c_p,
    )
    
    db_dv, db_dphi = compute_dbdu_single_transducer(
        trans, op.x, op.omega, op.rho0, op.coupling_alpha, op.bottom_rows, cfg.Nx, cfg.Ny
    )
    
    grads = adjoint_gradient_vectorized(op.adjoint_solve, dJ_dp, [db_dv, db_dphi])
    return grads[0], grads[1]  # dU_dv, dU_dphi


def clamp(val: float, lo: float, hi: float) -> float:
    """Clamp value to [lo, hi]."""
    return max(lo, min(hi, val))


def run_1step_steering(cfg: SteerConfig, fast: bool = False) -> dict:
    """Run one-step adjoint steering with line search."""
    
    if fast:
        cfg.Nx = 32
        cfg.Ny = 32
        cfg.alphas = (0.0, 0.1, 1.0)
    
    print("=" * 70)
    print("ONE-STEP ADJOINT STEERING: J = U(x_p)")
    print("=" * 70)
    
    # Build operator
    print("\n1. Building Helmholtz operator...")
    op = build_helmholtz_2d_forced_25d_operator(
        Lx=cfg.Lx, Ly=cfg.Ly, Nx=cfg.Nx, Ny=cfg.Ny,
        f=cfg.f, c0=cfg.c0, rho0=cfg.rho0,
    )
    print(f"   Grid: {cfg.Nx}×{cfg.Ny}")
    
    # Particle properties
    particle = ParticleProps(a=cfg.particle_a, rho_p=cfg.particle_rho_p, c_p=cfg.particle_c_p)
    
    # Particle position (fixed, in center of domain)
    x_p = cfg.Lx * 0.5
    y_p = cfg.Ly * 0.4  # slightly below center to be near transducer field
    print(f"\n2. Particle at ({x_p*1e3:.3f}, {y_p*1e3:.3f}) mm")
    
    # Baseline transducer configuration
    trans0 = TransducerParams(
        x=cfg.Lx * 0.5,
        y=0.02 * cfg.Ly,
        v=0.05,
        phi=0.0,
        sigma_x=cfg.sigma_x,
        sigma_y=cfg.sigma_y,
        gate=True,
    )
    print(f"\n3. Baseline control: v={trans0.v}, φ={trans0.phi:.4f}")
    
    # Forward solve at baseline
    print("\n4. Computing baseline J0...")
    field0, U0, Fx0, Fy0, ix, iy = compute_U_and_F(op, trans0, particle, x_p, y_p)
    print(f"   U0 = {U0:.6e} J")
    print(f"   F0 = ({Fx0:.4e}, {Fy0:.4e}) N")
    
    # Compute adjoint gradient
    print("\n5. Computing adjoint gradient...")
    dU_dv, dU_dphi = compute_adjoint_gradient(op, trans0, particle, field0, ix, iy, cfg)
    grad_norm = np.sqrt(dU_dv**2 + dU_dphi**2)
    print(f"   ∂U/∂v   = {dU_dv:+.6e}")
    print(f"   ∂U/∂phi = {dU_dphi:+.6e}")
    print(f"   ||g||   = {grad_norm:.6e}")
    
    # Normalize gradient for line search (scale so α=1 gives ~10% parameter change)
    # We'll use separate scales for v and phi
    if abs(dU_dv) > 1e-30:
        scale_v = 0.1 * trans0.v / abs(dU_dv)
    else:
        scale_v = 0.0
    if abs(dU_dphi) > 1e-30:
        scale_phi = 0.1 / abs(dU_dphi)  # ~0.1 rad for α=1
    else:
        scale_phi = 0.0
    
    # Line search
    print("\n6. Line search (minimizing U)...")
    print(f"   {'α':>8} | {'v_new':>10} | {'phi_new':>10} | {'U':>14} | {'ΔU':>14} | {'ΔU/U0':>12}")
    print("   " + "-" * 80)
    
    results = []
    for alpha in cfg.alphas:
        # Gradient descent: θ1 = θ0 - α * scaled_grad
        v_new = clamp(trans0.v - alpha * scale_v * dU_dv, cfg.v_min, cfg.v_max)
        phi_new = trans0.phi - alpha * scale_phi * dU_dphi
        # Wrap phi to [-π, π]
        phi_new = ((phi_new + np.pi) % (2 * np.pi)) - np.pi
        
        trans_new = TransducerParams(
            x=trans0.x, y=trans0.y, v=v_new, phi=phi_new,
            sigma_x=trans0.sigma_x, sigma_y=trans0.sigma_y, gate=True,
        )
        
        _, U_new, _, _, _, _ = compute_U_and_F(op, trans_new, particle, x_p, y_p)
        dU = U_new - U0
        dU_rel = dU / abs(U0) if abs(U0) > 1e-30 else 0.0
        
        results.append({
            'alpha': alpha,
            'v': v_new,
            'phi': phi_new,
            'U': U_new,
            'dU': dU,
            'dU_rel': dU_rel,
        })
        
        print(f"   {alpha:8.4f} | {v_new:10.6f} | {phi_new:+10.6f} | {U_new:+14.6e} | {dU:+14.6e} | {dU_rel:+12.4e}")
    
    # Find best α (minimum U, excluding α=0)
    valid_results = [r for r in results if r['alpha'] > 0]
    if valid_results:
        best = min(valid_results, key=lambda r: r['U'])
    else:
        best = results[0]
    
    print(f"\n   Best α = {best['alpha']:.4f}, U_best = {best['U']:.6e}, ΔU = {best['dU']:+.6e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    improvement = best['dU'] < 0
    print(f"   J0 = U(x_p; u0) = {U0:.6e} J")
    print(f"   J1 = U(x_p; u1) = {best['U']:.6e} J")
    print(f"   ΔJ = {best['dU']:+.6e} J ({best['dU_rel']*100:+.4f}%)")
    print(f"   Gradient norm: {grad_norm:.6e}")
    print(f"   Best α: {best['alpha']:.4f}")
    print(f"   Control change: Δv = {best['v'] - trans0.v:+.6f}, Δφ = {best['phi'] - trans0.phi:+.6f}")
    
    if improvement:
        print("\n   ✓ SUCCESS: Gradient descent reduced U (improved trapping)")
    else:
        print("\n   ✗ NOTE: U did not decrease (may be at local minimum or gradient too small)")
    
    return {
        'config': {
            'Nx': cfg.Nx, 'Ny': cfg.Ny, 'Lx': cfg.Lx, 'Ly': cfg.Ly,
            'particle_pos': {'x': x_p, 'y': y_p},
        },
        'baseline': {
            'v': trans0.v, 'phi': trans0.phi,
            'U0': U0, 'Fx0': Fx0, 'Fy0': Fy0,
        },
        'gradient': {
            'dU_dv': dU_dv, 'dU_dphi': dU_dphi, 'norm': grad_norm,
        },
        'line_search': results,
        'best': best,
        'improved': improvement,
    }


def main():
    parser = argparse.ArgumentParser(description="One-step adjoint steering")
    parser.add_argument('--fast', action='store_true', help="Fast mode with reduced resolution")
    args = parser.parse_args()
    
    cfg = SteerConfig()
    results = run_1step_steering(cfg, fast=args.fast)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = project_root / "results" / "adjoint_steer_1step" / f"run_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON summary
    with open(results_dir / "summary.json", 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    # Save CSV of line search
    import csv
    with open(results_dir / "alphas.csv", 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['alpha', 'v', 'phi', 'U', 'dU', 'dU_rel'])
        writer.writeheader()
        writer.writerows(results['line_search'])
    
    print(f"\nResults saved to {results_dir}")
    
    # Optional: simple plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        alphas = [r['alpha'] for r in results['line_search']]
        Us = [r['U'] for r in results['line_search']]
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(alphas, Us, 'o-', markersize=8)
        ax.axhline(results['baseline']['U0'], color='r', linestyle='--', label='U0 (baseline)')
        ax.set_xlabel('α (step size)')
        ax.set_ylabel('U (Gor\'kov potential) [J]')
        ax.set_title('One-Step Adjoint Steering: U vs α')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(results_dir / "U_vs_alpha.png", dpi=150)
        print(f"Plot saved to {results_dir / 'U_vs_alpha.png'}")
    except Exception as e:
        print(f"(Plotting skipped: {e})")
    
    return 0 if results['improved'] else 1


if __name__ == "__main__":
    sys.exit(main())
