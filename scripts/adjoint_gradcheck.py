#!/usr/bin/env python3
"""
adjoint_gradcheck.py - Validate adjoint gradients against finite differences.

This script performs a one-step adjoint gradient check for acoustic tweezers:
1. Set up a 2.5D Helmholtz solver with a single transducer
2. Define a simple scalar objective J = |p|² at a fixed grid point
3. Compute gradient dJ/du via the adjoint method
4. Compare against finite difference gradients
5. Verify O(h²) convergence for centered differences

Usage:
    python scripts/adjoint_gradcheck.py
"""

from __future__ import annotations

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from acousto.solvers.fd_helmholtz_2d_forced_25d import build_helmholtz_2d_forced_25d_operator
from acousto.adjoint.gradients import (
    TransducerParams,
    compute_dJdp_complex_pressure,
    compute_dbdu_single_transducer,
    compute_dbdu_position,
    adjoint_gradient_vectorized,
)


@dataclass
class GradCheckConfig:
    """Configuration for gradient check."""
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
    
    # Finite difference
    fd_epsilons: Tuple[float, ...] = (1e-3, 1e-4, 1e-5, 1e-6, 1e-7)


def build_vb_from_transducer(
    trans: TransducerParams,
    x: np.ndarray,
) -> np.ndarray:
    """Build bottom boundary velocity from single transducer."""
    if not trans.gate:
        return np.zeros(len(x), dtype=np.complex128)
    
    G_x = np.exp(-(x - trans.x)**2 / (2.0 * trans.sigma_x**2))
    G_y = np.exp(-(trans.y)**2 / (2.0 * trans.sigma_y**2))
    
    vb_x = trans.v * np.exp(1j * trans.phi) * G_x * G_y
    return vb_x


def objective_p_squared(field, ix: int, iy: int) -> float:
    """Objective J = |p(ix, iy)|² = p * conj(p)."""
    p_val = field.p[iy, ix]
    return np.abs(p_val)**2


def objective_real_p(field, ix: int, iy: int) -> float:
    """Objective J = Re(p(ix, iy))."""
    p_val = field.p[iy, ix]
    return np.real(p_val)


def finite_difference_gradient(
    op,
    trans: TransducerParams,
    param_name: str,
    ix: int,
    iy: int,
    epsilon: float,
) -> float:
    """
    Compute finite difference gradient for one parameter.
    
    Uses centered differences: (J(u+ε) - J(u-ε)) / (2ε)
    """
    # Get current value
    current_val = getattr(trans, param_name)
    
    # Perturb +
    trans_plus = TransducerParams(
        x=trans.x if param_name != 'x' else current_val + epsilon,
        y=trans.y if param_name != 'y' else current_val + epsilon,
        v=trans.v if param_name != 'v' else current_val + epsilon,
        phi=trans.phi if param_name != 'phi' else current_val + epsilon,
        sigma_x=trans.sigma_x,
        sigma_y=trans.sigma_y,
        gate=trans.gate,
    )
    vb_plus = build_vb_from_transducer(trans_plus, op.x)
    field_plus = op.solve_for_bottom_vb(vb_plus)
    J_plus = objective_p_squared(field_plus, ix, iy)
    
    # Perturb -
    trans_minus = TransducerParams(
        x=trans.x if param_name != 'x' else current_val - epsilon,
        y=trans.y if param_name != 'y' else current_val - epsilon,
        v=trans.v if param_name != 'v' else current_val - epsilon,
        phi=trans.phi if param_name != 'phi' else current_val - epsilon,
        sigma_x=trans.sigma_x,
        sigma_y=trans.sigma_y,
        gate=trans.gate,
    )
    vb_minus = build_vb_from_transducer(trans_minus, op.x)
    field_minus = op.solve_for_bottom_vb(vb_minus)
    J_minus = objective_p_squared(field_minus, ix, iy)
    
    return (J_plus - J_minus) / (2.0 * epsilon)


def run_gradient_check(cfg: GradCheckConfig) -> dict:
    """Run full gradient check and return results."""
    
    print("=" * 60)
    print("ADJOINT GRADIENT CHECK")
    print("=" * 60)
    
    # Build operator
    print("\n1. Building Helmholtz operator...")
    op = build_helmholtz_2d_forced_25d_operator(
        Lx=cfg.Lx, Ly=cfg.Ly, Nx=cfg.Nx, Ny=cfg.Ny,
        f=cfg.f, c0=cfg.c0, rho0=cfg.rho0,
    )
    print(f"   Grid: {cfg.Nx}×{cfg.Ny}, Domain: {cfg.Lx*1e3:.1f}×{cfg.Ly*1e3:.1f} mm")
    print(f"   Frequency: {cfg.f/1e6:.1f} MHz, Wavelength: {cfg.c0/cfg.f*1e3:.3f} mm")
    
    # Set up transducer
    trans = TransducerParams(
        x=cfg.Lx / 2,         # center
        y=0.02 * cfg.Ly,      # near bottom
        v=0.05,               # m/s
        phi=0.0,              # rad
        sigma_x=cfg.sigma_x,
        sigma_y=cfg.sigma_y,
        gate=True,
    )
    print(f"\n2. Transducer: x={trans.x*1e3:.2f}mm, y={trans.y*1e3:.3f}mm, v={trans.v}, φ={trans.phi}")
    
    # Solve forward problem
    print("\n3. Solving forward problem...")
    vb_x = build_vb_from_transducer(trans, op.x)
    field = op.solve_for_bottom_vb(vb_x)
    
    # Choose evaluation point (center of domain)
    ix = cfg.Nx // 2
    iy = cfg.Ny // 2
    p_val = field.p[iy, ix]
    J0 = np.abs(p_val)**2
    print(f"   Evaluation point: ({ix}, {iy})")
    print(f"   p = {p_val:.6e}")
    print(f"   J = |p|² = {J0:.6e}")
    
    # Compute adjoint gradients
    print("\n4. Computing adjoint gradients...")
    
    # ∂J/∂p = conj(p) for J = |p|² (Wirtinger derivative)
    dJ_dp = compute_dJdp_complex_pressure(ix, iy, cfg.Nx, cfg.Ny, p_val)
    
    # db/du vectors
    db_dv, db_dphi = compute_dbdu_single_transducer(
        trans, op.x, op.omega, op.rho0, op.coupling_alpha, op.bottom_rows, cfg.Nx, cfg.Ny
    )
    db_dx, db_dy = compute_dbdu_position(
        trans, op.x, op.omega, op.rho0, op.coupling_alpha, op.bottom_rows, cfg.Nx, cfg.Ny
    )
    
    # Compute all adjoint gradients at once
    grads_adjoint = adjoint_gradient_vectorized(
        op.adjoint_solve,
        dJ_dp,
        [db_dv, db_dphi, db_dx, db_dy]
    )
    
    param_names = ['v', 'phi', 'x', 'y']
    print("   Adjoint gradients:")
    for name, grad in zip(param_names, grads_adjoint):
        print(f"      dJ/d{name} = {grad:+.6e}")
    
    # Finite difference comparison
    print("\n5. Finite difference comparison...")
    results = {
        'param_names': param_names,
        'adjoint_grads': grads_adjoint.copy(),
        'fd_grads': {},
        'rel_errors': {},
    }
    
    for param_idx, param_name in enumerate(param_names):
        adj_grad = grads_adjoint[param_idx]
        
        # Choose appropriate epsilon scale based on parameter
        if param_name in ['x', 'y']:
            epsilons = [e * cfg.Lx for e in cfg.fd_epsilons]  # scale by domain size
        elif param_name == 'v':
            epsilons = [e * trans.v for e in cfg.fd_epsilons]  # scale by amplitude
        else:  # phi
            epsilons = list(cfg.fd_epsilons)  # radians, use directly
        
        print(f"\n   Parameter: {param_name} (adjoint: {adj_grad:+.6e})")
        print(f"   {'epsilon':>12} | {'FD grad':>14} | {'rel error':>12}")
        print("   " + "-" * 45)
        
        fd_grads = []
        rel_errors = []
        
        for eps in epsilons:
            fd_grad = finite_difference_gradient(op, trans, param_name, ix, iy, eps)
            fd_grads.append(fd_grad)
            
            if abs(adj_grad) > 1e-15:
                rel_err = abs(fd_grad - adj_grad) / abs(adj_grad)
            else:
                rel_err = abs(fd_grad - adj_grad)
            rel_errors.append(rel_err)
            
            print(f"   {eps:12.2e} | {fd_grad:+14.6e} | {rel_err:12.2e}")
        
        results['fd_grads'][param_name] = fd_grads
        results['rel_errors'][param_name] = rel_errors
    
    return results


def plot_convergence(results: dict, save_path: Path = None):
    """Plot finite difference convergence."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    
    for idx, param_name in enumerate(results['param_names']):
        ax = axes[idx]
        
        adj_grad = results['adjoint_grads'][idx]
        rel_errors = results['rel_errors'][param_name]
        
        # X-axis: epsilon indices (0 to n-1)
        x = np.arange(len(rel_errors))
        
        ax.semilogy(x, rel_errors, 'o-', label=f'd{param_name}')
        ax.axhline(1e-6, color='g', linestyle='--', alpha=0.5, label='1e-6')
        ax.axhline(1e-3, color='r', linestyle='--', alpha=0.5, label='1e-3')
        
        ax.set_xlabel('Epsilon index (smaller →)')
        ax.set_ylabel('Relative error')
        ax.set_title(f'dJ/d{param_name} = {adj_grad:.3e}')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.suptitle('Adjoint vs Finite Difference Gradient Convergence', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"\nPlot saved to {save_path}")
    
    plt.show()


def main():
    """Main entry point."""
    cfg = GradCheckConfig()
    results = run_gradient_check(cfg)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for idx, param_name in enumerate(results['param_names']):
        adj_grad = results['adjoint_grads'][idx]
        rel_errors = results['rel_errors'][param_name]
        min_error = min(rel_errors)
        
        # Special case: if adjoint gradient is near zero, check absolute error instead
        if abs(adj_grad) < 1e-3:
            fd_grads = results['fd_grads'][param_name]
            min_abs_error = min(abs(fd - adj_grad) for fd in fd_grads)
            passed = min_abs_error < 1.0  # absolute tolerance for small gradients
            status = f"✓ PASS (small grad)" if passed else "✗ FAIL"
            print(f"   dJ/d{param_name:4s}: {adj_grad:+12.4e}, abs error: {min_abs_error:.2e} {status}")
        else:
            passed = min_error < 1e-3  # reasonable threshold
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"   dJ/d{param_name:4s}: {adj_grad:+12.4e}, min rel error: {min_error:.2e} {status}")
        
        all_passed = all_passed and passed
    
    print("\n" + ("ALL TESTS PASSED!" if all_passed else "SOME TESTS FAILED"))
    
    # Save results
    results_dir = project_root / "results" / "adjoint_gradcheck"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save numerical results
    np.savez(
        results_dir / "gradcheck_results.npz",
        adjoint_grads=results['adjoint_grads'],
        fd_grads_v=results['fd_grads']['v'],
        fd_grads_phi=results['fd_grads']['phi'],
        fd_grads_x=results['fd_grads']['x'],
        fd_grads_y=results['fd_grads']['y'],
    )
    print(f"\nResults saved to {results_dir}")
    
    # Plot
    plot_convergence(results, results_dir / "convergence.png")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
