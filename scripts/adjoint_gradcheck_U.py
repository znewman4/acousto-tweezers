#!/usr/bin/env python3
"""
adjoint_gradcheck_U.py - Validate adjoint gradients for Gor'kov potential objective.

This script performs an adjoint gradient check for the control-relevant objective:
    J = U(x_p, y_p) = Gor'kov potential at the particle position

This is a more physically meaningful objective than |p|² because:
- The Gor'kov potential determines trap locations (U minima)
- Radiation force F = -∇U drives particle motion
- This is what the controller ultimately needs to optimize

The script:
1. Sets up a 2.5D Helmholtz solver with a single transducer
2. Computes U(x_p) from the pressure field
3. Computes ∂U/∂u via the adjoint method
4. Validates against finite difference gradients
5. Reports convergence behavior

Success criteria:
- Relative error between adjoint and FD gradients should decrease as O(ε²) 
  for centered differences until numerical precision limits are reached
- Final relative error < 1e-4 for at least one ε value

Usage:
    python scripts/adjoint_gradcheck_U.py
"""

from __future__ import annotations

import sys
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Tuple

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
    compute_dbdu_position,
    adjoint_gradient_vectorized,
)


@dataclass
class GradCheckConfig:
    """Configuration for Gor'kov potential gradient check."""
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
    
    # Finite difference epsilons (relative to parameter scale)
    fd_epsilons: Tuple[float, ...] = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)


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


def compute_gorkov_at_point(
    field,
    particle: ParticleProps,
    ix: int,
    iy: int,
) -> float:
    """Compute Gor'kov potential U at grid point (ix, iy)."""
    U, Fx, Fy = gorkov_potential_and_force_2d(field, particle)
    return U[iy, ix]


def finite_difference_gradient_U(
    op,
    trans: TransducerParams,
    particle: ParticleProps,
    param_name: str,
    ix: int,
    iy: int,
    epsilon: float,
) -> float:
    """
    Compute finite difference gradient of U w.r.t. one parameter.
    
    Uses centered differences: (J(u+ε) - J(u-ε)) / (2ε)
    """
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
    U_plus = compute_gorkov_at_point(field_plus, particle, ix, iy)
    
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
    U_minus = compute_gorkov_at_point(field_minus, particle, ix, iy)
    
    return (U_plus - U_minus) / (2.0 * epsilon)


def run_gradient_check(cfg: GradCheckConfig) -> dict:
    """Run full gradient check for Gor'kov potential objective."""
    
    print("=" * 70)
    print("ADJOINT GRADIENT CHECK: Gor'kov Potential Objective J = U(x_p)")
    print("=" * 70)
    
    # Build operator
    print("\n1. Building Helmholtz operator...")
    op = build_helmholtz_2d_forced_25d_operator(
        Lx=cfg.Lx, Ly=cfg.Ly, Nx=cfg.Nx, Ny=cfg.Ny,
        f=cfg.f, c0=cfg.c0, rho0=cfg.rho0,
    )
    dx = op.dx
    dy = op.dy
    print(f"   Grid: {cfg.Nx}×{cfg.Ny}, Domain: {cfg.Lx*1e3:.1f}×{cfg.Ly*1e3:.1f} mm")
    print(f"   Frequency: {cfg.f/1e6:.1f} MHz, Wavelength: {cfg.c0/cfg.f*1e3:.3f} mm")
    
    # Set up particle
    particle = ParticleProps(
        a=cfg.particle_a,
        rho_p=cfg.particle_rho_p,
        c_p=cfg.particle_c_p,
    )
    print(f"\n2. Particle: a={particle.a*1e6:.1f}µm, ρ={particle.rho_p} kg/m³")
    
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
    print(f"\n3. Transducer: x={trans.x*1e3:.2f}mm, y={trans.y*1e3:.3f}mm, v={trans.v}, φ={trans.phi}")
    
    # Solve forward problem
    print("\n4. Solving forward problem...")
    vb_x = build_vb_from_transducer(trans, op.x)
    field = op.solve_for_bottom_vb(vb_x)
    
    # Choose evaluation point (closer to transducer for more variation)
    # Using quarter-height to be in a more interesting region of the field
    ix = cfg.Nx // 2
    iy = cfg.Ny // 4  # closer to bottom where transducer is
    
    # Compute Gor'kov potential
    U, Fx, Fy = gorkov_potential_and_force_2d(field, particle)
    U0 = U[iy, ix]
    F_mag = np.sqrt(Fx[iy, ix]**2 + Fy[iy, ix]**2)
    
    print(f"   Particle position: ({ix}, {iy}) = ({op.x[ix]*1e3:.3f}, {op.y[iy]*1e3:.3f}) mm")
    print(f"   p = {field.p[iy, ix]:.4e}")
    print(f"   U = {U0:.6e} J (Gor'kov potential)")
    print(f"   |F| = {F_mag:.6e} N (radiation force magnitude)")
    
    # Compute adjoint gradients
    print("\n5. Computing adjoint gradients...")
    
    # ∂U/∂p (accounts for both E_pot and E_kin terms)
    dJ_dp = compute_dJdp_gorkov_potential(
        ix, iy, cfg.Nx, cfg.Ny, dx, dy,
        field.p, op.omega, cfg.rho0, cfg.c0,
        cfg.particle_a, cfg.particle_rho_p, cfg.particle_c_p,
    )
    
    # ∂b/∂u vectors
    db_dv, db_dphi = compute_dbdu_single_transducer(
        trans, op.x, op.omega, op.rho0, op.coupling_alpha, op.bottom_rows, cfg.Nx, cfg.Ny
    )
    db_dx, db_dy = compute_dbdu_position(
        trans, op.x, op.omega, op.rho0, op.coupling_alpha, op.bottom_rows, cfg.Nx, cfg.Ny
    )
    
    # Compute adjoint gradients
    grads_adjoint = adjoint_gradient_vectorized(
        op.adjoint_solve,
        dJ_dp,
        [db_dv, db_dphi, db_dx, db_dy]
    )
    
    param_names = ['v', 'phi', 'x', 'y']
    print("   Adjoint gradients:")
    for name, grad in zip(param_names, grads_adjoint):
        print(f"      ∂U/∂{name} = {grad:+.6e}")
    
    # Finite difference comparison
    print("\n6. Finite difference comparison...")
    print("   " + "=" * 65)
    
    results = {
        'config': asdict(cfg),
        'particle_pos': {'ix': ix, 'iy': iy, 'x_mm': op.x[ix]*1e3, 'y_mm': op.y[iy]*1e3},
        'U0': float(U0),
        'param_names': param_names,
        'adjoint_grads': grads_adjoint.tolist(),
        'fd_results': {},
    }
    
    for param_idx, param_name in enumerate(param_names):
        adj_grad = grads_adjoint[param_idx]
        
        # Choose appropriate epsilon scale
        if param_name == 'v':
            epsilons = [e * trans.v for e in cfg.fd_epsilons]
        elif param_name == 'phi':
            epsilons = list(cfg.fd_epsilons)  # radians
        elif param_name in ['x', 'y']:
            epsilons = [e * cfg.Lx for e in cfg.fd_epsilons]  # scale by domain
        else:
            epsilons = [e * cfg.Lx for e in cfg.fd_epsilons]
        
        print(f"\n   Parameter: {param_name} (adjoint: {adj_grad:+.6e})")
        print(f"   {'epsilon':>12} | {'FD grad':>14} | {'abs error':>12} | {'rel error':>12}")
        print("   " + "-" * 58)
        
        fd_grads = []
        abs_errors = []
        rel_errors = []
        
        for eps in epsilons:
            fd_grad = finite_difference_gradient_U(op, trans, particle, param_name, ix, iy, eps)
            fd_grads.append(fd_grad)
            
            abs_err = abs(fd_grad - adj_grad)
            abs_errors.append(abs_err)
            
            if abs(adj_grad) > 1e-20:
                rel_err = abs_err / abs(adj_grad)
            else:
                rel_err = abs_err
            rel_errors.append(rel_err)
            
            print(f"   {eps:12.2e} | {fd_grad:+14.6e} | {abs_err:12.2e} | {rel_err:12.2e}")
        
        results['fd_results'][param_name] = {
            'epsilons': epsilons,
            'fd_grads': fd_grads,
            'abs_errors': abs_errors,
            'rel_errors': rel_errors,
        }
    
    return results


def main():
    """Main entry point."""
    cfg = GradCheckConfig()
    results = run_gradient_check(cfg)
    
    # Summary and success check
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nSuccess criteria: relative error < 1e-4 for at least one ε value")
    print("(For centered FD, error should be O(ε²) until numerical limits)\n")
    
    all_passed = True
    for param_name in results['param_names']:
        param_idx = results['param_names'].index(param_name)
        adj_grad = results['adjoint_grads'][param_idx]
        rel_errors = results['fd_results'][param_name]['rel_errors']
        min_error = min(rel_errors)
        best_eps_idx = rel_errors.index(min_error)
        best_eps = results['fd_results'][param_name]['epsilons'][best_eps_idx]
        
        # Check if gradient is essentially zero
        if abs(adj_grad) < 1e-20:
            fd_grads = results['fd_results'][param_name]['fd_grads']
            max_fd = max(abs(g) for g in fd_grads)
            if max_fd < 1e-15:
                passed = True
                status = "✓ PASS (both ~0)"
            else:
                passed = False
                status = "✗ FAIL (adj~0 but FD not)"
        else:
            passed = min_error < 1e-4
            status = "✓ PASS" if passed else "✗ FAIL"
        
        all_passed = all_passed and passed
        print(f"   ∂U/∂{param_name:4s}: adj={adj_grad:+12.4e}, min rel err={min_error:.2e} @ ε={best_eps:.1e}  {status}")
    
    print("\n" + ("ALL TESTS PASSED!" if all_passed else "SOME TESTS FAILED"))
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = project_root / "results" / "adjoint_gradcheck_U" / f"run_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON summary
    with open(results_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save human-readable summary
    with open(results_dir / "summary.txt", 'w') as f:
        f.write("Adjoint Gradient Check: Gor'kov Potential Objective\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Objective: J = U(x_p, y_p) = {results['U0']:.6e} J\n")
        f.write(f"Particle position: ({results['particle_pos']['x_mm']:.3f}, {results['particle_pos']['y_mm']:.3f}) mm\n\n")
        
        for param_name in results['param_names']:
            param_idx = results['param_names'].index(param_name)
            adj_grad = results['adjoint_grads'][param_idx]
            min_err = min(results['fd_results'][param_name]['rel_errors'])
            f.write(f"∂U/∂{param_name}: adjoint={adj_grad:+.6e}, min_rel_error={min_err:.2e}\n")
        
        f.write(f"\nAll passed: {all_passed}\n")
    
    print(f"\nResults saved to {results_dir}")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
