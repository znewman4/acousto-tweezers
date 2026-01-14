#!/usr/bin/env python3
"""
Diagnose: Does the Gorkov potential U change when control u changes?

This script checks if the evaluator is correctly computing different U fields
for different control inputs. This is CRITICAL for path_follow to work.
"""

import numpy as np
from acousto.force import ParticleProps
from tweezers.control import (
    DishDomain,
    MediumProps,
    EvaluatorConfig,
    Control2Pucks,
    BottomFootprint25DEvaluator,
)

def main():
    # Setup
    domain = DishDomain(Lx=2e-3, Ly=2e-3, Nx=160, Ny=160)
    medium = MediumProps(
        f=2e6,
        c0=1500.0,
        rho0=1000.0,
        loss_eta=1e-3,
        kz=0.0,
        coupling_alpha=1.0,
    )
    particle = ParticleProps(a=5e-6, rho_p=1050.0, c_p=2350.0)
    
    cfg = EvaluatorConfig(
        sigma_x=0.10e-3,
        bottom_band=0.25e-3,
        dt=2e-3,
        viscosity=1e-3,
        alpha_g=1e6,
    )
    
    ev = BottomFootprint25DEvaluator(domain, medium, particle, cfg)
    
    # Test 1: Same control multiple times
    print("="*70)
    print("TEST 1: Same control applied 5 times (should give IDENTICAL U)")
    print("="*70)
    
    u_test = Control2Pucks(
        xA=0.5e-3, yA=0.15e-3,
        xB=1.5e-3, yB=0.15e-3,
        vA=5e-4, vB=5e-4,
        phiA=0.0, phiB=np.pi,
    )
    
    U_results = []
    for i in range(5):
        _, _, _, _, field, U, Fx, Fy = ev.step(
            xp=1e-3, yp=1e-3,
            target_x=1e-3, target_y=1e-3,
            u=u_test, u_prev=u_test,
            return_fields=True
        )
        U_min, U_max = float(np.min(U)), float(np.max(U))
        U_results.append((U_min, U_max))
        print(f"  Iteration {i}: U ∈ [{U_min:.3e}, {U_max:.3e}]")
    
    # Check if all U are identical
    U_identical = all(u == U_results[0] for u in U_results)
    print(f"  ✓ All U identical: {U_identical}")
    
    # Test 2: Different controls
    print("\n" + "="*70)
    print("TEST 2: Different control parameters (should give DIFFERENT U)")
    print("="*70)
    
    u_configs = [
        ("Control A: xA=0.3mm, xB=1.7mm", Control2Pucks(
            xA=0.3e-3, yA=0.15e-3, xB=1.7e-3, yB=0.15e-3,
            vA=5e-4, vB=5e-4, phiA=0.0, phiB=np.pi
        )),
        ("Control B: xA=0.5mm, xB=1.5mm", Control2Pucks(
            xA=0.5e-3, yA=0.15e-3, xB=1.5e-3, yB=0.15e-3,
            vA=5e-4, vB=5e-4, phiA=0.0, phiB=np.pi
        )),
        ("Control C: xA=0.7mm, xB=1.3mm", Control2Pucks(
            xA=0.7e-3, yA=0.15e-3, xB=1.3e-3, yB=0.15e-3,
            vA=5e-4, vB=5e-4, phiA=0.0, phiB=np.pi
        )),
        ("Control D: Different phases", Control2Pucks(
            xA=0.5e-3, yA=0.15e-3, xB=1.5e-3, yB=0.15e-3,
            vA=5e-4, vB=5e-4, phiA=0.0, phiB=0.0  # Same phase!
        )),
    ]
    
    U_results_control = []
    for label, u_ctrl in u_configs:
        _, _, _, _, field, U, Fx, Fy = ev.step(
            xp=1e-3, yp=1e-3,
            target_x=1e-3, target_y=1e-3,
            u=u_ctrl, u_prev=u_ctrl,
            return_fields=True
        )
        U_min, U_max = float(np.min(U)), float(np.max(U))
        U_mean = float(np.mean(np.abs(U)))
        U_std = float(np.std(U))
        U_results_control.append((U_min, U_max))
        print(f"  {label}")
        print(f"    U ∈ [{U_min:.3e}, {U_max:.3e}]  mean|U|={U_mean:.3e}  std={U_std:.3e}")
    
    # Check if U changes
    U_all_different = len(set(U_results_control)) == len(U_results_control)
    print(f"\n  ✓ All U different: {U_all_different}")
    
    if not U_all_different:
        print("\n  ⚠️  WARNING: Different controls give SAME U!")
        print("  This explains why the Gorkov landscape doesn't change in path_follow!")
        
        # Check if all U fields are nearly identical
        for i in range(len(U_results_control) - 1):
            if U_results_control[i] == U_results_control[i+1]:
                print(f"    Controls {i} and {i+1} produce IDENTICAL U")
    
    # Test 3: Check pressure field changes
    print("\n" + "="*70)
    print("TEST 3: Pressure field p (should change with control)")
    print("="*70)
    
    p_results = []
    for label, u_ctrl in u_configs:
        _, _, _, _, field, U, Fx, Fy = ev.step(
            xp=1e-3, yp=1e-3,
            target_x=1e-3, target_y=1e-3,
            u=u_ctrl, u_prev=u_ctrl,
            return_fields=True
        )
        p_min = float(np.abs(field.p).min())
        p_max = float(np.abs(field.p).max())
        p_mean = float(np.mean(np.abs(field.p)))
        p_results.append((p_min, p_max))
        print(f"  {label}")
        print(f"    |p| ∈ [{p_min:.3e}, {p_max:.3e}]  mean={p_mean:.3e}")
    
    # Check if pressure changes
    p_all_different = len(set(p_results)) == len(p_results)
    print(f"\n  ✓ All |p| different: {p_all_different}")
    
    if not p_all_different:
        print("\n  ⚠️  WARNING: Pressure field NOT changing with control!")
        print("  This suggests the solver is not updating the bottom velocity vb correctly")


if __name__ == "__main__":
    main()
