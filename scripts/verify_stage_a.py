#!/usr/bin/env python3
"""
STAGE A VERIFICATION: Y-Control Authority Probe

This script verifies that the Stage A upgrade (2D forcing band) has restored
y-control authority. It computes Jacobian sensitivities before and after
the upgrade to confirm:
  - ∂F/∂yA and ∂F/∂yB are now non-zero
  - Displacement Jacobian rank is >= 2 (x and y independently controllable)
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

from acousto.force import ParticleProps
from tweezers.control import (
    DishDomain, MediumProps, EvaluatorConfig,
    Control2Pucks, BottomFootprint25DEvaluator,
    ControlState, ControlVector, ControlBounds, ControlRateLimits,
)


def probe_jacobian(
    ev: BottomFootprint25DEvaluator,
    x: float,
    y: float,
    u0: Control2Pucks,
    name: str,
) -> dict:
    """
    Compute Jacobian sensitivities at a given state/control.
    
    Returns dict with:
    - jacobian_force: (2, 8) array
    - jacobian_disp: (2, 8) array
    - svd_force: singular values
    - svd_disp: singular values
    - dim_labels: control dimension names
    """
    dim_labels = ["xA", "yA", "xB", "yB", "vA", "vB", "phiA", "phiB"]
    
    # Epsilons for finite differences
    eps_pos = 5e-6    # 5 µm for positions
    eps_amp = 1e-5    # Small for amplitudes
    eps_phi = 0.05    # ~3 degrees for phases
    
    epsilons = np.array([
        eps_pos, eps_pos,   # xA, yA
        eps_pos, eps_pos,   # xB, yB
        eps_amp, eps_amp,   # vA, vB
        eps_phi, eps_phi,   # phiA, phiB
    ])
    
    # Baseline evaluation
    xp1_base, yp1_base, _, info_base = ev.step(
        xp=x, yp=y, target_x=x, target_y=y,
        u=u0, u_prev=None, return_fields=False,
    )
    fx_base, fy_base = info_base["fx"], info_base["fy"]
    dx_base = xp1_base - x
    dy_base = yp1_base - y
    
    # Storage
    n_dims = 8
    jacobian_force = np.zeros((2, n_dims))
    jacobian_disp = np.zeros((2, n_dims))
    
    u0_arr = np.array([u0.xA, u0.yA, u0.xB, u0.yB, u0.vA, u0.vB, u0.phiA, u0.phiB])
    
    for i in range(n_dims):
        eps = epsilons[i]
        
        # u+ perturbation
        u_plus_arr = u0_arr.copy()
        u_plus_arr[i] += eps
        u_plus = Control2Pucks(
            xA=u_plus_arr[0], yA=u_plus_arr[1],
            xB=u_plus_arr[2], yB=u_plus_arr[3],
            vA=u_plus_arr[4], vB=u_plus_arr[5],
            phiA=u_plus_arr[6], phiB=u_plus_arr[7],
        )
        
        # u- perturbation
        u_minus_arr = u0_arr.copy()
        u_minus_arr[i] -= eps
        u_minus = Control2Pucks(
            xA=u_minus_arr[0], yA=u_minus_arr[1],
            xB=u_minus_arr[2], yB=u_minus_arr[3],
            vA=u_minus_arr[4], vB=u_minus_arr[5],
            phiA=u_minus_arr[6], phiB=u_minus_arr[7],
        )
        
        # Evaluate at u+
        xp1_plus, yp1_plus, _, info_plus = ev.step(
            xp=x, yp=y, target_x=x, target_y=y,
            u=u_plus, u_prev=None, return_fields=False,
        )
        fx_plus, fy_plus = info_plus["fx"], info_plus["fy"]
        dx_plus = xp1_plus - x
        dy_plus = yp1_plus - y
        
        # Evaluate at u-
        xp1_minus, yp1_minus, _, info_minus = ev.step(
            xp=x, yp=y, target_x=x, target_y=y,
            u=u_minus, u_prev=None, return_fields=False,
        )
        fx_minus, fy_minus = info_minus["fx"], info_minus["fy"]
        dx_minus = xp1_minus - x
        dy_minus = yp1_minus - y
        
        # Central difference
        jacobian_force[0, i] = (fx_plus - fx_minus) / (2 * eps)
        jacobian_force[1, i] = (fy_plus - fy_minus) / (2 * eps)
        jacobian_disp[0, i] = (dx_plus - dx_minus) / (2 * eps)
        jacobian_disp[1, i] = (dy_plus - dy_minus) / (2 * eps)
    
    # SVD
    _, s_force, _ = np.linalg.svd(jacobian_force)
    _, s_disp, _ = np.linalg.svd(jacobian_disp)
    
    # Sensitivity norms
    force_norms = np.sqrt(jacobian_force[0]**2 + jacobian_force[1]**2)
    disp_norms = np.sqrt(jacobian_disp[0]**2 + jacobian_disp[1]**2)
    
    return {
        "name": name,
        "jacobian_force": jacobian_force,
        "jacobian_disp": jacobian_disp,
        "svd_force": s_force,
        "svd_disp": s_disp,
        "dim_labels": dim_labels,
        "force_norms": force_norms,
        "disp_norms": disp_norms,
    }


def main():
    print("=" * 70)
    print("STAGE A VERIFICATION: Y-Control Authority Probe")
    print("=" * 70)
    
    # Domain and physics
    domain = DishDomain(Lx=2e-3, Ly=2e-3, Nx=100, Ny=100)
    medium = MediumProps(f=2e6, c0=1500.0, rho0=1000.0, loss_eta=1e-3, kz=0.0, coupling_alpha=1.0)
    particle = ParticleProps(a=5e-6, rho_p=1050.0, c_p=2350.0)
    
    # Test position
    test_x, test_y = 1.0e-3, 1.0e-3
    
    # Test control - transducers near bottom boundary for good coupling
    # With sigma_y=0.15mm and y=0.05mm: coupling = exp(-0.05^2/(2*0.15^2)) ≈ 0.95
    u0 = Control2Pucks(
        xA=0.5e-3, yA=0.05e-3,
        xB=1.5e-3, yB=0.05e-3,
        vA=5e-4, vB=5e-4,
        phiA=0.0, phiB=np.pi,
    )
    
    print(f"\nTest position: ({test_x*1e3:.2f}, {test_y*1e3:.2f}) mm")
    print(f"Transducers: A=({u0.xA*1e3:.2f}, {u0.yA*1e3:.2f}) mm, B=({u0.xB*1e3:.2f}, {u0.yB*1e3:.2f}) mm")
    
    # ===== Test with LEGACY 1D forcing =====
    print("\n" + "-" * 70)
    print("1D FORCING (Legacy - use_2d_forcing=False)")
    print("-" * 70)
    
    cfg_1d = EvaluatorConfig(
        sigma_x=0.10e-3,
        sigma_y=0.05e-3,
        bottom_band=0.25e-3,
        dt=5e-3,
        viscosity=1e-3,
        alpha_g=1e6,
        max_step=0.05e-3,
        use_2d_forcing=False,  # LEGACY
    )
    ev_1d = BottomFootprint25DEvaluator(domain, medium, particle, cfg_1d)
    
    result_1d = probe_jacobian(ev_1d, test_x, test_y, u0, "1D Forcing")
    
    print("\nForce Jacobian ||dF/du|| by dimension:")
    for i, (lbl, val) in enumerate(zip(result_1d["dim_labels"], result_1d["force_norms"])):
        marker = "⚠️ ZERO" if val < 1e-20 else ""
        print(f"  {lbl:6s}: {val:.3e} {marker}")
    
    print(f"\nForce Jacobian SVD: σ₁={result_1d['svd_force'][0]:.3e}, σ₂={result_1d['svd_force'][1]:.3e}")
    print(f"Disp Jacobian SVD:  σ₁={result_1d['svd_disp'][0]:.3e}, σ₂={result_1d['svd_disp'][1]:.3e}")
    
    # Check y-authority
    yA_norm_1d = result_1d["force_norms"][1]  # yA is index 1
    yB_norm_1d = result_1d["force_norms"][3]  # yB is index 3
    print(f"\nyA sensitivity: {yA_norm_1d:.3e}")
    print(f"yB sensitivity: {yB_norm_1d:.3e}")
    
    # ===== Test with NEW 2D forcing =====
    print("\n" + "-" * 70)
    print("2D FORCING (Stage A upgrade - use_2d_forcing=True)")
    print("-" * 70)
    
    cfg_2d = EvaluatorConfig(
        sigma_x=0.10e-3,
        sigma_y=0.15e-3,  # Larger sigma_y for reasonable coupling at y=0.05mm
        bottom_band=0.25e-3,
        dt=5e-3,
        viscosity=1e-3,
        alpha_g=1e6,
        max_step=0.05e-3,
        use_2d_forcing=True,  # STAGE A UPGRADE
    )
    ev_2d = BottomFootprint25DEvaluator(domain, medium, particle, cfg_2d)
    
    result_2d = probe_jacobian(ev_2d, test_x, test_y, u0, "2D Forcing")
    
    print("\nForce Jacobian ||dF/du|| by dimension:")
    for i, (lbl, val) in enumerate(zip(result_2d["dim_labels"], result_2d["force_norms"])):
        marker = "⚠️ ZERO" if val < 1e-20 else ""
        print(f"  {lbl:6s}: {val:.3e} {marker}")
    
    print(f"\nForce Jacobian SVD: σ₁={result_2d['svd_force'][0]:.3e}, σ₂={result_2d['svd_force'][1]:.3e}")
    print(f"Disp Jacobian SVD:  σ₁={result_2d['svd_disp'][0]:.3e}, σ₂={result_2d['svd_disp'][1]:.3e}")
    
    # Check y-authority
    yA_norm_2d = result_2d["force_norms"][1]
    yB_norm_2d = result_2d["force_norms"][3]
    print(f"\nyA sensitivity: {yA_norm_2d:.3e}")
    print(f"yB sensitivity: {yB_norm_2d:.3e}")
    
    # ===== Comparison =====
    print("\n" + "=" * 70)
    print("COMPARISON: Before vs After Stage A")
    print("=" * 70)
    
    print("\n| Dimension |    1D (Legacy)   |    2D (Stage A)  | Improvement |")
    print("|-----------|------------------|------------------|-------------|")
    for i, lbl in enumerate(result_1d["dim_labels"]):
        v1 = result_1d["force_norms"][i]
        v2 = result_2d["force_norms"][i]
        if v1 > 1e-20:
            improvement = v2 / v1
            imp_str = f"{improvement:.1f}x"
        elif v2 > 1e-20:
            imp_str = "∞ (restored!)"
        else:
            imp_str = "N/A"
        print(f"| {lbl:9s} | {v1:16.3e} | {v2:16.3e} | {imp_str:11s} |")
    
    # Rank assessment
    rank_1d = np.sum(result_1d["svd_disp"] > 1e-15)
    rank_2d = np.sum(result_2d["svd_disp"] > 1e-15)
    
    print(f"\nDisplacement Jacobian Rank:")
    print(f"  1D (Legacy): {rank_1d} (should be 1)")
    print(f"  2D (Stage A): {rank_2d} (should be ≥2)")
    
    # Success criteria
    print("\n" + "=" * 70)
    print("VERIFICATION RESULTS")
    print("=" * 70)
    
    y_authority_restored = (yA_norm_2d > 1e-15 or yB_norm_2d > 1e-15)
    rank_improved = rank_2d >= 2
    
    if y_authority_restored:
        print("✅ Y-control authority RESTORED (yA and/or yB sensitivity is non-zero)")
    else:
        print("❌ Y-control authority NOT restored")
    
    if rank_improved:
        print("✅ Displacement Jacobian rank is ≥2 (x and y independently controllable)")
    else:
        print("❌ Displacement Jacobian rank < 2")
    
    if y_authority_restored and rank_improved:
        print("\n🎉 STAGE A VERIFICATION PASSED 🎉")
    else:
        print("\n⚠️  STAGE A VERIFICATION NEEDS ATTENTION")
    
    # Save results
    REPO = Path(__file__).resolve().parents[1]
    out_dir = REPO / "results" / "stage_a_verification"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(out_dir / "jacobian_force_1d.npy", result_1d["jacobian_force"])
    np.save(out_dir / "jacobian_force_2d.npy", result_2d["jacobian_force"])
    np.save(out_dir / "jacobian_disp_1d.npy", result_1d["jacobian_disp"])
    np.save(out_dir / "jacobian_disp_2d.npy", result_2d["jacobian_disp"])
    
    print(f"\nResults saved to: {out_dir}")


if __name__ == "__main__":
    main()
