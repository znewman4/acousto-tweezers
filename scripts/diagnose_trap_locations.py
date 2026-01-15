#!/usr/bin/env python3
"""
Diagnose where acoustic traps form for different transducer configurations.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from acousto.force import ParticleProps, gorkov_potential_and_force_2d
from tweezers.control import (
    DishDomain, MediumProps, EvaluatorConfig,
    Control2Pucks, BottomFootprint25DEvaluator,
)
from acousto.analysis import find_trap_center


def main():
    # Setup
    domain = DishDomain(Lx=2e-3, Ly=2e-3, Nx=120, Ny=120)
    medium = MediumProps(f=2e6, c0=1500.0, rho0=1000.0, loss_eta=1e-3, kz=0.0, coupling_alpha=1.0)
    particle = ParticleProps(a=5e-6, rho_p=1050.0, c_p=2350.0)
    
    cfg = EvaluatorConfig(
        sigma_x=0.10e-3,
        sigma_y=0.15e-3,
        bottom_band=0.25e-3,
        dt=5e-3,
        viscosity=1e-3,
        alpha_g=1e6,
        max_step=0.05e-3,
        use_2d_forcing=True,
    )
    ev = BottomFootprint25DEvaluator(domain, medium, particle, cfg)
    
    # Scan transducer x-positions and find trap locations
    print("Scanning transducer positions to map trap locations...")
    
    xA_values = np.linspace(0.3e-3, 0.8e-3, 6)
    xB_values = np.linspace(1.2e-3, 1.7e-3, 6)
    
    results = []
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (xA, xB) in enumerate(zip(xA_values, xB_values)):
        u = Control2Pucks(
            xA=float(xA), yA=0.05e-3,
            xB=float(xB), yB=0.05e-3,
            vA=5e-4, vB=5e-4,
            phiA=0.0, phiB=np.pi,  # Out of phase
        )
        
        # Get field
        vb_x = ev.control_to_forcing_band_vb(u)
        field = ev.op.solve_for_bottom_vb(vb_x)
        U, Fx, Fy = gorkov_potential_and_force_2d(field, particle)
        
        # Find trap centre
        search_x = (u.xA + u.xB) / 2.0
        search_y = domain.Ly * 0.5
        trap_result = find_trap_center(
            field.x, field.y, U, Fx, Fy,
            particle_x=search_x, particle_y=search_y,
            search_radius=0.5e-3,
            max_gradient_steps=30,
        )
        
        results.append({
            "xA": xA, "xB": xB,
            "trap_x": trap_result.x, "trap_y": trap_result.y,
            "is_stable": trap_result.is_stable,
            "stiffness": trap_result.stiffness_eigvals,
        })
        
        # Plot
        ax = axes[idx]
        x_mm = field.x * 1e3
        y_mm = field.y * 1e3
        X, Y = np.meshgrid(x_mm, y_mm)
        
        ax.contourf(X, Y, U * 1e15, levels=20, cmap="viridis")
        
        # Mark transducers
        ax.scatter([u.xA * 1e3, u.xB * 1e3], [u.yA * 1e3, u.yB * 1e3], 
                   marker="^", s=100, c="yellow", edgecolors="black", label="transducers")
        
        # Mark trap
        if trap_result.is_stable:
            ax.scatter(trap_result.x * 1e3, trap_result.y * 1e3,
                       marker="x", s=200, c="red", linewidths=3, label="trap")
        else:
            ax.scatter(trap_result.x * 1e3, trap_result.y * 1e3,
                       marker="x", s=200, c="orange", linewidths=3, label="trap (unstable)")
        
        # Target region
        ax.axhline(y=1.1, color="white", linestyle="--", alpha=0.5, label="target y=1.1mm")
        
        ax.set_title(f"xA={xA*1e3:.2f}, xB={xB*1e3:.2f}mm\n"
                     f"trap=({trap_result.x*1e3:.2f}, {trap_result.y*1e3:.2f})mm")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_aspect("equal")
    
    plt.tight_layout()
    out_path = Path(__file__).parents[1] / "results" / "trap_location_scan.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    print(f"Saved: {out_path}")
    plt.close()
    
    # Summary
    print("\nTrap Location Summary:")
    print("-" * 80)
    print("| xA (mm) | xB (mm) | trap_x (mm) | trap_y (mm) | stable | eig1      | eig2      |")
    print("-" * 80)
    for r in results:
        stable_str = "✓" if r["is_stable"] else "✗"
        eig1, eig2 = r["stiffness"] if len(r["stiffness"]) == 2 else (0, 0)
        print(f"| {r['xA']*1e3:7.2f} | {r['xB']*1e3:7.2f} | "
              f"{r['trap_x']*1e3:11.3f} | {r['trap_y']*1e3:11.3f} | {stable_str:6s} | "
              f"{eig1:9.2e} | {eig2:9.2e} |")
    print("-" * 80)
    
    # Check if any traps form at y ≈ 1.1mm (target region)
    y_target = 1.1e-3
    close_traps = [r for r in results if abs(r["trap_y"] - y_target) < 0.1e-3]
    
    if close_traps:
        print(f"\n✅ Found {len(close_traps)} trap(s) near target y={y_target*1e3:.1f}mm")
    else:
        print(f"\n⚠️  No traps found near target y={y_target*1e3:.1f}mm")
        print("   The 2-transducer bottom forcing may not be able to create traps at this height.")
        trap_ys = [r["trap_y"]*1e3 for r in results]
        print(f"   Observed trap y-range: [{min(trap_ys):.2f}, {max(trap_ys):.2f}] mm")


if __name__ == "__main__":
    main()
