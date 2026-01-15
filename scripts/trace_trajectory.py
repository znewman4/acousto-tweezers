#!/usr/bin/env python3
"""
Trace force at particle for a simulated trajectory.
"""

import numpy as np
from acousto.force import ParticleProps, gorkov_potential_and_force_2d, bilinear_sample_vec
from tweezers.control import (
    DishDomain, MediumProps, EvaluatorConfig,
    Control2Pucks, BottomFootprint25DEvaluator,
)


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
        alpha_g=1e3,
        max_step=0.05e-3,
        use_2d_forcing=True,
    )
    ev = BottomFootprint25DEvaluator(domain, medium, particle, cfg)
    
    # Simulate particle starting at x=0.56mm
    px, py = 0.56e-3, 1.1e-3
    v_amp = 0.05
    
    print("Simulating particle trajectory with guided transducers")
    print("=" * 80)
    print(f"Particle starts at ({px*1e3:.2f}, {py*1e3:.2f}) mm")
    print()
    
    gamma = 6.0 * np.pi * cfg.viscosity * particle.a
    
    for step in range(20):
        # Target moves from 0.6mm to 1.4mm over 20 steps
        target_x = 0.6e-3 + (1.4e-3 - 0.6e-3) * step / 19
        
        # Guided transducers: straddle target with 0.8mm separation
        sep = 0.8e-3
        xA = target_x - sep/2
        xB = target_x + sep/2
        
        # Clip to domain
        xA = np.clip(xA, 0.1e-3, 1.9e-3)
        xB = np.clip(xB, 0.1e-3, 1.9e-3)
        
        u = Control2Pucks(
            xA=xA, yA=0.05e-3,
            xB=xB, yB=0.05e-3,
            vA=v_amp, vB=v_amp,
            phiA=0.0, phiB=np.pi,
        )
        
        # Get force at particle position
        vb_x = ev.control_to_forcing_band_vb(u)
        field = ev.op.solve_for_bottom_vb(vb_x)
        U, Fx, Fy = gorkov_potential_and_force_2d(field, particle)
        fx, fy = bilinear_sample_vec(field.x, field.y, 
                                      Fx * cfg.alpha_g, Fy * cfg.alpha_g, px, py)
        
        # Calculate displacement
        vx = fx / gamma
        dx = vx * cfg.dt
        
        # Apply displacement (with step limiting)
        dr = np.sqrt(dx**2 + (fy/gamma*cfg.dt)**2)
        if dr > cfg.max_step:
            scale = cfg.max_step / dr
            dx *= scale
        
        direction = "→" if dx > 0 else "←"
        print(f"Step {step+1:2d}: tgt={target_x*1e3:.2f}mm, trans center={(xA+xB)/2*1e3:.2f}mm, "
              f"particle={px*1e3:.2f}mm, Fx={fx:.2e}N, dx={dx*1e6:.1f}µm {direction}")
        
        # Update particle position
        px += dx
        px = np.clip(px, 0.1e-3, 1.9e-3)
    
    print()
    print(f"Final particle position: {px*1e3:.2f} mm")
    print(f"Final target: {target_x*1e3:.2f} mm")
    print(f"Final error: {abs(px - target_x)*1e3:.2f} mm")


if __name__ == "__main__":
    main()
