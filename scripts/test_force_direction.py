#!/usr/bin/env python3
"""
Quick test: Can we move a particle by shifting transducers?
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
        alpha_g=1e6,
        max_step=0.1e-3,  # Allow larger steps
        use_2d_forcing=True,
    )
    ev = BottomFootprint25DEvaluator(domain, medium, particle, cfg)
    
    # Test: particle at x=0.6mm, transducers at different x-positions
    py = 1.0e-3  # Fixed y
    
    print("Testing force direction vs transducer positions")
    print("=" * 80)
    print(f"Particle at y={py*1e3:.1f}mm, vA=vB=0.05 m/s")
    print()
    
    v_amp = 0.05
    
    # Scan transducer centre positions
    for trans_center_x in [0.5e-3, 0.7e-3, 0.9e-3, 1.1e-3, 1.3e-3]:
        sep = 0.4e-3  # 0.4mm separation
        xA = trans_center_x - sep
        xB = trans_center_x + sep
        
        # Particle position - start at 0.6mm
        px = 0.6e-3
        
        u = Control2Pucks(
            xA=xA, yA=0.05e-3,
            xB=xB, yB=0.05e-3,
            vA=v_amp, vB=v_amp,
            phiA=0.0, phiB=np.pi,
        )
        
        # Get force at particle
        vb_x = ev.control_to_forcing_band_vb(u)
        field = ev.op.solve_for_bottom_vb(vb_x)
        U, Fx, Fy = gorkov_potential_and_force_2d(field, particle)
        fx, fy = bilinear_sample_vec(field.x, field.y, 
                                      Fx * cfg.alpha_g, Fy * cfg.alpha_g, px, py)
        
        # Stokes drag
        gamma = 6.0 * np.pi * cfg.viscosity * particle.a
        vx = fx / gamma
        dx_per_step = vx * cfg.dt
        
        # Direction
        direction = "→ RIGHT" if fx > 0 else "← LEFT" if fx < 0 else "   (none)"
        
        print(f"Transducers: xA={xA*1e3:.1f}mm, xB={xB*1e3:.1f}mm (center={trans_center_x*1e3:.1f}mm)")
        print(f"  Particle at x={px*1e3:.1f}mm: Fx={fx:.2e} N, "
              f"dx/step={dx_per_step*1e6:.1f}µm {direction}")
        
        # Also check if particle at center
        px2 = 1.0e-3
        fx2, _ = bilinear_sample_vec(field.x, field.y, 
                                      Fx * cfg.alpha_g, Fy * cfg.alpha_g, px2, py)
        vx2 = fx2 / gamma
        dx2_per_step = vx2 * cfg.dt
        direction2 = "→ RIGHT" if fx2 > 0 else "← LEFT" if fx2 < 0 else "   (none)"
        print(f"  Particle at x={px2*1e3:.1f}mm: Fx={fx2:.2e} N, "
              f"dx/step={dx2_per_step*1e6:.1f}µm {direction2}")
        print()


if __name__ == "__main__":
    main()
