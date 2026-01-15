#!/usr/bin/env python3
"""
Quick diagnostic to check force magnitudes and particle motion.
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
        max_step=0.05e-3,
        use_2d_forcing=True,
    )
    ev = BottomFootprint25DEvaluator(domain, medium, particle, cfg)
    
    # Particle at centre
    px, py = 1.0e-3, 1.0e-3
    
    # Transducers straddling particle
    u = Control2Pucks(
        xA=0.6e-3, yA=0.05e-3,
        xB=1.4e-3, yB=0.05e-3,
        vA=5e-4, vB=5e-4,
        phiA=0.0, phiB=np.pi,
    )
    
    # Get field and forces
    vb_x = ev.control_to_forcing_band_vb(u)
    field = ev.op.solve_for_bottom_vb(vb_x)
    U, Fx, Fy = gorkov_potential_and_force_2d(field, particle)
    
    # Raw force at particle position
    fx_raw, fy_raw = bilinear_sample_vec(field.x, field.y, Fx, Fy, px, py)
    
    # Scaled force
    fx_scaled = fx_raw * cfg.alpha_g
    fy_scaled = fy_raw * cfg.alpha_g
    
    # Stokes drag
    gamma = 6.0 * np.pi * cfg.viscosity * particle.a
    
    # Velocity
    vx = fx_scaled / gamma
    vy = fy_scaled / gamma
    
    # Displacement per timestep
    dx = vx * cfg.dt
    dy = vy * cfg.dt
    
    print("Force/Motion Diagnostic")
    print("=" * 60)
    print(f"\nParticle position: ({px*1e3:.2f}, {py*1e3:.2f}) mm")
    print(f"Transducers: A=({u.xA*1e3:.2f}, {u.yA*1e3:.2f}), B=({u.xB*1e3:.2f}, {u.yB*1e3:.2f}) mm")
    print(f"\nForce scaling (alpha_g): {cfg.alpha_g:.0e}")
    print(f"\nRaw Gor'kov force:")
    print(f"  Fx = {fx_raw:.3e} N")
    print(f"  Fy = {fy_raw:.3e} N")
    print(f"  |F| = {np.sqrt(fx_raw**2 + fy_raw**2):.3e} N")
    print(f"\nScaled force (alpha_g applied):")
    print(f"  Fx = {fx_scaled:.3e} N")
    print(f"  Fy = {fy_scaled:.3e} N")
    print(f"  |F| = {np.sqrt(fx_scaled**2 + fy_scaled**2):.3e} N")
    print(f"\nStokes drag coefficient: gamma = {gamma:.3e} Ns/m")
    print(f"\nVelocity:")
    print(f"  vx = {vx:.3e} m/s = {vx*1e3:.3f} mm/s")
    print(f"  vy = {vy:.3e} m/s = {vy*1e3:.3f} mm/s")
    print(f"\nDisplacement per step (dt={cfg.dt*1e3:.1f}ms):")
    print(f"  dx = {dx:.3e} m = {dx*1e6:.2f} µm")
    print(f"  dy = {dy:.3e} m = {dy*1e6:.2f} µm")
    print(f"  |d| = {np.sqrt(dx**2 + dy**2)*1e6:.2f} µm")
    
    # Check max force in field
    force_mag = np.sqrt(Fx**2 + Fy**2)
    print(f"\nMax raw force in field: {np.max(force_mag):.3e} N")
    print(f"Max scaled force: {np.max(force_mag) * cfg.alpha_g:.3e} N")


if __name__ == "__main__":
    main()
