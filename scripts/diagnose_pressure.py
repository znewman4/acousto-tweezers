#!/usr/bin/env python3
"""
Check pressure field magnitudes to diagnose weak forces.
"""

import numpy as np
from acousto.force import ParticleProps
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
    
    # Standard transducer config
    u = Control2Pucks(
        xA=0.6e-3, yA=0.05e-3,
        xB=1.4e-3, yB=0.05e-3,
        vA=5e-4, vB=5e-4,
        phiA=0.0, phiB=np.pi,
    )
    
    # Get field
    vb_x = ev.control_to_forcing_band_vb(u)
    field = ev.op.solve_for_bottom_vb(vb_x)
    
    # Pressure analysis
    p = field.p  # complex pressure field
    p_mag = np.abs(p)
    
    print("Pressure Field Diagnostic")
    print("=" * 60)
    print(f"\nTransducer velocity: vA = vB = {u.vA:.2e} m/s")
    print(f"Medium: c0 = {medium.c0} m/s, rho0 = {medium.rho0} kg/m³")
    print(f"Frequency: {medium.f/1e6:.1f} MHz")
    print(f"\nPressure field statistics:")
    print(f"  Max |p|: {np.max(p_mag):.3e} Pa")
    print(f"  Mean |p|: {np.mean(p_mag):.3e} Pa")
    print(f"  Min |p|: {np.min(p_mag):.3e} Pa")
    
    # Expected pressure from plane wave: p = rho * c * v
    p_expected = medium.rho0 * medium.c0 * u.vA
    print(f"\nExpected plane wave pressure: p = ρ₀ c₀ v = {p_expected:.3e} Pa")
    print(f"Ratio (max/expected): {np.max(p_mag)/p_expected:.3f}")
    
    # Reference: realistic acoustic tweezers use ~MPa pressures
    print(f"\n--- Reference values for acoustic tweezers ---")
    print(f"Typical MHz ultrasound: 0.1 - 1 MPa (10⁵ - 10⁶ Pa)")
    print(f"Our max pressure: {np.max(p_mag):.2e} Pa ({np.max(p_mag)/1e6:.3e} MPa)")
    
    # To achieve realistic pressures, what velocity would we need?
    target_pressure = 0.1e6  # 0.1 MPa
    needed_velocity = target_pressure / (medium.rho0 * medium.c0)
    print(f"\nTo achieve {target_pressure/1e6:.1f} MPa:")
    print(f"  Need transducer velocity: v = {needed_velocity:.2e} m/s")
    print(f"  Current velocity: {u.vA:.2e} m/s")
    print(f"  Ratio needed/current: {needed_velocity/u.vA:.0f}x")


if __name__ == "__main__":
    main()
