"""
STEP 5: Audit Gor'kov force computation.
Check pressure gradients, velocity magnitudes, energy densities, and forces.
"""
import sys
sys.path.insert(0, 'src')

import numpy as np
import matplotlib.pyplot as plt
from tweezers.control import DishDomain, MediumProps, EvaluatorConfig, Control2Pucks, BottomFootprint25DEvaluator
from acousto.force import ParticleProps, gorkov_potential_and_force_2d

print("=" * 80)
print("STEP 5: GOR'KOV FORCE COMPUTATION AUDIT")
print("=" * 80)

# ============================================================================
# SETUP
# ============================================================================
domain = DishDomain(2e-3, 2e-3, 80, 80)
medium = MediumProps(f=2e6, c0=1500, rho0=1000)
particle = ParticleProps(a=5e-6, rho_p=1050, c_p=2350)
cfg = EvaluatorConfig(sigma_x=0.10e-3, bottom_band=0.25e-3, dt=2e-3, viscosity=1e-3)

ev = BottomFootprint25DEvaluator(domain, medium, particle, cfg)

# Test control
u = Control2Pucks(
    xA=domain.Lx / 2.0,
    yA=0.15e-3,
    xB=0.0,
    yB=0.15e-3,
    vA=5e-4,
    vB=0.0,
    phiA=0.0,
    phiB=0.0
)

# Solve
field = ev.op.solve_for_bottom_vb(ev.control_to_bottom_vb(u))

# Compute forces (with velocity output)
U, Fx, Fy, vx, vy = gorkov_potential_and_force_2d(field, particle, return_velocity=True)

print("\n" + "=" * 80)
print("INTERMEDIATE COMPUTATIONS")
print("=" * 80)

p = field.p
omega = field.omega
rho0 = field.rho0
c0 = field.c0

# Pressure field
print(f"\nPressure field:")
print(f"  |p| range: {np.abs(p).min():.3e} to {np.abs(p).max():.3e} Pa")
print(f"  mean|p|: {np.abs(p).mean():.3e} Pa")

# Pressure gradients (manual computation)
dx = field.x[1] - field.x[0]
dy = field.y[1] - field.y[0]
dpdy, dpdx = np.gradient(p, dy, dx, edge_order=2)

print(f"\nPressure gradients:")
print(f"  max|∂p/∂x| = {np.abs(dpdx).max():.3e} Pa/m")
print(f"  max|∂p/∂y| = {np.abs(dpdy).max():.3e} Pa/m")
print(f"  mean|∂p/∂x| = {np.abs(dpdx).mean():.3e} Pa/m")
print(f"  mean|∂p/∂y| = {np.abs(dpdy).mean():.3e} Pa/m")

# Velocity
print(f"\nVelocity field (from ∇p):")
print(f"  |v_x| range: {np.abs(vx).min():.3e} to {np.abs(vx).max():.3e} m/s")
print(f"  |v_y| range: {np.abs(vy).min():.3e} to {np.abs(vy).max():.3e} m/s")
print(f"  |v|^2 range: {(np.abs(vx)**2 + np.abs(vy)**2).min():.3e} to {(np.abs(vx)**2 + np.abs(vy)**2).max():.3e}")

# Energy densities
kappa0 = 1.0 / (rho0 * c0**2)
kappap = 1.0 / (particle.rho_p * particle.c_p**2)
f1 = 1.0 - (kappap / kappa0)
f2 = 2.0 * (particle.rho_p - rho0) / (2.0 * particle.rho_p + rho0)

E_pot = 0.25 * (np.abs(p) ** 2) * kappa0
v2 = (np.abs(vx) ** 2 + np.abs(vy) ** 2)
E_kin = 0.25 * rho0 * v2

print(f"\nContrast factors:")
print(f"  f1 (compressibility) = {f1:.6f}")
print(f"  f2 (density) = {f2:.6f}")

print(f"\nEnergy densities:")
print(f"  E_pot = 0.25 |p|^2 * kappa0:")
print(f"    range: {E_pot.min():.3e} to {E_pot.max():.3e} J/m³")
print(f"    mean: {E_pot.mean():.3e} J/m³")
print(f"  E_kin = 0.25 rho0 |v|^2:")
print(f"    range: {E_kin.min():.3e} to {E_kin.max():.3e} J/m³")
print(f"    mean: {E_kin.mean():.3e} J/m³")

# Gor'kov potential
V = (4.0 / 3.0) * np.pi * (particle.a ** 3)
print(f"\nGor'kov potential U = V * (f1 * E_pot - 1.5 * f2 * E_kin):")
print(f"  V (particle volume) = {V:.3e} m³")
print(f"  f1 * E_pot range: {(f1 * E_pot).min():.3e} to {(f1 * E_pot).max():.3e}")
print(f"  1.5 * f2 * E_kin range: {(1.5 * f2 * E_kin).min():.3e} to {(1.5 * f2 * E_kin).max():.3e}")
print(f"  U range: {U.min():.3e} to {U.max():.3e} J")
print(f"  mean(U): {U.mean():.3e} J")

# U gradients
dUdy, dUdx = np.gradient(U, dy, dx, edge_order=2)
print(f"\nGor'kov potential gradients (∇U):")
print(f"  |∂U/∂x| range: {np.abs(dUdx).min():.3e} to {np.abs(dUdx).max():.3e}")
print(f"  |∂U/∂y| range: {np.abs(dUdy).min():.3e} to {np.abs(dUdy).max():.3e}")
print(f"  mean|∂U/∂x| = {np.abs(dUdx).mean():.3e}")
print(f"  mean|∂U/∂y| = {np.abs(dUdy).mean():.3e}")

# Forces
print(f"\nForces (F = -∇U):")
print(f"  Fx range: {Fx.min():.3e} to {Fx.max():.3e} N")
print(f"  Fy range: {Fy.min():.3e} to {Fy.max():.3e} N")
print(f"  mean|Fx| = {np.abs(Fx).mean():.3e} N")
print(f"  mean|Fy| = {np.abs(Fy).mean():.3e} N")
print(f"  |F| = sqrt(Fx² + Fy²):")
F_mag = np.sqrt(Fx**2 + Fy**2)
print(f"    range: {F_mag.min():.3e} to {F_mag.max():.3e} N")
print(f"    mean: {F_mag.mean():.3e} N")

# ============================================================================
# DIAGNOSE ZERO FORCES
# ============================================================================
print("\n" + "=" * 80)
print("DIAGNOSIS")
print("=" * 80)

# Check if E_kin dominates
e_pot_contrib = np.abs(f1 * E_pot)
e_kin_contrib = np.abs(1.5 * f2 * E_kin)

print(f"\nEnergy balance:")
print(f"  Potential contribution |f1*E_pot|: {e_pot_contrib.max():.3e} (max)")
print(f"  Kinetic contribution |1.5*f2*E_kin|: {e_kin_contrib.max():.3e} (max)")

if e_kin_contrib.max() > e_pot_contrib.max():
    print(f"  ⚠️  KINETIC ENERGY DOMINATES! Check if this is physical.")
else:
    print(f"  ✓ Potential energy dominates (typical).")

# Check if U is nearly uniform (would give ~0 gradients)
U_range = U.max() - U.min()
U_mean = np.abs(U.mean())
print(f"\nGor'kov potential uniformity:")
print(f"  U range (max-min): {U_range:.3e}")
print(f"  U mean (absolute): {U_mean:.3e}")
if U_mean > 0:
    rel_variation = U_range / U_mean
    print(f"  Relative variation: {rel_variation:.3e}")
    if rel_variation < 0.01:
        print(f"  ⚠️  U IS NEARLY UNIFORM (rel variation < 1%)")
        print(f"      This would explain ~0 forces!")
    else:
        print(f"  ✓ U has reasonable spatial variation ({rel_variation*100:.1f}%)")

# Check if force magnitudes are realistic
# Stokes drag: F_stokes = 6πμ*a*v_particle
a = particle.a
mu = 1e-3  # Water viscosity
typical_particle_velocity = 1e-4  # m/s (hoping for ~100 um/s)
typical_stokes_force = 6 * np.pi * mu * a * typical_particle_velocity
print(f"\nForce scale check:")
print(f"  Typical Stokes drag (6πμ*a*v_p) for v_p={typical_particle_velocity*1e3:.1f} mm/s:")
print(f"    F_stokes ≈ {typical_stokes_force:.3e} N")
print(f"  Actual force observed:")
print(f"    max|F| = {F_mag.max():.3e} N")
print(f"  Ratio (max|F| / F_stokes): {F_mag.max() / typical_stokes_force if typical_stokes_force > 0 else 0:.3e}")

if F_mag.max() < typical_stokes_force * 0.01:
    print(f"  ⚠️  FORCES ARE TOO WEAK! (< 1% of expected)")
elif F_mag.max() > typical_stokes_force * 10:
    print(f"  ✓ Forces are strong (> 10x Stokes drag)")
else:
    print(f"  ✓ Forces are reasonable scale")

# ============================================================================
# VISUALIZATION
# ============================================================================
print("\nGenerating detailed force visualizations...")
fig, axes = plt.subplots(2, 4, figsize=(18, 10))
fig.suptitle("Gor'kov Force Computation Audit", fontsize=14)

extent = [0, domain.Lx*1e3, 0, domain.Ly*1e3]
x_grid = np.linspace(0, domain.Lx, domain.Nx)
y_grid = np.linspace(0, domain.Ly, domain.Ny)

# Row 1: Energy densities
im0 = axes[0, 0].imshow(E_pot, extent=extent, origin='lower', cmap='viridis', aspect='auto')
axes[0, 0].set_title(r'$E_{pot} = \frac{1}{4}|p|^2 \kappa_0$')
axes[0, 0].set_ylabel('y (mm)')
plt.colorbar(im0, ax=axes[0, 0])

im1 = axes[0, 1].imshow(E_kin, extent=extent, origin='lower', cmap='viridis', aspect='auto')
axes[0, 1].set_title(r'$E_{kin} = \frac{1}{4}\rho_0 |v|^2$')
plt.colorbar(im1, ax=axes[0, 1])

# Gor'kov potential
im2 = axes[0, 2].imshow(U, extent=extent, origin='lower', cmap='RdBu_r', aspect='auto')
axes[0, 2].set_title(r'Gor\'kov potential $U$')
plt.colorbar(im2, ax=axes[0, 2])

# U in log scale
U_abs = np.abs(U)
U_abs_safe = np.where(U_abs > 1e-20, U_abs, 1e-20)
im3 = axes[0, 3].imshow(np.log10(U_abs_safe), extent=extent, origin='lower', cmap='viridis', aspect='auto')
axes[0, 3].set_title(r'$\log_{10}(|U|)$')
plt.colorbar(im3, ax=axes[0, 3])

# Row 2: Force components
im4 = axes[1, 0].imshow(Fx, extent=extent, origin='lower', cmap='RdBu_r', aspect='auto')
axes[1, 0].set_title(r'Force $F_x = -\frac{\partial U}{\partial x}$')
axes[1, 0].set_xlabel('x (mm)')
axes[1, 0].set_ylabel('y (mm)')
plt.colorbar(im4, ax=axes[1, 0])

im5 = axes[1, 1].imshow(Fy, extent=extent, origin='lower', cmap='RdBu_r', aspect='auto')
axes[1, 1].set_title(r'Force $F_y = -\frac{\partial U}{\partial y}$')
axes[1, 1].set_xlabel('x (mm)')
plt.colorbar(im5, ax=axes[1, 1])

# Force magnitude
im6 = axes[1, 2].imshow(F_mag, extent=extent, origin='lower', cmap='hot', aspect='auto')
axes[1, 2].set_title(r'Force magnitude $|F|$')
axes[1, 2].set_xlabel('x (mm)')
plt.colorbar(im6, ax=axes[1, 2])

# Force magnitude in log
F_mag_safe = np.where(F_mag > 1e-20, F_mag, 1e-20)
im7 = axes[1, 3].imshow(np.log10(F_mag_safe), extent=extent, origin='lower', cmap='hot', aspect='auto')
axes[1, 3].set_title(r'$\log_{10}(|F|)$ [N]')
axes[1, 3].set_xlabel('x (mm)')
plt.colorbar(im7, ax=axes[1, 3])

plt.tight_layout()
plt.savefig('/home/znewman4/projects/acousto-tweezers/results/diagnose_gorkov.png', dpi=150)
print("  ✓ Saved: results/diagnose_gorkov.png")
plt.close()

# ============================================================================
# ADDITIONAL: Check specific particle locations
# ============================================================================
print("\n" + "=" * 80)
print("FORCE AT SPECIFIC PARTICLE LOCATIONS")
print("=" * 80)

test_positions = [
    (domain.Lx/2, domain.Ly/2, "center"),
    (domain.Lx/2, 0.1e-3, "near bottom"),
    (domain.Lx/2, 1.9e-3, "near top"),
    (domain.Lx/4, domain.Ly/2, "left of center"),
    (3*domain.Lx/4, domain.Ly/2, "right of center"),
]

for x_pos, y_pos, label in test_positions:
    # Find nearest grid point
    ix = np.argmin(np.abs(x_grid - x_pos))
    iy = np.argmin(np.abs(y_grid - y_pos))
    
    print(f"\nPosition ({x_pos*1e3:.3f}, {y_pos*1e3:.3f}) mm [{label}]:")
    print(f"  Grid: [{iy}, {ix}]")
    print(f"  |p| = {np.abs(p[iy, ix]):.3e} Pa")
    print(f"  U = {U[iy, ix]:.3e} J")
    print(f"  Fx = {Fx[iy, ix]:.3e} N")
    print(f"  Fy = {Fy[iy, ix]:.3e} N")
    print(f"  |F| = {F_mag[iy, ix]:.3e} N")

print("\n" + "=" * 80)
