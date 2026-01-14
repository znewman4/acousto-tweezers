"""
Comprehensive force computation diagnostic.
STEPS 1-4: Visualize pressure field, check gradients, audit solver.
"""
import sys
sys.path.insert(0, 'src')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tweezers.control import DishDomain, MediumProps, EvaluatorConfig, Control2Pucks, BottomFootprint25DEvaluator
from acousto.force import ParticleProps

print("=" * 80)
print("DIAGNOSTIC: Force Computation Pipeline")
print("=" * 80)

# ============================================================================
# SETUP
# ============================================================================
domain = DishDomain(2e-3, 2e-3, 80, 80)
medium = MediumProps(f=2e6, c0=1500, rho0=1000)
particle = ParticleProps(a=5e-6, rho_p=1050, c_p=2350)
cfg = EvaluatorConfig(sigma_x=0.10e-3, bottom_band=0.25e-3, dt=2e-3, viscosity=1e-3)

ev = BottomFootprint25DEvaluator(domain, medium, particle, cfg)
dx = domain.Lx / (domain.Nx - 1)
dy = domain.Ly / (domain.Ny - 1)
print(f"\nDomain: Lx={domain.Lx*1e3:.2f} mm, Ly={domain.Ly*1e3:.2f} mm")
print(f"Grid: Nx={domain.Nx}, Ny={domain.Ny}, dx={dx*1e6:.2f} um, dy={dy*1e6:.2f} um")
print(f"Medium: f={medium.f/1e6:.1f} MHz, c0={medium.c0} m/s, rho0={medium.rho0} kg/m³")
print(f"Particle: a={particle.a*1e6:.1f} um, rho_p={particle.rho_p} kg/m³, c_p={particle.c_p} m/s")
print(f"Config: sigma_x={cfg.sigma_x*1e3:.3f} mm")

# Reference grid for visualization
x_grid = np.linspace(0, domain.Lx, domain.Nx)
y_grid = np.linspace(0, domain.Ly, domain.Ny)

# ============================================================================
# STEP 1 & 2: PRESSURE FIELD VISUALIZATION & GRADIENTS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 1-2: PRESSURE FIELD & GRADIENT ANALYSIS")
print("=" * 80)

# Use a strong, simple control: single puck at x = Lx/2
u = Control2Pucks(
    xA=domain.Lx / 2.0,  # Center of domain
    yA=0.15e-3,          # Near bottom
    xB=0.0,              # Inactive
    yB=0.15e-3,
    vA=5e-4,             # 0.5 mm/s
    vB=0.0,
    phiA=0.0,
    phiB=0.0
)

print(f"\nTest control: puck A at ({u.xA*1e3:.3f}, {u.yA*1e3:.3f}) mm, vA={u.vA*1e3:.3f} mm/s")

# Solve
field = ev.op.solve_for_bottom_vb(ev.control_to_bottom_vb(u))

p = field.p
p_abs = np.abs(p)
p_real = np.real(p)

print(f"\nPressure field statistics:")
print(f"  |p|: min={p_abs.min():.3e}, max={p_abs.max():.3e}, mean={p_abs.mean():.3e}")
print(f"  real(p): min={p_real.min():.3e}, max={p_real.max():.3e}, mean={p_real.mean():.3e}")
print(f"  imag(p): min={np.imag(p).min():.3e}, max={np.imag(p).max():.3e}, mean={np.imag(p).mean():.3e}")

# Compute gradients via finite differences
grad_x_abs = np.diff(p_abs, axis=1)
grad_y_abs = np.diff(p_abs, axis=0)
grad_x_magnitude = np.abs(grad_x_abs).max()
grad_y_magnitude = np.abs(grad_y_abs).max()

print(f"\nFinite-difference gradients of |p|:")
print(f"  max(∂|p|/∂x) = {grad_x_magnitude:.3e}")
print(f"  max(∂|p|/∂y) = {grad_y_magnitude:.3e}")
print(f"  Total gradient range = {np.ptp(np.abs(grad_x_abs)) + np.ptp(np.abs(grad_y_abs)):.3e}")

if grad_x_magnitude < 1e-10 and grad_y_magnitude < 1e-10:
    print("  ⚠️  WARNING: Pressure field is nearly FLAT! Gradients are ~0.")
    print("      This suggests the Neumann BC is not being applied correctly.")
else:
    print("  ✓ Pressure field has spatial variation (good sign).")

# ============================================================================
# VISUALIZATION
# ============================================================================
print("\nGenerating visualizations...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle(f"Pressure Field Diagnostics (puck at x={u.xA*1e3:.2f} mm)", fontsize=14)

# Row 1: |p| (3 views)
im1 = axes[0, 0].imshow(p_abs, extent=[0, domain.Lx*1e3, 0, domain.Ly*1e3], 
                         origin='lower', cmap='viridis', aspect='auto')
axes[0, 0].set_title(r'$|p|$ (magnitude)')
axes[0, 0].set_xlabel('x (mm)')
axes[0, 0].set_ylabel('y (mm)')
plt.colorbar(im1, ax=axes[0, 0])

# real(p)
im2 = axes[0, 1].imshow(p_real, extent=[0, domain.Lx*1e3, 0, domain.Ly*1e3],
                         origin='lower', cmap='RdBu_r', aspect='auto')
axes[0, 1].set_title('real(p)')
axes[0, 1].set_xlabel('x (mm)')
axes[0, 1].set_ylabel('y (mm)')
plt.colorbar(im2, ax=axes[0, 1])

# Contour of |p|
levels = np.linspace(p_abs.min(), p_abs.max(), 15)
X, Y = np.meshgrid(x_grid*1e3, y_grid*1e3)
cs = axes[0, 2].contour(X, Y, p_abs, levels=levels, cmap='viridis')
axes[0, 2].clabel(cs, inline=True, fontsize=8)
axes[0, 2].set_title(r'contour($|p|$)')
axes[0, 2].set_xlabel('x (mm)')
axes[0, 2].set_ylabel('y (mm)')
axes[0, 2].set_aspect('equal')

# Row 2: Gradients
im3 = axes[1, 0].imshow(np.abs(grad_x_abs), extent=[0, domain.Lx*1e3, 0, domain.Ly*1e3],
                         origin='lower', cmap='hot', aspect='auto')
axes[1, 0].set_title(r'$|\partial|p|/\partial x|$')
axes[1, 0].set_xlabel('x (mm)')
axes[1, 0].set_ylabel('y (mm)')
plt.colorbar(im3, ax=axes[1, 0])

im4 = axes[1, 1].imshow(np.abs(grad_y_abs), extent=[0, domain.Lx*1e3, 0, domain.Ly*1e3],
                         origin='lower', cmap='hot', aspect='auto')
axes[1, 1].set_title(r'$|\partial|p|/\partial y|$')
axes[1, 1].set_xlabel('x (mm)')
axes[1, 1].set_ylabel('y (mm)')
plt.colorbar(im4, ax=axes[1, 1])

# Log scale |p|
p_abs_safe = np.where(p_abs > 1e-20, p_abs, 1e-20)
im5 = axes[1, 2].imshow(np.log10(p_abs_safe), extent=[0, domain.Lx*1e3, 0, domain.Ly*1e3],
                         origin='lower', cmap='viridis', aspect='auto')
axes[1, 2].set_title(r'$\log_{10}(|p|)$')
axes[1, 2].set_xlabel('x (mm)')
axes[1, 2].set_ylabel('y (mm)')
plt.colorbar(im5, ax=axes[1, 2])

plt.tight_layout()
plt.savefig('/home/znewman4/projects/acousto-tweezers/results/diagnose_pressure_field.png', dpi=150)
print("  ✓ Saved: results/diagnose_pressure_field.png")
plt.close()

# ============================================================================
# STEP 4: HARD-TEST SINGLE BOTTOM PIXEL
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: HARD-TEST — SINGLE BOTTOM PIXEL VELOCITY")
print("=" * 80)

Nx = domain.Nx
vb_x_test = np.zeros(Nx, dtype=np.complex128)
vb_x_test[Nx // 2] = 1e-3  # Single spike at center

print(f"\nTest: vb_x with only vb_x[{Nx//2}] = 1e-3, all others zero")

field_test = ev.op.solve_for_bottom_vb(vb_x_test)
p_test = field_test.p
p_test_abs = np.abs(p_test)

print(f"Result: |p| min={p_test_abs.min():.3e}, max={p_test_abs.max():.3e}")
print(f"        Spatial range (max-min) = {p_test_abs.max() - p_test_abs.min():.3e}")

grad_x_test = np.diff(p_test_abs, axis=1)
grad_y_test = np.diff(p_test_abs, axis=0)
print(f"        max(∂|p|/∂x) = {np.abs(grad_x_test).max():.3e}")
print(f"        max(∂|p|/∂y) = {np.abs(grad_y_test).max():.3e}")

if np.abs(grad_x_test).max() < 1e-10 and np.abs(grad_y_test).max() < 1e-10:
    print("        ⚠️  FLAT FIELD: Single pixel forcing produces uniform field!")
    print("             Solver may be collapsing the Neumann BC.")
else:
    print("        ✓ Single-pixel forcing produces spatially varying field (good).")

# Visualize single-pixel test
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Single-Pixel Bottom Forcing Test", fontsize=14)

im1 = axes[0].imshow(p_test_abs, extent=[0, domain.Lx*1e3, 0, domain.Ly*1e3],
                      origin='lower', cmap='viridis', aspect='auto')
axes[0].set_title(r'$|p|$ from single vb spike')
axes[0].axvline(x_grid[Nx//2]*1e3, color='red', linestyle='--', linewidth=2, label='vb spike')
axes[0].set_xlabel('x (mm)')
axes[0].set_ylabel('y (mm)')
axes[0].legend()
plt.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(np.abs(grad_x_test), extent=[0, domain.Lx*1e3, 0, domain.Ly*1e3],
                      origin='lower', cmap='hot', aspect='auto')
axes[1].set_title(r'$|\partial|p|/\partial x|$')
axes[1].set_xlabel('x (mm)')
axes[1].set_ylabel('y (mm)')
plt.colorbar(im2, ax=axes[1])

im3 = axes[2].imshow(np.abs(grad_y_test), extent=[0, domain.Lx*1e3, 0, domain.Ly*1e3],
                      origin='lower', cmap='hot', aspect='auto')
axes[2].set_title(r'$|\partial|p|/\partial y|$')
axes[2].set_xlabel('x (mm)')
axes[2].set_ylabel('y (mm)')
plt.colorbar(im3, ax=axes[2])

plt.tight_layout()
plt.savefig('/home/znewman4/projects/acousto-tweezers/results/diagnose_single_pixel.png', dpi=150)
print(f"✓ Saved: results/diagnose_single_pixel.png")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("DIAGNOSTIC SUMMARY")
print("=" * 80)
print(f"\n1. Pressure field spatial variation:")
print(f"   |p| range: {p_abs.min():.3e} to {p_abs.max():.3e}")
print(f"   ∂|p|/∂x max: {grad_x_magnitude:.3e}")
print(f"   ∂|p|/∂y max: {grad_y_magnitude:.3e}")

if grad_x_magnitude > 1e-10 or grad_y_magnitude > 1e-10:
    print(f"   ✓ FIELD HAS SPATIAL GRADIENTS (Pressure field is OK)")
else:
    print(f"   ❌ FIELD IS FLAT (Neumann BC may be wrong)")

print(f"\n2. Single-pixel forcing response:")
if np.abs(grad_x_test).max() > 1e-10 or np.abs(grad_y_test).max() > 1e-10:
    print(f"   ✓ Single pixel creates spatial variation (Solver is OK)")
else:
    print(f"   ❌ Single pixel produces flat field (Solver issue)")

print(f"\nNext step: Check Gor'kov force computation (STEP 5)")
print("=" * 80)
