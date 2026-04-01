#!/usr/bin/env python3
"""
Test the Gorkov force field of the bridge lens reconstruction to see if it
would push particle A toward particle B.

Loads the reconstructed pressure field from the IASA lens design and computes:
  1. Gor'kov potential U(x,y)
  2. Force field F = -∇U
  3. Checks if force at A points toward B
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# ═══════════════════════════════════════════════════════════════════
# Physical constants
# ═══════════════════════════════════════════════════════════════════
C_WATER = 1484.0          # m/s
F_HZ    = 2.15e6          # Hz  (matching the IASA run)
OMEGA   = 2.0 * np.pi * F_HZ
RHO0    = 997.0            # kg/m³  water density

# Particle properties — polystyrene microsphere
RHO_P   = 1050.0          # kg/m³
C_P     = 2350.0          # m/s
A_PART  = 50.0e-6         # particle radius 50 µm (100 µm diameter)

# Gor'kov contrast factors
KAPPA_W = 1.0 / (RHO0 * C_WATER**2)
KAPPA_P = 1.0 / (RHO_P * C_P**2)
F1 = 1.0 - KAPPA_P / KAPPA_W
F2 = 2.0 * (RHO_P - RHO0) / (2.0 * RHO_P + RHO0)

print(f"[constants] F1={F1:.4f}, F2={F2:.4f}, a={A_PART*1e6:.1f}µm")

# ═══════════════════════════════════════════════════════════════════
# Most recent inverse bridge run
# ═══════════════════════════════════════════════════════════════════
RESULTS_DIR = PROJECT_ROOT / "results" / "dev"
bridge_results = sorted(RESULTS_DIR.glob("inverse_bridge_pressure_lens_replica_*"))
if not bridge_results:
    raise FileNotFoundError("No inverse_bridge_pressure_lens_replica_* results found")

latest_run = bridge_results[-1]
print(f"[load] using {latest_run.name}")

# Load reconstruction
npz_path = latest_run / "bridge_inverse_replica_fields.npz"
if not npz_path.exists():
    raise FileNotFoundError(f"{npz_path} not found")

d = np.load(npz_path)
recon_amp = d["recon_amp"]           # (400, 400) real
lens_field = d["lens_field"]         # (400, 400) complex
aperture_mask = d["aperture_mask"]   # (400, 400) bool

# Load manifest for trap positions and config
manifest_path = latest_run / "bridge_inverse_replica_manifest.json"
with open(manifest_path) as f:
    manifest = json.load(f)

config = manifest["config"]
metrics = manifest["metrics"]

print(f"\n[config] focal_distance_mm={config['focal_distance_mm']:.2f}")
print(f"[config] n_grid={config['n_grid']}, dx={config['transducer_diameter_mm']*1e3/(config['n_grid']*1e3):.2f} µm/px")

# Load the original bridge NPZ to get trap positions
bridge_npz_path = PROJECT_ROOT / "results" / "dev" / "bridge_pressure_field_standalone_scaled" / "bridge_pressure_fields.npz"
if bridge_npz_path.exists():
    d_bridge = np.load(bridge_npz_path)
    traps_raw = d_bridge["traps_m"].astype(float)
    idx_a = int(d_bridge["idx_a"])
    idx_b = int(d_bridge["idx_b"])
    x_full = d_bridge["x_full"].astype(float)
    y_full = d_bridge["y_full"].astype(float)
    
    # Get field centre
    x_center = 0.5 * (x_full[0] + x_full[-1])
    y_center = 0.5 * (y_full[0] + y_full[-1])
    
    # Trap positions relative to IASA grid centre
    a_pos_m = traps_raw[idx_a][:2] - np.array([x_center, y_center])
    b_pos_m = traps_raw[idx_b][:2] - np.array([x_center, y_center])
    print(f"\n[traps] A position: {a_pos_m*1e3} mm")
    print(f"[traps] B position: {b_pos_m*1e3} mm")
    print(f"[traps] A→B distance: {np.linalg.norm(b_pos_m - a_pos_m)*1e3:.4f} mm")
else:
    print("[warn] bridge NPZ not found; will skip trap-based analysis")
    a_pos_m = None
    b_pos_m = None

# ═══════════════════════════════════════════════════════════════════
# Gorkov potential and force
# ═══════════════════════════════════════════════════════════════════

def compute_gorkov(p_field, dx):
    """
    Gor'kov potential on a 2D grid.

    U = (f1 / (2 * rho * c²)) |p|² - (3*f2 / (4*omega²*rho)) |grad(p)|²

    Returns U (Ny, Nx).
    """
    p_abs2 = np.abs(p_field)**2

    dp_dx = np.gradient(p_field, dx, axis=1)
    dp_dy = np.gradient(p_field, dx, axis=0)
    grad_p_abs2 = np.abs(dp_dx)**2 + np.abs(dp_dy)**2

    coeff_p = F1 / (2.0 * RHO0 * C_WATER**2)
    coeff_k = 3.0 * F2 / (4.0 * OMEGA**2 * RHO0)

    U = coeff_p * p_abs2 - coeff_k * grad_p_abs2
    return U


def compute_force_field(U, dx):
    """F = -grad(U), returned as (Fx, Fy) each (Ny, Nx)."""
    dUdx = np.gradient(U, dx, axis=1)
    dUdy = np.gradient(U, dx, axis=0)
    return -dUdx, -dUdy


# Grid setup
n_grid = config["n_grid"]
aperture_diameter_m = config["transducer_diameter_mm"] * 1e-3
dx = aperture_diameter_m / n_grid

# Build coordinate grids
x_half = aperture_diameter_m / 2.0
y_half = aperture_diameter_m / 2.0
xg = np.linspace(-x_half, x_half, n_grid)
yg = np.linspace(-y_half, y_half, n_grid)

print(f"\n[grid] n_grid={n_grid}, dx={dx*1e6:.2f} µm, aperture={aperture_diameter_m*1e3:.1f} mm")

# Compute Gorkov potential from reconstructed complex pressure
# (Use the propagated lens field, not just amplitude)
print("\n[gorkov] computing Gor'kov potential...")
U = compute_gorkov(lens_field, dx)
Fx, Fy = compute_force_field(U, dx)

print(f"[gorkov] U range: [{U.min():.3e}, {U.max():.3e}] J")
print(f"[gorkov] |F| range: [0, {np.sqrt(Fx**2 + Fy**2).max():.3e}] N")

# ─────────────────────────────────────────────────────────────────
# Analysis: force at trap positions
# ─────────────────────────────────────────────────────────────────

if a_pos_m is not None and b_pos_m is not None:
    n_half = n_grid // 2
    
    # Convert physical positions to grid indices
    def pos_to_idx(pos_m):
        """Convert physical position (m) to grid index."""
        i = int(round(pos_m[1] / dx + n_half))  # y → row
        j = int(round(pos_m[0] / dx + n_half))  # x → col
        i = np.clip(i, 0, n_grid - 1)
        j = np.clip(j, 0, n_grid - 1)
        return (i, j)
    
    a_idx = pos_to_idx(a_pos_m)
    b_idx = pos_to_idx(b_pos_m)
    
    print(f"\n[analysis] trap A grid index: {a_idx}")
    print(f"[analysis] trap B grid index: {b_idx}")
    
    # Force at A
    F_at_A = np.array([Fx[a_idx], Fy[a_idx]])
    F_mag_A = np.linalg.norm(F_at_A)
    
    # Direction from A to B
    AB_vec = b_pos_m - a_pos_m
    AB_dist = np.linalg.norm(AB_vec)
    AB_dir = AB_vec / (AB_dist + 1e-12)
    
    # Angle between F_A and A→B direction
    cos_angle = np.dot(F_at_A / (F_mag_A + 1e-12), AB_dir)
    angle_deg = np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi
    
    print(f"\n[force_at_A]")
    print(f"  F_A = ({F_at_A[0]:.3e}, {F_at_A[1]:.3e}) N")
    print(f"  |F_A| = {F_mag_A:.3e} N")
    print(f"  A→B direction: ({AB_dir[0]:.4f}, {AB_dir[1]:.4f})")
    print(f"  cos(angle) = {cos_angle:.4f}")
    print(f"  angle = {angle_deg:.1f}°")
    
    if angle_deg < 90:
        print(f"  ✓ FORCE POINTS TOWARD B (angle < 90°)")
    else:
        print(f"  ✗ FORCE POINTS AWAY FROM B (angle ≥ 90°)")
    
    # Potential at A and B
    U_A = U[a_idx]
    U_B = U[b_idx]
    dU = U_B - U_A
    
    print(f"\n[potential]")
    print(f"  U_A = {U_A:.3e} J")
    print(f"  U_B = {U_B:.3e} J")
    print(f"  ΔU = U_B - U_A = {dU:.3e} J")
    
    if dU < 0:
        print(f"  ✓ POTENTIAL IS LOWER AT B (particle would drift A→B)")
    else:
        print(f"  ✗ POTENTIAL IS HIGHER AT B (particle would repel from B)")

# ─────────────────────────────────────────────────────────────────
# Visualizations
# ─────────────────────────────────────────────────────────────────

print("\n[figures] saving visualizations...")
out_dir = latest_run  # Save alongside existing results

# 1. Gorkov potential field
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(U * aperture_mask, origin="lower", cmap="RdBu_r", aspect="equal", extent=[-x_half*1e3, x_half*1e3, -y_half*1e3, y_half*1e3])
plt.colorbar(im, ax=ax, label="Potential [J]")
ax.set_title("Gor'kov Potential U(x,y)")
ax.set_xlabel("x [mm]")
ax.set_ylabel("y [mm]")
if a_pos_m is not None:
    ax.plot(a_pos_m[0]*1e3, a_pos_m[1]*1e3, "g*", markersize=15, label="A")
    ax.plot(b_pos_m[0]*1e3, b_pos_m[1]*1e3, "r*", markersize=15, label="B")
    ax.legend()
fig.tight_layout()
fig.savefig(out_dir / "gorkov_potential.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# 2. Force magnitude
F_mag = np.sqrt(Fx**2 + Fy**2)
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(F_mag * aperture_mask, origin="lower", cmap="inferno", aspect="equal", extent=[-x_half*1e3, x_half*1e3, -y_half*1e3, y_half*1e3])
plt.colorbar(im, ax=ax, label="|F| [N]")
ax.set_title("Radiation Force Magnitude")
ax.set_xlabel("x [mm]")
ax.set_ylabel("y [mm]")
if a_pos_m is not None:
    ax.plot(a_pos_m[0]*1e3, a_pos_m[1]*1e3, "g*", markersize=15, label="A")
    ax.plot(b_pos_m[0]*1e3, b_pos_m[1]*1e3, "r*", markersize=15, label="B")
    ax.legend()
fig.tight_layout()
fig.savefig(out_dir / "force_magnitude.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# 3. Force vectors (quiver plot)
stride = 8
fig, ax = plt.subplots(figsize=(10, 8))
ax.imshow(recon_amp * aperture_mask, origin="lower", cmap="gray", aspect="equal", extent=[-x_half*1e3, x_half*1e3, -y_half*1e3, y_half*1e3], vmin=0, vmax=np.percentile(recon_amp[aperture_mask], 99), alpha=0.6)

# Quiver plot of force field
X = xg[::stride] * 1e3
Y = yg[::stride] * 1e3
Fx_plot = Fx[::stride, ::stride]
Fy_plot = Fy[::stride, ::stride]

Q = ax.quiver(X, Y, Fx_plot, Fy_plot, np.sqrt(Fx_plot**2 + Fy_plot**2), cmap="hot", scale=1e-11, scale_units="xy")
plt.colorbar(Q, ax=ax, label="|F| [N]")

ax.set_title("Force Field F = -∇U (overlaid on pressure)")
ax.set_xlabel("x [mm]")
ax.set_ylabel("y [mm]")
ax.set_aspect("equal")

if a_pos_m is not None:
    ax.plot(a_pos_m[0]*1e3, a_pos_m[1]*1e3, "g*", markersize=20, label="A", markeredgecolor="black", markeredgewidth=1.5)
    ax.plot(b_pos_m[0]*1e3, b_pos_m[1]*1e3, "r*", markersize=20, label="B", markeredgecolor="black", markeredgewidth=1.5)
    # Draw A→B arrow
    ax.arrow(a_pos_m[0]*1e3, a_pos_m[1]*1e3,
             (b_pos_m[0]-a_pos_m[0])*1e3*0.9, (b_pos_m[1]-a_pos_m[1])*1e3*0.9,
             head_width=0.05, head_length=0.05, fc="cyan", ec="cyan", alpha=0.7, linewidth=2)
    ax.legend(fontsize=10)

fig.tight_layout()
fig.savefig(out_dir / "force_field_quiver.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"[figures] saved:")
print(f"  - gorkov_potential.png")
print(f"  - force_magnitude.png")
print(f"  - force_field_quiver.png")

# ─────────────────────────────────────────────────────────────────
# Save analysis to JSON
# ─────────────────────────────────────────────────────────────────

analysis = {
    "test_description": "Gorkov force field test: does force at A point toward B?",
    "lens_run": str(latest_run.name),
    "physical_constants": {
        "frequency_hz": F_HZ,
        "c_water_m_s": C_WATER,
        "rho0_kg_m3": RHO0,
        "particle_radius_um": A_PART * 1e6,
        "f1_monopole": float(F1),
        "f2_dipole": float(F2),
    },
    "grid": {
        "n_grid": n_grid,
        "dx_um": float(dx * 1e6),
        "aperture_diameter_mm": float(aperture_diameter_m * 1e3),
    },
    "potential_stats": {
        "U_min_J": float(U.min()),
        "U_max_J": float(U.max()),
        "U_mean_J": float(np.mean(U[aperture_mask])),
    },
    "force_stats": {
        "F_magnitude_max_N": float(np.sqrt(Fx**2 + Fy**2).max()),
        "F_magnitude_mean_N": float(np.mean(np.sqrt(Fx[aperture_mask]**2 + Fy[aperture_mask]**2))),
    },
}

if a_pos_m is not None:
    analysis["trap_positions"] = {
        "A_mm": [float(a_pos_m[0]*1e3), float(a_pos_m[1]*1e3)],
        "B_mm": [float(b_pos_m[0]*1e3), float(b_pos_m[1]*1e3)],
        "distance_AB_mm": float(AB_dist * 1e3),
    }
    analysis["force_analysis"] = {
        "F_at_A_N": [float(F_at_A[0]), float(F_at_A[1])],
        "F_magnitude_at_A_N": float(F_mag_A),
        "A_to_B_direction": [float(AB_dir[0]), float(AB_dir[1])],
        "angle_between_F_and_AB_deg": float(angle_deg),
        "force_points_toward_B": bool(angle_deg < 90),
    }
    analysis["potential_analysis"] = {
        "U_at_A_J": float(U_A),
        "U_at_B_J": float(U_B),
        "delta_U_J": float(dU),
        "potential_lower_at_B": bool(dU < 0),
        "interpretation": "If delta_U < 0, particle is attracted to B potential well"
    }

analysis_file = out_dir / "gorkov_force_analysis.json"
with open(analysis_file, "w") as f:
    json.dump(analysis, f, indent=2)

print(f"\n[results] analysis saved to gorkov_force_analysis.json")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
if a_pos_m is not None:
    print(f"Trap A→B distance: {AB_dist*1e3:.4f} mm")
    print(f"Force at A angle to A→B: {angle_deg:.1f}°")
    print(f"Potential difference: ΔU = {dU:.3e} J")
    
    if angle_deg < 90 and dU < 0:
        print("\n✓✓✓ GOOD: Force points to B AND potential is lower at B")
        print("         → Particle A would be transported to B")
    elif angle_deg < 90:
        print("\n⚠ MIXED: Force points to B but potential is higher at B")
        print("         → Force would push A→B but it's not a stable trap")
    elif dU < 0:
        print("\n⚠ MIXED: Potential lower at B but force points away")
        print("         → Potential trap at B but force initially repels")
    else:
        print("\n✗ BAD: Force points away AND potential is higher at B")
        print("       → Particle would NOT be attracted to B")
print("="*70)
