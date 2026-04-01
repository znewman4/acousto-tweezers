#!/usr/bin/env python3
"""
Compare Gorkov force field for:
  1. Original bridge pressure field (standalone simulation)
  2. IASA reconstructed lens field

This will tell us if the bridge field itself is good for A→B transport,
or if there's an issue with the IASA reconstruction.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# ═══════════════════════════════════════════════════════════════════
# Physical constants
# ═══════════════════════════════════════════════════════════════════
C_WATER = 1484.0
F_HZ    = 2.15e6
OMEGA   = 2.0 * np.pi * F_HZ
RHO0    = 997.0

RHO_P   = 1050.0
C_P     = 2350.0
A_PART  = 50.0e-6

KAPPA_W = 1.0 / (RHO0 * C_WATER**2)
KAPPA_P = 1.0 / (RHO_P * C_P**2)
F1 = 1.0 - KAPPA_P / KAPPA_W
F2 = 2.0 * (RHO_P - RHO0) / (2.0 * RHO_P + RHO0)

# ═══════════════════════════════════════════════════════════════════
# Load bridge pressure field (original, from standalone simulation)
# ═══════════════════════════════════════════════════════════════════

bridge_npz_path = PROJECT_ROOT / "results" / "dev" / "bridge_pressure_field_standalone_scaled" / "bridge_pressure_fields.npz"
if not bridge_npz_path.exists():
    print(f"[error] {bridge_npz_path} not found")
    sys.exit(1)

d_bridge = np.load(bridge_npz_path)
p_bridge_full = d_bridge["p_bridge_effective_full"]  # (400, 400) complex
x_full = d_bridge["x_full"].astype(float)
y_full = d_bridge["y_full"].astype(float)
traps_raw = d_bridge["traps_m"].astype(float)
idx_a = int(d_bridge["idx_a"])
idx_b = int(d_bridge["idx_b"])

x_span = x_full[-1] - x_full[0]
y_span = y_full[-1] - y_full[0]
dx_bridge = x_span / (p_bridge_full.shape[1] - 1)

x_center = 0.5 * (x_full[0] + x_full[-1])
y_center = 0.5 * (y_full[0] + y_full[-1])

a_pos_m = traps_raw[idx_a][:2] - np.array([x_center, y_center])
b_pos_m = traps_raw[idx_b][:2] - np.array([x_center, y_center])

print(f"[bridge] field shape: {p_bridge_full.shape}, extent: {x_span*1e3:.2f} x {y_span*1e3:.2f} mm")
print(f"[bridge] dx: {dx_bridge*1e6:.2f} µm")
print(f"[bridge] |p| range: [0, {np.abs(p_bridge_full).max():.1f}] Pa")
print(f"[bridge] trap A: {a_pos_m*1e3} mm")
print(f"[bridge] trap B: {b_pos_m*1e3} mm")

# ───────────────────────────────────────────────────────────────────
# Gorkov functions
# ───────────────────────────────────────────────────────────────────

def compute_gorkov(p_field, dx):
    """Gor'kov potential."""
    p_abs2 = np.abs(p_field)**2
    dp_dx = np.gradient(p_field, dx, axis=1)
    dp_dy = np.gradient(p_field, dx, axis=0)
    grad_p_abs2 = np.abs(dp_dx)**2 + np.abs(dp_dy)**2
    coeff_p = F1 / (2.0 * RHO0 * C_WATER**2)
    coeff_k = 3.0 * F2 / (4.0 * OMEGA**2 * RHO0)
    return coeff_p * p_abs2 - coeff_k * grad_p_abs2

def compute_force_field(U, dx):
    """Force field F = -∇U."""
    dUdx = np.gradient(U, dx, axis=1)
    dUdy = np.gradient(U, dx, axis=0)
    return -dUdx, -dUdy

def analyze_at_position(U, Fx, Fy, dx, pos_m, label):
    """Analyze force and potential at a position."""
    n = U.shape[0]
    n_half = n // 2
    i = int(round(pos_m[1] / dx + n_half))
    j = int(round(pos_m[0] / dx + n_half))
    i = np.clip(i, 0, n - 1)
    j = np.clip(j, 0, n - 1)
    
    F_at_pos = np.array([Fx[i, j], Fy[i, j]])
    F_mag = np.linalg.norm(F_at_pos)
    U_val = U[i, j]
    
    return {"pos": pos_m, "F": F_at_pos, "F_mag": F_mag, "U": U_val}

# ═══════════════════════════════════════════════════════════════════
# Compute Gorkov for bridge field
# ═══════════════════════════════════════════════════════════════════

print(f"\n[compute] Gorkov potential for ORIGINAL bridge field...")
U_bridge = compute_gorkov(p_bridge_full, dx_bridge)
Fx_bridge, Fy_bridge = compute_force_field(U_bridge, dx_bridge)

print(f"[bridge] U range: [{U_bridge.min():.3e}, {U_bridge.max():.3e}] J")
print(f"[bridge] |F| max: {np.sqrt(Fx_bridge**2 + Fy_bridge**2).max():.3e} N")

# Analyze at A and B
result_a = analyze_at_position(U_bridge, Fx_bridge, Fy_bridge, dx_bridge, a_pos_m, "A")
result_b = analyze_at_position(U_bridge, Fx_bridge, Fy_bridge, dx_bridge, b_pos_m, "B")

AB_vec = b_pos_m - a_pos_m
AB_dist = np.linalg.norm(AB_vec)
AB_dir = AB_vec / (AB_dist + 1e-12)

F_A = result_a["F"]
F_mag_A = result_a["F_mag"]
cos_angle_bridge = np.dot(F_A / (F_mag_A + 1e-12), AB_dir)
angle_bridge = np.arccos(np.clip(cos_angle_bridge, -1, 1)) * 180 / np.pi
dU_bridge = result_b["U"] - result_a["U"]

print(f"\n[BRIDGE FIELD ANALYSIS]")
print(f"  Force at A magnitude: {F_mag_A:.3e} N")
print(f"  Angle to A→B: {angle_bridge:.1f}°")
print(f"  ΔU (B - A): {dU_bridge:.3e} J")

if angle_bridge < 90 and dU_bridge < 0:
    print(f"  ✓ BRIDGE FIELD: Force A→B and lower potential at B")
elif angle_bridge < 90:
    print(f"  ~ BRIDGE FIELD: Force A→B but higher potential at B")
else:
    print(f"  ✗ BRIDGE FIELD: Force points away from B")

# ═══════════════════════════════════════════════════════════════════
# Load IASA reconstructed lens and compare
# ═══════════════════════════════════════════════════════════════════

# Most recent inverse bridge run
RESULTS_DIR = PROJECT_ROOT / "results" / "dev"
bridge_results = sorted(RESULTS_DIR.glob("inverse_bridge_pressure_lens_replica_*"))
if bridge_results:
    latest_run = bridge_results[-1]
    npz_path = latest_run / "bridge_inverse_replica_fields.npz"
    
    if npz_path.exists():
        d_recon = np.load(npz_path)
        lens_field = d_recon["lens_field"]  # (400, 400) complex
        aperture_mask = d_recon["aperture_mask"]
        
        # Grid for reconstruction
        n_grid = 400
        aperture_m = 0.020
        dx_recon = aperture_m / n_grid
        
        print(f"\n[compute] Gorkov potential for IASA RECONSTRUCTED lens field...")
        U_recon = compute_gorkov(lens_field, dx_recon)
        Fx_recon, Fy_recon = compute_force_field(U_recon, dx_recon)
        
        print(f"[recon] U range: [{U_recon.min():.3e}, {U_recon.max():.3e}] J")
        print(f"[recon] |F| max: {np.sqrt(Fx_recon**2 + Fy_recon**2).max():.3e} N")
        
        # Analyze at A and B (same physical positions, but in different grids)
        # Scale positions to reconstruction grid
        a_pos_recon = a_pos_m * (aperture_m / x_span)  # rescale
        b_pos_recon = b_pos_m * (aperture_m / x_span)
        
        result_a_recon = analyze_at_position(U_recon, Fx_recon, Fy_recon, dx_recon, a_pos_recon, "A_recon")
        result_b_recon = analyze_at_position(U_recon, Fx_recon, Fy_recon, dx_recon, b_pos_recon, "B_recon")
        
        F_A_recon = result_a_recon["F"]
        F_mag_A_recon = result_a_recon["F_mag"]
        cos_angle_recon = np.dot(F_A_recon / (F_mag_A_recon + 1e-12), AB_dir)
        angle_recon = np.arccos(np.clip(cos_angle_recon, -1, 1)) * 180 / np.pi
        dU_recon = result_b_recon["U"] - result_a_recon["U"]
        
        print(f"\n[IASA RECONSTRUCTED LENS ANALYSIS]")
        print(f"  Force at A magnitude: {F_mag_A_recon:.3e} N")
        print(f"  Angle to A→B: {angle_recon:.1f}°")
        print(f"  ΔU (B - A): {dU_recon:.3e} J")
        
        if angle_recon < 90 and dU_recon < 0:
            print(f"  ✓ RECON LENS: Force A→B and lower potential at B")
        elif angle_recon < 90:
            print(f"  ~ RECON LENS: Force A→B but higher potential at B")
        else:
            print(f"  ✗ RECON LENS: Force points away from B")

# ═══════════════════════════════════════════════════════════════════
# Comparison figure
# ═══════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Bridge field
ax = axes[0]
im0 = ax.imshow(U_bridge, origin="lower", cmap="RdBu_r", aspect="equal", 
                 extent=[x_full[0]*1e3, x_full[-1]*1e3, y_full[0]*1e3, y_full[-1]*1e3])
plt.colorbar(im0, ax=ax, label="U [J]")
ax.plot(a_pos_m[0]*1e3, a_pos_m[1]*1e3, "g*", markersize=15, label="A")
ax.plot(b_pos_m[0]*1e3, b_pos_m[1]*1e3, "r*", markersize=15, label="B")
ax.arrow(a_pos_m[0]*1e3, a_pos_m[1]*1e3,
         (b_pos_m[0]-a_pos_m[0])*1e3*0.7, (b_pos_m[1]-a_pos_m[1])*1e3*0.7,
         head_width=0.02, head_length=0.02, fc="cyan", ec="cyan", alpha=0.5)
ax.set_title(f"Original Bridge Field Gorkov Potential\n(angle={angle_bridge:.0f}°, ΔU={dU_bridge:.2e}J)")
ax.set_xlabel("x [mm]")
ax.set_ylabel("y [mm]")
ax.legend()

# IASA reconstruction
if bridge_results:
    ax = axes[1]
    # Rescale coordinates for visualization
    extent_recon = [-aperture_m/2*1e3, aperture_m/2*1e3, -aperture_m/2*1e3, aperture_m/2*1e3]
    im1 = ax.imshow(U_recon, origin="lower", cmap="RdBu_r", aspect="equal", extent=extent_recon)
    plt.colorbar(im1, ax=ax, label="U [J]")
    ax.plot(a_pos_recon[0]*1e3, a_pos_recon[1]*1e3, "g*", markersize=15, label="A")
    ax.plot(b_pos_recon[0]*1e3, b_pos_recon[1]*1e3, "r*", markersize=15, label="B")
    ax.arrow(a_pos_recon[0]*1e3, a_pos_recon[1]*1e3,
             (b_pos_recon[0]-a_pos_recon[0])*1e3*0.7, (b_pos_recon[1]-a_pos_recon[1])*1e3*0.7,
             head_width=0.02, head_length=0.02, fc="cyan", ec="cyan", alpha=0.5)
    ax.set_title(f"IASA Reconstructed Lens Gorkov Potential\n(angle={angle_recon:.0f}°, ΔU={dU_recon:.2e}J)")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.legend()

fig.suptitle("Bridge vs IASA Lens: Gorkov Potential Comparison", fontsize=12, weight="bold")
fig.tight_layout()
fig.savefig(latest_run / "gorkov_comparison_bridge_vs_iasa.png", dpi=150, bbox_inches="tight")
print(f"\n[figure] saved gorkov_comparison_bridge_vs_iasa.png")
plt.close(fig)

print("\n" + "="*70)
print("INTERPRETATION")
print("="*70)
if angle_bridge < 90 and dU_bridge < 0:
    print("✓ ORIGINAL BRIDGE FIELD: Good for A→B transport")
elif angle_bridge < 90:
    print("~ ORIGINAL BRIDGE FIELD: Force points A→B but potential opposes it")
else:
    print("✗ ORIGINAL BRIDGE FIELD: Force points away from B")
    print("  → The bridge field itself may not be designed for A→B particle transport")
    print("  → This suggests the bridge design goal is structural, not transport")
print("="*70)
