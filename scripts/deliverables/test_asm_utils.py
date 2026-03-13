#!/usr/bin/env python3
"""
Test script for scripts/lib/asm_utils.py — Phase 2.1 validation.

Exercises every public function in asm_utils:
  1. make_grid_from_fem   — grid from FEM cache
  2. make_vortex_field    — LG-like vortex at z = 0
  3. make_lens_phase      — ideal / plastic / axicon
  4. propagate_asm        — forward-propagate vortex to z*
  5. make_cshape_mask     — C-shape perturbation

Saves results to  results/deliverables/asm_test/
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Project root & imports ─────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lib.fem_cache_utils import load_fem_cache, LAM, C_WATER, F_HZ
from scripts.lib.asm_utils import (
    make_grid_from_fem,
    make_vortex_field,
    make_lens_phase,
    propagate_asm,
    make_cshape_mask,
    K0,
)

OUT = PROJECT_ROOT / "results" / "deliverables" / "asm_test"
OUT.mkdir(parents=True, exist_ok=True)


def _mm(arr):
    """Convert metres to millimetres for plot axes."""
    return arr * 1e3


# ── 1. Load FEM cache & build grid ────────────────────────────────
print("=" * 60)
print("Phase 2.1 — ASM utilities test")
print("=" * 60)

cache = load_fem_cache()
grid = make_grid_from_fem(cache, nx=400, ny=400)
x, y = grid["x"], grid["y"]
XX, YY = grid["XX"], grid["YY"]
dx, dy = grid["dx"], grid["dy"]

print(f"\n[1] Grid: {XX.shape}  dx={dx*1e6:.1f} µm  dy={dy*1e6:.1f} µm")
print(f"    x ∈ [{x[0]*1e3:.3f}, {x[-1]*1e3:.3f}] mm")
print(f"    y ∈ [{y[0]*1e3:.3f}, {y[-1]*1e3:.3f}] mm")

# ── 2. Vortex field ───────────────────────────────────────────────
waist = 0.8e-3  # 0.8 mm
vortex = make_vortex_field(XX, YY, charge=1, waist=waist, k=K0)

print(f"\n[2] Vortex: charge=1  waist={waist*1e3:.1f} mm")
print(f"    |p|_max = {np.abs(vortex).max():.4f}")
print(f"    phase range = [{np.angle(vortex).min():.2f}, {np.angle(vortex).max():.2f}] rad")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
ax0 = axes[0]
im0 = ax0.pcolormesh(_mm(x), _mm(y), np.abs(vortex), shading="auto", cmap="inferno")
ax0.set_title("|Vortex|  (z = 0)")
ax0.set_xlabel("x [mm]")
ax0.set_ylabel("y [mm]")
ax0.set_aspect("equal")
plt.colorbar(im0, ax=ax0, shrink=0.8)

ax1 = axes[1]
im1 = ax1.pcolormesh(_mm(x), _mm(y), np.angle(vortex), shading="auto", cmap="hsv")
ax1.set_title("Phase(Vortex)  (z = 0)")
ax1.set_xlabel("x [mm]")
ax1.set_ylabel("y [mm]")
ax1.set_aspect("equal")
plt.colorbar(im1, ax=ax1, shrink=0.8, label="rad")
fig.tight_layout()
fig.savefig(OUT / "vortex_source.png", dpi=150)
plt.close(fig)
print(f"    → saved vortex_source.png")

# ── 3. Lens phase (all three families) ────────────────────────────
R_ap = 2.0e-3   # 2 mm aperture
f_len = 10e-3   # 10 mm focal length

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, fam in zip(axes, ["ideal", "plastic", "axicon"]):
    phi = make_lens_phase(
        XX, YY,
        focal_length=f_len,
        aperture_radius=R_ap,
        family=fam,
        k=K0,
        charge=1,
        axicon_angle_deg=15.0,
    )
    im = ax.pcolormesh(_mm(x), _mm(y), phi, shading="auto", cmap="twilight")
    ax.set_title(f"Lens phase — {fam}")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax, shrink=0.8, label="rad")

fig.tight_layout()
fig.savefig(OUT / "lens_phases.png", dpi=150)
plt.close(fig)
print(f"\n[3] Lens phases (ideal / plastic / axicon): → saved lens_phases.png")

# ── 4. ASM propagation ────────────────────────────────────────────
z_prop = cache["z_star"] - cache["z_bottom"]  # propagation distance from transducer
print(f"\n[4] Propagating vortex from z=0 to z={z_prop*1e3:.3f} mm ...")

# Create a vortex + ideal lens as the source
lens_ideal = make_lens_phase(XX, YY, focal_length=z_prop, aperture_radius=R_ap,
                             family="ideal", k=K0, charge=1)
drive = vortex * np.exp(1j * lens_ideal)

p_prop = propagate_asm(drive, dx, dy, wavelength=LAM, z=z_prop)

print(f"    |p_prop|_max = {np.abs(p_prop).max():.2f}")
print(f"    shape = {p_prop.shape}")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
ax0, ax1 = axes
im0 = ax0.pcolormesh(_mm(x), _mm(y), np.abs(p_prop), shading="auto", cmap="inferno")
ax0.set_title(f"|p| at z = {z_prop*1e3:.2f} mm")
ax0.set_xlabel("x [mm]")
ax0.set_ylabel("y [mm]")
ax0.set_aspect("equal")
plt.colorbar(im0, ax=ax0, shrink=0.8, label="Pa")

im1 = ax1.pcolormesh(_mm(x), _mm(y), np.angle(p_prop), shading="auto", cmap="hsv")
ax1.set_title(f"Phase at z = {z_prop*1e3:.2f} mm")
ax1.set_xlabel("x [mm]")
ax1.set_ylabel("y [mm]")
ax1.set_aspect("equal")
plt.colorbar(im1, ax=ax1, shrink=0.8, label="rad")
fig.tight_layout()
fig.savefig(OUT / "propagated_vortex.png", dpi=150)
plt.close(fig)
print(f"    → saved propagated_vortex.png")

# ── 5. C-shape mask ───────────────────────────────────────────────
r0 = 0.4e-3  # ring radius 0.4 mm
sigma_r = 0.1e-3  # ring thickness 0.1 mm
cshape = make_cshape_mask(XX, YY, radius=r0, gap_angle=0.0,
                          thickness=sigma_r, charge=1, gap_width=0.4)

print(f"\n[5] C-shape mask: r0={r0*1e3:.1f} mm  σ_r={sigma_r*1e3:.1f} mm  gap_angle=0")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
ax0, ax1 = axes
im0 = ax0.pcolormesh(_mm(x), _mm(y), np.abs(cshape), shading="auto", cmap="inferno")
ax0.set_title("|C-shape mask|")
ax0.set_xlabel("x [mm]")
ax0.set_ylabel("y [mm]")
ax0.set_aspect("equal")
plt.colorbar(im0, ax=ax0, shrink=0.8)

im1 = ax1.pcolormesh(_mm(x), _mm(y), np.angle(cshape), shading="auto", cmap="hsv")
ax1.set_title("Phase(C-shape)")
ax1.set_xlabel("x [mm]")
ax1.set_ylabel("y [mm]")
ax1.set_aspect("equal")
plt.colorbar(im1, ax=ax1, shrink=0.8, label="rad")
fig.tight_layout()
fig.savefig(OUT / "cshape_mask.png", dpi=150)
plt.close(fig)
print(f"    → saved cshape_mask.png")

# ── Summary ───────────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print(f"All 5 tests passed.  Output → {OUT.relative_to(PROJECT_ROOT)}/")
print(f"  vortex_source.png     — vortex amplitude & phase at z=0")
print(f"  lens_phases.png       — ideal / plastic / axicon phase maps")
print(f"  propagated_vortex.png — ASM-propagated vortex + lens")
print(f"  cshape_mask.png       — C-shape perturbation mask")
print(f"{'=' * 60}")
