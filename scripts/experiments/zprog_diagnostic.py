#!/usr/bin/env python3
"""
Quick diagnostic: What do the actual pressure values look like
at each z-height?  Separates physical vs PML DOFs.
"""
from __future__ import annotations
import sys, os, time, gc
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

NTHREADS = str(min(os.cpu_count() or 4, 14))
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, NTHREADS)

from acoustweezers.experiments.farfield_petri_cuboid.presets import (
    CORRECTED_PRESET, PETSC_MUMPS,
)
from acoustweezers.experiments.farfield_petri_cuboid.config import FarFieldConfig
from acoustweezers.experiments.farfield_petri_cuboid.solve_pressure import solve_helmholtz

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import NearestNDInterpolator

PETSC_OPTS = {
    **PETSC_MUMPS,
    "mat_mumps_icntl_14": "100",
    "mat_mumps_icntl_23": "0",
}

H_TOP_OPTIMAL = 2.0085e-3
COMMON = {
    **CORRECTED_PRESET,
    "H_top": H_TOP_OPTIMAL,
    "elements_per_wavelength": 4,
}

# Solve combined
print("Solving combined case...")
cfg = FarFieldConfig(**COMMON)
t0 = time.time()
sol = solve_helmholtz(cfg, verbose=True, petsc_options=PETSC_OPTS, export_fields=False)
print(f"  Done in {time.time()-t0:.1f}s  max|p|={sol.max_pressure:.3f} Pa")

coords = sol.coords.copy()
p_vals = sol.p_values.copy()
pmag = np.abs(p_vals)

# PML bounds
t_xy = cfg.t_pml_xy
t_z = cfg.t_pml_z
print(f"\nDomain: [0, {cfg.Lx*1e3:.2f}] x [0, {cfg.Ly*1e3:.2f}] x [0, {cfg.H_total*1e3:.4f}] mm")
print(f"PML thickness xy: {t_xy*1e3:.3f} mm")
print(f"PML thickness z:  {t_z*1e3:.3f} mm")
print(f"Physical x: [{t_xy*1e3:.3f}, {(cfg.Lx-t_xy)*1e3:.3f}] mm")
print(f"Physical y: [{t_xy*1e3:.3f}, {(cfg.Ly-t_xy)*1e3:.3f}] mm")
print(f"Physical z: [{t_z*1e3:.3f}, {cfg.H_total*1e3:.4f}] mm (no top PML)")

# Classify DOFs
x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
in_pml_x = (x < t_xy) | (x > cfg.Lx - t_xy)
in_pml_y = (y < t_xy) | (y > cfg.Ly - t_xy)
in_pml_z = z < t_z
in_pml = in_pml_x | in_pml_y | in_pml_z
in_phys = ~in_pml

print(f"\nTotal DOFs: {len(pmag)}")
print(f"Physical DOFs: {in_phys.sum()}")
print(f"PML DOFs: {in_pml.sum()}")
print(f"\nGlobal max|p|:    {pmag.max():.4f} Pa  (at DOF {np.argmax(pmag)})")
print(f"Physical max|p|:  {pmag[in_phys].max():.4f} Pa")
print(f"PML max|p|:       {pmag[in_pml].max():.4f} Pa")

# Where is physical max?
phys_max_idx = np.where(in_phys)[0][np.argmax(pmag[in_phys])]
print(f"Physical max location: ({coords[phys_max_idx]*1e3}) mm")

# Where is global max?
glob_max_idx = np.argmax(pmag)
print(f"Global max location:   ({coords[glob_max_idx]*1e3}) mm")

# Z-height analysis
Z_HEIGHTS_MM = [0.5, 0.8, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
print(f"\n{'z [mm]':>8} | {'Region':>8} | {'All DOFs max':>12} | {'Phys DOFs max':>14} | "
      f"{'Phys mean':>10} | {'Phys median':>12} | {'# phys DOFs':>12}")
print("-" * 100)

for z_mm in Z_HEIGHTS_MM:
    z_m = z_mm * 1e-3
    dz = 0.15e-3  # half-slab thickness for selecting nearby DOFs
    mask_z = (z > z_m - dz) & (z < z_m + dz)
    mask_phys_z = mask_z & in_phys
    mask_all_z = mask_z
    region = "Petri" if z_mm >= cfg.H_under * 1e3 else "Bath"
    n_phys = mask_phys_z.sum()
    if n_phys > 0:
        phys_max = pmag[mask_phys_z].max()
        phys_mean = pmag[mask_phys_z].mean()
        phys_med = np.median(pmag[mask_phys_z])
    else:
        phys_max = phys_mean = phys_med = 0
    all_max = pmag[mask_all_z].max() if mask_all_z.sum() > 0 else 0
    print(f"  {z_mm:5.1f}  | {region:>8} | {all_max:12.4f} | {phys_max:14.4f} | "
          f"{phys_mean:10.4f} | {phys_med:12.4f} | {n_phys:12d}")

# Also check: interpolation grid comparison (full domain vs physical-only)
print("\n\nInterpolation grid comparison at z = 4.0 mm (trap plane):")
z_trap = cfg.H_under + cfg.H_top / 2
n = 300

# Full domain grid
xg_full = np.linspace(0, cfg.Lx, n)
yg_full = np.linspace(0, cfg.Ly, n)
X, Y = np.meshgrid(xg_full, yg_full)
pts = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, z_trap)])

interp_re = NearestNDInterpolator(coords, np.real(p_vals))
interp_im = NearestNDInterpolator(coords, np.imag(p_vals))
p_full = np.abs(interp_re(pts) + 1j * interp_im(pts)).reshape(X.shape)

# Physical-only grid
xg_phys = np.linspace(t_xy, cfg.Lx - t_xy, n)
yg_phys = np.linspace(t_xy, cfg.Ly - t_xy, n)
X2, Y2 = np.meshgrid(xg_phys, yg_phys)
pts2 = np.column_stack([X2.ravel(), Y2.ravel(), np.full(X2.size, z_trap)])
p_phys = np.abs(interp_re(pts2) + 1j * interp_im(pts2)).reshape(X2.shape)

# Physical-only interpolator (exclude PML DOFs)
interp_re_p = NearestNDInterpolator(coords[in_phys], np.real(p_vals[in_phys]))
interp_im_p = NearestNDInterpolator(coords[in_phys], np.imag(p_vals[in_phys]))
p_phys2 = np.abs(interp_re_p(pts2) + 1j * interp_im_p(pts2)).reshape(X2.shape)

print(f"  Full grid,   all DOFs interp: min={p_full.min():.4f}  max={p_full.max():.4f}  mean={p_full.mean():.4f}")
print(f"  Phys grid,   all DOFs interp: min={p_phys.min():.4f}  max={p_phys.max():.4f}  mean={p_phys.mean():.4f}")
print(f"  Phys grid, phys DOFs interp:  min={p_phys2.min():.4f}  max={p_phys2.max():.4f}  mean={p_phys2.mean():.4f}")

# Save diagnostic figure
fig, axes = plt.subplots(1, 3, figsize=(21, 7))
for ax, data, xg, yg, title in [
    (axes[0], p_full, xg_full, yg_full, f"Full grid, all DOFs\nmax={p_full.max():.4f}"),
    (axes[1], p_phys, xg_phys, yg_phys, f"Phys grid, all DOFs\nmax={p_phys.max():.4f}"),
    (axes[2], p_phys2, xg_phys, yg_phys, f"Phys grid, phys DOFs\nmax={p_phys2.max():.4f}"),
]:
    im = ax.pcolormesh(xg * 1e3, yg * 1e3, data,
                       shading="auto", cmap="inferno")
    ax.set_title(title)
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax, label="|p| [Pa]")
fig.suptitle(f"Diagnostic: XY at z = {z_trap*1e3:.1f} mm (trap plane)", fontsize=14)
fig.tight_layout()
fig.savefig(PROJECT_ROOT / "results" / "zprog_diagnostic.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# Also do a log-scale version
from matplotlib.colors import LogNorm
fig, axes = plt.subplots(1, 3, figsize=(21, 7))
for ax, data, xg, yg, title in [
    (axes[0], p_full, xg_full, yg_full, "Full grid (log)"),
    (axes[1], p_phys, xg_phys, yg_phys, "Phys grid (log)"),
    (axes[2], p_phys2, xg_phys, yg_phys, "Phys grid, phys DOFs (log)"),
]:
    dclip = np.clip(data, 1e-6, None)
    im = ax.pcolormesh(xg * 1e3, yg * 1e3, dclip,
                       shading="auto", cmap="inferno",
                       norm=LogNorm(vmin=max(dclip.min(), 1e-4), vmax=dclip.max()))
    ax.set_title(title)
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax, label="|p| [Pa]")
fig.suptitle(f"Diagnostic LOG: XY at z = {z_trap*1e3:.1f} mm", fontsize=14)
fig.tight_layout()
fig.savefig(PROJECT_ROOT / "results" / "zprog_diagnostic_log.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("\nSaved: results/zprog_diagnostic.png + _log.png")
