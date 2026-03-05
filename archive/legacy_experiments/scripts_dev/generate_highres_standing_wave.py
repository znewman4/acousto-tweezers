#!/usr/bin/env python3
"""
Generate a high-quality FEM standing-wave field (no vortex lens).

This is the "truth" standing-wave pattern used for hybrid RS+FEM validation.
The vortex disk source is disabled; only the standing-wave wall transducers
are active with the canonical antiphase pattern on both axes.

Usage:
    micromamba run -n acousto-complex python scripts/dev/generate_highres_standing_wave.py [--epl 6] [--grid_n 300]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# ── project root ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Thread control (before any FEniCSx import)
NTHREADS = str(min(os.cpu_count() or 4, 14))
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, NTHREADS)

from acoustweezers.experiments.farfield_petri_cuboid.config import FarFieldConfig
from acoustweezers.experiments.farfield_petri_cuboid.presets import (
    CORRECTED_PRESET,
    PETSC_MUMPS,
)
from acoustweezers.experiments.farfield_petri_cuboid.solve_pressure import solve_helmholtz
from scipy.interpolate import NearestNDInterpolator

# ── Physical constants ────────────────────────────────────────────
WATER_C   = 1484.0
WATER_RHO = 997.0
F_HZ      = 2.0e6
LAM       = WATER_C / F_HZ             # 0.742 mm
K_WATER   = 2.0 * np.pi * F_HZ / WATER_C

H_UNDER = 3.0e-3
H_TOP   = 2.0085e-3
H_TOTAL = H_UNDER + H_TOP
Z_MID   = H_UNDER + H_TOP / 2.0
Z_STAR  = Z_MID + 0.25 * LAM           # ≈ 4.190 mm

PETSC_OPTS = {
    **PETSC_MUMPS,
    "mat_mumps_icntl_14": "100",
    "mat_mumps_icntl_23": "0",
}


def parse_args():
    p = argparse.ArgumentParser(description="High-res FEM standing-wave generation")
    p.add_argument("--epl", type=int, default=6,
                   help="Elements per wavelength (default: 6)")
    p.add_argument("--grid_n", type=int, default=300,
                   help="Output grid N per axis (default: 300)")
    p.add_argument("--timestamp", default=None,
                   help="Override output timestamp")
    return p.parse_args()


def main():
    args = parse_args()
    TS = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    EPL = args.epl
    GRID_N = args.grid_n

    OUT_DIR = PROJECT_ROOT / "results" / f"rs_hybrid_validation_{TS}"
    FEM_DIR = OUT_DIR / "fem_standing"
    FIG_DIR = OUT_DIR / "figures"
    FEM_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("HIGH-QUALITY FEM STANDING WAVE — TRUTH GENERATION")
    print("=" * 72)
    print(f"Output       : {OUT_DIR.relative_to(PROJECT_ROOT)}")
    print(f"EPL          : {EPL}")
    print(f"Grid         : {GRID_N}×{GRID_N}")
    print(f"λ = {LAM*1e3:.4f} mm   z* = {Z_STAR*1e3:.3f} mm")
    print()

    # ── Configure standing-wave-only solve ────────────────────────
    overrides = {
        **CORRECTED_PRESET,
        "disk_velocity_amplitude": 0.0,         # disable vortex
        "elements_per_wavelength": EPL,
    }
    cfg = FarFieldConfig(**overrides)

    print(f"Mesh: nx={cfg.mesh_nx}, ny={cfg.mesh_ny}, nz={cfg.mesh_nz}")
    print(f"Domain: {cfg.Lx*1e3:.1f}×{cfg.Ly*1e3:.1f}×{cfg.H_total*1e3:.2f} mm")
    print(f"PML: t_xy={cfg.t_pml_xy*1e3:.3f} mm, t_z={cfg.t_pml_z*1e3:.3f} mm")
    print()

    # ── Solve ─────────────────────────────────────────────────────
    print("Solving Helmholtz (standing wave only)...")
    t0 = time.time()
    sol = solve_helmholtz(cfg, verbose=True, petsc_options=PETSC_OPTS)
    solve_t = time.time() - t0
    print(f"  Solve time: {solve_t:.1f}s")
    print(f"  KSP converged reason: {sol.ksp_converged_reason}")
    print(f"  Max |p|: {sol.max_pressure:.4f} Pa")
    print()

    # ── Filter PML DOFs ───────────────────────────────────────────
    coords = sol.coords.copy()
    p_vals = sol.p_values.copy()
    t_xy = cfg.t_pml_xy
    t_z  = cfg.t_pml_z

    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    in_pml_x = ((x < t_xy) | (x > cfg.Lx - t_xy)) & (z < H_UNDER)
    in_pml_y = ((y < t_xy) | (y > cfg.Ly - t_xy)) & (z < H_UNDER)
    in_pml_z = z < t_z
    is_physical = ~(in_pml_x | in_pml_y | in_pml_z)

    phys_coords = coords[is_physical]
    phys_p = p_vals[is_physical]
    print(f"  Physical DOFs: {is_physical.sum()}/{len(coords)}")
    print(f"  Phys max |p|: {np.abs(phys_p).max():.4f} Pa")

    # ── Interpolate to output grid ────────────────────────────────
    interp_re = NearestNDInterpolator(phys_coords, np.real(phys_p))
    interp_im = NearestNDInterpolator(phys_coords, np.imag(phys_p))

    xg = np.linspace(t_xy, cfg.Lx - t_xy, GRID_N)
    yg = np.linspace(t_xy, cfg.Ly - t_xy, GRID_N)
    dx_out = float(xg[1] - xg[0])
    dy_out = float(yg[1] - yg[0])

    # Z-planes: z*, z* ± 0.2mm
    z_planes = [Z_STAR - 0.2e-3, Z_STAR, Z_STAR + 0.2e-3]
    z_labels = ["zstar_m02", "zstar", "zstar_p02"]

    print("\nInterpolating to output grids...")
    XG, YG = np.meshgrid(xg, yg)
    p_xy_dict = {}
    for zp, zlbl in zip(z_planes, z_labels):
        pts = np.column_stack([XG.ravel(), YG.ravel(), np.full(XG.size, zp)])
        p_plane = (interp_re(pts) + 1j * interp_im(pts)).reshape(XG.shape)
        p_xy_dict[zlbl] = p_plane
        print(f"  z={zp*1e3:.3f} mm ({zlbl}): max|p|={np.abs(p_plane).max():.4f}")

    # ── XZ slice at mid-y ─────────────────────────────────────────
    mid_y = cfg.Ly / 2.0
    zg_xz = np.linspace(t_z, cfg.H_total, GRID_N)
    X_xz, Z_xz = np.meshgrid(xg, zg_xz)
    pts_xz = np.column_stack([X_xz.ravel(), np.full(X_xz.size, mid_y), Z_xz.ravel()])
    p_xz = (interp_re(pts_xz) + 1j * interp_im(pts_xz)).reshape(X_xz.shape)
    print(f"  XZ slice at y={mid_y*1e3:.2f} mm: max|p|={np.abs(p_xz).max():.4f}")

    # ── Save NPZ ──────────────────────────────────────────────────
    save_data = {
        "xg": xg, "yg": yg,
        "z_planes": np.array(z_planes),
        "z_labels": np.array(z_labels),
        "p_xy_zstar": p_xy_dict["zstar"],
        "p_xy_zstar_m02": p_xy_dict["zstar_m02"],
        "p_xy_zstar_p02": p_xy_dict["zstar_p02"],
        "xg_xz": xg, "zg_xz": zg_xz, "p_xz": p_xz,
        "z_star": np.array(Z_STAR),
    }
    npz_path = FEM_DIR / "standing_wave_zstar.npz"
    np.savez_compressed(str(npz_path), **save_data)
    print(f"\n  Saved: {npz_path.relative_to(PROJECT_ROOT)}")

    # ── Metadata ──────────────────────────────────────────────────
    metadata = {
        "description": "FEM standing-wave truth (no vortex)",
        "timestamp": TS,
        "elements_per_wavelength": EPL,
        "grid_n": GRID_N,
        "dx_um": round(dx_out * 1e6, 2),
        "dy_um": round(dy_out * 1e6, 2),
        "lambda_mm": round(LAM * 1e3, 4),
        "z_star_mm": round(Z_STAR * 1e3, 4),
        "H_under_mm": round(H_UNDER * 1e3, 2),
        "H_top_mm": round(H_TOP * 1e3, 4),
        "H_total_mm": round(H_TOTAL * 1e3, 4),
        "frequency_hz": F_HZ,
        "standing_velocity_amplitude": 10e-6,
        "standing_phase_pattern": "antiphase",
        "standing_axis": "both",
        "solve_time_s": round(solve_t, 1),
        "ksp_converged_reason": int(sol.ksp_converged_reason),
        "max_pressure_Pa": round(float(np.abs(phys_p).max()), 6),
        "mesh_nx": cfg.mesh_nx,
        "mesh_ny": cfg.mesh_ny,
        "mesh_nz": cfg.mesh_nz,
    }
    meta_path = FEM_DIR / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved: {meta_path.relative_to(PROJECT_ROOT)}")

    # ── Trap detection ────────────────────────────────────────────
    from scipy.ndimage import minimum_filter

    p_star = p_xy_dict["zstar"]
    amp_star = np.abs(p_star)
    min_sep_px = max(3, int(0.5 * LAM / dx_out))

    filt = minimum_filter(amp_star, size=min_sep_px)
    is_min = (amp_star == filt) & np.isfinite(amp_star)
    # Only keep minima with amp below median (traps are low-pressure)
    amp_thresh = np.median(amp_star) * 0.5
    is_min &= amp_star < amp_thresh
    iys, ixs = np.where(is_min)
    order = np.argsort(amp_star[iys, ixs])
    trap_ixs = ixs[order]
    trap_iys = iys[order]
    trap_xs = xg[trap_ixs]
    trap_ys = yg[trap_iys]

    print(f"\n  Trap minima detected: {len(trap_xs)}")
    if len(trap_xs) > 0:
        # Compute periodicity from trap positions in x
        cx = cfg.Lx / 2.0
        cy = cfg.Ly / 2.0
        # Find unique x-positions near centre row
        near_centre_y = np.abs(trap_ys - cy) < 0.5 * LAM
        x_centre = np.sort(trap_xs[near_centre_y])
        if len(x_centre) >= 2:
            spacings = np.diff(x_centre)
            median_spacing = float(np.median(spacings))
            print(f"  Median trap spacing (x near centre): {median_spacing*1e3:.3f} mm = {median_spacing/LAM:.2f} λ")
            metadata["trap_spacing_mm"] = round(median_spacing * 1e3, 4)
            metadata["trap_spacing_lambda"] = round(median_spacing / LAM, 3)
        metadata["n_traps_detected"] = int(len(trap_xs))

        # Re-save metadata with trap info
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

    # ── FIGURES ───────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    extent = [xg[0]*1e3, xg[-1]*1e3, yg[0]*1e3, yg[-1]*1e3]

    # Panel 1 — Amplitude + Phase at z*
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Standing Wave Truth — z* = {Z_STAR*1e3:.3f} mm  (EPL={EPL})",
                 fontsize=14, fontweight="bold")

    im0 = axes[0].imshow(amp_star, extent=extent, origin="lower", cmap="magma")
    axes[0].set_title("|p| at z*")
    axes[0].set_xlabel("x (mm)"); axes[0].set_ylabel("y (mm)")
    fig.colorbar(im0, ax=axes[0], shrink=0.8, label="Pa")

    im1 = axes[1].imshow(np.angle(p_star), extent=extent, origin="lower",
                          cmap="hsv", vmin=-np.pi, vmax=np.pi)
    axes[1].set_title("Phase at z*")
    axes[1].set_xlabel("x (mm)"); axes[1].set_ylabel("y (mm)")
    fig.colorbar(im1, ax=axes[1], shrink=0.8, label="rad")

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG_DIR / "panel_sw_amplitude_phase.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved panel_sw_amplitude_phase.png")

    # Panel 2 — Annotated trap centres
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(amp_star, extent=extent, origin="lower", cmap="magma")
    if len(trap_xs) > 0:
        ax.scatter(trap_xs * 1e3, trap_ys * 1e3, c="cyan", s=30, marker="x",
                   linewidths=1.5, label=f"Traps ({len(trap_xs)})")
        ax.legend(fontsize=10, loc="upper right")
    ax.set_title(f"Standing Wave |p| with Trap Centres — z* = {Z_STAR*1e3:.3f} mm",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "panel_sw_trap_centers.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved panel_sw_trap_centers.png")

    # ── Summary ───────────────────────────────────────────────────
    print()
    print("=" * 72)
    print(f"STANDING WAVE GENERATION COMPLETE  ({solve_t:.1f}s solve)")
    print("=" * 72)
    print(f"  Output: {OUT_DIR.relative_to(PROJECT_ROOT)}")
    print(f"  dx = {dx_out*1e6:.2f} μm  ({dx_out/LAM:.3f} λ)")
    print(f"  Grid: {GRID_N}×{GRID_N}  ({len(z_planes)} z-planes)")
    print(f"  Traps: {len(trap_xs)} detected")
    print()

    # Return the timestamp for downstream scripts
    return TS


if __name__ == "__main__":
    main()
