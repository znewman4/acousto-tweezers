#!/usr/bin/env python3
"""
Diagnostic z-sweep: compare trapping planes at several heights above z_bottom.

Generates side-by-side |p|, Gor'kov U, and trap overlays at:
  z_bottom + 50 µm
  z_bottom + 100 µm
  z_bottom + 200 µm

This confirms which Δz gives the best-defined trapping lattice.

Output
------
results/deliverables/standing_wave/z_sweep_diagnostic.png
results/deliverables/standing_wave/z_sweep_summary.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import minimum_filter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lib.fem_cache_utils import (
    load_fem_cache,
    interpolate_slice,
    gorkov_grid_2d,
    C_WATER, F_HZ, OMEGA, RHO0, LAM, H_TOP,
)

OUT_DIR = PROJECT_ROOT / "results" / "deliverables" / "standing_wave"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Particle — polystyrene
A_P = 50.0e-6
RHO_P, C_P = 1050.0, 2350.0
KAPPA_W = 1.0 / (RHO0 * C_WATER**2)
KAPPA_P = 1.0 / (RHO_P * C_P**2)
F1 = 1.0 - KAPPA_P / KAPPA_W
F2 = 2.0 * (RHO_P - RHO0) / (2.0 * RHO_P + RHO0)
TRAP_SP = LAM / 2.0

DZ_VALUES = [50e-6, 100e-6, 200e-6]   # offsets above z_bottom
DPI = 200
N_GRID = 800


def count_traps(U, dx):
    """Quick Hessian-stable trap count."""
    min_sep_pts = max(3, int(0.4 * TRAP_SP / dx))
    nbhd = 2 * min_sep_pts + 1
    local_min = minimum_filter(U, size=nbhd)
    mask = (U == local_min)
    border = max(5, min_sep_pts)
    mask[:border, :] = False
    mask[-border:, :] = False
    mask[:, :border] = False
    mask[:, -border:] = False
    U_range = U.max() - U.min()
    mask &= (U < U.min() + 0.50 * U_range)
    iy, ix = np.where(mask)

    dUdx = np.gradient(U, dx, axis=1)
    dUdy = np.gradient(U, dx, axis=0)
    Hxx = np.gradient(dUdx, dx, axis=1)
    Hyy = np.gradient(dUdy, dx, axis=0)
    Hxy = np.gradient(dUdx, dx, axis=0)

    n_stable = 0
    for ci in range(len(iy)):
        hxx = Hxx[iy[ci], ix[ci]]
        hyy = Hyy[iy[ci], ix[ci]]
        hxy = Hxy[iy[ci], ix[ci]]
        tr = hxx + hyy
        det = hxx * hyy - hxy**2
        disc = max(0.0, tr**2 - 4 * det)
        lam_min = (tr - np.sqrt(disc)) / 2.0
        if lam_min > 0:
            n_stable += 1
    return n_stable


def main() -> None:
    cache = load_fem_cache()
    z_bottom = cache["z_bottom"]
    print(f"z_bottom = {z_bottom*1e3:.3f} mm")

    fig, axes = plt.subplots(3, 3, figsize=(17, 15))
    summary = {"z_bottom_mm": z_bottom * 1e3, "sweeps": []}

    for row, dz in enumerate(DZ_VALUES):
        z = z_bottom + dz
        print(f"\n--- Δz = {dz*1e6:.0f} µm  →  z = {z*1e3:.4f} mm ---")
        sl = interpolate_slice(cache, z=z, n_grid=N_GRID)
        p_grid = sl["p_grid"]
        dx = sl["dx"]
        xg_mm = sl["xg"] * 1e3
        yg_mm = sl["yg"] * 1e3
        extent = [xg_mm[0], xg_mm[-1], yg_mm[0], yg_mm[-1]]

        U, Fx, Fy = gorkov_grid_2d(
            p_grid, dx, sl["dy"], OMEGA, RHO0, C_WATER, A_P, F1, F2,
        )
        F_mag = np.sqrt(Fx**2 + Fy**2)
        p_abs = np.abs(p_grid)
        n_traps = count_traps(U, dx)
        U_depth = float(U.max() - U.min())

        info = {
            "dz_um": dz * 1e6,
            "z_mm": z * 1e3,
            "p_max_Pa": float(p_abs.max()),
            "p_mean_Pa": float(p_abs.mean()),
            "U_min_J": float(U.min()),
            "U_max_J": float(U.max()),
            "U_depth_J": U_depth,
            "n_traps": n_traps,
        }
        summary["sweeps"].append(info)
        print(f"  |p|_max = {p_abs.max():.3f} Pa, U depth = {U_depth:.3e} J, "
              f"traps = {n_traps}")

        # Col 0: |p|
        ax = axes[row, 0]
        im = ax.imshow(p_abs, origin="lower", extent=extent,
                       cmap="viridis", aspect="equal")
        ax.set_title(f"|p|  —  Δz = {dz*1e6:.0f} µm")
        ax.set_ylabel("y  [mm]")
        div = make_axes_locatable(ax)
        plt.colorbar(im, cax=div.append_axes("right", size="4%", pad=0.06),
                     label="Pa")

        # Col 1: U
        ax = axes[row, 1]
        im = ax.imshow(U, origin="lower", extent=extent,
                       cmap="RdBu_r", aspect="equal")
        ax.set_title(f"Gor'kov U  —  {n_traps} traps")
        div = make_axes_locatable(ax)
        plt.colorbar(im, cax=div.append_axes("right", size="4%", pad=0.06),
                     label="J")

        # Col 2: |F|
        ax = axes[row, 2]
        im = ax.imshow(F_mag, origin="lower", extent=extent,
                       cmap="inferno", aspect="equal")
        ax.set_title("|F_rad|")
        div = make_axes_locatable(ax)
        plt.colorbar(im, cax=div.append_axes("right", size="4%", pad=0.06),
                     label="N")

    for ax in axes[-1, :]:
        ax.set_xlabel("x  [mm]")

    fig.suptitle(
        f"Trapping-plane z-sweep diagnostic\n"
        f"z_bottom = {z_bottom*1e3:.3f} mm  |  "
        f"λ/2 = {TRAP_SP*1e3:.3f} mm  |  epl5 cache",
        fontsize=14,
    )
    fig.tight_layout()
    fig_path = OUT_DIR / "z_sweep_diagnostic.png"
    fig.savefig(fig_path, dpi=DPI)
    plt.close(fig)
    print(f"\nSaved {fig_path}")

    json_path = OUT_DIR / "z_sweep_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {json_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print(f"{'Δz (µm)':>10} {'z (mm)':>10} {'|p|_max':>10} "
          f"{'U_depth':>12} {'n_traps':>8}")
    print("-" * 70)
    for s in summary["sweeps"]:
        print(f"{s['dz_um']:10.0f} {s['z_mm']:10.4f} {s['p_max_Pa']:10.3f} "
              f"{s['U_depth_J']:12.3e} {s['n_traps']:8d}")
    print("=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()
