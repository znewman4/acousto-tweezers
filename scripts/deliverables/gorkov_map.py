#!/usr/bin/env python3
"""
Gor'kov potential map from cached FEM standing wave.

Computes the 2-D Gor'kov potential on the XY trap plane (z*) using the
canonical ``gorkov_grid_2d`` from the physics library.

Output
------
results/deliverables/gorkov/gorkov_potential_xy.png
results/deliverables/gorkov/gorkov_fields.npz   (for downstream reuse)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

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
    C_WATER, F_HZ, OMEGA, RHO0, LAM,
)

OUT_DIR = PROJECT_ROOT / "results" / "deliverables" / "gorkov"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Particle properties — polystyrene in water
A_P = 50.0e-6           # radius [m]
RHO_P = 1050.0           # kg/m³
C_P = 2350.0             # m/s
KAPPA_W = 1.0 / (RHO0 * C_WATER**2)
KAPPA_P = 1.0 / (RHO_P * C_P**2)
F1 = 1.0 - KAPPA_P / KAPPA_W
F2 = 2.0 * (RHO_P - RHO0) / (2.0 * RHO_P + RHO0)

DPI = 200


def _add_colorbar(ax, im, label: str) -> None:
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="4%", pad=0.06)
    plt.colorbar(im, cax=cax, label=label)


def main() -> None:
    cache = load_fem_cache()
    sl = interpolate_slice(cache, z=cache["z_star"], n_grid=800)
    p_grid = sl["p_grid"]
    xg_mm = sl["xg"] * 1e3
    yg_mm = sl["yg"] * 1e3
    dx = sl["dx"]
    dy = sl["dy"]

    print(f"Computing Gor'kov potential (f1={F1:.4f}, f2={F2:.4f}) …")
    U, Fx, Fy = gorkov_grid_2d(
        p_grid, dx, dy, OMEGA, RHO0, C_WATER, A_P, F1, F2,
    )
    print(f"  U range: [{U.min():.3e}, {U.max():.3e}] J")

    # ── Save for reuse ──
    npz_path = OUT_DIR / "gorkov_fields.npz"
    np.savez_compressed(
        str(npz_path),
        U=U, Fx=Fx, Fy=Fy,
        p_grid_real=np.real(p_grid),
        p_grid_imag=np.imag(p_grid),
        xg=sl["xg"], yg=sl["yg"],
        z=sl["z"], dx=dx, dy=dy,
    )
    print(f"  saved {npz_path}")

    # ── Plot ──
    extent = [xg_mm[0], xg_mm[-1], yg_mm[0], yg_mm[-1]]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1) |p|
    ax = axes[0]
    im = ax.imshow(np.abs(p_grid), origin="lower", extent=extent,
                   cmap="viridis", aspect="equal")
    ax.set_title("|p|  [Pa]")
    ax.set_xlabel("x  [mm]"); ax.set_ylabel("y  [mm]")
    _add_colorbar(ax, im, "|p|  [Pa]")

    # 2) Gor'kov U
    ax = axes[1]
    im = ax.imshow(U, origin="lower", extent=extent,
                   cmap="RdBu_r", aspect="equal")
    ax.set_title("Gor'kov potential  U  [J]")
    ax.set_xlabel("x  [mm]"); ax.set_ylabel("y  [mm]")
    _add_colorbar(ax, im, "U  [J]")

    # 3) |F_rad|
    ax = axes[2]
    F_mag = np.sqrt(Fx**2 + Fy**2)
    im = ax.imshow(F_mag, origin="lower", extent=extent,
                   cmap="inferno", aspect="equal")
    ax.set_title("|F_rad|  [N]")
    ax.set_xlabel("x  [mm]"); ax.set_ylabel("y  [mm]")
    _add_colorbar(ax, im, "|F|  [N]")

    fig.suptitle(
        f"Gor'kov potential — z = {sl['z']*1e3:.2f} mm, "
        f"a = {A_P*1e6:.0f} µm polystyrene",
        fontsize=13,
    )
    fig.tight_layout()
    fig_path = OUT_DIR / "gorkov_potential_xy.png"
    fig.savefig(fig_path, dpi=DPI)
    plt.close(fig)
    print(f"  saved {fig_path}")
    print("Done.")


if __name__ == "__main__":
    main()
