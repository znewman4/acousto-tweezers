#!/usr/bin/env python3
"""
Pressure line profiles through detected traps.

Draws horizontal and vertical line profiles through the deepest trap,
plus a profile connecting two adjacent traps (showing λ/2 periodicity).

Output
------
results/deliverables/profiles/pressure_line_through_traps.png
results/deliverables/profiles/pressure_line_axial.png
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

from scripts.lib.fem_cache_utils import (
    load_fem_cache,
    interpolate_slice,
    interpolate_xz_slice,
    gorkov_grid_2d,
    C_WATER, F_HZ, OMEGA, RHO0, LAM,
)

OUT_DIR = PROJECT_ROOT / "results" / "deliverables" / "profiles"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Particle
A_P = 50.0e-6
RHO_P, C_P = 1050.0, 2350.0
KAPPA_W = 1.0 / (RHO0 * C_WATER**2)
KAPPA_P = 1.0 / (RHO_P * C_P**2)
F1 = 1.0 - KAPPA_P / KAPPA_W
F2 = 2.0 * (RHO_P - RHO0) / (2.0 * RHO_P + RHO0)

DPI = 200


def _find_deepest_trap_idx(U, xg, yg):
    """Return row, col of global U minimum in the interior."""
    border = 10
    U_inner = U.copy()
    U_inner[:border, :] = np.inf
    U_inner[-border:, :] = np.inf
    U_inner[:, :border] = np.inf
    U_inner[:, -border:] = np.inf
    iy, ix = np.unravel_index(np.argmin(U_inner), U_inner.shape)
    return iy, ix


def main() -> None:
    cache = load_fem_cache()

    # ── XY slice ────────────────────────────────────────────────────
    sl = interpolate_slice(cache, z=cache["z_star"], n_grid=800)
    p_grid = sl["p_grid"]
    dx, dy = sl["dx"], sl["dy"]
    xg_mm = sl["xg"] * 1e3
    yg_mm = sl["yg"] * 1e3

    U, _, _ = gorkov_grid_2d(
        p_grid, dx, dy, OMEGA, RHO0, C_WATER, A_P, F1, F2,
    )

    iy0, ix0 = _find_deepest_trap_idx(U, sl["xg"], sl["yg"])
    x0_mm = xg_mm[ix0]
    y0_mm = yg_mm[iy0]
    print(f"Deepest trap at ({x0_mm:.3f}, {y0_mm:.3f}) mm")

    # ── Horizontal & vertical line profiles ─────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Top-left: |p| along x at trap y
    ax = axes[0, 0]
    p_hline = np.abs(p_grid[iy0, :])
    ax.plot(xg_mm, p_hline, "b-", lw=1.2)
    ax.axvline(x0_mm, color="r", ls="--", lw=0.8, label=f"trap x = {x0_mm:.2f} mm")
    ax.set_xlabel("x  [mm]")
    ax.set_ylabel("|p|  [Pa]")
    ax.set_title(f"|p| line profile  (y = {y0_mm:.2f} mm)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Top-right: |p| along y at trap x
    ax = axes[0, 1]
    p_vline = np.abs(p_grid[:, ix0])
    ax.plot(yg_mm, p_vline, "b-", lw=1.2)
    ax.axvline(y0_mm, color="r", ls="--", lw=0.8, label=f"trap y = {y0_mm:.2f} mm")
    ax.set_xlabel("y  [mm]")
    ax.set_ylabel("|p|  [Pa]")
    ax.set_title(f"|p| line profile  (x = {x0_mm:.2f} mm)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom-left: U along x at trap y
    ax = axes[1, 0]
    ax.plot(xg_mm, U[iy0, :], "k-", lw=1.2)
    ax.axvline(x0_mm, color="r", ls="--", lw=0.8)
    ax.set_xlabel("x  [mm]")
    ax.set_ylabel("U  [J]")
    ax.set_title("Gor'kov U line profile  (horizontal)")
    ax.grid(True, alpha=0.3)

    # Bottom-right: U along y at trap x
    ax = axes[1, 1]
    ax.plot(yg_mm, U[:, ix0], "k-", lw=1.2)
    ax.axvline(y0_mm, color="r", ls="--", lw=0.8)
    ax.set_xlabel("y  [mm]")
    ax.set_ylabel("U  [J]")
    ax.set_title("Gor'kov U line profile  (vertical)")
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Pressure & Gor'kov line profiles through deepest trap\n"
        f"z = {sl['z']*1e3:.2f} mm  —  λ/2 = {LAM/2*1e3:.3f} mm",
        fontsize=13,
    )
    fig.tight_layout()
    path1 = OUT_DIR / "pressure_line_through_traps.png"
    fig.savefig(path1, dpi=DPI)
    plt.close(fig)
    print(f"  saved {path1}")

    # ── Axial (z) profile through the trap ──────────────────────────
    dom = cache["domain"]
    xz = interpolate_xz_slice(cache, y=sl["yg"][iy0], n_x=800, n_z=400)
    # Find column closest to trap x
    ix_xz = np.argmin(np.abs(xz["xg"] - sl["xg"][ix0]))
    p_z = np.abs(xz["p_grid"][:, ix_xz])
    zg_mm = xz["zg"] * 1e3

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(zg_mm, p_z, "b-", lw=1.2)
    ax.axvline(cache["z_star"] * 1e3, color="r", ls="--", lw=0.8,
               label=f"z* = {cache['z_star']*1e3:.2f} mm")
    ax.set_xlabel("z  [mm]")
    ax.set_ylabel("|p|  [Pa]")
    ax.set_title(
        f"Axial |p| through trap at "
        f"({x0_mm:.2f}, {y0_mm:.2f}) mm"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path2 = OUT_DIR / "pressure_line_axial.png"
    fig.savefig(path2, dpi=DPI)
    plt.close(fig)
    print(f"  saved {path2}")
    print("Done.")


if __name__ == "__main__":
    main()
