#!/usr/bin/env python3
"""
Standing-wave XY and XZ slice figures.

Generates publication-quality pressure-magnitude and phase plots on:
  1. XY plane at z* (trap plane)
  2. XZ plane at y = domain centre

Output
------
results/deliverables/standing_wave/sw_xy_magnitude.png
results/deliverables/standing_wave/sw_xy_phase.png
results/deliverables/standing_wave/sw_xz_magnitude.png
results/deliverables/standing_wave/sw_xz_phase.png
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
    interpolate_xz_slice,
    LAM,
)

OUT_DIR = PROJECT_ROOT / "results" / "deliverables" / "standing_wave"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Plot style
DPI = 200
CMAP_MAG = "viridis"
CMAP_PHASE = "twilight"


def _add_colorbar(ax, im, label: str) -> None:
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="4%", pad=0.06)
    plt.colorbar(im, cax=cax, label=label)


def plot_xy(cache: dict) -> None:
    """XY slice at z*."""
    sl = interpolate_slice(cache, z=cache["z_star"], n_grid=800)
    p = sl["p_grid"]
    xg_mm = sl["xg"] * 1e3
    yg_mm = sl["yg"] * 1e3
    extent = [xg_mm[0], xg_mm[-1], yg_mm[0], yg_mm[-1]]

    # ── Magnitude ──
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        np.abs(p), origin="lower", extent=extent,
        cmap=CMAP_MAG, aspect="equal",
    )
    ax.set_xlabel("x  [mm]")
    ax.set_ylabel("y  [mm]")
    ax.set_title(f"Standing wave |p|  —  z = {sl['z']*1e3:.2f} mm")
    _add_colorbar(ax, im, "|p|  [Pa]")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "sw_xy_magnitude.png", dpi=DPI)
    plt.close(fig)
    print(f"  saved sw_xy_magnitude.png")

    # ── Phase ──
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        np.angle(p), origin="lower", extent=extent,
        cmap=CMAP_PHASE, aspect="equal", vmin=-np.pi, vmax=np.pi,
    )
    ax.set_xlabel("x  [mm]")
    ax.set_ylabel("y  [mm]")
    ax.set_title(f"Standing wave arg(p)  —  z = {sl['z']*1e3:.2f} mm")
    _add_colorbar(ax, im, "phase  [rad]")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "sw_xy_phase.png", dpi=DPI)
    plt.close(fig)
    print(f"  saved sw_xy_phase.png")


def plot_xz(cache: dict) -> None:
    """XZ slice through domain centre."""
    dom = cache["domain"]
    y_mid = (dom["y_min"] + dom["y_max"]) / 2.0
    sl = interpolate_xz_slice(cache, y=y_mid, n_x=800, n_z=400)
    p = sl["p_grid"]
    xg_mm = sl["xg"] * 1e3
    zg_mm = sl["zg"] * 1e3
    extent = [xg_mm[0], xg_mm[-1], zg_mm[0], zg_mm[-1]]

    # ── Magnitude ──
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(
        np.abs(p), origin="lower", extent=extent,
        cmap=CMAP_MAG, aspect="auto",
    )
    ax.axhline(cache["z_star"] * 1e3, color="w", ls="--", lw=0.8, label="z*")
    ax.set_xlabel("x  [mm]")
    ax.set_ylabel("z  [mm]")
    ax.set_title(f"Standing wave |p|  —  XZ plane  (y = {y_mid*1e3:.2f} mm)")
    ax.legend(loc="upper right", fontsize=8)
    _add_colorbar(ax, im, "|p|  [Pa]")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "sw_xz_magnitude.png", dpi=DPI)
    plt.close(fig)
    print(f"  saved sw_xz_magnitude.png")

    # ── Phase ──
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(
        np.angle(p), origin="lower", extent=extent,
        cmap=CMAP_PHASE, aspect="auto", vmin=-np.pi, vmax=np.pi,
    )
    ax.axhline(cache["z_star"] * 1e3, color="w", ls="--", lw=0.8, label="z*")
    ax.set_xlabel("x  [mm]")
    ax.set_ylabel("z  [mm]")
    ax.set_title(f"Standing wave arg(p)  —  XZ plane  (y = {y_mid*1e3:.2f} mm)")
    ax.legend(loc="upper right", fontsize=8)
    _add_colorbar(ax, im, "phase  [rad]")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "sw_xz_phase.png", dpi=DPI)
    plt.close(fig)
    print(f"  saved sw_xz_phase.png")


def main() -> None:
    cache = load_fem_cache()
    print(f"Loaded {cache['n_dofs']:,} DOFs, |p|_max = {cache['p_max']:.2f} Pa")
    plot_xy(cache)
    plot_xz(cache)
    print("Done — all slices saved.")


if __name__ == "__main__":
    main()
