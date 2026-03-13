#!/usr/bin/env python3
"""
Trap detection on the FEM standing-wave Gor'kov potential.

Detects Hessian-stable local minima of U on the XY trap plane,
annotates them on a composite |p| + U figure, and saves trap data JSON.

Output
------
results/deliverables/trap_map/trap_localisation.png
results/deliverables/trap_map/trap_data.json
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
    C_WATER, F_HZ, OMEGA, RHO0, LAM,
)

OUT_DIR = PROJECT_ROOT / "results" / "deliverables" / "trap_map"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Particle — polystyrene
A_P = 50.0e-6
RHO_P, C_P = 1050.0, 2350.0
KAPPA_W = 1.0 / (RHO0 * C_WATER**2)
KAPPA_P = 1.0 / (RHO_P * C_P**2)
F1 = 1.0 - KAPPA_P / KAPPA_W
F2 = 2.0 * (RHO_P - RHO0) / (2.0 * RHO_P + RHO0)

TRAP_SP = LAM / 2.0   # expected λ/2 spacing
DPI = 200


def detect_traps(U: np.ndarray, xg: np.ndarray, yg: np.ndarray,
                 p_abs: np.ndarray):
    """
    Morphological minima + Hessian stability check.

    Returns list of dicts with trap info and (n, 2) array of positions.
    """
    dx = xg[1] - xg[0]
    min_sep_pts = max(3, int(0.4 * TRAP_SP / dx))
    nbhd = 2 * min_sep_pts + 1

    local_min = minimum_filter(U, size=nbhd)
    mask = (U == local_min)

    # Exclude borders
    border = max(5, min_sep_pts)
    mask[:border, :] = False
    mask[-border:, :] = False
    mask[:, :border] = False
    mask[:, -border:] = False

    # Only keep candidates below 50 % of U range
    U_range = U.max() - U.min()
    mask &= (U < U.min() + 0.50 * U_range)

    iy, ix = np.where(mask)
    print(f"  {len(iy)} candidates before Hessian check")

    # Hessian via numerical derivatives
    dUdx = np.gradient(U, dx, axis=1)
    dUdy = np.gradient(U, dx, axis=0)
    Hxx = np.gradient(dUdx, dx, axis=1)
    Hyy = np.gradient(dUdy, dx, axis=0)
    Hxy = np.gradient(dUdx, dx, axis=0)

    traps = []
    for ci in range(len(iy)):
        hxx = Hxx[iy[ci], ix[ci]]
        hyy = Hyy[iy[ci], ix[ci]]
        hxy = Hxy[iy[ci], ix[ci]]
        tr = hxx + hyy
        det = hxx * hyy - hxy**2
        disc = max(0.0, tr**2 - 4 * det)
        lam_min = (tr - np.sqrt(disc)) / 2.0
        lam_max = (tr + np.sqrt(disc)) / 2.0
        if lam_min > 0:
            traps.append({
                "x_mm": float(xg[ix[ci]] * 1e3),
                "y_mm": float(yg[iy[ci]] * 1e3),
                "p_abs_Pa": float(p_abs[iy[ci], ix[ci]]),
                "U_J": float(U[iy[ci], ix[ci]]),
                "lam_min": float(lam_min),
                "lam_max": float(lam_max),
            })

    positions = np.array([[t["x_mm"], t["y_mm"]] for t in traps]) if traps else np.zeros((0, 2))
    print(f"  {len(traps)} Hessian-stable traps detected")
    return traps, positions


def main() -> None:
    cache = load_fem_cache()
    sl = interpolate_slice(cache, z=cache["z_star"], n_grid=800)
    p_grid = sl["p_grid"]
    dx, dy = sl["dx"], sl["dy"]

    print("Computing Gor'kov potential …")
    U, Fx, Fy = gorkov_grid_2d(
        p_grid, dx, dy, OMEGA, RHO0, C_WATER, A_P, F1, F2,
    )

    print("Detecting traps …")
    traps, pos_mm = detect_traps(U, sl["xg"], sl["yg"], np.abs(p_grid))

    # ── Save JSON ──
    json_path = OUT_DIR / "trap_data.json"
    json_out = {
        "n_traps": len(traps),
        "z_mm": float(sl["z"] * 1e3),
        "f1": F1, "f2": F2,
        "particle_radius_um": A_P * 1e6,
        "traps": traps,
    }
    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"  saved {json_path}")

    # ── Plot: 3-panel ──
    xg_mm = sl["xg"] * 1e3
    yg_mm = sl["yg"] * 1e3
    extent = [xg_mm[0], xg_mm[-1], yg_mm[0], yg_mm[-1]]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # Panel 1: |p|
    ax = axes[0]
    im = ax.imshow(np.abs(p_grid), origin="lower", extent=extent,
                   cmap="viridis", aspect="equal")
    ax.set_title("|p|  [Pa]")
    ax.set_xlabel("x  [mm]"); ax.set_ylabel("y  [mm]")
    div = make_axes_locatable(ax)
    plt.colorbar(im, cax=div.append_axes("right", size="4%", pad=0.06), label="Pa")

    # Panel 2: Gor'kov U
    ax = axes[1]
    im = ax.imshow(U, origin="lower", extent=extent,
                   cmap="RdBu_r", aspect="equal")
    ax.set_title("Gor'kov U  [J]")
    ax.set_xlabel("x  [mm]"); ax.set_ylabel("y  [mm]")
    div = make_axes_locatable(ax)
    plt.colorbar(im, cax=div.append_axes("right", size="4%", pad=0.06), label="J")

    # Panel 3: |p| with traps
    ax = axes[2]
    im = ax.imshow(np.abs(p_grid), origin="lower", extent=extent,
                   cmap="viridis", aspect="equal")
    if len(pos_mm):
        ax.scatter(pos_mm[:, 0], pos_mm[:, 1],
                   s=50, facecolors="none", edgecolors="red", linewidths=1.2,
                   label=f"{len(traps)} traps")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.7)
    ax.set_title("|p| + detected traps")
    ax.set_xlabel("x  [mm]"); ax.set_ylabel("y  [mm]")
    div = make_axes_locatable(ax)
    plt.colorbar(im, cax=div.append_axes("right", size="4%", pad=0.06), label="Pa")

    fig.suptitle(
        f"Trap localisation — z = {sl['z']*1e3:.2f} mm, "
        f"λ/2 = {TRAP_SP*1e3:.3f} mm, "
        f"{len(traps)} stable traps",
        fontsize=13,
    )
    fig.tight_layout()
    fig_path = OUT_DIR / "trap_localisation.png"
    fig.savefig(fig_path, dpi=DPI)
    plt.close(fig)
    print(f"  saved {fig_path}")
    print("Done.")


if __name__ == "__main__":
    main()
