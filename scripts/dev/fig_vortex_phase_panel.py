#!/usr/bin/env python3
"""
Vortex phase interaction panel.

Produces a 1×3 panel showing how the Gor'kov potential landscape changes
with the vortex phase offset ψ.  The vortex is fixed at the grid centre;
no particles are drawn.  The wider ROI (±1.2 mm) lets the underlying
standing-wave lattice show through.

Output: results/dissertation_figures/vortex_entry_greedy/fig_vortex_phase_panel.{pdf,png}
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.patches import Rectangle

# ── Project path ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.lib.fem_cache_utils import (
    C_WATER, OMEGA, RHO0,
    default_particle_params, gorkov_grid_2d,
)
from acoustweezers.experiments.vortex_entry.fields.vortex_source import load_data
from acoustweezers.experiments.vortex_entry.fields.field_superposition import total_pressure

# ── Input data ─────────────────────────────────────────────────────────────────
FIELD_NPZ = (
    PROJECT_ROOT / "results" / "deliverables" / "vortex_stage_transport"
    / "transport" / "transport_case_for_gif.npz"
)

# ── Physical / source parameters ───────────────────────────────────────────────
ALPHA        = 2.0        # vortex amplitude weight
BETA         = 1.0        # standing-wave amplitude weight
APERTURE_M   = 3.5e-3
PROP_DIST_M  = 3.0e-3
FOCUS_M      = 3.0e-3
PPAR         = default_particle_params()

# ── Phase sweep ────────────────────────────────────────────────────────────────
PSI_VALUES = [0.0, 2 * np.pi / 3, 4 * np.pi / 3]
PSI_LABELS = ["0", "2π/3", "4π/3"]

# ── ROI half-width around vortex centre ────────────────────────────────────────
ROI_HALF_M = 1.2e-3   # ±1.2 mm  → 2.4 mm × 2.4 mm window

# ── Style ──────────────────────────────────────────────────────────────────────
CMAP          = "RdBu_r"
COL_BARRIER   = "#e67e22"
FIG_WIDTH_MM  = 190.0
DPI           = 300
SCALE_BAR_MM  = 0.5
FONT_SANS     = "DejaVu Sans"

plt.rcParams.update({
    "font.family":      FONT_SANS,
    "font.size":        11,
    "axes.labelsize":   12,
    "axes.labelweight": "bold",
    "axes.titlesize":   11,
    "figure.dpi":       DPI,
    "savefig.dpi":      DPI,
    "xtick.direction":  "in",
    "ytick.direction":  "in",
})

# ── Output ─────────────────────────────────────────────────────────────────────
OUT_DIR = PROJECT_ROOT / "results" / "dissertation_figures" / "vortex_entry_greedy"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _inches(mm: float) -> float:
    return mm / 25.4


def _gorkov(p_tot, dx, dy):
    return gorkov_grid_2d(
        p_tot, dx, dy, OMEGA, RHO0, C_WATER,
        PPAR["a"], PPAR["f1"], PPAR["f2"],
    )


def _roi_slices(xg, yg, x_min, x_max, y_min, y_max):
    ix_lo = max(0, int(np.searchsorted(xg, x_min)) - 1)
    ix_hi = min(len(xg), int(np.searchsorted(xg, x_max)) + 2)
    iy_lo = max(0, int(np.searchsorted(yg, y_min)) - 1)
    iy_hi = min(len(yg), int(np.searchsorted(yg, y_max)) + 2)
    return ix_lo, ix_hi, iy_lo, iy_hi


def _add_scale_bar(ax, x0_data: float, y0_data: float,
                   bar_len: float = SCALE_BAR_MM,
                   color: str = "white") -> None:
    bar_h = 0.025
    ax.add_patch(Rectangle(
        (x0_data, y0_data), bar_len, bar_h,
        facecolor=color, edgecolor="none", zorder=30, transform=ax.transData,
    ))
    ax.text(x0_data + bar_len / 2.0, y0_data + bar_h + 0.04,
            f"{bar_len:.1f} mm",
            ha="center", va="bottom", color=color, fontsize=9,
            fontweight="bold", zorder=31)


# ══════════════════════════════════════════════════════════════════════════════
# Main figure
# ══════════════════════════════════════════════════════════════════════════════

def make_figure(p_sw, vortex_gen, xg, yg, dx, dy, center_xy, r_barrier):
    cx, cy = center_xy

    # ROI bounds (clamped to grid)
    x_min = max(float(xg[0]),  cx - ROI_HALF_M)
    x_max = min(float(xg[-1]), cx + ROI_HALF_M)
    y_min = max(float(yg[0]),  cy - ROI_HALF_M)
    y_max = min(float(yg[-1]), cy + ROI_HALF_M)
    roi_w_mm = (x_max - x_min) * 1e3
    roi_h_mm = (y_max - y_min) * 1e3
    print(f"  ROI: {roi_w_mm:.2f} × {roi_h_mm:.2f} mm")

    ix_lo, ix_hi, iy_lo, iy_hi = _roi_slices(xg, yg, x_min, x_max, y_min, y_max)
    xg_roi = xg[ix_lo:ix_hi] * 1e3           # mm
    yg_roi = yg[iy_lo:iy_hi] * 1e3           # mm
    x_off  = xg_roi[0]
    y_off  = yg_roi[0]
    xg_roi -= x_off
    yg_roi -= y_off

    # Vortex centre in offset coordinates [mm]
    vc_mm = np.array([cx * 1e3 - x_off, cy * 1e3 - y_off])

    # ── Compute Gor'kov for each ψ ────────────────────────────────
    print("  Computing Gor'kov for 3 phases …")
    panel_data = []
    U_all = []
    for psi, label in zip(PSI_VALUES, PSI_LABELS):
        p_tot = total_pressure(p_sw, vortex_gen, psi, ALPHA, BETA, center_xy)
        U, Fx, Fy = _gorkov(p_tot, dx, dy)
        U_r  = U[iy_lo:iy_hi, ix_lo:ix_hi]
        Fx_r = Fx[iy_lo:iy_hi, ix_lo:ix_hi]
        Fy_r = Fy[iy_lo:iy_hi, ix_lo:ix_hi]
        panel_data.append((psi, label, U_r, Fx_r, Fy_r))
        U_all.append(U_r.ravel())
        print(f"    ψ = {label} done")

    # Shared colour limits
    U_stack = np.concatenate(U_all)
    vlo = float(np.percentile(U_stack, 2))
    vhi = float(np.percentile(U_stack, 98))
    cf_levels = np.linspace(vlo, vhi, 64)
    cr_levels = np.linspace(vlo, vhi, 8)

    # ── Layout ────────────────────────────────────────────────────
    fig_w_in = _inches(FIG_WIDTH_MM)
    aspect   = roi_h_mm / roi_w_mm
    fig_h_in = fig_w_in * aspect / 3.0 * 1.25

    fig, axes = plt.subplots(1, 3, figsize=(fig_w_in, fig_h_in),
                             constrained_layout=True)

    for ax, (psi, label, U_r, Fx_r, Fy_r) in zip(axes, panel_data):

        # Filled contours
        ax.contourf(xg_roi, yg_roi, U_r, levels=cf_levels,
                    cmap=CMAP, extend="both", zorder=1)

        # White iso-contour lines to highlight SW structure
        ax.contour(xg_roi, yg_roi, U_r, levels=cr_levels,
                   colors="white", linewidths=0.5, alpha=0.45, zorder=2)

        # Force arrows — direction only, white
        qs  = 7
        xs  = xg_roi[::qs]
        ys  = yg_roi[::qs]
        Fxs = Fx_r[::qs, ::qs]
        Fys = Fy_r[::qs, ::qs]
        mag = np.sqrt(Fxs**2 + Fys**2)
        ms  = np.where(mag > 0, mag, 1.0)
        ax.quiver(xs, ys, Fxs / ms, Fys / ms,
                  color="white", alpha=0.55, scale=24, width=0.0025,
                  headwidth=3.5, headlength=4.5, zorder=5)

        # Barrier ring (dashed, centred on vortex)
        ax.add_patch(mpatches.Circle(
            vc_mm, r_barrier * 1e3,
            fill=False, edgecolor=COL_BARRIER, linestyle="--",
            linewidth=1.2, alpha=0.85, zorder=8,
        ))

        # Scale bar
        _add_scale_bar(ax, roi_w_mm * 0.05, roi_h_mm * 0.04)

        # ψ label
        ax.text(0.03, 0.97, f"ψ = {label}",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=10, color="white",
                bbox=dict(facecolor="black", alpha=0.45, pad=2, edgecolor="none"),
                zorder=20)

        ax.set_xlim(0, roi_w_mm)
        ax.set_ylim(0, roi_h_mm)
        ax.set_xlabel("x [mm]", fontsize=12, fontweight="bold")
        ax.set_aspect("equal", adjustable="box")

    axes[0].set_ylabel("y [mm]", fontsize=12, fontweight="bold")
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)

    # Shared colorbar
    sm = plt.cm.ScalarMappable(
        cmap=CMAP, norm=mcolors.Normalize(vmin=vlo, vmax=vhi),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(),
                        orientation="vertical", shrink=0.9, pad=0.02)
    cbar.set_label("U  [J]", fontsize=11, fontweight="bold")
    cbar.ax.tick_params(labelsize=9)
    cbar.formatter = mticker.ScalarFormatter(useMathText=True)
    cbar.formatter.set_powerlimits((-20, -20))
    cbar.update_ticks()

    for ext in ("pdf", "png"):
        out = OUT_DIR / f"fig_vortex_phase_panel.{ext}"
        fig.savefig(str(out), dpi=DPI, bbox_inches="tight")
        print(f"  Saved: {out}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("Loading physics data …")
    (p_sw, xg, yg, dx, dy, traps_m, vortex_gen, r_barrier
     ) = load_data(APERTURE_M, PROP_DIST_M, FOCUS_M, field_npz=FIELD_NPZ)
    print(f"  Grid: {len(xg)}×{len(yg)}")
    print(f"  r_barrier = {r_barrier*1e6:.1f} µm")

    # Vortex fixed at grid centre
    center_xy = np.array([
        0.5 * (float(xg[0]) + float(xg[-1])),
        0.5 * (float(yg[0]) + float(yg[-1])),
    ])
    print(f"  Vortex centre: ({center_xy[0]*1e3:.3f}, {center_xy[1]*1e3:.3f}) mm")

    print("\nBuilding figure …")
    make_figure(p_sw, vortex_gen, xg, yg, dx, dy, center_xy, r_barrier)

    print(f"\nDone — output in: {OUT_DIR}")


if __name__ == "__main__":
    main()
