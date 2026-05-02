#!/usr/bin/env python3
"""
Static PNG of the 9-particle ROI with A (orange) and B (blue) labelled,
using Gorkov-potential background (RdBu_r colormap) matching the
vortex_entry_hires_mpc GIF visual style.

Output: results/dev/roi_9particle/roi_9particle.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.lib.fem_cache_utils import (
    C_WATER, OMEGA, RHO0,
    default_particle_params, gorkov_grid_2d,
)

# ── Data ──────────────────────────────────────────────────────────
ROI_NPZ = (
    PROJECT_ROOT / "results"
    / "c_shape_lens_15mm_overlay_study_20260310_170620"
    / "npz" / "roi_fields.npz"
)

data = np.load(str(ROI_NPZ))
p_sw    = data["p_sw"]           # (400, 400) complex
xg      = data["xg"]             # (400,) metres
yg      = data["yg"]             # (400,) metres
traps_m = data["traps_m"]        # (9, 2) metres
idx_A   = int(data["idx_A"])     # trap A index
idx_B   = int(data["idx_B"])     # trap B index

PPAR = default_particle_params()
dx = float(xg[1] - xg[0])
dy = float(yg[1] - yg[0])

# ── Gorkov potential (pure SW) ─────────────────────────────────────
U, _, _ = gorkov_grid_2d(
    p_sw, dx, dy, OMEGA, RHO0, C_WATER,
    PPAR["a"], PPAR["f1"], PPAR["f2"],
)

# ── View extent: 9 traps + margin ─────────────────────────────────
MARGIN_MM = 0.40
traps_mm = traps_m * 1e3
x_lo = traps_mm[:, 0].min() - MARGIN_MM
x_hi = traps_mm[:, 0].max() + MARGIN_MM
y_lo = traps_mm[:, 1].min() - MARGIN_MM
y_hi = traps_mm[:, 1].max() + MARGIN_MM

# Percentile-clipped colour limits over the ROI (same as GIF)
ix_lo = max(0, int(np.searchsorted(xg, x_lo * 1e-3)))
ix_hi = min(len(xg), int(np.searchsorted(xg, x_hi * 1e-3)))
iy_lo = max(0, int(np.searchsorted(yg, y_lo * 1e-3)))
iy_hi = min(len(yg), int(np.searchsorted(yg, y_hi * 1e-3)))
U_roi = U[iy_lo:iy_hi, ix_lo:ix_hi]
vmin = float(np.percentile(U_roi, 0.5))
vmax = float(np.percentile(U_roi, 99.5))

# ── Visual constants (matching vortex_entry_hires_mpc GIF) ─────────
CMAP             = "RdBu_r"
COL_A            = "#e67e22"   # orange  (user request + GIF tone)
COL_B            = "#3498db"   # blue
COL_OTHERS       = "#95a5a6"   # light grey for neighbour traps
PARTICLE_RADIUS  = 0.045       # mm

# ── Figure ────────────────────────────────────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(6.5, 6.0))

x_mm   = xg * 1e3
y_mm   = yg * 1e3
extent = [x_mm[0], x_mm[-1], y_mm[0], y_mm[-1]]

im = ax.imshow(
    U, origin="lower", extent=extent,
    cmap=CMAP, vmin=vmin, vmax=vmax,
    aspect="equal", interpolation="bicubic",
)

# Colorbar
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label("Gor'kov potential $U$ (J)", fontsize=10)

# ── Neighbour trap particles (grey circles) ────────────────────────
for i in range(len(traps_m)):
    if i in (idx_A, idx_B):
        continue
    ax.add_patch(mpatches.Circle(
        (traps_mm[i, 0], traps_mm[i, 1]), PARTICLE_RADIUS,
        facecolor=COL_OTHERS, edgecolor="white", linewidth=0.8,
        zorder=5, alpha=0.80,
    ))

# ── Particle B (blue) ──────────────────────────────────────────────
ax.add_patch(mpatches.Circle(
    (traps_mm[idx_B, 0], traps_mm[idx_B, 1]), PARTICLE_RADIUS,
    facecolor=COL_B, edgecolor="white", linewidth=1.0,
    zorder=7, alpha=0.95,
))
ax.annotate(
    "B",
    xy=(traps_mm[idx_B, 0], traps_mm[idx_B, 1]),
    fontsize=10, fontweight="bold", color="white",
    ha="center", va="center", zorder=9,
)

# ── Particle A (orange) ────────────────────────────────────────────
ax.add_patch(mpatches.Circle(
    (traps_mm[idx_A, 0], traps_mm[idx_A, 1]), PARTICLE_RADIUS,
    facecolor=COL_A, edgecolor="white", linewidth=1.0,
    zorder=7, alpha=0.95,
))
ax.annotate(
    "A",
    xy=(traps_mm[idx_A, 0], traps_mm[idx_A, 1]),
    fontsize=10, fontweight="bold", color="white",
    ha="center", va="center", zorder=9,
)

# ── Legend ─────────────────────────────────────────────────────────
ax.scatter([], [], c=COL_A, s=40, label="A (vortex)")
ax.scatter([], [], c=COL_B, s=40, label="B (SW trap)")
ax.scatter([], [], c=COL_OTHERS, s=40, label="Lattice traps")
ax.legend(loc="upper right", fontsize=9, framealpha=0.8)

# ── Axes ───────────────────────────────────────────────────────────
ax.set_xlim(x_lo, x_hi)
ax.set_ylim(y_lo, y_hi)
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("x [mm]", fontsize=11)
ax.set_ylabel("y [mm]", fontsize=11)
ax.set_title("ROI — 9-particle lattice (SW only, α = 0)", fontsize=12)

fig.tight_layout()

# ── Save ───────────────────────────────────────────────────────────
OUT_DIR = PROJECT_ROOT / "results" / "dev" / "roi_9particle"
OUT_DIR.mkdir(parents=True, exist_ok=True)
out_path = OUT_DIR / "roi_9particle.png"
fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path}")
