#!/usr/bin/env python3
"""
Focused side-by-side transport GIF: vortex vs C-shape.

Uses calibrated parameters from the full comparison run.
Both panels share a common vmax derived from the pure SW field so that
the standing-wave phase looks identical on both sides before any
perturbation is applied.

Top row: ROI total perturbed fields.
Bottom row: full-domain total perturbed fields.

Output
------
results/deliverables/transport_side_by_side/
    transport_vortex_vs_cshape.gif
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.interpolate import RegularGridInterpolator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lib.particle_dynamics_utils import (
    run_transport,
)
from scripts.lib.perturbation_cshape import CShapePerturbation
from scripts.lib.perturbation_vortex import VortexPerturbation

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUT_DIR = PROJECT_ROOT / "results" / "deliverables" / "transport_side_by_side"

VORTEX_NPZ = (
    PROJECT_ROOT / "results" / "deliverables" / "vortex_stage_transport"
    / "transport" / "transport_case_for_gif.npz"
)
OVERLAY_NPZ = (
    PROJECT_ROOT / "results" / "c_shape_lens_15mm_overlay_study_20260310_170620"
    / "npz" / "roi_fields.npz"
)

# ---------------------------------------------------------------------------
# Calibrated controls  (from overdamped recalibration sweep)
# ---------------------------------------------------------------------------
VORTEX_ALPHA = 2.10
VORTEX_PSI   = 0.5 * np.pi          # psi = 0.5π gave progress=0.995

CSHAPE_ALPHA   = 4.50
CSHAPE_PSI     = 1.5 * np.pi
CSHAPE_BSW_MIN = 0.42

SW_SCALE = 1.0

# Schedule (ms)
T_SW_ONLY_MS  = 120.0
T_RAMP_ON_MS  = 220.0
T_ACTIVE_MS   = 260.0
T_RAMP_OFF_MS = 220.0
T_SETTLE_MS   = 800.0

DT       = 1.0e-4   # s
N_FRAMES = 320

# GIF
GIF_DURATION_MS = 55
CMAP = "viridis"

COL_A     = "#e74c3c"
COL_B     = "#3498db"
COL_NEIGH = "#95a5a6"
COL_HOME  = "#2ecc71"
COL_CTR   = "#f39c12"


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------
def _vortex_phases(A: np.ndarray, B: np.ndarray) -> List[Dict[str, Any]]:
    return [
        {"duration_ms": T_SW_ONLY_MS,  "alpha_start": 0.0,          "alpha_end": 0.0,
         "bsw_start": SW_SCALE, "bsw_end": SW_SCALE,
         "ctr_start": A.copy(), "ctr_end": A.copy(), "label": "SW only"},
        {"duration_ms": T_RAMP_ON_MS,  "alpha_start": 0.0,          "alpha_end": VORTEX_ALPHA,
         "bsw_start": SW_SCALE, "bsw_end": SW_SCALE,
         "ctr_start": A.copy(), "ctr_end": A.copy(), "label": "ramp on"},
        {"duration_ms": T_ACTIVE_MS,   "alpha_start": VORTEX_ALPHA, "alpha_end": VORTEX_ALPHA,
         "bsw_start": SW_SCALE, "bsw_end": SW_SCALE,
         "ctr_start": A.copy(), "ctr_end": B.copy(), "label": "translate"},
        {"duration_ms": T_RAMP_OFF_MS, "alpha_start": VORTEX_ALPHA, "alpha_end": 0.0,
         "bsw_start": SW_SCALE, "bsw_end": SW_SCALE,
         "ctr_start": B.copy(), "ctr_end": B.copy(), "label": "ramp off"},
        {"duration_ms": T_SETTLE_MS,   "alpha_start": 0.0,          "alpha_end": 0.0,
         "bsw_start": SW_SCALE, "bsw_end": SW_SCALE,
         "ctr_start": B.copy(), "ctr_end": B.copy(), "label": "SW release"},
    ]


def _cshape_phases(A: np.ndarray, B: np.ndarray) -> List[Dict[str, Any]]:
    mid = 0.5 * (A + B)
    return [
        {"duration_ms": T_SW_ONLY_MS,  "alpha_start": 0.0,          "alpha_end": 0.0,
         "bsw_start": SW_SCALE,       "bsw_end": SW_SCALE,
         "ctr_start": mid.copy(), "ctr_end": mid.copy(), "label": "SW only"},
        {"duration_ms": T_RAMP_ON_MS,  "alpha_start": 0.0,          "alpha_end": CSHAPE_ALPHA,
         "bsw_start": SW_SCALE,       "bsw_end": CSHAPE_BSW_MIN,
         "ctr_start": mid.copy(), "ctr_end": mid.copy(), "label": "ramp on"},
        {"duration_ms": T_ACTIVE_MS,   "alpha_start": CSHAPE_ALPHA, "alpha_end": CSHAPE_ALPHA,
         "bsw_start": CSHAPE_BSW_MIN, "bsw_end": CSHAPE_BSW_MIN,
         "ctr_start": mid.copy(), "ctr_end": mid.copy(), "label": "hold"},
        {"duration_ms": T_RAMP_OFF_MS, "alpha_start": CSHAPE_ALPHA, "alpha_end": 0.0,
         "bsw_start": CSHAPE_BSW_MIN, "bsw_end": SW_SCALE,
         "ctr_start": mid.copy(), "ctr_end": mid.copy(), "label": "ramp off"},
        {"duration_ms": T_SETTLE_MS,   "alpha_start": 0.0,          "alpha_end": 0.0,
         "bsw_start": SW_SCALE,       "bsw_end": SW_SCALE,
         "ctr_start": mid.copy(), "ctr_end": mid.copy(), "label": "SW release"},
    ]


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
_PHASE_COL = {
    "sw only":     "#2e7d32",
    "ramp on":     "#ef6c00",
    "translate":   "#6a1b9a",
    "hold":        "#6a1b9a",
    "ramp off":    "#1565c0",
    "sw release":  "#2e7d32",
}


def _phase_color(label: str) -> str:
    return _PHASE_COL.get(label.lower().strip(), "#424242")


def _pressure_abs(p_sw, perturb_fn, psi, alpha_f, bsw_f, center_f):
    return np.abs(bsw_f * p_sw + alpha_f * np.exp(1j * psi) * perturb_fn(center_f))


def _draw_panel(ax, p_abs, vmax, extent, x_mm, y_mm,
                traps_mm, current_mm, centres_path_mm,
                idx_A, idx_B, neigh_idx, title):
    ax.imshow(p_abs, origin="lower", extent=extent,
              cmap=CMAP, vmin=0.0, vmax=vmax, aspect="equal")

    # Home trap markers
    ax.scatter(traps_mm[:, 0], traps_mm[:, 1],
               marker="x", c=COL_HOME, s=28, linewidths=0.8, zorder=4)

    # Home-to-current displacement vectors
    for i in neigh_idx:
        ax.plot([traps_mm[i, 0], current_mm[i, 0]],
                [traps_mm[i, 1], current_mm[i, 1]],
                color=COL_NEIGH, lw=0.8, alpha=0.65, zorder=3)
    for i, col in [(idx_A, COL_A), (idx_B, COL_B)]:
        ax.plot([traps_mm[i, 0], current_mm[i, 0]],
                [traps_mm[i, 1], current_mm[i, 1]],
                color=col, lw=1.2, alpha=0.9, zorder=3)

    # Particles
    ax.scatter(current_mm[neigh_idx, 0], current_mm[neigh_idx, 1],
               c=COL_NEIGH, s=28, zorder=6)
    ax.scatter([current_mm[idx_B, 0]], [current_mm[idx_B, 1]],
               c=COL_B, s=44, zorder=7)
    ax.scatter([current_mm[idx_A, 0]], [current_mm[idx_A, 1]],
               c=COL_A, s=44, zorder=7)

    # Perturbation centre trail
    ax.plot(centres_path_mm[:, 0], centres_path_mm[:, 1],
            color=COL_CTR, lw=0.9, alpha=0.75, zorder=5)
    ax.plot(centres_path_mm[-1, 0], centres_path_mm[-1, 1],
            marker="+", color=COL_CTR, ms=10, mew=1.5, zorder=8)

    ax.set_xlim(x_mm[0], x_mm[-1])
    ax.set_ylim(y_mm[0], y_mm[-1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # -- Load data ----------------------------------------------------------
    print("Loading data...")
    ov = np.load(OVERLAY_NPZ)
    xg       = ov["xg"].astype(float)
    yg       = ov["yg"].astype(float)
    p_sw     = ov["p_sw"].astype(complex)
    p_lens   = ov["p_lens_roi"].astype(complex)
    traps_m  = ov["traps_m"].astype(float)
    idx_A    = int(ov["idx_A"])
    idx_B    = int(ov["idx_B"])

    vd    = np.load(VORTEX_NPZ)
    xg_v  = vd["xg"].astype(float)
    yg_v  = vd["yg"].astype(float)
    p_sw_full = vd["p_sw"].astype(complex)
    p_v   = vd["p_vortex_centered"].astype(complex)

    neigh_idx = np.array(sorted(set(range(len(traps_m))) - {idx_A, idx_B}), dtype=int)

    A_xy = traps_m[idx_A]
    B_xy = traps_m[idx_B]

    print(f"  Trap A idx={idx_A}: {A_xy*1e3} mm")
    print(f"  Trap B idx={idx_B}: {B_xy*1e3} mm")
    print(f"  ROI domain: [{xg[0]*1e3:.3f}, {xg[-1]*1e3:.3f}] mm")
    print(f"  Full domain: [{xg_v[0]*1e3:.3f}, {xg_v[-1]*1e3:.3f}] mm")

    # -- Perturbation objects -----------------------------------------------
    gen_vortex = VortexPerturbation(p_v, xg_v, yg_v, out_xg=xg, out_yg=yg)
    gen_vortex_full = VortexPerturbation(p_v, xg_v, yg_v, out_xg=xg_v, out_yg=yg_v)
    gen_cshape = CShapePerturbation(p_lens, xg, yg)

    # Interpolate the ROI C-shape perturbation onto full-domain grid.
    yy_full, xx_full = np.meshgrid(yg_v, xg_v, indexing="ij")
    pts_full = np.column_stack([yy_full.ravel(), xx_full.ravel()])
    interp_re = RegularGridInterpolator((yg, xg), np.real(p_lens), bounds_error=False, fill_value=0.0)
    interp_im = RegularGridInterpolator((yg, xg), np.imag(p_lens), bounds_error=False, fill_value=0.0)
    p_lens_full = (interp_re(pts_full) + 1j * interp_im(pts_full)).reshape(yy_full.shape)

    # -- Schedules ----------------------------------------------------------
    phases_v = _vortex_phases(A_xy, B_xy)
    phases_c = _cshape_phases(A_xy, B_xy)

    # -- Simulate -----------------------------------------------------------
    print("Running VORTEX transport...")
    t0 = __import__("time").time()
    res_v = run_transport(
        p_sw=p_sw,
        perturbation_fn=gen_vortex.get_field,
        phases=phases_v,
        psi=VORTEX_PSI,
        initial_positions=traps_m.copy(),
        xg=xg,
        yg=yg,
        idx_A=idx_A,
        idx_B=idx_B,
        dt=DT,
        n_frames=N_FRAMES,
        n_keyframes=20,
    )
    print(f"  done in {__import__('time').time()-t0:.1f}s, {len(res_v.times_s)} frames")

    print("Running C-SHAPE transport...")
    t0 = __import__("time").time()
    res_c = run_transport(
        p_sw=p_sw,
        perturbation_fn=gen_cshape.get_field,
        phases=phases_c,
        psi=CSHAPE_PSI,
        initial_positions=traps_m.copy(),
        xg=xg,
        yg=yg,
        idx_A=idx_A,
        idx_B=idx_B,
        dt=DT,
        n_frames=N_FRAMES,
        n_keyframes=20,
    )
    print(f"  done in {__import__('time').time()-t0:.1f}s, {len(res_c.times_s)} frames")

    n_frames = min(len(res_v.times_s), len(res_c.times_s))

    # -- Shared scales from pure SW field ----------------------------------
    vmax = float(np.percentile(np.abs(SW_SCALE * p_sw), 99.5))
    vmax_full = float(np.percentile(np.abs(SW_SCALE * p_sw_full), 99.5))
    print(f"  Shared ROI vmax (SW-only 99.5th percentile): {vmax:.4f}")
    print(f"  Shared full-domain vmax (SW-only 99.5th percentile): {vmax_full:.4f}")

    # -- Render side-by-side GIF -------------------------------------------
    x_mm = xg * 1e3
    y_mm = yg * 1e3
    extent = [x_mm[0], x_mm[-1], y_mm[0], y_mm[-1]]

    x_full_mm = xg_v * 1e3
    y_full_mm = yg_v * 1e3
    extent_full = [x_full_mm[0], x_full_mm[-1], y_full_mm[0], y_full_mm[-1]]

    traps_mm = traps_m * 1e3

    out_path = OUT_DIR / "transport_vortex_vs_cshape.gif"
    print(f"Rendering {n_frames} frames -> {out_path.name} ...")
    frames: List[Image.Image] = []

    for fi in range(n_frames):
        label_v = res_v.phase_labels[fi]
        label_c = res_c.phase_labels[fi]
        t_ms = res_v.times_s[fi] * 1e3

        p_abs_v = _pressure_abs(
            p_sw, gen_vortex.get_field, VORTEX_PSI,
            float(res_v.alphas[fi]), float(res_v.betas_sw[fi]), res_v.centers[fi])
        p_abs_c = _pressure_abs(
            p_sw, gen_cshape.get_field, CSHAPE_PSI,
            float(res_c.alphas[fi]), float(res_c.betas_sw[fi]), res_c.centers[fi])

        p_abs_v_full = _pressure_abs(
            p_sw_full, gen_vortex_full.get_field, VORTEX_PSI,
            float(res_v.alphas[fi]), float(res_v.betas_sw[fi]), res_v.centers[fi])
        p_abs_c_full = np.abs(
            float(res_c.betas_sw[fi]) * p_sw_full
            + float(res_c.alphas[fi]) * np.exp(1j * CSHAPE_PSI) * p_lens_full
        )

        cur_v = res_v.trajectories[fi] * 1e3
        cur_c = res_c.trajectories[fi] * 1e3
        path_v = res_v.centers[: fi + 1] * 1e3
        path_c = res_c.centers[: fi + 1] * 1e3

        fig, axes = plt.subplots(2, 2, figsize=(11.2, 10.4))
        (ax_v, ax_c), (ax_vs, ax_cs) = axes

        # Top row: total perturbed acoustic field
        _draw_panel(ax_v, p_abs_v, vmax, extent, x_mm, y_mm,
                    traps_mm, cur_v, path_v, idx_A, idx_B, neigh_idx,
                    f"Vortex — total field  ({label_v})")
        _draw_panel(ax_c, p_abs_c, vmax, extent, x_mm, y_mm,
                    traps_mm, cur_c, path_c, idx_A, idx_B, neigh_idx,
                    f"C-shape — total field  ({label_c})")

        # Bottom row: full-domain total perturbed fields
        _draw_panel(ax_vs, p_abs_v_full, vmax_full, extent_full, x_full_mm, y_full_mm,
                    traps_mm, cur_v, path_v, idx_A, idx_B, neigh_idx,
                f"Vortex — total field (full domain)  ({label_v})")
        _draw_panel(ax_cs, p_abs_c_full, vmax_full, extent_full, x_full_mm, y_full_mm,
                    traps_mm, cur_c, path_c, idx_A, idx_B, neigh_idx,
                f"C-shape — total field (full domain)  ({label_c})")

        fig.suptitle(
            f"t = {t_ms:.0f} ms  |  frame {fi+1}/{n_frames}",
            fontsize=11,
            color=_phase_color(label_v),
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        frames.append(Image.fromarray(buf.reshape(h, w, 4)).convert("RGB"))
        plt.close(fig)

        if (fi + 1) % 50 == 0:
            print(f"  {fi+1}/{n_frames}")

    frames[0].save(
        str(out_path),
        save_all=True,
        append_images=frames[1:],
        duration=GIF_DURATION_MS,
        loop=0,
    )
    print(f"\nSaved: {out_path}")
    print(f"Output folder: {OUT_DIR}")


if __name__ == "__main__":
    main()
