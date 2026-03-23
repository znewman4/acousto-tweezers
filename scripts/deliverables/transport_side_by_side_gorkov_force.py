#!/usr/bin/env python3
"""
Focused side-by-side transport GIF: vortex vs C-shape.

Uses calibrated parameters from the full comparison run.
Background shows signed U_Gorkov from the pre-existing gorkov utility,
matching the visual style used in bridge study figures.

Top row: ROI total perturbed fields.
Bottom row: full-domain total perturbed fields.

Output
------
results/deliverables/transport_side_by_side/
    transport_vortex_vs_cshape_gorkov_bridge_style.gif
"""
from __future__ import annotations

import sys
import os
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
from scripts.lib.fem_cache_utils import (
    C_WATER,
    OMEGA,
    RHO0,
    default_particle_params,
    gorkov_grid_2d,
)
from scripts.lib.perturbation_cshape import CShapePerturbation
from scripts.lib.perturbation_vortex import VortexPerturbation

PPAR = default_particle_params()

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
VORTEX_ALPHA = float(os.getenv("VORTEX_ALPHA", "2.10"))
VORTEX_PSI   = float(os.getenv("VORTEX_PSI", str(0.5 * np.pi)))

CSHAPE_ALPHA   = float(os.getenv("CSHAPE_ALPHA", "4.50"))
CSHAPE_PSI     = float(os.getenv("CSHAPE_PSI", str(1.5 * np.pi)))
CSHAPE_BSW_MIN = float(os.getenv("CSHAPE_BSW_MIN", "0.62"))

SW_SCALE = float(os.getenv("SW_SCALE", "1.0"))
CSHAPE_CONST_SW = os.getenv("CSHAPE_CONST_SW", "0") == "1"

# Schedule (ms)
T_SW_ONLY_MS  = 120.0
T_RAMP_ON_MS  = 220.0
T_ACTIVE_MS   = 260.0
T_RAMP_OFF_MS = 220.0
T_SETTLE_MS   = 800.0

# C-shape-only timing and dynamics controls (vortex remains unchanged).
CSHAPE_T_RAMP_ON_MS = float(os.getenv("CSHAPE_T_RAMP_ON_MS", "420.0"))
CSHAPE_T_ACTIVE_MS  = float(os.getenv("CSHAPE_T_ACTIVE_MS", "620.0"))
CSHAPE_T_RAMP_OFF_MS = float(os.getenv("CSHAPE_T_RAMP_OFF_MS", "260.0"))
CSHAPE_T_SETTLE_MS  = float(os.getenv("CSHAPE_T_SETTLE_MS", str(T_SETTLE_MS)))
CSHAPE_MOBILITY_SCALE = max(0.0, float(os.getenv("CSHAPE_MOBILITY_SCALE", "0.05")))

DT       = 1.0e-4   # s
N_FRAMES = 320

# GIF
GIF_DURATION_MS   = max(10, int(os.getenv("GIF_DURATION_MS", "110")))  # ms per physics frame
DISPLAY_SUBFRAMES = max(1, int(os.getenv("DISPLAY_SUBFRAMES", "3")))
_max_frames_env = os.getenv("MAX_OUTPUT_FRAMES", "").strip()
MAX_OUTPUT_FRAMES = max(0, int(_max_frames_env)) if _max_frames_env else 0
CMAP = "RdBu_r"
IMSHOW_INTERP = "bicubic"

# Gor'kov color scaling: per-frame auto-scale from data range (matches particle-merge demo style).
GORKOV_CLIP_LO = 0.5   # lower percentile clip
GORKOV_CLIP_HI = 99.5  # upper percentile clip

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
    cshape_bsw = SW_SCALE if CSHAPE_CONST_SW else CSHAPE_BSW_MIN
    return [
        {"duration_ms": T_SW_ONLY_MS,  "alpha_start": 0.0,          "alpha_end": 0.0,
         "bsw_start": SW_SCALE,       "bsw_end": SW_SCALE,
         "ctr_start": mid.copy(), "ctr_end": mid.copy(), "label": "SW only"},
        {"duration_ms": CSHAPE_T_RAMP_ON_MS,  "alpha_start": 0.0,          "alpha_end": CSHAPE_ALPHA,
         "bsw_start": SW_SCALE,       "bsw_end": cshape_bsw,
         "ctr_start": mid.copy(), "ctr_end": mid.copy(), "label": "ramp on"},
        {"duration_ms": CSHAPE_T_ACTIVE_MS, "alpha_start": CSHAPE_ALPHA, "alpha_end": CSHAPE_ALPHA,
         "bsw_start": cshape_bsw, "bsw_end": cshape_bsw,
         "ctr_start": mid.copy(), "ctr_end": mid.copy(), "label": "hold"},
        {"duration_ms": CSHAPE_T_RAMP_OFF_MS, "alpha_start": CSHAPE_ALPHA, "alpha_end": 0.0,
         "bsw_start": cshape_bsw, "bsw_end": SW_SCALE,
         "ctr_start": mid.copy(), "ctr_end": mid.copy(), "label": "ramp off"},
        {"duration_ms": CSHAPE_T_SETTLE_MS, "alpha_start": 0.0,          "alpha_end": 0.0,
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


def _total_pressure(p_sw, perturb_fn, psi, alpha_f, bsw_f, center_f):
    return bsw_f * p_sw + alpha_f * np.exp(1j * psi) * perturb_fn(center_f)


def _gorkov_fields(p_total: np.ndarray, dx: float, dy: float):
    return gorkov_grid_2d(
        p_total,
        dx,
        dy,
        OMEGA,
        RHO0,
        C_WATER,
        PPAR["a"],
        PPAR["f1"],
        PPAR["f2"],
    )


def _auto_limits(*arrays: np.ndarray) -> tuple[float, float]:
    """Per-frame vmin/vmax from combined data percentile — no symmetry constraint."""
    combined = np.concatenate([a.ravel() for a in arrays])
    return (float(np.percentile(combined, GORKOV_CLIP_LO)),
            float(np.percentile(combined, GORKOV_CLIP_HI)))


def _lerp(a, b, t: float):
    """Linear interpolation helper for display-only subframes."""
    return (1.0 - t) * a + t * b


def _draw_panel(ax, field2d, vmin, vmax, extent, x_mm, y_mm,
                traps_mm, current_mm, centres_path_mm,
                idx_A, idx_B, neigh_idx, title):
    ax.imshow(field2d, origin="lower", extent=extent,
              cmap=CMAP, vmin=vmin, vmax=vmax, aspect="equal",
              interpolation=IMSHOW_INTERP)

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
        mobility_scale=CSHAPE_MOBILITY_SCALE,
    )
    print(f"  done in {__import__('time').time()-t0:.1f}s, {len(res_c.times_s)} frames")

    n_phys_frames = min(len(res_v.times_s), len(res_c.times_s))
    n_frames_nominal = (n_phys_frames - 1) * DISPLAY_SUBFRAMES + 1
    n_frames = min(n_frames_nominal, MAX_OUTPUT_FRAMES) if MAX_OUTPUT_FRAMES > 0 else n_frames_nominal

    # -- Shared sensitive scales from SW-only U_Gorkov ---------------------
    dx = float(xg[1] - xg[0])
    dy = float(yg[1] - yg[0])
    dx_full = float(xg_v[1] - xg_v[0])
    dy_full = float(yg_v[1] - yg_v[0])

    # -- Render side-by-side GIF -------------------------------------------
    x_mm = xg * 1e3
    y_mm = yg * 1e3
    extent = [x_mm[0], x_mm[-1], y_mm[0], y_mm[-1]]

    x_full_mm = xg_v * 1e3
    y_full_mm = yg_v * 1e3
    extent_full = [x_full_mm[0], x_full_mm[-1], y_full_mm[0], y_full_mm[-1]]

    traps_mm = traps_m * 1e3

    out_name = os.getenv("OUT_GIF_NAME", "transport_vortex_vs_cshape_gorkov_bridge_style.gif")
    out_path = OUT_DIR / out_name
    print(
        f"Rendering {n_frames} display frames "
        f"({n_phys_frames} physics frames, subframes={DISPLAY_SUBFRAMES}, cap={MAX_OUTPUT_FRAMES or 'none'}) "
        f"-> {out_path.name} ..."
    )
    frames: List[Image.Image] = []

    for fi in range(n_frames):
        # Display-only interpolation between simulated frames for smoother motion.
        u = fi / float(DISPLAY_SUBFRAMES)
        i0 = int(np.floor(u))
        i1 = min(i0 + 1, n_phys_frames - 1)
        tau = float(u - i0)

        label_v = res_v.phase_labels[i0]
        label_c = res_c.phase_labels[i0]
        t_ms = float(_lerp(res_v.times_s[i0], res_v.times_s[i1], tau) * 1e3)

        alpha_v = float(_lerp(res_v.alphas[i0], res_v.alphas[i1], tau))
        beta_v = float(_lerp(res_v.betas_sw[i0], res_v.betas_sw[i1], tau))
        ctr_v = _lerp(res_v.centers[i0], res_v.centers[i1], tau)

        alpha_c = float(_lerp(res_c.alphas[i0], res_c.alphas[i1], tau))
        beta_c = float(_lerp(res_c.betas_sw[i0], res_c.betas_sw[i1], tau))
        ctr_c = _lerp(res_c.centers[i0], res_c.centers[i1], tau)

        p_tot_v = _total_pressure(
            p_sw, gen_vortex.get_field, VORTEX_PSI,
            alpha_v, beta_v, ctr_v)
        p_tot_c = _total_pressure(
            p_sw, gen_cshape.get_field, CSHAPE_PSI,
            alpha_c, beta_c, ctr_c)

        p_tot_v_full = _total_pressure(
            p_sw_full, gen_vortex_full.get_field, VORTEX_PSI,
            alpha_v, beta_v, ctr_v)
        p_tot_c_full = (
            beta_c * p_sw_full
            + alpha_c * np.exp(1j * CSHAPE_PSI) * p_lens_full
        )

        U_v, _, _ = _gorkov_fields(p_tot_v, dx, dy)
        U_c, _, _ = _gorkov_fields(p_tot_c, dx, dy)
        U_v_full, _, _ = _gorkov_fields(p_tot_v_full, dx_full, dy_full)
        U_c_full, _, _ = _gorkov_fields(p_tot_c_full, dx_full, dy_full)

        # Per-frame auto limits from combined data (matches particle-merge demo style)
        vmin_roi, vmax_roi = _auto_limits(U_v, U_c)
        vmin_full, vmax_full = _auto_limits(U_v_full, U_c_full)

        cur_v = _lerp(res_v.trajectories[i0], res_v.trajectories[i1], tau) * 1e3
        cur_c = _lerp(res_c.trajectories[i0], res_c.trajectories[i1], tau) * 1e3

        if i1 == i0:
            path_v = res_v.centers[: i0 + 1] * 1e3
            path_c = res_c.centers[: i0 + 1] * 1e3
        else:
            path_v = np.vstack([res_v.centers[: i0 + 1], ctr_v]) * 1e3
            path_c = np.vstack([res_c.centers[: i0 + 1], ctr_c]) * 1e3

        fig, axes = plt.subplots(2, 2, figsize=(11.2, 10.4))
        (ax_v, ax_c), (ax_vs, ax_cs) = axes

        # Top row: ROI U_Gorkov from total perturbed field
        _draw_panel(ax_v, U_v, vmin_roi, vmax_roi, extent, x_mm, y_mm,
                    traps_mm, cur_v, path_v, idx_A, idx_B, neigh_idx,
                f"Vortex — U_Gorkov (ROI)  ({label_v})")
        _draw_panel(ax_c, U_c, vmin_roi, vmax_roi, extent, x_mm, y_mm,
                    traps_mm, cur_c, path_c, idx_A, idx_B, neigh_idx,
                f"C-shape — U_Gorkov (ROI)  ({label_c})")

        # Bottom row: full-domain U_Gorkov from total perturbed fields
        _draw_panel(ax_vs, U_v_full, vmin_full, vmax_full, extent_full, x_full_mm, y_full_mm,
                    traps_mm, cur_v, path_v, idx_A, idx_B, neigh_idx,
            f"Vortex — U_Gorkov (full domain)  ({label_v})")
        _draw_panel(ax_cs, U_c_full, vmin_full, vmax_full, extent_full, x_full_mm, y_full_mm,
                    traps_mm, cur_c, path_c, idx_A, idx_B, neigh_idx,
            f"C-shape — U_Gorkov (full domain)  ({label_c})")

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

    gif_frame_ms = max(10, int(round(GIF_DURATION_MS / float(DISPLAY_SUBFRAMES))))
    frames[0].save(
        str(out_path),
        save_all=True,
        append_images=frames[1:],
        duration=gif_frame_ms,
        loop=0,
    )
    print(f"\nSaved: {out_path}")
    print(f"Output folder: {OUT_DIR}")


if __name__ == "__main__":
    main()
