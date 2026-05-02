#!/usr/bin/env python3
"""
Dissertation figures — MPC vortex-entry particle merging.

Loads existing simulation timeseries CSV (or regenerates via source script)
and produces four publication-quality figures:

  fig1_MPC_gif.gif              — animated Gor'kov field with MPC ψ
  fig2_psi_sequence.pdf/.png       — ψ(t) step function + d(A,B) approach
  fig3_static_force_arrows.pdf/.png — vortex-only vs vortex+SW force arrows
  fig4_gorkov_sequence.pdf/.png    — Gor'kov potential at 3 key ψ transitions

Data:
  results/dev/vortex_entry_hires_inertial/20260402_183529/timeseries.csv
Physics re-built via load_data() (identical params to that run).

Output: results/dissertation_figures/vortex_entry_MPC/
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from PIL import Image

# ── Project path ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.lib.fem_cache_utils import (
    C_WATER, F_HZ, OMEGA, RHO0,
    default_particle_params, gorkov_grid_2d,
)
from acoustweezers.experiments.vortex_entry.fields.vortex_source import load_data
from acoustweezers.experiments.vortex_entry.fields.field_superposition import total_pressure

# ── Input data ────────────────────────────────────────────────────────────────
FIELD_NPZ = (
    PROJECT_ROOT / "results" / "deliverables" / "vortex_stage_transport"
    / "transport" / "transport_case_for_gif.npz"
)
TIMESERIES_CSV = (
    PROJECT_ROOT / "results" / "dev" / "vortex_entry_hires_inertial"
    / "20260402_183529" / "timeseries.csv"
)

# ── Physical constants (from run manifest) ────────────────────────────────────
ALPHA        = 2.0
BETA         = 1.0
R_BARRIER_M  = 382.4e-6        # m  (from manifest r_barrier_um: 382.4)
DT_S         = 1e-4            # s  per timestep
IDX_A        = 187
IDX_B        = 96
PPAR         = default_particle_params()

# ── Vortex source parameters (must match original run) ────────────────────────
APERTURE_M   = 3.5e-3
PROP_DIST_M  = 3.0e-3
FOCUS_M      = 3.0e-3

# ── Style constants ───────────────────────────────────────────────────────────
CMAP         = "RdBu_r"
COL_A        = "#e05c2e"   # orange-red for particle A
COL_B        = "#4a90d9"   # light blue for particle B
COL_NEIGH    = "#888888"
COL_BARRIER  = "#e67e22"
COL_VC       = "#f39c12"
FIG_WIDTH_MM = 190.0
DPI          = 300
PARTICLE_RADIUS_MM = 0.045
SCALE_BAR_MM = 0.5
FONT_SANS    = "DejaVu Sans"

plt.rcParams.update({
    "font.family":   FONT_SANS,
    "font.size":     11,
    "axes.labelsize": 12,
    "axes.labelweight": "bold",
    "axes.titlesize": 11,
    "figure.dpi":    DPI,
    "savefig.dpi":   DPI,
    "xtick.direction": "in",
    "ytick.direction": "in",
})

# ── Output directory ──────────────────────────────────────────────────────────
OUT_DIR = PROJECT_ROOT / "results" / "dissertation_figures" / "vortex_entry_MPC"
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Output directory: {OUT_DIR}")


# ══════════════════════════════════════════════════════════════════════════════
# Helper utilities
# ══════════════════════════════════════════════════════════════════════════════

def _inches(mm: float) -> float:
    return mm / 25.4


def gorkov(p_tot, dx, dy):
    """Compute (U, Fx, Fy) Gor'kov potential and force on the full grid."""
    return gorkov_grid_2d(
        p_tot, dx, dy, OMEGA, RHO0, C_WATER,
        PPAR["a"], PPAR["f1"], PPAR["f2"],
    )


def _compute_global_roi(df: pd.DataFrame, r_barrier: float,
                        xg: np.ndarray, yg: np.ndarray,
                        pad_factor: float = 1.5) -> Tuple[float, float, float, float]:
    """
    Fixed ROI encompassing all A/B positions across the simulation,
    padded by pad_factor × vortex_diameter (= 2*r_barrier).
    Returns (x_min, x_max, y_min, y_max) in metres, clamped to grid.
    """
    pad = pad_factor * 2.0 * r_barrier
    all_x = np.concatenate([df["A_x_m"].values, df["B_x_m"].values])
    all_y = np.concatenate([df["A_y_m"].values, df["B_y_m"].values])
    x_min = max(float(xg[0]),  float(all_x.min()) - pad)
    x_max = min(float(xg[-1]), float(all_x.max()) + pad)
    y_min = max(float(yg[0]),  float(all_y.min()) - pad)
    y_max = min(float(yg[-1]), float(all_y.max()) + pad)
    return x_min, x_max, y_min, y_max


def _compute_late_roi(df: pd.DataFrame, r_barrier: float,
                      xg: np.ndarray, yg: np.ndarray,
                      v_step_start: int = 16,
                      pad_m: float = None) -> Tuple[float, float, float, float]:
    """
    Tight ROI covering only the late-stage interaction (v_step >= v_step_start).
    Uses B and vortex-centre positions only (not A, which has wider range).
    pad_m defaults to one barrier radius.
    """
    if pad_m is None:
        pad_m = r_barrier
    active = df[df["v_step"] >= v_step_start]
    all_x = np.concatenate([active["B_x_m"].values, active["vc_x_m"].values])
    all_y = np.concatenate([active["B_y_m"].values, active["vc_y_m"].values])
    x_min = max(float(xg[0]),  float(all_x.min()) - pad_m)
    x_max = min(float(xg[-1]), float(all_x.max()) + pad_m)
    y_min = max(float(yg[0]),  float(all_y.min()) - pad_m)
    y_max = min(float(yg[-1]), float(all_y.max()) + pad_m)
    return x_min, x_max, y_min, y_max


def _roi_slices(xg, yg, x_min, x_max, y_min, y_max):
    """Return (ix_lo, ix_hi, iy_lo, iy_hi) index slices for the ROI."""
    ix_lo = max(0, int(np.searchsorted(xg, x_min)) - 1)
    ix_hi = min(len(xg), int(np.searchsorted(xg, x_max)) + 2)
    iy_lo = max(0, int(np.searchsorted(yg, y_min)) - 1)
    iy_hi = min(len(yg), int(np.searchsorted(yg, y_max)) + 2)
    return ix_lo, ix_hi, iy_lo, iy_hi


def _add_scale_bar(ax, x0_data: float, y0_data: float,
                   bar_len: float = SCALE_BAR_MM,
                   color: str = "white") -> None:
    """Add a horizontal scale bar at (x0_data, y0_data) in data coords [mm]."""
    bar_h = 0.025  # thin rectangle height in mm
    ax.add_patch(Rectangle(
        (x0_data, y0_data), bar_len, bar_h,
        facecolor=color, edgecolor="none", zorder=30, transform=ax.transData,
    ))
    ax.text(x0_data + bar_len / 2.0, y0_data + bar_h + 0.04,
            f"{bar_len:.1f} mm",
            ha="center", va="bottom", color=color, fontsize=9,
            fontweight="bold", zorder=31)


def _add_particles(ax, pos_A_mm, pos_B_mm, r_mm=PARTICLE_RADIUS_MM) -> None:
    """Draw A (orange-red) and B (blue) as filled circles."""
    ax.add_patch(mpatches.Circle(
        (pos_A_mm[0], pos_A_mm[1]), r_mm,
        facecolor=COL_A, edgecolor="white", linewidth=0.8, zorder=15))
    ax.add_patch(mpatches.Circle(
        (pos_B_mm[0], pos_B_mm[1]), r_mm,
        facecolor=COL_B, edgecolor="white", linewidth=0.8, zorder=15))


def _psi_label(psi_rad: float) -> str:
    """Format ψ as a fraction of π, e.g. '3π/4'."""
    frac = psi_rad / np.pi
    numer = round(frac * 4)
    denom = 4
    if numer == 0:
        return "0"
    g = np.gcd(abs(numer), denom)
    n, d = numer // g, denom // g
    if d == 1:
        return f"{n}π" if n != 1 else "π"
    return f"{n}π/{d}"


def _pi_yticks(ax, y_min: float = 0.0, y_max: float = 2 * np.pi,
               step: float = np.pi / 4) -> None:
    """Set yticks at step intervals with π-fraction labels."""
    ticks = np.arange(y_min, y_max + step * 0.01, step)
    ax.set_yticks(ticks)
    ax.set_yticklabels([_psi_label(t) for t in ticks])


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Animated GIF: Gor'kov field with MPC controller
# ══════════════════════════════════════════════════════════════════════════════

def _render_gif_frame(
    ax, psi, alpha, pos_A_m, pos_B_m, vc_m,
    p_sw, vortex_gen, xg, yg, dx, dy,
    roi, x_off, y_off, r_barrier_m,
    t_ms: float, b_status: str,
) -> None:
    x_min, x_max, y_min, y_max = roi
    ix_lo, ix_hi, iy_lo, iy_hi = _roi_slices(xg, yg, x_min, x_max, y_min, y_max)

    p_tot = total_pressure(p_sw, vortex_gen, psi, alpha, BETA, vc_m)
    U, Fx, Fy = gorkov(p_tot, dx, dy)

    U_roi    = U[iy_lo:iy_hi, ix_lo:ix_hi]
    Fx_roi   = Fx[iy_lo:iy_hi, ix_lo:ix_hi]
    Fy_roi   = Fy[iy_lo:iy_hi, ix_lo:ix_hi]
    xg_roi   = xg[ix_lo:ix_hi] * 1e3 - x_off
    yg_roi   = yg[iy_lo:iy_hi] * 1e3 - y_off
    extent   = [xg_roi[0], xg_roi[-1], yg_roi[0], yg_roi[-1]]

    vlo = float(np.percentile(U_roi, 1.0))
    vhi = float(np.percentile(U_roi, 99.0))
    ax.imshow(U_roi, origin="lower", extent=extent, cmap=CMAP,
              vmin=vlo, vmax=vhi, aspect="equal", interpolation="bicubic",
              zorder=1)

    # Force arrows — same style as fig4 (white, direction-only)
    _qs = 7
    xs_q  = xg_roi[::_qs]
    ys_q  = yg_roi[::_qs]
    Fxs   = Fx_roi[::_qs, ::_qs]
    Fys   = Fy_roi[::_qs, ::_qs]
    mag   = np.sqrt(Fxs**2 + Fys**2)
    safe  = np.where(mag > 0, mag, 1.0)
    ax.quiver(xs_q, ys_q, Fxs / safe, Fys / safe,
              color="white", alpha=0.60, scale=24, width=0.0025,
              headwidth=3.5, headlength=4.5, zorder=10)

    # Vortex barrier ring (dashed)
    vc_off = np.array(vc_m) * 1e3 - np.array([x_off, y_off])
    ax.add_patch(mpatches.Circle(
        vc_off, R_BARRIER_M * 1e3,
        fill=False, edgecolor=COL_BARRIER, linestyle="--",
        linewidth=1.2, alpha=0.85, zorder=8))

    # Particles
    A_off = pos_A_m * 1e3 - np.array([x_off, y_off])
    B_off = pos_B_m * 1e3 - np.array([x_off, y_off])
    _add_particles(ax, A_off, B_off)

    # Scale bar (bottom-left corner, 5% from edges)
    roi_w = (x_max - x_min) * 1e3
    roi_h = (y_max - y_min) * 1e3
    _add_scale_bar(ax, roi_w * 0.05, roi_h * 0.04)

    # Annotation: ψ and t
    psi_str = _psi_label(psi)
    bstat_str = f"  B: {b_status}" if b_status else ""
    ax.text(0.03, 0.97, f"ψ = {psi:.2f} rad ({psi_str})\nt = {t_ms:.1f} ms{bstat_str}",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=10, color="white",
            bbox=dict(facecolor="black", alpha=0.45, pad=3, edgecolor="none"),
            zorder=20)

    # Legend stubs
    ax.scatter([], [], c=COL_A, s=40, label="A (vortex)")
    ax.scatter([], [], c=COL_B, s=40, label="B (SW trap)")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.6,
              handletextpad=0.3, borderpad=0.4)

    ax.set_xlim(0, roi_w)
    ax.set_ylim(0, roi_h)
    ax.set_xlabel("x [mm]", fontsize=12, fontweight="bold")
    ax.set_ylabel("y [mm]", fontsize=12, fontweight="bold")
    ax.set_aspect("equal", adjustable="box")


def make_fig1_gif(df, p_sw, vortex_gen, xg, yg, dx, dy,
                  roi, x_off, y_off) -> None:
    print("\n[Figure 1] Generating GIF …")
    fig_w_in = _inches(FIG_WIDTH_MM)

    # Only LATE and HOLD phases — skip INTRO/EARLY approach; focus on capture event
    late_df = df[df["phase_label"].isin(["LATE", "HOLD"])]
    step_rows: List[pd.Series] = []

    for v_step, grp in late_df.groupby("v_step", sort=True):
        label = grp.iloc[0]["phase_label"]
        if label == "HOLD":
            idxs = np.linspace(0, len(grp) - 1, 4, dtype=int)
        else:  # LATE
            idxs = np.linspace(0, len(grp) - 1, 9, dtype=int)
        for i in idxs:
            step_rows.append(grp.iloc[i])

    all_rows = step_rows
    n_total = len(all_rows)
    print(f"  {n_total} GIF frames to render …")

    roi_w_mm = (roi[1] - roi[0]) * 1e3
    roi_h_mm = (roi[3] - roi[2]) * 1e3
    aspect   = roi_h_mm / roi_w_mm
    fig_h_in = fig_w_in * aspect

    frames: List[Image.Image] = []
    for fi, row in enumerate(all_rows):
        psi     = float(row["psi"])
        alpha_f = float(row["alpha"])
        pos_A   = np.array([row["A_x_m"], row["A_y_m"]])
        pos_B   = np.array([row["B_x_m"], row["B_y_m"]])
        vc      = np.array([row["vc_x_m"], row["vc_y_m"]])
        t_step  = int(row["t_step"])
        b_stat  = str(row["B_status"])

        # For INTRO frames use alpha=0 → pure SW
        if row["phase_label"] == "INTRO":
            alpha_f = 0.0

        t_ms = t_step * DT_S * 1e3

        fig, ax = plt.subplots(1, 1, figsize=(fig_w_in, fig_h_in))
        _render_gif_frame(
            ax, psi, alpha_f, pos_A, pos_B, vc,
            p_sw, vortex_gen, xg, yg, dx, dy,
            roi, x_off, y_off, R_BARRIER_M,
            t_ms, b_stat,
        )
        fig.tight_layout(pad=0.5)
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        w_px, h_px = fig.canvas.get_width_height()
        frames.append(Image.fromarray(buf.reshape(h_px, w_px, 4)).convert("RGB"))
        plt.close(fig)
        if (fi + 1) % 10 == 0:
            print(f"  … rendered {fi + 1}/{n_total}")

    out_path = OUT_DIR / "fig1_MPC_gif.gif"
    frames[0].save(
        str(out_path), save_all=True, append_images=frames[1:],
        duration=90, loop=0,
    )
    print(f"  Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — ψ sequence and B approach
# ══════════════════════════════════════════════════════════════════════════════

def make_fig2_psi_sequence(df: pd.DataFrame) -> None:
    print("\n[Figure 2] ψ sequence + d(A,B) …")

    main_df = df[df["phase_label"].isin(["EARLY", "LATE", "HOLD"])].copy()
    t0 = float(main_df["t_step"].iloc[0])
    main_df["t_ms"] = (main_df["t_step"].astype(float) - t0) * DT_S * 1e3
    main_df["d_AB_um"] = main_df["d_AB_m"].astype(float) * 1e6

    # Per-vortex-step ψ: first row of each v_step
    step_rows = main_df.groupby("v_step").first().reset_index()
    step_t_ms = step_rows["t_ms"].values
    step_psi  = step_rows["psi"].values

    # Full timeseries for d(A,B)
    t_all  = main_df["t_ms"].values
    d_all  = main_df["d_AB_um"].values

    # Detect ψ transitions
    psi_transitions = []
    for i in range(1, len(step_psi)):
        if abs(step_psi[i] - step_psi[i - 1]) > 0.05:
            psi_transitions.append(step_t_ms[i])

    # Detect first INSIDE event
    inside_mask = main_df["B_status"] == "INSIDE"
    t_capture = float(main_df.loc[inside_mask, "t_ms"].iloc[0]) if inside_mask.any() else None

    fig_w_in = _inches(FIG_WIDTH_MM)
    fig_h_in = _inches(100.0)  # 100mm tall
    fig, ax1 = plt.subplots(figsize=(fig_w_in, fig_h_in))
    ax2 = ax1.twinx()

    # ── Left axis: ψ step function ────────────────────────────────
    # Build step arrays from per-step values
    # step function: use ax1.step with all main timestamps
    psi_dense = np.interp(t_all, step_t_ms, step_psi, left=step_psi[0], right=step_psi[-1])
    ax1.step(t_all, psi_dense, color="black", lw=1.5, where="post", label="ψ (MPC)")
    ax1.set_ylabel("ψ [rad]", fontsize=12, fontweight="bold", color="black")
    ax1.tick_params(axis="y", colors="black")
    _pi_yticks(ax1, y_min=0.0, y_max=2.0 * np.pi + 0.01, step=np.pi / 4)
    ax1.set_ylim(-0.15, 2 * np.pi + 0.3)

    # ── Right axis: d(A,B) distance ──────────────────────────────
    ax2.plot(t_all, d_all, color="#2980b9", lw=1.4, label="d(A, B)")
    cap_um = R_BARRIER_M * 1e6
    ax2.axhline(cap_um, color="#c0392b", ls="--", lw=1.2,
                label=f"capture radius ({cap_um:.0f} µm)")
    ax2.set_ylabel("d(A, B)  [µm]", fontsize=12, fontweight="bold", color="#2980b9")
    ax2.tick_params(axis="y", colors="#2980b9")
    # Scale to visible data only (t >= t_start_ms, set below)
    t_start_ms = float(main_df[main_df["v_step"] >= 16]["t_ms"].iloc[0])
    d_visible_max = float(main_df[main_df["t_ms"] >= t_start_ms]["d_AB_um"].max())
    ax2.set_ylim(bottom=-10, top=d_visible_max * 1.10)

    # ── Vertical grey lines at ψ transitions ─────────────────────
    for t_trans in psi_transitions:
        ax1.axvline(t_trans, color="#aaaaaa", ls="--", lw=0.7, alpha=0.8, zorder=1)

    # ── Capture annotation ────────────────────────────────────────
    if t_capture is not None:
        ax1.axvline(t_capture, color="#27ae60", ls="-", lw=1.5, zorder=5)
        ax1.text(t_capture + 0.3, 2 * np.pi * 0.95, "capture",
                 color="#27ae60", fontsize=9, fontweight="bold",
                 va="top", ha="left")

    ax1.set_xlabel("t  [ms]", fontsize=12, fontweight="bold")
    # Trim to v_step ≥ 16 — remove the irrelevant long-range approach phase
    ax1.set_xlim(left=t_start_ms)

    # ── Combined legend ───────────────────────────────────────────
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=9,
               framealpha=0.85)

    ax1.grid(axis="x", color="#dddddd", lw=0.5)

    fig.tight_layout(pad=0.6)
    for ext in ("pdf", "png"):
        out = OUT_DIR / f"fig2_psi_sequence.{ext}"
        fig.savefig(str(out), dpi=DPI, bbox_inches="tight")
        print(f"  Saved: {out}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Force Arrow Maps: Static Phase Failure
# ══════════════════════════════════════════════════════════════════════════════

def _quiver_force_panel(ax, xg_roi, yg_roi, Fx_roi, Fy_roi, U_roi,
                        q_step: int = 10, u_alpha: float = 0.25,
                        norm_shared=None):
    """
    Render Gor'kov force arrows (normalised direction, magnitude in colour)
    on axes ax over a faint U background.  Uses ax.quiver for speed.
    Returns (norm_m, cmap_q) for shared colorbar.
    """
    vlo = float(np.percentile(U_roi, 2))
    vhi = float(np.percentile(U_roi, 98))
    extent = [xg_roi[0], xg_roi[-1], yg_roi[0], yg_roi[-1]]
    ax.imshow(U_roi, origin="lower", extent=extent, cmap=CMAP,
              vmin=vlo, vmax=vhi, alpha=u_alpha, aspect="equal",
              interpolation="bicubic", zorder=1)

    # Sub-sample for quiver
    ys = yg_roi[::q_step]
    xs = xg_roi[::q_step]
    Fxs = Fx_roi[::q_step, ::q_step]
    Fys = Fy_roi[::q_step, ::q_step]

    mag = np.sqrt(Fxs**2 + Fys**2)
    mag_safe = np.where(mag > 0, mag, 1.0)
    Fxn = Fxs / mag_safe
    Fyn = Fys / mag_safe

    # Colour by log magnitude
    cmap_q  = plt.get_cmap("YlOrRd")
    mag_log = np.log10(np.where(mag > 0, mag, 1e-40))
    if norm_shared is None:
        mag_log_finite = mag_log[mag > 0]
        vlo_m = float(np.percentile(mag_log_finite, 5)) if len(mag_log_finite) else -30
        vhi_m = float(np.percentile(mag_log_finite, 95)) if len(mag_log_finite) else 0
        norm_m = mcolors.Normalize(vmin=vlo_m, vmax=vhi_m)
    else:
        norm_m = norm_shared

    colors_2d   = cmap_q(norm_m(mag_log))       # (ny_q, nx_q, 4)
    colors_flat = colors_2d.reshape(-1, 4)
    XX, YY = np.meshgrid(xs, ys)
    ax.quiver(
        XX.ravel(), YY.ravel(), Fxn.ravel(), Fyn.ravel(),
        color=colors_flat,
        scale=24, scale_units="width",
        width=0.003, headwidth=3.5, headlength=4.0,
        headaxislength=3.5, alpha=0.90, zorder=10,
    )
    return norm_m, cmap_q


def make_fig3_static_force_arrows(
    df: pd.DataFrame,
    p_sw: np.ndarray,
    vortex_gen,
    xg: np.ndarray, yg: np.ndarray,
    dx: float, dy: float,
    traps_m: np.ndarray,
    late_roi: Tuple[float, float, float, float] = None,
) -> None:
    print("\n[Figure 3] Static force arrows …")

    # Most common ψ in the MPC sequence → treat as 'best static'
    main_df = df[df["phase_label"].isin(["EARLY", "LATE"])]
    step_psi  = main_df.groupby("v_step")["psi"].first()
    psi_best  = float(step_psi.mode().iloc[0])   # e.g. 7π/4
    psi_bad   = 0.0                               # arbitrary bad phase
    print(f"  Best-static ψ = {psi_best:.4f} rad ({_psi_label(psi_best)})")
    print(f"  Bad-static  ψ = {psi_bad:.4f} rad ({_psi_label(psi_bad)})")

    # Vortex centre: last LATE step where B is still OUTSIDE
    # (vortex right at the capture threshold — most illustrative for force fail)
    late_df   = df[df["phase_label"] == "LATE"]
    outside   = late_df[late_df["B_status"] == "OUTSIDE"]
    ref_row   = outside.iloc[-1] if len(outside) else late_df.iloc[-1]
    vc_ref    = np.array([ref_row["vc_x_m"], ref_row["vc_y_m"]])
    B_ref_m   = np.array([ref_row["B_x_m"],  ref_row["B_y_m"]])
    A_ref_m   = np.array([ref_row["A_x_m"],  ref_row["A_y_m"]])
    print(f"  Reference v_step={int(ref_row['v_step'])}: "
          f"vc=({vc_ref[0]*1e3:.3f},{vc_ref[1]*1e3:.3f}) mm  "
          f"d(vc,B)={np.linalg.norm(vc_ref-B_ref_m)*1e6:.0f} µm")

    # ROI: use the shared late_roi if supplied, else build a local tight window
    if late_roi is not None:
        roi3 = late_roi
    else:
        pad_m = R_BARRIER_M
        all_x = np.array([vc_ref[0], B_ref_m[0]])
        all_y = np.array([vc_ref[1], B_ref_m[1]])
        roi3 = (
            max(float(xg[0]),  all_x.min() - pad_m),
            min(float(xg[-1]), all_x.max() + pad_m),
            max(float(yg[0]),  all_y.min() - pad_m),
            min(float(yg[-1]), all_y.max() + pad_m),
        )
    x3_off = roi3[0] * 1e3
    y3_off = roi3[2] * 1e3
    ix_lo, ix_hi, iy_lo, iy_hi = _roi_slices(xg, yg, *roi3)
    xg_roi = xg[ix_lo:ix_hi] * 1e3 - x3_off
    yg_roi = yg[iy_lo:iy_hi] * 1e3 - y3_off
    roi_w  = (roi3[1] - roi3[0]) * 1e3
    roi_h  = (roi3[3] - roi3[2]) * 1e3

    # ── Compute both fields (vortex+SW, two static ψ values) ─────
    p_tot_bad  = total_pressure(p_sw, vortex_gen, psi_bad,  ALPHA, BETA, vc_ref)
    p_tot_best = total_pressure(p_sw, vortex_gen, psi_best, ALPHA, BETA, vc_ref)

    U_bad,  Fx_bad,  Fy_bad  = gorkov(p_tot_bad,  dx, dy)
    U_best, Fx_best, Fy_best = gorkov(p_tot_best, dx, dy)

    def _sl(arr):
        return arr[iy_lo:iy_hi, ix_lo:ix_hi]

    U_bad_r   = _sl(U_bad);   Fx_bad_r  = _sl(Fx_bad);   Fy_bad_r  = _sl(Fy_bad)
    U_best_r  = _sl(U_best);  Fx_best_r = _sl(Fx_best);  Fy_best_r = _sl(Fy_best)

    # Shared magnitude norm across both panels
    for_norm = np.concatenate([
        np.sqrt(Fx_bad_r**2 + Fy_bad_r**2).ravel(),
        np.sqrt(Fx_best_r**2 + Fy_best_r**2).ravel(),
    ])
    for_norm = for_norm[for_norm > 0]
    mag_log_all = np.log10(for_norm)
    shared_norm = mcolors.Normalize(
        vmin=float(np.percentile(mag_log_all, 5)),
        vmax=float(np.percentile(mag_log_all, 95)),
    )

    # ── Layout ────────────────────────────────────────────────────
    fig_w_in = _inches(FIG_WIDTH_MM)
    aspect   = roi_h / roi_w
    fig_h_in = fig_w_in * 0.5 * aspect * 1.15

    fig, axes = plt.subplots(1, 2, figsize=(fig_w_in, fig_h_in),
                             constrained_layout=True)

    vc_off = vc_ref  * 1e3 - np.array([x3_off, y3_off])
    B_off  = B_ref_m * 1e3 - np.array([x3_off, y3_off])
    A_off  = A_ref_m * 1e3 - np.array([x3_off, y3_off])

    panels = [
        (axes[0], U_bad_r,  Fx_bad_r,  Fy_bad_r,  psi_bad,  f"(a) ψ = {_psi_label(psi_bad)}"),
        (axes[1], U_best_r, Fx_best_r, Fy_best_r, psi_best, f"(b) ψ = {_psi_label(psi_best)} (static best)"),
    ]

    for col, (ax, U_r, Fx_r, Fy_r, psi_val, label) in enumerate(panels):
        norm_m, cmap_q = _quiver_force_panel(ax, xg_roi, yg_roi, Fx_r, Fy_r, U_r,
                                             q_step=8, u_alpha=0.28,
                                             norm_shared=shared_norm)

        # B position
        ax.add_patch(mpatches.Circle(
            B_off, PARTICLE_RADIUS_MM,
            facecolor=COL_B, edgecolor="white", linewidth=0.8, zorder=12, alpha=0.9))
        # A position
        ax.add_patch(mpatches.Circle(
            A_off, PARTICLE_RADIUS_MM,
            facecolor=COL_A, edgecolor="white", linewidth=0.8, zorder=12, alpha=0.9))

        # Vortex capture ring
        ax.add_patch(mpatches.Circle(
            vc_off, R_BARRIER_M * 1e3,
            fill=False, edgecolor=COL_BARRIER, linestyle="--",
            linewidth=1.2, alpha=0.85, zorder=8))

        # SW trap crosses in both panels
        traps_mm = traps_m * 1e3 - np.array([x3_off, y3_off])
        in_roi = (
            (traps_mm[:, 0] >= 0) & (traps_mm[:, 0] <= roi_w)
            & (traps_mm[:, 1] >= 0) & (traps_mm[:, 1] <= roi_h)
        )
        ax.scatter(traps_mm[in_roi, 0], traps_mm[in_roi, 1],
                   marker="+", c="grey", s=22, linewidths=0.7,
                   zorder=5, alpha=0.55)

        # Repulsion annotation on panel (a): arrow from B toward barrier
        if col == 0:
            B_to_vc = vc_off - B_off
            B_to_vc_n = B_to_vc / (np.linalg.norm(B_to_vc) + 1e-12)
            ann_xy      = B_off + 0.25 * B_to_vc_n
            ann_text_xy = B_off - 0.35 * B_to_vc_n
            ax.annotate("repulsion",
                xy=tuple(ann_xy),
                xytext=tuple(ann_text_xy),
                fontsize=9, color="#c0392b", fontweight="bold",
                ha="center",
                arrowprops=dict(arrowstyle="-|>, head_width=0.06",
                                color="#c0392b",
                                connectionstyle="arc3,rad=0.35",
                                lw=1.2),
                zorder=20)

        _add_scale_bar(ax, roi_w * 0.05, roi_h * 0.04)
        ax.set_xlim(0, roi_w)
        ax.set_ylim(0, roi_h)
        ax.set_xlabel("x [mm]", fontsize=12, fontweight="bold")
        if col == 0:
            ax.set_ylabel("y [mm]", fontsize=12, fontweight="bold")
        else:
            ax.tick_params(labelleft=False)
        ax.text(0.03, 0.97, label, transform=ax.transAxes,
                ha="left", va="top", fontsize=9.5, fontweight="bold",
                color="white",
                bbox=dict(facecolor="black", alpha=0.45, pad=2, edgecolor="none"),
                zorder=25)
        ax.set_aspect("equal", adjustable="box")

    # Shared magnitude colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap_q, norm=shared_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), orientation="vertical",
                        shrink=0.85, pad=0.02)
    cbar.set_label("|−∇U|  [log₁₀ scale]", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    for ext in ("pdf", "png"):
        out = OUT_DIR / f"fig3_static_force_arrows.{ext}"
        fig.savefig(str(out), dpi=DPI, bbox_inches="tight")
        print(f"  Saved: {out}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Gor'kov Potential at Key MPC Steps
# ══════════════════════════════════════════════════════════════════════════════

def make_fig4_gorkov_sequence(
    df: pd.DataFrame,
    p_sw: np.ndarray,
    vortex_gen,
    xg: np.ndarray, yg: np.ndarray,
    dx: float, dy: float,
    roi, x_off, y_off,
) -> None:
    print("\n[Figure 4] Gor'kov potential at key ψ transitions …")

    # B-approach sequence: step 16 (vortex first influences B),
    # step 22 (LATE phase start, vortex significantly closer),
    # step 29 (B crosses into vortex — first INSIDE)
    key_vsteps_sorted = [16, 22, 29]
    print(f"  Key vortex steps (B-approach sequence): {key_vsteps_sorted}")

    all_df = df[df["phase_label"].isin(["EARLY", "LATE", "HOLD"])]
    step_grp = all_df.groupby("v_step").first().reset_index()

    ix_lo, ix_hi, iy_lo, iy_hi = _roi_slices(xg, yg, *roi)
    xg_roi = xg[ix_lo:ix_hi] * 1e3 - x_off
    yg_roi = yg[iy_lo:iy_hi] * 1e3 - y_off
    roi_w  = (roi[1] - roi[0]) * 1e3
    roi_h  = (roi[3] - roi[2]) * 1e3

    fig_w_in = _inches(FIG_WIDTH_MM)
    aspect   = roi_h / roi_w
    fig_h_in = fig_w_in * aspect / 3.0 * 1.2  # 1×3 + colorbar

    fig, axes = plt.subplots(1, 3, figsize=(fig_w_in, fig_h_in),
                             constrained_layout=True)

    # Shared colour limits across all panels — compute once
    U_all_panels = []
    panel_data   = []
    for seq_n, v_step in enumerate(key_vsteps_sorted):
        row = step_grp[step_grp["v_step"] == v_step].iloc[0]
        psi = float(row["psi"])
        vc  = np.array([row["vc_x_m"], row["vc_y_m"]])
        pos_A = np.array([row["A_x_m"], row["A_y_m"]])
        pos_B = np.array([row["B_x_m"], row["B_y_m"]])
        p_tot = total_pressure(p_sw, vortex_gen, psi, ALPHA, BETA, vc)
        U, Fx, Fy = gorkov(p_tot, dx, dy)
        U_r  = U[iy_lo:iy_hi, ix_lo:ix_hi]
        Fx_r = Fx[iy_lo:iy_hi, ix_lo:ix_hi]
        Fy_r = Fy[iy_lo:iy_hi, ix_lo:ix_hi]
        panel_data.append((psi, vc, pos_A, pos_B, U_r, Fx_r, Fy_r, seq_n + 1, v_step))
        U_all_panels.append(U_r)

    U_stack = np.concatenate([u.ravel() for u in U_all_panels])
    vlo = float(np.percentile(U_stack, 2))
    vhi = float(np.percentile(U_stack, 98))

    # Fixed contour levels
    n_levels  = 8
    cr_levels = np.linspace(vlo, vhi, n_levels)
    # Fine contour-fill levels
    cf_levels = np.linspace(vlo, vhi, 64)

    for ax, (psi, vc, pos_A, pos_B, U_r, Fx_r, Fy_r, seq_n, v_step) in zip(axes, panel_data):
        extent = [xg_roi[0], xg_roi[-1], yg_roi[0], yg_roi[-1]]
        cf = ax.contourf(xg_roi, yg_roi, U_r, levels=cf_levels,
                         cmap=CMAP, extend="both", zorder=1)
        ax.contour(xg_roi, yg_roi, U_r, levels=cr_levels,
                   colors="white", linewidths=0.5, alpha=0.5, zorder=2)

        # Dense quiver on tight ROI — smaller step for more detail
        qs = 7
        ys = yg_roi[::qs]
        xs = xg_roi[::qs]
        Fxs = Fx_r[::qs, ::qs]
        Fys = Fy_r[::qs, ::qs]
        mag = np.sqrt(Fxs**2 + Fys**2)
        ms  = np.where(mag > 0, mag, 1.0)
        ax.quiver(xs, ys, Fxs / ms, Fys / ms,
                  color="white", alpha=0.60, scale=24, width=0.0025,
                  headwidth=3.5, headlength=4.5, zorder=5)

        # Particles
        A_off = pos_A * 1e3 - np.array([x_off, y_off])
        B_off = pos_B * 1e3 - np.array([x_off, y_off])
        _add_particles(ax, A_off, B_off)

        _add_scale_bar(ax, roi_w * 0.05, roi_h * 0.04)

        # Caption label
        psi_str = _psi_label(psi)
        ax.text(0.03, 0.97,
                f"ψ = {psi_str}\nstep {v_step}",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=9, color="white",
                bbox=dict(facecolor="black", alpha=0.45, pad=2, edgecolor="none"),
                zorder=25)

        ax.set_xlim(0, roi_w)
        ax.set_ylim(0, roi_h)
        ax.set_xlabel("x [mm]", fontsize=12, fontweight="bold")
        ax.set_aspect("equal", adjustable="box")

    axes[0].set_ylabel("y [mm]", fontsize=12, fontweight="bold")
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)

    # Shared colorbar
    sm = plt.cm.ScalarMappable(
        cmap=CMAP,
        norm=mcolors.Normalize(vmin=vlo, vmax=vhi),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), orientation="vertical",
                        shrink=0.9, pad=0.02)
    cbar.set_label("U  [J]", fontsize=11, fontweight="bold")
    cbar.ax.tick_params(labelsize=9)
    cbar.formatter = mticker.ScalarFormatter(useMathText=True)
    cbar.formatter.set_powerlimits((-20, -20))
    cbar.update_ticks()

    for ext in ("pdf", "png"):
        out = OUT_DIR / f"fig4_gorkov_sequence.{ext}"
        fig.savefig(str(out), dpi=DPI, bbox_inches="tight")
        print(f"  Saved: {out}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    t0 = __import__("time").time()

    # ── Load physics ─────────────────────────────────────────────
    print("Loading physics data …")
    (p_sw, xg, yg, dx, dy, traps_m, vortex_gen, r_barrier
     ) = load_data(APERTURE_M, PROP_DIST_M, FOCUS_M, field_npz=FIELD_NPZ)
    print(f"  FEM grid: {len(xg)}×{len(yg)}")
    print(f"  r_barrier = {r_barrier*1e6:.1f} µm  (ref: {R_BARRIER_M*1e6:.1f} µm)")

    # ── Load timeseries ──────────────────────────────────────────
    print(f"\nLoading timeseries: {TIMESERIES_CSV}")
    df = pd.read_csv(TIMESERIES_CSV)
    print(f"  {len(df)} rows, {df['v_step'].nunique()} unique v_steps")

    # ── Late interaction ROI: step 16 → capture (for Figs 3 & 4) ─
    late_roi   = _compute_late_roi(df, r_barrier, xg, yg, v_step_start=16)
    late_x_off = late_roi[0] * 1e3
    late_y_off = late_roi[2] * 1e3
    print(f"\nLate ROI  (v≥16): x=[{late_roi[0]*1e3:.3f}, {late_roi[1]*1e3:.3f}]  "
          f"y=[{late_roi[2]*1e3:.3f}, {late_roi[3]*1e3:.3f}] mm  "
          f"({(late_roi[1]-late_roi[0])*1e3:.2f}×{(late_roi[3]-late_roi[2])*1e3:.2f} mm)")

    # ── GIF ROI: 2× the base v≥22 region (doubled linear dimensions) ──
    _gif_base  = _compute_late_roi(df, r_barrier, xg, yg, v_step_start=22)
    _gif_cx    = (_gif_base[0] + _gif_base[1]) / 2
    _gif_cy    = (_gif_base[2] + _gif_base[3]) / 2
    _gif_hw    = (_gif_base[1] - _gif_base[0])      # half-width at 2× = full base width
    _gif_hh    = (_gif_base[3] - _gif_base[2])      # half-height at 2× = full base height
    gif_roi    = (
        max(float(xg[0]),  _gif_cx - _gif_hw),
        min(float(xg[-1]), _gif_cx + _gif_hw),
        max(float(yg[0]),  _gif_cy - _gif_hh),
        min(float(yg[-1]), _gif_cy + _gif_hh),
    )
    gif_x_off = gif_roi[0] * 1e3
    gif_y_off = gif_roi[2] * 1e3
    print(f"GIF  ROI  (2× v≥22): x=[{gif_roi[0]*1e3:.3f}, {gif_roi[1]*1e3:.3f}]  "
          f"y=[{gif_roi[2]*1e3:.3f}, {gif_roi[3]*1e3:.3f}] mm  "
          f"({(gif_roi[1]-gif_roi[0])*1e3:.2f}×{(gif_roi[3]-gif_roi[2])*1e3:.2f} mm)")

    # ── Generate figures ─────────────────────────────────────────
    make_fig2_psi_sequence(df)
    make_fig3_static_force_arrows(df, p_sw, vortex_gen, xg, yg, dx, dy, traps_m, late_roi)
    make_fig4_gorkov_sequence(df, p_sw, vortex_gen, xg, yg, dx, dy, late_roi, late_x_off, late_y_off)
    make_fig1_gif(df, p_sw, vortex_gen, xg, yg, dx, dy, gif_roi, gif_x_off, gif_y_off)

    elapsed = __import__("time").time() - t0
    print(f"\nAll figures saved to: {OUT_DIR}")
    print(f"Total elapsed: {elapsed:.1f} s")


if __name__ == "__main__":
    main()
