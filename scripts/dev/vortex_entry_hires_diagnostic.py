#!/usr/bin/env python3
"""
Vortex Entry High-Resolution Diagnostic — dense-frame replay of the
successful vortex entry / merge case for detailed analysis.

Same physics, field construction, scoring, and dynamics as
vortex_entry_test.py.  Adds:
  - Non-uniform vortex path: denser waypoints near the merge target
  - Adaptive frame capture: more GIF frames during late / merge stages
  - Optionally slower vortex near contact (more dyn steps per position)
  - Post-merge hold: continues dynamics after merge to observe stability
  - Per-sub-step time-series CSV
  - Proximity-vs-time plot (d_AB with phase / merge annotations)

Outputs:
    results/dev/vortex_entry_hires/<timestamp>/
        vortex_entry_hires_<label>.gif
        timeseries.csv
        proximity_vs_time.png
        manifest.json

Usage:
    python scripts/dev/vortex_entry_hires_diagnostic.py
    python scripts/dev/vortex_entry_hires_diagnostic.py --n_psi 36 --n_hold 200
    python scripts/dev/vortex_entry_hires_diagnostic.py --slow_late_factor 2.0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["figure.dpi"] = 72
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# ── Project path ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.lib.fem_cache_utils import (
    C_WATER, F_HZ, OMEGA, RHO0,
    default_particle_params, gorkov_grid_2d,
)
from scripts.lib.particle_dynamics_utils import LAM

from acoustweezers.experiments.vortex_entry.fields.vortex_source import load_data
from acoustweezers.experiments.vortex_entry.fields.field_superposition import total_pressure
from acoustweezers.experiments.vortex_entry.control.vortex_entry import (
    select_trap_pair,
    validate_geometry,
    phase_sweep,
    GEOM_MARGIN,
    OPENING_THRESHOLD,
)
from acoustweezers.experiments.vortex_entry.particles.dynamics import (
    update_particles,
    b_status as _b_status,
)
from acoustweezers.experiments.vortex_entry.diagnostics.scheduling import (
    dense_late_path,
    adaptive_snapshot_schedule,
)
from acoustweezers.experiments.vortex_entry.diagnostics.timeseries import (
    TimeSeriesRecorder,
)

# ── Data path ─────────────────────────────────────────────────────
FIELD_NPZ = (
    PROJECT_ROOT / "results" / "deliverables" / "vortex_stage_transport"
    / "transport" / "transport_case_for_gif.npz"
)

# ── Defaults (same successful case as vortex_entry_test.py) ───────
BESSEL_APERTURE_MM_DEFAULT  = 3.5
BESSEL_PROP_DIST_MM_DEFAULT = 3.0
FOCUS_MM_DEFAULT            = 3.0
N_PSI_DEFAULT        = 24
ALPHA_DEFAULT        = 2.0
BETA_FIXED           = 1.0
N_VORTEX_STEPS       = 20
N_DYN_STEPS_PER_POS  = 150
W_BARRIER_DEFAULT    = 1.0
W_PULL_DEFAULT       = 1.0
W_LATERAL_DEFAULT    = 0.5
W_RETAIN_DEFAULT     = 1.0
MAX_STEP_DEFAULT     = 2e-6

# ── Diagnostic defaults ──────────────────────────────────────────
LATE_FRAC_DEFAULT        = 0.3
DENSITY_BOOST_DEFAULT    = 3
BASE_FPS_DEFAULT         = 4
LATE_FPS_DEFAULT         = 12
SLOW_LATE_FACTOR_DEFAULT = 1.5
N_HOLD_DEFAULT           = 200
HOLD_FPS_DEFAULT         = 8
N_INTRO_FRAMES           = 15

# ── Rendering ─────────────────────────────────────────────────────
CMAP               = "RdBu_r"
COL_A              = "#e74c3c"
COL_B              = "#3498db"
COL_CTR            = "#f39c12"
COL_BARRIER        = "#e67e22"
COL_TRAP           = "#95a5a6"
COL_HOME_A         = "#e74c3c"
COL_HOME_B         = "#3498db"
GIF_DURATION_MS    = 80
PARTICLE_RADIUS_MM = 0.045
VIEW_MARGIN_MM     = 0.5

PPAR = default_particle_params()


# ══════════════════════════════════════════════════════════════════
# View + rendering (same visual style as vortex_entry_test.py)
# ══════════════════════════════════════════════════════════════════

def _compute_view(
    traps_m: np.ndarray,
    idx_A: int,
    idx_B: int,
    vortex_path: np.ndarray,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    key_pts = np.vstack([
        traps_m[idx_A][None, :],
        traps_m[idx_B][None, :],
        vortex_path,
    ]) * 1e3

    mid = 0.5 * (traps_m[idx_A] + traps_m[idx_B])
    d_to_mid = np.linalg.norm(traps_m - mid, axis=1)
    nearby_mask = d_to_mid < 1.5e-3
    if np.any(nearby_mask):
        key_pts = np.vstack([key_pts, traps_m[nearby_mask] * 1e3])

    view_xlim = (float(key_pts[:, 0].min()) - VIEW_MARGIN_MM,
                 float(key_pts[:, 0].max()) + VIEW_MARGIN_MM)
    view_ylim = (float(key_pts[:, 1].min()) - VIEW_MARGIN_MM,
                 float(key_pts[:, 1].max()) + VIEW_MARGIN_MM)
    return view_xlim, view_ylim


def _render_frame(
    ax, fd, xg, yg, dx, dy,
    p_sw, vortex_gen, traps_m, idx_A, idx_B,
    r_barrier_mm, extent, view_xlim, view_ylim,
    frame_idx, n_frames_total,
):
    pos_A_mm = fd["pos_A"] * 1e3
    pos_B_mm = fd["pos_B"] * 1e3
    vc_mm    = fd["vortex_center"] * 1e3
    psi      = fd["psi"]
    alpha    = fd["alpha"]
    d_AB_um  = fd["d_AB"] * 1e6
    trail_mm = np.array(fd["vortex_trail"]) * 1e3
    label    = fd.get("label", "SWEEP")
    b_stat   = fd.get("B_status", "")

    p_tot = total_pressure(p_sw, vortex_gen, psi, alpha, BETA_FIXED,
                           fd["vortex_center"])
    U_disp, _, _ = gorkov_grid_2d(
        p_tot, dx, dy, OMEGA, RHO0, C_WATER,
        PPAR["a"], PPAR["f1"], PPAR["f2"],
    )

    ix_lo = max(0, int(np.searchsorted(xg, (view_xlim[0] - 0.1) * 1e-3)))
    ix_hi = min(len(xg),
                int(np.searchsorted(xg, (view_xlim[1] + 0.1) * 1e-3)))
    iy_lo = max(0, int(np.searchsorted(yg, (view_ylim[0] - 0.1) * 1e-3)))
    iy_hi = min(len(yg),
                int(np.searchsorted(yg, (view_ylim[1] + 0.1) * 1e-3)))
    U_roi = U_disp[iy_lo:iy_hi, ix_lo:ix_hi]
    lo = float(np.percentile(U_roi, 0.5))
    hi = float(np.percentile(U_roi, 99.5))

    ax.imshow(
        U_disp, origin="lower", extent=extent, cmap=CMAP,
        vmin=lo, vmax=hi, aspect="equal", interpolation="bicubic",
    )

    traps_mm = traps_m * 1e3
    in_view = (
        (traps_mm[:, 0] >= view_xlim[0]) & (traps_mm[:, 0] <= view_xlim[1])
        & (traps_mm[:, 1] >= view_ylim[0]) & (traps_mm[:, 1] <= view_ylim[1])
    )
    if np.any(in_view):
        ax.scatter(traps_mm[in_view, 0], traps_mm[in_view, 1],
                   marker="x", c=COL_TRAP, s=28, linewidths=0.8,
                   zorder=4, alpha=0.5)

    ax.annotate("A-home", traps_mm[idx_A], fontsize=6, color=COL_HOME_A,
                alpha=0.8, ha="left", va="bottom",
                xytext=(3, 3), textcoords="offset points")
    ax.annotate("B-home", traps_mm[idx_B], fontsize=6, color=COL_HOME_B,
                alpha=0.8, ha="left", va="bottom",
                xytext=(3, 3), textcoords="offset points")

    if len(trail_mm) > 1:
        ax.plot(trail_mm[:, 0], trail_mm[:, 1], color=COL_CTR,
                lw=0.9, alpha=0.75, zorder=5)
    ax.plot(vc_mm[0], vc_mm[1], marker="+", color=COL_CTR,
            ms=12, mew=2.0, zorder=8)

    ax.add_patch(mpatches.Circle(
        (vc_mm[0], vc_mm[1]), r_barrier_mm,
        fill=False, edgecolor=COL_BARRIER, linestyle="--",
        linewidth=1.0, alpha=0.7, zorder=6))

    ax.add_patch(mpatches.Circle(
        (pos_A_mm[0], pos_A_mm[1]), PARTICLE_RADIUS_MM,
        facecolor=COL_A, edgecolor="white", linewidth=0.6,
        zorder=7, alpha=0.9))
    ax.add_patch(mpatches.Circle(
        (pos_B_mm[0], pos_B_mm[1]), PARTICLE_RADIUS_MM,
        facecolor=COL_B, edgecolor="white", linewidth=0.6,
        zorder=7, alpha=0.9))
    ax.scatter([], [], c=COL_A, s=30, label="A (vortex)")
    ax.scatter([], [], c=COL_B, s=30, label="B (SW trap)")

    ax.set_xlim(*view_xlim)
    ax.set_ylim(*view_ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    status_str = f" | B: {b_stat}" if b_stat else ""
    ax.set_title(
        f"[{label}] psi={psi:.2f} | alpha={alpha:.1f} | "
        f"d(A,B)={d_AB_um:.0f} um{status_str}\n"
        f"frame {frame_idx + 1}/{n_frames_total}",
        fontsize=9)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.7)


def render_gif(
    frames_data, xg, yg, dx, dy,
    p_sw, vortex_gen, traps_m,
    idx_A, idx_B, r_barrier_mm,
    view_xlim, view_ylim,
    out_path, duration_ms=GIF_DURATION_MS,
):
    x_mm = xg * 1e3
    y_mm = yg * 1e3
    extent = [x_mm[0], x_mm[-1], y_mm[0], y_mm[-1]]

    n_total = len(frames_data)
    frames: List[Image.Image] = []
    print(f"  Rendering {n_total} GIF frames...")

    for fi, fdat in enumerate(frames_data):
        fig, ax = plt.subplots(1, 1, figsize=(6.0, 6.0))
        _render_frame(
            ax, fdat, xg, yg, dx, dy, p_sw, vortex_gen,
            traps_m, idx_A, idx_B, r_barrier_mm,
            extent, view_xlim, view_ylim, fi, n_total,
        )
        fig.tight_layout()
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        frames.append(Image.fromarray(buf.reshape(h, w, 4)).convert("RGB"))
        plt.close(fig)

    frames[0].save(
        str(out_path), save_all=True, append_images=frames[1:],
        duration=duration_ms, loop=0)
    print(f"  Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════
# Proximity-vs-time plot
# ══════════════════════════════════════════════════════════════════

def plot_proximity(
    recorder: TimeSeriesRecorder,
    r_barrier: float,
    out_path: Path,
) -> None:
    t_steps = recorder.get_column("t_step").astype(float)
    d_AB    = recorder.get_column("d_AB_m").astype(float) * 1e6  # um
    d_B_vc  = recorder.get_column("d_B_vc_m").astype(float) * 1e6
    d_A_vc  = recorder.get_column("d_A_vc_m").astype(float) * 1e6
    labels  = recorder.get_column("phase_label")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # ── d(A,B) vs time ──────────────────────────────────────────
    ax1.plot(t_steps, d_AB, color="#2c3e50", lw=1.2, label="d(A,B)")
    ax1.axhline(r_barrier * 1e6, color=COL_BARRIER, ls="--", lw=0.8,
                label=f"r_barrier = {r_barrier*1e6:.0f} um")
    ax1.set_ylabel("d(A, B)  [um]")
    ax1.set_title("Particle separation vs time")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ── d(A, vc) and d(B, vc) vs time ───────────────────────────
    ax2.plot(t_steps, d_A_vc, color=COL_A, lw=1.0, label="d(A, vortex)")
    ax2.plot(t_steps, d_B_vc, color=COL_B, lw=1.0, label="d(B, vortex)")
    ax2.axhline(r_barrier * 1e6, color=COL_BARRIER, ls="--", lw=0.8,
                label=f"r_barrier")
    ax2.set_ylabel("distance to vortex  [um]")
    ax2.set_xlabel("time step")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ── Phase-transition lines ───────────────────────────────────
    phase_changes = []
    prev_label = None
    for i, lab in enumerate(labels):
        if lab != prev_label and prev_label is not None:
            phase_changes.append((float(t_steps[i]), str(lab)))
        prev_label = lab

    for t_change, lab in phase_changes:
        for ax in (ax1, ax2):
            ax.axvline(t_change, color="#7f8c8d", ls=":", lw=0.7, alpha=0.8)
        ax1.annotate(lab, (t_change, ax1.get_ylim()[1]),
                     fontsize=7, rotation=90, va="top", ha="right",
                     color="#7f8c8d", alpha=0.9)

    # ── Merge event (first time d_B_vc < 0.95 * r_barrier) ──────
    merge_mask = d_B_vc < r_barrier * 1e6 * 0.95
    if np.any(merge_mask):
        t_merge = float(t_steps[np.argmax(merge_mask)])
        for ax in (ax1, ax2):
            ax.axvline(t_merge, color="#27ae60", ls="-", lw=1.2, alpha=0.8)
        ax1.annotate("MERGE", (t_merge, ax1.get_ylim()[1] * 0.5),
                     fontsize=8, color="#27ae60", fontweight="bold",
                     rotation=90, va="center", ha="right")

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════
# Main experiment
# ══════════════════════════════════════════════════════════════════

def run_experiment(args) -> None:
    t0_total = time.time()

    aperture_m  = args.bessel_aperture_mm * 1e-3
    prop_dist_m = args.bessel_prop_dist_mm * 1e-3
    focus_f_m   = args.focus_mm * 1e-3

    # ── Load, calibrate ───────────────────────────────────────────
    (p_sw, xg, yg, dx, dy, traps_m,
     vortex_gen, r_barrier) = load_data(aperture_m, prop_dist_m, focus_f_m,
                                         field_npz=FIELD_NPZ)
    r_barrier_mm = r_barrier * 1e3

    # ── Select A/B pair ───────────────────────────────────────────
    idx_A, idx_B = select_trap_pair(
        traps_m, r_barrier, xg, yg,
        idx_B_override=args.idx_B,
        min_sep_factor=args.min_sep_factor,
    )
    geom_status = validate_geometry(traps_m, idx_A, idx_B, r_barrier)

    vortex_start = traps_m[idx_A].copy()
    vortex_end   = traps_m[idx_B].copy()

    # ── Build dense-late path ─────────────────────────────────────
    psi_values = np.linspace(0, 2 * np.pi, args.n_psi, endpoint=False)
    vortex_path = dense_late_path(
        vortex_start, vortex_end,
        n_steps=args.n_vortex_steps,
        late_frac=args.late_frac,
        density_boost=args.density_boost,
        path_mode=args.path_mode,
        arc_height_mm=args.arc_height_mm,
    )
    n_path = len(vortex_path)
    print(f"  Path: {n_path} waypoints (base {args.n_vortex_steps}, "
          f"late_frac={args.late_frac}, boost={args.density_boost}x)")

    # ── Late-stage boundary ───────────────────────────────────────
    late_start_idx = int(round(n_path * (1 - args.late_frac)))

    # ── Snapshot schedule ─────────────────────────────────────────
    snap_set = adaptive_snapshot_schedule(
        n_path, args.n_dyn_steps,
        base_fps=args.base_fps,
        late_fps=args.late_fps,
        late_frac=args.late_frac,
    )
    print(f"  Snapshot schedule: {len(snap_set)} capture points across "
          f"{n_path} vortex steps x {args.n_dyn_steps} dyn steps")

    # ── Compute view ──────────────────────────────────────────────
    view_xlim, view_ylim = _compute_view(
        traps_m, idx_A, idx_B, vortex_path)
    print(f"  Render view: x=[{view_xlim[0]:.2f}, {view_xlim[1]:.2f}] "
          f"y=[{view_ylim[0]:.2f}, {view_ylim[1]:.2f}] mm")

    # ── Output directory ──────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (PROJECT_ROOT / "results" / "dev"
               / "vortex_entry_hires" / timestamp)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {out_dir}")

    recorder = TimeSeriesRecorder()
    alpha = args.alpha
    pos_A = traps_m[idx_A].copy()
    pos_B = traps_m[idx_B].copy()
    frames_data: List[Dict] = []
    vortex_trail: List[np.ndarray] = []

    # ── Intro frames (alpha=0, standing wave only) ────────────────
    intro_vc = vortex_start.copy()
    b_stat_intro = _b_status(pos_B, intro_vc, r_barrier)
    for i_intro in range(N_INTRO_FRAMES):
        frames_data.append({
            "pos_A": traps_m[idx_A].copy(),
            "pos_B": traps_m[idx_B].copy(),
            "vortex_center": intro_vc.copy(),
            "vortex_trail": [intro_vc.copy()],
            "psi": 0.0, "alpha": 0.0,
            "d_AB": float(np.linalg.norm(
                traps_m[idx_A] - traps_m[idx_B])),
            "label": "INTRO", "B_status": b_stat_intro,
        })
        recorder.record(
            v_step=-1, dyn_i=i_intro, phase_label="INTRO",
            psi=0.0, alpha=0.0,
            pos_A=traps_m[idx_A].copy(),
            pos_B=traps_m[idx_B].copy(),
            vortex_center=intro_vc, r_barrier=r_barrier,
        )

    # ── Main transport loop ───────────────────────────────────────
    print(f"\n  Main loop: {n_path} vortex steps, alpha={alpha:.1f}")
    for v_step in range(n_path):
        vortex_center = vortex_path[v_step].copy()
        vortex_trail.append(vortex_center.copy())
        is_late = v_step >= late_start_idx
        phase_label = "LATE" if is_late else "EARLY"

        # Optionally run more dynamics in the late phase
        n_dyn = (int(round(args.n_dyn_steps * args.slow_late_factor))
                 if is_late else args.n_dyn_steps)

        t_sweep = time.time()
        best = phase_sweep(
            p_sw, vortex_gen, xg, yg, dx, dy,
            psi_values, alpha, BETA_FIXED,
            vortex_center, pos_A, pos_B, r_barrier,
            w_barrier=args.w_barrier,
            w_pull=args.w_pull,
            w_lateral=args.w_lateral,
            w_retain=args.w_retain,
            opening_threshold=args.opening_threshold,
        )
        psi_best = float(best["psi"])
        iFx, iFy = best["iFx"], best["iFy"]
        mech = best["mechanism"]
        dt_sweep = time.time() - t_sweep

        print(f"  [{phase_label}] step {v_step+1:03d}/{n_path} | "
              f"psi={psi_best:.2f} | dU={best['DeltaU']:.3e} | "
              f"F_in_B={best['F_in_B']:.3e} | "
              f"{mech['mechanism'].upper()} | "
              f"n_dyn={n_dyn} | {dt_sweep*1e3:.0f}ms")

        # ── Dynamics sub-steps ────────────────────────────────────
        # Re-build snapshot set for this step if n_dyn differs from
        # the base (late-phase slow-down)
        if is_late:
            step_snaps = set(
                np.linspace(0, n_dyn - 1, args.late_fps, dtype=int).tolist()
            )
        else:
            step_snaps = set(
                np.linspace(0, n_dyn - 1, args.base_fps, dtype=int).tolist()
            )

        for dyn_i in range(n_dyn):
            pos_A, pos_B = update_particles(
                pos_A, pos_B, iFx, iFy,
                args.max_step, xg, yg,
            )
            recorder.record(
                v_step=v_step, dyn_i=dyn_i,
                phase_label=phase_label,
                psi=psi_best, alpha=alpha,
                pos_A=pos_A.copy(), pos_B=pos_B.copy(),
                vortex_center=vortex_center,
                r_barrier=r_barrier,
            )

            if dyn_i in step_snaps:
                b_stat = _b_status(pos_B, vortex_center, r_barrier)
                frames_data.append({
                    "pos_A": pos_A.copy(),
                    "pos_B": pos_B.copy(),
                    "vortex_center": vortex_center.copy(),
                    "vortex_trail": [v.copy() for v in vortex_trail],
                    "psi": psi_best, "alpha": alpha,
                    "d_AB": float(np.linalg.norm(pos_A - pos_B)),
                    "label": phase_label,
                    "B_status": b_stat,
                })

    # ── Post-merge hold ───────────────────────────────────────────
    if args.n_hold > 0:
        print(f"\n  Post-merge hold: {args.n_hold} dyn steps at final "
              f"vortex position")
        hold_vc = vortex_path[-1].copy()
        # Re-sweep at hold position for current forces
        best_hold = phase_sweep(
            p_sw, vortex_gen, xg, yg, dx, dy,
            psi_values, alpha, BETA_FIXED,
            hold_vc, pos_A, pos_B, r_barrier,
            w_barrier=args.w_barrier,
            w_pull=args.w_pull,
            w_lateral=args.w_lateral,
            w_retain=args.w_retain,
            opening_threshold=args.opening_threshold,
        )
        psi_hold = float(best_hold["psi"])
        iFx_hold, iFy_hold = best_hold["iFx"], best_hold["iFy"]

        hold_snaps = set(
            np.linspace(0, args.n_hold - 1, args.hold_fps,
                        dtype=int).tolist()
        )

        for hi in range(args.n_hold):
            pos_A, pos_B = update_particles(
                pos_A, pos_B, iFx_hold, iFy_hold,
                args.max_step, xg, yg,
            )
            recorder.record(
                v_step=n_path, dyn_i=hi,
                phase_label="HOLD",
                psi=psi_hold, alpha=alpha,
                pos_A=pos_A.copy(), pos_B=pos_B.copy(),
                vortex_center=hold_vc,
                r_barrier=r_barrier,
            )

            if hi in hold_snaps:
                b_stat = _b_status(pos_B, hold_vc, r_barrier)
                frames_data.append({
                    "pos_A": pos_A.copy(),
                    "pos_B": pos_B.copy(),
                    "vortex_center": hold_vc.copy(),
                    "vortex_trail": [v.copy() for v in vortex_trail],
                    "psi": psi_hold, "alpha": alpha,
                    "d_AB": float(np.linalg.norm(pos_A - pos_B)),
                    "label": "HOLD",
                    "B_status": b_stat,
                })

        d_final = float(np.linalg.norm(pos_A - pos_B))
        d_A_vc = float(np.linalg.norm(pos_A - hold_vc))
        d_B_vc = float(np.linalg.norm(pos_B - hold_vc))
        print(f"  Hold end: d(A,B)={d_final*1e6:.1f} um | "
              f"d(A,vc)={d_A_vc*1e6:.1f} um | "
              f"d(B,vc)={d_B_vc*1e6:.1f} um | "
              f"A {'TRAPPED' if d_A_vc < r_barrier else 'drifted'} | "
              f"B {_b_status(pos_B, hold_vc, r_barrier)}")

    # ── Outputs ───────────────────────────────────────────────────
    gif_path = out_dir / f"vortex_entry_hires_alpha_{alpha:.1f}.gif"
    render_gif(frames_data, xg, yg, dx, dy, p_sw, vortex_gen,
               traps_m, idx_A, idx_B, r_barrier_mm,
               view_xlim, view_ylim, gif_path)

    csv_path = out_dir / "timeseries.csv"
    recorder.to_csv(csv_path)
    print(f"  Saved: {csv_path} ({len(recorder.rows)} rows)")

    plot_path = out_dir / "proximity_vs_time.png"
    plot_proximity(recorder, r_barrier, plot_path)

    # ── Manifest ──────────────────────────────────────────────────
    elapsed = time.time() - t0_total
    manifest = {
        "timestamp": timestamp,
        "elapsed_s": round(elapsed, 2),
        "script": "vortex_entry_hires_diagnostic.py",
        "alpha": alpha,
        "beta_fixed": BETA_FIXED,
        "n_psi": args.n_psi,
        "n_vortex_steps_base": args.n_vortex_steps,
        "n_vortex_steps_actual": n_path,
        "late_frac": args.late_frac,
        "density_boost": args.density_boost,
        "late_start_idx": late_start_idx,
        "n_dyn_steps_base": args.n_dyn_steps,
        "slow_late_factor": args.slow_late_factor,
        "n_hold": args.n_hold,
        "max_step_m": args.max_step,
        "w_barrier": args.w_barrier,
        "w_pull": args.w_pull,
        "w_lateral": args.w_lateral,
        "w_retain": args.w_retain,
        "path_mode": args.path_mode,
        "arc_height_mm": args.arc_height_mm,
        "base_fps": args.base_fps,
        "late_fps": args.late_fps,
        "hold_fps": args.hold_fps,
        "n_gif_frames": len(frames_data),
        "n_timeseries_rows": len(recorder.rows),
        "aperture_mm": args.bessel_aperture_mm,
        "prop_dist_mm": args.bessel_prop_dist_mm,
        "focus_mm": args.focus_mm,
        "r_barrier_um": round(r_barrier * 1e6, 1),
        "initial_geometry": geom_status,
        "idx_A": idx_A,
        "idx_B": idx_B,
        "n_traps_total": len(traps_m),
    }
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Saved: {manifest_path}")

    print(f"\nDone in {elapsed:.1f}s — {len(frames_data)} GIF frames, "
          f"{len(recorder.rows)} timeseries rows")
    print(f"Results: {out_dir}")


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Vortex entry — high-resolution diagnostic replay")

    g = parser.add_argument_group("vortex beam")
    g.add_argument("--bessel_aperture_mm", type=float,
                   default=BESSEL_APERTURE_MM_DEFAULT)
    g.add_argument("--bessel_prop_dist_mm", type=float,
                   default=BESSEL_PROP_DIST_MM_DEFAULT)
    g.add_argument("--focus_mm", type=float, default=FOCUS_MM_DEFAULT)

    g = parser.add_argument_group("trap selection")
    g.add_argument("--idx_B", type=int, default=None)
    g.add_argument("--min_sep_factor", type=float, default=GEOM_MARGIN)
    g.add_argument("--opening_threshold", type=float,
                   default=OPENING_THRESHOLD)

    g = parser.add_argument_group("sweep & dynamics")
    g.add_argument("--alpha", type=float, default=ALPHA_DEFAULT,
                   help="Vortex amplitude (default %(default)s)")
    g.add_argument("--n_psi", type=int, default=N_PSI_DEFAULT)
    g.add_argument("--n_vortex_steps", type=int, default=N_VORTEX_STEPS,
                   help="Base vortex step count before density boost "
                        "(default %(default)s)")
    g.add_argument("--n_dyn_steps", type=int, default=N_DYN_STEPS_PER_POS)
    g.add_argument("--max_step", type=float, default=MAX_STEP_DEFAULT)

    g = parser.add_argument_group("diagnostic: path density")
    g.add_argument("--late_frac", type=float, default=LATE_FRAC_DEFAULT,
                   help="Fraction of path considered 'late' for denser "
                        "sampling (default %(default)s)")
    g.add_argument("--density_boost", type=int, default=DENSITY_BOOST_DEFAULT,
                   help="Waypoint density multiplier in late region "
                        "(default %(default)s)")

    g = parser.add_argument_group("diagnostic: frame capture")
    g.add_argument("--base_fps", type=int, default=BASE_FPS_DEFAULT,
                   help="GIF frames per vortex step in early phase "
                        "(default %(default)s)")
    g.add_argument("--late_fps", type=int, default=LATE_FPS_DEFAULT,
                   help="GIF frames per vortex step in late phase "
                        "(default %(default)s)")

    g = parser.add_argument_group("diagnostic: slow late-stage")
    g.add_argument("--slow_late_factor", type=float,
                   default=SLOW_LATE_FACTOR_DEFAULT,
                   help="Multiply n_dyn_steps by this in late phase — "
                        "vortex lingers longer near contact "
                        "(default %(default)s)")

    g = parser.add_argument_group("diagnostic: post-merge hold")
    g.add_argument("--n_hold", type=int, default=N_HOLD_DEFAULT,
                   help="Extra dynamics steps after main loop ends "
                        "(default %(default)s)")
    g.add_argument("--hold_fps", type=int, default=HOLD_FPS_DEFAULT,
                   help="GIF frames during hold phase "
                        "(default %(default)s)")

    g = parser.add_argument_group("path shape")
    g.add_argument("--path_mode", type=str, default="straight",
                   choices=["straight", "arc_left", "arc_right"])
    g.add_argument("--arc_height_mm", type=float, default=0.15)

    g = parser.add_argument_group("score weights")
    g.add_argument("--w_barrier", type=float, default=W_BARRIER_DEFAULT)
    g.add_argument("--w_pull", type=float, default=W_PULL_DEFAULT)
    g.add_argument("--w_lateral", type=float, default=W_LATERAL_DEFAULT)
    g.add_argument("--w_retain", type=float, default=W_RETAIN_DEFAULT)

    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
