#!/usr/bin/env python3
"""
Vortex Entry High-Resolution Diagnostic — with inertial particle ensemble
and **true receding-horizon psi MPC**.

Same physics, field construction, and rendering as
vortex_entry_hires_inertial.py.  The control method is a proper K-step
receding-horizon MPC over phase ψ:

  Phase I  — Receding-horizon psi MPC: optimises a K-step sequence of
             ψ values at each replan step.  Position-based cost
             (‖pA_K − B_home‖²) with fully analytic adjoint gradient
             via ForceEvaluator.dF_du.  Vortex follows the prescribed
             dense_late_path.  Warm-started — no random restarts.
             Dynamics use feval.forces() (same model as optimizer).
  Phase II — Optional close-range receding-horizon (activation radius > 0).
  Phase III— Settling: α→0 ramp-down, then pure standing-wave recapture.

Key differences from the old per-waypoint adjoint sweep:
  • True K-step look-ahead (not per-waypoint greedy gradient step).
  • Position-based cost — optimises where A ends up, not instantaneous
    force.  Gradient captures multi-step dynamics.
  • No FD on the full pressure field — all gradients are analytic via
    the trilinear basis.
  • Warm-start shifts horizon forward one step each replan → single
    L-BFGS-B solve per step, 8× fewer function evaluations.
  • Basis cache shared across all replans at the same vortex position
    (rebuilt only when vortex moves, ≈20× for the full path).

Outputs:
    results/dev/vortex_entry_hires_inertial_mpc/<timestamp>/
        vortex_entry_hires_mpc_<label>.gif  — animated Gorkov potential
        timeseries.csv                      — per-step metrics
        proximity_vs_time.png               — proximity graph
        convergence.png                     — MPC horizon cost history
        particles.csv                       — inertial ensemble (if enabled)
        manifest.json                       — config + metrics

Usage:
    python scripts/dev/vortex_entry_hires_inertial_mpc.py
    python scripts/dev/vortex_entry_hires_inertial_mpc.py --K_phase1 12
    python scripts/dev/vortex_entry_hires_inertial_mpc.py --replan_every 5
    python scripts/dev/vortex_entry_hires_inertial_mpc.py --particle_grid_n 10
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
from scripts.lib.particle_dynamics_utils import (
    CAPTURE_RADIUS, DT_DEFAULT, LAM, SCALE, TRAP_SP,
    gorkov_normalised,
)
from scripts.lib.mpc_controller import (
    I_ALPHA, I_BETA, I_PSI, I_XV, I_YV, N_CTRL,
    ForceEvaluator,
)
from acoustweezers.experiments.vortex_entry.utils.interpolation import (
    make_interp, eval_at,
)
from scipy.optimize import minimize as sp_minimize

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
from acoustweezers.experiments.vortex_entry.particles.particle_state import (
    ParticleEnsemble,
    make_grid_ensemble,
)
from acoustweezers.experiments.vortex_entry.particles.particle_integrator import (
    advance_ensemble,
)
from acoustweezers.experiments.vortex_entry.particles.drag_models import (
    TAU_STOKES, ACCEL_SCALE, M_PARTICLE, PARTICLE_DIAMETER,
)
from acoustweezers.experiments.vortex_entry.diagnostics.scheduling import (
    dense_late_path,
    adaptive_snapshot_schedule,
)
from acoustweezers.experiments.vortex_entry.diagnostics.timeseries import (
    TimeSeriesRecorder,
)
from acoustweezers.experiments.vortex_entry.diagnostics.particle_recorder import (
    ParticleRecorder,
)

# ── Data path ─────────────────────────────────────────────────────
FIELD_NPZ = (
    PROJECT_ROOT / "results" / "deliverables" / "vortex_stage_transport"
    / "transport" / "transport_case_for_gif.npz"
)

# ── Defaults (same successful case as vortex_entry_hires_inertial.py) ──
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

# ── Adjoint optimiser defaults ────────────────────────────────────
MPC_N_ITERS_DEFAULT        = 15   # L-BFGS-B maxiter per start
MPC_ACTIVATION_RADIUS_LAM  = 0.0
MPC_T_SETTLE_DEFAULT       = 3000
MPC_N_NEIGH_RADIUS_DEFAULT = 3
MPC_N_VORTEX_STEPS_DEFAULT = 20
MPC_N_DYN_STEPS_DEFAULT    = 150
N_BARRIER_SAMPLES          = 12

# ── Particle ensemble defaults ────────────────────────────────────
PARTICLE_GRID_N_DEFAULT  = 0
PARTICLE_INTEGRATOR_DEFAULT = "rk4"

# ── Rendering ─────────────────────────────────────────────────────
CMAP               = "RdBu_r"
COL_A              = "#e74c3c"
COL_B              = "#3498db"
COL_CTR            = "#f39c12"
COL_BARRIER        = "#e67e22"
COL_TRAP           = "#95a5a6"
COL_HOME_A         = "#e74c3c"
COL_HOME_B         = "#3498db"
COL_NEIGH          = "#95a5a6"
GIF_DURATION_MS    = 80
PARTICLE_RADIUS_MM = 0.045
VIEW_MARGIN_MM     = 0.5

PPAR = default_particle_params()


# ══════════════════════════════════════════════════════════════════
# View + rendering  (identical to vortex_entry_hires_inertial.py)
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
    neigh_pos_m=None,
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

    # Neighbour particles (MPC phase only)
    if neigh_pos_m is not None:
        for ni_pos in neigh_pos_m:
            ax.add_patch(mpatches.Circle(
                (ni_pos[0] * 1e3, ni_pos[1] * 1e3), PARTICLE_RADIUS_MM,
                facecolor=COL_NEIGH, edgecolor="white", linewidth=0.6,
                zorder=7, alpha=0.7))

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
            neigh_pos_m=fdat.get("neigh_pos_m"),
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
# Proximity-vs-time plot  (identical to vortex_entry_hires_inertial.py)
# ══════════════════════════════════════════════════════════════════

def plot_proximity(
    recorder: TimeSeriesRecorder,
    r_barrier: float,
    out_path: Path,
) -> None:
    t_steps = recorder.get_column("t_step").astype(float)
    d_AB    = recorder.get_column("d_AB_m").astype(float) * 1e6
    d_B_vc  = recorder.get_column("d_B_vc_m").astype(float) * 1e6
    d_A_vc  = recorder.get_column("d_A_vc_m").astype(float) * 1e6
    labels  = recorder.get_column("phase_label")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax1.plot(t_steps, d_AB, color="#2c3e50", lw=1.2, label="d(A,B)")
    ax1.axhline(r_barrier * 1e6, color=COL_BARRIER, ls="--", lw=0.8,
                label=f"r_barrier = {r_barrier*1e6:.0f} um")
    ax1.set_ylabel("d(A, B)  [um]")
    ax1.set_title("Particle separation vs time")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.plot(t_steps, d_A_vc, color=COL_A, lw=1.0, label="d(A, vortex)")
    ax2.plot(t_steps, d_B_vc, color=COL_B, lw=1.0, label="d(B, vortex)")
    ax2.axhline(r_barrier * 1e6, color=COL_BARRIER, ls="--", lw=0.8,
                label=f"r_barrier")
    ax2.set_ylabel("distance to vortex  [um]")
    ax2.set_xlabel("time step")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

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
# MPC convergence plot
# ══════════════════════════════════════════════════════════════════

def plot_convergence(
    J_history: List[float],
    inner_J_histories: List[List[float]],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(J_history, ".-")
    ax.set_xlabel("Waypoint")
    ax.set_ylabel("Horizon cost J")
    ax.set_title("MPC per-waypoint cost")
    if J_history and max(J_history) > 0:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    n_show = min(8, len(inner_J_histories))
    step = max(1, len(inner_J_histories) // n_show)
    for i in range(0, len(inner_J_histories), step):
        hist = inner_J_histories[i]
        if len(hist) > 1:
            ax.plot(hist, label=f"wp {i}")
    ax.set_xlabel("Inner iteration")
    ax.set_ylabel("J")
    ax.set_title("Inner-loop convergence (sampled)")
    if inner_J_histories:
        ax.set_yscale("log")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"  Saved convergence plot: {out_path}")


# (Settling is now done inline in run_experiment — no standalone function needed.)


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

    # ── Build dense-late path for Phase I carry ───────────────────
    # Only carry up to MPC activation radius from B
    mpc_activation_dist = args.mpc_activation_radius * LAM
    transport_vec = vortex_end - vortex_start
    transport_dist = float(np.linalg.norm(transport_vec))
    transport_dir = transport_vec / max(transport_dist, 1e-12)
    ol_travel_dist = max(0.0, transport_dist - mpc_activation_dist)
    ol_end = vortex_start + transport_dir * ol_travel_dist

    psi_values = np.linspace(0, 2 * np.pi, args.n_psi, endpoint=False)
    vortex_path = dense_late_path(
        vortex_start, ol_end,
        n_steps=args.n_vortex_steps,
        late_frac=args.late_frac,
        density_boost=args.density_boost,
        path_mode=args.path_mode,
        arc_height_mm=args.arc_height_mm,
    )
    n_path = len(vortex_path)
    print(f"  Phase I path: {n_path} waypoints, carry {ol_travel_dist*1e3:.2f} mm")
    print(f"  MPC activates at {mpc_activation_dist*1e3:.2f} mm from B")

    late_start_idx = int(round(n_path * (1 - args.late_frac)))

    # ── Build Phase II MPC path (carry end → B) ──────────────────
    vortex_handoff = vortex_path[-1].copy() if n_path > 0 else vortex_start.copy()
    mpc_path = dense_late_path(
        vortex_handoff, vortex_end,
        n_steps=args.n_mpc_vortex_steps,
        late_frac=args.late_frac,
        density_boost=args.density_boost,
        path_mode=args.path_mode,
        arc_height_mm=args.arc_height_mm,
    )
    n_mpc_path = len(mpc_path)
    print(f"  Phase II path: {n_mpc_path} waypoints, travel "
          f"{mpc_activation_dist*1e3:.2f} mm")

    # ── Compute view (covers full A→B path) ──────────────────────
    full_path_for_view = np.vstack([vortex_path, mpc_path])
    view_xlim, view_ylim = _compute_view(
        traps_m, idx_A, idx_B, full_path_for_view)
    print(f"  Render view: x=[{view_xlim[0]:.2f}, {view_xlim[1]:.2f}] "
          f"y=[{view_ylim[0]:.2f}, {view_ylim[1]:.2f}] mm")

    # ── MPC neighbour selection ───────────────────────────────────
    mid_AB = 0.5 * (traps_m[idx_A] + traps_m[idx_B])
    neigh_radius = args.n_neigh_radius * TRAP_SP
    dists_to_mid = np.linalg.norm(traps_m - mid_AB, axis=1)
    neigh_idx = np.array(
        [i for i in range(len(traps_m))
         if i not in (idx_A, idx_B) and dists_to_mid[i] < neigh_radius],
        dtype=int,
    )
    print(f"  MPC neighbours: {len(neigh_idx)} traps within "
          f"{args.n_neigh_radius:.1f} trap spacings of A-B midpoint")

    # ── Inertial particle ensemble setup ─────────────────────────
    particles_active = args.particle_grid_n > 0
    ensemble: ParticleEnsemble | None = None
    particle_recorder: ParticleRecorder | None = None

    if particles_active:
        ensemble = make_grid_ensemble(
            view_xlim, view_ylim, args.particle_grid_n)
        particle_recorder = ParticleRecorder()
        print(f"\n  Inertial particle ensemble:")
        print(f"    N = {ensemble.N} particles  "
              f"({args.particle_grid_n}×{args.particle_grid_n} grid)")
        print(f"    diameter    = {PARTICLE_DIAMETER*1e6:.0f} µm")
        print(f"    τ_Stokes    = {TAU_STOKES*1e3:.3f} ms")
        print(f"    integrator  = {args.particle_integrator}")
    else:
        print(f"\n  Inertial particle ensemble: DISABLED")

    # ── Output directory ──────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (PROJECT_ROOT / "results" / "dev"
               / "vortex_entry_hires_inertial_mpc" / timestamp)
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

    # ══════════════════════════════════════════════════════════════
    #  MPC setup — shared by Phase I and Phase II
    # ══════════════════════════════════════════════════════════════

    feval        = ForceEvaluator(p_sw, vortex_gen, xg, yg)
    fixed_alpha  = float(alpha)
    fixed_beta   = float(BETA_FIXED)
    psi_current  = 0.0
    maxiter_c    = args.n_iters
    mpc_J_history: List[float] = []
    mpc_inner_histories: List[List[float]] = []

    K_H          = args.K_phase1      # horizon length (physics steps)
    REPLAN_EVERY = args.replan_every  # replan every N dynamics steps
    step_scale   = SCALE * DT_DEFAULT
    merge_time_s: Optional[float] = None
    mpc_elapsed  = 0.0

    # ── Greedy fallback threshold (0 = disabled) ─────────────────
    greedy_fallback_dist = args.greedy_late_dist * LAM   # metres

    # ── Shared receding-horizon psi solver ────────────────────────
    _w_pull    = float(args.w_pull)
    _w_lateral = float(args.w_lateral)
    _w_retain  = float(args.w_retain)
    _w_term    = float(args.w_term)

    def _solve_psi_horizon(
        psi_h0: np.ndarray,
        pA: np.ndarray,
        pB: np.ndarray,
        vc_seq: List[np.ndarray],
    ) -> Tuple[np.ndarray, float]:
        """
        Optimise K ψ values via a K-step receding-horizon loop.

        Forward: roll out K overdamped dynamics steps using feval.forces()
                 (basis, same model as execution — no model-reality gap).
        Cost:    RUNNING force-based sum over the horizon —
                 J = Σ_k [ -w_pull·F_in_B_k + w_lat·F_perp_B_k - w_ret·F_A_in_k ]
                 Same physical objective as greedy phase_sweep but with
                 K-step look-ahead and evolved states (not 1-step myopic).
        Gradient: ∂J/∂ψ_k = ∂J_k/∂ψ_k evaluated at evolved state traj[k]
                  (frozen dynamics Jacobian — valid for short horizons &
                  overdamped particles; avoids expensive dF/dx computation).
                  Uses feval.dF_dpsi(): analytic, no FD on the full field.
        ψ is unconstrained: periodicity handled by cos/sin in the basis.
        Warm-started from the shifted previous solution — no restarts.
        """
        def _obj(psi_flat: np.ndarray) -> Tuple[float, np.ndarray]:
            # ── Forward rollout ──────────────────────────────────
            pA_k = pA.copy()
            pB_k = pB.copy()
            traj_A = [pA_k.copy()]
            traj_B = [pB_k.copy()]
            for k in range(K_H):
                vc_k = vc_seq[k]
                u_k  = np.array([psi_flat[k], vc_k[0], vc_k[1],
                                  fixed_alpha, fixed_beta])
                pos_ab = np.vstack([pA_k[None, :], pB_k[None, :]])
                Fx_k, Fy_k = feval.forces(u_k, pos_ab)
                pA_k = pA_k + step_scale * np.array(
                    [float(Fx_k[0]), float(Fy_k[0])])
                pB_k = pB_k + step_scale * np.array(
                    [float(Fx_k[1]), float(Fy_k[1])])
                traj_A.append(pA_k.copy())
                traj_B.append(pB_k.copy())

            # ── Running force-based cost + gradient ──────────────
            # Frozen Jacobian: ∂J/∂ψ_k = ∂J_k/∂ψ_k at state traj[k]
            J = 0.0
            grads = np.zeros(K_H)
            dpsi_cache: List[np.ndarray] = []
            for k in range(K_H):
                vc_k   = vc_seq[k]
                pA_k   = traj_A[k]
                pB_k   = traj_B[k]
                u_k    = np.array([psi_flat[k], vc_k[0], vc_k[1],
                                    fixed_alpha, fixed_beta])
                pos_ab = np.vstack([pA_k[None, :], pB_k[None, :]])
                Fx_k, Fy_k = feval.forces(u_k, pos_ab)
                FA = np.array([float(Fx_k[0]), float(Fy_k[0])])
                FB = np.array([float(Fx_k[1]), float(Fy_k[1])])

                # Direction vectors (unit, vortex-relative)
                dA    = vc_k - pA_k
                nA    = float(np.linalg.norm(dA))
                e_A   = dA / max(nA, 1e-12)
                dB    = vc_k - pB_k
                nB    = float(np.linalg.norm(dB))
                e_B   = dB / max(nB, 1e-12)

                F_A_in  = float(np.dot(FA, e_A))
                F_in_B  = float(np.dot(FB, e_B))
                F_tang  = FB - F_in_B * e_B
                F_perp  = float(np.linalg.norm(F_tang))

                J += (-_w_pull * F_in_B
                      + _w_lateral * F_perp
                      - _w_retain * F_A_in)

                # Analytic gradient via dF_dpsi — no FD
                dpsi  = feval.dF_dpsi(u_k, pos_ab)  # (2, 2): [part, xy]
                dFA   = dpsi[0]    # [∂FAx/∂ψ, ∂FAy/∂ψ]
                dFB   = dpsi[1]    # [∂FBx/∂ψ, ∂FBy/∂ψ]

                dF_A_in  = float(np.dot(dFA, e_A))
                dF_in_B  = float(np.dot(dFB, e_B))
                dF_tang  = dFB - dF_in_B * e_B
                dF_perp  = (float(np.dot(F_tang, dF_tang))
                             / max(F_perp, 1e-30))

                grads[k] = (-_w_pull * dF_in_B
                             + _w_lateral * dF_perp
                             - _w_retain * dF_A_in)
                dpsi_cache.append(dpsi)

            # ── Terminal position cost: (||pA_K - pB_K|| / LAM)^2 ──
            # Normalised by λ² so w_term is O(1) (same scale as
            # running force terms).  Adjoint gradient (frozen Jacobian):
            #   ∂J_term/∂ψ_k ≈ 2*w_term/λ² * (pA_K - pB_K)^T
            #                   * step_scale*(∂FA_k/∂ψ_k - ∂FB_k/∂ψ_k)
            # No extra dF_dpsi calls — reuses dpsi_cache from above.
            if _w_term > 0.0:
                res_AB = traj_A[K_H] - traj_B[K_H]  # (2,)
                J += _w_term * float(np.dot(res_AB, res_AB)) / LAM**2
                term_c = _w_term * 2.0 * step_scale / LAM**2
                for k, dp in enumerate(dpsi_cache):
                    grads[k] += term_c * float(
                        np.dot(res_AB, dp[0] - dp[1]))

            return J, grads

        res = sp_minimize(
            _obj, psi_h0.copy(),
            jac=True, method="L-BFGS-B",
            options={"maxiter": maxiter_c, "ftol": 1e-15, "gtol": 1e-9},
        )
        return res.x, float(res.fun)

    # ══════════════════════════════════════════════════════════════
    #  Phase I — Receding-horizon psi MPC (full A→B path)
    #
    #  True K-step receding horizon over ψ alone.  Vortex follows
    #  the prescribed dense_late_path externally.
    #
    #  Key improvements over the old per-waypoint adjoint sweep:
    #   • Position-based cost (not force-based) — directly optimises
    #     where A ends up, not what force acts right now.
    #   • Fully analytic gradient via feval.dF_du — no FD on the
    #     full pressure field, no inconsistent mixed computation.
    #   • Warm-start: shift horizon forward one step each replan →
    #     single L-BFGS-B solve per step, no 8 random restarts.
    #   • Basis cache shared across all replans at the same vortex
    #     position — rebuilt only when the vortex moves (20×).
    #   • Dynamics use feval.forces() — identical model to optimizer
    #     (no model-reality gap from basis vs. full-field mismatch).
    # ══════════════════════════════════════════════════════════════

    # Flatten vortex schedule → per-physics-step list
    flat_schedule: List[Tuple[int, int, np.ndarray, bool, int]] = []
    for _vs in range(n_path):
        _vc_v   = vortex_path[_vs].copy()
        _late_v = _vs >= late_start_idx
        _nd_v   = (int(round(args.n_dyn_steps * args.slow_late_factor))
                   if _late_v else args.n_dyn_steps)
        for _di_v in range(_nd_v):
            flat_schedule.append((_vs, _di_v, _vc_v, _late_v, _nd_v))

    N_SCHED     = len(flat_schedule)
    psi_horizon = np.full(K_H, psi_current)   # warm-start at ψ = 0

    print(f"\n  Phase I — Receding-horizon psi MPC: "
          f"{N_SCHED} steps, K={K_H}, replan_every={REPLAN_EVERY}, "
          f"alpha={alpha:.1f}")

    prev_vc_key         = None
    best_J_i            = np.inf
    cur_snap_vstep      = -1
    step_snaps_set: set = set()
    greedy_mode         = False   # switched on when d(A,B) < greedy_fallback_dist
    iFx_greedy: Any     = None    # full-field force interpolators (greedy mode)
    iFy_greedy: Any     = None
    psi_greedy          = 0.0     # last greedy-mode psi (re-used within waypoint)
    greedy_vc_key       = None    # separate key — tracks when greedy field was built

    for global_i, (v_step, dyn_i, vc, is_late, n_dyn) in enumerate(
            flat_schedule):

        # Update snapshot set and vortex trail when waypoint advances
        if v_step != cur_snap_vstep:
            cur_snap_vstep = v_step
            n_snap = args.late_fps if is_late else args.base_fps
            step_snaps_set = set(
                np.linspace(0, n_dyn - 1, n_snap, dtype=int).tolist()
            )
            vortex_trail.append(vc.copy())

        # Clear basis cache only when vortex position changes (≈20×/run)
        vc_key = (round(float(vc[0]), 8), round(float(vc[1]), 8))
        if vc_key != prev_vc_key:
            feval.clear_cache()
            prev_vc_key = vc_key

        # Build K-step vortex look-ahead
        vc_horizon = [
            flat_schedule[min(global_i + k, N_SCHED - 1)][2]
            for k in range(K_H)
        ]

        # ── Check greedy-fallback threshold ──────────────────────
        d_AB_pre = float(np.linalg.norm(pos_A - pos_B))
        if greedy_fallback_dist > 0 and (not greedy_mode) and d_AB_pre < greedy_fallback_dist:
            greedy_mode = True
            print(f"  [GREEDY] Switching to greedy at step {global_i}, "
                  f"d(A,B)={d_AB_pre*1e6:.0f}µm < {greedy_fallback_dist*1e6:.0f}µm")

        if greedy_mode:
            # ── Greedy fallback: phase_sweep + full-field dynamics ──
            # Rebuild field only when vortex position changes (same
            # frequency as phase_sweep in the reference greedy script).
            # NOTE: use greedy_vc_key (not prev_vc_key) because prev_vc_key
            # is updated earlier in this iteration for the basis cache.
            if vc_key != greedy_vc_key or iFx_greedy is None:
                greedy_vc_key = vc_key
                best_g = phase_sweep(
                    p_sw, vortex_gen, xg, yg, dx, dy,
                    psi_values, fixed_alpha, fixed_beta,
                    vc, pos_A, pos_B, r_barrier,
                    w_barrier=0.0,
                    w_pull=_w_pull,
                    w_lateral=_w_lateral,
                    w_retain=_w_retain,
                )
                psi_greedy  = float(best_g["psi"])
                iFx_greedy  = best_g["iFx"]
                iFy_greedy  = best_g["iFy"]
                best_J_i    = float(best_g["score"])
            psi_apply   = psi_greedy
            psi_current = psi_apply
            pos_A, pos_B = update_particles(
                pos_A, pos_B, iFx_greedy, iFy_greedy,
                args.max_step, xg, yg,
            )

        else:
            # ── Receding-horizon psi MPC ──────────────────────────
            # Replan when scheduled
            if global_i % REPLAN_EVERY == 0:
                psi_horizon, best_J_i = _solve_psi_horizon(
                    psi_horizon, pos_A, pos_B, vc_horizon,
                )
                mpc_J_history.append(best_J_i)
                mpc_inner_histories.append([best_J_i])

            # Apply first ψ; warm-start shift
            psi_apply   = float(psi_horizon[0])
            psi_current = psi_apply
            psi_horizon = np.concatenate([psi_horizon[1:], [psi_horizon[-1]]])

            # Step A and B — fast basis, same model as optimizer
            u_apply      = np.array([psi_apply, vc[0], vc[1],
                                      fixed_alpha, fixed_beta])
            pos_ab       = np.vstack([pos_A[None, :], pos_B[None, :]])
            Fx_ab, Fy_ab = feval.forces(u_apply, pos_ab)
            dA  = step_scale * np.array([float(Fx_ab[0]), float(Fy_ab[0])])
            dB  = step_scale * np.array([float(Fx_ab[1]), float(Fy_ab[1])])
            nd_A = float(np.linalg.norm(dA))
            nd_B = float(np.linalg.norm(dB))
            if nd_A > args.max_step:
                dA *= args.max_step / nd_A
            if nd_B > args.max_step:
                dB *= args.max_step / nd_B
            pos_A = np.clip(pos_A + dA, [xg[0], yg[0]], [xg[-1], yg[-1]])
            pos_B = np.clip(pos_B + dB, [xg[0], yg[0]], [xg[-1], yg[-1]])

        # Inertial ensemble — rebuild full field only at snapshot steps
        if particles_active and dyn_i in step_snaps_set:
            p_snap = total_pressure(p_sw, vortex_gen, psi_apply,
                                    fixed_alpha, fixed_beta, vc)
            _, Fx_n, Fy_n = gorkov_normalised(p_snap, dx, dy)
            iFx_snap = make_interp(Fx_n, xg, yg)
            iFy_snap = make_interp(Fy_n, xg, yg)
            ensemble = advance_ensemble(
                ensemble, iFx_snap, iFy_snap, xg, yg,
                integrator=args.particle_integrator)

        # Merge detection
        d_AB_cur = float(np.linalg.norm(pos_A - pos_B))
        if merge_time_s is None and d_AB_cur < CAPTURE_RADIUS:
            merge_time_s = global_i * DT_DEFAULT
            print(f"  MERGE at step {global_i} "
                  f"(t={merge_time_s * 1e3:.1f} ms)")

        # Record timeseries
        _phase_lbl = "GREEDY" if greedy_mode else "CARRY"
        recorder.record(
            v_step=v_step, dyn_i=dyn_i,
            phase_label=_phase_lbl,
            psi=psi_apply, alpha=alpha,
            pos_A=pos_A.copy(), pos_B=pos_B.copy(),
            vortex_center=vc, r_barrier=r_barrier,
        )

        # GIF snapshot
        if dyn_i in step_snaps_set:
            b_stat = _b_status(pos_B, vc, r_barrier)
            frames_data.append({
                "pos_A": pos_A.copy(),
                "pos_B": pos_B.copy(),
                "vortex_center": vc.copy(),
                "vortex_trail": [v.copy() for v in vortex_trail],
                "psi": psi_apply, "alpha": alpha,
                "d_AB": float(np.linalg.norm(pos_A - pos_B)),
                "label": _phase_lbl,
                "B_status": b_stat,
            })
            if particles_active:
                particle_recorder.record_ensemble(ensemble)

        if (global_i + 1) % max(1, N_SCHED // 20) == 0:
            print(f"  [CARRY] step {global_i + 1:04d}/{N_SCHED} | "
                  f"ψ={psi_apply:.3f} | d(A,B)={d_AB_cur * 1e6:.1f}µm | "
                  f"J={best_J_i:.3e}")

    d_AB_post_carry = float(np.linalg.norm(pos_A - pos_B))
    d_AB_post_mpc   = d_AB_post_carry   # updated by Phase II if active
    n_mpc_dyn       = args.n_mpc_dyn_steps
    print(f"\n  Phase I complete: d(A,B) = {d_AB_post_carry * 1e6:.1f} µm")
    if merge_time_s is None and d_AB_post_carry < CAPTURE_RADIUS:
        merge_time_s = N_SCHED * DT_DEFAULT

    # ══════════════════════════════════════════════════════════════
    #  Phase II — Optional close-range adjoint (if activation radius > 0)
    # ══════════════════════════════════════════════════════════════
    if n_mpc_path > 0 and mpc_activation_dist > 1e-6:
        t0_mpc = time.time()
        mpc_late_start = int(round(n_mpc_path * (1 - args.late_frac)))

        # Flatten Phase II schedule
        flat_mpc: List[Tuple[int, int, np.ndarray, bool, int]] = []
        for _vm in range(n_mpc_path):
            _vc_m   = mpc_path[_vm].copy()
            _late_m = _vm >= mpc_late_start
            _nd_m   = (int(round(n_mpc_dyn * args.slow_late_factor))
                       if _late_m else n_mpc_dyn)
            for _di_m in range(_nd_m):
                flat_mpc.append((_vm, _di_m, _vc_m, _late_m, _nd_m))

        N_MPC = len(flat_mpc)
        # Warm-start horizon from Phase I final ψ
        psi_horizon_ii = np.full(K_H, psi_current)

        print(f"\n  Phase II — Receding-horizon psi MPC (close-range): "
              f"{N_MPC} steps, K={K_H}, replan_every={REPLAN_EVERY}")

        prev_vc_key_ii      = None
        best_J_ii           = np.inf
        cur_snap_vm         = -1
        step_snaps_ii: set  = set()

        for gii, (v_step, dyn_i, vc, is_late, n_dyn) in enumerate(flat_mpc):

            if v_step != cur_snap_vm:
                cur_snap_vm = v_step
                n_snap = args.late_fps if is_late else args.base_fps
                step_snaps_ii = set(
                    np.linspace(0, n_dyn - 1, n_snap, dtype=int).tolist()
                )
                vortex_trail.append(vc.copy())

            vc_key_ii = (round(float(vc[0]), 8), round(float(vc[1]), 8))
            if vc_key_ii != prev_vc_key_ii:
                feval.clear_cache()
                prev_vc_key_ii = vc_key_ii

            vc_hor_ii = [
                flat_mpc[min(gii + k, N_MPC - 1)][2]
                for k in range(K_H)
            ]

            if gii % REPLAN_EVERY == 0:
                psi_horizon_ii, best_J_ii = _solve_psi_horizon(
                    psi_horizon_ii, pos_A, pos_B, vc_hor_ii,
                )
                mpc_J_history.append(best_J_ii)
                mpc_inner_histories.append([best_J_ii])

            psi_apply_ii    = float(psi_horizon_ii[0])
            psi_current     = psi_apply_ii
            psi_horizon_ii  = np.concatenate(
                [psi_horizon_ii[1:], [psi_horizon_ii[-1]]])

            u_appl_ii        = np.array([psi_apply_ii, vc[0], vc[1],
                                          fixed_alpha, fixed_beta])
            pos_ab_ii        = np.vstack([pos_A[None, :], pos_B[None, :]])
            Fx_ii, Fy_ii     = feval.forces(u_appl_ii, pos_ab_ii)
            dA_ii = step_scale * np.array([float(Fx_ii[0]), float(Fy_ii[0])])
            dB_ii = step_scale * np.array([float(Fx_ii[1]), float(Fy_ii[1])])
            nd_Aii = float(np.linalg.norm(dA_ii))
            nd_Bii = float(np.linalg.norm(dB_ii))
            if nd_Aii > args.max_step:
                dA_ii *= args.max_step / nd_Aii
            if nd_Bii > args.max_step:
                dB_ii *= args.max_step / nd_Bii
            pos_A = np.clip(pos_A + dA_ii, [xg[0], yg[0]], [xg[-1], yg[-1]])
            pos_B = np.clip(pos_B + dB_ii, [xg[0], yg[0]], [xg[-1], yg[-1]])

            if particles_active and dyn_i in step_snaps_ii:
                p_snap_ii = total_pressure(p_sw, vortex_gen, psi_apply_ii,
                                           fixed_alpha, fixed_beta, vc)
                _, Fx_n, Fy_n = gorkov_normalised(p_snap_ii, dx, dy)
                iFx_s = make_interp(Fx_n, xg, yg)
                iFy_s = make_interp(Fy_n, xg, yg)
                ensemble = advance_ensemble(
                    ensemble, iFx_s, iFy_s, xg, yg,
                    integrator=args.particle_integrator)

            d_AB_ii = float(np.linalg.norm(pos_A - pos_B))
            if merge_time_s is None and d_AB_ii < CAPTURE_RADIUS:
                merge_time_s = (N_SCHED + gii) * DT_DEFAULT
                print(f"  MERGE (Phase II) at step {gii} "
                      f"(t={merge_time_s * 1e3:.1f} ms)")

            recorder.record(
                v_step=n_path + v_step, dyn_i=dyn_i,
                phase_label="MPC",
                psi=psi_apply_ii, alpha=fixed_alpha,
                pos_A=pos_A.copy(), pos_B=pos_B.copy(),
                vortex_center=vc, r_barrier=r_barrier)

            if dyn_i in step_snaps_ii:
                b_stat = _b_status(pos_B, vc, r_barrier)
                frames_data.append({
                    "pos_A": pos_A.copy(),
                    "pos_B": pos_B.copy(),
                    "vortex_center": vc.copy(),
                    "vortex_trail": [v.copy() for v in vortex_trail],
                    "psi": psi_apply_ii, "alpha": fixed_alpha,
                    "d_AB": float(np.linalg.norm(pos_A - pos_B)),
                    "label": "MPC", "B_status": b_stat})
                if particles_active:
                    particle_recorder.record_ensemble(ensemble)

        mpc_elapsed = time.time() - t0_mpc
        d_AB_post_mpc = float(np.linalg.norm(pos_A - pos_B))
        print(f"\n  Phase II complete: d(A,B) = {d_AB_post_mpc * 1e6:.1f} µm "
              f"({mpc_elapsed:.1f}s)")

    # ══════════════════════════════════════════════════════════════
    #  Phase III — Settling (α→0 ramp, SW recapture)
    # ══════════════════════════════════════════════════════════════
    if args.T_settle > 0:
        vc_final = (mpc_path[-1].copy() if (n_mpc_path > 0 and mpc_activation_dist > 1e-6)
                    else vortex_path[-1].copy() if n_path > 0
                    else vortex_start.copy())
        alpha_settle = fixed_alpha
        N_RAMP = min(50, args.T_settle)
        FIELD_REFRESH = 5  # rebuild field every N steps during ramp

        print(f"\n  Phase III — Settling: {args.T_settle} steps "
              f"(ramp {N_RAMP} + relax {args.T_settle - N_RAMP})")

        settle_snaps = set(
            np.linspace(0, args.T_settle - 1, 20, dtype=int).tolist()
        )

        iFx_s: Any = None
        iFy_s: Any = None
        prev_a_k: Optional[float] = None

        for k in range(args.T_settle):
            if k < N_RAMP:
                frac = (k + 1) / N_RAMP
                a_k = alpha_settle * (1.0 - frac)
            else:
                a_k = 0.0

            # Rebuild field when α changes enough (or first step)
            if iFx_s is None or (k < N_RAMP and k % FIELD_REFRESH == 0):
                p_s = total_pressure(p_sw, vortex_gen, psi_current,
                                     a_k, BETA_FIXED, vc_final)
                _, Fx_s, Fy_s = gorkov_normalised(p_s, dx, dy)
                iFx_s = make_interp(Fx_s, xg, yg)
                iFy_s = make_interp(Fy_s, xg, yg)
                prev_a_k = a_k

            pos_A, pos_B = update_particles(
                pos_A, pos_B, iFx_s, iFy_s,
                args.max_step, xg, yg,
            )

            if merge_time_s is None:
                d_check = float(np.linalg.norm(pos_A - pos_B))
                if d_check < CAPTURE_RADIUS:
                    merge_time_s = -1.0  # approximate

            recorder.record(
                v_step=n_path + n_mpc_path + k, dyn_i=0,
                phase_label="SETTLE",
                psi=psi_current, alpha=a_k,
                pos_A=pos_A.copy(), pos_B=pos_B.copy(),
                vortex_center=vc_final, r_barrier=r_barrier,
            )

            if k in settle_snaps:
                b_stat = _b_status(pos_B, vc_final, r_barrier)
                frames_data.append({
                    "pos_A": pos_A.copy(),
                    "pos_B": pos_B.copy(),
                    "vortex_center": vc_final.copy(),
                    "vortex_trail": [v.copy() for v in vortex_trail],
                    "psi": psi_current, "alpha": a_k,
                    "d_AB": float(np.linalg.norm(pos_A - pos_B)),
                    "label": "SETTLE",
                    "B_status": b_stat,
                })

        d_settle = float(np.linalg.norm(pos_A - pos_B))
        print(f"  Settle done. d(A,B) = {d_settle*1e6:.1f} µm")

    # ── Final status ──────────────────────────────────────────────
    d_final = float(np.linalg.norm(pos_A - pos_B))
    merged = d_final < CAPTURE_RADIUS
    print(f"\n  Final d(A,B) = {d_final*1e6:.1f} µm  "
          f"{'MERGED' if merged else 'NOT MERGED'}")

    # ── Outputs ───────────────────────────────────────────────────
    gif_path = out_dir / f"vortex_entry_hires_mpc_alpha_{alpha:.1f}.gif"
    render_gif(frames_data, xg, yg, dx, dy, p_sw, vortex_gen,
               traps_m, idx_A, idx_B, r_barrier_mm,
               view_xlim, view_ylim, gif_path)

    csv_path = out_dir / "timeseries.csv"
    recorder.to_csv(csv_path)
    print(f"  Saved: {csv_path} ({len(recorder.rows)} rows)")

    plot_path = out_dir / "proximity_vs_time.png"
    plot_proximity(recorder, r_barrier, plot_path)

    conv_path = out_dir / "convergence.png"
    plot_convergence(mpc_J_history, mpc_inner_histories, conv_path)

    if particles_active:
        particle_csv = out_dir / "particles.csv"
        particle_recorder.to_csv(particle_csv)
        print(f"  Saved: {particle_csv}")

    # ── Manifest ──────────────────────────────────────────────────
    elapsed = time.time() - t0_total
    manifest = {
        "timestamp": timestamp,
        "elapsed_s": round(elapsed, 2),
        "script": "vortex_entry_hires_inertial_mpc.py",
        "control_method": "receding_horizon_psi_mpc",
        "alpha": alpha,
        "beta_fixed": BETA_FIXED,
        # Phase I receding-horizon psi MPC config
        "phase_I_rh_mpc": {
            "optimizer": "L-BFGS-B",
            "horizon_K": K_H,
            "replan_every": REPLAN_EVERY,
            "maxiter_per_replan": maxiter_c,
            "cost": "running force: -w_pull*F_in_B + w_lat*F_perp_B - w_ret*F_A_in + w_term*||pA_K-pB_K||^2",
            "w_pull": _w_pull,
            "w_lateral": _w_lateral,
            "w_retain": _w_retain,
            "w_term": _w_term,
            "gradient": "analytic ∂J/∂ψ_k via feval.dF_dpsi (no FD on full field); terminal cost via frozen-Jacobian adjoint",
            "warm_start": True,
            "n_vortex_steps_base": args.n_vortex_steps,
            "n_vortex_steps_actual": n_path,
            "total_physics_steps": N_SCHED,
            "late_frac": args.late_frac,
            "density_boost": args.density_boost,
            "n_dyn_steps_base": args.n_dyn_steps,
            "slow_late_factor": args.slow_late_factor,
            "max_step_m": args.max_step,
            "alpha_fixed": fixed_alpha,
            "beta_fixed": fixed_beta,
            "path_mode": args.path_mode,
            "arc_height_mm": args.arc_height_mm,
            "mpc_activation_radius_lam": args.mpc_activation_radius,
            "greedy_late_dist_lam": args.greedy_late_dist,
            "greedy_mode_activated": greedy_mode,
            "d_AB_post_carry_um": round(d_AB_post_carry * 1e6, 1),
            "n_replans": len(mpc_J_history),
        },
        # Phase II close-range adjoint (optional)
        "phase_II_close_range": {
            "n_mpc_vortex_steps": n_mpc_path,
            "n_mpc_dyn_steps": n_mpc_dyn,
            "d_AB_post_mpc_um": round(d_AB_post_mpc * 1e6, 1),
            "mpc_elapsed_s": round(mpc_elapsed, 1),
        },
        # Phase III settle config
        "phase_III_settle": {
            "T_settle": args.T_settle,
        },
        # Results
        "d_final_um": round(d_final * 1e6, 1),
        "merged": merged,
        "merge_time_s": merge_time_s,
        # Geometry
        "aperture_mm": args.bessel_aperture_mm,
        "prop_dist_mm": args.bessel_prop_dist_mm,
        "focus_mm": args.focus_mm,
        "r_barrier_um": round(r_barrier * 1e6, 1),
        "initial_geometry": geom_status,
        "idx_A": idx_A,
        "idx_B": idx_B,
        "n_traps_total": len(traps_m),
        "n_neighbours": len(neigh_idx),
        "n_gif_frames": len(frames_data),
        "n_timeseries_rows": len(recorder.rows),
        # Inertial particles
        "particles_active": particles_active,
        "particle_grid_n": args.particle_grid_n,
        "particle_n_total": ensemble.N if particles_active else 0,
    }
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Saved: {manifest_path}")

    print(f"\nDone in {elapsed:.1f}s — {len(frames_data)} GIF frames")
    print(f"Results: {out_dir}")


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Vortex entry — high-res inertial with adjoint "
                    "ψ-optimisation (force-based cost, prescribed vortex path)")

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

    g = parser.add_argument_group("Phase I: carry")
    g.add_argument("--alpha", type=float, default=ALPHA_DEFAULT,
                   help="Vortex amplitude for Phase I carry (default %(default)s)")
    g.add_argument("--n_psi", type=int, default=N_PSI_DEFAULT,
                   help="(unused — kept for backward compat)")
    g.add_argument("--n_vortex_steps", type=int, default=N_VORTEX_STEPS)
    g.add_argument("--n_dyn_steps", type=int, default=N_DYN_STEPS_PER_POS)
    g.add_argument("--max_step", type=float, default=MAX_STEP_DEFAULT)
    g.add_argument("--mpc_activation_radius", type=float,
                   default=MPC_ACTIVATION_RADIUS_LAM,
                   help="Distance from B at which Phase II activates [λ] "
                        "(default %(default)s)")
    g.add_argument("--K_phase1", type=int, default=8,
                   help="Receding-horizon length K for Phase I psi MPC "
                        "(default %(default)s)")
    g.add_argument("--replan_every", type=int, default=1,
                   help="Replan every N physics steps (default %(default)s)")
    g.add_argument("--w_barrier", type=float, default=W_BARRIER_DEFAULT,
                   help="(legacy — unused in receding-horizon mode)")
    g.add_argument("--w_pull", type=float, default=W_PULL_DEFAULT,
                   help="(legacy — unused in receding-horizon mode)")
    g.add_argument("--w_lateral", type=float, default=W_LATERAL_DEFAULT,
                   help="(legacy — unused in receding-horizon mode)")
    g.add_argument("--w_retain", type=float, default=W_RETAIN_DEFAULT,
                   help="(legacy — unused in receding-horizon mode)")
    g.add_argument("--w_term", type=float, default=0.0,
                   help="Terminal position cost weight W_term*||pA_K - pB_K||^2 "
                        "(0=disabled; try 5-20 to fix late-stage stall)")
    g.add_argument("--greedy_late_dist", type=float, default=0.0,
                   help="Switch to greedy phase_sweep when d(A,B) < N*λ "
                        "(0=disabled; try 2.0 for reliable final merge)")

    g = parser.add_argument_group("Phase II: adjoint ψ-optimisation")
    g.add_argument("--n_iters", type=int, default=MPC_N_ITERS_DEFAULT,
                   help="L-BFGS-B max iterations per start (default %(default)s)")
    g.add_argument("--n_mpc_vortex_steps", type=int,
                   default=MPC_N_VORTEX_STEPS_DEFAULT,
                   help="Vortex waypoints in adjoint phase (default %(default)s)")
    g.add_argument("--n_mpc_dyn_steps", type=int,
                   default=MPC_N_DYN_STEPS_DEFAULT,
                   help="Dynamics steps per adjoint waypoint (default %(default)s)")
    g.add_argument("--n_neigh_radius", type=float,
                   default=MPC_N_NEIGH_RADIUS_DEFAULT,
                   help="Neighbour radius in trap spacings")

    g = parser.add_argument_group("Phase III: settle")
    g.add_argument("--T_settle", type=int, default=MPC_T_SETTLE_DEFAULT,
                   help="Settling phase steps (default %(default)s)")

    g = parser.add_argument_group("diagnostic: path density")
    g.add_argument("--late_frac", type=float, default=LATE_FRAC_DEFAULT)
    g.add_argument("--density_boost", type=int, default=DENSITY_BOOST_DEFAULT)

    g = parser.add_argument_group("diagnostic: frame capture")
    g.add_argument("--base_fps", type=int, default=BASE_FPS_DEFAULT)
    g.add_argument("--late_fps", type=int, default=LATE_FPS_DEFAULT)

    g = parser.add_argument_group("diagnostic: slow late-stage")
    g.add_argument("--slow_late_factor", type=float,
                   default=SLOW_LATE_FACTOR_DEFAULT)

    g = parser.add_argument_group("path shape")
    g.add_argument("--path_mode", type=str, default="straight",
                   choices=["straight", "arc_left", "arc_right"])
    g.add_argument("--arc_height_mm", type=float, default=0.15)

    g = parser.add_argument_group("inertial particle ensemble")
    g.add_argument("--particle_grid_n", type=int,
                   default=PARTICLE_GRID_N_DEFAULT)
    g.add_argument("--particle_integrator", type=str,
                   default=PARTICLE_INTEGRATOR_DEFAULT,
                   choices=["rk4", "euler"])

    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
