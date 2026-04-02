#!/usr/bin/env python3
"""
Vortex Entry Test — fast, reduced experiment (no MPC) to test whether
phase-tuning can open a vortex ring so particle B can enter the core
without being repelled, while particle A remains trapped inside.

Sweeps phase psi at each vortex position along a straight or curved path
from A's trap toward B's trap. Selects psi that:
  - passes the A-retention constraint (F_A_in > 0)
  - minimises score = w1*DeltaU_entry - w2*F_in + w3*F_perp

Includes:
  - Automatic A-trap selection: searches the full 197-trap lattice for a
    pair satisfying d(B, vortex_start) > 1.2 * r_barrier
  - Vortex calibration: measures barrier ring radius from radial profile
  - Geometry validation: ensures B starts outside the barrier ring
  - Mechanism test: samples U around the barrier ring to distinguish
    true phase-created opening from global basin merging

Outputs:
    results/dev/vortex_entry_test/<timestamp>/
        vortex_entry_alpha_<X>.gif   — animated Gorkov potential
        phase_sweep_results.csv      — per-step metrics
        manifest.json                — config

Usage:
    python scripts/dev/vortex_entry_test.py
    python scripts/dev/vortex_entry_test.py --alphas 2.0 3.0 --n_psi 36
    python scripts/dev/vortex_entry_test.py --focus_mm 2.0 --bessel_prop_dist_mm 2.0
    python scripts/dev/vortex_entry_test.py --bessel_aperture_mm 4.0 --focus_mm 3.0 --bessel_prop_dist_mm 3.0
"""
from __future__ import annotations

import argparse
import csv
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
    build_vortex_path,
    phase_sweep,
    GEOM_MARGIN,
    OPENING_THRESHOLD,
)
from acoustweezers.experiments.vortex_entry.particles.dynamics import (
    update_particles,
    b_status as _b_status,
)

# ── Data path (full-domain, 197 traps) ───────────────────────────
FIELD_NPZ = (
    PROJECT_ROOT / "results" / "deliverables" / "vortex_stage_transport"
    / "transport" / "transport_case_for_gif.npz"
)

# ── Vortex design defaults ────────────────────────────────────────
BESSEL_APERTURE_MM_DEFAULT  = 3.5   # aperture radius [mm]
BESSEL_PROP_DIST_MM_DEFAULT = 3.0   # ASM propagation distance [mm]
FOCUS_MM_DEFAULT            = 3.0   # lens focal length [mm]; set = prop_dist for focal-plane eval

# ── Experiment defaults ───────────────────────────────────────────
N_PSI_DEFAULT        = 24
# α=2.0 is the canonical working value (2026-04-02): A carried + B enters simultaneously
# with soft A-retention in score (w_retain=1.0); use --alphas 1.0 2.0 3.0 to sweep
ALPHAS_DEFAULT       = [2.0]
BETA_FIXED           = 1.0
N_VORTEX_STEPS       = 20
N_DYN_STEPS_PER_POS  = 150
W_BARRIER_DEFAULT    = 1.0
W_PULL_DEFAULT       = 1.0
W_LATERAL_DEFAULT    = 0.5
W_RETAIN_DEFAULT     = 1.0
MAX_STEP_DEFAULT     = 2e-6
N_INTRO_FRAMES       = 15
FRAMES_PER_VSTEP     = 6

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
# GIF rendering (kept in main script — rendering/file I/O not in src)
# ══════════════════════════════════════════════════════════════════

def _compute_view(
    traps_m: np.ndarray,
    idx_A: int,
    idx_B: int,
    vortex_path: np.ndarray,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Compute a fixed view that covers A, B, their path, and any
    neighbouring traps within a modest radius -- decoupled from the
    full simulation domain.
    """
    # Key points that must be visible
    key_pts = np.vstack([
        traps_m[idx_A][None, :],
        traps_m[idx_B][None, :],
        vortex_path,
    ]) * 1e3  # mm

    # Also include any traps within 1.5mm of the A-B midpoint
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

    # ── Trap markers (only those inside the view) ────────────────
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

    # ── Vortex trail ─────────────────────────────────────────────
    if len(trail_mm) > 1:
        ax.plot(trail_mm[:, 0], trail_mm[:, 1], color=COL_CTR,
                lw=0.9, alpha=0.75, zorder=5)
    ax.plot(vc_mm[0], vc_mm[1], marker="+", color=COL_CTR,
            ms=12, mew=2.0, zorder=8)

    # ── Barrier ring ─────────────────────────────────────────────
    ax.add_patch(mpatches.Circle(
        (vc_mm[0], vc_mm[1]), r_barrier_mm,
        fill=False, edgecolor=COL_BARRIER, linestyle="--",
        linewidth=1.0, alpha=0.7, zorder=6))

    # ── Particles ────────────────────────────────────────────────
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

    # ── Title ────────────────────────────────────────────────────
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

    # ── Select A/B pair satisfying geometry constraint ────────────
    idx_A, idx_B = select_trap_pair(
        traps_m, r_barrier, xg, yg,
        idx_B_override=args.idx_B,
        min_sep_factor=args.min_sep_factor,
    )

    # ── Validate ──────────────────────────────────────────────────
    geom_status = validate_geometry(traps_m, idx_A, idx_B, r_barrier)

    vortex_start = traps_m[idx_A].copy()
    vortex_end   = traps_m[idx_B].copy()

    # ── Build path ────────────────────────────────────────────────
    psi_values = np.linspace(0, 2 * np.pi, args.n_psi, endpoint=False)
    vortex_path = build_vortex_path(
        vortex_start, vortex_end, args.n_vortex_steps,
        path_mode=args.path_mode,
        arc_height_mm=args.arc_height_mm,
    )
    print(f"  Path mode: {args.path_mode}"
          + (f" (arc height {args.arc_height_mm:.2f} mm)"
             if args.path_mode != "straight" else ""))

    # ── Compute render view (decoupled from sim domain) ───────────
    view_xlim, view_ylim = _compute_view(
        traps_m, idx_A, idx_B, vortex_path)
    print(f"  Render view: x=[{view_xlim[0]:.2f}, {view_xlim[1]:.2f}] "
          f"y=[{view_ylim[0]:.2f}, {view_ylim[1]:.2f}] mm")

    # ── Output directory ──────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (PROJECT_ROOT / "results" / "dev"
               / "vortex_entry_test" / timestamp)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {out_dir}")

    snap_indices = set(
        np.linspace(0, args.n_dyn_steps - 1, FRAMES_PER_VSTEP,
                    dtype=int).tolist()
    )

    all_csv_rows: List[List] = []

    # ── Two-phase schedule ────────────────────────────────────────
    n_carry = max(1, int(args.n_vortex_steps * args.n_carry_frac))
    if args.two_phase:
        alpha_schedule = (
            [args.alpha_carry] * n_carry
            + [args.alpha_entry] * (args.n_vortex_steps - n_carry)
        )
        alphas_to_run = ["two_phase"]
        print(f"\n  Two-phase mode: carry α={args.alpha_carry:.1f} for "
              f"{n_carry} steps, entry α={args.alpha_entry:.1f} for "
              f"{args.n_vortex_steps - n_carry} steps")
    else:
        alpha_schedule = None
        alphas_to_run = args.alphas

    for alpha in alphas_to_run:
        print(f"\n{'='*60}")
        if args.two_phase:
            print(f"  TWO-PHASE: carry α={args.alpha_carry:.1f} "
                  f"({n_carry} steps) → entry α={args.alpha_entry:.1f} "
                  f"({args.n_vortex_steps - n_carry} steps)")
        else:
            print(f"  alpha = {alpha:.1f}")
        print(f"{'='*60}")

        pos_A = traps_m[idx_A].copy()
        pos_B = traps_m[idx_B].copy()
        frames_data: List[Dict] = []
        vortex_trail: List[np.ndarray] = []

        # ── Intro frames (alpha=0, standing wave only) ────────────
        intro_vc = vortex_start.copy()
        b_stat_intro = _b_status(pos_B, intro_vc, r_barrier)
        for _ in range(N_INTRO_FRAMES):
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

        # ── Main loop ─────────────────────────────────────────────
        for v_step in range(args.n_vortex_steps):
            vortex_center = vortex_path[v_step].copy()
            vortex_trail.append(vortex_center.copy())

            t_sweep = time.time()
            # ── Resolve per-step alpha in two-phase mode ──────────
            if args.two_phase:
                effective_alpha = alpha_schedule[v_step]
                step_phase = "carry" if v_step < n_carry else "entry"
            else:
                effective_alpha = alpha
                step_phase = "sweep"
            best = phase_sweep(
                p_sw, vortex_gen, xg, yg, dx, dy,
                psi_values, effective_alpha, BETA_FIXED,
                vortex_center, pos_A, pos_B, r_barrier,
                w_barrier=args.w_barrier,
                w_pull=args.w_pull,
                w_lateral=args.w_lateral,
                w_retain=args.w_retain,
                opening_threshold=args.opening_threshold,
            )
            psi_best = float(best["psi"])
            DeltaU   = float(best["DeltaU"])
            F_in_B   = float(best["F_in_B"])
            F_perp_B = float(best["F_perp_B"])
            F_A_in   = float(best["F_A_in"])
            score    = float(best["score"])
            iFx, iFy = best["iFx"], best["iFy"]
            mech     = best["mechanism"]
            dt_sweep = time.time() - t_sweep

            ret_s = f"F_A_in={F_A_in:+.3e}"
            opn_s = mech["mechanism"].upper()
            print(f"  [{step_phase}] v_step {v_step+1:02d}/{args.n_vortex_steps} | "
                  f"α={effective_alpha:.1f} | psi={psi_best:.2f} | dU={DeltaU:.3e} | "
                  f"F_in_B={F_in_B:.3e} | F_A_in={F_A_in:.3e} "
                  f"({ret_s}) | {opn_s} | {dt_sweep*1e3:.0f}ms")

            # ── Dynamics ──────────────────────────────────────────
            for dyn_i in range(args.n_dyn_steps):
                pos_A, pos_B = update_particles(
                    pos_A, pos_B, iFx, iFy,
                    args.max_step, xg, yg,
                )

                if dyn_i in snap_indices:
                    b_stat = _b_status(pos_B, vortex_center, r_barrier)
                    frames_data.append({
                        "pos_A": pos_A.copy(),
                        "pos_B": pos_B.copy(),
                        "vortex_center": vortex_center.copy(),
                        "vortex_trail": [v.copy() for v in vortex_trail],
                        "psi": psi_best, "alpha": effective_alpha,
                        "d_AB": float(np.linalg.norm(pos_A - pos_B)),
                        "label": step_phase.upper(),
                        "B_status": b_stat,
                    })

            d_AB = float(np.linalg.norm(pos_A - pos_B))
            d_A_vc = float(np.linalg.norm(pos_A - vortex_center))
            d_B_vc = float(np.linalg.norm(pos_B - vortex_center))
            delta_B = d_B_vc - r_barrier
            A_trapped = bool(d_A_vc < r_barrier)
            b_stat_end = _b_status(pos_B, vortex_center, r_barrier)

            all_csv_rows.append([
                effective_alpha, v_step,
                vortex_center[0] * 1e3, vortex_center[1] * 1e3,
                psi_best, DeltaU, F_in_B, F_perp_B, F_A_in, score,
                pos_A[0] * 1e3, pos_A[1] * 1e3,
                pos_B[0] * 1e3, pos_B[1] * 1e3,
                d_AB * 1e6, delta_B * 1e6, b_stat_end,
                A_trapped, mech["mechanism"],
                mech["U_B_sector"], mech["U_other"],
                step_phase,
            ])

        # ── GIF ───────────────────────────────────────────────────
        gif_run_label = (
            f"two_phase_carry{args.alpha_carry:.1f}_entry{args.alpha_entry:.1f}"
            if args.two_phase else f"alpha_{alpha:.1f}"
        )
        gif_path = out_dir / f"vortex_entry_{gif_run_label}.gif"
        render_gif(frames_data, xg, yg, dx, dy, p_sw, vortex_gen,
                   traps_m, idx_A, idx_B, r_barrier_mm,
                   view_xlim, view_ylim, gif_path)

        # ── Summary ───────────────────────────────────────────────
        final = frames_data[-1]
        d_B_core = float(np.linalg.norm(
            final["pos_B"] - final["vortex_center"]))
        d_A_core = float(np.linalg.norm(
            final["pos_A"] - final["vortex_center"]))
        if args.two_phase:
            n_openings = sum(
                1 for r in all_csv_rows if r[18] == "true_opening")
            run_label = (f"two_phase (carry α={args.alpha_carry:.1f} / "
                         f"entry α={args.alpha_entry:.1f})")
        else:
            n_openings = sum(
                1 for r in all_csv_rows
                if r[0] == alpha and r[18] == "true_opening")
            run_label = f"alpha={alpha:.1f}"
        print(f"\n  {run_label} summary:")
        print(f"    d(A, vortex) = {d_A_core*1e6:.1f} um  "
              f"({'TRAPPED' if d_A_core < r_barrier else 'drifted'})")
        print(f"    d(B, vortex) = {d_B_core*1e6:.1f} um  "
              f"(delta_B = {(d_B_core - r_barrier)*1e6:.1f} um) "
              f"({final.get('B_status', '?')})")
        print(f"    True openings: {n_openings}/{args.n_vortex_steps}")

    # ── CSV ───────────────────────────────────────────────────────
    csv_path = out_dir / "phase_sweep_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "alpha", "v_step", "vortex_x_mm", "vortex_y_mm",
            "psi_best", "DeltaU_entry", "F_in_B", "F_perp_B", "F_A_in",
            "score", "A_x_mm", "A_y_mm", "B_x_mm", "B_y_mm",
            "d_AB_um", "delta_B_um", "B_status",
            "A_trapped", "mechanism", "U_B_sector", "U_other",
            "phase",
        ])
        writer.writerows(all_csv_rows)
    print(f"\nSaved CSV: {csv_path}")

    # ── Manifest ──────────────────────────────────────────────────
    elapsed = time.time() - t0_total
    manifest = {
        "timestamp": timestamp,
        "elapsed_s": round(elapsed, 2),
        "two_phase": args.two_phase,
        "alphas": args.alphas,
        "alpha_carry": args.alpha_carry,
        "alpha_entry": args.alpha_entry,
        "n_carry_frac": args.n_carry_frac,
        "n_carry_steps": n_carry,
        "beta_fixed": BETA_FIXED,
        "n_psi": args.n_psi,
        "n_vortex_steps": args.n_vortex_steps,
        "n_dyn_steps": args.n_dyn_steps,
        "max_step_m": args.max_step,
        "w_barrier": args.w_barrier,
        "w_pull": args.w_pull,
        "w_lateral": args.w_lateral,
        "w_retain": args.w_retain,
        "path_mode": args.path_mode,
        "arc_height_mm": args.arc_height_mm,
        "vortex_charge": 1,
        "aperture_mm": args.bessel_aperture_mm,
        "prop_dist_mm": args.bessel_prop_dist_mm,
        "focus_mm": args.focus_mm,
        "r_barrier_um": round(r_barrier * 1e6, 1),
        "initial_geometry": geom_status,
        "idx_A": idx_A,
        "idx_B": idx_B,
        "n_traps_total": len(traps_m),
        "n_intro_frames": N_INTRO_FRAMES,
        "frames_per_vstep": FRAMES_PER_VSTEP,
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone in {elapsed:.1f}s — results in {out_dir}")


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Vortex entry test — phase sweep without MPC")

    g = parser.add_argument_group("vortex beam")
    g.add_argument("--bessel_aperture_mm", type=float,
                   default=BESSEL_APERTURE_MM_DEFAULT,
                   help="Lens aperture radius [mm] — sets numerical aperture "
                        "(default %(default)s)")
    g.add_argument("--bessel_prop_dist_mm", type=float,
                   default=BESSEL_PROP_DIST_MM_DEFAULT,
                   help="ASM propagation distance [mm] — set equal to "
                        "--focus_mm to evaluate at the focal plane "
                        "(default %(default)s)")
    g.add_argument("--focus_mm", type=float,
                   default=FOCUS_MM_DEFAULT,
                   help="Lens focal length [mm] — controls ring tightness; "
                        "shorter f → smaller ring. Use with "
                        "--bessel_prop_dist_mm equal to this value for "
                        "focal-plane evaluation. (default %(default)s)")

    g = parser.add_argument_group("trap selection")
    g.add_argument("--idx_B", type=int, default=None,
                   help="Override B trap index (default: nearest to "
                        "grid centre)")
    g.add_argument("--min_sep_factor", type=float, default=GEOM_MARGIN,
                   help="A must satisfy d(A,B) > factor * r_barrier "
                        "(default %(default)s)")
    g.add_argument("--opening_threshold", type=float, default=OPENING_THRESHOLD,
                   help="delta_B fraction required for 'true_opening' "
                        "classification (default %(default)s)")

    g = parser.add_argument_group("sweep")
    g.add_argument("--n_psi", type=int, default=N_PSI_DEFAULT)
    g.add_argument("--alphas", nargs="+", type=float,
                   default=ALPHAS_DEFAULT)
    g.add_argument("--n_vortex_steps", type=int, default=N_VORTEX_STEPS)
    g.add_argument("--n_dyn_steps", type=int, default=N_DYN_STEPS_PER_POS)
    g.add_argument("--max_step", type=float, default=MAX_STEP_DEFAULT,
                   help="Max displacement per dyn step [m]")

    g = parser.add_argument_group("two-phase schedule")
    g.add_argument("--two_phase", action="store_true",
                   help="Run a single two-phase schedule (carry then entry) "
                        "instead of sweeping --alphas")
    g.add_argument("--alpha_carry", type=float, default=2.0,
                   help="Vortex amplitude during carry phase "
                        "(default %(default)s)")
    g.add_argument("--alpha_entry", type=float, default=1.0,
                   help="Vortex amplitude during entry phase "
                        "(default %(default)s)")
    g.add_argument("--n_carry_frac", type=float, default=0.7,
                   help="Fraction of vortex steps spent in carry phase "
                        "(default %(default)s)")

    g = parser.add_argument_group("score weights")
    g.add_argument("--w_barrier", type=float, default=W_BARRIER_DEFAULT)
    g.add_argument("--w_pull", type=float, default=W_PULL_DEFAULT)
    g.add_argument("--w_lateral", type=float, default=W_LATERAL_DEFAULT)
    g.add_argument("--w_retain", type=float, default=W_RETAIN_DEFAULT,
                   help="Weight on -F_A_in in score: penalises ψ choices "
                        "that weaken A's restoring force (default %(default)s)")

    g = parser.add_argument_group("path")
    g.add_argument("--path_mode", type=str, default="straight",
                   choices=["straight", "arc_left", "arc_right"])
    g.add_argument("--arc_height_mm", type=float, default=0.15)

    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
