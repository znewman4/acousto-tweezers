#!/usr/bin/env python3
"""
MPC Vortex Merge — closed-loop controller for merging particle A into B's trap.

Uses an analytic LG vortex perturbation with 5 control DOFs per timestep:
    (ψ, x_v, y_v, α, β)
and a receding-horizon MPC with discrete adjoint gradients.

Outputs:
    results/dev/mpc_vortex_merge/<run_dir>/
        mpc_vortex_merge.gif      — animated Gor'kov potential + trajectories
        manifest.json             — full config + metrics
        controls.csv              — per-step control history
        trajectory.csv            — per-step particle positions
        convergence.png           — MPC inner-loop convergence

Usage:
    python scripts/dev/mpc_vortex_merge.py
    python scripts/dev/mpc_vortex_merge.py --K 15 --T 2000 --n_iters 25
    python scripts/dev/mpc_vortex_merge.py --gradcheck   # verify gradients only
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["figure.dpi"] = 72
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.lib.fem_cache_utils import (
    C_WATER,
    F_HZ,
    OMEGA,
    RHO0,
    default_particle_params,
    gorkov_grid_2d,
)
from scripts.lib.mpc_controller import (
    I_ALPHA,
    I_BETA,
    I_PSI,
    I_XV,
    I_YV,
    N_CTRL,
    ForceEvaluator,
    MPCConfig,
    MPCResult,
    mpc_result_to_transport,
    run_mpc,
)
from scripts.lib.mpc_gorkov_basis import build_basis_for_vortex_position
from scripts.lib.particle_dynamics_utils import (
    CAPTURE_RADIUS,
    LAM,
    SCALE,
    TRAP_SP,
    TransportResult,
    compute_metrics,
)
from scripts.lib.asm_utils import make_vortex_field, make_bessel_vortex_field, propagate_asm, LAM as ASM_LAM, K0 as ASM_K0
from scripts.lib.perturbation_vortex import VortexPerturbation

PPAR = default_particle_params()

# ── Data paths ────────────────────────────────────────────────────

VORTEX_NPZ = (
    PROJECT_ROOT / "results" / "deliverables" / "vortex_stage_transport"
    / "transport" / "transport_case_for_gif.npz"
)
OVERLAY_NPZ = (
    PROJECT_ROOT / "results" / "c_shape_lens_15mm_overlay_study_20260310_170620"
    / "npz" / "roi_fields.npz"
)

# ── Rendering constants ──────────────────────────────────────────

CMAP = "RdBu_r"
COL_A = "#e74c3c"
COL_B = "#3498db"
COL_NEIGH = "#95a5a6"
COL_HOME = "#2ecc71"
COL_CTR = "#f39c12"
GORKOV_CLIP_LO = 0.5
GORKOV_CLIP_HI = 99.5

# Settling-phase defaults
T_SETTLE_DEFAULT = 3000          # 3000 steps × 0.1ms = 300ms for α→0 ramp + SW relaxation


# ── Vortex design defaults ───────────────────────────────────────

VORTEX_CHARGE = 1
VORTEX_WAIST = 0.15e-3       # m  (LG fallback only)
VORTEX_APERTURE = 0.8e-3     # m  (LG fallback only)

# Bessel vortex defaults (primary model)
BESSEL_APERTURE_RADIUS = 2.0e-3  # m  (physical transducer radius, 2mm)
BESSEL_PROP_DIST = 5.0e-3        # m  (source → measurement plane)

# ── Remote particle / open-loop defaults ─────────────────────────

REMOTE_DISTANCE_LAM = 3.0        # start A this many λ away from B
MPC_ACTIVATION_RADIUS_LAM = 1.0  # activate MPC at this distance from B (in λ)  [>0.6λ ⇒ skip open-loop]
OPEN_LOOP_SPEED = 5.0e-3          # m/s  (5 mm/s max cruise — tethered to particle A)

# ── GIF settings ─────────────────────────────────────────────────

N_GIF_FRAMES = 200
GIF_DURATION_MS = 50


# ════════════════════════════════════════════════════════════════
# Physics helpers
# ════════════════════════════════════════════════════════════════

def total_pressure(
    p_sw: np.ndarray,
    vortex_gen: VortexPerturbation,
    psi: float,
    alpha: float,
    beta: float,
    center_xy: np.ndarray,
) -> np.ndarray:
    """Compute p_total = β·p_sw + α·exp(iψ)·p_v(center)."""
    p_v = vortex_gen.get_field(center_xy)
    return beta * p_sw + alpha * np.exp(1j * psi) * p_v


def gorkov_from_pressure(p: np.ndarray, dx: float, dy: float):
    """Gor'kov potential and force from total pressure."""
    return gorkov_grid_2d(
        p, dx, dy, OMEGA, RHO0, C_WATER,
        PPAR["a"], PPAR["f1"], PPAR["f2"],
    )


# ════════════════════════════════════════════════════════════════
# Gradient check
# ════════════════════════════════════════════════════════════════

def gradient_check(
    feval: ForceEvaluator,
    u0: np.ndarray,
    pos: np.ndarray,
    idx_A: int,
    idx_B: int,
    neigh_idx: np.ndarray,
    target_pos: np.ndarray,
    cfg: MPCConfig,
) -> bool:
    """
    Verify analytic MPC gradients match centered finite differences.

    Returns True if all relative errors < 5%.
    """
    from scripts.lib.mpc_controller import (
        compute_mpc_gradients,
        compute_total_cost,
        rollout,
    )

    controls = [u0.copy() for _ in range(cfg.K)]
    state = rollout(feval, controls, pos, cfg.dt)
    grads = compute_mpc_gradients(
        feval, state, idx_A, idx_B, neigh_idx, target_pos, cfg,
    )

    print("\n  Gradient check (analytic vs centered FD):")
    names = ["psi", "x_v", "y_v", "alpha", "beta"]
    eps_vals = [1e-4, 5e-6, 5e-6, 1e-4, 1e-4]
    all_ok = True

    # Check gradient for step t=0
    for dim in range(N_CTRL):
        eps = eps_vals[dim]
        controls_p = [u.copy() for u in controls]
        controls_m = [u.copy() for u in controls]
        controls_p[0][dim] += eps
        controls_m[0][dim] -= eps

        J_p = compute_total_cost(
            feval, controls_p, pos, idx_A, idx_B, neigh_idx, target_pos, cfg,
        )
        J_m = compute_total_cost(
            feval, controls_m, pos, idx_A, idx_B, neigh_idx, target_pos, cfg,
        )
        fd_grad = (J_p - J_m) / (2.0 * eps)
        analytic = grads[0][dim]
        rel_err = abs(analytic - fd_grad) / max(abs(fd_grad), 1e-30)
        ok = rel_err < 0.05 or abs(fd_grad) < 1e-20
        if not ok:
            all_ok = False
        print(f"    {names[dim]:>6s}: analytic={analytic:+.6e}  fd={fd_grad:+.6e}  "
              f"rel_err={rel_err:.4f}  {'OK' if ok else 'FAIL'}")

    return all_ok


# ════════════════════════════════════════════════════════════════
# GIF rendering
# ════════════════════════════════════════════════════════════════

def render_gif(
    result: MPCResult,
    p_sw: np.ndarray,
    vortex_gen: VortexPerturbation,
    xg: np.ndarray,
    yg: np.ndarray,
    traps_m: np.ndarray,
    idx_A: int,
    idx_B: int,
    neigh_idx: np.ndarray,
    out_path: Path,
    n_frames: int = N_GIF_FRAMES,
    duration_ms: int = GIF_DURATION_MS,
    open_loop_steps: int = 0,
) -> None:
    """Render transport GIF matching the C-shape study style."""
    dx = float(xg[1] - xg[0])
    dy = float(yg[1] - yg[0])
    x_mm = xg * 1e3
    y_mm = yg * 1e3
    extent = [x_mm[0], x_mm[-1], y_mm[0], y_mm[-1]]
    traps_mm = traps_m * 1e3

    # Adaptive view: fit all particle positions + some margin
    all_pos_mm = np.array([p * 1e3 for p in result.positions])
    pos_xmin = float(np.min(all_pos_mm[:, :, 0]))
    pos_xmax = float(np.max(all_pos_mm[:, :, 0]))
    pos_ymin = float(np.min(all_pos_mm[:, :, 1]))
    pos_ymax = float(np.max(all_pos_mm[:, :, 1]))
    margin_mm = 0.3
    view_xlim = (pos_xmin - margin_mm, pos_xmax + margin_mm)
    view_ylim = (pos_ymin - margin_mm, pos_ymax + margin_mm)

    T_total = len(result.applied_controls)
    frame_every = max(1, T_total // n_frames)
    frame_indices = list(range(0, T_total, frame_every))
    if frame_indices[-1] != T_total - 1:
        frame_indices.append(T_total - 1)

    frames: List[Image.Image] = []
    print(f"  Rendering {len(frame_indices)} GIF frames...")

    for fi, step in enumerate(frame_indices):
        u = result.applied_controls[step]
        psi, xv, yv, alpha, beta = u

        p_total = total_pressure(
            p_sw, vortex_gen, psi, alpha, beta, np.array([xv, yv]),
        )
        U, _, _ = gorkov_from_pressure(p_total, dx, dy)

        # Clip colorscale within the visible 2mm×2mm window so SW traps are visible.
        ix_lo = int(np.searchsorted(xg, (view_xlim[0] - 0.05) * 1e-3))
        ix_hi = int(np.searchsorted(xg, (view_xlim[1] + 0.05) * 1e-3))
        iy_lo = int(np.searchsorted(yg, (view_ylim[0] - 0.05) * 1e-3))
        iy_hi = int(np.searchsorted(yg, (view_ylim[1] + 0.05) * 1e-3))
        U_roi = U[iy_lo:iy_hi, ix_lo:ix_hi]
        lo = float(np.percentile(U_roi, GORKOV_CLIP_LO))
        hi = float(np.percentile(U_roi, GORKOV_CLIP_HI))

        pos_mm = result.positions[step] * 1e3
        t_ms = step * result.elapsed_s / max(T_total, 1) * 1e3 if result.elapsed_s > 0 else step * 0.1

        # Vortex centre trail — only last 30 steps to avoid visual clutter
        trail_start = max(0, step - 30)
        centers = np.array([[c[I_XV], c[I_YV]] for c in result.applied_controls[trail_start:step + 1]]) * 1e3

        fig, ax = plt.subplots(1, 1, figsize=(6.5, 6.0))
        ax.imshow(
            U, origin="lower", extent=extent, cmap=CMAP,
            vmin=lo, vmax=hi, aspect="equal", interpolation="bicubic",
        )

        # Home positions
        ax.scatter(traps_mm[:, 0], traps_mm[:, 1], marker="x", c=COL_HOME,
                   s=28, linewidths=0.8, zorder=4)

        # Neighbour tethers
        for i in neigh_idx:
            ax.plot([traps_mm[i, 0], pos_mm[i, 0]],
                    [traps_mm[i, 1], pos_mm[i, 1]],
                    color=COL_NEIGH, lw=0.8, alpha=0.65, zorder=3)

        # A and B tethers
        for i, col in [(idx_A, COL_A), (idx_B, COL_B)]:
            ax.plot([traps_mm[i, 0], pos_mm[i, 0]],
                    [traps_mm[i, 1], pos_mm[i, 1]],
                    color=col, lw=1.2, alpha=0.9, zorder=3)

        # Particles
        ax.scatter(pos_mm[neigh_idx, 0], pos_mm[neigh_idx, 1],
                   c=COL_NEIGH, s=28, zorder=6)
        ax.scatter([pos_mm[idx_B, 0]], [pos_mm[idx_B, 1]],
                   c=COL_B, s=44, zorder=7)
        ax.scatter([pos_mm[idx_A, 0]], [pos_mm[idx_A, 1]],
                   c=COL_A, s=44, zorder=7)

        # Vortex centre trail
        if len(centers) > 1:
            ax.plot(centers[:, 0], centers[:, 1], color=COL_CTR,
                    lw=0.9, alpha=0.75, zorder=5)
        ax.plot(centers[-1, 0], centers[-1, 1], marker="+",
                color=COL_CTR, ms=10, mew=1.5, zorder=8)

        ax.set_xlim(*view_xlim)
        ax.set_ylim(*view_ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")

        d_AB = float(np.linalg.norm(
            result.positions[step][idx_A] - result.positions[step][idx_B]
        ) * 1e6)
        # Phase label
        if step < open_loop_steps:
            phase_label = "OPEN-LOOP"
        elif step < open_loop_steps + (cfg_T if 'cfg_T' in dir() else 500):
            phase_label = "MPC"
        else:
            phase_label = "SETTLE"
        # Determine phase from step count vs open_loop_steps
        if step < open_loop_steps:
            phase_label = "OPEN-LOOP"
        else:
            phase_label = "MPC+SETTLE"
        ax.set_title(
            f"{phase_label} — step {step}/{T_total} | "
            f"d(A,B)={d_AB:.0f} µm\n"
            f"α={alpha:.2f}  β={beta:.2f}  ψ={psi:.2f} rad",
            fontsize=9,
        )

        fig.tight_layout()
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        frames.append(Image.fromarray(buf.reshape(h, w, 4)).convert("RGB"))
        plt.close(fig)

        if (fi + 1) % 50 == 0:
            print(f"    {fi + 1}/{len(frame_indices)}")

    frames[0].save(
        str(out_path),
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    print(f"  Saved GIF: {out_path}")


# ════════════════════════════════════════════════════════════════
# Convergence plot
# ════════════════════════════════════════════════════════════════

def plot_convergence(result: MPCResult, out_path: Path) -> None:
    """Plot MPC outer-loop cost and inner convergence."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Outer loop
    ax = axes[0]
    ax.plot(result.J_history, ".-")
    ax.set_xlabel("Replan step")
    ax.set_ylabel("Horizon cost J")
    ax.set_title("MPC outer-loop cost")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # Sample inner loops
    ax = axes[1]
    n_show = min(8, len(result.inner_J_histories))
    step = max(1, len(result.inner_J_histories) // n_show)
    for i in range(0, len(result.inner_J_histories), step):
        hist = result.inner_J_histories[i]
        if len(hist) > 1:
            ax.plot(hist, label=f"replan {i}")
    ax.set_xlabel("Inner iteration")
    ax.set_ylabel("J")
    ax.set_title("Inner-loop convergence (sampled)")
    ax.set_yscale("log")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"  Saved convergence plot: {out_path}")


# ════════════════════════════════════════════════════════════════
# Control & trajectory CSV
# ════════════════════════════════════════════════════════════════

def save_controls_csv(result: MPCResult, dt: float, path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "t_ms", "psi", "x_v", "y_v", "alpha", "beta"])
        for i, u in enumerate(result.applied_controls):
            w.writerow([i, i * dt * 1e3, *u])


def save_trajectory_csv(result: MPCResult, idx_A: int, idx_B: int, dt: float, path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "t_ms", "xA_mm", "yA_mm", "xB_mm", "yB_mm", "d_AB_um"])
        for i, pos in enumerate(result.positions):
            xA, yA = pos[idx_A] * 1e3
            xB, yB = pos[idx_B] * 1e3
            d = float(np.linalg.norm(pos[idx_A] - pos[idx_B]) * 1e6)
            w.writerow([i, i * dt * 1e3, xA, yA, xB, yB, d])


# ════════════════════════════════════════════════════════════════
# Settling phase — ramp α→0, let standing wave recapture
# ════════════════════════════════════════════════════════════════

def run_settling_phase(
    feval: ForceEvaluator,
    result: MPCResult,
    cfg: MPCConfig,
    idx_A: int,
    idx_B: int,
    T_settle: int = T_SETTLE_DEFAULT,
) -> MPCResult:
    """
    Append a settling phase to an MPC result.

    After the MPC transport phase:
      1. Rapidly ramp α → 0 over 50 steps (5 ms) to remove vortex influence.
      2. Hold α = 0, β = 1.0 for the remaining steps, letting the standing
         wave recapture all particles into their nearest trap nodes.

    If A has been brought within λ/4 of B's node, both settle into
    the same node → merge.

    The function mutates *result* in-place (appending to lists) and returns it.
    """
    step_scale = SCALE * cfg.dt

    pos = result.positions[-1].copy()
    u_final = result.applied_controls[-1].copy()
    alpha0 = u_final[I_ALPHA]
    beta0 = u_final[I_BETA]

    N_RAMP = min(50, T_settle)  # ramp α → 0 in 50 steps (5ms)

    print(f"\n  Settling phase: {T_settle} steps "
          f"(ramp {N_RAMP} + relax {T_settle - N_RAMP}), "
          f"α {alpha0:.3f}→0, β {beta0:.3f}→1.0")

    for k in range(T_settle):
        u = u_final.copy()

        if k < N_RAMP:
            # Quick ramp: α → 0, β → 1.0
            frac = (k + 1) / N_RAMP
            u[I_ALPHA] = alpha0 * (1.0 - frac)
            u[I_BETA] = beta0 + (1.0 - beta0) * frac
        else:
            # Pure SW relaxation
            u[I_ALPHA] = 0.0
            u[I_BETA] = 1.0

        u = np.clip(u, cfg.u_lo, cfg.u_hi)

        result.applied_controls.append(u.copy())
        Fx, Fy = feval.forces(u, pos)
        result.forces.append((Fx.copy(), Fy.copy()))
        pos = pos.copy()
        pos[:, 0] += step_scale * Fx
        pos[:, 1] += step_scale * Fy
        result.positions.append(pos.copy())

        # Check merge
        if result.merge_time_s is None:
            d_AB = float(np.linalg.norm(pos[idx_A] - pos[idx_B]))
            if d_AB < CAPTURE_RADIUS:
                result.merge_time_s = (cfg.T + k + 1) * cfg.dt

    d_final = float(np.linalg.norm(pos[idx_A] - pos[idx_B]) * 1e6)
    print(f"  Settle done. d(A,B) = {d_final:.1f} µm, "
          f"α_final = {result.applied_controls[-1][I_ALPHA]:.4f}")
    return result


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="MPC Vortex Merge")
    parser.add_argument("--K", type=int, default=10, help="MPC horizon")
    parser.add_argument("--T", type=int, default=3000, help="Total steps")
    parser.add_argument("--n_iters", type=int, default=10, help="Inner optimiser iterations")
    parser.add_argument("--replan_every", type=int, default=5, help="Replan frequency")
    parser.add_argument("--w_merge", type=float, default=1e12)
    parser.add_argument("--w_B_stable", type=float, default=1e12)
    parser.add_argument("--w_neigh", type=float, default=5e11)
    parser.add_argument("--w_smooth", type=float, default=1e8)
    parser.add_argument("--waist", type=float, default=VORTEX_WAIST * 1e3, help="Vortex waist [mm] (LG only)")
    parser.add_argument("--charge", type=int, default=VORTEX_CHARGE)
    parser.add_argument("--alpha_max", type=float, default=3.0, help="Max vortex amplitude")
    parser.add_argument("--alpha_init", type=float, default=3.0, help="Initial vortex amplitude (MPC phase)")
    parser.add_argument("--n_gif_frames", type=int, default=N_GIF_FRAMES)
    parser.add_argument("--gradcheck", action="store_true", help="Run gradient check only")
    parser.add_argument("--n_particles", type=int, default=9, help="Number of particles (2-9); default all")
    parser.add_argument("--T_settle", type=int, default=T_SETTLE_DEFAULT,
                        help="Settling-phase steps (α→0 ramp-down after MPC)")
    # ── New: vortex source selection ──
    parser.add_argument("--vortex_source", type=str, default="bessel",
                        choices=["lg", "bessel"],
                        help="Vortex field model: 'bessel' (Bessel from finite aperture, "
                             "default) or 'lg' (Laguerre-Gaussian)")
    parser.add_argument("--bessel_aperture", type=float, default=BESSEL_APERTURE_RADIUS * 1e3,
                        help="Bessel transducer aperture radius [mm]")
    parser.add_argument("--bessel_prop_dist", type=float, default=BESSEL_PROP_DIST * 1e3,
                        help="Bessel propagation distance [mm]")
    # ── New: remote particle + open-loop ──
    parser.add_argument("--remote_distance", type=float, default=REMOTE_DISTANCE_LAM,
                        help="Remote particle distance from cluster edge [λ]")
    parser.add_argument("--mpc_activation_radius", type=float, default=MPC_ACTIVATION_RADIUS_LAM,
                        help="Distance from cluster at which MPC activates [λ]")
    parser.add_argument("--open_loop_speed", type=float, default=OPEN_LOOP_SPEED * 1e3,
                        help="Open-loop vortex cruise speed [mm/s]")
    parser.add_argument("--ol_alpha", type=float, default=3.0,
                        help="Vortex amplitude α during open-loop (strong capture)")
    parser.add_argument("--ol_beta", type=float, default=1.0,
                        help="Standing-wave amplitude β during open-loop (full SW visible)")
    # ── New: per-step rate limits ──
    parser.add_argument("--vxy_rate", type=float, default=0.5,
                        help="MPC vortex position rate limit [µm/step]")
    parser.add_argument("--psi_rate", type=float, default=2.0 * np.pi,
                        help="MPC ψ rate limit [rad/step] (default 2π = free)")
    args = parser.parse_args()

    # ── Load data ─────────────────────────────────────────────────
    print("Loading standing-wave and trap data...")
    ov = np.load(OVERLAY_NPZ)
    traps_m_all = ov["traps_m"].astype(float)
    # Override: A = "top middle" (trap 8), B = "middle" (trap 4)
    idx_A_orig = 8   # top centre
    idx_B_orig = 4   # middle centre

    vd = np.load(VORTEX_NPZ)
    xg = vd["xg"].astype(float)
    yg = vd["yg"].astype(float)
    p_sw = vd["p_sw"].astype(complex)
    p_sw_peak = float(np.max(np.abs(p_sw)))
    print(f"  p_sw max amplitude: {p_sw_peak:.2f}")

    dx = float(xg[1] - xg[0])
    dy = float(yg[1] - yg[0])

    # ── Select particles (A = top-middle, B = middle) ─────────────
    B_xy = traps_m_all[idx_B_orig]
    A_orig_xy = traps_m_all[idx_A_orig]

    # All particles stay in the cluster; A starts at its actual trap.
    traps_m_cluster = traps_m_all.copy()
    init_pos = traps_m_cluster.copy()           # everyone at home

    # Target positions: A merges into B's trap, others stay home.
    traps_m = traps_m_cluster.copy()
    traps_m[idx_A_orig] = B_xy.copy()

    idx_A = idx_A_orig
    idx_B = idx_B_orig
    A_remote_xy = A_orig_xy.copy()              # kept for open-loop start
    neigh_idx = np.array([i for i in range(len(traps_m)) if i not in (idx_A, idx_B)], dtype=int)

    print(f"Particles: {len(traps_m)} (A={idx_A} [top-middle], B={idx_B} [middle], "
          f"neighbours={neigh_idx.tolist()})")
    print(f"  A (start):  {A_remote_xy*1e3} mm")
    print(f"  B (target): {B_xy*1e3} mm")
    d_AB_init = np.linalg.norm(A_remote_xy - B_xy)
    print(f"  d(A,B) = {d_AB_init*1e6:.1f} µm = {d_AB_init/LAM:.1f} λ")

    # ── Generate vortex field ─────────────────────────────────────
    grid_cx = 0.5 * (float(xg[0]) + float(xg[-1]))
    grid_cy = 0.5 * (float(yg[0]) + float(yg[-1]))
    XX, YY = np.meshgrid(xg, yg)

    if args.vortex_source == "bessel":
        # First-order Bessel vortex: uniform source + spiral phase, ASM-propagated
        bessel_R = args.bessel_aperture * 1e-3  # mm → m
        bessel_z = args.bessel_prop_dist * 1e-3
        p_source = make_bessel_vortex_field(
            XX, YY,
            charge=args.charge,
            aperture_radius=bessel_R,
            k=ASM_K0,
            center=(grid_cx, grid_cy),
            apodization="cosine_taper",
        )
        print(f"  Bessel source: R={bessel_R*1e3:.1f} mm, z_prop={bessel_z*1e3:.1f} mm")
        p_vortex_raw = propagate_asm(p_source, dx, dy, wavelength=ASM_LAM, z=bessel_z)
        vortex_family = "bessel_asm"
    else:
        # LG fallback
        vortex_waist_m = args.waist * 1e-3
        p_vortex_raw = make_vortex_field(
            XX, YY,
            charge=args.charge,
            waist=vortex_waist_m,
            center=(grid_cx, grid_cy),
        )
        vortex_family = "LG"

    # Scale amplitude to match p_sw peak so α=1.0 gives comparable force.
    p_vortex_peak = float(np.max(np.abs(p_vortex_raw)))
    if p_vortex_peak > 0:
        p_vortex_raw = p_vortex_raw * (p_sw_peak / p_vortex_peak)
    print(f"  Vortex ({vortex_family}): charge={args.charge}, "
          f"peak={float(np.max(np.abs(p_vortex_raw))):.2f} Pa")

    vortex_gen = VortexPerturbation(
        p_vortex_raw, xg, yg, out_xg=xg, out_yg=yg,
    )

    # ── Configure MPC ─────────────────────────────────────────────
    corridor_margin = 2.0 * TRAP_SP
    # Bounds span the full transport corridor: A_remote → B
    all_positions = np.vstack([A_remote_xy[np.newaxis, :], traps_m])

    cfg = MPCConfig(
        K=args.K,
        T=args.T,
        n_iters=args.n_iters,
        replan_every=args.replan_every,
        w_merge=args.w_merge,
        w_B_stable=args.w_B_stable,
        w_neigh=args.w_neigh,
        w_smooth=args.w_smooth,
        w_beta_high=1e4,
    )
    # Per-DOF smoothness: ψ effectively free, xy/α/β smooth
    cfg.w_smooth_vec = np.array([
        1.0,                   # ψ: minimal penalty (effectively free)
        args.w_smooth,         # x_v: full smoothness
        args.w_smooth,         # y_v: full smoothness
        args.w_smooth * 0.1,   # α: moderate smoothness
        args.w_smooth * 0.1,   # β: moderate smoothness
    ])

    cfg.u_lo = np.array([
        0.0,
        float(np.min(all_positions[:, 0])) - corridor_margin,
        float(np.min(all_positions[:, 1])) - corridor_margin,
        0.0,
        0.2,                          # β lower bound (must allow open-loop β)
    ])
    cfg.u_hi = np.array([
        2.0 * np.pi,
        float(np.max(all_positions[:, 0])) + corridor_margin,
        float(np.max(all_positions[:, 1])) + corridor_margin,
        args.alpha_max,
        1.0,
    ])

    # Per-step rate limits:
    #   ψ:   large (effectively free — optimizer can jump anywhere in [0, 2π])
    #   x,y: slow (smooth vortex translation, default 0.5 µm/step = 5 mm/s)
    #   α:   ±0.005/step (ramp 0→2 over 400 steps = 40ms)
    #   β:   ±0.005/step
    cfg.du_max = np.array([
        args.psi_rate,              # ψ: free (2π/step by default)
        args.vxy_rate * 1e-6,       # x_v:  µm/step → m/step
        args.vxy_rate * 1e-6,       # y_v
        0.005,                      # α
        0.005,                      # β
    ])

    feval = ForceEvaluator(p_sw, vortex_gen, xg, yg)

    # ── Gradient check ────────────────────────────────────────────
    if args.gradcheck:
        u0_check = np.array([0.0, A_remote_xy[0], A_remote_xy[1], args.alpha_init, 0.9])
        u0_check = np.clip(u0_check, cfg.u_lo, cfg.u_hi)
        ok = gradient_check(
            feval, u0_check, init_pos.copy(), idx_A, idx_B, neigh_idx,
            traps_m.copy(), cfg,
        )
        print(f"\n  Gradient check {'PASSED' if ok else 'FAILED'}")
        return

    # ════════════════════════════════════════════════════════════════
    #  Phase I — Open-loop pickup: move vortex from A_remote toward cluster
    #  Tethered approach: vortex only advances when particle A follows.
    # ════════════════════════════════════════════════════════════════
    open_loop_speed = args.open_loop_speed * 1e-3  # mm/s → m/s
    mpc_activation_dist = args.mpc_activation_radius * LAM
    LEASH = 0.3 * LAM   # max distance between vortex centre and particle A

    # Vector from A_remote toward B
    transport_vec = traps_m[idx_B] - A_remote_xy
    transport_dist = float(np.linalg.norm(transport_vec))
    transport_dir = transport_vec / max(transport_dist, 1e-12)

    # Stop open-loop when vortex center is mpc_activation_dist from B
    open_loop_travel = max(0.0, transport_dist - mpc_activation_dist)
    max_ol_steps = int(open_loop_travel / (open_loop_speed * cfg.dt)) + 500  # safety margin

    print(f"\n  Phase I — Open-loop pickup (tethered):")
    print(f"    Travel: {open_loop_travel*1e3:.2f} mm at max {open_loop_speed*1e3:.1f} mm/s")
    print(f"    Leash: {LEASH*1e6:.0f} µm ({LEASH/LAM:.2f} λ)")
    print(f"    MPC activates at {mpc_activation_dist*1e3:.2f} mm from B")

    # Tethered open-loop: vortex advances at max speed but waits for particle A.
    step_scale = SCALE * cfg.dt
    pos = init_pos.copy()
    ol_positions = [pos.copy()]
    ol_controls = []
    ol_forces = []

    vortex_pos = A_remote_xy.copy()
    max_advance = open_loop_speed * cfg.dt  # max vortex displacement per step [m]

    for k in range(max_ol_steps):
        # Desired vortex position: advance toward B
        desired = vortex_pos + transport_dir * max_advance
        # Progress along transport axis
        progress = float(np.dot(desired - A_remote_xy, transport_dir))
        if progress >= open_loop_travel:
            desired = A_remote_xy + transport_dir * open_loop_travel
            at_target = True
        else:
            at_target = False

        # Leash: clamp vortex so it stays within LEASH of particle A
        to_desired = desired - pos[idx_A]
        d_leash = float(np.linalg.norm(to_desired))
        if d_leash > LEASH:
            # Don't advance — hold vortex at leash boundary from A
            vortex_pos = pos[idx_A] + to_desired / d_leash * LEASH
        else:
            vortex_pos = desired.copy()

        u_ol = np.array([
            0.0,                        # psi = 0  (open-loop)
            vortex_pos[0],
            vortex_pos[1],
            args.ol_alpha,              # strong vortex for capture
            args.ol_beta,               # weak SW to release from trap
        ])
        u_ol = np.clip(u_ol, cfg.u_lo, cfg.u_hi)
        ol_controls.append(u_ol.copy())

        # Use fast spline-based force evaluation (no full-grid basis rebuild)
        Fx, Fy = feval._fast_forces_at_pts(
            pos, u_ol[1], u_ol[2], u_ol[3], u_ol[4], u_ol[0]
        )
        ol_forces.append((Fx.copy(), Fy.copy()))
        pos = pos.copy()
        pos[:, 0] += step_scale * Fx
        pos[:, 1] += step_scale * Fy
        ol_positions.append(pos.copy())

        # Check completion: vortex reached activation zone
        d_to_B = float(np.linalg.norm(vortex_pos - traps_m[idx_B]))
        if at_target or d_to_B <= mpc_activation_dist:
            break

    open_loop_steps = len(ol_controls)
    open_loop_time = open_loop_steps * cfg.dt

    # Position of vortex at end of open-loop
    if open_loop_steps > 0:
        vortex_handoff = ol_controls[-1][[I_XV, I_YV]]
    else:
        vortex_handoff = A_remote_xy.copy()

    d_AB_post_ol = float(np.linalg.norm(pos[idx_A] - pos[idx_B]) * 1e6)
    print(f"    Completed in {open_loop_steps} steps ({open_loop_time*1e3:.1f} ms)")
    print(f"    After open-loop: d(A,B) = {d_AB_post_ol:.1f} µm")

    # ════════════════════════════════════════════════════════════════
    #  Phase II — MPC-controlled approach
    # ════════════════════════════════════════════════════════════════
    u0_mpc = np.array([
        0.0,                         # ψ: start at 0, optimizer is free to explore
        vortex_handoff[0],
        vortex_handoff[1],
        args.alpha_init,
        1.0,                         # β = 1.0 during MPC (SW visible)
    ])
    u0_mpc = np.clip(u0_mpc, cfg.u_lo, cfg.u_hi)

    print(f"\n  Phase II — MPC approach (T={cfg.T}, K={cfg.K})...")
    mpc_result = run_mpc(
        feval=feval,
        x0=pos.copy(),
        u_init=u0_mpc,
        idx_A=idx_A,
        idx_B=idx_B,
        neigh_idx=neigh_idx,
        target_pos=traps_m.copy(),
        cfg=cfg,
        verbose=True,
    )

    # ════════════════════════════════════════════════════════════════
    #  Phase III — Settling (α→0 ramp, SW recapture)
    # ════════════════════════════════════════════════════════════════
    if args.T_settle > 0:
        mpc_result = run_settling_phase(
            feval, mpc_result, cfg, idx_A, idx_B, T_settle=args.T_settle,
        )

    # ── Merge open-loop + MPC results into a single MPCResult ─────
    all_positions = ol_positions[:-1] + mpc_result.positions  # avoid duplicate handoff frame
    all_controls = ol_controls + mpc_result.applied_controls
    all_forces = ol_forces + mpc_result.forces
    # Offset merge time to account for open-loop duration
    ol_duration = open_loop_steps * cfg.dt
    merge_time = None
    if mpc_result.merge_time_s is not None:
        merge_time = mpc_result.merge_time_s + ol_duration
    else:
        # Check if merge happened during open-loop
        for k in range(len(ol_positions)):
            d = float(np.linalg.norm(ol_positions[k][idx_A] - ol_positions[k][idx_B]))
            if d < CAPTURE_RADIUS:
                merge_time = k * cfg.dt
                break

    combined_result = MPCResult(
        positions=all_positions,
        applied_controls=all_controls,
        forces=all_forces,
        J_history=mpc_result.J_history,
        inner_J_histories=mpc_result.inner_J_histories,
        elapsed_s=mpc_result.elapsed_s,
        merge_time_s=merge_time,
    )

    # ── Metrics ───────────────────────────────────────────────────
    transport = mpc_result_to_transport(combined_result, traps_m, cfg.dt)
    metrics = compute_metrics(transport, idx_A, idx_B, neigh_idx)
    metrics["open_loop_steps"] = open_loop_steps
    metrics["open_loop_duration_ms"] = round(ol_duration * 1e3, 1)
    print(f"\nTransport metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # ── Output directory ──────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "results" / "dev" / "mpc_vortex_merge" / f"run_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Save outputs ──────────────────────────────────────────────
    save_controls_csv(combined_result, cfg.dt, out_dir / "controls.csv")
    save_trajectory_csv(combined_result, idx_A, idx_B, cfg.dt, out_dir / "trajectory.csv")
    plot_convergence(mpc_result, out_dir / "convergence.png")

    render_gif(
        combined_result, p_sw, vortex_gen, xg, yg, traps_m,
        idx_A, idx_B, neigh_idx,
        out_dir / "mpc_vortex_merge.gif",
        n_frames=args.n_gif_frames,
        open_loop_steps=open_loop_steps,
    )

    # ── Manifest ──────────────────────────────────────────────────
    manifest = {
        "method_id": "mpc_vortex_merge_v2",
        "script": "scripts/dev/mpc_vortex_merge.py",
        "timestamp": timestamp,
        "output_dir": str(out_dir.relative_to(PROJECT_ROOT)),
        "mpc_config": {
            "K": cfg.K,
            "T": cfg.T,
            "dt": cfg.dt,
            "replan_every": cfg.replan_every,
            "n_iters": cfg.n_iters,
            "w_merge": cfg.w_merge,
            "w_B_stable": cfg.w_B_stable,
            "w_neigh": cfg.w_neigh,
            "w_smooth": cfg.w_smooth,
            "w_smooth_vec": cfg.w_smooth_vec.tolist() if cfg.w_smooth_vec is not None else None,
            "w_beta_high": cfg.w_beta_high,
            "beta_terminal": cfg.beta_terminal,
            "u_lo": cfg.u_lo.tolist(),
            "u_hi": cfg.u_hi.tolist(),
            "du_max": cfg.du_max.tolist() if cfg.du_max is not None else None,
            "T_settle": args.T_settle,
        },
        "vortex_design": {
            "charge": args.charge,
            "family": vortex_family,
            "bessel_aperture_mm": args.bessel_aperture if args.vortex_source == "bessel" else None,
            "bessel_prop_dist_mm": args.bessel_prop_dist if args.vortex_source == "bessel" else None,
            "waist_mm": args.waist if args.vortex_source == "lg" else None,
        },
        "scenario": {
            "remote_distance_lam": args.remote_distance,
            "mpc_activation_radius_lam": args.mpc_activation_radius,
            "open_loop_speed_mm_s": args.open_loop_speed,
            "ol_alpha": args.ol_alpha,
            "ol_beta": args.ol_beta,
            "open_loop_steps": open_loop_steps,
            "open_loop_duration_ms": round(ol_duration * 1e3, 1),
            "vxy_rate_um_per_step": args.vxy_rate,
            "psi_rate_rad_per_step": args.psi_rate,
        },
        "particles": {
            "n_total": len(traps_m),
            "idx_A": idx_A,
            "idx_B": idx_B,
            "n_neighbours": len(neigh_idx),
            "d_AB_initial_um": round(d_AB_init * 1e6, 2),
        },
        "metrics": metrics,
        "performance": {
            "elapsed_s": round(combined_result.elapsed_s, 1),
            "n_replans": len(combined_result.J_history),
        },
    }

    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    print(f"\nAll outputs saved to: {out_dir}")
    print(f"Classification: {metrics.get('classification', 'unknown')}")


if __name__ == "__main__":
    main()
