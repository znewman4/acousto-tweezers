#!/usr/bin/env python3
"""
C-Shape Particle Merge GIF Demo
=================================

Demonstrates 9 particles initially sitting in 9 standing-wave traps,
then activating a crossfade transport protocol: the standing wave is
smoothly faded out while the C-shape 15 mm lens field ramps in,
sweeping particle A toward particle B's trap.  On restoration of the
standing wave A settles into B's basin — a successful merge.

Model:  p_comb(t) = β_sw(t) × p_sw + β_lens(t) × exp(iψ) × p_lens

Uses:
  - cached standing-wave + lens fields from overlay study
  - Gor'kov potential computed from first principles (same formula as repo)
  - overdamped particle dynamics: dx/dt = mu * F, F = -grad(U)

Outputs → results/c_shape_particle_merge_demo_<TS>/
"""
from __future__ import annotations

import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.ndimage import minimum_filter, uniform_filter
from scipy.interpolate import RegularGridInterpolator

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as mpatches

# ── Project root ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ═══════════════════════════════════════════════════════════════════
# Physical constants
# ═══════════════════════════════════════════════════════════════════
C_WATER = 1484.0          # m/s
F_HZ    = 2.0e6           # Hz
OMEGA   = 2.0 * np.pi * F_HZ
K_WATER = OMEGA / C_WATER
LAM     = C_WATER / F_HZ  # 0.742 mm
TRAP_SP = LAM / 2.0       # ~0.371 mm
RHO0    = 997.0            # kg/m³  water density

# Particle properties — polystyrene microsphere
RHO_P   = 1050.0          # kg/m³
C_P     = 2350.0          # m/s
A_PART  = 50.0e-6         # particle radius 50 µm (100 µm diameter)
ETA     = 1.0e-3          # dynamic viscosity of water (Pa·s)

# Gor'kov small-particle validity: ka ≪ 1
_KA = K_WATER * A_PART
assert _KA < 0.5, f"ka = {_KA:.3f} exceeds small-particle limit 0.5"
print(f"[validity] ka = {_KA:.3f}  (Gor'kov valid for ka < 0.5)")

# Stokes mobility  µ = 1/(6πηa)
MU_STOKES = 1.0 / (6.0 * np.pi * ETA * A_PART)

# Gor'kov particle-volume prefactor
# The grid-level Gor'kov formula (used in all repo scripts) omits the
# time-averaging 1/2 and the particle volume (4π/3)a³, so:
#   U_real = (V_p / 2) × U_code = (2π/3) a³ × U_code
# The force must carry this prefactor.
GORKOV_PREFACTOR = (2.0 * np.pi / 3.0) * A_PART**3

# Gor'kov contrast factors
KAPPA_W = 1.0 / (RHO0 * C_WATER**2)
KAPPA_P = 1.0 / (RHO_P * C_P**2)
F1 = 1.0 - KAPPA_P / KAPPA_W
F2 = 2.0 * (RHO_P - RHO0) / (2.0 * RHO_P + RHO0)

# ═══════════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════════
OVERLAY_NPZ = (PROJECT_ROOT / "results"
               / "c_shape_lens_15mm_overlay_study_20260310_170620"
               / "npz" / "roi_fields.npz")

# ═══════════════════════════════════════════════════════════════════
# Pressure scaling
# ═══════════════════════════════════════════════════════════════════
# The FEM standing wave has |p|_max ~ 46 Pa (unit-excitation).  Real
# acoustic tweezers operate at ~100 kPa–1 MPa.  We scale all pressures
# by P_SCALE so that forces are physically realistic.
#   U_real = P_SCALE² × U_code × GORKOV_PREFACTOR
#   F_real = P_SCALE² × F_code × GORKOV_PREFACTOR
P_SCALE = 3.0e3  # → |p_sw| ≈ 140 kPa  (typical piezo transducer)

# ═══════════════════════════════════════════════════════════════════
# Control schedule parameters
# ═══════════════════════════════════════════════════════════════════
# Crossfade mechanism: smoothly transition the pressure field from
# the pure standing wave to α × exp(iψ) × p_lens,  then back.
# β_sw(t) ramps 1 → 0 while β_lens(t) ramps 0 → α.  At β_sw = 0
# the λ/2 traps vanish; the lens potential sweeps A toward B.
# On restoration of the SW, A settles into B's basin.
# ψ = 3π/2 directs lens force A → B (verified by phase sweep).
ALPHA_TARGET = 5.0
PSI          = 3.0 * np.pi / 2.0

# Time schedule: crossfade-up → hold → crossfade-down → settle
# Ramp slowly for overdamped dynamics to track smoothly.
T_RAMP_UP    = 0.200      # 200 ms — crossfade SW→lens
T_HOLD       = 0.100      # 100 ms — let A reach B under lens field
T_RAMP_DOWN  = 0.200      # 200 ms — crossfade lens→SW
T_SETTLE     = 0.300      # 300 ms — final equilibration in SW traps

DT           = 1.0e-4     # 0.1 ms timestep
N_FRAMES     = 150        # GIF frames

# Success criteria
CAPTURE_RADIUS = 0.30 * TRAP_SP   # A within this of B → merge
TRAP_TOLERANCE = 0.50 * TRAP_SP   # B must stay within this of its initial pos
NEIGHBOUR_TOL  = 0.50 * TRAP_SP   # neighbours must stay within this

# ═══════════════════════════════════════════════════════════════════
# Output
# ═══════════════════════════════════════════════════════════════════
TS = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = PROJECT_ROOT / "results" / f"c_shape_particle_merge_demo_{TS}"
FIG_DIR = OUT_DIR / "figures"
NPZ_DIR = OUT_DIR / "npz"
for d in [OUT_DIR, FIG_DIR, NPZ_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════

def load_fields():
    """Load pre-computed ROI fields from overlay study."""
    d = np.load(OVERLAY_NPZ, allow_pickle=True)
    p_sw       = d["p_sw"]          # (400, 400) complex
    p_lens_roi = d["p_lens_roi"]    # (400, 400) complex
    xg         = d["xg"]            # (400,)
    yg         = d["yg"]            # (400,)
    traps_m    = d["traps_m"]       # (9, 2)
    idx_A      = int(d["idx_A"])
    idx_B      = int(d["idx_B"])
    midpoint   = d["midpoint"]      # (2,)

    print(f"[load] p_sw {p_sw.shape}, |max|={np.abs(p_sw).max():.2f} Pa")
    print(f"[load] p_lens {p_lens_roi.shape}, |max|={np.abs(p_lens_roi).max():.2f} Pa")
    print(f"[load] {len(traps_m)} traps, A={idx_A}, B={idx_B}")
    print(f"  trap A = {traps_m[idx_A]*1e3} mm")
    print(f"  trap B = {traps_m[idx_B]*1e3} mm")
    print(f"  d_AB = {np.linalg.norm(traps_m[idx_A]-traps_m[idx_B])*1e3:.4f} mm")

    return p_sw, p_lens_roi, xg, yg, traps_m, idx_A, idx_B, midpoint


# ═══════════════════════════════════════════════════════════════════
# Gor'kov potential & force
# ═══════════════════════════════════════════════════════════════════

def compute_gorkov(p_field, dx):
    """
    Gor'kov potential on a 2D grid.

    U = (f1 / (2 * rho * c²)) |p|² - (3*f2 / (4*omega²*rho)) |grad(p)|²

    Returns U (Ny, Nx) in Joules.
    """
    p_abs2 = np.abs(p_field)**2

    dp_dx = np.gradient(p_field, dx, axis=1)
    dp_dy = np.gradient(p_field, dx, axis=0)
    grad_p_abs2 = np.abs(dp_dx)**2 + np.abs(dp_dy)**2

    coeff_p = F1 / (2.0 * RHO0 * C_WATER**2)
    coeff_k = 3.0 * F2 / (4.0 * OMEGA**2 * RHO0)

    U = coeff_p * p_abs2 - coeff_k * grad_p_abs2
    return U


def compute_force_field(U, dx):
    """
    F = -grad(U), returned as (Fy, Fx) each (Ny, Nx).
    """
    dUdx = np.gradient(U, dx, axis=1)
    dUdy = np.gradient(U, dx, axis=0)
    return -dUdx, -dUdy


def make_force_interpolators(Fx, Fy, xg, yg):
    """Create interpolators for force components."""
    interp_Fx = RegularGridInterpolator(
        (yg, xg), Fx, bounds_error=False, fill_value=0.0)
    interp_Fy = RegularGridInterpolator(
        (yg, xg), Fy, bounds_error=False, fill_value=0.0)
    return interp_Fx, interp_Fy


# ═══════════════════════════════════════════════════════════════════
# Control schedule
# ═══════════════════════════════════════════════════════════════════

def crossfade_schedule(t):
    """
    Smooth crossfade schedule: SW fades out while lens ramps in, then reverses.

    Returns (beta_sw, beta_lens) where the combined field is:
        p_comb = beta_sw * p_sw + beta_lens * exp(i*PSI) * p_lens

    Uses half-cosine (raised cosine) for smooth transitions.
    """
    if t < 0:
        return 1.0, 0.0
    elif t < T_RAMP_UP:
        frac = 0.5 * (1.0 - np.cos(np.pi * t / T_RAMP_UP))
        return 1.0 - frac, ALPHA_TARGET * frac
    elif t < T_RAMP_UP + T_HOLD:
        return 0.0, ALPHA_TARGET
    elif t < T_RAMP_UP + T_HOLD + T_RAMP_DOWN:
        t_rd = t - T_RAMP_UP - T_HOLD
        frac = 0.5 * (1.0 - np.cos(np.pi * t_rd / T_RAMP_DOWN))
        return frac, ALPHA_TARGET * (1.0 - frac)
    else:
        return 1.0, 0.0


T_TOTAL = T_RAMP_UP + T_HOLD + T_RAMP_DOWN + T_SETTLE


# ═══════════════════════════════════════════════════════════════════
# Particle dynamics (overdamped Euler integrator)
# ═══════════════════════════════════════════════════════════════════

def simulate_particles(p_sw, p_lens_roi, xg, yg, traps_m, idx_A, idx_B):
    """
    Simulate overdamped motion of 9 particles under time-varying Gor'kov force.

    dx/dt = mu * F(x, t)

    where F = -grad(U_gorkov(p_comb(t)))
    """
    dx = xg[1] - xg[0]
    n_particles = len(traps_m)
    n_steps = int(T_TOTAL / DT)
    frame_every = max(1, n_steps // N_FRAMES)

    scale = MU_STOKES * GORKOV_PREFACTOR * P_SCALE**2

    print(f"\n[sim] n_steps={n_steps}, dt={DT*1e6:.1f} µs, T_total={T_TOTAL*1e3:.1f} ms")
    print(f"  frame_every={frame_every}, expected frames ≈ {n_steps // frame_every}")
    print(f"  mu_Stokes = {MU_STOKES:.3e} m/(N·s)")
    print(f"  GORKOV_PREFACTOR = {GORKOV_PREFACTOR:.3e}")
    print(f"  P_SCALE = {P_SCALE:.0f}  → |p_sw|_eff ≈ {P_SCALE*46:.0f} Pa")
    print(f"  velocity_coeff (mu*V_p*P²) = {scale:.3e}")
    print(f"  capture_radius = {CAPTURE_RADIUS*1e6:.1f} µm")
    print(f"  trap_tolerance = {TRAP_TOLERANCE*1e6:.1f} µm")

    # Initialise positions at trap minima
    pos = traps_m.copy()  # (9, 2)
    initial_pos = traps_m.copy()

    # Storage
    trajectories = [pos.copy()]
    times = [0.0]
    alphas = [0.0]
    betas_sw = [1.0]
    U_frames = []
    Fx_frames = []
    Fy_frames = []

    # Pre-compute the standing-wave-only Gor'kov for reference
    U_sw_only = compute_gorkov(p_sw, dx)

    t0 = time.time()
    merge_time = None

    for step in range(n_steps):
        t = step * DT
        beta_sw_t, beta_lens_t = crossfade_schedule(t)

        # Combined field: crossfade between SW and lens
        p_comb = beta_sw_t * p_sw + beta_lens_t * np.exp(1j * PSI) * p_lens_roi

        # Gor'kov potential & force
        U = compute_gorkov(p_comb, dx)
        Fx, Fy = compute_force_field(U, dx)

        # Interpolate force at particle positions
        interp_Fx, interp_Fy = make_force_interpolators(Fx, Fy, xg, yg)

        for ip in range(n_particles):
            pt = np.array([[pos[ip, 1], pos[ip, 0]]])  # (y, x) for interpolator
            fx = float(interp_Fx(pt)[0])
            fy = float(interp_Fy(pt)[0])
            # Overdamped update: v = µ × P² × (2π/3)a³ × F_code
            pos[ip, 0] += scale * fx * DT
            pos[ip, 1] += scale * fy * DT

        # Clamp to ROI
        pos[:, 0] = np.clip(pos[:, 0], xg[2], xg[-3])
        pos[:, 1] = np.clip(pos[:, 1], yg[2], yg[-3])

        # Check for merge
        if merge_time is None:
            d_AB = np.linalg.norm(pos[idx_A] - pos[idx_B])
            if d_AB < CAPTURE_RADIUS:
                merge_time = t
                print(f"  *** MERGE at t={t*1e3:.3f} ms, d_AB={d_AB*1e6:.1f} µm")

        # Store frame data
        if step % frame_every == 0 or step == n_steps - 1:
            trajectories.append(pos.copy())
            times.append(t)
            alphas.append(beta_lens_t)
            betas_sw.append(beta_sw_t)
            U_frames.append(U.copy())
            Fx_frames.append(Fx.copy())
            Fy_frames.append(Fy.copy())

    elapsed = time.time() - t0
    print(f"[sim] Done in {elapsed:.1f}s, {len(trajectories)} frames")

    return {
        "trajectories": np.array(trajectories),  # (n_frames, 9, 2)
        "times": np.array(times),
        "alphas": np.array(alphas),
        "betas_sw": np.array(betas_sw),
        "U_frames": U_frames,
        "Fx_frames": Fx_frames,
        "Fy_frames": Fy_frames,
        "initial_pos": initial_pos,
        "U_sw_only": U_sw_only,
        "merge_time": merge_time,
    }


# ═══════════════════════════════════════════════════════════════════
# Success analysis
# ═══════════════════════════════════════════════════════════════════

def analyse_success(sim, traps_m, idx_A, idx_B):
    """Compute quantitative success metrics."""
    traj = sim["trajectories"]
    initial = sim["initial_pos"]
    final_pos = traj[-1]

    # A-B metrics
    pos_A_init = initial[idx_A]
    pos_A_final = final_pos[idx_A]
    pos_B_init = initial[idx_B]
    pos_B_final = final_pos[idx_B]

    d_A_moved = np.linalg.norm(pos_A_final - pos_A_init)
    d_B_moved = np.linalg.norm(pos_B_final - pos_B_init)
    d_AB_init = np.linalg.norm(pos_A_init - pos_B_init)
    d_AB_final = np.linalg.norm(pos_A_final - pos_B_final)

    # Did A cross the midpoint toward B?
    AB_vec = pos_B_init - pos_A_init
    AB_hat = AB_vec / np.linalg.norm(AB_vec)
    proj_A = np.dot(pos_A_final - pos_A_init, AB_hat)
    crossed_midpoint = proj_A > 0.5 * d_AB_init

    # Did A enter capture region of B?
    a_in_capture = d_AB_final < CAPTURE_RADIUS

    # Did B stay trapped?
    b_stayed = d_B_moved < TRAP_TOLERANCE

    # Neighbour stability
    neighbour_mask = np.ones(len(traps_m), dtype=bool)
    neighbour_mask[idx_A] = False
    neighbour_mask[idx_B] = False
    neighbour_disps = np.linalg.norm(final_pos[neighbour_mask] - initial[neighbour_mask], axis=1)
    max_neigh_disp = float(neighbour_disps.max())
    mean_neigh_disp = float(neighbour_disps.mean())
    any_neigh_escaped = bool(np.any(neighbour_disps > NEIGHBOUR_TOL))

    # Classification
    if a_in_capture and b_stayed and not any_neigh_escaped:
        classification = "successful_merge"
    elif (crossed_midpoint or proj_A > 0.3 * d_AB_init) and b_stayed:
        classification = "partial_success"
    else:
        classification = "failure"

    metrics = {
        "pos_A_init_mm": (pos_A_init * 1e3).tolist(),
        "pos_A_final_mm": (pos_A_final * 1e3).tolist(),
        "pos_B_init_mm": (pos_B_init * 1e3).tolist(),
        "pos_B_final_mm": (pos_B_final * 1e3).tolist(),
        "d_A_moved_um": float(d_A_moved * 1e6),
        "d_B_moved_um": float(d_B_moved * 1e6),
        "d_AB_init_um": float(d_AB_init * 1e6),
        "d_AB_final_um": float(d_AB_final * 1e6),
        "proj_A_toward_B_um": float(proj_A * 1e6),
        "crossed_midpoint": bool(crossed_midpoint),
        "A_in_capture_region": bool(a_in_capture),
        "B_stayed_in_trap": bool(b_stayed),
        "max_neighbour_disp_um": float(max_neigh_disp * 1e6),
        "mean_neighbour_disp_um": float(mean_neigh_disp * 1e6),
        "any_neighbour_escaped": bool(any_neigh_escaped),
        "merge_time_ms": float(sim["merge_time"] * 1e3) if sim["merge_time"] else None,
        "classification": classification,
        "capture_radius_um": float(CAPTURE_RADIUS * 1e6),
        "trap_tolerance_um": float(TRAP_TOLERANCE * 1e6),
    }

    print(f"\n{'='*60}")
    print(f"  SUCCESS ANALYSIS")
    print(f"{'='*60}")
    print(f"  A moved:         {d_A_moved*1e6:.1f} µm")
    print(f"  B moved:         {d_B_moved*1e6:.1f} µm")
    print(f"  d_AB init:       {d_AB_init*1e6:.1f} µm")
    print(f"  d_AB final:      {d_AB_final*1e6:.1f} µm")
    print(f"  A proj toward B: {proj_A*1e6:.1f} µm")
    print(f"  Crossed midpoint: {crossed_midpoint}")
    print(f"  A in capture:    {a_in_capture}")
    print(f"  B stayed:        {b_stayed}")
    print(f"  Max neighbour:   {max_neigh_disp*1e6:.1f} µm")
    print(f"  Mean neighbour:  {mean_neigh_disp*1e6:.1f} µm")
    print(f"  Any escaped:     {any_neigh_escaped}")
    print(f"  Classification:  {classification}")
    if sim["merge_time"]:
        print(f"  Merge time:      {sim['merge_time']*1e3:.3f} ms")
    print(f"{'='*60}\n")

    return metrics


# ═══════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════

def _cbar(ax, im, label=""):
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="4%", pad=0.06)
    plt.colorbar(im, cax=cax, label=label)


def plot_frame(ax, U, xg, yg, positions, traps_m, idx_A, idx_B,
               alpha_t, t_ms, Fx=None, Fy=None, show_force=True,
               trajectories_so_far=None):
    """Render a single simulation frame on the given axes."""
    xg_mm = xg * 1e3
    yg_mm = yg * 1e3
    ext = [xg_mm[0], xg_mm[-1], yg_mm[0], yg_mm[-1]]

    # Gor'kov potential
    im = ax.imshow(U, origin="lower", extent=ext, cmap="RdBu_r",
                   aspect="equal")

    # Trap markers (small grey crosses)
    for i, t in enumerate(traps_m):
        ax.plot(t[0]*1e3, t[1]*1e3, "+", color="0.5", ms=5, mew=0.5, zorder=3)

    # Particle positions
    for ip in range(len(positions)):
        x_mm, y_mm = positions[ip] * 1e3
        if ip == idx_A:
            ax.plot(x_mm, y_mm, "ro", ms=8, mew=1.5, mfc="red",
                    zorder=10, label="A" if t_ms < 0.1 else "")
            ax.annotate("A", (x_mm, y_mm), fontsize=8, color="red",
                        fontweight="bold", ha="center", va="bottom",
                        xytext=(0, 6), textcoords="offset points", zorder=11)
        elif ip == idx_B:
            ax.plot(x_mm, y_mm, "bs", ms=8, mew=1.5, mfc="blue",
                    zorder=10, label="B" if t_ms < 0.1 else "")
            ax.annotate("B", (x_mm, y_mm), fontsize=8, color="blue",
                        fontweight="bold", ha="center", va="bottom",
                        xytext=(0, 6), textcoords="offset points", zorder=11)
        else:
            ax.plot(x_mm, y_mm, "o", color="lime", ms=5, mew=0.8,
                    mfc="lime", zorder=8)

    # Trajectory trace for A
    if trajectories_so_far is not None and len(trajectories_so_far) > 1:
        tA = trajectories_so_far[:, idx_A, :] * 1e3
        ax.plot(tA[:, 0], tA[:, 1], "-", color="red", lw=1.0, alpha=0.6, zorder=6)
        tB = trajectories_so_far[:, idx_B, :] * 1e3
        ax.plot(tB[:, 0], tB[:, 1], "-", color="blue", lw=1.0, alpha=0.6, zorder=6)

    # Force vectors (subsample for clarity)
    if show_force and Fx is not None and Fy is not None:
        skip = 20
        xx = xg_mm[::skip]
        yy = yg_mm[::skip]
        Fx_sub = Fx[::skip, ::skip]
        Fy_sub = Fy[::skip, ::skip]
        mag = np.sqrt(Fx_sub**2 + Fy_sub**2)
        mag_max = mag.max()
        if mag_max > 0:
            ax.quiver(xx, yy, Fx_sub / mag_max, Fy_sub / mag_max,
                      color="white", alpha=0.3, scale=25, width=0.003,
                      headwidth=3, zorder=4)

    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(f"t = {t_ms:.2f} ms  |  β_sw = {alpha_t:.3f}",
                 fontsize=10)
    return im


def save_static_frame(U, xg, yg, positions, traps_m, idx_A, idx_B,
                      alpha_t, t_ms, Fx, Fy, trajectories_so_far,
                      filename, title_extra=""):
    """Save a standalone PNG frame with colorbar."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    im = plot_frame(ax, U, xg, yg, positions, traps_m, idx_A, idx_B,
                    alpha_t, t_ms, Fx, Fy, show_force=True,
                    trajectories_so_far=trajectories_so_far)
    _cbar(ax, im, "U_gor (J)")
    if title_extra:
        fig.suptitle(title_extra, fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {filename}")


def render_gif(sim, xg, yg, traps_m, idx_A, idx_B):
    """Render the simulation as a GIF."""
    traj = sim["trajectories"]
    times_ms = sim["times"] * 1e3
    alphas = sim["alphas"]
    U_frames = sim["U_frames"]
    Fx_frames = sim["Fx_frames"]
    Fy_frames = sim["Fy_frames"]

    n_frames = len(U_frames)
    print(f"\n[gif] Rendering {n_frames} frames...")

    # Determine consistent colour scale from all frames
    U_all_min = min(U.min() for U in U_frames)
    U_all_max = max(U.max() for U in U_frames)

    frames_images = []

    for fi in range(n_frames):
        fig, ax = plt.subplots(1, 1, figsize=(7, 6))

        xg_mm = xg * 1e3
        yg_mm = yg * 1e3
        ext = [xg_mm[0], xg_mm[-1], yg_mm[0], yg_mm[-1]]

        im = ax.imshow(U_frames[fi], origin="lower", extent=ext,
                       cmap="RdBu_r", aspect="equal",
                       vmin=U_all_min, vmax=U_all_max)

        # Trap crosses
        for i, t in enumerate(traps_m):
            ax.plot(t[0]*1e3, t[1]*1e3, "+", color="0.6", ms=5, mew=0.5, zorder=3)

        # Trajectory traces
        if fi > 0:
            tA = traj[:fi+1, idx_A, :] * 1e3
            ax.plot(tA[:, 0], tA[:, 1], "-", color="red", lw=1.0, alpha=0.5, zorder=6)
            tB = traj[:fi+1, idx_B, :] * 1e3
            ax.plot(tB[:, 0], tB[:, 1], "-", color="blue", lw=0.8, alpha=0.5, zorder=6)

        # Particles
        positions = traj[fi]
        for ip in range(len(positions)):
            x_mm, y_mm = positions[ip] * 1e3
            if ip == idx_A:
                ax.plot(x_mm, y_mm, "o", color="red", ms=8, mew=1.5,
                        mfc="red", mec="white", zorder=10)
                ax.annotate("A", (x_mm, y_mm), fontsize=9, color="red",
                            fontweight="bold", ha="center", va="bottom",
                            xytext=(0, 6), textcoords="offset points", zorder=11)
            elif ip == idx_B:
                ax.plot(x_mm, y_mm, "s", color="blue", ms=8, mew=1.5,
                        mfc="blue", mec="white", zorder=10)
                ax.annotate("B", (x_mm, y_mm), fontsize=9, color="blue",
                            fontweight="bold", ha="center", va="bottom",
                            xytext=(0, 6), textcoords="offset points", zorder=11)
            else:
                ax.plot(x_mm, y_mm, "o", color="lime", ms=5, mec="white",
                        mew=0.5, zorder=8)

        # Force vectors
        skip = 25
        Fx_sub = Fx_frames[fi][::skip, ::skip]
        Fy_sub = Fy_frames[fi][::skip, ::skip]
        mag = np.sqrt(Fx_sub**2 + Fy_sub**2)
        mag_max = mag.max()
        if mag_max > 0:
            ax.quiver(xg_mm[::skip], yg_mm[::skip],
                      Fx_sub / mag_max, Fy_sub / mag_max,
                      color="white", alpha=0.25, scale=30, width=0.003, zorder=4)

        _cbar(ax, im, "U_gor (J)")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        bsw = sim["betas_sw"][fi] if fi < len(sim["betas_sw"]) else 1.0
        ax.set_title(f"t = {times_ms[fi]:.1f} ms  |  β_sw = {bsw:.2f}  β_lens = {alphas[fi]:.2f}")

        fig.tight_layout()

        # Render to RGBA array
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        frames_images.append(buf.copy())
        plt.close(fig)

        if (fi + 1) % 20 == 0:
            print(f"  rendered {fi+1}/{n_frames}")

    # Save GIF using PIL
    from PIL import Image
    pil_frames = [Image.fromarray(f) for f in frames_images]
    gif_path = OUT_DIR / "particle_merge_demo.gif"
    pil_frames[0].save(
        gif_path, save_all=True, append_images=pil_frames[1:],
        duration=80, loop=0, optimize=False)
    print(f"[save] {gif_path.name} ({len(pil_frames)} frames)")

    return gif_path


def plot_trajectory_overlay(sim, xg, yg, traps_m, idx_A, idx_B):
    """Plot all particle trajectories as an overlay on the SW-only Gor'kov potential."""
    traj = sim["trajectories"]
    U_sw = sim["U_sw_only"]
    xg_mm = xg * 1e3
    yg_mm = yg * 1e3
    ext = [xg_mm[0], xg_mm[-1], yg_mm[0], yg_mm[-1]]

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    im = ax.imshow(U_sw, origin="lower", extent=ext, cmap="RdBu_r", aspect="equal")
    _cbar(ax, im, "U_gor (J) [SW only]")

    # Trap markers
    for i, t in enumerate(traps_m):
        ax.plot(t[0]*1e3, t[1]*1e3, "+", color="grey", ms=8, mew=1, zorder=3)

    # Particle trajectories
    colors = plt.cm.Set1(np.linspace(0, 1, len(traps_m)))
    for ip in range(len(traps_m)):
        path = traj[:, ip, :] * 1e3
        if ip == idx_A:
            ax.plot(path[:, 0], path[:, 1], "-o", color="red", ms=2,
                    lw=1.5, label=f"A (particle {ip})", zorder=9)
            ax.plot(path[0, 0], path[0, 1], "r^", ms=10, zorder=10)
            ax.plot(path[-1, 0], path[-1, 1], "rv", ms=10, zorder=10)
        elif ip == idx_B:
            ax.plot(path[:, 0], path[:, 1], "-s", color="blue", ms=2,
                    lw=1.5, label=f"B (particle {ip})", zorder=9)
            ax.plot(path[0, 0], path[0, 1], "b^", ms=10, zorder=10)
            ax.plot(path[-1, 0], path[-1, 1], "bv", ms=10, zorder=10)
        else:
            ax.plot(path[:, 0], path[:, 1], "-", color=colors[ip],
                    lw=0.8, alpha=0.7, zorder=5)
            ax.plot(path[0, 0], path[0, 1], "^", color=colors[ip], ms=5, zorder=6)
            ax.plot(path[-1, 0], path[-1, 1], "v", color=colors[ip], ms=5, zorder=6)

    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("Particle Trajectories — All 9 Particles\n(▲ start, ▼ end)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "trajectory_overlay.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] trajectory_overlay.png")


def plot_displacement_summary(sim, traps_m, idx_A, idx_B):
    """Bar chart of total displacement for each particle."""
    traj = sim["trajectories"]
    initial = sim["initial_pos"]
    final_pos = traj[-1]
    disps_um = np.linalg.norm(final_pos - initial, axis=1) * 1e6

    fig, ax = plt.subplots(figsize=(8, 4))
    colours = ["red" if i == idx_A else "blue" if i == idx_B else "grey"
               for i in range(len(traps_m))]
    labels = [f"A ({i})" if i == idx_A else f"B ({i})" if i == idx_B else str(i)
              for i in range(len(traps_m))]
    bars = ax.bar(range(len(traps_m)), disps_um, color=colours)
    ax.set_xticks(range(len(traps_m)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Total displacement (µm)")
    ax.set_xlabel("Particle")
    ax.set_title("Final Displacement of Each Particle")
    ax.axhline(TRAP_TOLERANCE * 1e6, color="orange", ls="--", lw=1, label="tolerance")
    ax.axhline(CAPTURE_RADIUS * 1e6, color="green", ls="--", lw=1, label="capture radius")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "displacement_summary.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] displacement_summary.png")


def plot_alpha_schedule():
    """Plot the crossfade control schedule."""
    ts = np.linspace(0, T_TOTAL, 500)
    bsw_arr = []
    blens_arr = []
    for t in ts:
        bsw, blens = crossfade_schedule(t)
        bsw_arr.append(bsw)
        blens_arr.append(blens)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(np.array(ts) * 1e3, bsw_arr, "b-", lw=2, label="β_sw")
    ax.plot(np.array(ts) * 1e3, blens_arr, "r-", lw=2, label="β_lens")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Weight")
    ax.set_title(f"Crossfade Schedule — α={ALPHA_TARGET}, ψ={PSI/(np.pi):.1f}π")
    ax.legend(fontsize=9)
    ax.set_ylim(-0.1, max(ALPHA_TARGET, 1.0) * 1.15)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "alpha_schedule.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] alpha_schedule.png")


def plot_distance_vs_time(sim, traps_m, idx_A, idx_B):
    """Plot d_AB over time and A/B displacements from initial."""
    traj = sim["trajectories"]
    times_ms = sim["times"] * 1e3
    initial = sim["initial_pos"]

    d_AB = np.linalg.norm(traj[:, idx_A] - traj[:, idx_B], axis=1) * 1e6
    d_A_from_init = np.linalg.norm(traj[:, idx_A] - initial[idx_A], axis=1) * 1e6
    d_B_from_init = np.linalg.norm(traj[:, idx_B] - initial[idx_B], axis=1) * 1e6

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax = axes[0]
    ax.plot(times_ms, d_AB, "k-", lw=1.5, label="d(A,B)")
    ax.axhline(CAPTURE_RADIUS * 1e6, color="green", ls="--", lw=1, label="capture radius")
    ax.set_ylabel("d_AB (µm)")
    ax.legend(fontsize=8)
    ax.set_title("A–B Separation Over Time")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(times_ms, d_A_from_init, "r-", lw=1.5, label="A displacement")
    ax.plot(times_ms, d_B_from_init, "b-", lw=1.5, label="B displacement")
    ax.axhline(TRAP_TOLERANCE * 1e6, color="orange", ls="--", lw=1, label="tolerance")
    ax.set_ylabel("Displacement from init (µm)")
    ax.set_xlabel("Time (ms)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "distance_vs_time.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] distance_vs_time.png")


# ═══════════════════════════════════════════════════════════════════
# Output
# ═══════════════════════════════════════════════════════════════════

def save_config():
    """Save run configuration."""
    cfg = {
        "timestamp": TS,
        "overlay_npz": str(OVERLAY_NPZ),
        "model": "crossfade: p = beta_sw*p_sw + beta_lens*exp(i*psi)*p_lens",
        "alpha_target": ALPHA_TARGET,
        "psi": PSI,
        "psi_over_pi": PSI / np.pi,
        "T_ramp_up_ms": T_RAMP_UP * 1e3,
        "T_hold_ms": T_HOLD * 1e3,
        "T_ramp_down_ms": T_RAMP_DOWN * 1e3,
        "T_settle_ms": T_SETTLE * 1e3,
        "T_total_ms": T_TOTAL * 1e3,
        "dt_us": DT * 1e6,
        "n_frames_target": N_FRAMES,
        "particle_radius_um": A_PART * 1e6,
        "eta_Pa_s": ETA,
        "mu_Stokes": MU_STOKES,
        "P_SCALE": P_SCALE,
        "p_sw_effective_kPa": P_SCALE * 46 / 1e3,
        "ka": float(_KA),
        "gorkov_prefactor": GORKOV_PREFACTOR,
        "capture_radius_um": CAPTURE_RADIUS * 1e6,
        "trap_tolerance_um": TRAP_TOLERANCE * 1e6,
        "neighbour_tolerance_um": NEIGHBOUR_TOL * 1e6,
        "f1": F1,
        "f2": F2,
        "rho_p": RHO_P,
        "c_p": C_P,
    }
    with open(OUT_DIR / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[save] config.json")
    return cfg


def save_metrics(metrics):
    """Save metrics to CSV."""
    with open(OUT_DIR / "metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        for k, v in metrics.items():
            w.writerow([k, v])
    print(f"[save] metrics.csv")


def save_npz(sim, xg, yg, traps_m, idx_A, idx_B):
    """Save simulation data as NPZ."""
    np.savez_compressed(
        NPZ_DIR / "simulation_data.npz",
        trajectories=sim["trajectories"],
        times=sim["times"],
        alphas=sim["alphas"],
        betas_sw=sim["betas_sw"],
        initial_pos=sim["initial_pos"],
        U_sw_only=sim["U_sw_only"],
        xg=xg, yg=yg,
        traps_m=traps_m,
        idx_A=idx_A, idx_B=idx_B,
    )
    print(f"[save] simulation_data.npz")


def write_index(cfg, metrics):
    """Write INDEX.md summary."""
    lines = [
        "# C-Shape Particle Merge Demo",
        "",
        f"**Generated:** {TS}",
        "",
        "## Model",
        "",
        "**Crossfade transport**: ",
        "`p_comb = β_sw(t) × p_sw + β_lens(t) × exp(iψ) × p_lens`",
        "",
        "The standing-wave amplitude β_sw is smoothly crossfaded from 1 → 0",
        "while the lens amplitude β_lens ramps from 0 → α.  With the SW traps",
        "suppressed, the lens potential (at ψ = 3π/2) sweeps particle A toward B.",
        "On restoration of the SW (β_sw: 0 → 1), A settles into B's trap basin.",
        "",
        "## Sources",
        f"- Lens + SW fields: `{OVERLAY_NPZ.name}` (from overlay study)",
        f"- Standing-wave peak: ~46.4 Pa",
        f"- Lens peak in ROI: ~10.7 Pa",
        "",
        "## Trap Cluster",
        f"- 9 Gor'kov-stable traps in ROI",
        f"- Trap A: {metrics.get('pos_A_init_mm', '?')} mm",
        f"- Trap B: {metrics.get('pos_B_init_mm', '?')} mm",
        f"- d_AB initial: {metrics.get('d_AB_init_um', '?'):.1f} µm",
        "",
        "## Control Schedule",
        f"- α_target = {ALPHA_TARGET}",
        f"- ψ = {PSI/(np.pi):.1f}π = {PSI:.4f} rad",
        f"- Crossfade-up: {T_RAMP_UP*1e3:.0f} ms (β_sw: 1→0, β_lens: 0→α)",
        f"- Hold:         {T_HOLD*1e3:.0f} ms (β_sw=0, β_lens=α)",
        f"- Crossfade-dn: {T_RAMP_DOWN*1e3:.0f} ms (β_sw: 0→1, β_lens: α→0)",
        f"- Settle:       {T_SETTLE*1e3:.0f} ms (pure SW)",
        f"- Total:        {T_TOTAL*1e3:.0f} ms",
        "",
        "## Dynamics",
        f"- Overdamped: dx/dt = µ·F, µ = 1/(6πηa)",
        f"- Particle: polystyrene, a = {A_PART*1e6:.0f} µm, ρ = {RHO_P} kg/m³",
        f"- µ_Stokes = {MU_STOKES:.3e} m/(N·s)",
        f"- dt = {DT*1e6:.0f} µs",
        "",
        "## Success Criteria",
        f"- Capture radius: {CAPTURE_RADIUS*1e6:.1f} µm (0.30 × λ/2)",
        f"- Trap tolerance: {TRAP_TOLERANCE*1e6:.1f} µm (0.50 × λ/2)",
        "",
        "## Results",
        f"- **Classification: {metrics['classification']}**",
        f"- A moved: {metrics['d_A_moved_um']:.1f} µm",
        f"- B moved: {metrics['d_B_moved_um']:.1f} µm",
        f"- A crossed midpoint: {metrics['crossed_midpoint']}",
        f"- A in capture region: {metrics['A_in_capture_region']}",
        f"- B stayed in trap: {metrics['B_stayed_in_trap']}",
        f"- Max neighbour displacement: {metrics['max_neighbour_disp_um']:.1f} µm",
        f"- Mean neighbour displacement: {metrics['mean_neighbour_disp_um']:.1f} µm",
    ]
    if metrics.get("merge_time_ms"):
        lines.append(f"- Merge time: {metrics['merge_time_ms']:.3f} ms")

    lines += [
        "",
        "## Files",
        "- `particle_merge_demo.gif` — main animation",
        "- `figures/` — static PNGs",
        "- `npz/simulation_data.npz` — trajectories & fields",
        "- `config.json` — run parameters",
        "- `metrics.csv` — quantitative metrics",
    ]

    if metrics["classification"] != "successful_merge":
        lines += [
            "",
            "## Recommended Next Steps",
            "- Try slower crossfade ramp for better neighbour stability",
            "- Increase settle time to allow displaced neighbours to return",
            "- Consider partial crossfade (β_sw_min > 0) to retain some SW trapping",
            "- Consider optimised lens (iterative phase retrieval) for stronger local gradient",
        ]

    with open(OUT_DIR / "INDEX.md", "w") as f:
        f.write("\n".join(lines))
    print(f"[save] INDEX.md")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()

    # 0. Save config
    cfg = save_config()

    # 1. Load data
    p_sw, p_lens_roi, xg, yg, traps_m, idx_A, idx_B, midpoint = load_fields()

    # 2. Plot control schedule
    plot_alpha_schedule()

    # 3. Save initial state
    dx = xg[1] - xg[0]
    U_sw = compute_gorkov(p_sw, dx)
    Fx_sw, Fy_sw = compute_force_field(U_sw, dx)
    save_static_frame(
        U_sw, xg, yg, traps_m, traps_m, idx_A, idx_B,
        1.0, 0.0, Fx_sw, Fy_sw, None,
        "initial_state.png", "Initial State — SW Only (β_sw = 1)")

    # 4. Run simulation
    sim = simulate_particles(p_sw, p_lens_roi, xg, yg, traps_m, idx_A, idx_B)

    # 5. Analyse results
    metrics = analyse_success(sim, traps_m, idx_A, idx_B)

    # 6. Save static frames: initial, midpoint, final
    traj = sim["trajectories"]
    n_frames_total = len(sim["U_frames"])
    mid_frame = n_frames_total // 2

    save_static_frame(
        sim["U_frames"][0], xg, yg, traj[0], traps_m, idx_A, idx_B,
        sim["betas_sw"][0], sim["times"][0] * 1e3,
        sim["Fx_frames"][0], sim["Fy_frames"][0], traj[:1],
        "frame_initial.png", "Frame: Initial (t=0)")

    save_static_frame(
        sim["U_frames"][mid_frame], xg, yg, traj[mid_frame], traps_m,
        idx_A, idx_B,
        sim["betas_sw"][mid_frame], sim["times"][mid_frame] * 1e3,
        sim["Fx_frames"][mid_frame], sim["Fy_frames"][mid_frame],
        traj[:mid_frame+1],
        "frame_midpoint.png", "Frame: Midpoint")

    save_static_frame(
        sim["U_frames"][-1], xg, yg, traj[-1], traps_m, idx_A, idx_B,
        sim["betas_sw"][-1], sim["times"][-1] * 1e3,
        sim["Fx_frames"][-1], sim["Fy_frames"][-1], traj,
        "frame_final.png", "Frame: Final")

    # 7. Render GIF
    gif_path = render_gif(sim, xg, yg, traps_m, idx_A, idx_B)

    # 8. Additional plots
    plot_trajectory_overlay(sim, xg, yg, traps_m, idx_A, idx_B)
    plot_displacement_summary(sim, traps_m, idx_A, idx_B)
    plot_distance_vs_time(sim, traps_m, idx_A, idx_B)

    # 9. Save data
    save_metrics(metrics)
    save_npz(sim, xg, yg, traps_m, idx_A, idx_B)
    write_index(cfg, metrics)

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  Total runtime: {elapsed:.1f}s")
    print(f"  Output: {OUT_DIR}")
    print(f"  GIF: {gif_path}")
    print(f"  Classification: {metrics['classification']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
