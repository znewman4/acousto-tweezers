#!/usr/bin/env python3
"""
Vortex Transport Localisation Study
=====================================

Investigates whether changing the **physical footprint of the vortex lens**
(via aperture diameter) and reducing the standing wave **less aggressively**
can produce A→B transport while keeping neighbouring particles stationary.

Key insight from previous study (c_shape_transport_refinement):
  - Spatial windowing of the vortex field at the ROI scale breaks the
    phase structure needed for directional transport.
  - Only the full vortex lens produces transport A→B.
  - Best result: d_AB ≈ 58 µm but max neighbour displacement ≈ 189 µm.

Strategy:
  1) Keep the full vortex phase structure (no spatial windowing)
  2) Change the **physical footprint** by varying lens aperture diameter:
     Smaller aperture → less focused beam → different spatial distribution
  3) Reduce the standing wave **less aggressively** (higher β_sw_min)
  4) Sweep vortex strength (α) to find optimum
  5) Add relaxation phase after ramp-down to observe trap recovery

Lens field generation:
  - Load the lens-plane field from the manufacturing study
  - Apply circular aperture mask at each diameter
  - Forward-propagate via ASM to the target plane (z*)
  - Interpolate onto the ROI grid
  This generates physically correct fields for each aperture size.

Schedule:
  ramp_up → hold → ramp_down → relax
  where relax = 500 ms pure standing wave for trap recovery observation.

Model:
  p_comb(t) = β_sw(t) · p_sw + α · β_vortex(t) · exp(iψ) · p_lens

Outputs → results/vortex_transport_localisation_study_<TS>/
"""
from __future__ import annotations

import csv
import itertools
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ── Project root ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from acoustweezers.physics.acoustics.angular_spectrum import propagate_pressure_asm

# ═══════════════════════════════════════════════════════════════════
# Physical constants
# ═══════════════════════════════════════════════════════════════════
C_WATER = 1484.0          # m/s
F_HZ    = 2.0e6           # Hz
OMEGA   = 2.0 * np.pi * F_HZ
K_WATER = OMEGA / C_WATER
LAM     = C_WATER / F_HZ  # 0.742 mm
TRAP_SP = LAM / 2.0       # ~0.371 mm
RHO0    = 997.0            # kg/m³

# Particle: polystyrene 50 µm radius
RHO_P   = 1050.0          # kg/m³
C_P     = 2350.0          # m/s
A_PART  = 50.0e-6         # m
ETA     = 1.0e-3          # Pa·s

_KA = K_WATER * A_PART
assert _KA < 0.5, f"ka = {_KA:.3f} exceeds small-particle limit"

MU_STOKES = 1.0 / (6.0 * np.pi * ETA * A_PART)
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

LENS_PLANE_NPZ = (PROJECT_ROOT / "results"
                  / "c_shape_lens_15mm_manufacturing_study_20260310_153032"
                  / "npz" / "lens_plane.npz")

# ═══════════════════════════════════════════════════════════════════
# Pressure scaling & control params
# ═══════════════════════════════════════════════════════════════════
P_SCALE = 3.0e3            # → |p_sw|_eff ≈ 140 kPa
PSI = 3.0 * np.pi / 2.0   # empirically verified optimal phase offset

T_RAMP_UP   = 0.200       # 200 ms
T_HOLD      = 0.100       # 100 ms
T_RAMP_DOWN = 0.200       # 200 ms
T_RELAX     = 0.500       # 500 ms — NEW: observe trap recovery
DT          = 1.0e-4      # 0.1 ms
N_FRAMES    = 200

CAPTURE_RADIUS = 0.30 * TRAP_SP
TRAP_TOLERANCE = 0.50 * TRAP_SP
NEIGHBOUR_TOL  = 0.50 * TRAP_SP

# ═══════════════════════════════════════════════════════════════════
# Parameter sweep
# ═══════════════════════════════════════════════════════════════════

# Vortex footprint: lens diameter [mm]
LENS_RADII_MM = [15, 12, 10, 8]

# Standing wave reduction (less aggressive than previous study)
BETA_SW_MIN_VALUES = [0.3, 0.4, 0.5, 0.6]

# Vortex strength (normalised: α=1.0 → same effective amplitude as previous
# baseline α=5.0 with printable lens, i.e. 53.5 Pa effective peak)
ALPHA_VALUES = [0.4, 0.6, 0.8, 1.0]
ALPHA_REFERENCE_PA = 5.0 * 10.74  # previous baseline: α=5.0 × |p_printable|_max

# Ramp type
RAMP_VARIANTS = [
    {"label": "fast", "ramp_up": 0.200, "hold": 0.100, "ramp_down": 0.200},
]

# Translation mode (keep existing)
TRANSLATION_MODES = ["static"]

# ═══════════════════════════════════════════════════════════════════
# Lens field generation parameters
# ═══════════════════════════════════════════════════════════════════
LENS_N_GRID = 512
LENS_HALF   = 11.25e-3    # ±11.25 mm
LENS_Z_PROP = 6.1855e-3   # z* propagation distance
CX_ROI      = 3.0e-3      # ROI centre in FEM coords
CY_ROI      = 3.0e-3

# ═══════════════════════════════════════════════════════════════════
# Output
# ═══════════════════════════════════════════════════════════════════
TS = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = PROJECT_ROOT / "results" / f"vortex_transport_localisation_study_{TS}"
FIG_DIR = OUT_DIR / "figures"
NPZ_DIR = OUT_DIR / "npz"
GIF_DIR = OUT_DIR / "gif"
for d in [OUT_DIR, FIG_DIR, NPZ_DIR, GIF_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════

def load_sw_and_traps():
    """Load standing wave field and trap positions from overlay study."""
    d = np.load(OVERLAY_NPZ, allow_pickle=True)
    p_sw       = d["p_sw"]          # (400, 400) complex
    xg         = d["xg"]            # (400,)
    yg         = d["yg"]            # (400,)
    traps_m    = d["traps_m"]       # (9, 2)
    idx_A      = int(d["idx_A"])
    idx_B      = int(d["idx_B"])
    midpoint   = d["midpoint"]      # (2,)

    print(f"[load] p_sw {p_sw.shape}, |max|={np.abs(p_sw).max():.2f} Pa")
    print(f"[load] {len(traps_m)} traps, A={idx_A}, B={idx_B}")

    d_AB = np.linalg.norm(traps_m[idx_A] - traps_m[idx_B])
    e_AB = (traps_m[idx_B] - traps_m[idx_A]) / d_AB
    print(f"  d_AB = {d_AB*1e6:.1f} µm, e_AB = {e_AB}")

    return p_sw, xg, yg, traps_m, idx_A, idx_B, midpoint, d_AB, e_AB


def generate_lens_fields(xg_roi, yg_roi):
    """
    Generate lens fields for each aperture diameter.

    Pipeline:
      1) Load lens-plane backpropagated field
      2) Extract wrapped phase
      3) For each diameter: apply circular aperture → phase-only drive → ASM forward propagate
      4) Interpolate onto ROI grid

    Returns dict: lens_radius_mm → p_lens_roi (400,400) complex
    """
    print("\n=== Generating lens fields for each aperture ===\n")

    lp = np.load(LENS_PLANE_NPZ)
    p_lens_plane = lp["p_lens"]

    xg_l = np.linspace(-LENS_HALF, LENS_HALF, LENS_N_GRID)
    yg_l = np.linspace(-LENS_HALF, LENS_HALF, LENS_N_GRID)
    dx_l = xg_l[1] - xg_l[0]
    XX, YY = np.meshgrid(xg_l, yg_l)
    R = np.sqrt(XX**2 + YY**2)

    phase_raw = np.angle(p_lens_plane)
    phase_wrapped = np.mod(phase_raw, 2.0 * np.pi)

    lens_fields = {}
    for d_mm in LENS_RADII_MM:
        r = d_mm / 2.0 * 1e-3
        mask = R <= r

        # Phase-only drive within circular aperture
        p_drive = np.zeros_like(p_lens_plane)
        p_drive[mask] = np.exp(1j * phase_wrapped[mask])

        # Forward propagate to target plane
        p_recon = propagate_pressure_asm(
            p_drive, dx_l, dx_l, K_WATER, z=LENS_Z_PROP,
            pad_factor=2, include_evanescent=False)

        # Interpolate onto ROI grid
        interp_re = RegularGridInterpolator(
            (yg_l, xg_l), np.real(p_recon),
            bounds_error=False, fill_value=0.0)
        interp_im = RegularGridInterpolator(
            (yg_l, xg_l), np.imag(p_recon),
            bounds_error=False, fill_value=0.0)

        YR, XR = np.meshgrid(yg_roi, xg_roi, indexing="ij")
        pts = np.column_stack([
            (YR - CY_ROI).ravel(),
            (XR - CX_ROI).ravel()])

        p_roi = (interp_re(pts) + 1j * interp_im(pts)).reshape(
            len(yg_roi), len(xg_roi))

        # Normalise so that α=1.0 gives the same effective peak amplitude
        # as the previous baseline (α_prev=5.0 with printable lens ≈ 53.5 Pa).
        # This means the user's α directly controls the perturbation fraction.
        p_max = np.abs(p_roi).max()
        if p_max > 0:
            p_roi = p_roi * (ALPHA_REFERENCE_PA / p_max)
        print(f"  D={d_mm}mm → raw |p|_max = {p_max:.4f} Pa, "
              f"normalised to {np.abs(p_roi).max():.2f} Pa (at α=1)")

        lens_fields[d_mm] = p_roi

    return lens_fields


# ═══════════════════════════════════════════════════════════════════
# Gor'kov potential & force
# ═══════════════════════════════════════════════════════════════════

def compute_gorkov(p_field, dx):
    """Gor'kov potential U on 2D grid."""
    p_abs2 = np.abs(p_field)**2
    dp_dx = np.gradient(p_field, dx, axis=1)
    dp_dy = np.gradient(p_field, dx, axis=0)
    grad_p_abs2 = np.abs(dp_dx)**2 + np.abs(dp_dy)**2

    coeff_p = F1 / (2.0 * RHO0 * C_WATER**2)
    coeff_k = 3.0 * F2 / (4.0 * OMEGA**2 * RHO0)
    return coeff_p * p_abs2 - coeff_k * grad_p_abs2


def compute_force_field(U, dx):
    """F = -grad(U) → (Fx, Fy)."""
    return -np.gradient(U, dx, axis=1), -np.gradient(U, dx, axis=0)


def make_force_interpolators(Fx, Fy, xg, yg):
    """Create RegularGridInterpolator for force components."""
    interp_Fx = RegularGridInterpolator(
        (yg, xg), Fx, bounds_error=False, fill_value=0.0)
    interp_Fy = RegularGridInterpolator(
        (yg, xg), Fy, bounds_error=False, fill_value=0.0)
    return interp_Fx, interp_Fy


# ═══════════════════════════════════════════════════════════════════
# Bilinear Gorkov decomposition
# ═══════════════════════════════════════════════════════════════════

def precompute_bilinear_basis(p1, p2, dx):
    """
    Precompute Gorkov bilinear basis fields.

    U(β1·p1 + β2·p2) = β1²·U_11 + β2²·U_22 + 2·β1·β2·U_12

    Returns (Fx_11, Fy_11, Fx_22, Fy_22, Fx_12, Fy_12, U_11, U_22, U_12)
    """
    U_11 = compute_gorkov(p1, dx)
    U_22 = compute_gorkov(p2, dx)

    coeff_p = F1 / (2.0 * RHO0 * C_WATER**2)
    coeff_k = 3.0 * F2 / (4.0 * OMEGA**2 * RHO0)

    cross_p = np.real(np.conj(p1) * p2)
    dp1_dx = np.gradient(p1, dx, axis=1)
    dp1_dy = np.gradient(p1, dx, axis=0)
    dp2_dx = np.gradient(p2, dx, axis=1)
    dp2_dy = np.gradient(p2, dx, axis=0)
    cross_grad = (np.real(np.conj(dp1_dx) * dp2_dx)
                  + np.real(np.conj(dp1_dy) * dp2_dy))
    U_12 = coeff_p * cross_p - coeff_k * cross_grad

    Fx_11, Fy_11 = compute_force_field(U_11, dx)
    Fx_22, Fy_22 = compute_force_field(U_22, dx)
    Fx_12, Fy_12 = compute_force_field(U_12, dx)

    return (Fx_11, Fy_11, Fx_22, Fy_22, Fx_12, Fy_12,
            U_11, U_22, U_12)


# ═══════════════════════════════════════════════════════════════════
# Control schedule
# ═══════════════════════════════════════════════════════════════════

def make_schedule(beta_sw_min, alpha, ramp_up, hold, ramp_down, relax):
    """
    Return a schedule function (t) → (beta_sw, beta_lens)
    and total time.

    Phases: ramp_up → hold → ramp_down → relax (pure SW)
    """
    t_total = ramp_up + hold + ramp_down + relax

    def schedule(t):
        if t < 0:
            return 1.0, 0.0
        elif t < ramp_up:
            frac = 0.5 * (1.0 - np.cos(np.pi * t / ramp_up))
            beta_sw = 1.0 - (1.0 - beta_sw_min) * frac
            beta_lens = alpha * frac
            return beta_sw, beta_lens
        elif t < ramp_up + hold:
            return beta_sw_min, alpha
        elif t < ramp_up + hold + ramp_down:
            t_rd = t - ramp_up - hold
            frac = 0.5 * (1.0 - np.cos(np.pi * t_rd / ramp_down))
            beta_sw = beta_sw_min + (1.0 - beta_sw_min) * frac
            beta_lens = alpha * (1.0 - frac)
            return beta_sw, beta_lens
        else:
            # Relax phase: pure standing wave
            return 1.0, 0.0

    return schedule, t_total


# ═══════════════════════════════════════════════════════════════════
# Particle dynamics
# ═══════════════════════════════════════════════════════════════════

def simulate_particles(p_sw, p_lens_roi, xg, yg, traps_m,
                       idx_A, idx_B,
                       schedule_fn, t_total,
                       store_fields=False):
    """
    Simulate overdamped motion of 9 particles under time-varying field.

    Uses bilinear Gorkov decomposition (static centre, fast path).
    """
    dx = xg[1] - xg[0]
    n_particles = len(traps_m)
    n_steps = int(t_total / DT)
    frame_every = max(1, n_steps // N_FRAMES)
    scale = MU_STOKES * GORKOV_PREFACTOR * P_SCALE**2

    pos = traps_m.copy()
    initial_pos = traps_m.copy()

    trajectories = [pos.copy()]
    times = [0.0]
    betas_sw = [1.0]
    betas_lens = [0.0]

    U_frames = [] if store_fields else None
    Fx_frames = [] if store_fields else None
    Fy_frames = [] if store_fields else None

    merge_time = None

    # Precompute bilinear basis (full lens, no windowing)
    p_lens2 = np.exp(1j * PSI) * p_lens_roi

    (Fx_11, Fy_11, Fx_22, Fy_22, Fx_12, Fy_12,
     U_11, U_22, U_12) = precompute_bilinear_basis(p_sw, p_lens2, dx)

    iFx_11, iFy_11 = make_force_interpolators(Fx_11, Fy_11, xg, yg)
    iFx_22, iFy_22 = make_force_interpolators(Fx_22, Fy_22, xg, yg)
    iFx_12, iFy_12 = make_force_interpolators(Fx_12, Fy_12, xg, yg)

    for step in range(n_steps):
        t = step * DT
        beta_sw_t, beta_lens_t = schedule_fn(t)
        b1 = beta_sw_t
        b2 = beta_lens_t

        pts = np.column_stack([pos[:, 1], pos[:, 0]])  # (N, 2) as (y, x)

        fx = (b1**2 * iFx_11(pts) + b2**2 * iFx_22(pts)
              + 2 * b1 * b2 * iFx_12(pts))
        fy = (b1**2 * iFy_11(pts) + b2**2 * iFy_22(pts)
              + 2 * b1 * b2 * iFy_12(pts))

        pos[:, 0] += scale * fx * DT
        pos[:, 1] += scale * fy * DT
        pos[:, 0] = np.clip(pos[:, 0], xg[2], xg[-3])
        pos[:, 1] = np.clip(pos[:, 1], yg[2], yg[-3])

        if merge_time is None:
            d_AB_cur = np.linalg.norm(pos[idx_A] - pos[idx_B])
            if d_AB_cur < CAPTURE_RADIUS:
                merge_time = t

        if step % frame_every == 0 or step == n_steps - 1:
            trajectories.append(pos.copy())
            times.append(t)
            betas_sw.append(beta_sw_t)
            betas_lens.append(beta_lens_t)
            if store_fields:
                U = b1**2 * U_11 + b2**2 * U_22 + 2 * b1 * b2 * U_12
                Fx = b1**2 * Fx_11 + b2**2 * Fx_22 + 2 * b1 * b2 * Fx_12
                Fy = b1**2 * Fy_11 + b2**2 * Fy_22 + 2 * b1 * b2 * Fy_12
                U_frames.append(U)
                Fx_frames.append(Fx)
                Fy_frames.append(Fy)

    U_sw_only = compute_gorkov(p_sw, dx)

    return {
        "trajectories": np.array(trajectories),
        "times": np.array(times),
        "betas_sw": np.array(betas_sw),
        "betas_lens": np.array(betas_lens),
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

    pos_A_init = initial[idx_A]
    pos_A_final = final_pos[idx_A]
    pos_B_init = initial[idx_B]
    pos_B_final = final_pos[idx_B]

    d_A_moved = np.linalg.norm(pos_A_final - pos_A_init)
    d_B_moved = np.linalg.norm(pos_B_final - pos_B_init)
    d_AB_init = np.linalg.norm(pos_A_init - pos_B_init)
    d_AB_final = np.linalg.norm(pos_A_final - pos_B_final)

    AB_vec = pos_B_init - pos_A_init
    AB_hat = AB_vec / np.linalg.norm(AB_vec)
    proj_A = np.dot(pos_A_final - pos_A_init, AB_hat)
    crossed_midpoint = proj_A > 0.5 * d_AB_init
    a_in_capture = d_AB_final < CAPTURE_RADIUS
    b_stayed = d_B_moved < TRAP_TOLERANCE

    neighbour_mask = np.ones(len(traps_m), dtype=bool)
    neighbour_mask[idx_A] = False
    neighbour_mask[idx_B] = False
    neighbour_disps = np.linalg.norm(
        final_pos[neighbour_mask] - initial[neighbour_mask], axis=1)
    max_neigh = float(neighbour_disps.max())
    mean_neigh = float(neighbour_disps.mean())
    any_escaped = bool(np.any(neighbour_disps > NEIGHBOUR_TOL))

    if a_in_capture and b_stayed and not any_escaped:
        classification = "successful_merge"
    elif (crossed_midpoint or proj_A > 0.3 * d_AB_init) and b_stayed:
        classification = "partial_success"
    else:
        classification = "failure"

    return {
        "d_A_moved_um": float(d_A_moved * 1e6),
        "d_B_moved_um": float(d_B_moved * 1e6),
        "d_AB_init_um": float(d_AB_init * 1e6),
        "d_AB_final_um": float(d_AB_final * 1e6),
        "proj_A_toward_B_um": float(proj_A * 1e6),
        "crossed_midpoint": bool(crossed_midpoint),
        "A_in_capture_region": bool(a_in_capture),
        "B_stayed_in_trap": bool(b_stayed),
        "max_neighbour_disp_um": float(max_neigh * 1e6),
        "mean_neighbour_disp_um": float(mean_neigh * 1e6),
        "any_neighbour_escaped": bool(any_escaped),
        "merge_time_ms": float(sim["merge_time"] * 1e3) if sim["merge_time"] else None,
        "classification": classification,
    }


# ═══════════════════════════════════════════════════════════════════
# Trajectory plots (primary output)
# ═══════════════════════════════════════════════════════════════════

def _cbar(ax, im, label=""):
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="4%", pad=0.06)
    plt.colorbar(im, cax=cax, label=label)


def plot_trajectory(sim, xg, yg, traps_m, idx_A, idx_B, title, filename):
    """
    Trajectory overlay on SW-only Gor'kov potential.

    This is the primary diagnostic output — shows trap lattice,
    A particle path (red), B particle (blue), and all neighbours.
    """
    traj = sim["trajectories"]
    U_sw = sim["U_sw_only"]
    xg_mm = xg * 1e3
    yg_mm = yg * 1e3
    ext = [xg_mm[0], xg_mm[-1], yg_mm[0], yg_mm[-1]]

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    im = ax.imshow(U_sw, origin="lower", extent=ext, cmap="RdBu_r",
                   aspect="equal")
    _cbar(ax, im, "U_gor (J) [SW only]")

    # Trap centres
    for i, t in enumerate(traps_m):
        ax.plot(t[0]*1e3, t[1]*1e3, "+", color="grey", ms=8, mew=1, zorder=3)

    colors = plt.cm.Set1(np.linspace(0, 1, len(traps_m)))
    for ip in range(len(traps_m)):
        path = traj[:, ip, :] * 1e3
        if ip == idx_A:
            ax.plot(path[:, 0], path[:, 1], "-o", color="red", ms=2,
                    lw=1.5, label=f"A ({ip})", zorder=9)
            ax.plot(path[0, 0], path[0, 1], "r^", ms=10, zorder=10)
            ax.plot(path[-1, 0], path[-1, 1], "rv", ms=10, zorder=10)
        elif ip == idx_B:
            ax.plot(path[:, 0], path[:, 1], "-s", color="blue", ms=2,
                    lw=1.5, label=f"B ({ip})", zorder=9)
            ax.plot(path[0, 0], path[0, 1], "b^", ms=10, zorder=10)
            ax.plot(path[-1, 0], path[-1, 1], "bv", ms=10, zorder=10)
        else:
            ax.plot(path[:, 0], path[:, 1], "-", color=colors[ip],
                    lw=0.8, alpha=0.7, zorder=5)
            ax.plot(path[0, 0], path[0, 1], "^", color=colors[ip],
                    ms=5, zorder=6)
            ax.plot(path[-1, 0], path[-1, 1], "v", color=colors[ip],
                    ms=5, zorder=6)

    # Capture radius circle around B
    theta_c = np.linspace(0, 2*np.pi, 100)
    ax.plot(traps_m[idx_B][0]*1e3 + CAPTURE_RADIUS*1e3*np.cos(theta_c),
            traps_m[idx_B][1]*1e3 + CAPTURE_RADIUS*1e3*np.sin(theta_c),
            "--", color="green", lw=0.8, alpha=0.6, label="capture radius")

    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return FIG_DIR / filename


def plot_schedule_diagnostic(alpha, beta_sw_min, ramp_up, hold, ramp_down, relax):
    """Plot example schedule for the study."""
    sched, t_total = make_schedule(beta_sw_min, alpha, ramp_up, hold, ramp_down, relax)
    ts = np.linspace(0, t_total, 500)
    bsw = np.array([sched(t)[0] for t in ts])
    blens = np.array([sched(t)[1] for t in ts])

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(ts * 1e3, bsw, "b-", lw=2, label="β_sw")
    ax.plot(ts * 1e3, blens, "r-", lw=2, label="β_lens (= α·envelope)")
    ax.axvline((ramp_up)*1e3, color="grey", ls=":", lw=0.8)
    ax.axvline((ramp_up+hold)*1e3, color="grey", ls=":", lw=0.8)
    ax.axvline((ramp_up+hold+ramp_down)*1e3, color="grey", ls=":", lw=0.8)
    ax.text((ramp_up/2)*1e3, 0.05, "ramp\nup", ha="center", fontsize=7, color="grey")
    ax.text((ramp_up+hold/2)*1e3, 0.05, "hold", ha="center", fontsize=7, color="grey")
    ax.text((ramp_up+hold+ramp_down/2)*1e3, 0.05, "ramp\ndown", ha="center", fontsize=7, color="grey")
    ax.text((ramp_up+hold+ramp_down+relax/2)*1e3, 0.05, "RELAX\n(pure SW)", ha="center", fontsize=7, color="green")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Weight")
    ax.set_title(f"Schedule: α={alpha}, β_sw_min={beta_sw_min}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"schedule_a{alpha}_bsw{beta_sw_min}.png",
                dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_distance_vs_time(sim, traps_m, idx_A, idx_B, title, filename):
    """d_AB and displacements over time."""
    traj = sim["trajectories"]
    times_ms = sim["times"] * 1e3
    initial = sim["initial_pos"]

    d_AB = np.linalg.norm(traj[:, idx_A] - traj[:, idx_B], axis=1) * 1e6
    d_A = np.linalg.norm(traj[:, idx_A] - initial[idx_A], axis=1) * 1e6
    d_B = np.linalg.norm(traj[:, idx_B] - initial[idx_B], axis=1) * 1e6

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax = axes[0]
    ax.plot(times_ms, d_AB, "k-", lw=1.5, label="d(A,B)")
    ax.axhline(CAPTURE_RADIUS * 1e6, color="green", ls="--", lw=1,
               label="capture radius")
    ax.set_ylabel("d_AB (µm)")
    ax.legend(fontsize=8)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(times_ms, d_A, "r-", lw=1.5, label="A disp")
    ax.plot(times_ms, d_B, "b-", lw=1.5, label="B disp")

    # Also plot neighbour displacements
    n_particles = len(traps_m)
    for ip in range(n_particles):
        if ip == idx_A or ip == idx_B:
            continue
        d_n = np.linalg.norm(traj[:, ip] - initial[ip], axis=1) * 1e6
        ax.plot(times_ms, d_n, "-", color="grey", lw=0.6, alpha=0.5)

    ax.axhline(NEIGHBOUR_TOL * 1e6, color="orange", ls="--", lw=1,
               label="neighbour tol")
    ax.set_ylabel("Disp from init (µm)")
    ax.set_xlabel("Time (ms)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_lens_footprint_comparison(lens_fields, xg, yg, traps_m, idx_A, idx_B):
    """Plot spatial footprints of all lens diameters side by side."""
    n = len(LENS_RADII_MM)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
    axes = axes[0]

    xg_mm = xg * 1e3
    yg_mm = yg * 1e3
    ext = [xg_mm[0], xg_mm[-1], yg_mm[0], yg_mm[-1]]

    vmax = max(np.abs(pf).max() for pf in lens_fields.values())

    for vi, d_mm in enumerate(LENS_RADII_MM):
        ax = axes[vi]
        amp = np.abs(lens_fields[d_mm])
        im = ax.imshow(amp, origin="lower", extent=ext, cmap="inferno",
                       aspect="equal", vmin=0, vmax=vmax)
        ax.set_title(f"D = {d_mm} mm\n|p|_max = {amp.max():.2f} Pa", fontsize=9)

        for i, t in enumerate(traps_m):
            c = "red" if i == idx_A else "blue" if i == idx_B else "white"
            ax.plot(t[0]*1e3, t[1]*1e3, "+", color=c, ms=8, mew=1.5)

        ax.set_xlabel("x (mm)")
        if vi == 0:
            ax.set_ylabel("y (mm)")

    fig.suptitle("Vortex Lens Field Footprints — Aperture Diameter Sweep",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "footprint_comparison.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print("[save] footprint_comparison.png")


def plot_sweep_heatmap(all_results):
    """Heatmap of d_AB and max neighbour disp across parameter space."""
    for d_mm in LENS_RADII_MM:
        sub = [r for r in all_results if r["lens_radius"] == d_mm]
        if not sub:
            continue

        bsw_vals = sorted(set(r["beta_sw_min"] for r in sub))
        a_vals = sorted(set(r["alpha"] for r in sub))

        mat_dab = np.full((len(a_vals), len(bsw_vals)), np.nan)
        mat_neigh = np.full((len(a_vals), len(bsw_vals)), np.nan)

        for r in sub:
            ai = a_vals.index(r["alpha"])
            bi = bsw_vals.index(r["beta_sw_min"])
            mat_dab[ai, bi] = r["metrics"]["d_AB_final_um"]
            mat_neigh[ai, bi] = r["metrics"]["max_neighbour_disp_um"]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # d_AB heatmap
        ax = axes[0]
        im = ax.imshow(mat_dab, origin="lower", aspect="auto", cmap="RdYlGn_r")
        ax.set_xticks(range(len(bsw_vals)))
        ax.set_xticklabels([f"{v:.1f}" for v in bsw_vals])
        ax.set_yticks(range(len(a_vals)))
        ax.set_yticklabels([f"{v:.1f}" for v in a_vals])
        ax.set_xlabel("β_sw_min")
        ax.set_ylabel("α")
        ax.set_title(f"d_AB final (µm) — D={d_mm}mm")
        for ai in range(len(a_vals)):
            for bi in range(len(bsw_vals)):
                val = mat_dab[ai, bi]
                if not np.isnan(val):
                    m = [r for r in sub
                         if r["alpha"] == a_vals[ai]
                         and r["beta_sw_min"] == bsw_vals[bi]][0]
                    star = "★" if m["metrics"]["A_in_capture_region"] else ""
                    ax.text(bi, ai, f"{val:.0f}{star}",
                            ha="center", va="center", fontsize=7,
                            color="white" if val > 200 else "black")
        _cbar(ax, im, "d_AB (µm)")

        # max neighbour heatmap
        ax = axes[1]
        im = ax.imshow(mat_neigh, origin="lower", aspect="auto", cmap="RdYlGn_r")
        ax.set_xticks(range(len(bsw_vals)))
        ax.set_xticklabels([f"{v:.1f}" for v in bsw_vals])
        ax.set_yticks(range(len(a_vals)))
        ax.set_yticklabels([f"{v:.1f}" for v in a_vals])
        ax.set_xlabel("β_sw_min")
        ax.set_ylabel("α")
        ax.set_title(f"Max neighbour disp (µm) — D={d_mm}mm")
        for ai in range(len(a_vals)):
            for bi in range(len(bsw_vals)):
                val = mat_neigh[ai, bi]
                if not np.isnan(val):
                    ax.text(bi, ai, f"{val:.0f}",
                            ha="center", va="center", fontsize=7,
                            color="white" if val > 100 else "black")
        _cbar(ax, im, "max N disp (µm)")

        fig.tight_layout()
        fig.savefig(FIG_DIR / f"heatmap_D{d_mm}mm.png", dpi=200,
                    bbox_inches="tight")
        plt.close(fig)
        print(f"[save] heatmap_D{d_mm}mm.png")


def plot_comparison_bars(all_results):
    """Bar chart of all candidates sorted by composite score."""
    ranked = sorted(all_results,
                    key=lambda r: (
                        not r["metrics"]["A_in_capture_region"],
                        r["metrics"]["max_neighbour_disp_um"]))

    n = min(len(ranked), 20)
    ranked = ranked[:n]

    labels = []
    d_abs = []
    max_ns = []
    merged = []
    for r in ranked:
        labels.append(f"D{r['lens_radius']}\nα{r['alpha']}\nβ{r['beta_sw_min']}")
        d_abs.append(r["metrics"]["d_AB_final_um"])
        max_ns.append(r["metrics"]["max_neighbour_disp_um"])
        merged.append(r["metrics"]["A_in_capture_region"])

    colours = ["green" if m else "salmon" for m in merged]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    ax = axes[0]
    ax.bar(range(n), d_abs, color=colours, edgecolor="grey", linewidth=0.5)
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, fontsize=6, rotation=45, ha="right")
    ax.set_ylabel("d_AB final (µm)")
    ax.set_title("Final A–B Separation (green = merged)")
    ax.axhline(CAPTURE_RADIUS * 1e6, color="green", ls="--", lw=1)
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    ax.bar(range(n), max_ns, color=colours, edgecolor="grey", linewidth=0.5)
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, fontsize=6, rotation=45, ha="right")
    ax.set_ylabel("Max neighbour disp (µm)")
    ax.set_title("Max Neighbour Displacement (green = merged)")
    ax.axhline(NEIGHBOUR_TOL * 1e6, color="orange", ls="--", lw=1)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "sweep_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[save] sweep_comparison.png")


def render_gif(sim, xg, yg, traps_m, idx_A, idx_B, filename):
    """Render GIF for a simulation result (requires stored fields)."""
    traj = sim["trajectories"]
    times_ms = sim["times"] * 1e3
    betas_lens = sim["betas_lens"]
    U_frames = sim["U_frames"]
    Fx_frames = sim["Fx_frames"]
    Fy_frames = sim["Fy_frames"]

    if U_frames is None or len(U_frames) == 0:
        print(f"[gif] No stored fields for {filename}, skipping")
        return None

    n_frames = len(U_frames)
    print(f"\n[gif] Rendering {n_frames} frames for {filename}...")

    U_all_min = min(U.min() for U in U_frames)
    U_all_max = max(U.max() for U in U_frames)

    frames_images = []
    xg_mm = xg * 1e3
    yg_mm = yg * 1e3
    ext = [xg_mm[0], xg_mm[-1], yg_mm[0], yg_mm[-1]]

    for fi in range(n_frames):
        fig, ax = plt.subplots(1, 1, figsize=(7, 6))

        im = ax.imshow(U_frames[fi], origin="lower", extent=ext,
                       cmap="RdBu_r", aspect="equal",
                       vmin=U_all_min, vmax=U_all_max)

        for i, t in enumerate(traps_m):
            ax.plot(t[0]*1e3, t[1]*1e3, "+", color="0.6", ms=5,
                    mew=0.5, zorder=3)

        if fi > 0:
            tA = traj[:fi+1, idx_A, :] * 1e3
            ax.plot(tA[:, 0], tA[:, 1], "-", color="red", lw=1.0,
                    alpha=0.5, zorder=6)
            tB = traj[:fi+1, idx_B, :] * 1e3
            ax.plot(tB[:, 0], tB[:, 1], "-", color="blue", lw=0.8,
                    alpha=0.5, zorder=6)

        positions = traj[fi]
        for ip in range(len(positions)):
            x_mm, y_mm = positions[ip] * 1e3
            if ip == idx_A:
                ax.plot(x_mm, y_mm, "o", color="red", ms=8, mew=1.5,
                        mfc="red", mec="white", zorder=10)
                ax.annotate("A", (x_mm, y_mm), fontsize=9, color="red",
                            fontweight="bold", ha="center", va="bottom",
                            xytext=(0, 6), textcoords="offset points",
                            zorder=11)
            elif ip == idx_B:
                ax.plot(x_mm, y_mm, "s", color="blue", ms=8, mew=1.5,
                        mfc="blue", mec="white", zorder=10)
                ax.annotate("B", (x_mm, y_mm), fontsize=9, color="blue",
                            fontweight="bold", ha="center", va="bottom",
                            xytext=(0, 6), textcoords="offset points",
                            zorder=11)
            else:
                ax.plot(x_mm, y_mm, "o", color="lime", ms=5, mec="white",
                        mew=0.5, zorder=8)

        skip = 25
        Fx_sub = Fx_frames[fi][::skip, ::skip]
        Fy_sub = Fy_frames[fi][::skip, ::skip]
        mag = np.sqrt(Fx_sub**2 + Fy_sub**2)
        mag_max = mag.max()
        if mag_max > 0:
            ax.quiver(xg_mm[::skip], yg_mm[::skip],
                      Fx_sub / mag_max, Fy_sub / mag_max,
                      color="white", alpha=0.25, scale=30,
                      width=0.003, zorder=4)

        _cbar(ax, im, "U_gor (J)")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        bsw = sim["betas_sw"][fi] if fi < len(sim["betas_sw"]) else 1.0
        ax.set_title(
            f"t = {times_ms[fi]:.1f} ms  |  β_sw = {bsw:.2f}  "
            f"β_lens = {betas_lens[fi]:.2f}")
        fig.tight_layout()

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(),
                            dtype=np.uint8).reshape(h, w, 4)
        frames_images.append(buf.copy())
        plt.close(fig)

        if (fi + 1) % 50 == 0:
            print(f"  rendered {fi+1}/{n_frames}")

    from PIL import Image
    pil_frames = [Image.fromarray(f) for f in frames_images]
    gif_path = GIF_DIR / filename
    pil_frames[0].save(
        gif_path, save_all=True, append_images=pil_frames[1:],
        duration=80, loop=0, optimize=False)
    print(f"[save] {gif_path.name} ({len(pil_frames)} frames)")
    return gif_path


# ═══════════════════════════════════════════════════════════════════
# Output helpers
# ═══════════════════════════════════════════════════════════════════

def save_config(all_results):
    cfg = {
        "timestamp": TS,
        "study": "vortex_transport_localisation_study",
        "overlay_npz": str(OVERLAY_NPZ),
        "lens_plane_npz": str(LENS_PLANE_NPZ),
        "model": ("crossfade: p = beta_sw*p_sw "
                   "+ alpha*beta_vortex*exp(i*psi)*p_lens"),
        "psi_over_pi": PSI / np.pi,
        "lens_radii_mm": LENS_RADII_MM,
        "beta_sw_min_values": BETA_SW_MIN_VALUES,
        "alpha_values": ALPHA_VALUES,
        "T_ramp_up_ms": T_RAMP_UP * 1e3,
        "T_hold_ms": T_HOLD * 1e3,
        "T_ramp_down_ms": T_RAMP_DOWN * 1e3,
        "T_relax_ms": T_RELAX * 1e3,
        "dt_us": DT * 1e6,
        "P_SCALE": P_SCALE,
        "n_candidates": len(all_results),
        "capture_radius_um": CAPTURE_RADIUS * 1e6,
        "trap_tolerance_um": TRAP_TOLERANCE * 1e6,
        "neighbour_tolerance_um": NEIGHBOUR_TOL * 1e6,
    }
    with open(OUT_DIR / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print("[save] config.json")


def save_metrics_csv(all_results):
    """Save all candidate results to CSV."""
    if not all_results:
        return
    fieldnames = (["lens_radius", "beta_sw_min", "alpha", "ramp_label",
                   "translation_mode"]
                  + list(all_results[0]["metrics"].keys()))
    with open(OUT_DIR / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_results:
            row = {
                "lens_radius": r["lens_radius"],
                "beta_sw_min": r["beta_sw_min"],
                "alpha": r["alpha"],
                "ramp_label": r["ramp_label"],
                "translation_mode": r["translation_mode"],
            }
            row.update(r["metrics"])
            w.writerow(row)
    print("[save] metrics.csv")


def write_index(all_results, best_results_top3):
    """Write comprehensive INDEX.md."""
    lines = [
        "# Vortex Transport Localisation Study",
        "",
        f"**Generated:** {TS}",
        "",
        "## Goal",
        "",
        "Move particle A → B while keeping neighbouring particles as",
        "stationary as possible, by controlling:",
        "",
        "1. **Vortex footprint** — lens aperture diameter (physical beam size)",
        "2. **Standing wave reduction** — less aggressive β_sw_min",
        "3. **Vortex strength** — α scaling factor",
        "",
        "## Key Difference from Previous Study",
        "",
        "Previous study used **spatial windowing** (Gaussian envelope) at the",
        "ROI scale to localise the vortex field. This **destroyed the phase",
        "structure** needed for directional transport.",
        "",
        "This study instead changes the **physical lens aperture** — smaller",
        "apertures produce different diffraction patterns at the target plane,",
        "potentially more localised perturbations without breaking the phase.",
        "",
        "## Lens Field Generation",
        "",
        "For each aperture diameter D:",
        "1. Load lens-plane backpropagated field (from manufacturing study)",
        "2. Apply circular aperture mask at radius D/2",
        "3. Create phase-only drive: exp(iφ) within aperture",
        "4. Forward-propagate to target plane via Angular Spectrum Method",
        "5. Interpolate onto ROI grid",
        "",
        "## Schedule (NEW: includes relaxation phase)",
        "",
        "```",
        f"ramp_up:   {T_RAMP_UP*1e3:.0f} ms  (cosine taper)",
        f"hold:      {T_HOLD*1e3:.0f} ms",
        f"ramp_down: {T_RAMP_DOWN*1e3:.0f} ms  (cosine taper)",
        f"relax:     {T_RELAX*1e3:.0f} ms  (pure standing wave — observe trap recovery)",
        "```",
        "",
        "## Sweep Parameters",
        "",
        f"- **Lens diameter:** {LENS_RADII_MM} mm",
        f"- **β_sw_min:** {BETA_SW_MIN_VALUES}",
        f"- **α (vortex strength):** {ALPHA_VALUES}",
        f"- **Ramp:** fast only (200 ms up/down)",
        f"- **ψ:** {PSI/np.pi:.1f}π",
        "",
        f"**Total candidates:** {len(all_results)}",
        "",
    ]

    # Top 10 table
    ranked = sorted(all_results,
                    key=lambda r: (
                        not r["metrics"]["A_in_capture_region"],
                        r["metrics"]["max_neighbour_disp_um"]))

    merged = [r for r in all_results if r["metrics"]["A_in_capture_region"]]
    full_success = [r for r in all_results
                    if r["metrics"]["classification"] == "successful_merge"]

    lines += [
        f"**Achieved merge (A in capture):** {len(merged)}",
        f"**Full success (merge + B stable + neighbours OK):** {len(full_success)}",
        "",
        "## Top 10 Candidates",
        "",
        ("| Rank | D (mm) | α | β_sw_min | d_AB (µm) | A moved (µm) | "
         "max_N (µm) | mean_N (µm) | Class |"),
        ("|------|--------|---|----------|-----------|-------------|"
         "------------|------------|-------|"),
    ]
    for i, r in enumerate(ranked[:10]):
        m = r["metrics"]
        lines.append(
            f"| {i+1} | {r['lens_radius']} | {r['alpha']:.1f} "
            f"| {r['beta_sw_min']:.1f} "
            f"| {m['d_AB_final_um']:.0f} "
            f"| {m['d_A_moved_um']:.0f} "
            f"| {m['max_neighbour_disp_um']:.0f} "
            f"| {m['mean_neighbour_disp_um']:.0f} "
            f"| {m['classification']} |")

    # Best protocol
    if best_results_top3:
        best = best_results_top3[0]
        bm = best["metrics"]
        lines += [
            "",
            "## Best Protocol",
            "",
            f"- **Lens diameter:** {best['lens_radius']} mm",
            f"- **α:** {best['alpha']}",
            f"- **β_sw_min:** {best['beta_sw_min']}",
            f"- **d_AB final:** {bm['d_AB_final_um']:.1f} µm",
            f"- **A moved:** {bm['d_A_moved_um']:.1f} µm",
            f"- **B moved:** {bm['d_B_moved_um']:.1f} µm",
            f"- **Max neighbour:** {bm['max_neighbour_disp_um']:.1f} µm",
            f"- **Mean neighbour:** {bm['mean_neighbour_disp_um']:.1f} µm",
            f"- **Classification:** {bm['classification']}",
        ]
        if bm["merge_time_ms"]:
            lines.append(f"- **Merge time:** {bm['merge_time_ms']:.1f} ms")

    # Analysis by lens diameter
    lines += [
        "",
        "## Analysis by Lens Diameter",
        "",
    ]
    for d_mm in LENS_RADII_MM:
        sub = [r for r in all_results if r["lens_radius"] == d_mm]
        sub_merged = [r for r in sub if r["metrics"]["A_in_capture_region"]]
        if sub_merged:
            avg_maxn = np.mean([r["metrics"]["max_neighbour_disp_um"] for r in sub_merged])
            min_dab = min(r["metrics"]["d_AB_final_um"] for r in sub_merged)
            lines.append(f"- **D={d_mm}mm**: {len(sub_merged)}/{len(sub)} merged, "
                         f"avg max_N = {avg_maxn:.0f} µm, best d_AB = {min_dab:.0f} µm")
        else:
            lines.append(f"- **D={d_mm}mm**: 0/{len(sub)} merged")

    # Trajectory PNGs
    lines += [
        "",
        "## Trajectory Plots",
        "",
        "Each candidate has a trajectory PNG showing trap lattice, A particle",
        "path (red), B particle path (blue), and neighbour movements.",
        "",
    ]
    for i, r in enumerate(ranked[:10]):
        tag = r["tag"]
        lines.append(f"- Rank {i+1}: [trajectory_{tag}.png](figures/trajectory_{tag}.png)")

    # GIFs
    lines += [
        "",
        "## GIFs (Top 3 protocols)",
        "",
    ]
    for i, best in enumerate(best_results_top3):
        lines.append(f"- Best {i+1}: [transport_best_{i+1}.gif](gif/transport_best_{i+1}.gif) "
                     f"— D={best['lens_radius']}mm, α={best['alpha']}, β_sw={best['beta_sw_min']}")

    lines += [
        "",
        "## Files",
        "",
        "- `config.json` — full run parameters",
        "- `metrics.csv` — all candidate results",
        "- `figures/` — trajectory PNGs, heatmaps, diagnostics",
        "- `gif/` — top 3 GIF animations",
        "- `npz/` — simulation data for best candidates",
    ]

    with open(OUT_DIR / "INDEX.md", "w") as f:
        f.write("\n".join(lines))
    print("[save] INDEX.md")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()

    # ── 1. Load standing wave & traps ─────────────────────────────
    (p_sw, xg, yg, traps_m,
     idx_A, idx_B, midpoint, d_AB, e_AB) = load_sw_and_traps()
    trap_A = traps_m[idx_A]
    trap_B = traps_m[idx_B]

    # ── 2. Generate lens fields for each aperture ─────────────────
    lens_fields = generate_lens_fields(xg, yg)

    n_cand = (len(LENS_RADII_MM) * len(BETA_SW_MIN_VALUES)
              * len(ALPHA_VALUES) * len(RAMP_VARIANTS)
              * len(TRANSLATION_MODES))

    print(f"\n{'='*60}")
    print(f"  VORTEX TRANSPORT LOCALISATION STUDY")
    print(f"  Sweep: {len(LENS_RADII_MM)} diameters × "
          f"{len(BETA_SW_MIN_VALUES)} β_sw_min × "
          f"{len(ALPHA_VALUES)} α × "
          f"{len(RAMP_VARIANTS)} ramp × "
          f"{len(TRANSLATION_MODES)} trans = {n_cand} candidates")
    print(f"{'='*60}\n")

    # ── 3. Schedule diagnostics ──────────────────────────────────
    print("=== Schedule Diagnostics ===\n")
    for a in [ALPHA_VALUES[0], ALPHA_VALUES[-1]]:
        for bsw in [BETA_SW_MIN_VALUES[0], BETA_SW_MIN_VALUES[-1]]:
            plot_schedule_diagnostic(a, bsw, T_RAMP_UP, T_HOLD, T_RAMP_DOWN, T_RELAX)
    print("[save] schedule diagnostic plots")

    # ── 4. Lens footprint comparison ──────────────────────────────
    print("\n=== Lens Footprint Comparison ===\n")
    plot_lens_footprint_comparison(lens_fields, xg, yg, traps_m, idx_A, idx_B)

    # ── 5. Parameter sweep ────────────────────────────────────────
    print("\n=== Parameter Sweep ===\n")
    all_results = []
    run_idx = 0

    for d_mm in LENS_RADII_MM:
        p_lens = lens_fields[d_mm]

        for bsw_min in BETA_SW_MIN_VALUES:
            for alpha in ALPHA_VALUES:
                for rv in RAMP_VARIANTS:
                    for tm in TRANSLATION_MODES:
                        run_idx += 1
                        tag = (f"D{d_mm}_a{alpha:.1f}_bsw{bsw_min:.1f}"
                               f"_{rv['label']}_{tm}")
                        print(f"  [{run_idx}/{n_cand}] {tag}", end="")

                        sched, ttot = make_schedule(
                            bsw_min, alpha,
                            rv["ramp_up"], rv["hold"],
                            rv["ramp_down"], T_RELAX)

                        sim = simulate_particles(
                            p_sw, p_lens, xg, yg, traps_m,
                            idx_A, idx_B,
                            schedule_fn=sched, t_total=ttot,
                            store_fields=False)

                        metrics = analyse_success(sim, traps_m, idx_A, idx_B)

                        result = {
                            "lens_radius": d_mm,
                            "beta_sw_min": bsw_min,
                            "alpha": alpha,
                            "ramp_label": rv["label"],
                            "translation_mode": tm,
                            "ramp_up": rv["ramp_up"],
                            "ramp_down": rv["ramp_down"],
                            "metrics": metrics,
                            "tag": tag,
                        }
                        all_results.append(result)

                        # Generate trajectory PNG for every candidate
                        title = (f"D={d_mm}mm  α={alpha}  β_sw_min={bsw_min}\n"
                                 f"d_AB={metrics['d_AB_final_um']:.0f}µm  "
                                 f"max_N={metrics['max_neighbour_disp_um']:.0f}µm  "
                                 f"{metrics['classification']}")
                        plot_trajectory(
                            sim, xg, yg, traps_m, idx_A, idx_B,
                            title, f"trajectory_{tag}.png")

                        cls_short = metrics["classification"][:4]
                        print(f" → {cls_short}, d_AB={metrics['d_AB_final_um']:.0f}µm, "
                              f"max_N={metrics['max_neighbour_disp_um']:.0f}µm")

    # ── 6. Identify top 3 candidates ──────────────────────────────
    # Rank by: merged first, then lowest max neighbour displacement
    merged = [r for r in all_results if r["metrics"]["A_in_capture_region"]]
    if merged:
        ranked_merged = sorted(merged,
                               key=lambda r: r["metrics"]["max_neighbour_disp_um"])
        # If we have fewer than 3 merged, fill from best partial
        top3_merged = ranked_merged[:3]
    else:
        # No merges: rank by lowest d_AB + lowest neighbour disp
        top3_merged = sorted(all_results,
                             key=lambda r: (r["metrics"]["d_AB_final_um"]
                                            + r["metrics"]["max_neighbour_disp_um"]))[:3]

    print(f"\n{'='*60}")
    print(f"  TOP 3 CANDIDATES")
    for i, best in enumerate(top3_merged):
        bm = best["metrics"]
        print(f"  #{i+1}: {best['tag']}  → {bm['classification']}, "
              f"d_AB={bm['d_AB_final_um']:.1f}µm, "
              f"max_N={bm['max_neighbour_disp_um']:.1f}µm")
    print(f"{'='*60}\n")

    # ── 7. Re-run top 3 with field storage for GIFs ──────────────
    print("=== Re-running top 3 with field storage ===\n")
    top3_sims = []
    for i, best in enumerate(top3_merged):
        print(f"  Re-running #{i+1}: {best['tag']}")
        p_lens = lens_fields[best["lens_radius"]]
        sched, ttot = make_schedule(
            best["beta_sw_min"], best["alpha"],
            best["ramp_up"], best.get("hold", T_HOLD),
            best["ramp_down"], T_RELAX)
        sim = simulate_particles(
            p_sw, p_lens, xg, yg, traps_m,
            idx_A, idx_B,
            schedule_fn=sched, t_total=ttot,
            store_fields=True)
        top3_sims.append(sim)

        # Distance-vs-time plot for top 3
        plot_distance_vs_time(
            sim, traps_m, idx_A, idx_B,
            f"Top {i+1}: {best['tag']}",
            f"distance_vs_time_best_{i+1}.png")

    # ── 8. Render GIFs for top 3 ─────────────────────────────────
    print("\n=== Rendering GIFs for top 3 ===\n")
    for i, (best, sim) in enumerate(zip(top3_merged, top3_sims)):
        render_gif(sim, xg, yg, traps_m, idx_A, idx_B,
                   f"transport_best_{i+1}.gif")

    # ── 9. Diagnostic plots ──────────────────────────────────────
    print("\n=== Diagnostic Plots ===\n")
    plot_sweep_heatmap(all_results)
    plot_comparison_bars(all_results)

    # ── 10. Save outputs ──────────────────────────────────────────
    print("\n=== Saving Outputs ===\n")
    save_config(all_results)
    save_metrics_csv(all_results)

    # Save best simulation data
    for i, (best, sim) in enumerate(zip(top3_merged, top3_sims)):
        np.savez_compressed(
            NPZ_DIR / f"best_{i+1}_simulation.npz",
            trajectories=sim["trajectories"],
            times=sim["times"],
            betas_sw=sim["betas_sw"],
            betas_lens=sim["betas_lens"],
            initial_pos=sim["initial_pos"],
            U_sw_only=sim["U_sw_only"],
            xg=xg, yg=yg,
            traps_m=traps_m, idx_A=idx_A, idx_B=idx_B,
            tag=best["tag"],
        )
        print(f"[save] best_{i+1}_simulation.npz ({best['tag']})")

    write_index(all_results, top3_merged)

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  Total runtime: {elapsed:.1f}s")
    print(f"  Output: {OUT_DIR}")
    if top3_merged:
        print(f"  Best: {top3_merged[0]['tag']} → {top3_merged[0]['metrics']['classification']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
