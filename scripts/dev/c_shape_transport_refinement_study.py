#!/usr/bin/env python3
"""
C-Shape Transport Refinement Study
====================================

Investigates whether a **smaller, translated C-shape perturbation** with
**partial standing-wave reduction** gives a better merge result than the
Phase 1 baseline crossfade protocol.

Key change from previous attempt (20260310_212501):
  Analytic C-shape fields could NOT achieve merge — only the real
  backpropagated lens field had the correct amplitude/phase structure.
  This study therefore keeps the real lens field but applies a
  **Gaussian spatial window** to make it physically smaller, and
  **translates the window centre** from A toward B during the protocol.

Model:
  p_comb(t) = β_sw(t) · p_sw + β_lens(t) · exp(iψ) · W(x,y; c(t)) · p_lens

where W is a 2D Gaussian spatial window centred at c(t).

Parts:
  A — Smaller C-shape variants: Gaussian window radii sweep
  B — Translation: window centre moves A→B during active phase
  C — Partial SW reduction: β_sw_min sweep
  D — Control schedule: smooth ramps + translation
  E — Particle dynamics: 9-particle overdamped simulation
  F — Success criteria & comparison with baseline
  G — Visual outputs: trajectory plots, GIF, comparison bars

Outputs → results/c_shape_transport_refinement_study_<TS>/
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

# ═══════════════════════════════════════════════════════════════════
# Pressure scaling & control params
# ═══════════════════════════════════════════════════════════════════
P_SCALE = 3.0e3            # → |p_sw|_eff ≈ 140 kPa
ALPHA_TARGET = 5.0
PSI = 3.0 * np.pi / 2.0

T_RAMP_UP   = 0.200       # 200 ms
T_HOLD      = 0.100       # 100 ms
T_RAMP_DOWN = 0.200       # 200 ms
T_SETTLE    = 0.300       # 300 ms
T_TOTAL     = T_RAMP_UP + T_HOLD + T_RAMP_DOWN + T_SETTLE
DT          = 1.0e-4      # 0.1 ms
N_FRAMES    = 150

CAPTURE_RADIUS = 0.30 * TRAP_SP
TRAP_TOLERANCE = 0.50 * TRAP_SP
NEIGHBOUR_TOL  = 0.50 * TRAP_SP

# ═══════════════════════════════════════════════════════════════════
# Parameter sweep
# ═══════════════════════════════════════════════════════════════════

# Part A: spatial window radii (in units of d_AB)
# "full" means no windowing (baseline lens behaviour)
# INSIGHT from first sweep: spatial windowing breaks the lens phase
# structure and prevents merge. Instead, we keep full lens and focus
# on schedule refinement.
WINDOW_RADII_REL = [0.60, 1.00, None]
WINDOW_LABELS = ["medium_0.60", "wider_1.00", "full_lens"]

# Part B: translation mode
TRANSLATION_MODES = ["static"]

# Part C: minimum SW scaling
BETA_SW_MIN_VALUES = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6]

# Part extra: alpha boost for windowed variants
ALPHA_BY_WINDOW = {
    "medium_0.60": 12.0,
    "wider_1.00": 6.0,
    "full_lens": 5.0,
}

# Part D-extra: ramp time variants
# The baseline uses 200ms ramps — we also test slower
RAMP_VARIANTS = [
    {"label": "fast", "ramp_up": 0.200, "hold": 0.100, "ramp_down": 0.200, "settle": 0.300},
    {"label": "slow", "ramp_up": 0.400, "hold": 0.100, "ramp_down": 0.400, "settle": 0.300},
]

# ═══════════════════════════════════════════════════════════════════
# Output
# ═══════════════════════════════════════════════════════════════════
TS = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = PROJECT_ROOT / "results" / f"c_shape_transport_refinement_study_{TS}"
FIG_DIR = OUT_DIR / "figures"
NPZ_DIR = OUT_DIR / "npz"
GIF_DIR = OUT_DIR / "gif"
for d in [OUT_DIR, FIG_DIR, NPZ_DIR, GIF_DIR]:
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

    d_AB = np.linalg.norm(traps_m[idx_A] - traps_m[idx_B])
    e_AB = (traps_m[idx_B] - traps_m[idx_A]) / d_AB
    print(f"  d_AB = {d_AB*1e6:.1f} µm, e_AB = {e_AB}")

    return p_sw, p_lens_roi, xg, yg, traps_m, idx_A, idx_B, midpoint, d_AB, e_AB


# ═══════════════════════════════════════════════════════════════════
# Spatial windowing — makes the lens field more local
# ═══════════════════════════════════════════════════════════════════

def apply_spatial_window(p_lens, xg, yg, centre, sigma):
    """
    Apply a 2D Gaussian spatial window to the lens field.

    W(x,y) = exp(-((x-cx)² + (y-cy)²) / (2σ²))

    This preserves the lens phase structure but attenuates it
    away from the window centre, making the perturbation footprint
    smaller and more local.

    Parameters
    ----------
    p_lens : (Ny, Nx) complex array
    xg, yg : 1D coordinate arrays
    centre : (2,) array — window centre [cx, cy] in metres
    sigma : float — window radius in metres

    Returns
    -------
    p_windowed : (Ny, Nx) complex array
    W : (Ny, Nx) float array — the window itself
    """
    XX, YY = np.meshgrid(xg, yg)
    r2 = (XX - centre[0])**2 + (YY - centre[1])**2
    W = np.exp(-r2 / (2.0 * sigma**2))
    return p_lens * W, W


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
# Control schedule with partial SW and translation
# ═══════════════════════════════════════════════════════════════════

def make_schedule(beta_sw_min, translation_mode, trap_A, trap_B, d_AB,
                  alpha=None, ramp_up=None, hold=None, ramp_down=None,
                  settle=None):
    """
    Return a schedule function (t) → (beta_sw, beta_lens, centre)
    and total time.
    """
    if alpha is None:
        alpha = ALPHA_TARGET
    if ramp_up is None:
        ramp_up = T_RAMP_UP
    if hold is None:
        hold = T_HOLD
    if ramp_down is None:
        ramp_down = T_RAMP_DOWN
    if settle is None:
        settle = T_SETTLE
    t_total = ramp_up + hold + ramp_down + settle
    midpoint = 0.5 * (trap_A + trap_B)
    c_start = trap_A + 0.10 * (trap_B - trap_A)
    c_end   = trap_A + 0.90 * (trap_B - trap_A)

    def schedule(t):
        if t < 0:
            return 1.0, 0.0, midpoint
        elif t < ramp_up:
            frac = 0.5 * (1.0 - np.cos(np.pi * t / ramp_up))
            beta_sw = 1.0 - (1.0 - beta_sw_min) * frac
            beta_lens = alpha * frac
            t_frac = frac  # for translation
        elif t < ramp_up + hold:
            beta_sw = beta_sw_min
            beta_lens = alpha
            t_frac = 1.0
        elif t < ramp_up + hold + ramp_down:
            t_rd = t - ramp_up - hold
            frac = 0.5 * (1.0 - np.cos(np.pi * t_rd / ramp_down))
            beta_sw = beta_sw_min + (1.0 - beta_sw_min) * frac
            beta_lens = alpha * (1.0 - frac)
            t_frac = 1.0
        else:
            return 1.0, 0.0, midpoint

        # --- centre translation ---
        if translation_mode == "static":
            centre = midpoint
        else:
            t_active_total = ramp_up + hold + ramp_down
            s = np.clip(t / t_active_total, 0, 1)
            s_smooth = 0.5 * (1.0 - np.cos(np.pi * s))
            centre = c_start + s_smooth * (c_end - c_start)

        return beta_sw, beta_lens, centre

    return schedule, t_total


# ═══════════════════════════════════════════════════════════════════
# Bilinear Gorkov decomposition (key optimisation)
# ═══════════════════════════════════════════════════════════════════
#
# Since p_comb = β1·p_sw + β2·p_lens2  (where p_lens2 = exp(iψ)·W·p_lens),
# the Gorkov potential U ∝ |p|² and |∇p|² terms expand bilinearly:
#
#   U(p_comb) = β1²·U_11 + β2²·U_22 + 2·β1·β2·U_12
#
# where U_11 = U(p_sw), U_22 = U(p_lens2), and U_12 uses cross-terms.
# Similarly for forces.  This lets us precompute 3 basis force fields
# once and just blend scalars at each timestep — orders of magnitude
# faster than recomputing 400×400 gradients at every step.

def precompute_bilinear_basis(p1, p2, dx):
    """
    Precompute Gorkov bilinear basis fields.

    Returns (Fx_11, Fy_11), (Fx_22, Fy_22), (Fx_12, Fy_12), U_11, U_22, U_12
    """
    # Self terms
    U_11 = compute_gorkov(p1, dx)
    U_22 = compute_gorkov(p2, dx)

    # Cross term:
    # |p1+p2|² = |p1|² + |p2|² + 2 Re(p1*·p2)
    # So cross in Gorkov = coeff_p·Re(p1*·p2) - coeff_k·Re(∇p1*·∇p2)
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
# Particle dynamics
# ═══════════════════════════════════════════════════════════════════

def simulate_particles(p_sw, p_lens_roi, xg, yg, traps_m,
                       idx_A, idx_B, d_AB,
                       window_sigma, schedule_fn, t_total,
                       store_fields=False):
    """
    Simulate overdamped motion of 9 particles under time-varying field.

    Uses bilinear Gorkov decomposition for static-centre runs (fast)
    and direct computation for moving-centre runs.
    """
    dx = xg[1] - xg[0]
    n_particles = len(traps_m)
    n_steps = int(t_total / DT)
    frame_every = max(1, n_steps // N_FRAMES)
    scale = MU_STOKES * GORKOV_PREFACTOR * P_SCALE**2

    pos = traps_m.copy()
    initial_pos = traps_m.copy()
    midpoint = 0.5 * (traps_m[idx_A] + traps_m[idx_B])

    trajectories = [pos.copy()]
    times = [0.0]
    alphas = [0.0]
    betas_sw = [1.0]
    centres = [midpoint.copy()]

    U_frames = [] if store_fields else None
    Fx_frames = [] if store_fields else None
    Fy_frames = [] if store_fields else None

    merge_time = None

    # Detect if schedule uses static centre (check first and mid-point)
    _, _, c0 = schedule_fn(0.0)
    _, _, c1 = schedule_fn(T_RAMP_UP * 0.5)
    is_static_centre = np.allclose(c0, c1) or window_sigma is None

    if is_static_centre:
        # --- Bilinear fast path ---
        if window_sigma is not None:
            p_lens2, _ = apply_spatial_window(
                p_lens_roi, xg, yg, midpoint, window_sigma)
        else:
            p_lens2 = p_lens_roi.copy()
        p_lens2 = np.exp(1j * PSI) * p_lens2

        (Fx_11, Fy_11, Fx_22, Fy_22, Fx_12, Fy_12,
         U_11, U_22, U_12) = precompute_bilinear_basis(p_sw, p_lens2, dx)

        # Build interpolators for bilinear basis
        iFx_11, iFy_11 = make_force_interpolators(Fx_11, Fy_11, xg, yg)
        iFx_22, iFy_22 = make_force_interpolators(Fx_22, Fy_22, xg, yg)
        iFx_12, iFy_12 = make_force_interpolators(Fx_12, Fy_12, xg, yg)

        for step in range(n_steps):
            t = step * DT
            beta_sw_t, beta_lens_t, centre_t = schedule_fn(t)
            b1 = beta_sw_t
            b2 = beta_lens_t

            # Interpolate basis forces at particle positions
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
                alphas.append(beta_lens_t)
                betas_sw.append(beta_sw_t)
                centres.append(centre_t.copy() if isinstance(centre_t, np.ndarray)
                               else np.array(centre_t))
                if store_fields:
                    U = b1**2 * U_11 + b2**2 * U_22 + 2 * b1 * b2 * U_12
                    Fx = b1**2 * Fx_11 + b2**2 * Fx_22 + 2 * b1 * b2 * Fx_12
                    Fy = b1**2 * Fy_11 + b2**2 * Fy_22 + 2 * b1 * b2 * Fy_12
                    U_frames.append(U)
                    Fx_frames.append(Fx)
                    Fy_frames.append(Fy)

    else:
        # --- Moving centre: precompute at N_KEYFRAMES positions ---
        N_KF = 12
        kf_fracs = np.linspace(0, 1, N_KF)
        kf_times = kf_fracs * (T_RAMP_UP + T_HOLD + T_RAMP_DOWN)
        basis_cache = []
        for kf_t in kf_times:
            _, _, c_kf = schedule_fn(kf_t)
            if window_sigma is not None:
                p_lens2, _ = apply_spatial_window(
                    p_lens_roi, xg, yg, c_kf, window_sigma)
            else:
                p_lens2 = p_lens_roi.copy()
            p_lens2 = np.exp(1j * PSI) * p_lens2
            basis = precompute_bilinear_basis(p_sw, p_lens2, dx)
            basis_cache.append(basis)

        for step in range(n_steps):
            t = step * DT
            beta_sw_t, beta_lens_t, centre_t = schedule_fn(t)
            b1 = beta_sw_t
            b2 = beta_lens_t

            # Find nearest keyframe (or interpolate between two)
            t_active = T_RAMP_UP + T_HOLD + T_RAMP_DOWN
            if t < t_active and b2 > 0:
                frac = t / t_active
                idx_f = frac * (N_KF - 1)
                idx_lo = int(np.floor(idx_f))
                idx_hi = min(idx_lo + 1, N_KF - 1)
                w_hi = idx_f - idx_lo
                w_lo = 1.0 - w_hi

                blo = basis_cache[idx_lo]
                bhi = basis_cache[idx_hi]

                # Interpolate force fields between keyframes
                pts = np.column_stack([pos[:, 1], pos[:, 0]])

                iFx11_lo, iFy11_lo = make_force_interpolators(blo[0], blo[1], xg, yg)
                iFx22_lo, iFy22_lo = make_force_interpolators(blo[2], blo[3], xg, yg)
                iFx12_lo, iFy12_lo = make_force_interpolators(blo[4], blo[5], xg, yg)

                iFx11_hi, iFy11_hi = make_force_interpolators(bhi[0], bhi[1], xg, yg)
                iFx22_hi, iFy22_hi = make_force_interpolators(bhi[2], bhi[3], xg, yg)
                iFx12_hi, iFy12_hi = make_force_interpolators(bhi[4], bhi[5], xg, yg)

                fx = (b1**2 * (w_lo * iFx11_lo(pts) + w_hi * iFx11_hi(pts))
                      + b2**2 * (w_lo * iFx22_lo(pts) + w_hi * iFx22_hi(pts))
                      + 2*b1*b2 * (w_lo * iFx12_lo(pts) + w_hi * iFx12_hi(pts)))
                fy = (b1**2 * (w_lo * iFy11_lo(pts) + w_hi * iFy11_hi(pts))
                      + b2**2 * (w_lo * iFy22_lo(pts) + w_hi * iFy22_hi(pts))
                      + 2*b1*b2 * (w_lo * iFy12_lo(pts) + w_hi * iFy12_hi(pts)))
            else:
                # Settle phase: pure SW
                pts = np.column_stack([pos[:, 1], pos[:, 0]])
                b0 = basis_cache[0]
                iFx_sw, iFy_sw = make_force_interpolators(b0[0], b0[1], xg, yg)
                fx = b1**2 * iFx_sw(pts)
                fy = b1**2 * iFy_sw(pts)

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
                alphas.append(beta_lens_t)
                betas_sw.append(beta_sw_t)
                centres.append(centre_t.copy() if isinstance(centre_t, np.ndarray)
                               else np.array(centre_t))
                if store_fields:
                    if t < t_active and b2 > 0:
                        blo = basis_cache[idx_lo]
                        bhi = basis_cache[idx_hi]
                        U = (b1**2 * (w_lo*blo[6]+w_hi*bhi[6])
                             + b2**2 * (w_lo*blo[7]+w_hi*bhi[7])
                             + 2*b1*b2 * (w_lo*blo[8]+w_hi*bhi[8]))
                        Fx = (b1**2 * (w_lo*blo[0]+w_hi*bhi[0])
                              + b2**2 * (w_lo*blo[2]+w_hi*bhi[2])
                              + 2*b1*b2 * (w_lo*blo[4]+w_hi*bhi[4]))
                        Fy = (b1**2 * (w_lo*blo[1]+w_hi*bhi[1])
                              + b2**2 * (w_lo*blo[3]+w_hi*bhi[3])
                              + 2*b1*b2 * (w_lo*blo[5]+w_hi*bhi[5]))
                    else:
                        U = basis_cache[0][6]
                        Fx = basis_cache[0][0]
                        Fy = basis_cache[0][1]
                    U_frames.append(U)
                    Fx_frames.append(Fx)
                    Fy_frames.append(Fy)

    U_sw_only = compute_gorkov(p_sw, dx)

    return {
        "trajectories": np.array(trajectories),
        "times": np.array(times),
        "alphas": np.array(alphas),
        "betas_sw": np.array(betas_sw),
        "centres": np.array(centres),
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
# Part A — Footprint diagnostic plots
# ═══════════════════════════════════════════════════════════════════

def plot_footprint_comparison(p_lens_roi, xg, yg, traps_m, idx_A, idx_B,
                              d_AB, midpoint):
    """Plot spatial footprints of all window variants side by side."""
    n_variants = len(WINDOW_RADII_REL)
    fig, axes = plt.subplots(1, n_variants, figsize=(4 * n_variants, 4),
                             squeeze=False)
    axes = axes[0]

    xg_mm = xg * 1e3
    yg_mm = yg * 1e3
    ext = [xg_mm[0], xg_mm[-1], yg_mm[0], yg_mm[-1]]

    for vi, (r_rel, label) in enumerate(zip(WINDOW_RADII_REL, WINDOW_LABELS)):
        ax = axes[vi]
        if r_rel is not None:
            sigma = r_rel * d_AB
            p_w, W = apply_spatial_window(p_lens_roi, xg, yg, midpoint, sigma)
        else:
            p_w = p_lens_roi
            W = np.ones_like(np.abs(p_lens_roi))

        amp = np.abs(p_w)
        im = ax.imshow(amp, origin="lower", extent=ext, cmap="inferno",
                       aspect="equal")
        ax.set_title(f"{label}\nσ/d_AB={r_rel if r_rel else '∞'}", fontsize=9)

        # Mark traps
        for i, t in enumerate(traps_m):
            c = "red" if i == idx_A else "blue" if i == idx_B else "white"
            ax.plot(t[0]*1e3, t[1]*1e3, "+", color=c, ms=8, mew=1.5)

        # Mark midpoint
        ax.plot(midpoint[0]*1e3, midpoint[1]*1e3, "x", color="cyan",
                ms=8, mew=2)

        ax.set_xlabel("x (mm)")
        if vi == 0:
            ax.set_ylabel("y (mm)")

    fig.suptitle("Windowed Lens Field Footprints (|p| amplitude)", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "footprint_comparison.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print("[save] footprint_comparison.png")


def plot_footprint_1d_profiles(p_lens_roi, xg, yg, traps_m, idx_A, idx_B,
                               d_AB, midpoint, e_AB):
    """1D profiles along A→B axis for each variant."""
    # Sample along line from A through midpoint to B
    n_pts = 400
    trap_A = traps_m[idx_A]
    trap_B = traps_m[idx_B]
    extension = 0.5 * d_AB  # extend beyond A and B
    p_start = trap_A - extension * e_AB
    p_end = trap_B + extension * e_AB

    ss = np.linspace(0, 1, n_pts)
    pts_x = p_start[0] + ss * (p_end[0] - p_start[0])
    pts_y = p_start[1] + ss * (p_end[1] - p_start[1])
    dist_mm = np.sqrt((pts_x - trap_A[0])**2 + (pts_y - trap_A[1])**2) * 1e3

    fig, ax = plt.subplots(figsize=(8, 4))

    for r_rel, label in zip(WINDOW_RADII_REL, WINDOW_LABELS):
        if r_rel is not None:
            sigma = r_rel * d_AB
            p_w, _ = apply_spatial_window(p_lens_roi, xg, yg, midpoint, sigma)
        else:
            p_w = p_lens_roi

        # Interpolate amplitude along profile
        amp_interp = RegularGridInterpolator(
            (yg, xg), np.abs(p_w), bounds_error=False, fill_value=0.0)
        amp_vals = amp_interp(np.column_stack([pts_y, pts_x]))
        ax.plot(dist_mm, amp_vals, label=label, lw=1.5)

    # Mark trap positions
    ax.axvline(0, color="red", ls="--", lw=0.8, label="Trap A")
    d_AB_mm = d_AB * 1e3
    ax.axvline(d_AB_mm, color="blue", ls="--", lw=0.8, label="Trap B")

    ax.set_xlabel("Distance from A along A→B (mm)")
    ax.set_ylabel("|p_lens| (Pa)")
    ax.set_title("Lens Field Profiles Along A→B — Window Variants")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "footprint_1d_profiles.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print("[save] footprint_1d_profiles.png")


# ═══════════════════════════════════════════════════════════════════
# Part D — Schedule visualisation
# ═══════════════════════════════════════════════════════════════════

def plot_schedule(schedule_fn, label, filename):
    """Plot beta_sw, beta_lens, and centre position over time."""
    ts = np.linspace(0, T_TOTAL, 500)
    bsw_arr, blens_arr, cx_arr, cy_arr = [], [], [], []
    for t in ts:
        bsw, blens, c = schedule_fn(t)
        bsw_arr.append(bsw)
        blens_arr.append(blens)
        cx_arr.append(c[0] * 1e3)
        cy_arr.append(c[1] * 1e3)

    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)

    ax = axes[0]
    ax.plot(ts * 1e3, bsw_arr, "b-", lw=2, label="β_sw")
    ax.plot(ts * 1e3, blens_arr, "r-", lw=2, label="β_lens")
    ax.set_ylabel("Weight")
    ax.legend(fontsize=9)
    ax.set_title(f"Control Schedule — {label}")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(ts * 1e3, cx_arr, "-", lw=1.5, label="cx (mm)")
    ax.plot(ts * 1e3, cy_arr, "-", lw=1.5, label="cy (mm)")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Window centre (mm)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {filename}")


# ═══════════════════════════════════════════════════════════════════
# Part G — Visualisation
# ═══════════════════════════════════════════════════════════════════

def _cbar(ax, im, label=""):
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="4%", pad=0.06)
    plt.colorbar(im, cax=cax, label=label)


def plot_trajectories(sim, xg, yg, traps_m, idx_A, idx_B, title, filename):
    """Trajectory overlay on SW-only Gor'kov potential."""
    traj = sim["trajectories"]
    U_sw = sim["U_sw_only"]
    xg_mm = xg * 1e3
    yg_mm = yg * 1e3
    ext = [xg_mm[0], xg_mm[-1], yg_mm[0], yg_mm[-1]]

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    im = ax.imshow(U_sw, origin="lower", extent=ext, cmap="RdBu_r",
                   aspect="equal")
    _cbar(ax, im, "U_gor (J) [SW only]")

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

    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {filename}")


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
    ax.axhline(TRAP_TOLERANCE * 1e6, color="orange", ls="--", lw=1,
               label="tolerance")
    ax.set_ylabel("Disp from init (µm)")
    ax.set_xlabel("Time (ms)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {filename}")


def plot_displacement_comparison(all_results, idx_A, idx_B, n_traps):
    """Bar chart comparing neighbour displacements: baseline vs best."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: max neighbour displacement across all candidates
    ax = axes[0]
    labels = []
    max_ns = []
    mean_ns = []
    merged = []
    for r in all_results:
        lab = (f"{r['window_label'][:8]}\nβ={r['beta_sw_min']:.1f}"
               f"\n{r.get('ramp_label','fast')[:4]}\n{r['translation'][:3]}")
        labels.append(lab)
        max_ns.append(r["metrics"]["max_neighbour_disp_um"])
        mean_ns.append(r["metrics"]["mean_neighbour_disp_um"])
        merged.append(r["metrics"]["A_in_capture_region"])

    colours = ["green" if m else "salmon" for m in merged]
    ax.bar(range(len(labels)), max_ns, color=colours, edgecolor="grey",
           linewidth=0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=6, rotation=45, ha="right")
    ax.set_ylabel("Max neighbour disp (µm)")
    ax.set_title("Max Neighbour Displacement\n(green = A merged)")
    ax.axhline(NEIGHBOUR_TOL * 1e6, color="orange", ls="--", lw=1)
    ax.axhline(227.3, color="red", ls=":", lw=1, label="baseline max=227µm")
    ax.legend(fontsize=7)

    # Right: d_AB final
    ax = axes[1]
    d_abs = [r["metrics"]["d_AB_final_um"] for r in all_results]
    ax.bar(range(len(labels)), d_abs, color=colours, edgecolor="grey",
           linewidth=0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=6, rotation=45, ha="right")
    ax.set_ylabel("d_AB final (µm)")
    ax.set_title("Final A–B Separation\n(green = merged)")
    ax.axhline(CAPTURE_RADIUS * 1e6, color="green", ls="--", lw=1,
               label="capture radius")
    ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "sweep_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[save] sweep_comparison.png")


def plot_comparison_bars(baseline_metrics, best_metrics, best_label):
    """Side-by-side bars comparing baseline and best refined protocol."""
    keys = ["d_A_moved_um", "d_B_moved_um", "d_AB_final_um",
            "max_neighbour_disp_um", "mean_neighbour_disp_um"]
    labels = ["A moved", "B moved", "d_AB final", "Max N disp", "Mean N disp"]

    base_vals = [baseline_metrics[k] for k in keys]
    best_vals = [best_metrics[k] for k in keys]

    x = np.arange(len(keys))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w/2, base_vals, w, label="Baseline (full crossfade)",
           color="salmon", edgecolor="grey")
    ax.bar(x + w/2, best_vals, w, label=f"Best refined ({best_label})",
           color="steelblue", edgecolor="grey")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("µm")
    ax.set_title("Baseline vs Best Refined Protocol")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "comparison_bars.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[save] comparison_bars.png")


def plot_heatmap(all_results):
    """Heatmap of max neighbour displacement across parameter space."""
    # Group by (window, beta_sw_min) — separate panels per (translation, ramp)
    window_labels_unique = list(dict.fromkeys(
        r["window_label"] for r in all_results))
    bsw_vals = sorted(set(r["beta_sw_min"] for r in all_results))
    trans_modes = sorted(set(r["translation"] for r in all_results))
    ramp_labels = sorted(set(r.get("ramp_label", "fast") for r in all_results))

    for tm in trans_modes:
        for rl in ramp_labels:
            mat_neigh = np.full((len(window_labels_unique), len(bsw_vals)), np.nan)
            mat_merged = np.full((len(window_labels_unique), len(bsw_vals)), False)

            for r in all_results:
                if r["translation"] != tm:
                    continue
                if r.get("ramp_label", "fast") != rl:
                    continue
                wi = window_labels_unique.index(r["window_label"])
                bi = bsw_vals.index(r["beta_sw_min"])
                mat_neigh[wi, bi] = r["metrics"]["max_neighbour_disp_um"]
                mat_merged[wi, bi] = r["metrics"]["A_in_capture_region"]

            fig, ax = plt.subplots(figsize=(6, 4))
            im = ax.imshow(mat_neigh, origin="lower", aspect="auto",
                           cmap="RdYlGn_r")
            ax.set_xticks(range(len(bsw_vals)))
            ax.set_xticklabels([f"{v:.1f}" for v in bsw_vals])
            ax.set_yticks(range(len(window_labels_unique)))
            ax.set_yticklabels(window_labels_unique, fontsize=8)
            ax.set_xlabel("β_sw_min")
            ax.set_ylabel("Window variant")
            ax.set_title(f"Max Neighbour Disp (µm) — {tm}, {rl}\n"
                         "★ = A merged into B")

            for wi in range(len(window_labels_unique)):
                for bi in range(len(bsw_vals)):
                    val = mat_neigh[wi, bi]
                    if not np.isnan(val):
                        star = "★" if mat_merged[wi, bi] else ""
                        ax.text(bi, wi, f"{val:.0f}{star}",
                                ha="center", va="center", fontsize=8,
                                color="white" if val > 150 else "black")

            _cbar(ax, im, "max N disp (µm)")
            fig.tight_layout()
            fig.savefig(FIG_DIR / f"heatmap_{tm}_{rl}.png", dpi=200,
                        bbox_inches="tight")
            plt.close(fig)
            print(f"[save] heatmap_{tm}_{rl}.png")


def render_gif(sim, xg, yg, traps_m, idx_A, idx_B, filename):
    """Render GIF for a simulation result (requires stored fields)."""
    traj = sim["trajectories"]
    times_ms = sim["times"] * 1e3
    alphas = sim["alphas"]
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
            f"β_lens = {alphas[fi]:.2f}")
        fig.tight_layout()

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(),
                            dtype=np.uint8).reshape(h, w, 4)
        frames_images.append(buf.copy())
        plt.close(fig)

        if (fi + 1) % 30 == 0:
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
        "overlay_npz": str(OVERLAY_NPZ),
        "model": ("windowed crossfade: p = beta_sw*p_sw "
                   "+ beta_lens*exp(i*psi)*W(x,y;c(t))*p_lens"),
        "alpha_target": ALPHA_TARGET,
        "alpha_by_window": ALPHA_BY_WINDOW,
        "psi_over_pi": PSI / np.pi,
        "T_ramp_up_ms": T_RAMP_UP * 1e3,
        "T_hold_ms": T_HOLD * 1e3,
        "T_ramp_down_ms": T_RAMP_DOWN * 1e3,
        "T_settle_ms": T_SETTLE * 1e3,
        "T_total_ms": T_TOTAL * 1e3,
        "dt_us": DT * 1e6,
        "P_SCALE": P_SCALE,
        "window_radii_rel": [r if r else "full" for r in WINDOW_RADII_REL],
        "beta_sw_min_values": BETA_SW_MIN_VALUES,
        "translation_modes": TRANSLATION_MODES,
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
    fieldnames = (["window_label", "alpha", "beta_sw_min", "ramp_label", "translation"]
                  + list(all_results[0]["metrics"].keys()))
    with open(OUT_DIR / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_results:
            row = {
                "window_label": r["window_label"],
                "alpha": r.get("alpha", ALPHA_TARGET),
                "beta_sw_min": r["beta_sw_min"],
                "ramp_label": r.get("ramp_label", "fast"),
                "translation": r["translation"],
            }
            row.update(r["metrics"])
            w.writerow(row)
    print("[save] metrics.csv")


def write_index(all_results, baseline_metrics, best_result):
    """Write comprehensive INDEX.md."""
    lines = [
        "# C-Shape Transport Refinement Study",
        "",
        f"**Generated:** {TS}",
        "",
        "## Goal",
        "",
        "Test whether a **spatially windowed, translated lens perturbation**",
        "with **partial standing-wave reduction** gives a better (more selective)",
        "merge result than the Phase 1 baseline full-crossfade protocol.",
        "",
        "## Key Innovation (vs previous attempt 20260310_212501)",
        "",
        "Previous attempt used **analytic C-shape fields** — none achieved merge.",
        "This study uses the **real backpropagated 15 mm lens field** but applies",
        "a **Gaussian spatial window** W(x,y) = exp(-r²/2σ²) to localise it.",
        "This preserves the correct phase/amplitude structure that produces",
        "the asymmetric Gor'kov force while reducing the spatial footprint.",
        "",
        "## Approximation",
        "",
        "The spatial windowing is applied analytically on the ROI grid.",
        "This tests the *control hypothesis* (more local perturbation →",
        "less neighbour disruption) before investing in physical lens redesign.",
        "A real implementation would require a redesigned lens aperture.",
        "",
        "## C-Shape Window Variants (Part A)",
        "",
        "Windowed variants use higher α to compensate for reduced amplitude.",
        "",
        "| Variant | σ/d_AB | α | Description |",
        "|---------|--------|---|-------------|",
    ]
    for r_rel, label in zip(WINDOW_RADII_REL, WINDOW_LABELS):
        desc = ("Full lens, no windowing" if r_rel is None
                else f"Gaussian window σ = {r_rel:.2f} × d_AB")
        a = ALPHA_BY_WINDOW.get(label, ALPHA_TARGET)
        lines.append(f"| {label} | {r_rel if r_rel else '∞'} | {a:.0f} | {desc} |")

    lines += [
        "",
        "## Translation Schedule (Part B)",
        "",
        "- **Static**: window centre fixed at trap-pair midpoint",
        "- **Moving**: S-curve translation from 10% to 90% along A→B during active phase",
        "",
        "## SW Reduction Values (Part C)",
        "",
        f"β_sw_min ∈ {BETA_SW_MIN_VALUES}",
        "",
        "## Control Schedule (Part D)",
        "",
        "`p(t) = β_sw(t)·p_sw + β_lens(t)·exp(iψ)·W(x,y;c(t))·p_lens`",
        "",
        f"- Ramp up:   {T_RAMP_UP*1e3:.0f} ms",
        f"- Hold:      {T_HOLD*1e3:.0f} ms",
        f"- Ramp down: {T_RAMP_DOWN*1e3:.0f} ms",
        f"- Settle:    {T_SETTLE*1e3:.0f} ms",
        f"- α varies by window (see table), ψ = {PSI/np.pi:.1f}π",
        "",
        f"## Sweep Summary (Part E/F)",
        "",
        f"- **Total candidates:** {len(all_results)}",
    ]

    merged = [r for r in all_results if r["metrics"]["A_in_capture_region"]]
    full_success = [r for r in all_results
                    if r["metrics"]["classification"] == "successful_merge"]

    lines += [
        f"- **Achieved merge (A in capture):** {len(merged)}",
        f"- **Full success (merge + B stable + neighbours OK):** {len(full_success)}",
        "",
        "### Phase 1 Baseline (reference: full crossfade, β_sw_min=0)",
        "",
        f"- d_AB final: {baseline_metrics['d_AB_final_um']:.1f} µm",
        f"- A moved: {baseline_metrics['d_A_moved_um']:.1f} µm",
        f"- B moved: {baseline_metrics['d_B_moved_um']:.1f} µm",
        f"- Max neighbour: {baseline_metrics['max_neighbour_disp_um']:.1f} µm",
        f"- Mean neighbour: {baseline_metrics['mean_neighbour_disp_um']:.1f} µm",
        f"- Classification: {baseline_metrics['classification']}",
    ]

    # Top candidates ranked by: merged first, then lowest max_neighbour
    ranked = sorted(all_results,
                    key=lambda r: (
                        not r["metrics"]["A_in_capture_region"],
                        r["metrics"]["max_neighbour_disp_um"]))

    lines += [
        "",
        "### All Candidates Ranked (merged first, then lowest max neighbour)",
        "",
        ("| Rank | Window | α | β_sw_min | Ramp | Trans | d_AB (µm) | A moved | "
         "max_N (µm) | mean_N (µm) | Class |"),
        ("|------|--------|---|----------|------|-------|-----------|---------|"
         "------------|------------|-------|"),
    ]
    for i, r in enumerate(ranked):
        m = r["metrics"]
        lines.append(
            f"| {i+1} | {r['window_label']} | {r.get('alpha', ALPHA_TARGET):.0f} "
            f"| {r['beta_sw_min']:.1f} "
            f"| {r.get('ramp_label', 'fast')} "
            f"| {r['translation']} | {m['d_AB_final_um']:.0f} "
            f"| {m['d_A_moved_um']:.0f} "
            f"| {m['max_neighbour_disp_um']:.0f} "
            f"| {m['mean_neighbour_disp_um']:.0f} "
            f"| {m['classification']} |")

    # Best protocol
    bm = best_result["metrics"]
    bl = best_result["window_label"]
    lines += [
        "",
        "## Best Refined Protocol",
        "",
        f"- **Variant:** {bl}",
        f"- **α:** {best_result.get('alpha', ALPHA_TARGET)}",
        f"- **β_sw_min:** {best_result['beta_sw_min']}",
        f"- **Translation:** {best_result['translation']}",
        f"- **d_AB final:** {bm['d_AB_final_um']:.1f} µm",
        f"- **A moved:** {bm['d_A_moved_um']:.1f} µm",
        f"- **B moved:** {bm['d_B_moved_um']:.1f} µm",
        f"- **Max neighbour:** {bm['max_neighbour_disp_um']:.1f} µm",
        f"- **Mean neighbour:** {bm['mean_neighbour_disp_um']:.1f} µm",
        f"- **Classification:** {bm['classification']}",
    ]
    if bm["merge_time_ms"]:
        lines.append(f"- **Merge time:** {bm['merge_time_ms']:.1f} ms")

    # Improvement
    base_max_n = baseline_metrics["max_neighbour_disp_um"]
    best_max_n = bm["max_neighbour_disp_um"]
    base_mean_n = baseline_metrics["mean_neighbour_disp_um"]
    best_mean_n = bm["mean_neighbour_disp_um"]
    lines += [
        "",
        "### Improvement Over Baseline",
        "",
        f"- Max neighbour: {base_max_n:.0f} → {best_max_n:.0f} µm "
        f"({'%.0f' % (base_max_n - best_max_n)} µm reduction, "
        f"{(base_max_n - best_max_n)/base_max_n*100:.1f}%)"
        if base_max_n > 0 else "",
        f"- Mean neighbour: {base_mean_n:.0f} → {best_mean_n:.0f} µm "
        f"({'%.0f' % (base_mean_n - best_mean_n)} µm reduction)"
        if base_mean_n > 0 else "",
        f"- A reaches B: {bm['A_in_capture_region']}",
        f"- B stable: {bm['B_stayed_in_trap']}",
    ]

    # Key findings
    # Analyse: does smaller window help?
    merged_results = [r for r in all_results
                      if r["metrics"]["A_in_capture_region"]]
    window_merge_counts = {}
    for r in all_results:
        wl = r["window_label"]
        if wl not in window_merge_counts:
            window_merge_counts[wl] = {"total": 0, "merged": 0}
        window_merge_counts[wl]["total"] += 1
        if r["metrics"]["A_in_capture_region"]:
            window_merge_counts[wl]["merged"] += 1

    # Min SW needed for merge
    merged_bsw = [r["beta_sw_min"] for r in merged_results] if merged_results else []
    max_bsw_merged = max(merged_bsw) if merged_bsw else None

    lines += [
        "",
        "## Key Findings",
        "",
        "### 1. Does a smaller/windowed lens reduce neighbour disruption?",
        "",
    ]
    for wl, counts in window_merge_counts.items():
        sub = [r for r in merged_results if r["window_label"] == wl]
        if sub:
            avg_max_n = np.mean([r["metrics"]["max_neighbour_disp_um"]
                                 for r in sub])
            lines.append(f"- {wl}: {counts['merged']}/{counts['total']} "
                         f"merged, avg max_N = {avg_max_n:.0f} µm")
        else:
            lines.append(f"- {wl}: {counts['merged']}/{counts['total']} merged")

    lines += [
        "",
        "### 2. Does translation help?",
        "",
    ]
    for tm in TRANSLATION_MODES:
        sub = [r for r in merged_results if r["translation"] == tm]
        lines.append(f"- {tm}: {len(sub)} merges out of "
                     f"{sum(1 for r in all_results if r['translation'] == tm)}")

    lines += [
        "",
        "### 3. Minimum SW reduction for merge",
        "",
        f"- Highest β_sw_min achieving merge: "
        f"{max_bsw_merged if max_bsw_merged is not None else 'N/A'}",
        "",
        "### 4. Is this better than previous baseline?",
        "",
    ]
    if bm["A_in_capture_region"]:
        is_better = best_max_n < base_max_n
        lines.append(f"- {'YES' if is_better else 'NO'}: "
                     f"max neighbour {best_max_n:.0f} vs {base_max_n:.0f} µm "
                     f"({'improvement' if is_better else 'no improvement'})")
    else:
        lines.append("- Best candidate did not achieve merge")

    lines += [
        "",
        "## Files",
        "",
        "- `gif/` — animations of best candidate(s)",
        "- `figures/footprint_comparison.png` — window variant footprints",
        "- `figures/footprint_1d_profiles.png` — 1D profiles along A→B",
        "- `figures/heatmap_*.png` — parameter sweep results",
        "- `figures/sweep_comparison.png` — displacement comparison",
        "- `figures/comparison_bars.png` — baseline vs best",
        "- `figures/trajectory_best.png` — best candidate trajectories",
        "- `figures/distance_vs_time_best.png` — A-B separation over time",
        "- `figures/schedule_*.png` — control schedule plots",
        "- `npz/` — simulation data",
        "- `config.json` — full run parameters",
        "- `metrics.csv` — all candidate results",
    ]

    with open(OUT_DIR / "INDEX.md", "w") as f:
        f.write("\n".join(lines))
    print("[save] INDEX.md")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()

    # 1. Load data
    (p_sw, p_lens_roi, xg, yg, traps_m,
     idx_A, idx_B, midpoint, d_AB, e_AB) = load_fields()
    trap_A = traps_m[idx_A]
    trap_B = traps_m[idx_B]

    print(f"\n{'='*60}")
    print(f"  C-SHAPE TRANSPORT REFINEMENT STUDY")
    n_cand = (len(WINDOW_RADII_REL) * len(BETA_SW_MIN_VALUES)
              * len(TRANSLATION_MODES) * len(RAMP_VARIANTS))
    print(f"  Sweep: {len(WINDOW_RADII_REL)} windows × "
          f"{len(BETA_SW_MIN_VALUES)} β_sw_min × "
          f"{len(RAMP_VARIANTS)} ramp × "
          f"{len(TRANSLATION_MODES)} trans = {n_cand} candidates")
    print(f"{'='*60}\n")

    # 2. Part A — Footprint diagnostics
    print("\n=== Part A: Footprint Comparison ===\n")
    plot_footprint_comparison(p_lens_roi, xg, yg, traps_m, idx_A, idx_B,
                              d_AB, midpoint)
    plot_footprint_1d_profiles(p_lens_roi, xg, yg, traps_m, idx_A, idx_B,
                               d_AB, midpoint, e_AB)

    # 3. Part D — Schedule plots (representative examples)
    print("\n=== Part D: Schedule Plots ===\n")
    for rv in RAMP_VARIANTS:
        for bsw_min in [0.0, 0.4]:
            for tm in TRANSLATION_MODES:
                sched, ttot = make_schedule(
                    bsw_min, tm, trap_A, trap_B, d_AB,
                    ramp_up=rv["ramp_up"], hold=rv["hold"],
                    ramp_down=rv["ramp_down"], settle=rv["settle"])
                plot_schedule(sched,
                              f"β_sw={bsw_min:.1f}, {rv['label']}, {tm}",
                              f"schedule_bsw{bsw_min:.1f}_{rv['label']}_{tm}.png")

    # 4. Part E — Sweep
    print("\n=== Part E: Parameter Sweep ===\n")
    all_results = []
    baseline_metrics = None
    run_idx = 0

    for r_rel, w_label in zip(WINDOW_RADII_REL, WINDOW_LABELS):
        for bsw_min in BETA_SW_MIN_VALUES:
            for rv in RAMP_VARIANTS:
                for tm in TRANSLATION_MODES:
                    run_idx += 1
                    tag = (f"{w_label}_bsw{bsw_min:.1f}"
                           f"_{rv['label']}_{tm}")
                    print(f"\n--- [{run_idx}/{n_cand}] {tag} ---")

                    sigma = r_rel * d_AB if r_rel is not None else None
                    alpha_w = ALPHA_BY_WINDOW.get(w_label, ALPHA_TARGET)
                    sched, ttot = make_schedule(
                        bsw_min, tm, trap_A, trap_B, d_AB,
                        alpha=alpha_w,
                        ramp_up=rv["ramp_up"], hold=rv["hold"],
                        ramp_down=rv["ramp_down"], settle=rv["settle"])

                    sim = simulate_particles(
                        p_sw, p_lens_roi, xg, yg, traps_m,
                        idx_A, idx_B, d_AB,
                        window_sigma=sigma,
                        schedule_fn=sched, t_total=ttot,
                        store_fields=False)

                    metrics = analyse_success(sim, traps_m, idx_A, idx_B)

                    result = {
                        "window_label": w_label,
                        "window_sigma_rel": r_rel,
                        "beta_sw_min": bsw_min,
                        "ramp_label": rv["label"],
                        "ramp_up": rv["ramp_up"],
                        "ramp_down": rv["ramp_down"],
                        "translation": tm,
                        "alpha": alpha_w,
                        "metrics": metrics,
                        "tag": tag,
                    }
                    all_results.append(result)

                    # Identify baseline: full lens, bsw_min=0, fast, static
                    if (r_rel is None and bsw_min == 0.0
                            and rv["label"] == "fast"
                            and tm == "static"):
                        baseline_metrics = metrics

                    print(f"  class={metrics['classification']}, "
                          f"d_AB={metrics['d_AB_final_um']:.0f}µm, "
                          f"max_N={metrics['max_neighbour_disp_um']:.0f}µm")

    # 5. Identify best
    merged = [r for r in all_results
              if r["metrics"]["A_in_capture_region"]]
    if merged:
        best = min(merged,
                   key=lambda r: r["metrics"]["max_neighbour_disp_um"])
    else:
        best = max(all_results,
                   key=lambda r: r["metrics"]["d_A_moved_um"])

    print(f"\n{'='*60}")
    print(f"  BEST: {best['tag']}")
    print(f"  {best['metrics']['classification']}")
    print(f"  d_AB = {best['metrics']['d_AB_final_um']:.1f} µm")
    print(f"  max_N = {best['metrics']['max_neighbour_disp_um']:.1f} µm")
    print(f"{'='*60}\n")

    # 6. Re-run best with field storage for GIF
    print("=== Re-running best candidate with field storage ===\n")
    r_best = best["window_sigma_rel"]
    sigma_best = r_best * d_AB if r_best is not None else None
    alpha_best = best.get("alpha", ALPHA_TARGET)
    sched_best, ttot_best = make_schedule(
        best["beta_sw_min"], best["translation"],
        trap_A, trap_B, d_AB, alpha=alpha_best,
        ramp_up=best.get("ramp_up", T_RAMP_UP),
        ramp_down=best.get("ramp_down", T_RAMP_DOWN))
    sim_best = simulate_particles(
        p_sw, p_lens_roi, xg, yg, traps_m,
        idx_A, idx_B, d_AB,
        window_sigma=sigma_best,
        schedule_fn=sched_best, t_total=ttot_best,
        store_fields=True)

    # Also re-run baseline with fields for comparison GIF
    print("=== Re-running baseline with field storage ===\n")
    sched_base, ttot_base = make_schedule(0.0, "static", trap_A, trap_B, d_AB)
    sim_base = simulate_particles(
        p_sw, p_lens_roi, xg, yg, traps_m,
        idx_A, idx_B, d_AB,
        window_sigma=None,
        schedule_fn=sched_base, t_total=ttot_base,
        store_fields=True)

    # 7. Part G — Plots
    print("\n=== Part G: Visualisation ===\n")

    plot_trajectories(sim_best, xg, yg, traps_m, idx_A, idx_B,
                      f"Best: {best['tag']}", "trajectory_best.png")
    plot_distance_vs_time(sim_best, traps_m, idx_A, idx_B,
                          f"Best: {best['tag']}", "distance_vs_time_best.png")

    plot_trajectories(sim_base, xg, yg, traps_m, idx_A, idx_B,
                      "Baseline: full_lens, β_sw_min=0, static",
                      "trajectory_baseline.png")

    if baseline_metrics is not None:
        plot_comparison_bars(baseline_metrics, best["metrics"], best["tag"])

    plot_displacement_comparison(all_results, idx_A, idx_B, len(traps_m))
    plot_heatmap(all_results)

    # GIF for best
    gif_best = render_gif(sim_best, xg, yg, traps_m, idx_A, idx_B,
                          "refined_merge.gif")
    gif_base = render_gif(sim_base, xg, yg, traps_m, idx_A, idx_B,
                          "baseline_merge.gif")

    # 8. Save outputs
    print("\n=== Saving outputs ===\n")
    save_config(all_results)
    save_metrics_csv(all_results)

    np.savez_compressed(
        NPZ_DIR / "best_simulation.npz",
        trajectories=sim_best["trajectories"],
        times=sim_best["times"],
        alphas=sim_best["alphas"],
        betas_sw=sim_best["betas_sw"],
        centres=sim_best["centres"],
        initial_pos=sim_best["initial_pos"],
        U_sw_only=sim_best["U_sw_only"],
        xg=xg, yg=yg,
        traps_m=traps_m, idx_A=idx_A, idx_B=idx_B,
    )
    print("[save] best_simulation.npz")

    write_index(all_results, baseline_metrics, best)

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  Total runtime: {elapsed:.1f}s")
    print(f"  Output: {OUT_DIR}")
    print(f"  Best: {best['tag']} → {best['metrics']['classification']}")
    if gif_best:
        print(f"  GIF: {gif_best}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
