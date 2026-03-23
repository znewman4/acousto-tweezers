"""
Standardised overdamped particle transport engine.

This is the single, canonical particle dynamics implementation for all
transport studies.  It supersedes the quasi-static descend_potential
approach used in vortex_stage_transport_utils.py and replicates the
overdamped integrator from c_shape_transport_refinement_study.py.

Physics
-------
    dx/dt = mu_stokes * F_gorkov
    F_gorkov = -grad(U_gorkov)
    U_gorkov = (normalised U from Gor'kov) * GORKOV_PREFACTOR * P_SCALE^2

The total field at time t:
    p_total(t) = beta_sw(t) * p_sw + alpha(t) * exp(i*psi) * p_perturb(centre(t))

Bilinear decomposition (quadratic in p):
    U ∝ |b1*p1 + b2*p2|^2 = b1^2*U_11 + b2^2*U_22 + 2*b1*b2*U_12

This allows precomputing 3 Gor'kov basis fields once and evaluating
combined forces cheaply at every time step.

Schedule format
---------------
A list of phase dicts, each with:
    duration_ms : float
    alpha_start : float (default 0)
    alpha_end   : float (default = alpha_start)
    bsw_start   : float (default 1.0)   beta_sw at phase start
    bsw_end     : float (default = bsw_start)
    ctr_start   : ndarray(2) or None
    ctr_end     : ndarray(2) or None (default = ctr_start)
    ramp        : "cosine" | "linear"  (default "cosine")
    label       : str

All centres in metres.

Convention matches c_shape_transport_refinement_study.py exactly so that
running the C-shape through this engine reproduces the existing results.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import shift as nd_shift

# ── Physical constants ─────────────────────────────────────────────
C_WATER: float = 1484.0          # m/s
F_HZ:    float = 2.0e6           # Hz
OMEGA:   float = 2.0 * np.pi * F_HZ  # rad/s
RHO0:    float = 997.0            # kg/m³
LAM:     float = C_WATER / F_HZ   # 0.742 mm
TRAP_SP: float = LAM / 2.0        # ~0.371 mm

# ── Particle: polystyrene 50 µm in water ──────────────────────────
A_PART:  float = 50.0e-6          # radius [m]
RHO_P:   float = 1050.0           # kg/m³
C_P:     float = 2350.0           # m/s
ETA:     float = 1.0e-3           # Pa·s (dynamic viscosity of water)

_KAPPA_W: float = 1.0 / (RHO0 * C_WATER**2)
_KAPPA_P: float = 1.0 / (RHO_P * C_P**2)
F1: float = 1.0 - _KAPPA_P / _KAPPA_W          # monopole contrast ≈ 0.6214
F2: float = 2.0 * (RHO_P - RHO0) / (2.0 * RHO_P + RHO0)  # dipole contrast ≈ 0.0342

MU_STOKES:       float = 1.0 / (6.0 * np.pi * ETA * A_PART)   # m / (N·s)
GORKOV_PREFACTOR: float = (2.0 * np.pi / 3.0) * A_PART**3      # m³  (C-shape convention)

# Pressure scale – the stored FEM/ASM fields are at unit normalisation;
# P_SCALE is the physical peak pressure in Pa that they represent.
P_SCALE: float = 3000.0  # Pa

# Combined velocity scale factor: dx/dt = SCALE * F_gorkov_normalised
# where F_gorkov_normalised = -grad(U_gorkov(p_normalised))
SCALE: float = MU_STOKES * GORKOV_PREFACTOR * P_SCALE**2  # m/s per (normalised Gorkov force)

# ── Capture / stability radii ──────────────────────────────────────
CAPTURE_RADIUS:  float = 0.30 * TRAP_SP   # ≈ 111 µm — A in B trap
NEIGHBOUR_TOL:   float = 0.50 * TRAP_SP   # ≈ 185 µm — neighbour stability

DT_DEFAULT:    float = 1.0e-4   # 0.1 ms time step
N_FRAMES_DEFAULT: int = 280
N_KEYFRAMES:   int = 12         # keyframes for moving-centre basis interpolation


# ══════════════════════════════════════════════════════════════════
# Gor'kov helpers  (C-shape convention: no (4π/3)a³ prefactor here)
# ══════════════════════════════════════════════════════════════════

def gorkov_normalised(
    p: np.ndarray,
    dx: float,
    dy: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Gor'kov potential and force on normalised pressure field.

    Returns (U, Fx, Fy) in the same normalised units as the C-shape study.
    The actual physical force is GORKOV_PREFACTOR * P_SCALE**2 * (Fx, Fy).

    This is identical to compute_gorkov / compute_force_field in
    c_shape_transport_refinement_study.py.
    """
    if dy is None:
        dy = dx
    p = np.asarray(p, dtype=complex)
    coeff_p = F1 / (2.0 * RHO0 * C_WATER**2)
    coeff_k = 3.0 * F2 / (4.0 * OMEGA**2 * RHO0)

    p_abs2 = np.abs(p) ** 2
    dp_dx = np.gradient(p, dx, axis=1)
    dp_dy = np.gradient(p, dy, axis=0)
    grad_abs2 = np.abs(dp_dx) ** 2 + np.abs(dp_dy) ** 2

    U = coeff_p * p_abs2 - coeff_k * grad_abs2
    Fx = -np.gradient(U, dx, axis=1)
    Fy = -np.gradient(U, dy, axis=0)
    return U, Fx, Fy


# BasisTuple: (Fx_11, Fy_11, Fx_22, Fy_22, Fx_12, Fy_12, U_11, U_22, U_12)
BasisTuple = Tuple[
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray,
]


def precompute_bilinear_basis(
    p_sw: np.ndarray,
    p_perturb_eff: np.ndarray,
    dx: float,
    dy: Optional[float] = None,
) -> BasisTuple:
    """
    Precompute Gor'kov bilinear basis fields for the decomposition:

        p_total = b1 * p_sw + b2 * p_perturb_eff

    where p_perturb_eff already includes exp(i*psi) and any spatial window.

    Returns:
        (Fx_11, Fy_11, Fx_22, Fy_22, Fx_12, Fy_12, U_11, U_22, U_12)

    Force field reconstruction:
        Fx_total = b1**2 * Fx_11 + b2**2 * Fx_22 + 2*b1*b2 * Fx_12
    """
    if dy is None:
        dy = dx
    coeff_p = F1 / (2.0 * RHO0 * C_WATER**2)
    coeff_k = 3.0 * F2 / (4.0 * OMEGA**2 * RHO0)

    U_11, Fx_11, Fy_11 = gorkov_normalised(p_sw, dx, dy)
    U_22, Fx_22, Fy_22 = gorkov_normalised(p_perturb_eff, dx, dy)

    # Cross term: Re(p1* · p2) and Re(∇p1* · ∇p2)
    dp1_dx = np.gradient(p_sw, dx, axis=1)
    dp1_dy = np.gradient(p_sw, dy, axis=0)
    dp2_dx = np.gradient(p_perturb_eff, dx, axis=1)
    dp2_dy = np.gradient(p_perturb_eff, dy, axis=0)

    cross_p = np.real(np.conj(p_sw) * p_perturb_eff)
    cross_g = (np.real(np.conj(dp1_dx) * dp2_dx)
               + np.real(np.conj(dp1_dy) * dp2_dy))

    U_12 = coeff_p * cross_p - coeff_k * cross_g
    Fx_12 = -np.gradient(U_12, dx, axis=1)
    Fy_12 = -np.gradient(U_12, dy, axis=0)

    return (Fx_11, Fy_11, Fx_22, Fy_22, Fx_12, Fy_12, U_11, U_22, U_12)


def _make_interp(F: np.ndarray, xg: np.ndarray, yg: np.ndarray) -> RegularGridInterpolator:
    return RegularGridInterpolator(
        (yg, xg), F, bounds_error=False, fill_value=0.0
    )


# InterPSet: 6-tuple of pre-built RegularGridInterpolators for one basis
# order: (iFx_11, iFy_11, iFx_22, iFy_22, iFx_12, iFy_12)
InterpSet = Tuple[
    RegularGridInterpolator,
    RegularGridInterpolator,
    RegularGridInterpolator,
    RegularGridInterpolator,
    RegularGridInterpolator,
    RegularGridInterpolator,
]


def _basis_to_interpset(basis: BasisTuple, xg: np.ndarray, yg: np.ndarray) -> InterpSet:
    """Build 6 RegularGridInterpolators from a BasisTuple (once per keyframe)."""
    Fx_11, Fy_11, Fx_22, Fy_22, Fx_12, Fy_12 = basis[:6]
    return (
        _make_interp(Fx_11, xg, yg),
        _make_interp(Fy_11, xg, yg),
        _make_interp(Fx_22, xg, yg),
        _make_interp(Fy_22, xg, yg),
        _make_interp(Fx_12, xg, yg),
        _make_interp(Fy_12, xg, yg),
    )


def _eval_forces(
    iset: InterpSet,
    b1: float,
    b2: float,
    pts: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate combined force at query points using precomputed InterPSet."""
    iFx11, iFy11, iFx22, iFy22, iFx12, iFy12 = iset
    fx = b1**2 * iFx11(pts) + b2**2 * iFx22(pts) + 2.0 * b1 * b2 * iFx12(pts)
    fy = b1**2 * iFy11(pts) + b2**2 * iFy22(pts) + 2.0 * b1 * b2 * iFy12(pts)
    return fx, fy


def bilinear_forces_at_positions(
    basis: BasisTuple,
    b1: float,
    b2: float,
    pos_xy: np.ndarray,
    xg: np.ndarray,
    yg: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate combined x and y forces at particle positions using bilinear basis.

    Parameters
    ----------
    pos_xy : (n_particles, 2)  columns are [x, y] in metres
    Returns
    -------
    fx, fy : (n_particles,) arrays  (normalised force units)
    """
    # RegularGridInterpolator expects query points as (y, x)
    pts = np.column_stack([pos_xy[:, 1], pos_xy[:, 0]])
    iset = _basis_to_interpset(basis, xg, yg)
    return _eval_forces(iset, b1, b2, pts)


# ══════════════════════════════════════════════════════════════════
# Schedule expansion
# ══════════════════════════════════════════════════════════════════

def _cosine_ramp(t: np.ndarray, duration: float, v0: float, v1: float) -> np.ndarray:
    u = np.clip(t / max(duration, 1e-12), 0.0, 1.0)
    w = 0.5 * (1.0 - np.cos(np.pi * u))
    return v0 + (v1 - v0) * w


def expand_schedule(
    phases: List[Dict],
    dt: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Expand a phase list into per-time-step arrays.

    Returns
    -------
    times     : (n_steps,)   seconds
    alphas    : (n_steps,)   perturbation amplitude
    betas_sw  : (n_steps,)   SW scaling factor
    centers   : (n_steps, 2) perturbation centre [x, y] metres
    labels    : (n_steps,)   phase label strings
    """
    all_alphas:   List[float] = []
    all_betas_sw: List[float] = []
    all_centers:  List[np.ndarray] = []
    all_labels:   List[str] = []

    # Pass 1: collect default centres from adjacent phases
    prev_center: Optional[np.ndarray] = None
    resolved: List[Dict] = []
    for ph in phases:
        ph2 = dict(ph)
        c0 = ph2.get("ctr_start")
        c1 = ph2.get("ctr_end")
        if c0 is None:
            c0 = prev_center
        if c1 is None:
            c1 = c0
        ph2["ctr_start"] = np.asarray(c0, dtype=float) if c0 is not None else np.zeros(2)
        ph2["ctr_end"]   = np.asarray(c1, dtype=float) if c1 is not None else ph2["ctr_start"]
        prev_center = ph2["ctr_end"]
        resolved.append(ph2)

    for ph in resolved:
        dur_s  = float(ph["duration_ms"]) * 1e-3
        n_ph   = max(1, round(dur_s / dt))
        a0     = float(ph.get("alpha_start", 0.0))
        a1     = float(ph.get("alpha_end",   a0))
        b0     = float(ph.get("bsw_start",   1.0))
        b1_val = float(ph.get("bsw_end",     b0))
        c0     = np.asarray(ph["ctr_start"], dtype=float)
        c1     = np.asarray(ph["ctr_end"],   dtype=float)
        ramp   = str(ph.get("ramp", "cosine"))
        label  = str(ph.get("label", ""))

        t_local = np.arange(n_ph, dtype=float) * dt

        if ramp == "linear":
            u = np.clip(t_local / max(dur_s, 1e-12), 0.0, 1.0)
        else:  # cosine
            u_lin = np.clip(t_local / max(dur_s, 1e-12), 0.0, 1.0)
            u = 0.5 * (1.0 - np.cos(np.pi * u_lin))

        alphas_ph   = a0 + (a1 - a0) * u
        betas_ph    = b0 + (b1_val - b0) * u
        centers_ph  = c0[None, :] + u[:, None] * (c1 - c0)[None, :]

        # Smooth S-curve for translate centre (looks better in GIF)
        if np.any(c0 != c1) and ramp == "cosine":
            # already S-curved via cosine u — no extra needed
            pass

        all_alphas.extend(alphas_ph.tolist())
        all_betas_sw.extend(betas_ph.tolist())
        all_centers.extend(centers_ph.tolist())
        all_labels.extend([label] * n_ph)

    n = len(all_alphas)
    times = np.arange(n, dtype=float) * dt
    return (
        times,
        np.array(all_alphas, dtype=float),
        np.array(all_betas_sw, dtype=float),
        np.array(all_centers, dtype=float).reshape(n, 2),
        all_labels,
    )


# ══════════════════════════════════════════════════════════════════
# Field shift (for vortex perturbation)
# ══════════════════════════════════════════════════════════════════

def shift_field(
    field: np.ndarray,
    xg: np.ndarray,
    yg: np.ndarray,
    target_center: np.ndarray,
    source_center: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Shift a complex 2-D field so its centre moves to target_center.

    Uses scipy.ndimage.shift on real and imaginary parts separately.
    Fallback mode uses nearest-edge values to avoid hard zero seams.
    """
    if source_center is None:
        source_center = np.array(
            [0.5 * (xg[0] + xg[-1]), 0.5 * (yg[0] + yg[-1])], dtype=float
        )
    dx = float(xg[1] - xg[0])
    dy = float(yg[1] - yg[0])
    dxy = np.asarray(target_center, dtype=float) - np.asarray(source_center, dtype=float)
    shift_x_pix = float(dxy[0] / dx)
    shift_y_pix = float(dxy[1] / dy)

    re = nd_shift(np.real(field), shift=(shift_y_pix, shift_x_pix),
                  order=1, mode="nearest")
    im = nd_shift(np.imag(field), shift=(shift_y_pix, shift_x_pix),
                  order=1, mode="nearest")
    return re + 1j * im


# ══════════════════════════════════════════════════════════════════
# Transport result dataclass
# ══════════════════════════════════════════════════════════════════

@dataclass
class TransportResult:
    """Stores the outcome of one run_transport call."""
    times_s:         np.ndarray   # (n_frames,)
    trajectories:    np.ndarray   # (n_frames, n_particles, 2)
    alphas:          np.ndarray   # (n_frames,)
    betas_sw:        np.ndarray   # (n_frames,)
    centers:         np.ndarray   # (n_frames, 2)
    phase_labels:    List[str]    # len n_frames
    initial_positions: np.ndarray  # (n_particles, 2)
    merge_time_s:    Optional[float] = None  # when A entered capture of B


# ══════════════════════════════════════════════════════════════════
# Main transport runner
# ══════════════════════════════════════════════════════════════════

def run_transport(
    p_sw: np.ndarray,
    perturbation_fn: Callable[[np.ndarray], np.ndarray],
    phases: List[Dict],
    psi: float,
    initial_positions: np.ndarray,
    xg: np.ndarray,
    yg: np.ndarray,
    idx_A: int,
    idx_B: int,
    dt: float = DT_DEFAULT,
    n_frames: int = N_FRAMES_DEFAULT,
    n_keyframes: int = N_KEYFRAMES,
    mobility_scale: float = 1.0,
    p_bias: Optional[np.ndarray] = None,
) -> TransportResult:
    """
    Run overdamped particle transport under a time-varying field.

    Parameters
    ----------
    p_sw
        Complex standing-wave pressure field, shape (ny, nx).
    perturbation_fn
        Callable: centre_xy (2,) → complex perturbation field (ny, nx).
        The field is NOT yet scaled by alpha or psi — those are applied here.
    phases
        List of phase dicts (see module docstring).
    psi
        Phase offset applied to perturbation: p_eff = exp(i*psi) * p_perturb.
    initial_positions
        (n_particles, 2) array of starting [x, y] positions in metres.
    xg, yg
        1-D coordinate arrays (metres).
    idx_A, idx_B
        Indices into initial_positions for the transported (A) and
        target (B) particles — used to detect merge time.
    dt
        Integration time step in seconds.
    n_frames
        Number of frames stored in the result.
    n_keyframes
        Number of basis precomputation keyframes for moving-centre phases.
    mobility_scale
        Multiplier on particle drift speed (1.0 keeps canonical dynamics).
    p_bias
        Optional complex bias field added to standing-wave basis construction
        only: p_sw_eff = p_sw + p_bias.
    """
    p_sw = np.asarray(p_sw, dtype=complex)
    p_sw_eff = p_sw if p_bias is None else p_sw + np.asarray(p_bias, dtype=complex)
    pos = np.asarray(initial_positions, dtype=float).copy()
    n_particles = pos.shape[0]

    dx = float(xg[1] - xg[0])
    dy = float(yg[1] - yg[0])

    # Expand schedule
    times_all, alphas_all, betas_sw_all, centers_all, labels_all = expand_schedule(phases, dt)
    n_steps = len(alphas_all)
    frame_every = max(1, n_steps // n_frames)

    # ── Detect whether perturbation centre is static ──────────────
    active = alphas_all > 1e-12
    if np.any(active):
        centres_active = centers_all[active]
        is_static = bool(np.allclose(centres_active, centres_active[0:1], atol=1e-9))
    else:
        is_static = True

    ei_psi = np.exp(1j * float(psi))

    # Precompute basis+interpolators once per keyframe and reuse each step.
    if is_static:
        c_static = centers_all[active][0] if np.any(active) else np.zeros(2)
        p_eff_static = ei_psi * perturbation_fn(c_static)
        basis_static = precompute_bilinear_basis(p_sw_eff, p_eff_static, dx, dy)
        iset_static = _basis_to_interpset(basis_static, xg, yg)
        kf_times = np.array([0.0])
        iset_list = [iset_static]
    else:
        t_active = times_all[active]
        t_kf = np.linspace(float(t_active[0]), float(t_active[-1]), n_keyframes)
        t_kf = np.unique(np.concatenate([[0.0], t_kf, [times_all[-1]]]))
        kf_times = t_kf
        iset_list = []
        for t_k in t_kf:
            k_idx = int(np.argmin(np.abs(times_all - t_k)))
            c_k = centers_all[k_idx]
            p_eff = ei_psi * perturbation_fn(c_k)
            basis_k = precompute_bilinear_basis(p_sw_eff, p_eff, dx, dy)
            iset_list.append(_basis_to_interpset(basis_k, xg, yg))

    # Single-pass overdamped integration loop.
    traj_times:    List[float]      = [0.0]
    traj_pos:      List[np.ndarray] = [pos.copy()]
    traj_alphas:   List[float]      = [0.0]
    traj_bsw:      List[float]      = [1.0]
    traj_centers:  List[np.ndarray] = [centers_all[0].copy()]
    traj_labels:   List[str]        = [labels_all[0] if labels_all else ""]

    merge_time_s: Optional[float] = None

    x_lo, x_hi = float(xg[2]), float(xg[-3])
    y_lo, y_hi = float(yg[2]), float(yg[-3])
    n_kf = len(kf_times)

    step_scale = float(SCALE * max(mobility_scale, 0.0) * dt)

    for step in range(n_steps):
        t_now = float(times_all[step])
        alpha_k = float(alphas_all[step])
        bsw_k = float(betas_sw_all[step])

        pts = np.column_stack([pos[:, 1], pos[:, 0]])

        if is_static:
            fx, fy = _eval_forces(iset_list[0], bsw_k, alpha_k, pts)
        elif alpha_k < 1e-12:
            # Pure SW: only use "11" term (b2=0).
            iFx11, iFy11 = iset_list[0][0], iset_list[0][1]
            fx = bsw_k**2 * iFx11(pts)
            fy = bsw_k**2 * iFy11(pts)
        else:
            # Moving centre: interpolate force between bracketing keyframes.
            idx_lo = int(np.searchsorted(kf_times, t_now) - 1)
            idx_lo = max(0, min(idx_lo, n_kf - 2))
            idx_hi = idx_lo + 1
            t_lo, t_hi = kf_times[idx_lo], kf_times[idx_hi]
            w = float(np.clip((t_now - t_lo) / max(t_hi - t_lo, 1e-30), 0.0, 1.0))
            if w < 1e-6:
                fx, fy = _eval_forces(iset_list[idx_lo], bsw_k, alpha_k, pts)
            elif w > 1.0 - 1e-6:
                fx, fy = _eval_forces(iset_list[idx_hi], bsw_k, alpha_k, pts)
            else:
                fx_lo, fy_lo = _eval_forces(iset_list[idx_lo], bsw_k, alpha_k, pts)
                fx_hi, fy_hi = _eval_forces(iset_list[idx_hi], bsw_k, alpha_k, pts)
                fx = (1.0 - w) * fx_lo + w * fx_hi
                fy = (1.0 - w) * fy_lo + w * fy_hi

        pos[:, 0] = np.clip(pos[:, 0] + step_scale * fx, x_lo, x_hi)
        pos[:, 1] = np.clip(pos[:, 1] + step_scale * fy, y_lo, y_hi)

        if merge_time_s is None:
            d_AB = float(np.linalg.norm(pos[idx_A] - pos[idx_B]))
            if d_AB < CAPTURE_RADIUS:
                merge_time_s = t_now

        if (step + 1) % frame_every == 0 or step == n_steps - 1:
            traj_times.append(t_now)
            traj_pos.append(pos.copy())
            traj_alphas.append(alpha_k)
            traj_bsw.append(bsw_k)
            traj_centers.append(centers_all[step].copy())
            traj_labels.append(labels_all[step] if labels_all else "")

    return TransportResult(
        times_s=np.array(traj_times, dtype=float),
        trajectories=np.array(traj_pos, dtype=float),
        alphas=np.array(traj_alphas, dtype=float),
        betas_sw=np.array(traj_bsw, dtype=float),
        centers=np.array(traj_centers, dtype=float),
        phase_labels=traj_labels,
        initial_positions=np.asarray(initial_positions, dtype=float),
        merge_time_s=merge_time_s,
    )


# ══════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════

def compute_metrics(
    result: TransportResult,
    idx_A: int,
    idx_B: int,
    neighbour_idx: np.ndarray,
    capture_radius: float = CAPTURE_RADIUS,
    neighbour_tol: float = NEIGHBOUR_TOL,
) -> Dict[str, Any]:
    """
    Compute standardised transport success metrics.

    Returns a dict ready to be serialised as metrics.json.
    """
    pos0   = result.initial_positions                      # (n_p, 2)
    pos_f  = result.trajectories[-1]                       # (n_p, 2)
    A0, B0 = pos0[idx_A], pos0[idx_B]
    Af, Bf = pos_f[idx_A], pos_f[idx_B]

    d_AB_final  = float(np.linalg.norm(Af - Bf))
    A_moved     = float(np.linalg.norm(Af - A0))
    B_moved     = float(np.linalg.norm(Bf - B0))

    A_success   = bool(d_AB_final < capture_radius)
    B_stable    = bool(B_moved < neighbour_tol)

    nidx = np.asarray(neighbour_idx, dtype=int)
    if len(nidx) > 0:
        neigh_disps = np.linalg.norm(pos_f[nidx] - pos0[nidx], axis=1)
        max_neigh   = float(np.max(neigh_disps))
        mean_neigh  = float(np.mean(neigh_disps))
        rms_neigh   = float(np.sqrt(np.mean(neigh_disps**2)))
        neigh_ok    = bool(np.all(neigh_disps < neighbour_tol))
    else:
        max_neigh = mean_neigh = rms_neigh = 0.0
        neigh_ok = True

    success_score = (
        (1.0 if A_success else 0.0)
        - (1.0 if not B_stable else 0.0)
        - (1.0 if not neigh_ok else 0.0)
        - rms_neigh * 1e3           # RMS in mm contributes penalty
    )

    # Classification matching C-shape study convention
    if A_success and B_stable and neigh_ok:
        classification = "successful_merge"
    elif A_success:
        classification = "partial_success"
    else:
        classification = "failed"

    # transport_distance: projection of A movement onto A→B direction
    e_ab = (B0 - A0) / max(float(np.linalg.norm(B0 - A0)), 1e-30)
    transport_dist = float(np.dot(Af - A0, e_ab))

    return {
        "A_success":            A_success,
        "B_stable":             B_stable,
        "neighbour_stable":     neigh_ok,
        "d_AB_final_um":        round(d_AB_final * 1e6, 2),
        "A_moved_um":           round(A_moved * 1e6, 2),
        "B_moved_um":           round(B_moved * 1e6, 2),
        "transport_distance_um": round(transport_dist * 1e6, 2),
        "max_neighbour_disp_um": round(max_neigh * 1e6, 2),
        "mean_neighbour_disp_um": round(mean_neigh * 1e6, 2),
        "disturbance_rms_um":   round(rms_neigh * 1e6, 2),
        "success_score":        round(success_score, 4),
        "merge_time_ms":        round(result.merge_time_s * 1e3, 2) if result.merge_time_s is not None else None,
        "classification":       classification,
    }
