"""
Vortex entry control logic: trap selection, geometry validation,
path generation, phase sweep, and ring barrier test.

Extracted verbatim from vortex_entry_test.py.  No physics changes.
"""
from __future__ import annotations

import sys
from typing import Dict, Optional, Tuple

import numpy as np

from scripts.lib.particle_dynamics_utils import LAM, gorkov_normalised

from ..fields.field_superposition import total_pressure
from ..utils.interpolation import eval_at, make_interp


# ── Experiment defaults (same as original script) ─────────────────
N_BARRIER_SAMPLES    = 50
N_RING_THETA         = 72
OPENING_SECTOR_DEG   = 60.0
GEOM_MARGIN          = 1.5
OPENING_THRESHOLD    = 0.05


def select_trap_pair(
    traps_m: np.ndarray,
    r_barrier: float,
    xg: np.ndarray,
    yg: np.ndarray,
    idx_B_override: Optional[int] = None,
    min_sep_factor: float = GEOM_MARGIN,
) -> Tuple[int, int]:
    """
    Choose B (central trap) and A (farthest trap satisfying geometry).

    The vortex starts centred on A, so the constraint is:
        d(B, A) > min_sep_factor * r_barrier

    Among all valid A candidates, the farthest from B is picked to
    maximise initial clearance — B is then well outside the barrier,
    not marginally so.

    Raises SystemExit if no valid A trap exists.
    """
    grid_cx = 0.5 * (float(xg[0]) + float(xg[-1]))
    grid_cy = 0.5 * (float(yg[0]) + float(yg[-1]))
    center = np.array([grid_cx, grid_cy])

    # ── Pick B: nearest trap to grid centre ───────────────────────
    if idx_B_override is not None:
        idx_B = idx_B_override
    else:
        dists_to_center = np.linalg.norm(traps_m - center, axis=1)
        idx_B = int(np.argmin(dists_to_center))

    B_xy = traps_m[idx_B]
    threshold = min_sep_factor * r_barrier

    # ── Search for valid A ────────────────────────────────────────
    dists_to_B = np.linalg.norm(traps_m - B_xy, axis=1)
    valid_mask = dists_to_B > threshold
    valid_mask[idx_B] = False

    print(f"\n  --- Trap Pair Selection ---")
    print(f"  B: trap [{idx_B}] at "
          f"({B_xy[0]*1e3:.3f}, {B_xy[1]*1e3:.3f}) mm")
    print(f"  Geometry constraint: d(A, B) > {min_sep_factor:.1f} * "
          f"{r_barrier*1e6:.1f} um = {threshold*1e6:.1f} um")
    print(f"  Candidates: {int(valid_mask.sum())} of "
          f"{len(traps_m)} traps pass constraint")

    if not np.any(valid_mask):
        print(f"\n  FATAL: No trap satisfies d(A, B) > {threshold*1e6:.1f} um.")
        print(f"  Closest trap to B is [{int(np.argsort(dists_to_B)[1])}] "
              f"at {sorted(dists_to_B)[1]*1e6:.1f} um.")
        print(f"  Options:")
        print(f"    - Decrease --focus_mm (tighter ring → smaller r_barrier)")
        print(f"    - Increase --bessel_aperture_mm (larger NA → tighter ring)")
        print(f"    - Decrease --min_sep_factor (relax geometry constraint)")
        sys.exit(1)

    # Among valid, pick farthest from B (maximises initial clearance)
    dists_valid = dists_to_B.copy()
    dists_valid[~valid_mask] = -np.inf
    idx_A = int(np.argmax(dists_valid))
    A_xy = traps_m[idx_A]
    d_AB = float(dists_to_B[idx_A])

    print(f"  A: trap [{idx_A}] at "
          f"({A_xy[0]*1e3:.3f}, {A_xy[1]*1e3:.3f}) mm  (farthest valid)")
    print(f"  d(A, B) = {d_AB*1e6:.1f} um = {d_AB/LAM:.2f} lam")
    print(f"  d(A, B) / r_barrier = {d_AB/r_barrier:.2f}  (>{min_sep_factor:.1f}  OK)")

    return idx_A, idx_B


def validate_geometry(
    traps_m: np.ndarray,
    idx_A: int,
    idx_B: int,
    r_barrier: float,
) -> str:
    """Check B starts outside the barrier ring. Returns status string."""
    d_B_vc = float(np.linalg.norm(traps_m[idx_B] - traps_m[idx_A]))
    ratio = d_B_vc / r_barrier if r_barrier > 0 else float("inf")

    if ratio > 1.1:
        status = "OUTSIDE"
    elif ratio > 0.9:
        status = "ON"
    else:
        status = "INSIDE"

    print(f"\n  --- Geometry Validation ---")
    print(f"  d(B, vortex_start) = {d_B_vc*1e6:.1f} um")
    print(f"  r_barrier          = {r_barrier*1e6:.1f} um")
    print(f"  ratio              = {ratio:.2f}")
    print(f"  B status:          {status}")

    if status != "OUTSIDE":
        print(f"  WARNING: B is NOT safely outside the barrier ring!")
    else:
        print(f"  OK: B starts {(d_B_vc - r_barrier)*1e6:.1f} um "
              f"outside the barrier.")
    return status


def build_vortex_path(
    start: np.ndarray,
    end: np.ndarray,
    n_steps: int,
    path_mode: str = "straight",
    arc_height_mm: float = 0.0,
) -> np.ndarray:
    """Build the vortex centre path (straight or Bezier arc)."""
    t = np.linspace(0, 1, n_steps)
    if path_mode == "straight" or arc_height_mm == 0.0:
        return start[None, :] + t[:, None] * (end - start)[None, :]

    d = end - start
    perp = np.array([-d[1], d[0]])
    perp = perp / np.linalg.norm(perp)
    sign = 1.0 if path_mode == "arc_left" else -1.0
    mid_ctrl = 0.5 * (start + end) + sign * arc_height_mm * 1e-3 * perp

    path = np.zeros((n_steps, 2))
    for i, ti in enumerate(t):
        path[i] = ((1 - ti) ** 2 * start
                    + 2 * (1 - ti) * ti * mid_ctrl
                    + ti ** 2 * end)
    return path


def ring_barrier_test(
    iU,
    vortex_center: np.ndarray,
    pos_B: np.ndarray,
    r_barrier: float,
    n_theta: int = N_RING_THETA,
    sector_half_deg: float = OPENING_SECTOR_DEG,
    opening_threshold: float = OPENING_THRESHOLD,
) -> Dict:
    """
    Sample Gorkov potential around the barrier ring and test for
    a directional opening vs global basin merging.
    """
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    ring_pts = np.zeros((n_theta, 2))
    ring_pts[:, 0] = vortex_center[0] + r_barrier * np.cos(theta)
    ring_pts[:, 1] = vortex_center[1] + r_barrier * np.sin(theta)
    U_ring = eval_at(iU, ring_pts)

    dxy = pos_B - vortex_center
    theta_B = float(np.arctan2(dxy[1], dxy[0]))

    sector_half_rad = np.deg2rad(sector_half_deg)
    ang_diff = np.abs(np.angle(np.exp(1j * (theta - theta_B))))
    in_sector = ang_diff <= sector_half_rad

    U_B_sector = (float(np.mean(U_ring[in_sector]))
                  if np.any(in_sector) else float("nan"))
    U_other = (float(np.mean(U_ring[~in_sector]))
               if np.any(~in_sector) else float("nan"))

    denom = abs(U_other) if U_other != 0.0 else 1.0
    delta_B = float((U_other - U_B_sector) / denom)

    if delta_B > opening_threshold:
        mechanism = "true_opening"
    elif delta_B > 0.0:
        mechanism = "weak_asymmetry"
    else:
        mechanism = "basin_merging"

    return {
        "theta": theta, "U_ring": U_ring, "theta_B": theta_B,
        "U_B_sector": U_B_sector, "U_other": U_other,
        "delta_B": delta_B, "mechanism": mechanism,
        "U_ring_min": float(np.min(U_ring)),
        "U_ring_max": float(np.max(U_ring)),
    }


def phase_sweep(
    p_sw, vortex_gen, xg, yg, dx, dy,
    psi_values, alpha, beta,
    vortex_center, pos_A, pos_B, r_barrier,
    w_barrier=1.0, w_pull=1.0, w_lateral=0.5, w_retain=1.0,
    opening_threshold=OPENING_THRESHOLD,
    p_noise=None,
) -> Dict:
    e_B = vortex_center - pos_B
    norm_B = np.linalg.norm(e_B)
    e_B = e_B / norm_B if norm_B > 0 else np.array([0.0, 1.0])

    e_A = vortex_center - pos_A
    norm_A = np.linalg.norm(e_A)
    e_A = e_A / norm_A if norm_A > 0 else np.array([0.0, 1.0])

    t_samples = np.linspace(0.05, 0.95, N_BARRIER_SAMPLES)
    line_pts = (pos_B[None, :]
                + t_samples[:, None] * (vortex_center - pos_B)[None, :])

    results = []
    for psi in psi_values:
        p_tot = total_pressure(p_sw, vortex_gen, psi, alpha, beta,
                               vortex_center)
        if p_noise is not None:
            p_tot = p_tot + p_noise
        U_n, Fx_n, Fy_n = gorkov_normalised(p_tot, dx, dy)

        iU  = make_interp(U_n,  xg, yg)
        iFx = make_interp(Fx_n, xg, yg)
        iFy = make_interp(Fy_n, xg, yg)

        FA = np.array([float(eval_at(iFx, pos_A[None, :])[0]),
                       float(eval_at(iFy, pos_A[None, :])[0])])
        F_A_in = float(np.dot(FA, e_A))

        FB = np.array([float(eval_at(iFx, pos_B[None, :])[0]),
                       float(eval_at(iFy, pos_B[None, :])[0])])
        F_in_B   = float(np.dot(FB, e_B))
        F_tang_B = FB - F_in_B * e_B
        F_perp_B = float(np.linalg.norm(F_tang_B))

        U_path = eval_at(iU, line_pts)
        U_B    = float(eval_at(iU, pos_B[None, :])[0])
        DeltaU = float(np.max(U_path) - U_B)

        score = (w_barrier * DeltaU
                 - w_pull * F_in_B
                 + w_lateral * F_perp_B
                 - w_retain * F_A_in)

        results.append({
            "psi": psi, "F_A_in": F_A_in, "F_in_B": F_in_B,
            "F_perp_B": F_perp_B, "DeltaU": DeltaU, "score": score,
            "iFx": iFx, "iFy": iFy, "iU": iU,
        })

    best = min(results, key=lambda r: r["score"])

    best["mechanism"] = ring_barrier_test(
        best["iU"], vortex_center, pos_B, r_barrier,
        opening_threshold=opening_threshold)
    return best
