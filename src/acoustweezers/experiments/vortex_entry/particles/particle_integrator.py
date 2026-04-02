"""
Time integrator for the inertial particle ensemble.

Governing equations (Option C — direct Gor'kov body force):

    dx/dt = v
    dv/dt = ACCEL_SCALE * F_norm(x)  -  v / TAU_STOKES

where F_norm(x) is the normalised Gor'kov force evaluated via the
existing RegularGridInterpolator pair (iFx, iFy) from phase_sweep.

Integrators
-----------
"rk4"   — 4th-order Runge–Kutta (default, recommended).
            Requires 4 × 2 vectorised interpolator evaluations per step.
            Error: O((dt/τ)⁴) per step — negligible for dt/τ ≈ 0.17.

"euler" — Explicit Euler (fallback, simpler).
            Error: O(dt/τ) ≈ 17 % per step — accumulates.
            Stable (|1 − dt/τ| ≈ 0.83 < 1) but not recommended for
            production use.

Both are fully vectorised over N particles (no Python loops).

Boundary handling
-----------------
Positions are clamped to the grid interior [xg[2], xg[−3]] × [yg[2], yg[−3]]
after each step, matching the existing update_particles convention.
Velocities are NOT clamped — drag naturally limits speed to terminal velocity.
Particles reaching the grid boundary continue to be driven by local force
(which is zero outside the grid via fill_value=0.0).
"""
from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from scripts.lib.particle_dynamics_utils import DT_DEFAULT

from ..utils.interpolation import eval_at
from .drag_models import ACCEL_SCALE, TAU_STOKES
from .particle_state import ParticleEnsemble


def _eval_force(
    pos: np.ndarray,            # (N, 2)  [m]
    iFx: RegularGridInterpolator,
    iFy: RegularGridInterpolator,
) -> np.ndarray:                # (N, 2)  normalised force
    """Evaluate normalised Gor'kov force at N positions, vectorised."""
    return np.column_stack([
        eval_at(iFx, pos),   # (N,) x-component
        eval_at(iFy, pos),   # (N,) y-component
    ])


def _deriv(
    pos: np.ndarray,            # (N, 2)
    vel: np.ndarray,            # (N, 2)
    iFx: RegularGridInterpolator,
    iFy: RegularGridInterpolator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute time derivatives.

    Returns
    -------
    dpos_dt : (N, 2)  =  vel
    dvel_dt : (N, 2)  =  ACCEL_SCALE * F_norm(pos) - vel / TAU_STOKES
    """
    F = _eval_force(pos, iFx, iFy)
    dv = ACCEL_SCALE * F - vel / TAU_STOKES
    return vel.copy(), dv


def _euler_step(
    pos: np.ndarray,
    vel: np.ndarray,
    dt: float,
    iFx: RegularGridInterpolator,
    iFy: RegularGridInterpolator,
) -> tuple[np.ndarray, np.ndarray]:
    dpos, dvel = _deriv(pos, vel, iFx, iFy)
    return pos + dt * dpos, vel + dt * dvel


def _rk4_step(
    pos: np.ndarray,
    vel: np.ndarray,
    dt: float,
    iFx: RegularGridInterpolator,
    iFy: RegularGridInterpolator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Classical 4th-order Runge–Kutta step.

    Stage 1 (at t):
        k1_x = vel
        k1_v = ACCEL_SCALE * F(pos) - vel / τ

    Stage 2 (at t + dt/2, using k1 extrapolation):
        k2_x = vel + k1_v * dt/2
        k2_v = ACCEL_SCALE * F(pos + k1_x * dt/2) - (vel + k1_v*dt/2) / τ

    Stage 3 (at t + dt/2, using k2 extrapolation):
        k3_x = vel + k2_v * dt/2
        k3_v = ACCEL_SCALE * F(pos + k2_x * dt/2) - (vel + k2_v*dt/2) / τ

    Stage 4 (at t + dt, using k3 extrapolation):
        k4_x = vel + k3_v * dt
        k4_v = ACCEL_SCALE * F(pos + k3_x * dt) - (vel + k3_v*dt) / τ

    Update:
        pos_new = pos + (dt/6) * (k1_x + 2*k2_x + 2*k3_x + k4_x)
        vel_new = vel + (dt/6) * (k1_v + 2*k2_v + 2*k3_v + k4_v)

    Requires 4 vectorised force evaluations (8 interpolator calls total).
    """
    k1_x, k1_v = _deriv(pos,                            vel,              iFx, iFy)
    k2_x, k2_v = _deriv(pos + 0.5 * dt * k1_x,         vel + 0.5 * dt * k1_v, iFx, iFy)
    k3_x, k3_v = _deriv(pos + 0.5 * dt * k2_x,         vel + 0.5 * dt * k2_v, iFx, iFy)
    k4_x, k4_v = _deriv(pos +       dt * k3_x,          vel +       dt * k3_v, iFx, iFy)

    pos_new = pos + (dt / 6.0) * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x)
    vel_new = vel + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
    return pos_new, vel_new


def advance_ensemble(
    state: ParticleEnsemble,
    iFx: RegularGridInterpolator,
    iFy: RegularGridInterpolator,
    xg: np.ndarray,
    yg: np.ndarray,
    dt: float = DT_DEFAULT,
    integrator: Literal["rk4", "euler"] = "rk4",
) -> ParticleEnsemble:
    """
    Advance all particles by one timestep dt.

    Parameters
    ----------
    state : ParticleEnsemble
        Current particle positions and velocities.
    iFx, iFy : RegularGridInterpolator
        Normalised Gor'kov force components from phase_sweep.
        Same objects used by update_particles for A and B.
    xg, yg : 1-D arrays
        Coordinate grids (used for boundary clamping only).
    dt : float
        Timestep [s].  Default: DT_DEFAULT = 1e-4 s.
    integrator : "rk4" | "euler"
        Integration scheme.  "rk4" is recommended.

    Returns
    -------
    ParticleEnsemble
        Updated state (new arrays, original not mutated).
    """
    pos = state.pos.copy()
    vel = state.vel.copy()

    if integrator == "rk4":
        pos_new, vel_new = _rk4_step(pos, vel, dt, iFx, iFy)
    else:
        pos_new, vel_new = _euler_step(pos, vel, dt, iFx, iFy)

    # Clamp positions to grid interior — matches update_particles convention
    pos_new[:, 0] = np.clip(pos_new[:, 0], xg[2],  xg[-3])
    pos_new[:, 1] = np.clip(pos_new[:, 1], yg[2],  yg[-3])

    return ParticleEnsemble(pos=pos_new, vel=vel_new)
