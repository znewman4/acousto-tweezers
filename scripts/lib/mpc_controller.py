"""
Model Predictive Controller for vortex-assisted particle merging.

Implements a receding-horizon MPC that optimises vortex parameters
(phase ψ, position x_v/y_v, amplitudes α/β) to merge particle A
into particle B's trap while minimising neighbour disturbance.

Architecture mirrors adjoint_circle_track_mpc.py but with:
  - 5 control DOFs per step: (ψ, x_v, y_v, α, β)
  - Trilinear Gor'kov basis for analytic ∂F/∂(ψ, α, β)
  - FD-based ∂F/∂(x_v, y_v) via basis rebuild at perturbed positions
  - Multi-particle cost (merge A→B, stabilise neighbours)
  - Discrete adjoint backpropagation through overdamped dynamics

Dynamics: x_{t+1} = x_t + dt · SCALE · F(x_t, u_t)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from scipy.interpolate import RectBivariateSpline

from scripts.lib.mpc_gorkov_basis import (
    TriBasisInterp,
    build_basis_for_vortex_position,
)
from scripts.lib.particle_dynamics_utils import (
    C_WATER,
    CAPTURE_RADIUS,
    DT_DEFAULT,
    F1,
    F2,
    OMEGA,
    RHO0,
    SCALE,
    TransportResult,
)

# Gorkov coefficients (same as gorkov_normalised and precompute_trilinear_basis)
_COEFF_P = F1 / (2.0 * RHO0 * C_WATER ** 2)
_COEFF_K = 3.0 * F2 / (4.0 * OMEGA ** 2 * RHO0)

# Number of control DOFs per step
N_CTRL = 5
# Index names for readability
I_PSI, I_XV, I_YV, I_ALPHA, I_BETA = range(N_CTRL)


# ══════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════

@dataclass
class MPCConfig:
    """Configuration for the vortex-merge MPC controller."""
    # Horizon & timing
    K: int = 10                       # look-ahead horizon steps
    T: int = 3000                     # total execution steps
    dt: float = DT_DEFAULT            # time step [s]  (0.1 ms)
    replan_every: int = 1             # replan frequency
    n_iters: int = 20                 # inner optimiser iterations

    # Objective weights
    w_merge: float = 1.0e12           # pull A toward B₀
    w_B_stable: float = 1.0e12        # keep B at B₀
    w_neigh: float = 5.0e11           # keep neighbours stable
    w_smooth: float = 1.0e-2          # control smoothness (scalar fallback)
    # Per-DOF smoothness weights [ψ, x_v, y_v, α, β].  If set, overrides
    # the scalar w_smooth for fine-grained control (e.g. free ψ, smooth xy).
    w_smooth_vec: Optional[np.ndarray] = None
    w_beta_high: float = 1.0e6        # penalise β < 1
    beta_terminal: float = 2.0        # terminal cost multiplier

    # Control bounds  [ψ, x_v, y_v, α, β]
    u_lo: np.ndarray = field(default_factory=lambda: np.array([-4.0 * np.pi, 0.0, 0.0, 0.0, 0.5]))
    u_hi: np.ndarray = field(default_factory=lambda: np.array([4.0 * np.pi, 6e-3, 6e-3, 15.0, 1.0]))

    # Per-step rate limits |u_{t+1} - u_t| ≤ du_max  (None = unconstrained)
    # Physical motivation: robotic-stage slew rate, amplifier bandwidth.
    du_max: Optional[np.ndarray] = None   # shape (N_CTRL,)

    # FD step for vortex position gradients
    fd_eps_pos: float = 5.0e-6        # 5 µm

    # FD step for ∂F/∂x (state Jacobian)
    fd_eps_x: float = 1.0e-6          # 1 µm

    # Line-search alphas (fallback if not using scipy)
    alphas: Tuple[float, ...] = (0.0, 0.005, 0.01, 0.05, 0.1, 0.3, 1.0)


@dataclass
class MPCState:
    """Cached forward-pass state for adjoint computation."""
    positions: List[np.ndarray]     # (K+1,) each (n_particles, 2)
    controls: List[np.ndarray]      # (K,) each (N_CTRL,)
    forces: List[Tuple[np.ndarray, np.ndarray]]  # (K,) each (Fx, Fy)


# ══════════════════════════════════════════════════════════════════
# Core physics: evaluate forces at a given control + particle state
# ══════════════════════════════════════════════════════════════════

class ForceEvaluator:
    """
    Wraps the trilinear basis to provide F(x, u) and ∂F/∂u.

    For vortex-position changes, rebuilds the basis from the
    VortexPerturbation generator (cached per unique position to
    avoid redundant computation within one MPC horizon).
    """

    def __init__(
        self,
        p_sw: np.ndarray,
        vortex_gen,
        xg: np.ndarray,
        yg: np.ndarray,
    ) -> None:
        self.p_sw = p_sw
        self.vortex_gen = vortex_gen
        self.xg = xg
        self.yg = yg
        self._basis_cache: Dict[Tuple[float, float], TriBasisInterp] = {}

        # Pre-build SW field splines (constant — built once here).
        # Used by _fast_forces_at_pts for cheap point-wise FD over vortex position.
        self._sw_spl_re = RectBivariateSpline(yg, xg, np.real(p_sw), kx=3, ky=3)
        self._sw_spl_im = RectBivariateSpline(yg, xg, np.imag(p_sw), kx=3, ky=3)

        # Borrow splines directly from VortexPerturbation (already built).
        self._vortex_spl_re = vortex_gen._spline_re
        self._vortex_spl_im = vortex_gen._spline_im
        self._vortex_src_centre = vortex_gen._source_centre.copy()  # (2,) [x, y]

    def _get_basis(self, xv: float, yv: float) -> TriBasisInterp:
        """Get or build the trilinear basis for a vortex position."""
        key = (round(xv, 8), round(yv, 8))
        if key not in self._basis_cache:
            self._basis_cache[key] = build_basis_for_vortex_position(
                self.p_sw, self.vortex_gen,
                np.array([xv, yv]),
                self.xg, self.yg,
            )
        return self._basis_cache[key]

    def clear_cache(self) -> None:
        """Clear the basis cache (call between MPC replans)."""
        self._basis_cache.clear()

    def forces(
        self, u: np.ndarray, pos_xy: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate (Fx, Fy) at particle positions for control u."""
        psi, xv, yv, alpha, beta = u
        basis = self._get_basis(xv, yv)
        return basis.eval_forces(beta, alpha, psi, pos_xy)

    def _fast_forces_at_pts(
        self,
        pos_xy: np.ndarray,
        xv: float,
        yv: float,
        alpha: float,
        beta: float,
        psi: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fast force at particle positions using only spline point queries.

        Avoids the full 400×400 basis rebuild — uses a local 5-point stencil
        per particle. ~60× faster than _get_basis() for point-only force needs.

        Used exclusively for FD position (x_v, y_v) gradient computation.
        """
        LOC_EPS = 1.5e-6  # local stencil step [m] (well below grid spacing)
        n = pos_xy.shape[0]
        xp, yp = pos_xy[:, 0], pos_xy[:, 1]

        # 5 stencil points per particle: [center, +x, -x, +y, -y]
        xs = np.concatenate([xp, xp + LOC_EPS, xp - LOC_EPS, xp,            xp           ])
        ys = np.concatenate([yp, yp,            yp,            yp + LOC_EPS, yp - LOC_EPS])

        # Shift for vortex source coordinates
        sc_x, sc_y = self._vortex_src_centre
        xsq = np.clip(xs - (xv - sc_x), self.xg[0], self.xg[-1])
        ysq = np.clip(ys - (yv - sc_y), self.yg[0], self.yg[-1])

        # SW field + gradients at stencil points (use source coordinates = xs, ys)
        psw_r  = self._sw_spl_re.ev(ys, xs)
        psw_i  = self._sw_spl_im.ev(ys, xs)
        dpsw_x_r = self._sw_spl_re.ev(ys, xs, dx=1)
        dpsw_x_i = self._sw_spl_im.ev(ys, xs, dx=1)
        dpsw_y_r = self._sw_spl_re.ev(ys, xs, dy=1)
        dpsw_y_i = self._sw_spl_im.ev(ys, xs, dy=1)

        # Vortex field + gradients at shifted stencil points
        pv_r   = self._vortex_spl_re.ev(ysq, xsq)
        pv_i   = self._vortex_spl_im.ev(ysq, xsq)
        dpv_x_r = self._vortex_spl_re.ev(ysq, xsq, dx=1)
        dpv_x_i = self._vortex_spl_im.ev(ysq, xsq, dx=1)
        dpv_y_r = self._vortex_spl_re.ev(ysq, xsq, dy=1)
        dpv_y_i = self._vortex_spl_im.ev(ysq, xsq, dy=1)

        # Total field: p_tot = β p_sw + α exp(iψ) p_v
        cp, sp = float(np.cos(psi)), float(np.sin(psi))
        pre_r = alpha * cp;  pre_i = alpha * sp  # Re/Im of α exp(iψ)

        pt_r = beta * psw_r + pre_r * pv_r - pre_i * pv_i
        pt_i = beta * psw_i + pre_r * pv_i + pre_i * pv_r

        dx_r = beta * dpsw_x_r + pre_r * dpv_x_r - pre_i * dpv_x_i
        dx_i = beta * dpsw_x_i + pre_r * dpv_x_i + pre_i * dpv_x_r

        dy_r = beta * dpsw_y_r + pre_r * dpv_y_r - pre_i * dpv_y_i
        dy_i = beta * dpsw_y_i + pre_r * dpv_y_i + pre_i * dpv_y_r

        # Gorkov potential at each stencil point
        U_s = (_COEFF_P * (pt_r ** 2 + pt_i ** 2)
               - _COEFF_K * (dx_r ** 2 + dx_i ** 2 + dy_r ** 2 + dy_i ** 2))

        # Reshape to (5, n): rows are [center, +x, -x, +y, -y]
        U5 = U_s.reshape(5, n)
        Fx = -(U5[1] - U5[2]) / (2.0 * LOC_EPS)
        Fy = -(U5[3] - U5[4]) / (2.0 * LOC_EPS)
        return Fx, Fy

    def dF_du(
        self,
        u: np.ndarray,
        pos_xy: np.ndarray,
        fd_eps_pos: float = 5.0e-6,
    ) -> np.ndarray:
        """
        Compute ∂F/∂u — shape (n_particles, 2, N_CTRL).

        Analytic for ψ, α, β.  Finite-difference for x_v, y_v
        (uses fast spline point evaluator — no full-grid basis rebuild).
        """
        psi, xv, yv, alpha, beta = u
        basis = self._get_basis(xv, yv)
        n = pos_xy.shape[0]
        jac = np.zeros((n, 2, N_CTRL))

        # Analytic gradients for ψ, α, β
        grads = basis.eval_dF_dcontrols(beta, alpha, psi, pos_xy)
        jac[:, 0, I_PSI]   = grads["psi"][0]
        jac[:, 1, I_PSI]   = grads["psi"][1]
        jac[:, 0, I_ALPHA] = grads["alpha"][0]
        jac[:, 1, I_ALPHA] = grads["alpha"][1]
        jac[:, 0, I_BETA]  = grads["beta"][0]
        jac[:, 1, I_BETA]  = grads["beta"][1]

        # Fast FD for x_v (no full-grid rebuild)
        Fxp_x, Fyp_x = self._fast_forces_at_pts(pos_xy, xv + fd_eps_pos, yv, alpha, beta, psi)
        Fxm_x, Fym_x = self._fast_forces_at_pts(pos_xy, xv - fd_eps_pos, yv, alpha, beta, psi)
        jac[:, 0, I_XV] = (Fxp_x - Fxm_x) / (2.0 * fd_eps_pos)
        jac[:, 1, I_XV] = (Fyp_x - Fym_x) / (2.0 * fd_eps_pos)

        # Fast FD for y_v (no full-grid rebuild)
        Fxp_y, Fyp_y = self._fast_forces_at_pts(pos_xy, xv, yv + fd_eps_pos, alpha, beta, psi)
        Fxm_y, Fym_y = self._fast_forces_at_pts(pos_xy, xv, yv - fd_eps_pos, alpha, beta, psi)
        jac[:, 0, I_YV] = (Fxp_y - Fxm_y) / (2.0 * fd_eps_pos)
        jac[:, 1, I_YV] = (Fyp_y - Fym_y) / (2.0 * fd_eps_pos)

        return jac

    def dF_dx_fd(
        self,
        u: np.ndarray,
        pos_xy: np.ndarray,
        eps: float = 1.0e-6,
    ) -> np.ndarray:
        """
        Compute ∂F/∂x via centered FD — shape (n_particles, 2, 2).

        jac[i, :, 0] = ∂(Fx_i, Fy_i)/∂x_i
        jac[i, :, 1] = ∂(Fx_i, Fy_i)/∂y_i
        """
        n = pos_xy.shape[0]
        jac = np.zeros((n, 2, 2))

        for dim in range(2):
            pos_p = pos_xy.copy()
            pos_m = pos_xy.copy()
            pos_p[:, dim] += eps
            pos_m[:, dim] -= eps
            Fxp, Fyp = self.forces(u, pos_p)
            Fxm, Fym = self.forces(u, pos_m)
            jac[:, 0, dim] = (Fxp - Fxm) / (2.0 * eps)
            jac[:, 1, dim] = (Fyp - Fym) / (2.0 * eps)

        return jac


# ══════════════════════════════════════════════════════════════════
# Forward rollout
# ══════════════════════════════════════════════════════════════════

def rollout(
    feval: ForceEvaluator,
    controls: List[np.ndarray],
    x0: np.ndarray,
    dt: float,
) -> MPCState:
    """
    Roll out K steps of overdamped dynamics.

    Parameters
    ----------
    feval : ForceEvaluator
    controls : list of K arrays, each (N_CTRL,)
    x0 : (n_particles, 2) — initial positions
    dt : float — time step

    Returns
    -------
    MPCState with K+1 positions, K controls, K force pairs.
    """
    positions = [x0.copy()]
    forces_list = []
    pos = x0.copy()
    step_scale = SCALE * dt

    for u_t in controls:
        Fx, Fy = feval.forces(u_t, pos)
        forces_list.append((Fx, Fy))
        pos = pos.copy()
        pos[:, 0] += step_scale * Fx
        pos[:, 1] += step_scale * Fy
        positions.append(pos.copy())

    return MPCState(positions=positions, controls=list(controls), forces=forces_list)


# ══════════════════════════════════════════════════════════════════
# Cost function
# ══════════════════════════════════════════════════════════════════

def compute_step_cost(
    pos: np.ndarray,
    u: np.ndarray,
    u_prev: Optional[np.ndarray],
    idx_A: int,
    idx_B: int,
    neigh_idx: np.ndarray,
    target_pos: np.ndarray,
    cfg: MPCConfig,
) -> float:
    """Per-step cost L_t."""
    x_A = pos[idx_A]
    x_B = pos[idx_B]
    B0 = target_pos[idx_B]

    # Pull A toward B₀
    J = cfg.w_merge * float(np.sum((x_A - B0) ** 2))

    # Keep B at B₀
    J += cfg.w_B_stable * float(np.sum((x_B - B0) ** 2))

    # Keep neighbours at their start positions
    if len(neigh_idx) > 0:
        diffs = pos[neigh_idx] - target_pos[neigh_idx]
        J += cfg.w_neigh * float(np.sum(diffs ** 2))

    # Control smoothness (per-DOF if available, else scalar)
    if u_prev is not None:
        du = u - u_prev
        if cfg.w_smooth_vec is not None:
            J += float(np.sum(cfg.w_smooth_vec * du ** 2))
        else:
            J += cfg.w_smooth * float(np.sum(du ** 2))

    # Penalise reducing β below 1
    beta = u[I_BETA]
    if beta < 1.0:
        J += cfg.w_beta_high * (1.0 - beta) ** 2

    return J


def compute_step_cost_grad_x(
    pos: np.ndarray,
    idx_A: int,
    idx_B: int,
    neigh_idx: np.ndarray,
    target_pos: np.ndarray,
    cfg: MPCConfig,
) -> np.ndarray:
    """∂L_t/∂x — shape (n_particles, 2)."""
    n = pos.shape[0]
    grad = np.zeros((n, 2))

    B0 = target_pos[idx_B]
    grad[idx_A] += 2.0 * cfg.w_merge * (pos[idx_A] - B0)
    grad[idx_B] += 2.0 * cfg.w_B_stable * (pos[idx_B] - B0)

    if len(neigh_idx) > 0:
        grad[neigh_idx] += 2.0 * cfg.w_neigh * (pos[neigh_idx] - target_pos[neigh_idx])

    return grad


def compute_step_cost_grad_u(
    u: np.ndarray,
    u_prev: Optional[np.ndarray],
    cfg: MPCConfig,
) -> np.ndarray:
    """∂L_t/∂u — shape (N_CTRL,)."""
    grad = np.zeros(N_CTRL)

    if u_prev is not None:
        du = u - u_prev
        if cfg.w_smooth_vec is not None:
            grad += 2.0 * cfg.w_smooth_vec * du
        else:
            grad += 2.0 * cfg.w_smooth * du

    beta = u[I_BETA]
    if beta < 1.0:
        grad[I_BETA] += -2.0 * cfg.w_beta_high * (1.0 - beta)

    return grad


# ══════════════════════════════════════════════════════════════════
# Discrete adjoint gradient computation
# ══════════════════════════════════════════════════════════════════

def compute_mpc_gradients(
    feval: ForceEvaluator,
    state: MPCState,
    idx_A: int,
    idx_B: int,
    neigh_idx: np.ndarray,
    target_pos: np.ndarray,
    cfg: MPCConfig,
    u_prev_external: Optional[np.ndarray] = None,
) -> List[np.ndarray]:
    """
    Compute ∂J/∂u_t for t=0..K-1 via discrete adjoint backpropagation.

    Returns list of K gradient vectors, each shape (N_CTRL,).
    """
    K = len(state.controls)
    step_scale = SCALE * cfg.dt

    # ── Terminal adjoint λ_K = β_T · ∂L_K/∂x_K ──────────────────
    lam = cfg.beta_terminal * compute_step_cost_grad_x(
        state.positions[K], idx_A, idx_B, neigh_idx, target_pos, cfg
    )  # (n_particles, 2)

    gradients: List[np.ndarray] = []

    for t in reversed(range(K)):
        u_t = state.controls[t]
        pos_t = state.positions[t]

        u_prev_t = state.controls[t - 1] if t > 0 else u_prev_external

        # ── Direct gradient ∂L_t/∂u_t ────────────────────────────
        dL_du = compute_step_cost_grad_u(u_t, u_prev_t, cfg)

        # ── Smoothness from step t+1 pointing back to u_t ────────
        if t < K - 1:
            u_tp1 = state.controls[t + 1]
            if cfg.w_smooth_vec is not None:
                dL_du -= 2.0 * cfg.w_smooth_vec * (u_tp1 - u_t)
            else:
                dL_du -= 2.0 * cfg.w_smooth * (u_tp1 - u_t)

        # ── Adjoint contribution: (∂x_{t+1}/∂u_t)^T λ_{t+1} ─────
        #    ∂x_{t+1}/∂u_t = dt * SCALE * ∂F/∂u_t
        dF_du = feval.dF_du(u_t, pos_t, cfg.fd_eps_pos)  # (n, 2, N_CTRL)
        # Contract: sum over particles and xy: lam[i,j] * dF_du[i,j,k]
        adj_contrib = step_scale * np.einsum("ij,ijk->k", lam, dF_du)
        grad_u_t = dL_du + adj_contrib

        gradients.append(grad_u_t)

        # ── Update adjoint: λ_t = ∂L_t/∂x_t + (∂x_{t+1}/∂x_t)^T λ_{t+1}
        dL_dx = compute_step_cost_grad_x(
            pos_t, idx_A, idx_B, neigh_idx, target_pos, cfg
        )
        # ∂x_{t+1}/∂x_t = I + dt*SCALE * ∂F/∂x_t
        dF_dx = feval.dF_dx_fd(u_t, pos_t, cfg.fd_eps_x)  # (n, 2, 2)
        # (I + scale * dF_dx)^T @ lam  for each particle
        n = pos_t.shape[0]
        lam_new = dL_dx.copy()
        for i in range(n):
            Jac_T = np.eye(2) + step_scale * dF_dx[i].T  # (2, 2)
            lam_new[i] += Jac_T @ lam[i]
        lam = lam_new

    gradients.reverse()
    return gradients


# ══════════════════════════════════════════════════════════════════
# Horizon solver (inner MPC optimisation)
# ══════════════════════════════════════════════════════════════════

def compute_total_cost(
    feval: ForceEvaluator,
    controls: List[np.ndarray],
    x0: np.ndarray,
    idx_A: int,
    idx_B: int,
    neigh_idx: np.ndarray,
    target_pos: np.ndarray,
    cfg: MPCConfig,
    u_prev_external: Optional[np.ndarray] = None,
) -> float:
    """Evaluate total MPC objective J over K-step horizon."""
    state = rollout(feval, controls, x0, cfg.dt)
    J = 0.0
    for t in range(len(controls)):
        u_prev = controls[t - 1] if t > 0 else u_prev_external
        J += compute_step_cost(
            state.positions[t], controls[t], u_prev,
            idx_A, idx_B, neigh_idx, target_pos, cfg,
        )
    # Terminal cost
    J += cfg.beta_terminal * compute_step_cost(
        state.positions[-1], controls[-1], controls[-2] if len(controls) > 1 else u_prev_external,
        idx_A, idx_B, neigh_idx, target_pos, cfg,
    )
    return J


def solve_horizon(
    feval: ForceEvaluator,
    x0: np.ndarray,
    u_init: List[np.ndarray],
    idx_A: int,
    idx_B: int,
    neigh_idx: np.ndarray,
    target_pos: np.ndarray,
    cfg: MPCConfig,
    u_prev_external: Optional[np.ndarray] = None,
) -> Tuple[List[np.ndarray], float, List[float]]:
    """
    Solve the K-step MPC horizon optimisation via L-BFGS-B.

    Uses scipy.optimize.minimize with analytic adjoint gradients and
    Wolfe-condition line search.  Typically 3-5× fewer function evaluations
    than manual gradient descent.

    Returns
    -------
    controls : list of K arrays
    J_final : float
    J_history : list of floats
    """
    K = len(u_init)
    x0_flat = np.concatenate([u.copy() for u in u_init])
    J_history: List[float] = []

    # Bounds: repeat control bounds K times, narrowed by rate limits when available
    if cfg.du_max is not None and u_prev_external is not None:
        # Build per-step bounds that tighten based on cumulative rate limit
        bounds_list = []
        for k_step in range(K):
            # Controls can drift ±du_max per step from the one before.
            # From u_prev_external, step k can be at most (k+1)*du_max away.
            drift = (k_step + 1) * cfg.du_max
            for d in range(N_CTRL):
                lo_d = max(float(cfg.u_lo[d]), float(u_prev_external[d] - drift[d]))
                hi_d = min(float(cfg.u_hi[d]), float(u_prev_external[d] + drift[d]))
                bounds_list.append((lo_d, hi_d))
    else:
        bounds_list = [(lo, hi) for _ in range(K) for (lo, hi) in zip(cfg.u_lo, cfg.u_hi)]

    def _obj_and_grad(x_flat: np.ndarray) -> Tuple[float, np.ndarray]:
        controls_ = [np.clip(x_flat[t * N_CTRL:(t + 1) * N_CTRL], cfg.u_lo, cfg.u_hi)
                     for t in range(K)]
        # Note: cache is only cleared by run_mpc between replans, not here.
        # This lets L-BFGS-B reuse basis lookups across line-search evaluations
        # when the optimizer revisits similar vortex positions.
        state_ = rollout(feval, controls_, x0, cfg.dt)

        # Cost
        J = 0.0
        for t in range(K):
            u_prev = controls_[t - 1] if t > 0 else u_prev_external
            J += compute_step_cost(
                state_.positions[t], controls_[t], u_prev,
                idx_A, idx_B, neigh_idx, target_pos, cfg,
            )
        J += cfg.beta_terminal * compute_step_cost(
            state_.positions[-1], controls_[-1],
            controls_[-2] if K > 1 else u_prev_external,
            idx_A, idx_B, neigh_idx, target_pos, cfg,
        )
        J_history.append(J)

        # Adjoint gradient
        grads_ = compute_mpc_gradients(
            feval, state_, idx_A, idx_B, neigh_idx, target_pos, cfg,
            u_prev_external,
        )
        grad_flat = np.concatenate(grads_)
        return float(J), grad_flat

    result = minimize(
        _obj_and_grad,
        x0=x0_flat,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds_list,
        options={
            "maxiter": cfg.n_iters,
            "maxfun": 3 * cfg.n_iters,
            "ftol": 1e-15,
            "gtol": 1e-10,
        },
    )

    controls_out = [np.clip(result.x[t * N_CTRL:(t + 1) * N_CTRL], cfg.u_lo, cfg.u_hi)
                    for t in range(K)]
    return controls_out, float(result.fun), J_history


# ══════════════════════════════════════════════════════════════════
# Main MPC loop
# ══════════════════════════════════════════════════════════════════

@dataclass
class MPCResult:
    """Full result of an MPC run (convertible to TransportResult)."""
    positions: List[np.ndarray]           # (T+1,) each (n_particles, 2)
    applied_controls: List[np.ndarray]    # (T,) each (N_CTRL,)
    forces: List[Tuple[np.ndarray, np.ndarray]]  # (T,) each (Fx, Fy)
    J_history: List[float]                # outer-loop costs
    inner_J_histories: List[List[float]]  # per-replan convergence
    elapsed_s: float = 0.0
    merge_time_s: Optional[float] = None


def run_mpc(
    feval: ForceEvaluator,
    x0: np.ndarray,
    u_init: np.ndarray,
    idx_A: int,
    idx_B: int,
    neigh_idx: np.ndarray,
    target_pos: np.ndarray,
    cfg: MPCConfig,
    verbose: bool = True,
) -> MPCResult:
    """
    Run the full closed-loop MPC for T steps.

    Parameters
    ----------
    feval : ForceEvaluator
    x0 : (n_particles, 2) — initial positions
    u_init : (N_CTRL,) — initial control guess
    idx_A, idx_B : int — particle indices
    neigh_idx : int array — neighbour indices
    target_pos : (n_particles, 2) — home positions for all particles
    cfg : MPCConfig
    verbose : bool

    Returns
    -------
    MPCResult
    """
    pos = x0.copy()
    n_particles = pos.shape[0]
    step_scale = SCALE * cfg.dt

    # Initialise horizon with constant control
    u_horizon = [u_init.copy() for _ in range(cfg.K)]
    u_prev = None

    positions = [pos.copy()]
    applied_controls: List[np.ndarray] = []
    forces_list: List[Tuple[np.ndarray, np.ndarray]] = []
    J_history: List[float] = []
    inner_J_histories: List[List[float]] = []
    merge_time_s: Optional[float] = None

    t0 = time.perf_counter()

    if verbose:
        print(f"  MPC: T={cfg.T}, K={cfg.K}, replan_every={cfg.replan_every}, "
              f"n_iters={cfg.n_iters}")
        print(f"  Progress: ", end="", flush=True)

    for t in range(cfg.T):
        # Replan
        if t % cfg.replan_every == 0:
            feval.clear_cache()
            u_horizon, J_opt, J_hist = solve_horizon(
                feval, pos, u_horizon, idx_A, idx_B, neigh_idx,
                target_pos, cfg, u_prev,
            )
            J_history.append(J_opt)
            inner_J_histories.append(J_hist)

        # Apply first control — enforce per-step rate limit
        u_apply = u_horizon[0].copy()
        if cfg.du_max is not None and u_prev is not None:
            u_apply = np.clip(u_apply, u_prev - cfg.du_max, u_prev + cfg.du_max)
        u_apply = np.clip(u_apply, cfg.u_lo, cfg.u_hi)
        applied_controls.append(u_apply.copy())

        # Step dynamics
        Fx, Fy = feval.forces(u_apply, pos)
        forces_list.append((Fx.copy(), Fy.copy()))
        pos = pos.copy()
        pos[:, 0] += step_scale * Fx
        pos[:, 1] += step_scale * Fy
        positions.append(pos.copy())

        # Check merge
        if merge_time_s is None:
            d_AB = float(np.linalg.norm(pos[idx_A] - pos[idx_B]))
            if d_AB < CAPTURE_RADIUS:
                merge_time_s = (t + 1) * cfg.dt

        # Warm-start: shift horizon forward
        u_prev = u_apply.copy()
        u_horizon = u_horizon[1:] + [u_horizon[-1].copy()]

        if verbose and (t + 1) % max(1, cfg.T // 20) == 0:
            d_AB = float(np.linalg.norm(pos[idx_A] - pos[idx_B]) * 1e6)
            print(f"{t+1}", end=" ", flush=True)

    elapsed = time.perf_counter() - t0
    if verbose:
        print(f"\n  MPC done in {elapsed:.1f}s")

    return MPCResult(
        positions=positions,
        applied_controls=applied_controls,
        forces=forces_list,
        J_history=J_history,
        inner_J_histories=inner_J_histories,
        elapsed_s=elapsed,
        merge_time_s=merge_time_s,
    )


def mpc_result_to_transport(
    result: MPCResult,
    initial_positions: np.ndarray,
    dt: float,
) -> TransportResult:
    """Convert MPCResult to TransportResult for metrics / GIF rendering."""
    n_frames = len(result.positions)
    times = np.arange(n_frames, dtype=float) * dt
    trajectories = np.array(result.positions)  # (n_frames, n_particles, 2)

    controls = np.array(result.applied_controls) if result.applied_controls else np.zeros((0, N_CTRL))
    alphas = controls[:, I_ALPHA] if len(controls) > 0 else np.zeros(0)
    betas  = controls[:, I_BETA]  if len(controls) > 0 else np.ones(0)
    psis   = controls[:, I_PSI]   if len(controls) > 0 else np.zeros(0)
    centers = controls[:, [I_XV, I_YV]] if len(controls) > 0 else np.zeros((0, 2))

    # Pad to match positions length (positions has T+1 entries)
    if len(alphas) < n_frames:
        pad = n_frames - len(alphas)
        alphas  = np.concatenate([alphas,  np.zeros(pad)])
        betas   = np.concatenate([betas,   np.ones(pad)])
        centers = np.vstack([centers, np.tile(centers[-1:], (pad, 1))]) if len(centers) > 0 else np.zeros((n_frames, 2))

    labels = ["mpc"] * n_frames

    return TransportResult(
        times_s=times,
        trajectories=trajectories,
        alphas=alphas,
        betas_sw=betas,
        centers=centers,
        phase_labels=labels,
        initial_positions=initial_positions,
        merge_time_s=result.merge_time_s,
    )
