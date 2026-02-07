"""
3-Puck Evaluator for Acoustic Tweezers Control

Extends BottomFootprint25DEvaluator to support 3-transducer configurations.
Provides full 2D control authority via non-collinear forcing.

Usage:
    from tweezers.control.evaluator_3pucks import Evaluator3Pucks
    ev = Evaluator3Pucks(domain, medium, particle, cfg)
    result = ev.step(xp=..., yp=..., target_x=..., target_y=..., u=Control3Pucks(...))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

from .evaluator import DishDomain, MediumProps, EvaluatorConfig, BottomFootprint25DEvaluator
from .pucks_3 import Control3Pucks, control_to_forcing_band_vb_3pucks
from acousto.solvers import build_helmholtz_2d_forced_25d_operator
from acousto.force import ParticleProps, gorkov_potential_and_force_2d, bilinear_sample_vec
from acousto.analysis import find_trap_center, TrapCenterResult


@dataclass(frozen=True)
class EvaluatorConfig3Pucks(EvaluatorConfig):
    """
    Configuration for 3-puck evaluator.
    
    Inherits all parameters from EvaluatorConfig and adds 3-puck specific ones.
    """
    # Separation constraint penalty
    separation_penalty_weight: float = 1e6
    min_puck_separation: float = 0.1e-3


class Evaluator3Pucks:
    """
    Control evaluator for 3-transducer acoustic tweezers.
    
    Similar to BottomFootprint25DEvaluator but with:
    - 3 transducer support via Control3Pucks
    - Full 2D reachability (when transducer C is at different y)
    - Pairwise separation safety constraints
    """
    
    def __init__(
        self,
        domain: DishDomain,
        medium: MediumProps,
        particle: ParticleProps,
        cfg: EvaluatorConfig,
        *,
        left_type: str = "neumann",
        right_type: str = "neumann",
        bottom_type: str = "neumann",
        top_type: str = "neumann",
    ) -> None:
        self.domain = domain
        self.medium = medium
        self.particle = particle
        self.cfg = cfg
        
        self.op = build_helmholtz_2d_forced_25d_operator(
            Lx=domain.Lx,
            Ly=domain.Ly,
            Nx=domain.Nx,
            Ny=domain.Ny,
            f=medium.f,
            c0=medium.c0,
            rho0=medium.rho0,
            left_type=left_type,
            right_type=right_type,
            bottom_type=bottom_type,
            top_type=top_type,
            left=0.0,
            right=0.0,
            bottom=0.0,
            top=0.0,
            kz=medium.kz,
            coupling_alpha=medium.coupling_alpha,
            loss_eta=medium.loss_eta,
        )
        
        # Cached bounds
        self._x0 = float(self.op.x[0])
        self._x1 = float(self.op.x[-1])
        self._y0 = float(self.op.y[0])
        self._y1 = float(self.op.y[-1])
    
    def control_to_forcing_band_vb(self, u: Control3Pucks) -> np.ndarray:
        """Map 3-puck control to bottom boundary velocity profile."""
        sigma_x = float(self.cfg.sigma_x)
        sigma_y = float(getattr(self.cfg, 'sigma_y', sigma_x * 0.5))
        
        return control_to_forcing_band_vb_3pucks(u, self.op.x, sigma_x, sigma_y)
    
    def clip_control(self, u: Control3Pucks) -> Control3Pucks:
        """Project control into feasible region."""
        Lx = float(self.domain.Lx)
        Ly = float(self.domain.Ly)
        y_max = min(float(self.cfg.bottom_band), Ly)
        
        # For transducer C, allow higher y for non-collinear forcing
        y_max_C = Ly  # C can go anywhere in the domain
        
        def wrap_angle(a: float) -> float:
            return float((a + np.pi) % (2.0 * np.pi) - np.pi)
        
        return Control3Pucks(
            xA=float(np.clip(u.xA, 0.0, Lx)),
            yA=float(np.clip(u.yA, 0.0, y_max)),
            vA=float(max(0.0, u.vA)),
            phiA=wrap_angle(u.phiA),
            xB=float(np.clip(u.xB, 0.0, Lx)),
            yB=float(np.clip(u.yB, 0.0, y_max)),
            vB=float(max(0.0, u.vB)),
            phiB=wrap_angle(u.phiB),
            xC=float(np.clip(u.xC, 0.0, Lx)),
            yC=float(np.clip(u.yC, 0.0, y_max_C)),  # C can go higher
            vC=float(max(0.0, u.vC)),
            phiC=wrap_angle(u.phiC),
        )
    
    def compute_separation_penalty(self, u: Control3Pucks) -> float:
        """Compute penalty for transducers being too close."""
        min_sep = getattr(self.cfg, 'min_puck_separation', 0.1e-3)
        weight = getattr(self.cfg, 'separation_penalty_weight', 1e6)
        
        dAB = np.sqrt((u.xA - u.xB)**2 + (u.yA - u.yB)**2)
        dAC = np.sqrt((u.xA - u.xC)**2 + (u.yA - u.yC)**2)
        dBC = np.sqrt((u.xB - u.xC)**2 + (u.yB - u.yC)**2)
        
        penalty = 0.0
        for d in [dAB, dAC, dBC]:
            if d < min_sep:
                penalty += weight * (min_sep - d)**2
        
        return penalty
    
    def step(
        self,
        *,
        xp: float,
        yp: float,
        target_x: float,
        target_y: float,
        u: Control3Pucks,
        u_prev: Optional[Control3Pucks] = None,
        return_fields: bool = False,
        return_metrics: bool = True,
    ):
        """
        One control step with 3 transducers.
        
        Parameters
        ----------
        xp, yp : float
            Current particle position (meters).
        target_x, target_y : float
            Target particle position (meters).
        u : Control3Pucks
            Current 3-transducer control configuration.
        u_prev : Control3Pucks, optional
            Previous control for rate penalty.
        return_fields : bool
            If True, also return field, U, Fx, Fy.
        return_metrics : bool
            If True (default), include scalar field metrics in info dict.
        
        Returns
        -------
        (xp1, yp1, loss, info) by default.
        If return_fields=True: (xp1, yp1, loss, info, field, U, Fx, Fy)
        """
        import time
        solver_start = time.perf_counter()
        
        u = self.clip_control(u)
        
        # Compute boundary forcing from 3 transducers
        vb_x = self.control_to_forcing_band_vb(u)
        field = self.op.solve_for_bottom_vb(vb_x)
        
        solver_time_ms = (time.perf_counter() - solver_start) * 1000.0
        
        U, Fx, Fy = gorkov_potential_and_force_2d(field, self.particle)
        
        # Apply Gor'kov force scaling
        Fx = Fx * self.cfg.alpha_g
        Fy = Fy * self.cfg.alpha_g
        
        # Sample force at particle position
        fx, fy = bilinear_sample_vec(field.x, field.y, Fx, Fy, float(xp), float(yp))
        
        # Overdamped dynamics
        a = float(self.particle.a)
        gamma = 6.0 * np.pi * float(self.cfg.viscosity) * a
        
        dx_raw = self.cfg.dt * (fx / gamma)
        dy_raw = self.cfg.dt * (fy / gamma)
        raw_displacement = np.sqrt(dx_raw**2 + dy_raw**2)
        
        # Step limiting
        step_limited = False
        if raw_displacement > self.cfg.max_step and raw_displacement > 0:
            scale = self.cfg.max_step / raw_displacement
            dx_raw *= scale
            dy_raw *= scale
            step_limited = True
        
        xp1 = float(xp + dx_raw)
        yp1 = float(yp + dy_raw)
        
        # Penalties
        penalty = 0.0
        
        # Border penalty
        if (xp1 < self._x0) or (xp1 > self._x1) or (yp1 < self._y0) or (yp1 > self._y1):
            penalty += float(self.cfg.border_penalty)
        
        xp1 = float(np.clip(xp1, self._x0, self._x1))
        yp1 = float(np.clip(yp1, self._y0, self._y1))
        
        # Separation penalty
        penalty += self.compute_separation_penalty(u)
        
        # Tracking loss
        dx = xp1 - float(target_x)
        dy = yp1 - float(target_y)
        loss = dx * dx + dy * dy + penalty
        
        # Control rate penalty
        if u_prev is not None:
            u_prev = self.clip_control(u_prev)
            du = np.array([
                u.xA - u_prev.xA, u.yA - u_prev.yA, u.vA - u_prev.vA,
                u.xB - u_prev.xB, u.yB - u_prev.yB, u.vB - u_prev.vB,
                u.xC - u_prev.xC, u.yC - u_prev.yC, u.vC - u_prev.vC,
                # Phase differences (wrapped)
                self._wrap_angle(u.phiA - u_prev.phiA),
                self._wrap_angle(u.phiB - u_prev.phiB),
                self._wrap_angle(u.phiC - u_prev.phiC),
            ])
            loss += float(self.cfg.smooth_u) * np.dot(du, du)
        
        # Info dict
        info = {
            "fx": float(fx),
            "fy": float(fy),
            "penalty": float(penalty),
            "step_limited": step_limited,
            "xp1": float(xp1),
            "yp1": float(yp1),
            "displacement": float(np.sqrt(dx_raw**2 + dy_raw**2)),
        }
        
        # Add scalar field metrics if requested (default True)
        if return_metrics:
            info["metrics"] = self._compute_step_metrics(
                field, U, Fx, Fy, vb_x, solver_time_ms, xp, yp
            )
        
        if return_fields:
            return xp1, yp1, loss, info, field, U, Fx, Fy
        return xp1, yp1, loss, info
    
    def _compute_step_metrics(
        self,
        field,
        U: np.ndarray,
        Fx: np.ndarray,
        Fy: np.ndarray,
        vb: np.ndarray,
        solver_time_ms: float,
        particle_x: float,
        particle_y: float,
    ) -> dict:
        """
        Compute scalar metrics from fields for diagnostics.
        
        These metrics are computed every step regardless of whether
        field arrays are returned, enabling comprehensive diagnostics.
        """
        metrics = {}
        
        # Solver stats
        metrics["solver_time_ms"] = float(solver_time_ms)
        metrics["solver_residual"] = None  # Not available from direct solve
        
        # Forcing stats
        vb_abs = np.abs(vb) if np.iscomplexobj(vb) else vb
        vb_finite = vb_abs[np.isfinite(vb_abs)]
        if len(vb_finite) > 0:
            metrics["forcing_vb_min"] = float(np.min(vb_finite))
            metrics["forcing_vb_max"] = float(np.max(vb_finite))
            metrics["forcing_vb_mean"] = float(np.mean(vb_finite))
        else:
            metrics["forcing_vb_min"] = np.nan
            metrics["forcing_vb_max"] = np.nan
            metrics["forcing_vb_mean"] = np.nan
        
        # Pressure field stats (|p|)
        p_abs = np.abs(field.p) if np.iscomplexobj(field.p) else field.p
        total_p = p_abs.size
        nan_p = np.sum(np.isnan(p_abs))
        inf_p = np.sum(np.isinf(p_abs))
        metrics["p_nan_frac"] = float(nan_p / total_p)
        metrics["p_inf_frac"] = float(inf_p / total_p)
        
        p_finite = p_abs[np.isfinite(p_abs)]
        if len(p_finite) > 0:
            metrics["p_min"] = float(np.min(p_finite))
            metrics["p_max"] = float(np.max(p_finite))
            metrics["p_mean"] = float(np.mean(p_finite))
            metrics["p_std"] = float(np.std(p_finite))
        else:
            metrics["p_min"] = np.nan
            metrics["p_max"] = np.nan
            metrics["p_mean"] = np.nan
            metrics["p_std"] = np.nan
        
        # Gor'kov potential stats (raw U, before alpha_g scaling)
        total_U = U.size
        nan_U = np.sum(np.isnan(U))
        inf_U = np.sum(np.isinf(U))
        metrics["U_nan_frac"] = float(nan_U / total_U)
        metrics["U_inf_frac"] = float(inf_U / total_U)
        
        U_finite = U[np.isfinite(U)]
        if len(U_finite) > 0:
            metrics["U_min"] = float(np.min(U_finite))
            metrics["U_max"] = float(np.max(U_finite))
            metrics["U_ptp"] = float(np.ptp(U_finite))
            metrics["U_std"] = float(np.std(U_finite))
        else:
            metrics["U_min"] = np.nan
            metrics["U_max"] = np.nan
            metrics["U_ptp"] = np.nan
            metrics["U_std"] = np.nan
        
        # Force stats (scaled by alpha_g)
        Fx_scaled = Fx * self.cfg.alpha_g
        Fy_scaled = Fy * self.cfg.alpha_g
        
        Fx_finite = Fx_scaled[np.isfinite(Fx_scaled)]
        Fy_finite = Fy_scaled[np.isfinite(Fy_scaled)]
        
        if len(Fx_finite) > 0:
            metrics["Fx_max"] = float(np.max(np.abs(Fx_finite)))
            metrics["Fx_mean"] = float(np.mean(Fx_finite))
        else:
            metrics["Fx_max"] = np.nan
            metrics["Fx_mean"] = np.nan
        
        if len(Fy_finite) > 0:
            metrics["Fy_max"] = float(np.max(np.abs(Fy_finite)))
            metrics["Fy_mean"] = float(np.mean(Fy_finite))
        else:
            metrics["Fy_max"] = np.nan
            metrics["Fy_mean"] = np.nan
        
        Fmag = np.sqrt(Fx_scaled**2 + Fy_scaled**2)
        Fmag_finite = Fmag[np.isfinite(Fmag)]
        if len(Fmag_finite) > 0:
            metrics["Fmag_max"] = float(np.max(Fmag_finite))
        else:
            metrics["Fmag_max"] = np.nan
        
        # Trap metrics (find trap near current particle position)
        # Note: We pass raw U and Fx/Fy (not scaled by alpha_g) to trap finder
        # This is correct - trap finding should use the raw potential field
        trap_result = find_trap_center(
            field.x, field.y, U, Fx, Fy,
            particle_x=particle_x, particle_y=particle_y,
            search_radius=0.4e-3,
        )
        
        # Log first trap find attempt for debugging (only on first call)
        if not hasattr(self, '_trap_find_logged'):
            self._trap_find_logged = True
            # Log diagnostic info about the trap finding
            print(f"[Evaluator3Pucks] Trap finder diagnostic (first step):")
            print(f"  U range: [{metrics.get('U_min', np.nan):.2e}, {metrics.get('U_max', np.nan):.2e}]")
            print(f"  U_ptp: {metrics.get('U_ptp', np.nan):.2e}")
            print(f"  Trap found: {trap_result.x is not None and np.isfinite(trap_result.x)}")
            print(f"  Trap stable: {trap_result.is_stable}")
            if trap_result.x is not None and np.isfinite(trap_result.x):
                print(f"  Trap position: ({trap_result.x*1e3:.4f}, {trap_result.y*1e3:.4f}) mm")
                if trap_result.stiffness_eigvals is not None:
                    print(f"  Stiffness eigenvalues: {trap_result.stiffness_eigvals}")
                    print(f"  (positive = stable minimum, negative = unstable)")
                print(f"  Hessian: Uxx={trap_result.hess_xx:.2e}, Uxy={trap_result.hess_xy:.2e}, Uyy={trap_result.hess_yy:.2e}")
                print(f"  Gradient norm: {trap_result.grad_norm:.2e}")
        
        # Always report if trap was found (regardless of stability)
        trap_found = trap_result.x is not None and np.isfinite(trap_result.x)
        metrics["trap_found"] = trap_found
        metrics["trap_stable"] = trap_result.is_stable
        
        # trap_candidate_* - ALWAYS set when trap is found, regardless of stability
        # This is for controller use - always have a target to steer toward
        if trap_found:
            metrics["trap_candidate_x"] = float(trap_result.x)
            metrics["trap_candidate_y"] = float(trap_result.y)
            metrics["trap_candidate_depth"] = float(trap_result.depth) if np.isfinite(trap_result.depth) else np.nan
        else:
            metrics["trap_candidate_x"] = np.nan
            metrics["trap_candidate_y"] = np.nan
            metrics["trap_candidate_depth"] = np.nan
        
        # trap_x/trap_y - Only set when certified stable (for high-confidence operations)
        if trap_result.is_stable:
            metrics["trap_x"] = float(trap_result.x)
            metrics["trap_y"] = float(trap_result.y)
        else:
            metrics["trap_x"] = np.nan
            metrics["trap_y"] = np.nan
        
        # Stiffness eigenvalues (always report if found)
        if trap_found and trap_result.stiffness_eigvals is not None:
            eigs = np.sort(trap_result.stiffness_eigvals)
            metrics["stiff_eig_1"] = float(eigs[0])
            metrics["stiff_eig_2"] = float(eigs[1]) if len(eigs) > 1 else float(eigs[0])
            metrics["stiff_min"] = float(np.min(eigs))
        else:
            metrics["stiff_eig_1"] = np.nan
            metrics["stiff_eig_2"] = np.nan
            metrics["stiff_min"] = np.nan
        
        # Hessian components for debugging
        if trap_found:
            metrics["U_hess_xx"] = float(trap_result.hess_xx)
            metrics["U_hess_xy"] = float(trap_result.hess_xy)
            metrics["U_hess_yy"] = float(trap_result.hess_yy)
            metrics["trap_grad_norm"] = float(trap_result.grad_norm)
        else:
            metrics["U_hess_xx"] = np.nan
            metrics["U_hess_xy"] = np.nan
            metrics["U_hess_yy"] = np.nan
            metrics["trap_grad_norm"] = np.nan
        
        metrics["trap_depth"] = float(trap_result.depth) if trap_found and np.isfinite(trap_result.depth) else np.nan
        
        # Add proxy trap = global argmin(U) - always available as planning fallback
        U_finite = U.copy()
        U_finite[~np.isfinite(U_finite)] = np.inf
        if np.any(np.isfinite(U_finite)):
            proxy_idx = np.unravel_index(np.argmin(U_finite), U_finite.shape)
            metrics["proxy_trap_x"] = float(field.x[proxy_idx[1]])
            metrics["proxy_trap_y"] = float(field.y[proxy_idx[0]])
            metrics["proxy_trap_U"] = float(U[proxy_idx])
        else:
            metrics["proxy_trap_x"] = np.nan
            metrics["proxy_trap_y"] = np.nan
            metrics["proxy_trap_U"] = np.nan
        
        return metrics
    
    def find_trap(
        self,
        u: Control3Pucks,
        search_x: float,
        search_y: float,
        search_radius: float = 0.3e-3,
    ) -> TrapCenterResult:
        """Find trap center near a search point for given control."""
        u = self.clip_control(u)
        vb_x = self.control_to_forcing_band_vb(u)
        field = self.op.solve_for_bottom_vb(vb_x)
        U, Fx, Fy = gorkov_potential_and_force_2d(field, self.particle)
        
        return find_trap_center(
            field.x, field.y, U, Fx, Fy,
            particle_x=search_x, particle_y=search_y,
            search_radius=search_radius,
        )
    
    @staticmethod
    def _wrap_angle(a: float) -> float:
        """Wrap angle to [-π, π]."""
        return float((a + np.pi) % (2.0 * np.pi) - np.pi)
