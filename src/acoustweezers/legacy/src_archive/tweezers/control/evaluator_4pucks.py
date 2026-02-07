"""
4-Puck Evaluator for Acoustic Tweezers Control with Gating.

Extends Evaluator3Pucks to support 4-transducer configurations with ON/OFF gating.
Provides enhanced 2D control authority via additional transducer and dynamic gating.

Usage:
    from tweezers.control.evaluator_4pucks import Evaluator4Pucks
    ev = Evaluator4Pucks(domain, medium, particle, cfg)
    result = ev.step(xp=..., yp=..., target_x=..., target_y=..., u=Control4Pucks(...))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

from .evaluator import DishDomain, MediumProps, EvaluatorConfig
from .pucks_4 import Control4Pucks, control_to_forcing_band_vb_4pucks
from acousto.solvers import build_helmholtz_2d_forced_25d_operator
from acousto.force import ParticleProps, gorkov_potential_and_force_2d, bilinear_sample_vec
from acousto.analysis import find_trap_center, TrapCenterResult


@dataclass(frozen=True)
class EvaluatorConfig4Pucks(EvaluatorConfig):
    """
    Configuration for 4-puck evaluator.
    
    Inherits all parameters from EvaluatorConfig and adds 4-puck specific ones.
    """
    # Separation constraint penalty
    separation_penalty_weight: float = 1e6
    min_puck_separation: float = 0.1e-3


class Evaluator4Pucks:
    """
    Control evaluator for 4-transducer acoustic tweezers with gating.
    
    Similar to Evaluator3Pucks but with:
    - 4 transducer support via Control4Pucks
    - Per-transducer ON/OFF gating
    - Enhanced 2D reachability with transducer D
    - Pairwise separation safety constraints (only for active transducers)
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
    
    def control_to_forcing_band_vb(self, u: Control4Pucks) -> np.ndarray:
        """Map 4-puck control to bottom boundary velocity profile with gating."""
        sigma_x = float(self.cfg.sigma_x)
        sigma_y = float(getattr(self.cfg, 'sigma_y', sigma_x * 0.5))
        
        return control_to_forcing_band_vb_4pucks(u, self.op.x, sigma_x, sigma_y)
    
    def clip_control(self, u: Control4Pucks) -> Control4Pucks:
        """Project control into feasible region."""
        Lx = float(self.domain.Lx)
        Ly = float(self.domain.Ly)
        y_max = min(float(self.cfg.bottom_band), Ly)
        
        # For transducers C and D, allow higher y for non-collinear forcing
        y_max_CD = Ly  # C and D can go anywhere in the domain
        
        def wrap_angle(a: float) -> float:
            return float((a + np.pi) % (2.0 * np.pi) - np.pi)
        
        return Control4Pucks(
            xA=float(np.clip(u.xA, 0.0, Lx)),
            yA=float(np.clip(u.yA, 0.0, y_max)),
            vA=float(max(0.0, u.vA)),
            phiA=wrap_angle(u.phiA),
            gateA=u.gateA,
            xB=float(np.clip(u.xB, 0.0, Lx)),
            yB=float(np.clip(u.yB, 0.0, y_max)),
            vB=float(max(0.0, u.vB)),
            phiB=wrap_angle(u.phiB),
            gateB=u.gateB,
            xC=float(np.clip(u.xC, 0.0, Lx)),
            yC=float(np.clip(u.yC, 0.0, y_max_CD)),  # C can go higher
            vC=float(max(0.0, u.vC)),
            phiC=wrap_angle(u.phiC),
            gateC=u.gateC,
            xD=float(np.clip(u.xD, 0.0, Lx)),
            yD=float(np.clip(u.yD, 0.0, y_max_CD)),  # D can go higher
            vD=float(max(0.0, u.vD)),
            phiD=wrap_angle(u.phiD),
            gateD=u.gateD,
        )
    
    def compute_separation_penalty(self, u: Control4Pucks) -> float:
        """
        Compute penalty for active transducers being too close.
        
        Only penalizes pairs where both transducers are gated-on.
        """
        min_sep = getattr(self.cfg, 'min_puck_separation', 0.1e-3)
        weight = getattr(self.cfg, 'separation_penalty_weight', 1e6)
        
        # Only consider active transducers
        active_positions = []
        if u.gateA:
            active_positions.append((u.xA, u.yA, "A"))
        if u.gateB:
            active_positions.append((u.xB, u.yB, "B"))
        if u.gateC:
            active_positions.append((u.xC, u.yC, "C"))
        if u.gateD:
            active_positions.append((u.xD, u.yD, "D"))
        
        penalty = 0.0
        for i, (x1, y1, _) in enumerate(active_positions):
            for x2, y2, _ in active_positions[i+1:]:
                d = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
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
        u: Control4Pucks,
        u_prev: Optional[Control4Pucks] = None,
        return_fields: bool = False,
        return_metrics: bool = True,
    ):
        """
        One control step with 4 transducers and gating.
        
        Parameters
        ----------
        xp, yp : float
            Current particle position (meters).
        target_x, target_y : float
            Target particle position (meters).
        u : Control4Pucks
            Current 4-transducer control configuration with gates.
        u_prev : Control4Pucks, optional
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
        
        # Compute boundary forcing from 4 transducers (respecting gates)
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
        
        # Separation penalty (only for active transducers)
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
                u.xD - u_prev.xD, u.yD - u_prev.yD, u.vD - u_prev.vD,
                # Phase differences (wrapped)
                self._wrap_angle(u.phiA - u_prev.phiA),
                self._wrap_angle(u.phiB - u_prev.phiB),
                self._wrap_angle(u.phiC - u_prev.phiC),
                self._wrap_angle(u.phiD - u_prev.phiD),
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
            "active_transducers": u.active_count(),
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
        """
        metrics = {}
        
        # Solver stats
        metrics["solver_time_ms"] = float(solver_time_ms)
        metrics["solver_residual"] = None
        
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
        
        # Field stats
        p_abs = np.abs(field.p)
        metrics["pressure_max"] = float(np.max(p_abs))
        metrics["pressure_mean"] = float(np.mean(p_abs))
        
        # Gor'kov potential stats
        U_finite = U[np.isfinite(U)]
        if len(U_finite) > 0:
            metrics["U_min"] = float(np.min(U_finite))
            metrics["U_max"] = float(np.max(U_finite))
            metrics["U_mean"] = float(np.mean(U_finite))
            metrics["U_std"] = float(np.std(U_finite))
        else:
            metrics["U_min"] = np.nan
            metrics["U_max"] = np.nan
            metrics["U_mean"] = np.nan
            metrics["U_std"] = np.nan
        
        # Force stats
        F_mag = np.sqrt(Fx**2 + Fy**2)
        F_finite = F_mag[np.isfinite(F_mag)]
        if len(F_finite) > 0:
            metrics["F_max"] = float(np.max(F_finite))
            metrics["F_mean"] = float(np.mean(F_finite))
        else:
            metrics["F_max"] = np.nan
            metrics["F_mean"] = np.nan
        
        # Force at particle
        fx_at_p, fy_at_p = bilinear_sample_vec(
            field.x, field.y, Fx, Fy, float(particle_x), float(particle_y)
        )
        metrics["Fx_at_particle"] = float(fx_at_p)
        metrics["Fy_at_particle"] = float(fy_at_p)
        metrics["F_mag_at_particle"] = float(np.sqrt(fx_at_p**2 + fy_at_p**2))
        
        return metrics
    
    @staticmethod
    def _wrap_angle(a: float) -> float:
        """Wrap angle to [-π, π]."""
        return float((a + np.pi) % (2.0 * np.pi) - np.pi)
    
    def compute_force_at_point(
        self,
        u: Control4Pucks,
        xp: float,
        yp: float,
        return_fields: bool = False,
    ):
        """
        Compute the Gor'kov force at a specific point without stepping.
        
        Useful for force-field analysis and control authority diagnostics.
        
        Parameters
        ----------
        u : Control4Pucks
            Control configuration.
        xp, yp : float
            Point at which to evaluate force.
        return_fields : bool
            If True, also return field, U, Fx, Fy arrays.
        
        Returns
        -------
        (fx, fy) or (fx, fy, field, U, Fx, Fy) if return_fields=True
        """
        u = self.clip_control(u)
        vb_x = self.control_to_forcing_band_vb(u)
        field = self.op.solve_for_bottom_vb(vb_x)
        U, Fx, Fy = gorkov_potential_and_force_2d(field, self.particle)
        
        Fx = Fx * self.cfg.alpha_g
        Fy = Fy * self.cfg.alpha_g
        
        fx, fy = bilinear_sample_vec(field.x, field.y, Fx, Fy, float(xp), float(yp))
        
        if return_fields:
            return float(fx), float(fy), field, U, Fx, Fy
        return float(fx), float(fy)
