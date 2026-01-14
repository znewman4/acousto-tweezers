#tweezers/control/evaluator.py
from __future__ import annotations

"""Control-ready evaluator for trajectory optimisation.

This module wraps your existing pipeline:
    control -> forced 2.5D Helmholtz -> Gor'kov -> force -> overdamped step

into a *pure* callable that an optimiser can query thousands of times.

Important modelling choice (Option A):
We represent each bottom transducer as an *effective* complex-valued drive
footprint ``vb(x,y)`` confined to a thin band near the bottom (y≈0). This
lets the transducer move anywhere on the bottom *surface* (2D) while keeping
the current 2.5D planar solver.
"""

from dataclasses import dataclass
import numpy as np

from acousto.solvers import build_helmholtz_2d_forced_25d_operator
from acousto.force import ParticleProps, gorkov_potential_and_force_2d, bilinear_sample_vec


@dataclass(frozen=True)
class DishDomain:
    Lx: float
    Ly: float
    Nx: int
    Ny: int


@dataclass(frozen=True)
class MediumProps:
    f: float
    c0: float
    rho0: float
    loss_eta: float = 1e-3
    kz: float = 0.0
    coupling_alpha: float = 1.0


@dataclass(frozen=True)
class Control2Pucks:
    """Two transducers with 2D positions, amplitude and phase.

    Positions are in meters in the same coordinate system as the solver domain.
    Phases are in radians.
    """
    xA: float
    yA: float
    xB: float
    yB: float
    vA: float
    vB: float
    phiA: float
    phiB: float


@dataclass(frozen=True)
class EvaluatorConfig:
    sigma_x: float  # width of 1D bottom actuator footprint (meters)
    bottom_band: float
    dt: float
    viscosity: float
    border_penalty: float = 1e6
    smooth_u: float = 1e-2
    alpha_g: float = 1.0  # Gor'kov force scaling (1.0 = physical, >1 = scaled for development)
    max_step: float = 0.1e-3  # Maximum displacement per timestep (meters), prevents instability


class BottomFootprint25DEvaluator:
    """Fast per-step evaluator for optimisation / MPC.

    Outputs are *not* images. Each call returns:
      - predicted next particle position (xp1, yp1)
      - scalar loss (tracking + penalties)
      - diagnostics dict (optional introspection)

    Internally it still computes full-field U(x,y), Fx(x,y), Fy(x,y) each call.
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

    def control_to_bottom_vb(self, u: Control2Pucks) -> np.ndarray:
        """Map control to 1D bottom boundary normal velocity profile vb_x(x).
        
        Returns array of shape (Nx,) with bottom velocity at each x position.
        Uses Gaussian footprints centered at transducer x-positions.
        """
        x = self.op.x
        sigma_x = float(self.cfg.sigma_x)
        
        # Gaussian footprints for each transducer at their x positions
        gA = np.exp(-(x - u.xA)**2 / (2.0 * sigma_x * sigma_x))
        gB = np.exp(-(x - u.xB)**2 / (2.0 * sigma_x * sigma_x))
        
        # Combine: amplitude * phase * footprint
        vb_x = (u.vA * np.exp(1j * u.phiA) * gA) + (u.vB * np.exp(1j * u.phiB) * gB)
        return vb_x.astype(np.complex128)

    def clip_control(self, u: Control2Pucks) -> Control2Pucks:
        """Project control into feasible region (domain + bottom band + wrapped phase)."""
        Lx = float(self.domain.Lx)
        Ly = float(self.domain.Ly)
        y_max = min(float(self.cfg.bottom_band), Ly)

        xA = float(np.clip(u.xA, 0.0, Lx))
        xB = float(np.clip(u.xB, 0.0, Lx))
        yA = float(np.clip(u.yA, 0.0, y_max))
        yB = float(np.clip(u.yB, 0.0, y_max))

        vA = float(max(0.0, u.vA))
        vB = float(max(0.0, u.vB))

        phiA = self._wrap_angle(u.phiA)
        phiB = self._wrap_angle(u.phiB)

        return Control2Pucks(xA=xA, yA=yA, xB=xB, yB=yB, vA=vA, vB=vB, phiA=phiA, phiB=phiB)

    def step(
        self,
        *,
        xp: float,
        yp: float,
        target_x: float,
        target_y: float,
        u: Control2Pucks,
        u_prev: Control2Pucks | None = None,
        return_fields: bool = False,
    ):
        """One control step.

        Returns
        -------
        (xp1, yp1, loss, info) by default.
        If ``return_fields`` is True, returns (xp1, yp1, loss, info, field, U, Fx, Fy).
        """
        u = self.clip_control(u)
        vb_x = self.control_to_bottom_vb(u)  # 1D bottom velocity profile
        field = self.op.solve_for_bottom_vb(vb_x)  # Physically correct boundary condition
        U, Fx, Fy = gorkov_potential_and_force_2d(field, self.particle)
        
        # STEP 1 PATCH: Apply Gor'kov force scaling factor
        Fx = Fx * self.cfg.alpha_g
        Fy = Fy * self.cfg.alpha_g

        # Smooth force at particle position
        fx, fy = bilinear_sample_vec(field.x, field.y, Fx, Fy, float(xp), float(yp))

        # Overdamped update: xdot = F/gamma (Stokes drag)
        a = float(self.particle.a)
        gamma = 6.0 * np.pi * float(self.cfg.viscosity) * a

        # Compute raw displacement
        dx_raw = self.cfg.dt * (fx / gamma)
        dy_raw = self.cfg.dt * (fy / gamma)
        raw_displacement = np.sqrt(dx_raw**2 + dy_raw**2)
        
        # Velocity/displacement limiting to prevent instability with large alpha_g
        step_limited = False
        step_scale = 1.0
        if raw_displacement > self.cfg.max_step and raw_displacement > 0:
            step_scale = self.cfg.max_step / raw_displacement
            dx_raw *= step_scale
            dy_raw *= step_scale
            step_limited = True
        
        final_displacement = np.sqrt(dx_raw**2 + dy_raw**2)

        xp1 = float(xp + dx_raw)
        yp1 = float(yp + dy_raw)

        penalty = 0.0
        if (xp1 < self._x0) or (xp1 > self._x1) or (yp1 < self._y0) or (yp1 > self._y1):
            penalty += float(self.cfg.border_penalty)
        xp1 = float(np.clip(xp1, self._x0, self._x1))
        yp1 = float(np.clip(yp1, self._y0, self._y1))

        # Tracking loss at next step
        dx = xp1 - float(target_x)
        dy = yp1 - float(target_y)
        loss = dx * dx + dy * dy + penalty

        # Regularise control-rate (helps avoid jittery robot motion)
        if u_prev is not None:
            u_prev = self.clip_control(u_prev)
            du = np.array(
                [
                    u.xA - u_prev.xA,
                    u.yA - u_prev.yA,
                    u.xB - u_prev.xB,
                    u.yB - u_prev.yB,
                    u.vA - u_prev.vA,
                    u.vB - u_prev.vB,
                    self._wrap_angle(u.phiA - u_prev.phiA),
                    self._wrap_angle(u.phiB - u_prev.phiB),
                ],
                dtype=float,
            )
            loss += float(self.cfg.smooth_u) * float(np.dot(du, du))

        info = {
            "fx": float(fx),
            "fy": float(fy),
            "pmax": float(np.abs(field.p).max()),
            "Umin": float(np.min(U)),
            "Umax": float(np.max(U)),
            "control": u,
            # Step limiter diagnostics
            "step_limited": step_limited,
            "step_scale": float(step_scale),
            "raw_step_mm": float(raw_displacement * 1e3),
            "step_mm": float(final_displacement * 1e3),
        }

        if return_fields:
            return xp1, yp1, float(loss), info, field, U, Fx, Fy
        return xp1, yp1, float(loss), info

    def debug_force_at(self, xp: float, yp: float, u: Control2Pucks, warn_threshold: float = 1e-14):
        """Diagnostic: inspect forces at a given particle position.
        
        Parameters
        ----------
        xp, yp : float
            Particle position (meters).
        u : Control2Pucks
            Control input.
        warn_threshold : float
            Issue warning if |F| < warn_threshold (default 1e-14 N).
        
        Returns
        -------
        dict with keys:
            - 'p_mag': |p| at particle position (Pa)
            - 'grad_p_mag': max(|∇p|) in domain (Pa/m)
            - 'U': Gor'kov potential at particle (J)
            - 'grad_U_mag': max(|∇U|) in domain (J/m)
            - 'Fx', 'Fy': force at particle (N)
            - 'F_mag': |F| at particle (N)
            - 'warning': string if |F| < warn_threshold
        """
        u = self.clip_control(u)
        vb_x = self.control_to_bottom_vb(u)
        field = self.op.solve_for_bottom_vb(vb_x)
        U, Fx, Fy = gorkov_potential_and_force_2d(field, self.particle)
        
        # Apply force scaling
        Fx = Fx * self.cfg.alpha_g
        Fy = Fy * self.cfg.alpha_g
        
        # Interpolate at particle position
        p_real, p_imag = bilinear_sample_vec(field.x, field.y, field.p.real, field.p.imag, float(xp), float(yp))
        p_at_particle = complex(p_real, p_imag)
        U_real, U_imag = bilinear_sample_vec(field.x, field.y, U.real, U.imag, float(xp), float(yp))
        U_at_particle = complex(U_real, U_imag)
        fx, fy = bilinear_sample_vec(field.x, field.y, Fx, Fy, float(xp), float(yp))
        
        # Compute gradients
        dx = field.x[1] - field.x[0]
        dy = field.y[1] - field.y[0]
        dpdy, dpdx = np.gradient(field.p, dy, dx, edge_order=2)
        dUdy, dUdx = np.gradient(U, dy, dx, edge_order=2)
        
        grad_p_mag = float(np.sqrt(np.abs(dpdx)**2 + np.abs(dpdy)**2).max())
        grad_U_mag = float(np.sqrt(np.abs(dUdx)**2 + np.abs(dUdy)**2).max())
        F_mag = float(np.sqrt(fx**2 + fy**2))
        
        result = {
            'p_mag': float(np.abs(p_at_particle)),
            'grad_p_mag': grad_p_mag,
            'U': float(np.abs(U_at_particle)) if U_at_particle != 0 else 0.0,
            'grad_U_mag': grad_U_mag,
            'Fx': float(fx),
            'Fy': float(fy),
            'F_mag': F_mag,
            'alpha_g': self.cfg.alpha_g,
        }
        
        if F_mag < warn_threshold:
            result['warning'] = (
                f"⚠️  Force too small: |F| = {F_mag:.3e} N < {warn_threshold:.3e} N threshold.\n"
                f"    This may indicate an unphysical force model.\n"
                f"    Current alpha_g = {self.cfg.alpha_g:.3e}.\n"
                f"    Consider increasing alpha_g for development, or implement 3D solver for physics correction."
            )
        else:
            result['warning'] = None
        
        return result

    @staticmethod
    def _wrap_angle(a: float) -> float:
        return float((a + np.pi) % (2.0 * np.pi) - np.pi)
