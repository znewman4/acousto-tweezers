"""
3-Transducer (3-Puck) Extension for Acoustic Tweezers Control

This module extends the 2-transducer system to 3 transducers for better 2D control.
With 2 transducers, the interference pattern is fundamentally limited in y-control.
3 transducers provide:
- Full 2D control authority via non-collinear forcing
- Richer trap landscape with more stable traps
- Better reachability for large trajectories

Key additions:
- Control3Pucks: 3-transducer control dataclass
- ControlVectorNPucks: Generalized N-puck control vector (12 dims for N=3)
- control_to_forcing_band_vb_3pucks: Sum of 3 footprints
- Safety: pairwise separation constraints

Usage:
    from tweezers.control.pucks_3 import Control3Pucks, ControlVector3Pucks
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass(frozen=True)
class Control3Pucks:
    """
    Three transducers with 2D positions, amplitude and phase.
    
    Positions are in meters in the same coordinate system as the solver domain.
    Phases are in radians.
    
    Transducers A and B are on the bottom (y ≈ 0), transducer C can be
    positioned differently (e.g., side or top) to break the collinear forcing limit.
    """
    # Transducer A (bottom-left typical)
    xA: float
    yA: float
    vA: float
    phiA: float
    
    # Transducer B (bottom-right typical)
    xB: float
    yB: float
    vB: float
    phiB: float
    
    # Transducer C (can be at different y for better coverage)
    xC: float
    yC: float
    vC: float
    phiC: float
    
    def to_array(self) -> np.ndarray:
        """Convert to 12-element array."""
        return np.array([
            self.xA, self.yA, self.vA, self.phiA,
            self.xB, self.yB, self.vB, self.phiB,
            self.xC, self.yC, self.vC, self.phiC,
        ], dtype=np.float64)
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> "Control3Pucks":
        """Create from 12-element array."""
        return cls(
            xA=float(arr[0]), yA=float(arr[1]), vA=float(arr[2]), phiA=float(arr[3]),
            xB=float(arr[4]), yB=float(arr[5]), vB=float(arr[6]), phiB=float(arr[7]),
            xC=float(arr[8]), yC=float(arr[9]), vC=float(arr[10]), phiC=float(arr[11]),
        )


@dataclass
class ControlBounds3Pucks:
    """Bounds for 3-puck control variables."""
    x_min: float = 0.0
    x_max: float = 2e-3
    y_min: float = 0.0
    y_max: float = 0.25e-3   # Transducers confined to bottom band (can relax for C)
    y_max_C: float = 2e-3    # Transducer C can go higher for better coverage
    v_min: float = 0.0
    v_max: float = 2e-3
    phi_min: float = -np.pi
    phi_max: float = np.pi
    
    # Safety: minimum separation between any two transducers
    min_separation: float = 0.1e-3


@dataclass
class ControlRateLimits3Pucks:
    """Per-step rate limits for 3-puck control variables."""
    dx_max: float = 0.1e-3
    dy_max: float = 0.05e-3
    dv_max: float = 2e-4
    dphi_max: float = 0.5


@dataclass
class ControlVector3Pucks:
    """
    Full control parameterization for three-transducer system.
    
    Control vector u = [xA, yA, vA, phiA, xB, yB, vB, phiB, xC, yC, vC, phiC]
    
    This is a 12-dimensional control space (vs 8 for 2 pucks).
    """
    xA: float
    yA: float
    vA: float
    phiA: float
    
    xB: float
    yB: float
    vB: float
    phiB: float
    
    xC: float
    yC: float
    vC: float
    phiC: float
    
    bounds: ControlBounds3Pucks = field(default_factory=ControlBounds3Pucks)
    rate_limits: ControlRateLimits3Pucks = field(default_factory=ControlRateLimits3Pucks)
    
    @classmethod
    def from_control3pucks(
        cls,
        u: Control3Pucks,
        bounds: Optional[ControlBounds3Pucks] = None,
        rate_limits: Optional[ControlRateLimits3Pucks] = None,
    ) -> "ControlVector3Pucks":
        """Create ControlVector3Pucks from Control3Pucks."""
        return cls(
            xA=u.xA, yA=u.yA, vA=u.vA, phiA=u.phiA,
            xB=u.xB, yB=u.yB, vB=u.vB, phiB=u.phiB,
            xC=u.xC, yC=u.yC, vC=u.vC, phiC=u.phiC,
            bounds=bounds or ControlBounds3Pucks(),
            rate_limits=rate_limits or ControlRateLimits3Pucks(),
        )
    
    def to_control3pucks(self) -> Control3Pucks:
        """Convert to Control3Pucks."""
        return Control3Pucks(
            xA=self.xA, yA=self.yA, vA=self.vA, phiA=self.phiA,
            xB=self.xB, yB=self.yB, vB=self.vB, phiB=self.phiB,
            xC=self.xC, yC=self.yC, vC=self.vC, phiC=self.phiC,
        )
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array (12 elements)."""
        return np.array([
            self.xA, self.yA, self.vA, self.phiA,
            self.xB, self.yB, self.vB, self.phiB,
            self.xC, self.yC, self.vC, self.phiC,
        ], dtype=np.float64)
    
    @classmethod
    def from_array(
        cls,
        arr: np.ndarray,
        bounds: Optional[ControlBounds3Pucks] = None,
        rate_limits: Optional[ControlRateLimits3Pucks] = None,
    ) -> "ControlVector3Pucks":
        """Create from 12-element numpy array."""
        return cls(
            xA=float(arr[0]), yA=float(arr[1]), vA=float(arr[2]), phiA=float(arr[3]),
            xB=float(arr[4]), yB=float(arr[5]), vB=float(arr[6]), phiB=float(arr[7]),
            xC=float(arr[8]), yC=float(arr[9]), vC=float(arr[10]), phiC=float(arr[11]),
            bounds=bounds or ControlBounds3Pucks(),
            rate_limits=rate_limits or ControlRateLimits3Pucks(),
        )
    
    def clamp_to_bounds(self) -> "ControlVector3Pucks":
        """Return new ControlVector3Pucks clamped to bounds."""
        b = self.bounds
        return ControlVector3Pucks(
            xA=float(np.clip(self.xA, b.x_min, b.x_max)),
            yA=float(np.clip(self.yA, b.y_min, b.y_max)),
            vA=float(np.clip(self.vA, b.v_min, b.v_max)),
            phiA=float(self._wrap_angle(self.phiA)),
            xB=float(np.clip(self.xB, b.x_min, b.x_max)),
            yB=float(np.clip(self.yB, b.y_min, b.y_max)),
            vB=float(np.clip(self.vB, b.v_min, b.v_max)),
            phiB=float(self._wrap_angle(self.phiB)),
            xC=float(np.clip(self.xC, b.x_min, b.x_max)),
            yC=float(np.clip(self.yC, b.y_min, b.y_max_C)),  # C can go higher
            vC=float(np.clip(self.vC, b.v_min, b.v_max)),
            phiC=float(self._wrap_angle(self.phiC)),
            bounds=self.bounds,
            rate_limits=self.rate_limits,
        )
    
    def apply_rate_limits(self, u_prev: "ControlVector3Pucks") -> "ControlVector3Pucks":
        """Return new ControlVector3Pucks with rate limits applied relative to u_prev."""
        r = self.rate_limits
        
        def clamp_delta(new_val: float, old_val: float, max_delta: float) -> float:
            delta = new_val - old_val
            return old_val + float(np.clip(delta, -max_delta, max_delta))
        
        def clamp_angle_delta(new_phi: float, old_phi: float, max_delta: float) -> float:
            delta = self._wrap_angle(new_phi - old_phi)
            return self._wrap_angle(old_phi + float(np.clip(delta, -max_delta, max_delta)))
        
        return ControlVector3Pucks(
            xA=clamp_delta(self.xA, u_prev.xA, r.dx_max),
            yA=clamp_delta(self.yA, u_prev.yA, r.dy_max),
            vA=clamp_delta(self.vA, u_prev.vA, r.dv_max),
            phiA=clamp_angle_delta(self.phiA, u_prev.phiA, r.dphi_max),
            xB=clamp_delta(self.xB, u_prev.xB, r.dx_max),
            yB=clamp_delta(self.yB, u_prev.yB, r.dy_max),
            vB=clamp_delta(self.vB, u_prev.vB, r.dv_max),
            phiB=clamp_angle_delta(self.phiB, u_prev.phiB, r.dphi_max),
            xC=clamp_delta(self.xC, u_prev.xC, r.dx_max),
            yC=clamp_delta(self.yC, u_prev.yC, r.dy_max),
            vC=clamp_delta(self.vC, u_prev.vC, r.dv_max),
            phiC=clamp_angle_delta(self.phiC, u_prev.phiC, r.dphi_max),
            bounds=self.bounds,
            rate_limits=self.rate_limits,
        )
    
    def pairwise_separations(self) -> dict[str, float]:
        """Compute pairwise separations between transducers."""
        dAB = np.sqrt((self.xA - self.xB)**2 + (self.yA - self.yB)**2)
        dAC = np.sqrt((self.xA - self.xC)**2 + (self.yA - self.yC)**2)
        dBC = np.sqrt((self.xB - self.xC)**2 + (self.yB - self.yC)**2)
        return {"AB": float(dAB), "AC": float(dAC), "BC": float(dBC)}
    
    def min_separation(self) -> float:
        """Return minimum pairwise separation."""
        seps = self.pairwise_separations()
        return min(seps.values())
    
    def separation_penalty(self) -> float:
        """
        Penalty for transducers being too close.
        Returns 0 if all pairs are at least min_separation apart.
        """
        min_sep = self.bounds.min_separation
        seps = self.pairwise_separations()
        penalty = 0.0
        for d in seps.values():
            if d < min_sep:
                # Quadratic penalty for being too close
                penalty += (min_sep - d)**2
        return penalty
    
    @staticmethod
    def _wrap_angle(a: float) -> float:
        """Wrap angle to [-π, π]."""
        return float((a + np.pi) % (2.0 * np.pi) - np.pi)
    
    @property
    def dim(self) -> int:
        """Dimension of control vector."""
        return 12


def control_to_forcing_band_vb_3pucks(
    u: Control3Pucks,
    x: np.ndarray,
    sigma_x: float,
    sigma_y: float,
) -> np.ndarray:
    """
    Map 3-puck control to bottom boundary velocity field.
    
    Sum of 3 Gaussian footprints, each with y-dependent coupling.
    
    Parameters
    ----------
    u : Control3Pucks
        Control configuration for 3 transducers.
    x : np.ndarray
        X-coordinates of the boundary grid (shape Nx).
    sigma_x : float
        Width of Gaussian footprint in x.
    sigma_y : float
        Width of Gaussian for y-coupling (how y-position affects boundary coupling).
    
    Returns
    -------
    vb_x : np.ndarray, shape (Nx,), complex
        Bottom boundary velocity profile for the Helmholtz solver.
    """
    # X-direction Gaussian footprints
    gA_x = np.exp(-(x - u.xA)**2 / (2.0 * sigma_x * sigma_x))
    gB_x = np.exp(-(x - u.xB)**2 / (2.0 * sigma_x * sigma_x))
    gC_x = np.exp(-(x - u.xC)**2 / (2.0 * sigma_x * sigma_x))
    
    # Y-direction coupling factor (how much transducer couples to y=0 boundary)
    gA_y = np.exp(-(u.yA)**2 / (2.0 * sigma_y * sigma_y))
    gB_y = np.exp(-(u.yB)**2 / (2.0 * sigma_y * sigma_y))
    gC_y = np.exp(-(u.yC)**2 / (2.0 * sigma_y * sigma_y))
    
    # Sum contributions from all 3 transducers
    vb_x = (
        (u.vA * np.exp(1j * u.phiA) * gA_x * gA_y) +
        (u.vB * np.exp(1j * u.phiB) * gB_x * gB_y) +
        (u.vC * np.exp(1j * u.phiC) * gC_x * gC_y)
    )
    
    return vb_x.astype(np.complex128)


def default_3puck_config(Lx: float = 2e-3, Ly: float = 2e-3) -> Control3Pucks:
    """
    Create a sensible default 3-puck configuration.
    
    Transducers A and B at bottom corners, C at bottom center.
    Phases set for constructive interference in center.
    """
    return Control3Pucks(
        xA=0.25 * Lx, yA=0.02 * Ly, vA=0.05, phiA=0.0,
        xB=0.75 * Lx, yB=0.02 * Ly, vB=0.05, phiB=np.pi,
        xC=0.50 * Lx, yC=0.02 * Ly, vC=0.05, phiC=np.pi / 2,
    )


def default_3puck_spread(Lx: float = 2e-3, Ly: float = 2e-3) -> Control3Pucks:
    """
    3-puck config with C at a higher y for better y-authority.
    
    This configuration provides non-collinear forcing that can break
    the y-control limitation of bottom-only transducers.
    """
    return Control3Pucks(
        xA=0.25 * Lx, yA=0.02 * Ly, vA=0.05, phiA=0.0,
        xB=0.75 * Lx, yB=0.02 * Ly, vB=0.05, phiB=np.pi,
        xC=0.50 * Lx, yC=0.15 * Ly, vC=0.05, phiC=np.pi / 2,  # C higher up
    )
