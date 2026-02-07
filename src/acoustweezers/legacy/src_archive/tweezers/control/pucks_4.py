"""
4-Transducer (4-Puck) Extension for Acoustic Tweezers Control with ON/OFF Gating.

This module extends the 3-transducer system to 4 transducers with per-transducer
gating (ON/OFF) capability. This enables:
- Move-while-off actions (reposition a silent transducer, then enable)
- Toggle actions for dynamic interference pattern control
- Richer control authority via 4th transducer D

Key additions:
- Control4Pucks: 4-transducer control dataclass with enable gates
- control_to_forcing_band_vb_4pucks: Sum of 4 footprints with gating

Usage:
    from tweezers.control.pucks_4 import Control4Pucks, control_to_forcing_band_vb_4pucks
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass(frozen=True)
class Control4Pucks:
    """
    Four transducers with 2D positions, amplitude, phase, and enable gates.
    
    Positions are in meters in the same coordinate system as the solver domain.
    Phases are in radians.
    Gates are boolean: True = transducer active, False = silent.
    
    Transducers A and B are on the bottom (y ≈ 0).
    Transducer C can be at side/top for non-collinear forcing.
    Transducer D provides additional control authority.
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
    
    # Transducer D (additional authority)
    xD: float
    yD: float
    vD: float
    phiD: float
    
    # ON/OFF gates (default all on)
    gateA: bool = True
    gateB: bool = True
    gateC: bool = True
    gateD: bool = True
    
    def to_array(self) -> np.ndarray:
        """Convert to 16-element array (positions, amplitudes, phases only, no gates)."""
        return np.array([
            self.xA, self.yA, self.vA, self.phiA,
            self.xB, self.yB, self.vB, self.phiB,
            self.xC, self.yC, self.vC, self.phiC,
            self.xD, self.yD, self.vD, self.phiD,
        ], dtype=np.float64)
    
    def to_array_with_gates(self) -> np.ndarray:
        """Convert to 20-element array including gate states as 0.0/1.0."""
        return np.array([
            self.xA, self.yA, self.vA, self.phiA, float(self.gateA),
            self.xB, self.yB, self.vB, self.phiB, float(self.gateB),
            self.xC, self.yC, self.vC, self.phiC, float(self.gateC),
            self.xD, self.yD, self.vD, self.phiD, float(self.gateD),
        ], dtype=np.float64)
    
    @classmethod
    def from_array(cls, arr: np.ndarray, gates: tuple[bool, bool, bool, bool] = (True, True, True, True)) -> "Control4Pucks":
        """Create from 16-element array with optional gate specification."""
        return cls(
            xA=float(arr[0]), yA=float(arr[1]), vA=float(arr[2]), phiA=float(arr[3]),
            gateA=gates[0],
            xB=float(arr[4]), yB=float(arr[5]), vB=float(arr[6]), phiB=float(arr[7]),
            gateB=gates[1],
            xC=float(arr[8]), yC=float(arr[9]), vC=float(arr[10]), phiC=float(arr[11]),
            gateC=gates[2],
            xD=float(arr[12]), yD=float(arr[13]), vD=float(arr[14]), phiD=float(arr[15]),
            gateD=gates[3],
        )
    
    @classmethod
    def from_array_with_gates(cls, arr: np.ndarray) -> "Control4Pucks":
        """Create from 20-element array including gate states."""
        return cls(
            xA=float(arr[0]), yA=float(arr[1]), vA=float(arr[2]), phiA=float(arr[3]),
            gateA=bool(arr[4] > 0.5),
            xB=float(arr[5]), yB=float(arr[6]), vB=float(arr[7]), phiB=float(arr[8]),
            gateB=bool(arr[9] > 0.5),
            xC=float(arr[10]), yC=float(arr[11]), vC=float(arr[12]), phiC=float(arr[13]),
            gateC=bool(arr[14] > 0.5),
            xD=float(arr[15]), yD=float(arr[16]), vD=float(arr[17]), phiD=float(arr[18]),
            gateD=bool(arr[19] > 0.5),
        )
    
    def with_gate(self, transducer: str, enabled: bool) -> "Control4Pucks":
        """Return new Control4Pucks with specified transducer gate changed."""
        gates = {"A": self.gateA, "B": self.gateB, "C": self.gateC, "D": self.gateD}
        gates[transducer.upper()] = enabled
        return Control4Pucks(
            xA=self.xA, yA=self.yA, vA=self.vA, phiA=self.phiA, gateA=gates["A"],
            xB=self.xB, yB=self.yB, vB=self.vB, phiB=self.phiB, gateB=gates["B"],
            xC=self.xC, yC=self.yC, vC=self.vC, phiC=self.phiC, gateC=gates["C"],
            xD=self.xD, yD=self.yD, vD=self.vD, phiD=self.phiD, gateD=gates["D"],
        )
    
    def active_count(self) -> int:
        """Return number of active (gated-on) transducers."""
        return sum([self.gateA, self.gateB, self.gateC, self.gateD])
    
    @classmethod
    def from_control3pucks(cls, u3, xD: float = 0.5e-3, yD: float = 0.1e-3,
                           vD: float = 0.05, phiD: float = 0.0,
                           gateD: bool = True) -> "Control4Pucks":
        """
        Create Control4Pucks from Control3Pucks, adding transducer D.
        
        By default, D is positioned between A and C with moderate amplitude.
        """
        return cls(
            xA=u3.xA, yA=u3.yA, vA=u3.vA, phiA=u3.phiA, gateA=True,
            xB=u3.xB, yB=u3.yB, vB=u3.vB, phiB=u3.phiB, gateB=True,
            xC=u3.xC, yC=u3.yC, vC=u3.vC, phiC=u3.phiC, gateC=True,
            xD=xD, yD=yD, vD=vD, phiD=phiD, gateD=gateD,
        )


@dataclass
class ControlBounds4Pucks:
    """Bounds for 4-puck control variables."""
    x_min: float = 0.0
    x_max: float = 2e-3
    y_min: float = 0.0
    y_max: float = 0.25e-3   # Transducers A,B confined to bottom band
    y_max_CD: float = 2e-3   # Transducers C,D can go higher for better coverage
    v_min: float = 0.0
    v_max: float = 2e-3
    phi_min: float = -np.pi
    phi_max: float = np.pi
    
    # Safety: minimum separation between any two transducers
    min_separation: float = 0.1e-3


@dataclass
class ControlRateLimits4Pucks:
    """Per-step rate limits for 4-puck control variables."""
    dx_max: float = 0.1e-3
    dy_max: float = 0.05e-3
    dv_max: float = 2e-4
    dphi_max: float = 0.5


@dataclass
class ControlVector4Pucks:
    """
    Full control parameterization for four-transducer system with gating.
    
    Control vector u = [xA, yA, vA, phiA, xB, yB, vB, phiB, 
                        xC, yC, vC, phiC, xD, yD, vD, phiD]
    
    This is a 16-dimensional continuous control space + 4 binary gates.
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
    
    xD: float
    yD: float
    vD: float
    phiD: float
    
    # Gates (defaults: all on)
    gateA: bool = True
    gateB: bool = True
    gateC: bool = True
    gateD: bool = True
    
    bounds: ControlBounds4Pucks = field(default_factory=ControlBounds4Pucks)
    rate_limits: ControlRateLimits4Pucks = field(default_factory=ControlRateLimits4Pucks)
    
    @classmethod
    def from_control4pucks(
        cls,
        u: Control4Pucks,
        bounds: Optional[ControlBounds4Pucks] = None,
        rate_limits: Optional[ControlRateLimits4Pucks] = None,
    ) -> "ControlVector4Pucks":
        """Create ControlVector4Pucks from Control4Pucks."""
        return cls(
            xA=u.xA, yA=u.yA, vA=u.vA, phiA=u.phiA, gateA=u.gateA,
            xB=u.xB, yB=u.yB, vB=u.vB, phiB=u.phiB, gateB=u.gateB,
            xC=u.xC, yC=u.yC, vC=u.vC, phiC=u.phiC, gateC=u.gateC,
            xD=u.xD, yD=u.yD, vD=u.vD, phiD=u.phiD, gateD=u.gateD,
            bounds=bounds or ControlBounds4Pucks(),
            rate_limits=rate_limits or ControlRateLimits4Pucks(),
        )
    
    def to_control4pucks(self) -> Control4Pucks:
        """Convert to Control4Pucks."""
        return Control4Pucks(
            xA=self.xA, yA=self.yA, vA=self.vA, phiA=self.phiA, gateA=self.gateA,
            xB=self.xB, yB=self.yB, vB=self.vB, phiB=self.phiB, gateB=self.gateB,
            xC=self.xC, yC=self.yC, vC=self.vC, phiC=self.phiC, gateC=self.gateC,
            xD=self.xD, yD=self.yD, vD=self.vD, phiD=self.phiD, gateD=self.gateD,
        )
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array (16 elements, no gates)."""
        return np.array([
            self.xA, self.yA, self.vA, self.phiA,
            self.xB, self.yB, self.vB, self.phiB,
            self.xC, self.yC, self.vC, self.phiC,
            self.xD, self.yD, self.vD, self.phiD,
        ], dtype=np.float64)
    
    @classmethod
    def from_array(
        cls,
        arr: np.ndarray,
        gates: tuple[bool, bool, bool, bool] = (True, True, True, True),
        bounds: Optional[ControlBounds4Pucks] = None,
        rate_limits: Optional[ControlRateLimits4Pucks] = None,
    ) -> "ControlVector4Pucks":
        """Create from 16-element numpy array."""
        return cls(
            xA=float(arr[0]), yA=float(arr[1]), vA=float(arr[2]), phiA=float(arr[3]),
            gateA=gates[0],
            xB=float(arr[4]), yB=float(arr[5]), vB=float(arr[6]), phiB=float(arr[7]),
            gateB=gates[1],
            xC=float(arr[8]), yC=float(arr[9]), vC=float(arr[10]), phiC=float(arr[11]),
            gateC=gates[2],
            xD=float(arr[12]), yD=float(arr[13]), vD=float(arr[14]), phiD=float(arr[15]),
            gateD=gates[3],
            bounds=bounds or ControlBounds4Pucks(),
            rate_limits=rate_limits or ControlRateLimits4Pucks(),
        )
    
    def clamp_to_bounds(self) -> "ControlVector4Pucks":
        """Return new ControlVector4Pucks clamped to bounds."""
        b = self.bounds
        return ControlVector4Pucks(
            xA=float(np.clip(self.xA, b.x_min, b.x_max)),
            yA=float(np.clip(self.yA, b.y_min, b.y_max)),
            vA=float(np.clip(self.vA, b.v_min, b.v_max)),
            phiA=float(self._wrap_angle(self.phiA)),
            gateA=self.gateA,
            xB=float(np.clip(self.xB, b.x_min, b.x_max)),
            yB=float(np.clip(self.yB, b.y_min, b.y_max)),
            vB=float(np.clip(self.vB, b.v_min, b.v_max)),
            phiB=float(self._wrap_angle(self.phiB)),
            gateB=self.gateB,
            xC=float(np.clip(self.xC, b.x_min, b.x_max)),
            yC=float(np.clip(self.yC, b.y_min, b.y_max_CD)),  # C can go higher
            vC=float(np.clip(self.vC, b.v_min, b.v_max)),
            phiC=float(self._wrap_angle(self.phiC)),
            gateC=self.gateC,
            xD=float(np.clip(self.xD, b.x_min, b.x_max)),
            yD=float(np.clip(self.yD, b.y_min, b.y_max_CD)),  # D can go higher
            vD=float(np.clip(self.vD, b.v_min, b.v_max)),
            phiD=float(self._wrap_angle(self.phiD)),
            gateD=self.gateD,
            bounds=self.bounds,
            rate_limits=self.rate_limits,
        )
    
    def pairwise_separations(self) -> dict[str, float]:
        """Compute pairwise separations between transducers."""
        positions = {
            "A": (self.xA, self.yA),
            "B": (self.xB, self.yB),
            "C": (self.xC, self.yC),
            "D": (self.xD, self.yD),
        }
        result = {}
        for i, (n1, p1) in enumerate(positions.items()):
            for n2, p2 in list(positions.items())[i+1:]:
                d = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                result[f"{n1}{n2}"] = float(d)
        return result
    
    def min_separation(self) -> float:
        """Return minimum pairwise separation."""
        seps = self.pairwise_separations()
        return min(seps.values())
    
    def separation_penalty(self) -> float:
        """
        Penalty for transducers being too close.
        Returns 0 if all pairs are at least min_separation apart.
        Only penalizes active transducers.
        """
        min_sep = self.bounds.min_separation
        gates = {"A": self.gateA, "B": self.gateB, "C": self.gateC, "D": self.gateD}
        positions = {
            "A": (self.xA, self.yA),
            "B": (self.xB, self.yB),
            "C": (self.xC, self.yC),
            "D": (self.xD, self.yD),
        }
        
        penalty = 0.0
        names = list(positions.keys())
        for i, n1 in enumerate(names):
            for n2 in names[i+1:]:
                # Only penalize if both are active
                if gates[n1] and gates[n2]:
                    d = np.sqrt((positions[n1][0] - positions[n2][0])**2 + 
                               (positions[n1][1] - positions[n2][1])**2)
                    if d < min_sep:
                        penalty += (min_sep - d)**2
        return penalty
    
    @staticmethod
    def _wrap_angle(a: float) -> float:
        """Wrap angle to [-π, π]."""
        return float((a + np.pi) % (2.0 * np.pi) - np.pi)
    
    @property
    def dim(self) -> int:
        """Dimension of continuous control vector (excludes gates)."""
        return 16


def control_to_forcing_band_vb_4pucks(
    u: Control4Pucks,
    x: np.ndarray,
    sigma_x: float,
    sigma_y: float,
) -> np.ndarray:
    """
    Map 4-puck control to bottom boundary velocity field with gating.
    
    Sum of 4 Gaussian footprints, each with y-dependent coupling.
    Gated-off transducers contribute zero amplitude.
    
    Parameters
    ----------
    u : Control4Pucks
        Control configuration for 4 transducers.
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
    vb_x = np.zeros(len(x), dtype=np.complex128)
    
    # Transducer A
    if u.gateA:
        gA_x = np.exp(-(x - u.xA)**2 / (2.0 * sigma_x * sigma_x))
        gA_y = np.exp(-(u.yA)**2 / (2.0 * sigma_y * sigma_y))
        vb_x += u.vA * np.exp(1j * u.phiA) * gA_x * gA_y
    
    # Transducer B
    if u.gateB:
        gB_x = np.exp(-(x - u.xB)**2 / (2.0 * sigma_x * sigma_x))
        gB_y = np.exp(-(u.yB)**2 / (2.0 * sigma_y * sigma_y))
        vb_x += u.vB * np.exp(1j * u.phiB) * gB_x * gB_y
    
    # Transducer C
    if u.gateC:
        gC_x = np.exp(-(x - u.xC)**2 / (2.0 * sigma_x * sigma_x))
        gC_y = np.exp(-(u.yC)**2 / (2.0 * sigma_y * sigma_y))
        vb_x += u.vC * np.exp(1j * u.phiC) * gC_x * gC_y
    
    # Transducer D
    if u.gateD:
        gD_x = np.exp(-(x - u.xD)**2 / (2.0 * sigma_x * sigma_x))
        gD_y = np.exp(-(u.yD)**2 / (2.0 * sigma_y * sigma_y))
        vb_x += u.vD * np.exp(1j * u.phiD) * gD_x * gD_y
    
    return vb_x


def default_4puck_config(Lx: float = 2e-3, Ly: float = 2e-3) -> Control4Pucks:
    """
    Create a sensible default 4-puck configuration.
    
    Transducers A and B at bottom corners, C at bottom center, D at top center.
    Phases set for constructive interference in center.
    """
    return Control4Pucks(
        xA=0.25 * Lx, yA=0.02 * Ly, vA=0.05, phiA=0.0, gateA=True,
        xB=0.75 * Lx, yB=0.02 * Ly, vB=0.05, phiB=np.pi, gateB=True,
        xC=0.50 * Lx, yC=0.02 * Ly, vC=0.05, phiC=np.pi / 2, gateC=True,
        xD=0.50 * Lx, yD=0.15 * Ly, vD=0.05, phiD=0.0, gateD=True,
    )


def default_4puck_spread(Lx: float = 2e-3, Ly: float = 2e-3) -> Control4Pucks:
    """
    4-puck config with C and D at higher y for better y-authority.
    
    This configuration provides non-collinear forcing that can break
    the y-control limitation of bottom-only transducers.
    """
    return Control4Pucks(
        xA=0.25 * Lx, yA=0.02 * Ly, vA=0.05, phiA=0.0, gateA=True,
        xB=0.75 * Lx, yB=0.02 * Ly, vB=0.05, phiB=np.pi, gateB=True,
        xC=0.35 * Lx, yC=0.15 * Ly, vC=0.05, phiC=np.pi / 4, gateC=True,
        xD=0.65 * Lx, yD=0.15 * Ly, vD=0.05, phiD=-np.pi / 4, gateD=True,
    )


def default_4puck_with_d_off(Lx: float = 2e-3, Ly: float = 2e-3) -> Control4Pucks:
    """
    4-puck config where D starts gated-off.
    
    Useful for testing move-while-off actions: D can be repositioned
    while silent, then gated-on for a new interference pattern.
    """
    return Control4Pucks(
        xA=0.25 * Lx, yA=0.02 * Ly, vA=0.05, phiA=0.0, gateA=True,
        xB=0.75 * Lx, yB=0.02 * Ly, vB=0.05, phiB=np.pi, gateB=True,
        xC=0.50 * Lx, yC=0.02 * Ly, vC=0.05, phiC=np.pi / 2, gateC=True,
        xD=0.50 * Lx, yD=0.15 * Ly, vD=0.05, phiD=0.0, gateD=False,  # D starts off
    )
