#!/usr/bin/env python3
"""
Extended Macro Actions for 4-Puck Acoustic Tweezers Control with Gating.

STAGE 2+ IMPLEMENTATION:
Provides structured control primitives including:
- All 3-puck actions extended to 4 pucks
- Toggle ON/OFF actions per transducer
- Move-while-off actions (reposition silent transducer)
- Transducer D control actions

Macro action categories:
1. TRANSLATE_TRAP: Move all transducers together
2. ROTATE_INTERFERENCE: Rotate interference pattern
3. STRENGTHEN/WEAKEN_TRAP: Change amplitudes
4. WIDEN/NARROW: Change transducer spread
5. Individual puck moves: MOVE_A/B/C/D in X/Y
6. Phase shifts: PHASE_SHIFT_B/C/D
7. Toggle gates: TOGGLE_A/B/C/D_ON/OFF
8. Move-while-off: MOVE_A/B/C/D_*_OFF (move without emitting)

Usage:
    from scripts.macro_actions_4puck import MacroActionType4Puck, apply_macro_action_4puck
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, List
from pathlib import Path
import numpy as np

from tweezers.control import (
    Control4Pucks,
)


class MacroActionType4Puck(Enum):
    """Macro action types for 4-puck control with gating."""
    HOLD = auto()
    
    # ==========================================
    # Primary trap translation (all 4 together)
    # ==========================================
    TRANSLATE_TRAP_X_POS = auto()
    TRANSLATE_TRAP_X_NEG = auto()
    TRANSLATE_TRAP_Y_POS = auto()
    TRANSLATE_TRAP_Y_NEG = auto()
    
    # ==========================================
    # Interference pattern control
    # ==========================================
    ROTATE_INTERFERENCE_CW = auto()
    ROTATE_INTERFERENCE_CCW = auto()
    
    # ==========================================
    # Stiffness control
    # ==========================================
    STRENGTHEN_TRAP = auto()
    WEAKEN_TRAP = auto()
    
    # ==========================================
    # Spread control
    # ==========================================
    WIDEN = auto()
    NARROW = auto()
    
    # ==========================================
    # Individual puck position control
    # ==========================================
    MOVE_A_RIGHT = auto()
    MOVE_A_LEFT = auto()
    MOVE_B_RIGHT = auto()
    MOVE_B_LEFT = auto()
    MOVE_C_UP = auto()
    MOVE_C_DOWN = auto()
    MOVE_C_RIGHT = auto()
    MOVE_C_LEFT = auto()
    MOVE_D_UP = auto()
    MOVE_D_DOWN = auto()
    MOVE_D_RIGHT = auto()
    MOVE_D_LEFT = auto()
    
    # ==========================================
    # Phase control
    # ==========================================
    PHASE_SHIFT_B_POS = auto()
    PHASE_SHIFT_B_NEG = auto()
    PHASE_SHIFT_C_POS = auto()
    PHASE_SHIFT_C_NEG = auto()
    PHASE_SHIFT_D_POS = auto()
    PHASE_SHIFT_D_NEG = auto()
    
    # ==========================================
    # Toggle gates ON/OFF
    # ==========================================
    TOGGLE_A_ON = auto()
    TOGGLE_A_OFF = auto()
    TOGGLE_B_ON = auto()
    TOGGLE_B_OFF = auto()
    TOGGLE_C_ON = auto()
    TOGGLE_C_OFF = auto()
    TOGGLE_D_ON = auto()
    TOGGLE_D_OFF = auto()
    
    # ==========================================
    # Move-while-off actions
    # Reposition a gated-off transducer (no acoustic output change)
    # ==========================================
    MOVE_A_RIGHT_OFF = auto()
    MOVE_A_LEFT_OFF = auto()
    MOVE_A_UP_OFF = auto()
    MOVE_A_DOWN_OFF = auto()
    
    MOVE_B_RIGHT_OFF = auto()
    MOVE_B_LEFT_OFF = auto()
    MOVE_B_UP_OFF = auto()
    MOVE_B_DOWN_OFF = auto()
    
    MOVE_C_RIGHT_OFF = auto()
    MOVE_C_LEFT_OFF = auto()
    MOVE_C_UP_OFF = auto()
    MOVE_C_DOWN_OFF = auto()
    
    MOVE_D_RIGHT_OFF = auto()
    MOVE_D_LEFT_OFF = auto()
    MOVE_D_UP_OFF = auto()
    MOVE_D_DOWN_OFF = auto()


@dataclass
class MacroAction4Puck:
    """A structured control primitive for 4-puck system with gating."""
    action_type: MacroActionType4Puck
    magnitude: float = 0.05e-3   # Position step (m)
    phase_step: float = 0.15     # Phase step (rad)
    amplitude_step: float = 0.01  # Amplitude step


@dataclass
class MacroActionEffect4Puck:
    """Measured effect of a macro action on trap position."""
    action: MacroAction4Puck
    delta_trap_x: float
    delta_trap_y: float
    delta_stiffness: float
    trap_found: bool
    initial_trap_x: float
    initial_trap_y: float
    final_trap_x: float
    final_trap_y: float
    initial_stiffness: float
    final_stiffness: float
    delta_force_x: float = 0.0
    delta_force_y: float = 0.0


def apply_macro_action_4puck(u: Control4Pucks, action: MacroAction4Puck) -> Control4Pucks:
    """Apply a macro action to a 4-puck control configuration."""
    mag = action.magnitude
    phase_step = action.phase_step
    amp_step = action.amplitude_step
    
    match action.action_type:
        case MacroActionType4Puck.HOLD:
            return u
        
        # ==========================================
        # TRANSLATE_TRAP: Move all transducers together
        # ==========================================
        case MacroActionType4Puck.TRANSLATE_TRAP_X_POS:
            return Control4Pucks(
                xA=u.xA + mag, yA=u.yA, vA=u.vA, phiA=u.phiA, gateA=u.gateA,
                xB=u.xB + mag, yB=u.yB, vB=u.vB, phiB=u.phiB, gateB=u.gateB,
                xC=u.xC + mag, yC=u.yC, vC=u.vC, phiC=u.phiC, gateC=u.gateC,
                xD=u.xD + mag, yD=u.yD, vD=u.vD, phiD=u.phiD, gateD=u.gateD,
            )
        
        case MacroActionType4Puck.TRANSLATE_TRAP_X_NEG:
            return Control4Pucks(
                xA=u.xA - mag, yA=u.yA, vA=u.vA, phiA=u.phiA, gateA=u.gateA,
                xB=u.xB - mag, yB=u.yB, vB=u.vB, phiB=u.phiB, gateB=u.gateB,
                xC=u.xC - mag, yC=u.yC, vC=u.vC, phiC=u.phiC, gateC=u.gateC,
                xD=u.xD - mag, yD=u.yD, vD=u.vD, phiD=u.phiD, gateD=u.gateD,
            )
        
        case MacroActionType4Puck.TRANSLATE_TRAP_Y_POS:
            # Y-control via phase shift + moving elevated transducers
            return Control4Pucks(
                xA=u.xA, yA=u.yA, vA=u.vA, phiA=u.phiA, gateA=u.gateA,
                xB=u.xB, yB=u.yB, vB=u.vB, phiB=u.phiB + phase_step * 0.5, gateB=u.gateB,
                xC=u.xC, yC=u.yC + mag * 0.5, vC=u.vC, phiC=u.phiC - phase_step * 0.25, gateC=u.gateC,
                xD=u.xD, yD=u.yD + mag * 0.5, vD=u.vD, phiD=u.phiD - phase_step * 0.25, gateD=u.gateD,
            )
        
        case MacroActionType4Puck.TRANSLATE_TRAP_Y_NEG:
            return Control4Pucks(
                xA=u.xA, yA=u.yA, vA=u.vA, phiA=u.phiA, gateA=u.gateA,
                xB=u.xB, yB=u.yB, vB=u.vB, phiB=u.phiB - phase_step * 0.5, gateB=u.gateB,
                xC=u.xC, yC=max(0.01e-3, u.yC - mag * 0.5), vC=u.vC, phiC=u.phiC + phase_step * 0.25, gateC=u.gateC,
                xD=u.xD, yD=max(0.01e-3, u.yD - mag * 0.5), vD=u.vD, phiD=u.phiD + phase_step * 0.25, gateD=u.gateD,
            )
        
        # ==========================================
        # ROTATE_INTERFERENCE: Rotate interference pattern
        # ==========================================
        case MacroActionType4Puck.ROTATE_INTERFERENCE_CW:
            return Control4Pucks(
                xA=u.xA, yA=u.yA, vA=u.vA, phiA=u.phiA + phase_step, gateA=u.gateA,
                xB=u.xB, yB=u.yB, vB=u.vB, phiB=u.phiB - phase_step, gateB=u.gateB,
                xC=u.xC, yC=u.yC, vC=u.vC, phiC=u.phiC, gateC=u.gateC,
                xD=u.xD, yD=u.yD, vD=u.vD, phiD=u.phiD, gateD=u.gateD,
            )
        
        case MacroActionType4Puck.ROTATE_INTERFERENCE_CCW:
            return Control4Pucks(
                xA=u.xA, yA=u.yA, vA=u.vA, phiA=u.phiA - phase_step, gateA=u.gateA,
                xB=u.xB, yB=u.yB, vB=u.vB, phiB=u.phiB + phase_step, gateB=u.gateB,
                xC=u.xC, yC=u.yC, vC=u.vC, phiC=u.phiC, gateC=u.gateC,
                xD=u.xD, yD=u.yD, vD=u.vD, phiD=u.phiD, gateD=u.gateD,
            )
        
        # ==========================================
        # STRENGTHEN/WEAKEN: Change amplitudes
        # ==========================================
        case MacroActionType4Puck.STRENGTHEN_TRAP:
            return Control4Pucks(
                xA=u.xA, yA=u.yA, vA=u.vA + amp_step, phiA=u.phiA, gateA=u.gateA,
                xB=u.xB, yB=u.yB, vB=u.vB + amp_step, phiB=u.phiB, gateB=u.gateB,
                xC=u.xC, yC=u.yC, vC=u.vC + amp_step, phiC=u.phiC, gateC=u.gateC,
                xD=u.xD, yD=u.yD, vD=u.vD + amp_step, phiD=u.phiD, gateD=u.gateD,
            )
        
        case MacroActionType4Puck.WEAKEN_TRAP:
            return Control4Pucks(
                xA=u.xA, yA=u.yA, vA=max(0.01, u.vA - amp_step), phiA=u.phiA, gateA=u.gateA,
                xB=u.xB, yB=u.yB, vB=max(0.01, u.vB - amp_step), phiB=u.phiB, gateB=u.gateB,
                xC=u.xC, yC=u.yC, vC=max(0.01, u.vC - amp_step), phiC=u.phiC, gateC=u.gateC,
                xD=u.xD, yD=u.yD, vD=max(0.01, u.vD - amp_step), phiD=u.phiD, gateD=u.gateD,
            )
        
        # ==========================================
        # WIDEN/NARROW: Change transducer spread
        # ==========================================
        case MacroActionType4Puck.WIDEN:
            # A and B move apart, C and D move apart vertically
            return Control4Pucks(
                xA=u.xA - mag * 0.5, yA=u.yA, vA=u.vA, phiA=u.phiA, gateA=u.gateA,
                xB=u.xB + mag * 0.5, yB=u.yB, vB=u.vB, phiB=u.phiB, gateB=u.gateB,
                xC=u.xC - mag * 0.25, yC=u.yC, vC=u.vC, phiC=u.phiC, gateC=u.gateC,
                xD=u.xD + mag * 0.25, yD=u.yD, vD=u.vD, phiD=u.phiD, gateD=u.gateD,
            )
        
        case MacroActionType4Puck.NARROW:
            return Control4Pucks(
                xA=u.xA + mag * 0.5, yA=u.yA, vA=u.vA, phiA=u.phiA, gateA=u.gateA,
                xB=u.xB - mag * 0.5, yB=u.yB, vB=u.vB, phiB=u.phiB, gateB=u.gateB,
                xC=u.xC + mag * 0.25, yC=u.yC, vC=u.vC, phiC=u.phiC, gateC=u.gateC,
                xD=u.xD - mag * 0.25, yD=u.yD, vD=u.vD, phiD=u.phiD, gateD=u.gateD,
            )
        
        # ==========================================
        # Individual puck moves: A
        # ==========================================
        case MacroActionType4Puck.MOVE_A_RIGHT:
            return Control4Pucks(
                xA=u.xA + mag, yA=u.yA, vA=u.vA, phiA=u.phiA, gateA=u.gateA,
                xB=u.xB, yB=u.yB, vB=u.vB, phiB=u.phiB, gateB=u.gateB,
                xC=u.xC, yC=u.yC, vC=u.vC, phiC=u.phiC, gateC=u.gateC,
                xD=u.xD, yD=u.yD, vD=u.vD, phiD=u.phiD, gateD=u.gateD,
            )
        
        case MacroActionType4Puck.MOVE_A_LEFT:
            return Control4Pucks(
                xA=u.xA - mag, yA=u.yA, vA=u.vA, phiA=u.phiA, gateA=u.gateA,
                xB=u.xB, yB=u.yB, vB=u.vB, phiB=u.phiB, gateB=u.gateB,
                xC=u.xC, yC=u.yC, vC=u.vC, phiC=u.phiC, gateC=u.gateC,
                xD=u.xD, yD=u.yD, vD=u.vD, phiD=u.phiD, gateD=u.gateD,
            )
        
        # ==========================================
        # Individual puck moves: B
        # ==========================================
        case MacroActionType4Puck.MOVE_B_RIGHT:
            return Control4Pucks(
                xA=u.xA, yA=u.yA, vA=u.vA, phiA=u.phiA, gateA=u.gateA,
                xB=u.xB + mag, yB=u.yB, vB=u.vB, phiB=u.phiB, gateB=u.gateB,
                xC=u.xC, yC=u.yC, vC=u.vC, phiC=u.phiC, gateC=u.gateC,
                xD=u.xD, yD=u.yD, vD=u.vD, phiD=u.phiD, gateD=u.gateD,
            )
        
        case MacroActionType4Puck.MOVE_B_LEFT:
            return Control4Pucks(
                xA=u.xA, yA=u.yA, vA=u.vA, phiA=u.phiA, gateA=u.gateA,
                xB=u.xB - mag, yB=u.yB, vB=u.vB, phiB=u.phiB, gateB=u.gateB,
                xC=u.xC, yC=u.yC, vC=u.vC, phiC=u.phiC, gateC=u.gateC,
                xD=u.xD, yD=u.yD, vD=u.vD, phiD=u.phiD, gateD=u.gateD,
            )
        
        # ==========================================
        # Individual puck moves: C
        # ==========================================
        case MacroActionType4Puck.MOVE_C_UP:
            return Control4Pucks(
                xA=u.xA, yA=u.yA, vA=u.vA, phiA=u.phiA, gateA=u.gateA,
                xB=u.xB, yB=u.yB, vB=u.vB, phiB=u.phiB, gateB=u.gateB,
                xC=u.xC, yC=u.yC + mag, vC=u.vC, phiC=u.phiC, gateC=u.gateC,
                xD=u.xD, yD=u.yD, vD=u.vD, phiD=u.phiD, gateD=u.gateD,
            )
        
        case MacroActionType4Puck.MOVE_C_DOWN:
            return Control4Pucks(
                xA=u.xA, yA=u.yA, vA=u.vA, phiA=u.phiA, gateA=u.gateA,
                xB=u.xB, yB=u.yB, vB=u.vB, phiB=u.phiB, gateB=u.gateB,
                xC=u.xC, yC=max(0.01e-3, u.yC - mag), vC=u.vC, phiC=u.phiC, gateC=u.gateC,
                xD=u.xD, yD=u.yD, vD=u.vD, phiD=u.phiD, gateD=u.gateD,
            )
        
        case MacroActionType4Puck.MOVE_C_RIGHT:
            return Control4Pucks(
                xA=u.xA, yA=u.yA, vA=u.vA, phiA=u.phiA, gateA=u.gateA,
                xB=u.xB, yB=u.yB, vB=u.vB, phiB=u.phiB, gateB=u.gateB,
                xC=u.xC + mag, yC=u.yC, vC=u.vC, phiC=u.phiC, gateC=u.gateC,
                xD=u.xD, yD=u.yD, vD=u.vD, phiD=u.phiD, gateD=u.gateD,
            )
        
        case MacroActionType4Puck.MOVE_C_LEFT:
            return Control4Pucks(
                xA=u.xA, yA=u.yA, vA=u.vA, phiA=u.phiA, gateA=u.gateA,
                xB=u.xB, yB=u.yB, vB=u.vB, phiB=u.phiB, gateB=u.gateB,
                xC=u.xC - mag, yC=u.yC, vC=u.vC, phiC=u.phiC, gateC=u.gateC,
                xD=u.xD, yD=u.yD, vD=u.vD, phiD=u.phiD, gateD=u.gateD,
            )
        
        # ==========================================
        # Individual puck moves: D
        # ==========================================
        case MacroActionType4Puck.MOVE_D_UP:
            return Control4Pucks(
                xA=u.xA, yA=u.yA, vA=u.vA, phiA=u.phiA, gateA=u.gateA,
                xB=u.xB, yB=u.yB, vB=u.vB, phiB=u.phiB, gateB=u.gateB,
                xC=u.xC, yC=u.yC, vC=u.vC, phiC=u.phiC, gateC=u.gateC,
                xD=u.xD, yD=u.yD + mag, vD=u.vD, phiD=u.phiD, gateD=u.gateD,
            )
        
        case MacroActionType4Puck.MOVE_D_DOWN:
            return Control4Pucks(
                xA=u.xA, yA=u.yA, vA=u.vA, phiA=u.phiA, gateA=u.gateA,
                xB=u.xB, yB=u.yB, vB=u.vB, phiB=u.phiB, gateB=u.gateB,
                xC=u.xC, yC=u.yC, vC=u.vC, phiC=u.phiC, gateC=u.gateC,
                xD=u.xD, yD=max(0.01e-3, u.yD - mag), vD=u.vD, phiD=u.phiD, gateD=u.gateD,
            )
        
        case MacroActionType4Puck.MOVE_D_RIGHT:
            return Control4Pucks(
                xA=u.xA, yA=u.yA, vA=u.vA, phiA=u.phiA, gateA=u.gateA,
                xB=u.xB, yB=u.yB, vB=u.vB, phiB=u.phiB, gateB=u.gateB,
                xC=u.xC, yC=u.yC, vC=u.vC, phiC=u.phiC, gateC=u.gateC,
                xD=u.xD + mag, yD=u.yD, vD=u.vD, phiD=u.phiD, gateD=u.gateD,
            )
        
        case MacroActionType4Puck.MOVE_D_LEFT:
            return Control4Pucks(
                xA=u.xA, yA=u.yA, vA=u.vA, phiA=u.phiA, gateA=u.gateA,
                xB=u.xB, yB=u.yB, vB=u.vB, phiB=u.phiB, gateB=u.gateB,
                xC=u.xC, yC=u.yC, vC=u.vC, phiC=u.phiC, gateC=u.gateC,
                xD=u.xD - mag, yD=u.yD, vD=u.vD, phiD=u.phiD, gateD=u.gateD,
            )
        
        # ==========================================
        # Phase shifts: B
        # ==========================================
        case MacroActionType4Puck.PHASE_SHIFT_B_POS:
            return Control4Pucks(
                xA=u.xA, yA=u.yA, vA=u.vA, phiA=u.phiA, gateA=u.gateA,
                xB=u.xB, yB=u.yB, vB=u.vB, phiB=u.phiB + phase_step, gateB=u.gateB,
                xC=u.xC, yC=u.yC, vC=u.vC, phiC=u.phiC, gateC=u.gateC,
                xD=u.xD, yD=u.yD, vD=u.vD, phiD=u.phiD, gateD=u.gateD,
            )
        
        case MacroActionType4Puck.PHASE_SHIFT_B_NEG:
            return Control4Pucks(
                xA=u.xA, yA=u.yA, vA=u.vA, phiA=u.phiA, gateA=u.gateA,
                xB=u.xB, yB=u.yB, vB=u.vB, phiB=u.phiB - phase_step, gateB=u.gateB,
                xC=u.xC, yC=u.yC, vC=u.vC, phiC=u.phiC, gateC=u.gateC,
                xD=u.xD, yD=u.yD, vD=u.vD, phiD=u.phiD, gateD=u.gateD,
            )
        
        # ==========================================
        # Phase shifts: C
        # ==========================================
        case MacroActionType4Puck.PHASE_SHIFT_C_POS:
            return Control4Pucks(
                xA=u.xA, yA=u.yA, vA=u.vA, phiA=u.phiA, gateA=u.gateA,
                xB=u.xB, yB=u.yB, vB=u.vB, phiB=u.phiB, gateB=u.gateB,
                xC=u.xC, yC=u.yC, vC=u.vC, phiC=u.phiC + phase_step, gateC=u.gateC,
                xD=u.xD, yD=u.yD, vD=u.vD, phiD=u.phiD, gateD=u.gateD,
            )
        
        case MacroActionType4Puck.PHASE_SHIFT_C_NEG:
            return Control4Pucks(
                xA=u.xA, yA=u.yA, vA=u.vA, phiA=u.phiA, gateA=u.gateA,
                xB=u.xB, yB=u.yB, vB=u.vB, phiB=u.phiB, gateB=u.gateB,
                xC=u.xC, yC=u.yC, vC=u.vC, phiC=u.phiC - phase_step, gateC=u.gateC,
                xD=u.xD, yD=u.yD, vD=u.vD, phiD=u.phiD, gateD=u.gateD,
            )
        
        # ==========================================
        # Phase shifts: D
        # ==========================================
        case MacroActionType4Puck.PHASE_SHIFT_D_POS:
            return Control4Pucks(
                xA=u.xA, yA=u.yA, vA=u.vA, phiA=u.phiA, gateA=u.gateA,
                xB=u.xB, yB=u.yB, vB=u.vB, phiB=u.phiB, gateB=u.gateB,
                xC=u.xC, yC=u.yC, vC=u.vC, phiC=u.phiC, gateC=u.gateC,
                xD=u.xD, yD=u.yD, vD=u.vD, phiD=u.phiD + phase_step, gateD=u.gateD,
            )
        
        case MacroActionType4Puck.PHASE_SHIFT_D_NEG:
            return Control4Pucks(
                xA=u.xA, yA=u.yA, vA=u.vA, phiA=u.phiA, gateA=u.gateA,
                xB=u.xB, yB=u.yB, vB=u.vB, phiB=u.phiB, gateB=u.gateB,
                xC=u.xC, yC=u.yC, vC=u.vC, phiC=u.phiC, gateC=u.gateC,
                xD=u.xD, yD=u.yD, vD=u.vD, phiD=u.phiD - phase_step, gateD=u.gateD,
            )
        
        # ==========================================
        # Toggle gates ON/OFF
        # ==========================================
        case MacroActionType4Puck.TOGGLE_A_ON:
            return u.with_gate("A", True)
        
        case MacroActionType4Puck.TOGGLE_A_OFF:
            return u.with_gate("A", False)
        
        case MacroActionType4Puck.TOGGLE_B_ON:
            return u.with_gate("B", True)
        
        case MacroActionType4Puck.TOGGLE_B_OFF:
            return u.with_gate("B", False)
        
        case MacroActionType4Puck.TOGGLE_C_ON:
            return u.with_gate("C", True)
        
        case MacroActionType4Puck.TOGGLE_C_OFF:
            return u.with_gate("C", False)
        
        case MacroActionType4Puck.TOGGLE_D_ON:
            return u.with_gate("D", True)
        
        case MacroActionType4Puck.TOGGLE_D_OFF:
            return u.with_gate("D", False)
        
        # ==========================================
        # Move-while-off: A
        # These first gate-off, then move, in one action.
        # Useful for repositioning without affecting acoustic field.
        # ==========================================
        case MacroActionType4Puck.MOVE_A_RIGHT_OFF:
            return Control4Pucks(
                xA=u.xA + mag, yA=u.yA, vA=u.vA, phiA=u.phiA, gateA=False,
                xB=u.xB, yB=u.yB, vB=u.vB, phiB=u.phiB, gateB=u.gateB,
                xC=u.xC, yC=u.yC, vC=u.vC, phiC=u.phiC, gateC=u.gateC,
                xD=u.xD, yD=u.yD, vD=u.vD, phiD=u.phiD, gateD=u.gateD,
            )
        
        case MacroActionType4Puck.MOVE_A_LEFT_OFF:
            return Control4Pucks(
                xA=u.xA - mag, yA=u.yA, vA=u.vA, phiA=u.phiA, gateA=False,
                xB=u.xB, yB=u.yB, vB=u.vB, phiB=u.phiB, gateB=u.gateB,
                xC=u.xC, yC=u.yC, vC=u.vC, phiC=u.phiC, gateC=u.gateC,
                xD=u.xD, yD=u.yD, vD=u.vD, phiD=u.phiD, gateD=u.gateD,
            )
        
        case MacroActionType4Puck.MOVE_A_UP_OFF:
            return Control4Pucks(
                xA=u.xA, yA=u.yA + mag, vA=u.vA, phiA=u.phiA, gateA=False,
                xB=u.xB, yB=u.yB, vB=u.vB, phiB=u.phiB, gateB=u.gateB,
                xC=u.xC, yC=u.yC, vC=u.vC, phiC=u.phiC, gateC=u.gateC,
                xD=u.xD, yD=u.yD, vD=u.vD, phiD=u.phiD, gateD=u.gateD,
            )
        
        case MacroActionType4Puck.MOVE_A_DOWN_OFF:
            return Control4Pucks(
                xA=u.xA, yA=max(0.01e-3, u.yA - mag), vA=u.vA, phiA=u.phiA, gateA=False,
                xB=u.xB, yB=u.yB, vB=u.vB, phiB=u.phiB, gateB=u.gateB,
                xC=u.xC, yC=u.yC, vC=u.vC, phiC=u.phiC, gateC=u.gateC,
                xD=u.xD, yD=u.yD, vD=u.vD, phiD=u.phiD, gateD=u.gateD,
            )
        
        # ==========================================
        # Move-while-off: B
        # ==========================================
        case MacroActionType4Puck.MOVE_B_RIGHT_OFF:
            return Control4Pucks(
                xA=u.xA, yA=u.yA, vA=u.vA, phiA=u.phiA, gateA=u.gateA,
                xB=u.xB + mag, yB=u.yB, vB=u.vB, phiB=u.phiB, gateB=False,
                xC=u.xC, yC=u.yC, vC=u.vC, phiC=u.phiC, gateC=u.gateC,
                xD=u.xD, yD=u.yD, vD=u.vD, phiD=u.phiD, gateD=u.gateD,
            )
        
        case MacroActionType4Puck.MOVE_B_LEFT_OFF:
            return Control4Pucks(
                xA=u.xA, yA=u.yA, vA=u.vA, phiA=u.phiA, gateA=u.gateA,
                xB=u.xB - mag, yB=u.yB, vB=u.vB, phiB=u.phiB, gateB=False,
                xC=u.xC, yC=u.yC, vC=u.vC, phiC=u.phiC, gateC=u.gateC,
                xD=u.xD, yD=u.yD, vD=u.vD, phiD=u.phiD, gateD=u.gateD,
            )
        
        case MacroActionType4Puck.MOVE_B_UP_OFF:
            return Control4Pucks(
                xA=u.xA, yA=u.yA, vA=u.vA, phiA=u.phiA, gateA=u.gateA,
                xB=u.xB, yB=u.yB + mag, vB=u.vB, phiB=u.phiB, gateB=False,
                xC=u.xC, yC=u.yC, vC=u.vC, phiC=u.phiC, gateC=u.gateC,
                xD=u.xD, yD=u.yD, vD=u.vD, phiD=u.phiD, gateD=u.gateD,
            )
        
        case MacroActionType4Puck.MOVE_B_DOWN_OFF:
            return Control4Pucks(
                xA=u.xA, yA=u.yA, vA=u.vA, phiA=u.phiA, gateA=u.gateA,
                xB=u.xB, yB=max(0.01e-3, u.yB - mag), vB=u.vB, phiB=u.phiB, gateB=False,
                xC=u.xC, yC=u.yC, vC=u.vC, phiC=u.phiC, gateC=u.gateC,
                xD=u.xD, yD=u.yD, vD=u.vD, phiD=u.phiD, gateD=u.gateD,
            )
        
        # ==========================================
        # Move-while-off: C
        # ==========================================
        case MacroActionType4Puck.MOVE_C_RIGHT_OFF:
            return Control4Pucks(
                xA=u.xA, yA=u.yA, vA=u.vA, phiA=u.phiA, gateA=u.gateA,
                xB=u.xB, yB=u.yB, vB=u.vB, phiB=u.phiB, gateB=u.gateB,
                xC=u.xC + mag, yC=u.yC, vC=u.vC, phiC=u.phiC, gateC=False,
                xD=u.xD, yD=u.yD, vD=u.vD, phiD=u.phiD, gateD=u.gateD,
            )
        
        case MacroActionType4Puck.MOVE_C_LEFT_OFF:
            return Control4Pucks(
                xA=u.xA, yA=u.yA, vA=u.vA, phiA=u.phiA, gateA=u.gateA,
                xB=u.xB, yB=u.yB, vB=u.vB, phiB=u.phiB, gateB=u.gateB,
                xC=u.xC - mag, yC=u.yC, vC=u.vC, phiC=u.phiC, gateC=False,
                xD=u.xD, yD=u.yD, vD=u.vD, phiD=u.phiD, gateD=u.gateD,
            )
        
        case MacroActionType4Puck.MOVE_C_UP_OFF:
            return Control4Pucks(
                xA=u.xA, yA=u.yA, vA=u.vA, phiA=u.phiA, gateA=u.gateA,
                xB=u.xB, yB=u.yB, vB=u.vB, phiB=u.phiB, gateB=u.gateB,
                xC=u.xC, yC=u.yC + mag, vC=u.vC, phiC=u.phiC, gateC=False,
                xD=u.xD, yD=u.yD, vD=u.vD, phiD=u.phiD, gateD=u.gateD,
            )
        
        case MacroActionType4Puck.MOVE_C_DOWN_OFF:
            return Control4Pucks(
                xA=u.xA, yA=u.yA, vA=u.vA, phiA=u.phiA, gateA=u.gateA,
                xB=u.xB, yB=u.yB, vB=u.vB, phiB=u.phiB, gateB=u.gateB,
                xC=u.xC, yC=max(0.01e-3, u.yC - mag), vC=u.vC, phiC=u.phiC, gateC=False,
                xD=u.xD, yD=u.yD, vD=u.vD, phiD=u.phiD, gateD=u.gateD,
            )
        
        # ==========================================
        # Move-while-off: D
        # ==========================================
        case MacroActionType4Puck.MOVE_D_RIGHT_OFF:
            return Control4Pucks(
                xA=u.xA, yA=u.yA, vA=u.vA, phiA=u.phiA, gateA=u.gateA,
                xB=u.xB, yB=u.yB, vB=u.vB, phiB=u.phiB, gateB=u.gateB,
                xC=u.xC, yC=u.yC, vC=u.vC, phiC=u.phiC, gateC=u.gateC,
                xD=u.xD + mag, yD=u.yD, vD=u.vD, phiD=u.phiD, gateD=False,
            )
        
        case MacroActionType4Puck.MOVE_D_LEFT_OFF:
            return Control4Pucks(
                xA=u.xA, yA=u.yA, vA=u.vA, phiA=u.phiA, gateA=u.gateA,
                xB=u.xB, yB=u.yB, vB=u.vB, phiB=u.phiB, gateB=u.gateB,
                xC=u.xC, yC=u.yC, vC=u.vC, phiC=u.phiC, gateC=u.gateC,
                xD=u.xD - mag, yD=u.yD, vD=u.vD, phiD=u.phiD, gateD=False,
            )
        
        case MacroActionType4Puck.MOVE_D_UP_OFF:
            return Control4Pucks(
                xA=u.xA, yA=u.yA, vA=u.vA, phiA=u.phiA, gateA=u.gateA,
                xB=u.xB, yB=u.yB, vB=u.vB, phiB=u.phiB, gateB=u.gateB,
                xC=u.xC, yC=u.yC, vC=u.vC, phiC=u.phiC, gateC=u.gateC,
                xD=u.xD, yD=u.yD + mag, vD=u.vD, phiD=u.phiD, gateD=False,
            )
        
        case MacroActionType4Puck.MOVE_D_DOWN_OFF:
            return Control4Pucks(
                xA=u.xA, yA=u.yA, vA=u.vA, phiA=u.phiA, gateA=u.gateA,
                xB=u.xB, yB=u.yB, vB=u.vB, phiB=u.phiB, gateB=u.gateB,
                xC=u.xC, yC=u.yC, vC=u.vC, phiC=u.phiC, gateC=u.gateC,
                xD=u.xD, yD=max(0.01e-3, u.yD - mag), vD=u.vD, phiD=u.phiD, gateD=False,
            )
    
    return u  # Fallback


# ============================================================================
# Action subsets for different use cases
# ============================================================================

def get_all_actions_4puck() -> List[MacroActionType4Puck]:
    """Get all available macro action types."""
    return list(MacroActionType4Puck)


def get_standard_actions_4puck() -> List[MacroActionType4Puck]:
    """
    Get standard action set (excludes move-while-off).
    
    These are actions that immediately affect the acoustic field.
    """
    move_off_actions = {
        MacroActionType4Puck.MOVE_A_RIGHT_OFF,
        MacroActionType4Puck.MOVE_A_LEFT_OFF,
        MacroActionType4Puck.MOVE_A_UP_OFF,
        MacroActionType4Puck.MOVE_A_DOWN_OFF,
        MacroActionType4Puck.MOVE_B_RIGHT_OFF,
        MacroActionType4Puck.MOVE_B_LEFT_OFF,
        MacroActionType4Puck.MOVE_B_UP_OFF,
        MacroActionType4Puck.MOVE_B_DOWN_OFF,
        MacroActionType4Puck.MOVE_C_RIGHT_OFF,
        MacroActionType4Puck.MOVE_C_LEFT_OFF,
        MacroActionType4Puck.MOVE_C_UP_OFF,
        MacroActionType4Puck.MOVE_C_DOWN_OFF,
        MacroActionType4Puck.MOVE_D_RIGHT_OFF,
        MacroActionType4Puck.MOVE_D_LEFT_OFF,
        MacroActionType4Puck.MOVE_D_UP_OFF,
        MacroActionType4Puck.MOVE_D_DOWN_OFF,
    }
    return [a for a in MacroActionType4Puck if a not in move_off_actions]


def get_gating_actions_4puck() -> List[MacroActionType4Puck]:
    """Get only the toggle gate actions."""
    return [
        MacroActionType4Puck.TOGGLE_A_ON,
        MacroActionType4Puck.TOGGLE_A_OFF,
        MacroActionType4Puck.TOGGLE_B_ON,
        MacroActionType4Puck.TOGGLE_B_OFF,
        MacroActionType4Puck.TOGGLE_C_ON,
        MacroActionType4Puck.TOGGLE_C_OFF,
        MacroActionType4Puck.TOGGLE_D_ON,
        MacroActionType4Puck.TOGGLE_D_OFF,
    ]


def get_move_off_actions_4puck() -> List[MacroActionType4Puck]:
    """Get only the move-while-off actions."""
    return [
        MacroActionType4Puck.MOVE_A_RIGHT_OFF,
        MacroActionType4Puck.MOVE_A_LEFT_OFF,
        MacroActionType4Puck.MOVE_A_UP_OFF,
        MacroActionType4Puck.MOVE_A_DOWN_OFF,
        MacroActionType4Puck.MOVE_B_RIGHT_OFF,
        MacroActionType4Puck.MOVE_B_LEFT_OFF,
        MacroActionType4Puck.MOVE_B_UP_OFF,
        MacroActionType4Puck.MOVE_B_DOWN_OFF,
        MacroActionType4Puck.MOVE_C_RIGHT_OFF,
        MacroActionType4Puck.MOVE_C_LEFT_OFF,
        MacroActionType4Puck.MOVE_C_UP_OFF,
        MacroActionType4Puck.MOVE_C_DOWN_OFF,
        MacroActionType4Puck.MOVE_D_RIGHT_OFF,
        MacroActionType4Puck.MOVE_D_LEFT_OFF,
        MacroActionType4Puck.MOVE_D_UP_OFF,
        MacroActionType4Puck.MOVE_D_DOWN_OFF,
    ]


def get_3puck_compatible_actions_4puck() -> List[MacroActionType4Puck]:
    """
    Get actions that are equivalent to the 3-puck action set.
    
    Excludes D-specific actions and all gating/move-off actions.
    """
    return [
        MacroActionType4Puck.HOLD,
        MacroActionType4Puck.TRANSLATE_TRAP_X_POS,
        MacroActionType4Puck.TRANSLATE_TRAP_X_NEG,
        MacroActionType4Puck.TRANSLATE_TRAP_Y_POS,
        MacroActionType4Puck.TRANSLATE_TRAP_Y_NEG,
        MacroActionType4Puck.ROTATE_INTERFERENCE_CW,
        MacroActionType4Puck.ROTATE_INTERFERENCE_CCW,
        MacroActionType4Puck.STRENGTHEN_TRAP,
        MacroActionType4Puck.WEAKEN_TRAP,
        MacroActionType4Puck.WIDEN,
        MacroActionType4Puck.NARROW,
        MacroActionType4Puck.MOVE_A_RIGHT,
        MacroActionType4Puck.MOVE_A_LEFT,
        MacroActionType4Puck.MOVE_B_RIGHT,
        MacroActionType4Puck.MOVE_B_LEFT,
        MacroActionType4Puck.MOVE_C_UP,
        MacroActionType4Puck.MOVE_C_DOWN,
        MacroActionType4Puck.PHASE_SHIFT_B_POS,
        MacroActionType4Puck.PHASE_SHIFT_B_NEG,
        MacroActionType4Puck.PHASE_SHIFT_C_POS,
        MacroActionType4Puck.PHASE_SHIFT_C_NEG,
    ]


# ============================================================================
# Helper functions
# ============================================================================

def action_name_to_type(name: str) -> Optional[MacroActionType4Puck]:
    """Convert action name string to MacroActionType4Puck."""
    try:
        return MacroActionType4Puck[name]
    except KeyError:
        return None


def action_type_to_name(action_type: MacroActionType4Puck) -> str:
    """Convert MacroActionType4Puck to action name string."""
    return action_type.name


if __name__ == "__main__":
    # Quick test
    print(f"Total 4-puck macro actions: {len(MacroActionType4Puck)}")
    print(f"Standard actions: {len(get_standard_actions_4puck())}")
    print(f"Gating actions: {len(get_gating_actions_4puck())}")
    print(f"Move-off actions: {len(get_move_off_actions_4puck())}")
    print(f"3-puck compatible: {len(get_3puck_compatible_actions_4puck())}")
    
    # Test applying an action
    from tweezers.control import default_4puck_config
    u0 = default_4puck_config()
    print(f"\nInitial config: xD={u0.xD*1e3:.3f}mm, gateD={u0.gateD}")
    
    action = MacroAction4Puck(MacroActionType4Puck.MOVE_D_RIGHT_OFF)
    u1 = apply_macro_action_4puck(u0, action)
    print(f"After MOVE_D_RIGHT_OFF: xD={u1.xD*1e3:.3f}mm, gateD={u1.gateD}")
    
    action2 = MacroAction4Puck(MacroActionType4Puck.TOGGLE_D_ON)
    u2 = apply_macro_action_4puck(u1, action2)
    print(f"After TOGGLE_D_ON: xD={u2.xD*1e3:.3f}mm, gateD={u2.gateD}")
