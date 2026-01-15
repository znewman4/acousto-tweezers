#!/usr/bin/env python3
"""
Extended Macro Actions for 3-Puck Acoustic Tweezers Control.

STAGE 2 IMPLEMENTATION:
Provides structured control primitives that produce predictable trap effects.

Macro actions:
1. TRANSLATE_TRAP_X: Move trap in +X direction
2. TRANSLATE_TRAP_Y: Move trap in +Y direction  
3. ROTATE_INTERFERENCE: Rotate interference pattern
4. STRENGTHEN_TRAP: Increase trap stiffness
5. WEAKEN_TRAP: Decrease trap stiffness
6. WIDEN: Increase transducer spread
7. NARROW: Decrease transducer spread
8. PHASE_SHIFT_AB: Change phase between A and B
9. PHASE_SHIFT_C: Change phase of C

Each macro action:
- Has a deterministic control delta
- Has measurable trap effect
- Can be composed sequentially

Usage:
    python scripts/macro_actions_3puck.py --measure
    python scripts/macro_actions_3puck.py --visualize
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Callable
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from acousto.force import ParticleProps, gorkov_potential_and_force_2d
from acousto.analysis import find_trap_center
from tweezers.control import (
    DishDomain, MediumProps, EvaluatorConfig,
    Control3Pucks, Evaluator3Pucks, default_3puck_config,
)


class MacroActionType3Puck(Enum):
    """Macro action types for 3-puck control."""
    HOLD = auto()
    
    # Primary trap translation
    TRANSLATE_TRAP_X_POS = auto()
    TRANSLATE_TRAP_X_NEG = auto()
    TRANSLATE_TRAP_Y_POS = auto()
    TRANSLATE_TRAP_Y_NEG = auto()
    
    # Interference pattern control
    ROTATE_INTERFERENCE_CW = auto()
    ROTATE_INTERFERENCE_CCW = auto()
    
    # Stiffness control
    STRENGTHEN_TRAP = auto()
    WEAKEN_TRAP = auto()
    
    # Spread control
    WIDEN = auto()
    NARROW = auto()
    
    # Individual puck control
    MOVE_A_RIGHT = auto()
    MOVE_A_LEFT = auto()
    MOVE_B_RIGHT = auto()
    MOVE_B_LEFT = auto()
    MOVE_C_UP = auto()
    MOVE_C_DOWN = auto()
    
    # Phase control
    PHASE_SHIFT_B_POS = auto()
    PHASE_SHIFT_B_NEG = auto()
    PHASE_SHIFT_C_POS = auto()
    PHASE_SHIFT_C_NEG = auto()


@dataclass
class MacroAction3Puck:
    """A structured control primitive for 3-puck system."""
    action_type: MacroActionType3Puck
    magnitude: float = 0.05e-3   # Position step (m)
    phase_step: float = 0.15     # Phase step (rad)
    amplitude_step: float = 0.01  # Amplitude step


@dataclass
class MacroActionEffect3Puck:
    """Measured effect of a macro action on trap position."""
    action: MacroAction3Puck
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


def apply_macro_action_3puck(u: Control3Pucks, action: MacroAction3Puck) -> Control3Pucks:
    """Apply a macro action to a 3-puck control configuration."""
    mag = action.magnitude
    phase_step = action.phase_step
    amp_step = action.amplitude_step
    
    match action.action_type:
        case MacroActionType3Puck.HOLD:
            return u
        
        # === TRANSLATE_TRAP: Move all transducers together ===
        case MacroActionType3Puck.TRANSLATE_TRAP_X_POS:
            return Control3Pucks(
                xA=u.xA + mag, yA=u.yA, vA=u.vA, phiA=u.phiA,
                xB=u.xB + mag, yB=u.yB, vB=u.vB, phiB=u.phiB,
                xC=u.xC + mag, yC=u.yC, vC=u.vC, phiC=u.phiC,
            )
        
        case MacroActionType3Puck.TRANSLATE_TRAP_X_NEG:
            return Control3Pucks(
                xA=u.xA - mag, yA=u.yA, vA=u.vA, phiA=u.phiA,
                xB=u.xB - mag, yB=u.yB, vB=u.vB, phiB=u.phiB,
                xC=u.xC - mag, yC=u.yC, vC=u.vC, phiC=u.phiC,
            )
        
        case MacroActionType3Puck.TRANSLATE_TRAP_Y_POS:
            # Move via phase shift (y-control is phase-based)
            # Also slightly move C up since it has y-authority
            return Control3Pucks(
                xA=u.xA, yA=u.yA, vA=u.vA, phiA=u.phiA,
                xB=u.xB, yB=u.yB, vB=u.vB, phiB=u.phiB + phase_step * 0.5,
                xC=u.xC, yC=u.yC + mag * 0.5, vC=u.vC, phiC=u.phiC - phase_step * 0.5,
            )
        
        case MacroActionType3Puck.TRANSLATE_TRAP_Y_NEG:
            return Control3Pucks(
                xA=u.xA, yA=u.yA, vA=u.vA, phiA=u.phiA,
                xB=u.xB, yB=u.yB, vB=u.vB, phiB=u.phiB - phase_step * 0.5,
                xC=u.xC, yC=u.yC - mag * 0.5, vC=u.vC, phiC=u.phiC + phase_step * 0.5,
            )
        
        # === ROTATE_INTERFERENCE: Rotate interference pattern ===
        case MacroActionType3Puck.ROTATE_INTERFERENCE_CW:
            # Rotate by adjusting phases symmetrically
            return Control3Pucks(
                xA=u.xA, yA=u.yA, vA=u.vA, phiA=u.phiA + phase_step,
                xB=u.xB, yB=u.yB, vB=u.vB, phiB=u.phiB - phase_step,
                xC=u.xC, yC=u.yC, vC=u.vC, phiC=u.phiC,
            )
        
        case MacroActionType3Puck.ROTATE_INTERFERENCE_CCW:
            return Control3Pucks(
                xA=u.xA, yA=u.yA, vA=u.vA, phiA=u.phiA - phase_step,
                xB=u.xB, yB=u.yB, vB=u.vB, phiB=u.phiB + phase_step,
                xC=u.xC, yC=u.yC, vC=u.vC, phiC=u.phiC,
            )
        
        # === STRENGTHEN/WEAKEN: Change amplitudes ===
        case MacroActionType3Puck.STRENGTHEN_TRAP:
            return Control3Pucks(
                xA=u.xA, yA=u.yA, vA=u.vA + amp_step, phiA=u.phiA,
                xB=u.xB, yB=u.yB, vB=u.vB + amp_step, phiB=u.phiB,
                xC=u.xC, yC=u.yC, vC=u.vC + amp_step, phiC=u.phiC,
            )
        
        case MacroActionType3Puck.WEAKEN_TRAP:
            return Control3Pucks(
                xA=u.xA, yA=u.yA, vA=max(0.01, u.vA - amp_step), phiA=u.phiA,
                xB=u.xB, yB=u.yB, vB=max(0.01, u.vB - amp_step), phiB=u.phiB,
                xC=u.xC, yC=u.yC, vC=max(0.01, u.vC - amp_step), phiC=u.phiC,
            )
        
        # === WIDEN/NARROW: Change transducer spread ===
        case MacroActionType3Puck.WIDEN:
            return Control3Pucks(
                xA=u.xA - mag * 0.5, yA=u.yA, vA=u.vA, phiA=u.phiA,
                xB=u.xB + mag * 0.5, yB=u.yB, vB=u.vB, phiB=u.phiB,
                xC=u.xC, yC=u.yC, vC=u.vC, phiC=u.phiC,
            )
        
        case MacroActionType3Puck.NARROW:
            return Control3Pucks(
                xA=u.xA + mag * 0.5, yA=u.yA, vA=u.vA, phiA=u.phiA,
                xB=u.xB - mag * 0.5, yB=u.yB, vB=u.vB, phiB=u.phiB,
                xC=u.xC, yC=u.yC, vC=u.vC, phiC=u.phiC,
            )
        
        # === Individual puck moves ===
        case MacroActionType3Puck.MOVE_A_RIGHT:
            return Control3Pucks(
                xA=u.xA + mag, yA=u.yA, vA=u.vA, phiA=u.phiA,
                xB=u.xB, yB=u.yB, vB=u.vB, phiB=u.phiB,
                xC=u.xC, yC=u.yC, vC=u.vC, phiC=u.phiC,
            )
        
        case MacroActionType3Puck.MOVE_A_LEFT:
            return Control3Pucks(
                xA=u.xA - mag, yA=u.yA, vA=u.vA, phiA=u.phiA,
                xB=u.xB, yB=u.yB, vB=u.vB, phiB=u.phiB,
                xC=u.xC, yC=u.yC, vC=u.vC, phiC=u.phiC,
            )
        
        case MacroActionType3Puck.MOVE_B_RIGHT:
            return Control3Pucks(
                xA=u.xA, yA=u.yA, vA=u.vA, phiA=u.phiA,
                xB=u.xB + mag, yB=u.yB, vB=u.vB, phiB=u.phiB,
                xC=u.xC, yC=u.yC, vC=u.vC, phiC=u.phiC,
            )
        
        case MacroActionType3Puck.MOVE_B_LEFT:
            return Control3Pucks(
                xA=u.xA, yA=u.yA, vA=u.vA, phiA=u.phiA,
                xB=u.xB - mag, yB=u.yB, vB=u.vB, phiB=u.phiB,
                xC=u.xC, yC=u.yC, vC=u.vC, phiC=u.phiC,
            )
        
        case MacroActionType3Puck.MOVE_C_UP:
            return Control3Pucks(
                xA=u.xA, yA=u.yA, vA=u.vA, phiA=u.phiA,
                xB=u.xB, yB=u.yB, vB=u.vB, phiB=u.phiB,
                xC=u.xC, yC=u.yC + mag, vC=u.vC, phiC=u.phiC,
            )
        
        case MacroActionType3Puck.MOVE_C_DOWN:
            return Control3Pucks(
                xA=u.xA, yA=u.yA, vA=u.vA, phiA=u.phiA,
                xB=u.xB, yB=u.yB, vB=u.vB, phiB=u.phiB,
                xC=u.xC, yC=max(0.01e-3, u.yC - mag), vC=u.vC, phiC=u.phiC,
            )
        
        # === Phase shifts ===
        case MacroActionType3Puck.PHASE_SHIFT_B_POS:
            return Control3Pucks(
                xA=u.xA, yA=u.yA, vA=u.vA, phiA=u.phiA,
                xB=u.xB, yB=u.yB, vB=u.vB, phiB=u.phiB + phase_step,
                xC=u.xC, yC=u.yC, vC=u.vC, phiC=u.phiC,
            )
        
        case MacroActionType3Puck.PHASE_SHIFT_B_NEG:
            return Control3Pucks(
                xA=u.xA, yA=u.yA, vA=u.vA, phiA=u.phiA,
                xB=u.xB, yB=u.yB, vB=u.vB, phiB=u.phiB - phase_step,
                xC=u.xC, yC=u.yC, vC=u.vC, phiC=u.phiC,
            )
        
        case MacroActionType3Puck.PHASE_SHIFT_C_POS:
            return Control3Pucks(
                xA=u.xA, yA=u.yA, vA=u.vA, phiA=u.phiA,
                xB=u.xB, yB=u.yB, vB=u.vB, phiB=u.phiB,
                xC=u.xC, yC=u.yC, vC=u.vC, phiC=u.phiC + phase_step,
            )
        
        case MacroActionType3Puck.PHASE_SHIFT_C_NEG:
            return Control3Pucks(
                xA=u.xA, yA=u.yA, vA=u.vA, phiA=u.phiA,
                xB=u.xB, yB=u.yB, vB=u.vB, phiB=u.phiB,
                xC=u.xC, yC=u.yC, vC=u.vC, phiC=u.phiC - phase_step,
            )
    
    return u  # Fallback


def measure_action_effect_3puck(
    ev: Evaluator3Pucks,
    u_base: Control3Pucks,
    action: MacroAction3Puck,
    search_radius: float = 0.5e-3,
) -> MacroActionEffect3Puck:
    """
    Measure how a macro action affects trap position.
    
    1. Find trap for base control
    2. Apply action
    3. Find trap for new control
    4. Report difference
    """
    particle = ev.particle
    domain = ev.domain
    
    # Find trap for base control
    vb_base = ev.control_to_forcing_band_vb(u_base)
    field_base = ev.op.solve_for_bottom_vb(vb_base)
    U_base, Fx_base, Fy_base = gorkov_potential_and_force_2d(field_base, particle)
    
    # Search near centroid of transducers
    search_x = (u_base.xA + u_base.xB + u_base.xC) / 3
    search_y = domain.Ly / 2
    
    trap_base = find_trap_center(
        field_base.x, field_base.y, U_base, Fx_base, Fy_base,
        particle_x=search_x, particle_y=search_y,
        search_radius=search_radius,
    )
    
    if not trap_base.is_stable:
        return MacroActionEffect3Puck(
            action=action,
            delta_trap_x=np.nan, delta_trap_y=np.nan,
            delta_stiffness=np.nan, trap_found=False,
            initial_trap_x=np.nan, initial_trap_y=np.nan,
            final_trap_x=np.nan, final_trap_y=np.nan,
            initial_stiffness=np.nan, final_stiffness=np.nan,
        )
    
    stiff_base = np.mean(np.abs(trap_base.stiffness_eigvals))
    
    # Apply action
    u_new = apply_macro_action_3puck(u_base, action)
    u_new = ev.clip_control(u_new)
    
    # Find trap for new control
    vb_new = ev.control_to_forcing_band_vb(u_new)
    field_new = ev.op.solve_for_bottom_vb(vb_new)
    U_new, Fx_new, Fy_new = gorkov_potential_and_force_2d(field_new, particle)
    
    trap_new = find_trap_center(
        field_new.x, field_new.y, U_new, Fx_new, Fy_new,
        particle_x=trap_base.x, particle_y=trap_base.y,
        search_radius=search_radius,
    )
    
    if not trap_new.is_stable:
        return MacroActionEffect3Puck(
            action=action,
            delta_trap_x=np.nan, delta_trap_y=np.nan,
            delta_stiffness=np.nan, trap_found=False,
            initial_trap_x=trap_base.x, initial_trap_y=trap_base.y,
            final_trap_x=np.nan, final_trap_y=np.nan,
            initial_stiffness=stiff_base, final_stiffness=np.nan,
        )
    
    stiff_new = np.mean(np.abs(trap_new.stiffness_eigvals))
    
    return MacroActionEffect3Puck(
        action=action,
        delta_trap_x=trap_new.x - trap_base.x,
        delta_trap_y=trap_new.y - trap_base.y,
        delta_stiffness=stiff_new - stiff_base,
        trap_found=True,
        initial_trap_x=trap_base.x,
        initial_trap_y=trap_base.y,
        final_trap_x=trap_new.x,
        final_trap_y=trap_new.y,
        initial_stiffness=stiff_base,
        final_stiffness=stiff_new,
    )


def measure_all_primitives_3puck(
    ev: Evaluator3Pucks,
    u_base: Control3Pucks,
    magnitudes: list[float] | None = None,
) -> list[MacroActionEffect3Puck]:
    """Measure effects of all primitive actions at different magnitudes."""
    if magnitudes is None:
        magnitudes = [0.03e-3, 0.05e-3, 0.08e-3]
    
    results = []
    
    # Key actions to measure (skip HOLD)
    key_actions = [
        MacroActionType3Puck.TRANSLATE_TRAP_X_POS,
        MacroActionType3Puck.TRANSLATE_TRAP_X_NEG,
        MacroActionType3Puck.TRANSLATE_TRAP_Y_POS,
        MacroActionType3Puck.TRANSLATE_TRAP_Y_NEG,
        MacroActionType3Puck.ROTATE_INTERFERENCE_CW,
        MacroActionType3Puck.ROTATE_INTERFERENCE_CCW,
        MacroActionType3Puck.STRENGTHEN_TRAP,
        MacroActionType3Puck.WEAKEN_TRAP,
        MacroActionType3Puck.WIDEN,
        MacroActionType3Puck.NARROW,
        MacroActionType3Puck.MOVE_C_UP,
        MacroActionType3Puck.MOVE_C_DOWN,
        MacroActionType3Puck.PHASE_SHIFT_B_POS,
        MacroActionType3Puck.PHASE_SHIFT_C_POS,
    ]
    
    for action_type in key_actions:
        for mag in magnitudes:
            action = MacroAction3Puck(
                action_type=action_type,
                magnitude=mag,
                phase_step=mag * 2000,  # Scale for phase
                amplitude_step=mag * 200,  # Scale for amplitude
            )
            
            effect = measure_action_effect_3puck(ev, u_base, action)
            results.append(effect)
    
    return results


def select_best_action_3puck(
    target_dx: float,
    target_dy: float,
    effects: list[MacroActionEffect3Puck],
) -> MacroAction3Puck | None:
    """
    Select the macro action that best moves the trap toward target.
    
    Uses dot product of target direction with action effect.
    """
    target_mag = np.sqrt(target_dx**2 + target_dy**2)
    if target_mag < 1e-9:
        return MacroAction3Puck(action_type=MacroActionType3Puck.HOLD)
    
    target_dir = np.array([target_dx, target_dy]) / target_mag
    
    best_score = -np.inf
    best_action = None
    
    for effect in effects:
        if not effect.trap_found:
            continue
        if np.isnan(effect.delta_trap_x) or np.isnan(effect.delta_trap_y):
            continue
        
        action_vec = np.array([effect.delta_trap_x, effect.delta_trap_y])
        
        # Score = dot(target_dir, action_vec)
        # This favors actions that move in the right direction with large magnitude
        score = np.dot(target_dir, action_vec)
        
        if score > best_score:
            best_score = score
            best_action = effect.action
    
    return best_action


def build_action_effect_lookup(
    ev: Evaluator3Pucks,
    n_grid: int = 5,
) -> dict:
    """
    Build a lookup table of action effects at different configurations.
    
    Returns dict mapping (approximate position) -> effects list.
    """
    Lx = ev.domain.Lx
    Ly = ev.domain.Ly
    
    lookup = {}
    
    x_positions = np.linspace(0.25 * Lx, 0.75 * Lx, n_grid)
    
    for x_center in x_positions:
        # Create base control centered at x_center
        spread = 0.4e-3
        u_base = Control3Pucks(
            xA=x_center - spread, yA=0.03e-3, vA=0.08, phiA=0.0,
            xB=x_center + spread, yB=0.03e-3, vB=0.08, phiB=np.pi,
            xC=x_center, yC=0.15e-3, vC=0.08, phiC=np.pi/2,
        )
        
        effects = measure_all_primitives_3puck(ev, u_base)
        
        # Key by approximate position
        key = round(x_center * 1e6)  # µm
        lookup[key] = effects
    
    return lookup


def plot_action_effects_3puck(
    effects: list[MacroActionEffect3Puck],
    output_path: Path,
) -> None:
    """Visualize macro action effects."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    valid = [e for e in effects if e.trap_found]
    
    if not valid:
        print("No valid effects to plot")
        return
    
    # Group by action type
    from collections import defaultdict
    by_type = defaultdict(list)
    for e in valid:
        by_type[e.action.action_type].append(e)
    
    # === Panel 1: Displacement vectors ===
    ax = axes[0]
    colors = plt.cm.tab20.colors
    
    for i, (action_type, effects_list) in enumerate(by_type.items()):
        color = colors[i % len(colors)]
        
        for effect in effects_list:
            ax.arrow(
                effect.initial_trap_x * 1e3,
                effect.initial_trap_y * 1e3,
                effect.delta_trap_x * 1e3,
                effect.delta_trap_y * 1e3,
                head_width=0.02, head_length=0.01,
                fc=color, ec=color, alpha=0.7,
            )
        
        ax.scatter([], [], c=[color], label=action_type.name[:15])
    
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("Trap Displacement by Macro Action")
    ax.legend(fontsize=6, ncol=2)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # === Panel 2: Displacement magnitude by action type ===
    ax = axes[1]
    
    action_names = []
    mean_displacements = []
    
    for action_type, effects_list in by_type.items():
        disps = [np.sqrt(e.delta_trap_x**2 + e.delta_trap_y**2) * 1e3 
                 for e in effects_list]
        action_names.append(action_type.name[:12])
        mean_displacements.append(np.mean(disps))
    
    x_pos = np.arange(len(action_names))
    ax.barh(x_pos, mean_displacements, color='steelblue')
    ax.set_yticks(x_pos)
    ax.set_yticklabels(action_names, fontsize=7)
    ax.set_xlabel("Mean Displacement (mm)")
    ax.set_title("Effectiveness of Actions")
    ax.grid(True, alpha=0.3, axis='x')
    
    # === Panel 3: Direction histogram ===
    ax = axes[2]
    
    angles = []
    for e in valid:
        if np.sqrt(e.delta_trap_x**2 + e.delta_trap_y**2) > 1e-9:
            angle = np.arctan2(e.delta_trap_y, e.delta_trap_x)
            angles.append(angle)
    
    if angles:
        ax.hist(angles, bins=16, range=(-np.pi, np.pi), color='coral', edgecolor='black')
        ax.set_xlabel("Direction (rad)")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Trap Movement Directions")
        ax.axvline(0, color='k', linestyle='--', alpha=0.5, label='Right')
        ax.axvline(np.pi/2, color='r', linestyle='--', alpha=0.5, label='Up')
        ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path / "action_effects_3puck.png", dpi=150)
    plt.close()
    print(f"Saved: {output_path / 'action_effects_3puck.png'}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--measure", action="store_true", help="Measure all primitives")
    parser.add_argument("--visualize", action="store_true", help="Visualize effects")
    parser.add_argument("--output", type=str, default="results/macro_actions_3puck")
    args = parser.parse_args()
    
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Setup
    domain = DishDomain(Lx=2e-3, Ly=2e-3, Nx=100, Ny=100)
    medium = MediumProps(f=2e6, c0=1500.0, rho0=1000.0, loss_eta=1e-3, kz=0.0, coupling_alpha=1.0)
    particle = ParticleProps(a=5e-6, rho_p=1050.0, c_p=2350.0)
    
    cfg = EvaluatorConfig(
        sigma_x=0.10e-3,
        sigma_y=0.15e-3,
        bottom_band=0.25e-3,
        dt=5e-3,
        viscosity=1e-3,
        alpha_g=2e3,
        max_step=0.08e-3,
        use_2d_forcing=True,
    )
    
    ev = Evaluator3Pucks(domain, medium, particle, cfg)
    
    # Base control (3-puck spread configuration)
    u_base = Control3Pucks(
        xA=0.5e-3, yA=0.03e-3, vA=0.08, phiA=0.0,
        xB=1.5e-3, yB=0.03e-3, vB=0.08, phiB=np.pi,
        xC=1.0e-3, yC=0.15e-3, vC=0.08, phiC=np.pi/2,
    )
    
    if args.measure or args.visualize:
        print("=" * 60)
        print("3-PUCK MACRO ACTION MEASUREMENT")
        print("=" * 60)
        
        effects = measure_all_primitives_3puck(ev, u_base)
        
        # Print summary
        print("\n" + "=" * 80)
        print("MACRO ACTION EFFECTS SUMMARY")
        print("=" * 80)
        
        n_valid = sum(1 for e in effects if e.trap_found)
        print(f"Total measured: {len(effects)}, Valid: {n_valid}")
        print()
        
        for effect in effects:
            if effect.trap_found:
                disp = np.sqrt(effect.delta_trap_x**2 + effect.delta_trap_y**2)
                print(f"{effect.action.action_type.name:25} "
                      f"mag={effect.action.magnitude*1e3:+.3f}mm -> "
                      f"Δtrap=({effect.delta_trap_x*1e3:+.4f}, {effect.delta_trap_y*1e3:+.4f}) mm "
                      f"[{disp*1e3:.4f} mm]")
            else:
                print(f"{effect.action.action_type.name:25} "
                      f"mag={effect.action.magnitude*1e3:+.3f}mm -> NO TRAP FOUND")
        
        if args.visualize:
            plot_action_effects_3puck(effects, output_path)
        
        # Save data
        import json
        data = []
        for e in effects:
            data.append({
                "action_type": e.action.action_type.name,
                "magnitude_mm": e.action.magnitude * 1e3,
                "delta_trap_x_mm": e.delta_trap_x * 1e3 if e.trap_found else None,
                "delta_trap_y_mm": e.delta_trap_y * 1e3 if e.trap_found else None,
                "trap_found": e.trap_found,
            })
        
        with open(output_path / "action_effects.json", "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nSaved: {output_path / 'action_effects.json'}")
    
    else:
        print("Run with --measure or --visualize")


if __name__ == "__main__":
    main()
