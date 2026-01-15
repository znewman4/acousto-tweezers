#!/usr/bin/env python3
"""
Macro Actions for Acoustic Tweezers Control

Instead of random-shooting MPC at every step, this module defines structured
"macro actions" or primitives that move the trap in predictable ways.

Macro actions:
1. TRANSLATE_BOTH: Move both transducers in same direction (shifts trap center)
2. WIDEN: Increase separation (may change trap stiffness)
3. NARROW: Decrease separation
4. PHASE_ROTATE: Rotate phase difference (shifts interference pattern)
5. HOLD: Keep control constant

This allows:
- More interpretable control
- Pre-computed lookup of "which macro moves trap toward target"
- Hierarchical control: high-level chooses macro, low-level refines

Usage:
    python scripts/macro_actions.py --diagnose
    python scripts/macro_actions.py --measure-primitives
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
    Control2Pucks, BottomFootprint25DEvaluator,
)


class MacroActionType(Enum):
    """Types of macro actions for trap steering."""
    HOLD = auto()           # Keep control constant
    TRANSLATE_X = auto()    # Shift both transducers in x
    TRANSLATE_Y = auto()    # Shift both transducers in y (coupling change)
    WIDEN = auto()          # Increase separation between transducers
    NARROW = auto()         # Decrease separation
    PHASE_SHIFT = auto()    # Change phase difference
    BOOST_A = auto()        # Increase amplitude of transducer A
    BOOST_B = auto()        # Increase amplitude of transducer B


@dataclass
class MacroAction:
    """A structured control primitive."""
    action_type: MacroActionType
    magnitude: float = 0.05e-3  # Step size for position changes (m)
    phase_step: float = 0.1     # Step size for phase changes (rad)
    amplitude_step: float = 0.005  # Step size for amplitude changes


@dataclass
class MacroActionEffect:
    """Measured effect of a macro action on trap position."""
    action: MacroAction
    delta_trap_x: float
    delta_trap_y: float
    delta_stiffness: float
    trap_found: bool
    initial_trap_x: float
    initial_trap_y: float
    final_trap_x: float
    final_trap_y: float


def apply_macro_action(u: Control2Pucks, action: MacroAction) -> Control2Pucks:
    """Apply a macro action to a control configuration."""
    mag = action.magnitude
    phase_step = action.phase_step
    amp_step = action.amplitude_step
    
    match action.action_type:
        case MacroActionType.HOLD:
            return u
        
        case MacroActionType.TRANSLATE_X:
            return Control2Pucks(
                xA=u.xA + mag, yA=u.yA,
                xB=u.xB + mag, yB=u.yB,
                vA=u.vA, vB=u.vB,
                phiA=u.phiA, phiB=u.phiB,
            )
        
        case MacroActionType.TRANSLATE_Y:
            return Control2Pucks(
                xA=u.xA, yA=u.yA + mag,
                xB=u.xB, yB=u.yB + mag,
                vA=u.vA, vB=u.vB,
                phiA=u.phiA, phiB=u.phiB,
            )
        
        case MacroActionType.WIDEN:
            # Move A left, B right (symmetric widening)
            return Control2Pucks(
                xA=u.xA - mag/2, yA=u.yA,
                xB=u.xB + mag/2, yB=u.yB,
                vA=u.vA, vB=u.vB,
                phiA=u.phiA, phiB=u.phiB,
            )
        
        case MacroActionType.NARROW:
            # Move A right, B left (symmetric narrowing)
            return Control2Pucks(
                xA=u.xA + mag/2, yA=u.yA,
                xB=u.xB - mag/2, yB=u.yB,
                vA=u.vA, vB=u.vB,
                phiA=u.phiA, phiB=u.phiB,
            )
        
        case MacroActionType.PHASE_SHIFT:
            return Control2Pucks(
                xA=u.xA, yA=u.yA,
                xB=u.xB, yB=u.yB,
                vA=u.vA, vB=u.vB,
                phiA=u.phiA, phiB=u.phiB + phase_step,
            )
        
        case MacroActionType.BOOST_A:
            return Control2Pucks(
                xA=u.xA, yA=u.yA,
                xB=u.xB, yB=u.yB,
                vA=u.vA + amp_step, vB=u.vB,
                phiA=u.phiA, phiB=u.phiB,
            )
        
        case MacroActionType.BOOST_B:
            return Control2Pucks(
                xA=u.xA, yA=u.yA,
                xB=u.xB, yB=u.yB,
                vA=u.vA, vB=u.vB + amp_step,
                phiA=u.phiA, phiB=u.phiB,
            )


def measure_action_effect(
    ev: BottomFootprint25DEvaluator,
    u_base: Control2Pucks,
    action: MacroAction,
    search_radius: float = 0.5e-3,
) -> MacroActionEffect:
    """
    Measure how a macro action affects trap position.
    
    1. Compute field and find trap for u_base
    2. Apply action to get u_new
    3. Compute field and find trap for u_new
    4. Report the difference
    """
    particle = ev.particle
    
    # Find trap for base control
    vb_base = ev.control_to_forcing_band_vb(u_base)
    field_base = ev.op.solve_for_bottom_vb(vb_base)
    U_base, Fx_base, Fy_base = gorkov_potential_and_force_2d(field_base, particle)
    
    # Search near center of transducers
    search_x = (u_base.xA + u_base.xB) / 2
    search_y = ev.domain.Ly / 2
    
    trap_base = find_trap_center(
        field_base.x, field_base.y, U_base, Fx_base, Fy_base,
        particle_x=search_x, particle_y=search_y,
        search_radius=search_radius,
    )
    
    if not trap_base.is_stable:
        return MacroActionEffect(
            action=action,
            delta_trap_x=np.nan, delta_trap_y=np.nan,
            delta_stiffness=np.nan, trap_found=False,
            initial_trap_x=np.nan, initial_trap_y=np.nan,
            final_trap_x=np.nan, final_trap_y=np.nan,
        )
    
    # Apply action
    u_new = apply_macro_action(u_base, action)
    u_new = ev.clip_control(u_new)
    
    # Find trap for new control
    vb_new = ev.control_to_forcing_band_vb(u_new)
    field_new = ev.op.solve_for_bottom_vb(vb_new)
    U_new, Fx_new, Fy_new = gorkov_potential_and_force_2d(field_new, particle)
    
    trap_new = find_trap_center(
        field_new.x, field_new.y, U_new, Fx_new, Fy_new,
        particle_x=trap_base.x, particle_y=trap_base.y,  # Search near old trap
        search_radius=search_radius,
    )
    
    if not trap_new.is_stable:
        return MacroActionEffect(
            action=action,
            delta_trap_x=np.nan, delta_trap_y=np.nan,
            delta_stiffness=np.nan, trap_found=False,
            initial_trap_x=trap_base.x, initial_trap_y=trap_base.y,
            final_trap_x=np.nan, final_trap_y=np.nan,
        )
    
    # Compute effects
    delta_x = trap_new.x - trap_base.x
    delta_y = trap_new.y - trap_base.y
    
    # Stiffness from eigenvalues
    stiff_base = np.mean(np.abs(trap_base.stiffness_eigvals))
    stiff_new = np.mean(np.abs(trap_new.stiffness_eigvals))
    delta_stiff = stiff_new - stiff_base
    
    return MacroActionEffect(
        action=action,
        delta_trap_x=delta_x,
        delta_trap_y=delta_y,
        delta_stiffness=delta_stiff,
        trap_found=True,
        initial_trap_x=trap_base.x,
        initial_trap_y=trap_base.y,
        final_trap_x=trap_new.x,
        final_trap_y=trap_new.y,
    )


def measure_all_primitives(
    ev: BottomFootprint25DEvaluator,
    u_base: Control2Pucks,
    magnitudes: list[float] | None = None,
) -> list[MacroActionEffect]:
    """Measure effects of all primitive actions at different magnitudes."""
    if magnitudes is None:
        magnitudes = [0.05e-3, 0.1e-3, 0.15e-3]
    
    results = []
    
    for action_type in MacroActionType:
        if action_type == MacroActionType.HOLD:
            continue
        
        for mag in magnitudes:
            action = MacroAction(
                action_type=action_type,
                magnitude=mag,
                phase_step=mag * 1000,  # Scale for phase (radians ~= mm * 1000)
                amplitude_step=mag * 100,  # Scale for amplitude
            )
            
            effect = measure_action_effect(ev, u_base, action)
            results.append(effect)
            
            # Also try negative magnitude
            action_neg = MacroAction(
                action_type=action_type,
                magnitude=-mag,
                phase_step=-mag * 1000,
                amplitude_step=-mag * 100,
            )
            effect_neg = measure_action_effect(ev, u_base, action_neg)
            results.append(effect_neg)
    
    return results


def select_best_action(
    target_dx: float,
    target_dy: float,
    effects: list[MacroActionEffect],
) -> MacroAction | None:
    """
    Select the macro action that best moves the trap toward the target.
    
    Uses dot product of (target direction) · (action effect) as score.
    """
    target_mag = np.sqrt(target_dx**2 + target_dy**2)
    if target_mag < 1e-9:
        return MacroAction(action_type=MacroActionType.HOLD)
    
    target_dir = np.array([target_dx, target_dy]) / target_mag
    
    best_score = -np.inf
    best_action = None
    
    for effect in effects:
        if not effect.trap_found:
            continue
        
        action_vec = np.array([effect.delta_trap_x, effect.delta_trap_y])
        action_mag = np.linalg.norm(action_vec)
        
        if action_mag < 1e-9:
            continue
        
        # Score = dot(target_dir, action_dir) * action_magnitude
        # This favors actions that move in the right direction with large magnitude
        score = np.dot(target_dir, action_vec)
        
        if score > best_score:
            best_score = score
            best_action = effect.action
    
    return best_action


def plot_action_effects(
    effects: list[MacroActionEffect],
    output_path: Path,
) -> None:
    """Visualize macro action effects as vectors from initial trap position."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Filter to successful measurements
    valid = [e for e in effects if e.trap_found]
    
    if not valid:
        print("No valid action effects to plot")
        return
    
    # Group by action type
    from collections import defaultdict
    by_type = defaultdict(list)
    for e in valid:
        by_type[e.action.action_type].append(e)
    
    # Plot 1: Trap displacement vectors
    ax = axes[0]
    colors = plt.cm.tab10.colors
    
    for i, (action_type, effects_list) in enumerate(by_type.items()):
        color = colors[i % len(colors)]
        
        for effect in effects_list:
            # Arrow from initial to final trap position
            ax.arrow(
                effect.initial_trap_x * 1e3,
                effect.initial_trap_y * 1e3,
                effect.delta_trap_x * 1e3,
                effect.delta_trap_y * 1e3,
                head_width=0.02, head_length=0.01,
                fc=color, ec=color, alpha=0.7,
            )
        
        # Legend entry
        ax.scatter([], [], c=[color], label=action_type.name)
    
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("Trap Displacement by Macro Action")
    ax.legend(fontsize=8)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Bar chart of displacement magnitude by action type
    ax = axes[1]
    
    action_names = []
    mean_displacements = []
    
    for action_type, effects_list in by_type.items():
        disps = [np.sqrt(e.delta_trap_x**2 + e.delta_trap_y**2) * 1e3 
                 for e in effects_list]
        action_names.append(action_type.name)
        mean_displacements.append(np.mean(disps))
    
    x_pos = np.arange(len(action_names))
    ax.bar(x_pos, mean_displacements, color='steelblue')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(action_names, rotation=45, ha='right')
    ax.set_ylabel("Mean Trap Displacement (mm)")
    ax.set_title("Effectiveness of Macro Actions")
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


class MacroActionController:
    """
    High-level controller that uses macro actions instead of random shooting.
    
    At each step:
    1. Compute current trap position
    2. Compute target displacement
    3. Look up which macro action moves trap closest to target
    4. Apply that action
    """
    
    def __init__(
        self,
        ev: BottomFootprint25DEvaluator,
        magnitudes: list[float] | None = None,
    ):
        self.ev = ev
        self.magnitudes = magnitudes or [0.02e-3, 0.05e-3, 0.1e-3]
        
        # Pre-computed action effects (will be filled on demand)
        self._action_cache: dict[tuple[float, float], list[MacroActionEffect]] = {}
    
    def _get_action_effects(self, u: Control2Pucks) -> list[MacroActionEffect]:
        """Get cached or compute action effects for control configuration."""
        key = (round(u.xA, 6), round(u.xB, 6))  # Approximate caching
        
        if key not in self._action_cache:
            effects = measure_all_primitives(self.ev, u, self.magnitudes)
            self._action_cache[key] = effects
        
        return self._action_cache[key]
    
    def step(
        self,
        u: Control2Pucks,
        target_x: float,
        target_y: float,
        trap_x: float,
        trap_y: float,
    ) -> Control2Pucks:
        """
        Take one macro action step toward target.
        
        Parameters
        ----------
        u : Control2Pucks
            Current control configuration.
        target_x, target_y : float
            Target trap position.
        trap_x, trap_y : float
            Current trap position.
        
        Returns
        -------
        u_new : Control2Pucks
            Updated control configuration after applying best macro action.
        """
        # Compute desired trap displacement
        dx_target = target_x - trap_x
        dy_target = target_y - trap_y
        
        # Get action effects (cached or computed)
        effects = self._get_action_effects(u)
        
        # Select best action
        best_action = select_best_action(dx_target, dy_target, effects)
        
        if best_action is None:
            return u  # No good action found, hold
        
        # Apply action
        u_new = apply_macro_action(u, best_action)
        return self.ev.clip_control(u_new)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnose", action="store_true", help="Run diagnostic")
    parser.add_argument("--measure-primitives", action="store_true", 
                       help="Measure all primitive effects")
    args = parser.parse_args()
    
    # Setup
    domain = DishDomain(Lx=2e-3, Ly=2e-3, Nx=100, Ny=100)
    medium = MediumProps(f=2e6, c0=1500.0, rho0=1000.0, loss_eta=1e-3)
    particle = ParticleProps(a=5e-6, rho_p=1050.0, c_p=2350.0)
    
    cfg = EvaluatorConfig(
        sigma_x=0.10e-3,
        sigma_y=0.15e-3,
        bottom_band=0.25e-3,
        dt=5e-3,
        viscosity=1e-3,
        alpha_g=1e3,
        max_step=0.05e-3,
        use_2d_forcing=True,
    )
    ev = BottomFootprint25DEvaluator(domain, medium, particle, cfg)
    
    # Base control
    u_base = Control2Pucks(
        xA=0.5e-3, yA=0.05e-3,
        xB=1.5e-3, yB=0.05e-3,
        vA=0.05, vB=0.05,
        phiA=0.0, phiB=np.pi,
    )
    
    output_dir = Path(__file__).parents[1] / "results" / "macro_actions"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.diagnose or args.measure_primitives:
        print("Measuring macro action effects...")
        
        effects = measure_all_primitives(ev, u_base)
        
        # Print summary
        print("\n" + "=" * 60)
        print("MACRO ACTION EFFECTS SUMMARY")
        print("=" * 60)
        
        for effect in effects:
            if effect.trap_found:
                print(f"{effect.action.action_type.name:15} "
                      f"mag={effect.action.magnitude*1e3:+.3f}mm -> "
                      f"Δtrap=({effect.delta_trap_x*1e3:+.4f}, {effect.delta_trap_y*1e3:+.4f}) mm")
            else:
                print(f"{effect.action.action_type.name:15} "
                      f"mag={effect.action.magnitude*1e3:+.3f}mm -> NO TRAP FOUND")
        
        # Plot
        plot_action_effects(effects, output_dir / "action_effects.png")
        
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
        
        with open(output_dir / "action_effects.json", "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nSaved: {output_dir / 'action_effects.json'}")
    
    else:
        print("Run with --diagnose or --measure-primitives")


if __name__ == "__main__":
    main()
