#!/usr/bin/env python3
"""
Control Authority Diagnostic Experiment for 4-Puck System with Gating

PURPOSE:
Quantify which control knobs (macro actions) actually matter for shaping
the force field at the particle. Extended version supporting:
- 4 transducers (A, B, C, D)
- ON/OFF gating actions
- Move-while-off actions

OUTPUTS:
results/control_authority_4puck/run_YYYYMMDD_HHMMSS/
    - control_authority.csv: One row per action with all metrics
    - summary_bars.png: Bar charts ranking actions
    - printed summary at end

Usage:
    python scripts/control_authority_diagnostic_4puck.py
    python scripts/control_authority_diagnostic_4puck.py --fast
    python scripts/control_authority_diagnostic_4puck.py --action_subset standard
    python scripts/control_authority_diagnostic_4puck.py --action_subset all
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List

# Add project root to path
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from acousto.force import ParticleProps, gorkov_potential_and_force_2d, bilinear_sample_vec
from acousto.analysis import find_trap_center

from tweezers.control import (
    DishDomain, MediumProps, EvaluatorConfig,
    Control4Pucks, default_4puck_config, default_4puck_spread,
    Evaluator4Pucks,
)

from scripts.macro_actions_4puck import (
    MacroAction4Puck,
    MacroActionType4Puck,
    apply_macro_action_4puck,
    get_all_actions_4puck,
    get_standard_actions_4puck,
    get_3puck_compatible_actions_4puck,
    get_gating_actions_4puck,
    get_move_off_actions_4puck,
)


# =============================================================================
# Data structures for metrics
# =============================================================================

@dataclass
class ActionMetrics:
    """Comprehensive metrics for one action's control authority."""
    action_name: str
    
    # Local metrics (at particle position)
    baseline_force_mag: float
    delta_force_mag: float
    delta_force_mag_rel: float
    delta_force_x: float
    delta_force_y: float
    
    # Directional control
    delta_force_proj_x: float      # Fixed-direction projection toward +x
    delta_force_proj_y: float      # Fixed-direction projection toward +y  
    max_delta_force_proj: float    # Maximum projection over all test directions
    best_direction_deg: float
    
    # Global field metrics
    baseline_U_std: float
    delta_U_std: float
    baseline_U_range_p95_p5: float
    delta_U_range: float
    
    # Trap metrics
    baseline_trap_present: bool
    action_trap_present: bool
    trap_appeared: bool
    trap_disappeared: bool
    delta_stiffness_min: float
    delta_trap_x: float
    delta_trap_y: float
    
    # Gating info
    active_transducers_before: int
    active_transducers_after: int
    
    # Solver time
    solver_time_ms: float


@dataclass
class BaselineState:
    """Baseline state from which all actions are compared."""
    particle_x: float
    particle_y: float
    control: Control4Pucks
    
    # Baseline field solution
    field: object
    U: np.ndarray
    Fx: np.ndarray
    Fy: np.ndarray
    Fx_scaled: np.ndarray
    Fy_scaled: np.ndarray
    
    # Force at particle
    fx_at_particle: float
    fy_at_particle: float
    force_mag_at_particle: float
    
    # Field statistics
    U_std: float
    U_p5: float
    U_p95: float
    U_range_p95_p5: float
    
    # Trap info
    trap_present: bool
    trap_x: float
    trap_y: float
    stiffness_min: float
    
    # Active transducers
    active_transducers: int


# =============================================================================
# Core evaluation functions
# =============================================================================

def compute_baseline(
    ev: Evaluator4Pucks,
    particle_x: float,
    particle_y: float,
    ctrl: Control4Pucks,
) -> BaselineState:
    """Solve baseline field and extract all reference metrics."""
    
    vb = ev.control_to_forcing_band_vb(ctrl)
    field = ev.op.solve_for_bottom_vb(vb)
    U, Fx, Fy = gorkov_potential_and_force_2d(field, ev.particle)
    
    Fx_scaled = Fx * ev.cfg.alpha_g
    Fy_scaled = Fy * ev.cfg.alpha_g
    
    fx, fy = bilinear_sample_vec(field.x, field.y, Fx_scaled, Fy_scaled, particle_x, particle_y)
    force_mag = np.sqrt(fx**2 + fy**2)
    
    U_flat = U.flatten()
    U_std = float(np.std(U_flat))
    U_p5 = float(np.percentile(U_flat, 5))
    U_p95 = float(np.percentile(U_flat, 95))
    
    trap = find_trap_center(
        field.x, field.y, U, Fx, Fy,
        particle_x=particle_x, particle_y=particle_y,
        search_radius=0.5e-3,
    )
    trap_present = trap.is_stable
    trap_x = trap.x if trap.x is not None else np.nan
    trap_y = trap.y if trap.y is not None else np.nan
    stiffness_min = float(np.min(trap.stiffness_eigvals)) if trap.stiffness_eigvals is not None else np.nan
    
    return BaselineState(
        particle_x=particle_x,
        particle_y=particle_y,
        control=ctrl,
        field=field,
        U=U,
        Fx=Fx,
        Fy=Fy,
        Fx_scaled=Fx_scaled,
        Fy_scaled=Fy_scaled,
        fx_at_particle=fx,
        fy_at_particle=fy,
        force_mag_at_particle=force_mag,
        U_std=U_std,
        U_p5=U_p5,
        U_p95=U_p95,
        U_range_p95_p5=U_p95 - U_p5,
        trap_present=trap_present,
        trap_x=trap_x,
        trap_y=trap_y,
        stiffness_min=stiffness_min,
        active_transducers=ctrl.active_count(),
    )


def evaluate_action(
    ev: Evaluator4Pucks,
    baseline: BaselineState,
    action_type: MacroActionType4Puck,
    macro_magnitude: float,
    macro_phase_step: float,
    macro_amplitude_step: float,
    test_directions: np.ndarray,
) -> ActionMetrics:
    """Evaluate one action's control authority relative to baseline."""
    
    t0 = time.perf_counter()
    
    action = MacroAction4Puck(
        action_type=action_type,
        magnitude=macro_magnitude,
        phase_step=macro_phase_step,
        amplitude_step=macro_amplitude_step,
    )
    u_new = apply_macro_action_4puck(baseline.control, action)
    u_new = ev.clip_control(u_new)
    
    vb = ev.control_to_forcing_band_vb(u_new)
    field = ev.op.solve_for_bottom_vb(vb)
    U, Fx, Fy = gorkov_potential_and_force_2d(field, ev.particle)
    
    Fx_scaled = Fx * ev.cfg.alpha_g
    Fy_scaled = Fy * ev.cfg.alpha_g
    
    solver_time_ms = (time.perf_counter() - t0) * 1000.0
    
    fx, fy = bilinear_sample_vec(
        field.x, field.y, Fx_scaled, Fy_scaled,
        baseline.particle_x, baseline.particle_y
    )
    force_mag = np.sqrt(fx**2 + fy**2)
    
    delta_fx = fx - baseline.fx_at_particle
    delta_fy = fy - baseline.fy_at_particle
    delta_force_mag = force_mag - baseline.force_mag_at_particle
    delta_force_mag_rel = delta_force_mag / (baseline.force_mag_at_particle + 1e-15)
    
    # Directional control
    delta_force_projs = []
    for d_hat in test_directions:
        proj_baseline = baseline.fx_at_particle * d_hat[0] + baseline.fy_at_particle * d_hat[1]
        proj_action = fx * d_hat[0] + fy * d_hat[1]
        delta_force_projs.append(proj_action - proj_baseline)
    
    delta_force_projs = np.array(delta_force_projs)
    # Fixed-direction projections (more meaningful than avg which cancels to ~0)
    delta_force_proj_x = float(delta_fx)  # Projection onto +x
    delta_force_proj_y = float(delta_fy)  # Projection onto +y
    max_idx = int(np.argmax(delta_force_projs))
    max_delta_force_proj = float(delta_force_projs[max_idx])
    best_direction_deg = float(np.degrees(np.arctan2(test_directions[max_idx, 1], test_directions[max_idx, 0])))
    
    U_flat = U.flatten()
    U_std = float(np.std(U_flat))
    U_p5 = float(np.percentile(U_flat, 5))
    U_p95 = float(np.percentile(U_flat, 95))
    
    delta_U_std = U_std - baseline.U_std
    delta_U_range = (U_p95 - U_p5) - baseline.U_range_p95_p5
    
    trap = find_trap_center(
        field.x, field.y, U, Fx, Fy,
        particle_x=baseline.particle_x, particle_y=baseline.particle_y,
        search_radius=0.5e-3,
    )
    action_trap_present = trap.is_stable
    action_trap_x = trap.x if trap.x is not None else np.nan
    action_trap_y = trap.y if trap.y is not None else np.nan
    action_stiffness_min = float(np.min(trap.stiffness_eigvals)) if trap.stiffness_eigvals is not None else np.nan
    
    trap_appeared = (not baseline.trap_present) and action_trap_present
    trap_disappeared = baseline.trap_present and (not action_trap_present)
    
    if baseline.trap_present and action_trap_present:
        delta_stiffness_min = action_stiffness_min - baseline.stiffness_min
        delta_trap_x = action_trap_x - baseline.trap_x
        delta_trap_y = action_trap_y - baseline.trap_y
    else:
        delta_stiffness_min = np.nan
        delta_trap_x = np.nan
        delta_trap_y = np.nan
    
    return ActionMetrics(
        action_name=action_type.name,
        baseline_force_mag=baseline.force_mag_at_particle,
        delta_force_mag=delta_force_mag,
        delta_force_mag_rel=delta_force_mag_rel,
        delta_force_x=delta_fx,
        delta_force_y=delta_fy,
        delta_force_proj_x=delta_force_proj_x,
        delta_force_proj_y=delta_force_proj_y,
        max_delta_force_proj=max_delta_force_proj,
        best_direction_deg=best_direction_deg,
        baseline_U_std=baseline.U_std,
        delta_U_std=delta_U_std,
        baseline_U_range_p95_p5=baseline.U_range_p95_p5,
        delta_U_range=delta_U_range,
        baseline_trap_present=baseline.trap_present,
        action_trap_present=action_trap_present,
        trap_appeared=trap_appeared,
        trap_disappeared=trap_disappeared,
        delta_stiffness_min=delta_stiffness_min,
        delta_trap_x=delta_trap_x,
        delta_trap_y=delta_trap_y,
        active_transducers_before=baseline.active_transducers,
        active_transducers_after=u_new.active_count(),
        solver_time_ms=solver_time_ms,
    )


# =============================================================================
# Output generation
# =============================================================================

def write_csv(metrics: list[ActionMetrics], path: Path) -> None:
    """Write metrics to CSV file."""
    fieldnames = [
        "action_name",
        "delta_force_mag",
        "delta_force_mag_rel",
        "delta_force_x",
        "delta_force_y",
        "delta_force_proj_x",
        "delta_force_proj_y",
        "max_delta_force_proj",
        "best_direction_deg",
        "delta_U_std",
        "delta_U_range",
        "baseline_trap_present",
        "action_trap_present",
        "trap_appeared",
        "trap_disappeared",
        "delta_stiffness_min",
        "delta_trap_x",
        "delta_trap_y",
        "active_transducers_before",
        "active_transducers_after",
        "solver_time_ms",
    ]
    
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in metrics:
            row = {
                "action_name": m.action_name,
                "delta_force_mag": f"{m.delta_force_mag:.6e}",
                "delta_force_mag_rel": f"{m.delta_force_mag_rel:.4f}",
                "delta_force_x": f"{m.delta_force_x:.6e}",
                "delta_force_y": f"{m.delta_force_y:.6e}",
                "delta_force_proj_x": f"{m.delta_force_proj_x:.6e}",
                "delta_force_proj_y": f"{m.delta_force_proj_y:.6e}",
                "max_delta_force_proj": f"{m.max_delta_force_proj:.6e}",
                "best_direction_deg": f"{m.best_direction_deg:.1f}",
                "delta_U_std": f"{m.delta_U_std:.6e}",
                "delta_U_range": f"{m.delta_U_range:.6e}",
                "baseline_trap_present": str(m.baseline_trap_present),
                "action_trap_present": str(m.action_trap_present),
                "trap_appeared": str(m.trap_appeared),
                "trap_disappeared": str(m.trap_disappeared),
                "delta_stiffness_min": f"{m.delta_stiffness_min:.6e}" if not np.isnan(m.delta_stiffness_min) else "nan",
                "delta_trap_x": f"{m.delta_trap_x:.6e}" if not np.isnan(m.delta_trap_x) else "nan",
                "delta_trap_y": f"{m.delta_trap_y:.6e}" if not np.isnan(m.delta_trap_y) else "nan",
                "active_transducers_before": m.active_transducers_before,
                "active_transducers_after": m.active_transducers_after,
                "solver_time_ms": f"{m.solver_time_ms:.1f}",
            }
            writer.writerow(row)


def plot_summary_bars(metrics: list[ActionMetrics], baseline: BaselineState, path: Path) -> None:
    """Create bar charts ranking actions by various metrics."""
    
    names = [m.action_name for m in metrics]
    n_actions = len(metrics)
    
    # Use larger figure for many actions
    fig_height = max(10, n_actions * 0.25)
    fig, axes = plt.subplots(2, 2, figsize=(16, fig_height))
    fig.suptitle(
        f"Control Authority Diagnostic (4-Puck System)\n"
        f"Particle at ({baseline.particle_x*1e3:.2f}, {baseline.particle_y*1e3:.2f}) mm, "
        f"Baseline |F| = {baseline.force_mag_at_particle:.2e} N, "
        f"Active: {baseline.active_transducers}/4",
        fontsize=12,
    )
    
    colors = plt.cm.tab20(np.linspace(0, 1, min(20, n_actions)))
    if n_actions > 20:
        colors = np.tile(colors, (n_actions // 20 + 1, 1))[:n_actions]
    
    # Panel 1: Force magnitude change
    ax1 = axes[0, 0]
    delta_mags = [m.delta_force_mag for m in metrics]
    sorted_idx = np.argsort(delta_mags)[::-1]
    sorted_names = [names[i] for i in sorted_idx]
    sorted_vals = [delta_mags[i] for i in sorted_idx]
    
    ax1.barh(range(n_actions), sorted_vals, color=[colors[i % len(colors)] for i in range(n_actions)])
    ax1.set_yticks(range(n_actions))
    ax1.set_yticklabels(sorted_names, fontsize=7)
    ax1.set_xlabel("Δ|F| (N)")
    ax1.set_title("Force Magnitude Change")
    ax1.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    ax1.invert_yaxis()
    
    # Panel 2: Max directional control
    ax2 = axes[0, 1]
    max_projs = [m.max_delta_force_proj for m in metrics]
    sorted_idx = np.argsort(max_projs)[::-1]
    sorted_names = [names[i] for i in sorted_idx]
    sorted_vals = [max_projs[i] for i in sorted_idx]
    
    ax2.barh(range(n_actions), sorted_vals, color=[colors[i % len(colors)] for i in range(n_actions)])
    ax2.set_yticks(range(n_actions))
    ax2.set_yticklabels(sorted_names, fontsize=7)
    ax2.set_xlabel("Max Δ(F·d̂) (N)")
    ax2.set_title("Best Directional Control")
    ax2.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    ax2.invert_yaxis()
    
    # Panel 3: Global field change
    ax3 = axes[1, 0]
    delta_stds = [m.delta_U_std for m in metrics]
    sorted_idx = np.argsort(np.abs(delta_stds))[::-1]
    sorted_names = [names[i] for i in sorted_idx]
    sorted_vals = [delta_stds[i] for i in sorted_idx]
    
    ax3.barh(range(n_actions), sorted_vals, color=[colors[i % len(colors)] for i in range(n_actions)])
    ax3.set_yticks(range(n_actions))
    ax3.set_yticklabels(sorted_names, fontsize=7)
    ax3.set_xlabel("Δstd(U)")
    ax3.set_title("Global Field Change")
    ax3.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    ax3.invert_yaxis()
    
    # Panel 4: Aggregate score
    ax4 = axes[1, 1]
    
    abs_delta_mags = np.abs(delta_mags)
    abs_max_projs = np.abs(max_projs)
    abs_delta_stds = np.abs(delta_stds)
    
    norm_mags = abs_delta_mags / (np.max(abs_delta_mags) + 1e-15)
    norm_projs = abs_max_projs / (np.max(abs_max_projs) + 1e-15)
    norm_stds = abs_delta_stds / (np.max(abs_delta_stds) + 1e-15)
    
    aggregate_scores = norm_mags + norm_projs + norm_stds
    
    sorted_idx = np.argsort(aggregate_scores)[::-1]
    sorted_names = [names[i] for i in sorted_idx]
    sorted_vals = [aggregate_scores[i] for i in sorted_idx]
    
    ax4.barh(range(n_actions), sorted_vals, color=[colors[i % len(colors)] for i in range(n_actions)])
    ax4.set_yticks(range(n_actions))
    ax4.set_yticklabels(sorted_names, fontsize=7)
    ax4.set_xlabel("Aggregate Score (normalized)")
    ax4.set_title("Combined Control Authority")
    ax4.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def print_summary(metrics: list[ActionMetrics], baseline: BaselineState, top_k: int = 5) -> None:
    """Print interpretable summary to stdout."""
    
    print("\n" + "=" * 70)
    print("CONTROL AUTHORITY DIAGNOSTIC - 4-PUCK SYSTEM - SUMMARY")
    print("=" * 70)
    
    print(f"\nBaseline State:")
    print(f"  Particle position: ({baseline.particle_x*1e3:.2f}, {baseline.particle_y*1e3:.2f}) mm")
    print(f"  Force at particle: ({baseline.fx_at_particle:.2e}, {baseline.fy_at_particle:.2e}) N")
    print(f"  Force magnitude: {baseline.force_mag_at_particle:.2e} N")
    print(f"  Active transducers: {baseline.active_transducers}/4")
    print(f"  Trap present: {baseline.trap_present}")
    
    # Rank by max directional control
    by_dir_ctrl = sorted(metrics, key=lambda m: m.max_delta_force_proj, reverse=True)
    
    print(f"\n--- Top {top_k} Actions by DIRECTIONAL CONTROL (max Δ(F·d̂)) ---")
    for i, m in enumerate(by_dir_ctrl[:top_k]):
        print(f"  {i+1}. {m.action_name:35s}  Δ(F·d̂) = {m.max_delta_force_proj:+.2e} N  @ {m.best_direction_deg:+6.1f}°")
    
    # Rank by force magnitude change
    by_force_mag = sorted(metrics, key=lambda m: abs(m.delta_force_mag), reverse=True)
    
    print(f"\n--- Top {top_k} Actions by FORCE MAGNITUDE CHANGE (|Δ|F||) ---")
    for i, m in enumerate(by_force_mag[:top_k]):
        print(f"  {i+1}. {m.action_name:35s}  Δ|F| = {m.delta_force_mag:+.2e} N  ({m.delta_force_mag_rel:+.1%})")
    
    # Low-effect actions
    threshold = 0.05 * max(abs(m.max_delta_force_proj) for m in metrics)
    low_effect = [m for m in metrics if abs(m.max_delta_force_proj) < threshold]
    
    print(f"\n--- Actions with NEAR-ZERO Effect (candidates for pruning) ---")
    if low_effect:
        for m in low_effect:
            print(f"  - {m.action_name:35s}  Δ(F·d̂) = {m.max_delta_force_proj:+.2e} N")
    else:
        print("  (None - all actions have significant effect)")
    
    # Gating actions analysis
    gating_actions = [m for m in metrics if "TOGGLE" in m.action_name]
    if gating_actions:
        print(f"\n--- GATING ACTIONS Analysis ---")
        for m in gating_actions:
            change = "ON" if m.active_transducers_after > m.active_transducers_before else "OFF"
            print(f"  {m.action_name:25s}  {m.active_transducers_before}→{m.active_transducers_after} transducers  "
                  f"Δ|F| = {m.delta_force_mag:+.2e} N")
    
    # Move-while-off actions
    move_off_actions = [m for m in metrics if "_OFF" in m.action_name and "TOGGLE" not in m.action_name]
    if move_off_actions:
        print(f"\n--- MOVE-WHILE-OFF ACTIONS Analysis ---")
        # These should have near-zero effect since transducer is off
        for m in move_off_actions:
            print(f"  {m.action_name:25s}  Δ|F| = {m.delta_force_mag:+.2e} N  "
                  f"(expect ~0 since transducer off)")
    
    # Transducer D actions
    d_actions = [m for m in metrics if "_D_" in m.action_name or m.action_name.startswith("MOVE_D")]
    if d_actions:
        print(f"\n--- TRANSDUCER D ACTIONS Analysis ---")
        d_sorted = sorted(d_actions, key=lambda m: abs(m.max_delta_force_proj), reverse=True)
        for m in d_sorted[:5]:
            print(f"  {m.action_name:25s}  Δ(F·d̂) = {m.max_delta_force_proj:+.2e} N")
    
    # Category comparison
    print(f"\n--- CATEGORY COMPARISON ---")
    
    position_actions = [m for m in metrics if any(x in m.action_name for x in ["TRANSLATE", "MOVE_"]) 
                       and "OFF" not in m.action_name]
    phase_actions = [m for m in metrics if "PHASE" in m.action_name or "ROTATE" in m.action_name]
    toggle_actions = [m for m in metrics if "TOGGLE" in m.action_name]
    
    if position_actions:
        avg_pos = np.mean([abs(m.max_delta_force_proj) for m in position_actions])
        print(f"  Position actions (TRANSLATE, MOVE): avg Δ(F·d̂) = {avg_pos:.2e} N")
    
    if phase_actions:
        avg_phase = np.mean([abs(m.max_delta_force_proj) for m in phase_actions])
        print(f"  Phase actions (PHASE, ROTATE): avg Δ(F·d̂) = {avg_phase:.2e} N")
    
    if toggle_actions:
        avg_toggle = np.mean([abs(m.max_delta_force_proj) for m in toggle_actions])
        print(f"  Toggle actions (ON/OFF): avg Δ(F·d̂) = {avg_toggle:.2e} N")
    
    print("\n" + "=" * 70)


# =============================================================================
# Main entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Control authority diagnostic for 4-puck system")
    
    parser.add_argument("--fast", action="store_true", help="Use coarse grid")
    parser.add_argument("--particle_x", type=float, default=1.0, help="Particle x position in mm")
    parser.add_argument("--particle_y", type=float, default=1.0, help="Particle y position in mm")
    parser.add_argument("--macro_step_pos_um", type=float, default=50, help="Position step in µm")
    parser.add_argument("--macro_step_phase_rad", type=float, default=0.15, help="Phase step in rad")
    parser.add_argument("--macro_amplitude_step", type=float, default=0.01, help="Amplitude step")
    parser.add_argument("--directions", type=int, default=8, help="Number of test directions")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top actions to highlight")
    
    parser.add_argument(
        "--action_subset",
        choices=["all", "standard", "3puck_compatible", "gating_only", "move_off_only"],
        default="standard",
        help="Which action subset to evaluate"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("CONTROL AUTHORITY DIAGNOSTIC - 4-PUCK SYSTEM")
    print("=" * 70)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path("results/control_authority_4puck") / f"run_{timestamp}"
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {outdir}")
    
    # Domain + Physics
    if args.fast:
        domain = DishDomain(Lx=2e-3, Ly=2e-3, Nx=80, Ny=80)
    else:
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
    
    ev = Evaluator4Pucks(domain, medium, particle, cfg)
    
    print(f"\nPhysics:")
    print(f"  Domain: {domain.Lx*1e3:.1f} x {domain.Ly*1e3:.1f} mm")
    print(f"  Grid: {domain.Nx} x {domain.Ny}")
    print(f"  alpha_g: {cfg.alpha_g:.0e}")
    
    # Baseline control (4-puck spread configuration)
    particle_x = args.particle_x * 1e-3
    particle_y = args.particle_y * 1e-3
    
    ctrl_baseline = Control4Pucks(
        xA=particle_x - 0.4e-3, yA=0.03e-3, vA=0.08, phiA=0.0, gateA=True,
        xB=particle_x + 0.4e-3, yB=0.03e-3, vB=0.08, phiB=np.pi, gateB=True,
        xC=particle_x - 0.2e-3, yC=0.20e-3, vC=0.08, phiC=np.pi/4, gateC=True,
        xD=particle_x + 0.2e-3, yD=0.20e-3, vD=0.08, phiD=-np.pi/4, gateD=True,
    )
    ctrl_baseline = ev.clip_control(ctrl_baseline)
    
    print(f"\nBaseline state:")
    print(f"  Particle: ({particle_x*1e3:.2f}, {particle_y*1e3:.2f}) mm")
    print(f"  Transducers:")
    print(f"    A: ({ctrl_baseline.xA*1e3:.2f}, {ctrl_baseline.yA*1e3:.2f}) mm, gate={ctrl_baseline.gateA}")
    print(f"    B: ({ctrl_baseline.xB*1e3:.2f}, {ctrl_baseline.yB*1e3:.2f}) mm, gate={ctrl_baseline.gateB}")
    print(f"    C: ({ctrl_baseline.xC*1e3:.2f}, {ctrl_baseline.yC*1e3:.2f}) mm, gate={ctrl_baseline.gateC}")
    print(f"    D: ({ctrl_baseline.xD*1e3:.2f}, {ctrl_baseline.yD*1e3:.2f}) mm, gate={ctrl_baseline.gateD}")
    
    # Compute baseline
    print("\nComputing baseline field...")
    t0 = time.perf_counter()
    baseline = compute_baseline(ev, particle_x, particle_y, ctrl_baseline)
    baseline_time = (time.perf_counter() - t0) * 1000.0
    print(f"  Baseline solve: {baseline_time:.1f} ms")
    print(f"  Force at particle: {baseline.force_mag_at_particle:.2e} N")
    print(f"  Active transducers: {baseline.active_transducers}/4")
    
    # Test directions
    n_dirs = args.directions
    angles = np.linspace(0, 2 * np.pi, n_dirs, endpoint=False)
    test_directions = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Action set
    if args.action_subset == "all":
        action_types = get_all_actions_4puck()
    elif args.action_subset == "standard":
        action_types = get_standard_actions_4puck()
    elif args.action_subset == "3puck_compatible":
        action_types = get_3puck_compatible_actions_4puck()
    elif args.action_subset == "gating_only":
        action_types = get_gating_actions_4puck()
    elif args.action_subset == "move_off_only":
        action_types = get_move_off_actions_4puck()
    else:
        action_types = get_standard_actions_4puck()
    
    print(f"\nAction subset: {args.action_subset} ({len(action_types)} actions)")
    
    # Macro action parameters
    macro_magnitude = args.macro_step_pos_um * 1e-6
    macro_phase_step = args.macro_step_phase_rad
    macro_amplitude_step = args.macro_amplitude_step
    
    # Evaluate all actions
    print(f"\nEvaluating actions...")
    metrics: list[ActionMetrics] = []
    
    for i, action_type in enumerate(action_types):
        t0 = time.perf_counter()
        m = evaluate_action(
            ev, baseline, action_type,
            macro_magnitude, macro_phase_step, macro_amplitude_step,
            test_directions,
        )
        metrics.append(m)
        elapsed = (time.perf_counter() - t0) * 1000.0
        gates_str = f"{m.active_transducers_before}→{m.active_transducers_after}"
        print(f"  [{i+1:2d}/{len(action_types)}] {action_type.name:35s}  "
              f"Δ|F|={m.delta_force_mag:+.2e} N  gates:{gates_str}  ({elapsed:.0f} ms)")
    
    # Write outputs
    print(f"\nWriting outputs...")
    
    csv_path = outdir / "control_authority.csv"
    write_csv(metrics, csv_path)
    print(f"  CSV: {csv_path}")
    
    bars_path = outdir / "summary_bars.png"
    plot_summary_bars(metrics, baseline, bars_path)
    print(f"  Bar charts: {bars_path}")
    
    # Print summary
    print_summary(metrics, baseline, top_k=args.top_k)
    
    print(f"\nDone. Results saved to: {outdir}")


if __name__ == "__main__":
    main()
