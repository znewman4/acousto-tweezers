#!/usr/bin/env python3
"""
Plot Control Diagnostics: Analyze flight recorder data.

Reads steps.csv from a flight recorder run and generates diagnostic plots:
- Tracking error vs time
- Trap→target distance vs time  
- Particle→trap distance vs time
- Stiffness (stiff_min) vs time
- Control delta norm (Δu) vs time
- Control jitter (ΔΔu) vs time
- Candidate cost statistics vs time
- Field metrics (p_max, U_ptp) vs time to correlate with flat frames

Usage:
    python scripts/plot_control_diagnostics.py results/demo_large_circle/run_20250114_123456
    python scripts/plot_control_diagnostics.py --latest  # Most recent run
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def find_latest_run(base_dir: Path) -> Path:
    """Find the most recent run_* directory."""
    runs = sorted(base_dir.glob("run_*"))
    if not runs:
        raise FileNotFoundError(f"No run_* directories found in {base_dir}")
    return runs[-1]


def load_steps_csv(run_dir: Path) -> pd.DataFrame:
    """Load steps.csv with proper handling of NaN values."""
    csv_path = run_dir / "steps.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No steps.csv found in {run_dir}")
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} steps from {csv_path}")
    return df


def plot_tracking(ax, df: pd.DataFrame):
    """Plot tracking error over time."""
    steps = df["step_idx"]
    err = df["tracking_error"] * 1e3  # Convert to mm if in meters
    
    # Check if already in mm (values > 1e-3 suggest mm)
    if err.mean() < 1e-3:
        err = err * 1e3
        unit = "mm"
    else:
        unit = "mm"
    
    ax.plot(steps, err, "b-", lw=1.5, label="tracking error")
    ax.axhline(err.mean(), color="r", linestyle="--", lw=1, label=f"mean={err.mean():.4f}")
    ax.fill_between(steps, 0, err, alpha=0.2)
    
    ax.set_xlabel("Step")
    ax.set_ylabel(f"Error ({unit})")
    ax.set_title("Tracking Error")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(steps.min(), steps.max())
    ax.set_ylim(bottom=0)


def plot_trap_target(ax, df: pd.DataFrame):
    """Plot trap→target distance over time."""
    steps = df["step_idx"]
    
    trap_x = df["trap_x"]
    trap_y = df["trap_y"]
    target_x = df["target_x"]
    target_y = df["target_y"]
    
    dist = np.sqrt((trap_x - target_x)**2 + (trap_y - target_y)**2) * 1e3
    
    valid_mask = np.isfinite(dist)
    
    ax.plot(steps[valid_mask], dist[valid_mask], "g-", lw=1.5, label="trap→target")
    
    if valid_mask.sum() > 0:
        ax.axhline(dist[valid_mask].mean(), color="r", linestyle="--", lw=1,
                  label=f"mean={dist[valid_mask].mean():.4f}")
    
    # Mark where trap was not found
    missing = ~valid_mask
    if missing.sum() > 0:
        ax.scatter(steps[missing], np.zeros(missing.sum()), c="red", s=10, 
                  marker="x", label=f"trap missing ({missing.sum()})")
    
    ax.set_xlabel("Step")
    ax.set_ylabel("Distance (mm)")
    ax.set_title("Trap → Target Distance")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_particle_trap(ax, df: pd.DataFrame):
    """Plot particle→trap distance over time."""
    steps = df["step_idx"]
    
    particle_x = df["particle_x"]
    particle_y = df["particle_y"]
    trap_x = df["trap_x"]
    trap_y = df["trap_y"]
    
    dist = np.sqrt((particle_x - trap_x)**2 + (particle_y - trap_y)**2) * 1e3
    
    valid_mask = np.isfinite(dist)
    
    ax.plot(steps[valid_mask], dist[valid_mask], "m-", lw=1.5, label="particle→trap")
    
    if valid_mask.sum() > 0:
        ax.axhline(dist[valid_mask].mean(), color="r", linestyle="--", lw=1,
                  label=f"mean={dist[valid_mask].mean():.4f}")
    
    ax.set_xlabel("Step")
    ax.set_ylabel("Distance (mm)")
    ax.set_title("Particle → Trap Distance")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_stiffness(ax, df: pd.DataFrame):
    """Plot trap stiffness over time."""
    steps = df["step_idx"]
    stiff = df["stiff_min"]
    
    valid_mask = np.isfinite(stiff)
    
    if valid_mask.sum() > 0:
        ax.plot(steps[valid_mask], stiff[valid_mask], "c-", lw=1.5, label="stiff_min")
        ax.axhline(0, color="k", linestyle="-", lw=0.5)
        
        # Mark unstable (positive eigenvalue = saddle)
        unstable = stiff > 0
        if unstable.sum() > 0:
            ax.scatter(steps[unstable & valid_mask], stiff[unstable & valid_mask],
                      c="red", s=20, marker="x", label=f"unstable ({unstable.sum()})")
    
    # Mark missing stiffness
    missing = ~valid_mask
    if missing.sum() > 0:
        ax.axhspan(-0.01, 0.01, alpha=0.1, color="gray", label=f"no trap ({missing.sum()})")
    
    ax.set_xlabel("Step")
    ax.set_ylabel("Min Eigenvalue")
    ax.set_title("Trap Stiffness (stiff_min)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_control_delta(ax, df: pd.DataFrame):
    """Plot control delta norm over time."""
    steps = df["step_idx"]
    delta_u = df["delta_u_norm"]
    
    valid_mask = np.isfinite(delta_u)
    
    if valid_mask.sum() > 0:
        ax.plot(steps[valid_mask], delta_u[valid_mask] * 1e6, "orange", lw=1.5, label="|Δu|")
        ax.axhline(delta_u[valid_mask].mean() * 1e6, color="r", linestyle="--", lw=1,
                  label=f"mean={delta_u[valid_mask].mean()*1e6:.2f}")
    
    ax.set_xlabel("Step")
    ax.set_ylabel("|Δu| (µm-equiv)")
    ax.set_title("Control Rate (|Δu|)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_control_jitter(ax, df: pd.DataFrame):
    """Plot control jitter (ΔΔu) over time."""
    steps = df["step_idx"]
    jitter = df["delta_delta_u_norm"]
    
    valid_mask = np.isfinite(jitter)
    
    if valid_mask.sum() > 0:
        ax.plot(steps[valid_mask], jitter[valid_mask] * 1e6, "purple", lw=1.5, label="|ΔΔu|")
        ax.axhline(jitter[valid_mask].mean() * 1e6, color="r", linestyle="--", lw=1,
                  label=f"mean={jitter[valid_mask].mean()*1e6:.2f}")
    
    ax.set_xlabel("Step")
    ax.set_ylabel("|ΔΔu| (µm-equiv)")
    ax.set_title("Control Jitter (|ΔΔu|)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_candidate_costs(ax, df: pd.DataFrame):
    """Plot MPC candidate cost statistics over time."""
    steps = df["step_idx"]
    
    cost_min = df.get("candidate_cost_min", pd.Series([np.nan] * len(df)))
    cost_mean = df.get("candidate_cost_mean", pd.Series([np.nan] * len(df)))
    cost_std = df.get("candidate_cost_std", pd.Series([np.nan] * len(df)))
    
    valid_mask = np.isfinite(cost_mean)
    
    if valid_mask.sum() > 0:
        ax.fill_between(steps[valid_mask], 
                       (cost_mean - cost_std)[valid_mask],
                       (cost_mean + cost_std)[valid_mask],
                       alpha=0.3, color="blue", label="±1σ")
        ax.plot(steps[valid_mask], cost_mean[valid_mask], "b-", lw=1.5, label="mean cost")
        ax.plot(steps[valid_mask], cost_min[valid_mask], "g--", lw=1, label="min cost")
        
        # Flag if all costs are equal (std ≈ 0)
        degenerate = cost_std < 1e-10
        if degenerate.sum() > 0:
            ax.scatter(steps[degenerate & valid_mask], 
                      cost_mean[degenerate & valid_mask],
                      c="red", s=20, marker="x", label=f"degenerate ({degenerate.sum()})")
    
    ax.set_xlabel("Step")
    ax.set_ylabel("Cost")
    ax.set_title("MPC Candidate Costs")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_field_metrics(ax, df: pd.DataFrame):
    """Plot field metrics (p_max, U_ptp) to correlate with flat frames."""
    steps = df["step_idx"]
    
    # Create twin axis for different scales
    ax2 = ax.twinx()
    
    # U_ptp (potential range)
    U_ptp = df.get("U_ptp", pd.Series([np.nan] * len(df)))
    valid_U = np.isfinite(U_ptp)
    if valid_U.sum() > 0:
        ax.semilogy(steps[valid_U], U_ptp[valid_U], "b-", lw=1.5, label="U_ptp")
    
    # p_max (pressure max)
    p_max = df.get("p_max", pd.Series([np.nan] * len(df)))
    valid_p = np.isfinite(p_max)
    if valid_p.sum() > 0:
        ax2.semilogy(steps[valid_p], p_max[valid_p], "r-", lw=1.5, alpha=0.7, label="p_max")
    
    # Mark flat frames
    flat_flag = df.get("render_flat_flag", pd.Series([False] * len(df)))
    flat_steps = steps[flat_flag == True]
    if len(flat_steps) > 0:
        for s in flat_steps:
            ax.axvline(s, color="gray", alpha=0.3, lw=0.5)
    
    ax.set_xlabel("Step")
    ax.set_ylabel("U_ptp (blue)", color="blue")
    ax2.set_ylabel("p_max (red)", color="red")
    ax.set_title(f"Field Metrics (flat frames: {len(flat_steps)})")
    
    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_control_positions(ax, df: pd.DataFrame):
    """Plot transducer positions over time."""
    steps = df["step_idx"]
    
    xA = df.get("ctrl_xA", pd.Series([np.nan] * len(df))) * 1e3
    xB = df.get("ctrl_xB", pd.Series([np.nan] * len(df))) * 1e3
    xC = df.get("ctrl_xC", pd.Series([np.nan] * len(df))) * 1e3
    
    ax.plot(steps, xA, "orange", lw=1.5, label="xA")
    ax.plot(steps, xB, "blue", lw=1.5, label="xB")
    ax.plot(steps, xC, "magenta", lw=1.5, label="xC")
    
    ax.set_xlabel("Step")
    ax.set_ylabel("x position (mm)")
    ax.set_title("Transducer X Positions")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_summary_trajectory(ax, df: pd.DataFrame):
    """Plot 2D trajectory: desired vs actual."""
    particle_x = df["particle_x"] * 1e3
    particle_y = df["particle_y"] * 1e3
    target_x = df["target_x"] * 1e3
    target_y = df["target_y"] * 1e3
    
    ax.plot(target_x, target_y, "k--", lw=2, label="target path")
    ax.plot(particle_x, particle_y, "b-", lw=1.5, label="actual")
    
    ax.scatter(particle_x.iloc[0], particle_y.iloc[0], s=100, c="green", marker="o", 
              label="start", zorder=10)
    ax.scatter(particle_x.iloc[-1], particle_y.iloc[-1], s=100, c="red", marker="s",
              label="end", zorder=10)
    
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("Trajectory")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)


def generate_diagnostics(run_dir: Path, output_dir: Path = None):
    """Generate all diagnostic plots for a run."""
    
    if output_dir is None:
        output_dir = run_dir
    
    df = load_steps_csv(run_dir)
    
    # Main diagnostics figure (3x3)
    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    
    plot_tracking(axes[0, 0], df)
    plot_trap_target(axes[0, 1], df)
    plot_particle_trap(axes[0, 2], df)
    
    plot_stiffness(axes[1, 0], df)
    plot_control_delta(axes[1, 1], df)
    plot_control_jitter(axes[1, 2], df)
    
    plot_candidate_costs(axes[2, 0], df)
    plot_field_metrics(axes[2, 1], df)
    plot_control_positions(axes[2, 2], df)
    
    fig.suptitle(f"Control Diagnostics: {run_dir.name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    out_path = output_dir / "diagnostics_panel.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")
    
    # Trajectory figure
    fig, ax = plt.subplots(figsize=(10, 10))
    plot_summary_trajectory(ax, df)
    fig.tight_layout()
    
    traj_path = output_dir / "diagnostics_trajectory.png"
    plt.savefig(traj_path, dpi=150)
    plt.close()
    print(f"Saved: {traj_path}")
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)
    
    # Tracking stats
    err = df["tracking_error"]
    valid_err = err[np.isfinite(err)]
    if len(valid_err) > 0:
        print(f"\nTracking Error:")
        print(f"  Mean:  {valid_err.mean():.6f}")
        print(f"  Std:   {valid_err.std():.6f}")
        print(f"  Max:   {valid_err.max():.6f}")
    
    # Trap stats
    trap_found = df.get("trap_found", pd.Series([False] * len(df)))
    trap_stable = df.get("trap_stable", pd.Series([False] * len(df)))
    print(f"\nTrap Detection:")
    print(f"  Found:   {trap_found.sum()}/{len(df)} ({trap_found.mean()*100:.1f}%)")
    print(f"  Stable:  {trap_stable.sum()}/{len(df)} ({trap_stable.mean()*100:.1f}%)")
    
    # Flat frames
    flat_flag = df.get("render_flat_flag", pd.Series([False] * len(df)))
    nan_flag = df.get("render_nan_flag", pd.Series([False] * len(df)))
    print(f"\nField Issues:")
    print(f"  Flat frames: {flat_flag.sum()} ({flat_flag.mean()*100:.1f}%)")
    print(f"  NaN frames:  {nan_flag.sum()} ({nan_flag.mean()*100:.1f}%)")
    
    # Control stats
    delta_u = df.get("delta_u_norm", pd.Series([np.nan] * len(df)))
    jitter = df.get("delta_delta_u_norm", pd.Series([np.nan] * len(df)))
    valid_delta = delta_u[np.isfinite(delta_u)]
    valid_jitter = jitter[np.isfinite(jitter)]
    
    if len(valid_delta) > 0:
        print(f"\nControl Smoothness:")
        print(f"  Mean |Δu|:   {valid_delta.mean()*1e6:.2f} µm-equiv")
        print(f"  Mean |ΔΔu|:  {valid_jitter.mean()*1e6:.2f} µm-equiv")
    
    # Rate limiting
    rate_limited = df.get("rate_limited", pd.Series([False] * len(df)))
    max_step_clipped = df.get("max_step_clipped", pd.Series([False] * len(df)))
    print(f"\nClipping:")
    print(f"  Rate limited:    {rate_limited.sum()} ({rate_limited.mean()*100:.1f}%)")
    print(f"  Max step clip:   {max_step_clipped.sum()} ({max_step_clipped.mean()*100:.1f}%)")
    
    # Mode distribution
    modes = df.get("control_mode", pd.Series([""] * len(df)))
    mode_counts = modes.value_counts()
    if len(mode_counts) > 0:
        print(f"\nControl Modes:")
        for mode, count in mode_counts.items():
            if mode:
                print(f"  {mode}: {count} ({count/len(df)*100:.1f}%)")
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Plot control diagnostics from flight recorder data")
    parser.add_argument("run_dir", nargs="?", help="Path to run directory with steps.csv")
    parser.add_argument("--latest", action="store_true", help="Use most recent run")
    parser.add_argument("--base", type=str, default="results/demo_large_circle",
                       help="Base directory to search for runs")
    args = parser.parse_args()
    
    if args.latest:
        run_dir = find_latest_run(Path(args.base))
    elif args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        # Try to find latest in default location
        try:
            run_dir = find_latest_run(Path(args.base))
        except FileNotFoundError:
            # Fall back to base directory if it has steps.csv
            run_dir = Path(args.base)
    
    if not run_dir.exists():
        print(f"Error: {run_dir} does not exist")
        sys.exit(1)
    
    if not (run_dir / "steps.csv").exists():
        print(f"Error: No steps.csv found in {run_dir}")
        print("Run demo_large_circle.py with --record first")
        sys.exit(1)
    
    generate_diagnostics(run_dir)


if __name__ == "__main__":
    main()
