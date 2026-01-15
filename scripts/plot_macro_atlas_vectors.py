#!/usr/bin/env python3
"""
Plot macro atlas vectors: trap displacement and multi-probe surf force vectors.

Supports the multi-probe surf atlas format where each (control config, action)
produces K rows for K probe points across the domain.

Usage:
    python scripts/plot_macro_atlas_vectors.py results/reachability_3puck/macro_action_atlas.csv
    python scripts/plot_macro_atlas_vectors.py results/reachability_3puck/macro_action_atlas.csv --stable-only
    python scripts/plot_macro_atlas_vectors.py results/reachability_3puck/macro_action_atlas.csv --action TRANSLATE_TRAP_X_POS
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


def plot_trap_displacement_vectors(df: pd.DataFrame, output_dir: Path, title_suffix: str = ""):
    """
    Plot 1: Trap displacement vectors.
    
    Scatter init_trap position with quiver arrows for (delta_trap_x, delta_trap_y).
    Color by final_stable.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Use init_trap_x/y if available, else fall back to init_proxy_trap
    if "init_trap_x" in df.columns:
        x = df["init_trap_x"].values * 1e3  # mm
        y = df["init_trap_y"].values * 1e3
    else:
        x = df["init_proxy_trap_x"].values * 1e3
        y = df["init_proxy_trap_y"].values * 1e3
    
    dx = df["delta_trap_x"].values * 1e3  # mm
    dy = df["delta_trap_y"].values * 1e3
    
    # Color by final_stable
    if "final_stable" in df.columns:
        colors = df["final_stable"].astype(float).values
        cmap = "RdYlGn"
    elif "final_stiff_min" in df.columns:
        colors = df["final_stiff_min"].values
        cmap = "viridis"
    else:
        colors = np.ones(len(df))
        cmap = "gray"
    
    # Filter valid entries
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(dx) & np.isfinite(dy)
    x, y, dx, dy, colors = x[valid], y[valid], dx[valid], dy[valid], colors[valid]
    
    if len(x) == 0:
        ax.text(0.5, 0.5, "No valid data", ha='center', va='center', transform=ax.transAxes)
    else:
        # Quiver plot
        q = ax.quiver(x, y, dx, dy,
                      colors,
                      cmap=cmap,
                      angles='xy', scale_units='xy', scale=1,
                      alpha=0.7, width=0.003)
        
        # Colorbar
        sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.7)
        cbar.set_label("Final Stable" if "final_stable" in df.columns else "Stiffness")
    
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_aspect('equal')
    ax.set_xlabel("X position (mm)")
    ax.set_ylabel("Y position (mm)")
    ax.set_title(f"Trap Displacement Vectors{title_suffix}\n(arrow = Δtrap after macro action)")
    ax.grid(True, alpha=0.3)
    
    output_path = output_dir / "atlas_trap_displacement.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")
    return output_path


def plot_surf_force_vectors(df: pd.DataFrame, output_dir: Path, title_suffix: str = ""):
    """
    Plot 2: Multi-probe surf force vectors.
    
    For multi-probe atlas: aggregates force across all control configs per probe point,
    showing the mean force direction at each spatial location.
    
    Arrows are normalized (direction only). Color by mean init_Fp_hat_dot_d.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Check for multi-probe format
    if "probe_id" in df.columns:
        n_probes = df["probe_id"].nunique()
        is_multiprobe = n_probes > 1
    else:
        is_multiprobe = False
    
    # Surf particle position
    if "surf_particle_x" in df.columns:
        x = df["surf_particle_x"].values * 1e3
        y = df["surf_particle_y"].values * 1e3
    else:
        x = df["init_particle_x"].values * 1e3 if "init_particle_x" in df.columns else np.full(len(df), 1.0)
        y = df["init_particle_y"].values * 1e3 if "init_particle_y" in df.columns else np.full(len(df), 1.0)
    
    # SURF force components
    if "init_Fp_x" in df.columns:
        fx = df["init_Fp_x"].values
        fy = df["init_Fp_y"].values
    else:
        fx = df["init_Fx_p"].values if "init_Fx_p" in df.columns else np.zeros(len(df))
        fy = df["init_Fy_p"].values if "init_Fy_p" in df.columns else np.zeros(len(df))
    
    # Color by Fp_hat_dot_d
    if "init_Fp_hat_dot_d" in df.columns:
        colors = df["init_Fp_hat_dot_d"].values
    elif "init_Fhat_dot_d" in df.columns:
        colors = df["init_Fhat_dot_d"].values
    else:
        colors = np.zeros(len(df))
    
    # Normalize force to unit vectors
    fmag = np.sqrt(fx**2 + fy**2)
    eps = 1e-15
    fx_norm = fx / (fmag + eps)
    fy_norm = fy / (fmag + eps)
    
    # Filter valid entries
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(fx_norm) & np.isfinite(fy_norm)
    x, y = x[valid], y[valid]
    fx_norm, fy_norm = fx_norm[valid], fy_norm[valid]
    colors = colors[valid]
    
    if len(x) == 0:
        ax.text(0.5, 0.5, "No valid data", ha='center', va='center', transform=ax.transAxes)
    else:
        # Fixed arrow length for direction visualization
        arrow_scale = 0.08  # mm - smaller to avoid overlap
        
        q = ax.quiver(x, y, fx_norm * arrow_scale, fy_norm * arrow_scale,
                      colors,
                      cmap="RdBu",
                      clim=(-1, 1),
                      angles='xy', scale_units='xy', scale=1,
                      alpha=0.6, width=0.003)
        
        # Colorbar
        sm = ScalarMappable(cmap="RdBu", norm=Normalize(vmin=-1, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.7)
        cbar.set_label("F̂·d (force alignment with desired dir)")
    
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_aspect('equal')
    ax.set_xlabel("X position (mm)")
    ax.set_ylabel("Y position (mm)")
    
    if is_multiprobe:
        title = f"Multi-Probe Surf Force Vectors{title_suffix}\n({n_probes} probe points, all configs overlaid)"
    else:
        title = f"Surf Force Vectors (FIXED position){title_suffix}\n(normalized direction, color=alignment with desired)"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    output_path = output_dir / "atlas_surf_forces.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")
    return output_path


def plot_surf_per_action(df: pd.DataFrame, output_dir: Path, title_suffix: str = ""):
    """
    Plot 3: Multi-panel surf force vectors, one per action type.
    
    Shows spatial surf force field for each macro action separately.
    This reveals which actions push particles in which directions at each location.
    """
    if "action_type" not in df.columns:
        print("Warning: action_type column not found, skipping per-action plots")
        return None
    
    action_types = sorted(df["action_type"].unique())
    n_actions = len(action_types)
    
    if n_actions == 0:
        return None
    
    # Create subplot grid
    ncols = 4
    nrows = (n_actions + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)
    axes_flat = axes.flatten()
    
    for idx, action_type in enumerate(action_types):
        ax = axes_flat[idx]
        df_act = df[df["action_type"] == action_type]
        
        # Get surf positions and force
        x = df_act["surf_particle_x"].values * 1e3 if "surf_particle_x" in df.columns else np.full(len(df_act), 1.0)
        y = df_act["surf_particle_y"].values * 1e3 if "surf_particle_y" in df.columns else np.full(len(df_act), 1.0)
        
        if "init_Fp_x" in df.columns:
            fx = df_act["init_Fp_x"].values
            fy = df_act["init_Fp_y"].values
        else:
            fx = df_act["init_Fx_p"].values if "init_Fx_p" in df.columns else np.zeros(len(df_act))
            fy = df_act["init_Fy_p"].values if "init_Fy_p" in df.columns else np.zeros(len(df_act))
        
        # Alignment color
        if "init_Fp_hat_dot_d" in df.columns:
            colors = df_act["init_Fp_hat_dot_d"].values
        else:
            colors = np.zeros(len(df_act))
        
        # Normalize
        fmag = np.sqrt(fx**2 + fy**2)
        eps = 1e-15
        fx_norm = fx / (fmag + eps)
        fy_norm = fy / (fmag + eps)
        
        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(fx_norm) & np.isfinite(fy_norm)
        
        if valid.sum() > 0:
            arrow_scale = 0.12
            ax.quiver(x[valid], y[valid],
                     fx_norm[valid] * arrow_scale, fy_norm[valid] * arrow_scale,
                     colors[valid],
                     cmap="RdBu", clim=(-1, 1),
                     angles='xy', scale_units='xy', scale=1,
                     alpha=0.7, width=0.008)
            
            # Add desired direction arrow (if translate action)
            if "desired_dir_x" in df.columns:
                dx = df_act["desired_dir_x"].iloc[0]
                dy = df_act["desired_dir_y"].iloc[0]
                if dx != 0 or dy != 0:
                    ax.arrow(0.2, 0.2, dx * 0.3, dy * 0.3,
                            head_width=0.08, head_length=0.05,
                            fc='black', ec='black', lw=2, zorder=10)
                    ax.text(0.2, 0.1, "desired", fontsize=7, ha='center')
        
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)
        ax.set_aspect('equal')
        ax.set_title(action_type.replace("_", "\n"), fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
    
    # Hide unused axes
    for idx in range(n_actions, len(axes_flat)):
        axes_flat[idx].axis('off')
    
    plt.suptitle(f"Multi-Probe Surf Force by Action Type{title_suffix}\n(color: alignment F̂·d, black arrow: desired direction)", fontsize=12)
    plt.tight_layout()
    
    output_path = output_dir / "atlas_surf_per_action.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")
    return output_path


def plot_alignment_histogram(df: pd.DataFrame, output_dir: Path, title_suffix: str = ""):
    """
    Plot 4: Histogram of init_Fp_hat_dot_d per action_type.
    
    Shows distribution of force alignment with desired direction.
    """
    if "action_type" not in df.columns or "init_Fp_hat_dot_d" not in df.columns:
        print("Warning: Required columns not found, skipping histogram")
        return None
    
    # Focus on translate actions (they have defined desired directions)
    translate_actions = [a for a in df["action_type"].unique() if a.startswith("TRANSLATE")]
    
    if len(translate_actions) == 0:
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes_flat = axes.flatten()
    
    for idx, action_type in enumerate(translate_actions[:4]):
        ax = axes_flat[idx]
        df_act = df[df["action_type"] == action_type]
        align = df_act["init_Fp_hat_dot_d"].dropna()
        
        if len(align) > 0:
            ax.hist(align, bins=20, range=(-1, 1), color='steelblue', alpha=0.7, edgecolor='black')
            ax.axvline(0, color='gray', linestyle='--', lw=1)
            ax.axvline(align.mean(), color='red', linestyle='-', lw=2, label=f'mean={align.mean():.2f}')
            
            # Annotate percentages
            pct_aligned = 100 * (align > 0).sum() / len(align)
            pct_high = 100 * (align > 0.5).sum() / len(align)
            ax.text(0.95, 0.95, f'aligned (>0): {pct_aligned:.0f}%\nstrong (>0.5): {pct_high:.0f}%',
                   transform=ax.transAxes, ha='right', va='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax.legend(loc='upper left')
        
        ax.set_xlim(-1, 1)
        ax.set_xlabel("F̂·d (alignment)")
        ax.set_ylabel("Count")
        ax.set_title(action_type)
    
    plt.suptitle(f"Surf Force Alignment Distribution{title_suffix}", fontsize=14)
    plt.tight_layout()
    
    output_path = output_dir / "atlas_alignment_histogram.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")
    return output_path


def plot_combined(df: pd.DataFrame, output_dir: Path, title_suffix: str = ""):
    """
    Combined 2-panel plot: trap displacement + surf force vectors.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # === Panel 1: Trap displacement ===
    ax = axes[0]
    x = df["init_trap_x"].values * 1e3 if "init_trap_x" in df.columns else df["init_proxy_trap_x"].values * 1e3
    y = df["init_trap_y"].values * 1e3 if "init_trap_y" in df.columns else df["init_proxy_trap_y"].values * 1e3
    dx = df["delta_trap_x"].values * 1e3
    dy = df["delta_trap_y"].values * 1e3
    
    colors = df["final_stable"].astype(float).values if "final_stable" in df.columns else np.ones(len(df))
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(dx) & np.isfinite(dy)
    
    if valid.sum() > 0:
        ax.quiver(x[valid], y[valid], dx[valid], dy[valid],
                  colors[valid], cmap="RdYlGn",
                  angles='xy', scale_units='xy', scale=1,
                  alpha=0.7, width=0.003)
    
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_aspect('equal')
    ax.set_xlabel("X position (mm)")
    ax.set_ylabel("Y position (mm)")
    ax.set_title("Trap Displacement Vectors")
    ax.grid(True, alpha=0.3)
    
    # === Panel 2: Surf force vectors at FIXED position ===
    ax = axes[1]
    # Use surf_particle position (fixed, not at trap)
    if "surf_particle_x" in df.columns:
        px = df["surf_particle_x"].values * 1e3
        py = df["surf_particle_y"].values * 1e3
    else:
        px = df["init_particle_x"].values * 1e3 if "init_particle_x" in df.columns else x
        py = df["init_particle_y"].values * 1e3 if "init_particle_y" in df.columns else y
    
    # Use Fp_* (force at fixed position)
    if "init_Fp_x" in df.columns:
        fx = df["init_Fp_x"].values
        fy = df["init_Fp_y"].values
    else:
        fx = df["init_Fx_p"].values if "init_Fx_p" in df.columns else np.zeros(len(df))
        fy = df["init_Fy_p"].values if "init_Fy_p" in df.columns else np.zeros(len(df))
    
    fmag = np.sqrt(fx**2 + fy**2)
    eps = 1e-15
    fx_norm = fx / (fmag + eps)
    fy_norm = fy / (fmag + eps)
    
    # Color by Fp_hat_dot_d
    if "init_Fp_hat_dot_d" in df.columns:
        colors2 = df["init_Fp_hat_dot_d"].values
    elif "init_Fhat_dot_d" in df.columns:
        colors2 = df["init_Fhat_dot_d"].values
    else:
        colors2 = np.zeros(len(df))
    valid2 = np.isfinite(px) & np.isfinite(py) & np.isfinite(fx_norm) & np.isfinite(fy_norm)
    
    if valid2.sum() > 0:
        arrow_scale = 0.1
        ax.quiver(px[valid2], py[valid2],
                  fx_norm[valid2] * arrow_scale, fy_norm[valid2] * arrow_scale,
                  colors2[valid2], cmap="RdBu", clim=(-1, 1),
                  angles='xy', scale_units='xy', scale=1,
                  alpha=0.7, width=0.003)
    
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_aspect('equal')
    ax.set_xlabel("X position (mm)")
    ax.set_ylabel("Y position (mm)")
    ax.set_title("Surf Force Vectors (normalized)")
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f"Macro Atlas Vectors{title_suffix}", fontsize=14)
    plt.tight_layout()
    
    output_path = output_dir / "atlas_vectors.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Plot macro atlas vectors (supports multi-probe format)")
    parser.add_argument("csv_path", type=str, help="Path to macro atlas CSV")
    parser.add_argument("--stable-only", action="store_true",
                        help="Filter to init_stable & final_stable entries only")
    parser.add_argument("--action", type=str, default=None,
                        help="Filter to specific action type (e.g., TRANSLATE_TRAP_X_POS)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (defaults to same as CSV)")
    args = parser.parse_args()
    
    # Load CSV
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}")
        return 1
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    
    # Check for multi-probe format
    if "probe_id" in df.columns:
        n_probes = df["probe_id"].nunique()
        n_unique_probe_positions = df[["surf_particle_x", "surf_particle_y"]].drop_duplicates().shape[0]
        print(f"\n*** MULTI-PROBE FORMAT DETECTED ***")
        print(f"  Unique probe IDs: {n_probes}")
        print(f"  Unique probe positions: {n_unique_probe_positions}")
        probe_x_mm = np.sort(df["surf_particle_x"].unique()) * 1e3
        probe_y_mm = np.sort(df["surf_particle_y"].unique()) * 1e3
        print(f"  Probe X (mm): {probe_x_mm}")
        print(f"  Probe Y (mm): {probe_y_mm}")
    else:
        n_probes = 1
        print("(Single-probe/legacy format)")
    
    print(f"\nColumns: {list(df.columns)}")
    
    # Apply filters
    title_suffix = ""
    if args.stable_only:
        if "init_stable" in df.columns and "final_stable" in df.columns:
            df = df[df["init_stable"] & df["final_stable"]]
            title_suffix += " [stable→stable]"
            print(f"After stable-only filter: {len(df)} rows")
        else:
            print("Warning: stable columns not found, skipping filter")
    
    if args.action:
        if "action_type" in df.columns:
            df = df[df["action_type"] == args.action]
            title_suffix += f" [{args.action}]"
            print(f"After action filter: {len(df)} rows")
        else:
            print("Warning: action_type column not found, skipping filter")
    
    if len(df) == 0:
        print("ERROR: No data after filtering!")
        return 1
    
    # Output directory
    output_dir = Path(args.output_dir) if args.output_dir else csv_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    print("\n=== GENERATING PLOTS ===")
    plot_combined(df, output_dir, title_suffix)
    plot_trap_displacement_vectors(df, output_dir, title_suffix)
    plot_surf_force_vectors(df, output_dir, title_suffix)
    
    # Multi-probe specific plots
    if n_probes > 1:
        plot_surf_per_action(df, output_dir, title_suffix)
        plot_alignment_histogram(df, output_dir, title_suffix)
    
    # Print summary statistics
    print("\n" + "="*60)
    print("ATLAS SUMMARY")
    print("="*60)
    print(f"Total rows: {len(df)}")
    
    if "probe_id" in df.columns:
        print(f"Unique probe points: {df[['surf_particle_x','surf_particle_y']].drop_duplicates().shape[0]}")
    
    if "init_stable" in df.columns and "final_stable" in df.columns:
        stable_stable = (df["init_stable"] & df["final_stable"]).sum()
        print(f"Stable→Stable: {stable_stable} ({100*stable_stable/len(df):.1f}%)")
    
    # Report SURF force metrics
    translate_mask = df["action_type"].str.startswith("TRANSLATE") if "action_type" in df.columns else pd.Series([True]*len(df))
    
    if "init_Fp_hat_dot_d" in df.columns:
        fhat = df.loc[translate_mask, "init_Fp_hat_dot_d"]
        fhat_valid = fhat[np.isfinite(fhat)]
        if len(fhat_valid) > 0:
            print(f"\n=== SURF METRICS (multi-probe) ===")
            print(f"Translate actions - init_Fp_hat_dot_d:")
            print(f"  min: {fhat_valid.min():.3f}")
            print(f"  max: {fhat_valid.max():.3f}")
            print(f"  mean: {fhat_valid.mean():.3f}")
            print(f"  >0 (aligned): {(fhat_valid > 0).sum()}/{len(fhat_valid)} ({100*(fhat_valid > 0).sum()/len(fhat_valid):.1f}%)")
            print(f"  >0.5 (strong): {(fhat_valid > 0.5).sum()}/{len(fhat_valid)} ({100*(fhat_valid > 0.5).sum()/len(fhat_valid):.1f}%)")
    
    if "init_Fp_mag" in df.columns:
        fmag = df.loc[translate_mask, "init_Fp_mag"]
        fmag_valid = fmag[np.isfinite(fmag)]
        if len(fmag_valid) > 0:
            print(f"\nForce magnitude at probe points:")
            print(f"  min: {fmag_valid.min():.2e}")
            print(f"  max: {fmag_valid.max():.2e}")
            print(f"  mean: {fmag_valid.mean():.2e}")
    
    # Per-action summary
    if "action_type" in df.columns and "init_Fp_hat_dot_d" in df.columns:
        print("\n--- Per-Action Mean Alignment ---")
        for action in sorted(df["action_type"].unique()):
            df_act = df[df["action_type"] == action]
            fhat = df_act["init_Fp_hat_dot_d"].dropna()
            if len(fhat) > 0:
                print(f"  {action}: mean={fhat.mean():+.3f}, >0.5: {100*(fhat>0.5).sum()/len(fhat):.0f}%")
    
    print("="*60)
    
    return 0


if __name__ == "__main__":
    exit(main())
