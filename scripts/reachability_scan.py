#!/usr/bin/env python3
"""
Reachability Atlas: Map what trap positions are achievable with 2 transducers.

This script scans the control space (xA, xB, Δφ) and for each setting:
1. Computes the acoustic field
2. Finds trap centers (local U minima)
3. Computes trap stiffness (stability)
4. Records reachable (x, y) points

Output:
- Scatter plot of reachable trap centers, colored by stiffness
- Overlay of target trajectory (e.g., circle) to see coverage
- Summary: what fraction of trajectory is reachable

Usage:
    python scripts/reachability_scan.py --output results/reachability/
    python scripts/reachability_scan.py --trajectory circle --output results/reachability/
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from tqdm import tqdm
import json

from acousto.force import ParticleProps, gorkov_potential_and_force_2d
from acousto.analysis import find_trap_center, find_traps_from_force
from tweezers.control import (
    DishDomain, MediumProps, EvaluatorConfig,
    Control2Pucks, BottomFootprint25DEvaluator,
)


@dataclass
class TrapPoint:
    """A reachable trap position with metadata."""
    trap_x: float  # m
    trap_y: float  # m
    stiffness_min: float  # min eigenvalue (negative = stable)
    stiffness_max: float
    is_stable: bool
    control_xA: float
    control_xB: float
    control_yA: float
    control_yB: float
    control_dphi: float  # phiB - phiA


def scan_reachability(
    ev: BottomFootprint25DEvaluator,
    particle: ParticleProps,
    *,
    n_xA: int = 15,
    n_xB: int = 15,
    n_dphi: int = 8,
    y_fixed: float = 0.05e-3,
    v_amp: float = 0.05,
    verbose: bool = True,
) -> list[TrapPoint]:
    """
    Scan control space and find reachable trap positions.
    
    Parameters
    ----------
    ev : BottomFootprint25DEvaluator
        Evaluator for field computation
    particle : ParticleProps
        Particle properties for Gor'kov computation
    n_xA, n_xB : int
        Number of scan points for transducer x-positions
    n_dphi : int
        Number of phase difference values to scan
    y_fixed : float
        Fixed y-position for transducers
    v_amp : float
        Transducer velocity amplitude
    verbose : bool
        Print progress
    
    Returns
    -------
    List of TrapPoint describing reachable positions
    """
    Lx = ev.domain.Lx
    Ly = ev.domain.Ly
    
    # Scan ranges
    margin = 0.1e-3
    xA_values = np.linspace(margin, Lx/2 - margin, n_xA)
    xB_values = np.linspace(Lx/2 + margin, Lx - margin, n_xB)
    dphi_values = np.linspace(0, 2*np.pi, n_dphi, endpoint=False)
    
    results: list[TrapPoint] = []
    
    total = n_xA * n_xB * n_dphi
    iterator = tqdm(total=total, desc="Scanning") if verbose else range(total)
    
    for xA in xA_values:
        for xB in xB_values:
            for dphi in dphi_values:
                if verbose:
                    iterator.update(1)
                
                # Skip if transducers too close
                if abs(xB - xA) < 0.15e-3:
                    continue
                
                u = Control2Pucks(
                    xA=float(xA), yA=y_fixed,
                    xB=float(xB), yB=y_fixed,
                    vA=v_amp, vB=v_amp,
                    phiA=0.0, phiB=float(dphi),
                )
                
                # Compute field
                vb_x = ev.control_to_forcing_band_vb(u)
                field = ev.op.solve_for_bottom_vb(vb_x)
                U, Fx, Fy = gorkov_potential_and_force_2d(field, particle)
                
                # Find traps (local minima)
                # Search in interior region
                search_x = (xA + xB) / 2
                search_y = Ly / 2
                
                try:
                    trap_result = find_trap_center(
                        field.x, field.y, U, Fx, Fy,
                        particle_x=search_x, particle_y=search_y,
                        search_radius=0.6e-3,
                    )
                    
                    # Skip if trap is at boundary
                    if trap_result.x < margin or trap_result.x > Lx - margin:
                        continue
                    if trap_result.y < margin or trap_result.y > Ly - margin:
                        continue
                    
                    eigvals = trap_result.stiffness_eigvals
                    stiff_min = float(np.min(eigvals))
                    stiff_max = float(np.max(eigvals))
                    
                    results.append(TrapPoint(
                        trap_x=trap_result.x,
                        trap_y=trap_result.y,
                        stiffness_min=stiff_min,
                        stiffness_max=stiff_max,
                        is_stable=trap_result.is_stable,
                        control_xA=xA,
                        control_xB=xB,
                        control_yA=y_fixed,
                        control_yB=y_fixed,
                        control_dphi=dphi,
                    ))
                except Exception:
                    continue
    
    if verbose:
        iterator.close()
    
    return results


def generate_target_trajectory(
    trajectory_type: str,
    Lx: float,
    Ly: float,
    n_points: int = 100,
) -> np.ndarray:
    """Generate a target trajectory for reachability comparison."""
    if trajectory_type == "circle":
        radius = 0.35 * min(Lx, Ly)
        center_x = 0.5 * Lx
        center_y = 0.55 * Ly
        theta = np.linspace(0, 2*np.pi, n_points)
        x = center_x + radius * np.cos(theta)
        y = center_y + radius * np.sin(theta)
        return np.column_stack([x, y])
    
    elif trajectory_type == "sweep_x":
        x = np.linspace(0.3 * Lx, 0.7 * Lx, n_points)
        y = np.full(n_points, 0.55 * Ly)
        return np.column_stack([x, y])
    
    elif trajectory_type == "sweep_y":
        x = np.full(n_points, 0.5 * Lx)
        y = np.linspace(0.3 * Ly, 0.7 * Ly, n_points)
        return np.column_stack([x, y])
    
    else:
        raise ValueError(f"Unknown trajectory type: {trajectory_type}")


def compute_reachability_fraction(
    trap_points: list[TrapPoint],
    trajectory: np.ndarray,
    tolerance: float = 0.1e-3,
    require_stable: bool = False,
) -> tuple[float, np.ndarray]:
    """
    Compute what fraction of trajectory is reachable.
    
    Parameters
    ----------
    trap_points : list[TrapPoint]
        All found trap positions
    trajectory : np.ndarray
        Shape (N, 2) target trajectory in meters
    tolerance : float
        Distance threshold for "reachable" (m)
    require_stable : bool
        If True, only count stable traps
    
    Returns
    -------
    fraction : float
        Fraction of trajectory points that are reachable
    reachable_mask : np.ndarray
        Boolean mask of shape (N,)
    """
    if require_stable:
        trap_positions = np.array([
            [tp.trap_x, tp.trap_y] for tp in trap_points if tp.is_stable
        ])
    else:
        trap_positions = np.array([
            [tp.trap_x, tp.trap_y] for tp in trap_points
        ])
    
    if len(trap_positions) == 0:
        return 0.0, np.zeros(len(trajectory), dtype=bool)
    
    reachable_mask = np.zeros(len(trajectory), dtype=bool)
    
    for i, (tx, ty) in enumerate(trajectory):
        distances = np.sqrt((trap_positions[:, 0] - tx)**2 + 
                           (trap_positions[:, 1] - ty)**2)
        if np.min(distances) < tolerance:
            reachable_mask[i] = True
    
    fraction = np.mean(reachable_mask)
    return fraction, reachable_mask


def plot_reachability_atlas(
    trap_points: list[TrapPoint],
    trajectory: Optional[np.ndarray],
    output_path: Path,
    domain_Lx: float,
    domain_Ly: float,
):
    """
    Create visualization of reachability atlas.
    
    Plots:
    1. Scatter of all trap positions, colored by min stiffness
    2. Target trajectory overlay
    3. Reachable/unreachable portions of trajectory
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # === Panel 1: All trap positions colored by stiffness ===
    ax1 = axes[0]
    
    trap_x = np.array([tp.trap_x for tp in trap_points]) * 1e3
    trap_y = np.array([tp.trap_y for tp in trap_points]) * 1e3
    stiffness = np.array([tp.stiffness_min for tp in trap_points])
    is_stable = np.array([tp.is_stable for tp in trap_points])
    
    # Color by stiffness magnitude (log scale)
    stiff_mag = np.abs(stiffness) + 1e-20
    colors = np.log10(stiff_mag)
    
    sc = ax1.scatter(trap_x, trap_y, c=colors, s=15, alpha=0.5, cmap='viridis')
    plt.colorbar(sc, ax=ax1, label='log₁₀|stiffness|')
    
    # Mark stable traps
    stable_x = trap_x[is_stable]
    stable_y = trap_y[is_stable]
    ax1.scatter(stable_x, stable_y, s=30, marker='x', c='red', 
                alpha=0.7, label='stable')
    
    # Overlay trajectory if provided
    if trajectory is not None:
        traj_x = trajectory[:, 0] * 1e3
        traj_y = trajectory[:, 1] * 1e3
        ax1.plot(traj_x, traj_y, 'w-', linewidth=2, label='target trajectory')
        ax1.plot(traj_x, traj_y, 'k--', linewidth=1)
    
    ax1.set_xlim(0, domain_Lx * 1e3)
    ax1.set_ylim(0, domain_Ly * 1e3)
    ax1.set_xlabel('x (mm)')
    ax1.set_ylabel('y (mm)')
    ax1.set_title('Reachability Atlas: Trap Positions')
    ax1.set_aspect('equal')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # === Panel 2: Reachability along trajectory ===
    ax2 = axes[1]
    
    if trajectory is not None:
        # Compute reachability
        fraction, reachable_mask = compute_reachability_fraction(
            trap_points, trajectory, tolerance=0.1e-3, require_stable=False
        )
        fraction_stable, reachable_stable = compute_reachability_fraction(
            trap_points, trajectory, tolerance=0.1e-3, require_stable=True
        )
        
        traj_x = trajectory[:, 0] * 1e3
        traj_y = trajectory[:, 1] * 1e3
        
        # Color trajectory by reachability
        for i in range(len(trajectory) - 1):
            color = 'green' if reachable_mask[i] else 'red'
            ax2.plot(traj_x[i:i+2], traj_y[i:i+2], color=color, linewidth=3)
        
        ax2.set_title(f'Trajectory Coverage\n'
                      f'Any trap: {fraction*100:.1f}% | '
                      f'Stable: {fraction_stable*100:.1f}%')
    else:
        ax2.text(0.5, 0.5, 'No trajectory specified', 
                 transform=ax2.transAxes, ha='center')
        ax2.set_title('Trajectory Coverage')
    
    ax2.set_xlim(0, domain_Lx * 1e3)
    ax2.set_ylim(0, domain_Ly * 1e3)
    ax2.set_xlabel('x (mm)')
    ax2.set_ylabel('y (mm)')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / "reachability_atlas.png", dpi=150)
    plt.close()
    print(f"Saved: {output_path / 'reachability_atlas.png'}")


def plot_trap_x_vs_control(
    trap_points: list[TrapPoint],
    output_path: Path,
):
    """
    Plot how trap x-position varies with transducer positions.
    
    This helps understand the mapping from control to trap position.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    trap_x = np.array([tp.trap_x for tp in trap_points]) * 1e3
    trap_y = np.array([tp.trap_y for tp in trap_points]) * 1e3
    xA = np.array([tp.control_xA for tp in trap_points]) * 1e3
    xB = np.array([tp.control_xB for tp in trap_points]) * 1e3
    dphi = np.array([tp.control_dphi for tp in trap_points])
    
    # Transducer center vs trap x
    ax = axes[0]
    center_x = (xA + xB) / 2
    ax.scatter(center_x, trap_x, c=dphi, s=10, alpha=0.3, cmap='hsv')
    ax.plot([0, 2], [0, 2], 'k--', alpha=0.5, label='ideal (trap=center)')
    ax.set_xlabel('Transducer center x (mm)')
    ax.set_ylabel('Trap x (mm)')
    ax.set_title('Trap X vs Transducer Center')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Phase difference vs trap y
    ax = axes[1]
    sc = ax.scatter(dphi, trap_y, c=center_x, s=10, alpha=0.3, cmap='viridis')
    plt.colorbar(sc, ax=ax, label='trans center x (mm)')
    ax.set_xlabel('Phase difference φB - φA (rad)')
    ax.set_ylabel('Trap y (mm)')
    ax.set_title('Trap Y vs Phase Difference')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / "trap_vs_control.png", dpi=150)
    plt.close()
    print(f"Saved: {output_path / 'trap_vs_control.png'}")


def main():
    parser = argparse.ArgumentParser(description="Reachability Atlas Scanner")
    parser.add_argument("--output", type=str, default="results/reachability",
                        help="Output directory")
    parser.add_argument("--trajectory", type=str, default="circle",
                        choices=["circle", "sweep_x", "sweep_y", "none"],
                        help="Target trajectory for comparison")
    parser.add_argument("--n_xA", type=int, default=15, 
                        help="Number of xA scan points")
    parser.add_argument("--n_xB", type=int, default=15,
                        help="Number of xB scan points")
    parser.add_argument("--n_dphi", type=int, default=8,
                        help="Number of phase difference scan points")
    parser.add_argument("--coarse", action="store_true",
                        help="Use coarse scan (fast but less accurate)")
    args = parser.parse_args()
    
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Coarse scan overrides
    if args.coarse:
        args.n_xA = 10
        args.n_xB = 10
        args.n_dphi = 4
    
    print("=" * 60)
    print("REACHABILITY ATLAS SCANNER")
    print("=" * 60)
    
    # Setup
    domain = DishDomain(Lx=2e-3, Ly=2e-3, Nx=120, Ny=120)
    medium = MediumProps(f=2e6, c0=1500.0, rho0=1000.0, loss_eta=1e-3, kz=0.0, coupling_alpha=1.0)
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
    
    print(f"Domain: {domain.Lx*1e3:.1f} x {domain.Ly*1e3:.1f} mm")
    print(f"Scan grid: {args.n_xA} x {args.n_xB} x {args.n_dphi} = "
          f"{args.n_xA * args.n_xB * args.n_dphi} control points")
    print()
    
    # Run scan
    trap_points = scan_reachability(
        ev, particle,
        n_xA=args.n_xA,
        n_xB=args.n_xB,
        n_dphi=args.n_dphi,
    )
    
    print(f"\nFound {len(trap_points)} trap positions")
    n_stable = sum(1 for tp in trap_points if tp.is_stable)
    print(f"  Stable: {n_stable} ({100*n_stable/max(1,len(trap_points)):.1f}%)")
    
    # Generate trajectory
    trajectory = None
    if args.trajectory != "none":
        trajectory = generate_target_trajectory(
            args.trajectory, domain.Lx, domain.Ly
        )
        
        # Compute reachability
        frac, mask = compute_reachability_fraction(
            trap_points, trajectory, tolerance=0.1e-3
        )
        frac_stable, _ = compute_reachability_fraction(
            trap_points, trajectory, tolerance=0.1e-3, require_stable=True
        )
        
        print(f"\nTrajectory: {args.trajectory}")
        print(f"  Reachability (any trap): {100*frac:.1f}%")
        print(f"  Reachability (stable):   {100*frac_stable:.1f}%")
    
    # Save results
    results_data = {
        "n_traps": len(trap_points),
        "n_stable": n_stable,
        "trajectory_type": args.trajectory,
        "scan_params": {
            "n_xA": args.n_xA,
            "n_xB": args.n_xB,
            "n_dphi": args.n_dphi,
        },
    }
    if trajectory is not None:
        results_data["reachability_any"] = float(frac)
        results_data["reachability_stable"] = float(frac_stable)
    
    with open(output_path / "summary.json", "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"Saved: {output_path / 'summary.json'}")
    
    # Create plots
    plot_reachability_atlas(trap_points, trajectory, output_path, 
                           domain.Lx, domain.Ly)
    plot_trap_x_vs_control(trap_points, output_path)
    
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
