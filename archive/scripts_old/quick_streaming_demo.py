#!/usr/bin/env python3
"""
Quick streaming solver test with diagnostics and visualization.
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dolfinx import mesh
from mpi4py import MPI
from acoustweezers.experiments.shallow_square_dish.config import ShallowDishConfig
from acoustweezers.experiments.shallow_square_dish.solve_pressure import (
    create_mesh, solve_helmholtz,
)
from acoustweezers.experiments.shallow_square_dish.streaming import (
    solve_streaming,
)

def main():
    """Run quick streaming demo."""
    
    # Create output directory
    out_dir = Path("results/quick_streaming_demo")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Configuration
    cfg = ShallowDishConfig(
        domain_lxy=0.01,  # 1 cm square
        domain_h=0.001,   # 1 mm deep
        freq_hz=500_000,  # 500 kHz
        vortex_amp=10e-6,  # 10 μm/s
        vortex_charge=1,
        vortex_aperture=0.002,  # 2 mm radius (smaller for speed)
    )
    
    print("="*70)
    print("QUICK STREAMING DEMO")
    print("="*70)
    print(f"Domain: {cfg.domain_lxy*1000:.1f} mm × {cfg.domain_h*1000:.1f} mm")
    print(f"Mesh density: {cfg.mesh_elements_per_wavelength} elem/λ")
    
    # Create mesh
    print("\nCreating mesh...")
    domain = create_mesh(cfg)
    print(f"  ✓ Mesh: {domain.topology.index_map(3).size_global:,} cells, "
          f"{domain.topology.index_map(0).size_global:,} vertices")
    
    # Solve pressure
    print("\nSolving pressure field...")
    p_sol = solve_helmholtz(domain, cfg, verbose=False)
    print(f"  ✓ max|p| = {np.max(np.abs(p_sol.p_vals)):.2f} Pa")
    
    # Solve streaming
    print("\nSolving acoustic streaming...")
    streaming = solve_streaming(
        p_solution=p_sol,
        domain=domain,
        cfg=cfg,
        downsample=1,
        forcing_scale=1.0,
        verbose=True,
    )
    
    # Extract diagnostics
    diags = streaming.get("diagnostics", {})
    u_h = streaming.get("u_h")
    
    print("\n" + "="*70)
    print("DIAGNOSTICS")
    print("="*70)
    
    # Save diagnostics to JSON
    diags_file = out_dir / "streaming_diagnostics.json"
    with open(diags_file, "w") as f:
        # Convert numpy types to native Python types for JSON
        diags_json = {}
        for k, v in diags.items():
            if isinstance(v, (np.ndarray, np.integer, np.floating)):
                diags_json[k] = float(v) if np.isscalar(v) else v.tolist()
            else:
                diags_json[k] = v
        json.dump(diags_json, f, indent=2)
    
    print(f"\nSaved diagnostics to: {diags_file}")
    
    # Extract velocity data
    if u_h is not None:
        u_vals = u_h.x.array
        n_udofs = len(u_vals) // 3
        u_vec = u_vals.reshape((n_udofs, 3))
        u_mag = np.linalg.norm(u_vec, axis=1)
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Velocity magnitude histogram
        axes[0, 0].hist(u_mag * 1e6, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('Velocity magnitude (μm/s)')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Streaming Velocity Distribution')
        axes[0, 0].grid(alpha=0.3)
        
        # 2. Velocity components
        axes[0, 1].hist([u_vec[:, 0]*1e6, u_vec[:, 1]*1e6, u_vec[:, 2]*1e6],
                       label=['u_x', 'u_y', 'u_z'], bins=30, alpha=0.6)
        axes[0, 1].set_xlabel('Velocity (μm/s)')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Velocity Components')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # 3. Convergence (KSP iterations)
        ksp_iter = diags.get('ksp_iterations', 0)
        ksp_reason = diags.get('ksp_reason_str', 'unknown')
        axes[1, 0].text(0.5, 0.5, f'KSP Convergence\n\nIterations: {ksp_iter}\n'
                                  f'Reason: {ksp_reason}',
                       ha='center', va='center', fontsize=14,
                       bbox=dict(boxstyle='round', facecolor='lightblue'))
        axes[1, 0].axis('off')
        
        # 4. Statistics box
        stats_text = (f"Streaming Statistics\n\n"
                     f"max|u| = {diags.get('max_u_um_s', 0):.2f} μm/s\n"
                     f"mean|u| = {diags.get('mean_u_um_s', 0):.2f} μm/s\n"
                     f"median|u| = {diags.get('median_u_um_s', 0):.2f} μm/s\n"
                     f"∇·u (L2) = {diags.get('divergence_l2_norm', 0):.2e}\n"
                     f"∇·u (rel) = {diags.get('divergence_l2_norm_relative', 0):.2e}\n")
        axes[1, 1].text(0.5, 0.5, stats_text,
                       ha='center', va='center', fontsize=12, family='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightgreen'))
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plot_file = out_dir / "streaming_diagnostics.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {plot_file}")
        plt.close()
    
    print("\n" + "="*70)
    print("✓ QUICK DEMO COMPLETE")
    print("="*70)
    print(f"Output directory: {out_dir}")

if __name__ == "__main__":
    main()
