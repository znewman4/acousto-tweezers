#!/usr/bin/env python3
"""
Export deposition experiment results to XDMF (ParaView-friendly format).

Re-runs minimal simulation to regenerate functions, then exports to XDMF.

Usage:
    micromamba run -n acousto-complex python scripts/visualization/export_deposition_xdmf.py \
        --run_dir results/deposition_20260209_194235 \
        --output_dir results/deposition_20260209_194235/paraview

Author: Acousto-Tweezers Project
Date: 2026-02-09
"""

import argparse
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2] / 'src'))

from mpi4py import MPI
from dolfinx import io, fem

from acoustweezers.experiments.shallow_square_dish.config import ShallowDishConfig
from acoustweezers.experiments.shallow_square_dish.solve_pressure import solve_helmholtz, create_mesh
from acoustweezers.experiments.shallow_square_dish.streaming import solve_streaming
from acoustweezers.experiments.shallow_square_dish.particles import compute_gorkov_potential


def export_deposition_xdmf(run_dir: Path, output_dir: Path):
    """Export deposition results to XDMF format."""
    
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    # Use same configuration as deposition experiment
    cfg = ShallowDishConfig(
        L=0.01,
        H=0.001,
        frequency_hz=500_000,
        standing_velocity_amplitude=1.0e-6,
        vortex_velocity_amplitude=2.0e-6,
        vortex_topological_charge=1,
    )
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if rank == 0:
        print("="*70)
        print("EXPORTING DEPOSITION RESULTS TO XDMF (PARAVIEW FORMAT)")
        print("="*70)
        print(f"  Output: {output_dir}")
    
    # Create mesh (same as deposition experiment)
    if rank == 0:
        print("\nCreating mesh...")
    domain, tags, _ = create_mesh(cfg, verbose=(rank==0))
    
    # Solve for combined mode (same as phase 2)
    if rank == 0:
        print("\nSolving Helmholtz (combined mode)...")
    p_sol = solve_helmholtz(domain, tags, cfg, mode="combined", verbose=(rank==0))
    
    # Solve streaming
    if rank == 0:
        print("\nSolving streaming...")
    stream_result = solve_streaming(p_sol, domain=domain, cfg=cfg, verbose=(rank==0))
    
    # Compute Gorkov potential
    if rank == 0:
        print("\nComputing Gorkov potential...")
    gorkov = compute_gorkov_potential(p_sol, verbose=(rank==0))
    
    # Extract the streaming solution
    stream_sol = stream_result.get("streaming_solution")
    
    if rank == 0:
        print("\n" + "="*70)
        print("EXPORTING XDMF FILES")
        print("="*70)
    
    # Create a real-valued function space for export
    from dolfinx.fem import functionspace
    from basix.ufl import element
    cg1 = element("Lagrange", domain.topology.cell_name(), 1)
    V_scalar = functionspace(domain, cg1)
    
    # Export pressure magnitude
    p_func = p_sol.p_function
    p_vals = p_func.x.array
    
    p_mag_func = fem.Function(V_scalar, name="pressure_magnitude")
    p_mag_func.x.array[:] = np.abs(p_vals[:len(p_mag_func.x.array)])
    
    with io.XDMFFile(comm, output_dir / "pressure_magnitude.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(p_mag_func)
    
    if rank == 0:
        print(f"  Wrote: pressure_magnitude.xdmf")
    
    # Export pressure phase
    p_phase_func = fem.Function(V_scalar, name="pressure_phase")
    p_phase_func.x.array[:] = np.angle(p_vals[:len(p_phase_func.x.array)])
    
    with io.XDMFFile(comm, output_dir / "pressure_phase.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(p_phase_func)
    
    if rank == 0:
        print(f"  Wrote: pressure_phase.xdmf")
    
    # Export Gorkov potential
    U_func = gorkov.U_function
    U_vals = U_func.x.array
    
    U_real_func = fem.Function(V_scalar, name="gorkov_potential")
    if np.iscomplexobj(U_vals):
        U_real_func.x.array[:] = np.real(U_vals[:len(U_real_func.x.array)])
    else:
        U_real_func.x.array[:] = U_vals[:len(U_real_func.x.array)]
    
    with io.XDMFFile(comm, output_dir / "gorkov_potential.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(U_real_func)
    
    if rank == 0:
        print(f"  Wrote: gorkov_potential.xdmf")
    
    # Export streaming velocity if available
    if stream_sol is not None:
        u_func = stream_sol.u_function
        
        # Interpolate to P1 vector space for XDMF compatibility
        cg1_vec = element("Lagrange", domain.topology.cell_name(), 1, shape=(3,))
        V_vec = functionspace(domain, cg1_vec)
        u_interp = fem.Function(V_vec, name="streaming_velocity")
        u_interp.interpolate(u_func)
        
        with io.XDMFFile(comm, output_dir / "streaming_velocity.xdmf", "w") as xdmf:
            xdmf.write_mesh(domain)
            xdmf.write_function(u_interp)
        
        if rank == 0:
            print(f"  Wrote: streaming_velocity.xdmf")
    
    if rank == 0:
        print("\n" + "="*70)
        print("EXPORT COMPLETE")
        print("="*70)
        print(f"\nFiles in: {output_dir}")
        print("  - pressure_magnitude.xdmf + .h5")
        print("  - pressure_phase.xdmf + .h5")
        print("  - gorkov_potential.xdmf + .h5")
        print("  - streaming_velocity.xdmf + .h5")
        print("\nOpen in ParaView: File → Open → Select .xdmf files")


def main():
    parser = argparse.ArgumentParser(description="Export deposition results to XDMF format")
    parser.add_argument("--run_dir", type=str, default="results/deposition_20260209_194235",
                       help="Run directory with deposition results")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for XDMF files (default: run_dir/paraview)")
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "paraview"
    
    export_deposition_xdmf(run_dir, output_dir)


if __name__ == "__main__":
    main()
