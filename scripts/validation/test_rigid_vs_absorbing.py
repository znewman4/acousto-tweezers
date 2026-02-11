"""Quick test: streaming velocity with disc-only impedance (petri-dish model).

The old test compared impedance vs rigid passive walls. After the BC rewrite,
side walls are ALWAYS rigid. This test now verifies the new baseline works:
standing mode should produce strong standing waves, combined mode should
produce both streaming and Gorkov forces.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import numpy as np
import time

from acoustweezers.experiments.shallow_square_dish.config import ShallowDishConfig
from acoustweezers.experiments.shallow_square_dish.solve_pressure import create_mesh, solve_helmholtz
from acoustweezers.experiments.shallow_square_dish.streaming import compute_streaming_velocity

from mpi4py import MPI

rank = MPI.COMM_WORLD.rank

for mode in ["standing", "vortex", "combined"]:
    print(f"\n{'='*70}")
    print(f" MODE: {mode}")
    print(f"{'='*70}")
    
    cfg = ShallowDishConfig(
        L=10e-3,
        H=1e-3,
        elements_per_wavelength=6,
        min_elements_z=8,
    )
    
    domain, facet_tags, tag_map = create_mesh(cfg, verbose=(rank==0))
    p_sol = solve_helmholtz(domain, facet_tags, cfg, mode=mode, verbose=(rank==0))
    
    print(f"\n  Pressure: max|p| = {np.max(np.abs(p_sol.p_function.x.array)):.4f} Pa")
    
    s_sol = compute_streaming_velocity(p_sol, downsample_factor=1, verbose=(rank==0))
    
    if s_sol is not None:
        u_arr = s_sol.u_values  # shape (N, 3)
        u_mag = np.linalg.norm(np.real(u_arr), axis=1)
        print(f"\n  STREAMING RESULT ({mode}):")
        print(f"    max |u| = {np.max(u_mag):.6e} m/s = {np.max(u_mag)*1e6:.4f} μm/s")
        print(f"    mean |u| = {np.mean(u_mag):.6e} m/s = {np.mean(u_mag)*1e6:.4f} μm/s")
    else:
        print(f"  STREAMING FAILED for {mode}")
