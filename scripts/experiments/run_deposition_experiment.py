#!/usr/bin/env python3
"""
In-Silico Particle Deposition Experiment

Three-phase protocol:
1. Vortex-only: Move particle toward center/lift/guide
2. Add standing wave: Trap forms
3. Stability test: Streaming stays on, particle should remain trapped

Physics checks:
- U(t) decreases when trap is active
- Particle spirals into minimum (not random oscillation)
- χ(t) drops below 1 once trapped
- Particle stays trapped for O(seconds)

Outputs:
- particles_timeseries.csv (U, |F|, |u_s|, χ, dist_to_min)
- 3D trajectory VTU
- Pass/fail assertions
- ParaView-ready visualization

Usage:
    micromamba run -n acousto-complex python scripts/validation/run_deposition_experiment.py

Author: Acousto-Tweezers Project
Date: 2026-02-09
"""

from __future__ import annotations

import sys
import csv
import json
import pandas as pd
from datetime import datetime
from pathlib import Path

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import io

from acoustweezers.experiments.shallow_square_dish.config import ShallowDishConfig
from acoustweezers.experiments.shallow_square_dish.solve_pressure import (
    create_mesh, solve_helmholtz
)
from acoustweezers.experiments.shallow_square_dish.streaming import solve_streaming
from acoustweezers.experiments.shallow_square_dish.particles import (
    compute_gorkov_potential, ParticleDynamics
)


def run_deposition_experiment():
    """Run the three-phase deposition experiment."""
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"results/deposition_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if rank == 0:
        print("\n" + "#"*70)
        print("# PARTICLE DEPOSITION EXPERIMENT")
        print("#"*70)
        print(f"\nOutput: {output_dir}\n")
    
    # =========================================================================
    # SETUP: Configuration
    # =========================================================================
    if rank == 0:
        print("="*70)
        print("CONFIGURATION")
        print("="*70)
    
    cfg = ShallowDishConfig(
        L=10e-3,                    # 10 mm square dish
        H=1e-3,                     # 1 mm depth
        frequency_hz=500e3,         # 500 kHz
        elements_per_wavelength=6,
        min_elements_z=8,
        
        # Vortex parameters
        vortex_velocity_amplitude=2e-6,      # 2 μm/s
        vortex_topological_charge=1,
        vortex_aperture_radius=3e-3,
        
        # Standing wave parameters
        standing_velocity_amplitude=1e-6,    # 1 μm/s (weaker)
        standing_phase_pattern="antiphase",
        standing_axis="x",
        
        # Particle parameters
        particle_radius=5e-6,                # 5 μm polystyrene
        particle_density=1050.0,
        particle_dt=1e-5,                    # 10 μs
        particle_t_max=2.0,                  # 2 seconds max
    )
    
    if rank == 0:
        print(f"  Domain: {cfg.L*1e3:.1f} mm × {cfg.H*1e3:.2f} mm")
        print(f"  Frequency: {cfg.frequency_hz/1e3:.0f} kHz")
        print(f"  Particle: {cfg.particle_radius*1e6:.1f} μm radius")
        print(f"  Integration: dt={cfg.particle_dt*1e6:.1f} μs, t_max={cfg.particle_t_max:.1f} s")
    
    # Create mesh once
    domain, facet_tags, tag_map = create_mesh(cfg, verbose=(rank==0))
    
    # =========================================================================
    # PHASE 1: Vortex only (guide particle)
    # =========================================================================
    if rank == 0:
        print("\n" + "="*70)
        print("PHASE 1: VORTEX ONLY (Guidance)")
        print("="*70)
    
    p1 = solve_helmholtz(domain, facet_tags, cfg, mode="vortex", verbose=(rank==0))
    stream1 = solve_streaming(p1, domain=domain, cfg=cfg, verbose=(rank==0))
    gorkov1 = compute_gorkov_potential(p1, verbose=(rank==0))
    
    # Initial position: offset from center
    x0_phase1 = np.array([cfg.L/2 + 2e-3, cfg.L/2 + 1e-3, cfg.H/2])
    
    dynamics1 = ParticleDynamics(gorkov1, stream1, cfg)
    
    if rank == 0:
        print(f"\n  Initial position: ({x0_phase1[0]*1e3:.2f}, {x0_phase1[1]*1e3:.2f}, {x0_phase1[2]*1e3:.2f}) mm")
        print(f"  Integrating for {cfg.particle_t_max:.2f} s...")
    
    traj1 = dynamics1.integrate(
        x0_phase1,
        t_max=0.5,  # 0.5 s for phase 1
        track_diagnostics=True
    )
    
    if rank == 0:
        final1 = traj1.final_position
        print(f"  Final position: ({final1[0]*1e3:.2f}, {final1[1]*1e3:.2f}, {final1[2]*1e3:.2f}) mm")
        print(f"  Displacement: {traj1.displacement*1e3:.2f} mm")
        print(f"  U(t=0): {traj1.U[0]:.2e} J")
        print(f"  U(final): {traj1.U[-1]:.2e} J")
        print(f"  ΔU: {traj1.U[-1] - traj1.U[0]:.2e} J")
    
    # =========================================================================
    # PHASE 2: Vortex + Standing Wave (trap forms)
    # =========================================================================
    if rank == 0:
        print("\n" + "="*70)
        print("PHASE 2: VORTEX + STANDING WAVE (Trapping)")
        print("="*70)
    
    p2 = solve_helmholtz(domain, facet_tags, cfg, mode="combined", verbose=(rank==0))
    stream2 = solve_streaming(p2, domain=domain, cfg=cfg, verbose=(rank==0))
    gorkov2 = compute_gorkov_potential(p2, verbose=(rank==0))
    
    # Start where phase 1 ended
    x0_phase2 = traj1.final_position
    
    dynamics2 = ParticleDynamics(gorkov2, stream2, cfg)
    
    if rank == 0:
        print(f"\n  Starting position: ({x0_phase2[0]*1e3:.2f}, {x0_phase2[1]*1e3:.2f}, {x0_phase2[2]*1e3:.2f}) mm")
        print(f"  Integrating for 1.0 s...")
    
    traj2 = dynamics2.integrate(
        x0_phase2,
        t_max=1.0,  # 1.0 s for phase 2
        track_diagnostics=True
    )
    
    if rank == 0:
        final2 = traj2.final_position
        print(f"  Final position: ({final2[0]*1e3:.2f}, {final2[1]*1e3:.2f}, {final2[2]*1e3:.2f}) mm")
        print(f"  Displacement: {traj2.displacement*1e3:.2f} mm")
        print(f"  U(t=0): {traj2.U[0]:.2e} J")
        print(f"  U(final): {traj2.U[-1]:.2e} J")
        print(f"  ΔU: {traj2.U[-1] - traj2.U[0]:.2e} J")
        
        # Check chi evolution
        chi_start = traj2.chi[0]
        chi_end = np.median(traj2.chi[-1000:])  # Last 10 ms
        print(f"  χ(t=0): {chi_start:.3f}")
        print(f"  χ(final): {chi_end:.3f}")
    
    # =========================================================================
    # PHASE 3: Stability test (stay trapped with streaming)
    # =========================================================================
    if rank == 0:
        print("\n" + "="*70)
        print("PHASE 3: STABILITY TEST (Streaming Competition)")
        print("="*70)
    
    # Continue from phase 2 with same fields
    x0_phase3 = traj2.final_position
    
    if rank == 0:
        print(f"\n  Starting position: ({x0_phase3[0]*1e3:.2f}, {x0_phase3[1]*1e3:.2f}, {x0_phase3[2]*1e3:.2f}) mm")
        print(f"  Integrating for 0.5 s...")
    
    traj3 = dynamics2.integrate(  # Use same dynamics as phase 2
        x0_phase3,
        t_max=0.5,  # 0.5 s for phase 3
        track_diagnostics=True
    )
    
    if rank == 0:
        final3 = traj3.final_position
        print(f"  Final position: ({final3[0]*1e3:.2f}, {final3[1]*1e3:.2f}, {final3[2]*1e3:.2f}) mm")
        print(f"  Displacement from phase 2 end: {np.linalg.norm(final3 - x0_phase3)*1e3:.2f} mm")
        
        # Stability metric: did particle stay near trap?
        stability_radius = traj3.displacement
        print(f"  Stability radius: {stability_radius*1e6:.1f} μm")
    
    # =========================================================================
    # CONCATENATE TRAJECTORIES
    # =========================================================================
    if rank == 0:
        print("\n" + "="*70)
        print("CONCATENATING TRAJECTORIES")
        print("="*70)
    
    # Offset times for phase 2 and 3
    t_all = np.concatenate([
        traj1.t,
        traj1.t[-1] + traj2.t[1:],  # Skip duplicate t=0
        traj1.t[-1] + traj2.t[-1] + traj3.t[1:]
    ])
    
    x_all = np.concatenate([traj1.x, traj2.x[1:], traj3.x[1:]])
    y_all = np.concatenate([traj1.y, traj2.y[1:], traj3.y[1:]])
    z_all = np.concatenate([traj1.z, traj2.z[1:], traj3.z[1:]])
    
    U_all = np.concatenate([traj1.U, traj2.U[1:], traj3.U[1:]])
    F_rad_all = np.concatenate([traj1.F_rad_mag, traj2.F_rad_mag[1:], traj3.F_rad_mag[1:]])
    u_stream_all = np.concatenate([traj1.u_stream_mag, traj2.u_stream_mag[1:], traj3.u_stream_mag[1:]])
    chi_all = np.concatenate([traj1.chi, traj2.chi[1:], traj3.chi[1:]])
    dist_all = np.concatenate([traj1.dist_to_min, traj2.dist_to_min[1:], traj3.dist_to_min[1:]])
    
    # Phase labels
    phase_labels = np.concatenate([
        np.ones(len(traj1.t)),
        2 * np.ones(len(traj2.t) - 1),
        3 * np.ones(len(traj3.t) - 1)
    ])
    
    if rank == 0:
        print(f"  Total trajectory points: {len(t_all)}")
        print(f"  Total time: {t_all[-1]:.3f} s")
    
    # =========================================================================
    # EXPORT CSV
    # =========================================================================
    if rank == 0:
        print("\n" + "="*70)
        print("EXPORTING CSV")
        print("="*70)
    
    csv_path = output_dir / "particles_timeseries.csv"
    
    if rank == 0:
        df = pd.DataFrame({
            't_s': t_all,
            'phase': phase_labels.astype(int),
            'x_m': x_all,
            'y_m': y_all,
            'z_m': z_all,
            'U_J': U_all,
            'F_rad_mag_N': F_rad_all,
            'u_stream_mag_m_per_s': u_stream_all,
            'chi': chi_all,
            'dist_to_min_m': dist_all,
        })
        
        df.to_csv(csv_path, index=False)
        print(f"  Wrote: {csv_path}")
        print(f"  Rows: {len(df)}")
    
    # =========================================================================
    # PHYSICS CHECKS
    # =========================================================================
    if rank == 0:
        print("\n" + "="*70)
        print("PHYSICS CHECKS")
        print("="*70)
    
    checks = {}
    
    # Check 1: U(t) decreases in phase 2 (trap formation)
    U_phase2_start = traj2.U[0]
    U_phase2_end = traj2.U[-1]
    dU_phase2 = U_phase2_end - U_phase2_start
    check1 = bool(dU_phase2 < 0)
    checks['U_decreases_in_trap'] = check1
    
    if rank == 0:
        status = "✓ PASS" if check1 else "✗ FAIL"
        print(f"  1. U decreases when trap active: ΔU = {dU_phase2:.2e} J {status}")
    
    # Check 2: χ drops below 1 in phase 2 (trapping wins)
    chi_phase2_final = np.median(traj2.chi[-1000:])
    check2 = bool(chi_phase2_final < 1.0)
    checks['chi_below_1_when_trapped'] = check2
    
    if rank == 0:
        status = "✓ PASS" if check2 else "✗ FAIL"
        print(f"  2. χ < 1 when trapped: χ = {chi_phase2_final:.3f} {status}")
    
    # Check 3: Particle stays trapped in phase 3 (stability)
    stability_threshold = 100e-6  # 100 μm
    check3 = bool(stability_radius < stability_threshold)
    checks['stays_trapped'] = check3
    
    if rank == 0:
        status = "✓ PASS" if check3 else "✗ FAIL"
        print(f"  3. Stays trapped (drift < {stability_threshold*1e6:.0f} μm): {stability_radius*1e6:.1f} μm {status}")
    
    # Check 4: Distance to minimum decreases over time
    dist_phase2_start = traj2.dist_to_min[0]
    dist_phase2_end = traj2.dist_to_min[-1]
    check4 = bool(dist_phase2_end < dist_phase2_start)
    checks['approaches_minimum'] = check4
    
    if rank == 0:
        status = "✓ PASS" if check4 else "✗ FAIL"
        print(f"  4. Approaches minimum: Δdist = {(dist_phase2_end - dist_phase2_start)*1e6:.1f} μm {status}")
    
    all_passed = all(checks.values())
    
    # =========================================================================
    # EXPORT VTU
    # =========================================================================
    if rank == 0:
        print("\n" + "="*70)
        print("EXPORTING VTU FILES")
        print("="*70)
    
    vtu_dir = output_dir / "vtu"
    vtu_dir.mkdir(exist_ok=True)
    
    # Export final (phase 2) fields
    with io.VTXWriter(comm, vtu_dir / "pressure.bp", [p2.p_function], engine="BP4") as vtx:
        vtx.write(0.0)
    
    # stream2 is a dict with "streaming_solution" key containing the StreamingSolution object
    stream_sol = stream2.get("streaming_solution")
    if stream_sol is not None:
        with io.VTXWriter(comm, vtu_dir / "streaming.bp", [stream_sol.u_function], engine="BP4") as vtx:
            vtx.write(0.0)
    else:
        if rank == 0:
            print("  Warning: No streaming solution available for VTU export")
    
    with io.VTXWriter(comm, vtu_dir / "gorkov.bp", [gorkov2.U_function], engine="BP4") as vtx:
        vtx.write(0.0)
    
    if rank == 0:
        print(f"  Exported: pressure.bp, streaming.bp, gorkov.bp")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    if rank == 0:
        print("\n" + "="*70)
        if all_passed:
            print("✓ ALL PHYSICS CHECKS PASSED")
        else:
            print("✗ SOME PHYSICS CHECKS FAILED")
        print("="*70)
        
        summary = {
            'timestamp': timestamp,
            'total_time_s': float(t_all[-1]),
            'n_steps': int(len(t_all)),
            'checks': checks,
            'phase1_displacement_mm': float(traj1.displacement * 1e3),
            'phase2_displacement_mm': float(traj2.displacement * 1e3),
            'phase3_displacement_mm': float(traj3.displacement * 1e3),
            'phase2_dU_J': float(dU_phase2),
            'phase2_chi_final': float(chi_phase2_final),
            'stability_radius_um': float(stability_radius * 1e6),
        }
        
        with open(output_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nResults: {output_dir}")
        print(f"  - particles_timeseries.csv")
        print(f"  - vtu/ (pressure, streaming, gorkov)")
        print(f"  - summary.json")
    
    return 0 if all_passed else 1


def main():
    """Entry point."""
    return run_deposition_experiment()


if __name__ == "__main__":
    sys.exit(main())
