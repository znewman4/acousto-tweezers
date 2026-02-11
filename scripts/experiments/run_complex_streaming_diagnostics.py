#!/usr/bin/env python3
"""
Complex Streaming Diagnostics

Runs a canonical vortex case with complex phasor pressure and verifies:
1. PETSc complex scalar type
2. Phase winding matches topological charge
3. Non-zero pressure and streaming velocities
4. Divergence metrics

Outputs:
- CSV file with quantitative diagnostics
- VTU files for ParaView visualization
- Pass/fail assertions

Usage:
    micromamba run -n acousto-complex python scripts/validation/run_complex_streaming_diagnostics.py

Author: Acousto-Tweezers Project
Date: 2026-02-09
"""

from __future__ import annotations

import sys
import csv
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import numpy as np


def get_git_info() -> str:
    """Get current git commit hash or 'dirty' if uncommitted changes."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        commit = result.stdout.strip() if result.returncode == 0 else "unknown"
        
        # Check for uncommitted changes
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            commit += "-dirty"
        
        return commit
    except Exception:
        return "unknown"


def run_diagnostics():
    """Run the complex streaming diagnostics."""
    from mpi4py import MPI
    from petsc4py import PETSc
    import dolfinx
    from dolfinx import fem, io
    
    from acoustweezers.experiments.shallow_square_dish.config import ShallowDishConfig
    from acoustweezers.experiments.shallow_square_dish.solve_pressure import (
        create_mesh, solve_helmholtz, compute_phase_winding
    )
    from acoustweezers.experiments.shallow_square_dish.streaming import solve_streaming
    
    # Import energy diagnostics
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from diagnostics.energy_budget import compute_energy_budget
    
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"results/complex_diagnostics_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Also create main diagnostics folder
    diag_dir = Path("results/diagnostics")
    diag_dir.mkdir(parents=True, exist_ok=True)
    
    if rank == 0:
        print("\n" + "#"*70)
        print("# COMPLEX STREAMING DIAGNOSTICS")
        print("#"*70)
        print(f"\nOutput directory: {output_dir}")
    
    # =========================================================================
    # STEP 1: Verify complex backend
    # =========================================================================
    if rank == 0:
        print("\n" + "="*70)
        print("STEP 1: Verify Complex Backend")
        print("="*70)
    
    petsc_scalar_type = str(PETSc.ScalarType)
    is_complex = np.issubdtype(PETSc.ScalarType, np.complexfloating)
    dolfinx_version = dolfinx.__version__
    
    # Get PETSc version
    try:
        petsc_version = PETSc.Sys.getVersion()
        petsc_version_str = f"{petsc_version[0]}.{petsc_version[1]}.{petsc_version[2]}"
    except:
        petsc_version_str = "unknown"
    
    if rank == 0:
        print(f"  PETSc ScalarType: {petsc_scalar_type}")
        print(f"  Is complex: {is_complex}")
        print(f"  DOLFINx version: {dolfinx_version}")
        print(f"  PETSc version: {petsc_version_str}")
    
    if not is_complex:
        if rank == 0:
            print("\n  ✗ FATAL: PETSc not compiled with complex scalars!")
            print("  Use: micromamba run -n acousto-complex python <script>")
        return 1
    
    if rank == 0:
        print("  ✓ Complex backend verified")
    
    # =========================================================================
    # STEP 2: Set up canonical vortex case
    # =========================================================================
    if rank == 0:
        print("\n" + "="*70)
        print("STEP 2: Configure Canonical Vortex Case")
        print("="*70)
    
    cfg = ShallowDishConfig(
        L=10e-3,                     # 10 mm square dish
        H=1e-3,                      # 1 mm depth
        frequency_hz=500e3,          # 500 kHz
        elements_per_wavelength=6,   # Mesh density
        min_elements_z=8,            # Min z elements
        standing_velocity_amplitude=1e-6,  # 1 μm/s
        vortex_velocity_amplitude=1e-6,    # 1 μm/s
        vortex_topological_charge=1,       # ℓ = 1
        vortex_aperture_radius=3e-3,       # 3 mm
    )
    
    if rank == 0:
        print(f"  Domain: {cfg.L*1e3:.1f} mm × {cfg.L*1e3:.1f} mm × {cfg.H*1e3:.2f} mm")
        print(f"  Frequency: {cfg.frequency_hz/1e3:.0f} kHz")
        print(f"  Vortex charge ℓ: {cfg.vortex_topological_charge}")
        print(f"  Wavelength: {cfg.wavelength*1e3:.2f} mm")
    
    # =========================================================================
    # STEP 3: Create mesh and solve Helmholtz (vortex mode)
    # =========================================================================
    if rank == 0:
        print("\n" + "="*70)
        print("STEP 3: Solve Complex Helmholtz (Vortex Mode)")
        print("="*70)
    
    domain, facet_tags, tag_map = create_mesh(cfg, verbose=(rank==0))
    
    p_solution = solve_helmholtz(
        domain, facet_tags, cfg,
        mode="vortex",  # Pure vortex to test phase winding
        verbose=(rank==0)
    )
    
    # Verify complex solution
    p_vals = p_solution.p_values
    p_is_complex = np.issubdtype(p_vals.dtype, np.complexfloating)
    max_abs_p = np.max(np.abs(p_vals))
    
    if rank == 0:
        print(f"\n  Pressure solution:")
        print(f"    dtype: {p_vals.dtype}")
        print(f"    is_complex: {p_is_complex}")
        print(f"    max|p|: {max_abs_p:.4f} Pa")
    
    # =========================================================================
    # STEP 4: Compute phase winding
    # =========================================================================
    if rank == 0:
        print("\n" + "="*70)
        print("STEP 4: Compute Phase Winding")
        print("="*70)
    
    center_xy = (cfg.L/2, cfg.L/2)
    sample_radius = cfg.vortex_aperture_radius * 0.5  # Inside aperture
    sample_z = cfg.H / 2  # Mid-height
    
    winding_number = compute_phase_winding(
        p_solution, center_xy, sample_radius, sample_z, n_samples=200
    )
    winding_error = abs(winding_number - cfg.vortex_topological_charge)
    
    if rank == 0:
        print(f"  Sample circle:")
        print(f"    center: ({center_xy[0]*1e3:.2f}, {center_xy[1]*1e3:.2f}) mm")
        print(f"    radius: {sample_radius*1e3:.2f} mm")
        print(f"    z: {sample_z*1e3:.2f} mm")
        print(f"  Phase winding number: {winding_number:.3f}")
        print(f"  Expected (ℓ): {cfg.vortex_topological_charge}")
        print(f"  Error: {winding_error:.3f}")
    
    # =========================================================================
    # STEP 5: Solve streaming
    # =========================================================================
    if rank == 0:
        print("\n" + "="*70)
        print("STEP 5: Solve Acoustic Streaming")
        print("="*70)
    
    streaming_result = solve_streaming(
        p_solution, domain=domain, cfg=cfg, verbose=(rank==0)
    )
    
    max_stream_u = streaming_result.get('max_speed', 0.0)
    stream_sol = streaming_result.get('streaming_solution')
    
    # Get median and other stats
    if stream_sol is not None:
        u_vals = stream_sol.u_function.x.array
        n_u = len(u_vals) // 3
        u_mag = np.linalg.norm(np.real(u_vals).reshape((n_u, 3)), axis=1)
        median_stream_u = np.median(u_mag)
        
        # Divergence (from diagnostics)
        diags = stream_sol.diagnostics or {}
        div_l2 = diags.get('divergence_l2_norm', 0.0)
        div_rel = diags.get('divergence_l2_norm_relative', 0.0)
    else:
        median_stream_u = 0.0
        div_l2 = 0.0
        div_rel = 0.0
    
    if rank == 0:
        print(f"\n  Streaming solution:")
        print(f"    max|u_s|: {max_stream_u*1e6:.2f} μm/s")
        print(f"    median|u_s|: {median_stream_u*1e6:.2f} μm/s")
        print(f"    div(u) L2: {div_l2:.2e}")
        print(f"    div(u) relative: {div_rel:.2e}")
    
    # =========================================================================
    # STEP 6: Compute additional diagnostics
    # =========================================================================
    if rank == 0:
        print("\n" + "="*70)
        print("STEP 6: Additional Diagnostics")
        print("="*70)
    
    # Gradient of pressure
    coords = p_solution.coords
    grad_p_mag = np.zeros(len(coords))
    
    # Rough estimate using finite differences on DOF coordinates
    # (This is approximate; proper gradient would use FEM projection)
    p_abs = np.abs(p_vals)
    for i in range(len(coords)):
        # Find nearby DOFs
        dists = np.linalg.norm(coords - coords[i], axis=1)
        near = (dists > 0) & (dists < 1e-4)  # Within 0.1 mm
        if np.any(near):
            dp = np.max(np.abs(p_abs[near] - p_abs[i]))
            dx = np.min(dists[near])
            grad_p_mag[i] = dp / dx if dx > 0 else 0
    
    max_abs_gradp = np.max(grad_p_mag)
    
    # First-order velocity magnitude
    max_abs_v1 = max_abs_gradp / (cfg.omega * cfg.rho)
    
    # Reynolds number estimate
    L_char = cfg.H  # Use depth as characteristic length
    Re_streaming = cfg.rho * max_stream_u * L_char / cfg.mu
    
    if rank == 0:
        print(f"  max|∇p|: {max_abs_gradp:.2e} Pa/m (approx)")
        print(f"  max|v₁|: {max_abs_v1*1e6:.2f} μm/s (approx)")
        print(f"  Re_streaming: {Re_streaming:.2e}")
    
    # =========================================================================
    # STEP 6b: Energy Budget Diagnostics
    # =========================================================================
    if rank == 0:
        print("\n" + "="*70)
        print("STEP 6b: Energy Budget Diagnostics")
        print("="*70)
    
    energy_budget = compute_energy_budget(p_solution, stream_sol, cfg, verbose=(rank==0))
    
    # Write energy budget to JSON
    energy_json_path = output_dir / "energy_budget.json"
    if rank == 0:
        with open(energy_json_path, 'w') as f:
            json.dump(energy_budget, f, indent=2)
        print(f"\n  Wrote: {energy_json_path}")
    
    # =========================================================================
    # STEP 7: Export fields for ParaView
    # =========================================================================
    if rank == 0:
        print("\n" + "="*70)
        print("STEP 7: Export Fields for ParaView")
        print("="*70)
    
    # Export pressure fields
    p_func = p_solution.p_function
    V = p_func.function_space
    
    # Create real/imag/abs/phase functions
    p_real_func = fem.Function(V, name="p_real")
    p_imag_func = fem.Function(V, name="p_imag")
    p_abs_func = fem.Function(V, name="p_abs")
    p_phase_func = fem.Function(V, name="p_phase")
    
    p_real_func.x.array[:] = np.real(p_vals)
    p_imag_func.x.array[:] = np.imag(p_vals)
    p_abs_func.x.array[:] = np.abs(p_vals)
    p_phase_func.x.array[:] = np.angle(p_vals)
    
    # Write pressure VTU
    with io.VTXWriter(comm, output_dir / "pressure_fields.bp", [p_real_func, p_imag_func, p_abs_func, p_phase_func], engine="BP4") as vtx:
        vtx.write(0.0)
    
    if rank == 0:
        print(f"  Exported: pressure_fields.bp")
    
    # Export streaming fields
    if stream_sol is not None:
        u_func = stream_sol.u_function
        
        # Compute magnitude
        V_streaming = u_func.function_space
        u_vals_real = np.real(u_func.x.array)
        n_u = len(u_vals_real) // 3
        u_reshaped = u_vals_real.reshape((n_u, 3))
        
        with io.VTXWriter(comm, output_dir / "streaming_fields.bp", [u_func], engine="BP4") as vtx:
            vtx.write(0.0)
        
        if rank == 0:
            print(f"  Exported: streaming_fields.bp")
    
    # =========================================================================
    # STEP 8: Write CSV diagnostics
    # =========================================================================
    if rank == 0:
        print("\n" + "="*70)
        print("STEP 8: Write CSV Diagnostics")
        print("="*70)
    
    diag_row = {
        'timestamp': timestamp,
        'git_commit': get_git_info(),
        'petsc_scalar_type': 'complex' if is_complex else 'real',
        'dolfinx_version': dolfinx_version,
        'petsc_version': petsc_version_str,
        'frequency_hz': cfg.frequency_hz,
        'omega': cfg.omega,
        'vortex_charge_l': cfg.vortex_topological_charge,
        'max_abs_p_Pa': float(max_abs_p),
        'max_abs_gradp_Pa_per_m': float(max_abs_gradp),
        'max_abs_v1_m_per_s': float(max_abs_v1),
        'max_abs_forcing_N_per_m3': float(diags.get('max_forcing_pa_m', 0.0)) if diags else 0.0,
        'max_stream_u_m_per_s': float(max_stream_u),
        'median_stream_u_m_per_s': float(median_stream_u),
        'Re_streaming': float(Re_streaming),
        'div_u_L2': float(div_l2),
        'div_u_rel': float(div_rel),
        'phase_winding_number': float(winding_number),
        'phase_winding_error': float(winding_error),
        # Energy budget
        'viscous_dissipation_W': energy_budget['viscous_dissipation_W'],
        'dissipation_density_max_W_per_m3': energy_budget['dissipation_density_max_W_per_m3'],
        'acoustic_intensity_max_W_per_m2': energy_budget['acoustic_intensity_max_W_per_m2'],
        'dissipation_to_acoustic_ratio': energy_budget['dissipation_to_acoustic_ratio'],
        'notes': 'canonical_vortex_l1',
    }
    
    # Write to run folder
    csv_path = output_dir / "complex_streaming_proof.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=diag_row.keys())
        writer.writeheader()
        writer.writerow(diag_row)
    
    if rank == 0:
        print(f"  Wrote: {csv_path}")
    
    # Also write to main diagnostics folder
    main_csv = diag_dir / "complex_streaming_proof.csv"
    file_exists = main_csv.exists()
    with open(main_csv, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=diag_row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(diag_row)
    
    if rank == 0:
        print(f"  Appended to: {main_csv}")
    
    # Write JSON summary
    json_path = output_dir / "diagnostics.json"
    with open(json_path, 'w') as f:
        json.dump(diag_row, f, indent=2)
    
    if rank == 0:
        print(f"  Wrote: {json_path}")
    
    # =========================================================================
    # STEP 9: Write ParaView README
    # =========================================================================
    readme_content = f"""# ParaView Visualization Guide

## Generated Files

- `pressure_fields.bp` - Complex pressure field (BP4/ADIOS2 format)
  - `p_real`: Real part of pressure phasor
  - `p_imag`: Imaginary part of pressure phasor
  - `p_abs`: Magnitude |p|
  - `p_phase`: Phase angle arg(p)

- `streaming_fields.bp` - Acoustic streaming velocity (BP4/ADIOS2 format)
  - `u_stream`: Streaming velocity vector

## Loading in ParaView

1. Open ParaView
2. File → Open → Select `pressure_fields.bp` or `streaming_fields.bp`
3. Click Apply

## Suggested Visualizations

### Vortex Phase Singularity
1. Load `pressure_fields.bp`
2. Create a Slice at z = {cfg.H/2*1e3:.2f} mm (mid-height)
3. Color by `p_phase` 
4. Use HSV color map to show phase winding from -π to π
5. Look for the phase singularity at vortex center

### Streaming Velocity
1. Load `streaming_fields.bp`
2. Apply Glyph filter with Arrow type
3. Scale by velocity magnitude
4. Color by `u_stream` magnitude

## Diagnostics Summary

- Frequency: {cfg.frequency_hz/1e3:.0f} kHz
- Vortex charge ℓ: {cfg.vortex_topological_charge}
- Phase winding measured: {winding_number:.3f}
- max|p|: {max_abs_p:.4f} Pa
- max|u_s|: {max_stream_u*1e6:.2f} μm/s

Generated: {timestamp}
"""
    
    with open(output_dir / "PARAVIEW_README.md", 'w') as f:
        f.write(readme_content)
    
    if rank == 0:
        print(f"  Wrote: PARAVIEW_README.md")
    
    # =========================================================================
    # STEP 10: Assertions and final summary
    # =========================================================================
    if rank == 0:
        print("\n" + "="*70)
        print("ASSERTIONS AND SUMMARY")
        print("="*70)
    
    all_passed = True
    
    # Assertion 1: Complex scalar type
    assert1 = is_complex
    if rank == 0:
        status = "✓ PASS" if assert1 else "✗ FAIL"
        print(f"  1. PETSc scalar type is complex: {status}")
    all_passed = all_passed and assert1
    
    # Assertion 2: Phase winding within tolerance
    winding_tol = 0.3  # Allow ±0.3 error on coarse mesh
    assert2 = winding_error < winding_tol
    if rank == 0:
        status = "✓ PASS" if assert2 else "✗ FAIL"
        print(f"  2. Phase winding error < {winding_tol}: {winding_error:.3f} {status}")
    all_passed = all_passed and assert2
    
    # Assertion 3: Non-zero pressure
    assert3 = max_abs_p > 0
    if rank == 0:
        status = "✓ PASS" if assert3 else "✗ FAIL"
        print(f"  3. max|p| > 0: {max_abs_p:.4f} Pa {status}")
    all_passed = all_passed and assert3
    
    # Assertion 4: Non-zero streaming
    assert4 = max_stream_u > 0
    if rank == 0:
        status = "✓ PASS" if assert4 else "✗ FAIL"
        print(f"  4. max|u_s| > 0: {max_stream_u*1e6:.2f} μm/s {status}")
    all_passed = all_passed and assert4
    
    # Assertion 5: Divergence check (warn only)
    div_warning = div_rel > 1.0  # Warn if relative divergence > 1
    if rank == 0:
        status = "⚠ WARN" if div_warning else "✓ OK"
        print(f"  5. Divergence (relative): {div_rel:.2e} {status}")
    
    # Final result
    if rank == 0:
        print("\n" + "="*70)
        if all_passed:
            print("✓ ALL ASSERTIONS PASSED")
        else:
            print("✗ SOME ASSERTIONS FAILED")
        print("="*70)
        print(f"\nResults saved to: {output_dir}")
        print(f"CSV appended to: {main_csv}")
    
    return 0 if all_passed else 1


def main():
    """Entry point."""
    return run_diagnostics()


if __name__ == "__main__":
    sys.exit(main())
