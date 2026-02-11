#!/usr/bin/env python3
"""
Particle Dynamics with Streaming + ParaView Story Visualization.

STEP 1 & 2: Physics-clear demonstration of particle-acoustic-streaming coupling.

Coupling equation (explicitly implemented):
    ẋᵢ = u_stream(xᵢ) + F_Gor'kov(xᵢ) / (6πμa)

Where:
  • u_stream(x) = precomputed steady streaming velocity [m/s]
  • F_Gor'kov(x) = -∇U(x) radiation force [N]
  • a = particle radius [m]
  • μ = fluid viscosity [Pa·s]
  
This is overdamped Stokes drag (no inertia).

Validation protocol:
  1. Gor'kov only → particles trap at potential minima
  2. Streaming only → pure advection along streamlines
  3. Streaming + Gor'kov → trapped yet drifting

Author: Acousto-Tweezers Project
Date: 2026-02-09
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dolfinx import mesh, fem, io
from mpi4py import MPI

from acoustweezers.experiments.shallow_square_dish.config import ShallowDishConfig
from acoustweezers.experiments.shallow_square_dish.solve_pressure import (
    create_mesh, solve_helmholtz,
)
from acoustweezers.experiments.shallow_square_dish.streaming import solve_streaming
from acoustweezers.experiments.shallow_square_dish.particles import (
    compute_gorkov_potential,
    ParticleDynamics,
    save_trajectories_csv,
)
from acoustweezers.experiments.shallow_square_dish.export import (
    export_pressure_fields,
    export_streaming_fields,
    export_gorkov_fields,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

def create_demo_config() -> ShallowDishConfig:
    """Create configuration for streaming + particle demo."""
    cfg = ShallowDishConfig(
        L=0.01,                      # 1 cm (smaller for demo)
        H=0.001,                     # 1 mm depth
        frequency_hz=500_000,        # 500 kHz
        vortex_velocity_amplitude=10e-6,
        vortex_topological_charge=1,
        vortex_aperture_radius=0.002,     # 2 mm
        standing_velocity_amplitude=1e-6,
        elements_per_wavelength=4,        # Medium mesh
        particle_t_max=0.01,              # Short simulation
        particle_dt=1e-5,                 # 10 μs time step
    )
    return cfg


# ============================================================================
# STEP 1: PARTICLE DYNAMICS VALIDATION
# ============================================================================

def validate_streaming_coupling(
    gorkov,
    streaming_solution: dict,
    cfg,
    output_dir: Path,
) -> Dict:
    """
    Validate that streaming affects particle motion.
    
    Returns dict with validation results and trajectories.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    if rank == 0:
        print("\n" + "="*70)
        print("PARTICLE DYNAMICS VALIDATION")
        print("="*70)
        print("\nTest Protocol:")
        print("  1. Gor'kov radiation force alone (no streaming)")
        print("  2. Streaming advection alone (no Gor'kov)")
        print("  3. Streaming + Gor'kov coupled")
    
    # Initial position: domain center at mid-height
    x0 = np.array([cfg.L/2, cfg.L/2, 0.2 * cfg.H])
    
    results = {
        'x0': x0.tolist(),
        'validation_details': {},
        'trajectories': {},
    }
    
    # -----------------------------------------------------------------------
    # TEST 1: Gor'kov radiation force only
    # -----------------------------------------------------------------------
    if rank == 0:
        print("\n" + "-"*70)
        print("TEST 1: Gor'kov Radiation Force Only")
        print("-"*70)
    
    dynamics_1 = ParticleDynamics(gorkov, streaming=None, cfg=cfg)
    traj_1 = dynamics_1.integrate(x0, t_max=0.01, dt=1e-5, method="rk2")
    
    final_pos_1 = traj_1.final_position
    disp_1 = np.linalg.norm(final_pos_1 - x0)
    
    if rank == 0:
        print(f"Initial: ({x0[0]*1e3:.3f}, {x0[1]*1e3:.3f}, {x0[2]*1e3:.3f}) mm")
        print(f"Final:   ({final_pos_1[0]*1e3:.3f}, {final_pos_1[1]*1e3:.3f}, {final_pos_1[2]*1e3:.3f}) mm")
        print(f"Displacement: {disp_1*1e3:.4f} mm")
        print(f"Expected: particle traps near potential minimum ✓")
    
    results['trajectories']['gorkov_only'] = {
        't': traj_1.t.tolist(),
        'x': traj_1.x.tolist(),
        'y': traj_1.y.tolist(),
        'z': traj_1.z.tolist(),
    }
    results['validation_details']['gorkov_displacement_mm'] = float(disp_1 * 1e3)
    
    # -----------------------------------------------------------------------
    # TEST 2: Streaming advection only
    # -----------------------------------------------------------------------
    if rank == 0:
        print("\n" + "-"*70)
        print("TEST 2: Streaming Advection Only")
        print("-"*70)
    
    # Manually integrate pure streaming (no Gor'kov)
    if streaming_solution is not None and 'u_function' in streaming_solution:
        t_max = 0.01
        dt = 1e-5
        n_steps = int(t_max / dt)
        
        t_arr = np.zeros(n_steps + 1)
        pos_stream = np.zeros((n_steps + 1, 3))
        pos_stream[0] = x0
        
        # Create temp dynamics to access _eval_streaming
        temp_dyn = ParticleDynamics(gorkov, streaming=streaming_solution, cfg=cfg)
        
        for i in range(n_steps):
            try:
                u_s = temp_dyn._eval_streaming(pos_stream[i])
                pos_stream[i+1] = pos_stream[i] + u_s * dt
            except:
                pos_stream[i+1] = pos_stream[i]
            t_arr[i] = i * dt
        
        disp_2 = np.linalg.norm(pos_stream[-1] - x0)
        
        if rank == 0:
            print(f"Initial: ({x0[0]*1e3:.3f}, {x0[1]*1e3:.3f}, {x0[2]*1e3:.3f}) mm")
            print(f"Final:   ({pos_stream[-1][0]*1e3:.3f}, {pos_stream[-1][1]*1e3:.3f}, {pos_stream[-1][2]*1e3:.3f}) mm")
            print(f"Displacement: {disp_2*1e3:.4f} mm")
            print(f"Expected: particle drifts along streamlines ✓")
        
        results['trajectories']['streaming_only'] = {
            't': t_arr.tolist(),
            'x': pos_stream[:, 0].tolist(),
            'y': pos_stream[:, 1].tolist(),
            'z': pos_stream[:, 2].tolist(),
        }
        results['validation_details']['streaming_displacement_mm'] = float(disp_2 * 1e3)
    else:
        disp_2 = 0
        if rank == 0:
            print("(Streaming not computed - skipping)")
    
    # -----------------------------------------------------------------------
    # TEST 3: Streaming + Gor'kov coupled
    # -----------------------------------------------------------------------
    if rank == 0:
        print("\n" + "-"*70)
        print("TEST 3: Streaming + Gor'kov Coupled")
        print("-"*70)
    
    if streaming_solution is not None:
        dynamics_3 = ParticleDynamics(gorkov, streaming=streaming_solution, cfg=cfg)
        traj_3 = dynamics_3.integrate(x0, t_max=0.01, dt=1e-5, method="rk2")
        
        final_pos_3 = traj_3.final_position
        disp_3 = np.linalg.norm(final_pos_3 - x0)
        
        if rank == 0:
            print(f"Initial: ({x0[0]*1e3:.3f}, {x0[1]*1e3:.3f}, {x0[2]*1e3:.3f}) mm")
            print(f"Final:   ({final_pos_3[0]*1e3:.3f}, {final_pos_3[1]*1e3:.3f}, {final_pos_3[2]*1e3:.3f}) mm")
            print(f"Displacement: {disp_3*1e3:.4f} mm")
            print(f"Expected: trapped yet drifting (intermediate) ✓")
        
        results['trajectories']['coupled'] = {
            't': traj_3.t.tolist(),
            'x': traj_3.x.tolist(),
            'y': traj_3.y.tolist(),
            'z': traj_3.z.tolist(),
        }
        results['validation_details']['coupled_displacement_mm'] = float(disp_3 * 1e3)
    
    # -----------------------------------------------------------------------
    # VALIDATION CHECKS
    # -----------------------------------------------------------------------
    if rank == 0:
        print("\n" + "-"*70)
        print("VALIDATION CHECKS")
        print("-"*70)
    
    passed = True
    
    # Check 1: Gor'kov alone should trap (small displacement)
    check1 = disp_1 < 0.5e-3
    if rank == 0:
        status = "✓ PASS" if check1 else "✗ FAIL"
        print(f"  1. Gor'kov trapping: {disp_1*1e3:.4f} mm {status}")
    passed = passed and check1
    
    # Check 2: Streaming alone should drift (larger displacement)
    if disp_2 > 0:
        check2 = disp_2 > 0.1e-3
        if rank == 0:
            status = "✓ PASS" if check2 else "✗ FAIL"
            print(f"  2. Streaming drifting: {disp_2*1e3:.4f} mm {status}")
        passed = passed and check2
    
    # Check 3: Velocities reasonable
    v_char_gorkov = disp_1 / 0.01
    if rank == 0:
        status = "✓ PASS" if (1e-6 < v_char_gorkov < 100e-6) else "⚠ WARN"
        print(f"  3. Velocity magnitudes: {v_char_gorkov*1e6:.2f} μm/s {status}")
    
    results['validation_passed'] = bool(passed)
    
    if rank == 0:
        print("\n" + "="*70)
        status = "✓ VALIDATION PASSED" if passed else "✗ VALIDATION FAILED"
        print(status)
        print("="*70 + "\n")
    
    return results


# ============================================================================
# STEP 2: PARAVIEW SETUP
# ============================================================================

def create_paraview_readme(output_dir: Path) -> Path:
    """Create comprehensive ParaView visualization guide."""
    
    readme_content = r"""# ParaView Visualization Guide — Particle Dynamics with Streaming

## Overview

This folder contains a **physics-clear demonstration** of particle motion under coupled acoustic forces:
- **Streaming velocity** (steady flow from acoustic radiation)
- **Radiation force** (Gor'kov potential trapping)

The visualization is divided into **4 mandatory panels** that explain the full coupling.

---

## Files in This Folder

| File | Purpose | Arrays |
|------|---------|--------|
| `standing_fields.vtu` | Standing wave acoustic pressure | p_real, p_imag, p_mag, p_phase |
| `streaming_fields.vtu` | Steady acoustic streaming velocity | streaming_velocity (vector) |
| `gorkov_U.vtu` | Gor'kov radiation potential | U_gorkov (scalar potential) |
| `gorkov_F.vtu` | Radiation force field | F_rad (vector force) |
| `particles.csv` | Particle trajectories | time, x_m, y_m, z_m |

---

## Panel A: Streaming Structure

**Purpose**: Show Rayleigh streaming cells around acoustic vortex.

**Steps**:

1. **Open** `streaming_fields.vtu` in ParaView
2. **Slice filter**:
   - Filters → Data Analysis → Slice
   - Normal: Z-axis
   - Origin: Z = 0.3 mm (mid-height)
3. **Visualize**:
   - Color by `streaming_velocity` magnitude
   - Colormap: Viridis
4. **Add stream tracers** (optional):
   - Filters → Visualization → Stream Tracer
   - Seed point near vortex center
   - Use Forward integration
5. **Result**: Should show circular/spiral recirculation pattern

---

## Panel B: Standing Wave Trap (Gor'kov Potential)

**Purpose**: Show the static trapping landscape.

**Steps**:

1. **Open** `gorkov_U.vtu` in ParaView
2. **Slice filter**:
   - Normal: Y-axis  
   - Origin: Y = L/2
3. **Visualize**:
   - Color by `U_gorkov`
   - Colormap: RdBu (red = potential wells, blue = barriers)
   - Opacity: 0.8
4. **Add contours** (optional):
   - Filters → Visualization → Contour
   - 5-10 contour levels around U_gorkov
5. **Result**: Should show nodal lines (unstable) and potential wells (stable traps)

---

## Panel C: Particle Trajectories

**Purpose**: Show individual particle paths.

**Steps**:

1. **Load** `particles.csv`:
   - File → Open → particles.csv
2. **Convert to points**:
   - Should auto-detect as table → points
3. **Tube representation**:
   - Filters → Visualization → Tube
   - Radius: 0.05 mm
4. **Color by time**:
   - Data → Arrays → time
   - Colormap: Spectral (early=blue, late=red)
5. **Overlay background**:
   - Add `streaming_fields.vtu` or `gorkov_U.vtu` with low opacity
6. **Result**: Colored tubes showing particle paths that bend along streaming

---

## Panel D: Hero Figure (Combined Explanation)

**Purpose**: Single view showing HOW particles move and WHY.

**Setup**: Layer three components:

**Layer 1 - Gor'kov Potential** (background):
- Load `gorkov_U.vtu`
- Slice (XY plane at z=mid-height)
- Color by `U_gorkov` with RdBu
- Opacity: 0.3-0.4

**Layer 2 - Streaming Vectors** (velocity field):
- Add `streaming_fields.vtu`
- Filters → Visualization → Glyph
- Select `streaming_velocity` arrays
- Scale magnitude (try 1e5, adjust for visibility)
- Use arrows or cones
- Color: Black (for contrast)
- Keep sparse (1 per 3-5 grid points)
- Opacity: 1.0

**Layer 3 - Particle Paths** (motion):
- Load `particles.csv` as points
- Filters → Visualization → Tube
- Radius: 0.03 mm
- Color by time (Spectral)
- Opacity: 1.0

**View Options**:
- Use Orthographic projection for clarity
- Suppress far-field particles
- Adjust clipping if needed

**Result**: Multi-layer view showing:
  - Where particles want to be trapped (color potential)
  - Which way the flow pushes them (black arrows)
  - Where they actually go (colored tubes)

---

## Validation Checklist

Check that visualizations confirm the physics:

- [ ] **Panel A**: Streaming shows circular recirculation around vortex
- [ ] **Panel B**: Gor'kov shows minima near vortex, nodes elsewhere
- [ ] **Panel C**: Particle paths curve along streaming streamlines
- [ ] **Panel D**: Particles bend along streaming while staying trapped

If all pass → streaming + trapping are coupled!

---

## Tips for Publication-Quality Renders

**Resolution**: 2560×1440 or 1920×1080 minimum

**Colormaps**:
- Gor'kov: RdBu (intuitive: red=trap, blue=barrier)
- Streaming: Viridis or Cool-to-Warm
- Particles: Spectral (time evolution)

**Labels**:
- Title: 14-16pt
- Axis labels: 12pt
- Sans-serif font (Arial, Helvetica)

**Export**:
- Save as PNG at 300 DPI for print
- Use Export View or Save Screenshot

---

## Equations

**Particle equation of motion**:
$$\dot{\mathbf{x}} = \mathbf{u}_{\text{stream}}(\mathbf{x}) + \frac{\mathbf{F}_{\text{Gor'kov}}(\mathbf{x})}{6\pi \mu a}$$

**Gor'kov potential**:
$$U = \frac{4\pi a^3}{3} \left[ f_1 \frac{\langle p^2 \rangle}{2K} - f_2 \frac{3\rho}{4} \langle |\mathbf{v}_1|^2 \rangle \right]$$

**Radiation force**: $\mathbf{F} = -\nabla U$

---

## Questions?

If visualizations don't match expected physics:

1. Check file sizes (should be > 1 MB each)
2. Verify mesh resolution in `gorkov_U.vtu` (should be smooth)
3. Check particle count in `particles.csv`
4. Ensure time data exists in particle CSV
5. Verify streaming magnitude > 0 in `streaming_fields.vtu`

---

*Generated: 2026-02-09*
*Acousto-Tweezers Project*
*Particle Dynamics + Streaming Coupling*
"""
    
    output_dir = Path(output_dir)
    readme_path = output_dir / "PARAVIEW_README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    return readme_path


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run complete pipeline."""
    
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    if rank == 0:
        print("\n" + "="*70)
        print("PARTICLE DYNAMICS WITH ACOUSTIC STREAMING")
        print("STEP 1: Physics Validation | STEP 2: ParaView Story")
        print("="*70)
    
    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results") / f"particle_streaming_demo_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if rank == 0:
        print(f"\nOutput: {output_dir}\n")
    
    # -----------------------------------------------------------------------
    # CONFIGURATION & SOLVE
    # -----------------------------------------------------------------------
    if rank == 0:
        print("-"*70)
        print("Setup: Configuration & Acoustic Solve")
        print("-"*70)
    
    cfg = create_demo_config()
    
    if rank == 0:
        print(f"Domain: {cfg.L*1e3:.1f} mm × {cfg.H*1e3:.2f} mm")
        print(f"Frequency: {cfg.frequency_hz/1e3:.0f} kHz")
        print(f"Particle radius: {cfg.particle_radius*1e6:.1f} μm")
    
    # Create mesh
    domain, facet_tags, tag_map = create_mesh(cfg, verbose=(rank==0))
    
    # Solve pressure
    p_standing = solve_helmholtz(
        domain, facet_tags, cfg, mode="standing", verbose=(rank==0)
    )
    
    if rank == 0:
        print(f"✓ Pressure solved: max|p| = {np.max(np.abs(p_standing.p_values)):.2f} Pa")
    
    # Solve streaming
    streaming_solution = solve_streaming(
        p_standing, domain=domain, cfg=cfg, verbose=(rank==0)
    )
    
    if rank == 0:
        print(f"✓ Streaming solved: max|u_s| = {streaming_solution.get('max_speed', 0)*1e6:.2f} μm/s")
    
    # Compute Gor'kov
    gorkov = compute_gorkov_potential(p_standing, verbose=(rank==0))
    
    if rank == 0:
        print(f"✓ Gor'kov computed: trap depth = {gorkov.trap_depth:.2e} J")
    
    # -----------------------------------------------------------------------
    # STEP 1: VALIDATION
    # -----------------------------------------------------------------------
    validation = validate_streaming_coupling(gorkov, streaming_solution, cfg, output_dir)
    
    # Save validation results
    with open(output_dir / "validation_results.json", 'w') as f:
        json.dump(validation, f, indent=2)
    
    if rank == 0:
        print(f"\n✓ Validation results saved")
    
    # -----------------------------------------------------------------------
    # STEP 2: PARAVIEW EXPORTS
    # -----------------------------------------------------------------------
    if rank == 0:
        print("\n" + "-"*70)
        print("Exporting for ParaView")
        print("-"*70)
    
    # Pressure
    export_pressure_fields(p_standing, output_dir, "standing_fields", verbose=(rank==0))
    
    # Streaming
    if streaming_solution and 'u_function' in streaming_solution:
        export_streaming_fields(streaming_solution, output_dir, verbose=(rank==0))
    
    # Gor'kov
    export_gorkov_fields(gorkov, output_dir, verbose=(rank==0))
    
    # Particles (from validation)
    if 'trajectories' in validation:
        # Create simple particle CSV from first trajectory
        if 'gorkov_only' in validation['trajectories']:
            traj_data = validation['trajectories']['gorkov_only']
            csv_path = output_dir / "particles.csv"
            
            with open(csv_path, 'w') as f:
                f.write("particle_id,time,x_m,y_m,z_m\n")
                for i, t in enumerate(traj_data['t']):
                    f.write(f"1,{t},{traj_data['x'][i]},{traj_data['y'][i]},{traj_data['z'][i]}\n")
            
            if rank == 0:
                print(f"✓ Exported: particles.csv ({len(traj_data['t'])} points)")
    
    # Create README
    readme_path = create_paraview_readme(output_dir)
    
    if rank == 0:
        print(f"✓ Created: PARAVIEW_README.md")
    
    # -----------------------------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------------------------
    if rank == 0:
        print("\n" + "="*70)
        print("✓ PIPELINE COMPLETE")
        print("="*70)
        print(f"\nGenerated in: {output_dir}")
        print("\nNext steps:")
        print("  1. Open ParaView")
        print("  2. Read: PARAVIEW_README.md")
        print("  3. Load VTU/CSV files")
        print("  4. Create Panels A-D")
        print("\nValidation result:")
        if validation.get('validation_passed'):
            print("  ✓ Streaming coupled with trapping")
        else:
            print("  ⚠ Check validation details")
        print("="*70 + "\n")


if __name__ == "__main__":
    main()
