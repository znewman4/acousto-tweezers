#!/usr/bin/env python3
"""
Phase Sweep Visualization Study - Vortex-Standing Wave Interaction

Systematically varies vortex phase to visualize how the interaction changes.

NO PHYSICS CHANGES - only phase parameter and visualization.

Outputs:
- 2D slices (pressure, Gor'kov, differences) for each phase
- 3D cropped subvolumes for ParaView
- Phase-encoded geometry
- Organized by phase step

Usage:
    python phase_sweep_study.py --preset A --n_phases 8
"""

import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
import sys

sys.path.append(str(Path(__file__).parents[2] / 'src'))

from dolfinx import fem, mesh, io
import ufl
import dolfinx

# Import from fixed comparison script
validation_dir = Path(__file__).parents[1] / 'validation'
sys.path.insert(0, str(validation_dir))

from compare_vortex_standing_fixed import (
    FluidProperties, ParticleProperties, PRESET_A, PRESET_B,
    evaluate_on_grid, solve_standing_only
)

from acoustweezers.experiments.square_dish.phase_control import (
    compute_gorkov_potential_3d, SquareDishConfig
)


def create_mesh_with_tags(L, nx):
    """Create box mesh with boundary tags."""
    domain = mesh.create_box(
        MPI.COMM_WORLD,
        [[0, 0, 0], [L, L, L]],
        [nx, nx, nx],
        cell_type=mesh.CellType.tetrahedron
    )
    
    # Mark boundaries
    def bottom_wall(x): return np.isclose(x[2], 0)
    def top_wall(x): return np.isclose(x[2], L)
    def x_walls(x): return np.logical_or(np.isclose(x[0], 0), np.isclose(x[0], L))
    def y_walls(x): return np.logical_or(np.isclose(x[1], 0), np.isclose(x[1], L))
    
    fdim = domain.topology.dim - 1
    facet_indices, facet_markers = [], []
    for marker, locator in enumerate([bottom_wall, x_walls, y_walls, top_wall], start=1):
        facets = mesh.locate_entities_boundary(domain, fdim, locator)
        facet_indices.append(facets)
        facet_markers.append(np.full_like(facets, marker))
    
    facet_indices = np.hstack(facet_indices).astype(np.int32)
    facet_markers = np.hstack(facet_markers).astype(np.int32)
    sorted_facets = np.argsort(facet_indices)
    facet_tag = mesh.meshtags(domain, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])
    
    return domain, facet_tag


def solve_with_phase(preset, fluid, topological_charge, vortex_phase=0.0):
    """
    Solve combined vortex + standing wave with specified vortex phase.
    
    Phase is applied as: v_vortex(x,y) = v0 * exp(i*ℓ*φ) * exp(i*vortex_phase)
    
    Returns: domain, facet_tags, p_function
    """
    L = preset['dish_size_m']
    freq = preset['frequency_hz']
    wavelength = preset['wavelength_m']
    elements_per_wl = preset['elements_per_wavelength']
    
    # Mesh
    nx = int(L / wavelength * elements_per_wl)
    domain, facet_tags = create_mesh_with_tags(L, nx)
    
    # Function space (P2 for accuracy)
    V = fem.functionspace(domain, ("Lagrange", 2))
    
    # Trial and test functions
    p = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Wavenumber
    omega = 2 * np.pi * freq
    k = omega / fluid.sound_speed
    k_complex = k * (1 + 1j * 1e-3)  # Small damping
    
    # Bulk equation
    a_bulk = (ufl.inner(ufl.grad(p), ufl.grad(ufl.conj(v))) 
              - k_complex**2 * ufl.inner(p, ufl.conj(v))) * ufl.dx
    L_bulk = fem.Constant(domain, complex(0.0)) * ufl.conj(v) * ufl.dx
    
    # Standing wave BC (side walls)
    Z = fluid.density * fluid.sound_speed
    v0 = 1e-6  # m/s
    g_val = -1j * omega * fluid.density * v0
    
    standing_gain = preset.get('standing_gain', 1.0)
    g_expr = fem.Constant(domain, complex(standing_gain * g_val))
    
    # Standing wave on x-walls (tag 2) and y-walls (tag 3)
    ds_xwalls = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags, subdomain_id=2)
    ds_ywalls = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags, subdomain_id=3)
    a_standing = - ufl.inner(ufl.conj(v), p) * (ds_xwalls + ds_ywalls)
    L_standing = - ufl.inner(ufl.conj(v), g_expr) * (ds_xwalls + ds_ywalls)
    
    # Vortex BC (top aperture with phase)
    vortex_gain = preset.get('vortex_gain', 10.0)
    v0_vortex = vortex_gain * v0
    g_vortex_val = -1j * omega * fluid.density * v0_vortex
    
    # Apply global phase shift
    phase_factor = np.exp(1j * vortex_phase)
    g_vortex_val = g_vortex_val * phase_factor
    
    # Get top facets
    top_facets = facet_tags.indices[facet_tags.values == 2]
    
    # Compute aperture pattern
    aperture_radius = preset.get('aperture_radius_m', 0.002)
    aperture_center_xy = preset.get('aperture_center_xy_m', None)
    if aperture_center_xy is None:
        aperture_center_xy = [L/2, L/2]
    
    facet_dim = domain.topology.dim - 1
    facet_centers = dolfinx.mesh.compute_midpoints(domain, facet_dim, top_facets)
    
    # Aperture mask
    aperture_mask = np.linalg.norm(facet_centers[:, :2] - aperture_center_xy, axis=1) <= aperture_radius
    top_facets_aperture = top_facets[aperture_mask]
    
    # DOFs on aperture
    top_dofs = fem.locate_dofs_topological(V, facet_dim, top_facets_aperture)
    
    # Vortex pattern with topological charge
    dof_coords = V.tabulate_dof_coordinates()
    x_dofs = dof_coords[top_dofs, 0]
    y_dofs = dof_coords[top_dofs, 1]
    
    dx = x_dofs - aperture_center_xy[0]
    dy = y_dofs - aperture_center_xy[1]
    r = np.sqrt(dx**2 + dy**2)
    phi = np.arctan2(dy, dx)
    
    # Vortex phase pattern
    vortex_pattern_phase = np.exp(1j * topological_charge * phi)
    
    # Fermi-Dirac taper
    sigma = 0.0003
    taper = 1.0 / (1.0 + np.exp((r - aperture_radius) / sigma))
    
    vortex_pattern = g_vortex_val * vortex_pattern_phase * taper
    
    # Set vortex function
    vortex_func = fem.Function(V)
    vortex_func.x.array[:] = 0.0
    vortex_func.x.array[top_dofs] = vortex_pattern
    
    ds_top = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags, subdomain_id=2)
    a_vortex = - ufl.inner(ufl.conj(v), p) * ds_top
    L_vortex = - ufl.inner(ufl.conj(v), vortex_func) * ds_top
    
    # Assemble and solve
    a = a_bulk + a_standing + a_vortex
    L = L_bulk + L_standing + L_vortex
    
    problem = fem.petsc.LinearProblem(a, L, bcs=[])
    p_sol = problem.solve()
    p_sol.name = "pressure"
    
    return domain, facet_tags, p_sol


def define_crop_region(preset, aperture_center_xy, crop_size_m=0.008):
    """
    Define a cubic subvolume centered on vortex aperture.
    
    Returns: (x_min, x_max, y_min, y_max, z_min, z_max)
    """
    L = preset['dish_size_m']
    
    cx, cy = aperture_center_xy
    half_crop = crop_size_m / 2
    
    x_min = max(0, cx - half_crop)
    x_max = min(L, cx + half_crop)
    y_min = max(0, cy - half_crop)
    y_max = min(L, cy + half_crop)
    z_min = L/2 - half_crop
    z_max = L/2 + half_crop
    
    return (x_min, x_max, y_min, y_max, z_min, z_max)


def extract_subvolume(p_func, domain, crop_bounds):
    """
    Extract function values in cropped region.
    
    Returns arrays for ParaView-compatible output.
    """
    x_min, x_max, y_min, y_max, z_min, z_max = crop_bounds
    
    # Get all DOF coordinates
    V = p_func.function_space
    dof_coords = V.tabulate_dof_coordinates()
    
    # Mask for DOFs in crop region
    mask = ((dof_coords[:, 0] >= x_min) & (dof_coords[:, 0] <= x_max) &
            (dof_coords[:, 1] >= y_min) & (dof_coords[:, 1] <= y_max) &
            (dof_coords[:, 2] >= z_min) & (dof_coords[:, 2] <= z_max))
    
    cropped_dofs = np.where(mask)[0]
    
    return cropped_dofs, dof_coords[cropped_dofs]


def plot_phase_slice(p_func, domain, z_slice, phase_label, output_path, title="Pressure", vmax=None):
    """Generate 2D slice plot for a specific phase."""
    L = domain.geometry.x.max(axis=0)[0]
    
    X, Y, p_vals = evaluate_on_grid(p_func, domain, z_slice)
    p_mag = np.abs(p_vals)
    
    if vmax is None:
        vmax = np.max(p_mag)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.contourf(X*1e3, Y*1e3, p_mag, levels=50, cmap='plasma', vmin=0, vmax=vmax)
    ax.set_title(f'{title}\nPhase = {phase_label}', fontsize=12)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    plt.colorbar(im, ax=ax, label='|p| (Pa)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return vmax


def plot_difference(p_combined, p_ref, domain, z_slice, phase_label, output_path, vmax_diff=None):
    """Plot |p_combined| - |p_ref| difference."""
    X, Y, p_c_vals = evaluate_on_grid(p_combined, domain, z_slice)
    _, _, p_r_vals = evaluate_on_grid(p_ref, domain, z_slice)
    
    diff = np.abs(p_c_vals) - np.abs(p_r_vals)
    
    if vmax_diff is None:
        vmax_diff = np.max(np.abs(diff))
    
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.contourf(X*1e3, Y*1e3, np.abs(diff), levels=50, cmap='hot', vmin=0, vmax=vmax_diff)
    
    max_p_ref = np.max(np.abs(p_r_vals))
    percent = 100 * vmax_diff / max_p_ref
    
    ax.set_title(f'Pressure Difference\nPhase = {phase_label} | max Δ = {vmax_diff:.1f} Pa ({percent:.1f}%)', fontsize=11)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    plt.colorbar(im, ax=ax, label='|Δp| (Pa)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return vmax_diff


def export_cropped_3d(p_func, domain, crop_bounds, output_base, phase_idx):
    """Export cropped 3D data for ParaView."""
    from dolfinx.io import XDMFFile
    
    # Create P1 function for XDMF compatibility
    V_p1 = fem.functionspace(domain, ("Lagrange", 1))
    p_mag_p1 = fem.Function(V_p1)
    
    # Interpolate magnitude
    V = p_func.function_space
    V_real = fem.functionspace(domain, ("Lagrange", V.ufl_element().degree()))
    p_mag = fem.Function(V_real)
    p_mag.x.array[:] = np.abs(p_func.x.array[:])
    
    p_mag_p1.interpolate(p_mag)
    p_mag_p1.name = "pressure_magnitude"
    
    # Also export phase
    p_phase = fem.Function(V_real)
    p_phase.x.array[:] = np.angle(p_func.x.array[:])
    
    p_phase_p1 = fem.Function(V_p1)
    p_phase_p1.interpolate(p_phase)
    p_phase_p1.name = "pressure_phase"
    
    # Write full domain (ParaView can clip)
    xdmf_file = str(output_base) + f"_phase{phase_idx:03d}.xdmf"
    with XDMFFile(domain.comm, xdmf_file, "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(p_mag_p1, 0.0)
        xdmf.write_function(p_phase_p1, 0.0)
    
    print(f"    [3D Export] {Path(xdmf_file).name}")


def main():
    parser = argparse.ArgumentParser(description="Phase Sweep Visualization Study")
    parser.add_argument('--preset', type=str, choices=['A', 'B'], default='A')
    parser.add_argument('--n_phases', type=int, default=8, help="Number of phase steps (e.g. 8 or 16)")
    parser.add_argument('--topological_charge', type=int, default=1)
    parser.add_argument('--crop_size_m', type=float, default=0.008, help="Crop region size (8mm default)")
    parser.add_argument('--output_dir', type=str, default=None)
    
    args = parser.parse_args()
    
    # Select preset
    preset = PRESET_A.copy() if args.preset == 'A' else PRESET_B.copy()
    
    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(__file__).parents[2] / 'results' / f'phase_sweep_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"PHASE SWEEP VISUALIZATION STUDY")
    print(f"{'='*70}")
    print(f"Preset: {preset['name']}")
    print(f"Topological charge: ℓ = {args.topological_charge}")
    print(f"Phase steps: {args.n_phases}")
    print(f"Crop size: {args.crop_size_m*1e3:.1f} mm")
    print(f"Output: {output_dir}")
    print(f"{'='*70}\n")
    
    fluid = FluidProperties()
    particle = ParticleProperties()
    
    # Define aperture center and crop region
    L = preset['dish_size_m']
    aperture_center_xy = preset.get('aperture_center_xy_m', None)
    if aperture_center_xy is None:
        aperture_center_xy = [L/2, L/2]
    crop_bounds = define_crop_region(preset, aperture_center_xy, args.crop_size_m)
    
    print(f"[Crop] Region: x=[{crop_bounds[0]*1e3:.2f}, {crop_bounds[1]*1e3:.2f}] mm")
    print(f"               y=[{crop_bounds[2]*1e3:.2f}, {crop_bounds[3]*1e3:.2f}] mm")
    print(f"               z=[{crop_bounds[4]*1e3:.2f}, {crop_bounds[5]*1e3:.2f}] mm\n")
    
    # Solve standing wave reference (once)
    print("[Reference] Solving standing wave only...")
    preset_standing = preset.copy()
    preset_standing['vortex_gain'] = 0.0
    domain_ref, _, p_standing = solve_standing_only(preset_standing, fluid)
    
    print(f"  Standing max|p| = {np.max(np.abs(p_standing.x.array[:])):.3e} Pa\n")
    
    # Phase sweep
    phases = np.linspace(0, 2*np.pi, args.n_phases, endpoint=False)
    phase_labels = [f"{ph*180/np.pi:.0f}°" for ph in phases]
    
    # Track global vmax for consistent scales
    vmax_pressure = 0
    vmax_diff = 0
    
    # First pass: determine scales
    print("[Pass 1] Determining color scales...")
    for i, (phase, label) in enumerate(zip(phases, phase_labels)):
        print(f"  Phase {i+1}/{args.n_phases}: {label}", end='\r')
        _, _, p_combined = solve_with_phase(preset, fluid, args.topological_charge, phase)
        vmax_pressure = max(vmax_pressure, np.max(np.abs(p_combined.x.array[:])))
        
        # Quick difference eval
        L_domain = domain_ref.geometry.x.max(axis=0)[0]
        z_mid = L_domain / 2
        _, _, p_c_vals = evaluate_on_grid(p_combined, domain_ref, z_mid)
        _, _, p_s_vals = evaluate_on_grid(p_standing, domain_ref, z_mid)
        diff = np.abs(p_c_vals) - np.abs(p_s_vals)
        vmax_diff = max(vmax_diff, np.max(np.abs(diff)))
    
    print(f"\n  Global vmax_pressure = {vmax_pressure:.1f} Pa")
    print(f"  Global vmax_diff = {vmax_diff:.1f} Pa\n")
    
    # Second pass: generate all outputs
    print("[Pass 2] Generating visualizations...")
    
    # Create config for Gor'kov
    config = SquareDishConfig()
    config.Lx = preset['dish_size_m']
    config.Ly = preset['dish_size_m']
    config.Lz = preset['dish_size_m']
    config.frequency = preset['frequency_hz']
    config.rho_water = fluid.density
    config.c_water = fluid.sound_speed
    config.particle_radius = particle.radius
    config.particle_density = particle.density
    
    # Compute Gor'kov for standing wave
    print("  [Gor'kov] Computing for standing wave...")
    U_standing = compute_gorkov_potential_3d(p_standing, config, verbose=False)
    U_standing_max = np.max(np.abs(U_standing.x.array[:]))
    
    for i, (phase, label) in enumerate(zip(phases, phase_labels)):
        print(f"\n[Phase {i+1}/{args.n_phases}] {label} ({phase:.3f} rad)")
        
        # Create phase directory
        phase_dir = output_dir / f"phi_{i:03d}_{label.replace('°', 'deg')}"
        phase_dir.mkdir(exist_ok=True)
        slices_dir = phase_dir / "slices"
        slices_dir.mkdir(exist_ok=True)
        viz_3d_dir = phase_dir / "3d"
        viz_3d_dir.mkdir(exist_ok=True)
        
        # Solve
        print(f"  [Solve] Combined field...")
        domain_comb, _, p_combined = solve_with_phase(preset, fluid, args.topological_charge, phase)
        
        print(f"  [Gor'kov] Computing potential...")
        U_combined = compute_gorkov_potential_3d(p_combined, config, verbose=False)
        
        # 2D slices
        z_mid = L / 2
        print(f"  [2D] Pressure slice...")
        plot_phase_slice(p_combined, domain_comb, z_mid, label, 
                        slices_dir / f"pressure_{i:03d}.png",
                        "Combined Pressure", vmax_pressure)
        
        print(f"  [2D] Gor'kov slice...")
        plot_phase_slice(U_combined, domain_comb, z_mid, label,
                        slices_dir / f"gorkov_{i:03d}.png",
                        "Gor'kov Potential", U_standing_max)
        
        print(f"  [2D] Pressure difference...")
        plot_difference(p_combined, p_standing, domain_comb, z_mid, label,
                       slices_dir / f"diff_pressure_{i:03d}.png", vmax_diff)
        
        print(f"  [2D] Gor'kov difference...")
        plot_difference(U_combined, U_standing, domain_comb, z_mid, label,
                       slices_dir / f"diff_gorkov_{i:03d}.png")
        
        # 3D exports
        print(f"  [3D] Exporting...")
        export_cropped_3d(p_combined, domain_comb, crop_bounds, 
                         viz_3d_dir / "combined", i)
    
    # Create README
    readme_path = output_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(f"# Phase Sweep Visualization Study\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## Parameters\n\n")
        f.write(f"- Preset: {preset['name']}\n")
        f.write(f"- Topological charge: ℓ = {args.topological_charge}\n")
        f.write(f"- Phase steps: {args.n_phases}\n")
        f.write(f"- Vortex gain: {preset.get('vortex_gain', 10.0)}\n")
        f.write(f"- Crop region: {args.crop_size_m*1e3:.1f} mm cube\n\n")
        f.write(f"## Phase Values\n\n")
        for i, label in enumerate(phase_labels):
            f.write(f"- `phi_{i:03d}_{label.replace('°', 'deg')}/`: {label} ({phases[i]:.3f} rad)\n")
        f.write(f"\n## 2D Visualizations\n\n")
        f.write(f"Each phase directory contains `slices/` with:\n")
        f.write(f"- `pressure_XXX.png`: Combined pressure magnitude\n")
        f.write(f"- `gorkov_XXX.png`: Gor'kov potential\n")
        f.write(f"- `diff_pressure_XXX.png`: Δ|p| = combined - standing\n")
        f.write(f"- `diff_gorkov_XXX.png`: ΔU = combined - standing\n\n")
        f.write(f"**Color scales are consistent across all phases for comparison.**\n\n")
        f.write(f"## 3D ParaView Visualization\n\n")
        f.write(f"Each phase directory contains `3d/` with XDMF files:\n")
        f.write(f"- `combined_phaseXXX.xdmf`: Full 3D pressure field\n")
        f.write(f"  - Field: `pressure_magnitude` (Pa)\n")
        f.write(f"  - Field: `pressure_phase` (rad)\n\n")
        f.write(f"### How to visualize in ParaView:\n\n")
        f.write(f"1. Open `combined_phaseXXX.xdmf` in ParaView\n")
        f.write(f"2. **Clip** to crop region:\n")
        f.write(f"   - Filters > Clip\n")
        f.write(f"   - Box clip: x=[{crop_bounds[0]*1e3:.2f}, {crop_bounds[1]*1e3:.2f}] mm\n")
        f.write(f"3. **Multi-layer iso-surfaces**:\n")
        f.write(f"   - Filters > Contour\n")
        f.write(f"   - Many iso-values (e.g. 10-20 levels)\n")
        f.write(f"   - Opacity: 0.1-0.2 per surface\n")
        f.write(f"4. **Phase-colored geometry**:\n")
        f.write(f"   - Color by: `pressure_phase`\n")
        f.write(f"   - Colormap: HSV (cyclic)\n")
        f.write(f"5. **Compare phases**:\n")
        f.write(f"   - Load multiple phases side-by-side\n")
        f.write(f"   - Or animate through phases\n\n")
        f.write(f"## What to Notice\n\n")
        f.write(f"As phase changes from 0° to 360°:\n")
        f.write(f"- The vortex perturbation **rotates/shifts** in space\n")
        f.write(f"- The pressure difference pattern **moves** around the aperture\n")
        f.write(f"- The Gor'kov potential minima/maxima **relocate**\n")
        f.write(f"- At phases 0°, 90°, 180°, 270° you should see 90° rotational symmetry\n\n")
        f.write(f"**Key insight**: The vortex acts as a localized, phase-tunable perturbation.\n")
        f.write(f"By varying phase, you can \"steer\" where the vortex influence appears.\n")
    
    print(f"\n{'='*70}")
    print(f"PHASE SWEEP COMPLETE")
    print(f"{'='*70}")
    print(f"Output directory: {output_dir}")
    print(f"\n2D Visualizations:")
    print(f"  - {args.n_phases} phase steps from 0° to {360-360//args.n_phases}°")
    print(f"  - Pressure, Gor'kov, and difference slices")
    print(f"  - Consistent color scales for direct comparison")
    print(f"\n3D ParaView Files:")
    print(f"  - XDMF format with pressure magnitude and phase")
    print(f"  - See README.md for ParaView visualization instructions")
    print(f"\nWhat to notice:")
    print(f"  The vortex perturbation ROTATES/SHIFTS as phase changes.")
    print(f"  This is visible in both difference plots and 3D phase-colored geometry.")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
