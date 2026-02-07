#!/usr/bin/env python3
"""
Scale-Calibrated Vortex + Standing Wave Superposition Comparison - FIXED

CRITICAL FIX: Vortex boundary term now properly enters weak form
- Proper velocity amplitude scaling
- Diagnostics verify non-zero BC contribution
- Coherent complex field superposition

Three-case comparison study:
1. Standing wave only (side walls)
2. Vortex only (top aperture)
3. Coherent superposition (both)
"""

import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
from dolfinx import fem, mesh
import ufl

import sys
sys.path.append(str(Path(__file__).parents[2] / 'src'))

from acoustweezers.physics.acoustics.vortex_lens import VortexLensConfig, compute_azimuthal_phase, compute_amplitude_profile
from acoustweezers.experiments.square_dish.phase_control import SquareDishConfig, compute_gorkov_potential_3d


# Material properties
class FluidProperties:
    def __init__(self):
        self.density = 997.0  # kg/m³
        self.sound_speed = 1484.0  # m/s
        self.viscosity = 0.001  # Pa·s


# Particle properties (polystyrene in water)
class ParticleProperties:
    def __init__(self):
        self.radius = 5e-6  # m
        self.density = 1050.0  # kg/m³
        self.f1 = 0.464  # Monopole coefficient
        self.f2 = 0.034  # Dipole coefficient


# =============================================================================
# DISH SIZE PRESETS
# =============================================================================

PRESET_A = {
    'name': 'Preset A: 2cm dish, 500 kHz',
    'dish_size_m': 0.02,
    'frequency_hz': 500e3,
    'wavelength_m': 2.968e-3,  # c=1484 m/s water
    'aperture_radius_m': 0.002,  # 2mm
    'aperture_center_xy_m': None,  # None = dish center, or [x, y]
    'vortex_gain': 10.0,  # Velocity amplitude multiplier - INCREASED for visibility
    'standing_gain': 1.0,
    'elements_per_wavelength': 6,
}

PRESET_B = {
    'name': 'Preset B: 3mm dish, 2 MHz',
    'dish_size_m': 0.003,
    'frequency_hz': 2.0e6,
    'wavelength_m': 0.742e-3,
    'aperture_radius_m': 0.0007,  # 0.7mm
    'aperture_center_xy_m': None,
    'vortex_gain': 1.0,
    'standing_gain': 1.0,
    'elements_per_wavelength': 6,
}


# =============================================================================
# SOLVER FUNCTIONS
# =============================================================================

def solve_standing_only(preset: dict, fluid: FluidProperties):
    """Solve standing wave (side walls actuation only)."""
    
    L = preset['dish_size_m']
    freq = preset['frequency_hz']
    omega = 2 * np.pi * freq
    k = omega / fluid.sound_speed
    gain = preset['standing_gain']
    
    # Create mesh
    nx = int(L / preset['wavelength_m'] * preset['elements_per_wavelength'])
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
    
    # Function space (complex)
    V = fem.functionspace(domain, ("Lagrange", 2))
    
    # Standing wave actuation
    v0 = 1e-6 * gain  # 1 μm/s velocity amplitude
    
    # Impedance BC parameter
    Z = fluid.density * fluid.sound_speed
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tag)
    
    # Weak form
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Volume terms
    a = (ufl.inner(ufl.grad(u), ufl.grad(v)) / fluid.density - 
         k**2 * ufl.inner(u, v) / fluid.density) * ufl.dx
    
    # Impedance BCs on all boundaries
    for tag in [1, 2, 3, 4]:
        a += -1j * omega * ufl.inner(u, v) / Z * ds(tag)
    
    # Actuation on x-walls (tag 2) and y-walls (tag 3)
    g_standing = -1j * omega * fluid.density * v0
    L_form = g_standing * ufl.inner(1.0, v) * (ds(2) + ds(3))
    
    # Solve
    problem = fem.petsc.LinearProblem(a, L_form, bcs=[], petsc_options={
        "ksp_type": "gmres",
        "ksp_rtol": 1e-8,
        "ksp_max_it": 2000,
        "pc_type": "ilu"
    })
    p = problem.solve()
    
    # Diagnostics
    max_p = np.max(np.abs(p.x.array[:]))
    max_bc = np.abs(g_standing)
    print(f"  [Diagnostics] max|p| = {max_p:.3e} Pa")
    print(f"  [Diagnostics] Standing BC |g| = {max_bc:.3e} Pa·s/m")
    
    return domain, facet_tag, p


def solve_vortex_only(preset: dict, fluid: FluidProperties, topological_charge: int = 1):
    """Solve vortex lens (top aperture only)."""
    
    L = preset['dish_size_m']
    freq = preset['frequency_hz']
    omega = 2 * np.pi * freq
    k = omega / fluid.sound_speed
    gain = preset['vortex_gain']
    
    # Create mesh
    nx = int(L / preset['wavelength_m'] * preset['elements_per_wavelength'])
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
    
    # Function space (complex)
    V = fem.functionspace(domain, ("Lagrange", 2))
    
    # Vortex configuration
    if preset.get('aperture_center_xy_m') is not None:
        vortex_center = np.array([preset['aperture_center_xy_m'][0], 
                                  preset['aperture_center_xy_m'][1], L])
    else:
        vortex_center = np.array([L/2, L/2, L])
    
    vortex_config = VortexLensConfig(
        topological_charge=topological_charge,
        center=tuple(vortex_center),
        amplitude=1e-6 * gain,  # Velocity amplitude m/s
        aperture_radius=preset['aperture_radius_m'],
        apodization='cosine_taper',
        axis='z'
    )
    
    # Get top boundary DOFs
    top_facets = mesh.locate_entities_boundary(domain, fdim, top_wall)
    top_dofs = fem.locate_dofs_topological(V, fdim, top_facets)
    
    # Extract DOF coordinates
    dof_coords = V.tabulate_dof_coordinates()
    top_coords = dof_coords[top_dofs]
    
    # Compute vortex pattern
    phi_vortex = compute_azimuthal_phase(top_coords, vortex_config, vortex_center)
    amp_vortex = compute_amplitude_profile(top_coords, vortex_config, vortex_center)
    vortex_pattern = amp_vortex * np.exp(1j * phi_vortex)
    
    # Create vortex function (zero everywhere except top boundary)
    vortex_func = fem.Function(V)
    vortex_func.x.array[:] = 0.0
    vortex_func.x.array[top_dofs] = vortex_pattern
    
    # Impedance BC
    Z = fluid.density * fluid.sound_speed
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tag)
    
    # Weak form
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Volume terms
    a = (ufl.inner(ufl.grad(u), ufl.grad(v)) / fluid.density - 
         k**2 * ufl.inner(u, v) / fluid.density) * ufl.dx
    
    # Impedance BCs on all boundaries
    for tag in [1, 2, 3, 4]:
        a += -1j * omega * ufl.inner(u, v) / Z * ds(tag)
    
    # Vortex actuation on top (tag 4)
    g_vortex = -1j * omega * fluid.density * vortex_func
    L_form = ufl.inner(g_vortex, v) * ds(4)
    
    # Solve
    problem = fem.petsc.LinearProblem(a, L_form, bcs=[], petsc_options={
        "ksp_type": "gmres",
        "ksp_rtol": 1e-8,
        "ksp_max_it": 2000,
        "pc_type": "ilu"
    })
    p = problem.solve()
    
    # Diagnostics
    max_p = np.max(np.abs(p.x.array[:]))
    max_bc_vortex = np.max(np.abs(vortex_pattern))
    active_dofs = np.sum(np.abs(vortex_pattern) > 1e-12)
    print(f"  [Diagnostics] max|p| = {max_p:.3e} Pa")
    print(f"  [Diagnostics] Vortex BC max|v0*pattern| = {max_bc_vortex:.3e} m/s")
    print(f"  [Diagnostics] Vortex BC active DOFs = {active_dofs} / {len(vortex_pattern)}")
    
    return domain, facet_tag, p


def solve_combined(preset: dict, fluid: FluidProperties, topological_charge: int = 1):
    """Solve coherent superposition (standing + vortex)."""
    
    L = preset['dish_size_m']
    freq = preset['frequency_hz']
    omega = 2 * np.pi * freq
    k = omega / fluid.sound_speed
    v_gain = preset['vortex_gain']
    s_gain = preset['standing_gain']
    
    # Create mesh
    nx = int(L / preset['wavelength_m'] * preset['elements_per_wavelength'])
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
    
    # Function space (complex)
    V = fem.functionspace(domain, ("Lagrange", 2))
    
    # Vortex configuration
    if preset.get('aperture_center_xy_m') is not None:
        vortex_center = np.array([preset['aperture_center_xy_m'][0], 
                                  preset['aperture_center_xy_m'][1], L])
    else:
        vortex_center = np.array([L/2, L/2, L])
    
    vortex_config = VortexLensConfig(
        topological_charge=topological_charge,
        center=tuple(vortex_center),
        amplitude=1e-6 * v_gain,
        aperture_radius=preset['aperture_radius_m'],
        apodization='cosine_taper',
        axis='z'
    )
    
    # Get top boundary DOFs
    top_facets = mesh.locate_entities_boundary(domain, fdim, top_wall)
    top_dofs = fem.locate_dofs_topological(V, fdim, top_facets)
    
    # Extract DOF coordinates
    dof_coords = V.tabulate_dof_coordinates()
    top_coords = dof_coords[top_dofs]
    
    # Compute vortex pattern
    phi_vortex = compute_azimuthal_phase(top_coords, vortex_config, vortex_center)
    amp_vortex = compute_amplitude_profile(top_coords, vortex_config, vortex_center)
    vortex_pattern = amp_vortex * np.exp(1j * phi_vortex)
    
    # Create vortex function
    vortex_func = fem.Function(V)
    vortex_func.x.array[:] = 0.0
    vortex_func.x.array[top_dofs] = vortex_pattern
    
    # Impedance BC
    Z = fluid.density * fluid.sound_speed
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tag)
    
    # Weak form
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Volume terms
    a = (ufl.inner(ufl.grad(u), ufl.grad(v)) / fluid.density - 
         k**2 * ufl.inner(u, v) / fluid.density) * ufl.dx
    
    # Impedance BCs on all boundaries
    for tag in [1, 2, 3, 4]:
        a += -1j * omega * ufl.inner(u, v) / Z * ds(tag)
    
    # Combined actuation: standing on sides + vortex on top
    v0_standing = 1e-6 * s_gain
    g_standing = -1j * omega * fluid.density * v0_standing
    g_vortex = -1j * omega * fluid.density * vortex_func
    
    L_form = (g_standing * ufl.inner(1.0, v) * (ds(2) + ds(3)) +  # Standing on x/y walls
              ufl.inner(g_vortex, v) * ds(4))  # Vortex on top
    
    # Solve
    problem = fem.petsc.LinearProblem(a, L_form, bcs=[], petsc_options={
        "ksp_type": "gmres",
        "ksp_rtol": 1e-8,
        "ksp_max_it": 2000,
        "pc_type": "ilu"
    })
    p = problem.solve()
    
    # Diagnostics
    max_p = np.max(np.abs(p.x.array[:]))
    max_bc_standing = np.abs(g_standing)
    max_bc_vortex = np.max(np.abs(vortex_pattern))
    active_dofs = np.sum(np.abs(vortex_pattern) > 1e-12)
    print(f"  [Diagnostics] max|p| = {max_p:.3e} Pa")
    print(f"  [Diagnostics] Standing BC |g| = {max_bc_standing:.3e} Pa·s/m")
    print(f"  [Diagnostics] Vortex BC max|v0*pattern| = {max_bc_vortex:.3e} m/s")
    print(f"  [Diagnostics] Vortex BC active DOFs = {active_dofs} / {len(vortex_pattern)}")
    
    return domain, facet_tag, p


# =============================================================================
# VISUALIZATION
# =============================================================================

def evaluate_on_grid(func, domain, z_slice, n_points=150):
    """Evaluate function on 2D grid at given z-height."""
    bounds = domain.geometry.x.min(axis=0), domain.geometry.x.max(axis=0)
    x_min, x_max = bounds[0][0], bounds[1][0]
    y_min, y_max = bounds[0][1], bounds[1][1]
    
    x_grid = np.linspace(x_min, x_max, n_points)
    y_grid = np.linspace(y_min, y_max, n_points)
    X, Y = np.meshgrid(x_grid, y_grid)
    points = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, z_slice)])
    
    # Evaluate
    bb_tree = dolfinx.geometry.bb_tree(domain, domain.topology.dim)
    vals = np.zeros(points.shape[0], dtype=np.complex128)
    
    cells = []
    points_on_proc = []
    cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = dolfinx.geometry.compute_colliding_cells(domain, cell_candidates, points)
    
    for i, point in enumerate(points):
        if len(colliding_cells.links(i)) > 0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])
    
    if len(points_on_proc) > 0:
        points_on_proc = np.array(points_on_proc, dtype=np.float64)
        eval_result = func.eval(points_on_proc, cells)
        if eval_result.ndim == 2:
            eval_result = eval_result[:, 0]
        vals[range(len(points_on_proc))] = eval_result
    
    return X, Y, vals.reshape(X.shape)


def plot_difference_field(p_ref, p_combined, domain, output_path, title="Pressure Difference"):
    """Plot |p|_combined - |p|_ref."""
    L = domain.geometry.x.max(axis=0)[0]
    z_mid = L / 2
    
    X, Y, p_ref_vals = evaluate_on_grid(p_ref, domain, z_mid)
    _, _, p_comb_vals = evaluate_on_grid(p_combined, domain, z_mid)
    
    diff = np.abs(p_comb_vals) - np.abs(p_ref_vals)
    
    fig, ax = plt.subplots(figsize=(8, 6.5))
    vmax_diff = np.max(np.abs(diff))
    im = ax.contourf(X*1e3, Y*1e3, np.abs(diff), levels=50, cmap='hot', 
                     vmin=0, vmax=vmax_diff)
    
    # Calculate percentage
    max_p_ref = np.max(np.abs(p_ref_vals))
    percent_diff = 100 * vmax_diff / max_p_ref
    
    ax.set_title(f'{title}\nmax Δ = {vmax_diff:.1f} Pa ({percent_diff:.1f}% of reference)', fontsize=12)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    plt.colorbar(im, ax=ax, label='|Δp| (Pa)')
    
    # Add text with statistics
    mean_diff = np.mean(np.abs(diff))
    textstr = f'max: {vmax_diff:.1f} Pa\nmean: {mean_diff:.1f} Pa\n{percent_diff:.1f}% of reference'
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes,
            verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'))
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[Plot] Saved {output_path}")
    plt.close()
    
    # Report statistics
    print(f"  [Difference Stats] max |Δ|p|| = {vmax_diff:.3e} Pa ({percent_diff:.2f}% of reference)")
    print(f"  [Difference Stats] mean |Δ|p|| = {mean_diff:.3e} Pa")


def plot_comparison_slices(p_standing, p_vortex, p_combined, domain, output_dir):
    """Plot three-panel comparison."""
    L = domain.geometry.x.max(axis=0)[0]
    z_mid = L / 2
    
    X, Y, p_s = evaluate_on_grid(p_standing, domain, z_mid)
    _, _, p_v = evaluate_on_grid(p_vortex, domain, z_mid)
    _, _, p_c = evaluate_on_grid(p_combined, domain, z_mid)
    
    p_s_mag = np.abs(p_s)
    p_v_mag = np.abs(p_v)
    p_c_mag = np.abs(p_c)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Standing wave - full scale
    vmax_s = np.max(p_s_mag)
    im1 = axes[0].contourf(X*1e3, Y*1e3, p_s_mag, levels=50, cmap='plasma', vmin=0, vmax=vmax_s)
    axes[0].set_title(f'Standing Wave Only\nmax|p|={vmax_s:.1f} Pa', fontsize=11)
    axes[0].set_xlabel('x (mm)')
    axes[0].set_ylabel('y (mm)')
    plt.colorbar(im1, ax=axes[0], label='|p| (Pa)')
    
    # Vortex only - separate scale to see structure
    vmax_v = np.max(p_v_mag)
    im2 = axes[1].contourf(X*1e3, Y*1e3, p_v_mag, levels=50, cmap='plasma', vmin=0, vmax=vmax_v)
    axes[1].set_title(f'Vortex Only\nmax|p|={vmax_v:.1f} Pa', fontsize=11)
    axes[1].set_xlabel('x (mm)')
    plt.colorbar(im2, ax=axes[1], label='|p| (Pa)')
    
    # Combined - full scale
    vmax_c = np.max(p_c_mag)
    im3 = axes[2].contourf(X*1e3, Y*1e3, p_c_mag, levels=50, cmap='plasma', vmin=0, vmax=vmax_c)
    axes[2].set_title(f'Combined\nmax|p|={vmax_c:.1f} Pa', fontsize=11)
    axes[2].set_xlabel('x (mm)')
    plt.colorbar(im3, ax=axes[2], label='|p| (Pa)')
    
    plt.tight_layout()
    output_path = output_dir / 'pressure_comparison_slice.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[Plot] Saved {output_path}")
    plt.close()


# =============================================================================
# EXPORT
# =============================================================================

def export_field(p_func, domain, filename_base):
    """Export pressure magnitude to BP4 and XDMF formats."""
    from dolfinx.io import VTXWriter, XDMFFile
    
    # Create P2 real function for magnitude
    V_p2 = p_func.function_space
    V_real_p2 = fem.functionspace(domain, ("Lagrange", 2))
    p_mag_p2 = fem.Function(V_real_p2)
    p_mag_p2.x.array[:] = np.abs(p_func.x.array[:])
    p_mag_p2.name = "pressure_magnitude"
    
    # Export to BP4 (P2 accuracy)
    bp_file = str(filename_base) + ".bp"
    with VTXWriter(domain.comm, bp_file, [p_mag_p2], engine="BP4") as vtx:
        vtx.write(0.0)
    print(f"[Export] Saved {bp_file}")
    
    # Export to XDMF (P1 for PyVista compatibility)
    V_real_p1 = fem.functionspace(domain, ("Lagrange", 1))
    p_mag_p1 = fem.Function(V_real_p1)
    
    # Interpolate P2 -> P1 using DOLFINx interpolation
    p_mag_p1.interpolate(p_mag_p2)
    p_mag_p1.name = "pressure_magnitude"
    
    xdmf_file = str(filename_base) + ".xdmf"
    with XDMFFile(domain.comm, xdmf_file, "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(p_mag_p1, 0.0)
    print(f"[Export] Saved {xdmf_file}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preset', type=str, choices=['A', 'B'], default='A')
    parser.add_argument('--topological_charge', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default=None)
    
    args = parser.parse_args()
    
    preset = PRESET_A if args.preset == 'A' else PRESET_B
    preset['topological_charge'] = args.topological_charge
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(__file__).parents[2] / 'results' / f'comparison_{args.preset}_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"VORTEX + STANDING WAVE COMPARISON (FIXED)")
    print(f"{'='*70}")
    print(f"Preset: {preset['name']}")
    print(f"Topological charge: ℓ = {args.topological_charge}")
    print(f"Output: {output_dir}")
    print(f"{'='*70}\n")
    
    fluid = FluidProperties()
    
    print("[1/3] Solving standing wave only...")
    domain_s, facet_s, p_standing = solve_standing_only(preset, fluid)
    
    print("\n[2/3] Solving vortex only...")
    domain_v, facet_v, p_vortex = solve_vortex_only(preset, fluid, args.topological_charge)
    
    print("\n[3/3] Solving combined...")
    domain_c, facet_c, p_combined = solve_combined(preset, fluid, args.topological_charge)
    
    print("\n[Viz] Generating plots...")
    plot_comparison_slices(p_standing, p_vortex, p_combined, domain_c, output_dir)
    plot_difference_field(p_standing, p_combined, domain_c, 
                         output_dir / 'pressure_difference.png',
                         'Pressure Difference: |p|_combined - |p|_standing')
    
    print("\n[Export] Writing BP4 files...")
    export_field(p_standing, domain_s, output_dir / 'standing_only')
    export_field(p_vortex, domain_v, output_dir / 'vortex_only')
    export_field(p_combined, domain_c, output_dir / 'combined')
    
    print(f"\n{'='*70}")
    print(f"DONE. Results in: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
