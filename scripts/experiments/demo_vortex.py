#!/usr/bin/env python3
"""
Acoustic Vortex Demonstration
==============================

Extends square_dish with vortex actuation on top boundary.
Shows vortex structure, Gor'kov landscape, and particle behavior.

Based on: square_dish/phase_control.py
Added: Vortex lens boundary actuation

Author: Acousto-Tweezers Project
Date: February 2026
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from mpi4py import MPI

# Add project modules
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

# Import square_dish module properly
from acoustweezers.experiments.square_dish.phase_control import (
    SquareDishConfig, create_square_dish_mesh,
    compute_gorkov_potential_3d
)

from petsc4py import PETSc
if not np.issubdtype(PETSc.ScalarType, np.complexfloating):
    raise RuntimeError("Requires complex PETSc")

from dolfinx import fem
from dolfinx.fem import Function, Constant, form, assemble_scalar
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
import ufl
from ufl import inner, grad, dx, ds, TestFunction, TrialFunction

# Import vortex module
from acoustweezers.physics.acoustics.vortex_lens import compute_azimuthal_phase


def solve_vortex_helmholtz(mesh, facet_tags, config, topological_charge=1):
    """
    Solve Helmholtz with vortex actuation on top boundary.
    
    Parameters
    ----------
    mesh, facet_tags : From create_square_dish_mesh
    config : SquareDishConfig
    topological_charge : int
        Vortex winding number ℓ
    """
    comm = mesh.comm
    rank = comm.rank
    
    # Function space
    V = fem.functionspace(mesh, ("Lagrange", 2))
    
    if rank == 0:
        print(f"[Vortex] Topological charge ℓ = {topological_charge}")
        print(f"[Vortex] DOFs: {V.dofmap.index_map.size_global}")
    
    # Trial/test
    p = TrialFunction(V)
    phi = TestFunction(V)
    
    # Constants
    rho = Constant(mesh, PETSc.ScalarType(config.rho_water))
    k = Constant(mesh, PETSc.ScalarType(config.k_water))
    Z_bottom = Constant(mesh, PETSc.ScalarType(config.Z_bottom))
    Z_air = Constant(mesh, PETSc.ScalarType(config.Z_air))
    
    # Vortex velocity amplitude on top
    v0_vortex = Constant(mesh, PETSc.ScalarType(config.v0_amplitude))
    
    # Create vortex phase pattern
    vortex_phase_func = Function(V)
    coords = V.tabulate_dof_coordinates()
    
    # Compute azimuthal angle relative to domain center
    center = np.array([config.Lx/2, config.Ly/2, config.Lz])
    dx_coords = coords[:, 0] - center[0]
    dy_coords = coords[:, 1] - center[1]
    theta = np.arctan2(dy_coords, dx_coords)
    
    # Vortex phase: exp(i ℓ θ)
    phase_pattern = np.exp(1j * topological_charge * theta)
    vortex_phase_func.x.array[:] = phase_pattern
    
    # Measures
    dx_mesh = dx(domain=mesh)
    ds_mesh = ds(domain=mesh, subdomain_data=facet_tags)
    
    # Bulk + impedance BCs (same as square_dish)
    a_bulk = (
        (1/rho) * inner(grad(p), grad(phi)) * dx_mesh
        - (k**2 / rho) * inner(p, phi) * dx_mesh
    )
    
    a_bottom = (-1j * k / Z_bottom) * inner(p, phi) * ds_mesh(5)
    a_top = (-1j * k / Z_air) * inner(p, phi) * ds_mesh(6)
    
    # Side walls: no actuation (impedance BC or rigid)
    # For simplicity, make sides rigid (∂p/∂n = 0)
    a_sides = 0 * inner(p, phi) * ds_mesh(1)  # Natural BC, zero contribution
    
    a = a_bulk + a_bottom + a_top + a_sides
    
    # RHS: Vortex actuation on top (tag=6)
    omega = Constant(mesh, PETSc.ScalarType(config.omega))
    g_vortex = -1j * omega * rho * v0_vortex * vortex_phase_func
    
    # Use inner() to ensure proper complex conjugation
    L = inner(g_vortex, phi) * ds_mesh(6)
    
    # Assemble
    if rank == 0:
        print("[Vortex] Assembling...")
    
    a_form = form(a)
    L_form = form(L)
    
    A = assemble_matrix(a_form)
    A.assemble()
    b = assemble_vector(L_form)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    
    # Solve
    p_solution = Function(V)
    
    ksp = PETSc.KSP().create(mesh.comm)
    ksp.setOperators(A)
    ksp.setType("gmres")
    ksp.getPC().setType("ilu")
    ksp.setTolerances(rtol=1e-9, max_it=1000)
    
    if rank == 0:
        print("[Vortex] Solving...")
    
    ksp.solve(b, p_solution.x.petsc_vec)
    p_solution.x.scatter_forward()
    
    # Stats
    p_max = comm.allreduce(np.max(np.abs(p_solution.x.array)), op=MPI.MAX)
    if rank == 0:
        print(f"[Vortex] max|p| = {p_max:.3e} Pa\n")
    
    return p_solution


try:
    import pyvista as pv
    pv.start_xvfb()  # For headless rendering
    pv.global_theme.background = 'white'
    PYVISTA_AVAILABLE = True
except (ImportError, OSError):
    PYVISTA_AVAILABLE = False
    print("[Warning] PyVista not available - 3D rendering disabled")


def render_3d_isosurface(p_function, config, output_dir, ell):
    """Render 3D iso-surfaces of pressure magnitude."""
    rank = MPI.COMM_WORLD.rank
    if rank != 0:
        return
    
    if not PYVISTA_AVAILABLE:
        print("[Plot] Skipping 3D rendering (PyVista not available)")
        return
    
    from acoustweezers.viz.plots_3d import extract_pyvista_mesh, add_field_to_mesh
    
    # Extract mesh
    mesh_dolfinx = p_function.function_space.mesh
    grid = extract_pyvista_mesh(mesh_dolfinx)
    
    # Add pressure field
    p_vals = np.abs(p_function.x.array)
    grid.point_data['|p|'] = p_vals
    
    # Create plotter
    plotter = pv.Plotter(off_screen=True, window_size=[1920, 1080])
    
    # Compute iso-surfaces at different levels
    p_max = np.max(p_vals)
    levels = [0.1 * p_max, 0.3 * p_max, 0.5 * p_max, 0.7 * p_max]
    
    # Add iso-surfaces
    for i, level in enumerate(levels):
        contour = grid.contour([level], scalars='|p|')
        opacity = 0.2 + 0.2 * i / len(levels)
        plotter.add_mesh(contour, opacity=opacity, color='blue', show_edges=False)
    
    # Add bounding box
    bounds = grid.bounds
    plotter.add_bounding_box(color='black', line_width=2)
    
    # Add axes
    plotter.add_axes(xlabel='x', ylabel='y', zlabel='z')
    
    # Camera position
    plotter.camera_position = 'iso'
    
    # Title
    plotter.add_text(f'Vortex ℓ={ell}: |p| Iso-surfaces', position='upper_edge', font_size=12)
    
    # Save
    plotter.screenshot(output_dir / f"vortex_ell{ell}_3d.png", return_img=False)
    plotter.close()
    
    print(f"[Plot] Saved 3D rendering to {output_dir / f'vortex_ell{ell}_3d.png'}")


def plot_gorkov_slice(U_function, config, output_dir, ell):
    """Plot Gor'kov potential at mid-height."""
    rank = MPI.COMM_WORLD.rank
    if rank != 0:
        return
    
    z_slice = config.Lz / 2
    
    # Grid
    N = 150
    x = np.linspace(0, config.Lx, N)
    y = np.linspace(0, config.Ly, N)
    X, Y = np.meshgrid(x, y)
    Z = np.full_like(X, z_slice)
    
    points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    
    # Evaluate
    from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
    mesh = U_function.function_space.mesh
    tree = bb_tree(mesh, mesh.topology.dim)
    cell_candidates = compute_collisions_points(tree, points)
    cells = compute_colliding_cells(mesh, cell_candidates, points)
    
    U_vals = np.zeros(len(points), dtype=np.float64)
    for i, point in enumerate(points):
        if len(cells.links(i)) > 0:
            U_vals[i] = U_function.eval(point, cells.links(i)[0])[0]
    
    U = U_vals.reshape(X.shape)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot potential
    levels = np.linspace(np.min(U), np.max(U), 30)
    contour = ax.contourf(X*1e3, Y*1e3, U*1e18, levels=30, cmap='RdBu_r')
    ax.contour(X*1e3, Y*1e3, U*1e18, levels=10, colors='k', linewidths=0.5, alpha=0.3)
    
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_title(f"Gor'kov Potential U (x10^-18 J) - Vortex ℓ={ell}")
    ax.set_aspect('equal')
    plt.colorbar(contour, ax=ax, label='U (x10^-18 J)')
    
    # Mark center
    ax.plot(config.Lx*1e3/2, config.Ly*1e3/2, 'k+', markersize=15, markeredgewidth=2)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"gorkov_ell{ell}_slice.png", dpi=200)
    plt.close()
    
    print(f"[Plot] Saved Gor'kov slice to {output_dir / f'gorkov_ell{ell}_slice.png'}")


def plot_vortex_slices(p_function, config, output_dir, ell):
    """Plot pressure magnitude and phase at mid-height."""
    rank = MPI.COMM_WORLD.rank
    if rank != 0:
        return
    
    z_slice = config.Lz / 2
    
    # Grid
    N = 150
    x = np.linspace(0, config.Lx, N)
    y = np.linspace(0, config.Ly, N)
    X, Y = np.meshgrid(x, y)
    Z = np.full_like(X, z_slice)
    
    points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    
    # Evaluate
    from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
    mesh = p_function.function_space.mesh
    tree = bb_tree(mesh, mesh.topology.dim)
    cell_candidates = compute_collisions_points(tree, points)
    cells = compute_colliding_cells(mesh, cell_candidates, points)
    
    p_vals = np.zeros(len(points), dtype=np.complex128)
    for i, point in enumerate(points):
        if len(cells.links(i)) > 0:
            p_vals[i] = p_function.eval(point, cells.links(i)[0])[0]
    
    p_mag = np.abs(p_vals).reshape(X.shape)
    p_phase = np.angle(p_vals).reshape(X.shape)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    im0 = axes[0].pcolormesh(X*1e3, Y*1e3, p_mag*1e-6, shading='gouraud', cmap='viridis')
    axes[0].set_xlabel('x (mm)')
    axes[0].set_ylabel('y (mm)')
    axes[0].set_title(f'|p| (MPa) - Vortex ℓ={ell}')
    axes[0].set_aspect('equal')
    plt.colorbar(im0, ax=axes[0], label='|p| (MPa)')
    
    im1 = axes[1].pcolormesh(X*1e3, Y*1e3, p_phase, shading='gouraud', cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axes[1].set_xlabel('x (mm)')
    axes[1].set_ylabel('y (mm)')
    axes[1].set_title(f'phase(p) - Vortex ℓ={ell}')
    axes[1].set_aspect('equal')
    plt.colorbar(im1, ax=axes[1], label='phase (rad)')
    
    plt.tight_layout()
    plt.savefig(output_dir / f"vortex_ell{ell}_slice.png", dpi=200)
    plt.close()
    
    print(f"[Plot] Saved vortex slice to {output_dir / f'vortex_ell{ell}_slice.png'}")


def main():
    parser = argparse.ArgumentParser(description="Acoustic vortex demonstration")
    parser.add_argument('--topological_charge', type=int, default=1, help='Vortex winding number ℓ')
    parser.add_argument('--elements_per_wavelength', type=float, default=8.0)
    parser.add_argument('--output_dir', type=str, default='results/vortex_demo')
    
    args = parser.parse_args()
    
    rank = MPI.COMM_WORLD.rank
    
    # Configuration
    config = SquareDishConfig(
        elements_per_wavelength=args.elements_per_wavelength,
        frequency=2e6,  # 2 MHz
        v0_amplitude=0.1  # 0.1 m/s
    )
    
    if rank == 0:
        print("="*70)
        print("ACOUSTIC VORTEX DEMONSTRATION")
        print("="*70)
        print(f"Domain: {config.Lx*1e3:.1f} × {config.Ly*1e3:.1f} × {config.Lz*1e3:.1f} mm")
        print(f"Frequency: {config.frequency*1e-6:.2f} MHz")
        print(f"Wavelength: {config.wavelength*1e3:.3f} mm")
        print(f"Topological charge: ℓ = {args.topological_charge}")
        print(f"Elements per wavelength: {config.elements_per_wavelength:.1f}")
        print("="*70 + "\n")
    
    # Create mesh
    if rank == 0:
        print("[Main] Creating mesh...")
    mesh, cell_tags, facet_tags = create_square_dish_mesh(config, verbose=(rank==0))
    
    # Solve
    if rank == 0:
        print("\n[Main] Solving vortex Helmholtz...")
    p_solution = solve_vortex_helmholtz(mesh, facet_tags, config, args.topological_charge)
    
    # Compute Gor'kov
    if rank == 0:
        print("[Main] Computing Gor'kov potential...")
    U_solution = compute_gorkov_potential_3d(p_solution, config, verbose=(rank==0))
    
    # Visualize
    if rank == 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(args.output_dir) / f"run_{timestamp}_ell{args.topological_charge}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("[Main] Generating plots...")
        plot_vortex_slices(p_solution, config, output_dir, args.topological_charge)
        plot_gorkov_slice(U_solution, config, output_dir, args.topological_charge)
        render_3d_isosurface(p_solution, config, output_dir, args.topological_charge)
        
        print(f"\n{'='*70}")
        print(f"VORTEX DEMONSTRATION COMPLETE")
        print(f"Results: {output_dir}")
        print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
