#!/usr/bin/env python3
"""
Acoustic Vortex Lens - Simplified Validation

Demonstrates vortex lens using Dirichlet BC (prescribed pressure).
This is simpler and more physical than Neumann BC for lens actuation.

Geometry: Box domain with vortex actuation on top boundary
Physics: Helmholtz with prescribed vortex field at top

Author: Acousto-Tweezers Project  
Date: February 2026
"""

from pathlib import Path
import sys
from datetime import datetime
import argparse

import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

from petsc4py import PETSc
if not np.issubdtype(PETSc.ScalarType, np.complexfloating):
    raise RuntimeError("Requires complex PETSc")

import dolfinx
from dolfinx import fem, mesh as dmesh
from dolfinx.fem import Function, FunctionSpace, Constant, form, locate_dofs_topological
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
import ufl
from ufl import inner, grad, dx, ds, TestFunction, TrialFunction

# Import vortex lens module
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from acoustweezers.physics.acoustics.vortex_lens import compute_azimuthal_phase

# =============================================================================
# Simple box mesh with gmsh
# =============================================================================

def create_box_mesh(Lx, Ly, Lz, nx, ny, nz):
    """Create structured box mesh."""
    from dolfinx.mesh import create_box, CellType
    
    comm = MPI.COMM_WORLD
    mesh = create_box(
        comm,
        [[0, 0, 0], [Lx, Ly, Lz]],
        [nx, ny, nz],
        CellType.tetrahedron
    )
    return mesh


def mark_boundaries(mesh):
    """
    Mark boundaries of box mesh.
    
    Returns facet tags:
    - 1: x=0
    - 2: x=Lx
    - 3: y=0
    - 4: y=Ly
    - 5: z=0 (bottom)
    - 6: z=Lz (top - vortex)
    """
    tdim = mesh.topology.dim
    fdim = tdim - 1
    
    # Get all facets
    mesh.topology.create_connectivity(fdim, tdim)
    boundary_facets = dmesh.exterior_facet_indices(mesh.topology)
    
    # Get coordinates
    x = mesh.geometry.x
    
    # Get facet midpoints
    facet_midpoints = dmesh.compute_midpoints(mesh, fdim, boundary_facets)
    
    # Get domain extents
    coords = mesh.geometry.x
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    z_min, z_max = coords[:, 2].min(), coords[:, 2].max()
    
    tol = 1e-10
    
    # Classify facets
    facet_values = np.zeros(len(boundary_facets), dtype=np.int32)
    
    for i, facet_idx in enumerate(boundary_facets):
        mp = facet_midpoints[i]
        if abs(mp[0] - x_min) < tol:
            facet_values[i] = 1
        elif abs(mp[0] - x_max) < tol:
            facet_values[i] = 2
        elif abs(mp[1] - y_min) < tol:
            facet_values[i] = 3
        elif abs(mp[1] - y_max) < tol:
            facet_values[i] = 4
        elif abs(mp[2] - z_min) < tol:
            facet_values[i] = 5
        elif abs(mp[2] - z_max) < tol:
            facet_values[i] = 6
    
    facet_tags = dmesh.meshtags(mesh, fdim, boundary_facets, facet_values)
    
    return facet_tags


# =============================================================================
# Solver
# =============================================================================

def solve_vortex(mesh, facet_tags, config):
    """Solve Helmholtz with vortex Dirichlet BC on top."""
    comm = mesh.comm
    rank = comm.rank
    
    # Function space
    V = fem.functionspace(mesh, ("Lagrange", 2))
    
    # Material constants
    rho = config['rho']
    c = config['c']
    omega = config['omega']
    k = omega / c
    
    # Create vortex boundary function
    p_vortex_bc = Function(V)
    coords = V.tabulate_dof_coordinates()
    
    # Compute vortex phase: φ(θ) = ℓθ
    ell = config['topological_charge']
    center = np.array([config['Lx']/2, config['Ly']/2, config['Lz']])
    
    # Compute azimuthal angle
    dx = coords[:, 0] - center[0]
    dy = coords[:, 1] - center[1]
    theta = np.arctan2(dy, dx)
    
    # Vortex field: A exp(i ℓ θ)
    A = config['amplitude']
    p_vortex_bc.x.array[:] = A * np.exp(1j * ell * theta)
    
    # Locate DOFs on top boundary (tag=6)
    top_dofs = locate_dofs_topological(V, facet_tags.dim, facet_tags.find(6))
    bc = fem.dirichletbc(p_vortex_bc, top_dofs)
    
    # Trial and test
    p = TrialFunction(V)
    phi = TestFunction(V)
    
    # Forms
    rho_const = Constant(mesh, PETSc.ScalarType(rho))
    c_const = Constant(mesh, PETSc.ScalarType(c))
    omega_const = Constant(mesh, PETSc.ScalarType(omega))
    k_const = omega_const / c_const
    
    # Build forms using exact pattern from square_dish
    Z = Constant(mesh, PETSc.ScalarType(rho * c))
    ds_mesh = ds(domain=mesh, subdomain_data=facet_tags)
    
    # Bilinear form - build piece by piece like square_dish
    a_bulk = (
        (1/rho_const) * inner(grad(p), grad(phi)) * dx
        - (k_const**2 / rho_const) * inner(p, phi) * dx
    )
    
    a_bc1 = (-1j * k_const / Z) * inner(p, phi) * ds_mesh(1)
    a_bc2 = (-1j * k_const / Z) * inner(p, phi) * ds_mesh(2)
    a_bc3 = (-1j * k_const / Z) * inner(p, phi) * ds_mesh(3)
    a_bc4 = (-1j * k_const / Z) * inner(p, phi) * ds_mesh(4)
    a_bc5 = (-1j * k_const / Z) * inner(p, phi) * ds_mesh(5)
    
    # Combine
    a = a_bulk + a_bc1 + a_bc2 + a_bc3 + a_bc4 + a_bc5
    
    L = phi * Constant(mesh, PETSc.ScalarType(0.0)) * dx
    
    # Assemble
    if rank == 0:
        print("[Solver] Assembling...")
    
    a_form = form(a)
    L_form = form(L)
    
    A = assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = assemble_vector(L_form)
    fem.petsc.apply_lifting(b, [a_form], [[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    fem.petsc.set_bc(b, [bc])
    
    # Solve
    p_solution = Function(V)
    
    ksp = PETSc.KSP().create(mesh.comm)
    ksp.setOperators(A)
    ksp.setType("gmres")
    ksp.getPC().setType("ilu")
    ksp.setTolerances(rtol=1e-9, max_it=1000)
    
    if rank == 0:
        print("[Solver] Solving...")
    
    ksp.solve(b, p_solution.x.petsc_vec)
    p_solution.x.scatter_forward()
    
    # Stats
    p_max = comm.allreduce(np.max(np.abs(p_solution.x.array)), op=MPI.MAX)
    if rank == 0:
        print(f"[Solver] max|p| = {p_max:.3e} Pa\n")
    
    return p_solution


# =============================================================================
# Visualization
# =============================================================================

def plot_axial_slice(p_function, config, output_path, z_slice=None):
    """Plot axial slice."""
    if z_slice is None:
        z_slice = config['Lz'] / 2
    
    # Create grid
    N = 150
    x = np.linspace(0, config['Lx'], N)
    y = np.linspace(0, config['Ly'], N)
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
    axes[0].set_title(f'|p| (MPa) at z={z_slice*1e3:.2f} mm')
    axes[0].set_aspect('equal')
    plt.colorbar(im0, ax=axes[0], label='|p| (MPa)')
    
    im1 = axes[1].pcolormesh(X*1e3, Y*1e3, p_phase, shading='gouraud', cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axes[1].set_xlabel('x (mm)')
    axes[1].set_ylabel('y (mm)')
    axes[1].set_title(f'phase(p) at z={z_slice*1e3:.2f} mm')
    axes[1].set_aspect('equal')
    plt.colorbar(im1, ax=axes[1], label='phase (rad)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    
    print(f"[Plot] Saved to {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--topological_charge', type=int, default=1)
    parser.add_argument('--frequency', type=float, default=2.0, help='MHz')
    parser.add_argument('--n_elements', type=int, default=20, help='Elements per side')
    parser.add_argument('--output_dir', type=str, default='results/vortex_simple')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'Lx': 2e-3,
        'Ly': 2e-3,
        'Lz': 2e-3,
        'rho': 998.0,
        'c': 1500.0,
        'frequency': args.frequency * 1e6,
        'topological_charge': args.topological_charge,
        'amplitude': 1e6
    }
    config['omega'] = 2 * np.pi * config['frequency']
    config['wavelength'] = config['c'] / config['frequency']
    
    rank = MPI.COMM_WORLD.rank
    if rank == 0:
        print("="*70)
        print("VORTEX LENS - SIMPLE VALIDATION")
        print("="*70)
        print(f"Domain: {config['Lx']*1e3:.1f} × {config['Ly']*1e3:.1f} × {config['Lz']*1e3:.1f} mm")
        print(f"Frequency: {config['frequency']*1e-6:.2f} MHz")
        print(f"Wavelength: {config['wavelength']*1e3:.3f} mm")
        print(f"Topological charge: ℓ = {config['topological_charge']}")
        print(f"Amplitude: {config['amplitude']*1e-6:.2f} MPa")
        print("="*70 + "\n")
    
    # Create mesh
    if rank == 0:
        print("[Main] Creating mesh...")
    n = args.n_elements
    mesh = create_box_mesh(config['Lx'], config['Ly'], config['Lz'], n, n, n)
    facet_tags = mark_boundaries(mesh)
    
    if rank == 0:
        num_cells = mesh.topology.index_map(mesh.topology.dim).size_global
        print(f"[Main] Mesh: {num_cells} cells\n")
    
    # Solve
    p_solution = solve_vortex(mesh, facet_tags, config)
    
    # Visualize
    if rank == 0:
        print("[Main] Generating plots...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(args.output_dir) / f"run_{timestamp}_ell{config['topological_charge']}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for z_frac in [0.1, 0.5, 0.9]:
            plot_axial_slice(
                p_solution, config,
                output_dir / f"slice_z{z_frac:.1f}.png",
                z_slice=z_frac * config['Lz']
            )
        
        print(f"\n{'='*70}")
        print(f"Results saved to: {output_dir}")
        print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
