"""
Visualization Demo with GIF Generation
======================================
Creates visualization outputs including 3D slice plots and rotation GIF.
"""

import numpy as np
from pathlib import Path
import sys

from mpi4py import MPI
from petsc4py import PETSc

print("=" * 70)
print("CREATING VISUALIZATION OUTPUTS")
print("=" * 70)

# Verify complex backend
if not np.issubdtype(PETSc.ScalarType, np.complexfloating):
    print("❌ ERROR: Requires complex PETSc")
    sys.exit(1)

import dolfinx
from dolfinx import mesh, fem
from dolfinx.fem.petsc import LinearProblem
import ufl

# Import visualization module
from tweezers.fenicsx.visualization import (
    plot_pressure_field_3d,
    create_animation_frames,
    frames_to_gif
)


def create_demo_solution():
    """Create a simple 3D acoustic solution for visualization."""
    print("\n[1] Creating 3D demo mesh...")
    
    # Create a small 3D box
    comm = MPI.COMM_WORLD
    domain = mesh.create_box(
        comm,
        [[0.0, 0.0, 0.0], [0.01, 0.01, 0.003]],
        [10, 10, 6],
        mesh.CellType.tetrahedron
    )
    
    ncells = domain.topology.index_map(domain.topology.dim).size_local
    print(f"    Mesh: {ncells} cells")
    
    # Create function space
    V = fem.functionspace(domain, ("Lagrange", 1))
    ndofs = V.dofmap.index_map.size_local
    print(f"    DOFs: {ndofs}")
    
    # Solve Helmholtz equation
    print("\n[2] Solving Helmholtz equation...")
    
    p = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    freq = 40e3  # 40 kHz
    omega = 2 * np.pi * freq
    c = 1500.0
    k = omega / c
    
    # Bilinear form
    k_sq = fem.Constant(domain, PETSc.ScalarType(k**2))
    a = ufl.inner(ufl.grad(p), ufl.grad(v)) * ufl.dx
    a -= k_sq * ufl.inner(p, v) * ufl.dx
    
    # Find bottom boundary
    fdim = domain.topology.dim - 1
    domain.topology.create_connectivity(fdim, domain.topology.dim)
    
    def bottom(x):
        return np.isclose(x[2], 0.0)
    
    def top(x):
        return np.isclose(x[2], 0.003)
    
    bottom_facets = mesh.locate_entities_boundary(domain, fdim, bottom)
    top_facets = mesh.locate_entities_boundary(domain, fdim, top)
    
    # BC: p = 1 at bottom, p = 0 at top
    p_bottom = fem.Constant(domain, PETSc.ScalarType(1.0))
    p_top = fem.Constant(domain, PETSc.ScalarType(0.0))
    
    bottom_dofs = fem.locate_dofs_topological(V, fdim, bottom_facets)
    top_dofs = fem.locate_dofs_topological(V, fdim, top_facets)
    
    bc_bottom = fem.dirichletbc(p_bottom, bottom_dofs, V)
    bc_top = fem.dirichletbc(p_top, top_dofs, V)
    
    # RHS
    f = fem.Function(V)
    f.x.array[:] = 0
    L = ufl.inner(f, v) * ufl.dx
    
    # Solve
    problem = LinearProblem(a, L, bcs=[bc_bottom, bc_top], petsc_options={
        "ksp_type": "preonly",
        "pc_type": "lu"
    })
    
    p_h = problem.solve()
    
    print(f"    max|p| = {np.max(np.abs(p_h.x.array)):.4f}")
    
    return domain, p_h


def main():
    """Generate all visualization outputs."""
    
    # Create output directory
    output_dir = Path("results/visualization_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate solution
    domain, pressure = create_demo_solution()
    
    # 1. Create 3D slice plot
    print("\n[3] Creating 3D slice visualization...")
    plot_pressure_field_3d(
        domain, pressure,
        output_path=output_dir / "pressure_3d_slice.png",
        slice_normal="z",
        title="Pressure Field |p| - Z-slice"
    )
    
    # 2. Create cross-section
    print("\n[4] Creating cross-section plot...")
    from tweezers.fenicsx.visualization import plot_cross_section
    plot_cross_section(
        domain, pressure,
        axis="z",
        position=0.0015,  # Middle
        output_path=output_dir / "pressure_cross_section.png",
        title="Pressure Cross-Section at z=1.5mm"
    )
    
    # 3. Create rotation animation frames
    print("\n[5] Creating rotation animation...")
    print("    Generating frames (this may take a minute)...")
    
    frame_paths = create_animation_frames(
        domain, pressure,
        output_dir=output_dir / "frames",
        n_frames=36,  # 36 frames = 10° per frame
        prefix="rotation"
    )
    
    # 4. Convert to GIF
    print("\n[6] Creating GIF...")
    try:
        frames_to_gif(
            frame_paths,
            output_path=output_dir / "pressure_rotation.gif",
            duration=0.1  # 100ms per frame = 3.6s total
        )
    except ImportError:
        print("    ⚠ PIL not available, frames saved but GIF not created")
        print("    Install with: pip install pillow")
    
    # Summary
    print("\n" + "=" * 70)
    print("VISUALIZATION OUTPUTS")
    print("=" * 70)
    print(f"\nAll outputs saved to: {output_dir.absolute()}")
    print("\nFiles created:")
    print(f"  ✓ pressure_3d_slice.png       - 3D visualization with slice")
    print(f"  ✓ pressure_cross_section.png  - 2D cross-section")
    print(f"  ✓ pressure_rotation.gif       - 360° rotation animation")
    print(f"  ✓ frames/rotation_*.png       - {len(frame_paths)} animation frames")
    print("=" * 70)


if __name__ == "__main__":
    main()
