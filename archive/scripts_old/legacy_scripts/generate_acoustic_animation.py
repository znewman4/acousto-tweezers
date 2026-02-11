#!/usr/bin/env python3
"""
Generate animated GIF showing time evolution of 2D acoustic field.

This script solves the Helmholtz equation and then animates the
time-harmonic solution to show wave propagation.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpi4py import MPI
import dolfinx
from dolfinx import fem, mesh
from dolfinx.io import gmsh as gmshio
from dolfinx.fem import petsc as fem_petsc
import ufl
from petsc4py import PETSc
import gmsh

# Output directory
OUTPUT_DIR = Path("results/demo_2d_acoustics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_2d_petri_mesh(R_inner: float, R_outer: float, water_depth: float, lc: float):
    """
    Create a simplified 2D cross-section of Petri dish.
    
    This is a rectangular water domain with semicircular ends.
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("petri_2d")
    
    # Simple rectangular water domain
    width = 2 * R_inner
    height = water_depth
    
    rect = gmsh.model.occ.addRectangle(-R_inner, 0, 0, width, height)
    gmsh.model.occ.synchronize()
    
    # Physical groups
    gmsh.model.addPhysicalGroup(2, [rect], tag=1, name="Water")
    
    # Get boundary curves and classify
    boundary = gmsh.model.getBoundary([(2, rect)], oriented=False)
    
    # Use a tolerance based on the element size
    tol = lc * 0.1
    
    bottom_curves = []
    top_curves = []
    side_curves = []
    
    for b in boundary:
        curve_tag = b[1]
        bounds = gmsh.model.getBoundingBox(1, curve_tag)
        ymin, ymax = bounds[1], bounds[4]
        
        # Bottom: both ymin and ymax close to 0
        if abs(ymin) < tol and abs(ymax) < tol:
            bottom_curves.append(curve_tag)
        # Top: both ymin and ymax close to height
        elif abs(ymin - height) < tol and abs(ymax - height) < tol:
            top_curves.append(curve_tag)
        else:
            side_curves.append(curve_tag)
    
    # Create physical groups for each boundary type
    if bottom_curves:
        gmsh.model.addPhysicalGroup(1, bottom_curves, tag=101, name="Transducer")
    if top_curves:
        gmsh.model.addPhysicalGroup(1, top_curves, tag=102, name="AirInterface")
    if side_curves:
        gmsh.model.addPhysicalGroup(1, side_curves, tag=103, name="Walls")
    
    # Mesh
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc * 0.5)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc * 2.0)
    gmsh.model.mesh.generate(2)
    
    # Import to DOLFINx
    mesh_data = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
    msh = mesh_data.mesh
    cell_tags = mesh_data.cell_tags
    facet_tags = mesh_data.facet_tags
    
    gmsh.finalize()
    
    return msh, cell_tags, facet_tags


def solve_standing_wave(msh, facet_tags, omega: float, c: float) -> fem.Function:
    """
    Solve for standing wave pattern with driven bottom boundary.
    
    We solve: ∇²p + k²p = 0
    with p = p_bc on bottom boundary (transducer)
    and ∂p/∂n = 0 on sides/top (hard walls)
    
    The key is that for resonance to occur, k*H should be close to nπ/2.
    """
    k = omega / c
    
    # Function space
    V = fem.functionspace(msh, ("Lagrange", 2))
    
    # Trial and test
    p = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dx = ufl.Measure("dx", domain=msh)
    ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_tags)
    
    # Helmholtz equation: -∇²p - k²p = 0
    # Weak form: ∫ ∇p·∇v dx - k² ∫ p·v dx = 0
    a = ufl.inner(ufl.grad(p), ufl.grad(v)) * dx - k**2 * ufl.inner(p, v) * dx
    
    # Dirichlet on bottom (transducer): prescribed pressure amplitude = 1
    p_bc = fem.Function(V)
    p_bc.x.array[:] = 1.0  # Uniform pressure BC
    
    boundary_dofs = fem.locate_dofs_topological(
        V, msh.topology.dim - 1,
        facet_tags.find(101)
    )
    bc = fem.dirichletbc(p_bc, boundary_dofs)
    
    # Zero RHS - natural BC on other boundaries (Neumann: ∂p/∂n = 0)
    L = fem.Constant(msh, PETSc.ScalarType(0.0)) * v * dx
    
    # Assemble and solve
    a_compiled = fem.form(a)
    L_compiled = fem.form(L)
    
    A = fem_petsc.assemble_matrix(a_compiled, bcs=[bc])
    A.assemble()
    
    b = fem_petsc.assemble_vector(L_compiled)
    fem_petsc.apply_lifting(b, [a_compiled], [[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    fem_petsc.set_bc(b, [bc])
    
    p_h = fem.Function(V)
    
    solver = PETSc.KSP().create(msh.comm)
    solver.setOperators(A)
    solver.setType("preonly")
    solver.getPC().setType("lu")
    solver.getPC().setFactorSolverType("mumps")
    solver.solve(b, p_h.x.petsc_vec)
    p_h.x.scatter_forward()
    
    solver.destroy()
    A.destroy()
    b.destroy()
    
    return p_h


def create_animation(p_h: fem.Function, omega: float, output_path: Path, 
                     n_frames: int = 60, fps: int = 20):
    """
    Create animated GIF showing time evolution of pressure field.
    
    For a standing wave pattern, animate as:
    p(x,t) = p_real(x) * cos(omega*t)
    """
    print(f"Creating animation with {n_frames} frames at {fps} fps...")
    
    V = p_h.function_space
    msh = V.mesh
    
    # Get DOF coordinates
    dof_coords = V.tabulate_dof_coordinates()
    x = dof_coords[:, 0] * 1000  # Convert to mm
    y = dof_coords[:, 1] * 1000
    
    # Real pressure values (standing wave amplitude)
    p_amplitude = np.real(p_h.x.array.copy())
    p_max = np.max(np.abs(p_amplitude)) * 1.1  # Add margin
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Initial plot
    t = 0
    p_real = p_amplitude * np.cos(omega * t)
    
    scatter = ax.scatter(x, y, c=p_real, cmap='RdBu_r', 
                         vmin=-p_max, vmax=p_max, s=3)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('z (mm)')
    ax.set_aspect('equal')
    cbar = plt.colorbar(scatter, ax=ax, label='Pressure (Pa)')
    
    # Time period
    T = 2 * np.pi / omega
    times = np.linspace(0, T, n_frames, endpoint=False)
    
    def update(frame):
        t = times[frame]
        p_real = p_amplitude * np.cos(omega * t)
        scatter.set_array(p_real)
        ax.set_title(f'Acoustic Pressure Field (t = {t*1e6:.2f} μs, T = {T*1e6:.2f} μs)')
        return scatter,
    
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000/fps, blit=True)
    
    # Save as GIF
    print(f"Saving to {output_path}...")
    anim.save(str(output_path), writer='pillow', fps=fps)
    plt.close()
    
    print(f"Animation saved: {output_path}")


def main():
    print("=" * 60)
    print("2D Acoustic Wave Animation Generator")
    print("=" * 60)
    
    # Physical parameters
    R_inner = 17.0e-3    # 17 mm inner radius (35mm diameter dish)
    R_outer = 17.5e-3    # Include wall
    water_depth = 2.0e-3 # 2 mm water depth
    
    f = 2.0e6            # 2 MHz frequency
    c = 1480.0           # Sound speed in water (m/s)
    omega = 2 * np.pi * f
    k = omega / c
    wavelength = c / f
    
    print(f"\nParameters:")
    print(f"  Domain width: {2*R_inner*1000:.1f} mm")
    print(f"  Water depth: {water_depth*1000:.1f} mm")
    print(f"  Frequency: {f/1e6:.2f} MHz")
    print(f"  Wavelength: {wavelength*1e6:.1f} μm")
    
    # Mesh
    ppw = 12
    lc = wavelength / ppw
    
    print(f"\nMesh:")
    print(f"  Points/wavelength: {ppw}")
    print(f"  Element size: {lc*1e6:.1f} μm")
    
    print("\nCreating mesh...")
    msh, cell_tags, facet_tags = create_2d_petri_mesh(R_inner, R_outer, water_depth, lc)
    num_cells = msh.topology.index_map(msh.topology.dim).size_local
    print(f"  Cells: {num_cells}")
    
    V = fem.functionspace(msh, ("Lagrange", 2))
    print(f"  DOFs: {V.dofmap.index_map.size_global}")
    
    print("\nSolving for standing wave pattern...")
    p_h = solve_standing_wave(msh, facet_tags, omega, c)
    
    p_vals = p_h.x.array
    print(f"\nResults:")
    print(f"  Max |p|: {np.max(np.abs(p_vals)):.4f} Pa")
    print(f"  Mean |p|: {np.mean(np.abs(p_vals)):.4f} Pa")
    
    # Create animation
    print("\nGenerating animation...")
    create_animation(
        p_h, omega,
        OUTPUT_DIR / "acoustic_wave.gif",
        n_frames=60,
        fps=20
    )
    
    # Also save a static image
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    
    dof_coords = V.tabulate_dof_coordinates()
    x = dof_coords[:, 0] * 1000
    y = dof_coords[:, 1] * 1000
    p_vals = p_h.x.array
    
    sc1 = axes[0].scatter(x, y, c=np.real(p_vals), cmap='RdBu_r', s=3)
    axes[0].set_xlabel('x (mm)')
    axes[0].set_ylabel('z (mm)')
    axes[0].set_title('Real(p)')
    axes[0].set_aspect('equal')
    plt.colorbar(sc1, ax=axes[0], label='Pa')
    
    sc2 = axes[1].scatter(x, y, c=np.abs(p_vals), cmap='hot', s=3)
    axes[1].set_xlabel('x (mm)')
    axes[1].set_ylabel('z (mm)')
    axes[1].set_title('|p| (Pressure Magnitude)')
    axes[1].set_aspect('equal')
    plt.colorbar(sc2, ax=axes[1], label='Pa')
    
    plt.suptitle(f'2D Petri Dish Acoustics (f={f/1e6:.1f} MHz)', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "standing_wave.png", dpi=150)
    plt.close()
    
    print(f"Static image saved: {OUTPUT_DIR / 'standing_wave.png'}")
    
    print("\n" + "=" * 60)
    print("ANIMATION COMPLETE")
    print("=" * 60)
    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print(f"  - acoustic_wave.gif (animated)")
    print(f"  - standing_wave.png (static)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
