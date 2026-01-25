#!/usr/bin/env python3
"""
Simple 2D acoustics demonstration using FEniCSx.

This script demonstrates the FEniCSx acoustic solver with a 2D disk geometry.
It's designed to run quickly and produce visualization output.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
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


def create_2d_disk_mesh(radius: float, lc: float) -> tuple:
    """Create a 2D disk mesh with Gmsh."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("disk")
    
    # Create disk
    disk = gmsh.model.occ.addDisk(0, 0, 0, radius, radius)
    gmsh.model.occ.synchronize()
    
    # Physical groups
    gmsh.model.addPhysicalGroup(2, [disk], tag=1, name="Domain")
    boundary = gmsh.model.getBoundary([(2, disk)], oriented=False)
    boundary_tags = [b[1] for b in boundary]
    gmsh.model.addPhysicalGroup(1, boundary_tags, tag=101, name="Boundary")
    
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


def solve_helmholtz_2d(
    msh,
    facet_tags,
    omega: float,
    c: float,
    rho: float,
    bc_amplitude: float = 1.0,
) -> fem.Function:
    """
    Solve 2D Helmholtz equation.
    
    -∇·(1/ρ ∇p) - (ω²/ρc²)p = 0
    
    with Dirichlet BC on boundary.
    """
    k = omega / c
    
    # Function space (complex-valued)
    V = fem.functionspace(msh, ("Lagrange", 2))
    
    # Trial and test functions
    p = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dx = ufl.Measure("dx", domain=msh)
    
    # Weak form: ∫(1/ρ)∇p·∇v dx - (ω²/ρc²)∫p·v dx = 0
    # Simplified (constant properties): ∫∇p·∇v dx - k²∫p·v dx = 0
    a = ufl.inner(ufl.grad(p), ufl.grad(v)) * dx - k**2 * ufl.inner(p, v) * dx
    
    # Dirichlet BC: oscillating pressure on boundary
    # p = A * exp(i * k * x) (plane wave approximation)
    def bc_expression(x):
        return bc_amplitude * np.exp(1j * k * x[0])
    
    p_bc = fem.Function(V)
    p_bc.interpolate(bc_expression)
    
    # Find boundary DOFs
    boundary_dofs = fem.locate_dofs_topological(
        V, msh.topology.dim - 1, 
        facet_tags.find(101)
    )
    bc = fem.dirichletbc(p_bc, boundary_dofs)
    
    # Zero RHS
    L = fem.Constant(msh, PETSc.ScalarType(0.0)) * v * dx
    
    # Assemble
    a_compiled = fem.form(a)
    L_compiled = fem.form(L)
    
    A = fem_petsc.assemble_matrix(a_compiled, bcs=[bc])
    A.assemble()
    
    b = fem_petsc.assemble_vector(L_compiled)
    fem_petsc.apply_lifting(b, [a_compiled], [[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    fem_petsc.set_bc(b, [bc])
    
    # Solution function
    p_h = fem.Function(V)
    
    # Solve
    solver = PETSc.KSP().create(msh.comm)
    solver.setOperators(A)
    solver.setType("preonly")
    solver.getPC().setType("lu")
    solver.getPC().setFactorSolverType("mumps")
    solver.solve(b, p_h.x.petsc_vec)
    p_h.x.scatter_forward()
    
    # Cleanup
    solver.destroy()
    A.destroy()
    b.destroy()
    
    return p_h


def plot_solution(p_h: fem.Function, output_path: Path, title: str = "Pressure Field"):
    """Plot the solution using matplotlib."""
    # Extract mesh coordinates and values
    V = p_h.function_space
    msh = V.mesh
    
    # Get DOF coordinates
    dof_coords = V.tabulate_dof_coordinates()
    x = dof_coords[:, 0]
    y = dof_coords[:, 1]
    
    # Get solution values
    p_vals = p_h.x.array
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Real part
    sc1 = axes[0].scatter(x * 1000, y * 1000, c=np.real(p_vals), 
                          cmap='RdBu_r', s=5)
    axes[0].set_xlabel('x (mm)')
    axes[0].set_ylabel('y (mm)')
    axes[0].set_title('Real(p)')
    axes[0].set_aspect('equal')
    plt.colorbar(sc1, ax=axes[0], label='Pa')
    
    # Magnitude
    sc2 = axes[1].scatter(x * 1000, y * 1000, c=np.abs(p_vals), 
                          cmap='viridis', s=5)
    axes[1].set_xlabel('x (mm)')
    axes[1].set_ylabel('y (mm)')
    axes[1].set_title('|p|')
    axes[1].set_aspect('equal')
    plt.colorbar(sc2, ax=axes[1], label='Pa')
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Saved plot to: {output_path}")


def main():
    print("=" * 60)
    print("2D FEniCSx Acoustic Demonstration")
    print("=" * 60)
    
    # Physical parameters
    R = 10.0e-3          # 10 mm radius
    f = 1.0e6            # 1 MHz frequency
    c = 1480.0           # Sound speed in water (m/s)
    rho = 998.0          # Water density (kg/m³)
    omega = 2 * np.pi * f
    k = omega / c
    wavelength = c / f
    
    print(f"\nParameters:")
    print(f"  Disk radius: {R*1000:.1f} mm")
    print(f"  Frequency: {f/1e6:.2f} MHz")
    print(f"  Wavelength: {wavelength*1e6:.1f} μm")
    print(f"  Wavenumber: k = {k:.1f} rad/m")
    print(f"  ka = {k*R:.2f}")
    
    # Mesh parameters
    ppw = 10  # Points per wavelength
    lc = wavelength / ppw
    
    print(f"\nMesh:")
    print(f"  Points/wavelength: {ppw}")
    print(f"  Element size: {lc*1e6:.1f} μm")
    
    # Create mesh
    print("\nCreating mesh...")
    msh, cell_tags, facet_tags = create_2d_disk_mesh(R, lc)
    num_cells = msh.topology.index_map(msh.topology.dim).size_local
    print(f"  Cells: {num_cells}")
    
    # Create function space to count DOFs
    V = fem.functionspace(msh, ("Lagrange", 2))
    num_dofs = V.dofmap.index_map.size_global
    print(f"  DOFs: {num_dofs}")
    
    # Solve
    print("\nSolving Helmholtz equation...")
    p_h = solve_helmholtz_2d(msh, facet_tags, omega, c, rho)
    
    # Results
    p_vals = p_h.x.array
    p_max = np.max(np.abs(p_vals))
    p_mean = np.mean(np.abs(p_vals))
    
    print(f"\nResults:")
    print(f"  Max |p|: {p_max:.4f} Pa")
    print(f"  Mean |p|: {p_mean:.4f} Pa")
    print(f"  Min |p|: {np.min(np.abs(p_vals)):.4f} Pa")
    
    # Plot
    print("\nGenerating visualization...")
    plot_solution(
        p_h, 
        OUTPUT_DIR / "pressure_field_2d.png",
        f"2D Acoustic Field (f={f/1e6:.1f} MHz, λ={wavelength*1e6:.0f} μm)"
    )
    
    # Save solution as VTK for Paraview
    try:
        from dolfinx.io import VTXWriter
        
        # Need to create a real-valued function for VTK output
        V_real = fem.functionspace(msh, ("Lagrange", 2, (2,)))  # Vector for real/imag
        p_vtk = fem.Function(V_real)
        p_vtk.x.array[0::2] = np.real(p_h.x.array)
        p_vtk.x.array[1::2] = np.imag(p_h.x.array)
        
        print(f"  Note: VTK output requires separate handling for complex values")
    except Exception as e:
        print(f"  Note: VTK output not available ({e})")
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print(f"\nOutput saved to: {OUTPUT_DIR}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
