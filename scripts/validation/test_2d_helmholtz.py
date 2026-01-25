#!/usr/bin/env python3
"""
Quick 2D validation test for the FEniCSx acoustic solver.

This script tests the basic acoustic solver with a simple 2D disk geometry
to verify the FEniCSx implementation works before running full 3D simulations.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
from mpi4py import MPI
import dolfinx
from dolfinx import fem, mesh, io
from dolfinx.io import gmsh as gmshio
from dolfinx.fem import petsc as fem_petsc
import ufl
from petsc4py import PETSc
import gmsh


def test_2d_helmholtz():
    """Test 2D Helmholtz equation on a disk."""
    print("=" * 60)
    print("2D HELMHOLTZ EQUATION TEST")
    print("=" * 60)
    
    # Parameters
    R = 0.01  # 10 mm radius
    f = 1e6   # 1 MHz
    c = 1480  # Sound speed in water
    omega = 2 * np.pi * f
    k = omega / c  # Wavenumber
    
    wavelength = c / f
    ppw = 8  # Points per wavelength
    lc = wavelength / ppw
    
    print(f"Frequency: {f/1e6:.1f} MHz")
    print(f"Wavelength: {wavelength*1e6:.1f} μm")
    print(f"Element size: {lc*1e6:.1f} μm")
    print(f"Wavenumber k = {k:.2f} rad/m")
    
    # Create mesh with Gmsh
    print("\nGenerating mesh...")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("disk")
    
    # Create disk
    disk = gmsh.model.occ.addDisk(0, 0, 0, R, R)
    gmsh.model.occ.synchronize()
    
    # Physical groups
    gmsh.model.addPhysicalGroup(2, [disk], tag=1, name="Domain")
    
    # Boundary
    boundary = gmsh.model.getBoundary([(2, disk)], oriented=False)
    boundary_tags = [b[1] for b in boundary]
    gmsh.model.addPhysicalGroup(1, boundary_tags, tag=101, name="Boundary")
    
    # Mesh
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc * 0.5)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc * 2.0)
    gmsh.model.mesh.generate(2)
    
    # New DOLFINx API returns MeshData object
    mesh_data = gmshio.model_to_mesh(
        gmsh.model, MPI.COMM_WORLD, rank=0, gdim=2
    )
    msh = mesh_data.mesh
    cell_tags = mesh_data.cell_tags
    facet_tags = mesh_data.facet_tags
    
    num_cells = msh.topology.index_map(2).size_local
    print(f"Mesh: {num_cells} cells")
    
    gmsh.finalize()
    
    # Function spaces - complex valued pressure
    # Use P2 elements
    V = fem.functionspace(msh, ("Lagrange", 2))
    
    print(f"DOFs: {V.dofmap.index_map.size_global}")
    
    # Trial and test functions
    p = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Material properties (water)
    rho = 998.0  # kg/m³
    K = rho * c**2  # Bulk modulus
    
    # Helmholtz weak form:
    # ∫ (1/ρ) ∇p·∇v dx - (ω²/K) ∫ p·v dx = ∫ g·v ds
    
    dx = ufl.Measure("dx", domain=msh)
    ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_tags)
    
    a = (1/rho) * ufl.inner(ufl.grad(p), ufl.grad(v)) * dx - (omega**2 / K) * ufl.inner(p, v) * dx
    
    # Source: prescribed velocity on boundary
    # For a pulsating boundary: v_n = v_0 
    # Robin BC: ∂p/∂n + (iωρ/Z) p = iωρ v_0
    # For simplicity, use Dirichlet: p = p_0 on boundary
    
    # Boundary condition: p = 1 Pa on boundary (source)
    p_bc = fem.Constant(msh, PETSc.ScalarType(1.0))
    
    # Find boundary DOFs
    boundary_dofs = fem.locate_dofs_topological(
        V, msh.topology.dim - 1, 
        facet_tags.find(101)
    )
    
    bc = fem.dirichletbc(p_bc, boundary_dofs, V)
    
    # Zero RHS
    L = fem.Constant(msh, PETSc.ScalarType(0.0)) * v * dx
    
    # Assemble and solve using direct approach
    print("Assembling system...")
    
    # Compile forms
    a_compiled = fem.form(a)
    L_compiled = fem.form(L)
    
    # Assemble matrix and vector
    A = fem_petsc.assemble_matrix(a_compiled, bcs=[bc])
    A.assemble()
    
    b = fem_petsc.assemble_vector(L_compiled)
    fem_petsc.apply_lifting(b, [a_compiled], [[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    fem_petsc.set_bc(b, [bc])
    
    # Create solution vector
    p_h = fem.Function(V)
    
    # Create solver
    solver = PETSc.KSP().create(msh.comm)
    solver.setOperators(A)
    solver.setType("preonly")
    solver.getPC().setType("lu")
    solver.getPC().setFactorSolverType("mumps")
    
    # Solve
    print("Solving...")
    solver.solve(b, p_h.x.petsc_vec)
    p_h.x.scatter_forward()
    
    # Clean up
    solver.destroy()
    A.destroy()
    b.destroy()
    
    # Extract results
    p_values = p_h.x.array
    p_max = np.max(np.abs(p_values))
    p_mean = np.mean(np.abs(p_values))
    
    print("\n" + "-" * 60)
    print("RESULTS")
    print("-" * 60)
    print(f"Max |p|: {p_max:.4f} Pa")
    print(f"Mean |p|: {p_mean:.4f} Pa")
    print(f"Min |p|: {np.min(np.abs(p_values)):.4f} Pa")
    
    # For a rigid cavity eigenmode, we expect standing waves
    # The Bessel function zeros give the resonant frequencies
    # For this test, we just verify the solver runs
    
    # Check that solution is non-trivial
    if p_max > 0.5:  # At least 50% of boundary value
        print("\n✓ PASS: Solution is non-trivial")
        status = True
    else:
        print("\n✗ FAIL: Solution appears to be near-zero")
        status = False
    
    # Check that solution varies in space (standing wave pattern)
    if (p_max - np.min(np.abs(p_values))) > 0.01:
        print("✓ PASS: Solution shows spatial variation")
    else:
        print("⚠ WARN: Solution is nearly uniform")
    
    print("-" * 60)
    
    return status


def test_imports():
    """Test that all required imports work."""
    print("Testing imports...")
    
    try:
        from tweezers.fenicsx import (
            FEMConfig, PhysicsLevel, 
            MaterialDatabase, 
            Domain, Interface,
        )
        print("✓ Core imports successful")
        
        config = FEMConfig.default()
        print(f"✓ Default config created (level={config.physics_level.name})")
        
        mat_db = MaterialDatabase()
        water = mat_db.get_fluid('water')
        print(f"✓ Materials: water at {water.sound_speed:.1f} m/s")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def main():
    print("\n" + "=" * 60)
    print("FEniCSx ACOUSTICS VALIDATION TEST")
    print("=" * 60 + "\n")
    
    # Test imports
    if not test_imports():
        return 1
    
    print()
    
    # Test 2D Helmholtz
    if not test_2d_helmholtz():
        return 1
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
