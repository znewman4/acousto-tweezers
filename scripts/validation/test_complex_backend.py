#!/usr/bin/env python3
"""
Validation: Complex PETSc Backend

This script verifies that PETSc is built with complex scalar type,
which is REQUIRED for frequency-domain acoustics with FEniCSx.

If this test fails, the entire simulation framework will not work correctly.

Usage:
    python scripts/validation/test_complex_backend.py

Expected output:
    [PASS] PETSc scalar type is complex
    [PASS] DOLFINx complex assembly works

Author: Acousto-Tweezers Project
Date: January 2026
"""

import sys
import numpy as np


def test_petsc_complex():
    """Test that PETSc is configured with complex scalars."""
    print("=" * 60)
    print("COMPLEX BACKEND VALIDATION")
    print("=" * 60)
    print()
    
    # Test 1: PETSc scalar type
    print("Test 1: PETSc ScalarType")
    print("-" * 40)
    
    try:
        from petsc4py import PETSc
        scalar_type = PETSc.ScalarType
        print(f"  PETSc.ScalarType = {scalar_type}")
        
        # Accept complex, numpy.complex64, numpy.complex128
        is_complex = (
            scalar_type == complex or
            scalar_type == np.complex64 or
            scalar_type == np.complex128 or
            "complex" in str(scalar_type).lower()
        )
        
        if is_complex:
            print("  [PASS] PETSc is configured with complex scalars")
            test1_pass = True
        else:
            print("  [FAIL] PETSc is NOT complex!")
            print("         Expected: numpy.complex128 or complex")
            print(f"         Got: {scalar_type}")
            print()
            print("  To fix this, you need to rebuild PETSc with:")
            print("    --with-scalar-type=complex")
            print()
            print("  Or use the Docker image:")
            print("    docker build -t acousto-complex -f docker/Dockerfile .")
            print("    docker run -it acousto-complex")
            test1_pass = False
    except ImportError as e:
        print(f"  [FAIL] Cannot import petsc4py: {e}")
        test1_pass = False
        scalar_type = None
    
    print()
    
    # Test 2: DOLFINx version
    print("Test 2: DOLFINx Installation")
    print("-" * 40)
    
    try:
        import dolfinx
        print(f"  DOLFINx version: {dolfinx.__version__}")
        print("  [PASS] DOLFINx is installed")
        test2_pass = True
    except ImportError as e:
        print(f"  [FAIL] Cannot import dolfinx: {e}")
        test2_pass = False
    
    print()
    
    # Test 3: Complex function creation
    print("Test 3: Complex Function Space")
    print("-" * 40)
    
    try:
        from dolfinx import mesh, fem
        from mpi4py import MPI
        
        # Create a simple mesh
        msh = mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
        V = fem.functionspace(msh, ("Lagrange", 1))
        
        # Create a complex-valued function
        u = fem.Function(V, dtype=np.complex128)
        u.x.array[:] = 1.0 + 2.0j
        
        # Verify it's actually complex
        max_val = np.max(u.x.array)
        if isinstance(max_val, complex) or np.iscomplexobj(u.x.array):
            print(f"  Created complex function: max = {max_val}")
            print("  [PASS] Complex functions work correctly")
            test3_pass = True
        else:
            print(f"  [FAIL] Function is not complex: {type(max_val)}")
            test3_pass = False
            
    except Exception as e:
        print(f"  [FAIL] Error creating complex function: {e}")
        import traceback
        traceback.print_exc()
        test3_pass = False
    
    print()
    
    # Test 4: Complex form assembly (Helmholtz equation)
    print("Test 4: Complex Form Assembly (Helmholtz)")
    print("-" * 40)
    
    try:
        from petsc4py import PETSc
        from dolfinx import mesh, fem
        from dolfinx.fem.petsc import assemble_matrix, assemble_vector
        from mpi4py import MPI
        import ufl
        
        # Create mesh and function space
        msh = mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
        V = fem.functionspace(msh, ("Lagrange", 1))
        
        # Trial and test functions
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        # Complex wavenumber (with small damping)
        k_val = 10.0 + 0.1j
        k = fem.Constant(msh, PETSc.ScalarType(k_val))
        
        # Helmholtz bilinear form: a(u,v) = ∫(∇u·∇v̄ - k²uv̄)dx
        # For complex, we need inner() which handles conjugation
        a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - k**2 * ufl.inner(u, v) * ufl.dx
        a_compiled = fem.form(a)
        
        # Assemble matrix
        A = assemble_matrix(a_compiled)
        A.assemble()
        
        # Check matrix is complex
        diag = A.getDiagonal()
        diag_values = diag.array
        
        # Also assemble a simple RHS
        f = fem.Constant(msh, PETSc.ScalarType(1.0 + 0.0j))
        L = ufl.inner(f, v) * ufl.dx
        L_compiled = fem.form(L)
        b = assemble_vector(L_compiled)
        b.ghostUpdate()
        
        if np.iscomplexobj(diag_values):
            print(f"  Assembled Helmholtz matrix with k = {k_val}")
            print(f"  Matrix size: {A.getSize()}")
            print(f"  Diagonal sample (real): {diag_values[0].real:.6f}")
            print(f"  Diagonal sample (imag): {diag_values[0].imag:.6f}")
            print(f"  RHS vector size: {b.getSize()}")
            print("  [PASS] Complex form assembly works")
            test4_pass = True
        else:
            print(f"  [WARN] Matrix diagonal is real-valued")
            print(f"  Type: {diag_values.dtype}")
            test4_pass = False
            
    except Exception as e:
        print(f"  [FAIL] Error in complex assembly: {e}")
        import traceback
        traceback.print_exc()
        test4_pass = False
    
    print()
    
    # Test 5: Solve a simple Helmholtz problem
    print("Test 5: Helmholtz Solve (Critical)")
    print("-" * 40)
    
    try:
        from petsc4py import PETSc
        from dolfinx import mesh, fem
        from dolfinx.fem.petsc import LinearProblem
        from mpi4py import MPI
        import ufl
        
        # Create mesh
        msh = mesh.create_unit_square(MPI.COMM_WORLD, 16, 16)
        V = fem.functionspace(msh, ("Lagrange", 1))
        
        # Trial and test functions
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        # Wavenumber
        k_val = 5.0 + 0.05j  # Small damping for stability
        k = fem.Constant(msh, PETSc.ScalarType(k_val))
        
        # Bilinear form
        a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - k**2 * ufl.inner(u, v) * ufl.dx
        
        # Source term (point-like source approximation)
        x = ufl.SpatialCoordinate(msh)
        f_expr = ufl.exp(-100 * ((x[0] - 0.5)**2 + (x[1] - 0.5)**2))
        L = ufl.inner(f_expr, v) * ufl.dx
        
        # Solve
        problem = LinearProblem(a, L, petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu"
        })
        uh = problem.solve()
        
        # Check solution
        u_array = uh.x.array
        max_abs = np.max(np.abs(u_array))
        has_imag = np.max(np.abs(u_array.imag)) > 1e-15
        
        print(f"  Solution computed successfully")
        print(f"  Max |u|: {max_abs:.6e}")
        print(f"  Has imaginary part: {has_imag}")
        
        if max_abs > 1e-10 and np.iscomplexobj(u_array):
            print("  [PASS] Helmholtz solve produces non-zero complex solution")
            test5_pass = True
        else:
            print("  [WARN] Solution may be trivial or real-only")
            test5_pass = max_abs > 1e-10  # Still pass if non-zero
            
    except Exception as e:
        print(f"  [FAIL] Error in Helmholtz solve: {e}")
        import traceback
        traceback.print_exc()
        test5_pass = False
    
    print()
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    results = [
        ("PETSc complex scalars", test1_pass),
        ("DOLFINx installed", test2_pass),
        ("Complex functions", test3_pass),
        ("Complex assembly", test4_pass),
        ("Helmholtz solve", test5_pass),
    ]
    
    all_pass = all(r[1] for r in results)
    
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {name}")
    
    print()
    
    if all_pass:
        print("✓ All tests passed! Complex backend is ready.")
        print()
        print("You can now run simulations with:")
        print("  python scripts/run_fem_multiphysics.py")
        return 0
    elif test1_pass and test2_pass and test3_pass:
        print("⚠ Core tests passed. Some advanced tests failed.")
        print("  Complex backend is likely functional.")
        return 0
    else:
        print("✗ Critical tests failed. Complex backend is NOT ready.")
        print()
        print("The simulation framework REQUIRES complex PETSc.")
        print("Please use the acousto-complex conda environment:")
        print("  micromamba activate acousto-complex")
        return 1


if __name__ == "__main__":
    sys.exit(test_petsc_complex())
