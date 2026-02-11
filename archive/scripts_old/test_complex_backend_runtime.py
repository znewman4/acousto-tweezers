#!/usr/bin/env python3
"""
Test Complex Backend Runtime

Verifies that the current Python environment has:
1. PETSc compiled with complex scalar support
2. DOLFINx capable of complex-valued assembly and solve

Usage:
    micromamba run -n acousto-complex python scripts/validation/test_complex_backend_runtime.py

Expected output:
    All assertions pass, confirming complex scalar support.
"""

from __future__ import annotations

import sys
import numpy as np


def test_petsc_complex_scalars() -> bool:
    """Test that PETSc is built with complex scalar type."""
    print("\n" + "="*70)
    print("TEST 1: PETSc Complex Scalar Type")
    print("="*70)
    
    from petsc4py import PETSc
    
    scalar_type = PETSc.ScalarType
    is_complex = np.issubdtype(scalar_type, np.complexfloating)
    
    print(f"  PETSc.ScalarType: {scalar_type}")
    print(f"  Is complex floating: {is_complex}")
    
    if not is_complex:
        print("\n  ✗ FAIL: PETSc is NOT built with complex scalars!")
        print("  You must use the 'acousto-complex' environment:")
        print("    micromamba activate acousto-complex")
        return False
    
    print("  ✓ PASS: PETSc has complex scalar support")
    return True


def test_dolfinx_complex_assembly() -> bool:
    """Test that DOLFINx can assemble and solve complex-valued forms."""
    print("\n" + "="*70)
    print("TEST 2: DOLFINx Complex Assembly & Solve")
    print("="*70)
    
    from mpi4py import MPI
    import dolfinx
    from dolfinx import fem, mesh
    from dolfinx.fem.petsc import LinearProblem
    import ufl
    from ufl import inner, grad, dx, TrialFunction, TestFunction
    
    print(f"  DOLFINx version: {dolfinx.__version__}")
    
    # Create simple 1D mesh
    domain = mesh.create_unit_interval(MPI.COMM_WORLD, 10)
    V = fem.functionspace(domain, ("Lagrange", 1))
    
    u = TrialFunction(V)
    v = TestFunction(V)
    
    # Complex-valued bilinear form: (1 + i) * u * v
    # This will fail if DOLFINx/PETSc doesn't support complex
    try:
        a = (1.0 + 1j) * inner(u, v) * dx
        L = inner(fem.Constant(domain, complex(1.0, 0.0)), v) * dx
        
        # Solve using LinearProblem
        # Note: API differs between DOLFINx versions
        problem = LinearProblem(
            a, L, bcs=[],
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
        )
        uh = problem.solve()
        
        # Check solution is complex
        solution_values = uh.x.array
        is_complex_solution = np.issubdtype(solution_values.dtype, np.complexfloating)
        
        print(f"  Solution dtype: {solution_values.dtype}")
        print(f"  Solution is complex: {is_complex_solution}")
        print(f"  Sample value: {solution_values[0]}")
        
        if not is_complex_solution:
            print("\n  ✗ FAIL: DOLFINx solution is not complex!")
            return False
        
        print("  ✓ PASS: DOLFINx complex assembly and solve works")
        return True
        
    except Exception as e:
        print(f"\n  ✗ FAIL: Complex assembly failed with error:")
        print(f"    {type(e).__name__}: {e}")
        return False


def test_complex_function_interpolation() -> bool:
    """Test that complex functions can be created and interpolated."""
    print("\n" + "="*70)
    print("TEST 3: Complex Function Interpolation")
    print("="*70)
    
    from mpi4py import MPI
    from dolfinx import fem, mesh
    import numpy as np
    
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 5, 5)
    V = fem.functionspace(domain, ("Lagrange", 1))
    
    # Create complex function
    f = fem.Function(V)
    
    # Interpolate complex expression: exp(i * x)
    def complex_expr(x):
        return np.exp(1j * x[0])
    
    try:
        f.interpolate(complex_expr)
        
        vals = f.x.array
        is_complex = np.issubdtype(vals.dtype, np.complexfloating)
        has_imag = np.any(np.abs(vals.imag) > 1e-10)
        
        print(f"  Function dtype: {vals.dtype}")
        print(f"  Has imaginary part: {has_imag}")
        print(f"  Sample value: {vals[0]}")
        
        if not is_complex or not has_imag:
            print("\n  ✗ FAIL: Complex interpolation failed!")
            return False
        
        print("  ✓ PASS: Complex function interpolation works")
        return True
        
    except Exception as e:
        print(f"\n  ✗ FAIL: Complex interpolation failed with error:")
        print(f"    {type(e).__name__}: {e}")
        return False


def main():
    """Run all complex backend tests."""
    print("\n" + "#"*70)
    print("# COMPLEX BACKEND RUNTIME VALIDATION")
    print("#"*70)
    
    # Run tests
    results = []
    results.append(("PETSc Complex Scalars", test_petsc_complex_scalars()))
    results.append(("DOLFINx Complex Assembly", test_dolfinx_complex_assembly()))
    results.append(("Complex Interpolation", test_complex_function_interpolation()))
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        all_passed = all_passed and passed
    
    print("\n" + "="*70)
    if all_passed:
        print("✓ ALL TESTS PASSED - Complex backend is ready")
        print("="*70 + "\n")
        return 0
    else:
        print("✗ SOME TESTS FAILED - Fix environment before proceeding")
        print("="*70 + "\n")
        print("To fix: Use the acousto-complex environment:")
        print("  micromamba activate acousto-complex")
        print("  # or")
        print("  micromamba run -n acousto-complex python <script>")
        return 1


if __name__ == "__main__":
    sys.exit(main())
