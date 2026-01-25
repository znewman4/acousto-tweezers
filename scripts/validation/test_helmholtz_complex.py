#!/usr/bin/env python3
"""
Quick end-to-end validation test for acoustic simulation with complex backend.

This test verifies:
1. Mesh generation works
2. Helmholtz equation assembles with complex scalars
3. Solver produces non-zero pressure field
4. Basic diagnostics compute correctly

Usage:
    micromamba activate acousto-complex
    python scripts/validation/test_helmholtz_complex.py

Expected:
    - Non-zero pressure field
    - max|p| > 0
    - Complex-valued solution

Author: Acousto-Tweezers Project
Date: January 2026
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def run_simple_helmholtz_test():
    """
    Simple Helmholtz test in a box with point source.
    """
    print("=" * 60)
    print("SIMPLE HELMHOLTZ TEST (Complex Backend)")
    print("=" * 60)
    print()
    
    # Import FEniCSx
    from petsc4py import PETSc
    from mpi4py import MPI
    from dolfinx import mesh, fem
    import ufl
    
    print(f"PETSc.ScalarType = {PETSc.ScalarType}")
    
    # Create simple mesh
    print("\n[1] Creating mesh...")
    L = 0.01  # 1 cm box
    nx, ny, nz = 10, 10, 10
    domain = mesh.create_box(
        MPI.COMM_WORLD,
        [np.array([0, 0, 0]), np.array([L, L, L])],
        [nx, ny, nz],
        cell_type=mesh.CellType.tetrahedron
    )
    print(f"    Created {nx}x{ny}x{nz} box mesh")
    print(f"    Cells: {domain.topology.index_map(3).size_global}")
    
    # Function space
    print("\n[2] Setting up function space...")
    V = fem.functionspace(domain, ("Lagrange", 2))
    print(f"    DOFs: {V.dofmap.index_map.size_global}")
    
    # Parameters
    print("\n[3] Setting up Helmholtz problem...")
    omega = 2 * np.pi * 1e6  # 1 MHz
    c = 1480  # Water
    rho = 1000
    k = omega / c
    K = rho * c**2
    wavelength = 2 * np.pi / k
    
    print(f"    Frequency: {omega / (2*np.pi) / 1e6:.2f} MHz")
    print(f"    Wavenumber: {k:.1f} rad/m")
    print(f"    Wavelength: {wavelength * 1e3:.3f} mm")
    print(f"    k*L = {k * L:.2f}")
    
    # Variational form
    # Helmholtz: ∇²p + k²p = f
    # Weak form (complex): ∫ ∇v̄·∇p dx - k² ∫ v̄ p dx = ∫ v̄ f dx
    # 
    # IMPORTANT: In DOLFINx with complex scalars, ufl.inner(v, p) automatically
    # handles conjugation. We need to be careful about the form structure.
    
    p = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Bilinear form (with complex k for slight damping)
    # inner(grad(u), grad(v)) = ∑ᵢ (∂u/∂xᵢ)(∂v̄/∂xᵢ) with conjugation
    k_complex = fem.Constant(domain, PETSc.ScalarType(k + 0.01j * k))
    
    # Standard Helmholtz sesquilinear form
    # a(p, v) = ∫ ∇p · ∇v̄ dx - k² ∫ p v̄ dx
    a = ufl.inner(ufl.grad(p), ufl.grad(v)) * ufl.dx - k_complex**2 * ufl.inner(p, v) * ufl.dx
    
    # Source term - Gaussian near center
    x = ufl.SpatialCoordinate(domain)
    x0, y0, z0 = L/2, L/2, L/2
    sigma = L / 20
    f_expr = ufl.exp(-((x[0]-x0)**2 + (x[1]-y0)**2 + (x[2]-z0)**2) / (2*sigma**2))
    f = fem.Constant(domain, PETSc.ScalarType(1e6))  # 1 MPa amplitude
    L_form = ufl.inner(f * f_expr, v) * ufl.dx
    
    # Absorbing BC on boundary
    # First-order ABC: ∂p/∂n + ik p = 0
    # In weak form: adds ik ∫_Γ p v̄ ds
    a = a + 1j * k * ufl.inner(p, v) * ufl.ds
    
    # Compile forms
    print("\n[4] Compiling forms...")
    a_compiled = fem.form(a)
    L_compiled = fem.form(L_form)
    print("    Forms compiled successfully")
    
    # Assemble
    print("\n[5] Assembling system...")
    from dolfinx.fem.petsc import assemble_matrix, assemble_vector
    
    A = assemble_matrix(a_compiled)
    A.assemble()
    
    b = assemble_vector(L_compiled)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    
    print(f"    Matrix size: {A.getSize()}")
    print(f"    Matrix type: {A.getType()}")
    
    # Check matrix is complex
    diag = A.getDiagonal()
    diag_vals = diag.array
    print(f"    Diagonal dtype: {diag_vals.dtype}")
    print(f"    Diagonal sample: {diag_vals[0]}")
    
    # Solve
    print("\n[6] Solving...")
    p_h = fem.Function(V)
    
    solver = PETSc.KSP().create(MPI.COMM_WORLD)
    solver.setOperators(A)
    solver.setType("preonly")
    pc = solver.getPC()
    pc.setType("lu")
    pc.setFactorSolverType("mumps")
    solver.setFromOptions()
    
    solver.solve(b, p_h.x.petsc_vec)
    p_h.x.scatter_forward()
    
    converged = solver.getConvergedReason()
    print(f"    Converged reason: {converged}")
    
    # Analyze solution
    print("\n[7] Analyzing solution...")
    p_array = p_h.x.array
    
    print(f"    Solution dtype: {p_array.dtype}")
    print(f"    Solution size: {len(p_array)}")
    
    p_abs = np.abs(p_array)
    p_real = np.real(p_array)
    p_imag = np.imag(p_array)
    
    print(f"    max|p|: {np.max(p_abs):.4e} Pa")
    print(f"    mean|p|: {np.mean(p_abs):.4e} Pa")
    print(f"    rms|p|: {np.sqrt(np.mean(p_abs**2)):.4e} Pa")
    print(f"    Re(p) range: [{np.min(p_real):.4e}, {np.max(p_real):.4e}] Pa")
    print(f"    Im(p) range: [{np.min(p_imag):.4e}, {np.max(p_imag):.4e}] Pa")
    print(f"    Has imaginary part: {np.max(np.abs(p_imag)) > 1e-15}")
    
    # Validation
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    
    passed = True
    
    # Test 1: Non-zero solution
    if np.max(p_abs) > 1e-10:
        print("[PASS] Solution is non-zero")
    else:
        print("[FAIL] Solution is zero or near-zero")
        passed = False
    
    # Test 2: Complex-valued
    if np.iscomplexobj(p_array):
        print("[PASS] Solution is complex-valued")
    else:
        print("[FAIL] Solution is not complex-valued")
        passed = False
    
    # Test 3: Has imaginary part (due to damping/ABC)
    if np.max(np.abs(p_imag)) > 1e-10 * np.max(p_abs):
        print("[PASS] Solution has non-trivial imaginary part")
    else:
        print("[WARN] Solution imaginary part is very small")
    
    # Test 4: Physically reasonable magnitude
    # For 1 MPa source, expect some acoustic pressure
    if 1e-3 < np.max(p_abs) < 1e12:
        print("[PASS] Pressure magnitude is physically reasonable")
    else:
        print("[WARN] Pressure magnitude may be unrealistic")
    
    print()
    if passed:
        print("✓ All critical tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(run_simple_helmholtz_test())
