#!/usr/bin/env python3
"""
Environment validation: Complex PETSc gate.

This test MUST pass before any other validation tests run.
Time-harmonic acoustics with PML requires complex scalars.

Usage:
    python scripts/validation/test_env_complex_petsc.py

Expected output:
    Complex: True

If this fails, reinstall environment:
    micromamba env create -f environment/complex-fenicsx.yml
    micromamba activate acousto-complex
"""

import sys

def main():
    print("=" * 70)
    print("ENVIRONMENT VALIDATION: Complex PETSc")
    print("=" * 70)
    print()
    
    # Check numpy first
    try:
        import numpy as np
        print(f"numpy version: {np.__version__}")
    except ImportError as e:
        print(f"FATAL: Cannot import numpy: {e}")
        sys.exit(1)
    
    # Check dolfinx
    try:
        import dolfinx
        print(f"dolfinx version: {dolfinx.__version__}")
    except ImportError as e:
        print(f"FATAL: Cannot import dolfinx: {e}")
        print("\nInstall with:")
        print("  micromamba env create -f environment/complex-fenicsx.yml")
        print("  micromamba activate acousto-complex")
        sys.exit(1)
    
    # Check PETSc
    try:
        from petsc4py import PETSc
        print(f"PETSc.ScalarType: {PETSc.ScalarType}")
    except ImportError as e:
        print(f"FATAL: Cannot import petsc4py: {e}")
        print("\nInstall with:")
        print("  micromamba env create -f environment/complex-fenicsx.yml")
        print("  micromamba activate acousto-complex")
        sys.exit(1)
    
    # The critical check
    is_complex = np.issubdtype(PETSc.ScalarType, np.complexfloating)
    print(f"Complex: {is_complex}")
    print()
    
    if not is_complex:
        print("=" * 70)
        print("FAIL: PETSc is NOT complex!")
        print("=" * 70)
        print()
        print(f"Current PETSc.ScalarType: {PETSc.ScalarType}")
        print("Required: numpy.complex128 (or complex64)")
        print()
        print("Time-harmonic acoustics with PML REQUIRES complex scalars.")
        print("The PML coordinate stretching s_x = 1 + i*sigma/omega is complex-valued.")
        print()
        print("FIX: Reinstall environment with complex PETSc:")
        print()
        print("  micromamba env create -f environment/complex-fenicsx.yml")
        print("  micromamba activate acousto-complex")
        print()
        print("Or manually install complex PETSc:")
        print()
        print("  micromamba install -c conda-forge petsc=3.21.*=*complex* petsc4py=3.21.*=*complex*")
        print()
        print("=" * 70)
        sys.exit(1)
    
    # Additional checks
    print("Additional environment info:")
    print("-" * 40)
    
    try:
        import gmsh
        print(f"gmsh version: {gmsh.__version__ if hasattr(gmsh, '__version__') else 'available'}")
    except ImportError:
        print("gmsh: NOT INSTALLED (needed for meshing)")
    
    try:
        import pyvista
        print(f"pyvista version: {pyvista.__version__}")
    except ImportError:
        print("pyvista: NOT INSTALLED (needed for visualization)")
    
    try:
        from mpi4py import MPI
        print(f"mpi4py: available (size={MPI.COMM_WORLD.size})")
    except ImportError:
        print("mpi4py: NOT INSTALLED")
    
    try:
        import basix
        print(f"basix version: {basix.__version__}")
    except ImportError:
        print("basix: NOT INSTALLED")
    
    print()
    print("=" * 70)
    print("PASS: Complex PETSc environment verified")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
