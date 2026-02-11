#!/usr/bin/env python3
"""
Minimal test: Check if -div(stress) assembles to zero in mixed Stokes.
"""

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, mesh
from dolfinx.fem.petsc import assemble_vector
import ufl
from ufl import inner, grad, div, dx
import basix.ufl

comm = MPI.COMM_WORLD

# Simple cube mesh
domain = mesh.create_unit_cube(comm, 5, 5, 5)
cell_type = domain.basix_cell()

# Taylor-Hood
P2_vec = basix.ufl.element("Lagrange", cell_type, 2, shape=(3,))
P1 = basix.ufl.element("Lagrange", cell_type, 1)
TH = basix.ufl.mixed_element([P2_vec, P1])
W = fem.functionspace(domain, TH)
(u_t, p_t) = ufl.TrialFunctions(W)
(w, q) = ufl.TestFunctions(W)

# Test 1: Constant forcing
print("--- Test 1: Constant body force f = (1, 0, 0) ---")
f_const = fem.Constant(domain, PETSc.ScalarType((1.0, 0.0, 0.0)))
L1 = inner(f_const, w) * dx
b1 = assemble_vector(fem.form(L1))
b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
print(f"  b norm: {b1.norm():.4e}")
print(f"  b max: {np.max(np.abs(b1.array)):.4e}")

# Test 2: Use -div(stress) where stress = outer(v, v)
# with v being a known function
print("\n--- Test 2: f = -div(stress) with known stress ---")
V_vec = fem.functionspace(domain, ("Lagrange", 2, (3,)))
v_func = fem.Function(V_vec)

# Set v to something non-trivial: v = (x, y, 0)
def set_v(x):
    vals = np.zeros((3, x.shape[1]))
    vals[0] = x[0]
    vals[1] = x[1]
    return vals

v_func.interpolate(set_v)

# stress = outer(v, v) = [[x², xy, 0], [xy, y², 0], [0, 0, 0]]
# div(stress) = [[2x, y], [x, 2y], [0, 0]] summed along rows
# div(stress)_x = ∂(x²)/∂x + ∂(xy)/∂y = 2x + x = 3x
# div(stress)_y = ∂(xy)/∂x + ∂(y²)/∂y = y + 2y = 3y
# So f = -div(stress) = (-3x, -3y, 0) — non-trivial!
stress = ufl.outer(v_func, v_func)
f_div = -ufl.div(stress)

L2 = inner(f_div, w) * dx
b2 = assemble_vector(fem.form(L2))
b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
print(f"  b norm: {b2.norm():.4e}")
print(f"  b max: {np.max(np.abs(b2.array)):.4e}")

# Test 3: Use -div(stress) where stress involves COMPLEX functions
print("\n--- Test 3: Complex v, stress = 0.5*Re(outer(v, conj(v))) ---")
V_vec_complex = fem.functionspace(domain, ("Lagrange", 2, (3,)))
v_complex = fem.Function(V_vec_complex)

def set_v_complex(x):
    vals = np.zeros((3, x.shape[1]), dtype=np.complex128)
    vals[0] = x[0] + 1j * x[1]
    vals[1] = x[1] - 1j * x[0]
    return vals

v_complex.interpolate(set_v_complex)

# Check that v is complex
print(f"  v dtype: {v_complex.x.array.dtype}")
print(f"  v max real: {np.max(np.abs(np.real(v_complex.x.array))):.4e}")
print(f"  v max imag: {np.max(np.abs(np.imag(v_complex.x.array))):.4e}")

# Method A: ufl.real(outer(v, conj(v)))
stress_A = 0.5 * ufl.real(ufl.outer(v_complex, ufl.conj(v_complex)))
f_A = -ufl.div(stress_A)
L3A = inner(f_A, w) * dx
b3A = assemble_vector(fem.form(L3A))
b3A.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
print(f"\n  Method A (ufl.real + ufl.conj):")
print(f"    b norm: {b3A.norm():.4e}")
print(f"    b max: {np.max(np.abs(b3A.array)):.4e}")

# Method B: manual Re(v)⊗Re(v) + Im(v)⊗Im(v)
v_re = ufl.real(v_complex)
v_im = ufl.imag(v_complex)
stress_B = 0.5 * (ufl.outer(v_re, v_re) + ufl.outer(v_im, v_im))
f_B = -ufl.div(stress_B)
L3B = inner(f_B, w) * dx
b3B = assemble_vector(fem.form(L3B))
b3B.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
print(f"\n  Method B (manual Re/Im decomposition):")
print(f"    b norm: {b3B.norm():.4e}")
print(f"    b max: {np.max(np.abs(b3B.array)):.4e}")

# Test 4: What if we DON'T use div, but instead use IBP form?
# inner(-div(stress), w) = inner(stress, grad(w)) - ∫(stress·n)·w ds
print("\n--- Test 4: IBP form: inner(stress, grad(w)) ---")
L4A = inner(stress_A, grad(w)) * dx
b4A = assemble_vector(fem.form(L4A))
b4A.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
print(f"  Method A IBP (ufl.real):")
print(f"    b norm: {b4A.norm():.4e}")
print(f"    b max: {np.max(np.abs(b4A.array)):.4e}")

L4B = inner(stress_B, grad(w)) * dx
b4B = assemble_vector(fem.form(L4B))
b4B.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
print(f"\n  Method B IBP (manual Re/Im):")
print(f"    b norm: {b4B.norm():.4e}")
print(f"    b max: {np.max(np.abs(b4B.array)):.4e}")

# Cleanup
for b in [b1, b2, b3A, b3B, b4A, b4B]:
    b.destroy()
