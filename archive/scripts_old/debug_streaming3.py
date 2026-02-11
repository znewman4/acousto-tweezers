#!/usr/bin/env python3
"""
Minimal test: is the forcing non-zero when assembled in a NON-MIXED space?
This isolates whether the issue is in the mixed element or the forcing itself.
"""

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, mesh
from dolfinx.fem.petsc import assemble_vector
import ufl
from ufl import inner, grad, div, dx
import basix.ufl

from acoustweezers.experiments.shallow_square_dish.config import ShallowDishConfig
from acoustweezers.experiments.shallow_square_dish.solve_pressure import (
    create_mesh, solve_helmholtz
)

comm = MPI.COMM_WORLD

cfg = ShallowDishConfig()
cfg.Nx = 20
cfg.Ny = 20
cfg.Nz = 8
cfg.L = 10e-3
cfg.H = 1e-3
domain, facet_tags, _ = create_mesh(cfg, verbose=False)

p_sol = solve_helmholtz(domain, facet_tags, cfg, mode="combined", verbose=False)
p_func = p_sol.p_function

rho = cfg.rho
omega = cfg.omega

# Build stress from pre-interpolated v1 functions
V_vec = fem.functionspace(domain, ("Lagrange", 1, (3,)))

v1_ufl = ufl.grad(p_func) / (1j * omega * rho)
v1re_func = fem.Function(V_vec)
v1re_func.interpolate(fem.Expression(ufl.real(v1_ufl), V_vec.element.interpolation_points()))
v1im_func = fem.Function(V_vec)
v1im_func.interpolate(fem.Expression(ufl.imag(v1_ufl), V_vec.element.interpolation_points()))

# Make them explicitly real-valued
v1re_func.x.array[:] = np.real(v1re_func.x.array)
v1im_func.x.array[:] = np.real(v1im_func.x.array)

print(f"v1re: dtype={v1re_func.x.array.dtype}, max={np.max(np.abs(v1re_func.x.array)):.4e}")
print(f"v1im: dtype={v1im_func.x.array.dtype}, max={np.max(np.abs(v1im_func.x.array)):.4e}")

stress = 0.5 * rho * (ufl.outer(v1re_func, v1re_func) + ufl.outer(v1im_func, v1im_func))
f_force = -ufl.div(stress)

# Test 1: Non-mixed P2 vector space
print("\n--- Test 1: Simple P2 vector space (not mixed) ---")
V_test = fem.functionspace(domain, ("Lagrange", 2, (3,)))
w_simple = ufl.TestFunction(V_test)

L_simple = inner(f_force, w_simple) * dx
b_simple = assemble_vector(fem.form(L_simple))
b_simple.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
print(f"  b norm: {b_simple.norm():.4e}")
print(f"  b max: {np.max(np.abs(b_simple.array)):.4e}")

# Test 2: Non-mixed P1 vector space (same order as stress functions)
print("\n--- Test 2: Simple P1 vector space ---")
V_test1 = fem.functionspace(domain, ("Lagrange", 1, (3,)))
w_simple1 = ufl.TestFunction(V_test1)

L_simple1 = inner(f_force, w_simple1) * dx
b_simple1 = assemble_vector(fem.form(L_simple1))
b_simple1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
print(f"  b norm: {b_simple1.norm():.4e}")
print(f"  b max: {np.max(np.abs(b_simple1.array)):.4e}")

# Test 3: IBP form in P2 vector space
print("\n--- Test 3: IBP form in P2 vector space ---")
L_ibp = inner(stress, grad(w_simple)) * dx
b_ibp = assemble_vector(fem.form(L_ibp))
b_ibp.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
print(f"  b norm: {b_ibp.norm():.4e}")
print(f"  b max: {np.max(np.abs(b_ibp.array)):.4e}")

# Test 4: Constant forcing in mixed space (sanity check)
print("\n--- Test 4: Constant forcing in mixed space ---")
cell_type = domain.basix_cell()
P2_vec = basix.ufl.element("Lagrange", cell_type, 2, shape=(3,))
P1 = basix.ufl.element("Lagrange", cell_type, 1)
TH = basix.ufl.mixed_element([P2_vec, P1])
W = fem.functionspace(domain, TH)
(u_t, p_t) = ufl.TrialFunctions(W)
(w_mix, q) = ufl.TestFunctions(W)

f_const = fem.Constant(domain, PETSc.ScalarType((1.0, 0.0, 0.0)))
L_const = inner(f_const, w_mix) * dx
b_const = assemble_vector(fem.form(L_const))
b_const.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
print(f"  b norm: {b_const.norm():.4e}")
print(f"  b max: {np.max(np.abs(b_const.array)):.4e}")

# Test 5: Our actual forcing in the MIXED space
print("\n--- Test 5: Actual forcing in mixed space ---")
L_mix = inner(f_force, w_mix) * dx
b_mix = assemble_vector(fem.form(L_mix))
b_mix.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
print(f"  b norm: {b_mix.norm():.4e}")
print(f"  b max: {np.max(np.abs(b_mix.array)):.4e}")

# Test 6: Use the projected forcing function (not UFL expression)
print("\n--- Test 6: Projected forcing function in mixed space ---")
f_proj = fem.Function(V_vec)
f_proj.interpolate(fem.Expression(f_force, V_vec.element.interpolation_points()))
f_proj.x.array[:] = np.real(f_proj.x.array)
print(f"  f_proj max: {np.max(np.abs(f_proj.x.array)):.4e}")

L_proj = inner(f_proj, w_mix) * dx
b_proj = assemble_vector(fem.form(L_proj))
b_proj.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
print(f"  b norm: {b_proj.norm():.4e}")
print(f"  b max: {np.max(np.abs(b_proj.array)):.4e}")

# Test 7: Use the projected forcing function in non-mixed space
print("\n--- Test 7: Projected forcing function in P2 vector space ---")
L_proj2 = inner(f_proj, w_simple) * dx
b_proj2 = assemble_vector(fem.form(L_proj2))
b_proj2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
print(f"  b norm: {b_proj2.norm():.4e}")
print(f"  b max: {np.max(np.abs(b_proj2.array)):.4e}")

for b in [b_simple, b_simple1, b_ibp, b_const, b_mix, b_proj, b_proj2]:
    b.destroy()
