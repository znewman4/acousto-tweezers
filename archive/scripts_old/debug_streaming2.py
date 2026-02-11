#!/usr/bin/env python3
"""
Debug: check if pressure gradient / v1 produces meaningful forcing on the ACTUAL mesh.
Focus on WHY inner(-div(stress), w)*dx == 0 on the acoustic mesh.
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

# Small mesh
cfg = ShallowDishConfig()
cfg.Nx = 20
cfg.Ny = 20
cfg.Nz = 8
cfg.L = 10e-3
cfg.H = 1e-3
domain, facet_tags, _ = create_mesh(cfg, verbose=False)

# Solve Helmholtz
p_sol = solve_helmholtz(domain, facet_tags, cfg, mode="combined", verbose=False)
p_func = p_sol.p_function

# Check pressure solution
p_vals = p_func.x.array
print(f"Pressure: max|p| = {np.max(np.abs(p_vals)):.4f} Pa")
print(f"  dtype: {p_vals.dtype}")
print(f"  max Re: {np.max(np.abs(np.real(p_vals))):.4e}")
print(f"  max Im: {np.max(np.abs(np.imag(p_vals))):.4e}")

rho = cfg.rho
omega = cfg.omega

# v1 = grad(p) / (iωρ) 
grad_p = ufl.grad(p_func)
v1_ufl = grad_p / (1j * omega * rho)

# Check v1 values by interpolation
V_vec = fem.functionspace(domain, ("Lagrange", 1, (3,)))
v1_func = fem.Function(V_vec)
v1_expr = fem.Expression(v1_ufl, V_vec.element.interpolation_points())
v1_func.interpolate(v1_expr)
v1_vals = v1_func.x.array
print(f"\nFirst-order velocity v1:")
print(f"  dtype: {v1_vals.dtype}")
n_v = len(v1_vals) // 3
v1_3d = v1_vals.reshape(n_v, 3)
v1_mag = np.linalg.norm(np.abs(v1_3d), axis=1)
print(f"  max |v1|: {np.max(v1_mag):.4e} m/s = {np.max(v1_mag)*1e6:.4f} μm/s")
print(f"  max Re(v1): {np.max(np.abs(np.real(v1_3d))):.4e}")
print(f"  max Im(v1): {np.max(np.abs(np.imag(v1_3d))):.4e}")

# Method B: manual decomposition
v1_re = ufl.real(v1_ufl)
v1_im = ufl.imag(v1_ufl)

# Interpolate Re(v1) and Im(v1)
v1re_func = fem.Function(V_vec)
v1re_func.interpolate(fem.Expression(v1_re, V_vec.element.interpolation_points()))
v1im_func = fem.Function(V_vec)
v1im_func.interpolate(fem.Expression(v1_im, V_vec.element.interpolation_points()))

print(f"\n  max |Re(v1)|: {np.max(np.abs(v1re_func.x.array)):.4e}")
print(f"  max |Im(v1)|: {np.max(np.abs(v1im_func.x.array)):.4e}")

# Now: instead of using UFL div(stress), project forcing to a Function first,
# THEN use it in the RHS. This avoids the div(outer()) UFL expression tree issue.

print("\n--- Strategy: Project forcing to Function, then use in RHS ---")

# Compute stress components and project to a tensor function
# stress = 0.5 * rho * (Re(v1)⊗Re(v1) + Im(v1)⊗Im(v1))
# This is a symmetric 3x3 tensor. For the RHS, we need f = -div(stress).

# Alternative: use the IBP form directly
# inner(-div(stress), w)*dx = inner(stress, grad(w))*dx - ∫(stress·n)·w ds
# The surface integral vanishes if w=0 on boundaries (Dirichlet BCs applied later).
# BUT: in assembly without BCs, the surface integral is NOT zero!

# Actually, that's the key insight:
# When we assemble WITHOUT BCs, inner(-div(stress), w)*dx gives the volume integral.
# The divergence theorem says:
#   inner(-div(S), w)*dx = inner(S, grad(w))*dx - ∫(S·n)·w ds
# The inner(-div(S), w)*dx form does NOT include the surface integral.
# It's literally ∫(-div(S))·w dx.
# If div(S) is zero elementwise, then the integral is zero.
# But div(S) should NOT be zero for a non-uniform stress field!

# Let me check: is the stress field nearly uniform?
stress_expr = 0.5 * rho * (ufl.outer(v1re_func, v1re_func) + ufl.outer(v1im_func, v1im_func))

# Project div(stress) to see its values
f_force_func = fem.Function(V_vec)
f_force_expr = fem.Expression(-ufl.div(stress_expr), V_vec.element.interpolation_points())
f_force_func.interpolate(f_force_expr)
f_vals = np.real(f_force_func.x.array)
n_f = len(f_vals) // 3
f_3d = f_vals.reshape(n_f, 3)
f_mag = np.linalg.norm(f_3d, axis=1)

print(f"  Projected |f|: max={np.max(f_mag):.4e}, mean={np.mean(f_mag):.4e}")
print(f"  Non-zero entries: {np.sum(f_mag > 1e-15)}/{n_f}")

# NOW use the projected function in the RHS
cell_type = domain.basix_cell()
P2_vec = basix.ufl.element("Lagrange", cell_type, 2, shape=(3,))
P1 = basix.ufl.element("Lagrange", cell_type, 1)
TH = basix.ufl.mixed_element([P2_vec, P1])
W = fem.functionspace(domain, TH)
(u_t, p_t) = ufl.TrialFunctions(W)
(w, q) = ufl.TestFunctions(W)

# Method 1: Direct UFL expression in RHS
v1_ufl_from_p = ufl.grad(p_func) / (1j * omega * rho)
v1_re_ufl = ufl.real(v1_ufl_from_p)
v1_im_ufl = ufl.imag(v1_ufl_from_p)
stress_ufl = 0.5 * rho * (ufl.outer(v1_re_ufl, v1_re_ufl) + ufl.outer(v1_im_ufl, v1_im_ufl))
f_ufl = -ufl.div(stress_ufl)

L1 = inner(f_ufl, w) * dx
b1 = assemble_vector(fem.form(L1))
b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
print(f"\n  Direct UFL -div(stress) RHS:")
print(f"    b norm: {b1.norm():.4e}")

# Method 2: Use projected Function in RHS — skip for now due to element mismatch
print(f"\n  (Method 2 skipped — element mismatch)")

# Method 3: IBP form (no div)
L3 = inner(stress_ufl, grad(w)) * dx
b3 = assemble_vector(fem.form(L3))
b3.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
print(f"\n  IBP form inner(stress, grad(w)) RHS:")
print(f"    b norm: {b3.norm():.4e}")

# Method 4: Use pre-interpolated functions (not UFL expressions from p)
stress_from_funcs = 0.5 * rho * (ufl.outer(v1re_func, v1re_func) + ufl.outer(v1im_func, v1im_func))
f_from_funcs = -ufl.div(stress_from_funcs)
L4 = inner(f_from_funcs, w) * dx
b4 = assemble_vector(fem.form(L4))
b4.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
print(f"\n  Pre-interpolated v1 functions RHS:")
print(f"    b norm: {b4.norm():.4e}")

# Method 5: IBP with pre-interpolated functions
L5 = inner(stress_from_funcs, grad(w)) * dx
b5 = assemble_vector(fem.form(L5))
b5.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
print(f"\n  IBP with pre-interpolated v1 RHS:")
print(f"    b norm: {b5.norm():.4e}")

# Cleanup
for b in [b1, b3, b4, b5]:
    b.destroy()
