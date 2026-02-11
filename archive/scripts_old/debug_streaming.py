#!/usr/bin/env python3
"""
Debug script: Check why streaming solver produces zero velocity
despite non-zero forcing.
"""

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, mesh
import ufl
from ufl import inner, grad, div, dx
import basix.ufl

from acoustweezers.experiments.shallow_square_dish.config import ShallowDishConfig
from acoustweezers.experiments.shallow_square_dish.solve_pressure import (
    create_mesh, solve_helmholtz
)

comm = MPI.COMM_WORLD

# Create mesh and solve pressure
cfg = ShallowDishConfig()
cfg.Nx = 20
cfg.Ny = 20
cfg.Nz = 8
cfg.L = 10e-3
cfg.H = 1e-3
domain, facet_tags, _ = create_mesh(cfg, verbose=True)
p_sol = solve_helmholtz(domain, facet_tags, cfg, mode="combined", verbose=True)

# Streaming setup
rho = cfg.rho
omega = cfg.omega
mu = cfg.mu
H = cfg.H
L = cfg.L

# First-order velocity (UFL)
p_func = p_sol.p_function
grad_p = ufl.grad(p_func)
v1_ufl = grad_p / (1j * omega * rho)

# Reynolds stress
v1_conj = ufl.conj(v1_ufl)
stress_complex = 0.5 * rho * ufl.outer(v1_ufl, v1_conj)
stress_real = ufl.real(stress_complex)
f_ufl = -ufl.div(stress_real)

# Alternative: manual real/imag decomposition
v1_re = ufl.real(v1_ufl)
v1_im = ufl.imag(v1_ufl)
stress_alt = 0.5 * rho * (ufl.outer(v1_re, v1_re) + ufl.outer(v1_im, v1_im))
f_alt = -ufl.div(stress_alt)

# Project both forcings to check
V_forcing = fem.functionspace(domain, ("Lagrange", 1, (3,)))

print("\n--- Checking forcing method 1: ufl.real(v1 outer conj(v1)) ---")
f_expr1 = fem.Expression(f_ufl, V_forcing.element.interpolation_points())
f_func1 = fem.Function(V_forcing)
f_func1.interpolate(f_expr1)
f_vals1 = f_func1.x.array
print(f"  dtype: {f_vals1.dtype}")
print(f"  max abs(real): {np.max(np.abs(np.real(f_vals1))):.4e}")
print(f"  max abs(imag): {np.max(np.abs(np.imag(f_vals1))):.4e}")
n1 = len(f_vals1) // 3
f_mag1 = np.linalg.norm(np.real(f_vals1).reshape(n1, 3), axis=1)
print(f"  max |f|: {np.max(f_mag1):.4e} Pa/m")

print("\n--- Checking forcing method 2: Re(v1)⊗Re(v1) + Im(v1)⊗Im(v1) ---")
f_expr2 = fem.Expression(f_alt, V_forcing.element.interpolation_points())
f_func2 = fem.Function(V_forcing)
f_func2.interpolate(f_expr2)
f_vals2 = f_func2.x.array
print(f"  dtype: {f_vals2.dtype}")
print(f"  max abs(real): {np.max(np.abs(np.real(f_vals2))):.4e}")
print(f"  max abs(imag): {np.max(np.abs(np.imag(f_vals2))):.4e}")
n2 = len(f_vals2) // 3
f_mag2 = np.linalg.norm(np.real(f_vals2).reshape(n2, 3), axis=1)
print(f"  max |f|: {np.max(f_mag2):.4e} Pa/m")

# Now check: assemble the linear form and inspect the RHS vector
print("\n--- Checking assembled RHS vector ---")
cell_type = domain.basix_cell()
P2_vec = basix.ufl.element("Lagrange", cell_type, 2, shape=(3,))
P1 = basix.ufl.element("Lagrange", cell_type, 1)
TH = basix.ufl.mixed_element([P2_vec, P1])
W = fem.functionspace(domain, TH)

(u_t, p_t) = ufl.TrialFunctions(W)
(w, q) = ufl.TestFunctions(W)

# Method 1: original
L1 = inner(f_ufl, w) * dx
L1_form = fem.form(L1)
b1 = fem.petsc.assemble_vector(L1_form)
b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
print(f"\n  Method 1 RHS (ufl.real):")
print(f"    b norm: {b1.norm():.4e}")
print(f"    b max abs: {np.max(np.abs(b1.array)):.4e}")
print(f"    b max real: {np.max(np.abs(np.real(b1.array))):.4e}")
print(f"    b max imag: {np.max(np.abs(np.imag(b1.array))):.4e}")

# Method 2: manual decomposition
L2 = inner(f_alt, w) * dx
L2_form = fem.form(L2)
b2 = fem.petsc.assemble_vector(L2_form)
b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
print(f"\n  Method 2 RHS (manual Re/Im):")
print(f"    b norm: {b2.norm():.4e}")
print(f"    b max abs: {np.max(np.abs(b2.array)):.4e}")
print(f"    b max real: {np.max(np.abs(np.real(b2.array))):.4e}")
print(f"    b max imag: {np.max(np.abs(np.imag(b2.array))):.4e}")

# Check what ufl.real actually does to the expression
print("\n--- UFL expression analysis ---")
print(f"  f_ufl type: {type(f_ufl)}")
print(f"  f_alt type: {type(f_alt)}")
print(f"  stress_real type: {type(stress_real)}")
print(f"  stress_alt type: {type(stress_alt)}")

# Check if the issue is that f_ufl and f_alt differ
diff_vals = np.real(f_vals1) - np.real(f_vals2)
print(f"\n  Max |f_method1 - f_method2|: {np.max(np.abs(diff_vals)):.4e}")

b1.destroy()
b2.destroy()
