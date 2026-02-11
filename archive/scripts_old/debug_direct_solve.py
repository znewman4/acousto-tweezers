"""Debug: solve streaming with DIRECT solver to get ground truth."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import numpy as np
import time

from acoustweezers.experiments.shallow_square_dish.config import ShallowDishConfig
from acoustweezers.experiments.shallow_square_dish.solve_pressure import create_mesh, solve_helmholtz

import dolfinx.fem as fem
import dolfinx.fem.petsc
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
import ufl
from ufl import inner, grad, div, dx
import basix.ufl
from petsc4py import PETSc
from dolfinx import mesh as dmesh

# Small deposition mesh
cfg = ShallowDishConfig()
cfg.L = 0.01
cfg.H = 0.001
cfg.nx = 20
cfg.ny = 20
cfg.nz = 8

domain, facet_tags, tag_map = create_mesh(cfg, verbose=True)
p_sol = solve_helmholtz(domain, facet_tags, cfg, mode="vortex", verbose=True)

rho = cfg.rho
omega = cfg.omega
mu = cfg.mu

p_func = p_sol.p_function
grad_p = ufl.grad(p_func)
v1_ufl = grad_p / (1j * omega * rho)
v1_conj = ufl.conj(v1_ufl)
stress_complex = 0.5 * rho * ufl.outer(v1_ufl, v1_conj)
stress_ufl = ufl.real(stress_complex)
f_ufl = -ufl.div(stress_ufl)

# Project forcing (with scaling)
V_forcing = fem.functionspace(domain, ("Lagrange", 1, (3,)))
f_expr = fem.Expression(f_ufl, V_forcing.element.interpolation_points())
f_func = fem.Function(V_forcing)
f_func.interpolate(f_expr)
f_func.x.array[:] = np.real(f_func.x.array)

f_vals = f_func.x.array.copy()
n_f = len(f_vals) // 3
f_mag = np.linalg.norm(np.real(f_vals).reshape((n_f, 3)), axis=1)
max_f = float(np.max(f_mag))
print(f"\nForcing: max |f| = {max_f:.3e} Pa/m")

# Taylor-Hood
cell_type = domain.basix_cell()
P2_vec = basix.ufl.element("Lagrange", cell_type, 2, shape=(3,))
P1 = basix.ufl.element("Lagrange", cell_type, 1)
TH = basix.ufl.mixed_element([P2_vec, P1])
W = fem.functionspace(domain, TH)

(u, p_trial) = ufl.TrialFunctions(W)
(w, q) = ufl.TestFunctions(W)

a_form = mu * inner(grad(u), grad(w)) * dx - inner(p_trial, div(w)) * dx - inner(div(u), q) * dx

# Use the f_func directly (no additional scaling)
L_form = inner(f_func, w) * dx

# BCs: no-slip on bottom and sides, u_z=0 on top
fdim = domain.topology.dim - 1
bcs = []
W0 = W.sub(0)
W0_collapsed, _ = W0.collapse()

# Create zero function for Dirichlet BCs
u_zero_func = fem.Function(W0_collapsed)
u_zero_func.x.array[:] = 0.0

H = cfg.H
L_dim = cfg.L

# No-slip on bottom
def bottom(x): return np.isclose(x[2], 0.0, atol=H*1e-6)
bottom_facets = dmesh.locate_entities_boundary(domain, fdim, bottom)
bottom_dofs = fem.locate_dofs_topological((W0, W0_collapsed), fdim, bottom_facets)
bc_bottom = fem.dirichletbc(u_zero_func, bottom_dofs, W0)
bcs.append(bc_bottom)

# No-slip on sides
def x0(x): return np.isclose(x[0], 0.0, atol=L_dim*1e-6)
def xL(x): return np.isclose(x[0], L_dim, atol=L_dim*1e-6)
def y0(x): return np.isclose(x[1], 0.0, atol=L_dim*1e-6)
def yL(x): return np.isclose(x[1], L_dim, atol=L_dim*1e-6)

for loc in [x0, xL, y0, yL]:
    facets_t = dmesh.locate_entities_boundary(domain, fdim, loc)
    dofs = fem.locate_dofs_topological((W0, W0_collapsed), fdim, facets_t)
    bc = fem.dirichletbc(u_zero_func, dofs, W0)
    bcs.append(bc)

# Top: u_z=0 (free-slip simplified)
def top(x): return np.isclose(x[2], H, atol=H*1e-6)
top_facets = dmesh.locate_entities_boundary(domain, fdim, top)
W0_z_collapsed, _ = W0.sub(2).collapse()
top_dofs_z = fem.locate_dofs_topological((W0.sub(2), W0_z_collapsed), fdim, top_facets)
u_z_zero_func = fem.Function(W0_z_collapsed)
u_z_zero_func.x.array[:] = 0.0
bc_top = fem.dirichletbc(u_z_zero_func, top_dofs_z, W0.sub(2))
bcs.append(bc_top)

# Assemble
a_compiled = fem.form(a_form)
L_compiled = fem.form(L_form)

A = assemble_matrix(a_compiled, bcs=bcs)
A.assemble()

b = assemble_vector(L_compiled)
fem.petsc.apply_lifting(b, [a_compiled], [bcs])
b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
fem.petsc.set_bc(b, bcs)

print(f"\nAfter BCs:")
print(f"  ||b||_2 = {b.norm():.6e}")
print(f"  ||b||_inf = {b.norm(PETSc.NormType.NORM_INFINITY):.6e}")
print(f"  ||A||_inf = {A.norm(PETSc.NormType.NORM_INFINITY):.6e}")

# Direct solve with MUMPS
print(f"\nSolving with direct solver (MUMPS)...")
ksp = PETSc.KSP().create(domain.comm)
ksp.setOperators(A)
ksp.setType("preonly")
pc = ksp.getPC()
pc.setType("lu")
pc.setFactorSolverType("mumps")

wh = fem.Function(W)
t0 = time.time()
ksp.solve(b, wh.x.petsc_vec)
wh.x.scatter_forward()
t_solve = time.time() - t0

reason = ksp.getConvergedReason()
print(f"  Solve time: {t_solve:.2f} s")
print(f"  KSP reason: {reason}")

# Extract velocity
u_h = wh.sub(0).collapse()
u_arr = np.real(u_h.x.array)
n_u = len(u_arr) // 3
u_mag = np.linalg.norm(u_arr.reshape((n_u, 3)), axis=1)
max_u = float(np.max(u_mag))
mean_u = float(np.mean(u_mag))
median_u = float(np.median(u_mag))

print(f"\n{'='*60}")
print(f"STREAMING VELOCITY (DIRECT SOLVER - ground truth):")
print(f"  max |u| = {max_u:.6e} m/s = {max_u*1e6:.4f} μm/s")
print(f"  mean |u| = {mean_u:.6e} m/s = {mean_u*1e6:.4f} μm/s")
print(f"  median |u| = {median_u:.6e} m/s = {median_u*1e6:.4f} μm/s")
print(f"  Expected (f*H²/μ): {max_f * cfg.H**2 / mu:.6e} m/s = {max_f * cfg.H**2 / mu * 1e6:.4f} μm/s")

# Also extract pressure
p_h = wh.sub(1).collapse()
p_arr = np.real(p_h.x.array)
print(f"\n  Streaming pressure:")
print(f"  max |p| = {np.max(np.abs(p_arr)):.6e} Pa")
print(f"  mean p = {np.mean(p_arr):.6e} Pa")

# Check residual
res = b.duplicate()
A.mult(wh.x.petsc_vec, res)
res.axpy(-1.0, b)
print(f"\n  Residual ||Ax-b||/||b|| = {res.norm()/b.norm():.3e}")

A.destroy()
b.destroy()
ksp.destroy()
