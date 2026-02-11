"""Debug: check assembled RHS vector norm in streaming solver."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import numpy as np

from acoustweezers.experiments.shallow_square_dish.config import ShallowDishConfig
from acoustweezers.experiments.shallow_square_dish.solve_pressure import create_mesh, solve_helmholtz

cfg = ShallowDishConfig()
domain, facet_tags, _ = create_mesh(cfg, verbose=True)

# Solve pressure
p_sol = solve_helmholtz(domain, facet_tags, cfg, mode="vortex", verbose=True)

# Now manually step through streaming to check the RHS
import dolfinx.fem as fem
import dolfinx.fem.petsc
import ufl
from ufl import inner, grad, div, dx
import basix.ufl
from petsc4py import PETSc

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

# Create a simple vector space and interpolate
V_forcing = fem.functionspace(domain, ("Lagrange", 1, (3,)))
f_expr = fem.Expression(f_ufl, V_forcing.element.interpolation_points())
f_func = fem.Function(V_forcing)
f_func.interpolate(f_expr)
f_func.x.array[:] = np.real(f_func.x.array)

f_vals = f_func.x.array.copy()
n_f = len(f_vals) // 3
f_mag = np.linalg.norm(np.real(f_vals).reshape((n_f, 3)), axis=1)
max_f = float(np.max(f_mag))
print(f"\nForcing stats BEFORE scaling:")
print(f"  max |f| = {max_f:.3e}")
print(f"  f_func dtype: {f_func.x.array.dtype}")
print(f"  f_func min/max: {np.min(np.real(f_func.x.array)):.3e} / {np.max(np.real(f_func.x.array)):.3e}")
print(f"  f_func imag min/max: {np.min(np.imag(f_func.x.array)):.3e} / {np.max(np.imag(f_func.x.array)):.3e}")

numerical_scale = 1.0 / max_f if max_f > 0 else 1.0
f_func.x.array[:] *= numerical_scale
print(f"\nAfter scaling (scale = {numerical_scale:.2e}):")
print(f"  f_func min/max: {np.min(np.real(f_func.x.array)):.3e} / {np.max(np.real(f_func.x.array)):.3e}")

# Create Taylor-Hood mixed space
cell_type = domain.basix_cell()
P2_vec = basix.ufl.element("Lagrange", cell_type, 2, shape=(3,))
P1 = basix.ufl.element("Lagrange", cell_type, 1)
TH = basix.ufl.mixed_element([P2_vec, P1])
W = fem.functionspace(domain, TH)

(u, p_trial) = ufl.TrialFunctions(W)
(w, q) = ufl.TestFunctions(W)

a_form_ufl = (
    mu * inner(grad(u), grad(w)) * dx
    - inner(p_trial, div(w)) * dx
    - inner(div(u), q) * dx
)

L_form_ufl = inner(f_func, w) * dx

# No BCs first — just check the raw assembly
a_compiled = fem.form(a_form_ufl)
L_compiled = fem.form(L_form_ufl)

b = fem.petsc.assemble_vector(L_compiled)
b.assemble()

print(f"\nAssembled RHS (no BCs):")
print(f"  ||b||_2 = {b.norm():.6e}")
print(f"  ||b||_inf = {b.norm(PETSc.NormType.NORM_INFINITY):.6e}")
print(f"  size = {b.getSize()}")

b_arr = b.getArray()
print(f"  b array dtype: {b_arr.dtype}")
print(f"  b real min/max: {np.min(np.real(b_arr)):.3e} / {np.max(np.real(b_arr)):.3e}")
print(f"  b imag min/max: {np.min(np.imag(b_arr)):.3e} / {np.max(np.imag(b_arr)):.3e}")
print(f"  # nonzero entries (|b_i| > 1e-15): {np.sum(np.abs(b_arr) > 1e-15)}")

# Also assemble the matrix to check
A = fem.petsc.assemble_matrix(a_compiled)
A.assemble()
print(f"\nAssembled matrix A:")
print(f"  ||A||_inf = {A.norm(PETSc.NormType.NORM_INFINITY):.6e}")

# Also try direct UFL assembly (NOT using projected function)
L_direct = inner(f_ufl, w) * dx
L_direct_compiled = fem.form(L_direct)
b_direct = fem.petsc.assemble_vector(L_direct_compiled)
b_direct.assemble()
print(f"\nDirect UFL assembly (no projection, no scaling):")
print(f"  ||b_direct||_2 = {b_direct.norm():.6e}")
print(f"  ||b_direct||_inf = {b_direct.norm(PETSc.NormType.NORM_INFINITY):.6e}")

b_d = b_direct.getArray()
print(f"  b_direct real min/max: {np.min(np.real(b_d)):.3e} / {np.max(np.real(b_d)):.3e}")
print(f"  b_direct imag min/max: {np.min(np.imag(b_d)):.3e} / {np.max(np.imag(b_d)):.3e}")
print(f"  # nonzero entries: {np.sum(np.abs(b_d) > 1e-15)}")

print("\n=== SUMMARY ===")
print(f"Projection+scaling RHS norm: {b.norm():.6e}")
print(f"Direct UFL RHS norm: {b_direct.norm():.6e}")
if b.norm() < 1e-12 and b_direct.norm() < 1e-12:
    print("BOTH RHS norms are effectively zero — problem is upstream in forcing!")
elif b.norm() < 1e-12:
    print("Projected RHS is zero but direct UFL is not — projection problem!")
elif b_direct.norm() < 1e-12:
    print("Direct UFL is zero but projected is not — UFL integration problem!")
else:
    print("Both have non-zero norms — issue is elsewhere (solver/BCs)")
