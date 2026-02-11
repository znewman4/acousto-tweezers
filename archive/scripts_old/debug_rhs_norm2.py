"""Debug: check assembled RHS vector norm — SMALL mesh for deposition."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import numpy as np

from acoustweezers.experiments.shallow_square_dish.config import ShallowDishConfig
from acoustweezers.experiments.shallow_square_dish.solve_pressure import solve_helmholtz

# Use same config as deposition experiment
cfg = ShallowDishConfig()
cfg.L = 0.01       # 10 mm
cfg.H = 0.001      # 1 mm  
cfg.nx = 20
cfg.ny = 20
cfg.nz = 8

# Import mesh creation
from acoustweezers.experiments.shallow_square_dish.solve_pressure import create_mesh

domain, facet_tags, tag_map = create_mesh(cfg, verbose=True)

# Solve pressure
p_sol = solve_helmholtz(domain, facet_tags, cfg, mode="vortex", verbose=True)

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
print(f"\n{'='*60}")
print(f"Forcing stats BEFORE scaling:")
print(f"  max |f| = {max_f:.3e}")
print(f"  f_func dtype: {f_func.x.array.dtype}")

numerical_scale = 1.0 / max_f if max_f > 0 else 1.0
f_func.x.array[:] *= numerical_scale
print(f"\nAfter scaling (scale = {numerical_scale:.2e}):")
print(f"  f_func real min/max: {np.min(np.real(f_func.x.array)):.3e} / {np.max(np.real(f_func.x.array)):.3e}")

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

a_compiled = fem.form(a_form_ufl)
L_compiled = fem.form(L_form_ufl)

b = fem.petsc.assemble_vector(L_compiled)
b.assemble()

A = fem.petsc.assemble_matrix(a_compiled)
A.assemble()

print(f"\n{'='*60}")
print(f"Assembled system (no BCs):")
print(f"  ||b||_2 = {b.norm():.6e}")
print(f"  ||b||_inf = {b.norm(PETSc.NormType.NORM_INFINITY):.6e}")
print(f"  ||A||_inf = {A.norm(PETSc.NormType.NORM_INFINITY):.6e}")
print(f"  ||b||/||A|| = {b.norm()/A.norm(PETSc.NormType.NORM_INFINITY):.6e}")
print(f"  DOFs = {W.dofmap.index_map.size_global * W.dofmap.index_map_bs}")

# Examine cell volumes
cells = domain.topology.index_map(3)
n_cells = cells.size_local
coords = domain.geometry.x
cell_to_vertex = domain.topology.connectivity(3, 0)

# Sample cell volume (tet)
from dolfinx.cpp.mesh import cell_num_vertices
# Compute approximate cell volume
from dolfinx.mesh import compute_midpoints
print(f"\n  Mesh info:")
print(f"    # cells: {n_cells}")
print(f"    coordinates range x: [{coords[:,0].min():.4e}, {coords[:,0].max():.4e}]")
print(f"    coordinates range y: [{coords[:,1].min():.4e}, {coords[:,1].max():.4e}]")
print(f"    coordinates range z: [{coords[:,2].min():.4e}, {coords[:,2].max():.4e}]")

# Estimate typical cell size
h_x = (coords[:,0].max() - coords[:,0].min()) / 20  # nx=20
h_y = (coords[:,1].max() - coords[:,1].min()) / 20  # ny=20
h_z = (coords[:,2].max() - coords[:,2].min()) / 8   # nz=8
cell_vol_est = h_x * h_y * h_z / 6  # tet = hex/6
print(f"    estimated cell volume: {cell_vol_est:.3e} m³")
print(f"    h_x = {h_x:.3e}, h_y = {h_y:.3e}, h_z = {h_z:.3e}")

# Now the key question: what is ||b|| composed of?
b_arr = b.getArray()
print(f"\n  RHS vector:")
print(f"    dtype: {b_arr.dtype}")
print(f"    real min/max: {np.min(np.real(b_arr)):.6e} / {np.max(np.real(b_arr)):.6e}")
if np.any(np.imag(b_arr) != 0):
    print(f"    imag min/max: {np.min(np.imag(b_arr)):.6e} / {np.max(np.imag(b_arr)):.6e}")
else:
    print(f"    imaginary: all zero")
    
# Count how many entries are in what range
b_abs = np.abs(np.real(b_arr))
for threshold in [1e-5, 1e-8, 1e-10, 1e-12, 1e-14, 1e-16]:
    count = np.sum(b_abs > threshold)
    print(f"    |b_i| > {threshold:.0e}: {count}/{len(b_arr)}")

# Expected solution magnitude:
# u ~ f * H^2 / mu (Stokes scaling)
H = cfg.H
f_phys = max_f
u_expected = f_phys * H**2 / mu
print(f"\n  Physical estimates:")
print(f"    f_physical = {f_phys:.3e} Pa/m")
print(f"    u_expected = f*H²/μ = {u_expected:.3e} m/s = {u_expected*1e6:.3f} μm/s")
print(f"    u_expected * numerical_scale = {u_expected * numerical_scale:.3e}")

# The issue: in complex mode, the matrix and vector are complex.
# Even if we only put real values, the solver treats everything as complex.
# Let's check the matrix diagonal
d = A.getDiagonal()
d_arr = d.getArray()
print(f"\n  Matrix diagonal:")
print(f"    dtype: {d_arr.dtype}")
print(f"    real min/max: {np.min(np.real(d_arr)):.6e} / {np.max(np.real(d_arr)):.6e}")
print(f"    imag min/max: {np.min(np.imag(d_arr)):.6e} / {np.max(np.imag(d_arr)):.6e}")
print(f"    # zeros on diag: {np.sum(np.abs(d_arr) < 1e-20)}")

# NOW try solving with proper absolute tolerance
print(f"\n{'='*60}")
print("Solving with explicit atol = 1e-20, rtol=1e-8...")
ksp = PETSc.KSP().create(domain.comm)
ksp.setOperators(A)
ksp.setType("gmres")
pc = ksp.getPC()
pc.setType("ilu")  # Simple preconditioner for testing
ksp.setTolerances(rtol=1e-8, atol=1e-20, max_it=2000)
ksp.setFromOptions()

wh = fem.Function(W)
ksp.solve(b, wh.x.petsc_vec)
wh.x.scatter_forward()

reason = ksp.getConvergedReason()
its = ksp.getIterationNumber()
print(f"  KSP: {its} iterations, reason={reason}")

u_h = wh.sub(0).collapse()
u_arr = np.real(u_h.x.array)
n_u = len(u_arr) // 3
u_mag = np.linalg.norm(u_arr.reshape((n_u, 3)), axis=1)
max_u_scaled = float(np.max(u_mag))
max_u = max_u_scaled / numerical_scale

print(f"  Scaled velocity max: {max_u_scaled:.6e}")
print(f"  Physical velocity max: {max_u:.6e} m/s = {max_u*1e6:.4f} μm/s")

A.destroy()
b.destroy()
ksp.destroy()
