"""Debug: check forcing function spatial distribution."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import numpy as np

from acoustweezers.experiments.shallow_square_dish.config import ShallowDishConfig
from acoustweezers.experiments.shallow_square_dish.solve_pressure import create_mesh, solve_helmholtz

import dolfinx.fem as fem
import ufl

cfg = ShallowDishConfig()
cfg.L = 0.01
cfg.H = 0.001
cfg.nx = 20
cfg.ny = 20
cfg.nz = 8

domain, facet_tags, tag_map = create_mesh(cfg, verbose=False)
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

V_forcing = fem.functionspace(domain, ("Lagrange", 1, (3,)))
f_expr = fem.Expression(f_ufl, V_forcing.element.interpolation_points())
f_func = fem.Function(V_forcing)
f_func.interpolate(f_expr)
f_func.x.array[:] = np.real(f_func.x.array)

# Get forcing at each DOF and its coordinates
coords = V_forcing.tabulate_dof_coordinates()
f_vals = np.real(f_func.x.array.copy())
n_dofs = len(coords)
f_3d = f_vals.reshape((n_dofs, 3))
f_mag = np.linalg.norm(f_3d, axis=1)

# Pressure distribution
p_coords = p_func.function_space.tabulate_dof_coordinates()
p_mag = np.abs(p_func.x.array)

print(f"\nPressure field:")
print(f"  max |p| = {np.max(p_mag):.4f} Pa")
print(f"  mean |p| = {np.mean(p_mag):.4f} Pa")
print(f"  median |p| = {np.median(p_mag):.4f} Pa")

# z-layer analysis
H = cfg.H
z = coords[:, 2]
for z_frac in [0.0, 0.125, 0.25, 0.5, 0.75, 1.0]:
    z_target = z_frac * H
    mask = np.isclose(z, z_target, atol=H*0.01)
    if np.any(mask):
        layer_f = f_mag[mask]
        print(f"\n  z = {z_target*1e3:.2f} mm (z/H = {z_frac:.3f}):")
        print(f"    # DOFs: {np.sum(mask)}")
        print(f"    max |f| = {np.max(layer_f):.4e}")
        print(f"    mean |f| = {np.mean(layer_f):.4e}")
        print(f"    % of total max: {np.max(layer_f)/np.max(f_mag)*100:.1f}%")

# Check which DOFs have significant forcing
print(f"\n\nForcing distribution:")
for thresh_frac in [0.5, 0.1, 0.01, 0.001, 1e-4, 1e-6]:
    thresh = thresh_frac * np.max(f_mag)
    n_above = np.sum(f_mag > thresh)
    # z-range of those DOFs
    if n_above > 0:
        z_above = z[f_mag > thresh]
        print(f"  |f| > {thresh_frac:.0e} * max ({thresh:.2e}): {n_above}/{n_dofs} DOFs, z=[{z_above.min()*1e3:.2f}, {z_above.max()*1e3:.2f}] mm")

# Now the key: how many forcing DOFs are on no-slip boundaries?
L_dim = cfg.L
on_bottom = np.isclose(z, 0.0, atol=H*1e-4)
on_sides = (np.isclose(coords[:,0], 0.0, atol=L_dim*1e-4) | 
            np.isclose(coords[:,0], L_dim, atol=L_dim*1e-4) |
            np.isclose(coords[:,1], 0.0, atol=L_dim*1e-4) |
            np.isclose(coords[:,1], L_dim, atol=L_dim*1e-4))
on_boundary = on_bottom | on_sides

interior = ~on_boundary

print(f"\n\nBoundary analysis:")
print(f"  Total forcing DOFs: {n_dofs}")
print(f"  On no-slip boundary: {np.sum(on_boundary)} ({100*np.sum(on_boundary)/n_dofs:.1f}%)")
print(f"  Interior: {np.sum(interior)} ({100*np.sum(interior)/n_dofs:.1f}%)")
print(f"  max |f| on boundary: {np.max(f_mag[on_boundary]):.4e}")
print(f"  max |f| in interior: {np.max(f_mag[interior]):.4e}")
print(f"  mean |f| on boundary: {np.mean(f_mag[on_boundary]):.4e}")
print(f"  mean |f| in interior: {np.mean(f_mag[interior]):.4e}")

# Key question: is the forcing localized to the bottom (source) boundary?
has_significant = f_mag > 0.01 * np.max(f_mag)
print(f"\n  DOFs with |f| > 1% of max:")
print(f"    Total: {np.sum(has_significant)}")
print(f"    On no-slip boundary: {np.sum(has_significant & on_boundary)}")
print(f"    Interior: {np.sum(has_significant & interior)}")
print(f"    On bottom: {np.sum(has_significant & on_bottom)}")
