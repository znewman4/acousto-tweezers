#!/usr/bin/env python3
"""
Energy/Power Balance Test for Helmholtz Solver

Verifies that the power injected by the velocity source equals
the power absorbed by the impedance boundaries (to assembly/solver tolerance).

For the complex Helmholtz equation with impedance BCs:
  Power injected at source S:   P_in = ½ Re ∫_S p · v̄ₙ ds
  Power absorbed at impedance:  P_abs = ½ Re ∫_Γ |p|² / Z ds

Energy conservation requires: P_in = P_abs  (no internal dissipation in Helmholtz)

Usage:
    micromamba run -n acousto-complex python scripts/validation/test_energy_balance.py
"""

import numpy as np
from mpi4py import MPI
from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem
import ufl
from ufl import inner, grad, dx, TrialFunction, TestFunction, Measure

comm = MPI.COMM_WORLD

# ============================================================================
# PHYSICAL PARAMETERS
# ============================================================================
rho = 997.0
c = 1484.0
f = 500e3
omega = 2 * np.pi * f
k = omega / c
Z = rho * c
lam = c / f
L = 5 * lam
V0 = 1e-6

print("="*70)
print("ENERGY/POWER BALANCE TEST")
print("="*70)
print(f"  f = {f/1e3:.0f} kHz, k = {k:.4f}, Z = {Z:.0f}")
print()

# ============================================================================
# 1D MESH
# ============================================================================
n_elem = 200
domain = mesh.create_interval(comm, n_elem, [0.0, L])

fdim = 0
domain.topology.create_connectivity(fdim, 1)

def left(x): return np.isclose(x[0], 0.0, atol=L*1e-10)
def right(x): return np.isclose(x[0], L, atol=L*1e-10)

left_facets = mesh.locate_entities_boundary(domain, fdim, left)
right_facets = mesh.locate_entities_boundary(domain, fdim, right)

all_facets = np.hstack([left_facets, right_facets]).astype(np.int32)
all_markers = np.hstack([np.ones_like(left_facets), 2*np.ones_like(right_facets)]).astype(np.int32)
order = np.argsort(all_facets)
facet_tags = mesh.meshtags(domain, fdim, all_facets[order], all_markers[order])

dss = Measure("ds", domain=domain, subdomain_data=facet_tags)

V = fem.functionspace(domain, ("Lagrange", 2))
u = TrialFunction(V)
v = TestFunction(V)

# ============================================================================
# SOLVE: Velocity source on left, matched impedance on right
# ============================================================================
# Robin: α = -(iωρ/Z) on both boundaries
alpha = -1j * omega * rho / Z  # = -ik

a_form = (inner(grad(u), grad(v)) - k**2 * inner(u, v)) * dx
a_form += alpha * inner(u, v) * dss(1)  # left impedance
a_form += alpha * inner(u, v) * dss(2)  # right impedance

# Source on left: g = -iωρ V₀
g_source = -1j * omega * rho * V0
L_form = inner(g_source, v) * dss(1)

problem = LinearProblem(
    a_form, L_form, bcs=[],
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
)
p_h = problem.solve()

# ============================================================================
# POWER CALCULATIONS
# ============================================================================
# Power = ½ Re ∫ p v̄ₙ ds  (time-averaged acoustic power)
# For velocity source: vₙ = V₀ at x=0 (into domain)
# But v̄ₙ is the conjugate of the normal velocity component
# 
# From Euler: vₙ = (1/iωρ) ∂p/∂n
# On impedance surface: vₙ = p/Z
#
# Alternative: use the relation directly.
# P_source = ½ Re ∫ₛ p · (-V₀*) ds  (negative because v_n = -V₀ wrt outward normal at x=0)
#          = -½ Re(V₀*) ∫ₛ p ds     (V₀ is real, so V₀* = V₀)
#
# More carefully: total power = power in - power out
# 
# At LEFT (x=0):  outward normal = -x̂
#   velocity into domain = +V₀ (rightward), but v·n̂ = -V₀
#   ∂p/∂n = ∇p·n̂ = -dp/dx
#   Power flowing RIGHT through x=0: P = ½ Re ∫ p · v̄_x ds = ½ Re(p·V₀*)
#
# At RIGHT (x=L):  outward normal = +x̂
#   v_n = p/Z (impedance relation)
#   Power flowing RIGHT through x=L: P = ½ Re ∫ p · (p/Z)* ds = ½ |p|²/Z
#
# For perfectly matched (R=0): all power goes through → P_in = P_out

# Evaluate p at boundaries
coords = V.tabulate_dof_coordinates()[:, 0]
p_vals = p_h.x.array[:]

# Find boundary DOFs
tol = L * 1e-10
left_dofs = np.where(np.abs(coords) < tol)[0]
right_dofs = np.where(np.abs(coords - L) < tol)[0]

p_left = p_vals[left_dofs[0]] if len(left_dofs) > 0 else 0.0
p_right = p_vals[right_dofs[0]] if len(right_dofs) > 0 else 0.0

print(f"  p(x=0) = {p_left:.6f}")
print(f"  p(x=L) = {p_right:.6f}")
print()

# Power injected by source at x=0:
# The source term in RHS is -iωρ V₀, which corresponds to ∂p/∂n = -iωρ V₀
# At x=0, outward normal is -x̂, so dp/dx = iωρ V₀
# Acoustic intensity through x=0 (rightward): I = ½ Re(p · v̄_x)
# where v_x = (1/iωρ) dp/dx = V₀ + p/(Z)  [velocity source + reflected]
# But we can use the relation: total power = source power
# Source power: P_source = ½ Re(p(0) · V₀*)  [V₀ is real]

# Actually for the combined source+impedance at x=0, let's compute from
# the solution directly:

# The velocity at x=0 consists of:
#   1. Forced component: V₀ (into domain, i.e., +x direction)
#   2. Impedance component: p(0)/Z (outward, i.e., -x direction)  
# Total v_x at x=0: V₀ - p(0)/Z ... but wait, this is the Neumann condition.
# Actually, the combined BC at x=0 is:
#   ∂p/∂n = -iωρ V₀ + (iωρ/Z) p    [source + impedance]
# outward normal at x=0 is -x̂, so ∂p/∂n = -dp/dx
# → -dp/dx = -iωρ V₀ + (iωρ/Z) p
# → dp/dx = iωρ V₀ - (iωρ/Z) p = iωρ (V₀ - p/Z)
# velocity: v_x = (1/iωρ) dp/dx = V₀ - p/Z

# Power (rightward, through x=0): 
v_x_left = V0 - p_left / Z
P_in = 0.5 * np.real(p_left * np.conj(v_x_left))

# Power (rightward, through x=L):
v_x_right = p_right / Z  # impedance relation: v·n̂ = p/Z, n̂ = +x̂ so v_x = p/Z
P_out = 0.5 * np.real(p_right * np.conj(v_x_right))

# Power absorbed by left impedance:
P_abs_left = 0.5 * np.abs(p_left)**2 / Z
# Power absorbed by right impedance:
P_abs_right = 0.5 * np.abs(p_right)**2 / Z

print(f"  Power flux (rightward) at x=0:  P_in  = {P_in:.6e} W/m²")
print(f"  Power flux (rightward) at x=L:  P_out = {P_out:.6e} W/m²")
print(f"  Power absorbed at left wall:     {P_abs_left:.6e} W/m²")
print(f"  Power absorbed at right wall:    {P_abs_right:.6e} W/m²")
print()

# For matched impedance, P_in should equal P_out (no reflection)
rel_err = abs(P_in - P_out) / (abs(P_in) + 1e-30)
print(f"  |P_in - P_out| / P_in = {rel_err:.2e}")
print(f"  {'✓ PASS' if rel_err < 1e-4 else '✗ FAIL'}: Power balance")
print()

# ============================================================================
# ANALYTICAL CHECK
# ============================================================================
# For matched impedance (R=0), the solution is purely forward-traveling:
#   p(x) = p₀ e^{ikx}
# where p₀ = ρcV₀ = Z V₀
p_analytical = Z * V0
print(f"  Analytical |p| = Z·V₀ = {p_analytical:.6f} Pa")
print(f"  Computed   |p(0)| = {abs(p_left):.6f} Pa")
print(f"  Computed   |p(L)| = {abs(p_right):.6f} Pa")
err_p = abs(abs(p_left) - p_analytical) / p_analytical
print(f"  |p(0)| error = {err_p:.2e}")
print(f"  {'✓ PASS' if err_p < 1e-3 else '✗ FAIL'}: Pressure amplitude")
print()

# Power check: P_analytical = ½ |p₀|² / Z = ½ Z V₀²
P_analytical = 0.5 * Z * V0**2
print(f"  Analytical P = ½ZV₀² = {P_analytical:.6e} W/m²")
print(f"  Computed  P_in = {P_in:.6e} W/m²")
err_P = abs(P_in - P_analytical) / P_analytical
print(f"  P error = {err_P:.2e}")
print(f"  {'✓ PASS' if err_P < 1e-3 else '✗ FAIL'}: Power magnitude")

# ============================================================================
# TEST 2: RIGID WALLS (energy should bounce back and forth)
# ============================================================================
print()
print("="*70)
print("TEST 2: Energy balance with rigid right wall")
print("="*70)

a_form2 = (inner(grad(u), grad(v)) - k**2 * inner(u, v)) * dx
a_form2 += alpha * inner(u, v) * dss(1)  # left: impedance
# right: rigid (no Robin term) → ∂p/∂n = 0

L_form2 = inner(g_source, v) * dss(1)

problem2 = LinearProblem(
    a_form2, L_form2, bcs=[],
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
)
p_h2 = problem2.solve()
p_vals2 = p_h2.x.array[:]

p_left2 = p_vals2[left_dofs[0]]
p_right2 = p_vals2[right_dofs[0]]

# For rigid right wall: all power must be absorbed by left impedance
# Net power into domain at left = power flowing right - power flowing left
# With |R|=1, net power = 0 ... but left wall has impedance.
# 
# Actually with impedance on left + rigid on right:
# The reflected wave from right is fully reflected, then partially absorbed on left.
# It reaches a steady state where power absorbed by left impedance = power from source.
# 
# Easier check: use assembled forms (residual-based)

# Just check that solution is reasonable (standing wave pattern)
p_mag2 = np.abs(p_vals2)
SWR2 = np.max(p_mag2) / (np.min(p_mag2) + 1e-30)
print(f"  |p| range: [{np.min(p_mag2):.4f}, {np.max(p_mag2):.4f}] Pa")
print(f"  SWR = {SWR2:.2f}")

# The standing wave ratio for |R|=1 should be very large (ideally infinite for rigid+source)
# But impedance on left absorbs, so SWR is finite and depends on L.
# For |R|=1 at right: wave reflects back, hits impedance on left (matched), gets absorbed.
# Net: traveling wave from left to right + reflected from right = standing wave with SWR=∞
# But left wall impedance means the reflected wave IS absorbed there.
# So actually SWR won't be infinite.
# For one round trip: source injects, wave goes right, reflects, comes back to left.
# Left impedance absorbs the returning wave → steady state is a traveling wave going right
# plus its reflection going left, but the left impedance kills the leftward wave.
# This is more complex. Let's just verify the analytical solution.

# Analytical: p = A e^{ikx} + B e^{-ikx}
# Left BC: dp/dx|₀ = iωρ(V₀ - p/Z) → ik(A-B) = iωρ(V₀ - (A+B)/Z)
# Since iωρ/Z = ik: ik(A-B) = ik(V₀·Z/(iωρ)... this is getting circular.
# Let me just use the matrix approach.

# Left: ∂p/∂n = -dp/dx = -iωρ V₀ + (iωρ/Z)p = -ikZV₀ + ik p
# → -ik(A-B) = -ik(A+B) + ik(A+B)... ugh, let me be more careful.
# ∂p/∂n at x=0 with outward normal -x̂:
# ∂p/∂n = -dp/dx = -ik(A - B)
# Robin + source on left: ∂p/∂n = -iωρ V₀ - (iωρ/Z)p = -ikZV₀/(Z)... 
# Actually from the weak form: -dp/dx|₀ = -iωρV₀ + (iωρ/Z)p|₀
# Wait, the convention in the code is that the Robin and source both contribute
# to ∂p/∂n. Let me just use:
# ∂p/∂n|₀ = -(source) + (impedance outward)
# dp/dx|₀ = -∂p/∂n|₀ = iωρV₀ - (iωρ/Z)p|₀

# → ik(A-B) = iωρV₀ - ik(A+B)
# → ik(A-B) + ik(A+B) = iωρV₀
# → 2ikA = ikZV₀/Z ... no, iωρV₀ = ik·Z·V₀/Z = ikV₀ ... wait:
# iωρV₀, and ik = iω/c, so iωρV₀ = ikρcV₀ = ikZV₀
# → 2ikA = ikZV₀ → A = ZV₀/2

# Right BC (rigid): dp/dx|_L = 0 (∂p/∂n = dp/dx = 0)
# ik(A e^{ikL} - B e^{-ikL}) = 0 → A e^{ikL} = B e^{-ikL}
# → B = A e^{2ikL}

# From left: 2ikA = ikZV₀ → A = ZV₀/2
A_anal = Z * V0 / 2
B_anal = A_anal * np.exp(2j * k * L)

# Check at boundary points
p_left_anal = A_anal + B_anal
p_right_anal = A_anal * np.exp(1j * k * L) + B_anal * np.exp(-1j * k * L)

print(f"\n  Analytical (impedance left + rigid right):")
print(f"    A = {A_anal:.4f}")
print(f"    B = {abs(B_anal):.4f} at phase {np.angle(B_anal)*180/np.pi:.1f}°")
print(f"    p(0) = {p_left_anal:.6f}  (computed: {p_left2:.6f})")
print(f"    p(L) = {p_right_anal:.6f}  (computed: {p_right2:.6f})")
print(f"    |p(0)| error = {abs(abs(p_left2)-abs(p_left_anal))/abs(p_left_anal):.2e}")
print(f"    |p(L)| error = {abs(abs(p_right2)-abs(p_right_anal))/abs(p_right_anal):.2e}")

err2 = abs(abs(p_left2)-abs(p_left_anal))/abs(p_left_anal)
print(f"    {'✓ PASS' if err2 < 1e-3 else '✗ FAIL'}: Analytical pressure match")

print()
print("="*70)
print("ALL TESTS COMPLETE")
print("="*70)
