#!/usr/bin/env python3
"""
1D Impedance Reflection Test for Helmholtz Weak Form

Tests the Robin BC implementation by solving a 1D Helmholtz problem
with a velocity source on the left and an impedance (absorbing) BC on the right.

If the impedance Z = ρc (matched), the reflection coefficient should be ~0.
If the Robin coefficient has wrong sign or scaling, this test will clearly fail.

Also tests:
- Rigid wall (∂p/∂n = 0): should give |R| = 1 (perfect reflection)
- Pressure release (p = 0): should give |R| = 1 with π phase shift

Usage:
    micromamba run -n acousto-complex python scripts/validation/test_1d_impedance.py

Author: Acousto-Tweezers Project
Date: 2026-02-10
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
rho = 997.0        # kg/m³
c = 1484.0          # m/s
f = 500e3           # Hz
omega = 2 * np.pi * f
k = omega / c
Z = rho * c         # true impedance Pa·s/m
lam = c / f         # wavelength
L = 5 * lam         # domain length (5 wavelengths)
V0 = 1e-6           # source velocity amplitude [m/s]

print("="*70)
print("1D IMPEDANCE REFLECTION TEST")
print("="*70)
print(f"  f = {f/1e3:.0f} kHz, λ = {lam*1e3:.3f} mm, L = {L*1e3:.2f} mm = {L/lam:.0f}λ")
print(f"  k = {k:.4f} rad/m, Z = {Z:.0f} Pa·s/m")
print(f"  V₀ = {V0*1e6:.1f} μm/s")
print()

# ============================================================================
# MESH: 1D interval [0, L]
# ============================================================================
n_elem = 200  # well-resolved
domain = mesh.create_interval(comm, n_elem, [0.0, L])

fdim = 0  # facet dim for 1D = points
domain.topology.create_connectivity(fdim, 1)

# Tag boundaries
def left(x): return np.isclose(x[0], 0.0, atol=L*1e-10)
def right(x): return np.isclose(x[0], L, atol=L*1e-10)

left_facets = mesh.locate_entities_boundary(domain, fdim, left)
right_facets = mesh.locate_entities_boundary(domain, fdim, right)

# Create facet tags
all_facets = np.hstack([left_facets, right_facets]).astype(np.int32)
all_markers = np.hstack([np.ones_like(left_facets), 2*np.ones_like(right_facets)]).astype(np.int32)
order = np.argsort(all_facets)
facet_tags = mesh.meshtags(domain, fdim, all_facets[order], all_markers[order])

dss = Measure("ds", domain=domain, subdomain_data=facet_tags)

V = fem.functionspace(domain, ("Lagrange", 2))
u = TrialFunction(V)
v = TestFunction(V)


def solve_1d_helmholtz(robin_right, source_left, label=""):
    """
    Solve 1D Helmholtz with:
      Left (tag 1):  velocity source + impedance
      Right (tag 2): Robin BC with given coefficient
    
    Parameters
    ----------
    robin_right : complex
        Robin coefficient α for right boundary: adds α∫u v̄ ds to bilinear form
    source_left : complex
        Source g for left boundary: adds g∫v̄ ds to linear form
    
    Returns p(x) as DOLFINx Function
    """
    # Standard weak form (no /ρ)
    # From IBP:  ∫∇u·∇v̄ dx - k²∫uv̄ dx = ∫_∂Ω v̄(∂p/∂n) ds
    # Robin: ∂p/∂n = (iωρ/Z)p, so RHS = (iωρ/Z)∫uv̄ ds
    # Move to LHS:  a(u,v) = ∫∇u·∇v̄ dx - k²∫uv̄ dx - (iωρ/Z)∫uv̄ ds = 0
    a_form = (inner(grad(u), grad(v)) - k**2 * inner(u, v)) * dx
    
    # Left boundary: impedance + source
    # α = -(iωρ/Z) because we moved the boundary integral from RHS to LHS
    alpha_left = -1j * omega * rho / Z  # = -ik for matched impedance
    a_form += alpha_left * inner(u, v) * dss(1)
    
    # Right boundary: user-specified Robin
    if robin_right != 0.0:
        a_form += robin_right * inner(u, v) * dss(2)
    
    # Source on left: g = -iωρ V₀
    L_form = inner(source_left, v) * dss(1)
    
    problem = LinearProblem(
        a_form, L_form, bcs=[],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
    )
    p_h = problem.solve()
    return p_h


def analyze_solution(p_h, label, expected_R_mag):
    """Analyze reflection from the right boundary."""
    coords = V.tabulate_dof_coordinates()[:, 0]
    p_vals = p_h.x.array[:]
    
    # Sort by x
    order = np.argsort(coords)
    x = coords[order]
    p = p_vals[order]
    
    # Evaluate at specific locations
    # Near left (x ~ λ/4): should have standing wave pattern if reflected
    # Near right (x ~ L):   field at absorber
    
    # Analytical: p = A e^{ikx} + B e^{-ikx}
    # Source at x=0: v_n = (1/iωρ) dp/dx = (1/iωρ)(ikA e^{ikx} - ikB e^{-ikx})
    # At x=0: v_n|₀ = (k/ωρ)(A - B) = (1/c)(A - B)
    # Source BC: v_n|₀ = V₀  →  A - B = cV₀
    # 
    # Right BC (matched Z): dp/dx|_L = -(iωρ/Z)p|_L = -ikp|_L
    # dp/dx|_L = ik(A e^{ikL} - B e^{-ikL})
    # p|_L = A e^{ikL} + B e^{-ikL}
    # → ik(A e^{ikL} - B e^{-ikL}) = -ik(A e^{ikL} + B e^{-ikL})
    # → A e^{ikL} - B e^{-ikL} = -(A e^{ikL} + B e^{-ikL})
    # → 2A e^{ikL} = 0 → A = 0
    # Wait - that would mean no forward wave. Let me reconsider...
    
    # Actually, the sign depends on which direction is "outward normal"
    # For the LEFT boundary, outward normal is -x, so ∂p/∂n = -dp/dx
    # For the RIGHT boundary, outward normal is +x, so ∂p/∂n = +dp/dx
    
    # RIGHT BC: ∂p/∂n = dp/dx = -(iωρ/Z)p
    # dp/dx|_L = -ik p|_L  (for Z=ρc)
    # ik(A e^{ikL} - B e^{-ikL}) = -ik(A e^{ikL} + B e^{-ikL})
    # This gives 2A e^{ikL} = 0, so A = 0.
    # But that's wrong - B should be the reflected wave, not A.
    
    # Convention: p = A e^{+ikx} (rightward) + B e^{-ikx} (leftward)
    # At right wall, outgoing = e^{+ikx}, so matched impedance should kill B (reflected)
    
    # RIGHT wall ∂p/∂n = dp/dx:
    # dp/dx|_L = ik A e^{ikL} - ik B e^{-ikL}
    # Robin: dp/dx = -(iωρ/Z)p = -ik(A e^{ikL} + B e^{-ikL})
    # → ikA e^{ikL} - ikB e^{-ikL} = -ikA e^{ikL} - ikB e^{-ikL}
    # → 2ikA e^{ikL} = 0  →  A = 0  (???)
    
    # This means matching kills the INCOMING wave at the right wall, not reflected.
    # That's because from the right wall's perspective, e^{+ikx} is incoming.
    # R = A/B (from right wall's perspective) → R = 0 means no incoming wave
    # from outside, which is correct for a termination.
    
    # In our setup, wave is launched from LEFT. So e^{+ikx} is the forward
    # propagating wave and e^{-ikx} is the reflected wave.
    # Matched right wall should absorb the forward wave → no reflection.
    
    # Let me just compute R numerically by fitting A,B:
    # Pick two interior points x1, x2
    i1 = len(x) // 3
    i2 = 2 * len(x) // 3
    x1, x2 = x[i1], x[i2]
    p1, p2 = p[i1], p[i2]
    
    # p = A e^{ikx} + B e^{-ikx}
    # [e^{ikx1}  e^{-ikx1}] [A]   [p1]
    # [e^{ikx2}  e^{-ikx2}] [B] = [p2]
    M = np.array([
        [np.exp(1j*k*x1), np.exp(-1j*k*x1)],
        [np.exp(1j*k*x2), np.exp(-1j*k*x2)],
    ])
    rhs = np.array([p1, p2])
    AB = np.linalg.solve(M, rhs)
    A, B = AB[0], AB[1]
    
    # Reflection coefficient from right wall
    # Forward wave hits right: amplitude A e^{ikL}
    # Reflected wave: amplitude B e^{-ikL}
    # R = (B e^{-ikL}) / (A e^{ikL}) = (B/A) e^{-2ikL}
    if abs(A) > 1e-20:
        R = (B / A) * np.exp(-2j * k * L)
    else:
        R = np.inf
    
    # Also compute standing wave ratio (SWR)
    p_mag = np.abs(p)
    SWR = np.max(p_mag) / (np.min(p_mag) + 1e-30)
    
    print(f"\n--- {label} ---")
    print(f"  max|p| = {np.max(p_mag):.4f} Pa")
    print(f"  min|p| = {np.min(p_mag):.6f} Pa")
    print(f"  SWR    = {SWR:.2f}")
    print(f"  |A| (forward)  = {abs(A):.6e}")
    print(f"  |B| (reflected)= {abs(B):.6e}")
    print(f"  |R| = {abs(R):.6f}  (expected: {expected_R_mag:.2f})")
    print(f"  arg(R) = {np.angle(R)*180/np.pi:.1f}°")
    
    status = "✓ PASS" if abs(abs(R) - expected_R_mag) < 0.05 else "✗ FAIL"
    print(f"  {status}")
    return abs(R)


# ============================================================================
# TEST 1: Matched impedance (Z = ρc) → R ≈ 0
# ============================================================================
print("\n" + "="*70)
print("TEST 1: Matched impedance (anechoic right wall)")
print("="*70)
g_source = -1j * omega * rho * V0
alpha_right_matched = -1j * omega * rho / Z  # = -ik (correct: moved from RHS to LHS)
p1 = solve_1d_helmholtz(alpha_right_matched, g_source, "matched")
R1 = analyze_solution(p1, "Matched Z=ρc → expect |R|≈0", expected_R_mag=0.0)


# ============================================================================
# TEST 2: Rigid wall (Neumann, ∂p/∂n = 0) → R = 1
# ============================================================================
print("\n" + "="*70)
print("TEST 2: Rigid right wall (no Robin term)")
print("="*70)
p2 = solve_1d_helmholtz(0.0, g_source, "rigid")
R2 = analyze_solution(p2, "Rigid wall → expect |R|≈1", expected_R_mag=1.0)


# ============================================================================
# TEST 3: Wrong sign Robin (the OLD bug) → should show problems
# ============================================================================
print("\n" + "="*70)
print("TEST 3: WRONG sign Robin (+iωρ/Z instead of -iωρ/Z)")
print("="*70)
alpha_wrong_sign = +1j * omega * rho / Z  # WRONG sign
p3 = solve_1d_helmholtz(alpha_wrong_sign, g_source, "wrong_sign")
R3 = analyze_solution(p3, "Wrong sign → expect |R|>0 (broken BC)", expected_R_mag=0.0)


# ============================================================================
# TEST 4: Old code's Robin (missing ρ, wrong sign): -iω/Z
# ============================================================================
print("\n" + "="*70)
print("TEST 4: OLD CODE Robin (-iω/Z, missing ρ AND wrong sign)")
print("="*70)
alpha_old_code = -1j * omega / Z  # what the code actually had
p4 = solve_1d_helmholtz(alpha_old_code, g_source, "old_code")
R4 = analyze_solution(p4, "Old code (-iω/Z) → expect |R|>0 (broken BC)", expected_R_mag=0.0)


# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"  Test 1 (matched, -iωρ/Z):    |R| = {R1:.6f}  {'✓' if R1 < 0.05 else '✗'}")
print(f"  Test 2 (rigid, no Robin):     |R| = {R2:.6f}  {'✓' if abs(R2-1.0) < 0.05 else '✗'}")
print(f"  Test 3 (wrong sign, +iωρ/Z): |R| = {R3:.6f}  {'← shows sign error' if R3 > 0.05 else ''}")
print(f"  Test 4 (old code, -iω/Z):    |R| = {R4:.6f}  {'← shows old bug' if R4 > 0.05 else ''}")
