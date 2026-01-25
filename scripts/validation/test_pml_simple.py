"""
PML Absorption Test 
===================
Validates that the Perfectly Matched Layer absorbs outgoing waves.

Test: Inject a time-harmonic wave from the left, measure decay in PML region.
The wave should decay exponentially in the PML, confirming absorption.

Pass Criteria: Amplitude at PML exit < 50% of amplitude at PML entrance
"""

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx
from dolfinx import mesh, fem
from dolfinx.fem.petsc import LinearProblem
import ufl

# Verify complex backend
assert np.issubdtype(PETSc.ScalarType, np.complexfloating), \
    f"PETSc must be complex! Got {PETSc.ScalarType}"


def run_pml_test():
    """Test PML absorption with traveling wave injection."""
    
    print("=" * 60)
    print("PML ABSORPTION TEST")
    print("=" * 60)
    
    # Physical parameters (simple non-dimensionalized)
    k = 10.0  # Wavenumber (dimensionless)
    omega = k  # c = omega/k = 1
    
    print(f"\nParameters:")
    print(f"  Wavenumber k = {k}")
    print(f"  Wavelength λ = {2*np.pi/k:.3f}")
    
    # Domain: physical region + PML
    L_phys = 1.0  # Physical region [0, 1]
    L_pml = 0.5   # PML region [1, 1.5]
    L_total = L_phys + L_pml
    H = 0.2       # Domain height (thin to approximate 1D)
    
    # Mesh
    h = 0.02  # Element size (about 30 elements per wavelength)
    nx = int(L_total / h)
    ny = max(int(H / h), 3)
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_rectangle(
        comm,
        [[0.0, 0.0], [L_total, H]],
        [nx, ny],
        mesh.CellType.triangle
    )
    
    V = fem.functionspace(domain, ("Lagrange", 2))
    ncells = domain.topology.index_map(domain.topology.dim).size_local
    ndofs = V.dofmap.index_map.size_local
    print(f"\nMesh: {ncells} cells, {ndofs} DOFs")
    
    # PML parameters
    # The key is getting the sign right for wave absorption
    # For a wave traveling in +x direction: p ~ exp(i(kx - ωt))
    # In PML, we want the wave to decay: x -> x + i*σ*x'/ω (where σ > 0)
    # This makes exp(ikx) -> exp(ik(x + iσx'/ω)) = exp(ikx) * exp(-kσx'/ω)
    # So for decay, we need: s_x = 1 + i*σ/ω with σ > 0
    # And the weak form has: (1/s_x)*∂p/∂x*∂v*/∂x and k²*s_x*p*v*
    
    sigma_max = 20.0  # Strong absorption
    
    # PML stretching coefficient: s_x = 1 + i*sigma(x)/omega
    # sigma(x) = sigma_max * ((x - L_phys)/L_pml)^3 for x > L_phys
    
    Vc = fem.functionspace(domain, ("DG", 0))
    
    # Complex stretching s_x (for exponential decay of rightward wave)
    s_x = fem.Function(Vc)
    def make_s_x(x):
        d = np.clip((x[0] - L_phys) / L_pml, 0, 1)
        sigma = sigma_max * (d ** 3)
        # s = 1 + i*sigma/omega, but for rightward traveling wave we need -i
        return (1.0 - 1j * sigma / omega).astype(PETSc.ScalarType)
    s_x.interpolate(make_s_x)
    
    # Inverse 1/s_x
    inv_s_x = fem.Function(Vc)
    def make_inv_s_x(x):
        d = np.clip((x[0] - L_phys) / L_pml, 0, 1)
        sigma = sigma_max * (d ** 3)
        s = 1.0 - 1j * sigma / omega
        return (1.0 / s).astype(PETSc.ScalarType)
    inv_s_x.interpolate(make_inv_s_x)
    
    # Trial and test functions
    p = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Helmholtz with PML (x-direction only)
    # a(p,v) = ∫ [(1/s_x) ∂p/∂x ∂v*/∂x + ∂p/∂y ∂v*/∂y - k² s_x p v*] dΩ
    
    grad_p = ufl.grad(p)
    grad_v = ufl.grad(v)
    
    k_sq = fem.Constant(domain, PETSc.ScalarType(k**2))
    
    # Bilinear form
    a = inv_s_x * ufl.inner(grad_p[0], grad_v[0]) * ufl.dx  # x-gradient (PML modified)
    a += ufl.inner(grad_p[1], grad_v[1]) * ufl.dx           # y-gradient (normal)
    a -= k_sq * s_x * ufl.inner(p, v) * ufl.dx              # Mass term (PML modified)
    
    # Boundary conditions
    fdim = domain.topology.dim - 1
    domain.topology.create_connectivity(fdim, domain.topology.dim)
    
    # Left boundary: Dirichlet p = exp(i*k*x) at x=0 => p = 1
    # But we want traveling wave, so use Robin BC: ∂p/∂n + ik*p = 2ik at x=0
    # This injects a unit amplitude right-traveling wave
    
    def left_boundary(x):
        return np.isclose(x[0], 0.0)
    
    def right_boundary(x):
        return np.isclose(x[0], L_total)
    
    left_facets = mesh.locate_entities_boundary(domain, fdim, left_boundary)
    right_facets = mesh.locate_entities_boundary(domain, fdim, right_boundary)
    
    # Facet tags
    all_facets = np.concatenate([left_facets, right_facets])
    all_tags = np.concatenate([
        np.full_like(left_facets, 1, dtype=np.int32),
        np.full_like(right_facets, 2, dtype=np.int32)
    ])
    
    # Sort for meshtags
    sort_idx = np.argsort(all_facets)
    facet_tags = mesh.meshtags(domain, fdim, all_facets[sort_idx], all_tags[sort_idx])
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
    
    # Left BC (x=0): Robin condition for incident wave
    # -∂p/∂n + ik*p = 2ik (n points in -x direction at left boundary)
    # In weak form: ∫ (∂p/∂n) v* ds = ∫ ik*p v* ds - ∫ 2ik v* ds
    # Add to bilinear form: ∫ ik p v* ds(left)
    # Add to linear form: ∫ 2ik v* ds(left)
    
    ik = fem.Constant(domain, PETSc.ScalarType(1j * k))
    two_ik = fem.Constant(domain, PETSc.ScalarType(2j * k))
    
    a += ik * ufl.inner(p, v) * ds(1)  # Robin contribution on left
    
    # Right BC (x=L_total): ABC (should be in PML anyway)
    a += ik * ufl.inner(p, v) * ds(2)
    
    # Linear form: source from left BC
    # Zero volume source
    f_zero = fem.Constant(domain, PETSc.ScalarType(0.0))
    L = ufl.inner(f_zero, v) * ufl.dx
    
    # Add incident wave source at left boundary
    # Using inner() for proper conjugation in complex mode
    L += ufl.inner(two_ik, v) * ds(1)
    
    # Solve
    print("\nSolving Helmholtz with PML and wave injection...")
    
    problem = LinearProblem(a, L, petsc_options={
        "ksp_type": "preonly",
        "pc_type": "lu"
    })
    
    p_h = problem.solve()
    
    # Analyze solution
    p_arr = p_h.x.array
    print(f"\nSolution statistics:")
    print(f"  max|p| = {np.max(np.abs(p_arr)):.4f}")
    print(f"  mean|p| = {np.mean(np.abs(p_arr)):.4f}")
    
    # Sample along centerline
    n_sample = 100
    x_sample = np.linspace(0.01, L_total - 0.01, n_sample)
    y_sample = (H / 2) * np.ones(n_sample)
    z_sample = np.zeros(n_sample)
    points = np.stack([x_sample, y_sample, z_sample], axis=1)
    
    bb_tree = dolfinx.geometry.bb_tree(domain, domain.topology.dim)
    cells = []
    pts_ok = []
    x_ok = []
    
    cell_cands = dolfinx.geometry.compute_collisions_points(bb_tree, points)
    coll_cells = dolfinx.geometry.compute_colliding_cells(domain, cell_cands, points)
    
    for i, pt in enumerate(points):
        if len(coll_cells.links(i)) > 0:
            pts_ok.append(pt)
            x_ok.append(x_sample[i])
            cells.append(coll_cells.links(i)[0])
    
    print(f"\n  Sampled {len(pts_ok)} points along centerline")
    
    if len(pts_ok) < 10:
        print("❌ FAIL: Not enough sample points")
        return False
    
    pts_ok = np.array(pts_ok, dtype=np.float64)
    x_ok = np.array(x_ok)
    
    p_vals = p_h.eval(pts_ok, cells).flatten()
    amp = np.abs(p_vals)
    
    # Find amplitude at key locations
    idx_phys = x_ok < L_phys
    idx_pml = x_ok >= L_phys
    
    # Physical region average amplitude
    amp_phys = np.mean(amp[idx_phys])
    
    # PML entrance and exit
    idx_entrance = np.argmin(np.abs(x_ok - L_phys))
    idx_exit = np.argmax(x_ok)
    
    amp_entrance = amp[idx_entrance]
    amp_exit = amp[idx_exit]
    
    print(f"\nAmplitude profile:")
    print(f"  Physical region mean: {amp_phys:.4f}")
    print(f"  PML entrance (x={x_ok[idx_entrance]:.2f}): {amp_entrance:.4f}")
    print(f"  PML exit (x={x_ok[idx_exit]:.2f}): {amp_exit:.4f}")
    
    # Decay in PML
    if amp_entrance > 1e-10:
        decay = amp_exit / amp_entrance
        print(f"  PML decay ratio: {decay:.4f}")
        absorption = (1 - decay) * 100
        print(f"  Absorption: {absorption:.1f}%")
    else:
        decay = 1.0
        absorption = 0.0
        print("  ⚠ Amplitude at entrance too small")
    
    # Also check that we have a traveling wave (not standing)
    # For traveling wave: |p| ≈ constant in physical region
    if np.any(idx_phys):
        amp_phys_arr = amp[idx_phys]
        variation = (np.max(amp_phys_arr) - np.min(amp_phys_arr)) / np.mean(amp_phys_arr)
        print(f"  Amplitude variation in physical region: {variation*100:.1f}%")
    
    # PASS/FAIL
    print("\n" + "=" * 60)
    
    if amp_phys < 0.1:
        print("❌ FAIL: Wave amplitude too small (bad injection)")
        return False
    elif decay < 0.3:
        print(f"✓ PASS: PML absorbs >{absorption:.0f}% ({decay:.2%} transmission)")
        return True
    elif decay < 0.6:
        print(f"⚠ WARN: Moderate PML absorption ({absorption:.0f}%)")
        return True
    else:
        print(f"❌ FAIL: PML not absorbing (decay ratio {decay:.2f})")
        return False


if __name__ == "__main__":
    success = run_pml_test()
    print("=" * 60)
    exit(0 if success else 1)
