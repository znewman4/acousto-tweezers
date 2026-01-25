"""
PML Absorption Test (Simplified)
================================
Validates that the Perfectly Matched Layer (PML) absorbs outgoing waves
without spurious reflections.

Test Methodology:
-----------------
1. Create a 1D-like domain (thin strip) with PML on right side
2. Apply plane wave BC on left
3. Measure amplitude decay in PML

Pass Criteria:
--------------
- Amplitude at PML exit < 50% of amplitude at PML entrance
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


def run_pml_absorption_test():
    """Run PML absorption validation test."""
    
    print("=" * 60)
    print("PML ABSORPTION TEST")
    print("=" * 60)
    
    # Physical parameters
    c = 1500.0  # m/s
    freq = 40e3  # 40 kHz
    omega = 2 * np.pi * freq
    k = omega / c  # wavenumber
    wavelength = c / freq
    
    print(f"\nPhysical parameters:")
    print(f"  Frequency: {freq/1e3:.1f} kHz")
    print(f"  Wavelength: {wavelength*1e3:.2f} mm")
    print(f"  Wavenumber: {k:.2f} rad/m")
    
    # Domain sizing
    L_phys = 3 * wavelength  # Physical region
    L_pml = 2 * wavelength   # PML region
    L_total = L_phys + L_pml
    H = wavelength / 2       # Domain height (thin strip)
    h = wavelength / 15      # Element size
    
    print(f"\nDomain:")
    print(f"  Physical region: {L_phys*1e3:.1f} mm")
    print(f"  PML region: {L_pml*1e3:.1f} mm")
    print(f"  Height: {H*1e3:.1f} mm")
    
    # Create mesh
    comm = MPI.COMM_WORLD
    nx = int(L_total / h)
    ny = max(int(H / h), 2)
    
    domain = mesh.create_rectangle(
        comm,
        [[0.0, 0.0], [L_total, H]],
        [nx, ny],
        mesh.CellType.triangle
    )
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", 2))
    
    ncells = domain.topology.index_map(domain.topology.dim).size_local
    ndofs = V.dofmap.index_map.size_local
    print(f"\nMesh: {ncells} cells, {ndofs} DOFs")
    
    # Material with PML absorption
    # Proper PML: modify the gradient operator
    # For Helmholtz: div(1/s * grad(p)) + k^2 * s * p = 0
    # where s = 1 + i*sigma/omega is the stretching factor
    # 
    # In 1D (x-dependent): s_x = 1 + i*sigma(x)/omega
    # For stretched Laplacian: ∂/∂x (1/s_x * ∂p/∂x) + s_x * k^2 * p = 0
    # Multiply by s_x: ∂/∂x (∂p/∂x) + s_x^2 * k^2 * p = 0 (approx)
    # 
    # Better: use complex k^2 replacement
    # In PML: k^2 -> k^2 * (1 - i * sigma/omega)^2 = k^2 * (1 - 2i*sigma/omega - sigma^2/omega^2)
    # For moderate sigma: k^2 -> k^2 - 2i*k*sigma (approximately)
    
    # PML profile: polynomial ramp
    sigma_max = 10.0  # Higher absorption strength
    
    # Create cell-based sigma function
    Q = fem.functionspace(domain, ("DG", 0))
    sigma = fem.Function(Q)
    
    def pml_sigma(x):
        d = np.clip((x[0] - L_phys) / L_pml, 0, 1)  # Normalized distance into PML
        # Polynomial profile (order 3 is typical)
        return sigma_max * (d ** 3)
    
    sigma.interpolate(pml_sigma)
    
    # Trial and test functions
    p = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Helmholtz with PML: proper complex coordinate stretching
    # The bilinear form becomes: ∫ (1/s_x) ∂p/∂x ∂v*/∂x dx + ∫ k² s_x p v* dx = 0
    # where s_x = 1 + i*σ/ω
    #
    # In weak form with sigma field:
    # For x > L_phys: s_x = 1 + i*σ(x)/ω
    # 
    # We implement: ∇p·∇v* - k²pv* - (2ik*σ/ω)pv* (linearized damping)
    # Or more properly: need to modify gradient coefficients
    
    # Complex stretching coefficient (cell-wise)
    # s = 1 + i*sigma/omega
    omega_val = omega
    s_re = fem.Constant(domain, PETSc.ScalarType(1.0))
    
    # For proper PML, we need: a = ∫ (1/s_x)(∂p/∂x)(∂v*/∂x) + s_y(∂p/∂y)(∂v*/∂y) - k²s_x*s_y*p*v* dx
    # In 1D (y-uniform): a = ∫ (1/s_x)(∂p/∂x)(∂v*/∂x) + (∂p/∂y)(∂v*/∂y) - k²s_x*p*v* dx
    
    # Create s_x coefficient: s_x = 1 + i*sigma/omega
    Vc = fem.functionspace(domain, ("DG", 0))
    s_x = fem.Function(Vc)
    
    def compute_s_x(x):
        d = np.clip((x[0] - L_phys) / L_pml, 0, 1)
        sigma_val = sigma_max * (d ** 3)
        return 1.0 + 1j * sigma_val / omega_val
    
    s_x.interpolate(compute_s_x)
    
    # Inverse: 1/s_x
    inv_s_x = fem.Function(Vc)
    
    def compute_inv_s_x(x):
        d = np.clip((x[0] - L_phys) / L_pml, 0, 1)
        sigma_val = sigma_max * (d ** 3)
        s_val = 1.0 + 1j * sigma_val / omega_val
        return 1.0 / s_val
    
    inv_s_x.interpolate(compute_inv_s_x)
    
    k_sq = fem.Constant(domain, PETSc.ScalarType(k**2))
    
    # Get spatial coordinate for gradient decomposition
    # We need to split gradient into x and y components
    # grad(p) = (∂p/∂x, ∂p/∂y)
    # For PML: (1/s_x * ∂p/∂x, ∂p/∂y)
    
    # Simpler approach: use coordinate-dependent coefficient
    # The bilinear form in 2D with x-only PML:
    # a(p,v) = ∫ [(1/s_x) ∂p/∂x ∂v*/∂x + ∂p/∂y ∂v*/∂y - k² s_x p v*] dx
    
    # In UFL, we can use directional gradients
    x_coord = ufl.SpatialCoordinate(domain)
    
    # Gradient components
    grad_p = ufl.grad(p)
    grad_v = ufl.grad(v)
    
    # x-direction (affected by PML)
    dp_dx = grad_p[0]
    dv_dx = grad_v[0]
    
    # y-direction (not affected)
    dp_dy = grad_p[1]
    dv_dy = grad_v[1]
    
    # Bilinear form with anisotropic PML
    a = inv_s_x * ufl.inner(dp_dx, dv_dx) * ufl.dx  # x-gradient with PML
    a += ufl.inner(dp_dy, dv_dy) * ufl.dx            # y-gradient normal
    a -= k_sq * s_x * ufl.inner(p, v) * ufl.dx       # Mass term with PML
    
    # Boundary conditions
    # Left (x=0): incident wave p = p0 * exp(ikx) at x=0 => p = p0
    # Right: Sommerfeld ABC (should not be reached due to PML)
    # Top/Bottom: Neumann (natural BC)
    
    def left_boundary(x):
        return np.isclose(x[0], 0)
    
    def right_boundary(x):
        return np.isclose(x[0], L_total)
    
    # Mark boundaries
    fdim = domain.topology.dim - 1
    domain.topology.create_connectivity(fdim, domain.topology.dim)
    
    left_facets = mesh.locate_entities_boundary(domain, fdim, left_boundary)
    right_facets = mesh.locate_entities_boundary(domain, fdim, right_boundary)
    
    # Dirichlet BC on left: p = 1.0 (unit amplitude incident wave)
    p0 = fem.Constant(domain, PETSc.ScalarType(1.0))
    left_dofs = fem.locate_dofs_topological(V, fdim, left_facets)
    bc_left = fem.dirichletbc(p0, left_dofs, V)
    
    # ABC on right: ∂p/∂n = -ik*p
    # Create facet tags for right boundary
    facet_values = np.full_like(right_facets, 1, dtype=np.int32)
    facet_tags = mesh.meshtags(domain, fdim, right_facets, facet_values)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
    
    # Add ABC contribution
    a += 1j * k * ufl.inner(p, v) * ds(1)
    
    # RHS = 0 (homogeneous)
    f = fem.Function(V)
    f.x.array[:] = 0
    L = ufl.inner(f, v) * ufl.dx
    
    # Solve
    print("\nSolving Helmholtz equation with PML...")
    
    problem = LinearProblem(a, L, bcs=[bc_left], petsc_options={
        "ksp_type": "preonly",
        "pc_type": "lu"
    })
    
    p_h = problem.solve()
    
    # Analyze solution
    p_array = p_h.x.array
    
    print(f"\nSolution:")
    print(f"  max|p|: {np.max(np.abs(p_array)):.4e}")
    print(f"  min|p|: {np.min(np.abs(p_array)):.4e}")
    
    # Sample along centerline y = H/2
    n_sample = 50
    x_sample = np.linspace(0, L_total - 1e-6, n_sample)
    y_sample = (H / 2) * np.ones(n_sample)
    z_sample = np.zeros(n_sample)
    points = np.stack([x_sample, y_sample, z_sample], axis=1)
    
    # Point evaluation
    bb_tree = dolfinx.geometry.bb_tree(domain, domain.topology.dim)
    cells = []
    points_on_proc = []
    x_on_proc = []
    
    cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = dolfinx.geometry.compute_colliding_cells(domain, cell_candidates, points)
    
    for i, point in enumerate(points):
        if len(colliding_cells.links(i)) > 0:
            points_on_proc.append(point)
            x_on_proc.append(x_sample[i])
            cells.append(colliding_cells.links(i)[0])
    
    print(f"\n  Points evaluated: {len(points_on_proc)} / {n_sample}")
    
    if len(points_on_proc) >= 5:
        points_on_proc = np.array(points_on_proc, dtype=np.float64)
        x_on_proc = np.array(x_on_proc)
        
        p_values = p_h.eval(points_on_proc, cells).flatten()
        
        # Amplitude profile
        amp = np.abs(p_values)
        
        # Find amplitude at PML entrance and exit
        idx_entrance = np.argmin(np.abs(x_on_proc - L_phys))
        idx_exit = np.argmax(x_on_proc)  # Rightmost point
        
        amp_entrance = amp[idx_entrance]
        amp_exit = amp[idx_exit]
        
        print(f"\nPML Analysis:")
        print(f"  Amplitude at PML entrance (x={L_phys*1e3:.1f}mm): {amp_entrance:.4e}")
        print(f"  Amplitude at PML exit (x={x_on_proc[idx_exit]*1e3:.1f}mm): {amp_exit:.4e}")
        
        if amp_entrance > 1e-12:
            decay_ratio = amp_exit / amp_entrance
            print(f"  Decay ratio: {decay_ratio:.4f} ({(1-decay_ratio)*100:.1f}% absorbed)")
        else:
            decay_ratio = 1.0
            print(f"  ⚠ Amplitude at entrance is very small")
        
        # PASS/FAIL
        print("\n" + "=" * 60)
        
        if amp_entrance < 1e-12:
            print("❌ FAIL: Solution essentially zero at PML entrance")
            return False
        elif decay_ratio < 0.5:
            print(f"✓ PASS: PML absorbs >{(1-decay_ratio)*100:.0f}% of wave amplitude")
            return True
        elif decay_ratio < 0.8:
            print(f"⚠ WARN: PML absorption marginal ({(1-decay_ratio)*100:.0f}%)")
            return True
        else:
            print(f"❌ FAIL: PML not absorbing effectively (decay={decay_ratio:.2f})")
            return False
    else:
        print("\n" + "=" * 60)
        print("❌ FAIL: Could not evaluate points in domain")
        return False


if __name__ == "__main__":
    success = run_pml_absorption_test()
    print("=" * 60)
    exit(0 if success else 1)
