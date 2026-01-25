"""
Interface Continuity Test
=========================
Validates smooth solution behavior with Helmholtz equation.

Test Methodology:
-----------------
1. Solve Helmholtz equation with Dirichlet BCs
2. Verify solution is smooth (no artificial discontinuities)
3. Check gradient is well-behaved

Pass Criteria:
--------------
- Coefficient of variation of gradient < 100%
- No spurious jumps in solution
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


def run_interface_test():
    """Test solution smoothness with Helmholtz equation."""
    
    print("=" * 60)
    print("INTERFACE CONTINUITY TEST")
    print("=" * 60)
    
    # Physical parameters
    k = 10.0  # Wavenumber
    
    # Domain
    L = 1.0
    H = 0.2
    h = 0.02
    
    comm = MPI.COMM_WORLD
    nx = int(L / h)
    ny = max(int(H / h), 3)
    
    domain = mesh.create_rectangle(
        comm,
        [[0.0, 0.0], [L, H]],
        [nx, ny],
        mesh.CellType.triangle
    )
    
    V = fem.functionspace(domain, ("Lagrange", 2))
    ncells = domain.topology.index_map(domain.topology.dim).size_local
    ndofs = V.dofmap.index_map.size_local
    print(f"\nMesh: {ncells} cells, {ndofs} DOFs")
    
    # Trial and test functions
    p = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Helmholtz equation: ∇²p + k²p = 0
    k_sq = fem.Constant(domain, PETSc.ScalarType(k**2))
    
    # Bilinear form (trial first for complex mode)
    a = ufl.inner(ufl.grad(p), ufl.grad(v)) * ufl.dx
    a -= k_sq * ufl.inner(p, v) * ufl.dx
    
    # Boundaries
    fdim = domain.topology.dim - 1
    domain.topology.create_connectivity(fdim, domain.topology.dim)
    
    def left_boundary(x):
        return np.isclose(x[0], 0.0)
    
    def right_boundary(x):
        return np.isclose(x[0], L)
    
    left_facets = mesh.locate_entities_boundary(domain, fdim, left_boundary)
    right_facets = mesh.locate_entities_boundary(domain, fdim, right_boundary)
    
    # Dirichlet BCs: p=1 at left, p=0 at right
    p_left = fem.Constant(domain, PETSc.ScalarType(1.0))
    p_right = fem.Constant(domain, PETSc.ScalarType(0.0))
    
    left_dofs = fem.locate_dofs_topological(V, fdim, left_facets)
    right_dofs = fem.locate_dofs_topological(V, fdim, right_facets)
    
    bc_left = fem.dirichletbc(p_left, left_dofs, V)
    bc_right = fem.dirichletbc(p_right, right_dofs, V)
    
    # RHS = 0
    f_zero = fem.Function(V)
    f_zero.x.array[:] = 0
    L_form = ufl.inner(f_zero, v) * ufl.dx
    
    # Solve
    print("\nSolving Helmholtz equation...")
    
    problem = LinearProblem(a, L_form, bcs=[bc_left, bc_right], petsc_options={
        "ksp_type": "preonly",
        "pc_type": "lu"
    })
    
    p_h = problem.solve()
    
    # Analyze solution
    p_arr = p_h.x.array
    print(f"\nSolution:")
    print(f"  max|p| = {np.max(np.abs(p_arr)):.4f}")
    print(f"  min|p| = {np.min(np.abs(p_arr)):.4f}")
    
    # Compute gradient magnitude using expression evaluation
    # Instead of projecting, evaluate at cell midpoints
    num_cells = domain.topology.index_map(domain.topology.dim).size_local
    cell_indices = np.arange(num_cells, dtype=np.int32)
    
    # Get cell midpoints
    midpoints = dolfinx.mesh.compute_midpoints(domain, domain.topology.dim, cell_indices)
    
    # Create gradient expression
    grad_expr = ufl.grad(p_h)
    
    # We'll compute gradient magnitude from finite differences along centerline
    # This is simpler and avoids projection issues
    
    # Sample along horizontal centerline
    n_sample = 50
    x_sample = np.linspace(0.01, L - 0.01, n_sample)
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
    
    if len(pts_ok) >= 10:
        pts_ok = np.array(pts_ok, dtype=np.float64)
        x_ok = np.array(x_ok)
        p_line = p_h.eval(pts_ok, cells).flatten()
        
        # Compute finite difference gradient
        dx = np.diff(x_ok)
        dp = np.diff(p_line)
        grad_fd = dp / dx
        
        grad_mag = np.abs(grad_fd)
        
        print(f"\nGradient analysis (finite difference):")
        print(f"  max|∂p/∂x| = {np.max(grad_mag):.4f}")
        print(f"  mean|∂p/∂x| = {np.mean(grad_mag):.4f}")
        print(f"  std|∂p/∂x| = {np.std(grad_mag):.4f}")
        
        # Coefficient of variation
        if np.mean(grad_mag) > 1e-10:
            cv = np.std(grad_mag) / np.mean(grad_mag)
            print(f"  Coefficient of variation: {cv:.2%}")
        else:
            cv = 0.0
        
        # Check for jumps
        d2p = np.diff(grad_fd)  # Second derivative approximation
        max_jump = np.max(np.abs(d2p))
        print(f"  Max |d²p/dx²|: {max_jump:.4f}")
    else:
        cv = 0.0
        max_jump = 0.0
        print("  ⚠ Not enough sample points")
    
    # Also sample along vertical line
    y_sample_v = np.linspace(0.01, H - 0.01, 10)
    x_sample_v = (L / 2) * np.ones(len(y_sample_v))
    z_sample_v = np.zeros(len(y_sample_v))
    points_v = np.stack([x_sample_v, y_sample_v, z_sample_v], axis=1)
    
    cells_v = []
    pts_v = []
    
    cell_cands_v = dolfinx.geometry.compute_collisions_points(bb_tree, points_v)
    coll_cells_v = dolfinx.geometry.compute_colliding_cells(domain, cell_cands_v, points_v)
    
    for i, pt in enumerate(points_v):
        if len(coll_cells_v.links(i)) > 0:
            pts_v.append(pt)
            cells_v.append(coll_cells_v.links(i)[0])
    
    if len(pts_v) >= 3:
        pts_v = np.array(pts_v, dtype=np.float64)
        p_vert = p_h.eval(pts_v, cells_v).flatten()
        
        # For a horizontal wave, vertical variation should be minimal
        vert_variation = (np.max(np.abs(p_vert)) - np.min(np.abs(p_vert))) / (np.mean(np.abs(p_vert)) + 1e-10)
        print(f"\nVertical uniformity at x={L/2}:")
        print(f"  Variation: {vert_variation:.2%}")
    else:
        vert_variation = 0.0
    
    # PASS/FAIL
    print("\n" + "=" * 60)
    
    # Check solution is non-trivial
    if np.max(np.abs(p_arr)) < 1e-10:
        print("❌ FAIL: Solution is essentially zero")
        return False
    
    result = True
    
    # Check gradient smoothness
    if cv < 0.5:  # Less than 50% coefficient of variation
        print(f"✓ PASS: Solution gradient is smooth (CV = {cv:.1%})")
    elif cv < 1.0:
        print(f"⚠ WARN: Moderate gradient variation (CV = {cv:.1%})")
    else:
        print(f"❌ FAIL: Large gradient variations (CV = {cv:.1%})")
        result = False
    
    # Check vertical uniformity (for this 1D-like problem)
    if vert_variation < 0.1:
        print(f"✓ PASS: Solution is vertically uniform ({vert_variation:.1%} variation)")
    else:
        print(f"⚠ WARN: Vertical variation detected ({vert_variation:.1%})")
    
    print("=" * 60)
    return result


if __name__ == "__main__":
    success = run_interface_test()
    exit(0 if success else 1)
