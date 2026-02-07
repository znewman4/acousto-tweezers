"""
Truth-validated PML test using proper reflection coefficient.

Uses 2-wave fitting along a probe line to compute actual reflection
coefficient R = |B|/|A| for incident/reflected waves.

This replaces the misleading "single-point amplitude ratio" from v2.3.0.
"""

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import gmsh
from pathlib import Path
import json
from datetime import datetime
import sys

# Check for complex PETSc (REQUIRED)
if not np.issubdtype(PETSc.ScalarType, np.complexfloating):
    print("=" * 60)
    print("ERROR: PETSc must be built with complex scalar support!")
    print(f"  Current PETSc.ScalarType: {PETSc.ScalarType}")
    print("=" * 60)
    sys.exit(1)

from dolfinx import fem, mesh as dmesh
from dolfinx.fem import (
    Function, FunctionSpace, Constant,
    form
)
import ufl
from ufl import inner, grad, ds, Measure

# Import production PML code (SINGLE SOURCE OF TRUTH)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tweezers.fenicsx.pml import (
    build_pml_stretch_dg0,
    helmholtz_anisotropic_pml_forms
)

# ============================================================
# PARAMETERS
# ============================================================

# Frequency and material
FREQ = 1.0e6  # 1 MHz
OMEGA = 2 * np.pi * FREQ
RHO_WATER = 1000.0  # kg/m³
C_WATER = 1480.0  # m/s
WAVELENGTH = C_WATER / FREQ  # ~1.5 mm

# Geometry (rectangular box)
L = 3.0 * WAVELENGTH  # 4.5 mm physical domain
PML_THICKNESS = 1.5 * WAVELENGTH  # 2.25 mm PML
L_TOTAL = L + PML_THICKNESS

# PML parameters
SIGMA_MAX = np.pi * 1e6  # ~3.14e6
PML_POWER = 2

# Mesh resolution
PPW = 5  # Points per wavelength
H = WAVELENGTH / PPW  # Element size

# Actuation
ACTUATION_VELOCITY = 1.0e-6  # m/s (1 μm/s)

# Tags
TAG_WATER = 1
TAG_PML = 2
TAG_ACTUATION = 10
TAG_OUTER = 20

# Output
OUTPUT_DIR = Path("results/validation/pml_reflection_fit")
RUN_ID = datetime.now().strftime("run_%Y%m%d_%H%M%S")
RUN_DIR = OUTPUT_DIR / RUN_ID


def create_box_mesh_with_pml(L, pml_thickness, h, comm=MPI.COMM_WORLD):
    """
    Create 3D box mesh with PML on +x side only.
    
    Water: [0, L]^3
    PML:   [L, L+pml] × [0, L] × [0, L]
    """
    if comm.rank == 0:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("pml_box")
        
        # Create boxes
        water_box = gmsh.model.occ.addBox(0, 0, 0, L, L, L)
        pml_box = gmsh.model.occ.addBox(L, 0, 0, pml_thickness, L, L)
        
        # Boolean fragment for conforming interface
        gmsh.model.occ.fragment([(3, water_box), (3, pml_box)], [])
        gmsh.model.occ.synchronize()
        
        # Get volumes and assign tags
        volumes = gmsh.model.getEntities(dim=3)
        
        # Classify by bounding box
        for dim, tag in volumes:
            bbox = gmsh.model.getBoundingBox(dim, tag)
            x_min, y_min, z_min, x_max, y_max, z_max = bbox
            x_center = (x_min + x_max) / 2
            
            if x_center < L - 1e-6:
                # Water region
                gmsh.model.addPhysicalGroup(3, [tag], TAG_WATER)
            else:
                # PML region
                gmsh.model.addPhysicalGroup(3, [tag], TAG_PML)
        
        # Find boundaries
        gmsh.model.occ.synchronize()
        surfaces = gmsh.model.getEntities(dim=2)
        
        actuation_faces = []
        outer_faces = []
        
        for dim, tag in surfaces:
            com = gmsh.model.occ.getCenterOfMass(dim, tag)
            x_com = com[0]
            
            if abs(x_com) < 1e-6:
                # x=0 face (actuation)
                actuation_faces.append(tag)
            elif abs(x_com - (L + pml_thickness)) < 1e-6:
                # x=L+pml face (outer boundary)
                outer_faces.append(tag)
        
        if actuation_faces:
            gmsh.model.addPhysicalGroup(2, actuation_faces, TAG_ACTUATION)
        if outer_faces:
            gmsh.model.addPhysicalGroup(2, outer_faces, TAG_OUTER)
        
        # Set mesh size
        gmsh.model.mesh.setSize(gmsh.model.getEntities(dim=0), h)
        
        # Generate mesh
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.setOrder(1)
        
        from dolfinx.io import gmshio
        mesh, cell_tags, facet_tags = gmshio.model_to_mesh(gmsh.model, comm, 0, gdim=3)
        
        gmsh.finalize()
        
        return mesh, cell_tags, facet_tags
    else:
        from dolfinx.io import gmshio
        mesh_data = gmshio.model_to_mesh(gmsh.model, comm, 0, gdim=3)
        mesh = mesh_data.mesh
        cell_tags = mesh_data.cell_tags
        facet_tags = mesh_data.facet_tags
        
        return mesh, cell_tags, facet_tags


def solve_helmholtz_pml(mesh, cell_tags, facet_tags, pml_active=True):
    """
    Solve Helmholtz with optional PML using production code.
    
    Returns
    -------
    p_h : Function
        Pressure field
    converged : bool
        Solver convergence status
    iterations : int
        Number of iterations
    pml_stats : dict
        PML diagnostics (Im(s) in water/PML)
    """
    # Function space (CG1) with complex dtype
    V = fem.functionspace(mesh, ("Lagrange", 1))
    p = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Create complex solution function
    p_h = fem.Function(V, dtype=np.complex128)
    
    # Wavenumber
    k = OMEGA / C_WATER
    
    # Domain measures
    dx_water = Measure("dx", domain=mesh, subdomain_data=cell_tags, subdomain_id=TAG_WATER)
    dx_pml = Measure("dx", domain=mesh, subdomain_data=cell_tags, subdomain_id=TAG_PML)
    ds_act = Measure("ds", domain=mesh, subdomain_data=facet_tags)
    
    # Build PML stretch if active
    if pml_active:
        s_x, s_x_inv, im_s_water, im_s_pml = build_pml_stretch_dg0(
            mesh, cell_tags, TAG_PML, L, PML_THICKNESS, OMEGA, SIGMA_MAX, PML_POWER, TAG_WATER
        )
        pml_stats = {"im_s_water": im_s_water, "im_s_pml": im_s_pml}
    else:
        # No PML: s_x = 1 everywhere
        DG0 = fem.functionspace(mesh, ("DG", 0))
        s_x = fem.Function(DG0, dtype=np.complex128)
        s_x_inv = fem.Function(DG0, dtype=np.complex128)
        s_x.x.array[:] = 1.0 + 0j
        s_x_inv.x.array[:] = 1.0 + 0j
        pml_stats = {"im_s_water": 0.0, "im_s_pml": 0.0}
    
    # Build anisotropic PML form (production)
    a_form, _ = helmholtz_anisotropic_pml_forms(
        p, v, mesh, k, RHO_WATER, OMEGA,
        s_x, s_x_inv,
        dx_water, dx_pml,
        source_form=None
    )
    
    # RHS: Actuation on x=0 face
    actuation_neumann = PETSc.ScalarType(-1j * OMEGA * RHO_WATER * ACTUATION_VELOCITY)
    g = Constant(mesh, actuation_neumann)
    L_form = inner(g, v) * ds_act(TAG_ACTUATION)
    
    # Compile
    a_compiled = form(a_form)
    L_compiled = form(L_form)
    
    # Assemble
    from dolfinx.fem.petsc import create_matrix, create_vector, assemble_matrix, assemble_vector as assemble_vector_petsc
    A_petsc = create_matrix(a_compiled)
    A_petsc.zeroEntries()
    assemble_matrix(A_petsc, a_compiled, bcs=[])
    A_petsc.assemble()
    
    b_petsc = create_vector(L_compiled)
    assemble_vector_petsc(b_petsc, L_compiled)
    b_petsc.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    
    # Solve
    p_h = fem.Function(V)
    ksp = PETSc.KSP().create(mesh.comm)
    ksp.setOperators(A_petsc)
    
    # Use direct solver for reliability
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.setFromOptions()
    
    ksp.solve(b_petsc, p_h.x.petsc_vec)
    
    converged = ksp.getConvergedReason() > 0
    iterations = ksp.getIterationNumber()
    
    return p_h, converged, iterations, pml_stats


def evaluate_field_at_point(field, mesh, point):
    """Evaluate field at a single point (MPI-safe)."""
    from dolfinx import geometry
    bb_tree = geometry.bb_tree(mesh, mesh.topology.dim)
    cells = geometry.compute_collisions_points(bb_tree, point.reshape(1, 3))
    if len(cells.links(0)) > 0:
        cell_idx = cells.links(0)[0]
        value = field.eval(point, cell_idx)
        return value[0]
    else:
        return None


def fit_two_wave_model(x_probe, p_probe, k):
    """
    Fit pressure samples to 2-wave model: p(x) = A*exp(-ikx) + B*exp(+ikx)
    
    Uses least squares to solve for complex A, B.
    
    Parameters
    ----------
    x_probe : array
        x-coordinates of probe points
    p_probe : array
        Complex pressure at probe points
    k : float
        Wavenumber
    
    Returns
    -------
    A : complex
        Forward wave amplitude
    B : complex
        Backward wave amplitude
    R : float
        Reflection coefficient |B|/|A|
    residual : float
        RMS fit residual
    """
    # Design matrix: [exp(-ikx), exp(+ikx)]
    X = np.column_stack([
        np.exp(-1j * k * x_probe),
        np.exp(1j * k * x_probe)
    ])
    
    # Least squares: X @ [A, B]^T = p
    # Normal equations: X^H @ X @ [A,B]^T = X^H @ p
    XH_X = X.conj().T @ X
    XH_p = X.conj().T @ p_probe
    
    # Solve 2x2 system
    AB = np.linalg.solve(XH_X, XH_p)
    A, B = AB[0], AB[1]
    
    # Reflection coefficient
    R = np.abs(B) / np.abs(A) if np.abs(A) > 1e-12 else np.inf
    
    # Residual
    p_fit = X @ AB
    residual = np.sqrt(np.mean(np.abs(p_probe - p_fit)**2))
    
    return A, B, R, residual


def run_validation():
    """Run PML validation with reflection coefficient."""
    
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    if rank == 0:
        print("=" * 70)
        print("PML Reflection Coefficient Validation (2-Wave Fit)")
        print("=" * 70)
        print(f"Frequency: {FREQ/1e6:.2f} MHz")
        print(f"Wavelength: {WAVELENGTH*1e3:.3f} mm")
        print(f"Domain: {L*1e3:.2f} mm (water) + {PML_THICKNESS*1e3:.2f} mm (PML)")
        print(f"Resolution: {PPW} PPW")
        print()
    
    # ============================================================
    # STEP 1: Create Mesh
    # ============================================================
    if rank == 0:
        print("STEP 1: Creating mesh...")
    
    mesh, cell_tags, facet_tags = create_box_mesh_with_pml(L, PML_THICKNESS, H, comm)
    
    num_cells = mesh.topology.index_map(mesh.topology.dim).size_global
    if rank == 0:
        print(f"  Cells: {num_cells}")
        print()
    
    # ============================================================
    # STEP 2: Solve with PML ON
    # ============================================================
    if rank == 0:
        print("STEP 2: Solving with PML ON...")
    
    p_h_on, conv_on, iter_on, pml_stats_on = solve_helmholtz_pml(
        mesh, cell_tags, facet_tags, pml_active=True
    )
    
    if rank == 0:
        print(f"  Converged: {conv_on} ({iter_on} iterations)")
        print(f"  Im(s) in water: {pml_stats_on['im_s_water']:.6e}")
        print(f"  Im(s) in PML: {pml_stats_on['im_s_pml']:.6e}")
        print()
    
    # ============================================================
    # STEP 3: Solve with PML OFF
    # ============================================================
    if rank == 0:
        print("STEP 3: Solving with PML OFF...")
    
    p_h_off, conv_off, iter_off, pml_stats_off = solve_helmholtz_pml(
        mesh, cell_tags, facet_tags, pml_active=False
    )
    
    if rank == 0:
        print(f"  Converged: {conv_off} ({iter_off} iterations)")
        print()
    
    # ============================================================
    # STEP 4: Two-Wave Fit for Reflection Coefficient
    # ============================================================
    if rank == 0:
        print("STEP 4: Computing reflection coefficient...")
    
    # Probe line in water near PML interface
    x_start = L - 1.5 * WAVELENGTH  # Start 1.5λ before interface
    x_end = L - 0.1 * WAVELENGTH    # End 0.1λ before interface
    n_probes = 100
    
    x_probe_vals = np.linspace(x_start, x_end, n_probes)
    y_probe = L / 2
    z_probe = L / 2
    
    # Evaluate fields along probe line
    p_on_vals = []
    p_off_vals = []
    
    for xi in x_probe_vals:
        point = np.array([xi, y_probe, z_probe])
        
        p_on = evaluate_field_at_point(p_h_on, mesh, point)
        p_off = evaluate_field_at_point(p_h_off, mesh, point)
        
        if p_on is not None:
            p_on_vals.append(p_on)
            p_off_vals.append(p_off)
    
    # Gather on rank 0
    p_on_vals = comm.gather(p_on_vals, root=0)
    p_off_vals = comm.gather(p_off_vals, root=0)
    
    if rank == 0:
        # Flatten and filter
        p_on_all = [v for sublist in p_on_vals for v in sublist if v is not None]
        p_off_all = [v for sublist in p_off_vals for v in sublist if v is not None]
        
        p_on_array = np.array(p_on_all)
        p_off_array = np.array(p_off_all)
        
        # Fit 2-wave model
        k = OMEGA / C_WATER
        
        A_on, B_on, R_on, res_on = fit_two_wave_model(x_probe_vals, p_on_array, k)
        A_off, B_off, R_off, res_off = fit_two_wave_model(x_probe_vals, p_off_array, k)
        
        print(f"  Probe line: x ∈ [{x_start*1e3:.3f}, {x_end*1e3:.3f}] mm")
        print(f"  Number of probes: {n_probes}")
        print()
        print(f"  PML ON:")
        print(f"    Forward amplitude |A|: {np.abs(A_on):.3e} Pa")
        print(f"    Backward amplitude |B|: {np.abs(B_on):.3e} Pa")
        print(f"    Reflection coefficient R: {R_on:.4f}")
        print(f"    Fit residual: {res_on:.3e}")
        print()
        print(f"  PML OFF:")
        print(f"    Forward amplitude |A|: {np.abs(A_off):.3e} Pa")
        print(f"    Backward amplitude |B|: {np.abs(B_off):.3e} Pa")
        print(f"    Reflection coefficient R: {R_off:.4f}")
        print(f"    Fit residual: {res_off:.3e}")
        print()
        
        # ============================================================
        # STEP 5: Validation Criteria
        # ============================================================
        print("=" * 70)
        print("VALIDATION")
        print("=" * 70)
        
        # Thresholds
        R_on_max = 0.10  # PML should give R < 10%
        R_off_min = 0.20  # No PML should give R > 20% (strong reflection)
        
        test_passed = True
        
        # Check 1: PML active
        if pml_stats_on['im_s_pml'] < 1e-6:
            print("  ✗ FAIL: PML not active (Im(s) = 0 in PML region)")
            test_passed = False
        else:
            print(f"  ✓ PASS: PML active (Im(s) = {pml_stats_on['im_s_pml']:.3f})")
        
        # Check 2: Reflection reduced
        if R_on < R_on_max:
            print(f"  ✓ PASS: Reflection with PML: R = {R_on:.4f} < {R_on_max}")
        else:
            print(f"  ✗ FAIL: Reflection with PML: R = {R_on:.4f} ≥ {R_on_max}")
            test_passed = False
        
        # Check 3: Baseline reflection exists
        if R_off > R_off_min:
            print(f"  ✓ PASS: Reflection without PML: R = {R_off:.4f} > {R_off_min}")
        else:
            print(f"  ⚠ WARN: Reflection without PML: R = {R_off:.4f} ≤ {R_off_min} (expected higher)")
        
        # Check 4: Reduction factor
        reduction = R_off / R_on if R_on > 1e-6 else np.inf
        if reduction > 2.0:
            print(f"  ✓ PASS: Reduction factor: {reduction:.2f}x")
        else:
            print(f"  ⚠ WARN: Reduction factor only {reduction:.2f}x")
        
        print()
        if test_passed:
            print("  ✓ TEST PASSED: PML reduces reflection")
        else:
            print("  ✗ TEST FAILED: PML not functioning correctly")
        
        print("=" * 70)
        
        # ============================================================
        # STEP 6: Save Results
        # ============================================================
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        
        diagnostics = {
            "timestamp": datetime.now().isoformat(),
            "frequency_Hz": FREQ,
            "wavelength_m": WAVELENGTH,
            "domain_length_m": L,
            "pml_thickness_m": PML_THICKNESS,
            "resolution_ppw": PPW,
            "num_cells": int(num_cells),
            "pml_on": {
                "reflection_coefficient": float(R_on),
                "forward_amplitude": float(np.abs(A_on)),
                "backward_amplitude": float(np.abs(B_on)),
                "fit_residual": float(res_on),
                "im_s_water": float(pml_stats_on['im_s_water']),
                "im_s_pml": float(pml_stats_on['im_s_pml']),
            },
            "pml_off": {
                "reflection_coefficient": float(R_off),
                "forward_amplitude": float(np.abs(A_off)),
                "backward_amplitude": float(np.abs(B_off)),
                "fit_residual": float(res_off),
            },
            "reduction_factor": float(reduction),
            "test_passed": test_passed,
            "probe_line": {
                "x_start": float(x_start),
                "x_end": float(x_end),
                "n_points": n_probes,
                "y": float(y_probe),
                "z": float(z_probe),
            }
        }
        
        with open(RUN_DIR / "diagnostics.json", "w") as f:
            json.dump(diagnostics, f, indent=2)
        
        print(f"\nResults saved to: {RUN_DIR}")


if __name__ == "__main__":
    run_validation()
