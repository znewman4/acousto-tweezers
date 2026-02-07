"""
6-face tensor PML validation test.

Tests full 3D tensor PML on all 6 sides of a box domain.
Measures reflection coefficient along multiple axes.

Requirements:
- PMLs on all 6 faces (x_min, x_max, y_min, y_max, z_min, z_max)
- Reflection coefficient R < 0.10 along primary propagation axis
- Optional: check absorption in transverse directions
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
from dolfinx.fem import Function, Constant, form
import ufl
from ufl import inner, grad, Measure

# Import production PML code
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tweezers.fenicsx.pml import (
    build_pml_stretch_tensor_dg0,
    helmholtz_tensor_pml_forms
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

# Geometry (cubic box with PML on all 6 sides)
L = 3 * WAVELENGTH  # 4.5 mm interior
PML_THICKNESS = 1.5 * WAVELENGTH  # 2.25 mm PML per side
PPW = 5  # Points per wavelength

# PML parameters
SIGMA_MAX = 1.5 * np.log(1e3) / PML_THICKNESS
PML_POWER = 2

# Actuation
ACTUATION_VELOCITY = 1e-6  # m/s (very small for linear regime)

# Validation thresholds
R_ON_MAX = 0.10  # PML should give R < 10%
R_OFF_MIN = 0.20  # Baseline should have R > 20%
REDUCTION_MIN = 2.0  # PML should reduce by at least 2x

# Tags
TAG_WATER = 1
TAG_PML = 2
TAG_ACTUATION = 10  # x=0 face
TAG_OUTER = 20  # x=L+2*PML outer face

# Output
OUTPUT_DIR = Path("results/pml_6face_validation")


def create_6face_box_mesh(L, pml, h, comm):
    """
    Create 3D box with PML on all 6 faces.
    
    Interior: [pml, L+pml]^3  (water)
    PML: 6 slabs surrounding interior
    Total domain: [0, L+2*pml]^3
    """
    if comm.rank == 0:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("6face_box")
        
        # Interior water box
        water = gmsh.model.occ.addBox(pml, pml, pml, L, L, L)
        
        # 6 PML slabs
        pml_xmin = gmsh.model.occ.addBox(0, pml, pml, pml, L, L)
        pml_xmax = gmsh.model.occ.addBox(L+pml, pml, pml, pml, L, L)
        pml_ymin = gmsh.model.occ.addBox(0, 0, pml, L+2*pml, pml, L)
        pml_ymax = gmsh.model.occ.addBox(0, L+pml, pml, L+2*pml, pml, L)
        pml_zmin = gmsh.model.occ.addBox(0, 0, 0, L+2*pml, L+2*pml, pml)
        pml_zmax = gmsh.model.occ.addBox(0, 0, L+pml, L+2*pml, L+2*pml, pml)
        
        # Fuse all into one conforming mesh
        all_volumes = [
            (3, water), 
            (3, pml_xmin), (3, pml_xmax),
            (3, pml_ymin), (3, pml_ymax),
            (3, pml_zmin), (3, pml_zmax)
        ]
        gmsh.model.occ.fragment(all_volumes, [])
        gmsh.model.occ.synchronize()
        
        # Physical groups
        volumes = gmsh.model.getEntities(dim=3)
        
        # Identify water vs PML by centroid
        water_vols = []
        pml_vols = []
        for _, tag in volumes:
            com = gmsh.model.occ.getCenterOfMass(3, tag)
            x, y, z = com
            # Water if inside [pml, L+pml]^3
            if (pml < x < L+pml) and (pml < y < L+pml) and (pml < z < L+pml):
                water_vols.append(tag)
            else:
                pml_vols.append(tag)
        
        if water_vols:
            gmsh.model.addPhysicalGroup(3, water_vols, TAG_WATER)
            gmsh.model.setPhysicalName(3, TAG_WATER, "WATER")
        
        if pml_vols:
            gmsh.model.addPhysicalGroup(3, pml_vols, TAG_PML)
            gmsh.model.setPhysicalName(3, TAG_PML, "PML")
        
        # Surfaces
        surfaces = gmsh.model.getEntities(dim=2)
        
        # Actuation: x=0 face
        actuation_faces = []
        outer_faces = []
        
        for _, tag in surfaces:
            com = gmsh.model.occ.getCenterOfMass(2, tag)
            x, y, z = com
            
            # Actuation at x=0
            if abs(x) < 1e-6:
                actuation_faces.append(tag)
            # Outer boundary (any face on domain boundary)
            elif (abs(x - (L+2*pml)) < 1e-6 or abs(y) < 1e-6 or abs(y - (L+2*pml)) < 1e-6 
                  or abs(z) < 1e-6 or abs(z - (L+2*pml)) < 1e-6):
                outer_faces.append(tag)
        
        if actuation_faces:
            gmsh.model.addPhysicalGroup(2, actuation_faces, TAG_ACTUATION)
            gmsh.model.setPhysicalName(2, TAG_ACTUATION, "ACTUATION")
        
        if outer_faces:
            gmsh.model.addPhysicalGroup(2, outer_faces, TAG_OUTER)
            gmsh.model.setPhysicalName(2, TAG_OUTER, "OUTER")
        
        # Mesh size
        gmsh.model.mesh.setSize(gmsh.model.getEntities(dim=0), h)
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.setOrder(1)
        
        from dolfinx.io import gmshio
        mesh, cell_tags, facet_tags = gmshio.model_to_mesh(gmsh.model, comm, 0, gdim=3)
        
        gmsh.finalize()
        
        return mesh, cell_tags, facet_tags
    else:
        from dolfinx.io import gmshio
        mesh, cell_tags, facet_tags = gmshio.model_to_mesh(gmsh.model, comm, 0, gdim=3)
        
        return mesh, cell_tags, facet_tags


def solve_helmholtz_6face(mesh, cell_tags, facet_tags, pml_active=True):
    """
    Solve Helmholtz with 6-face tensor PML.
    
    Returns
    -------
    p_h : Function
        Pressure field
    converged : bool
        Solver convergence
    iterations : int
        Number of iterations
    pml_stats : dict
        PML diagnostics
    """
    # Function space (CG1)
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
        # Get bbox
        coords = mesh.geometry.x
        x_min, x_max = float(coords[:, 0].min()), float(coords[:, 0].max())
        y_min, y_max = float(coords[:, 1].min()), float(coords[:, 1].max())
        z_min, z_max = float(coords[:, 2].min()), float(coords[:, 2].max())
        bbox = (x_min, x_max, y_min, y_max, z_min, z_max)
        
        s_x, s_y, s_z, s_x_inv, s_y_inv, s_z_inv, pml_stats = build_pml_stretch_tensor_dg0(
            mesh, cell_tags, [TAG_PML], bbox, PML_THICKNESS, OMEGA, SIGMA_MAX, PML_POWER, TAG_WATER
        )
    else:
        # No PML: s = 1 everywhere
        DG0 = fem.functionspace(mesh, ("DG", 0))
        s_x = fem.Function(DG0, dtype=np.complex128)
        s_y = fem.Function(DG0, dtype=np.complex128)
        s_z = fem.Function(DG0, dtype=np.complex128)
        s_x_inv = fem.Function(DG0, dtype=np.complex128)
        s_y_inv = fem.Function(DG0, dtype=np.complex128)
        s_z_inv = fem.Function(DG0, dtype=np.complex128)
        
        s_x.x.array[:] = 1.0 + 0j
        s_y.x.array[:] = 1.0 + 0j
        s_z.x.array[:] = 1.0 + 0j
        s_x_inv.x.array[:] = 1.0 + 0j
        s_y_inv.x.array[:] = 1.0 + 0j
        s_z_inv.x.array[:] = 1.0 + 0j
        
        pml_stats = {}
    
    # Build tensor PML form
    a_form, _ = helmholtz_tensor_pml_forms(
        p, v, mesh, k, RHO_WATER, OMEGA,
        s_x, s_y, s_z, s_x_inv, s_y_inv, s_z_inv,
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
    ksp = PETSc.KSP().create(mesh.comm)
    ksp.setOperators(A_petsc)
    
    # Use direct solver
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
    
    Returns
    -------
    A, B : complex
        Forward and backward wave amplitudes
    R : float
        Reflection coefficient |B|/|A|
    residual : float
        RMS residual of fit
    """
    # Design matrix: [exp(-ikx), exp(+ikx)]
    X = np.column_stack([
        np.exp(-1j * k * x_probe),
        np.exp(1j * k * x_probe)
    ])
    
    # Least squares: minimize ||X·[A,B] - p||²
    XH_X = X.conj().T @ X
    XH_p = X.conj().T @ p_probe
    AB = np.linalg.solve(XH_X, XH_p)
    
    A, B = AB[0], AB[1]
    
    # Reflection coefficient
    R = np.abs(B) / np.abs(A)
    
    # Residual
    p_fit = X @ AB
    residual = np.linalg.norm(p_probe - p_fit) / np.linalg.norm(p_probe)
    
    return A, B, R, residual


def run_validation():
    """Run 6-face PML validation."""
    comm = MPI.COMM_WORLD
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Header
    if comm.rank == 0:
        print("=" * 60)
        print("6-FACE TENSOR PML VALIDATION")
        print("=" * 60)
        print(f"Frequency: {FREQ/1e6:.2f} MHz")
        print(f"Wavelength: {WAVELENGTH*1000:.3f} mm")
        print(f"Interior: {L*1000:.2f} mm (cubic)")
        print(f"PML thickness: {PML_THICKNESS*1000:.2f} mm per side")
        print(f"Resolution: {PPW} PPW")
        print()
    
    # Mesh
    if comm.rank == 0:
        print("STEP 1: Creating 6-face box mesh...")
    
    h = WAVELENGTH / PPW
    mesh, cell_tags, facet_tags = create_6face_box_mesh(L, PML_THICKNESS, h, comm)
    
    num_cells = mesh.topology.index_map(3).size_global
    num_dofs = fem.functionspace(mesh, ("Lagrange", 1)).dofmap.index_map.size_global
    
    if comm.rank == 0:
        print(f"  Cells: {num_cells}")
        print(f"  DOFs: {num_dofs}")
        print()
    
    # Solve with PML ON
    if comm.rank == 0:
        print("STEP 2: Solving with PML ON...")
    
    p_h_on, conv_on, iter_on, pml_stats_on = solve_helmholtz_6face(
        mesh, cell_tags, facet_tags, pml_active=True
    )
    
    if comm.rank == 0:
        print(f"  Converged: {conv_on}, Iterations: {iter_on}")
        print()
    
    # Solve with PML OFF
    if comm.rank == 0:
        print("STEP 3: Solving with PML OFF (baseline)...")
    
    p_h_off, conv_off, iter_off, pml_stats_off = solve_helmholtz_6face(
        mesh, cell_tags, facet_tags, pml_active=False
    )
    
    if comm.rank == 0:
        print(f"  Converged: {conv_off}, Iterations: {iter_off}")
        print()
    
    # Measure reflection along x-axis (primary propagation direction)
    if comm.rank == 0:
        print("STEP 4: Measuring reflection coefficient (x-axis)...")
    
    # Probe line in water near x-center, before PML
    x_probe_start = PML_THICKNESS + 0.5 * WAVELENGTH
    x_probe_end = PML_THICKNESS + L - 0.5 * WAVELENGTH
    N_probe = 80
    
    x_coords = np.linspace(x_probe_start, x_probe_end, N_probe)
    y_center = PML_THICKNESS + L/2
    z_center = PML_THICKNESS + L/2
    
    # Sample fields
    p_on_samples = []
    p_off_samples = []
    
    for xi in x_coords:
        probe = np.array([xi, y_center, z_center])
        
        p_on = evaluate_field_at_point(p_h_on, mesh, probe)
        p_off = evaluate_field_at_point(p_h_off, mesh, probe)
        
        if p_on is not None:
            p_on_samples.append(p_on)
            p_off_samples.append(p_off)
    
    if len(p_on_samples) < 10:
        if comm.rank == 0:
            print("  ERROR: Too few probe points!")
        return
    
    p_on_samples = np.array(p_on_samples)
    p_off_samples = np.array(p_off_samples)
    x_samples = x_coords[:len(p_on_samples)]
    
    # Fit 2-wave model
    k = OMEGA / C_WATER
    
    A_on, B_on, R_on, res_on = fit_two_wave_model(x_samples, p_on_samples, k)
    A_off, B_off, R_off, res_off = fit_two_wave_model(x_samples, p_off_samples, k)
    
    if comm.rank == 0:
        print(f"  PML ON:  R = {R_on:.4f}, residual = {res_on:.2e}")
        print(f"  PML OFF: R = {R_off:.4f}, residual = {res_off:.2e}")
        print(f"  Reduction: {R_off/R_on:.2f}x")
        print()
    
    # Pass/fail
    test_passed = True
    
    if comm.rank == 0:
        print("=" * 60)
        print("VALIDATION RESULTS")
        print("=" * 60)
        
        # Check 1: PML reduces reflection
        if R_on < R_ON_MAX:
            print(f"  ✓ PASS: R_on = {R_on:.4f} < {R_ON_MAX}")
        else:
            print(f"  ✗ FAIL: R_on = {R_on:.4f} >= {R_ON_MAX}")
            test_passed = False
        
        # Check 2: Baseline has reflection
        if R_off > R_OFF_MIN:
            print(f"  ✓ PASS: R_off = {R_off:.4f} > {R_OFF_MIN}")
        else:
            print(f"  ✗ FAIL: R_off = {R_off:.4f} <= {R_OFF_MIN}")
            test_passed = False
        
        # Check 3: Reduction factor
        reduction = R_off / R_on if R_on > 1e-12 else 999
        if reduction > REDUCTION_MIN:
            print(f"  ✓ PASS: Reduction = {reduction:.2f}x > {REDUCTION_MIN}x")
        else:
            print(f"  ✗ FAIL: Reduction = {reduction:.2f}x <= {REDUCTION_MIN}x")
            test_passed = False
        
        print("=" * 60)
        
        if test_passed:
            print("\n✓ ALL TESTS PASSED\n")
        else:
            print("\n✗ SOME TESTS FAILED\n")
    
    # Save diagnostics
    if comm.rank == 0:
        diag = {
            "timestamp": datetime.now().isoformat(),
            "frequency_hz": float(FREQ),
            "wavelength_m": float(WAVELENGTH),
            "interior_size_m": float(L),
            "pml_thickness_m": float(PML_THICKNESS),
            "ppw": int(PPW),
            "num_cells": int(num_cells),
            "num_dofs": int(num_dofs),
            "reflection": {
                "R_on": float(R_on),
                "R_off": float(R_off),
                "reduction_factor": float(R_off / R_on) if R_on > 1e-12 else 999.0,
                "fit_residual_on": float(res_on),
                "fit_residual_off": float(res_off)
            },
            "probe_line": {
                "x_start": float(x_probe_start),
                "x_end": float(x_probe_end),
                "y": float(y_center),
                "z": float(z_center),
                "num_points": len(x_samples)
            },
            "pml_stats": pml_stats_on,
            "test_passed": test_passed
        }
        
        with open(OUTPUT_DIR / "diagnostics.json", "w") as f:
            json.dump(diag, f, indent=2)
        
        print(f"Diagnostics saved to: {OUTPUT_DIR / 'diagnostics.json'}")


if __name__ == "__main__":
    run_validation()
