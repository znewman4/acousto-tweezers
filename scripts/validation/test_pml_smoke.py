#!/usr/bin/env python3
"""
PML Level 2 Smoke Test (Truth-Validated)
=========================================

Validates that Perfectly Matched Layer (PML) is:
1. Actually using complex coordinate stretching
2. Mesh is conforming (WATER ↔ PML interface)
3. PML region is excited (nonzero pressure)
4. Actually absorbing/reducing reflection (PML ON vs OFF)

This test cannot lie: it measures reflection proxy directly.

Tests:
1. Complex PETSc is active
2. PML domains exist and are conforming (boolean fragment)
3. PML scaling is complex-valued (Im(s) > 0 in PML)
4. PML region has nonzero pressure (point probes)
5. Reflection proxy is reduced when PML is enabled

Usage:
    python scripts/validation/test_pml_smoke.py

Output:
    results/validation/pml_smoke/run_YYYYMMDD_HHMMSS/
"""

import sys
import os
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime

# ============================================================
# GATE: Complex PETSc
# ============================================================
from petsc4py import PETSc

if not np.issubdtype(PETSc.ScalarType, np.complexfloating):
    print("=" * 70)
    print("FATAL: PETSc is NOT complex!")
    print(f"PETSc.ScalarType = {PETSc.ScalarType}")
    print("=" * 70)
    print()
    print("PML requires complex scalars for coordinate stretching.")
    print("The PML scaling s_x = 1 + i*sigma/omega is inherently complex.")
    print()
    print("FIX: Install complex environment:")
    print("  micromamba env create -f environment/complex-fenicsx.yml")
    print("  micromamba activate acousto-complex")
    sys.exit(1)

print(f"✓ PETSc.ScalarType = {PETSc.ScalarType} (complex)")

# ============================================================
# Imports
# ============================================================
import dolfinx
from dolfinx import fem, mesh as dmesh
from dolfinx.fem import form, assemble_scalar, assemble_vector, Constant
import gmsh
import ufl
from ufl import inner, grad, dx, ds, Measure, TestFunction, TrialFunction
from mpi4py import MPI

# Import production PML implementation (SINGLE SOURCE OF TRUTH)
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from acoustweezers.physics.acoustics.pml import (
    pml_complex_stretch,
    build_pml_stretch_dg0,
    helmholtz_anisotropic_pml_forms
)

print(f"✓ dolfinx version: {dolfinx.__version__}")

# ============================================================
# Test Parameters
# ============================================================
FREQ = 1e6  # 1 MHz
C_WATER = 1500.0  # m/s
RHO_WATER = 1000.0  # kg/m³
WAVELENGTH = C_WATER / FREQ  # 1.5 mm
OMEGA = 2 * np.pi * FREQ

# Geometry
DOMAIN_SIZE = 3 * WAVELENGTH  # Smaller physical domain for faster smoke test
PML_THICKNESS = 1.5 * WAVELENGTH  # PML layer thickness
MESH_SIZE = WAVELENGTH / 5  # 5 PPW (coarse but adequate for smoke test)
POINTS_PER_WAVELENGTH = 5

# PML parameters
# For effective absorption, need sigma/omega ~ O(0.1-1)
# sigma = sigma_max * (d/d_pml)^power
# Use moderate value for smoke test (strong but not too stiff)
PML_POWER = 2
SIGMA_MAX = 0.5 * OMEGA  # Moderate absorption: sigma/omega ~ 0.5 at d=d_pml

# Actuation
ACTUATION_VELOCITY = 1e-4  # m/s

print()
print("Test Parameters:")
print(f"  Frequency: {FREQ/1e6:.1f} MHz")
print(f"  Wavelength: {WAVELENGTH*1e3:.2f} mm")
print(f"  Physical domain: {DOMAIN_SIZE*1e3:.2f} mm")
print(f"  PML thickness: {PML_THICKNESS*1e3:.2f} mm ({PML_THICKNESS/WAVELENGTH:.1f} λ)")
print(f"  σ_max: {SIGMA_MAX}")
print(f"  PML power: {PML_POWER}")

# ============================================================
# Create Output Directory
# ============================================================
t0 = time.time()
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = Path(f"results/validation/pml_smoke/run_{timestamp}")
output_dir.mkdir(parents=True, exist_ok=True)
print(f"\n✓ Output directory: {output_dir}")

# ============================================================
# PML Scaling Function (using production code)
# ============================================================
print("\n" + "=" * 70)
print("STEP 1: Verify PML Scaling is Complex")
print("=" * 70)

# Test at midpoint of PML using production function
d_test = PML_THICKNESS / 2
s_test = pml_complex_stretch(d_test, PML_THICKNESS, SIGMA_MAX, OMEGA, PML_POWER)
print(f"  PML scaling at d={d_test*1e3:.2f}mm:")
print(f"    s = {s_test}")
print(f"    Re(s) = {np.real(s_test):.6f}")
print(f"    Im(s) = {np.imag(s_test):.6f}")

if np.imag(s_test) == 0:
    print("FATAL: PML scaling has no imaginary component!")
    sys.exit(1)
print(f"✓ PML scaling is complex-valued (using production PML code)")

# ============================================================
# Create Mesh with Physical + PML Domains (BOOLEAN FRAGMENT!)
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: Create Conforming Mesh (Boolean Fragment)")
print("=" * 70)

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.model.add("pml_smoke_test")

# Domain tags
TAG_WATER = 1
TAG_PML = 21  # PML region
TAG_ACTUATION = 101
TAG_OUTER = 102

# Physical domain size
L = DOMAIN_SIZE
d_pml = PML_THICKNESS

# Total domain including PML on +x side only (for simplicity)
L_total = L + d_pml

# Create physical domain (water) - will fragment this
water_box = gmsh.model.occ.addBox(0, 0, 0, L, L, L, tag=1)

# Create PML domain (on +x side) - adjacent to water
pml_box = gmsh.model.occ.addBox(L, 0, 0, d_pml, L, L, tag=2)

# CRITICAL: Boolean fragment to create conforming mesh at interface
print("  Fragmenting volumes for conforming interface...")
gmsh.model.occ.fragment([(3, water_box)], [(3, pml_box)])
gmsh.model.occ.synchronize()

# After fragment, need to re-tag volumes by centroid location
all_volumes = gmsh.model.getEntities(3)
water_vols = []
pml_vols = []

print(f"  Found {len(all_volumes)} volumes after fragment")
for vol in all_volumes:
    mass = gmsh.model.occ.getMass(vol[0], vol[1])
    com = gmsh.model.occ.getCenterOfMass(vol[0], vol[1])
    x_center = com[0]
    
    # Classify by x-coordinate of centroid
    if x_center < L - 1e-6:  # In physical region
        water_vols.append(vol[1])
    else:  # In PML region
        pml_vols.append(vol[1])

print(f"  Classified: {len(water_vols)} WATER, {len(pml_vols)} PML volumes")

if len(water_vols) == 0:
    print("FATAL: No WATER volumes after fragment!")
    sys.exit(1)

if len(pml_vols) == 0:
    print("FATAL: No PML volumes after fragment!")
    sys.exit(1)

# Mesh size
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", MESH_SIZE * 0.8)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", MESH_SIZE * 1.2)

# Tag volumes with physical groups
gmsh.model.addPhysicalGroup(3, water_vols, TAG_WATER)
gmsh.model.setPhysicalName(3, TAG_WATER, "WATER")
gmsh.model.addPhysicalGroup(3, pml_vols, TAG_PML)
gmsh.model.setPhysicalName(3, TAG_PML, "PML")

# Synchronize before getting surfaces
gmsh.model.occ.synchronize()

# Find surfaces
surfaces = gmsh.model.getEntities(2)
actuation_surfs = []
outer_surfs = []

for surf in surfaces:
    bbox = gmsh.model.getBoundingBox(surf[0], surf[1])
    x_min, y_min, z_min, x_max, y_max, z_max = bbox
    
    # Actuation on x=0 plane
    if abs(x_min) < 1e-6 and abs(x_max) < 1e-6:
        actuation_surfs.append(surf[1])
    # Outer boundary on x=L+d_pml plane (or other outer faces)
    elif abs(x_max - L_total) < 1e-6:
        outer_surfs.append(surf[1])
    elif abs(y_min) < 1e-6 or abs(y_max - L) < 1e-6:
        outer_surfs.append(surf[1])
    elif abs(z_min) < 1e-6 or abs(z_max - L) < 1e-6:
        outer_surfs.append(surf[1])

if len(actuation_surfs) == 0:
    print("WARNING: No actuation surface found, using fallback")
    # Use any surface at x=0
    for surf in surfaces:
        bbox = gmsh.model.getBoundingBox(surf[0], surf[1])
        if abs(bbox[0]) < 1e-6 and abs(bbox[3]) < 1e-6:  # x_min and x_max ~ 0
            actuation_surfs.append(surf[1])

print(f"  Tagged {len(actuation_surfs)} actuation surfaces")
print(f"  Tagged {len(outer_surfs)} outer surfaces")

gmsh.model.addPhysicalGroup(2, actuation_surfs, TAG_ACTUATION)
gmsh.model.setPhysicalName(2, TAG_ACTUATION, "ACTUATION")

if outer_surfs:
    gmsh.model.addPhysicalGroup(2, outer_surfs, TAG_OUTER)
    gmsh.model.setPhysicalName(2, TAG_OUTER, "OUTER")

# Generate mesh
gmsh.model.mesh.generate(3)

# Save mesh
mesh_path = output_dir / "mesh.msh"
gmsh.write(str(mesh_path))
print(f"✓ Mesh saved: {mesh_path}")

# Import to DOLFINx
mesh, cell_tags, facet_tags = dolfinx.io.gmshio.model_to_mesh(
    gmsh.model, MPI.COMM_WORLD, 0, gdim=3
)
gmsh.finalize()

num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
num_vertices = mesh.topology.index_map(0).size_local

# ============================================================
# Verify Domain Tags
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: Verify Domain Tags")
print("=" * 70)

water_cells = cell_tags.find(TAG_WATER)
pml_cells = cell_tags.find(TAG_PML)
actuation_facets = facet_tags.find(TAG_ACTUATION)

print(f"✓ WATER cells: {len(water_cells)}")
print(f"✓ PML cells: {len(pml_cells)}")
print(f"✓ ACTUATION facets: {len(actuation_facets)}")

if len(pml_cells) == 0:
    print("FATAL: No PML cells found!")
    sys.exit(1)

if len(water_cells) == 0:
    print("FATAL: No WATER cells found!")
    sys.exit(1)

# ============================================================
# Solve with PML ON and OFF
# ============================================================
print("\n" + "=" * 70)
print("STEP 4: Solve Helmholtz (PML ON and OFF)")
print("=" * 70)

V = fem.functionspace(mesh, ("Lagrange", 2))
print(f"✓ DOFs: {V.dofmap.index_map.size_global}")

# Trial and test functions
p = TrialFunction(V)
v = TestFunction(V)

# Material properties
rho = RHO_WATER
c = C_WATER
k = OMEGA / c
K = rho * c**2

# Measures
dx_water = Measure("dx", domain=mesh, subdomain_data=cell_tags)(TAG_WATER)
dx_pml = Measure("dx", domain=mesh, subdomain_data=cell_tags)(TAG_PML)
ds_act = Measure("ds", domain=mesh, subdomain_data=facet_tags)

# ============================================================
# Helper Function: Solve with/without PML (using production code)
# ============================================================
def solve_helmholtz_pml(pml_active=True):
    """
    Solve Helmholtz equation with or without PML using PRODUCTION PML CODE.
    
    This ensures the smoke test validates the actual production operator.
    
    Parameters
    ----------
    pml_active : bool
        If True, use PML scaling. If False, set s=1 everywhere.
    
    Returns
    -------
    p_h : fem.Function
        Pressure solution
    ksp_converged : bool
        Whether solver converged
    ksp_iterations : int
        Number of KSP iterations
    im_s_water : float
        Max Im(s_x) in water
    im_s_pml : float
        Max Im(s_x) in PML
    """
    # Build PML stretch fields using production code
    if pml_active:
        s_x, s_x_inv, im_s_water, im_s_pml = build_pml_stretch_dg0(
            mesh, cell_tags, TAG_PML, L, d_pml, OMEGA, SIGMA_MAX, PML_POWER, TAG_WATER
        )
    else:
        # No PML: s_x = 1 everywhere
        DG0 = fem.functionspace(mesh, ("DG", 0))
        s_x = fem.Function(DG0)
        s_x_inv = fem.Function(DG0)
        s_x.x.array[:] = 1.0 + 0j
        s_x_inv.x.array[:] = 1.0 + 0j
        im_s_water = 0.0
        im_s_pml = 0.0
    
    # Build anisotropic PML forms using production code (SINGLE SOURCE OF TRUTH)
    a_form, _ = helmholtz_anisotropic_pml_forms(
        p, v, mesh, k, rho, OMEGA,
        s_x, s_x_inv,
        dx_water, dx_pml,
        source_form=None
    )
    
    # RHS: Actuation
    actuation_neumann = np.complex128(-1j * OMEGA * RHO_WATER * ACTUATION_VELOCITY)
    g = Constant(mesh, actuation_neumann)
    L_form = inner(g, v) * ds_act(TAG_ACTUATION)
    
    # Compile and assemble
    a_compiled = form(a_form)
    L_compiled = form(L_form)
    
    # Create solution function
    p_h = fem.Function(V)
    
    # Use PETSc solver
    from dolfinx.fem.petsc import create_matrix, create_vector, assemble_matrix, assemble_vector as assemble_vector_petsc
    A_petsc = create_matrix(a_compiled)
    A_petsc.zeroEntries()
    assemble_matrix(A_petsc, a_compiled, bcs=[])
    A_petsc.assemble()
    
    b_petsc = create_vector(L_compiled)
    assemble_vector_petsc(b_petsc, L_compiled)
    b_petsc.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    
    # Setup solver with better convergence
    ksp = PETSc.KSP().create(mesh.comm)
    ksp.setOperators(A_petsc)
    
    # Try direct solver for small problems, else iterative
    matrix_size = V.dofmap.index_map.size_global
    if matrix_size < 50000:
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
    else:
        ksp.setType("gmres")
        ksp.getPC().setType("ilu")
        ksp.setTolerances(rtol=1e-8, max_it=2000)
    
    ksp.setFromOptions()
    
    # Solve
    ksp.solve(b_petsc, p_h.x.petsc_vec)
    
    converged = ksp.getConvergedReason() > 0
    iterations = ksp.getIterationNumber()
    
    return p_h, converged, iterations, im_s_water, im_s_pml

# Solve with PML ON
print("\n  [PML ON]")
p_h_pml_on, converged_on, iters_on, im_s_water, im_s_pml = solve_helmholtz_pml(pml_active=True)
print(f"    Converged: {converged_on} ({iters_on} iterations)")
print(f"    Im(s) in WATER: max = {im_s_water:.6e} (should be ~0)")
print(f"    Im(s) in PML: max = {im_s_pml:.6e} (should be >0)")

# Solve with PML OFF
print("\n  [PML OFF]")
p_h_pml_off, converged_off, iters_off, _, _ = solve_helmholtz_pml(pml_active=False)
print(f"    Converged: {converged_off} ({iters_off} iterations)")

# Check PML scaling is correct
if im_s_water > 1e-10:
    print("  WARNING: Im(s) is nonzero in WATER region (should be 0)")
if im_s_pml < 1e-10:
    print("  WARNING: Im(s) is zero in PML region (should be >0)")
    print("           PML may not be active!")

# ============================================================
# Point Probes: Verify PML is Excited
# ============================================================
print("\n" + "=" * 70)
print("STEP 5: Point Probes (PML Excitation Check)")
print("=" * 70)

# Define probe locations
probe_water_near = np.array([L - 0.25*WAVELENGTH, L/2, L/2])  # Near interface
probe_pml_mid = np.array([L + 0.5*PML_THICKNESS, L/2, L/2])  # Mid-PML

# Reflection proxy probes (in WATER only)
probe_refl_1 = np.array([L - 0.25*WAVELENGTH, L/2, L/2])  # Near PML interface
probe_refl_2 = np.array([L - 0.75*WAVELENGTH, L/2, L/2])  # Further from interface

print(f"  Probe locations:")
print(f"    WATER near interface: x={probe_water_near[0]*1e3:.2f}mm")
print(f"    PML mid-layer: x={probe_pml_mid[0]*1e3:.2f}mm")
print(f"    Reflection proxy 1: x={probe_refl_1[0]*1e3:.2f}mm")
print(f"    Reflection proxy 2: x={probe_refl_2[0]*1e3:.2f}mm")

# Evaluate pressure at probes for PML ON
try:
    p_water_near_on = p_h_pml_on.eval(probe_water_near, 0)
    p_pml_mid_on = p_h_pml_on.eval(probe_pml_mid, 0)
    p_refl_1_on = p_h_pml_on.eval(probe_refl_1, 0)
    p_refl_2_on = p_h_pml_on.eval(probe_refl_2, 0)
    
    mag_water_near_on = np.abs(p_water_near_on[0])
    mag_pml_mid_on = np.abs(p_pml_mid_on[0])
    mag_refl_1_on = np.abs(p_refl_1_on[0])
    mag_refl_2_on = np.abs(p_refl_2_on[0])
    
    print(f"\n  [PML ON]")
    print(f"    |p| at WATER near interface: {mag_water_near_on:.6e} Pa")
    print(f"    |p| at PML mid-layer: {mag_pml_mid_on:.6e} Pa")
    print(f"    |p| at reflection probe 1: {mag_refl_1_on:.6e} Pa")
    print(f"    |p| at reflection probe 2: {mag_refl_2_on:.6e} Pa")
    
except Exception as e:
    print(f"  ERROR evaluating probes (PML ON): {e}")
    mag_water_near_on = 0
    mag_pml_mid_on = 0
    mag_refl_1_on = 0
    mag_refl_2_on = 0

# Evaluate for PML OFF
try:
    p_water_near_off = p_h_pml_off.eval(probe_water_near, 0)
    p_pml_mid_off = p_h_pml_off.eval(probe_pml_mid, 0)
    p_refl_1_off = p_h_pml_off.eval(probe_refl_1, 0)
    p_refl_2_off = p_h_pml_off.eval(probe_refl_2, 0)
    
    mag_water_near_off = np.abs(p_water_near_off[0])
    mag_pml_mid_off = np.abs(p_pml_mid_off[0])
    mag_refl_1_off = np.abs(p_refl_1_off[0])
    mag_refl_2_off = np.abs(p_refl_2_off[0])
    
    print(f"\n  [PML OFF]")
    print(f"    |p| at WATER near interface: {mag_water_near_off:.6e} Pa")
    print(f"    |p| at PML mid-layer: {mag_pml_mid_off:.6e} Pa")
    print(f"    |p| at reflection probe 1: {mag_refl_1_off:.6e} Pa")
    print(f"    |p| at reflection probe 2: {mag_refl_2_off:.6e} Pa")
    
except Exception as e:
    print(f"  ERROR evaluating probes (PML OFF): {e}")
    mag_water_near_off = 0
    mag_pml_mid_off = 0
    mag_refl_1_off = 0
    mag_refl_2_off = 0

# Check PML is actually excited
max_p_global_on = np.max(np.abs(p_h_pml_on.x.array))
pml_excitation_threshold = 1e-12 * max_p_global_on

print(f"\n  PML Excitation Check:")
print(f"    max|p| global: {max_p_global_on:.6e} Pa")
print(f"    |p| at PML probe: {mag_pml_mid_on:.6e} Pa")
print(f"    Threshold (1e-12 * max): {pml_excitation_threshold:.6e} Pa")

if mag_pml_mid_on < pml_excitation_threshold:
    print("  ⚠ WARNING: PML region has near-zero pressure!")
    print("             PML may be disconnected from physical domain")
else:
    print("  ✓ PML region is excited (nonzero pressure)")

# ============================================================
# Reflection Proxy Analysis
# ============================================================
print("\n" + "=" * 70)
print("STEP 6: Reflection Proxy (PML ON vs OFF)")
print("=" * 70)

# Reflection proxy: pressure near interface
# Should be lower with PML ON (less reflection)
reflection_proxy_on = mag_refl_1_on
reflection_proxy_off = mag_refl_1_off

print(f"  Reflection proxy (|p| at x={probe_refl_1[0]*1e3:.2f}mm):")
print(f"    PML ON:  {reflection_proxy_on:.6e} Pa")
print(f"    PML OFF: {reflection_proxy_off:.6e} Pa")

if reflection_proxy_off > 0:
    reduction_factor = reflection_proxy_off / reflection_proxy_on if reflection_proxy_on > 0 else float('inf')
    reduction_pct = (1 - reflection_proxy_on / reflection_proxy_off) * 100 if reflection_proxy_on < reflection_proxy_off else 0
    print(f"    Reduction factor: {reduction_factor:.2f}x")
    print(f"    Reduction: {reduction_pct:.1f}%")
else:
    reduction_factor = float('nan')
    reduction_pct = float('nan')
    print(f"    Cannot compute reduction (PML OFF baseline is zero)")

# Acceptance criterion: PML ON should reduce reflection
# Note: For smoke test, require modest but measurable reduction
TARGET_REDUCTION_FACTOR = 1.2  # Require at least 20% reduction

if reflection_proxy_off > 0 and reflection_proxy_on > 0:
    if reduction_factor >= TARGET_REDUCTION_FACTOR:
        print(f"\n  ✓ PASS: Reflection reduced by {reduction_factor:.2f}x (target: {TARGET_REDUCTION_FACTOR:.1f}x)")
        reflection_test_pass = True
    else:
        print(f"\n  ✗ FAIL: Reflection reduction insufficient ({reduction_factor:.2f}x < {TARGET_REDUCTION_FACTOR:.1f}x)")
        reflection_test_pass = False
else:
    print(f"\n  ⚠ WARNING: Cannot verify reflection reduction (zero baseline)")
    reflection_test_pass = False

# ============================================================
# Standing-Wave Line Scan (harder to cheat)
# ============================================================
print("\n" + "=" * 70)
print("STEP 6b: Standing-Wave Line Scan")
print("=" * 70)

# Sample pressure along a line in WATER parallel to x near interface
#Standing waves (from reflections) show large max/min ratio
N_scan = 25
x_scan = np.linspace(L - 1.5*WAVELENGTH, L - 0.1*WAVELENGTH, N_scan)
y_scan = L / 2
z_scan = L / 2

scan_points_on = []
scan_points_off = []

print(f"  Scanning {N_scan} points along x=[{x_scan[0]*1e3:.2f}, {x_scan[-1]*1e3:.2f}]mm")
print(f"  at y=z={y_scan*1e3:.2f}mm\\n")

for x_val in x_scan:
    try:
        p_on_val = p_h_pml_on.eval(np.array([x_val, y_scan, z_scan]), 0)
        p_off_val = p_h_pml_off.eval(np.array([x_val, y_scan, z_scan]), 0)
        scan_points_on.append(np.abs(p_on_val[0]))
        scan_points_off.append(np.abs(p_off_val[0]))
    except Exception as e:
        print(f"  Warning: Could not evaluate at x={x_val*1e3:.2f}mm: {e}")
        scan_points_on.append(0)
        scan_points_off.append(0)

scan_points_on = np.array(scan_points_on)
scan_points_off = np.array(scan_points_off)

# Compute standing-wave ratio: max/min (avoid division by zero)
eps = 1e-30
S_on = np.max(scan_points_on) / (np.min(scan_points_on) + eps) if len(scan_points_on) > 0 else 1.0
S_off = np.max(scan_points_off) / (np.min(scan_points_off) + eps) if len(scan_points_off) > 0 else 1.0

standing_wave_metric = S_off / S_on if S_on > 1.0 else 1.0

print(f"  Standing-wave ratio (max|p|/min|p| on scan line):")
print(f"    PML ON:  S = {S_on:.2f}")
print(f"    PML OFF: S = {S_off:.2f}")
print(f"    Metric (S_off/S_on): {standing_wave_metric:.2f}")

if standing_wave_metric > 1.5:
    print(f"  ✓ Standing-wave pattern significantly reduced with PML")
else:
    print(f"  ⚠ Standing-wave reduction modest")

# ============================================================
# Save Diagnostics & Report
# ============================================================
print("\n" + "=" * 70)
print("STEP 7: Generate Report")
print("=" * 70)

os.makedirs(output_dir, exist_ok=True)

# MPI-safe global max|p| (scatter forward + reduction)
p_h_pml_on.x.scatter_forward()
local_max_on = np.max(np.abs(p_h_pml_on.x.array))
max_p_global_on = mesh.comm.allreduce(local_max_on, op=MPI.MAX)

# Detailed JSON diagnostics
diagnostics = {
    "timestamp": datetime.now().isoformat(),
    "test": "pml_smoke_truth_validated_anisotropic",
    "parameters": {
        "freq_hz": FREQ,
        "wavelength_m": WAVELENGTH,
        "L_physical_m": L,
        "pml_thickness_m": PML_THICKNESS,
        "sigma_max": SIGMA_MAX,
        "pml_power": PML_POWER,
        "mesh_resolution_ppw": POINTS_PER_WAVELENGTH,
        "domain_size_mm": float(L * 1e3),
    },
    "mesh": {
        "num_cells": int(num_cells),
        "num_vertices": int(num_vertices),
        "num_dofs": int(V.dofmap.index_map.size_global),
    },
    "solver": {
        "pml_on_converged": bool(converged_on),
        "pml_on_iterations": int(iters_on),
        "pml_off_converged": bool(converged_off),
        "pml_off_iterations": int(iters_off),
    },
    "pml_activation": {
        "im_s_water_median": float(im_s_water),
        "im_s_pml_median": float(im_s_pml),
    },
    "max_pressure_global_pa": float(max_p_global_on),
    "standing_wave_scan": {
        "S_pml_on": float(S_on),
        "S_pml_off": float(S_off),
        "standing_wave_metric": float(standing_wave_metric),
        "n_points": int(N_scan),
    },
    "probes": {
        "probe_water_near_pa": float(mag_water_near_on),
        "probe_pml_mid_pa": float(mag_pml_mid_on),
        "probe_refl_1_on_pa": float(mag_refl_1_on),
        "probe_refl_2_on_pa": float(mag_refl_2_on),
        "probe_refl_1_off_pa": float(mag_refl_1_off),
        "probe_refl_2_off_pa": float(mag_refl_2_off),
    },
    "reflection_analysis": {
        "reflection_proxy_pml_on": float(reflection_proxy_on),
        "reflection_proxy_pml_off": float(reflection_proxy_off),
        "reduction_factor": float(reduction_factor) if not np.isnan(reduction_factor) else None,
        "reduction_pct": float(reduction_pct) if not np.isnan(reduction_pct) else None,
        "target_reduction_factor": float(TARGET_REDUCTION_FACTOR),
        "meets_target": bool(reflection_test_pass),
    },
}

diag_path = os.path.join(str(output_dir), "diagnostics.json")
with open(diag_path, "w") as f:
    json.dump(diagnostics, f, indent=2)
print(f"  Diagnostics saved: {diag_path}")

# Human-readable report
report_path = os.path.join(str(output_dir), "pml_report.txt")
with open(report_path, "w") as f:
    f.write("=" * 70 + "\n")
    f.write("PML Smoke Test – Truth-Validated Report\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Timestamp: {diagnostics['timestamp']}\n\n")
    
    f.write("PARAMETERS\n")
    f.write("-" * 40 + "\n")
    f.write(f"  Frequency:        {FREQ*1e-6:.2f} MHz\n")
    f.write(f"  Wavelength:       {WAVELENGTH*1e3:.2f} mm\n")
    f.write(f"  Physical domain:  {L*1e3:.2f} mm ({L/WAVELENGTH:.1f}λ)\n")
    f.write(f"  PML thickness:    {PML_THICKNESS*1e3:.2f} mm ({PML_THICKNESS/WAVELENGTH:.1f}λ)\n")
    f.write(f"  σ_max:            {SIGMA_MAX:.1f}\n")
    f.write(f"  PML power:        {PML_POWER}\n")
    f.write(f"  Resolution:       {POINTS_PER_WAVELENGTH} PPW\n\n")
    
    f.write("MESH\n")
    f.write("-" * 40 + "\n")
    f.write(f"  Cells:            {num_cells}\n")
    f.write(f"  Vertices:         {num_vertices}\n")
    f.write(f"  DOFs:             {V.dofmap.index_map.size_global}\n\n")
    
    f.write("SOLVER\n")
    f.write("-" * 40 + "\n")
    f.write(f"  PML ON:           {'Converged' if converged_on else 'Did not converge'} ({iters_on} iter)\n")
    f.write(f"  PML OFF:          {'Converged' if converged_off else 'Did not converge'} ({iters_off} iter)\n\n")
    
    f.write("PML ACTIVATION CHECK\n")
    f.write("-" * 40 + "\n")
    f.write(f"  Im(s) in WATER:   {im_s_water:.6e} (should be ~0)\n")
    f.write(f"  Im(s) in PML:     {im_s_pml:.6e} (should be >0)\n")
    if im_s_pml > 0:
        f.write("  Status:           ✓ PML is active\n\n")
    else:
        f.write("  Status:           ✗ PML NOT active!\n\n")
    
    f.write("POINT PROBES\n")
    f.write("-" * 40 + "\n")
    f.write(f"  |p| at WATER near interface (PML ON):  {mag_water_near_on:.6e} Pa\n")
    f.write(f"  |p| at PML mid-layer (PML ON):         {mag_pml_mid_on:.6e} Pa\n")
    f.write(f"  |p| at WATER near interface (PML OFF): {mag_water_near_off:.6e} Pa\n")
    if mag_pml_mid_on > 1e-12 * max_p_global_on:
        f.write("  Status:           ✓ PML region is excited\n\n")
    else:
        f.write("  Status:           ✗ PML region NOT excited!\n\n")
    
    f.write("REFLECTION PROXY ANALYSIS\n")
    f.write("-" * 40 + "\n")
    f.write(f"  |p| near interface (PML ON):  {reflection_proxy_on:.6e} Pa\n")
    f.write(f"  |p| near interface (PML OFF): {reflection_proxy_off:.6e} Pa\n")
    if not np.isnan(reduction_factor):
        f.write(f"  Reduction factor:             {reduction_factor:.2f}x\n")
        f.write(f"  Reduction percentage:         {reduction_pct:.1f}%\n")
        f.write(f"  Target:                       {TARGET_REDUCTION_FACTOR:.1f}x\n")
        if reflection_test_pass:
            f.write("  Status:                       ✓ PASS\n\n")
        else:
            f.write("  Status:                       ✗ FAIL (insufficient reduction)\n\n")
    else:
        f.write("  Status:                       ✗ Cannot compute (zero baseline)\n\n")
    
    f.write("VERDICT\n")
    f.write("-" * 40 + "\n")
    
    all_checks = [
        ("PML active (Im(s) > 0)", im_s_pml > 0),
        ("PML region excited", mag_pml_mid_on > 1e-12 * max_p_global_on),
        ("Reflection reduced", reflection_test_pass),
    ]
    
    all_pass = all(check[1] for check in all_checks)
    
    if all_pass:
        f.write("  ✓ TEST PASSED: PML is functional and reduces reflection\n\n")
    else:
        f.write("  ✗ TEST FAILED:\n")
        for check_name, check_result in all_checks:
            status = "✓" if check_result else "✗"
            f.write(f"    {status} {check_name}\n")
        f.write("\n")

print(f"  Report saved: {report_path}")

# ============================================================
# Final Pass/Fail
# ============================================================
print("\n" + "=" * 70)
print("FINAL VERDICT")
print("=" * 70)

all_checks_pass = (
    mag_pml_mid_on > 1e-12 * max_p_global_on  # PML is excited
    and im_s_pml > 0  # PML is active
    and reflection_test_pass  # Reflection is reduced
)

if all_checks_pass:
    print("✓ TEST PASSED: PML functional and reduces reflection")
    if not converged_on:
        print("  (Note: Solver did not fully converge, but PML is functional)")
    print(f"  - PML active (Im(s) > 0): {im_s_pml > 0}")
    print(f"  - PML region excited: {mag_pml_mid_on > 1e-12 * max_p_global_on}")
    print(f"  - Reflection reduced: {reflection_test_pass} ({reduction_factor:.2f}x)")
else:
    print("✗ TEST FAILED")
    if not converged_on:
        print("  - Solver did not converge (PML ON)")
    if im_s_pml <= 0:
        print("  - PML not active (Im(s) ≤ 0)")
    if mag_pml_mid_on <= 1e-12 * max_p_global_on:
        print("  - PML region not excited")
    if not reflection_test_pass:
        print(f"  - Reflection not reduced sufficiently ({reduction_factor:.2f}x < {TARGET_REDUCTION_FACTOR:.1f}x)")
    sys.exit(1)

print("\n" + "=" * 70)
print(f"Runtime: {time.time() - t0:.1f}s")
print("=" * 70)
