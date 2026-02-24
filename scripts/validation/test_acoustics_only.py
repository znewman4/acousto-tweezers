#!/usr/bin/env python3
"""
Quick test for ACOUSTICS_ONLY physics level using the full solver stack.

This tests:
1. Mesh generation with proper domain tagging  
2. AcousticSolver with complex backend
3. Basic diagnostics

Usage:
    micromamba activate acousto-complex
    python scripts/validation/test_acoustics_only.py

Author: Acousto-Tweezers Project
Date: January 2026
"""

import sys
from pathlib import Path
import numpy as np
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def run_acoustics_only_test():
    """
    Test ACOUSTICS_ONLY level with simple box geometry.
    """
    print("=" * 60)
    print("ACOUSTICS_ONLY LEVEL TEST")
    print("=" * 60)
    print()
    
    # Check complex backend first
    from petsc4py import PETSc
    print(f"PETSc.ScalarType = {PETSc.ScalarType}")
    if "complex" not in str(PETSc.ScalarType).lower():
        print("[FAIL] PETSc is not complex!")
        return 1
    print("[PASS] Complex backend verified")
    print()
    
    # Import modules
    print("[1] Importing modules...")
    from mpi4py import MPI
    from dolfinx import mesh, fem
    from dolfinx.io import gmshio
    import gmsh
    import ufl
    
    from acoustweezers.legacy.src_archive.tweezers.fenicsx.config import FEMConfig, PhysicsLevel, GeometryConfig, PhysicsConfig
    from acoustweezers.legacy.src_archive.tweezers.fenicsx.domains import Domain, Interface
    from acoustweezers.legacy.src_archive.tweezers.fenicsx.materials import MaterialDatabase
    from acoustweezers.legacy.src_archive.tweezers.fenicsx.acoustics import AcousticSolver, AcousticField
    
    print("    Imports successful")
    print()
    
    # Create configuration
    print("[2] Creating configuration...")
    config = FEMConfig(
        physics_level=PhysicsLevel.ACOUSTICS_ONLY,
        geometry=GeometryConfig(
            dish_diameter=12e-3,        # 12mm diameter
            dish_wall_thickness=1e-3,   # 1mm walls
            dish_bottom_thickness=0.5e-3,  # 0.5mm plate
            water_depth=3e-3,           # 3mm water depth
            air_height=2e-3,
            bath_depth=2e-3,
            pml_thickness=1e-3,
            elements_per_wavelength=6,
            min_element_size=50e-6,
            max_element_size=500e-6,
        ),
        physics=PhysicsConfig(
            frequency=1e6,  # 1 MHz
        ),
    )
    
    omega = config.physics.omega
    print(f"    Frequency: {config.physics.frequency / 1e6:.1f} MHz")
    print(f"    Angular freq: {omega:.2e} rad/s")
    print()
    
    # Create simple mesh with Gmsh
    print("[3] Creating mesh with Gmsh...")
    gmsh.initialize()
    gmsh.model.add("test_box")
    
    # Simple box for water domain
    L = config.geometry.dish_inner_radius * 2
    H = config.geometry.water_depth
    box = gmsh.model.occ.addBox(-L/2, -L/2, 0, L, L, H)
    gmsh.model.occ.synchronize()
    
    # Tag as water
    pg_water = gmsh.model.addPhysicalGroup(3, [box], Domain.WATER.gmsh_tag)
    gmsh.model.setPhysicalName(3, pg_water, "WATER")
    
    # Tag bottom surface for actuation (WATER_PLATE = 111)
    # Need to find surface at z=0 (bottom of water box)
    bottom_surfs = []
    all_surfs = gmsh.model.getEntities(dim=2)
    print(f"    Total surfaces: {len(all_surfs)}")
    
    tol = 1e-6
    for dim, tag in all_surfs:
        bbox = gmsh.model.getBoundingBox(2, tag)
        z_min, z_max = bbox[2], bbox[5]
        # Surface is at z=0 if both zmin and zmax are near 0
        if abs(z_min) < tol and abs(z_max) < tol:
            bottom_surfs.append(tag)
            print(f"    Found bottom surface: tag={tag}, z_range=[{z_min:.2e}, {z_max:.2e}]")
        else:
            print(f"    Surface {tag}: z_range=[{z_min:.6f}, {z_max:.6f}]")
    
    if bottom_surfs:
        pg_act = gmsh.model.addPhysicalGroup(2, bottom_surfs, Interface.WATER_PLATE.gmsh_tag)
        gmsh.model.setPhysicalName(2, pg_act, "WATER_PLATE")
        print(f"    Tagged {len(bottom_surfs)} bottom surfaces as WATER_PLATE (tag={Interface.WATER_PLATE.gmsh_tag})")
    else:
        print("    WARNING: No bottom surfaces found!")
        # Debug: print all surface z-ranges
        print("    Trying to find surfaces near z=0:")
        for dim, tag in all_surfs:
            bbox = gmsh.model.getBoundingBox(2, tag)
            if bbox[2] < 1e-4 and bbox[5] < 1e-4:  # More lenient check
                print(f"    Surface {tag} is near z=0: bbox z = [{bbox[2]}, {bbox[5]}]")
                bottom_surfs.append(tag)
        if bottom_surfs:
            pg_act = gmsh.model.addPhysicalGroup(2, bottom_surfs, Interface.WATER_PLATE.gmsh_tag)
            gmsh.model.setPhysicalName(2, pg_act, "WATER_PLATE")
    
    # Find outer surfaces for BC
    outer_surfs = []
    for dim, tag in all_surfs:
        if tag not in bottom_surfs:
            outer_surfs.append(tag)
    if outer_surfs:
        pg_outer = gmsh.model.addPhysicalGroup(2, outer_surfs, Interface.PML_OUTER.gmsh_tag)
        gmsh.model.setPhysicalName(2, pg_outer, "PML_OUTER")
        print(f"    Tagged {len(outer_surfs)} outer surfaces as PML_OUTER (tag={Interface.PML_OUTER.gmsh_tag})")
    
    # Mesh
    lc = 1e-3  # 1 mm (coarser for testing)
    gmsh.option.setNumber("Mesh.MeshSizeMax", lc)
    gmsh.model.mesh.generate(3)
    
    # Import to DOLFINx
    mesh_dolfinx, cell_tags, facet_tags = gmshio.model_to_mesh(
        gmsh.model, MPI.COMM_WORLD, rank=0, gdim=3
    )
    gmsh.finalize()
    
    num_cells = mesh_dolfinx.topology.index_map(3).size_global
    print(f"    Mesh cells: {num_cells}")
    print(f"    Cell tags: {np.unique(cell_tags.values)}")
    print(f"    Facet tags: {np.unique(facet_tags.values)}")
    print()
    
    # Create materials
    print("[4] Creating material database...")
    materials = MaterialDatabase()
    print(f"    Water: ρ={materials.water.density}, c={materials.water.sound_speed}")
    print()
    
    # Create solver
    print("[5] Creating AcousticSolver...")
    solver = AcousticSolver(
        config=config,
        mesh=mesh_dolfinx,
        cell_tags=cell_tags,
        facet_tags=facet_tags,
        materials=materials,
    )
    print(f"    DOFs: {solver.V.dofmap.index_map.size_global}")
    print()
    
    # Solve with actuation
    print("[6] Solving with actuation...")
    actuation_amplitude = 1e-9  # 1 nm displacement -> velocity
    result = solver.solve_with_actuation(
        actuation_amplitude=actuation_amplitude,
        actuation_phase=0.0,
    )
    
    # Analyze results
    print("\n[7] Analyzing results...")
    p_array = result.p
    print(f"    Solution dtype: {p_array.dtype}")
    
    p_abs = np.abs(p_array)
    p_real = np.real(p_array)
    p_imag = np.imag(p_array)
    
    print(f"    max|p|: {result.max_pressure:.4e} Pa")
    print(f"    mean|p|: {result.mean_pressure:.4e} Pa")
    print(f"    rms|p|: {result.rms_pressure:.4e} Pa")
    print(f"    Re(p) range: [{np.min(p_real):.4e}, {np.max(p_real):.4e}] Pa")
    print(f"    Im(p) range: [{np.min(p_imag):.4e}, {np.max(p_imag):.4e}] Pa")
    
    # Validation
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    
    passed = True
    
    # Test 1: Non-zero solution
    if result.max_pressure > 1e-10:
        print("[PASS] Pressure field is non-zero")
    else:
        print("[FAIL] Pressure field is zero or near-zero")
        passed = False
    
    # Test 2: Complex-valued
    if np.iscomplexobj(p_array):
        print("[PASS] Pressure is complex-valued")
    else:
        print("[FAIL] Pressure is not complex-valued")
        passed = False
    
    # Test 3: Has imaginary part
    if np.max(np.abs(p_imag)) > 1e-15 * np.max(p_abs):
        print("[PASS] Has non-trivial imaginary part")
    else:
        print("[WARN] Imaginary part very small")
    
    # Test 4: Reasonable dynamic range
    if result.max_pressure / (result.mean_pressure + 1e-30) < 1e6:
        print("[PASS] Dynamic range is reasonable")
    else:
        print("[WARN] Large dynamic range (may indicate issues)")
    
    print()
    if passed:
        print("✓ ACOUSTICS_ONLY level test passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(run_acoustics_only_test())
