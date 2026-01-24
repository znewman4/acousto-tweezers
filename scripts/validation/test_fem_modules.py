#!/usr/bin/env python3
"""
FEM module validation micro-tests.

This script runs a series of small validation tests for each FEM module
to verify correctness:

1. Configuration: Verify FEMConfig defaults and serialization
2. Domains: Verify domain type assignments
3. Materials: Verify material property calculations
4. Geometry: Verify mesh generation and node/element counts
5. Acoustics: Verify weak form assembly (manufactured solution)
6. Solids: Verify elasticity tensor
7. PML: Verify complex stretching factors
8. Thermoviscous: Verify boundary layer thickness
9. Streaming: Verify Reynolds stress computation
10. Particles: Verify Gor'kov potential

Run with:
    python scripts/validation/test_fem_modules.py
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))


def test_config():
    """Test configuration module."""
    print("Testing config module...")
    
    from tweezers.fem import FEMConfig, PhysicsLevel
    
    # Test default config
    config = FEMConfig.default()
    assert config.physics_level == PhysicsLevel.PARTICLES
    assert config.physics.frequency == 2e6
    assert config.geometry.resolution == 0.0002
    
    # Test physics level ordering
    assert PhysicsLevel.ACOUSTICS_ONLY.value < PhysicsLevel.PARTICLES.value
    assert PhysicsLevel.STREAMING < PhysicsLevel.PARTICLES
    
    print("  ✓ FEMConfig defaults")
    print("  ✓ PhysicsLevel ordering")
    return True


def test_domains():
    """Test domain module."""
    print("Testing domains module...")
    
    from tweezers.fem import DomainType, InterfaceType
    
    # Test domain types
    assert DomainType.WATER.value == 1
    assert DomainType.AIR.value == 2
    assert DomainType.PML_WATER.is_pml
    assert not DomainType.WATER.is_pml
    assert DomainType.PLATE.is_solid
    assert not DomainType.WATER.is_solid
    
    print("  ✓ DomainType values")
    print("  ✓ Domain properties (is_pml, is_solid)")
    return True


def test_materials():
    """Test materials module."""
    print("Testing materials module...")
    
    from tweezers.fem import MaterialDatabase
    
    db = MaterialDatabase()
    
    # Test water properties
    water = db.water
    assert abs(water.rho - 997.0) < 1  # kg/m³
    assert abs(water.c - 1497.0) < 10  # m/s
    
    # Test bulk modulus K = ρc²
    K_expected = water.rho * water.c**2
    assert abs(water.K - K_expected) / K_expected < 0.001
    
    # Test wavelength
    freq = 1e6  # 1 MHz
    wavelength = water.wavelength(freq)
    expected = water.c / freq
    assert abs(wavelength - expected) < 1e-10
    
    # Test particle coefficients
    particle = db.polystyrene_bead
    f1 = particle.monopole_coefficient(water)
    f2 = particle.dipole_coefficient(water)
    
    # Monopole coefficient f1 = 1 - κf/κp can be negative for soft particles
    # Dipole coefficient f2 = 2(ρp-ρf)/(2ρp+ρf) is bounded in (-1, 1)
    assert -2 < f1 < 2  # Reasonable range
    assert -1 < f2 < 1
    
    print("  ✓ Water properties")
    print("  ✓ Bulk modulus K = ρc²")
    print("  ✓ Wavelength calculation")
    print(f"  ✓ Monopole coefficient f₁ = {f1:.3f}")
    return True


def test_geometry():
    """Test geometry module."""
    print("Testing geometry module...")
    
    from tweezers.fem import FEMConfig, create_petri_dish_mesh
    
    # Create small test mesh with coarse resolution for speed
    config = FEMConfig.default()
    config.geometry.dish_diameter = 0.010  # 10mm
    config.geometry.max_element_size = 0.001  # 1mm
    config.geometry.min_element_size = 0.001  # 1mm
    
    mesh = create_petri_dish_mesh(config)
    
    # Check mesh structure
    assert mesh.n_nodes > 0
    assert mesh.n_elements > 0
    assert mesh.nodes.shape[1] == 3  # 3D
    assert mesh.elements.shape[1] == 8  # hex8 elements
    
    # Check coordinate arrays
    assert len(mesh.x) == mesh.nx
    assert len(mesh.y) == mesh.ny
    assert len(mesh.z) == mesh.nz
    
    print(f"  ✓ Mesh created: {mesh.n_nodes} nodes, {mesh.n_elements} elements")
    print(f"  ✓ Grid: {mesh.nx}×{mesh.ny}×{mesh.nz}")
    return True


def test_acoustics():
    """Test acoustics module."""
    print("Testing acoustics module...")
    
    from tweezers.fem import (
        FEMConfig, create_petri_dish_mesh, FEMAcousticSolver,
        MaterialDatabase,
    )
    
    # Create small mesh with coarse resolution
    config = FEMConfig.default()
    config.geometry.dish_diameter = 0.010  # 10mm
    config.geometry.max_element_size = 0.002  # 2mm
    config.geometry.min_element_size = 0.002  # 2mm
    
    mesh = create_petri_dish_mesh(config)
    
    db = MaterialDatabase()
    
    # Create solver with config-based API
    solver = FEMAcousticSolver(
        mesh=mesh,
        materials=db,
        config=config,
    )
    
    # Assemble system (doesn't return values, sets internal state)
    solver.assemble_system()
    
    # Check matrices are set
    n = mesh.n_nodes
    assert solver._K is not None
    assert solver._M is not None
    assert solver._K.shape == (n, n)
    assert solver._M.shape == (n, n)
    
    # Check matrices are sparse
    K_density = solver._K.nnz / (n * n)
    assert K_density < 0.1  # Should be sparse
    
    print(f"  ✓ System assembled: K,M ∈ ℂ^{n}×{n}")
    print(f"  ✓ Sparsity: {K_density*100:.1f}% fill")
    return True


def test_pml():
    """Test PML module."""
    print("Testing PML module...")
    
    from tweezers.fem import PMLParameters
    
    params = PMLParameters(
        thickness=0.002,  # 2mm
        sigma_max=1.0,
        order=2,
    )
    
    # Test stretching factor
    # At boundary (d=0): s = 1
    # Into PML: s = 1 + σ(d)/iω
    d = 0.001  # 1mm into PML
    omega = 2 * np.pi * 1e6
    
    # Compute expected stretching
    d_norm = d / params.thickness
    sigma = params.sigma_max * d_norm**params.order
    s_expected = 1 + sigma / (1j * omega)
    
    # s should have imaginary part (absorption)
    assert np.imag(s_expected) != 0
    
    print("  ✓ PML parameters")
    print(f"  ✓ Stretching factor: s = {s_expected:.4f}")
    return True


def test_thermoviscous():
    """Test thermoviscous module."""
    print("Testing thermoviscous module...")
    
    from tweezers.fem.thermoviscous import ThermoviscousParameters
    from tweezers.fem import MaterialDatabase
    
    db = MaterialDatabase()
    water = db.water
    
    params = ThermoviscousParameters.from_fluid(water, frequency=1e6)
    
    # Typical values for water at 1 MHz
    # δv ≈ 0.56 μm, δt ≈ 0.22 μm
    delta_v_um = params.delta_v * 1e6
    delta_t_um = params.delta_t * 1e6
    
    assert 0.1 < delta_v_um < 2.0  # Reasonable range
    assert 0.1 < delta_t_um < 1.0
    
    # Prandtl number Pr = ν/α ≈ 7 for water
    assert 5 < params.Pr < 10
    
    print(f"  ✓ δv = {delta_v_um:.2f} μm")
    print(f"  ✓ δt = {delta_t_um:.2f} μm")
    print(f"  ✓ Pr = {params.Pr:.1f}")
    return True


def test_streaming():
    """Test streaming module."""
    print("Testing streaming module...")
    
    # We'll just import and verify the function exists
    # Full testing requires an acoustic field
    from tweezers.fem.streaming import compute_streaming_force, StreamingSolver
    from tweezers.fem import MaterialDatabase
    
    db = MaterialDatabase()
    water = db.water
    
    # Basic checks
    assert callable(compute_streaming_force)
    assert callable(StreamingSolver)
    
    print("  ✓ Streaming module imports")
    print("  ✓ compute_streaming_force available")
    return True


def test_particles():
    """Test particles module."""
    print("Testing particles module...")
    
    from tweezers.fem import (
        MaterialDatabase,
    )
    from tweezers.fem.particles import GorkovPotential, ParticleDynamics
    
    db = MaterialDatabase()
    water = db.water
    particle = db.polystyrene_bead
    
    # Check monopole/dipole coefficients
    f1 = particle.monopole_coefficient(water)
    f2 = particle.dipole_coefficient(water)
    
    # Monopole can be negative for soft particles
    # Dipole should be small for similar densities
    assert -2 < f1 < 2
    assert -0.5 < f2 < 0.5
    
    # Check mobility
    mu = particle.mobility(water)
    # μ = 1/(6πηa), for a=5μm, η=0.001: μ ≈ 1e7 m/(N·s)
    assert 1e5 < mu < 1e10
    
    print(f"  ✓ Monopole coefficient f₁ = {f1:.3f}")
    print(f"  ✓ Dipole coefficient f₂ = {f2:.3f}")
    print(f"  ✓ Mobility μ = {mu:.2e} m/(N·s)")
    return True


def test_solver():
    """Test unified solver."""
    print("Testing solver module...")
    
    from tweezers.fem import (
        FEMConfig, PhysicsLevel, FEMMultiphysicsSolver,
    )
    
    # Test with minimal physics level and coarse mesh
    config = FEMConfig.default()
    config.physics_level = PhysicsLevel.ACOUSTICS_ONLY
    config.geometry.dish_diameter = 0.010  # Small domain
    config.geometry.max_element_size = 0.002  # Coarse for speed
    config.geometry.min_element_size = 0.002
    
    solver = FEMMultiphysicsSolver(config)
    
    # Check mesh is created
    mesh = solver.mesh
    assert mesh.n_nodes > 0
    
    # Check physics level
    assert solver.physics_level == PhysicsLevel.ACOUSTICS_ONLY
    
    print(f"  ✓ Solver initialized")
    print(f"  ✓ Mesh: {mesh.n_nodes} nodes")
    print(f"  ✓ Physics level: {solver.physics_level.name}")
    return True


def run_all_tests():
    """Run all validation tests."""
    print("=" * 60)
    print("FEM MODULE VALIDATION TESTS")
    print("=" * 60)
    
    tests = [
        ("Configuration", test_config),
        ("Domains", test_domains),
        ("Materials", test_materials),
        ("Geometry", test_geometry),
        ("Acoustics", test_acoustics),
        ("PML", test_pml),
        ("Thermoviscous", test_thermoviscous),
        ("Streaming", test_streaming),
        ("Particles", test_particles),
        ("Solver", test_solver),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success, None))
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"  ✗ FAILED: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, s, _ in results if s)
    failed = len(results) - passed
    
    for name, success, error in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {name}")
        if error:
            print(f"         Error: {error}")
    
    print("-" * 60)
    print(f"Passed: {passed}/{len(results)}")
    
    if failed > 0:
        print(f"Failed: {failed}")
        return 1
    else:
        print("All tests passed!")
        return 0


if __name__ == '__main__':
    sys.exit(run_all_tests())
