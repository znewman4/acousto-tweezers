"""
Test suite for multiphysics acoustic trapping simulation.

Tests cover:
- Material properties and contrast factors
- PML coordinate stretching
- Geometry construction
- Acoustic solver convergence
- Gor'kov potential physics
- Particle dynamics
"""
from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

# Import test utilities
def approx_equal(a, b, rel_tol=1e-5, abs_tol=1e-10):
    """Check approximate equality."""
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


# ============================================================================
# Material Properties Tests
# ============================================================================

class TestMaterials:
    """Test material property calculations."""
    
    def test_water_properties(self):
        """Test water properties at standard conditions."""
        from tweezers.physics.acoustics.materials import MaterialDatabase
        
        water = MaterialDatabase.water(25.0)
        
        # Check reasonable ranges
        assert 990 < water.rho < 1010, "Water density out of range"
        assert 1480 < water.c < 1500, "Water sound speed out of range"
        assert 8e-4 < water.eta < 1e-3, "Water viscosity out of range"
    
    def test_air_properties(self):
        """Test air properties."""
        from tweezers.physics.acoustics.materials import MaterialDatabase
        
        air = MaterialDatabase.air(25.0)
        
        assert 1.1 < air.rho < 1.3, "Air density out of range"
        assert 340 < air.c < 350, "Air sound speed out of range"
    
    def test_contrast_factors_polystyrene(self):
        """Test contrast factors for polystyrene in water."""
        from tweezers.physics.acoustics.materials import MaterialDatabase
        from tweezers.physics.particle.properties import (
            ParticleDatabase, compute_contrast_factors
        )
        
        water = MaterialDatabase.water(25.0)
        ps = ParticleDatabase.polystyrene_bead(5.0)
        
        contrast = compute_contrast_factors(ps, water)
        
        # Polystyrene should have positive contrast (moves to nodes)
        assert contrast.acoustic_contrast_factor > 0
        assert contrast.is_positive_contrast
        
        # Literature values: f1 ≈ 0.6, f2 ≈ 0.03
        assert 0.4 < contrast.f1 < 0.8
        assert -0.1 < contrast.f2 < 0.2
    
    def test_contrast_factors_bubble(self):
        """Test contrast factors for air bubble in water."""
        from tweezers.physics.acoustics.materials import MaterialDatabase
        from tweezers.physics.particle.properties import (
            ParticleDatabase, compute_contrast_factors
        )
        
        water = MaterialDatabase.water(25.0)
        bubble = ParticleDatabase.air_bubble(10.0)
        
        contrast = compute_contrast_factors(bubble, water)
        
        # Bubbles should have very negative contrast (moves to antinodes)
        assert contrast.acoustic_contrast_factor < -10
        assert not contrast.is_positive_contrast


# ============================================================================
# PML Tests
# ============================================================================

class TestPML:
    """Test PML implementation."""
    
    def test_pml_stretching(self):
        """Test PML coordinate stretching function."""
        from tweezers.physics.acoustics.pml import PMLRegion, PMLParameters
        
        params = PMLParameters(thickness=10, R0=1e-6)
        region = PMLRegion(params, axis=0, direction=1, start=0.0, dx=0.001)
        
        omega = 2 * np.pi * 1e6  # 1 MHz
        
        # Test at boundary (no stretching)
        s0 = region.stretching_factor(0.0, omega)
        assert approx_equal(s0.real, 1.0)
        assert s0.imag == 0.0
        
        # Test inside PML (should have imaginary part)
        s_inside = region.stretching_factor(0.005, omega)
        assert s_inside.real >= 1.0
        assert s_inside.imag > 0  # Absorbing
    
    def test_pml_reflection_coefficient(self):
        """Test PML reflection estimate."""
        from tweezers.physics.acoustics.pml import compute_pml_reflection_test
        
        # With reasonable PML, reflection should be small
        R = compute_pml_reflection_test(
            thickness=10,
            R0=1e-6,
            omega=2*np.pi*1e6,
            c=1500.0,
            dx=50e-6,
        )
        
        assert R < 0.01, "PML reflection coefficient too large"


# ============================================================================
# Geometry Tests
# ============================================================================

class TestGeometry:
    """Test domain geometry construction."""
    
    def test_standard_dish_geometry(self):
        """Test standard dish geometry creation."""
        from tweezers.physics.acoustics.geometry import create_standard_dish_geometry
        
        geom = create_standard_dish_geometry(
            dish_radius=10e-3,
            water_depth=1e-3,
            resolution=100e-6,
        )
        
        # Check grid created
        assert len(geom.grid_x) > 0
        assert len(geom.grid_y) > 0
        assert len(geom.grid_z) > 0
        
        # Check domains created
        assert len(geom.domains) >= 2  # At least water and air
    
    def test_domain_mask(self):
        """Test domain type assignment."""
        from tweezers.physics.acoustics.geometry import (
            create_standard_dish_geometry, DomainType
        )
        
        geom = create_standard_dish_geometry(
            dish_radius=10e-3,
            water_depth=1e-3,
            resolution=200e-6,
        )
        
        domain_types = geom.get_domain_type_array()
        
        # Check domain types are assigned
        has_water = np.any(domain_types == DomainType.WATER_DISH.value)
        has_air = np.any(domain_types == DomainType.AIR.value)
        
        assert has_water, "No water domain found"
        assert has_air, "No air domain found"


# ============================================================================
# Acoustic Solver Tests
# ============================================================================

class TestAcousticSolver:
    """Test acoustic field solver."""
    
    def test_plane_wave_solution(self):
        """Test solver with known plane wave solution."""
        from tweezers.physics.acoustics.solver import AcousticField3D
        from tweezers.physics.particle.interpolation import Grid3D
        import numpy as np
        
        # Create simple grid
        nx, ny, nz = 50, 50, 50
        x = np.linspace(-5e-3, 5e-3, nx)
        y = np.linspace(-5e-3, 5e-3, ny)
        z = np.linspace(0, 10e-3, nz)
        
        # Create plane wave pressure field
        k = 2 * np.pi * 1e6 / 1500  # 1 MHz in water
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        p0 = 1000.0  # Pa
        pressure = p0 * np.exp(1j * k * Z)
        
        # Create field object
        field = AcousticField3D(pressure)
        
        # Check velocity computation
        rho = 1000.0
        omega = 2 * np.pi * 1e6
        vx, vy, vz = field.velocity(x, y, z, rho, omega)
        
        # For plane wave in z, only vz should be non-zero
        assert np.max(np.abs(vx)) < np.max(np.abs(vz)) * 0.1
        assert np.max(np.abs(vy)) < np.max(np.abs(vz)) * 0.1
        
        # Check energy density
        E_pot, E_kin = field.energy_density(x, y, z, rho, 1500.0)
        
        # For plane wave, E_pot ≈ E_kin (equipartition)
        ratio = np.mean(E_pot) / np.mean(E_kin)
        assert 0.5 < ratio < 2.0, "Energy equipartition violated"


# ============================================================================
# Gor'kov Potential Tests
# ============================================================================

class TestGorkovPotential:
    """Test Gor'kov radiation force calculations."""
    
    def test_standing_wave_trapping(self):
        """Test that particles are trapped at correct locations in standing wave."""
        from tweezers.physics.particle.interpolation import Grid3D
        from tweezers.physics.particle.gorkov import GorkovPotential3D
        from tweezers.physics.particle.properties import ParticleDatabase
        from tweezers.physics.acoustics.materials import MaterialDatabase
        
        # Create 1D standing wave
        nz = 100
        z = np.linspace(0, 1e-3, nz)  # 1 mm domain
        x = np.array([0.0])
        y = np.array([0.0])
        
        grid = Grid3D(x, y, z)
        
        # Standing wave at 1.5 MHz in water (λ = 1 mm)
        k = 2 * np.pi / 1e-3
        omega = k * 1500
        freq = omega / (2 * np.pi)
        
        # p = p0 * sin(kz)
        p0 = 1000.0
        Z = np.zeros((1, 1, nz))
        Z[0, 0, :] = z
        pressure = p0 * np.sin(k * Z)
        
        water = MaterialDatabase.water(25.0)
        particle = ParticleDatabase.polystyrene_bead(5.0)
        
        gorkov = GorkovPotential3D(grid, pressure, water, omega)
        U = gorkov.compute_potential(particle)
        
        # Find minima locations
        U_1d = U[0, 0, :]
        min_idx = np.argmin(U_1d[5:-5]) + 5  # Avoid boundary effects
        min_z = z[min_idx]
        
        # For positive contrast particle, minimum should be at pressure node
        # Nodes at z = n*λ/2 = 0, 0.5 mm
        expected_node = 0.5e-3
        
        assert abs(min_z - expected_node) < 0.05e-3, \
            f"Trap at {min_z*1e3:.2f} mm, expected {expected_node*1e3:.2f} mm"
    
    def test_force_scale(self):
        """Test radiation force magnitude estimate."""
        from tweezers.physics.particle.gorkov import estimate_max_radiation_force
        from tweezers.physics.particle.properties import ParticleDatabase
        from tweezers.physics.acoustics.materials import MaterialDatabase
        
        water = MaterialDatabase.water(25.0)
        particle = ParticleDatabase.polystyrene_bead(5.0)
        
        F_max = estimate_max_radiation_force(
            particle, water,
            frequency=1e6,  # 1 MHz
            pressure_amplitude=1e6,  # 1 MPa
        )
        
        # Order of magnitude check (should be ~pN range for 5 μm particle at 1 MPa)
        assert 1e-12 < F_max < 1e-9, f"Force {F_max:.2e} N out of expected range"


# ============================================================================
# Particle Dynamics Tests
# ============================================================================

class TestParticleDynamics:
    """Test particle trajectory integration."""
    
    def test_stokes_drag(self):
        """Test Stokes drag coefficient calculation."""
        from tweezers.physics.particle.dynamics import StokesianDynamics
        from tweezers.physics.particle.properties import ParticleDatabase
        from tweezers.physics.acoustics.materials import MaterialDatabase
        
        water = MaterialDatabase.water(25.0)
        particle = ParticleDatabase.polystyrene_bead(5.0)
        
        stokes = StokesianDynamics(particle, water)
        
        # Analytical Stokes drag: γ = 6πηa
        gamma_analytical = 6 * np.pi * water.eta * particle.a
        
        assert approx_equal(stokes.gamma, gamma_analytical, rel_tol=1e-10)
    
    def test_relaxation_time(self):
        """Test inertial relaxation time."""
        from tweezers.physics.particle.dynamics import StokesianDynamics
        from tweezers.physics.particle.properties import ParticleDatabase
        from tweezers.physics.acoustics.materials import MaterialDatabase
        
        water = MaterialDatabase.water(25.0)
        particle = ParticleDatabase.polystyrene_bead(5.0)
        
        stokes = StokesianDynamics(particle, water)
        tau = stokes.relaxation_time
        
        # For 5 μm particle in water, τ ≈ 1 μs
        assert 1e-7 < tau < 1e-5, f"Relaxation time {tau:.2e} s out of range"
    
    def test_trajectory_convergence(self):
        """Test that particle reaches equilibrium in uniform force field."""
        from tweezers.physics.particle.interpolation import Grid3D
        from tweezers.physics.particle.gorkov import GorkovPotential3D
        from tweezers.physics.particle.dynamics import ParticleDynamics3D
        from tweezers.physics.particle.properties import ParticleDatabase
        from tweezers.physics.acoustics.materials import MaterialDatabase
        
        # Create grid
        n = 20
        x = np.linspace(-1e-3, 1e-3, n)
        y = np.linspace(-1e-3, 1e-3, n)
        z = np.linspace(0, 1e-3, n)
        grid = Grid3D(x, y, z)
        
        # Standing wave with trap at center
        k = 2 * np.pi / 1e-3
        omega = k * 1500
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        pressure = 1000.0 * np.sin(k * Z)
        
        water = MaterialDatabase.water(25.0)
        particle = ParticleDatabase.polystyrene_bead(5.0)
        
        gorkov = GorkovPotential3D(grid, pressure, water, omega)
        
        dynamics = ParticleDynamics3D(
            grid, gorkov, particle, water,
            streaming_velocity=None,
        )
        
        # Start away from equilibrium
        initial_pos = np.array([0.0, 0.0, 0.3e-3])  # Near node at 0.5 mm
        
        traj = dynamics.simulate(
            initial_pos,
            duration=0.01,  # 10 ms
            dt=1e-5,
        )
        
        # Should move toward z = 0.5 mm
        final_z = traj.positions[-1, 2]
        assert final_z > 0.35e-3, "Particle did not move toward trap"


# ============================================================================
# Interpolation Tests
# ============================================================================

class TestInterpolation:
    """Test 3D interpolation routines."""
    
    def test_trilinear_exact_values(self):
        """Test trilinear interpolation returns exact values at nodes."""
        from tweezers.physics.particle.interpolation import (
            Grid3D, TrilinearInterpolator
        )
        
        # Create small grid
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 1.0, 2.0])
        z = np.array([0.0, 1.0, 2.0])
        grid = Grid3D(x, y, z)
        
        # Create field with known values
        field = np.arange(27).reshape(3, 3, 3).astype(float)
        
        interp = TrilinearInterpolator(grid, field)
        
        # Check node values
        assert interp(np.array([[0.0, 0.0, 0.0]])) == field[0, 0, 0]
        assert interp(np.array([[1.0, 1.0, 1.0]])) == field[1, 1, 1]
        assert interp(np.array([[2.0, 2.0, 2.0]])) == field[2, 2, 2]
    
    def test_trilinear_midpoint(self):
        """Test trilinear interpolation at midpoint."""
        from tweezers.physics.particle.interpolation import (
            Grid3D, TrilinearInterpolator
        )
        
        x = np.array([0.0, 1.0])
        y = np.array([0.0, 1.0])
        z = np.array([0.0, 1.0])
        grid = Grid3D(x, y, z)
        
        # Uniform field
        field = np.ones((2, 2, 2))
        interp = TrilinearInterpolator(grid, field)
        
        # Midpoint should also be 1
        mid_val = interp(np.array([[0.5, 0.5, 0.5]]))
        assert approx_equal(mid_val, 1.0)


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for full simulation pipeline."""
    
    @pytest.mark.slow
    def test_full_pipeline_runs(self):
        """Test that full simulation pipeline runs without error."""
        from tweezers.physics import MultiphysicsSolver, SimulationParameters
        
        # Use very coarse resolution for speed
        params = SimulationParameters(
            frequency=1e6,
            grid_resolution=500e-6,  # Very coarse
            dish_radius=5e-3,
            water_depth=1e-3,
        )
        
        solver = MultiphysicsSolver(params, verbose=False)
        
        # Run minimal simulation
        results = solver.solve(
            solve_streaming=False,  # Skip for speed
            compute_gorkov=True,
            simulate_particles=False,
        )
        
        # Check results exist
        assert results.acoustic_field is not None
        assert results.geometry is not None
        assert results.gorkov_potential is not None


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
