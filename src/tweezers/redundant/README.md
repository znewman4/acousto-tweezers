# Redundant Modules (Deprecated)

This directory contains modules that have been superseded by the new FEM-based
implementation in `src/tweezers/fem/`.

## Migration Summary

The following modules were replaced:

| Old Location | New FEM Module | Notes |
|-------------|----------------|-------|
| `physics/acoustics/solver.py` | `fem/acoustics.py` | FD → FEM weak form |
| `physics/acoustics/pml.py` | `fem/pml.py` | PML with complex stretching |
| `physics/acoustics/materials.py` | `fem/materials.py` | MaterialDatabase class |
| `physics/acoustics/geometry.py` | `fem/geometry.py` | create_petri_dish_mesh() |
| `physics/acoustics/thermoviscous.py` | `fem/thermoviscous.py` | Same physics, cleaner API |
| `physics/solver.py` | `fem/solver.py` | FEMMultiphysicsSolver |
| `physics/streaming/` | `fem/streaming.py` | StreamingSolver class |
| `physics/particle/` | `fem/particles.py` | GorkovPotential, ParticleDynamics |
| `grid/grid3d.py` | `fem/geometry.py` | Hex8 mesh instead of FD grid |

## Reason for Deprecation

The original implementation used Finite Differences (FD) which:
- Had poor boundary handling at material interfaces
- Required excessive resolution for accuracy
- Could not properly implement PML boundaries
- Suffered from staircase artifacts at curved boundaries

The new FEM implementation uses:
- Weak form Helmholtz equation with Galerkin FEM
- Hexahedral (hex8) elements with 2×2×2 Gauss quadrature
- Complex coordinate stretching for PML
- Proper domain tagging for multi-material systems
- Thermoviscous boundary layer corrections

## Files Preserved

The old modules are kept for reference but should NOT be used.
All new code should import from `tweezers.fem` instead.

## Date: January 2026
