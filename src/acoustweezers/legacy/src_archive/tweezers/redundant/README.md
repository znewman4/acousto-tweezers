# Redundant Modules (Deprecated)

This directory contains modules that have been superseded by the new FEniCSx-based
implementation in `src/tweezers/fenicsx/`.

## Migration Summary (January 2026)

### Phase 1: Grid → FEM Transition
The following modules in `grid/` and `physics/` were first replaced with a homebrew
FEM approach in `fem/`.

### Phase 2: FEM → FEniCSx Transition (Current)
The homebrew FEM code in `fem_old/` has been replaced with a proper FEniCSx
(DOLFINx + PETSc) implementation in `fenicsx/`.

| Old Location | New FEniCSx Module | Notes |
|-------------|-------------------|-------|
| `fem_old/acoustics.py` | `fenicsx/acoustics.py` | Helmholtz solver |
| `fem_old/pml.py` | `fenicsx/pml.py` | Complex coordinate stretching |
| `fem_old/materials.py` | `fenicsx/materials.py` | MaterialDatabase class |
| `fem_old/geometry.py` | `fenicsx/geometry.py` | Gmsh mesh generation |
| `fem_old/solids.py` | `fenicsx/solids.py` | Linear elasticity |
| `fem_old/coupling.py` | `fenicsx/coupling.py` | Fluid-solid coupling |
| `fem_old/thermoviscous.py` | `fenicsx/thermoviscous.py` | Boundary layers |
| `fem_old/streaming.py` | `fenicsx/streaming.py` | Acoustic streaming |
| `fem_old/particles.py` | `fenicsx/particles.py` | Gorkov potential, dynamics |
| `fem_old/solver.py` | `fenicsx/solver.py` | FEMMultiphysicsSolver |

## Why FEniCSx?

The homebrew FEM approach was replaced because:
- **No code generation**: Hand-written assembly is error-prone and slow
- **No optimized kernels**: FFCx generates optimized C code
- **Limited elements**: FEniCSx supports arbitrary polynomial degrees
- **No complex support**: Need complex-valued Helmholtz for absorbing BCs
- **Maintenance burden**: FEniCSx is actively maintained by experts

The new FEniCSx implementation provides:
- **UFL weak forms**: Physics defined in symbolic Python
- **FFCx code generation**: Auto-optimized kernels
- **PETSc solvers**: Industrial-strength linear algebra
- **Gmsh integration**: Proper geometry with physical groups
- **DOLFINx 0.10.0**: Latest stable FEniCSx release

## Files Preserved

- `grid/` - Original finite difference grid (deprecated)
- `physics/` - Original FD-based physics modules (deprecated)
- `fem_old/` - Intermediate homebrew FEM (deprecated)

All new code should import from `tweezers.fenicsx` instead.

## Date: January 2026
