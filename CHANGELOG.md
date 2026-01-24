# Changelog

All notable changes to the Acousto-Tweezers project.

---

## [Unreleased] - January 2026

### Added: FEM Multiphysics Framework

Complete rewrite of the physics simulation using Finite Element Method (FEM) for
research-grade accuracy. This replaces the finite difference (FD) approach.

#### New Package: `src/tweezers/fem/`

| Module | Description |
|--------|-------------|
| `config.py` | Single authoritative configuration with `FEMConfig` dataclass |
| `domains.py` | `DomainType` and `InterfaceType` enums for multi-domain tagging |
| `materials.py` | `MaterialDatabase` with temperature-dependent properties |
| `geometry.py` | Hex8 mesh generation via `create_petri_dish_mesh()` |
| `acoustics.py` | Helmholtz weak form with `FEMAcousticSolver` |
| `solids.py` | Elastic wave equation with `FEMSolidSolver` |
| `pml.py` | Perfectly Matched Layer with complex coordinate stretching |
| `thermoviscous.py` | Viscous/thermal boundary layer corrections |
| `streaming.py` | Acoustic streaming (Eckart + Reynolds stress) |
| `particles.py` | Gor'kov potential and `ParticleDynamics` |
| `solver.py` | `FEMMultiphysicsSolver` orchestrating all physics |
| `diagnostics.py` | Mesh quality, energy balance, PML reflection checks |

#### Physics Ladder

```
Level 7: PARTICLES         ← Particle dynamics with Stokes drag
Level 6: RADIATION_FORCE   ← Gor'kov potential from acoustic field
Level 5: STREAMING         ← Acoustic streaming velocity field
Level 4: THERMOVISCOUS     ← Boundary layer loss corrections
Level 3: PML               ← Perfectly Matched Layer boundaries
Level 2: SOLID_COUPLING    ← Elastic waves in dish structure
Level 1: ACOUSTICS_ONLY    ← Helmholtz equation in water domain
```

Each level includes all physics from levels below it.

#### Domain Schematic

```
                        ┌─────────────────────────────────┐
                        │           PML_TOP               │
     ┌──────────────────┼─────────────────────────────────┼──────────────────┐
     │                  │             AIR                 │                  │
     │    PML_LEFT      ├────────┬───────────────┬────────┤    PML_RIGHT     │
     │                  │  WALL  │     WATER     │  WALL  │                  │
     │                  │        │   (target)    │        │                  │
     │                  │        └───────────────┘        │                  │
     │                  │              PLATE              │                  │
     │                  ├─────────────────────────────────┤                  │
     │                  │              BATH               │                  │
     │                  │          (transducers)          │                  │
     └──────────────────┼─────────────────────────────────┼──────────────────┘
                        │          PML_BOTTOM             │
                        └─────────────────────────────────┘
```

#### Key Features

- **Weak form FEM**: Galerkin discretization with hex8 elements
- **2×2×2 Gauss quadrature**: Accurate volume integration  
- **PML boundaries**: < 1% reflection target with complex stretching
- **Material interfaces**: Proper fluid-solid coupling conditions
- **Temperature dependence**: Material properties vary with temperature
- **Diagnostics**: Mesh quality, energy conservation, convergence checks

### New Scripts

- `scripts/run_fem_multiphysics.py`: CLI entry point for FEM simulations
- `scripts/validation/test_fem_modules.py`: Module validation micro-tests

### Deprecated

The following modules are moved to `src/tweezers/redundant/`:

- `tweezers.physics.acoustics` → Use `tweezers.fem.acoustics`
- `tweezers.physics.solver` → Use `tweezers.fem.solver`  
- `tweezers.physics.streaming` → Use `tweezers.fem.streaming`
- `tweezers.physics.particle` → Use `tweezers.fem.particles`
- `tweezers.grid` → Use `tweezers.fem.geometry`

These modules used finite differences and had several limitations:
- Poor accuracy at material interfaces (staircase artifacts)
- No proper PML implementation
- No thermoviscous effects
- Required very fine grids for convergence

### Migration Guide

**Before (deprecated):**
```python
from tweezers.physics import MultiphysicsSolver, SimulationParameters

params = SimulationParameters(frequency=2e6, grid_resolution=50e-6)
solver = MultiphysicsSolver(params)
results = solver.solve()
```

**After (new):**
```python
from tweezers.fem import FEMConfig, PhysicsLevel, FEMMultiphysicsSolver

config = FEMConfig.default()
config.physics_level = PhysicsLevel.PARTICLES
config.physics.frequency = 2e6

solver = FEMMultiphysicsSolver(config)
result = solver.run_simulation()
```

---

## [0.1.0] - January 2026 (Prior Work)

### Added
- Finite difference Helmholtz solver (2.5D forced)
- Gor'kov potential and radiation force computation
- Greedy surf controller with macro-actions
- Adjoint-based gradient computation
- K-step and MPC controllers
- 4-puck transducer array support
- Circle and path tracking demos
- Visualization (GIF animation, contour plots)

### Known Issues
- FD solver struggles with fine resolution (memory, accuracy)
- No proper PML boundaries (anechoic BC approximation)
- No thermoviscous or streaming effects
- Requires coarse grid for reasonable runtime
