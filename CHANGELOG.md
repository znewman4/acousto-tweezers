# Changelog

All notable changes to the Acousto-Tweezers project.

---

## [2.2.0] - 26th January 2026

### 🔧 Environment & Validation Overhaul

**Root Cause Identified: Complex PETSc Required**
The January 25 validation tests reported "PASS" but were running with `PETSc.ScalarType = float64` (REAL), not `complex128` as required. This means the Helmholtz solver was incorrectly treating fields as real, producing physically meaningless results.

**Environment Fix**
- Created `environment/complex-fenicsx.yml` with explicit `petsc=3.21.*=*complex*`
- Created `environment/setup_env_complex.sh` automated setup script
- Created `scripts/validation/test_env_complex_petsc.py` as mandatory runtime gate

**New Validation Suite**
- `test_env_complex_petsc.py`: Runtime gate - fails fast if PETSc is real
- `test_acoustics_smoke.py`: Level 1 smoke test that verifies nonzero complex fields
- `test_pml_smoke.py`: PML validation with complexity proof (Im(s) ≠ 0)
- Updated `run_all_tests.py` to run env gate first and fail fast

**Visualization Fixes**
- Fixed color scaling: clim computed ONCE and applied to ALL frames (no flicker)
- Added frame stamps: max|p|, PPW, timestamp, run_id
- Added headless rendering support with `pv.start_xvfb()`

**Diagnostics Fixes**
- Fixed DOF count reporting: Added `_get_pressure_dofs()` and `_get_displacement_dofs()` helpers
- Now correctly extracts DOF count from function space instead of returning 0

**Honesty Checkpoint**
- README updated with ⚠️ STATUS section and complex PETSc verification
- CHANGELOG updated to reflect actual validated state
- No claims of "PASS" without evidence from complex PETSc environment

**Status: BLOCKED**
Tests cannot pass until environment is switched to `acousto-complex` with proper complex PETSc.

---

## [2.1.0] - 25th January 2026

### ⚠️ Complex PETSc Backend (VALIDATION INCOMPLETE)

**NOTE: This session ran with REAL PETSc (float64), so reported "PASS" results are invalid.**

**Attempted Fixes (correct code, wrong environment):**
- Changed environment spec to use `petsc=3.21.*=complex*`
- Fixed UFL form ordering: `inner(trial, test)` for proper conjugation
- Geometry module rewrite with Gmsh physical groups
- Form fixes for complex mode

**Validation Test Suite Created (ran with wrong PETSc):**
- `test_acoustics_only.py`
- `test_pml_simple.py`
- `test_interface_continuity.py`
- `test_fluid_solid_coupled.py`

The code architecture is correct, but results were computed with real scalars.

---

## [2.0.0] - 25th January 2026

### 🚀 Major Refactor: FEniCSx Integration

Complete rewrite of the physics simulation using **FEniCSx (DOLFINx + PETSc)** for
research-grade accuracy. This replaces the previous homebrew FEM approach.

#### New Package: `src/tweezers/fenicsx/`

| Module | Description |
|--------|-------------|
| `config.py` | `FEMConfig` dataclass with physics ladder configuration |
| `domains.py` | `Domain` and `Interface` enums for multi-domain tagging |
| `materials.py` | `MaterialDatabase` with temperature-dependent properties |
| `geometry.py` | Gmsh mesh generation via `create_petri_dish_geometry()` |
| `acoustics.py` | Helmholtz solver with UFL weak forms |
| `solids.py` | Linear elasticity with UFL weak forms |
| `coupling.py` | Monolithic fluid-solid coupling solver |
| `pml.py` | PML with complex coordinate stretching |
| `thermoviscous.py` | Viscous/thermal boundary layer corrections |
| `streaming.py` | Acoustic streaming (Stokes solver) |
| `particles.py` | Gorkov potential and `ParticleDynamics` |
| `solver.py` | `FEMMultiphysicsSolver` orchestrating all physics |
| `diagnostics.py` | Mesh quality, energy balance, convergence checks |
| `solver_utils.py` | PETSc linear system utilities |

#### Physics Ladder (7 Levels)

```
Level 7: PARTICLES         ← Particle dynamics with Stokes drag
Level 6: STREAMING         ← Acoustic streaming velocity field
Level 5: THERMOVISCOUS     ← Boundary layer loss corrections
Level 4: FLUID_SOLID       ← Elastic waves in dish structure
Level 3: FLUID_AIR_BATH    ← Multi-fluid domains
Level 2: ACOUSTICS_PML     ← Helmholtz with PML boundaries
Level 1: ACOUSTICS_ONLY    ← Helmholtz equation in water domain
```

#### Key Features

- **FEniCSx 0.10.0**: Latest stable DOLFINx release
- **UFL weak forms**: Physics defined in symbolic Python
- **FFCx code generation**: Auto-optimized assembly kernels
- **PETSc KSP solvers**: Industrial-strength linear algebra
- **Gmsh integration**: Proper geometry with physical groups
- **Complex support**: Time-harmonic acoustics with absorbing BCs

### Demos

- `scripts/demo_2d_acoustics.py`: Quick 2D validation demo
- `scripts/generate_acoustic_animation.py`: Generates animated GIF
- `scripts/validation/test_2d_helmholtz.py`: 2D Helmholtz validation

### Outputs Generated

- `results/demo_2d_acoustics/acoustic_wave.gif`: Animated pressure field
- `results/demo_2d_acoustics/standing_wave.png`: Static visualization

### Deprecated

The following modules are moved to `src/tweezers/redundant/`:

- `tweezers.fem` → `redundant/fem_old/` (homebrew FEM)
- `tweezers.physics` → `redundant/physics/` (finite differences)
- `tweezers.grid` → `redundant/grid/` (FD grid)

### Migration Guide

**Before (deprecated):**
```python
from tweezers.fem import FEMConfig, FEMMultiphysicsSolver
```

**After (FEniCSx):**
```python
from tweezers.fenicsx import FEMConfig, FEMMultiphysicsSolver
```

---

## [1.0.0] - 24th January 2026

### Added: Homebrew FEM Framework (Now Deprecated)

The initial FEM implementation using custom assembly. This has been superseded
by the FEniCSx implementation in v2.0.0.

#### Package: `src/tweezers/fem/` (Now at `redundant/fem_old/`)

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

This was superseded by FEniCSx due to:
- No optimized code generation (slow assembly)
- Limited element types (only hex8)
- No complex number support
- Maintenance burden

---

## [0.x] - Prior to January 2026

### Legacy Implementation

Original finite difference implementation for acoustic simulations.
Now archived in `src/tweezers/redundant/physics/` and `redundant/grid/`.

Limitations that led to FEM rewrite:
- Poor accuracy at material interfaces (staircase artifacts)
- No proper PML implementation  
- No thermoviscous effects
- Required very fine grids for convergence

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
result = solver.solve()
```

### Enhanced Output Format & Diagnostics (24 Jan 2026)

Added comprehensive output structure and automated diagnostics per original MASTER BRIEF:

#### New Script: `run_fem_enhanced.py`

Full production-ready simulation with automatic result logging and validation:

```bash
python run_fem_enhanced.py
```

Generates timestamped output directory: `results/run_YYYYMMDD_HHMMSS/`

```
run_YYYYMMDD_HHMMSS/
  ├── config.json                   # Configuration parameters
  ├── run.log                        # Complete execution log
  ├── summary.csv                    # All computed metrics
  ├── traj.csv                       # 50 particle trajectories
  ├── anim_U_contours.gif           # Pressure field animation (when 3D)
  ├── anim_streaming.gif            # Streaming velocity animation
  └── diagnostics/
      ├── sanity_report.txt         # Physics validation summary
      ├── pml_reflection.txt        # PML boundary performance
      ├── interface_residuals.txt   # Fluid-solid coupling errors
      └── energy_budget.txt         # Energy conservation check
```

#### Automatic Diagnostics Computed

After every run, the following are computed and saved:

1. **Mesh Quality**
   - Wavelength λ
   - Grid spacing h
   - Points per wavelength (PPW) — recommend > 10

2. **Acoustic Field Statistics**
   - max|p|, mean|p|, rms|p| (pressure extrema and statistics)
   - Detected amplitude range and field uniformity

3. **Streaming Field Statistics**
   - min/max streaming velocity |ū|
   - Velocity gradient ∇ū (shear rate)
   - Reynolds number for streaming regime assessment

4. **Particle Dynamics**
   - Mean and max displacement per particle
   - Estimated particle velocity per timestep

5. **PML Boundary Performance**
   - Reflection coefficient (target < 1%)
   - Quality assessment vs. target

6. **Physical Validation**
   - Sanity report with PASS/WARN/FAIL for each metric
   - Energy conservation (preliminary)
   - Interface residuals (pressure and velocity continuity)

#### Summary Metrics CSV

`summary.csv` contains all computed metrics in tabular form:
```csv
metric,value,unit
frequency_Hz,2000000.0,
wavelength_m,0.00074,
grid_spacing_m,0.003,
points_per_wavelength,0.2,
p_max_Pa,0.0,
p_mean_Pa,0.0,
...
```

#### Particle Trajectories CSV

`traj.csv` contains full temporal history of all particles:
```csv
particle_id,time,x_m,y_m,z_m
0,0,0.001,0.002,0.003
0,1,0.001,0.002,0.004
...
```

Enables post-processing: trajectory analysis, trapping efficiency, clustering.

#### Visual Outputs

- `anim_U_contours.gif`: Acoustic pressure field at multiple z-slices (when 3D data available)
- `anim_streaming.gif`: Streaming velocity field animation
  - Uses normalized color scale [−1, +1]
  - Frame rate: 5 frames/sec
  - Auto-generated for 2D/3D slices

### Known Issues
- FD solver struggles with fine resolution (memory, accuracy)
- No proper PML boundaries (anechoic BC approximation)
- No thermoviscous or streaming effects
- Requires coarse grid for reasonable runtime
