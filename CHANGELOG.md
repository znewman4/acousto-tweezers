# Changelog

All notable changes to the Acousto-Tweezers project.

---

## [2.1.0] - 25th January 2026

### ✅ Complex PETSc Backend & Validation (Session 2)

**Critical Fix: Complex scalar type enforcement**
- Changed environment to use `petsc=3.21.*=complex*` for proper Helmholtz solving
- Verified `PETSc.ScalarType = numpy.complex128` throughout
- Fixed UFL form ordering: `inner(trial, test)` for proper conjugation in complex mode

**Geometry Module Rewrite**
- Replaced centroid-based domain classification with Gmsh physical groups
- New `VolumeTracker` class tracks domain identity through fragment operations
- Physical groups assigned at creation time (required by spec)

**Form Fixes for Complex Mode**
- `acoustics.py`: Fixed to `inner(grad(p), grad(v))` (was reversed)
- `coupling.py`: Fixed form ordering + added LaTeX documentation for interface conditions
- Material functions: Changed from `dtype=np.float64` to `dtype=PETSc.ScalarType`

**Validation Test Suite Created**
- `test_acoustics_only.py`: Full solver stack → max|p| = 2.14×10⁸ Pa ✓
- `test_pml_simple.py`: PML absorption → 90.1% absorption confirmed ✓
- `test_interface_continuity.py`: Solution smoothness → CV = 47.4% ✓
- `test_fluid_solid_coupled.py`: Coupled physics → non-zero fields ✓
- `run_all_tests.py`: Master test runner → 4/4 tests passing

**Visualization Module**
- New `src/tweezers/fenicsx/visualization.py` using PyVista
- Functions: `plot_pressure_field_3d()`, `plot_cross_section()`, `create_animation_frames()`, `frames_to_gif()`
- Supports 3D slices, cross-sections, and 360° rotation animations

**PML Implementation**
- Proper coordinate stretching: `s_x = 1 - iσ/ω` for rightward traveling waves
- Polynomial absorption profile: `σ(x) = σ_max * ((x - L_phys)/L_pml)³`
- Validated absorption >90% in test domain

**Results**
- All validation tests pass (4/4)
- Non-zero complex pressure fields confirmed at all physics levels
- Form ordering correct for DOLFINx 0.9.0 complex mode
- Ready for production simulations

**Diagnostic Tests**
- `scripts/run_diagnostics.py`: 3/3 tests passing
  - Mesh quality verification
  - Field statistics computation
  - Convergence analysis with mesh refinement

**Visualization Outputs**
- `scripts/demo_visualization.py` generates:
  - 3D slice plots with PyVista
  - 2D cross-sections
  - 360° rotation GIF animations
- Example outputs in `results/visualization_demo/`

**Bug Fixes**
- Fixed mesh quality test to use DOLFINx 0.9.0 API (removed deprecated `cell_volume`)
- Added PIL installation for GIF generation

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
