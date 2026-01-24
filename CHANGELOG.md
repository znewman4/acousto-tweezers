# Changelog

All notable changes to the Acousto-Tweezers project.

---

## [Unreleased] - 24th January 2026

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
