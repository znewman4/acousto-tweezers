# Acousto-Tweezers: FEniCSx Multiphysics Simulator

A research-grade FEM-based multiphysics simulator for acoustic tweezers using **FEniCSx (dolfinx + PETSc)**.

---

## Quick Start

### Installation

```bash
git clone <repo-url>
cd acousto-tweezers

# Create conda environment with FEniCSx
conda create -n fenicsx python=3.11
conda activate fenicsx
conda install -c conda-forge fenics-dolfinx petsc4py gmsh

# Install package
pip install -e .
```

### Run a Simulation

```bash
# The ONLY blessed entry point:
python scripts/run_fem_multiphysics.py --level ACOUSTICS_ONLY --quick

# Full 3D simulation (requires more memory):
python scripts/run_fem_multiphysics.py --level ACOUSTICS_PML --ppw 10
```

### Output Structure

All runs are saved to `results/fem_multiphysics/run_YYYYMMDD_HHMMSS/`:

```
run_20260125_123456/
├── config.json              # Configuration used
├── summary.csv              # Key metrics
├── diagnostics/
│   ├── sanity_report.txt    # Physics sanity checks
│   ├── mesh_report.txt      # Mesh quality metrics
│   ├── solver_report.txt    # Solver performance
│   ├── acoustics_report.txt # Field statistics
│   └── pml_report.txt       # PML validation
├── figures/
│   ├── p_slice.png          # Pressure field slice
│   └── anim_U_contours.gif  # 3D animated pressure
├── mesh/
├── fields/
└── logs/
    └── run.log
```

---

## Physics Ladder

The simulator implements a 7-level physics ladder. Each level includes all physics from lower levels:

| Level | Name | Description |
|-------|------|-------------|
| 1 | `ACOUSTICS_ONLY` | Helmholtz equation in water |
| 2 | `ACOUSTICS_PML` | + PML boundary conditions |
| 3 | `FLUID_AIR_BATH` | + Air and bath domains |
| 4 | `FLUID_SOLID` | + Elastic plate/wall coupling |
| 5 | `THERMOVISCOUS` | + Boundary layer corrections |
| 6 | `STREAMING` | + Acoustic streaming |
| 7 | `PARTICLES` | + Radiation force & dynamics |

---

## Architecture

```
src/tweezers/
├── core/                 # Common utilities
│   ├── io.py            # Run directory management
│   └── logging.py       # Logging setup
├── fenicsx/             # PRIMARY FEM backend
│   ├── config.py        # FEMConfig dataclass
│   ├── domains.py       # Domain/Interface enums
│   ├── materials.py     # Material database
│   ├── geometry.py      # Gmsh mesh generation
│   ├── acoustics.py     # Helmholtz solver
│   ├── solids.py        # Elasticity solver
│   ├── coupling.py      # Fluid-solid coupling
│   ├── pml.py           # PML implementation
│   ├── streaming.py     # Acoustic streaming
│   ├── particles.py     # Gor'kov & dynamics
│   ├── solver.py        # Multiphysics orchestrator
│   └── diagnostics.py   # Expanded diagnostics
└── redundant/           # Deprecated code (archived)
```

---

## Validation Tests

Run validation micro-tests:

```bash
# PML reflection test (<1% target)
python scripts/validation/test_pml_reflection.py

# Interface continuity test
python scripts/validation/test_interface_continuity.py

# 2D Helmholtz validation
python scripts/validation/test_2d_helmholtz.py
```

---

## Legacy Control Module

The control module (`tweezers.control`) is preserved for MPC/trajectory planning but uses the legacy FD solver. See `scripts/redundant/` for old control scripts.

---

## References

- FEniCSx: https://fenicsproject.org/
- DOLFINx: https://github.com/FEniCS/dolfinx
- Gmsh: https://gmsh.info/

---

## License

MIT License

### Visualisation
- GIF output with particle trail, target marker, force vectors
- Gor'kov potential contour overlay (stable colour scaling)
- Comparison plots for multi-method runs

---

## Adjoint Gradient System

The adjoint module (`src/acousto/adjoint/`) provides exact gradients for control optimisation:

- **Direct gradients:** ∂U/∂u via adjoint of Helmholtz solve
- **Trajectory gradients:** Discrete-time adjoint backpropagation through dynamics
- **Verified:** Matches finite differences to <1% relative error

Key scripts:
```bash
# MPC Controllers (recommended)
python scripts/adjoint_circle_track_mpc.py            # MPC circle tracking
python scripts/adjoint_path_track_mpc_compare.py      # MPC path tracking
python scripts/mpc_vs_greedy_4puck.py                 # MPC vs greedy comparison

# K-step optimisation (simpler, no receding horizon)
python scripts/adjoint_steer_kstep.py --fast          # K-step U minimisation
python scripts/adjoint_circle_track_kstep.py --fast   # Circle tracking

# Verification
python scripts/adjoint_gradcheck.py                   # Gradient verification
```

---

## Next Steps

### Completed ✅
- **Adjoint MPC:** Receding-horizon optimisation with exact gradients — working
- **Path tracking:** Arbitrary parametric paths, not just circles — working
- **MPC vs Greedy comparisons:** Benchmarking scripts with quantitative metrics

### Immediate Goals
- **Multi-particle adjoint MPC:** Extend to N particles with shared control
- **Collision avoidance:** Inter-particle proximity constraints
- **Warm-starting:** Shift and reuse solutions for faster MPC solves

---

## FEM Multiphysics Simulator (January 2026 - NEW)

A complete rewrite using **Finite Element Method (FEM)** for research-grade accuracy:

```python
from tweezers.fem import FEMConfig, PhysicsLevel, FEMMultiphysicsSolver

# Configure simulation
config = FEMConfig.default()
config.physics_level = PhysicsLevel.PARTICLES  # Full physics ladder
config.geometry.dish_diameter = 35e-3          # 35mm Petri dish
config.physics.frequency = 2.0e6               # 2 MHz

# Run simulation
solver = FEMMultiphysicsSolver(config)
result = solver.run_simulation()

# Access results
print(f"Pressure field: {result.pressure.shape}")
print(f"Particle positions: {result.particle_positions}")
```

### Physics Ladder

```
Level 7: PARTICLES         ← Particle dynamics with Stokes drag
Level 6: RADIATION_FORCE   ← Gor'kov potential
Level 5: STREAMING         ← Acoustic streaming (Eckart forcing)  
Level 4: THERMOVISCOUS     ← Boundary layer corrections
Level 3: PML               ← Perfectly Matched Layer (< 1% reflection)
Level 2: SOLID_COUPLING    ← Elastic waves in dish
Level 1: ACOUSTICS_ONLY    ← Helmholtz equation in water
```

### Domain Structure

```
                    ┌─────────────────────────────────┐
                    │           PML_TOP               │
     ┌──────────────┼─────────────────────────────────┼──────────────┐
     │   PML_LEFT   │             AIR                 │  PML_RIGHT   │
     │              ├─────────────────────────────────┤              │
     │              │           WATER                 │              │ 
     │              │ ┌─────────────────────────────┐ │              │
     │              │ │          PLATE              │ │              │
     │              ├─┴─────────────────────────────┴─┤              │
     │              │             BATH                │              │
     └──────────────┼─────────────────────────────────┼──────────────┘
                    │          PML_BOTTOM             │
                    └─────────────────────────────────┘
```

### Key Improvements over FD

| Feature | Old (FD) | New (FEM) |
|---------|----------|-----------|
| Accuracy | O(h²) | O(h⁴) with hex8 |
| PML boundaries | ❌ | ✅ < 1% reflection |
| Interface conditions | Staircase | Proper weak form |
| Multi-domain | Approximated | Tagged domains |
| Thermoviscous | ❌ | ✅ Boundary layers |
| Streaming | ❌ | ✅ Eckart + Reynolds |

### Run the Demo

```bash
# Entry point script
python scripts/run_fem_multiphysics.py --physics-level 7 --frequency 2e6

# Validation tests
python scripts/validation/test_fem_modules.py
```

See [docs/INDEX.md](docs/INDEX.md) for full documentation.

---

### Future Direction
- **Learned value functions:** Replace long horizons with short horizon + terminal V(x)
- **Learned surrogates:** Accelerate PDE solves for real-time control
- **Second-order methods:** L-BFGS or Gauss-Newton for faster convergence
- **Experimental validation:** Close the loop with real hardware

---

## Repository Structure

```
acousto-tweezers/
├── README.md
├── pyproject.toml
├── .gitignore
│
├── scripts/                      # Runnable demos and experiments
│   ├── run_fem_multiphysics.py      # ★ FEM entry point (NEW)
│   ├── validation/                  # Module validation tests
│   │   └── test_fem_modules.py
│   ├── 4puck_demo_surf_greedy.py    # Main 4-puck circle demo
│   ├── adjoint_circle_track_kstep.py # Adjoint circle tracking
│   └── ...                           # Various diagnostics/experiments
│
├── src/
│   ├── acousto/                  # Core physics library (legacy)
│   │   ├── solvers/                 # Helmholtz PDE solvers
│   │   ├── force/                   # Gor'kov potential & radiation force
│   │   ├── adjoint/                 # Adjoint gradient computation
│   │   ├── analysis/                # Trap finding, stiffness
│   │   └── dynamics/                # Particle motion
│   │
│   └── tweezers/                 # Control & FEM multiphysics
│       ├── fem/                     # ★ FEM modules (NEW)
│       │   ├── config.py               # Single authoritative config
│       │   ├── domains.py              # Domain/interface types
│       │   ├── materials.py            # MaterialDatabase
│       │   ├── geometry.py             # Hex8 mesh generation
│       │   ├── acoustics.py            # Helmholtz weak form
│       │   ├── solids.py               # Elastic solid mechanics
│       │   ├── pml.py                  # Perfectly Matched Layer
│       │   ├── thermoviscous.py        # Boundary layer corrections
│       │   ├── streaming.py            # Acoustic streaming
│       │   ├── particles.py            # Gor'kov + particle dynamics
│       │   ├── solver.py               # FEMMultiphysicsSolver
│       │   └── diagnostics.py          # Analysis tools
│       ├── control/                 # Controllers (MPC, greedy)
│       ├── actuation/               # Transducer models
│       ├── viz/                     # 2D/3D rendering
│       ├── diagnostics/             # Analysis tools
│       └── redundant/               # Old FD modules (deprecated)
│
├── docs/                         # Documentation
│   ├── INDEX.md                     # Documentation index
│   └── ...
│
└── results/                      # Output folder (mostly gitignored)
    └── ...
```

---

## License

[Add license info]
