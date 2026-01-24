# Acoustic Tweezers: Modelling + Control (Robot-Moved Transducers)

A COMSOL-independent modelling and control engine for robotic acoustic tweezers. Transducers are treated as control inputs that can be moved by robots to reshape the acoustic field in real time.

**Core loop:** actuation parameters → acoustic field (Helmholtz PDE) → Gor'kov potential → radiation force → overdamped particle motion → visualisation

---

## Quick Start

### Installation

```bash
git clone <repo-url>
cd acousto-tweezers
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

**Requirements:** Python 3.10+, works on Linux/macOS/Windows.

### View Pre-Computed Demo Results

The best demo results are committed to the repo:

```bash
# 4-puck surface greedy controller tracking a circle (18MB GIF)
results/4puck_demo_surf_greedy/run_20260116_165630/4puck_demo_surf_greedy.gif

# Path-following with controlled transducers (8MB GIF)
results/path_follow_controlled/path_follow_controlled.gif
```

Open these in any image viewer or browser to see the particle being steered around a circular path while the Gor'kov potential contours evolve.

### Run the Demos Yourself

```bash
# Activate environment
source .venv/bin/activate

# 4-puck demo (greedy surf controller, ~2-3 min)
python scripts/4puck_demo_surf_greedy.py

# Path-following demo
python scripts/path_follow_controlled.py

# Adjoint-based circle tracking (new!)
python scripts/adjoint_circle_track_kstep.py --fast
```

Each script creates a timestamped folder in `results/` containing:
- Animated GIF of the run
- Summary plot (PNG)
- Step-by-step CSV log
- JSON summary with metrics

---

## What's Working (End-to-End)

### Physics Engine
- **2.5D forced Helmholtz solver** using finite differences on a rectangular domain
- Moving transducers represented as spatially-localised velocity boundary sources
- **Gor'kov potential and radiation force** computed from each solved pressure field
- **Bilinear interpolation** for smooth force evaluation at arbitrary particle positions
- **Overdamped particle dynamics** with domain clamping

### Controllers

**1. Greedy Surf Controller** (`scripts/4puck_demo_surf_greedy.py`, `scripts/demo_surf_greedy.py`)
- Enumerates discrete macro-actions (transducer position changes)
- Evaluates each via full PDE solve
- Scores by force alignment with desired direction
- Selects best action per timestep
- Proves particles can be steered by "surfing" a changing force field

**2. Adjoint-Based K-Step Controller** (`scripts/adjoint_circle_track_kstep.py`)
- Optimises control sequence over K-step horizon
- Uses discrete-time adjoint backpropagation for exact gradients
- Circle-tracking objective: radial error + tangent progress + trap stability
- Outperforms greedy on trajectory optimisation

**3. Adjoint MPC Controller** (`scripts/adjoint_circle_track_mpc.py`, `scripts/adjoint_path_track_mpc_compare.py`)
- Receding-horizon model predictive control with gradient-based optimisation
- Two-level adjoint: PDE-level (∂U/∂u) + trajectory-level (backprop through dynamics)
- Supports arbitrary path tracking, not just circles
- Control smoothness penalties for physically realisable signals
- Comparison scripts: `mpc_vs_greedy_4puck.py`, `path_tracking_comparison.py`

**4. Bayesian Acceleration Layer** (in `demo_surf_greedy.py`)
- Surrogate model predicts action quality
- UCB acquisition selects subset for PDE evaluation
- Reduces compute while maintaining accuracy

### Circle Tracking Features
- Tangent + radial correction for stable circular motion
- Adaptive target advancement based on angular progress
- Works with both greedy and adjoint controllers

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
