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

**3. Bayesian Acceleration Layer** (in `demo_surf_greedy.py`)
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
python scripts/adjoint_steer_kstep.py --fast          # K-step U minimisation
python scripts/adjoint_circle_track_kstep.py --fast   # Circle tracking
python scripts/adjoint_gradcheck.py                   # Gradient verification
```

---

## Next Steps

### Immediate Goals
- Reliable, visually compelling circle-tracking demos
- Parameter tuning for consistent good runs
- Documentation of working configurations

### Future Direction
- **Adjoint MPC:** Optimise over rolling horizon with constraints
- **Learned surrogates:** Accelerate PDE solves for real-time control
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
│   ├── 4puck_demo_surf_greedy.py    # Main 4-puck circle demo
│   ├── path_follow_controlled.py     # Path following demo
│   ├── adjoint_circle_track_kstep.py # Adjoint circle tracking
│   ├── adjoint_steer_kstep.py        # K-step adjoint optimiser
│   ├── adjoint_gradcheck.py          # Gradient verification
│   ├── demo_surf_greedy.py           # Original surf controller
│   └── ...                           # Various diagnostics/experiments
│
├── src/
│   ├── acousto/                  # Core physics library
│   │   ├── solvers/                 # Helmholtz PDE solvers
│   │   │   ├── fd_helmholtz_1d.py
│   │   │   ├── fd_helmholtz_2d_forced_25d.py
│   │   │   └── helmholtz_3d_simple.py
│   │   ├── force/                   # Gor'kov potential & radiation force
│   │   │   ├── gorkov_1d.py
│   │   │   ├── gorkov_2d.py
│   │   │   └── gorkov_3d.py
│   │   ├── adjoint/                 # Adjoint gradient computation
│   │   │   ├── gradients.py
│   │   │   └── trajectory.py
│   │   ├── analysis/                # Trap finding, stiffness
│   │   │   └── traps_2d.py
│   │   └── dynamics/                # Particle motion
│   │
│   └── tweezers/                 # Control & visualisation
│       ├── control/                 # Controllers
│       ├── actuation/               # Transducer models
│       ├── viz/                     # 2D/3D rendering
│       │   ├── render_2d.py
│       │   └── render_3d.py
│       └── diagnostics/             # Analysis tools
│
└── results/                      # Output folder (mostly gitignored)
    ├── 4puck_demo_surf_greedy/      # Committed demo results
    │   └── run_20260116_165630/
    └── path_follow_controlled/
        └── path_follow_controlled.gif
```

---

## License

[Add license info]
