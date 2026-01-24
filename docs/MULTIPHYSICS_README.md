# Multiphysics 3D Acoustic Trapping Simulator

This document describes the complete multiphysics simulation module for 3D acoustic particle trapping.

## Overview

The `tweezers.physics` package provides a physically realistic 3D acoustic trapping simulator featuring:

- **Multi-domain acoustics**: Explicit water, air, plate, and wall domains
- **PML boundaries**: Perfectly Matched Layers for open boundaries (no fake anechoic BC)
- **Solid mechanics**: Elastic wave propagation in dish plate and walls
- **Fluid-solid coupling**: Proper interface conditions at boundaries
- **Thermoviscous effects**: Viscous and thermal boundary layer corrections
- **Acoustic streaming**: Second-order mean flow from acoustic forcing
- **Radiation force**: Gor'kov potential and force field computation
- **Particle dynamics**: Full trajectory integration with drag and streaming

## Quick Start

```python
from tweezers.physics import MultiphysicsSolver, SimulationParameters

# Create simulation parameters
params = SimulationParameters(
    frequency=2.0e6,          # 2 MHz
    dish_radius=17.5e-3,      # 35 mm diameter
    water_depth=2.0e-3,       # 2 mm water
    grid_resolution=50e-6,    # 50 μm resolution
)

# Run simulation
solver = MultiphysicsSolver(params)
results = solver.solve(
    solve_streaming=True,
    compute_gorkov=True,
    simulate_particles=True,
)

# Results contain:
# - results.acoustic_field: Complex pressure and velocity
# - results.streaming_field: Mean streaming velocity
# - results.gorkov_potential: Radiation potential field
# - results.particle_trajectories: Simulated paths
```

## Command Line Demo

```bash
# Full simulation with default parameters
python scripts/demo_helmholtz3d_multiphysics.py

# Quick test run
python scripts/demo_helmholtz3d_multiphysics.py --quick

# Custom parameters
python scripts/demo_helmholtz3d_multiphysics.py \
    --frequency 1.5 \
    --resolution 75 \
    --n-particles 20
```

## Module Structure

```
src/tweezers/physics/
├── __init__.py              # Package exports
├── solver.py                # MultiphysicsSolver (unified orchestrator)
├── visualization.py         # Plotting and animation
│
├── acoustics/               # Multi-domain acoustic solver
│   ├── geometry.py          # MultiDomainGeometry, DomainType
│   ├── materials.py         # FluidMaterial, MaterialDatabase
│   ├── pml.py               # PMLManager, coordinate stretching
│   ├── thermoviscous.py     # Boundary layer corrections
│   └── solver.py            # MultiDomainAcousticSolver
│
├── solid/                   # Elastic solid mechanics
│   ├── materials.py         # SolidMaterial, viscoelastic
│   ├── solver.py            # ElasticSolver (frequency domain)
│   └── coupling.py          # FluidSolidCoupling, interfaces
│
├── streaming/               # Acoustic streaming
│   ├── forcing.py           # Eckart + Reynolds stress
│   └── solver.py            # StokesSolver, StreamingSolver
│
├── particle/                # Radiation force & dynamics
│   ├── properties.py        # Particle3D, contrast factors
│   ├── interpolation.py     # Grid3D, TrilinearInterpolator
│   ├── gorkov.py            # GorkovPotential3D
│   └── dynamics.py          # ParticleDynamics3D, trajectories
│
└── tests/                   # Test suite
    └── test_multiphysics.py
```

## Physical Model

### Governing Equations

1. **Helmholtz equation** (time-harmonic acoustics):
   ```
   ∇·(1/ρ ∇p) + ω²/(ρc²) p = 0
   ```

2. **Elastic wave equation** (frequency domain):
   ```
   ∇·σ(u) + ρω²u = 0
   σ = λ(∇·u)I + μ(∇u + ∇uᵀ)
   ```

3. **Stokes equation** (streaming):
   ```
   -∇P + η∇²u + f_streaming = 0
   ∇·u = 0
   ```

4. **Gor'kov potential**:
   ```
   U = V_p [f₁/(4ρc²)⟨p²⟩ - 3f₂/(8ρ)⟨v²⟩]
   F = -∇U
   ```

5. **Particle dynamics** (overdamped):
   ```
   dx/dt = μF_rad + u_stream
   μ = 1/(6πηa)    (Stokes mobility)
   ```

### Domain Types

| Domain | Symbol | Material | Physics |
|--------|--------|----------|---------|
| Dish water | Ω_w | Water | Helmholtz + streaming |
| Air | Ω_a | Air | Helmholtz |
| Bath water | Ω_b | Water | Helmholtz + PML |
| Plate | Ω_p | Glass | Elastic |
| Walls | Ω_s | Polystyrene | Elastic |

### PML Implementation

The Perfectly Matched Layer uses complex coordinate stretching:
```
s(x) = 1 + (iσ_max/ω)(x/L)^m
```

where:
- σ_max controls absorption strength
- L is PML thickness
- m is polynomial order (typically 2-3)

Target reflection coefficient: R₀ ≈ 10⁻⁶

### Interface Conditions

At fluid-solid interfaces:
- **Continuity of normal velocity**: v_n,fluid = iωu_n,solid
- **Continuity of traction**: p·n = σ·n

## Output Structure

```
results/helmholtz3d_multiphysics/run_YYYYMMDD_HHMMSS/
├── results.npz              # Full simulation data
├── parameters.txt           # Parameter summary
├── pressure_slices.png      # XY, XZ, YZ pressure plots
├── gorkov_potential.png     # Trap potential with minima
├── streaming_field.png      # Velocity magnitude and vectors
├── trajectories_xy.png      # Particle paths (top view)
├── trajectories_xz.png      # Particle paths (side view)
├── anim_particles_xy.gif    # Animated particle motion
├── energy_budget.png        # Power flow analysis
└── summary.png              # Comprehensive overview
```

## Key Classes

### SimulationParameters

```python
@dataclass
class SimulationParameters:
    frequency: float = 2.0e6          # [Hz]
    actuation_amplitude: float = 1e-6  # [m/s]
    dish_radius: float = 17.5e-3      # [m]
    water_depth: float = 2.0e-3       # [m]
    air_height: float = 5.0e-3        # [m]
    plate_thickness: float = 1.0e-3   # [m]
    wall_thickness: float = 1.5e-3    # [m]
    grid_resolution: float = 50e-6    # [m]
    pml_thickness: int = 10           # grid points
    temperature: float = 25.0         # [°C]
```

### MultiphysicsResults

```python
@dataclass
class MultiphysicsResults:
    parameters: SimulationParameters
    geometry: MultiDomainGeometry
    acoustic_field: AcousticField3D
    displacement_field: DisplacementField3D  # optional
    streaming_field: StreamingField          # optional
    gorkov_potential: np.ndarray             # optional
    particle_trajectories: List[ParticleTrajectory]  # optional
    computation_times: Dict[str, float]
    energy_budget: Dict[str, float]
```

## Material Database

Built-in material properties:

**Fluids:**
- Water (temperature-dependent ρ, c, η)
- Air (temperature-dependent)

**Solids:**
- Borosilicate glass (dish plate)
- Polystyrene (walls)

**Particles:**
- Polystyrene microspheres
- Silica beads
- Biological cells
- Air bubbles
- Lipid droplets

## Testing

```bash
# Run all tests
pytest src/tweezers/physics/tests/ -v

# Run specific test class
pytest src/tweezers/physics/tests/test_multiphysics.py::TestGorkovPotential -v

# Run with coverage
pytest --cov=tweezers.physics --cov-report=html
```

## Performance Notes

- **Grid resolution**: λ/20 recommended for accuracy
- **PML thickness**: 10-15 points typical
- **Solver**: GMRES with ILU preconditioner
- **Particle integration**: RK45 adaptive stepping

Typical computation times (on modern CPU):
- 100 μm resolution, 35 mm dish: ~30 s acoustics, ~60 s streaming
- 50 μm resolution: ~4x longer

## References

1. Gor'kov, L. P. (1962). "On the forces acting on a small particle in an acoustical field in an ideal fluid."
2. Bruus, H. (2012). "Acoustofluidics 7: The acoustic radiation force on small particles."
3. Muller et al. (2012). "A numerical study of microparticle acoustophoresis."
4. Settnes & Bruus (2012). "Forces acting on a small particle in an acoustical field."
