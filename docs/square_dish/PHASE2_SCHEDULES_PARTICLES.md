# Phase 2: Time-Varying Phase Schedules with Particle Dynamics

**Date:** February 2026  
**Status:** Complete and tested

## Overview

Phase 2 implements **quasi-static time evolution** where acoustic phase patterns change over time and particles respond via overdamped Stokes dynamics. At each "macro time step":

1. Solve Helmholtz equation for current wall phases φ(t)
2. Compute Gor'kov potential U and force field F = -∇U on midplane
3. Advance particle positions using overdamped dynamics with sub-stepping
4. Output diagnostics (CSV/JSON) and visualizations (PNG)

**Key features:**
- 3 predefined phase schedules (step, ramp, sinusoidal)
- 5 particles with deterministic initial placement (cross pattern)
- Overdamped Stokes drag: ẋ = μF where μ = 1/(6πηa)
- Safety constraints: speed clamping and wall collision detection
- Per-step output: |p| and U visualizations with particle overlays

## Implementation Details

### Phase Schedules

Three schedules are available via `--schedule` flag:

#### 1. **step_lr** - Step Transition
Sudden switch from uniform to left-right opposite phase at t = T/2

```
t < T/2:  φ = [0, 0, 0, 0]
t ≥ T/2:  φ = [0, π, 0, π]  (L-R opposite)
```

**Use case:** Test particle response to sudden field reconfiguration

#### 2. **ramp_quadrature** - Smooth Ramp (DEFAULT)
Linear transition from uniform to quadrature phase over total time T

```
φ(t) = α(t) × [0, π/2, π, 3π/2]
where α(t) = t/T clipped to [0, 1]
```

**Use case:** Gradual steering toward quadrature trap configuration

#### 3. **sine_pushpull** - Sinusoidal Oscillation
Left and right walls oscillate in anti-phase

```
φ_L(t) = (π/2) × sin(2πt/T)
φ_R(t) = -(π/2) × sin(2πt/T)
φ_F(t) = 0
φ_B(t) = 0
```

**Use case:** Periodic "push-pull" motion along x-axis

### Particle Dynamics

**Overdamped Stokes equation:**
```
ẋ = μ × F
where:
  μ = 1/(6πηa)  [Stokes mobility]
  η = 8.9×10⁻⁴ Pa·s  [water viscosity]
  a = 40 μm  [particle radius, default]
  F = -∇U  [Gor'kov radiation force]
```

**Initial positions (deterministic cross pattern):**
- Particle 1: center (L/2, L/2)
- Particle 2: center + offset right
- Particle 3: center + offset left
- Particle 4: center + offset up
- Particle 5: center + offset down

Default offset: 0.25 × L = 0.5 mm (configurable via `--initial_offset`)

**Motion constraints:**
- **2D midplane only:** z = H/2 fixed (no vertical motion yet)
- **Speed clamp:** max 10 mm/s (prevents numerical blowup)
- **Wall margin:** 50 μm minimum distance from boundaries
- **Sub-stepping:** 10 substeps per macro step for accuracy (configurable via `--n_substeps`)

### Time Integration

**Macro time steps:**
- User specifies: `--T_total` (total time) and `--n_steps` (number of steps)
- Macro dt = T_total / n_steps
- Helmholtz solved once per macro step (quasi-static assumption)

**Sub-stepping:**
- Particle motion integrated with smaller dt = (macro dt) / n_substeps
- Force field remains constant during substeps (quasi-static)
- Default: 10 substeps per macro step

**Example timing:**
```
T_total = 1.0 s, n_steps = 20
  → macro dt = 0.05 s
  → 10 substeps → substep dt = 0.005 s
  → 21 total Helmholtz solves (t = 0, 0.05, 0.1, ..., 1.0)
```

## Outputs

### Directory Structure
```
results/phase2_{schedule}/run_{timestamp}/
├── config.json               # Full configuration
├── time_evolution.csv        # Per-step diagnostics (rows)
├── time_evolution.json       # Complete diagnostics (nested)
├── pressure_step_0000.png    # |p| at t=0
├── gorkov_step_0000.png      # U at t=0
├── pressure_step_0001.png    # |p| at next saved step
├── gorkov_step_0001.png      # U at next saved step
└── ...
```

### CSV Columns
Each row contains one time step:
```
step, time,
phi_left, phi_right, phi_front, phi_back,
max_p, mean_p, l2_p,
deepest_U, trap_depth,
max_particle_speed, speed_clamp_triggered,
x1, y1, x2, y2, x3, y3, x4, y4, x5, y5
```

### JSON Structure
```json
{
  "config": { ... },
  "diagnostics": [
    {
      "step": 0,
      "time": 0.0,
      "phases": {"left": 0.0, "right": 0.0, ...},
      "field": {"max_p": 1.23e7, "mean_p": 2.56e6, "l2_p": 1.45e9},
      "gorkov": {"deepest_U": -7.73e-11, "trap_depth": 1.79e-9},
      "particles": {
        "positions": [[0.001, 0.001], ...],
        "max_speed": 0.0023,
        "speed_clamped": false
      }
    },
    ...
  ],
  "particle_summary": {
    "total_wall_hits": 0,
    "total_speed_clamps": 2
  }
}
```

### Visualizations
Each saved frame generates 2 PNGs:

**pressure_step_NNNN.png:**
- Midplane |p| contour plot
- Red circles: current particle positions
- Colorbar: pressure magnitude (Pa)

**gorkov_step_NNNN.png:**
- Midplane Gor'kov potential U
- Red circles: current particle positions
- Colorbar: potential energy (J)
- Minima (dark blue) indicate stable traps

## Usage Examples

### Example 1: Short Step Schedule Test
```bash
python scripts/phase2_time_evolution.py \
  --schedule step_lr \
  --T_total 0.5 \
  --n_steps 10 \
  --save_every 2
```

**What it does:**
- Runs step_lr schedule for 0.5 seconds
- 10 macro steps → dt = 0.05 s
- Saves plots every 2 steps → 6 PNG pairs
- Particles start in uniform field, then jump to L-R opposite at t=0.25s

**Expected runtime:** ~2-3 minutes

### Example 2: Ramp to Quadrature (Default)
```bash
python scripts/phase2_time_evolution.py \
  --schedule ramp_quadrature \
  --T_total 1.0 \
  --n_steps 20 \
  --n_substeps 10 \
  --save_every 1
```

**What it does:**
- Smoothly transitions from [0,0,0,0] to [0,π/2,π,3π/2] over 1 second
- 20 macro steps → dt = 0.05 s per step
- 10 substeps → particle dt = 0.005 s
- Saves all 21 frames (every step)

**Expected runtime:** ~5-7 minutes  
**Output:** 42 PNG images + CSV + JSON

### Example 3: Sine Push-Pull with Custom Particles
```bash
python scripts/phase2_time_evolution.py \
  --schedule sine_pushpull \
  --T_total 2.0 \
  --n_steps 40 \
  --save_every 4 \
  --particle_radius 50e-6 \
  --initial_offset 0.3
```

**What it does:**
- Sinusoidal left-right oscillation over 2 seconds
- 40 steps → dt = 0.05 s
- Larger particles (50 μm instead of 40 μm)
- Wider initial spread (0.6 mm instead of 0.5 mm)
- Saves every 4th step → 11 PNG pairs

**Expected runtime:** ~10-12 minutes

### Example 4: High-Resolution Short Run
```bash
python scripts/phase2_time_evolution.py \
  --schedule ramp_quadrature \
  --T_total 0.2 \
  --n_steps 10 \
  --elements_per_wavelength 16 \
  --save_every 1
```

**What it does:**
- Short 0.2 s run with finer mesh (16 elem/λ vs default 12)
- Better force field accuracy but slower
- Good for convergence testing

**Expected runtime:** ~8-10 minutes (finer mesh)

## Configuration Parameters

### Time Evolution
| Parameter | Flag | Default | Description |
|-----------|------|---------|-------------|
| Schedule | `--schedule` | `ramp_quadrature` | Phase schedule type |
| Total time | `--T_total` | 1.0 | Simulation duration (s) |
| Macro steps | `--n_steps` | 20 | Number of Helmholtz solves |
| Substeps | `--n_substeps` | 10 | Particle integration substeps per macro |
| Save frequency | `--save_every` | 1 | Save plots every N steps |

### Particles
| Parameter | Flag | Default | Description |
|-----------|------|---------|-------------|
| Radius | `--particle_radius` | 40e-6 | Particle radius (m) |
| Initial offset | `--initial_offset` | 0.25 | Cross pattern offset (fraction of L) |

### Mesh
| Parameter | Flag | Default | Description |
|-----------|------|---------|-------------|
| Resolution | `--elements_per_wavelength` | 12.0 | Mesh density |

## Physics Parameters (Fixed in Code)

### Geometry
- Domain: 2 mm × 2 mm × 2 mm square dish
- Particle motion: midplane z = 1 mm only

### Acoustics
- Frequency: 2 MHz
- Water: ρ=997 kg/m³, c=1497 m/s
- Actuation velocity: v₀ = 1 mm/s

### Impedance BCs
- Bottom (polystyrene): Z_b = 2.468 MPa·s/m
- Top (air): Z_a = 411.6 Pa·s/m

### Particles (Polystyrene)
- Density: 1050 kg/m³
- Sound speed: 2350 m/s
- Gor'kov contrast factors: f₁=0.464, f₂=0.034

### Viscosity
- Water: η = 8.9×10⁻⁴ Pa·s

## Validation Checks

**Healthy run indicators:**
1. ✅ No NaN in CSV columns
2. ✅ Particle positions remain inside domain (> margin, < L-margin)
3. ✅ Speed clamp triggers ≤ 5% of substeps
4. ✅ Wall hits = 0 (or very few)
5. ✅ max_particle_speed < 5 mm/s typical (< 10 mm/s always)
6. ✅ PNG shows particles moving toward/along trap minima (blue regions)

**Warning signs:**
- ⚠️ Frequent speed clamping → reduce dt or check force magnitude
- ⚠️ Particles stuck at walls → increase wall_margin or adjust initial_offset
- ⚠️ No visible motion → check phase schedule amplitudes
- ⚠️ Erratic motion → increase n_substeps or reduce T_total

## Efficiency Notes

**Computational cost:**
- Helmholtz solve: ~5-15 seconds per step (depends on mesh)
- Particle integration: negligible (< 0.1 s per step)
- Plot generation: ~2 s per frame

**Bottleneck:** Helmholtz assembly and solve

**Optimization tips:**
1. Use coarser mesh (e.g., `--elements_per_wavelength 10`) for prototyping
2. Increase `--save_every` to reduce PNG overhead
3. Reduce `--n_steps` for quick tests
4. Consider parallel execution (not yet implemented)

**Current state:**
- Matrix reassembled every macro step (phases change BCs)
- No factorization reuse between steps
- Future: exploit BC-only changes to reuse LU factorization

## Next Steps (Phase 3 and Beyond)

**Planned enhancements:**
1. 3D particle motion (enable z-dynamics)
2. Particle-particle interactions (hard-sphere or hydrodynamic)
3. Adaptive time stepping based on force magnitude
4. Custom schedule specification (JSON file input)
5. Multi-objective optimization of phase sequences
6. Real-time visualization (GIF/video generation)

## Troubleshooting

### Issue: Particles fly off immediately
**Cause:** Force field too strong or dt too large  
**Fix:** Reduce `--T_total` or increase `--n_steps` (smaller dt)

### Issue: Particles don't move
**Cause:** Phase schedule has zero gradients or particles already at minima  
**Fix:** Check schedule amplitudes; try different `--initial_offset`

### Issue: Solver fails to converge
**Cause:** Mesh too coarse or phase values causing singularity  
**Fix:** Increase `--elements_per_wavelength`; check phase schedule validity

### Issue: Speed clamp triggered every step
**Cause:** Normal for first few steps if particles start far from equilibrium  
**Fix:** If persists beyond ~5 steps, reduce dt or check force computation

### Issue: Wall hits accumulate
**Cause:** Initial positions too close to boundaries  
**Fix:** Reduce `--initial_offset` or increase `wall_margin` in code

## References

**Gor'kov potential:**
- L.P. Gor'kov, "On the forces acting on a small particle in an acoustical field in an ideal fluid," Sov. Phys. Dokl. 6, 773 (1962)

**Overdamped dynamics:**
- Stokes drag: F_drag = 6πηav
- Reynolds number Re = ρvd/η << 1 for microparticles

**FEniCSx documentation:**
- https://fenicsproject.org/

## File Locations

**Main script:** [scripts/phase2_time_evolution.py](../scripts/phase2_time_evolution.py)  
**Phase 1 reference:** [scripts/square_dish_phase_control.py](../scripts/square_dish_phase_control.py)  
**Diagnostics utilities:** [scripts/diagnostics_utils.py](../scripts/diagnostics_utils.py)  
**Phase 1.5 docs:** [PHASE1_5_DIAGNOSTICS.md](PHASE1_5_DIAGNOSTICS.md)

---

**Author:** Acousto-Tweezers Project  
**Last updated:** February 6, 2026
