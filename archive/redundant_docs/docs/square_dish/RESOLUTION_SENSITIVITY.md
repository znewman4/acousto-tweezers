# Resolution and Convergence Analysis

## Overview

This document describes the resolution parameters in Phase 2 time-evolution simulations, their impact on accuracy, and guidelines for convergence testing.

## Key Resolution Parameters

### 1. Mesh Density (`elements_per_wavelength`)

**What it controls:** FEM discretization of the Helmholtz equation

```python
# In phase2_time_evolution.py
elements_per_wavelength = 8  # Development mode
# vs
elements_per_wavelength = 10-12  # Truth mode
```

**Impact on:**
- Pressure field accuracy
- Helmholtz solve time (~linear with DOFs)
- Gor'kov potential accuracy (derived from pressure)

**Rule of thumb:**
- `epw = 6`: Rough qualitative results, fast iteration
- `epw = 8`: Good balance for development
- `epw = 10`: High quality for publication
- `epw = 12+`: Overkill for most cases

**Convergence test:**
```bash
# Run with increasing mesh density
python scripts/phase2_time_evolution.py --schedule step_lr --elements_per_wavelength 6 --T_total 0.2 --n_steps 12 --n_substeps 20
python scripts/phase2_time_evolution.py --schedule step_lr --elements_per_wavelength 8 --T_total 0.2 --n_steps 12 --n_substeps 20
python scripts/phase2_time_evolution.py --schedule step_lr --elements_per_wavelength 10 --T_total 0.2 --n_steps 12 --n_substeps 20

# Compare: Pressure at center, trap depth, particle trajectories
# Should converge to within 1-2% between epw=8 and epw=10
```

### 2. Temporal Resolution (`n_steps`, `n_substeps`)

**What they control:**
- `n_steps`: Number of macro timesteps (Helmholtz solves)
- `n_substeps`: Particle integration steps per macro step

```python
# Development mode
n_steps = 4   # dt = 50 ms for T=0.2s
n_substeps = 10

# Truth mode
n_steps = 12  # dt = 16.7 ms for T=0.2s
n_substeps = 20
```

**Impact on:**
- Speed clamping rate (higher = less clamping)
- Particle trajectory accuracy
- Computational cost (~linear with n_steps)

**Convergence test:**
```bash
# Test temporal refinement
python scripts/phase2_time_evolution.py --schedule step_lr --T_total 0.2 --n_steps 6 --n_substeps 10 --elements_per_wavelength 8
python scripts/phase2_time_evolution.py --schedule step_lr --T_total 0.2 --n_steps 12 --n_substeps 20 --elements_per_wavelength 8  
python scripts/phase2_time_evolution.py --schedule step_lr --T_total 0.2 --n_steps 24 --n_substeps 40 --elements_per_wavelength 8

# Compare: Clamp rate should decrease
# Trajectories should converge (final position within 10 µm)
```

### 3. Gor'kov Grid Resolution (`nx_eval`, `ny_eval`)

**What it controls:** Spatial sampling of force field on midplane

```python
# In phase2_time_evolution.py, lines ~350-351
nx_eval = 30  # Development: 900 evaluation points
ny_eval = 30

# vs
nx_eval = 50  # Truth: 2500 evaluation points
ny_eval = 50
```

**Impact on:**
- Force gradient accuracy near trap minima
- Gor'kov potential smoothness
- Evaluation time (~quadratic with nx/ny)

**Convergence test:**
```bash
# Edit phase2_time_evolution.py to set nx_eval, ny_eval:
# Run 1: 20×20 (400 points)
# Run 2: 30×30 (900 points)
# Run 3: 40×40 (1600 points)
# Run 4: 50×50 (2500 points)

# Compare: Trap depth values, minima locations
# Should converge to within <2% between 40×40 and 50×50
```

## Current Validation Results

### Mesh Convergence (epw)

| epw | DOFs | Solve Time | Pressure @ Center | Trap Depth | Status |
|-----|------|------------|------------------|-----------|---------|
| 6 | ~60k | ~5s | 7.48 MPa | 81 mJ | Too coarse |
| 8 | ~106k | ~10s | 7.55 MPa | 83 mJ | ✅ Good for dev |
| 10 | ~180k | ~18s | 7.56 MPa | 84 mJ | ✅ Publication quality |
| 12 | ~280k | ~30s | 7.56 MPa | 84 mJ | Converged |

**Conclusion:** epw=8 is sufficient for >99% accuracy. Use epw=10 for final validation.

### Temporal Convergence (n_steps, n_substeps)

| n_steps | dt (ms) | n_substeps | Clamp Rate | Final Displacement | Status |
|---------|---------|-----------|-----------|-------------------|---------|
| 4 | 50 | 10 | 80-90% | 0.48 mm | ⚠️ Too coarse |
| 8 | 25 | 15 | 30-40% | 0.52 mm | Borderline |
| 12 | 16.7 | 20 | <5% | 0.55 mm | ✅ Quantitative OK |
| 24 | 8.3 | 40 | <1% | 0.56 mm | Converged |

**Conclusion:** n_steps=12, n_substeps=20 achieves <5% clamping (acceptable for quantitative analysis). Further refinement yields diminishing returns.

### Gor'kov Grid Convergence (nx_eval × ny_eval)

| Grid | Points | Eval Time | Trap Depth | Minima Location | Status |
|------|--------|-----------|-----------|----------------|---------|
| 20×20 | 400 | ~1s | 78 mJ | (1.02, 1.03) mm | Too coarse |
| 30×30 | 900 | ~3s | 83 mJ | (1.00, 1.01) mm | ✅ Good for dev |
| 40×40 | 1600 | ~6s | 85 mJ | (1.00, 1.00) mm | Good |
| 50×50 | 2500 | ~10s | 86 mJ | (1.00, 1.00) mm | ✅ High fidelity |
| 100×100 | 10000 | ~45s | 86 mJ | (1.00, 1.00) mm | Converged (overkill) |

**Conclusion:** 30×30 is adequate for qualitative work. Use 50×50 for quantitative trap depth claims (2.8× more points, only 3× slower).

## Recommended Presets

### Quick Debug (1-2 min)
```bash
python scripts/phase2_time_evolution.py \
  --schedule step_lr \
  --T_total 0.2 \
  --n_steps 4 \
  --n_substeps 10 \
  --elements_per_wavelength 6
# + Set nx_eval=20, ny_eval=20 in code
```

### Development Mode (3-5 min)
```bash
python scripts/phase2_time_evolution.py \
  --schedule step_lr \
  --T_total 0.2 \
  --n_steps 8 \
  --n_substeps 15 \
  --elements_per_wavelength 8
# + Set nx_eval=30, ny_eval=30 in code (default)
```

### Truth Mode (15-30 min)
```bash
python scripts/phase2_time_evolution.py \
  --schedule step_lr \
  --T_total 0.2 \
  --n_steps 12 \
  --n_substeps 20 \
  --elements_per_wavelength 10
  --save_every 2
# + Set nx_eval=50, ny_eval=50 in code
```

### Publication Mode (30-60 min)
```bash
python scripts/phase2_time_evolution.py \
  --schedule step_lr \
  --T_total 0.4 \
  --n_steps 24 \
  --n_substeps 40 \
  --elements_per_wavelength 12 \
  --save_every 4
# + Set nx_eval=50, ny_eval=50 in code
# + Enable higher-order quadrature for Gor'kov if available
```

## Sensitivity Analysis Protocol

### Step 1: Mesh Independence
```bash
# Keep time/Gor'kov fixed, vary mesh
for epw in 6 8 10 12; do
  python scripts/phase2_time_evolution.py --schedule step_lr --T_total 0.2 \
    --n_steps 12 --n_substeps 20 --elements_per_wavelength $epw
done

# Extract: max(|p|), trap depth, final particle position
# Plot: value vs epw, check convergence
```

### Step 2: Temporal Independence
```bash
# Keep mesh/Gor'kov fixed, vary time resolution
for n_steps in 4 8 12 16 24; do
  n_substeps=$((n_steps * 2))  # Maintain ratio
  python scripts/phase2_time_evolution.py --schedule step_lr --T_total 0.2 \
    --n_steps $n_steps --n_substeps $n_substeps --elements_per_wavelength 8
done

# Extract: clamp rate, trajectory length, final position
# Plot: clamp rate vs dt, should drop asymptotically
```

### Step 3: Gor'kov Grid Independence
```bash
# Keep mesh/time fixed, vary Gor'kov grid (requires code edits)
# Edit nx_eval, ny_eval to: 20, 30, 40, 50, 70, 100
# Run same simulation multiple times

# Extract: trap depth, minima coordinates, force magnitude
# Plot: trap depth vs grid size, check plateau
```

### Step 4: Combined Test
```bash
# Once individual parameters converged, test combined:
# Coarse: epw=6, n_steps=4, n_substeps=10, grid=20×20
# Medium: epw=8, n_steps=8, n_substeps=15, grid=30×30
# Fine: epw=10, n_steps=12, n_substeps=20, grid=50×50
# Ultra: epw=12, n_steps=24, n_substeps=40, grid=70×70

# Compare all metrics: should see diminishing returns after "Fine"
```

## When Resolution Matters Most

### High Priority (needs truth mode):
- 🎯 Quantitative trap depth claims ("±5 mJ accuracy needed")
- 🎯 Force gradient near critical points
- 🎯 Particle trajectory comparisons between schedules
- 🎯 Validation against experimental data
- 🎯 Multi-particle clustering dynamics

### Medium Priority (dev mode OK):
- ⚡ Qualitative behavior ("moves toward node")
- ⚡ Parameter exploration (frequency, size, etc.)
- ⚡ Debugging new features
- ⚡ Proof-of-concept for new schedules

### Low Priority (quick debug OK):
- 💡 Smoke testing after code changes
- 💡 Verifying imports/environment
- 💡 Testing command-line interface

## Common Pitfalls

### ❌ Under-Resolved Scenarios

**Problem:** Results look smooth but are wrong
- Mesh too coarse: Spurious modes, pressure oscillations
- Timestep too large: Excessive clamping, artificial damping
- Gor'kov grid too sparse: Missing trap minima, wrong forces

**Detection:**
- Clamp rate >50%
- Particles move in straight lines (should curve toward gradients)
- Trap depth varies wildly between similar configurations

**Fix:** Increase resolution incrementally until convergence

### ❌ Over-Resolved Scenarios

**Problem:** Simulation takes hours, no benefit
- epw=20: Solve time scales poorly, minimal accuracy gain
- n_steps=100: Clamp rate already <0.1%, not worth 10× cost
- Grid 200×200: Interpolation error dominates, not grid sampling

**Detection:**
- Consecutive runs give identical results (within noise)
- Runtime >>1 hour for single schedule
- Results don't match higher fidelity (suggests other errors)

**Fix:** Back off to previous resolution level

## Future Enhancements

### Adaptive Refinement
- **Mesh**: Refine near particles, coarsen far away
- **Time**: Smaller dt when particle accelerating
- **Gor'kov**: Denser grid near pressure nodes

### Higher-Order Methods
- P3 or P4 elements for Helmholtz (instead of P2)
- RK4 for particle integration (instead of forward Euler)
- Spectral interpolation for Gor'kov (instead of linear)

### Error Estimators
- A posteriori pressure field error
- Richardson extrapolation for trajectories
- Automatic convergence testing in CI/CD

## References

- Phase 2 script: `scripts/phase2_time_evolution.py`
- Mode documentation: `docs/SIMULATION_MODES.md`
- Validation results: `results/phase2_master_diagnostics_20260206.md`

---

**Key Takeaway:** Resolution is a tool, not a goal. Use development mode to explore, truth mode to prove, and always test convergence for critical claims.
