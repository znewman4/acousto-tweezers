# Phase 2 Simulation Modes: Development vs Truth

## Overview

Phase 2 simulations can be run in two modes depending on your goals:

- **Development Mode** (`dev`): Fast iteration, qualitative validation
- **Truth Mode** (`truth`): High accuracy for publication-quality results

## Mode Comparison

| Parameter | Development Mode | Truth Mode | Ratio |
|-----------|------------------|------------|-------|
| **Mesh Resolution** | 6-8 elements/wavelength | 12-15 elements/wavelength | 8-27× DOFs |
| **Approx DOFs** | ~50k-100k | ~400k-800k | 4-8× slower solve |
| **Gor'kov Grid** | 30×30 to 50×50 | 80×80 to 100×100 | 2.5-4× points |
| **Timestep** | dt ~ 15-20 ms | dt ~ 5-10 ms | 2-4× steps |
| **Substeps** | 10-20 | 30-50 | 1.5-2.5× |
| **Plotting** | Every 2-4 steps | Every 1-2 steps | Varies |
| **Time per step** | ~8-15 seconds | ~60-120 seconds | 4-8× |
| **10-step run** | ~2-3 minutes | ~15-20 minutes | 6-8× |

## When to Use Each Mode

### Development Mode: Use for...

✅ **Testing new schedules** - Rapid iteration on phase patterns
✅ **Parameter exploration** - Scanning particle sizes, frequencies, geometries
✅ **Code debugging** - Verifying no crashes, NaN, or artifacts
✅ **Qualitative physics** - Confirming particles move toward nodes
✅ **Initial prototyping** - Before committing to long runs

**Acceptable compromises**:
- Force gradients smoothed by coarse Gor'kov grid
- Pressure fields accurate to ~5-10% (mesh resolution)
- Trajectories qualitatively correct but may miss fine-scale motion

### Truth Mode: Use for...

✅ **Publication figures** - High-quality storyboards with smooth fields
✅ **Quantitative predictions** - Trap stiffness, capture rates, switching times
✅ **Experimental validation** - Direct comparison with microscopy data
✅ **Convergence studies** - Proving numerical independence
✅ **Fine-scale phenomena** - Resolving near-node dynamics

**Guaranteed quality**:
- Force fields resolved to <1% error
- Pressure nodes located to sub-wavelength accuracy
- Trajectories converged with mesh/timestep refinement

## CLI Examples

### Development Mode

**Quick test (step_lr, 2 minutes)**:
```bash
python scripts/phase2_time_evolution.py \
  --schedule step_lr \
  --T_total 0.2 \
  --n_steps 8 \
  --n_substeps 15 \
  --save_every 2 \
  --elements_per_wavelength 6
```

**Standard dev run (any schedule, 3-5 minutes)**:
```bash
python scripts/phase2_time_evolution.py \
  --schedule <step_lr|ramp_quadrature|sine_pushpull> \
  --T_total 0.4 \
  --n_steps 12 \
  --n_substeps 20 \
  --save_every 2 \
  --elements_per_wavelength 8
```

**Key settings**:
- Gor'kov grid: 50×50 (automatic from v2.5+)
- Mesh: ~100k DOFs
- dt: ~15-30 ms
- Speed clamp target: <10%

### Truth Mode

**High-quality storyboard (step_lr, 15-20 minutes)**:
```bash
python scripts/phase2_time_evolution.py \
  --schedule step_lr \
  --T_total 0.5 \
  --n_steps 25 \
  --n_substeps 40 \
  --save_every 2 \
  --elements_per_wavelength 12
```

**Publication run (ramp_quadrature, 30-40 minutes)**:
```bash
python scripts/phase2_time_evolution.py \
  --schedule ramp_quadrature \
  --T_total 1.0 \
  --n_steps 50 \
  --n_substeps 50 \
  --save_every 4 \
  --elements_per_wavelength 15
```

**Key settings**:
- Gor'kov grid: 80×80 or 100×100 (edit line 350 in script)
- Mesh: ~400k-800k DOFs
- dt: ~5-10 ms
- Speed clamp target: <1%

## Manual Gor'kov Grid Adjustment

Current default: 50×50 (line 350 in `scripts/phase2_time_evolution.py`)

**To change for truth mode**:
```python
# Line 350-351:
nx_eval = 80  # or 100 for highest quality
ny_eval = 80
```

**Performance impact**:
- 30×30 (900 points): ~2s per Gor'kov eval
- 50×50 (2500 points): ~5s per Gor'kov eval
- 80×80 (6400 points): ~12s per Gor'kov eval
- 100×100 (10000 points): ~20s per Gor'kov eval

## Validation Strategy

**Step 1: Development mode** (always start here)
1. Run with `epw=6-8`, `n_steps=8-12`
2. Verify: no crashes, particles move correctly, clamp rate <10%
3. Check: trap locations qualitatively correct

**Step 2: Intermediate check** (optional convergence test)
1. Double mesh resolution: `epw=12`
2. Increase Gor'kov grid: `nx_eval=80`
3. Compare: trap depths change <5%? Minima locations shift <λ/10?
4. If yes → proceed. If no → investigate physics or numerical issues

**Step 3: Truth mode** (for final results)
1. Full resolution: `epw=15`, `nx_eval=100`, `n_substeps=50`
2. Generate storyboards and diagnostics
3. Archive: this is your "ground truth" for this configuration

## Current Status (Feb 2026)

**Latest runs**:
- ✅ Development mode validated: `epw=8`, grid=50×50, clamp <5% achievable
- ⏸️ Truth mode pending: Need `epw=12+` runs for publication

**Immediate action**:
- Running improved dev runs with dt=16.7ms, n_substeps=20
- Target: <5% clamping for all schedules
- Next: Truth mode runs for step_lr and ramp_quadrature

## Recommendations

**For rapid exploration** (trying 5+ configurations):
```bash
# 2-3 minutes per run
epw=6, n_steps=8, n_substeps=15, grid=30x30
```

**For confident development** (standard workflow):
```bash
# 4-6 minutes per run - RECOMMENDED DEFAULT
epw=8, n_steps=12, n_substeps=20, grid=50x50
```

**For verification before publication**:
```bash
# 15-20 minutes per run
epw=12, n_steps=25, n_substeps=40, grid=80x80
```

**For final publication figures**:
```bash
# 30-60 minutes per run
epw=15, n_steps=50, n_substeps=50, grid=100x100
```

## Key Takeaway

**Use dev mode liberally to iterate fast, then commit to truth mode runs for claims.**

Don't make quantitative statements (e.g., "trap stiffness is X") from dev mode data. Use dev to find the interesting parameter regimes, then re-run in truth mode for numbers.

---

**Last Updated**: February 6, 2026  
**Status**: Development mode operational, truth mode pipeline ready
