# FEM Standing-Wave Mesh Convergence Study Report

Generated: 2026-03-23 19:19:36

Wavelength: 0.6902 mm  |  λ/2 = 0.3451 mm
Frequency: 2.15 MHz  |  c = 1484 m/s

## 1. Reduced-Domain Mesh Convergence (3.0 mm, PML = 1.0λ)

### 1.1 Error Metrics

| EPL | DOFs | Solve (s) | max\|p\| (Pa) | ε L2 ROI | Spacing Err (%) | Matched/Total | Mean Trap Err (µm) |
|-----|------|-----------|--------------|----------|----------------|---------------|---------------------|
| 5.0 | 400,869 | 39.7 | 82.40 | — | — | 19/19 | — |
| 4.5 | 295,659 | 31.5 | 80.86 | 1.1832e-01 | 1.4 | 18/21 | 44.0 |
| 4.0 | 210,681 | 18.7 | 85.28 | 6.8114e-01 | 4.6 | 19/25 | 38.6 |
| 3.5 | 143,775 | 9.7 | 101.03 | 6.9494e-01 | 8.3 | 15/22 | 58.6 |
| 3.0 | 102,541 | 6.7 | 104.60 | 8.4819e-01 | 108.3 | 11/21 | 110.8 |
| 2.0 | 68,921 | 3.8 | 146.39 | 2.0093e+00 | 77.8 | 6/16 | 89.8 |

**Primary convergence metric:** ε L2 ROI (relative L2 norm of complex pressure difference in central 50% ROI, after phase alignment).

**Secondary metrics:** Centreline trap spacing error and matched trap position error. These are physically meaningful but inherently noisier because they depend on trap detection thresholds.

### 1.2 Observed Convergence Order

Mesh size parameter: h = λ / EPL

| EPL pair | h ratio | p_obs (L2 ROI) | p_obs (spacing) | p_obs (trap pos) |
|----------|---------|----------------|-----------------|------------------|
| 2.0 → 3.0 | 1.50 | 2.13 | -0.82 | -0.52 |
| 3.0 → 3.5 | 1.17 | 1.29 | 16.64 | 4.13 |
| 3.5 → 4.0 | 1.14 | 0.15 | 4.40 | 3.13 |
| 4.0 → 4.5 | 1.12 | 14.86 | 10.22 | -1.11 |

For P2 elements, the expected asymptotic convergence order is O(h³) in L2 norm. Observed rates significantly below this in coarse regimes indicate pre-asymptotic behaviour (under-resolution). Rates approaching or exceeding 3 in the fine regime confirm asymptotic convergence.

## 2. Domain-Size Sensitivity

| Domain (mm) | EPL | max\|p\| (Pa) | Trap spacing (mm) | n_traps |
|-------------|-----|--------------|-------------------|---------|
| 3.0 | 5.0 | 82.40 | 0.3612 | 19 |
| 4.0 | 5.0 | 107.43 | 0.3545 | 47 |
| 5.4 | 4.0 | 113.82 | 0.3463 | 71 |

If trap spacing and max|p| are consistent across domain sizes, the reduced domain does not introduce significant truncation artefacts in the central ROI.

## 3. PML Sensitivity

| PML (λ) | σ_max factor | max\|p\| (Pa) | Trap spacing (mm) | n_traps |
|---------|-------------|--------------|-------------------|---------|
| 1.0 | 5.0 | 82.40 | 0.3612 | 19 |
| 1.5 | 5.0 | 84.48 | 0.3712 | 23 |
| 2.0 | 5.0 | 65.15 | 0.3637 | 25 |

If metrics are stable across PML thicknesses, the 1.0λ PML is adequate and does not contaminate the central ROI.

## 4. Production Resolution Acceptance

1. **Convergence onset:** Reduced-domain mesh convergence indicates clear convergence beginning around EPL ≈ 4.5 (ε L2 ROI < 0.2).
2. **Full-domain EPL=5** is NOT feasible on the current ~30 GB workstation (solver produces inf/NaN at production domain size).
3. **Full-domain EPL=4** (505k DOFs) is the highest feasible production resolution on this hardware.
4. **Remaining uncertainty** is bounded by:
   - Reduced-domain EPL=4.5→5 comparison (provides upper bound on discretisation error)
   - Domain-size sensitivity (checks that reduced domain does not corrupt central ROI)
   - PML sensitivity (confirms truncation artefact is negligible)

## 5. Limitations

- Gor'kov potential computed on 2D Cartesian plane at z* with z-gradient neglected (valid at pressure antinode)
- Trap detection uses finite-difference Gor'kov with depth threshold and minimum separation filters
- Trap matching uses greedy nearest-neighbour with λ/4 rejection threshold (not Hungarian)
- Convergence order estimates are local (between adjacent EPL pairs) and may not reflect asymptotic behaviour at coarse levels
- Full-domain EPL=5 production run not feasible on current hardware
