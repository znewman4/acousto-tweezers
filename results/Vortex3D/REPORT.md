# Vortex 3-D Physics Validation Report
Generated: 20260219_192955

## Solver
- Type: MUMPS direct
- DOFs: 347733
- max|p|: 17.76 Pa
- Elements/λ: 5
- Solve time: 33.9s

## Part 2: Topological Charge
- **ℓ = 1.0000** (expected: 1.0)
- Total winding: 6.2832 rad
- Status: **PASS**

## Part 3: Axial Null
- **core_ratio = 0.0707** (expected: < 0.3)
- max|p| on axis: 0.4941 Pa
- max|p| off-axis: 6.9856 Pa
- Status: **PASS**

## Part 4: Power-Flow Direction
- **mean(I_z) above source: 2.252824e-06 W/m²**
- mean(I_z) below source: 1.004055e-05 W/m²
- Fraction positive: 1.0000
- Status: **PASS**

## Part 5: PML Absorption
- **decay_ratio = 0.000990** (expected: < 0.1)
- mean|p|² inner: 1.810025e+00
- mean|p|² outer: 1.792637e-03
- Status: **PASS**

## Summary
- **PASS: 4**
- **FAIL: 0**

## Failure Conditions Checked
| Condition | Threshold | Actual | Status |
|-----------|-----------|--------|--------|
| ℓ ≈ 1 | |ℓ−1| < 0.1 | 1.0000 | PASS |
| core_ratio < 0.3 | 0.3 | 0.0707 | PASS |
| mean(I_z_above) > 0 | 0 | 2.25e-06 | PASS |
| frac positive > 0.8 | 0.8 | 1.0000 | PASS |
| decay_ratio < 0.1 | 0.1 | 0.000990 | PASS |
