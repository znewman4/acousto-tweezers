# Validation Guide

This document explains how to run and interpret the regression/verification
tests for the acousto-tweezers solver stack.

## Prerequisites

All tests require the `acousto-complex` conda/micromamba environment, which
provides DOLFINx built against a complex-scalar PETSc.

```bash
micromamba activate acousto-complex
# or prefix every command with: micromamba run -n acousto-complex python ...
```

The package must be installed in editable mode:

```bash
pip install -e .
```

---

## Core Tests

### 1. Environment Gate

```bash
python scripts/validation/test_env_complex_petsc.py
```

Confirms that `PETSc.ScalarType` is `numpy.complex128`.  If this fails,
no other test will produce meaningful results.

### 2. 1D Impedance Reflection

```bash
python scripts/validation/test_1d_impedance.py
```

Solves a 1D Helmholtz problem (velocity source on left, impedance BC on right)
and measures the reflection coefficient.

| Sub-test | Expected |R| | What it validates |
|----------|-----------|-------------------|
| Matched impedance (Z = ρc) | ≈ 0 | Robin coefficient α = −iωρ/Z is correct |
| Rigid wall (∂p/∂n = 0) | 1.0 | Natural Neumann gives perfect reflection |
| Wrong sign (+iωρ/Z) | ≫ 0 | Demonstrates sign error is detectable |
| Old code (−iω/Z, missing ρ) | ≈ 1 | Demonstrates old bug is detectable |

**Tests 1 and 2 must pass.  Tests 3 and 4 are expected to fail** — they serve
as negative controls showing what broken BCs look like.

### 3. Energy / Power Balance

```bash
python scripts/validation/test_energy_balance.py
```

Verifies that the time-averaged power injected by the velocity source equals
the power absorbed at impedance boundaries, to 14-digit precision.

### 4. Petri-Dish BC Smoke Test

```bash
python scripts/validation/test_petri_dish_bcs.py
```

Verifies the petri-dish boundary-condition model:

- Bottom facet segmentation produces nonzero disc and rigid facet counts.
- Standing-mode peak pressure is ≫ 0.5 Pa (rigid reflecting walls form a
  resonant cavity).
- Vortex and combined modes produce nonzero pressure.

### 5. Full Suite

```bash
python scripts/validation/run_all_tests.py
```

Runs all available validation tests in sequence and reports a pass/fail
summary.

---

## Supplementary Tests

| Test | Script | What It Checks |
|------|--------|----------------|
| Complex backend | `test_complex_backend.py` | DOLFINx complex assembly |
| 2D Helmholtz | `test_2d_helmholtz.py` | Convergence on circular domain |
| FEM modules | `test_fem_modules.py` | Config, domains, materials, geometry |
| Acoustics only | `test_acoustics_only.py` | ACOUSTICS_ONLY physics level |
| Acoustics smoke | `test_acoustics_smoke.py` | RHS norm, mesh, solution nonzero |
| Helmholtz complex | `test_helmholtz_complex.py` | End-to-end complex solve |
| Interface continuity | `test_interface_continuity.py` | Smooth solution |
| Fluid–solid coupled | `test_fluid_solid_coupled.py` | Coupling interface |
| Vortex lens | `validate_vortex_lens.py` | Core null, phase winding |
| PML reflection | `test_pml_reflection_fit.py` | |R| < 1 % |
| PML absorption | `test_pml_absorption.py` | Amplitude decay |
| PML smoke | `test_pml_smoke.py` | PML assembly works |
| Streaming smoke | `test_streaming_stokes_smoke.py` | Stokes solver works |
| Rigid vs absorbing | `test_rigid_vs_absorbing.py` | Mode comparison |
| Vortex simple | `vortex_simple.py` | Dirichlet-BC vortex |

---

## Interpreting Failures

- **1D impedance test fails on sub-test 1 or 2:** the Robin BC coefficient
  has been incorrectly modified.  Check the sign and scaling of `alpha` in
  `solve_pressure.py`.
- **Energy balance fails:** either the bilinear form or the RHS assembly has
  a bug that breaks power conservation.
- **Petri-dish test fails on standing pressure:** side walls may have been
  given an impedance Robin term (absorbing instead of reflecting).
- **Streaming tests fail:** check that the MUMPS solver is available and that
  the nullspace is correctly constructed.
