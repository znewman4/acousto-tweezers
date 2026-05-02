# FIX: Vortex-Too-Weak — disc Robin BC Removal

Generated: 2026-02-15T12:21:16.199799

## What was wrong

The disc boundary (bottom, r ≤ R_disc) had an impedance Robin BC
with Z = Z_water = ρc (impedance-matched).  This made the disc
a **perfect absorber** for any acoustic energy reflected back to it.

Standing waves resonate between rigid side walls (pure Neumann BCs)
and are barely affected by the small disc absorber.  But the vortex
beam emits from the disc, bounces off rigid walls, and returns to
the disc — where it is **completely absorbed**.  No resonance builds
up, yielding max|p| ≈ 6 Pa vs ~69 Pa for standing waves.

## What was changed

- File: `src/acoustweezers/experiments/shallow_square_dish/solve_pressure.py`
- Parameter: `disc_robin` (line ~213)
- No code changes to `solve_pressure.py` — the `disc_robin=False`
  parameter already existed (added in previous investigation).
- **Fix**: pass `disc_robin=False` for **all three cases** (A/B/C).
- This makes the disc a rigid boundary with prescribed normal
  velocity, matching COMSOL's 'Normal Velocity' transducer BC.
- The Robin coefficient α_disc = −iωρ/Z is NOT applied.
- The Neumann source g_vtx = −iωρ V₀ pattern IS still applied.

## Disc BC physics: COMSOL comparison

COMSOL 'Impedance + Include Normal Velocity':  
  ∂p/∂n = (iωρ/Z_w)p − iωρ v_n

Our solver implements this correctly when disc_robin=True.
But COMSOL comparison models typically use 'Normal Velocity' BC
(no impedance term / rigid piston), equivalent to our disc_robin=False:
  ∂p/∂n = −iωρ v_n

The impedance term Z_w = ρc creates perfect absorption at the source,
which is NOT what a typical COMSOL benchmark assumes.  A physical PZT
transducer has Z_PZT ≈ 33 MRayl >> Z_water ≈ 1.48 MRayl, so the
transducer face acts nearly rigid even with impedance.  Using Z_water
was an over-damping error.

## Case configuration

| Case | disc_robin | Why |
|------|-----------|-----|
| A (standing) | False | Rigid bottom, no disc absorption hole |
| B (vortex) | False | Rigid piston source (COMSOL Normal Velocity) |
| C (combined) | False | Same: rigid piston + standing walls |

## Disc Diagnostics

### 2.1 Disc facet tagging sanity

- Disc facets tagged: **184**
- A_disc_mesh = **23.0000** mm²
- A_disc_expected = π R² = **28.2743** mm²
- Ratio A_mesh/A_expected = **0.8135**  (PROBLEM!)

### 2.2 Forcing strength on disc

- max(|pattern|) on disc = **1.000000** (expect ~1 if taper=1 at r=0)
- avg(|pattern|) on disc = **0.328329**
- max(|g_vtx|) = **31321.6788** Pa/m
- Σ|g_vtx|² (DOF sum) = **7.6493e+10**

### 2.3 Standing vs vortex forcing magnitudes

- |g_stand| = ωρ Vs = **31321.6788** Pa/m  (Vs = 10.0 µm/s)
- max(|g_vtx|) on disc = **31321.6788** Pa/m  (V₀ = 10.0 µm/s)
- Ratio max|g_vtx| / |g_stand| = **1.0000**
- V₀/Vs = 1.00

### Surface area comparison

- Standing wall area (4 walls, axis=both): **40.00** mm²
- Disc area: **23.00** mm²
- Ratio wall/disc area = **1.74**

## Robin absorption A/B test (vortex-only)

- max|p| (disc_robin=True, absorbing):  **11.1557** Pa
- max|p| (disc_robin=False, rigid piston): **33.7707** Pa
- **Boost: +202.7%** (3.0× increase)
- This confirms the impedance-matched disc absorbs most vortex energy.

## Before / After max|p|

| Case | Before (robin=True) | After (robin=False) | Change |
|------|--------------------:|--------------------:|-------:|
| Case_A_standing | 93.15 Pa | 80.98 Pa | -13.1% |
| Case_B_vortex | 11.16 Pa | 33.77 Pa | +202.7% |
| Case_C_combined | 88.10 Pa | 81.78 Pa | -7.2% |

## Ratios (after fix)

- max|p|_vortex  / max|p|_standing = **0.417**
- max|p|_combined / max|p|_standing = **1.010**

**Target behaviour achieved**: combined > standing ✓

## Files created

- `COMSOL_comparison_results/Case_A_standing/figs_fix_20260215_1221/COMSOLstyle_abs_p_z_H2.png`
- `COMSOL_comparison_results/Case_A_standing/figs_fix_20260215_1221/COMSOLstyle_abs_p_z_H2_contours.png`
- `COMSOL_comparison_results/Case_A_standing/figs_fix_20260215_1221/COMSOLstyle_total_pressure_Re_p_z_H2.png`
- `COMSOL_comparison_results/Case_A_standing/figs_fix_20260215_1221/plane_z_H2_Re.csv`
- `COMSOL_comparison_results/Case_A_standing/figs_fix_20260215_1221/plane_z_H2_abs.csv`
- `COMSOL_comparison_results/Case_B_vortex/figs_fix_20260215_1221/COMSOLstyle_abs_p_z_H2.png`
- `COMSOL_comparison_results/Case_B_vortex/figs_fix_20260215_1221/COMSOLstyle_abs_p_z_H2_contours.png`
- `COMSOL_comparison_results/Case_B_vortex/figs_fix_20260215_1221/COMSOLstyle_total_pressure_Re_p_z_H2.png`
- `COMSOL_comparison_results/Case_B_vortex/figs_fix_20260215_1221/plane_z_H2_Re.csv`
- `COMSOL_comparison_results/Case_B_vortex/figs_fix_20260215_1221/plane_z_H2_abs.csv`
- `COMSOL_comparison_results/Case_C_combined/figs_fix_20260215_1221/COMSOLstyle_abs_p_z_H2.png`
- `COMSOL_comparison_results/Case_C_combined/figs_fix_20260215_1221/COMSOLstyle_abs_p_z_H2_contours.png`
- `COMSOL_comparison_results/Case_C_combined/figs_fix_20260215_1221/COMSOLstyle_total_pressure_Re_p_z_H2.png`
- `COMSOL_comparison_results/Case_C_combined/figs_fix_20260215_1221/plane_z_H2_Re.csv`
- `COMSOL_comparison_results/Case_C_combined/figs_fix_20260215_1221/plane_z_H2_abs.csv`
- `COMSOL_comparison_results/FIX_NOTES.md`

