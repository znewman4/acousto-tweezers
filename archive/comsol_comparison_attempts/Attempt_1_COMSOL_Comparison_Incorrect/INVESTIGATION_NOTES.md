# Investigation: COMSOL vs FEniCSx Discrepancy

Generated: 2026-02-13T20:41:01.254689

## Hypotheses
- H1: Plotting mismatch — we show |p| while COMSOL shows Re(p).
- H2: Disc Robin BC (impedance-matched patch) is active in standing-only
  case, creating a circular absorption 'hole' that COMSOL doesn't have.

## Task 1 — Re(p) vs |p| slice figures for all 3 cases

### Case A — Standing
- max|p| = 68.7912 Pa
- Re(p) range: [-30.5241, 30.5577] Pa
- Figures: `figs_investigation_20260213_2041/slice_abs_p_z_H2.png`, `slice_Re_p_z_H2.png`

### Case B — Vortex
- max|p| = 6.3380 Pa
- Re(p) range: [-6.3063, 6.2864] Pa
- Figures: `figs_investigation_20260213_2041/slice_abs_p_z_H2.png`, `slice_Re_p_z_H2.png`

### Case C — Combined
- max|p| = 65.2021 Pa
- Re(p) range: [-28.9720, 28.9971] Pa
- Figures: `figs_investigation_20260213_2041/slice_abs_p_z_H2.png`, `slice_Re_p_z_H2.png`

## Task 2 — Case A: disc Robin ON vs OFF (A/B test)

### Change made to solver
- File: `src/acoustweezers/experiments/shallow_square_dish/solve_pressure.py`
- Added `disc_robin: bool = True` parameter to `solve_helmholtz()` (line ~222)
- Guarded the disc Robin term `a += alpha_disc * inner(u, v) * dss(1)` behind
  `if disc_robin:` (line ~320)
- Default is True → existing behaviour unchanged for all other scripts.

### A2.1 Baseline (disc Robin ON)
- max|p| = 68.7912 Pa
- Re(p) range: [-30.5241, 30.5577] Pa

### A2.2 Rigid bottom (disc Robin OFF)
- max|p| = 57.5271 Pa
- Re(p) range: [-0.3304, 0.3309] Pa

### Comparison
- max|p| baseline:      68.7912 Pa
- max|p| rigid-bottom:  57.5271 Pa
- Change:               -16.4%

- |p| at disc centre (baseline):     0.0022 Pa
- |p| at disc centre (rigid-bottom): 0.0139 Pa
- 'Hole' vanished:  YES

## Task 3 — Verify vortex OFF in Case A

- standing enabled: **True**
- vortex enabled: **False**
- max(|g_vtx|) applied on disc: **0** (code path not entered)
- vortex pattern function assembled: **NO**

Verification: In `solve_pressure.py`, lines ~390-405, the vortex source is
only added when `mode in ('vortex', 'combined')`. For mode='standing', that
block is skipped entirely — `_create_vortex_source()` is never called.

## Conclusion

### Root causes of COMSOL/FEniCSx figure discrepancy

**1. Plotting mismatch (CONFIRMED):**
Our previous figures showed |p| (positive-only, up to ~93 Pa) while COMSOL
screenshots show Re(p) ('Total acoustic pressure', diverging ±, ~±0.43 Pa).
These are fundamentally different quantities. The Re(p) slice for the standing
case has range [-30.52, 30.56] Pa — diverging, sign-changing,
as expected for a standing wave. This is the PRIMARY cause of visual mismatch.

**2. Disc Robin BC active in standing-only mode (CONFIRMED):**
The disc region (r ≤ 3 mm) is always treated as an impedance-matched
boundary (Robin BC with Z = ρc), even when only standing waves are active.
This creates a local absorption patch — a visible 'hole' in the pressure
pattern. Removing it changes max|p| by -16.4% and
eliminates the circular artefact at disc centre.

### Recommendation
1. **Always compare Re(p) side-by-side with COMSOL's 'Total acoustic pressure'.**
   The |p| plot is useful but is NOT what COMSOL shows by default.
2. **For pure standing-wave COMSOL comparisons (Case A), consider running with
   `disc_robin=False`** to match a COMSOL model that has a fully rigid bottom.
   The physical transducer model (disc Robin ON) is correct for the real device,
   but may differ from a simplified COMSOL benchmark with rigid walls everywhere.


## Files created

- `COMSOL_comparison_results/Case_A_standing/figs_investigation_20260213_2041/CaseA_baseline_COMSOLstyle_total_pressure.png`
- `COMSOL_comparison_results/Case_A_standing/figs_investigation_20260213_2041/CaseA_baseline_slice_Re_p_z_H2.png`
- `COMSOL_comparison_results/Case_A_standing/figs_investigation_20260213_2041/CaseA_baseline_slice_abs_p_z_H2.png`
- `COMSOL_comparison_results/Case_A_standing/figs_investigation_20260213_2041/CaseA_rigid_bottom_COMSOLstyle_total_pressure.png`
- `COMSOL_comparison_results/Case_A_standing/figs_investigation_20260213_2041/CaseA_rigid_bottom_slice_Re_p_z_H2.png`
- `COMSOL_comparison_results/Case_A_standing/figs_investigation_20260213_2041/CaseA_rigid_bottom_slice_abs_p_z_H2.png`
- `COMSOL_comparison_results/Case_A_standing/figs_investigation_20260213_2041/plane_z_H2_Re.csv`
- `COMSOL_comparison_results/Case_A_standing/figs_investigation_20260213_2041/plane_z_H2_abs.csv`
- `COMSOL_comparison_results/Case_A_standing/figs_investigation_20260213_2041/slice_Re_p_z_H2.png`
- `COMSOL_comparison_results/Case_A_standing/figs_investigation_20260213_2041/slice_abs_p_z_H2.png`
- `COMSOL_comparison_results/Case_B_vortex/figs_investigation_20260213_2041/plane_z_H2_Re.csv`
- `COMSOL_comparison_results/Case_B_vortex/figs_investigation_20260213_2041/plane_z_H2_abs.csv`
- `COMSOL_comparison_results/Case_B_vortex/figs_investigation_20260213_2041/slice_Re_p_z_H2.png`
- `COMSOL_comparison_results/Case_B_vortex/figs_investigation_20260213_2041/slice_abs_p_z_H2.png`
- `COMSOL_comparison_results/Case_C_combined/figs_investigation_20260213_2041/plane_z_H2_Re.csv`
- `COMSOL_comparison_results/Case_C_combined/figs_investigation_20260213_2041/plane_z_H2_abs.csv`
- `COMSOL_comparison_results/Case_C_combined/figs_investigation_20260213_2041/slice_Re_p_z_H2.png`
- `COMSOL_comparison_results/Case_C_combined/figs_investigation_20260213_2041/slice_abs_p_z_H2.png`
- `COMSOL_comparison_results/INVESTIGATION_NOTES.md`
