# Changelog

## [Fix] Standing-Wave Boundary Condition Height Restriction

### Problem
Standing-wave drive boundary condition was applied to entire vertical sidewalls
(bath + petri), causing incorrect cavity excitation and non-physical standing patterns.

### Root Cause
Facet selection logic did not restrict BC to petri slab z-range.

### Fix
Added z-range filtering to standing-wall facet selection:
`z >= H_under and z <= H_under + H_top`

### Impact
Standing-wave excitation now correctly confined to petri slab only.
Bath region no longer directly driven.
Produces physically consistent modal structure.

### Files Modified
- src/acoustweezers/experiments/farfield_petri_cuboid/solve_pressure.py
- src/acoustweezers/experiments/farfield_petri_cuboid/mesh.py

### Regression Check
- standing_only: PASS
- vortex_only: PASS
- combined: PASS

---

## 2026-02-17 — Far-Field Hardening + Day-2 Deliverables

### Part A: PML Coefficient Refactoring

- **A1**: Refactored PML metric coefficients (Λ\_x, Λ\_y, Λ\_z, J) from DOF-array
  arithmetic to **UFL expressions**.  FFCx now evaluates `Λ_x = s_y·s_z/s_x` at
  quadrature points directly from interpolated σ Functions, avoiding P2 projection
  error on rational expressions.  σ\_x/y/z remain as `fem.Function(V)`.
- **A2**: PML operator check script (`farfield_pml_operator_check.py`):
  PML vs rigid walls at 4 elem/λ — **PASS** (98.4% diff in max|p|, 7 vs 5000 GMRES iters).

### Part B: Top BC Sensitivity

- **B1**: Swept H\_under ∈ {1, 2, 3} mm × {impedance, dirichlet} with PML on.
  Top BC has **< 0.4%** effect on all metrics — confirms PML domination.
  Script: `farfield_s4_topbc_sensitivity.py`.

### Part C: Solver Robustness & Reporting

- **C1**: `solve_helmholtz()` now extracts KSP diagnostics after each solve:
  converged reason, iteration count, residual norm, timing breakdown
  (mesh / assemble+solve / total).  Added `mesh_time`, `ksp_iterations`,
  `ksp_converged_reason`, `ksp_residual_norm` fields to `PressureSolution`.
- **C2**: `fast_mode_config()` factory returns FarFieldConfig with 4 elem/λ
  (+ warning).  `demo_config()` provides standard 6×6×4 mm configuration.
- **C3**: CLI args `--rtol`, `--restart`, `--maxit`, `--fast` added to
  `farfield_vortex_plus_standing.py`.  PETSc options stored in `config.json`.

### Part D: Plastic Lens Hardening

- **D1**: `compute_plastic_lens_thickness()` — auto-safe base thickness `t0`:
  when `t0=None` (new default), computes `2π/|dk| + safety_margin`.  Warning
  if user-provided `t0` is too small.  All thicknesses now guaranteed positive.
- **D2**: Three lens presets:
  - **Preset A**: pure vortex (l=1, f=50 mm, no focusing offset)
  - **Preset B**: focused  (l=1, f=10 mm)
  - **Preset C**: off-axis (l=1, f=10 mm, xf=0.2 mm)
  `LENS_PRESETS` dict + `export_lens_maps()` for NPY/CSV batch export.
- **D3**: Gallery script (`farfield_plastic_lens_gallery.py`): 6-panel plot +
  individual PNGs + NPY arrays + summary CSV.  All presets: thickness
  [0.200, 1.848] mm — verified positive.

### Part E: Plastic vs Ideal Comparison + Slice Exports

- **E1**: `farfield_plastic_vs_ideal.py` — side-by-side solve with centerline,
  XY, XZ comparison plots + .npz slice exports.  At 4 elem/λ: plastic ≈ ideal
  (0.0% diff in max|p|, 14.7% diff in cl\_max from numerical noise near axis null).
- **E2**: `export_slice_xy()` and `export_slice_xz()` added to `post.py`.
  Output: `.npz` with keys `x`, `y`/`z`, `p_mag`, `p_phase`, `p_complex`.

### Part F: Documentation

- README §9.2 updated: UFL PML coefficient formulation documented.
- README §9.5 updated: new CLI arguments listed.
- README: new §9.8 (Lens Presets) and §9.9 (Diagnostic Scripts).
- CHANGELOG updated with full session log.

### Files Modified

| File | Changes |
|------|---------|
| `solve_pressure.py` | UFL PML coefficients, KSP diagnostics, `PressureSolution` fields |
| `config.py` | `verbose_solver`, `fast_mode_config()`, `demo_config()` |
| `vortex_lens.py` | Safe thickness, 3 presets, `LENS_PRESETS`, `export_lens_maps()` |
| `post.py` | `export_slice_xy()`, `export_slice_xz()` |
| `farfield_vortex_plus_standing.py` | CLI args `--rtol/--restart/--maxit/--fast`, `petsc_opts` passthrough |

### Scripts Created

| Script | Purpose | Key Result |
|--------|---------|------------|
| `farfield_pml_operator_check.py` | A2: PML vs rigid | 98.4% diff, PASS |
| `farfield_s4_topbc_sensitivity.py` | B1: impedance vs Dirichlet | < 0.4% diff with PML |
| `farfield_plastic_lens_gallery.py` | D3: lens preset visual gallery | thickness ∈ [0.200, 1.848] mm |
| `farfield_plastic_vs_ideal.py` | E1: plastic vs ideal comparison | ~0% diff at 4 elem/λ |

### Remaining Risks & TODOs

1. **UFL Division performance** — FFCx may raise quadrature degree for the
   rational PML expressions, potentially slowing assembly.  Not observed at
   4 elem/λ.  Monitor at 5+ elem/λ production runs.
2. **Mesh resolution** — At 4 elem/λ plastic and ideal fields are nearly
   identical (0.0% diff).  Finer mesh (5–6 elem/λ) is needed to resolve the
   lens-driven phase/amplitude differences and confirm the plastic lens adds
   physical value beyond the ideal vortex.
3. **Memory** — 5 elem/λ = ~348K DOFs ≈ 1.8 GB.  Two simultaneous solves on
   7.5 GB machine will OOM.  Scripts use `del sol; gc.collect()` between solves.

---

## 2026-02-16 — Plastic Lens + PML Diagnostics (Far-Field 2 MHz)

### Part 1: PML Diagnostic Audit (S1–S4)

Systematic diagnostics confirmed the far-field PML implementation is correct:

- **S1 PASS**: σ_z = 0 at all DOFs near the top face and in the petri slab.
  Only σ_x, σ_y are nonzero (in lateral PML bands).
- **S2 PASS**: σ_z = 0 for all 5530 DOFs in the disk column (r ≤ R, z < t_pml).
  Outside the disk: 12016 DOFs with σ_z up to 6.28e+07.
- **S3 FIXED**: GMRES(30)+ILU completely stagnated at residual ~124 after
  4800 iterations — residual was constant to 13 digits.  Fixed by changing
  `ksp_gmres_restart` from 30 → 200.  Tolerance sweep: rtol=1e-3 and
  rtol=1e-5 give identical max|p| = 18.0069 Pa (0.00% difference).
  Default rtol changed to 1e-4.
- **S4 PASS** (by design): Impedance vs pressure-release top BC gives only
  0.02% difference in max|p|.  PML absorption dominates — top BC is cosmetic
  at current domain depth.

Diagnostic script: `scripts/experiments/farfield_part1_diagnostics.py`

### Part 2: Plastic Lens Vortex Drive

Replaced the placeholder ideal vortex drive with a fabricable plastic lens model:

**Physics:**
- φ_target = ℓ·θ + k_water·(√((x−xf)²+(y−yf)²+f²) − f)
- φ_plastic = mod(φ_target, 2π) — fabricable via thickness variation
- v_n = V₀·A(r)·exp(i·φ_plastic) with cosine_taper/tukey/uniform apodization

**Files modified:**
- `src/acoustweezers/physics/acoustics/vortex_lens.py` — Added `PlasticLensConfig`,
  `compute_plastic_lens_phase()`, `compute_plastic_lens_amplitude()`,
  `compute_plastic_lens_thickness()`, `create_plastic_lens_drive()`.
  Original `VortexLensConfig` and functions preserved intact.
- `src/acoustweezers/experiments/farfield_petri_cuboid/config.py` — Added 8 lens
  fields: `lens_drive`, `lens_l`, `lens_focal_length`, `lens_focus_offset_x/y`,
  `lens_c_lens`, `lens_apodization`, `lens_apodization_strength`.
- `src/acoustweezers/experiments/farfield_petri_cuboid/solve_pressure.py` —
  Replaced `_create_disk_vortex_source` with `_create_disk_source` dispatcher
  (plastic vs ideal).  Solver defaults: GMRES(200), rtol=1e-4.
- `src/acoustweezers/experiments/farfield_petri_cuboid/post.py` — 4-panel disk
  drive diagnostic: disk_amplitude.png, disk_phase.png, disk_real.png, disk_imag.png.
- `scripts/experiments/farfield_vortex_plus_standing.py` — Rewritten with
  `--ideal` CLI flag; defaults to plastic lens.

**No changes to shallow_square_dish code.**

---

## 2026-02-10 — Boundary-Condition Overhaul (Petri-Dish Model)

### Helmholtz Weak Form — Boundary Condition Model

The Helmholtz pressure solver (`src/acoustweezers/experiments/shallow_square_dish/solve_pressure.py`)
has been updated to implement physically correct boundary conditions for a
shallow petri-dish acoustic tweezer.  Prior to this change **all** walls
(bottom, sides) carried an impedance-matched Robin term `α = −iωρ/Z`, which
turned the dish into an anechoic chamber and suppressed standing-wave formation.

**Changes:**

- **Side walls (x = 0, x = L, y = 0, y = L):**
  Side walls are now **rigid reflectors** by default (natural Neumann,
  `∂p/∂n = 0`).  When a wall pair is an active transducer (standing-wave mode
  or combined mode) it receives a **pure Neumann source** term in the RHS
  (`∂p/∂n = −iωρ V_n`).  Side walls **never** carry an impedance Robin term.

- **Bottom wall (z = 0) — segmented:**
  The bottom is now split into two tagged regions during mesh generation:
  - `TAG_BOTTOM_DISC` (tag 1): circular transducer patch of radius
    `cfg.bottom_disc_radius_effective` (defaults to `vortex_aperture_radius`).
    Carries an impedance Robin term `α = −iωρ/Z` (Z = ρc) at all times,
    and a vortex Neumann source when active (vortex or combined mode).
  - `TAG_BOTTOM_RIGID` (tag 7): the rigid floor surrounding the disc
    (natural Neumann, `∂p/∂n = 0`).
  - `TAG_BOTTOM` is retained as an alias for `TAG_BOTTOM_DISC` for backward
    compatibility in the streaming solver's no-slip boundary conditions.

- **Top wall (z = H):** unchanged — low-impedance Robin BC modelling the
  air–water interface (`Z_top = 0.001 × Z_water`).

- **Mode logic (explicit):**
  - `standing`: x-walls (and y-walls if `axis="both"`) active; disc inactive.
  - `vortex`: disc active; all side walls rigid.
  - `combined`: both active.

### Mesh Generation

- `create_mesh()` now segments bottom-boundary facets by computing the radial
  distance of each facet's midpoint from the vortex centre.  Facets inside
  the disc radius receive `TAG_BOTTOM_DISC`; the remainder receive
  `TAG_BOTTOM_RIGID`.

### Vortex Source

- `_create_vortex_source()` now selects DOFs from `TAG_BOTTOM_DISC` only
  (previously used the entire bottom boundary).
- The surface integral in the RHS is taken over `dss(TAG_BOTTOM_DISC)`.

### Configuration

- **Removed:** `passive_bc_type`, `bottom_bc_type` — no longer applicable.
- **Added:** `bottom_disc_radius: float = None` (defaults to
  `vortex_aperture_radius`); exposed via `bottom_disc_radius_effective`
  property.

### Downstream Updates

- `streaming.py`: import updated to include `TAG_BOTTOM_DISC`,
  `TAG_BOTTOM_RIGID`.  Streaming BCs (no-slip on all solid walls) unchanged.
- `test_rigid_vs_absorbing.py`: rewritten to test all three modes (standing,
  vortex, combined) under the new BC model.

### Repository Cleanup

- Root-level session reports, completion summaries, and planning docs (23 files)
  moved to `archive/redundant_docs/`.
- `docs/` subdirectories (archive, physics, refactor, square_dish) and
  superseded docs moved to `archive/redundant_docs/docs/`.
- `scripts/` reorganised into `validation/`, `experiments/`, `analysis/`.
  Debug and obsolete scripts moved to `archive/scripts_old/`.
- Old result directories moved to `archive/results/`.
- `README.md` rewritten with full system documentation.

---

### Validation

| Test | Script | Verifies |
|------|--------|----------|
| 1D impedance reflection | `scripts/validation/test_1d_impedance.py` | Robin coefficient `α = −iωρ/Z` gives \|R\| ≈ 0 for Z = ρc |
| Energy / power balance | `scripts/validation/test_energy_balance.py` | P_in = P_abs to machine precision |
| Petri-dish BC smoke test | `scripts/validation/test_petri_dish_bcs.py` | Bottom segmentation counts; standing-mode pressure ≫ 0.5 Pa; all modes nonzero |

All three tests pass after the changes above.

Standing-mode peak pressure increased from ~0.15 Pa (absorbing walls) to
~58 Pa (rigid reflecting walls) at 10 μm/s source amplitude — consistent with
physical expectations for a resonant cavity.

---

### Known Limitations

- Streaming velocity magnitudes remain sensitive to mesh resolution, Robin BC
  impedance values, and solver tolerances.  The MUMPS direct solver is used
  for robustness but does not scale to very fine meshes.
- Bottom disc segmentation uses facet-midpoint distance, which introduces a
  mesh-dependent jagged boundary at the disc edge.  This is acceptable at
  current resolutions (6 elements per wavelength) but would benefit from a
  conforming mesh for high-accuracy studies.
- PML (perfectly matched layer) infrastructure exists in the codebase but is
  not used in the current petri-dish model and has known accuracy issues
  documented in `archive/redundant_docs/docs/physics/PML_TECHNICAL_AUDIT.md`.
