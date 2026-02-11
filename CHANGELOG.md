# Changelog

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
