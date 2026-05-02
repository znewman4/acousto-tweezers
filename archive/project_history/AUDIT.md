# AUDIT.md — acousto-tweezers Linux-Ready Audit

*Branch:* `chore/linux-ready-audit`
*Date:* 2026-02-20
*Author:* Claude (automated refactoring session)

---

## A. Config Drift Identified and Fixed

### Problem

Two run scripts existed with **different physics configurations**:

| Parameter | `corrected_model_sweep.py` (correct) | `production_farfield_run.py` (old) |
|-----------|--------------------------------------|-------------------------------------|
| `H_top` (petri slab) | **2 mm** | 1 mm |
| Top BC | Physical water–air Robin (`Z_air = 411.6 Pa·s/m`) | `top_impedance_Zrel = 0.001` (artificial) |
| `top_bc_type` | (not set — uses new default) | `"impedance"` (deprecated) |

The sweep script used the **corrected** two-region model (H_petri = 2 mm,
physical Robin BC), while the production script still used the **old** model
(H_top = 1 mm, artificial impedance).

### Fix

Created a single source of truth in
`src/acoustweezers/experiments/farfield_petri_cuboid/presets.py`:

```python
CORRECTED_PRESET: dict = dict(
    Lx=6e-3, Ly=6e-3,
    H_under=3e-3,          # water-bath depth
    H_top=2e-3,            # petri slab — FIXED at 2 mm
    frequency_hz=2.0e6,
    disk_radius=1.0e-3,
    disk_velocity_amplitude=1e-6,       # 1 µm/s
    standing_velocity_amplitude=10e-6,  # 10 µm/s
    ...
)
```

Both scripts now `from ...presets import CORRECTED_PRESET` and derive their
`BASE_CFG` / module constants from this single dict.  The old
`top_bc_type="impedance"` and `top_impedance_Zrel=0.001` fields are gone; the
`FarFieldConfig` dataclass marks them as **DEPRECATED** and the solver always
uses the physical water–air Robin BC.

---

## B. What Each Script Does

### `scripts/experiments/corrected_model_sweep.py`

**Purpose:** Geometry sweep over `H_bath × f_lens` grid to find the best
bath depth and focal length, then runs a vortex + standing-wave interaction
check at the best geometry.

**Steps:**
1. *Phase 1 — Vortex-only sweep:* 5 × 5 grid (H_bath ∈ {3,4,5,6,7} mm,
   f_lens ∈ {2,3,4,5,6} mm), skipping infeasible combos
   (f_lens ≥ H_bath).  Solves Helmholtz with PML, records `max|p|` in bath
   and petri regions.
2. *Best-geometry selection:* picks the combo that maximises pressure in
   the bath while keeping the focus comfortably below the petri slab.
3. *Phase 2 — Interaction check:* at the best geometry, runs standing-only,
   vortex-only, combined, and difference-field cases.  Computes ROI metrics
   in the petri mid-plane.
4. *Output:* INDEX.md, config.json, CSV summaries, ~20 PNG figures.

**Typical runtime:** ~20–40 min at 4 elem/λ (depends on RAM).

### `scripts/experiments/production_farfield_run.py`

**Purpose:** Full 8-step production verification pipeline at higher
resolution.

**Steps:**
1. Environment validation (external)
2. Solver config lock (MUMPS, thread controls)
3. Production mesh setup (default 5 elem/λ)
4. Canonical cases: standing_only, vortex_only, combined, rigid_combined
5. Free-space vortex propagation verification
6. PML stability check (1λ vs 2λ thickness)
7. Interaction metrics (Δ|p|, ΔU, selectivity, Hessian)
8. Particle scaling (10–100 µm)

**Typical runtime:** ~1–2 hours at 5 elem/λ on 32 GB RAM.

---

## C. CLI Arguments

Both scripts now accept the same flags:

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--out DIR` | str | auto-timestamped | Output directory path |
| `--elem-per-lambda N` | int | 4 (sweep) / 5 (production) | Elements per wavelength |
| `--threads N` | int | `ncores // 2` | OMP thread count |
| `--tag STRING` | str | `""` | Appended to auto-generated dir name |
| `--overwrite` | flag | off | Allow writing into existing dir |

**Examples:**
```bash
# Sweep with defaults
python scripts/experiments/corrected_model_sweep.py

# Production at 6 elem/λ, 4 threads, tagged "nightly"
python scripts/experiments/production_farfield_run.py \
    --elem-per-lambda 6 --threads 4 --tag nightly

# One-shot wrapper that runs both
bash scripts/run_linux_all.sh --tag batch1
```

---

## D. Known Gaps / Next Steps

1. **PML sensitivity:**  The Feb 20 Linux run showed a 27.9 % pressure
   difference between 1λ and 2λ PML, well above the 10 % pass threshold.
   Investigation needed: either increase default PML to 2λ, tune
   `sigma_max_factor`, or accept and document.

2. **Axicon lens path untested:**  `lens_drive="axicon"` is wired up in
   `solve_pressure.py` and `vortex_lens.py`, but no integration test
   exercises it.

3. **No automated regression tests:**  The scripts produce figures and CSVs
   but there is no `pytest` suite that checks solver convergence or output
   values.  Adding a lightweight smoke test (e.g., coarse 2 elem/λ solve,
   assert `max|p| > 0`) would catch import breakage early.

4. **Memory at high resolution:**  5 elem/λ on a 6 × 6 × 9 mm domain
   produces ~500 k DOFs.  6 elem/λ pushes past 1 M DOFs and may exceed
   32 GB with MUMPS.  The `--elem-per-lambda` flag lets users dial this
   down, but a heuristic memory guard (estimate DOFs → warn if > threshold)
   would be a nice guardrail.

5. **Stokes streaming / particle transport:**  The current pipeline stops
   at the acoustic pressure field.  Gorkov potential and radiation force
   are computed post-hoc.  Second-order streaming and full particle
   trajectory integration are not yet implemented in the FEniCSx stack.

6. **COMSOL comparison validation:**  Three prior comparison attempts
   (in `COMSOL_comparison_archive/`) were marked incorrect.  A
   `COMSOL_comparison_results/` directory exists with Case A/B/C but
   has not been cross-validated against the corrected model.
