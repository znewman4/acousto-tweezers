# Results Index

> Auto-maintained index of simulation results in `results/`.
> Last updated: 2026-02-23

---

## Active / Latest Runs

### `vortex_only_hires_20260223_145831/`
**Vortex-only high-resolution snapshots** — 23 Feb 2026  
- Standing wave OFF, plastic lens vortex l=1, offset 0.2 mm  
- 5 elem/λ, 439,587 DOFs, MUMPS direct solver  
- Solver converged (REASON=4), max|p| = 1.77 Pa  
- **Outputs:** 20 PNG figures (XY at 5 z-heights + XZ midplane, linear/log/phase), 2 CSV profiles  
- **Audit:** `VORTEX_ONLY_AUDIT.md` — full BC/geometry/solver documentation  
- **Script:** `scripts/experiments/vortex_only_hires.py`

### `run_20260222_132559_maxcap/`
**Production run (maxcap)** — 22 Feb 2026  
- Corrected model sweep + production farfield  
- 5 elem/λ, 14 threads  
- Contains `corrected_sweep/` and `production/` subdirectories

---

## Historical Runs

### `corrected_model_20260220_17*/`
Corrected model sweeps (20 Feb 2026). Two runs with corrected PML + BC model.

### `farfield_production_20260220_*/`
Six production farfield runs (20 Feb 2026). Iterative debugging of production pipeline.

### `diagnostic_20260219_*/`
Diagnostic pipeline runs (19 Feb 2026).

### `Deliverable1_FarFieldValidation/`
Far-field validation deliverable.

### `Deliverable2_LensPropagation/`
Lens propagation deliverable.

### `Deliverable3_Interaction/`
Vortex–standing interaction deliverable.

### `LinuxConfirmation/`
Initial Linux environment confirmation run.

### `Vortex3D/`
3D vortex visualisation data.

---

## Symlinks

- `farfield_production_latest` → symlink to latest farfield production run
