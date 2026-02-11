# Archive Notes

This file documents what was moved into `archive/` during the 2026-02-10
repository cleanup and why.

## archive/redundant_docs/

Root-level markdown files that were session logs, completion reports, planning
documents, or superseded summaries.  None of these are referenced by the
current solver code.  All were generated between January and February 2026
during iterative development.

Moved files (23):
- `DELIVERABLES_COMPLETE_20260206.md` — Phase 2 deliverables status
- `DELIVERABLES_STEPS_1_2.md` — Steps 1–2 checklist
- `DELIVERABLES_SUMMARY.md` — deliverables location summary
- `HONEST_AUDIT_20260126.md` — January audit of repo state
- `INDEX.md` — streaming solver deliverables index
- `INDEX_STEPS_1_2.md` — particle dynamics index
- `LEVEL2_STOKES_COMPLETION_REPORT_20260209.md` — streaming completion report
- `PARAVIEW_HANDOFF.md` — ParaView workflow transition
- `PARTICLE_STREAMING_IMPLEMENTATION.md` — coupling implementation guide
- `PHYSICS_REALITY_CHECK.md` — 3D vs 2D interrogation
- `PHYSICS_STATUS.md` — validated physics summary
- `README_NEW.md` — draft README
- `RESULTS_POINTER.md` — Phase 2 result pointers
- `RESULTS_SUMMARY.md` — Jan 25 session results
- `SESSION_SUMMARY_20260209.md` — Feb 9 session log
- `STEPS_1_2_COMPLETE.md` — Steps 1–2 completion
- `STREAMING_COMPLETE_SUMMARY.md` — streaming summary
- `STREAMING_DELIVERABLES.md` — streaming deliverables checklist
- `STREAMING_IMPLEMENTATION_REPORT.md` — streaming validation report
- `TASK_COMPLETION_SUMMARY_20260208.md` — Feb 8 task completion
- `VALIDATION_REPORT_20260126.md` — Jan 26 validation report
- `RESULTS_LOCATIONS.txt` — result file listings
- `STREAMING_FILES_SUMMARY.txt` — streaming file manifest
- `CHANGELOG_old.md` — previous multi-version CHANGELOG
- `README_old.md` — previous README

### archive/redundant_docs/docs/

Documentation subdirectories and files from `docs/` that were superseded or
no longer part of the main narrative:

- `docs/archive/` — old 3D Helmholtz completion reports
- `docs/physics/` — PML docs (incl. audit exposing errors), multiphysics README
- `docs/refactor/` — package unification report
- `docs/square_dish/` — Phase 1/2 implementation reports
- `docs/DEVLOG_20260208_streaming_particles.md` — dev session log
- `docs/PHASE_SWEEP_STATUS.md` — superseded exploratory study
- `docs/PYVISTA_VIZ.md` — superseded by ParaView workflow
- `docs/VORTEX_*.md` — superseded vortex implementation docs
- `docs/README.md` — old docs index

## archive/scripts_old/

Debug scripts, duplicates, and obsolete one-off scripts:

- `debug_direct_solve.py` — one-off MUMPS ground truth test
- `debug_forcing_dist.py` — forcing distribution diagnostic
- `debug_rhs_norm.py`, `debug_rhs_norm2.py` — RHS norm checks
- `debug_streaming.py`, `debug_streaming2.py`, `debug_streaming3.py` — streaming debugs
- `debug_streaming_minimal.py` — unit-cube Stokes test
- `compare_vortex_standing_fixed.py` — superseded by compare_vortex_standing.py
- `test_complex_backend_runtime.py` — duplicate of test_complex_backend.py
- `quick_streaming_demo.py` — obsolete quick test
- `square_dish_phase_control.py` — monolithic Phase 1 script
- `test_phase2_minimal.py` — minimal smoke test (replaced by petri_dish_bcs)
- `run_phase1_5.py` (square_dish) — duplicate of root version
- `run_phase2_storyboard.py` (square_dish) — duplicate of root version

### archive/scripts_old/legacy_scripts/

Entire `scripts/legacy_scripts/` directory containing adjoint, MPC, greedy
controller, FD-based 2D solver, and early FEM demo scripts.  None of these
are used by the current solver stack.

### archive/scripts_old/setup/

Environment setup scripts (`scripts/setup/`).

## archive/results/

Old result directories from prior sessions:

- `ARCHIVE_OLD/` — pre-3D results
- `ARCHIVE_PRE_3D_PARAVIEW_WORKFLOW/` — pre-ParaView results (2.5 GB)
- `ARCHIVE_PRE_DEPOSITION_MODEL/` — pre-deposition model (1.4 GB)
- `ARCHIVE_PRE_STOKES_STREAMING_FIX/` — pre-streaming fix
- `deposition_20260209_*` — Feb 9 deposition runs (superseded)
- `deposition_20260210_*` — Feb 10 intermediate runs (superseded)
- `complex_diagnostics_*` — diagnostics runs
- `diagnostics/`, `logs/` — misc output
- `INDEX.md`, `*.json`, `*.md` — result metadata

Only `results/deposition_20260210_203721/` (the latest successful run) was
retained in the main `results/` directory.

## Deleted Files

- `scripts/analysis/visualization/create_streaming_summary.py` — contained
  hardcoded snapshot data baked into source; not reusable.
