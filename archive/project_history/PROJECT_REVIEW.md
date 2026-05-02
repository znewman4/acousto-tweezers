Now I have all the information needed. Let me compile the comprehensive report. 

Updated todo list

---

# ACOUSTO-TWEEZERS — Project Inventory & Structured Report

---

## PART 1 — REPO ARCHITECTURE

### Overview

The repository is a FEniCSx-based acoustic tweezers simulation engine for selective micro-particle manipulation in shallow liquid-filled cavities. It combines standing-wave lattices with movable vortex-beam lenses.

**Codebase size:** 154 source files (~53,500 lines), 135 script files (~76,200 lines), 79 git commits spanning 2025-12-19 to 2026-03-11.

### Directory Map

| Directory | Role | Contents | Workflow Role |
|-----------|------|----------|---------------|
| acoustweezers | **Core library** | 23 modules across 5 subsystems: core, physics, numerics, io, viz | Foundation — all scripts import from here |
| core | Infrastructure | Config, case loader, diagnostics, report, audit, export, logging | Reproducibility & reporting pipeline |
| acoustics | Acoustic solvers | Helmholtz, PML, ASM, streaming, thermoviscous, vortex lens, impedance | Physics engine — frequency-domain PDE solving |
| particles | Particle dynamics | Gor'kov potential, radiation force, overdamped trajectory solver | Particle manipulation — force fields & transport |
| numerics | FEM infrastructure | FEniCSx assembly, PETSc solvers, Gmsh mesh generation, domain enums | Numerical backbone — 762K DOF capability |
| experiments | Experiment runners | farfield_petri_cuboid, shallow_square_dish, square_dish, path_tracking | High-level experiment orchestration |
| dev | **Active development** | 18 scripts: C-shape studies, bridge formation, RS investigation, lens design | Current research frontier |
| experiments | **Experiment library** | 52 scripts: sweeps, diagnostics, validation, demos, galleries | Systematic parameter exploration |
| analysis | Analysis & rendering | diagnostics_utils, COMSOL comparison, batch outputs, ParaView rendering | Post-processing & cross-validation |
| validation | **Test suite** | 28 scripts: 1D impedance, PML reflection, energy balance, Helmholtz convergence | Physics verification gate |
| viz | Visualisation | Far-field figure generation | Publication figure export |
| results | **Output repository** | 35+ study folders, 2,491 PNGs, 58 CSVs, 16 GIFs, 266 NPZs, 49 JSONs | Evidence base for final document |
| fem_standing_wave_cache | FEM cache | Standing wave solutions at epl4/5/6 (762K DOFs, 6 elem/λ) | Reusable truth data — avoids re-solving |
| _archive | Superseded results | 23 archived directories from early phases | Historical record |
| _mobility_fem_cache | Grid cache | Pre-gridded standing + vortex fields for mobility studies | Fast particle dynamics input |
| COMSOL_comparison_results | **Validation** | Cases A/B/C: standing, vortex, combined — figures + CSV + metadata | Independent cross-validation evidence |
| cases | Case definitions | canonical_farfield.json | Parametric study input |
| docs | Documentation | COMSOL spec, Linux setup, validation guide, standing wave audit | Knowledge base |
| docker | Container | Dockerfile for FEniCSx environment | Reproducible deployment |

### Key Architectural Properties

1. **Physics hierarchy enforcement**: 7-level ladder (Acoustics → PML → Coupling → Thermoviscous → Streaming → Particles) enforced at runtime
2. **UFL + DOLFINx native**: No homebrew FEM — all PDEs in UFL, solved by DOLFINx + PETSc (complex scalars, MUMPS direct solver)
3. **Single source of truth**: Domain labels from Gmsh physical groups, not inference
4. **Full audit trail**: Every run generates MANIFEST.json, config.json, INDEX.md, metrics.csv
5. **FEM + ASM hybrid**: Standing waves via FEM (expensive, cached), vortex propagation via Angular Spectrum Method (fast, no FEM needed)

---

## PART 2 — SCRIPT INVENTORY

### scripts/dev/ (18 scripts — Active Development)

| Script | Purpose | Key Inputs | Outputs | Category | Status |
|--------|---------|------------|---------|----------|--------|
| bridge_master_study.py | 19,360-config sweep of bridge barrier energy (α, φ₀, x₀, y₀) | FEM cache, α×16 φ₀×11²offsets | PNG, CSV, JSON, INDEX.md | parameter sweep | core |
| bridge_phase_offset_study.py | Phase/offset bridge formation (800 configs) | FEM cache, α×φ₀×offsets | PNG, JSON, REPORT.md | parameter sweep | core |
| c_shape_backprop_phase_to_thickness.py | ASM backpropagation: target field → lens plane → printable thickness | C-shape best candidate, ASM config | NPZ, PNG, CSV, INDEX.md | field generation | core |
| c_shape_lens_15mm_manufacturing_study.py | Manufacturing-ready lens: masked smoothing, slope limiting, 3D CAD | Source candidate, smoothing sweep | CSV, PNG (44 figs), NPZ, INDEX.md | field generation / viz | core |
| c_shape_lens_15mm_overlay_study.py | Lens × standing-wave interaction: α and ψ sweeps | Cached SW + printable lens | CSV, PNG (21 figs), NPZ, INDEX.md | propagation / sweep | useful |
| c_shape_on_cached_sw_geometry_study.py | 144 C-shape geometry candidates screened on real trap geometry | FEM cache, geometry params | CSV, PNG (432+ figs), NPZ, INDEX.md | field generation / metrics | core |
| c_shape_particle_merge_gif_demo.py | Crossfade particle merge: 9-particle overdamped dynamics with GIF | Cached fields, particle props | GIF, PNG (8 figs), CSV, JSON | particle dynamics / demo | useful |
| c_shape_transport_refinement_study.py | Windowed lens + partial SW retention parameter sweep (36 configs) | Lens field, window params | GIF (2), PNG (11 figs), CSV, INDEX.md | particle dynamics / sweep | experimental |
| cleanup_rs_investigation_artifacts.py | Archive 23 obsolete result directories to _archive/ | Results directory | ARCHIVE_LOG.md | utility | useful |
| fem_standing_plus_asm_vortex_local_3x3.py | FEM + ASM overlay on local 3×3 trap grid with α sweep | FEM solver/cache, ASM config | PNG, JSON, REPORT.md | field gen / validation | core |
| full_domain_gorkov_diagnostic.py | Full-domain trap detection via Gor'kov and Hessian analysis | FEM cache (800×800 grid) | PNG (3), JSON, NPZ | metrics / validation | useful |
| rs_aperture_scaling_vtu_debug.py | Aperture requirement analysis + VTU geometry debug | ASM config, R sweep [1-5]mm | PNG, VTU, JSON | validation / export | experimental |
| rs_aperture_sweep_vtu_fix.py | Aperture scaling validation with corrected VTU export | Lens config, R sweep | PNG, VTU | validation / export | experimental |
| rs_free_space_vortex_truth.py | Definitive free-space vortex validation pack (ℓ=2) | Plastic lens, 120 z-planes | PNG, VTU, JSON | field gen / validation | core |
| rs_hourglass_investigation.py | Stage 1 hourglass integrity: VTU ordering, phase topology, ring tracking | CORRECTED_PRESET params | PNG (4 panels), REPORT.md | validation / metrics | experimental |
| rs_hourglass_proof.py | Visual convergence→waist→divergence proof for vortex | Plastic lens (R=5, f=4, ℓ=2) | PNG (XY+XZ), VTU, JSON | visualisation / export | useful |
| rs_plastic_lens_hourglass_demo.py | Canonical ASM hourglass demo without FEM | Plastic lens preset | PNG (XY+XZ), VTU, JSON, REPORT.md | field gen / viz / export | core |
| waist_align_zstar_then_xy_alpha_sweeps.py | Waist alignment sweep + XY α sweep on 3×3 ROI | Best lens config, z₀ sweep | PNG, JSON, REPORT.md | sweep / validation | experimental |

### scripts/experiments/ (52 scripts — Experiment Library)

| Script | Purpose | Category | Status |
|--------|---------|----------|--------|
| _solve_worker.py | FEM subprocess worker (single case) | field generation | core |
| _solve_worker_multi_z.py | Multi-z FEM worker with XZ slice | field generation | core |
| audit_farfield.py | Comprehensive PML diagnostics (Q A1–E2) | validation | core |
| compare_vortex_standing.py | Scale-calibrated vortex + standing superposition | experiment demo | experimental |
| corrected_model_sweep.py | Corrected H_bath × f sweep with interaction | parameter sweep | core |
| demo_vortex.py | Acoustic vortex demonstration | experiment demo | experimental |
| diagnostic_pipeline.py | 6-step convergence/PML/amplitude pipeline | validation | core |
| farfield_part1_diagnostics.py | PML/source/solver issue diagnostics (S1-S4) | validation | core |
| farfield_plastic_lens_gallery.py | Plastic lens gallery (3 presets) | visualisation | useful |
| farfield_plastic_vs_ideal.py | Plastic vs ideal lens metrics comparison | validation | useful |
| farfield_pml_operator_check.py | PML operator verification (>10% difference) | validation | core |
| farfield_s3_s4_only.py | S3/S4 diagnostics only | validation | useful |
| farfield_s4_topbc_sensitivity.py | Top BC sensitivity sweep (H_under, PML, BC type) | parameter sweep | useful |
| farfield_vortex_plus_standing.py | Far-field combined solve (PML on/off) | experiment demo | core |
| fixed_vortex_gallery.py | Fixed vortex gallery (no PML artefacts) | visualisation | useful |
| focused_vortex_gallery.py | Focused vortex gallery (f sweep 2–6 mm) | visualisation | useful |
| golden_run.py | Golden run: 3 canonical PML cases with audit | experiment demo | core |
| impedance_sweep.py | Wall + bottom impedance sweep (lossy cavity) | parameter sweep | experimental |
| linux_confirmation_diagnostics.py | Lens propagation checks (A–C) | validation | core |
| optimal_geometry_gallery.py | Optimal geometry z-progression | visualisation | useful |
| phase0_baseline_sweep.py | Baseline disc diameter sweep (D 2/3/4 mm) | parameter sweep | experimental |
| phase1_square_dish.py | Square dish phase control entrypoint | field generation | experimental |
| phase1_sweep.py | Transducer/dish architecture sweep | parameter sweep | experimental |
| phase2_freq_sweep.py | Frequency sweep 500 kHz–2 MHz | parameter sweep | experimental |
| phase2_time_evolution.py | Time evolution entrypoint | field generation | experimental |
| phase2b_resonance_investigation.py | Fine resonance characterisation | parameter sweep | experimental |
| physics_affirmation.py | Gor'kov old-vs-new, lens focus, winding, superposition | validation | core |
| physics_baseline_hires.py | XY/XZ/YZ hi-res slices (300 DPI) | visualisation | useful |
| pluck_demo.py | Local particle extraction from trap | experiment demo | experimental |
| pre_lens_affirmation.py | Pre-lens: trap plane, 3D focus, charge, alpha calibration | validation | core |
| production_farfield_run.py | Production run: MUMPS, full verification, particle scaling | experiment demo | core |
| run_axicon_lens_demo.py | Axicon (Bessel-like) vortex demo | experiment demo | experimental |
| run_complex_streaming_diagnostics.py | Streaming PETSc complex type check | validation | useful |
| run_deposition_experiment.py | In-silico particle deposition protocol | particle dynamics | experimental |
| run_device_demo.py | Complete device workflow (all physics levels) | experiment demo | experimental |
| run_particle_streaming_demo.py | Particle + streaming ParaView pipeline | particle dynamics | experimental |
| run_phase1_5.py | Enhanced diagnostics runner | experiment demo | experimental |
| standing_clarity_diagnostics.py | Standing-wave clarity S1–S4 | validation | core |
| standing_resonance_sweep.py | Fine H_top sweep for vertical resonance | parameter sweep | useful |
| trap_localisation_debug_standing.py | Debug unstable trap counts + z-plane sensitivity | metrics analysis | experimental |
| trap_localisation_validation_study.py | Validate detected traps as real stable minima | metrics analysis | experimental |
| vortex_3d_diagnostics.py | Vortex topology, power-flow, PML absorption | validation | core |
| vortex_3d_hires.py | 3D hi-def snapshots + VTU export | export paraview | useful |
| vortex_bridge_design_study.py | Lens config ranking for bridge creation | experiment demo | experimental |
| vortex_convergence_spotcheck.py | Ranking stability + vortex authority gate | validation | core |
| vortex_function_audit.py | 20-question vortex superposition audit (§1–6) | metrics analysis | core |
| vortex_lens_sweep.py | Lens family sweep: LG/Bessel/BG/Plastic | parameter sweep | experimental |
| vortex_minimum_mobility.py | Locking→sliding transition (offset × α) | metrics analysis | experimental |
| vortex_only_hires.py | Hi-res vortex-only snapshot | visualisation | useful |
| vortex_standing_balance.py | Magnitude mismatch + amplitude sweep | metrics analysis | core |
| vortex_static_authority.py | Static particle authority comparison | metrics analysis | core |
| zprog_diagnostic.py | z-height pressure diagnostic | metrics analysis | experimental |

### scripts/analysis/ (20 scripts)

| Script | Purpose | Category | Status |
|--------|---------|----------|--------|
| diagnostics_utils.py | Shared utilities for Gor'kov minima, convergence metrics | utility | core |
| debug_disc_bc_case_a.py / debug_disc_neumann_vs_robin.py / debug_vortex_sign.py | BC debugging | validation | useful |
| energy_budget.py | Power/energy analysis | metrics analysis | useful |
| export_comsol_parallel_figures.py | COMSOL comparison figure export | visualisation | core |
| fix_vortex_comsol_match.py | COMSOL vortex sign fix | validation | useful |
| investigate_comsol_discrepancy.py | COMSOL discrepancy forensics | validation | useful |
| rebuild_comsol_comparison.py (×3) | Iterative COMSOL comparison rebuilds | validation | core |
| run_batch1_outputs.py / run_batch2a_trap_atlas.py | Batch output generation | visualisation | useful |
| run_comsol_validation_lockdown.py | Lock COMSOL validation cases | validation | core |
| run_phase2_storyboard.py | Phase 2 storyboard | visualisation | useful |
| render/bp4_to_vtu.py, render_field_pyvista.py, render_vortex_3d.py | ParaView rendering | export paraview | useful |
| visualization/* | Phase sweep, canonical rendering, 2D validation | visualisation | useful |

### scripts/validation/ (28 scripts)

| Script | Purpose | Category | Status |
|--------|---------|----------|--------|
| test_1d_impedance.py | 1D impedance reflection coefficient | validation | core |
| test_2d_helmholtz.py | 2D Helmholtz convergence | validation | core |
| test_energy_balance.py | Power injection = power absorbed | validation | core |
| test_pml_*.py (6 scripts) | PML absorption, reflection, smoke tests | validation | core |
| test_fem_modules.py / test_helmholtz_complex.py | Core solver tests | validation | core |
| test_petri_dish_bcs.py / test_interface_continuity.py | BC/coupling verification | validation | core |
| deliverable1/2/3_*.py | Deliverable validation scripts | validation | core |
| export_vortex_3d.py / vortex_simple.py | Vortex export & verification | export/validation | useful |

---

## PART 3 — CAPABILITY INVENTORY

| # | Capability | How Produced | Key Scripts | Outputs | Importance |
|---|-----------|-------------|-------------|---------|------------|
| 1 | Standing wave field generation (FEM) | FEniCSx Helmholtz + PML at 6 elem/λ, 762K DOFs, MUMPS | fem_standing_plus_asm_vortex_local_3x3.py | NPZ cache (15 MB), PNG field plots | **Critical** — foundation for all trap studies |
| 2 | Trap detection & validation | Morphological minima + sub-grid quadratic fit + Hessian eigenvalue check | trap_localisation_validation_study.py, full_domain_gorkov_diagnostic.py | JSON (trap coords), CSV (metrics), PNG (overlays) | **Critical** — 15 real traps vs 203 false positives (old) |
| 3 | Gor'kov potential & radiation force | Corrected formula: $U = \frac{2\pi}{3}a^3[f_1\frac{|p|^2}{2\rho c^2} - f_2\frac{3\rho|\nabla p|^2}{4\omega^2\rho^2}]$ | gorkov.py, physics_affirmation.py | NPZ fields, PNG potential maps | **Critical** — validated Feb 2026 |
| 4 | ASM (Angular Spectrum) propagation | FFT-based Rayleigh–Sommerfeld diffraction | angular_spectrum.py | Complex pressure at arbitrary z-planes | **Critical** — enables vortex analysis without FEM |
| 5 | Vortex lens generation & families | Phase-winding $\phi = \ell\theta$, plastic lens thickness encoding | vortex_lens.py, vortex_lens_sweep.py | Lens profiles, field galleries | **High** — core manipulation element |
| 6 | FEM + ASM hybrid superposition | Standing wave (FEM) + vortex (ASM) overlay at matched resolution | fem_standing_plus_asm_vortex_local_3x3.py | Combined field PNG, α sweep panels | **High** — validates combination physics |
| 7 | Vortex hourglass characterisation | ASM propagation through focal plane, waist metrics | rs_free_space_vortex_truth.py, rs_hourglass_proof.py | XY/XZ panels, VTU export, JSON waist metrics | **High** — proves correct vortex structure |
| 8 | Bridge barrier analysis | Gor'kov saddle-point energy between trap pair | bridge_master_study.py, bridge_phase_offset_study.py | Heatmaps (α,φ₀,offset), CSV metrics, REPORT.md | **High** — 89% barrier reduction demonstrated |
| 9 | C-shape selective perturbation design | Asymmetric angular winding: $p_C = A(r)W(\theta)e^{im\theta}$ | c_shape_on_cached_sw_geometry_study.py | 144-candidate sweep, leakage metrics, best geometry | **High** — asymmetry > 0.99 achieved |
| 10 | Lens backpropagation (ASM inverse) | Target field → hologram via ASM reverse propagation | c_shape_backprop_phase_to_thickness.py | Phase lens, thickness profile, reconstruction fidelity | **Medium** — 50% amplitude correlation (phase-only ceiling) |
| 11 | Manufacturing-ready lens CAD | Flat-bottom lens: masked smoothing, slope limiting, Fresnel analysis | c_shape_lens_15mm_manufacturing_study.py | 3D solid renders, thickness maps, smoothing sweep CSV | **Medium** — ready for fabrication attempt |
| 12 | Lens × standing-wave node-amplification physics | Combined field $p_{comb} = p_{sw} + \alpha e^{i\psi} p_{lens}$ at trap nodes | c_shape_lens_15mm_overlay_study.py | α/ψ sweep panels, perturbation maps, asymmetry metrics | **High** — node-amplified effect (+539% at 23% peak) |
| 13 | Particle transport via crossfade protocol | Overdamped dynamics with ramped $\beta_{sw}(t), \beta_{lens}(t)$ schedule | c_shape_particle_merge_gif_demo.py | GIF animations, trajectory plots, distance-vs-time CSV | **High** — visual proof of merge concept |
| 14 | Transport optimisation (windowed, partial SW) | 36-config sweep: window radius, β_sw_min, ramp speed | c_shape_transport_refinement_study.py | GIF (baseline vs refined), comparison bars, heatmaps | **Medium** — 17% neighbour improvement |
| 15 | COMSOL cross-validation | Independent solver comparison (3 cases: standing, vortex, combined) | rebuild_comsol_comparison.py, compare_to_comsol.py | Side-by-side PNGs, CSV metrics, MANIFEST.txt | **Critical** — validates entire physics stack |
| 16 | PML verification suite | Reflection coefficient, absorption profile, operator checks | test_pml_*.py, farfield_pml_operator_check.py | Pass/fail tests, reflection metrics | **Critical** — ensures non-reflecting boundaries |
| 17 | Acoustic streaming | Reynolds stress → Stokes equations for time-averaged flow | streaming.py, run_complex_streaming_diagnostics.py | Streaming velocity fields | **Medium** — physics complete but not central to story |
| 18 | ParaView 3D export | VTU/XDMF with real/imag/mag/phase decomposition | export_paraview.py, export_fields.py, render_*.py | VTU/XDMF + manifest, PyVista renderings | **Medium** — 3D visualisation capability |
| 19 | Automated INDEX/CSV/JSON reporting | Per-run metadata, metrics tables, audit reports | report.py, audit.py | INDEX.md, metrics.csv, config.json, MANIFEST.json | **High** — reproducibility infrastructure |
| 20 | Multi-z field visualisation | Z-stack slice galleries and XZ meridional cross-sections | _solve_worker_multi_z.py, optimal_geometry_gallery.py | Multi-panel PNG figures | **Medium** — depth understanding |

---

## PART 4 — RESULTS INVENTORY

### Active Results (35 directories)

| Result Folder | Study Type | Script Used | Key Outputs | Key Conclusion | Quality |
|---------------|------------|-------------|-------------|----------------|---------|
| fem_standing_wave_cache/ | FEM cache | fem_standing_plus_asm_vortex_local_3x3 | NPZ (762K DOFs, epl6), INFO.txt | Production cache at 6 elem/λ, max\|p\| = 60.6 Pa | **final-quality** |
| COMSOL_comparison_results/ | Cross-validation | rebuild_comsol_comparison | 3 cases × (figs + CSV + meta) | Standing 81.68 Pa matches COMSOL theory; physics validated | **final-quality** |
| trap_localisation_validation_20260308/ | Trap audit | trap_localisation_validation_study | report.md, 11 CSVs, figs | 15 real traps (vs 203 false), sub-grid Hessian refinement works | **final-quality** |
| trap_localisation_debug_standing_20260308/ | Trap debug | trap_localisation_debug_standing | report.md, 4 CSVs, figs | IDW artefact discovered: F_max ∝ N², force criterion invalid | **useful evidence** |
| full_domain_gorkov_diagnostic_20260310/ | Full-domain trap map | full_domain_gorkov_diagnostic | 3 PNGs, JSON, NPZ | 15 traps with spacing histogram; foundation for targeting | **useful evidence** |
| bridge_master_study_20260305/ | Bridge parameter sweep | bridge_master_study | REPORT.md, data/, figs/ | 89.4% barrier reduction at α=0.50, φ₀=158° — barrier NOT eliminated | **useful evidence** |
| bridge_phase_offset_study_20260305/ | Bridge phase study | bridge_phase_offset_study | REPORT.md, data/, figs/ | 53.8% barrier reduction at α=0.20 — confirms limitation | **useful evidence** |
| c_shape_on_cached_sw_..._20260310_102151/ | Geometry screening | c_shape_on_cached_sw_geometry | INDEX.md, 432+ PNGs, CSV, NPZ | Best: asymmetry=0.9935, B_supp=0.9934, leak_max≈0 | **final-quality** |
| c_shape_on_cached_sw_..._20260309_* (×3) | Earlier geometry runs | same | INDEX.md, PNGs, CSV | Iterative refinement → final 102151 run was definitive | **exploratory** |
| c_shape_backprop_..._20260310_112549/ | Lens design | c_shape_backprop_phase_to_thickness | INDEX.md, NPZ, PNGs, CSV | 10mm lens: energy capture 90.7%, amp_corr 0.55 (phase-only limit) | **useful evidence** |
| c_shape_backprop_..._20260310_112336/ | Earlier lens run | same | Same structure | Superseded by 112549 | **exploratory** |
| c_shape_lens_15mm_mfg_20260310/ | Manufacturing study | c_shape_lens_15mm_manufacturing | INDEX.md, 44 PNGs, CSV, NPZ | 15mm lens: 95.7% energy capture, 50% fidelity; Fresnel steps dominate | **useful evidence** |
| c_shape_lens_15mm_overlay_20260310/ | Overlay physics | c_shape_lens_15mm_overlay | INDEX.md, 21 PNGs, CSV, NPZ | Node-amplified perturbation: +539% at trap A, −3.4% at B; ψ=3π/2 best | **final-quality** |
| c_shape_particle_merge_..._20260310_182422/ | Merge demo | c_shape_particle_merge_gif_demo | GIF, 8 PNGs, CSV, INDEX.md | Particle A merged into B (344.9 µm), but neighbours 227 µm disturbed | **useful evidence** |
| c_shape_particle_merge_..._20260310_* (×6) | Earlier merge runs | same | GIFs, PNGs | Iterative refinement of merge protocol | **exploratory** |
| c_shape_transport_refine_20260311_091608/ | Transport refinement | c_shape_transport_refinement | 2 GIFs, 11 PNGs, CSV, INDEX.md | Best: β_sw_min=0.3 → neighbour 189 µm (17% improvement); 8/36 merges | **useful evidence** |
| c_shape_transport_refine_20260311_* (×5) | Earlier refinement runs | same | GIFs, PNGs | Progressive debugging of transport protocol | **exploratory** |
| c_shape_transport_refine_20260310_* (×2) | Initial refinement | same | GIF, PNGs | First attempts at windowed transport | **exploratory** |
| _mobility_fem_cache/ | Grid cache | (internal) | 18 NPZ files | Pre-gridded fields for mobility experiments | **useful evidence** |
| _archive/ (23 items) | Superseded | Various | ARCHIVE_LOG.md | Physics affirmation, early vortex audits, resonance sweeps archived | **obsolete** |

### Output Artifact Summary

| Type | Count | Example |
|------|-------|---------|
| PNG figures | 2,491 | Field plots, parameter heatmaps, trajectory overlays |
| CSV tables | 58 | Trap data, metrics sweeps, smoothing diagnostics |
| GIF animations | 16 | Particle merge demos, baseline/refined comparisons |
| NPZ data | 266 | Field caches, grid data, lens profiles |
| INDEX.md | 19 | Per-study structured summaries |
| JSON configs | 49 | Run parameters, trap coordinates, audit manifests |

---

## PART 5 — PROJECT DEVELOPMENT TIMELINE

### Phase 1 — 2D Helmholtz & Path Control (Dec 2025 – Jan 2026)

**Goal:** Build foundational 2D acoustic solver and demonstrate particle control.

**Developed:**
- 2D Neumann Helmholtz solver (Dec 29)
- 2.5D model with moving transducers (Dec 31)
- Gor'kov landscape, Bayes optimisation, greedy + MPC controllers (Jan 14–17)
- K-step adjoint trajectory optimiser (Jan 16)
- Path-tracking comparison: MPC clearly outperforms greedy (Jan 17–18)

**Learned:** MPC steering is feasible but 2D surrogate model insufficient for real physics; need full 3D FEM.

### Phase 2 — 3D FEM Implementation (Jan 23 – Feb 7)

**Goal:** Build 3D FEniCSx solver for realistic petri-dish geometry.

**Developed:**
- Robin-BC Helmholtz 3D framework (Jan 23)
- Multiple failed FEniCSx attempts (Jan 24–26: "also not working lol", "shite version", "non functioning fem what da hell")
- Breakthrough: functioning 3D standing waves confirmed (Feb 7)
- ParaView rendering, phase sweeping, vortex lens overlays (Feb 7)

**Learned:** FEniCSx complex-scalar PETSc setup is non-trivial; PML placement critical ("removed PML on the inside of physical domain duh").

### Phase 3 — Streaming, COMSOL Validation, & Linux Deployment (Feb 11 – Feb 22)

**Goal:** Add streaming physics, validate against COMSOL, deploy to Linux lab machine.

**Developed:**
- Overdamped + Reynolds streaming (Feb 11)
- Full 3D FEM with streaming confirmed (Feb 12)
- COMSOL coherence achieved for 10mm dish (Feb 15)
- Linux deployment: conda environment, CLI, guardrails, AUDIT.md (Feb 19–22)

**Learned:** COMSOL validates physics stack; standing-wave BC height bug discovered and fixed via PR #1.

### Phase 4 — Vortex Investigation & Lens Families (Feb 23 – Feb 25)

**Goal:** Explore vortex beam physics and characterise lens types.

**Developed:**
- Vortex-only hires runs, 3D z-axis scans (Feb 23–24)
- Lens family analysis: LG, Bessel, BG, Plastic (Feb 25)
- Vortex function audit (20-question systematic evaluation)
- Physics affirmation sprint: Gor'kov correction validated

**Learned:** FEM resolution too low for vortex beams → need Rayleigh-Sommerfeld / ASM approach.

### Phase 5 — ASM Canonical Module & Hourglass Validation (Mar 1 – Mar 3)

**Goal:** Build production ASM propagator and prove correct vortex hourglass.

**Developed:**
- Canonical ASM module with plastic lens preset (Mar 1–3)
- VTU geometry bug fix, hourglass forensics
- Free-space vortex truth pack (ℓ=2 validation)
- **Sign convention discovery**: focusing phase must be NEGATIVE
- Aperture scaling analysis (R=1–5 mm)
- Hourglass visual proof: XY panel + XZ meridional

**Learned:** Correct sign convention is critical; hourglass requires R ≥ 3mm for convergent structure.

### Phase 6 — FEM–ASM Hybrid & Standing Wave Cache (Mar 4 – Mar 5)

**Goal:** Overlay ASM vortex onto FEM standing wave at production resolution.

**Developed:**
- 6 elem/λ standing-wave cache (762K DOFs, 151s solve, reusable)
- MUMPS icntl prefix bug fix (critical for >440K DOFs)
- 3D LinearND → 2D slab interpolation fix (26 min → 2s)
- Phase investigation: overlay of vortex and standing successful
- Bridge mechanism initiated

**Learned:** MUMPS options require DOLFINx-specific prefix; 3D interpolation must be projected to 2D slabs.

### Phase 7 — Bridge Formation Studies (Mar 5 – Mar 8)

**Goal:** Eliminate Gor'kov barrier between adjacent traps using vortex interference.

**Developed:**
- Bridge phase offset study: 800 configs, 53.8% barrier reduction
- Bridge master study: 19,360 configs, 89.4% barrier reduction
- Functioning FEM+RS overlay confirmed

**Learned:** Barrier reduction is monotonic with α but never reaches zero → bridge approach insufficient for full elimination.

### Phase 8 — Trap Localisation & IDW Artefact Discovery (Mar 8 – Mar 9)

**Goal:** Validate trap detection before proceeding with manipulation.

**Developed:**
- Trap localisation validation: 15 real traps confirmed (old method: 203 false)
- IDW interpolation artefact discovered: sub-DOF sampling creates artificial gradients
- F_max ∝ N² (resolution-dependent force criterion)
- Production epl6 FEM checkpoint (7mm depth)

**Learned:** **Critical**: IDW interpolation creates C⁰ gradients that corrupt force-based metrics at sub-DOF scales; force criterion must be removed or replaced.

### Phase 9 — C-Shape Perturbation Strategy (Mar 9 – Mar 10)

**Goal:** Design asymmetric pressure perturbation that selectively affects one trap.

**Developed:**
- C-shape geometry screening: 144 candidates → best asymmetry 0.9935
- ASM backpropagation: target field → lens plane (8.3λ propagation)
- Printable thickness profile (10mm: amp_corr 0.55; 15mm: amp_corr 0.50)
- Full-domain Gor'kov diagnostic (15 traps confirmed)

**Learned:** C-shape angular windowing achieves near-perfect selectivity (asymmetry > 0.99); single-pass phase-only reconstruction hits 50% fidelity ceiling.

### Phase 10 — Manufacturing, Overlay Physics, & Particle Transport (Mar 10)

**Goal:** Physical lens design + demonstrate particle manipulation.

**Developed:**
- 15mm manufacturing study: masked smoothing, slope limiting, 3D CAD render
- Overlay study: node-amplified perturbation (+539% at trap A, −3.4% at B)
- ψ=3π/2 optimal phase offset
- Particle merge GIF demo: crossfade protocol, particle A merged into B
- 7 iterative merge demo runs

**Learned:** Node-amplified effect is the key physics: weak lens at pressure nodes creates large relative perturbation; neighbours are the challenge (227 µm displacement).

### Phase 11 — Transport Refinement & Protocol Optimisation (Mar 10 – Mar 11)

**Goal:** Reduce neighbour disruption during particle transport.

**Developed:**
- 36-config parameter sweep: window radius, β_sw_min, ramp speed
- Best: full lens + 30% SW retention → 17% neighbour improvement (189 µm vs 227 µm)
- Windowed lens variants: 0/12 merges (insufficient force)
- Translation protocols: no improvement over static
- 8 iterative refinement runs

**Learned:** Partial SW retention (β_sw_min=0.3) is the most effective strategy; spatial windowing is too aggressive; protocol remains partial_success.

---

## PART 6 — STRONGEST RESULTS

### 1. COMSOL Cross-Validation

**Why important:** Independent verification of the entire FEM physics stack against a commercial solver. Standing-wave peak pressure matches COMSOL theory (81.68 Pa). This validates every downstream result.

**Key figures:** figs, standing/vortex/combined isosurface comparisons.

**Should appear in final report:** Yes — essential validation section. Shows side-by-side pressure field comparisons.

### 2. Full-Domain Trap Detection (15 Traps)

**Why important:** Demonstrates that the validated standing wave produces a regular trap lattice with physically meaningful 0.371 mm spacing (λ/2). The old-vs-new comparison (203 → 15 traps) demonstrates rigorous methodology.

**Key figures:** full_domain_gorkov_traps.png, trap spacing histogram, centre zoom.

**Should appear in final report:** Yes — establishes the baseline trap environment.

### 3. C-Shape Geometry Screening (Asymmetry > 0.99)

**Why important:** Systematic 144-candidate sweep proving that angular-windowed C-shape perturbation can achieve near-perfect selectivity (asymmetry 0.9935, B-suppression 0.9934, leakage ≈ 0).

**Key figures:** figures — best candidate overlays showing targeted energy redistribution.

**Should appear in final report:** Yes — demonstrates design methodology and optimal parameters.

### 4. Node-Amplified Perturbation Physics

**Why important:** The most surprising result — a lens field at only 23% of SW peak produces +539% perturbation at trap nodes. This is the mechanism enabling selective manipulation without global field disruption.

**Key figures:** D_alpha_sweep_summary.png, E_psi_sweep_summary.png.

**Should appear in final report:** Yes — central physics insight driving the manipulation strategy.

### 5. Particle Merge GIF Demonstration

**Why important:** Visual proof-of-concept that particle A can be transported 345 µm into trap B's basin while B remains trapped (2.4 µm displacement). First successful merge in the project.

**Key figures:** particle_merge_demo.gif, trajectory overlay, distance-vs-time plot.

**Should appear in final report:** Yes — compelling visual for the primary narrative.

### 6. Vortex Hourglass Proof

**Why important:** Validates the ASM propagation framework by showing correct convergence→waist→divergence structure with topological charge ℓ=2.

**Key figures:** XY cross-section panel + XZ meridional slice from rs_free_space_vortex_truth.py and rs_hourglass_proof.py.

**Should appear in final report:** Yes — validates propagation method.

### 7. Bridge Barrier Reduction Heatmaps

**Why important:** 19,360-configuration systematic parameter sweep demonstrating 89.4% barrier reduction. While bridge is not fully eliminated, the monotonic α-dependence and phase sensitivity are informative for the narrative.

**Key figures:** bridge_metric_heatmaps.png, best bridge Gor'kov profile.

**Should appear in final report:** Yes, but framed as motivation for evolving toward C-shape approach.

### 8. Transport Refinement Comparison

**Why important:** Systematic 36-configuration optimisation showing that partial SW retention improves neighbour stability by 17%. Includes baseline-vs-refined GIF comparison.

**Key figures:** gif (baseline_merge.gif, refined_merge.gif), comparison bars, heatmaps.

**Should appear in final report:** Yes — demonstrates optimisation methodology and current limitations.

---

## PART 7 — MISSING BUT EASY ADDITIONS

### 1. Summary Comparison Panel of Manipulation Strategies

**Why it helps:** A single figure showing bridge approach (89.4% barrier reduction) vs C-shape approach (0.99 asymmetry + successful merge) would make the narrative progression visually clear.

**Effort:** Low — data already exists in two result folders; needs a single matplotlib script compositing existing PNGs or recomputing from cached NPZ fields.

### 2. Final-Quality 15-Trap Lattice Diagram

**Why it helps:** A publication-quality figure showing the full 6×6 mm domain with all 15 traps, λ/2 spacing annotations, and the selected A-B pair highlighted would anchor the introduction.

**Effort:** Low — data in full_domain_gorkov_diagnostic NPZ and JSON; needs one focused plotting script.

### 3. Clean Parameter Sweep Summary Figure

**Why it helps:** The 144-candidate C-shape sweep and 19,360-config bridge sweep both exist as raw heatmaps; a curated 2–3 panel figure with clear axis labels and colourbar legend would be publication-ready.

**Effort:** Low — replot from existing CSV/NPZ data with tighter formatting.

### 4. α-ψ Phase Space Map (Overlay Study)

**Why it helps:** Combining the α sweep (4 values) and ψ sweep (4 values) into a single 4×4 grid panel would provide a complete view of the perturbation parameter space.

**Effort:** Low — all 16 panels already exist as individual PNGs; needs compositing script.

### 5. Baseline-vs-Refined Transport Side-by-Side Figure

**Why it helps:** The two GIFs (baseline_merge.gif and refined_merge.gif) could be rendered as a multi-panel static figure with key frames at t=0, t=200ms (peak), t=500ms (settled).

**Effort:** Low — extract frames from existing GIFs using imageio/PIL.

### 6. Pipeline/Workflow Diagram

**Why it helps:** A Mermaid or TikZ diagram showing: FEM solve → Cache → ASM propagation → C-shape design → Lens backprop → Manufacturing → Overlay → Particle dynamics would clarify the methodology.

**Effort:** Very low — diagram only, no code to run.

### 7. Rerun Best Merge at Higher Time Resolution

**Why it helps:** Current merge GIFs use coarse time steps; running the best configuration (α=5, β_sw_min=0.3) at 2× time resolution would produce smoother animation and more precise distance-vs-time curves.

**Effort:** Low — modify dt in existing script, ~5 min runtime.

### 8. IDW Artefact Ablation Figure

**Why it helps:** The IDW artefact discovery (F_max ∝ N²) is a key methodological finding. A clean 3-panel figure showing force field at N=200, N=800, N=1600 would clearly illustrate the problem.

**Effort:** Low — data already exists in trap_localisation_debug results; needs formatting.

### 9. Standardise Older Bridge Studies to INDEX/CSV Format

**Why it helps:** The bridge studies (Mar 5) use REPORT.md format while newer studies use INDEX.md + metrics.csv; standardising would unify the results directory.

**Effort:** Low — write a retrospective INDEX.md from existing REPORT.md content.

### 10. Manufacture Comparison Table

**Why it helps:** A table comparing 10mm vs 15mm lens: energy capture, fidelity, Fresnel step count, slope stats. Data already in manufacturing study.

**Effort:** Very low — extract from existing metrics.csv.

---

## PART 8 — PROPOSED FINAL REPORT STRUCTURE

### 1. Introduction & Motivation

**Content:** Acoustic tweezer concept; selective micro-particle manipulation goal; why standing-wave lattice + vortex lens combination; project scope.

**Repo evidence:** README.md §1.

### 2. Acoustic Environment & Baseline Trap Lattice

**Content:** Standing-wave generation in shallow petri dish (6×6 mm, 2 MHz); Gor'kov potential theory; trap detection methodology (old vs new); full-domain 15-trap lattice characterisation.

**Repo evidence:** FEM cache (epl6), full_domain_gorkov_diagnostic, trap_localisation_validation.

### 3. Modelling Framework

**Content:** FEniCSx architecture; Helmholtz + PML weak form; complex PETSc scalars; MUMPS direct solver; mesh convergence (4–6 elem/λ); physics hierarchy (7 levels); ASM module for vortex propagation.

**Repo evidence:** src/acoustweezers module structure, CHANGELOG MUMPS fix, validation tests.

### 4. Validation Against COMSOL

**Content:** 3-case comparison (standing, vortex, combined); pressure field agreement; independent verification of physics model.

**Repo evidence:** COMSOL_comparison_results Cases A/B/C, COMSOL_RECREATION_SPEC.md.

### 5. Vortex Beam Characterisation

**Content:** Lens families (LG, Bessel, BG, Plastic); ASM propagation validation; hourglass proof; sign convention discovery; aperture scaling.

**Repo evidence:** rs_free_space_vortex_truth, rs_hourglass_proof, rs_plastic_lens_hourglass_demo.

### 6. Early Manipulation Strategy: Bridge Formation

**Content:** Hypothesis: vortex + standing wave interference can eliminate inter-trap barrier; 19,360-config parameter sweep; 89.4% barrier reduction; why barrier is never fully eliminated → motivates C-shape approach.

**Repo evidence:** bridge_master_study, bridge_phase_offset_study.

### 7. C-Shape Selective Perturbation Design

**Content:** Asymmetric angular winding; 144-candidate geometry screening (asymmetry > 0.99); design rationale; leakage metrics; best geometry selection.

**Repo evidence:** c_shape_on_cached_sw_geometry_study_20260310_102151.

### 8. Holographic Lens Design & Manufacturing

**Content:** ASM backpropagation; phase-only reconstruction (50% fidelity ceiling); 15mm manufacturing study; masked smoothing; slope limiting; Fresnel step analysis.

**Repo evidence:** c_shape_backprop_phase_to_thickness, c_shape_lens_15mm_manufacturing_study.

### 9. Node-Amplified Perturbation Physics

**Content:** Key discovery: weak lens at pressure nodes creates disproportionate relative perturbation (+539% at trap A with only 23% absolute contribution); α and ψ parameter space; optimal ψ=3π/2.

**Repo evidence:** c_shape_lens_15mm_overlay_study.

### 10. Particle Transport & Merge Demonstration

**Content:** Crossfade protocol; overdamped dynamics; successful merge (345 µm transport, B stationary); neighbour disruption challenge (227 µm → 189 µm with optimisation); 36-config refinement sweep.

**Repo evidence:** c_shape_particle_merge_demo_20260310_182422, c_shape_transport_refinement_study_20260311_091608.

### 11. Metrics, Workflow, & Reproducibility

**Content:** Automated INDEX/CSV/JSON reporting; config.json per run; MANIFEST.json for environment; 28 validation tests; standardised diagnostics pipeline.

**Repo evidence:** core/report.py, core/audit.py, validation test suite.

### 12. Discussion & Limitations

**Content:** IDW interpolation artefact; force criterion invalidity; phase-only lens fidelity ceiling; neighbour stability as primary remaining challenge; windowed lens insufficient; bridge approach evolutionary dead end.

**Repo evidence:** trap_localisation_debug, transport refinement negative results.

### 13. Future Work

**Content:** Gerchberg-Saxton iterative phase retrieval; FEM shape-function interpolation (replacing IDW); multi-particle ensemble simulations; experimental lens fabrication and testing; streaming-coupled transport.

**Repo evidence:** Deferred items noted in INDEX.md files and CHANGELOG.

---

*Report generated from systematic exploration of 79 commits, 289 source/script files (~130K lines), 35+ result directories, 2,491 PNG figures, 16 GIF animations, and 58 CSV metric tables across the acousto-tweezers repository.* 

Completed: *Compile final report* (8/8)

The report above covers all 8 requested parts. Key highlights:

- **289 source/script files** (~130K lines of Python) across a production-grade FEniCSx acoustic tweezer stack
- **35+ result directories** containing 2,491 PNGs, 16 GIFs, 58 CSVs, 266 NPZs
- **11 development phases** from 2D Helmholtz (Dec 2025) through transport refinement (Mar 11, 2026)
- **8 strongest results** identified for the final document, led by COMSOL validation, node-amplified perturbation physics, and particle merge GIF demonstrations
- **10 easy additions** that would strengthen the narrative using existing data
- **13-section final report structure** proposed as a research narrative from motivation through future work