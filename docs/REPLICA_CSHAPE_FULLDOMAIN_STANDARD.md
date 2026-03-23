# Canonical Method Report: Replica C-Shape Full-Domain Translated Transport

## Decision
As of 2026-03-17, this is the canonical repo method for replica-lens C-shape transport side-by-side studies.

This method replaces ROI-only synthetic C-shape overlays for this workflow.
All new side-by-side replica C-shape deliverables must follow this pipeline unless a documented change request is approved.

## Why This Was Standardized
The standardized run produced strong transport behavior with physically consistent field construction:
- Selected alpha: 10.5
- A_move_max: 393.85 um
- A_move_final: 389.20 um
- d_AB_final: 0.00 um

Source run outputs:
- results/deliverables/transport_side_by_side/transport_vortex_vs_cshape_gorkov_const_sw1_translated_replica_fullfield.gif
- results/deliverables/transport_side_by_side/replica_cshape_full_domain_field.npz
- results/deliverables/transport_side_by_side/replica_cshape_alpha_tuning.json

## Scope
This standard applies to:
- side-by-side transport comparisons where the C-shape perturbation is derived from the replica lens pipeline
- full-domain Gor'kov rendering and ROI crop views in the same figure
- translated C-shape transport schedules (A->B) with alpha tuning for measurable A movement

This standard does not replace:
- vortex-only transport pipelines
- legacy ROI-local C-shape exploration scripts used for historical comparison

## Exact Canonical Pipeline
Implemented in:
- scripts/deliverables/transport_side_by_side_replica_cshape.py

Step 1: Build replica lens using IASA
- Use ReplicaConfig and run_iasa from scripts/dev/inverse_c_shape_lens_replica.py.
- Solve for lens_field on the physical lens aperture grid.

Step 2: Convert wrapped phase to thickness profile
- phi_wrapped = mod(angle(lens_field), 2*pi)
- thickness = h_base + h_max * (phi_wrapped / (2*pi)) inside aperture

Step 3: Convert thickness to transmitted aperture phase
- phase_delay = abs(k_lens - k_water) * (thickness - h_base)
- aperture_field = exp(i * mod(phase_delay, 2*pi)) inside aperture

Step 4: Propagate real lens aperture field with ASM
- Propagate aperture_field to focal plane on the lens/full domain basis.
- This is the only C-shape perturbation source used downstream.

Step 5: Resample once onto full transport grid
- Interpolate propagated field to full transport coordinates.
- Save as centered full-domain perturbation basis field.

Step 6: Use translated full-domain C-shape field in transport
- Use translation-capable perturbation generator on full grid.
- During active phase, move center from trap A toward trap B (when translation enabled).

Step 7: Auto-tune alpha until A moves
- Evaluate alpha candidates from initial alpha with multiplicative growth.
- Stop at first candidate where A_move_max >= A_MOVE_THRESHOLD_UM.
- Save tuning table and selected alpha.

Step 8: Render full-domain and ROI-crop consistently
- Compute U_Gorkov on full domain for both vortex and C-shape arms.
- Top-row ROI panels must be direct crops from full-domain U arrays.
- Bottom-row panels are full-domain U arrays.

## Non-Negotiable Method Invariants
1. No separate ROI-only C-shape solve for production replica side-by-side outputs.
2. No synthetic C-shape overlay used to represent full-domain physics.
3. ROI must be a crop of the same full-domain arrays shown in bottom panels.
4. C-shape perturbation must be translated in-plane for transport push studies.
5. Alpha selection must be logged and traceable.

## Canonical Preset
Preset file:
- configs/cases/replica_cshape_fullfield_transport_standard.json

Primary run wrapper:
- scripts/deliverables/replica_cshape_transport_run_all.py

The preset fixes method-critical controls (full-domain pipeline, translation, auto-tuning, replica lens parameters, and output naming).

## Standard Reproduction Commands
Run canonical method:

```bash
python scripts/deliverables/replica_cshape_transport_run_all.py
```

Optional override for output filename:

```bash
python scripts/deliverables/replica_cshape_transport_run_all.py \
  --out-gif-name transport_vortex_vs_cshape_gorkov_const_sw1_translated_replica_fullfield_rerun.gif
```

Dry-run (inspect resolved canonical env values):

```bash
python scripts/deliverables/replica_cshape_transport_run_all.py --dry-run
```

## Output Contract (Required Artifacts)
A valid canonical run must produce all of:
- results/deliverables/transport_side_by_side/transport_vortex_vs_cshape_gorkov_const_sw1_translated_replica_fullfield.gif
- results/deliverables/transport_side_by_side/replica_cshape_full_domain_field.npz
- results/deliverables/transport_side_by_side/replica_cshape_alpha_tuning.json
- results/deliverables/transport_side_by_side/replica_cshape_method_manifest.json

## Acceptance Checklist
- [ ] Full-domain field package exists and contains phase_wrapped + thickness + p_cshape_full_centered.
- [ ] Alpha tuning log exists and records selected alpha with movement metrics.
- [ ] Method manifest exists and records canonical method id + run controls.
- [ ] Generated GIF uses ROI crops from full-domain arrays, not independent ROI solves.
- [ ] C-shape translation is enabled for transport push runs.

## Repository Integration Points
- scripts/deliverables/transport_side_by_side_replica_cshape.py
- scripts/deliverables/replica_cshape_transport_run_all.py
- configs/cases/replica_cshape_fullfield_transport_standard.json
- docs/REPLICA_CSHAPE_FULLDOMAIN_STANDARD.md
- README.md (canonical method references)

## Change Control
If this method is changed, update all of:
1. configs/cases/replica_cshape_fullfield_transport_standard.json
2. scripts/deliverables/replica_cshape_transport_run_all.py
3. scripts/deliverables/transport_side_by_side_replica_cshape.py
4. docs/REPLICA_CSHAPE_FULLDOMAIN_STANDARD.md
5. README.md

Any merge that changes only one of these is considered incomplete.
