# Report-Facing Results

These folders gather the result outputs most closely related to the final
report narrative.

## Lens Design

`lens_design/`

- `c_shape_backprop_phase_to_thickness_20260310_112549/` - C-shape back
  propagation, phase maps, thickness maps and reconstruction comparisons.
- `c_shape_lens_15mm_manufacturing_study_20260310_153032/` - printable lens
  geometry, heightmaps, STL exports and manufacturing-resolution studies.
- `c_shape_lens_15mm_overlay_study_20260310_170620/` - C-shape lens overlay
  with the standing-wave reference field.
- `c_shape_particle_merge_demo_20260310_182422/` - C-shape particle transport
  and merge demonstration outputs.

## Integrated Model

`integrated_model/`

- `mesh_convergence/` - mesh convergence figures and tables used to justify
  the finite-element resolution.
- `vortex_phase_control/` - MPC and greedy vortex phase-control figures,
  Gorkov landscape sequences and phase-offset plots.
- `trap_localisation_validation_20260308_132247/` - standing-wave trap
  localisation validation outputs.
- `trap_localisation_debug_standing_20260308_140901/` - trap detection and
  z-plane sensitivity diagnostics.
- `vortex_transport_localisation_study_20260311_142210/` - vortex transport
  sweep figures, trajectory plots and GIFs.
- `full_domain_gorkov_diagnostic_20260310_110953/` - full-domain Gorkov trap
  diagnostic outputs.

## Report Figures

`report_figures/`

Contains generated figures used while assembling the report story, including
phase-barrier, C-shape failure, IASA progression and bridge/corridor diagnostics.

## Experimental Validation

The report's experimental photos and lab images are primarily writing assets,
not all of which are stored in this code repository. Current computational
validation against COMSOL is in `../comsol_comparison/`; older failed attempts
are archived in `../../archive/comsol_comparison_attempts/`.
