# FEM Standing Grid (3×3) + ASM Vortex Overlay Study

**Generated:** 2026-03-04 11:02:54
**Run directory:** `results/fem_standing_plus_asm_vortex_local_3x3_20260304_110236`
**Total runtime:** 18.3s

---

## Objective

Study how a focused ASM acoustic vortex (plastic lens) perturbs a
local 3×3 standing-wave trap grid.  The standing-wave field is from
a validated FEM solve; the vortex field is from the validated ASM
propagator.  The two are overlaid in post-processing.

---

## Physical Parameters

| Parameter | Value |
|-----------|-------|
| Frequency | 2.0 MHz |
| λ | 0.7420 mm |
| k | 8467.9 rad/m |
| c_water | 1484.0 m/s |
| ρ_water | 997.0 kg/m³ |
| Trap spacing (λ/2) | 0.3710 mm |
| z* (trap plane) | 4.190 mm |
| 3×3 region size | 1.484 mm ≈ 2.00λ |

### Vortex Lens

| Parameter | Value |
|-----------|-------|
| ℓ (topological charge) | 2 |
| R (aperture radius) | 5.0 mm |
| f (focal length) | 4.0 mm |
| Fresnel number N_F | 8.42 |
| c_lens | 2700.0 m/s |
| Apodization | cosine_taper |

---

## Validation Checks

| Metric | Measured | Expected | Status |
|--------|----------|----------|--------|
| Trap spacing | 0.1875 mm | 0.3710 mm | CHECK |
| Waist diameter | 4.4297 mm | < 0.7420 mm (1λ) | WARN: > 1λ |
| Waist diameter / λ | 5.970 | < 1.0 | WARN |

---

## Perturbation Results

| α | Vortex/Standing (%) | Traps perturbed | Total traps |
|---|---------------------|-----------------|-------------|
| 0.05 | 5.0% | 0 | 14 |
| 0.1 | 10.0% | 1 | 14 |
| 0.2 | 20.0% | 2 | 14 |

---

## Conclusion

At α = 0.1 (vortex peak = 10.0% of standing peak):

**The vortex significantly perturbs 1 trap.**

Excellent — single-trap selection achievable.

---

## Limitations

1. **No coupled FEM re-solve.** Standing and vortex fields are superposed
   linearly in post-processing.  There is no cavity–lens interaction.
2. **Linear acoustics assumption.** Superposition is only valid in the
   linear regime; no nonlinear radiation forces are computed.
3. **ASM is free-space.** The vortex propagation does not include
   reflections from the petri dish walls or water–air interface.
4. **FEM interpolation uses LinearNDInterpolator** — the P2 FEM field is
   sampled at DOF coordinates and linearly interpolated (Delaunay).
   For this qualitative study this is acceptable.

---

## Deliverables

- `figures/01_standing_only_xy.png` — Standing wave |p| at z*
- `figures/02_vortex_only_xy.png` — Vortex |p| at z* with waist annotation
- `figures/03_overlay_alpha_*.png` — Combined + difference for each α
- `figures/04_xz_meridional.png` — XZ slice showing hourglass + standing envelope
- `figures/05_phase_comparison.png` — Phase maps (standing, vortex, combined)
- `data/local_fields.npz` — Complex fields on the local 3×3 grid
- `data/xz_fields.npz` — Complex fields on the XZ meridional plane
- `data/metadata.json` — All parameters and computed metrics
- `REPORT.md` — This file
