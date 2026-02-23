# VORTEX_ONLY_AUDIT.md

Generated: 2026-02-23T14:59:39.182769
Run directory: `vortex_only_hires_20260223_145831`
Git branch: LinuxTest

---

## A) Geometry Summary

| Parameter | Value |
|-----------|-------|
| Lx | 6.0 mm |
| Ly | 6.0 mm |
| H_under (bath depth) | 3.0 mm |
| H_top (petri slab) | 2.0 mm |
| H_total | 5.0 mm |
| Disk radius | 1.00 mm |
| Disk centre | (3.0, 3.0) mm |
| Wavelength (λ) | 0.7420 mm |
| Mesh resolution | 5 elements/λ |
| Mesh grid | 40 × 40 × 33 |
| DOF count | 439,587 |
| PML thickness (xy) | 0.7420 mm (1.0λ) |
| PML thickness (z) | 0.7420 mm (1.0λ) |

---

## B) Boundary Conditions

### B.1 — Bottom Disk (TAG_BOTTOM_DISK = 1)

- **Type:** Neumann (velocity source)
- **Mathematical form:** ∂p/∂n = −iωρ V_disk · Φ(x,y)
  - Where Φ(x,y) is the plastic-lens vortex phase profile
  - V_disk = 1.0 μm/s
- **Physical interpretation:** Ultrasonic transducer disk driving an upward-propagating vortex beam through a plastic spiral-phase lens. The phase profile Φ encodes topological charge l=1, focal length f=10.0 mm, and offset (0.20, 0.00) mm.

### B.2 — Bottom outside disk (TAG_BOTTOM_OUTSIDE = 7)

- **Type:** Natural Neumann (zero flux)
- **Mathematical form:** ∂p/∂n = 0
- **Physical interpretation:** Rigid backing / no acoustic source outside the disk aperture. PML z-layer absorbs below this surface (outside disk column).

### B.3 — Petri sidewalls (TAG_STAND_X0=13, TAG_STAND_XL=14, TAG_STAND_Y0=15, TAG_STAND_YL=16)

- **Type:** Neumann (velocity source) — but **amplitude = 0**
- **Mathematical form:** ∂p/∂n = −iωρ · V_stand · pattern(x)  with V_stand = 0.0e+00
- **Physical interpretation:** Standing-wave transducers on petri slab walls, z ∈ [H_under, H_under + H_top]. **DISABLED** for this run (V_stand = 0). These walls effectively become rigid (zero normal velocity) because the Neumann load is zero.

### B.4 — Bath sidewalls (TAG_X0=3, TAG_XL=4, TAG_Y0=5, TAG_YL=6)

- **Type:** Natural Neumann (zero flux if not in PML) / PML absorption
- **Mathematical form:** ∂p/∂n = 0 (physical region); PML complex-coordinate stretching absorbs outgoing waves
- **Physical interpretation:** Far-field boundary. Lateral PML layers of thickness 0.742 mm absorb outgoing waves to prevent artificial reflections.

### B.5 — Top boundary (TAG_TOP = 2)

- **Type:** Robin (impedance)
- **Mathematical form:** ∂p/∂n + α·p = 0  where α = −iωρ/Z_air
  - Z_air = ρ_air · c_air = 1.2 × 343.0 = 411.6 Pa·s/m
  - α = −i·1.26e+07·997.0/(cfg.Z_air) = complex coefficient
  - Z_rel = Z_air/Z_water = 0.000278
- **Physical interpretation:** Water–air interface. Nearly total reflection (Z_air ≪ Z_water), approximating a pressure-release boundary with small transmission loss.

### B.6 — PML boundaries (all exterior faces of PML region)

- **Type:** Complex-coordinate stretching (absorbed into bilinear form)
- **Mathematical form:** Coordinate mapping x → x + (i/ω)∫σ(x')dx' with σ_max = 6.28e+07 (factor 5.0), polynomial degree 2
- **Physical interpretation:** Perfectly Matched Layer absorbs outgoing waves. Applied on 1.0λ lateral bands and 1.0λ below the disk (outside disk column). PML regions: X, Y, Z, XY, XZ, YZ, XYZ corners.

---

## C) Input Forcing Summary

| Parameter | Value |
|-----------|-------|
| Vortex topological charge (l) | 1 |
| Lens topological charge (lens_l) | 1 |
| Disk velocity amplitude | 1.0 μm/s |
| Lens drive model | plastic |
| Lens focal length | 10.0 mm |
| Lens offset (x, y) | (0.20, 0.00) mm |
| Lens c_lens (plastic) | 2700 m/s |
| Lens apodization | cosine_taper (strength 1.0) |
| Standing velocity amplitude | 0.0 μm/s **(OFF)** |
| Standing phase pattern | antiphase |
| Standing axis | both |
| Frequency | 2.00 MHz |
| Water density | 997.0 kg/m³ |
| Water sound speed | 1484.0 m/s |

---

## D) Solver Configuration

| Parameter | Value |
|-----------|-------|
| PETSc scalar type | complex128 |
| Linear solver (KSP) | preonly |
| Preconditioner (PC) | lu |
| Direct solver | mumps |
| MUMPS ICNTL(14) | 100 (% workspace increase) |
| MUMPS ICNTL(23) | 0 (max memory MB, 0=auto) |
| MUMPS ICNTL(28) | 2 (parallel analysis) |
| MUMPS ICNTL(29) | 2 (ParMETIS ordering) |
| FE order | P2 Lagrange |
| DOFs | 439,587 |
| KSP convergence reason | 4 (UNKNOWN) |
| KSP iterations | 1 |
| KSP residual norm | 0.00e+00 |
| Wall time (total) | 61.3 s |
| Mesh time | 0.4 s |
| max|p| | 1.7729 Pa |
| Memory (RSS peak) | 15279 MB |

---

## E) Confirmation Checks

| Check | Result | Detail |
|-------|--------|--------|
| Standing-wave BC disabled | ✅ PASS | V_stand = 0.0 m/s |
| No petri wall excitation | ✅ PASS | Petri Neumann load = 0 when V_stand = 0 |
| Disk is active drive | ✅ PASS | V_disk = 1.0 μm/s |
| PML enabled | ✅ PASS | pml_enabled = True |
| PML thickness (xy) | ✅ | 0.7420 mm = 1.0λ |
| PML thickness (z) | ✅ | 0.7420 mm = 1.0λ |
| Top Robin BC | ✅ | Z_air = 411.6 Pa·s/m, Z_rel = 0.000278 |
| Solver converged | ✅ PASS | reason = 4 |
| No divergence | ✅ PASS | max|p| = 1.7729 Pa |
| Vortex phase winding | ✅ PASS | winding number = 1.00 |
| Central null | ✅ PASS | |p|_center / |p|_ring = 0.1601 |
| Asymmetry matches offset | ✅ PASS | Δ|p| ratio = 0.0534 |

---

## F) Energy Partition

| Region | Σ|p|² |
|--------|-------|
| Physical | 1.5912e+04 |
| PML | 2.0309e+03 |
| PML/Physical ratio | 0.1276 |

---

## G) Output Files

### Figures (in `figures/`)

| File | Description |
|------|-------------|
| vortex_only_xy_trap_linear.png | XY |p| at trap plane |
| vortex_only_xy_trap_log.png | XY log₁₀|p| at trap plane |
| vortex_only_xy_trap_phase.png | XY arg(p) at trap plane |
| vortex_only_xy_z1mm_*.png | XY slices at z=1mm |
| vortex_only_xy_z2mm_*.png | XY slices at z=2mm |
| vortex_only_xy_z3mm_*.png | XY slices at z=3mm |
| vortex_only_xy_z4mm_*.png | XY slices at z=4mm |
| vortex_only_xz_linear.png | XZ |p| midplane |
| vortex_only_xz_log.png | XZ log₁₀|p| midplane |
| vortex_only_xz_phase.png | XZ arg(p) midplane |

### CSVs (in `csv/`)

| File | Columns |
|------|---------|
| vortex_centerline_z.csv | z_m, abs_p, phase_rad, real_p, imag_p |
| vortex_radial_profile.csv | r_m, abs_p, phase_rad, real_p, imag_p |

### Field exports

XDMF export skipped (P2 → XDMF requires interpolation to P1; use `.npz` slices or ParaView VTK instead).
