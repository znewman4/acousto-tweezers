# Corrected Physical Model — Vortex + Standing-Wave Interaction

**Date:** 2026-02-20T17:36:12.598198

## Physical Model

- **Two-region cuboid**: water bath (below) + petri slab (above)
- **H_petri** = 2 mm (fixed)
- **H_bath** = swept  {3, 4, 5, 6, 7} mm
- **Standing-wave BC**: ONLY on petri slab side walls (z ∈ [H_bath, H_total])
- **Bath side walls**: passive (no excitation)
- **Top BC**: water–air Robin (FIXED, not tunable)
  - ρ_air = 1.2 kg/m³,  c_air = 343 m/s
  - Z_air = ρ_air · c_air = 411.6 Pa·s/m
  - Z_water = 997 × 1484 = 1,479,548 Pa·s/m
  - Z_rel = Z_air / Z_water = 0.000278
  - Robin: ∂p/∂n + ik(ρ_water c_water)/(ρ_air c_air) p = 0
- **Frequency**: 2.0 MHz
- **Domain**: 6 × 6 mm lateral
- **Resolution**: 4 elem/λ
- **Solver**: MUMPS direct

## H_bath × f_lens Sweep (Vortex-Only)

| H_bath [mm] | f [mm] | H_total [mm] | z_focus [mm] | below_petri [mm] | max|p| bath | max|p| petri | status |
|------------|--------|-------------|-------------|-----------------|-------------|-------------|--------|
| 3 | 2 | 5 | 1.30 | 1.70 | 0.2359 | 0.1630 | OK |
| 3 | 3 | 5 | nan | nan | nan | nan | SKIPPED (f>=H_bath) |
| 3 | 4 | 5 | nan | nan | nan | nan | SKIPPED (f>=H_bath) |
| 3 | 5 | 5 | nan | nan | nan | nan | SKIPPED (f>=H_bath) |
| 3 | 6 | 5 | nan | nan | nan | nan | SKIPPED (f>=H_bath) |
| 4 | 2 | 6 | 1.18 | 2.82 | 0.2533 | 0.1219 | OK |
| 4 | 3 | 6 | 1.18 | 2.82 | 0.1737 | 0.0831 | OK |
| 4 | 4 | 6 | nan | nan | nan | nan | SKIPPED (f>=H_bath) |
| 4 | 5 | 6 | nan | nan | nan | nan | SKIPPED (f>=H_bath) |
| 4 | 6 | 6 | nan | nan | nan | nan | SKIPPED (f>=H_bath) |
| 5 | 2 | 7 | 1.47 | 3.53 | 0.3842 | 0.1719 | OK |
| 5 | 3 | 7 | 1.47 | 3.53 | 0.2692 | 0.1180 | OK |
| 5 | 4 | 7 | 1.47 | 3.53 | 0.2057 | 0.0890 | OK |
| 5 | 5 | 7 | nan | nan | nan | nan | SKIPPED (f>=H_bath) |
| 5 | 6 | 7 | nan | nan | nan | nan | SKIPPED (f>=H_bath) |
| 6 | 2 | 8 | 1.36 | 4.64 | 0.2407 | 0.1020 | OK |
| 6 | 3 | 8 | 1.36 | 4.64 | 0.1723 | 0.0712 | OK |
| 6 | 4 | 8 | 1.36 | 4.64 | 0.1329 | 0.0546 | OK |
| 6 | 5 | 8 | 1.36 | 4.64 | 0.1078 | 0.0445 | OK |
| 6 | 6 | 8 | nan | nan | nan | nan | SKIPPED (f>=H_bath) |
| 7 | 2 | 9 | 1.17 | 5.83 | 0.2411 | 0.0788 | OK |
| 7 | 3 | 9 | 1.17 | 5.83 | 0.1663 | 0.0536 | OK |
| 7 | 4 | 9 | 1.17 | 5.83 | 0.1261 | 0.0403 | OK |
| 7 | 5 | 9 | 1.28 | 5.72 | 0.1014 | 0.0322 | OK |
| 7 | 6 | 9 | 1.28 | 5.72 | 0.0848 | 0.0269 | OK |

## Selected Geometry

- H_bath = 3 mm
- f_lens = 2 mm
- z_focus = 1.30 mm (1.70 mm below petri)

## Interaction Check

- V_disk = 1.0 µm/s
- V_stand = 10.0 µm/s

See `csv/roi_metrics.csv` and `figures/` for full results.

## Figures

- ![centerline_z](figures/centerline_z.png)
- ![xy_combined](figures/xy_combined.png)
- ![xy_delta](figures/xy_delta.png)
- ![xy_standing_only](figures/xy_standing_only.png)
- ![xy_vortex_only](figures/xy_vortex_only.png)
- ![xz_combined_linear](figures/xz_combined_linear.png)
- ![xz_combined_log](figures/xz_combined_log.png)
- ![xz_standing_only_linear](figures/xz_standing_only_linear.png)
- ![xz_standing_only_log](figures/xz_standing_only_log.png)
- ![xz_vortex_linear_Hb3_f2](figures/xz_vortex_linear_Hb3_f2.png)
- ![xz_vortex_linear_Hb4_f2](figures/xz_vortex_linear_Hb4_f2.png)
- ![xz_vortex_linear_Hb4_f3](figures/xz_vortex_linear_Hb4_f3.png)
- ![xz_vortex_linear_Hb5_f2](figures/xz_vortex_linear_Hb5_f2.png)
- ![xz_vortex_linear_Hb5_f3](figures/xz_vortex_linear_Hb5_f3.png)
- ![xz_vortex_linear_Hb5_f4](figures/xz_vortex_linear_Hb5_f4.png)
- ![xz_vortex_linear_Hb6_f2](figures/xz_vortex_linear_Hb6_f2.png)
- ![xz_vortex_linear_Hb6_f3](figures/xz_vortex_linear_Hb6_f3.png)
- ![xz_vortex_linear_Hb6_f4](figures/xz_vortex_linear_Hb6_f4.png)
- ![xz_vortex_linear_Hb6_f5](figures/xz_vortex_linear_Hb6_f5.png)
- ![xz_vortex_linear_Hb7_f2](figures/xz_vortex_linear_Hb7_f2.png)
- ![xz_vortex_linear_Hb7_f3](figures/xz_vortex_linear_Hb7_f3.png)
- ![xz_vortex_linear_Hb7_f4](figures/xz_vortex_linear_Hb7_f4.png)
- ![xz_vortex_linear_Hb7_f5](figures/xz_vortex_linear_Hb7_f5.png)
- ![xz_vortex_linear_Hb7_f6](figures/xz_vortex_linear_Hb7_f6.png)
- ![xz_vortex_log_Hb3_f2](figures/xz_vortex_log_Hb3_f2.png)
- ![xz_vortex_log_Hb4_f2](figures/xz_vortex_log_Hb4_f2.png)
- ![xz_vortex_log_Hb4_f3](figures/xz_vortex_log_Hb4_f3.png)
- ![xz_vortex_log_Hb5_f2](figures/xz_vortex_log_Hb5_f2.png)
- ![xz_vortex_log_Hb5_f3](figures/xz_vortex_log_Hb5_f3.png)
- ![xz_vortex_log_Hb5_f4](figures/xz_vortex_log_Hb5_f4.png)
- ![xz_vortex_log_Hb6_f2](figures/xz_vortex_log_Hb6_f2.png)
- ![xz_vortex_log_Hb6_f3](figures/xz_vortex_log_Hb6_f3.png)
- ![xz_vortex_log_Hb6_f4](figures/xz_vortex_log_Hb6_f4.png)
- ![xz_vortex_log_Hb6_f5](figures/xz_vortex_log_Hb6_f5.png)
- ![xz_vortex_log_Hb7_f2](figures/xz_vortex_log_Hb7_f2.png)
- ![xz_vortex_log_Hb7_f3](figures/xz_vortex_log_Hb7_f3.png)
- ![xz_vortex_log_Hb7_f4](figures/xz_vortex_log_Hb7_f4.png)
- ![xz_vortex_log_Hb7_f5](figures/xz_vortex_log_Hb7_f5.png)
- ![xz_vortex_log_Hb7_f6](figures/xz_vortex_log_Hb7_f6.png)
- ![xz_vortex_only_linear](figures/xz_vortex_only_linear.png)
- ![xz_vortex_only_log](figures/xz_vortex_only_log.png)
