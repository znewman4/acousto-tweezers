    # COMSOL Comparison Results — Attempt 4

    Generated: 2026-02-15T14:49:44.683424

    ## What Changed from Attempt 3 and Why

    **Bug fixed:** In Attempts 2–3, Cases B (vortex) and C (combined) used
    `disc_robin=True`, meaning the disc had an impedance Robin BC ($Z = \rho c$)
    plus a Neumann velocity source. The impedance term absorbed standing-wave
    energy inside the disc region, creating a "hole" in the combined pressure
    field at V₀ and V₀×2.

    **Root cause (same as Attempt 3, now extended to ALL cases):**
    The impedance Robin term $\partial p/\partial n = (i\omega\rho/Z)p$ with
    $Z = \rho c$ (i.e., $\alpha = -ik$) acts as an energy absorber. In combined
    mode (Case C), this absorbed the standing-wave component inside the disc
    circle, reducing avg|p| by 2×:

    | Config | avg|p| inside disc | % of standing |
    |--------|-------------------|---------------|
    | A standing (rigid) | 23.01 Pa | 100% |
    | C V₀ Robin (impedance+vortex) | 11.16 Pa | 48.5% |
    | C V₀ Neumann (pure source) | 25.07 Pa | **108.9%** |

    **Fix:** ALL cases now use `disc_robin=False`. The disc is a **pure Neumann
    velocity source** (∂p/∂n = −iωρ v_n) with no impedance absorption.
    This matches the physical COMSOL setup where the disc applies a normal
    velocity without impedance.

    ## Physics

    Helmholtz equation in 3D frequency domain:
      ∇²p + k²p = 0

    Time convention: e^{-iωt}

    ## Boundary Conditions

    | Surface | Case A (standing) | Cases B, C (vortex/combined) |
    |---------|-------------------|------------------------------|
    | Top (z=H) | Robin: Z_top = 0.001 × ρc | Robin: Z_top = 0.001 × ρc |
    | Side walls (x±, y±) | Neumann source (antiphase) | Neumann source (active in combined) |
    | Bottom disc (r ≤ R) | **Rigid** (∂p/∂n = 0) | **Pure Neumann**: ∂p/∂n = −iωρ v_n |
    | Bottom rigid (r > R) | Rigid | Rigid |

    **Key change:** The disc has NO impedance Robin term. It is a pure velocity
    source (Neumann) in Cases B/C and rigid in Case A.

    Side wall BC (pure Neumann source):
      ∂p/∂n = −iωρ V_s  (no impedance term)

    ## Parameters

    | Parameter | Value |
    |-----------|-------|
    | L | 10 mm |
    | H | 1 mm |
    | f | 500 kHz |
    | λ | 2.968 mm |
    | ρ | 997 kg/m³ |
    | c | 1484 m/s |
    | Z_w = ρc | 1479548 Pa·s/m (NOT used for disc BC) |
    | Z_top | 1479.5 Pa·s/m |
    | V₀ (vortex) | 10 µm/s |
    | Vs (standing) | 10 µm/s |
    | ℓ | 1 |
    | R_disc | 3 mm |
    | Apodization | cosine taper: A(r) = 0.5(1+cos(πr/R)) |

    ## Mesh

    - elements_per_wavelength: 10
    - nx × ny × nz: 33 × 33 × 8
    - Element type: P2 (quadratic Lagrange tetrahedra)
    - Satisfies: max element size ≤ λ/6

    ## Solver

    MUMPS direct LU factorization (via PETSc)

    ## Diagnostics

    ```
    Disc facets:          550
Disc area (mesh):     25.2525 mm²
Disc area (πR²):      28.2743 mm²
Area ratio:           0.8931
max(|pattern|):       1.000000
avg(|pattern|):       0.314239
|g_stand| = ωρVs:     31321.6788 Pa/m
max|g_vtx| = ωρV₀max: 31321.6788 Pa/m
Forcing ratio g_vtx/g_stand: 1.0000
Wall area (4 walls):  40.00 mm²
Wall/disc area ratio: 1.58
    ```

    ## Results

    | Case | max|p| (3D) | max|p| (z=H/2) | |p| disc centre |
|------|------------|----------------|-----------------|
| Case_A_standing | 81.68 Pa | 57.76 Pa | 0.0079 Pa |
| Case_B_vortex | 35.81 Pa | 25.33 Pa | 2.6265 Pa |
| Case_C_combined | 82.78 Pa | 64.49 Pa | 2.6206 Pa |
| Case_C_V0x1 | 82.78 Pa | 64.49 Pa | 2.6206 Pa |
| Case_C_V0x2 | 114.38 Pa | 86.54 Pa | 5.2471 Pa |
| Case_C_V0x3 | 147.21 Pa | 109.54 Pa | 7.8737 Pa |
| Case_C_V0x6 | 248.14 Pa | 180.54 Pa | 15.7533 Pa |
| Case_C_V0x10 | 384.41 Pa | 276.69 Pa | 26.2594 Pa |
| Case_C_V0x20 | 726.81 Pa | 518.62 Pa | 52.5248 Pa |

    ## Structure

    ```
    COMSOL_comparison_results/
      Case_A_standing/   csv/ figs/ meta/
      Case_B_vortex/     csv/ figs/ meta/
      Case_C_combined/   csv/ figs/ meta/
      README.md
      MANIFEST.txt
    ```

    ## How to regenerate

    ```bash
    cd /home/znewman4/projects/acousto-tweezers
    micromamba run -n acousto-complex python scripts/analysis/rebuild_comsol_comparison_attempt4.py
    ```
