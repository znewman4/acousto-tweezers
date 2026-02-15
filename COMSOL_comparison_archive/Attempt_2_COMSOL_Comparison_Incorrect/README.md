    # COMSOL Comparison Results — Attempt 2 (Authoritative)

    Generated: 2026-02-15T13:32:50.929241

    ## Physics

    Helmholtz equation in 3D frequency domain:
      ∇²p + k²p = 0

    Time convention: e^{-iωt}

    ## Boundary Conditions

    | Surface | Type | Detail |
    |---------|------|--------|
    | Top (z=H) | Robin (impedance) | Z_top = 0.001 × ρc |
    | Side walls (x±, y±) | Neumann (velocity source) | antiphase, both axes |
    | Bottom disc (r ≤ R_disc) | Robin (impedance) + Neumann (velocity) | Z_w = ρc, v_n = V₀ A(r) e^{iℓθ} |
    | Bottom rigid (r > R_disc) | Rigid | ∂p/∂n = 0 (natural Neumann) |

    Disc BC (COMSOL "Impedance + Normal Velocity"):
      ∂p/∂n = (iωρ/Z_w)p − iωρ v_n

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
    | Z_w = ρc | 1479548 Pa·s/m |
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

    | Case | max|p| (3D) | max|p| (z=H/2 plane) |
|------|------------|----------------------|
| Case_A_standing | 89.59 Pa | 66.45 Pa |
| Case_B_vortex | 10.88 Pa | 6.46 Pa |
| Case_C_combined | 87.73 Pa | 65.14 Pa |
| Case_C_V0x2 | 85.91 Pa | 63.84 Pa |
| Case_C_V0x3 | 84.11 Pa | 62.56 Pa |
| Case_C_V0x6 | 78.91 Pa | 58.88 Pa |
| Case_C_V0x10 | 109.92 Pa | 67.15 Pa |
| Case_C_V0x20 | 218.64 Pa | 131.63 Pa |

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
    micromamba run -n acousto-complex python scripts/analysis/rebuild_comsol_comparison.py
    ```
