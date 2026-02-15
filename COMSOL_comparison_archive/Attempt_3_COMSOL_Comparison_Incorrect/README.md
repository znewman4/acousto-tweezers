    # COMSOL Comparison Results — Attempt 3

    Generated: 2026-02-15T14:00:28.908611

    ## What Changed from Attempt 2 and Why

    **Bug fixed:** In Attempt 2, Case A (standing-only) used `disc_robin=True`,
    meaning the disc region on the bottom boundary had an impedance Robin BC
    ($Z = \rho c$). This caused the disc to act as an **absorbing patch**,
    creating a large dead zone (low |p|) inside the disc circle that does not
    appear in the COMSOL standing-only reference.

    **Root cause:** The impedance Robin term $\partial p/\partial n = (i\omega\rho/Z)p$
    with $Z = \rho c$ (i.e., $\alpha = -ik$) absorbs energy at the disc boundary.
    In standing-only mode (no vortex forcing), the disc has no source term to
    compensate. The debug analysis showed:
    - Disc absorbed **55× more power** than the top boundary
    - |p| at disc centre was **13× lower** with disc impedance ON vs OFF
    - Making $Z_{disc} \to \infty$ (rigid) recovered the correct pattern

    **Fix:** Case A now uses `disc_robin=False` (rigid bottom everywhere).
    Cases B and C keep `disc_robin=True` because the disc impedance is part of
    the physical transducer model when vortex forcing is active.

    **COMSOL interpretation:** In COMSOL standing-only, the disc impedance BC is
    not enabled. The "Impedance + Include normal velocity" boundary only matters
    when the normal velocity source is active (vortex/combined modes).

    ## Physics

    Helmholtz equation in 3D frequency domain:
      ∇²p + k²p = 0

    Time convention: e^{-iωt}

    ## Boundary Conditions

    | Surface | Case A (standing) | Cases B, C (vortex/combined) |
    |---------|-------------------|------------------------------|
    | Top (z=H) | Robin: Z_top = 0.001 × ρc | Robin: Z_top = 0.001 × ρc |
    | Side walls (x±, y±) | Neumann source (antiphase) | Neumann source (active in combined) |
    | Bottom disc (r ≤ R) | **Rigid** (∂p/∂n = 0) | Robin: Z_w = ρc + Neumann: v_n |
    | Bottom rigid (r > R) | Rigid | Rigid |

    Disc BC (Cases B, C — COMSOL "Impedance + Normal Velocity"):
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

    | Case | max|p| (3D) | max|p| (z=H/2) | |p| disc centre |
|------|------------|----------------|-----------------|
| Case_A_standing | 81.68 Pa | 57.76 Pa | 0.0079 Pa |
| Case_B_vortex | 10.88 Pa | 6.46 Pa | 0.2035 Pa |
| Case_C_combined | 87.73 Pa | 65.14 Pa | 0.2038 Pa |
| Case_C_V0x1 | 87.73 Pa | 65.14 Pa | 0.2038 Pa |
| Case_C_V0x2 | 85.91 Pa | 63.84 Pa | 0.4073 Pa |
| Case_C_V0x3 | 84.11 Pa | 62.56 Pa | 0.6107 Pa |
| Case_C_V0x6 | 78.91 Pa | 58.88 Pa | 1.2211 Pa |
| Case_C_V0x10 | 109.92 Pa | 67.15 Pa | 2.0349 Pa |
| Case_C_V0x20 | 218.64 Pa | 131.63 Pa | 4.0694 Pa |

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
    micromamba run -n acousto-complex python scripts/analysis/rebuild_comsol_comparison_attempt3.py
    ```
