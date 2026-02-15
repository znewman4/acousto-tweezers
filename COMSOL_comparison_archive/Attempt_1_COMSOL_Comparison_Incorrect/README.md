# COMSOL Comparison Results

Generated: 2026-02-13T19:47:05.222200

## How to regenerate

```bash
cd /home/znewman4/projects/acousto-tweezers
micromamba run -n acousto-complex python scripts/analysis/export_comsol_parallel_figures.py
```

## Structure

```
COMSOL_comparison_results/
  Case_A_standing/   — standing wave only
  Case_B_vortex/     — vortex beam only
  Case_C_combined/   — standing + vortex
  README.md          — this file
  MANIFEST.txt       — full file list
```

Each case contains:
- `figs/`  — 4 PNG figures (slice abs, slice arg, iso abs, iso Re)
- `csv/`   — plane grid + 1D line exports
- `meta/`  — config_used.json, solver_report.txt

## Parameters (locked)
- L = 10 mm, H = 1 mm, f = 500 kHz
- rho = 997 kg/m³, c = 1484 m/s
- V_s = V_0 = 10 µm/s, ℓ = 1
- R_disc = 3 mm, Z_top = 0.001·ρc
- P2 elements, 20×20×8 structured tet mesh
