# COMSOL Comparison — Export Requirements

This document describes exactly what must be exported from COMSOL so
the automated `compare_to_comsol.py` pipeline can generate quantitative
comparison metrics against the FEniCSx results.

## 1. Geometry & Frequency

All exports must use the **canonical case** defined in
`configs/cases/canonical_farfield.json`.  Key parameters:

| Parameter        | Value        |
|-----------------|-------------|
| Domain Lx × Ly  | 6 × 6 mm    |
| H_under (bath)   | 3 mm         |
| H_top (petri)    | 2 mm         |
| Frequency        | 2 MHz        |
| Disk radius      | 1 mm         |
| V_disk           | 1 µm/s       |
| V_standing       | 10 µm/s      |
| Vortex charge    | ℓ = 1        |
| Lens             | plastic, f = 10 mm |
| Top BC           | water–air Robin (Z_air = 411.6 Pa·s/m) |

---

## 2. Required Slices

Export each slice as a **CSV** file with columns: `x, y, value`
(or `x, z, value` / `y, z, value` for non-XY slices).
Coordinates must be in **metres** (SI).

### Slice A — XY mid-plane (petri slab centre)
- z = H_under + H_top/2 = 4.0 mm
- x ∈ [0, 6 mm], y ∈ [0, 6 mm]
- Grid: 300 × 300 (or finer)
- Export: `|p|` (Pa) and `arg(p)` (radians)
- Filename: `xy_midplane_pmag.csv`, `xy_midplane_phase.csv`

### Slice B — XZ centre-plane
- y = Ly/2 = 3 mm
- x ∈ [0, 6 mm], z ∈ [0, 5 mm]
- Grid: 300 × 300
- Export: `|p|` (Pa)
- Filename: `xz_center_pmag.csv`

### Slice C — YZ centre-plane
- x = Lx/2 = 3 mm
- y ∈ [0, 6 mm], z ∈ [0, 5 mm]
- Grid: 300 × 300
- Export: `|p|` (Pa)
- Filename: `yz_center_pmag.csv`

---

## 3. Required Cases

Export the slices above for each case:

1. `standing_only/` — disk velocity = 0
2. `vortex_only/` — standing velocity = 0
3. `combined/` — both active

Directory layout:

```
comsol_exports/
  standing_only/
    xy_midplane_pmag.csv
    xy_midplane_phase.csv
    xz_center_pmag.csv
    yz_center_pmag.csv
  vortex_only/
    ...
  combined/
    ...
```

---

## 4. CSV Format

Plain CSV, no header comments, first row is header:

```csv
x,y,value
0.0,0.0,12345.67
0.00002,0.0,12346.01
...
```

- Coordinates in metres (e.g., `0.003` for 3 mm)
- Values: pressure magnitude in Pa, phase in radians
- NaN for any points outside the domain

---

## 5. Units Summary

| Quantity | Unit |
|----------|------|
| Coordinates x,y,z | metres [m] |
| Pressure magnitude | Pa |
| Phase | radians |
| Frequency | Hz |

---

## 6. Running the Comparison

After exporting COMSOL data:

```bash
python scripts/comsol/compare_to_comsol.py \
    --fenics-run results/<run_id> \
    --comsol-dir /path/to/comsol_exports
```

This produces:
- Difference heatmaps (PNG)
- Relative error norms (L2, L∞) per slice
- `REPORT.md` with pass/fail metrics

---

## 7. Pass/Fail Criteria (Advisory)

| Metric | Threshold | Notes |
|--------|-----------|-------|
| Focus z-location error | < 0.5 mm | argmax(|p|) on centerline |
| Max |p| error in ROI | < 20% relative | physical region only |
| Spatial correlation (XY magnitude) | > 0.85 | Pearson r of |p| patterns |
| L2 relative error (any slice) | < 30% | ‖Δ‖₂ / ‖COMSOL‖₂ |
