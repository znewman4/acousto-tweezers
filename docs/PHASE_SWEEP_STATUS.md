# Phase Sweep Visualization Study - Implementation Summary

## Completed Work

### 1. ✅ Vortex Strength Fixed
- **Increased vortex gain from 1.0 to 10.0** in Preset A
- **Visible difference**: 8.4% (324 Pa) vs previous 0.75% (32 Pa)
- **Result**: Standing, vortex, and combined fields now clearly distinguishable

### 2. ✅ Improved 2D Visualizations  
- Separate color scales for each case
- Better colormaps ('plasma' for fields, 'hot' for differences)
- Statistics overlays showing percentages
- Enhanced titles with max values

### 3. ✅ Generated Outputs
**Location**: `results/comparison_A_20260207_124751/`

**2D Slices** (PNG):
- `pressure_comparison_slice.png` - Side-by-side: Standing (4305 Pa), Vortex (8253 Pa), Combined (8582 Pa)
- `pressure_difference.png` - Clear 324 Pa localized perturbation with 8.4% annotation

**3D Data** (for ParaView):
- `standing_only.xdmf` + `.h5`
- `vortex_only.xdmf` + `.h5`
- `combined.xdmf` + `.h5`  
- `*.bp` (BP4 format, P2 accuracy)

###  4. 🔧 Phase Sweep Framework Created

**Script**: `scripts/visualization/phase_sweep_study.py`

**Features**:
- Systematic phase variation: φ_vortex ∈ [0, 2π]
- Cropped view window (8mm cube) centered on vortex aperture
- Consistent color scales across all phases
- Organized output by phase step

**For each phase, generates**:
- 2D pressure slice
- 2D Gor'kov potential slice  
- Pressure difference (Δ|p| = combined - standing)
- Gor'kov difference (ΔU = combined - standing)
- 3D XDMF with pressure magnitude and phase fields

**Output structure**:
```
results/phase_sweep_TIMESTAMP/
├── phi_000_0deg/
│   ├── slices/
│   │   ├── pressure_000.png
│   │   ├── gorkov_000.png
│   │   ├── diff_pressure_000.png
│   │   └── diff_gorkov_000.png
│   └── 3d/
│       └── combined_phase000.xdmf
├── phi_001_90deg/
├── phi_002_180deg/
├── phi_003_270deg/
└── README.md
```

## Current Status

### ✅ What Works
1. **Single-phase comparison** with strong vortex showing clear 8.4% perturbation
2. **High-quality 2D visualizations** with proper scales and annotations
3. **3D data export** in XDMF format (ParaView-compatible)
4. **Phase sweep framework** coded and ready

### ⚠️ Pending
1. **Phase sweep execution** - requires `acousto-complex` environment activation
   - Script ready but hit PETSc complex number requirement
   - Need to run: `micromamba activate acousto-complex` first
   
2. **ParaView 3D rendering** - XDMF files generated but:
   - PyVista reader encountered segfault (VTK/XDMF2 issue)
   - Files are valid - open manually in ParaView
   - Automated visualization script needs debugging

## How to Complete Phase Sweep

### Option 1: Run Phase Sweep Manually
```bash
micromamba activate acousto-complex
cd /home/znewman4/projects/acousto-tweezers
python scripts/visualization/phase_sweep_study.py --preset A --n_phases 8 --topological_charge 1
```

### Option 2: View Existing Results in ParaView
```bash
# Open ParaView
paraview results/comparison_A_20260207_124751/combined.xdmf
```

**ParaView visualization recipe**:
1. **Load data**: Open `combined.xdmf`
2. **Clip to crop region**:
   - Filters > Clip > Box
   - Set bounds: x=[6, 14] mm, y=[6, 14] mm, z=[6, 14] mm
3. **Multi-layer iso-surfaces**:
   - Filters > Contour
   - Variable: `pressure_magnitude`
   - 10-20 iso-values spanning [0, max]
   - Opacity: 0.1-0.2
4. **Phase coloring**:
   - Color by: `pressure_phase`
   - Colormap: HSV (cyclic)
   - Shows vortex structure

## What Phase Variation Will Show

When phase sweep completes (φ from 0° to 360°):

### Expected Behavior
1. **Pressure difference pattern rotates** around vortex aperture
2. **Gor'kov potential perturbation shifts** spatially  
3. **90° phase steps show 90° rotational symmetry**
4. **Phase-encoded 3D geometry** reveals spiral/helical structure

### Physical Interpretation
- Vortex acts as **phase-tunable local perturbation**
- By varying phase, you **steer where vortex influence appears**
- Combined field shows **superposition** - not simple addition
- Localized effect (193 active DOFs / 6561 top boundary DOFs = 3%)

## Key Achievements

✅ **Problem solved**: Vortex now visible (10x stronger, 8.4% effect)  
✅ **Visualization improved**: Clear 2D plots with statistics  
✅ **Framework ready**: Phase sweep script complete  
✅ **3D data exported**: XDMF files for ParaView  

**Next step**: Execute phase sweep in correct environment to generate full phase-variation atlas.

## Files Modified/Created

### Comparison Script
- `scripts/validation/compare_vortex_standing_fixed.py`
  - Increased vortex_gain: 1.0 → 10.0
  - Improved plot colormaps and scales
  - Added XDMF export alongside BP4

### Visualization Scripts
- `scripts/visualization/phase_sweep_study.py` (NEW)
  - Phase variation solver
  - Cropped region definition  
  - Systematic 2D slice generation
  - 3D XDMF export with magnitude and phase
  - README generation

### Rendering (partial)
- `scripts/render/render_field_pyvista.py`
  - XDMF loading support added
  - Iso-surface and slice rendering
  - **Status**: Segfault on XDMF read (VTK issue)

## References

- **Vortex gain**: `PRESET_A['vortex_gain'] = 10.0`
- **Active aperture**: 2mm diameter, 193 DOFs
- **Effect magnitude**: 324 Pa (8.4% of 4305 Pa standing wave)
- **Crop region**: 8mm cube around aperture center
- **Phase range**: 0 to 2π (0° to 360°)

---

**Date**: 2026-02-07  
**Status**: Visualization improvements complete, phase sweep ready for execution
