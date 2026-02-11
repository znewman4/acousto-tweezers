# Visualization Scripts

PyVista-based 3D visualization and batch rendering for acousto-tweezers results.

## Quick Start

### 1. Install Dependencies

```bash
conda install pyvista matplotlib ffmpeg
```

### 2. Render Single Run

```bash
python render_pyvista_batch.py --run_dir ../../results/comparison_A_20260207_124751/
```

**Output** (in `renders/`):
- `combined_iso_phase.png` - Main 3D view (transparent iso-surfaces, phase-colored)
- `standing_iso_phase.png` - Standing wave only
- `diff_mag_slice.png` - Difference visualization
- `combined_2d_panel.png` - 2D slice panels

**What you'll see:**
- ✅ Transparent multi-layer iso-surfaces (NO opaque cube)
- ✅ Phase-colored geometry (HSV colormap)
- ✅ Cropped to 8mm cube around vortex
- ✅ Context slice at ROI center
- ✅ Consistent camera positioning

### 3. Render Phase Sweep with Animation

```bash
python render_pyvista_batch.py --phase_sweep_dir ../../results/phase_sweep_*/ --make_animation
```

**Output** (in `batch_renders/`):
- `phi_000_0deg_3d.png`, `phi_045_45deg_3d.png`, ... (one per phase)
- `phase_sweep_*_phase_sweep.mp4` - Animation

---

## Summary Commands

```bash
# Install dependencies
conda install pyvista matplotlib ffmpeg

# Single run
python render_pyvista_batch.py --run_dir ../../results/comparison_A_20260207_124751/

# Phase sweep with animation
python render_pyvista_batch.py --phase_sweep_dir ../../results/phase_sweep_*/ --make_animation

# Custom ROI and resolution
python render_pyvista_batch.py \
    --run_dir ../../results/comparison_A_20260207_124751/ \
    --roi_center 0.01 0.01 0.012 \
    --roi_size 0.010 \
    --resolution 3840 2160
```

---

## See Also

- [docs/PYVISTA_VIZ.md](../../docs/PYVISTA_VIZ.md) - Complete documentation
- [docs/PHASE_SWEEP_STATUS.md](../../docs/PHASE_SWEEP_STATUS.md) - Phase variation study
