⚠️ **Historical Document** — PyVista visualization approach documented here has been **superseded by ParaView-native visualization** as of v3.0.0. The current recommended workflow uses VTU export and ParaView (see [README.md](../README.md) § "Current Working Workflow"). PyVista scripts in `scripts/visualization/` are unmaintained.

---

# PyVista 3D Visualization Guide

**Intuitive 3D renders for acousto-tweezers using PyVista**

Fast, transparent, phase-aware visualizations with batch rendering support.

---

## Quick Start

### Install Dependencies

```bash
conda install pyvista matplotlib ffmpeg
# or
pip install pyvista matplotlib
```

### Single Run Rendering

```bash
python scripts/visualization/render_pyvista_batch.py \
    --run_dir results/comparison_A_20260207_124751/
```

**Output** (in `results/.../renders/`):
- `combined_iso_phase.png` - Main 3D view (transparent iso-surfaces, phase-colored)
- `standing_iso_phase.png` - Standing wave only
- `vortex_iso_phase.png` - Vortex only (if available)
- `diff_mag_slice.png` - Difference visualization (combined - standing)
- `combined_2d_panel.png` - 2D slice panels

### Phase Sweep Rendering

```bash
python scripts/visualization/render_pyvista_batch.py \
    --phase_sweep_dir results/phase_sweep_*/ \
    --make_animation
```

**Output** (in `results/.../batch_renders/`):
- `phi_000_0deg_3d.png`, `phi_045_45deg_3d.png`, ... (one per phase)
- `phi_000_0deg_2d.png`, `phi_045_45deg_2d.png`, ... (2D panels)
- `phase_sweep_*_phase_sweep.mp4` - Animation (requires ffmpeg)
- `phase_sweep_*_phase_sweep.gif` - GIF animation

---

## Features

### Intuitive 3D Visualization

The canonical view includes:

1. **ROI Cropping**
   - Clips to configurable box around vortex aperture
   - Default: 8mm cube, auto-centered
   - Focuses on local vortex perturbation

2. **Multi-Layer Iso-Surfaces**
   - 12-20 transparent surfaces
   - Contour levels: 25%-85% of magnitude range
   - Opacity: 0.05-0.12 per surface
   - **No opaque cubes** - structure is transparent

3. **Phase-Aware Coloring**
   - HSV cyclic colormap for phase field
   - Range: [-π, π]
   - Reveals helical/spiral vortex geometry
   - Fallback: Plasma colormap for magnitude

4. **Context Slice**
   - XY plane through ROI center
   - Low opacity (0.2)
   - Colored by magnitude
   - Anchors spatial understanding

5. **Consistent Camera**
   - Positioned at (center + 1.5×ROI, -1.5×ROI, 1.2×ROI)
   - Focal point: ROI center
   - View up: +Z
   - White background

### Robust Data Loading

Supports multiple input formats:

- **XDMF + H5** (preferred) - Direct PyVista loading
- **Fallback VTU conversion** - Via dolfinx if XDMF fails

**Field detection:**
- Magnitude: `pressure_magnitude`, `p_abs`, `p_mag`, `|p|`
- Phase: `pressure_phase`, `arg_p`, `phase`, `phi`
- Gor'kov: `gorkov_potential`, `U`, `gorkov`
- Complex: Computes from `p_real`/`p_imag` if needed

### Batch Rendering

**Single run:**
- All available datasets (combined, standing, vortex)
- Difference visualization if both combined & standing exist
- 2D panel summaries

**Phase sweep:**
- Iterates all `phi_*` subdirectories
- Consistent camera and ROI across phases
- Optional MP4/GIF animation (ffmpeg required)

---

## Usage Examples

### Basic Rendering

```bash
# Render single run with defaults
python scripts/visualization/render_pyvista_batch.py \
    --run_dir results/comparison_A_20260207_124751/
```

### Custom ROI

```bash
# Specify ROI center and size
python scripts/visualization/render_pyvista_batch.py \
    --run_dir results/comparison_A_20260207_124751/ \
    --roi_center 0.01 0.01 0.012 \
    --roi_size 0.010
```

- `--roi_center x y z` - Center in meters
- `--roi_size L` - Cube size in meters (default: 0.008 = 8mm)

### High-Resolution Rendering

```bash
# 4K resolution
python scripts/visualization/render_pyvista_batch.py \
    --run_dir results/comparison_A_20260207_124751/ \
    --resolution 3840 2160
```

### Phase Sweep with Animation

```bash
# Render all phases and create MP4
python scripts/visualization/render_pyvista_batch.py \
    --phase_sweep_dir results/phase_sweep_20260207_123456/ \
    --make_animation \
    --roi_size 0.008
```

---

## Python API

### Direct Use in Scripts

```python
from acousto.viz import batch_render_run, batch_render_phase_sweep

# Single run
saved_files = batch_render_run(
    'results/comparison_A_20260207_124751/',
    roi_center=[0.01, 0.01, 0.01],
    roi_size=0.008,
    resolution=(1920, 1080)
)

# Phase sweep
frames = batch_render_phase_sweep(
    'results/phase_sweep_20260207_123456/',
    roi_center=[0.01, 0.01, 0.01],
    roi_size=0.008,
    make_animation=True
)
```

### Custom Views

```python
from acousto.viz import load_dataset, detect_fields, create_intuitive_view

# Load dataset
mesh = load_dataset('results/comparison_A_20260207_124751/combined.xdmf')

# Detect fields
mag_field, phase_field = detect_fields(mesh)

# Create view
plotter = create_intuitive_view(
    mesh,
    mag_field,
    phase_field,
    roi_center=[0.01, 0.01, 0.01],
    roi_size=0.008,
    n_contours=20,
    opacity=0.06
)

# Save screenshot
plotter.screenshot('custom_view.png')
plotter.close()
```

---

## Module Structure

```
src/acousto/viz/
├── __init__.py       - Package exports
├── loaders.py        - Data loading and field detection
├── views.py          - View builders and rendering functions
└── batch.py          - Batch rendering utilities
```

### Key Functions

**loaders.py:**
- `find_dataset(run_dir, prefer='combined')` - Locate XDMF files
- `load_dataset(filepath, convert_if_needed=True)` - Load mesh with fallback
- `detect_fields(mesh)` - Auto-detect field names
- `compute_difference(mesh_combined, mesh_standing)` - Difference field

**views.py:**
- `create_intuitive_view(mesh, mag_field, phase_field, ...)` - Main view builder
- `render_comparison(mesh_combined, mesh_standing, output_dir, ...)` - Compare views
- `render_2d_panel(mesh, output_file, ...)` - 2D slice panels

**batch.py:**
- `batch_render_run(run_dir, ...)` - Single run batch rendering
- `batch_render_phase_sweep(sweep_dir, ...)` - Phase sweep rendering
- `create_animation(frame_files, output_dir, sweep_name)` - MP4/GIF creation

---

## Expected Outputs

### Single Run

**Input:**
```
results/comparison_A_20260207_124751/
├── combined.xdmf
├── combined.h5
├── standing_only.xdmf
├── standing_only.h5
└── ...
```

**After rendering:**
```
results/comparison_A_20260207_124751/
└── renders/
    ├── combined_iso_phase.png ← Main 3D view
    ├── combined_2d_panel.png
    ├── standing_iso_phase.png
    ├── vortex_iso_phase.png (if available)
    └── diff_mag_slice.png
```

### Phase Sweep

**Input:**
```
results/phase_sweep_20260207_123456/
├── phi_000_0deg/
│   └── 3d/combined_phase000.xdmf
├── phi_045_45deg/
│   └── 3d/combined_phase045.xdmf
└── ...
```

**After rendering:**
```
results/phase_sweep_20260207_123456/
└── batch_renders/
    ├── phi_000_0deg_3d.png ← 3D views
    ├── phi_000_0deg_2d.png ← 2D panels
    ├── phi_045_45deg_3d.png
    ├── phi_045_45deg_2d.png
    ├── ...
    ├── phase_sweep_20260207_123456_phase_sweep.mp4 ← Animation
    └── phase_sweep_20260207_123456_phase_sweep.gif
```

---

## Troubleshooting

### ImportError: No module named 'pyvista'

**Solution:**
```bash
conda install pyvista
# or
pip install pyvista
```

### XDMF Loading Fails

**Problem:** PyVista can't read XDMF directly

**Solution:** The loader automatically falls back to dolfinx conversion:
```python
mesh = load_dataset('file.xdmf', convert_if_needed=True)
```

If dolfinx conversion also fails, manually convert:
```bash
python scripts/visualization/export_vizpack.py \
    --xdmf file.xdmf \
    --output file.vtu
```

Then render the VTU:
```bash
python scripts/visualization/render_pyvista_batch.py \
    --run_dir path/to/vtu_files/
```

### No magnitude field found

**Problem:** Field names don't match expected patterns

**Check available fields:**
```python
from acousto.viz import load_dataset
mesh = load_dataset('file.xdmf')
print(mesh.point_data.keys())
```

**Solution:** Edit `loaders.py` to add your field name to detection list.

### Animation not created

**Problem:** ffmpeg not installed

**Solution:**
```bash
conda install ffmpeg
```

Frames are still saved individually even without animation.

### Rendering is slow

**Solutions:**
- Reduce `n_contours` (default: 16)
- Reduce `resolution` (default: 1920×1080)
- Increase opacity (fewer transparent layers to render)

```bash
python scripts/visualization/render_pyvista_batch.py \
    --run_dir results/... \
    --resolution 1280 720
```

---

## Performance Tips

### Caching VTU Files

If XDMF loading is slow, convert once and cache:

```python
from acousto.viz import load_dataset, export_to_vtu

mesh = load_dataset('combined.xdmf')
export_to_vtu(mesh, 'combined.vtu')
```

Then render from VTU (much faster):
```bash
python scripts/visualization/render_pyvista_batch.py \
    --run_dir path/to/vtu_files/
```

### Parallel Rendering

For phase sweeps, render phases in parallel:

```python
from multiprocessing import Pool
from acousto.viz import batch_render_run

phase_dirs = list(Path('results/phase_sweep_*/').glob('phi_*'))

with Pool(4) as pool:  # 4 parallel renders
    pool.map(batch_render_run, phase_dirs)
```

---

## Comparison: PyVista vs ParaView

| Feature | PyVista | ParaView (removed) |
|---------|---------|-------------------|
| **Installation** | `conda install pyvista` | Large GUI application |
| **Batch rendering** | Python API, scriptable | pvpython required |
| **Speed** | Fast (NumPy/VTK) | Slower (GUI overhead) |
| **Customization** | Full Python control | XML state files |
| **Automation** | Native | Requires state generation |
| **Dependencies** | Lightweight | Heavy (Qt, full VTK) |

**Result:** PyVista is faster, more flexible, and easier to integrate into automated workflows.

---

## Summary Commands

```bash
# Install
conda install pyvista matplotlib ffmpeg

# Single run
python scripts/visualization/render_pyvista_batch.py \
    --run_dir results/comparison_A_20260207_124751/

# Phase sweep with animation
python scripts/visualization/render_pyvista_batch.py \
    --phase_sweep_dir results/phase_sweep_*/ \
    --make_animation

# Custom ROI and resolution
python scripts/visualization/render_pyvista_batch.py \
    --run_dir results/comparison_A_20260207_124751/ \
    --roi_center 0.01 0.01 0.012 \
    --roi_size 0.010 \
    --resolution 3840 2160
```

---

## See Also

- [scripts/visualization/README.md](../../scripts/visualization/README.md) - Quick reference
- [Phase Sweep Study](PHASE_SWEEP_STATUS.md) - Phase variation framework
- [PyVista Documentation](https://docs.pyvista.org/) - Official PyVista docs
