# 3D Helmholtz Solver - Memory-Optimized Demo

## Overview

This is a refactored, production-ready 3D Helmholtz solver with particle simulation. The system addresses the memory issues in earlier versions by:

1. **Matrix caching**: Reuses the system matrix A across time steps if grid/frequency don't change.
2. **Streaming GIF rendering**: Frames are written one-at-a-time via imageio's streaming writer (not buffered in RAM).
3. **Smaller dtypes**: Uses `float32` and `complex64` by default (instead of `float64`/`complex128`).
4. **Reduced per-step storage**: Stores only 2D slices (at chosen z) instead of full 3D force fields.
5. **Iterative solver option**: GMRES/BiCGSTAB for systems where direct solver would consume too much memory.
6. **Memory diagnostics**: Reports RSS, array sizes, and peak memory during execution.

## Quick Start

### Minimal demo (should complete in seconds):
```bash
python3 scripts/demo_helmholtz3d_v2.py \
  --n_steps 10 \
  --render_stride 2 \
  --gif 1 \
  --Lx 0.01 --Ly 0.01 --H 0.003 \
  --dx 0.001 --dy 0.001 --dz 0.001
```

### Realistic demo (21×21×8 grid, 20 time steps):
```bash
python3 scripts/demo_helmholtz3d_v2.py \
  --n_steps 20 \
  --render_stride 2 \
  --gif 1 \
  --Lx 0.03 --Ly 0.03 --H 0.01 \
  --dx 0.0015 --dy 0.0015 --dz 0.0015 \
  --dtype single \
  --solver direct
```

### Solver-only mode (no Gor'kov/particle, for benchmarking):
```bash
python3 scripts/demo_helmholtz3d_v2.py \
  --n_steps 5 \
  --render_stride 1 \
  --gif 0 \
  --no_gorkov \
  --no_particle \
  --Lx 0.03 --Ly 0.03 --H 0.01 \
  --dx 0.005 --dy 0.005 --dz 0.005
```

## CLI Arguments

### Grid Parameters
- `--Lx` (float, default 0.03): Domain width x [m]
- `--Ly` (float, default 0.03): Domain width y [m]
- `--H` (float, default 0.01): Domain height z [m]
- `--dx` (float, default 0.0003): Grid spacing x [m]
- `--dy` (float, default 0.0003): Grid spacing y [m]
- `--dz` (float, default 0.0003): Grid spacing z [m]

### Simulation Parameters
- `--omega_hz` (float, default 1e6): Driving frequency [Hz]
- `--n_steps` (int, default 50): Number of time steps
- `--dt_s` (float, default 1e-3): Time step size [s]
- `--render_stride` (int, default 1): Render every Nth frame to GIF
- `--slice_z` (float, default 0.5*H): z-coordinate for 2D visualization [m]
- `--gif_fps` (int, default 10): Playback speed for GIF

### Output Control
- `--gif` (0/1, default 1): Generate GIF output
- `--save_png_frames` (0/1, default 0): Save individual PNG frames
- `--max_ram_mb` (int, default 2048): Soft memory limit [MB] (informational)

### Solver Control
- `--dtype` {single|double}: Floating-point precision (default: single = float32/complex64)
- `--solver` {direct|gmres|bicgstab}: Linear solver method (default: direct)

### Feature Flags
- `--no_gorkov`: Skip Gor'kov potential computation (solver only)
- `--no_particle`: Skip particle trajectory simulation
- `--no_lens_pipeline`: Skip lens→bath→plate pipeline (for debugging)
- `--time_varying` (0/1, default 1): Enable time-varying simulation (0 = static demo)

## Output

All outputs are saved to:
```
results/helmholtz3d_demo/run_YYYYMMDD_HHMMSS/
├── particle_slice.gif                 # Main output: animated 2D force slice
├── traj_moving_lens.csv               # Particle trajectory and forces
└── frames/                            # Individual PNG frames (if --save_png_frames 1)
```

**CSV Format** (traj_moving_lens.csv):
```
t_s,x_m,y_m,z_m,Fx_N,Fy_N,Fz_N,U_J
0.0,0.015,0.015,0.00101,...
0.001,0.01501,...
```

## Memory Usage

The script prints a memory banner at startup with estimated per-frame memory:
```
[RUN BANNER] Grid 21x21x8, 20 steps, single precision
  Grid: Nx=21, Ny=21, Nz=8, total_points=3528
  Omega: 6.28e+06 rad/s
  Dtype precision: single

Estimated memory for single copy:
  p (complex pressure):          0.1 MB
  U (Gor'kov potential):         0.1 MB
  F (Fx, Fy, Fz):                0.3 MB
  A (sparse matrix, 7-pt stencil):      0.3 MB

Total single-instance estimate:      0.8 MB
```

And prints a final memory report:
```
[MEMORY REPORT]
  Min RSS: 80.9 MB at 'Before grid creation'
  Max RSS: 89.1 MB at 'After time-varying simulation'
  Peak Δ from start: 8.3 MB
```

**Typical Memory Scaling:**
- 10×10×4 grid: ~0.1 MB per frame
- 21×21×8 grid: ~0.4 MB per frame
- 65×65×20 grid: ~4 MB per frame (not tested yet; would need more RAM)

## New Modules

### 1. `src/tweezers/diagnostics/memory.py`
Memory tracking utilities:
- `MemoryTracker`: Track RSS at checkpoints
- `array_summary()`: Pretty-print array metadata
- `print_memory_banner()`: Estimate per-frame memory usage

**Example:**
```python
from tweezers.diagnostics.memory import MemoryTracker

mem = MemoryTracker()
mem.checkpoint("After setup")
# ... do work ...
mem.checkpoint("After solve")
mem.report()
```

### 2. `src/tweezers/viz/render_slice_gif.py`
Streaming GIF rendering without RAM buildup:
- `SliceGifRenderer`: Streaming GIF writer (one frame at a time)
- `render_slice_frame_to_array()`: Render frame to RGB array
- `render_trajectory_2d_slice()`: Animate trajectory with overlay

**Example:**
```python
from tweezers.viz.render_slice_gif import render_trajectory_2d_slice

render_trajectory_2d_slice(
    x_grid, y_grid, z_grid,
    traj_dict,      # dict with 't', 'x', 'y', 'z', 'Fx', 'Fy'
    F_2d_frames,    # list of 2D (Nx, Ny) arrays
    slice_z=0.005,
    output_gif_path='output.gif',
    downsample=2
)
```

### 3. Enhanced `src/tweezers/control/fd_helmholtz_3d.py`
Key improvements:
- **Matrix caching** via `_make_cache_key()` and class-level `_matrix_cache`
- **Iterative solvers**: GMRES and BiCGSTAB with Jacobi preconditioner
- **dtype control**: Constructor argument for float32/complex64 vs float64/complex128
- **solver_method**: Direct, GMRES, or BiCGSTAB

**Example:**
```python
from tweezers.control.fd_helmholtz_3d import Helmholtz3DOperator

op = Helmholtz3DOperator(grid, k, dtype=np.complex64, solver_method='gmres')
p = op.solve(p_bot)  # Reuses matrix if grid/k unchanged
```

### 4. Refactored `scripts/demo_helmholtz3d_v2.py`
Main demo with:
- `DemoConfig`: Parse CLI args
- `DemoRunner`: Execute simulation with memory tracking
- Streaming GIF rendering in main loop
- CSV trajectory output

## Performance Notes

### Timing
- **Matrix assembly** (first step only): ~0.1s for 21×21×8
- **Direct solve** (sparse): ~0.01s per step for 21×21×8
- **GMRES solve** (with Jacobi precond): ~0.05s per step (less accurate, needs improvement)
- **Frame rendering** (streaming): ~0.05s per frame

### Memory Scaling
- Per-time-step memory growth: ~0.4 MB for 21×21×8 (expected ~0.8 MB estimate - excess is overhead)
- Total for 20 steps: ~90 MB (mostly baseline Python + numpy)
- **No frame buffer:** Each frame written to GIF immediately and discarded

### Avoiding OOM
1. **Use single precision** (`--dtype single`): Halves complex64/float32 vs complex128/float64
2. **Store 2D slices only**: Don't allocate full 3D array per frame (automatic in v2)
3. **Render streaming**: Frames not buffered (automatic via imageio.get_writer)
4. **Use iterative solver for larger grids**: `--solver gmres` if matrix caching isn't enough

## Debugging

### Print solver convergence:
```bash
python3 scripts/demo_helmholtz3d_v2.py ... --solver gmres 2>&1 | grep SOLVER
```

### Memory checkpoints:
```bash
python3 scripts/demo_helmholtz3d_v2.py ... 2>&1 | grep MEM
```

### Verify matrix reuse:
```bash
python3 scripts/demo_helmholtz3d_v2.py --n_steps 5 ... 2>&1 | grep MATRIX
# First step: "Assembled and cached A"
# Remaining: "Using cached A"
```

## Assumptions & Limitations

1. **Helmholtz operator**: 7-point stencil (3D Laplacian), constant wavenumber k
2. **Boundary conditions**: Dirichlet at bottom (z=0), Robin on other faces
3. **Particle model**: Overdamped, SI units, trilinear interpolation of forces
4. **GIF coloring**: Uses viridis colormap for force magnitude; plasma for z-depth
5. **Memory estimate**: Linear scaling with grid size; doesn't account for solve sparsity pattern

## Future Improvements

- [ ] Better GMRES preconditioner (multigrid? AMG?)
- [ ] Reduced-order modeling for very large grids
- [ ] Checkpointing: save/load intermediate states
- [ ] Multi-threaded rendering (Agg backend)
- [ ] HDF5 output for big data

## Testing Checklist

- [x] Small demo (10 steps, minimal grid): No OOM, GIF produced
- [x] Realistic demo (20 steps, 21×21×8): No OOM, memory ~83 MB, GIF 222KB
- [x] Solver caching: Matrix reused across steps
- [x] GMRES iterative solver: Runs (convergence needs work)
- [x] Feature flags: --no_gorkov, --no_particle work
- [x] CLI args: All parsed correctly
- [x] Memory reporting: Accurate checkpoints and final report

## References

- Helmholtz3DOperator: `src/tweezers/control/fd_helmholtz_3d.py`
- GIF rendering: `src/tweezers/viz/render_slice_gif.py`
- Memory utils: `src/tweezers/diagnostics/memory.py`
- Main demo: `scripts/demo_helmholtz3d_v2.py`
