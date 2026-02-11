# DEVLOG: Streaming + Particles Implementation
**Date Started**: 2026-02-08  
**Goal**: Implement device-realistic 3D shallow dish model with streaming and particle dynamics

---

## Chronological Log

### 2026-02-08: Initial Setup and Full Implementation

#### Files Created/Modified

**New Files:**
- `docs/DEVLOG_20260208_streaming_particles.md` (this file)
- `results/INDEX.md` — Index of canonical runs and results structure
- `src/acoustweezers/core/export_paraview.py` — Consolidated ParaView export utilities
- `src/acoustweezers/experiments/shallow_square_dish/__init__.py` — Module init
- `src/acoustweezers/experiments/shallow_square_dish/config.py` — Device configuration
- `src/acoustweezers/experiments/shallow_square_dish/solve_pressure.py` — Helmholtz solver
- `src/acoustweezers/experiments/shallow_square_dish/streaming.py` — Reynolds stress streaming
- `src/acoustweezers/experiments/shallow_square_dish/particles.py` — Gor'kov + trajectories
- `src/acoustweezers/experiments/shallow_square_dish/export.py` — VTU export
- `scripts/shallow_dish/run_device_demo.py` — Main entry point

**Modified Files:**
- `scripts/tools/cleanup_results.py` — Extended with auto_archive_exploratory() and generate_index()
- `README.md` — Added shallow dish workflow section

#### Key Modelling Assumptions

1. **Device geometry**: 5 cm × 5 cm × 5 mm shallow dish (aspect ratio 10:1)
2. **Bottom (z=0)**: Vortex lens actuation with moveable center
   - Neumann BC: ∂p/∂n = -iωρ v_n(x,y)
   - Vortex pattern: v_n = V_vtx * A(r) * exp(i*ℓ*θ)
3. **Side walls**: Standing wave transducers
   - Anti-phase pattern by default (x=0 vs x=L)
   - Full-wall actuation (simplified; aperture mask optional)
4. **Top (z=H)**: Free surface / air interface
   - Low impedance Robin BC: Z_top = 0.001 * Z_water
5. **Frequency**: 500 kHz (λ ≈ 2.97 mm in water)
6. **Streaming**: Reynolds stress forcing → Stokes solve
   - f = -∇·⟨ρ v₁⊗v₁⟩
   - -μ∇²u_s + ∇p_s = f, ∇·u_s = 0
7. **Particles**: Overdamped dynamics
   - ẋ = u_s(x) + μ F_rad(x)
   - μ = 1/(6πηa) Stokes mobility

---

## What Changed

### A) Repository Organization
1. Created `results/INDEX.md` documenting canonical runs
2. Extended `scripts/tools/cleanup_results.py` for auto-archiving
3. Consolidated ParaView export in `src/acoustweezers/core/export_paraview.py`

### B) New Experiment Module
Created `src/acoustweezers/experiments/shallow_square_dish/` with:
- `config.py`: ShallowDishConfig dataclass with all parameters
- `solve_pressure.py`: Complex Helmholtz solver with device-aligned BCs
- `streaming.py`: Reynolds stress → Stokes streaming solver
- `particles.py`: Gor'kov potential, radiation force, trajectory integration
- `export.py`: VTU/XDMF export with all required arrays

### C) Main Entry Point
Created `scripts/shallow_dish/run_device_demo.py`:
- Single command to run complete workflow
- CLI arguments for all key parameters
- Generates all VTU files for ParaView

---

## How to Reproduce

```bash
# Activate environment
micromamba activate acousto-complex

# Run the complete device demo (with defaults)
python scripts/shallow_dish/run_device_demo.py

# Or with custom parameters:
python scripts/shallow_dish/run_device_demo.py \
  --L 0.05 --H 0.005 --freq 500e3 \
  --standing_gain 1.0 \
  --vortex_gain 10.0 --ell 1 --aperture_radius_mm 4 \
  --n_particles 5 --t_max 0.1 \
  --out results/device_shallow_custom/

# Path tracking (vortex moves along line):
python scripts/shallow_dish/run_device_demo.py \
  --vortex_path line --n_steps 20

# Skip streaming (faster, just pressure + Gor'kov):
python scripts/shallow_dish/run_device_demo.py --skip_streaming
```

---

## What to Look at in ParaView

### Step 1: Load Combined Pressure Field

1. **File → Open** → `combined_fields.vtu`
2. Click **Apply**
3. Color by `p_mag` to see standing wave pattern
4. **Filters → Slice** at z = 2.5 mm to see horizontal pattern
5. Color by `p_phase` using **twilight** colormap to see phase structure

### Step 2: Visualize Streaming (Tornado-like Flow)

1. **File → Open** → `streaming_fields.vtu`
2. Click **Apply**
3. **Filters → Stream Tracer**
   - Seed Type: **Point Cloud**
   - Center: (25, 25, 1) mm (near vortex core)
   - Radius: 5 mm
   - Number of Points: 50
4. Color streamlines by `streaming_velocity` magnitude

### Step 3: Overlay Gor'kov Traps + Particle Path

1. **File → Open** → `gorkov_fields.vtu`
2. **Contour** by `U_gorkov` at low values (trap wells)
3. **File → Open** → `particles.csv`
4. **Filters → Table to Points** (x_m, y_m, z_m columns)
5. **Filters → Tube** for visibility
6. Color by `time` to show progression

### Step 4: Compare Delta Fields

1. Load `delta_fields.vtu`
2. **Filters → Slice** at z = 0.5, 2.5, 4.5 mm
3. Color by `delta_p_mag` to verify bulk penetration
4. Expect: Vortex perturbation strongest near bottom, decays toward top

---

## Known Limitations

1. **No thermoviscous effects**: Boundary layer losses not included
2. **Linearized streaming**: Uses slow streaming approximation (Re_s << 1)
3. **No particle-particle interaction**: Single-particle dynamics only
4. **Plane wave Gor'kov approximation**: Velocity term uses gradient projection
5. **Stokes streaming only**: No Rayleigh streaming boundary corrections
6. **Fixed time step**: Adaptive integration not implemented
7. **Uniform mesh**: No refinement near boundaries
8. **Simplified free-slip BC at top**: Only normal velocity constrained

---

## Validation Checks

- [x] Standing-only shows lateral standing wave pattern (nodes at half-wavelength spacing) — **max|p| = 1564 Pa**
- [x] Vortex-only shows phase winding at bottom (azimuthal 2π wrap for ℓ=1) — **max|p| = 8593 Pa**
- [x] Combined shows modulated standing pattern — **max|p| = 8225 Pa**
- [x] Delta fields show vortex contribution propagating through depth — **max|Δp| = 8593 Pa**
- [ ] Streaming shows vortical structures — *Stokes solve requires specialized preconditioner; documented as known limitation*
- [x] Particle trajectories respond to trap (motion ~nm/timestep = correct order of magnitude)
- [x] Gor'kov potential computed correctly — **trap depth = 2.48e-18 J**, **max|F| = 5.47e-15 N**
- [x] Diagnostics JSON contains all required statistics
- [x] All VTU exports created successfully

---

## Known Issues & Limitations

### Streaming Solver (Current Limitation)

The Stokes saddle-point system for acoustic streaming requires a specialized block preconditioner. The current implementation uses GMRES with block Jacobi which fails to converge for this problem. 

**Future improvements:**
1. Implement Schur complement preconditioner
2. Use specialized Stokes preconditioner from PETSc (fieldsplit with LSC)
3. Reduce to P1-P1 with stabilization (PSPG) for simpler solve

**Workaround:** Use `--skip_streaming` flag to bypass streaming computation. The radiation force from Gor'kov potential still provides particle dynamics.

---

## File Manifest

```
src/acoustweezers/experiments/shallow_square_dish/
├── __init__.py              # Module exports
├── config.py                # ShallowDishConfig dataclass
├── solve_pressure.py        # Helmholtz solver + boundary conditions
├── streaming.py             # Reynolds stress + Stokes streaming
├── particles.py             # Gor'kov potential + trajectory integration
└── export.py                # VTU/XDMF export + diagnostics

scripts/shallow_dish/
└── run_device_demo.py       # Main entry point

src/acoustweezers/core/
└── export_paraview.py       # Consolidated export utilities

results/
└── INDEX.md                 # Results index
```
