# Acousto-Tweezers: Complex Phasor Acoustic Simulation

FEniCSx-based multiphysics simulator for acoustic tweezers with validated vortex topology and streaming.

---

## ✅ Validated Physics (February 2026)

| Module | Status | Implementation |
|--------|--------|----------------|
| **Complex Phasor Helmholtz** | ✓ Validated | Time-harmonic with complex scalars (PETSc complex) |
| **Vortex Topology** | ✓ Validated | Phase winding = topological charge exactly |
| **Level-2 Acoustic Streaming** | ✓ Validated | Phasor-based Reynolds stress, Stokes flow |
| **Gor'kov Radiation Force** | ✓ Validated | Monopole + dipole contrast factors |
| **Overdamped Particle Dynamics** | ✓ Implemented | Stokes drag + streaming advection |

**Evidence:** See [PHYSICS_STATUS.md](PHYSICS_STATUS.md) for full documentation.

---

## 🚀 Quick Start

### Requirements

**CRITICAL:** Must use PETSc with complex scalar support.

```bash
# Verify environment:
python -c "from petsc4py import PETSc; print(PETSc.ScalarType)"
# Must output: <class 'numpy.complex128'>
```

**Environment:** Use `acousto-complex` environment (see [environment/complex-fenicsx.yml](environment/complex-fenicsx.yml))

```bash
micromamba env create -f environment/complex-fenicsx.yml
micromamba activate acousto-complex
```

### Validation Scripts

```bash
# 1. Verify complex backend
micromamba run -n acousto-complex python scripts/validation/test_complex_backend_runtime.py

# 2. Run streaming diagnostics (with energy budget)
micromamba run -n acousto-complex python scripts/validation/run_complex_streaming_diagnostics.py

# 3. Run particle deposition experiment (3-phase protocol)
micromamba run -n acousto-complex python scripts/validation/run_deposition_experiment.py
```

### Outputs

**Diagnostics CSV:**
- `results/diagnostics/complex_streaming_proof.csv`
- Columns: petsc_scalar_type, phase_winding_number, pressure/streaming magnitudes, energy budget

**VTU Files (ParaView):**
- `pressure_fields.bp` — Complex pressure (real, imag, abs, phase)
- `streaming_fields.bp` — Acoustic streaming velocity
- `gorkov.bp` — Gor'kov potential
- `particles_timeseries.csv` — U(t), |F|(t), |u_s|(t), χ(t), dist_to_min

---

## 📊 Key Results

### Complex Streaming Validation

Latest validated run (2026-02-09):

| Metric | Value | Status |
|--------|-------|--------|
| PETSc scalar type | complex | ✓ |
| Phase winding (ℓ=1) | 1.000 | ✓ (exact) |
| max\|p\| | 1108 Pa | ✓ |
| max\|u_s\| | 20.8 μm/s | ✓ |
| Dissipation/Acoustic ratio | 2.1×10⁻⁷ | ✓ |

### Deposition Experiment Protocol

**Phase 1 (0.5s):** Vortex only → guide particle toward center  
**Phase 2 (1.0s):** Add standing wave → trap forms, U(t) decreases  
**Phase 3 (0.5s):** Stability test → χ < 1, particle stays trapped

**Pass Criteria:**
- U(t) decreases when trap activates
- χ(t) drops below 1
- Particle stays within 100 μm for O(seconds)
- Approaches Gor'kov minimum

---

## 📁 Repository Structure

```
acousto-tweezers/
├── src/acoustweezers/experiments/shallow_square_dish/
│   ├── config.py           # Configuration dataclass
│   ├── solve_pressure.py   # Complex Helmholtz with vortex source
│   ├── streaming.py        # Level-2 Stokes streaming
│   └── particles.py        # Gor'kov + overdamped dynamics
├── scripts/
│   ├── validation/         # Validation and diagnostics scripts
│   └── diagnostics/        # Energy budget utilities
├── results/
│   ├── diagnostics/        # CSV/JSON outputs
│   └── ARCHIVE_*/          # Archived runs
├── environment/
│   └── complex-fenicsx.yml # PETSc complex environment
├── PHYSICS_STATUS.md       # Detailed physics documentation
└── README.md               # This file
```

---

## 🧪 Physics Details

### Complex Phasor Helmholtz

**Equation:** ∇²p + k²p = 0  
**Boundary Conditions:**
- Robin (impedance): ∂p/∂n = -iωp/Z
- Neumann (actuation): ∂p/∂n = -iωρ v_n(x)

**Vortex Source:** v_n(r,θ) = A(r) exp(iℓθ)  
**Validation:** Phase winding ∮ d(arg p) = 2πℓ

### Acoustic Streaming

**First-order velocity (phasor):** v₁ = ∇p / (iωρ)  
**Time-averaged Reynolds stress:** ⟨ρ v⊗v⟩ = (ρ/2) Re(v₁ ⊗ v₁*)  
**Forcing:** f = -∇·⟨ρ v⊗v⟩  
**Stokes solve:** -μΔu + ∇q = f, ∇·u = 0

### Gor'kov Potential

**Potential:** U = V_p [f₁⟨p²⟩/(2K) - f₂(3ρ/4)⟨v²⟩]  
**Radiation Force:** F_rad = -∇U  
**Particle Motion:** ẋ = u_s(x) + μ F_rad(x)

**Dimensionless competition:** χ = |u_s| / (|F_rad|/(6πηa))  
- χ < 1: Trapping dominates
- χ > 1: Streaming dominates

---

## 📖 Documentation

- **[PHYSICS_STATUS.md](PHYSICS_STATUS.md)** — Validated physics, equations, evidence
- **[CHANGELOG.md](CHANGELOG.md)** — Development history
- **[docs/](docs/)** — Archived reports and implementation details

---

## ❌ Not Implemented (Future Work)

- Thermoviscous boundary layers (requires BL-resolved mesh)
- Inter-particle forces (Bjerknes, hydrodynamic coupling)
- Free surface with gravity (currently simplified impedance BC)
- Transient acoustics (fully time-dependent)

---

## 🔬 How to Reproduce Figures

### Figure 1: Vortex Phase Singularity

```bash
micromamba run -n acousto-complex python scripts/validation/run_complex_streaming_diagnostics.py
# Open ParaView: results/complex_diagnostics_*/pressure_fields.bp
# Slice at z=H/2, color by p_phase (use HSV colormap)
```

### Figure 2: Streaming Velocity Field

```bash
# Same run as above
# Open streaming_fields.bp in ParaView
# Apply Glyph filter (arrows) or Stream Tracer
```

### Figure 3: Particle Deposition

```bash
micromamba run -n acousto-complex python scripts/validation/run_deposition_experiment.py
# CSV: results/deposition_*/particles_timeseries.csv
# Plot U(t), chi(t), position vs time
```

---

## 📝 Citation

If you use this code, please cite the following key references:

**Gor'kov Theory:**  
Gor'kov, L.P. (1962). On the forces acting on a small particle in an acoustical field in an ideal fluid. *Soviet Physics Doklady*, 6, 773-775.

**Acoustic Streaming:**  
Lighthill, M.J. (1978). Acoustic streaming. *Journal of Sound and Vibration*, 61(3), 391-418.

**Acoustic Vortices:**  
Hefner, B.T., & Marston, P.L. (1999). An acoustical helicoidal wave transducer. *The Journal of the Acoustical Society of America*, 106(6), 3313-3316.

---

## 👤 Author

**Project:** Acousto-Tweezers (2026 Major Project)  
**Institution:** University of Bristol

For implementation details, see commit history and `PHYSICS_STATUS.md`.
