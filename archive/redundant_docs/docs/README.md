⚠️ **Note**: This documentation index includes historical reports from February 2026. For the **current working physics**, see [../README.md](../README.md) § "Current Working Workflow: 3D Acoustic Fields & ParaView Visualisation".

---

# Acousto-Tweezers Documentation

**Project:** Advanced Acoustic Manipulation Framework  
**Platform:** FEniCSx + Complex PETSc  
**Last Updated:** February 8, 2026 (v3.0.0)

---

## Quick Navigation

### Current Active Documentation

- **[../README.md](../README.md) § "Current Working Workflow"** ⭐ **START HERE FOR PRODUCTION USE**
  - Validated physics (Helmholtz, Gor'kov, particles)
  - ParaView visualization workflow with step-by-step instructions
  - Known limitations and future work
  - Typical value ranges for all quantities

- **[DEVLOG_20260208_streaming_particles.md](DEVLOG_20260208_streaming_particles.md)**
  - Shallow dish implementation details (v3.0.0)
  - File manifest, reproduction instructions
  - Validation results from February 8, 2026

### Historical Reports (Superseded but Retained for Provenance)

⚠️ These documents chronicle the implementation journey. Physics is now integrated into v3.0.0. See [../results/ARCHIVE_PRE_3D_PARAVIEW_WORKFLOW/](../results/ARCHIVE_PRE_3D_PARAVIEW_WORKFLOW/) for associated results.

- **[VORTEX_COMPLETE_SUMMARY.md](VORTEX_COMPLETE_SUMMARY.md)** (February 7, 2026)
  - Complete vortex lens implementation summary (Stages 1+2)
  - ✅ Now integrated into v3.0.0 workflow

- **[VORTEX_SUPERPOSITION_REPORT.md](VORTEX_SUPERPOSITION_REPORT.md)** (Stage 2)
  - Standing wave + vortex superposition validation
  - ✅ Now integrated into v3.0.0 workflow

- **[VORTEX_IMPLEMENTATION_REPORT.md](VORTEX_IMPLEMENTATION_REPORT.md)** (Stage 1)
  - Basic vortex lens physics implementation
  - ✅ Now integrated into v3.0.0 workflow

### Legacy Visualizations

- **[PYVISTA_VIZ.md](PYVISTA_VIZ.md)** (superseded by ParaView)
  - PyVista 3D visualization guide
  - ⚠️ Use ParaView instead (see main README)

- **[PHASE_SWEEP_STATUS.md](PHASE_SWEEP_STATUS.md)**
  - Phase variation studies from early February
  - ⚠️ Results archived; physics now unified in v3.0.0

### Core Documentation

- **Physics Models**
  - [Helmholtz 3D](physics/helmholtz_3d.md) - Frequency-domain acoustics
  - [PML Theory](physics/pml_theory.md) - Perfectly Matched Layers
  - [Gor'kov Potential](physics/gorkov_radiation_force.md) - Radiation force theory

- **Numerical Methods**
  - [FEM Implementation](refactor/fem_implementation.md)
  - [Complex-Valued Problems](refactor/complex_petsc.md)
  - [Mesh Generation](refactor/mesh_generation.md)

- **Experiments**
  - [Square Dish](square_dish/README.md) - Phase control arrays
  - [Vortex Lens](VORTEX_COMPLETE_SUMMARY.md) - Topological phase patterns

### Legacy Documentation

- **[Archive](archive/)** - Historical implementation notes and deprecated code

---

## Implementation Status

### ✅ Completed Features

| Feature | Status | Documentation |
|---------|--------|---------------|
| Helmholtz 3D Solver | ✅ Complete | [HELMHOLTZ3D_README.md](HELMHOLTZ3D_README.md) |
| PML Boundaries | ✅ Complete | [PML_README.md](PML_README.md) |
| Square Dish Arrays | ✅ Complete | [square_dish/](square_dish/) |
| Gor'kov Computation | ✅ Complete | Integrated in solvers |
| **Vortex Lens (Stage 1)** | ✅ Complete | [VORTEX_IMPLEMENTATION_REPORT.md](VORTEX_IMPLEMENTATION_REPORT.md) |
| **Vortex Superposition (Stage 2)** | ✅ Complete | [VORTEX_SUPERPOSITION_REPORT.md](VORTEX_SUPERPOSITION_REPORT.md) |
| Time Evolution | ✅ Complete | `phase2_time_evolution.py` |
| Particle Dynamics | ✅ Complete | Overdamped integrator |
| 3D Visualization | ⚠️ Partial | PyVista (requires Xvfb) |

### 🔄 In Progress

| Feature | Status | Notes |
|---------|--------|-------|
| Vortex Particles | 🔄 Framework ready | Stage 3 optional |
| Gor'kov Difference | 🔄 Framework ready | Stage 3 optional |
| Gain Calibration | 🔄 Manual | Auto-calibration Stage 3 |

### 📋 Planned Features

| Feature | Priority | Estimated Time |
|---------|----------|----------------|
| Multi-vortex arrays | Medium | 2-3 hours |
| Time-domain vortex | Low | 4-6 hours |
| Optimization framework | Medium | 8-10 hours |
| Experimental validation | High | N/A (hardware dependent) |

---

## Usage Examples

### Basic Vortex Demo

```bash
# Single vortex (ℓ=1)
python scripts/validation/demo_vortex.py --topological_charge 1

# Higher-order vortex (ℓ=2)
python scripts/validation/demo_vortex.py --topological_charge 2
```

**Outputs:** `results/vortex_demo/run_*/`

### Vortex + Standing Wave Comparison

```bash
# Preset A: 2cm dish, 500 kHz
python scripts/validation/compare_vortex_standing.py --preset A

# Preset B: 3mm dish, 2 MHz
python scripts/validation/compare_vortex_standing.py --preset B --topological_charge 2
```

**Outputs:** `results/comparison_<preset>_<timestamp>/`

### Square Dish Phase Control

```bash
# 4-puck surface greedy optimization
python scripts/4puck_demo_surf_greedy.py --num_elements 4 --elements_per_wavelength 6

# MPC tracking
python scripts/macro_actions_4puck.py --control_type mpc
```

**Outputs:** `results/4puck_demo_surf_greedy/`, `results/mpc_*/`

---

## Documentation Structure

```
docs/
├── README.md (this file)
│
├── VORTEX_COMPLETE_SUMMARY.md ⭐ Complete vortex implementation overview
├── VORTEX_SUPERPOSITION_REPORT.md (Stage 2: superposition + presets)
├── VORTEX_IMPLEMENTATION_REPORT.md (Stage 1: basic vortex)
│
├── HELMHOLTZ3D_README.md (3D frequency-domain solver)
├── PML_README.md (Perfectly Matched Layers)
├── MULTIPHYSICS_README.md (FEM + particle dynamics)
│
├── physics/ (Theory and equations)
│   ├── helmholtz_3d.md
│   ├── pml_theory.md
│   └── gorkov_radiation_force.md
│
├── refactor/ (Implementation details)
│   ├── fem_implementation.md
│   ├── complex_petsc.md
│   └── mesh_generation.md
│
├── square_dish/ (Experiment-specific)
│   └── README.md
│
└── archive/ (Legacy documentation)
```

---

## Technical Specifications

### Software Requirements

**Core Dependencies:**
- FEniCSx v0.7+ (with complex PETSc)
- Python 3.10+
- NumPy, SciPy, Matplotlib
- MPI4Py, PETSc4Py

**Optional:**
- PyVista (3D visualization)
- ADIOS2 (parallel I/O)
- Xvfb (headless rendering)

### Hardware Recommendations

**Minimum:**
- 4 GB RAM
- Single core sufficient for small problems
- ~1 GB disk space for results

**Recommended:**
- 16 GB RAM (for large meshes)
- Multi-core CPU (MPI parallelism)
- ~10 GB disk space (with full result archive)

### Performance Benchmarks

| Problem Size | DOFs | Elements/λ | Solve Time (1 core) | Memory |
|--------------|------|------------|---------------------|--------|
| Small (5mm) | ~200k | 6 | ~10 sec | ~500 MB |
| Medium (2cm) | ~1.7M | 6 | ~60 sec | ~1.5 GB |
| Large (10cm) | ~20M | 6 | ~600 sec | ~15 GB |

*P2 Lagrange elements, GMRES+ILU solver, complex scalars*

---

## Citing This Work

If you use this framework in your research, please cite:

```bibtex
@software{acousto_tweezers_2026,
  title = {Acousto-Tweezers: Advanced Acoustic Manipulation Framework},
  author = {[Your Name]},
  year = {2026},
  url = {https://github.com/your-repo/acousto-tweezers},
  note = {FEniCSx-based implementation of acoustic vortex lenses and phase control}
}
```

**Key Publications:**
- Gor'kov, L. P. (1962). "On the forces acting on a small particle in an acoustical field in an ideal fluid." *Soviet Physics Doklady*, 6, 773-775.
- Marzo, A., et al. (2015). "Holographic acoustic elements for manipulation of levitated objects." *Nature Communications*, 6, 8661.

---

## Contributing

This is a research code under active development. Contributions welcome:

1. **Bug Reports:** Open issue with minimal reproducible example
2. **Feature Requests:** Describe use case and expected behavior
3. **Code Contributions:** Fork, implement, test, submit PR

**Code Style:**
- Follow PEP 8 for Python
- Document all functions with NumPy-style docstrings
- Add unit tests for new solvers
- Update relevant documentation

---

## License

[Specify license here, e.g., MIT, GPL-3.0, Apache-2.0]

---

## Contact

**Primary Maintainer:** [Your Name]  
**Email:** [your.email@institution.edu]  
**Lab:** [Your Research Group]  
**Institution:** [Your University]

---

## Changelog

### v2.0 - February 7, 2026
- ✨ Added acoustic vortex lens implementation (Stage 1+2)
- ✨ Finite aperture with cosine taper
- ✨ Standing wave + vortex superposition
- ✨ Dish size presets (Preset A, B)
- ✨ 3D field export (ADIOS2 BP4)
- ✨ Difference plot visualization
- 📚 Three comprehensive reports (1,100+ lines)
- 🐛 Fixed UFL complex conjugation issues
- 🐛 Fixed func.eval() shape mismatch

### v1.5 - January 2026
- ✨ Square dish phase control
- ✨ 4-puck surface greedy optimization
- ✨ MPC trajectory tracking
- 🔧 PML boundary refinement

### v1.0 - December 2025
- 🎉 Initial release
- ✨ Helmholtz 3D solver with PML
- ✨ Gor'kov potential computation
- ✨ Particle dynamics (overdamped)

---

**End of Documentation Index**
