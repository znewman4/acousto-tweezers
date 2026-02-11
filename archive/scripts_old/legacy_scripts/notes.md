# Redundant Scripts

This folder contains scripts that are no longer the primary entry points.
They are preserved for reference but should NOT be used for production runs.

## The ONLY blessed entry point is:
```bash
python scripts/run_fem_multiphysics.py --level ACOUSTICS_ONLY --quick
```

---

## Files and Why They're Redundant

### Control/MPC Scripts (Legacy Control Module)
| File | Purpose | Status |
|------|---------|--------|
| `4puck_demo_surf_greedy.py` | 4-puck surface greedy control demo | Uses legacy FD acoustics |
| `adjoint_*.py` | Adjoint-based gradient computation | Uses legacy FD solver |
| `bc_sensitivity_compare.py` | BC sensitivity analysis | Uses legacy FD grid |
| `control_authority_diagnostic.py` | Control authority analysis | Uses legacy control module |
| `control_authority_diagnostic_4puck.py` | 4-puck control analysis | Uses legacy control module |
| `macro_actions_4puck.py` | Macro action generation | Uses legacy control module |
| `mpc_vs_greedy_4puck.py` | MPC vs greedy comparison | Uses legacy control module |
| `optimized_mpc_comparison.py` | Optimized MPC comparison | Uses legacy control module |
| `path_tracking_comparison.py` | Path tracking comparison | Uses legacy control module |

### Development/Demo Scripts  
| File | Purpose | Status |
|------|---------|--------|
| `demo_2d_acoustics.py` | 2D acoustics demo | Development aid, not production |
| `demo_helmholtz3d.py` | 3D Helmholtz demo | Development aid |
| `demo_helmholtz3d_v2.py` | 3D Helmholtz v2 | Development aid |
| `demo_helmholtz3d_multiphysics.py` | Multiphysics demo | Development aid |
| `demo_surf_greedy.py` | Surface greedy demo | Uses legacy FD solver |
| `generate_acoustic_animation.py` | GIF generation | Merged into main solver |
| `render_particle_gif.py` | Particle trajectory GIF | Merged into main solver |

### Validation (Legacy)
| File | Purpose | Status |
|------|---------|--------|
| `validate_actuation_pipeline.py` | Actuation validation | Uses legacy approach |
| `validate_helmholtz3d.sh` | Shell validation script | Replaced by validation/ tests |

---

## Migration Guide

If you were using one of these scripts, migrate to:

```python
# All physics levels through blessed entry point:
python scripts/run_fem_multiphysics.py --level ACOUSTICS_ONLY
python scripts/run_fem_multiphysics.py --level ACOUSTICS_PML
python scripts/run_fem_multiphysics.py --level FLUID_SOLID
python scripts/run_fem_multiphysics.py --level PARTICLES

# For validation:
python scripts/validation/test_pml_reflection.py
python scripts/validation/test_interface_continuity.py
python scripts/validation/test_cavity_eigenmode.py
python scripts/validation/test_plate_impedance.py
```

---

Last updated: January 2026
