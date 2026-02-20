# Linux Setup Guide — Acousto-Tweezers

This guide covers setting up and running the acousto-tweezers project on
Bristol Linux desktops.

---

## 1  Clone & Environment

```bash
git clone <repo-url> ~/Desktop/acousto-tweezers
cd ~/Desktop/acousto-tweezers

# Create the FEniCSx environment with complex PETSc
micromamba create -f environment/complex-fenicsx.yml
micromamba activate fenicsx

# Install the package in editable mode
pip install -e .
```

> **Note:** The environment is called `fenicsx`.  All solver commands must use
> `micromamba run -n fenicsx python <script>` to ensure complex PETSc scalars
> are available.

---

## 2  Verify Environment

### 2.1  Check DOLFINx import

```bash
micromamba run -n fenicsx python -c "import dolfinx; print(f'DOLFINx {dolfinx.__version__}')"
```

### 2.2  Check complex PETSc

```bash
micromamba run -n fenicsx python -c "
from petsc4py import PETSc
import numpy as np
ok = np.issubdtype(PETSc.ScalarType, np.complexfloating)
print(f'PETSc ScalarType = {PETSc.ScalarType}  complex = {ok}')
assert ok, 'ERROR: PETSc must be built with complex scalars'
print('PASS')
"
```

### 2.3  Check MUMPS availability

```bash
micromamba run -n fenicsx python -c "
from petsc4py import PETSc
ksp = PETSc.KSP().create()
pc = ksp.getPC()
pc.setType('lu')
pc.setFactorSolverType('mumps')
print('MUMPS: available')
ksp.destroy()
"
```

### 2.4  30-second smoke test

```bash
micromamba run -n fenicsx python -c "
import sys, numpy as np
sys.path.insert(0, 'src')
from acoustweezers.experiments.farfield_petri_cuboid.config import FarFieldConfig
from acoustweezers.experiments.farfield_petri_cuboid.solve_pressure import solve_helmholtz

cfg = FarFieldConfig(
    Lx=6e-3, Ly=6e-3, H_under=3e-3, H_top=1e-3,
    frequency_hz=2e6, disk_radius=1e-3,
    disk_velocity_amplitude=1e-6,
    standing_velocity_amplitude=0.0,
    elements_per_wavelength=3,
    lens_drive='plastic', lens_l=1,
    lens_focal_length=10e-3,
    pml_enabled=True,
)
sol = solve_helmholtz(cfg, verbose=True, petsc_options={
    'ksp_type': 'preonly', 'pc_type': 'lu',
    'pc_factor_mat_solver_type': 'mumps',
})
print(f'max|p| = {sol.max_pressure:.2f} Pa  DOFs = {sol.dofs}')
assert sol.max_pressure > 0.1, 'Pressure too low — solver may have failed'
print('SMOKE TEST PASSED')
"
```

Expected output: converged solve (~8 s), max|p| ≈ 2 Pa, 79 233 DOFs.

---

## 3  Output Folder Location

**All heavy 3D visualisation output (VTU, PVTU, H5, XDMF) is saved to:**

```
~/OneDrive - University of Bristol/Major Project Onedrive/Research/Vortex 3D visualisation
```

This directory is **outside the git repository** so large binary files are
never committed.  OneDrive syncs these files automatically to the cloud for
backup and cross-device access.

Scripts auto-create subdirectories:

| Script | Output subfolder |
|--------|-----------------|
| `export_vortex_3d.py` | `.../Vortex3D/` |
| `diagnostics_lens_propagation.py` | `.../Diagnostics_LensPropagation/` |
| `diagnostics_interaction.py` | `.../Diagnostics_Interaction/` |
| `run_axicon_lens_demo.py` | `.../AxiconLensDemo/` |

Light results (CSV, PNG, JSON) still go to `results/` inside the repo but are
git-ignored.

---

## 4  Troubleshooting

### 4.1  MUMPS OOM (Out-of-Memory)

MUMPS direct solver requires substantial RAM for large meshes.

| Mesh | Approx DOFs | RAM needed |
|------|-------------|-----------|
| 3 elem/λ | 79 K | ~3 GB |
| 5 elem/λ | 348 K | ~16 GB |
| 8 elem/λ | 1.4 M | ~64 GB |

**Workarounds:**

- Use `elements_per_wavelength=3` for quick tests.
- Add MUMPS memory hints:
  ```python
  petsc_options = {
      "ksp_type": "preonly", "pc_type": "lu",
      "pc_factor_mat_solver_type": "mumps",
      "mat_mumps_icntl_14": "80",    # % increase working space
      "mat_mumps_icntl_23": "4000",  # max working memory MB
  }
  ```
- On machines with < 8 GB RAM, stick to 3 elem/λ.

### 4.2  MPI vs Serial

The codebase runs in **serial** by default (single MPI rank).  If you see
MPI-related errors:

```bash
# Force serial execution
micromamba run -n fenicsx python script.py

# Do NOT use mpirun unless the script explicitly supports it
# mpirun -n 4 python script.py   # ← may cause issues
```

For PVTU (parallel VTU) output, MPI with >1 rank is needed but is optional —
the default VTU export works in serial.

### 4.3  ParaView Loading

1. Open ParaView (≥ 5.11 recommended).
2. **File → Open** → navigate to the OneDrive output folder.
3. Load `.vtu` files directly.  For `.xdmf` files, ensure the corresponding
   `.h5` file is in the same directory.
4. Recommended colormaps:
   - `p_abs` → **viridis** (0 to max)
   - `p_arg` → **twilight** (−π to +π, cyclic)
   - `Iz` → **coolwarm** (diverging, centred on 0)

### 4.4  OneDrive Sync Delays

OneDrive on Linux may take 30–120 seconds to sync new files.  If files appear
missing on another device:

- Check `~/.config/onedrive/` logs.
- Force sync: `onedrive --synchronize --single-directory 'Major Project Onedrive/Research/Vortex 3D visualisation'`
- If OneDrive is not installed, files are still written locally at the path
  above and can be copied manually.

### 4.5  Display / Matplotlib Errors

On headless Linux (SSH without X forwarding), matplotlib may fail.  All scripts
use `matplotlib.use("Agg")` to avoid this.  If you still get display errors:

```bash
export MPLBACKEND=Agg
```
