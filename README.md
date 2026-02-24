# Acousto-Tweezers

A finite-element simulation engine for acoustic-tweezer devices that combine
standing-wave lattices with movable vortex-beam lenses for selective
micro-particle manipulation in shallow liquid-filled cavities.

Built on **FEniCSx / DOLFINx (v0.9)** with complex PETSc scalars.

---

## Quick Start (Linux)

```bash
# 1. Clone and enter
git clone <repo-url> && cd acousto-tweezers

# 2. Create environment (requires micromamba)
micromamba create -n fenicsx -f environment/complex-fenicsx.yml
micromamba activate fenicsx
pip install -e .

# 3. Verify complex PETSc
micromamba run -n fenicsx python -c \
  "from petsc4py import PETSc; assert PETSc.ScalarType == complex"

# 4. Smoke test
micromamba run -n fenicsx python scripts/validation/test_1d_impedance.py
```

> **Full setup guide:** [docs/LINUX_SETUP.md](docs/LINUX_SETUP.md)

### Where Heavy Results Go

All large 3D files (VTU, H5, XDMF) are written to the OneDrive folder:

```
~/OneDrive - University of Bristol/Major Project Onedrive/
  Research/Vortex 3D visualisation/
```

**Do NOT commit heavy 3D files to the repo.**  The `.gitignore` excludes
`*.vtu`, `*.pvtu`, `*.h5`, and `*.xdmf`.  Lightweight PNG/CSV/JSON results
stay in `results/` inside the repo.

---

## 1  Project Goal

This codebase models the physics of an acoustic tweezer system operating in
water at ultrasonic frequencies (typically 500 kHz – 2 MHz).  The device
geometry is a shallow square petri dish (~6-30 mm lateral, ~1-4 mm deep)
instrumented with:

- **Side-wall transducer pairs** that generate one- or two-axis standing-wave
  pressure fields, creating a periodic lattice of Gor'kov potential minima
  (traps).
- **A bottom-mounted circular vortex transducer** (the "lens") that
  superimposes an orbital-angular-momentum beam, producing localised streaming
  flow and a pressure null at its core.

Three lens models are available:

| Model | Description | Config |
|-------|-------------|--------|
| **Ideal** | Pure $e^{i\ell\theta}$ — no focusing | `lens_drive="ideal"` |
| **Plastic** | Converging spiral-phase lens (fabricable) | `lens_drive="plastic"` |
| **Axicon** | Bessel-like non-diffracting vortex beam | `lens_drive="axicon"` |

By varying the relative amplitudes, phases, and spatial position of the vortex
lens, individual particles can be extracted from the lattice, transported along
programmed paths, and re-deposited — while neighbouring particles remain
trapped.

---

## 2  Physical Model

### 2.1  Geometry and Coordinate System

The computational domain is a rectangular box
$[0, L] \times [0, L] \times [0, H]$ with six planar boundaries:

| Boundary | Location | Physical Role |
|----------|----------|---------------|
| Bottom disc | $z = 0$, $r \le R_\text{disc}$ | Circular vortex transducer (impedance + source) |
| Bottom rigid | $z = 0$, $r > R_\text{disc}$ | Rigid floor (no-flux) |
| Top | $z = H$ | Free surface / air interface (low-impedance) |
| Side walls ($x_\pm$, $y_\pm$) | $x = 0, L$; $y = 0, L$ | Rigid reflectors or standing-wave transducers |

The centre of the disc transducer defaults to $(L/2, L/2, 0)$ and its radius
$R_\text{disc}$ defaults to `vortex_aperture_radius` (4 mm).

### 2.2  Governing Equation

The acoustic pressure phasor $p(\mathbf{x})$ satisfies the time-harmonic
Helmholtz equation in the fluid domain:

$$\nabla^2 p + k^2 p = 0, \qquad k = \omega / c$$

where $\omega = 2\pi f$ is the angular frequency and $c$ the speed of sound in
water.

### 2.3  Boundary Conditions

Three types of boundary condition are used, all arising from the linearised
Euler equation $\nabla p = i\omega\rho\,\mathbf{v}$:

1. **Neumann (velocity source).**  On an active transducer face with prescribed
   normal velocity $v_n$ (positive into the domain):
   $$\frac{\partial p}{\partial n} = -i\omega\rho\,v_n$$
   Standing-wave transducers use a uniform real $v_n$ with a phase pattern
   (anti-phase, in-phase, or quadrature) between opposing walls.  The vortex
   transducer uses
   $v_n(r,\theta) = A(r)\,\exp(i\ell\theta + i\varphi_0)$
   where $\ell$ is the topological charge and $A(r)$ is a cosine-taper
   apodisation profile.

2. **Robin (impedance).**  On the top surface and optionally side walls:
   $$\frac{\partial p}{\partial n} = \frac{i\omega\rho}{Z}\,p$$
   This is moved to the bilinear form as $\alpha\langle u, v\rangle_{ds}$ with
   the Robin coefficient $\alpha = -i\omega\rho / Z$.  The top uses
   $Z_\text{top} = 0.001\,Z_\text{water}$ (modelling the water–air interface).
   Side walls and the bottom rigid region can optionally be given finite
   impedance via `wall_impedance_Zrel` and `bottom_rigid_impedance_Zrel` in
   `ShallowDishConfig` (both default to `None` → rigid).

3. **Rigid (natural Neumann).**  On inactive side walls and the bottom floor
   outside the disc: $\partial p / \partial n = 0$.  No term is added to either
   the bilinear form or the RHS — this is the natural boundary condition of the
   Helmholtz weak form.

**Mode logic (default — rigid walls):**

| Mode | Side walls | Bottom disc |
|------|------------|-------------|
| `standing` | Active Neumann source | Rigid (∂p/∂n = 0) |
| `vortex` | Rigid | Pure Neumann vortex source |
| `combined` | Active Neumann source | Pure Neumann vortex source |

When `wall_impedance_Zrel` is set to a finite value, a Robin impedance term
is added to all four side walls *in addition* to any Neumann source terms.
This enables the Phase 2 lossy-cavity study.

---

## 3  Numerical Method

### 3.1  Discretisation

The weak form is discretised with **FEniCSx / DOLFINx (v0.9)** using:

- Quadratic Lagrange elements (P2) on an unstructured tetrahedral mesh
  generated by `dolfinx.mesh.create_box`.
- **Complex scalar type**: PETSc is built with `--with-scalar-type=complex`
  so that the complex pressure phasor is a first-class degree of freedom.  UFL's
  `inner(u, v)` automatically applies complex conjugation to the test function.

The weak form reads:

$$a(u,v) = \int_\Omega \nabla u \cdot \nabla \bar{v}\,dx
         - k^2 \int_\Omega u\,\bar{v}\,dx
         + \alpha_\text{top}\int_{\Gamma_\text{top}} u\,\bar{v}\,ds
         + \alpha_\text{disc}\int_{\Gamma_\text{disc}} u\,\bar{v}\,ds
         + \sum_{\text{lossy walls}} \alpha_\text{wall}\int_{\Gamma_j} u\,\bar{v}\,ds$$

$$L(v) = \sum_{\text{active walls}} \int_{\Gamma_i} g_i\,\bar{v}\,ds$$

where $g_i = -i\omega\rho\,v_{n,i}$ is the Neumann data and
$\alpha_\text{wall} = -i\omega\rho / Z_\text{wall}$ (only when
`wall_impedance_Zrel` is set; omitted for rigid walls).

### 3.2  Mesh and Facet Tagging

Bottom-boundary facets are segmented at mesh-generation time: each facet's
midpoint is tested against $r \le R_\text{disc}$ to assign either
`TAG_BOTTOM_DISC` (1) or `TAG_BOTTOM_RIGID` (7).  Side walls receive tags
3–6.  The top surface is tag 2.  Duplicate facets at edges/corners are resolved
by first-assignment priority, with the disc tag evaluated first.

### 3.3  Linear Solver

The assembled complex-valued system is solved with **MUMPS direct LU** by
default.  GMRES + ILU is available but diverges on the PML-Helmholtz system
(the complex-indefinite operator defeats ILU preconditioning).  MUMPS requires
more RAM (~7 GB for 79K DOFs at 3 elem/λ) but converges reliably.

---

## 4  Acoustic Streaming (Second-Order Model)

After solving the linear Helmholtz problem for $p(\mathbf{x})$, the code
computes the time-averaged **acoustic streaming velocity** $\langle\mathbf{u}\rangle$
via a Stokes flow driven by the Reynolds stress:

$$-\mu\,\nabla^2\langle\mathbf{u}\rangle + \nabla\langle p_2\rangle
  = -\rho\,\langle(\mathbf{v}_1 \cdot \nabla)\mathbf{v}_1\rangle$$

$$\nabla \cdot \langle\mathbf{u}\rangle = 0$$

where $\mathbf{v}_1 = \nabla p / (i\omega\rho)$ is the first-order acoustic
velocity (from the Helmholtz phasor) and the Reynolds stress forcing is
$\mathbf{f} = -\frac{1}{2}\text{Re}[\rho(\mathbf{v}_1 \cdot \nabla)\bar{\mathbf{v}}_1]$.

### Implementation

- Discretised with Taylor–Hood (P2–P1) mixed elements for velocity–pressure.
- Assembled using DOLFINx with complex-to-real projection of the forcing.
- Solved with MUMPS direct LU factorisation (fieldsplit Schur complement was
  found to fail at SI-scale magnitudes due to absolute tolerance issues).
- Pressure nullspace (constant mode) is removed explicitly.
- **Boundary conditions:** no-slip ($\mathbf{u} = 0$) on bottom and side
  walls; free-slip ($u_z = 0$) on the top surface.

### Caveats

- Streaming velocities are $O(10^{-3}\text{–}10^{-1})$ μm/s at typical
  transducer amplitudes (1–10 μm/s), consistent with Reynolds-stress scaling
  but difficult to validate without COMSOL cross-checks.
- The model neglects boundary-layer streaming (Schlichting streaming within
  the viscous boundary layer) and uses the outer-streaming approximation.

---

## 5  Particle Dynamics

### 5.1  Gor'kov Radiation Force

The time-averaged radiation force on a small compressible sphere is derived
from the Gor'kov potential:

$$U = \frac{4\pi a^3}{3}\left[
  f_1\frac{\langle|p|^2\rangle}{4\rho c^2}
  - f_2\frac{3\rho\langle|\mathbf{v}|^2\rangle}{8}
\right]$$

$$\mathbf{F}_\text{rad} = -\nabla U$$

where $a$ is the particle radius, and $f_1$, $f_2$ are the monopole and dipole
contrast factors (functions of particle and fluid properties).

### 5.2  Overdamped Trajectory Integration

The particle trajectory is integrated in the overdamped (Stokesian) regime:

$$\frac{d\mathbf{x}}{dt} = \mu_\text{Stokes}\,\mathbf{F}_\text{rad}
                           + \langle\mathbf{u}\rangle(\mathbf{x})$$

where $\mu_\text{Stokes} = 1/(6\pi\eta a)$ is the Stokes mobility.  The
streaming drag enters as a body-velocity correction.

### 5.3  Outputs

- **Trap stiffness:** curvature of the Gor'kov potential at lattice minima.
- **Barrier height:** potential difference between adjacent minimum and saddle
  point.
- **Safe transport speed:** maximum vortex-lens translation rate that does not
  lose the particle (estimated from barrier height and Stokes drag).
- **Selectivity ratio:** ratio of vortex extraction force to lattice restoring
  force.

---

## 6  Validation and Regression Tests

All tests are run with `micromamba run -n fenicsx python <script>`.

| Test | Script | What It Verifies |
|------|--------|------------------|
| Complex PETSc gate | `scripts/validation/test_env_complex_petsc.py` | PETSc has complex scalar type |
| Complex backend | `scripts/validation/test_complex_backend.py` | DOLFINx complex assembly works |
| 1D impedance | `scripts/validation/test_1d_impedance.py` | Robin BC `α = −iωρ/Z` gives \|R\| ≈ 0 for matched impedance |
| Energy balance | `scripts/validation/test_energy_balance.py` | Power in = power absorbed (to machine precision) |
| Petri-dish BCs | `scripts/validation/test_petri_dish_bcs.py` | Bottom segmentation; standing ~58 Pa; all modes nonzero |
| Vortex lens | `scripts/validation/validate_vortex_lens.py` | Pressure null at core; phase winds by 2πℓ |
| 2D Helmholtz | `scripts/validation/test_2d_helmholtz.py` | Convergence on circular domain |
| Streaming smoke | `scripts/validation/test_streaming_stokes_smoke.py` | Stokes solver converges; velocity nonzero |
| PML reflection | `scripts/validation/test_pml_reflection_fit.py` | \|R\| < 1 % for plane-wave incidence |
| Full suite | `scripts/validation/run_all_tests.py` | Runs all above in sequence |

---

## 7  How to Run

### 7.1  Environment Setup

The project requires a DOLFINx environment with **complex PETSc scalars**.
See [docs/LINUX_SETUP.md](docs/LINUX_SETUP.md) for detailed Linux instructions.

```bash
micromamba create -n fenicsx -f environment/complex-fenicsx.yml
micromamba activate fenicsx
pip install -e .
```

All solver commands must be prefixed with `micromamba run -n fenicsx` to
ensure the correct PETSc scalar type is used.

### 7.2  Validation Tests

```bash
# Core regression tests
micromamba run -n fenicsx python scripts/validation/test_1d_impedance.py
micromamba run -n fenicsx python scripts/validation/test_energy_balance.py
micromamba run -n fenicsx python scripts/validation/test_petri_dish_bcs.py

# Run the full validation suite
micromamba run -n fenicsx python scripts/validation/run_all_tests.py
```

### 7.3  Deliverable Scripts

These are the primary outputs.  All 3D exports go to the OneDrive folder.

```bash
# Vortex 3D Export — VTU primary (plastic lens, standing OFF)
micromamba run -n fenicsx python scripts/validation/export_vortex_3d.py

# Lens Propagation Diagnostics — z-stack, winding, core ratio, PML decay
micromamba run -n fenicsx python scripts/validation/diagnostics_lens_propagation.py

# Interaction Diagnostics — standing vs vortex vs combined
micromamba run -n fenicsx python scripts/validation/diagnostics_interaction.py

# Axicon Lens Demo — Bessel-beam vs plastic lens comparison
micromamba run -n fenicsx python scripts/experiments/run_axicon_lens_demo.py
```

### 7.4  Experiment Sweeps

```bash
# Phase 0 — 10×10 mm baseline with disc diameter sweep
micromamba run -n fenicsx python scripts/experiments/phase0_baseline_sweep.py

# Phase 1 — Transducer size & dish size architecture sweep
micromamba run -n fenicsx python scripts/experiments/phase1_sweep.py

# Phase 2 — Wall impedance sweep (lossy cavity)
micromamba run -n fenicsx python scripts/experiments/impedance_sweep.py

# Far-field PML demo (2 MHz vortex + standing)
micromamba run -n fenicsx python scripts/experiments/farfield_vortex_plus_standing.py
```

### 7.5  Repository Cleanup

```bash
# Preview cleanup (dry run — no files moved)
micromamba run -n fenicsx python scripts/maintenance/cleanup_repo.py --dry-run

# Execute cleanup (archives old results)
micromamba run -n fenicsx python scripts/maintenance/cleanup_repo.py
```

### 7.6  Configuration

All simulation parameters are controlled through `ShallowDishConfig`
(`src/acoustweezers/experiments/shallow_square_dish/config.py`).  Key fields:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `L` | 0.05 m | Lateral dish size |
| `H` | 0.005 m | Dish depth |
| `frequency_hz` | 500 kHz | Operating frequency |
| `vortex_velocity_amplitude` | 10 μm/s | Vortex transducer amplitude |
| `vortex_topological_charge` | 1 | Orbital angular momentum order |
| `vortex_aperture_radius` | 4 mm | Disc transducer radius |
| `standing_velocity_amplitude` | 1 μm/s | Standing-wave amplitude |
| `standing_phase_pattern` | `"antiphase"` | Phase relation between walls |
| `standing_axis` | `"x"` | Active wall pair(s) |
| `elements_per_wavelength` | 6 | Mesh resolution |
| `top_impedance_factor` | 0.001 | Z_top / Z_water |
| `wall_impedance_Zrel` | `None` | Relative wall impedance Z_wall/(ρc); `None` = rigid |
| `bottom_rigid_impedance_Zrel` | `None` | Z_rel on bottom rigid region; `None` = rigid |

#### Phase 0 Presets (10×10 mm Baseline)

Three presets lock the 10×10 mm testbed with varying disc diameter:

| Preset | L | D_disc | R_disc | Coverage |
|--------|---|--------|--------|----------|
| `L10_D02` | 10 mm | 2 mm | 1.0 mm | 3.1% |
| `L10_D03` | 10 mm | 3 mm | 1.5 mm | 7.1% |
| `L10_D04` | 10 mm | 4 mm | 2.0 mm | 12.6% |

All use: H=1 mm, f=500 kHz, 10 elem/λ, both-axis antiphase standing,
cosine-taper vortex, V_stand=V_vtx=10 μm/s.

Access via:
```python
from acoustweezers.experiments.shallow_square_dish.config import PHASE0_PRESETS
cfg = PHASE0_PRESETS["L10_D03"]()
```

---

## 8  Repository Structure

```
acousto-tweezers/
├── CHANGELOG.md              Current session changes
├── README.md                 This file
├── pyproject.toml            Package metadata & dependencies
├── environment.yml           Conda environment (real-scalar fallback)
│
├── docs/
│   ├── LINUX_SETUP.md            Linux setup guide (Bristol desktops)
│   ├── COMSOL_RECREATION_SPEC.md
│   └── validation.md
│
├── src/acoustweezers/
│   ├── experiments/
│   │   ├── shallow_square_dish/
│   │   │   ├── config.py          ShallowDishConfig + Phase 0 presets
│   │   │   ├── solve_pressure.py  Helmholtz solver + mesh + BCs
│   │   │   ├── streaming.py       Stokes streaming solver
│   │   │   ├── particles.py       Gor'kov + trajectory integration
│   │   │   └── export.py          VTU/XDMF export
│   │   └── farfield_petri_cuboid/
│   │       ├── config.py          FarFieldConfig (+axicon, fast_mode_config)
│   │       ├── mesh.py            Mesh with PML cell/facet tags
│   │       ├── solve_pressure.py  PML-Helmholtz (MUMPS, plastic/axicon/ideal)
│   │       └── post.py            Slicing, plotting, CSV, .npz export
│   ├── physics/
│   │   └── acoustics/
│   │       └── vortex_lens.py     Ideal + Plastic + Axicon lens models
│   ├── numerics/                  FEM assembly utilities
│   └── legacy/                    Archived older solver stacks
│
├── scripts/
│   ├── validation/
│   │   ├── export_vortex_3d.py           VTU primary export (→ OneDrive)
│   │   ├── diagnostics_lens_propagation.py  Z-stack + winding (→ OneDrive)
│   │   ├── diagnostics_interaction.py    3-case comparison (→ OneDrive)
│   │   ├── run_all_tests.py              Full test suite
│   │   ├── test_1d_impedance.py          Robin BC verification
│   │   ├── test_energy_balance.py        Power conservation
│   │   ├── test_petri_dish_bcs.py        Bottom segmentation
│   │   └── ...                           Other validation scripts
│   ├── experiments/
│   │   ├── run_axicon_lens_demo.py       Axicon vs plastic comparison (→ OneDrive)
│   │   ├── farfield_vortex_plus_standing.py  Far-field PML demo
│   │   ├── phase0_baseline_sweep.py      Phase 0 disc-diameter sweep
│   │   ├── phase1_sweep.py               Phase 1 architecture sweep
│   │   ├── impedance_sweep.py            Phase 2 impedance sweep
│   │   └── ...                           Other experiment pipelines
│   ├── maintenance/
│   │   └── cleanup_repo.py               Archive old results + report
│   └── analysis/                         Postprocessing, plotting
│
├── results/                              Lightweight outputs (PNG/CSV/JSON)
│   ├── *_latest -> *_<timestamp>/        Symlinks to latest runs
│   └── ...
│
├── archive/                              Quarantined old docs, scripts, results
├── docker/Dockerfile
└── environment/
    ├── complex-fenicsx.yml               Complex PETSc environment spec
    └── setup_env_complex.sh
```

The active solvers live in `src/acoustweezers/experiments/shallow_square_dish/`
(shallow cavity, 500 kHz) and `src/acoustweezers/experiments/farfield_petri_cuboid/`
(far-field PML, 2 MHz).  Legacy code under `src/acoustweezers/legacy/` is
retained for reference but is not imported by any current script.

---

## 9  Far-Field Petri Cuboid (2 MHz PML Demo)

### 9.1  Domain Layout

A taller cuboid domain (default 6×6×5.0085 mm) models upward propagation from a
bottom-mounted vortex disc through a water under-bath ($H_\text{under}$ = 3 mm)
into a thin petri slab ($H_\text{top}$ = 2.0085 mm, tuned to the m = 14
quarter-wave resonance) at the top.  Perfectly Matched Layers (PML) absorb
outgoing waves on the bottom face (outside the disc column) and on the four
lateral faces **in the water-bath region only** ($z < H_\text{under}$).  In the
petri slab ($z \ge H_\text{under}$), the lateral faces are rigid walls (or
standing-wave transducers) with no PML absorption — this is essential because
the standing-wave BCs sit on the mesh boundary and must not be damped.  The top
face carries a water–air impedance Robin BC ($Z_\text{air} \approx 413$ Pa·s/m).

> **Critical fix (2026-02-24):** Prior to this date, lateral PML extended the
> full domain height, absorbing standing waves before they reached the physical
> interior.  See CHANGELOG for details.

### 9.2  PML Formulation

Complex coordinate stretching: $\tilde{x}_\alpha = x_\alpha + \frac{i}{\omega}\int_0^{x_\alpha} \sigma_\alpha(s)\,ds$.
The absorption profile $\sigma_\alpha$ uses a polynomial ramp of degree 2
with $\sigma_\text{max} = 5\omega$.  PML thickness defaults to 1 wavelength
(λ ≈ 0.742 mm at 2 MHz in water).

The stretch factors $s_\alpha = 1 + i\sigma_\alpha / \omega$ and the PML metric
tensor $\Lambda_x = s_y s_z / s_x$ and Jacobian $J = s_x s_y s_z$ are built
as **UFL expressions** from the σ `fem.Function` objects.  FFCx evaluates these
rational expressions at quadrature points, avoiding P2 projection error that
would arise from DOF-array arithmetic on the products of P2 functions.

**Z-filter (2026-02-24 fix):** The lateral absorption profiles σ_x and σ_y are
set to **zero** for $z \ge H_\text{under}$.  This confines the lateral PML to
the water-bath region and leaves the petri slab as a pure rigid-walled cavity
where standing waves can resonate freely.  The bottom PML (σ_z) is unchanged.

### 9.3  Boundary Conditions

| Boundary | BC |
|----------|----|
| Bottom disc ($z=0$, $r \le R_\text{disc}$) | Neumann: $\partial_n p = -i\omega\rho\,v_n(\mathbf{x})$ (plastic lens or ideal vortex) |
| Bottom outside disc | PML absorbing layer (σ_z ramp) |
| Lateral faces, $z < H_\text{under}$ (bath) | PML absorbing layer (σ_x, σ_y ramps) |
| Lateral faces, $z \ge H_\text{under}$ (petri) | Rigid walls or standing-wave Neumann transducers |
| Top ($z = H$) | Robin: $\partial_n p = +i\omega\rho\,p / Z_\text{top}$ (i.e. $+ik/Z_\text{rel}$) |

### 9.4  Plastic Lens Vortex Drive (Day 2)

The default bottom-disc drive models a **fabricable plastic spiral-phase lens**
that encodes both vortex and focusing phases via thickness variation:

$$\varphi_\mathrm{target}(x,y) = \ell\,\theta + k_w\left(\sqrt{(x - x_f)^2 + (y - y_f)^2 + f^2} - f\right)$$

$$\varphi_\mathrm{plastic} = \mathrm{mod}(\varphi_\mathrm{target},\, 2\pi)$$

The boundary velocity is $v_n(x,y) = V_0\,A(r)\,\exp(i\varphi_\mathrm{plastic})$,
where $A(r)$ is one of `cosine_taper`, `tukey`, or `uniform` apodization.

Physical thickness for fabrication:

$$t(x,y) = t_0 + \frac{\mathrm{mod}(\varphi_\mathrm{target},\,2\pi)}{k_\mathrm{lens} - k_\mathrm{water}}$$

Configuration fields (in `FarFieldConfig`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lens_drive` | `"plastic"` | `"plastic"`, `"axicon"`, or `"ideal"` |
| `lens_l` | 1 | Topological charge $\ell$ |
| `lens_focal_length` | 2 mm | Focusing focal length $f$ (plastic only) |
| `lens_focus_offset_x` | 0.0 | Off-axis focus $x_f$ (plastic only) |
| `lens_focus_offset_y` | 0.0 | Off-axis focus $y_f$ (plastic only) |
| `lens_c_lens` | 2700 m/s | Speed of sound in plastic |
| `lens_axicon_angle_deg` | 15.0 | Axicon half-angle $\alpha$ in degrees (axicon only) |
| `lens_apodization` | `"cosine_taper"` | Amplitude taper profile |
| `lens_apodization_strength` | 1.0 | Tukey taper parameter |

The physics module lives in `src/acoustweezers/physics/acoustics/vortex_lens.py`
(`PlasticLensConfig`, `AxiconLensConfig`, `create_plastic_lens_drive()`,
`create_axicon_lens_drive()`).

### 9.5  Axicon (Bessel-like) Vortex Lens

The axicon lens imparts a radial phase:

$$\varphi_\mathrm{axicon}(r,\theta) = \ell\,\theta + k_r\,r, \qquad k_r = k_0\sin\alpha$$

where $\alpha$ is the axicon half-angle.  This produces a non-diffracting
Bessel-like vortex beam whose core diameter is determined by $\alpha$ rather
than focal geometry.  A larger $\alpha$ gives a tighter core.

Run the comparison demo:

```bash
micromamba run -n fenicsx python scripts/experiments/run_axicon_lens_demo.py
```

Output goes to `~/OneDrive - .../AxiconLensDemo/` with VTU fields,
radial-profile comparison, and per-case XY/XZ PNG slices.

### 9.6  Running

```bash
# Default: plastic lens drive (focused vortex)
micromamba run -n fenicsx python scripts/experiments/farfield_vortex_plus_standing.py

# Fallback: legacy ideal vortex (pure exp(i ℓ θ), no focusing)
micromamba run -n fenicsx python scripts/experiments/farfield_vortex_plus_standing.py --ideal

# Fast mode: 4 elem/λ for quick qualitative checks
micromamba run -n fenicsx python scripts/experiments/farfield_vortex_plus_standing.py --fast

# Custom solver parameters
micromamba run -n fenicsx python scripts/experiments/farfield_vortex_plus_standing.py \
    --rtol 1e-5 --restart 300 --maxit 8000
```

**CLI arguments** (added 2026-02-17):

| Flag | Default | Description |
|------|---------|-------------|
| `--ideal` | off | Use ideal vortex instead of plastic lens |
| `--fast` | off | 4 elem/λ (qualitative only, ~5× faster) |
| `--rtol` | 1e-4 | GMRES relative tolerance |
| `--restart` | 200 | GMRES subspace restart |
| `--maxit` | 5000 | Maximum GMRES iterations |

Outputs land in `results/farfield_vortex_standing_<timestamp>/` (symlinked as
`results/farfield_latest`).  Generates:

- Diagnostic PNGs: XY/XZ slices, centerline, energy, disk drive patterns
  (disk_amplitude.png, disk_phase.png, disk_real.png, disk_imag.png)
- `csv/summary.csv` — PML vs rigid comparison metrics
- `config.json` — full configuration dump

### 9.7  PML Diagnostic Findings (Part 1)

Systematic diagnostics (`scripts/experiments/farfield_part1_diagnostics.py`)
confirmed the PML implementation is correct:

| Suspect | Test | Verdict |
|---------|------|---------|
| **S1**: PML leaks into top | σ_z = 0 at all DOFs near top face and in petri slab; only σ_x, σ_y nonzero (in lateral PML bands) | **PASS** |
| **S2**: Disk source damped by PML | σ_z = 0 for all 5530 DOFs in disk column ($r \le R$, $z < t_\mathrm{pml}$); σ_z active only outside disk | **PASS** |
| **S3**: GMRES convergence | GMRES(30)+ILU stagnated at residual ~124 after 4800 iters.  GMRES(200)+ILU converges: rtol=1e-3 → 1e-5 gives identical max\|p\| to 4 d.p. (18.0069 Pa) | **FIXED** |
| **S4**: Top BC toggleable | Impedance vs pressure-release gives only 0.02% difference — PML domination makes top BC nearly irrelevant | **PASS** (by design) |

**S3 critical fix**: GMRES restart increased 30 → 200, default rtol changed to
1e-4.  The complex-indefinite Helmholtz+PML system requires a large Krylov
subspace.  rtol=1e-3 to 1e-5 tolerance sweep shows solutions are converged
(0.00% variation in max|p| and centerline max).

**S4 note**: The near-identical impedance vs pressure-release results confirm
that the PML + water column geometry dominates the physics — by the time acoustic
energy reaches the top face, it has already been absorbed or spread laterally.
The top BC choice is cosmetic at the current domain depth (3 mm underbath).

### 9.8  Memory Notes

At 2 MHz (λ = 0.742 mm) the mesh is dense.  On a 7.5 GB machine:

- **5 elem/λ** (348K DOFs): OOM-kills MUMPS.  Use GMRES(200)+ILU if attempted.
- **3 elem/λ** (79K DOFs): Works with MUMPS direct.  This is the default for
  all deliverable scripts (they auto-fallback from 5 → 3 if OOM).
- The driver scripts explicitly free memory with `gc.collect()` between cases.

### 9.9  Lens Presets

Three presets cover common use cases.  Access via:

```python
from acoustweezers.physics.acoustics.vortex_lens import LENS_PRESETS
lens_cfg = LENS_PRESETS["B"]()  # focused preset
```

| Preset | l | f (mm) | Offset (x, y) mm | Description |
|--------|---|--------|-------------------|-------------|
| A | 1 | 50 | (0, 0) | Weak focus / pure vortex |
| B | 1 | 10 | (0, 0) | Strong focus |
| C | 1 | 10 | (0.2, 0) | Off-axis focus (biased transport) |

Gallery: `python scripts/experiments/farfield_plastic_lens_gallery.py`
→ `results/farfield_lens_gallery_latest/`

### 9.10  Diagnostic Scripts

| Script | Purpose | Key Result |
|--------|---------|------------|
| `farfield_pml_operator_check.py` | PML vs rigid comparison (A2) | 98.4% diff in max\|p\|, 7 vs 5000 iters |
| `farfield_s4_topbc_sensitivity.py` | Impedance vs Dirichlet sweep (B1) | < 0.4% diff with PML on |
| `farfield_plastic_lens_gallery.py` | Lens preset visual gallery (D3) | Thickness [0.200, 1.848] mm |
| `farfield_plastic_vs_ideal.py` | Plastic vs ideal comparison (E1) | ~0% diff at 4 elem/λ |

All scripts are in `scripts/experiments/` and output timestamped results with
a `_latest` symlink.

### 9.11  Gallery Script (Corrected Model)

```bash
micromamba run -n acousto-complex python scripts/experiments/fixed_vortex_gallery.py
```

Generates a comprehensive 127-PNG gallery for the corrected far-field model
using the `CORRECTED_PRESET` (H_top = 2.0085 mm, f = 2 mm, plastic lens).
For each case (standing, vortex, combined) it produces:

- XY and XZ slices (linear + log scale)
- Centerline pressure profiles
- 3-way comparison panels
- Z-progression panels (9 heights through the petri slab)
- Physics audit bar chart (physical vs PML max|p|)

Key features: physical-domain-only interpolation (PML DOFs filtered out),
per-case colorscales, and automatic PML/physical ratio diagnostics.

Results: `results/fixed_gallery_<timestamp>/`

**Current best results (2026-02-24):** standing max|p| = 59.9 Pa,
vortex max|p| = 1.3 Pa, combined max|p| = 59.7 Pa — all in the physical domain.

---

## 10  Roadmap

### Phase 0 — 10×10 mm Baseline + Disc Diameter Sweep

| ID | Task | Status |
|----|------|--------|
| P0.1 | Define L10_D02/D03/D04 presets (L=10 mm, D=2/3/4 mm) | **Done** |
| P0.2 | Sweep script: A/B/C + amplitude sweep per preset | **Done** |
| P0.3 | Comparison table: trap count, authority, selectivity | **Done** |
| P0.4 | Acceptance: all discs < 15 % coverage, stable runtime | **Done** |

Results: `results/phase0_baseline_latest/`

### Phase 1 — Realistic Lens Boundary Model

| ID | Task | Status |
|----|------|--------|
| P1.1 | Plastic lens module (`vortex_lens.py`) with `PlasticLensConfig` + `create_plastic_lens_drive()` | **Done** |
| P1.2 | Amplitude + phase maps (`disk_amplitude.png`, `disk_phase.png`, `disk_real.png`, `disk_imag.png`) | **Done** |
| P1.3 | Regression: `--ideal` flag reproduces legacy ideal vortex baseline | **Done** |
| P1.4 | PML S1–S4 diagnostics (`farfield_part1_diagnostics.py`) | **Done** |
| P1.5 | GMRES stagnation fix: restart 30→200, rtol 1e-8→1e-4 | **Done** |

### Phase 2 — Wall + Bottom Impedance Sweep (Lossy Cavity)

| ID | Task | Status |
|----|------|--------|
| P2.1 | Robin impedance terms on side walls (solver change) | **Done** |
| P2.2 | Impedance sweep script (`impedance_sweep.py`) | **Done** |
| P2.3 | Comparison: |p|, trap depth, selectivity vs Z_rel | **Done** |
| P2.4 | Confirm: resonance peaks soften as walls become less rigid | **Done** |

Results: `results/phase2_impedance_latest/`

Key finding: as Z_rel decreases from ∞ → 1, max|p| drops 81.7 → 57.0 Pa,
selectivity improves 0.73 → 1.60, confirming lossy walls reduce global
recirculation while preserving local vortex authority.

### Phase 3 — Frequency Separation (Two-Frequency Operation)

| ID | Task | Status |
|----|------|--------|
| P3.1 | Dual Helmholtz solves (f_s for standing, f_v for vortex) | Planned |
| P3.2 | Gor'kov superposition (no cross terms) | Planned |
| P3.3 | Demo: lattice stable + local vortex transport | Planned |

### Phase 4 — Integration (Best Available Model)

| ID | Task | Status |
|----|------|--------|
| P4.1 | Combine: realistic lens + impedance + dual-frequency | Planned |
| P4.2 | Reference config + reproducible report | Planned |

### Completed (earlier)

- COMSOL cross-validation (Attempt 4) — pressure fields for Cases A/B/C match
  expected physics (disc as pure Neumann source, no impedance absorption)
- Phase 1 architecture sweep (5/10/20 mm piezo, 10/30 mm dish) — see
  `scripts/experiments/phase1_sweep.py`
- **Far-field PML demo (2 MHz)** — upward-propagating vortex + standing waves
  with PML absorption; see Section 9 and `results/farfield_latest/`
- **Lateral PML z-filter fix (2026-02-24)** — standing waves now reach the
  physical interior; max|p| increased from 0.77 → 59.9 Pa
- **Resonance optimisation** — H_top = 2.0085 mm (m = 14 quarter-wave),
  f = 2 mm focal length (optimal vortex ring size ≈ 1.2λ)

---

## License

See `LICENSE` file.
