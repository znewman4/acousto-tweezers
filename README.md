# Investigation and Development of a Low-Cost Acoustic Tweezer System

This repository contains the modelling code, validation studies, and curated
results used for the 2025/26 final MEng Major Project 1 report:



The project investigates a low-cost acoustic tweezer platform that combines a
bulk standing-wave trap lattice with a robotically translated acoustic lens.
The aim is to hold many microparticles at stable positions while selectively
manipulating and merging one target particle in an open dish environment.

## Project Threads

The report is organised around five objectives. The repository now follows the
same structure.

| Report thread | What is in this repository |
| --- | --- |
| Physics characterisation | Linear acoustics, Gorkov force, drag and secondary-effect assumptions in `src/acoustweezers/physics/` |
| Computational framework | FEniCSx finite-element standing-wave model, ASM beam/lens propagation, FEM-ASM superposition |
| Manipulation strategy evaluation | Vortex phase-offset control, C-shape lens studies, Monte Carlo robustness analysis |
| Experimental platform | Notes and manufacturing outputs for the acoustic lens and translated lens platform |
| Experimental validation | COMSOL comparison outputs and experimental-result pointers |

## Key Findings Reflected in the Report

- **FZP vortex lens:** produced an on-axis vortex null with a 0.957 mm ring
  radius. The modelled peak Gorkov force was 2759 pN, giving a 3.3x trapping
  margin at 1 mm/s for 65 um radius polystyrene particles.
- **C-shape hologram lens:** IASA reduced NMSE from 0.070 at iteration 1 to
  0.048 at iteration 100, a 31 percent improvement. The modelled peak Gorkov
  force was 3346.59 pN, giving a 2.73x trapping margin at 1 mm/s.
- **Petri-dish transmission:** the simplified dish model gives an intensity
  transmission coefficient of 0.87, reducing Gorkov force by about 13 percent.
- **Controllable standing waves:** frequency modulation provides long-range
  translation but poor stopping precision; phase modulation provides precise
  positioning over about half a wavelength. This motivated the final
  phase-offset concept.
- **Integrated FEM-ASM model:** the final simulation domain is a 6 x 6 x 7 mm
  water cuboid at 2.15 MHz. A 5.75 elements-per-wavelength mesh was selected
  as the practical standing-wave resolution.
- **Vortex phase-offset merging:** MPC-controlled relative phase offset lets a
  particle cross the vortex boundary rather than being repelled by it.
  Robustness tests reported 67 percent success over 100 random starting
  positions, and a noise study dropped from about 75 percent success at 0
  percent noise to about 44 percent at maximum tested noise.
- **Experimental status:** the low-cost experimental platform, standing-wave
  chamber, translated lens assembly, FZP lens and C-shape lens were produced.
  FZP/C-shape lens behaviour was tested with cinnamon and fluorescent
  polystyrene particles. Vortex/standing-wave phase-offset merging remains a
  computational candidate because matched-frequency transducers were not
  available for experimental validation.

## Repository Map

```text
.
|-- README.md
|-- pyproject.toml
|-- report/
|   |-- Final_Major_Project_1_Report.tex
|   `-- README.md
|-- src/
|   |-- acoustweezers/          # current simulation package
|   `-- acousto/                # legacy visualisation helpers still used by old scripts
|-- scripts/
|   |-- dev/                    # report-generation and exploratory study scripts
|   |-- deliverables/           # scripted result builders
|   |-- validation/             # smoke, physics and numerical validation checks
|   |-- analysis/               # COMSOL/debug analysis utilities
|   `-- lib/                    # shared script utilities
|-- results/
|   |-- README.md
|   |-- report/                 # curated report-facing figures and study outputs
|   |-- comsol_comparison/      # current COMSOL comparison outputs
|   |-- fem_standing_wave_cache/# cached FEM standing-wave fields
|   `-- archive_studies/        # older exploratory or superseded result runs
|-- docs/                       # setup, validation and modelling notes
|-- configs/                    # reusable simulation case definitions
|-- environment/                # reproducible environment files
|-- docker/
|-- tests/
`-- archive/                    # old logs, reviews, duplicate snapshots and failed attempts
```

## Where To Look For Report Evidence

| Question | Start here |
| --- | --- |
| What exactly was submitted as the report? | `report/Final_Major_Project_1_Report.tex` |
| Which figures/results support the report story? | `results/report/README.md` |
| Where are the C-shape and lens-design studies? | `results/report/lens_design/` |
| Where are the FEM-ASM, mesh and phase-control studies? | `results/report/integrated_model/` |
| Where are the COMSOL validation outputs? | `results/comsol_comparison/` |
| Where did old logs, review notes and duplicate deliverables go? | `archive/README.md` |

## Environment

The project uses Python 3.10+ with FEniCSx/DOLFINx for finite-element solves.
The main project environment is:

```bash
micromamba create -n fenicsx -f environment/complex-fenicsx.yml
micromamba activate fenicsx
pip install -e .
```

Older environment snapshots are kept in `environment/legacy/` for provenance.

## Useful Commands

Verify the package imports:

```bash
python -c "import acoustweezers; print(acoustweezers.__version__)"
```

Run a lightweight validation check:

```bash
python -c "from petsc4py import PETSc; assert PETSc.ScalarType.__name__ == 'complex128'"
python scripts/validation/test_1d_impedance.py
```

The FEniCSx validation scripts require a complex PETSc build. If the scalar
check reports `float64`, activate the complex FEniCSx environment first.

Run the validation collection after confirming complex PETSc:

```bash
python scripts/validation/run_all_tests.py
```

## Notes On Large Outputs

`results/` is intentionally organised for navigation, not for a small source
checkout. Heavy field files, caches and older exploratory outputs are separated
from the report-facing results. The curated report results live in
`results/report/`; old or superseded runs live in `results/archive_studies/` or
`archive/`.
