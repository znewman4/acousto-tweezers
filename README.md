
Acoustic Tweezers Modelling Engine

Linear acoustics → radiation force → trap dynamics → optimisation

1. Purpose of this repository

This repository implements a COMSOL-independent modelling engine for acoustic tweezers and particle manipulation.

The goal is to build a predictive, optimisation-ready model that maps:

actuation parameters → acoustic field → radiation force → particle motion

with an emphasis on:

linear acoustic field modelling

physically justified radiation force modelling

linearised trap dynamics (stiffness, time constants)

optimisation, reduced-order modelling, and control-readiness

COMSOL is used only as a validation reference, not as the core solver.

2. What the final product is

The final product is not just a solver, but a modelling platform that can be used by the rest of the project.

At the end of development, the codebase will allow a user to:

define a geometry and actuation scheme

compute the acoustic pressure field

compute the radiation force field on particles

automatically identify trapping points

extract linearised trap stiffness and dynamics

simulate particle trajectories

optimise actuation parameters to achieve design goals

interactively explore the system in real time

This makes the model useful for:

design decisions

control algorithm development

experimental parameter selection

demonstration and visualisation

3. Architectural overview (conceptual)

The repository is structured as a layered modelling pipeline.

3.1 Core abstractions (physics-agnostic)

These define what is being modelled, not how.

Geometry
Domain description (e.g. 2D rectangular microchannel)

Actuation
Transducer parameters (frequency, phase, amplitude, boundary location)

Field
Acoustic pressure and velocity fields

Force model
Radiation force derived from the field

Linear model
Local linearisation of forces around traps (stiffness matrices, time constants)

These abstractions allow multiple solvers and optimisation methods to plug into the same framework.

3.2 Forward solvers (replace COMSOL)

Multiple backends are supported:

Fast solver (finite difference / spectral)

Rectangular domains

Extremely fast

Used for optimisation, ROM, and UI

FEM solver (FEniCS)

Arbitrary geometry

Higher fidelity

Used for validation and realism

COMSOL reference (optional)

Used only to validate pressure fields and trap stiffness

Never used inside optimisation loops

All solvers return the same Field object, ensuring downstream code is solver-independent.

3.3 Physics-to-mechanics mapping (core modelling contribution)

This is the scientifically central part of the project.

Linear acoustic field is solved (Helmholtz equation)

Gor’kov radiation potential is computed

Radiation force field is obtained

Equilibrium points (traps) are detected

Forces are linearised around traps, yielding:

stiffness matrices

stable/unstable directions

characteristic time constants

This step explicitly separates:

nonlinear radiation force physics
from

locally linear particle dynamics, which are the focus of control and optimisation

3.4 Particle dynamics

Particle motion is modelled using:

overdamped dynamics (Stokes drag dominated)

optional Brownian diffusion (future extension)

This produces:

trajectories

capture behaviour

convergence rates

3.5 Optimisation and reduced-order modelling (advanced layer)

This layer makes the model useful and fast.

Bayesian optimisation

automatically tunes phases/amplitudes

objectives: trap location, stiffness, robustness

Adjoint / gradient-based optimisation (stretch)

gradients via fenics-adjoint or autodiff

efficient high-dimensional optimisation

Reduced-order models (ROMs)

snapshot-based POD / PCA

near real-time evaluation

enables interactive UI and fast optimisation

3.6 Interfaces

The system can be used in three ways:

Python API
Used by other project components (control, tracking, experiments)

CLI
Batch runs, parameter sweeps, reproducible figures

Interactive UI (Streamlit)
Real-time sliders for phase, amplitude, frequency
Live visualisation of fields, forces, and trajectories

4. Why this approach (and why not COMSOL-only)

Using COMSOL alone:

hides numerical details

limits optimisation and ML integration

reduces learning and originality

This repository:

owns the physics

enables optimisation and control

allows deep understanding and validation

still uses COMSOL where it is strongest (reference solutions)

This hybrid approach mirrors real research workflows.

5. Chronological development game plan

The project is built in strict stages, to avoid over-complexity.

Phase 1 — Minimal end-to-end model

Goal: A complete pipeline exists.

2D rectangular domain

Finite-difference Helmholtz solver

Single boundary transducer

Field → radiation force → trap detection

Linearised stiffness and time constant

Simple particle trajectory simulation

Exit condition:
The model produces traps and predicts convergence dynamics.

Phase 2 — Physics correctness & validation

Goal: Trust the results.

Analytical validation (1D standing wave)

Symmetry and conservation tests

Grid refinement checks

Direct comparison with COMSOL for:

pressure magnitude

trap locations

stiffness values

Exit condition:
First-order agreement with COMSOL and theory.

Phase 3 — FEM backend

Goal: Geometry realism.

Implement Helmholtz FEM solver (FEniCS)

Same interface as FD solver

Mesh refinement studies

Exit condition:
FD and FEM solvers produce consistent trap predictions.

Phase 4 — Linear dynamics extraction

Goal: Control-ready modelling.

Automated Jacobian / stiffness extraction

Eigenvalue analysis

Time constants and stability classification

Exit condition:
Local linear state models are available for any trap.

Phase 5 — Bayesian optimisation

Goal: Automated design.

Gaussian process surrogate

Optimisation of phases/amplitudes

Objectives:

place trap at target location

maximise stiffness

minimise off-target forces

Exit condition:
Trap design works without manual tuning.

Phase 6 — Reduced-order modelling

Goal: Speed and interactivity.

Snapshot generation

POD / PCA basis

Fast surrogate evaluation

Exit condition:
Near real-time field and force updates.

Phase 7 — Adjoint / differentiable optimisation (stretch)

Goal: Demonstrate depth.

fenics-adjoint or autodiff-based gradients

Gradient-based optimisation outperforming BO

Exit condition:
Sensitivity-based optimisation demonstrated on at least one case.

Phase 8 — UI & integration

Goal: Communication and impact.

Interactive dashboard

Live parameter tuning

Trajectory previews

Exit condition:
System can be demonstrated live.

6. What success looks like

At completion, the project will have:

a clean, modular modelling codebase

validated physics

optimisation-ready architecture

control-ready linear models

demonstrable interactivity

clear pathways to experimental integration

This represents a full modelling workflow, not just a numerical experiment.


# =========================
# Repo scaffold (copy/paste)
# =========================
acousto-tweezers/
  README.md
  LICENSE
  .gitignore
  pyproject.toml
  Makefile
  configs/
    base.yaml
    cases/
      case_rect_fd.yaml
      case_rect_fem.yaml
  examples/
    00_quickstart.ipynb
    01_validate_1d_standing_wave.ipynb
    02_trap_stiffness_demo.ipynb
    03_bayesopt_trap_target.ipynb
    04_rom_realtime_demo.ipynb
  scripts/
    export_figures.py
    run_case.py
    run_sweep.py
    run_bayesopt.py
    build_rom.py
  src/
    acousto/
      __init__.py
      api.py
      cli.py
      logging.py
      types.py
      utils/
        __init__.py
        units.py
        numerics.py
        grid.py
      geometry/
        __init__.py
        primitives.py
        boundary_tags.py
        gmsh_tools.py
      solvers/
        __init__.py
        base.py
        fd_helmholtz.py
        fem_helmholtz_fenics.py
        comsol_reference.py
      acoustics/
        __init__.py
        field.py
        relations.py
      force/
        __init__.py
        gorkov.py
        radiation_force.py
      analysis/
        __init__.py
        traps.py
        linearise.py
        metrics.py
        validation.py
      dynamics/
        __init__.py
        overdamped.py
        brownian.py
        integrators.py
      optim/
        __init__.py
        bayesopt.py
        objectives.py
        constraints.py
        adjoint_fenics.py
      rom/
        __init__.py
        snapshots.py
        pod.py
        surrogate.py
      ui/
        __init__.py
        streamlit_app.py
  tests/
    conftest.py
    test_units.py
    test_fd_helmholtz_1d.py
    test_symmetry_force.py
    test_trap_linearisation.py
    test_convergence_fd.py
  docs/
    architecture.md
    methodology.md
    roadmap.md
    references.md
