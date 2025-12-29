
Acoustic Tweezers Modelling Engine

Linear acoustics → radiation force → trap dynamics → optimisation

1. Purpose of this repository
This repository implements a COMSOL-independent modelling engine for acoustic tweezers and particle manipulation. The goal is to build a predictive, optimisation-ready model that maps actuation parameters to acoustic fields, radiation forces, and resulting particle motion. The emphasis is on linear acoustic field modelling, physically justified radiation force modelling, and the extraction of linearised trap dynamics (including stiffness matrices and characteristic time constants), with a view toward optimisation, reduced-order modelling, and control readiness. COMSOL is used only as a validation reference and not as the core solver, ensuring that the physics and numerical methods remain transparent, extensible, and suitable for integration with optimisation, machine learning, and control algorithms.

2. What the final product is
The final product is not just a numerical solver, but a modelling platform intended to support the wider project. At completion, the codebase will allow a user to define geometries and actuation schemes, compute acoustic pressure and velocity fields, derive radiation force fields acting on particles, automatically identify trapping points, extract linearised trap stiffness and dynamics, simulate particle trajectories, optimise actuation parameters to meet design objectives, and interactively explore system behaviour in real time. Beyond offline analysis, the modelling engine is explicitly designed to operate as part of a robotic manipulation system. In the intended workflow, particle positions are observed using a camera-based perception system, and actuation parameters are updated in closed loop to achieve target configurations and assembled structures. The model therefore provides a predictive mapping from actuation to particle motion, enabling model-based control, online calibration, and autonomous acoustic assembly.

3. Architectural overview (conceptual)
The repository is structured as a layered modelling pipeline. Core abstractions define what is being modelled rather than how: geometry describes the domain (for example, a 2D rectangular microchannel), actuation specifies transducer parameters such as frequency, phase, amplitude, and boundary location, the field represents acoustic pressure and velocity fields, the force model derives radiation forces from the field, and the linear model captures local linearisations of forces around traps in the form of stiffness matrices and time constants. These abstractions allow multiple solvers and optimisation methods to plug into a common framework without altering downstream components.

Forward solvers replace COMSOL as the primary modelling tool. A fast finite-difference or spectral solver is used for rectangular domains, providing extremely rapid evaluations suitable for optimisation, reduced-order modelling, and interactive use. A finite-element solver based on FEniCS supports arbitrary geometries and higher-fidelity simulations for validation and realism. COMSOL is optionally used as a reference to validate pressure fields and trap stiffness, but is never included inside optimisation loops. All solvers return the same Field object, ensuring solver-independent downstream analysis.

4. Physics-to-mechanics mapping (core modelling contribution)
The scientifically central contribution of the project lies in the mapping from acoustic physics to particle mechanics. A linear acoustic field is solved using the Helmholtz equation, from which the Gor’kov radiation potential is computed. The radiation force field is obtained as the gradient of this potential, and equilibrium points (traps) are detected automatically. Forces are then linearised about these traps to obtain stiffness matrices, stable and unstable directions, and characteristic time constants. This explicitly separates the nonlinear physics of acoustic radiation forces from the locally linear particle dynamics that are most relevant for control and optimisation. Interpreting the Gor’kov potential as an energy landscape also enables powerful geometric intuition: stable traps correspond to local minima, unstable equilibria appear as saddle points, and stiffness eigenvalues correspond to local curvatures. Visualising this potential as a three-dimensional surface therefore provides an intuitive way to understand trapping, stability, and control authority, and directly supports the design of gradient-based and model-predictive control strategies.

5. Particle dynamics
Particle motion is modelled using overdamped dynamics appropriate for microscale acoustofluidic systems, where Stokes drag dominates inertia. This produces particle trajectories, capture behaviour, and convergence rates toward traps. Optional extensions, such as Brownian diffusion, are identified as future work but are not required for the initial control-focused modelling framework.

6. Optimisation and reduced-order modelling
An advanced optimisation layer makes the model practical for design and control. Bayesian optimisation is used to automatically tune phases and amplitudes to achieve objectives such as trap placement, stiffness maximisation, and robustness to disturbances. Gradient-based optimisation using adjoint methods or automatic differentiation is identified as a stretch goal for high-dimensional parameter spaces. Reduced-order models based on snapshot generation and POD/PCA are used to enable near real-time evaluation of fields and forces, which is essential for interactive visualisation and closed-loop control.

7. Interfaces
The system is designed to be used through multiple interfaces. A Python API allows integration with control algorithms, tracking pipelines, and experimental software. A command-line interface supports batch runs, parameter sweeps, and reproducible figure generation. An interactive user interface, implemented using Streamlit, provides real-time sliders for phase, amplitude, and frequency, along with live visualisation of fields, forces, trajectories, and potential surfaces, supporting both development and demonstration.

8. Why this approach (and why not COMSOL-only)
Using COMSOL alone obscures numerical details, limits optimisation and machine-learning integration, and reduces learning and originality. By contrast, this repository owns the physics, enables optimisation and control, allows deep understanding and systematic validation, and still leverages COMSOL where it is strongest as a reference solution. This hybrid approach mirrors real research workflows and supports the ultimate goal of autonomous, robot-controlled acoustic manipulation.

9. What success looks like
At completion, the project will deliver a clean, modular modelling codebase with validated physics, an optimisation-ready architecture, control-ready linear models, intuitive three-dimensional visualisations of trapping landscapes, and demonstrable interactivity. Crucially, it will provide a clear pathway to experimental integration, supporting camera-based perception and closed-loop robotic control for the autonomous assembly of particle structures using acoustic tweezers. This represents a full modelling-to-control workflow rather than a standalone numerical experiment.

# =========================
# Repo scaffold
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
