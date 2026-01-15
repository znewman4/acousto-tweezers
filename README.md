# ============================================================
# Acoustic Tweezers: Modelling + Control (Robot-Moved Transducers)
# ============================================================

This repository is a COMSOL-independent modelling and control engine for robotic acoustic tweezers. Transducers are not fixed: they are treated as control inputs and are intended to be moved by robots to reshape the acoustic field in real time. The core loop is: actuation parameters → acoustic field → Gor’kov potential → radiation force → overdamped particle motion → visualisation and logging.


# ============================================================
# What we have right now (working end-to-end)
# ============================================================

The codebase can run repeated fast forward solves of a 2.5D forced Helmholtz model on a planar domain, where moving transducers are represented as spatially localised velocity boundary sources. From each solved field, the Gor’kov radiation potential and radiation force are computed, and the particle is advanced under overdamped dynamics using interpolated forces at the particle position. The system supports trap finding and stiffness extraction, but control does not rely on trap stability and can move even through unstable regions.

A complete path-following demo exists in scripts/demo_surf_greedy.py that produces a reproducible “control run” with outputs saved to results/demo_surf_greedy/run_YYYYMMDD_HHMMSS. Each run produces an animated GIF, a summary plot, a step-by-step CSV log, and a JSON summary.


# ============================================================
# The baseline controller that proves the physics can steer particles
# ============================================================

The baseline controller is a truth-model greedy “surf controller”. At every timestep it enumerates a discrete set of macro actions. Each macro action modifies actuation parameters (transducer positions and optionally phase/amplitude knobs depending on the action set). For each candidate action, the solver is called, the radiation force at the particle position is sampled, a score is computed based on alignment and push along the desired direction, and the best action is chosen. The chosen action’s field is then used to integrate particle dynamics.

This controller is intentionally expensive but reliable, because it uses the PDE as the oracle. It establishes the key fact: the particle can be moved by “surfing” a changing force field, rather than requiring a stable static trap.


# ============================================================
# Circle tracking improvements that are now implemented
# ============================================================

Circle tracking is handled differently from a straight line. A waypoint-chasing direction can cause oscillation on a circle, so the demo supports a circle-specific desired direction mode that combines tangential motion with a radial correction back toward the circle. This stabilises progress around the loop.

Circle target advancement is also handled explicitly. Instead of the target point running away or freezing, the circle target index can advance based on angle progress so that the target remains consistently “ahead” of the particle around the loop. This produces more realistic desired directions and better-looking runs.


# ============================================================
# Visualisation that exists now
# ============================================================

The GIF output overlays the particle trajectory, target marker, force vectors, and a 2D contour of the Gor’kov potential U for the chosen action at each timestep. The contour colour scaling is designed to be stable enough to watch changes over time instead of flickering wildly between frames, and the contour displayed should always correspond to the chosen action’s field for that timestep.


# ============================================================
# Bayesian acceleration layer (working, but not yet the final answer)
# ============================================================

A Bayesian action-selection layer exists in the same demo script. It does not change the physics, scoring, dynamics, or path logic. Its only job is to reduce how many candidate actions require a full PDE solve.

The Bayes controller uses a learned surrogate model to predict action quality and chooses only K candidate actions per step for truth evaluation using an acquisition rule (UCB). The chosen action is still selected by the true PDE score among the evaluated subset, so the Bayes layer is an acceleration mechanism rather than a replacement controller.

In short runs, the Bayes layer achieves a clear reduction in PDE solves per step and still produces sensible motion. In longer runs, Bayes can sometimes lock onto a locally good action and become less robust than the greedy oracle, especially on closed paths where long-horizon exploration matters.


# ============================================================
# What we are trying to achieve next (the immediate practical goal)
# ============================================================

The immediate goal is a dependable “final demo run” that is visually compelling and technically honest: the particle completes the full circle with good cross-track error and smooth progress, while the Gor’kov contours clearly evolve as the controller applies different actions.

At the moment the system can produce runs that either complete large angle progress but track poorly, or track well but stall later. The next step is to understand the parameter/setting differences behind “good looking” runs (for example, comparing runs like run_20260115_221636 against later runs), and to stabilise those conditions into a reliable configuration.


# ============================================================
# What comes after that (toward the real final system)
# ============================================================

Once we have repeatable robust path-following, the project transitions from discrete surfing toward continuous optimal control.

The intended endgame is adjoint-based model predictive control, where actuation parameters (including robot transducer positions and phase/amplitude knobs) are optimised over a short horizon to minimise tracking error and enforce constraints. Bayesian optimisation and learned surrogates remain useful in that future system as accelerators, warm-start tools, and model-mismatch compensators, while reinforcement learning becomes relevant when experimental feedback and unmodelled dynamics dominate.

The direction is therefore hybrid: keep the physics exact, add gradient-based planning for control authority, and use learning methods to reduce compute and handle uncertainty.


# ============================================================
# How to run the current demos
# ============================================================

The main demo lives in scripts/demo_surf_greedy.py. It supports line and circle paths, greedy or Bayes controllers, controllable rendering stride, and produces a timestamped results folder containing demo_surf_greedy.gif, summary.png, steps.csv, and summary.json.

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
