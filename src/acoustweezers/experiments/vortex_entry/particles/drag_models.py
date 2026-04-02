"""
Drag model constants for inertial polystyrene particles in water.

Physics model (Option C):
    m dv/dt = F_gorkov(x) - 6π η a v
    dx/dt   = v

Equivalently:
    dv/dt = ACCEL_SCALE * F_norm(x)  -  v / TAU_STOKES

where:
    F_norm  — normalised Gor'kov force from gorkov_normalised() / iFx, iFy
              units: Pa⁻¹ m⁻¹ (dimensionless-pressure-based, see below)
    ACCEL_SCALE = GORKOV_PREFACTOR * P_SCALE² / m_particle   [m/s²  per  Pa⁻¹ m⁻¹]
    TAU_STOKES  = 2 ρ_p a² / (9 η)                           [s]

Regime assumptions (explicitly flagged):
  1. Stokes drag: Re_p = ρ_f a |v - u| / η ≪ 1.
       Typical |v| ~ 20 µm/s, a = 50 µm → Re_p ~ 10⁻³.  Satisfied.
  2. Spherical particle.
  3. No Basset history force, no added-mass term.
       Density ratio ρ_p/ρ_f ≈ 1.05 (near-neutral), so added-mass O(ρ_p/ρ_f)
       correction is ~5 %.  Acceptable for diagnostic use.
  4. One-way coupling: particle does not perturb the acoustic field.
  5. No inter-particle interactions (dilute ensemble).

Consistency check:
    At terminal velocity (dv/dt = 0):
        v_term = ACCEL_SCALE * F_norm * TAU_STOKES
               = (GORKOV_PREFACTOR * P_SCALE² / m) * (2 ρ_p a² / (9η)) * F_norm
               = GORKOV_PREFACTOR * P_SCALE² * (1/(6π η a)) * F_norm
               = SCALE * F_norm    ← identical to the overdamped model in dynamics.py
    The two models converge at steady state.  Inertia only matters during transients
    with timescale τ ≈ 0.58 ms ≈ 6 × DT_DEFAULT.
"""
from __future__ import annotations

import numpy as np

from scripts.lib.particle_dynamics_utils import (
    A_PART, RHO_P, ETA,
    GORKOV_PREFACTOR, P_SCALE,
)

# ── Particle geometry ──────────────────────────────────────────────
PARTICLE_DIAMETER: float = 2.0 * A_PART           # 100 µm  [m]
PARTICLE_RADIUS:   float = A_PART                  # 50 µm   [m]

# ── Particle mass ──────────────────────────────────────────────────
M_PARTICLE: float = RHO_P * (4.0 / 3.0) * np.pi * A_PART ** 3  # kg
# ≈ 1050 * (4π/3) * (50e-6)³ ≈ 5.50e-10 kg

# ── Stokes relaxation time ─────────────────────────────────────────
TAU_STOKES: float = 2.0 * RHO_P * A_PART ** 2 / (9.0 * ETA)  # s
# = 2 * 1050 * (50e-6)² / (9 * 1e-3)
# = 2 * 1050 * 2.5e-9 / 9e-3
# ≈ 5.83e-4 s  ≈ 0.583 ms

# ── Acceleration scale ─────────────────────────────────────────────
# Converts normalised Gor'kov force (Pa⁻¹ m⁻¹) to physical acceleration (m/s²).
#
# F_physical = GORKOV_PREFACTOR * P_SCALE² * F_norm   [N]
# a_physical = F_physical / M_PARTICLE                [m/s²]
#
# Simplifies to P_SCALE² / (2 * RHO_P) because:
#   GORKOV_PREFACTOR / M_PARTICLE
#   = (2π/3 a³) / (RHO_P * 4π/3 * a³)
#   = 1 / (2 * RHO_P)
ACCEL_SCALE: float = GORKOV_PREFACTOR * P_SCALE ** 2 / M_PARTICLE  # m/s² per (Pa⁻¹ m⁻¹)
# = P_SCALE² / (2 * RHO_P)
# ≈ 9e6 / 2100 ≈ 4286 m/s²

# ── Sanity check (relationship to overdamped SCALE) ───────────────
# SCALE (from particle_dynamics_utils) = MU_STOKES * GORKOV_PREFACTOR * P_SCALE²
# TAU_STOKES / M_PARTICLE = 1 / (6π η a) = MU_STOKES
# Therefore: ACCEL_SCALE * TAU_STOKES = GORKOV_PREFACTOR * P_SCALE² / M_PARTICLE * TAU_STOKES
#           = GORKOV_PREFACTOR * P_SCALE² * (1/(6π η a)) = SCALE  ✓
