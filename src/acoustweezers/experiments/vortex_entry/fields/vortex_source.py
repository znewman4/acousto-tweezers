"""
Focused vortex source generation and data loading.

Extracted verbatim from vortex_entry_test.py.  No physics changes.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import numpy as np

from scripts.lib.fem_cache_utils import (
    C_WATER, F_HZ, OMEGA, RHO0,
    default_particle_params, gorkov_grid_2d,
)
from scripts.lib.asm_utils import (
    propagate_asm,
    LAM as ASM_LAM, K0 as ASM_K0,
)
from scripts.lib.overlay_utils import estimate_ring_radius
from scripts.lib.perturbation_vortex import VortexPerturbation
from scripts.lib.particle_dynamics_utils import LAM


# ── Vortex design defaults ────────────────────────────────────────
VORTEX_CHARGE = 1


def make_focused_vortex_source(
    XX: np.ndarray,
    YY: np.ndarray,
    charge: int,
    aperture_radius: float,
    k: float,
    focal_length: float,
    center: Tuple[float, float],
    apodization: str = "cosine_taper",
    apod_width: float = 0.15,
) -> np.ndarray:
    """
    Phase-only focused vortex source field.

    Aperture-plane phase:
        phi(r, theta) = charge * theta  -  k * r^2 / (2 * focal_length)

    The first term is the spiral phase (topological charge).
    The second term is a converging thin-lens phase.  Together they
    produce a tighter, more localised vortex at the focal plane
    compared to free-propagation spiral-phase alone.

    The amplitude is uniform inside the aperture (phase-only model),
    directly interpretable as a passive holographic acoustic lens
    that imprints combined spiral + converging-lens phase onto an
    incident plane wave.

    Ring radius at focal plane (first-order vortex, uniform aperture):
        r_ring ≈ 0.42 * lambda * focal_length / (2 * aperture_radius)
    """
    cx, cy = center
    rx = XX - cx
    ry = YY - cy
    r = np.sqrt(rx ** 2 + ry ** 2)
    theta = np.arctan2(ry, rx)

    amp = np.ones_like(r)
    amp[r > aperture_radius] = 0.0

    if apodization == "cosine_taper" and apod_width > 0:
        r0 = aperture_radius * (1.0 - apod_width)
        mask = (r > r0) & (r <= aperture_radius)
        amp[mask] = 0.5 * (1.0 + np.cos(
            np.pi * (r[mask] - r0) / (aperture_radius * apod_width)
        ))

    a_max = float(amp.max())
    if a_max > 0:
        amp = amp / a_max

    phase = charge * theta - k * r ** 2 / (2.0 * focal_length)
    return amp * np.exp(1j * phase)


def load_data(
    aperture_m: float,
    prop_dist_m: float,
    focus_f_m: float,
    field_npz: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float,
           np.ndarray, VortexPerturbation, float]:
    """
    Load full-domain standing wave, full trap lattice, build focused
    vortex, and calibrate barrier radius.

    Returns (p_sw, xg, yg, dx, dy, traps_m, vortex_gen, r_barrier).
    """
    print("Loading full-domain standing-wave and trap lattice...")
    fd = np.load(field_npz)
    xg      = fd["xg"].astype(float)
    yg      = fd["yg"].astype(float)
    p_sw    = fd["p_sw"].astype(complex)
    traps_m = fd["traps_m"].astype(float)
    p_sw_peak = float(np.max(np.abs(p_sw)))

    dx = float(xg[1] - xg[0])
    dy = float(yg[1] - yg[0])

    print(f"  Grid: {len(xg)}x{len(yg)}, "
          f"domain [{xg[0]*1e3:.2f}, {xg[-1]*1e3:.2f}] x "
          f"[{yg[0]*1e3:.2f}, {yg[-1]*1e3:.2f}] mm, "
          f"dx={dx*1e6:.1f} um")
    print(f"  p_sw max amplitude: {p_sw_peak:.4f}")
    print(f"  Trap lattice: {len(traps_m)} traps")

    # ── Generate Bessel vortex ────────────────────────────────────
    grid_cx = 0.5 * (float(xg[0]) + float(xg[-1]))
    grid_cy = 0.5 * (float(yg[0]) + float(yg[-1]))
    XX, YY = np.meshgrid(xg, yg)

    print(f"  Generating focused vortex (R={aperture_m*1e3:.2f} mm, "
          f"f={focus_f_m*1e3:.2f} mm, z={prop_dist_m*1e3:.2f} mm)...")
    p_source = make_focused_vortex_source(
        XX, YY,
        charge=VORTEX_CHARGE,
        aperture_radius=aperture_m,
        k=ASM_K0,
        focal_length=focus_f_m,
        center=(grid_cx, grid_cy),
        apodization="cosine_taper",
    )
    p_vortex_raw = propagate_asm(p_source, dx, dy, wavelength=ASM_LAM,
                                  z=prop_dist_m)

    p_vortex_peak = float(np.max(np.abs(p_vortex_raw)))
    if p_vortex_peak > 0:
        p_vortex_raw = p_vortex_raw * (p_sw_peak / p_vortex_peak)

    # ── Calibrate: measure barrier ring radius ────────────────────
    mag = np.abs(p_vortex_raw)
    center_xy = np.array([grid_cx, grid_cy])
    ring_info = estimate_ring_radius(
        mag, xg, yg, center=center_xy,
        r_min=0.05e-3, r_max=1.0e-3, n_bins=300,
    )
    r_barrier = float(ring_info["ring_radius_m"])
    ring_peak = float(ring_info["ring_peak"])

    r_theory = 0.42 * ASM_LAM * focus_f_m / (2.0 * aperture_m)
    print(f"\n  --- Focused Vortex Calibration ---")
    print(f"  Aperture radius:      {aperture_m*1e3:.2f} mm")
    print(f"  Focal length:         {focus_f_m*1e3:.2f} mm")
    print(f"  Propagation distance: {prop_dist_m*1e3:.2f} mm  "
          f"({'at focus' if abs(prop_dist_m - focus_f_m) < 0.05e-3 else 'off-focus'})")
    print(f"  Pressure ring radius: {r_barrier*1e6:.1f} um "
          f"({r_barrier/LAM:.2f} lam)")
    print(f"  Ring peak amplitude:  {ring_peak:.4f}")
    print(f"  Theory (focal-plane): {r_theory*1e6:.1f} um  "
          f"[0.42*lam*f/(2R)]")

    vortex_gen = VortexPerturbation(p_vortex_raw, xg, yg,
                                     out_xg=xg, out_yg=yg)
    return p_sw, xg, yg, dx, dy, traps_m, vortex_gen, r_barrier
