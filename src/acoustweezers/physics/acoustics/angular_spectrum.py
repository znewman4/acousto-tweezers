"""
Angular Spectrum Method (ASM) Propagation
==========================================

Canonical implementation of monochromatic scalar-field propagation via
the angular spectrum method.  This is the spectral form of the
Rayleigh–Sommerfeld Type-I diffraction integral.

**Physical setup:**

    A complex scalar field D(x, y) is specified on the plane z = 0.
    The field propagates into the half-space z > 0 in a homogeneous
    medium with wavenumber k = ω / c.

**Method:**

    p(x, y, z) = F⁻¹[ F[D] · H ]

    where H(kx, ky; z) = exp(i kz z)
whyve 
          kz = √(k² − kx² − ky²)        for propagating modes (kx²+ky² ≤ k²)
          kz = i √(kx² + ky² − k²)       for evanescent modes  (kx²+ky² > k²)

    This is the *exact* transfer function for the Helmholtz equation
    in a homogeneous medium — no paraxial / Fresnel approximation.

**Boundary interpretation:**

    - ``propagate_pressure_asm`` treats D as a pressure field p(x,y,0).
      H = exp(i kz z)  (RS-I kernel).

    - ``propagate_velocity_asm`` treats D as normal velocity v_z(x,y,0).
      H = (ωρ/kz) exp(i kz z)  (Rayleigh integral for a vibrating
      planar source).  This is the spectral counterpart of the FEM
      Neumann-BC source model.

**Wraparound control:**

    The FFT assumes periodicity.  Zero-padding by ``pad_factor`` in
    each dimension converts the circular convolution to a (larger)
    linear convolution, pushing wraparound aliases outside the region
    of interest.  ``pad_factor=2`` (default) doubles each axis.

**Evanescent modes:**

    When ``include_evanescent=True`` (default), evanescent waves are
    retained with the exponentially decaying kz.  For large z these
    contributions are negligible; for near-field analysis they can
    matter.  Setting ``include_evanescent=False`` zeroes them out.

Author: Acousto-Tweezers Project
Date: March 2026
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


# =====================================================================
# Helper: spectral grids
# =====================================================================

def make_k_grids(
    nx: int,
    ny: int,
    dx: float,
    dy: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return 2-D wavenumber grids (KX, KY) matching an (ny, nx) spatial grid.

    Parameters
    ----------
    nx, ny : int
        Number of grid points along x and y.
    dx, dy : float
        Grid spacing [m] along x and y.

    Returns
    -------
    KX, KY : ndarray, shape (ny, nx)
        Wavenumber grids [rad/m] ordered for ``numpy.fft``.
    """
    fx = np.fft.fftfreq(nx, d=dx)
    fy = np.fft.fftfreq(ny, d=dy)
    FX, FY = np.meshgrid(fx, fy)
    return 2.0 * np.pi * FX, 2.0 * np.pi * FY


# =====================================================================
# Core: kz computation
# =====================================================================

def _compute_kz(
    kx: np.ndarray,
    ky: np.ndarray,
    k: float,
    include_evanescent: bool = True,
) -> np.ndarray:
    """
    Compute the longitudinal wavenumber kz for each (kx, ky) pair.

    Propagating modes:  kz = sqrt(k² − kx² − ky²)    (real, ≥ 0)
    Evanescent modes:   kz = i sqrt(kx² + ky² − k²)  (pure imaginary)

    If ``include_evanescent`` is False, evanescent kz is set to zero
    (=> H = 1 for those modes — effectively frozen, not decaying).
    """
    kt2 = kx ** 2 + ky ** 2
    kz2 = k ** 2 - kt2

    propagating = kz2 >= 0
    kz = np.zeros_like(kz2, dtype=complex)
    kz[propagating] = np.sqrt(kz2[propagating])

    if include_evanescent:
        kz[~propagating] = 1j * np.sqrt(-kz2[~propagating])
    # else: leave kz = 0 for evanescent → H = exp(0) = 1 (no decay)

    return kz


# =====================================================================
# Pressure-mode propagator (RS-I)
# =====================================================================

def propagate_pressure_asm(
    D_xy: np.ndarray,
    dx: float,
    dy: float,
    k: float,
    z: float,
    pad_factor: int = 2,
    include_evanescent: bool = True,
) -> np.ndarray:
    """
    Propagate a pressure source field D(x,y) from z=0 to height z.

    Parameters
    ----------
    D_xy : ndarray, shape (Ny, Nx), complex
        Source-plane complex pressure field at z = 0.
    dx, dy : float
        Grid spacing [m].
    k : float
        Medium wavenumber [rad/m].  k = 2π f / c.
    z : float
        Propagation distance [m].  Positive = forward.
    pad_factor : int
        Multiply each dimension by this factor with zero-padding
        to suppress wraparound.  1 = no padding.
    include_evanescent : bool
        If True, evanescent modes decay exponentially (correct physics).
        If False, evanescent modes are frozen (H=1).

    Returns
    -------
    p : ndarray, shape (Ny, Nx), complex
        Propagated field at height z.
    """
    Ny, Nx = D_xy.shape

    # ── zero-pad ──────────────────────────────────────────────────
    Ny_p = Ny * pad_factor
    Nx_p = Nx * pad_factor
    D_pad = np.zeros((Ny_p, Nx_p), dtype=complex)
    D_pad[:Ny, :Nx] = D_xy

    # ── spectral grids & transfer function ────────────────────────
    KX, KY = make_k_grids(Nx_p, Ny_p, dx, dy)
    kz = _compute_kz(KX, KY, k, include_evanescent=include_evanescent)
    H = np.exp(1j * kz * z)

    # ── propagate ─────────────────────────────────────────────────
    p_pad = np.fft.ifft2(np.fft.fft2(D_pad) * H)

    return p_pad[:Ny, :Nx]


# =====================================================================
# Velocity-mode propagator (Rayleigh integral)
# =====================================================================

def propagate_velocity_asm(
    Vn_xy: np.ndarray,
    dx: float,
    dy: float,
    k: float,
    z: float,
    omega: float,
    rho: float,
    pad_factor: int = 2,
    kz_floor_frac: float = 0.05,
) -> np.ndarray:
    """
    Propagate a velocity source field v_z(x,y) from z=0 to height z.

    Uses the Rayleigh integral transfer function:

        H(kx,ky; z) = (ωρ / kz) · exp(i kz z)

    for propagating modes.  Evanescent modes are zeroed out.
    The 1/kz singularity at grazing incidence is regularised by
    clamping kz ≥ kz_floor_frac · k.

    Parameters
    ----------
    Vn_xy : ndarray, shape (Ny, Nx), complex
        Source-plane complex normal velocity field at z = 0.
    dx, dy : float
        Grid spacing [m].
    k : float
        Medium wavenumber [rad/m].
    z : float
        Propagation distance [m].
    omega : float
        Angular frequency [rad/s].
    rho : float
        Medium density [kg/m³].
    pad_factor : int
        Zero-padding factor (default 2).
    kz_floor_frac : float
        Minimum kz as fraction of k (regularisation).

    Returns
    -------
    p : ndarray, shape (Ny, Nx), complex
        Propagated pressure at height z.
    """
    Ny, Nx = Vn_xy.shape

    Ny_p = Ny * pad_factor
    Nx_p = Nx * pad_factor
    D_pad = np.zeros((Ny_p, Nx_p), dtype=complex)
    D_pad[:Ny, :Nx] = Vn_xy

    KX, KY = make_k_grids(Nx_p, Ny_p, dx, dy)
    kz = _compute_kz(KX, KY, k, include_evanescent=False)

    # regularised denominator
    kz_real = np.real(kz)
    kz_floor = k * kz_floor_frac
    kz_denom = np.maximum(kz_real, kz_floor)

    propagating = kz_real > 0
    H = np.zeros((Ny_p, Nx_p), dtype=complex)
    H[propagating] = (
        (omega * rho / kz_denom[propagating])
        * np.exp(1j * kz[propagating] * z)
    )

    p_pad = np.fft.ifft2(np.fft.fft2(D_pad) * H)
    return p_pad[:Ny, :Nx]


# =====================================================================
# Self-check (run with:  python -m acoustweezers.physics.acoustics.angular_spectrum)
# =====================================================================

if __name__ == "__main__":
    print("Angular Spectrum Module — self-check")
    print("=" * 50)

    # Parameters: 2 MHz in water
    c = 1484.0
    f_hz = 2.0e6
    lam = c / f_hz
    k = 2.0 * np.pi * f_hz / c

    N = 128
    L = 4.0e-3
    dx = dy = L / N
    x = np.linspace(0, L, N, endpoint=False)
    y = np.linspace(0, L, N, endpoint=False)
    XX, YY = np.meshgrid(x, y)

    # Test 1: propagating z=0 should return the input
    D = np.exp(-(((XX - L / 2) ** 2 + (YY - L / 2) ** 2) / (0.5e-3) ** 2))
    p0 = propagate_pressure_asm(D, dx, dy, k, z=0.0, pad_factor=2)
    err = np.max(np.abs(p0 - D)) / np.max(np.abs(D))
    status = "PASS" if err < 1e-10 else "FAIL"
    print(f"  Test 1 (z=0 identity):  max relative error = {err:.2e}  [{status}]")

    # Test 2: energy should not blow up for moderate z
    p_far = propagate_pressure_asm(D, dx, dy, k, z=3.0e-3, pad_factor=2)
    energy_in = np.sum(np.abs(D) ** 2)
    energy_out = np.sum(np.abs(p_far) ** 2)
    ratio = energy_out / energy_in
    status = "PASS" if ratio < 2.0 else "FAIL"
    print(f"  Test 2 (energy bound):  E_out/E_in = {ratio:.4f}  [{status}]")

    # Test 3: plane wave propagation phase
    D_plane = np.ones((N, N), dtype=complex)
    z_test = 1.0e-3
    p_plane = propagate_pressure_asm(D_plane, dx, dy, k, z=z_test, pad_factor=1)
    # Central pixel should have phase ≈ k*z
    phase_centre = np.angle(p_plane[N // 2, N // 2])
    expected_phase = (k * z_test) % (2.0 * np.pi)
    if expected_phase > np.pi:
        expected_phase -= 2.0 * np.pi
    phase_err = abs(phase_centre - expected_phase)
    if phase_err > np.pi:
        phase_err = 2.0 * np.pi - phase_err
    status = "PASS" if phase_err < 0.01 else "FAIL"
    print(f"  Test 3 (plane-wave phase):  |Δφ| = {phase_err:.4f} rad  [{status}]")

    print("\nDone.")
