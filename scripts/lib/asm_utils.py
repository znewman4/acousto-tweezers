"""
ASM Utilities — reusable module for Angular Spectrum Method workflows.

Provides FEniCS-independent functions for:

1. Grid generation from FEM cache metadata
2. Vortex beam field generation (LG-like)
3. Lens phase profile computation (ideal / plastic / axicon)
4. ASM propagation (wraps canonical propagate_pressure_asm)
5. C-shape perturbation mask generation

All functions operate on numpy arrays and meshgrid coordinates.
No FEniCS / DOLFINx dependency.

Usage
-----
    from scripts.lib.asm_utils import (
        make_grid_from_fem, make_vortex_field, make_lens_phase,
        propagate_asm, make_cshape_mask,
    )
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

# ── Ensure src/ is importable ──────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
_src = str(PROJECT_ROOT / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from acoustweezers.physics.acoustics.angular_spectrum import (
    propagate_pressure_asm,
)

# ── Physical constants (match fem_cache_utils) ──────────────────────
C_WATER = 1484.0          # m/s
F_HZ = 2.0e6              # Hz
LAM = C_WATER / F_HZ      # 0.742 mm
K0 = 2.0 * np.pi / LAM    # wavenumber [rad/m]


# ════════════════════════════════════════════════════════════════════
# 1. Grid generation from FEM cache
# ════════════════════════════════════════════════════════════════════

def make_grid_from_fem(
    cache: Dict[str, Any],
    nx: int = 400,
    ny: int = 400,
    margin: float = 50e-6,
) -> Dict[str, Any]:
    """
    Build a uniform XY meshgrid covering the FEM domain.

    Parameters
    ----------
    cache : dict
        Output of ``load_fem_cache()``.  Must contain ``domain`` dict
        with keys ``x_min, x_max, y_min, y_max``.
    nx, ny : int
        Number of grid points along x and y.
    margin : float
        Inset from domain edges [m] to avoid boundary artefacts.

    Returns
    -------
    dict with keys
        ``x``  : 1-D array (nx,)
        ``y``  : 1-D array (ny,)
        ``XX`` : 2-D meshgrid (ny, nx)
        ``YY`` : 2-D meshgrid (ny, nx)
        ``dx`` : float — grid spacing along x [m]
        ``dy`` : float — grid spacing along y [m]
        ``Lx`` : float — usable domain width [m]
        ``Ly`` : float — usable domain height [m]
    """
    dom = cache["domain"]
    x = np.linspace(dom["x_min"] + margin, dom["x_max"] - margin, nx)
    y = np.linspace(dom["y_min"] + margin, dom["y_max"] - margin, ny)
    XX, YY = np.meshgrid(x, y)
    dx = float(x[1] - x[0]) if nx > 1 else 1.0
    dy = float(y[1] - y[0]) if ny > 1 else 1.0
    return {
        "x": x, "y": y, "XX": XX, "YY": YY,
        "dx": dx, "dy": dy,
        "Lx": float(x[-1] - x[0]),
        "Ly": float(y[-1] - y[0]),
    }


# ════════════════════════════════════════════════════════════════════
# 2. Vortex field generation (LG-like, FEniCS-free)
# ════════════════════════════════════════════════════════════════════

def make_vortex_field(
    x: np.ndarray,
    y: np.ndarray,
    charge: int = 1,
    waist: float = 0.6e-3,
    k: float = K0,
    center: Optional[Tuple[float, float]] = None,
    aperture_radius: Optional[float] = None,
) -> np.ndarray:
    """
    Generate an LG-like vortex beam field on coordinate arrays.

    Amplitude envelope (Laguerre–Gaussian zero-radial-order mode):

        A(r) = (r/w)^|ℓ| · exp(−r²/w²),   normalised to max = 1

    Phase:

        φ(x,y) = ℓ · arctan2(y − cy, x − cx)

    Parameters
    ----------
    x, y : ndarray
        Coordinate arrays (meshgrid or flat).  Same shape.
    charge : int
        Topological charge ℓ.
    waist : float
        Beam waist w [m].
    k : float
        Wavenumber [rad/m].  Stored for API consistency; not used in
        the amplitude/phase formula.
    center : (cx, cy) or None
        Centre of the vortex.  Defaults to domain centre.
    aperture_radius : float or None
        Hard aperture radius [m].  Field zeroed outside.

    Returns
    -------
    field : ndarray (complex128), same shape as x
    """
    if center is None:
        cx = float(0.5 * (x.min() + x.max()))
        cy = float(0.5 * (y.min() + y.max()))
    else:
        cx, cy = center

    dx = x - cx
    dy = y - cy
    r = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)

    ell = abs(charge)
    rw = r / waist
    amp = (rw ** ell) * np.exp(-(r ** 2) / (waist ** 2))

    if aperture_radius is not None:
        amp[r > aperture_radius] = 0.0

    a_max = amp.max()
    if a_max > 0:
        amp /= a_max

    return amp * np.exp(1j * charge * theta)


# ════════════════════════════════════════════════════════════════════
# 3. Lens phase generation (ideal / plastic / axicon)
# ════════════════════════════════════════════════════════════════════

def make_lens_phase(
    x: np.ndarray,
    y: np.ndarray,
    focal_length: float = 10.0e-3,
    aperture_radius: float = 1.0e-3,
    curvature: Optional[float] = None,
    family: str = "ideal",
    k: float = K0,
    charge: int = 0,
    axicon_angle_deg: float = 15.0,
    center: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """
    Compute a lens phase profile on coordinate arrays.

    Families
    --------
    ``"ideal"`` — Spherical converging lens (exact):
        φ = ℓθ + k(√(r² + f²) − f)

    ``"plastic"`` — Wrapped fabricable lens:
        φ = mod(φ_ideal, 2π)

    ``"axicon"`` — Conical (Bessel-generating) lens:
        φ = ℓθ + k·sin(α)·r

    Parameters
    ----------
    x, y : ndarray
        Coordinate arrays (meshgrid or flat).  Same shape.
    focal_length : float
        Focal length f [m].  Used by ``ideal`` and ``plastic``.
    aperture_radius : float
        Lens aperture radius R [m].  Phase zeroed outside.
    curvature : float or None
        If not None, overrides the focusing term with the parabolic
        thin-lens approximation: φ_focus = curvature · k · r² / 2.
        Units: 1/m (≈ 1/f for a thin lens).  Only affects ``ideal``
        and ``plastic`` families.
    family : {"ideal", "plastic", "axicon"}
        Phase profile family.
    k : float
        Medium wavenumber [rad/m].
    charge : int
        Vortex topological charge ℓ (0 for pure focusing lens).
    axicon_angle_deg : float
        Axicon half-angle α [degrees].  Only used for ``"axicon"``.
    center : (cx, cy) or None
        Centre of the lens.  Defaults to domain centre.

    Returns
    -------
    phase : ndarray (float64), same shape as x
        Phase values [rad].  Zero outside aperture.
    """
    if center is None:
        cx = float(0.5 * (x.min() + x.max()))
        cy = float(0.5 * (y.min() + y.max()))
    else:
        cx, cy = center

    rx = x - cx
    ry = y - cy
    r = np.sqrt(rx**2 + ry**2)
    theta = np.arctan2(ry, rx)
    inside = r <= aperture_radius

    phi = np.zeros_like(r, dtype=np.float64)

    if family in ("ideal", "plastic"):
        if curvature is not None:
            phi_focus = curvature * k * r**2 / 2.0
        else:
            phi_focus = k * (np.sqrt(r**2 + focal_length**2) - focal_length)
        phi_raw = charge * theta + phi_focus
        if family == "plastic":
            phi[inside] = np.mod(phi_raw[inside], 2.0 * np.pi)
        else:
            phi[inside] = phi_raw[inside]

    elif family == "axicon":
        alpha = np.deg2rad(axicon_angle_deg)
        k_r = k * np.sin(alpha)
        phi_raw = charge * theta + k_r * r
        phi[inside] = phi_raw[inside]

    else:
        raise ValueError(
            f"Unknown lens family: {family!r}. "
            f"Expected 'ideal', 'plastic', or 'axicon'."
        )

    return phi


# ════════════════════════════════════════════════════════════════════
# 4. ASM propagation (wrapper)
# ════════════════════════════════════════════════════════════════════

def propagate_asm(
    field_xy: np.ndarray,
    dx: float,
    dy: float,
    wavelength: float = LAM,
    z: float = 5.0e-3,
    pad_factor: int = 2,
) -> np.ndarray:
    """
    Propagate a 2-D complex field via the angular spectrum method.

    Thin wrapper around the canonical ``propagate_pressure_asm``
    (Rayleigh–Sommerfeld Type-I, with zero-padding).

    Parameters
    ----------
    field_xy : ndarray (Ny, Nx), complex
        Source-plane complex field at z = 0.
    dx, dy : float
        Grid spacing [m].
    wavelength : float
        Medium wavelength [m].
    z : float
        Propagation distance [m].  Positive = forward.
    pad_factor : int
        Zero-padding multiplier (default 2).

    Returns
    -------
    p : ndarray (Ny, Nx), complex
        Propagated field at height z.
    """
    k = 2.0 * np.pi / wavelength
    return propagate_pressure_asm(
        field_xy, dx, dy, k, z, pad_factor=pad_factor,
    )


# ════════════════════════════════════════════════════════════════════
# 5. C-shape perturbation mask
# ════════════════════════════════════════════════════════════════════

def make_cshape_mask(
    x: np.ndarray,
    y: np.ndarray,
    radius: float,
    gap_angle: float,
    thickness: float,
    charge: int = 1,
    gap_width: float = 0.3,
    beta: float = 1.0,
    center: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """
    Generate a C-shaped perturbation mask.

    The mask is a Gaussian ring with an angular gap:

        p_C = A(r) · W(θ) · exp(i·m·θ)

        A(r) = exp(−(r − r₀)² / (2σ_r²))     radial Gaussian ring
        W(θ) = 1 − β·exp(−Δθ² / (2σ_θ²))     angular suppression

    Parameters
    ----------
    x, y : ndarray
        Coordinate arrays (meshgrid or flat).  Same shape.
    radius : float
        Ring radius r₀ [m].
    gap_angle : float
        Centre of the angular gap [rad].
    thickness : float
        Radial ring width σ_r [m].
    charge : int
        Azimuthal winding number m.
    gap_width : float
        Angular gap half-width σ_θ [rad].
    beta : float
        Gap depth (0 = no gap, 1 = full suppression).
    center : (cx, cy) or None
        Centre of the C-shape.  Defaults to domain centre.

    Returns
    -------
    mask : ndarray (complex128), same shape as x
    """
    if center is None:
        cx = float(0.5 * (x.min() + x.max()))
        cy = float(0.5 * (y.min() + y.max()))
    else:
        cx, cy = center

    rx = x - cx
    ry = y - cy
    r = np.sqrt(rx**2 + ry**2)
    theta = np.arctan2(ry, rx)

    # Radial Gaussian ring
    A_r = np.exp(-0.5 * ((r - radius) / thickness) ** 2)

    # Angular gap (wrapped to [-π, π])
    dtheta = theta - gap_angle
    dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
    W_theta = 1.0 - beta * np.exp(-0.5 * (dtheta / gap_width) ** 2)

    return A_r * W_theta * np.exp(1j * charge * theta)
