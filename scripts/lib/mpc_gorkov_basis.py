"""
Trilinear Gor'kov decomposition for MPC vortex-merge control.

Extends the bilinear basis from particle_dynamics_utils.py by separating
the cross-term into real (cosine) and imaginary (sine) components, enabling
**analytic gradients** w.r.t. the phase delay ψ, vortex amplitude α, and
standing-wave amplitude β — without re-computing any fields.

Decomposition
-------------
The total pressure field is:

    p_total = β · p_sw  +  α · exp(iψ) · p_v(x_v, y_v)

Since Gor'kov U is quadratic in p, we can write:

    U_total = β² U₁₁ + α² U₂₂ + 2βα [cos(ψ) U₁₂c + sin(ψ) U₁₂s]

where:
    U₁₁ = Gorkov(p_sw)                — standing-wave self-term
    U₂₂ = Gorkov(p_v)                 — vortex self-term
    U₁₂c = cross-term from Re(p_sw* · p_v)
    U₁₂s = cross-term from Im(p_sw* · p_v)

The same decomposition holds for forces Fx, Fy = −∇U.

Analytic gradients
------------------
    ∂F/∂β  = 2β Fx₁₁ + 2α [cos ψ · Fx₁₂c + sin ψ · Fx₁₂s]
    ∂F/∂α  = 2α Fx₂₂ + 2β [cos ψ · Fx₁₂c + sin ψ · Fx₁₂s]
    ∂F/∂ψ  = 2βα [−sin ψ · Fx₁₂c + cos ψ · Fx₁₂s]

Position gradients ∂F/∂(x_v, y_v) are computed via finite differences
on the shifted vortex field.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from scripts.lib.particle_dynamics_utils import (
    C_WATER,
    F1,
    F2,
    OMEGA,
    RHO0,
    gorkov_normalised,
)

# ── Basis tuple type ──────────────────────────────────────────────
# (U_11, Fx_11, Fy_11,
#  U_22, Fx_22, Fy_22,
#  U_12c, Fx_12c, Fy_12c,
#  U_12s, Fx_12s, Fy_12s)
TriBasis = Tuple[
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray,
]


def precompute_trilinear_basis(
    p_sw: np.ndarray,
    p_vortex: np.ndarray,
    dx: float,
    dy: Optional[float] = None,
) -> TriBasis:
    """
    Precompute the 12-component trilinear Gor'kov basis.

    Parameters
    ----------
    p_sw : complex ndarray (ny, nx)
        Standing-wave pressure field.
    p_vortex : complex ndarray (ny, nx)
        Vortex pressure field (centred, **no** exp(iψ) applied).
    dx, dy : float
        Grid spacings [m].

    Returns
    -------
    TriBasis — 12 fields, each (ny, nx).
    """
    if dy is None:
        dy = dx

    coeff_p = F1 / (2.0 * RHO0 * C_WATER ** 2)
    coeff_k = 3.0 * F2 / (4.0 * OMEGA ** 2 * RHO0)

    # Self-terms
    U_11, Fx_11, Fy_11 = gorkov_normalised(p_sw, dx, dy)
    U_22, Fx_22, Fy_22 = gorkov_normalised(p_vortex, dx, dy)

    # Gradients of both fields
    dp1_dx = np.gradient(p_sw, dx, axis=1)
    dp1_dy = np.gradient(p_sw, dy, axis=0)
    dp2_dx = np.gradient(p_vortex, dx, axis=1)
    dp2_dy = np.gradient(p_vortex, dy, axis=0)

    # Cross product  p_sw* · p_v  =  Re(...) + i Im(...)
    cross_complex = np.conj(p_sw) * p_vortex
    grad_cross_complex = (
        np.conj(dp1_dx) * dp2_dx + np.conj(dp1_dy) * dp2_dy
    )

    # Real (cosine) cross-term
    cross_p_c = np.real(cross_complex)
    cross_g_c = np.real(grad_cross_complex)
    U_12c = coeff_p * cross_p_c - coeff_k * cross_g_c
    Fx_12c = -np.gradient(U_12c, dx, axis=1)
    Fy_12c = -np.gradient(U_12c, dy, axis=0)

    # Imaginary (sine) cross-term
    cross_p_s = np.imag(cross_complex)
    cross_g_s = np.imag(grad_cross_complex)
    U_12s = coeff_p * cross_p_s - coeff_k * cross_g_s
    Fx_12s = -np.gradient(U_12s, dx, axis=1)
    Fy_12s = -np.gradient(U_12s, dy, axis=0)

    return (
        U_11, Fx_11, Fy_11,
        U_22, Fx_22, Fy_22,
        U_12c, Fx_12c, Fy_12c,
        U_12s, Fx_12s, Fy_12s,
    )


# ── Interpolator set ──────────────────────────────────────────────

def _make_interp(F: np.ndarray, xg: np.ndarray, yg: np.ndarray) -> RegularGridInterpolator:
    """Build a RegularGridInterpolator with zero fill outside domain."""
    return RegularGridInterpolator(
        (yg, xg), F, bounds_error=False, fill_value=0.0
    )


class TriBasisInterp:
    """
    Pre-built interpolators for all 12 basis fields.

    Allows fast evaluation of combined forces and analytic control
    gradients at arbitrary particle (x, y) positions.
    """

    def __init__(self, basis: TriBasis, xg: np.ndarray, yg: np.ndarray) -> None:
        (
            U_11, Fx_11, Fy_11,
            U_22, Fx_22, Fy_22,
            U_12c, Fx_12c, Fy_12c,
            U_12s, Fx_12s, Fy_12s,
        ) = basis

        self.iFx_11  = _make_interp(Fx_11, xg, yg)
        self.iFy_11  = _make_interp(Fy_11, xg, yg)
        self.iFx_22  = _make_interp(Fx_22, xg, yg)
        self.iFy_22  = _make_interp(Fy_22, xg, yg)
        self.iFx_12c = _make_interp(Fx_12c, xg, yg)
        self.iFy_12c = _make_interp(Fy_12c, xg, yg)
        self.iFx_12s = _make_interp(Fx_12s, xg, yg)
        self.iFy_12s = _make_interp(Fy_12s, xg, yg)

    def _pts(self, pos_xy: np.ndarray) -> np.ndarray:
        """Convert (n, 2) [x, y] to (n, 2) [y, x] for interpolator."""
        return np.column_stack([pos_xy[:, 1], pos_xy[:, 0]])

    # ── Component lookups ─────────────────────────────────────────

    def _lookup_all(self, pts: np.ndarray) -> Dict[str, np.ndarray]:
        """Evaluate all 8 force-component interpolators at pts (y, x)."""
        return {
            "Fx_11":  self.iFx_11(pts),
            "Fy_11":  self.iFy_11(pts),
            "Fx_22":  self.iFx_22(pts),
            "Fy_22":  self.iFy_22(pts),
            "Fx_12c": self.iFx_12c(pts),
            "Fy_12c": self.iFy_12c(pts),
            "Fx_12s": self.iFx_12s(pts),
            "Fy_12s": self.iFy_12s(pts),
        }

    # ── Combined force evaluation ─────────────────────────────────

    def eval_forces(
        self,
        beta: float,
        alpha: float,
        psi: float,
        pos_xy: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate total Gor'kov force (Fx, Fy) at particle positions.

        Parameters
        ----------
        beta : float   — standing-wave amplitude
        alpha : float  — vortex amplitude
        psi : float    — phase delay [rad]
        pos_xy : (n, 2) float  — particle [x, y] positions [m]

        Returns
        -------
        Fx, Fy : (n,) arrays — normalised Gor'kov force components
        """
        pts = self._pts(pos_xy)
        c = self._lookup_all(pts)
        cp = np.cos(psi)
        sp = np.sin(psi)
        Fx = (beta ** 2 * c["Fx_11"]
              + alpha ** 2 * c["Fx_22"]
              + 2 * beta * alpha * (cp * c["Fx_12c"] + sp * c["Fx_12s"]))
        Fy = (beta ** 2 * c["Fy_11"]
              + alpha ** 2 * c["Fy_22"]
              + 2 * beta * alpha * (cp * c["Fy_12c"] + sp * c["Fy_12s"]))
        return Fx, Fy

    # ── Analytic control gradients ────────────────────────────────

    def eval_dF_dcontrols(
        self,
        beta: float,
        alpha: float,
        psi: float,
        pos_xy: np.ndarray,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Analytic gradients ∂(Fx, Fy)/∂u for u ∈ {β, α, ψ}.

        Returns
        -------
        dict with keys "beta", "alpha", "psi", each mapping to
        (dFx_du, dFy_du) arrays of shape (n_particles,).
        """
        pts = self._pts(pos_xy)
        c = self._lookup_all(pts)
        cp = np.cos(psi)
        sp = np.sin(psi)

        cross_Fx = cp * c["Fx_12c"] + sp * c["Fx_12s"]
        cross_Fy = cp * c["Fy_12c"] + sp * c["Fy_12s"]

        # ∂F/∂β = 2β F₁₁ + 2α cross
        dFx_dbeta  = 2.0 * beta * c["Fx_11"] + 2.0 * alpha * cross_Fx
        dFy_dbeta  = 2.0 * beta * c["Fy_11"] + 2.0 * alpha * cross_Fy

        # ∂F/∂α = 2α F₂₂ + 2β cross
        dFx_dalpha = 2.0 * alpha * c["Fx_22"] + 2.0 * beta * cross_Fx
        dFy_dalpha = 2.0 * alpha * c["Fy_22"] + 2.0 * beta * cross_Fy

        # ∂F/∂ψ = 2βα [−sin ψ · F₁₂c + cos ψ · F₁₂s]
        coeff = 2.0 * beta * alpha
        dFx_dpsi = coeff * (-sp * c["Fx_12c"] + cp * c["Fx_12s"])
        dFy_dpsi = coeff * (-sp * c["Fy_12c"] + cp * c["Fy_12s"])

        return {
            "beta":  (dFx_dbeta,  dFy_dbeta),
            "alpha": (dFx_dalpha, dFy_dalpha),
            "psi":   (dFx_dpsi,   dFy_dpsi),
        }


def build_basis_for_vortex_position(
    p_sw: np.ndarray,
    vortex_gen,
    center_xy: np.ndarray,
    xg: np.ndarray,
    yg: np.ndarray,
) -> TriBasisInterp:
    """
    Build a TriBasisInterp for a given vortex centre position.

    Parameters
    ----------
    p_sw : complex (ny, nx) — standing-wave field
    vortex_gen : VortexPerturbation — provides .get_field(center_xy)
    center_xy : (2,) — vortex centre [x, y] in metres
    xg, yg : 1-D coordinate arrays
    """
    dx = float(xg[1] - xg[0])
    dy = float(yg[1] - yg[0])
    p_v = vortex_gen.get_field(np.asarray(center_xy, dtype=float))
    basis = precompute_trilinear_basis(p_sw, p_v, dx, dy)
    return TriBasisInterp(basis, xg, yg)
