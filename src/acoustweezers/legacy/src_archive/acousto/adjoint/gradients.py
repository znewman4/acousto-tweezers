# src/acousto/adjoint/gradients.py
"""
Adjoint gradient computations for acousto-tweezers.

This module implements the adjoint method for computing gradients of
scalar objectives with respect to control parameters (transducer amplitudes
and phases).

The forward problem is:
    A p = b(u)
    
where:
    - A is the Helmholtz operator (sparse matrix)
    - p is the complex pressure field (vector)
    - b(u) is the RHS depending on control u through bottom boundary velocity
    
The objective is J(p), a scalar function of the pressure field.

Adjoint gradient formula:
    dJ/du = Re(λ^H ∂b/∂u)
    
where λ solves the adjoint equation:
    A^H λ = ∂J/∂p^*   (conjugate of dJ/dp)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np


@dataclass
class TransducerParams:
    """Parameters for a single transducer."""
    x: float          # x position (m)
    y: float          # y position (m)
    v: float          # velocity amplitude (m/s)
    phi: float        # phase (rad)
    sigma_x: float    # Gaussian width in x (m)
    sigma_y: float    # Gaussian width for y coupling (m)
    gate: bool = True # enable flag


def compute_dJdp_pressure_at_point(
    ix: int,
    iy: int,
    Nx: int,
    Ny: int,
) -> np.ndarray:
    """
    Compute ∂J/∂p where J = |p(x_p, y_p)|² = p * conj(p) at grid point (ix, iy).
    
    dJ/dp[k] = conj(p)[k] if k = (iy, ix), else 0
    
    But for adjoint we need ∂J/∂p^* = p[k]
    
    Since J = |p|² = p * p^*, we have:
        ∂J/∂p = p^*   (treating p and p^* as independent)
        ∂J/∂p^* = p
    
    The adjoint equation uses ∂J/∂p^* as RHS.
    
    Returns
    -------
    dJdp_conj : np.ndarray, shape (Nx * Ny,)
        The vector ∂J/∂p^* = p at the target point, to be used as RHS for adjoint.
        
    Note: Caller must multiply by actual p[iy, ix] to get final RHS.
    """
    idx = iy * Nx + ix
    indicator = np.zeros(Nx * Ny, dtype=np.complex128)
    indicator[idx] = 1.0
    return indicator


def compute_dJdp_simple_real_pressure(
    ix: int,
    iy: int, 
    Nx: int,
    Ny: int,
) -> np.ndarray:
    """
    For J = Re(p) at point (ix, iy).
    
    J = (p + p^*) / 2
    ∂J/∂p = 1/2
    ∂J/∂p^* = 1/2
    
    Returns indicator * 0.5 for the adjoint RHS.
    """
    idx = iy * Nx + ix
    indicator = np.zeros(Nx * Ny, dtype=np.complex128)
    indicator[idx] = 0.5
    return indicator


def compute_dJdp_complex_pressure(
    ix: int,
    iy: int,
    Nx: int,
    Ny: int,
    p_val: complex,
) -> np.ndarray:
    """
    For J = |p|² at point (ix, iy), return ∂J/∂p = conj(p).
    
    Using Wirtinger calculus: J = p * conj(p), so ∂J/∂p = conj(p).
    
    Parameters
    ----------
    ix, iy : int
        Grid indices of the evaluation point.
    Nx, Ny : int
        Grid dimensions.
    p_val : complex
        Value of p at the evaluation point.
        
    Returns
    -------
    rhs : np.ndarray, shape (Nx * Ny,)
        Gradient ∂J/∂p = conj(p) at the target point.
    """
    idx = iy * Nx + ix
    rhs = np.zeros(Nx * Ny, dtype=np.complex128)
    rhs[idx] = np.conj(p_val)  # ∂J/∂p = p̄
    return rhs


def compute_dbdu_single_transducer(
    trans: TransducerParams,
    x: np.ndarray,
    omega: float,
    rho0: float,
    coupling_alpha: float,
    bottom_rows: np.ndarray,
    Nx: int,
    Ny: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute ∂b/∂v and ∂b/∂φ for a single transducer.
    
    The boundary velocity from one transducer is:
        vb[i] = v * exp(j*φ) * G_x(x[i] - x_t) * G_y(y_t)
    
    where G_x, G_y are Gaussians.
    
    The RHS contribution is:
        b[bottom_rows] = coupling_alpha * (-j ω ρ₀) * vb
    
    Derivatives:
        ∂vb/∂v = exp(j*φ) * G_x * G_y
        ∂vb/∂φ = j * v * exp(j*φ) * G_x * G_y = j * vb
        
        ∂b/∂v = coupling_alpha * (-j ω ρ₀) * ∂vb/∂v
        ∂b/∂φ = coupling_alpha * (-j ω ρ₀) * ∂vb/∂φ
    
    Parameters
    ----------
    trans : TransducerParams
        Transducer parameters.
    x : np.ndarray, shape (Nx,)
        X-coordinates of the grid.
    omega : float
        Angular frequency.
    rho0 : float
        Medium density.
    coupling_alpha : float
        Boundary coupling coefficient.
    bottom_rows : np.ndarray
        Row indices in RHS vector for bottom boundary.
    Nx, Ny : int
        Grid dimensions.
        
    Returns
    -------
    db_dv : np.ndarray, shape (Nx * Ny,)
        Gradient of RHS w.r.t. velocity amplitude.
    db_dphi : np.ndarray, shape (Nx * Ny,)
        Gradient of RHS w.r.t. phase.
    """
    if not trans.gate:
        # Transducer is gated off - zero gradient
        return np.zeros(Nx * Ny, dtype=np.complex128), np.zeros(Nx * Ny, dtype=np.complex128)
    
    # Gaussian footprint
    G_x = np.exp(-(x - trans.x)**2 / (2.0 * trans.sigma_x**2))
    G_y = np.exp(-(trans.y)**2 / (2.0 * trans.sigma_y**2))
    
    # Boundary velocity contribution
    phasor = np.exp(1j * trans.phi)
    vb = trans.v * phasor * G_x * G_y  # shape (Nx,)
    
    # Prefactor for boundary condition
    prefactor = coupling_alpha * (-1j * omega * rho0)
    
    # ∂vb/∂v = phasor * G_x * G_y
    dvb_dv = phasor * G_x * G_y
    
    # ∂vb/∂φ = j * vb
    dvb_dphi = 1j * vb
    
    # Full db vectors
    db_dv = np.zeros(Nx * Ny, dtype=np.complex128)
    db_dphi = np.zeros(Nx * Ny, dtype=np.complex128)
    
    db_dv[bottom_rows] = prefactor * dvb_dv
    db_dphi[bottom_rows] = prefactor * dvb_dphi
    
    return db_dv, db_dphi


def compute_dbdu_position(
    trans: TransducerParams,
    x: np.ndarray,
    omega: float,
    rho0: float,
    coupling_alpha: float,
    bottom_rows: np.ndarray,
    Nx: int,
    Ny: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute ∂b/∂x_t and ∂b/∂y_t for a single transducer.
    
    The boundary velocity from one transducer is:
        vb[i] = v * exp(j*φ) * G_x(x[i] - x_t) * G_y(y_t)
    
    where:
        G_x(d) = exp(-d² / (2 σ_x²))
        G_y(y) = exp(-y² / (2 σ_y²))
    
    Derivatives:
        ∂G_x/∂x_t = G_x * (x[i] - x_t) / σ_x²  (chain rule, minus sign from d = x - x_t)
        ∂G_y/∂y_t = G_y * (-y_t / σ_y²)
        
    Returns
    -------
    db_dx : np.ndarray, shape (Nx * Ny,)
        Gradient of RHS w.r.t. transducer x position.
    db_dy : np.ndarray, shape (Nx * Ny,)
        Gradient of RHS w.r.t. transducer y position.
    """
    if not trans.gate:
        return np.zeros(Nx * Ny, dtype=np.complex128), np.zeros(Nx * Ny, dtype=np.complex128)
    
    # Gaussian footprints
    d_x = x - trans.x
    G_x = np.exp(-d_x**2 / (2.0 * trans.sigma_x**2))
    G_y = np.exp(-(trans.y)**2 / (2.0 * trans.sigma_y**2))
    
    phasor = np.exp(1j * trans.phi)
    base = trans.v * phasor * G_y
    
    # Prefactor for boundary condition
    prefactor = coupling_alpha * (-1j * omega * rho0)
    
    # ∂G_x/∂x_t = G_x * (x - x_t) / σ_x²
    dGx_dxt = G_x * d_x / (trans.sigma_x**2)
    
    # ∂G_y/∂y_t = G_y * (-y_t) / σ_y²
    dGy_dyt = G_y * (-trans.y) / (trans.sigma_y**2)
    
    # ∂vb/∂x_t = v * phasor * ∂G_x/∂x_t * G_y
    dvb_dxt = base * dGx_dxt
    
    # ∂vb/∂y_t = v * phasor * G_x * ∂G_y/∂y_t
    dvb_dyt = trans.v * phasor * G_x * dGy_dyt
    
    db_dx = np.zeros(Nx * Ny, dtype=np.complex128)
    db_dy = np.zeros(Nx * Ny, dtype=np.complex128)
    
    db_dx[bottom_rows] = prefactor * dvb_dxt
    db_dy[bottom_rows] = prefactor * dvb_dyt
    
    return db_dx, db_dy


def adjoint_gradient(
    adjoint_solve: Callable[[np.ndarray], np.ndarray],
    dJ_dp: np.ndarray,
    db_du: np.ndarray,
) -> float:
    """
    Compute adjoint gradient dJ/du for a single control parameter.
    
    For a real-valued objective J(p) of a complex field satisfying A p = b(u):
        dJ/du = 2 Re(λ^T ∂b/∂u)
        
    where λ = (A^T)^{-1} (∂J/∂p)
    
    Parameters
    ----------
    adjoint_solve : callable
        Solves A^T λ = rhs and returns λ.
    dJ_dp : np.ndarray
        The sensitivity ∂J/∂p (gradient of objective w.r.t. state).
    db_du : np.ndarray
        Gradient of RHS w.r.t. control parameter.
        
    Returns
    -------
    grad : float
        The gradient dJ/du.
    """
    # Solve adjoint equation A^T λ = ∂J/∂p
    lam = adjoint_solve(dJ_dp)
    
    # Compute gradient: dJ/du = 2 Re(λ^T ∂b/∂u)
    # np.vdot(a, b) = sum(conj(a) * b), so we need np.dot instead
    grad = 2.0 * np.real(np.dot(lam, db_du))
    
    return grad


def adjoint_gradient_vectorized(
    adjoint_solve: Callable[[np.ndarray], np.ndarray],
    dJ_dp: np.ndarray,
    db_du_list: list[np.ndarray],
) -> np.ndarray:
    """
    Compute adjoint gradients for multiple control parameters.
    
    This is efficient because we only solve the adjoint equation once.
    
    For a real-valued objective J(p) of a complex field:
        dJ/du = 2 Re(λ^T ∂b/∂u) where A^T λ = ∂J/∂p
    
    Parameters
    ----------
    adjoint_solve : callable
        Solves A^T λ = rhs and returns λ.
    dJ_dp : np.ndarray
        The sensitivity ∂J/∂p (gradient of objective w.r.t. state).
    db_du_list : list of np.ndarray
        List of gradients ∂b/∂u_i for each control parameter.
        
    Returns
    -------
    grads : np.ndarray, shape (n_controls,)
        Gradient dJ/du for each control.
    """
    # Solve adjoint equation ONCE: A^T λ = ∂J/∂p
    lam = adjoint_solve(dJ_dp)
    
    # Compute all gradients with factor of 2
    # λ^T db/du = sum_i λ_i * (db/du)_i (no conjugation)
    grads = np.array([2.0 * np.real(np.dot(lam, db_du)) for db_du in db_du_list])
    
    return grads


def compute_dJdp_gorkov_potential(
    ix: int,
    iy: int,
    Nx: int,
    Ny: int,
    dx: float,
    dy: float,
    p: np.ndarray,
    omega: float,
    rho0: float,
    c0: float,
    particle_a: float,
    particle_rho_p: float,
    particle_c_p: float,
) -> np.ndarray:
    """
    Compute ∂J/∂p where J = U(x_p, y_p), the Gor'kov potential at grid point (ix, iy).
    
    The Gor'kov potential is:
        U = V * (f1 * E_pot - 1.5 * f2 * E_kin)
    
    where:
        E_pot = 0.25 * κ0 * |p|²
        E_kin = 0.25 * ρ0 * |v|² = 0.25 * |∇p|² / (ω² ρ0)
    
    The derivative ∂U/∂p involves:
        1. ∂E_pot/∂p = 0.25 * κ0 * conj(p)  at the particle point
        2. ∂E_kin/∂p involves the gradient stencil (neighbors contribute)
    
    For np.gradient with edge_order=2, the stencil at interior points is central difference:
        ∂p/∂x ≈ (p[i+1] - p[i-1]) / (2*dx)
    
    So |∇p|² = |∂p/∂x|² + |∂p/∂y|², and we need to differentiate this w.r.t. each p[j,i].
    
    Parameters
    ----------
    ix, iy : int
        Grid indices of the particle position.
    Nx, Ny : int  
        Grid dimensions.
    dx, dy : float
        Grid spacing.
    p : np.ndarray, shape (Ny, Nx)
        Complex pressure field.
    omega : float
        Angular frequency.
    rho0 : float
        Fluid density.
    c0 : float
        Fluid sound speed.
    particle_a, particle_rho_p, particle_c_p : float
        Particle radius, density, and sound speed.
        
    Returns
    -------
    dJ_dp : np.ndarray, shape (Nx * Ny,)
        Gradient ∂U/∂p as a flattened vector.
    """
    # Material constants
    kappa0 = 1.0 / (rho0 * c0**2)
    kappap = 1.0 / (particle_rho_p * particle_c_p**2)
    f1 = 1.0 - (kappap / kappa0)
    f2 = 2.0 * (particle_rho_p - rho0) / (2.0 * particle_rho_p + rho0)
    V = (4.0 / 3.0) * np.pi * (particle_a ** 3)
    
    # Prefactors
    # E_pot = 0.25 * kappa0 * |p|²  =>  ∂E_pot/∂p = 0.25 * kappa0 * conj(p)
    # E_kin = 0.25 * rho0 * |v|² where v = ∇p / (i*ω*ρ0)
    #       = 0.25 * rho0 * |∇p|² / (ω² * ρ0²)
    #       = 0.25 * |∇p|² / (ω² * ρ0)
    # So ∂E_kin/∂p requires differentiating |∇p|² w.r.t. p
    
    coef_pot = V * f1 * 0.25 * kappa0
    coef_kin = V * (-1.5) * f2 * 0.25 / (omega**2 * rho0)
    
    # Initialize gradient vector
    dJ_dp = np.zeros(Nx * Ny, dtype=np.complex128)
    
    def idx(j: int, i: int) -> int:
        return j * Nx + i
    
    # --- Contribution from E_pot term ---
    # ∂(|p|²)/∂p at (iy, ix) = conj(p[iy, ix])
    dJ_dp[idx(iy, ix)] += coef_pot * np.conj(p[iy, ix])
    
    # --- Contribution from E_kin term ---
    # |∇p|² = |∂p/∂x|² + |∂p/∂y|²
    #
    # Using central differences at interior points:
    #   ∂p/∂x at (iy, ix) ≈ (p[iy, ix+1] - p[iy, ix-1]) / (2*dx)
    #   ∂p/∂y at (iy, ix) ≈ (p[iy+1, ix] - p[iy-1, ix]) / (2*dy)
    #
    # Let gx = ∂p/∂x, gy = ∂p/∂y at the particle point
    # |∇p|² = gx * conj(gx) + gy * conj(gy)
    #
    # The Wirtinger derivative ∂(|g|²)/∂p_k = conj(g) * (∂g/∂p_k)
    # where ∂gx/∂p_k depends on the stencil
    
    # Compute gradients at particle point using central differences
    # (matching np.gradient behavior at interior points)
    if 1 <= ix <= Nx - 2:
        gx = (p[iy, ix + 1] - p[iy, ix - 1]) / (2.0 * dx)
        dgx_dp_ip1 = 1.0 / (2.0 * dx)   # ∂gx/∂p[iy, ix+1]
        dgx_dp_im1 = -1.0 / (2.0 * dx)  # ∂gx/∂p[iy, ix-1]
    elif ix == 0:
        # Forward difference at left boundary
        gx = (p[iy, ix + 1] - p[iy, ix]) / dx
        dgx_dp_ip1 = 1.0 / dx
        dgx_dp_i = -1.0 / dx
        dgx_dp_im1 = 0.0
    else:  # ix == Nx - 1
        # Backward difference at right boundary
        gx = (p[iy, ix] - p[iy, ix - 1]) / dx
        dgx_dp_i = 1.0 / dx
        dgx_dp_im1 = -1.0 / dx
        dgx_dp_ip1 = 0.0
    
    if 1 <= iy <= Ny - 2:
        gy = (p[iy + 1, ix] - p[iy - 1, ix]) / (2.0 * dy)
        dgy_dp_jp1 = 1.0 / (2.0 * dy)   # ∂gy/∂p[iy+1, ix]
        dgy_dp_jm1 = -1.0 / (2.0 * dy)  # ∂gy/∂p[iy-1, ix]
    elif iy == 0:
        # Forward difference at bottom boundary
        gy = (p[iy + 1, ix] - p[iy, ix]) / dy
        dgy_dp_jp1 = 1.0 / dy
        dgy_dp_j = -1.0 / dy
        dgy_dp_jm1 = 0.0
    else:  # iy == Ny - 1
        # Backward difference at top boundary
        gy = (p[iy, ix] - p[iy - 1, ix]) / dy
        dgy_dp_j = 1.0 / dy
        dgy_dp_jm1 = -1.0 / dy
        dgy_dp_jp1 = 0.0
    
    # ∂(|gx|²)/∂p_k = conj(gx) * ∂gx/∂p_k
    # ∂(|gy|²)/∂p_k = conj(gy) * ∂gy/∂p_k
    gx_conj = np.conj(gx)
    gy_conj = np.conj(gy)
    
    # Add contributions from x-gradient stencil
    if 1 <= ix <= Nx - 2:
        # Interior: central difference
        dJ_dp[idx(iy, ix + 1)] += coef_kin * gx_conj * dgx_dp_ip1
        dJ_dp[idx(iy, ix - 1)] += coef_kin * gx_conj * dgx_dp_im1
    elif ix == 0:
        dJ_dp[idx(iy, ix + 1)] += coef_kin * gx_conj * dgx_dp_ip1
        dJ_dp[idx(iy, ix)] += coef_kin * gx_conj * dgx_dp_i
    else:  # ix == Nx - 1
        dJ_dp[idx(iy, ix)] += coef_kin * gx_conj * dgx_dp_i
        dJ_dp[idx(iy, ix - 1)] += coef_kin * gx_conj * dgx_dp_im1
    
    # Add contributions from y-gradient stencil
    if 1 <= iy <= Ny - 2:
        # Interior: central difference
        dJ_dp[idx(iy + 1, ix)] += coef_kin * gy_conj * dgy_dp_jp1
        dJ_dp[idx(iy - 1, ix)] += coef_kin * gy_conj * dgy_dp_jm1
    elif iy == 0:
        dJ_dp[idx(iy + 1, ix)] += coef_kin * gy_conj * dgy_dp_jp1
        dJ_dp[idx(iy, ix)] += coef_kin * gy_conj * dgy_dp_j
    else:  # iy == Ny - 1
        dJ_dp[idx(iy, ix)] += coef_kin * gy_conj * dgy_dp_j
        dJ_dp[idx(iy - 1, ix)] += coef_kin * gy_conj * dgy_dp_jm1
    
    return dJ_dp
