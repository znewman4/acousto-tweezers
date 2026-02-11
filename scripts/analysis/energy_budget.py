#!/usr/bin/env python3
"""
Energy Budget and Dissipation Diagnostics

Computes order-of-magnitude energy scales:
- Domain-integrated viscous dissipation
- Acoustic intensity proxy (max/median)
- Energy ratio: dissipation / acoustic energy

This catches silent nonsense early and provides physical context.

Usage:
    from scripts.diagnostics.energy_budget import compute_energy_budget
    budget = compute_energy_budget(p_solution, streaming_solution, cfg)

Author: Acousto-Tweezers Project
Date: 2026-02-09
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any

from dolfinx import fem
import ufl

from acoustweezers.experiments.shallow_square_dish.config import ShallowDishConfig
from acoustweezers.experiments.shallow_square_dish.solve_pressure import PressureSolution
from acoustweezers.experiments.shallow_square_dish.streaming import StreamingSolution


def compute_energy_budget(
    p_solution: PressureSolution,
    streaming_solution: StreamingSolution,
    cfg: ShallowDishConfig,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Compute energy budget and dissipation diagnostics.
    
    Parameters
    ----------
    p_solution : PressureSolution
        Pressure solution
    streaming_solution : StreamingSolution
        Streaming solution
    cfg : ShallowDishConfig
        Configuration
    verbose : bool
        Print diagnostics
        
    Returns
    -------
    budget : dict
        Energy budget with keys:
        - viscous_dissipation_W: Total viscous dissipation [W]
        - dissipation_density_max_W_per_m3: Max dissipation density
        - dissipation_density_median_W_per_m3: Median dissipation density
        - acoustic_intensity_max_W_per_m2: Max acoustic intensity proxy
        - acoustic_intensity_median_W_per_m2: Median acoustic intensity
        - acoustic_energy_density_max_J_per_m3: Max energy density
        - acoustic_energy_density_median_J_per_m3: Median energy density
        - dissipation_to_acoustic_ratio: Dissipation / (acoustic power scale)
    """
    if verbose:
        print("\n" + "="*70)
        print("ENERGY BUDGET DIAGNOSTICS")
        print("="*70)
    
    rho = cfg.rho
    mu = cfg.mu
    omega = cfg.omega
    c = cfg.c
    
    domain = p_solution.p_function.function_space.mesh
    
    # =========================================================================
    # 1. VISCOUS DISSIPATION
    # =========================================================================
    # Viscous dissipation rate density: Φ = μ |∇u + (∇u)ᵀ|²
    # For incompressible flow: Φ = 2μ |ε(u)|² where ε = (∇u + ∇uᵀ)/2
    
    u_func = streaming_solution.u_function
    
    # Strain rate tensor ε = (∇u + ∇uᵀ)/2
    grad_u = ufl.grad(u_func)
    epsilon = (grad_u + grad_u.T) / 2
    
    # Dissipation density: Φ = 2μ ε:ε = 2μ |ε|²
    dissipation_density = 2 * mu * ufl.inner(epsilon, epsilon)
    
    # Integrate over domain
    total_dissipation = fem.assemble_scalar(fem.form(dissipation_density * ufl.dx))
    
    # Also get point-wise values for statistics
    V_scalar = fem.functionspace(domain, ("Lagrange", 1))
    dissip_func = fem.Function(V_scalar)
    dissip_expr = fem.Expression(
        ufl.real(dissipation_density),
        V_scalar.element.interpolation_points()
    )
    dissip_func.interpolate(dissip_expr)
    dissip_vals = np.real(dissip_func.x.array)
    
    dissip_max = np.max(dissip_vals)
    dissip_median = np.median(dissip_vals)
    
    if verbose:
        print(f"\n  Viscous Dissipation:")
        print(f"    Total: {total_dissipation:.2e} W")
        print(f"    Max density: {dissip_max:.2e} W/m³")
        print(f"    Median density: {dissip_median:.2e} W/m³")
    
    # =========================================================================
    # 2. ACOUSTIC INTENSITY PROXY
    # =========================================================================
    # Time-averaged acoustic intensity: I = (1/2) Re(p v₁*)
    # For plane wave: I ≈ |p|²/(2ρc)
    # Use this as order-of-magnitude proxy
    
    p_vals = p_solution.p_values
    p_abs_sq = np.abs(p_vals)**2
    
    intensity_proxy = p_abs_sq / (2 * rho * c)
    
    intensity_max = np.max(intensity_proxy)
    intensity_median = np.median(intensity_proxy)
    
    if verbose:
        print(f"\n  Acoustic Intensity (plane-wave proxy):")
        print(f"    Max: {intensity_max:.2e} W/m²")
        print(f"    Median: {intensity_median:.2e} W/m²")
    
    # =========================================================================
    # 3. ACOUSTIC ENERGY DENSITY
    # =========================================================================
    # Time-averaged energy density: E = (1/2) (|p|²/K + ρ|v₁|²)
    # where K = ρc² is bulk modulus
    # For plane wave: E ≈ |p|²/(ρc²)
    
    K = cfg.fluid_bulk_modulus
    energy_density_proxy = p_abs_sq / (2 * K)
    
    energy_max = np.max(energy_density_proxy)
    energy_median = np.median(energy_density_proxy)
    
    if verbose:
        print(f"\n  Acoustic Energy Density (plane-wave proxy):")
        print(f"    Max: {energy_max:.2e} J/m³")
        print(f"    Median: {energy_median:.2e} J/m³")
    
    # =========================================================================
    # 4. DISSIPATION TO ACOUSTIC RATIO
    # =========================================================================
    # Compare dissipation to acoustic power scale
    # Acoustic power scale ≈ intensity × area ≈ I_max × L²
    
    L = cfg.L
    acoustic_power_scale = intensity_max * L**2
    
    if acoustic_power_scale > 0:
        dissip_ratio = total_dissipation / acoustic_power_scale
    else:
        dissip_ratio = np.inf
    
    if verbose:
        print(f"\n  Energy Ratios:")
        print(f"    Acoustic power scale: {acoustic_power_scale:.2e} W")
        print(f"    Dissipation / Acoustic: {dissip_ratio:.2e}")
        
        # Sanity check
        if dissip_ratio > 1.0:
            print(f"    ⚠ WARNING: Dissipation > acoustic power (unphysical)")
        elif dissip_ratio < 1e-6:
            print(f"    ⚠ WARNING: Dissipation too small (numerical issue?)")
        else:
            print(f"    ✓ Reasonable dissipation scale")
    
    # =========================================================================
    # 5. STREAMING REYNOLDS NUMBER
    # =========================================================================
    u_vals = streaming_solution.u_function.x.array
    n_u = len(u_vals) // 3
    u_mag = np.linalg.norm(np.real(u_vals).reshape((n_u, 3)), axis=1)
    
    u_max = np.max(u_mag)
    u_median = np.median(u_mag)
    
    L_char = cfg.H  # Use depth as characteristic length
    Re_streaming = rho * u_max * L_char / mu
    
    if verbose:
        print(f"\n  Streaming Velocity:")
        print(f"    Max: {u_max*1e6:.2f} μm/s")
        print(f"    Median: {u_median*1e6:.2f} μm/s")
        print(f"    Re_streaming: {Re_streaming:.2e}")
        
        if Re_streaming > 1:
            print(f"    ⚠ NOTE: Re > 1, inertial effects may matter")
        else:
            print(f"    ✓ Re << 1, Stokes approximation valid")
    
    if verbose:
        print("="*70 + "\n")
    
    return {
        'viscous_dissipation_W': float(total_dissipation),
        'dissipation_density_max_W_per_m3': float(dissip_max),
        'dissipation_density_median_W_per_m3': float(dissip_median),
        'acoustic_intensity_max_W_per_m2': float(intensity_max),
        'acoustic_intensity_median_W_per_m2': float(intensity_median),
        'acoustic_energy_density_max_J_per_m3': float(energy_max),
        'acoustic_energy_density_median_J_per_m3': float(energy_median),
        'acoustic_power_scale_W': float(acoustic_power_scale),
        'dissipation_to_acoustic_ratio': float(dissip_ratio),
        'streaming_velocity_max_m_per_s': float(u_max),
        'streaming_velocity_median_m_per_s': float(u_median),
        'Re_streaming': float(Re_streaming),
    }


def export_dissipation_field(
    streaming_solution: StreamingSolution,
    cfg: ShallowDishConfig,
    output_path: str,
) -> None:
    """
    Export dissipation density field for visualization.
    
    Parameters
    ----------
    streaming_solution : StreamingSolution
        Streaming solution
    cfg : ShallowDishConfig
        Configuration
    output_path : str
        Output file path (e.g., "dissipation.bp")
    """
    from dolfinx import io
    from mpi4py import MPI
    
    domain = streaming_solution.u_function.function_space.mesh
    u_func = streaming_solution.u_function
    mu = cfg.mu
    
    # Strain rate tensor
    grad_u = ufl.grad(u_func)
    epsilon = (grad_u + grad_u.T) / 2
    
    # Dissipation density
    dissipation_density = 2 * mu * ufl.inner(epsilon, epsilon)
    
    # Project to scalar function
    V_scalar = fem.functionspace(domain, ("Lagrange", 1))
    dissip_func = fem.Function(V_scalar, name="dissipation_density")
    dissip_expr = fem.Expression(
        ufl.real(dissipation_density),
        V_scalar.element.interpolation_points()
    )
    dissip_func.interpolate(dissip_expr)
    
    # Write to file
    with io.VTXWriter(MPI.COMM_WORLD, output_path, [dissip_func], engine="BP4") as vtx:
        vtx.write(0.0)
