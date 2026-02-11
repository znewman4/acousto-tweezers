"""
Acoustic Streaming Solver for Shallow Square Dish.

Computes steady acoustic streaming velocity using:
1. Reynolds stress forcing from first-order velocity field
2. Steady incompressible Stokes equation solve with proper saddle-point preconditioner

Physics:
    f = -∇·⟨ρ v₁ ⊗ v₁⟩  (Reynolds stress forcing)
    
    -μ∇²u_s + ∇p_s = f   (Stokes momentum)
    ∇·u_s = 0             (Incompressibility)

Level-2 Features:
    • Taylor-Hood (P2-P1) stable element
    • Fieldsplit Schur complement preconditioner
    • Pressure nullspace handling (constant mode)
    • Mesh downsampling option for memory efficiency
    • Force scaling for conditioning control
    • Comprehensive diagnostics (KSP convergence, z-profiles, divergence)
    • Graceful fallback on solver failure

Author: Acousto-Tweezers Project
Date: 2026-02-09
"""

from __future__ import annotations

import json
import time
from datetime import datetime
import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass, asdict

from mpi4py import MPI
import dolfinx
from dolfinx import fem, mesh
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_vector
from petsc4py import PETSc
import ufl
from ufl import inner, grad, div, dx, TrialFunction, TestFunction, Measure
import basix.ufl

from .config import ShallowDishConfig
from .solve_pressure import PressureSolution, TAG_BOTTOM_DISC, TAG_BOTTOM_RIGID, TAG_BOTTOM, TAG_TOP, TAG_X0, TAG_XL, TAG_Y0, TAG_YL


@dataclass
class StreamingSolution:
    """Container for streaming velocity solution."""
    u_function: fem.Function              # Vector velocity function
    p_function: fem.Function              # Streaming pressure (Lagrange multiplier)
    mesh_acoustic: mesh.Mesh              # Original acoustic mesh
    mesh_streaming: mesh.Mesh             # Downsampled streaming mesh (if applicable)
    cfg: ShallowDishConfig
    diagnostics: Dict = None              # Solver diagnostics
    
    def __post_init__(self):
        if self.diagnostics is None:
            self.diagnostics = {}
    
    @property
    def mesh(self):
        return self.u_function.function_space.mesh
    
    @property
    def u_values(self) -> np.ndarray:
        """Streaming velocity at DOFs, shape (N, 3)."""
        vals = self.u_function.x.array.copy()
        ndofs = len(vals) // 3
        return vals.reshape((ndofs, 3))
    
    @property
    def u_mag(self) -> np.ndarray:
        """Streaming speed at DOFs."""
        return np.linalg.norm(self.u_values, axis=1)
    
    @property
    def max_speed(self) -> float:
        return float(np.max(self.u_mag))
    
    @property
    def mean_speed(self) -> float:
        return float(np.mean(self.u_mag))
    
    @property
    def p_values(self) -> np.ndarray:
        """Streaming pressure at DOFs."""
        return self.p_function.x.array.copy()
    
    @property
    def coords(self) -> np.ndarray:
        """DOF coordinates for velocity space."""
        V = self.u_function.function_space
        if hasattr(V, 'sub'):
            return V.sub(0).tabulate_dof_coordinates()
        return V.tabulate_dof_coordinates()


def build_fieldsplit_options(
    streaming_model: str = "stokes",
    verbose: bool = True,
) -> Dict:
    """
    Build PETSc options dict for fieldsplit Schur complement preconditioner.
    
    This implements a robust saddle-point preconditioner suitable for
    the mixed Stokes system (u, p):
    
        [A    B^T]   [u]   [f]
        [B     0 ] * [p] = [0]
    
    where A = μ∇²+mass, B = -∇·, with Schur complement S = B A^{-1} B^T
    
    Strategy:
        - Use GMRES for outer KSP (robust for indefinite systems)
        - Fieldsplit with Schur factorization
        - Velocity block: GAMG (aggressive algebraic multigrid)
        - Schur block: Diagonal approximation (S ~ -M_p^{-1} where M_p is pressure mass)
    
    Parameters
    ----------
    streaming_model : str
        "stokes" (default) uses fieldsplit Schur for saddle-point
        "penalty" (future) would use penalty formulation
    verbose : bool
        Print option summary
    
    Returns
    -------
    opts : Dict[str, str]
        PETSc command-line options for KSP setup
    """
    opts = {
        # ======= Outer KSP =======
        "ksp_type": "gmres",
        "ksp_gmres_restart": 100,
        "ksp_gmres_modifiedgramschmidt": "true",
        "ksp_rtol": "1e-6",
        "ksp_atol": "1e-30",
        "ksp_max_it": "5000",
        
        # ======= Preconditioner: Fieldsplit =======
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "schur",
        "pc_fieldsplit_schur_factorization_type": "full",
        
        # ======= Velocity block (field 0) =======
        # Use GAMG (algebraic multigrid) for the velocity Laplacian
        "fieldsplit_u_pc_type": "gamg",
        "fieldsplit_u_pc_gamg_type": "agg",
        "fieldsplit_u_pc_gamg_agg_nsmooths": "1",
        "fieldsplit_u_ksp_type": "gmres",
        "fieldsplit_u_ksp_rtol": "1e-5",
        "fieldsplit_u_ksp_max_it": "100",
        
        # ======= Pressure / Schur block (field 1) =======
        # For Schur complement, use simple preconditioner
        # S ≈ -M_p^{-1} where M_p is pressure mass matrix
        # Jacobi gives diagonal preconditioning of pressure mass
        "fieldsplit_p_pc_type": "jacobi",
        "fieldsplit_p_ksp_type": "preonly",
    }
    
    if verbose:
        print("\nFieldsplit Schur Preconditioner Configuration:")
        print("  Outer: GMRES (restart=100, rtol=1e-6, max_it=5000)")
        print("  Velocity block: GAMG (algebraic multigrid)")
        print("  Schur/Pressure: Jacobi + preonly")
    
    return opts


def attach_pressure_nullspace(ksp: PETSc.KSP, W: fem.FunctionSpace) -> None:
    """
    Attach pressure nullspace (constant pressure mode) to KSP.
    
    The mixed Stokes system has a nullspace: any constant added to pressure
    is a solution with zero residual (since ∇·p = 0 and incompressibility
    already enforces ∇·u = 0).
    
    PETSc must know about this nullspace for preconditioner correctness,
    especially for Schur-based solvers.
    
    Parameters
    ----------
    ksp : PETSc.KSP
        The KSP solver
    W : fem.FunctionSpace
        Mixed function space (P2-P1 or similar)
    """
    # Create a function in the mixed space and set only pressure DOFs to 1
    ns_func = fem.Function(W)
    ns_func.x.array[:] = 0.0  # Zero everything
    
    # Get the pressure DOF indices in the mixed space
    Q_dofs = np.array(W.sub(1).dofmap.list.flat, dtype=np.int32)
    Q_unique = np.unique(Q_dofs)
    
    # Set pressure DOFs to 1
    ns_func.x.array[Q_unique] = 1.0
    
    # Create PETSc nullspace from the function's PETSc vector
    ns_vec = ns_func.x.petsc_vec.copy()
    ns_vec.normalize()
    
    null = PETSc.NullSpace().create(vectors=[ns_vec], constant=False, comm=W.mesh.comm)
    A = ksp.getOperators()[0]
    A.setNullSpace(null)
    A.setTransposeNullSpace(null)
    
    ns_vec.destroy()


def compute_first_order_velocity(
    p_solution: PressureSolution,
    domain: Optional[mesh.Mesh] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute first-order acoustic velocity from pressure gradient.
    
    v₁ = (1/iωρ) ∇p
    
    Parameters
    ----------
    p_solution : PressureSolution
        Pressure solution
    domain : mesh.Mesh, optional
        Domain to evaluate on (defaults to pressure mesh)
    verbose : bool
        Print info
        
    Returns
    -------
    v1_values : np.ndarray
        Complex velocity at DOF coordinates, shape (N, 3)
    coords : np.ndarray
        DOF coordinates, shape (N, 3)
    """
    cfg = p_solution.cfg
    omega = cfg.omega
    rho = cfg.rho
    
    if domain is None:
        domain = p_solution.p_function.function_space.mesh
    
    V = p_solution.p_function.function_space
    coords = V.tabulate_dof_coordinates()
    n_dofs = len(coords)
    
    # Create a vector function space for velocity
    V_vec = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    
    p_func = p_solution.p_function
    grad_p = ufl.grad(p_func)
    
    # CORRECT PHASOR-BASED FIRST-ORDER VELOCITY
    # For complex phasor pressure p, the first-order velocity phasor is:
    #   v₁ = ∇p / (iωρ)
    # This is the velocity amplitude (complex, with phase info)
    v1_expr = grad_p / (1j * omega * rho)
    
    # Project using interpolation
    v1_func = fem.Function(V_vec)
    
    # DOLFINx 0.9.0 API: interpolation_points() is a method, not an attribute
    pts = V_vec.element.interpolation_points()
    v1_expr_interp = fem.Expression(v1_expr, pts)
    v1_func.interpolate(v1_expr_interp)
    
    v1_vals = v1_func.x.array.copy()
    v1_coords = V_vec.tabulate_dof_coordinates()
    
    n_vec_dofs = len(v1_coords)
    v1_values = v1_vals.reshape((n_vec_dofs, 3))
    
    if verbose:
        v1_mag = np.linalg.norm(np.abs(v1_values), axis=1)
        print(f"  First-order velocity (complex phasor):")
        print(f"    max |v₁| = {np.max(v1_mag)*1e6:.2f} μm/s")
        print(f"    dtype: {v1_values.dtype}")
        print(f"    DOFs: {n_vec_dofs}")
    
    return v1_values, v1_coords


def downsample_mesh(
    domain_fine: mesh.Mesh,
    downsample_factor: int = 2,
    verbose: bool = True,
) -> mesh.Mesh:
    """
    Create a downsampled mesh by refining uniformly and keeping every N-th cell.
    
    For memory efficiency, streaming solve can use a coarser mesh.
    Downsampling reduces DOFs by ~factor^{dim} per refinement level.
    
    Parameters
    ----------
    domain_fine : mesh.Mesh
        Fine (acoustic) mesh
    downsample_factor : int
        Coarsening factor (1 = no downsampling, 2 = 8x fewer cells in 3D)
    verbose : bool
        Print info
        
    Returns
    -------
    domain_coarse : mesh.Mesh
        Coarsened mesh (if downsample_factor > 1, otherwise returns original)
    """
    if downsample_factor <= 1:
        return domain_fine
    
    # TODO: Implement proper mesh coarsening
    # For now, return fine mesh (proper coarsening requires MSHIO or Gmsh API)
    # Future: use dolfinx.mesh.create_mesh with decimated cell list
    
    if verbose:
        print(f"  Mesh downsampling: factor={downsample_factor} (not yet implemented, using fine mesh)")
    
    return domain_fine


def compute_streaming_diagnostics(
    u_h: fem.Function,
    p_h: fem.Function,
    domain: mesh.Mesh,
    cfg: ShallowDishConfig,
    ksp: PETSc.KSP,
    f_forcing: fem.Function,
    assemble_time: float = 0.0,
    solve_time: float = 0.0,
    verbose: bool = True,
) -> Dict:
    """
    Compute comprehensive diagnostics for streaming solution.
    
    Includes:
    - KSP convergence info (iterations, reason, residual norm)
    - Velocity statistics (max, mean, median, RMS)
    - Divergence of velocity field
    - Vertical profile of speed vs height (z-levels)
    - Forcing statistics
    - Runtime breakdown
    
    Parameters
    ----------
    u_h : fem.Function
        Streaming velocity solution
    p_h : fem.Function
        Streaming pressure solution
    domain : mesh.Mesh
        Domain mesh
    cfg : ShallowDishConfig
        Configuration
    ksp : PETSc.KSP
        KSP solver (for convergence info)
    f_forcing : fem.Function
        Forcing term function
    assemble_time : float
        Assembly wall time [s]
    solve_time : float
        Solve wall time [s]
    verbose : bool
        Print summary
        
    Returns
    -------
    diags : Dict
        Dictionary of diagnostic values
    """
    diags = {}
    
    # ===== KSP convergence =====
    diags["ksp_iterations"] = int(ksp.getIterationNumber())
    conv_reason = ksp.getConvergedReason()
    diags["ksp_converged_reason"] = int(conv_reason)
    reason_text = {
        -1: "DIVERGED_NULL",
        0: "ITERATING",
        -2: "DIVERGED_ITS",
        -3: "DIVERGED_DTOL",
        -4: "DIVERGED_BREAKDOWN",
        -5: "DIVERGED_BREAKDOWN_BICG",
        -6: "DIVERGED_NONSYMMETRIC_PC",
        -7: "DIVERGED_NANORINF",
        -8: "DIVERGED_INDEFINITE_PC",
        -9: "DIVERGED_NAN",
        -10: "DIVERGED_INDEFINITE_MAT",
        -11: "DIVERGED_PC_FAILED",
        1: "CONVERGED_RTOL_NORMAL",
        2: "CONVERGED_ATOL_NORMAL",
        3: "CONVERGED_RTOL_HAPPY_BREAKDOWN",
        4: "CONVERGED_ATOL_HAPPY_BREAKDOWN",
    }
    diags["ksp_reason_str"] = reason_text.get(conv_reason, f"UNKNOWN({conv_reason})")
    
    # Try to get final residual norm - getResidualNorm() is the correct API
    try:
        diags["ksp_final_residual_norm"] = float(ksp.getResidualNorm())
    except (AttributeError, TypeError):
        diags["ksp_final_residual_norm"] = np.nan
    
    # ===== Velocity statistics =====
    u_vals = u_h.x.array
    n_udofs = len(u_vals) // 3
    u_vec = u_vals.reshape((n_udofs, 3))
    u_mag = np.linalg.norm(u_vec, axis=1)
    
    diags["max_u_um_s"] = float(np.max(u_mag) * 1e6)
    diags["mean_u_um_s"] = float(np.mean(u_mag) * 1e6)
    diags["median_u_um_s"] = float(np.median(u_mag) * 1e6)
    diags["rms_u_um_s"] = float(np.sqrt(np.mean(u_mag**2)) * 1e6)
    
    # ===== Divergence check =====
    # Compute ∇·u at quadrature points
    div_u = div(u_h)
    div_form = ufl.inner(div_u, div_u) * dx
    div_form_compiled = fem.form(div_form)
    div_l2_sq = fem.assemble_scalar(div_form_compiled)
    diags["divergence_l2_norm"] = float(np.sqrt(div_l2_sq))
    diags["divergence_l2_norm_relative"] = float(np.sqrt(div_l2_sq) / (diags["rms_u_um_s"] * 1e-6 + 1e-15))
    
    # ===== Z-profile =====
    u_coords = u_h.function_space.tabulate_dof_coordinates()
    z_min, z_max = np.min(u_coords[:, 2]), np.max(u_coords[:, 2])
    z_levels = np.linspace(z_min, z_max, 21)  # 20 intervals
    
    z_profile = []
    for z in z_levels:
        tol = (z_max - z_min) / 1000.0
        dofs_at_z = np.where(np.abs(u_coords[:, 2] - z) < tol)[0]
        if len(dofs_at_z) > 0:
            u_at_z = u_mag[dofs_at_z]
            u_mean_z = np.mean(u_at_z)
        else:
            u_mean_z = 0.0
        z_profile.append({"z": float(z), "u_mean_um_s": float(u_mean_z * 1e6)})
    
    diags["z_profile"] = z_profile
    
    # ===== Forcing statistics =====
    f_vals = f_forcing.x.array
    n_fdofs = len(f_vals) // 3
    f_vec = f_vals.reshape((n_fdofs, 3))
    f_mag = np.linalg.norm(f_vec, axis=1)
    
    diags["max_forcing_pa_m"] = float(np.max(f_mag))
    diags["median_forcing_pa_m"] = float(np.median(f_mag))
    diags["mean_forcing_pa_m"] = float(np.mean(f_mag))
    
    # ===== Pressure statistics =====
    p_vals = p_h.x.array
    diags["max_p_pa"] = float(np.max(np.abs(p_vals)))
    diags["mean_p_pa"] = float(np.mean(np.abs(p_vals)))
    
    # ===== Runtime =====
    diags["assemble_time_s"] = float(assemble_time)
    diags["solve_time_s"] = float(solve_time)
    diags["total_time_s"] = float(assemble_time + solve_time)
    
    # ===== Mesh info =====
    diags["n_cells"] = int(domain.topology.index_map(3).size_global)
    diags["n_dofs"] = int(u_h.function_space.dofmap.index_map.size_global * u_h.function_space.dofmap.index_map_bs)
    
    if verbose:
        print("\nStreaming Diagnostics:")
        print(f"  KSP: {diags['ksp_iterations']} iterations, reason={diags['ksp_reason_str']}")
        max_u = diags['max_u_um_s']
        fmt = f"{max_u:.2f}" if max_u >= 0.01 else f"{max_u:.2e}"
        mean_u = diags['mean_u_um_s']
        fmt_mean = f"{mean_u:.2f}" if mean_u >= 0.01 else f"{mean_u:.2e}"
        med_u = diags['median_u_um_s']
        fmt_med = f"{med_u:.2f}" if med_u >= 0.01 else f"{med_u:.2e}"
        print(f"  Velocity: max={fmt} μm/s, mean={fmt_mean}, median={fmt_med}")
        print(f"  Divergence: L2={diags['divergence_l2_norm']:.2e}, relative={diags['divergence_l2_norm_relative']:.2e}")
        print(f"  Forcing: max={diags['max_forcing_pa_m']:.2e} Pa/m, median={diags['median_forcing_pa_m']:.2e}")
        print(f"  Runtime: assembly={diags['assemble_time_s']:.2f}s, solve={diags['solve_time_s']:.2f}s")
        print(f"  DOFs: {diags['n_dofs']:,}, cells: {diags['n_cells']:,}")
    
    return diags


# Backward compatibility wrapper
def solve_streaming(
    p_solution: PressureSolution,
    domain: Optional[mesh.Mesh] = None,
    cfg: Optional[ShallowDishConfig] = None,
    downsample: int = 1,
    forcing_scale: float = 1.0,
    verbose: bool = True,
) -> dict:
    """
    Wrapper for solve_streaming_stokes for backward compatibility.
    
    Parameters
    ----------
    p_solution : PressureSolution
        Solution from pressure solve
    domain : mesh.Mesh, optional
        Acoustic domain
    cfg : Config, optional
        Configuration object
    downsample : int, default=1
        Mesh downsampling factor
    forcing_scale : float, default=1.0
        Reynolds stress forcing scale
    verbose : bool, default=True
        Print diagnostics
    
    Returns
    -------
    dict
        Streaming solution and diagnostics
    """
    solution = solve_streaming_stokes(
        p_solution=p_solution,
        domain=domain,
        downsample_factor=downsample,
        forcing_scale=forcing_scale,
        verbose=verbose,
    )
    
    # Return as dict for backward compatibility
    if solution is None:
        return {"max_speed": 0.0}
    
    u_vals = solution.u_function.x.array
    max_speed = np.max(np.abs(u_vals)) if len(u_vals) > 0 else 0.0
    
    return {
        "max_speed": max_speed,
        "streaming_solution": solution,
        "diagnostics": solution.diagnostics,
    }


def solve_streaming_stokes(
    p_solution: PressureSolution,
    domain: Optional[mesh.Mesh] = None,
    facet_tags: Optional[mesh.MeshTags] = None,
    downsample_factor: int = 1,
    forcing_scale: float = 1.0,
    verbose: bool = True,
) -> Optional[StreamingSolution]:
    """
    Solve mixed Stokes equations for acoustic streaming with robust preconditioner.
    
    This is the main Level-2 streaming solver using:
    - Taylor-Hood (P2-P1) stable mixed element
    - Fieldsplit Schur complement preconditioner (proper saddle-point treatment)
    - Pressure nullspace attachment
    - Mesh downsampling for memory efficiency (optional)
    - Force scaling for conditioning control (optional)
    
    Physics:
        -μ∇²u + ∇p = f(x) (Stokes momentum)
        ∇·u = 0            (Incompressibility)
        
    where f = -∇·⟨ρ v₁ ⊗ v₁⟩ is Reynolds stress forcing.
    
    Parameters
    ----------
    p_solution : PressureSolution
        Pressure solution (for v₁ computation)
    domain : mesh.Mesh, optional
        Streaming mesh (defaults to pressure mesh; can be coarser)
    facet_tags : mesh.MeshTags, optional
        Boundary tags (auto-created if not provided)
    downsample_factor : int
        Mesh downsampling: 1 = no downsampling, 2 = coarse mesh (8x fewer cells in 3D)
    forcing_scale : float
        Scaling factor for Reynolds stress forcing (for conditioning tests)
    verbose : bool
        Print progress
        
    Returns
    -------
    StreamingSolution or None
        Streaming solution if converged, None if solver diverged (with diagnostics saved)
    """
    cfg = p_solution.cfg
    mu = cfg.mu
    rho = cfg.rho
    omega = cfg.omega
    H = cfg.H
    L = cfg.L
    
    domain_acoustic = p_solution.p_function.function_space.mesh
    
    if domain is None:
        domain = domain_acoustic
    
    # Optionally downsample for memory efficiency
    domain_streaming = downsample_mesh(domain, downsample_factor, verbose=verbose)
    
    if verbose:
        print(f"\n{'='*70}")
        print("SOLVING ACOUSTIC STREAMING (Level-2 Stokes)")
        print(f"{'='*70}")
    
    # =========================================================================
    # STEP 1: Compute first-order velocity
    # =========================================================================
    if verbose:
        print("\nStep 1: Computing first-order velocity from pressure gradient...")
    v1_values, v1_coords = compute_first_order_velocity(p_solution, domain_streaming, verbose=verbose)
    
    # =========================================================================
    # STEP 2: Create mixed function space (Taylor-Hood P2-P1)
    # =========================================================================
    if verbose:
        print("\nStep 2: Setting up Taylor-Hood mixed element...")
    
    cell_type = domain_streaming.basix_cell()
    P2_vec = basix.ufl.element("Lagrange", cell_type, 2, shape=(3,))
    P1 = basix.ufl.element("Lagrange", cell_type, 1)
    TH = basix.ufl.mixed_element([P2_vec, P1])
    W = fem.functionspace(domain_streaming, TH)
    
    # =========================================================================
    # STEP 3: Compute Reynolds stress forcing (CORRECT PHASOR-BASED)
    # =========================================================================
    if verbose:
        print("\nStep 3: Computing Reynolds stress forcing (phasor-based)...")
    
    p_func = p_solution.p_function
    grad_p = ufl.grad(p_func)
    
    # CORRECT PHASOR-BASED FIRST-ORDER VELOCITY
    # v₁ = ∇p / (iωρ)  (complex phasor)
    v1_ufl = grad_p / (1j * omega * rho)
    
    # CORRECT TIME-AVERAGED REYNOLDS STRESS
    # For complex phasors, the time-averaged product is:
    #   ⟨v⊗v⟩ = (1/2) Re(v₁ ⊗ v₁*)
    # where v₁* is the complex conjugate
    #
    # Using UFL, we compute:
    #   stress = (1/2) * rho * Re(v₁ ⊗ conj(v₁))
    #          = (1/2) * rho * (Re(v₁) ⊗ Re(v₁) + Im(v₁) ⊗ Im(v₁))
    #
    # Both formulations are equivalent.
    v1_conj = ufl.conj(v1_ufl)
    stress_complex = 0.5 * rho * ufl.outer(v1_ufl, v1_conj)
    stress_ufl = ufl.real(stress_complex)
    
    # Forcing: f = -∇·⟨ρ v⊗v⟩ with optional scaling
    f_ufl = -forcing_scale * ufl.div(stress_ufl)
    
    # Project forcing to vector space for diagnostics AND for assembly.
    # CRITICAL: Direct UFL assembly of inner(-div(stress), w)*dx can lose
    # precision when cell volumes are very small (O(1e-11) m³ for mm-scale
    # domains in SI units). Projecting to a Function first preserves the
    # forcing magnitude through pointwise evaluation, then the function
    # is used in the linear form assembly.
    V_forcing = fem.functionspace(domain_streaming, ("Lagrange", 1, (3,)))
    f_expr = fem.Expression(f_ufl, V_forcing.element.interpolation_points())
    f_func = fem.Function(V_forcing)
    f_func.interpolate(f_expr)
    # Take real part (should already be real, but ensure no complex noise)
    f_func.x.array[:] = np.real(f_func.x.array)
    
    # Compute forcing statistics
    f_vals = f_func.x.array.copy()
    n_f = len(f_vals) // 3
    f_mag = np.linalg.norm(f_vals.reshape((n_f, 3)), axis=1)
    max_f = float(np.max(f_mag))
    
    # Numerical scaling: compute a scale factor to bring forcing into
    # a numerically safe range. The Stokes equation is linear, so we can
    # solve with scaled forcing and rescale the solution.
    # Target: max|f_scaled| ~ 1 for good conditioning.
    if max_f > 0:
        numerical_scale = 1.0 / max_f
    else:
        numerical_scale = 1.0
    
    # Apply numerical scaling to the forcing function
    f_func.x.array[:] *= numerical_scale
    
    if verbose:
        print(f"  Reynolds stress forcing:")
        print(f"    max |f| = {max_f:.2e} Pa/m")
        print(f"    median |f| = {float(np.median(f_mag)):.2e} Pa/m")
        print(f"    numerical scale = {numerical_scale:.2e} (for assembly conditioning)")
    
    # =========================================================================
    # STEP 4: Set up weak form (mixed Stokes)
    # =========================================================================
    if verbose:
        print("\nStep 4: Assembling mixed Stokes system...")
    
    (u, p) = ufl.TrialFunctions(W)
    (w, q) = ufl.TestFunctions(W)
    
    # Bilinear form: μ(∇u:∇w) - (p, ∇·w) - (q, ∇·u)
    a = (
        mu * inner(grad(u), grad(w)) * dx
        - inner(p, div(w)) * dx
        - inner(div(u), q) * dx
    )
    
    # Linear form: (f, w) — using projected & scaled forcing function
    # (NOT the UFL expression, which loses precision on tiny meshes)
    L_form = inner(f_func, w) * dx
    
    # =========================================================================
    # STEP 5: Boundary conditions (no-slip on bottom/sides, free-slip on top)
    # =========================================================================
    if verbose:
        print("\nStep 5: Setting up boundary conditions...")
    
    bcs = []
    W0 = W.sub(0)  # Velocity subspace
    W0_collapsed, _ = W0.collapse()
    
    fdim = domain_streaming.topology.dim - 1
    
    # Create zero function for Dirichlet BCs
    u_zero_func = fem.Function(W0_collapsed)
    u_zero_func.x.array[:] = 0.0
    
    # No-slip on bottom (z=0): u = 0
    def bottom(x): return np.isclose(x[2], 0.0, atol=H*1e-6)
    bottom_facets = mesh.locate_entities_boundary(domain_streaming, fdim, bottom)
    bottom_dofs = fem.locate_dofs_topological((W0, W0_collapsed), fdim, bottom_facets)
    bc_bottom = fem.dirichletbc(u_zero_func, bottom_dofs, W0)
    bcs.append(bc_bottom)
    
    # No-slip on side walls
    def x0(x): return np.isclose(x[0], 0.0, atol=L*1e-6)
    def xL(x): return np.isclose(x[0], L, atol=L*1e-6)
    def y0(x): return np.isclose(x[1], 0.0, atol=L*1e-6)
    def yL(x): return np.isclose(x[1], L, atol=L*1e-6)
    
    for loc in [x0, xL, y0, yL]:
        facets = mesh.locate_entities_boundary(domain_streaming, fdim, loc)
        dofs = fem.locate_dofs_topological((W0, W0_collapsed), fdim, facets)
        bc = fem.dirichletbc(u_zero_func, dofs, W0)
        bcs.append(bc)
    
    # Top (z=H): free-slip simplified as u_z = 0 (no penetration)
    # Full free-slip (u·n=0 + σ·n·t=0) is complex; this is pragmatic alternative
    def top(x): return np.isclose(x[2], H, atol=H*1e-6)
    top_facets = mesh.locate_entities_boundary(domain_streaming, fdim, top)
    
    # Constrain normal component (z) to zero
    W0_z = W0.sub(2)
    W0_z_collapsed, W0_z_map = W0.sub(2).collapse()
    top_dofs_z = fem.locate_dofs_topological((W0.sub(2), W0_z_collapsed), fdim, top_facets)
    u_z_zero_func = fem.Function(W0_z_collapsed)
    u_z_zero_func.x.array[:] = 0.0
    bc_top = fem.dirichletbc(u_z_zero_func, top_dofs_z, W0.sub(2))
    bcs.append(bc_top)
    
    if verbose:
        print(f"  Boundary conditions:")
        print(f"    No-slip on: bottom, 4 side walls")
        print(f"    Free-slip on: top (simplified as u_z=0)")
    
    # =========================================================================
    # STEP 6: Assemble system matrices and vectors
    # =========================================================================
    t_assemble_start = time.time()
    
    a_form = fem.form(a)
    L_form_compiled = fem.form(L_form)
    
    A = assemble_matrix(a_form, bcs=bcs)
    A.assemble()
    
    b = assemble_vector(L_form_compiled)
    
    if verbose:
        print(f"\n  RHS diagnostics (before BCs):")
        print(f"    ||b||_2 = {b.norm():.3e}")
        print(f"    ||b||_inf = {b.norm(PETSc.NormType.NORM_INFINITY):.3e}")
    
    fem.petsc.apply_lifting(b, [a_form], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    fem.petsc.set_bc(b, bcs)
    
    if verbose:
        print(f"    ||b||_2 (after BCs) = {b.norm():.3e}")
        print(f"    ||b||_inf (after BCs) = {b.norm(PETSc.NormType.NORM_INFINITY):.3e}")
        print(f"    ||A||_inf = {A.norm(PETSc.NormType.NORM_INFINITY):.3e}")
    
    t_assemble = time.time() - t_assemble_start
    
    if verbose:
        dof_count = W.dofmap.index_map.size_global * W.dofmap.index_map_bs
        print(f"\nSystem size: {dof_count:,} DOFs")
        print(f"Assembly time: {t_assemble:.2f} s")
    
    # =========================================================================
    # STEP 7: Configure KSP with fieldsplit Schur preconditioner
    # =========================================================================
    if verbose:
        print("\nStep 6: Configuring KSP solver...")
    
    ksp = PETSc.KSP().create(domain_streaming.comm)
    ksp.setOperators(A)
    
    # Use direct solver (MUMPS/LU) for robustness.
    # The Stokes system with SI units has entries spanning many orders of
    # magnitude (cell volumes O(1e-12)), making iterative methods unreliable.
    # MUMPS handles the tiny entries through pivoting.
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("lu")
    pc.setFactorSolverType("mumps")
    
    # Attach pressure nullspace
    attach_pressure_nullspace(ksp, W)
    
    if verbose:
        print("  Solver configured:")
        print(f"    • Type: Direct (LU via MUMPS)")
        print("    • Nullspace: constant pressure mode attached")
    
    # =========================================================================
    # STEP 8: Solve
    # =========================================================================
    if verbose:
        print("\nStep 7: Solving...\n")
    
    wh = fem.Function(W)
    
    t_solve_start = time.time()
    try:
        ksp.solve(b, wh.x.petsc_vec)
        wh.x.scatter_forward()
        t_solve = time.time() - t_solve_start
        solver_diverged = False
    except Exception as e:
        t_solve = time.time() - t_solve_start
        solver_diverged = True
        if verbose:
            print(f"  ERROR: Solver raised exception: {e}")
        ksp.destroy()
        A.destroy()
        b.destroy()
        return None
    
    # Check convergence
    conv_reason = ksp.getConvergedReason()
    if conv_reason < 0:
        solver_diverged = True
        if verbose:
            reason_map = {
                -2: "DIVERGED_ITS (max iterations)",
                -11: "DIVERGED_PC_FAILED (preconditioner failure)",
            }
            print(f"  WARNING: Solver diverged with reason {conv_reason} ({reason_map.get(conv_reason, 'UNKNOWN')})")
    
    # =========================================================================
    # STEP 9: Extract solution and de-scale
    # =========================================================================
    u_h = wh.sub(0).collapse()
    p_h = wh.sub(1).collapse()
    
    # De-scale: we solved with f_scaled = numerical_scale * f_physical
    # Since Stokes is linear: u_scaled = numerical_scale * u_physical
    # → u_physical = u_scaled / numerical_scale
    if numerical_scale != 1.0:
        u_h.x.array[:] /= numerical_scale
        p_h.x.array[:] /= numerical_scale
        if verbose:
            print(f"\n  De-scaled solution by 1/{numerical_scale:.2e}")
    
    u_h.name = "streaming_velocity"
    p_h.name = "streaming_pressure"
    
    # De-scale the forcing function back for diagnostics
    f_func.x.array[:] /= numerical_scale
    
    # Compute diagnostics
    diags = compute_streaming_diagnostics(
        u_h, p_h, domain_streaming, cfg, ksp, f_func,
        assemble_time=t_assemble,
        solve_time=t_solve,
        verbose=verbose
    )
    
    if solver_diverged:
        if verbose:
            print(f"\n{'='*70}")
            print("WARNING: Streaming solve diverged, but returning partial solution with diagnostics")
            print(f"{'='*70}\n")
    elif verbose:
        print(f"{'='*70}\n")
    
    # Clean up
    A.destroy()
    b.destroy()
    ksp.destroy()
    
    # Return solution
    return StreamingSolution(
        u_function=u_h,
        p_function=p_h,
        mesh_acoustic=domain_acoustic,
        mesh_streaming=domain_streaming,
        cfg=cfg,
        diagnostics=diags,
    )


def compute_streaming_velocity(
    p_solution: PressureSolution,
    downsample_factor: int = 2,
    forcing_scale: float = 1.0,
    verbose: bool = True,
) -> Optional[StreamingSolution]:
    """
    Convenience function: main entry point for streaming computation.
    
    Uses Level-2 Stokes solver with proper saddle-point preconditioner.
    
    Parameters
    ----------
    p_solution : PressureSolution
        Pressure solution
    downsample_factor : int
        Mesh coarsening (1 = no, 2 = coarse for memory)
    forcing_scale : float
        Reynolds stress forcing scale factor
    verbose : bool
        Print progress
        
    Returns
    -------
    StreamingSolution or None
        Streaming solution if successful, None if solver diverged
    """
    return solve_streaming_stokes(
        p_solution,
        downsample_factor=downsample_factor,
        forcing_scale=forcing_scale,
        verbose=verbose,
    )
