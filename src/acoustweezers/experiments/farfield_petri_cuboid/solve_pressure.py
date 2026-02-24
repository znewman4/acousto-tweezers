"""
Helmholtz pressure solver for the far-field petri cuboid with PML.

Complex-coordinate-stretch PML formulation for Helmholtz
=========================================================

Standard Helmholtz in physical coordinates:

    ∇²p + k²p = 0

With PML we replace each real coordinate by a complex-stretched one:

    x̃ = x + (i/ω) ∫₀ˣ σ_x(x') dx'
    (similarly for ỹ, z̃)

which modifies the differential operators.  The resulting weak form
(after factoring out the Jacobian of the complex stretch) is:

    ∫ [ Λ · (∇p · ∇v̄) ] dx  −  k² ∫ J p v̄ dx  =  ∮ (source terms)

where
    sα = 1 + i σα/ω         for α ∈ {x, y, z}
    Λ  = diag(sy sz/sx,  sx sz/sy,  sx sy/sz)     (metric tensor)
    J  = sx sy sz                                   (Jacobian)

In the physical region σα = 0 ⇒ sα = 1, Λ = I, J = 1 and the form
reduces to the standard Helmholtz.

σα ramp profile (polynomial):
    σα(ξ) = σ_max · (ξ/L_pml)^d
where ξ is the depth into the PML layer and d is the polynomial degree.

Author: Acousto-Tweezers Project
Date: 2026-02-16
"""

from __future__ import annotations

import time
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict

from mpi4py import MPI
import dolfinx
from dolfinx import fem, mesh, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
import ufl
from ufl import (inner, grad, dx, ds, TrialFunction, TestFunction,
                 Measure, as_vector, Dx)

from .config import FarFieldConfig
from .mesh import (
    create_mesh,
    TAG_BOTTOM_DISK, TAG_BOTTOM_OUTSIDE, TAG_TOP,
    TAG_X0, TAG_XL, TAG_Y0, TAG_YL,
    TAG_STAND_X0, TAG_STAND_XL, TAG_STAND_Y0, TAG_STAND_YL,
    CELL_PHYSICAL,
)


# =====================================================================
# PressureSolution dataclass (same pattern as shallow_square_dish)
# =====================================================================

@dataclass
class PressureSolution:
    """Container for the computed pressure field."""
    p_function: fem.Function
    cfg: FarFieldConfig
    solver_time: float = 0.0
    dofs: int = 0
    mesh_time: float = 0.0
    ksp_iterations: int = 0
    ksp_converged_reason: int = 0
    ksp_residual_norm: float = 0.0

    @property
    def p_values(self) -> np.ndarray:
        return self.p_function.x.array.copy()

    @property
    def p_mag(self) -> np.ndarray:
        return np.abs(self.p_values)

    @property
    def p_phase(self) -> np.ndarray:
        return np.angle(self.p_values)

    @property
    def max_pressure(self) -> float:
        return float(np.max(self.p_mag))

    @property
    def coords(self) -> np.ndarray:
        return self.p_function.function_space.tabulate_dof_coordinates()


# =====================================================================
# σ-profile helpers  (evaluated at mesh coordinates → fem.Function)
# =====================================================================

def _build_sigma_functions(V: fem.FunctionSpace, cfg: FarFieldConfig):
    """
    Return three ``fem.Function`` objects (σ_x, σ_y, σ_z) defined over *V*.

    σ_x(x) is nonzero in the PML strips at x < t and x > Lx-t.
    σ_y(y)  is nonzero in the PML strips at y < t and y > Ly-t.
    σ_z(z)  is nonzero below t_pml_z  AND  outside the disk column (r > R).
    """
    ndofs = len(V.tabulate_dof_coordinates())
    sigma_x = fem.Function(V)
    sigma_y = fem.Function(V)
    sigma_z = fem.Function(V)

    # When PML is disabled, return all-zero σ fields → s=1, Λ=I, J=1
    if not cfg.pml_enabled:
        sigma_x.x.array[:] = np.zeros(ndofs, dtype=np.complex128)
        sigma_y.x.array[:] = np.zeros(ndofs, dtype=np.complex128)
        sigma_z.x.array[:] = np.zeros(ndofs, dtype=np.complex128)
        return sigma_x, sigma_y, sigma_z

    coords = V.tabulate_dof_coordinates()
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    sigma_max = cfg.sigma_max
    t_xy = cfg.t_pml_xy
    t_z = cfg.t_pml_z
    Lx, Ly = cfg.Lx, cfg.Ly
    deg = cfg.pml_degree
    R = cfg.disk_radius
    cx, cy = cfg.disk_center_x, cfg.disk_center_y

    def _ramp(distance, thickness):
        """Polynomial ramp: σ_max * (d/L)^deg  clamped to [0, σ_max]."""
        if thickness <= 0:
            return np.zeros_like(distance)
        frac = np.clip(distance / thickness, 0.0, 1.0)
        return sigma_max * frac ** deg

    # σ_x — lateral PML only in water bath (z < H_under).
    # Petri slab has physical walls (transducers), no absorption.
    H_under = cfg.H_under
    sx_arr = np.zeros_like(x)
    mask_lo = (x < t_xy) & (z < H_under)
    mask_hi = (x > Lx - t_xy) & (z < H_under)
    sx_arr[mask_lo] = _ramp(t_xy - x[mask_lo], t_xy)
    sx_arr[mask_hi] = _ramp(x[mask_hi] - (Lx - t_xy), t_xy)

    # σ_y — same z-filter
    sy_arr = np.zeros_like(y)
    mask_lo = (y < t_xy) & (z < H_under)
    mask_hi = (y > Ly - t_xy) & (z < H_under)
    sy_arr[mask_lo] = _ramp(t_xy - y[mask_lo], t_xy)
    sy_arr[mask_hi] = _ramp(y[mask_hi] - (Ly - t_xy), t_xy)

    # σ_z (only below t_pml_z AND outside disk column)
    sz_arr = np.zeros_like(z)
    mask_z = z < t_z
    r2 = (x - cx)**2 + (y - cy)**2
    mask_out = r2 > R**2
    mask = mask_z & mask_out
    sz_arr[mask] = _ramp(t_z - z[mask], t_z)

    sigma_x.x.array[:] = sx_arr.astype(np.complex128)
    sigma_y.x.array[:] = sy_arr.astype(np.complex128)
    sigma_z.x.array[:] = sz_arr.astype(np.complex128)

    return sigma_x, sigma_y, sigma_z


# =====================================================================
# Main solver
# =====================================================================

def solve_helmholtz(
    cfg: FarFieldConfig,
    verbose: bool = True,
    petsc_options: Optional[dict] = None,
    export_fields: bool = False,
    export_dir: Optional[str] = None,
) -> PressureSolution:
    """
    Solve the PML-Helmholtz system for the far-field cuboid.

    Parameters
    ----------
    cfg : FarFieldConfig
    verbose : bool
    petsc_options : dict, optional
        PETSc solver options.  Default uses MUMPS direct solver.
    export_fields : bool
        If True, export XDMF field files after solving.
    export_dir : str, optional
        Directory for XDMF exports (required if export_fields=True).

    Returns
    -------
    PressureSolution
    """
    from petsc4py import PETSc

    if not np.issubdtype(PETSc.ScalarType, np.complexfloating):
        raise RuntimeError(
            "This solver requires PETSc with complex scalars.\n"
            "Use: micromamba run -n acousto-complex python <script>"
        )

    t0 = time.time()

    # ── mesh ──────────────────────────────────────────────────────────
    domain, facet_tags, cell_tags, tag_info = create_mesh(cfg, verbose=verbose)
    t_mesh = time.time() - t0

    omega = cfg.omega
    k = cfg.k
    rho = cfg.rho
    Lx, Ly = cfg.Lx, cfg.Ly
    H = cfg.H_total
    H_under = cfg.H_under

    # P2 function space
    V = fem.functionspace(domain, ("Lagrange", 2))
    ndofs = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
    if verbose:
        print(f"  DOFs: {ndofs}")

    # ── PML σ fields ──────────────────────────────────────────────────
    sigma_x, sigma_y, sigma_z = _build_sigma_functions(V, cfg)

    # PML stretch factors as *UFL expressions* (not projected Functions).
    #   s_α = 1 + i σ_α / ω
    # Building Λ and J as UFL expressions means FFCx evaluates them
    # at quadrature points from the interpolated σ values, rather than
    # interpolating a P2 projection of the product.  This avoids
    # projection error for the rational expressions Λ_x = sy·sz/sx.
    one = fem.Constant(domain, PETSc.ScalarType(1.0))
    inv_omega_i = fem.Constant(domain, PETSc.ScalarType(1j / omega))

    sx = one + inv_omega_i * sigma_x   # UFL expr
    sy = one + inv_omega_i * sigma_y
    sz = one + inv_omega_i * sigma_z

    # Λ_x = sy·sz / sx   (anisotropic PML metric)
    # J   = sx·sy·sz      (PML Jacobian)
    Lam_x = sy * sz / sx
    Lam_y = sx * sz / sy
    Lam_z = sx * sy / sz
    Jac   = sx * sy * sz

    if verbose:
        any_sigma = (
            (np.abs(sigma_x.x.array) > 0)
            | (np.abs(sigma_y.x.array) > 0)
            | (np.abs(sigma_z.x.array) > 0)
        )
        n_pml = int(np.sum(any_sigma))
        n_phys = ndofs - n_pml
        print(f"  PML DOFs: {n_pml},  Physical DOFs: {n_phys}")

    # ── measures ──────────────────────────────────────────────────────
    dss = Measure("ds", domain=domain, subdomain_data=facet_tags)

    u = TrialFunction(V)
    v = TestFunction(V)

    # ── bilinear form (PML-Helmholtz) ─────────────────────────────────
    # a(u,v) = ∫ [Λx u_x v̄_x + Λy u_y v̄_y + Λz u_z v̄_z] dx
    #        - k² ∫ J u v̄ dx
    #        + impedance surface integrals
    #
    # Note: inner(a, b) in complex UFL mode computes a·conj(b).
    a_form = (
        Lam_x * inner(Dx(u, 0), Dx(v, 0))
        + Lam_y * inner(Dx(u, 1), Dx(v, 1))
        + Lam_z * inner(Dx(u, 2), Dx(v, 2))
        - k**2 * Jac * inner(u, v)
    ) * dx

    # ── top face: water–air Robin BC (fixed) ──────────────────────
    # ∂p/∂n + ik(ρ_water c_water)/(ρ_air c_air) p = 0
    # ⇒ α = -iωρ / Z_air   (Z_air = ρ_air * c_air)
    Z_air = cfg.Z_air   # = ρ_air · c_air = 411.6 Pa·s/m
    alpha_top = fem.Constant(domain, PETSc.ScalarType(-1j * omega * rho / Z_air))
    a_form += alpha_top * inner(u, v) * dss(TAG_TOP)
    if verbose:
        Z_rel = Z_air / cfg.Z_water
        print(f"  Top BC: water–air Robin  Z_air={Z_air:.1f} Pa·s/m  "
              f"Z_rel={Z_rel:.6f}")

    # ── RHS: Neumann sources ──────────────────────────────────────────
    L_terms = []

    # 1) Bottom disk: vortex drive  ∂p/∂n = −iωρ v_n(x)
    # Bottom face outward normal is -z.
    # For upward-propagating wave: v_z > 0 → v_n = v·n̂ = -v_z < 0
    # ∂p/∂n = iωρ v_n = -iωρ v_z, so g = -iωρ V_disk * pattern.
    V_disk = cfg.disk_velocity_amplitude
    g_disk = _create_disk_source(V, domain, facet_tags, cfg, verbose)
    # scale by −iωρ V_disk  (v_n < 0 on bottom face → wave propagates +z)
    g_disk.x.array[:] *= -1j * omega * rho * V_disk
    L_terms.append(inner(g_disk, v) * dss(TAG_BOTTOM_DISK))

    # 2) Standing-wave patches on petri-slab side walls
    V_stand = cfg.standing_velocity_amplitude
    g_stand = fem.Constant(domain, PETSc.ScalarType(-1j * omega * rho * V_stand))

    stand_x_tags = [TAG_STAND_X0, TAG_STAND_XL]
    stand_y_tags = [TAG_STAND_Y0, TAG_STAND_YL]

    if cfg.standing_phase_pattern == "antiphase":
        L_terms.append(inner(g_stand, v) * dss(TAG_STAND_X0))
        L_terms.append(inner(-g_stand, v) * dss(TAG_STAND_XL))
    elif cfg.standing_phase_pattern == "inphase":
        L_terms.append(inner(g_stand, v) * dss(TAG_STAND_X0))
        L_terms.append(inner(g_stand, v) * dss(TAG_STAND_XL))
    elif cfg.standing_phase_pattern == "quadrature":
        L_terms.append(inner(g_stand, v) * dss(TAG_STAND_X0))
        g_stand_q = fem.Constant(domain, PETSc.ScalarType(-1j * omega * rho * V_stand * 1j))
        L_terms.append(inner(g_stand_q, v) * dss(TAG_STAND_XL))

    if cfg.standing_axis == "both":
        if cfg.standing_phase_pattern == "antiphase":
            L_terms.append(inner(g_stand, v) * dss(TAG_STAND_Y0))
            L_terms.append(inner(-g_stand, v) * dss(TAG_STAND_YL))
        elif cfg.standing_phase_pattern == "inphase":
            L_terms.append(inner(g_stand, v) * dss(TAG_STAND_Y0))
            L_terms.append(inner(g_stand, v) * dss(TAG_STAND_YL))
        elif cfg.standing_phase_pattern == "quadrature":
            L_terms.append(inner(g_stand, v) * dss(TAG_STAND_Y0))
            L_terms.append(inner(g_stand_q, v) * dss(TAG_STAND_YL))

    if verbose:
        print(f"  Standing wave: V={V_stand*1e6:.1f} μm/s  "
              f"pattern={cfg.standing_phase_pattern}  axis={cfg.standing_axis}")

    # Build L
    L_form = L_terms[0]
    for t in L_terms[1:]:
        L_form = L_form + t

    # ── Dirichlet BCs ─────────────────────────────────────────────
    bcs = []  # no Dirichlet BCs (Robin on top, Neumann elsewhere)

    # ── solve ─────────────────────────────────────────────────────────
    # NOTE: MUMPS direct solver is the reliable default.
    # GMRES+ILU diverges on the PML-Helmholtz system (see diagnostic
    # pipeline Step 1, 2026-02-18: DIVERGED_BREAKDOWN at 600 iters,
    # max|p| = 8.37 vs correct 20.11 Pa from direct solve).
    if petsc_options is None:
        petsc_options = {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        }

    if verbose:
        print("  Assembling & solving …")

    t_asm = time.time()
    problem = LinearProblem(a_form, L_form, bcs=bcs, petsc_options=petsc_options)
    ph = problem.solve()
    ph.name = "pressure"
    t_solve_end = time.time()

    # Extract KSP diagnostics
    ksp = problem.solver
    ksp_its = ksp.getIterationNumber()
    ksp_reason = int(ksp.getConvergedReason())
    try:
        ksp_rnorm = float(ksp.getResidualNorm())
    except Exception:
        ksp_rnorm = float("nan")

    t_total = t_solve_end - t0
    t_solve_only = t_solve_end - t_asm
    max_p = float(np.max(np.abs(ph.x.array)))

    _KSP_REASONS = {
        1: "CONVERGED_RTOL_NORMAL", 2: "CONVERGED_RTOL",
        3: "CONVERGED_ATOL", 9: "CONVERGED_ITERATING",
        -3: "DIVERGED_ITS", -4: "DIVERGED_DTOL",
        -5: "DIVERGED_BREAKDOWN", -9: "DIVERGED_NANORINF",
    }

    if verbose:
        reason_str = _KSP_REASONS.get(ksp_reason, f"REASON={ksp_reason}")
        print(f"  KSP converged: {reason_str}  "
              f"iters={ksp_its}  |r|={ksp_rnorm:.2e}")
        print(f"  Timing: mesh {t_mesh:.1f}s  assemble+solve {t_solve_only:.1f}s  "
              f"total {t_total:.1f}s")
        print(f"  max|p| = {max_p:.2f} Pa   DOFs = {ndofs}")
        print(f"{'='*70}\n")

    sol = PressureSolution(
        p_function=ph,
        cfg=cfg,
        solver_time=t_total,
        dofs=ndofs,
        mesh_time=t_mesh,
        ksp_iterations=ksp_its,
        ksp_converged_reason=ksp_reason,
        ksp_residual_norm=ksp_rnorm,
    )
    # attach mesh objects for postprocessing
    sol.domain = domain
    sol.facet_tags = facet_tags
    sol.cell_tags = cell_tags
    sol.tag_info = tag_info
    sol.V = V
    sol.sigma_x = sigma_x
    sol.sigma_y = sigma_y
    sol.sigma_z = sigma_z

    # Optional field export
    if export_fields and export_dir is not None:
        from acoustweezers.io.export_fields import export_pressure_fields
        if verbose:
            print(f"  Exporting fields to {export_dir} …")
        export_pressure_fields(sol, export_dir, verbose=verbose)

    return sol


# =====================================================================
# Disk source (dispatches between ideal vortex and plastic lens)
# =====================================================================

def _create_disk_source(
    V: fem.FunctionSpace,
    domain: mesh.Mesh,
    facet_tags: mesh.MeshTags,
    cfg: FarFieldConfig,
    verbose: bool = True,
) -> fem.Function:
    """
    Build complex boundary pattern on the bottom disk facets.

    Dispatches to:
      - ideal:   v_n = A(r) · exp(i ℓ θ)        (pure vortex, no focusing)
      - plastic:  v_n = A(r) · exp(i φ_plastic)  (focused vortex via thickness lens)

    Returns a fem.Function; caller multiplies by −iωρ V_disk.
    """
    fdim = domain.topology.dim - 1
    disk_facets = facet_tags.indices[facet_tags.values == TAG_BOTTOM_DISK]
    disk_dofs = fem.locate_dofs_topological(V, fdim, disk_facets)

    coords = V.tabulate_dof_coordinates()
    cx, cy = cfg.disk_center_x, cfg.disk_center_y

    func = fem.Function(V)
    func.x.array[:] = 0.0 + 0.0j

    if cfg.lens_drive == "axicon":
        from acoustweezers.physics.acoustics.vortex_lens import (
            AxiconLensConfig, create_axicon_lens_drive,
        )
        axicon_cfg = AxiconLensConfig(
            topological_charge=cfg.lens_l,
            axicon_angle_deg=cfg.lens_axicon_angle_deg,
            c_water=cfg.c,
            frequency_hz=cfg.frequency_hz,
            aperture_radius=cfg.disk_radius,
            center=None,
            apodization=cfg.lens_apodization,
            apodization_strength=cfg.lens_apodization_strength,
        )
        x_d = coords[disk_dofs, 0]
        y_d = coords[disk_dofs, 1]
        pattern = create_axicon_lens_drive(
            x_d, y_d, axicon_cfg,
            center_x=cx, center_y=cy, verbose=verbose,
        )
        func.x.array[disk_dofs] = pattern.astype(np.complex128)

        if verbose:
            n_active = int(np.sum(np.abs(pattern) > 1e-10))
            print(f"  Disk source: axicon lens  l={cfg.lens_l}  "
                  f"alpha={cfg.lens_axicon_angle_deg:.1f} deg  "
                  f"active={n_active}/{len(disk_dofs)}")

    elif cfg.lens_drive == "plastic":
        from acoustweezers.physics.acoustics.vortex_lens import (
            PlasticLensConfig, create_plastic_lens_drive,
        )
        lens_cfg = PlasticLensConfig(
            topological_charge=cfg.lens_l,
            focal_length=cfg.lens_focal_length,
            focus_offset_x=cfg.lens_focus_offset_x,
            focus_offset_y=cfg.lens_focus_offset_y,
            c_lens=cfg.lens_c_lens,
            c_water=cfg.c,
            frequency_hz=cfg.frequency_hz,
            aperture_radius=cfg.disk_radius,
            center=None,  # use explicit center
            apodization=cfg.lens_apodization,
            apodization_strength=cfg.lens_apodization_strength,
        )
        x_d = coords[disk_dofs, 0]
        y_d = coords[disk_dofs, 1]
        pattern = create_plastic_lens_drive(
            x_d, y_d, lens_cfg,
            center_x=cx, center_y=cy, verbose=verbose,
        )
        func.x.array[disk_dofs] = pattern.astype(np.complex128)

        if verbose:
            n_active = int(np.sum(np.abs(pattern) > 1e-10))
            print(f"  Disk source: plastic lens  l={cfg.lens_l}  "
                  f"f={cfg.lens_focal_length*1e3:.1f} mm  "
                  f"active={n_active}/{len(disk_dofs)}")

    else:
        # Legacy "ideal" pure vortex: A(r) exp(i ℓ θ)
        dx_a = coords[disk_dofs, 0] - cx
        dy_a = coords[disk_dofs, 1] - cy
        r = np.sqrt(dx_a**2 + dy_a**2)
        theta = np.arctan2(dy_a, dx_a)
        R = cfg.disk_radius
        inside = r <= R

        amp = np.zeros_like(r)
        if cfg.vortex_apodization == "cosine_taper":
            amp[inside] = 0.5 * (1 + np.cos(np.pi * r[inside] / R))
        elif cfg.vortex_apodization == "gaussian":
            sigma = R / 2
            amp = np.exp(-r**2 / (2 * sigma**2))
        else:
            amp[inside] = 1.0

        ell = cfg.vortex_topological_charge
        pattern = amp * np.exp(1j * ell * theta)
        func.x.array[disk_dofs] = pattern.astype(np.complex128)

        if verbose:
            n_active = int(np.sum(np.abs(pattern) > 1e-10))
            print(f"  Disk source: ideal vortex  l={ell}  R={R*1e3:.2f} mm  "
                  f"active={n_active}/{len(disk_dofs)}")

    return func
