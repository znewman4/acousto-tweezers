#!/usr/bin/env python3
"""
Debug Script — Disc Impedance Absorption Investigation (Case A standing-only).

Tests three hypotheses for why FEniCSx Case A shows "empty centre" (low |p|
inside disc circle) that COMSOL standing-only does NOT show:

Variant 1: disc_robin=True   (current Attempt 2 — expected to show dead zone)
Variant 2: disc_robin=False  (rigid bottom everywhere — expected to match COMSOL)
Variant 3: disc_robin=True, Z_disc → 1000×Z_w (approx rigid, continuity check)

Also computes:
- H2: time-averaged power flux through disc and top impedance boundaries
- H3: disc area tagging verification (area, outline plot)

Usage:
    micromamba run -n acousto-complex python scripts/analysis/debug_disc_bc_case_a.py
"""
from __future__ import annotations

import sys, os, time, json
from pathlib import Path
from datetime import datetime
from dataclasses import replace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np

from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
import ufl
from ufl import inner, grad, dx, TrialFunction, TestFunction, Measure, FacetNormal

from acoustweezers.experiments.shallow_square_dish.config import ShallowDishConfig
from acoustweezers.experiments.shallow_square_dish.solve_pressure import (
    create_mesh,
    solve_helmholtz,
    _create_vortex_source,
    TAG_BOTTOM_DISC,
    TAG_BOTTOM_RIGID,
    TAG_TOP,
    TAG_X0, TAG_XL, TAG_Y0, TAG_YL,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import TwoSlopeNorm

# ─── globals ────────────────────────────────────────────────────
comm = MPI.COMM_WORLD
rank = comm.rank
ROOT = Path(__file__).resolve().parents[2]
NOW = datetime.now()
STAMP = NOW.strftime("%Y%m%d_%H%M")
OUTDIR = ROOT / "COMSOL_comparison_debug" / f"Case_A_disc_BC_tests_{STAMP}"
N_PLANE = 201

MUMPS_OPTS = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}


def log(msg="", end="\n"):
    if rank == 0:
        print(msg, end=end, flush=True)


# ═══════════════════════════════════════════════════════════════
# CONFIG — same as Attempt 2
# ═══════════════════════════════════════════════════════════════
CFG = ShallowDishConfig(
    L=10e-3,
    H=1e-3,
    frequency_hz=500e3,
    elements_per_wavelength=10,
    min_elements_z=8,
    rho=997.0,
    c=1484.0,
    mu=1.002e-3,
    vortex_velocity_amplitude=10e-6,
    standing_velocity_amplitude=10e-6,
    vortex_topological_charge=1,
    vortex_aperture_radius=3e-3,
    vortex_apodization="cosine_taper",
    vortex_phase_offset=0.0,
    standing_axis="both",
    standing_phase_pattern="antiphase",
    top_bc_type="impedance",
    top_impedance_factor=0.001,
    bottom_disc_radius=None,
    standing_full_wall=True,
    particle_radius=5e-6,
    particle_density=1050.0,
    particle_compressibility=2.4e-10,
)


# ═══════════════════════════════════════════════════════════════
# SAMPLING
# ═══════════════════════════════════════════════════════════════
def _make_plane_grid(cfg, z_val, N):
    xs = np.linspace(0, cfg.L, N)
    ys = np.linspace(0, cfg.L, N)
    X, Y = np.meshgrid(xs, ys)
    return np.column_stack([X.ravel(), Y.ravel(), np.full(N * N, z_val)])


def _sample_pressure(p_func, pts):
    domain = p_func.function_space.mesh
    tree = bb_tree(domain, domain.topology.dim)
    cands = compute_collisions_points(tree, pts)
    cells = compute_colliding_cells(domain, cands, pts)
    vals = np.full(len(pts), np.nan + 0j, dtype=np.complex128)
    for i in range(len(pts)):
        links = cells.links(i)
        if len(links) == 0:
            continue
        vals[i] = complex(p_func.eval(pts[i], links[0])[0])
    return vals


def _disc_circle(cfg, color="white", ls="--"):
    cx = cfg.L / 2 * 1e3
    cy = cfg.L / 2 * 1e3
    R = cfg.bottom_disc_radius_effective * 1e3
    return Circle((cx, cy), R, fill=False, edgecolor=color,
                  linewidth=1.2, linestyle=ls)


# ═══════════════════════════════════════════════════════════════
# CUSTOM SOLVER — disc Robin with arbitrary impedance
# ═══════════════════════════════════════════════════════════════
def solve_standing_custom_disc(domain, facet_tags, cfg,
                               disc_robin=True, Z_disc_override=None,
                               verbose=True):
    """
    Solve standing-only Helmholtz with configurable disc impedance.
    
    Parameters
    ----------
    disc_robin : bool
        If True, add impedance Robin on disc. If False, rigid everywhere.
    Z_disc_override : float or None
        If not None, use this impedance instead of Z_water for the disc Robin.
    """
    L = cfg.L
    H = cfg.H
    omega = cfg.omega
    k = cfg.k
    rho = cfg.rho
    Z = cfg.Z_water
    Z_top = cfg.Z_top

    V = fem.functionspace(domain, ("Lagrange", 2))
    dss = Measure("ds", domain=domain, subdomain_data=facet_tags)
    u = TrialFunction(V)
    v = TestFunction(V)

    # Bilinear form
    a = (inner(grad(u), grad(v)) - k**2 * inner(u, v)) * dx

    # Top Robin
    if cfg.top_bc_type == "impedance":
        alpha_top = -1j * omega * rho / Z_top
        a += alpha_top * inner(u, v) * dss(TAG_TOP)

    # Disc Robin
    Z_disc = Z_disc_override if Z_disc_override is not None else Z
    if disc_robin:
        alpha_disc = -1j * omega * rho / Z_disc
        a += alpha_disc * inner(u, v) * dss(TAG_BOTTOM_DISC)
        if verbose:
            log(f"    Disc Robin ON: Z_disc = {Z_disc:.1f} Pa·s/m "
                f"(Z_disc/Z_w = {Z_disc/Z:.1f})")
    else:
        if verbose:
            log(f"    Disc Robin OFF: entire bottom rigid")

    # RHS: standing wave actuation on side walls (antiphase, both axes)
    V_stand = cfg.standing_velocity_amplitude
    g_stand = -1j * omega * rho * V_stand

    L_terms = []
    L_terms.append(inner(g_stand, v) * dss(TAG_X0))
    L_terms.append(inner(-g_stand, v) * dss(TAG_XL))
    L_terms.append(inner(g_stand, v) * dss(TAG_Y0))
    L_terms.append(inner(-g_stand, v) * dss(TAG_YL))

    L_form = L_terms[0]
    for term in L_terms[1:]:
        L_form = L_form + term

    problem = LinearProblem(a, L_form, bcs=[], petsc_options=MUMPS_OPTS)
    p_sol = problem.solve()
    p_sol.name = "pressure_standing"

    return p_sol, V


# ═══════════════════════════════════════════════════════════════
# POWER FLUX COMPUTATION (H2)
# ═══════════════════════════════════════════════════════════════
def compute_power_flux(p_func, V_space, domain, facet_tags, cfg, tag, label):
    """
    Compute time-averaged acoustic power flux through a tagged surface.

    For an impedance boundary ∂p/∂n = (iωρ/Z)p, the normal velocity is:
        v_n = (1/(iωρ)) ∂p/∂n = p/Z

    Time-averaged power per unit area:
        ⟨I⟩ = ½ Re(p · v_n*)

    For impedance BC:  ⟨I⟩ = ½ Re(p · (p*/Z*)) = |p|²/(2Z)  [for real Z]
    
    We compute the surface integral ∫_S ½ Re(p · v_n*) dS using UFL.
    """
    omega = cfg.omega
    rho = cfg.rho
    Z = cfg.Z_water
    Z_top = cfg.Z_top

    dss = Measure("ds", domain=domain, subdomain_data=facet_tags)

    # For impedance boundary: v_n = p / Z
    # Power = ∫ ½ Re(p * conj(p/Z)) dS = ∫ |p|²/(2Z) dS  (for real Z)
    if tag == TAG_TOP:
        Z_bc = Z_top
    elif tag == TAG_BOTTOM_DISC:
        Z_bc = Z
    else:
        Z_bc = Z

    # Compute ∫ |p|² dS over the tagged surface
    p_abs_sq = ufl.real(p_func * ufl.conj(p_func))
    power_form = fem.form(0.5 / Z_bc * p_abs_sq * dss(tag))
    power = abs(fem.assemble_scalar(power_form))

    return power


# ═══════════════════════════════════════════════════════════════
# DISC AREA DIAGNOSTICS (H3)
# ═══════════════════════════════════════════════════════════════
def disc_area_diagnostics(domain, facet_tags, cfg, outdir):
    """Compute disc area metrics and plot disc facet outline."""
    R_disc = cfg.bottom_disc_radius_effective
    L = cfg.L

    dss = Measure("ds", domain=domain, subdomain_data=facet_tags)
    one = fem.Constant(domain, complex(1.0, 0.0))

    # Disc area
    area_form = fem.form(one * dss(TAG_BOTTOM_DISC))
    A_disc_mesh = abs(fem.assemble_scalar(area_form))
    A_disc_expected = np.pi * R_disc**2
    ratio = A_disc_mesh / A_disc_expected

    # Rigid area
    area_rigid_form = fem.form(one * dss(TAG_BOTTOM_RIGID))
    A_rigid_mesh = abs(fem.assemble_scalar(area_rigid_form))
    A_bottom_total = L * L
    A_rigid_expected = A_bottom_total - A_disc_expected

    # Top area
    area_top_form = fem.form(one * dss(TAG_TOP))
    A_top_mesh = abs(fem.assemble_scalar(area_top_form))

    lines = [
        f"Disc area (mesh):     {A_disc_mesh*1e6:.4f} mm²",
        f"Disc area (πR²):      {A_disc_expected*1e6:.4f} mm²",
        f"Area ratio:           {ratio:.4f}",
        f"Rigid area (mesh):    {A_rigid_mesh*1e6:.4f} mm²",
        f"Rigid area (expect):  {A_rigid_expected*1e6:.4f} mm²",
        f"Top area (mesh):      {A_top_mesh*1e6:.4f} mm²",
        f"Top area (expect):    {A_bottom_total*1e6:.4f} mm²",
        f"R_disc:               {R_disc*1e3:.2f} mm",
        f"Centre:               ({L/2*1e3:.2f}, {L/2*1e3:.2f}) mm",
    ]

    # Plot disc facet outline
    fdim = domain.topology.dim - 1
    disc_facets = facet_tags.indices[facet_tags.values == TAG_BOTTOM_DISC]
    rigid_facets = facet_tags.indices[facet_tags.values == TAG_BOTTOM_RIGID]

    # Get midpoints of disc facets
    domain.topology.create_connectivity(fdim, domain.topology.dim)
    disc_midpoints = mesh.compute_midpoints(domain, fdim, disc_facets)
    rigid_midpoints = mesh.compute_midpoints(domain, fdim, rigid_facets)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(disc_midpoints[:, 0] * 1e3, disc_midpoints[:, 1] * 1e3,
               s=1.5, c="tab:blue", alpha=0.6, label=f"Disc (tag={TAG_BOTTOM_DISC})")
    if len(rigid_midpoints) > 0:
        ax.scatter(rigid_midpoints[:, 0] * 1e3, rigid_midpoints[:, 1] * 1e3,
                   s=0.5, c="tab:gray", alpha=0.3, label=f"Rigid (tag={TAG_BOTTOM_RIGID})")

    # Expected circle
    theta = np.linspace(0, 2 * np.pi, 200)
    cx, cy = L / 2 * 1e3, L / 2 * 1e3
    ax.plot(cx + R_disc * 1e3 * np.cos(theta),
            cy + R_disc * 1e3 * np.sin(theta),
            "r-", lw=1.5, label=f"Expected R={R_disc*1e3:.1f} mm")

    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("Bottom Boundary: Disc vs Rigid Facet Classification")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "disc_facet_classification.png", dpi=200)
    plt.close(fig)

    return lines, ratio


# ═══════════════════════════════════════════════════════════════
# FIGURE GENERATION
# ═══════════════════════════════════════════════════════════════
def plot_contourf_abs_p(pvals_2d, cfg, title, path, n_levels=20):
    xs_mm = np.linspace(0, cfg.L * 1e3, pvals_2d.shape[1])
    ys_mm = np.linspace(0, cfg.L * 1e3, pvals_2d.shape[0])
    X, Y = np.meshgrid(xs_mm, ys_mm)
    data = np.abs(pvals_2d)
    maxp = np.nanmax(data)
    fig, ax = plt.subplots(figsize=(7, 5.8))
    cf = ax.contourf(X, Y, data, levels=n_levels, cmap="jet")
    ax.contour(X, Y, data, levels=n_levels, colors="k",
               linewidths=0.35, alpha=0.55)
    cb = fig.colorbar(cf, ax=ax, shrink=0.82, pad=0.03)
    cb.set_label("|p| (Pa)", fontsize=11)
    ax.add_patch(_disc_circle(cfg, "white"))
    ax.set_xlabel("x (mm)", fontsize=11)
    ax.set_ylabel("y (mm)", fontsize=11)
    ax.set_title(f"{title}\nIsosurface |p|  max = {maxp:.2f} Pa", fontsize=12)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_contourf_Re_p(pvals_2d, cfg, title, path, n_levels=20):
    xs_mm = np.linspace(0, cfg.L * 1e3, pvals_2d.shape[1])
    ys_mm = np.linspace(0, cfg.L * 1e3, pvals_2d.shape[0])
    X, Y = np.meshgrid(xs_mm, ys_mm)
    data = np.real(pvals_2d)
    vmax = max(abs(np.nanmin(data)), abs(np.nanmax(data)))
    if vmax < 1e-15:
        vmax = 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    fig, ax = plt.subplots(figsize=(7, 5.8))
    cf = ax.contourf(X, Y, data, levels=n_levels, cmap="RdBu_r", norm=norm)
    ax.contour(X, Y, data, levels=n_levels, colors="k",
               linewidths=0.3, alpha=0.5)
    cb = fig.colorbar(cf, ax=ax, shrink=0.82, pad=0.03)
    cb.set_label("Re(p) (Pa)", fontsize=11)
    ax.add_patch(_disc_circle(cfg, "black"))
    ax.set_xlabel("x (mm)", fontsize=11)
    ax.set_ylabel("y (mm)", fontsize=11)
    re_range = f"[{np.nanmin(data):.2f}, {np.nanmax(data):.2f}]"
    ax.set_title(f"{title}\nTotal pressure Re(p) {re_range} Pa", fontsize=12)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_radial_comparison(all_results, cfg, path):
    """Radial |p| profile comparison across all 3 variants."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, info in all_results.items():
        p2d = info["p2d"]
        mid_row = N_PLANE // 2
        radial = np.abs(p2d[mid_row, :])
        xs_mm = np.linspace(0, cfg.L * 1e3, N_PLANE)
        ax.plot(xs_mm, radial, label=label, linewidth=1.5)

    R_disc_mm = cfg.bottom_disc_radius_effective * 1e3
    cx_mm = cfg.L / 2 * 1e3
    ax.axvline(cx_mm - R_disc_mm, color="gray", ls="--", lw=0.8, alpha=0.6)
    ax.axvline(cx_mm + R_disc_mm, color="gray", ls="--", lw=0.8, alpha=0.6,
               label="disc edge")
    ax.axvline(cx_mm, color="gray", ls=":", lw=0.8, alpha=0.4)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("|p| (Pa)")
    ax.set_title("Radial |p| Profile at y = L/2, z = H/2 — Three Variants")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    cfg = CFG

    log(f"\n{'#'*70}")
    log(f"  DISC BC DEBUG — Case A (Standing-Only)")
    log(f"  Output: {OUTDIR.relative_to(ROOT)}")
    log(f"{'#'*70}\n")

    OUTDIR.mkdir(parents=True, exist_ok=True)

    # ── Mesh ──
    domain, facet_tags, tag_map = create_mesh(cfg, verbose=True)

    # ── H3: Disc area diagnostics ──
    log(f"\n{'='*70}")
    log("  H3 — DISC AREA TAGGING DIAGNOSTICS")
    log(f"{'='*70}")
    area_lines, area_ratio = disc_area_diagnostics(domain, facet_tags, cfg, OUTDIR)
    for line in area_lines:
        log(f"  {line}")

    # ── Sampling grid ──
    z_H2 = cfg.H / 2.0
    pts_H2 = _make_plane_grid(cfg, z_H2, N_PLANE)

    # Disc centre coordinates (for point sampling)
    cx = cfg.L / 2
    cy = cfg.L / 2
    disc_center_3d = np.array([[cx, cy, z_H2]])

    # ═══════════════════════════════════════════════════════════
    # THREE VARIANTS
    # ═══════════════════════════════════════════════════════════
    variants = {
        "V1_disc_robin_ON": {
            "disc_robin": True,
            "Z_disc_override": None,
            "label": "Variant 1: disc_robin=True (current Attempt 2)",
        },
        "V2_disc_robin_OFF": {
            "disc_robin": False,
            "Z_disc_override": None,
            "label": "Variant 2: disc_robin=False (rigid bottom)",
        },
        "V3_disc_robin_LARGE_Z": {
            "disc_robin": True,
            "Z_disc_override": 1000.0 * cfg.Z_water,
            "label": "Variant 3: disc_robin=True, Z_disc=1000×Z_w (≈rigid)",
        },
    }

    all_results = {}

    for vname, vspec in variants.items():
        log(f"\n{'='*70}")
        log(f"  {vspec['label']}")
        log(f"{'='*70}")

        vdir = OUTDIR / vname
        vdir.mkdir(parents=True, exist_ok=True)

        # Solve
        p_func, V_space = solve_standing_custom_disc(
            domain, facet_tags, cfg,
            disc_robin=vspec["disc_robin"],
            Z_disc_override=vspec["Z_disc_override"],
            verbose=True,
        )

        # Global metrics
        p_vals = p_func.x.array[:]
        maxp = np.max(np.abs(p_vals))
        log(f"  max|p| (3D) = {maxp:.4f} Pa")

        # Sample z = H/2 plane
        log("  Sampling z=H/2 plane...", end=" ")
        p_H2 = _sample_pressure(p_func, pts_H2)
        p2d = p_H2.reshape(N_PLANE, N_PLANE)
        plane_maxp = np.nanmax(np.abs(p_H2))
        log(f"max|p|(plane) = {plane_maxp:.4f}")

        # Disc centre |p|
        p_center = _sample_pressure(p_func, disc_center_3d)
        abs_p_center = np.abs(p_center[0]) if not np.isnan(p_center[0]) else 0.0
        log(f"  |p| at disc centre = {abs_p_center:.4f} Pa")

        # ── H2: Power flux ──
        if vspec["disc_robin"]:
            Z_disc_used = vspec["Z_disc_override"] if vspec["Z_disc_override"] else cfg.Z_water
            power_disc = compute_power_flux(p_func, V_space, domain, facet_tags, cfg,
                                            TAG_BOTTOM_DISC, "disc")
            log(f"  Power flux through disc: {power_disc:.6e} W")
        else:
            power_disc = 0.0
            log(f"  Power flux through disc: N/A (rigid)")

        power_top = compute_power_flux(p_func, V_space, domain, facet_tags, cfg,
                                       TAG_TOP, "top")
        log(f"  Power flux through top:  {power_top:.6e} W")

        if power_disc > 0 and power_top > 0:
            log(f"  Power ratio disc/top:    {power_disc/power_top:.4f}")

        # ── Figures ──
        title = vspec["label"]
        plot_contourf_abs_p(p2d, cfg, title,
                            vdir / "COMSOLstyle_abs_p_z_H2_contours.png")
        plot_contourf_Re_p(p2d, cfg, title,
                           vdir / "COMSOLstyle_total_pressure_Re_p_z_H2.png")
        log(f"  Saved figures to {vdir.relative_to(ROOT)}")

        all_results[vspec["label"]] = {
            "vname": vname,
            "maxp_3d": maxp,
            "maxp_plane": plane_maxp,
            "abs_p_center": abs_p_center,
            "power_disc": power_disc,
            "power_top": power_top,
            "area_ratio": area_ratio,
            "disc_robin": vspec["disc_robin"],
            "Z_disc_override": vspec["Z_disc_override"],
            "p2d": p2d,
        }

    # ── Radial comparison plot ──
    plot_radial_comparison(all_results, cfg, OUTDIR / "radial_comparison.png")
    log(f"\n  Saved radial comparison plot")

    # ═══════════════════════════════════════════════════════════
    # SUMMARY REPORT
    # ═══════════════════════════════════════════════════════════
    log(f"\n{'#'*70}")
    log(f"  SUMMARY REPORT")
    log(f"{'#'*70}\n")

    log(f"  {'Variant':<50} {'max|p|':>10} {'|p|_center':>12} "
        f"{'P_disc':>12} {'P_top':>12}")
    log(f"  {'-'*96}")
    for label, info in all_results.items():
        log(f"  {label:<50} {info['maxp_3d']:>10.4f} "
            f"{info['abs_p_center']:>12.4f} "
            f"{info['power_disc']:>12.4e} "
            f"{info['power_top']:>12.4e}")

    log(f"\n  Area ratio (disc mesh / πR²): {area_ratio:.4f}")
    log(f"  Area lines:")
    for line in area_lines:
        log(f"    {line}")

    # ── Decision logic ──
    v1 = all_results[list(all_results.keys())[0]]
    v2 = all_results[list(all_results.keys())[1]]
    v3 = all_results[list(all_results.keys())[2]]

    log(f"\n{'='*70}")
    log("  DIAGNOSIS")
    log(f"{'='*70}")

    # Check if V2 (rigid) has much higher centre pressure than V1 (impedance)
    ratio_center = v2["abs_p_center"] / max(v1["abs_p_center"], 1e-15)
    ratio_maxp = v2["maxp_3d"] / max(v1["maxp_3d"], 1e-15)

    log(f"  |p|_center ratio (rigid/impedance): {ratio_center:.2f}")
    log(f"  max|p| ratio (rigid/impedance):     {ratio_maxp:.2f}")

    if ratio_center > 2.0:
        log(f"\n  *** H1 CONFIRMED: disc Robin absorbs standing-wave energy ***")
        log(f"  The 'empty centre' disappears when disc is rigid.")
        log(f"  Fix: Case A benchmark should use disc_robin=False.")
        log(f"  COMSOL standing-only likely has rigid bottom everywhere.")
    else:
        log(f"\n  H1 not clearly confirmed (centre ratio = {ratio_center:.2f})")

    # Check V3 (large Z ≈ rigid) matches V2
    ratio_v3v2 = v3["abs_p_center"] / max(v2["abs_p_center"], 1e-15)
    log(f"  |p|_center ratio (large_Z / rigid): {ratio_v3v2:.4f}")
    if abs(ratio_v3v2 - 1.0) < 0.1:
        log(f"  V3 (Z→∞) matches V2 (rigid) — continuity check PASSED")
    else:
        log(f"  V3 (Z→∞) differs from V2 (rigid) by {abs(ratio_v3v2-1)*100:.1f}%")

    # Power analysis
    if v1["power_disc"] > 0:
        disc_to_top = v1["power_disc"] / max(v1["power_top"], 1e-30)
        log(f"\n  H2: Disc power / Top power = {disc_to_top:.2f}")
        if disc_to_top > 0.5:
            log(f"  *** Disc is absorbing substantial energy (> 50% of top) ***")
        elif disc_to_top > 0.1:
            log(f"  Disc absorbing noticeable energy ({disc_to_top*100:.1f}% of top)")
        else:
            log(f"  Disc absorption relatively small ({disc_to_top*100:.1f}% of top)")

    # Write JSON summary
    summary = {}
    for label, info in all_results.items():
        info_clean = {k: v for k, v in info.items() if k != "p2d"}
        # Convert numpy types for JSON
        for k, val in info_clean.items():
            if isinstance(val, (np.floating, np.integer)):
                info_clean[k] = float(val)
        summary[info["vname"]] = info_clean
    summary["area_diagnostics"] = {
        "lines": area_lines,
        "ratio": float(area_ratio),
    }
    summary["diagnosis"] = {
        "center_ratio_rigid_over_impedance": float(ratio_center),
        "maxp_ratio_rigid_over_impedance": float(ratio_maxp),
        "v3_v2_center_ratio": float(ratio_v3v2),
        "H1_confirmed": bool(ratio_center > 2.0),
    }

    with open(OUTDIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    elapsed = time.time() - t0
    log(f"\n{'#'*70}")
    log(f"  Done in {elapsed:.1f} s")
    log(f"  Output: {OUTDIR.relative_to(ROOT)}")
    log(f"{'#'*70}\n")


if __name__ == "__main__":
    main()
