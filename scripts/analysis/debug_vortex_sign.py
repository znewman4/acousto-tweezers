#!/usr/bin/env python3
"""
Diagnostic: vortex sign convention investigation for Case C "hole".

Steps:
  1. Confirm combined is single solve (print RHS norms)
  2. Cancellation diagnostic: Δp = p_comb - p_stand, plot Re/Im/|·|/arg
  3. Run sign-flipped vortex (g_vtx = +iωρ V₀ pattern) for B and C
  4. Compare original vs flipped at V₀ and V₀×2

Usage:
    micromamba run -n acousto-complex python scripts/analysis/debug_vortex_sign.py
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
from ufl import inner, grad, dx, TrialFunction, TestFunction, Measure

from acoustweezers.experiments.shallow_square_dish.config import ShallowDishConfig
from acoustweezers.experiments.shallow_square_dish.solve_pressure import (
    create_mesh,
    _create_vortex_source,
    TAG_BOTTOM_DISC, TAG_BOTTOM_RIGID,
    TAG_TOP, TAG_X0, TAG_XL, TAG_Y0, TAG_YL,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import TwoSlopeNorm

comm = MPI.COMM_WORLD
rank = comm.rank
ROOT = Path(__file__).resolve().parents[2]
NOW = datetime.now()
STAMP = NOW.strftime("%Y%m%d_%H%M")
OUTDIR = ROOT / "COMSOL_comparison_debug" / f"vortex_sign_investigation_{STAMP}"

N_PLANE = 201
MUMPS_OPTS = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}


def log(msg="", end="\n"):
    if rank == 0:
        print(msg, end=end, flush=True)


CFG = ShallowDishConfig(
    L=10e-3, H=1e-3, frequency_hz=500e3,
    elements_per_wavelength=10, min_elements_z=8,
    rho=997.0, c=1484.0, mu=1.002e-3,
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
# SOLVER with sign control
# ═══════════════════════════════════════════════════════════════
def solve_helmholtz_signtest(
    domain, facet_tags, cfg,
    mode="combined",
    disc_robin=True,
    vortex_sign=-1,       # -1 = original (-iωρ), +1 = flipped (+iωρ)
    verbose=True,
    label="",
):
    """
    Helmholtz solver with explicit vortex forcing sign control.

    vortex_sign = -1  →  g_vtx = -iωρ V₀ pattern   (original)
    vortex_sign = +1  →  g_vtx = +iωρ V₀ pattern   (sign-flipped)
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

    if verbose:
        log(f"\n  SOLVE [{mode.upper()}] disc_robin={disc_robin} "
            f"vortex_sign={vortex_sign:+d} {label}")

    # Bilinear form
    a = (inner(grad(u), grad(v)) - k**2 * inner(u, v)) * dx

    # Top Robin
    if cfg.top_bc_type == "impedance":
        alpha_top = -1j * omega * rho / Z_top
        a += alpha_top * inner(u, v) * dss(TAG_TOP)

    # Disc Robin
    if disc_robin:
        alpha_disc = -1j * omega * rho / Z
        a += alpha_disc * inner(u, v) * dss(TAG_BOTTOM_DISC)

    # RHS terms
    L_terms = []
    rhs_norms = {}

    # Standing wave (side walls)
    if mode in ("standing", "combined"):
        V_stand = cfg.standing_velocity_amplitude
        g_stand = -1j * omega * rho * V_stand

        if cfg.standing_phase_pattern == "antiphase":
            L_terms.append(inner(g_stand, v) * dss(TAG_X0))
            L_terms.append(inner(-g_stand, v) * dss(TAG_XL))
        if cfg.standing_axis == "both":
            if cfg.standing_phase_pattern == "antiphase":
                L_terms.append(inner(g_stand, v) * dss(TAG_Y0))
                L_terms.append(inner(-g_stand, v) * dss(TAG_YL))

        # Assemble standing-only RHS norm for diagnostics
        L_stand = inner(g_stand, v) * dss(TAG_X0) + inner(-g_stand, v) * dss(TAG_XL)
        if cfg.standing_axis == "both":
            L_stand = L_stand + inner(g_stand, v) * dss(TAG_Y0) + inner(-g_stand, v) * dss(TAG_YL)
        b_stand_vec = fem.petsc.assemble_vector(fem.form(L_stand))
        b_stand_vec.ghostUpdate()
        rhs_norms["standing"] = b_stand_vec.norm()
        b_stand_vec.destroy()

    # Vortex (disc)
    if mode in ("vortex", "combined"):
        V_vtx = cfg.vortex_velocity_amplitude

        g_vtx = _create_vortex_source(V, domain, facet_tags, cfg,
                                       verbose=False)
        # Apply sign-controlled scaling
        g_vtx.x.array[:] *= vortex_sign * 1j * omega * rho * V_vtx

        L_vtx_form = inner(g_vtx, v) * dss(TAG_BOTTOM_DISC)
        L_terms.append(L_vtx_form)

        # Assemble vortex-only RHS norm
        b_vtx_vec = fem.petsc.assemble_vector(fem.form(L_vtx_form))
        b_vtx_vec.ghostUpdate()
        rhs_norms["vortex"] = b_vtx_vec.norm()
        b_vtx_vec.destroy()

    if len(L_terms) == 0:
        raise ValueError(f"No source for mode={mode}")

    L_form = L_terms[0]
    for term in L_terms[1:]:
        L_form = L_form + term

    # Combined RHS norm
    b_comb_vec = fem.petsc.assemble_vector(fem.form(L_form))
    b_comb_vec.ghostUpdate()
    rhs_norms["combined"] = b_comb_vec.norm()
    b_comb_vec.destroy()

    if verbose:
        for k_name, val in rhs_norms.items():
            log(f"    RHS norm ({k_name}): {val:.6e}")

    problem = LinearProblem(a, L_form, bcs=[], petsc_options=MUMPS_OPTS)
    p_sol = problem.solve()

    maxp = np.max(np.abs(p_sol.x.array[:]))
    if verbose:
        log(f"    max|p| = {maxp:.4f} Pa")

    return p_sol, V, rhs_norms


# ═══════════════════════════════════════════════════════════════
# FIGURE HELPERS
# ═══════════════════════════════════════════════════════════════
def plot_contourf(data_2d, cfg, title, path, cmap="jet", label="|p|",
                  n_levels=20, diverging=False):
    xs_mm = np.linspace(0, cfg.L * 1e3, data_2d.shape[1])
    ys_mm = np.linspace(0, cfg.L * 1e3, data_2d.shape[0])
    X, Y = np.meshgrid(xs_mm, ys_mm)
    vmax = max(abs(np.nanmin(data_2d)), abs(np.nanmax(data_2d)))
    if vmax < 1e-15:
        vmax = 1.0

    fig, ax = plt.subplots(figsize=(7, 5.8))
    if diverging:
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        cf = ax.contourf(X, Y, data_2d, levels=n_levels, cmap=cmap, norm=norm)
    else:
        cf = ax.contourf(X, Y, data_2d, levels=n_levels, cmap=cmap)
    ax.contour(X, Y, data_2d, levels=n_levels, colors="k",
               linewidths=0.3, alpha=0.5)
    cb = fig.colorbar(cf, ax=ax, shrink=0.82, pad=0.03)
    cb.set_label(label, fontsize=11)
    disc_color = "black" if diverging else "white"
    ax.add_patch(_disc_circle(cfg, disc_color))
    ax.set_xlabel("x (mm)", fontsize=11)
    ax.set_ylabel("y (mm)", fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_radial(profiles, cfg, title, path, ylabel="|p| (Pa)"):
    fig, ax = plt.subplots(figsize=(10, 5))
    xs_mm = np.linspace(0, cfg.L * 1e3, N_PLANE)
    for label, data in profiles.items():
        ax.plot(xs_mm, data, label=label, linewidth=1.3)
    R_mm = cfg.bottom_disc_radius_effective * 1e3
    cx_mm = cfg.L / 2 * 1e3
    ax.axvline(cx_mm - R_mm, color="gray", ls="--", lw=0.7, alpha=0.5)
    ax.axvline(cx_mm + R_mm, color="gray", ls="--", lw=0.7, alpha=0.5,
               label="disc edge")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
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
    OUTDIR.mkdir(parents=True, exist_ok=True)

    log(f"\n{'#'*70}")
    log(f"  VORTEX SIGN INVESTIGATION — Case C hole diagnostic")
    log(f"  Output: {OUTDIR.relative_to(ROOT)}")
    log(f"{'#'*70}\n")

    # Mesh
    domain, facet_tags, tag_map = create_mesh(cfg, verbose=True)
    z_H2 = cfg.H / 2.0
    pts_H2 = _make_plane_grid(cfg, z_H2, N_PLANE)
    disc_center_pts = np.array([[cfg.L / 2, cfg.L / 2, z_H2]])

    # ══════════════════════════════════════════════════════════
    # Step 1: Compute reference fields
    # ══════════════════════════════════════════════════════════
    log(f"\n{'='*70}")
    log("  STEP 1 — Reference solves (original sign)")
    log(f"{'='*70}")

    # Case A standing (rigid bottom) — reference
    p_stand, _, norms_stand = solve_helmholtz_signtest(
        domain, facet_tags, cfg,
        mode="standing", disc_robin=False, vortex_sign=-1,
        label="Case A (standing, rigid)")

    p_stand_H2 = _sample_pressure(p_stand, pts_H2)
    p_stand_2d = p_stand_H2.reshape(N_PLANE, N_PLANE)

    # Case B vortex (original sign)
    p_vtx_orig, _, norms_vtx = solve_helmholtz_signtest(
        domain, facet_tags, cfg,
        mode="vortex", disc_robin=True, vortex_sign=-1,
        label="Case B orig")

    p_vtx_orig_H2 = _sample_pressure(p_vtx_orig, pts_H2)

    # Case C combined (original sign, V₀)
    p_comb_orig, _, norms_comb = solve_helmholtz_signtest(
        domain, facet_tags, cfg,
        mode="combined", disc_robin=True, vortex_sign=-1,
        label="Case C orig V₀")

    p_comb_orig_H2 = _sample_pressure(p_comb_orig, pts_H2)
    p_comb_orig_2d = p_comb_orig_H2.reshape(N_PLANE, N_PLANE)

    # Case C combined (original sign, V₀×2)
    cfg_v2 = replace(cfg, vortex_velocity_amplitude=2 * cfg.vortex_velocity_amplitude)
    p_comb_orig_v2, _, _ = solve_helmholtz_signtest(
        domain, facet_tags, cfg_v2,
        mode="combined", disc_robin=True, vortex_sign=-1,
        label="Case C orig V₀×2")

    p_comb_orig_v2_H2 = _sample_pressure(p_comb_orig_v2, pts_H2)

    # ══════════════════════════════════════════════════════════
    # Step 2: Cancellation diagnostic
    # ══════════════════════════════════════════════════════════
    log(f"\n{'='*70}")
    log("  STEP 2 — Cancellation diagnostic: Δp = p_comb - p_stand")
    log(f"{'='*70}")

    delta_p = p_comb_orig_H2 - p_stand_H2
    delta_2d = delta_p.reshape(N_PLANE, N_PLANE)

    # Disc-centre values
    mid = N_PLANE // 2
    log(f"  p_stand at centre:  {p_stand_2d[mid,mid]:.4f}")
    log(f"  p_comb  at centre:  {p_comb_orig_2d[mid,mid]:.4f}")
    log(f"  Δp      at centre:  {delta_2d[mid,mid]:.4f}")
    log(f"  |Δp|    at centre:  {abs(delta_2d[mid,mid]):.4f}")

    # Check correlation with -p_stand inside disc
    R_disc = cfg.bottom_disc_radius_effective
    cx_mm = cfg.L / 2 * 1e3
    xs_mm = np.linspace(0, cfg.L * 1e3, N_PLANE)
    ys_mm = np.linspace(0, cfg.L * 1e3, N_PLANE)
    Xg, Yg = np.meshgrid(xs_mm, ys_mm)
    r_from_center = np.sqrt((Xg - cx_mm)**2 + (Yg - cx_mm)**2)
    inside_disc = r_from_center <= R_disc * 1e3

    # Correlation: is Δp ≈ -p_stand inside disc?
    delta_flat = delta_2d[inside_disc]
    stand_flat = p_stand_2d[inside_disc]
    neg_stand_flat = -stand_flat

    if np.any(np.isfinite(delta_flat)) and np.any(np.isfinite(neg_stand_flat)):
        mask = np.isfinite(delta_flat) & np.isfinite(neg_stand_flat)
        corr = np.abs(np.vdot(delta_flat[mask], neg_stand_flat[mask])) / (
            np.linalg.norm(delta_flat[mask]) * np.linalg.norm(neg_stand_flat[mask]) + 1e-30)
        log(f"  Correlation(Δp, -p_stand) inside disc: {corr:.4f}")
        log(f"    (1.0 = perfect cancellation, Δp ~ -p_stand)")

    # Δp plots
    ddir = OUTDIR / "step2_cancellation"
    ddir.mkdir(parents=True, exist_ok=True)

    plot_contourf(np.real(delta_2d), cfg,
                  "Δp = p_comb − p_stand : Re(Δp)",
                  ddir / "delta_p_Re.png", cmap="RdBu_r",
                  label="Re(Δp) (Pa)", diverging=True)
    plot_contourf(np.imag(delta_2d), cfg,
                  "Δp = p_comb − p_stand : Im(Δp)",
                  ddir / "delta_p_Im.png", cmap="RdBu_r",
                  label="Im(Δp) (Pa)", diverging=True)
    plot_contourf(np.abs(delta_2d), cfg,
                  "Δp = p_comb − p_stand : |Δp|",
                  ddir / "delta_p_abs.png", cmap="jet",
                  label="|Δp| (Pa)")
    plot_contourf(np.angle(delta_2d), cfg,
                  "Δp = p_comb − p_stand : arg(Δp)",
                  ddir / "delta_p_arg.png", cmap="twilight_shifted",
                  label="arg(Δp) (rad)")
    log("  Saved Δp plots")

    # ══════════════════════════════════════════════════════════
    # Step 3: Sign-flipped vortex
    # ══════════════════════════════════════════════════════════
    log(f"\n{'='*70}")
    log("  STEP 3 — Sign-flipped vortex (g_vtx = +iωρ V₀ pattern)")
    log(f"{'='*70}")

    # Case B flipped
    p_vtx_flip, _, _ = solve_helmholtz_signtest(
        domain, facet_tags, cfg,
        mode="vortex", disc_robin=True, vortex_sign=+1,
        label="Case B FLIPPED")

    p_vtx_flip_H2 = _sample_pressure(p_vtx_flip, pts_H2)

    # Case C flipped V₀
    p_comb_flip, _, norms_comb_flip = solve_helmholtz_signtest(
        domain, facet_tags, cfg,
        mode="combined", disc_robin=True, vortex_sign=+1,
        label="Case C FLIPPED V₀")

    p_comb_flip_H2 = _sample_pressure(p_comb_flip, pts_H2)
    p_comb_flip_2d = p_comb_flip_H2.reshape(N_PLANE, N_PLANE)

    # Case C flipped V₀×2
    p_comb_flip_v2, _, _ = solve_helmholtz_signtest(
        domain, facet_tags, cfg_v2,
        mode="combined", disc_robin=True, vortex_sign=+1,
        label="Case C FLIPPED V₀×2")

    p_comb_flip_v2_H2 = _sample_pressure(p_comb_flip_v2, pts_H2)

    # ══════════════════════════════════════════════════════════
    # Step 3b: Δp for flipped sign
    # ══════════════════════════════════════════════════════════
    delta_flip = p_comb_flip_H2 - p_stand_H2
    delta_flip_2d = delta_flip.reshape(N_PLANE, N_PLANE)

    log(f"\n  Flipped Δp at centre: {delta_flip_2d[mid,mid]:.4f}")
    log(f"  Flipped |Δp| at centre: {abs(delta_flip_2d[mid,mid]):.4f}")

    # ══════════════════════════════════════════════════════════
    # Figures: compare original vs flipped
    # ══════════════════════════════════════════════════════════
    fdir = OUTDIR / "step3_sign_comparison"
    fdir.mkdir(parents=True, exist_ok=True)

    # Case B
    p_vtx_orig_2d = p_vtx_orig_H2.reshape(N_PLANE, N_PLANE)
    p_vtx_flip_2d = p_vtx_flip_H2.reshape(N_PLANE, N_PLANE)
    plot_contourf(np.abs(p_vtx_orig_2d), cfg,
                  "Case B Vortex (original sign -iωρ) |p|",
                  fdir / "B_orig_abs_p.png")
    plot_contourf(np.abs(p_vtx_flip_2d), cfg,
                  "Case B Vortex (flipped sign +iωρ) |p|",
                  fdir / "B_flip_abs_p.png")

    # Case C V₀ original vs flipped
    plot_contourf(np.abs(p_comb_orig_2d), cfg,
                  "Case C Combined V₀ (original -iωρ) |p|",
                  fdir / "C_V0_orig_abs_p.png")
    plot_contourf(np.abs(p_comb_flip_2d), cfg,
                  "Case C Combined V₀ (flipped +iωρ) |p|",
                  fdir / "C_V0_flip_abs_p.png")

    # Re(p) for C V₀
    plot_contourf(np.real(p_comb_orig_2d), cfg,
                  "Case C Combined V₀ (original -iωρ) Re(p)",
                  fdir / "C_V0_orig_Re_p.png", cmap="RdBu_r",
                  label="Re(p) (Pa)", diverging=True)
    plot_contourf(np.real(p_comb_flip_2d), cfg,
                  "Case C Combined V₀ (flipped +iωρ) Re(p)",
                  fdir / "C_V0_flip_Re_p.png", cmap="RdBu_r",
                  label="Re(p) (Pa)", diverging=True)

    # Case C V₀×2 original vs flipped
    p_comb_orig_v2_2d = p_comb_orig_v2_H2.reshape(N_PLANE, N_PLANE)
    p_comb_flip_v2_2d = p_comb_flip_v2_H2.reshape(N_PLANE, N_PLANE)

    plot_contourf(np.abs(p_comb_orig_v2_2d), cfg,
                  "Case C Combined V₀×2 (original -iωρ) |p|",
                  fdir / "C_V0x2_orig_abs_p.png")
    plot_contourf(np.abs(p_comb_flip_v2_2d), cfg,
                  "Case C Combined V₀×2 (flipped +iωρ) |p|",
                  fdir / "C_V0x2_flip_abs_p.png")

    plot_contourf(np.real(p_comb_orig_v2_2d), cfg,
                  "Case C Combined V₀×2 (original -iωρ) Re(p)",
                  fdir / "C_V0x2_orig_Re_p.png", cmap="RdBu_r",
                  label="Re(p) (Pa)", diverging=True)
    plot_contourf(np.real(p_comb_flip_v2_2d), cfg,
                  "Case C Combined V₀×2 (flipped +iωρ) Re(p)",
                  fdir / "C_V0x2_flip_Re_p.png", cmap="RdBu_r",
                  label="Re(p) (Pa)", diverging=True)

    # Δp for flipped
    plot_contourf(np.abs(delta_flip_2d), cfg,
                  "Δp_flip = p_comb_flip − p_stand : |Δp|",
                  fdir / "delta_p_flip_abs.png", cmap="jet",
                  label="|Δp| (Pa)")

    # Radial comparison
    mid_row = N_PLANE // 2
    profiles_abs = {
        "A standing (rigid)": np.abs(p_stand_2d[mid_row, :]),
        "C orig V₀ (−iωρ)": np.abs(p_comb_orig_2d[mid_row, :]),
        "C flip V₀ (+iωρ)": np.abs(p_comb_flip_2d[mid_row, :]),
        "C orig V₀×2 (−iωρ)": np.abs(p_comb_orig_v2_2d[mid_row, :]),
        "C flip V₀×2 (+iωρ)": np.abs(p_comb_flip_v2_2d[mid_row, :]),
    }
    plot_radial(profiles_abs, cfg,
                "Radial |p| comparison: original vs flipped vortex sign",
                fdir / "radial_comparison_abs.png")

    log("  Saved comparison figures")

    # ══════════════════════════════════════════════════════════
    # Summary metrics
    # ══════════════════════════════════════════════════════════
    log(f"\n{'#'*70}")
    log("  SUMMARY")
    log(f"{'#'*70}")

    def pmetrics(label, p_H2_vals):
        p2d = p_H2_vals.reshape(N_PLANE, N_PLANE)
        maxp = np.nanmax(np.abs(p_H2_vals))
        cp = abs(p2d[mid, mid])
        return maxp, cp

    rows = [
        ("A standing (rigid)",    *pmetrics("A", p_stand_H2)),
        ("B vortex orig (-iωρ)",  *pmetrics("B", p_vtx_orig_H2)),
        ("B vortex flip (+iωρ)",  *pmetrics("Bf", p_vtx_flip_H2)),
        ("C comb orig V₀",       *pmetrics("Co", p_comb_orig_H2)),
        ("C comb flip V₀",       *pmetrics("Cf", p_comb_flip_H2)),
        ("C comb orig V₀×2",     *pmetrics("Co2", p_comb_orig_v2_H2)),
        ("C comb flip V₀×2",     *pmetrics("Cf2", p_comb_flip_v2_H2)),
    ]

    log(f"\n  {'Case':<30} {'max|p|(plane)':>14} {'|p|_center':>12}")
    log(f"  {'-'*58}")
    for label, maxp, cp in rows:
        log(f"  {label:<30} {maxp:>12.4f} Pa {cp:>10.4f} Pa")

    # Check: does flipping help the "hole"?
    _, cp_orig = pmetrics("", p_comb_orig_H2)
    _, cp_flip = pmetrics("", p_comb_flip_H2)
    _, cp_stand = pmetrics("", p_stand_H2)

    log(f"\n  DECISION CRITERIA:")
    log(f"    |p|_center(stand):      {cp_stand:.4f}")
    log(f"    |p|_center(comb orig):  {cp_orig:.4f}")
    log(f"    |p|_center(comb flip):  {cp_flip:.4f}")

    # Inside-disc average |p| comparison
    orig_disc_avg = np.nanmean(np.abs(p_comb_orig_2d[inside_disc]))
    flip_disc_avg = np.nanmean(np.abs(p_comb_flip_2d[inside_disc]))
    stand_disc_avg = np.nanmean(np.abs(p_stand_2d[inside_disc]))

    log(f"    avg|p|_disc(stand):     {stand_disc_avg:.4f}")
    log(f"    avg|p|_disc(comb orig): {orig_disc_avg:.4f}")
    log(f"    avg|p|_disc(comb flip): {flip_disc_avg:.4f}")

    if flip_disc_avg > orig_disc_avg * 1.5:
        log(f"\n  >>> SIGN FLIP HELPS: disc-avg |p| improved by "
            f"{flip_disc_avg/orig_disc_avg:.1f}×")
        log(f"  >>> Recommend: change vortex forcing sign to +iωρ V₀ pattern")
    elif orig_disc_avg > flip_disc_avg * 1.5:
        log(f"\n  >>> SIGN FLIP MAKES IT WORSE")
        log(f"  >>> Original sign is correct; issue is elsewhere")
    else:
        log(f"\n  >>> SIGN FLIP has marginal effect ({flip_disc_avg/orig_disc_avg:.2f}×)")
        log(f"  >>> Issue likely not sign-related")

    # RHS norms
    log(f"\n  RHS norms (combined, original sign):")
    for k_name, val in norms_comb.items():
        log(f"    {k_name}: {val:.6e}")
    log(f"  RHS norms (combined, flipped sign):")
    for k_name, val in norms_comb_flip.items():
        log(f"    {k_name}: {val:.6e}")

    # Save summary JSON
    summary = {
        "cases": {r[0]: {"maxp_plane": float(r[1]), "p_center": float(r[2])} for r in rows},
        "disc_avg_abs_p": {
            "standing": float(stand_disc_avg),
            "comb_orig": float(orig_disc_avg),
            "comb_flip": float(flip_disc_avg),
        },
        "rhs_norms_orig": {k: float(v) for k, v in norms_comb.items()},
        "rhs_norms_flip": {k: float(v) for k, v in norms_comb_flip.items()},
    }
    with open(OUTDIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    elapsed = time.time() - t0
    log(f"\n  Done in {elapsed:.1f} s")
    log(f"  Output: {OUTDIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
