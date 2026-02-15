#!/usr/bin/env python3
"""
Diagnostic: Disc BC mode investigation — impedance Robin vs pure Neumann.

The sign-flip test showed the "hole" is NOT a sign issue.
The cancellation diagnostic showed Δp correlates 0.89 with -p_stand inside disc.
The disc Robin absorbs standing-wave energy in combined mode (avg |p| drops 2×).

New hypothesis: COMSOL disc BC is a PURE VELOCITY SOURCE (Neumann), not
impedance + velocity (Robin + Neumann). The disc should be:
  - Standing-only (A):  rigid (disc_robin=False, no vortex) ← already fixed
  - Vortex-only (B):    pure Neumann source (disc_robin=False + vortex forcing)
  - Combined (C):       pure Neumann source (disc_robin=False + vortex forcing)

Variants tested:
  V1: disc Robin ON  + vortex (current)   = impedance absorber + source
  V2: disc Robin OFF + vortex (proposed)   = pure velocity source
  V3: disc Robin OFF, no vortex            = Case A reference (rigid)

For each: plot |p|, Re(p) at z=H/2, radial cross-section.

Usage:
    micromamba run -n acousto-complex python scripts/analysis/debug_disc_neumann_vs_robin.py
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
OUTDIR = ROOT / "COMSOL_comparison_debug" / f"disc_neumann_vs_robin_{STAMP}"
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


def solve_custom(domain, facet_tags, cfg, mode="combined",
                 disc_robin=True, include_vortex=True, verbose=True, label=""):
    """
    Helmholtz solve with full control over disc BC.

    disc_robin=True  + include_vortex=True  → Robin impedance + vortex source
    disc_robin=False + include_vortex=True  → pure Neumann vortex source (rigid + source)
    disc_robin=False + include_vortex=False → rigid bottom (Case A)
    """
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
            f"vortex={include_vortex} {label}")

    # Bilinear form
    a = (inner(grad(u), grad(v)) - k**2 * inner(u, v)) * dx

    # Top Robin
    if cfg.top_bc_type == "impedance":
        alpha_top = -1j * omega * rho / Z_top
        a += alpha_top * inner(u, v) * dss(TAG_TOP)

    # Disc Robin (impedance absorber)
    if disc_robin:
        alpha_disc = -1j * omega * rho / Z
        a += alpha_disc * inner(u, v) * dss(TAG_BOTTOM_DISC)

    # RHS terms
    L_terms = []

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

    # Vortex (disc)
    if include_vortex and mode in ("vortex", "combined"):
        V_vtx = cfg.vortex_velocity_amplitude
        g_vtx = _create_vortex_source(V, domain, facet_tags, cfg, verbose=False)
        g_vtx.x.array[:] *= -1j * omega * rho * V_vtx
        L_terms.append(inner(g_vtx, v) * dss(TAG_BOTTOM_DISC))

    if len(L_terms) == 0:
        raise ValueError(f"No source for mode={mode}, vortex={include_vortex}")

    L_form = L_terms[0]
    for term in L_terms[1:]:
        L_form = L_form + term

    problem = LinearProblem(a, L_form, bcs=[], petsc_options=MUMPS_OPTS)
    p_sol = problem.solve()

    maxp = np.max(np.abs(p_sol.x.array[:]))
    if verbose:
        log(f"    max|p| = {maxp:.4f} Pa")

    return p_sol, V


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
    for name, data in profiles.items():
        ax.plot(xs_mm, data, label=name, linewidth=1.3)
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


def main():
    t0 = time.time()
    cfg = CFG
    OUTDIR.mkdir(parents=True, exist_ok=True)

    log(f"\n{'#'*70}")
    log(f"  DISC NEUMANN vs ROBIN INVESTIGATION")
    log(f"  Output: {OUTDIR.relative_to(ROOT)}")
    log(f"{'#'*70}\n")

    domain, facet_tags, _ = create_mesh(cfg, verbose=True)
    z_H2 = cfg.H / 2.0
    pts_H2 = _make_plane_grid(cfg, z_H2, N_PLANE)
    mid = N_PLANE // 2

    # Disc mask for averaging
    xs_mm = np.linspace(0, cfg.L * 1e3, N_PLANE)
    ys_mm = np.linspace(0, cfg.L * 1e3, N_PLANE)
    Xg, Yg = np.meshgrid(xs_mm, ys_mm)
    cx_mm = cfg.L / 2 * 1e3
    R_mm = cfg.bottom_disc_radius_effective * 1e3
    inside_disc = np.sqrt((Xg - cx_mm)**2 + (Yg - cx_mm)**2) <= R_mm

    # ═══════════════════════════════════════════════════════
    # Reference: Case A standing (rigid bottom)
    # ═══════════════════════════════════════════════════════
    log(f"\n{'='*70}")
    log("  REFERENCE — Case A standing (rigid bottom)")
    log(f"{'='*70}")

    p_A, _ = solve_custom(domain, facet_tags, cfg,
                          mode="standing", disc_robin=False, include_vortex=False,
                          label="Case A (rigid)")
    p_A_H2 = _sample_pressure(p_A, pts_H2)
    p_A_2d = p_A_H2.reshape(N_PLANE, N_PLANE)

    # ═══════════════════════════════════════════════════════
    # Case B vortex: Robin vs pure Neumann
    # ═══════════════════════════════════════════════════════
    log(f"\n{'='*70}")
    log("  CASE B VORTEX — Robin vs Pure Neumann")
    log(f"{'='*70}")

    p_B_robin, _ = solve_custom(domain, facet_tags, cfg,
                                mode="vortex", disc_robin=True, include_vortex=True,
                                label="B: Robin+vortex")
    p_B_robin_H2 = _sample_pressure(p_B_robin, pts_H2)

    p_B_neumann, _ = solve_custom(domain, facet_tags, cfg,
                                  mode="vortex", disc_robin=False, include_vortex=True,
                                  label="B: Pure Neumann vortex")
    p_B_neumann_H2 = _sample_pressure(p_B_neumann, pts_H2)

    # ═══════════════════════════════════════════════════════
    # Case C combined V₀: Robin vs pure Neumann
    # ═══════════════════════════════════════════════════════
    log(f"\n{'='*70}")
    log("  CASE C COMBINED V₀ — Robin vs Pure Neumann")
    log(f"{'='*70}")

    p_C_robin, _ = solve_custom(domain, facet_tags, cfg,
                                mode="combined", disc_robin=True, include_vortex=True,
                                label="C: Robin+vortex V₀")
    p_C_robin_H2 = _sample_pressure(p_C_robin, pts_H2)
    p_C_robin_2d = p_C_robin_H2.reshape(N_PLANE, N_PLANE)

    p_C_neumann, _ = solve_custom(domain, facet_tags, cfg,
                                  mode="combined", disc_robin=False, include_vortex=True,
                                  label="C: Pure Neumann vortex V₀")
    p_C_neumann_H2 = _sample_pressure(p_C_neumann, pts_H2)
    p_C_neumann_2d = p_C_neumann_H2.reshape(N_PLANE, N_PLANE)

    # ═══════════════════════════════════════════════════════
    # Case C combined V₀×2: Robin vs pure Neumann
    # ═══════════════════════════════════════════════════════
    log(f"\n{'='*70}")
    log("  CASE C COMBINED V₀×2 — Robin vs Pure Neumann")
    log(f"{'='*70}")

    cfg_v2 = replace(cfg, vortex_velocity_amplitude=2 * cfg.vortex_velocity_amplitude)

    p_C2_robin, _ = solve_custom(domain, facet_tags, cfg_v2,
                                 mode="combined", disc_robin=True, include_vortex=True,
                                 label="C: Robin+vortex V₀×2")
    p_C2_robin_H2 = _sample_pressure(p_C2_robin, pts_H2)

    p_C2_neumann, _ = solve_custom(domain, facet_tags, cfg_v2,
                                   mode="combined", disc_robin=False, include_vortex=True,
                                   label="C: Pure Neumann vortex V₀×2")
    p_C2_neumann_H2 = _sample_pressure(p_C2_neumann, pts_H2)

    # ═══════════════════════════════════════════════════════
    # Case C combined V₀×3 and V₀×6
    # ═══════════════════════════════════════════════════════
    log(f"\n{'='*70}")
    log("  CASE C COMBINED V₀×3, V₀×6 — Pure Neumann")
    log(f"{'='*70}")

    cfg_v3 = replace(cfg, vortex_velocity_amplitude=3 * cfg.vortex_velocity_amplitude)
    p_C3_neumann, _ = solve_custom(domain, facet_tags, cfg_v3,
                                   mode="combined", disc_robin=False, include_vortex=True,
                                   label="C: Pure Neumann V₀×3")
    p_C3_neumann_H2 = _sample_pressure(p_C3_neumann, pts_H2)

    cfg_v6 = replace(cfg, vortex_velocity_amplitude=6 * cfg.vortex_velocity_amplitude)
    p_C6_neumann, _ = solve_custom(domain, facet_tags, cfg_v6,
                                   mode="combined", disc_robin=False, include_vortex=True,
                                   label="C: Pure Neumann V₀×6")
    p_C6_neumann_H2 = _sample_pressure(p_C6_neumann, pts_H2)

    # ═══════════════════════════════════════════════════════
    # Figures
    # ═══════════════════════════════════════════════════════
    log(f"\n  Generating figures...")

    # Case A reference
    plot_contourf(np.abs(p_A_2d), cfg, "Case A Standing (rigid bottom) |p|",
                  OUTDIR / "A_standing_rigid_abs_p.png")

    # Case C Robin vs Neumann |p|
    plot_contourf(np.abs(p_C_robin_2d), cfg,
                  "Case C V₀ Robin (impedance+vortex) |p|",
                  OUTDIR / "C_V0_robin_abs_p.png")
    plot_contourf(np.abs(p_C_neumann_2d), cfg,
                  "Case C V₀ Pure Neumann (vortex only) |p|",
                  OUTDIR / "C_V0_neumann_abs_p.png")

    # Case C Robin vs Neumann Re(p)
    plot_contourf(np.real(p_C_robin_2d), cfg,
                  "Case C V₀ Robin (impedance+vortex) Re(p)",
                  OUTDIR / "C_V0_robin_Re_p.png", cmap="RdBu_r",
                  label="Re(p) (Pa)", diverging=True)
    plot_contourf(np.real(p_C_neumann_2d), cfg,
                  "Case C V₀ Pure Neumann (vortex only) Re(p)",
                  OUTDIR / "C_V0_neumann_Re_p.png", cmap="RdBu_r",
                  label="Re(p) (Pa)", diverging=True)

    # V₀×2
    p_C2_robin_2d = p_C2_robin_H2.reshape(N_PLANE, N_PLANE)
    p_C2_neumann_2d = p_C2_neumann_H2.reshape(N_PLANE, N_PLANE)
    plot_contourf(np.abs(p_C2_robin_2d), cfg,
                  "Case C V₀×2 Robin |p|",
                  OUTDIR / "C_V0x2_robin_abs_p.png")
    plot_contourf(np.abs(p_C2_neumann_2d), cfg,
                  "Case C V₀×2 Pure Neumann |p|",
                  OUTDIR / "C_V0x2_neumann_abs_p.png")

    # V₀×3 and V₀×6 (Neumann only)
    p_C3_neumann_2d = p_C3_neumann_H2.reshape(N_PLANE, N_PLANE)
    p_C6_neumann_2d = p_C6_neumann_H2.reshape(N_PLANE, N_PLANE)
    plot_contourf(np.abs(p_C3_neumann_2d), cfg,
                  "Case C V₀×3 Pure Neumann |p|",
                  OUTDIR / "C_V0x3_neumann_abs_p.png")
    plot_contourf(np.abs(p_C6_neumann_2d), cfg,
                  "Case C V₀×6 Pure Neumann |p|",
                  OUTDIR / "C_V0x6_neumann_abs_p.png")
    plot_contourf(np.real(p_C3_neumann_2d), cfg,
                  "Case C V₀×3 Pure Neumann Re(p)",
                  OUTDIR / "C_V0x3_neumann_Re_p.png", cmap="RdBu_r",
                  label="Re(p) (Pa)", diverging=True)
    plot_contourf(np.real(p_C6_neumann_2d), cfg,
                  "Case C V₀×6 Pure Neumann Re(p)",
                  OUTDIR / "C_V0x6_neumann_Re_p.png", cmap="RdBu_r",
                  label="Re(p) (Pa)", diverging=True)

    # Radial comparison
    mid_row = N_PLANE // 2
    radial_profiles = {
        "A standing (rigid)": np.abs(p_A_2d[mid_row, :]),
        "C V₀ Robin": np.abs(p_C_robin_2d[mid_row, :]),
        "C V₀ Neumann": np.abs(p_C_neumann_2d[mid_row, :]),
        "C V₀×2 Robin": np.abs(p_C2_robin_2d[mid_row, :]),
        "C V₀×2 Neumann": np.abs(p_C2_neumann_2d[mid_row, :]),
    }
    plot_radial(radial_profiles, cfg,
                "Radial |p| at y=L/2, z=H/2 — Robin vs Neumann disc",
                OUTDIR / "radial_robin_vs_neumann.png")

    radial_sweep = {
        "A standing": np.abs(p_A_2d[mid_row, :]),
        "C V₀ Neumann": np.abs(p_C_neumann_2d[mid_row, :]),
        "C V₀×2 Neumann": np.abs(p_C2_neumann_2d[mid_row, :]),
        "C V₀×3 Neumann": np.abs(p_C3_neumann_2d[mid_row, :]),
        "C V₀×6 Neumann": np.abs(p_C6_neumann_2d[mid_row, :]),
    }
    plot_radial(radial_sweep, cfg,
                "Amplitude sweep (Neumann disc) — radial |p|",
                OUTDIR / "radial_neumann_sweep.png")

    # ═══════════════════════════════════════════════════════
    # Summary metrics
    # ═══════════════════════════════════════════════════════
    log(f"\n{'#'*70}")
    log("  SUMMARY")
    log(f"{'#'*70}")

    def metrics(p_H2_vals):
        p2d = p_H2_vals.reshape(N_PLANE, N_PLANE)
        mp = np.nanmax(np.abs(p_H2_vals))
        cp = abs(p2d[mid, mid])
        disc_avg = np.nanmean(np.abs(p2d[inside_disc]))
        return mp, cp, disc_avg

    rows = [
        ("A standing (rigid)",      *metrics(p_A_H2)),
        ("B vortex Robin",          *metrics(p_B_robin_H2)),
        ("B vortex Neumann",        *metrics(p_B_neumann_H2)),
        ("C V₀ Robin",             *metrics(p_C_robin_H2)),
        ("C V₀ Neumann",           *metrics(p_C_neumann_H2)),
        ("C V₀×2 Robin",           *metrics(p_C2_robin_H2)),
        ("C V₀×2 Neumann",         *metrics(p_C2_neumann_H2)),
        ("C V₀×3 Neumann",         *metrics(p_C3_neumann_H2)),
        ("C V₀×6 Neumann",         *metrics(p_C6_neumann_H2)),
    ]

    log(f"\n  {'Case':<26} {'max|p|':>10} {'|p|_ctr':>10} {'avg|p|_disc':>12}")
    log(f"  {'-'*60}")
    for label, mp, cp, da in rows:
        log(f"  {label:<26} {mp:>8.2f} Pa {cp:>8.4f} Pa {da:>10.4f} Pa")

    # Robin vs Neumann disc average for C V₀
    _, _, robin_avg = metrics(p_C_robin_H2)
    _, _, neumann_avg = metrics(p_C_neumann_H2)
    _, _, stand_avg = metrics(p_A_H2)

    log(f"\n  KEY COMPARISON (disc-average |p|):")
    log(f"    A standing (rigid):     {stand_avg:.4f} Pa")
    log(f"    C V₀ Robin:             {robin_avg:.4f} Pa  "
        f"(= {robin_avg/stand_avg:.1%} of standing)")
    log(f"    C V₀ Neumann:           {neumann_avg:.4f} Pa  "
        f"(= {neumann_avg/stand_avg:.1%} of standing)")

    if neumann_avg / stand_avg > 0.85:
        log(f"\n  >>> NEUMANN DISC PRESERVES STANDING PATTERN <<<")
        log(f"  >>> Fix: use disc_robin=False for ALL cases <<<")
        log(f"  >>> Disc is a pure velocity source, not an impedance absorber <<<")
    elif neumann_avg > robin_avg * 1.3:
        log(f"\n  >>> NEUMANN is better than ROBIN (disc avg +{(neumann_avg/robin_avg-1)*100:.0f}%)")
    else:
        log(f"\n  >>> No significant improvement from Neumann ({neumann_avg/robin_avg:.2f}×)")

    # Save JSON
    summary = {}
    for label, mp, cp, da in rows:
        summary[label] = {"maxp_plane": float(mp), "p_center": float(cp),
                          "disc_avg": float(da)}
    with open(OUTDIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    elapsed = time.time() - t0
    log(f"\n  Done in {elapsed:.1f} s")
    log(f"  Output: {OUTDIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
