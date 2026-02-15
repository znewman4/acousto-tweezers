#!/usr/bin/env python3
"""
Streaming FEM norms, boundary flux, CSV summary, and mesh-refinement symmetry test.

Computes proper L2 norms via assembled UFL forms (not pointwise sampling):
  ‖∇·u‖_L2, ‖u‖_L2, ‖∇u‖_L2, net boundary flux ∫u·n dS

Then reruns standing-only at 30×30×12 and reports symmetry metric drop.

Usage:
    micromamba run -n acousto-complex python scripts/analysis/run_streaming_norms.py
"""
from __future__ import annotations
import sys, os, csv, time
from pathlib import Path
from datetime import datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from mpi4py import MPI
import ufl
from ufl import inner, grad, div, dx
from dolfinx import fem, mesh as dmesh

from acoustweezers.experiments.shallow_square_dish.config import ShallowDishConfig
from acoustweezers.experiments.shallow_square_dish.solve_pressure import create_mesh, solve_helmholtz
from acoustweezers.experiments.shallow_square_dish.streaming import solve_streaming_stokes

comm = MPI.COMM_WORLD
TODAY = datetime.now().strftime("%Y-%m-%d")
OUTDIR = Path(f"results/latest/streaming_norms_{TODAY}")

def log(msg=""):
    if comm.rank == 0:
        print(msg, flush=True)


# ════════════════════════════════════════════════════════════════════
# Configs
# ════════════════════════════════════════════════════════════════════
def make_cfg(epw=6, min_z=8):
    return ShallowDishConfig(
        L=10e-3, H=1e-3,
        frequency_hz=500e3,
        elements_per_wavelength=epw,
        min_elements_z=min_z,
        vortex_velocity_amplitude=10e-6,
        standing_velocity_amplitude=10e-6,
        vortex_aperture_radius=3e-3,
        standing_axis="both",
        standing_phase_pattern="antiphase",
    )


# ════════════════════════════════════════════════════════════════════
# FEM norm computation
# ════════════════════════════════════════════════════════════════════
def compute_fem_norms(u_func):
    """Compute ‖u‖_L2, ‖∇u‖_L2, ‖∇·u‖_L2 via assembled UFL forms."""
    u = u_func
    # ‖u‖²_L2 = ∫ u·u dx
    u_l2_sq = np.real(fem.assemble_scalar(fem.form(inner(u, u) * dx)))
    u_l2 = float(np.sqrt(max(u_l2_sq, 0.0)))

    # ‖∇u‖²_L2 = ∫ ∇u : ∇u dx
    gradu_l2_sq = np.real(fem.assemble_scalar(fem.form(inner(grad(u), grad(u)) * dx)))
    gradu_l2 = float(np.sqrt(max(gradu_l2_sq, 0.0)))

    # ‖∇·u‖²_L2 = ∫ (∇·u)² dx
    divu = div(u)
    divu_l2_sq = np.real(fem.assemble_scalar(fem.form(inner(divu, divu) * dx)))
    divu_l2 = float(np.sqrt(max(divu_l2_sq, 0.0)))

    return u_l2, gradu_l2, divu_l2


def compute_boundary_flux(u_func, domain):
    """Compute net flux ∫ u·n dS over the entire boundary."""
    n = ufl.FacetNormal(domain)
    ds = ufl.Measure("ds", domain=domain)
    flux_form = fem.form(inner(u_func, n) * ds)
    flux = np.real(fem.assemble_scalar(flux_form))
    return float(flux)


def compute_velocity_stats(u_func):
    """max|u|, mean|u|, 95th percentile |u|."""
    vals = u_func.x.array.copy()
    n = len(vals) // 3
    umag = np.linalg.norm(vals.reshape((n, 3)), axis=1)
    return {
        "max_u": float(np.max(umag)),
        "mean_u": float(np.mean(umag)),
        "p95_u": float(np.percentile(umag, 95)),
    }


def compute_div_stats(u_func, domain, divu_l2, gradu_l2):
    """max|div u| pointwise + ratio L2(div)/L2(grad)."""
    # Project div(u) to P1 for pointwise max
    V_scalar = fem.functionspace(domain, ("Lagrange", 1))
    divu_expr = fem.Expression(div(u_func), V_scalar.element.interpolation_points())
    divu_func = fem.Function(V_scalar)
    divu_func.interpolate(divu_expr)
    max_divu = float(np.max(np.abs(divu_func.x.array)))
    ratio = divu_l2 / (gradu_l2 + 1e-30)
    return max_divu, ratio


# ════════════════════════════════════════════════════════════════════
# Symmetry metric (x-mirror about L/2)
# ════════════════════════════════════════════════════════════════════
def compute_symmetry_metric(u_func, cfg, N=201):
    """Evaluate |u| on mid-plane grid, return mean|u_left - u_right_flip|/mean|u|."""
    from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells

    domain = u_func.function_space.mesh
    z_mid = cfg.H / 2
    xs = np.linspace(0, cfg.L, N)
    ys = np.linspace(0, cfg.L, N)
    X, Y = np.meshgrid(xs, ys)
    pts = np.column_stack([X.ravel(), Y.ravel(), np.full(N*N, z_mid)])

    tree = bb_tree(domain, domain.topology.dim)
    cands = compute_collisions_points(tree, pts)
    cells = compute_colliding_cells(domain, cands, pts)

    umag = np.full(N*N, np.nan)
    for i in range(len(pts)):
        links = cells.links(i)
        if len(links) > 0:
            val = u_func.eval(pts[i], links[0])[:3]
            umag[i] = np.linalg.norm(val)

    umag_2d = umag.reshape(N, N)
    left = umag_2d[:, :N//2]
    right = umag_2d[:, N//2+1:][:, ::-1]
    sz = min(left.shape[1], right.shape[1])
    diff = float(np.nanmean(np.abs(left[:, :sz] - right[:, :sz])))
    ref = float(np.nanmean(umag_2d)) + 1e-30
    return diff / ref


# ════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════
def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    modes = ["standing", "vortex", "combined"]

    # ── Part 1: standard mesh (20×20×8) ──
    log(f"\n{'='*65}")
    log(f"  STREAMING FEM NORMS & DIAGNOSTICS — {TODAY}")
    log(f"{'='*65}\n")

    cfg = make_cfg(epw=6, min_z=8)
    log(f"[1/4] Creating mesh ({cfg.mesh_nx}×{cfg.mesh_nx}×{cfg.mesh_nz})...")
    domain, facet_tags, _ = create_mesh(cfg, verbose=False)

    log("[2/4] Solving Helmholtz + streaming (3 modes)...")
    summary_rows = []
    sym_coarse = None

    for mode in modes:
        log(f"\n  --- {mode} ---")
        p_sol = solve_helmholtz(domain, facet_tags, cfg, mode=mode, verbose=False)
        s_sol = solve_streaming_stokes(p_sol, domain=domain, verbose=False)
        if s_sol is None:
            log(f"  *** SOLVER RETURNED None for {mode} ***")
            continue

        u = s_sol.u_function

        # FEM L2 norms
        u_l2, gradu_l2, divu_l2 = compute_fem_norms(u)

        # Boundary flux
        flux = compute_boundary_flux(u, u.function_space.mesh)

        # Velocity stats
        vstats = compute_velocity_stats(u)

        # Div stats
        max_divu, ratio_div_grad = compute_div_stats(
            u, u.function_space.mesh, divu_l2, gradu_l2)

        log(f"  ‖u‖_L2          = {u_l2:.6e}")
        log(f"  ‖∇u‖_L2         = {gradu_l2:.6e}")
        log(f"  ‖∇·u‖_L2        = {divu_l2:.6e}")
        log(f"  net flux ∫u·n dS = {flux:.6e}")
        log(f"  max|u|           = {vstats['max_u']:.6e} m/s")
        log(f"  mean|u|          = {vstats['mean_u']:.6e} m/s")
        log(f"  95th %ile |u|    = {vstats['p95_u']:.6e} m/s")
        log(f"  max|div u|       = {max_divu:.6e} 1/s")
        log(f"  L2(div u)        = {divu_l2:.6e}")
        log(f"  L2(div)/L2(grad) = {ratio_div_grad:.6e}")

        row = {
            "mode": mode,
            "mesh": f"{cfg.mesh_nx}x{cfg.mesh_nx}x{cfg.mesh_nz}",
            "max_u_m_s": f"{vstats['max_u']:.6e}",
            "mean_u_m_s": f"{vstats['mean_u']:.6e}",
            "p95_u_m_s": f"{vstats['p95_u']:.6e}",
            "u_L2": f"{u_l2:.6e}",
            "gradu_L2": f"{gradu_l2:.6e}",
            "divu_L2": f"{divu_l2:.6e}",
            "max_divu": f"{max_divu:.6e}",
            "ratio_divu_gradu": f"{ratio_div_grad:.6e}",
            "net_flux": f"{flux:.6e}",
        }
        summary_rows.append(row)

        if mode == "standing":
            sym_coarse = compute_symmetry_metric(u, cfg)
            log(f"  symmetry metric  = {sym_coarse:.4f}")
            row["symmetry_metric"] = f"{sym_coarse:.4f}"

    # ── Part 2: refined mesh standing-only (30×30×12) ──
    log(f"\n{'='*65}")
    log(f"[3/4] Refined mesh standing-only (23×23×8)...")
    log(f"{'='*65}\n")

    cfg_fine = make_cfg(epw=7, min_z=8)
    log(f"  Mesh: {cfg_fine.mesh_nx}×{cfg_fine.mesh_nx}×{cfg_fine.mesh_nz}")
    domain_f, ftags_f, _ = create_mesh(cfg_fine, verbose=False)

    p_sol_f = solve_helmholtz(domain_f, ftags_f, cfg_fine, mode="standing", verbose=False)
    s_sol_f = solve_streaming_stokes(p_sol_f, domain=domain_f, verbose=False)

    if s_sol_f is not None:
        u_f = s_sol_f.u_function
        u_l2_f, gradu_l2_f, divu_l2_f = compute_fem_norms(u_f)
        flux_f = compute_boundary_flux(u_f, u_f.function_space.mesh)
        vstats_f = compute_velocity_stats(u_f)
        max_divu_f, ratio_f = compute_div_stats(
            u_f, u_f.function_space.mesh, divu_l2_f, gradu_l2_f)
        sym_fine = compute_symmetry_metric(u_f, cfg_fine)

        log(f"  ‖u‖_L2          = {u_l2_f:.6e}")
        log(f"  ‖∇u‖_L2         = {gradu_l2_f:.6e}")
        log(f"  ‖∇·u‖_L2        = {divu_l2_f:.6e}")
        log(f"  net flux ∫u·n dS = {flux_f:.6e}")
        log(f"  max|u|           = {vstats_f['max_u']:.6e} m/s")
        log(f"  mean|u|          = {vstats_f['mean_u']:.6e} m/s")
        log(f"  95th %ile |u|    = {vstats_f['p95_u']:.6e} m/s")
        log(f"  max|div u|       = {max_divu_f:.6e} 1/s")
        log(f"  L2(div)/L2(grad) = {ratio_f:.6e}")
        log(f"  symmetry metric  = {sym_fine:.4f}")

        row_f = {
            "mode": "standing",
            "mesh": f"{cfg_fine.mesh_nx}x{cfg_fine.mesh_nx}x{cfg_fine.mesh_nz}",
            "max_u_m_s": f"{vstats_f['max_u']:.6e}",
            "mean_u_m_s": f"{vstats_f['mean_u']:.6e}",
            "p95_u_m_s": f"{vstats_f['p95_u']:.6e}",
            "u_L2": f"{u_l2_f:.6e}",
            "gradu_L2": f"{gradu_l2_f:.6e}",
            "divu_L2": f"{divu_l2_f:.6e}",
            "max_divu": f"{max_divu_f:.6e}",
            "ratio_divu_gradu": f"{ratio_f:.6e}",
            "net_flux": f"{flux_f:.6e}",
            "symmetry_metric": f"{sym_fine:.4f}",
        }
        summary_rows.append(row_f)

        # ── Symmetry comparison ──
        log(f"\n{'='*65}")
        log(f"  SYMMETRY CONVERGENCE (standing x-mirror)")
        log(f"{'='*65}")
        log(f"  Coarse 20×20×8 :  symmetry = {sym_coarse:.4f}")
        log(f"  Fine   {cfg_fine.mesh_nx}×{cfg_fine.mesh_nx}×{cfg_fine.mesh_nz}:  symmetry = {sym_fine:.4f}")
        if sym_coarse is not None and sym_coarse > 0:
            drop = (sym_coarse - sym_fine) / sym_coarse * 100
            log(f"  Drop           :  {drop:+.1f}%  ({'improved' if drop > 0 else 'worsened'})")
        log(f"{'='*65}")
    else:
        log("  *** Refined solver returned None ***")

    # ── Part 3: Write CSV ──
    log(f"\n[4/4] Writing CSV summary...")
    csv_path = OUTDIR / "streaming_norms_summary.csv"
    if summary_rows:
        keys = list(summary_rows[-1].keys())  # use widest row
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(summary_rows)
    log(f"  {csv_path}")

    total = time.time() - t0
    log(f"\n  Done in {total:.1f} s.  Output: {OUTDIR.resolve()}\n")


if __name__ == "__main__":
    main()
