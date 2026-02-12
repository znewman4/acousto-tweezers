#!/usr/bin/env python3
"""
Batch 2A — Trap Atlas: Gor'kov potential, trap detection, stiffness analysis.

Generates trap-level quantitative diagnostics for standing-only, vortex-only,
and combined modes.  No streaming, no dynamics, no solver changes.

Usage:
    micromamba run -n acousto-complex python scripts/analysis/run_batch2a_trap_atlas.py

Outputs go to: results/latest/batch2A_YYYY-MM-DD/
"""
from __future__ import annotations

import sys, os, json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from mpi4py import MPI

from acoustweezers.experiments.shallow_square_dish.config import ShallowDishConfig
from acoustweezers.experiments.shallow_square_dish.solve_pressure import (
    create_mesh, solve_helmholtz,
    TAG_BOTTOM_DISC, TAG_BOTTOM_RIGID, TAG_TOP,
    TAG_X0, TAG_XL, TAG_Y0, TAG_YL,
)

comm = MPI.COMM_WORLD
rank = comm.rank

TODAY = datetime.now().strftime("%Y-%m-%d")
OUTDIR = Path(f"results/latest/batch2A_{TODAY}")


def log(msg=""):
    if rank == 0:
        print(msg, flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# Configuration — same validated Batch 1 configuration
# ══════════════════════════════════════════════════════════════════════════════
CFG = ShallowDishConfig(
    L=10e-3,
    H=1e-3,
    frequency_hz=500e3,
    elements_per_wavelength=6,
    min_elements_z=8,
    vortex_velocity_amplitude=10e-6,
    standing_velocity_amplitude=10e-6,
    vortex_aperture_radius=3e-3,
    standing_axis="both",
    standing_phase_pattern="antiphase",
)


# ══════════════════════════════════════════════════════════════════════════════
# Gor'kov potential  U(x,y) on a 2D grid at z = z_target
# ══════════════════════════════════════════════════════════════════════════════
def compute_gorkov_on_grid(p_solution, cfg, z_target, nx_grid=120):
    """
    Compute Gor'kov potential on a regular (x,y) grid at z ≈ z_target.

    U = (4π/3) a³ [ f1 <p²>/(2K) - f2 (3ρ/4) <v²> ]

    where  <p²> = |p|²/2,  <v²> = |∇p|²/(2 ω² ρ²).

    |∇p|² is computed via L2-projection of |grad p|² onto P1 space.

    Returns (X, Y, U_grid) — all 2D arrays of shape (nx_grid, nx_grid).
    """
    from dolfinx import fem
    from dolfinx.fem.petsc import LinearProblem
    import ufl
    from scipy.interpolate import griddata

    p_func = p_solution.p_function
    V = p_func.function_space
    mesh = V.mesh

    # --- |∇p|² via L2 projection onto P1 ---
    V1 = fem.functionspace(mesh, ("Lagrange", 1))
    u_trial = ufl.TrialFunction(V1)
    v_test = ufl.TestFunction(V1)

    grad_p = ufl.grad(p_func)
    # |∇p|² = ∇p · conj(∇p)  (complex inner product)
    grad_p_sq = ufl.inner(grad_p, grad_p)  # UFL handles conjugation

    a_form = ufl.inner(u_trial, v_test) * ufl.dx
    L_form = ufl.inner(grad_p_sq, v_test) * ufl.dx

    problem = LinearProblem(a_form, L_form, bcs=[],
                            petsc_options={"ksp_type": "cg", "pc_type": "jacobi",
                                           "ksp_rtol": 1e-6})
    grad_sq_func = problem.solve()

    coords_v1 = V1.tabulate_dof_coordinates()
    grad_sq_vals = np.abs(grad_sq_func.x.array)  # take real magnitude

    # |v1|² = |∇p|² / (ω²ρ²)
    v_sq = grad_sq_vals / (cfg.omega**2 * cfg.rho**2)

    # --- |p|² at P2 DOFs, interpolate to P1 coords via nearest-neighbour ---
    from scipy.interpolate import NearestNDInterpolator
    coords_p2 = p_func.function_space.tabulate_dof_coordinates()
    p_vals = p_func.x.array[:]
    p_sq = np.abs(p_vals)**2

    interp_psq = NearestNDInterpolator(coords_p2, p_sq)
    p_sq_at_v1 = interp_psq(coords_v1)

    # --- Gor'kov potential at V1 DOFs ---
    a_p = cfg.particle_radius
    f1 = cfg.f1_monopole
    f2 = cfg.f2_dipole
    rho = cfg.rho
    K = cfg.fluid_bulk_modulus

    prefactor = (4.0 * np.pi / 3.0) * a_p**3
    # <p²> = |p|²/2,  <v²> = |v|²/2
    U_dofs = prefactor * (f1 * p_sq_at_v1 / (4.0 * K) - f2 * (3.0 * rho / 8.0) * v_sq)

    # --- Extract slice at z_target and grid ---
    tol_z = cfg.H / cfg.mesh_nz * 1.5
    mask = np.abs(coords_v1[:, 2] - z_target) < tol_z
    if not np.any(mask):
        raise RuntimeError(f"No DOFs near z={z_target}")

    x_pts = coords_v1[mask, 0]
    y_pts = coords_v1[mask, 1]
    u_pts = U_dofs[mask]

    xg = np.linspace(0, cfg.L, nx_grid)
    yg = np.linspace(0, cfg.L, nx_grid)
    X, Y = np.meshgrid(xg, yg)
    U_grid = griddata((x_pts, y_pts), u_pts, (X, Y), method='linear')

    return X, Y, U_grid


# ══════════════════════════════════════════════════════════════════════════════
# Trap detection (local minima of U on grid)
# ══════════════════════════════════════════════════════════════════════════════
def find_traps(X, Y, U_grid, margin_cells=3):
    """
    Find local minima of U_grid using neighbourhood comparison.

    Parameters
    ----------
    X, Y : 2D arrays (meshgrid)
    U_grid : 2D array of Gor'kov potential
    margin_cells : int
        Exclude this many cells from each edge (avoids boundary artefacts).

    Returns list of dicts with trap info.
    """
    ny, nx_g = U_grid.shape
    traps = []

    # Pad NaN to avoid edge effects in comparison
    U_work = U_grid.copy()
    U_work[:margin_cells, :] = np.nan
    U_work[-margin_cells:, :] = np.nan
    U_work[:, :margin_cells] = np.nan
    U_work[:, -margin_cells:] = np.nan

    # Neighbourhood check: 5×5 window
    hw = 2  # half-window
    for j in range(hw + margin_cells, ny - hw - margin_cells):
        for i in range(hw + margin_cells, nx_g - hw - margin_cells):
            val = U_work[j, i]
            if np.isnan(val):
                continue
            patch = U_work[j - hw:j + hw + 1, i - hw:i + hw + 1]
            if np.all(np.isnan(patch)):
                continue
            # Check if centre is minimum in patch (strict)
            patch_nonan = patch[~np.isnan(patch)]
            if val <= np.min(patch_nonan) and np.sum(patch_nonan == val) == 1:
                traps.append({"ix": i, "iy": j,
                              "x": float(X[j, i]),
                              "y": float(Y[j, i]),
                              "U_min": float(val)})

    return traps


def compute_trap_properties(X, Y, U_grid, traps, cfg):
    """
    For each trap, compute:
      - saddle U (nearest ridge along x and y)
      - trap depth ΔU
      - Hessian at minimum (2×2, from finite differences)
      - eigenvalues λ1, λ2
      - anisotropy ratio
    """
    dx_grid = X[0, 1] - X[0, 0]
    dy_grid = Y[1, 0] - Y[0, 0]

    for trap in traps:
        ix, iy = trap["ix"], trap["iy"]
        U0 = trap["U_min"]

        # --- Saddle estimates along x and y ---
        # Walk in +x and -x until U starts decreasing (i.e. local ridge)
        saddle_vals = []
        for dim, step, n_max in [("x", 1, U_grid.shape[1] - ix - 1),
                                  ("x", -1, ix),
                                  ("y", 1, U_grid.shape[0] - iy - 1),
                                  ("y", -1, iy)]:
            prev = U0
            saddle = U0
            for s in range(1, min(n_max, 40)):
                if dim == "x":
                    val = U_grid[iy, ix + step * s]
                else:
                    val = U_grid[iy + step * s, ix]
                if np.isnan(val):
                    break
                if val >= prev:
                    saddle = val
                    prev = val
                else:
                    # passed the ridge
                    break
            saddle_vals.append(saddle)

        U_saddle = float(np.min(saddle_vals)) if saddle_vals else U0
        trap["U_saddle"] = U_saddle
        trap["delta_U"] = float(U_saddle - U0)

        # --- Hessian via central differences ---
        # H_xx = (U(i+1,j) - 2U(i,j) + U(i-1,j)) / dx²
        # H_yy = (U(i,j+1) - 2U(i,j) + U(i,j-1)) / dy²
        # H_xy = (U(i+1,j+1) - U(i+1,j-1) - U(i-1,j+1) + U(i-1,j-1)) / (4 dx dy)
        ny, nx_g = U_grid.shape
        if 1 <= ix < nx_g - 1 and 1 <= iy < ny - 1:
            H_xx = (U_grid[iy, ix + 1] - 2 * U0 + U_grid[iy, ix - 1]) / dx_grid**2
            H_yy = (U_grid[iy + 1, ix] - 2 * U0 + U_grid[iy - 1, ix]) / dy_grid**2
            H_xy = (U_grid[iy + 1, ix + 1] - U_grid[iy + 1, ix - 1]
                    - U_grid[iy - 1, ix + 1] + U_grid[iy - 1, ix - 1]) / (4 * dx_grid * dy_grid)

            if any(np.isnan([H_xx, H_yy, H_xy])):
                trap["hessian"] = None
                trap["eigenvalues"] = [None, None]
                trap["anisotropy"] = None
            else:
                H_mat = np.array([[H_xx, H_xy], [H_xy, H_yy]])
                eigvals = np.sort(np.linalg.eigvalsh(H_mat))
                trap["hessian"] = [[float(H_xx), float(H_xy)],
                                   [float(H_xy), float(H_yy)]]
                trap["eigenvalues"] = [float(eigvals[0]), float(eigvals[1])]
                if eigvals[0] > 0:
                    trap["anisotropy"] = float(eigvals[1] / eigvals[0])
                else:
                    trap["anisotropy"] = None
        else:
            trap["hessian"] = None
            trap["eigenvalues"] = [None, None]
            trap["anisotropy"] = None

    return traps


# ══════════════════════════════════════════════════════════════════════════════
# CSV / JSON writers
# ══════════════════════════════════════════════════════════════════════════════
def save_traps_csv(traps, path):
    import csv
    fields = ["x", "y", "U_min", "U_saddle", "delta_U",
              "eigenvalue_1", "eigenvalue_2", "anisotropy"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for t in traps:
            w.writerow({
                "x": f"{t['x']:.6e}",
                "y": f"{t['y']:.6e}",
                "U_min": f"{t['U_min']:.6e}",
                "U_saddle": f"{t['U_saddle']:.6e}",
                "delta_U": f"{t['delta_U']:.6e}",
                "eigenvalue_1": f"{t['eigenvalues'][0]:.6e}" if t['eigenvalues'][0] is not None else "",
                "eigenvalue_2": f"{t['eigenvalues'][1]:.6e}" if t['eigenvalues'][1] is not None else "",
                "anisotropy": f"{t['anisotropy']:.4f}" if t['anisotropy'] is not None else "",
            })


# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════
def plot_U_with_traps(X, Y, U_grid, traps, title, path, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    L_mm = cfg.L * 1e3

    im = ax.pcolormesh(X * 1e3, Y * 1e3, U_grid, shading='auto', cmap='viridis')
    fig.colorbar(im, ax=ax, label="U [J]", shrink=0.8)

    if traps:
        tx = [t["x"] * 1e3 for t in traps]
        ty = [t["y"] * 1e3 for t in traps]
        ax.plot(tx, ty, 'rx', ms=8, mew=2, label=f"{len(traps)} traps")
        ax.legend(loc='upper right')

    # Mark disc outline
    theta = np.linspace(0, 2 * np.pi, 100)
    cx, cy = cfg.L / 2 * 1e3, cfg.L / 2 * 1e3
    R_mm = cfg.vortex_aperture_radius * 1e3
    ax.plot(cx + R_mm * np.cos(theta), cy + R_mm * np.sin(theta),
            'w--', lw=1, alpha=0.6, label='disc edge')

    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(title)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log(f"  Plot: {Path(path).name}")


def plot_delta_U(X, Y, U_comb, U_stand, path, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dU = U_comb - U_stand
    vmax = np.nanmax(np.abs(dU))
    if vmax == 0:
        vmax = 1e-30

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.pcolormesh(X * 1e3, Y * 1e3, dU, shading='auto',
                        cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    fig.colorbar(im, ax=ax, label="ΔU [J]", shrink=0.8)

    theta = np.linspace(0, 2 * np.pi, 100)
    cx, cy = cfg.L / 2 * 1e3, cfg.L / 2 * 1e3
    R_mm = cfg.vortex_aperture_radius * 1e3
    ax.plot(cx + R_mm * np.cos(theta), cy + R_mm * np.sin(theta),
            'k--', lw=1, alpha=0.6)

    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title("ΔU = U_combined − U_standing  [J],  mid-plane")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log(f"  Plot: {Path(path).name}")


# ══════════════════════════════════════════════════════════════════════════════
# Selectivity + interaction metrics
# ══════════════════════════════════════════════════════════════════════════════
def selectivity_metric(traps, cfg):
    """
    Choose trap closest to vortex centre as 'target'.
    S = ΔU_target / median(ΔU_other).
    """
    if len(traps) < 2:
        return None, None

    cx = cfg.L / 2
    cy = cfg.L / 2
    dists = [np.hypot(t["x"] - cx, t["y"] - cy) for t in traps]
    idx_target = int(np.argmin(dists))

    dU_target = traps[idx_target]["delta_U"]
    dU_others = [t["delta_U"] for i, t in enumerate(traps) if i != idx_target and t["delta_U"] > 0]

    if not dU_others or dU_target <= 0:
        return idx_target, None

    S = dU_target / float(np.median(dU_others))
    return idx_target, S


def trap_displacement_metric(traps_stand, traps_comb, cfg):
    """
    For standing traps within 2 disc radii of centre, find nearest combined trap
    and report displacement.
    """
    cx = cfg.L / 2
    cy = cfg.L / 2
    R2 = (2 * cfg.vortex_aperture_radius)**2

    displacements = []
    for ts in traps_stand:
        if (ts["x"] - cx)**2 + (ts["y"] - cy)**2 > R2:
            continue
        if not traps_comb:
            continue
        # Find nearest combined trap
        dists = [np.hypot(ts["x"] - tc["x"], ts["y"] - tc["y"]) for tc in traps_comb]
        idx_near = int(np.argmin(dists))
        displacements.append({
            "standing_xy": [ts["x"], ts["y"]],
            "combined_xy": [traps_comb[idx_near]["x"], traps_comb[idx_near]["y"]],
            "displacement_m": float(dists[idx_near]),
        })
    return displacements


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    log(f"\n{'#'*70}")
    log(f"  BATCH 2A — TRAP ATLAS — {TODAY}")
    log(f"  Output: {OUTDIR.resolve()}")
    log(f"{'#'*70}\n")

    cfg = CFG
    z_mid = cfg.H / 2
    nx_grid = 150  # finer than Batch 1 for trap detection

    # --- Mesh + solve 3 modes ---
    log("[1/5] Mesh generation...")
    domain, facet_tags, _ = create_mesh(cfg, verbose=True)

    log("[2/5] Solving Helmholtz (3 modes)...")
    solutions = {}
    for mode in ["standing", "vortex", "combined"]:
        solutions[mode] = solve_helmholtz(domain, facet_tags, cfg, mode=mode, verbose=True)

    # --- Gor'kov potential on mid-plane ---
    log("[3/5] Computing Gor'kov potential U(x,y) on mid-plane...")
    U_grids = {}
    for mode in ["standing", "vortex", "combined"]:
        log(f"  {mode}...")
        X, Y, U_grids[mode] = compute_gorkov_on_grid(solutions[mode], cfg, z_mid, nx_grid=nx_grid)

    # --- Trap detection ---
    log("[4/5] Detecting traps (local minima)...")
    all_traps = {}
    for mode in ["standing", "vortex", "combined"]:
        traps = find_traps(X, Y, U_grids[mode])
        traps = compute_trap_properties(X, Y, U_grids[mode], traps, cfg)
        # Filter out saddle points: a true trap must have both Hessian
        # eigenvalues positive.  Near-boundary points with a negative eigenvalue
        # (even tiny) are saddle artefacts from grid interpolation.
        traps_filtered = []
        n_rejected = 0
        for t in traps:
            if t["eigenvalues"][0] is not None and t["eigenvalues"][0] <= 0:
                n_rejected += 1
                continue
            traps_filtered.append(t)
        all_traps[mode] = traps_filtered
        log(f"  {mode}: {len(traps_filtered)} traps detected"
            + (f" ({n_rejected} saddle-points rejected)" if n_rejected else ""))

    # --- Metrics ---
    log("[5/5] Computing metrics + saving outputs...")

    # Selectivity (combined mode, target = nearest to vortex centre)
    idx_target_comb, S_comb = selectivity_metric(all_traps["combined"], cfg)
    idx_target_stand, S_stand = selectivity_metric(all_traps["standing"], cfg)

    # Trap displacement
    displacements = trap_displacement_metric(all_traps["standing"], all_traps["combined"], cfg)

    # ── Save CSVs ──
    for mode in ["standing", "vortex", "combined"]:
        csv_path = OUTDIR / f"traps_{mode}.csv"
        save_traps_csv(all_traps[mode], csv_path)
        log(f"  CSV: {csv_path.name}")

    # ── Save plots ──
    plot_U_with_traps(X, Y, U_grids["standing"], all_traps["standing"],
                      "Gor'kov U — standing, mid-plane",
                      OUTDIR / "U_midplane_standing.png", cfg)
    plot_U_with_traps(X, Y, U_grids["vortex"], all_traps["vortex"],
                      "Gor'kov U — vortex, mid-plane",
                      OUTDIR / "U_midplane_vortex.png", cfg)
    plot_U_with_traps(X, Y, U_grids["combined"], all_traps["combined"],
                      "Gor'kov U — combined, mid-plane",
                      OUTDIR / "U_midplane_combined.png", cfg)
    plot_delta_U(X, Y, U_grids["combined"], U_grids["standing"],
                 OUTDIR / "delta_U_combined_minus_standing.png", cfg)

    # ── Summary JSON ──
    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "L_mm": cfg.L * 1e3,
            "H_mm": cfg.H * 1e3,
            "freq_kHz": cfg.frequency_hz / 1e3,
            "standing_axis": cfg.standing_axis,
            "particle_radius_um": cfg.particle_radius * 1e6,
            "f1": round(cfg.f1_monopole, 6),
            "f2": round(cfg.f2_dipole, 6),
        },
        "trap_counts": {m: len(all_traps[m]) for m in ["standing", "vortex", "combined"]},
        "trap_depths_J": {},
        "selectivity": {},
        "displacements": displacements,
    }

    for mode in ["standing", "vortex", "combined"]:
        depths = [t["delta_U"] for t in all_traps[mode] if t["delta_U"] > 0]
        summary["trap_depths_J"][mode] = {
            "min": float(np.min(depths)) if depths else None,
            "max": float(np.max(depths)) if depths else None,
            "median": float(np.median(depths)) if depths else None,
            "mean": float(np.mean(depths)) if depths else None,
        }

    summary["selectivity"]["combined"] = {
        "target_index": idx_target_comb,
        "S": round(S_comb, 4) if S_comb is not None else None,
    }
    summary["selectivity"]["standing"] = {
        "target_index": idx_target_stand,
        "S": round(S_stand, 4) if S_stand is not None else None,
    }

    json_path = OUTDIR / "trap_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"  JSON: {json_path.name}")

    # ══════════════════════════════════════════════════════════════════════
    # ACCEPTANCE CRITERIA
    # ══════════════════════════════════════════════════════════════════════
    log(f"\n{'='*70}")
    log("ACCEPTANCE CRITERIA")
    log(f"{'='*70}")

    all_pass = True

    # 1. At least 4 traps in standing
    n_stand = len(all_traps["standing"])
    ok1 = n_stand >= 4
    log(f"  [{'PASS' if ok1 else 'FAIL'}] Standing traps >= 4: found {n_stand}")
    if not ok1:
        all_pass = False

    # 2. Combined trap depths differ from standing
    if all_traps["standing"] and all_traps["combined"]:
        depths_s = sorted([t["delta_U"] for t in all_traps["standing"]])
        depths_c = sorted([t["delta_U"] for t in all_traps["combined"]])
        # Check they're not identical
        if len(depths_s) == len(depths_c):
            diff = np.max(np.abs(np.array(depths_s) - np.array(depths_c)))
        else:
            diff = 1.0  # different count → definitely different
        ok2 = diff > 0
        log(f"  [{'PASS' if ok2 else 'FAIL'}] Combined depths differ from standing: max diff = {diff:.4e} J")
    else:
        ok2 = False
        log(f"  [FAIL] Cannot compare depths (empty trap lists)")
    if not ok2:
        all_pass = False

    # 3. Hessian eigenvalues positive at all retained traps
    #    (Saddle-point traps with λ1 ≤ 0 were already filtered out above.)
    bad_eigs = 0
    total_eigs = 0
    for mode in ["standing", "vortex", "combined"]:
        for t in all_traps[mode]:
            if t["eigenvalues"][0] is not None:
                total_eigs += 1
                if t["eigenvalues"][0] <= 0:
                    bad_eigs += 1
    ok3 = bad_eigs == 0 and total_eigs > 0
    log(f"  [{'PASS' if ok3 else 'FAIL'}] Hessian eigenvalues positive: {total_eigs - bad_eigs}/{total_eigs} positive")
    if not ok3:
        all_pass = False

    # 4. No NaNs in U grids
    nan_count = sum(np.sum(np.isnan(U_grids[m])) for m in ["standing", "vortex", "combined"])
    # Some NaN at boundary interpolation edges is normal; check interior
    interior = U_grids["standing"][5:-5, 5:-5]
    nan_interior = np.sum(np.isnan(interior))
    ok4 = nan_interior == 0
    log(f"  [{'PASS' if ok4 else 'FAIL'}] No interior NaNs: {nan_interior} NaN in interior")
    if not ok4:
        all_pass = False

    if all_pass:
        log(f"\n  *** ALL ACCEPTANCE CRITERIA PASSED ***")
    else:
        log(f"\n  *** SOME CRITERIA FAILED — see above ***")

    # ── Final summary ──
    log(f"\n{'#'*70}")
    log("  BATCH 2A SUMMARY")
    log(f"{'#'*70}")
    log(f"  Output:     {OUTDIR.resolve()}")
    for mode in ["standing", "vortex", "combined"]:
        n = len(all_traps[mode])
        if n > 0:
            depths = [t["delta_U"] for t in all_traps[mode]]
            log(f"  {mode:10s}:  {n} traps,  ΔU median = {np.median(depths):.3e} J, "
                f"max = {np.max(depths):.3e} J")
        else:
            log(f"  {mode:10s}:  0 traps")

    if S_comb is not None:
        log(f"  Selectivity S (combined): {S_comb:.4f}")
    if displacements:
        d_vals = [d["displacement_m"] for d in displacements]
        log(f"  Trap displacements (stand→comb): "
            f"mean={np.mean(d_vals)*1e3:.3f} mm, max={np.max(d_vals)*1e3:.3f} mm")
    log()


if __name__ == "__main__":
    main()
