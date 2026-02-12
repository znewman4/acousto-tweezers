#!/usr/bin/env python3
"""
Streaming Validation Dump — dense plane-sampled CSV + visual diagnostics.

Solves standing/vortex/combined, computes streaming, exports every field
on uniform grids, generates diagnostic plots and a sanity report.

Usage:
    micromamba run -n acousto-complex python scripts/analysis/run_streaming_validation_dump.py

Outputs: results/latest/streaming_validation_YYYY-MM-DD/
"""
from __future__ import annotations
import sys, os, json, csv, time, textwrap
from pathlib import Path
from datetime import datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from mpi4py import MPI

from acoustweezers.experiments.shallow_square_dish.config import ShallowDishConfig
from acoustweezers.experiments.shallow_square_dish.solve_pressure import (
    create_mesh, solve_helmholtz,
)
from acoustweezers.experiments.shallow_square_dish.streaming import (
    solve_streaming_stokes, compute_first_order_velocity,
)

comm = MPI.COMM_WORLD
rank = comm.rank
TODAY = datetime.now().strftime("%Y-%m-%d")
OUTDIR = Path(f"results/latest/streaming_validation_{TODAY}")

def log(msg=""):
    if rank == 0:
        print(msg, flush=True)

# ════════════════════════════════════════════════════════════════════
# Config — same as Batch1/2A
# ════════════════════════════════════════════════════════════════════
def make_cfg(amp_scale=1.0):
    return ShallowDishConfig(
        L=10e-3, H=1e-3,
        frequency_hz=500e3,
        elements_per_wavelength=6,
        min_elements_z=8,
        vortex_velocity_amplitude=10e-6 * amp_scale,
        standing_velocity_amplitude=10e-6 * amp_scale,
        vortex_aperture_radius=3e-3,
        standing_axis="both",
        standing_phase_pattern="antiphase",
    )

CFG = make_cfg(1.0)

# ════════════════════════════════════════════════════════════════════
# Helpers: sample FEM functions onto a uniform 2D grid
# ════════════════════════════════════════════════════════════════════
def _sample_scalar_at_points(func, pts_3d):
    """Evaluate a scalar fem.Function at an array of 3D points.
    Returns array of same length; NaN where outside mesh."""
    from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
    mesh_obj = func.function_space.mesh
    tree = bb_tree(mesh_obj, mesh_obj.topology.dim)
    cell_candidates = compute_collisions_points(tree, pts_3d)
    cells = compute_colliding_cells(mesh_obj, cell_candidates, pts_3d)
    vals = np.full(len(pts_3d), np.nan, dtype=func.x.array.dtype)
    for i in range(len(pts_3d)):
        links = cells.links(i)
        if len(links) > 0:
            vals[i] = func.eval(pts_3d[i], links[0])[0]
    return vals

def _sample_vector_at_points(func, pts_3d):
    """Evaluate a 3-component vector fem.Function. Returns (N,3) array."""
    from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
    mesh_obj = func.function_space.mesh
    tree = bb_tree(mesh_obj, mesh_obj.topology.dim)
    cell_candidates = compute_collisions_points(tree, pts_3d)
    cells = compute_colliding_cells(mesh_obj, cell_candidates, pts_3d)
    vals = np.full((len(pts_3d), 3), np.nan, dtype=func.x.array.dtype)
    for i in range(len(pts_3d)):
        links = cells.links(i)
        if len(links) > 0:
            vals[i] = func.eval(pts_3d[i], links[0])[:3]
    return vals

def make_grid(cfg, z_val, N=201):
    """Return (pts_3d shape (N*N,3), X shape (N,N), Y shape (N,N))."""
    xs = np.linspace(0, cfg.L, N)
    ys = np.linspace(0, cfg.L, N)
    X, Y = np.meshgrid(xs, ys)
    pts = np.column_stack([X.ravel(), Y.ravel(), np.full(N*N, z_val)])
    return pts, X, Y

# ════════════════════════════════════════════════════════════════════
# Project grad(p) and forcing to Functions for pointwise eval
# ════════════════════════════════════════════════════════════════════
def project_gradient(p_func, domain, cfg):
    """Return a fem.Function (P1-vec,3) containing grad(p) (complex)."""
    from dolfinx import fem
    import ufl
    V_vec = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    gp_expr = fem.Expression(ufl.grad(p_func),
                             V_vec.element.interpolation_points())
    gp = fem.Function(V_vec)
    gp.interpolate(gp_expr)
    return gp

def project_forcing(p_func, domain, cfg, forcing_scale=1.0):
    """Return f_func (P1-vec,3) = -div(Re(0.5*rho*v1 outer conj(v1))).
    This is the EXACT same expression used in solve_streaming_stokes."""
    from dolfinx import fem
    import ufl
    omega = cfg.omega
    rho = cfg.rho
    grad_p = ufl.grad(p_func)
    v1 = grad_p / (1j * omega * rho)
    stress = 0.5 * rho * ufl.outer(v1, ufl.conj(v1))
    stress_re = ufl.real(stress)
    f_ufl = -forcing_scale * ufl.div(stress_re)
    V_f = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    f_expr = fem.Expression(f_ufl, V_f.element.interpolation_points())
    f_func = fem.Function(V_f)
    f_func.interpolate(f_expr)
    f_func.x.array[:] = np.real(f_func.x.array)
    return f_func

# ════════════════════════════════════════════════════════════════════
# Build CSV for one (mode, plane)
# ════════════════════════════════════════════════════════════════════
def build_plane_csv(mode, plane_name, z_val,
                    cfg, p_func, gp_func, f_func, u_str_func,
                    N=201):
    """Sample all fields on uniform grid and return list-of-dicts rows."""
    pts, X, Y = make_grid(cfg, z_val, N)
    omega = cfg.omega; rho = cfg.rho; mu = cfg.mu; c = cfg.c

    # ── p (complex scalar in P2 space) ──
    p_vals = _sample_scalar_at_points(p_func, pts)
    p_re = np.real(p_vals);  p_im = np.imag(p_vals)
    p_abs = np.abs(p_vals);  p_phase = np.angle(p_vals)

    # ── grad(p) (complex vector in P1 space) ──
    gp = _sample_vector_at_points(gp_func, pts)
    gp_abs = np.sqrt(np.sum(np.abs(gp)**2, axis=1))

    # ── v1 = grad(p)/(i*omega*rho), complex ──
    v1 = gp / (1j * omega * rho)
    v1_abs = np.sqrt(np.sum(np.abs(v1)**2, axis=1))

    # ── forcing f_str (real vector, P1) ──
    f_v = _sample_vector_at_points(f_func, pts)
    f_abs = np.sqrt(np.sum(np.real(f_v)**2, axis=1))

    # ── streaming velocity u_str (real vector, P2 collapsed) ──
    u_v = _sample_vector_at_points(u_str_func, pts)
    u_abs = np.sqrt(np.sum(np.real(u_v)**2, axis=1))

    # ── div(u_str), curl(u_str) via finite diff on grid ──
    dx_g = cfg.L / (N - 1)
    def _reshape(arr):
        return arr.reshape(N, N)
    ux = _reshape(np.real(u_v[:, 0]))
    uy = _reshape(np.real(u_v[:, 1]))
    uz = _reshape(np.real(u_v[:, 2]))
    # div ≈ ∂ux/∂x + ∂uy/∂y  (no ∂uz/∂z on a single plane)
    dux_dx = np.gradient(ux, dx_g, axis=1)
    duy_dy = np.gradient(uy, dx_g, axis=0)
    div_u = dux_dx + duy_dy  # partial 2D approximation

    # curl components (only z-component is fully in-plane)
    duy_dx = np.gradient(uy, dx_g, axis=1)
    dux_dy = np.gradient(ux, dx_g, axis=0)
    duz_dx = np.gradient(uz, dx_g, axis=1)
    duz_dy = np.gradient(uz, dx_g, axis=0)
    curl_x = duz_dy           # ∂uz/∂y (partial)
    curl_y = -duz_dx          # -∂uz/∂x (partial)
    curl_z = duy_dx - dux_dy  # ∂uy/∂x - ∂ux/∂y (full in-plane)
    curl_abs = np.sqrt(curl_x**2 + curl_y**2 + curl_z**2)

    # ── derived quantities ──
    p2 = p_abs**2           # |p|²
    floor = 1e-30
    u_ratio = u_abs / np.maximum(v1_abs, floor)
    Re_local = (rho * u_abs * cfg.L) / (mu + floor)
    # mask_near_boundary: within 1 grid cell of edge
    eps_bnd = 1.5 * dx_g
    x_flat = pts[:, 0]; y_flat = pts[:, 1]
    near = ((x_flat < eps_bnd) | (x_flat > cfg.L - eps_bnd) |
            (y_flat < eps_bnd) | (y_flat > cfg.L - eps_bnd))
    mask_bnd = near.astype(int)

    # ── flatten grid quantities ──
    div_u_flat = div_u.ravel()
    curl_x_flat = curl_x.ravel(); curl_y_flat = curl_y.ravel()
    curl_z_flat = curl_z.ravel(); curl_abs_flat = curl_abs.ravel()

    rows = []
    for k in range(N * N):
        rows.append({
            "mode": mode, "plane": plane_name,
            "x_m": f"{pts[k,0]:.8e}", "y_m": f"{pts[k,1]:.8e}", "z_m": f"{pts[k,2]:.8e}",
            "freq_hz": f"{cfg.frequency_hz:.1f}",
            "omega": f"{omega:.4f}", "rho": f"{rho:.1f}",
            "mu": f"{mu:.6e}", "c": f"{c:.1f}",
            "p_re": f"{p_re[k]:.8e}", "p_im": f"{p_im[k]:.8e}",
            "p_abs": f"{p_abs[k]:.8e}", "p_phase": f"{p_phase[k]:.8e}",
            "dpdx_re": f"{np.real(gp[k,0]):.8e}", "dpdx_im": f"{np.imag(gp[k,0]):.8e}",
            "dpdy_re": f"{np.real(gp[k,1]):.8e}", "dpdy_im": f"{np.imag(gp[k,1]):.8e}",
            "dpdz_re": f"{np.real(gp[k,2]):.8e}", "dpdz_im": f"{np.imag(gp[k,2]):.8e}",
            "gradp_abs": f"{gp_abs[k]:.8e}",
            "v1x_re": f"{np.real(v1[k,0]):.8e}", "v1x_im": f"{np.imag(v1[k,0]):.8e}",
            "v1y_re": f"{np.real(v1[k,1]):.8e}", "v1y_im": f"{np.imag(v1[k,1]):.8e}",
            "v1z_re": f"{np.real(v1[k,2]):.8e}", "v1z_im": f"{np.imag(v1[k,2]):.8e}",
            "v1_abs": f"{v1_abs[k]:.8e}",
            "u_str_x": f"{np.real(u_v[k,0]):.8e}", "u_str_y": f"{np.real(u_v[k,1]):.8e}",
            "u_str_z": f"{np.real(u_v[k,2]):.8e}", "u_str_abs": f"{u_abs[k]:.8e}",
            "p2": f"{p2[k]:.8e}",
            "div_u_str": f"{div_u_flat[k]:.8e}",
            "curl_u_str_x": f"{curl_x_flat[k]:.8e}", "curl_u_str_y": f"{curl_y_flat[k]:.8e}",
            "curl_u_str_z": f"{curl_z_flat[k]:.8e}", "curl_u_str_abs": f"{curl_abs_flat[k]:.8e}",
            "f_str_x": f"{np.real(f_v[k,0]):.8e}", "f_str_y": f"{np.real(f_v[k,1]):.8e}",
            "f_str_z": f"{np.real(f_v[k,2]):.8e}", "f_str_abs": f"{f_abs[k]:.8e}",
            "u_ratio_str_to_v1": f"{u_ratio[k]:.8e}",
            "Re_str_local": f"{Re_local[k]:.8e}",
            "mask_near_boundary": str(mask_bnd[k]),
        })
    return rows, div_u, curl_abs, ux, uy, uz, u_abs.reshape(N, N), f_abs.reshape(N, N)

# ════════════════════════════════════════════════════════════════════
# Plotting helpers
# ════════════════════════════════════════════════════════════════════
def _setup_mpl():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt

def heatmap(X, Y, Z, title, path, cmap='viridis', label="", symmetric=False):
    plt = _setup_mpl()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    if symmetric:
        vmax = np.nanmax(np.abs(Z)); vmin = -vmax
    else:
        vmin = None; vmax = None
    im = ax.pcolormesh(X*1e3, Y*1e3, Z, shading='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, label=label, shrink=0.8)
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    ax.set_title(title); ax.set_aspect("equal")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)

def quiver_overlay(X, Y, Umag, Ux, Uy, title, path, sparse=25):
    plt = _setup_mpl()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.pcolormesh(X*1e3, Y*1e3, Umag, shading='auto', cmap='viridis')
    fig.colorbar(im, ax=ax, label="|u_str| [m/s]", shrink=0.8)
    step = max(1, X.shape[0] // sparse)
    ax.quiver(X[::step,::step]*1e3, Y[::step,::step]*1e3,
              Ux[::step,::step], Uy[::step,::step],
              color='w', alpha=0.8, scale_units='xy')
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    ax.set_title(title); ax.set_aspect("equal")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)

# ════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════
def main():
    t0_all = time.time()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "csv").mkdir(exist_ok=True)
    (OUTDIR / "plots").mkdir(exist_ok=True)
    (OUTDIR / "diagnostics").mkdir(exist_ok=True)

    cfg = CFG
    N = 201  # grid resolution per axis

    log(f"\n{'#'*70}")
    log(f"  STREAMING VALIDATION DUMP — {TODAY}")
    log(f"  Output: {OUTDIR.resolve()}")
    log(f"{'#'*70}\n")

    # ── 1. Mesh ──
    log("[1/7] Mesh generation...")
    domain, facet_tags, _ = create_mesh(cfg, verbose=True)

    # ── 2. Solve Helmholtz for 3 modes ──
    log("[2/7] Solving Helmholtz (3 modes)...")
    p_solutions = {}
    for mode in ["standing", "vortex", "combined"]:
        p_solutions[mode] = solve_helmholtz(domain, facet_tags, cfg,
                                            mode=mode, verbose=True)

    # ── 3. Solve streaming for 3 modes ──
    log("[3/7] Solving streaming Stokes (3 modes)...")
    str_solutions = {}
    f_funcs = {}
    gp_funcs = {}
    for mode in ["standing", "vortex", "combined"]:
        log(f"\n  --- {mode} ---")
        sol = solve_streaming_stokes(p_solutions[mode], domain=domain,
                                     verbose=True)
        if sol is None:
            log(f"  *** RED FLAG: streaming solver returned None for {mode} ***")
            _write_red_flag(f"Streaming solver returned None for mode={mode}. "
                            "Likely KSP divergence. Check streaming.py solve_streaming_stokes.",
                            OUTDIR)
            return
        str_solutions[mode] = sol
        # project grad(p) and forcing for pointwise eval
        gp_funcs[mode] = project_gradient(p_solutions[mode].p_function, domain, cfg)
        f_funcs[mode] = project_forcing(p_solutions[mode].p_function, domain, cfg)

    # ── Quick NaN/Inf check ──
    for mode in ["standing", "vortex", "combined"]:
        u_arr = str_solutions[mode].u_function.x.array
        if np.any(np.isnan(u_arr)) or np.any(np.isinf(u_arr)):
            _write_red_flag(f"NaN/Inf in streaming velocity for mode={mode}!", OUTDIR)
            return

    # ── 4. Build CSVs and collect grid data for plots ──
    log("\n[4/7] Sampling fields on uniform grids + writing CSVs...")
    z_mid = cfg.H / 2
    z_bot = 0.1 * cfg.H
    planes = [("mid", z_mid), ("bottom", z_bot)]

    grid_data = {}   # (mode, plane) -> dict of 2D arrays
    csv_all_rows = {"mid": [], "bottom": []}

    pts_mid, X, Y = make_grid(cfg, z_mid, N)

    for mode in ["standing", "vortex", "combined"]:
        for pname, zv in planes:
            log(f"  {mode} / {pname}...")
            rows, div_u, curl_abs, ux, uy, uz, u_abs_2d, f_abs_2d = \
                build_plane_csv(mode, pname, zv, cfg,
                                p_solutions[mode].p_function,
                                gp_funcs[mode], f_funcs[mode],
                                str_solutions[mode].u_function, N=N)
            csv_all_rows[pname].extend(rows)
            grid_data[(mode, pname)] = {
                "div_u": div_u, "curl_abs": curl_abs,
                "ux": ux, "uy": uy, "uz": uz,
                "u_abs": u_abs_2d, "f_abs": f_abs_2d,
            }

    # Write CSVs
    for pname in ["mid", "bottom"]:
        csv_path = OUTDIR / "csv" / f"streaming_plane_{pname}.csv"
        if csv_all_rows[pname]:
            fieldnames = list(csv_all_rows[pname][0].keys())
            with open(csv_path, "w", newline="") as fout:
                w = csv.DictWriter(fout, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(csv_all_rows[pname])
        log(f"  CSV: {csv_path.name} ({len(csv_all_rows[pname])} rows)")

    # ── 5. Plots ──
    log("\n[5/7] Generating diagnostic plots...")
    pdir = OUTDIR / "plots"
    for mode in ["standing", "vortex", "combined"]:
        for pname, _ in planes:
            gd = grid_data[(mode, pname)]
            tag = f"{mode}_{pname}"
            heatmap(X, Y, gd["u_abs"], f"|u_str| — {mode} {pname}",
                    pdir / f"u_str_abs_{tag}.png", label="|u_str| [m/s]")
            quiver_overlay(X, Y, gd["u_abs"], gd["ux"], gd["uy"],
                           f"u_str quiver — {mode} {pname}",
                           pdir / f"u_str_quiver_{tag}.png")
            heatmap(X, Y, gd["curl_abs"],
                    f"|curl u_str| — {mode} {pname}",
                    pdir / f"curl_u_str_{tag}.png", label="|∇×u| [1/s]")
        # div and f_str only for mid-plane
        gd_mid = grid_data[(mode, "mid")]
        heatmap(X, Y, gd_mid["div_u"],
                f"div(u_str) — {mode} mid", pdir / f"div_u_str_{mode}_mid.png",
                cmap='RdBu_r', label="div(u) [1/s]", symmetric=True)
        heatmap(X, Y, gd_mid["f_abs"],
                f"|f_str| — {mode} mid", pdir / f"f_str_abs_{mode}_mid.png",
                label="|f| [Pa/m]")
    log(f"  Wrote {3*2*2 + 3*2} plot PNGs")

    # ── 6. Amplitude scaling check ──
    log("\n[6/7] Amplitude scaling check (0.5x, 1.0x, 2.0x)...")
    scales = [0.5, 1.0, 2.0]
    scaling_results = {s: {} for s in scales}

    for sc in scales:
        cfg_sc = make_cfg(sc)
        for mode in ["standing", "vortex"]:
            p_sol = solve_helmholtz(domain, facet_tags, cfg_sc,
                                    mode=mode, verbose=False)
            v1_vals, _ = compute_first_order_velocity(p_sol, domain, verbose=False)
            max_v1 = float(np.max(np.linalg.norm(np.abs(v1_vals), axis=1)))

            s_sol = solve_streaming_stokes(p_sol, domain=domain, verbose=False)
            max_u_str = float(s_sol.max_speed) if s_sol else 0.0
            scaling_results[sc][mode] = {"max_v1": max_v1, "max_u_str": max_u_str}
        log(f"  scale={sc}x done")

    # Scaling plots
    plt = _setup_mpl()
    for mode in ["standing", "vortex"]:
        v1_vals = [scaling_results[s][mode]["max_v1"] for s in scales]
        u_vals = [scaling_results[s][mode]["max_u_str"] for s in scales]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        fig.suptitle(f"Amplitude scaling — {mode}", fontsize=13)
        axes[0].plot(scales, v1_vals, 'o-', lw=2)
        axes[0].set_xlabel("Amplitude scale"); axes[0].set_ylabel("max|v1| [m/s]")
        axes[0].set_title("v1 vs scale (expect linear)")
        axes[1].plot(scales, u_vals, 's-', lw=2, color='C1')
        axes[1].set_xlabel("Amplitude scale"); axes[1].set_ylabel("max|u_str| [m/s]")
        axes[1].set_title("u_str vs scale (expect quadratic)")
        fig.tight_layout()
        fig.savefig(pdir / f"scaling_{mode}.png", dpi=140)
        plt.close(fig)
    log(f"  Wrote scaling PNGs")

    # ── 7. Sanity report ──
    log("\n[7/7] Writing sanity report...")
    report = _build_sanity_report(cfg, p_solutions, str_solutions,
                                   grid_data, scaling_results, scales, N)
    rpt_path = OUTDIR / "diagnostics" / "STREAMING_SANITY_REPORT.txt"
    with open(rpt_path, "w") as f:
        f.write(report)
    log(f"  Report: {rpt_path.name}")

    # Also save streaming solver diags as JSON
    for mode in ["standing", "vortex", "combined"]:
        d = str_solutions[mode].diagnostics
        dp = OUTDIR / "diagnostics" / f"streaming_diags_{mode}.json"
        with open(dp, "w") as f:
            # filter non-serializable
            json.dump({k: v for k, v in d.items()
                       if isinstance(v, (int, float, str, list, dict, bool))}, f, indent=2)

    total_t = time.time() - t0_all
    log(f"\n{'#'*70}")
    log(f"  STREAMING VALIDATION COMPLETE — {total_t:.1f} s total")
    log(f"  Output: {OUTDIR.resolve()}")
    log(f"{'#'*70}\n")

    # Print key numbers
    for mode in ["standing", "vortex", "combined"]:
        ss = str_solutions[mode]
        log(f"  {mode:10s}: max|u_str|={ss.max_speed*1e6:.3f} μm/s, "
            f"max|f|={ss.diagnostics['max_forcing_pa_m']:.2e} Pa/m, "
            f"div_L2_rel={ss.diagnostics['divergence_l2_norm_relative']:.2e}")


# ════════════════════════════════════════════════════════════════════
# Red flag writer
# ════════════════════════════════════════════════════════════════════
def _write_red_flag(msg, outdir):
    log(f"\n*** RED FLAG ***\n{msg}")
    rpt = outdir / "diagnostics" / "RED_FLAG.txt"
    (outdir / "diagnostics").mkdir(parents=True, exist_ok=True)
    with open(rpt, "w") as f:
        f.write(f"RED FLAG — {datetime.now().isoformat()}\n\n{msg}\n")


# ════════════════════════════════════════════════════════════════════
# Build sanity report text
# ════════════════════════════════════════════════════════════════════
def _build_sanity_report(cfg, p_solutions, str_solutions,
                          grid_data, scaling_results, scales, N):
    lines = []
    L = cfg.L
    def a(s): lines.append(s)

    a(f"STREAMING SANITY REPORT — {TODAY}")
    a("=" * 60)

    # ── Units sanity ──
    a("\n1. UNITS SANITY")
    a("-" * 40)
    for mode in ["standing", "vortex", "combined"]:
        ps = p_solutions[mode]
        ss = str_solutions[mode]
        d = ss.diagnostics
        a(f"  {mode}:")
        a(f"    max|p|        = {ps.max_pressure:.2f} Pa")
        a(f"    max|v1|       ~ {d['max_forcing_pa_m']:.2e} Pa/m  (forcing)")
        a(f"    max|u_str|    = {ss.max_speed:.4e} m/s = {ss.max_speed*1e6:.3f} μm/s")
        a(f"    mean|u_str|   = {ss.mean_speed:.4e} m/s")
        a(f"    Stokes ratio u_str/v1_typical ~ {ss.max_speed / (ps.max_pressure/(cfg.omega*cfg.rho) + 1e-30):.2e}")

    # ── Scaling checks ──
    a("\n2. AMPLITUDE SCALING")
    a("-" * 40)
    red_scaling = False
    for mode in ["standing", "vortex"]:
        v1_at = [scaling_results[s][mode]["max_v1"] for s in scales]
        u_at  = [scaling_results[s][mode]["max_u_str"] for s in scales]
        # v1 should scale linearly: v1(2x)/v1(0.5x) ≈ 4
        v1_ratio = v1_at[2] / (v1_at[0] + 1e-30)
        u_ratio  = u_at[2]  / (u_at[0]  + 1e-30)
        a(f"  {mode}:")
        a(f"    v1 ratio (2x/0.5x) = {v1_ratio:.3f}  (expect 4.0 for linear)")
        a(f"    u_str ratio (2x/0.5x) = {u_ratio:.3f}  (expect 16.0 for quadratic)")
        if abs(v1_ratio - 4.0) > 0.5:
            a(f"    *** RED FLAG: v1 scaling deviates from linear ***")
            red_scaling = True
        if abs(u_ratio - 16.0) > 4.0:
            a(f"    *** RED FLAG: u_str scaling deviates from quadratic ***")
            red_scaling = True

    # ── Incompressibility ──
    a("\n3. INCOMPRESSIBILITY (div u_str on mid-plane)")
    a("-" * 40)
    red_div = False
    for mode in ["standing", "vortex", "combined"]:
        gd = grid_data[(mode, "mid")]
        d = gd["div_u"]
        uabs = gd["u_abs"]
        max_div = float(np.nanmax(np.abs(d)))
        med_div = float(np.nanmedian(np.abs(d)))
        char_vel = float(np.nanmax(uabs)) + 1e-30
        char_grad = char_vel / L
        a(f"  {mode}:")
        a(f"    max |div(u)|   = {max_div:.4e} 1/s")
        a(f"    median |div(u)| = {med_div:.4e} 1/s")
        a(f"    char |u|/L     = {char_grad:.4e} 1/s")
        ratio = max_div / char_grad
        a(f"    max|div|/(|u|/L) = {ratio:.4f}")
        if ratio > 0.5:
            a(f"    *** RED FLAG: div(u) is NOT small relative to |u|/L ***")
            red_div = True
        else:
            a(f"    OK (ratio < 0.5)")

    # ── Symmetry ──
    a("\n4. SYMMETRY / STRUCTURE")
    a("-" * 40)
    red_struct = False
    for mode in ["standing", "vortex", "combined"]:
        gd = grid_data[(mode, "mid")]
        uabs = gd["u_abs"]
        # Check if basically zero
        if np.nanmax(uabs) < 1e-20:
            a(f"  {mode}: *** RED FLAG: u_str is effectively zero ***")
            red_struct = True
            continue
        # Standing: check x-reflection symmetry about L/2
        if mode == "standing":
            left = uabs[:, :N//2]
            right = uabs[:, N//2+1:][:, ::-1]
            sz = min(left.shape[1], right.shape[1])
            diff = np.nanmean(np.abs(left[:, :sz] - right[:, :sz]))
            ref = np.nanmean(uabs) + 1e-30
            a(f"  standing: mean |u_left - u_right_flipped| / mean|u| = {diff/ref:.4f}")
            if diff / ref > 0.3:
                a(f"    *** RED FLAG: poor x-mirror symmetry ***")
                red_struct = True
        # Vortex: check if curl is substantial (swirling)
        if mode == "vortex":
            curl_mid = gd["curl_abs"]
            max_curl = float(np.nanmax(curl_mid))
            a(f"  vortex: max|curl(u_str)| = {max_curl:.4e} 1/s")
            if max_curl < 1e-20:
                a(f"    *** RED FLAG: vortex streaming has no curl (no swirl) ***")
                red_struct = True
            else:
                a(f"    OK — non-zero curl present")

    # ── Forcing ↔ response correlation ──
    a("\n5. FORCING vs RESPONSE CORRELATION")
    a("-" * 40)
    red_corr = False
    for mode in ["standing", "vortex", "combined"]:
        d = str_solutions[mode].diagnostics
        max_f = d["max_forcing_pa_m"]
        max_u = str_solutions[mode].max_speed
        a(f"  {mode}: max|f| = {max_f:.4e} Pa/m, max|u| = {max_u:.4e} m/s")
        if max_f > 0 and max_u < 1e-30:
            a(f"    *** RED FLAG: non-zero forcing but zero response ***")
            red_corr = True

    # ── Red flags summary ──
    a("\n" + "=" * 60)
    flags = []
    if red_scaling: flags.append("SCALING")
    if red_div:     flags.append("DIVERGENCE")
    if red_struct:  flags.append("STRUCTURE/SYMMETRY")
    if red_corr:    flags.append("FORCING-RESPONSE")
    if flags:
        a(f"RED FLAGS RAISED: {', '.join(flags)}")
    else:
        a("NO RED FLAGS — all sanity checks passed.")
    a("=" * 60)

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
