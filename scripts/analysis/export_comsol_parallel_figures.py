#!/usr/bin/env python3
"""
Export COMSOL-comparison figures and CSVs for 3 cases.

Generates 4 figures per case (abs(p) slice, arg(p) slice,
isosurface-style abs(p), isosurface-style Re(p)) plus CSV plane/line
exports and meta information.

Output: COMSOL_comparison_results/

Usage:
    micromamba run -n acousto-complex python scripts/analysis/export_comsol_parallel_figures.py
"""
from __future__ import annotations

import sys, os, csv, json, time, textwrap
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np

# Must import before dolfinx to guarantee mpi4py init
from mpi4py import MPI

from dolfinx import fem, mesh
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
import ufl

from acoustweezers.experiments.shallow_square_dish.config import ShallowDishConfig
from acoustweezers.experiments.shallow_square_dish.solve_pressure import (
    create_mesh,
    solve_helmholtz,
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import TwoSlopeNorm

# ─── globals ────────────────────────────────────────────────────────
comm = MPI.COMM_WORLD
rank = comm.rank
ROOT = Path(__file__).resolve().parents[2]
OUTROOT = ROOT / "COMSOL_comparison_results"

CASES = {
    "Case_A_standing": "standing",
    "Case_B_vortex": "vortex",
    "Case_C_combined": "combined",
}

CASE_TITLES = {
    "Case_A_standing": "Case A — Standing",
    "Case_B_vortex": "Case B — Vortex",
    "Case_C_combined": "Case C — Combined",
}

N_PLANE = 201       # 201 × 201 grid for plane CSVs
N_LINE = 1001       # 1001 points for line CSVs


def log(msg="", end="\n"):
    if rank == 0:
        print(msg, end=end, flush=True)


# ═══════════════════════════════════════════════════════════════════
# CONFIG (locked — mirrors run_comsol_validation_lockdown.py)
# ═══════════════════════════════════════════════════════════════════
CFG = ShallowDishConfig(
    L=10e-3,
    H=1e-3,
    frequency_hz=500e3,
    elements_per_wavelength=6,
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


# ═══════════════════════════════════════════════════════════════════
# SAMPLING UTILITIES
# ═══════════════════════════════════════════════════════════════════
def _make_plane_grid(cfg, z_val, N):
    """Return (N*N, 3) array of sample points on a z-plane."""
    xs = np.linspace(0, cfg.L, N)
    ys = np.linspace(0, cfg.L, N)
    X, Y = np.meshgrid(xs, ys)
    return np.column_stack([X.ravel(), Y.ravel(), np.full(N * N, z_val)])


def _make_line_radial(cfg, z_val, N):
    """x from 0→L, y=L/2, z=z_val."""
    xs = np.linspace(0, cfg.L, N)
    ys = np.full(N, cfg.L / 2)
    zs = np.full(N, z_val)
    s = xs.copy()
    return s, np.column_stack([xs, ys, zs])


def _make_line_diagonal(cfg, z_val, N):
    """(0,0)→(L,L), z=z_val."""
    t = np.linspace(0, 1, N)
    xs = t * cfg.L
    ys = t * cfg.L
    zs = np.full(N, z_val)
    s = np.sqrt(xs ** 2 + ys ** 2)
    return s, np.column_stack([xs, ys, zs])


def _sample_pressure(p_func, pts):
    """Sample complex pressure at pts. Returns (N,) complex array; NaN where not found."""
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


# ═══════════════════════════════════════════════════════════════════
# CSV EXPORT
# ═══════════════════════════════════════════════════════════════════
def _write_plane_csv(path, pts, pvals):
    """Write plane CSV with columns x,y,z,Re(p),Im(p),abs(p),arg(p)."""
    mask = ~np.isnan(pvals)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "y", "z", "Re(p)", "Im(p)", "abs(p)", "arg(p)"])
        for i in np.where(mask)[0]:
            p = pvals[i]
            w.writerow([
                f"{pts[i,0]:.8e}", f"{pts[i,1]:.8e}", f"{pts[i,2]:.8e}",
                f"{p.real:.8e}", f"{p.imag:.8e}", f"{abs(p):.8e}", f"{np.angle(p):.8e}",
            ])
    return int(mask.sum())


def _write_line_csv(path, s_arr, pts, pvals):
    """Write line CSV with columns s,x,y,z,Re(p),Im(p),abs(p),arg(p)."""
    mask = ~np.isnan(pvals)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["s", "x", "y", "z", "Re(p)", "Im(p)", "abs(p)", "arg(p)"])
        for i in np.where(mask)[0]:
            p = pvals[i]
            w.writerow([
                f"{s_arr[i]:.8e}",
                f"{pts[i,0]:.8e}", f"{pts[i,1]:.8e}", f"{pts[i,2]:.8e}",
                f"{p.real:.8e}", f"{p.imag:.8e}", f"{abs(p):.8e}", f"{np.angle(p):.8e}",
            ])
    return int(mask.sum())


# ═══════════════════════════════════════════════════════════════════
# FIGURE GENERATION
# ═══════════════════════════════════════════════════════════════════
def _disc_circle(cfg):
    """Return a matplotlib Circle for the disc in mm."""
    cx = cfg.L / 2 * 1e3
    cy = cfg.L / 2 * 1e3
    R = cfg.bottom_disc_radius_effective * 1e3
    return Circle((cx, cy), R, fill=False, edgecolor="white", linewidth=1.2, linestyle="--")


def _reshape_plane(pvals, N):
    """Reshape (N*N,) → (N, N) with NaN fill."""
    return pvals.reshape(N, N)


def _make_extent_mm(cfg):
    """(left, right, bottom, top) in mm for imshow."""
    return [0, cfg.L * 1e3, 0, cfg.L * 1e3]


def plot_slice_abs(pvals_2d, cfg, title, path):
    """Slice figure: abs(p) with disc overlay."""
    ext = _make_extent_mm(cfg)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    data = np.abs(pvals_2d)
    im = ax.imshow(data, origin="lower", extent=ext, cmap="inferno", aspect="equal")
    cb = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.03)
    cb.set_label("|p|  (Pa)", fontsize=11)
    ax.add_patch(_disc_circle(cfg))
    ax.set_xlabel("x  (mm)", fontsize=11)
    ax.set_ylabel("y  (mm)", fontsize=11)
    ax.set_title(f"{title}\nSlice z = H/2 — |p|", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_slice_arg(pvals_2d, cfg, title, path):
    """Slice figure: arg(p) with disc overlay."""
    ext = _make_extent_mm(cfg)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    data = np.angle(pvals_2d)
    im = ax.imshow(data, origin="lower", extent=ext, cmap="twilight_shifted",
                   vmin=-np.pi, vmax=np.pi, aspect="equal")
    cb = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.03)
    cb.set_label("arg(p)  (rad)", fontsize=11)
    cb.set_ticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    cb.set_ticklabels(["-π", "-π/2", "0", "π/2", "π"])
    ax.add_patch(_disc_circle(cfg))
    ax.set_xlabel("x  (mm)", fontsize=11)
    ax.set_ylabel("y  (mm)", fontsize=11)
    ax.set_title(f"{title}\nSlice z = H/2 — arg(p)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_iso_abs(pvals_2d, cfg, title, path):
    """Isosurface-style abs(p): filled contour + contour lines, positive-only colorbar."""
    ext = _make_extent_mm(cfg)
    xs_mm = np.linspace(0, cfg.L * 1e3, pvals_2d.shape[1])
    ys_mm = np.linspace(0, cfg.L * 1e3, pvals_2d.shape[0])
    X, Y = np.meshgrid(xs_mm, ys_mm)
    data = np.abs(pvals_2d)
    n_levels = 20

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    cf = ax.contourf(X, Y, data, levels=n_levels, cmap="inferno")
    ax.contour(X, Y, data, levels=n_levels, colors="k", linewidths=0.35, alpha=0.5)
    cb = fig.colorbar(cf, ax=ax, shrink=0.82, pad=0.03)
    cb.set_label("|p|  (Pa)", fontsize=11)
    ax.add_patch(_disc_circle(cfg))
    ax.set_xlabel("x  (mm)", fontsize=11)
    ax.set_ylabel("y  (mm)", fontsize=11)
    ax.set_title(f"{title}\nIsosurface — |p|", fontsize=12)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_iso_total(pvals_2d, cfg, title, path):
    """Isosurface-style Re(p): filled contour + contour lines, diverging ± colorbar."""
    ext = _make_extent_mm(cfg)
    xs_mm = np.linspace(0, cfg.L * 1e3, pvals_2d.shape[1])
    ys_mm = np.linspace(0, cfg.L * 1e3, pvals_2d.shape[0])
    X, Y = np.meshgrid(xs_mm, ys_mm)
    data = np.real(pvals_2d)
    n_levels = 20

    vmax = max(abs(np.nanmin(data)), abs(np.nanmax(data)))
    if vmax < 1e-15:
        vmax = 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    cf = ax.contourf(X, Y, data, levels=n_levels, cmap="RdBu_r", norm=norm)
    ax.contour(X, Y, data, levels=n_levels, colors="k", linewidths=0.35, alpha=0.5)
    cb = fig.colorbar(cf, ax=ax, shrink=0.82, pad=0.03)
    cb.set_label("Re(p)  (Pa)", fontsize=11)
    ax.add_patch(_disc_circle(cfg))
    ax.set_xlabel("x  (mm)", fontsize=11)
    ax.set_ylabel("y  (mm)", fontsize=11)
    ax.set_title(f"{title}\nIsosurface — Total acoustic pressure", fontsize=12)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# META / README / MANIFEST
# ═══════════════════════════════════════════════════════════════════
def write_config_json(cfg, path):
    """Dump config to JSON."""
    d = cfg.to_dict()
    # Add derived values
    d["_derived"] = {
        "omega": cfg.omega,
        "k": cfg.k,
        "wavelength": cfg.wavelength,
        "Z_water": cfg.Z_water,
        "Z_top": cfg.Z_top,
        "mesh_nx": cfg.mesh_nx,
        "mesh_nz": cfg.mesh_nz,
    }
    with open(path, "w") as f:
        json.dump(d, f, indent=2, default=str)


def write_solver_report(cfg, p_sol, mode, n_cells, path):
    """Write solver_report.txt."""
    pvals = p_sol.p_values
    max_p = p_sol.max_pressure
    coords = p_sol.coords
    idx_max = np.argmax(np.abs(pvals))
    loc = coords[idx_max]
    lines = [
        f"Solver Report — {mode}",
        f"Generated: {datetime.now().isoformat()}",
        "",
        f"Mode:            {mode}",
        f"DOFs:            {len(pvals)}",
        f"Mesh cells:      {n_cells}",
        f"Element order:   P2 (Lagrange 2)",
        f"Solver:          GMRES + ILU(0)",
        f"rtol:            1e-8",
        f"max iterations:  3000",
        "",
        f"max(|p|):        {max_p:.4f} Pa",
        f"  location:      ({loc[0]*1e3:.3f}, {loc[1]*1e3:.3f}, {loc[2]*1e3:.3f}) mm",
        "",
        f"Frequency:       {cfg.frequency_hz/1e3:.0f} kHz",
        f"k:               {cfg.k:.6f} rad/m",
        f"Wavelength:      {cfg.wavelength*1e3:.3f} mm",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_readme(outroot):
    text = textwrap.dedent(f"""\
    # COMSOL Comparison Results

    Generated: {datetime.now().isoformat()}

    ## How to regenerate

    ```bash
    cd {ROOT}
    micromamba run -n acousto-complex python scripts/analysis/export_comsol_parallel_figures.py
    ```

    ## Structure

    ```
    COMSOL_comparison_results/
      Case_A_standing/   — standing wave only
      Case_B_vortex/     — vortex beam only
      Case_C_combined/   — standing + vortex
      README.md          — this file
      MANIFEST.txt       — full file list
    ```

    Each case contains:
    - `figs/`  — 4 PNG figures (slice abs, slice arg, iso abs, iso Re)
    - `csv/`   — plane grid + 1D line exports
    - `meta/`  — config_used.json, solver_report.txt

    ## Parameters (locked)
    - L = 10 mm, H = 1 mm, f = 500 kHz
    - rho = 997 kg/m³, c = 1484 m/s
    - V_s = V_0 = 10 µm/s, ℓ = 1
    - R_disc = 3 mm, Z_top = 0.001·ρc
    - P2 elements, 20×20×8 structured tet mesh
    """)
    with open(outroot / "README.md", "w") as f:
        f.write(text)


def write_manifest(outroot):
    """Write MANIFEST.txt and return its contents."""
    lines = []
    for p in sorted(outroot.rglob("*")):
        if p.is_file():
            rel = p.relative_to(outroot)
            lines.append(str(rel))
    lines.append("MANIFEST.txt")
    lines = sorted(set(lines))
    text = "\n".join(lines) + "\n"
    with open(outroot / "MANIFEST.txt", "w") as f:
        f.write(text)
    return text


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    cfg = CFG

    log(f"\n{'#'*70}")
    log(f"  COMSOL COMPARISON FIGURE EXPORT")
    log(f"  Output: {OUTROOT}")
    log(f"{'#'*70}\n")

    # Create directory structure
    for case_dir in CASES:
        for sub in ("figs", "csv", "meta"):
            (OUTROOT / case_dir / sub).mkdir(parents=True, exist_ok=True)

    # Create mesh once (shared across all 3 cases)
    domain, facet_tags, tag_map = create_mesh(cfg, verbose=True)
    n_cells = domain.topology.index_map(domain.topology.dim).size_global

    # Precompute sample point arrays
    z_mid = cfg.H / 2.0
    z_bot = cfg.H / 10.0
    pts_plane_mid = _make_plane_grid(cfg, z_mid, N_PLANE)
    pts_plane_bot = _make_plane_grid(cfg, z_bot, N_PLANE)
    s_rad, pts_rad = _make_line_radial(cfg, z_mid, N_LINE)
    s_diag, pts_diag = _make_line_diagonal(cfg, z_mid, N_LINE)

    for case_dir, mode in CASES.items():
        title = CASE_TITLES[case_dir]
        cdir = OUTROOT / case_dir

        log(f"\n{'='*70}")
        log(f"  {title}  (mode={mode})")
        log(f"{'='*70}")

        # ── Solve Helmholtz ──
        p_sol = solve_helmholtz(domain, facet_tags, cfg, mode=mode, verbose=True)

        # ── Meta ──
        write_config_json(cfg, cdir / "meta" / "config_used.json")
        write_solver_report(cfg, p_sol, mode, n_cells, cdir / "meta" / "solver_report.txt")

        # ── Sample planes ──
        log("  Sampling z=H/2 plane...", end=" ")
        p_mid = _sample_pressure(p_sol.p_function, pts_plane_mid)
        nrows = _write_plane_csv(cdir / "csv" / "plane_z_H2.csv", pts_plane_mid, p_mid)
        log(f"{nrows} rows")

        log("  Sampling z=H/10 plane...", end=" ")
        p_bot = _sample_pressure(p_sol.p_function, pts_plane_bot)
        nrows = _write_plane_csv(cdir / "csv" / "plane_z_H10.csv", pts_plane_bot, p_bot)
        log(f"{nrows} rows")

        # ── Sample lines ──
        log("  Sampling radial line...", end=" ")
        p_rad = _sample_pressure(p_sol.p_function, pts_rad)
        nrows = _write_line_csv(cdir / "csv" / "line_radial_y_mid_z_H2.csv", s_rad, pts_rad, p_rad)
        log(f"{nrows} rows")

        log("  Sampling diagonal line...", end=" ")
        p_diag = _sample_pressure(p_sol.p_function, pts_diag)
        nrows = _write_line_csv(cdir / "csv" / "line_diagonal_z_H2.csv", s_diag, pts_diag, p_diag)
        log(f"{nrows} rows")

        # ── Figures ──
        log("  Generating figures...")
        p2d = _reshape_plane(p_mid, N_PLANE)

        plot_slice_abs(p2d, cfg, title, cdir / "figs" / "slice_abs_p_z_H2.png")
        log("    → slice_abs_p_z_H2.png")

        plot_slice_arg(p2d, cfg, title, cdir / "figs" / "slice_arg_p_z_H2.png")
        log("    → slice_arg_p_z_H2.png")

        plot_iso_abs(p2d, cfg, title, cdir / "figs" / "isosurface_abs_p.png")
        log("    → isosurface_abs_p.png")

        plot_iso_total(p2d, cfg, title, cdir / "figs" / "isosurface_total_pressure.png")
        log("    → isosurface_total_pressure.png")

    # ── Top-level outputs ──
    write_readme(OUTROOT)
    log("\n  Wrote README.md")

    manifest = write_manifest(OUTROOT)
    log("  Wrote MANIFEST.txt")

    elapsed = time.time() - t0
    log(f"\n{'#'*70}")
    log(f"  DONE: outputs in {OUTROOT}")
    log(f"  Elapsed: {elapsed:.1f} s")
    log(f"{'#'*70}\n")

    log("MANIFEST:")
    log(manifest)


if __name__ == "__main__":
    main()
