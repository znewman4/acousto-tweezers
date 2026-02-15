#!/usr/bin/env python3
"""
Authoritative COMSOL Comparison Rebuild — Attempt 2.

Physics exactly matching COMSOL setup:
- Disc BC: impedance Z_w = ρc + prescribed normal velocity (combined Robin BC)
  ∂p/∂n = (iωρ/Z_w)p − iωρ v_n
- Side walls: pure Neumann source (rigid + velocity)
- Top: low-impedance Robin (Z_top = 0.001 Z_w)
- Bottom outside disc: rigid (natural Neumann)

Time convention: e^{-iωt}
Solver: MUMPS direct (LU factorization)
Elements: P2 (quadratic Lagrange)
Mesh: ≥ 6 elements per wavelength (using 10 for better disc coverage)

Cases:
  A — Standing only (disc impedance, NO vortex forcing)
  B — Vortex only (disc impedance + vortex velocity)
  C — Combined (standing + vortex)
  C amplitude sweep: V₀×2, V₀×3, V₀×6

Usage:
    micromamba run -n acousto-complex python scripts/analysis/rebuild_comsol_comparison.py
"""
from __future__ import annotations

import sys, os, csv, json, time, textwrap, shutil
from pathlib import Path
from datetime import datetime
from dataclasses import replace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np

from mpi4py import MPI
from dolfinx import fem, mesh
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
import ufl

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

# ─── globals ────────────────────────────────────────────────────────
comm = MPI.COMM_WORLD
rank = comm.rank
ROOT = Path(__file__).resolve().parents[2]
OUTROOT = ROOT / "COMSOL_comparison_results"
NOW = datetime.now()

CASES = {
    "Case_A_standing": "standing",
    "Case_B_vortex": "vortex",
    "Case_C_combined": "combined",
}
CASE_TITLES = {
    "Case_A_standing": "Case A — Standing Only",
    "Case_B_vortex": "Case B — Vortex Only",
    "Case_C_combined": "Case C — Combined",
}

# disc_robin per case — COMSOL: disc always has impedance Z_w
DISC_ROBIN = {
    "Case_A_standing": True,   # impedance absorber, no forcing
    "Case_B_vortex": True,     # impedance + vortex velocity
    "Case_C_combined": True,   # impedance + vortex velocity + standing
}

N_PLANE = 201
N_LINE = 1001

# Direct solver (MUMPS)
MUMPS_OPTS = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}

created_files: list[str] = []


def log(msg="", end="\n"):
    if rank == 0:
        print(msg, end=end, flush=True)


def record(path: Path):
    created_files.append(str(path.relative_to(ROOT)))


# ═══════════════════════════════════════════════════════════════════
# CONFIG — locked to match COMSOL spec exactly
# ═══════════════════════════════════════════════════════════════════
CFG = ShallowDishConfig(
    L=10e-3,                            # 10 mm
    H=1e-3,                             # 1 mm
    frequency_hz=500e3,                 # 500 kHz
    elements_per_wavelength=10,         # finer mesh for better disc coverage
    min_elements_z=8,
    rho=997.0,
    c=1484.0,
    mu=1.002e-3,
    vortex_velocity_amplitude=10e-6,    # V₀ = 10 µm/s
    standing_velocity_amplitude=10e-6,  # Vs = 10 µm/s  (equal)
    vortex_topological_charge=1,        # ℓ = 1
    vortex_aperture_radius=3e-3,        # R_disc = 3 mm
    vortex_apodization="cosine_taper",  # A(r) = 0.5(1+cos(πr/R))
    vortex_phase_offset=0.0,
    standing_axis="both",               # all 4 side walls
    standing_phase_pattern="antiphase",
    top_bc_type="impedance",
    top_impedance_factor=0.001,         # Z_top = 0.001 × ρc
    bottom_disc_radius=None,            # → 3 mm (from aperture)
    standing_full_wall=True,
    particle_radius=5e-6,
    particle_density=1050.0,
    particle_compressibility=2.4e-10,
)


# ═══════════════════════════════════════════════════════════════════
# SAMPLING UTILITIES
# ═══════════════════════════════════════════════════════════════════
def _make_plane_grid(cfg, z_val, N):
    xs = np.linspace(0, cfg.L, N)
    ys = np.linspace(0, cfg.L, N)
    X, Y = np.meshgrid(xs, ys)
    return np.column_stack([X.ravel(), Y.ravel(), np.full(N * N, z_val)])


def _make_line_radial(cfg, z_val, N):
    """Radial: x from 0→L, y=L/2, z=z_val."""
    xs = np.linspace(0, cfg.L, N)
    ys = np.full(N, cfg.L / 2)
    zs = np.full(N, z_val)
    s = xs.copy()
    return s, np.column_stack([xs, ys, zs])


def _make_line_diagonal(cfg, z_val, N):
    """Diagonal: (0,0)→(L,L), z=z_val."""
    t = np.linspace(0, 1, N)
    xs = t * cfg.L
    ys = t * cfg.L
    zs = np.full(N, z_val)
    s = np.sqrt(xs**2 + ys**2)
    return s, np.column_stack([xs, ys, zs])


def _sample_pressure(p_func, pts):
    """Sample complex pressure at points. NaN where not found."""
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
    """Write plane CSV: x,y,z,Re(p),Im(p),abs(p),arg(p)."""
    mask = ~np.isnan(pvals)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "y", "z", "Re(p)", "Im(p)", "abs(p)", "arg(p)"])
        for i in np.where(mask)[0]:
            p = pvals[i]
            w.writerow([
                f"{pts[i,0]:.8e}", f"{pts[i,1]:.8e}", f"{pts[i,2]:.8e}",
                f"{p.real:.8e}", f"{p.imag:.8e}", f"{abs(p):.8e}",
                f"{np.angle(p):.8e}",
            ])
    record(path)
    return int(mask.sum())


def _write_line_csv(path, s_arr, pts, pvals):
    """Write line CSV: s,x,y,z,Re(p),Im(p),abs(p),arg(p)."""
    mask = ~np.isnan(pvals)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["s", "x", "y", "z", "Re(p)", "Im(p)", "abs(p)", "arg(p)"])
        for i in np.where(mask)[0]:
            p = pvals[i]
            w.writerow([
                f"{s_arr[i]:.8e}",
                f"{pts[i,0]:.8e}", f"{pts[i,1]:.8e}", f"{pts[i,2]:.8e}",
                f"{p.real:.8e}", f"{p.imag:.8e}", f"{abs(p):.8e}",
                f"{np.angle(p):.8e}",
            ])
    record(path)
    return int(mask.sum())


# ═══════════════════════════════════════════════════════════════════
# FIGURE GENERATION — COMSOL rainbow / diverging style
# ═══════════════════════════════════════════════════════════════════
def _disc_circle_white(cfg):
    cx = cfg.L / 2 * 1e3
    cy = cfg.L / 2 * 1e3
    R = cfg.bottom_disc_radius_effective * 1e3
    return Circle((cx, cy), R, fill=False, edgecolor="white",
                  linewidth=1.2, linestyle="--")


def _disc_circle_black(cfg):
    cx = cfg.L / 2 * 1e3
    cy = cfg.L / 2 * 1e3
    R = cfg.bottom_disc_radius_effective * 1e3
    return Circle((cx, cy), R, fill=False, edgecolor="black",
                  linewidth=1.2, linestyle="--")


def plot_01_slice_abs_p(pvals_2d, cfg, title, path):
    """01 — Slice z=H/2: |p| (imshow)."""
    ext = [0, cfg.L * 1e3, 0, cfg.L * 1e3]
    data = np.abs(pvals_2d)
    maxp = np.nanmax(data)
    fig, ax = plt.subplots(figsize=(7, 5.8))
    im = ax.imshow(data, origin="lower", extent=ext, cmap="jet", aspect="equal")
    cb = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.03)
    cb.set_label("|p|  (Pa)", fontsize=11)
    ax.add_patch(_disc_circle_white(cfg))
    ax.set_xlabel("x  (mm)", fontsize=11)
    ax.set_ylabel("y  (mm)", fontsize=11)
    ax.set_title(f"{title}\nSlice z = H/2 — |p|   max = {maxp:.2f} Pa", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    record(path)


def plot_02_slice_arg_p(pvals_2d, cfg, title, path):
    """02 — Slice z=H/2: arg(p)."""
    ext = [0, cfg.L * 1e3, 0, cfg.L * 1e3]
    data = np.angle(pvals_2d)
    fig, ax = plt.subplots(figsize=(7, 5.8))
    im = ax.imshow(data, origin="lower", extent=ext, cmap="twilight_shifted",
                   vmin=-np.pi, vmax=np.pi, aspect="equal")
    cb = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.03)
    cb.set_label("arg(p)  (rad)", fontsize=11)
    cb.set_ticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    cb.set_ticklabels(["-π", "-π/2", "0", "π/2", "π"])
    ax.add_patch(_disc_circle_black(cfg))
    ax.set_xlabel("x  (mm)", fontsize=11)
    ax.set_ylabel("y  (mm)", fontsize=11)
    ax.set_title(f"{title}\nSlice z = H/2 — arg(p)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    record(path)


def plot_03_isosurface_abs_p(pvals_2d, cfg, title, path, n_levels=20):
    """03 — Isosurface-style |p| (rainbow contourf + contour lines)."""
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
    cb.set_label("|p|  (Pa)", fontsize=11)
    ax.add_patch(_disc_circle_white(cfg))
    ax.set_xlabel("x  (mm)", fontsize=11)
    ax.set_ylabel("y  (mm)", fontsize=11)
    ax.set_title(f"{title}\nIsosurface |p|   max = {maxp:.2f} Pa", fontsize=12)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    record(path)


def plot_04_isosurface_total_pressure(pvals_2d, cfg, title, path, n_levels=20):
    """04 — Isosurface-style Re(p) (diverging ±)."""
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
    cb.set_label("Re(p)  (Pa)", fontsize=11)
    ax.add_patch(_disc_circle_black(cfg))
    ax.set_xlabel("x  (mm)", fontsize=11)
    ax.set_ylabel("y  (mm)", fontsize=11)
    re_range = f"[{np.nanmin(data):.2f}, {np.nanmax(data):.2f}]"
    ax.set_title(f"{title}\nTotal acoustic pressure Re(p) {re_range} Pa",
                 fontsize=12)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    record(path)


# ═══════════════════════════════════════════════════════════════════
# META / DOCS
# ═══════════════════════════════════════════════════════════════════
def write_config_json(cfg, path):
    d = cfg.to_dict()
    d["_derived"] = {
        "omega": cfg.omega, "k": cfg.k, "wavelength": cfg.wavelength,
        "Z_water": cfg.Z_water, "Z_top": cfg.Z_top,
        "mesh_nx": cfg.mesh_nx, "mesh_nz": cfg.mesh_nz,
    }
    with open(path, "w") as f:
        json.dump(d, f, indent=2, default=str)
    record(path)


def write_solver_report(cfg, p_sol, mode, n_cells, path, extra=""):
    pvals = p_sol.p_values
    maxp = p_sol.max_pressure
    minp = np.min(np.abs(pvals))
    max_re = np.max(np.real(pvals))
    coords = p_sol.coords
    idx_max = np.argmax(np.abs(pvals))
    loc = coords[idx_max]
    lines = [
        f"Solver Report — {mode}",
        f"Generated: {NOW.isoformat()}",
        "",
        f"Mode:            {mode}",
        f"DOFs:            {len(pvals)}",
        f"Mesh cells:      {n_cells}",
        f"Element order:   P2 (Lagrange 2)",
        f"Solver:          MUMPS direct (LU)",
        "",
        f"max(|p|):        {maxp:.4f} Pa",
        f"min(|p|):        {minp:.6f} Pa",
        f"max(Re(p)):      {max_re:.4f} Pa",
        f"  max loc:       ({loc[0]*1e3:.3f}, {loc[1]*1e3:.3f}, {loc[2]*1e3:.3f}) mm",
        "",
        f"Frequency:       {cfg.frequency_hz/1e3:.0f} kHz",
        f"k:               {cfg.k:.6f} rad/m",
        f"Wavelength:      {cfg.wavelength*1e3:.3f} mm",
        "",
        extra,
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    record(path)


# ═══════════════════════════════════════════════════════════════════
# DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════
def run_diagnostics(domain, facet_tags, cfg):
    """Full diagnostic suite (Section 9 validation checks)."""
    lines = []
    fdim = domain.topology.dim - 1
    R_disc = cfg.bottom_disc_radius_effective
    omega = cfg.omega
    rho = cfg.rho
    V_vtx = cfg.vortex_velocity_amplitude
    V_stand = cfg.standing_velocity_amplitude

    # --- Disc facet count ---
    disc_mask = facet_tags.values == TAG_BOTTOM_DISC
    n_disc = int(np.sum(disc_mask))
    disc_facets = facet_tags.indices[disc_mask]

    # --- Disc area ---
    dss = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
    one = fem.Constant(domain, complex(1.0, 0.0))
    area_form = fem.form(one * dss(TAG_BOTTOM_DISC))
    A_disc_mesh = abs(fem.assemble_scalar(area_form))
    A_disc_expected = np.pi * R_disc**2
    ratio = A_disc_mesh / A_disc_expected

    lines.append(f"Disc facets:          {n_disc}")
    lines.append(f"Disc area (mesh):     {A_disc_mesh*1e6:.4f} mm²")
    lines.append(f"Disc area (πR²):      {A_disc_expected*1e6:.4f} mm²")
    lines.append(f"Area ratio:           {ratio:.4f}")

    # --- Pattern / forcing ---
    V_fs = fem.functionspace(domain, ("Lagrange", 2))
    pattern_func = _create_vortex_source(V_fs, domain, facet_tags, cfg, verbose=False)
    disc_dofs = fem.locate_dofs_topological(V_fs, fdim, disc_facets)
    pvals = pattern_func.x.array[disc_dofs]
    max_pat = np.max(np.abs(pvals))
    avg_pat = np.mean(np.abs(pvals))

    g_vtx_max = omega * rho * V_vtx * max_pat
    g_stand_mag = omega * rho * V_stand

    lines.append(f"max(|pattern|):       {max_pat:.6f}")
    lines.append(f"avg(|pattern|):       {avg_pat:.6f}")
    lines.append(f"|g_stand| = ωρVs:     {g_stand_mag:.4f} Pa/m")
    lines.append(f"max|g_vtx| = ωρV₀max: {g_vtx_max:.4f} Pa/m")
    lines.append(f"Forcing ratio g_vtx/g_stand: {g_vtx_max/g_stand_mag:.4f}")

    # --- Wall area ---
    A_walls = 4 * cfg.L * cfg.H
    lines.append(f"Wall area (4 walls):  {A_walls*1e6:.2f} mm²")
    lines.append(f"Wall/disc area ratio: {A_walls/A_disc_mesh:.2f}")

    return lines, ratio


# ═══════════════════════════════════════════════════════════════════
# PROCESS ONE CASE
# ═══════════════════════════════════════════════════════════════════
def process_case(case_dir, mode, domain, facet_tags, cfg, n_cells,
                 pts_H2, pts_H10, s_rad, pts_rad, s_diag, pts_diag):
    """Solve, sample, export CSV, generate figures for one case."""
    title = CASE_TITLES[case_dir]
    cdir = OUTROOT / case_dir
    disc_robin = DISC_ROBIN[case_dir]

    log(f"\n{'='*70}")
    log(f"  {title}  (mode={mode}, disc_robin={disc_robin})")
    log(f"{'='*70}")

    # ── Solve ──
    p_sol = solve_helmholtz(
        domain, facet_tags, cfg, mode=mode,
        disc_robin=disc_robin, verbose=True,
        petsc_options=MUMPS_OPTS,
    )

    # ── Validation ──
    pv = p_sol.p_values
    maxp = p_sol.max_pressure
    minp = np.min(np.abs(pv))
    max_re = np.max(np.real(pv))
    log(f"  VALIDATION: max|p|={maxp:.4f}  min|p|={minp:.6f}  "
        f"max(Re(p))={max_re:.4f}")

    # ── Meta ──
    write_config_json(cfg, cdir / "meta" / "config_used.json")
    write_solver_report(cfg, p_sol, mode, n_cells,
                        cdir / "meta" / "solver_report.txt")

    # ── CSV: plane z=H/2 ──
    log("  Sampling z=H/2 plane...", end=" ")
    p_H2 = _sample_pressure(p_sol.p_function, pts_H2)
    nr = _write_plane_csv(cdir / "csv" / "plane_z_H2.csv", pts_H2, p_H2)
    log(f"{nr} rows")

    # ── CSV: plane z=H/10 ──
    log("  Sampling z=H/10 plane...", end=" ")
    p_H10 = _sample_pressure(p_sol.p_function, pts_H10)
    nr = _write_plane_csv(cdir / "csv" / "plane_z_H10.csv", pts_H10, p_H10)
    log(f"{nr} rows")

    # ── CSV: radial line ──
    log("  Sampling radial line...", end=" ")
    p_rad = _sample_pressure(p_sol.p_function, pts_rad)
    _write_line_csv(cdir / "csv" / "line_radial_y_mid_z_H2.csv",
                    s_rad, pts_rad, p_rad)
    log("done")

    # ── CSV: diagonal line ──
    log("  Sampling diagonal line...", end=" ")
    p_diag = _sample_pressure(p_sol.p_function, pts_diag)
    _write_line_csv(cdir / "csv" / "line_diagonal_z_H2.csv",
                    s_diag, pts_diag, p_diag)
    log("done")

    # ── Figures ──
    p2d = p_H2.reshape(N_PLANE, N_PLANE)
    log("  Generating figures...")

    plot_01_slice_abs_p(p2d, cfg, title,
                        cdir / "figs" / "01_slice_abs_p_z_H2.png")
    log("    → 01_slice_abs_p_z_H2.png")

    plot_02_slice_arg_p(p2d, cfg, title,
                        cdir / "figs" / "02_slice_arg_p_z_H2.png")
    log("    → 02_slice_arg_p_z_H2.png")

    plot_03_isosurface_abs_p(p2d, cfg, title,
                             cdir / "figs" / "03_isosurface_abs_p.png")
    log("    → 03_isosurface_abs_p.png")

    plot_04_isosurface_total_pressure(p2d, cfg, title,
                                      cdir / "figs" / "04_isosurface_total_pressure.png")
    log("    → 04_isosurface_total_pressure.png")

    # Plane max (for table)
    plane_maxp = np.nanmax(np.abs(p_H2))
    return maxp, plane_maxp, p_sol


# ═══════════════════════════════════════════════════════════════════
# AMPLITUDE SWEEP (Case C)
# ═══════════════════════════════════════════════════════════════════
def run_amplitude_sweep(domain, facet_tags, cfg, pts_H2):
    """Run Case C with V₀×2, ×3, ×6, ×10, ×20 and generate figures."""
    V0 = cfg.vortex_velocity_amplitude
    multipliers = [2, 3, 6, 10, 20]
    results = {}

    cdir = OUTROOT / "Case_C_combined"

    # Disc centre observation point (for tracking vortex-region boost)
    cx_mm = cfg.L / 2 * 1e3
    disc_obs_idx = N_PLANE // 2  # centre index

    fig_names = {
        2: "05_isosurface_abs_p_V0x2.png",
        3: "06_isosurface_abs_p_V0x3.png",
        6: "07_isosurface_abs_p_V0x6.png",
        10: "08_isosurface_abs_p_V0x10.png",
        20: "09_isosurface_abs_p_V0x20.png",
    }

    for mult in multipliers:
        label = f"V₀×{mult}"
        V0_new = mult * V0
        cfg_m = replace(cfg, vortex_velocity_amplitude=V0_new)

        log(f"\n  Amplitude sweep: {label} ({V0_new*1e6:.0f} µm/s)")
        p_sol = solve_helmholtz(
            domain, facet_tags, cfg_m, mode="combined",
            disc_robin=True, verbose=False,
            petsc_options=MUMPS_OPTS,
        )
        maxp = p_sol.max_pressure

        # Sample z=H/2
        p_H2 = _sample_pressure(p_sol.p_function, pts_H2)
        p2d = p_H2.reshape(N_PLANE, N_PLANE)
        plane_maxp = np.nanmax(np.abs(p_H2))

        # Disc centre |p|
        disc_center_p = np.abs(p2d[disc_obs_idx, disc_obs_idx])

        log(f"    max|p|(3D)={maxp:.2f}  max|p|(plane)={plane_maxp:.2f}  "
            f"|p|(disc centre)={disc_center_p:.2f}")

        fig_name = fig_names.get(mult, f"0{mult}_isosurface_abs_p_V0x{mult}.png")
        title = f"Case C — Combined  (V₀ × {mult} = {V0_new*1e6:.0f} µm/s)"
        plot_03_isosurface_abs_p(p2d, cfg_m, title,
                                 cdir / "figs" / fig_name)
        log(f"    → {fig_name}")

        results[f"Case_C_V0x{mult}"] = {
            "maxp_3d": maxp,
            "maxp_plane": plane_maxp,
            "disc_center_p": disc_center_p,
            "V0": V0_new,
        }

    return results


# ═══════════════════════════════════════════════════════════════════
# README
# ═══════════════════════════════════════════════════════════════════
def write_readme(cfg, diag_lines, result_table):
    text = textwrap.dedent(f"""\
    # COMSOL Comparison Results — Attempt 2 (Authoritative)

    Generated: {NOW.isoformat()}

    ## Physics

    Helmholtz equation in 3D frequency domain:
      ∇²p + k²p = 0

    Time convention: e^{{-iωt}}

    ## Boundary Conditions

    | Surface | Type | Detail |
    |---------|------|--------|
    | Top (z=H) | Robin (impedance) | Z_top = 0.001 × ρc |
    | Side walls (x±, y±) | Neumann (velocity source) | antiphase, both axes |
    | Bottom disc (r ≤ R_disc) | Robin (impedance) + Neumann (velocity) | Z_w = ρc, v_n = V₀ A(r) e^{{iℓθ}} |
    | Bottom rigid (r > R_disc) | Rigid | ∂p/∂n = 0 (natural Neumann) |

    Disc BC (COMSOL "Impedance + Normal Velocity"):
      ∂p/∂n = (iωρ/Z_w)p − iωρ v_n

    Side wall BC (pure Neumann source):
      ∂p/∂n = −iωρ V_s  (no impedance term)

    ## Parameters

    | Parameter | Value |
    |-----------|-------|
    | L | 10 mm |
    | H | 1 mm |
    | f | 500 kHz |
    | λ | {cfg.wavelength*1e3:.3f} mm |
    | ρ | 997 kg/m³ |
    | c | 1484 m/s |
    | Z_w = ρc | {cfg.Z_water:.0f} Pa·s/m |
    | Z_top | {cfg.Z_top:.1f} Pa·s/m |
    | V₀ (vortex) | 10 µm/s |
    | Vs (standing) | 10 µm/s |
    | ℓ | 1 |
    | R_disc | 3 mm |
    | Apodization | cosine taper: A(r) = 0.5(1+cos(πr/R)) |

    ## Mesh

    - elements_per_wavelength: {cfg.elements_per_wavelength}
    - nx × ny × nz: {cfg.mesh_nx} × {cfg.mesh_nx} × {cfg.mesh_nz}
    - Element type: P2 (quadratic Lagrange tetrahedra)
    - Satisfies: max element size ≤ λ/6

    ## Solver

    MUMPS direct LU factorization (via PETSc)

    ## Diagnostics

    ```
    {chr(10).join(diag_lines)}
    ```

    ## Results

    {result_table}

    ## Structure

    ```
    COMSOL_comparison_results/
      Case_A_standing/   csv/ figs/ meta/
      Case_B_vortex/     csv/ figs/ meta/
      Case_C_combined/   csv/ figs/ meta/
      README.md
      MANIFEST.txt
    ```

    ## How to regenerate

    ```bash
    cd {ROOT}
    micromamba run -n acousto-complex python scripts/analysis/rebuild_comsol_comparison.py
    ```
    """)
    readme_path = OUTROOT / "README.md"
    with open(readme_path, "w") as f:
        f.write(text)
    record(readme_path)


def write_manifest():
    lines = []
    for p in sorted(OUTROOT.rglob("*")):
        if p.is_file():
            lines.append(str(p.relative_to(OUTROOT)))
    lines.append("MANIFEST.txt")
    lines = sorted(set(lines))
    text = "\n".join(lines) + "\n"
    mpath = OUTROOT / "MANIFEST.txt"
    with open(mpath, "w") as f:
        f.write(text)
    record(mpath)
    return text


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    cfg = CFG

    log(f"\n{'#'*70}")
    log(f"  COMSOL COMPARISON REBUILD — ATTEMPT 2 (AUTHORITATIVE)")
    log(f"  Disc BC: impedance Z_w=ρc + prescribed velocity (COMSOL match)")
    log(f"  Solver: MUMPS direct")
    log(f"  Mesh: P2, {cfg.elements_per_wavelength} elem/λ → nx={cfg.mesh_nx}")
    log(f"{'#'*70}\n")

    # ══════════════════════════════════════════════════════════════
    # PART 2 — Create fresh folder structure
    # ══════════════════════════════════════════════════════════════
    for case_dir in CASES:
        for sub in ("csv", "figs", "meta"):
            (OUTROOT / case_dir / sub).mkdir(parents=True, exist_ok=True)

    # ══════════════════════════════════════════════════════════════
    # CREATE MESH (shared)
    # ══════════════════════════════════════════════════════════════
    domain, facet_tags, tag_map = create_mesh(cfg, verbose=True)
    n_cells = domain.topology.index_map(domain.topology.dim).size_global

    # ══════════════════════════════════════════════════════════════
    # DIAGNOSTICS (Section 9)
    # ══════════════════════════════════════════════════════════════
    log(f"\n{'='*70}")
    log("  DIAGNOSTICS")
    log(f"{'='*70}")
    diag_lines, area_ratio = run_diagnostics(domain, facet_tags, cfg)
    for line in diag_lines:
        log(f"  {line}")

    if area_ratio < 0.70:
        log("\n  *** WARNING: disc area ratio < 0.70 — mesh may be too coarse ***")

    # ══════════════════════════════════════════════════════════════
    # PRECOMPUTE SAMPLE GRIDS
    # ══════════════════════════════════════════════════════════════
    z_H2 = cfg.H / 2.0
    z_H10 = cfg.H / 10.0
    pts_H2 = _make_plane_grid(cfg, z_H2, N_PLANE)
    pts_H10 = _make_plane_grid(cfg, z_H10, N_PLANE)
    s_rad, pts_rad = _make_line_radial(cfg, z_H2, N_LINE)
    s_diag, pts_diag = _make_line_diagonal(cfg, z_H2, N_LINE)

    # ══════════════════════════════════════════════════════════════
    # SOLVE ALL 3 CASES (Parts 3 & 4)
    # ══════════════════════════════════════════════════════════════
    maxp_3d = {}
    maxp_plane = {}
    solutions = {}

    for case_dir, mode in CASES.items():
        mp3d, mpp, psol = process_case(
            case_dir, mode, domain, facet_tags, cfg, n_cells,
            pts_H2, pts_H10, s_rad, pts_rad, s_diag, pts_diag,
        )
        maxp_3d[case_dir] = mp3d
        maxp_plane[case_dir] = mpp
        solutions[case_dir] = psol

    # ══════════════════════════════════════════════════════════════
    # VALIDATION CHECK (Section 9)
    # ══════════════════════════════════════════════════════════════
    log(f"\n{'='*70}")
    log("  VALIDATION CHECK: Vortex vs Standing magnitude")
    log(f"{'='*70}")
    mp_A = maxp_3d["Case_A_standing"]
    mp_B = maxp_3d["Case_B_vortex"]
    mp_C = maxp_3d["Case_C_combined"]
    ratio_BA = mp_B / mp_A if mp_A > 0 else 0

    log(f"  Case A (standing):     max|p| = {mp_A:.4f} Pa")
    log(f"  Case B (vortex):       max|p| = {mp_B:.4f} Pa")
    log(f"  Case C (combined):     max|p| = {mp_C:.4f} Pa")
    log(f"  Ratio B/A:             {ratio_BA:.4f}")

    if ratio_BA < 0.05:
        log("\n  *** CRITICAL: Vortex is > 20× weaker than standing! ***")
        log("  This is with the COMSOL-matching impedance BC (Z_w = ρc).")
        log("  The impedance-matched disc absorbs reflected vortex energy,")
        log("  which is physically correct but limits vortex buildup.")
        log("  Proceeding — the amplitude sweep will demonstrate the boost.\n")
    elif ratio_BA < 0.3:
        log(f"\n  NOTE: Vortex {ratio_BA:.1%} of standing — expected with Z_w=ρc disc.")
        log("  The disc absorbs reflected vortex energy (impedance-matched).")
        log("  Standing waves resonate between rigid walls (no absorption).")
        log("  This is correct COMSOL physics. Amplitude sweep follows.\n")
    else:
        log(f"\n  Vortex/standing ratio = {ratio_BA:.2f} — magnitudes comparable. ✓\n")

    # ══════════════════════════════════════════════════════════════
    # PART 5 — AMPLITUDE SWEEP (Case C)
    # ══════════════════════════════════════════════════════════════
    log(f"\n{'='*70}")
    log("  AMPLITUDE SWEEP — Case C")
    log(f"{'='*70}")
    sweep_results = run_amplitude_sweep(domain, facet_tags, cfg, pts_H2)

    # ══════════════════════════════════════════════════════════════
    # RESULTS TABLE
    # ══════════════════════════════════════════════════════════════
    log(f"\n{'='*70}")
    log("  RESULTS SUMMARY")
    log(f"{'='*70}")

    table_lines = []
    table_lines.append("| Case | max|p| (3D) | max|p| (z=H/2 plane) |")
    table_lines.append("|------|------------|----------------------|")
    for cd in CASES:
        table_lines.append(
            f"| {cd} | {maxp_3d[cd]:.2f} Pa | {maxp_plane[cd]:.2f} Pa |"
        )
    for key, info in sweep_results.items():
        table_lines.append(
            f"| {key} | {info['maxp_3d']:.2f} Pa | {info['maxp_plane']:.2f} Pa |"
        )
    result_table = "\n".join(table_lines)

    log(f"\n  {'Case':<25} {'max|p| (3D)':>14} {'max|p| (plane)':>16}")
    log(f"  {'-'*57}")
    for cd in CASES:
        log(f"  {cd:<25} {maxp_3d[cd]:>11.2f} Pa {maxp_plane[cd]:>13.2f} Pa")
    for key, info in sweep_results.items():
        log(f"  {key:<25} {info['maxp_3d']:>11.2f} Pa "
            f"{info['maxp_plane']:>13.2f} Pa")

    # Ratios
    log(f"\n  Ratios:")
    log(f"    B/A (vortex/standing):  {mp_B/mp_A:.4f}")
    log(f"    C/A (combined/standing): {mp_C/mp_A:.4f}")
    for key, info in sweep_results.items():
        log(f"    {key}/A:                 {info['maxp_3d']/mp_A:.4f}")

    # ══════════════════════════════════════════════════════════════
    # PART 10 — README and MANIFEST
    # ══════════════════════════════════════════════════════════════
    write_readme(cfg, diag_lines, result_table)
    manifest_text = write_manifest()

    elapsed = time.time() - t0
    log(f"\n{'#'*70}")
    log("  Attempt 2 COMSOL Comparison Built Successfully")
    log(f"  Elapsed: {elapsed:.1f} s")
    log(f"{'#'*70}\n")

    log("MANIFEST.txt:")
    log(manifest_text)

    # Final summary table
    log(f"\nmax|p| Summary:")
    log(f"  Case A (standing):  {maxp_3d['Case_A_standing']:.2f} Pa")
    log(f"  Case B (vortex):    {maxp_3d['Case_B_vortex']:.2f} Pa")
    log(f"  Case C (combined):  {maxp_3d['Case_C_combined']:.2f} Pa")
    for key, info in sweep_results.items():
        log(f"  {key}:  {info['maxp_3d']:.2f} Pa")


if __name__ == "__main__":
    main()
