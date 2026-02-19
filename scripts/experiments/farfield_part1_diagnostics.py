#!/usr/bin/env python3
"""
Part 1 diagnostics for far-field petri cuboid PML experiment.

Confirms / disproves suspected issues S1–S4:
  S1) Is PML accidentally affecting the TOP region?
  S2) Is the bottom disk source being damped by PML?
  S3) GMRES+ILU solver reliability / convergence
  S4) Top boundary impedance vs pressure-release comparison

Usage:
    micromamba run -n acousto-complex python scripts/experiments/farfield_part1_diagnostics.py

Author: Acousto-Tweezers Project
Date: 2026-02-16
"""

from __future__ import annotations

import json
import sys
import time
import csv
import gc
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from acoustweezers.experiments.farfield_petri_cuboid.config import FarFieldConfig
from acoustweezers.experiments.farfield_petri_cuboid.solve_pressure import (
    solve_helmholtz, _build_sigma_functions, PressureSolution,
)
from acoustweezers.experiments.farfield_petri_cuboid.mesh import (
    create_mesh, CELL_PHYSICAL, CELL_PML_X, CELL_PML_Y, CELL_PML_Z,
    CELL_PML_XY, CELL_PML_XZ, CELL_PML_YZ, CELL_PML_XYZ,
    TAG_BOTTOM_DISK, TAG_TOP,
)
from acoustweezers.experiments.farfield_petri_cuboid.post import centerline_z


def make_cfg():
    """Standard test config (same as driver script)."""
    return FarFieldConfig(
        Lx=6e-3, Ly=6e-3,
        H_under=3e-3, H_top=1e-3,
        frequency_hz=2.0e6,
        disk_radius=1.0e-3,
        disk_velocity_amplitude=10e-6,
        vortex_topological_charge=1,
        standing_velocity_amplitude=1e-6,
        standing_phase_pattern="antiphase",
        standing_axis="both",
        top_bc_type="impedance",
        top_impedance_Zrel=0.001,
        pml_n_wavelengths_xy=1.0,
        pml_n_wavelengths_z=1.0,
        pml_degree=2,
        pml_sigma_max_factor=5.0,
        pml_enabled=True,
        elements_per_wavelength=5,
    )


# =====================================================================
# S1: PML top region check
# =====================================================================
def diagnostic_s1(cfg, out_dir):
    """Check that sigma fields are zero near the top face and in the petri slab."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from dolfinx import fem

    print("\n" + "=" * 70)
    print("S1: PML TOP REGION CHECK")
    print("=" * 70)

    domain, facet_tags, cell_tags, tag_info = create_mesh(cfg, verbose=False)
    V = fem.functionspace(domain, ("Lagrange", 2))
    sigma_x, sigma_y, sigma_z = _build_sigma_functions(V, cfg)

    coords = V.tabulate_dof_coordinates()
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    sx = np.real(sigma_x.x.array)
    sy = np.real(sigma_y.x.array)
    sz = np.real(sigma_z.x.array)

    H_under = cfg.H_under
    H_total = cfg.H_total
    tol = 1e-6

    petri_mask = (z >= H_under - tol)
    top_mask = (z > H_total - 0.1e-3)
    full_mask = np.ones(len(z), dtype=bool)

    regions = {"petri_slab": petri_mask, "near_top": top_mask, "full_domain": full_mask}

    csv_rows = []
    for name, mask in regions.items():
        for label, arr in [("sigma_x", sx), ("sigma_y", sy), ("sigma_z", sz)]:
            vals = arr[mask]
            nz_count = int(np.sum(np.abs(vals) > 1e-10))
            row = {"region": name, "field": label, "n_dofs": int(np.sum(mask)),
                   "nonzero_count": nz_count, "min": f"{vals.min():.6e}",
                   "max": f"{vals.max():.6e}", "mean": f"{np.mean(vals):.6e}"}
            csv_rows.append(row)
            print(f"  {name:15s} {label}: nonzero={nz_count:6d}  "
                  f"min={vals.min():.3e}  max={vals.max():.3e}")

    csv_path = out_dir / "pml_sigma_summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        w.writeheader()
        w.writerows(csv_rows)
    print(f"  -> Saved: {csv_path}")

    # sigma along centerline
    cx, cy = cfg.disk_center_x, cfg.disk_center_y
    r_from_center = np.sqrt((x - cx)**2 + (y - cy)**2)
    cl_mask = r_from_center < 0.3e-3
    if np.sum(cl_mask) > 2:
        z_cl = z[cl_mask]; sx_cl = sx[cl_mask]; sy_cl = sy[cl_mask]; sz_cl = sz[cl_mask]
        order = np.argsort(z_cl)
        z_cl, sx_cl, sy_cl, sz_cl = z_cl[order], sx_cl[order], sy_cl[order], sz_cl[order]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(z_cl * 1e3, sx_cl, "r-", alpha=0.7, label="sigma_x")
        ax.plot(z_cl * 1e3, sy_cl, "g--", alpha=0.7, label="sigma_y")
        ax.plot(z_cl * 1e3, sz_cl, "b-", lw=2, label="sigma_z")
        ax.axvline(cfg.t_pml_z * 1e3, color="gray", ls=":", lw=0.7, label="PML-z limit")
        ax.axvline(H_under * 1e3, color="cyan", ls=":", lw=0.8, label="petri base")
        ax.set_xlabel("z [mm]"); ax.set_ylabel("sigma [1/s]")
        ax.set_title("sigma along centerline (x~Lx/2, y~Ly/2)")
        ax.legend(); fig.tight_layout()
        fig.savefig(out_dir / "sigma_centerline.png", dpi=150); plt.close(fig)
        print(f"  -> Saved: {out_dir / 'sigma_centerline.png'}")

    petri_sz_nz = int(np.sum(np.abs(sz[petri_mask]) > 1e-10))
    top_sz_nz = int(np.sum(np.abs(sz[top_mask]) > 1e-10))
    above_pml = z > cfg.t_pml_z + tol
    sz_above_nz = int(np.sum(np.abs(sz[above_pml]) > 1e-10))
    print(f"\n  VERDICT S1:")
    print(f"    sigma_z near top: {top_sz_nz} nonzero DOFs -> "
          f"{'PASS (zero)' if top_sz_nz == 0 else 'FAIL'}")
    print(f"    sigma_z in petri slab: {petri_sz_nz} nonzero DOFs")
    print(f"    sigma_z above z=t_pml_z: nonzero={sz_above_nz}")

    del domain, facet_tags, cell_tags, V, sigma_x, sigma_y, sigma_z
    gc.collect()
    return csv_rows


# =====================================================================
# S2: Disk column PML exclusion check
# =====================================================================
def diagnostic_s2(cfg, out_dir):
    """Check that disk column is excluded from bottom PML."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from dolfinx import fem
    from scipy.interpolate import NearestNDInterpolator

    print("\n" + "=" * 70)
    print("S2: BOTTOM DISK PML EXCLUSION CHECK")
    print("=" * 70)

    domain, facet_tags, cell_tags, tag_info = create_mesh(cfg, verbose=False)
    V = fem.functionspace(domain, ("Lagrange", 2))
    sigma_x, sigma_y, sigma_z = _build_sigma_functions(V, cfg)

    coords = V.tabulate_dof_coordinates()
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    sz = np.real(sigma_z.x.array)
    cx, cy = cfg.disk_center_x, cfg.disk_center_y
    R = cfg.disk_radius; t_z = cfg.t_pml_z

    r2 = (x - cx)**2 + (y - cy)**2
    disk_col_mask = (r2 <= R**2) & (z < t_z + 1e-6)
    disk_col_sz = sz[disk_col_mask]
    bottom_mask = z < 0.1e-3
    outside_disk_bottom = bottom_mask & (r2 > R**2)

    print(f"  Disk column (r<=R, z<t_pml_z):")
    print(f"    DOFs: {int(np.sum(disk_col_mask))}")
    print(f"    sigma_z nonzero: {int(np.sum(np.abs(disk_col_sz) > 1e-10))}")
    print(f"    sigma_z max: {disk_col_sz.max():.3e}")
    if np.sum(outside_disk_bottom) > 0:
        sz_out = sz[outside_disk_bottom]
        print(f"  Bottom-outside-disk (r>R, z<0.1mm):")
        print(f"    DOFs: {int(np.sum(outside_disk_bottom))}")
        print(f"    sigma_z nonzero: {int(np.sum(np.abs(sz_out) > 1e-10))}")
        print(f"    sigma_z max: {sz_out.max():.3e}")

    # 2D map
    bottom_all = z < 0.2e-3
    interp = NearestNDInterpolator(
        np.column_stack([x[bottom_all], y[bottom_all]]), sz[bottom_all])
    ng = 200
    xg = np.linspace(0, cfg.Lx, ng); yg = np.linspace(0, cfg.Ly, ng)
    X, Y = np.meshgrid(xg, yg)
    SZ = interp(np.column_stack([X.ravel(), Y.ravel()])).reshape(X.shape)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.pcolormesh(xg * 1e3, yg * 1e3, SZ, shading="auto", cmap="Reds")
    tc = np.linspace(0, 2 * np.pi, 100)
    ax.plot(cx * 1e3 + R * 1e3 * np.cos(tc), cy * 1e3 + R * 1e3 * np.sin(tc),
            "k--", lw=1.5, label="disk edge")
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    ax.set_title("sigma_z on bottom plane -- disk exclusion")
    ax.set_aspect("equal"); ax.legend()
    plt.colorbar(im, ax=ax, label="sigma_z [1/s]")
    fig.tight_layout()
    fig.savefig(out_dir / "pml_bottom_mask.png", dpi=150); plt.close(fig)
    print(f"  -> Saved: {out_dir / 'pml_bottom_mask.png'}")

    disk_ok = int(np.sum(np.abs(disk_col_sz) > 1e-10)) == 0
    print(f"\n  VERDICT S2: disk column sigma_z = 0 -> "
          f"{'PASS' if disk_ok else 'FAIL'}")

    del domain, facet_tags, cell_tags, V, sigma_x, sigma_y, sigma_z
    gc.collect()


# =====================================================================
# S3: GMRES solver convergence
# =====================================================================
def diagnostic_s3(cfg, out_dir):
    """Test solver convergence with two tolerance levels using GMRES(200)."""
    print("\n" + "=" * 70)
    print("S3: GMRES(200)+ILU SOLVER RELIABILITY")
    print("=" * 70)

    results = []
    for rtol_str, rtol in [("1e-5", 1e-5), ("1e-7", 1e-7)]:
        print(f"\n  Solving with rtol = {rtol_str} ...")
        petsc_opts = {
            "ksp_type": "gmres",
            "ksp_rtol": rtol,
            "ksp_atol": 1e-50,
            "ksp_max_it": 5000,
            "ksp_gmres_restart": 200,
            "pc_type": "ilu",
            "ksp_converged_reason": "",
        }
        t0 = time.time()
        sol = solve_helmholtz(cfg, verbose=False, petsc_options=petsc_opts)
        t_solve = time.time() - t0
        max_p = sol.max_pressure
        zc, pc = centerline_z(sol, nz=200)
        cl_max = float(pc.max())

        print(f"    solver: GMRES(200)+ILU,  rtol={rtol}")
        print(f"    DOFs: {sol.dofs},  time: {t_solve:.1f} s")
        print(f"    max|p|: {max_p:.4f} Pa,  centerline max: {cl_max:.4f} Pa")

        results.append({"rtol": rtol_str, "solver": "GMRES(200)", "pc": "ILU",
                         "dofs": sol.dofs, "solver_time_s": f"{t_solve:.1f}",
                         "max_p_Pa": f"{max_p:.4f}", "centerline_max_Pa": f"{cl_max:.4f}"})
        del sol; gc.collect()

    mp1 = float(results[0]["max_p_Pa"]); mp2 = float(results[1]["max_p_Pa"])
    cl1 = float(results[0]["centerline_max_Pa"]); cl2 = float(results[1]["centerline_max_Pa"])
    pct_maxp = abs(mp1 - mp2) / max(mp1, mp2, 1e-30) * 100
    pct_cl = abs(cl1 - cl2) / max(cl1, cl2, 1e-30) * 100

    print(f"\n  TOLERANCE SENSITIVITY:")
    print(f"    max|p| change: {pct_maxp:.2f}%  ({mp1:.4f} -> {mp2:.4f})")
    print(f"    centerline max change: {pct_cl:.2f}%  ({cl1:.4f} -> {cl2:.4f})")
    verdict = "PASS (<2%)" if max(pct_maxp, pct_cl) < 2.0 else "MARGINAL"
    print(f"    VERDICT S3: {verdict}")

    csv_path = out_dir / "solver_convergence.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    print(f"  -> Saved: {csv_path}")
    return pct_maxp, pct_cl


# =====================================================================
# S4: Top BC impedance vs pressure-release
# =====================================================================
def diagnostic_s4(cfg, out_dir):
    """Compare impedance top BC vs pressure-release (Dirichlet p=0)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from acoustweezers.experiments.farfield_petri_cuboid.post import slice_xy
    from dataclasses import replace

    print("\n" + "=" * 70)
    print("S4: TOP BC IMPEDANCE vs PRESSURE-RELEASE")
    print("=" * 70)

    cfg_imp = replace(cfg, top_bc_type="impedance")
    print(f"\n  Solve A: top_bc = impedance (Z_rel = {cfg_imp.top_impedance_Zrel})")
    sol_imp = solve_helmholtz(cfg_imp, verbose=True)
    zc_imp, pc_imp = centerline_z(sol_imp, nz=300)
    max_p_imp = sol_imp.max_pressure
    _, _, ptop_imp, _ = slice_xy(sol_imp, cfg.H_total - 0.01e-3, nx=100, ny=100)
    del sol_imp; gc.collect()

    cfg_pr = replace(cfg, top_bc_type="dirichlet")
    print(f"\n  Solve B: top_bc = pressure_release (p=0)")
    sol_pr = solve_helmholtz(cfg_pr, verbose=True)
    zc_pr, pc_pr = centerline_z(sol_pr, nz=300)
    max_p_pr = sol_pr.max_pressure
    _, _, ptop_pr, _ = slice_xy(sol_pr, cfg.H_total - 0.01e-3, nx=100, ny=100)
    del sol_pr; gc.collect()

    print(f"\n  COMPARISON:")
    print(f"    Impedance:        max|p| = {max_p_imp:.2f} Pa,  cl_max = {pc_imp.max():.2f} Pa")
    print(f"    Pressure-release: max|p| = {max_p_pr:.2f} Pa,  cl_max = {pc_pr.max():.2f} Pa")
    print(f"    Top-plane max|p|:  imp={ptop_imp.max():.4f}  pr={ptop_pr.max():.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(zc_imp * 1e3, pc_imp, "b-", label="Impedance")
    axes[0].plot(zc_pr * 1e3, pc_pr, "r--", label="Pressure-release")
    axes[0].axvline(cfg.H_under * 1e3, color="cyan", ls=":", lw=0.8, label="petri base")
    axes[0].set_xlabel("z [mm]"); axes[0].set_ylabel("|p| [Pa]")
    axes[0].set_title("Centerline |p|"); axes[0].legend(fontsize=8)

    xg = np.linspace(0, cfg.Lx, 100); yg = np.linspace(0, cfg.Ly, 100)
    vmax = max(ptop_imp.max(), ptop_pr.max())
    im1 = axes[1].pcolormesh(xg * 1e3, yg * 1e3, ptop_imp, shading="auto",
                              cmap="inferno", vmin=0, vmax=vmax)
    axes[1].set_title("Top plane -- Impedance"); axes[1].set_aspect("equal")
    axes[1].set_xlabel("x [mm]"); axes[1].set_ylabel("y [mm]")
    plt.colorbar(im1, ax=axes[1], label="Pa")
    im2 = axes[2].pcolormesh(xg * 1e3, yg * 1e3, ptop_pr, shading="auto",
                              cmap="inferno", vmin=0, vmax=vmax)
    axes[2].set_title("Top plane -- Pressure-release"); axes[2].set_aspect("equal")
    axes[2].set_xlabel("x [mm]"); axes[2].set_ylabel("y [mm]")
    plt.colorbar(im2, ax=axes[2], label="Pa")
    fig.suptitle("S4: Top BC comparison", fontsize=12); fig.tight_layout()
    fig.savefig(out_dir / "top_bc_comparison.png", dpi=150); plt.close(fig)
    print(f"  -> Saved: {out_dir / 'top_bc_comparison.png'}")

    csv_path = out_dir / "top_bc_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["top_bc", "max_p_Pa", "centerline_max_Pa", "top_plane_max_Pa"])
        w.writeheader()
        w.writerow({"top_bc": f"impedance(Z_rel={cfg.top_impedance_Zrel})",
                     "max_p_Pa": f"{max_p_imp:.2f}", "centerline_max_Pa": f"{pc_imp.max():.2f}",
                     "top_plane_max_Pa": f"{ptop_imp.max():.4f}"})
        w.writerow({"top_bc": "pressure_release(p=0)",
                     "max_p_Pa": f"{max_p_pr:.2f}", "centerline_max_Pa": f"{pc_pr.max():.2f}",
                     "top_plane_max_Pa": f"{ptop_pr.max():.4f}"})
    print(f"  -> Saved: {csv_path}")

    same = abs(max_p_imp - max_p_pr) / max(max_p_imp, max_p_pr) < 0.01
    print(f"\n  VERDICT S4: {'NOT toggleable (FAIL)' if same else 'toggleable (PASS)'}")


# =====================================================================
# MAIN
# =====================================================================
def main():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("results") / f"farfield_part1_diag_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#' * 70}")
    print(f"  PART 1 DIAGNOSTICS -- FAR-FIELD PETRI CUBOID")
    print(f"  Output: {out_dir}")
    print(f"{'#' * 70}")

    cfg = make_cfg()
    diagnostic_s1(cfg, out_dir)
    diagnostic_s2(cfg, out_dir)
    diagnostic_s3(cfg, out_dir)
    diagnostic_s4(cfg, out_dir)

    print(f"\n{'#' * 70}")
    print(f"  DIAGNOSTICS COMPLETE -- output: {out_dir}")
    print(f"{'#' * 70}\n")

    latest = Path("results") / "farfield_part1_diag_latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(out_dir.name)
    print(f"  Symlink: {latest} -> {out_dir.name}")
    return out_dir


if __name__ == "__main__":
    main()
