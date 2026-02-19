#!/usr/bin/env python3
"""
Golden Run — one clean results folder for the farfield_petri_cuboid model.

Runs exactly three PML-on cases:
  1) standing-only   (disk_velocity_amplitude = 0)
  2) vortex-only     (standing_velocity_amplitude = 0, plastic lens)
  3) combined        (both active)

For each case generates:
  A) Slice plots:  XZ |p|, XY |p| at disk, XY |p| at petri mid, XY arg(p) at petri mid
  B) Centerline:   |p|(z) on-axis, |p|(z) off-axis (x = center + lens_focus_offset_x)
  C) ROI metrics CSV (petri slab minus lateral PML)
  D) Correctness audit (facet counts, sigma audit, PML vs rigid comparison table)

Output:   results/farfield_<timestamp>/  (symlinked as results/farfield_latest)

Usage:
    micromamba run -n acousto-complex python scripts/experiments/golden_run.py
"""
from __future__ import annotations

import gc
import json
import csv
import sys
import time
import numpy as np
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── project imports ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from acoustweezers.experiments.farfield_petri_cuboid.config import FarFieldConfig
from acoustweezers.experiments.farfield_petri_cuboid.solve_pressure import solve_helmholtz
from acoustweezers.experiments.farfield_petri_cuboid.post import (
    slice_xy, slice_xz, centerline_z, energy_physical_vs_pml,
)

# ═════════════════════════════════════════════════════════════════════
#  Config shared by all three cases
# ═════════════════════════════════════════════════════════════════════
BASE_CFG = dict(
    Lx=6e-3,
    Ly=6e-3,
    H_under=3e-3,
    H_top=1e-3,
    frequency_hz=2.0e6,
    disk_radius=1.0e-3,
    # Balanced amplitudes: diagnostic Step 4 found d1_s10 gives best
    # interaction signal (Δ mean|p| = 0.015 Pa, interaction strength 27%).
    disk_velocity_amplitude=1e-6,
    vortex_topological_charge=1,
    standing_velocity_amplitude=10e-6,
    standing_phase_pattern="antiphase",
    standing_axis="both",
    top_bc_type="impedance",
    top_impedance_Zrel=0.001,
    # PML — always on
    pml_n_wavelengths_xy=1.0,
    pml_n_wavelengths_z=1.0,
    pml_degree=2,
    pml_sigma_max_factor=5.0,
    pml_enabled=True,
    # 3 elem/λ (coarse): ~79K DOFs, fits MUMPS in 8 GB RAM.
    # Production 5 elem/λ (348K DOFs) OOM-kills with MUMPS direct.
    elements_per_wavelength=3,
    # plastic lens
    lens_drive="plastic",
    lens_l=1,
    lens_focal_length=10e-3,
    lens_focus_offset_x=0.2e-3,
    lens_focus_offset_y=0.0,
    lens_c_lens=2700.0,
    lens_apodization="cosine_taper",
    lens_apodization_strength=1.0,
)

PETSC_OPTS = {
    # MUMPS direct — proven reliable in diagnostic Step 1.
    # GMRES+ILU diverges on the PML-Helmholtz system.
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    # MUMPS memory: increase workspace estimate, enable OOC if needed
    "mat_mumps_icntl_14": "80",     # % increase of estimated working space
    "mat_mumps_icntl_23": "4000",   # max working memory (MB)
}


# ═════════════════════════════════════════════════════════════════════
#  Helpers
# ═════════════════════════════════════════════════════════════════════

def _off_axis_centerline_z(sol, x_offset, nz=500):
    """
    |p|(z) along a vertical line at (disk_center_x + x_offset, disk_center_y).
    """
    from scipy.interpolate import NearestNDInterpolator
    coords = sol.coords
    pv = sol.p_values
    interp_re = NearestNDInterpolator(coords, np.real(pv))
    interp_im = NearestNDInterpolator(coords, np.imag(pv))
    cfg = sol.cfg
    zg = np.linspace(0, cfg.H_total, nz)
    cx = cfg.disk_center_x + x_offset
    cy = cfg.disk_center_y
    pts = np.column_stack([np.full(nz, cx), np.full(nz, cy), zg])
    pr = interp_re(pts)
    pi = interp_im(pts)
    return zg, np.abs(pr + 1j * pi)


def _roi_mask_xy(xg, yg, cfg):
    """
    Boolean mask (ny, nx) for ROI = petri slab minus lateral PML bands.
    """
    t_xy = cfg.t_pml_xy
    X, Y = np.meshgrid(xg, yg)
    inside = (
        (X >= t_xy) & (X <= cfg.Lx - t_xy) &
        (Y >= t_xy) & (Y <= cfg.Ly - t_xy)
    )
    return inside


def _roi_metrics(sol, z_val, nx=200, ny=200):
    """
    Compute ROI metrics at slice z = z_val.

    Returns dict with mean_abs_p, max_abs_p.
    """
    xg, yg, pmag, _ = slice_xy(sol, z_val, nx, ny)
    mask = _roi_mask_xy(xg, yg, sol.cfg)
    pmag_roi = pmag[mask]
    return {
        "mean_abs_p": float(np.mean(pmag_roi)),
        "max_abs_p": float(np.max(pmag_roi)),
    }


def _sigma_audit(sol):
    """
    Verify σ_z = 0 in the disk column and in the top (petri) slab.
    Returns dict of audit lines.
    """
    cfg = sol.cfg
    coords = sol.V.tabulate_dof_coordinates()
    sz = np.real(sol.sigma_z.x.array)  # σ_z is real on mesh coords
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    cx, cy = cfg.disk_center_x, cfg.disk_center_y
    R = cfg.disk_radius

    # Disk column: r <= R
    r2 = (x - cx)**2 + (y - cy)**2
    in_col = r2 <= R**2
    sz_col_max = float(np.max(np.abs(sz[in_col]))) if np.any(in_col) else 0.0

    # Top slab: z >= H_under
    in_top = z >= cfg.H_under
    sz_top_max = float(np.max(np.abs(sz[in_top]))) if np.any(in_top) else 0.0

    return {
        "sigma_z_max_disk_column": sz_col_max,
        "sigma_z_zero_in_disk_column": sz_col_max < 1e-10,
        "sigma_z_max_top_slab": sz_top_max,
        "sigma_z_zero_in_top_slab": sz_top_max < 1e-10,
    }


def _facet_count_table(sol):
    """
    Return list-of-dicts: [{"tag": ..., "name": ..., "count": ...}, ...]
    """
    tag_info = sol.tag_info
    rows = []
    for tag, name in tag_info["facet"].items():
        c = int(np.sum(sol.facet_tags.values == tag))
        rows.append({"tag": tag, "name": name, "count": c})
    for tag, name in tag_info["cell"].items():
        c = int(np.sum(sol.cell_tags.values == tag))
        if c > 0:
            rows.append({"tag": f"cell_{tag}", "name": name, "count": c})
    return rows


# ═════════════════════════════════════════════════════════════════════
#  Plotting (consistent scales)
# ═════════════════════════════════════════════════════════════════════

def _consistent_plots(results: dict, fig_dir: Path, cfg: FarFieldConfig):
    """
    Generate all requested slice / centerline plots with matched colour scales.

    results: dict  case_name -> dict with keys pmag_xz, pmag_xy_disk,
             pmag_xy_petri, pphase_xy_petri, zc_on, pc_on, zc_off, pc_off,
             and grid arrays.
    """
    case_names = list(results.keys())

    # ── determine common colour scales ────────────────────────────────
    vmax_xz = max(r["pmag_xz"].max() for r in results.values())
    vmax_xy_disk = max(r["pmag_xy_disk"].max() for r in results.values())
    vmax_xy_petri = max(r["pmag_xy_petri"].max() for r in results.values())

    # ──────────────────────────────────────────────────────────────────
    # A1) XZ mid-plane |p|
    # ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, cn in zip(axes, case_names):
        r = results[cn]
        im = ax.pcolormesh(r["xg_xz"]*1e3, r["zg_xz"]*1e3, r["pmag_xz"],
                           shading="auto", cmap="inferno", vmin=0, vmax=vmax_xz)
        ax.set_title(f"|p| XZ — {cn}")
        ax.set_xlabel("x [mm]"); ax.set_ylabel("z [mm]")
        _mark_pml(ax, cfg)
    plt.colorbar(im, ax=axes.tolist(), label="Pa", shrink=0.8)
    fig.tight_layout()
    fig.savefig(fig_dir / "slice_xz_pmag.png", dpi=150)
    plt.close(fig)

    # ──────────────────────────────────────────────────────────────────
    # A2) XY at disk plane |p|
    # ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, cn in zip(axes, case_names):
        r = results[cn]
        im = ax.pcolormesh(r["xg_xy"]*1e3, r["yg_xy"]*1e3, r["pmag_xy_disk"],
                           shading="auto", cmap="inferno", vmin=0, vmax=vmax_xy_disk)
        ax.set_title(f"|p| XY z≈0 — {cn}")
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
        ax.set_aspect("equal")
    plt.colorbar(im, ax=axes.tolist(), label="Pa", shrink=0.8)
    fig.tight_layout()
    fig.savefig(fig_dir / "slice_xy_disk_pmag.png", dpi=150)
    plt.close(fig)

    # ──────────────────────────────────────────────────────────────────
    # A3) XY at petri mid-plane |p|
    # ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, cn in zip(axes, case_names):
        r = results[cn]
        im = ax.pcolormesh(r["xg_xy"]*1e3, r["yg_xy"]*1e3, r["pmag_xy_petri"],
                           shading="auto", cmap="inferno", vmin=0, vmax=vmax_xy_petri)
        ax.set_title(f"|p| XY petri mid — {cn}")
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
        ax.set_aspect("equal")
    plt.colorbar(im, ax=axes.tolist(), label="Pa", shrink=0.8)
    fig.tight_layout()
    fig.savefig(fig_dir / "slice_xy_petri_pmag.png", dpi=150)
    plt.close(fig)

    # ──────────────────────────────────────────────────────────────────
    # A4) XY at petri mid-plane arg(p)
    # ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, cn in zip(axes, case_names):
        r = results[cn]
        im = ax.pcolormesh(r["xg_xy"]*1e3, r["yg_xy"]*1e3, r["pphase_xy_petri"],
                           shading="auto", cmap="twilight", vmin=-np.pi, vmax=np.pi)
        ax.set_title(f"arg(p) XY petri mid — {cn}")
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
        ax.set_aspect("equal")
    plt.colorbar(im, ax=axes.tolist(), label="rad", shrink=0.8)
    fig.tight_layout()
    fig.savefig(fig_dir / "slice_xy_petri_phase.png", dpi=150)
    plt.close(fig)

    # ──────────────────────────────────────────────────────────────────
    # B) Centerline plots
    # ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for cn in case_names:
        r = results[cn]
        axes[0].plot(r["zc_on"]*1e3, r["pc_on"], label=cn)
        axes[1].plot(r["zc_off"]*1e3, r["pc_off"], label=cn)
    for ax, title in zip(axes, ["|p|(z) on-axis (disk centre)",
                                 f"|p|(z) off-axis (x+{cfg.lens_focus_offset_x*1e3:.2f}mm)"]):
        ax.axvline(cfg.H_under*1e3, color="cyan", ls=":", lw=0.8, label="petri base")
        if cfg.pml_enabled:
            ax.axvline(cfg.t_pml_z*1e3, color="gray", ls=":", lw=0.7, label="PML-z top")
        ax.set_xlabel("z [mm]"); ax.set_ylabel("|p| [Pa]")
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "centerline_z.png", dpi=150)
    plt.close(fig)


def _mark_pml(ax, cfg):
    """Draw PML boundary lines on XZ plot."""
    t_xy = cfg.t_pml_xy * 1e3
    t_z = cfg.t_pml_z * 1e3
    ax.axvline(t_xy, color="w", ls="--", lw=0.7, alpha=0.6)
    ax.axvline((cfg.Lx - cfg.t_pml_xy)*1e3, color="w", ls="--", lw=0.7, alpha=0.6)
    ax.axhline(t_z, color="w", ls="--", lw=0.7, alpha=0.6)
    ax.axhline(cfg.H_under*1e3, color="cyan", ls=":", lw=0.8)


# ═════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════

def main():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path("results") / f"farfield_{stamp}"
    out_root.mkdir(parents=True, exist_ok=True)
    fig_dir = out_root / "figs"
    csv_dir = out_root / "csv"
    fig_dir.mkdir(exist_ok=True)
    csv_dir.mkdir(exist_ok=True)

    print(f"\n{'#'*70}")
    print(f"  GOLDEN RUN — farfield_petri_cuboid")
    print(f"  Output: {out_root}")
    print(f"{'#'*70}\n")

    # ── define the three cases ────────────────────────────────────────
    CASES = {
        "standing_only": dict(disk_velocity_amplitude=0.0),
        "vortex_only":   dict(standing_velocity_amplitude=0.0),
        "combined":      {},    # both active, use BASE_CFG as-is
    }

    # Reference config for geometry constants
    ref_cfg = FarFieldConfig(**BASE_CFG)
    z_disk = 0.0                          # disk sits at z = 0
    z_petri_mid = ref_cfg.H_under + ref_cfg.H_top / 2

    # Store per-case data
    results = {}        # for plotting
    roi_rows = []       # for ROI CSV
    solver_meta = {}    # for audit comparison
    facet_tables = {}   # for correctness pack

    # ── save shared config ────────────────────────────────────────────
    config_out = ref_cfg.to_dict()
    config_out["petsc_options"] = PETSC_OPTS
    config_out["cases"] = list(CASES.keys())
    with open(out_root / "config.json", "w") as f:
        json.dump(config_out, f, indent=2, default=str)

    # ══════════════════════════════════════════════════════════════════
    #  Run each case (PML ON)
    # ══════════════════════════════════════════════════════════════════
    for case_name, overrides in CASES.items():
        case_params = {**BASE_CFG, **overrides}
        cfg = FarFieldConfig(**case_params)

        print(f"\n{'='*70}")
        print(f"  CASE: {case_name}")
        print(f"{'='*70}")
        print(cfg.describe())

        sol = solve_helmholtz(cfg, verbose=True, petsc_options=PETSC_OPTS)

        # ── Extract slices ────────────────────────────────────────────
        xg_xz, zg_xz, pmag_xz, _ = slice_xz(sol, cfg.Ly / 2)
        xg_xy, yg_xy, pmag_xy_disk, _ = slice_xy(sol, z_disk)
        _, _, pmag_xy_petri, pphase_xy_petri = slice_xy(sol, z_petri_mid)

        # ── Centerlines ──────────────────────────────────────────────
        zc_on, pc_on = centerline_z(sol)
        zc_off, pc_off = _off_axis_centerline_z(sol, cfg.lens_focus_offset_x)

        results[case_name] = {
            "xg_xz": xg_xz, "zg_xz": zg_xz, "pmag_xz": pmag_xz,
            "xg_xy": xg_xy, "yg_xy": yg_xy,
            "pmag_xy_disk": pmag_xy_disk,
            "pmag_xy_petri": pmag_xy_petri,
            "pphase_xy_petri": pphase_xy_petri,
            "zc_on": zc_on, "pc_on": pc_on,
            "zc_off": zc_off, "pc_off": pc_off,
        }

        # ── ROI metrics ──────────────────────────────────────────────
        roi = _roi_metrics(sol, z_petri_mid)
        en = energy_physical_vs_pml(sol)
        roi_rows.append({
            "case": case_name,
            "mean_abs_p_roi": f"{roi['mean_abs_p']:.4f}",
            "max_abs_p_roi": f"{roi['max_abs_p']:.4f}",
            "max_abs_p_global": f"{sol.max_pressure:.4f}",
            "energy_phys": f"{en['physical']:.6e}",
            "energy_pml": f"{en['pml']:.6e}",
            "energy_ratio_pml_phys": f"{en['ratio']:.6f}",
        })

        # ── Solver/audit metadata ────────────────────────────────────
        sigma_aud = _sigma_audit(sol)
        solver_meta[case_name] = {
            "max_abs_p": sol.max_pressure,
            "dofs": sol.dofs,
            "ksp_iters": sol.ksp_iterations,
            "ksp_converged_reason": sol.ksp_converged_reason,
            "ksp_residual_norm": sol.ksp_residual_norm,
            "solver_time_s": sol.solver_time,
            "sigma_audit": sigma_aud,
            "roi": roi,
            "energy": en,
        }
        facet_tables[case_name] = _facet_count_table(sol)

        # Free PETSc memory
        del sol; gc.collect()

    # ══════════════════════════════════════════════════════════════════
    #  Also run ONE rigid-wall case (combined) for PML comparison table
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  REFERENCE: combined (rigid walls / PML OFF)")
    print(f"{'='*70}")
    cfg_rigid = FarFieldConfig(**{**BASE_CFG, "pml_enabled": False})
    sol_rigid = solve_helmholtz(cfg_rigid, verbose=True, petsc_options=PETSC_OPTS)
    roi_rigid = _roi_metrics(sol_rigid, z_petri_mid)
    en_rigid = energy_physical_vs_pml(sol_rigid)
    rigid_meta = {
        "max_abs_p": sol_rigid.max_pressure,
        "dofs": sol_rigid.dofs,
        "ksp_iters": sol_rigid.ksp_iterations,
        "ksp_converged_reason": sol_rigid.ksp_converged_reason,
        "ksp_residual_norm": sol_rigid.ksp_residual_norm,
        "solver_time_s": sol_rigid.solver_time,
        "roi": roi_rigid,
        "energy": en_rigid,
    }
    del sol_rigid; gc.collect()

    # ══════════════════════════════════════════════════════════════════
    #  C) Generate consistent plots
    # ══════════════════════════════════════════════════════════════════
    print("\n  Generating plots …")
    _consistent_plots(results, fig_dir, ref_cfg)

    # ══════════════════════════════════════════════════════════════════
    #  C) ROI metrics CSV + Δ rows
    # ══════════════════════════════════════════════════════════════════
    # Add Δ row: combined − standing_only
    stand_roi = solver_meta["standing_only"]["roi"]
    comb_roi  = solver_meta["combined"]["roi"]
    delta_row = {
        "case": "delta_combined_minus_standing",
        "mean_abs_p_roi": f"{comb_roi['mean_abs_p'] - stand_roi['mean_abs_p']:.4f}",
        "max_abs_p_roi": f"{comb_roi['max_abs_p'] - stand_roi['max_abs_p']:.4f}",
        "max_abs_p_global": "",
        "energy_phys": "",
        "energy_pml": "",
        "energy_ratio_pml_phys": "",
    }
    roi_rows.append(delta_row)

    roi_csv_path = csv_dir / "roi_metrics.csv"
    with open(roi_csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=roi_rows[0].keys())
        w.writeheader()
        w.writerows(roi_rows)

    # ══════════════════════════════════════════════════════════════════
    #  D) Correctness pack — audit_summary.csv
    # ══════════════════════════════════════════════════════════════════
    audit_rows = []

    # D1: Facet / tag counts (use combined as representative — same mesh)
    for r in facet_tables.get("combined", []):
        audit_rows.append({
            "section": "facet_cell_counts",
            "key": f"{r['tag']}_{r['name'][:30]}",
            "value": str(r["count"]),
        })

    # D2: PML sigma audit
    for case_name in CASES:
        sa = solver_meta[case_name]["sigma_audit"]
        audit_rows.append({
            "section": f"sigma_audit_{case_name}",
            "key": "sigma_z_max_disk_column",
            "value": f"{sa['sigma_z_max_disk_column']:.2e}",
        })
        audit_rows.append({
            "section": f"sigma_audit_{case_name}",
            "key": "sigma_z_zero_in_disk_column",
            "value": str(sa["sigma_z_zero_in_disk_column"]),
        })
        audit_rows.append({
            "section": f"sigma_audit_{case_name}",
            "key": "sigma_z_max_top_slab",
            "value": f"{sa['sigma_z_max_top_slab']:.2e}",
        })
        audit_rows.append({
            "section": f"sigma_audit_{case_name}",
            "key": "sigma_z_zero_in_top_slab",
            "value": str(sa["sigma_z_zero_in_top_slab"]),
        })

    # D3: PML vs rigid comparison (combined case)
    cm = solver_meta["combined"]
    KSP_REASONS = {
        1: "RTOL_NORMAL", 2: "RTOL", 3: "ATOL", 9: "ITERATING",
        -3: "DIVERGED_ITS", -4: "DIVERGED_DTOL",
        -5: "DIVERGED_BREAKDOWN", -9: "DIVERGED_NANORINF",
    }
    pml_reason = KSP_REASONS.get(cm["ksp_converged_reason"],
                                  str(cm["ksp_converged_reason"]))
    rigid_reason = KSP_REASONS.get(rigid_meta["ksp_converged_reason"],
                                    str(rigid_meta["ksp_converged_reason"]))

    for label, meta in [("pml_combined", cm), ("rigid_combined", rigid_meta)]:
        audit_rows.append({"section": "pml_vs_rigid", "key": f"{label}_max_abs_p",
                           "value": f"{meta['max_abs_p']:.4f}"})
        audit_rows.append({"section": "pml_vs_rigid", "key": f"{label}_max_abs_p_roi",
                           "value": f"{meta['roi']['max_abs_p']:.4f}"})
        audit_rows.append({"section": "pml_vs_rigid", "key": f"{label}_energy_ratio",
                           "value": f"{meta['energy']['ratio']:.6f}"})
        reason = pml_reason if "pml" in label else rigid_reason
        audit_rows.append({"section": "pml_vs_rigid", "key": f"{label}_ksp_iters",
                           "value": str(meta["ksp_iters"])})
        audit_rows.append({"section": "pml_vs_rigid", "key": f"{label}_ksp_converged",
                           "value": reason})

    audit_csv_path = csv_dir / "audit_summary.csv"
    with open(audit_csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["section", "key", "value"])
        w.writeheader()
        w.writerows(audit_rows)

    # ══════════════════════════════════════════════════════════════════
    #  INDEX.md
    # ══════════════════════════════════════════════════════════════════
    _write_index(out_root, ref_cfg, solver_meta, rigid_meta, roi_rows)

    # ── symlink ───────────────────────────────────────────────────────
    latest = Path("results") / "farfield_latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(out_root.name)
    print(f"\n  Symlink: {latest} → {out_root.name}")

    print(f"\n{'='*70}")
    print(f"  GOLDEN RUN COMPLETE")
    print(f"  {out_root}")
    print(f"{'='*70}\n")


def _write_index(out_root, cfg, solver_meta, rigid_meta, roi_rows):
    """Write INDEX.md summarising what was run and key numbers."""
    lines = []
    lines.append("# Golden Run — farfield_petri_cuboid\n")
    lines.append(f"**Date:** {datetime.now().isoformat()}\n")
    lines.append("")
    lines.append("## Configuration\n")
    lines.append(f"- Domain: {cfg.Lx*1e3:.1f} × {cfg.Ly*1e3:.1f} × {cfg.H_total*1e3:.1f} mm")
    lines.append(f"- H_under = {cfg.H_under*1e3:.1f} mm,  H_top = {cfg.H_top*1e3:.1f} mm")
    lines.append(f"- Frequency: {cfg.frequency_hz/1e6:.2f} MHz,  λ = {cfg.wavelength*1e3:.3f} mm")
    lines.append(f"- Mesh: {cfg.mesh_nx}×{cfg.mesh_ny}×{cfg.mesh_nz}  ({cfg.elements_per_wavelength} elem/λ)")
    lines.append(f"- PML: {cfg.pml_n_wavelengths_xy:.1f}λ xy, {cfg.pml_n_wavelengths_z:.1f}λ z, σ_max_factor = {cfg.pml_sigma_max_factor}")
    lines.append(f"- Lens: plastic l={cfg.lens_l}, f={cfg.lens_focal_length*1e3:.1f} mm, offset_x={cfg.lens_focus_offset_x*1e3:.2f} mm")
    lines.append(f"- Standing: V={cfg.standing_velocity_amplitude*1e6:.1f} µm/s, pattern={cfg.standing_phase_pattern}, axis={cfg.standing_axis}")
    lines.append(f"- Vortex disk: V={cfg.disk_velocity_amplitude*1e6:.1f} µm/s")
    lines.append(f"- Top BC: {cfg.top_bc_type} (Z_rel = {cfg.top_impedance_Zrel})")
    lines.append("")

    lines.append("## Cases\n")
    lines.append("| Case | max|p| [Pa] | DOFs | GMRES iters | Converged | Time [s] |")
    lines.append("|------|------------|------|-------------|-----------|----------|")
    KSP_REASONS = {
        1: "RTOL_NORMAL", 2: "RTOL", 3: "ATOL", 9: "ITERATING",
        -3: "DIV_ITS", -4: "DIV_DTOL", -5: "DIV_BRK", -9: "DIV_NAN",
    }
    for cn, m in solver_meta.items():
        reason = KSP_REASONS.get(m["ksp_converged_reason"], str(m["ksp_converged_reason"]))
        lines.append(f"| {cn} | {m['max_abs_p']:.2f} | {m['dofs']} | "
                     f"{m['ksp_iters']} | {reason} | {m['solver_time_s']:.1f} |")
    reason_r = KSP_REASONS.get(rigid_meta["ksp_converged_reason"],
                                str(rigid_meta["ksp_converged_reason"]))
    lines.append(f"| rigid (combined) | {rigid_meta['max_abs_p']:.2f} | {rigid_meta['dofs']} | "
                 f"{rigid_meta['ksp_iters']} | {reason_r} | {rigid_meta['solver_time_s']:.1f} |")
    lines.append("")

    lines.append("## ROI Metrics (petri mid-plane, excluding PML)\n")
    lines.append("| Case | mean|p| | max|p| |")
    lines.append("|------|---------|--------|")
    for r in roi_rows:
        lines.append(f"| {r['case']} | {r['mean_abs_p_roi']} | {r['max_abs_p_roi']} |")
    lines.append("")

    lines.append("## PML σ_z Audit\n")
    for cn in ["standing_only", "vortex_only", "combined"]:
        sa = solver_meta[cn]["sigma_audit"]
        lines.append(f"- **{cn}**: σ_z max in disk column = {sa['sigma_z_max_disk_column']:.2e} "
                     f"({'PASS' if sa['sigma_z_zero_in_disk_column'] else 'FAIL'}), "
                     f"σ_z max in top slab = {sa['sigma_z_max_top_slab']:.2e} "
                     f"({'PASS' if sa['sigma_z_zero_in_top_slab'] else 'FAIL'})")
    lines.append("")

    lines.append("## PML vs Rigid Comparison (combined case)\n")
    cm = solver_meta["combined"]
    lines.append(f"| Metric | PML | Rigid |")
    lines.append(f"|--------|-----|-------|")
    lines.append(f"| max|p| global | {cm['max_abs_p']:.2f} | {rigid_meta['max_abs_p']:.2f} |")
    lines.append(f"| max|p| ROI | {cm['roi']['max_abs_p']:.4f} | {rigid_meta['roi']['max_abs_p']:.4f} |")
    lines.append(f"| energy ratio (PML/phys) | {cm['energy']['ratio']:.6f} | {rigid_meta['energy']['ratio']:.6f} |")
    lines.append(f"| GMRES iters | {cm['ksp_iters']} | {rigid_meta['ksp_iters']} |")
    lines.append("")

    lines.append("## What Plots to Look At\n")
    lines.append("1. **slice_xz_pmag.png** — XZ mid-plane |p|. Compare the vortex beam shape across cases.")
    lines.append("   Look for: focused beam in vortex_only and combined; standing wave nodes in standing_only.")
    lines.append("2. **slice_xy_disk_pmag.png** — XY at z≈0. Vortex ring pattern should appear in vortex_only & combined.")
    lines.append("3. **slice_xy_petri_pmag.png** — XY at petri mid-plane. Standing wave grid pattern in standing_only & combined.")
    lines.append("4. **slice_xy_petri_phase.png** — Phase at petri mid. Helical phase in vortex cases, antiphase grid in standing.")
    lines.append("5. **centerline_z.png** — |p|(z) on-axis and off-axis. Focal region should be visible for vortex cases.")
    lines.append("")

    # Similarity analysis
    lines.append("## Notes on Between-Case Similarity\n")
    # Compare vortex petri slab vs standing petri slab
    sm = solver_meta
    v_roi_max = sm["vortex_only"]["roi"]["max_abs_p"]
    s_roi_max = sm["standing_only"]["roi"]["max_abs_p"]
    c_roi_max = sm["combined"]["roi"]["max_abs_p"]
    if v_roi_max < s_roi_max * 0.01:
        lines.append("- **Vortex signal in petri slab is negligible** relative to standing wave.")
        lines.append("  The vortex beam focuses at f=10 mm, far above H_under=3 mm. "
                     "The XY petri-mid plots for vortex_only and combined may look nearly identical "
                     "to standing_only.")
        lines.append("  **Next knob:** Reduce H_under (e.g. 1 mm) to bring the petri slab closer "
                     "to the disk, or reduce lens_focal_length to focus within the domain.")
    elif v_roi_max > s_roi_max * 100:
        lines.append("- **Standing wave is negligible** relative to the vortex in the petri slab.")
        lines.append("  **Next knob:** Increase standing_velocity_amplitude or reduce disk_velocity_amplitude.")
    else:
        lines.append("- Both sources contribute meaningfully in the petri slab ROI. Good interaction evidence.")
    lines.append("")

    with open(out_root / "INDEX.md", "w") as f:
        f.write("\n".join(lines))
    print(f"  Wrote {out_root / 'INDEX.md'}")


if __name__ == "__main__":
    main()
