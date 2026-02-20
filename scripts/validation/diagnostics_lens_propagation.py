#!/usr/bin/env python3
"""
Diagnostics — Lens Propagation
===============================

Analyses the vortex lens field propagation through the cuboid domain.

Generates:
    - Disk boundary maps (amplitude, phase, real, imag)
    - Z-stack slices (|p|, arg(p), Iz at multiple z-heights)
    - Winding number (ell) at each z-slice
    - Core ratio (on/off axis) at each z-slice
    - Power direction metric (fraction of Iz > 0)
    - PML decay metric (energy ratio)

Output saved to:
    ~/OneDrive - University of Bristol/Major Project Onedrive/
      Research/Vortex 3D visualisation/Diagnostics_LensPropagation/

Usage:
    micromamba run -n fenicsx python scripts/validation/diagnostics_lens_propagation.py
"""
from __future__ import annotations

import json
import sys
import gc
import csv
import numpy as np
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from acoustweezers.experiments.farfield_petri_cuboid.config import FarFieldConfig
from acoustweezers.experiments.farfield_petri_cuboid.solve_pressure import solve_helmholtz
from acoustweezers.experiments.farfield_petri_cuboid.post import (
    slice_xy, slice_xz, centerline_z, energy_physical_vs_pml,
)

ONEDRIVE_BASE = Path.home() / (
    "OneDrive - University of Bristol/Major Project Onedrive/"
    "Research/Vortex 3D visualisation"
)
OUTPUT_DIR = ONEDRIVE_BASE / "Diagnostics_LensPropagation"

PETSC_OPTS = {
    "ksp_type": "preonly", "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "mat_mumps_icntl_14": "80",
    "mat_mumps_icntl_23": "4000",
}


def _build_cfg(epw: int = 3) -> FarFieldConfig:
    return FarFieldConfig(
        Lx=6e-3, Ly=6e-3, H_under=3e-3, H_top=1e-3,
        frequency_hz=2.0e6, disk_radius=1.0e-3,
        disk_velocity_amplitude=1e-6,
        vortex_topological_charge=1,
        standing_velocity_amplitude=0.0,
        standing_phase_pattern="antiphase", standing_axis="both",
        top_bc_type="impedance", top_impedance_Zrel=0.001,
        pml_n_wavelengths_xy=1.0, pml_n_wavelengths_z=1.0,
        pml_degree=2, pml_sigma_max_factor=5.0, pml_enabled=True,
        elements_per_wavelength=epw,
        lens_drive="plastic", lens_l=1,
        lens_focal_length=10e-3,
        lens_focus_offset_x=0.2e-3, lens_focus_offset_y=0.0,
        lens_c_lens=2700.0, lens_apodization="cosine_taper",
        lens_apodization_strength=1.0,
    )


def _compute_Iz_slice(sol, z_val, nx=200, ny=200):
    """Approximate Iz at a z-slice using finite difference."""
    from scipy.interpolate import NearestNDInterpolator

    coords = sol.coords
    p_vals = sol.p_values
    cfg = sol.cfg
    omega, rho = cfg.omega, cfg.rho

    xg = np.linspace(0, cfg.Lx, nx)
    yg = np.linspace(0, cfg.Ly, ny)
    XX, YY = np.meshgrid(xg, yg)

    pts = np.column_stack([XX.ravel(), YY.ravel(), np.full(nx*ny, z_val)])
    interp_re = NearestNDInterpolator(coords, np.real(p_vals))
    interp_im = NearestNDInterpolator(coords, np.imag(p_vals))
    p_re = interp_re(pts)
    p_im = interp_im(pts)
    p_grid = p_re + 1j * p_im

    # dP/dz via finite difference
    eps = 5e-5
    pts_up = pts.copy(); pts_up[:, 2] += eps
    pts_dn = pts.copy(); pts_dn[:, 2] -= eps
    p_up = interp_re(pts_up) + 1j * interp_im(pts_up)
    p_dn = interp_re(pts_dn) + 1j * interp_im(pts_dn)
    dp_dz = (p_up - p_dn) / (2 * eps)

    vz = dp_dz / (1j * omega * rho)
    Iz = 0.5 * np.real(p_grid * np.conj(vz))

    return xg, yg, Iz.reshape(ny, nx)


def _winding_at_z(sol, z_val, nx=200, ny=200):
    """Measure phase winding ell at a given z-height."""
    xg, yg, pmag, pphase = slice_xy(sol, z_val, nx, ny)
    cfg = sol.cfg
    cx, cy = cfg.disk_center_x, cfg.disk_center_y

    ring_r = cfg.disk_radius * 0.5
    n_ring = 360
    theta_ring = np.linspace(0, 2*np.pi, n_ring, endpoint=False)
    x_ring = cx + ring_r * np.cos(theta_ring)
    y_ring = cy + ring_r * np.sin(theta_ring)

    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator(
        (yg, xg), pphase, method="nearest", bounds_error=False, fill_value=0)
    pts = np.column_stack([y_ring, x_ring])
    phase_ring = interp(pts)

    dphi = np.diff(phase_ring)
    dphi = (dphi + np.pi) % (2*np.pi) - np.pi
    total = np.sum(dphi)
    return float(total / (2*np.pi))


def _core_ratio_at_z(sol, z_val, nx=200, ny=200):
    """On-axis / off-axis |p| ratio."""
    xg, yg, pmag, _ = slice_xy(sol, z_val, nx, ny)
    cfg = sol.cfg
    ci = np.argmin(np.abs(xg - cfg.disk_center_x))
    cj = np.argmin(np.abs(yg - cfg.disk_center_y))
    on = pmag[cj, ci]

    ring_r_px = max(1, int(0.5 * cfg.disk_radius / (xg[1]-xg[0])))
    vals = []
    for a in np.linspace(0, 2*np.pi, 36, endpoint=False):
        ri = cj + int(ring_r_px * np.sin(a))
        rci = ci + int(ring_r_px * np.cos(a))
        if 0 <= ri < len(yg) and 0 <= rci < len(xg):
            vals.append(pmag[ri, rci])
    off = np.mean(vals) if vals else 1.0
    return float(on / off) if off > 0 else float("inf")


def _disk_boundary_maps(sol, out_dir: Path):
    """Plot amplitude, phase, real, imag on the disk boundary DOFs."""
    from dolfinx import fem
    from acoustweezers.experiments.farfield_petri_cuboid.mesh import TAG_BOTTOM_DISK

    V = sol.V
    fdim = sol.domain.topology.dim - 1
    disk_facets = sol.facet_tags.indices[sol.facet_tags.values == TAG_BOTTOM_DISK]
    disk_dofs = fem.locate_dofs_topological(V, fdim, disk_facets)
    coords = V.tabulate_dof_coordinates()
    p_disk = sol.p_values[disk_dofs]
    x_d = coords[disk_dofs, 0] * 1e3
    y_d = coords[disk_dofs, 1] * 1e3

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fields = [
        (np.abs(p_disk), "Amplitude |p|", "viridis"),
        (np.angle(p_disk), "Phase arg(p)", "twilight"),
        (np.real(p_disk), "Real(p)", "RdBu_r"),
        (np.imag(p_disk), "Imag(p)", "RdBu_r"),
    ]
    for ax, (vals, title, cmap) in zip(axes.flat, fields):
        sc = ax.scatter(x_d, y_d, c=vals, s=4, cmap=cmap)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
        plt.colorbar(sc, ax=ax)
    fig.tight_layout()
    path = out_dir / "disk_boundary_maps.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path.resolve()}")


def main():
    print(f"\n{'#'*70}")
    print(f"  DIAGNOSTICS — LENS PROPAGATION")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'#'*70}\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Solve ─────────────────────────────────────────────────────────
    for epw in [5, 3]:
        cfg = _build_cfg(epw)
        try:
            sol = solve_helmholtz(cfg, verbose=True, petsc_options=PETSC_OPTS)
            break
        except Exception as e:
            print(f"  {epw} elem/lambda failed: {e}")
            gc.collect()
    else:
        raise RuntimeError("All mesh resolutions failed")

    cfg = sol.cfg

    # ── Disk boundary maps ────────────────────────────────────────────
    _disk_boundary_maps(sol, OUTPUT_DIR)

    # ── Z-stack slices ────────────────────────────────────────────────
    z_heights = np.linspace(cfg.t_pml_z + 0.1e-3, cfg.H_total - 0.1e-3, 8)
    z_heights_mm = z_heights * 1e3

    stack_data = []
    for zi, z_val in enumerate(z_heights):
        xg, yg, pmag, pphase = slice_xy(sol, z_val)
        _, _, Iz_map = _compute_Iz_slice(sol, z_val)
        ell_meas = _winding_at_z(sol, z_val)
        cr = _core_ratio_at_z(sol, z_val)

        # Power direction: fraction of Iz > 0 in physical ROI
        t_xy = cfg.t_pml_xy
        X, Y = np.meshgrid(xg, yg)
        roi = (X >= t_xy) & (X <= cfg.Lx - t_xy) & (Y >= t_xy) & (Y <= cfg.Ly - t_xy)
        frac_up = float(np.sum(Iz_map[roi] > 0) / max(np.sum(roi), 1))

        stack_data.append({
            "z_mm": float(z_val * 1e3),
            "ell_measured": ell_meas,
            "core_ratio": cr,
            "frac_Iz_upward": frac_up,
            "mean_pmag_roi": float(np.mean(pmag[roi])),
            "max_pmag_roi": float(np.max(pmag[roi])),
        })

        # Save individual slice plots
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        im0 = axes[0].pcolormesh(xg*1e3, yg*1e3, pmag, cmap="inferno", shading="auto")
        axes[0].set_title(f"|p| z={z_val*1e3:.1f}mm"); axes[0].set_aspect("equal")
        plt.colorbar(im0, ax=axes[0], label="Pa")

        im1 = axes[1].pcolormesh(xg*1e3, yg*1e3, pphase, cmap="twilight",
                                 shading="auto", vmin=-np.pi, vmax=np.pi)
        axes[1].set_title(f"arg(p) z={z_val*1e3:.1f}mm"); axes[1].set_aspect("equal")
        plt.colorbar(im1, ax=axes[1], label="rad")

        vmax_Iz = max(np.abs(Iz_map).max(), 1e-20)
        im2 = axes[2].pcolormesh(xg*1e3, yg*1e3, Iz_map, cmap="coolwarm",
                                 shading="auto", vmin=-vmax_Iz, vmax=vmax_Iz)
        axes[2].set_title(f"Iz z={z_val*1e3:.1f}mm"); axes[2].set_aspect("equal")
        plt.colorbar(im2, ax=axes[2], label="W/m^2")

        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"zstack_{zi:02d}_z{z_val*1e3:.1f}mm.png", dpi=150)
        plt.close(fig)

    # ── Summary plots ─────────────────────────────────────────────────
    zz = [d["z_mm"] for d in stack_data]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(zz, [d["ell_measured"] for d in stack_data], "o-")
    axes[0, 0].axhline(1.0, color="gray", ls="--", lw=0.7)
    axes[0, 0].set_ylabel("ell (measured)"); axes[0, 0].set_title("Winding Number vs z")

    axes[0, 1].plot(zz, [d["core_ratio"] for d in stack_data], "o-")
    axes[0, 1].axhline(0.3, color="red", ls="--", lw=0.7, label="threshold")
    axes[0, 1].set_ylabel("core ratio"); axes[0, 1].set_title("Core Ratio vs z")
    axes[0, 1].legend()

    axes[1, 0].plot(zz, [d["frac_Iz_upward"] for d in stack_data], "o-")
    axes[1, 0].set_ylabel("frac Iz > 0"); axes[1, 0].set_title("Power Direction vs z")

    axes[1, 1].plot(zz, [d["mean_pmag_roi"] for d in stack_data], "o-", label="mean")
    axes[1, 1].plot(zz, [d["max_pmag_roi"] for d in stack_data], "s--", label="max")
    axes[1, 1].set_ylabel("|p| [Pa]"); axes[1, 1].set_title("|p| in ROI vs z")
    axes[1, 1].legend()

    for ax in axes.flat:
        ax.set_xlabel("z [mm]")
        ax.axvline(cfg.H_under*1e3, color="cyan", ls=":", lw=0.7, label="petri base")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "propagation_summary.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {OUTPUT_DIR / 'propagation_summary.png'}")

    # ── PML decay ─────────────────────────────────────────────────────
    en = energy_physical_vs_pml(sol)
    pml_metric = {
        "energy_physical": float(en["physical"]),
        "energy_pml": float(en["pml"]),
        "ratio": float(en["ratio"]),
    }

    # ── CSV output ────────────────────────────────────────────────────
    csv_path = OUTPUT_DIR / "zstack_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=stack_data[0].keys())
        w.writeheader()
        w.writerows(stack_data)
    print(f"  Saved: {csv_path.resolve()}")

    # ── REPORT.md ─────────────────────────────────────────────────────
    lines = [
        "# Diagnostics — Lens Propagation", "",
        f"**Date:** {datetime.now().isoformat()}", "",
        "## Z-Stack Metrics", "",
        "| z [mm] | ell | core_ratio | frac_Iz_up | mean|p| | max|p| |",
        "|--------|-----|------------|------------|---------|--------|",
    ]
    for d in stack_data:
        lines.append(
            f"| {d['z_mm']:.1f} | {d['ell_measured']:.2f} | "
            f"{d['core_ratio']:.4f} | {d['frac_Iz_upward']:.3f} | "
            f"{d['mean_pmag_roi']:.4f} | {d['max_pmag_roi']:.4f} |"
        )
    lines.extend([
        "", "## PML Decay", "",
        f"- Energy physical: {pml_metric['energy_physical']:.4e}",
        f"- Energy PML: {pml_metric['energy_pml']:.4e}",
        f"- Ratio: {pml_metric['ratio']:.4f}", "",
        "## Files", "",
        "- `disk_boundary_maps.png` — amplitude, phase, real, imag on disk",
        "- `zstack_*.png` — |p|, arg(p), Iz at 8 z-heights",
        "- `propagation_summary.png` — ell, core_ratio, power direction vs z",
        "- `zstack_metrics.csv` — tabulated metrics",
        "",
    ])
    (OUTPUT_DIR / "REPORT.md").write_text("\n".join(lines))
    print(f"  Saved: {OUTPUT_DIR / 'REPORT.md'}")

    # ── results.json ──────────────────────────────────────────────────
    results = {
        "date": datetime.now().isoformat(),
        "dofs": sol.dofs,
        "max_pressure_Pa": sol.max_pressure,
        "pml_decay": pml_metric,
        "zstack": stack_data,
    }
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"  LENS PROPAGATION DIAGNOSTICS COMPLETE")
    print(f"  Output: {OUTPUT_DIR.resolve()}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
