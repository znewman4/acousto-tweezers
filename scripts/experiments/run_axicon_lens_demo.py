#!/usr/bin/env python3
"""
Axicon (Bessel-like) Vortex Lens Demo
======================================

Runs the far-field simulation with an axicon lens instead of the
default converging plastic lens.  The axicon imparts a radial phase
φ(r,θ) = ℓθ + k_r · r  (k_r = k0 sin α), producing a non-diffracting
Bessel-like vortex beam.

Standing wave is OFF so the vortex structure is visible in isolation.

Three sub-cases are exported:

  1. axicon_15deg  — default axicon half-angle α = 15°
  2. axicon_10deg  — shallower cone → wider Bessel core
  3. plastic_ref   — reference plastic converging lens (same ℓ)

Outputs go to:
    ~/OneDrive - University of Bristol/Major Project Onedrive/
      Research/Vortex 3D visualisation/AxiconLensDemo/

Usage:
    micromamba run -n fenicsx python scripts/experiments/run_axicon_lens_demo.py
"""
from __future__ import annotations

import gc
import json
import sys
import time
import numpy as np
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── project path ──────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from acoustweezers.experiments.farfield_petri_cuboid.config import FarFieldConfig
from acoustweezers.experiments.farfield_petri_cuboid.solve_pressure import solve_helmholtz
from acoustweezers.experiments.farfield_petri_cuboid.post import (
    slice_xy, slice_xz, centerline_z, energy_physical_vs_pml,
)

# ── output path (OneDrive) ────────────────────────────────────────────
ONEDRIVE_BASE = Path.home() / (
    "OneDrive - University of Bristol/Major Project Onedrive/"
    "Research/Vortex 3D visualisation"
)
OUTPUT_DIR = ONEDRIVE_BASE / "AxiconLensDemo"

# ── solver config ─────────────────────────────────────────────────────
PETSC_OPTS = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "mat_mumps_icntl_14": "80",
    "mat_mumps_icntl_23": "4000",
}

# Try 5 elem/λ first; fall back to 3 if MUMPS OOM
ELEM_PER_WAVELENGTH_TARGETS = [5, 3]


# ─────────────────────────────────────────────────────────────────────
# Config builders
# ─────────────────────────────────────────────────────────────────────
_COMMON = dict(
    Lx=6e-3, Ly=6e-3, H_under=3e-3, H_top=1e-3,
    frequency_hz=2.0e6,
    disk_radius=1.0e-3,
    disk_velocity_amplitude=1e-6,
    vortex_topological_charge=1,
    standing_velocity_amplitude=0.0,      # standing OFF
    standing_phase_pattern="antiphase",
    standing_axis="both",
    top_bc_type="impedance",
    top_impedance_Zrel=0.001,
    pml_n_wavelengths_xy=1.0,
    pml_n_wavelengths_z=1.0,
    pml_degree=2,
    pml_sigma_max_factor=5.0,
    pml_enabled=True,
    lens_l=1,
    lens_apodization="cosine_taper",
    lens_apodization_strength=1.0,
)


CASES = {
    "axicon_15deg": dict(
        lens_drive="axicon",
        lens_axicon_angle_deg=15.0,
    ),
    "axicon_10deg": dict(
        lens_drive="axicon",
        lens_axicon_angle_deg=10.0,
    ),
    "plastic_ref": dict(
        lens_drive="plastic",
        lens_focal_length=10e-3,
        lens_focus_offset_x=0.2e-3,
        lens_focus_offset_y=0.0,
        lens_c_lens=2700.0,
    ),
}


def _build_cfg(case_overrides: dict, epw: int) -> FarFieldConfig:
    kw = {**_COMMON, **case_overrides, "elements_per_wavelength": epw}
    return FarFieldConfig(**kw)


# ─────────────────────────────────────────────────────────────────────
# VTU export (meshio point cloud)
# ─────────────────────────────────────────────────────────────────────
def _export_vtu(sol, out_dir: Path, tag: str) -> Path:
    import meshio

    out_dir.mkdir(parents=True, exist_ok=True)
    coords = sol.coords
    p_vals = sol.p_values

    point_data = {
        "p_re":  np.real(p_vals).astype(np.float64),
        "p_im":  np.imag(p_vals).astype(np.float64),
        "p_abs": np.abs(p_vals).astype(np.float64),
        "p_arg": np.angle(p_vals).astype(np.float64),
    }

    vtu_path = out_dir / f"{tag}.vtu"
    m = meshio.Mesh(points=coords, cells=[], point_data=point_data)
    m.write(str(vtu_path))
    print(f"  VTU saved: {vtu_path.resolve()}  ({vtu_path.stat().st_size/1e6:.1f} MB)")
    return vtu_path


# ─────────────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────────────
def _plot_xy(sol, z_mm: float, tag: str, out_dir: Path):
    """Plot |p| and arg(p) at an XY slice, save PNG."""
    cfg = sol.cfg
    z = z_mm * 1e-3
    xg, yg, pmag, pphase = slice_xy(sol, z, 200, 200)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    im = ax.pcolormesh(xg * 1e3, yg * 1e3, pmag, cmap="viridis", shading="auto")
    fig.colorbar(im, ax=ax, label="|p| [Pa]")
    ax.set_title(f"|p|  z = {z_mm:.1f} mm")
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    ax.set_aspect("equal")

    ax = axes[1]
    im = ax.pcolormesh(xg * 1e3, yg * 1e3, pphase, cmap="twilight",
                       vmin=-np.pi, vmax=np.pi, shading="auto")
    fig.colorbar(im, ax=ax, label="arg(p) [rad]")
    ax.set_title(f"arg(p)  z = {z_mm:.1f} mm")
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    ax.set_aspect("equal")

    fig.tight_layout()
    png = out_dir / f"{tag}_xy_z{z_mm:.1f}mm.png"
    fig.savefig(str(png), dpi=150)
    plt.close(fig)
    print(f"  PNG: {png.name}")


def _plot_xz(sol, tag: str, out_dir: Path):
    """Plot |p| in the XZ mid-plane."""
    xg, zg, pmag_xz, _ = slice_xz(sol, 200, 200)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.pcolormesh(xg * 1e3, zg * 1e3, pmag_xz, cmap="viridis", shading="auto")
    fig.colorbar(im, ax=ax, label="|p| [Pa]")
    ax.set_title(f"XZ mid-plane |p|")
    ax.set_xlabel("x [mm]"); ax.set_ylabel("z [mm]")
    ax.set_aspect("equal")
    fig.tight_layout()
    png = out_dir / f"{tag}_xz.png"
    fig.savefig(str(png), dpi=150)
    plt.close(fig)
    print(f"  PNG: {png.name}")


def _plot_centerline(sol, tag: str, out_dir: Path):
    """Plot on-axis |p|(z)."""
    zvals, pz = centerline_z(sol, 400)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(zvals * 1e3, np.abs(pz), "b-", linewidth=1)
    ax.set_xlabel("z [mm]"); ax.set_ylabel("|p| [Pa]")
    ax.set_title("On-axis |p|(z)")
    ax.axvline(sol.cfg.H_under * 1e3, color="gray", ls="--", alpha=0.5, label="petri top")
    ax.legend()
    fig.tight_layout()
    png = out_dir / f"{tag}_centerline.png"
    fig.savefig(str(png), dpi=150)
    plt.close(fig)
    print(f"  PNG: {png.name}")


# ─────────────────────────────────────────────────────────────────────
# Radial profile comparison
# ─────────────────────────────────────────────────────────────────────
def _radial_profile(sol, z: float, nr: int = 200):
    """Compute azimuthally-averaged |p|(r) at given z height."""
    cfg = sol.cfg
    xg, yg, pmag, _ = slice_xy(sol, z, 200, 200)
    cx, cy = cfg.disk_center_x, cfg.disk_center_y

    # Build radial bins
    xx, yy = np.meshgrid(xg - cx, yg - cy)
    rr = np.sqrt(xx**2 + yy**2)
    r_edges = np.linspace(0, cfg.disk_radius, nr + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    p_avg = np.zeros(nr)
    for i in range(nr):
        mask = (rr >= r_edges[i]) & (rr < r_edges[i + 1])
        if np.any(mask):
            p_avg[i] = np.mean(pmag[mask])
    return r_centers, p_avg


def _plot_radial_comparison(solutions: dict, out_dir: Path):
    """Compare radial |p| profiles for all cases."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"axicon_15deg": "C0", "axicon_10deg": "C1", "plastic_ref": "C2"}

    for tag, sol in solutions.items():
        if sol is None:
            continue
        z = sol.cfg.H_under + sol.cfg.H_top / 2
        r, p_avg = _radial_profile(sol, z)
        ax.plot(r * 1e3, p_avg, label=tag, color=colors.get(tag, "k"), linewidth=1.5)

    ax.set_xlabel("r [mm]")
    ax.set_ylabel("|p| [Pa]  (azimuthal average)")
    ax.set_title("Radial Profile at Petri Mid-Plane")
    ax.legend()
    fig.tight_layout()
    png = out_dir / "radial_comparison.png"
    fig.savefig(str(png), dpi=150)
    plt.close(fig)
    print(f"  PNG: {png.name}")


# ─────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────
def _write_report(out_dir: Path, results: list[dict]):
    lines = [
        "# Axicon Lens Demo — REPORT",
        "",
        f"**Date:** {datetime.now().isoformat()}",
        f"**Script:** `scripts/experiments/run_axicon_lens_demo.py`",
        "",
    ]
    for r in results:
        lines.extend([
            f"## {r['tag']}",
            "",
            f"- Lens: {r['lens_drive']}"
            + (f"  α = {r.get('axicon_angle', '—')}°" if r['lens_drive'] == "axicon" else ""),
            f"- elements/λ: {r['epw']}",
            f"- DOFs: {r['dofs']}",
            f"- max|p|: {r['max_p']:.2f} Pa",
            f"- KSP reason: {r['ksp_reason']}",
            f"- Solver time: {r['solver_time']:.1f} s",
            f"- VTU: `{r['vtu_name']}`",
            "",
        ])

    lines.extend([
        "## How to Compare in ParaView",
        "",
        "1. Open all three `.vtu` files",
        "2. Color each by `p_abs` with the same range",
        "3. Use **Linked Views** to sync cameras",
        "4. Compare vortex core diameter: axicon beams should be",
        "   narrower (non-diffracting Bessel core) vs converging lens",
        "",
    ])

    rpt = out_dir / "REPORT.md"
    rpt.write_text("\n".join(lines))
    print(f"  REPORT.md: {rpt.resolve()}")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'#'*70}")
    print(f"  AXICON LENS DEMO")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'#'*70}\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    solutions: dict[str, object] = {}
    all_results: list[dict] = []

    for tag, overrides in CASES.items():
        print(f"\n{'='*60}")
        print(f"  Case: {tag}")
        print(f"{'='*60}")

        sol = None
        used_epw = None
        for epw in ELEM_PER_WAVELENGTH_TARGETS:
            cfg = _build_cfg(overrides, epw)
            print(f"  {epw} elem/λ  ({cfg.mesh_nx}×{cfg.mesh_ny}×{cfg.mesh_nz}) ...")
            try:
                sol = solve_helmholtz(cfg, verbose=True, petsc_options=PETSC_OPTS)
                used_epw = epw
                break
            except Exception as e:
                print(f"  FAILED at {epw} elem/λ: {e}")
                gc.collect()
                continue

        if sol is None:
            print(f"  *** All resolutions failed for {tag} — skipping ***")
            solutions[tag] = None
            continue

        solutions[tag] = sol
        print(f"  Solved: max|p| = {sol.max_pressure:.2f} Pa, DOFs = {sol.dofs}")

        # ── VTU export ────────────────────────────────────────────────
        vtu_path = _export_vtu(sol, OUTPUT_DIR, tag)

        # ── Per-case plots ────────────────────────────────────────────
        z_petri_mm = (cfg.H_under + cfg.H_top / 2) * 1e3
        _plot_xy(sol, z_petri_mm, tag, OUTPUT_DIR)
        _plot_xz(sol, tag, OUTPUT_DIR)
        _plot_centerline(sol, tag, OUTPUT_DIR)

        info = {
            "tag": tag,
            "lens_drive": overrides.get("lens_drive", "plastic"),
            "axicon_angle": overrides.get("lens_axicon_angle_deg", None),
            "epw": used_epw,
            "dofs": sol.dofs,
            "max_p": sol.max_pressure,
            "ksp_reason": sol.ksp_converged_reason,
            "solver_time": sol.solver_time,
            "vtu_name": vtu_path.name,
        }
        all_results.append(info)
        gc.collect()

    # ── Cross-case comparison plot ────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Generating radial comparison …")
    _plot_radial_comparison(solutions, OUTPUT_DIR)

    # ── Report & JSON ─────────────────────────────────────────────────
    _write_report(OUTPUT_DIR, all_results)

    json_path = OUTPUT_DIR / "results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  results.json: {json_path.resolve()}")

    # ── Validation ────────────────────────────────────────────────────
    for r in all_results:
        vf = OUTPUT_DIR / r["vtu_name"]
        if not vf.exists() or vf.stat().st_size < 500:
            raise RuntimeError(f"VTU incomplete: {vf}")

    print(f"\n{'='*70}")
    print(f"  AXICON DEMO COMPLETE — {len(all_results)} cases exported")
    print(f"  Output: {OUTPUT_DIR.resolve()}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
