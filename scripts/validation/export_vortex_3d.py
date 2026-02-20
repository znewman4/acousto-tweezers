#!/usr/bin/env python3
"""
Robust 3D ParaView Export — Vortex-Only Case (VTU Primary)
==========================================================

Runs lens-only (standing OFF), 5 elem/λ (or 3 if OOM), MUMPS direct.

Exports scalar and vector fields to VTU in the OneDrive output folder:

    ~/OneDrive - University of Bristol/Major Project Onedrive/
      Research/Vortex 3D visualisation/Vortex3D/

Fields exported:
    p_re, p_im, p_abs, p_arg        (scalar)
    Iz                               (scalar — z-component of intensity)
    I_abs                            (scalar — |I|)
    I_vec                            (vector — acoustic intensity)

Optional XDMF export with .h5 sanity check.

Usage:
    micromamba run -n fenicsx python scripts/validation/export_vortex_3d.py
"""
from __future__ import annotations

import json
import sys
import time
import gc
import numpy as np
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

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
OUTPUT_DIR = ONEDRIVE_BASE / "Vortex3D"

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


def _build_cfg(epw: int) -> FarFieldConfig:
    """Build vortex-only config at given elements_per_wavelength."""
    return FarFieldConfig(
        Lx=6e-3, Ly=6e-3, H_under=3e-3, H_top=1e-3,
        frequency_hz=2.0e6,
        disk_radius=1.0e-3,
        disk_velocity_amplitude=1e-6,
        vortex_topological_charge=1,
        standing_velocity_amplitude=0.0,  # standing OFF
        standing_phase_pattern="antiphase",
        standing_axis="both",
        top_bc_type="impedance",
        top_impedance_Zrel=0.001,
        pml_n_wavelengths_xy=1.0,
        pml_n_wavelengths_z=1.0,
        pml_degree=2,
        pml_sigma_max_factor=5.0,
        pml_enabled=True,
        elements_per_wavelength=epw,
        lens_drive="plastic",
        lens_l=1,
        lens_focal_length=10e-3,
        lens_focus_offset_x=0.2e-3,
        lens_focus_offset_y=0.0,
        lens_c_lens=2700.0,
        lens_apodization="cosine_taper",
        lens_apodization_strength=1.0,
    )


def _compute_intensity(p_vals: np.ndarray, coords: np.ndarray,
                       omega: float, rho: float, k: float):
    """
    Approximate acoustic intensity I = (1/2) Re[p v*].

    Uses finite-difference gradient on DOF coordinates for v = ∇p/(iωρ).
    Falls back to plane-wave approximation |I| ≈ |p|²/(2ρc) if FD fails.

    Returns I_vec (N,3), I_abs (N,), Iz (N,).
    """
    from scipy.interpolate import NearestNDInterpolator

    N = len(p_vals)
    c = omega / k

    # Plane-wave magnitude as fallback
    I_pw_mag = np.abs(p_vals)**2 / (2 * rho * c)

    try:
        # Build NearestNDInterpolator for p_re and p_im separately
        interp_re = NearestNDInterpolator(coords, np.real(p_vals))
        interp_im = NearestNDInterpolator(coords, np.imag(p_vals))

        # Estimate gradient via small perturbation
        eps = 1e-6  # 1 µm step
        grad_p = np.zeros((N, 3), dtype=np.complex128)
        for dim in range(3):
            pts_plus = coords.copy()
            pts_plus[:, dim] += eps
            p_re_plus = interp_re(pts_plus)
            p_im_plus = interp_im(pts_plus)
            grad_p[:, dim] = ((p_re_plus - np.real(p_vals))
                              + 1j * (p_im_plus - np.imag(p_vals))) / eps

        # v = grad(p) / (i ω ρ)
        v = grad_p / (1j * omega * rho)

        # I = (1/2) Re[p conj(v)]
        I_vec = 0.5 * np.real(p_vals[:, None] * np.conj(v))
        I_abs = np.linalg.norm(I_vec, axis=1)
        Iz = I_vec[:, 2]

    except Exception:
        # Fallback: assume upward propagation
        I_vec = np.zeros((N, 3), dtype=np.float64)
        I_vec[:, 2] = I_pw_mag
        I_abs = I_pw_mag
        Iz = I_pw_mag

    return I_vec, I_abs, Iz


def _export_vtu(sol, output_dir: Path) -> Path:
    """Export all fields to VTU using meshio."""
    import meshio

    output_dir.mkdir(parents=True, exist_ok=True)

    coords = sol.coords
    p_vals = sol.p_values
    cfg = sol.cfg

    # ── Scalar fields ──────────────────────────────────────────────────
    I_vec, I_abs, Iz = _compute_intensity(
        p_vals, coords, cfg.omega, cfg.rho, cfg.k)

    point_data = {
        "p_re":  np.real(p_vals).astype(np.float64),
        "p_im":  np.imag(p_vals).astype(np.float64),
        "p_abs": np.abs(p_vals).astype(np.float64),
        "p_arg": np.angle(p_vals).astype(np.float64),
        "Iz":    Iz.astype(np.float64),
        "I_abs": I_abs.astype(np.float64),
        "I_vec": I_vec.astype(np.float64),   # (N,3) vector
    }

    # ── Extract mesh topology ─────────────────────────────────────────
    domain = sol.domain
    topology = domain.topology
    tdim = topology.dim
    topology.create_connectivity(tdim, 0)
    cell_map = topology.index_map(tdim)
    num_cells = cell_map.size_local
    cells_local = domain.geometry.dofmap[:num_cells]

    # Map geometry DOFs → P2 DOFs via coordinates
    geom_coords = domain.geometry.x[:, :3]

    # For P2 elements, DOFs don't align 1:1 with geometry nodes.
    # Export as point cloud (no topology) for maximum compatibility.
    vtu_path = output_dir / "fields_0000.vtu"
    m = meshio.Mesh(
        points=coords,
        cells=[],
        point_data=point_data,
    )
    m.write(str(vtu_path))
    print(f"  VTU saved: {vtu_path.resolve()}")
    print(f"  VTU size:  {vtu_path.stat().st_size / 1e6:.1f} MB")
    return vtu_path


def _export_xdmf_optional(sol, output_dir: Path) -> Path | None:
    """Optionally export XDMF+H5. Sanity-check h5 existence and size."""
    from dolfinx.io import XDMFFile

    output_dir.mkdir(parents=True, exist_ok=True)
    xdmf_path = output_dir / "fields.xdmf"
    h5_path = output_dir / "fields.h5"

    try:
        with XDMFFile(sol.domain.comm, str(xdmf_path), "w") as xf:
            xf.write_mesh(sol.domain)
            xf.write_function(sol.p_function)
        # Sanity checks
        if not h5_path.exists():
            raise RuntimeError(f"XDMF export produced no .h5 file at {h5_path}")
        h5_size = h5_path.stat().st_size
        if h5_size < 1_000_000:  # 1 MB
            raise RuntimeError(
                f"H5 file too small ({h5_size} bytes) — likely corrupt or empty"
            )
        print(f"  XDMF saved: {xdmf_path.resolve()}")
        print(f"  H5 size:    {h5_size / 1e6:.1f} MB")
        return xdmf_path
    except Exception as e:
        print(f"  XDMF export skipped/failed: {e}")
        # Clean up partial files
        for p in [xdmf_path, h5_path]:
            if p.exists():
                p.unlink()
        return None


def _winding_check(sol, z_slice: float, nx: int = 200, ny: int = 200):
    """Check that arg(p) winds by 2πℓ around the vortex axis."""
    xg, yg, pmag, pphase = slice_xy(sol, z_slice, nx, ny)
    cfg = sol.cfg
    cx, cy = cfg.disk_center_x, cfg.disk_center_y

    # Sample phase on a ring around the axis
    ring_r = cfg.disk_radius * 0.5
    n_ring = 360
    theta_ring = np.linspace(0, 2 * np.pi, n_ring, endpoint=False)
    x_ring = cx + ring_r * np.cos(theta_ring)
    y_ring = cy + ring_r * np.sin(theta_ring)

    from scipy.interpolate import RegularGridInterpolator
    phase_interp = RegularGridInterpolator(
        (yg, xg), pphase, method="nearest", bounds_error=False, fill_value=0.0)
    pts_ring = np.column_stack([y_ring, x_ring])
    phase_ring = phase_interp(pts_ring)

    # Total winding = sum of angle differences
    dphi = np.diff(phase_ring)
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi  # wrap to [-π, π]
    total_winding = np.sum(dphi)
    ell_measured = total_winding / (2 * np.pi)

    return {
        "ell_expected": cfg.vortex_topological_charge,
        "ell_measured": float(ell_measured),
        "winding_2pi": float(total_winding),
        "pass": abs(ell_measured - cfg.vortex_topological_charge) < 0.3,
    }


def _core_ratio(sol, z_slice: float, nx: int = 200, ny: int = 200):
    """Compute on-axis / off-axis |p| ratio (vortex null quality)."""
    xg, yg, pmag, _ = slice_xy(sol, z_slice, nx, ny)
    cfg = sol.cfg
    cx_idx = np.argmin(np.abs(xg - cfg.disk_center_x))
    cy_idx = np.argmin(np.abs(yg - cfg.disk_center_y))

    on_axis = pmag[cy_idx, cx_idx]
    # Off-axis: mean on a ring at r ≈ 0.5 * disk_radius
    ring_r_px = int(0.5 * cfg.disk_radius / (xg[1] - xg[0]))
    ring_vals = []
    for angle in np.linspace(0, 2 * np.pi, 36, endpoint=False):
        ri = cy_idx + int(ring_r_px * np.sin(angle))
        ci = cx_idx + int(ring_r_px * np.cos(angle))
        if 0 <= ri < len(yg) and 0 <= ci < len(xg):
            ring_vals.append(pmag[ri, ci])
    off_axis = np.mean(ring_vals) if ring_vals else 1.0

    return {
        "on_axis": float(on_axis),
        "off_axis": float(off_axis),
        "core_ratio": float(on_axis / off_axis) if off_axis > 0 else float("inf"),
    }


def _pml_decay(sol):
    """Measure PML absorption: energy ratio PML/physical."""
    en = energy_physical_vs_pml(sol)
    return {
        "energy_physical": float(en["physical"]),
        "energy_pml": float(en["pml"]),
        "ratio": float(en["ratio"]),
    }


def _mean_iz_above(sol, z_above: float):
    """Mean Iz in the region z > z_above (petri slab)."""
    coords = sol.coords
    p_vals = sol.p_values
    cfg = sol.cfg
    mask = coords[:, 2] > z_above
    _, _, Iz = _compute_intensity(p_vals, coords, cfg.omega, cfg.rho, cfg.k)
    return float(np.mean(Iz[mask])) if np.any(mask) else 0.0


def _write_report(output_dir: Path, cfg, sol, winding, core, pml,
                  mean_iz, vtu_path, xdmf_path):
    """Write REPORT.md with diagnostics and ParaView instructions."""
    lines = [
        "# Vortex 3D Export — REPORT",
        "",
        f"**Date:** {datetime.now().isoformat()}",
        f"**Script:** `scripts/validation/export_vortex_3d.py`",
        "",
        "## Configuration",
        "",
        f"- Domain: {cfg.Lx*1e3:.1f} x {cfg.Ly*1e3:.1f} x {cfg.H_total*1e3:.1f} mm",
        f"- Frequency: {cfg.frequency_hz/1e6:.2f} MHz, lambda = {cfg.wavelength*1e3:.3f} mm",
        f"- Mesh: {cfg.mesh_nx}x{cfg.mesh_ny}x{cfg.mesh_nz} ({cfg.elements_per_wavelength} elem/lambda)",
        f"- DOFs: {sol.dofs}",
        f"- Solver: MUMPS direct, KSP reason={sol.ksp_converged_reason}, time={sol.solver_time:.1f}s",
        f"- Lens: plastic l={cfg.lens_l}, f={cfg.lens_focal_length*1e3:.1f} mm",
        f"- Standing: OFF",
        f"- max|p| = {sol.max_pressure:.2f} Pa",
        "",
        "## Winding Check",
        "",
        f"- Expected ell: {winding['ell_expected']}",
        f"- Measured ell: {winding['ell_measured']:.2f}",
        f"- Total winding: {winding['winding_2pi']:.2f} rad",
        f"- **{'PASS' if winding['pass'] else 'FAIL'}**",
        "",
        "## Core Ratio (Vortex Null Quality)",
        "",
        f"- On-axis |p|: {core['on_axis']:.4f} Pa",
        f"- Off-axis |p|: {core['off_axis']:.4f} Pa",
        f"- core_ratio: {core['core_ratio']:.4f}",
        f"  (< 0.3 is good)",
        "",
        "## PML Decay",
        "",
        f"- Energy physical: {pml['energy_physical']:.4e}",
        f"- Energy PML: {pml['energy_pml']:.4e}",
        f"- Ratio PML/phys: {pml['ratio']:.4f}",
        "",
        "## Mean Iz (petri slab, z > H_under)",
        "",
        f"- mean(Iz): {mean_iz:.6e} W/m^2",
        "",
        "## Exported Files",
        "",
        f"- VTU: `{vtu_path.name}`",
    ]
    if xdmf_path:
        lines.append(f"- XDMF: `{xdmf_path.name}` + `.h5`")
    lines.extend([
        "",
        "## ParaView Instructions",
        "",
        "1. **File > Open** -> select `fields_0000.vtu`",
        "2. Click **Apply**",
        "3. In the **Properties** panel, choose field to color by:",
        "   - `p_abs` -> viridis (0 to max)",
        "   - `p_arg` -> twilight (-pi to +pi, cyclic)",
        "   - `Iz` -> coolwarm (diverging, centred on 0)",
        "   - `I_abs` -> plasma (0 to max)",
        "4. For vector field intensity, select `I_vec` with Glyph filter",
        "5. Use **Slice** filter (normal Z) at z = 3.5 mm for petri slab",
        "6. Use **Contour** on `p_abs` to see isosurfaces",
        "",
    ])

    report_path = output_dir / "REPORT.md"
    report_path.write_text("\n".join(lines))
    print(f"  REPORT.md saved: {report_path.resolve()}")


def main():
    print(f"\n{'#'*70}")
    print(f"  VORTEX 3D EXPORT — VTU Primary")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'#'*70}\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Solve with fallback on mesh resolution ────────────────────────
    sol = None
    used_epw = None
    for epw in ELEM_PER_WAVELENGTH_TARGETS:
        cfg = _build_cfg(epw)
        print(f"  Attempting {epw} elem/lambda ({cfg.mesh_nx}x{cfg.mesh_ny}"
              f"x{cfg.mesh_nz} grid) ...")
        try:
            sol = solve_helmholtz(cfg, verbose=True, petsc_options=PETSC_OPTS)
            used_epw = epw
            break
        except Exception as e:
            print(f"  FAILED at {epw} elem/lambda: {e}")
            gc.collect()
            continue

    if sol is None:
        raise RuntimeError("All mesh resolutions failed. Check RAM and MUMPS.")

    cfg = sol.cfg
    print(f"\n  Solved at {used_epw} elem/lambda, max|p| = {sol.max_pressure:.2f} Pa\n")

    # ── Export VTU ────────────────────────────────────────────────────
    vtu_path = _export_vtu(sol, OUTPUT_DIR)

    # ── Optional XDMF ────────────────────────────────────────────────
    xdmf_path = _export_xdmf_optional(sol, OUTPUT_DIR)

    # ── Diagnostics ───────────────────────────────────────────────────
    z_petri_mid = cfg.H_under + cfg.H_top / 2

    winding = _winding_check(sol, z_petri_mid)
    print(f"  Winding: ell_measured={winding['ell_measured']:.2f}  "
          f"{'PASS' if winding['pass'] else 'FAIL'}")

    core = _core_ratio(sol, z_petri_mid)
    print(f"  Core ratio: {core['core_ratio']:.4f}")

    pml = _pml_decay(sol)
    print(f"  PML energy ratio: {pml['ratio']:.4f}")

    mean_iz = _mean_iz_above(sol, cfg.H_under)
    print(f"  Mean Iz above H_under: {mean_iz:.6e}")

    # ── REPORT.md ─────────────────────────────────────────────────────
    _write_report(OUTPUT_DIR, cfg, sol, winding, core, pml,
                  mean_iz, vtu_path, xdmf_path)

    # ── results.json ──────────────────────────────────────────────────
    results = {
        "date": datetime.now().isoformat(),
        "elements_per_wavelength": used_epw,
        "dofs": sol.dofs,
        "max_pressure_Pa": sol.max_pressure,
        "ksp_converged_reason": sol.ksp_converged_reason,
        "solver_time_s": sol.solver_time,
        "winding": winding,
        "core_ratio": core,
        "pml_decay": pml,
        "mean_Iz_petri": mean_iz,
        "vtu_file": str(vtu_path.name),
        "vtu_size_MB": vtu_path.stat().st_size / 1e6,
    }
    json_path = OUTPUT_DIR / "results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  results.json saved: {json_path.resolve()}")

    # ── Final validation ──────────────────────────────────────────────
    if not vtu_path.exists() or vtu_path.stat().st_size < 1000:
        raise RuntimeError(f"VTU export incomplete: {vtu_path}")

    print(f"\n{'='*70}")
    print(f"  EXPORT COMPLETE")
    print(f"  Output: {OUTPUT_DIR.resolve()}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
