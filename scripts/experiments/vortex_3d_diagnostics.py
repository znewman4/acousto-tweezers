#!/usr/bin/env python3
"""
Vortex 3-D Physics Validation — Lens-only diagnostic suite.

This is a PHYSICS VALIDATION script, not a visualization script.

Validates:
  1. Vortex topology (ℓ = 1 winding number)
  2. Power-flow direction (I_z sign)
  3. PML absorption behaviour
  4. Boundary sign correctness

Parts:
  PART 1 — Export 3-D volumetric fields for ParaView (XDMF + VTX)
  PART 2 — Topological charge check (discrete winding number)
  PART 3 — Axial null check (core_ratio)
  PART 4 — Power-flow direction check (mean I_z)
  PART 5 — PML absorption check (decay ratio)
  PART 6 — Z-stack XY slice export

Output structure:
  results/Vortex3D/
      fields.xdmf / fields.h5      (XDMF scalars + vector)
      fields_vtx.bp/               (ADIOS2/VTX for ParaView 5.12+)
      REPORT.md
      results.json
      slices/

Usage:
    module load anaconda/3-2025
    conda activate acousto-complex
    python scripts/experiments/vortex_3d_diagnostics.py
"""
from __future__ import annotations

import gc
import json
import sys
import time
import traceback
import numpy as np
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mpi4py import MPI
import dolfinx
from dolfinx import fem, io
import basix

from acoustweezers.experiments.farfield_petri_cuboid.config import (
    FarFieldConfig, demo_config,
)
from acoustweezers.experiments.farfield_petri_cuboid.solve_pressure import (
    solve_helmholtz, PressureSolution,
)

REPO = Path(__file__).resolve().parents[2]
OUT_ROOT = REPO / "results" / "Vortex3D"

# ═════════════════════════════════════════════════════════════════════
#  Helpers
# ═════════════════════════════════════════════════════════════════════

def _lens_only_config(elements_per_wavelength: int = 5) -> FarFieldConfig:
    """Lens-only: standing OFF, impedance-matched top, PML ON."""
    return demo_config(
        standing_velocity_amplitude=0.0,
        elements_per_wavelength=elements_per_wavelength,
        pml_enabled=True,
        top_impedance_Zrel=1.0,
    )


def _interp_complex(sol, pts):
    """Nearest-neighbour interpolation of complex p at arbitrary points."""
    from scipy.interpolate import NearestNDInterpolator
    coords = sol.coords
    pv = sol.p_values
    interp_re = NearestNDInterpolator(coords, np.real(pv))
    interp_im = NearestNDInterpolator(coords, np.imag(pv))
    return interp_re(pts) + 1j * interp_im(pts)


def _interp_complex_xy(sol, z_val, nx=200, ny=200):
    """Return (xg, yg, p_complex) on XY grid at given z."""
    cfg = sol.cfg
    xg = np.linspace(0, cfg.Lx, nx)
    yg = np.linspace(0, cfg.Ly, ny)
    X, Y = np.meshgrid(xg, yg)
    Z = np.full_like(X, z_val)
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    pc = _interp_complex(sol, pts).reshape(X.shape)
    return xg, yg, pc


def _save_fig(fig, path, dpi=150):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════
#  Intensity computation on DOF coordinates
# ═════════════════════════════════════════════════════════════════════

def _compute_intensity_on_dofs(sol: PressureSolution):
    """
    Compute time-averaged acoustic intensity vector on P1 DOF coordinates.

    I = 0.5 * Re(p · conj(v))
    v = -(1 / (i ω ρ)) ∇p

    Uses finite-difference gradient from nearest-neighbour interpolation.

    Returns
    -------
    coords_p1 : (N, 3) array
    Ix, Iy, Iz : (N,) real arrays
    """
    from scipy.interpolate import NearestNDInterpolator

    cfg = sol.cfg
    omega = cfg.omega
    rho = cfg.rho

    # Use P1 grid for output (smaller than P2)
    V1 = fem.functionspace(sol.domain, ("Lagrange", 1))
    coords_p1 = V1.tabulate_dof_coordinates()
    n = len(coords_p1)

    # Interpolate complex p at P1 nodes
    p_vals = _interp_complex(sol, coords_p1)

    # Finite-difference gradient: h must be > nearest-DOF spacing
    # so that NearestNDInterpolator snaps to *different* DOFs at ±h.
    # Use 1.5× element size to guarantee separation.
    h = 1.5 * cfg.wavelength / cfg.elements_per_wavelength

    dp_dx = np.zeros(n, dtype=np.complex128)
    dp_dy = np.zeros(n, dtype=np.complex128)
    dp_dz = np.zeros(n, dtype=np.complex128)

    for dim, dp in [(0, dp_dx), (1, dp_dy), (2, dp_dz)]:
        pts_plus = coords_p1.copy()
        pts_minus = coords_p1.copy()
        pts_plus[:, dim] += h
        pts_minus[:, dim] -= h
        p_plus = _interp_complex(sol, pts_plus)
        p_minus = _interp_complex(sol, pts_minus)
        dp[:] = (p_plus - p_minus) / (2 * h)

    # v = (1/(iωρ)) ∇p   [linearised Euler eqn, exp(-iωt) convention]
    factor = 1.0 / (1j * omega * rho)
    vx = factor * dp_dx
    vy = factor * dp_dy
    vz = factor * dp_dz

    # I = 0.5 * Re(p * conj(v))
    Ix = 0.5 * np.real(p_vals * np.conj(vx))
    Iy = 0.5 * np.real(p_vals * np.conj(vy))
    Iz = 0.5 * np.real(p_vals * np.conj(vz))

    return coords_p1, V1, p_vals, Ix, Iy, Iz


# ═════════════════════════════════════════════════════════════════════
#  PART 1 — Export 3-D Fields for ParaView
# ═════════════════════════════════════════════════════════════════════

def part1_export_3d_fields(sol: PressureSolution, out_dir: Path,
                           coords_p1, V1, p_at_p1, Ix, Iy, Iz):
    """
    Export volumetric fields to XDMF and VTX for ParaView.

    Scalars: Re(p), Im(p), |p|, arg(p), I_z, |I|
    Vector:  I = (Ix, Iy, Iz)
    """
    print("\n  ── PART 1: Export 3-D Fields for ParaView ──")

    domain = sol.domain
    n = len(coords_p1)

    # --- Build scalar P1 functions (real-valued for export) ---
    def _make_scalar(name, data):
        f = fem.Function(V1, name=name)
        f.x.array[:] = data.astype(np.float64)
        return f

    f_re_p   = _make_scalar("Re_p",   np.real(p_at_p1))
    f_im_p   = _make_scalar("Im_p",   np.imag(p_at_p1))
    f_abs_p  = _make_scalar("abs_p",  np.abs(p_at_p1))
    f_arg_p  = _make_scalar("arg_p",  np.angle(p_at_p1))
    f_Iz     = _make_scalar("I_z",    Iz)
    f_Imag   = _make_scalar("abs_I",  np.sqrt(Ix**2 + Iy**2 + Iz**2))

    scalars = [f_re_p, f_im_p, f_abs_p, f_arg_p, f_Iz, f_Imag]

    # --- Build vector P1 function ---
    el_vec = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1,
                               shape=(3,))
    Vvec = fem.functionspace(domain, el_vec)
    f_I_vec = fem.Function(Vvec, name="I_vector")
    # Vvec has block size 3: [Ix0, Iy0, Iz0, Ix1, Iy1, Iz1, …]
    f_I_vec.x.array[:] = np.column_stack([Ix, Iy, Iz]).ravel().astype(np.float64)

    # --- XDMF export (widest ParaView compatibility) ---
    xdmf_path = out_dir / "fields.xdmf"
    with io.XDMFFile(MPI.COMM_WORLD, str(xdmf_path), "w") as xf:
        xf.write_mesh(domain)
        for f in scalars:
            xf.write_function(f)
        xf.write_function(f_I_vec)
    print(f"    XDMF written: {xdmf_path.name}")

    # --- VTX export (ADIOS2, better for large data) ---
    vtx_path = out_dir / "fields_vtx.bp"
    with io.VTXWriter(MPI.COMM_WORLD, str(vtx_path),
                      scalars + [f_I_vec]) as vtx:
        vtx.write(0.0)
    print(f"    VTX written:  {vtx_path.name}/")


# ═════════════════════════════════════════════════════════════════════
#  PART 2 — Topological Charge Check
# ═════════════════════════════════════════════════════════════════════

def part2_topological_charge(sol: PressureSolution):
    """
    Discrete winding-number computation on a horizontal ring at mid-height.

    Samples p on a circular ring around the beam axis, unwraps phase,
    computes Δφ/(2π).
    """
    print("\n  ── PART 2: Topological Charge Check ──")
    cfg = sol.cfg

    cx, cy = cfg.disk_center_x, cfg.disk_center_y
    z_mid = cfg.H_total / 2

    # Ring radius: ~0.3 mm (well within beam, outside the null)
    ring_r = 0.3e-3
    n_pts = 360
    theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)

    x_ring = cx + ring_r * np.cos(theta)
    y_ring = cy + ring_r * np.sin(theta)
    z_ring = np.full(n_pts, z_mid)
    pts = np.column_stack([x_ring, y_ring, z_ring])

    p_ring = _interp_complex(sol, pts)
    phase = np.angle(p_ring)
    unwrapped = np.unwrap(phase)

    delta_phi = unwrapped[-1] - unwrapped[0]
    # Account for the missing step (endpoint=False):
    step_phase = np.angle(p_ring[0]) - np.angle(p_ring[-1])
    # Use total unwrapped change + wrap of closing step
    total_wind = unwrapped[-1] - unwrapped[0]
    # Better: use sum of wrapped increments
    diffs = np.diff(phase)
    diffs_wrapped = (diffs + np.pi) % (2 * np.pi) - np.pi
    total_wind = np.sum(diffs_wrapped)
    # Include closing step
    close_diff = phase[0] - phase[-1]
    close_wrapped = (close_diff + np.pi) % (2 * np.pi) - np.pi
    total_wind += close_wrapped

    ell = total_wind / (2 * np.pi)

    status = "PASS" if abs(ell - 1.0) < 0.1 else "FAIL"
    print(f"    Ring radius: {ring_r*1e3:.2f} mm at z = {z_mid*1e3:.2f} mm")
    print(f"    Total phase winding: {total_wind:.4f} rad")
    print(f"    Topological charge ℓ = {ell:.4f}")
    print(f"    Status: {status}")
    if status == "FAIL":
        print(f"    ⚠ WARNING: |ℓ − 1| = {abs(ell-1):.4f} > 0.1 — VORTEX TOPOLOGY INCORRECT")

    return {"ell": float(ell), "total_winding_rad": float(total_wind),
            "ring_r_mm": ring_r * 1e3, "z_mm": z_mid * 1e3, "status": status}


# ═════════════════════════════════════════════════════════════════════
#  PART 3 — Axial Null Check
# ═════════════════════════════════════════════════════════════════════

def part3_axial_null(sol: PressureSolution):
    """
    Extract |p| on beam axis and on offset ring, compute core_ratio.
    """
    print("\n  ── PART 3: Axial Null Check ──")
    cfg = sol.cfg
    cx, cy = cfg.disk_center_x, cfg.disk_center_y

    nz = 500
    z_min = cfg.t_pml_z + 0.1e-3 if cfg.pml_enabled else 0.1e-3
    z_max = cfg.H_total - 0.05e-3
    zg = np.linspace(z_min, z_max, nz)

    # On-axis
    pts_axis = np.column_stack([np.full(nz, cx), np.full(nz, cy), zg])
    p_axis = np.abs(_interp_complex(sol, pts_axis))

    # Off-axis ring (average over 8 points at r = 0.4 mm)
    ring_r = 0.4e-3
    n_ring = 8
    theta_ring = np.linspace(0, 2 * np.pi, n_ring, endpoint=False)
    p_ring_max = np.zeros(nz)
    for iz, zv in enumerate(zg):
        x_r = cx + ring_r * np.cos(theta_ring)
        y_r = cy + ring_r * np.sin(theta_ring)
        pts_r = np.column_stack([x_r, y_r, np.full(n_ring, zv)])
        p_r = np.abs(_interp_complex(sol, pts_r))
        p_ring_max[iz] = np.max(p_r)

    max_on_axis = float(np.max(p_axis))
    max_off_axis = float(np.max(p_ring_max))
    core_ratio = max_on_axis / max_off_axis if max_off_axis > 0 else float("inf")

    status = "PASS" if core_ratio < 0.3 else "FAIL"
    print(f"    max|p| on axis:    {max_on_axis:.4f} Pa")
    print(f"    max|p| off-axis:   {max_off_axis:.4f} Pa")
    print(f"    core_ratio:        {core_ratio:.4f}")
    print(f"    Status: {status}")
    if status == "FAIL":
        print(f"    ⚠ WARNING: core_ratio = {core_ratio:.4f} > 0.3 — VORTEX NULL IS WEAK")

    return {
        "max_on_axis_Pa": max_on_axis,
        "max_off_axis_Pa": max_off_axis,
        "core_ratio": core_ratio,
        "ring_r_mm": ring_r * 1e3,
        "status": status,
        "p_axis": p_axis.tolist(),
        "p_ring_max": p_ring_max.tolist(),
        "z_mm": (zg * 1e3).tolist(),
    }


# ═════════════════════════════════════════════════════════════════════
#  PART 4 — Power-Flow Direction Check
# ═════════════════════════════════════════════════════════════════════

def part4_power_flow(sol: PressureSolution, coords_p1, Iz, out_dir: Path):
    """
    Compute mean I_z above and below the source.
    """
    print("\n  ── PART 4: Power-Flow Direction Check ──")
    fig_dir = out_dir / "slices"
    fig_dir.mkdir(parents=True, exist_ok=True)

    cfg = sol.cfg
    cx, cy = cfg.disk_center_x, cfg.disk_center_y
    R = cfg.disk_radius

    x, y, z = coords_p1[:, 0], coords_p1[:, 1], coords_p1[:, 2]
    r = np.sqrt((x - cx)**2 + (y - cy)**2)

    # Beam ROI: within disk radius
    beam_mask = r <= R

    # Above source (z > pml_z_top + small margin, z < H_under)
    z_above_lo = cfg.t_pml_z + 0.2e-3 if cfg.pml_enabled else 0.2e-3
    z_above_hi = cfg.H_under
    above_mask = beam_mask & (z > z_above_lo) & (z < z_above_hi)

    # Below source (in bottom PML, if it exists)
    z_below_hi = cfg.t_pml_z if cfg.pml_enabled else 0.0
    below_mask = beam_mask & (z < z_below_hi) & (z > 0.1e-3)

    Iz_above = Iz[above_mask]
    Iz_below = Iz[below_mask] if np.any(below_mask) else np.array([0.0])

    mean_Iz_above = float(np.mean(Iz_above)) if len(Iz_above) > 0 else 0.0
    mean_Iz_below = float(np.mean(Iz_below)) if len(Iz_below) > 0 else 0.0
    frac_positive = float(np.mean(Iz_above > 0)) if len(Iz_above) > 0 else 0.0

    status = "PASS" if mean_Iz_above > 0 and frac_positive > 0.8 else "FAIL"
    print(f"    DOFs in beam above source: {np.sum(above_mask)}")
    print(f"    DOFs in beam below source: {np.sum(below_mask)}")
    print(f"    mean(I_z) above: {mean_Iz_above:.6e} W/m²")
    print(f"    mean(I_z) below: {mean_Iz_below:.6e} W/m²")
    print(f"    Fraction I_z > 0 in beam (above): {frac_positive:.4f}")
    print(f"    Status: {status}")
    if status == "FAIL":
        if mean_Iz_above < 0:
            print("    ⚠ WARNING: mean(I_z_above) < 0 — BOUNDARY SIGN LIKELY WRONG")
        if frac_positive < 0.8:
            print(f"    ⚠ WARNING: Only {frac_positive*100:.1f}% positive — significant back-reflection or wrong sign")

    return {
        "mean_Iz_above": mean_Iz_above,
        "mean_Iz_below": mean_Iz_below,
        "positive_Iz_fraction": frac_positive,
        "n_dofs_above": int(np.sum(above_mask)),
        "n_dofs_below": int(np.sum(below_mask)),
        "status": status,
    }


# ═════════════════════════════════════════════════════════════════════
#  PART 5 — PML Absorption Check
# ═════════════════════════════════════════════════════════════════════

def part5_pml_absorption(sol: PressureSolution):
    """
    Compute decay ratio: mean|p|² near outer PML boundary / near inner PML interface.
    """
    print("\n  ── PART 5: PML Absorption Check ──")
    cfg = sol.cfg

    if not cfg.pml_enabled:
        print("    PML disabled — skipping.")
        return {"decay_ratio": float("nan"), "status": "SKIP"}

    coords = sol.coords
    pv = sol.p_values
    p2 = np.abs(pv)**2
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    t_xy = cfg.t_pml_xy
    Lx, Ly = cfg.Lx, cfg.Ly

    # Inner shell: just inside PML interface (within 0.1 mm of interface)
    margin = 0.1e-3
    inner_x = ((np.abs(x - t_xy) < margin) | (np.abs(x - (Lx - t_xy)) < margin))
    inner_y = ((np.abs(y - t_xy) < margin) | (np.abs(y - (Ly - t_xy)) < margin))
    inner_mask = inner_x | inner_y
    # Restrict to mid-height to avoid disk/PML-z overlap
    z_mid_lo = cfg.H_total * 0.3
    z_mid_hi = cfg.H_total * 0.7
    inner_mask = inner_mask & (z > z_mid_lo) & (z < z_mid_hi)

    # Outer shell: near the actual domain boundary
    outer_x = (x < margin) | (x > Lx - margin)
    outer_y = (y < margin) | (y > Ly - margin)
    outer_mask = (outer_x | outer_y) & (z > z_mid_lo) & (z < z_mid_hi)

    p2_inner = float(np.mean(p2[inner_mask])) if np.any(inner_mask) else 1e-30
    p2_outer = float(np.mean(p2[outer_mask])) if np.any(outer_mask) else 0.0

    decay_ratio = p2_outer / p2_inner if p2_inner > 0 else float("inf")

    status = "PASS" if decay_ratio < 0.1 else "FAIL"
    print(f"    Inner shell DOFs: {np.sum(inner_mask)}")
    print(f"    Outer shell DOFs: {np.sum(outer_mask)}")
    print(f"    mean|p|² inner:  {p2_inner:.6e}")
    print(f"    mean|p|² outer:  {p2_outer:.6e}")
    print(f"    decay_ratio:     {decay_ratio:.6f}")
    print(f"    Status: {status}")
    if status == "FAIL":
        print(f"    ⚠ WARNING: decay_ratio = {decay_ratio:.4f} > 0.1 — PML MAY BE INSUFFICIENT")

    return {
        "decay_ratio": decay_ratio,
        "p2_inner": p2_inner,
        "p2_outer": p2_outer,
        "n_inner": int(np.sum(inner_mask)),
        "n_outer": int(np.sum(outer_mask)),
        "status": status,
    }


# ═════════════════════════════════════════════════════════════════════
#  PART 6 — Z-Stack Slice Export
# ═════════════════════════════════════════════════════════════════════

def part6_zstack_slices(sol: PressureSolution, coords_p1, Iz_dof, out_dir: Path):
    """
    Save XY slices at 6 z-levels: |p|, arg(p), I_z with consistent color limits.
    """
    print("\n  ── PART 6: Z-Stack Slice Export ──")
    fig_dir = out_dir / "slices"
    fig_dir.mkdir(parents=True, exist_ok=True)

    cfg = sol.cfg
    z_lo = cfg.t_pml_z + 0.1e-3 if cfg.pml_enabled else 0.1e-3
    z_hi = cfg.H_total - 0.05e-3
    z_levels = np.linspace(z_lo, z_hi, 6)

    nx, ny = 180, 180

    # Precompute all slices for consistent colour scaling
    all_pc = []
    all_Iz = []
    for zv in z_levels:
        xg, yg, pc = _interp_complex_xy(sol, zv, nx, ny)
        all_pc.append(pc)

        # Compute I_z on this slice by FD
        _, _, pc_up = _interp_complex_xy(sol, zv + 0.02e-3, nx, ny)
        dp_dz = (pc_up - pc) / 0.02e-3
        vz = 1.0 / (1j * cfg.omega * cfg.rho) * dp_dz
        Iz_slice = 0.5 * np.real(pc * np.conj(vz))
        all_Iz.append(Iz_slice)

    # Global colour limits
    vmax_abs = max(np.abs(pc).max() for pc in all_pc) * 1.02
    Iz_absmax = max(np.percentile(np.abs(Iz_s), 99) for Iz_s in all_Iz)
    if Iz_absmax < 1e-20:
        Iz_absmax = 1.0

    for iz, zv in enumerate(z_levels):
        pc = all_pc[iz]
        Iz_s = all_Iz[iz]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        im0 = axes[0].pcolormesh(xg * 1e3, yg * 1e3, np.abs(pc),
                                  shading="auto", cmap="inferno",
                                  vmin=0, vmax=vmax_abs)
        axes[0].set_title(f"|p|  z={zv*1e3:.2f} mm")
        plt.colorbar(im0, ax=axes[0], label="Pa")

        im1 = axes[1].pcolormesh(xg * 1e3, yg * 1e3, np.angle(pc),
                                  shading="auto", cmap="twilight",
                                  vmin=-np.pi, vmax=np.pi)
        axes[1].set_title(f"arg(p)  z={zv*1e3:.2f} mm")
        plt.colorbar(im1, ax=axes[1], label="rad")

        im2 = axes[2].pcolormesh(xg * 1e3, yg * 1e3, Iz_s,
                                  shading="auto", cmap="RdBu_r",
                                  vmin=-Iz_absmax, vmax=Iz_absmax)
        axes[2].set_title(f"$I_z$  z={zv*1e3:.2f} mm")
        plt.colorbar(im2, ax=axes[2], label="W/m²")

        for ax in axes:
            ax.set_xlabel("x [mm]")
            ax.set_ylabel("y [mm]")
            ax.set_aspect("equal")
            if cfg.pml_enabled:
                t = cfg.t_pml_xy * 1e3
                for bnd in [t, (cfg.Lx - cfg.t_pml_xy) * 1e3]:
                    ax.axvline(bnd, color="w", ls="--", lw=0.4, alpha=0.4)
                for bnd in [t, (cfg.Ly - cfg.t_pml_xy) * 1e3]:
                    ax.axhline(bnd, color="w", ls="--", lw=0.4, alpha=0.4)

        fig.suptitle(f"Z-slice {iz+1}/6  z = {zv*1e3:.2f} mm", fontsize=13)
        fig.tight_layout()
        _save_fig(fig, fig_dir / f"zslice_{iz:02d}_{zv*1e3:.2f}mm.png")

    # Also save an XZ mid-plane overview
    xg_xz, zg_xz, pc_xz = _interp_complex_xy.__wrapped__(sol, cfg.Ly / 2) \
        if hasattr(_interp_complex_xy, '__wrapped__') else _do_xz(sol)

    print(f"    Saved 6 z-slices to slices/")


def _do_xz(sol):
    """Helper: XZ midplane complex field."""
    from scipy.interpolate import NearestNDInterpolator
    cfg = sol.cfg
    coords = sol.coords
    pv = sol.p_values
    interp_re = NearestNDInterpolator(coords, np.real(pv))
    interp_im = NearestNDInterpolator(coords, np.imag(pv))
    xg = np.linspace(0, cfg.Lx, 200)
    zg = np.linspace(0, cfg.H_total, 200)
    X, Z = np.meshgrid(xg, zg)
    Y = np.full_like(X, cfg.Ly / 2)
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    pc = (interp_re(pts) + 1j * interp_im(pts)).reshape(X.shape)
    return xg, zg, pc


# ═════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════

def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("  VORTEX 3-D PHYSICS VALIDATION")
    print(f"  Output: {out_dir}")
    print(f"  Time:   {ts}")
    print(f"{'='*70}")

    results = {"timestamp": ts}
    warnings = []

    # ── Solve ─────────────────────────────────────────────────────────
    print("\n  ═══ Solving lens-only (standing OFF, PML ON) ═══")
    cfg = _lens_only_config(elements_per_wavelength=5)
    print(cfg.describe())
    t0 = time.time()
    sol = solve_helmholtz(cfg, verbose=True)
    solve_time = time.time() - t0

    results["solver"] = {
        "type": "MUMPS direct",
        "DOFs": sol.dofs,
        "max_p_Pa": sol.max_pressure,
        "solve_time_s": solve_time,
        "elements_per_wavelength": cfg.elements_per_wavelength,
        "ksp_converged_reason": sol.ksp_converged_reason,
    }

    # ── Compute intensity on P1 DOFs (shared across parts) ───────────
    print("\n  Computing intensity field on P1 DOFs …")
    t_int = time.time()
    coords_p1, V1, p_at_p1, Ix, Iy, Iz = _compute_intensity_on_dofs(sol)
    print(f"    Intensity computed in {time.time()-t_int:.1f}s  "
          f"({len(coords_p1)} P1 DOFs)")

    # ── PART 1 ────────────────────────────────────────────────────────
    try:
        part1_export_3d_fields(sol, out_dir, coords_p1, V1, p_at_p1, Ix, Iy, Iz)
    except Exception as e:
        print(f"  *** PART 1 FAILED: {e}")
        traceback.print_exc()
        warnings.append(f"PART 1 export failed: {e}")

    # ── PART 2 ────────────────────────────────────────────────────────
    try:
        r2 = part2_topological_charge(sol)
        results["part2_topological_charge"] = r2
        if r2["status"] == "FAIL":
            warnings.append(f"PART 2: ℓ = {r2['ell']:.4f}, deviates from 1")
    except Exception as e:
        print(f"  *** PART 2 FAILED: {e}")
        traceback.print_exc()
        warnings.append(f"PART 2 failed: {e}")

    # ── PART 3 ────────────────────────────────────────────────────────
    try:
        r3 = part3_axial_null(sol)
        # Don't store the full arrays in the top-level JSON (too large)
        r3_summary = {k: v for k, v in r3.items()
                      if k not in ("p_axis", "p_ring_max", "z_mm")}
        results["part3_axial_null"] = r3_summary
        if r3["status"] == "FAIL":
            warnings.append(f"PART 3: core_ratio = {r3['core_ratio']:.4f} > 0.3")
    except Exception as e:
        print(f"  *** PART 3 FAILED: {e}")
        traceback.print_exc()
        warnings.append(f"PART 3 failed: {e}")

    # ── PART 4 ────────────────────────────────────────────────────────
    try:
        r4 = part4_power_flow(sol, coords_p1, Iz, out_dir)
        results["part4_power_flow"] = r4
        if r4["status"] == "FAIL":
            warnings.append(f"PART 4: mean(I_z_above) = {r4['mean_Iz_above']:.6e}, "
                            f"positive fraction = {r4['positive_Iz_fraction']:.4f}")
    except Exception as e:
        print(f"  *** PART 4 FAILED: {e}")
        traceback.print_exc()
        warnings.append(f"PART 4 failed: {e}")

    # ── PART 5 ────────────────────────────────────────────────────────
    try:
        r5 = part5_pml_absorption(sol)
        results["part5_pml_absorption"] = r5
        if r5["status"] == "FAIL":
            warnings.append(f"PART 5: decay_ratio = {r5['decay_ratio']:.4f} > 0.1")
    except Exception as e:
        print(f"  *** PART 5 FAILED: {e}")
        traceback.print_exc()
        warnings.append(f"PART 5 failed: {e}")

    # ── PART 6 ────────────────────────────────────────────────────────
    try:
        part6_zstack_slices(sol, coords_p1, Iz, out_dir)
    except Exception as e:
        print(f"  *** PART 6 FAILED: {e}")
        traceback.print_exc()
        warnings.append(f"PART 6 failed: {e}")

    # ── Aggregate warnings ────────────────────────────────────────────
    results["warnings"] = warnings
    n_pass = sum(1 for k in results if k.startswith("part") and
                 isinstance(results[k], dict) and results[k].get("status") == "PASS")
    n_fail = sum(1 for k in results if k.startswith("part") and
                 isinstance(results[k], dict) and results[k].get("status") == "FAIL")
    results["summary"] = {"pass": n_pass, "fail": n_fail}

    # ── Write results.json ────────────────────────────────────────────
    def _convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        return obj

    json_path = out_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(_convert(results), f, indent=2)

    # ── Write REPORT.md ───────────────────────────────────────────────
    report = _build_report(results, cfg, ts)
    report_path = out_dir / "REPORT.md"
    with open(report_path, "w") as f:
        f.write(report)

    # ── Final summary ─────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  VORTEX 3-D DIAGNOSTICS COMPLETE")
    print(f"  PASS: {n_pass}  FAIL: {n_fail}")
    if warnings:
        print(f"\n  ⚠ WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"    • {w}")
    print(f"\n  Report:  {report_path}")
    print(f"  JSON:    {json_path}")
    print(f"  XDMF:    {out_dir / 'fields.xdmf'}")
    print(f"  VTX:     {out_dir / 'fields_vtx.bp/'}")
    print(f"  Slices:  {out_dir / 'slices/'}")
    print(f"{'='*70}\n")


def _build_report(results, cfg, ts):
    """Build REPORT.md content from results dict."""
    lines = [
        "# Vortex 3-D Physics Validation Report\n",
        f"Generated: {ts}\n\n",
        "## Solver\n",
        f"- Type: {results['solver']['type']}\n",
        f"- DOFs: {results['solver']['DOFs']}\n",
        f"- max|p|: {results['solver']['max_p_Pa']:.2f} Pa\n",
        f"- Elements/λ: {results['solver']['elements_per_wavelength']}\n",
        f"- Solve time: {results['solver']['solve_time_s']:.1f}s\n\n",
    ]

    # Part 2
    if "part2_topological_charge" in results:
        r = results["part2_topological_charge"]
        lines.append("## Part 2: Topological Charge\n")
        lines.append(f"- **ℓ = {r['ell']:.4f}** (expected: 1.0)\n")
        lines.append(f"- Total winding: {r['total_winding_rad']:.4f} rad\n")
        lines.append(f"- Status: **{r['status']}**\n\n")

    # Part 3
    if "part3_axial_null" in results:
        r = results["part3_axial_null"]
        lines.append("## Part 3: Axial Null\n")
        lines.append(f"- **core_ratio = {r['core_ratio']:.4f}** (expected: < 0.3)\n")
        lines.append(f"- max|p| on axis: {r['max_on_axis_Pa']:.4f} Pa\n")
        lines.append(f"- max|p| off-axis: {r['max_off_axis_Pa']:.4f} Pa\n")
        lines.append(f"- Status: **{r['status']}**\n\n")

    # Part 4
    if "part4_power_flow" in results:
        r = results["part4_power_flow"]
        lines.append("## Part 4: Power-Flow Direction\n")
        lines.append(f"- **mean(I_z) above source: {r['mean_Iz_above']:.6e} W/m²**\n")
        lines.append(f"- mean(I_z) below source: {r['mean_Iz_below']:.6e} W/m²\n")
        lines.append(f"- Fraction positive: {r['positive_Iz_fraction']:.4f}\n")
        lines.append(f"- Status: **{r['status']}**\n\n")

    # Part 5
    if "part5_pml_absorption" in results:
        r = results["part5_pml_absorption"]
        lines.append("## Part 5: PML Absorption\n")
        lines.append(f"- **decay_ratio = {r['decay_ratio']:.6f}** (expected: < 0.1)\n")
        lines.append(f"- mean|p|² inner: {r['p2_inner']:.6e}\n")
        lines.append(f"- mean|p|² outer: {r['p2_outer']:.6e}\n")
        lines.append(f"- Status: **{r['status']}**\n\n")

    # Summary
    s = results.get("summary", {})
    lines.append("## Summary\n")
    lines.append(f"- **PASS: {s.get('pass', 0)}**\n")
    lines.append(f"- **FAIL: {s.get('fail', 0)}**\n\n")

    if results.get("warnings"):
        lines.append("## Warnings\n")
        for w in results["warnings"]:
            lines.append(f"- ⚠ {w}\n")
        lines.append("\n")

    # Failure conditions
    lines.append("## Failure Conditions Checked\n")
    lines.append("| Condition | Threshold | Actual | Status |\n")
    lines.append("|-----------|-----------|--------|--------|\n")
    if "part2_topological_charge" in results:
        r = results["part2_topological_charge"]
        lines.append(f"| ℓ ≈ 1 | |ℓ−1| < 0.1 | {r['ell']:.4f} | {r['status']} |\n")
    if "part3_axial_null" in results:
        r = results["part3_axial_null"]
        lines.append(f"| core_ratio < 0.3 | 0.3 | {r['core_ratio']:.4f} | {r['status']} |\n")
    if "part4_power_flow" in results:
        r = results["part4_power_flow"]
        lines.append(f"| mean(I_z_above) > 0 | 0 | {r['mean_Iz_above']:.2e} | {r['status']} |\n")
        lines.append(f"| frac positive > 0.8 | 0.8 | {r['positive_Iz_fraction']:.4f} | {r['status']} |\n")
    if "part5_pml_absorption" in results:
        r = results["part5_pml_absorption"]
        lines.append(f"| decay_ratio < 0.1 | 0.1 | {r['decay_ratio']:.6f} | {r['status']} |\n")

    return "".join(lines)


if __name__ == "__main__":
    main()
