"""
Post-processing / diagnostics helpers for the far-field petri cuboid.

Provides:
  - 2-D field slicing on structured grids (nearest-neighbour on P2 DOFs)
  - Centerline profile extraction
  - Energy integrals (physical vs PML region)
  - Convenience plotting routines that write to disk

Author: Acousto-Tweezers Project
Date: 2026-02-16
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Optional, Tuple

from .config import FarFieldConfig
from .solve_pressure import PressureSolution
from .mesh import CELL_PHYSICAL


# =====================================================================
# Slicing
# =====================================================================

def slice_xy(sol: PressureSolution, z_val: float,
             nx: int = 200, ny: int = 200):
    """
    Return |p| and arg(p) on a regular x-y grid at fixed z.

    Returns (xg, yg, pmag, pphase) where xg/yg are 1-D arrays.
    """
    from scipy.interpolate import NearestNDInterpolator
    coords = sol.coords
    pv = sol.p_values
    interp_re = NearestNDInterpolator(coords, np.real(pv))
    interp_im = NearestNDInterpolator(coords, np.imag(pv))

    cfg = sol.cfg
    xg = np.linspace(0, cfg.Lx, nx)
    yg = np.linspace(0, cfg.Ly, ny)
    X, Y = np.meshgrid(xg, yg)
    Z = np.full_like(X, z_val)
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    pr = interp_re(pts).reshape(X.shape)
    pi = interp_im(pts).reshape(X.shape)
    pc = pr + 1j * pi
    return xg, yg, np.abs(pc), np.angle(pc)


def slice_xz(sol: PressureSolution, y_val: float,
             nx: int = 200, nz: int = 200):
    """
    Return |p| and arg(p) on a regular x-z grid at fixed y.
    """
    from scipy.interpolate import NearestNDInterpolator
    coords = sol.coords
    pv = sol.p_values
    interp_re = NearestNDInterpolator(coords, np.real(pv))
    interp_im = NearestNDInterpolator(coords, np.imag(pv))

    cfg = sol.cfg
    xg = np.linspace(0, cfg.Lx, nx)
    zg = np.linspace(0, cfg.H_total, nz)
    X, Z = np.meshgrid(xg, zg)
    Y = np.full_like(X, y_val)
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    pr = interp_re(pts).reshape(X.shape)
    pi = interp_im(pts).reshape(X.shape)
    pc = pr + 1j * pi
    return xg, zg, np.abs(pc), np.angle(pc)


def centerline_z(sol: PressureSolution, nz: int = 500):
    """
    |p| along the vertical centerline (x=Lx/2, y=Ly/2).
    """
    from scipy.interpolate import NearestNDInterpolator
    coords = sol.coords
    pv = sol.p_values
    interp_re = NearestNDInterpolator(coords, np.real(pv))
    interp_im = NearestNDInterpolator(coords, np.imag(pv))

    cfg = sol.cfg
    zg = np.linspace(0, cfg.H_total, nz)
    cx = cfg.disk_center_x
    cy = cfg.disk_center_y
    pts = np.column_stack([np.full(nz, cx), np.full(nz, cy), zg])
    pr = interp_re(pts)
    pi = interp_im(pts)
    return zg, np.abs(pr + 1j * pi)


def energy_physical_vs_pml(sol: PressureSolution):
    """
    Integrated |p|^2 in physical cells vs PML cells (cheap estimate).

    Uses DOF coordinates + cell tags to classify DOFs.
    Returns dict with keys 'physical', 'pml', 'ratio'.
    """
    coords = sol.coords
    pv = sol.p_values
    p2 = np.abs(pv)**2

    # Classify DOFs by whether they sit in physical region
    cfg = sol.cfg
    t_xy = cfg.t_pml_xy if cfg.pml_enabled else 0.0
    t_z  = cfg.t_pml_z  if cfg.pml_enabled else 0.0
    cx, cy = cfg.disk_center_x, cfg.disk_center_y
    R = cfg.disk_radius

    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    in_pml_x = (x < t_xy) | (x > cfg.Lx - t_xy)
    in_pml_y = (y < t_xy) | (y > cfg.Ly - t_xy)
    r2 = (x - cx)**2 + (y - cy)**2
    in_pml_z = (z < t_z) & (r2 > R**2)
    in_pml = in_pml_x | in_pml_y | in_pml_z

    e_phys = float(np.sum(p2[~in_pml]))
    e_pml  = float(np.sum(p2[in_pml]))
    ratio = e_pml / (e_phys + 1e-30)
    return {"physical": e_phys, "pml": e_pml, "ratio": ratio}


# =====================================================================
# Slice export (npz)
# =====================================================================

def export_slice_xy(sol: PressureSolution, z_val: float, out_path,
                    nx: int = 200, ny: int = 200):
    """Export XY slice to .npz with keys: x, y, p_mag, p_phase, p_complex."""
    xg, yg, pmag, pphi = slice_xy(sol, z_val, nx, ny)
    # Also get complex values
    from scipy.interpolate import NearestNDInterpolator
    coords = sol.coords
    pv = sol.p_values
    interp_re = NearestNDInterpolator(coords, np.real(pv))
    interp_im = NearestNDInterpolator(coords, np.imag(pv))
    X, Y = np.meshgrid(xg, yg)
    Z = np.full_like(X, z_val)
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    pc = (interp_re(pts) + 1j * interp_im(pts)).reshape(X.shape)

    np.savez(out_path, x=xg, y=yg, p_mag=pmag, p_phase=pphi, p_complex=pc,
             z_val=z_val)
    return xg, yg, pmag, pphi


def export_slice_xz(sol: PressureSolution, y_val: float, out_path,
                    nx: int = 200, nz: int = 200):
    """Export XZ slice to .npz with keys: x, z, p_mag, p_phase, p_complex."""
    xg, zg, pmag, pphi = slice_xz(sol, y_val, nx, nz)
    from scipy.interpolate import NearestNDInterpolator
    coords = sol.coords
    pv = sol.p_values
    interp_re = NearestNDInterpolator(coords, np.real(pv))
    interp_im = NearestNDInterpolator(coords, np.imag(pv))
    X, Z = np.meshgrid(xg, zg)
    Y = np.full_like(X, y_val)
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    pc = (interp_re(pts) + 1j * interp_im(pts)).reshape(X.shape)

    np.savez(out_path, x=xg, z=zg, p_mag=pmag, p_phase=pphi, p_complex=pc,
             y_val=y_val)
    return xg, zg, pmag, pphi


# =====================================================================
# Plotting
# =====================================================================

def plot_all_diagnostics(sol: PressureSolution, out_dir: Path, label: str = ""):
    """
    Generate all standard diagnostic plots and save to *out_dir/figs/*.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)
    cfg = sol.cfg
    prefix = f"{label}_" if label else ""

    # ── 1) ZX mid-plane ──────────────────────────────────────────────
    xg, zg, pmag_xz, pphi_xz = slice_xz(sol, cfg.Ly / 2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    im0 = axes[0].pcolormesh(xg * 1e3, zg * 1e3, pmag_xz, shading="auto", cmap="inferno")
    axes[0].set_title("|p| — xz mid-plane")
    axes[0].set_xlabel("x [mm]"); axes[0].set_ylabel("z [mm]")
    plt.colorbar(im0, ax=axes[0], label="Pa")
    # mark PML boundaries
    t_xy = cfg.t_pml_xy * 1e3
    t_z  = cfg.t_pml_z * 1e3
    for ax in axes:
        ax.axvline(t_xy, color="w", ls="--", lw=0.7, alpha=0.6)
        ax.axvline((cfg.Lx - cfg.t_pml_xy) * 1e3, color="w", ls="--", lw=0.7, alpha=0.6)
        ax.axhline(t_z, color="w", ls="--", lw=0.7, alpha=0.6)
        ax.axhline(cfg.H_under * 1e3, color="cyan", ls=":", lw=0.8, label="petri base")

    im1 = axes[1].pcolormesh(xg * 1e3, zg * 1e3, pphi_xz, shading="auto", cmap="twilight")
    axes[1].set_title("arg(p) — xz mid-plane")
    axes[1].set_xlabel("x [mm]"); axes[1].set_ylabel("z [mm]")
    plt.colorbar(im1, ax=axes[1], label="rad")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{prefix}xz_midplane.png", dpi=150)
    plt.close(fig)

    # ── 2) XY slices at multiple z heights ────────────────────────────
    z_levels = {
        "z_above_disk": cfg.t_pml_z + 0.2e-3 if cfg.pml_enabled else 0.2e-3,
        "z_mid_under": cfg.H_under / 2,
        "z_petri_mid": cfg.H_under + cfg.H_top / 2,
        "z_near_top": cfg.H_total - 0.05e-3,
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, (name, zv) in zip(axes.flat, z_levels.items()):
        xg2, yg2, pm2, _ = slice_xy(sol, zv)
        im = ax.pcolormesh(xg2 * 1e3, yg2 * 1e3, pm2, shading="auto", cmap="inferno")
        ax.set_title(f"|p| at z={zv*1e3:.2f} mm  ({name})")
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, label="Pa")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{prefix}xy_slices.png", dpi=150)
    plt.close(fig)

    # ── 3) Centerline |p| vs z ────────────────────────────────────────
    zc, pc = centerline_z(sol)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(zc * 1e3, pc, "k-")
    ax.axvline(cfg.t_pml_z * 1e3, color="r", ls="--", lw=0.7, label="PML-z top")
    ax.axvline(cfg.H_under * 1e3, color="cyan", ls=":", lw=0.8, label="petri base")
    ax.set_xlabel("z [mm]"); ax.set_ylabel("|p| [Pa]")
    ax.set_title("Centerline |p| vs z")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / f"{prefix}centerline_z.png", dpi=150)
    plt.close(fig)

    # ── 4) Disk source amplitude map ──────────────────────────────────
    _plot_disk_source(sol, fig_dir, prefix)

    # ── 5) Energy physical vs PML ─────────────────────────────────────
    en = energy_physical_vs_pml(sol)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(["Physical", "PML"], [en["physical"], en["pml"]], color=["steelblue", "salmon"])
    ax.set_ylabel("Σ|p|²  (DOF sum)")
    ax.set_title(f"Energy partition (PML/phys ratio = {en['ratio']:.3f})")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{prefix}energy_partition.png", dpi=150)
    plt.close(fig)

    return z_levels, en


def _plot_disk_source(sol: PressureSolution, fig_dir: Path, prefix: str):
    """Plot amplitude, phase, real, imag of the disk drive pattern."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .mesh import TAG_BOTTOM_DISK

    V = sol.V
    domain = sol.domain
    facet_tags = sol.facet_tags
    fdim = domain.topology.dim - 1
    cfg = sol.cfg

    disk_facets = facet_tags.indices[facet_tags.values == TAG_BOTTOM_DISK]
    disk_dofs = fem.locate_dofs_topological(V, fdim, disk_facets)
    coords = V.tabulate_dof_coordinates()
    x = coords[disk_dofs, 0]
    y = coords[disk_dofs, 1]

    cx, cy = cfg.disk_center_x, cfg.disk_center_y
    R = cfg.disk_radius

    # Reconstruct drive pattern based on lens mode
    if cfg.lens_drive == "plastic":
        from acoustweezers.physics.acoustics.vortex_lens import (
            PlasticLensConfig, compute_plastic_lens_phase,
            compute_plastic_lens_amplitude, create_plastic_lens_drive,
        )
        lens_cfg = PlasticLensConfig(
            topological_charge=cfg.lens_l,
            focal_length=cfg.lens_focal_length,
            focus_offset_x=cfg.lens_focus_offset_x,
            focus_offset_y=cfg.lens_focus_offset_y,
            c_lens=cfg.lens_c_lens,
            c_water=cfg.c,
            frequency_hz=cfg.frequency_hz,
            aperture_radius=cfg.disk_radius,
            center=None,
            apodization=cfg.lens_apodization,
            apodization_strength=cfg.lens_apodization_strength,
        )
        pattern = create_plastic_lens_drive(
            x, y, lens_cfg, center_x=cx, center_y=cy, verbose=False)
        amp = np.abs(pattern)
        phase = np.angle(pattern)
        drive_label = f"Plastic Lens (l={cfg.lens_l}, f={cfg.lens_focal_length*1e3:.1f}mm)"
    else:
        dx_a = x - cx
        dy_a = y - cy
        r = np.sqrt(dx_a**2 + dy_a**2)
        theta = np.arctan2(dy_a, dx_a)
        inside = r <= R
        amp = np.zeros_like(r)
        if cfg.vortex_apodization == "cosine_taper":
            amp[inside] = 0.5 * (1 + np.cos(np.pi * r[inside] / R))
        else:
            amp[inside] = 1.0
        phase = cfg.vortex_topological_charge * theta
        pattern = amp * np.exp(1j * phase)
        drive_label = f"Ideal Vortex (l={cfg.vortex_topological_charge})"

    # Four-panel plot: amplitude, phase, real, imag
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    sc0 = axes[0, 0].scatter(x * 1e3, y * 1e3, c=amp, s=2, cmap="viridis")
    axes[0, 0].set_title("Amplitude A(r)")
    axes[0, 0].set_aspect("equal")
    plt.colorbar(sc0, ax=axes[0, 0])

    sc1 = axes[0, 1].scatter(x * 1e3, y * 1e3, c=phase, s=2, cmap="twilight")
    axes[0, 1].set_title("Phase [rad]")
    axes[0, 1].set_aspect("equal")
    plt.colorbar(sc1, ax=axes[0, 1])

    sc2 = axes[1, 0].scatter(x * 1e3, y * 1e3, c=np.real(pattern), s=2, cmap="RdBu_r")
    axes[1, 0].set_title("Re(v_n)")
    axes[1, 0].set_aspect("equal")
    plt.colorbar(sc2, ax=axes[1, 0])

    sc3 = axes[1, 1].scatter(x * 1e3, y * 1e3, c=np.imag(pattern), s=2, cmap="RdBu_r")
    axes[1, 1].set_title("Im(v_n)")
    axes[1, 1].set_aspect("equal")
    plt.colorbar(sc3, ax=axes[1, 1])

    for ax in axes.flat:
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")

    fig.suptitle(f"Disk Drive — {drive_label}", fontsize=12)
    fig.tight_layout()

    # Save as combined and individual files
    fig.savefig(fig_dir / f"{prefix}disk_source.png", dpi=150)
    plt.close(fig)

    # Also save individual plots
    for name, data, cmap in [
        ("disk_amplitude", amp, "viridis"),
        ("disk_phase", phase, "twilight"),
        ("disk_real", np.real(pattern), "RdBu_r"),
        ("disk_imag", np.imag(pattern), "RdBu_r"),
    ]:
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        sc = ax2.scatter(x * 1e3, y * 1e3, c=data, s=2, cmap=cmap)
        ax2.set_xlabel("x [mm]"); ax2.set_ylabel("y [mm]")
        ax2.set_title(name.replace("_", " ").title())
        ax2.set_aspect("equal")
        plt.colorbar(sc, ax=ax2)
        fig2.tight_layout()
        fig2.savefig(fig_dir / f"{prefix}{name}.png", dpi=150)
        plt.close(fig2)


# ── need this import for _plot_disk_source to work ──
from dolfinx import fem
