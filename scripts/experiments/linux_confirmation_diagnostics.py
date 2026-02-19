#!/usr/bin/env python3
"""
Linux Confirmation Diagnostics — Comprehensive lens-only propagation checks.

This script implements all diagnostics from task A–C:

  A. Lens-only free-space case (standing sources OFF, petri slab OFF-equivalent)
  B. Five diagnostic categories:
     B1. Disk boundary condition verification (|v_n|, arg, Re, Im)
     B2. Z-stack XY slices (propagation movie) with fixed color scales
     B3. Propagation sanity: standing-ness index + acoustic intensity I_z
     B4. PML reflection diagnostics (PML vs rigid-wall comparison)
     B5. Mesh convergence check (5 vs 6 elem/λ)
  C. Engine fault checks (normal sign, phase convention, PML consistency)

All results saved to: results/LinuxConfirmation/

Usage:
    module load anaconda/3-2025
    conda activate acousto-complex
    python scripts/experiments/linux_confirmation_diagnostics.py
"""
from __future__ import annotations

import gc
import json
import sys
import time
import traceback
import numpy as np
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from acoustweezers.experiments.farfield_petri_cuboid.config import FarFieldConfig, demo_config
from acoustweezers.experiments.farfield_petri_cuboid.solve_pressure import (
    solve_helmholtz, PressureSolution,
)
from acoustweezers.experiments.farfield_petri_cuboid.post import (
    slice_xy, slice_xz, centerline_z, energy_physical_vs_pml,
)
from acoustweezers.experiments.farfield_petri_cuboid.mesh import TAG_BOTTOM_DISK

REPO = Path(__file__).resolve().parents[2]
OUT_ROOT = REPO / "results" / "LinuxConfirmation"


# ═════════════════════════════════════════════════════════════════════
#  Shared helpers
# ═════════════════════════════════════════════════════════════════════

def _interp_complex_xy(sol, z_val, nx=200, ny=200):
    """Return complex pressure on XY grid at given z."""
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
    pc = (interp_re(pts) + 1j * interp_im(pts)).reshape(X.shape)
    return xg, yg, pc


def _interp_complex_xz(sol, y_val, nx=200, nz=300):
    """Return complex pressure on XZ grid at given y."""
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
    pc = (interp_re(pts) + 1j * interp_im(pts)).reshape(X.shape)
    return xg, zg, pc


def _interp_complex_line(sol, x_vals, y_vals, z_vals):
    """Return complex pressure along an arbitrary line of points."""
    from scipy.interpolate import NearestNDInterpolator
    coords = sol.coords
    pv = sol.p_values
    interp_re = NearestNDInterpolator(coords, np.real(pv))
    interp_im = NearestNDInterpolator(coords, np.imag(pv))
    pts = np.column_stack([x_vals, y_vals, z_vals])
    return interp_re(pts) + 1j * interp_im(pts)


def _make_lens_only_config(elements_per_wavelength=5, pml_enabled=True):
    """
    Create a lens-only config: standing sources OFF, petri slab essentially
    transparent (impedance-matched top).

    Uses the same geometry/solver settings as the production demo_config
    but with standing_velocity_amplitude = 0.
    """
    cfg = demo_config(
        standing_velocity_amplitude=0.0,   # ← standing OFF
        elements_per_wavelength=elements_per_wavelength,
        pml_enabled=pml_enabled,
        # Use impedance-matched top to minimize reflections (free-space-like):
        top_impedance_Zrel=1.0,
    )
    return cfg


def _save_fig(fig, path, dpi=150):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {path.name}")


# ═════════════════════════════════════════════════════════════════════
#  B1: Disk Boundary Condition Verification
# ═════════════════════════════════════════════════════════════════════

def step_B1_disk_bc_verification(sol: PressureSolution, out_dir: Path):
    """
    Plot the disk BC pattern: |v_n|, arg(v_n), Re(v_n), Im(v_n).
    Verifies that the lens pattern is what we intend to drive.
    """
    print("\n  ── B1: Disk Boundary Condition Verification ──")
    fig_dir = out_dir / "B1_disk_bc"
    fig_dir.mkdir(parents=True, exist_ok=True)

    from dolfinx import fem
    V = sol.V
    domain = sol.domain
    facet_tags = sol.facet_tags
    cfg = sol.cfg
    fdim = domain.topology.dim - 1

    disk_facets = facet_tags.indices[facet_tags.values == TAG_BOTTOM_DISK]
    disk_dofs = fem.locate_dofs_topological(V, fdim, disk_facets)
    coords = V.tabulate_dof_coordinates()
    x = coords[disk_dofs, 0]
    y = coords[disk_dofs, 1]
    cx, cy = cfg.disk_center_x, cfg.disk_center_y

    # Reconstruct the drive pattern (same logic as solver)
    if cfg.lens_drive == "plastic":
        from acoustweezers.physics.acoustics.vortex_lens import (
            PlasticLensConfig, create_plastic_lens_drive,
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
            x, y, lens_cfg, center_x=cx, center_y=cy, verbose=True)
    else:
        dx_a = x - cx
        dy_a = y - cy
        r = np.sqrt(dx_a**2 + dy_a**2)
        theta = np.arctan2(dy_a, dx_a)
        R = cfg.disk_radius
        inside = r <= R
        amp = np.zeros_like(r)
        if cfg.vortex_apodization == "cosine_taper":
            amp[inside] = 0.5 * (1 + np.cos(np.pi * r[inside] / R))
        else:
            amp[inside] = 1.0
        ell = cfg.vortex_topological_charge
        pattern = amp * np.exp(1j * ell * theta)

    # Scale to physical velocity (multiply by V_disk as solver does)
    V_disk = cfg.disk_velocity_amplitude
    v_n = pattern * V_disk  # physical v_n [m/s]

    amp = np.abs(v_n)
    phase = np.angle(v_n)
    re_part = np.real(v_n)
    im_part = np.imag(v_n)

    # ── Four-panel combined plot ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    sc0 = axes[0, 0].scatter((x - cx) * 1e3, (y - cy) * 1e3, c=amp * 1e6,
                              s=4, cmap="viridis")
    axes[0, 0].set_title("|v_n(x,y)| [μm/s]")
    plt.colorbar(sc0, ax=axes[0, 0])

    sc1 = axes[0, 1].scatter((x - cx) * 1e3, (y - cy) * 1e3, c=phase,
                              s=4, cmap="twilight", vmin=-np.pi, vmax=np.pi)
    axes[0, 1].set_title("arg(v_n(x,y)) [rad]")
    plt.colorbar(sc1, ax=axes[0, 1])

    vmax_re = max(abs(re_part.max()), abs(re_part.min())) * 1e6
    sc2 = axes[1, 0].scatter((x - cx) * 1e3, (y - cy) * 1e3, c=re_part * 1e6,
                              s=4, cmap="RdBu_r", vmin=-vmax_re, vmax=vmax_re)
    axes[1, 0].set_title("Re(v_n(x,y)) [μm/s]")
    plt.colorbar(sc2, ax=axes[1, 0])

    vmax_im = max(abs(im_part.max()), abs(im_part.min())) * 1e6
    sc3 = axes[1, 1].scatter((x - cx) * 1e3, (y - cy) * 1e3, c=im_part * 1e6,
                              s=4, cmap="RdBu_r", vmin=-vmax_im, vmax=vmax_im)
    axes[1, 1].set_title("Im(v_n(x,y)) [μm/s]")
    plt.colorbar(sc3, ax=axes[1, 1])

    for ax in axes.flat:
        ax.set_xlabel("x − cx [mm]")
        ax.set_ylabel("y − cy [mm]")
        ax.set_aspect("equal")
        # Draw disk boundary circle
        theta_c = np.linspace(0, 2 * np.pi, 200)
        ax.plot(cfg.disk_radius * 1e3 * np.cos(theta_c),
                cfg.disk_radius * 1e3 * np.sin(theta_c),
                "k--", lw=0.8, alpha=0.5)

    drive_label = (f"Plastic Lens l={cfg.lens_l}" if cfg.lens_drive == "plastic"
                   else f"Ideal Vortex l={cfg.vortex_topological_charge}")
    fig.suptitle(f"B1: Disk BC Verification — {drive_label}", fontsize=14)
    fig.tight_layout()
    _save_fig(fig, fig_dir / "disk_bc_4panel.png")

    # ── Phase winding check (radial profile) ──
    r_from_center = np.sqrt((x - cx)**2 + (y - cy)**2)
    theta_from_center = np.arctan2(y - cy, x - cx)
    # Select DOFs within the active region
    active = amp > 1e-10 * amp.max()
    if np.any(active):
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        sc = ax2.scatter(theta_from_center[active], phase[active],
                         c=r_from_center[active] * 1e3, s=2, cmap="plasma", alpha=0.7)
        ax2.set_xlabel("θ from center [rad]")
        ax2.set_ylabel("arg(v_n) [rad]")
        ax2.set_title("Phase winding check: should show ~2π wrap for l=1")
        plt.colorbar(sc, ax=ax2, label="r [mm]")
        fig2.tight_layout()
        _save_fig(fig2, fig_dir / "phase_winding.png")

    # ── Radial amplitude profile ──
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    sort_idx = np.argsort(r_from_center)
    ax3.plot(r_from_center[sort_idx] * 1e3, amp[sort_idx] * 1e6, "k.", ms=1, alpha=0.3)
    ax3.set_xlabel("r from center [mm]")
    ax3.set_ylabel("|v_n| [μm/s]")
    ax3.set_title("Radial amplitude profile (should be ring-like for plastic lens)")
    ax3.axvline(cfg.disk_radius * 1e3, color="r", ls="--", label="R_disk")
    ax3.legend()
    fig3.tight_layout()
    _save_fig(fig3, fig_dir / "radial_amplitude.png")

    # Quantitative checks
    checks = {
        "n_disk_dofs": len(disk_dofs),
        "n_active_dofs": int(np.sum(active)),
        "max_amplitude_um_s": float(amp.max() * 1e6),
        "min_active_amplitude_um_s": float(amp[active].min() * 1e6) if np.any(active) else 0,
        "phase_range_rad": float(phase[active].max() - phase[active].min()) if np.any(active) else 0,
    }
    print(f"    Disk DOFs: {checks['n_disk_dofs']}, Active: {checks['n_active_dofs']}")
    print(f"    Max |v_n|: {checks['max_amplitude_um_s']:.3f} μm/s")
    print(f"    Phase range: {checks['phase_range_rad']:.3f} rad (expect ~6.28 for l=1)")
    return checks


# ═════════════════════════════════════════════════════════════════════
#  B2: Z-Stack XY Slices (propagation movie)
# ═════════════════════════════════════════════════════════════════════

def step_B2_zstack_xy_slices(sol: PressureSolution, out_dir: Path, n_planes=15):
    """
    XY slices at multiple z-levels with fixed color limits.
    Generates individual PNGs + summary montage + scalar metrics per plane.
    """
    print("\n  ── B2: Z-Stack XY Slices ──")
    fig_dir = out_dir / "B2_zstack"
    fig_dir.mkdir(parents=True, exist_ok=True)

    cfg = sol.cfg
    z_bot = cfg.t_pml_z if cfg.pml_enabled else 0.0
    z_top = cfg.H_total - 0.05e-3  # just below top face

    z_levels = np.linspace(z_bot + 0.1e-3, z_top, n_planes)

    # First pass: compute all slices and find global color scale
    slices = []
    metrics = []
    for iz, zv in enumerate(z_levels):
        xg, yg, pc = _interp_complex_xy(sol, zv, nx=150, ny=150)
        pmag = np.abs(pc)
        pphase = np.angle(pc)
        slices.append((xg, yg, pmag, pphase, pc))

        # Compute ROI metrics (physical region only)
        x_lo, x_hi = cfg.physical_x_range
        y_lo, y_hi = cfg.physical_y_range
        ix_lo = np.searchsorted(xg, x_lo)
        ix_hi = np.searchsorted(xg, x_hi)
        iy_lo = np.searchsorted(yg, y_lo)
        iy_hi = np.searchsorted(yg, y_hi)
        roi = pmag[iy_lo:iy_hi, ix_lo:ix_hi]
        metrics.append({
            "z_mm": float(zv * 1e3),
            "max_p_Pa": float(roi.max()) if roi.size > 0 else 0,
            "mean_p_Pa": float(roi.mean()) if roi.size > 0 else 0,
        })

    global_vmax = max(m["max_p_Pa"] for m in metrics) * 1.05

    # Second pass: plot with fixed color scale
    for iz, (zv, (xg, yg, pmag, pphase, pc)) in enumerate(zip(z_levels, slices)):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        im0 = axes[0].pcolormesh(xg * 1e3, yg * 1e3, pmag,
                                  shading="auto", cmap="inferno",
                                  vmin=0, vmax=global_vmax)
        axes[0].set_title(f"|p| at z={zv*1e3:.2f} mm")
        plt.colorbar(im0, ax=axes[0], label="Pa")

        im1 = axes[1].pcolormesh(xg * 1e3, yg * 1e3, pphase,
                                  shading="auto", cmap="twilight",
                                  vmin=-np.pi, vmax=np.pi)
        axes[1].set_title(f"arg(p) at z={zv*1e3:.2f} mm")
        plt.colorbar(im1, ax=axes[1], label="rad")

        for ax in axes:
            ax.set_xlabel("x [mm]")
            ax.set_ylabel("y [mm]")
            ax.set_aspect("equal")
            # Mark PML boundaries
            if cfg.pml_enabled:
                t = cfg.t_pml_xy * 1e3
                ax.axvline(t, color="w", ls="--", lw=0.5, alpha=0.5)
                ax.axvline((cfg.Lx - cfg.t_pml_xy) * 1e3, color="w", ls="--", lw=0.5, alpha=0.5)
                ax.axhline(t, color="w", ls="--", lw=0.5, alpha=0.5)
                ax.axhline((cfg.Ly - cfg.t_pml_xy) * 1e3, color="w", ls="--", lw=0.5, alpha=0.5)

        fig.suptitle(f"Plane {iz+1}/{n_planes}  z = {zv*1e3:.2f} mm", fontsize=12)
        fig.tight_layout()
        _save_fig(fig, fig_dir / f"xy_z{iz:02d}_{zv*1e3:.2f}mm.png")

    # Scalar metrics plot
    z_arr = np.array([m["z_mm"] for m in metrics])
    max_arr = np.array([m["max_p_Pa"] for m in metrics])
    mean_arr = np.array([m["mean_p_Pa"] for m in metrics])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(z_arr, max_arr, "bo-", label="max |p| in ROI")
    ax.plot(z_arr, mean_arr, "rs-", label="mean |p| in ROI")
    ax.set_xlabel("z [mm]")
    ax.set_ylabel("|p| [Pa]")
    ax.set_title("B2: ROI pressure vs z (propagation envelope)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, fig_dir / "roi_pressure_vs_z.png")

    # Save metrics
    import csv
    csv_path = fig_dir / "zstack_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["z_mm", "max_p_Pa", "mean_p_Pa"])
        w.writeheader()
        w.writerows(metrics)
    print(f"    Saved {n_planes} XY slices + metrics CSV")

    return metrics


# ═════════════════════════════════════════════════════════════════════
#  B3: Propagation Sanity — Standing-ness + Acoustic Intensity
# ═════════════════════════════════════════════════════════════════════

def step_B3_propagation_sanity(sol: PressureSolution, out_dir: Path):
    """
    B3a: Standing-ness index along centerline.
    B3b: Acoustic intensity I_z on XY planes.
    """
    print("\n  ── B3: Propagation Sanity Metrics ──")
    fig_dir = out_dir / "B3_propagation"
    fig_dir.mkdir(parents=True, exist_ok=True)

    cfg = sol.cfg

    # ── B3a: Standing-ness index ──
    # Sample complex pressure along centerline
    nz = 500
    zg = np.linspace(0, cfg.H_total, nz)
    cx, cy = cfg.disk_center_x, cfg.disk_center_y
    p_center = _interp_complex_line(
        sol,
        np.full(nz, cx), np.full(nz, cy), zg
    )
    p_mag = np.abs(p_center)

    # Standing-ness: ratio of min/max in sliding windows
    # For a pure travelling wave, |p| is roughly constant;
    # for a standing wave, it oscillates between 0 and 2*A.
    window = max(3, int(0.5 * cfg.wavelength / (cfg.H_total / nz)))  # ~half wavelength
    standing_index = np.zeros(nz)
    for i in range(window, nz - window):
        local = p_mag[i - window:i + window + 1]
        if local.max() > 1e-10:
            standing_index[i] = 1.0 - local.min() / local.max()
        else:
            standing_index[i] = 0.0

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    axes[0].plot(zg * 1e3, p_mag, "k-", lw=0.8)
    axes[0].set_ylabel("|p| [Pa]")
    axes[0].set_title("B3a: Centerline |p|(z) — Lens Only")
    if cfg.pml_enabled:
        axes[0].axvline(cfg.t_pml_z * 1e3, color="r", ls="--", lw=0.7, label="PML-z top")
    axes[0].axvline(cfg.H_under * 1e3, color="cyan", ls=":", lw=0.8, label="petri base")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(zg * 1e3, standing_index, "b-", lw=0.8)
    axes[1].set_xlabel("z [mm]")
    axes[1].set_ylabel("Standing-ness index")
    axes[1].set_title("Standing-ness (0=travelling, 1=pure standing)")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    _save_fig(fig, fig_dir / "centerline_standingness.png")

    # Global standing-ness in physical region
    phys_mask = zg >= (cfg.t_pml_z if cfg.pml_enabled else 0)
    mean_standing = float(np.mean(standing_index[phys_mask & (standing_index > 0)]))
    print(f"    Mean standing-ness (physical): {mean_standing:.3f}")

    # ── B3b: Acoustic intensity I_z ──
    # I_z ∝ Re(p · conj(v_z)) where v_z = -(1/(iωρ)) ∂p/∂z
    # Approximate ∂p/∂z via finite differences on XZ mid-plane

    xg, zg_xz, pc_xz = _interp_complex_xz(sol, cfg.Ly / 2, nx=150, nz=300)
    dz = zg_xz[1] - zg_xz[0]
    dp_dz = np.gradient(pc_xz, dz, axis=0)

    omega = cfg.omega
    rho = cfg.rho
    # v_z = 1/(i ω ρ) * ∂p/∂z   [linearised Euler, exp(-iωt)]
    v_z = 1.0 / (1j * omega * rho) * dp_dz
    # I_z = Re(p * conj(v_z))
    I_z = np.real(pc_xz * np.conj(v_z))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    im0 = axes[0].pcolormesh(xg * 1e3, zg_xz * 1e3, np.abs(pc_xz),
                              shading="auto", cmap="inferno")
    axes[0].set_title("|p| — XZ mid-plane")
    plt.colorbar(im0, ax=axes[0], label="Pa")

    # I_z with diverging colormap
    Iz_max = np.percentile(np.abs(I_z), 99)
    im1 = axes[1].pcolormesh(xg * 1e3, zg_xz * 1e3, I_z,
                              shading="auto", cmap="RdBu_r",
                              vmin=-Iz_max, vmax=Iz_max)
    axes[1].set_title("$I_z$ (time-averaged power flow, z-component)")
    plt.colorbar(im1, ax=axes[1], label="W/m²")

    for ax in axes:
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("z [mm]")
        if cfg.pml_enabled:
            t_xy = cfg.t_pml_xy * 1e3
            t_z = cfg.t_pml_z * 1e3
            ax.axvline(t_xy, color="w", ls="--", lw=0.5, alpha=0.5)
            ax.axvline((cfg.Lx - cfg.t_pml_xy) * 1e3, color="w", ls="--", lw=0.5, alpha=0.5)
            ax.axhline(t_z, color="w", ls="--", lw=0.5, alpha=0.5)

    fig.tight_layout()
    _save_fig(fig, fig_dir / "xz_intensity_Iz.png")

    # I_z on XY planes at several z-levels
    z_probe = [cfg.t_pml_z + 0.3e-3, cfg.H_under * 0.5,
               cfg.H_under * 0.75, cfg.H_under - 0.1e-3]
    z_probe = [z for z in z_probe if 0 < z < cfg.H_total]

    fig, axes = plt.subplots(1, len(z_probe), figsize=(5 * len(z_probe), 4))
    if len(z_probe) == 1:
        axes = [axes]

    for ax, zv in zip(axes, z_probe):
        xg2, yg2, pc2 = _interp_complex_xy(sol, zv, nx=120, ny=120)
        # Compute I_z by finite difference in z
        dz_small = 0.02e-3  # 20 μm step
        _, _, pc2_up = _interp_complex_xy(sol, zv + dz_small, nx=120, ny=120)
        dp_dz_xy = (pc2_up - pc2) / dz_small
        v_z_xy = 1.0 / (1j * omega * rho) * dp_dz_xy
        Iz_xy = np.real(pc2 * np.conj(v_z_xy))

        Iz_max_xy = np.percentile(np.abs(Iz_xy), 98)
        if Iz_max_xy < 1e-20:
            Iz_max_xy = 1.0
        im = ax.pcolormesh(xg2 * 1e3, yg2 * 1e3, Iz_xy,
                           shading="auto", cmap="RdBu_r",
                           vmin=-Iz_max_xy, vmax=Iz_max_xy)
        ax.set_title(f"$I_z$ at z={zv*1e3:.2f} mm")
        ax.set_aspect("equal")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        plt.colorbar(im, ax=ax, label="W/m²")

    fig.suptitle("B3b: I_z (acoustic intensity z-component) on XY planes", fontsize=12)
    fig.tight_layout()
    _save_fig(fig, fig_dir / "Iz_xy_planes.png")

    # Mean I_z in disk ROI vs z
    nz_flux = 50
    z_flux = np.linspace(cfg.t_pml_z + 0.1e-3 if cfg.pml_enabled else 0.1e-3,
                          cfg.H_total - 0.1e-3, nz_flux)
    mean_Iz = np.zeros(nz_flux)
    roi_r = cfg.disk_radius * 0.8

    for iz, zv in enumerate(z_flux):
        xg3, yg3, pc3 = _interp_complex_xy(sol, zv, nx=60, ny=60)
        _, _, pc3_up = _interp_complex_xy(sol, zv + 0.02e-3, nx=60, ny=60)
        dp_dz3 = (pc3_up - pc3) / 0.02e-3
        v_z3 = 1.0 / (1j * omega * rho) * dp_dz3
        Iz3 = np.real(pc3 * np.conj(v_z3))
        X3, Y3 = np.meshgrid(xg3, yg3)
        r3 = np.sqrt((X3 - cfg.disk_center_x)**2 + (Y3 - cfg.disk_center_y)**2)
        mask = r3 <= roi_r
        if np.any(mask):
            mean_Iz[iz] = float(np.mean(Iz3[mask]))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(z_flux * 1e3, mean_Iz, "b-o", ms=3)
    ax.axhline(0, color="k", ls="-", lw=0.5)
    ax.set_xlabel("z [mm]")
    ax.set_ylabel("Mean $I_z$ in beam ROI [W/m²]")
    ax.set_title("B3b: Net z-directed power flow vs depth")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, fig_dir / "mean_Iz_vs_z.png")

    # Fraction of positive I_z (one-way-ness metric)
    positive_frac = float(np.mean(mean_Iz > 0))
    print(f"    Fraction of z-planes with positive mean I_z: {positive_frac:.2f}")

    return {
        "mean_standing_index": mean_standing,
        "positive_Iz_fraction": positive_frac,
    }

# ═════════════════════════════════════════════════════════════════════
#  B4: PML Reflection Diagnostics
# ═════════════════════════════════════════════════════════════════════

def step_B4_pml_reflection(sol_pml: PressureSolution, out_dir: Path):
    """
    Compare PML lens-only vs rigid-wall lens-only.
    Check |p| on thin shell just inside PML boundaries.
    """
    print("\n  ── B4: PML Reflection Diagnostics ──")
    fig_dir = out_dir / "B4_pml_reflection"
    fig_dir.mkdir(parents=True, exist_ok=True)

    cfg_pml = sol_pml.cfg

    # Solve rigid-wall reference (PML OFF)
    print("    Solving rigid-wall reference (PML OFF) …")
    cfg_rigid = _make_lens_only_config(
        elements_per_wavelength=cfg_pml.elements_per_wavelength,
        pml_enabled=False,
    )
    sol_rigid = solve_helmholtz(cfg_rigid, verbose=True)

    # Compare centerlines
    zc_pml, pc_pml = centerline_z(sol_pml)
    zc_rig, pc_rig = centerline_z(sol_rigid)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(zc_pml * 1e3, pc_pml, "b-", label="PML", lw=1)
    ax.plot(zc_rig * 1e3, pc_rig, "r-", label="Rigid walls", lw=1, alpha=0.8)
    ax.set_xlabel("z [mm]")
    ax.set_ylabel("|p| [Pa]")
    ax.set_title("B4: PML vs Rigid — Centerline |p|(z)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, fig_dir / "centerline_pml_vs_rigid.png")

    # Compare XZ mid-planes side by side
    xg_p, zg_p, pc_p = _interp_complex_xz(sol_pml, cfg_pml.Ly / 2)
    xg_r, zg_r, pc_r = _interp_complex_xz(sol_rigid, cfg_rigid.Ly / 2)

    vmax = max(np.abs(pc_p).max(), np.abs(pc_r).max())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    im0 = axes[0].pcolormesh(xg_p * 1e3, zg_p * 1e3, np.abs(pc_p),
                              shading="auto", cmap="inferno", vmin=0, vmax=vmax)
    axes[0].set_title("PML: |p| XZ mid-plane")
    plt.colorbar(im0, ax=axes[0], label="Pa")

    im1 = axes[1].pcolormesh(xg_r * 1e3, zg_r * 1e3, np.abs(pc_r),
                              shading="auto", cmap="inferno", vmin=0, vmax=vmax)
    axes[1].set_title("Rigid: |p| XZ mid-plane")
    plt.colorbar(im1, ax=axes[1], label="Pa")

    for ax in axes:
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("z [mm]")

    fig.suptitle("B4: PML vs Rigid Wall Comparison", fontsize=12)
    fig.tight_layout()
    _save_fig(fig, fig_dir / "xz_pml_vs_rigid.png")

    # |p| on thin shell just inside PML boundary
    # Sample at x = t_pml + 0.02mm (near left PML boundary)
    if cfg_pml.pml_enabled:
        shell_x = cfg_pml.t_pml_xy + 0.02e-3
        nz_shell = 100
        ny_shell = 60
        yg_s = np.linspace(0, cfg_pml.Ly, ny_shell)
        zg_s = np.linspace(0, cfg_pml.H_total, nz_shell)
        Ys, Zs = np.meshgrid(yg_s, zg_s)
        Xs = np.full_like(Ys, shell_x)

        p_shell_pml = _interp_complex_line(
            sol_pml, Xs.ravel(), Ys.ravel(), Zs.ravel()
        ).reshape(Ys.shape)
        p_shell_rigid = _interp_complex_line(
            sol_rigid, Xs.ravel(), Ys.ravel(), Zs.ravel()
        ).reshape(Ys.shape)

        vmax_shell = max(np.abs(p_shell_pml).max(), np.abs(p_shell_rigid).max())

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        im0 = axes[0].pcolormesh(yg_s * 1e3, zg_s * 1e3, np.abs(p_shell_pml),
                                  shading="auto", cmap="inferno", vmin=0, vmax=vmax_shell)
        axes[0].set_title(f"PML: |p| at x={shell_x*1e3:.2f} mm (near boundary)")
        plt.colorbar(im0, ax=axes[0], label="Pa")

        im1 = axes[1].pcolormesh(yg_s * 1e3, zg_s * 1e3, np.abs(p_shell_rigid),
                                  shading="auto", cmap="inferno", vmin=0, vmax=vmax_shell)
        axes[1].set_title(f"Rigid: |p| at x={shell_x*1e3:.2f} mm")
        plt.colorbar(im1, ax=axes[1], label="Pa")

        for ax in axes:
            ax.set_xlabel("y [mm]")
            ax.set_ylabel("z [mm]")

        fig.suptitle("B4: Boundary shell |p| (just inside PML layer)", fontsize=12)
        fig.tight_layout()
        _save_fig(fig, fig_dir / "boundary_shell_pml_vs_rigid.png")

    # Energy comparison
    en_pml = energy_physical_vs_pml(sol_pml)
    en_rig = energy_physical_vs_pml(sol_rigid)

    print(f"    PML:   physical energy={en_pml['physical']:.2e}, PML energy={en_pml['pml']:.2e}, ratio={en_pml['ratio']:.4f}")
    print(f"    Rigid: physical energy={en_rig['physical']:.2e}")
    print(f"    Rigid/PML physical ratio: {en_rig['physical']/en_pml['physical']:.3f}")

    del sol_rigid
    gc.collect()

    return {
        "pml_energy_ratio": en_pml["ratio"],
        "rigid_vs_pml_phys_ratio": en_rig["physical"] / en_pml["physical"],
    }


# ═════════════════════════════════════════════════════════════════════
#  B5: Mesh Convergence Check
# ═════════════════════════════════════════════════════════════════════

def step_B5_mesh_convergence(out_dir: Path):
    """
    Compare lens-only at 5 elem/λ vs 6 elem/λ.
    Check that beam shape doesn't shift materially.
    """
    print("\n  ── B5: Mesh Convergence Check ──")
    fig_dir = out_dir / "B5_mesh_convergence"
    fig_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    centerlines = {}
    xy_slices = {}

    for epw in [5, 6]:
        print(f"    Solving with {epw} elem/λ …")
        cfg = _make_lens_only_config(elements_per_wavelength=epw)
        sol = solve_helmholtz(cfg, verbose=True)
        results[epw] = {
            "dofs": sol.dofs,
            "max_p": sol.max_pressure,
            "solver_time": sol.solver_time,
        }

        # Centerline
        zc, pc = centerline_z(sol, nz=500)
        centerlines[epw] = (zc, pc)

        # XY slice at trapping plane (mid of under-bath)
        z_trap = cfg.H_under * 0.7  # 70% up the under-bath
        xg, yg, pmag, _ = slice_xy(sol, z_trap, nx=150, ny=150)
        xy_slices[epw] = (xg, yg, pmag, z_trap)

        del sol
        gc.collect()

    # Plot centerline comparison
    fig, ax = plt.subplots(figsize=(10, 4))
    for epw, (zc, pc) in centerlines.items():
        ax.plot(zc * 1e3, pc, label=f"{epw} elem/λ  (DOFs={results[epw]['dofs']})", lw=1)
    ax.set_xlabel("z [mm]")
    ax.set_ylabel("|p| [Pa]")
    ax.set_title("B5: Mesh Convergence — Centerline |p|(z)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, fig_dir / "centerline_convergence.png")

    # Relative difference
    zc5, pc5 = centerlines[5]
    zc6, pc6 = centerlines[6]
    # Interpolate to common grid if needed
    from scipy.interpolate import interp1d
    pc5_interp = interp1d(zc5, pc5, fill_value="extrapolate")(zc6)
    rel_diff = np.abs(pc5_interp - pc6) / (pc6 + 1e-10)
    max_rel_diff = float(np.max(rel_diff[pc6 > 0.1 * pc6.max()]))

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(zc6 * 1e3, rel_diff * 100, "r-", lw=0.8)
    ax.set_xlabel("z [mm]")
    ax.set_ylabel("Relative difference [%]")
    ax.set_title(f"B5: Relative |p| difference (5 vs 6 elem/λ), max={max_rel_diff*100:.1f}%")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, min(100, max_rel_diff * 200))
    fig.tight_layout()
    _save_fig(fig, fig_dir / "centerline_relative_diff.png")

    # Plot XY slice comparison at trapping plane
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    vmax = max(xy_slices[5][2].max(), xy_slices[6][2].max())
    for ax, epw in zip(axes, [5, 6]):
        xg, yg, pmag, z_trap = xy_slices[epw]
        im = ax.pcolormesh(xg * 1e3, yg * 1e3, pmag,
                           shading="auto", cmap="inferno", vmin=0, vmax=vmax)
        ax.set_title(f"{epw} elem/λ  |p| at z={z_trap*1e3:.2f} mm")
        ax.set_aspect("equal")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        plt.colorbar(im, ax=ax, label="Pa")

    fig.suptitle("B5: Mesh Convergence — XY slice at trapping plane", fontsize=12)
    fig.tight_layout()
    _save_fig(fig, fig_dir / "xy_slice_convergence.png")

    print(f"    5 elem/λ: DOFs={results[5]['dofs']}, max|p|={results[5]['max_p']:.2f} Pa")
    print(f"    6 elem/λ: DOFs={results[6]['dofs']}, max|p|={results[6]['max_p']:.2f} Pa")
    print(f"    Max relative diff (centerline, where |p|>10% of peak): {max_rel_diff*100:.1f}%")

    return {
        "results": results,
        "max_relative_diff_pct": max_rel_diff * 100,
    }


# ═════════════════════════════════════════════════════════════════════
#  C: Engine Fault Checks
# ═════════════════════════════════════════════════════════════════════

def step_C_engine_checks(sol: PressureSolution, out_dir: Path):
    """
    Automated checks for common FEniCSx/PETSc Helmholtz + PML faults.
    """
    print("\n  ── C: Engine Fault Checks ──")
    fig_dir = out_dir / "C_engine_checks"
    fig_dir.mkdir(parents=True, exist_ok=True)

    cfg = sol.cfg
    checks = {}

    # C1: Neumann sign convention check
    # If the normal is wrong, the beam propagates downward instead of upward.
    # Check: pressure should be higher above the disk than below.
    nz = 500
    zg = np.linspace(0, cfg.H_total, nz)
    cx, cy = cfg.disk_center_x, cfg.disk_center_y
    p_center = _interp_complex_line(
        sol, np.full(nz, cx), np.full(nz, cy), zg)

    z_disk = cfg.t_pml_z if cfg.pml_enabled else 0.0
    above = (zg > z_disk + 0.1e-3) & (zg < cfg.H_under)
    below = (zg >= 0) & (zg < z_disk - 0.1e-3) if z_disk > 0.2e-3 else np.zeros(nz, dtype=bool)

    p_above = np.abs(p_center[above]).mean() if np.any(above) else 0
    p_below = np.abs(p_center[below]).mean() if np.any(below) else 0

    checks["C1_normal_sign"] = "PASS" if p_above > p_below or p_below == 0 else "FAIL"
    print(f"    C1 Normal sign: mean|p| above disk={p_above:.2f}, below={p_below:.2f} → {checks['C1_normal_sign']}")

    # C2: Phase convention check
    # For a focused vortex lens with l=1, the phase at the disk should wind by 2π.
    # Already checked in B1, but verify the actual solved field shows a phase singularity
    # near the axis just above the disk.
    z_check = z_disk + 0.5e-3
    xg2, yg2, pc2 = _interp_complex_xy(sol, z_check, nx=100, ny=100)
    phase_xy = np.angle(pc2)

    # Check for phase singularity: the gradient of phase should be large near center
    X2, Y2 = np.meshgrid(xg2, yg2)
    r2 = np.sqrt((X2 - cx)**2 + (Y2 - cy)**2)
    # Near the vortex core, magnitude should be low
    core_mask = r2 < 0.3e-3
    outer_mask = (r2 > 0.3e-3) & (r2 < cfg.disk_radius)
    core_mag = np.abs(pc2[core_mask]).mean() if np.any(core_mask) else 0
    outer_mag = np.abs(pc2[outer_mask]).mean() if np.any(outer_mask) else 1
    null_depth = core_mag / outer_mag if outer_mag > 0 else 999

    checks["C2_vortex_null"] = "PASS" if null_depth < 0.5 else "WARN"
    print(f"    C2 Vortex null: core/outer ratio = {null_depth:.3f} (expect < 0.5) → {checks['C2_vortex_null']}")

    # C3: PML consistency check
    # In PML region, |p| should decay into the absorbing layer.
    # Sample along x at fixed y=Ly/2, z=H_under/2 — through the left PML.
    if cfg.pml_enabled:
        nx_pml = 100
        x_probe = np.linspace(0, cfg.t_pml_xy * 2, nx_pml)
        z_mid = cfg.H_under / 2
        p_pml_line = _interp_complex_line(
            sol,
            x_probe, np.full(nx_pml, cfg.Ly / 2), np.full(nx_pml, z_mid)
        )
        p_pml_mag = np.abs(p_pml_line)
        # Inside PML (x < t_pml_xy), field should generally decrease toward x=0
        in_pml = x_probe < cfg.t_pml_xy
        if np.sum(in_pml) > 5:
            p_boundary = p_pml_mag[in_pml][0]  # at x=0
            p_inner = p_pml_mag[in_pml][-1]     # at PML/physical interface
            decay_ratio = p_boundary / (p_inner + 1e-10)
            checks["C3_pml_decay"] = "PASS" if decay_ratio < 0.3 else "WARN"
            print(f"    C3 PML decay: |p| at boundary/interface = {decay_ratio:.3f} (expect < 0.3) → {checks['C3_pml_decay']}")

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(x_probe * 1e3, p_pml_mag, "b-")
            ax.axvline(cfg.t_pml_xy * 1e3, color="r", ls="--", label="PML boundary")
            ax.set_xlabel("x [mm]")
            ax.set_ylabel("|p| [Pa]")
            ax.set_title("C3: PML decay profile (left side)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            _save_fig(fig, fig_dir / "pml_decay_profile.png")
        else:
            checks["C3_pml_decay"] = "SKIP"

    # C4: PML σ field audit
    sigma_x = sol.sigma_x
    sigma_y = sol.sigma_y
    sigma_z = sol.sigma_z
    sx_max = float(np.max(np.abs(sigma_x.x.array)))
    sy_max = float(np.max(np.abs(sigma_y.x.array)))
    sz_max = float(np.max(np.abs(sigma_z.x.array)))
    sigma_expected = cfg.sigma_max if cfg.pml_enabled else 0.0

    checks["C4_sigma_x_max"] = float(sx_max)
    checks["C4_sigma_y_max"] = float(sy_max)
    checks["C4_sigma_z_max"] = float(sz_max)
    checks["C4_sigma_expected"] = float(sigma_expected)

    if cfg.pml_enabled:
        tol = 0.1
        sx_ok = abs(sx_max - sigma_expected) / sigma_expected < tol if sigma_expected > 0 else True
        sy_ok = abs(sy_max - sigma_expected) / sigma_expected < tol if sigma_expected > 0 else True
        checks["C4_sigma_audit"] = "PASS" if (sx_ok and sy_ok) else "WARN"
        print(f"    C4 σ audit: σ_x_max={sx_max:.2e}, σ_y_max={sy_max:.2e}, σ_z_max={sz_max:.2e}  "
              f"(expected ~{sigma_expected:.2e}) → {checks['C4_sigma_audit']}")
    else:
        checks["C4_sigma_audit"] = "N/A"

    # C5: Under-resolution check
    # Measure effective elements per wavelength based on DOFs and physical volume
    phys_vol = ((cfg.Lx - 2 * cfg.t_pml_xy) *
                (cfg.Ly - 2 * cfg.t_pml_xy) *
                cfg.H_total)
    wl3 = cfg.wavelength ** 3
    n_wavelength_cubes = phys_vol / wl3
    dofs = sol.dofs
    dofs_per_wl3 = dofs / n_wavelength_cubes if n_wavelength_cubes > 0 else 0
    checks["C5_dofs_per_wl_cube"] = float(dofs_per_wl3)
    # For P2 elements with N elem/λ per dimension: expect ~(2N)^3 / 6 DOFs per λ³ (tet)
    expected_tet = (2 * cfg.elements_per_wavelength) ** 3 / 6
    checks["C5_expected_dofs_per_wl3"] = float(expected_tet)
    checks["C5_resolution"] = "OK" if dofs_per_wl3 > expected_tet * 0.5 else "LOW"

    print(f"    C5 Resolution: DOFs/λ³ ≈ {dofs_per_wl3:.0f} (expected ~{expected_tet:.0f}) → {checks['C5_resolution']}")

    # Save summary
    with open(fig_dir / "engine_checks.json", "w") as f:
        json.dump(checks, f, indent=2)

    return checks


# ═════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════

def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*70}")
    print(f"  LINUX CONFIRMATION DIAGNOSTICS")
    print(f"  Output: {out_dir}")
    print(f"  Time: {ts}")
    print(f"{'='*70}")

    report_lines = [
        f"# Linux Confirmation Diagnostics Report\n",
        f"Generated: {ts}\n\n",
    ]

    all_results = {}

    # ── A: Lens-only solve (primary) ──────────────────────────────────
    print("\n  ═══ A: Lens-Only Free-Space Solve ═══")
    t0 = time.time()
    cfg_primary = _make_lens_only_config(elements_per_wavelength=5, pml_enabled=True)
    print(cfg_primary.describe())
    sol = solve_helmholtz(cfg_primary, verbose=True)
    t_primary = time.time() - t0
    print(f"  Primary solve complete in {t_primary:.1f}s, DOFs={sol.dofs}, max|p|={sol.max_pressure:.2f} Pa")

    report_lines.append("## A: Lens-Only Solve\n")
    report_lines.append(f"- Elements/λ: {cfg_primary.elements_per_wavelength}\n")
    report_lines.append(f"- DOFs: {sol.dofs}\n")
    report_lines.append(f"- max|p|: {sol.max_pressure:.2f} Pa\n")
    report_lines.append(f"- Solve time: {t_primary:.1f}s\n")
    report_lines.append(f"- Standing sources: **OFF**\n")
    report_lines.append(f"- Top BC: impedance Z_rel={cfg_primary.top_impedance_Zrel}\n\n")

    # ── B1 ────────────────────────────────────────────────────────────
    try:
        b1 = step_B1_disk_bc_verification(sol, out_dir)
        all_results["B1"] = b1
        report_lines.append("## B1: Disk BC Verification\n")
        for k, v in b1.items():
            report_lines.append(f"- {k}: {v}\n")
        report_lines.append("\n")
    except Exception as e:
        print(f"  *** B1 FAILED: {e}")
        traceback.print_exc()
        report_lines.append(f"## B1: FAILED\n{e}\n\n")

    # ── B2 ────────────────────────────────────────────────────────────
    try:
        b2 = step_B2_zstack_xy_slices(sol, out_dir, n_planes=15)
        all_results["B2"] = b2
        report_lines.append("## B2: Z-Stack XY Slices\n")
        report_lines.append(f"- {len(b2)} planes generated\n")
        report_lines.append(f"- Peak pressure: {max(m['max_p_Pa'] for m in b2):.2f} Pa\n\n")
    except Exception as e:
        print(f"  *** B2 FAILED: {e}")
        traceback.print_exc()
        report_lines.append(f"## B2: FAILED\n{e}\n\n")

    # ── B3 ────────────────────────────────────────────────────────────
    try:
        b3 = step_B3_propagation_sanity(sol, out_dir)
        all_results["B3"] = b3
        report_lines.append("## B3: Propagation Sanity\n")
        for k, v in b3.items():
            report_lines.append(f"- {k}: {v:.4f}\n")
        report_lines.append("\n")
    except Exception as e:
        print(f"  *** B3 FAILED: {e}")
        traceback.print_exc()
        report_lines.append(f"## B3: FAILED\n{e}\n\n")

    # ── B4 ────────────────────────────────────────────────────────────
    try:
        b4 = step_B4_pml_reflection(sol, out_dir)
        all_results["B4"] = b4
        report_lines.append("## B4: PML Reflection Test\n")
        for k, v in b4.items():
            report_lines.append(f"- {k}: {v:.4f}\n")
        report_lines.append("\n")
    except Exception as e:
        print(f"  *** B4 FAILED: {e}")
        traceback.print_exc()
        report_lines.append(f"## B4: FAILED\n{e}\n\n")

    # Free primary solution before mesh convergence (needs RAM)
    del sol
    gc.collect()

    # ── B5 ────────────────────────────────────────────────────────────
    try:
        b5 = step_B5_mesh_convergence(out_dir)
        all_results["B5"] = b5
        report_lines.append("## B5: Mesh Convergence\n")
        for epw, res in b5["results"].items():
            report_lines.append(f"- {epw} elem/λ: DOFs={res['dofs']}, max|p|={res['max_p']:.2f} Pa, time={res['solver_time']:.1f}s\n")
        report_lines.append(f"- Max relative difference: {b5['max_relative_diff_pct']:.1f}%\n\n")
    except Exception as e:
        print(f"  *** B5 FAILED: {e}")
        traceback.print_exc()
        report_lines.append(f"## B5: FAILED\n{e}\n\n")

    # ── C: Engine checks (re-solve at 5 elem/λ for this) ─────────────
    print("\n  ═══ Re-solving for engine checks ═══")
    cfg_c = _make_lens_only_config(elements_per_wavelength=5, pml_enabled=True)
    sol_c = solve_helmholtz(cfg_c, verbose=True)
    try:
        c_checks = step_C_engine_checks(sol_c, out_dir)
        all_results["C"] = c_checks
        report_lines.append("## C: Engine Fault Checks\n")
        for k, v in c_checks.items():
            report_lines.append(f"- {k}: {v}\n")
        report_lines.append("\n")
    except Exception as e:
        print(f"  *** C FAILED: {e}")
        traceback.print_exc()
        report_lines.append(f"## C: FAILED\n{e}\n\n")

    del sol_c
    gc.collect()

    # ── Write report ──────────────────────────────────────────────────
    report_path = out_dir / "REPORT.md"
    with open(report_path, "w") as f:
        f.writelines(report_lines)

    # Also save raw results as JSON
    json_path = out_dir / "results.json"
    # Convert numpy types for JSON serialization
    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        return obj

    with open(json_path, "w") as f:
        json.dump(_convert(all_results), f, indent=2)

    print(f"\n{'='*70}")
    print(f"  ALL DIAGNOSTICS COMPLETE")
    print(f"  Report: {report_path}")
    print(f"  Results JSON: {json_path}")
    print(f"  Figures: {out_dir}/B*/  and  {out_dir}/C*/")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
