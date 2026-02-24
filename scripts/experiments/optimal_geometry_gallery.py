#!/usr/bin/env python3
"""
Optimal Geometry Gallery — Standing / Vortex / Combined + Z-height progression
================================================================================

Uses the resonance-optimised H_top = 2.0085 mm (m=14 quarter-wave mode,
Mz ≈ 6.39) as the canonical geometry going forward.

Generates:
  1. Standing-only  — XY trap, XZ mid, YZ mid, centerline
  2. Vortex-only    — same set
  3. Combined       — same set
  4. Z-progression  — XY slices at 8 heights from just above the vortex
                      source (z ≈ 0.5 mm) up through the petri slab,
                      showing how the beam evolves

Output → results/optimal_gallery_<timestamp>/

Memory-safe: solves one case at a time, extracts lightweight arrays,
releases FEniCSx objects immediately.
"""

from __future__ import annotations

import sys, os, time, json, gc
import numpy as np
from pathlib import Path
from datetime import datetime

# ── Project root ──
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ── Thread control ──
NTHREADS = str(min(os.cpu_count() or 4, 14))
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, NTHREADS)

# ── Output ──
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = PROJECT_ROOT / "results" / f"optimal_gallery_{TIMESTAMP}"
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
print(f"Output → {OUT_DIR}")

# ====================================================================
# Imports
# ====================================================================
from acoustweezers.experiments.farfield_petri_cuboid.presets import (
    CORRECTED_PRESET, PETSC_MUMPS,
)
from acoustweezers.experiments.farfield_petri_cuboid.config import FarFieldConfig
from acoustweezers.experiments.farfield_petri_cuboid.solve_pressure import solve_helmholtz
from acoustweezers.experiments.farfield_petri_cuboid.post import (
    centerline_z, energy_physical_vs_pml,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import NearestNDInterpolator

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})

# ====================================================================
# Constants
# ====================================================================
ELEM_PER_LAMBDA = 4
NGRID = 500           # interpolation grid per axis
H_TOP_OPTIMAL = 2.0085e-3   # m=14 quarter-wave resonance

PETSC_OPTS = {
    **PETSC_MUMPS,
    "mat_mumps_icntl_14": "100",
    "mat_mumps_icntl_23": "0",
}

# ====================================================================
# Cases
# ====================================================================
COMMON = {
    **CORRECTED_PRESET,
    "H_top": H_TOP_OPTIMAL,
    "elements_per_wavelength": ELEM_PER_LAMBDA,
}

CASES = {
    "standing_only": {
        **COMMON,
        "disk_velocity_amplitude": 0.0,    # vortex OFF
    },
    "vortex_only": {
        **COMMON,
        "standing_velocity_amplitude": 0.0, # standing OFF
    },
    "combined": {
        **COMMON,
        # both sources ON — use preset amplitudes
    },
}

# Z-heights for the progression (mm) — from near-source up through petri
# H_under=3mm, H_top=2.0085mm, H_total=5.0085mm
# PML_z ≈ 1.0λ ≈ 0.742mm at bottom
Z_PROGRESSION_MM = [0.8, 1.2, 1.8, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]


# ====================================================================
# Lightweight solution container
# ====================================================================
class LightSol:
    """Holds only numpy arrays + config — no FEniCSx references."""
    def __init__(self, sol):
        self.coords = sol.coords.copy()
        self.p_values = sol.p_values.copy()
        self.cfg = sol.cfg
        self.dofs = sol.dofs
        self.max_pressure = sol.max_pressure
        self.ksp_converged_reason = sol.ksp_converged_reason


# ====================================================================
# Slicing helpers (return complex field)
# ====================================================================
def complex_slice_xy(lsol, z_val, n=NGRID):
    coords, pv = lsol.coords, lsol.p_values
    interp_re = NearestNDInterpolator(coords, np.real(pv))
    interp_im = NearestNDInterpolator(coords, np.imag(pv))
    xg = np.linspace(0, lsol.cfg.Lx, n)
    yg = np.linspace(0, lsol.cfg.Ly, n)
    X, Y = np.meshgrid(xg, yg)
    pts = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, z_val)])
    pc = (interp_re(pts) + 1j * interp_im(pts)).reshape(X.shape)
    return xg, yg, pc


def complex_slice_xz(lsol, y_val, n=NGRID):
    coords, pv = lsol.coords, lsol.p_values
    interp_re = NearestNDInterpolator(coords, np.real(pv))
    interp_im = NearestNDInterpolator(coords, np.imag(pv))
    xg = np.linspace(0, lsol.cfg.Lx, n)
    zg = np.linspace(0, lsol.cfg.H_total, n)
    X, Z = np.meshgrid(xg, zg)
    pts = np.column_stack([X.ravel(), np.full(X.size, y_val), Z.ravel()])
    pc = (interp_re(pts) + 1j * interp_im(pts)).reshape(X.shape)
    return xg, zg, pc


def complex_slice_yz(lsol, x_val, n=NGRID):
    coords, pv = lsol.coords, lsol.p_values
    interp_re = NearestNDInterpolator(coords, np.real(pv))
    interp_im = NearestNDInterpolator(coords, np.imag(pv))
    yg = np.linspace(0, lsol.cfg.Ly, n)
    zg = np.linspace(0, lsol.cfg.H_total, n)
    Y, Z = np.meshgrid(yg, zg)
    pts = np.column_stack([np.full(Y.size, x_val), Y.ravel(), Z.ravel()])
    pc = (interp_re(pts) + 1j * interp_im(pts)).reshape(Y.shape)
    return yg, zg, pc


# ====================================================================
# Plotting helpers
# ====================================================================
def plot_slice(xg, yg, pc, xlabel, ylabel, title, fname,
               x_scale=1e3, y_scale=1e3, aspect="equal",
               vmax_override=None):
    """
    Generate 3 PNGs: magnitude, log-magnitude, phase.
    Returns (fname_mag, fname_logmag, fname_phase).
    """
    pmag = np.abs(pc)
    pphase = np.angle(pc)

    vmax = vmax_override if vmax_override else None

    # --- Linear magnitude ---
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.pcolormesh(xg * x_scale, yg * y_scale, pmag,
                       shading="auto", cmap="inferno",
                       vmin=0, vmax=vmax)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(f"{title} — |p|")
    if aspect:
        ax.set_aspect(aspect)
    plt.colorbar(im, ax=ax, label="|p| [Pa]", shrink=0.85)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{fname}_mag.png", bbox_inches="tight")
    plt.close(fig)

    # --- Phase ---
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.pcolormesh(xg * x_scale, yg * y_scale, pphase,
                       shading="auto", cmap="twilight",
                       vmin=-np.pi, vmax=np.pi)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(f"{title} — arg(p)")
    if aspect:
        ax.set_aspect(aspect)
    plt.colorbar(im, ax=ax, label="Phase [rad]", shrink=0.85)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{fname}_phase.png", bbox_inches="tight")
    plt.close(fig)

    return 2  # 2 PNGs


def plot_xz_annotated(xg, zg, pc, title, fname, cfg, vmax_override=None):
    """XZ slice with petri-slab region annotated."""
    pmag = np.abs(pc)
    pphase = np.angle(pc)
    vmax = vmax_override if vmax_override else None

    for suffix, data, cmap, label, vkw in [
        ("mag",   pmag,   "inferno", "|p| [Pa]",     dict(vmin=0, vmax=vmax)),
        ("phase", pphase, "twilight", "Phase [rad]",  dict(vmin=-np.pi, vmax=np.pi)),
    ]:
        fig, ax = plt.subplots(figsize=(9, 7))
        im = ax.pcolormesh(xg * 1e3, zg * 1e3, data,
                           shading="auto", cmap=cmap, **vkw)
        # Annotate petri slab
        H_u = cfg.H_under * 1e3
        H_t = cfg.H_total * 1e3
        ax.axhspan(H_u, H_t, alpha=0.10, color="cyan", label="Petri slab")
        ax.axhline(H_u, color="cyan", ls="--", lw=0.8, alpha=0.6)
        ax.axhline(H_t, color="cyan", ls="--", lw=0.8, alpha=0.6)
        ax.set_xlabel("x [mm]"); ax.set_ylabel("z [mm]")
        ax.set_title(f"{title} — {label.split('[')[0].strip()}")
        plt.colorbar(im, ax=ax, label=label, shrink=0.85)
        ax.legend(loc="lower right", fontsize=9)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"{fname}_{suffix}.png", bbox_inches="tight")
        plt.close(fig)


# ====================================================================
# MAIN
# ====================================================================
def main():
    png_count = 0
    all_metrics = {}
    solutions = {}

    print("=" * 72)
    print("OPTIMAL GEOMETRY GALLERY")
    print(f"  H_top = {H_TOP_OPTIMAL*1e3:.4f} mm  (m=14 quarter-wave resonance)")
    print(f"  Resolution: {ELEM_PER_LAMBDA} elem/λ,  grid {NGRID}×{NGRID}")
    print("=" * 72)

    # ── Solve each case, extract lightweight data, free FEniCSx ──
    for case_name, overrides in CASES.items():
        print(f"\n{'─'*72}")
        print(f"  Solving: {case_name}")
        print(f"{'─'*72}")

        cfg = FarFieldConfig(**overrides)
        print(cfg.describe())

        t0 = time.time()
        sol = solve_helmholtz(cfg, verbose=True, petsc_options=PETSC_OPTS,
                              export_fields=False)
        dt = time.time() - t0

        lsol = LightSol(sol)
        del sol
        gc.collect()

        solutions[case_name] = lsol
        all_metrics[case_name] = {
            "H_top_mm": cfg.H_top * 1e3,
            "dofs": lsol.dofs,
            "max_p_Pa": round(lsol.max_pressure, 3),
            "ksp_reason": lsol.ksp_converged_reason,
            "wall_time_s": round(dt, 1),
        }
        print(f"  max|p| = {lsol.max_pressure:.3f} Pa   time = {dt:.1f} s")

    # ── Per-case standard slices ──
    print(f"\n{'='*72}")
    print("GENERATING STANDARD SLICES")
    print(f"{'='*72}")

    # Find global vmax across all cases for consistent colour scale
    global_max_p = max(m["max_p_Pa"] for m in all_metrics.values())
    vmax = global_max_p * 1.05

    for case_name, lsol in solutions.items():
        cfg = lsol.cfg
        trap_z = cfg.H_under + cfg.H_top / 2
        y_mid = cfg.Ly / 2
        x_mid = cfg.Lx / 2

        label = case_name.replace("_", " ").title()

        # XY at trap plane
        print(f"\n  {case_name}: XY at z={trap_z*1e3:.2f} mm (trap)")
        xg, yg, pc = complex_slice_xy(lsol, trap_z)
        png_count += plot_slice(xg, yg, pc, "x [mm]", "y [mm]",
                                f"{label} — XY trap z={trap_z*1e3:.2f}mm",
                                f"{case_name}_xy_trap", vmax_override=vmax)

        # XY at bath midplane
        z_bath = cfg.H_under / 2
        print(f"  {case_name}: XY at z={z_bath*1e3:.1f} mm (bath mid)")
        xg, yg, pc = complex_slice_xy(lsol, z_bath)
        png_count += plot_slice(xg, yg, pc, "x [mm]", "y [mm]",
                                f"{label} — XY bath z={z_bath*1e3:.1f}mm",
                                f"{case_name}_xy_bath", vmax_override=vmax)

        # XZ mid-plane
        print(f"  {case_name}: XZ at y={y_mid*1e3:.1f} mm")
        xg, zg, pc = complex_slice_xz(lsol, y_mid)
        plot_xz_annotated(xg, zg, pc, f"{label} — XZ mid",
                          f"{case_name}_xz_mid", cfg, vmax_override=vmax)
        png_count += 2

        # YZ mid-plane
        print(f"  {case_name}: YZ at x={x_mid*1e3:.1f} mm")
        yg, zg, pc = complex_slice_yz(lsol, x_mid)
        plot_xz_annotated(yg, zg, pc, f"{label} — YZ mid",
                          f"{case_name}_yz_mid", cfg, vmax_override=vmax)
        png_count += 2

        # Centerline z-profile
        print(f"  {case_name}: centerline z-profile")
        zg_cl = np.linspace(0, cfg.H_total, 600)
        cx, cy = cfg.Lx / 2, cfg.Ly / 2
        interp_re = NearestNDInterpolator(lsol.coords, np.real(lsol.p_values))
        interp_im = NearestNDInterpolator(lsol.coords, np.imag(lsol.p_values))
        pts = np.column_stack([np.full(600, cx), np.full(600, cy), zg_cl])
        pmag_cl = np.abs(interp_re(pts) + 1j * interp_im(pts))

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(zg_cl * 1e3, pmag_cl, "k-", lw=1.5)
        ax.axvspan(cfg.H_under * 1e3, cfg.H_total * 1e3,
                   alpha=0.12, color="cyan", label="Petri slab")
        ax.set_xlabel("z [mm]"); ax.set_ylabel("|p| [Pa]")
        ax.set_title(f"{label} — centerline |p|(z)")
        ax.legend(); fig.tight_layout()
        fig.savefig(FIG_DIR / f"{case_name}_centerline_z.png", bbox_inches="tight")
        plt.close(fig)
        png_count += 1

    # ── 3-case comparison panels ──
    print(f"\n{'='*72}")
    print("COMPARISON PANELS")
    print(f"{'='*72}")

    # XY trap-plane comparison: magnitude
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    for ax, cn in zip(axes, ["standing_only", "vortex_only", "combined"]):
        ls = solutions[cn]
        trap_z = ls.cfg.H_under + ls.cfg.H_top / 2
        xg, yg, pc = complex_slice_xy(ls, trap_z, n=400)
        im = ax.pcolormesh(xg * 1e3, yg * 1e3, np.abs(pc),
                           shading="auto", cmap="inferno", vmin=0, vmax=vmax)
        ax.set_title(cn.replace("_", " ").title(), fontsize=14)
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, label="|p| [Pa]", shrink=0.85)
    fig.suptitle(f"XY Trap Plane  |p|  — H_top = {H_TOP_OPTIMAL*1e3:.4f} mm (optimal)",
                 fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "comparison_xy_trap_mag.png", bbox_inches="tight")
    plt.close(fig)
    png_count += 1

    # XY trap-plane comparison: phase
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    for ax, cn in zip(axes, ["standing_only", "vortex_only", "combined"]):
        ls = solutions[cn]
        trap_z = ls.cfg.H_under + ls.cfg.H_top / 2
        xg, yg, pc = complex_slice_xy(ls, trap_z, n=400)
        im = ax.pcolormesh(xg * 1e3, yg * 1e3, np.angle(pc),
                           shading="auto", cmap="twilight",
                           vmin=-np.pi, vmax=np.pi)
        ax.set_title(cn.replace("_", " ").title(), fontsize=14)
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, label="Phase [rad]", shrink=0.85)
    fig.suptitle(f"XY Trap Plane  Phase  — H_top = {H_TOP_OPTIMAL*1e3:.4f} mm (optimal)",
                 fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "comparison_xy_trap_phase.png", bbox_inches="tight")
    plt.close(fig)
    png_count += 1

    # XZ comparison panel
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    for ax, cn in zip(axes, ["standing_only", "vortex_only", "combined"]):
        ls = solutions[cn]
        y_mid = ls.cfg.Ly / 2
        xg, zg, pc = complex_slice_xz(ls, y_mid, n=400)
        im = ax.pcolormesh(xg * 1e3, zg * 1e3, np.abs(pc),
                           shading="auto", cmap="inferno", vmin=0, vmax=vmax)
        ax.axhline(ls.cfg.H_under * 1e3, color="cyan", ls="--", lw=0.8, alpha=0.6)
        ax.axhline(ls.cfg.H_total * 1e3, color="cyan", ls="--", lw=0.8, alpha=0.6)
        ax.set_title(cn.replace("_", " ").title(), fontsize=14)
        ax.set_xlabel("x [mm]"); ax.set_ylabel("z [mm]")
        plt.colorbar(im, ax=ax, label="|p| [Pa]", shrink=0.85)
    fig.suptitle(f"XZ Mid-Plane  |p|  — H_top = {H_TOP_OPTIMAL*1e3:.4f} mm (optimal)",
                 fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "comparison_xz_mid_mag.png", bbox_inches="tight")
    plt.close(fig)
    png_count += 1

    # Centerline overlay
    fig, ax = plt.subplots(figsize=(10, 6))
    for cn, col in [("standing_only", "blue"), ("vortex_only", "orange"), ("combined", "green")]:
        ls = solutions[cn]
        cfg = ls.cfg
        zg_cl = np.linspace(0, cfg.H_total, 600)
        cx, cy = cfg.Lx / 2, cfg.Ly / 2
        interp_re = NearestNDInterpolator(ls.coords, np.real(ls.p_values))
        interp_im = NearestNDInterpolator(ls.coords, np.imag(ls.p_values))
        pts = np.column_stack([np.full(600, cx), np.full(600, cy), zg_cl])
        pmag_cl = np.abs(interp_re(pts) + 1j * interp_im(pts))
        ax.plot(zg_cl * 1e3, pmag_cl, color=col, lw=1.5,
                label=cn.replace("_", " ").title())
    cfg0 = solutions["standing_only"].cfg
    ax.axvspan(cfg0.H_under * 1e3, cfg0.H_total * 1e3,
               alpha=0.10, color="cyan", label="Petri slab")
    ax.set_xlabel("z [mm]"); ax.set_ylabel("|p| [Pa]")
    ax.set_title(f"Centerline |p|(z) — All Cases — H_top = {H_TOP_OPTIMAL*1e3:.4f} mm")
    ax.legend(); fig.tight_layout()
    fig.savefig(FIG_DIR / "comparison_centerline_z.png", bbox_inches="tight")
    plt.close(fig)
    png_count += 1

    # ── Z-height progression (combined field) ──
    print(f"\n{'='*72}")
    print("Z-HEIGHT PROGRESSION (combined field)")
    print(f"  Heights: {Z_PROGRESSION_MM} mm")
    print(f"{'='*72}")

    ls_comb = solutions["combined"]
    cfg_c = ls_comb.cfg

    # Individual XY slices at each z
    for z_mm in Z_PROGRESSION_MM:
        z_m = z_mm * 1e-3
        if z_m > cfg_c.H_total:
            z_m = cfg_c.H_total - 1e-5  # clamp
        print(f"  z = {z_mm:.1f} mm")
        xg, yg, pc = complex_slice_xy(ls_comb, z_m, n=400)
        png_count += plot_slice(
            xg, yg, pc, "x [mm]", "y [mm]",
            f"Combined — XY at z = {z_mm:.1f} mm",
            f"zprog_combined_z{z_mm:.1f}mm".replace(".", "p"),
            vmax_override=vmax,
        )

    # Multi-panel progression figure (3 rows × 3 cols)
    n_z = len(Z_PROGRESSION_MM)
    ncols = 3
    nrows = (n_z + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6.5 * nrows))
    axes = axes.flatten()

    for i, z_mm in enumerate(Z_PROGRESSION_MM):
        z_m = z_mm * 1e-3
        if z_m > cfg_c.H_total:
            z_m = cfg_c.H_total - 1e-5
        xg, yg, pc = complex_slice_xy(ls_comb, z_m, n=300)
        pmag = np.abs(pc)
        ax = axes[i]
        im = ax.pcolormesh(xg * 1e3, yg * 1e3, pmag,
                           shading="auto", cmap="inferno",
                           vmin=0, vmax=vmax)
        # Mark which region this z is in
        region = "Petri" if z_mm >= cfg_c.H_under * 1e3 else "Bath"
        if z_mm < cfg_c.t_pml_z * 1e3:
            region = "PML"
        ax.set_title(f"z = {z_mm:.1f} mm  ({region})", fontsize=13)
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, label="|p| [Pa]", shrink=0.80)

    # Hide unused axes
    for j in range(n_z, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"Combined Field — XY Slices Travelling Upward\n"
        f"H_top = {H_TOP_OPTIMAL*1e3:.4f} mm (optimal resonance)",
        fontsize=16, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "zprog_combined_panel.png", bbox_inches="tight")
    plt.close(fig)
    png_count += 1

    # Phase progression panel
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6.5 * nrows))
    axes = axes.flatten()

    for i, z_mm in enumerate(Z_PROGRESSION_MM):
        z_m = z_mm * 1e-3
        if z_m > cfg_c.H_total:
            z_m = cfg_c.H_total - 1e-5
        xg, yg, pc = complex_slice_xy(ls_comb, z_m, n=300)
        pphase = np.angle(pc)
        ax = axes[i]
        im = ax.pcolormesh(xg * 1e3, yg * 1e3, pphase,
                           shading="auto", cmap="twilight",
                           vmin=-np.pi, vmax=np.pi)
        region = "Petri" if z_mm >= cfg_c.H_under * 1e3 else "Bath"
        if z_mm < cfg_c.t_pml_z * 1e3:
            region = "PML"
        ax.set_title(f"z = {z_mm:.1f} mm  ({region})", fontsize=13)
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, label="Phase [rad]", shrink=0.80)

    for j in range(n_z, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"Combined Field — Phase Slices Travelling Upward\n"
        f"H_top = {H_TOP_OPTIMAL*1e3:.4f} mm (optimal resonance)",
        fontsize=16, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "zprog_combined_phase_panel.png", bbox_inches="tight")
    plt.close(fig)
    png_count += 1

    # ── Vortex-only Z progression for comparison ──
    print(f"\n  Vortex-only Z-progression …")
    ls_vort = solutions["vortex_only"]
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6.5 * nrows))
    axes = axes.flatten()

    for i, z_mm in enumerate(Z_PROGRESSION_MM):
        z_m = z_mm * 1e-3
        if z_m > ls_vort.cfg.H_total:
            z_m = ls_vort.cfg.H_total - 1e-5
        xg, yg, pc = complex_slice_xy(ls_vort, z_m, n=300)
        pmag = np.abs(pc)
        ax = axes[i]
        im = ax.pcolormesh(xg * 1e3, yg * 1e3, pmag,
                           shading="auto", cmap="inferno",
                           vmin=0, vmax=vmax)
        region = "Petri" if z_mm >= ls_vort.cfg.H_under * 1e3 else "Bath"
        if z_mm < ls_vort.cfg.t_pml_z * 1e3:
            region = "PML"
        ax.set_title(f"z = {z_mm:.1f} mm  ({region})", fontsize=13)
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, label="|p| [Pa]", shrink=0.80)

    for j in range(n_z, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"Vortex Only — XY Slices Travelling Upward\n"
        f"H_top = {H_TOP_OPTIMAL*1e3:.4f} mm",
        fontsize=16, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "zprog_vortex_panel.png", bbox_inches="tight")
    plt.close(fig)
    png_count += 1

    # ── Standing-only Z progression for comparison ──
    print(f"  Standing-only Z-progression …")
    ls_stand = solutions["standing_only"]
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6.5 * nrows))
    axes = axes.flatten()

    for i, z_mm in enumerate(Z_PROGRESSION_MM):
        z_m = z_mm * 1e-3
        if z_m > ls_stand.cfg.H_total:
            z_m = ls_stand.cfg.H_total - 1e-5
        xg, yg, pc = complex_slice_xy(ls_stand, z_m, n=300)
        pmag = np.abs(pc)
        ax = axes[i]
        im = ax.pcolormesh(xg * 1e3, yg * 1e3, pmag,
                           shading="auto", cmap="inferno",
                           vmin=0, vmax=vmax)
        region = "Petri" if z_mm >= ls_stand.cfg.H_under * 1e3 else "Bath"
        if z_mm < ls_stand.cfg.t_pml_z * 1e3:
            region = "PML"
        ax.set_title(f"z = {z_mm:.1f} mm  ({region})", fontsize=13)
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, label="|p| [Pa]", shrink=0.80)

    for j in range(n_z, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"Standing Only — XY Slices Travelling Upward\n"
        f"H_top = {H_TOP_OPTIMAL*1e3:.4f} mm (optimal resonance)",
        fontsize=16, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "zprog_standing_panel.png", bbox_inches="tight")
    plt.close(fig)
    png_count += 1

    # ── XZ side-by-side at each z height (combined vs vortex vs standing) ──
    print(f"\n  3-way XZ comparison with z-height markers …")
    fig, axes = plt.subplots(1, 3, figsize=(26, 8))
    for ax, cn, title in zip(axes,
                              ["standing_only", "vortex_only", "combined"],
                              ["Standing Only", "Vortex Only", "Combined"]):
        ls = solutions[cn]
        y_mid = ls.cfg.Ly / 2
        xg, zg, pc = complex_slice_xz(ls, y_mid, n=400)
        im = ax.pcolormesh(xg * 1e3, zg * 1e3, np.abs(pc),
                           shading="auto", cmap="inferno", vmin=0, vmax=vmax)
        # Mark z-progression heights
        for z_mm in Z_PROGRESSION_MM:
            ax.axhline(z_mm, color="white", ls=":", lw=0.6, alpha=0.7)
        ax.axhline(ls.cfg.H_under * 1e3, color="cyan", ls="--", lw=1.0, alpha=0.7)
        ax.axhline(ls.cfg.H_total * 1e3, color="cyan", ls="--", lw=1.0, alpha=0.7)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("x [mm]"); ax.set_ylabel("z [mm]")
        plt.colorbar(im, ax=ax, label="|p| [Pa]", shrink=0.85)
    fig.suptitle(
        f"XZ Cross-Section with Z-slice Heights Marked\n"
        f"H_top = {H_TOP_OPTIMAL*1e3:.4f} mm (optimal)",
        fontsize=15, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "comparison_xz_with_zslices.png", bbox_inches="tight")
    plt.close(fig)
    png_count += 1

    # ── Save metadata ──
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    with open(OUT_DIR / "gallery_info.json", "w") as f:
        json.dump({
            "H_top_optimal_mm": H_TOP_OPTIMAL * 1e3,
            "resonance_mode": "m=14 quarter-wave",
            "elements_per_wavelength": ELEM_PER_LAMBDA,
            "grid_size": NGRID,
            "z_progression_mm": Z_PROGRESSION_MM,
            "total_pngs": png_count,
            "timestamp": TIMESTAMP,
        }, f, indent=2)

    print(f"\n{'='*72}")
    print(f"  DONE — {png_count} PNGs generated")
    print(f"  Output: {OUT_DIR}")
    print(f"{'='*72}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
