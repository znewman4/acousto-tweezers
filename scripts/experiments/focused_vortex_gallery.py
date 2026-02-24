#!/usr/bin/env python3
"""
Focused Vortex Gallery — f=4mm targeting trap plane
=====================================================

After diagnostics revealed the vortex ring was 2.2 mm (3λ) wide because
f=10 mm placed the focus 5 mm below the domain, this script:

  1. Reference solves: standing-only, vortex-only (f=4mm), combined
     — all with per-case colourscales (no more black figures)
  2. Focal length sweep: f = 2,3,4,5,6 mm to find tightest ring at trap plane
  3. Best-focus gallery: XY slices at z-heights travelling upward
  4. Amplitude balance check at best focus

Output → results/focused_gallery_<timestamp>/
"""

from __future__ import annotations
import sys, os, time, json, gc
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

NTHREADS = str(min(os.cpu_count() or 4, 14))
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, NTHREADS)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = PROJECT_ROOT / "results" / f"focused_gallery_{TIMESTAMP}"
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
print(f"Output → {OUT_DIR}")

from acoustweezers.experiments.farfield_petri_cuboid.presets import (
    CORRECTED_PRESET, PETSC_MUMPS,
)
from acoustweezers.experiments.farfield_petri_cuboid.config import FarFieldConfig
from acoustweezers.experiments.farfield_petri_cuboid.solve_pressure import solve_helmholtz

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import NearestNDInterpolator
from scipy.signal import argrelextrema

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200,
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
})

ELEM_PER_LAMBDA = 4
H_TOP_OPTIMAL = 2.0085e-3
NGRID = 400

PETSC_OPTS = {
    **PETSC_MUMPS,
    "mat_mumps_icntl_14": "100",
    "mat_mumps_icntl_23": "0",
}

COMMON = {
    **CORRECTED_PRESET,
    "H_top": H_TOP_OPTIMAL,
    "elements_per_wavelength": ELEM_PER_LAMBDA,
}


# ===================================================================
class LightSol:
    def __init__(self, sol):
        self.coords = sol.coords.copy()
        self.p_values = sol.p_values.copy()
        self.cfg = sol.cfg
        self.dofs = sol.dofs
        self.max_pressure = sol.max_pressure
        self.ksp_converged_reason = sol.ksp_converged_reason


def solve_case(overrides, label=""):
    cfg = FarFieldConfig(**overrides)
    t0 = time.time()
    sol = solve_helmholtz(cfg, verbose=True, petsc_options=PETSC_OPTS,
                          export_fields=False)
    dt = time.time() - t0
    lsol = LightSol(sol)
    del sol; gc.collect()
    print(f"  [{label}] max|p|={lsol.max_pressure:.3f} Pa  "
          f"KSP={lsol.ksp_converged_reason}  {dt:.1f}s")
    return lsol


def complex_slice_xy(lsol, z_val, n=NGRID):
    interp_re = NearestNDInterpolator(lsol.coords, np.real(lsol.p_values))
    interp_im = NearestNDInterpolator(lsol.coords, np.imag(lsol.p_values))
    xg = np.linspace(0, lsol.cfg.Lx, n)
    yg = np.linspace(0, lsol.cfg.Ly, n)
    X, Y = np.meshgrid(xg, yg)
    pts = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, z_val)])
    return xg, yg, (interp_re(pts) + 1j * interp_im(pts)).reshape(X.shape)


def complex_slice_xz(lsol, y_val, n=NGRID):
    interp_re = NearestNDInterpolator(lsol.coords, np.real(lsol.p_values))
    interp_im = NearestNDInterpolator(lsol.coords, np.imag(lsol.p_values))
    xg = np.linspace(0, lsol.cfg.Lx, n)
    zg = np.linspace(0, lsol.cfg.H_total, n)
    X, Z = np.meshgrid(xg, zg)
    pts = np.column_stack([X.ravel(), np.full(X.size, y_val), Z.ravel()])
    return xg, zg, (interp_re(pts) + 1j * interp_im(pts)).reshape(X.shape)


def measure_vortex_ring(lsol, z_val, n=500):
    """Return (ring_radius_mm, ring_peak_Pa, fwhm_mm) at given z."""
    xg, yg, pc = complex_slice_xy(lsol, z_val, n=n)
    pmag = np.abs(pc)
    cx_i = n // 2
    cy_i = n // 2
    cx_m = lsol.cfg.Lx / 2

    profile = pmag[cy_i, :]
    half = profile[cx_i:]

    if len(half) < 3:
        return 0, 0, 0

    peak_idx = np.argmax(half)
    ring_r = (xg[cx_i + peak_idx] - cx_m) * 1e3
    ring_peak = half[peak_idx]

    # FWHM
    half_max = ring_peak / 2
    above = half > half_max
    if np.any(above):
        first = np.argmax(above)
        last = len(above) - 1 - np.argmax(above[::-1])
        fwhm = (xg[cx_i + last] - xg[cx_i + first]) * 1e3
    else:
        fwhm = 0

    return ring_r, ring_peak, fwhm


def plot_xy(xg, yg, pc, title, fname, vmax=None):
    """Plot magnitude + phase with per-case colorscale. Returns 2."""
    pmag = np.abs(pc)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.pcolormesh(xg * 1e3, yg * 1e3, pmag,
                       shading="auto", cmap="inferno",
                       vmin=0, vmax=vmax)
    ax.set_title(f"{title}\nmax|p| = {pmag.max():.3f} Pa")
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax, label="|p| [Pa]")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{fname}_mag.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.pcolormesh(xg * 1e3, yg * 1e3, np.angle(pc),
                       shading="auto", cmap="twilight",
                       vmin=-np.pi, vmax=np.pi)
    ax.set_title(f"{title} — Phase")
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax, label="Phase [rad]")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{fname}_phase.png", bbox_inches="tight")
    plt.close(fig)
    return 2


def plot_xz(xg, zg, pc, title, fname, cfg, vmax=None):
    """XZ slice with petri annotation. Returns 2."""
    pmag = np.abs(pc)
    for suffix, data, cmap, label, vkw in [
        ("mag",   pmag,        "inferno", "|p| [Pa]",    dict(vmin=0, vmax=vmax)),
        ("phase", np.angle(pc), "twilight", "Phase [rad]", dict(vmin=-np.pi, vmax=np.pi)),
    ]:
        fig, ax = plt.subplots(figsize=(9, 7))
        im = ax.pcolormesh(xg * 1e3, zg * 1e3, data,
                           shading="auto", cmap=cmap, **vkw)
        ax.axhspan(cfg.H_under * 1e3, cfg.H_total * 1e3,
                   alpha=0.08, color="cyan", label="Petri slab")
        ax.axhline(cfg.H_under * 1e3, color="cyan", ls="--", lw=0.8, alpha=0.6)
        ax.axhline(cfg.H_total * 1e3, color="cyan", ls="--", lw=0.8, alpha=0.6)
        ax.set_xlabel("x [mm]"); ax.set_ylabel("z [mm]")
        ax.set_title(f"{title} — {label.split('[')[0].strip()}\nmax = {pmag.max():.3f} Pa")
        plt.colorbar(im, ax=ax, label=label)
        ax.legend(loc="lower right", fontsize=9)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"{fname}_{suffix}.png", bbox_inches="tight")
        plt.close(fig)
    return 2


# ===================================================================
# MAIN
# ===================================================================
def main():
    t_start = time.time()
    png_count = 0
    all_results = {}

    cfg_base = FarFieldConfig(**COMMON)
    trap_z = cfg_base.H_under + cfg_base.H_top / 2
    lam_mm = cfg_base.wavelength * 1e3

    print("=" * 72)
    print("FOCUSED VORTEX GALLERY")
    print(f"  Trap plane z = {trap_z*1e3:.2f} mm")
    print(f"  New focal length = {COMMON['lens_focal_length']*1e3:.0f} mm")
    print(f"  λ = {lam_mm:.3f} mm")
    print("=" * 72)

    # ==============================================================
    # PHASE 1: Focal length sweep (vortex-only) to find tightest ring
    # ==============================================================
    print(f"\n{'='*72}")
    print("PHASE 1: Focal length sweep (vortex-only)")
    print(f"{'='*72}")

    f_values_mm = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0]
    f_results = []

    for f_mm in f_values_mm:
        print(f"\n  f = {f_mm} mm")
        lsol = solve_case({
            **COMMON,
            "standing_velocity_amplitude": 0.0,
            "lens_focal_length": f_mm * 1e-3,
        }, label=f"vortex_f={f_mm}mm")

        ring_r, ring_peak, fwhm = measure_vortex_ring(lsol, trap_z)

        # Also XZ to see beam shape
        xg, zg, pc_xz = complex_slice_xz(lsol, lsol.cfg.Ly / 2, n=400)
        png_count += plot_xz(xg, zg, pc_xz,
                             f"Vortex Only — f={f_mm}mm",
                             f"fsweep_f{f_mm:.0f}mm_xz", lsol.cfg)

        # XY at trap
        xg, yg, pc_xy = complex_slice_xy(lsol, trap_z)
        png_count += plot_xy(xg, yg, pc_xy,
                             f"Vortex Only — f={f_mm}mm — z={trap_z*1e3:.1f}mm",
                             f"fsweep_f{f_mm:.0f}mm_xy")

        r = {
            "f_mm": f_mm,
            "max_p_Pa": float(lsol.max_pressure),
            "ring_radius_mm": ring_r,
            "ring_peak_Pa": float(ring_peak),
            "ring_fwhm_mm": fwhm,
            "ring_radius_lambda": ring_r / lam_mm,
        }
        f_results.append(r)
        print(f"    Ring R = {ring_r:.3f} mm ({ring_r/lam_mm:.2f}λ)  "
              f"peak = {ring_peak:.4f} Pa  FWHM = {fwhm:.3f} mm")

        del lsol; gc.collect()

    all_results["focal_sweep"] = f_results

    # Plot focal sweep summary
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fs = [r["f_mm"] for r in f_results]
    ax1.plot(fs, [r["ring_radius_mm"] for r in f_results], "ro-", ms=8, lw=2)
    ax1.axhline(lam_mm / 2, color="gray", ls="--", lw=1, label=f"λ/2 = {lam_mm/2:.3f} mm")
    ax1.set_xlabel("Focal length [mm]"); ax1.set_ylabel("Ring radius [mm]")
    ax1.set_title("Vortex Ring Radius vs Focal Length")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(fs, [r["ring_peak_Pa"] for r in f_results], "bs-", ms=8, lw=2)
    ax2.set_xlabel("Focal length [mm]"); ax2.set_ylabel("Ring peak |p| [Pa]")
    ax2.set_title("Vortex Ring Peak Pressure vs Focal Length")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"Focal Length Sweep — Vortex Only at Trap Plane z={trap_z*1e3:.1f}mm",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "focal_sweep_summary.png", bbox_inches="tight")
    plt.close(fig)
    png_count += 1

    # Find best focal length (minimum ring radius or tightest focus)
    best_f = min(f_results, key=lambda r: r["ring_radius_mm"])
    print(f"\n  ** Best focus: f = {best_f['f_mm']} mm → "
          f"ring R = {best_f['ring_radius_mm']:.3f} mm ({best_f['ring_radius_lambda']:.2f}λ)")

    BEST_F = best_f["f_mm"] * 1e-3
    all_results["best_focal_length_mm"] = best_f["f_mm"]

    # ==============================================================
    # PHASE 2: Reference field gallery at best focus
    # ==============================================================
    print(f"\n{'='*72}")
    print(f"PHASE 2: Gallery at best f = {best_f['f_mm']} mm")
    print(f"{'='*72}")

    cases = {
        "standing_only": {
            **COMMON,
            "disk_velocity_amplitude": 0.0,
            "lens_focal_length": BEST_F,
        },
        "vortex_only": {
            **COMMON,
            "standing_velocity_amplitude": 0.0,
            "lens_focal_length": BEST_F,
        },
        "combined": {
            **COMMON,
            "lens_focal_length": BEST_F,
        },
    }

    solutions = {}
    metrics = {}
    for cn, ov in cases.items():
        lsol = solve_case(ov, label=cn)
        solutions[cn] = lsol
        metrics[cn] = {"max_p_Pa": float(lsol.max_pressure)}

    all_results["case_metrics"] = metrics

    # Per-case slices with OWN colourscale
    for cn, lsol in solutions.items():
        cfg = lsol.cfg
        label = cn.replace("_", " ").title()
        y_mid = cfg.Ly / 2
        x_mid = cfg.Lx / 2

        # XY at trap
        xg, yg, pc = complex_slice_xy(lsol, trap_z)
        png_count += plot_xy(xg, yg, pc,
                             f"{label} — XY trap z={trap_z*1e3:.1f}mm",
                             f"{cn}_xy_trap")

        # XY at bath mid
        z_bath = cfg.H_under / 2
        xg, yg, pc = complex_slice_xy(lsol, z_bath)
        png_count += plot_xy(xg, yg, pc,
                             f"{label} — XY bath z={z_bath*1e3:.1f}mm",
                             f"{cn}_xy_bath")

        # XZ mid
        xg, zg, pc = complex_slice_xz(lsol, y_mid)
        png_count += plot_xz(xg, zg, pc,
                             f"{label} — XZ mid y={y_mid*1e3:.1f}mm",
                             f"{cn}_xz_mid", cfg)

        # Centerline
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
        fig.savefig(FIG_DIR / f"{cn}_centerline.png", bbox_inches="tight")
        plt.close(fig)
        png_count += 1

    # ==============================================================
    # PHASE 3: Comparison panels (3-way, own scales)
    # ==============================================================
    print(f"\n{'='*72}")
    print("PHASE 3: Comparison panels")
    print(f"{'='*72}")

    # XY trap-plane 3-way comparison
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    for ax, cn in zip(axes, ["standing_only", "vortex_only", "combined"]):
        ls = solutions[cn]
        xg, yg, pc = complex_slice_xy(ls, trap_z, n=400)
        pmag = np.abs(pc)
        im = ax.pcolormesh(xg * 1e3, yg * 1e3, pmag,
                           shading="auto", cmap="inferno")
        ax.set_title(f"{cn.replace('_',' ').title()}\nmax|p| = {pmag.max():.3f} Pa", fontsize=13)
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, label="|p| [Pa]")
    fig.suptitle(f"XY Trap Plane — Per-Case Colourscales — f={best_f['f_mm']}mm",
                 fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "comparison_xy_trap.png", bbox_inches="tight")
    plt.close(fig)
    png_count += 1

    # Phase comparison
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    for ax, cn in zip(axes, ["standing_only", "vortex_only", "combined"]):
        ls = solutions[cn]
        xg, yg, pc = complex_slice_xy(ls, trap_z, n=400)
        im = ax.pcolormesh(xg * 1e3, yg * 1e3, np.angle(pc),
                           shading="auto", cmap="twilight",
                           vmin=-np.pi, vmax=np.pi)
        ax.set_title(f"{cn.replace('_',' ').title()}", fontsize=13)
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, label="Phase [rad]")
    fig.suptitle(f"Phase at Trap Plane — f={best_f['f_mm']}mm", fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "comparison_phase_trap.png", bbox_inches="tight")
    plt.close(fig)
    png_count += 1

    # XZ 3-way comparison
    fig, axes = plt.subplots(1, 3, figsize=(26, 7))
    for ax, cn in zip(axes, ["standing_only", "vortex_only", "combined"]):
        ls = solutions[cn]
        xg, zg, pc = complex_slice_xz(ls, ls.cfg.Ly / 2, n=400)
        pmag = np.abs(pc)
        im = ax.pcolormesh(xg * 1e3, zg * 1e3, pmag,
                           shading="auto", cmap="inferno")
        ax.axhline(ls.cfg.H_under * 1e3, color="cyan", ls="--", lw=0.8, alpha=0.7)
        ax.axhline(ls.cfg.H_total * 1e3, color="cyan", ls="--", lw=0.8, alpha=0.7)
        ax.set_title(f"{cn.replace('_',' ').title()}\nmax|p| = {pmag.max():.3f} Pa", fontsize=13)
        ax.set_xlabel("x [mm]"); ax.set_ylabel("z [mm]")
        plt.colorbar(im, ax=ax, label="|p| [Pa]")
    fig.suptitle(f"XZ Mid-Plane — Per-Case Colourscales — f={best_f['f_mm']}mm",
                 fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "comparison_xz_mid.png", bbox_inches="tight")
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
               alpha=0.1, color="cyan", label="Petri slab")
    ax.set_xlabel("z [mm]"); ax.set_ylabel("|p| [Pa]")
    ax.set_title(f"Centerline |p|(z) — f={best_f['f_mm']}mm")
    ax.legend(); fig.tight_layout()
    fig.savefig(FIG_DIR / "comparison_centerline.png", bbox_inches="tight")
    plt.close(fig)
    png_count += 1

    # ==============================================================
    # PHASE 4: Z-height progression (combined)
    # ==============================================================
    print(f"\n{'='*72}")
    print("PHASE 4: Z-height progression")
    print(f"{'='*72}")

    Z_HEIGHTS_MM = [0.8, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    ls_comb = solutions["combined"]
    cfg_c = ls_comb.cfg

    # Individual slices
    for z_mm in Z_HEIGHTS_MM:
        z_m = min(z_mm * 1e-3, cfg_c.H_total - 1e-5)
        xg, yg, pc = complex_slice_xy(ls_comb, z_m, n=400)
        png_count += plot_xy(xg, yg, pc,
                             f"Combined — z = {z_mm:.1f} mm",
                             f"zprog_z{z_mm:.1f}".replace(".", "p"))

    # Multi-panel: magnitude
    ncols = 3
    nrows = (len(Z_HEIGHTS_MM) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7*ncols, 6.5*nrows))
    axes_flat = axes.flatten()
    for i, z_mm in enumerate(Z_HEIGHTS_MM):
        z_m = min(z_mm * 1e-3, cfg_c.H_total - 1e-5)
        xg, yg, pc = complex_slice_xy(ls_comb, z_m, n=300)
        pmag = np.abs(pc)
        ax = axes_flat[i]
        im = ax.pcolormesh(xg * 1e3, yg * 1e3, pmag,
                           shading="auto", cmap="inferno")
        region = "Petri" if z_mm >= cfg_c.H_under * 1e3 else "Bath"
        ax.set_title(f"z = {z_mm:.1f} mm  ({region})\nmax = {pmag.max():.2f} Pa")
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, label="|p| [Pa]", shrink=0.80)
    for j in range(len(Z_HEIGHTS_MM), len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.suptitle(f"Combined — XY Slices Travelling Upward\n"
                 f"f = {best_f['f_mm']} mm,  H_top = {H_TOP_OPTIMAL*1e3:.4f} mm",
                 fontsize=16, y=1.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "zprog_combined_panel.png", bbox_inches="tight")
    plt.close(fig)
    png_count += 1

    # Phase panel
    fig, axes = plt.subplots(nrows, ncols, figsize=(7*ncols, 6.5*nrows))
    axes_flat = axes.flatten()
    for i, z_mm in enumerate(Z_HEIGHTS_MM):
        z_m = min(z_mm * 1e-3, cfg_c.H_total - 1e-5)
        xg, yg, pc = complex_slice_xy(ls_comb, z_m, n=300)
        ax = axes_flat[i]
        im = ax.pcolormesh(xg * 1e3, yg * 1e3, np.angle(pc),
                           shading="auto", cmap="twilight", vmin=-np.pi, vmax=np.pi)
        region = "Petri" if z_mm >= cfg_c.H_under * 1e3 else "Bath"
        ax.set_title(f"z = {z_mm:.1f} mm  ({region})")
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, label="Phase [rad]", shrink=0.80)
    for j in range(len(Z_HEIGHTS_MM), len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.suptitle(f"Combined — Phase Travelling Upward\n"
                 f"f = {best_f['f_mm']} mm", fontsize=16, y=1.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "zprog_combined_phase_panel.png", bbox_inches="tight")
    plt.close(fig)
    png_count += 1

    # Vortex-only Z panel
    ls_vort = solutions["vortex_only"]
    fig, axes = plt.subplots(nrows, ncols, figsize=(7*ncols, 6.5*nrows))
    axes_flat = axes.flatten()
    for i, z_mm in enumerate(Z_HEIGHTS_MM):
        z_m = min(z_mm * 1e-3, ls_vort.cfg.H_total - 1e-5)
        xg, yg, pc = complex_slice_xy(ls_vort, z_m, n=300)
        pmag = np.abs(pc)
        ax = axes_flat[i]
        im = ax.pcolormesh(xg * 1e3, yg * 1e3, pmag,
                           shading="auto", cmap="inferno")
        region = "Petri" if z_mm >= ls_vort.cfg.H_under * 1e3 else "Bath"
        ax.set_title(f"z = {z_mm:.1f} mm  ({region})\nmax = {pmag.max():.3f} Pa")
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, label="|p| [Pa]", shrink=0.80)
    for j in range(len(Z_HEIGHTS_MM), len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.suptitle(f"Vortex Only — XY Travelling Upward — f = {best_f['f_mm']} mm",
                 fontsize=16, y=1.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "zprog_vortex_panel.png", bbox_inches="tight")
    plt.close(fig)
    png_count += 1

    # Standing-only Z panel
    ls_stand = solutions["standing_only"]
    fig, axes = plt.subplots(nrows, ncols, figsize=(7*ncols, 6.5*nrows))
    axes_flat = axes.flatten()
    for i, z_mm in enumerate(Z_HEIGHTS_MM):
        z_m = min(z_mm * 1e-3, ls_stand.cfg.H_total - 1e-5)
        xg, yg, pc = complex_slice_xy(ls_stand, z_m, n=300)
        pmag = np.abs(pc)
        ax = axes_flat[i]
        im = ax.pcolormesh(xg * 1e3, yg * 1e3, pmag,
                           shading="auto", cmap="inferno")
        region = "Petri" if z_mm >= ls_stand.cfg.H_under * 1e3 else "Bath"
        ax.set_title(f"z = {z_mm:.1f} mm  ({region})\nmax = {pmag.max():.2f} Pa")
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, label="|p| [Pa]", shrink=0.80)
    for j in range(len(Z_HEIGHTS_MM), len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.suptitle(f"Standing Only — XY Travelling Upward", fontsize=16, y=1.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "zprog_standing_panel.png", bbox_inches="tight")
    plt.close(fig)
    png_count += 1

    # ==============================================================
    # PHASE 5: XZ with z-height markers (3-way)
    # ==============================================================
    print(f"\n  3-way XZ with z-markers …")
    fig, axes = plt.subplots(1, 3, figsize=(26, 8))
    for ax, cn in zip(axes, ["standing_only", "vortex_only", "combined"]):
        ls = solutions[cn]
        xg, zg, pc = complex_slice_xz(ls, ls.cfg.Ly / 2, n=400)
        pmag = np.abs(pc)
        im = ax.pcolormesh(xg * 1e3, zg * 1e3, pmag,
                           shading="auto", cmap="inferno")
        for z_mm in Z_HEIGHTS_MM:
            ax.axhline(z_mm, color="white", ls=":", lw=0.5, alpha=0.6)
        ax.axhline(ls.cfg.H_under * 1e3, color="cyan", ls="--", lw=1)
        ax.axhline(ls.cfg.H_total * 1e3, color="cyan", ls="--", lw=1)
        ax.set_title(f"{cn.replace('_',' ').title()}\nmax = {pmag.max():.3f} Pa")
        ax.set_xlabel("x [mm]"); ax.set_ylabel("z [mm]")
        plt.colorbar(im, ax=ax, label="|p| [Pa]")
    fig.suptitle(f"XZ with Z-slice Markers — f = {best_f['f_mm']} mm",
                 fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "comparison_xz_zslices.png", bbox_inches="tight")
    plt.close(fig)
    png_count += 1

    # ==============================================================
    # Save metadata
    # ==============================================================
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    dt = time.time() - t_start
    print(f"\n{'='*72}")
    print(f"  DONE — {png_count} PNGs in {dt:.0f}s")
    print(f"  Output: {OUT_DIR}")
    print(f"{'='*72}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
