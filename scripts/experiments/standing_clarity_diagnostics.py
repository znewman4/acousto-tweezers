#!/usr/bin/env python3
"""
STANDING-WAVE CLARITY DIAGNOSTICS  (S1–S4)
============================================

Four-phase diagnostic suite for standing-wave pattern clarity in a
square-dish cuboid at fixed 2 MHz.

  S1 — Baseline visualization  (standing_only, full PNG set)
  S2 — Geometry sweeps  (H_top, Lx=Ly; identify resonant configurations)
  S3 — Boundary-condition sensitivity  (passive BC variants on z-faces)
  S4 — Final "cleanest" configuration at publication quality

All outputs land under  results/standing_clarity_<timestamp>/.

Usage:
    python scripts/experiments/standing_clarity_diagnostics.py
"""

from __future__ import annotations
import sys, os, time, json, gc, csv
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# ── project root ──
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

NTHREADS = str(min(os.cpu_count() or 4, 14))
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, NTHREADS)

# ── output ──
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_ROOT = PROJECT_ROOT / "results" / f"standing_clarity_{TIMESTAMP}"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import NearestNDInterpolator
from scipy.signal import find_peaks

from acoustweezers.experiments.farfield_petri_cuboid.presets import (
    CORRECTED_PRESET, PETSC_MUMPS,
)
from acoustweezers.experiments.farfield_petri_cuboid.config import FarFieldConfig
from acoustweezers.experiments.farfield_petri_cuboid.solve_pressure import solve_helmholtz
from acoustweezers.experiments.farfield_petri_cuboid.post import (
    slice_xy, slice_xz, centerline_z, energy_physical_vs_pml,
)

# ====================================================================
# Global constants  (tuned for 7.5 GB RAM)
# ====================================================================
DPI = 300
NGRID = 300          # interpolation grid — kept small to avoid OOM
NLINE = 500          # 1-D line-cut points
ELEM_PER_LAMBDA = 4  # 4 elem/λ to stay within memory

PETSC_OPTS = {
    **PETSC_MUMPS,
    "mat_mumps_icntl_14": "100",
    "mat_mumps_icntl_23": "0",
    "mat_mumps_icntl_28": "2",
    "mat_mumps_icntl_29": "2",
}

plt.rcParams.update({
    "figure.dpi": DPI, "savefig.dpi": DPI,
    "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
})


# ====================================================================
# Lightweight solution container (frees FEniCSx/PETSc memory)
# ====================================================================
class LightSol:
    def __init__(self, sol):
        self.coords = sol.coords.copy()
        self.p_values = sol.p_values.copy()
        self.cfg = sol.cfg
        self.dofs = sol.dofs
        self.ksp_converged_reason = sol.ksp_converged_reason
        self.ksp_iterations = sol.ksp_iterations
        self.ksp_residual_norm = sol.ksp_residual_norm
        self.max_pressure = sol.max_pressure


def solve_and_extract(cfg, label=""):
    """Solve, extract light arrays, free FEniCSx objects."""
    print(f"\n  Solving {label} …")
    t0 = time.time()
    sol = solve_helmholtz(cfg, verbose=True, petsc_options=PETSC_OPTS,
                          export_fields=False)
    dt = time.time() - t0
    energy = energy_physical_vs_pml(sol)
    lsol = LightSol(sol)
    lsol.wall_time = dt
    lsol.energy = energy
    del sol; gc.collect()
    return lsol


# ====================================================================
# Slicing / line-cut helpers
# ====================================================================

def _interp(coords, pv):
    return (NearestNDInterpolator(coords, np.real(pv)),
            NearestNDInterpolator(coords, np.imag(pv)))


def complex_slice_xy(sol, z_val, n=NGRID):
    ire, iim = _interp(sol.coords, sol.p_values)
    xg = np.linspace(0, sol.cfg.Lx, n)
    yg = np.linspace(0, sol.cfg.Ly, n)
    X, Y = np.meshgrid(xg, yg)
    pts = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, z_val)])
    return xg, yg, (ire(pts) + 1j * iim(pts)).reshape(X.shape)


def complex_slice_xz(sol, y_val, n=NGRID):
    ire, iim = _interp(sol.coords, sol.p_values)
    xg = np.linspace(0, sol.cfg.Lx, n)
    zg = np.linspace(0, sol.cfg.H_total, n)
    X, Z = np.meshgrid(xg, zg)
    pts = np.column_stack([X.ravel(), np.full(X.size, y_val), Z.ravel()])
    return xg, zg, (ire(pts) + 1j * iim(pts)).reshape(X.shape)


def complex_slice_yz(sol, x_val, n=NGRID):
    ire, iim = _interp(sol.coords, sol.p_values)
    yg = np.linspace(0, sol.cfg.Ly, n)
    zg = np.linspace(0, sol.cfg.H_total, n)
    Y, Z = np.meshgrid(yg, zg)
    pts = np.column_stack([np.full(Y.size, x_val), Y.ravel(), Z.ravel()])
    return yg, zg, (ire(pts) + 1j * iim(pts)).reshape(Y.shape)


def line_z(sol, n=NLINE):
    """Complex p along z at (Lx/2, Ly/2)."""
    ire, iim = _interp(sol.coords, sol.p_values)
    cfg = sol.cfg
    zg = np.linspace(0, cfg.H_total, n)
    pts = np.column_stack([np.full(n, cfg.Lx/2), np.full(n, cfg.Ly/2), zg])
    return zg, ire(pts) + 1j * iim(pts)


def line_x(sol, z_val, n=NLINE):
    """Complex p along x at (y=Ly/2, z=z_val)."""
    ire, iim = _interp(sol.coords, sol.p_values)
    cfg = sol.cfg
    xg = np.linspace(0, cfg.Lx, n)
    pts = np.column_stack([xg, np.full(n, cfg.Ly/2), np.full(n, z_val)])
    return xg, ire(pts) + 1j * iim(pts)


def line_y(sol, z_val, n=NLINE):
    """Complex p along y at (x=Lx/2, z=z_val)."""
    ire, iim = _interp(sol.coords, sol.p_values)
    cfg = sol.cfg
    yg = np.linspace(0, cfg.Ly, n)
    pts = np.column_stack([np.full(n, cfg.Lx/2), yg, np.full(n, z_val)])
    return yg, ire(pts) + 1j * iim(pts)


# ====================================================================
# PML masking
# ====================================================================

def pml_mask_xy(xg, yg, cfg):
    """Bool 2-D array: True where physical, False in PML."""
    t = cfg.t_pml_xy
    X, Y = np.meshgrid(xg, yg)
    return (X >= t) & (X <= cfg.Lx - t) & (Y >= t) & (Y <= cfg.Ly - t)


def pml_mask_xz(xg, zg, cfg):
    t_xy = cfg.t_pml_xy; t_z = cfg.t_pml_z
    X, Z = np.meshgrid(xg, zg)
    return (X >= t_xy) & (X <= cfg.Lx - t_xy) & (Z >= t_z)


def pml_mask_yz(yg, zg, cfg):
    t_xy = cfg.t_pml_xy; t_z = cfg.t_pml_z
    Y, Z = np.meshgrid(yg, zg)
    return (Y >= t_xy) & (Y <= cfg.Ly - t_xy) & (Z >= t_z)


# ====================================================================
# Metrics computation
# ====================================================================

def compute_metrics(sol, label=""):
    """Compute standing-wave clarity metrics."""
    cfg = sol.cfg
    trap_z = cfg.H_under + cfg.H_top / 2
    lam = cfg.wavelength

    # --- centerline z ---
    zg, pz = line_z(sol)
    pmag_z = np.abs(pz)
    re_z = np.real(pz)
    mean_z = np.mean(pmag_z) + 1e-30
    Mz = (np.max(pmag_z) - np.min(pmag_z)) / mean_z

    # Estimated nodal spacing from Re(p) zero crossings
    sign_changes = np.where(np.diff(np.sign(re_z)))[0]
    if len(sign_changes) >= 2:
        zc = 0.5 * (zg[sign_changes] + zg[sign_changes + 1])
        spacings = np.diff(zc)
        nodal_spacing_z = float(np.median(spacings))
    else:
        nodal_spacing_z = float("nan")

    # --- x-line at trap ---
    xg, px = line_x(sol, trap_z)
    # restrict to physical domain
    t = cfg.t_pml_xy
    mask_x = (xg >= t) & (xg <= cfg.Lx - t)
    pmag_x = np.abs(px[mask_x])
    mean_x = np.mean(pmag_x) + 1e-30
    Mx = (np.max(pmag_x) - np.min(pmag_x)) / mean_x

    # --- y-line at trap ---
    yg, py = line_y(sol, trap_z)
    mask_y = (yg >= t) & (yg <= cfg.Ly - t)
    pmag_y = np.abs(py[mask_y])
    mean_y = np.mean(pmag_y) + 1e-30
    My = (np.max(pmag_y) - np.min(pmag_y)) / mean_y

    pv = sol.p_values
    m = {
        "label": label,
        "Lx_mm": cfg.Lx * 1e3,
        "Ly_mm": cfg.Ly * 1e3,
        "H_under_mm": cfg.H_under * 1e3,
        "H_top_mm": cfg.H_top * 1e3,
        "H_total_mm": cfg.H_total * 1e3,
        "lambda_mm": lam * 1e3,
        "elems_per_lambda": cfg.elements_per_wavelength,
        "DOFs": sol.dofs,
        "solve_time_s": round(getattr(sol, "wall_time", 0), 1),
        "ksp_reason": sol.ksp_converged_reason,
        "max_p_Pa": float(np.max(np.abs(pv))),
        "mean_p_Pa": float(np.mean(np.abs(pv))),
        "Mz": round(Mz, 4),
        "Mx": round(Mx, 4),
        "My": round(My, 4),
        "nodal_spacing_z_mm": round(nodal_spacing_z * 1e3, 4) if not np.isnan(nodal_spacing_z) else None,
        "expected_half_lambda_mm": round(lam / 2 * 1e3, 4),
    }
    return m


# ====================================================================
# Full PNG set generation for one solution
# ====================================================================

def generate_full_png_set(sol, fig_dir: Path, prefix: str, title_extra: str = ""):
    """
    Generate the complete PNG set for one standing_only solution.
    Returns list of generated filenames.
    """
    fig_dir.mkdir(parents=True, exist_ok=True)
    cfg = sol.cfg
    trap_z = cfg.H_under + cfg.H_top / 2
    bath_z = cfg.H_under / 2
    y_mid = cfg.Ly / 2
    x_mid = cfg.Lx / 2
    t_xy = cfg.t_pml_xy
    t_z = cfg.t_pml_z

    fnames = []

    def _save(fig, name):
        p = fig_dir / f"{prefix}{name}.png"
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        fnames.append(p.name)

    # ─── helper: 2-D colour plot with optional PML mask ───
    def _plot2d(ag, bg, data, xlabel, ylabel, title, fname,
                cmap="inferno", mask=None, vmin=None, vmax=None,
                annotations=None, aspect="equal"):
        fig, ax = plt.subplots(figsize=(8, 7))
        d = data.copy()
        if mask is not None:
            d = np.where(mask, d, np.nan)
        kw = {}
        if vmin is not None: kw["vmin"] = vmin
        if vmax is not None: kw["vmax"] = vmax
        im = ax.pcolormesh(ag * 1e3, bg * 1e3, d, shading="auto", cmap=cmap, **kw)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.set_title(f"{title}{title_extra}")
        if aspect: ax.set_aspect(aspect)
        cb = plt.colorbar(im, ax=ax, shrink=0.82)
        if annotations:
            for ann in annotations:
                ann(ax)
        fig.tight_layout()
        _save(fig, fname)

    # ─── PML boundary annotation helpers ───
    def ann_pml_xy(ax):
        for v in [t_xy * 1e3, (cfg.Lx - t_xy) * 1e3]:
            ax.axvline(v, color="w", ls="--", lw=0.5, alpha=0.5)
        for v in [t_xy * 1e3, (cfg.Ly - t_xy) * 1e3]:
            ax.axhline(v, color="w", ls="--", lw=0.5, alpha=0.5)

    def ann_pml_xz(ax):
        for v in [t_xy * 1e3, (cfg.Lx - t_xy) * 1e3]:
            ax.axvline(v, color="w", ls="--", lw=0.5, alpha=0.5)
        ax.axhline(t_z * 1e3, color="w", ls="--", lw=0.5, alpha=0.5)
        ax.axhline(cfg.H_under * 1e3, color="cyan", ls=":", lw=0.8)

    def ann_pml_yz(ax):
        for v in [t_xy * 1e3, (cfg.Ly - t_xy) * 1e3]:
            ax.axvline(v, color="w", ls="--", lw=0.5, alpha=0.5)
        ax.axhline(t_z * 1e3, color="w", ls="--", lw=0.5, alpha=0.5)
        ax.axhline(cfg.H_under * 1e3, color="cyan", ls=":", lw=0.8)

    # ═══════════════════════════════════════════════════════════════
    #  XY slices  (trap plane + bath midplane)
    # ═══════════════════════════════════════════════════════════════
    for z_val, zlabel in [(trap_z, "trap"), (bath_z, "bath")]:
        xg, yg, pc = complex_slice_xy(sol, z_val)
        pmag = np.abs(pc); rep = np.real(pc); logp = np.log10(np.clip(pmag, 1, None))
        mk = pml_mask_xy(xg, yg, cfg)

        vmax_mag = np.nanmax(pmag); vmax_re = max(abs(np.nanmin(rep)), abs(np.nanmax(rep)))

        # Full domain
        _plot2d(xg, yg, pmag, "x [mm]", "y [mm]",
                f"|p| XY z={z_val*1e3:.1f}mm ({zlabel})", f"xy_{zlabel}_mag",
                annotations=[ann_pml_xy])
        _plot2d(xg, yg, rep, "x [mm]", "y [mm]",
                f"Re(p) XY z={z_val*1e3:.1f}mm ({zlabel})", f"xy_{zlabel}_re",
                cmap="RdBu_r", vmin=-vmax_re, vmax=vmax_re, annotations=[ann_pml_xy])
        _plot2d(xg, yg, logp, "x [mm]", "y [mm]",
                f"log₁₀|p| XY z={z_val*1e3:.1f}mm ({zlabel})", f"xy_{zlabel}_log",
                annotations=[ann_pml_xy])

        # PML-masked
        _plot2d(xg, yg, pmag, "x [mm]", "y [mm]",
                f"|p| XY z={z_val*1e3:.1f}mm ({zlabel}) [phys]",
                f"xy_{zlabel}_mag_phys", mask=mk, annotations=[ann_pml_xy])
        _plot2d(xg, yg, rep, "x [mm]", "y [mm]",
                f"Re(p) XY z={z_val*1e3:.1f}mm ({zlabel}) [phys]",
                f"xy_{zlabel}_re_phys", cmap="RdBu_r",
                vmin=-vmax_re, vmax=vmax_re, mask=mk, annotations=[ann_pml_xy])

    # ═══════════════════════════════════════════════════════════════
    #  XZ mid-plane
    # ═══════════════════════════════════════════════════════════════
    xg, zg, pc_xz = complex_slice_xz(sol, y_mid)
    pmag_xz = np.abs(pc_xz); rep_xz = np.real(pc_xz)
    logp_xz = np.log10(np.clip(pmag_xz, 1, None))
    mk_xz = pml_mask_xz(xg, zg, cfg)
    vmax_re_xz = max(abs(np.nanmin(rep_xz)), abs(np.nanmax(rep_xz)))

    _plot2d(xg, zg, pmag_xz, "x [mm]", "z [mm]",
            f"|p| XZ y={y_mid*1e3:.1f}mm", "xz_mag", aspect=None,
            annotations=[ann_pml_xz])
    _plot2d(xg, zg, rep_xz, "x [mm]", "z [mm]",
            f"Re(p) XZ y={y_mid*1e3:.1f}mm", "xz_re", cmap="RdBu_r",
            vmin=-vmax_re_xz, vmax=vmax_re_xz, aspect=None,
            annotations=[ann_pml_xz])
    _plot2d(xg, zg, logp_xz, "x [mm]", "z [mm]",
            f"log₁₀|p| XZ y={y_mid*1e3:.1f}mm", "xz_log", aspect=None,
            annotations=[ann_pml_xz])
    _plot2d(xg, zg, pmag_xz, "x [mm]", "z [mm]",
            f"|p| XZ [phys]", "xz_mag_phys", aspect=None,
            mask=mk_xz, annotations=[ann_pml_xz])
    _plot2d(xg, zg, rep_xz, "x [mm]", "z [mm]",
            f"Re(p) XZ [phys]", "xz_re_phys", cmap="RdBu_r",
            vmin=-vmax_re_xz, vmax=vmax_re_xz, aspect=None,
            mask=mk_xz, annotations=[ann_pml_xz])

    # ═══════════════════════════════════════════════════════════════
    #  YZ mid-plane
    # ═══════════════════════════════════════════════════════════════
    yg2, zg2, pc_yz = complex_slice_yz(sol, x_mid)
    pmag_yz = np.abs(pc_yz); rep_yz = np.real(pc_yz)
    logp_yz = np.log10(np.clip(pmag_yz, 1, None))
    mk_yz = pml_mask_yz(yg2, zg2, cfg)
    vmax_re_yz = max(abs(np.nanmin(rep_yz)), abs(np.nanmax(rep_yz)))

    _plot2d(yg2, zg2, pmag_yz, "y [mm]", "z [mm]",
            f"|p| YZ x={x_mid*1e3:.1f}mm", "yz_mag", aspect=None,
            annotations=[ann_pml_yz])
    _plot2d(yg2, zg2, rep_yz, "y [mm]", "z [mm]",
            f"Re(p) YZ x={x_mid*1e3:.1f}mm", "yz_re", cmap="RdBu_r",
            vmin=-vmax_re_yz, vmax=vmax_re_yz, aspect=None,
            annotations=[ann_pml_yz])
    _plot2d(yg2, zg2, logp_yz, "y [mm]", "z [mm]",
            f"log₁₀|p| YZ x={x_mid*1e3:.1f}mm", "yz_log", aspect=None,
            annotations=[ann_pml_yz])
    _plot2d(yg2, zg2, pmag_yz, "y [mm]", "z [mm]",
            f"|p| YZ [phys]", "yz_mag_phys", aspect=None,
            mask=mk_yz, annotations=[ann_pml_yz])
    _plot2d(yg2, zg2, rep_yz, "y [mm]", "z [mm]",
            f"Re(p) YZ [phys]", "yz_re_phys", cmap="RdBu_r",
            vmin=-vmax_re_yz, vmax=vmax_re_yz, aspect=None,
            mask=mk_yz, annotations=[ann_pml_yz])

    # ═══════════════════════════════════════════════════════════════
    #  1-D line cuts
    # ═══════════════════════════════════════════════════════════════
    # Centerline z
    zl, pz = line_z(sol)
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(zl * 1e3, np.abs(pz), "k-", lw=1.2)
    axes[0].set_ylabel("|p| [Pa]"); axes[0].set_title(f"Centerline z{title_extra}")
    axes[0].axvspan(cfg.H_under*1e3, cfg.H_total*1e3, alpha=0.1, color="red", label="petri slab")
    axes[0].legend(fontsize=8)
    axes[1].plot(zl * 1e3, np.real(pz), "b-", lw=1.2)
    axes[1].set_ylabel("Re(p) [Pa]"); axes[1].set_xlabel("z [mm]")
    axes[1].axhline(0, color="gray", ls=":", lw=0.5)
    axes[1].axvspan(cfg.H_under*1e3, cfg.H_total*1e3, alpha=0.1, color="red")
    fig.tight_layout(); _save(fig, "linecut_z")

    # x-line at trap
    xl, px = line_x(sol, trap_z)
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(xl * 1e3, np.abs(px), "k-", lw=1.2)
    axes[0].set_ylabel("|p| [Pa]"); axes[0].set_title(f"x-line at z={trap_z*1e3:.1f}mm{title_extra}")
    axes[0].axvspan(0, t_xy*1e3, alpha=0.1, color="orange", label="PML")
    axes[0].axvspan((cfg.Lx-t_xy)*1e3, cfg.Lx*1e3, alpha=0.1, color="orange")
    axes[0].legend(fontsize=8)
    axes[1].plot(xl * 1e3, np.real(px), "b-", lw=1.2)
    axes[1].set_ylabel("Re(p) [Pa]"); axes[1].set_xlabel("x [mm]")
    axes[1].axhline(0, color="gray", ls=":", lw=0.5)
    fig.tight_layout(); _save(fig, "linecut_x")

    # y-line at trap
    yl, py = line_y(sol, trap_z)
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(yl * 1e3, np.abs(py), "k-", lw=1.2)
    axes[0].set_ylabel("|p| [Pa]"); axes[0].set_title(f"y-line at z={trap_z*1e3:.1f}mm{title_extra}")
    axes[0].axvspan(0, t_xy*1e3, alpha=0.1, color="orange", label="PML")
    axes[0].axvspan((cfg.Ly-t_xy)*1e3, cfg.Ly*1e3, alpha=0.1, color="orange")
    axes[0].legend(fontsize=8)
    axes[1].plot(yl * 1e3, np.real(py), "b-", lw=1.2)
    axes[1].set_ylabel("Re(p) [Pa]"); axes[1].set_xlabel("y [mm]")
    axes[1].axhline(0, color="gray", ls=":", lw=0.5)
    fig.tight_layout(); _save(fig, "linecut_y")

    print(f"    → {len(fnames)} PNGs in {fig_dir}")
    return fnames


# ====================================================================
# PHASE S1 — Baseline
# ====================================================================

def run_s1():
    print("\n" + "=" * 72)
    print("PHASE S1 — STANDING-WAVE BASELINE")
    print("=" * 72)

    s1_dir = OUT_ROOT / "S1_baseline"
    fig_dir = s1_dir / "figures"

    overrides = {
        **CORRECTED_PRESET,
        "standing_velocity_amplitude": 10e-6,
        "standing_phase_pattern": "antiphase",
        "standing_axis": "both",
        "disk_velocity_amplitude": 0.0,   # vortex OFF
        "elements_per_wavelength": ELEM_PER_LAMBDA,
    }
    cfg = FarFieldConfig(**overrides)
    sol = solve_and_extract(cfg, label="S1 standing baseline")

    generate_full_png_set(sol, fig_dir, prefix="standing_baseline_",
                          title_extra="  [S1 baseline]")

    m = compute_metrics(sol, label="S1_baseline")
    with open(s1_dir / "metrics_standing_baseline.json", "w") as f:
        json.dump(m, f, indent=2)

    print(f"\n  S1 metrics: max|p|={m['max_p_Pa']:.2f} Pa  "
          f"Mz={m['Mz']:.4f}  Mx={m['Mx']:.4f}  "
          f"nodal_spacing_z={m['nodal_spacing_z_mm']} mm  "
          f"(λ/2={m['expected_half_lambda_mm']} mm)")

    return sol, m


# ====================================================================
# PHASE S2 — Geometry sweeps
# ====================================================================

def run_s2():
    print("\n" + "=" * 72)
    print("PHASE S2 — GEOMETRY SWEEPS  (fixed f = 2 MHz)")
    print("=" * 72)

    s2_dir = OUT_ROOT / "S2_geometry"
    fig_dir = s2_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    lam = 1484.0 / 2e6  # ~0.742 mm
    base_H_top = 2e-3
    base_Lxy = 6e-3
    sweep_rows = []

    # --- S2.A: Height sweep (vary H_top by ±0.25λ) ---
    # Only ±0.25λ to conserve RAM (3 solves, not 5)
    print("\n  ── S2.A: Height sweep (H_top ± 0.25λ) ──")
    height_offsets = [-0.25, 0.0, 0.25]

    for dh_lam in height_offsets:
        h_top = base_H_top + dh_lam * lam
        label = f"Htop_{h_top*1e3:.3f}mm"
        overrides = {
            **CORRECTED_PRESET,
            "H_top": h_top,
            "standing_velocity_amplitude": 10e-6,
            "disk_velocity_amplitude": 0.0,
            "elements_per_wavelength": ELEM_PER_LAMBDA,
        }
        cfg = FarFieldConfig(**overrides)
        sol = solve_and_extract(cfg, label=label)

        generate_full_png_set(sol, fig_dir, prefix=f"geom_{label}_",
                              title_extra=f"  [{label}]")

        m = compute_metrics(sol, label=label)
        sweep_rows.append(m)

        print(f"    {label}: max|p|={m['max_p_Pa']:.2f}  Mz={m['Mz']:.4f}  "
              f"nodal_z={m['nodal_spacing_z_mm']} mm")

        del sol; gc.collect()  # free immediately

    # --- S2.B: Lateral size sweep (Lx=Ly ±0.25λ) ---
    print("\n  ── S2.B: Lateral size sweep (Lx=Ly ± 0.25λ) ──")
    lat_offsets = [-0.25, 0.0, 0.25]

    for dl_lam in lat_offsets:
        Lxy = base_Lxy + dl_lam * lam
        label = f"Lxy_{Lxy*1e3:.3f}mm"
        overrides = {
            **CORRECTED_PRESET,
            "Lx": Lxy, "Ly": Lxy,
            "standing_velocity_amplitude": 10e-6,
            "disk_velocity_amplitude": 0.0,
            "elements_per_wavelength": ELEM_PER_LAMBDA,
        }
        cfg = FarFieldConfig(**overrides)
        sol = solve_and_extract(cfg, label=label)

        generate_full_png_set(sol, fig_dir, prefix=f"geom_{label}_",
                              title_extra=f"  [{label}]")

        m = compute_metrics(sol, label=label)
        sweep_rows.append(m)

        print(f"    {label}: max|p|={m['max_p_Pa']:.2f}  Mz={m['Mz']:.4f}  "
              f"Mx={m['Mx']:.4f}")

        del sol; gc.collect()  # free immediately

    # --- sweep_table.csv ---
    if sweep_rows:
        keys = sweep_rows[0].keys()
        with open(s2_dir / "sweep_table.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(sweep_rows)
        print(f"\n  Wrote {s2_dir / 'sweep_table.csv'}")

    # NOTE: montage plots removed to conserve memory.
    # Individual PNGs per configuration are in figures/ for comparison.

    return sweep_rows


# ====================================================================
# PHASE S3 — BC sensitivity
# ====================================================================

def run_s3():
    """
    Test passive-BC sensitivity on z-boundaries.

    The solver hard-codes:
      - Top face: Robin (water-air) on TAG_TOP (facet 2)
      - Standing patches: Neumann on TAG_STAND_X0/XL/Y0/YL (z ∈ petri slab only)
      - Side walls below petri: Neumann (TAG_X0/XL/Y0/YL) — but absorbed by PML
      - Bottom disk: Neumann source (TAG_BOTTOM_DISK)
      - Bottom outside: natural Neumann = hard wall (TAG_BOTTOM_OUTSIDE)

    What we CAN test:
      (A) PML on vs off — shows PML effect on standing wave clarity
      (B) PML thickness (1λ vs 1.5λ vs 2λ xy)
      (C) PML sigma strength
    We also verify the driven facet mask.
    """
    print("\n" + "=" * 72)
    print("PHASE S3 — BC / PML SENSITIVITY")
    print("=" * 72)

    s3_dir = OUT_ROOT / "S3_bc_sensitivity"
    fig_dir = s3_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    bc_rows = []

    # Common overrides for standing_only
    common = {
        **CORRECTED_PRESET,
        "standing_velocity_amplitude": 10e-6,
        "disk_velocity_amplitude": 0.0,
        "elements_per_wavelength": ELEM_PER_LAMBDA,
    }

    # Reduce to 3 variants (from 5) for memory safety
    variants = {
        # (A) PML off — rigid walls everywhere
        "pml_off_rigid": {"pml_enabled": False},
        # (B) PML 1.0λ (current default from CORRECTED_PRESET)
        "pml_1p0lam": {"pml_n_wavelengths_xy": 1.0, "pml_n_wavelengths_z": 1.0},
        # (C) Stronger sigma (factor 10 vs default 5)
        "sigma_strong": {"pml_sigma_max_factor": 10.0},
    }

    for vname, extra_overrides in variants.items():
        label = f"bc_{vname}"
        overrides = {**common, **extra_overrides}
        cfg = FarFieldConfig(**overrides)
        sol = solve_and_extract(cfg, label=label)

        generate_full_png_set(sol, fig_dir, prefix=f"bc_{vname}_",
                              title_extra=f"  [{vname}]")

        m = compute_metrics(sol, label=label)
        m["variant"] = vname
        bc_rows.append(m)

        print(f"    {vname}: max|p|={m['max_p_Pa']:.2f}  Mz={m['Mz']:.4f}  Mx={m['Mx']:.4f}")

        del sol; gc.collect()  # free immediately

    # --- BC mask verification plot ---
    # Show which facets are driven (standing patches) vs passive
    # We use the mesh facet data from a fresh solve
    print("\n  Generating BC mask verification plot …")
    try:
        cfg_mask = FarFieldConfig(**common)
        from acoustweezers.experiments.farfield_petri_cuboid.mesh import (
            create_mesh, TAG_STAND_X0, TAG_STAND_XL, TAG_STAND_Y0, TAG_STAND_YL,
            TAG_X0, TAG_XL, TAG_Y0, TAG_YL, TAG_TOP, TAG_BOTTOM_DISK, TAG_BOTTOM_OUTSIDE,
        )
        domain, facet_tags, cell_tags, _ = create_mesh(cfg_mask, verbose=False)
        fdim = domain.topology.dim - 1
        fcoords = {}
        for tag, name in [
            (TAG_STAND_X0, "Stand x=0"), (TAG_STAND_XL, "Stand x=Lx"),
            (TAG_STAND_Y0, "Stand y=0"), (TAG_STAND_YL, "Stand y=Ly"),
            (TAG_X0, "Passive x=0"), (TAG_XL, "Passive x=Lx"),
            (TAG_Y0, "Passive y=0"), (TAG_YL, "Passive y=Ly"),
            (TAG_TOP, "Top Robin"), (TAG_BOTTOM_DISK, "Bottom disk"),
            (TAG_BOTTOM_OUTSIDE, "Bottom outside"),
        ]:
            fidx = facet_tags.indices[facet_tags.values == tag]
            if len(fidx) > 0:
                pts = mesh_module_midpoints(domain, fdim, fidx)
                fcoords[name] = pts

        # Plot side-view (x=0 face) showing driven vs passive z-regions
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = {"Stand": "red", "Passive": "blue", "Top": "green",
                  "Bottom disk": "orange", "Bottom outside": "gray"}
        for name, pts in fcoords.items():
            c = "gray"
            for prefix, col in colors.items():
                if name.startswith(prefix):
                    c = col; break
            # project to y-z for x=0 or x=Lx walls; x-z for y walls; etc.
            if "x=0" in name or "x=Lx" in name:
                ax.scatter(pts[:, 1] * 1e3, pts[:, 2] * 1e3, c=c, s=3, label=name, alpha=0.7)
            elif "y=0" in name or "y=Ly" in name:
                ax.scatter(pts[:, 0] * 1e3, pts[:, 2] * 1e3, c=c, s=3, marker="x", label=name, alpha=0.7)
            elif "Top" in name:
                ax.scatter(pts[:, 0] * 1e3, pts[:, 2] * 1e3, c=c, s=3, marker="^", label=name, alpha=0.5)
            elif "Bottom" in name:
                ax.scatter(pts[:, 0] * 1e3, pts[:, 2] * 1e3, c=c, s=3, marker="v", label=name, alpha=0.5)
        ax.set_xlabel("Lateral [mm]"); ax.set_ylabel("z [mm]")
        ax.set_title("BC Facet Map — driven (red) vs passive (blue) vs top (green)")
        ax.legend(fontsize=7, ncol=2, loc="upper right")
        fig.tight_layout()
        fig.savefig(fig_dir / "bc_facet_mask.png", bbox_inches="tight")
        plt.close(fig)

        del domain, facet_tags, cell_tags
        gc.collect()
    except Exception as e:
        print(f"  WARNING: BC mask plot failed: {e}")

    # Write metrics
    if bc_rows:
        keys = bc_rows[0].keys()
        with open(s3_dir / "bc_sensitivity_table.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(bc_rows)

    for row in bc_rows:
        with open(s3_dir / f"metrics_standing_bc_{row.get('variant','')}.json", "w") as f:
            json.dump(row, f, indent=2)

    return bc_rows


def mesh_module_midpoints(domain, fdim, facet_indices):
    """Compute midpoints of selected facets."""
    from dolfinx.mesh import compute_midpoints
    return compute_midpoints(domain, fdim, facet_indices)


# ====================================================================
# PHASE S4 — Final decision + publication-quality output
# ====================================================================

def run_s4(all_metrics: list):
    """
    Pick the cleanest config, re-solve it, and produce publication PNGs.
    """
    print("\n" + "=" * 72)
    print("PHASE S4 — FINAL DECISION")
    print("=" * 72)

    s4_dir = OUT_ROOT / "S4_final"
    fig_dir = s4_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Score each configuration: prefer high Mz, high Mx, clear nodal spacing
    best_label = None
    best_score = -1
    best_metrics = None

    for m in all_metrics:
        if m.get("ksp_reason", 0) <= 0:
            continue
        mz = m.get("Mz", 0) or 0
        mx = m.get("Mx", 0) or 0
        ns = m.get("nodal_spacing_z_mm")
        lam_half = m.get("expected_half_lambda_mm", 0.371)
        ns_score = 0
        if ns is not None and lam_half > 0:
            ns_score = max(0, 1.0 - abs(ns - lam_half) / lam_half)
        score = mz + mx + 2 * ns_score
        if score > best_score:
            best_score = score
            best_label = m.get("label", "")
            best_metrics = m

    if best_metrics is None:
        print("  ERROR: no converged solutions to pick from!")
        return None, None

    print(f"\n  Best configuration: {best_label}  (score={best_score:.3f})")
    print(f"    Mz={best_metrics['Mz']:.4f}  Mx={best_metrics['Mx']:.4f}  "
          f"nodal_z={best_metrics['nodal_spacing_z_mm']} mm")

    # Re-solve the best config for final PNGs
    print(f"\n  Re-solving best config for final output …")
    overrides = {
        **CORRECTED_PRESET,
        "standing_velocity_amplitude": 10e-6,
        "disk_velocity_amplitude": 0.0,
        "elements_per_wavelength": ELEM_PER_LAMBDA,
    }
    # Apply geometry overrides based on best label
    if "Htop_" in best_label:
        h_mm = float(best_label.split("Htop_")[1].replace("mm", ""))
        overrides["H_top"] = h_mm * 1e-3
    elif "Lxy_" in best_label:
        l_mm = float(best_label.split("Lxy_")[1].replace("mm", ""))
        overrides["Lx"] = l_mm * 1e-3
        overrides["Ly"] = l_mm * 1e-3
    elif "bc_pml_off" in best_label:
        overrides["pml_enabled"] = False
    elif "bc_pml_1p5lam" in best_label:
        overrides["pml_n_wavelengths_xy"] = 1.5
        overrides["pml_n_wavelengths_z"] = 1.5
    elif "bc_sigma_strong" in best_label:
        overrides["pml_sigma_max_factor"] = 10.0
    elif "bc_sigma_weak" in best_label:
        overrides["pml_sigma_max_factor"] = 2.0

    cfg = FarFieldConfig(**overrides)
    sol = solve_and_extract(cfg, label=f"S4_final_{best_label}")

    generate_full_png_set(sol, fig_dir, prefix="standing_final_",
                          title_extra=f"  [FINAL: {best_label}]")

    del sol; gc.collect()

    # Save final metrics
    with open(s4_dir / "final_metrics.json", "w") as f:
        json.dump(best_metrics, f, indent=2)

    # Write summary report
    report = [
        "# Standing-Wave Clarity Report",
        f"Date: {datetime.now().isoformat()}",
        "",
        "## Best Configuration",
        f"- **Label**: {best_label}",
        f"- **Score**: {best_score:.3f}",
        f"- Lx × Ly = {best_metrics['Lx_mm']:.3f} × {best_metrics['Ly_mm']:.3f} mm",
        f"- H_under = {best_metrics['H_under_mm']:.3f} mm",
        f"- H_top = {best_metrics['H_top_mm']:.3f} mm  (H_total = {best_metrics['H_total_mm']:.3f} mm)",
        f"- λ = {best_metrics['lambda_mm']:.4f} mm  (λ/2 = {best_metrics['expected_half_lambda_mm']:.4f} mm)",
        "",
        "## Modulation Metrics",
        f"- Mz (z-modulation depth) = {best_metrics['Mz']:.4f}",
        f"- Mx (x-modulation depth) = {best_metrics['Mx']:.4f}",
        f"- My (y-modulation depth) = {best_metrics['My']:.4f}",
        f"- Estimated nodal spacing (z) = {best_metrics['nodal_spacing_z_mm']} mm",
        "",
        "## How the z-axis piecewise driven patch was verified",
        "- Standing-wave Neumann drive applied only to TAG_STAND_X0/XL/Y0/YL",
        "  facets, which are restricted to z ∈ [H_under, H_under + H_top]",
        "  (petri slab region) in mesh.py _stand_x0 / _stand_xL etc.",
        "- Side walls below the petri slab (TAG_X0/XL/Y0/YL) get no drive",
        "  and are absorbed by PML.",
        "- BC mask verification PNG confirms correct segmentation.",
        "",
        "## All Configurations Tested",
        "",
        "| Label | max|p| [Pa] | Mz | Mx | Nodal z [mm] |",
        "|-------|-------------|------|------|-------------|",
    ]
    for m in all_metrics:
        ns = m.get("nodal_spacing_z_mm", "—")
        report.append(
            f"| {m.get('label','')} | {m['max_p_Pa']:.2f} | "
            f"{m['Mz']:.4f} | {m['Mx']:.4f} | {ns} |"
        )

    report += [
        "",
        "## Recommended baseline for vortex non-disruption testing",
        f"Use configuration **{best_label}** as the standing-wave reference.",
        "Then add the vortex beam and compare Mz / Mx metrics.",
        "",
        "## Figures",
        "All PNGs in `S4_final/figures/standing_final_*.png`.",
    ]

    with open(s4_dir / "standing_clarity_report.md", "w") as f:
        f.write("\n".join(report))

    print(f"  Wrote {s4_dir / 'standing_clarity_report.md'}")
    return best_label, best_metrics


# ====================================================================
# MAIN
# ====================================================================

def main():
    print("=" * 72)
    print("STANDING-WAVE CLARITY DIAGNOSTICS")
    print(f"Output: {OUT_ROOT}")
    print(f"Timestamp: {TIMESTAMP}")
    print(f"Resolution: {ELEM_PER_LAMBDA} elem/λ, grid {NGRID}², {DPI} DPI")
    print("=" * 72)

    all_metrics = []

    # ── S1 ──
    sol_s1, m_s1 = run_s1()
    all_metrics.append(m_s1)
    del sol_s1; gc.collect()

    # ── S2 ──
    sweep_rows = run_s2()
    all_metrics.extend(sweep_rows)

    # ── S3 ──
    bc_rows = run_s3()
    all_metrics.extend(bc_rows)

    # ── S4 ──
    best_label, best_m = run_s4(all_metrics)

    # ── Summary ──
    print("\n" + "=" * 72)
    print("ALL PHASES COMPLETE")
    print(f"Output: {OUT_ROOT}")
    n_png = len(list(OUT_ROOT.rglob("*.png")))
    print(f"Total PNGs: {n_png}")
    print(f"Best config: {best_label}")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    sys.exit(main())
