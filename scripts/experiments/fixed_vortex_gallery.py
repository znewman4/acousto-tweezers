#!/usr/bin/env python3
"""
Fixed Vortex Gallery — Physical domain only (no PML artefacts)
===============================================================

Diagnosis (Feb 24):
  The standing-wave BCs sit at x=0 / x=Lx (outer mesh boundary),
  which is INSIDE the PML sponge layer.  The PML absorbs the
  standing waves before they reach the physical interior.

  Result: physical-domain max|p| ≈ 1.0 Pa (vortex beam only),
          while PML max|p| = 24.5 Pa (standing-wave artifact).

  Previous galleries sampled [0, Lx] × [0, Ly] including PML,
  making panels appear black (physical centre ≈ 2 % of PML-edge scale).

Fix:
  • Clip interpolation grid to physical domain only
  • Filter interpolator DOFs to physical-domain only
  • Per-panel auto-scale within physical domain → patterns visible
  • Add log-scale panels for dynamic-range overview

Also generates a physics-audit section comparing PML vs physical pressure.

Output → results/fixed_gallery_<timestamp>/
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
OUT_DIR = PROJECT_ROOT / "results" / f"fixed_gallery_{TIMESTAMP}"
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
from matplotlib.colors import LogNorm
from scipy.interpolate import NearestNDInterpolator

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
    """Memory-efficient snapshot of a FEniCSx solution."""
    def __init__(self, sol):
        self.coords = sol.coords.copy()
        self.p_values = sol.p_values.copy()
        self.cfg = sol.cfg
        self.dofs = sol.dofs
        self.max_pressure = sol.max_pressure
        self.ksp_converged_reason = sol.ksp_converged_reason

        # Pre-compute physical domain mask
        # Lateral PML only below petri dish (z < H_under)
        t_xy = self.cfg.t_pml_xy
        t_z = self.cfg.t_pml_z
        H_under = self.cfg.H_under
        x, y, z = self.coords[:, 0], self.coords[:, 1], self.coords[:, 2]
        in_pml_x = ((x < t_xy) | (x > self.cfg.Lx - t_xy)) & (z < H_under)
        in_pml_y = ((y < t_xy) | (y > self.cfg.Ly - t_xy)) & (z < H_under)
        in_pml_z = z < t_z
        self.is_physical = ~(in_pml_x | in_pml_y | in_pml_z)

        self.phys_coords = self.coords[self.is_physical]
        self.phys_p = self.p_values[self.is_physical]
        self.phys_max_pressure = float(np.abs(self.phys_p).max())

        # Physical domain bounds
        self.phys_xmin = t_xy
        self.phys_xmax = self.cfg.Lx - t_xy
        self.phys_ymin = t_xy
        self.phys_ymax = self.cfg.Ly - t_xy
        self.phys_zmin = t_z
        self.phys_zmax = self.cfg.H_total  # no top PML


def solve_case(overrides, label=""):
    cfg = FarFieldConfig(**overrides)
    t0 = time.time()
    sol = solve_helmholtz(cfg, verbose=True, petsc_options=PETSC_OPTS,
                          export_fields=False)
    dt = time.time() - t0
    lsol = LightSol(sol)
    del sol; gc.collect()
    print(f"  [{label}] global max|p|={lsol.max_pressure:.3f} Pa  "
          f"phys max|p|={lsol.phys_max_pressure:.3f} Pa  "
          f"KSP={lsol.ksp_converged_reason}  {dt:.1f}s")
    return lsol


def phys_slice_xy(lsol, z_val, n=NGRID):
    """XY slice using only PHYSICAL-domain DOFs, clipped to physical bounds."""
    interp_re = NearestNDInterpolator(lsol.phys_coords, np.real(lsol.phys_p))
    interp_im = NearestNDInterpolator(lsol.phys_coords, np.imag(lsol.phys_p))
    xg = np.linspace(lsol.phys_xmin, lsol.phys_xmax, n)
    yg = np.linspace(lsol.phys_ymin, lsol.phys_ymax, n)
    X, Y = np.meshgrid(xg, yg)
    pts = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, z_val)])
    return xg, yg, (interp_re(pts) + 1j * interp_im(pts)).reshape(X.shape)


def phys_slice_xz(lsol, y_val, n=NGRID):
    """XZ slice using only PHYSICAL-domain DOFs, clipped to physical bounds."""
    interp_re = NearestNDInterpolator(lsol.phys_coords, np.real(lsol.phys_p))
    interp_im = NearestNDInterpolator(lsol.phys_coords, np.imag(lsol.phys_p))
    xg = np.linspace(lsol.phys_xmin, lsol.phys_xmax, n)
    zg = np.linspace(lsol.phys_zmin, lsol.phys_zmax, n)
    X, Z = np.meshgrid(xg, zg)
    pts = np.column_stack([X.ravel(), np.full(X.size, y_val), Z.ravel()])
    return xg, zg, (interp_re(pts) + 1j * interp_im(pts)).reshape(X.shape)


def plot_xy(xg, yg, pc, title, fname, vmax=None, log_floor=None):
    """Plot magnitude + phase, physical domain only. Returns png count."""
    pmag = np.abs(pc)
    count = 0

    # Linear magnitude
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.pcolormesh(xg * 1e3, yg * 1e3, pmag,
                       shading="auto", cmap="inferno",
                       vmin=0, vmax=vmax)
    ax.set_title(f"{title}\nmax|p| = {pmag.max():.4f} Pa")
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax, label="|p| [Pa]")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{fname}_mag.png", bbox_inches="tight")
    plt.close(fig)
    count += 1

    # Log magnitude
    fig, ax = plt.subplots(figsize=(8, 7))
    floor = log_floor if log_floor else max(pmag[pmag > 0].min() if np.any(pmag > 0) else 1e-6, 1e-6)
    pmag_log = np.clip(pmag, floor, None)
    im = ax.pcolormesh(xg * 1e3, yg * 1e3, pmag_log,
                       shading="auto", cmap="inferno",
                       norm=LogNorm(vmin=floor, vmax=max(pmag.max(), floor * 10)))
    ax.set_title(f"{title} — Log Scale")
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax, label="|p| [Pa]")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{fname}_log.png", bbox_inches="tight")
    plt.close(fig)
    count += 1

    # Phase
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
    count += 1

    return count


def plot_xz(xg, zg, pc, title, fname, cfg, vmax=None):
    """XZ slice with petri annotation. Returns png count."""
    pmag = np.abs(pc)
    count = 0
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
        ax.set_title(f"{title} — {label.split('[')[0].strip()}\nphys max = {pmag.max():.4f} Pa")
        plt.colorbar(im, ax=ax, label=label)
        ax.legend(loc="lower right", fontsize=9)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"{fname}_{suffix}.png", bbox_inches="tight")
        plt.close(fig)
        count += 1

    # Log magnitude
    fig, ax = plt.subplots(figsize=(9, 7))
    floor = max(pmag[pmag > 0].min() if np.any(pmag > 0) else 1e-6, 1e-6)
    pmag_log = np.clip(pmag, floor, None)
    im = ax.pcolormesh(xg * 1e3, zg * 1e3, pmag_log,
                       shading="auto", cmap="inferno",
                       norm=LogNorm(vmin=floor, vmax=max(pmag.max(), floor * 10)))
    ax.axhspan(cfg.H_under * 1e3, cfg.H_total * 1e3,
               alpha=0.08, color="cyan", label="Petri slab")
    ax.axhline(cfg.H_under * 1e3, color="cyan", ls="--", lw=0.8, alpha=0.6)
    ax.axhline(cfg.H_total * 1e3, color="cyan", ls="--", lw=0.8, alpha=0.6)
    ax.set_xlabel("x [mm]"); ax.set_ylabel("z [mm]")
    ax.set_title(f"{title} — Log |p|")
    plt.colorbar(im, ax=ax, label="|p| [Pa]")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{fname}_log.png", bbox_inches="tight")
    plt.close(fig)
    count += 1

    return count


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
    t_xy_mm = cfg_base.t_pml_xy * 1e3

    print("=" * 72)
    print("FIXED VORTEX GALLERY — Physical Domain Only")
    print(f"  Trap plane z = {trap_z*1e3:.2f} mm")
    print(f"  f = {COMMON['lens_focal_length']*1e3:.0f} mm")
    print(f"  λ = {lam_mm:.3f} mm")
    print(f"  PML thickness = {t_xy_mm:.3f} mm")
    print(f"  Physical x: [{t_xy_mm:.3f}, {(cfg_base.Lx-cfg_base.t_pml_xy)*1e3:.3f}] mm")
    print("=" * 72)

    # ==============================================================
    # PHASE 1: Reference solves
    # ==============================================================
    print(f"\n{'='*72}")
    print("PHASE 1: Reference solves (standing-only, vortex-only, combined)")
    print(f"{'='*72}")

    cases = {
        "standing_only": {
            **COMMON,
            "disk_velocity_amplitude": 0.0,
        },
        "vortex_only": {
            **COMMON,
            "standing_velocity_amplitude": 0.0,
        },
        "combined": {
            **COMMON,
        },
    }

    solutions = {}
    metrics = {}
    for cn, ov in cases.items():
        lsol = solve_case(ov, label=cn)
        solutions[cn] = lsol
        metrics[cn] = {
            "global_max_p_Pa": float(lsol.max_pressure),
            "physical_max_p_Pa": float(lsol.phys_max_pressure),
        }

    all_results["case_metrics"] = metrics

    # ==============================================================
    # PHASE 2: Physics audit — PML vs Physical
    # ==============================================================
    print(f"\n{'='*72}")
    print("PHASE 2: Physics Audit — PML vs Physical domain")
    print(f"{'='*72}")

    audit = {}
    for cn, ls in solutions.items():
        pml_max = float(np.abs(ls.p_values[~ls.is_physical]).max()) if (~ls.is_physical).any() else 0
        phys_max = float(ls.phys_max_pressure)
        ratio = pml_max / phys_max if phys_max > 0 else float('inf')
        print(f"  {cn:15s}: global={ls.max_pressure:.3f}  PML={pml_max:.3f}  "
              f"Physical={phys_max:.3f}  PML/Phys={ratio:.1f}x")
        audit[cn] = {
            "global_max": float(ls.max_pressure),
            "pml_max": pml_max,
            "physical_max": phys_max,
            "pml_to_phys_ratio": ratio,
        }

        # Find physical max location
        phys_idx = np.where(ls.is_physical)[0]
        best_phys = phys_idx[np.argmax(np.abs(ls.p_values[ls.is_physical]))]
        audit[cn]["physical_max_location_mm"] = (ls.coords[best_phys] * 1e3).tolist()

    all_results["physics_audit"] = audit

    # Audit bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    names = list(audit.keys())
    x = np.arange(len(names))
    pml_vals = [audit[n]["pml_max"] for n in names]
    phys_vals = [audit[n]["physical_max"] for n in names]
    w = 0.35
    ax.bar(x - w / 2, pml_vals, w, label="PML max |p|", color="red", alpha=0.7)
    ax.bar(x + w / 2, phys_vals, w, label="Physical max |p|", color="blue", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", "\n") for n in names])
    ax.set_ylabel("|p| [Pa]")
    ax.set_title("PML vs Physical Domain — Max Pressure\n"
                 "Standing waves trapped in PML (BC at mesh boundary inside PML)")
    ax.legend()
    for i, (pv, phv) in enumerate(zip(pml_vals, phys_vals)):
        ax.text(i - w / 2, pv + 0.3, f"{pv:.2f}", ha="center", fontsize=9)
        ax.text(i + w / 2, phv + 0.3, f"{phv:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "audit_pml_vs_physical.png", bbox_inches="tight")
    plt.close(fig)
    png_count += 1

    # ==============================================================
    # PHASE 3: Per-case gallery (physical domain only)
    # ==============================================================
    print(f"\n{'='*72}")
    print("PHASE 3: Per-case gallery (physical domain)")
    print(f"{'='*72}")

    for cn, lsol in solutions.items():
        cfg = lsol.cfg
        label = cn.replace("_", " ").title()
        y_mid = (cfg.Ly / 2)

        # XY at trap plane
        xg, yg, pc = phys_slice_xy(lsol, trap_z)
        png_count += plot_xy(xg, yg, pc,
                             f"{label} — XY trap z={trap_z*1e3:.1f}mm",
                             f"{cn}_xy_trap")

        # XY at vortex focus plane (z = f_lens)
        z_focus = min(COMMON["lens_focal_length"], cfg.H_total - 1e-5)
        xg, yg, pc = phys_slice_xy(lsol, z_focus)
        png_count += plot_xy(xg, yg, pc,
                             f"{label} — XY focus z={z_focus*1e3:.1f}mm",
                             f"{cn}_xy_focus")

        # XZ mid
        xg, zg, pc = phys_slice_xz(lsol, y_mid)
        png_count += plot_xz(xg, zg, pc,
                             f"{label} — XZ y={y_mid*1e3:.1f}mm",
                             f"{cn}_xz_mid", cfg)

        # Centerline (physical domain z-range)
        zg_cl = np.linspace(lsol.phys_zmin, lsol.phys_zmax, 600)
        cx, cy = cfg.Lx / 2, cfg.Ly / 2
        interp_re = NearestNDInterpolator(lsol.phys_coords, np.real(lsol.phys_p))
        interp_im = NearestNDInterpolator(lsol.phys_coords, np.imag(lsol.phys_p))
        pts = np.column_stack([np.full(600, cx), np.full(600, cy), zg_cl])
        pmag_cl = np.abs(interp_re(pts) + 1j * interp_im(pts))
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(zg_cl * 1e3, pmag_cl, "k-", lw=1.5)
        ax.axvspan(cfg.H_under * 1e3, cfg.H_total * 1e3,
                   alpha=0.12, color="cyan", label="Petri slab")
        ax.axvline(COMMON["lens_focal_length"] * 1e3, color="red", ls=":",
                   lw=1, alpha=0.6, label=f"f = {COMMON['lens_focal_length']*1e3:.0f} mm")
        ax.set_xlabel("z [mm]"); ax.set_ylabel("|p| [Pa]")
        ax.set_title(f"{label} — centerline |p|(z)  [Physical domain]")
        ax.legend(); fig.tight_layout()
        fig.savefig(FIG_DIR / f"{cn}_centerline.png", bbox_inches="tight")
        plt.close(fig)
        png_count += 1

    # ==============================================================
    # PHASE 4: 3-way comparison panels
    # ==============================================================
    print(f"\n{'='*72}")
    print("PHASE 4: Comparison panels")
    print(f"{'='*72}")

    # XY trap-plane comparison (linear)
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    for ax, cn in zip(axes, ["standing_only", "vortex_only", "combined"]):
        ls = solutions[cn]
        xg, yg, pc = phys_slice_xy(ls, trap_z, n=400)
        pmag = np.abs(pc)
        im = ax.pcolormesh(xg * 1e3, yg * 1e3, pmag,
                           shading="auto", cmap="inferno")
        ax.set_title(f"{cn.replace('_',' ').title()}\nphys max = {pmag.max():.4f} Pa", fontsize=13)
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, label="|p| [Pa]")
    fig.suptitle(f"XY Trap Plane z={trap_z*1e3:.1f}mm — Physical Domain Only\n"
                 f"f = {COMMON['lens_focal_length']*1e3:.0f}mm",
                 fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "comparison_xy_trap.png", bbox_inches="tight")
    plt.close(fig)
    png_count += 1

    # XY trap-plane comparison (log-scale)
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    for ax, cn in zip(axes, ["standing_only", "vortex_only", "combined"]):
        ls = solutions[cn]
        xg, yg, pc = phys_slice_xy(ls, trap_z, n=400)
        pmag = np.abs(pc)
        floor = max(pmag[pmag > 0].min() if np.any(pmag > 0) else 1e-6, 1e-4)
        pmag_log = np.clip(pmag, floor, None)
        im = ax.pcolormesh(xg * 1e3, yg * 1e3, pmag_log,
                           shading="auto", cmap="inferno",
                           norm=LogNorm(vmin=floor, vmax=max(pmag.max(), floor * 10)))
        ax.set_title(f"{cn.replace('_',' ').title()}\nphys max = {pmag.max():.4f} Pa", fontsize=13)
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, label="|p| [Pa]")
    fig.suptitle(f"XY Trap Plane z={trap_z*1e3:.1f}mm — Log Scale — Physical Domain Only",
                 fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "comparison_xy_trap_log.png", bbox_inches="tight")
    plt.close(fig)
    png_count += 1

    # Phase comparison
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    for ax, cn in zip(axes, ["standing_only", "vortex_only", "combined"]):
        ls = solutions[cn]
        xg, yg, pc = phys_slice_xy(ls, trap_z, n=400)
        im = ax.pcolormesh(xg * 1e3, yg * 1e3, np.angle(pc),
                           shading="auto", cmap="twilight",
                           vmin=-np.pi, vmax=np.pi)
        ax.set_title(f"{cn.replace('_',' ').title()}", fontsize=13)
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, label="Phase [rad]")
    fig.suptitle(f"Phase — Trap Plane — Physical Domain Only", fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "comparison_phase_trap.png", bbox_inches="tight")
    plt.close(fig)
    png_count += 1

    # XZ 3-way comparison
    fig, axes = plt.subplots(1, 3, figsize=(26, 7))
    for ax, cn in zip(axes, ["standing_only", "vortex_only", "combined"]):
        ls = solutions[cn]
        xg, zg, pc = phys_slice_xz(ls, ls.cfg.Ly / 2, n=400)
        pmag = np.abs(pc)
        im = ax.pcolormesh(xg * 1e3, zg * 1e3, pmag,
                           shading="auto", cmap="inferno")
        ax.axhline(ls.cfg.H_under * 1e3, color="cyan", ls="--", lw=0.8, alpha=0.7)
        ax.axhline(ls.cfg.H_total * 1e3, color="cyan", ls="--", lw=0.8, alpha=0.7)
        ax.set_title(f"{cn.replace('_',' ').title()}\nphys max = {pmag.max():.4f} Pa", fontsize=13)
        ax.set_xlabel("x [mm]"); ax.set_ylabel("z [mm]")
        plt.colorbar(im, ax=ax, label="|p| [Pa]")
    fig.suptitle(f"XZ Mid-Plane — Physical Domain Only", fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "comparison_xz_mid.png", bbox_inches="tight")
    plt.close(fig)
    png_count += 1

    # Centerline overlay
    fig, ax = plt.subplots(figsize=(10, 6))
    for cn, col in [("standing_only", "blue"), ("vortex_only", "orange"), ("combined", "green")]:
        ls = solutions[cn]
        cfg = ls.cfg
        zg_cl = np.linspace(ls.phys_zmin, ls.phys_zmax, 600)
        cx, cy = cfg.Lx / 2, cfg.Ly / 2
        interp_re = NearestNDInterpolator(ls.phys_coords, np.real(ls.phys_p))
        interp_im = NearestNDInterpolator(ls.phys_coords, np.imag(ls.phys_p))
        pts = np.column_stack([np.full(600, cx), np.full(600, cy), zg_cl])
        pmag_cl = np.abs(interp_re(pts) + 1j * interp_im(pts))
        ax.plot(zg_cl * 1e3, pmag_cl, color=col, lw=1.5,
                label=f"{cn.replace('_', ' ').title()} (phys max={ls.phys_max_pressure:.3f})")
    cfg0 = solutions["standing_only"].cfg
    ax.axvspan(cfg0.H_under * 1e3, cfg0.H_total * 1e3,
               alpha=0.1, color="cyan", label="Petri slab")
    ax.axvline(COMMON["lens_focal_length"] * 1e3, color="red", ls=":", lw=1,
               alpha=0.6, label=f"Focus z = {COMMON['lens_focal_length']*1e3:.0f} mm")
    ax.set_xlabel("z [mm]"); ax.set_ylabel("|p| [Pa]")
    ax.set_title("Centerline |p|(z) — Physical Domain Only")
    ax.legend(fontsize=9); fig.tight_layout()
    fig.savefig(FIG_DIR / "comparison_centerline.png", bbox_inches="tight")
    plt.close(fig)
    png_count += 1

    # ==============================================================
    # PHASE 5: Z-height progression (all 3 cases)
    # ==============================================================
    print(f"\n{'='*72}")
    print("PHASE 5: Z-height progression")
    print(f"{'='*72}")

    Z_HEIGHTS_MM = [0.8, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    ncols = 3
    nrows = (len(Z_HEIGHTS_MM) + ncols - 1) // ncols

    for cn in ["standing_only", "vortex_only", "combined"]:
        ls = solutions[cn]
        cfg = ls.cfg
        label = cn.replace("_", " ").title()

        # Individual slices
        for z_mm in Z_HEIGHTS_MM:
            z_m = min(z_mm * 1e-3, cfg.H_total - 1e-5)
            xg, yg, pc = phys_slice_xy(ls, z_m, n=400)
            png_count += plot_xy(xg, yg, pc,
                                 f"{label} — z = {z_mm:.1f} mm",
                                 f"zprog_{cn}_z{z_mm:.1f}".replace(".", "p"))

        # Multi-panel: linear magnitude
        fig, axes = plt.subplots(nrows, ncols, figsize=(7*ncols, 6.5*nrows))
        axes_flat = axes.flatten()
        for i, z_mm in enumerate(Z_HEIGHTS_MM):
            z_m = min(z_mm * 1e-3, cfg.H_total - 1e-5)
            xg, yg, pc = phys_slice_xy(ls, z_m, n=300)
            pmag = np.abs(pc)
            ax = axes_flat[i]
            im = ax.pcolormesh(xg * 1e3, yg * 1e3, pmag,
                               shading="auto", cmap="inferno")
            region = "Petri" if z_mm >= cfg.H_under * 1e3 else "Bath"
            ax.set_title(f"z = {z_mm:.1f} mm  ({region})\nmax = {pmag.max():.4f} Pa")
            ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
            ax.set_aspect("equal")
            plt.colorbar(im, ax=ax, label="|p| [Pa]", shrink=0.80)
        for j in range(len(Z_HEIGHTS_MM), len(axes_flat)):
            axes_flat[j].set_visible(False)
        fig.suptitle(f"{label} — XY Slices (Physical Domain)\n"
                     f"f = {COMMON['lens_focal_length']*1e3:.0f} mm",
                     fontsize=16, y=1.01)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"zprog_{cn}_panel.png", bbox_inches="tight")
        plt.close(fig)
        png_count += 1

        # Multi-panel: LOG magnitude
        fig, axes = plt.subplots(nrows, ncols, figsize=(7*ncols, 6.5*nrows))
        axes_flat = axes.flatten()
        for i, z_mm in enumerate(Z_HEIGHTS_MM):
            z_m = min(z_mm * 1e-3, cfg.H_total - 1e-5)
            xg, yg, pc = phys_slice_xy(ls, z_m, n=300)
            pmag = np.abs(pc)
            ax = axes_flat[i]
            floor = max(pmag[pmag > 0].min() if np.any(pmag > 0) else 1e-6, 1e-4)
            pmag_log = np.clip(pmag, floor, None)
            im = ax.pcolormesh(xg * 1e3, yg * 1e3, pmag_log,
                               shading="auto", cmap="inferno",
                               norm=LogNorm(vmin=floor, vmax=max(pmag.max(), floor*10)))
            region = "Petri" if z_mm >= cfg.H_under * 1e3 else "Bath"
            ax.set_title(f"z = {z_mm:.1f} mm  ({region})\nmax = {pmag.max():.4f} Pa")
            ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
            ax.set_aspect("equal")
            plt.colorbar(im, ax=ax, label="|p| [Pa]", shrink=0.80)
        for j in range(len(Z_HEIGHTS_MM), len(axes_flat)):
            axes_flat[j].set_visible(False)
        fig.suptitle(f"{label} — Log |p| (Physical Domain)\n"
                     f"f = {COMMON['lens_focal_length']*1e3:.0f} mm",
                     fontsize=16, y=1.01)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"zprog_{cn}_panel_log.png", bbox_inches="tight")
        plt.close(fig)
        png_count += 1

        # Phase panel
        fig, axes = plt.subplots(nrows, ncols, figsize=(7*ncols, 6.5*nrows))
        axes_flat = axes.flatten()
        for i, z_mm in enumerate(Z_HEIGHTS_MM):
            z_m = min(z_mm * 1e-3, cfg.H_total - 1e-5)
            xg, yg, pc = phys_slice_xy(ls, z_m, n=300)
            ax = axes_flat[i]
            im = ax.pcolormesh(xg * 1e3, yg * 1e3, np.angle(pc),
                               shading="auto", cmap="twilight",
                               vmin=-np.pi, vmax=np.pi)
            region = "Petri" if z_mm >= cfg.H_under * 1e3 else "Bath"
            ax.set_title(f"z = {z_mm:.1f} mm  ({region})")
            ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
            ax.set_aspect("equal")
            plt.colorbar(im, ax=ax, label="Phase [rad]", shrink=0.80)
        for j in range(len(Z_HEIGHTS_MM), len(axes_flat)):
            axes_flat[j].set_visible(False)
        fig.suptitle(f"{label} — Phase (Physical Domain)", fontsize=16, y=1.01)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"zprog_{cn}_phase_panel.png", bbox_inches="tight")
        plt.close(fig)
        png_count += 1

    # ==============================================================
    # PHASE 6: XZ with z-height markers (3-way)
    # ==============================================================
    print(f"\n  3-way XZ with z-markers …")
    fig, axes = plt.subplots(1, 3, figsize=(26, 8))
    for ax, cn in zip(axes, ["standing_only", "vortex_only", "combined"]):
        ls = solutions[cn]
        xg, zg, pc = phys_slice_xz(ls, ls.cfg.Ly / 2, n=400)
        pmag = np.abs(pc)
        im = ax.pcolormesh(xg * 1e3, zg * 1e3, pmag,
                           shading="auto", cmap="inferno")
        for z_mm in Z_HEIGHTS_MM:
            ax.axhline(z_mm, color="white", ls=":", lw=0.5, alpha=0.6)
        ax.axhline(ls.cfg.H_under * 1e3, color="cyan", ls="--", lw=1)
        ax.axhline(ls.cfg.H_total * 1e3, color="cyan", ls="--", lw=1)
        ax.set_title(f"{cn.replace('_',' ').title()}\nphys max = {pmag.max():.4f} Pa")
        ax.set_xlabel("x [mm]"); ax.set_ylabel("z [mm]")
        plt.colorbar(im, ax=ax, label="|p| [Pa]")
    fig.suptitle("XZ Physical Domain Only — with Z-slice Markers",
                 fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "comparison_xz_zslices.png", bbox_inches="tight")
    plt.close(fig)
    png_count += 1

    # ==============================================================
    # Save metadata
    # ==============================================================
    all_results["notes"] = (
        "Standing-wave BCs are at mesh boundary (x=0, x=Lx), which sit inside "
        "the PML sponge layer. PML absorbs standing waves before they reach the "
        "physical domain. Physical-domain max|p| ~ 1 Pa (vortex beam only). "
        "PML max|p| ~ 24 Pa (standing-wave artifact). "
        "All plots now clipped to physical domain with physical-only DOF interpolation."
    )

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
