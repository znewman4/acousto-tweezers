#!/usr/bin/env python3
"""
Corrected Physical Model — H_bath × f Sweep + Interaction Check
================================================================

Implements the corrected two-region model:
  - H_petri = 2 mm (fixed)
  - H_bath  ∈ {3, 4, 5, 6, 7} mm (sweep)
  - f_lens  ∈ {2, 3, 4, 5, 6} mm (sweep, must be < H_bath)
  - Standing-wave BC ONLY in petri slab (z ∈ [H_bath, H_total])
  - Top BC: fixed water–air Robin  (Z_air = ρ_air · c_air = 411.6 Pa·s/m)
  - V_disk = 1 µm/s, V_stand = 10 µm/s

After sweep, selects best geometry and runs interaction check.

Usage:
    python scripts/experiments/corrected_model_sweep.py
"""
from __future__ import annotations

import gc
import csv
import json
import os
import sys
import time
import traceback
import numpy as np
from dataclasses import replace
from datetime import datetime
from pathlib import Path

# Thread controls – cap at 8 to reduce MUMPS per-thread memory
NCORES = os.cpu_count() or 8
OMP_THREADS = min(8, max(1, NCORES // 2))
os.environ["OMP_NUM_THREADS"] = str(OMP_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(OMP_THREADS)
os.environ["MKL_NUM_THREADS"] = str(OMP_THREADS)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from acoustweezers.experiments.farfield_petri_cuboid.config import FarFieldConfig
from acoustweezers.experiments.farfield_petri_cuboid.solve_pressure import solve_helmholtz
from acoustweezers.experiments.farfield_petri_cuboid.post import (
    slice_xy, slice_xz, centerline_z,
)

# ═════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═════════════════════════════════════════════════════════════════════
H_PETRI = 2e-3          # fixed petri slab thickness [m]
LX = 6e-3               # lateral domain
LY = 6e-3
FREQ = 2.0e6
DISK_RADIUS = 1.0e-3
ELEM_PER_LAMBDA = 4     # eps=4: fits in 30 GB RAM for all domains

# Amplitudes for interaction check
V_DISK = 1e-6            # 1 µm/s
V_STAND = 10e-6          # 10 µm/s

# Sweep parameters
H_BATH_LIST = [3e-3, 4e-3, 5e-3, 6e-3, 7e-3]   # m
F_LENS_LIST = [2e-3, 3e-3, 4e-3, 5e-3, 6e-3]    # m

PETSC_OPTS = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "mat_mumps_icntl_14": 30,    # low memory overhead (30% relaxation)
    "mat_mumps_icntl_23": 0,     # let MUMPS manage memory automatically
    "mat_mumps_icntl_28": 2,     # parallel analysis
    "mat_mumps_icntl_29": 2,     # ParMETIS ordering (less fill-in)
}


# ═════════════════════════════════════════════════════════════════════
#  HELPERS
# ═════════════════════════════════════════════════════════════════════

def make_cfg(H_bath, f_lens, V_disk=V_DISK, V_stand=0.0, pml_enabled=True,
             eps=ELEM_PER_LAMBDA):
    """Build a FarFieldConfig for the corrected model."""
    return FarFieldConfig(
        Lx=LX, Ly=LY,
        H_under=H_bath,
        H_top=H_PETRI,
        frequency_hz=FREQ,
        disk_radius=DISK_RADIUS,
        disk_velocity_amplitude=V_disk,
        vortex_topological_charge=1,
        standing_velocity_amplitude=V_stand,
        standing_phase_pattern="antiphase",
        standing_axis="both",
        pml_n_wavelengths_xy=1.0,
        pml_n_wavelengths_z=1.0,
        pml_degree=2,
        pml_sigma_max_factor=5.0,
        pml_enabled=pml_enabled,
        elements_per_wavelength=eps,
        lens_drive="plastic",
        lens_l=1,
        lens_focal_length=f_lens,
        lens_focus_offset_x=0.2e-3,
        lens_focus_offset_y=0.0,
        lens_c_lens=2700.0,
        lens_apodization="cosine_taper",
        lens_apodization_strength=1.0,
    )


def physical_mask_xz(xg, zg, cfg):
    """Boolean masks for physical region (exclude PML)."""
    t_xy = cfg.t_pml_xy
    t_z = cfg.t_pml_z
    x_phys = (xg >= t_xy) & (xg <= cfg.Lx - t_xy)
    z_phys = zg >= t_z
    return x_phys, z_phys


def physical_mask_xy(xg, yg, cfg):
    """Boolean mask for physical XY region (exclude PML)."""
    t_xy = cfg.t_pml_xy
    X, Y = np.meshgrid(xg, yg)
    return (X >= t_xy) & (X <= cfg.Lx - t_xy) & (Y >= t_xy) & (Y <= cfg.Ly - t_xy)


def find_focus_z(sol, cfg):
    """
    Find z_focus = argmax(|p|) on the centerline inside the bath
    (excluding PML), and also max |p| in bath and petri.
    """
    zc, pc = centerline_z(sol, nz=500)
    t_z = cfg.t_pml_z
    H_bath = cfg.H_under
    H_total = cfg.H_total

    # Bath region: z ∈ [t_pml_z, H_bath]
    bath_mask = (zc >= t_z) & (zc <= H_bath)
    if not np.any(bath_mask):
        return float("nan"), 0.0, 0.0, zc, pc

    pc_bath = pc[bath_mask]
    z_bath = zc[bath_mask]
    i_max = np.argmax(pc_bath)
    z_focus = float(z_bath[i_max])
    max_p_bath = float(pc_bath[i_max])

    # Petri region: z ∈ [H_bath, H_total]
    petri_mask = (zc >= H_bath) & (zc <= H_total)
    max_p_petri = float(np.max(pc[petri_mask])) if np.any(petri_mask) else 0.0

    return z_focus, max_p_bath, max_p_petri, zc, pc


def plot_xz_vortex(sol, cfg, fig_dir, label, z_focus=None):
    """Generate XZ |p| linear and log plots with physical-region scaling."""
    xg, zg, pmag, _ = slice_xz(sol, cfg.Ly / 2, nx=300, nz=300)
    x_phys, z_phys = physical_mask_xz(xg, zg, cfg)
    phys_region = pmag[np.ix_(z_phys, x_phys)]
    vmax = float(phys_region.max()) if phys_region.size > 0 else 1.0

    # Guard against NaN/inf from solver failures
    if not np.isfinite(vmax) or vmax <= 0:
        print(f"  WARNING: vmax={vmax} — skipping XZ plot for {label}")
        return

    H_bath = cfg.H_under
    t_xy = cfg.t_pml_xy * 1e3
    t_z = cfg.t_pml_z * 1e3

    for mode in ["linear", "log"]:
        fig, ax = plt.subplots(figsize=(8, 6))
        if mode == "linear":
            im = ax.pcolormesh(xg*1e3, zg*1e3, pmag, shading="auto",
                               cmap="inferno", vmin=0, vmax=vmax)
        else:
            pmag_c = np.clip(pmag, 1e-6, None)
            im = ax.pcolormesh(xg*1e3, zg*1e3, pmag_c, shading="auto",
                               cmap="inferno",
                               norm=LogNorm(vmin=max(1e-5, vmax*1e-4),
                                            vmax=vmax*2))
        # PML boundaries
        ax.axvline(t_xy, color="w", ls="--", lw=0.7, alpha=0.6, label="PML")
        ax.axvline((cfg.Lx - cfg.t_pml_xy)*1e3, color="w", ls="--", lw=0.7, alpha=0.6)
        ax.axhline(t_z, color="w", ls="--", lw=0.7, alpha=0.6)
        # Petri interface
        ax.axhline(H_bath*1e3, color="cyan", ls="-", lw=1.5, label="petri interface")
        ax.axhline(cfg.H_total*1e3, color="cyan", ls=":", lw=0.8)
        # Focus marker
        if z_focus is not None and not np.isnan(z_focus):
            ax.axhline(z_focus*1e3, color="lime", ls="-.", lw=1.2,
                        label=f"z_focus={z_focus*1e3:.2f} mm")
            ax.plot(cfg.Lx/2*1e3, z_focus*1e3, "x", color="lime", ms=10, mew=2)

        ax.set_title(f"|p| XZ ({mode}) — {label}\n"
                     f"vmax(phys)={vmax:.4f} Pa", fontsize=10)
        ax.set_xlabel("x [mm]"); ax.set_ylabel("z [mm]")
        plt.colorbar(im, ax=ax, label="Pa")
        ax.legend(fontsize=7, loc="upper right")
        fig.tight_layout()
        fig.savefig(fig_dir / f"xz_vortex_{mode}_{label}.png", dpi=200)
        plt.close(fig)


def plot_centerline(solutions_dict, fig_dir, cfg_ref, title="Centerline"):
    """Plot centerline |p|(z) for multiple cases on one axes."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, data in solutions_dict.items():
        ax.plot(data["zc"]*1e3, data["pc"], label=label)
    ax.axvline(cfg_ref.H_under*1e3, color="cyan", ls="-", lw=1.5,
               label="petri interface")
    ax.axvline(cfg_ref.t_pml_z*1e3, color="gray", ls=":", lw=0.7,
               label="PML-z")
    ax.set_xlabel("z [mm]"); ax.set_ylabel("|p| [Pa]")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "centerline_z.png", dpi=200)
    plt.close(fig)


def plot_xy_case(sol, cfg, fig_dir, label, vmax=None):
    """Generate XY |p| at petri mid-plane."""
    z_petri_mid = cfg.H_under + H_PETRI / 2
    xg, yg, pmag, pphi = slice_xy(sol, z_petri_mid, nx=200, ny=200)
    mask = physical_mask_xy(xg, yg, cfg)
    if vmax is None:
        vmax = pmag[mask].max() if mask.any() else 1.0

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.pcolormesh(xg*1e3, yg*1e3, pmag, shading="auto",
                       cmap="inferno", vmin=0, vmax=vmax)
    ax.set_title(f"|p| XY petri mid — {label}")
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax, label="Pa")
    fig.tight_layout()
    fig.savefig(fig_dir / f"xy_{label}.png", dpi=200)
    plt.close(fig)
    return xg, yg, pmag, pphi, mask


def slice_xy_complex(sol, z_val, nx=200, ny=200):
    """Return (xg, yg, p_complex_2d) at fixed z."""
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


# ═════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════

def main():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path("results") / f"corrected_model_{stamp}"
    out_root.mkdir(parents=True, exist_ok=True)
    fig_dir = out_root / "figures"
    csv_dir = out_root / "csv"
    fig_dir.mkdir(exist_ok=True)
    csv_dir.mkdir(exist_ok=True)

    print(f"\n{'#'*72}")
    print(f"  CORRECTED MODEL SWEEP — {stamp}")
    print(f"  H_petri = {H_PETRI*1e3:.0f} mm (fixed)")
    print(f"  Top BC: water–air Robin  Z_air = {1.2*343:.1f} Pa·s/m")
    print(f"  V_disk = {V_DISK*1e6:.1f} µm/s,  V_stand = {V_STAND*1e6:.1f} µm/s")
    print(f"  OMP_NUM_THREADS = {OMP_THREADS} / {NCORES} cores")
    print(f"  Output: {out_root}")
    print(f"{'#'*72}\n")

    # ══════════════════════════════════════════════════════════════════
    #  PHASE 1: Vortex-Only Sweep (H_bath × f_lens)
    # ══════════════════════════════════════════════════════════════════
    print(f"{'='*72}")
    print(f"  PHASE 1: Vortex-Only H_bath × f_lens Sweep")
    print(f"{'='*72}\n")

    sweep_rows = []

    for H_bath in H_BATH_LIST:
        for f_lens in F_LENS_LIST:
            # Constraint: focus must be inside bath
            if f_lens >= H_bath:
                print(f"  SKIP H_bath={H_bath*1e3:.0f}mm, f={f_lens*1e3:.0f}mm "
                      f"(f >= H_bath)")
                sweep_rows.append({
                    "H_bath_mm": H_bath*1e3,
                    "f_lens_mm": f_lens*1e3,
                    "H_total_mm": (H_bath + H_PETRI)*1e3,
                    "z_focus_mm": float("nan"),
                    "max_p_bath": float("nan"),
                    "max_p_petri": float("nan"),
                    "focus_below_petri_mm": float("nan"),
                    "dofs": 0,
                    "solve_time_s": 0,
                    "status": "SKIPPED (f>=H_bath)",
                })
                continue

            label = f"Hb{H_bath*1e3:.0f}_f{f_lens*1e3:.0f}"
            print(f"\n{'─'*60}")
            print(f"  H_bath={H_bath*1e3:.0f}mm  f={f_lens*1e3:.0f}mm  "
                  f"H_total={(H_bath+H_PETRI)*1e3:.0f}mm")
            print(f"{'─'*60}")

            cfg = make_cfg(H_bath, f_lens, V_disk=V_DISK, V_stand=0.0)
            print(cfg.describe())

            t0 = time.time()
            try:
                sol = solve_helmholtz(cfg, verbose=True, petsc_options=PETSC_OPTS)
            except Exception as e:
                print(f"  *** SOLVE FAILED: {e}")
                traceback.print_exc()
                sweep_rows.append({
                    "H_bath_mm": H_bath*1e3,
                    "f_lens_mm": f_lens*1e3,
                    "H_total_mm": (H_bath + H_PETRI)*1e3,
                    "z_focus_mm": float("nan"),
                    "max_p_bath": float("nan"),
                    "max_p_petri": float("nan"),
                    "focus_below_petri_mm": float("nan"),
                    "dofs": 0,
                    "solve_time_s": time.time() - t0,
                    "status": f"FAILED: {e}",
                })
                continue

            wall = time.time() - t0

            # Check solver convergence
            if sol.ksp_converged_reason < 0:
                print(f"  *** SOLVER DIVERGED (reason={sol.ksp_converged_reason})")
                sweep_rows.append({
                    "H_bath_mm": H_bath*1e3,
                    "f_lens_mm": f_lens*1e3,
                    "H_total_mm": (H_bath + H_PETRI)*1e3,
                    "z_focus_mm": float("nan"),
                    "max_p_bath": float("nan"),
                    "max_p_petri": float("nan"),
                    "focus_below_petri_mm": float("nan"),
                    "dofs": sol.dofs,
                    "solve_time_s": wall,
                    "status": f"DIVERGED(reason={sol.ksp_converged_reason})",
                })
                del sol; gc.collect()
                continue
            z_focus, max_p_bath, max_p_petri, zc, pc = find_focus_z(sol, cfg)
            focus_below_petri = (H_bath - z_focus) if not np.isnan(z_focus) else float("nan")

            print(f"  z_focus = {z_focus*1e3:.2f} mm  "
                  f"({focus_below_petri*1e3:.2f} mm below petri)")
            print(f"  max|p| bath = {max_p_bath:.4f} Pa")
            print(f"  max|p| petri = {max_p_petri:.4f} Pa")

            # Plots
            plot_xz_vortex(sol, cfg, fig_dir, label, z_focus=z_focus)

            sweep_rows.append({
                "H_bath_mm": H_bath*1e3,
                "f_lens_mm": f_lens*1e3,
                "H_total_mm": (H_bath + H_PETRI)*1e3,
                "z_focus_mm": z_focus*1e3,
                "max_p_bath": max_p_bath,
                "max_p_petri": max_p_petri,
                "focus_below_petri_mm": focus_below_petri*1e3,
                "dofs": sol.dofs,
                "solve_time_s": wall,
                "status": "OK",
            })

            del sol; gc.collect()

    # Write sweep CSV
    if sweep_rows:
        with open(csv_dir / "summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sweep_rows[0].keys())
            w.writeheader()
            for row in sweep_rows:
                w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v)
                            for k, v in row.items()})

    # Print sweep summary table
    print(f"\n{'='*72}")
    print("  SWEEP SUMMARY")
    print(f"{'='*72}")
    print(f"{'H_bath':>8} {'f_lens':>8} {'z_focus':>10} {'below_petri':>12} "
          f"{'max_p_bath':>12} {'max_p_petri':>12} {'status':>10}")
    print("-" * 80)
    for r in sweep_rows:
        print(f"{r['H_bath_mm']:7.0f}mm {r['f_lens_mm']:7.0f}mm "
              f"{r['z_focus_mm']:9.2f}mm {r['focus_below_petri_mm']:11.2f}mm "
              f"{r['max_p_bath']:11.4f} {r['max_p_petri']:11.4f} "
              f"{r['status']:>10}")

    # ══════════════════════════════════════════════════════════════════
    #  SELECT BEST GEOMETRY
    # ══════════════════════════════════════════════════════════════════
    # Target: focus 0.5–1.0 mm below petri, no cavity blow-up
    ok_rows = [r for r in sweep_rows if r["status"] == "OK"
               and not np.isnan(r["focus_below_petri_mm"])]

    best = None
    best_score = float("inf")
    for r in ok_rows:
        d = r["focus_below_petri_mm"]
        # Score: distance from ideal range [0.5, 1.0] mm below petri
        if 0.5 <= d <= 1.0:
            score = 0  # perfect
        elif d < 0.5:
            score = (0.5 - d)**2
        else:
            score = (d - 1.0)**2
        # Penalise cavity blow-up (petri >> bath suggests resonance)
        if r["max_p_petri"] > 5 * r["max_p_bath"]:
            score += 10
        if score < best_score:
            best_score = score
            best = r

    if best is None:
        print("\n  *** No valid sweep result found for interaction check.")
        _write_index(out_root, sweep_rows, None, {}, stamp)
        _write_config(out_root, sweep_rows)
        print(f"\n  Output: {out_root}")
        return

    print(f"\n  SELECTED: H_bath={best['H_bath_mm']:.0f}mm  "
          f"f={best['f_lens_mm']:.0f}mm  "
          f"focus {best['focus_below_petri_mm']:.2f}mm below petri")

    # ══════════════════════════════════════════════════════════════════
    #  PHASE 2: Interaction Check (best geometry)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*72}")
    print(f"  PHASE 2: Interaction Check")
    print(f"  H_bath={best['H_bath_mm']:.0f}mm  f={best['f_lens_mm']:.0f}mm")
    print(f"{'='*72}\n")

    H_bath_best = best["H_bath_mm"] * 1e-3
    f_best = best["f_lens_mm"] * 1e-3

    interaction_cases = {
        "standing_only": dict(V_disk=0.0, V_stand=V_STAND),
        "vortex_only":   dict(V_disk=V_DISK, V_stand=0.0),
        "combined":      dict(V_disk=V_DISK, V_stand=V_STAND),
    }

    interaction_data = {}
    roi_rows = []

    for case_name, amps in interaction_cases.items():
        print(f"\n  {'─'*40} {case_name}")
        cfg = make_cfg(H_bath_best, f_best,
                       V_disk=amps["V_disk"], V_stand=amps["V_stand"])
        print(cfg.describe())

        t0 = time.time()
        try:
            sol = solve_helmholtz(cfg, verbose=True, petsc_options=PETSC_OPTS)
        except Exception as e:
            print(f"  *** SOLVE FAILED: {e}")
            traceback.print_exc()
            continue
        wall = time.time() - t0

        # Check solver convergence
        if sol.ksp_converged_reason < 0:
            print(f"  *** SOLVER DIVERGED in interaction case '{case_name}' "
                  f"(reason={sol.ksp_converged_reason})")
            del sol; gc.collect()
            continue

        # Extract data
        z_petri_mid = cfg.H_under + H_PETRI / 2
        xg, yg, pmag, pphi = slice_xy(sol, z_petri_mid, nx=200, ny=200)
        xg_c, yg_c, pc = slice_xy_complex(sol, z_petri_mid, nx=200, ny=200)
        mask = physical_mask_xy(xg, yg, cfg)
        zc, pcl = centerline_z(sol)

        interaction_data[case_name] = {
            "sol": sol, "cfg": cfg,
            "xg": xg, "yg": yg, "pmag": pmag, "pphi": pphi,
            "pc": pc, "mask": mask,
            "zc": zc, "pcl": pcl,
            "max_p": sol.max_pressure,
        }

        # ROI metrics
        pmag_roi = pmag[mask]
        roi_rows.append({
            "case": case_name,
            "mean_abs_p_roi": float(np.mean(pmag_roi)),
            "max_abs_p_roi": float(np.max(pmag_roi)),
            "max_abs_p_global": sol.max_pressure,
        })

    # ── Interaction Plots ─────────────────────────────────────────────
    if interaction_data:
        # Shared vmax from physical region (exclude standing wall sources)
        all_phys_max = []
        for cn, d in interaction_data.items():
            v = float(d["pmag"][d["mask"]].max()) if d["mask"].any() else 0.0
            if np.isfinite(v):
                all_phys_max.append(v)
        vmax_shared = max(all_phys_max) if all_phys_max else 1.0
        if not np.isfinite(vmax_shared) or vmax_shared <= 0:
            vmax_shared = 1.0

        # XY plots for each case
        for cn, d in interaction_data.items():
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.pcolormesh(d["xg"]*1e3, d["yg"]*1e3, d["pmag"],
                               shading="auto", cmap="inferno",
                               vmin=0, vmax=vmax_shared)
            ax.set_title(f"|p| XY petri mid — {cn}")
            ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
            ax.set_aspect("equal")
            plt.colorbar(im, ax=ax, label="Pa")
            fig.tight_layout()
            fig.savefig(fig_dir / f"xy_{cn}.png", dpi=200)
            plt.close(fig)

        # Delta map: |p_combined| − |p_standing|
        if "combined" in interaction_data and "standing_only" in interaction_data:
            dc = interaction_data["combined"]
            ds = interaction_data["standing_only"]
            delta = np.abs(dc["pc"]) - np.abs(ds["pc"])
            vabs = max(abs(np.nanmin(delta[dc["mask"]])),
                       abs(np.nanmax(delta[dc["mask"]])))
            if vabs < 1e-30:
                vabs = 1.0

            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.pcolormesh(dc["xg"]*1e3, dc["yg"]*1e3, delta,
                               shading="auto", cmap="RdBu_r",
                               vmin=-vabs, vmax=vabs)
            ax.set_title("Δ|p| = |p_combined| − |p_standing|")
            ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
            ax.set_aspect("equal")
            plt.colorbar(im, ax=ax, label="Pa")
            fig.tight_layout()
            fig.savefig(fig_dir / "xy_delta.png", dpi=200)
            plt.close(fig)

        # Centerline
        cl_dict = {}
        for cn, d in interaction_data.items():
            cl_dict[cn] = {"zc": d["zc"], "pc": d["pcl"]}
        cfg_best = make_cfg(H_bath_best, f_best)
        plot_centerline(cl_dict, fig_dir, cfg_best,
                        title=f"Centerline — H_bath={H_bath_best*1e3:.0f}mm "
                              f"f={f_best*1e3:.0f}mm")

        # XZ plots for interaction cases
        for cn, d in interaction_data.items():
            xg_xz, zg_xz, pmag_xz, _ = slice_xz(
                d["sol"], d["cfg"].Ly / 2, nx=300, nz=300)
            x_phys, z_phys = physical_mask_xz(xg_xz, zg_xz, d["cfg"])
            phys = pmag_xz[np.ix_(z_phys, x_phys)]
            vm = float(phys.max()) if phys.size > 0 else 1.0
            if not np.isfinite(vm) or vm <= 0:
                print(f"  WARNING: interaction XZ vmax={vm} — skipping {cn}")
                continue

            for mode in ["linear", "log"]:
                fig, ax = plt.subplots(figsize=(8, 6))
                if mode == "linear":
                    im = ax.pcolormesh(xg_xz*1e3, zg_xz*1e3, pmag_xz,
                                       shading="auto", cmap="inferno",
                                       vmin=0, vmax=vm)
                else:
                    pmag_c = np.clip(pmag_xz, 1e-6, None)
                    im = ax.pcolormesh(xg_xz*1e3, zg_xz*1e3, pmag_c,
                                       shading="auto", cmap="inferno",
                                       norm=LogNorm(
                                           vmin=max(1e-5, vm*1e-4),
                                           vmax=vm*2))
                ax.axvline(d["cfg"].t_pml_xy*1e3, color="w", ls="--",
                           lw=0.7, alpha=0.6)
                ax.axvline((d["cfg"].Lx - d["cfg"].t_pml_xy)*1e3,
                           color="w", ls="--", lw=0.7, alpha=0.6)
                ax.axhline(d["cfg"].t_pml_z*1e3, color="w", ls="--",
                           lw=0.7, alpha=0.6)
                ax.axhline(d["cfg"].H_under*1e3, color="cyan", ls="-",
                           lw=1.5, label="petri interface")
                ax.set_title(f"|p| XZ ({mode}) — {cn}", fontsize=10)
                ax.set_xlabel("x [mm]"); ax.set_ylabel("z [mm]")
                ax.legend(fontsize=7)
                plt.colorbar(im, ax=ax, label="Pa")
                fig.tight_layout()
                fig.savefig(fig_dir / f"xz_{cn}_{mode}.png", dpi=200)
                plt.close(fig)

    # ── Write ROI CSV ─────────────────────────────────────────────────
    if roi_rows:
        with open(csv_dir / "roi_metrics.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=roi_rows[0].keys())
            w.writeheader()
            for row in roi_rows:
                w.writerow({k: f"{v:.6f}" if isinstance(v, float) else v
                            for k, v in row.items()})

    # ── Write outputs ─────────────────────────────────────────────────
    _write_index(out_root, sweep_rows, best, interaction_data, stamp)
    _write_config(out_root, sweep_rows)

    # Clean up sols
    for cn in list(interaction_data.keys()):
        if "sol" in interaction_data[cn]:
            del interaction_data[cn]["sol"]
    gc.collect()

    print(f"\n{'#'*72}")
    print(f"  CORRECTED MODEL SWEEP COMPLETE")
    print(f"  Output: {out_root}")
    print(f"{'#'*72}\n")


# ═════════════════════════════════════════════════════════════════════
#  OUTPUT WRITERS
# ═════════════════════════════════════════════════════════════════════

def _write_index(out_root, sweep_rows, best, interaction_data, stamp):
    lines = []
    lines.append("# Corrected Physical Model — Vortex + Standing-Wave Interaction\n")
    lines.append(f"**Date:** {datetime.now().isoformat()}\n")
    lines.append("## Physical Model\n")
    lines.append("- **Two-region cuboid**: water bath (below) + petri slab (above)")
    lines.append(f"- **H_petri** = {H_PETRI*1e3:.0f} mm (fixed)")
    lines.append("- **H_bath** = swept  {3, 4, 5, 6, 7} mm")
    lines.append("- **Standing-wave BC**: ONLY on petri slab side walls "
                 "(z ∈ [H_bath, H_total])")
    lines.append("- **Bath side walls**: passive (no excitation)")
    lines.append("- **Top BC**: water–air Robin (FIXED, not tunable)")
    lines.append(f"  - ρ_air = 1.2 kg/m³,  c_air = 343 m/s")
    lines.append(f"  - Z_air = ρ_air · c_air = 411.6 Pa·s/m")
    lines.append(f"  - Z_water = 997 × 1484 = 1,479,548 Pa·s/m")
    lines.append(f"  - Z_rel = Z_air / Z_water = 0.000278")
    lines.append(f"  - Robin: ∂p/∂n + ik(ρ_water c_water)/(ρ_air c_air) p = 0")
    lines.append(f"- **Frequency**: {FREQ/1e6:.1f} MHz")
    lines.append(f"- **Domain**: {LX*1e3:.0f} × {LY*1e3:.0f} mm lateral")
    lines.append(f"- **Resolution**: {ELEM_PER_LAMBDA} elem/λ")
    lines.append(f"- **Solver**: MUMPS direct\n")

    lines.append("## H_bath × f_lens Sweep (Vortex-Only)\n")
    lines.append("| H_bath [mm] | f [mm] | H_total [mm] | z_focus [mm] | "
                 "below_petri [mm] | max|p| bath | max|p| petri | status |")
    lines.append("|------------|--------|-------------|-------------|"
                 "-----------------|-------------|-------------|--------|")
    for r in sweep_rows:
        lines.append(
            f"| {r['H_bath_mm']:.0f} | {r['f_lens_mm']:.0f} | "
            f"{r['H_total_mm']:.0f} | "
            f"{r['z_focus_mm']:.2f} | {r['focus_below_petri_mm']:.2f} | "
            f"{r['max_p_bath']:.4f} | {r['max_p_petri']:.4f} | "
            f"{r['status']} |"
        )
    lines.append("")

    if best:
        lines.append(f"## Selected Geometry\n")
        lines.append(f"- H_bath = {best['H_bath_mm']:.0f} mm")
        lines.append(f"- f_lens = {best['f_lens_mm']:.0f} mm")
        lines.append(f"- z_focus = {best['z_focus_mm']:.2f} mm "
                     f"({best['focus_below_petri_mm']:.2f} mm below petri)")
        lines.append("")

    if interaction_data:
        lines.append("## Interaction Check\n")
        lines.append(f"- V_disk = {V_DISK*1e6:.1f} µm/s")
        lines.append(f"- V_stand = {V_STAND*1e6:.1f} µm/s\n")
        lines.append("See `csv/roi_metrics.csv` and `figures/` for full results.\n")

    lines.append("## Figures\n")
    fig_dir = out_root / "figures"
    if fig_dir.exists():
        for fn in sorted(fig_dir.iterdir()):
            if fn.suffix == ".png":
                lines.append(f"- ![{fn.stem}](figures/{fn.name})")
    lines.append("")

    (out_root / "INDEX.md").write_text("\n".join(lines))
    print(f"\n  Wrote {out_root / 'INDEX.md'}")


def _write_config(out_root, sweep_rows):
    cfg_example = make_cfg(H_BATH_LIST[0], F_LENS_LIST[0])
    config_dict = cfg_example.to_dict()
    config_dict["sweep_H_bath_mm"] = [h*1e3 for h in H_BATH_LIST]
    config_dict["sweep_f_lens_mm"] = [f*1e3 for f in F_LENS_LIST]
    config_dict["V_disk_mps"] = V_DISK
    config_dict["V_stand_mps"] = V_STAND
    config_dict["H_petri_mm"] = H_PETRI * 1e3
    with open(out_root / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2, default=str)


if __name__ == "__main__":
    main()
