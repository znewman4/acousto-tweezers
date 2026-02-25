#!/usr/bin/env python3
"""
Pre-Lens "Green-Light" Affirmation Sprint
==========================================

Five targeted checks before starting the full lens investigation sweep:
  TASK 1 — Trap plane z* confirmation (z-sweep of Gor'kov)
  TASK 2 — 3-D focus & vortex-core location (vortex-only, ℓ ∈ {1,2,3})
  TASK 3 — Robust net topological charge in ROI
  TASK 4 — Metric stability vs resolution (elem/λ 4→6, grid 200→300)
  TASK 5 — Alpha authority calibration (α ∈ {0.02, 0.05, 0.10, 0.20})

Produces:
  results/pre_lens_affirmation_YYYYMMDD_HHMMSS/
    ├── csv/           (all .csv deliverables)
    ├── figures/        (all .png deliverables)
    └── results.json   (machine-readable summary)

Author: Acousto-Tweezers Pre-Lens Affirmation Sprint
"""
from __future__ import annotations
import sys, os, time, json, subprocess, gc, csv, io
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, List, Dict
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

NTHREADS = str(min(os.cpu_count() or 4, 14))
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, NTHREADS)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import binary_dilation

from acoustweezers.experiments.farfield_petri_cuboid.presets import (
    CORRECTED_PRESET, PETSC_MUMPS,
)
from acoustweezers.experiments.farfield_petri_cuboid.config import FarFieldConfig

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150,
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
})

# =====================================================================
# Constants
# =====================================================================
FREQ_HZ         = 2.0e6
WATER_DENSITY   = 997.0
WATER_C         = 1484.0
OMEGA           = 2 * np.pi * FREQ_HZ
WAVELENGTH      = WATER_C / FREQ_HZ        # 0.742 mm
K_WATER         = OMEGA / WATER_C

PARTICLE_RADIUS  = 5e-6
PARTICLE_DENSITY = 1050.0
PARTICLE_C       = 2350.0
_kappa_p = 1 / (PARTICLE_DENSITY * PARTICLE_C**2)
_kappa_f = 1 / (WATER_DENSITY * WATER_C**2)
F1 = 1 - _kappa_p / _kappa_f
F2 = 2 * (PARTICLE_DENSITY - WATER_DENSITY) / (2 * PARTICLE_DENSITY + WATER_DENSITY)

V_STAND = CORRECTED_PRESET["standing_velocity_amplitude"]   # 10 µm/s
V_VORTEX = CORRECTED_PRESET["disk_velocity_amplitude"]      # 1 µm/s

WORKER_SCRIPT = str(PROJECT_ROOT / "scripts" / "experiments" / "_solve_worker.py")

# ── Output directory ──────────────────────────────────────────────
ts = time.strftime("%Y%m%d_%H%M%S")
OUT_DIR = PROJECT_ROOT / "results" / f"pre_lens_affirmation_{ts}"
FIG_DIR = OUT_DIR / "figures"
CSV_DIR = OUT_DIR / "csv"
for d in (OUT_DIR, FIG_DIR, CSV_DIR):
    d.mkdir(parents=True, exist_ok=True)


# =====================================================================
# Subprocess solver (reuses existing _solve_worker)
# =====================================================================

def solve_to_grid(overrides: dict, label: str,
                  trap_z: float, mid_y: float, n_xy: int = 200):
    """Solve in subprocess → return (xg, yg, p_xy, xg_xz, zg_xz, p_xz,
       phys_max, solve_time)."""
    args_file   = str(OUT_DIR / f"_args_{label}.json")
    result_file = str(OUT_DIR / f"_grid_{label}.npz")

    with open(args_file, "w") as f:
        json.dump({
            "overrides": overrides,
            "label": label,
            "trap_z": trap_z,
            "mid_y": mid_y,
            "n_xy": n_xy,
            "result_file": result_file,
        }, f)

    proc = subprocess.run(
        [sys.executable, WORKER_SCRIPT, args_file],
        capture_output=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Solve for '{label}' failed (rc={proc.returncode})")

    data = np.load(result_file)
    result = (data["xg"], data["yg"], data["p_xy"],
              data["xg_xz"], data["zg_xz"], data["p_xz"],
              float(data["phys_max"]), float(data["solve_time"]))
    os.unlink(args_file)
    return result


# =====================================================================
# Gor'kov (gridded, correct ∇p-based velocity)
# =====================================================================

def gorkov_2d(p_grid, dx, dy, omega=OMEGA, rho=WATER_DENSITY, c=WATER_C,
              a=PARTICLE_RADIUS):
    """Correct Gor'kov: ⟨v²⟩ = |∇p̂|² / (2ω²ρ²). Returns U, Fx, Fy."""
    K = rho * c**2
    p2_avg = np.abs(p_grid)**2 / 2
    dp_dx = np.gradient(p_grid, dx, axis=1)
    dp_dy = np.gradient(p_grid, dy, axis=0)
    gpm2 = np.abs(dp_dx)**2 + np.abs(dp_dy)**2
    v2_avg = gpm2 / (2 * omega**2 * rho**2)
    pf = (4 * np.pi / 3) * a**3
    U = pf * (F1 * p2_avg / (2 * K) - F2 * (3 * rho / 4) * v2_avg)
    Fx = -np.gradient(U, dx, axis=1)
    Fy = -np.gradient(U, dy, axis=0)
    return U, Fx, Fy


# =====================================================================
# Winding number
# =====================================================================

def winding_number(p_grid, xg, yg, cx, cy, radius, n_angles=720):
    """Phase winding around a circle. Returns float ~ integer charge."""
    interp_re = RegularGridInterpolator((yg, xg), np.real(p_grid),
                                         method='linear', bounds_error=False,
                                         fill_value=0.0)
    interp_im = RegularGridInterpolator((yg, xg), np.imag(p_grid),
                                         method='linear', bounds_error=False,
                                         fill_value=0.0)
    theta = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    px = cx + radius * np.cos(theta)
    py = cy + radius * np.sin(theta)
    pts = np.column_stack([py, px])
    p_circ = interp_re(pts) + 1j * interp_im(pts)
    phi = np.angle(p_circ)
    dphi = np.diff(phi)
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
    return float(np.sum(dphi) / (2 * np.pi))


# =====================================================================
# Singularity detection on a 2-D complex field
# =====================================================================

def detect_singularities(p_grid, xg, yg, cx, cy, roi_radius):
    """
    Detect phase singularities inside a circular ROI.

    Strategy: scan every 2×2 cell of the phase grid.  For each plaquette,
    compute the lattice circulation Δφ_12 + Δφ_23 + Δφ_34 + Δφ_41 (wrapped).
    If |circulation| > π  →  contains a singularity with charge sign(circ)/2π.
    """
    phase = np.angle(p_grid)
    ny, nx = phase.shape
    sings = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            # Plaquette corners: (j,i), (j,i+1), (j+1,i+1), (j+1,i)
            ph = [phase[j, i], phase[j, i+1], phase[j+1, i+1], phase[j+1, i]]
            circ = 0.0
            for k in range(4):
                dp = ph[(k+1) % 4] - ph[k]
                dp = (dp + np.pi) % (2 * np.pi) - np.pi
                circ += dp
            if abs(circ) > np.pi:
                x_c = 0.5 * (xg[i] + xg[min(i+1, nx-1)])
                y_c = 0.5 * (yg[j] + yg[min(j+1, ny-1)])
                if (x_c - cx)**2 + (y_c - cy)**2 <= roi_radius**2:
                    charge = int(round(circ / (2 * np.pi)))
                    sings.append({"x": float(x_c), "y": float(y_c),
                                  "charge": charge})
    return sings


# =====================================================================
# ROI & metric helpers
# =====================================================================

def make_roi(xg, yg, cx, cy, radius):
    X, Y = np.meshgrid(xg, yg)
    return (X - cx)**2 + (Y - cy)**2 <= radius**2


def compute_metrics(p_stand_xy, p_vortex_xy, p_comb_xy, xg, yg, roi_mask,
                    omega=OMEGA, label=""):
    """Compute the standard ranking metrics (§1-§3 of vortex_function_audit)."""
    dx = xg[1] - xg[0]
    dy = yg[1] - yg[0]
    m = {"label": label}
    out = ~roi_mask
    roi = roi_mask

    # eta_out
    delta = p_comb_xy - p_stand_xy
    ns = np.sqrt(np.sum(np.abs(p_stand_xy[out])**2))
    nd = np.sqrt(np.sum(np.abs(delta[out])**2))
    m["eta_out"] = float(nd / ns) if ns > 0 else float('nan')

    # corr_out
    s_out = np.abs(p_stand_xy[out])
    c_out = np.abs(p_comb_xy[out])
    if np.std(s_out) > 0 and np.std(c_out) > 0:
        m["corr_out"] = float(np.corrcoef(s_out, c_out)[0, 1])
    else:
        m["corr_out"] = float('nan')

    # Gor'kov
    U_s, Fx_s, Fy_s = gorkov_2d(p_stand_xy, dx, dy, omega)
    U_c, Fx_c, Fy_c = gorkov_2d(p_comb_xy, dx, dy, omega)

    F_mag_s = np.sqrt(Fx_s**2 + Fy_s**2)
    F_mag_c = np.sqrt(Fx_c**2 + Fy_c**2)

    if roi.any():
        m["trap_depth_stand_J"] = float(np.max(U_s[roi]) - np.min(U_s[roi]))
        m["trap_depth_comb_J"]  = float(np.max(U_c[roi]) - np.min(U_c[roi]))
        m["mean_F_stand_N"]     = float(np.mean(F_mag_s[roi]))
        m["mean_F_comb_N"]      = float(np.mean(F_mag_c[roi]))

        # barrier
        Umin_s = float(np.min(U_s[roi]))
        Umin_c = float(np.min(U_c[roi]))
        bnd = binary_dilation(roi, iterations=2) & (~roi)
        if bnd.any():
            Us_s = float(np.max(U_s[bnd]))
            Us_c = float(np.max(U_c[bnd]))
        else:
            Us_s = float(np.max(U_s[roi]))
            Us_c = float(np.max(U_c[roi]))
        m["barrier_stand_J"] = Us_s - Umin_s
        m["barrier_comb_J"]  = Us_c - Umin_c
        m["barrier_reduction_pct"] = float(
            100 * (1 - m["barrier_comb_J"] / m["barrier_stand_J"])
        ) if m["barrier_stand_J"] > 0 else float('nan')
    else:
        for k in ("trap_depth_stand_J", "trap_depth_comb_J", "mean_F_stand_N",
                   "mean_F_comb_N", "barrier_stand_J", "barrier_comb_J",
                   "barrier_reduction_pct"):
            m[k] = float('nan')

    # ROI energy fraction
    ve = np.abs(p_vortex_xy)**2
    tot = np.sum(ve)
    m["vortex_energy_in_roi_pct"] = float(100 * np.sum(ve[roi]) / tot) if tot > 0 else 0.0

    # Directional bias: signed mean of F_x along ê_x inside ROI
    if roi.any():
        Fdelta_x = Fx_c - Fx_s
        Fdelta_y = Fy_c - Fy_s
        m["bias_x_N"] = float(np.mean(Fdelta_x[roi]))
        m["bias_y_N"] = float(np.mean(Fdelta_y[roi]))
        m["bias_mag_N"] = float(np.sqrt(m["bias_x_N"]**2 + m["bias_y_N"]**2))
        # Sign consistency: fraction of ROI pixels where F_delta·ê > 0
        # (ê = direction of max bias)
        if m["bias_mag_N"] > 0:
            ex = m["bias_x_N"] / m["bias_mag_N"]
            ey = m["bias_y_N"] / m["bias_mag_N"]
            dot = Fdelta_x[roi] * ex + Fdelta_y[roi] * ey
            m["sign_consistency"] = float(np.mean(dot > 0))
        else:
            m["sign_consistency"] = float('nan')
    else:
        m["bias_x_N"] = m["bias_y_N"] = m["bias_mag_N"] = float('nan')
        m["sign_consistency"] = float('nan')

    return m


# =====================================================================
# Superposition helper
# =====================================================================

def combine(p_stand, p_vortex, alpha):
    """p_combined = p_stand + α·(V_stand/V_vortex)·p_vortex"""
    scale = alpha * V_STAND / V_VORTEX
    return p_stand + scale * p_vortex


# =====================================================================
# MAIN
# =====================================================================

def main():
    t0_global = time.time()
    results = {}
    solve_log = []

    # Common config base
    COMMON_4 = {**CORRECTED_PRESET, "elements_per_wavelength": 4}
    H_under = COMMON_4["H_under"]
    H_top   = COMMON_4["H_top"]
    Lx      = COMMON_4["Lx"]
    Ly      = COMMON_4["Ly"]
    lam     = WAVELENGTH
    cx, cy  = Lx / 2, Ly / 2
    # Intended target (with focus offset)
    x_star = cx + COMMON_4.get("lens_focus_offset_x", 0.0)
    y_star = cy + COMMON_4.get("lens_focus_offset_y", 0.0)

    mid_y = Ly / 2
    z_mid = H_under + H_top / 2   # candidate z*

    # ==================================================================
    # TASK 1 — Trap plane z* confirmation
    # ==================================================================
    print("=" * 72)
    print("TASK 1: Trap plane z* confirmation  (standing-only, z-sweep)")
    print("=" * 72)

    z_offsets_lam = [0.0, 0.25, 0.5, -0.25, -0.5]
    z_values = [z_mid + off * lam for off in z_offsets_lam]
    z_labels = [f"z_mid{off:+.2f}lam" for off in z_offsets_lam]

    stand_cfg = {**COMMON_4, "disk_velocity_amplitude": 0.0}
    n_grid = 200

    task1_rows = []
    task1_grids = {}

    for z_val, z_lab, z_off in zip(z_values, z_labels, z_offsets_lam):
        print(f"\n  Solving standing-only at z = {z_val*1e3:.4f} mm  ({z_lab})")
        xg, yg, p_xy, xg_xz, zg_xz, p_xz, phys_max, st = \
            solve_to_grid(stand_cfg, f"stand_{z_lab}", z_val, mid_y, n_grid)
        solve_log.append((f"stand_{z_lab}", st))

        dx = xg[1] - xg[0]; dy = yg[1] - yg[0]
        U, Fx, Fy = gorkov_2d(p_xy, dx, dy)
        roi = make_roi(xg, yg, cx, cy, 2 * lam)
        X, Y = np.meshgrid(xg, yg)

        # Trap depth & barrier
        if roi.any():
            td = float(np.max(U[roi]) - np.min(U[roi]))
            Umin = float(np.min(U[roi]))
            bnd = binary_dilation(roi, iterations=2) & (~roi)
            Usaddle = float(np.max(U[bnd])) if bnd.any() else float(np.max(U[roi]))
            barrier = Usaddle - Umin
            # Location of U minimum
            idx_min = np.unravel_index(np.argmin(np.where(roi, U, np.inf)), U.shape)
            xmin = float(xg[idx_min[1]])
            ymin = float(yg[idx_min[0]])
        else:
            td = barrier = xmin = ymin = float('nan')

        row = {
            "z_mm": round(z_val * 1e3, 4),
            "z_offset_lam": z_off,
            "trap_depth_J": td,
            "barrier_J": barrier,
            "x_Umin_mm": round(xmin * 1e3, 4),
            "y_Umin_mm": round(ymin * 1e3, 4),
            "phys_max_Pa": phys_max,
        }
        task1_rows.append(row)
        task1_grids[z_off] = (xg, yg, p_xy, U)
        print(f"    trap_depth={td:.4e} J   barrier={barrier:.4e} J   "
              f"max|p|={phys_max:.2f} Pa   Umin at ({xmin*1e3:.3f},{ymin*1e3:.3f}) mm")

    # Write CSV
    with open(CSV_DIR / "z_sensitivity.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(task1_rows[0].keys()))
        w.writeheader(); w.writerows(task1_rows)

    # Best z*: maximises barrier (most confining trap)
    best_row = max(task1_rows, key=lambda r: r["barrier_J"])
    z_star = best_row["z_mm"] / 1e3
    results["task1"] = {
        "z_star_mm": best_row["z_mm"],
        "best_barrier_J": best_row["barrier_J"],
        "best_trap_depth_J": best_row["trap_depth_J"],
        "rows": task1_rows,
    }
    print(f"\n  >>> Chosen z* = {best_row['z_mm']:.4f} mm  "
          f"(barrier = {best_row['barrier_J']:.4e} J)")

    # PNG: z-sweep panels (U maps for z_mid and the best offset)
    offsets_to_plot = sorted(task1_grids.keys())
    n_panels = len(offsets_to_plot)
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 3.5))
    if n_panels == 1:
        axes = [axes]
    for ax, off in zip(axes, offsets_to_plot):
        xg_p, yg_p, _, U_p = task1_grids[off]
        z_mm = (z_mid + off * lam) * 1e3
        im = ax.pcolormesh(xg_p * 1e3, yg_p * 1e3, U_p, shading='auto', cmap='RdBu_r')
        ax.set_title(f"U  z={z_mm:.3f} mm\n(Δz={off:+.2f}λ)", fontsize=8)
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]"); ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, label="J", shrink=0.8)
    fig.suptitle("TASK 1: Gor'kov potential vs z-plane", fontsize=11)
    fig.tight_layout()
    fig.savefig(str(FIG_DIR / "trap_plane_z_sweep_panels.png"))
    plt.close(fig)

    # ==================================================================
    # TASK 2 — 3-D focus + core location (vortex-only)
    # ==================================================================
    print(f"\n{'='*72}")
    print("TASK 2: 3-D focus + vortex-core location  (ℓ ∈ {1,2,3})")
    print("=" * 72)

    focus_3d_rows = []
    # Cache vortex grids for Task 3
    vortex_grids = {}

    for ell in [1, 2, 3]:
        print(f"\n  --- ℓ = {ell} ---")
        vortex_cfg = {
            **COMMON_4,
            "standing_velocity_amplitude": 0.0,
            "disk_velocity_amplitude": 10e-6,
            "lens_l": ell,
            "vortex_topological_charge": ell,
        }
        lab = f"vortex_l{ell}"

        # Solve at z* for XY; XZ through domain centre
        xg, yg, p_xy, xg_xz, zg_xz, p_xz, phys_max, st = \
            solve_to_grid(vortex_cfg, lab, z_star, mid_y, n_grid)
        solve_log.append((lab, st))
        vortex_grids[ell] = (xg, yg, p_xy)

        # ── XY plane analysis ──
        p_mag = np.abs(p_xy)
        X, Y = np.meshgrid(xg, yg)

        # Peak |p| (XY at z*)
        idx_peak = np.unravel_index(np.argmax(p_mag), p_mag.shape)
        x_peak = float(xg[idx_peak[1]])
        y_peak = float(yg[idx_peak[0]])
        val_peak = float(p_mag[idx_peak])

        # Vortex core (minimum |p| near centre, search radius 1.5λ)
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
        search = dist < 1.5 * lam
        if search.any():
            masked = np.where(search, p_mag, np.inf)
            idx_core = np.unravel_index(np.argmin(masked), p_mag.shape)
            x_core = float(xg[idx_core[1]])
            y_core = float(yg[idx_core[0]])
            val_core = float(p_mag[idx_core])
        else:
            x_core = y_core = val_core = float('nan')

        # ── XZ plane analysis ──  peak |p| along z
        p_xz_mag = np.abs(p_xz)
        idx_xz_peak = np.unravel_index(np.argmax(p_xz_mag), p_xz_mag.shape)
        x_xz_peak = float(xg_xz[idx_xz_peak[1]])
        z_xz_peak = float(zg_xz[idx_xz_peak[0]])
        val_xz_peak = float(p_xz_mag[idx_xz_peak])

        # Centroid
        intensity = np.abs(p_xy)**2
        tot = np.sum(intensity)
        cx_cent = float(np.sum(X * intensity) / tot) if tot > 0 else cx
        cy_cent = float(np.sum(Y * intensity) / tot) if tot > 0 else cy

        row = {
            "ell": ell,
            "x_peak_mm": round(x_peak * 1e3, 4),
            "y_peak_mm": round(y_peak * 1e3, 4),
            "z_peak_mm": round(z_star * 1e3, 4),
            "peak_Pa": round(val_peak, 3),
            "x_core_mm": round(x_core * 1e3, 4),
            "y_core_mm": round(y_core * 1e3, 4),
            "core_min_Pa": round(val_core, 4),
            "centroid_x_mm": round(cx_cent * 1e3, 4),
            "centroid_y_mm": round(cy_cent * 1e3, 4),
            "xz_x_peak_mm": round(x_xz_peak * 1e3, 4),
            "xz_z_peak_mm": round(z_xz_peak * 1e3, 4),
            "xz_peak_Pa": round(val_xz_peak, 3),
            "offset_core_x_mm": round((x_core - x_star) * 1e3, 4),
            "offset_core_y_mm": round((y_core - y_star) * 1e3, 4),
            "phys_max_Pa": round(phys_max, 3),
        }
        focus_3d_rows.append(row)
        print(f"    Peak XY: ({x_peak*1e3:.3f}, {y_peak*1e3:.3f}) mm  |p|={val_peak:.3f} Pa")
        print(f"    Core:    ({x_core*1e3:.3f}, {y_core*1e3:.3f}) mm  min|p|={val_core:.4f} Pa")
        print(f"    XZ peak: x={x_xz_peak*1e3:.3f} mm  z={z_xz_peak*1e3:.3f} mm  |p|={val_xz_peak:.3f} Pa")
        print(f"    Centroid: ({cx_cent*1e3:.3f}, {cy_cent*1e3:.3f}) mm")

        # ── PNGs ──
        # (a) |p| XY
        fig, ax = plt.subplots(figsize=(5, 4.5))
        im = ax.pcolormesh(xg * 1e3, yg * 1e3, p_mag, shading='auto', cmap='inferno')
        ax.plot(x_core * 1e3, y_core * 1e3, 'cx', ms=8, mew=2, label='core')
        ax.plot(cx_cent * 1e3, cy_cent * 1e3, 'g+', ms=10, mew=2, label='centroid')
        ax.set_title(f"|p| XY  ℓ={ell}  z*={z_star*1e3:.3f} mm")
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]"); ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, label="Pa"); ax.legend(fontsize=7, loc='upper right')
        fig.tight_layout()
        fig.savefig(str(FIG_DIR / f"vortex_l{ell}_xy_pmag_zstar.png"))
        plt.close(fig)

        # (b) Phase XY
        fig, ax = plt.subplots(figsize=(5, 4.5))
        im = ax.pcolormesh(xg * 1e3, yg * 1e3, np.angle(p_xy), shading='auto',
                           cmap='hsv', vmin=-np.pi, vmax=np.pi)
        ax.plot(x_core * 1e3, y_core * 1e3, 'kx', ms=8, mew=2, label='core')
        ax.set_title(f"arg(p) XY  ℓ={ell}  z*={z_star*1e3:.3f} mm")
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]"); ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, label="rad"); ax.legend(fontsize=7, loc='upper right')
        fig.tight_layout()
        fig.savefig(str(FIG_DIR / f"vortex_l{ell}_xy_phase_zstar.png"))
        plt.close(fig)

        # (c) |p| XZ through core y
        fig, ax = plt.subplots(figsize=(6, 3.5))
        im = ax.pcolormesh(xg_xz * 1e3, zg_xz * 1e3, p_xz_mag, shading='auto',
                           cmap='inferno')
        ax.axhline(z_star * 1e3, color='w', ls='--', lw=0.8, label=f"z*={z_star*1e3:.3f}")
        ax.set_title(f"|p| XZ  ℓ={ell}  (y = mid)")
        ax.set_xlabel("x [mm]"); ax.set_ylabel("z [mm]")
        plt.colorbar(im, ax=ax, label="Pa"); ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(str(FIG_DIR / f"vortex_l{ell}_xz_pmag_through_core.png"))
        plt.close(fig)

    with open(CSV_DIR / "focus_3d_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(focus_3d_rows[0].keys()))
        w.writeheader(); w.writerows(focus_3d_rows)
    results["task2"] = focus_3d_rows

    # ==================================================================
    # TASK 3 — Robust net topological charge in ROI
    # ==================================================================
    print(f"\n{'='*72}")
    print("TASK 3: Net topological charge in ROI  (singularity detection)")
    print("=" * 72)

    sing_all_rows = []
    charge_summary = []

    for ell in [1, 2, 3]:
        xg, yg, p_xy = vortex_grids[ell]
        print(f"\n  --- ℓ = {ell} ---")

        roi_radii_lam = [1.0, 1.5, 2.0]
        for roi_r_lam in roi_radii_lam:
            roi_r = roi_r_lam * lam
            sings = detect_singularities(p_xy, xg, yg, cx, cy, roi_r)
            total_q = sum(s["charge"] for s in sings)
            n_pos = sum(1 for s in sings if s["charge"] > 0)
            n_neg = sum(1 for s in sings if s["charge"] < 0)
            print(f"    ROI={roi_r_lam:.1f}λ:  {len(sings)} singularities  "
                  f"(+{n_pos}, −{n_neg})  Σq = {total_q}")
            for s in sings:
                sing_all_rows.append({
                    "ell": ell,
                    "roi_radius_lam": roi_r_lam,
                    "x_mm": round(s["x"] * 1e3, 4),
                    "y_mm": round(s["y"] * 1e3, 4),
                    "charge": s["charge"],
                })
            charge_summary.append({
                "ell": ell,
                "roi_radius_lam": roi_r_lam,
                "n_singularities": len(sings),
                "n_positive": n_pos,
                "n_negative": n_neg,
                "total_charge": total_q,
                "magnitude_match": abs(total_q) == ell,
            })

        # Also cross-check with winding number around centroid
        cx_c, cy_c = cx, cy  # use domain centre
        for r_lam in [0.5, 1.0, 1.5, 2.0]:
            w = winding_number(p_xy, xg, yg, cx_c, cy_c, r_lam * lam)
            print(f"    Winding(r={r_lam:.1f}λ, centre) = {w:.3f}")

    with open(CSV_DIR / "singularities_roi.csv", "w", newline="") as f:
        if sing_all_rows:
            w = csv.DictWriter(f, fieldnames=list(sing_all_rows[0].keys()))
            w.writeheader(); w.writerows(sing_all_rows)
        else:
            f.write("ell,roi_radius_lam,x_mm,y_mm,charge\n")
    results["task3"] = {"charge_summary": charge_summary}

    # Determine robust-ℓ status
    robust = True
    for cs in charge_summary:
        if cs["roi_radius_lam"] == 1.5:
            if abs(cs["total_charge"]) != cs["ell"]:
                robust = False
    results["task3"]["robust_net_charge"] = robust
    print(f"\n  >>> Net charge robustly = ℓ at 1.5λ ROI: {robust}")

    # PNG: charge map overlay (phase + singularity markers)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for idx, ell in enumerate([1, 2, 3]):
        ax = axes[idx]
        xg, yg, p_xy = vortex_grids[ell]
        phase = np.angle(p_xy)
        im = ax.pcolormesh(xg * 1e3, yg * 1e3, phase, shading='auto',
                           cmap='hsv', vmin=-np.pi, vmax=np.pi)
        # Plot singularities for 1.5λ ROI
        for row in sing_all_rows:
            if row["ell"] == ell and row["roi_radius_lam"] == 1.5:
                marker = '^' if row["charge"] > 0 else 'v'
                color = 'lime' if row["charge"] > 0 else 'red'
                ax.plot(row["x_mm"], row["y_mm"], marker, color=color,
                        ms=7, mew=1.5, mec='k')
        # ROI circle
        theta_c = np.linspace(0, 2 * np.pi, 100)
        ax.plot(cx * 1e3 + 1.5 * lam * 1e3 * np.cos(theta_c),
                cy * 1e3 + 1.5 * lam * 1e3 * np.sin(theta_c),
                'w--', lw=1)
        ax.set_title(f"ℓ={ell}  Σq (1.5λ) = "
                     f"{sum(r['charge'] for r in sing_all_rows if r['ell']==ell and r['roi_radius_lam']==1.5)}")
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]"); ax.set_aspect('equal')
    fig.suptitle("TASK 3: Phase + singularity charge map (▲ +1, ▼ −1)", fontsize=11)
    fig.tight_layout()
    fig.savefig(str(FIG_DIR / "charge_map_overlay.png"))
    plt.close(fig)

    # ==================================================================
    # TASK 4 — Metric stability vs resolution
    # ==================================================================
    print(f"\n{'='*72}")
    print("TASK 4: Metric stability vs resolution")
    print("=" * 72)

    alpha_ref = 0.10
    ell_ref = 1
    comb_cfg_base = {**COMMON_4}  # combined: both sources active

    resolution_rows = []

    for n_grid_res in [200, 300, 400]:
        elem_per_lam = 4  # fixed; higher OOMs on this machine
        label = f"e{elem_per_lam}_g{n_grid_res}"
        print(f"\n  --- elem/λ={elem_per_lam},  grid={n_grid_res} ---")

        cfg_res = {**CORRECTED_PRESET, "elements_per_wavelength": elem_per_lam}

        # Solve standing-only
        stand_cfg_r = {**cfg_res, "disk_velocity_amplitude": 0.0}
        xg_s, yg_s, p_s, _, _, _, pm_s, st_s = \
            solve_to_grid(stand_cfg_r, f"stand_{label}", z_star, mid_y, n_grid_res)
        solve_log.append((f"stand_{label}", st_s))

        # Solve vortex-only (ℓ=1 at base amplitude)
        vortex_cfg_r = {**cfg_res, "standing_velocity_amplitude": 0.0}
        xg_v, yg_v, p_v, _, _, _, pm_v, st_v = \
            solve_to_grid(vortex_cfg_r, f"vortex_{label}", z_star, mid_y, n_grid_res)
        solve_log.append((f"vortex_{label}", st_v))

        # Combine at α=0.10
        p_c = combine(p_s, p_v, alpha_ref)

        roi_mask = make_roi(xg_s, yg_s, cx, cy, 1.5 * lam)
        m = compute_metrics(p_s, p_v, p_c, xg_s, yg_s, roi_mask, label=label)
        m["elem_per_lam"] = elem_per_lam
        m["n_grid"] = n_grid_res
        resolution_rows.append(m)

        print(f"    eta_out={m['eta_out']:.4f}  corr={m['corr_out']:.4f}  "
              f"barrier_red={m['barrier_reduction_pct']:.1f}%  "
              f"bias={m['bias_mag_N']:.3e}  sign_con={m['sign_consistency']:.3f}")

    # Write CSV
    res_keys = ["elem_per_lam", "n_grid", "eta_out", "corr_out",
                "barrier_reduction_pct", "vortex_energy_in_roi_pct",
                "bias_mag_N", "sign_consistency", "mean_F_comb_N",
                "trap_depth_comb_J", "barrier_stand_J", "barrier_comb_J"]
    with open(CSV_DIR / "resolution_stability.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=res_keys, extrasaction='ignore')
        w.writeheader()
        for row in resolution_rows:
            w.writerow(row)

    # Compute % change between baseline (e4,g200) and finest (e6,g300)
    def find_row(epl, ng):
        for r in resolution_rows:
            if r["elem_per_lam"] == epl and r["n_grid"] == ng:
                return r
        return None

    r_base = find_row(4, 200)
    r_fine = find_row(4, 400)
    pct_changes = {}
    if r_base and r_fine:
        for k in ["eta_out", "corr_out", "barrier_reduction_pct",
                   "vortex_energy_in_roi_pct", "bias_mag_N", "sign_consistency"]:
            v_b = r_base.get(k, 0)
            v_f = r_fine.get(k, 0)
            if abs(v_b) > 1e-40:
                pct_changes[k] = round(100 * (v_f - v_b) / abs(v_b), 2)
            else:
                pct_changes[k] = float('nan')
        print(f"\n  % change (4,200) \u2192 (4,400):")
        for k, v in pct_changes.items():
            print(f"    {k}: {v:+.2f}%")
    results["task4"] = {
        "rows": [{k: r[k] for k in res_keys if k in r} for r in resolution_rows],
        "pct_change_4_200_vs_4_400": pct_changes,
    }

    # Stability verdict
    max_pct = max(abs(v) for v in pct_changes.values() if not np.isnan(v)) if pct_changes else 999
    stable = max_pct < 25  # allow up to 25% change
    results["task4"]["stable"] = stable
    results["task4"]["max_pct_change"] = max_pct
    print(f"\n  >>> Resolution stable enough for lens ranking: {stable}  "
          f"(max Δ = {max_pct:.1f}%)")

    # PNG: resolution comparison panels
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    metrics_to_show = ["eta_out", "barrier_reduction_pct", "bias_mag_N", "sign_consistency"]
    for ax, mk in zip(axes.flat, metrics_to_show):
        vals = []
        labels_r = []
        for r in resolution_rows:
            vals.append(r.get(mk, 0))
            labels_r.append(f"e{r['elem_per_lam']}\ng{r['n_grid']}")
        ax.bar(range(len(vals)), vals, color=['#4c72b0', '#55a868', '#c44e52', '#8172b2'])
        ax.set_xticks(range(len(vals))); ax.set_xticklabels(labels_r, fontsize=8)
        ax.set_title(mk, fontsize=9)
        ax.grid(axis='y', alpha=0.3)
    fig.suptitle("TASK 4: Metric stability vs resolution", fontsize=11)
    fig.tight_layout()
    fig.savefig(str(FIG_DIR / "resolution_comparison_panels.png"))
    plt.close(fig)

    # ==================================================================
    # TASK 5 — Alpha authority calibration
    # ==================================================================
    print(f"\n{'='*72}")
    print("TASK 5: Alpha authority calibration")
    print("=" * 72)

    # Use the 4 elem/λ, 200 grid solves from Task 4 baseline
    alpha_values = [0.02, 0.05, 0.10, 0.20]

    # Re-use standing & vortex grids from Task 4 baseline (e4,g200)
    stand_lab = "stand_e4_g200"
    vortex_lab = "vortex_e4_g200"
    stand_npz = OUT_DIR / f"_grid_{stand_lab}.npz"
    vortex_npz = OUT_DIR / f"_grid_{vortex_lab}.npz"

    data_s = np.load(str(stand_npz))
    data_v = np.load(str(vortex_npz))
    xg_a = data_s["xg"]; yg_a = data_s["yg"]
    p_s_a = data_s["p_xy"]; p_v_a = data_v["p_xy"]

    alpha_rows = []
    roi_mask_a = make_roi(xg_a, yg_a, cx, cy, 1.5 * lam)

    for alpha in alpha_values:
        p_c_a = combine(p_s_a, p_v_a, alpha)
        m = compute_metrics(p_s_a, p_v_a, p_c_a, xg_a, yg_a, roi_mask_a,
                            label=f"alpha={alpha:.2f}")
        m["alpha"] = alpha
        alpha_rows.append(m)
        print(f"  α={alpha:.2f}  eta_out={m['eta_out']:.4f}  "
              f"barrier_red={m['barrier_reduction_pct']:.1f}%  "
              f"bias={m['bias_mag_N']:.3e}  sign_con={m['sign_consistency']:.3f}")

    alpha_keys = ["alpha", "eta_out", "corr_out", "barrier_reduction_pct",
                  "vortex_energy_in_roi_pct", "bias_mag_N", "sign_consistency"]
    with open(CSV_DIR / "alpha_calibration.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=alpha_keys, extrasaction='ignore')
        w.writeheader()
        for row in alpha_rows:
            w.writerow(row)
    results["task5"] = [{k: r[k] for k in alpha_keys if k in r} for r in alpha_rows]

    # PNG: alpha calibration curves
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    alphas = [r["alpha"] for r in alpha_rows]
    axes[0].plot(alphas, [r["barrier_reduction_pct"] for r in alpha_rows], 'o-')
    axes[0].set_xlabel("α"); axes[0].set_ylabel("barrier_reduction_pct [%]")
    axes[0].set_title("Barrier reduction vs α"); axes[0].grid(alpha=0.3)

    axes[1].plot(alphas, [r["bias_mag_N"] for r in alpha_rows], 's-', color='C1')
    axes[1].set_xlabel("α"); axes[1].set_ylabel("|bias| [N]")
    axes[1].set_title("Directional bias vs α"); axes[1].grid(alpha=0.3)
    axes[1].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

    axes[2].plot(alphas, [r["eta_out"] for r in alpha_rows], 'D-', color='C2')
    axes[2].set_xlabel("α"); axes[2].set_ylabel("η_out")
    axes[2].set_title("η_out vs α"); axes[2].grid(alpha=0.3)

    fig.suptitle("TASK 5: Alpha authority calibration (ℓ=1)", fontsize=11)
    fig.tight_layout()
    fig.savefig(str(FIG_DIR / "alpha_calibration_curves.png"))
    plt.close(fig)

    # ==================================================================
    # Save master results
    # ==================================================================
    results["solve_log"] = solve_log
    results["total_time_s"] = time.time() - t0_global
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    total = time.time() - t0_global
    print(f"\n{'='*72}")
    print(f"Pre-Lens Affirmation complete.  {total:.0f}s total.")
    print(f"Results:  {OUT_DIR}")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
