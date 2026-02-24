#!/usr/bin/env python3
"""
Vortex-Standing Balance Diagnostics
=====================================

Investigates:
  1. Colorscale fix: per-case colourbars (not global vmax)
  2. Magnitude mismatch: standing max|p|=24.5 Pa vs vortex max|p|=1.8 Pa (14×)
  3. Vortex focus geometry: does f=10mm produce a tight-enough spot?
  4. Amplitude sweep: find ratio where vortex perturbs but doesn't destroy
     the standing wave pattern

Key physics goal:
  Standing wave creates a lattice of traps.
  Vortex adds orbital angular momentum to *pull* a particle out of a trap.
  Vortex should NOT destroy the standing-wave nodal structure — it should
  be a perturbation, not a dominator.

Output → results/vortex_balance_<timestamp>/
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
OUT_DIR = PROJECT_ROOT / "results" / f"vortex_balance_{TIMESTAMP}"
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
# Lightweight container
# ===================================================================
class LightSol:
    def __init__(self, sol):
        self.coords = sol.coords.copy()
        self.p_values = sol.p_values.copy()
        self.cfg = sol.cfg
        self.dofs = sol.dofs
        self.max_pressure = sol.max_pressure
        self.ksp_converged_reason = sol.ksp_converged_reason


def solve_and_extract(overrides, label=""):
    cfg = FarFieldConfig(**overrides)
    t0 = time.time()
    sol = solve_helmholtz(cfg, verbose=True, petsc_options=PETSC_OPTS,
                          export_fields=False)
    dt = time.time() - t0
    lsol = LightSol(sol)
    del sol; gc.collect()
    print(f"  [{label}] max|p| = {lsol.max_pressure:.3f} Pa   "
          f"KSP={lsol.ksp_converged_reason}   time={dt:.1f}s")
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


# ===================================================================
# PHASE 1: Baseline reference — standing only (proper colorscale)
# ===================================================================
def phase1_references():
    """Solve standing-only and vortex-only with proper per-case colorscales."""
    print("\n" + "=" * 72)
    print("PHASE 1: Reference solutions with per-case colourscales")
    print("=" * 72)

    standing = solve_and_extract({
        **COMMON,
        "disk_velocity_amplitude": 0.0,
    }, label="standing_only")

    vortex = solve_and_extract({
        **COMMON,
        "standing_velocity_amplitude": 0.0,
    }, label="vortex_only")

    return standing, vortex


# ===================================================================
# PHASE 2: Analyze vortex spot size and pressure profile
# ===================================================================
def phase2_vortex_analysis(vortex_sol):
    """Characterize the vortex beam: spot size, ring diameter, peak pressure."""
    print("\n" + "=" * 72)
    print("PHASE 2: Vortex beam characterization")
    print("=" * 72)

    cfg = vortex_sol.cfg
    trap_z = cfg.H_under + cfg.H_top / 2

    # XY at trap plane
    xg, yg, pc = complex_slice_xy(vortex_sol, trap_z, n=500)
    pmag = np.abs(pc)

    # Radial profile from center
    cx_i, cy_i = len(xg) // 2, len(yg) // 2
    cx_m, cy_m = cfg.Lx / 2, cfg.Ly / 2

    # Extract radial profile along x (y=center)
    profile_x = pmag[cy_i, :]
    x_mm = xg * 1e3
    r_mm = x_mm - cx_m * 1e3

    # Find the vortex ring radius (first peak away from center)
    # Center region should be a null (topological singularity)
    center_val = profile_x[cx_i]
    half = profile_x[cx_i:]
    if len(half) > 5:
        peak_idx = np.argmax(half)
        ring_radius_mm = (xg[cx_i + peak_idx] - cx_m) * 1e3
        ring_peak_Pa = half[peak_idx]
    else:
        ring_radius_mm = 0
        ring_peak_Pa = 0

    # FWHM of the ring
    half_max = ring_peak_Pa / 2
    above = half > half_max
    if np.any(above):
        first = np.argmax(above)
        last = len(above) - 1 - np.argmax(above[::-1])
        fwhm_mm = (xg[cx_i + last] - xg[cx_i + first]) * 1e3
    else:
        fwhm_mm = 0

    lam = cfg.wavelength * 1e3  # mm

    print(f"  Vortex at trap plane (z = {trap_z*1e3:.2f} mm):")
    print(f"    Center null |p|     : {center_val:.4f} Pa")
    print(f"    Ring radius         : {ring_radius_mm:.3f} mm = {ring_radius_mm/lam:.2f}λ")
    print(f"    Ring peak |p|       : {ring_peak_Pa:.3f} Pa")
    print(f"    Ring FWHM           : {fwhm_mm:.3f} mm = {fwhm_mm/lam:.2f}λ")
    print(f"    Aperture R          : {cfg.disk_radius*1e3:.2f} mm = {cfg.disk_radius/cfg.wavelength:.1f}λ")
    print(f"    Focal length        : {cfg.lens_focal_length*1e3:.1f} mm")
    print(f"    Focus-to-trap dist  : {(trap_z)*1e3:.2f} mm")
    print(f"    f/# = f/(2R)        : {cfg.lens_focal_length/(2*cfg.disk_radius):.1f}")

    # Plot the radial profile
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(r_mm, profile_x, "b-", lw=1.5, label="Radial |p| profile")
    ax.axhline(ring_peak_Pa, color="red", ls=":", lw=0.8, alpha=0.5)
    ax.axvline(ring_radius_mm, color="red", ls="--", lw=0.8, alpha=0.5,
               label=f"Ring R = {ring_radius_mm:.3f} mm")
    ax.axvline(-ring_radius_mm, color="red", ls="--", lw=0.8, alpha=0.5)
    ax.set_xlabel("r from center [mm]")
    ax.set_ylabel("|p| [Pa]")
    ax.set_title(f"Vortex Radial Profile at Trap Plane (z={trap_z*1e3:.2f} mm)\n"
                 f"Ring R = {ring_radius_mm:.3f} mm ({ring_radius_mm/lam:.2f}λ), "
                 f"FWHM = {fwhm_mm:.3f} mm, peak = {ring_peak_Pa:.3f} Pa")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "vortex_radial_profile.png")
    plt.close(fig)

    # XY magnitude with own colorscale
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.pcolormesh(xg * 1e3, yg * 1e3, pmag,
                       shading="auto", cmap="inferno")
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    ax.set_title(f"Vortex Only — XY at trap z={trap_z*1e3:.2f} mm\n"
                 f"max|p| = {pmag.max():.3f} Pa  (own colorscale)")
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax, label="|p| [Pa]")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "vortex_xy_trap_own_scale.png")
    plt.close(fig)

    # Phase at trap — should show ℓ=1 winding
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.pcolormesh(xg * 1e3, yg * 1e3, np.angle(pc),
                       shading="auto", cmap="twilight",
                       vmin=-np.pi, vmax=np.pi)
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    ax.set_title(f"Vortex Only — Phase at trap z={trap_z*1e3:.2f} mm")
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax, label="Phase [rad]")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "vortex_phase_trap.png")
    plt.close(fig)

    return {
        "ring_radius_mm": ring_radius_mm,
        "ring_peak_Pa": ring_peak_Pa,
        "ring_fwhm_mm": fwhm_mm,
        "center_null_Pa": float(center_val),
        "f_number": cfg.lens_focal_length / (2 * cfg.disk_radius),
    }


# ===================================================================
# PHASE 3: Standing wave characterization
# ===================================================================
def phase3_standing_analysis(standing_sol):
    """Characterize standing-wave trap lattice: node spacing, contrast."""
    print("\n" + "=" * 72)
    print("PHASE 3: Standing wave characterization")
    print("=" * 72)

    cfg = standing_sol.cfg
    trap_z = cfg.H_under + cfg.H_top / 2

    xg, yg, pc = complex_slice_xy(standing_sol, trap_z, n=500)
    pmag = np.abs(pc)

    # Line cut along x at y=center
    cy_i = len(yg) // 2
    profile_x = pmag[cy_i, :]

    # Find nodes (local minima) and antinodes (local maxima)
    from scipy.signal import argrelextrema
    maxima_idx = argrelextrema(profile_x, np.greater, order=5)[0]
    minima_idx = argrelextrema(profile_x, np.less, order=5)[0]

    if len(maxima_idx) >= 2:
        spacings = np.diff(xg[maxima_idx]) * 1e3
        mean_spacing = np.mean(spacings)
    else:
        mean_spacing = 0

    antinode_vals = profile_x[maxima_idx] if len(maxima_idx) > 0 else []
    node_vals = profile_x[minima_idx] if len(minima_idx) > 0 else []

    if len(antinode_vals) > 0 and len(node_vals) > 0:
        mean_antinode = np.mean(antinode_vals)
        mean_node = np.mean(node_vals)
        contrast = (mean_antinode - mean_node) / (mean_antinode + mean_node + 1e-30)
    else:
        mean_antinode = mean_node = contrast = 0

    lam = cfg.wavelength * 1e3
    print(f"  Standing wave at trap plane (z = {trap_z*1e3:.2f} mm):")
    print(f"    Antinode spacing    : {mean_spacing:.3f} mm = {mean_spacing/lam:.2f}λ  (expect λ/2 = {lam/2:.3f})")
    print(f"    Mean antinode |p|   : {mean_antinode:.3f} Pa")
    print(f"    Mean node |p|       : {mean_node:.3f} Pa")
    print(f"    Michelson contrast  : {contrast:.3f}")

    # XY with own colorscale
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.pcolormesh(xg * 1e3, yg * 1e3, pmag,
                       shading="auto", cmap="inferno")
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    ax.set_title(f"Standing Only — XY at trap z={trap_z*1e3:.2f} mm\n"
                 f"Contrast = {contrast:.3f}, spacing = {mean_spacing:.3f} mm")
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax, label="|p| [Pa]")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "standing_xy_trap_own_scale.png")
    plt.close(fig)

    # Line cut
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(xg * 1e3, profile_x, "r-", lw=1.5)
    if len(maxima_idx) > 0:
        ax.plot(xg[maxima_idx] * 1e3, profile_x[maxima_idx], "rv",
                ms=8, label="antinodes")
    if len(minima_idx) > 0:
        ax.plot(xg[minima_idx] * 1e3, profile_x[minima_idx], "b^",
                ms=8, label="nodes")
    ax.set_xlabel("x [mm]"); ax.set_ylabel("|p| [Pa]")
    ax.set_title(f"Standing Wave — X-profile at trap plane")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "standing_x_profile.png")
    plt.close(fig)

    return {
        "antinode_spacing_mm": mean_spacing,
        "mean_antinode_Pa": float(mean_antinode),
        "mean_node_Pa": float(mean_node),
        "contrast": float(contrast),
    }


# ===================================================================
# PHASE 4: Amplitude ratio sweep — find the sweet spot
# ===================================================================
def phase4_amplitude_sweep(standing_sol):
    """
    Sweep vortex amplitude from very weak to moderate.
    Measure standing-wave contrast preservation at each level.
    """
    print("\n" + "=" * 72)
    print("PHASE 4: Vortex amplitude sweep  (standing = 10 µm/s fixed)")
    print("=" * 72)

    cfg_ref = standing_sol.cfg
    trap_z = cfg_ref.H_under + cfg_ref.H_top / 2

    # Reference standing-only pattern
    xg_ref, yg_ref, pc_ref = complex_slice_xy(standing_sol, trap_z, n=300)
    pmag_ref = np.abs(pc_ref)
    cy_i = len(yg_ref) // 2

    from scipy.signal import argrelextrema

    # Standing V = 10 µm/s.  Sweep vortex disk_velocity_amplitude.
    # Current ratio: disk=1 µm/s / standing=10 µm/s = 0.1
    # But vortex max|p|=1.77 Pa and standing max|p|=24.5 Pa → pressure ratio ~0.07
    #
    # Sweep: V_disk from 0.01 to 2.0 µm/s (ratio 0.001 to 0.2)
    # Also try smaller disk radius (tighter focus) and shorter focal length

    V_disk_sweep = [0.05e-6, 0.1e-6, 0.2e-6, 0.5e-6, 1.0e-6, 2.0e-6]
    labels = ["0.05", "0.1", "0.2", "0.5", "1.0", "2.0"]

    results = []

    for V_disk, lab in zip(V_disk_sweep, labels):
        print(f"\n  V_disk = {lab} µm/s  (ratio = {V_disk/10e-6:.3f})")

        lsol = solve_and_extract({
            **COMMON,
            "disk_velocity_amplitude": V_disk,
        }, label=f"combined_Vd={lab}")

        xg, yg, pc = complex_slice_xy(lsol, trap_z, n=300)
        pmag = np.abs(pc)
        profile = pmag[cy_i, :]

        # Measure contrast
        maxima_idx = argrelextrema(profile, np.greater, order=5)[0]
        minima_idx = argrelextrema(profile, np.less, order=5)[0]
        if len(maxima_idx) > 0 and len(minima_idx) > 0:
            an = np.mean(profile[maxima_idx])
            nd = np.mean(profile[minima_idx])
            contrast = (an - nd) / (an + nd + 1e-30)
        else:
            an = nd = contrast = 0

        # Also check phase winding to see if vortex is detectable
        pphase = np.angle(pc)
        # Check winding around center at radius ~1mm
        n_pts = len(xg)
        cx_i = n_pts // 2
        r_px = int(1.0e-3 / (lsol.cfg.Lx / n_pts))  # 1mm radius in pixels
        theta_pts = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        ix = np.clip((cx_i + r_px * np.cos(theta_pts)).astype(int), 0, n_pts - 1)
        iy = np.clip((cx_i + r_px * np.sin(theta_pts)).astype(int), 0, n_pts - 1)
        dph = np.diff(np.unwrap(pphase[iy, ix]))
        winding = abs(np.sum(dph)) / (2 * np.pi)

        r = {
            "V_disk_um_s": V_disk * 1e6,
            "ratio": V_disk / 10e-6,
            "max_p_Pa": float(lsol.max_pressure),
            "contrast": contrast,
            "phase_winding": winding,
        }
        results.append(r)
        print(f"    max|p| = {lsol.max_pressure:.3f} Pa   "
              f"contrast = {contrast:.3f}   winding = {winding:.2f}")

        del lsol; gc.collect()

    return results


# ===================================================================
# PHASE 5: Focal length / aperture radius sweep
# ===================================================================
def phase5_focus_sweep():
    """
    Test smaller disk radius and shorter focal length for tighter vortex spot.
    """
    print("\n" + "=" * 72)
    print("PHASE 5: Vortex focus geometry sweep")
    print("=" * 72)

    cfg_base = FarFieldConfig(**COMMON)
    trap_z = cfg_base.H_under + cfg_base.H_top / 2

    from scipy.signal import argrelextrema

    # Test variations:
    # 1. Default: R=1mm, f=10mm (f/# = 5)
    # 2. Tighter focus: R=1mm, f=5mm (f/# = 2.5)
    # 3. Smaller aperture: R=0.5mm, f=5mm (f/# = 5)
    # 4. Tiny aperture: R=0.3mm, f=3mm (f/# = 5)
    # Keep vortex at 0.5 µm/s (good balance from phase 4 likely)

    V_VORT = 0.5e-6   # moderate vortex

    variants = [
        {"label": "R=1.0mm_f=10mm",  "disk_radius": 1.0e-3, "lens_focal_length": 10e-3},
        {"label": "R=1.0mm_f=5mm",   "disk_radius": 1.0e-3, "lens_focal_length": 5e-3},
        {"label": "R=0.5mm_f=5mm",   "disk_radius": 0.5e-3, "lens_focal_length": 5e-3},
        {"label": "R=0.5mm_f=3mm",   "disk_radius": 0.5e-3, "lens_focal_length": 3e-3},
    ]

    results = []

    for v in variants:
        label = v["label"]
        print(f"\n  {label}  (V_disk = {V_VORT*1e6:.1f} µm/s)")

        overrides = {
            **COMMON,
            "disk_velocity_amplitude": V_VORT,
            "disk_radius": v["disk_radius"],
            "lens_focal_length": v["lens_focal_length"],
        }
        lsol = solve_and_extract(overrides, label=label)

        # XY at trap
        xg, yg, pc = complex_slice_xy(lsol, trap_z, n=400)
        pmag = np.abs(pc)
        cy_i = len(yg) // 2

        # Also solve vortex-only with this geometry for ring characterization
        overrides_v = {
            **COMMON,
            "disk_velocity_amplitude": V_VORT,
            "standing_velocity_amplitude": 0.0,
            "disk_radius": v["disk_radius"],
            "lens_focal_length": v["lens_focal_length"],
        }
        lsol_v = solve_and_extract(overrides_v, label=f"{label}_vortex_only")

        xg_v, yg_v, pc_v = complex_slice_xy(lsol_v, trap_z, n=400)
        pmag_v = np.abs(pc_v)
        profile_v = pmag_v[len(yg_v)//2, :]

        # Vortex ring radius
        half_v = profile_v[len(profile_v)//2:]
        peak_idx = np.argmax(half_v)
        ring_r = (xg_v[len(profile_v)//2 + peak_idx] - lsol_v.cfg.Lx/2) * 1e3
        ring_peak = half_v[peak_idx]

        # Combined contrast
        profile = pmag[cy_i, :]
        maxima_idx = argrelextrema(profile, np.greater, order=5)[0]
        minima_idx = argrelextrema(profile, np.less, order=5)[0]
        if len(maxima_idx) > 0 and len(minima_idx) > 0:
            contrast = (np.mean(profile[maxima_idx]) - np.mean(profile[minima_idx])) / \
                       (np.mean(profile[maxima_idx]) + np.mean(profile[minima_idx]) + 1e-30)
        else:
            contrast = 0

        # Plot combined XY (own scale)
        fig, axes = plt.subplots(1, 3, figsize=(21, 6))

        # vortex alone
        ax = axes[0]
        im = ax.pcolormesh(xg_v * 1e3, yg_v * 1e3, pmag_v,
                           shading="auto", cmap="inferno")
        ax.set_title(f"Vortex Only\nmax={pmag_v.max():.3f} Pa, ring R={ring_r:.2f}mm")
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, label="|p| [Pa]")

        # combined
        ax = axes[1]
        im = ax.pcolormesh(xg * 1e3, yg * 1e3, pmag,
                           shading="auto", cmap="inferno")
        ax.set_title(f"Combined\nmax={pmag.max():.3f} Pa, contrast={contrast:.3f}")
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, label="|p| [Pa]")

        # phase for combined
        ax = axes[2]
        im = ax.pcolormesh(xg * 1e3, yg * 1e3, np.angle(pc),
                           shading="auto", cmap="twilight",
                           vmin=-np.pi, vmax=np.pi)
        ax.set_title(f"Combined Phase")
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, label="Phase [rad]")

        fig.suptitle(f"Focus Geometry: {label}   (V_disk={V_VORT*1e6:.1f} µm/s)",
                     fontsize=14)
        fig.tight_layout()
        fname = label.replace("=", "").replace(".", "p").replace("mm", "")
        fig.savefig(FIG_DIR / f"focus_{fname}.png", bbox_inches="tight")
        plt.close(fig)

        r = {
            "label": label,
            "disk_radius_mm": v["disk_radius"] * 1e3,
            "focal_length_mm": v["lens_focal_length"] * 1e3,
            "f_number": v["lens_focal_length"] / (2 * v["disk_radius"]),
            "ring_radius_mm": ring_r,
            "ring_peak_Pa": float(ring_peak),
            "combined_contrast": contrast,
            "combined_max_p": float(lsol.max_pressure),
        }
        results.append(r)
        print(f"    Ring R = {ring_r:.3f} mm,  peak = {ring_peak:.3f} Pa,  "
              f"contrast = {contrast:.3f}")

        del lsol, lsol_v; gc.collect()

    return results


# ===================================================================
# PHASE 6: Summary panel — amplitude sweep visualization
# ===================================================================
def phase6_summary_panel(amp_results, standing_sol, vortex_sol):
    """Generate summary plots from the amplitude sweep."""
    print("\n" + "=" * 72)
    print("PHASE 6: Summary panels")
    print("=" * 72)

    cfg = standing_sol.cfg
    trap_z = cfg.H_under + cfg.H_top / 2

    # ── Standing + Vortex side by side with OWN colourscales ──
    xg_s, yg_s, pc_s = complex_slice_xy(standing_sol, trap_z, n=400)
    xg_v, yg_v, pc_v = complex_slice_xy(vortex_sol, trap_z, n=400)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax = axes[0]
    im = ax.pcolormesh(xg_s * 1e3, yg_s * 1e3, np.abs(pc_s),
                       shading="auto", cmap="inferno")
    ax.set_title(f"Standing Only\nmax|p| = {np.abs(pc_s).max():.2f} Pa", fontsize=13)
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax, label="|p| [Pa]")

    ax = axes[1]
    im = ax.pcolormesh(xg_v * 1e3, yg_v * 1e3, np.abs(pc_v),
                       shading="auto", cmap="inferno")
    ax.set_title(f"Vortex Only\nmax|p| = {np.abs(pc_v).max():.2f} Pa", fontsize=13)
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax, label="|p| [Pa]")

    fig.suptitle(
        f"Per-Case Colourscales — Note 14× Pressure Difference\n"
        f"Standing: V=10 µm/s → {np.abs(pc_s).max():.1f} Pa    "
        f"Vortex: V=1 µm/s → {np.abs(pc_v).max():.1f} Pa",
        fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "own_scale_comparison.png", bbox_inches="tight")
    plt.close(fig)

    # ── Amplitude sweep: contrast & winding vs V_disk ──
    if amp_results:
        ratios = [r["ratio"] for r in amp_results]
        contrasts = [r["contrast"] for r in amp_results]
        windings = [r["phase_winding"] for r in amp_results]

        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()

        ax1.semilogx(ratios, contrasts, "rs-", ms=8, lw=2, label="Standing-wave contrast")
        ax2.semilogx(ratios, windings, "b^--", ms=8, lw=1.5, label="Phase winding (ℓ)")

        ax1.set_xlabel("V_disk / V_standing  (amplitude ratio)", fontsize=12)
        ax1.set_ylabel("Standing-wave contrast (Michelson)", fontsize=12, color="red")
        ax2.set_ylabel("Phase winding at 1mm radius", fontsize=12, color="blue")
        ax1.axhline(0.5, color="red", ls=":", alpha=0.3, label="50% contrast threshold")
        ax2.axhline(0.5, color="blue", ls=":", alpha=0.3, label="Detectable winding")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

        ax1.set_title("Vortex-Standing Balance: Contrast vs Amplitude Ratio\n"
                      "Sweet spot: high contrast + detectable winding")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "amplitude_balance_sweep.png", bbox_inches="tight")
        plt.close(fig)

    # ── XZ comparisons with per-case scale ──
    y_mid = cfg.Ly / 2
    xg_xs, zg_xs, pc_xs = complex_slice_xz(standing_sol, y_mid, n=400)
    xg_xv, zg_xv, pc_xv = complex_slice_xz(vortex_sol, y_mid, n=400)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    for ax, xg, zg, pc, title in [
        (axes[0], xg_xs, zg_xs, pc_xs, "Standing Only"),
        (axes[1], xg_xv, zg_xv, pc_xv, "Vortex Only"),
    ]:
        pmag = np.abs(pc)
        im = ax.pcolormesh(xg * 1e3, zg * 1e3, pmag,
                           shading="auto", cmap="inferno")
        ax.axhline(cfg.H_under * 1e3, color="cyan", ls="--", lw=0.8, alpha=0.7)
        ax.axhline(cfg.H_total * 1e3, color="cyan", ls="--", lw=0.8, alpha=0.7)
        ax.set_title(f"{title}\nmax |p| = {pmag.max():.2f} Pa")
        ax.set_xlabel("x [mm]"); ax.set_ylabel("z [mm]")
        plt.colorbar(im, ax=ax, label="|p| [Pa]")

    fig.suptitle("XZ Mid-Plane — Per-Case Colourscales", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "xz_own_scale_comparison.png", bbox_inches="tight")
    plt.close(fig)


# ===================================================================
# MAIN
# ===================================================================
def main():
    t_start = time.time()

    # Phase 1 & 2 & 3: Core references
    standing, vortex = phase1_references()
    vortex_info = phase2_vortex_analysis(vortex)
    standing_info = phase3_standing_analysis(standing)

    # Phase 4: Amplitude sweep
    amp_results = phase4_amplitude_sweep(standing)

    # Phase 5: Focus geometry sweep
    focus_results = phase5_focus_sweep()

    # Phase 6: Summary panels
    phase6_summary_panel(amp_results, standing, vortex)

    # Save all results
    all_results = {
        "vortex_beam": vortex_info,
        "standing_wave": standing_info,
        "amplitude_sweep": amp_results,
        "focus_sweep": focus_results,
        "current_config": {
            "standing_V_um_s": 10.0,
            "vortex_V_um_s": 1.0,
            "disk_radius_mm": 1.0,
            "focal_length_mm": 10.0,
            "pressure_ratio": vortex.max_pressure / standing.max_pressure,
        },
    }
    with open(OUT_DIR / "balance_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    dt = time.time() - t_start
    print(f"\n{'='*72}")
    print(f"  DONE in {dt:.0f}s")
    print(f"  Output: {OUT_DIR}")
    print(f"{'='*72}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
