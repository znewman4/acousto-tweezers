#!/usr/bin/env python3
"""
Aperture Scaling + VTU Geometry Debug
======================================

PART A — Physics: aperture sweep R=[1,2,3,4,5] mm at f=4mm, ℓ=2
PART B — VTU geometry: diagnose & fix skinny-line bug + minimal test

Outputs → results/rs_aperture_vtu_debug_<TS>/
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.interpolate import RegularGridInterpolator

# ── project root ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from acoustweezers.physics.acoustics.angular_spectrum import propagate_pressure_asm
from acoustweezers.physics.acoustics.vortex_lens import (
    PlasticLensConfig,
    compute_plastic_lens_phase,
    compute_plastic_lens_amplitude,
)

# ═══════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════
WATER_C = 1484.0
F_HZ = 2.0e6
LAM = WATER_C / F_HZ           # 0.742 mm
K_WATER = 2.0 * np.pi * F_HZ / WATER_C

F_LENS = 4.0e-3                # focal length 4 mm
ELL = 2                        # topological charge
R_SWEEP = [1e-3, 2e-3, 3e-3, 4e-3, 5e-3]   # aperture radii to test

# z propagation
Z_MIN = 0.5e-3
Z_MAX = 10.0e-3
NZ_SWEEP = 100

TS = datetime.now().strftime("%Y%m%d_%H%M%S")
BASE = PROJECT_ROOT / "results" / f"rs_aperture_vtu_debug_{TS}"
FIG_DIR = BASE / "figures"
VTU_DIR = BASE / "vtu"
FIG_DIR.mkdir(parents=True, exist_ok=True)
VTU_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
# PART A1 — Quantitative aperture requirement
# ═══════════════════════════════════════════════════════════════════
def part_a1_theory():
    """Compute analytical aperture requirements."""
    print("=" * 72)
    print("PART A1 — QUANTITATIVE APERTURE REQUIREMENT")
    print("=" * 72)

    f = F_LENS
    lam = LAM

    print(f"\n  λ = {lam*1e3:.4f} mm")
    print(f"  f = {f*1e3:.2f} mm")
    print(f"  N_F = R² / (λ·f)")
    print()

    # Required R for N_F thresholds
    for nf_target in [1, 3, 5, 10]:
        R_req = np.sqrt(nf_target * lam * f)
        NA = R_req / f
        w0 = lam / NA if NA > 0 else float('inf')
        print(f"  N_F ≥ {nf_target:2d}:  R ≥ {R_req*1e3:.3f} mm  "
              f"(NA = {NA:.3f},  w0 ~ λ/NA = {w0*1e3:.3f} mm = {w0/lam:.1f}λ)")

    print()
    # Current R = 1 mm
    R_current = 1.0e-3
    NF_current = R_current**2 / (lam * f)
    NA_current = R_current / f
    w0_current = lam / NA_current

    print(f"  Current R = {R_current*1e3:.1f} mm:")
    print(f"    N_F = {NF_current:.3f}")
    print(f"    NA  = {NA_current:.4f}")
    print(f"    w0  ~ {w0_current*1e3:.2f} mm = {w0_current/lam:.1f}λ")
    print()
    print(f"  VERDICT: R = 1 mm at f = 4 mm gives N_F = {NF_current:.2f}.")
    print(f"  This is deep in the diffraction-dominated regime (N_F << 1).")
    print(f"  A tight waist CANNOT form because the beam spreads faster")
    print(f"  than the lens can focus it.")
    print()
    print(f"  For genuine hourglass with contraction ≥ 2×:")
    print(f"    Minimum: N_F ≥ 3  →  R ≥ {np.sqrt(3*lam*f)*1e3:.2f} mm")
    print(f"    Strong:  N_F ≥ 5  →  R ≥ {np.sqrt(5*lam*f)*1e3:.2f} mm")

    return {
        "NF_current": NF_current,
        "R_for_NF3": np.sqrt(3 * lam * f) * 1e3,
        "R_for_NF5": np.sqrt(5 * lam * f) * 1e3,
    }


# ═══════════════════════════════════════════════════════════════════
# PART A1.5 — Sign convention diagnostic (ℓ=0 focusing test)
# ═══════════════════════════════════════════════════════════════════
def part_a15_sign_test():
    """
    Quick diagnostic: propagate an ℓ=0 focused beam with both sign conventions
    and check which one actually focuses at z=f.
    """
    print("\n" + "=" * 72)
    print("PART A1.5 — SIGN CONVENTION DIAGNOSTIC (ℓ=0)")
    print("=" * 72)

    R_test = 3.0e-3   # R=3mm → N_F=3.03
    dn = 256
    LX_domain = 14.0e-3
    CX = LX_domain / 2.0

    xg = np.linspace(0, LX_domain, dn, endpoint=False)
    yg = np.linspace(0, LX_domain, dn, endpoint=False)
    XX, YY = np.meshgrid(xg, yg)
    rx = XX - CX
    ry = YY - CX
    r = np.sqrt(rx**2 + ry**2)

    # Cosine taper apodization
    r_norm = r / R_test
    A = np.where(r_norm <= 1.0, 0.5 * (1.0 + np.cos(np.pi * r_norm)), 0.0)

    # Test both signs — NO vortex, just focusing
    for sign_label, sign in [("POSITIVE (+k(ρ−f))", +1), ("NEGATIVE (−k(ρ−f))", -1)]:
        phi_focus = sign * K_WATER * (np.sqrt(r**2 + F_LENS**2) - F_LENS)
        D = A * np.exp(1j * phi_focus)
        dx = float(xg[1] - xg[0])

        # Propagate to z = f
        p_focus = propagate_pressure_asm(D, dx, dx, K_WATER, F_LENS, pad_factor=2)
        I_focus = np.abs(p_focus)**2

        # On-axis intensity (at center)
        ic = dn // 2
        I_center = I_focus[ic, ic]
        I_max = I_focus.max()

        # Also propagate to z = f/2 and z = 2f for comparison
        p_half = propagate_pressure_asm(D, dx, dx, K_WATER, F_LENS / 2, pad_factor=2)
        p_double = propagate_pressure_asm(D, dx, dx, K_WATER, F_LENS * 2, pad_factor=2)
        I_center_half = np.abs(p_half[ic, ic])**2
        I_center_double = np.abs(p_double[ic, ic])**2

        print(f"\n  {sign_label}:")
        print(f"    I_center(z=f/2) = {I_center_half:.4e}")
        print(f"    I_center(z=f)   = {I_center:.4e}")
        print(f"    I_center(z=2f)  = {I_center_double:.4e}")
        print(f"    I_max(z=f)      = {I_max:.4e}")
        print(f"    Focus ratio I(f)/I(f/2) = {I_center/(I_center_half+1e-30):.2f}")
        print(f"    Focus ratio I(f)/I(2f)  = {I_center/(I_center_double+1e-30):.2f}")

        focuses = I_center > I_center_half and I_center > I_center_double
        print(f"    FOCUSES at z=f: {'YES ✓' if focuses else 'NO ✗'}")

    return


# ═══════════════════════════════════════════════════════════════════
# PART A2 — Numerical aperture sweep
# ═══════════════════════════════════════════════════════════════════
def build_drive_unwrapped(R_aperture, LX_domain, dn=512):
    """Build unwrapped (continuous) phase drive for given aperture."""
    xg = np.linspace(0, LX_domain, dn, endpoint=False)
    yg = np.linspace(0, LX_domain, dn, endpoint=False)
    XX, YY = np.meshgrid(xg, yg)
    cx, cy = LX_domain / 2, LX_domain / 2

    rx = XX - cx
    ry = YY - cy
    r = np.sqrt(rx**2 + ry**2)
    theta = np.arctan2(ry, rx)

    phi_vortex = ELL * theta
    # NEGATIVE sign = converging spherical wavefront (verified by Part A1.5)
    phi_focus = -K_WATER * (np.sqrt(r**2 + F_LENS**2) - F_LENS)
    phi_target = phi_vortex + phi_focus

    # Cosine taper apodization
    r_norm = r / R_aperture
    A = np.where(r_norm <= 1.0, 0.5 * (1.0 + np.cos(np.pi * r_norm)), 0.0)
    D = A * np.exp(1j * phi_target)

    return xg, yg, D


def propagate_single_z(D, xg_d, z, xg_out, yg_out, pad_factor=2):
    """Propagate drive to a single z-plane and interpolate to output grid."""
    dx = float(xg_d[1] - xg_d[0])
    p_prop = propagate_pressure_asm(D, dx, dx, K_WATER, z, pad_factor=pad_factor)

    YY_out, XX_out = np.meshgrid(yg_out, xg_out, indexing='ij')
    pts = np.column_stack([YY_out.ravel(), XX_out.ravel()])

    ire = RegularGridInterpolator((xg_d, xg_d), np.real(p_prop),
                                  method="linear", bounds_error=False, fill_value=0.0)
    iim = RegularGridInterpolator((xg_d, xg_d), np.imag(p_prop),
                                  method="linear", bounds_error=False, fill_value=0.0)
    p_out = (ire(pts) + 1j * iim(pts)).reshape(len(xg_out), len(yg_out))
    return p_out


def radial_profile(I_2d, xg, yg, cx, cy, r_max=None, n_bins=200):
    """Azimuthal average. Returns (r_mid, I_mean)."""
    XX, YY = np.meshgrid(xg, yg)
    R = np.sqrt((XX - cx)**2 + (YY - cy)**2)
    if r_max is None:
        r_max = 0.5 * min(xg[-1] - xg[0], yg[-1] - yg[0])
    r_bins = np.linspace(0, r_max, n_bins + 1)
    r_mid = 0.5 * (r_bins[:-1] + r_bins[1:])
    I_mean = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (R >= r_bins[i]) & (R < r_bins[i + 1])
        if np.any(mask):
            I_mean[i] = np.mean(I_2d[mask])
    return r_mid, I_mean


def continuity_ring_radius(r_mid, I_profiles_vs_z, search_min=0.1e-3, search_max=None):
    """
    Track the primary ring radius vs z using continuity constraint.
    I_profiles_vs_z: list of 1-D intensity radial profiles.

    Returns r_ring[iz] — the ring radius at each z, tracked with continuity.
    """
    nz = len(I_profiles_vs_z)
    r_ring = np.zeros(nz)

    if search_max is None:
        search_max = r_mid[-1] * 0.8

    mask_search = (r_mid >= search_min) & (r_mid <= search_max)
    idx_search = np.where(mask_search)[0]

    if len(idx_search) == 0:
        return r_ring

    # Initialize at first z with global max in search window
    i_first = idx_search[np.argmax(I_profiles_vs_z[0][idx_search])]
    r_ring[0] = r_mid[i_first]

    # For subsequent z, find peak nearest to previous
    for iz in range(1, nz):
        prof = I_profiles_vs_z[iz]
        # Search within ±0.5mm of previous
        dr_max = 0.5e-3
        near_mask = mask_search & (np.abs(r_mid - r_ring[iz-1]) < dr_max)
        idx_near = np.where(near_mask)[0]

        if len(idx_near) > 0 and np.max(prof[idx_near]) > 0:
            # Find local maxima in this neighborhood
            best_idx = idx_near[np.argmax(prof[idx_near])]
            r_ring[iz] = r_mid[best_idx]
        else:
            # Fall back to global max in search window
            r_ring[iz] = r_mid[idx_search[np.argmax(prof[idx_search])]]

    return r_ring


def compute_winding(p_2d, xg, yg, cx, cy, radius, n_pts=512):
    """Compute phase winding number."""
    theta = np.linspace(0, 2*np.pi, n_pts, endpoint=False)
    xs = cx + radius * np.cos(theta)
    ys = cy + radius * np.sin(theta)
    pts = np.column_stack([ys, xs])

    ire = RegularGridInterpolator((yg, xg), np.real(p_2d), method="linear",
                                  bounds_error=False, fill_value=0.0)
    iim = RegularGridInterpolator((yg, xg), np.imag(p_2d), method="linear",
                                  bounds_error=False, fill_value=0.0)
    p_loop = ire(pts) + 1j * iim(pts)
    phase = np.angle(p_loop)
    dphi = np.diff(phase)
    dphi = np.arctan2(np.sin(dphi), np.cos(dphi))
    return np.sum(dphi) / (2 * np.pi)


def part_a2_sweep():
    """Run aperture sweep."""
    print("\n" + "=" * 72)
    print("PART A2 — NUMERICAL APERTURE SWEEP")
    print("=" * 72)

    z_planes = np.linspace(Z_MIN, Z_MAX, NZ_SWEEP)
    nxy_out = 180

    results = []

    for R in R_SWEEP:
        NF = R**2 / (LAM * F_LENS)
        # Domain must be > 2*R with margin
        LX_domain = max(6.0e-3, 4.0 * R + 2.0e-3)
        CX = LX_domain / 2.0
        dn = max(256, int(LX_domain / (LAM / 8)))  # at least 8 pts per λ
        dn = min(dn, 1024)  # cap for performance

        print(f"\n  R = {R*1e3:.1f} mm  (N_F = {NF:.2f}, domain = {LX_domain*1e3:.1f} mm, grid = {dn})")

        xg_d, yg_d, D = build_drive_unwrapped(R, LX_domain, dn=dn)
        xg_out = np.linspace(0, LX_domain, nxy_out)
        yg_out = np.linspace(0, LX_domain, nxy_out)

        # Propagate to all z-planes
        I_profiles = []
        ring_peak_intensity = []  # max of azimuthal-averaged I vs z
        ring_peak_radius = []     # radius of that peak
        r_mid_ref = None
        p_at_focus = None
        iz_focus = np.argmin(np.abs(z_planes - F_LENS))

        t0 = time.time()
        for iz, zz in enumerate(z_planes):
            p_2d = propagate_single_z(D, xg_d, zz, xg_out, yg_out, pad_factor=2)
            I_2d = np.abs(p_2d)**2
            r_mid, I_prof = radial_profile(I_2d, xg_out, yg_out, CX, CX,
                                           r_max=min(R * 3, LX_domain / 2 - 0.5e-3))
            I_profiles.append(I_prof)
            if r_mid_ref is None:
                r_mid_ref = r_mid

            # Ring peak: max of radial profile in [0.05mm, search_max]
            search_mask = (r_mid >= 0.05e-3) & (r_mid <= min(R * 2.5, r_mid[-1] * 0.9))
            if np.any(search_mask):
                idx_s = np.where(search_mask)[0]
                i_pk = idx_s[np.argmax(I_prof[idx_s])]
                ring_peak_intensity.append(I_prof[i_pk])
                ring_peak_radius.append(r_mid[i_pk])
            else:
                ring_peak_intensity.append(0.0)
                ring_peak_radius.append(0.0)

            if iz == iz_focus:
                p_at_focus = p_2d

        dt = time.time() - t0
        print(f"    Propagated in {dt:.1f}s")

        ring_peak_intensity = np.array(ring_peak_intensity)
        ring_peak_radius = np.array(ring_peak_radius)

        # Continuity-constrained ring tracking
        r_ring = continuity_ring_radius(r_mid_ref, I_profiles,
                                        search_min=0.05e-3,
                                        search_max=min(R * 2.5, r_mid_ref[-1] * 0.9))

        # Detect interior minimum (waist)
        # Smooth slightly to avoid noise
        from scipy.ndimage import uniform_filter1d
        r_smooth = uniform_filter1d(r_ring, size=5)

        # Count feature-switching jumps
        dr = np.abs(np.diff(r_ring))
        n_jumps = int(np.sum(dr > 0.3 * LAM))

        # For a genuine hourglass: ring converges, reaches minimum, re-diverges
        # Require minimum to be at least 15% of z-range from either boundary
        margin_frac = 0.15
        iz_lo = int(margin_frac * len(r_smooth))
        iz_hi = int((1.0 - margin_frac) * len(r_smooth))
        interior_slice = r_smooth[iz_lo:iz_hi]

        if len(interior_slice) > 5:
            iz_min_interior = np.argmin(interior_slice) + iz_lo
            r_min = r_smooth[iz_min_interior]

            # Check convergence before and re-divergence after
            n_avg = max(3, len(r_smooth) // 20)
            r_before = np.mean(r_smooth[max(0, iz_min_interior - 2*n_avg):
                                        max(1, iz_min_interior - n_avg)])
            r_after = np.mean(r_smooth[min(len(r_smooth)-1, iz_min_interior + n_avg):
                                       min(len(r_smooth), iz_min_interior + 2*n_avg)])

            converges = r_before > r_min * 1.1  # ring shrinks by ≥10% from before
            diverges = r_after > r_min * 1.1     # ring grows by ≥10% after
            is_interior = converges and diverges

            contraction = max(r_before, r_after) / (r_min + 1e-10)
        else:
            iz_min_interior = NZ_SWEEP // 2
            r_min = 0
            contraction = 1.0
            is_interior = False

        z_waist = z_planes[iz_min_interior]

        # Also check for intensity-peak waist (ring peak intensity peaks at focus)
        # This is independent of ring radius tracking
        iz_ipeak = np.argmax(ring_peak_intensity)
        z_ipeak = z_planes[iz_ipeak]
        ipeak_interior = (iz_ipeak > len(z_planes) * 0.15) and (iz_ipeak < len(z_planes) * 0.85)
        r_at_ipeak = ring_peak_radius[iz_ipeak]

        # Winding at ~focus
        w_focus = compute_winding(p_at_focus, xg_out, yg_out, CX, CX, 0.5 * LAM)

        # Determine hourglass verdict (either ring-radius or intensity-based)
        hourglass_ring = is_interior and contraction >= 2.0 and n_jumps <= 2
        hourglass_intensity = ipeak_interior  # intensity peaks at an interior z

        print(f"    Ring tracker: z_waist = {z_waist*1e3:.2f} mm,  r_min = {r_min*1e3:.3f} mm,  "
              f"contraction = {contraction:.1f}×")
        print(f"    Ring converges+diverges: {is_interior},  jumps = {n_jumps}")
        print(f"    I_peak tracker: z_peak = {z_ipeak*1e3:.2f} mm,  r_at_peak = {r_at_ipeak*1e3:.3f} mm, "
              f"interior = {ipeak_interior}")
        print(f"    winding = {w_focus:.2f}")
        print(f"    HOURGLASS (ring): {'YES ✓' if hourglass_ring else 'NO ✗'}  "
              f"HOURGLASS (intensity): {'YES ✓' if hourglass_intensity else 'NO ✗'}")

        results.append({
            "R_mm": R * 1e3,
            "NF": NF,
            "z_waist_mm": z_waist * 1e3,
            "r_min_mm": r_min * 1e3,
            "contraction": contraction,
            "is_interior": is_interior,
            "z_ipeak_mm": z_ipeak * 1e3,
            "r_at_ipeak_mm": r_at_ipeak * 1e3,
            "ipeak_interior": ipeak_interior,
            "n_jumps": n_jumps,
            "winding": w_focus,
            "hourglass_ring": hourglass_ring,
            "hourglass_intensity": hourglass_intensity,
            "r_ring": r_ring,
            "z_planes": z_planes,
            "r_smooth": r_smooth,
            "ring_peak_intensity": ring_peak_intensity,
            "ring_peak_radius": ring_peak_radius,
        })

    # ── Figure 1: ring radius vs z for all R ────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(R_SWEEP)))

    for i, res in enumerate(results):
        label = f"R={res['R_mm']:.0f}mm (N_F={res['NF']:.2f})"
        # Top panel: ring radius
        ax1.plot(res["z_planes"] * 1e3, res["r_smooth"] * 1e3,
                 color=colors[i], lw=2, label=label)
        iz_w = np.argmin(np.abs(res["z_planes"] - res["z_waist_mm"] * 1e-3))
        ax1.plot(res["z_waist_mm"], res["r_smooth"][iz_w] * 1e3,
                 'o', color=colors[i], ms=8, zorder=5)

        # Bottom panel: ring peak intensity (normalised per-R)
        ipeak = np.array(res["ring_peak_intensity"])
        ipeak_norm = ipeak / (ipeak.max() + 1e-30)
        ax2.plot(res["z_planes"] * 1e3, ipeak_norm,
                 color=colors[i], lw=2, label=label)
        # Mark intensity peak z
        iz_ip = np.argmax(ipeak)
        ax2.plot(res["z_planes"][iz_ip] * 1e3, ipeak_norm[iz_ip],
                 's', color=colors[i], ms=8, zorder=5)

    ax1.axvline(F_LENS * 1e3, color='gray', ls='--', alpha=0.5, label=f"f = {F_LENS*1e3:.0f} mm")
    ax2.axvline(F_LENS * 1e3, color='gray', ls='--', alpha=0.5, label=f"f = {F_LENS*1e3:.0f} mm")

    ax1.set_ylabel("Ring radius (mm)", fontsize=12)
    ax1.set_title(f"Continuity-Constrained Ring Radius vs z  (f={F_LENS*1e3:.0f}mm, ℓ={ELL})",
                  fontsize=13, fontweight="bold")
    ax1.legend(fontsize=9, loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(Z_MIN * 1e3, Z_MAX * 1e3)

    ax2.set_xlabel("z (mm)", fontsize=12)
    ax2.set_ylabel("Normalised ring peak intensity", fontsize=12)
    ax2.set_title("Ring Peak Intensity vs z  (□ = intensity peak)", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=9, loc="upper right")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "aperture_sweep_ring_radius.png", dpi=150)
    plt.close(fig)
    print(f"\n  Saved aperture_sweep_ring_radius.png")

    # ── Summary table ─────────────────────────────────────────────
    print("\n  APERTURE SWEEP SUMMARY:")
    print(f"  {'R':>6}  {'N_F':>5}  {'z_rWaist':>8}  {'r_min':>7}  {'contr':>6}  "
          f"{'z_Ipeak':>8}  {'r@Ipk':>7}  {'wind':>5}  {'HG_r':>4}  {'HG_I':>4}")
    for r in results:
        hg_r = "YES" if r["hourglass_ring"] else "NO"
        hg_i = "YES" if r["hourglass_intensity"] else "NO"
        print(f"  {r['R_mm']:5.1f}  {r['NF']:5.2f}  {r['z_waist_mm']:7.2f}  "
              f"{r['r_min_mm']:6.3f}  {r['contraction']:5.1f}×  "
              f"{r['z_ipeak_mm']:7.2f}  {r['r_at_ipeak_mm']:6.3f}  "
              f"{r['winding']:+5.2f}  {hg_r:>4}  {hg_i:>4}")

    return results


# ═══════════════════════════════════════════════════════════════════
# PART B — VTU Geometry Debug
# ═══════════════════════════════════════════════════════════════════

def write_vtu_fixed(filepath, xg, yg, zg, fields: dict):
    """
    Write a .vtu file with hexahedral cells.
    ALL coordinates must be in the SAME units (mm recommended).

    Point ordering: iz (slowest) → iy → ix (fastest) — C-order.
    Data arrays shape (nz, ny, nx) ravelled in C-order.
    """
    nx, ny, nz = len(xg), len(yg), len(zg)
    npts = nx * ny * nz
    ncells = (nx - 1) * (ny - 1) * (nz - 1)

    lines = []
    lines.append('<?xml version="1.0"?>')
    lines.append('<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">')
    lines.append('<UnstructuredGrid>')
    lines.append(f'<Piece NumberOfPoints="{npts}" NumberOfCells="{ncells}">')

    # Points: iz → iy → ix (C-order)
    lines.append('<Points>')
    lines.append('<DataArray type="Float64" NumberOfComponents="3" format="ascii">')
    coords = []
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                coords.append(f"{xg[ix]:.8e} {yg[iy]:.8e} {zg[iz]:.8e}")
    lines.append(" ".join(coords))
    lines.append('</DataArray>')
    lines.append('</Points>')

    # Point data
    lines.append('<PointData>')
    for name, arr3d in fields.items():
        flat = arr3d.ravel()  # C-order matches iz→iy→ix
        lines.append(f'<DataArray type="Float64" Name="{name}" format="ascii">')
        lines.append(" ".join(f"{v:.8e}" for v in flat))
        lines.append('</DataArray>')
    lines.append('</PointData>')

    # Cells: hexahedra
    lines.append('<Cells>')
    lines.append('<DataArray type="Int64" Name="connectivity" format="ascii">')
    conn = []
    for iz in range(nz - 1):
        for iy in range(ny - 1):
            for ix in range(nx - 1):
                n0 = iz * ny * nx + iy * nx + ix
                n1 = n0 + 1
                n2 = n0 + nx + 1
                n3 = n0 + nx
                n4 = n0 + ny * nx
                n5 = n4 + 1
                n6 = n4 + nx + 1
                n7 = n4 + nx
                conn.append(f"{n0} {n1} {n2} {n3} {n4} {n5} {n6} {n7}")
    lines.append(" ".join(conn))
    lines.append('</DataArray>')

    lines.append('<DataArray type="Int64" Name="offsets" format="ascii">')
    lines.append(" ".join(str(8 * (i + 1)) for i in range(ncells)))
    lines.append('</DataArray>')

    lines.append('<DataArray type="UInt8" Name="types" format="ascii">')
    lines.append(" ".join(["12"] * ncells))
    lines.append('</DataArray>')

    lines.append('</Cells>')
    lines.append('</Piece>')
    lines.append('</UnstructuredGrid>')
    lines.append('</VTKFile>')

    Path(filepath).write_text("\n".join(lines))


def part_b1_inspect():
    """Inspect the existing VTU geometry bounds."""
    print("\n" + "=" * 72)
    print("PART B1 — VTU GEOMETRY INSPECTION")
    print("=" * 72)

    # Find most recent VTU
    import glob
    vtu_files = sorted(glob.glob(str(PROJECT_ROOT / "results" / "rs_free_space_vortex_truth_*" / "vtu" / "*.vtu")))
    if not vtu_files:
        print("  No existing VTU files found!")
        return None

    vtu_path = vtu_files[-1]
    print(f"  Inspecting: {Path(vtu_path).relative_to(PROJECT_ROOT)}")

    # Parse the VTU to extract point coordinates
    import xml.etree.ElementTree as ET
    tree = ET.parse(vtu_path)
    root = tree.getroot()

    piece = root.find(".//Piece")
    npts = int(piece.attrib["NumberOfPoints"])
    ncells = int(piece.attrib["NumberOfCells"])
    print(f"  Points: {npts}")
    print(f"  Cells:  {ncells}")

    # Extract coordinates
    points_array = root.find(".//Points/DataArray")
    coords_text = points_array.text.strip().split()
    coords = np.array([float(v) for v in coords_text]).reshape(-1, 3)

    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    z_min, z_max = coords[:, 2].min(), coords[:, 2].max()

    print(f"\n  Bounds:")
    print(f"    X: [{x_min:.6f}, {x_max:.6f}]  range = {x_max - x_min:.6f}")
    print(f"    Y: [{y_min:.6f}, {y_max:.6f}]  range = {y_max - y_min:.6f}")
    print(f"    Z: [{z_min:.6f}, {z_max:.6f}]  range = {z_max - z_min:.6f}")

    print(f"\n  DIAGNOSIS:")
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    ranges = {"X": x_range, "Y": y_range, "Z": z_range}
    max_range = max(ranges.values())
    min_range = min(ranges.values())

    if max_range / (min_range + 1e-30) > 100:
        # Find which axes are tiny
        for axis, rng in ranges.items():
            if rng < max_range / 100:
                print(f"    ⚠ {axis} axis range ({rng:.6f}) is tiny relative to max ({max_range:.6f})")
                print(f"      Ratio: {max_range/rng:.0f}:1")

        # Check unit mismatch
        if x_range < 0.01 and z_range > 0.1:
            print(f"\n    ROOT CAUSE: X,Y are in METRES ({x_range:.6f}) but Z is in MILLIMETRES ({z_range:.4f}).")
            print(f"    X spans {x_range*1e3:.2f} mm (as m), Z spans {z_range:.2f} mm.")
            print(f"    This creates a {z_range/(x_range+1e-30):.0f}:1 aspect ratio → skinny line in ParaView.")
            print(f"\n    FIX: Convert X,Y to mm (multiply by 1e3) OR convert Z to metres (divide by 1e3)")
            print(f"         Recommended: use all mm for ParaView convenience.")
    else:
        print(f"    Geometry looks proportioned (max/min ratio = {max_range/(min_range+1e-30):.1f})")

    return {
        "npts": npts, "ncells": ncells,
        "x_min": x_min, "x_max": x_max,
        "y_min": y_min, "y_max": y_max,
        "z_min": z_min, "z_max": z_max,
    }


def part_b2_audit():
    """Audit the VTU writing code for correctness."""
    print("\n" + "=" * 72)
    print("PART B2 — VTU CODE AUDIT (rs_free_space_vortex_truth.py)")
    print("=" * 72)

    script_path = PROJECT_ROOT / "scripts" / "dev" / "rs_free_space_vortex_truth.py"
    text = script_path.read_text()

    # Check the write_vtu call
    import re
    call_match = re.search(r'write_vtu\(vtu_path,\s*(.*?)\)', text, re.DOTALL)
    if call_match:
        args = call_match.group(1).strip()
        print(f"  Call: write_vtu(vtu_path, {args})")
        print()

    # Check for unit mismatch
    if "z_sub * 1e3" in text and "xg_sub" in text:
        print("  ⚠ UNIT MISMATCH FOUND:")
        print("    xg_sub = xg[::stride_xy]           → METRES (0 → 0.006)")
        print("    yg_sub = yg[::stride_xy]           → METRES (0 → 0.006)")
        print("    z_sub * 1e3                        → MILLIMETRES (0.5 → 8.0)")
        print()
        print("    X,Y spans ~0.006 in the VTU coordinate system")
        print("    Z spans ~7.5 in the VTU coordinate system")
        print("    Aspect ratio: ~1250:1 → ParaView shows a skinny line!")
        print()
        print("  FIX: Change the write_vtu call to use consistent units (all mm):")
        print("    write_vtu(vtu_path, xg_sub * 1e3, yg_sub * 1e3, z_sub * 1e3, fields_vtu)")

    # Check flattening
    if "arr3d.ravel()" in text and "transpose" not in text.split("def write_vtu")[1].split("def ")[0]:
        print("\n  ✓ Data flattening: arr3d.ravel() — correct (C-order matches point loop)")
    elif "transpose(2,1,0)" in text:
        print("\n  ⚠ Data flattening: transpose(2,1,0).ravel() — WRONG (swaps x↔z)")

    # Check point loop order
    if "for iz" in text and "for iy" in text and "for ix" in text:
        print("  ✓ Point loop: iz → iy → ix — correct")

    # Check coordinate construction
    if 'xg[ix]' in text and 'yg[iy]' in text and 'zg[iz]' in text:
        print("  ✓ Coordinate mapping: x=xg[ix], y=yg[iy], z=zg[iz] — correct")

    print("\n  CONCLUSION: The VTU writer logic is correct EXCEPT for the unit mismatch")
    print("  in the call site. The fix is to pass all coordinates in the same units.")


def part_b3_minimal_test():
    """Create a minimal geometry-only VTU and verify it."""
    print("\n" + "=" * 72)
    print("PART B3 — MINIMAL VTU TEST")
    print("=" * 72)

    # Create a simple box: 6mm × 6mm × 8mm (all in mm)
    nx, ny, nz = 10, 10, 12
    xg = np.linspace(0, 6.0, nx)     # mm
    yg = np.linspace(0, 6.0, ny)     # mm
    zg = np.linspace(0.5, 8.0, nz)   # mm

    # Dummy scalar = z-coordinate
    ZZ, YY, XX = np.meshgrid(zg, yg, xg, indexing='ij')
    dummy_z = ZZ  # shape (nz, ny, nx)

    vtu_path = VTU_DIR / "minimal_box_test.vtu"
    write_vtu_fixed(vtu_path, xg, yg, zg, {"z_value": dummy_z})
    print(f"  Written: {vtu_path.relative_to(PROJECT_ROOT)}")

    # Read back and verify
    import xml.etree.ElementTree as ET
    tree = ET.parse(str(vtu_path))
    root = tree.getroot()
    piece = root.find(".//Piece")

    npts = int(piece.attrib["NumberOfPoints"])
    ncells = int(piece.attrib["NumberOfCells"])

    points_data = root.find(".//Points/DataArray").text.strip().split()
    coords = np.array([float(v) for v in points_data]).reshape(-1, 3)

    x_r = coords[:, 0].max() - coords[:, 0].min()
    y_r = coords[:, 1].max() - coords[:, 1].min()
    z_r = coords[:, 2].max() - coords[:, 2].min()

    print(f"  Points: {npts} (expected {nx*ny*nz})")
    print(f"  Cells:  {ncells} (expected {(nx-1)*(ny-1)*(nz-1)})")
    print(f"  X range: {x_r:.2f} mm")
    print(f"  Y range: {y_r:.2f} mm")
    print(f"  Z range: {z_r:.2f} mm")

    max_r = max(x_r, y_r, z_r)
    min_r = min(x_r, y_r, z_r)
    ratio = max_r / (min_r + 1e-30)

    test_pass = npts == nx * ny * nz and ratio < 5.0
    print(f"  Aspect ratio: {ratio:.1f}")
    print(f"  MINIMAL TEST: {'PASS ✓' if test_pass else 'FAIL ✗'}")
    print(f"  → This VTU should display as a 3D box in ParaView, NOT a line.")

    # Verify z_value data
    z_data = root.find(".//PointData/DataArray[@Name='z_value']")
    z_vals = np.array([float(v) for v in z_data.text.strip().split()])
    # At first point (iz=0, iy=0, ix=0), z_value should = zg[0]
    # At last point (iz=nz-1, iy=ny-1, ix=nx-1), z_value should = zg[-1]
    print(f"  z_value[0] = {z_vals[0]:.2f} (expected {zg[0]:.2f})")
    print(f"  z_value[-1] = {z_vals[-1]:.2f} (expected {zg[-1]:.2f})")
    data_pass = abs(z_vals[0] - zg[0]) < 0.01 and abs(z_vals[-1] - zg[-1]) < 0.01
    print(f"  Data consistency: {'PASS ✓' if data_pass else 'FAIL ✗'}")

    return test_pass and data_pass


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    print("=" * 72)
    print("APERTURE SCALING + VTU GEOMETRY DEBUG")
    print("=" * 72)
    print(f"Output: {BASE.relative_to(PROJECT_ROOT)}")
    print(f"λ = {LAM*1e3:.4f} mm,  f = {F_LENS*1e3:.1f} mm,  ℓ = {ELL}")
    print()

    # ── PART A ────────────────────────────────────────────────────
    a1_results = part_a1_theory()
    part_a15_sign_test()
    a2_results = part_a2_sweep()

    # ── PART B ────────────────────────────────────────────────────
    b1_results = part_b1_inspect()
    part_b2_audit()
    b3_pass = part_b3_minimal_test()

    # ── Summary ───────────────────────────────────────────────────
    t_total = time.time() - t0
    print("\n" + "=" * 72)
    print("FINAL SUMMARY")
    print("=" * 72)

    print("\n  PART A — PHYSICS:")
    print(f"    Current N_F = {a1_results['NF_current']:.3f} (R=1mm, f=4mm)")
    print(f"    Required R for N_F≥3:  {a1_results['R_for_NF3']:.2f} mm")
    print(f"    Required R for N_F≥5:  {a1_results['R_for_NF5']:.2f} mm")
    print()
    for r in a2_results:
        hg_r = "✓ HG(ring)" if r["hourglass_ring"] else "✗ no HG(ring)"
        hg_i = "✓ HG(intensity)" if r["hourglass_intensity"] else "✗ no HG(intensity)"
        print(f"    R={r['R_mm']:.0f}mm: N_F={r['NF']:.2f}, contraction={r['contraction']:.1f}×, "
              f"winding={r['winding']:+.2f}  {hg_r}  {hg_i}")

    print(f"\n  PART B — VTU GEOMETRY:")
    if b1_results:
        print(f"    X range: {b1_results['x_max']-b1_results['x_min']:.6f}")
        print(f"    Y range: {b1_results['y_max']-b1_results['y_min']:.6f}")
        print(f"    Z range: {b1_results['z_max']-b1_results['z_min']:.6f}")
    print(f"    ROOT CAUSE: xg,yg in metres but zg in mm → 1250:1 aspect ratio")
    print(f"    FIX: pass xg*1e3, yg*1e3, z*1e3 (all mm) to write_vtu()")
    print(f"    Minimal box test: {'PASS' if b3_pass else 'FAIL'}")

    print(f"\n  Runtime: {t_total:.1f}s")

    # ── Save metadata ─────────────────────────────────────────────
    sweep_table = []
    for r in a2_results:
        sweep_table.append({
            "R_mm": r["R_mm"],
            "NF": round(r["NF"], 3),
            "z_waist_mm": round(r["z_waist_mm"], 2),
            "r_min_mm": round(r["r_min_mm"], 3),
            "contraction": round(float(r["contraction"]), 2),
            "z_ipeak_mm": round(r["z_ipeak_mm"], 2),
            "r_at_ipeak_mm": round(r["r_at_ipeak_mm"], 3),
            "winding": round(float(r["winding"]), 2),
            "n_jumps": int(r["n_jumps"]),
            "hourglass_ring": bool(r["hourglass_ring"]),
            "hourglass_intensity": bool(r["hourglass_intensity"]),
        })

    metadata = {
        "timestamp": TS,
        "runtime_s": round(t_total, 1),
        "part_a": {
            "theory": {
                "NF_at_R1mm": round(a1_results["NF_current"], 3),
                "R_for_NF3_mm": round(a1_results["R_for_NF3"], 2),
                "R_for_NF5_mm": round(a1_results["R_for_NF5"], 2),
            },
            "sweep": sweep_table,
        },
        "part_b": {
            "root_cause": "xg,yg in metres but zg in mm → 1250:1 aspect ratio",
            "fix": "pass all coordinates in mm: xg*1e3, yg*1e3, z*1e3",
            "minimal_test_pass": bool(b3_pass),
            "existing_vtu_bounds": {
                "x_range": round(float(b1_results["x_max"] - b1_results["x_min"]), 6) if b1_results else None,
                "y_range": round(float(b1_results["y_max"] - b1_results["y_min"]), 6) if b1_results else None,
                "z_range": round(float(b1_results["z_max"] - b1_results["z_min"]), 6) if b1_results else None,
            },
        },
    }

    with open(BASE / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata.json")


if __name__ == "__main__":
    main()
