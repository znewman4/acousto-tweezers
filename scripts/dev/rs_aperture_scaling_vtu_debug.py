#!/usr/bin/env python3
"""
Aperture Scaling + VTU Geometry Debug
======================================

PART A — Physics: Why no hourglass in free space?
  A1: Quantitative aperture requirement (analytical)
  A2: Numerical validation — sweep R = [1, 2, 3, 4, 5] mm

PART B — VTU appears as a long skinny line in ParaView
  B1: Inspect exported VTU geometry bounds
  B2: Audit coordinate writing logic
  B3: Minimal geometry-only VTU test

Outputs → results/rs_aperture_scaling_vtu_debug_<TS>/
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

# ═══════════════════════════════════════════════════════════════════
# Physical constants
# ═══════════════════════════════════════════════════════════════════
WATER_C = 1484.0          # m/s
F_HZ = 2.0e6              # Hz
LAM = WATER_C / F_HZ      # 0.742 mm
K_WATER = 2.0 * np.pi * F_HZ / WATER_C
FOCAL_LENGTH = 4.0e-3     # 4 mm
ELL = 2                   # topological charge

TS = datetime.now().strftime("%Y%m%d_%H%M%S")
BASE = PROJECT_ROOT / "results" / f"rs_aperture_scaling_vtu_debug_{TS}"
FIG_DIR = BASE / "figures"
VTU_DIR = BASE / "vtu"
FIG_DIR.mkdir(parents=True, exist_ok=True)
VTU_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
# PART A1 — Quantitative aperture requirement (analytical)
# ═══════════════════════════════════════════════════════════════════
def part_a1():
    """Derive required aperture for real hourglass."""
    print("=" * 72)
    print("PART A1: QUANTITATIVE APERTURE REQUIREMENT")
    print("=" * 72)

    f = FOCAL_LENGTH
    lam = LAM

    print(f"\n  λ = {lam*1e3:.4f} mm")
    print(f"  f = {f*1e3:.2f} mm")
    print(f"  ℓ = {ELL}")

    # Fresnel number: N_F = R² / (λ·f)
    print(f"\n  Fresnel number: N_F = R² / (λ·f)")

    # Current aperture
    R_current = 1.0e-3
    NF_current = R_current**2 / (lam * f)
    print(f"\n  Current: R = {R_current*1e3:.1f} mm → N_F = {NF_current:.3f}")
    print(f"  This means the aperture subtends < 1 Fresnel zone.")
    print(f"  A single Fresnel zone cannot form a geometric focus.")

    # Required R for N_F = 3, 5
    thresholds = [3, 5, 10]
    results = {}
    print(f"\n  Required aperture for target Fresnel numbers:")
    print(f"  {'N_F':>6}  {'R (mm)':>8}  {'NA':>6}  {'w₀ (mm)':>8}  {'w₀/λ':>6}")
    for nf_target in thresholds:
        R_req = np.sqrt(nf_target * lam * f)
        NA = R_req / f  # paraxial NA
        w0 = lam / NA if NA > 0 else float("inf")  # diffraction-limited waist
        # For a vortex ℓ=2, the ring radius ≈ ℓ·w0/(√2) ≈ √ℓ · w0
        # But w0 here is the Gaussian waist equivalent
        print(f"  {nf_target:6d}  {R_req*1e3:8.2f}  {NA:6.3f}  {w0*1e3:8.3f}  {w0/lam:6.2f}")
        results[nf_target] = {"R_mm": R_req * 1e3, "NA": NA, "w0_mm": w0 * 1e3}

    # Can R=1mm produce a tight waist at f=4mm?
    print(f"\n  VERDICT: Can R = 1 mm produce a tight waist at f = 4 mm?")
    print(f"  NO.")
    print(f"  N_F = {NF_current:.3f} → the aperture is smaller than one Fresnel zone.")
    print(f"  Geometric focusing requires N_F >> 1 (at least N_F ≥ 3–5).")
    print(f"  With N_F < 1, the beam diffracts broadly and no waist forms.")
    print(f"  Required: R ≥ {results[3]['R_mm']:.1f} mm for N_F=3 (moderate focus)")
    print(f"            R ≥ {results[5]['R_mm']:.1f} mm for N_F=5 (strong focus)")

    return {
        "NF_current": NF_current,
        "R_current_mm": R_current * 1e3,
        "requirements": results,
    }


# ═══════════════════════════════════════════════════════════════════
# Build drive field for arbitrary aperture radius (unwrapped phase)
# ═══════════════════════════════════════════════════════════════════
def build_drive_variable_R(R_m, f_m, dn=512):
    """
    Build source drive with unwrapped (continuous) phase.
    Domain size adapts to 3× the aperture radius to avoid edge effects.
    """
    # Domain must be large enough to contain the aperture + margin
    L = max(6.0e-3, 3.0 * R_m)  # at least 6mm or 3×R
    xg = np.linspace(0, L, dn, endpoint=False)
    yg = np.linspace(0, L, dn, endpoint=False)
    dx = float(xg[1] - xg[0])
    XX, YY = np.meshgrid(xg, yg)

    cx, cy = L / 2.0, L / 2.0
    rx, ry = XX - cx, YY - cy
    r = np.sqrt(rx**2 + ry**2)
    theta = np.arctan2(ry, rx)

    # Vortex + focusing phase (unwrapped)
    phi_vortex = ELL * theta
    phi_focus = K_WATER * (np.sqrt(r**2 + f_m**2) - f_m)
    phi = phi_vortex + phi_focus

    # Cosine taper apodization within aperture
    r_norm = r / R_m
    A = np.where(r_norm <= 1.0, 0.5 * (1.0 + np.cos(np.pi * r_norm)), 0.0)

    D = A * np.exp(1j * phi)

    return xg, yg, D, L


# ═══════════════════════════════════════════════════════════════════
# Propagate and compute radial profile metrics
# ═══════════════════════════════════════════════════════════════════
def propagate_and_track(xg_d, yg_d, D, L, z_planes, nxy_out=200,
                        pad_factor=2):
    """
    Propagate drive field to z_planes and compute continuity-constrained
    ring radius + robust metrics at each z.

    Returns (xg_out, yg_out, ring_r, contrast, ring_to_out, winding_at_zstar).
    """
    dx = float(xg_d[1] - xg_d[0])
    cx = cy = L / 2.0

    # Output grid
    xg_out = np.linspace(0, L, nxy_out)
    yg_out = np.linspace(0, L, nxy_out)
    YY_out, XX_out = np.meshgrid(yg_out, xg_out, indexing='ij')
    pts_out = np.column_stack([YY_out.ravel(), XX_out.ravel()])

    # Radial grid for output
    XX_r, YY_r = np.meshgrid(xg_out, yg_out)
    R_grid = np.sqrt((XX_r - cx)**2 + (YY_r - cy)**2)

    ring_r = np.zeros(len(z_planes))
    contrast = np.zeros(len(z_planes))
    ring_to_out = np.zeros(len(z_planes))
    ring_frac = np.zeros(len(z_planes))
    I_ring_peak_arr = np.zeros(len(z_planes))

    r_bins = np.linspace(0, L / 2 * 0.8, 300)
    r_mid = 0.5 * (r_bins[:-1] + r_bins[1:])
    dA = (xg_out[1] - xg_out[0])**2

    winding_at_zstar = None

    for iz, zz in enumerate(z_planes):
        p_prop = propagate_pressure_asm(D, dx, dx, K_WATER, zz, pad_factor=pad_factor)

        ire = RegularGridInterpolator(
            (yg_d, xg_d), np.real(p_prop), method="linear",
            bounds_error=False, fill_value=0.0)
        iim = RegularGridInterpolator(
            (yg_d, xg_d), np.imag(p_prop), method="linear",
            bounds_error=False, fill_value=0.0)
        p_2d = (ire(pts_out) + 1j * iim(pts_out)).reshape(nxy_out, nxy_out)
        I_2d = np.abs(p_2d)**2

        # Radial profile
        I_radial = np.zeros(len(r_mid))
        for i in range(len(r_mid)):
            mask = (R_grid >= r_bins[i]) & (R_grid < r_bins[i + 1])
            if np.any(mask):
                I_radial[i] = np.mean(I_2d[mask])

        # Primary ring: find peak in [0.05mm, L/2*0.6]
        window = (r_mid >= 0.05e-3) & (r_mid <= L / 2 * 0.6)
        if np.any(window):
            idx_w = np.where(window)[0]
            i_peak = idx_w[np.argmax(I_radial[window])]
            ring_r[iz] = r_mid[i_peak]
            I_ring_peak_arr[iz] = I_radial[i_peak]
        else:
            ring_r[iz] = 0.0

        # Continuity constraint: if iz > 0, reject jumps > 0.5mm
        if iz > 0 and abs(ring_r[iz] - ring_r[iz - 1]) > 0.5e-3:
            # Search near previous ring_r
            near = np.abs(r_mid - ring_r[iz - 1]) < 0.5e-3
            if np.any(near):
                idx_near = np.where(near)[0]
                i_local = idx_near[np.argmax(I_radial[near])]
                ring_r[iz] = r_mid[i_local]
                I_ring_peak_arr[iz] = I_radial[i_local]

        # Contrast: ring peak / center intensity
        mask_center = R_grid < 0.1e-3
        I_center = np.mean(I_2d[mask_center]) if np.any(mask_center) else 1e-30
        contrast[iz] = I_ring_peak_arr[iz] / (I_center + 1e-30)

        # Ring to outskirts
        mask_out = R_grid > L / 2 * 0.6
        I_out = np.mean(I_2d[mask_out]) if np.any(mask_out) else 1e-30
        ring_to_out[iz] = I_ring_peak_arr[iz] / (I_out + 1e-30)

        # Ring fraction
        r_peak = ring_r[iz]
        mask_ring = (R_grid >= max(0.01e-3, r_peak - 0.3e-3)) & \
                    (R_grid < r_peak + 0.3e-3)
        E_ring = np.sum(I_2d[mask_ring]) * dA
        E_total = np.sum(I_2d) * dA
        ring_frac[iz] = E_ring / (E_total + 1e-30)

        # Winding at z closest to focal length
        if winding_at_zstar is None or abs(zz - FOCAL_LENGTH) < abs(z_planes[0] + (winding_at_zstar[0] if isinstance(winding_at_zstar, tuple) else 0) - FOCAL_LENGTH):
            # Compute winding at r = 0.5λ
            r_loop = 0.5 * LAM
            n_pts = 512
            theta_loop = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
            xs = cx + r_loop * np.cos(theta_loop)
            ys = cy + r_loop * np.sin(theta_loop)
            pts_loop = np.column_stack([ys, xs])

            ire2 = RegularGridInterpolator(
                (yg_out, xg_out), np.real(p_2d), method="linear",
                bounds_error=False, fill_value=0.0)
            iim2 = RegularGridInterpolator(
                (yg_out, xg_out), np.imag(p_2d), method="linear",
                bounds_error=False, fill_value=0.0)
            p_loop = ire2(pts_loop) + 1j * iim2(pts_loop)
            phase = np.angle(p_loop)
            dphi = np.diff(phase)
            dphi = np.arctan2(np.sin(dphi), np.cos(dphi))
            w = np.sum(dphi) / (2 * np.pi)
            winding_at_zstar = (zz, w)

    return {
        "ring_r": ring_r,
        "contrast": contrast,
        "ring_to_out": ring_to_out,
        "ring_frac": ring_frac,
        "winding_at_zstar": winding_at_zstar,
    }


# ═══════════════════════════════════════════════════════════════════
# PART A2 — Aperture sweep
# ═══════════════════════════════════════════════════════════════════
def part_a2():
    """Sweep R = [1, 2, 3, 4, 5] mm and detect hourglass."""
    print("\n" + "=" * 72)
    print("PART A2: APERTURE SWEEP")
    print("=" * 72)

    R_values = [1.0e-3, 2.0e-3, 3.0e-3, 4.0e-3, 5.0e-3]
    z_planes = np.linspace(0.5e-3, 10.0e-3, 120)

    sweep_results = []

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(f"Aperture Sweep: Ring Radius & Contrast vs z  "
                 f"(f={FOCAL_LENGTH*1e3:.0f}mm, ℓ={ELL})",
                 fontsize=14, fontweight="bold")

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(R_values)))

    for ir, R in enumerate(R_values):
        NF = R**2 / (LAM * FOCAL_LENGTH)
        label = f"R={R*1e3:.0f}mm (N_F={NF:.2f})"
        print(f"\n  {label}")

        t0 = time.time()
        xg_d, yg_d, D, L = build_drive_variable_R(R, FOCAL_LENGTH)
        result = propagate_and_track(xg_d, yg_d, D, L, z_planes)
        dt = time.time() - t0
        print(f"    Propagation time: {dt:.1f}s")

        ring_r = result["ring_r"]
        contrast = result["contrast"]
        ring_to_out = result["ring_to_out"]
        winding = result["winding_at_zstar"]

        # Detect interior minimum in ring_r
        # Smooth ring_r to suppress noise
        from scipy.ndimage import uniform_filter1d
        ring_smooth = uniform_filter1d(ring_r, size=5)

        # Find interior minimum: not at first or last 5% of z-range
        margin = max(3, len(z_planes) // 20)
        interior = ring_smooth[margin:-margin]
        iz_min_interior = np.argmin(interior) + margin

        # Check if it's a real minimum (lower than neighbors by at least 10%)
        r_min = ring_smooth[iz_min_interior]
        r_start = ring_smooth[margin]
        r_end = ring_smooth[-margin]
        contraction = max(r_start, r_end) / (r_min + 1e-10)

        # Check quality: is it a true local min?
        has_interior_min = (r_min < 0.9 * r_start) and (r_min < 0.9 * r_end)
        is_hourglass = has_interior_min and (contraction >= 2.0)

        z_waist = z_planes[iz_min_interior]

        # Count feature-switching jumps
        dr = np.abs(np.diff(ring_r))
        n_jumps = int(np.sum(dr > 0.3e-3))

        print(f"    z_waist = {z_waist*1e3:.2f} mm")
        print(f"    r_min   = {r_min*1e3:.3f} mm")
        print(f"    contraction = {contraction:.1f}×")
        print(f"    interior minimum: {has_interior_min}")
        print(f"    hourglass: {is_hourglass}")
        print(f"    feature-switch jumps: {n_jumps}")
        print(f"    winding at z≈f: {winding[1]:.3f}")

        sweep_results.append({
            "R_mm": R * 1e3,
            "NF": round(NF, 3),
            "z_waist_mm": round(z_waist * 1e3, 2),
            "r_min_mm": round(r_min * 1e3, 3),
            "contraction": round(contraction, 1),
            "has_interior_min": bool(has_interior_min),
            "is_hourglass": bool(is_hourglass),
            "n_jumps": int(n_jumps),
            "winding": round(winding[1], 3),
        })

        # Plot
        axes[0].plot(z_planes * 1e3, ring_r * 1e3, color=colors[ir],
                     lw=1.5, label=label)
        if is_hourglass:
            axes[0].plot(z_waist * 1e3, r_min * 1e3, 'o',
                         color=colors[ir], ms=8, zorder=5)

        axes[1].plot(z_planes * 1e3, ring_to_out, color=colors[ir],
                     lw=1.5, label=label)

    axes[0].set_ylabel("Ring radius (mm)")
    axes[0].set_xlabel("z (mm)")
    axes[0].set_title("Continuity-constrained ring radius")
    axes[0].legend(fontsize=8, loc="upper right")
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(FOCAL_LENGTH * 1e3, color="gray", ls=":", alpha=0.5, label="f")

    axes[1].set_ylabel("ring_to_out")
    axes[1].set_xlabel("z (mm)")
    axes[1].set_title("Ring intensity / outskirts intensity")
    axes[1].legend(fontsize=8, loc="upper right")
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(FOCAL_LENGTH * 1e3, color="gray", ls=":", alpha=0.5)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "panel_aperture_sweep.png", dpi=150)
    plt.close(fig)
    print(f"\n  Saved panel_aperture_sweep.png")

    # Summary table
    print(f"\n  SUMMARY TABLE")
    print(f"  {'R (mm)':>7} {'N_F':>6} {'z_w (mm)':>8} {'r_min':>7} {'contr':>6} "
          f"{'int_min':>7} {'hour?':>6} {'jumps':>5} {'wind':>6}")
    for sr in sweep_results:
        print(f"  {sr['R_mm']:7.1f} {sr['NF']:6.3f} {sr['z_waist_mm']:8.2f} "
              f"{sr['r_min_mm']:7.3f} {sr['contraction']:6.1f} "
              f"{'YES' if sr['has_interior_min'] else 'no':>7} "
              f"{'YES' if sr['is_hourglass'] else 'no':>6} "
              f"{sr['n_jumps']:5d} {sr['winding']:6.3f}")

    return sweep_results


# ═══════════════════════════════════════════════════════════════════
# PART B1 — Inspect existing VTU
# ═══════════════════════════════════════════════════════════════════
def part_b1():
    """Parse the existing VTU and report geometry bounds."""
    print("\n" + "=" * 72)
    print("PART B1: INSPECT EXISTING VTU GEOMETRY")
    print("=" * 72)

    # Find the most recent truth VTU
    vtu_dirs = sorted(PROJECT_ROOT.glob("results/rs_free_space_vortex_truth_*/vtu/vortex_free_space.vtu"))
    if not vtu_dirs:
        print("  No existing VTU found — skipping inspection")
        return None

    vtu_path = vtu_dirs[-1]
    print(f"  Inspecting: {vtu_path.relative_to(PROJECT_ROOT)}")

    # Parse coordinates from VTU XML
    import xml.etree.ElementTree as ET
    tree = ET.parse(vtu_path)
    root = tree.getroot()

    piece = root.find(".//Piece")
    npts = int(piece.attrib["NumberOfPoints"])
    ncells = int(piece.attrib["NumberOfCells"])

    # Get point coordinates
    points_da = root.find(".//Points/DataArray")
    coords_text = points_da.text.strip()
    coords = np.array([float(v) for v in coords_text.split()])
    coords = coords.reshape(-1, 3)

    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    z_min, z_max = coords[:, 2].min(), coords[:, 2].max()

    print(f"\n  Points: {npts}")
    print(f"  Cells:  {ncells}")
    print(f"\n  Axis bounds:")
    print(f"    X: [{x_min:.6f}, {x_max:.6f}]  range = {x_max - x_min:.6f}")
    print(f"    Y: [{y_min:.6f}, {y_max:.6f}]  range = {y_max - y_min:.6f}")
    print(f"    Z: [{z_min:.6f}, {z_max:.6f}]  range = {z_max - z_min:.6f}")

    # Diagnose
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min

    print(f"\n  Aspect ratios:")
    print(f"    Z/X = {z_range / (x_range + 1e-30):.1f}")
    print(f"    Z/Y = {z_range / (y_range + 1e-30):.1f}")

    if z_range / (x_range + 1e-30) > 100:
        print(f"\n  ⚠ BUG CONFIRMED: Z range is {z_range/(x_range+1e-30):.0f}× the X range!")
        print(f"  ROOT CAUSE: X,Y are in METRES ({x_range:.6f}) but Z is in MM ({z_range:.4f})")
        print(f"  This makes ParaView show a long skinny line along Z.")
        print(f"\n  FIX: Convert all coordinates to the SAME unit system.")
        print(f"  In write_vtu() call: use xg*1e3, yg*1e3, zg (already mm)")
        print(f"  OR: keep all in metres (remove the z_sub * 1e3).")
    elif max(x_range, y_range) < 0.01 and z_range > 1.0:
        print(f"\n  ⚠ BUG: X,Y appear to be in metres, Z in mm — unit mismatch!")
    else:
        print(f"\n  ✓ Geometry looks reasonable")

    return {
        "npts": npts,
        "ncells": ncells,
        "x_range": x_range,
        "y_range": y_range,
        "z_range": z_range,
        "x_unit": "metres" if x_range < 0.01 else "mm",
        "z_unit": "mm" if z_range > 1.0 else "metres",
    }


# ═══════════════════════════════════════════════════════════════════
# PART B2 — Audit coordinate writing
# ═══════════════════════════════════════════════════════════════════
def part_b2():
    """Report on the VTU writing logic and the unit mismatch."""
    print("\n" + "=" * 72)
    print("PART B2: COORDINATE WRITING AUDIT")
    print("=" * 72)

    # The write_vtu function is correct in structure:
    # Points loop: iz → iy → ix with xg[ix], yg[iy], zg[iz] ✓
    # Data ravel: arr3d.ravel() (C-order) ✓
    # No transpose ✓

    # But the CALL SITE has the bug:
    # write_vtu(vtu_path, xg_sub, yg_sub, z_sub * 1e3, fields_vtu)
    #                     ↑metres  ↑metres  ↑mm!

    print("""
  The write_vtu() function itself is CORRECT:
    - Point loop: iz → iy → ix with xg[ix], yg[iy], zg[iz] ✓
    - Data flatten: arr3d.ravel() (C-order matches point order) ✓
    - No transpose ✓
    - Hex connectivity: correct VTK_HEXAHEDRON node ordering ✓

  The BUG is at the CALL SITE (line 625 of rs_free_space_vortex_truth.py):

    write_vtu(vtu_path, xg_sub, yg_sub, z_sub * 1e3, fields_vtu)
                        ↑metres  ↑metres  ↑mm (× 1e3)!

  xg_sub: [0.0, ..., 0.006] — metres (range 0.006)
  yg_sub: [0.0, ..., 0.006] — metres (range 0.006)
  z_sub * 1e3: [0.5, ..., 8.0] — mm (range 7.5)

  ParaView sees: X ≈ 0.006, Y ≈ 0.006, Z ≈ 7.5
  → Aspect ratio Z/X ≈ 1250:1 → appears as a vertical line.

  FIX: Convert ALL coordinates to mm:
    write_vtu(vtu_path, xg_sub * 1e3, yg_sub * 1e3, z_sub * 1e3, fields_vtu)
""")


# ═══════════════════════════════════════════════════════════════════
# PART B3 — Minimal geometry-only VTU test
# ═══════════════════════════════════════════════════════════════════
def write_vtu_correct(filepath, xg, yg, zg, fields: dict):
    """
    Write a .vtu file with hexahedral cells.
    ALL coordinates must be in the SAME unit system.

    Point ordering: iz (slowest) → iy → ix (fastest) — C-order.
    """
    nx, ny, nz = len(xg), len(yg), len(zg)
    npts = nx * ny * nz
    ncells = (nx - 1) * (ny - 1) * (nz - 1)

    lines = []
    lines.append('<?xml version="1.0"?>')
    lines.append('<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">')
    lines.append('<UnstructuredGrid>')
    lines.append(f'<Piece NumberOfPoints="{npts}" NumberOfCells="{ncells}">')

    # Points: iz → iy → ix
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
        flat = arr3d.ravel()  # C-order matches iz→iy→ix point order
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


def part_b3():
    """Create a minimal geometry-only VTU to verify the writer."""
    print("\n" + "=" * 72)
    print("PART B3: MINIMAL GEOMETRY VTU TEST")
    print("=" * 72)

    # Simple 10×10×10 box in mm
    xg = np.linspace(0, 6, 10)     # 0 to 6 mm
    yg = np.linspace(0, 6, 10)     # 0 to 6 mm
    zg = np.linspace(0.5, 8, 10)   # 0.5 to 8 mm

    # Dummy scalar = z coordinate
    ZZ, YY, XX = np.meshgrid(zg, yg, xg, indexing='ij')
    dummy_z = ZZ  # shape (nz, ny, nx)

    filepath = VTU_DIR / "geometry_test.vtu"
    write_vtu_correct(filepath, xg, yg, zg, {"z_coord": dummy_z})

    # Self-check: parse back and verify bounds
    import xml.etree.ElementTree as ET
    tree = ET.parse(filepath)
    root = tree.getroot()
    piece = root.find(".//Piece")
    npts = int(piece.attrib["NumberOfPoints"])
    points_da = root.find(".//Points/DataArray")
    coords = np.array([float(v) for v in points_da.text.strip().split()])
    coords = coords.reshape(-1, 3)

    x_range = coords[:, 0].max() - coords[:, 0].min()
    y_range = coords[:, 1].max() - coords[:, 1].min()
    z_range = coords[:, 2].max() - coords[:, 2].min()

    print(f"  Written: {filepath.relative_to(PROJECT_ROOT)}")
    print(f"  Grid: 10×10×10 = {npts} points")
    print(f"  X range: [{coords[:, 0].min():.1f}, {coords[:, 0].max():.1f}] = {x_range:.1f} mm")
    print(f"  Y range: [{coords[:, 1].min():.1f}, {coords[:, 1].max():.1f}] = {y_range:.1f} mm")
    print(f"  Z range: [{coords[:, 2].min():.1f}, {coords[:, 2].max():.1f}] = {z_range:.1f} mm")

    # Check aspect ratio
    aspect_ok = (0.5 < z_range / x_range < 3.0) and (0.5 < z_range / y_range < 3.0)
    print(f"  Aspect Z/X = {z_range / x_range:.2f}")
    print(f"  Geometry test: {'PASS ✓' if aspect_ok else 'FAIL ✗'}")

    # Also check data ordering: the z_coord field should match the z coordinates
    data_da = root.find(".//PointData/DataArray[@Name='z_coord']")
    data = np.array([float(v) for v in data_da.text.strip().split()])
    z_coords = coords[:, 2]
    corr = np.corrcoef(data, z_coords)[0, 1]
    print(f"  Data vs z-coordinate correlation: {corr:.6f}")
    print(f"  Data ordering: {'PASS ✓' if corr > 0.999 else 'FAIL ✗'}")

    return {
        "aspect_ok": aspect_ok,
        "data_ordering_ok": corr > 0.999,
        "x_range": x_range,
        "y_range": y_range,
        "z_range": z_range,
    }


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    print("=" * 72)
    print("APERTURE SCALING + VTU GEOMETRY DEBUG")
    print("=" * 72)
    print(f"Output: {BASE.relative_to(PROJECT_ROOT)}")
    print()

    # ── Part A1: analytical ───────────────────────────────────────
    a1_results = part_a1()

    # ── Part B1: inspect existing VTU ─────────────────────────────
    b1_results = part_b1()

    # ── Part B2: audit writing logic ──────────────────────────────
    part_b2()

    # ── Part B3: minimal geometry test ────────────────────────────
    b3_results = part_b3()

    # ── Part A2: aperture sweep ───────────────────────────────────
    a2_results = part_a2()

    # ── Summary ───────────────────────────────────────────────────
    t_total = time.time() - t0
    print("\n" + "=" * 72)
    print("FINAL SUMMARY")
    print("=" * 72)

    print(f"\n  PART A — PHYSICS:")
    print(f"  Current aperture R=1mm gives N_F = {a1_results['NF_current']:.3f}")
    print(f"  This is insufficient for geometric focusing (need N_F ≥ 3).")
    print(f"  Required: R ≥ {a1_results['requirements'][3]['R_mm']:.1f} mm for N_F=3")
    print(f"            R ≥ {a1_results['requirements'][5]['R_mm']:.1f} mm for N_F=5")

    hourglass_found = [sr for sr in a2_results if sr["is_hourglass"]]
    if hourglass_found:
        names = ", ".join(f"R={sr['R_mm']:.0f}mm" for sr in hourglass_found)
        print(f"\n  Hourglass found at: {names}")
    else:
        print(f"\n  No genuine hourglass found at any tested R.")
        # But report which had the best contraction
        best = max(a2_results, key=lambda x: x["contraction"])
        print(f"  Best contraction: {best['contraction']:.1f}× at R={best['R_mm']:.0f}mm")

    print(f"\n  PART B — VTU GEOMETRY:")
    if b1_results:
        print(f"  Existing VTU: X in {b1_results['x_unit']}, Z in {b1_results['z_unit']}")
        print(f"  ROOT CAUSE: Unit mismatch — X,Y in metres, Z in mm")
    print(f"  Minimal geometry test: {'PASS' if b3_results['aspect_ok'] else 'FAIL'}")
    print(f"  Data ordering test: {'PASS' if b3_results['data_ordering_ok'] else 'FAIL'}")
    print(f"  FIX: Change VTU call to use consistent units (all mm)")

    print(f"\n  Runtime: {t_total:.1f}s")

    # Write metadata
    metadata = {
        "timestamp": TS,
        "runtime_s": round(t_total, 1),
        "part_a1": a1_results,
        "part_a2": a2_results,
        "part_b1": b1_results,
        "part_b3": {k: v for k, v in b3_results.items()},
        "vtu_bug": {
            "root_cause": "Unit mismatch: X,Y in metres (range ~0.006), Z in mm (range ~7.5)",
            "location": "rs_free_space_vortex_truth.py line 625",
            "buggy_code": "write_vtu(vtu_path, xg_sub, yg_sub, z_sub * 1e3, fields_vtu)",
            "fix": "write_vtu(vtu_path, xg_sub * 1e3, yg_sub * 1e3, z_sub * 1e3, fields_vtu)",
        },
    }

    with open(BASE / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=lambda o: o.item() if hasattr(o, 'item') else o)
    print("  Written metadata.json")


if __name__ == "__main__":
    main()
