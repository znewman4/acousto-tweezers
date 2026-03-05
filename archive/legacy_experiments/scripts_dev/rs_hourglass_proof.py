#!/usr/bin/env python3
"""
Hourglass Proof — Visual evidence of vortex beam convergence + divergence
==========================================================================

Using the CORRECTED sign convention (φ_focus = −k(√(r²+f²)−f)):
  R = 5 mm, f = 4 mm, ℓ = 2  →  N_F = 8.42

Produces:
  1. XY cross-section panel: symmetric z-slices around the waist
  2. XZ meridional cross-section showing convergence → waist → divergence
  3. Corrected VTU file (all coordinates in mm)

Output → results/rs_hourglass_proof_<TS>/
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
from matplotlib.gridspec import GridSpec

from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import uniform_filter1d

# ── project root ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from acoustweezers.physics.acoustics.angular_spectrum import propagate_pressure_asm

# ═══════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════
WATER_C = 1484.0
F_HZ = 2.0e6
LAM = WATER_C / F_HZ           # 0.742 mm
K_WATER = 2.0 * np.pi * F_HZ / WATER_C

F_LENS = 4.0e-3                # focal length 4 mm
ELL = 2                        # topological charge

# Use R=5mm (N_F=8.42) for clearest hourglass
R_APT = 5.0e-3
NF = R_APT**2 / (LAM * F_LENS)

# Domain
LX = 22.0e-3                   # 4*R + 2mm margin
CX = LX / 2.0

# Grids
DN_DRIVE = 512                  # source grid
NXY_OUT = 256                   # output xy grid
NZ_FINE = 200                   # z-planes for high-res propagation

# z range: 0.3mm to 7mm — captures full convergence + divergence
Z_MIN = 0.3e-3
Z_MAX = 7.0e-3

TS = datetime.now().strftime("%Y%m%d_%H%M%S")
BASE = PROJECT_ROOT / "results" / f"rs_hourglass_proof_{TS}"
FIG_DIR = BASE / "figures"
VTU_DIR = BASE / "vtu"
FIG_DIR.mkdir(parents=True, exist_ok=True)
VTU_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
# Build source
# ═══════════════════════════════════════════════════════════════════
def build_drive():
    """Build converging vortex drive with CORRECT sign convention."""
    xg = np.linspace(0, LX, DN_DRIVE, endpoint=False)
    yg = np.linspace(0, LX, DN_DRIVE, endpoint=False)
    XX, YY = np.meshgrid(xg, yg)

    rx = XX - CX
    ry = YY - CX
    r = np.sqrt(rx**2 + ry**2)
    theta = np.arctan2(ry, rx)

    phi_vortex = ELL * theta
    # NEGATIVE sign = converging spherical wavefront
    phi_focus = -K_WATER * (np.sqrt(r**2 + F_LENS**2) - F_LENS)
    phi_target = phi_vortex + phi_focus

    # Cosine taper apodization
    r_norm = r / R_APT
    A = np.where(r_norm <= 1.0, 0.5 * (1.0 + np.cos(np.pi * r_norm)), 0.0)
    D = A * np.exp(1j * phi_target)

    return xg, yg, D


def propagate_single_z(D, xg_d, z, xg_out, yg_out, pad_factor=2):
    """Propagate to a single z-plane with interpolation to output grid."""
    dx = float(xg_d[1] - xg_d[0])
    p_prop = propagate_pressure_asm(D, dx, dx, K_WATER, z, pad_factor=pad_factor)

    YY_out, XX_out = np.meshgrid(yg_out, xg_out, indexing='ij')
    pts = np.column_stack([YY_out.ravel(), XX_out.ravel()])

    ire = RegularGridInterpolator((xg_d, xg_d), np.real(p_prop),
                                  method="linear", bounds_error=False, fill_value=0.0)
    iim = RegularGridInterpolator((xg_d, xg_d), np.imag(p_prop),
                                  method="linear", bounds_error=False, fill_value=0.0)
    p_out = (ire(pts) + 1j * iim(pts)).reshape(len(yg_out), len(xg_out))
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


def write_vtu(filepath, xg, yg, zg, fields: dict):
    """
    Write .vtu with hexahedral cells.
    ALL coordinates must be in the SAME units (mm).
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


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    t0_total = time.time()
    print("=" * 72)
    print("HOURGLASS PROOF — R=5mm, f=4mm, ℓ=2, N_F=8.42")
    print("=" * 72)
    print(f"Output: {BASE.relative_to(PROJECT_ROOT)}")
    print(f"λ = {LAM*1e3:.4f} mm,  k = {K_WATER:.1f} rad/m")
    print(f"R = {R_APT*1e3:.1f} mm,  f = {F_LENS*1e3:.1f} mm,  N_F = {NF:.2f}")
    print(f"Sign convention: φ_focus = −k(√(r²+f²)−f)  [CONVERGING]")
    print()

    # ── Build source ──────────────────────────────────────────────
    print("Building source drive...")
    xg_d, yg_d, D = build_drive()
    print(f"  Drive grid: {DN_DRIVE}×{DN_DRIVE}, domain = {LX*1e3:.1f} mm")

    # ── Output grid (xy) ────────────────────────────────────────
    # Wide grid for tracking + VTU, narrow grid for zoomed images
    xy_half_wide = min(R_APT * 2, LX / 2 - 1e-3)  # wide enough to capture the full ring
    xg_wide = np.linspace(CX - xy_half_wide, CX + xy_half_wide, NXY_OUT)
    yg_wide = np.linspace(CX - xy_half_wide, CX + xy_half_wide, NXY_OUT)

    # Zoomed grid for nice images (±2mm around centre)
    xy_half_zoom = 2.5e-3
    nxy_zoom = 200
    xg_zoom = np.linspace(CX - xy_half_zoom, CX + xy_half_zoom, nxy_zoom)
    yg_zoom = np.linspace(CX - xy_half_zoom, CX + xy_half_zoom, nxy_zoom)

    # ── Propagate to dense z-planes ──────────────────────────────
    z_planes = np.linspace(Z_MIN, Z_MAX, NZ_FINE)
    print(f"Propagating to {NZ_FINE} z-planes ({Z_MIN*1e3:.1f} → {Z_MAX*1e3:.1f} mm)...")

    p3d_wide = np.zeros((NZ_FINE, NXY_OUT, NXY_OUT), dtype=complex)
    p3d_zoom = np.zeros((NZ_FINE, nxy_zoom, nxy_zoom), dtype=complex)
    ring_peak_intensity = np.zeros(NZ_FINE)
    ring_peak_radius = np.zeros(NZ_FINE)

    t0 = time.time()
    for iz, zz in enumerate(z_planes):
        # Wide-field propagation for ring tracking + VTU
        p_wide = propagate_single_z(D, xg_d, zz, xg_wide, yg_wide, pad_factor=2)
        p3d_wide[iz] = p_wide
        I_wide = np.abs(p_wide)**2

        # Zoomed propagation for sharp XY images (interpolated from wide)
        p_zoom = propagate_single_z(D, xg_d, zz, xg_zoom, yg_zoom, pad_factor=2)
        p3d_zoom[iz] = p_zoom

        # Ring peak from radial profile on wide grid
        cx_out = CX
        cy_out = CX
        r_mid, I_prof = radial_profile(I_wide, xg_wide, yg_wide, cx_out, cy_out,
                                       r_max=xy_half_wide * 0.9, n_bins=250)
        search_mask = (r_mid >= 0.05e-3) & (r_mid <= xy_half_wide * 0.7)
        if np.any(search_mask):
            idx_s = np.where(search_mask)[0]
            i_pk = idx_s[np.argmax(I_prof[idx_s])]
            ring_peak_intensity[iz] = I_prof[i_pk]
            ring_peak_radius[iz] = r_mid[i_pk]

        if (iz + 1) % 50 == 0:
            print(f"  ... {iz+1}/{NZ_FINE} planes done")

    dt = time.time() - t0
    print(f"  Propagation complete in {dt:.1f}s")

    # ── Find the waist (intensity-peak z) ─────────────────────────
    iz_waist = np.argmax(ring_peak_intensity)
    z_waist = z_planes[iz_waist]
    r_waist = ring_peak_radius[iz_waist]

    print(f"\n  WAIST LOCATION:")
    print(f"    z_waist = {z_waist*1e3:.2f} mm  (geometric focus f = {F_LENS*1e3:.1f} mm)")
    print(f"    ring radius at waist = {r_waist*1e3:.3f} mm")
    print(f"    peak intensity at waist = {ring_peak_intensity[iz_waist]:.4e}")

    # Also find ring radius at first and last z for comparison
    r_max_all = ring_peak_radius.max()
    iz_rmax = np.argmax(ring_peak_radius)
    z_rmax = z_planes[iz_rmax]
    print(f"    ring radius at z_min ({Z_MIN*1e3:.1f}mm) = {ring_peak_radius[0]*1e3:.3f} mm")
    print(f"    ring radius at z_max ({Z_MAX*1e3:.1f}mm) = {ring_peak_radius[-1]*1e3:.3f} mm")
    print(f"    MAX ring radius = {r_max_all*1e3:.3f} mm at z = {z_rmax*1e3:.2f} mm")
    contraction = r_max_all / (r_waist + 1e-30)
    intensity_ratio = ring_peak_intensity[iz_waist] / (ring_peak_intensity[-1] + 1e-30)
    print(f"    ring contraction (max → waist): {contraction:.2f}×")
    print(f"    intensity concentration (waist / exit): {intensity_ratio:.1f}×")

    # ══════════════════════════════════════════════════════════════
    # FIGURE 1: XY cross-section panel, symmetric around waist
    # ══════════════════════════════════════════════════════════════
    print("\nGenerating XY cross-section panel...")

    # Choose z-slices: symmetric offsets around waist
    # Use offsets that clearly show convergence and divergence
    # Go from well before the waist to well after it
    dz_max = min(z_waist - Z_MIN, Z_MAX - z_waist) * 0.95  # stay within bounds
    offsets = np.array([-dz_max, -dz_max * 0.67, -dz_max * 0.33,
                        0.0,
                        dz_max * 0.33, dz_max * 0.67, dz_max])
    z_slices = z_waist + offsets
    n_slices = len(z_slices)
    n_slices_per_side = 3  # index of waist in z_slices

    # Find nearest z-plane indices
    iz_slices = [np.argmin(np.abs(z_planes - zz)) for zz in z_slices]

    # Compute intensity fields from zoomed grid
    I_slices = [np.abs(p3d_zoom[iz])**2 for iz in iz_slices]

    # Global color scale (from all slices)
    vmax = max(I.max() for I in I_slices)

    fig, axes = plt.subplots(2, n_slices, figsize=(3.2 * n_slices, 7.5),
                             gridspec_kw={"height_ratios": [1, 0.05]})

    # Convert to mm for plotting (relative to centre)
    extent_mm = [(xg_zoom[0] - CX) * 1e3, (xg_zoom[-1] - CX) * 1e3,
                 (yg_zoom[0] - CX) * 1e3, (yg_zoom[-1] - CX) * 1e3]

    for i in range(n_slices):
        ax = axes[0, i]
        z_actual = z_planes[iz_slices[i]]
        im = ax.imshow(I_slices[i], extent=extent_mm, origin='lower',
                       cmap='inferno', vmin=0, vmax=vmax, aspect='equal')

        # Ring radius circle
        r_ring_mm = ring_peak_radius[iz_slices[i]] * 1e3
        circle = plt.Circle((0, 0), r_ring_mm, fill=False, ec='cyan',
                             lw=1.5, ls='--', alpha=0.8)
        ax.add_patch(circle)

        # Label
        is_waist = (i == n_slices_per_side)
        label = f"z = {z_actual*1e3:.2f} mm"
        if is_waist:
            label += "\n★ WAIST ★"
            ax.set_title(label, fontsize=10, fontweight='bold', color='red')
        else:
            ax.set_title(label, fontsize=10)

        ax.set_xlabel("x (mm)" if i == n_slices // 2 else "", fontsize=9)
        if i == 0:
            ax.set_ylabel("y (mm)", fontsize=9)
        else:
            ax.set_yticklabels([])
        ax.tick_params(labelsize=8)

        # Ring radius annotation
        ax.text(0.02, 0.02, f"r_ring={r_ring_mm:.2f}mm",
                transform=ax.transAxes, fontsize=7, color='cyan',
                va='bottom', ha='left',
                bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.6))

    # Colorbar spanning full width
    for i in range(n_slices):
        axes[1, i].axis('off')
    cbar_ax = fig.add_axes([0.08, 0.04, 0.84, 0.02])
    fig.colorbar(im, cax=cbar_ax, orientation='horizontal', label="|p|²")

    fig.suptitle(f"XY Cross-Sections — Converging Vortex Beam\n"
                 f"R={R_APT*1e3:.0f}mm, f={F_LENS*1e3:.0f}mm, ℓ={ELL}, "
                 f"N_F={NF:.2f}  |  waist at z={z_waist*1e3:.2f}mm",
                 fontsize=13, fontweight='bold', y=0.99)
    fig.tight_layout(rect=[0, 0.07, 1, 0.94])
    fig.savefig(FIG_DIR / "xy_cross_sections.png", dpi=180)
    plt.close(fig)
    print(f"  Saved xy_cross_sections.png")

    # ══════════════════════════════════════════════════════════════
    # FIGURE 2: XZ meridional cross-section (through beam centre)
    # ══════════════════════════════════════════════════════════════
    print("Generating XZ meridional cross-section...")

    # Extract the central y-row (y=CX) from each z-plane → I(x, z) using zoomed grid
    iy_center = nxy_zoom // 2
    I_xz = np.zeros((NZ_FINE, nxy_zoom))
    for iz in range(NZ_FINE):
        I_xz[iz, :] = np.abs(p3d_zoom[iz, iy_center, :])**2

    fig, ax = plt.subplots(figsize=(14, 5))

    x_mm = (xg_zoom - CX) * 1e3
    z_mm = z_planes * 1e3

    # Plot I(x,z) as 2D image
    im = ax.pcolormesh(z_mm, x_mm, I_xz.T, cmap='inferno', shading='auto')
    cb = fig.colorbar(im, ax=ax, label="|p|²", shrink=0.8)

    # Overlay ring radius envelope (positive and negative)
    r_ring_mm = ring_peak_radius * 1e3
    ax.plot(z_mm, r_ring_mm, 'c--', lw=2.0, label='ring radius (upper)')
    ax.plot(z_mm, -r_ring_mm, 'c--', lw=2.0, label='ring radius (lower)')

    # Mark waist
    ax.axvline(z_waist * 1e3, color='lime', ls=':', lw=2.0, alpha=0.8, label=f'waist z={z_waist*1e3:.2f}mm')
    ax.plot(z_waist * 1e3, r_waist * 1e3, 'v', color='lime', ms=12, zorder=10)
    ax.plot(z_waist * 1e3, -r_waist * 1e3, '^', color='lime', ms=12, zorder=10)

    # Mark focal length
    ax.axvline(F_LENS * 1e3, color='white', ls='--', lw=1.0, alpha=0.5, label=f'f={F_LENS*1e3:.0f}mm')

    ax.set_xlabel("z (mm)", fontsize=13)
    ax.set_ylabel("x (mm)", fontsize=13)
    ax.set_title(f"XZ Meridional Cross-Section — Hourglass Shape\n"
                 f"R={R_APT*1e3:.0f}mm, f={F_LENS*1e3:.0f}mm, ℓ={ELL}, N_F={NF:.2f}",
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')

    # Annotate contraction ratio
    ax.annotate(f"ring contracts {contraction:.1f}× to waist\nintensity peaks {intensity_ratio:.0f}× at waist",
                xy=(z_waist * 1e3, r_waist * 1e3 + 0.3),
                fontsize=10, color='lime', fontweight='bold',
                ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', fc='black', alpha=0.7))

    fig.tight_layout()
    fig.savefig(FIG_DIR / "xz_meridional.png", dpi=180)
    plt.close(fig)
    print(f"  Saved xz_meridional.png")

    # ══════════════════════════════════════════════════════════════
    # FIGURE 3: Ring peak intensity + radius vs z
    # ══════════════════════════════════════════════════════════════
    print("Generating intensity/radius vs z plot...")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    ax1.plot(z_mm, ring_peak_intensity / ring_peak_intensity.max(), 'b-', lw=2)
    ax1.axvline(z_waist * 1e3, color='red', ls='--', lw=1.5, label=f'waist z={z_waist*1e3:.2f}mm')
    ax1.axvline(F_LENS * 1e3, color='gray', ls=':', lw=1, label=f'f={F_LENS*1e3:.0f}mm')
    ax1.set_ylabel("Normalised ring peak intensity", fontsize=12)
    ax1.set_title("Ring metrics vs z  (peak = focus)", fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.plot(z_mm, ring_peak_radius * 1e3, 'b-', lw=2)
    ax2.axvline(z_waist * 1e3, color='red', ls='--', lw=1.5)
    ax2.axvline(F_LENS * 1e3, color='gray', ls=':', lw=1)
    ax2.set_xlabel("z (mm)", fontsize=12)
    ax2.set_ylabel("Ring peak radius (mm)", fontsize=12)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "ring_metrics_vs_z.png", dpi=150)
    plt.close(fig)
    print(f"  Saved ring_metrics_vs_z.png")

    # ══════════════════════════════════════════════════════════════
    # VTU file — correct units (all mm)
    # ══════════════════════════════════════════════════════════════
    print("\nWriting VTU file (all coordinates in mm)...")

    # Subsample for manageable VTU size
    stride_xy = max(1, NXY_OUT // 60)  # ~60 pts per axis → ~60×60×80 = ~290k pts
    stride_z = max(1, NZ_FINE // 80)   # ~80 z-planes

    xg_vtu = xg_wide[::stride_xy] * 1e3   # mm
    yg_vtu = yg_wide[::stride_xy] * 1e3   # mm
    zg_vtu = z_planes[::stride_z] * 1e3   # mm

    p_sub = p3d_wide[::stride_z, ::stride_xy, ::stride_xy]

    fields_vtu = {
        "p_abs": np.abs(p_sub),
        "p_abs2": np.abs(p_sub)**2,
        "p_real": np.real(p_sub),
        "p_imag": np.imag(p_sub),
        "phase": np.angle(p_sub),
    }

    nx_v, ny_v, nz_v = len(xg_vtu), len(yg_vtu), len(zg_vtu)
    npts_vtu = nx_v * ny_v * nz_v
    print(f"  VTU grid: {nx_v}×{ny_v}×{nz_v} = {npts_vtu} points")
    print(f"  X range: [{xg_vtu[0]:.2f}, {xg_vtu[-1]:.2f}] mm")
    print(f"  Y range: [{yg_vtu[0]:.2f}, {yg_vtu[-1]:.2f}] mm")
    print(f"  Z range: [{zg_vtu[0]:.2f}, {zg_vtu[-1]:.2f}] mm")
    x_range = xg_vtu[-1] - xg_vtu[0]
    y_range = yg_vtu[-1] - yg_vtu[0]
    z_range = zg_vtu[-1] - zg_vtu[0]
    aspect = max(x_range, y_range, z_range) / min(x_range, y_range, z_range)
    print(f"  Aspect ratio: {aspect:.1f}  (should be < 5 for 3D rendering)")

    vtu_path = VTU_DIR / "hourglass_proof.vtu"
    write_vtu(vtu_path, xg_vtu, yg_vtu, zg_vtu, fields_vtu)
    print(f"  Written: {vtu_path.relative_to(PROJECT_ROOT)}")

    # Verify by reading back
    import xml.etree.ElementTree as ET
    tree = ET.parse(str(vtu_path))
    root = tree.getroot()
    piece = root.find(".//Piece")
    npts_check = int(piece.attrib["NumberOfPoints"])
    coords_text = root.find(".//Points/DataArray").text.strip().split()
    coords = np.array([float(v) for v in coords_text]).reshape(-1, 3)
    x_r = coords[:, 0].max() - coords[:, 0].min()
    y_r = coords[:, 1].max() - coords[:, 1].min()
    z_r = coords[:, 2].max() - coords[:, 2].min()
    print(f"\n  VTU SELF-CHECK:")
    print(f"    Points: {npts_check} (expected {npts_vtu})")
    print(f"    X range: {x_r:.2f} mm")
    print(f"    Y range: {y_r:.2f} mm")
    print(f"    Z range: {z_r:.2f} mm")
    aspect_check = max(x_r, y_r, z_r) / min(x_r, y_r, z_r)
    print(f"    Aspect ratio: {aspect_check:.1f}")
    vtu_ok = npts_check == npts_vtu and aspect_check < 5.0
    print(f"    VTU GEOMETRY: {'PASS ✓' if vtu_ok else 'FAIL ✗'}")

    # ── Summary ───────────────────────────────────────────────────
    t_total = time.time() - t0_total
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"  Parameters: R={R_APT*1e3:.0f}mm, f={F_LENS*1e3:.0f}mm, ℓ={ELL}, N_F={NF:.2f}")
    print(f"  Sign: φ_focus = −k(√(r²+f²)−f)  [CONVERGING]")
    print()
    print(f"  WAIST:")
    print(f"    z_waist = {z_waist*1e3:.2f} mm  (target f = {F_LENS*1e3:.1f} mm)")
    print(f"    ring radius at waist = {r_waist*1e3:.3f} mm")
    print(f"    MAX ring radius = {r_max_all*1e3:.3f} mm at z = {z_rmax*1e3:.2f} mm")
    print(f"    ring contraction (max → waist): {contraction:.2f}×")
    print(f"    intensity concentration (waist/exit): {intensity_ratio:.1f}×")
    print()
    print(f"  OUTPUTS:")
    print(f"    {FIG_DIR.relative_to(PROJECT_ROOT)}/xy_cross_sections.png")
    print(f"    {FIG_DIR.relative_to(PROJECT_ROOT)}/xz_meridional.png")
    print(f"    {FIG_DIR.relative_to(PROJECT_ROOT)}/ring_metrics_vs_z.png")
    print(f"    {VTU_DIR.relative_to(PROJECT_ROOT)}/hourglass_proof.vtu")
    print(f"  VTU geometry: {'PASS ✓' if vtu_ok else 'FAIL ✗'}")
    print(f"  Runtime: {t_total:.1f}s")

    # Save metadata
    metadata = {
        "timestamp": TS,
        "runtime_s": round(t_total, 1),
        "parameters": {
            "R_mm": R_APT * 1e3,
            "f_mm": F_LENS * 1e3,
            "ell": ELL,
            "NF": round(NF, 3),
            "lambda_mm": round(LAM * 1e3, 4),
            "sign_convention": "phi_focus = -k(sqrt(r^2+f^2) - f)  [CONVERGING]",
        },
        "waist": {
            "z_waist_mm": round(z_waist * 1e3, 2),
            "r_ring_at_waist_mm": round(r_waist * 1e3, 3),
            "r_ring_at_zmin_mm": round(ring_peak_radius[0] * 1e3, 3),
            "r_ring_max_mm": round(r_max_all * 1e3, 3),
            "z_rmax_mm": round(z_rmax * 1e3, 2),
            "ring_contraction": round(float(contraction), 2),
            "intensity_concentration": round(float(intensity_ratio), 1),
        },),
        },
        "vtu": {
            "file": "vtu/hourglass_proof.vtu",
            "all_units_mm": True,
            "aspect_ratio": round(float(aspect_check), 1),
            "geometry_pass": bool(vtu_ok),
        },
    }
    with open(BASE / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata.json")


if __name__ == "__main__":
    main()
