#!/usr/bin/env python3
"""
RS Plastic Lens Hourglass Demo — Stage 1
==========================================

Demonstrates that a plastic-lens vortex drive (ℓ = 2, cosine taper,
focal length set to z*) produces a true hourglass when propagated
via the canonical ASM module — no FEM needed.

Outputs
-------
results/rs_lens_hourglass_stage1_<TS>/
    figures/panel_xy_slices.png     — XY |p| at ~9 z-planes
    figures/panel_xz_hourglass.png  — XZ slice through y = centre
    vtu/vortex_pressure_3d.vtu      — 3-D VTU for ParaView
    metadata.json                   — all parameters
    report.md                       — "what vortex is this?" write-up

Usage
-----
  python scripts/dev/rs_plastic_lens_hourglass_demo.py [--timestamp TS]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── project root ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from acoustweezers.physics.acoustics.angular_spectrum import (
    propagate_pressure_asm,
)
from acoustweezers.physics.acoustics.vortex_lens import (
    PlasticLensConfig,
    create_plastic_lens_drive,
)
from acoustweezers.experiments.farfield_petri_cuboid.presets import (
    CORRECTED_PRESET,
)


# ═══════════════════════════════════════════════════════════════════
# Physical constants (from canonical preset)
# ═══════════════════════════════════════════════════════════════════
WATER_C   = 1484.0          # m/s
WATER_RHO = 997.0           # kg/m³
F_HZ      = 2.0e6           # Hz (canonical)
LAM       = WATER_C / F_HZ  # 0.000742 m
K_WATER   = 2.0 * np.pi * F_HZ / WATER_C
OMEGA     = 2.0 * np.pi * F_HZ

# Domain
LX = float(CORRECTED_PRESET["Lx"])        # 6 mm
LY = float(CORRECTED_PRESET["Ly"])        # 6 mm
H_UNDER = float(CORRECTED_PRESET["H_under"])  # 3 mm
H_TOP   = float(CORRECTED_PRESET["H_top"])    # 2.0085 mm

# Trap plane z* = mid-petri + λ/4  (same definition used everywhere)
Z_STAR = H_UNDER + H_TOP / 2.0 + 0.25 * LAM

# Disk
DISK_R = float(CORRECTED_PRESET["disk_radius"])  # 1 mm

# Lens parameters for this demo
LENS_ELL          = 2             # ℓ = 2 (as specified)
# Fresnel number N_F = R²/(λ·f) determines whether an hourglass forms.
# For R=1mm, λ=0.742mm:
#   f=2.0mm → N_F≈0.67  (NO hourglass — ring expands monotonically)
#   f=0.3mm → N_F≈4.49  (CLEAR hourglass — 5× ring contraction at waist)
# A shorter geometric f pushes the actual waist (large focal shift)
# up toward z* thanks to the N_F ≲ 1 focal-shift formula.
LENS_FOCAL_LENGTH_DEFAULT = 0.30e-3   # 0.30 mm — tuned for hourglass at z*
LENS_APODIZATION  = "cosine_taper"
LENS_C_LENS       = float(CORRECTED_PRESET.get("lens_c_lens", 2700.0))

# Grid sizes
DRIVE_N = 512     # source-plane FFT grid (power of 2)
OUT_NXY = 200     # output XY resolution
OUT_NZ  = 120     # output z-planes for 3-D (fine enough for waist)

# z-scan range: full propagation window from near-source to well beyond z*
Z_MIN = 1.5e-3                        # well into far-field
Z_MAX = 7.0e-3                        # beyond z* to see ring re-expand

# Number of XY-slice planes for the panel figure
N_PANEL_PLANES = 9


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="Plastic lens hourglass Stage 1")
    p.add_argument("--timestamp", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    p.add_argument("--out-nxy", type=int, default=OUT_NXY)
    p.add_argument("--out-nz", type=int, default=OUT_NZ)
    p.add_argument("--drive-n", type=int, default=DRIVE_N)
    p.add_argument("--focal-mm", type=float, default=LENS_FOCAL_LENGTH_DEFAULT * 1e3,
                   help="Lens geometric focal length in mm (default: 2.0)")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════
# VTU export helper (structured grid → unstructured VTK)
# ═══════════════════════════════════════════════════════════════════
def _write_vtu(filepath: Path, xg, yg, zg, fields: dict):
    """
    Write a rectilinear 3-D field to VTU (unstructured hexahedral XML).

    Parameters
    ----------
    filepath : Path
        Output .vtu path.
    xg, yg, zg : 1-D arrays
        Grid coordinates.
    fields : dict[str, ndarray(nz, ny, nx)]
        Scalar fields to include as PointData.
    """
    nx, ny, nz = len(xg), len(yg), len(zg)
    npts = nx * ny * nz
    ncells = (nx - 1) * (ny - 1) * (nz - 1)

    # Build point coordinates (Fortran-order: x fastest)
    coords = np.empty((npts, 3), dtype=np.float64)
    idx = 0
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                coords[idx] = (xg[ix], yg[iy], zg[iz])
                idx += 1

    def _pt(ix, iy, iz):
        return iz * ny * nx + iy * nx + ix

    # Connectivity (VTK_HEXAHEDRON = 12)
    conn = np.empty((ncells, 8), dtype=np.int64)
    offsets = np.empty(ncells, dtype=np.int64)
    ci = 0
    for iz in range(nz - 1):
        for iy in range(ny - 1):
            for ix in range(nx - 1):
                conn[ci] = [
                    _pt(ix, iy, iz), _pt(ix + 1, iy, iz),
                    _pt(ix + 1, iy + 1, iz), _pt(ix, iy + 1, iz),
                    _pt(ix, iy, iz + 1), _pt(ix + 1, iy, iz + 1),
                    _pt(ix + 1, iy + 1, iz + 1), _pt(ix, iy + 1, iz + 1),
                ]
                offsets[ci] = (ci + 1) * 8
                ci += 1

    # Build XML
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">',
        '<UnstructuredGrid>',
        f'<Piece NumberOfPoints="{npts}" NumberOfCells="{ncells}">',
        '<Points>',
        f'<DataArray type="Float64" NumberOfComponents="3" format="ascii">',
    ]
    # Write coords in chunks to avoid enormous single lines
    for i in range(npts):
        lines.append(f"{coords[i, 0]:.8e} {coords[i, 1]:.8e} {coords[i, 2]:.8e}")
    lines += ['</DataArray>', '</Points>']

    # Cells
    lines.append('<Cells>')
    lines.append('<DataArray type="Int64" Name="connectivity" format="ascii">')
    for ci in range(ncells):
        lines.append(" ".join(str(v) for v in conn[ci]))
    lines.append('</DataArray>')
    lines.append('<DataArray type="Int64" Name="offsets" format="ascii">')
    lines.append(" ".join(str(v) for v in offsets))
    lines.append('</DataArray>')
    lines.append('<DataArray type="UInt8" Name="types" format="ascii">')
    lines.append(" ".join(["12"] * ncells))
    lines.append('</DataArray>')
    lines.append('</Cells>')

    # Point data
    lines.append('<PointData>')
    for name, arr3d in fields.items():
        # arr3d is (nz, ny, nx); point loop is iz→iy→ix (C-order for that shape)
        flat = arr3d.ravel()  # C-order matches iz*ny*nx + iy*nx + ix
        lines.append(f'<DataArray type="Float64" Name="{name}" format="ascii">')
        for v in flat:
            lines.append(f"{v:.8e}")
        lines.append('</DataArray>')
    lines += ['</PointData>', '</Piece>', '</UnstructuredGrid>', '</VTKFile>']

    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text("\n".join(lines))


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    args = parse_args()
    TS = args.timestamp
    NXY = args.out_nxy
    NZ = args.out_nz
    DN = args.drive_n
    LENS_FOCAL_LENGTH = args.focal_mm * 1e-3   # convert mm → m

    BASE = PROJECT_ROOT / "results" / f"rs_lens_hourglass_stage1_{TS}"
    FIG_DIR = BASE / "figures"
    VTU_DIR = BASE / "vtu"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    VTU_DIR.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    print("=" * 72)
    print("RS PLASTIC LENS HOURGLASS DEMO — Stage 1")
    print("=" * 72)
    print(f"Output       : {BASE.relative_to(PROJECT_ROOT)}")
    print(f"f            : {F_HZ / 1e6:.1f} MHz")
    print(f"λ            : {LAM * 1e3:.4f} mm")
    print(f"k            : {K_WATER:.1f} rad/m")
    print(f"Domain       : {LX * 1e3:.0f} × {LY * 1e3:.0f} mm")
    print(f"Disk R       : {DISK_R * 1e3:.1f} mm")
    print(f"Lens ℓ       : {LENS_ELL}")
    print(f"Focal length : {LENS_FOCAL_LENGTH * 1e3:.3f} mm  (geometric)")
    NF_header = DISK_R ** 2 / (LAM * LENS_FOCAL_LENGTH)
    print(f"Fresnel N_F  : {NF_header:.2f}  (R²/λf)")
    print(f"z*           : {Z_STAR * 1e3:.3f} mm")
    print(f"z range      : [{Z_MIN * 1e3:.3f}, {Z_MAX * 1e3:.3f}] mm")
    print(f"Drive grid   : {DN}×{DN}")
    print(f"Output grid  : {NXY}×{NXY}×{NZ}")
    print()

    # ──────────────────────────────────────────────────────────────
    # 1. Build plastic lens drive on source grid
    # ──────────────────────────────────────────────────────────────
    print("Building plastic lens drive pattern...")
    CX, CY = LX / 2, LY / 2
    xg_d = np.linspace(0, LX, DN, endpoint=False)
    yg_d = np.linspace(0, LY, DN, endpoint=False)
    dx_d = float(xg_d[1] - xg_d[0])
    dy_d = float(yg_d[1] - yg_d[0])
    XX_d, YY_d = np.meshgrid(xg_d, yg_d)

    lens_cfg = PlasticLensConfig(
        topological_charge=LENS_ELL,
        focal_length=LENS_FOCAL_LENGTH,
        focus_offset_x=0.0,           # on-axis for hourglass demo
        focus_offset_y=0.0,
        c_lens=LENS_C_LENS,
        c_water=WATER_C,
        frequency_hz=F_HZ,
        aperture_radius=DISK_R,
        center=None,
        apodization=LENS_APODIZATION,
        apodization_strength=1.0,
    )

    D = create_plastic_lens_drive(
        XX_d.ravel(), YY_d.ravel(), lens_cfg,
        center_x=CX, center_y=CY, verbose=True,
    ).reshape(XX_d.shape)

    print(f"  Drive max |D| = {np.abs(D).max():.6f}")
    print()

    # ──────────────────────────────────────────────────────────────
    # 2. Output grids
    # ──────────────────────────────────────────────────────────────
    xg = np.linspace(0, LX, NXY)
    yg = np.linspace(0, LY, NXY)
    zg = np.linspace(Z_MIN, Z_MAX, NZ)
    dx_out = float(xg[1] - xg[0])
    dy_out = float(yg[1] - yg[0])

    # z-planes for the XY panel figure (9 evenly spaced)
    z_panel = np.linspace(Z_MIN, Z_MAX, N_PANEL_PLANES)

    # ──────────────────────────────────────────────────────────────
    # 3. Propagate to all z-planes (3-D volume + panel slices)
    # ──────────────────────────────────────────────────────────────
    print("Propagating to 3-D volume...")
    from scipy.interpolate import RegularGridInterpolator

    # We'll propagate on the drive grid and interpolate to the output grid
    vol_amp = np.zeros((NZ, NXY, NXY), dtype=np.float64)
    vol_re = np.zeros_like(vol_amp)
    vol_im = np.zeros_like(vol_amp)

    XX_out, YY_out = np.meshgrid(xg, yg)
    pts_out = np.column_stack([YY_out.ravel(), XX_out.ravel()])

    t0 = time.time()
    for iz, zz in enumerate(zg):
        p_drive = propagate_pressure_asm(D, dx_d, dy_d, K_WATER, zz,
                                         pad_factor=2, include_evanescent=True)
        # Interpolate to output grid
        ire = RegularGridInterpolator(
            (yg_d, xg_d), np.real(p_drive),
            method="linear", bounds_error=False, fill_value=0.0,
        )
        iim = RegularGridInterpolator(
            (yg_d, xg_d), np.imag(p_drive),
            method="linear", bounds_error=False, fill_value=0.0,
        )
        p_out = (ire(pts_out) + 1j * iim(pts_out)).reshape(NXY, NXY)
        vol_re[iz] = np.real(p_out)
        vol_im[iz] = np.imag(p_out)
        vol_amp[iz] = np.abs(p_out)

        if (iz + 1) % 20 == 0 or iz == 0 or iz == NZ - 1:
            print(f"  z[{iz:3d}] = {zz * 1e3:6.3f} mm  "
                  f"max|p| = {vol_amp[iz].max():.6f}  "
                  f"({time.time() - t0:.1f}s)")
    t_prop = time.time() - t0
    print(f"  3-D propagation done in {t_prop:.1f}s")
    print()

    # ──────────────────────────────────────────────────────────────
    # 4. Waist analysis: ring radius vs z
    # ──────────────────────────────────────────────────────────────
    # Strategy: at each z-plane, compute the azimuthal-average radial
    # profile and find the global maximum (dominant ring radius).
    # The "waist" is the interior minimum of ring-radius-vs-z where
    # the inner vortex ring becomes the dominant peak, contracting
    # inward then re-expanding — the classic hourglass shape.
    print("Computing ring radius vs z (waist analysis)...")
    NF_FRESNEL = DISK_R ** 2 / (LAM * LENS_FOCAL_LENGTH)
    print(f"  Fresnel number N_F = R²/(λf) = {NF_FRESNEL:.2f}")

    RR_out = np.sqrt((XX_out - CX) ** 2 + (YY_out - CY) ** 2)
    N_RBINS = 150
    r_bins = np.linspace(0, 2.5 * LAM, N_RBINS + 1)
    r_mid = 0.5 * (r_bins[:-1] + r_bins[1:])

    ring_radii = np.full(NZ, np.nan)
    ring_peak_amp = np.full(NZ, np.nan)
    rad_profiles = np.zeros((NZ, N_RBINS))

    for iz in range(NZ):
        amp_plane = vol_amp[iz]
        rad_mean = np.zeros(N_RBINS)
        for ib in range(N_RBINS):
            sel = (RR_out >= r_bins[ib]) & (RR_out < r_bins[ib + 1])
            if sel.any():
                rad_mean[ib] = np.mean(amp_plane[sel])
        rad_profiles[iz] = rad_mean

        if rad_mean.max() > 0:
            peak_idx = np.argmax(rad_mean)
            ring_radii[iz] = r_mid[peak_idx]
            ring_peak_amp[iz] = rad_mean[peak_idx]

    valid = ~np.isnan(ring_radii)
    if valid.any():
        rr_valid = ring_radii[valid]
        zg_valid = zg[valid]
        # Look for interior minimum
        interior_min_idx = None
        for ii in range(1, len(rr_valid) - 1):
            if rr_valid[ii] <= rr_valid[ii - 1] and rr_valid[ii] <= rr_valid[ii + 1]:
                if interior_min_idx is None or rr_valid[ii] < rr_valid[interior_min_idx]:
                    interior_min_idx = ii
        if interior_min_idx is not None:
            z_waist = zg_valid[interior_min_idx]
            r_waist = rr_valid[interior_min_idx]
            waist_tag = "interior"
        else:
            waist_idx = np.argmin(rr_valid)
            z_waist = zg_valid[waist_idx]
            r_waist = rr_valid[waist_idx]
            waist_tag = "edge"
        delta_z = z_waist - Z_STAR

        # Contraction ratio
        r_max_val = np.max(rr_valid)
        contraction = r_max_val / r_waist if r_waist > 0 else np.nan

        print(f"  Waist ring radius : {r_waist * 1e3:.4f} mm at z = {z_waist * 1e3:.3f} mm  [{waist_tag}]")
        print(f"  z* target         : {Z_STAR * 1e3:.3f} mm")
        print(f"  Focal shift       : {delta_z * 1e3:.3f} mm = {delta_z / LAM:.3f}λ")
        print(f"  Max ring radius   : {r_max_val * 1e3:.4f} mm")
        print(f"  Contraction ratio : {contraction:.2f}×")
    else:
        z_waist = np.nan
        r_waist = np.nan
        delta_z = np.nan
        contraction = np.nan
        waist_tag = "none"
        print("  WARNING: could not determine waist")
    print()

    # ──────────────────────────────────────────────────────────────
    # 5. Figures
    # ──────────────────────────────────────────────────────────────
    print("Generating figures...")
    extent_xy = [xg[0] * 1e3, xg[-1] * 1e3, yg[0] * 1e3, yg[-1] * 1e3]

    # ── Panel A: XY slices at N_PANEL_PLANES z values ─────────────
    ncols = 3
    nrows = (N_PANEL_PLANES + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.8 * nrows))
    fig.suptitle(
        f"XY |p| slices — Plastic Lens ℓ={LENS_ELL}, "
        f"f={LENS_FOCAL_LENGTH * 1e3:.2f} mm",
        fontsize=14, fontweight="bold",
    )
    axes_flat = axes.ravel()

    # Find z-indices closest to each panel z
    vmax_global = vol_amp.max()
    for ip, zp in enumerate(z_panel):
        iz = int(np.argmin(np.abs(zg - zp)))
        ax = axes_flat[ip]
        im = ax.imshow(
            vol_amp[iz], extent=extent_xy, origin="lower",
            cmap="magma", vmin=0, vmax=vmax_global,
        )
        dz_lam = (zg[iz] - Z_STAR) / LAM
        ax.set_title(f"z = {zg[iz] * 1e3:.3f} mm  (Δz = {dz_lam:+.2f}λ)",
                      fontsize=10)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        fig.colorbar(im, ax=ax, shrink=0.75)

    # Hide unused axes
    for ip in range(len(z_panel), len(axes_flat)):
        axes_flat[ip].set_visible(False)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG_DIR / "panel_xy_slices.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved panel_xy_slices.png")

    # ── Panel B: XZ hourglass (y = centre) ────────────────────────
    iy_cen = NXY // 2
    xz_amp = vol_amp[:, iy_cen, :]  # (nz, nx)

    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5),
                              gridspec_kw={"width_ratios": [1.2, 1, 1]})
    fig.suptitle(
        f"XZ Hourglass — ℓ={LENS_ELL}, "
        f"f={LENS_FOCAL_LENGTH * 1e3:.2f} mm, "
        f"N_F={NF_FRESNEL:.2f}",
        fontsize=14, fontweight="bold",
    )

    extent_xz = [xg[0] * 1e3, xg[-1] * 1e3, zg[0] * 1e3, zg[-1] * 1e3]
    im = axes[0].imshow(
        xz_amp, extent=extent_xz, origin="lower", cmap="magma",
        aspect="auto",
    )
    axes[0].axhline(Z_STAR * 1e3, color="cyan", ls="--", lw=1.2,
                     alpha=0.8, label=f"z* = {Z_STAR * 1e3:.3f} mm")
    if not np.isnan(z_waist):
        axes[0].axhline(z_waist * 1e3, color="lime", ls=":", lw=1.2,
                         alpha=0.8, label=f"waist = {z_waist * 1e3:.3f} mm")
    axes[0].set_xlabel("x (mm)")
    axes[0].set_ylabel("z (mm)")
    axes[0].set_title("XZ |p| (y = centre)")
    axes[0].legend(fontsize=9, loc="upper right")
    fig.colorbar(im, ax=axes[0], shrink=0.8)

    # Ring radius vs z
    axes[1].plot(ring_radii[valid] * 1e3, zg[valid] * 1e3, "b-o", ms=2, lw=1.2)
    axes[1].axhline(Z_STAR * 1e3, color="cyan", ls="--", lw=1,
                     alpha=0.7, label="z*")
    if not np.isnan(z_waist):
        axes[1].axhline(z_waist * 1e3, color="lime", ls=":", lw=1,
                         alpha=0.7, label=f"waist {z_waist * 1e3:.2f} mm")
        axes[1].axvline(r_waist * 1e3, color="red", ls=":", lw=0.8,
                         alpha=0.5, label=f"r_min={r_waist * 1e3:.3f} mm")
    axes[1].set_xlabel("Ring radius (mm)")
    axes[1].set_ylabel("z (mm)")
    axes[1].set_title("Vortex ring radius vs z")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # Radial profile waterfall at selected z-planes
    n_profiles = 8
    cmap_prof = plt.cm.viridis
    z_prof_idx = np.linspace(0, NZ - 1, n_profiles, dtype=int)
    for ii, iz in enumerate(z_prof_idx):
        color = cmap_prof(ii / (n_profiles - 1))
        label = f"z={zg[iz]*1e3:.2f}"
        axes[2].plot(r_mid * 1e3, rad_profiles[iz] / max(rad_profiles[iz].max(), 1e-30),
                     color=color, lw=1.0, label=label)
    axes[2].set_xlabel("r (mm)")
    axes[2].set_ylabel("Normalised |p|")
    axes[2].set_title("Radial profiles")
    axes[2].legend(fontsize=6, ncol=2)
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG_DIR / "panel_xz_hourglass.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved panel_xz_hourglass.png")

    # ──────────────────────────────────────────────────────────────
    # 6. VTU export (reduced grid for manageable file size)
    # ──────────────────────────────────────────────────────────────
    print("Exporting VTU...")
    # Subsample to keep VTU < ~50 MB
    vtu_stride_xy = max(1, NXY // 80)
    vtu_stride_z = max(1, NZ // 60)
    xg_v = xg[::vtu_stride_xy]
    yg_v = yg[::vtu_stride_xy]
    zg_v = zg[::vtu_stride_z]
    amp_v = vol_amp[::vtu_stride_z, ::vtu_stride_xy, ::vtu_stride_xy]
    re_v = vol_re[::vtu_stride_z, ::vtu_stride_xy, ::vtu_stride_xy]
    im_v = vol_im[::vtu_stride_z, ::vtu_stride_xy, ::vtu_stride_xy]

    _write_vtu(
        VTU_DIR / "vortex_pressure_3d.vtu",
        xg_v, yg_v, zg_v,
        {
            "pressure_amplitude": amp_v,
            "pressure_real": re_v,
            "pressure_imag": im_v,
            "pressure_intensity": amp_v ** 2,
        },
    )
    n_vtu_pts = len(xg_v) * len(yg_v) * len(zg_v)
    print(f"  VTU grid: {len(xg_v)}×{len(yg_v)}×{len(zg_v)} = {n_vtu_pts} pts")
    print(f"  Saved vortex_pressure_3d.vtu")
    print()

    # ──────────────────────────────────────────────────────────────
    # 7. metadata.json
    # ──────────────────────────────────────────────────────────────
    metadata = {
        "timestamp": TS,
        "description": "Stage 1 plastic lens hourglass demo (vortex-only, ASM)",
        "physics": {
            "frequency_hz": F_HZ,
            "c_water": WATER_C,
            "rho_water": WATER_RHO,
            "wavelength_m": LAM,
            "k_water": K_WATER,
        },
        "lens": {
            "type": "PlasticLensConfig",
            "topological_charge": LENS_ELL,
            "focal_length_m": LENS_FOCAL_LENGTH,
            "aperture_radius_m": DISK_R,
            "apodization": LENS_APODIZATION,
            "c_lens": LENS_C_LENS,
            "focus_offset_x": 0.0,
            "focus_offset_y": 0.0,
        },
        "domain": {
            "Lx": LX, "Ly": LY,
            "z_star": Z_STAR,
            "z_min": Z_MIN, "z_max": Z_MAX,
        },
        "grid": {
            "drive_n": DN, "drive_dx": dx_d, "drive_dy": dy_d,
            "output_nxy": NXY, "output_nz": NZ,
            "output_dx": dx_out, "output_dy": dy_out,
        },
        "asm": {
            "pad_factor": 2,
            "include_evanescent": True,
            "method": "pressure-mode ASM (RS-I kernel: H = exp(i kz z))",
        },
        "results": {
            "z_waist_found_m": float(z_waist) if not np.isnan(z_waist) else None,
            "r_waist_found_m": float(r_waist) if not np.isnan(r_waist) else None,
            "focal_shift_m": float(delta_z) if not np.isnan(delta_z) else None,
            "focal_shift_lambda": float(delta_z / LAM) if not np.isnan(delta_z) else None,
            "contraction_ratio": float(contraction) if not np.isnan(contraction) else None,
            "waist_type": waist_tag,
            "fresnel_number": float(NF_FRESNEL),
        },
    }
    meta_path = BASE / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))

    # ──────────────────────────────────────────────────────────────
    # 8. report.md
    # ──────────────────────────────────────────────────────────────
    waist_str = "NOT FOUND" if np.isnan(z_waist) else (
        f"{z_waist * 1e3:.3f} mm  (Δz = {delta_z / LAM:+.3f}λ from z*)"
    )
    contraction_str = f"{contraction:.1f}×" if not np.isnan(contraction) else "N/A"
    passfail = (
        "**PASS:** Waist is an interior minimum (true hourglass) with "
        f"{contraction_str} ring contraction."
        if waist_tag == "interior" else
        "**FAIL:** No interior waist found — ring radius is monotonic "
        "(Fresnel number too low for hourglass to form)."
    )
    report = f"""\
# RS Plastic Lens Hourglass — Stage 1 Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Run directory:** `{BASE.relative_to(PROJECT_ROOT)}`

---

## What vortex is this?

This is **not** a free-space Laguerre–Gaussian beam.  It is the field
produced by a **plastic spiral-phase lens** with integrated focusing,
modelled as a phase mask on a finite circular aperture, propagated via
the angular spectrum method (ASM).

### Source drive

The drive pattern on the disk at z = 0 is:

    D(x,y) = A(r) · exp(i · mod(φ_target, 2π))

where:

    φ_target(x,y) = ℓ·θ  +  k_water·(√(Δx² + Δy² + f²) − f)
                     ↑              ↑
              vortex phase    converging spherical phase (focus at f)

- **ℓ = {LENS_ELL}** (topological charge)
- **f = {LENS_FOCAL_LENGTH * 1e3:.3f} mm** (geometric focal length)
- **R = {DISK_R * 1e3:.1f} mm** (aperture radius)
- **Apodization:** cosine taper, A(r) = 0.5·(1 + cos(πr/R)) for r ≤ R

The 2π wrapping models a stepped (Fresnel) plastic lens fabricated via:

    t(x,y) = t₀ + mod(φ_target, 2π) / (k_lens − k_water)

with c_lens = {LENS_C_LENS} m/s.

### Propagation method

Angular Spectrum Method (pressure-mode, RS-I kernel):

    p(x,y,z) = IFFT[ FFT[D] · exp(i·kz·z) ]

    kz = √(k² − kx² − ky²)          (propagating)
    kz = i·√(kx² + ky² − k²)         (evanescent, exponential decay)

- 2× zero-padding in each dimension (wraparound suppression)
- All evanescent modes retained (correct decay)
- No paraxial / Fresnel approximation

**Implementation:** `src/acoustweezers/physics/acoustics/angular_spectrum.py`
→ `propagate_pressure_asm()`

---

## Parameters

| Parameter | Value |
|-----------|-------|
| Frequency | {F_HZ / 1e6:.1f} MHz |
| λ | {LAM * 1e3:.4f} mm |
| k | {K_WATER:.1f} rad/m |
| c_water | {WATER_C} m/s |
| ρ_water | {WATER_RHO} kg/m³ |
| Lens ℓ | {LENS_ELL} |
| Focal length f | {LENS_FOCAL_LENGTH * 1e3:.3f} mm |
| **Fresnel number N_F** | **{NF_FRESNEL:.2f}** |
| Aperture R | {DISK_R * 1e3:.1f} mm |
| Apodization | {LENS_APODIZATION} |
| c_lens | {LENS_C_LENS} m/s |
| z* (target) | {Z_STAR * 1e3:.3f} mm |
| z range | [{Z_MIN * 1e3:.3f}, {Z_MAX * 1e3:.3f}] mm |
| Drive grid | {DN}×{DN} (dx = {dx_d * 1e6:.1f} µm) |
| Output grid | {NXY}×{NXY}×{NZ} |

---

## Waist verification

**Method:** Global maximum of the azimuthally-averaged |p| radial
profile (the dominant vortex ring) is tracked across z-planes. The
waist is the z where this ring radius reaches an interior minimum.

| Metric | Value |
|--------|-------|
| z_waist found | {waist_str} |
| r_waist (ring radius at waist) | {r_waist * 1e3:.4f} mm = {r_waist / LAM:.2f}λ |
| z* (target) | {Z_STAR * 1e3:.3f} mm |
| Focal shift | {delta_z / LAM:+.3f}λ |
| Contraction ratio | {contraction_str} |
| Waist type | {waist_tag} |

{passfail}

---

## Fresnel Number Analysis

The Fresnel number **N_F = R² / (λ·f)** determines whether the
focused vortex produces a true hourglass:

| N_F | Behavior |
|-----|----------|
| < 1 | Diffraction-dominated; ring radius increases monotonically (no hourglass). The canonical preset f=2mm gives N_F≈0.67 — **no hourglass in free-space ASM**. |
| 1–2 | Transition; weak ring contraction but hard to resolve |
| > 2 | Clear hourglass; ring contracts to interior minimum then re-expands |
| > 4 | Strong hourglass; geometric focus is a good approximation |

This run uses **N_F = {NF_FRESNEL:.2f}** → {"clear hourglass expected" if NF_FRESNEL > 2 else "no hourglass expected"}.

**Note:** The canonical FEM preset uses f=2mm (N_F≈0.67). In the
full FEM model, the cavity standing-wave pattern and reflections
produce a tight vortex ring at z* — this is a **cavity effect**, not
free-space focusing. The ASM proof requires a higher N_F to
demonstrate the hourglass in free space.

---

## Deliverables

- `figures/panel_xy_slices.png` — XY |p| at {N_PANEL_PLANES} z-planes
- `figures/panel_xz_hourglass.png` — XZ slice + ring radius vs z + radial profiles
- `vtu/vortex_pressure_3d.vtu` — 3-D field for ParaView
- `metadata.json` — all parameters (machine-readable)
- `report.md` — this file
"""
    report_path = BASE / "report.md"
    report_path.write_text(report)

    # ──────────────────────────────────────────────────────────────
    # Summary
    # ──────────────────────────────────────────────────────────────
    t_total = time.time() - t_start
    print("=" * 72)
    print("STAGE 1 COMPLETE")
    print("=" * 72)
    print(f"  Fresnel number    : {NF_FRESNEL:.2f}")
    print(f"  Waist at z        : {z_waist * 1e3:.3f} mm  "
          f"(Δz = {delta_z / LAM:+.3f}λ from z*)  [{waist_tag}]")
    print(f"  Ring radius       : {r_waist * 1e3:.4f} mm = {r_waist / LAM:.2f}λ")
    print(f"  Contraction ratio : {contraction:.1f}×")
    print(f"  Total time        : {t_total:.1f}s")
    print(f"  Output            : {BASE.relative_to(PROJECT_ROOT)}")
    print()


if __name__ == "__main__":
    main()
