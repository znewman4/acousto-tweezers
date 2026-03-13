#!/usr/bin/env python3
"""
Phase 2.2D — Vortex 3D volumetric export for ParaView.

Computes a vortex+lens field on a regular 3D grid via ASM propagation
at many z-planes, then exports the volume as:
  1. VTI (structured grid) via meshio — native ParaView format
  2. Compressed NPZ as backup

Output  → results/deliverables/vortex/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# ── Project root & imports ─────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lib.fem_cache_utils import load_fem_cache, LAM, C_WATER, F_HZ
from scripts.lib.asm_utils import (
    make_grid_from_fem,
    make_vortex_field,
    make_lens_phase,
    propagate_asm,
    K0,
)

OUT = PROJECT_ROOT / "results" / "deliverables" / "vortex"
OUT.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════
# Parameters
# ══════════════════════════════════════════════════════════════════
CHARGE = 1
WAIST = 2.0e-3
R_AP = 2.5e-3
FOCAL_LENGTH = 5.0e-3

# Grid resolution — moderate for 3D (memory-conscious)
NX_3D, NY_3D = 200, 200
Z_MIN_3D, Z_MAX_3D = 1.0e-3, 9.0e-3
NZ_3D = 60


def main() -> None:
    print("=" * 60)
    print("Phase 2.2D — Vortex 3D ParaView export")
    print("=" * 60)

    # ── Load FEM cache for grid ────────────────────────────────────
    cache = load_fem_cache()
    grid = make_grid_from_fem(cache, nx=NX_3D, ny=NY_3D)
    x, y = grid["x"], grid["y"]
    XX, YY = grid["XX"], grid["YY"]
    dx, dy = grid["dx"], grid["dy"]
    z_arr = np.linspace(Z_MIN_3D, Z_MAX_3D, NZ_3D)
    dz = float(z_arr[1] - z_arr[0]) if NZ_3D > 1 else 1.0

    print(f"Grid: {NX_3D}×{NY_3D}×{NZ_3D}  "
          f"dx={dx*1e6:.1f} µm  dy={dy*1e6:.1f} µm  dz={dz*1e6:.0f} µm")
    print(f"z ∈ [{Z_MIN_3D*1e3:.1f}, {Z_MAX_3D*1e3:.1f}] mm")
    vol_mb = NX_3D * NY_3D * NZ_3D * 16 / 1e6  # complex128 = 16 bytes
    print(f"Volume: {NX_3D*NY_3D*NZ_3D/1e6:.1f} M voxels  "
          f"(~{vol_mb:.0f} MB complex128)")

    # ── Build source field ─────────────────────────────────────────
    vortex = make_vortex_field(XX, YY, charge=CHARGE, waist=WAIST, k=K0,
                               aperture_radius=R_AP)
    lens = make_lens_phase(XX, YY, focal_length=FOCAL_LENGTH,
                           aperture_radius=R_AP, family="ideal", k=K0,
                           charge=0)
    source = vortex * np.exp(-1j * lens)  # converging sign

    # ── Propagate to each z-plane ─────────────────────────────────
    vol_mag = np.zeros((NZ_3D, NY_3D, NX_3D), dtype=np.float32)
    vol_phase = np.zeros((NZ_3D, NY_3D, NX_3D), dtype=np.float32)
    vol_real = np.zeros((NZ_3D, NY_3D, NX_3D), dtype=np.float32)
    vol_imag = np.zeros((NZ_3D, NY_3D, NX_3D), dtype=np.float32)

    for iz, z_val in enumerate(z_arr):
        p = propagate_asm(source, dx, dy, wavelength=LAM, z=z_val)
        vol_mag[iz] = np.abs(p).astype(np.float32)
        vol_phase[iz] = np.angle(p).astype(np.float32)
        vol_real[iz] = np.real(p).astype(np.float32)
        vol_imag[iz] = np.imag(p).astype(np.float32)
        if iz % 10 == 0:
            print(f"  z = {z_val*1e3:5.1f} mm  ({iz+1}/{NZ_3D})  "
                  f"|p|_max = {vol_mag[iz].max():.4f}")

    print(f"\nVolume peak |p| = {vol_mag.max():.5f}")

    # ── Export VTI (ImageData) via meshio ─────────────────────────
    vti_path = OUT / "vortex_3d.vti"
    try:
        import meshio

        # meshio expects point data on a structured grid (Nz+1, Ny+1, Nx+1)
        # for cell-centred data, or (Nz, Ny, Nx) for point data on the cell grid.
        # We use the VTK ImageData writer directly for correctness.
        _write_vti_raw(
            vti_path, vol_mag, vol_phase, vol_real, vol_imag,
            origin=(x[0], y[0], z_arr[0]),
            spacing=(dx, dy, dz),
        )
        print(f"  → saved {vti_path.name}  "
              f"({vti_path.stat().st_size / 1e6:.1f} MB)")
    except Exception as exc:
        print(f"  VTI export failed ({exc}); falling back to NPZ only.")

    # ── Compressed NPZ backup ─────────────────────────────────────
    npz_path = OUT / "vortex_3d_volume.npz"
    np.savez_compressed(
        npz_path,
        p_mag=vol_mag,
        p_phase=vol_phase,
        p_real=vol_real,
        p_imag=vol_imag,
        x_mm=x * 1e3,
        y_mm=y * 1e3,
        z_mm=z_arr * 1e3,
        dx_m=dx, dy_m=dy, dz_m=dz,
    )
    print(f"  → saved {npz_path.name}  "
          f"({npz_path.stat().st_size / 1e6:.1f} MB)")

    # ── ParaView README ───────────────────────────────────────────
    readme_text = f"""\
# Vortex 3D — ParaView Guide

## Files

| File | Format | Description |
|------|--------|-------------|
| `vortex_3d.vti` | VTK ImageData | Structured 3D grid, ParaView-native |
| `vortex_3d_volume.npz` | NumPy compressed | Backup / post-processing |

## Grid

- **Dimensions:** {NX_3D} × {NY_3D} × {NZ_3D}
- **Spacing:** dx = {dx*1e6:.1f} µm, dy = {dy*1e6:.1f} µm, dz = {dz*1e6:.0f} µm
- **Origin:** ({x[0]*1e3:.3f}, {y[0]*1e3:.3f}, {Z_MIN_3D*1e3:.1f}) mm
- **Extent:** x ∈ [{x[0]*1e3:.3f}, {x[-1]*1e3:.3f}] mm, \
y ∈ [{y[0]*1e3:.3f}, {y[-1]*1e3:.3f}] mm, \
z ∈ [{Z_MIN_3D*1e3:.1f}, {Z_MAX_3D*1e3:.1f}] mm

## Physical parameters

- Topological charge ℓ = {CHARGE}
- Beam waist w = {WAIST*1e3:.1f} mm
- Aperture radius = {R_AP*1e3:.1f} mm
- Focal length = {FOCAL_LENGTH*1e3:.1f} mm
- Wavelength = {LAM*1e3:.3f} mm  (f = {F_HZ/1e6:.1f} MHz, c = {C_WATER:.0f} m/s)

## Fields

| Name     | Type    | Description                  |
|----------|---------|------------------------------|
| p_mag    | float32 | Pressure magnitude |p| [a.u.]|
| p_phase  | float32 | Phase arg(p) [rad]           |
| p_real   | float32 | Re(p) [a.u.]                |
| p_imag   | float32 | Im(p) [a.u.]                |

## Quick start (VTI file)

1. **File → Open** → `vortex_3d.vti` → Apply
2. Representation: **Volume** or **Slice**
3. Color by **p_mag** → Viridis or Inferno colormap
4. Add **Slice** filter at z = {FOCAL_LENGTH*1e3:.1f} mm to see the focal ring
5. Add **Contour** filter on p_mag to visualise the hourglass iso-surface
6. **Clip** filter to expose the interior structure

## Quick start (NPZ file — Python)

```python
import numpy as np
d = np.load("vortex_3d_volume.npz")
p_mag = d["p_mag"]          # shape ({NZ_3D}, {NY_3D}, {NX_3D})
x_mm, y_mm, z_mm = d["x_mm"], d["y_mm"], d["z_mm"]
```
"""
    (OUT / "PARAVIEW_README.md").write_text(readme_text)
    print("  → saved PARAVIEW_README.md")

    # ── Summary JSON ──────────────────────────────────────────────
    summary = {
        "charge": CHARGE,
        "waist_mm": WAIST * 1e3,
        "aperture_radius_mm": R_AP * 1e3,
        "focal_length_mm": FOCAL_LENGTH * 1e3,
        "wavelength_mm": LAM * 1e3,
        "grid": {"nx": NX_3D, "ny": NY_3D, "nz": NZ_3D},
        "spacing_um": {"dx": dx * 1e6, "dy": dy * 1e6, "dz": dz * 1e6},
        "z_range_mm": [Z_MIN_3D * 1e3, Z_MAX_3D * 1e3],
        "peak_magnitude": float(vol_mag.max()),
        "files": ["vortex_3d.vti", "vortex_3d_volume.npz", "PARAVIEW_README.md"],
    }
    with open(OUT / "vortex_3d_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("  → saved vortex_3d_summary.json")

    print(f"\n{'='*60}")
    print(f"Done. Outputs → {OUT.relative_to(PROJECT_ROOT)}/")
    print(f"{'='*60}")


# ══════════════════════════════════════════════════════════════════
# VTI writer (pure XML + binary, no meshio dependency for ImageData)
# ══════════════════════════════════════════════════════════════════

def _write_vti_raw(
    path: Path,
    mag: np.ndarray,
    phase: np.ndarray,
    real: np.ndarray,
    imag: np.ndarray,
    origin: tuple,
    spacing: tuple,
) -> None:
    """
    Write a VTK ImageData (.vti) file with appended binary data.

    VTI is the standard structured-grid format for ParaView.
    Data is stored as appended raw binary (base64-encoded).
    """
    import base64
    import struct

    nz, ny, nx = mag.shape  # VTK ordering: z slowest
    # VTK expects WholeExtent as (x_min, x_max, y_min, y_max, z_min, z_max)
    extent = f"0 {nx-1} 0 {ny-1} 0 {nz-1}"
    ox, oy, oz = origin
    sx, sy, sz = spacing

    arrays = [
        ("p_mag", mag),
        ("p_phase", phase),
        ("p_real", real),
        ("p_imag", imag),
    ]

    # Build appended data buffer
    offsets = []
    raw_chunks = []
    offset = 0
    for name, arr in arrays:
        data = arr.astype(np.float32).ravel(order="F")  # Fortran order for VTK
        raw = data.tobytes()
        header = struct.pack("<I", len(raw))  # 4-byte little-endian size header
        offsets.append(offset)
        raw_chunks.append(header + raw)
        offset += len(header) + len(raw)

    # Concatenate all chunks
    appended_raw = b"".join(raw_chunks)
    appended_b64 = base64.b64encode(appended_raw).decode("ascii")

    # Build XML
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian" '
        'header_type="UInt32">',
        f'  <ImageData WholeExtent="{extent}" '
        f'Origin="{ox} {oy} {oz}" Spacing="{sx} {sy} {sz}">',
        f'    <Piece Extent="{extent}">',
        '      <PointData>',
    ]
    for i, (name, _) in enumerate(arrays):
        lines.append(
            f'        <DataArray type="Float32" Name="{name}" '
            f'format="appended" offset="{offsets[i]}"/>'
        )
    lines += [
        '      </PointData>',
        '    </Piece>',
        '  </ImageData>',
        '  <AppendedData encoding="base64">',
        f'   _{appended_b64}',
        '  </AppendedData>',
        '</VTKFile>',
    ]

    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
