#!/usr/bin/env python3
"""
Standing-wave 3D ParaView export.

Reads the FEM cache and writes a VTU point-cloud with pressure
magnitude, phase, real, and imaginary parts for ParaView visualisation.

Output
------
results/deliverables/standing_wave/standing_wave_3d.vtu
results/deliverables/standing_wave/PARAVIEW_README.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lib.fem_cache_utils import load_fem_cache

try:
    import meshio
except ImportError:
    sys.exit("meshio is required: pip install meshio")


OUT_DIR = PROJECT_ROOT / "results" / "deliverables" / "standing_wave"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    cache = load_fem_cache()
    coords = cache["coords"]   # (N, 3)
    p = cache["p"]             # (N,) complex

    print(f"Exporting {cache['n_dofs']:,} DOFs to VTU …")

    point_data = {
        "p_mag":   np.abs(p).astype(np.float32),
        "p_phase": np.angle(p).astype(np.float32),
        "p_real":  np.real(p).astype(np.float32),
        "p_imag":  np.imag(p).astype(np.float32),
    }

    # meshio point cloud (vertex cells)
    n = len(coords)
    cells = [("vertex", np.arange(n).reshape(-1, 1))]
    mesh = meshio.Mesh(coords, cells, point_data=point_data)

    vtu_path = OUT_DIR / "standing_wave_3d.vtu"
    mesh.write(str(vtu_path))
    print(f"  wrote {vtu_path}  ({vtu_path.stat().st_size / 1e6:.1f} MB)")

    # Write ParaView README
    readme = OUT_DIR / "PARAVIEW_README.md"
    readme.write_text(f"""\
# Standing-Wave 3D — ParaView Guide

**File:** `standing_wave_3d.vtu`
**DOFs:** {cache['n_dofs']:,}
**|p|_max:** {cache['p_max']:.2f} Pa
**Domain:** {cache['domain']['x_max']*1e3:.1f} × {cache['domain']['y_max']*1e3:.1f} × {cache['domain']['z_max']*1e3:.1f} mm

## Quick start

1. **File → Open** → `standing_wave_3d.vtu` → Apply
2. Representation: **Point Gaussian** (size ≈ 0.05 mm)
3. Color by **p_mag** → use *Cool to Warm* or *Viridis* colormap
4. Add **Clip** filter to view interior slices
5. Use **Slice** at z = {cache['z_star']*1e3:.2f} mm to see the trap plane

## Fields

| Name      | Description                |
|-----------|----------------------------|
| p_mag     | Pressure magnitude |p| [Pa] |
| p_phase   | Pressure phase arg(p) [rad] |
| p_real    | Re(p) [Pa]                 |
| p_imag    | Im(p) [Pa]                 |
""")
    print(f"  wrote {readme}")
    print("Done.")


if __name__ == "__main__":
    main()
