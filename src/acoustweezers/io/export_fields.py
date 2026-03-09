"""
Field export utilities for FEniCSx pressure solutions.

Exports complex pressure fields to XDMF + HDF5 (ParaView-friendly)
as separate real-valued datasets: p_real, p_imag, p_mag, p_phase.

Usage::

    from acoustweezers.io.export_fields import export_pressure_fields
    export_pressure_fields(solution, Path("results/run_001/fields/combined"))

The output directory will contain::

    mesh.xdmf + mesh.h5
    p_real.xdmf + p_real.h5
    p_imag.xdmf + p_imag.h5
    p_mag.xdmf + p_mag.h5
    p_phase.xdmf + p_phase.h5
"""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import Optional

from mpi4py import MPI


def export_pressure_fields(
    sol,
    export_dir: str | Path,
    *,
    verbose: bool = True,
) -> Path:
    """
    Export all pressure-derived fields from a PressureSolution to XDMF.

    Parameters
    ----------
    sol : PressureSolution
        Solution object with ``.p_function``, ``.domain``, ``.V``, ``.cfg``.
    export_dir : str or Path
        Output directory (created if needed).
    verbose : bool
        Print progress.

    Returns
    -------
    Path
        The export directory.
    """
    from dolfinx import fem
    from dolfinx.io import XDMFFile

    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    domain = sol.domain
    V = sol.V
    p_complex = sol.p_function.x.array.copy()  # complex128

    # dolfinx 0.9.0 XDMFFile.write_function requires the function degree to
    # equal the mesh geometry degree (1 for linear tets from create_box).
    # We therefore export at P1 (vertex values only) for XDMF / ParaView.
    # The full P2 DOF accuracy is preserved in the .npz DOF-scatter cache.
    V_real_p2 = fem.functionspace(domain, ("Lagrange", 2))  # source: P2
    V_real_p1 = fem.functionspace(domain, ("Lagrange", 1))  # target: P1 (XDMF)

    def _write_field(name: str, values: np.ndarray):
        """Write a real-valued P1 field to XDMF + HDF5 (ParaView-friendly).

        The values array comes from P2 DOF extraction.  We build a P2
        Function and interpolate it down to P1 so dolfinx can write it
        to XDMF (which requires function degree == mesh-geometry degree).
        """
        # Build P2 function with the DOF values
        func_p2 = fem.Function(V_real_p2)
        func_p2.name = name
        func_p2.x.array[:] = values.astype(np.float64)

        # Interpolate to P1 for XDMF compatibility
        func_p1 = fem.Function(V_real_p1)
        func_p1.name = name
        func_p1.interpolate(func_p2)

        xdmf_path = export_dir / f"{name}.xdmf"
        with XDMFFile(domain.comm, str(xdmf_path), "w") as xf:
            xf.write_mesh(domain)
            xf.write_function(func_p1)

        if verbose:
            print(f"    Wrote {xdmf_path}  (P1 interpolation for XDMF)")

    # Mesh-only XDMF
    mesh_path = export_dir / "mesh.xdmf"
    with XDMFFile(domain.comm, str(mesh_path), "w") as xf:
        xf.write_mesh(domain)
    if verbose:
        print(f"    Wrote {mesh_path}")

    # Derived arrays
    p_real = np.real(p_complex)
    p_imag = np.imag(p_complex)
    p_mag = np.abs(p_complex)
    p_phase = np.angle(p_complex)

    _write_field("p_real", p_real)
    _write_field("p_imag", p_imag)
    _write_field("p_mag", p_mag)
    _write_field("p_phase", p_phase)

    # Write a small manifest
    manifest = {
        "fields": ["p_real", "p_imag", "p_mag", "p_phase"],
        "mesh": "mesh.xdmf",
        "dofs_p2": int(V.dofmap.index_map.size_global * V.dofmap.index_map_bs),
        "element_order_source": 2,
        "element_order_xdmf": 1,
        "notes": "XDMF fields are P1 interpolations (for ParaView). "
                 "Full P2 accuracy is preserved in the .npz DOF-scatter cache. "
                 "Reload path: mesh.xdmf + .npz DOF arrays -> P2 Function -> eval().",
    }
    with open(export_dir / "fields_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    if verbose:
        print(f"    Field export complete → {export_dir}")

    return export_dir


def export_pressure_fields_from_arrays(
    domain,
    V,
    p_complex: np.ndarray,
    export_dir: str | Path,
    *,
    verbose: bool = True,
) -> Path:
    """
    Export fields when you only have the mesh, function space, and raw array.

    Parameters
    ----------
    domain : dolfinx.mesh.Mesh
    V : dolfinx.fem.FunctionSpace
        The complex P2 function space the solution was computed on.
    p_complex : np.ndarray
        Complex pressure DOF values.
    export_dir : str or Path

    Returns
    -------
    Path
    """
    from dolfinx import fem
    from dolfinx.io import XDMFFile

    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    V_real = fem.functionspace(domain, ("Lagrange", 2))

    def _write_field(name: str, values: np.ndarray):
        func = fem.Function(V_real)
        func.name = name
        func.x.array[:] = values.astype(np.float64)
        xdmf_path = export_dir / f"{name}.xdmf"
        with XDMFFile(domain.comm, str(xdmf_path), "w") as xf:
            xf.write_mesh(domain)
            xf.write_function(func)
        if verbose:
            print(f"    Wrote {xdmf_path}")

    mesh_path = export_dir / "mesh.xdmf"
    with XDMFFile(domain.comm, str(mesh_path), "w") as xf:
        xf.write_mesh(domain)

    _write_field("p_real", np.real(p_complex))
    _write_field("p_imag", np.imag(p_complex))
    _write_field("p_mag", np.abs(p_complex))
    _write_field("p_phase", np.angle(p_complex))

    return export_dir
