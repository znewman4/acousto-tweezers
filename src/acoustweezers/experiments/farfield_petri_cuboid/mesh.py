"""
Mesh generation and boundary / cell tagging for the far-field petri cuboid.

Creates a structured tetrahedral box mesh with:
  - Facet tags: bottom_disk, top_face, side walls (standing patches vs rest)
  - Cell  tags: PML regions (side-x, side-y, bottom-z) vs physical interior

PML geometry
------------
Side PML (x):  cells with  x < t_pml_xy  OR  x > Lx - t_pml_xy
Side PML (y):  cells with  y < t_pml_xy  OR  y > Ly - t_pml_xy
Bottom PML (z): cells with  z < t_pml_z  AND  r > disk_radius
  (a "column" directly above the disk is excluded so the source is in
   the physical domain)
Top: NO PML  (uses impedance Robin BC)

Author: Acousto-Tweezers Project
Date: 2026-02-16
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Dict
from mpi4py import MPI
from dolfinx import mesh, fem

from .config import FarFieldConfig

# ── facet tags ────────────────────────────────────────────────────────
TAG_BOTTOM_DISK = 1
TAG_TOP = 2
TAG_X0 = 3
TAG_XL = 4
TAG_Y0 = 5
TAG_YL = 6
TAG_BOTTOM_OUTSIDE = 7

# derived standing-patch tags (same as side, with subregion z filter)
TAG_STAND_X0 = 13
TAG_STAND_XL = 14
TAG_STAND_Y0 = 15
TAG_STAND_YL = 16

# ── cell (region) tags ────────────────────────────────────────────────
CELL_PHYSICAL = 0
CELL_PML_X = 1
CELL_PML_Y = 2
CELL_PML_Z = 3
CELL_PML_XY = 4     # corner overlap
CELL_PML_XZ = 5
CELL_PML_YZ = 6
CELL_PML_XYZ = 7


def create_mesh(cfg: FarFieldConfig, verbose: bool = True):
    """
    Build structured tet mesh with facet and cell tags.

    Returns
    -------
    domain : dolfinx.mesh.Mesh
    facet_tags : dolfinx.mesh.MeshTags
    cell_tags  : dolfinx.mesh.MeshTags
    tag_info   : dict   (maps for both facet and cell tags)
    """
    Lx, Ly = cfg.Lx, cfg.Ly
    H = cfg.H_total
    nx, ny, nz = cfg.mesh_nx, cfg.mesh_ny, cfg.mesh_nz

    comm = MPI.COMM_WORLD
    rank = comm.rank

    if verbose and rank == 0:
        print(f"\n{'='*70}")
        print("MESH GENERATION  (far-field petri cuboid)")
        print(f"{'='*70}")
        print(f"  Box: {Lx*1e3:.2f} × {Ly*1e3:.2f} × {H*1e3:.2f} mm")
        print(f"  Grid: {nx} × {ny} × {nz}  (tet)")
        print(f"  λ = {cfg.wavelength*1e3:.4f} mm   elem/λ = {cfg.elements_per_wavelength}")

    domain = mesh.create_box(
        comm,
        [[0.0, 0.0, 0.0], [Lx, Ly, H]],
        [nx, ny, nz],
        cell_type=mesh.CellType.tetrahedron,
    )

    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    domain.topology.create_connectivity(tdim, fdim)

    tol = min(Lx, Ly, H) * 1e-6
    R_disk = cfg.disk_radius
    cx, cy = cfg.disk_center_x, cfg.disk_center_y
    H_under = cfg.H_under

    # ── facet tagging ─────────────────────────────────────────────────
    def _bottom_disk(x):
        on = np.isclose(x[2], 0.0, atol=tol)
        r2 = (x[0] - cx)**2 + (x[1] - cy)**2
        return on & (r2 <= R_disk**2 + tol)

    def _bottom_outside(x):
        on = np.isclose(x[2], 0.0, atol=tol)
        r2 = (x[0] - cx)**2 + (x[1] - cy)**2
        return on & (r2 > R_disk**2 + tol)

    def _top(x):
        return np.isclose(x[2], H, atol=tol)

    def _x0(x):
        return np.isclose(x[0], 0.0, atol=tol)
    def _xL(x):
        return np.isclose(x[0], Lx, atol=tol)
    def _y0(x):
        return np.isclose(x[1], 0.0, atol=tol)
    def _yL(x):
        return np.isclose(x[1], Ly, atol=tol)

    # Standing-wave patches: side walls restricted to petri slab  z ∈ [H_under, H_under + H_top]
    H_top = cfg.H_top
    def _stand_x0(x):
        return np.isclose(x[0], 0.0, atol=tol) & (x[2] >= H_under - tol) & (x[2] <= H_under + H_top + tol)
    def _stand_xL(x):
        return np.isclose(x[0], Lx, atol=tol)  & (x[2] >= H_under - tol) & (x[2] <= H_under + H_top + tol)
    def _stand_y0(x):
        return np.isclose(x[1], 0.0, atol=tol) & (x[2] >= H_under - tol) & (x[2] <= H_under + H_top + tol)
    def _stand_yL(x):
        return np.isclose(x[1], Ly, atol=tol)  & (x[2] >= H_under - tol) & (x[2] <= H_under + H_top + tol)

    # Build (standing patches AFTER full side tags so they overwrite in petri slab)
    boundaries = [
        (TAG_BOTTOM_DISK, _bottom_disk),
        (TAG_BOTTOM_OUTSIDE, _bottom_outside),
        (TAG_TOP, _top),
        (TAG_X0, _x0),
        (TAG_XL, _xL),
        (TAG_Y0, _y0),
        (TAG_YL, _yL),
        (TAG_STAND_X0, _stand_x0),
        (TAG_STAND_XL, _stand_xL),
        (TAG_STAND_Y0, _stand_y0),
        (TAG_STAND_YL, _stand_yL),
    ]

    facet_indices_list, facet_markers_list = [], []
    for tag, loc in boundaries:
        f = mesh.locate_entities_boundary(domain, fdim, loc)
        facet_indices_list.append(f)
        facet_markers_list.append(np.full_like(f, tag))

    all_idx = np.hstack(facet_indices_list).astype(np.int32)
    all_mk = np.hstack(facet_markers_list).astype(np.int32)

    # de-dup: last assignment wins (standing patches overwrite generic side)
    _, keep = np.unique(all_idx[::-1], return_index=True)
    keep = len(all_idx) - 1 - keep          # un-reverse
    order = np.argsort(all_idx[keep])
    facet_tags = mesh.meshtags(domain, fdim, all_idx[keep][order], all_mk[keep][order])

    facet_tag_map = {
        TAG_BOTTOM_DISK:    f"bottom disk (R={R_disk*1e3:.2f} mm)",
        TAG_BOTTOM_OUTSIDE: "bottom outside disk",
        TAG_TOP:            "top face (impedance)",
        TAG_X0:             "x=0 (below petri)",
        TAG_XL:             f"x={Lx*1e3:.1f} (below petri)",
        TAG_Y0:             "y=0 (below petri)",
        TAG_YL:             f"y={Ly*1e3:.1f} (below petri)",
        TAG_STAND_X0:       "x=0 standing patch (petri slab)",
        TAG_STAND_XL:       f"x={Lx*1e3:.1f} standing patch",
        TAG_STAND_Y0:       "y=0 standing patch",
        TAG_STAND_YL:       f"y={Ly*1e3:.1f} standing patch",
    }

    # ── cell tagging (PML vs physical) ────────────────────────────────
    t_xy = cfg.t_pml_xy if cfg.pml_enabled else 0.0
    t_z  = cfg.t_pml_z  if cfg.pml_enabled else 0.0

    midpoints = mesh.compute_midpoints(domain, tdim, np.arange(
        domain.topology.index_map(tdim).size_local, dtype=np.int32))

    xm, ym, zm = midpoints[:, 0], midpoints[:, 1], midpoints[:, 2]

    in_pml_x = (xm < t_xy) | (xm > Lx - t_xy)
    in_pml_y = (ym < t_xy) | (ym > Ly - t_xy)

    # Bottom PML: only outside the disk column (r > R_disk)
    r2_m = (xm - cx)**2 + (ym - cy)**2
    in_pml_z = (zm < t_z) & (r2_m > R_disk**2)

    tags_arr = np.zeros(len(xm), dtype=np.int32)
    # Set composite tags (order matters: corners last)
    tags_arr[in_pml_x & ~in_pml_y & ~in_pml_z] = CELL_PML_X
    tags_arr[~in_pml_x & in_pml_y & ~in_pml_z] = CELL_PML_Y
    tags_arr[~in_pml_x & ~in_pml_y & in_pml_z] = CELL_PML_Z
    tags_arr[in_pml_x & in_pml_y & ~in_pml_z]  = CELL_PML_XY
    tags_arr[in_pml_x & ~in_pml_y & in_pml_z]  = CELL_PML_XZ
    tags_arr[~in_pml_x & in_pml_y & in_pml_z]  = CELL_PML_YZ
    tags_arr[in_pml_x & in_pml_y & in_pml_z]   = CELL_PML_XYZ

    cell_indices = np.arange(len(xm), dtype=np.int32)
    cell_tags = mesh.meshtags(domain, tdim, cell_indices, tags_arr)

    cell_tag_map = {
        CELL_PHYSICAL: "physical",
        CELL_PML_X: "PML-x",
        CELL_PML_Y: "PML-y",
        CELL_PML_Z: "PML-z",
        CELL_PML_XY: "PML-xy",
        CELL_PML_XZ: "PML-xz",
        CELL_PML_YZ: "PML-yz",
        CELL_PML_XYZ: "PML-xyz",
    }

    if verbose and rank == 0:
        n_cells = domain.topology.index_map(tdim).size_global
        n_verts = domain.topology.index_map(0).size_global
        print(f"  Cells: {n_cells},  Vertices: {n_verts}")
        print(f"\n  Facet tags:")
        for tag, name in facet_tag_map.items():
            c = np.sum(facet_tags.values == tag)
            print(f"    {tag:2d}: {name}  ({c} facets)")
        print(f"\n  Cell tags:")
        for tag, name in cell_tag_map.items():
            c = np.sum(cell_tags.values == tag)
            if c > 0:
                print(f"    {tag}: {name}  ({c} cells)")
        print(f"{'='*70}\n")

    tag_info = {"facet": facet_tag_map, "cell": cell_tag_map}
    return domain, facet_tags, cell_tags, tag_info
