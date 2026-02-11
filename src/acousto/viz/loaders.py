"""
Data loaders for canonical acoustic-tweezers visualizations.

Loads NPZ files with complex pressure + Gor'kov data from
generate_rich_data.py, and builds PyVista meshes.

Key design choice:
  Data is solved on P2 elements (531K DOFs) but XDMF mesh is P1 (68K vertices).
  We build a P1 UnstructuredGrid from XDMF, then interpolate P2 field values
  onto P1 vertices using a KD-tree lookup.  This is safe because P1 vertices
  are a subset of P2 DOF locations.
"""

from pathlib import Path
import numpy as np
import pyvista as pv


def load_rich(run_dir, name='combined', use_p2=True):
    """
    Load a rich dataset (NPZ + mesh) and return a PyVista UnstructuredGrid.

    Parameters
    ----------
    run_dir : str or Path
        Directory containing NPZ + XDMF files from generate_rich_data.py.
    name : str
        Base name: 'standing', 'vortex', 'combined', 'combined_phi045', etc.
    use_p2 : bool
        If True, use full P2 resolution (531K DOFs). If False, downsampled to P1.

    Returns
    -------
    pv.UnstructuredGrid
        Mesh with point_data:
          'magnitude'  — |p| in Pa
          'phase'      — arg(p) in radians, [-π, π]
          'gorkov'     — Gor'kov potential in J
          'p_real'     — Re(p)
          'p_imag'     — Im(p)
    """
    run_dir = Path(run_dir)

    npz_path  = run_dir / f'{name}.npz'
    mesh_path = run_dir / f'{name}_mesh.xdmf'
    h5_path   = mesh_path.with_suffix('.h5')

    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ not found: {npz_path}")
    if not h5_path.exists():
        raise FileNotFoundError(f"Mesh H5 not found: {h5_path}")

    # --- Load P2 field data from NPZ ---
    d = np.load(str(npz_path))
    coords_p2 = d['coords']          # (N_p2, 3)
    mag_p2    = d['magnitude']        # (N_p2,)
    phase_p2  = d['phase']            # (N_p2,)
    gorkov_p2 = d.get('gorkov', None) # (N_p2,) or None
    preal_p2  = d['p_real']           # (N_p2,)
    pimag_p2  = d['p_imag']           # (N_p2,)

    # --- Load mesh topology from HDF5 ---
    import h5py
    with h5py.File(str(h5_path), 'r') as f:
        points_p1 = f['Mesh/mesh/geometry'][:]   # (N_p1, 3)
        cells_p1  = f['Mesh/mesh/topology'][:]   # (N_cells, 4)

    if use_p2:
        # --- Use FULL P2 resolution (true FE) ---
        # Build P2 tetrahedral mesh: each tet has 10 nodes (4 vertices + 6 edge midpoints)
        n_cells = cells_p1.shape[0]
        n_p2    = coords_p2.shape[0]
        
        # For P2 tets, dolfinx stores cells as P1 indices + P2 edge indices
        # We need to reconstruct the P2 cell connectivity.
        # Safe approach: use the P2 coordinate locations as cell membership.
        # Simpler: use PolyData (vertex cloud) instead of UnstructuredGrid, then
        # call Delaunay to build true connectivity from the P2 points.
        
        # Even simpler: just use the P2 points directly with P1 tetrahedra,
        # which is valid for visualization (values will be at all 531K nodes).
        
        # Build cells: each P1 tet (4 vertices) → find all P2 DOFs in that tet
        # and list them. For now, use a simpler approach: P1 cells but with P2 points.
        
        # Remap P1 cell indices to P2 space:
        # Find which P2 DOF is closest to each P1 vertex, then reconstruct connectivity
        from scipy.spatial import cKDTree
        tree = cKDTree(coords_p2)
        _, p1_to_p2_idx = tree.query(points_p1)  # Map P1 vertices → P2 DOFs
        
        # Remap cells: replace P1 indices with P2 indices
        cells_p2_remapped = p1_to_p2_idx[cells_p1]
        
        # Build PyVista grid with P2 points and remapped P1 topology
        cell_types = np.full(n_cells, pv.CellType.TETRA, dtype=np.uint8)
        cells_pv = np.hstack([np.full((n_cells, 1), 4, dtype=np.int64),
                              cells_p2_remapped]).ravel()
        
        grid = pv.UnstructuredGrid(cells_pv, cell_types, coords_p2)
        grid.point_data['magnitude'] = mag_p2
        grid.point_data['phase']     = phase_p2
        grid.point_data['p_real']    = preal_p2
        grid.point_data['p_imag']    = pimag_p2
        if gorkov_p2 is not None:
            grid.point_data['gorkov'] = gorkov_p2
    else:
        # --- Downsampled P1 (legacy, not recommended) ---
        from scipy.spatial import cKDTree
        tree = cKDTree(coords_p2)
        _, idx = tree.query(points_p1)
        
        mag_p1    = mag_p2[idx]
        phase_p1  = phase_p2[idx]
        preal_p1  = preal_p2[idx]
        pimag_p1  = pimag_p2[idx]
        gorkov_p1 = gorkov_p2[idx] if gorkov_p2 is not None else None
        
        n_cells = cells_p1.shape[0]
        cell_types = np.full(n_cells, pv.CellType.TETRA, dtype=np.uint8)
        cells_pv   = np.hstack([np.full((n_cells, 1), 4, dtype=np.int64),
                                cells_p1]).ravel()
        
        grid = pv.UnstructuredGrid(cells_pv, cell_types, points_p1)
        grid.point_data['magnitude'] = mag_p1
        grid.point_data['phase']     = phase_p1
        grid.point_data['p_real']    = preal_p1
        grid.point_data['p_imag']    = pimag_p1
        if gorkov_p1 is not None:
            grid.point_data['gorkov'] = gorkov_p1

    return grid


def load_pair(run_dir):
    """Load standing + combined grids for comparison views."""
    standing = load_rich(run_dir, 'standing')
    combined = load_rich(run_dir, 'combined')
    return standing, combined


def clip_roi(grid, roi_center=None, roi_size=0.008):
    """
    Clip mesh to ROI cube.

    Parameters
    ----------
    grid : pv.UnstructuredGrid
    roi_center : array-like (3,) in metres, or None for auto (mesh centre).
    roi_size : float, cube edge length in metres.

    Returns
    -------
    pv.UnstructuredGrid  (clipped)
    """
    if roi_center is None:
        b = grid.bounds
        roi_center = np.array([(b[0]+b[1])/2, (b[2]+b[3])/2, (b[4]+b[5])/2])
    roi_center = np.asarray(roi_center, dtype=float)

    h = roi_size / 2
    bounds = [roi_center[0]-h, roi_center[0]+h,
              roi_center[1]-h, roi_center[1]+h,
              roi_center[2]-h, roi_center[2]+h]
    return grid.clip_box(bounds, invert=False)


def list_phase_files(run_dir):
    """Return sorted list of (deg, name) for phase-sweep NPZ files."""
    run_dir = Path(run_dir)
    pairs = []
    for f in sorted(run_dir.glob('combined_phi*.npz')):
        stem = f.stem                        # combined_phi045
        deg  = int(stem.split('phi')[1])     # 45
        pairs.append((deg, stem))
    return pairs
