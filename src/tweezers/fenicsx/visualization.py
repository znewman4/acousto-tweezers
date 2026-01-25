"""
PyVista Visualization Module
============================
Clean, documented visualization utilities for acousto-tweezers results.

Features:
- 3D pressure field rendering with slices
- Cross-section plots
- GIF animation generation
- Publication-quality figure export
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List

import pyvista as pv
from dolfinx import fem, mesh
from mpi4py import MPI


def extract_pyvista_mesh(domain: mesh.Mesh) -> pv.UnstructuredGrid:
    """Convert DOLFINx mesh to PyVista UnstructuredGrid.
    
    Parameters
    ----------
    domain : dolfinx.mesh.Mesh
        DOLFINx mesh object
        
    Returns
    -------
    pv.UnstructuredGrid
        PyVista mesh suitable for visualization
    """
    # Get mesh topology
    tdim = domain.topology.dim
    num_cells = domain.topology.index_map(tdim).size_local
    
    # Get cell connectivity
    domain.topology.create_connectivity(tdim, 0)
    cell_to_vertex = domain.topology.connectivity(tdim, 0)
    
    # Get vertex coordinates
    coords = domain.geometry.x
    
    # Build PyVista cells array
    cells = []
    celltypes = []
    
    for i in range(num_cells):
        vertices = cell_to_vertex.links(i)
        cells.append(len(vertices))
        cells.extend(vertices)
        
        # Determine cell type
        if tdim == 3:
            if len(vertices) == 4:
                celltypes.append(pv.CellType.TETRA)
            elif len(vertices) == 8:
                celltypes.append(pv.CellType.HEXAHEDRON)
        elif tdim == 2:
            if len(vertices) == 3:
                celltypes.append(pv.CellType.TRIANGLE)
            elif len(vertices) == 4:
                celltypes.append(pv.CellType.QUAD)
    
    # Create PyVista mesh
    # Pad 2D coordinates to 3D if needed
    if coords.shape[1] == 2:
        coords_3d = np.zeros((coords.shape[0], 3))
        coords_3d[:, :2] = coords
    else:
        coords_3d = coords
    
    grid = pv.UnstructuredGrid(cells, celltypes, coords_3d)
    
    return grid


def add_field_to_mesh(
    grid: pv.UnstructuredGrid,
    field: fem.Function,
    name: str = "field"
) -> None:
    """Add a scalar field to PyVista mesh.
    
    Parameters
    ----------
    grid : pv.UnstructuredGrid
        PyVista mesh to add field to
    field : dolfinx.fem.Function
        DOLFINx function to visualize
    name : str
        Name for the field in PyVista
    """
    values = field.x.array
    
    # Handle complex values
    if np.iscomplexobj(values):
        # Add magnitude
        grid.point_data[f"{name}_mag"] = np.abs(values).real
        grid.point_data[f"{name}_real"] = values.real
        grid.point_data[f"{name}_imag"] = values.imag
        grid.point_data[f"{name}_phase"] = np.angle(values)
    else:
        grid.point_data[name] = values.real


def plot_pressure_field_3d(
    domain: mesh.Mesh,
    pressure: fem.Function,
    output_path: Optional[Path] = None,
    clim: Optional[Tuple[float, float]] = None,
    show_edges: bool = False,
    slice_normal: str = "z",
    slice_origin: Optional[Tuple[float, float, float]] = None,
    title: str = "Pressure Field |p|"
) -> Optional[pv.Plotter]:
    """Create 3D visualization of pressure field with slice.
    
    Parameters
    ----------
    domain : dolfinx.mesh.Mesh
        Computational mesh
    pressure : dolfinx.fem.Function
        Complex pressure field
    output_path : Path, optional
        If provided, save image to this path
    clim : tuple, optional
        Color limits (min, max). If None, auto-scale.
    show_edges : bool
        Whether to show mesh edges
    slice_normal : str
        Normal direction for slice plane ('x', 'y', or 'z')
    slice_origin : tuple, optional
        Origin point for slice. If None, use mesh center.
    title : str
        Plot title
        
    Returns
    -------
    pv.Plotter or None
        Plotter object if not saving, None if saved to file
    """
    # Convert mesh
    grid = extract_pyvista_mesh(domain)
    
    # Add pressure magnitude
    p_array = pressure.x.array
    if np.iscomplexobj(p_array):
        p_mag = np.abs(p_array).real
    else:
        p_mag = np.abs(p_array)
    
    grid.point_data["pressure_mag"] = p_mag
    
    # Determine slice origin
    if slice_origin is None:
        bounds = grid.bounds
        slice_origin = (
            (bounds[0] + bounds[1]) / 2,
            (bounds[2] + bounds[3]) / 2,
            (bounds[4] + bounds[5]) / 2
        )
    
    # Create plotter
    pl = pv.Plotter(off_screen=output_path is not None)
    pl.set_background("white")
    
    # Create slice
    normal_map = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}
    normal = normal_map.get(slice_normal.lower(), [0, 0, 1])
    
    sliced = grid.slice(normal=normal, origin=slice_origin)
    
    # Add slice to plot
    if clim is None:
        clim = [0, np.max(p_mag)]
    
    pl.add_mesh(
        sliced,
        scalars="pressure_mag",
        clim=clim,
        cmap="viridis",
        show_edges=show_edges,
        scalar_bar_args={
            "title": "|p| [Pa]",
            "vertical": True,
            "position_x": 0.85,
            "position_y": 0.2,
        }
    )
    
    # Add outline
    pl.add_mesh(grid.outline(), color="black", line_width=2)
    
    pl.add_title(title, font_size=12)
    pl.view_isometric()
    
    if output_path is not None:
        pl.screenshot(str(output_path), transparent_background=False)
        pl.close()
        print(f"Saved: {output_path}")
        return None
    else:
        return pl


def plot_cross_section(
    domain: mesh.Mesh,
    pressure: fem.Function,
    axis: str = "z",
    position: float = 0.0,
    output_path: Optional[Path] = None,
    clim: Optional[Tuple[float, float]] = None,
    title: str = "Cross Section"
) -> Optional[pv.Plotter]:
    """Create 2D cross-section plot.
    
    Parameters
    ----------
    domain : dolfinx.mesh.Mesh
        Computational mesh
    pressure : dolfinx.fem.Function
        Complex pressure field
    axis : str
        Axis perpendicular to slice ('x', 'y', or 'z')
    position : float
        Position along axis for slice
    output_path : Path, optional
        If provided, save image to this path
    clim : tuple, optional
        Color limits
    title : str
        Plot title
    """
    grid = extract_pyvista_mesh(domain)
    
    # Add pressure
    p_array = pressure.x.array
    p_mag = np.abs(p_array).real if np.iscomplexobj(p_array) else np.abs(p_array)
    grid.point_data["pressure_mag"] = p_mag
    
    # Determine slice parameters
    bounds = grid.bounds
    axis_map = {"x": 0, "y": 1, "z": 2}
    axis_idx = axis_map.get(axis.lower(), 2)
    
    origin = [0, 0, 0]
    origin[axis_idx] = position
    
    normal = [0, 0, 0]
    normal[axis_idx] = 1
    
    # Create slice
    sliced = grid.slice(normal=normal, origin=origin)
    
    if sliced.n_points == 0:
        print(f"Warning: No points in slice at {axis}={position}")
        return None
    
    # Create plotter
    pl = pv.Plotter(off_screen=output_path is not None)
    pl.set_background("white")
    
    if clim is None:
        clim = [0, np.max(p_mag)]
    
    pl.add_mesh(
        sliced,
        scalars="pressure_mag",
        clim=clim,
        cmap="viridis",
        show_edges=False,
        scalar_bar_args={"title": "|p| [Pa]"}
    )
    
    # Set view perpendicular to slice
    view_map = {
        "x": "yz",
        "y": "xz", 
        "z": "xy"
    }
    
    pl.add_title(title, font_size=12)
    pl.view_vector(normal, viewup=[0, 1, 0] if axis != "y" else [0, 0, 1])
    
    if output_path is not None:
        pl.screenshot(str(output_path), transparent_background=False)
        pl.close()
        print(f"Saved: {output_path}")
        return None
    else:
        return pl


def create_animation_frames(
    domain: mesh.Mesh,
    pressure: fem.Function,
    output_dir: Path,
    n_frames: int = 36,
    prefix: str = "frame"
) -> List[Path]:
    """Create animation frames rotating around the solution.
    
    Parameters
    ----------
    domain : dolfinx.mesh.Mesh
        Computational mesh
    pressure : dolfinx.fem.Function
        Complex pressure field
    output_dir : Path
        Directory to save frames
    n_frames : int
        Number of frames for full rotation
    prefix : str
        Filename prefix for frames
        
    Returns
    -------
    list of Path
        Paths to generated frame images
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    grid = extract_pyvista_mesh(domain)
    
    p_array = pressure.x.array
    p_mag = np.abs(p_array).real if np.iscomplexobj(p_array) else np.abs(p_array)
    grid.point_data["pressure_mag"] = p_mag
    
    clim = [0, np.max(p_mag)]
    
    frame_paths = []
    
    for i in range(n_frames):
        angle = i * 360 / n_frames
        
        pl = pv.Plotter(off_screen=True)
        pl.set_background("white")
        
        pl.add_mesh(
            grid,
            scalars="pressure_mag",
            clim=clim,
            cmap="viridis",
            opacity=0.8,
            scalar_bar_args={"title": "|p| [Pa]"}
        )
        
        pl.add_mesh(grid.outline(), color="black", line_width=2)
        
        # Rotate camera
        pl.camera.azimuth = angle
        pl.camera.elevation = 20
        
        # Save frame
        frame_path = output_dir / f"{prefix}_{i:04d}.png"
        pl.screenshot(str(frame_path))
        pl.close()
        
        frame_paths.append(frame_path)
    
    print(f"Generated {len(frame_paths)} frames in {output_dir}")
    return frame_paths


def frames_to_gif(
    frame_paths: List[Path],
    output_path: Path,
    duration: float = 0.1
) -> None:
    """Convert animation frames to GIF.
    
    Parameters
    ----------
    frame_paths : list of Path
        Paths to frame images
    output_path : Path
        Output GIF path
    duration : float
        Duration per frame in seconds
    """
    try:
        from PIL import Image
    except ImportError:
        print("PIL not installed, cannot create GIF")
        return
    
    images = [Image.open(p) for p in sorted(frame_paths)]
    
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=int(duration * 1000),
        loop=0
    )
    
    print(f"Saved GIF: {output_path}")


# Convenience function for quick visualization
def quick_plot(
    domain: mesh.Mesh,
    pressure: fem.Function,
    output: Optional[str] = None
) -> None:
    """Quick visualization of pressure field.
    
    Parameters
    ----------
    domain : dolfinx.mesh.Mesh
        Computational mesh
    pressure : dolfinx.fem.Function
        Pressure field
    output : str, optional
        Output filename. If None, display interactively.
    """
    if output:
        plot_pressure_field_3d(
            domain, pressure,
            output_path=Path(output),
            title="Pressure Field"
        )
    else:
        pl = plot_pressure_field_3d(domain, pressure, title="Pressure Field")
        if pl is not None:
            pl.show()


if __name__ == "__main__":
    # Demo/test the visualization module
    print("PyVista Visualization Module")
    print("============================")
    print("\nThis module provides:")
    print("  - extract_pyvista_mesh(): Convert DOLFINx mesh to PyVista")
    print("  - plot_pressure_field_3d(): 3D visualization with slices")
    print("  - plot_cross_section(): 2D cross-section plots")
    print("  - create_animation_frames(): Generate animation frames")
    print("  - frames_to_gif(): Convert frames to GIF")
    print("  - quick_plot(): Convenience function for quick visualization")
    print("\nImport and use with your DOLFINx solutions.")
