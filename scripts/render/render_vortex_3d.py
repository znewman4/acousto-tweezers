#!/usr/bin/env python3
"""
3D Rendering Script for Vortex + Standing Wave Comparison

Reads VTU/BP4 files exported from comparison script and generates:
- Iso-surface renderings at multiple pressure levels
- Volume slices (axial, radial)
- Combined visualizations

Requires PyVista with offscreen rendering support.
For headless servers: use `xvfb-run python render_vortex_3d.py ...`

Usage:
    python render_vortex_3d.py results/comparison_A_20260207_112806/
"""

import argparse
from pathlib import Path
import numpy as np
import sys

try:
    import pyvista as pv
    pv.set_plot_theme("document")
    pv.OFF_SCREEN = True
except ImportError:
    print("ERROR: PyVista not installed. Install with: pip install pyvista")
    sys.exit(1)


def load_bp4_dataset(bp_dir: Path):
    """Load BP4 dataset from VTXWriter output.
    
    Parameters
    ----------
    bp_dir : Path
        Directory containing .bp files
    
    Returns
    -------
    pv.UnstructuredGrid
        PyVista mesh with pressure_magnitude field
    """
    # BP4 files are typically read through ADIOS2 reader
    # For now, try direct VTK reading
    vtk_files = list(bp_dir.glob("*.vtu"))
    
    if vtk_files:
        # Found VTU files
        mesh = pv.read(vtk_files[0])
        return mesh
    else:
        # Try reading BP directory structure
        # Note: May require ADIOS2-aware PyVista build
        try:
            from adios2 import Stream
            # This is a placeholder - actual implementation depends on ADIOS2 setup
            print(f"Warning: BP4 reading not fully implemented. Directory: {bp_dir}")
            return None
        except ImportError:
            print(f"Warning: Cannot read BP4 without ADIOS2. Directory: {bp_dir}")
            return None


def render_isosurfaces(mesh: pv.UnstructuredGrid, output_path: Path, title: str = "Pressure Field"):
    """Render iso-surfaces at multiple pressure levels.
    
    Parameters
    ----------
    mesh : pv.UnstructuredGrid
        Mesh with 'pressure_magnitude' field
    output_path : Path
        Output image path
    title : str
        Plot title
    """
    if mesh is None:
        print(f"Skipping {title}: mesh not loaded")
        return
    
    # Get pressure field
    if 'pressure_magnitude' not in mesh.array_names:
        print(f"Warning: 'pressure_magnitude' not found in mesh. Available: {mesh.array_names}")
        return
    
    p_mag = mesh['pressure_magnitude']
    p_max = np.max(p_mag)
    p_min = np.min(p_mag)
    
    # Create plotter
    plotter = pv.Plotter(off_screen=True, window_size=[1200, 800])
    
    # Add iso-surfaces at 30%, 50%, 70% of max
    levels = [0.3, 0.5, 0.7]
    colors = ['lightblue', 'cyan', 'blue']
    opacities = [0.3, 0.5, 0.7]
    
    for level, color, opacity in zip(levels, colors, opacities):
        contour = mesh.contour([level * p_max], scalars='pressure_magnitude')
        if contour.n_points > 0:
            plotter.add_mesh(contour, color=color, opacity=opacity, 
                           show_scalar_bar=False, label=f'{level*100:.0f}% max')
    
    # Add bounding box
    plotter.add_mesh(mesh.outline(), color='black', line_width=2)
    
    # Configure view
    plotter.add_title(f"{title}\nmax|p| = {p_max:.2e} Pa", font_size=12)
    plotter.add_legend()
    plotter.view_isometric()
    plotter.add_axes()
    
    # Render and save
    plotter.screenshot(output_path, transparent_background=False)
    plotter.close()
    
    print(f"[Render] Saved {output_path}")


def render_slices(mesh: pv.UnstructuredGrid, output_path: Path, title: str = "Pressure Slices"):
    """Render axial and radial slices.
    
    Parameters
    ----------
    mesh : pv.UnstructuredGrid
        Mesh with 'pressure_magnitude' field
    output_path : Path
        Output image path
    title : str
        Plot title
    """
    if mesh is None:
        print(f"Skipping {title}: mesh not loaded")
        return
    
    if 'pressure_magnitude' not in mesh.array_names:
        print(f"Warning: 'pressure_magnitude' not found in mesh")
        return
    
    # Get domain bounds
    bounds = mesh.bounds  # [xmin, xmax, ymin, ymax, zmin, zmax]
    x_mid = (bounds[0] + bounds[1]) / 2
    y_mid = (bounds[2] + bounds[3]) / 2
    z_mid = (bounds[4] + bounds[5]) / 2
    
    # Create plotter with 2x2 subplots
    plotter = pv.Plotter(shape=(2, 2), off_screen=True, window_size=[1600, 1600])
    
    # Top-left: XY slice at mid-z
    plotter.subplot(0, 0)
    slice_xy = mesh.slice(normal='z', origin=[x_mid, y_mid, z_mid])
    plotter.add_mesh(slice_xy, scalars='pressure_magnitude', cmap='viridis', 
                    show_scalar_bar=True, scalar_bar_args={'title': '|p| (Pa)'})
    plotter.add_title("XY Slice (mid-height)")
    plotter.view_xy()
    
    # Top-right: XZ slice at mid-y
    plotter.subplot(0, 1)
    slice_xz = mesh.slice(normal='y', origin=[x_mid, y_mid, z_mid])
    plotter.add_mesh(slice_xz, scalars='pressure_magnitude', cmap='viridis', 
                    show_scalar_bar=True, scalar_bar_args={'title': '|p| (Pa)'})
    plotter.add_title("XZ Slice (mid-y)")
    plotter.view_xz()
    
    # Bottom-left: YZ slice at mid-x
    plotter.subplot(1, 0)
    slice_yz = mesh.slice(normal='x', origin=[x_mid, y_mid, z_mid])
    plotter.add_mesh(slice_yz, scalars='pressure_magnitude', cmap='viridis', 
                    show_scalar_bar=True, scalar_bar_args={'title': '|p| (Pa)'})
    plotter.add_title("YZ Slice (mid-x)")
    plotter.view_yz()
    
    # Bottom-right: 3D view with slice
    plotter.subplot(1, 1)
    plotter.add_mesh(slice_xy, scalars='pressure_magnitude', cmap='viridis', opacity=0.8)
    plotter.add_mesh(mesh.outline(), color='black', line_width=2)
    plotter.add_title("3D Context")
    plotter.view_isometric()
    
    # Add overall title
    plotter.add_text(title, position='upper_edge', font_size=14)
    
    # Render and save
    plotter.screenshot(output_path, transparent_background=False)
    plotter.close()
    
    print(f"[Render] Saved {output_path}")


def render_comparison_grid(mesh_standing, mesh_vortex, mesh_combined, output_path: Path):
    """Render 3x1 grid comparing three cases.
    
    Parameters
    ----------
    mesh_standing, mesh_vortex, mesh_combined : pv.UnstructuredGrid
        Meshes for three cases
    output_path : Path
        Output image path
    """
    plotter = pv.Plotter(shape=(1, 3), off_screen=True, window_size=[1800, 600])
    
    for i, (mesh, title) in enumerate([(mesh_standing, "Standing Wave"),
                                       (mesh_vortex, "Vortex"),
                                       (mesh_combined, "Combined")]):
        plotter.subplot(0, i)
        
        if mesh is None or 'pressure_magnitude' not in mesh.array_names:
            plotter.add_text(f"{title}\n(not available)", position='center')
            continue
        
        # Get bounds
        bounds = mesh.bounds
        x_mid = (bounds[0] + bounds[1]) / 2
        y_mid = (bounds[2] + bounds[3]) / 2
        z_mid = (bounds[4] + bounds[5]) / 2
        
        # Add slice
        slice_xy = mesh.slice(normal='z', origin=[x_mid, y_mid, z_mid])
        plotter.add_mesh(slice_xy, scalars='pressure_magnitude', cmap='viridis',
                        show_scalar_bar=True, scalar_bar_args={'title': '|p| (Pa)', 'vertical': True})
        plotter.add_title(title)
        plotter.view_xy()
    
    plotter.add_text("Pressure Magnitude Comparison (XY slice)", position='upper_edge', font_size=14)
    
    # Render
    plotter.screenshot(output_path, transparent_background=False)
    plotter.close()
    
    print(f"[Render] Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(description="3D Rendering for Vortex Comparison")
    parser.add_argument('result_dir', type=str,
                       help="Result directory containing .bp files")
    parser.add_argument('--output_dir', type=str, default=None,
                       help="Output directory for renders (default: same as result_dir)")
    
    args = parser.parse_args()
    
    result_dir = Path(args.result_dir)
    if not result_dir.exists():
        print(f"ERROR: Result directory not found: {result_dir}")
        sys.exit(1)
    
    output_dir = Path(args.output_dir) if args.output_dir else result_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"3D RENDERING: VORTEX + STANDING WAVE")
    print(f"{'='*60}")
    print(f"Input:  {result_dir}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")
    
    # Note: BP4 reading requires specific setup
    # For now, this is a template. Actual reading depends on FEniCSx export format
    print("Note: BP4 reading not fully implemented.")
    print("To use this script:")
    print("1. Export to XDMF format (add to compare_vortex_standing.py)")
    print("2. Or use ParaView to convert BP4 -> VTU")
    print("3. Then load VTU files with pv.read()")
    
    # Try to load meshes (placeholder)
    mesh_standing = None
    mesh_vortex = None
    mesh_combined = None
    
    # Look for VTU files as fallback
    vtu_standing = list(result_dir.glob("standing*.vtu"))
    vtu_vortex = list(result_dir.glob("vortex*.vtu"))
    vtu_combined = list(result_dir.glob("combined*.vtu"))
    
    if vtu_standing:
        mesh_standing = pv.read(vtu_standing[0])
        print(f"Loaded: {vtu_standing[0].name}")
    
    if vtu_vortex:
        mesh_vortex = pv.read(vtu_vortex[0])
        print(f"Loaded: {vtu_vortex[0].name}")
    
    if vtu_combined:
        mesh_combined = pv.read(vtu_combined[0])
        print(f"Loaded: {vtu_combined[0].name}")
    
    # Render if meshes loaded
    if mesh_standing:
        render_isosurfaces(mesh_standing, output_dir / "standing_iso.png", "Standing Wave")
        render_slices(mesh_standing, output_dir / "standing_slices.png", "Standing Wave Slices")
    
    if mesh_vortex:
        render_isosurfaces(mesh_vortex, output_dir / "vortex_iso.png", "Vortex Lens")
        render_slices(mesh_vortex, output_dir / "vortex_slices.png", "Vortex Lens Slices")
    
    if mesh_combined:
        render_isosurfaces(mesh_combined, output_dir / "combined_iso.png", "Combined")
        render_slices(mesh_combined, output_dir / "combined_slices.png", "Combined Slices")
    
    # Comparison grid
    if any([mesh_standing, mesh_vortex, mesh_combined]):
        render_comparison_grid(mesh_standing, mesh_vortex, mesh_combined,
                             output_dir / "comparison_grid.png")
    
    print(f"\n{'='*60}")
    print(f"RENDERING COMPLETE")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
