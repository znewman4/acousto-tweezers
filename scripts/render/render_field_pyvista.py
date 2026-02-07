#!/usr/bin/env python3
"""
PyVista 3D Rendering for Vortex + Standing Wave Fields

Loads BP4 files and generates:
- Iso-surface renderings at multiple pressure levels
- Axial and radial slices
- Side-by-side comparisons

Usage:
    python render_field_pyvista.py results/comparison_A_20260207_121651/
"""

import argparse
from pathlib import Path
import numpy as np
import sys

# Check for PyVista
try:
    import pyvista as pv
    pv.set_plot_theme("document")
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    print("WARNING: PyVista not installed. Install with: pip install pyvista")

# Check for ADIOS2
try:
    from adios2 import Stream
    ADIOS2_AVAILABLE = True
except ImportError:
    ADIOS2_AVAILABLE = False


def detect_offscreen_capable():
    """Detect if offscreen rendering is possible."""
    if not PYVISTA_AVAILABLE:
        return False
    
    try:
        # Try creating an offscreen plotter
        plotter = pv.Plotter(off_screen=True, window_size=[800, 600])
        plotter.close()
        return True
    except:
        return False


def load_bp4_field(bp_dir: Path):
    """
    Load pressure field from BP4 directory.
    
    Returns PyVista UnstructuredGrid or None if not readable.
    """
    if not bp_dir.exists():
        print(f"ERROR: Directory not found: {bp_dir}")
        return None
    
    # Try reading .vtu files (if converted from BP4)
    vtu_files = list(bp_dir.glob("*.vtu"))
    if vtu_files:
        try:
            mesh = pv.read(vtu_files[0])
            print(f"[Load] Read VTU: {vtu_files[0].name}")
            return mesh
        except Exception as e:
            print(f"[Load] Failed to read VTU: {e}")
            return None
    
    # BP4 reading requires ADIOS2
    if not ADIOS2_AVAILABLE:
        print(f"[Load] Cannot read BP4 without ADIOS2. Convert to VTU with ParaView.")
        return None
    
    # TODO: Implement BP4 reading via ADIOS2
    print(f"[Load] BP4 reading not yet implemented.")
    return None


def render_isosurfaces(mesh: pv.UnstructuredGrid, output_path: Path, 
                      title: str = "Pressure Field", offscreen: bool = True):
    """Render iso-surfaces at multiple pressure levels."""
    
    if mesh is None:
        print(f"[Render] Skipping {title}: mesh not loaded")
        return
    
    # Get pressure field
    if 'pressure_magnitude' not in mesh.array_names:
        print(f"[Render] WARNING: 'pressure_magnitude' not in mesh. Available: {mesh.array_names}")
        # Try first array
        if len(mesh.array_names) > 0:
            field_name = mesh.array_names[0]
            print(f"[Render] Using field: {field_name}")
        else:
            print(f"[Render] No fields available")
            return
    else:
        field_name = 'pressure_magnitude'
    
    p_mag = mesh[field_name]
    p_max = np.max(p_mag)
    p_min = np.min(p_mag)
    
    print(f"[Render] Field range: [{p_min:.3e}, {p_max:.3e}] Pa")
    
    # Create plotter
    plotter = pv.Plotter(off_screen=offscreen, window_size=[1200, 800])
    
    # Add iso-surfaces at 30%, 50%, 70% of max
    levels = [0.3, 0.5, 0.7]
    colors = ['lightblue', 'cyan', 'blue']
    opacities = [0.3, 0.5, 0.7]
    
    for level, color, opacity in zip(levels, colors, opacities):
        try:
            contour = mesh.contour([level * p_max], scalars=field_name)
            if contour.n_points > 0:
                plotter.add_mesh(contour, color=color, opacity=opacity, 
                               show_scalar_bar=False, label=f'{level*100:.0f}% max')
        except Exception as e:
            print(f"[Render] Failed to create contour at {level}: {e}")
    
    # Add bounding box
    plotter.add_mesh(mesh.outline(), color='black', line_width=2)
    
    # Configure view
    plotter.add_title(f"{title}\nmax|p| = {p_max:.2e} Pa", font_size=12)
    plotter.add_legend(loc='upper right')
    plotter.view_isometric()
    plotter.add_axes()
    
    # Render and save
    try:
        plotter.screenshot(output_path, transparent_background=False)
        print(f"[Render] Saved {output_path}")
    except Exception as e:
        print(f"[Render] Failed to save screenshot: {e}")
    finally:
        plotter.close()


def render_slices(mesh: pv.UnstructuredGrid, output_path: Path, 
                 title: str = "Pressure Slices", offscreen: bool = True):
    """Render axial and radial slices."""
    
    if mesh is None:
        print(f"[Render] Skipping {title}: mesh not loaded")
        return
    
    if 'pressure_magnitude' not in mesh.array_names:
        if len(mesh.array_names) == 0:
            print(f"[Render] No fields available")
            return
        field_name = mesh.array_names[0]
    else:
        field_name = 'pressure_magnitude'
    
    # Get domain bounds
    bounds = mesh.bounds  # [xmin, xmax, ymin, ymax, zmin, zmax]
    x_mid = (bounds[0] + bounds[1]) / 2
    y_mid = (bounds[2] + bounds[3]) / 2
    z_mid = (bounds[4] + bounds[5]) / 2
    
    # Create plotter with 2x2 layout
    plotter = pv.Plotter(shape=(2, 2), off_screen=offscreen, window_size=[1600, 1600])
    
    # Top-left: XY slice
    plotter.subplot(0, 0)
    try:
        slice_xy = mesh.slice(normal='z', origin=[x_mid, y_mid, z_mid])
        plotter.add_mesh(slice_xy, scalars=field_name, cmap='viridis', 
                        show_scalar_bar=True, scalar_bar_args={'title': '|p| (Pa)'})
        plotter.add_title("XY Slice (mid-height)")
        plotter.view_xy()
    except Exception as e:
        print(f"[Render] Failed XY slice: {e}")
    
    # Top-right: XZ slice
    plotter.subplot(0, 1)
    try:
        slice_xz = mesh.slice(normal='y', origin=[x_mid, y_mid, z_mid])
        plotter.add_mesh(slice_xz, scalars=field_name, cmap='viridis', 
                        show_scalar_bar=True, scalar_bar_args={'title': '|p| (Pa)'})
        plotter.add_title("XZ Slice (mid-y)")
        plotter.view_xz()
    except Exception as e:
        print(f"[Render] Failed XZ slice: {e}")
    
    # Bottom-left: YZ slice
    plotter.subplot(1, 0)
    try:
        slice_yz = mesh.slice(normal='x', origin=[x_mid, y_mid, z_mid])
        plotter.add_mesh(slice_yz, scalars=field_name, cmap='viridis', 
                        show_scalar_bar=True, scalar_bar_args={'title': '|p| (Pa)'})
        plotter.add_title("YZ Slice (mid-x)")
        plotter.view_yz()
    except Exception as e:
        print(f"[Render] Failed YZ slice: {e}")
    
    # Bottom-right: 3D view
    plotter.subplot(1, 1)
    try:
        plotter.add_mesh(slice_xy, scalars=field_name, cmap='viridis', opacity=0.8)
        plotter.add_mesh(mesh.outline(), color='black', line_width=2)
        plotter.add_title("3D Context")
        plotter.view_isometric()
    except Exception as e:
        print(f"[Render] Failed 3D view: {e}")
    
    # Add overall title
    plotter.add_text(title, position='upper_edge', font_size=14)
    
    # Render
    try:
        plotter.screenshot(output_path, transparent_background=False)
        print(f"[Render] Saved {output_path}")
    except Exception as e:
        print(f"[Render] Failed to save screenshot: {e}")
    finally:
        plotter.close()


def main():
    parser = argparse.ArgumentParser(description="PyVista 3D Rendering")
    parser.add_argument('result_dir', type=str, help="Result directory with .bp files")
    parser.add_argument('--output_dir', type=str, default=None, help="Output directory for PNGs")
    
    args = parser.parse_args()
    
    result_dir = Path(args.result_dir)
    if not result_dir.exists():
        print(f"ERROR: Directory not found: {result_dir}")
        sys.exit(1)
    
    output_dir = Path(args.output_dir) if args.output_dir else result_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"PyVista 3D RENDERING")
    print(f"{'='*70}")
    print(f"Input:  {result_dir}")
    print(f"Output: {output_dir}")
    
    if not PYVISTA_AVAILABLE:
        print(f"\nERROR: PyVista not installed.")
        print(f"Install with: pip install pyvista")
        print(f"{'='*70}\n")
        sys.exit(1)
    
    # Detect offscreen capability
    offscreen = detect_offscreen_capable()
    if offscreen:
        print(f"Rendering: Offscreen (headless)")
        pv.OFF_SCREEN = True
    else:
        print(f"Rendering: Onscreen (requires display)")
        print(f"For headless server, run: xvfb-run python {sys.argv[0]} ...")
    
    print(f"{'='*70}\n")
    
    # Look for XDMF files, VTU files, or BP4 directories
    xdmf_files = {
        'standing': result_dir / 'standing_only.xdmf',
        'vortex': result_dir / 'vortex_only.xdmf',
        'combined': result_dir / 'combined.xdmf'
    }
    
    vtu_files = {
        'standing': result_dir / 'standing_only.vtu',
        'vortex': result_dir / 'vortex_only.vtu',
        'combined': result_dir / 'combined.vtu'
    }
    
    bp_dirs = {
        'standing': result_dir / 'standing_only.bp',
        'vortex': result_dir / 'vortex_only.bp',
        'combined': result_dir / 'combined.bp'
    }
    
    # Try loading meshes - prefer XDMF, then VTU, then BP4
    meshes = {}
    for name in ['standing', 'vortex', 'combined']:
        xdmf_file = xdmf_files[name]
        vtu_file = vtu_files[name]
        bp_dir = bp_dirs[name]
        
        # Try XDMF first (best compatibility)
        if xdmf_file.exists():
            print(f"[Load] Attempting to load {name} from {xdmf_file.name}")
            try:
                mesh = pv.read(xdmf_file)
                meshes[name] = mesh
                print(f"[Load] Success: {name}")
                continue
            except Exception as e:
                print(f"[Load] Failed to read XDMF: {e}")
        
        # Try VTU
        if vtu_file.exists():
            print(f"[Load] Attempting to load {name} from {vtu_file.name}")
            try:
                mesh = pv.read(vtu_file)
                meshes[name] = mesh
                print(f"[Load] Success: {name}")
                continue
            except Exception as e:
                print(f"[Load] Failed to read VTU: {e}")
        
        # Try BP4 directory
        if bp_dir.exists():
            print(f"[Load] Attempting to load {name} from {bp_dir.name}")
            mesh = load_bp4_field(bp_dir)
            if mesh is not None:
                meshes[name] = mesh
            else:
                print(f"[Load] Failed to load {name}")
        else:
            print(f"[Load] Not found: {name}")
    
    if not meshes:
        print(f"\nERROR: No meshes loaded. BP4 reading requires:")
        print(f"1. Convert BP4 to VTU using ParaView")
        print(f"2. Place .vtu files in BP4 directories")
        print(f"3. Or implement ADIOS2 reader")
        print(f"\n{'='*70}\n")
        sys.exit(1)
    
    # Render each case
    if 'standing' in meshes:
        print(f"\n[Render] Standing wave...")
        render_isosurfaces(meshes['standing'], output_dir / 'standing_iso.png', 
                          "Standing Wave", offscreen)
        render_slices(meshes['standing'], output_dir / 'standing_slices.png',
                     "Standing Wave Slices", offscreen)
    
    if 'vortex' in meshes:
        print(f"\n[Render] Vortex...")
        render_isosurfaces(meshes['vortex'], output_dir / 'vortex_iso.png',
                          "Vortex Lens", offscreen)
        render_slices(meshes['vortex'], output_dir / 'vortex_slices.png',
                     "Vortex Lens Slices", offscreen)
    
    if 'combined' in meshes:
        print(f"\n[Render] Combined...")
        render_isosurfaces(meshes['combined'], output_dir / 'combined_iso.png',
                          "Combined Field", offscreen)
        render_slices(meshes['combined'], output_dir / 'combined_slices.png',
                     "Combined Field Slices", offscreen)
    
    print(f"\n{'='*70}")
    print(f"RENDERING COMPLETE")
    print(f"Check: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
