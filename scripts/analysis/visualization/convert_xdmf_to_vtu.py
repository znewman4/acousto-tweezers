#!/usr/bin/env python3
"""
Convert XDMF files to VTU format for PyVista visualization

Uses dolfinx to read XDMF and exports to VTU for reliable PyVista loading.

Usage:
    python convert_xdmf_to_vtu.py --xdmf_file results/comparison_A_20260207_124751/combined.xdmf
    python convert_xdmf_to_vtu.py --run_dir results/comparison_A_20260207_124751/
"""

import argparse
from pathlib import Path
import numpy as np
from mpi4py import MPI
from dolfinx import io
import pyvista as pv


def convert_xdmf_to_vtu(xdmf_file, output_file=None):
    """
    Convert XDMF file to VTU format.
    
    Parameters:
    - xdmf_file: Path to XDMF file
    - output_file: Path to output VTU (None = auto-generate)
    
    Returns:
    - Path to VTU file
    """
    xdmf_file = Path(xdmf_file)
    
    if output_file is None:
        output_file = xdmf_file.with_suffix('.vtu')
    else:
        output_file = Path(output_file)
    
    print(f"[Convert] {xdmf_file.name} → {output_file.name}")
    
    # Read XDMF with dolfinx
    from dolfinx import fem
    import basix.ufl
    
    with io.XDMFFile(MPI.COMM_WORLD, str(xdmf_file), "r") as file:
        mesh = file.read_mesh()
        
        # Determine element type from mesh
        cell_type = mesh.topology.cell_name()
        
        # Create function space (P1 for compatibility)
        element = basix.ufl.element("Lagrange", mesh.topology.cell_name(), 1)
        V = fem.functionspace(mesh, element)
        
        u = fem.Function(V)
        
        # Try to read pressure field (complex-valued)
        try:
            # The XDMF file should have the function data
            u.name = "p"
            file.read_function(u, "p")
            
            print(f"[Read] Function 'p' from XDMF")
            
            # Get values
            values = u.x.array
            
            # Check if complex
            is_complex = hasattr(values.dtype, 'char') and values.dtype.char == 'D'
            
            if is_complex or np.iscomplexobj(values):
                # Complex pressure - extract magnitude and phase
                p_real = values.real
                p_imag = values.imag
                magnitude = np.abs(values)
                phase = np.angle(values)
                
                print(f"[Data] Complex pressure: {len(values)} values")
                print(f"[Data] Magnitude range: [{magnitude.min():.1f}, {magnitude.max():.1f}]")
            else:
                # Real-valued
                magnitude = np.abs(values)
                phase = np.zeros_like(values)
                p_real = values
                p_imag = np.zeros_like(values)
                
                print(f"[Data] Real-valued pressure: {len(values)} values")
                print(f"[Data] Range: [{magnitude.min():.1f}, {magnitude.max():.1f}]")
            
        except Exception as e:
            print(f"[Warning] Could not read function: {e}")
            print(f"[Warning] Creating dummy data")
            
            # Create dummy data
            magnitude = np.ones(mesh.geometry.x.shape[0])
            phase = np.zeros_like(magnitude)
            p_real = magnitude
            p_imag = np.zeros_like(magnitude)
    
    # Create PyVista mesh
    points = mesh.geometry.x
    
    # Get cell connectivity
    cells = mesh.geometry.dofmap.array
    cell_type_pv = mesh.topology.cell_name()
    
    # Map cell types
    if cell_type_pv == 'tetrahedron':
        cells_reshaped = cells.reshape(-1, 4)
        cell_types = np.full(len(cells_reshaped), pv.CellType.TETRA, dtype=np.uint8)
        cells_pv = np.hstack([np.full((len(cells_reshaped), 1), 4), cells_reshaped]).ravel()
    elif cell_type_pv == 'hexahedron':
        cells_reshaped = cells.reshape(-1, 8)
        cell_types = np.full(len(cells_reshaped), pv.CellType.HEXAHEDRON, dtype=np.uint8)
        cells_pv = np.hstack([np.full((len(cells_reshaped), 1), 8), cells_reshaped]).ravel()
    else:
        raise ValueError(f"Unsupported cell type: {cell_type_pv}")
    
    # Create UnstructuredGrid
    pv_mesh = pv.UnstructuredGrid(cells_pv, cell_types, points)
    
    # Add fields
    pv_mesh.point_data['pressure_magnitude'] = magnitude
    pv_mesh.point_data['pressure_phase'] = phase
    pv_mesh.point_data['p_real'] = p_real
    pv_mesh.point_data['p_imag'] = p_imag
    
    # Save to VTU
    pv_mesh.save(str(output_file))
    
    print(f"[Saved] {output_file}")
    print(f"[VTU] {pv_mesh.n_points} points, {pv_mesh.n_cells} cells")
    
    return output_file


def convert_run_dir(run_dir):
    """
    Convert all XDMF files in a run directory to VTU.
    
    Parameters:
    - run_dir: Path to results directory
    
    Returns:
    - List of converted VTU files
    """
    run_dir = Path(run_dir)
    
    xdmf_files = list(run_dir.glob('*.xdmf'))
    
    if not xdmf_files:
        print(f"No XDMF files found in {run_dir}")
        return []
    
    print(f"\n{'='*70}")
    print(f"CONVERT RUN DIRECTORY: {run_dir.name}")
    print(f"Found {len(xdmf_files)} XDMF files")
    print(f"{'='*70}\n")
    
    vtu_files = []
    
    for xdmf_file in xdmf_files:
        try:
            vtu_file = convert_xdmf_to_vtu(xdmf_file)
            vtu_files.append(vtu_file)
        except Exception as e:
            print(f"[Error] Failed to convert {xdmf_file.name}: {e}")
    
    print(f"\n{'='*70}")
    print(f"COMPLETE: Converted {len(vtu_files)} files")
    print(f"{'='*70}\n")
    
    return vtu_files


def main():
    parser = argparse.ArgumentParser(description="Convert XDMF to VTU for PyVista")
    
    parser.add_argument('--xdmf_file', help='Single XDMF file to convert')
    parser.add_argument('--run_dir', help='Convert all XDMF files in directory')
    parser.add_argument('--output', help='Output VTU file (for single file mode)')
    
    args = parser.parse_args()
    
    if args.xdmf_file:
        convert_xdmf_to_vtu(args.xdmf_file, args.output)
    elif args.run_dir:
        convert_run_dir(args.run_dir)
    else:
        print("ERROR: Must specify --xdmf_file or --run_dir")
        return 1
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
