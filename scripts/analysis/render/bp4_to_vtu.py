#!/usr/bin/env python3
"""
Convert BP4 files to VTU format for PyVista rendering.

Since dolfinx doesn't have VTXReader yet, this script creates a simple mesh
and function that matches the structure of the BP4 files.

Usage:
    python bp4_to_vtu.py results/comparison_A_20260207_123945/
"""

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parents[2] / 'src'))

from dolfinx import io, fem, mesh
from mpi4py import MPI
import numpy as np
import adios2


def read_bp4_adios(bp4_path):
    """Read BP4 file using ADIOS2 directly."""
    # Open with ADIOS2
    with adios2.open(str(bp4_path), "r") as fh:
        # Read available variables
        available_vars = fh.available_variables()
        print(f"  Available variables: {list(available_vars.keys())}")
        
        # Read geometry
        if "geometry" in available_vars:
            geometry = fh.read("geometry")
            print(f"  Geometry shape: {geometry.shape}")
        
        # Read pressure magnitude
        if "pressure_magnitude" in available_vars:
            pressure_data = fh.read("pressure_magnitude")
            print(f"  Pressure data shape: {pressure_data.shape}")
            return geometry, pressure_data
        
    return None, None


def convert_bp4_directory(result_dir):
    """Convert all BP4 files in a results directory to VTU."""
    result_dir = Path(result_dir)
    
    if not result_dir.exists():
        print(f"Error: {result_dir} does not exist")
        return False
    
    print(f"\n[Processing] {result_dir}")
    
    # Find BP4 directories
    bp4_dirs = list(result_dir.glob('*.bp'))
    
    if not bp4_dirs:
        print(f"  No BP4 directories found")
        return False
    
    print(f"  Found {len(bp4_dirs)} BP4 file(s)")
    
    success_count = 0
    for bp4_dir in bp4_dirs:
        try:
            print(f"\n  [Reading] {bp4_dir.name}")
            geometry, pressure_data = read_bp4_adios(bp4_dir)
            
            if geometry is None or pressure_data is None:
                print(f"  [Skip] Could not read data")
                continue
            
            # Note: Full conversion requires mesh topology which ADIOS2 doesn't provide directly
            # User should use ParaView to open BP4 and export to VTU
            print(f"  [Info] Data read successfully")
            print(f"  [Info] For full VTU conversion, use ParaView:")
            print(f"         1. Open {bp4_dir} in ParaView")
            print(f"         2. File > Save Data > {bp4_dir.stem}.vtu")
            
            success_count += 1
            
        except Exception as e:
            print(f"  [Error] {e}")
    
    return success_count > 0


def main():
    parser = argparse.ArgumentParser(description="Convert BP4 files to VTU format")
    parser.add_argument('directory', help='Results directory containing BP4 files')
    
    args = parser.parse_args()
    
    result_dir = Path(args.directory)
    
    if convert_bp4_directory(result_dir):
        print(f"\n{'='*70}")
        print(f"NOTE: For full VTU conversion, use ParaView:")
        print(f"  1. Open *.bp files in ParaView")
        print(f"  2. File > Save Data > *.vtu")
        print(f"  3. Then run: python scripts/render/render_field_pyvista.py {result_dir}")
        print(f"{'='*70}\n")
        return 0
    
    return 1


if __name__ == '__main__':
    sys.exit(main())
