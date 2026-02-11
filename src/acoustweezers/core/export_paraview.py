"""
ParaView Export Utilities for Acousto-Tweezers.

This is the SINGLE AUTHORITATIVE module for ParaView-ready exports.
All VTU/XDMF exports should use functions from this module.

PyVista visualization scripts are SECONDARY and should not be expanded.

Author: Acousto-Tweezers Project
Date: 2026-02-08
"""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime

from dolfinx import fem, mesh as dmesh
from dolfinx.io import XDMFFile, VTKFile
from mpi4py import MPI

# Try to import meshio for more flexible VTU export
try:
    import meshio
    HAS_MESHIO = True
except ImportError:
    HAS_MESHIO = False


@dataclass
class FieldData:
    """Container for field data to export."""
    name: str
    values: np.ndarray
    is_vector: bool = False
    description: str = ""


@dataclass
class ExportConfig:
    """Configuration for ParaView export."""
    output_dir: Path
    mesh_name: str = "mesh"
    include_mesh_xdmf: bool = True
    include_mesh_vtu: bool = True
    float_precision: int = 8  # decimal places
    compress_vtu: bool = True


def export_mesh_xdmf(
    domain: dmesh.Mesh,
    output_path: Path,
    name: str = "mesh"
) -> Path:
    """
    Export mesh to XDMF format for ParaView.
    
    Parameters
    ----------
    domain : dolfinx.mesh.Mesh
        The computational mesh
    output_path : Path
        Output directory
    name : str
        Base name for mesh files
        
    Returns
    -------
    Path
        Path to the created XDMF file
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    xdmf_path = output_path / f"{name}.xdmf"
    
    with XDMFFile(domain.comm, str(xdmf_path), "w") as xdmf:
        xdmf.write_mesh(domain)
    
    return xdmf_path


def export_function_vtu(
    func: fem.Function,
    output_path: Path,
    name: str,
    additional_fields: Optional[Dict[str, np.ndarray]] = None,
) -> Path:
    """
    Export a FEniCSx function to VTU format.
    
    Parameters
    ----------
    func : fem.Function
        The function to export (used for mesh/coordinates)
    output_path : Path
        Output directory
    name : str
        Base name for output file
    additional_fields : dict, optional
        Additional scalar/vector fields to include
        
    Returns
    -------
    Path
        Path to the created VTU file
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    vtu_path = output_path / f"{name}.vtu"
    
    # Get mesh and coordinates
    V = func.function_space
    domain = V.mesh
    coords = V.tabulate_dof_coordinates()
    
    # For complex functions, we need to handle real/imag separately
    with VTKFile(domain.comm, str(vtu_path), "w") as vtk:
        vtk.write_function(func)
    
    return vtu_path


def export_fields_to_vtu(
    domain: dmesh.Mesh,
    coords: np.ndarray,
    fields: Dict[str, np.ndarray],
    output_path: Path,
    name: str,
    mesh_topology: Optional[np.ndarray] = None,
) -> Path:
    """
    Export multiple fields to VTU using meshio.
    
    This provides more control over the output format than dolfinx.io.
    
    Parameters
    ----------
    domain : dolfinx.mesh.Mesh
        The mesh (used for topology if not provided)
    coords : np.ndarray
        Point coordinates (N, 3)
    fields : dict
        Dictionary of field_name -> values
        Scalars: (N,) array
        Vectors: (N, 3) array
    output_path : Path
        Output directory
    name : str
        Base name for output file
    mesh_topology : np.ndarray, optional
        Cell connectivity. If None, uses point cloud.
        
    Returns
    -------
    Path
        Path to the created VTU file
    """
    if not HAS_MESHIO:
        raise ImportError("meshio required for flexible VTU export. Install with: pip install meshio")
    
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    vtu_path = output_path / f"{name}.vtu"
    
    # Prepare point data
    point_data = {}
    for field_name, values in fields.items():
        if np.iscomplexobj(values):
            # Split complex into real/imag
            point_data[f"{field_name}_real"] = np.real(values).astype(np.float64)
            point_data[f"{field_name}_imag"] = np.imag(values).astype(np.float64)
        else:
            point_data[field_name] = values.astype(np.float64)
    
    if mesh_topology is not None:
        # Create mesh with cells
        cells = [("tetra", mesh_topology)]
        m = meshio.Mesh(coords, cells, point_data=point_data)
    else:
        # Point cloud (no cells)
        m = meshio.Mesh(coords, [], point_data=point_data)
    
    m.write(str(vtu_path))
    return vtu_path


def create_pressure_fields_dict(
    p_values: np.ndarray,
    omega: float,
    rho: float,
    coords: np.ndarray,
    prefix: str = "p",
    include_velocity: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Create dictionary of pressure-derived fields for VTU export.
    
    Parameters
    ----------
    p_values : np.ndarray
        Complex pressure values at DOFs
    omega : float
        Angular frequency
    rho : float
        Fluid density
    coords : np.ndarray
        DOF coordinates (N, 3)
    prefix : str
        Field name prefix (e.g., 'p', 'delta_p')
    include_velocity : bool
        Whether to compute first-order velocity
        
    Returns
    -------
    dict
        Dictionary of field arrays ready for VTU export
    """
    fields = {}
    
    # Pressure components
    fields[f"{prefix}_real"] = np.real(p_values)
    fields[f"{prefix}_imag"] = np.imag(p_values)
    fields[f"{prefix}_mag"] = np.abs(p_values)
    fields[f"{prefix}_phase"] = np.angle(p_values)
    
    if include_velocity:
        # First-order velocity: v1 = (1/iωρ) ∇p
        # Approximate gradient using finite differences on DOF coordinates
        # This is a simplified approach - proper gradient projection is more accurate
        
        # For now, use a simplified plane-wave approximation
        # |v1| ≈ |p| / (ρc) in far field
        c = 1484.0  # m/s (water)
        v1_mag = np.abs(p_values) / (rho * c)
        fields[f"v1_mag_approx"] = v1_mag
    
    return fields


def create_streaming_fields_dict(
    u_s_values: np.ndarray,
    prefix: str = "u_s",
) -> Dict[str, np.ndarray]:
    """
    Create dictionary of streaming fields for VTU export.
    
    Parameters
    ----------
    u_s_values : np.ndarray
        Streaming velocity at DOFs, shape (N, 3)
    prefix : str
        Field name prefix
        
    Returns
    -------
    dict
        Dictionary of field arrays
    """
    fields = {}
    
    # Vector components
    fields[f"{prefix}_x"] = u_s_values[:, 0]
    fields[f"{prefix}_y"] = u_s_values[:, 1]
    fields[f"{prefix}_z"] = u_s_values[:, 2]
    
    # Magnitude
    fields[f"{prefix}_mag"] = np.linalg.norm(u_s_values, axis=1)
    
    return fields


def create_gorkov_fields_dict(
    U_values: np.ndarray,
    F_values: Optional[np.ndarray] = None,
    prefix: str = "",
) -> Dict[str, np.ndarray]:
    """
    Create dictionary of Gor'kov fields for VTU export.
    
    Parameters
    ----------
    U_values : np.ndarray
        Gor'kov potential at DOFs
    F_values : np.ndarray, optional
        Radiation force at DOFs, shape (N, 3)
    prefix : str
        Field name prefix
        
    Returns
    -------
    dict
        Dictionary of field arrays
    """
    pre = f"{prefix}_" if prefix else ""
    fields = {}
    
    fields[f"{pre}U_gorkov"] = U_values
    
    if F_values is not None:
        fields[f"{pre}F_rad_x"] = F_values[:, 0]
        fields[f"{pre}F_rad_y"] = F_values[:, 1]
        fields[f"{pre}F_rad_z"] = F_values[:, 2]
        fields[f"{pre}F_rad_mag"] = np.linalg.norm(F_values, axis=1)
    
    return fields


def write_paraview_readme(
    output_dir: Path,
    config: Dict[str, Any],
    files_created: List[str],
) -> Path:
    """
    Write PARAVIEW_README.md with step-by-step instructions.
    
    Parameters
    ----------
    output_dir : Path
        Output directory
    config : dict
        Simulation configuration
    files_created : list
        List of VTU/XDMF files created
        
    Returns
    -------
    Path
        Path to the created README
    """
    output_dir = Path(output_dir)
    readme_path = output_dir / "PARAVIEW_README.md"
    
    content = f"""# ParaView Visualization Guide

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}  
**Configuration**: {config.get('name', 'Shallow Dish Demo')}

---

## Files Created

{chr(10).join(f'- `{f}`' for f in files_created)}

---

## Step-by-Step Instructions

### 1. Load Combined Pressure Field

1. **File → Open** → `combined_fields.vtu`
2. Click **Apply**
3. **Filters → Alphabetical → Contour**
   - Contour By: `p_mag`
   - Value: {config.get('p_contour_value', 'Set to 90th percentile')}
4. Color by `p_phase` using **twilight** colormap

### 2. Visualize Streaming (Tornado Flow)

1. **File → Open** → `streaming_fields.vtu`
2. Click **Apply**
3. **Filters → Alphabetical → Stream Tracer**
   - Seed Type: **Point Cloud**
   - Center: `({config.get('vortex_center_x', 0.025)}, {config.get('vortex_center_y', 0.025)}, {config.get('vortex_center_z', 0.002)})`
   - Radius: `0.01`
   - Number of Points: `50`
4. Color streamlines by `u_s_mag`

### 3. Overlay Gor'kov Traps

1. **File → Open** → `gorkov_fields.vtu`
2. Click **Apply**
3. **Filters → Contour**
   - Contour By: `U_gorkov`
   - Value: Low values (trap wells)
4. Color by `U_gorkov` or `F_rad_mag`

### 4. View Particle Trajectories

1. **File → Open** → `particles.csv`
2. Click **Apply** (opens as Table)
3. **Filters → Table To Points**
   - X Column: `x_m`
   - Y Column: `y_m`
   - Z Column: `z_m`
4. Apply **Tube** filter for visibility
5. Color by `time` to show progression

### 5. Compare Standing vs Combined (Delta)

1. Load `delta_fields.vtu`
2. **Filters → Slice**
   - Plane Normal: Z
   - Z positions: 0.001, 0.0025, 0.004 m
3. Color by `delta_p_mag` to verify vortex bulk penetration

---

## Recommended Colormaps

| Field | Colormap | Range |
|-------|----------|-------|
| `p_mag` | viridis | 0 to max |
| `p_phase` | twilight (cyclic) | -π to π |
| `u_s_mag` | plasma | 0 to max |
| `U_gorkov` | coolwarm (diverging) | centered on 0 |
| `F_rad_mag` | inferno | 0 to max |

---

## Expected Physics

1. **Standing pattern**: Lateral standing waves (nodal planes in x or y)
2. **Vortex pattern**: Phase winding around core, propagation through depth
3. **Streaming**: Vortical mean flow near boundaries and vortex core
4. **Particles**: Should track toward trap wells, advected by streaming

---

*Generated by acousto-tweezers export utilities*
"""
    
    with open(readme_path, 'w') as f:
        f.write(content)
    
    return readme_path


def write_diagnostics_json(
    output_dir: Path,
    diagnostics: Dict[str, Any],
) -> Path:
    """
    Write diagnostics.json with quantitative sanity checks.
    
    Parameters
    ----------
    output_dir : Path
        Output directory (meta subdirectory will be used)
    diagnostics : dict
        Diagnostics data
        
    Returns
    -------
    Path
        Path to the created JSON file
    """
    output_dir = Path(output_dir)
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    
    json_path = meta_dir / "diagnostics.json"
    
    # Ensure all values are JSON-serializable
    def make_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [make_serializable(v) for v in obj]
        return obj
    
    serializable = make_serializable(diagnostics)
    
    with open(json_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    
    return json_path


def write_config_json(
    output_dir: Path,
    config: Dict[str, Any],
) -> Path:
    """
    Write config.json with full simulation parameters.
    """
    output_dir = Path(output_dir)
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    
    json_path = meta_dir / "config.json"
    
    # Convert dataclass if needed
    if hasattr(config, '__dataclass_fields__'):
        config = asdict(config)
    
    # Ensure serializable
    def make_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [make_serializable(v) for v in obj]
        return obj
    
    serializable = make_serializable(config)
    
    with open(json_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    
    return json_path
