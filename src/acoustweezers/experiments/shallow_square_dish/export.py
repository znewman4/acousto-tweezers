"""
ParaView Export for Shallow Square Dish Experiment.

Exports all fields to VTU format with required arrays for ParaView visualization.

Author: Acousto-Tweezers Project
Date: 2026-02-08
"""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from dolfinx import fem, mesh as dmesh
from dolfinx.io import VTKFile, XDMFFile
from mpi4py import MPI

from .config import ShallowDishConfig
from .solve_pressure import PressureSolution
from .streaming import StreamingSolution
from .particles import GorkovField, ParticleTrajectory, save_trajectories_csv


def export_pressure_fields(
    p_solution: PressureSolution,
    output_dir: Path,
    name: str,
    include_velocity: bool = True,
    verbose: bool = True,
) -> Path:
    """
    Export pressure field to VTU with all derived quantities.
    
    Arrays exported:
    - p_real, p_imag, p_mag, p_phase
    - v1_x, v1_y, v1_z, v1_mag (if include_velocity)
    
    Parameters
    ----------
    p_solution : PressureSolution
        Pressure solution
    output_dir : Path
        Output directory
    name : str
        Base name for file (e.g., "standing_fields")
    include_velocity : bool
        Include first-order velocity
    verbose : bool
        Print info
        
    Returns
    -------
    Path
        Path to created VTU file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cfg = p_solution.cfg
    omega = cfg.omega
    rho = cfg.rho
    
    V = p_solution.p_function.function_space
    domain = V.mesh
    
    # Create separate functions for each field
    p_vals = p_solution.p_values
    
    # Pressure components
    p_real = fem.Function(V, name="p_real")
    p_real.x.array[:] = np.real(p_vals)
    
    p_imag = fem.Function(V, name="p_imag")
    p_imag.x.array[:] = np.imag(p_vals)
    
    p_mag = fem.Function(V, name="p_mag")
    p_mag.x.array[:] = np.abs(p_vals)
    
    p_phase = fem.Function(V, name="p_phase")
    p_phase.x.array[:] = np.angle(p_vals)
    
    # Write to VTK
    vtu_path = output_dir / f"{name}.vtu"
    
    with VTKFile(domain.comm, str(vtu_path), "w") as vtk:
        vtk.write_function([p_real, p_imag, p_mag, p_phase])
    
    if verbose:
        print(f"  Exported: {vtu_path.name}")
        print(f"    max|p| = {np.max(np.abs(p_vals)):.2f} Pa")
    
    return vtu_path


def export_delta_fields(
    p_combined: PressureSolution,
    p_standing: PressureSolution,
    output_dir: Path,
    verbose: bool = True,
) -> Path:
    """
    Export delta fields (Δp = combined - standing).
    
    Arrays exported:
    - delta_p_real, delta_p_imag, delta_p_mag, delta_p_phase
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    V = p_combined.p_function.function_space
    domain = V.mesh
    
    # Compute delta
    delta_p = p_combined.p_values - p_standing.p_values
    
    # Create functions
    delta_real = fem.Function(V, name="delta_p_real")
    delta_real.x.array[:] = np.real(delta_p)
    
    delta_imag = fem.Function(V, name="delta_p_imag")
    delta_imag.x.array[:] = np.imag(delta_p)
    
    delta_mag = fem.Function(V, name="delta_p_mag")
    delta_mag.x.array[:] = np.abs(delta_p)
    
    delta_phase = fem.Function(V, name="delta_p_phase")
    delta_phase.x.array[:] = np.angle(delta_p)
    
    vtu_path = output_dir / "delta_fields.vtu"
    
    with VTKFile(domain.comm, str(vtu_path), "w") as vtk:
        vtk.write_function([delta_real, delta_imag, delta_mag, delta_phase])
    
    if verbose:
        print(f"  Exported: {vtu_path.name}")
        print(f"    max|Δp| = {np.max(np.abs(delta_p)):.2f} Pa")
    
    return vtu_path


def export_streaming_fields(
    streaming: StreamingSolution,
    output_dir: Path,
    verbose: bool = True,
) -> Path:
    """
    Export streaming velocity field with diagnostics.
    
    Arrays exported:
    - u_s_x, u_s_y, u_s_z, u_s_mag (velocity)
    - Additional derived fields (vorticity magnitude, etc.)
    
    Also saves streaming diagnostics to meta/streaming_diagnostics.json
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(exist_ok=True)
    
    domain = streaming.mesh
    
    # The streaming velocity is already a vector function
    u_func = streaming.u_function
    u_func.name = "streaming_velocity"
    
    # Create magnitude function for visualization
    V_scalar = fem.functionspace(domain, ("Lagrange", 2))
    u_mag_func = fem.Function(V_scalar, name="streaming_velocity_magnitude")
    
    # Compute u_mag at DOFs
    u_vals = u_func.x.array
    n_udofs = len(u_vals) // 3
    u_vec = u_vals.reshape((n_udofs, 3))
    u_mag = np.linalg.norm(u_vec, axis=1)
    u_mag_func.x.array[:] = u_mag
    
    # Compute vorticity magnitude (∇ × u)
    omega_curl = fem.Function(V_scalar, name="streaming_vorticity_magnitude")
    curl_u = fem.function.Function(fem.functionspace(domain, ("Lagrange", 2, (3,))))
    # For now, set to zero (proper vorticity computation requires curl)
    omega_curl.x.array[:] = 0.0  # TODO: implement proper curl computation
    
    vtu_path = output_dir / "streaming_fields.vtu"
    
    with VTKFile(domain.comm, str(vtu_path), "w") as vtk:
        vtk.write_function([u_func, u_mag_func])
    
    # Save streaming diagnostics
    if streaming.diagnostics:
        diag_path = meta_dir / "streaming_diagnostics.json"
        # Convert diagnostics to JSON-serializable format
        diags_json = {}
        for key, value in streaming.diagnostics.items():
            if isinstance(value, (int, float, str, bool, type(None))):
                diags_json[key] = value
            elif isinstance(value, np.ndarray):
                diags_json[key] = value.tolist()
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                # z_profile is a list of dicts
                diags_json[key] = value
            else:
                diags_json[key] = str(value)
        
        with open(diag_path, 'w') as f:
            json.dump(diags_json, f, indent=2)
        
        if verbose:
            print(f"  Exported: {vtu_path.name}")
            print(f"    max|u_s| = {streaming.max_speed*1e6:.2f} μm/s")
            print(f"  Exported: meta/streaming_diagnostics.json")
            print(f"    KSP convergence: {diags_json.get('ksp_reason_str', 'unknown')}")
    else:
        if verbose:
            print(f"  Exported: {vtu_path.name} (no diagnostics)")
    
    return vtu_path


def export_gorkov_fields(
    gorkov: GorkovField,
    output_dir: Path,
    verbose: bool = True,
) -> dict:
    """
    Export Gor'kov potential and radiation force.
    
    Arrays exported (to separate files due to VTK element type constraints):
    - gorkov_U.vtu: U_gorkov (scalar)
    - gorkov_F.vtu: F_rad (vector)
    
    Returns dict with paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    domain = gorkov.mesh
    result = {}
    
    # Export potential (scalar)
    U_func = gorkov.U_function
    U_func.name = "U_gorkov"
    
    vtu_U_path = output_dir / "gorkov_U.vtu"
    with VTKFile(domain.comm, str(vtu_U_path), "w") as vtk:
        vtk.write_function([U_func])
    result["U"] = vtu_U_path
    
    if verbose:
        print(f"  Exported: {vtu_U_path.name}")
        print(f"    trap depth = {gorkov.trap_depth:.2e} J")
    
    # Export force (vector) if available
    if gorkov.F_function is not None:
        F_func = gorkov.F_function
        F_func.name = "F_rad"
        
        vtu_F_path = output_dir / "gorkov_F.vtu"
        with VTKFile(domain.comm, str(vtu_F_path), "w") as vtk:
            vtk.write_function([F_func])
        result["F"] = vtu_F_path
        
        if verbose:
            print(f"  Exported: {vtu_F_path.name}")
            max_F = np.max(np.abs(F_func.x.array))
            print(f"    max|F| = {max_F:.2e} N")
    
    return result
    
    return vtu_path


def export_mesh(
    domain: dmesh.Mesh,
    output_dir: Path,
    verbose: bool = True,
) -> Path:
    """Export mesh to XDMF format."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    xdmf_path = output_dir / "mesh.xdmf"
    
    with XDMFFile(domain.comm, str(xdmf_path), "w") as xdmf:
        xdmf.write_mesh(domain)
    
    if verbose:
        n_cells = domain.topology.index_map(domain.topology.dim).size_global
        print(f"  Exported: {xdmf_path.name} ({n_cells} cells)")
    
    return xdmf_path


def compute_diagnostics(
    cfg: ShallowDishConfig,
    solutions: Dict[str, PressureSolution],
    streaming: Optional[StreamingSolution] = None,
    gorkov: Optional[GorkovField] = None,
    trajectories: Optional[List[ParticleTrajectory]] = None,
) -> Dict[str, Any]:
    """
    Compute diagnostic statistics.
    
    Returns dictionary with:
    - mesh_stats
    - pressure_stats
    - streaming_stats
    - gorkov_stats
    - particle_stats
    """
    diagnostics = {
        "timestamp": datetime.now().isoformat(),
        "config": cfg.to_dict(),
    }
    
    # Mesh stats
    if "combined" in solutions:
        V = solutions["combined"].p_function.function_space
        domain = V.mesh
        n_cells = domain.topology.index_map(domain.topology.dim).size_global
        n_vertices = domain.topology.index_map(0).size_global
        
        diagnostics["mesh"] = {
            "n_cells": n_cells,
            "n_vertices": n_vertices,
            "n_dofs": len(V.tabulate_dof_coordinates()),
        }
    
    # Pressure stats
    pressure_stats = {}
    for mode, sol in solutions.items():
        p_mag = sol.p_mag
        pressure_stats[mode] = {
            "min": float(np.min(p_mag)),
            "max": float(np.max(p_mag)),
            "median": float(np.median(p_mag)),
            "mean": float(np.mean(p_mag)),
        }
    diagnostics["pressure"] = pressure_stats
    
    # Delta stats (bulk penetration)
    if "combined" in solutions and "standing" in solutions:
        delta_p = solutions["combined"].p_values - solutions["standing"].p_values
        delta_mag = np.abs(delta_p)
        
        coords = solutions["combined"].coords
        z = coords[:, 2]
        H = cfg.H
        
        # Slab statistics
        bottom_slab = z < 0.1 * H
        mid_slab = (z > 0.4 * H) & (z < 0.6 * H)
        top_slab = z > 0.9 * H
        
        diagnostics["delta"] = {
            "max_bottom": float(np.max(delta_mag[bottom_slab])) if np.any(bottom_slab) else 0,
            "max_mid": float(np.max(delta_mag[mid_slab])) if np.any(mid_slab) else 0,
            "max_top": float(np.max(delta_mag[top_slab])) if np.any(top_slab) else 0,
            "bulk_penetration_ratio": float(
                np.max(delta_mag[mid_slab]) / np.max(delta_mag[bottom_slab])
            ) if np.any(bottom_slab) and np.max(delta_mag[bottom_slab]) > 0 else 0,
        }
    
    # Streaming stats
    if streaming is not None:
        u_mag = streaming.u_mag
        diagnostics["streaming"] = {
            "max_speed": float(streaming.max_speed),
            "mean_speed": float(np.mean(u_mag)),
            "median_speed": float(np.median(u_mag)),
        }
    
    # Gor'kov stats
    if gorkov is not None:
        U = gorkov.U_values
        diagnostics["gorkov"] = {
            "min_potential": float(np.min(U)),
            "max_potential": float(np.max(U)),
            "trap_depth": float(gorkov.trap_depth),
        }
        
        if gorkov.F_function is not None:
            F_mag = np.linalg.norm(gorkov.F_values, axis=1)
            diagnostics["gorkov"]["max_force"] = float(np.max(F_mag))
            diagnostics["gorkov"]["mean_force"] = float(np.mean(F_mag))
            
            # Characteristic velocity from force
            v_char = cfg.stokes_mobility * np.max(F_mag)
            diagnostics["gorkov"]["characteristic_velocity"] = float(v_char)
    
    # Particle stats
    if trajectories is not None and len(trajectories) > 0:
        displacements = [t.displacement for t in trajectories]
        path_lengths = [t.path_length for t in trajectories]
        
        diagnostics["particles"] = {
            "n_particles": len(trajectories),
            "mean_displacement": float(np.mean(displacements)),
            "max_displacement": float(np.max(displacements)),
            "mean_path_length": float(np.mean(path_lengths)),
        }
    
    return diagnostics


def write_paraview_readme(output_dir: Path, cfg: ShallowDishConfig) -> Path:
    """Write ParaView visualization instructions."""
    output_dir = Path(output_dir)
    readme_path = output_dir / "PARAVIEW_README.md"
    
    content = f"""# ParaView Visualization Guide

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}

---

## Configuration Summary

- **Domain**: {cfg.L*1e3:.1f} mm × {cfg.L*1e3:.1f} mm × {cfg.H*1e3:.2f} mm
- **Frequency**: {cfg.frequency_hz/1e3:.0f} kHz
- **Wavelength**: {cfg.wavelength*1e3:.2f} mm
- **Vortex charge**: ℓ = {cfg.vortex_topological_charge}

---

## Files

| File | Description | Key Arrays |
|------|-------------|------------|
| `mesh.xdmf` | Mesh geometry | - |
| `standing_fields.vtu` | Standing wave only | p_real, p_imag, p_mag, p_phase |
| `vortex_fields.vtu` | Vortex only | p_real, p_imag, p_mag, p_phase |
| `combined_fields.vtu` | Standing + vortex | p_real, p_imag, p_mag, p_phase |
| `delta_fields.vtu` | Δp = combined - standing | delta_p_real, delta_p_imag, delta_p_mag, delta_p_phase |
| `streaming_fields.vtu` | Acoustic streaming | streaming_velocity (vector) |
| `gorkov_fields.vtu` | Radiation force | U_gorkov, F_rad (vector) |
| `particles.csv` | Particle trajectories | time, x_m, y_m, z_m |

---

## Visualization Steps

### 1. Combined Pressure Field

1. **File → Open** → `combined_fields.vtu`
2. Click **Apply**
3. Color by `p_mag`
4. Use **Slice** filter at z = {cfg.H/2*1e3:.2f} mm to see horizontal pattern

### 2. Streaming (Tornado-like Flow)

1. **File → Open** → `streaming_fields.vtu`
2. Click **Apply**
3. **Filters → Stream Tracer**
   - Seed Type: Point Cloud
   - Center: ({cfg.L/2*1e3:.1f}, {cfg.L/2*1e3:.1f}, {cfg.H*0.2*1e3:.2f}) mm
   - Radius: {cfg.L*0.1*1e3:.1f} mm
4. Color by magnitude

### 3. Gor'kov Traps

1. Load `gorkov_fields.vtu`
2. **Contour** by `U_gorkov` at low values
3. These show trap locations

### 4. Particle Trajectories

1. Load `particles.csv` as Table
2. **Table To Points** filter
3. **Tube** filter for visibility

---

## Recommended Colormaps

| Field | Colormap |
|-------|----------|
| p_mag | viridis |
| p_phase | twilight (cyclic, -π to π) |
| delta_p_mag | plasma |
| streaming_velocity magnitude | inferno |
| U_gorkov | coolwarm (centered on 0) |

---

## Physics Checks

1. **Standing pattern**: Look for lateral nodal planes
2. **Vortex signature**: Phase winding in delta_p_phase
3. **Bulk penetration**: Compare delta_p_mag at bottom vs top
4. **Streaming vortex**: Streamlines should show recirculation

---

*Generated by acousto-tweezers shallow dish experiment*
"""
    
    with open(readme_path, 'w') as f:
        f.write(content)
    
    return readme_path


def export_all_fields(
    output_dir: Path,
    cfg: ShallowDishConfig,
    solutions: Dict[str, PressureSolution],
    streaming: Optional[StreamingSolution] = None,
    gorkov: Optional[GorkovField] = None,
    trajectories: Optional[List[ParticleTrajectory]] = None,
    verbose: bool = True,
) -> Dict[str, Path]:
    """
    Export all fields to ParaView-ready format.
    
    Parameters
    ----------
    output_dir : Path
        Output directory
    cfg : ShallowDishConfig
        Configuration
    solutions : dict
        {"standing": ..., "vortex": ..., "combined": ...}
    streaming : StreamingSolution, optional
        Streaming velocity
    gorkov : GorkovField, optional
        Gor'kov potential
    trajectories : list, optional
        Particle trajectories
    verbose : bool
        Print progress
        
    Returns
    -------
    dict
        Dictionary of exported file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(exist_ok=True)
    
    if verbose:
        print(f"\n{'='*70}")
        print("EXPORTING PARAVIEW DATA")
        print(f"{'='*70}")
        print(f"  Output: {output_dir}")
    
    exported = {}
    
    # Mesh
    if "combined" in solutions:
        domain = solutions["combined"].p_function.function_space.mesh
        exported["mesh"] = export_mesh(domain, output_dir, verbose)
    
    # Pressure fields
    for mode, sol in solutions.items():
        name = f"{mode}_fields"
        exported[name] = export_pressure_fields(sol, output_dir, name, verbose=verbose)
    
    # Delta fields
    if "combined" in solutions and "standing" in solutions:
        exported["delta"] = export_delta_fields(
            solutions["combined"], solutions["standing"], output_dir, verbose
        )
    
    # Streaming
    if streaming is not None:
        exported["streaming"] = export_streaming_fields(streaming, output_dir, verbose)
    
    # Gor'kov
    if gorkov is not None:
        exported["gorkov"] = export_gorkov_fields(gorkov, output_dir, verbose)
    
    # Particles
    if trajectories is not None and len(trajectories) > 0:
        csv_path = output_dir / "particles.csv"
        save_trajectories_csv(trajectories, str(csv_path))
        exported["particles"] = csv_path
        if verbose:
            print(f"  Exported: particles.csv ({len(trajectories)} trajectories)")
    
    # Diagnostics
    diagnostics = compute_diagnostics(cfg, solutions, streaming, gorkov, trajectories)
    diag_path = meta_dir / "diagnostics.json"
    with open(diag_path, 'w') as f:
        json.dump(diagnostics, f, indent=2)
    exported["diagnostics"] = diag_path
    if verbose:
        print(f"  Exported: meta/diagnostics.json")
    
    # Config
    config_path = meta_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(cfg.to_dict(), f, indent=2)
    exported["config"] = config_path
    if verbose:
        print(f"  Exported: meta/config.json")
    
    # ParaView README
    readme_path = write_paraview_readme(output_dir, cfg)
    exported["readme"] = readme_path
    if verbose:
        print(f"  Exported: PARAVIEW_README.md")
    
    if verbose:
        print(f"{'='*70}\n")
    
    return exported
