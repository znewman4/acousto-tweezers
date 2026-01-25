"""
Gmsh geometry generation for FEniCSx acoustic tweezers simulation.

Creates a 3D Petri dish geometry with all required domains and interfaces:
- Volumetric domains: water, air, bath, plate, wall, lens, PML regions
- Interface surfaces for coupling conditions

The geometry is created using the Gmsh Python API and imported into
DOLFINx using dolfinx.io.gmsh.

Author: Acousto-Tweezers Project
Date: January 2026
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

import gmsh
from mpi4py import MPI

from .config import FEMConfig, GeometryConfig
from .domains import Domain, Interface


@dataclass
class MeshInfo:
    """Information about the generated mesh."""
    num_nodes: int
    num_cells: int
    domain_counts: Dict[str, int]
    interface_counts: Dict[str, int]
    min_element_size: float
    max_element_size: float
    mesh_file: Optional[str] = None


def create_petri_dish_geometry(config: FEMConfig, 
                                output_dir: Optional[str] = None,
                                verbose: bool = True) -> Tuple[any, any, any, MeshInfo]:
    """
    Create 3D Petri dish geometry with Gmsh.
    
    Creates volumetric domains for all physics regions and tags
    interfaces for boundary conditions and coupling.
    
    Parameters
    ----------
    config : FEMConfig
        Simulation configuration
    output_dir : str, optional
        Directory to save mesh files
    verbose : bool
        Print progress information
        
    Returns
    -------
    mesh : dolfinx.mesh.Mesh
        The DOLFINx mesh
    cell_tags : dolfinx.mesh.MeshTags
        Cell (domain) tags
    facet_tags : dolfinx.mesh.MeshTags
        Facet (interface) tags
    mesh_info : MeshInfo
        Mesh statistics
    """
    from dolfinx.io import gmsh as gmshio
    
    geo = config.geometry
    physics_level = config.physics_level
    
    # Initialize Gmsh
    gmsh.initialize()
    if not verbose:
        gmsh.option.setNumber("General.Terminal", 0)
    
    gmsh.model.add("petri_dish")
    
    # Compute wavelength for mesh sizing
    omega = config.physics.omega
    c_water = 1480.0  # Approximate sound speed in water
    wavelength = 2 * np.pi * c_water / omega
    
    # Mesh size based on elements per wavelength
    lc = min(
        wavelength / geo.elements_per_wavelength,
        geo.max_element_size
    )
    lc = max(lc, geo.min_element_size)
    
    if verbose:
        print(f"Wavelength: {wavelength*1e6:.1f} μm")
        print(f"Target element size: {lc*1e6:.1f} μm")
    
    # =========================================================
    # GEOMETRY CONSTRUCTION
    # =========================================================
    
    # Key dimensions
    R_inner = geo.dish_inner_radius      # Inner radius of dish
    R_outer = geo.dish_outer_radius      # Outer radius (with wall)
    t_wall = geo.dish_wall_thickness     # Wall thickness
    t_plate = geo.dish_bottom_thickness  # Plate thickness
    h_water = geo.water_depth            # Water depth
    h_air = geo.air_height               # Air height above water
    h_bath = geo.bath_depth              # Bath depth below plate
    L_pml = geo.pml_thickness            # PML thickness
    
    # Domain extents in z
    z_bath_bottom = -h_bath - t_plate - L_pml
    z_plate_bottom = -t_plate
    z_plate_top = 0.0
    z_water_top = h_water
    z_air_top = h_water + h_air
    z_pml_top = z_air_top + L_pml
    
    # Domain extents in r (radial, we'll make it 3D)
    r_pml_outer = R_outer + L_pml
    
    # For 3D, we'll create a rectangular domain with cylindrical dish
    # Using symmetry: model quarter geometry if desired, or full
    # Here we do full 3D for generality
    
    # Create the geometry using OpenCASCADE kernel for boolean ops
    gmsh.model.occ.synchronize()
    
    # =========================================================
    # CREATE VOLUMETRIC DOMAINS
    # =========================================================
    
    # We'll build this in layers from bottom to top
    volumes = {}
    
    # --- PML Bottom layer ---
    if physics_level.value >= 2:
        box_pml_bottom = gmsh.model.occ.addBox(
            -r_pml_outer, -r_pml_outer, z_bath_bottom,
            2*r_pml_outer, 2*r_pml_outer, L_pml
        )
        volumes['pml_bottom'] = box_pml_bottom
    
    # --- Bath layer (below plate) ---
    z_bath_top = z_plate_bottom
    if physics_level.value >= 3:
        box_bath = gmsh.model.occ.addBox(
            -r_pml_outer, -r_pml_outer, z_bath_bottom + L_pml,
            2*r_pml_outer, 2*r_pml_outer, h_bath
        )
        volumes['bath'] = box_bath
    
    # --- Plate (solid, volumetric) ---
    if physics_level.value >= 4:
        # Plate is a cylinder at the bottom of the dish
        cyl_plate = gmsh.model.occ.addCylinder(
            0, 0, z_plate_bottom,  # Base center
            0, 0, t_plate,         # Height direction
            R_outer                # Radius (full dish including wall base)
        )
        volumes['plate'] = cyl_plate
    
    # --- Wall (solid, volumetric) ---
    if physics_level.value >= 4:
        # Wall is an annular cylinder
        cyl_wall_outer = gmsh.model.occ.addCylinder(
            0, 0, z_plate_top,
            0, 0, h_water + h_air,
            R_outer
        )
        cyl_wall_inner = gmsh.model.occ.addCylinder(
            0, 0, z_plate_top,
            0, 0, h_water + h_air,
            R_inner
        )
        wall_cut = gmsh.model.occ.cut(
            [(3, cyl_wall_outer)], 
            [(3, cyl_wall_inner)],
            removeObject=True, removeTool=False
        )
        volumes['wall'] = wall_cut[0][0][1]
        # Keep inner cylinder for water
        cyl_water_air = cyl_wall_inner
    else:
        # Without solids, water cylinder is created directly
        # No separate cyl_water_air needed - we just create water
        # (and air separately if physics level requires it)
        pass
    
    # --- Water domain ---
    cyl_water = gmsh.model.occ.addCylinder(
        0, 0, z_plate_top,
        0, 0, h_water,
        R_inner
    )
    volumes['water'] = cyl_water
    
    # --- Air domain (only for level 3+) ---
    if physics_level.value >= 3:
        cyl_air = gmsh.model.occ.addCylinder(
            0, 0, z_water_top,
            0, 0, h_air,
            R_inner
        )
        volumes['air'] = cyl_air
    
    # --- PML regions (top, sides) ---
    if physics_level.value >= 2:
        # Top PML
        box_pml_top = gmsh.model.occ.addBox(
            -r_pml_outer, -r_pml_outer, z_air_top,
            2*r_pml_outer, 2*r_pml_outer, L_pml
        )
        volumes['pml_top'] = box_pml_top
        
        # Side PML (shell around the domain)
        # This is more complex - we'll create it as the difference of boxes
        box_outer = gmsh.model.occ.addBox(
            -r_pml_outer, -r_pml_outer, z_bath_bottom + L_pml,
            2*r_pml_outer, 2*r_pml_outer, z_air_top - z_bath_bottom - L_pml
        )
        box_inner = gmsh.model.occ.addBox(
            -(R_outer + 0.001), -(R_outer + 0.001), z_bath_bottom + L_pml,
            2*(R_outer + 0.001), 2*(R_outer + 0.001), z_air_top - z_bath_bottom - L_pml
        )
        pml_side_cut = gmsh.model.occ.cut(
            [(3, box_outer)],
            [(3, box_inner)],
            removeObject=True, removeTool=True
        )
        if pml_side_cut[0]:
            volumes['pml_side'] = pml_side_cut[0][0][1]
    
    gmsh.model.occ.synchronize()
    
    # =========================================================
    # FRAGMENT ALL VOLUMES FOR CONFORMING MESH
    # =========================================================
    
    all_volumes = [(3, v) for v in volumes.values() if isinstance(v, int)]
    if len(all_volumes) > 1:
        gmsh.model.occ.fragment(all_volumes, [])
        gmsh.model.occ.synchronize()
    
    # =========================================================
    # ASSIGN PHYSICAL GROUPS (DOMAIN TAGS)
    # =========================================================
    
    # Get all volumes after fragmentation
    all_vols = gmsh.model.getEntities(dim=3)
    
    # Function to determine domain from volume centroid
    def get_domain_from_centroid(vol_tag: int) -> Domain:
        """Determine domain type from volume centroid location."""
        bbox = gmsh.model.getBoundingBox(3, vol_tag)
        cx = (bbox[0] + bbox[3]) / 2
        cy = (bbox[1] + bbox[4]) / 2
        cz = (bbox[2] + bbox[5]) / 2
        r = np.sqrt(cx**2 + cy**2)
        
        # Check PML regions first (outermost) - only if PML enabled
        if physics_level.value >= 2:
            if cz < z_bath_bottom + L_pml + 0.001:
                return Domain.PML_BOTTOM
            if cz > z_air_top - 0.001:
                return Domain.PML_TOP
            if r > R_outer + 0.001:
                return Domain.PML_WATER  # Side PML
        
        # Bath region - only if enabled
        if physics_level.value >= 3:
            if cz < z_plate_bottom - 0.001:
                return Domain.BATH
        
        # Plate region - only if solids enabled
        if physics_level.value >= 4:
            if z_plate_bottom - 0.001 < cz < z_plate_top + 0.001 and r < R_outer:
                return Domain.PLATE
        
        # Wall region (annular) - only if solids enabled
        if physics_level.value >= 4:
            if cz > z_plate_top - 0.001 and R_inner < r < R_outer:
                return Domain.WALL
        
        # Air region - only if air domain enabled
        if physics_level.value >= 3:
            if z_water_top - 0.001 < cz < z_air_top + 0.001 and r < R_inner:
                return Domain.AIR
        
        # Water region - the default for level 1 and 2
        # For higher levels, need to check z range
        if r < R_inner + 0.001:  # Inside dish
            if physics_level.value >= 3:
                # With air, water is below water surface
                if z_plate_top - 0.001 < cz < z_water_top + 0.001:
                    return Domain.WATER
            else:
                # Without air, everything inside is water
                return Domain.WATER
        
        # Default to water for unclassified (shouldn't happen)
        return Domain.WATER
    
    # Create physical groups for each domain
    domain_volumes: Dict[Domain, list] = {}
    for dim, tag in all_vols:
        domain = get_domain_from_centroid(tag)
        if domain not in domain_volumes:
            domain_volumes[domain] = []
        domain_volumes[domain].append(tag)
    
    for domain, vol_tags in domain_volumes.items():
        pg = gmsh.model.addPhysicalGroup(3, vol_tags, domain.gmsh_tag)
        gmsh.model.setPhysicalName(3, pg, domain.name)
        if verbose:
            print(f"Domain {domain.name}: {len(vol_tags)} volumes, tag={domain.gmsh_tag}")
    
    # =========================================================
    # ASSIGN INTERFACE TAGS
    # =========================================================
    
    all_surfs = gmsh.model.getEntities(dim=2)
    
    def get_interface_from_surface(surf_tag: int) -> Optional[Interface]:
        """Determine interface type from surface properties."""
        bbox = gmsh.model.getBoundingBox(2, surf_tag)
        cx = (bbox[0] + bbox[3]) / 2
        cy = (bbox[1] + bbox[4]) / 2
        cz = (bbox[2] + bbox[5]) / 2
        r = np.sqrt(cx**2 + cy**2)
        
        zmin, zmax = bbox[2], bbox[5]
        
        # Water-air interface (horizontal at z = water_depth)
        if abs(cz - z_water_top) < 0.001 and r < R_inner:
            return Interface.WATER_AIR
        
        # Water-plate interface (bottom of water)
        if abs(cz - z_plate_top) < 0.001 and r < R_inner:
            return Interface.WATER_PLATE
        
        # Outer boundary
        if r > r_pml_outer - 0.001:
            return Interface.PML_OUTER
        
        # Bath-plate interface
        if abs(cz - z_plate_bottom) < 0.001 and r < R_outer:
            return Interface.BATH_PLATE
        
        return None
    
    interface_surfaces: Dict[Interface, list] = {}
    for dim, tag in all_surfs:
        interface = get_interface_from_surface(tag)
        if interface is not None:
            if interface not in interface_surfaces:
                interface_surfaces[interface] = []
            interface_surfaces[interface].append(tag)
    
    for interface, surf_tags in interface_surfaces.items():
        pg = gmsh.model.addPhysicalGroup(2, surf_tags, interface.gmsh_tag)
        gmsh.model.setPhysicalName(2, pg, interface.name)
        if verbose:
            print(f"Interface {interface.name}: {len(surf_tags)} surfaces, tag={interface.gmsh_tag}")
    
    # =========================================================
    # MESH GENERATION
    # =========================================================
    
    # Set mesh size
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc * 0.5)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc * 2.0)
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # Delaunay
    
    # Generate mesh
    if verbose:
        print("Generating 3D mesh...")
    gmsh.model.mesh.generate(3)
    
    # Get mesh statistics
    node_tags, _, _ = gmsh.model.mesh.getNodes()
    elem_types, elem_tags, _ = gmsh.model.mesh.getElements(dim=3)
    num_cells = sum(len(et) for et in elem_tags)
    
    # Mesh quality
    gmsh.model.mesh.optimize("Netgen")
    
    # Save mesh file if requested
    mesh_file = None
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        mesh_file = str(output_path / "mesh.msh")
        gmsh.write(mesh_file)
        if verbose:
            print(f"Mesh saved to: {mesh_file}")
    
    # =========================================================
    # IMPORT TO DOLFINX
    # =========================================================
    
    if verbose:
        print("Importing mesh to DOLFINx...")
    
    # New DOLFINx API returns MeshData object
    mesh_data = gmshio.model_to_mesh(
        gmsh.model, 
        MPI.COMM_WORLD, 
        rank=0,
        gdim=3
    )
    mesh = mesh_data.mesh
    cell_tags = mesh_data.cell_tags
    facet_tags = mesh_data.facet_tags
    
    # Gather mesh info
    domain_counts = {}
    for domain in Domain:
        count = np.sum(cell_tags.values == domain.gmsh_tag)
        if count > 0:
            domain_counts[domain.name] = int(count)
    
    interface_counts = {}
    for interface in Interface:
        count = np.sum(facet_tags.values == interface.gmsh_tag)
        if count > 0:
            interface_counts[interface.name] = int(count)
    
    mesh_info = MeshInfo(
        num_nodes=len(node_tags),
        num_cells=num_cells,
        domain_counts=domain_counts,
        interface_counts=interface_counts,
        min_element_size=lc * 0.5,
        max_element_size=lc * 2.0,
        mesh_file=mesh_file,
    )
    
    gmsh.finalize()
    
    if verbose:
        print(f"Mesh created: {mesh_info.num_nodes} nodes, {mesh_info.num_cells} cells")
        print(f"Domains: {mesh_info.domain_counts}")
    
    return mesh, cell_tags, facet_tags, mesh_info


def create_simple_box_mesh(Lx: float, Ly: float, Lz: float,
                           nx: int, ny: int, nz: int,
                           comm=None) -> Tuple[any, any]:
    """
    Create a simple box mesh for testing.
    
    Parameters
    ----------
    Lx, Ly, Lz : float
        Box dimensions
    nx, ny, nz : int
        Number of elements in each direction
    comm : MPI communicator
        MPI communicator (default: COMM_WORLD)
        
    Returns
    -------
    mesh : dolfinx.mesh.Mesh
        The mesh
    cell_tags : dolfinx.mesh.MeshTags
        All cells tagged as WATER domain
    """
    from dolfinx import mesh as dmesh
    from dolfinx.mesh import create_box, CellType, meshtags
    
    if comm is None:
        comm = MPI.COMM_WORLD
    
    mesh = create_box(
        comm,
        [[0.0, 0.0, 0.0], [Lx, Ly, Lz]],
        [nx, ny, nz],
        CellType.tetrahedron
    )
    
    # Tag all cells as WATER
    num_cells = mesh.topology.index_map(3).size_local
    cell_indices = np.arange(num_cells, dtype=np.int32)
    cell_values = np.full(num_cells, Domain.WATER.value, dtype=np.int32)
    
    mesh.topology.create_connectivity(3, 0)
    cell_tags = meshtags(mesh, 3, cell_indices, cell_values)
    
    return mesh, cell_tags


def create_2d_disk_mesh(radius: float, lc: float, 
                        comm=None) -> Tuple[any, any, any]:
    """
    Create a 2D disk mesh for axisymmetric testing.
    
    Parameters
    ----------
    radius : float
        Disk radius
    lc : float
        Characteristic mesh size
    comm : MPI communicator
        
    Returns
    -------
    mesh, cell_tags, facet_tags
    """
    from dolfinx.io import gmsh as gmshio
    
    if comm is None:
        comm = MPI.COMM_WORLD
    
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("disk")
    
    # Create disk
    disk = gmsh.model.occ.addDisk(0, 0, 0, radius, radius)
    gmsh.model.occ.synchronize()
    
    # Physical groups
    gmsh.model.addPhysicalGroup(2, [disk], Domain.WATER.value)
    
    # Boundary
    boundary = gmsh.model.getBoundary([(2, disk)], oriented=False)
    boundary_tags = [b[1] for b in boundary]
    gmsh.model.addPhysicalGroup(1, boundary_tags, Interface.OUTER.value)
    
    # Mesh
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc * 0.5)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc * 2.0)
    gmsh.model.mesh.generate(2)
    
    # New DOLFINx API returns MeshData object
    mesh_data = gmshio.model_to_mesh(
        gmsh.model, comm, rank=0, gdim=2
    )
    mesh = mesh_data.mesh
    cell_tags = mesh_data.cell_tags
    facet_tags = mesh_data.facet_tags
    
    gmsh.finalize()
    
    return mesh, cell_tags, facet_tags
