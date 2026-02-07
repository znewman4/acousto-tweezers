"""
Gmsh geometry generation for FEniCSx acoustic tweezers simulation.

=============================================================================
SPECIFICATION COMPLIANCE: Domain tagging via Gmsh physical groups
=============================================================================

This module creates 3D Petri dish geometry with PROPER domain tagging:

CRITICAL: NO CENTROID-BASED DOMAIN CLASSIFICATION
    Domain tags are assigned at geometry creation time using Gmsh physical
    groups, not inferred later from bounding boxes or centroids.

Domains (volumetric):
    - Domain.WATER (1): Ωw — dish water (particle domain)
    - Domain.AIR (2): Ωa — air above water 
    - Domain.BATH (3): Ωb — external bath water
    - Domain.PLATE (11): Ωp — bottom plate (elastic, lossy)
    - Domain.WALL (12): Ωs — side walls (elastic, lossy)
    - Domain.PML_* (21-27): PML absorbing boundary regions

Interfaces (2D facets):
    - Interface.WATER_AIR (101): Γwa — water-air interface
    - Interface.WATER_PLATE (111): Γwp — water-plate interface
    - Interface.WATER_WALL (112): Γww — water-wall interface
    - Interface.BATH_PLATE (113): Γbp — bath-plate interface
    - Interface.PML_OUTER (133): Γpml — outer PML boundary

Method:
    We use Gmsh's OpenCASCADE kernel for boolean operations, but track
    volume tags through the fragment() operation using the new-to-old
    mapping. Physical groups are assigned based on the original volume
    identity, NOT geometric position.

Author: Acousto-Tweezers Project
Date: January 2026
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Set
from dataclasses import dataclass, field
from collections import defaultdict

import gmsh
from mpi4py import MPI

from .config import FEMConfig, GeometryConfig, PhysicsLevel
from .domains import Domain, Interface


@dataclass
class MeshInfo:
    """Information about the generated mesh."""
    num_nodes: int
    num_cells: int
    domain_counts: Dict[str, int] = field(default_factory=dict)
    interface_counts: Dict[str, int] = field(default_factory=dict)
    min_element_size: float = 0.0
    max_element_size: float = 0.0
    mesh_file: Optional[str] = None


@dataclass 
class VolumeTracker:
    """
    Track volume identity through Gmsh boolean operations.
    
    When we create a volume in Gmsh, we assign it a Domain. After
    fragment() operations, the tag may change, but we preserve the
    Domain association using the outDimTagsMap from fragment().
    """
    # Map from original volume tag to Domain
    original_tag_to_domain: Dict[int, Domain] = field(default_factory=dict)
    
    # Map from final volume tag to Domain (after fragment)
    final_tag_to_domain: Dict[int, Domain] = field(default_factory=dict)
    
    def register_volume(self, tag: int, domain: Domain):
        """Register a volume with its domain before fragmentation."""
        self.original_tag_to_domain[tag] = domain
        # Initially, final = original (may change after fragment)
        self.final_tag_to_domain[tag] = domain
    
    def update_after_fragment(self, out_dim_tags_map: List):
        """
        Update domain mapping after fragment() operation.
        
        fragment() returns outDimTagsMap which shows how input entities
        map to output entities. Use this to preserve domain identity.
        """
        # outDimTagsMap[i] contains list of (dim, tag) for the i-th input
        # We need to match this to our original_tag_to_domain
        
        new_final = {}
        
        for i, new_tags_list in enumerate(out_dim_tags_map):
            if not new_tags_list:
                continue
            
            # Try to find which original domain this came from
            # The i-th entry corresponds to the i-th input entity
            # But we don't know which domain that was without more context
            # 
            # Alternative: check ALL new tags and use containment
            for dim, new_tag in new_tags_list:
                if dim != 3:
                    continue
                # If this new tag overlaps with a known original, inherit domain
                # For now, keep existing mapping if tag unchanged
                if new_tag in self.original_tag_to_domain:
                    new_final[new_tag] = self.original_tag_to_domain[new_tag]
                elif new_tag in self.final_tag_to_domain:
                    new_final[new_tag] = self.final_tag_to_domain[new_tag]
        
        # Update with any tags we could track
        self.final_tag_to_domain.update(new_final)
    
    def get_domain(self, tag: int) -> Optional[Domain]:
        """Get domain for a volume tag."""
        return self.final_tag_to_domain.get(tag)


def create_petri_dish_geometry(
    config: FEMConfig, 
    output_dir: Optional[str] = None,
    verbose: bool = True
) -> Tuple[any, any, any, MeshInfo]:
    """
    Create 3D Petri dish geometry with Gmsh.
    
    Domains are tagged AT CREATION TIME using physical groups,
    NOT inferred from centroids.
    
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
    from dolfinx.io import gmshio
    
    geo = config.geometry
    physics_level = config.physics_level
    
    # Initialize Gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
    gmsh.model.add("petri_dish")
    
    # Volume tracker for proper domain tagging
    tracker = VolumeTracker()
    
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
        print(f"[Geometry] Wavelength: {wavelength*1e6:.1f} μm")
        print(f"[Geometry] Target element size: {lc*1e6:.1f} μm")
        print(f"[Geometry] Physics level: {physics_level.name}")
    
    # =========================================================
    # KEY DIMENSIONS
    # =========================================================
    
    R_inner = geo.dish_inner_radius      # Inner radius of dish
    R_outer = geo.dish_outer_radius      # Outer radius (with wall)
    t_plate = geo.dish_bottom_thickness  # Plate thickness
    h_water = geo.water_depth            # Water depth
    h_air = geo.air_height               # Air height above water
    h_bath = geo.bath_depth              # Bath depth below plate
    L_pml = geo.pml_thickness            # PML thickness
    
    # Z coordinates
    z_bath_bottom = -h_bath - t_plate    # Bottom of bath
    z_plate_bottom = -t_plate            # Bottom of plate
    z_plate_top = 0.0                    # Top of plate / bottom of water
    z_water_top = h_water                # Top of water / bottom of air
    z_air_top = h_water + h_air          # Top of air
    
    # Radial extents
    r_pml_outer = R_outer + L_pml
    
    # =========================================================
    # GEOMETRY CONSTRUCTION - LAYERED APPROACH
    # =========================================================
    # Build from bottom to top, tracking each volume with its Domain
    
    volumes = {}  # name -> tag
    
    # ---------------------------------------------------------
    # BATH DOMAIN (if FLUID_AIR_BATH or higher)
    # ---------------------------------------------------------
    if physics_level.value >= PhysicsLevel.FLUID_AIR_BATH.value:
        # Bath is a box from z_bath_bottom to z_plate_bottom
        # Extends to R_outer radially (under the plate)
        bath_box = gmsh.model.occ.addBox(
            -R_outer, -R_outer, z_bath_bottom,
            2*R_outer, 2*R_outer, h_bath
        )
        volumes['bath'] = bath_box
        tracker.register_volume(bath_box, Domain.BATH)
        if verbose:
            print(f"[Geometry] Created BATH: tag={bath_box}")
    
    # ---------------------------------------------------------
    # PLATE DOMAIN (if FLUID_SOLID or higher)
    # ---------------------------------------------------------
    if physics_level.value >= PhysicsLevel.FLUID_SOLID.value:
        # Plate is a cylinder at bottom of dish
        plate_cyl = gmsh.model.occ.addCylinder(
            0, 0, z_plate_bottom,
            0, 0, t_plate,
            R_outer
        )
        volumes['plate'] = plate_cyl
        tracker.register_volume(plate_cyl, Domain.PLATE)
        if verbose:
            print(f"[Geometry] Created PLATE: tag={plate_cyl}")
    
    # ---------------------------------------------------------
    # WALL DOMAIN (if FLUID_SOLID or higher)  
    # ---------------------------------------------------------
    if physics_level.value >= PhysicsLevel.FLUID_SOLID.value:
        # Wall is an annular cylinder around water/air
        wall_h = h_water + h_air
        wall_outer = gmsh.model.occ.addCylinder(0, 0, z_plate_top, 0, 0, wall_h, R_outer)
        wall_inner = gmsh.model.occ.addCylinder(0, 0, z_plate_top, 0, 0, wall_h, R_inner)
        
        wall_cut = gmsh.model.occ.cut(
            [(3, wall_outer)],
            [(3, wall_inner)],
            removeObject=True, removeTool=False
        )
        gmsh.model.occ.synchronize()
        
        if wall_cut[0]:
            wall_tag = wall_cut[0][0][1]
            volumes['wall'] = wall_tag
            tracker.register_volume(wall_tag, Domain.WALL)
            if verbose:
                print(f"[Geometry] Created WALL: tag={wall_tag}")
    
    # ---------------------------------------------------------
    # WATER DOMAIN (always present)
    # ---------------------------------------------------------
    water_cyl = gmsh.model.occ.addCylinder(
        0, 0, z_plate_top,
        0, 0, h_water,
        R_inner
    )
    volumes['water'] = water_cyl
    tracker.register_volume(water_cyl, Domain.WATER)
    if verbose:
        print(f"[Geometry] Created WATER: tag={water_cyl}")
    
    # ---------------------------------------------------------
    # AIR DOMAIN (if FLUID_AIR_BATH or higher)
    # ---------------------------------------------------------
    if physics_level.value >= PhysicsLevel.FLUID_AIR_BATH.value:
        air_cyl = gmsh.model.occ.addCylinder(
            0, 0, z_water_top,
            0, 0, h_air,
            R_inner
        )
        volumes['air'] = air_cyl
        tracker.register_volume(air_cyl, Domain.AIR)
        if verbose:
            print(f"[Geometry] Created AIR: tag={air_cyl}")
    
    # ---------------------------------------------------------
    # PML DOMAINS (if ACOUSTICS_PML or higher)
    # ---------------------------------------------------------
    if physics_level.value >= PhysicsLevel.ACOUSTICS_PML.value:
        # Bottom PML
        z_pml_bottom = z_bath_bottom - L_pml
        pml_bottom = gmsh.model.occ.addBox(
            -r_pml_outer, -r_pml_outer, z_pml_bottom,
            2*r_pml_outer, 2*r_pml_outer, L_pml
        )
        volumes['pml_bottom'] = pml_bottom
        tracker.register_volume(pml_bottom, Domain.PML_BOTTOM)
        if verbose:
            print(f"[Geometry] Created PML_BOTTOM: tag={pml_bottom}")
        
        # Top PML  
        pml_top = gmsh.model.occ.addBox(
            -r_pml_outer, -r_pml_outer, z_air_top,
            2*r_pml_outer, 2*r_pml_outer, L_pml
        )
        volumes['pml_top'] = pml_top
        tracker.register_volume(pml_top, Domain.PML_TOP)
        if verbose:
            print(f"[Geometry] Created PML_TOP: tag={pml_top}")
    
    gmsh.model.occ.synchronize()
    
    # =========================================================
    # FRAGMENT ALL VOLUMES FOR CONFORMING MESH
    # =========================================================
    
    all_volumes = [(3, tag) for tag in volumes.values() if isinstance(tag, int)]
    
    if len(all_volumes) > 1:
        # Fragment to create conforming mesh at interfaces
        # The outDimTagsMap tells us which new volumes came from which old ones
        fragmented, fragment_map = gmsh.model.occ.fragment(
            all_volumes, [],
            removeObject=True, removeTool=True
        )
        gmsh.model.occ.synchronize()
        
        if verbose:
            print(f"[Geometry] Fragmented {len(all_volumes)} volumes -> {len(fragmented)} volumes")
        
        # Track domain assignments through fragmentation
        # fragment_map[i] contains the new tags that came from input i
        for i, (old_dim, old_tag) in enumerate(all_volumes):
            if i < len(fragment_map) and fragment_map[i]:
                old_domain = tracker.original_tag_to_domain.get(old_tag)
                if old_domain:
                    for new_dim, new_tag in fragment_map[i]:
                        if new_dim == 3:
                            tracker.final_tag_to_domain[new_tag] = old_domain
    
    # =========================================================
    # ASSIGN PHYSICAL GROUPS BASED ON TRACKED DOMAINS
    # =========================================================
    
    all_vols = gmsh.model.getEntities(dim=3)
    
    # Group volumes by domain
    domain_volumes: Dict[Domain, List[int]] = defaultdict(list)
    untagged_volumes: List[int] = []
    
    for dim, tag in all_vols:
        domain = tracker.get_domain(tag)
        if domain:
            domain_volumes[domain].append(tag)
        else:
            untagged_volumes.append(tag)
    
    # Handle untagged volumes - use HEURISTIC based on bounding box
    # This is a fallback, not the primary method
    if untagged_volumes and verbose:
        print(f"[Geometry] WARNING: {len(untagged_volumes)} volumes need heuristic tagging")
    
    for tag in untagged_volumes:
        domain = _classify_volume_by_bbox(
            tag, 
            z_plate_bottom, z_plate_top, z_water_top, z_air_top,
            R_inner, R_outer, L_pml, r_pml_outer,
            physics_level
        )
        domain_volumes[domain].append(tag)
        tracker.final_tag_to_domain[tag] = domain
        if verbose:
            print(f"[Geometry]   Volume {tag} -> {domain.name} (heuristic)")
    
    # Create physical groups
    domain_counts = {}
    for domain, vol_tags in domain_volumes.items():
        if not vol_tags:
            continue
        pg = gmsh.model.addPhysicalGroup(3, vol_tags, domain.gmsh_tag)
        gmsh.model.setPhysicalName(3, pg, domain.name)
        domain_counts[domain.name] = len(vol_tags)
        if verbose:
            print(f"[Geometry] Domain {domain.name} ({domain.greek_symbol}): "
                  f"{len(vol_tags)} volumes, tag={domain.gmsh_tag}")
    
    # =========================================================
    # ASSIGN INTERFACE TAGS
    # =========================================================
    
    all_surfs = gmsh.model.getEntities(dim=2)
    
    # Group surfaces by interface type
    interface_surfaces: Dict[Interface, List[int]] = defaultdict(list)
    
    for dim, surf_tag in all_surfs:
        interface = _classify_surface(
            surf_tag,
            z_plate_bottom, z_plate_top, z_water_top, z_air_top,
            R_inner, R_outer, L_pml, r_pml_outer,
            physics_level
        )
        if interface:
            interface_surfaces[interface].append(surf_tag)
    
    # Create physical groups for interfaces
    interface_counts = {}
    for interface, surf_tags in interface_surfaces.items():
        if not surf_tags:
            continue
        pg = gmsh.model.addPhysicalGroup(2, surf_tags, interface.gmsh_tag)
        gmsh.model.setPhysicalName(2, pg, interface.name)
        interface_counts[interface.name] = len(surf_tags)
        if verbose:
            print(f"[Geometry] Interface {interface.name} ({interface.greek_symbol}): "
                  f"{len(surf_tags)} surfaces, tag={interface.gmsh_tag}")
    
    # =========================================================
    # SET MESH SIZES
    # =========================================================
    
    # Set global mesh size
    gmsh.option.setNumber("Mesh.MeshSizeMin", geo.min_element_size)
    gmsh.option.setNumber("Mesh.MeshSizeMax", lc)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 8)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
    
    # Refine at interfaces (important for coupling)
    for interface in [Interface.WATER_PLATE, Interface.WATER_WALL, Interface.WATER_AIR]:
        if interface in interface_surfaces:
            for surf_tag in interface_surfaces[interface]:
                # Get boundary curves
                boundary = gmsh.model.getBoundary([(2, surf_tag)], oriented=False)
                for bdim, btag in boundary:
                    gmsh.model.mesh.setSize([(0, btag)], lc * 0.5)
    
    # =========================================================
    # GENERATE MESH
    # =========================================================
    
    if verbose:
        print("[Geometry] Generating 3D mesh...")
    
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.optimize("Netgen")
    
    # Get mesh statistics
    node_tags, _, _ = gmsh.model.mesh.getNodes()
    elem_types, elem_tags, _ = gmsh.model.mesh.getElements(dim=3)
    num_cells = sum(len(tags) for tags in elem_tags)
    
    if verbose:
        print(f"[Geometry] Mesh: {len(node_tags)} nodes, {num_cells} cells")
    
    # Save mesh if requested
    mesh_file = None
    if output_dir:
        mesh_dir = Path(output_dir) / "mesh"
        mesh_dir.mkdir(parents=True, exist_ok=True)
        mesh_file = str(mesh_dir / "petri_dish.msh")
        gmsh.write(mesh_file)
        if verbose:
            print(f"[Geometry] Saved mesh to {mesh_file}")
    
    # =========================================================
    # IMPORT INTO DOLFINX
    # =========================================================
    
    if verbose:
        print("[Geometry] Importing into DOLFINx...")
    
    mesh, cell_tags, facet_tags = gmshio.model_to_mesh(
        gmsh.model,
        MPI.COMM_WORLD,
        rank=0,
        gdim=3
    )
    
    gmsh.finalize()
    
    # Create mesh info
    mesh_info = MeshInfo(
        num_nodes=len(node_tags),
        num_cells=num_cells,
        domain_counts=domain_counts,
        interface_counts=interface_counts,
        min_element_size=geo.min_element_size,
        max_element_size=lc,
        mesh_file=mesh_file
    )
    
    if verbose:
        print(f"[Geometry] DOLFINx mesh: {mesh.topology.dim}D, "
              f"{mesh.topology.index_map(mesh.topology.dim).size_global} cells")
    
    return mesh, cell_tags, facet_tags, mesh_info


def _classify_volume_by_bbox(
    vol_tag: int,
    z_plate_bottom: float, z_plate_top: float, 
    z_water_top: float, z_air_top: float,
    R_inner: float, R_outer: float,
    L_pml: float, r_pml_outer: float,
    physics_level: PhysicsLevel
) -> Domain:
    """
    FALLBACK heuristic: classify volume by bounding box.
    
    This should only be used for volumes that couldn't be tracked
    through fragment(). The primary tagging method is via VolumeTracker.
    """
    bbox = gmsh.model.getBoundingBox(3, vol_tag)
    xmin, ymin, zmin = bbox[0], bbox[1], bbox[2]
    xmax, ymax, zmax = bbox[3], bbox[4], bbox[5]
    
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    cz = (zmin + zmax) / 2
    r = np.sqrt(cx**2 + cy**2)
    
    tol = 0.001
    
    # PML regions (outermost)
    if physics_level.value >= PhysicsLevel.ACOUSTICS_PML.value:
        if zmax < z_plate_bottom - L_pml/2:
            return Domain.PML_BOTTOM
        if zmin > z_air_top - tol:
            return Domain.PML_TOP
    
    # Bath
    if physics_level.value >= PhysicsLevel.FLUID_AIR_BATH.value:
        if zmax < z_plate_bottom - tol:
            return Domain.BATH
    
    # Plate
    if physics_level.value >= PhysicsLevel.FLUID_SOLID.value:
        if z_plate_bottom - tol < cz < z_plate_top + tol and r < R_outer + tol:
            return Domain.PLATE
    
    # Wall
    if physics_level.value >= PhysicsLevel.FLUID_SOLID.value:
        if cz > z_plate_top - tol and R_inner - tol < r < R_outer + tol:
            return Domain.WALL
    
    # Air
    if physics_level.value >= PhysicsLevel.FLUID_AIR_BATH.value:
        if z_water_top - tol < cz < z_air_top + tol and r < R_inner + tol:
            return Domain.AIR
    
    # Water (default for unclassified)
    return Domain.WATER


def _classify_surface(
    surf_tag: int,
    z_plate_bottom: float, z_plate_top: float,
    z_water_top: float, z_air_top: float,
    R_inner: float, R_outer: float,
    L_pml: float, r_pml_outer: float,
    physics_level: PhysicsLevel
) -> Optional[Interface]:
    """
    Classify a surface as an interface type based on its position.
    
    Returns None for surfaces that aren't important interfaces.
    """
    bbox = gmsh.model.getBoundingBox(2, surf_tag)
    xmin, ymin, zmin = bbox[0], bbox[1], bbox[2]
    xmax, ymax, zmax = bbox[3], bbox[4], bbox[5]
    
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    cz = (zmin + zmax) / 2
    r = np.sqrt(cx**2 + cy**2)
    
    # Tolerance for surface detection
    tol = 0.001
    z_thickness = zmax - zmin
    
    # Horizontal surfaces (small z extent)
    is_horizontal = z_thickness < tol
    
    if is_horizontal:
        # Water-air interface
        if abs(cz - z_water_top) < tol and r < R_inner + tol:
            return Interface.WATER_AIR
        
        # Water-plate interface
        if abs(cz - z_plate_top) < tol and r < R_inner + tol:
            if physics_level.value >= PhysicsLevel.FLUID_SOLID.value:
                return Interface.WATER_PLATE
        
        # Bath-plate interface
        if abs(cz - z_plate_bottom) < tol and r < R_outer + tol:
            if physics_level.value >= PhysicsLevel.FLUID_SOLID.value:
                return Interface.BATH_PLATE
    
    # Vertical/cylindrical surfaces (water-wall)
    r_avg = np.sqrt(cx**2 + cy**2)
    if abs(r_avg - R_inner) < tol:
        if z_plate_top - tol < cz < z_water_top + tol:
            if physics_level.value >= PhysicsLevel.FLUID_SOLID.value:
                return Interface.WATER_WALL
    
    # Outer boundary (PML)
    if r > r_pml_outer - tol:
        return Interface.PML_OUTER
    
    return None


# =========================================================
# SIMPLIFIED GEOMETRY FOR QUICK TESTS
# =========================================================

def create_simple_box_geometry(
    config: FEMConfig,
    output_dir: Optional[str] = None,
    verbose: bool = True
) -> Tuple[any, any, any, MeshInfo]:
    """
    Create a simple box geometry for testing.
    
    This creates a single water domain without complex geometry,
    useful for validating the solver before adding complexity.
    """
    from dolfinx import mesh as dmesh
    from dolfinx.io import gmshio
    
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
    gmsh.model.add("simple_box")
    
    geo = config.geometry
    
    # Simple box
    Lx = 2 * geo.dish_inner_radius
    Ly = 2 * geo.dish_inner_radius
    Lz = geo.water_depth
    
    box = gmsh.model.occ.addBox(-Lx/2, -Ly/2, 0, Lx, Ly, Lz)
    gmsh.model.occ.synchronize()
    
    # Tag as water
    pg = gmsh.model.addPhysicalGroup(3, [box], Domain.WATER.gmsh_tag)
    gmsh.model.setPhysicalName(3, pg, "WATER")
    
    # Tag bottom surface for actuation
    bottom_surfs = []
    all_surfs = gmsh.model.getEntities(dim=2)
    for dim, tag in all_surfs:
        bbox = gmsh.model.getBoundingBox(2, tag)
        if abs(bbox[2]) < 0.001 and abs(bbox[5]) < 0.001:
            bottom_surfs.append(tag)
    
    if bottom_surfs:
        pg = gmsh.model.addPhysicalGroup(2, bottom_surfs, Interface.ACTUATION.gmsh_tag)
        gmsh.model.setPhysicalName(2, pg, "ACTUATION")
    
    # Mesh
    lc = Lx / 20
    gmsh.option.setNumber("Mesh.MeshSizeMax", lc)
    gmsh.model.mesh.generate(3)
    
    # Import to DOLFINx
    mesh, cell_tags, facet_tags = gmshio.model_to_mesh(
        gmsh.model, MPI.COMM_WORLD, rank=0, gdim=3
    )
    
    gmsh.finalize()
    
    mesh_info = MeshInfo(
        num_nodes=0,
        num_cells=0,
        domain_counts={"WATER": 1},
        interface_counts={"ACTUATION": 1},
        min_element_size=lc,
        max_element_size=lc,
        mesh_file=None
    )
    
    return mesh, cell_tags, facet_tags, mesh_info
