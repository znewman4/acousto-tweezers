"""
FEM mesh and geometry module for acoustic tweezers simulation.

This module provides:
1. Structured mesh generation for the Petri dish geometry
2. Domain assignment based on physical regions
3. Interface detection and normal computation
4. Mesh quality metrics and refinement

Geometry (from MASTER BRIEF):
- Ωw: dish water (particle domain)
- Ωa: air above water
- Ωb: external bath water
- Ωp: bottom plate (elastic)
- Ωs: side walls (elastic)
- PML regions surrounding open boundaries

Coordinate System:
- Origin at center of dish bottom interior surface
- z-axis pointing upward
- Axisymmetric about z-axis (but full 3D mesh)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
from scipy import sparse

from .domains import (
    DomainType, InterfaceType, Domain, Interface,
    DomainInfo, InterfaceInfo, get_interface_type
)
from .config import GeometryConfig, FEMConfig
from .materials import MaterialDatabase, FluidMaterial, SolidMaterial


@dataclass
class Node:
    """A single mesh node."""
    id: int
    x: float
    y: float
    z: float
    
    def __array__(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])
    
    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])


@dataclass
class Element:
    """
    A finite element (hexahedral for structured mesh).
    
    Node ordering follows standard FEM convention:
    
        7-------6
       /|      /|
      4-------5 |
      | 3-----|-2
      |/      |/
      0-------1
      
    Local coordinates: (ξ, η, ζ) ∈ [-1, 1]³
    """
    id: int
    node_ids: np.ndarray  # 8 node IDs for hexahedron
    domain: DomainType
    
    def __post_init__(self):
        self.node_ids = np.asarray(self.node_ids, dtype=np.int64)
        if len(self.node_ids) != 8:
            raise ValueError(f"Hexahedral element requires 8 nodes, got {len(self.node_ids)}")


@dataclass
class Facet:
    """
    A boundary facet (quadrilateral face of hexahedron).
    
    Used for interface conditions and boundary conditions.
    """
    id: int
    node_ids: np.ndarray  # 4 node IDs for quad
    element_id: int       # Parent element
    local_face: int       # Which face (0-5)
    normal: np.ndarray    # Outward normal
    
    def __post_init__(self):
        self.node_ids = np.asarray(self.node_ids, dtype=np.int64)
        self.normal = np.asarray(self.normal, dtype=np.float64)


@dataclass
class FEMMesh:
    """
    Complete finite element mesh.
    
    Contains nodes, elements, and connectivity information.
    """
    # Node data
    nodes: np.ndarray          # (N_nodes, 3) coordinates
    num_nodes: int
    
    # Element data
    elements: np.ndarray       # (N_elements, 8) node indices
    element_domains: np.ndarray  # (N_elements,) domain IDs
    num_elements: int
    
    # Grid info (for structured mesh)
    nx: int
    ny: int
    nz: int
    dx: float
    dy: float
    dz: float
    
    # Coordinate arrays
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    
    # Domain and interface info
    domain_info: Dict[DomainType, DomainInfo] = field(default_factory=dict)
    interface_info: Dict[InterfaceType, InterfaceInfo] = field(default_factory=dict)
    
    # Facets (computed on demand)
    facets: Optional[List[Facet]] = None
    boundary_facets: Optional[np.ndarray] = None
    
    @property
    def h(self) -> float:
        """Characteristic element size."""
        return min(self.dx, self.dy, self.dz)
    
    @property
    def n_nodes(self) -> int:
        """Alias for num_nodes."""
        return self.num_nodes
    
    @property
    def n_elements(self) -> int:
        """Alias for num_elements."""
        return self.num_elements
    
    @property
    def Lx(self) -> float:
        """Domain size in x."""
        return self.x[-1] - self.x[0]
    
    @property
    def Ly(self) -> float:
        """Domain size in y."""
        return self.y[-1] - self.y[0]
    
    @property
    def Lz(self) -> float:
        """Domain size in z."""
        return self.z[-1] - self.z[0]
    
    def get_node_coords(self, node_id: int) -> np.ndarray:
        """Get coordinates of a node."""
        return self.nodes[node_id]
    
    def get_element_nodes(self, element_id: int) -> np.ndarray:
        """Get node IDs of an element."""
        return self.elements[element_id]
    
    def get_element_coords(self, element_id: int) -> np.ndarray:
        """Get (8, 3) array of node coordinates for element."""
        node_ids = self.elements[element_id]
        return self.nodes[node_ids]
    
    def get_element_centroid(self, element_id: int) -> np.ndarray:
        """Get centroid of an element."""
        coords = self.get_element_coords(element_id)
        return np.mean(coords, axis=0)
    
    def get_domain_elements(self, domain: DomainType) -> np.ndarray:
        """Get element IDs belonging to a domain."""
        return np.where(self.element_domains == domain.value)[0]
    
    def get_domain_nodes(self, domain: DomainType) -> np.ndarray:
        """Get unique node IDs belonging to a domain."""
        elem_ids = self.get_domain_elements(domain)
        node_ids = self.elements[elem_ids].flatten()
        return np.unique(node_ids)
    
    def point_to_element(self, point: np.ndarray) -> int:
        """
        Find element containing a point.
        
        For structured mesh, use direct indexing.
        """
        # Structured mesh: direct lookup
        ix = int((point[0] - self.x[0]) / self.dx)
        iy = int((point[1] - self.y[0]) / self.dy)
        iz = int((point[2] - self.z[0]) / self.dz)
        
        # Clamp to valid range
        ix = max(0, min(ix, self.nx - 2))
        iy = max(0, min(iy, self.ny - 2))
        iz = max(0, min(iz, self.nz - 2))
        
        # Element index (structured ordering)
        return ix * (self.ny - 1) * (self.nz - 1) + iy * (self.nz - 1) + iz
    
    def summary(self) -> str:
        """Return mesh summary string."""
        lines = [
            "FEM Mesh Summary",
            "=" * 40,
            f"Nodes:    {self.num_nodes:,}",
            f"Elements: {self.num_elements:,}",
            f"Grid:     {self.nx} × {self.ny} × {self.nz}",
            f"Spacing:  {self.dx*1e3:.3f} × {self.dy*1e3:.3f} × {self.dz*1e3:.3f} mm",
            f"Domain:   [{self.x[0]*1e3:.1f}, {self.x[-1]*1e3:.1f}] × "
            f"[{self.y[0]*1e3:.1f}, {self.y[-1]*1e3:.1f}] × "
            f"[{self.z[0]*1e3:.1f}, {self.z[-1]*1e3:.1f}] mm",
            "",
            "Domains:",
        ]
        for domain, info in self.domain_info.items():
            lines.append(f"  {domain.name}: {info.num_elements:,} elements")
        
        if self.interface_info:
            lines.append("")
            lines.append("Interfaces:")
            for iface, info in self.interface_info.items():
                lines.append(f"  {iface.name}: {info.num_facets:,} facets")
        
        return "\n".join(lines)


def create_petri_dish_mesh(
    config: Union[GeometryConfig, FEMConfig],
    materials: Optional[MaterialDatabase] = None,
) -> FEMMesh:
    """
    Create structured hexahedral mesh for Petri dish geometry.
    
    Geometry layout (cross-section at y=0):
    
    z
    ^                  PML (air)
    |     ----------------------------------------
    |     |   AIR (Ωa)                           |
    |     |                                      |
    |     |   - - - - water surface - - - - - - |
    |     |   WATER (Ωw)                        |
    |     |======================================|  <- dish bottom (plate)
    |     ||    PLATE (Ωp)                     ||
    |     ||===================================||
    |     |   BATH (Ωb)                         |
    |     |                                      |
    |     ----------------------------------------
    |     |          PML (bath)                  |
    +-----------------------------------------------------> x
          |<--- bath extent --->|<- dish ->|
    
    Parameters
    ----------
    config : GeometryConfig or FEMConfig
        Geometry configuration.
    materials : MaterialDatabase, optional
        Material properties for determining mesh density.
    
    Returns
    -------
    mesh : FEMMesh
        Complete finite element mesh.
    """
    if isinstance(config, FEMConfig):
        geom = config.geometry
        phys = config.physics
        temperature = phys.temperature
    else:
        geom = config
        phys = None
        temperature = 25.0
    
    # Get material properties for mesh sizing
    if materials is None:
        materials = MaterialDatabase(temperature=temperature)
    
    water = materials.water
    
    # Determine mesh spacing based on wavelength
    frequency = phys.frequency if phys else 2.0e6
    wavelength = water.wavelength(frequency)
    h_target = wavelength / geom.elements_per_wavelength
    
    # Clamp to min/max
    h = max(geom.min_element_size, min(h_target, geom.max_element_size))
    
    # Domain extents
    R = geom.dish_radius
    wall_t = geom.dish_wall_thickness
    plate_t = geom.dish_bottom_thickness
    bath_ext = geom.bath_lateral_extent
    pml_t = geom.pml_thickness
    
    # Total domain size
    x_min = -(R + wall_t + bath_ext + pml_t)
    x_max = R + wall_t + bath_ext + pml_t
    y_min = x_min
    y_max = x_max
    z_min = -(geom.bath_depth + pml_t)
    z_max = geom.water_depth + geom.air_height + pml_t
    
    # Number of elements in each direction
    nx = int(np.ceil((x_max - x_min) / h)) + 1
    ny = int(np.ceil((y_max - y_min) / h)) + 1
    nz = int(np.ceil((z_max - z_min) / h)) + 1
    
    # Actual spacing
    dx = (x_max - x_min) / (nx - 1)
    dy = (y_max - y_min) / (ny - 1)
    dz = (z_max - z_min) / (nz - 1)
    
    # Create coordinate arrays
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    z = np.linspace(z_min, z_max, nz)
    
    # Create nodes (structured grid)
    num_nodes = nx * ny * nz
    nodes = np.zeros((num_nodes, 3), dtype=np.float64)
    
    node_idx = 0
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                nodes[node_idx] = [x[ix], y[iy], z[iz]]
                node_idx += 1
    
    # Create elements (hexahedra)
    num_elements = (nx - 1) * (ny - 1) * (nz - 1)
    elements = np.zeros((num_elements, 8), dtype=np.int64)
    element_domains = np.zeros(num_elements, dtype=np.int32)
    
    def node_index(ix, iy, iz):
        return iz * ny * nx + iy * nx + ix
    
    elem_idx = 0
    for iz in range(nz - 1):
        for iy in range(ny - 1):
            for ix in range(nx - 1):
                # 8 corners of hexahedron
                n0 = node_index(ix, iy, iz)
                n1 = node_index(ix + 1, iy, iz)
                n2 = node_index(ix + 1, iy + 1, iz)
                n3 = node_index(ix, iy + 1, iz)
                n4 = node_index(ix, iy, iz + 1)
                n5 = node_index(ix + 1, iy, iz + 1)
                n6 = node_index(ix + 1, iy + 1, iz + 1)
                n7 = node_index(ix, iy + 1, iz + 1)
                
                elements[elem_idx] = [n0, n1, n2, n3, n4, n5, n6, n7]
                
                # Assign domain based on centroid
                cx = (x[ix] + x[ix + 1]) / 2
                cy = (y[iy] + y[iy + 1]) / 2
                cz = (z[iz] + z[iz + 1]) / 2
                
                element_domains[elem_idx] = _classify_point(
                    cx, cy, cz, geom
                ).value
                
                elem_idx += 1
    
    # Build domain info
    domain_info = {}
    for domain in DomainType:
        elem_ids = np.where(element_domains == domain.value)[0]
        if len(elem_ids) > 0:
            # Get unique nodes
            node_ids = np.unique(elements[elem_ids].flatten())
            
            # Compute volume
            volume = len(elem_ids) * dx * dy * dz
            
            # Material ID
            if domain.is_fluid:
                mat_id = "water" if domain.is_water_like else "air"
            elif domain.is_solid:
                mat_id = "borosilicate_glass"
            else:
                mat_id = "pml"
            
            domain_info[domain] = DomainInfo(
                domain_type=domain,
                name=domain.name,
                material_id=mat_id,
                element_ids=elem_ids,
                node_ids=node_ids,
                volume=volume,
            )
    
    mesh = FEMMesh(
        nodes=nodes,
        num_nodes=num_nodes,
        elements=elements,
        element_domains=element_domains,
        num_elements=num_elements,
        nx=nx,
        ny=ny,
        nz=nz,
        dx=dx,
        dy=dy,
        dz=dz,
        x=x,
        y=y,
        z=z,
        domain_info=domain_info,
    )
    
    # Detect and build interface info
    mesh.interface_info = _build_interfaces(mesh, geom)
    
    return mesh


def _classify_point(
    x: float,
    y: float,
    z: float,
    geom: GeometryConfig,
) -> DomainType:
    """
    Classify a point into a domain.
    
    This is the core geometry definition.
    """
    R = geom.dish_radius
    wall_t = geom.dish_wall_thickness
    plate_t = geom.dish_bottom_thickness
    bath_ext = geom.bath_lateral_extent
    pml_t = geom.pml_thickness
    
    # Radial distance from z-axis
    r = np.sqrt(x**2 + y**2)
    
    # Z-coordinates of key surfaces
    z_bath_bottom = -geom.bath_depth
    z_plate_bottom = -plate_t
    z_plate_top = 0.0
    z_water_surface = geom.water_depth
    z_air_top = geom.water_depth + geom.air_height
    
    # Check if in PML region (outer boundary)
    outer_r = R + wall_t + bath_ext
    in_pml_lateral = r > outer_r
    in_pml_bottom = z < z_bath_bottom
    in_pml_top = z > z_air_top
    
    # PML classification
    if in_pml_lateral or in_pml_bottom or in_pml_top:
        if z < z_plate_top:
            return DomainType.PML_BATH
        elif z > z_water_surface:
            return DomainType.PML_AIR
        else:
            return DomainType.PML_WATER
    
    # Inside plate (bottom of dish)
    if z_plate_bottom <= z <= z_plate_top and r <= R + wall_t:
        return DomainType.PLATE
    
    # Side wall
    if R <= r <= R + wall_t and z_plate_top <= z <= z_air_top:
        return DomainType.WALL
    
    # Bath (below dish, outside dish footprint)
    if z < z_plate_bottom and r > R + wall_t:
        return DomainType.BATH
    if z_plate_bottom <= z <= z_plate_top and r > R + wall_t:
        return DomainType.BATH
    
    # Water inside dish
    if r < R and z_plate_top <= z <= z_water_surface:
        return DomainType.WATER
    
    # Air above water (inside dish)
    if r < R and z_water_surface < z <= z_air_top:
        return DomainType.AIR
    
    # Air outside dish (above plate level)
    if r >= R + wall_t and z > z_water_surface:
        return DomainType.AIR
    
    # Bath (default for remaining underwater regions outside dish)
    if z <= z_water_surface:
        return DomainType.BATH
    
    # Default: air
    return DomainType.AIR


def _build_interfaces(
    mesh: FEMMesh,
    geom: GeometryConfig,
) -> Dict[InterfaceType, InterfaceInfo]:
    """
    Detect interfaces between domains.
    
    An interface exists where adjacent elements have different domains.
    """
    interfaces = {}
    
    # For each pair of adjacent elements, check if domains differ
    # In structured mesh, adjacent elements differ by 1 in one direction
    nx, ny, nz = mesh.nx - 1, mesh.ny - 1, mesh.nz - 1  # Element counts
    
    def elem_index(ix, iy, iz):
        return iz * ny * nx + iy * nx + ix
    
    # Track facets for each interface type
    interface_facets = {itype: [] for itype in InterfaceType}
    
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                e1 = elem_index(ix, iy, iz)
                d1 = DomainType(mesh.element_domains[e1])
                
                # Check z+ neighbor
                if iz < nz - 1:
                    e2 = elem_index(ix, iy, iz + 1)
                    d2 = DomainType(mesh.element_domains[e2])
                    if d1 != d2:
                        itype = get_interface_type(d1, d2)
                        if itype:
                            interface_facets[itype].append((e1, e2, 'z'))
                
                # Check y+ neighbor
                if iy < ny - 1:
                    e2 = elem_index(ix, iy + 1, iz)
                    d2 = DomainType(mesh.element_domains[e2])
                    if d1 != d2:
                        itype = get_interface_type(d1, d2)
                        if itype:
                            interface_facets[itype].append((e1, e2, 'y'))
                
                # Check x+ neighbor
                if ix < nx - 1:
                    e2 = elem_index(ix + 1, iy, iz)
                    d2 = DomainType(mesh.element_domains[e2])
                    if d1 != d2:
                        itype = get_interface_type(d1, d2)
                        if itype:
                            interface_facets[itype].append((e1, e2, 'x'))
    
    # Build InterfaceInfo for non-empty interfaces
    for itype, facets in interface_facets.items():
        if facets:
            facet_array = np.array([(f[0], f[1]) for f in facets])
            normal_dirs = [f[2] for f in facets]
            primary_normal = max(set(normal_dirs), key=normal_dirs.count)
            
            # Determine domain pair
            if itype.is_fluid_fluid:
                d1, d2 = DomainType.WATER, DomainType.AIR
            elif itype == InterfaceType.WATER_PLATE:
                d1, d2 = DomainType.WATER, DomainType.PLATE
            elif itype == InterfaceType.WATER_WALL:
                d1, d2 = DomainType.WATER, DomainType.WALL
            elif itype == InterfaceType.BATH_PLATE:
                d1, d2 = DomainType.BATH, DomainType.PLATE
            elif itype == InterfaceType.BATH_WALL:
                d1, d2 = DomainType.BATH, DomainType.WALL
            else:
                d1, d2 = DomainType.WATER, DomainType.AIR
            
            interfaces[itype] = InterfaceInfo(
                interface_type=itype,
                name=itype.name,
                domain_minus=d1,
                domain_plus=d2,
                facet_ids=facet_array,
                normal_direction=primary_normal,
                area=len(facets) * mesh.dx * mesh.dy,  # Approximate
            )
    
    return interfaces


def get_element_jacobian(
    mesh: FEMMesh,
    element_id: int,
) -> Tuple[np.ndarray, float]:
    """
    Compute Jacobian matrix and determinant for an element.
    
    For a structured mesh with uniform spacing, this is constant.
    
    Returns
    -------
    J : np.ndarray
        (3, 3) Jacobian matrix ∂x/∂ξ
    det_J : float
        Determinant of Jacobian
    """
    # For uniform structured mesh
    J = np.diag([mesh.dx / 2, mesh.dy / 2, mesh.dz / 2])
    det_J = (mesh.dx * mesh.dy * mesh.dz) / 8
    return J, det_J


def get_shape_functions_hex8(xi: np.ndarray) -> np.ndarray:
    """
    Evaluate trilinear shape functions at local coordinates.
    
    Parameters
    ----------
    xi : np.ndarray
        Local coordinates (ξ, η, ζ) ∈ [-1, 1]³
    
    Returns
    -------
    N : np.ndarray
        Shape function values (8,)
    """
    xi_, eta, zeta = xi
    
    N = np.array([
        (1 - xi_) * (1 - eta) * (1 - zeta),  # N0
        (1 + xi_) * (1 - eta) * (1 - zeta),  # N1
        (1 + xi_) * (1 + eta) * (1 - zeta),  # N2
        (1 - xi_) * (1 + eta) * (1 - zeta),  # N3
        (1 - xi_) * (1 - eta) * (1 + zeta),  # N4
        (1 + xi_) * (1 - eta) * (1 + zeta),  # N5
        (1 + xi_) * (1 + eta) * (1 + zeta),  # N6
        (1 - xi_) * (1 + eta) * (1 + zeta),  # N7
    ]) / 8
    
    return N


def get_shape_gradients_hex8(xi: np.ndarray) -> np.ndarray:
    """
    Evaluate shape function gradients at local coordinates.
    
    Parameters
    ----------
    xi : np.ndarray
        Local coordinates (ξ, η, ζ) ∈ [-1, 1]³
    
    Returns
    -------
    dN : np.ndarray
        Shape function gradients (8, 3) = [∂N/∂ξ, ∂N/∂η, ∂N/∂ζ]
    """
    xi_, eta, zeta = xi
    
    dN = np.array([
        [-(1 - eta) * (1 - zeta), -(1 - xi_) * (1 - zeta), -(1 - xi_) * (1 - eta)],
        [(1 - eta) * (1 - zeta), -(1 + xi_) * (1 - zeta), -(1 + xi_) * (1 - eta)],
        [(1 + eta) * (1 - zeta), (1 + xi_) * (1 - zeta), -(1 + xi_) * (1 + eta)],
        [-(1 + eta) * (1 - zeta), (1 - xi_) * (1 - zeta), -(1 - xi_) * (1 + eta)],
        [-(1 - eta) * (1 + zeta), -(1 - xi_) * (1 + zeta), (1 - xi_) * (1 - eta)],
        [(1 - eta) * (1 + zeta), -(1 + xi_) * (1 + zeta), (1 + xi_) * (1 - eta)],
        [(1 + eta) * (1 + zeta), (1 + xi_) * (1 + zeta), (1 + xi_) * (1 + eta)],
        [-(1 + eta) * (1 + zeta), (1 - xi_) * (1 + zeta), (1 - xi_) * (1 + eta)],
    ]) / 8
    
    return dN


# Gauss quadrature points and weights for hexahedra (2×2×2)
GAUSS_POINTS_HEX8 = np.array([
    [-1, -1, -1],
    [1, -1, -1],
    [1, 1, -1],
    [-1, 1, -1],
    [-1, -1, 1],
    [1, -1, 1],
    [1, 1, 1],
    [-1, 1, 1],
]) / np.sqrt(3)

GAUSS_WEIGHTS_HEX8 = np.ones(8)  # All weights = 1 for 2×2×2 Gauss
