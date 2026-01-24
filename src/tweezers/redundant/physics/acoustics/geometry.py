"""
Multi-domain geometry definition for coupled acoustic-solid simulations.

Defines:
- Domain types (Ωw, Ωa, Ωb, Ωp, Ωs)
- Interface types (Γwa, Γws, Γbs, Γba)
- Domain masks and interface extraction
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple, Dict, List
import numpy as np


class DomainType(Enum):
    """Types of physical domains in the simulation."""
    WATER_DISH = auto()      # Ωw: dish water volume (where particle lives)
    AIR = auto()             # Ωa: air volume above dish water
    WATER_BATH = auto()      # Ωb: bath water outside/around dish
    PLATE = auto()           # Ωp: dish bottom plate (plastic)
    SIDEWALL = auto()        # Ωs: dish side walls (plastic)
    PML = auto()             # Perfectly Matched Layer (absorbing)
    TRANSDUCER = auto()      # Transducer solid (optional)
    LENS = auto()            # Acoustic lens (optional)


class InterfaceType(Enum):
    """Types of interfaces between domains."""
    WATER_AIR = auto()       # Γwa: water-air free surface
    WATER_SOLID = auto()     # Γws: water-solid (inner dish surfaces)
    BATH_SOLID = auto()      # Γbs: bath water-solid (outer dish surfaces)
    BATH_AIR = auto()        # Γba: bath water-air interface
    SOLID_SOLID = auto()     # Solid-solid coupling (e.g., transducer-plate)
    PML_FLUID = auto()       # PML-fluid interface


@dataclass
class DomainRegion:
    """Definition of a single domain region."""
    domain_type: DomainType
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float
    
    # Optional: cylindrical shape for dish
    is_cylindrical: bool = False
    center_x: float = 0.0
    center_y: float = 0.0
    inner_radius: float = 0.0
    outer_radius: float = 0.0
    
    def contains_point(self, x: float, y: float, z: float) -> bool:
        """Check if a point is inside this region."""
        if not (self.z_min <= z <= self.z_max):
            return False
        
        if self.is_cylindrical:
            r = np.sqrt((x - self.center_x)**2 + (y - self.center_y)**2)
            return self.inner_radius <= r <= self.outer_radius
        else:
            return (self.x_min <= x <= self.x_max and 
                    self.y_min <= y <= self.y_max)
    
    def get_mask(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """Get boolean mask for grid points inside this region."""
        z_mask = (Z >= self.z_min) & (Z <= self.z_max)
        
        if self.is_cylindrical:
            R = np.sqrt((X - self.center_x)**2 + (Y - self.center_y)**2)
            return z_mask & (R >= self.inner_radius) & (R <= self.outer_radius)
        else:
            return (z_mask & 
                    (X >= self.x_min) & (X <= self.x_max) &
                    (Y >= self.y_min) & (Y <= self.y_max))


@dataclass
class InterfaceRegion:
    """Definition of an interface between two domains."""
    interface_type: InterfaceType
    domain1: DomainType
    domain2: DomainType
    normal_direction: str  # 'x', 'y', 'z', or 'r' (radial for cylindrical)
    position: float  # Position along normal direction
    
    # Extent of interface
    extent_min_1: float = 0.0  # Min in first tangent direction
    extent_max_1: float = 0.0  # Max in first tangent direction
    extent_min_2: float = 0.0  # Min in second tangent direction
    extent_max_2: float = 0.0  # Max in second tangent direction


@dataclass
class MultiDomainGeometry:
    """
    Complete multi-domain geometry for coupled acoustic-solid simulation.
    
    Contains:
    - All domain regions (fluids and solids)
    - All interface definitions
    - Grid generation utilities
    
    Geometry layout (vertical cross-section):
    
        PML_top (air)
        ===============================
              AIR DOMAIN (Ωa)
        -------------------------------  <- Γwa (water-air interface)
              WATER DISH (Ωw)
        |-----|               |-----|
        | wall|               | wall|   <- Ωs (sidewalls)  
        |-----|               |-----|
        ===============================  <- Γws (water-solid, bottom)
             PLATE DOMAIN (Ωp)
        ===============================  <- Γbs (bath-solid, bottom)
              BATH WATER (Ωb)
        ===============================
             PML_bottom (bath)
    """
    
    # Overall domain bounds (including PML)
    Lx: float  # Total x extent
    Ly: float  # Total y extent
    Lz: float  # Total z extent
    
    # Grid resolution
    dx: float
    dy: float
    dz: float
    
    # Dish parameters
    dish_radius: float           # Inner radius of dish
    dish_wall_thickness: float   # Thickness of sidewalls
    dish_height: float           # Height of water column in dish
    plate_thickness: float       # Bottom plate thickness
    
    # Air parameters
    air_height: float            # Height of air domain above dish water
    
    # Bath parameters
    bath_depth: float            # Depth of bath below plate
    
    # PML parameters
    pml_thickness: float = 0.005  # Thickness of PML layers
    
    # Computed quantities (filled by __post_init__)
    regions: Dict[DomainType, DomainRegion] = field(default_factory=dict)
    interfaces: List[InterfaceRegion] = field(default_factory=list)
    
    # Grid arrays (computed)
    x: np.ndarray = field(default=None, repr=False)
    y: np.ndarray = field(default=None, repr=False)
    z: np.ndarray = field(default=None, repr=False)
    X: np.ndarray = field(default=None, repr=False)
    Y: np.ndarray = field(default=None, repr=False)
    Z: np.ndarray = field(default=None, repr=False)
    
    def __post_init__(self):
        """Build domain regions and interfaces."""
        self._build_grid()
        self._build_regions()
        self._build_interfaces()
    
    def _build_grid(self):
        """Create coordinate arrays."""
        self.Nx = int(round(self.Lx / self.dx)) + 1
        self.Ny = int(round(self.Ly / self.dy)) + 1
        self.Nz = int(round(self.Lz / self.dz)) + 1
        
        self.x = np.linspace(0, self.Lx, self.Nx)
        self.y = np.linspace(0, self.Ly, self.Ny)
        self.z = np.linspace(0, self.Lz, self.Nz)
        
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')
    
    def _build_regions(self):
        """Build all domain regions."""
        # Center of domain
        cx, cy = self.Lx / 2, self.Ly / 2
        
        # Z coordinates (bottom to top)
        z_pml_bot = 0.0
        z_bath_bot = self.pml_thickness
        z_plate_bot = z_bath_bot + self.bath_depth
        z_plate_top = z_plate_bot + self.plate_thickness
        z_water_top = z_plate_top + self.dish_height
        z_air_top = z_water_top + self.air_height
        z_pml_top = z_air_top + self.pml_thickness
        
        # Store key z-levels
        self.z_levels = {
            'pml_bot': z_pml_bot,
            'bath_bot': z_bath_bot,
            'plate_bot': z_plate_bot,
            'plate_top': z_plate_top,
            'water_top': z_water_top,
            'air_top': z_air_top,
            'pml_top': z_pml_top,
        }
        
        # 1. Dish water domain (Ωw) - cylindrical
        self.regions[DomainType.WATER_DISH] = DomainRegion(
            domain_type=DomainType.WATER_DISH,
            x_min=cx - self.dish_radius, x_max=cx + self.dish_radius,
            y_min=cy - self.dish_radius, y_max=cy + self.dish_radius,
            z_min=z_plate_top, z_max=z_water_top,
            is_cylindrical=True,
            center_x=cx, center_y=cy,
            inner_radius=0.0, outer_radius=self.dish_radius,
        )
        
        # 2. Air domain (Ωa) - extends above entire dish and to PML
        self.regions[DomainType.AIR] = DomainRegion(
            domain_type=DomainType.AIR,
            x_min=self.pml_thickness, x_max=self.Lx - self.pml_thickness,
            y_min=self.pml_thickness, y_max=self.Ly - self.pml_thickness,
            z_min=z_water_top, z_max=z_air_top,
        )
        
        # 3. Bath water domain (Ωb) - surrounds dish externally
        self.regions[DomainType.WATER_BATH] = DomainRegion(
            domain_type=DomainType.WATER_BATH,
            x_min=self.pml_thickness, x_max=self.Lx - self.pml_thickness,
            y_min=self.pml_thickness, y_max=self.Ly - self.pml_thickness,
            z_min=z_bath_bot, z_max=z_plate_bot,
        )
        
        # 4. Dish plate (Ωp) - bottom of dish
        outer_r = self.dish_radius + self.dish_wall_thickness
        self.regions[DomainType.PLATE] = DomainRegion(
            domain_type=DomainType.PLATE,
            x_min=cx - outer_r, x_max=cx + outer_r,
            y_min=cy - outer_r, y_max=cy + outer_r,
            z_min=z_plate_bot, z_max=z_plate_top,
            is_cylindrical=True,
            center_x=cx, center_y=cy,
            inner_radius=0.0, outer_radius=outer_r,
        )
        
        # 5. Dish sidewalls (Ωs) - annular region
        self.regions[DomainType.SIDEWALL] = DomainRegion(
            domain_type=DomainType.SIDEWALL,
            x_min=cx - outer_r, x_max=cx + outer_r,
            y_min=cy - outer_r, y_max=cy + outer_r,
            z_min=z_plate_top, z_max=z_water_top,
            is_cylindrical=True,
            center_x=cx, center_y=cy,
            inner_radius=self.dish_radius,
            outer_radius=outer_r,
        )
        
        # 6. PML regions (stored as one combined region for simplicity)
        # In practice, we identify PML by checking if point is within pml_thickness of boundaries
        
    def _build_interfaces(self):
        """Build interface definitions."""
        cx, cy = self.Lx / 2, self.Ly / 2
        z = self.z_levels
        
        # 1. Water-air interface (Γwa) - top of dish water
        self.interfaces.append(InterfaceRegion(
            interface_type=InterfaceType.WATER_AIR,
            domain1=DomainType.WATER_DISH,
            domain2=DomainType.AIR,
            normal_direction='z',
            position=z['water_top'],
            extent_min_1=cx - self.dish_radius,
            extent_max_1=cx + self.dish_radius,
            extent_min_2=cy - self.dish_radius,
            extent_max_2=cy + self.dish_radius,
        ))
        
        # 2. Water-solid interface (Γws) - dish water touching plate and walls
        # Bottom interface (plate top)
        self.interfaces.append(InterfaceRegion(
            interface_type=InterfaceType.WATER_SOLID,
            domain1=DomainType.WATER_DISH,
            domain2=DomainType.PLATE,
            normal_direction='z',
            position=z['plate_top'],
            extent_min_1=cx - self.dish_radius,
            extent_max_1=cx + self.dish_radius,
            extent_min_2=cy - self.dish_radius,
            extent_max_2=cy + self.dish_radius,
        ))
        
        # Sidewall interface (radial)
        self.interfaces.append(InterfaceRegion(
            interface_type=InterfaceType.WATER_SOLID,
            domain1=DomainType.WATER_DISH,
            domain2=DomainType.SIDEWALL,
            normal_direction='r',
            position=self.dish_radius,
            extent_min_1=z['plate_top'],
            extent_max_1=z['water_top'],
            extent_min_2=0.0,
            extent_max_2=2 * np.pi,  # Full circle
        ))
        
        # 3. Bath-solid interface (Γbs) - bath water touching plate bottom
        self.interfaces.append(InterfaceRegion(
            interface_type=InterfaceType.BATH_SOLID,
            domain1=DomainType.WATER_BATH,
            domain2=DomainType.PLATE,
            normal_direction='z',
            position=z['plate_bot'],
            extent_min_1=cx - (self.dish_radius + self.dish_wall_thickness),
            extent_max_1=cx + (self.dish_radius + self.dish_wall_thickness),
            extent_min_2=cy - (self.dish_radius + self.dish_wall_thickness),
            extent_max_2=cy + (self.dish_radius + self.dish_wall_thickness),
        ))
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        """Grid shape (Nx, Ny, Nz)."""
        return (self.Nx, self.Ny, self.Nz)
    
    def get_domain_mask(self, domain_type: DomainType) -> np.ndarray:
        """Get boolean mask for a specific domain."""
        if domain_type not in self.regions:
            return np.zeros(self.shape, dtype=bool)
        return self.regions[domain_type].get_mask(self.X, self.Y, self.Z)
    
    def get_all_domain_masks(self) -> Dict[DomainType, np.ndarray]:
        """Get masks for all domains."""
        return {dt: self.get_domain_mask(dt) for dt in self.regions}
    
    def get_domain_index_field(self) -> np.ndarray:
        """
        Get integer field indicating domain type at each grid point.
        0 = undefined, 1+ = domain type enum value
        """
        domain_field = np.zeros(self.shape, dtype=np.int32)
        for domain_type, region in self.regions.items():
            mask = region.get_mask(self.X, self.Y, self.Z)
            domain_field[mask] = domain_type.value
        return domain_field
    
    def is_in_pml(self, x: float, y: float, z: float) -> bool:
        """Check if a point is within the PML region."""
        return (x < self.pml_thickness or x > self.Lx - self.pml_thickness or
                y < self.pml_thickness or y > self.Ly - self.pml_thickness or
                z < self.pml_thickness or z > self.Lz - self.pml_thickness)
    
    def get_pml_mask(self) -> np.ndarray:
        """Get boolean mask for PML region."""
        return ((self.X < self.pml_thickness) | (self.X > self.Lx - self.pml_thickness) |
                (self.Y < self.pml_thickness) | (self.Y > self.Ly - self.pml_thickness) |
                (self.Z < self.pml_thickness) | (self.Z > self.Lz - self.pml_thickness))
    
    def get_fluid_mask(self) -> np.ndarray:
        """Get mask for all fluid regions (water dish, air, bath)."""
        return (self.get_domain_mask(DomainType.WATER_DISH) |
                self.get_domain_mask(DomainType.AIR) |
                self.get_domain_mask(DomainType.WATER_BATH))
    
    def get_solid_mask(self) -> np.ndarray:
        """Get mask for all solid regions (plate, sidewalls)."""
        return (self.get_domain_mask(DomainType.PLATE) |
                self.get_domain_mask(DomainType.SIDEWALL))
    
    def find_interface_nodes(self, interface: InterfaceRegion) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find grid node indices on either side of an interface.
        
        Returns
        -------
        (indices_1, indices_2) : Tuple of arrays of shape (N, 3)
            Grid indices [ix, iy, iz] for nodes on domain1 and domain2 sides.
        """
        # Implementation depends on interface normal direction
        # This is a simplified version; full implementation handles cylindrical interfaces
        
        indices_1 = []
        indices_2 = []
        
        if interface.normal_direction == 'z':
            # Find z-index closest to interface position
            iz = np.argmin(np.abs(self.z - interface.position))
            
            for ix in range(self.Nx):
                for iy in range(self.Ny):
                    x, y = self.x[ix], self.y[iy]
                    # Check if within interface extent
                    if (interface.extent_min_1 <= x <= interface.extent_max_1 and
                        interface.extent_min_2 <= y <= interface.extent_max_2):
                        if iz > 0:
                            indices_1.append([ix, iy, iz - 1])
                        if iz < self.Nz - 1:
                            indices_2.append([ix, iy, iz])
        
        return np.array(indices_1), np.array(indices_2)
    
    def summary(self) -> str:
        """Return a summary string of the geometry."""
        lines = [
            "=" * 60,
            "Multi-Domain Geometry Summary",
            "=" * 60,
            f"Grid: {self.Nx} × {self.Ny} × {self.Nz} = {self.Nx * self.Ny * self.Nz:,} nodes",
            f"Spacing: dx={self.dx*1e3:.2f}mm, dy={self.dy*1e3:.2f}mm, dz={self.dz*1e3:.2f}mm",
            f"Total size: {self.Lx*1e3:.1f} × {self.Ly*1e3:.1f} × {self.Lz*1e3:.1f} mm³",
            "",
            "Domain Regions:",
        ]
        
        for domain_type, region in self.regions.items():
            mask = region.get_mask(self.X, self.Y, self.Z)
            n_nodes = np.sum(mask)
            lines.append(f"  {domain_type.name}: {n_nodes:,} nodes")
        
        lines.append("")
        lines.append("Interfaces:")
        for interface in self.interfaces:
            lines.append(f"  {interface.interface_type.name}: "
                        f"{interface.domain1.name} ↔ {interface.domain2.name}")
        
        lines.append("")
        lines.append(f"PML thickness: {self.pml_thickness*1e3:.1f}mm")
        lines.append("=" * 60)
        
        return "\n".join(lines)


def create_standard_dish_geometry(
    dish_diameter_mm: float = 35.0,
    dish_height_mm: float = 10.0,
    wall_thickness_mm: float = 1.0,
    plate_thickness_mm: float = 1.0,
    air_height_mm: float = 5.0,
    bath_depth_mm: float = 5.0,
    pml_thickness_mm: float = 3.0,
    resolution_mm: float = 0.5,
) -> MultiDomainGeometry:
    """
    Create a standard petri dish geometry with typical parameters.
    
    Parameters
    ----------
    dish_diameter_mm : float
        Inner diameter of the dish.
    dish_height_mm : float
        Height of water column in dish.
    wall_thickness_mm : float
        Thickness of dish sidewalls.
    plate_thickness_mm : float
        Thickness of bottom plate.
    air_height_mm : float
        Height of air domain above water.
    bath_depth_mm : float
        Depth of bath water below plate.
    pml_thickness_mm : float
        Thickness of PML absorbing layers.
    resolution_mm : float
        Grid resolution (uniform dx=dy=dz).
    
    Returns
    -------
    MultiDomainGeometry
    """
    # Convert to meters
    mm = 1e-3
    
    dish_radius = 0.5 * dish_diameter_mm * mm
    dish_height = dish_height_mm * mm
    wall_thickness = wall_thickness_mm * mm
    plate_thickness = plate_thickness_mm * mm
    air_height = air_height_mm * mm
    bath_depth = bath_depth_mm * mm
    pml_thickness = pml_thickness_mm * mm
    resolution = resolution_mm * mm
    
    # Total domain size
    outer_radius = dish_radius + wall_thickness
    Lx = 2 * (outer_radius + pml_thickness + 2 * resolution)  # Add margin
    Ly = Lx  # Square domain
    Lz = pml_thickness + bath_depth + plate_thickness + dish_height + air_height + pml_thickness
    
    return MultiDomainGeometry(
        Lx=Lx,
        Ly=Ly,
        Lz=Lz,
        dx=resolution,
        dy=resolution,
        dz=resolution,
        dish_radius=dish_radius,
        dish_wall_thickness=wall_thickness,
        dish_height=dish_height,
        plate_thickness=plate_thickness,
        air_height=air_height,
        bath_depth=bath_depth,
        pml_thickness=pml_thickness,
    )


if __name__ == "__main__":
    # Demo: create and inspect geometry
    geom = create_standard_dish_geometry(
        dish_diameter_mm=20.0,
        dish_height_mm=5.0,
        resolution_mm=1.0,
    )
    print(geom.summary())
