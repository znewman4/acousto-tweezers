"""
Domain and interface tagging abstraction for FEniCSx simulation.

This module provides the SINGLE source of truth for domain labels.
NO MAGIC INTEGERS in variational forms - always use Domain/Interface enums.

Domains (from MASTER BRIEF):
- Domain.WATER: Ωw — dish water (particle domain)
- Domain.AIR: Ωa — air above water (explicit volumetric)
- Domain.BATH: Ωb — external bath water
- Domain.PLATE: Ωp — bottom plate (elastic, lossy) - VOLUMETRIC SOLID
- Domain.WALL: Ωs — side walls (elastic, lossy) - VOLUMETRIC SOLID
- Domain.PML_*: PML absorbing boundary regions
- Domain.LENS: Acoustic lens domain
- Domain.TRANSDUCER: Transducer region for actuation

Interfaces (from MASTER BRIEF):
- Interface.WATER_AIR (Γwa): water–air interface
- Interface.WATER_SOLID (Γws): water–solid (plate and walls)
- Interface.BATH_SOLID (Γbs): bath–solid
- Interface.BATH_AIR (Γba): bath–air
- Interface.ACTUATION: Where mechanical actuation is applied
- Interface.OUTER: External boundary (for PML)

CRITICAL: Ωp and Ωs are NEVER boundary conditions. They are volumetric
solids with displacement DOFs.

Author: Acousto-Tweezers Project
Date: January 2026
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Set, Tuple, Optional
import numpy as np


class Domain(IntEnum):
    """
    Domain type enumeration for physical regions.
    
    Use this enum EXCLUSIVELY for domain identification.
    Never use raw integers for domain IDs in variational forms.
    
    Numbering convention:
    - Fluids: 1-10
    - Solids: 11-20
    - PML regions: 21-30
    - Special: 91+
    
    All domains are VOLUMETRIC (3D).
    """
    # Fluid domains
    WATER = 1       # Ωw: dish water where particles live
    AIR = 2         # Ωa: air above water surface
    BATH = 3        # Ωb: external coupling bath below dish
    
    # Solid domains (VOLUMETRIC - not boundary conditions!)
    PLATE = 11      # Ωp: dish bottom plate (elastic, lossy)
    WALL = 12       # Ωs: dish side walls (elastic, lossy)
    LENS = 13       # Acoustic lens (elastic or fluid)
    
    # PML domains (absorbing boundaries)
    PML_WATER = 21  # PML for water domain
    PML_AIR = 22    # PML for air domain
    PML_BATH = 23   # PML for bath domain
    PML_TOP = 24    # PML top boundary
    PML_BOTTOM = 25 # PML bottom boundary
    PML_LEFT = 26   # PML left boundary
    PML_RIGHT = 27  # PML right boundary
    
    # Special regions
    TRANSDUCER = 91  # Transducer actuation region
    VOID = 99        # Empty/exterior (not meshed)
    
    def __str__(self) -> str:
        return self.name
    
    @property
    def is_fluid(self) -> bool:
        """Check if domain is a fluid."""
        return self.value in (1, 2, 3)
    
    @property
    def is_solid(self) -> bool:
        """Check if domain is an elastic solid."""
        return self.value in (11, 12, 13)
    
    @property
    def is_pml(self) -> bool:
        """Check if domain is a PML region."""
        return 21 <= self.value <= 30
    
    @property
    def is_water_like(self) -> bool:
        """Check if domain has water acoustic properties."""
        return self.value in (1, 3)
    
    @property
    def is_air_like(self) -> bool:
        """Check if domain has air acoustic properties."""
        return self.value == 2
    
    @property
    def greek_symbol(self) -> str:
        """Return mathematical notation for domain."""
        symbols = {
            1: "Ωw",
            2: "Ωa", 
            3: "Ωb",
            11: "Ωp",
            12: "Ωs",
            13: "Ωlens",
            21: "Ωpml,w",
            22: "Ωpml,a",
            23: "Ωpml,b",
            24: "Ωpml,top",
            25: "Ωpml,bot",
            26: "Ωpml,left",
            27: "Ωpml,right",
            91: "Ωtrans",
        }
        return symbols.get(self.value, "Ω?")
    
    @property 
    def gmsh_tag(self) -> int:
        """Gmsh physical group tag for this domain."""
        return self.value


class Interface(IntEnum):
    """
    Interface type enumeration for boundaries between domains.
    
    Use this enum EXCLUSIVELY for interface identification.
    Never use raw integers for interface IDs.
    
    Numbering convention:
    - Fluid-fluid: 101-110
    - Fluid-solid: 111-120
    - Solid-solid: 121-130
    - External: 131-140
    - Special: 191+
    
    All interfaces are 2D surfaces (facets) in 3D.
    """
    # Fluid-fluid interfaces
    WATER_AIR = 101      # Γwa: water–air interface
    WATER_BATH = 102     # Γwb: water–bath (through plate)
    BATH_AIR = 103       # Γba: bath–air (external)
    
    # Fluid-solid interfaces (CRITICAL for coupling)
    WATER_PLATE = 111    # Γwp: water–plate bottom interface
    WATER_WALL = 112     # Γww: water–side wall interface  
    BATH_PLATE = 113     # Γbp: bath–plate bottom interface
    BATH_LENS = 114      # Γbl: bath–lens interface
    AIR_WALL = 115       # Γaw: air–wall interface
    
    # Solid-solid interfaces
    PLATE_WALL = 121     # Junction between plate and wall
    
    # External boundaries
    OUTER = 131          # External boundary (terminated by PML)
    PML_INNER = 132      # Inner boundary of PML region
    PML_OUTER = 133      # Outer boundary of PML (zero BC)
    
    # Special boundaries
    ACTUATION = 191      # Where mechanical actuation is applied
    SYMMETRY = 192       # Symmetry plane (if used)
    
    def __str__(self) -> str:
        return self.name
    
    @property
    def is_fluid_fluid(self) -> bool:
        """Check if interface is between two fluids."""
        return 101 <= self.value <= 110
    
    @property
    def is_fluid_solid(self) -> bool:
        """Check if interface is fluid-solid coupling."""
        return 111 <= self.value <= 120
    
    @property
    def is_solid_solid(self) -> bool:
        """Check if interface is between two solids."""
        return 121 <= self.value <= 130
    
    @property
    def is_external(self) -> bool:
        """Check if interface is an external boundary."""
        return 131 <= self.value <= 140
    
    @property
    def greek_symbol(self) -> str:
        """Return mathematical notation for interface."""
        symbols = {
            101: "Γwa",
            102: "Γwb",
            103: "Γba",
            111: "Γwp",
            112: "Γww",
            113: "Γbp",
            114: "Γbl",
            115: "Γaw",
            121: "Γpw",
            131: "Γouter",
            132: "Γpml,in",
            133: "Γpml,out",
            191: "Γact",
            192: "Γsym",
        }
        return symbols.get(self.value, "Γ?")
    
    @property
    def gmsh_tag(self) -> int:
        """Gmsh physical group tag for this interface."""
        return self.value


@dataclass
class DomainInfo:
    """
    Information about a domain in the mesh.
    """
    domain: Domain
    num_cells: int = 0
    volume: float = 0.0
    centroid: Optional[np.ndarray] = None
    
    @property
    def name(self) -> str:
        return self.domain.name
    
    @property
    def tag(self) -> int:
        return self.domain.gmsh_tag


@dataclass
class InterfaceInfo:
    """
    Information about an interface in the mesh.
    """
    interface: Interface
    domain_minus: Domain  # Domain on negative normal side
    domain_plus: Domain   # Domain on positive normal side
    num_facets: int = 0
    area: float = 0.0
    
    @property
    def name(self) -> str:
        return self.interface.name
    
    @property
    def tag(self) -> int:
        return self.interface.gmsh_tag


def get_domains_for_physics_level(level: int) -> Set[Domain]:
    """
    Get the set of domains required for a physics level.
    
    Parameters
    ----------
    level : int
        Physics level (1-7)
        
    Returns
    -------
    Set[Domain]
        Set of required domains
    """
    from .config import PhysicsLevel
    
    # Level 1: ACOUSTICS_ONLY - just water
    domains = {Domain.WATER}
    
    # Level 2: ACOUSTICS_PML - add PML
    if level >= PhysicsLevel.ACOUSTICS_PML:
        domains.update({Domain.PML_WATER, Domain.PML_TOP, Domain.PML_BOTTOM,
                       Domain.PML_LEFT, Domain.PML_RIGHT})
    
    # Level 3: FLUID_AIR_BATH - add air and bath
    if level >= PhysicsLevel.FLUID_AIR_BATH:
        domains.update({Domain.AIR, Domain.BATH, Domain.PML_AIR, Domain.PML_BATH})
    
    # Level 4: FLUID_SOLID - add plate and walls
    if level >= PhysicsLevel.FLUID_SOLID:
        domains.update({Domain.PLATE, Domain.WALL, Domain.LENS, Domain.TRANSDUCER})
    
    return domains


def get_interfaces_for_physics_level(level: int) -> Set[Interface]:
    """
    Get the set of interfaces required for a physics level.
    
    Parameters
    ----------
    level : int
        Physics level (1-7)
        
    Returns
    -------
    Set[Interface]
        Set of required interfaces
    """
    from .config import PhysicsLevel
    
    # Level 1-2: Just external boundaries
    interfaces = {Interface.OUTER}
    
    # Level 2: PML boundaries
    if level >= PhysicsLevel.ACOUSTICS_PML:
        interfaces.update({Interface.PML_INNER, Interface.PML_OUTER})
    
    # Level 3: Fluid-fluid interfaces
    if level >= PhysicsLevel.FLUID_AIR_BATH:
        interfaces.update({Interface.WATER_AIR, Interface.BATH_AIR})
    
    # Level 4: Fluid-solid coupling interfaces
    if level >= PhysicsLevel.FLUID_SOLID:
        interfaces.update({
            Interface.WATER_PLATE, Interface.WATER_WALL,
            Interface.BATH_PLATE, Interface.BATH_LENS,
            Interface.AIR_WALL, Interface.PLATE_WALL,
            Interface.ACTUATION
        })
    
    return interfaces


# Mapping from interface to the two domains it connects
INTERFACE_DOMAINS: Dict[Interface, Tuple[Domain, Domain]] = {
    Interface.WATER_AIR: (Domain.WATER, Domain.AIR),
    Interface.WATER_BATH: (Domain.WATER, Domain.BATH),
    Interface.BATH_AIR: (Domain.BATH, Domain.AIR),
    Interface.WATER_PLATE: (Domain.WATER, Domain.PLATE),
    Interface.WATER_WALL: (Domain.WATER, Domain.WALL),
    Interface.BATH_PLATE: (Domain.BATH, Domain.PLATE),
    Interface.BATH_LENS: (Domain.BATH, Domain.LENS),
    Interface.AIR_WALL: (Domain.AIR, Domain.WALL),
    Interface.PLATE_WALL: (Domain.PLATE, Domain.WALL),
}


def get_interface_between(domain1: Domain, domain2: Domain) -> Optional[Interface]:
    """
    Get the interface between two domains.
    
    Parameters
    ----------
    domain1, domain2 : Domain
        The two domains
        
    Returns
    -------
    Interface or None
        The interface connecting them, or None if not directly connected
    """
    for interface, (d1, d2) in INTERFACE_DOMAINS.items():
        if (d1 == domain1 and d2 == domain2) or (d1 == domain2 and d2 == domain1):
            return interface
    return None


def validate_domain_tags(cell_tags: "np.ndarray") -> Dict[str, int]:
    """
    Validate that all cell tags correspond to known domains.
    
    Parameters
    ----------
    cell_tags : np.ndarray
        Array of cell domain tags from mesh
        
    Returns
    -------
    Dict[str, int]
        Dictionary mapping domain names to counts
    """
    unique_tags = np.unique(cell_tags)
    valid_tags = {d.value for d in Domain}
    
    counts = {}
    for tag in unique_tags:
        if tag not in valid_tags:
            raise ValueError(f"Unknown domain tag: {tag}. Valid tags: {valid_tags}")
        domain = Domain(tag)
        counts[domain.name] = np.sum(cell_tags == tag)
    
    return counts
