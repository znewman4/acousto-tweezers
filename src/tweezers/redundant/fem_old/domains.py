"""
Domain tagging abstraction for FEM simulation.

This module provides the SINGLE source of truth for domain labels.
NO MAGIC INTEGERS in FEM forms - always use Domain enum.

Domains (from MASTER BRIEF):
- Domain.WATER: Ωw — dish water (particle lives here)
- Domain.AIR: Ωa — air above water (explicit)
- Domain.BATH: Ωb — external bath water
- Domain.PLATE: Ωp — bottom plate (elastic, lossy)
- Domain.WALL: Ωs — side walls (elastic, lossy)
- Domain.PML: PML absorbing boundary regions

Interfaces (from MASTER BRIEF):
- Γwa — water–air
- Γws — water–solid (plate and walls)
- Γbs — bath–solid
- Γba — bath–air (if present)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, auto
from typing import Tuple, List, Optional
import numpy as np


class DomainType(IntEnum):
    """
    Domain type enumeration.
    
    Use this enum EXCLUSIVELY for domain identification.
    Never use raw integers for domain IDs.
    
    Numbering convention:
    - Fluids: 1-10
    - Solids: 11-20
    - PML regions: 21-30
    - Special: 90+
    """
    # Fluid domains
    WATER = 1       # Ωw: dish water where particles live
    AIR = 2         # Ωa: air above water surface
    BATH = 3        # Ωb: external coupling bath below dish
    
    # Solid domains
    PLATE = 11      # Ωp: dish bottom plate (elastic)
    WALL = 12       # Ωs: dish side walls (elastic)
    
    # PML domains (absorbing boundaries)
    PML_WATER = 21  # PML for water domain
    PML_AIR = 22    # PML for air domain
    PML_BATH = 23   # PML for bath domain
    
    # Special
    TRANSDUCER = 91  # Transducer source region
    VOID = 99        # Empty/exterior
    
    def __str__(self) -> str:
        return self.name
    
    @property
    def is_fluid(self) -> bool:
        """Check if domain is a fluid."""
        return self.value in (1, 2, 3)
    
    @property
    def is_solid(self) -> bool:
        """Check if domain is a solid."""
        return self.value in (11, 12, 91)
    
    @property
    def is_pml(self) -> bool:
        """Check if domain is a PML region."""
        return 21 <= self.value <= 30
    
    @property
    def is_water_like(self) -> bool:
        """Check if domain has water properties (water or bath)."""
        return self.value in (1, 3)
    
    @property
    def greek_symbol(self) -> str:
        """Return mathematical notation."""
        symbols = {
            1: "Ωw",
            2: "Ωa",
            3: "Ωb",
            11: "Ωp",
            12: "Ωs",
            21: "Ωpml,w",
            22: "Ωpml,a",
            23: "Ωpml,b",
        }
        return symbols.get(self.value, "Ω?")


class InterfaceType(IntEnum):
    """
    Interface type enumeration.
    
    Defines the type of coupling at each interface.
    """
    # Fluid-fluid interfaces
    WATER_AIR = 1       # Γwa: water-air (pressure & normal velocity continuity)
    BATH_AIR = 2        # Γba: bath-air (if present)
    
    # Fluid-solid interfaces  
    WATER_PLATE = 11    # Γwp: water-plate bottom
    WATER_WALL = 12     # Γws: water-sidewall
    BATH_PLATE = 13     # Γbp: bath-plate bottom
    BATH_WALL = 14      # Γbs: bath-sidewall
    
    # PML interfaces (internal, for bookkeeping)
    WATER_PML = 21
    AIR_PML = 22
    BATH_PML = 23
    
    def __str__(self) -> str:
        return self.name
    
    @property
    def is_fluid_fluid(self) -> bool:
        """Check if interface couples two fluids."""
        return self.value in (1, 2)
    
    @property
    def is_fluid_solid(self) -> bool:
        """Check if interface couples fluid and solid."""
        return 11 <= self.value <= 20
    
    @property
    def greek_symbol(self) -> str:
        """Return mathematical notation."""
        symbols = {
            1: "Γwa",
            2: "Γba",
            11: "Γwp",
            12: "Γws",
            13: "Γbp",
            14: "Γbs",
        }
        return symbols.get(self.value, "Γ?")


# Convenient aliases for backward compatibility and readability
class Domain:
    """
    Namespace for domain constants.
    
    Usage:
        from tweezers.fem.domains import Domain
        
        if element.domain == Domain.WATER:
            ...
    """
    WATER = DomainType.WATER
    AIR = DomainType.AIR
    BATH = DomainType.BATH
    PLATE = DomainType.PLATE
    WALL = DomainType.WALL
    PML_WATER = DomainType.PML_WATER
    PML_AIR = DomainType.PML_AIR
    PML_BATH = DomainType.PML_BATH
    TRANSDUCER = DomainType.TRANSDUCER
    VOID = DomainType.VOID


class Interface:
    """
    Namespace for interface constants.
    """
    WATER_AIR = InterfaceType.WATER_AIR
    BATH_AIR = InterfaceType.BATH_AIR
    WATER_PLATE = InterfaceType.WATER_PLATE
    WATER_WALL = InterfaceType.WATER_WALL
    BATH_PLATE = InterfaceType.BATH_PLATE
    BATH_WALL = InterfaceType.BATH_WALL


@dataclass
class DomainInfo:
    """
    Complete information about a domain region.
    
    Attributes
    ----------
    domain_type : DomainType
        Type identifier for the domain.
    name : str
        Human-readable name.
    material_id : str
        Key into material database.
    element_ids : np.ndarray
        Array of element indices belonging to this domain.
    node_ids : np.ndarray
        Array of node indices in this domain (computed from elements).
    volume : float
        Total volume of the domain [m³].
    """
    domain_type: DomainType
    name: str
    material_id: str
    element_ids: np.ndarray
    node_ids: Optional[np.ndarray] = None
    volume: float = 0.0
    
    def __post_init__(self):
        if self.element_ids is not None:
            self.element_ids = np.asarray(self.element_ids)
        if self.node_ids is not None:
            self.node_ids = np.asarray(self.node_ids)
    
    @property
    def num_elements(self) -> int:
        return len(self.element_ids) if self.element_ids is not None else 0
    
    @property
    def num_nodes(self) -> int:
        return len(self.node_ids) if self.node_ids is not None else 0


@dataclass
class InterfaceInfo:
    """
    Complete information about an interface between domains.
    
    Attributes
    ----------
    interface_type : InterfaceType
        Type identifier for the interface.
    name : str
        Human-readable name.
    domain_minus : DomainType
        Domain on the '-' side of the interface.
    domain_plus : DomainType
        Domain on the '+' side of the interface.
    facet_ids : np.ndarray
        Array of facet (face element) indices on this interface.
    normal_direction : str
        Primary normal direction ('x', 'y', 'z', or 'r' for radial).
    area : float
        Total area of the interface [m²].
    """
    interface_type: InterfaceType
    name: str
    domain_minus: DomainType
    domain_plus: DomainType
    facet_ids: np.ndarray
    normal_direction: str = "z"
    area: float = 0.0
    
    def __post_init__(self):
        if self.facet_ids is not None:
            self.facet_ids = np.asarray(self.facet_ids)
    
    @property
    def num_facets(self) -> int:
        return len(self.facet_ids) if self.facet_ids is not None else 0
    
    @property
    def domains(self) -> Tuple[DomainType, DomainType]:
        """Return the two domains this interface connects."""
        return (self.domain_minus, self.domain_plus)


def get_interface_type(domain1: DomainType, domain2: DomainType) -> Optional[InterfaceType]:
    """
    Determine the interface type between two domains.
    
    Parameters
    ----------
    domain1, domain2 : DomainType
        The two domains that meet at the interface.
    
    Returns
    -------
    interface_type : InterfaceType or None
        The interface type, or None if no valid interface exists.
    """
    # Normalize order (smaller value first)
    d1, d2 = sorted([domain1, domain2], key=lambda x: x.value)
    
    # Lookup table
    interfaces = {
        (DomainType.WATER, DomainType.AIR): InterfaceType.WATER_AIR,
        (DomainType.AIR, DomainType.BATH): InterfaceType.BATH_AIR,
        (DomainType.WATER, DomainType.PLATE): InterfaceType.WATER_PLATE,
        (DomainType.WATER, DomainType.WALL): InterfaceType.WATER_WALL,
        (DomainType.BATH, DomainType.PLATE): InterfaceType.BATH_PLATE,
        (DomainType.BATH, DomainType.WALL): InterfaceType.BATH_WALL,
    }
    
    return interfaces.get((d1, d2))


def get_domain_neighbors(domain: DomainType) -> List[DomainType]:
    """
    Return list of domains that can neighbor the given domain.
    
    This defines the valid domain topology for the Petri dish setup.
    """
    neighbors = {
        DomainType.WATER: [DomainType.AIR, DomainType.PLATE, DomainType.WALL],
        DomainType.AIR: [DomainType.WATER, DomainType.WALL, DomainType.BATH],
        DomainType.BATH: [DomainType.PLATE, DomainType.WALL, DomainType.AIR],
        DomainType.PLATE: [DomainType.WATER, DomainType.BATH, DomainType.WALL],
        DomainType.WALL: [DomainType.WATER, DomainType.AIR, DomainType.BATH, DomainType.PLATE],
    }
    return neighbors.get(domain, [])


def validate_domain_topology(domains: List[DomainInfo], interfaces: List[InterfaceInfo]) -> List[str]:
    """
    Validate that the domain topology is physically consistent.
    
    Returns list of error messages (empty if valid).
    """
    errors = []
    
    # Check that all interfaces connect valid domain pairs
    for iface in interfaces:
        expected = get_interface_type(iface.domain_minus, iface.domain_plus)
        if expected is None:
            errors.append(
                f"Invalid interface: {iface.domain_minus} and {iface.domain_plus} "
                f"cannot share an interface"
            )
        elif expected != iface.interface_type:
            errors.append(
                f"Interface type mismatch: {iface.domain_minus}-{iface.domain_plus} "
                f"should be {expected}, got {iface.interface_type}"
            )
    
    # Check that water domain exists (required)
    domain_types = {d.domain_type for d in domains}
    if DomainType.WATER not in domain_types:
        errors.append("Water domain (Ωw) is required but not present")
    
    return errors
