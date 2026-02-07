#!/usr/bin/env python3
"""
Phase 1: Square Dish Acoustic Tweezers with Side-Wall Actuation

GEOMETRY:
- Square water-filled domain (2mm × 2mm × 2mm)
- Four vertical side walls = four transducers (full-span actuation)
- Bottom = realistic impedance BC (polystyrene substrate)
- Top = water-air interface impedance BC

BOUNDARY CONDITIONS:
- Side walls: ∂p/∂n = -iωρ v₀ exp(iφᵢ)  [velocity actuation]
- Bottom: ∂p/∂n = -ik (1/Z_bottom) p    [impedance BC]
- Top: ∂p/∂n = -ik (1/Z_air) p          [impedance BC]

PHYSICS:
- Frequency-domain Helmholtz: ∇²p + k²p = 0
- Linear acoustics only
- No PML (closed cavity modes)
- No elastic solids yet
- No streaming yet

OUTPUTS:
- Static images of pressure magnitude, Gor'kov potential, particle positions
- Multiple phase configurations for validation
- Mesh resolution study

Author: Acousto-Tweezers Project
Date: February 2026
"""

from pathlib import Path
import sys
from datetime import datetime
from dataclasses import dataclass
from typing import Tuple, List, Optional
from enum import Enum
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

# Add project root to path
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Check complex PETSc
from petsc4py import PETSc
if not np.issubdtype(PETSc.ScalarType, np.complexfloating):
    raise RuntimeError(
        f"This script requires complex PETSc. Got {PETSc.ScalarType}.\n"
        "Activate the complex environment: micromamba activate acousto-complex"
    )

import dolfinx
from dolfinx import fem, mesh as dmesh
from dolfinx.fem import (
    Function, FunctionSpace, Constant,
    dirichletbc, locate_dofs_topological, form, assemble_scalar
)
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
import ufl
from ufl import (
    inner, grad, dx, ds,
    TestFunction, TrialFunction
)

import gmsh
from dolfinx.io import gmshio

# Import diagnostics utilities
try:
    from diagnostics_utils import (
        find_gorkov_minima_2d, save_minima_data,
        compute_convergence_metrics
    )
    DIAGNOSTICS_AVAILABLE = True
except ImportError:
    DIAGNOSTICS_AVAILABLE = False
    print("Warning: diagnostics_utils not found. Minima detection disabled.")


# =============================================================================
# Configuration
# =============================================================================

from enum import Enum

class ImpedanceBCMode(Enum):
    """Bottom boundary condition modes for impedance verification."""
    IMPEDANCE = "impedance"  # Standard impedance BC: ∂p/∂n = -ik/Z * p
    RIGID = "rigid"          # Sound-hard rigid wall: ∂p/∂n = 0

@dataclass
class SquareDishConfig:
    """Configuration for square dish acoustic tweezers."""
    
    # Geometry (all in meters)
    Lx: float = 2.0e-3          # Width (x-direction)
    Ly: float = 2.0e-3          # Depth (y-direction)
    Lz: float = 2.0e-3          # Height (z-direction, water depth)
    
    # Physics
    frequency: float = 2.0e6     # 2 MHz
    temperature: float = 25.0    # 25°C
    
    # Water properties (25°C)
    rho_water: float = 997.0     # kg/m³
    c_water: float = 1497.0      # m/s (at 25°C)
    
    # Loss factor (small damping)
    loss_eta: float = 1e-3
    
    # Actuation
    v0_amplitude: float = 1.0e-3  # 1 mm/s transducer velocity amplitude
    
    # Bottom impedance (polystyrene substrate)
    # Z_bottom ≈ ρ_polystyrene × c_polystyrene
    # These are REAL impedances (no imaginary part for now)
    rho_polystyrene: float = 1050.0  # kg/m³
    c_polystyrene: float = 2350.0    # m/s (longitudinal)
    
    # Top impedance (water-air interface)
    rho_air: float = 1.2         # kg/m³
    c_air: float = 343.0         # m/s
    
    # Impedance boundary condition modes
    bottom_bc_mode: ImpedanceBCMode = ImpedanceBCMode.IMPEDANCE
    
    # Mesh resolution
    elements_per_wavelength: float = 12.0
    
    # Convergence study
    convergence_mode: bool = False
    mesh_refinement_levels: tuple = (0.8, 1.0, 1.25)  # Multipliers for elements_per_wavelength
    
    # Particles
    num_particles: int = 5
    particle_radius: float = 40.0e-6   # 40 μm (conservative for Gor'kov validity: ka ≈ 0.34)
    particle_density: float = 1050.0   # kg/m³ (polystyrene beads)
    particle_compressibility: float = 2.4e-10  # Pa⁻¹
    
    @property
    def omega(self) -> float:
        """Angular frequency [rad/s]."""
        return 2 * np.pi * self.frequency
    
    @property
    def wavelength(self) -> float:
        """Wavelength in water [m]."""
        return self.c_water / self.frequency
    
    @property
    def k_water(self) -> complex:
        """Complex wavenumber in water."""
        k_real = self.omega / self.c_water
        return k_real * (1 + 1j * self.loss_eta / 2)
    
    @property
    def Z_water(self) -> float:
        """Acoustic impedance of water [Pa·s/m]."""
        return self.rho_water * self.c_water
    
    @property
    def Z_bottom(self) -> float:
        """Acoustic impedance of bottom substrate [Pa·s/m]."""
        return self.rho_polystyrene * self.c_polystyrene
    
    @property
    def Z_air(self) -> float:
        """Acoustic impedance of air [Pa·s/m]."""
        return self.rho_air * self.c_air
    
    @property
    def mesh_size(self) -> float:
        """Target mesh element size [m]."""
        return self.wavelength / self.elements_per_wavelength
    
    def print_summary(self):
        """Print configuration summary."""
        print("\n" + "=" * 70)
        print("SQUARE DISH CONFIGURATION")
        print("=" * 70)
        print(f"\nGEOMETRY:")
        print(f"  Domain: {self.Lx*1e3:.2f} × {self.Ly*1e3:.2f} × {self.Lz*1e3:.2f} mm³")
        print(f"  Volume: {self.Lx*self.Ly*self.Lz*1e9:.3f} mm³")
        print(f"\nACOUSTICS:")
        print(f"  Frequency: {self.frequency/1e6:.2f} MHz")
        print(f"  Wavelength: {self.wavelength*1e3:.3f} mm")
        print(f"  Wavenumber: k = {np.real(self.k_water):.1f} + {np.imag(self.k_water):.3f}i m⁻¹")
        print(f"  Loss factor: η = {self.loss_eta:.1e}")
        print(f"\nIMPEDANCES (REAL, no imaginary part):")
        print(f"  Water: Z_w = {self.Z_water/1e6:.3f} MPa·s/m = {self.Z_water:.3e} Pa·s/m")
        print(f"  Bottom: Z_b = {self.Z_bottom/1e6:.3f} MPa·s/m = {self.Z_bottom:.3e} Pa·s/m")
        print(f"    (polystyrene: ρ={self.rho_polystyrene} kg/m³, c={self.c_polystyrene} m/s)")
        print(f"  Top: Z_a = {self.Z_air:.1f} Pa·s/m = {self.Z_air:.3e} Pa·s/m")
        print(f"    (air: ρ={self.rho_air} kg/m³, c={self.c_air} m/s)")
        print(f"  Reflection coeff (bottom): R = {abs((self.Z_bottom - self.Z_water)/(self.Z_bottom + self.Z_water)):.3f}")
        print(f"  Reflection coeff (top): R = {abs((self.Z_air - self.Z_water)/(self.Z_air + self.Z_water)):.3f}")
        print(f"  Bottom BC mode: {self.bottom_bc_mode.value}")
        print(f"\nACTUATION:")
        print(f"  Velocity amplitude: v₀ = {self.v0_amplitude*1e3:.2f} mm/s")
        print(f"  Four side walls (full span)")
        print(f"\nMESH:")
        print(f"  Elements per wavelength: {self.elements_per_wavelength:.1f}")
        print(f"  Target element size: {self.mesh_size*1e6:.1f} μm")
        print(f"\nPARTICLES:")
        print(f"  Number: {self.num_particles}")
        print(f"  Radius: {self.particle_radius*1e6:.1f} μm")
        print(f"  Density: {self.particle_density} kg/m³")
        # Gor'kov validity: ka << 1 (typically ka < 0.5 for good approximation)
        ka = np.abs(self.k_water) * self.particle_radius
        print(f"  Dimensionless size: ka = {ka:.3f} (Gor'kov valid if << 1)")
        print("=" * 70 + "\n")


# =============================================================================
# Geometry & Mesh Generation
# =============================================================================

def create_square_dish_mesh(
    config: SquareDishConfig,
    mesh_size: Optional[float] = None,
    verbose: bool = True
) -> Tuple[dmesh.Mesh, dmesh.MeshTags, dmesh.MeshTags]:
    """
    Create square box mesh with tagged boundaries.
    
    BOUNDARY TAGS:
    - 1: Left wall (x = 0)
    - 2: Right wall (x = Lx)
    - 3: Front wall (y = 0)
    - 4: Back wall (y = Ly)
    - 5: Bottom (z = 0)
    - 6: Top (z = Lz)
    
    Parameters
    ----------
    config : SquareDishConfig
        Configuration object
    mesh_size : float, optional
        Override mesh element size
    verbose : bool
        Print mesh info
        
    Returns
    -------
    mesh : dolfinx.mesh.Mesh
    cell_tags : dolfinx.mesh.MeshTags
    facet_tags : dolfinx.mesh.MeshTags
    """
    if mesh_size is None:
        mesh_size = config.mesh_size
    
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    if rank == 0:
        gmsh.initialize()
        gmsh.model.add("square_dish")
        
        # Create box
        box = gmsh.model.occ.addBox(0, 0, 0, config.Lx, config.Ly, config.Lz)
        gmsh.model.occ.synchronize()
        
        # Get all surfaces
        surfaces = gmsh.model.getEntities(dim=2)
        
        if verbose:
            print(f"[Mesh Debug] Total surfaces found: {len(surfaces)}")
        
        # Identify surfaces by bounding box
        tol = 1e-9
        left_surfaces = []
        right_surfaces = []
        front_surfaces = []
        back_surfaces = []
        bottom_surfaces = []
        top_surfaces = []
        
        for dim, tag in surfaces:
            # Get center of mass to identify which face this is
            com = gmsh.model.occ.getCenterOfMass(dim, tag)
            x, y, z = com
            
            if verbose:
                print(f"[Mesh Debug] Surface {tag}: center=({x*1e3:.4f}, {y*1e3:.4f}, {z*1e3:.4f}) mm")
            
            # Check which face based on center coordinates
            # Left: x ≈ 0
            if abs(x) < tol:
                left_surfaces.append(tag)
            # Right: x ≈ Lx
            elif abs(x - config.Lx) < tol:
                right_surfaces.append(tag)
            # Front: y ≈ 0
            elif abs(y) < tol:
                front_surfaces.append(tag)
            # Back: y ≈ Ly
            elif abs(y - config.Ly) < tol:
                back_surfaces.append(tag)
            # Bottom: z ≈ 0
            elif abs(z) < tol:
                bottom_surfaces.append(tag)
            # Top: z ≈ Lz
            elif abs(z - config.Lz) < tol:
                top_surfaces.append(tag)
        
        # Create physical groups for boundaries
        if verbose:
            print(f"[Mesh Debug] Found surfaces: Left={len(left_surfaces)}, Right={len(right_surfaces)}, "
                  f"Front={len(front_surfaces)}, Back={len(back_surfaces)}, "
                  f"Bottom={len(bottom_surfaces)}, Top={len(top_surfaces)}")
        
        if left_surfaces:
            gmsh.model.addPhysicalGroup(2, left_surfaces, 1)
            gmsh.model.setPhysicalName(2, 1, "LeftWall")
        if right_surfaces:
            gmsh.model.addPhysicalGroup(2, right_surfaces, 2)
            gmsh.model.setPhysicalName(2, 2, "RightWall")
        if front_surfaces:
            gmsh.model.addPhysicalGroup(2, front_surfaces, 3)
            gmsh.model.setPhysicalName(2, 3, "FrontWall")
        if back_surfaces:
            gmsh.model.addPhysicalGroup(2, back_surfaces, 4)
            gmsh.model.setPhysicalName(2, 4, "BackWall")
        if bottom_surfaces:
            gmsh.model.addPhysicalGroup(2, bottom_surfaces, 5)
            gmsh.model.setPhysicalName(2, 5, "Bottom")
        if top_surfaces:
            gmsh.model.addPhysicalGroup(2, top_surfaces, 6)
            gmsh.model.setPhysicalName(2, 6, "Top")
        
        # Physical group for volume
        volumes = [tag for dim, tag in gmsh.model.getEntities(dim=3)]
        gmsh.model.addPhysicalGroup(3, volumes, 1)
        gmsh.model.setPhysicalName(3, 1, "Water")
        
        # Set mesh size
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)
        
        # Generate mesh
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.optimize("Netgen")
        
        if verbose:
            print(f"\n[Mesh] Generated with element size {mesh_size*1e6:.1f} μm")
            print(f"[Mesh] Approximate elements per edge: {config.Lx / mesh_size:.1f}")
    
    # Import to DOLFINx
    mesh, cell_tags, facet_tags = gmshio.model_to_mesh(
        gmsh.model, comm, rank=0, gdim=3
    )
    
    if rank == 0:
        gmsh.finalize()
    
    if verbose and rank == 0:
        print(f"[Mesh] DOLFINx mesh: {mesh.topology.index_map(3).size_local} cells")
        unique_facet_values = np.unique(facet_tags.values)
        print(f"[Mesh] Facet tags: {len(unique_facet_values)} unique boundaries")
        if len(unique_facet_values) > 0:
            print(f"[Mesh] Boundary tags found: {unique_facet_values}")
        else:
            print(f"[Mesh] WARNING: No boundary tags found! Boundary conditions will not work.")
    
    return mesh, cell_tags, facet_tags


# =============================================================================
# Helmholtz Solver with Impedance BCs
# =============================================================================

@dataclass
class PhaseConfiguration:
    """Phase configuration for four transducers."""
    name: str
    phases: Tuple[float, float, float, float]  # (left, right, front, back) in radians
    description: str


def solve_helmholtz_square_dish(
    config: SquareDishConfig,
    mesh: dmesh.Mesh,
    facet_tags: dmesh.MeshTags,
    phase_config: PhaseConfiguration,
    verbose: bool = True
) -> Function:
    """
    Solve 3D Helmholtz equation with side-wall actuation and impedance BCs.
    
    WEAK FORM:
    ∫ (1/ρ) ∇φ·∇p dV - ∫ (k²/ρ) φ p dV
    + ∫_walls (-iωρ v₀ exp(iφᵢ)) φ dS     [actuation]
    + ∫_bottom (-ik/Z_b) φ p dS            [impedance]
    + ∫_top (-ik/Z_a) φ p dS               [impedance]
    = 0
    
    Parameters
    ----------
    config : SquareDishConfig
    mesh : dolfinx.mesh.Mesh
    facet_tags : dolfinx.mesh.MeshTags
    phase_config : PhaseConfiguration
    verbose : bool
        
    Returns
    -------
    p : dolfinx.fem.Function
        Pressure field solution
    """
    comm = mesh.comm
    rank = comm.rank
    
    if verbose and rank == 0:
        print(f"\n{'='*70}")
        print(f"SOLVING: {phase_config.name}")
        print(f"{'='*70}")
        print(f"Description: {phase_config.description}")
        print(f"Phases: Left={phase_config.phases[0]/np.pi:.2f}π, "
              f"Right={phase_config.phases[1]/np.pi:.2f}π, "
              f"Front={phase_config.phases[2]/np.pi:.2f}π, "
              f"Back={phase_config.phases[3]/np.pi:.2f}π")
    
    # Function space (P2 for accuracy)
    V = fem.functionspace(mesh, ("Lagrange", 2))
    
    if verbose and rank == 0:
        print(f"[Solver] DOFs: {V.dofmap.index_map.size_global}")
    
    # Trial and test functions
    p = TrialFunction(V)
    phi = TestFunction(V)
    
    # Constants
    rho = Constant(mesh, PETSc.ScalarType(config.rho_water))
    k = Constant(mesh, PETSc.ScalarType(config.k_water))
    omega = Constant(mesh, PETSc.ScalarType(config.omega))
    v0 = Constant(mesh, PETSc.ScalarType(config.v0_amplitude))
    
    # Impedances
    Z_bottom = Constant(mesh, PETSc.ScalarType(config.Z_bottom))
    Z_air = Constant(mesh, PETSc.ScalarType(config.Z_air))
    
    # Phases as complex exponentials
    phase_left = Constant(mesh, PETSc.ScalarType(np.exp(1j * phase_config.phases[0])))
    phase_right = Constant(mesh, PETSc.ScalarType(np.exp(1j * phase_config.phases[1])))
    phase_front = Constant(mesh, PETSc.ScalarType(np.exp(1j * phase_config.phases[2])))
    phase_back = Constant(mesh, PETSc.ScalarType(np.exp(1j * phase_config.phases[3])))
    
    # Measures
    dx_mesh = dx(domain=mesh)
    ds_mesh = ds(domain=mesh, subdomain_data=facet_tags)
    
    # BULK TERM: ∫ (1/ρ) ∇φ·∇p dV - ∫ (k²/ρ) φ p dV
    # NOTE: inner() in complex mode automatically conjugates the second argument
    a_bulk = (
        (1/rho) * inner(grad(p), grad(phi)) * dx_mesh
        - (k**2 / rho) * inner(p, phi) * dx_mesh
    )
    
    # IMPEDANCE BC (BOTTOM):
    # Standard impedance: ∂p/∂n = -ik (1/Z_b) p
    # Weak form contribution: ∫ (-ik/Z_b) p φ̄ dS_bottom
    # 
    # For RIGID mode: ∂p/∂n = 0 (no term added)
    if config.bottom_bc_mode == ImpedanceBCMode.IMPEDANCE:
        a_bottom = (-1j * k / Z_bottom) * inner(p, phi) * ds_mesh(5)
    else:  # RIGID mode
        a_bottom = 0 * inner(p, phi) * ds_mesh(5)  # Zero contribution (natural BC)
    
    # IMPEDANCE BC (TOP): ∫ (-ik/Z_a) φ p dS
    # This comes from ∂p/∂n = -ik (1/Z_a) p
    a_top = (-1j * k / Z_air) * inner(p, phi) * ds_mesh(6)
    
    # Bilinear form
    a = a_bulk + a_bottom + a_top
    
    # LINEAR FORM (RHS): Actuation on four side walls
    # Each wall: ∫ g φ̄ dS where g = -iωρ v₀ exp(iφᵢ)
    # Using inner(g, phi) which correctly handles conjugation in complex mode
    g_left = (-1j * omega * rho * v0 * phase_left)
    g_right = (-1j * omega * rho * v0 * phase_right)
    g_front = (-1j * omega * rho * v0 * phase_front)
    g_back = (-1j * omega * rho * v0 * phase_back)
    
    actuation_left = inner(g_left, phi) * ds_mesh(1)
    actuation_right = inner(g_right, phi) * ds_mesh(2)
    actuation_front = inner(g_front, phi) * ds_mesh(3)
    actuation_back = inner(g_back, phi) * ds_mesh(4)
    
    L = actuation_left + actuation_right + actuation_front + actuation_back
    
    # Assemble and solve
    if verbose and rank == 0:
        print("[Solver] Assembling system...")
    
    # Assemble system matrices
    a_compiled = form(a)
    L_compiled = form(L)
    
    A = assemble_matrix(a_compiled, bcs=[])
    A.assemble()
    b = assemble_vector(L_compiled)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    
    # Create solution vector
    p_solution = Function(V)
    
    # Setup KSP solver
    ksp = PETSc.KSP().create(mesh.comm)
    ksp.setOperators(A)
    ksp.setType("gmres")
    ksp.getPC().setType("ilu")
    ksp.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)
    if verbose:
        ksp.setMonitor(lambda ksp, its, rnorm: print(f"  Iteration {its}: residual = {rnorm:.3e}") if its % 10 == 0 else None)
    
    if verbose and rank == 0:
        print("[Solver] Solving...")
    
    # Solve
    ksp.solve(b, p_solution.x.petsc_vec)
    p_solution.x.scatter_forward()
    
    # Statistics
    p_array = p_solution.x.array
    p_max = comm.allreduce(np.max(np.abs(p_array)), op=MPI.MAX)
    p_mean = comm.allreduce(np.mean(np.abs(p_array)), op=MPI.SUM) / comm.size
    
    # L2 norm
    p_abs_sq = np.abs(p_array)**2
    p_l2_local = np.sqrt(np.sum(p_abs_sq))
    p_l2 = comm.allreduce(p_l2_local**2, op=MPI.SUM)
    p_l2 = np.sqrt(p_l2)
    
    if verbose and rank == 0:
        print(f"[Solver] Solution statistics:")
        print(f"  max|p| = {p_max:.3e} Pa")
        print(f"  mean|p| = {p_mean:.3e} Pa")
        print(f"  L2(|p|) = {p_l2:.3e}")
        print(f"{'='*70}\n")
    
    # Store diagnostics
    diagnostics = {
        'max_p': float(p_max),
        'mean_p': float(p_mean),
        'l2_p': float(p_l2)
    }
    
    return p_solution, diagnostics


# =============================================================================
# Gor'kov Potential Computation
# =============================================================================

def compute_gorkov_potential_3d(
    p_function: Function,
    config: SquareDishConfig,
    verbose: bool = True
) -> Function:
    """
    Compute Gor'kov potential from pressure field.
    
    U = (4π/3)a³ [f₁·⟨p²⟩/(2ρc²) - f₂·(3ρ/4)·⟨v²⟩]
    
    where:
    - f₁ = 1 - κ_p/κ_f  (monopole contrast)
    - f₂ = 2(ρ_p - ρ_f)/(2ρ_p + ρ_f)  (dipole contrast)
    - v = -1/(iωρ) ∇p
    
    Parameters
    ----------
    p_function : Function
        Pressure field
    config : SquareDishConfig
    verbose : bool
        
    Returns
    -------
    U_function : Function
        Gor'kov potential field
    """
    comm = p_function.function_space.mesh.comm
    rank = comm.rank
    
    if verbose and rank == 0:
        print("[Gor'kov] Computing potential...")
    
    # Particle properties
    a = config.particle_radius
    rho_p = config.particle_density
    kappa_p = config.particle_compressibility
    
    # Fluid properties
    rho_f = config.rho_water
    c_f = config.c_water
    kappa_f = 1.0 / (rho_f * c_f**2)
    omega = config.omega
    
    # Contrast factors
    f1 = 1.0 - (kappa_p / kappa_f)
    f2 = 2.0 * (rho_p - rho_f) / (2.0 * rho_p + rho_f)
    
    # Particle volume
    V_particle = (4.0 / 3.0) * np.pi * a**3
    
    if verbose and rank == 0:
        print(f"[Gor'kov] Contrast factors: f₁={f1:.3f}, f₂={f2:.3f}")
        print(f"[Gor'kov] Particle volume: {V_particle*1e18:.3f} × 10⁻¹⁸ m³")
    
    # Create function space for U (same as pressure)
    V = p_function.function_space
    U_function = Function(V)
    
    # Interpolate Gor'kov expression
    # This is done point-wise on the DOFs
    
    # Get coordinates and pressure values
    coords = V.tabulate_dof_coordinates()
    p_vals = p_function.x.array
    
    # Compute energy densities
    # E_pot = 0.25 * |p|² * κ_f
    E_pot = 0.25 * np.abs(p_vals)**2 * kappa_f
    
    # For velocity, we need gradients
    # This is tricky in DOLFINx - we'll use projection
    # Create DG space for velocity components
    V_DG = fem.functionspace(V.mesh, ("DG", 1, (3,)))
    v_field = Function(V_DG)
    
    # Project velocity: v = -1/(iωρ) ∇p
    v_expr = -1.0 / (1j * omega * rho_f) * grad(p_function)
    v_proj = fem.Expression(v_expr, V_DG.element.interpolation_points())
    v_field.interpolate(v_proj)
    
    # Get velocity at DOFs of original space
    # We need to evaluate v_field at pressure DOF coordinates
    # For simplicity, interpolate |v|² onto V space
    v_sq_expr = inner(v_field, v_field)
    V_DG0 = fem.functionspace(V.mesh, ("DG", 0))
    v_sq_func = Function(V_DG0)
    
    # Project |v|² to DG0 first, then to V
    v_sq_proj_expr = fem.Expression(v_sq_expr, V_DG0.element.interpolation_points())
    v_sq_func.interpolate(v_sq_proj_expr)
    
    # Now project to V
    v_sq_V = Function(V)
    v_sq_V_expr = fem.Expression(v_sq_func, V.element.interpolation_points())
    v_sq_V.interpolate(v_sq_V_expr)
    
    v_sq_vals = np.abs(v_sq_V.x.array)  # Take abs since it's complex type but should be real
    E_kin = 0.25 * rho_f * v_sq_vals
    
    # Gor'kov potential
    U_vals = V_particle * (f1 * E_pot - 1.5 * f2 * E_kin)
    
    # Store in function (convert to real if needed)
    U_function.x.array[:] = np.real(U_vals)
    
    # Statistics
    U_min = comm.allreduce(np.min(U_vals.real), op=MPI.MIN)
    U_max = comm.allreduce(np.max(U_vals.real), op=MPI.MAX)
    trap_depth = U_max - U_min
    
    if verbose and rank == 0:
        print(f"[Gor'kov] min(U) = {U_min:.3e} J")
        print(f"[Gor'kov] max(U) = {U_max:.3e} J")
        print(f"[Gor'kov] Trap depth = {trap_depth:.3e} J")
        print(f"[Gor'kov] Trap depth / kT (300K) = {trap_depth / (1.38e-23 * 300):.1f}\n")
    
    return U_function


# =============================================================================
# Particle Placement and Relaxation
# =============================================================================

def place_particles_initial(
    config: SquareDishConfig,
    mesh: dmesh.Mesh,
    seed: int = 42
) -> np.ndarray:
    """
    Place particles at initial positions.
    
    For now, place them in a regular pattern at mid-height.
    
    Returns
    -------
    positions : np.ndarray
        Shape (N, 3) particle positions [x, y, z]
    """
    np.random.seed(seed)
    
    # Place particles in a cross pattern at mid-height
    z_mid = config.Lz / 2
    
    if config.num_particles == 5:
        # Quinc pattern
        positions = np.array([
            [config.Lx/2, config.Ly/2, z_mid],  # Center
            [config.Lx/4, config.Ly/2, z_mid],  # Left
            [3*config.Lx/4, config.Ly/2, z_mid],  # Right
            [config.Lx/2, config.Ly/4, z_mid],  # Front
            [config.Lx/2, 3*config.Ly/4, z_mid],  # Back
        ])
    else:
        # Random placement with margin
        margin = 0.2e-3  # 0.2 mm from walls
        positions = np.random.uniform(
            [margin, margin, margin],
            [config.Lx - margin, config.Ly - margin, config.Lz - margin],
            size=(config.num_particles, 3)
        )
    
    return positions


def relax_particles_to_minima(
    positions: np.ndarray,
    U_function: Function,
    config: SquareDishConfig,
    dt: float = 1.0e-5,
    steps: int = 1000,
    verbose: bool = True
) -> np.ndarray:
    """
    Relax particles to nearby Gor'kov minima via overdamped dynamics.
    
    dx/dt = F_rad / γ = -∇U / γ
    
    where γ = 6πηa is Stokes drag coefficient.
    
    Parameters
    ----------
    positions : np.ndarray
        Initial positions, shape (N, 3)
    U_function : Function
        Gor'kov potential field
    config : SquareDishConfig
    dt : float
        Timestep for relaxation
    steps : int
        Number of relaxation steps
    verbose : bool
        
    Returns
    -------
    relaxed_positions : np.ndarray
        Final positions after relaxation
    """
    comm = U_function.function_space.mesh.comm
    rank = comm.rank
    
    if verbose and rank == 0:
        print(f"[Particles] Relaxing {config.num_particles} particles to minima...")
        print(f"[Particles] dt={dt:.2e} s, steps={steps}")
    
    # Stokes drag: γ = 6πηa
    # η_water ≈ 0.89e-3 Pa·s at 25°C
    eta = 0.89e-3  # Pa·s
    gamma = 6 * np.pi * eta * config.particle_radius
    mobility = 1.0 / gamma
    
    if verbose and rank == 0:
        print(f"[Particles] Stokes mobility: μ = {mobility:.3e} m/(N·s)")
    
    # Create DG space for gradient
    V = U_function.function_space
    mesh = V.mesh
    V_grad = fem.functionspace(mesh, ("DG", 1, (3,)))
    grad_U = Function(V_grad)
    
    # Project gradient
    grad_U_expr = -grad(U_function)  # Force = -∇U
    grad_U_proj = fem.Expression(grad_U_expr, V_grad.element.interpolation_points())
    grad_U.interpolate(grad_U_proj)
    
    # Relaxation loop
    from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
    
    tree = bb_tree(mesh, mesh.topology.dim)
    
    relaxed_positions = positions.copy()
    
    for step in range(steps):
        # Evaluate force at each particle position
        for i in range(config.num_particles):
            pos = relaxed_positions[i:i+1, :]  # Shape (1, 3)
            
            # Find cell containing point
            cell_candidates = compute_collisions_points(tree, pos)
            cells = compute_colliding_cells(mesh, cell_candidates, pos)
            
            if len(cells.links(0)) > 0:
                cell = cells.links(0)[0]
                
                # Evaluate gradient (force) and take real part
                # (Gradient of real field may be stored in complex function space)
                force = np.real(grad_U.eval(pos[0], cell))
                
                # Update position: x += μ F dt
                relaxed_positions[i] += mobility * force * dt
                
                # Clamp to domain boundaries
                relaxed_positions[i, 0] = np.clip(relaxed_positions[i, 0], 0, config.Lx)
                relaxed_positions[i, 1] = np.clip(relaxed_positions[i, 1], 0, config.Ly)
                relaxed_positions[i, 2] = np.clip(relaxed_positions[i, 2], 0, config.Lz)
    
    if verbose and rank == 0:
        displacements = np.linalg.norm(relaxed_positions - positions, axis=1)
        print(f"[Particles] Mean displacement: {np.mean(displacements)*1e6:.2f} μm")
        print(f"[Particles] Max displacement: {np.max(displacements)*1e6:.2f} μm\n")
    
    return relaxed_positions


# =============================================================================
# Visualization
# =============================================================================

def visualize_results(
    mesh: dmesh.Mesh,
    p_function: Function,
    U_function: Function,
    particle_positions: np.ndarray,
    config: SquareDishConfig,
    phase_config: PhaseConfiguration,
    output_dir: Path
):
    """
    Generate static visualization plots.
    
    Creates three plots:
    1. Pressure magnitude |p| at mid-height slice
    2. Gor'kov potential U at mid-height slice
    3. Combined plot with particle positions
    """
    comm = mesh.comm
    rank = comm.rank
    
    if rank != 0:
        return  # Only rank 0 creates plots
    
    print(f"[Visualization] Generating plots for {phase_config.name}...")
    
    # Create grid for interpolation
    z_slice = config.Lz / 2
    n_points = 200
    
    x_grid = np.linspace(0, config.Lx, n_points)
    y_grid = np.linspace(0, config.Ly, n_points)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Points to evaluate (shape N, 3)
    points = np.column_stack([
        X.ravel(),
        Y.ravel(),
        np.full(n_points**2, z_slice)
    ])
    
    # Evaluate pressure
    from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
    tree = bb_tree(mesh, mesh.topology.dim)
    cell_candidates = compute_collisions_points(tree, points)
    cells = compute_colliding_cells(mesh, cell_candidates, points)
    
    p_vals = np.full(len(points), np.nan, dtype=complex)
    U_vals = np.full(len(points), np.nan)
    
    for i in range(len(points)):
        if len(cells.links(i)) > 0:
            cell = cells.links(i)[0]
            p_vals[i] = p_function.eval(points[i], cell)[0]
            U_vals[i] = U_function.eval(points[i], cell)[0]
    
    # Reshape to grid
    P_grid = np.abs(p_vals).reshape(n_points, n_points)
    U_grid = U_vals.reshape(n_points, n_points)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Pressure magnitude
    ax = axes[0]
    im1 = ax.contourf(X*1e3, Y*1e3, P_grid, levels=50, cmap='viridis')
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    ax.set_title(f'Pressure |p| at z = {z_slice*1e3:.2f} mm')
    ax.set_aspect('equal')
    plt.colorbar(im1, ax=ax, label='|p| [Pa]')
    
    # 2. Gor'kov potential
    ax = axes[1]
    im2 = ax.contourf(X*1e3, Y*1e3, U_grid*1e15, levels=50, cmap='RdBu_r')
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    ax.set_title(f'Gor\'kov Potential U at z = {z_slice*1e3:.2f} mm')
    ax.set_aspect('equal')
    plt.colorbar(im2, ax=ax, label='U [fJ]')
    
    # 3. Combined with particles
    ax = axes[2]
    im3 = ax.contourf(X*1e3, Y*1e3, U_grid*1e15, levels=50, cmap='RdBu_r', alpha=0.7)
    
    # Overlay pressure contours
    ax.contour(X*1e3, Y*1e3, P_grid, levels=10, colors='black', alpha=0.3, linewidths=0.5)
    
    # Plot particles at mid-height
    particles_at_slice = particle_positions[np.abs(particle_positions[:, 2] - z_slice) < config.Lz/10]
    if len(particles_at_slice) > 0:
        ax.scatter(particles_at_slice[:, 0]*1e3, particles_at_slice[:, 1]*1e3,
                  s=200, c='red', marker='o', edgecolors='white', linewidths=2,
                  label='Particles', zorder=10)
    
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    ax.set_title(f'Combined: U + Particles')
    ax.set_aspect('equal')
    ax.legend()
    plt.colorbar(im3, ax=ax, label='U [fJ]')
    
    # Overall title
    fig.suptitle(f'{phase_config.name}: {phase_config.description}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    filename = f"{phase_config.name.lower().replace(' ', '_')}.png"
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[Visualization] Saved: {filepath}")


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main execution for Phase 1."""
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results/square_dish_phase1") / f"run_{timestamp}"
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[Output] Results will be saved to: {output_dir}")
    
    # Configuration
    config = SquareDishConfig()
    if rank == 0:
        config.print_summary()
        
        # Save config
        config_dict = {
            'geometry': {
                'Lx': config.Lx,
                'Ly': config.Ly,
                'Lz': config.Lz,
            },
            'physics': {
                'frequency': config.frequency,
                'wavelength': config.wavelength,
                'omega': config.omega,
                'loss_eta': config.loss_eta,
            },
            'impedances': {
                'Z_water': config.Z_water,
                'Z_bottom': config.Z_bottom,
                'Z_air': config.Z_air,
                'rho_polystyrene': config.rho_polystyrene,
                'c_polystyrene': config.c_polystyrene,
                'rho_air': config.rho_air,
                'c_air': config.c_air,
                'bottom_bc_mode': config.bottom_bc_mode.value,
                'R_bottom': float(abs((config.Z_bottom - config.Z_water)/(config.Z_bottom + config.Z_water))),
                'R_top': float(abs((config.Z_air - config.Z_water)/(config.Z_air + config.Z_water))),
            },
            'mesh': {
                'elements_per_wavelength': config.elements_per_wavelength,
                'mesh_size': config.mesh_size,
            },
            'particles': {
                'num_particles': config.num_particles,
                'radius': config.particle_radius,
                'density': config.particle_density,
                'ka': float(np.abs(config.k_water) * config.particle_radius),
            },
            'convergence': {
                'mode': config.convergence_mode,
                'levels': list(config.mesh_refinement_levels) if hasattr(config, 'mesh_refinement_levels') else [],
            }
        }
        with open(output_dir / "config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    # Generate mesh
    if rank == 0:
        print("\n" + "="*70)
        print("MESH GENERATION")
        print("="*70)
    
    mesh, cell_tags, facet_tags = create_square_dish_mesh(config, verbose=True)
    
    # Define phase configurations to test
    phase_configs = [
        PhaseConfiguration(
            name="All_In_Phase",
            phases=(0.0, 0.0, 0.0, 0.0),
            description="All walls in phase (uniform excitation)"
        ),
        PhaseConfiguration(
            name="LR_Opposite",
            phases=(0.0, np.pi, 0.0, np.pi),
            description="Left-Right out of phase (x-direction standing wave)"
        ),
        PhaseConfiguration(
            name="FB_Opposite",
            phases=(0.0, 0.0, np.pi, np.pi),
            description="Front-Back out of phase (y-direction standing wave)"
        ),
        PhaseConfiguration(
            name="Quadrature",
            phases=(0.0, np.pi/2, np.pi, 3*np.pi/2),
            description="90° phase shifts (rotating pattern)"
        ),
    ]
    
    # Place particles (initial positions)
    particle_positions_initial = place_particles_initial(config, mesh)
    
    if rank == 0:
        print("\n" + "="*70)
        print("INITIAL PARTICLE POSITIONS")
        print("="*70)
        for i, pos in enumerate(particle_positions_initial):
            print(f"  Particle {i+1}: ({pos[0]*1e3:.3f}, {pos[1]*1e3:.3f}, {pos[2]*1e3:.3f}) mm")
    
    # Solve for each phase configuration
    all_diagnostics = {}
    
    for phase_config in phase_configs:
        # Solve Helmholtz
        p_solution, field_diag = solve_helmholtz_square_dish(
            config, mesh, facet_tags, phase_config, verbose=True
        )
        
        # Store field diagnostics
        all_diagnostics[phase_config.name] = {
            'field': field_diag,
            'phase': {
                'left': float(phase_config.phases[0]),
                'right': float(phase_config.phases[1]),
                'front': float(phase_config.phases[2]),
                'back': float(phase_config.phases[3]),
            }
        }
        
        # Compute Gor'kov potential
        U_solution = compute_gorkov_potential_3d(p_solution, config, verbose=True)
        
        # Find Gor'kov minima on mid-plane
        if rank == 0 and DIAGNOSTICS_AVAILABLE:
            # Sample U on mid-plane
            z_mid = config.Lz / 2
            nx, ny = 100, 100
            x_coords = np.linspace(0, config.Lx, nx)
            y_coords = np.linspace(0, config.Ly, ny)
            U_grid = np.zeros((ny, nx))
            
            # Evaluate U at grid points
            V = U_solution.function_space
            from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
            tree = bb_tree(mesh, mesh.topology.dim)
            
            for i, x in enumerate(x_coords):
                for j, y in enumerate(y_coords):
                    pos = np.array([[x, y, z_mid]])
                    cell_candidates = compute_collisions_points(tree, pos)
                    cells = compute_colliding_cells(mesh, cell_candidates, pos)
                    if len(cells.links(0)) > 0:
                        cell = cells.links(0)[0]
                        U_grid[j, i] = U_solution.eval(pos[0], cell)[0]
            
            # Find minima
            minima_pos, minima_vals = find_gorkov_minima_2d(
                U_grid, x_coords, y_coords, n_minima=10
            )
            
            # Save minima data
            minima_file = output_dir / f"minima_{phase_config.name.lower()}.json"
            save_minima_data(
                minima_pos, minima_vals, str(minima_file),
                metadata={
                    'phase_config': phase_config.name,
                    'z_plane': z_mid,
                    'n_minima_found': len(minima_vals)
                }
            )
            
            # Store in diagnostics
            if len(minima_vals) > 0:
                all_diagnostics[phase_config.name]['gorkov'] = {
                    'n_minima': len(minima_vals),
                    'trap_depth': float(np.max(minima_vals) - np.min(minima_vals)) if len(minima_vals) > 1 else 0.0,
                    'deepest_value': float(np.min(minima_vals)),
                }
        
        # Relax particles to minima
        if rank == 0:
            particle_positions_relaxed = relax_particles_to_minima(
                particle_positions_initial,
                U_solution,
                config,
                dt=1.0e-5,
                steps=500,
                verbose=True
            )
        else:
            particle_positions_relaxed = particle_positions_initial
        
        # Visualize
        if rank == 0:
            visualize_results(
                mesh, p_solution, U_solution,
                particle_positions_relaxed,
                config, phase_config,
                output_dir
            )
    
    # Save all diagnostics
    if rank == 0:
        diag_file = output_dir / "diagnostics.json"
        with open(diag_file, 'w') as f:
            json.dump(all_diagnostics, f, indent=2)
        print(f"\\n[Diagnostics] Saved to: {diag_file}")
    
    if rank == 0:
        print("\n" + "="*70)
        print("PHASE 1.5 COMPLETE - DIAGNOSTICS ENABLED")
        print("="*70)
        print(f"\nResults saved to: {output_dir}")
        print(f"\nGenerated {len(phase_configs)} diagnostic configurations")
        print(f"\nDiagnostics:")
        print(f"  ✓ Field scalars (max|p|, mean|p|, L2 norm)")
        print(f"  ✓ Gor'kov minima detection (mid-plane)")
        print(f"  ✓ Impedance BC documentation")
        print(f"\nImpedance BCs:")
        print(f"  Bottom: {config.bottom_bc_mode.value}")
        print(f"    Z_b = {config.Z_bottom/1e6:.3f} MPa·s/m")
        print(f"    R = {abs((config.Z_bottom - config.Z_water)/(config.Z_bottom + config.Z_water)):.3f}")
        print(f"  Top: impedance")
        print(f"    Z_a = {config.Z_air:.1f} Pa·s/m") 
        print(f"    R = {abs((config.Z_air - config.Z_water)/(config.Z_air + config.Z_water)):.3f}")
        print(f"\nParticle radius: {config.particle_radius*1e6:.1f} μm")
        print(f"  ka = {np.abs(config.k_water) * config.particle_radius:.3f} (Gor'kov valid if << 1)")
        print(f"\nNext steps:")
        print(f"  1. Review diagnostics.json and minima_*.json")
        print(f"  2. Run convergence: python scripts/run_phase1_5.py --convergence")
        print(f"  3. Test impedance: python scripts/run_phase1_5.py --impedance-test")
        print("="*70 + "\n")


if __name__ == "__main__":
    main()
