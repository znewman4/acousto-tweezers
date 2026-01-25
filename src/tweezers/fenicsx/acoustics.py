"""
FEniCSx acoustic solver for first-order pressure acoustics.

Implements the Helmholtz equation in the frequency domain using UFL
and DOLFINx:

    ∇·(1/ρ ∇p) + ω²/(ρc²) p = 0

Weak form (from MASTER BRIEF):
    ∫_Ω (1/ρ) ∇φ·∇p dV - ∫_Ω (ω²/K) φ p dV + boundary terms = 0

where K = ρc² is the bulk modulus.

Fluid-fluid interface conditions:
- Pressure continuity: p₁ = p₂  (natural via continuous elements)
- Normal velocity continuity: (1/ρ₁) ∂p₁/∂n = (1/ρ₂) ∂p₂/∂n

IMPORTANT: All PDEs are defined in UFL and assembled/solved by DOLFINx + PETSc.
No homebrew FEM, no Python element loops, no manual sparse assembly.

Author: Acousto-Tweezers Project
Date: January 2026
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Callable
import numpy as np

from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
from dolfinx import fem, mesh as dmesh
from dolfinx.fem import (
    Function, FunctionSpace, Constant,
    dirichletbc, locate_dofs_topological, form
)
from .solver_utils import solve_linear_system
import ufl
from ufl import (
    inner, grad, dx, ds, dS,
    TestFunction, TrialFunction,
    FacetNormal, avg, jump,
    Measure
)

from .config import FEMConfig, PhysicsLevel
from .domains import Domain, Interface
from .materials import MaterialDatabase, FluidMaterial


@dataclass
class AcousticField:
    """
    Solution of the acoustic field.
    
    Contains pressure and derived quantities computed from the FEniCSx solution.
    """
    # DOLFINx function containing pressure solution
    p_function: Function
    
    # Angular frequency
    omega: float
    
    # Material property functions
    rho_function: Optional[Function] = None
    c_function: Optional[Function] = None
    
    # Cached numpy arrays for post-processing
    _p_array: Optional[np.ndarray] = None
    _coords: Optional[np.ndarray] = None
    
    @property
    def p(self) -> np.ndarray:
        """Pressure values at DOFs (complex)."""
        if self._p_array is None:
            self._p_array = self.p_function.x.array.copy()
        return self._p_array
    
    @property
    def mesh(self):
        """The underlying mesh."""
        return self.p_function.function_space.mesh
    
    @property
    def coords(self) -> np.ndarray:
        """Coordinates of DOF points."""
        if self._coords is None:
            V = self.p_function.function_space
            self._coords = V.tabulate_dof_coordinates()
        return self._coords
    
    @property
    def max_pressure(self) -> float:
        """Maximum pressure amplitude."""
        return np.max(np.abs(self.p))
    
    @property
    def mean_pressure(self) -> float:
        """Mean pressure amplitude."""
        return np.mean(np.abs(self.p))
    
    @property
    def rms_pressure(self) -> float:
        """RMS pressure."""
        return np.sqrt(np.mean(np.abs(self.p)**2))
    
    def evaluate_at_points(self, points: np.ndarray) -> np.ndarray:
        """
        Evaluate pressure at arbitrary points.
        
        Parameters
        ----------
        points : np.ndarray
            Points to evaluate at, shape (N, 3)
            
        Returns
        -------
        np.ndarray
            Pressure values at points (complex)
        """
        from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
        
        mesh = self.mesh
        tree = bb_tree(mesh, mesh.topology.dim)
        
        # Find cells containing points
        cell_candidates = compute_collisions_points(tree, points)
        cells = compute_colliding_cells(mesh, cell_candidates, points)
        
        # Evaluate
        p_values = np.zeros(len(points), dtype=np.complex128)
        for i, point in enumerate(points):
            if len(cells.links(i)) > 0:
                cell = cells.links(i)[0]
                p_values[i] = self.p_function.eval(point, cell)[0]
        
        return p_values


class AcousticSolver:
    """
    FEniCSx solver for the Helmholtz equation.
    
    Solves:
        ∇·(1/ρ ∇p) + ω²/K p = 0
        
    with domain-dependent material properties and appropriate
    boundary conditions.
    """
    
    def __init__(self, config: FEMConfig,
                 mesh: dmesh.Mesh,
                 cell_tags: dmesh.MeshTags,
                 facet_tags: dmesh.MeshTags,
                 materials: MaterialDatabase):
        """
        Initialize acoustic solver.
        
        Parameters
        ----------
        config : FEMConfig
            Simulation configuration
        mesh : dolfinx.mesh.Mesh
            The computational mesh
        cell_tags : dolfinx.mesh.MeshTags
            Domain tags for cells
        facet_tags : dolfinx.mesh.MeshTags
            Interface tags for facets
        materials : MaterialDatabase
            Material property database
        """
        self.config = config
        self.mesh = mesh
        self.cell_tags = cell_tags
        self.facet_tags = facet_tags
        self.materials = materials
        
        self.omega = config.physics.omega
        self.comm = mesh.comm
        
        # Create function space (complex-valued for frequency domain)
        # Using P2 elements for better accuracy
        self.V = fem.functionspace(mesh, ("Lagrange", 2))
        
        # Setup material property fields
        self._setup_materials()
        
        # Create measures with domain/facet tags
        self._setup_measures()
        
    def _setup_materials(self):
        """Set up material property functions."""
        from petsc4py import PETSc
        
        # Create DG0 function space for piecewise constant properties
        DG0 = fem.functionspace(self.mesh, ("DG", 0))
        
        # IMPORTANT: For complex PETSc, material properties should also be complex
        # to avoid mixed float64/complex128 issues in form compilation
        dtype = PETSc.ScalarType
        
        # Density field
        self.rho = Function(DG0, dtype=dtype)
        # Bulk modulus field (K = ρc²)
        self.K = Function(DG0, dtype=dtype)
        # Inverse density for weak form
        self.inv_rho = Function(DG0, dtype=dtype)
        
        # Get cell-to-tag mapping
        cells = self.cell_tags.indices
        values = self.cell_tags.values
        
        # Create lookup for all cells
        num_cells = self.mesh.topology.index_map(self.mesh.topology.dim).size_local
        cell_domains = np.zeros(num_cells, dtype=np.int32)
        cell_domains[cells] = values
        
        # Assign material properties based on domain
        rho_values = np.zeros(num_cells)
        K_values = np.zeros(num_cells)
        
        for i in range(num_cells):
            domain_tag = cell_domains[i]
            try:
                domain = Domain(domain_tag)
                mat = self.materials.get_material_for_domain(domain)
                if hasattr(mat, 'sound_speed'):  # FluidMaterial
                    rho_values[i] = mat.density
                    K_values[i] = mat.bulk_modulus
                else:  # SolidMaterial - use longitudinal properties
                    rho_values[i] = mat.density
                    K_values[i] = mat.bulk_modulus
            except (ValueError, KeyError):
                # Default to water
                rho_values[i] = self.materials.water.density
                K_values[i] = self.materials.water.bulk_modulus
        
        self.rho.x.array[:] = rho_values
        self.K.x.array[:] = K_values
        self.inv_rho.x.array[:] = 1.0 / rho_values
        
    def _setup_measures(self):
        """Set up integration measures with tags."""
        self.dx = Measure("dx", domain=self.mesh, subdomain_data=self.cell_tags)
        self.ds = Measure("ds", domain=self.mesh, subdomain_data=self.facet_tags)
        self.dS = Measure("dS", domain=self.mesh, subdomain_data=self.facet_tags)
        
    def solve(self, 
              source_function: Optional[Callable] = None,
              boundary_conditions: Optional[Dict] = None) -> AcousticField:
        """
        Solve the Helmholtz equation.
        
        Parameters
        ----------
        source_function : callable, optional
            Source term function f(x) for ∇·(1/ρ ∇p) + ω²/K p = f
        boundary_conditions : dict, optional
            Dictionary of boundary conditions
            
        Returns
        -------
        AcousticField
            Solution containing pressure field
        """
        # Trial and test functions
        p = TrialFunction(self.V)
        v = TestFunction(self.V)
        
        # Weak form of Helmholtz equation (complex sesquilinear form):
        #
        # LaTeX:
        # $$
        # a(p, v) = \int_\Omega \frac{1}{\rho} \nabla p \cdot \nabla \bar{v} \, dV 
        #           - \int_\Omega \frac{\omega^2}{K} p \bar{v} \, dV
        # $$
        #
        # where K = ρc² is the bulk modulus.
        #
        # IMPORTANT: For DOLFINx complex mode, ufl.inner(a, b) computes a·b̄
        # So we write inner(p, v) which gives p·v̄ with conjugation on test.
        # Similarly inner(grad(p), grad(v)) gives ∇p · ∇v̄.
        #
        # Reference: Ihlenburg (1998), Finite Element Analysis of Acoustic Scattering
        
        omega = self.omega
        
        # Bilinear form (sesquilinear in complex)
        # a(p, v) = ∫(1/ρ)∇p·∇v̄ - (ω²/K)p·v̄ dV
        a = (
            inner(self.inv_rho * grad(p), grad(v)) * dx
            - (omega**2 / self.K) * inner(p, v) * dx
        )
        
        # Right-hand side (source terms)
        if source_function is not None:
            # Create source function
            f = Function(self.V)
            f.interpolate(source_function)
            L = inner(f, v) * dx
        else:
            # Zero RHS - actuation comes through BCs
            L = inner(Constant(self.mesh, PETSc.ScalarType(0.0)), v) * dx
        
        # Boundary conditions
        bcs = []
        
        if boundary_conditions is not None:
            for bc_type, bc_data in boundary_conditions.items():
                if bc_type == "dirichlet":
                    # Dirichlet BC: p = p_0 on boundary
                    for tag, value in bc_data.items():
                        facets = self.facet_tags.find(tag)
                        dofs = locate_dofs_topological(
                            self.V, self.mesh.topology.dim - 1, facets
                        )
                        if callable(value):
                            bc_func = Function(self.V)
                            bc_func.interpolate(value)
                            bcs.append(dirichletbc(bc_func, dofs))
                        else:
                            bcs.append(dirichletbc(
                                PETSc.ScalarType(value), dofs, self.V
                            ))
                            
                elif bc_type == "neumann":
                    # Neumann BC: (1/ρ)∂p/∂n = g on boundary
                    # Add surface integral to RHS
                    for tag, value in bc_data.items():
                        if callable(value):
                            g = Function(self.V)
                            g.interpolate(value)
                        else:
                            g = Constant(self.mesh, PETSc.ScalarType(value))
                        # Note: this modifies L (RHS) - proper ordering for complex
                        L = L + inner(g, v) * self.ds(tag)
        
        # Add PML or absorbing boundary condition on outer boundaries
        # First-order absorbing BC (Sommerfeld radiation condition):
        #
        # LaTeX:
        # $$\frac{\partial p}{\partial n} + ik p = 0 \quad \text{on } \Gamma$$
        #
        # In weak form: adds ik ∫_Γ p v̄ ds to bilinear form
        if self.config.physics_level.value >= PhysicsLevel.ACOUSTICS_PML:
            # Get wavenumber for water
            c_water = self.materials.water.sound_speed
            k = omega / c_water
            
            # Apply ABC on outer boundary (proper ordering for complex)
            outer_tag = Interface.PML_OUTER.gmsh_tag
            a = a + 1j * k * inner(p, v) * self.ds(outer_tag)
        
        # Assemble and solve
        p_solution = solve_linear_system(
            a, L, bcs=bcs,
            V=self.V,
            petsc_options={
                "ksp_type": self.config.solver.ksp_type,
                "pc_type": self.config.solver.pc_type,
                "pc_factor_mat_solver_type": self.config.solver.pc_factor_solver_type,
            }
        )
        
        # Create result object
        result = AcousticField(
            p_function=p_solution,
            omega=omega,
            rho_function=self.rho,
            c_function=None,  # Could compute from K/rho
        )
        
        return result
    
    def solve_with_actuation(self, 
                             actuation_amplitude: float,
                             actuation_phase: float = 0.0) -> AcousticField:
        """
        Solve with mechanical actuation boundary condition.
        
        The actuation is applied as a prescribed normal velocity on the
        actuation boundary, which translates to a Neumann condition:
        
            (1/ρ) ∂p/∂n = -iω v_n
            
        where v_n is the normal velocity amplitude.
        
        Parameters
        ----------
        actuation_amplitude : float
            Normal velocity amplitude [m/s]
        actuation_phase : float
            Phase of actuation [radians]
            
        Returns
        -------
        AcousticField
            Solution
        """
        # Convert displacement to velocity: v = iω u
        # If user specifies displacement amplitude u_0:
        # v_0 = ω u_0
        if self.config.physics.actuation_type == "displacement":
            v_amplitude = self.omega * actuation_amplitude
        else:
            v_amplitude = actuation_amplitude
        
        # Complex amplitude with phase
        v_n = v_amplitude * np.exp(1j * actuation_phase)
        
        # Neumann BC: (1/ρ) ∂p/∂n = -iω v_n
        # The flux value is: g = -iω ρ v_n
        rho_bath = self.materials.water.density  # Bath material
        g_value = -1j * self.omega * rho_bath * v_n
        
        # For ACOUSTICS_ONLY (level 1), actuation is on WATER_PLATE interface
        # For higher levels, use dedicated ACTUATION interface
        if self.config.physics_level.value < 4:
            # Use bottom of water (WATER_PLATE) as actuation surface
            actuation_tag = Interface.WATER_PLATE.gmsh_tag
        else:
            # Use dedicated actuation interface
            actuation_tag = Interface.ACTUATION.gmsh_tag
        
        boundary_conditions = {
            "neumann": {
                actuation_tag: g_value,
            }
        }
        
        return self.solve(boundary_conditions=boundary_conditions)


def solve_helmholtz_simple(mesh: dmesh.Mesh,
                           omega: float,
                           rho: float,
                           c: float,
                           source: Optional[Callable] = None) -> Function:
    """
    Simple Helmholtz solver for testing.
    
    Solves in a single-material domain with zero Dirichlet BC on boundary.
    
    Parameters
    ----------
    mesh : Mesh
        The mesh
    omega : float
        Angular frequency
    rho : float
        Density
    c : float
        Sound speed
    source : callable, optional
        Source function
        
    Returns
    -------
    Function
        Pressure solution
    """
    V = fem.functionspace(mesh, ("Lagrange", 2))
    
    p = TrialFunction(V)
    v = TestFunction(V)
    
    K = rho * c**2
    k = omega / c
    
    # Weak form
    a = inner(grad(v), grad(p)) * dx - k**2 * inner(v, p) * dx
    
    if source is not None:
        f = Function(V)
        f.interpolate(source)
        L = inner(v, f) * dx
    else:
        L = inner(v, Constant(mesh, PETSc.ScalarType(0.0))) * dx
    
    # Zero Dirichlet BC on boundary
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    boundary_facets = dmesh.exterior_facet_indices(mesh.topology)
    boundary_dofs = locate_dofs_topological(
        V, mesh.topology.dim - 1, boundary_facets
    )
    bc = dirichletbc(PETSc.ScalarType(0.0), boundary_dofs, V)
    
    return solve_linear_system(a, L, bcs=[bc], V=V)
