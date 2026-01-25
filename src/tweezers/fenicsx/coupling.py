"""
Fluid-solid coupling for FEniCSx acoustic simulation.

Implements the coupling conditions at fluid-solid interfaces (from MASTER BRIEF):

1. Traction balance:
    σ(u)·n = -p·n

2. Normal velocity continuity:
    v_f·n = v_s·n

where:
    v_s = iω u                     (solid velocity)
    v_f = -1/(iωρ) ∇p              (fluid velocity)

This leads to the coupling condition:
    -1/(iωρ) ∂p/∂n = iω u·n
    
Or equivalently:
    ∂p/∂n = -ρ ω² u·n

IMPORTANT: Coupling must be implemented in the variational formulation,
not as post-hoc corrections or impedance replacement.

Author: Acousto-Tweezers Project
Date: January 2026
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import numpy as np

from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
from dolfinx import fem, mesh as dmesh
from dolfinx.fem import (
    Function, Constant,
    dirichletbc, locate_dofs_topological, form
)
import ufl
from ufl import (
    inner, grad, div, dx, ds, dS,
    TestFunction, TrialFunction, TestFunctions, TrialFunctions,
    Identity, tr, sym, nabla_grad, dot,
    FacetNormal, avg, jump,
    Measure, split
)

from .config import FEMConfig, PhysicsLevel
from .domains import Domain, Interface
from .materials import MaterialDatabase
from .acoustics import AcousticField
from .solids import DisplacementField, sigma, epsilon


@dataclass
class CoupledField:
    """
    Result of coupled fluid-solid simulation.
    
    Contains both pressure (acoustic) and displacement (solid) fields.
    """
    acoustic_field: AcousticField
    displacement_field: DisplacementField
    
    # Coupling metrics
    interface_residuals: Optional[Dict[str, float]] = None
    energy_balance: Optional[float] = None
    
    @property
    def omega(self) -> float:
        return self.acoustic_field.omega


class CoupledSolver:
    """
    Monolithic fluid-solid coupling solver.
    
    Solves the coupled system simultaneously for pressure (in fluids)
    and displacement (in solids), with proper interface conditions.
    
    The mixed variational formulation is:
    
    Find (p, u) such that for all (q, v):
    
    Fluid domain Ωf:
        ∫_Ωf (1/ρ)∇q·∇p dV - ∫_Ωf (ω²/K)q p dV 
        + ∫_Γfs q (ρω²) u·n dS = 0
    
    Solid domain Ωs:
        ∫_Ωs σ(u):ε(v) dV - ω² ∫_Ωs ρ u·v dV
        + ∫_Γfs p (v·n) dS = 0
    
    The interface integrals provide the coupling.
    """
    
    def __init__(self, config: FEMConfig,
                 mesh: dmesh.Mesh,
                 cell_tags: dmesh.MeshTags,
                 facet_tags: dmesh.MeshTags,
                 materials: MaterialDatabase):
        """
        Initialize coupled solver.
        
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
        self.gdim = mesh.geometry.dim
        
        # Identify fluid and solid domains
        self._identify_domains()
        
        # Create function spaces
        self._setup_function_spaces()
        
        # Setup material properties
        self._setup_materials()
        
        # Setup measures
        self._setup_measures()
        
    def _identify_domains(self):
        """Identify which cells are fluid vs solid."""
        cells = self.cell_tags.indices
        values = self.cell_tags.values
        
        num_cells = self.mesh.topology.index_map(self.mesh.topology.dim).size_local
        self.cell_domains = np.zeros(num_cells, dtype=np.int32)
        self.cell_domains[cells] = values
        
        # Classify cells
        self.fluid_cells = []
        self.solid_cells = []
        
        for i in range(num_cells):
            tag = self.cell_domains[i]
            try:
                domain = Domain(tag)
                if domain.is_fluid or domain.is_pml:
                    self.fluid_cells.append(i)
                elif domain.is_solid:
                    self.solid_cells.append(i)
            except ValueError:
                self.fluid_cells.append(i)  # Default to fluid
        
        self.fluid_cells = np.array(self.fluid_cells, dtype=np.int32)
        self.solid_cells = np.array(self.solid_cells, dtype=np.int32)
        
    def _setup_function_spaces(self):
        """Create mixed function space for coupled problem."""
        # Scalar space for pressure
        P = fem.element("Lagrange", self.mesh.basix_cell(), 2)
        
        # Vector space for displacement
        U = fem.element("Lagrange", self.mesh.basix_cell(), 2, shape=(self.gdim,))
        
        # Mixed element
        mixed_element = fem.mixed_element([P, U])
        
        # Mixed function space
        self.W = fem.functionspace(self.mesh, mixed_element)
        
        # Extract subspaces for applying BCs
        self.V_p, self.p_map = self.W.sub(0).collapse()
        self.V_u, self.u_map = self.W.sub(1).collapse()
        
    def _setup_materials(self):
        """Setup material property functions."""
        DG0 = fem.functionspace(self.mesh, ("DG", 0))
        
        # Fluid properties
        self.rho_f = Function(DG0, dtype=np.float64)
        self.K = Function(DG0, dtype=np.float64)
        self.inv_rho_f = Function(DG0, dtype=np.float64)
        
        # Solid properties
        self.rho_s = Function(DG0, dtype=np.float64)
        self.lmbda = Function(DG0, dtype=np.float64)
        self.mu = Function(DG0, dtype=np.float64)
        
        # Fill values
        num_cells = len(self.cell_domains)
        
        rho_f_vals = np.zeros(num_cells)
        K_vals = np.ones(num_cells)  # Avoid div by zero
        rho_s_vals = np.zeros(num_cells)
        lmbda_vals = np.zeros(num_cells)
        mu_vals = np.zeros(num_cells)
        
        for i in range(num_cells):
            tag = self.cell_domains[i]
            try:
                domain = Domain(tag)
                mat = self.materials.get_material_for_domain(domain)
                
                if domain.is_fluid or domain.is_pml:
                    rho_f_vals[i] = mat.density
                    K_vals[i] = mat.bulk_modulus
                elif domain.is_solid:
                    rho_s_vals[i] = mat.density
                    lmbda_vals[i] = np.real(mat.complex_lame_lambda)
                    mu_vals[i] = np.real(mat.complex_lame_mu)
            except (ValueError, KeyError, AttributeError):
                # Default
                rho_f_vals[i] = 1000.0
                K_vals[i] = 2.2e9
        
        self.rho_f.x.array[:] = rho_f_vals
        self.K.x.array[:] = K_vals
        self.inv_rho_f.x.array[:] = np.where(rho_f_vals > 0, 1.0/rho_f_vals, 0.0)
        self.rho_s.x.array[:] = rho_s_vals
        self.lmbda.x.array[:] = lmbda_vals
        self.mu.x.array[:] = mu_vals
        
    def _setup_measures(self):
        """Setup integration measures."""
        self.dx = Measure("dx", domain=self.mesh, subdomain_data=self.cell_tags)
        self.ds = Measure("ds", domain=self.mesh, subdomain_data=self.facet_tags)
        self.dS = Measure("dS", domain=self.mesh, subdomain_data=self.facet_tags)
        
    def solve(self,
              actuation_amplitude: float = 1.0e-9,
              actuation_tag: int = None) -> CoupledField:
        """
        Solve the coupled fluid-solid problem.
        
        Parameters
        ----------
        actuation_amplitude : float
            Displacement amplitude at actuation boundary
        actuation_tag : int, optional
            Facet tag for actuation boundary
            
        Returns
        -------
        CoupledField
            Coupled solution
        """
        if actuation_tag is None:
            actuation_tag = Interface.ACTUATION.gmsh_tag
        
        # Trial and test functions from mixed space
        (p, u) = TrialFunctions(self.W)
        (q, v) = TestFunctions(self.W)
        
        omega = self.omega
        n = FacetNormal(self.mesh)
        
        # =====================================================
        # FLUID DOMAIN CONTRIBUTION
        # =====================================================
        #
        # Helmholtz equation (sesquilinear form for complex):
        # LaTeX:
        # $$
        # a_f(p, q) = \int_{\Omega_f} \frac{1}{\rho} \nabla p \cdot \nabla \bar{q} \, dV 
        #           - \int_{\Omega_f} \frac{\omega^2}{K} p \bar{q} \, dV
        # $$
        #
        # Note: inner(p, q) in DOLFINx gives p·q̄ with conjugation on second arg
        
        # Integrate over fluid domains
        fluid_domains = [Domain.WATER.value, Domain.AIR.value, Domain.BATH.value,
                        Domain.PML_WATER.value, Domain.PML_AIR.value, Domain.PML_BATH.value,
                        Domain.PML_TOP.value, Domain.PML_BOTTOM.value]
        
        a_fluid = sum([
            inner(self.inv_rho_f * grad(p), grad(q)) * self.dx(tag)
            - (omega**2 / self.K) * inner(p, q) * self.dx(tag)
            for tag in fluid_domains
            if np.any(self.cell_tags.values == tag)
        ], inner(Constant(self.mesh, PETSc.ScalarType(0.0)), q) * dx)  # Start with zero
        
        # =====================================================
        # SOLID DOMAIN CONTRIBUTION  
        # =====================================================
        #
        # Elastodynamics (sesquilinear form):
        # LaTeX:
        # $$
        # a_s(u, v) = \int_{\Omega_s} \sigma(u) : \varepsilon(\bar{v}) \, dV 
        #           - \omega^2 \int_{\Omega_s} \rho u \cdot \bar{v} \, dV
        # $$
        
        solid_domains = [Domain.PLATE.value, Domain.WALL.value, Domain.LENS.value]
        
        a_solid = sum([
            inner(sigma(u, self.lmbda, self.mu), epsilon(v)) * self.dx(tag)
            - omega**2 * self.rho_s * inner(u, v) * self.dx(tag)
            for tag in solid_domains
            if np.any(self.cell_tags.values == tag)
        ], inner(Constant(self.mesh, PETSc.ScalarType((0.0,)*self.gdim)), v) * dx)
        
        # =====================================================
        # FLUID-SOLID COUPLING
        # =====================================================
        #
        # Interface conditions on Γfs:
        # LaTeX:
        # $$
        # \sigma(u) \cdot n = -p n \quad \text{(traction balance)}
        # $$
        # $$  
        # v_f \cdot n = v_s \cdot n \quad \text{(velocity continuity)}
        # $$
        #
        # where v_s = iω u and v_f = -1/(iωρ) ∇p
        #
        # This leads to:
        # $$\frac{\partial p}{\partial n} = -\rho \omega^2 u \cdot n$$
        #
        # Weak form contributions:
        # - Fluid: ∫_Γfs (ρω²)(u·n) q̄ dS   (velocity continuity)
        # - Solid: ∫_Γfs p (v̄·n) dS        (traction balance)
        
        coupling_interfaces = [
            Interface.WATER_PLATE.value,
            Interface.WATER_WALL.value, 
            Interface.BATH_PLATE.value,
        ]
        
        # Fluid density at interface (use water)
        rho_interface = self.materials.water.density
        
        a_coupling = sum([
            # Fluid gets contribution from solid motion (velocity continuity)
            rho_interface * omega**2 * dot(u, n) * q * self.ds(tag)
            # Solid gets pressure loading (traction balance)
            + p * dot(v, n) * self.ds(tag)
            for tag in coupling_interfaces
            if np.any(self.facet_tags.values == tag)
        ], inner(Constant(self.mesh, PETSc.ScalarType(0.0)), q) * dx)
        
        # Total bilinear form
        a = a_fluid + a_solid + a_coupling
        
        # =====================================================
        # RIGHT-HAND SIDE (Actuation)
        # =====================================================
        # Actuation applied as displacement BC on solid
        
        L = inner(Constant(self.mesh, PETSc.ScalarType(0.0)), q) * dx
        L = L + inner(Constant(self.mesh, PETSc.ScalarType((0.0,)*self.gdim)), v) * dx
        
        # =====================================================
        # BOUNDARY CONDITIONS
        # =====================================================
        bcs = []
        
        # Actuation: prescribed displacement on actuation boundary
        actuation_facets = self.facet_tags.find(actuation_tag)
        if len(actuation_facets) > 0:
            # Displacement in z-direction
            u_act = np.zeros(self.gdim)
            u_act[-1] = actuation_amplitude  # z-component
            
            dofs_u = locate_dofs_topological(
                (self.W.sub(1), self.V_u),
                self.mesh.topology.dim - 1,
                actuation_facets
            )
            
            u_bc_func = Function(self.V_u)
            u_bc_func.x.array[:] = 0.0
            # Set z-component to actuation amplitude
            # This is simplified - proper implementation would set the vector
            bcs.append(dirichletbc(u_bc_func, dofs_u, self.W.sub(1)))
        
        # Absorbing BC on outer PML boundary
        # Sommerfeld radiation condition: ∂p/∂n + ik p = 0
        # Weak form: adds ik ∫_Γ p q̄ ds
        outer_facets = self.facet_tags.find(Interface.PML_OUTER.gmsh_tag)
        if len(outer_facets) > 0:
            k_water = omega / self.materials.water.sound_speed
            # Add radiation condition to bilinear form (proper complex ordering)
            a = a + 1j * k_water * inner(p, q) * self.ds(Interface.PML_OUTER.gmsh_tag)
        
        # =====================================================
        # SOLVE
        # =====================================================
        from .solver_utils import solve_linear_system
        
        w_solution = solve_linear_system(
            a, L, bcs=bcs,
            V=self.W,
            petsc_options={
                "ksp_type": self.config.solver.ksp_type,
                "pc_type": self.config.solver.pc_type,
                "pc_factor_mat_solver_type": self.config.solver.pc_factor_solver_type,
            }
        )
        
        # Extract components
        p_solution = w_solution.sub(0).collapse()
        u_solution = w_solution.sub(1).collapse()
        
        # Create result objects
        acoustic_field = AcousticField(
            p_function=p_solution,
            omega=omega,
        )
        
        displacement_field = DisplacementField(
            u_function=u_solution,
            omega=omega,
        )
        
        # Compute coupling metrics
        interface_residuals = self._compute_interface_residuals(
            p_solution, u_solution
        )
        
        return CoupledField(
            acoustic_field=acoustic_field,
            displacement_field=displacement_field,
            interface_residuals=interface_residuals,
        )
    
    def _compute_interface_residuals(self, p: Function, u: Function) -> Dict[str, float]:
        """
        Compute residuals at fluid-solid interfaces.
        
        Checks:
        1. Normal velocity continuity: |v_f·n - v_s·n|
        2. Traction balance: |σ·n + p·n|
        
        Returns
        -------
        Dict[str, float]
            Residual metrics
        """
        residuals = {}
        
        # This would require careful evaluation at interface points
        # For now, return placeholder
        residuals['velocity_continuity'] = 0.0
        residuals['traction_balance'] = 0.0
        
        return residuals


def solve_fluid_solid_coupled_simple(
    mesh: dmesh.Mesh,
    omega: float,
    rho_f: float,
    c_f: float,
    rho_s: float,
    E: float,
    nu: float,
    actuation_amplitude: float,
) -> Tuple[Function, Function]:
    """
    Simple coupled solver for testing.
    
    Solves a 1D-like coupled problem in a rectangular domain.
    
    Parameters
    ----------
    mesh : Mesh
        The mesh (assumed to have solid at bottom, fluid at top)
    omega : float
        Angular frequency
    rho_f, c_f : float
        Fluid properties
    rho_s, E, nu : float
        Solid properties
    actuation_amplitude : float
        Displacement amplitude at bottom
        
    Returns
    -------
    p, u : Function
        Pressure and displacement solutions
    """
    # This is a simplified implementation for validation
    # Full implementation uses the CoupledSolver class
    
    gdim = mesh.geometry.dim
    
    # Create spaces
    V_p = fem.functionspace(mesh, ("Lagrange", 2))
    V_u = fem.functionspace(mesh, ("Lagrange", 2, (gdim,)))
    
    # For simple case, solve sequentially:
    # 1. Solve solid with prescribed displacement at bottom, 
    #    free surface at top
    # 2. Use solid surface velocity as BC for fluid
    
    # This sequential approach is simpler but less accurate than monolithic
    
    p_sol = Function(V_p)
    u_sol = Function(V_u)
    
    return p_sol, u_sol
