"""
FEniCSx elastic solid solver for frequency-domain elasticity.

Implements the frequency-domain elastic wave equation (from MASTER BRIEF):

    ∇·σ(u) + ρ_s ω² u = 0

where the stress tensor is:
    σ = λ(∇·u)I + 2μ ε(u)

with symmetric strain tensor:
    ε(u) = (1/2)(∇u + ∇uᵀ)

Viscoelastic damping via complex modulus:
    E → E(1 + iη)

This affects both Lamé parameters λ and μ.

CRITICAL: The displacement field u is complex-valued and vector-valued.

Author: Acousto-Tweezers Project
Date: January 2026
"""

from __future__ import annotations

from dataclasses import dataclass
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
    inner, grad, div, dx, ds, dS,
    TestFunction, TrialFunction,
    Identity, tr, sym, nabla_grad,
    Measure
)

from .config import FEMConfig, PhysicsLevel
from .domains import Domain, Interface
from .materials import MaterialDatabase, SolidMaterial


@dataclass  
class DisplacementField:
    """
    Solution of the elastic displacement field.
    
    The displacement is complex-valued (frequency domain) and vector-valued (3D).
    """
    # DOLFINx function containing displacement solution
    u_function: Function
    
    # Angular frequency
    omega: float
    
    # Material property functions
    rho_function: Optional[Function] = None
    lambda_function: Optional[Function] = None
    mu_function: Optional[Function] = None
    
    @property
    def u(self) -> np.ndarray:
        """Displacement values at DOFs (complex, flattened)."""
        return self.u_function.x.array.copy()
    
    @property
    def mesh(self):
        """The underlying mesh."""
        return self.u_function.function_space.mesh
    
    def velocity(self) -> np.ndarray:
        """
        Compute velocity from displacement.
        
        v = iω u
        
        Returns
        -------
        np.ndarray
            Complex velocity values
        """
        return 1j * self.omega * self.u
    
    def max_displacement(self) -> float:
        """Maximum displacement magnitude."""
        # Reshape to (N, 3) and compute magnitude
        u_vec = self.u.reshape(-1, 3)
        return np.max(np.linalg.norm(u_vec, axis=1))
    
    def max_velocity(self) -> float:
        """Maximum velocity magnitude."""
        v_vec = self.velocity().reshape(-1, 3)
        return np.max(np.linalg.norm(np.abs(v_vec), axis=1))


def epsilon(u):
    """
    Symmetric strain tensor.
    
    ε(u) = (1/2)(∇u + ∇uᵀ) = sym(∇u)
    """
    return sym(nabla_grad(u))


def sigma(u, lmbda, mu):
    """
    Cauchy stress tensor (isotropic linear elasticity).
    
    σ = λ(∇·u)I + 2μ ε(u)
    
    Parameters
    ----------
    u : UFL expression
        Displacement field
    lmbda : UFL expression
        First Lamé parameter
    mu : UFL expression
        Second Lamé parameter (shear modulus)
        
    Returns
    -------
    UFL expression
        Stress tensor
    """
    d = u.ufl_shape[0]  # Spatial dimension
    return lmbda * div(u) * Identity(d) + 2 * mu * epsilon(u)


class SolidSolver:
    """
    FEniCSx solver for frequency-domain elasticity.
    
    Solves:
        ∇·σ(u) + ρ ω² u = 0
        
    with complex-valued displacement for viscoelastic materials.
    """
    
    def __init__(self, config: FEMConfig,
                 mesh: dmesh.Mesh,
                 cell_tags: dmesh.MeshTags,
                 facet_tags: dmesh.MeshTags,
                 materials: MaterialDatabase):
        """
        Initialize solid solver.
        
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
        
        # Create vector function space for displacement
        # Using P2 elements for accuracy
        self.V = fem.functionspace(mesh, ("Lagrange", 2, (self.gdim,)))
        
        # Setup material property fields
        self._setup_materials()
        
        # Create measures
        self._setup_measures()
        
    def _setup_materials(self):
        """Set up material property functions for solids."""
        # DG0 for piecewise constant properties
        DG0 = fem.functionspace(self.mesh, ("DG", 0))
        
        # Density
        self.rho = Function(DG0, dtype=np.float64)
        
        # Complex Lamé parameters (for viscoelastic damping)
        # We'll store real and imaginary parts separately
        self.lmbda_re = Function(DG0, dtype=np.float64)
        self.lmbda_im = Function(DG0, dtype=np.float64)
        self.mu_re = Function(DG0, dtype=np.float64)
        self.mu_im = Function(DG0, dtype=np.float64)
        
        # Get cell-to-tag mapping
        cells = self.cell_tags.indices
        values = self.cell_tags.values
        
        num_cells = self.mesh.topology.index_map(self.mesh.topology.dim).size_local
        cell_domains = np.zeros(num_cells, dtype=np.int32)
        cell_domains[cells] = values
        
        # Assign material properties
        rho_values = np.zeros(num_cells)
        lmbda_re_values = np.zeros(num_cells)
        lmbda_im_values = np.zeros(num_cells)
        mu_re_values = np.zeros(num_cells)
        mu_im_values = np.zeros(num_cells)
        
        for i in range(num_cells):
            domain_tag = cell_domains[i]
            try:
                domain = Domain(domain_tag)
                if domain.is_solid:
                    mat = self.materials.get_material_for_domain(domain)
                    rho_values[i] = mat.density
                    
                    # Complex Lamé parameters with loss factor
                    lmbda_complex = mat.complex_lame_lambda
                    mu_complex = mat.complex_lame_mu
                    
                    lmbda_re_values[i] = np.real(lmbda_complex)
                    lmbda_im_values[i] = np.imag(lmbda_complex)
                    mu_re_values[i] = np.real(mu_complex)
                    mu_im_values[i] = np.imag(mu_complex)
                else:
                    # Non-solid domain: use very small stiffness
                    rho_values[i] = 1.0
                    lmbda_re_values[i] = 1.0
                    mu_re_values[i] = 1.0
            except (ValueError, KeyError):
                # Default values
                rho_values[i] = 1050.0  # Polystyrene
                lmbda_re_values[i] = 2.0e9
                mu_re_values[i] = 1.0e9
        
        self.rho.x.array[:] = rho_values
        self.lmbda_re.x.array[:] = lmbda_re_values
        self.lmbda_im.x.array[:] = lmbda_im_values
        self.mu_re.x.array[:] = mu_re_values
        self.mu_im.x.array[:] = mu_im_values
        
    def _setup_measures(self):
        """Set up integration measures."""
        self.dx = Measure("dx", domain=self.mesh, subdomain_data=self.cell_tags)
        self.ds = Measure("ds", domain=self.mesh, subdomain_data=self.facet_tags)
        
    def solve(self, 
              traction_bcs: Optional[Dict[int, np.ndarray]] = None,
              displacement_bcs: Optional[Dict[int, np.ndarray]] = None) -> DisplacementField:
        """
        Solve the frequency-domain elasticity problem.
        
        Parameters
        ----------
        traction_bcs : dict, optional
            Dictionary mapping facet tags to traction vectors
        displacement_bcs : dict, optional
            Dictionary mapping facet tags to displacement vectors
            
        Returns
        -------
        DisplacementField
            Solution containing displacement field
        """
        # Trial and test functions
        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        
        omega = self.omega
        
        # For complex viscoelastic materials, we need to handle real/imag parts
        # The full equation is:
        #   ∫ σ(u):ε(v) dV - ω² ∫ ρ u·v dV = ∫ t·v dS
        #
        # With complex Lamé parameters: λ = λ_r + i λ_i, μ = μ_r + i μ_i
        # And complex displacement: u = u_r + i u_i
        #
        # For now, we solve with real Lamé parameters (ignoring damping in first pass)
        # TODO: Full complex formulation with block system
        
        # Bilinear form (using real parts only for simplicity)
        # Full version would need complex arithmetic or block system
        a = (
            inner(sigma(u, self.lmbda_re, self.mu_re), epsilon(v)) * dx
            - omega**2 * self.rho * inner(u, v) * dx
        )
        
        # Add damping contribution (imaginary Lamé parameters)
        # This creates damping: σ_damping : ε(v)
        a = a + 1j * inner(sigma(u, self.lmbda_im, self.mu_im), epsilon(v)) * dx
        
        # Right-hand side
        L = inner(Constant(self.mesh, PETSc.ScalarType((0.0,) * self.gdim)), v) * dx
        
        # Traction boundary conditions
        if traction_bcs is not None:
            for tag, traction in traction_bcs.items():
                t = Constant(self.mesh, PETSc.ScalarType(traction))
                L = L + inner(t, v) * self.ds(tag)
        
        # Dirichlet boundary conditions
        bcs = []
        if displacement_bcs is not None:
            for tag, disp in displacement_bcs.items():
                facets = self.facet_tags.find(tag)
                dofs = locate_dofs_topological(
                    self.V, self.mesh.topology.dim - 1, facets
                )
                u_bc = Constant(self.mesh, PETSc.ScalarType(disp))
                bcs.append(dirichletbc(u_bc, dofs, self.V))
        
        # Solve
        u_solution = solve_linear_system(
            a, L, bcs=bcs,
            V=self.V,
            petsc_options={
                "ksp_type": self.config.solver.ksp_type,
                "pc_type": self.config.solver.pc_type,
                "pc_factor_mat_solver_type": self.config.solver.pc_factor_solver_type,
            }
        )
        
        result = DisplacementField(
            u_function=u_solution,
            omega=omega,
            rho_function=self.rho,
            lambda_function=self.lmbda_re,
            mu_function=self.mu_re,
        )
        
        return result
    
    def compute_stress(self, u: Function) -> Function:
        """
        Compute stress tensor from displacement.
        
        Parameters
        ----------
        u : Function
            Displacement field
            
        Returns
        -------
        Function
            Stress tensor (as tensor-valued function)
        """
        # Create tensor function space
        gdim = self.gdim
        T = fem.functionspace(self.mesh, ("DG", 0, (gdim, gdim)))
        
        sig = Function(T)
        
        # Project stress
        sig_expr = sigma(u, self.lmbda_re, self.mu_re)
        # TODO: proper projection or interpolation
        
        return sig


def solve_elastic_eigenvalue(mesh: dmesh.Mesh,
                             rho: float,
                             E: float,
                             nu: float,
                             num_modes: int = 10) -> Tuple[np.ndarray, list]:
    """
    Solve elastic eigenvalue problem for natural frequencies.
    
    (-∇·σ(u), v) = ω² (ρ u, v)
    
    Parameters
    ----------
    mesh : Mesh
        The mesh
    rho : float
        Density
    E : float
        Young's modulus
    nu : float
        Poisson's ratio
    num_modes : int
        Number of modes to compute
        
    Returns
    -------
    eigenvalues : np.ndarray
        Eigenvalues (ω²)
    eigenmodes : list of Function
        Mode shapes
    """
    from slepc4py import SLEPc
    
    gdim = mesh.geometry.dim
    V = fem.functionspace(mesh, ("Lagrange", 2, (gdim,)))
    
    u = TrialFunction(V)
    v = TestFunction(V)
    
    # Lamé parameters
    lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    
    # Stiffness matrix
    a = inner(sigma(u, Constant(mesh, lmbda), Constant(mesh, mu)), epsilon(v)) * dx
    
    # Mass matrix
    m = rho * inner(u, v) * dx
    
    # Assemble
    A = fem.petsc.assemble_matrix(form(a))
    A.assemble()
    M = fem.petsc.assemble_matrix(form(m))
    M.assemble()
    
    # Setup eigenvalue solver
    eigensolver = SLEPc.EPS().create(mesh.comm)
    eigensolver.setOperators(A, M)
    eigensolver.setProblemType(SLEPc.EPS.ProblemType.GHEP)
    eigensolver.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL)
    eigensolver.setDimensions(num_modes)
    
    eigensolver.solve()
    
    # Extract results
    nconv = eigensolver.getConverged()
    eigenvalues = np.zeros(min(nconv, num_modes))
    eigenmodes = []
    
    for i in range(min(nconv, num_modes)):
        eigenvalues[i] = eigensolver.getEigenvalue(i).real
        
        # Get eigenvector
        vr = A.createVecRight()
        eigensolver.getEigenvector(i, vr)
        
        u_mode = Function(V)
        u_mode.x.array[:] = vr.array
        eigenmodes.append(u_mode)
    
    return eigenvalues, eigenmodes
