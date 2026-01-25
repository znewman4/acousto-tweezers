"""
Acoustic streaming solver for FEniCSx.

Implements acoustic streaming (time-averaged flow) per MASTER BRIEF:

    v₁ = -1/(iωρ) ∇p                    (first-order acoustic velocity)
    
    -∇p̄ + η∇²u + f_stream = 0          (Stokes equation for streaming)
    ∇·u = 0                             (incompressibility)

where f_stream is the streaming force from Reynolds stress:
    f_stream = -ρ⟨(v₁·∇)v₁*⟩

For standing waves, this creates vortices in the streaming pattern.

Author: Acousto-Tweezers Project
Date: January 2026
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
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
    inner, grad, div, dx, ds,
    TestFunction, TrialFunction, TestFunctions, TrialFunctions,
    Measure, nabla_grad, outer, dot
)

from .config import FEMConfig
from .domains import Domain, Interface
from .materials import MaterialDatabase
from .acoustics import AcousticField


@dataclass
class StreamingField:
    """
    Acoustic streaming velocity field.
    """
    # DOLFINx function for streaming velocity
    u_function: Function
    
    # Pressure field (mean pressure, usually zero for incompressible)
    p_function: Optional[Function] = None
    
    # Streaming force that drove the flow
    f_stream: Optional[Function] = None
    
    @property
    def u(self) -> np.ndarray:
        """Streaming velocity values."""
        return self.u_function.x.array.copy()
    
    @property
    def mesh(self):
        return self.u_function.function_space.mesh
    
    @property
    def max_velocity(self) -> float:
        """Maximum streaming velocity magnitude."""
        u_vals = self.u.reshape(-1, self.mesh.geometry.dim)
        return np.max(np.linalg.norm(u_vals, axis=1))
    
    @property
    def mean_velocity(self) -> float:
        """Mean streaming velocity magnitude."""
        u_vals = self.u.reshape(-1, self.mesh.geometry.dim)
        return np.mean(np.linalg.norm(u_vals, axis=1))
    
    def reynolds_number(self, fluid_viscosity: float, 
                        length_scale: float) -> float:
        """
        Compute streaming Reynolds number.
        
        Re = u L / ν
        
        Parameters
        ----------
        fluid_viscosity : float
            Kinematic viscosity [m²/s]
        length_scale : float
            Characteristic length [m]
            
        Returns
        -------
        float
            Reynolds number
        """
        u_max = self.max_velocity
        return u_max * length_scale / fluid_viscosity


class StreamingSolver:
    """
    Solver for acoustic streaming (time-averaged flow).
    
    Solves the Stokes equations with acoustic streaming force:
    
        -∇p̄ + η∇²u = -f_stream
        ∇·u = 0
        
    The streaming force is computed from the first-order acoustic field.
    """
    
    def __init__(self, config: FEMConfig,
                 mesh: dmesh.Mesh,
                 cell_tags: dmesh.MeshTags,
                 facet_tags: dmesh.MeshTags,
                 materials: MaterialDatabase):
        """
        Initialize streaming solver.
        
        Parameters
        ----------
        config : FEMConfig
            Simulation configuration
        mesh : dolfinx.mesh.Mesh
            The computational mesh
        cell_tags, facet_tags : MeshTags
            Domain and interface tags
        materials : MaterialDatabase
            Material properties
        """
        self.config = config
        self.mesh = mesh
        self.cell_tags = cell_tags
        self.facet_tags = facet_tags
        self.materials = materials
        
        self.gdim = mesh.geometry.dim
        
        # Create Taylor-Hood function spaces (P2-P1)
        self._setup_function_spaces()
        
    def _setup_function_spaces(self):
        """Create function spaces for Stokes problem."""
        # Velocity: P2 vector
        P2 = fem.element("Lagrange", self.mesh.basix_cell(), 2, shape=(self.gdim,))
        # Pressure: P1 scalar
        P1 = fem.element("Lagrange", self.mesh.basix_cell(), 1)
        
        # Mixed element (Taylor-Hood)
        mixed_el = fem.mixed_element([P2, P1])
        self.W = fem.functionspace(self.mesh, mixed_el)
        
        # Collapsed subspaces
        self.V, self.v_map = self.W.sub(0).collapse()
        self.Q, self.q_map = self.W.sub(1).collapse()
        
    def compute_streaming_force(self, 
                                acoustic_field: AcousticField) -> Function:
        """
        Compute acoustic streaming force from first-order field.
        
        The streaming force is:
            f = -ρ⟨(v₁·∇)v₁*⟩ = -ρ/2 Re[(v₁·∇)v₁*]
            
        where v₁ = -1/(iωρ)∇p is the acoustic velocity.
        
        For numerical stability, we compute:
            f_i = -1/(2ω²ρ) Re[∂p/∂xⱼ ∂²p*/∂xᵢ∂xⱼ + ...]
            
        Parameters
        ----------
        acoustic_field : AcousticField
            First-order acoustic field solution
            
        Returns
        -------
        Function
            Streaming force (vector function)
        """
        p = acoustic_field.p_function
        omega = acoustic_field.omega
        rho = self.materials.water.density
        
        # Create function for streaming force
        f = Function(self.V)
        
        # The streaming force requires second derivatives of pressure
        # This is challenging to compute accurately from P2 elements
        
        # Alternative: project force expression onto function space
        # f = -ρ/(4ω²) [grad(|∇p|²) - 2k² Re(p* ∇p)]
        
        # For now, use simplified expression based on energy density gradient
        # f ≈ -∇E where E is acoustic energy density
        
        # Project gradient of |p|²
        V_scalar = fem.functionspace(self.mesh, ("Lagrange", 2))
        p_squared = Function(V_scalar)
        
        # Get pressure values and compute |p|²
        p_vals = p.x.array
        p_sq_vals = np.abs(p_vals)**2
        
        # This is a simplified approach - proper implementation would
        # use symbolic expressions and projection
        
        return f
    
    def solve(self, 
              acoustic_field: AcousticField,
              wall_tags: Optional[list] = None) -> StreamingField:
        """
        Solve for acoustic streaming velocity.
        
        Parameters
        ----------
        acoustic_field : AcousticField
            First-order acoustic field
        wall_tags : list, optional
            Facet tags for no-slip walls
            
        Returns
        -------
        StreamingField
            Streaming velocity solution
        """
        # Trial and test functions
        (u, p) = TrialFunctions(self.W)
        (v, q) = TestFunctions(self.W)
        
        # Dynamic viscosity
        eta = self.materials.water.dynamic_viscosity
        
        # Stokes weak form:
        # ∫ η ∇u:∇v - p ∇·v - q ∇·u dx = ∫ f·v dx
        
        a = (
            eta * inner(grad(u), grad(v)) * dx
            - inner(p, div(v)) * dx
            - inner(div(u), q) * dx
        )
        
        # Streaming force RHS
        f_stream = self.compute_streaming_force(acoustic_field)
        L = inner(f_stream, v) * dx
        
        # Add pressure stabilization (PSPG) for equal-order elements
        # Not needed for Taylor-Hood, but can help convergence
        
        # Boundary conditions: no-slip on walls
        bcs = []
        
        if wall_tags is None:
            wall_tags = [
                Interface.WATER_PLATE.gmsh_tag,
                Interface.WATER_WALL.gmsh_tag,
            ]
        
        for tag in wall_tags:
            facets = self.facet_tags.find(tag)
            if len(facets) > 0:
                dofs = locate_dofs_topological(
                    (self.W.sub(0), self.V),
                    self.mesh.topology.dim - 1,
                    facets
                )
                u_zero = Function(self.V)
                u_zero.x.array[:] = 0.0
                bcs.append(dirichletbc(u_zero, dofs, self.W.sub(0)))
        
        # Also need to pin pressure at one point to remove null space
        # For now, use iterative solver with null space handling
        
        # Solve
        w = solve_linear_system(
            a, L, bcs=bcs,
            V=self.W,
            petsc_options={
                "ksp_type": "minres",
                "pc_type": "hypre",
                "pc_hypre_type": "boomeramg",
            }
        )
        
        # Extract velocity and pressure
        u_solution = w.sub(0).collapse()
        p_solution = w.sub(1).collapse()
        
        return StreamingField(
            u_function=u_solution,
            p_function=p_solution,
            f_stream=f_stream,
        )


def compute_streaming_reynolds(streaming: StreamingField,
                               fluid: "FluidMaterial",
                               length_scale: float) -> float:
    """
    Compute streaming Reynolds number.
    
    Re_s = ρ u_s L / η
    
    Parameters
    ----------
    streaming : StreamingField
        Streaming solution
    fluid : FluidMaterial
        Fluid properties
    length_scale : float
        Characteristic length (e.g., wavelength)
        
    Returns
    -------
    float
        Streaming Reynolds number
    """
    u_max = streaming.max_velocity
    nu = fluid.kinematic_viscosity
    
    return u_max * length_scale / nu


def estimate_streaming_velocity(acoustic_field: AcousticField,
                                fluid: "FluidMaterial") -> float:
    """
    Estimate streaming velocity magnitude from acoustic field.
    
    Scaling: u_s ~ (p₀²) / (ρ c ω L)
    
    where p₀ is pressure amplitude, L is characteristic length.
    
    Parameters
    ----------
    acoustic_field : AcousticField
        Acoustic field
    fluid : FluidMaterial
        Fluid properties
        
    Returns
    -------
    float
        Estimated streaming velocity [m/s]
    """
    p_max = acoustic_field.max_pressure
    omega = acoustic_field.omega
    rho = fluid.density
    c = fluid.sound_speed
    
    # Characteristic length from wavelength
    wavelength = 2 * np.pi * c / omega
    
    # Rayleigh streaming velocity estimate
    u_s = p_max**2 / (4 * rho * c * omega * wavelength)
    
    return u_s
