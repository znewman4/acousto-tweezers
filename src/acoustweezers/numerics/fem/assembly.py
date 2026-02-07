"""
Solver utilities for FEniCSx.

This module provides helper functions for assembling and solving linear systems
using PETSc directly, avoiding DOLFINx LinearProblem API changes.
"""

from typing import Optional, List, Dict, Any
import numpy as np

from dolfinx import fem
from dolfinx.fem import petsc as fem_petsc
from petsc4py import PETSc
import ufl


def solve_linear_system(
    a: ufl.Form,
    L: ufl.Form,
    bcs: Optional[List[fem.DirichletBC]] = None,
    u: Optional[fem.Function] = None,
    V: Optional[fem.FunctionSpace] = None,
    petsc_options: Optional[Dict[str, Any]] = None,
) -> fem.Function:
    """
    Solve a linear variational problem using PETSc.
    
    This replaces fem.petsc.LinearProblem to avoid API compatibility issues
    across DOLFINx versions.
    
    Parameters
    ----------
    a : ufl.Form
        Bilinear form (LHS)
    L : ufl.Form
        Linear form (RHS)
    bcs : list of DirichletBC, optional
        Boundary conditions
    u : Function, optional
        Solution function (created if not provided)
    V : FunctionSpace, optional
        Function space (required if u is not provided)
    petsc_options : dict, optional
        PETSc solver options. Defaults to direct LU solver with MUMPS.
        
    Returns
    -------
    Function
        Solution function
    """
    if bcs is None:
        bcs = []
    
    if petsc_options is None:
        petsc_options = {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        }
    
    # Compile forms
    a_compiled = fem.form(a)
    L_compiled = fem.form(L)
    
    # Create solution function if not provided
    if u is None:
        if V is None:
            raise ValueError("Either u or V must be provided to solve_linear_system")
        u = fem.Function(V)
    else:
        # Get V from provided function
        V = u.function_space
    
    # Get mesh communicator
    mesh = V.mesh
    comm = mesh.comm
    
    # Assemble matrix
    A = fem_petsc.assemble_matrix(a_compiled, bcs=bcs)
    A.assemble()
    
    # Assemble RHS vector
    b = fem_petsc.assemble_vector(L_compiled)
    
    # Apply lifting for non-homogeneous BCs
    fem_petsc.apply_lifting(b, [a_compiled], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    fem_petsc.set_bc(b, bcs)
    
    # Create and configure solver
    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    
    # Apply PETSc options
    solver.setType(petsc_options.get("ksp_type", "preonly"))
    pc = solver.getPC()
    pc.setType(petsc_options.get("pc_type", "lu"))
    
    # Set direct solver type if using LU
    if pc.getType() == "lu":
        solver_type = petsc_options.get("pc_factor_mat_solver_type", "mumps")
        pc.setFactorSolverType(solver_type)
    
    # Handle Hypre options
    if pc.getType() == "hypre":
        hypre_type = petsc_options.get("pc_hypre_type", "boomeramg")
        pc.setHYPREType(hypre_type)
    
    solver.setFromOptions()
    
    # Solve
    solver.solve(b, u.x.petsc_vec)
    u.x.scatter_forward()
    
    # Report convergence
    converged_reason = solver.getConvergedReason()
    if converged_reason < 0:
        import warnings
        warnings.warn(f"PETSc solver did not converge (reason: {converged_reason})")
    
    # Clean up
    solver.destroy()
    A.destroy()
    b.destroy()
    
    return u


def solve_mixed_system(
    a: ufl.Form,
    L: ufl.Form,
    W: fem.FunctionSpace,
    bcs: Optional[List[fem.DirichletBC]] = None,
    petsc_options: Optional[Dict[str, Any]] = None,
) -> fem.Function:
    """
    Solve a mixed linear variational problem using PETSc.
    
    For Stokes-like problems with velocity-pressure or similar mixed spaces.
    
    Parameters
    ----------
    a : ufl.Form
        Bilinear form (LHS)
    L : ufl.Form
        Linear form (RHS)
    W : FunctionSpace
        Mixed function space
    bcs : list of DirichletBC, optional
        Boundary conditions
    petsc_options : dict, optional
        PETSc solver options. Defaults to minres with hypre AMG.
        
    Returns
    -------
    Function
        Solution function in mixed space
    """
    if bcs is None:
        bcs = []
    
    if petsc_options is None:
        # Good default for saddle-point problems
        petsc_options = {
            "ksp_type": "minres",
            "pc_type": "hypre",
            "pc_hypre_type": "boomeramg",
        }
    
    w = fem.Function(W)
    return solve_linear_system(a, L, bcs=bcs, u=w, petsc_options=petsc_options)


def create_null_space(V: fem.FunctionSpace, 
                      null_components: List[int]) -> PETSc.NullSpace:
    """
    Create a PETSc null space for a function space.
    
    Parameters
    ----------
    V : FunctionSpace
        Function space
    null_components : list of int
        Components that span the null space (e.g., [0] for constant pressure)
        
    Returns
    -------
    PETSc.NullSpace
        The null space object
    """
    null_vectors = []
    for comp in null_components:
        null_vec = fem.Function(V)
        null_vec.x.array[:] = 0
        # Set component to constant
        # This is a simplified version - for mixed spaces, need more care
        null_vec.x.array[:] = 1.0 / np.sqrt(len(null_vec.x.array))
        null_vectors.append(null_vec.x.petsc_vec)
    
    nullspace = PETSc.NullSpace().create(constant=False, vectors=null_vectors)
    return nullspace
