"""
FEM solid mechanics solver for frequency-domain elastodynamics.

Implements the frequency-domain elasticity equation:

    ∇·σ(u) + ρ_s ω² u = 0

with viscoelastic damping:

    E → E(1 + iη)

Fluid-solid coupling (from MASTER BRIEF):

1. Traction balance on interface:
   σ(u)·n = -p·n

2. Normal velocity continuity:
   v·n = iω u·n
   
   where v = -1/(iωρ) ∇p is the fluid velocity.

These coupling conditions are implemented through:
- Weak form surface integrals at fluid-solid interfaces
- Consistent treatment of normal tractions

References
----------
- Kaltenbacher, M. (2015): Numerical Simulation of Mechatronic Sensors
- Zienkiewicz, O.C. (2005): The Finite Element Method for Solid Mechanics
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

from .domains import DomainType, InterfaceType, Domain
from .geometry import (
    FEMMesh,
    get_shape_functions_hex8,
    get_shape_gradients_hex8,
    GAUSS_POINTS_HEX8,
    GAUSS_WEIGHTS_HEX8,
)
from .materials import SolidMaterial, FluidMaterial, MaterialDatabase
from .config import FEMConfig
from .acoustics import AcousticField


@dataclass
class DisplacementField:
    """
    Displacement field solution in solid domains.
    
    Contains displacement and derived quantities (strain, stress).
    """
    # Grid coordinates
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    
    # Displacement components (complex, nodal values)
    ux: np.ndarray  # (num_nodes,) complex
    uy: np.ndarray
    uz: np.ndarray
    
    # Frequency
    omega: float
    
    # Material properties at nodes (solid nodes only)
    rho: np.ndarray
    E: np.ndarray
    nu: np.ndarray
    
    # Mesh reference
    mesh: Optional[FEMMesh] = None
    
    @property
    def u_amplitude(self) -> np.ndarray:
        """Displacement amplitude |u|."""
        return np.sqrt(np.abs(self.ux)**2 + np.abs(self.uy)**2 + np.abs(self.uz)**2)
    
    def compute_strain(self) -> Dict[str, np.ndarray]:
        """
        Compute strain tensor components.
        
        ε_ij = (1/2)(∂u_i/∂x_j + ∂u_j/∂x_i)
        
        Returns
        -------
        strain : dict
            Strain components {exx, eyy, ezz, exy, exz, eyz}
        """
        if self.mesh is None:
            raise ValueError("Mesh required for strain computation")
        
        dx, dy, dz = self.mesh.dx, self.mesh.dy, self.mesh.dz
        nx, ny, nz = self.mesh.nx, self.mesh.ny, self.mesh.nz
        
        # Reshape to 3D
        ux = self.ux.reshape((nz, ny, nx)).transpose((2, 1, 0))
        uy = self.uy.reshape((nz, ny, nx)).transpose((2, 1, 0))
        uz = self.uz.reshape((nz, ny, nx)).transpose((2, 1, 0))
        
        # Compute gradients (central differences)
        def grad_x(f):
            g = np.zeros_like(f)
            g[1:-1, :, :] = (f[2:, :, :] - f[:-2, :, :]) / (2 * dx)
            g[0, :, :] = (f[1, :, :] - f[0, :, :]) / dx
            g[-1, :, :] = (f[-1, :, :] - f[-2, :, :]) / dx
            return g
        
        def grad_y(f):
            g = np.zeros_like(f)
            g[:, 1:-1, :] = (f[:, 2:, :] - f[:, :-2, :]) / (2 * dy)
            g[:, 0, :] = (f[:, 1, :] - f[:, 0, :]) / dy
            g[:, -1, :] = (f[:, -1, :] - f[:, -2, :]) / dy
            return g
        
        def grad_z(f):
            g = np.zeros_like(f)
            g[:, :, 1:-1] = (f[:, :, 2:] - f[:, :, :-2]) / (2 * dz)
            g[:, :, 0] = (f[:, :, 1] - f[:, :, 0]) / dz
            g[:, :, -1] = (f[:, :, -1] - f[:, :, -2]) / dz
            return g
        
        # Normal strains
        exx = grad_x(ux)
        eyy = grad_y(uy)
        ezz = grad_z(uz)
        
        # Shear strains (engineering notation)
        exy = 0.5 * (grad_y(ux) + grad_x(uy))
        exz = 0.5 * (grad_z(ux) + grad_x(uz))
        eyz = 0.5 * (grad_z(uy) + grad_y(uz))
        
        return {
            'exx': exx, 'eyy': eyy, 'ezz': ezz,
            'exy': exy, 'exz': exz, 'eyz': eyz,
        }
    
    def compute_stress(self, material: SolidMaterial) -> Dict[str, np.ndarray]:
        """
        Compute stress tensor components.
        
        σ = λ tr(ε) I + 2μ ε
        
        with complex moduli for damping.
        """
        strain = self.compute_strain()
        
        # Complex Lamé parameters
        lam = material.lambda_lame_complex
        mu = material.mu_lame_complex
        
        # Volumetric strain
        tr_e = strain['exx'] + strain['eyy'] + strain['ezz']
        
        # Stress components
        sxx = lam * tr_e + 2 * mu * strain['exx']
        syy = lam * tr_e + 2 * mu * strain['eyy']
        szz = lam * tr_e + 2 * mu * strain['ezz']
        sxy = 2 * mu * strain['exy']
        sxz = 2 * mu * strain['exz']
        syz = 2 * mu * strain['eyz']
        
        return {
            'sxx': sxx, 'syy': syy, 'szz': szz,
            'sxy': sxy, 'sxz': sxz, 'syz': syz,
        }


class FEMSolidSolver:
    """
    Finite Element Method solver for frequency-domain elastodynamics.
    
    Solves:
        ∇·σ(u) + ρω²u = 0
    
    with σ = C:ε and complex-valued C for viscoelastic damping.
    
    Coupling to acoustic field via interface conditions.
    """
    
    def __init__(
        self,
        mesh: FEMMesh,
        materials: MaterialDatabase,
        config: FEMConfig,
    ):
        """
        Initialize solid mechanics solver.
        
        Parameters
        ----------
        mesh : FEMMesh
            Finite element mesh with domain assignments.
        materials : MaterialDatabase
            Material properties.
        config : FEMConfig
            Simulation configuration.
        """
        self.mesh = mesh
        self.materials = materials
        self.config = config
        
        self.omega = config.physics.omega
        
        # Get solid material (assume glass for dish)
        self.solid_material = materials.borosilicate_glass
        
        # Build material property arrays for solid nodes
        self._build_material_fields()
        
        # System matrices
        self._K: Optional[sparse.csr_matrix] = None  # Stiffness
        self._M: Optional[sparse.csr_matrix] = None  # Mass
        self._assembled = False
    
    def _build_material_fields(self):
        """Build nodal arrays of material properties for solid domains."""
        n_nodes = self.mesh.num_nodes
        
        self.rho = np.zeros(n_nodes, dtype=np.float64)
        self.E = np.zeros(n_nodes, dtype=np.complex128)
        self.nu = np.zeros(n_nodes, dtype=np.float64)
        
        mat = self.solid_material
        
        # Assign to solid domain nodes
        solid_domains = [DomainType.PLATE, DomainType.WALL]
        for domain in solid_domains:
            if domain in self.mesh.domain_info:
                for node_id in self.mesh.domain_info[domain].node_ids:
                    self.rho[node_id] = mat.rho
                    self.E[node_id] = mat.E_complex
                    self.nu[node_id] = mat.nu
    
    def assemble_system(self) -> None:
        """
        Assemble global stiffness and mass matrices for solid domains.
        
        Weak form:
            ∫_Ω ε(φ):C:ε(u) dV = stiffness → K
            ∫_Ω ρ φ·u dV = mass → M
        
        System: (K - ω²M) u = f
        """
        if self._assembled:
            return
        
        n_nodes = self.mesh.num_nodes
        n_dof = 3 * n_nodes  # 3 DOF per node (ux, uy, uz)
        
        # Triplet format
        rows = []
        cols = []
        K_vals = []
        M_vals = []
        
        # Jacobian (constant for structured mesh)
        J_inv = np.diag([2 / self.mesh.dx, 2 / self.mesh.dy, 2 / self.mesh.dz])
        det_J = (self.mesh.dx * self.mesh.dy * self.mesh.dz) / 8
        
        # Material matrix (isotropic, complex for damping)
        mat = self.solid_material
        lam = mat.lambda_lame_complex
        mu = mat.mu_lame_complex
        
        # Constitutive matrix D for isotropic material (Voigt notation)
        # σ = D ε, where ε = [εxx, εyy, εzz, 2εxy, 2εxz, 2εyz]
        D = np.array([
            [lam + 2*mu, lam, lam, 0, 0, 0],
            [lam, lam + 2*mu, lam, 0, 0, 0],
            [lam, lam, lam + 2*mu, 0, 0, 0],
            [0, 0, 0, mu, 0, 0],
            [0, 0, 0, 0, mu, 0],
            [0, 0, 0, 0, 0, mu],
        ], dtype=np.complex128)
        
        # Loop over solid elements
        for e in range(self.mesh.num_elements):
            domain = DomainType(self.mesh.element_domains[e])
            if not domain.is_solid:
                continue
            
            node_ids = self.mesh.elements[e]
            rho_e = mat.rho
            
            # Local matrices (24×24 for 8 nodes × 3 DOF)
            Ke = np.zeros((24, 24), dtype=np.complex128)
            Me = np.zeros((24, 24), dtype=np.complex128)
            
            # Gauss quadrature
            for g, (xi, w) in enumerate(zip(GAUSS_POINTS_HEX8, GAUSS_WEIGHTS_HEX8)):
                N = get_shape_functions_hex8(xi)
                dN_local = get_shape_gradients_hex8(xi)
                dN = dN_local @ J_inv  # (8, 3)
                
                # Build B matrix (6×24): ε = B u
                B = self._build_B_matrix(dN)
                
                # Build N matrix (3×24): u = N ũ
                N_mat = self._build_N_matrix(N)
                
                # Stiffness: ∫ Bᵀ D B dV
                Ke += (B.T @ D @ B) * det_J * w
                
                # Mass: ∫ ρ Nᵀ N dV
                Me += rho_e * (N_mat.T @ N_mat) * det_J * w
            
            # Map local to global DOFs
            for i in range(8):
                for j in range(8):
                    for di in range(3):
                        for dj in range(3):
                            gi = 3 * node_ids[i] + di
                            gj = 3 * node_ids[j] + dj
                            li = 3 * i + di
                            lj = 3 * j + dj
                            
                            rows.append(gi)
                            cols.append(gj)
                            K_vals.append(Ke[li, lj])
                            M_vals.append(Me[li, lj])
        
        self._K = sparse.csr_matrix(
            (K_vals, (rows, cols)),
            shape=(n_dof, n_dof),
            dtype=np.complex128,
        )
        
        self._M = sparse.csr_matrix(
            (M_vals, (rows, cols)),
            shape=(n_dof, n_dof),
            dtype=np.complex128,
        )
        
        self._assembled = True
    
    def _build_B_matrix(self, dN: np.ndarray) -> np.ndarray:
        """
        Build strain-displacement matrix B.
        
        ε = [εxx, εyy, εzz, 2εxy, 2εxz, 2εyz]ᵀ = B u
        
        Parameters
        ----------
        dN : np.ndarray
            Shape function gradients (8, 3).
        
        Returns
        -------
        B : np.ndarray
            (6, 24) strain-displacement matrix.
        """
        B = np.zeros((6, 24), dtype=np.float64)
        
        for i in range(8):
            col = 3 * i
            # εxx = ∂ux/∂x
            B[0, col] = dN[i, 0]
            # εyy = ∂uy/∂y
            B[1, col + 1] = dN[i, 1]
            # εzz = ∂uz/∂z
            B[2, col + 2] = dN[i, 2]
            # 2εxy = ∂ux/∂y + ∂uy/∂x
            B[3, col] = dN[i, 1]
            B[3, col + 1] = dN[i, 0]
            # 2εxz = ∂ux/∂z + ∂uz/∂x
            B[4, col] = dN[i, 2]
            B[4, col + 2] = dN[i, 0]
            # 2εyz = ∂uy/∂z + ∂uz/∂y
            B[5, col + 1] = dN[i, 2]
            B[5, col + 2] = dN[i, 1]
        
        return B
    
    def _build_N_matrix(self, N: np.ndarray) -> np.ndarray:
        """
        Build shape function matrix for displacement.
        
        u = [ux, uy, uz]ᵀ = N_mat ũ
        
        Parameters
        ----------
        N : np.ndarray
            Shape function values (8,).
        
        Returns
        -------
        N_mat : np.ndarray
            (3, 24) shape function matrix.
        """
        N_mat = np.zeros((3, 24), dtype=np.float64)
        
        for i in range(8):
            col = 3 * i
            N_mat[0, col] = N[i]
            N_mat[1, col + 1] = N[i]
            N_mat[2, col + 2] = N[i]
        
        return N_mat
    
    def solve_coupled(
        self,
        acoustic_field: AcousticField,
    ) -> DisplacementField:
        """
        Solve solid mechanics with coupling to acoustic field.
        
        Coupling condition:
            σ·n = -p·n on fluid-solid interfaces
        
        Parameters
        ----------
        acoustic_field : AcousticField
            Solved acoustic pressure field.
        
        Returns
        -------
        displacement : DisplacementField
            Displacement field in solid domains.
        """
        self.assemble_system()
        
        n_nodes = self.mesh.num_nodes
        n_dof = 3 * n_nodes
        
        # System matrix
        A = self._K - self.omega**2 * self._M
        
        # Right-hand side from acoustic pressure
        f = self._compute_interface_traction(acoustic_field)
        
        # Apply boundary conditions
        A, f = self._apply_boundary_conditions(A.tolil(), f)
        
        # Solve
        if self.config.solver.linear_solver == "direct":
            u = spla.spsolve(A.tocsr(), f)
        else:
            u, info = spla.gmres(
                A.tocsr(), f,
                rtol=self.config.solver.iterative_tol,
                maxiter=self.config.solver.iterative_maxiter,
            )
        
        # Extract components
        ux = u[0::3]
        uy = u[1::3]
        uz = u[2::3]
        
        return DisplacementField(
            x=self.mesh.x,
            y=self.mesh.y,
            z=self.mesh.z,
            ux=ux,
            uy=uy,
            uz=uz,
            omega=self.omega,
            rho=self.rho,
            E=self.E,
            nu=self.nu,
            mesh=self.mesh,
        )
    
    def _compute_interface_traction(
        self,
        acoustic_field: AcousticField,
    ) -> np.ndarray:
        """
        Compute traction force from acoustic pressure at interfaces.
        
        f = -∫_Γ p n φ dA
        """
        n_dof = 3 * self.mesh.num_nodes
        f = np.zeros(n_dof, dtype=np.complex128)
        
        # Get interface facets
        fluid_solid_interfaces = [
            InterfaceType.WATER_PLATE,
            InterfaceType.WATER_WALL,
            InterfaceType.BATH_PLATE,
            InterfaceType.BATH_WALL,
        ]
        
        for itype in fluid_solid_interfaces:
            if itype not in self.mesh.interface_info:
                continue
            
            interface = self.mesh.interface_info[itype]
            normal_dir = interface.normal_direction
            
            # Get normal vector
            if normal_dir == 'z':
                n = np.array([0, 0, 1])
            elif normal_dir == 'x':
                n = np.array([1, 0, 0])
            elif normal_dir == 'y':
                n = np.array([0, 1, 0])
            else:
                n = np.array([0, 0, 1])
            
            # Loop over interface facets
            for facet in interface.facet_ids:
                elem_id = facet[0]  # Element on one side
                node_ids = self.mesh.elements[elem_id]
                
                # Get pressure at nodes (interpolate from acoustic field)
                p_nodes = acoustic_field.p[node_ids]
                p_avg = np.mean(p_nodes)
                
                # Face area (approximate)
                if normal_dir == 'z':
                    face_area = self.mesh.dx * self.mesh.dy
                elif normal_dir == 'x':
                    face_area = self.mesh.dy * self.mesh.dz
                else:
                    face_area = self.mesh.dx * self.mesh.dz
                
                # Distribute traction to nodes
                # f_i = -p * n * area / 8 (uniform distribution to 8 nodes)
                for i, node_id in enumerate(node_ids):
                    for d in range(3):
                        dof = 3 * node_id + d
                        f[dof] -= p_avg * n[d] * face_area / 8
        
        return f
    
    def _apply_boundary_conditions(
        self,
        A: sparse.lil_matrix,
        f: np.ndarray,
    ) -> Tuple[sparse.lil_matrix, np.ndarray]:
        """
        Apply displacement boundary conditions.
        
        For now: fix nodes on outer boundaries.
        """
        # Fix displacement on exterior boundaries
        x_min, x_max = self.mesh.x[0], self.mesh.x[-1]
        y_min, y_max = self.mesh.y[0], self.mesh.y[-1]
        z_min = self.mesh.z[0]
        
        tol = self.mesh.dx / 10
        
        for node_id in range(self.mesh.num_nodes):
            x, y, z = self.mesh.nodes[node_id]
            
            # Fix bottom and lateral boundaries
            is_boundary = (
                abs(x - x_min) < tol or abs(x - x_max) < tol or
                abs(y - y_min) < tol or abs(y - y_max) < tol or
                abs(z - z_min) < tol
            )
            
            if is_boundary:
                for d in range(3):
                    dof = 3 * node_id + d
                    A[dof, :] = 0
                    A[dof, dof] = 1.0
                    f[dof] = 0.0
        
        return A, f
