import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

class Helmholtz3DOperator:
    """
    3D Helmholtz operator for uniform Cartesian grid.
    Solves: (∇^2 + k^2) p = 0
    Boundary conditions:
      - Bottom (z=0): Dirichlet, p(x,y,0) = p_bot(x,y)
      - Other faces: Absorbing Robin, ∂p/∂n = i k p
    
    Features:
      - Matrix caching: reuses A if grid/k haven't changed.
      - Iterative solvers: GMRES with Jacobi preconditioning.
      - dtype control: store A as complex64 or complex128.
    """
    # Class-level cache for assembled matrices
    _matrix_cache = {}
    
    def __init__(self, grid, k, dtype=np.complex128, solver_method='direct'):
        self.grid = grid
        self.k = k
        self.Nx, self.Ny, self.Nz = grid.Nx, grid.Ny, grid.Nz
        self.dx, self.dy, self.dz = grid.dx, grid.dy, grid.dz
        self.size = self.Nx * self.Ny * self.Nz
        self.dtype = dtype
        self.solver_method = solver_method  # 'direct', 'gmres', 'bicgstab'
        self.A = None  # Cached matrix
        self.cache_key = None

    def _flatten_index(self, ix, iy, iz):
        return (ix * self.Ny + iy) * self.Nz + iz

    def _make_cache_key(self):
        """Generate cache key for matrix reuse."""
        return (self.Nx, self.Ny, self.Nz, self.dx, self.dy, self.dz, float(self.k), str(self.dtype))
    
    def assemble_system(self, p_bot):
        """
        Assemble or retrieve cached system matrix A and RHS vector b.
        Uses cache to avoid repeated assembly if grid/k unchanged.
        """
        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
        dx, dy, dz = self.dx, self.dy, self.dz
        k = self.k
        size = self.size
        
        # Check cache
        cache_key = self._make_cache_key()
        if cache_key in Helmholtz3DOperator._matrix_cache:
            A = Helmholtz3DOperator._matrix_cache[cache_key]
            print(f"[MATRIX] Using cached A (shape={A.shape}, nnz={A.nnz})")
            self.A = A
        else:
            A = self._assemble_matrix()
            Helmholtz3DOperator._matrix_cache[cache_key] = A
            self.A = A
            print(f"[MATRIX] Assembled and cached A (shape={A.shape}, nnz={A.nnz})")
        
        # Build RHS
        b = np.zeros(size, dtype=self.dtype)
        for ix in range(Nx):
            for iy in range(Ny):
                iz = 0  # Bottom (Dirichlet)
                idx = self._flatten_index(ix, iy, iz)
                b[idx] = p_bot[ix, iy]
        
        return A, b
    
    def _assemble_matrix(self):
        """Assemble the system matrix A. Called once and cached."""
        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
        dx, dy, dz = self.dx, self.dy, self.dz
        k = self.k
        size = self.size

        rows = []
        cols = []
        data = []

        for ix in range(Nx):
            for iy in range(Ny):
                for iz in range(Nz):
                    idx = self._flatten_index(ix, iy, iz)
                    # Dirichlet: bottom face only
                    if iz == 0:
                        rows.append(idx)
                        cols.append(idx)
                        data.append(1.0)
                        continue
                    # Robin: top face
                    if iz == Nz - 1:
                        idx_in = self._flatten_index(ix, iy, iz - 1)
                        rows.append(idx)
                        cols.append(idx)
                        data.append(1.0 / dz)
                        rows.append(idx)
                        cols.append(idx_in)
                        data.append(-1.0 / dz)
                        continue
                    # Robin: x-min
                    if ix == 0:
                        idx_in = self._flatten_index(ix + 1, iy, iz)
                        rows.append(idx)
                        cols.append(idx)
                        data.append(1.0 / dx)
                        rows.append(idx)
                        cols.append(idx_in)
                        data.append(-1.0 / dx)
                        continue
                    # Robin: x-max
                    if ix == Nx - 1:
                        idx_in = self._flatten_index(ix - 1, iy, iz)
                        rows.append(idx)
                        cols.append(idx)
                        data.append(1.0 / dx)
                        rows.append(idx)
                        cols.append(idx_in)
                        data.append(-1.0 / dx)
                        continue
                    # Robin: y-min
                    if iy == 0:
                        idx_in = self._flatten_index(ix, iy + 1, iz)
                        rows.append(idx)
                        cols.append(idx)
                        data.append(1.0 / dy)
                        rows.append(idx)
                        cols.append(idx_in)
                        data.append(-1.0 / dy)
                        continue
                    # Robin: y-max
                    if iy == Ny - 1:
                        idx_in = self._flatten_index(ix, iy - 1, iz)
                        rows.append(idx)
                        cols.append(idx)
                        data.append(1.0 / dy)
                        rows.append(idx)
                        cols.append(idx_in)
                        data.append(-1.0 / dy)
                        continue
                    # Interior: 7-point stencil
                    idx_xm = self._flatten_index(ix - 1, iy, iz)
                    idx_xp = self._flatten_index(ix + 1, iy, iz)
                    idx_ym = self._flatten_index(ix, iy - 1, iz)
                    idx_yp = self._flatten_index(ix, iy + 1, iz)
                    idx_zm = self._flatten_index(ix, iy, iz - 1)
                    idx_zp = self._flatten_index(ix, iy, iz + 1)
                    rows.extend([idx] * 7)
                    cols.extend([idx_xm, idx_xp, idx_ym, idx_yp, idx_zm, idx_zp, idx])
                    data.extend([
                        1.0 / dx**2,
                        1.0 / dx**2,
                        1.0 / dy**2,
                        1.0 / dy**2,
                        1.0 / dz**2,
                        1.0 / dz**2,
                        -2.0 * (1.0 / dx**2 + 1.0 / dy**2 + 1.0 / dz**2) + k**2
                    ])

        # Debug assertions
        A = sp.coo_matrix((data, (rows, cols)), shape=(size, size), dtype=self.dtype).tocsr()
        if Nx > 2 and Ny > 2 and Nz > 2:
            ix, iy, iz = 1, 1, Nz-1
            idx = self._flatten_index(ix, iy, iz)
            row = A.getrow(idx)
            nnz = row.count_nonzero()
            assert nnz == 2, f"Top face node should have 2 nonzeros, got {nnz}"
        
        return A

    def solve(self, p_bot):
        """Solve the 3D Helmholtz problem."""
        A, b = self.assemble_system(p_bot)
        
        if self.solver_method == 'direct':
            p_flat = spla.spsolve(A, b)
        elif self.solver_method in ['gmres', 'bicgstab']:
            # Use iterative solver with Jacobi preconditioning
            diag = np.abs(A.diagonal())
            diag[diag == 0] = 1.0  # Avoid division by zero
            M_inv = sp.diags(1.0 / diag, format='csr')
            
            if self.solver_method == 'gmres':
                p_flat, info = spla.gmres(A, b, M=M_inv, restart=30, maxiter=1000, atol=1e-4)
            else:  # bicgstab
                p_flat, info = spla.bicgstab(A, b, M=M_inv, maxiter=1000, atol=1e-4)
            
            if info != 0:
                print(f"[SOLVER] Warning: {self.solver_method} converged with info={info}")
        else:
            raise ValueError(f"Unknown solver method: {self.solver_method}")
        
        p = p_flat.reshape((self.Nx, self.Ny, self.Nz))
        return p

