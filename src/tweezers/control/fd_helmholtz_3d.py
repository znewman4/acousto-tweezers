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
    """
    def __init__(self, grid, k):
        self.grid = grid
        self.k = k
        self.Nx, self.Ny, self.Nz = grid.Nx, grid.Ny, grid.Nz
        self.dx, self.dy, self.dz = grid.dx, grid.dy, grid.dz
        self.size = self.Nx * self.Ny * self.Nz

    def _flatten_index(self, ix, iy, iz):
        return (ix * self.Ny + iy) * self.Nz + iz

    def assemble_system(self, p_bot):
        print(f"[DEBUG] k type: {type(self.k)}, value: {self.k}")
        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
        dx, dy, dz = self.dx, self.dy, self.dz
        k = self.k
        size = self.size

        rows = []
        cols = []
        data = []
        b = np.zeros(size, dtype=np.complex128)

        for ix in range(Nx):
            for iy in range(Ny):
                for iz in range(Nz):
                    idx = self._flatten_index(ix, iy, iz)
                    # Dirichlet: bottom face only
                    if iz == 0:
                        rows.append(idx)
                        cols.append(idx)
                        data.append(1.0)
                        b[idx] = p_bot[ix, iy]
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
                        b[idx] = 0.0
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
                        b[idx] = 0.0
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
                        b[idx] = 0.0
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
                        b[idx] = 0.0
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
                        b[idx] = 0.0
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
                    b[idx] = 0.0

        # Debug assertions: check Robin row structure for a representative node on each face
        A = sp.coo_matrix((data, (rows, cols)), shape=(size, size)).tocsr()
        # Top face (not at edge/corner)
        if Nx > 2 and Ny > 2 and Nz > 2:
            ix, iy, iz = 1, 1, Nz-1
            idx = self._flatten_index(ix, iy, iz)
            row = A.getrow(idx)
            nnz = row.count_nonzero()
            assert nnz == 2, f"Top face node should have 2 nonzeros, got {nnz}"
        # x-min
            ix, iy, iz = 0, 1, 1
            idx = self._flatten_index(ix, iy, iz)
            row = A.getrow(idx)
            nnz = row.count_nonzero()
            assert nnz == 2, f"x-min face node should have 2 nonzeros, got {nnz}"
        # x-max
            ix, iy, iz = Nx-1, 1, 1
            idx = self._flatten_index(ix, iy, iz)
            row = A.getrow(idx)
            nnz = row.count_nonzero()
            assert nnz == 2, f"x-max face node should have 2 nonzeros, got {nnz}"
        # y-min
            ix, iy, iz = 1, 0, 1
            idx = self._flatten_index(ix, iy, iz)
            row = A.getrow(idx)
            nnz = row.count_nonzero()
            assert nnz == 2, f"y-min face node should have 2 nonzeros, got {nnz}"
        # y-max
            ix, iy, iz = 1, Ny-1, 1
            idx = self._flatten_index(ix, iy, iz)
            row = A.getrow(idx)
            nnz = row.count_nonzero()
            assert nnz == 2, f"y-max face node should have 2 nonzeros, got {nnz}"

        return A, b

    def solve(self, p_bot):
        A, b = self.assemble_system(p_bot)
        p_flat = spla.spsolve(A, b)
        p = p_flat.reshape((self.Nx, self.Ny, self.Nz))
        return p
