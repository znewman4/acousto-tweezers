import numpy as np
from tweezers.grid.grid3d import Grid3D
from tweezers.control.fd_helmholtz_3d import Helmholtz3DOperator


import numpy as np

class ExtendedField3D:
    """
    Stores 3D pressure field and metadata for Helmholtz3DSolver.
    """
    def __init__(self, p, grid, k, omega, medium_props):
        self.p = p  # complex pressure field, shape (Nx, Ny, Nz)
        self.grid = grid  # Grid3D instance
        self.k = k
        self.omega = omega
        self.medium_props = medium_props

    @property
    def shape(self):
        return self.p.shape

    @property
    def Nx(self):
        return self.grid.Nx
    @property
    def Ny(self):
        return self.grid.Ny
    @property
    def Nz(self):
        return self.grid.Nz

    def diagnostics(self, p_bot=None, eps=1e-12, operator=None, debug_Ab=False, face_mode='strict', debug_print=True):
        """
        Compute PDE and BC residuals for validation. Returns dict of floats.
        Args:
            p_bot: (Nx, Ny) array, required for Dirichlet BC check
            operator: Helmholtz3DOperator instance (optional, for debug)
            debug_Ab: if True, compute and return A, b, and system residuals
            face_mode: 'strict' (exclude edges/corners) or 'all' (all face nodes)
            debug_print: print detailed row info for one node per face
        """
        p = self.p
        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
        dx, dy, dz = self.grid.dx, self.grid.dy, self.grid.dz
        k = self.k
        metrics = {}

        # A1: Interior PDE residual (exclude all boundaries)
        r = np.zeros((Nx, Ny, Nz), dtype=np.complex128)
        for ix in range(1, Nx-1):
            for iy in range(1, Ny-1):
                for iz in range(1, Nz-1):
                    lap = (
                        (p[ix+1,iy,iz] - 2*p[ix,iy,iz] + p[ix-1,iy,iz]) / dx**2 +
                        (p[ix,iy+1,iz] - 2*p[ix,iy,iz] + p[ix,iy-1,iz]) / dy**2 +
                        (p[ix,iy,iz+1] - 2*p[ix,iy,iz] + p[ix,iy,iz-1]) / dz**2
                    )
                    r[ix,iy,iz] = lap + k**2 * p[ix,iy,iz]
        mask = np.zeros((Nx, Ny, Nz), dtype=bool)
        mask[1:-1,1:-1,1:-1] = True
        kp_int = (k**2 * p)[mask]
        r_int = r[mask]
        metrics['res_L2_rel'] = np.linalg.norm(r_int) / (np.linalg.norm(kp_int) + eps)
        metrics['res_Linf_rel'] = np.max(np.abs(r_int)) / (np.max(np.abs(kp_int)) + eps)

        # A2: Dirichlet enforcement error (bottom face)
        if p_bot is not None:
            eD = p[:,:,0] - p_bot
            metrics['dirichlet_L2_rel'] = np.linalg.norm(eD) / (np.linalg.norm(p_bot) + eps)
            metrics['dirichlet_Linf_rel'] = np.max(np.abs(eD)) / (np.max(np.abs(p_bot)) + eps)

        # --- Matrix-based face residuals ---
        if debug_Ab and operator is not None:
            A, b = operator.assemble_system(p_bot)
            u = p.ravel()
            r = A @ u - b
            face_defs = {
                'top':   lambda ix,iy,iz: iz == Nz-1,
                'xmin':  lambda ix,iy,iz: ix == 0,
                'xmax':  lambda ix,iy,iz: ix == Nx-1,
                'ymin':  lambda ix,iy,iz: iy == 0,
                'ymax':  lambda ix,iy,iz: iy == Ny-1,
            }
            # For each face, get all and strict indices
            for face, cond in face_defs.items():
                all_idx = []
                strict_idx = []
                for ix in range(Nx):
                    for iy in range(Ny):
                        for iz in range(Nz):
                            if cond(ix,iy,iz):
                                idx = operator._flatten_index(ix,iy,iz)
                                all_idx.append(idx)
                                # strict: not on any other boundary
                                if face == 'top' and (ix>0 and ix<Nx-1 and iy>0 and iy<Ny-1):
                                    strict_idx.append(idx)
                                elif face == 'xmin' and (iy>0 and iy<Ny-1 and iz>0 and iz<Nz-1):
                                    strict_idx.append(idx)
                                elif face == 'xmax' and (iy>0 and iy<Ny-1 and iz>0 and iz<Nz-1):
                                    strict_idx.append(idx)
                                elif face == 'ymin' and (ix>0 and ix<Nx-1 and iz>0 and iz<Nz-1):
                                    strict_idx.append(idx)
                                elif face == 'ymax' and (ix>0 and ix<Nx-1 and iz>0 and iz<Nz-1):
                                    strict_idx.append(idx)
                all_idx = np.array(all_idx)
                strict_idx = np.array(strict_idx)
                if len(all_idx) > 0:
                    metrics[f'face_{face}_lsys_rms_all'] = np.sqrt(np.mean(np.abs(r[all_idx])**2))
                    metrics[f'face_{face}_lsys_max_all'] = np.max(np.abs(r[all_idx]))
                if len(strict_idx) > 0:
                    metrics[f'face_{face}_lsys_rms_strict'] = np.sqrt(np.mean(np.abs(r[strict_idx])**2))
                    metrics[f'face_{face}_lsys_max_strict'] = np.max(np.abs(r[strict_idx]))

            # --- Debug print for one node per face ---
            if debug_print:
                print("\n[DEBUG] Per-face matrix row and manual residuals:")
                for face, cond in face_defs.items():
                    # Pick a representative strict node
                    if face == 'top':
                        ix, iy, iz = 1, 1, Nz-1
                        delta = dz
                        p_b = p[ix,iy,iz]
                        p_in = p[ix,iy,iz-1]
                    elif face == 'xmin':
                        ix, iy, iz = 0, 1, 1
                        delta = dx
                        p_b = p[ix,iy,iz]
                        p_in = p[ix+1,iy,iz]
                    elif face == 'xmax':
                        ix, iy, iz = Nx-1, 1, 1
                        delta = dx
                        p_b = p[ix,iy,iz]
                        p_in = p[ix-1,iy,iz]
                    elif face == 'ymin':
                        ix, iy, iz = 1, 0, 1
                        delta = dy
                        p_b = p[ix,iy,iz]
                        p_in = p[ix,iy+1,iz]
                    elif face == 'ymax':
                        ix, iy, iz = 1, Ny-1, 1
                        delta = dy
                        p_b = p[ix,iy,iz]
                        p_in = p[ix,iy-1,iz]
                    idx = operator._flatten_index(ix,iy,iz)
                    row = A.getrow(idx).toarray().ravel()
                    nonzero_idx = np.nonzero(row)[0]
                    print(f"  Face: {face}, node=({ix},{iy},{iz}), k={k:.3g}, Δ={delta:.3g}")
                    print(f"    p_b={p_b:.3g}, p_in={p_in:.3g}")
                    # Manual row residual (should match matrix)
                    if face == 'top':
                        manual = (1/dz-1j*k)*p_b - (1/dz)*p_in
                    else:
                        manual = (1/delta-1j*k)*p_b - (1/delta)*p_in
                    print(f"    Manual row residual: {manual:.3g}")
                    print(f"    Matrix row residual: {(A@u-b)[idx]:.3g}")
                    print(f"    Matrix row nonzeros:")
                    for j in nonzero_idx:
                        print(f"      col={j}, value={row[j]:.3g}")
                    print(f"    b[idx]={b[idx]:.3g}")
        return metrics

    def compute_gradients(self):
        """Compute dpdx, dpdy, dpdz using central differences (one-sided at boundaries)."""
        p = self.p
        dx, dy, dz = self.grid.dx, self.grid.dy, self.grid.dz
        dpdx = np.zeros_like(p, dtype=np.complex128)
        dpdy = np.zeros_like(p, dtype=np.complex128)
        dpdz = np.zeros_like(p, dtype=np.complex128)
        # x
        dpdx[1:-1,:,:] = (p[2:,:,:] - p[:-2,:,:]) / (2*dx)
        dpdx[0,:,:]    = (p[1,:,:] - p[0,:,:]) / dx
        dpdx[-1,:,:]   = (p[-1,:,:] - p[-2,:,:]) / dx
        # y
        dpdy[:,1:-1,:] = (p[:,2:,:] - p[:,:-2,:]) / (2*dy)
        dpdy[:,0,:]    = (p[:,1,:] - p[:,0,:]) / dy
        dpdy[:,-1,:]   = (p[:,-1,:] - p[:,-2,:]) / dy
        # z
        dpdz[:,:,1:-1] = (p[:,:,2:] - p[:,:,:-2]) / (2*dz)
        dpdz[:,:,0]    = (p[:,:,1] - p[:,:,0]) / dz
        dpdz[:,:,-1]   = (p[:,:,-1] - p[:,:,-2]) / dz
        return dpdx, dpdy, dpdz

    def compute_velocity_magnitude2(self):
        """Compute |v|^2 from grad(p)."""
        dpdx, dpdy, dpdz = self.compute_gradients()
        omega = self.omega
        rho0 = self.medium_props.rho0
        v_x = -(1/(1j*omega*rho0)) * dpdx
        v_y = -(1/(1j*omega*rho0)) * dpdy
        v_z = -(1/(1j*omega*rho0)) * dpdz
        v2 = np.abs(v_x)**2 + np.abs(v_y)**2 + np.abs(v_z)**2
        return v2, v_x, v_y, v_z

    def compute_gorkov_potential(self, a, f1, f2):
        """Compute Gor'kov potential U[ix,iy,iz] for given particle radius a and contrast factors f1, f2."""
        p = self.p
        rho0 = self.medium_props.rho0
        c0 = self.medium_props.c0
        v2, _, _, _ = self.compute_velocity_magnitude2()
        U = (4 * np.pi * a**3 / 3) * (
            (f1 / (2 * rho0 * c0**2)) * np.abs(p)**2 - (3 * f2 / 4) * rho0 * v2
        )
        self.U = U
        return U

    def compute_radiation_force(self):
        """Compute Fx, Fy, Fz = -grad(U) using central differences."""
        U = self.U
        dx, dy, dz = self.grid.dx, self.grid.dy, self.grid.dz
        Fx = np.zeros_like(U)
        Fy = np.zeros_like(U)
        Fz = np.zeros_like(U)
        # x
        Fx[1:-1,:,:] = -(U[2:,:,:] - U[:-2,:,:]) / (2*dx)
        Fx[0,:,:]    = -(U[1,:,:] - U[0,:,:]) / dx
        Fx[-1,:,:]   = -(U[-1,:,:] - U[-2,:,:]) / dx
        # y
        Fy[:,1:-1,:] = -(U[:,2:,:] - U[:,:-2,:]) / (2*dy)
        Fy[:,0,:]    = -(U[:,1,:] - U[:,0,:]) / dy
        Fy[:,-1,:]   = -(U[:,-1,:] - U[:,-2,:]) / dy
        # z
        Fz[:,:,1:-1] = -(U[:,:,2:] - U[:,:,:-2]) / (2*dz)
        Fz[:,:,0]    = -(U[:,:,1] - U[:,:,0]) / dz
        Fz[:,:,-1]   = -(U[:,:,-1] - U[:,:,-2]) / dz
        self.Fx = Fx
        self.Fy = Fy
        self.Fz = Fz
        return Fx, Fy, Fz


class Helmholtz3DSolver:
    """
    Wrapper for 3D Helmholtz operator, mirroring Helmholtz25DSolver pattern.
    """
    def __init__(self, grid: Grid3D, omega: float, medium_props):
        self.grid = grid
        self.omega = omega
        self.medium_props = medium_props
        self.c0 = getattr(medium_props, 'c0', 1500.0)
        self.k = omega / self.c0
        self.op = Helmholtz3DOperator(grid, self.k)

    def solve(self, p_bot: np.ndarray):
        """
        Solve 3D Helmholtz problem with bottom Dirichlet drive.
        Args:
            p_bot: complex array, shape (Nx, Ny)
        Returns:
            ExtendedField3D
        """
        p = self.op.solve(p_bot)
        return ExtendedField3D(p, self.grid, self.k, self.omega, self.medium_props)
