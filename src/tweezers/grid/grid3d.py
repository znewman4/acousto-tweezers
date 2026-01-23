import numpy as np

class Grid3D:
    """
    Uniform 3D Cartesian grid for a dish-water acoustic domain.
    Coordinates: (x, y, z) in meters.
    Indexing: p[ix, iy, iz] with shape (Nx, Ny, Nz)
    Domain extents:
        x in [0, Lx], y in [0, Ly], z in [0, H]
        Lx = (Nx-1)*dx, Ly = (Ny-1)*dy, H = (Nz-1)*dz
    """
    def __init__(self, Lx, Ly, H, dx, dy, dz):
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.Nx = int(round(Lx / dx)) + 1
        self.Ny = int(round(Ly / dy)) + 1
        self.Nz = int(round(H / dz)) + 1
        self.x = np.linspace(0, Lx, self.Nx)
        self.y = np.linspace(0, Ly, self.Ny)
        self.z = np.linspace(0, H, self.Nz)

    @property
    def shape(self):
        return (self.Nx, self.Ny, self.Nz)

    @property
    def bottom_shape(self):
        return (self.Nx, self.Ny)

    # Face index helpers
    @property
    def bottom(self):
        return 0
    @property
    def top(self):
        return self.Nz - 1
    @property
    def x_min(self):
        return 0
    @property
    def x_max(self):
        return self.Nx - 1
    @property
    def y_min(self):
        return 0
    @property
    def y_max(self):
        return self.Ny - 1

    # Optional: meshgrid views (not stored)
    def meshgrid(self, indexing='ij'):
        return np.meshgrid(self.x, self.y, self.z, indexing=indexing)

if __name__ == "__main__":
    # Demo/self-test
    Lx, Ly, H = 0.1, 0.1, 0.02
    dx, dy, dz = 0.002, 0.002, 0.002
    grid = Grid3D(Lx, Ly, H, dx, dy, dz)
    print("Grid shape:", grid.shape)
    print("Grid spacings:", grid.dx, grid.dy, grid.dz)
    print("Bottom face shape (z=0):", (grid.Nx, grid.Ny))
    print("Top face shape (z=H):", (grid.Nx, grid.Ny))
    print("x-min face shape (x=0):", (grid.Ny, grid.Nz))
    print("x-max face shape (x=Lx):", (grid.Ny, grid.Nz))
    print("y-min face shape (y=0):", (grid.Nx, grid.Nz))
    print("y-max face shape (y=Ly):", (grid.Nx, grid.Nz))
