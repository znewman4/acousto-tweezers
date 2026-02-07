import numpy as np

def drive_constant(grid, amp=1.0, phase=0.0):
    """Constant Dirichlet drive: p_bot = amp * exp(i*phase) everywhere."""
    return amp * np.exp(1j * phase) * np.ones((grid.Nx, grid.Ny), dtype=np.complex128)

def drive_gaussian(grid, x0, y0, sigma, amp=1.0, phase=0.0):
    """Gaussian spot centered at (x0, y0)."""
    X, Y = np.meshgrid(grid.x, grid.y, indexing='ij')
    r2 = (X - x0)**2 + (Y - y0)**2
    return amp * np.exp(1j * phase) * np.exp(-0.5 * r2 / sigma**2)

def drive_plane_wave(grid, kx, ky, amp=1.0):
    """Plane wave: exp(i*(kx*x + ky*y))."""
    X, Y = np.meshgrid(grid.x, grid.y, indexing='ij')
    return amp * np.exp(1j * (kx * X + ky * Y))

def drive_vortex(grid, x0, y0, ell, sigma, amp=1.0):
    """Vortex: (r/sigma)^|ell| * exp(-r^2/(2*sigma^2)) * exp(i*ell*theta)"""
    X, Y = np.meshgrid(grid.x, grid.y, indexing='ij')
    dx = X - x0
    dy = Y - y0
    eps = 1e-12
    r = np.sqrt(dx**2 + dy**2) + eps
    theta = np.arctan2(dy, dx)
    envelope = np.exp(-0.5 * r**2 / sigma**2)
    p_bot = amp * (r/sigma)**abs(ell) * envelope * np.exp(1j * ell * theta)
    return p_bot

def drive_axicon(grid, x0, y0, alpha, sigma, amp=1.0):
    """Axicon: exp(-r^2/(2*sigma^2)) * exp(i*alpha*r)"""
    X, Y = np.meshgrid(grid.x, grid.y, indexing='ij')
    dx = X - x0
    dy = Y - y0
    r = np.sqrt(dx**2 + dy**2)
    envelope = np.exp(-0.5 * r**2 / sigma**2)
    p_bot = amp * envelope * np.exp(1j * alpha * r)
    return p_bot
