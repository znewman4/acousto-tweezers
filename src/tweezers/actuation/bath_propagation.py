import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift

def angular_spectrum_propagate(p0, dx, dy, d, k):
    """
    Angular spectrum propagation of a field p0 over distance d.
    Args:
        p0: 2D complex array (pressure on lens plane)
        dx, dy: grid spacing (m)
        d: propagation distance (m)
        k: wavenumber in bath (1/m)
    Returns:
        p: propagated field at distance d
    """
    ny, nx = p0.shape
    fx = np.fft.fftfreq(nx, d=dx)
    fy = np.fft.fftfreq(ny, d=dy)
    FX, FY = np.meshgrid(fx, fy)
    KX = 2 * np.pi * FX
    KY = 2 * np.pi * FY
    KZ = np.sqrt(np.maximum(0, k**2 - KX**2 - KY**2))
    P0 = fft2(p0)
    H = np.exp(1j * KZ * d)
    P = P0 * H
    p = ifft2(P)
    return p
