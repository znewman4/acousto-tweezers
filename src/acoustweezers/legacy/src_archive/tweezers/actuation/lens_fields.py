import numpy as np

def lens_focus(x, y, x0, y0, f, sigma, k):
    """
    Generates a unit-amplitude focused field on the lens plane.
    Args:
        x, y: 2D meshgrid arrays (m)
        x0, y0: focus center (m)
        f: focal length (m)
        sigma: Gaussian envelope width (m)
        k: wavenumber (1/m)
    Returns:
        p_lens: complex pressure field (unit amplitude)
    """
    r2 = (x - x0)**2 + (y - y0)**2
    phase = -k * ((r2) / (2 * f))
    envelope = np.exp(-r2 / (2 * sigma**2))
    return envelope * np.exp(1j * phase)

def lens_vortex(x, y, x0, y0, ell, sigma, k):
    """
    Generates a unit-amplitude vortex field on the lens plane.
    Args:
        x, y: 2D meshgrid arrays (m)
        x0, y0: vortex center (m)
        ell: topological charge (int)
        sigma: Gaussian envelope width (m)
        k: wavenumber (1/m)
    Returns:
        p_lens: complex pressure field (unit amplitude)
    """
    X = x - x0
    Y = y - y0
    r2 = X**2 + Y**2
    theta = np.arctan2(Y, X)
    envelope = np.exp(-r2 / (2 * sigma**2))
    return envelope * np.exp(1j * (ell * theta + k * 0 * X))

def lens_axicon(x, y, x0, y0, alpha, sigma, k):
    """
    Generates a unit-amplitude axicon (Bessel-like) field on the lens plane.
    Args:
        x, y: 2D meshgrid arrays (m)
        x0, y0: axicon center (m)
        alpha: axicon angle (rad)
        sigma: Gaussian envelope width (m)
        k: wavenumber (1/m)
    Returns:
        p_lens: complex pressure field (unit amplitude)
    """
    X = x - x0
    Y = y - y0
    r = np.sqrt(X**2 + Y**2)
    envelope = np.exp(-r**2 / (2 * sigma**2))
    phase = k * r * np.sin(alpha)
    return envelope * np.exp(1j * phase)
