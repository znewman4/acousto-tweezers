import numpy as np

def plate_transmission_operator(k, k_plate, rho, rho_plate, c, c_plate, h):
    """
    Computes the transmission coefficient for a fluid-plate-fluid system (longitudinal waves).
    Args:
        k: 2D array, wavenumber in fluid (1/m)
        k_plate: 2D array, wavenumber in plate (1/m)
        rho: fluid density (kg/m^3)
        rho_plate: plate density (kg/m^3)
        c: fluid sound speed (m/s)
        c_plate: plate sound speed (m/s)
        h: plate thickness (m)
    Returns:
        T: 2D array, transmission coefficient (complex)
    """
    Z = rho * c
    Zp = rho_plate * c_plate
    r = Zp / Z
    phi = k_plate * h
    numerator = 4 * r * np.exp(-1j * k * h)
    denominator = (r + 1)**2 * np.exp(-1j * k * h) - (r - 1)**2 * np.exp(1j * k * h)
    T = numerator / denominator
    return T

def apply_plate_transmission(p_bath, dx, dy, k, k_plate, rho, rho_plate, c, c_plate, h):
    """
    Applies the plate transmission operator in the Fourier domain.
    Args:
        p_bath: 2D complex array (pressure after bath)
        dx, dy: grid spacing (m)
        k, k_plate: wavenumbers (1/m)
        rho, rho_plate: densities (kg/m^3)
        c, c_plate: sound speeds (m/s)
        h: plate thickness (m)
    Returns:
        p_bot: 2D complex array (pressure after plate)
    """
    ny, nx = p_bath.shape
    fx = np.fft.fftfreq(nx, d=dx)
    fy = np.fft.fftfreq(ny, d=dy)
    FX, FY = np.meshgrid(fx, fy)
    KX = 2 * np.pi * FX
    KY = 2 * np.pi * FY
    K = np.sqrt(KX**2 + KY**2)
    K_plate = K * c / c_plate
    P_bath = np.fft.fft2(p_bath)
    T = plate_transmission_operator(K, K_plate, rho, rho_plate, c, c_plate, h)
    P_bot = P_bath * T
    p_bot = np.fft.ifft2(P_bot)
    return p_bot
