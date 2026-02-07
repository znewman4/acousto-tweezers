"""
Fluid-solid coupling for acoustic-elastic simulations.

Implements the interface conditions between fluid and solid domains:

1. Dynamic continuity (traction balance):
   σ·n = -p·n    (solid traction = fluid pressure)

2. Kinematic continuity (normal velocity):
   v·n = iω·u·n  (fluid normal velocity = solid normal velocity)

These conditions are essential for correctly modeling:
- Energy transfer between fluid and solid
- Standing wave patterns in coupled system
- Resonances in dish plate and walls
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict, Optional
import numpy as np
import scipy.sparse as sp

from .materials import SolidMaterial
from ..acoustics.materials import FluidMaterial


@dataclass
class CouplingInterface:
    """
    Definition of a fluid-solid coupling interface.
    
    Parameters
    ----------
    fluid_side : str
        Which side has fluid ('top' or 'bottom' for z-normal interfaces).
    normal : np.ndarray
        Unit normal pointing from solid into fluid.
    z_position : float
        Z-coordinate of the interface.
    """
    fluid_side: str
    normal: np.ndarray
    z_position: float
    
    # Material properties
    fluid: Optional[FluidMaterial] = None
    solid: Optional[SolidMaterial] = None


class FluidSolidCoupling:
    """
    Manages fluid-solid coupling at interfaces.
    
    This class builds the coupling matrices that connect fluid pressure
    to solid displacement at interface nodes.
    """
    
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        fluid: FluidMaterial,
        solid: SolidMaterial,
    ):
        """
        Initialize coupling.
        
        Parameters
        ----------
        x, y : np.ndarray
            Interface coordinate arrays.
        fluid : FluidMaterial
            Fluid material properties.
        solid : SolidMaterial
            Solid material properties.
        """
        self.x = x
        self.y = y
        self.Nx = len(x)
        self.Ny = len(y)
        self.dx = x[1] - x[0] if len(x) > 1 else 1.0
        self.dy = y[1] - y[0] if len(y) > 1 else 1.0
        
        self.fluid = fluid
        self.solid = solid
        
        # Interface area per node (for integration)
        self.dA = self.dx * self.dy
    
    @property
    def n_nodes(self) -> int:
        """Number of interface nodes."""
        return self.Nx * self.Ny
    
    def pressure_to_traction(
        self,
        p: np.ndarray,
        normal: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert fluid pressure to traction on solid surface.
        
        Implements: t = σ·n = -p·n
        
        Parameters
        ----------
        p : np.ndarray
            Pressure at interface, shape (Nx, Ny).
        normal : np.ndarray
            Unit normal from solid into fluid.
        
        Returns
        -------
        tx, ty, tz : np.ndarray
            Traction components, shape (Nx, Ny).
        """
        return (-p * normal[0], -p * normal[1], -p * normal[2])
    
    def displacement_to_velocity(
        self,
        ux: np.ndarray,
        uy: np.ndarray,
        uz: np.ndarray,
        omega: float,
        normal: np.ndarray,
    ) -> np.ndarray:
        """
        Convert solid displacement to normal velocity.
        
        Implements: v_n = iω·u·n
        
        Parameters
        ----------
        ux, uy, uz : np.ndarray
            Displacement at interface, shape (Nx, Ny).
        omega : float
            Angular frequency.
        normal : np.ndarray
            Unit normal.
        
        Returns
        -------
        v_n : np.ndarray
            Normal velocity, shape (Nx, Ny).
        """
        u_n = ux * normal[0] + uy * normal[1] + uz * normal[2]
        return 1j * omega * u_n
    
    def velocity_to_pressure_gradient(
        self,
        v_n: np.ndarray,
        omega: float,
    ) -> np.ndarray:
        """
        Convert normal velocity to pressure gradient BC.
        
        From: v = -(1/iωρ)∇p
        We get: ∂p/∂n = -iωρ·v_n
        
        Parameters
        ----------
        v_n : np.ndarray
            Normal velocity at interface.
        omega : float
            Angular frequency.
        
        Returns
        -------
        dpdn : np.ndarray
            Normal pressure gradient.
        """
        return -1j * omega * self.fluid.rho * v_n
    
    def transmission_coefficient(self, omega: float) -> complex:
        """
        Compute pressure transmission coefficient at interface.
        
        For normal incidence plane wave:
        T = 2*Z_s / (Z_s + Z_f)
        
        Parameters
        ----------
        omega : float
            Angular frequency (for frequency-dependent materials).
        
        Returns
        -------
        T : complex
            Transmission coefficient.
        """
        Z_f = self.fluid.Z  # Fluid impedance
        Z_s = self.solid.Z_longitudinal  # Solid longitudinal impedance
        
        return 2 * Z_s / (Z_s + Z_f)
    
    def reflection_coefficient(self, omega: float) -> complex:
        """
        Compute pressure reflection coefficient at interface.
        
        R = (Z_s - Z_f) / (Z_s + Z_f)
        
        Parameters
        ----------
        omega : float
            Angular frequency.
        
        Returns
        -------
        R : complex
            Reflection coefficient.
        """
        Z_f = self.fluid.Z
        Z_s = self.solid.Z_longitudinal
        
        return (Z_s - Z_f) / (Z_s + Z_f)
    
    def build_coupling_matrix_fluid_to_solid(
        self,
        omega: float,
        normal: np.ndarray,
    ) -> sp.csr_matrix:
        """
        Build matrix that maps fluid pressure to solid RHS.
        
        For each solid displacement DOF at the interface, this matrix
        gives the contribution from fluid pressure (as traction).
        
        Parameters
        ----------
        omega : float
            Angular frequency.
        normal : np.ndarray
            Interface normal (solid to fluid).
        
        Returns
        -------
        C : sp.csr_matrix
            Coupling matrix, shape (3*n_nodes, n_nodes).
        """
        n = self.n_nodes
        C = sp.lil_matrix((3 * n, n), dtype=np.complex128)
        
        for idx in range(n):
            # Pressure -> traction: t = -p*n
            # RHS contribution: traction * area
            C[3*idx + 0, idx] = -normal[0] * self.dA  # tx from p
            C[3*idx + 1, idx] = -normal[1] * self.dA  # ty from p
            C[3*idx + 2, idx] = -normal[2] * self.dA  # tz from p
        
        return C.tocsr()
    
    def build_coupling_matrix_solid_to_fluid(
        self,
        omega: float,
        normal: np.ndarray,
    ) -> sp.csr_matrix:
        """
        Build matrix that maps solid displacement to fluid BC.
        
        Converts solid normal displacement to Neumann BC for pressure.
        
        Parameters
        ----------
        omega : float
            Angular frequency.
        normal : np.ndarray
            Interface normal.
        
        Returns
        -------
        C : sp.csr_matrix
            Coupling matrix, shape (n_nodes, 3*n_nodes).
        """
        n = self.n_nodes
        C = sp.lil_matrix((n, 3 * n), dtype=np.complex128)
        
        # v_n = iω(u·n) -> ∂p/∂n = -iωρ*v_n = ω²ρ(u·n)
        factor = omega**2 * self.fluid.rho
        
        for idx in range(n):
            C[idx, 3*idx + 0] = factor * normal[0]  # from ux
            C[idx, 3*idx + 1] = factor * normal[1]  # from uy
            C[idx, 3*idx + 2] = factor * normal[2]  # from uz
        
        return C.tocsr()
    
    def interface_energy_flux(
        self,
        p: np.ndarray,
        v_n: np.ndarray,
    ) -> float:
        """
        Compute time-averaged energy flux through interface.
        
        I = 0.5 * Re(p * conj(v_n))
        
        Parameters
        ----------
        p : np.ndarray
            Pressure at interface.
        v_n : np.ndarray
            Normal velocity at interface.
        
        Returns
        -------
        P : float
            Total power [W].
        """
        I = 0.5 * np.real(p * np.conj(v_n))
        return np.sum(I) * self.dA
    
    def interface_continuity_error(
        self,
        p_fluid: np.ndarray,
        v_fluid_n: np.ndarray,
        stress_solid: np.ndarray,
        v_solid_n: np.ndarray,
        normal: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute residuals of interface conditions.
        
        Parameters
        ----------
        p_fluid : np.ndarray
            Fluid pressure at interface.
        v_fluid_n : np.ndarray
            Fluid normal velocity.
        stress_solid : np.ndarray
            Solid normal stress (σ·n)·n.
        v_solid_n : np.ndarray
            Solid normal velocity.
        normal : np.ndarray
            Interface normal.
        
        Returns
        -------
        errors : Dict[str, float]
            Residual metrics.
        """
        # Traction balance: σ_nn = -p
        traction_error = stress_solid + p_fluid
        traction_residual = np.sqrt(np.mean(np.abs(traction_error)**2))
        
        # Velocity continuity: v_f·n = v_s·n
        velocity_error = v_fluid_n - v_solid_n
        velocity_residual = np.sqrt(np.mean(np.abs(velocity_error)**2))
        
        # Normalize by magnitude
        p_scale = np.sqrt(np.mean(np.abs(p_fluid)**2)) + 1e-20
        v_scale = np.sqrt(np.mean(np.abs(v_fluid_n)**2)) + 1e-20
        
        return {
            'traction_residual': float(traction_residual),
            'traction_relative': float(traction_residual / p_scale),
            'velocity_residual': float(velocity_residual),
            'velocity_relative': float(velocity_residual / v_scale),
        }


class PlateTransmission:
    """
    Analytical model for acoustic transmission through elastic plate.
    
    For plane wave at normal incidence through a plate of thickness h:
    T = 4*r*exp(-ik*h) / [(r+1)²*exp(-ik_p*h) - (r-1)²*exp(+ik_p*h)]
    
    where r = Z_plate/Z_fluid is the impedance ratio.
    """
    
    def __init__(
        self,
        fluid: FluidMaterial,
        plate: SolidMaterial,
        thickness: float,
    ):
        """
        Initialize plate transmission model.
        
        Parameters
        ----------
        fluid : FluidMaterial
            Surrounding fluid (assumed same on both sides).
        plate : SolidMaterial
            Plate material.
        thickness : float
            Plate thickness [m].
        """
        self.fluid = fluid
        self.plate = plate
        self.h = thickness
    
    def transmission_coefficient(self, omega: float) -> complex:
        """
        Compute complex transmission coefficient.
        
        Parameters
        ----------
        omega : float
            Angular frequency [rad/s].
        
        Returns
        -------
        T : complex
            Transmission coefficient (pressure ratio).
        """
        Z_f = self.fluid.Z
        Z_p = self.plate.Z_longitudinal
        r = Z_p / Z_f
        
        k_f = omega / self.fluid.c
        
        # Complex wavenumber in plate (includes damping)
        c_p = np.sqrt(self.plate.mu_complex / self.plate.rho)  # Approximate
        k_p = omega / c_p
        
        h = self.h
        
        numerator = 4 * r * np.exp(-1j * k_f * h)
        denominator = (r + 1)**2 * np.exp(-1j * k_p * h) - (r - 1)**2 * np.exp(1j * k_p * h)
        
        return numerator / denominator
    
    def transmission_loss_db(self, omega: float) -> float:
        """
        Compute transmission loss in dB.
        
        TL = -20*log10(|T|)
        """
        T = self.transmission_coefficient(omega)
        return -20 * np.log10(np.abs(T) + 1e-20)
    
    def resonance_frequencies(self, n_modes: int = 5) -> np.ndarray:
        """
        Compute plate thickness resonance frequencies.
        
        At resonance: k_p * h = n*π -> f = n*c_p/(2*h)
        
        Parameters
        ----------
        n_modes : int
            Number of modes to compute.
        
        Returns
        -------
        f_res : np.ndarray
            Resonance frequencies [Hz].
        """
        c_p = self.plate.c_longitudinal
        return np.array([n * c_p / (2 * self.h) for n in range(1, n_modes + 1)])


if __name__ == "__main__":
    from ..acoustics.materials import MaterialDatabase
    from .materials import SolidMaterialDatabase
    
    # Demo: compute plate transmission
    water = MaterialDatabase.water(25.0)
    ps = SolidMaterialDatabase.polystyrene()
    
    plate = PlateTransmission(
        fluid=water,
        plate=ps,
        thickness=1e-3,  # 1mm
    )
    
    # Transmission vs frequency
    freqs = np.logspace(4, 7, 100)  # 10 kHz to 10 MHz
    
    print("Plate transmission analysis (1mm polystyrene in water):")
    print(f"Resonance frequencies: {plate.resonance_frequencies(3)/1e6} MHz")
    
    # At 1 MHz
    f = 1e6
    omega = 2 * np.pi * f
    T = plate.transmission_coefficient(omega)
    TL = plate.transmission_loss_db(omega)
    
    print(f"\nAt {f/1e6:.1f} MHz:")
    print(f"  |T| = {np.abs(T):.3f}")
    print(f"  TL = {TL:.1f} dB")
    print(f"  Phase = {np.angle(T)*180/np.pi:.1f}°")
