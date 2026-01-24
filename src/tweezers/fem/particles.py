"""
Radiation force and particle dynamics for acoustic trapping.

Implements the Gor'kov radiation force and particle trajectory integration.

Gor'kov force (from MASTER BRIEF):

    F_rad = -∇U

    U = V [f₁ κ/3 ⟨p²⟩ - f₂ ρ/2 ⟨v²⟩]

where:
    V = 4πa³/3 (particle volume)
    f₁ = 1 - κₚ/κ (monopole coefficient)
    f₂ = 2(ρₚ-ρ)/(2ρₚ+ρ) (dipole coefficient)
    ⟨·⟩ denotes time average

Particle ODE (overdamped limit):

    ẋ = u_stream(x) + μ F_rad(x)

where μ = 1/(6πηa) is the Stokes mobility.

References
----------
- Gor'kov (1962): On the forces acting on a small particle in an acoustical
  field in an ideal fluid
- Settnes & Bruus (2012): Forces acting on a small particle in an acoustical
  field in a viscous fluid
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .domains import DomainType
from .geometry import FEMMesh
from .materials import FluidMaterial, ParticleMaterial, MaterialDatabase
from .config import FEMConfig
from .acoustics import AcousticField
from .streaming import StreamingField


@dataclass
class GorkovPotential:
    """
    Gor'kov potential field for radiation force calculation.
    """
    # Grid coordinates
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    
    # Potential (real, units: Joules)
    U: np.ndarray  # (nx, ny, nz)
    
    # Force components (negative gradient of U)
    Fx: np.ndarray
    Fy: np.ndarray
    Fz: np.ndarray
    
    # Particle properties used
    particle_radius: float
    particle_density: float
    
    # Mesh reference
    mesh: Optional[FEMMesh] = None
    
    @property
    def U_min(self) -> float:
        """Minimum potential (trap location)."""
        return float(np.min(self.U))
    
    @property
    def U_max(self) -> float:
        """Maximum potential."""
        return float(np.max(self.U))
    
    @property
    def trap_depth(self) -> float:
        """Trap depth ΔU = U_max - U_min [J]."""
        return self.U_max - self.U_min
    
    def find_trap_locations(self, threshold: float = 0.1) -> List[np.ndarray]:
        """
        Find trap locations (local minima of U).
        
        Parameters
        ----------
        threshold : float
            Fraction of trap depth to consider as trap region.
        
        Returns
        -------
        locations : list of np.ndarray
            List of (x, y, z) trap positions.
        """
        # Simple approach: find points where U is close to minimum
        U_threshold = self.U_min + threshold * self.trap_depth
        
        # Find local minima using gradient
        trap_mask = self.U < U_threshold
        
        # Get coordinates of trap region
        traps = []
        if np.any(trap_mask):
            # Center of mass of trap region
            X, Y, Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')
            trap_x = np.mean(X[trap_mask])
            trap_y = np.mean(Y[trap_mask])
            trap_z = np.mean(Z[trap_mask])
            traps.append(np.array([trap_x, trap_y, trap_z]))
        
        return traps


@dataclass
class ParticleTrajectory:
    """
    Trajectory of a single particle.
    """
    # Time array
    t: np.ndarray
    
    # Position arrays
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    
    # Initial position
    x0: np.ndarray
    
    @property
    def positions(self) -> np.ndarray:
        """(N, 3) array of positions."""
        return np.column_stack([self.x, self.y, self.z])
    
    @property
    def final_position(self) -> np.ndarray:
        """Final (x, y, z) position."""
        return np.array([self.x[-1], self.y[-1], self.z[-1]])
    
    @property
    def distance_traveled(self) -> float:
        """Total path length traveled."""
        dx = np.diff(self.x)
        dy = np.diff(self.y)
        dz = np.diff(self.z)
        return float(np.sum(np.sqrt(dx**2 + dy**2 + dz**2)))


def compute_gorkov_potential(
    acoustic_field: AcousticField,
    fluid: FluidMaterial,
    particle: ParticleMaterial,
) -> GorkovPotential:
    """
    Compute Gor'kov potential from acoustic field.
    
    U = V [f₁ κ/3 ⟨p²⟩ - f₂ ρ/2 ⟨v²⟩]
    
    For time-harmonic fields:
        ⟨p²⟩ = |p̂|²/2
        ⟨v²⟩ = |v̂|²/2
    
    Parameters
    ----------
    acoustic_field : AcousticField
        Acoustic pressure field.
    fluid : FluidMaterial
        Fluid properties.
    particle : ParticleMaterial
        Particle properties.
    
    Returns
    -------
    gorkov : GorkovPotential
        Gor'kov potential and force field.
    """
    mesh = acoustic_field.mesh
    
    # Get pressure amplitude (3D grid)
    p_3d = acoustic_field.p_grid
    p2_avg = np.abs(p_3d)**2 / 2  # Time-averaged ⟨p²⟩
    
    # Get velocity amplitude
    vx, vy, vz = acoustic_field.compute_velocity()
    v2_avg = (np.abs(vx)**2 + np.abs(vy)**2 + np.abs(vz)**2) / 2
    
    # Monopole and dipole coefficients
    f1 = particle.monopole_coefficient(fluid)
    f2 = particle.dipole_coefficient(fluid)
    
    # Compressibility and density
    kappa = fluid.compressibility()
    rho = fluid.rho
    
    # Particle volume
    V = particle.volume
    
    # Gor'kov potential
    U = V * (f1 * kappa / 3 * p2_avg - f2 * rho / 2 * v2_avg)
    
    # Compute force as negative gradient
    dx, dy, dz = mesh.dx, mesh.dy, mesh.dz
    
    Fx = np.zeros_like(U)
    Fy = np.zeros_like(U)
    Fz = np.zeros_like(U)
    
    # Central differences for gradient
    Fx[1:-1, :, :] = -(U[2:, :, :] - U[:-2, :, :]) / (2 * dx)
    Fy[:, 1:-1, :] = -(U[:, 2:, :] - U[:, :-2, :]) / (2 * dy)
    Fz[:, :, 1:-1] = -(U[:, :, 2:] - U[:, :, :-2]) / (2 * dz)
    
    # One-sided at boundaries
    Fx[0, :, :] = -(U[1, :, :] - U[0, :, :]) / dx
    Fx[-1, :, :] = -(U[-1, :, :] - U[-2, :, :]) / dx
    Fy[:, 0, :] = -(U[:, 1, :] - U[:, 0, :]) / dy
    Fy[:, -1, :] = -(U[:, -1, :] - U[:, -2, :]) / dy
    Fz[:, :, 0] = -(U[:, :, 1] - U[:, :, 0]) / dz
    Fz[:, :, -1] = -(U[:, :, -1] - U[:, :, -2]) / dz
    
    return GorkovPotential(
        x=mesh.x,
        y=mesh.y,
        z=mesh.z,
        U=U.real,  # U should be real
        Fx=Fx.real,
        Fy=Fy.real,
        Fz=Fz.real,
        particle_radius=particle.radius,
        particle_density=particle.rho,
        mesh=mesh,
    )


class ParticleDynamics:
    """
    Particle trajectory integration.
    
    Solves the overdamped equation:
        ẋ = u_stream(x) + μ F_rad(x)
    
    using explicit Euler or RK4 integration.
    """
    
    def __init__(
        self,
        gorkov: GorkovPotential,
        streaming: Optional[StreamingField],
        fluid: FluidMaterial,
        particle: ParticleMaterial,
    ):
        """
        Initialize particle dynamics.
        
        Parameters
        ----------
        gorkov : GorkovPotential
            Radiation force field.
        streaming : StreamingField, optional
            Streaming velocity field.
        fluid : FluidMaterial
            Fluid properties.
        particle : ParticleMaterial
            Particle properties.
        """
        self.gorkov = gorkov
        self.streaming = streaming
        self.fluid = fluid
        self.particle = particle
        
        # Stokes mobility
        self.mobility = particle.mobility(fluid)
        
        # Build interpolators for force field
        self._build_interpolators()
    
    def _build_interpolators(self):
        """Build interpolation functions for fields."""
        x, y, z = self.gorkov.x, self.gorkov.y, self.gorkov.z
        
        # Force interpolators
        self._interp_Fx = RegularGridInterpolator(
            (x, y, z), self.gorkov.Fx,
            method='linear', bounds_error=False, fill_value=0.0
        )
        self._interp_Fy = RegularGridInterpolator(
            (x, y, z), self.gorkov.Fy,
            method='linear', bounds_error=False, fill_value=0.0
        )
        self._interp_Fz = RegularGridInterpolator(
            (x, y, z), self.gorkov.Fz,
            method='linear', bounds_error=False, fill_value=0.0
        )
        
        # Streaming velocity interpolators
        if self.streaming is not None:
            mesh = self.streaming.mesh
            nx, ny, nz = mesh.nx, mesh.ny, mesh.nz
            
            ux = self.streaming.ux.reshape((nz, ny, nx)).transpose((2, 1, 0))
            uy = self.streaming.uy.reshape((nz, ny, nx)).transpose((2, 1, 0))
            uz = self.streaming.uz.reshape((nz, ny, nx)).transpose((2, 1, 0))
            
            self._interp_ux = RegularGridInterpolator(
                (x, y, z), ux,
                method='linear', bounds_error=False, fill_value=0.0
            )
            self._interp_uy = RegularGridInterpolator(
                (x, y, z), uy,
                method='linear', bounds_error=False, fill_value=0.0
            )
            self._interp_uz = RegularGridInterpolator(
                (x, y, z), uz,
                method='linear', bounds_error=False, fill_value=0.0
            )
        else:
            self._interp_ux = None
            self._interp_uy = None
            self._interp_uz = None
    
    def velocity(self, pos: np.ndarray) -> np.ndarray:
        """
        Compute particle velocity at position.
        
        v = u_stream + μ F_rad
        
        Parameters
        ----------
        pos : np.ndarray
            Position (x, y, z).
        
        Returns
        -------
        vel : np.ndarray
            Velocity (vx, vy, vz).
        """
        # Radiation force contribution
        Fx = float(self._interp_Fx(pos))
        Fy = float(self._interp_Fy(pos))
        Fz = float(self._interp_Fz(pos))
        
        vx = self.mobility * Fx
        vy = self.mobility * Fy
        vz = self.mobility * Fz
        
        # Streaming contribution
        if self._interp_ux is not None:
            vx += float(self._interp_ux(pos))
            vy += float(self._interp_uy(pos))
            vz += float(self._interp_uz(pos))
        
        return np.array([vx, vy, vz])
    
    def integrate(
        self,
        x0: np.ndarray,
        t_final: float,
        dt: float,
        method: str = 'rk4',
    ) -> ParticleTrajectory:
        """
        Integrate particle trajectory.
        
        Parameters
        ----------
        x0 : np.ndarray
            Initial position (x, y, z).
        t_final : float
            Final time [s].
        dt : float
            Time step [s].
        method : str
            Integration method ('euler' or 'rk4').
        
        Returns
        -------
        trajectory : ParticleTrajectory
            Particle trajectory.
        """
        n_steps = int(t_final / dt) + 1
        t = np.linspace(0, t_final, n_steps)
        
        positions = np.zeros((n_steps, 3))
        positions[0] = x0
        
        x = x0.copy()
        
        for i in range(1, n_steps):
            if method == 'euler':
                v = self.velocity(x)
                x = x + dt * v
            elif method == 'rk4':
                k1 = self.velocity(x)
                k2 = self.velocity(x + 0.5 * dt * k1)
                k3 = self.velocity(x + 0.5 * dt * k2)
                k4 = self.velocity(x + dt * k3)
                x = x + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
            
            # Clamp to domain
            x[0] = np.clip(x[0], self.gorkov.x[0], self.gorkov.x[-1])
            x[1] = np.clip(x[1], self.gorkov.y[0], self.gorkov.y[-1])
            x[2] = np.clip(x[2], self.gorkov.z[0], self.gorkov.z[-1])
            
            positions[i] = x
        
        return ParticleTrajectory(
            t=t,
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            x0=x0,
        )
    
    def integrate_ensemble(
        self,
        initial_positions: np.ndarray,
        t_final: float,
        dt: float,
        method: str = 'rk4',
    ) -> List[ParticleTrajectory]:
        """
        Integrate trajectories for multiple particles.
        
        Parameters
        ----------
        initial_positions : np.ndarray
            (N, 3) array of initial positions.
        t_final : float
            Final time [s].
        dt : float
            Time step [s].
        method : str
            Integration method.
        
        Returns
        -------
        trajectories : list of ParticleTrajectory
            Trajectories for each particle.
        """
        trajectories = []
        for x0 in initial_positions:
            traj = self.integrate(x0, t_final, dt, method)
            trajectories.append(traj)
        return trajectories


def generate_random_initial_positions(
    mesh: FEMMesh,
    n_particles: int,
    domain: DomainType = DomainType.WATER,
    margin: float = 0.1,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate random initial particle positions within a domain.
    
    Parameters
    ----------
    mesh : FEMMesh
        Finite element mesh.
    n_particles : int
        Number of particles.
    domain : DomainType
        Domain to place particles in.
    margin : float
        Fraction of domain to exclude at boundaries.
    seed : int, optional
        Random seed for reproducibility.
    
    Returns
    -------
    positions : np.ndarray
        (n_particles, 3) array of positions.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Get domain extent
    if domain in mesh.domain_info:
        elem_ids = mesh.domain_info[domain].element_ids
        node_ids = mesh.elements[elem_ids].flatten()
        coords = mesh.nodes[np.unique(node_ids)]
        
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        z_min, z_max = coords[:, 2].min(), coords[:, 2].max()
    else:
        # Default to full domain
        x_min, x_max = mesh.x[0], mesh.x[-1]
        y_min, y_max = mesh.y[0], mesh.y[-1]
        z_min, z_max = mesh.z[0], mesh.z[-1]
    
    # Apply margin
    dx = (x_max - x_min) * margin
    dy = (y_max - y_min) * margin
    dz = (z_max - z_min) * margin
    
    x_min += dx
    x_max -= dx
    y_min += dy
    y_max -= dy
    z_min += dz
    z_max -= dz
    
    # Generate random positions
    positions = np.zeros((n_particles, 3))
    positions[:, 0] = np.random.uniform(x_min, x_max, n_particles)
    positions[:, 1] = np.random.uniform(y_min, y_max, n_particles)
    positions[:, 2] = np.random.uniform(z_min, z_max, n_particles)
    
    return positions
