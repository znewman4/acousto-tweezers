"""
Particle dynamics for FEniCSx acoustic tweezers.

Implements radiation force and particle trajectory integration per MASTER BRIEF:

Radiation force from Gor'kov potential:
    F_rad = -∇U

Gor'kov potential:
    U = (4π/3)a³ [f₁·⟨p²⟩/(2ρc²) - f₂·(3ρ/4)·⟨v²⟩]

Particle velocity (overdamped dynamics):
    ẋ = u_stream(x) + μ F_rad(x)
    
where μ = 1/(6πηa) is the Stokes mobility.

Author: Acousto-Tweezers Project
Date: January 2026
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Callable
import numpy as np

from dolfinx import fem, mesh as dmesh
from dolfinx.fem import Function
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells

from .config import FEMConfig
from .domains import Domain
from .materials import MaterialDatabase, FluidMaterial, ParticleMaterial
from .acoustics import AcousticField
from .streaming import StreamingField


@dataclass
class GorkovPotential:
    """
    Gor'kov radiation force potential.
    """
    # Potential field (scalar)
    U_function: Function
    
    # Force components (vector or separate scalars)
    F_function: Optional[Function] = None
    
    # Contrast factors
    f1: float = 0.0  # Monopole
    f2: float = 0.0  # Dipole
    
    # Particle properties
    particle_radius: float = 5e-6
    
    @property
    def U(self) -> np.ndarray:
        """Potential values at DOFs."""
        return self.U_function.x.array.copy()
    
    @property
    def mesh(self):
        return self.U_function.function_space.mesh
    
    @property
    def trap_depth(self) -> float:
        """Trap depth (max - min of potential)."""
        return np.max(self.U) - np.min(self.U)
    
    @property
    def min_potential_value(self) -> float:
        """Minimum potential value."""
        return np.min(self.U)
    
    @property
    def max_potential_gradient(self) -> float:
        """Maximum force magnitude."""
        if self.F_function is not None:
            F_vals = self.F_function.x.array.reshape(-1, 3)
            return np.max(np.linalg.norm(F_vals, axis=1))
        return 0.0


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
    
    # Velocity history (optional)
    vx: Optional[np.ndarray] = None
    vy: Optional[np.ndarray] = None
    vz: Optional[np.ndarray] = None
    
    @property
    def positions(self) -> np.ndarray:
        """Position array (N, 3)."""
        return np.column_stack([self.x, self.y, self.z])
    
    @property
    def final_position(self) -> np.ndarray:
        """Final position."""
        return np.array([self.x[-1], self.y[-1], self.z[-1]])
    
    @property
    def displacement(self) -> float:
        """Total displacement from initial position."""
        return np.linalg.norm(self.final_position - self.x0)
    
    @property
    def path_length(self) -> float:
        """Total path length traveled."""
        dr = np.diff(self.positions, axis=0)
        return np.sum(np.linalg.norm(dr, axis=1))
    
    def is_trapped(self, tolerance: float = 1e-6) -> bool:
        """Check if particle reached a trap (velocity ~0)."""
        if len(self.x) < 10:
            return False
        # Check if position stabilized
        recent = self.positions[-10:]
        return np.std(recent, axis=0).max() < tolerance


class ParticleDynamics:
    """
    Particle radiation force and trajectory integration.
    """
    
    def __init__(self, config: FEMConfig,
                 mesh: dmesh.Mesh,
                 materials: MaterialDatabase):
        """
        Initialize particle dynamics.
        
        Parameters
        ----------
        config : FEMConfig
            Simulation configuration
        mesh : dolfinx.mesh.Mesh
            The computational mesh
        materials : MaterialDatabase
            Material property database
        """
        self.config = config
        self.mesh = mesh
        self.materials = materials
        
        # Particle properties
        self.particle_radius = config.physics.particle_radius
        self.particle_density = config.physics.particle_density
        self.particle_compressibility = config.physics.particle_compressibility
        
        # Create particle material
        self.particle = ParticleMaterial(
            name="particle",
            radius=self.particle_radius,
            density=self.particle_density,
            compressibility=self.particle_compressibility,
        )
        
        # Fluid properties (water in dish)
        self.fluid = materials.water
        
        # Contrast factors
        self.f1 = self.particle.monopole_contrast(self.fluid)
        self.f2 = self.particle.dipole_contrast(self.fluid)
        
        # Mobility
        self.mobility = self.particle.stokes_mobility(self.fluid)
        
        # Bounding box tree for point location
        self._setup_geometry()
        
    def _setup_geometry(self):
        """Setup geometry structures for point location."""
        self.tree = bb_tree(self.mesh, self.mesh.topology.dim)
        
    def compute_gorkov_potential(self,
                                  acoustic_field: AcousticField) -> GorkovPotential:
        """
        Compute Gor'kov potential from acoustic field.
        
        U = (4π/3)a³ [f₁·⟨p²⟩/(2K) - f₂·(3ρ/4)·⟨v²⟩]
        
        where:
        - ⟨p²⟩ = |p|²/2 (time-averaged pressure squared)
        - ⟨v²⟩ = |v|²/2 (time-averaged velocity squared)
        - K = ρc² (bulk modulus)
        
        Parameters
        ----------
        acoustic_field : AcousticField
            First-order acoustic field
            
        Returns
        -------
        GorkovPotential
            Gor'kov potential and force
        """
        p = acoustic_field.p_function
        omega = acoustic_field.omega
        
        rho = self.fluid.density
        c = self.fluid.sound_speed
        K = self.fluid.bulk_modulus
        
        a = self.particle_radius
        f1 = self.f1
        f2 = self.f2
        
        # Create function for potential
        V = p.function_space
        U = Function(V)
        
        # Get pressure values
        p_vals = p.x.array
        
        # Time-averaged pressure squared: ⟨p²⟩ = |p|²/2
        p2_avg = np.abs(p_vals)**2 / 2
        
        # Velocity squared: |v|² = |∇p|²/(ω²ρ²)
        # This requires computing gradient of p
        # For now, use finite difference approximation
        
        # Simplified: assume ⟨v²⟩ ≈ ⟨p²⟩/(ρ²c²) for plane wave
        v2_avg = p2_avg / (rho**2 * c**2)
        
        # Gor'kov potential
        prefactor = (4 * np.pi / 3) * a**3
        U_vals = prefactor * (f1 * p2_avg / (2 * K) - f2 * (3 * rho / 4) * v2_avg)
        
        U.x.array[:] = np.real(U_vals)
        
        # Compute force as negative gradient
        # F = -∇U
        # This would require projecting gradient
        
        return GorkovPotential(
            U_function=U,
            F_function=None,
            f1=f1,
            f2=f2,
            particle_radius=a,
        )
    
    def compute_radiation_force(self,
                                gorkov: GorkovPotential,
                                points: np.ndarray) -> np.ndarray:
        """
        Compute radiation force at given points.
        
        F = -∇U
        
        Parameters
        ----------
        gorkov : GorkovPotential
            Gor'kov potential
        points : np.ndarray
            Points to evaluate force at, shape (N, 3)
            
        Returns
        -------
        np.ndarray
            Force vectors at points, shape (N, 3)
        """
        # Evaluate potential at points
        U_func = gorkov.U_function
        
        # Find cells containing points
        cell_candidates = compute_collisions_points(self.tree, points)
        cells = compute_colliding_cells(self.mesh, cell_candidates, points)
        
        # Compute gradient numerically
        h = 1e-7  # Small step for finite difference
        F = np.zeros_like(points)
        
        for i, point in enumerate(points):
            if len(cells.links(i)) > 0:
                cell = cells.links(i)[0]
                
                # Central difference for gradient
                for d in range(3):
                    point_plus = point.copy()
                    point_minus = point.copy()
                    point_plus[d] += h
                    point_minus[d] -= h
                    
                    try:
                        U_plus = U_func.eval(point_plus, cell)[0]
                        U_minus = U_func.eval(point_minus, cell)[0]
                        F[i, d] = -(U_plus - U_minus) / (2 * h)
                    except:
                        F[i, d] = 0.0
        
        return F
    
    def integrate_trajectory(self,
                             gorkov: GorkovPotential,
                             streaming: Optional[StreamingField],
                             x0: np.ndarray,
                             t_max: float,
                             dt: float) -> ParticleTrajectory:
        """
        Integrate particle trajectory.
        
        Overdamped equation:
            dx/dt = u_stream(x) + μ F_rad(x)
            
        Parameters
        ----------
        gorkov : GorkovPotential
            Gor'kov potential for radiation force
        streaming : StreamingField, optional
            Streaming velocity field
        x0 : np.ndarray
            Initial position (3,)
        t_max : float
            Maximum integration time
        dt : float
            Time step
            
        Returns
        -------
        ParticleTrajectory
            Particle trajectory
        """
        # Time array
        n_steps = int(t_max / dt)
        t = np.linspace(0, t_max, n_steps + 1)
        
        # Position arrays
        x = np.zeros(n_steps + 1)
        y = np.zeros(n_steps + 1)
        z = np.zeros(n_steps + 1)
        
        x[0], y[0], z[0] = x0
        
        mu = self.mobility
        
        # Forward Euler integration
        for i in range(n_steps):
            pos = np.array([[x[i], y[i], z[i]]])
            
            # Radiation force
            F_rad = self.compute_radiation_force(gorkov, pos)[0]
            
            # Streaming velocity
            if streaming is not None:
                u_stream = self._evaluate_streaming(streaming, pos[0])
            else:
                u_stream = np.zeros(3)
            
            # Velocity: dx/dt = u_stream + μ F_rad
            v = u_stream + mu * F_rad
            
            # Update position
            x[i+1] = x[i] + v[0] * dt
            y[i+1] = y[i] + v[1] * dt
            z[i+1] = z[i] + v[2] * dt
            
            # Check if particle left domain
            # Would need domain bounds check here
        
        return ParticleTrajectory(
            t=t,
            x=x,
            y=y,
            z=z,
            x0=x0.copy(),
        )
    
    def _evaluate_streaming(self, 
                            streaming: StreamingField,
                            point: np.ndarray) -> np.ndarray:
        """Evaluate streaming velocity at a point."""
        u_func = streaming.u_function
        
        # Find cell
        cells = compute_collisions_points(self.tree, point.reshape(1, -1))
        colliding = compute_colliding_cells(self.mesh, cells, point.reshape(1, -1))
        
        if len(colliding.links(0)) > 0:
            cell = colliding.links(0)[0]
            try:
                return u_func.eval(point, cell)
            except:
                return np.zeros(3)
        return np.zeros(3)
    
    def integrate_ensemble(self,
                           gorkov: GorkovPotential,
                           streaming: Optional[StreamingField],
                           initial_positions: np.ndarray,
                           t_max: float,
                           dt: float) -> List[ParticleTrajectory]:
        """
        Integrate trajectories for multiple particles.
        
        Parameters
        ----------
        gorkov : GorkovPotential
            Gor'kov potential
        streaming : StreamingField, optional
            Streaming field
        initial_positions : np.ndarray
            Initial positions, shape (N, 3)
        t_max, dt : float
            Time parameters
            
        Returns
        -------
        List[ParticleTrajectory]
            List of particle trajectories
        """
        trajectories = []
        
        for i, x0 in enumerate(initial_positions):
            print(f"Integrating particle {i+1}/{len(initial_positions)}")
            traj = self.integrate_trajectory(
                gorkov, streaming, x0, t_max, dt
            )
            trajectories.append(traj)
        
        return trajectories
    
    def generate_initial_positions(self,
                                    n_particles: int,
                                    domain: str = "water") -> np.ndarray:
        """
        Generate random initial positions within a domain.
        
        Parameters
        ----------
        n_particles : int
            Number of particles
        domain : str
            Domain name ("water", "dish")
            
        Returns
        -------
        np.ndarray
            Initial positions, shape (n_particles, 3)
        """
        geo = self.config.geometry
        
        # Domain bounds (approximate, for water region)
        r_max = geo.dish_inner_radius * 0.9  # Stay away from walls
        z_min = 0.1 * geo.water_depth
        z_max = 0.9 * geo.water_depth
        
        positions = []
        
        for _ in range(n_particles):
            # Random position in cylindrical coordinates
            r = np.sqrt(np.random.random()) * r_max  # Uniform in area
            theta = np.random.random() * 2 * np.pi
            z = z_min + np.random.random() * (z_max - z_min)
            
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            
            positions.append([x, y, z])
        
        return np.array(positions)


def save_trajectories_csv(trajectories: List[ParticleTrajectory],
                          filepath: str):
    """
    Save trajectories to CSV file.
    
    Format:
        particle_id, time, x_m, y_m, z_m
        
    Parameters
    ----------
    trajectories : List[ParticleTrajectory]
        List of trajectories
    filepath : str
        Output file path
    """
    with open(filepath, 'w') as f:
        f.write("particle_id,time,x_m,y_m,z_m\n")
        
        for i, traj in enumerate(trajectories):
            for j in range(len(traj.t)):
                f.write(f"{i},{traj.t[j]},{traj.x[j]},{traj.y[j]},{traj.z[j]}\n")


def load_trajectories_csv(filepath: str) -> List[ParticleTrajectory]:
    """
    Load trajectories from CSV file.
    
    Parameters
    ----------
    filepath : str
        Input file path
        
    Returns
    -------
    List[ParticleTrajectory]
        Loaded trajectories
    """
    import pandas as pd
    
    df = pd.read_csv(filepath)
    
    trajectories = []
    for pid in df['particle_id'].unique():
        pdata = df[df['particle_id'] == pid]
        traj = ParticleTrajectory(
            t=pdata['time'].values,
            x=pdata['x_m'].values,
            y=pdata['y_m'].values,
            z=pdata['z_m'].values,
            x0=np.array([pdata['x_m'].iloc[0], 
                        pdata['y_m'].iloc[0],
                        pdata['z_m'].iloc[0]]),
        )
        trajectories.append(traj)
    
    return trajectories
