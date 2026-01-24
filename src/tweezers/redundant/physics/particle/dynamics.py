"""
Particle dynamics for acoustic manipulation.

Implements particle trajectory integration including:
- Acoustic radiation force
- Acoustic streaming drag
- Stokes drag
- Brownian motion (optional)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, Callable, List
import numpy as np
from scipy.integrate import solve_ivp

from .properties import Particle3D, compute_contrast_factors
from .interpolation import Grid3D, TrilinearInterpolator, VectorFieldInterpolator
from .gorkov import GorkovPotential3D
from ..acoustics.materials import FluidMaterial


@dataclass
class ParticleState:
    """
    Particle state at a given time.
    
    Attributes
    ----------
    time : float
        Current time [s].
    position : np.ndarray
        Position (x, y, z) [m].
    velocity : np.ndarray
        Velocity (vx, vy, vz) [m/s].
    """
    time: float
    position: np.ndarray
    velocity: np.ndarray


@dataclass
class ParticleTrajectory:
    """
    Complete particle trajectory.
    
    Attributes
    ----------
    times : np.ndarray
        Time points [s].
    positions : np.ndarray
        Position history, shape (n_times, 3) [m].
    velocities : np.ndarray
        Velocity history, shape (n_times, 3) [m/s].
    forces : np.ndarray
        Force history, shape (n_times, 3) [N].
    """
    times: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    forces: np.ndarray
    
    @property
    def duration(self) -> float:
        """Total trajectory duration [s]."""
        return self.times[-1] - self.times[0]
    
    @property
    def distance_traveled(self) -> float:
        """Total distance traveled [m]."""
        diffs = np.diff(self.positions, axis=0)
        return np.sum(np.linalg.norm(diffs, axis=1))
    
    def position_at(self, t: float) -> np.ndarray:
        """Interpolate position at time t."""
        return np.array([
            np.interp(t, self.times, self.positions[:, i])
            for i in range(3)
        ])
    
    def velocity_at(self, t: float) -> np.ndarray:
        """Interpolate velocity at time t."""
        return np.array([
            np.interp(t, self.times, self.velocities[:, i])
            for i in range(3)
        ])


class StokesianDynamics:
    """
    Stokesian dynamics for spherical particle.
    
    Equation of motion (overdamped limit):
        dx/dt = μ*F_rad + u_stream
    
    where μ = 1/(6πηa) is the Stokes mobility.
    
    Full equation (with inertia):
        m*dv/dt = F_rad - γ*(v - u_stream) + F_Brownian
    
    where γ = 6πηa is the drag coefficient.
    """
    
    def __init__(
        self,
        particle: Particle3D,
        fluid: FluidMaterial,
        include_inertia: bool = False,
        include_brownian: bool = False,
        temperature: float = 300.0,
    ):
        """
        Initialize dynamics.
        
        Parameters
        ----------
        particle : Particle3D
            Particle properties.
        fluid : FluidMaterial
            Fluid properties.
        include_inertia : bool
            Include inertial terms (ma) in dynamics.
        include_brownian : bool
            Include Brownian motion.
        temperature : float
            Temperature [K] for Brownian motion.
        """
        self.particle = particle
        self.fluid = fluid
        self.include_inertia = include_inertia
        self.include_brownian = include_brownian
        self.temperature = temperature
        
        # Compute drag coefficient and mobility
        self.gamma = 6.0 * np.pi * fluid.eta * particle.a  # Stokes drag [N·s/m]
        self.mobility = 1.0 / self.gamma  # [m/(N·s)]
        
        # Brownian diffusion coefficient
        if include_brownian:
            kB = 1.380649e-23  # Boltzmann constant [J/K]
            self.D = kB * temperature * self.mobility
        else:
            self.D = 0.0
    
    @property
    def relaxation_time(self) -> float:
        """
        Inertial relaxation time τ = m/γ.
        
        Time for particle to reach terminal velocity.
        """
        return self.particle.mass / self.gamma
    
    @property
    def diffusion_length_scale(self) -> float:
        """
        Characteristic diffusion length √(2Dτ).
        
        How far particle diffuses in one relaxation time.
        """
        if self.D > 0:
            return np.sqrt(2.0 * self.D * self.relaxation_time)
        return 0.0


class ParticleDynamics3D:
    """
    Complete particle dynamics with radiation force and streaming.
    
    Integrates particle trajectory under:
    - Acoustic radiation force from Gor'kov potential
    - Acoustic streaming velocity field
    - Stokes drag
    - Optional Brownian motion
    """
    
    def __init__(
        self,
        grid: Grid3D,
        gorkov: GorkovPotential3D,
        particle: Particle3D,
        fluid: FluidMaterial,
        streaming_velocity: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
        include_inertia: bool = False,
        include_brownian: bool = False,
        temperature: float = 300.0,
    ):
        """
        Initialize dynamics solver.
        
        Parameters
        ----------
        grid : Grid3D
            Spatial grid.
        gorkov : GorkovPotential3D
            Gor'kov potential for radiation force.
        particle : Particle3D
            Particle properties.
        fluid : FluidMaterial
            Fluid properties.
        streaming_velocity : tuple of 3 arrays, optional
            Streaming velocity field (ux, uy, uz).
        include_inertia : bool
            Include particle inertia.
        include_brownian : bool
            Include Brownian motion.
        temperature : float
            Temperature [K].
        """
        self.grid = grid
        self.gorkov = gorkov
        self.particle = particle
        self.fluid = fluid
        
        # Stokesian dynamics
        self.stokes = StokesianDynamics(
            particle, fluid,
            include_inertia=include_inertia,
            include_brownian=include_brownian,
            temperature=temperature,
        )
        
        # Precompute radiation force field and interpolators
        Fx, Fy, Fz = gorkov.compute_force(particle)
        self._force_interp = VectorFieldInterpolator(grid, Fx, Fy, Fz)
        
        # Streaming velocity interpolator
        if streaming_velocity is not None:
            self._streaming_interp = VectorFieldInterpolator(
                grid, *streaming_velocity
            )
        else:
            self._streaming_interp = None
        
        # Store for analysis
        self._force_field = (Fx, Fy, Fz)
    
    def force_at(self, position: np.ndarray) -> np.ndarray:
        """Get radiation force at position."""
        return self._force_interp(position.reshape(1, 3)).flatten()
    
    def streaming_at(self, position: np.ndarray) -> np.ndarray:
        """Get streaming velocity at position."""
        if self._streaming_interp is not None:
            return self._streaming_interp(position.reshape(1, 3)).flatten()
        return np.zeros(3)
    
    def _equations_of_motion_overdamped(
        self,
        t: float,
        state: np.ndarray,
    ) -> np.ndarray:
        """
        Overdamped equations of motion.
        
        dx/dt = μ*F_rad + u_stream
        """
        position = state[:3]
        
        F_rad = self.force_at(position)
        u_stream = self.streaming_at(position)
        
        dxdt = self.stokes.mobility * F_rad + u_stream
        
        return dxdt
    
    def _equations_of_motion_full(
        self,
        t: float,
        state: np.ndarray,
    ) -> np.ndarray:
        """
        Full equations of motion with inertia.
        
        dx/dt = v
        m*dv/dt = F_rad - γ*(v - u_stream)
        """
        position = state[:3]
        velocity = state[3:6]
        
        F_rad = self.force_at(position)
        u_stream = self.streaming_at(position)
        
        dxdt = velocity
        dvdt = (F_rad - self.stokes.gamma * (velocity - u_stream)) / self.particle.mass
        
        return np.concatenate([dxdt, dvdt])
    
    def simulate(
        self,
        initial_position: np.ndarray,
        duration: float,
        dt: float = 1e-4,
        initial_velocity: Optional[np.ndarray] = None,
        method: str = 'RK45',
        max_step: Optional[float] = None,
        events: Optional[List[Callable]] = None,
    ) -> ParticleTrajectory:
        """
        Simulate particle trajectory.
        
        Parameters
        ----------
        initial_position : np.ndarray
            Starting position (x, y, z) [m].
        duration : float
            Simulation duration [s].
        dt : float
            Output time step [s].
        initial_velocity : np.ndarray, optional
            Starting velocity. Default: zero for overdamped, equilibrium for full.
        method : str
            Integration method ('RK45', 'RK23', 'DOP853').
        max_step : float, optional
            Maximum integration step.
        events : list of callables, optional
            Event functions for termination.
        
        Returns
        -------
        trajectory : ParticleTrajectory
            Complete trajectory.
        """
        # Set up initial conditions
        if self.stokes.include_inertia:
            if initial_velocity is None:
                # Start at equilibrium velocity
                F0 = self.force_at(initial_position)
                u0 = self.streaming_at(initial_position)
                initial_velocity = self.stokes.mobility * F0 + u0
            y0 = np.concatenate([initial_position, initial_velocity])
            eom = self._equations_of_motion_full
        else:
            y0 = initial_position.copy()
            eom = self._equations_of_motion_overdamped
        
        # Time points
        t_span = (0.0, duration)
        t_eval = np.arange(0, duration + dt, dt)
        
        # Integrate
        if max_step is None:
            max_step = dt
        
        sol = solve_ivp(
            eom,
            t_span,
            y0,
            method=method,
            t_eval=t_eval,
            max_step=max_step,
            events=events,
        )
        
        # Extract results
        times = sol.t
        n_times = len(times)
        
        if self.stokes.include_inertia:
            positions = sol.y[:3, :].T
            velocities = sol.y[3:6, :].T
        else:
            positions = sol.y[:3, :].T
            # Compute velocities from positions
            velocities = np.zeros((n_times, 3))
            for i in range(n_times):
                velocities[i] = self._equations_of_motion_overdamped(times[i], positions[i])
        
        # Compute forces
        forces = np.zeros((n_times, 3))
        for i in range(n_times):
            forces[i] = self.force_at(positions[i])
        
        return ParticleTrajectory(
            times=times,
            positions=positions,
            velocities=velocities,
            forces=forces,
        )
    
    def estimate_equilibrium_time(self, position: np.ndarray) -> float:
        """
        Estimate time to reach equilibrium near a trap.
        
        Uses local potential curvature to estimate trapping time.
        """
        from .gorkov import compute_stiffness
        
        kappa = compute_stiffness(self.gorkov, self.particle, position)
        kappa_max = np.max(np.abs(kappa))
        
        if kappa_max > 0:
            # Overdamped oscillator time constant: τ = γ/κ
            return self.stokes.gamma / kappa_max
        else:
            return np.inf


def make_boundary_event(grid: Grid3D, margin: float = 0.0):
    """
    Create event function to stop simulation at boundary.
    
    Parameters
    ----------
    grid : Grid3D
        Spatial grid.
    margin : float
        Margin from boundary [m].
    
    Returns
    -------
    event : callable
        Event function for solve_ivp.
    """
    bounds = grid.bounds
    
    def boundary_event(t, state):
        x, y, z = state[:3]
        
        # Return negative when outside bounds
        dist = min(
            x - (bounds[0][0] + margin),
            (bounds[0][1] - margin) - x,
            y - (bounds[1][0] + margin),
            (bounds[1][1] - margin) - y,
            z - (bounds[2][0] + margin),
            (bounds[2][1] - margin) - z,
        )
        return dist
    
    boundary_event.terminal = True
    boundary_event.direction = -1
    
    return boundary_event


def simulate_multiple_particles(
    dynamics: ParticleDynamics3D,
    initial_positions: np.ndarray,
    duration: float,
    dt: float = 1e-4,
    **kwargs,
) -> List[ParticleTrajectory]:
    """
    Simulate multiple non-interacting particles.
    
    Parameters
    ----------
    dynamics : ParticleDynamics3D
        Dynamics solver.
    initial_positions : np.ndarray
        Initial positions, shape (n_particles, 3).
    duration : float
        Simulation duration [s].
    dt : float
        Output time step [s].
    **kwargs
        Additional arguments for simulate().
    
    Returns
    -------
    trajectories : list of ParticleTrajectory
        Trajectories for each particle.
    """
    n_particles = initial_positions.shape[0]
    trajectories = []
    
    for i in range(n_particles):
        traj = dynamics.simulate(
            initial_positions[i],
            duration,
            dt,
            **kwargs,
        )
        trajectories.append(traj)
    
    return trajectories


def compute_characteristic_velocity(
    particle: Particle3D,
    fluid: FluidMaterial,
    frequency: float,
    pressure_amplitude: float,
) -> float:
    """
    Estimate characteristic particle velocity scale.
    
    v_char = μ * F_max = F_max / (6πηa)
    
    Parameters
    ----------
    particle : Particle3D
        Particle properties.
    fluid : FluidMaterial
        Fluid properties.
    frequency : float
        Acoustic frequency [Hz].
    pressure_amplitude : float
        Pressure amplitude [Pa].
    
    Returns
    -------
    v_char : float
        Characteristic velocity [m/s].
    """
    from .gorkov import estimate_max_radiation_force
    
    F_max = estimate_max_radiation_force(
        particle, fluid, frequency, pressure_amplitude
    )
    mobility = 1.0 / (6.0 * np.pi * fluid.eta * particle.a)
    
    return mobility * F_max
