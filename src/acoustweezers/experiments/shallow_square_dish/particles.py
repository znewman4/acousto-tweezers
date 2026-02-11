"""
Particle Dynamics for Shallow Square Dish.

Implements overdamped particle motion with:
- Gor'kov radiation force (from acoustic field)
- Stokes drag
- Advection by acoustic streaming

Equation of motion:
    ẋ = u_s(x) + μ F_rad(x)
    
where μ = 1/(6πηa) is the Stokes mobility.

Author: Acousto-Tweezers Project
Date: 2026-02-08
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, List, Optional, Callable
from dataclasses import dataclass, field

from dolfinx import fem, mesh
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
import ufl

from .config import ShallowDishConfig
from .solve_pressure import PressureSolution
from .streaming import StreamingSolution


@dataclass
class GorkovField:
    """Container for Gor'kov potential and radiation force."""
    U_function: fem.Function       # Gor'kov potential U
    F_function: Optional[fem.Function] = None  # Radiation force F = -∇U
    cfg: Optional[ShallowDishConfig] = None
    
    @property
    def mesh(self):
        return self.U_function.function_space.mesh
    
    @property
    def U_values(self) -> np.ndarray:
        return np.real(self.U_function.x.array)
    
    @property
    def coords(self) -> np.ndarray:
        return self.U_function.function_space.tabulate_dof_coordinates()
    
    @property
    def trap_depth(self) -> float:
        """Trap depth (max - min of potential)."""
        U = self.U_values
        return np.max(U) - np.min(U)
    
    @property
    def F_values(self) -> Optional[np.ndarray]:
        """Radiation force at DOFs, shape (N, 3)."""
        if self.F_function is None:
            return None
        vals = self.F_function.x.array.copy()
        n = len(vals) // 3
        return vals.reshape((n, 3))


@dataclass
class ParticleTrajectory:
    """Container for particle trajectory data with full diagnostics."""
    t: np.ndarray              # Time array [s]
    x: np.ndarray              # x-coordinates [m]
    y: np.ndarray              # y-coordinates [m]
    z: np.ndarray              # z-coordinates [m]
    x0: np.ndarray             # Initial position [m]
    
    # Full physics diagnostics
    U: Optional[np.ndarray] = None              # Gor'kov potential [J]
    F_rad_mag: Optional[np.ndarray] = None      # |F_rad| [N]
    u_stream_mag: Optional[np.ndarray] = None   # |u_s| [m/s]
    chi: Optional[np.ndarray] = None            # χ = |u_s| / (|F|/(6πηa))
    dist_to_min: Optional[np.ndarray] = None    # Distance to nearest potential minimum [m]
    
    # Optional velocity components
    vx: Optional[np.ndarray] = None
    vy: Optional[np.ndarray] = None
    vz: Optional[np.ndarray] = None
    
    @property
    def positions(self) -> np.ndarray:
        """Position array (N, 3)."""
        return np.column_stack([self.x, self.y, self.z])
    
    @property
    def final_position(self) -> np.ndarray:
        return np.array([self.x[-1], self.y[-1], self.z[-1]])
    
    @property
    def displacement(self) -> float:
        """Total displacement from start."""
        return np.linalg.norm(self.final_position - self.x0)
    
    @property
    def path_length(self) -> float:
        """Total path length traveled."""
        dr = np.diff(self.positions, axis=0)
        return np.sum(np.linalg.norm(dr, axis=1))
    
    def to_dict(self) -> dict:
        """Convert to dictionary for CSV export with full diagnostics."""
        data = {
            't_s': self.t,
            'x_m': self.x,
            'y_m': self.y,
            'z_m': self.z,
        }
        
        if self.U is not None:
            data['U_J'] = self.U
        if self.F_rad_mag is not None:
            data['F_rad_mag_N'] = self.F_rad_mag
        if self.u_stream_mag is not None:
            data['u_stream_mag_m_per_s'] = self.u_stream_mag
        if self.chi is not None:
            data['chi'] = self.chi
        if self.dist_to_min is not None:
            data['dist_to_min_m'] = self.dist_to_min
            
        return data


def compute_gorkov_potential(
    p_solution: PressureSolution,
    use_velocity_term: bool = True,
    verbose: bool = True,
) -> GorkovField:
    """
    Compute Gor'kov potential from pressure field.
    
    U = (4π/3)a³ [f₁·⟨p²⟩/(2K) - f₂·(3ρ/4)·⟨v²⟩]
    
    where:
    - ⟨p²⟩ = |p|²/2 (time-averaged pressure squared)
    - ⟨v²⟩ = |v₁|²/2 (time-averaged velocity squared)
    - K = ρc² (bulk modulus)
    
    Parameters
    ----------
    p_solution : PressureSolution
        Pressure solution
    use_velocity_term : bool
        If True, include dipole term with proper velocity gradient
        If False, use plane-wave approximation
    verbose : bool
        Print info
        
    Returns
    -------
    GorkovField
        Gor'kov potential and force
    """
    cfg = p_solution.cfg
    rho = cfg.rho
    c = cfg.c
    omega = cfg.omega
    K = cfg.fluid_bulk_modulus
    kappa_f = cfg.fluid_compressibility
    
    a = cfg.particle_radius
    f1 = cfg.f1_monopole
    f2 = cfg.f2_dipole
    V_p = cfg.particle_volume
    
    if verbose:
        print(f"\n{'='*70}")
        print("COMPUTING GOR'KOV POTENTIAL")
        print(f"{'='*70}")
        print(f"  Particle: a = {a*1e6:.1f} μm, f₁ = {f1:.3f}, f₂ = {f2:.3f}")
    
    # Get pressure
    V = p_solution.p_function.function_space
    p_vals = p_solution.p_values
    
    # Time-averaged pressure squared: ⟨p²⟩ = |p|²/2
    p2_avg = np.abs(p_vals)**2 / 2
    
    # Velocity squared
    if use_velocity_term:
        # Compute |v₁|² = |∇p|²/(ω²ρ²)
        # Use UFL for gradient computation
        p_func = p_solution.p_function
        domain = V.mesh
        
        grad_p = ufl.grad(p_func)
        grad_p_mag_sq = ufl.inner(grad_p, ufl.conj(grad_p))
        
        # Project to scalar space
        grad_p_sq_func = fem.Function(V)
        grad_p_sq_expr = fem.Expression(
            ufl.real(grad_p_mag_sq), 
            V.element.interpolation_points()
        )
        grad_p_sq_func.interpolate(grad_p_sq_expr)
        
        # |v₁|² = |∇p|²/(ω²ρ²)
        grad_p_sq = grad_p_sq_func.x.array
        v2 = grad_p_sq / (omega**2 * rho**2)
        v2_avg = v2 / 2  # time average
    else:
        # Plane-wave approximation: |v₁|² ≈ |p|²/(ρ²c²)
        v2_avg = p2_avg / (rho**2 * c**2)
    
    # Gor'kov potential
    # U = V_p [f₁·⟨p²⟩/(2K) - f₂·(3ρ/4)·⟨v²⟩]
    U_vals = V_p * (f1 * p2_avg / (2 * K) - f2 * (3 * rho / 4) * v2_avg)
    
    # Create function
    U_func = fem.Function(V)
    U_func.x.array[:] = np.real(U_vals)
    U_func.name = "gorkov_potential"
    
    if verbose:
        print(f"  Gor'kov potential:")
        print(f"    min U = {np.min(U_vals):.2e} J")
        print(f"    max U = {np.max(U_vals):.2e} J")
        print(f"    trap depth = {np.max(U_vals) - np.min(U_vals):.2e} J")
    
    # Compute radiation force F = -∇U
    # Use vector function space
    domain = V.mesh
    V_vec = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    
    # F = -∇U
    grad_U = ufl.grad(U_func)
    F_expr = fem.Expression(-grad_U, V_vec.element.interpolation_points())
    
    F_func = fem.Function(V_vec)
    F_func.interpolate(F_expr)
    F_func.name = "radiation_force"
    
    if verbose:
        F_vals = F_func.x.array.reshape((-1, 3))
        F_mag = np.linalg.norm(F_vals, axis=1)
        print(f"  Radiation force:")
        print(f"    max |F| = {np.max(F_mag):.2e} N")
        print(f"    mean |F| = {np.mean(F_mag):.2e} N")
        print(f"{'='*70}\n")
    
    return GorkovField(
        U_function=U_func,
        F_function=F_func,
        cfg=cfg,
    )


class ParticleDynamics:
    """
    Particle trajectory integration.
    
    Overdamped equation of motion:
        ẋ = u_s(x) + μ F_rad(x)
    """
    
    def __init__(
        self,
        gorkov: GorkovField,
        streaming: Optional[StreamingSolution] = None,
        cfg: Optional[ShallowDishConfig] = None,
    ):
        """
        Initialize particle dynamics.
        
        Parameters
        ----------
        gorkov : GorkovField
            Gor'kov potential and force
        streaming : StreamingSolution, optional
            Streaming velocity field
        cfg : ShallowDishConfig, optional
            Configuration (defaults to gorkov.cfg)
        """
        self.gorkov = gorkov
        self.streaming = streaming
        self.cfg = cfg or gorkov.cfg
        
        # Stokes mobility
        self.mu = self.cfg.stokes_mobility
        
        # Setup geometry for point queries
        self._setup_geometry()
    
    def _setup_geometry(self):
        """Setup bounding box tree for point location."""
        self.mesh = self.gorkov.mesh
        self.tree = bb_tree(self.mesh, self.mesh.topology.dim)
        
        # Get domain bounds
        coords = self.gorkov.coords
        self.x_min = np.min(coords[:, 0])
        self.x_max = np.max(coords[:, 0])
        self.y_min = np.min(coords[:, 1])
        self.y_max = np.max(coords[:, 1])
        self.z_min = np.min(coords[:, 2])
        self.z_max = np.max(coords[:, 2])
    
    def _is_in_domain(self, pos: np.ndarray) -> bool:
        """Check if position is inside domain."""
        margin = 1e-9
        return (
            self.x_min + margin <= pos[0] <= self.x_max - margin and
            self.y_min + margin <= pos[1] <= self.y_max - margin and
            self.z_min + margin <= pos[2] <= self.z_max - margin
        )
    
    def _eval_force(self, pos: np.ndarray) -> np.ndarray:
        """Evaluate radiation force at position."""
        if self.gorkov.F_function is None:
            return np.zeros(3)
        
        # Ensure proper shape and type for DOLFINx geometry functions
        pos_2d = np.ascontiguousarray(pos.reshape(1, 3), dtype=np.float64)
        cells = compute_collisions_points(self.tree, pos_2d)
        colliding = compute_colliding_cells(self.mesh, cells, pos_2d)
        
        if len(colliding.links(0)) > 0:
            cell = colliding.links(0)[0]
            try:
                return self.gorkov.F_function.eval(pos, cell)
            except:
                return np.zeros(3)
        return np.zeros(3)
    
    def _eval_streaming(self, pos: np.ndarray) -> np.ndarray:
        """Evaluate streaming velocity at position."""
        if self.streaming is None:
            return np.zeros(3)
        
        # Ensure proper shape and type for DOLFINx geometry functions
        pos_2d = np.ascontiguousarray(pos.reshape(1, 3), dtype=np.float64)
        cells = compute_collisions_points(self.tree, pos_2d)
        colliding = compute_colliding_cells(self.mesh, cells, pos_2d)
        
        if len(colliding.links(0)) > 0:
            cell = colliding.links(0)[0]
            try:
                return self.streaming.u_function.eval(pos, cell)
            except:
                return np.zeros(3)
        return np.zeros(3)
    
    def _eval_gorkov_potential(self, pos: np.ndarray) -> float:
        """Evaluate Gor'kov potential at position."""
        # Ensure proper shape and type for DOLFINx geometry functions
        pos_2d = np.ascontiguousarray(pos.reshape(1, 3), dtype=np.float64)
        cells = compute_collisions_points(self.tree, pos_2d)
        colliding = compute_colliding_cells(self.mesh, cells, pos_2d)
        
        if len(colliding.links(0)) > 0:
            cell = colliding.links(0)[0]
            try:
                return self.gorkov.U_function.eval(pos, cell)[0]
            except:
                return 0.0
        return 0.0
    
    def find_nearest_minimum(self, pos: np.ndarray, search_radius: float = 2e-3) -> Tuple[np.ndarray, float]:
        """
        Find nearest local minimum of Gor'kov potential.
        
        Parameters
        ----------
        pos : np.ndarray
            Current position (3,)
        search_radius : float
            Search radius [m]
            
        Returns
        -------
        min_pos : np.ndarray
            Position of nearest minimum (3,)
        min_U : float
            Potential at minimum [J]
        """
        # Sample potential on a grid around current position
        n_samples = 10
        x_range = np.linspace(max(self.x_min, pos[0] - search_radius),
                              min(self.x_max, pos[0] + search_radius), n_samples)
        y_range = np.linspace(max(self.y_min, pos[1] - search_radius),
                              min(self.y_max, pos[1] + search_radius), n_samples)
        z_range = np.linspace(max(self.z_min, pos[2] - search_radius),
                              min(self.z_max, pos[2] + search_radius), n_samples)
        
        min_U = np.inf
        min_pos = pos.copy()
        
        for x in x_range:
            for y in y_range:
                for z in z_range:
                    test_pos = np.array([x, y, z])
                    U = self._eval_gorkov_potential(test_pos)
                    if U < min_U:
                        min_U = U
                        min_pos = test_pos
        
        return min_pos, min_U
    
    def velocity(self, pos: np.ndarray) -> np.ndarray:
        """
        Compute particle velocity at position.
        
        ẋ = u_s(x) + μ F_rad(x)
        """
        F_rad = self._eval_force(pos)
        u_s = self._eval_streaming(pos)
        return u_s + self.mu * F_rad
    
    def integrate(
        self,
        x0: np.ndarray,
        t_max: float = None,
        dt: float = None,
        method: str = "rk2",
        stop_at_boundary: bool = True,
        track_diagnostics: bool = True,
    ) -> ParticleTrajectory:
        """
        Integrate particle trajectory with full diagnostics.
        
        Parameters
        ----------
        x0 : np.ndarray
            Initial position (3,)
        t_max : float
            Maximum integration time [s]
        dt : float
            Time step [s]
        method : str
            Integration method: "euler", "rk2", "rk4"
        stop_at_boundary : bool
            Stop if particle exits domain
        track_diagnostics : bool
            Track U(t), |F|(t), |u_s|(t), chi(t), dist_to_min(t)
            
        Returns
        -------
        ParticleTrajectory
            Integrated trajectory with diagnostics
        """
        if t_max is None:
            t_max = self.cfg.particle_t_max
        if dt is None:
            dt = self.cfg.particle_dt
        
        n_steps = int(t_max / dt)
        
        t = np.zeros(n_steps + 1)
        x = np.zeros(n_steps + 1)
        y = np.zeros(n_steps + 1)
        z = np.zeros(n_steps + 1)
        
        # Diagnostic arrays
        if track_diagnostics:
            U_arr = np.zeros(n_steps + 1)
            F_rad_mag_arr = np.zeros(n_steps + 1)
            u_stream_mag_arr = np.zeros(n_steps + 1)
            chi_arr = np.zeros(n_steps + 1)
            dist_to_min_arr = np.zeros(n_steps + 1)
            
            # Find global minimum once (for distance tracking)
            min_pos, _ = self.find_nearest_minimum(x0, search_radius=self.cfg.L)
        
        pos = np.array(x0, dtype=np.float64)
        x[0], y[0], z[0] = pos
        
        # Initial diagnostics
        if track_diagnostics:
            U_arr[0] = self._eval_gorkov_potential(pos)
            F_rad = self._eval_force(pos)
            F_rad_mag_arr[0] = np.linalg.norm(F_rad)
            u_s = self._eval_streaming(pos)
            u_stream_mag_arr[0] = np.linalg.norm(u_s)
            
            # χ = |u_s| / (|F|/(6πηa))
            if F_rad_mag_arr[0] > 1e-20:
                chi_arr[0] = u_stream_mag_arr[0] / (F_rad_mag_arr[0] * self.mu)
            else:
                chi_arr[0] = np.inf
            
            dist_to_min_arr[0] = np.linalg.norm(pos - min_pos)
        
        for i in range(n_steps):
            t[i] = i * dt
            
            # Check domain bounds
            if stop_at_boundary and not self._is_in_domain(pos):
                # Truncate arrays
                t = t[:i+1]
                x = x[:i+1]
                y = y[:i+1]
                z = z[:i+1]
                if track_diagnostics:
                    U_arr = U_arr[:i+1]
                    F_rad_mag_arr = F_rad_mag_arr[:i+1]
                    u_stream_mag_arr = u_stream_mag_arr[:i+1]
                    chi_arr = chi_arr[:i+1]
                    dist_to_min_arr = dist_to_min_arr[:i+1]
                break
            
            # Integration step
            if method == "euler":
                v = self.velocity(pos)
                pos = pos + v * dt
            
            elif method == "rk2":
                k1 = self.velocity(pos)
                k2 = self.velocity(pos + 0.5 * dt * k1)
                pos = pos + dt * k2
            
            elif method == "rk4":
                k1 = self.velocity(pos)
                k2 = self.velocity(pos + 0.5 * dt * k1)
                k3 = self.velocity(pos + 0.5 * dt * k2)
                k4 = self.velocity(pos + dt * k3)
                pos = pos + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
            
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Clamp to domain
            pos[0] = np.clip(pos[0], self.x_min, self.x_max)
            pos[1] = np.clip(pos[1], self.y_min, self.y_max)
            pos[2] = np.clip(pos[2], self.z_min, self.z_max)
            
            x[i+1], y[i+1], z[i+1] = pos
            
            # Update diagnostics
            if track_diagnostics:
                U_arr[i+1] = self._eval_gorkov_potential(pos)
                F_rad = self._eval_force(pos)
                F_rad_mag_arr[i+1] = np.linalg.norm(F_rad)
                u_s = self._eval_streaming(pos)
                u_stream_mag_arr[i+1] = np.linalg.norm(u_s)
                
                # χ = |u_s| / (|F|/(6πηa))
                if F_rad_mag_arr[i+1] > 1e-20:
                    chi_arr[i+1] = u_stream_mag_arr[i+1] / (F_rad_mag_arr[i+1] * self.mu)
                else:
                    chi_arr[i+1] = np.inf
                
                dist_to_min_arr[i+1] = np.linalg.norm(pos - min_pos)
        
        t[-1] = len(t) * dt
        
        return ParticleTrajectory(
            t=t,
            x=x,
            y=y,
            z=z,
            x0=np.array(x0),
            U=U_arr if track_diagnostics else None,
            F_rad_mag=F_rad_mag_arr if track_diagnostics else None,
            u_stream_mag=u_stream_mag_arr if track_diagnostics else None,
            chi=chi_arr if track_diagnostics else None,
            dist_to_min=dist_to_min_arr if track_diagnostics else None,
        )
    
    def integrate_ensemble(
        self,
        initial_positions: np.ndarray,
        t_max: float = None,
        dt: float = None,
        method: str = "rk2",
        verbose: bool = True,
    ) -> List[ParticleTrajectory]:
        """
        Integrate multiple particle trajectories.
        
        Parameters
        ----------
        initial_positions : np.ndarray
            Initial positions, shape (N, 3)
        t_max, dt : float
            Time parameters
        method : str
            Integration method
        verbose : bool
            Print progress
            
        Returns
        -------
        List[ParticleTrajectory]
            List of trajectories
        """
        trajectories = []
        n = len(initial_positions)
        
        for i, x0 in enumerate(initial_positions):
            if verbose and (i % 10 == 0 or i == n - 1):
                print(f"  Integrating particle {i+1}/{n}")
            
            traj = self.integrate(x0, t_max, dt, method)
            trajectories.append(traj)
        
        return trajectories


def integrate_particle_trajectory(
    gorkov: GorkovField,
    streaming: Optional[StreamingSolution] = None,
    x0: np.ndarray = None,
    cfg: Optional[ShallowDishConfig] = None,
    verbose: bool = True,
) -> ParticleTrajectory:
    """
    Convenience function to integrate a single particle trajectory.
    
    Parameters
    ----------
    gorkov : GorkovField
        Gor'kov potential and force
    streaming : StreamingSolution, optional
        Streaming velocity
    x0 : np.ndarray, optional
        Initial position (defaults to center of domain)
    cfg : ShallowDishConfig, optional
        Configuration
    verbose : bool
        Print progress
        
    Returns
    -------
    ParticleTrajectory
        Integrated trajectory
    """
    cfg = cfg or gorkov.cfg
    
    if x0 is None:
        # Default: start near vortex core, slightly above bottom
        x0 = np.array([cfg.L/2, cfg.L/2, 0.1 * cfg.H])
    
    if verbose:
        print(f"\n{'='*70}")
        print("INTEGRATING PARTICLE TRAJECTORY")
        print(f"{'='*70}")
        print(f"  Initial position: ({x0[0]*1e3:.2f}, {x0[1]*1e3:.2f}, {x0[2]*1e3:.2f}) mm")
        print(f"  Mobility μ = {cfg.stokes_mobility:.2e} m/(N·s)")
        print(f"  t_max = {cfg.particle_t_max:.3f} s, dt = {cfg.particle_dt:.2e} s")
    
    dynamics = ParticleDynamics(gorkov, streaming, cfg)
    traj = dynamics.integrate(x0, method="rk2")
    
    if verbose:
        print(f"\n  Final position: ({traj.x[-1]*1e3:.2f}, {traj.y[-1]*1e3:.2f}, {traj.z[-1]*1e3:.2f}) mm")
        print(f"  Displacement: {traj.displacement*1e3:.2f} mm")
        print(f"  Path length: {traj.path_length*1e3:.2f} mm")
        print(f"  Steps: {len(traj.t)}")
        print(f"{'='*70}\n")
    
    return traj


def save_trajectory_csv(
    trajectory: ParticleTrajectory,
    filepath: str,
    particle_id: int = 0,
) -> None:
    """
    Save trajectory to CSV file.
    
    Format: particle_id, time, x_m, y_m, z_m
    """
    import csv
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['particle_id', 'time', 'x_m', 'y_m', 'z_m'])
        
        for i in range(len(trajectory.t)):
            writer.writerow([
                particle_id,
                trajectory.t[i],
                trajectory.x[i],
                trajectory.y[i],
                trajectory.z[i],
            ])


def save_trajectories_csv(
    trajectories: List[ParticleTrajectory],
    filepath: str,
) -> None:
    """
    Save multiple trajectories to CSV file.
    """
    import csv
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['particle_id', 'time', 'x_m', 'y_m', 'z_m'])
        
        for pid, traj in enumerate(trajectories):
            for i in range(len(traj.t)):
                writer.writerow([
                    pid,
                    traj.t[i],
                    traj.x[i],
                    traj.y[i],
                    traj.z[i],
                ])
