"""
Visualization tools for 3D multiphysics simulation.

Provides:
- 2D slice plots of pressure, velocity, potential
- 3D isosurface plots
- Particle trajectory animations
- Energy budget plots
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any
import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors
    from matplotlib.animation import FuncAnimation
    import matplotlib.gridspec as gridspec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False


def _check_matplotlib():
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization")


class MultiphysicsVisualizer:
    """
    Visualization for multiphysics simulation results.
    """
    
    def __init__(
        self,
        results: "MultiphysicsResults",  # Forward reference
        output_dir: Optional[Path] = None,
        dpi: int = 150,
    ):
        """
        Initialize visualizer.
        
        Parameters
        ----------
        results : MultiphysicsResults
            Simulation results from MultiphysicsSolver.
        output_dir : Path, optional
            Output directory for saved figures.
        dpi : int
            Resolution for saved figures.
        """
        _check_matplotlib()
        
        self.results = results
        self.output_dir = Path(output_dir) if output_dir else Path(".")
        self.dpi = dpi
        
        # Grid data
        self.x = results.geometry.grid_x
        self.y = results.geometry.grid_y
        self.z = results.geometry.grid_z
        
        # Convenience
        self.nx, self.ny, self.nz = len(self.x), len(self.y), len(self.z)
    
    def _save_fig(self, fig: plt.Figure, name: str) -> None:
        """Save figure to output directory."""
        path = self.output_dir / f"{name}.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches='tight')
        print(f"Saved: {path}")
    
    def plot_pressure_slices(
        self,
        z_idx: Optional[int] = None,
        y_idx: Optional[int] = None,
        x_idx: Optional[int] = None,
        save: bool = True,
    ) -> plt.Figure:
        """
        Plot pressure field slices.
        
        Parameters
        ----------
        z_idx : int, optional
            z-index for xy slice. Default: middle.
        y_idx : int, optional
            y-index for xz slice. Default: middle.
        x_idx : int, optional
            x-index for yz slice. Default: middle.
        save : bool
            Save figure to file.
        
        Returns
        -------
        fig : Figure
            Matplotlib figure.
        """
        if z_idx is None:
            z_idx = self.nz // 2
        if y_idx is None:
            y_idx = self.ny // 2
        if x_idx is None:
            x_idx = self.nx // 2
        
        pressure = self.results.acoustic_field.pressure
        p_mag = np.abs(pressure)
        vmax = np.max(p_mag)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # XY slice (top view)
        ax = axes[0]
        im = ax.pcolormesh(
            self.x * 1e3, self.y * 1e3, p_mag[:, :, z_idx].T,
            shading='auto', cmap='viridis', vmin=0, vmax=vmax
        )
        ax.set_xlabel('x [mm]')
        ax.set_ylabel('y [mm]')
        ax.set_title(f'|p| at z = {self.z[z_idx]*1e3:.2f} mm')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, label='|p| [Pa]')
        
        # XZ slice (side view)
        ax = axes[1]
        im = ax.pcolormesh(
            self.x * 1e3, self.z * 1e3, p_mag[:, y_idx, :].T,
            shading='auto', cmap='viridis', vmin=0, vmax=vmax
        )
        ax.set_xlabel('x [mm]')
        ax.set_ylabel('z [mm]')
        ax.set_title(f'|p| at y = {self.y[y_idx]*1e3:.2f} mm')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, label='|p| [Pa]')
        
        # YZ slice
        ax = axes[2]
        im = ax.pcolormesh(
            self.y * 1e3, self.z * 1e3, p_mag[x_idx, :, :].T,
            shading='auto', cmap='viridis', vmin=0, vmax=vmax
        )
        ax.set_xlabel('y [mm]')
        ax.set_ylabel('z [mm]')
        ax.set_title(f'|p| at x = {self.x[x_idx]*1e3:.2f} mm')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, label='|p| [Pa]')
        
        plt.tight_layout()
        
        if save:
            self._save_fig(fig, 'pressure_slices')
        
        return fig
    
    def plot_gorkov_potential(
        self,
        z_idx: Optional[int] = None,
        save: bool = True,
    ) -> plt.Figure:
        """
        Plot Gor'kov potential with minima (trap locations).
        
        Parameters
        ----------
        z_idx : int, optional
            z-index for slice. Default: middle of water domain.
        save : bool
            Save figure to file.
        
        Returns
        -------
        fig : Figure
            Matplotlib figure.
        """
        if self.results.gorkov_potential is None:
            raise ValueError("Gor'kov potential not computed")
        
        U = self.results.gorkov_potential
        
        if z_idx is None:
            z_idx = self.nz // 2
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot potential slice
        U_slice = U[:, :, z_idx].real
        vmin, vmax = np.percentile(U_slice, [5, 95])
        
        im = ax.pcolormesh(
            self.x * 1e3, self.y * 1e3, U_slice.T,
            shading='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax
        )
        ax.set_xlabel('x [mm]')
        ax.set_ylabel('y [mm]')
        ax.set_title(f'Gor\'kov Potential at z = {self.z[z_idx]*1e3:.2f} mm')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, label='U [J]')
        
        # Find and mark local minima (traps)
        from scipy.ndimage import minimum_filter
        filtered = minimum_filter(U_slice, size=5, mode='constant', cval=np.inf)
        minima_mask = (U_slice == filtered) & (U_slice < vmin + 0.3 * (vmax - vmin))
        
        min_indices = np.argwhere(minima_mask)
        if len(min_indices) > 0:
            min_x = self.x[min_indices[:, 0]] * 1e3
            min_y = self.y[min_indices[:, 1]] * 1e3
            ax.scatter(min_x, min_y, c='k', marker='x', s=50, linewidths=2,
                      label='Trap locations')
            ax.legend()
        
        plt.tight_layout()
        
        if save:
            self._save_fig(fig, 'gorkov_potential')
        
        return fig
    
    def plot_streaming_field(
        self,
        z_idx: Optional[int] = None,
        quiver_density: int = 10,
        save: bool = True,
    ) -> plt.Figure:
        """
        Plot streaming velocity field.
        
        Parameters
        ----------
        z_idx : int, optional
            z-index for slice. Default: middle.
        quiver_density : int
            Subsampling for quiver plot.
        save : bool
            Save figure to file.
        
        Returns
        -------
        fig : Figure
            Matplotlib figure.
        """
        if self.results.streaming_field is None:
            raise ValueError("Streaming field not computed")
        
        streaming = self.results.streaming_field
        
        if z_idx is None:
            z_idx = self.nz // 2
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Velocity magnitude
        v_mag = np.sqrt(
            streaming.vx**2 + streaming.vy**2 + streaming.vz**2
        )
        
        ax = axes[0]
        im = ax.pcolormesh(
            self.x * 1e3, self.y * 1e3, v_mag[:, :, z_idx].T * 1e6,
            shading='auto', cmap='magma'
        )
        ax.set_xlabel('x [mm]')
        ax.set_ylabel('y [mm]')
        ax.set_title(f'|u_stream| at z = {self.z[z_idx]*1e3:.2f} mm')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, label='|u| [μm/s]')
        
        # Quiver plot
        ax = axes[1]
        skip = quiver_density
        X, Y = np.meshgrid(self.x, self.y, indexing='ij')
        
        ax.quiver(
            X[::skip, ::skip] * 1e3,
            Y[::skip, ::skip] * 1e3,
            streaming.vx[::skip, ::skip, z_idx],
            streaming.vy[::skip, ::skip, z_idx],
            v_mag[::skip, ::skip, z_idx],
            cmap='viridis',
        )
        ax.set_xlabel('x [mm]')
        ax.set_ylabel('y [mm]')
        ax.set_title('Streaming velocity vectors')
        ax.set_aspect('equal')
        
        plt.tight_layout()
        
        if save:
            self._save_fig(fig, 'streaming_field')
        
        return fig
    
    def plot_particle_trajectories(
        self,
        projection: str = 'xy',
        save: bool = True,
    ) -> plt.Figure:
        """
        Plot particle trajectories.
        
        Parameters
        ----------
        projection : str
            Projection: 'xy', 'xz', 'yz', or '3d'.
        save : bool
            Save figure to file.
        
        Returns
        -------
        fig : Figure
            Matplotlib figure.
        """
        if self.results.particle_trajectories is None:
            raise ValueError("Particle trajectories not computed")
        
        trajectories = self.results.particle_trajectories
        
        if projection == '3d':
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            for i, traj in enumerate(trajectories):
                ax.plot(
                    traj.positions[:, 0] * 1e3,
                    traj.positions[:, 1] * 1e3,
                    traj.positions[:, 2] * 1e3,
                    alpha=0.7, label=f'P{i+1}'
                )
                ax.scatter(
                    traj.positions[0, 0] * 1e3,
                    traj.positions[0, 1] * 1e3,
                    traj.positions[0, 2] * 1e3,
                    marker='o', s=50
                )
                ax.scatter(
                    traj.positions[-1, 0] * 1e3,
                    traj.positions[-1, 1] * 1e3,
                    traj.positions[-1, 2] * 1e3,
                    marker='x', s=50
                )
            
            ax.set_xlabel('x [mm]')
            ax.set_ylabel('y [mm]')
            ax.set_zlabel('z [mm]')
            ax.set_title('Particle Trajectories')
            
        else:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            idx_map = {'xy': (0, 1), 'xz': (0, 2), 'yz': (1, 2)}
            label_map = {'xy': ('x', 'y'), 'xz': ('x', 'z'), 'yz': ('y', 'z')}
            
            i1, i2 = idx_map[projection]
            l1, l2 = label_map[projection]
            
            cmap = plt.cm.tab10
            for i, traj in enumerate(trajectories):
                color = cmap(i % 10)
                ax.plot(
                    traj.positions[:, i1] * 1e3,
                    traj.positions[:, i2] * 1e3,
                    color=color, alpha=0.7, label=f'P{i+1}'
                )
                ax.scatter(
                    traj.positions[0, i1] * 1e3,
                    traj.positions[0, i2] * 1e3,
                    color=color, marker='o', s=50
                )
                ax.scatter(
                    traj.positions[-1, i1] * 1e3,
                    traj.positions[-1, i2] * 1e3,
                    color=color, marker='x', s=50
                )
            
            ax.set_xlabel(f'{l1} [mm]')
            ax.set_ylabel(f'{l2} [mm]')
            ax.set_title(f'Particle Trajectories ({projection.upper()} view)')
            ax.set_aspect('equal')
            if len(trajectories) <= 10:
                ax.legend()
        
        plt.tight_layout()
        
        if save:
            self._save_fig(fig, f'trajectories_{projection}')
        
        return fig
    
    def animate_particles(
        self,
        output_path: Optional[Path] = None,
        fps: int = 30,
        duration: float = 5.0,
        projection: str = 'xy',
    ) -> None:
        """
        Create animated GIF of particle trajectories.
        
        Parameters
        ----------
        output_path : Path, optional
            Output path for GIF. Default: output_dir/particles.gif.
        fps : int
            Frames per second.
        duration : float
            Animation duration [s].
        projection : str
            Projection for 2D view.
        """
        if not HAS_IMAGEIO:
            raise ImportError("imageio required for animation")
        
        if self.results.particle_trajectories is None:
            raise ValueError("Particle trajectories not computed")
        
        trajectories = self.results.particle_trajectories
        
        if output_path is None:
            output_path = self.output_dir / f"anim_particles_{projection}.gif"
        else:
            output_path = Path(output_path)
        
        n_frames = int(fps * duration)
        frames = []
        
        idx_map = {'xy': (0, 1), 'xz': (0, 2), 'yz': (1, 2)}
        label_map = {'xy': ('x', 'y'), 'xz': ('x', 'z'), 'yz': ('y', 'z')}
        i1, i2 = idx_map[projection]
        l1, l2 = label_map[projection]
        
        # Time range
        t_max = max(traj.times[-1] for traj in trajectories)
        times = np.linspace(0, t_max, n_frames)
        
        # Compute bounds
        all_pos = np.vstack([traj.positions for traj in trajectories])
        x_range = (all_pos[:, i1].min() * 1e3 - 0.5, all_pos[:, i1].max() * 1e3 + 0.5)
        y_range = (all_pos[:, i2].min() * 1e3 - 0.5, all_pos[:, i2].max() * 1e3 + 0.5)
        
        cmap = plt.cm.tab10
        
        for frame_idx, t in enumerate(times):
            fig, ax = plt.subplots(figsize=(8, 6))
            
            for i, traj in enumerate(trajectories):
                color = cmap(i % 10)
                
                # Find position at this time
                if t <= traj.times[-1]:
                    pos = traj.position_at(t)
                else:
                    pos = traj.positions[-1]
                
                # Plot trail
                mask = traj.times <= t
                if np.any(mask):
                    ax.plot(
                        traj.positions[mask, i1] * 1e3,
                        traj.positions[mask, i2] * 1e3,
                        color=color, alpha=0.3, linewidth=1
                    )
                
                # Plot current position
                ax.scatter(
                    pos[i1] * 1e3, pos[i2] * 1e3,
                    color=color, s=100, edgecolors='k', zorder=10
                )
            
            ax.set_xlim(x_range)
            ax.set_ylim(y_range)
            ax.set_xlabel(f'{l1} [mm]')
            ax.set_ylabel(f'{l2} [mm]')
            ax.set_title(f't = {t*1e3:.2f} ms')
            ax.set_aspect('equal')
            
            # Convert to image
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(image)
            plt.close(fig)
        
        # Save GIF
        imageio.mimsave(output_path, frames, fps=fps, loop=0)
        print(f"Saved animation: {output_path}")
    
    def plot_energy_budget(self, save: bool = True) -> plt.Figure:
        """
        Plot energy budget analysis.
        
        Returns
        -------
        fig : Figure
            Matplotlib figure.
        """
        energy = self.results.energy_budget
        
        if not energy:
            raise ValueError("Energy budget not computed")
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Bar chart of energy components
        ax = axes[0]
        labels = list(energy.keys())
        values = [energy[k] for k in labels]
        
        colors = ['steelblue' if v >= 0 else 'salmon' for v in values]
        ax.barh(labels, values, color=colors)
        ax.set_xlabel('Energy [J]')
        ax.set_title('Energy Budget')
        ax.axvline(0, color='k', linewidth=0.5)
        
        # Pie chart (absolute values)
        ax = axes[1]
        abs_values = [abs(v) for v in values]
        ax.pie(abs_values, labels=labels, autopct='%1.1f%%')
        ax.set_title('Energy Distribution')
        
        plt.tight_layout()
        
        if save:
            self._save_fig(fig, 'energy_budget')
        
        return fig
    
    def plot_summary(self, save: bool = True) -> plt.Figure:
        """
        Create comprehensive summary plot.
        
        Returns
        -------
        fig : Figure
            Matplotlib figure.
        """
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig)
        
        pressure = self.results.acoustic_field.pressure
        p_mag = np.abs(pressure)
        z_mid = self.nz // 2
        y_mid = self.ny // 2
        
        # Pressure XY slice
        ax1 = fig.add_subplot(gs[0, 0])
        im = ax1.pcolormesh(
            self.x * 1e3, self.y * 1e3, p_mag[:, :, z_mid].T,
            shading='auto', cmap='viridis'
        )
        ax1.set_xlabel('x [mm]')
        ax1.set_ylabel('y [mm]')
        ax1.set_title(f'|p| (z={self.z[z_mid]*1e3:.1f}mm)')
        ax1.set_aspect('equal')
        plt.colorbar(im, ax=ax1)
        
        # Pressure XZ slice
        ax2 = fig.add_subplot(gs[0, 1])
        im = ax2.pcolormesh(
            self.x * 1e3, self.z * 1e3, p_mag[:, y_mid, :].T,
            shading='auto', cmap='viridis'
        )
        ax2.set_xlabel('x [mm]')
        ax2.set_ylabel('z [mm]')
        ax2.set_title('|p| (y=0)')
        ax2.set_aspect('equal')
        plt.colorbar(im, ax=ax2)
        
        # Gor'kov potential
        ax3 = fig.add_subplot(gs[0, 2])
        if self.results.gorkov_potential is not None:
            U = self.results.gorkov_potential[:, :, z_mid].real
            vmin, vmax = np.percentile(U, [5, 95])
            im = ax3.pcolormesh(
                self.x * 1e3, self.y * 1e3, U.T,
                shading='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax
            )
            ax3.set_title('Gor\'kov Potential')
            plt.colorbar(im, ax=ax3)
        ax3.set_xlabel('x [mm]')
        ax3.set_ylabel('y [mm]')
        ax3.set_aspect('equal')
        
        # Streaming velocity
        ax4 = fig.add_subplot(gs[1, 0])
        if self.results.streaming_field is not None:
            streaming = self.results.streaming_field
            v_mag = np.sqrt(
                streaming.vx**2 + streaming.vy**2 + streaming.vz**2
            )
            im = ax4.pcolormesh(
                self.x * 1e3, self.y * 1e3, v_mag[:, :, z_mid].T * 1e6,
                shading='auto', cmap='magma'
            )
            ax4.set_title('|u_stream| [μm/s]')
            plt.colorbar(im, ax=ax4)
        ax4.set_xlabel('x [mm]')
        ax4.set_ylabel('y [mm]')
        ax4.set_aspect('equal')
        
        # Streaming vectors
        ax5 = fig.add_subplot(gs[1, 1])
        if self.results.streaming_field is not None:
            skip = max(1, min(self.nx, self.ny) // 15)
            X, Y = np.meshgrid(self.x, self.y, indexing='ij')
            ax5.quiver(
                X[::skip, ::skip] * 1e3,
                Y[::skip, ::skip] * 1e3,
                streaming.vx[::skip, ::skip, z_mid],
                streaming.vy[::skip, ::skip, z_mid],
            )
            ax5.set_title('Streaming vectors')
        ax5.set_xlabel('x [mm]')
        ax5.set_ylabel('y [mm]')
        ax5.set_aspect('equal')
        
        # Particle trajectories
        ax6 = fig.add_subplot(gs[1, 2])
        if self.results.particle_trajectories is not None:
            cmap = plt.cm.tab10
            for i, traj in enumerate(self.results.particle_trajectories[:10]):
                color = cmap(i % 10)
                ax6.plot(
                    traj.positions[:, 0] * 1e3,
                    traj.positions[:, 1] * 1e3,
                    color=color, alpha=0.7
                )
                ax6.scatter(
                    traj.positions[0, 0] * 1e3,
                    traj.positions[0, 1] * 1e3,
                    color=color, marker='o', s=30
                )
                ax6.scatter(
                    traj.positions[-1, 0] * 1e3,
                    traj.positions[-1, 1] * 1e3,
                    color=color, marker='x', s=30
                )
            ax6.set_title('Particle Trajectories')
        ax6.set_xlabel('x [mm]')
        ax6.set_ylabel('y [mm]')
        ax6.set_aspect('equal')
        
        # Computation times
        ax7 = fig.add_subplot(gs[2, 0])
        times = self.results.computation_times
        if times:
            ax7.barh(list(times.keys()), list(times.values()), color='steelblue')
            ax7.set_xlabel('Time [s]')
            ax7.set_title('Computation Time')
        
        # Parameters text
        ax8 = fig.add_subplot(gs[2, 1])
        params = self.results.parameters
        text = (
            f"Frequency: {params.frequency/1e6:.2f} MHz\n"
            f"Dish radius: {params.dish_radius*1e3:.1f} mm\n"
            f"Water depth: {params.water_depth*1e3:.1f} mm\n"
            f"Grid resolution: {params.grid_resolution*1e6:.0f} μm\n"
            f"Grid shape: {self.results.geometry.shape}\n"
            f"PML thickness: {params.pml_thickness} pts\n"
            f"Temperature: {params.temperature:.1f} °C"
        )
        ax8.text(0.1, 0.5, text, transform=ax8.transAxes,
                fontsize=10, verticalalignment='center', family='monospace')
        ax8.axis('off')
        ax8.set_title('Parameters')
        
        # Energy budget
        ax9 = fig.add_subplot(gs[2, 2])
        energy = self.results.energy_budget
        if energy:
            labels = list(energy.keys())[:5]
            values = [energy[k] for k in labels]
            colors = ['steelblue' if v >= 0 else 'salmon' for v in values]
            ax9.barh(labels, values, color=colors)
            ax9.set_xlabel('Energy [J]')
            ax9.set_title('Energy Budget')
            ax9.axvline(0, color='k', linewidth=0.5)
        
        plt.tight_layout()
        
        if save:
            self._save_fig(fig, 'summary')
        
        return fig


def create_all_plots(results: "MultiphysicsResults", output_dir: Path) -> None:
    """
    Create all visualization plots for results.
    
    Parameters
    ----------
    results : MultiphysicsResults
        Simulation results.
    output_dir : Path
        Output directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    viz = MultiphysicsVisualizer(results, output_dir)
    
    # Static plots
    viz.plot_pressure_slices()
    
    if results.gorkov_potential is not None:
        viz.plot_gorkov_potential()
    
    if results.streaming_field is not None:
        viz.plot_streaming_field()
    
    if results.particle_trajectories is not None:
        viz.plot_particle_trajectories(projection='xy')
        viz.plot_particle_trajectories(projection='xz')
        
        if HAS_IMAGEIO:
            viz.animate_particles(projection='xy')
    
    if results.energy_budget:
        viz.plot_energy_budget()
    
    viz.plot_summary()
    
    print(f"\nAll plots saved to: {output_dir}")
