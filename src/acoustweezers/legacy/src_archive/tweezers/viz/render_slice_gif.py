"""Streaming GIF renderer for 2D slices without storing all frames in RAM."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import imageio
import os


class SliceGifRenderer:
    """
    Render a 2D slice animation to a GIF using streaming writer.
    Frames are encoded one at a time and written immediately, avoiding RAM buildup.
    """
    
    def __init__(self, output_path, figsize=(8, 6), dpi=100, fps=10):
        """
        Initialize the GIF renderer.
        Args:
            output_path: Where to save the GIF.
            figsize: Figure size (width, height) in inches.
            dpi: DPI for rendering.
            fps: Frames per second.
        """
        self.output_path = output_path
        self.figsize = figsize
        self.dpi = dpi
        self.fps = fps
        self.duration = 1.0 / fps  # imageio uses duration in seconds per frame
        self.writer = None
        self.frame_count = 0
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    def start(self):
        """Start the GIF writer."""
        self.writer = imageio.get_writer(self.output_path, fps=self.fps)
    
    def add_frame(self, img_array):
        """
        Add a single frame (as RGB numpy array, shape (H, W, 3)) to the GIF.
        Args:
            img_array: numpy array, shape (H, W, 3), dtype uint8 or float [0,1]
        """
        if self.writer is None:
            self.start()
        # Ensure uint8
        if img_array.dtype != np.uint8:
            img_array = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)
        self.writer.append_data(img_array)
        self.frame_count += 1
    
    def finish(self):
        """Close the GIF writer and finalize the output."""
        if self.writer is not None:
            self.writer.close()
            print(f"[GIF] Wrote {self.frame_count} frames to {self.output_path}")
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args):
        self.finish()


def render_slice_frame_to_array(F_slice, x_grid, y_grid, particle_pos_xy, particle_z, 
                                  Fx, Fy, t, color_map='viridis', norm=None,
                                  title_suffix=""):
    """
    Render a 2D force/potential slice with particle overlay to an RGB array.
    Does NOT display; only returns pixel array.
    
    Args:
        F_slice: 2D array (Nx, Ny), the field to visualize.
        x_grid, y_grid: 1D arrays for axis labels.
        particle_pos_xy: (x, y) position of particle.
        particle_z: z position (for display only).
        Fx, Fy: Force components at current particle position.
        t: Time (for display).
        color_map: matplotlib colormap name.
        norm: matplotlib Normalize object (for consistent coloring across frames).
        title_suffix: Optional suffix for title.
    
    Returns:
        frame: RGB numpy array, shape (H, W, 3), dtype uint8.
    """
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    
    if norm is None:
        vmin, vmax = np.min(F_slice), np.max(F_slice)
        norm = plt.Normalize(vmin, vmax)
    
    cmap = cm.get_cmap(color_map)
    im = ax.imshow(F_slice.T, origin='lower', extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]],
                    cmap=cmap, norm=norm)
    
    # Particle position marker
    z_norm = np.clip((particle_z - 0.0001) / 0.001, 0, 1)  # Assume z in ~[0, 0.01]
    particle_color = cm.plasma(z_norm)
    ax.scatter(particle_pos_xy[0], particle_pos_xy[1], s=80, color=particle_color, 
               edgecolor='white', linewidth=1.5, zorder=5, label='Particle')
    
    # Optional: force vector
    arrow_scale = 1e4  # Adjust to visualize force
    ax.arrow(particle_pos_xy[0], particle_pos_xy[1], 
             Fx*arrow_scale, Fy*arrow_scale, 
             head_width=0.0003, head_length=0.0003, color='red', alpha=0.7, zorder=4)
    
    # Labels and title
    ax.set_xlabel('x [m]', fontsize=10)
    ax.set_ylabel('y [m]', fontsize=10)
    title = f'Slice at t={t:.3f}s, z={particle_z*1e3:.1f}mm {title_suffix}'
    ax.set_title(title, fontsize=11)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('|F| [N]', fontsize=9)
    
    ax.legend(loc='upper right', fontsize=9)
    fig.tight_layout()
    
    # Render to array
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    frame = np.frombuffer(buf, dtype=np.uint8)
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    frame_rgb = frame[..., :3]  # Drop alpha
    
    plt.close(fig)
    
    return frame_rgb


def render_trajectory_2d_slice(x_grid, y_grid, z_grid, 
                               traj_csv_or_dict, force_2d_slice_list, 
                               slice_z, output_gif_path, 
                               downsample=1, title_suffix=""):
    """
    Render a 2D trajectory animation from trajectory data and force slices.
    Uses streaming writer to avoid RAM buildup.
    
    Args:
        x_grid, y_grid, z_grid: Grid coordinate arrays.
        traj_csv_or_dict: Either path to CSV (columns: t, x, y, z, Fx, Fy, Fz, U) 
                          or dict with keys 't', 'x', 'y', 'z', 'Fx', 'Fy'.
        force_2d_slice_list: List of 2D force slices (one per trajectory point).
        slice_z: z coordinate of the slice (for reference in title).
        output_gif_path: Where to save the GIF.
        downsample: Render every downsample-th frame.
        title_suffix: Optional string to append to frame title.
    """
    # Load trajectory
    if isinstance(traj_csv_or_dict, str):
        traj = np.loadtxt(traj_csv_or_dict, delimiter=',', skiprows=1)
        t = traj[:, 0]
        x = traj[:, 1]
        y = traj[:, 2]
        z = traj[:, 3]
        Fx = traj[:, 4]
        Fy = traj[:, 5]
    else:
        t = traj_csv_or_dict['t']
        x = traj_csv_or_dict['x']
        y = traj_csv_or_dict['y']
        z = traj_csv_or_dict['z']
        Fx = traj_csv_or_dict['Fx']
        Fy = traj_csv_or_dict['Fy']
    
    n_steps = len(x)
    assert len(force_2d_slice_list) == n_steps, \
        f"force_2d_slice_list ({len(force_2d_slice_list)}) != n_steps ({n_steps})"
    
    # Compute normalization for consistent coloring
    F_min = min(np.min(F) for F in force_2d_slice_list if F.size > 0)
    F_max = max(np.max(F) for F in force_2d_slice_list if F.size > 0)
    norm = plt.Normalize(F_min, F_max)
    
    # Use streaming writer
    os.makedirs(os.path.dirname(output_gif_path), exist_ok=True)
    writer = imageio.get_writer(output_gif_path, fps=10)
    
    for i in range(0, n_steps, downsample):
        F_slice = force_2d_slice_list[i]
        frame = render_slice_frame_to_array(
            F_slice, x_grid, y_grid,
            (x[i], y[i]), z[i],
            Fx[i], Fy[i], t[i],
            norm=norm,
            title_suffix=title_suffix
        )
        writer.append_data(frame)
    
    writer.close()
    n_frames = (n_steps + downsample - 1) // downsample
    print(f"[GIF] Rendered {n_frames} frames to {output_gif_path}")
