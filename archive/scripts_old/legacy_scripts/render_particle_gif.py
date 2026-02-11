import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
import csv
from matplotlib import cm

def load_traj(csv_path):
    data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
    t = data[:,0]
    x = data[:,1]
    y = data[:,2]
    z = data[:,3]
    Fx = data[:,4]
    Fy = data[:,5]
    Fz = data[:,6]
    U = data[:,7]
    return t, x, y, z, Fx, Fy, Fz, U

def load_force_volume(npz_path):
    arrs = np.load(npz_path)
    if 'Fmag2d' in arrs:
        # 2D slice mode
        Fmag2d = arrs['Fmag2d']
        x = arrs['x']
        y = arrs['y']
        z_slice = arrs['z_slice'].item() if hasattr(arrs['z_slice'], 'item') else float(arrs['z_slice'])
        return Fmag2d, x, y, z_slice
    else:
        Fmag = arrs['Fmag']
        x = arrs['x']
        y = arrs['y']
        z = arrs['z']
        return Fmag, x, y, z

def render_gif(traj_csv, force_npz, out_gif, slice_z=None, downsample=5, title='Force slice + particle'):
    t, x, y, z, Fx, Fy, Fz, U = load_traj(traj_csv)
    Fmag, x_grid, y_grid, z_info = load_force_volume(force_npz)
    # If Fmag is 3D (full), select z slice; if 2D, use as is
    if Fmag.ndim == 3:
        # (n_steps, Nx, Ny) 2D slices
        F_slices = Fmag
        z_slice = z_info if slice_z is None else slice_z
    else:
        # (n_steps, Nx, Ny) 2D slices
        F_slices = Fmag
        z_slice = z_info
    frames = []
    cmap = cm.viridis
    norm = plt.Normalize(np.min(F_slices), np.max(F_slices))
    # Encode z as color
    if np.max(z) > np.min(z):
        z_colors = cm.plasma((z - np.min(z)) / (np.max(z) - np.min(z)))
    else:
        z_colors = np.tile(cm.plasma(0.5), (len(z), 1))
    for i in range(0, len(x), downsample):
        fig, ax = plt.subplots(figsize=(6,5))
        F_slice = F_slices[i]
        im = ax.imshow(F_slice.T, origin='lower', extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]], cmap=cmap, norm=norm)
        ax.plot(x[:i+1], y[:i+1], color='white', lw=1.5, alpha=0.7, label='Trajectory')
        ax.scatter(x[i], y[i], color=z_colors[i], s=60, edgecolor='black', label='Particle')
        ax.set_title(f'{title} (t={t[i]:.3f}s, z={z[i]*1e3:.1f}mm)')
        plt.colorbar(im, ax=ax, label='|F| [N]')
        # Optional: draw local force arrow
        ax.arrow(x[i], y[i], Fx[i]*1e4, Fy[i]*1e4, color='red', head_width=0.0005, length_includes_head=True)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.legend()
        fig.tight_layout()
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        frame = np.frombuffer(buf, dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        frame = frame[...,:3]
        frames.append(frame)
        plt.close(fig)
    imageio.mimsave(out_gif, frames, duration=0.08)
    print(f'[GIF] Saved {out_gif} with {len(frames)} frames.')
    # --- Diagnostics ---
    dx = x_grid[1] - x_grid[0]
    step_sizes = np.sqrt(np.diff(x)**2 + np.diff(y)**2 + np.diff(z)**2)
    print(f'[DIAG] Median step size / dx: {np.median(step_sizes)/dx:.2f}, Max: {np.max(step_sizes)/dx:.2f}')
    U_traj = U
    print(f'[DIAG] U(t) min: {np.min(U_traj):.3e}, max: {np.max(U_traj):.3e}, start: {U_traj[0]:.3e}, end: {U_traj[-1]:.3e}')
    F_traj = np.sqrt(Fx**2 + Fy**2 + Fz**2)
    print(f'[DIAG] Max |F| along traj: {np.max(F_traj):.3e}')
    clamped = np.sum((x==x_grid[0])|(x==x_grid[-1])|(y==y_grid[0])|(y==y_grid[-1])|(z==z_grid[0])|(z==z_grid[-1]))
    print(f'[DIAG] % clamped steps: {100*clamped/len(x):.1f}%')

if __name__ == "__main__":
    # Moving lens (time-varying transducer) high-res GIF
    traj_csv = 'results/helmholtz3d_demo/traj_moving_lens.csv'
    force_npz = 'results/helmholtz3d_demo/force_moving_lens.npz'
    out_gif = 'results/helmholtz3d_demo/gif_force_slice_moving_lens.gif'
    render_gif(traj_csv, force_npz, out_gif, slice_z=0.015, downsample=3, title='Moving lens |F| slice')
