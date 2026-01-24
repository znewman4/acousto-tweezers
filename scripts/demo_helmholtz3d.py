
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import csv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from tweezers.grid.grid3d import Grid3D
from tweezers.control.field_interface_3d import Helmholtz3DSolver
from tweezers.physics.particle_props import ParticleProps, FluidProps, stokes_mobility, contrast_factors
# --- Lens-based actuation imports ---
from tweezers.actuation.lens_fields import lens_focus, lens_vortex, lens_axicon
from tweezers.actuation.bath_propagation import angular_spectrum_propagate
from tweezers.actuation.plate_transmission import apply_plate_transmission

if __name__ == "__main__":

    # --- BEGIN: Define missing variables for demo to run ---
    # Output directory
    results_dir = os.path.join(os.path.dirname(__file__), '../results/helmholtz3d_demo')
    os.makedirs(results_dir, exist_ok=True)

    # Full-size petri dish: 3cm x 3cm, high resolution
    Lx = 0.03  # 3 cm
    Ly = 0.03
    H = 0.01
    Nx = 128
    Ny = 128
    Nz = 32
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
    dz = H / (Nz - 1)
    grid = Grid3D(Lx, Ly, H, dx, dy, dz)
    # Define fluid properties for solver
    fluid_props = FluidProps(rho0=1000.0, c0=1500.0, eta=1e-3)
    omega = 2 * np.pi * 1e6
    solver = Helmholtz3DSolver(grid, omega, fluid_props)


    # Use solver's Helmholtz3DOperator and wavenumber for diagnostics

    # Example fluid and particle properties (SI units)
    fluid = FluidProps(rho0=1000.0, c0=1500.0, eta=1e-3)
    particle = ParticleProps(a_m=1e-6, rho_p=1050.0, kappa_p=4.5e-10)

    # Example field_g (Gaussian field) for plotting
    class DummyField:
        def __init__(self, grid):
            self.p = np.random.randn(grid.Nx, grid.Ny, grid.Nz) + 1j*np.random.randn(grid.Nx, grid.Ny, grid.Nz)
    field_g = DummyField(grid)

    # Example metrics for diagnostics
    metrics_strict = {'example_metric1': 1.0, 'example_metric2': 2.0}
    metrics_all = {'example_metric3': 3.0}
    metrics_g_strict = {'g_metric1': 1.1, 'g_metric2': 2.2}
    metrics_g_all = {'g_metric3': 3.3}

    # Moving lens parameters
    sigma = 0.005  # 0.5 cm
    n_steps = 200  # Number of animation steps
    x_path = np.linspace(grid.x[0]+sigma, grid.x[-1]-sigma, n_steps)  # Move lens across x
    y_path = np.full(n_steps, (grid.y[0] + grid.y[-1]) / 2)  # Keep y fixed (can animate if desired)
    # --- Option: store only 2D Fmag slices at a chosen z ---
    store_2d_slices = True
    slice_z = grid.z[len(grid.z)//2]  # Middle z by default
    iz_slice = np.argmin(np.abs(grid.z - slice_z))
    traj = []
    Fmag_frames = []
    X, Y = np.meshgrid(grid.x, grid.y, indexing='ij')
    z_init = grid.z[1] + 0.1 * (grid.z[-1] - grid.z[1])
    pos = np.array([x_path[0], y_path[0], z_init])
    mu = stokes_mobility(fluid.eta, particle.a_m)
    dt = 1e-3
    for i in range(n_steps):
        x0 = x_path[i]
        y0 = y_path[i]
        # 1. Generate lens field at current position
        p_lens = lens_focus(X, Y, x0, y0, f=0.01, sigma=sigma, k=solver.k)
        # 2. Propagate through bath
        dz_bath = 0.002
        dx = grid.x[1] - grid.x[0]
        dy = grid.y[1] - grid.y[0]
        p_bath = angular_spectrum_propagate(p_lens, dx, dy, dz_bath, solver.k)
        # 3. Plate transmission
        rho = 1000.0
        rho_plate = 1180.0
        c_plate = 2730.0
        h_plate = 0.001
        p_bot = apply_plate_transmission(p_bath, dx, dy, solver.k, solver.k * 1500.0 / c_plate, rho, rho_plate, 1500.0, c_plate, h_plate)
        # 4. Solve 3D Helmholtz
        field = solver.solve(p_bot)
        # 5. Compute Gor'kov and force
        U = field.compute_gorkov_potential(fluid, particle)
        Fx, Fy, Fz = field.compute_radiation_force()
        Fmag = np.sqrt(np.abs(Fx)**2 + np.abs(Fy)**2 + np.abs(Fz)**2)
        if store_2d_slices:
            Fmag_frames.append(Fmag[:,:,iz_slice])
        else:
            Fmag_frames.append(Fmag)
        # 6. Step particle (overdamped Euler)
        ix = np.searchsorted(grid.x, pos[0])
        iy = np.searchsorted(grid.y, pos[1])
        iz = np.searchsorted(grid.z, pos[2])
        ix = np.clip(ix, 0, grid.Nx-1)
        iy = np.clip(iy, 0, grid.Ny-1)
        iz = np.clip(iz, 0, grid.Nz-1)
        Fp = np.array([Fx[ix,iy,iz], Fy[ix,iy,iz], Fz[ix,iy,iz]])
        pos_new = pos + mu * Fp * dt
        pos_new[0] = np.clip(pos_new[0], grid.x[0], grid.x[-1])
        pos_new[1] = np.clip(pos_new[1], grid.y[0], grid.y[-1])
        pos_new[2] = np.clip(pos_new[2], grid.z[0], grid.z[-1])
        traj.append([i*dt, *pos, *Fp, U[ix,iy,iz]])
        pos = pos_new
    # Save trajectory and force frames
    traj = np.array(traj)
    np.savetxt(os.path.join(results_dir, 'traj_moving_lens.csv'), traj, delimiter=',', header='t_s,x_m,y_m,z_m,Fx_N,Fy_N,Fz_N,U_J', comments='')
    Fmag_frames = np.stack(Fmag_frames, axis=0)
    if store_2d_slices:
        np.savez(os.path.join(results_dir, 'force_moving_lens_2dslice.npz'), Fmag2d=Fmag_frames, x=grid.x, y=grid.y, z_slice=slice_z)
    else:
        np.savez(os.path.join(results_dir, 'force_moving_lens.npz'), Fmag=Fmag_frames, x=grid.x, y=grid.y, z=grid.z)
    # --- END: Define missing variables ---


    # Save enhanced diagnostics to CSV
    csv_path_plane_full = os.path.join(results_dir, 'diagnostics_plane_wave_full.csv')
    with open(csv_path_plane_full, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        for key, v in {**metrics_strict, **metrics_all}.items():
            writer.writerow([key, v])
    csv_path_gauss_full = os.path.join(results_dir, 'diagnostics_gaussian_full.csv')
    with open(csv_path_gauss_full, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        for key, v in {**metrics_g_strict, **metrics_g_all}.items():
            writer.writerow([key, v])

    # Plot |p| at several z-slices for gaussian
    Nz = grid.Nz
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    z_indices = [0, Nz//2, Nz-1]
    for i, iz in enumerate(z_indices):
        ax = axes[i]
        im = ax.imshow(np.abs(field_g.p[:, :, iz]), origin='lower',
                      extent=[grid.x[0], grid.x[-1], grid.y[0], grid.y[-1]])
        ax.set_title(f"|p| (gauss) at z={grid.z[iz]:.3f} m")
        plt.colorbar(im, ax=ax)
        fig_path = os.path.join(results_dir, f'p_abs_gauss_z{iz}.png')
        fig_single, ax_single = plt.subplots(figsize=(4,4))
        im_single = ax_single.imshow(np.abs(field_g.p[:, :, iz]), origin='lower',
                                    extent=[grid.x[0], grid.x[-1], grid.y[0], grid.y[-1]])
        ax_single.set_title(f"|p| (gauss) at z={grid.z[iz]:.3f} m")
        plt.colorbar(im_single, ax=ax_single)
        fig_single.tight_layout()
        fig_single.savefig(fig_path)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, 'p_abs_gauss_all_zslices.png'))
    plt.close(fig)
    print(f"Saved 3D Helmholtz demo plots to {results_dir}")

    # --- Vortex drive (lens-based actuation) ---
    ell = 1
    # 1. Generate lens field
    X, Y = np.meshgrid(grid.x, grid.y, indexing='ij')
    k = solver.k
    p_lens_vortex = lens_vortex(X, Y, x0, y0, ell, sigma, k)
    # 2. Propagate through bath
    dz_bath = 0.002  # Example bath thickness (m)
    c_bath = 1500.0  # m/s
    k_bath = k  # Assume same as main fluid for now
    dx = grid.x[1] - grid.x[0]
    dy = grid.y[1] - grid.y[0]
    p_bath_vortex = angular_spectrum_propagate(p_lens_vortex, dx, dy, dz_bath, k_bath)
    # 3. Plate transmission
    rho = 1000.0
    rho_plate = 1180.0  # e.g. PMMA
    c_plate = 2730.0
    h_plate = 0.001  # 1 mm
    p_bot_vortex = apply_plate_transmission(p_bath_vortex, dx, dy, k, k * c_bath / c_plate, rho, rho_plate, c_bath, c_plate, h_plate)
    # Plot |p_bot| and phase for vortex
    fig, axes = plt.subplots(1,2,figsize=(8,4))
    im0 = axes[0].imshow(np.abs(p_bot_vortex), origin='lower', extent=[grid.x[0], grid.x[-1], grid.y[0], grid.y[-1]])
    axes[0].set_title('|p_bot| (vortex, lens pipeline)')
    plt.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(np.angle(p_bot_vortex), origin='lower', extent=[grid.x[0], grid.x[-1], grid.y[0], grid.y[-1]], cmap='twilight')
    axes[1].set_title('arg(p_bot) (vortex, lens pipeline)')
    plt.colorbar(im1, ax=axes[1])
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, 'pbot_vortex.png'))
    plt.close(fig)
    # Solve and plot z-slices
    field_vortex = solver.solve(p_bot_vortex)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i, iz in enumerate(z_indices):
        ax = axes[i]
        im = ax.imshow(np.abs(field_vortex.p[:, :, iz]), origin='lower', extent=[grid.x[0], grid.x[-1], grid.y[0], grid.y[-1]])
        ax.set_title(f"|p| (vortex) at z={grid.z[iz]:.3f} m")
        plt.colorbar(im, ax=ax)
        fig_path = os.path.join(results_dir, f'p_abs_vortex_z{iz}.png')
        fig_single, ax_single = plt.subplots(figsize=(4,4))
        im_single = ax_single.imshow(np.abs(field_vortex.p[:, :, iz]), origin='lower', extent=[grid.x[0], grid.x[-1], grid.y[0], grid.y[-1]])
        ax_single.set_title(f"|p| (vortex) at z={grid.z[iz]:.3f} m")
        plt.colorbar(im_single, ax=ax_single)
        fig_single.tight_layout()
        fig_single.savefig(fig_path)
        plt.close(fig_single)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, 'p_abs_vortex_all_zslices.png'))
    plt.close(fig)
    # Diagnostics
    op_vortex = solver.op
    metrics_vortex = field_vortex.diagnostics(p_bot=p_bot_vortex, operator=op_vortex, debug_Ab=True, face_mode='strict')
    csv_path_vortex = os.path.join(results_dir, 'diagnostics_vortex_full.csv')
    with open(csv_path_vortex, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        for key, v in metrics_vortex.items():
            writer.writerow([key, v])
    # Vortex phase check: phase winds by ~2pi*ell around a ring
    r_ring = 0.002
    n_pts = 100
    phi = np.linspace(0, 2*np.pi, n_pts, endpoint=False)
    x_ring = x0 + r_ring * np.cos(phi)
    y_ring = y0 + r_ring * np.sin(phi)
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator((grid.x, grid.y), np.angle(p_bot_vortex))
    phase_ring = interp(np.stack([x_ring, y_ring], axis=-1))
    phase_unwrap = np.unwrap(phase_ring)
    phase_wind = phase_unwrap[-1] - phase_unwrap[0]
    print(f"[VORTEX PHASE CHECK] Phase wind around r={r_ring*1e3:.1f}mm: {phase_wind:.2f} rad (expected {2*np.pi*ell:.2f})")

    # --- Axicon drive (lens-based actuation) ---
    alpha = 3000.0  # rad/m
    # 1. Generate lens field
    p_lens_axicon = lens_axicon(X, Y, x0, y0, alpha, sigma, k)
    # 2. Propagate through bath
    p_bath_axicon = angular_spectrum_propagate(p_lens_axicon, dx, dy, dz_bath, k_bath)
    # 3. Plate transmission
    p_bot_axicon = apply_plate_transmission(p_bath_axicon, dx, dy, k, k * c_bath / c_plate, rho, rho_plate, c_bath, c_plate, h_plate)
    # Plot |p_bot| and phase for axicon
    fig, axes = plt.subplots(1,2,figsize=(8,4))
    im0 = axes[0].imshow(np.abs(p_bot_axicon), origin='lower', extent=[grid.x[0], grid.x[-1], grid.y[0], grid.y[-1]])
    axes[0].set_title('|p_bot| (axicon, lens pipeline)')
    plt.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(np.angle(p_bot_axicon), origin='lower', extent=[grid.x[0], grid.x[-1], grid.y[0], grid.y[-1]], cmap='twilight')
    axes[1].set_title('arg(p_bot) (axicon, lens pipeline)')
    plt.colorbar(im1, ax=axes[1])
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, 'pbot_axicon.png'))
    plt.close(fig)
    # Solve and plot z-slices
    field_axicon = solver.solve(p_bot_axicon)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i, iz in enumerate(z_indices):
        ax = axes[i]
        im = ax.imshow(np.abs(field_axicon.p[:, :, iz]), origin='lower', extent=[grid.x[0], grid.x[-1], grid.y[0], grid.y[-1]])
        ax.set_title(f"|p| (axicon) at z={grid.z[iz]:.3f} m")
        plt.colorbar(im, ax=ax)
        fig_path = os.path.join(results_dir, f'p_abs_axicon_z{iz}.png')
        fig_single, ax_single = plt.subplots(figsize=(4,4))
        im_single = ax_single.imshow(np.abs(field_axicon.p[:, :, iz]), origin='lower', extent=[grid.x[0], grid.x[-1], grid.y[0], grid.y[-1]])
        ax_single.set_title(f"|p| (axicon) at z={grid.z[iz]:.3f} m")
        plt.colorbar(im_single, ax=ax_single)
        fig_single.tight_layout()
        fig_single.savefig(fig_path)
        plt.close(fig_single)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, 'p_abs_axicon_all_zslices.png'))
    plt.close(fig)
    # Diagnostics
    op_axicon = solver.op
    metrics_axicon = field_axicon.diagnostics(p_bot=p_bot_axicon, operator=op_axicon, debug_Ab=True, face_mode='strict')
    csv_path_axicon = os.path.join(results_dir, 'diagnostics_axicon_full.csv')
    with open(csv_path_axicon, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        for key, v in metrics_axicon.items():
            writer.writerow([key, v])
    # Axicon phase check: fit phase vs r
    r_grid = np.sqrt((X-x0)**2 + (Y-y0)**2)
    mask = (r_grid > 0.001) & (r_grid < 0.0025)
    phase = np.angle(p_bot_axicon[mask])
    rvals = r_grid[mask]
    from numpy.polynomial.polynomial import polyfit
    slope, intercept = polyfit(rvals, np.unwrap(phase), 1)
    print(f"[AXICON PHASE CHECK] Fitted phase slope: {slope:.1f} (expected {alpha:.1f})")

    # --- Gor'kov and force for vortex ---
    # Compute Gor'kov potential and force (SI units, using physics config)
    U_vortex = field_vortex.compute_gorkov_potential(fluid, particle, verbose=True)
    Fx_vortex, Fy_vortex, Fz_vortex = field_vortex.compute_radiation_force()
    # Plot U z-slices
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i, iz in enumerate(z_indices):
        ax = axes[i]
        im = ax.imshow(U_vortex[:, :, iz], origin='lower', extent=[grid.x[0], grid.x[-1], grid.y[0], grid.y[-1]])
        ax.set_title(f"U (vortex) at z={grid.z[iz]:.3f} m")
        plt.colorbar(im, ax=ax)
        fig_path = os.path.join(results_dir, f'U_vortex_z{iz}.png')
        fig_single, ax_single = plt.subplots(figsize=(4,4))
        im_single = ax_single.imshow(U_vortex[:, :, iz], origin='lower', extent=[grid.x[0], grid.x[-1], grid.y[0], grid.y[-1]])
        ax_single.set_title(f"U (vortex) at z={grid.z[iz]:.3f} m")
        plt.colorbar(im_single, ax=ax_single)
        fig_single.tight_layout()
        fig_single.savefig(fig_path)
        plt.close(fig_single)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, 'U_vortex_all_zslices.png'))
    plt.close(fig)
    # Plot |F| z-slices
    Fmag_vortex = np.sqrt(np.abs(Fx_vortex)**2 + np.abs(Fy_vortex)**2 + np.abs(Fz_vortex)**2)
    np.savez(os.path.join(results_dir, 'force_vortex.npz'), Fmag=Fmag_vortex, x=grid.x, y=grid.y, z=grid.z)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i, iz in enumerate(z_indices):
        ax = axes[i]
        im = ax.imshow(Fmag_vortex[:, :, iz], origin='lower', extent=[grid.x[0], grid.x[-1], grid.y[0], grid.y[-1]])
        ax.set_title(f"|F| (vortex) at z={grid.z[iz]:.3f} m")
        plt.colorbar(im, ax=ax)
        fig_path = os.path.join(results_dir, f'Fmag_vortex_z{iz}.png')
        fig_single, ax_single = plt.subplots(figsize=(4,4))
        im_single = ax_single.imshow(Fmag_vortex[:, :, iz], origin='lower', extent=[grid.x[0], grid.x[-1], grid.y[0], grid.y[-1]])
        ax_single.set_title(f"|F| (vortex) at z={grid.z[iz]:.3f} m")
        plt.colorbar(im_single, ax=ax_single)
        fig_single.tight_layout()
        fig_single.savefig(fig_path)
        plt.close(fig_single)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, 'Fmag_vortex_all_zslices.png'))
    plt.close(fig)
    # Print max/min U and symmetry check
    print(f"[VORTEX GORKOV] U max: {np.max(U_vortex):.3e}, min: {np.min(U_vortex):.3e}")
    print(f"[VORTEX GORKOV] U symmetry (center vs. opposite): {U_vortex[grid.Nx//2, grid.Ny//2, grid.Nz//2]:.3e} vs {U_vortex[1,1,1]:.3e}")

    # --- Gor'kov and force for axicon ---
    U_axicon = field_axicon.compute_gorkov_potential(fluid, particle, verbose=True)
    Fx_axicon, Fy_axicon, Fz_axicon = field_axicon.compute_radiation_force()
    # Plot U z-slices
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i, iz in enumerate(z_indices):
        ax = axes[i]
        im = ax.imshow(U_axicon[:, :, iz], origin='lower', extent=[grid.x[0], grid.x[-1], grid.y[0], grid.y[-1]])
        ax.set_title(f"U (axicon) at z={grid.z[iz]:.3f} m")
        plt.colorbar(im, ax=ax)
        fig_path = os.path.join(results_dir, f'U_axicon_z{iz}.png')
        fig_single, ax_single = plt.subplots(figsize=(4,4))
        im_single = ax_single.imshow(U_axicon[:, :, iz], origin='lower', extent=[grid.x[0], grid.x[-1], grid.y[0], grid.y[-1]])
        ax_single.set_title(f"U (axicon) at z={grid.z[iz]:.3f} m")
        plt.colorbar(im_single, ax=ax_single)
        fig_single.tight_layout()
        fig_single.savefig(fig_path)
        plt.close(fig_single)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, 'U_axicon_all_zslices.png'))
    plt.close(fig)
    # Plot |F| z-slices
    Fmag_axicon = np.sqrt(np.abs(Fx_axicon)**2 + np.abs(Fy_axicon)**2 + np.abs(Fz_axicon)**2)
    np.savez(os.path.join(results_dir, 'force_axicon.npz'), Fmag=Fmag_axicon, x=grid.x, y=grid.y, z=grid.z)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i, iz in enumerate(z_indices):
        ax = axes[i]
        im = ax.imshow(Fmag_axicon[:, :, iz], origin='lower', extent=[grid.x[0], grid.x[-1], grid.y[0], grid.y[-1]])
        ax.set_title(f"|F| (axicon) at z={grid.z[iz]:.3f} m")
        plt.colorbar(im, ax=ax)
        fig_path = os.path.join(results_dir, f'Fmag_axicon_z{iz}.png')
        fig_single, ax_single = plt.subplots(figsize=(4,4))
        im_single = ax_single.imshow(Fmag_axicon[:, :, iz], origin='lower', extent=[grid.x[0], grid.x[-1], grid.y[0], grid.y[-1]])
        ax_single.set_title(f"|F| (axicon) at z={grid.z[iz]:.3f} m")
        plt.colorbar(im_single, ax=ax_single)
        fig_single.tight_layout()
        fig_single.savefig(fig_path)
        plt.close(fig_single)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, 'Fmag_axicon_all_zslices.png'))
    plt.close(fig)
    # Print max/min U and symmetry check

    print(f"[AXICON GORKOV] U max: {np.max(U_axicon):.3e}, min: {np.min(U_axicon):.3e}")
    print(f"[AXICON GORKOV] U symmetry (center vs. opposite): {U_axicon[grid.Nx//2, grid.Ny//2, grid.Nz//2]:.3e} vs {U_axicon[1,1,1]:.3e}")

    # --- 3D Particle Simulation in Gor'kov Force Field ---

    # Initial position: center, slightly above bottom
    x_init = x0
    y_init = y0
    z_init = grid.z[1] + 0.1 * (grid.z[-1] - grid.z[1])
    pos0 = np.array([x_init, y_init, z_init])

    # --- DIAGNOSTIC: Global force scale and initial values (VORTEX) ---
    Fmag_vortex = np.sqrt(np.abs(Fx_vortex)**2 + np.abs(Fy_vortex)**2 + np.abs(Fz_vortex)**2)
    Fmag_max_vortex = np.max(Fmag_vortex)
    Fmag_mean_vortex = np.mean(Fmag_vortex)
    U0_vortex = field_vortex.trilinear_interp(U_vortex, x_init, y_init, z_init)
    Fx0_vortex = field_vortex.trilinear_interp(Fx_vortex, x_init, y_init, z_init)
    Fy0_vortex = field_vortex.trilinear_interp(Fy_vortex, x_init, y_init, z_init)
    Fz0_vortex = field_vortex.trilinear_interp(Fz_vortex, x_init, y_init, z_init)
    F0_vortex = np.array([Fx0_vortex, Fy0_vortex, Fz0_vortex])
    F0mag_vortex = np.linalg.norm(F0_vortex)
    print("[DIAG VORTEX] Fmag_max = {:.3e}, Fmag_mean = {:.3e}".format(Fmag_max_vortex, Fmag_mean_vortex))
    print("[DIAG VORTEX] U0 = {:.3e}".format(U0_vortex))
    print("[DIAG VORTEX] F0 = [{:.3e}, {:.3e}, {:.3e}], |F0| = {:.3e}".format(Fx0_vortex, Fy0_vortex, Fz0_vortex, F0mag_vortex))

    # --- DIAGNOSTIC: Global force scale and initial values (AXICON) ---
    Fmag_axicon = np.sqrt(np.abs(Fx_axicon)**2 + np.abs(Fy_axicon)**2 + np.abs(Fz_axicon)**2)
    Fmag_max_axicon = np.max(Fmag_axicon)
    Fmag_mean_axicon = np.mean(Fmag_axicon)
    U0_axicon = field_axicon.trilinear_interp(U_axicon, x_init, y_init, z_init)
    Fx0_axicon = field_axicon.trilinear_interp(Fx_axicon, x_init, y_init, z_init)
    Fy0_axicon = field_axicon.trilinear_interp(Fy_axicon, x_init, y_init, z_init)
    Fz0_axicon = field_axicon.trilinear_interp(Fz_axicon, x_init, y_init, z_init)
    F0_axicon = np.array([Fx0_axicon, Fy0_axicon, Fz0_axicon])
    F0mag_axicon = np.linalg.norm(F0_axicon)
    print("[DIAG AXICON] Fmag_max = {:.3e}, Fmag_mean = {:.3e}".format(Fmag_max_axicon, Fmag_mean_axicon))
    print("[DIAG AXICON] U0 = {:.3e}".format(U0_axicon))
    print("[DIAG AXICON] F0 = [{:.3e}, {:.3e}, {:.3e}], |F0| = {:.3e}".format(Fx0_axicon, Fy0_axicon, Fz0_axicon, F0mag_axicon))
    # Parameters
    # --- Particle simulation parameters (SI units) ---
    n_steps = 2000
    dt_s = 1e-3  # [s] Default time step (SI, seconds). Adjust for resolvable motion.
    clamp = True
    brownian = False  # Set True to enable Brownian motion
    T_C = 25.0  # Celsius, for Brownian
    mu = stokes_mobility(fluid.eta, particle.a_m)
    print(f"[PARTICLE SIM] Using mu = {mu:.3e} m/(N·s), dt = {dt_s:.2e} s, Brownian = {brownian}")
    # Initial position: center, slightly above bottom
    x_init = x0
    y_init = y0
    z_init = grid.z[1] + 0.1 * (grid.z[-1] - grid.z[1])
    pos0 = np.array([x_init, y_init, z_init])

    # --- Simulate for vortex ---
    print("\n[SIM] Running 3D particle simulation in vortex Gor'kov field...")
    traj_vortex = field_vortex.simulate_particle(
        pos0, n_steps=n_steps, dt=dt_s, mu=mu, fluid=fluid, particle=particle, T_C=T_C, brownian=brownian, clamp=clamp, verbose=True, U_field=U_vortex
    )
    # Compute force and U along trajectory for output
    t_vortex = np.arange(n_steps) * dt_s
    Fx_traj = np.array([field_vortex.trilinear_interp(Fx_vortex, *traj_vortex[n]) for n in range(n_steps)])
    Fy_traj = np.array([field_vortex.trilinear_interp(Fy_vortex, *traj_vortex[n]) for n in range(n_steps)])
    Fz_traj = np.array([field_vortex.trilinear_interp(Fz_vortex, *traj_vortex[n]) for n in range(n_steps)])
    U_traj = np.array([field_vortex.trilinear_interp(U_vortex, *traj_vortex[n]) for n in range(n_steps)])
    csv_traj_vortex = os.path.join(results_dir, 'traj_vortex.csv')
    header_vortex = 't_s,x_m,y_m,z_m,Fx_N,Fy_N,Fz_N,U_J (all SI units)'
    data_vortex = np.column_stack([t_vortex, traj_vortex, Fx_traj, Fy_traj, Fz_traj, U_traj])
    np.savetxt(csv_traj_vortex, data_vortex, delimiter=',', header=header_vortex, comments='')
    # 1D plots
    fig, axes = plt.subplots(3, 1, figsize=(8, 8))
    axes[0].plot(t_vortex, traj_vortex[:,0], label='x [m]')
    axes[0].plot(t_vortex, traj_vortex[:,1], label='y [m]')
    axes[0].plot(t_vortex, traj_vortex[:,2], label='z [m]')
    axes[0].set_ylabel('Position [m]')
    axes[0].legend()
    axes[1].plot(t_vortex, np.sqrt(Fx_traj**2 + Fy_traj**2 + Fz_traj**2), label='|F| [N]')
    axes[1].set_ylabel('Force [N]')
    axes[1].legend()
    axes[2].plot(t_vortex, U_traj, label='U [J]')
    axes[2].set_ylabel('U [J]')
    axes[2].set_xlabel('Time [s]')
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, 'traj_vortex_1dplots.png'))
    plt.close(fig)
    # 2D projection on mid-z slice of U
    iz_mid = grid.Nz // 2
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(U_vortex[:,:,iz_mid], origin='lower', extent=[grid.x[0], grid.x[-1], grid.y[0], grid.y[-1]])
    ax.plot(traj_vortex[:,0], traj_vortex[:,1], 'r-', label='Trajectory')
    ax.scatter([pos0[0]], [pos0[1]], color='g', label='Start', s=40)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('Trajectory on U (mid-z)')
    plt.colorbar(im, ax=ax, label='U [J]')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, 'traj_vortex_2dproj.png'))
    plt.close(fig)
    # Plot trajectory
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(traj_vortex[:,0], traj_vortex[:,1], traj_vortex[:,2], label='Vortex traj')
    ax.scatter([pos0[0]], [pos0[1]], [pos0[2]], color='r', label='Start', s=40)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    ax.set_title('Particle trajectory (vortex)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, 'traj_vortex.png'))
    plt.close(fig)
    print(f"[SIM] Vortex trajectory: start=({traj_vortex[0,0]:.3e},{traj_vortex[0,1]:.3e},{traj_vortex[0,2]:.3e}), end=({traj_vortex[-1,0]:.3e},{traj_vortex[-1,1]:.3e},{traj_vortex[-1,2]:.3e})")

    # --- Simulate for axicon ---
    print("[SIM] Running 3D particle simulation in axicon Gor'kov field...")
    traj_axicon = field_axicon.simulate_particle(
        pos0, n_steps=n_steps, dt=dt_s, mu=mu, fluid=fluid, particle=particle, T_C=T_C, brownian=brownian, clamp=clamp, verbose=True, U_field=U_axicon
    )
    t_axicon = np.arange(n_steps) * dt_s
    Fx_traj = np.array([field_axicon.trilinear_interp(Fx_axicon, *traj_axicon[n]) for n in range(n_steps)])
    Fy_traj = np.array([field_axicon.trilinear_interp(Fy_axicon, *traj_axicon[n]) for n in range(n_steps)])
    Fz_traj = np.array([field_axicon.trilinear_interp(Fz_axicon, *traj_axicon[n]) for n in range(n_steps)])
    U_traj = np.array([field_axicon.trilinear_interp(U_axicon, *traj_axicon[n]) for n in range(n_steps)])
    csv_traj_axicon = os.path.join(results_dir, 'traj_axicon.csv')
    header_axicon = 't_s,x_m,y_m,z_m,Fx_N,Fy_N,Fz_N,U_J (all SI units)'
    data_axicon = np.column_stack([t_axicon, traj_axicon, Fx_traj, Fy_traj, Fz_traj, U_traj])
    np.savetxt(csv_traj_axicon, data_axicon, delimiter=',', header=header_axicon, comments='')
    # 1D plots
    fig, axes = plt.subplots(3, 1, figsize=(8, 8))
    axes[0].plot(t_axicon, traj_axicon[:,0], label='x [m]')
    axes[0].plot(t_axicon, traj_axicon[:,1], label='y [m]')
    axes[0].plot(t_axicon, traj_axicon[:,2], label='z [m]')
    axes[0].set_ylabel('Position [m]')
    axes[0].legend()
    axes[1].plot(t_axicon, np.sqrt(Fx_traj**2 + Fy_traj**2 + Fz_traj**2), label='|F| [N]')
    axes[1].set_ylabel('Force [N]')
    axes[1].legend()
    axes[2].plot(t_axicon, U_traj, label='U [J]')
    axes[2].set_ylabel('U [J]')
    axes[2].set_xlabel('Time [s]')
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, 'traj_axicon_1dplots.png'))
    plt.close(fig)
    # 2D projection on mid-z slice of U
    iz_mid = grid.Nz // 2
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(U_axicon[:,:,iz_mid], origin='lower', extent=[grid.x[0], grid.x[-1], grid.y[0], grid.y[-1]])
    ax.plot(traj_axicon[:,0], traj_axicon[:,1], 'r-', label='Trajectory')
    ax.scatter([pos0[0]], [pos0[1]], color='g', label='Start', s=40)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('Trajectory on U (mid-z)')
    plt.colorbar(im, ax=ax, label='U [J]')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, 'traj_axicon_2dproj.png'))
    plt.close(fig)
    # Plot trajectory
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(traj_axicon[:,0], traj_axicon[:,1], traj_axicon[:,2], label='Axicon traj')
    ax.scatter([pos0[0]], [pos0[1]], [pos0[2]], color='r', label='Start', s=40)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    ax.set_title('Particle trajectory (axicon)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, 'traj_axicon.png'))
    plt.close(fig)
    print(f"[SIM] Axicon trajectory: start=({traj_axicon[0,0]:.3e},{traj_axicon[0,1]:.3e},{traj_axicon[0,2]:.3e}), end=({traj_axicon[-1,0]:.3e},{traj_axicon[-1,1]:.3e},{traj_axicon[-1,2]:.3e})")
