import numpy as np

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from tweezers.grid.grid3d import Grid3D
from tweezers.control.field_interface_3d import Helmholtz3DSolver
from tweezers.control import bottom_drives

if __name__ == "__main__":
    # Grid parameters (small for speed)
    Lx, Ly, H = 0.006, 0.006, 0.003  # 6mm x 6mm x 3mm
    dx = dy = dz = 0.001  # 1mm spacing
    grid = Grid3D(Lx, Ly, H, dx, dy, dz)

    # Physical parameters
    omega = 2 * np.pi * 1e6  # 1 MHz
    class MediumProps:
        c0 = 1500.0
        rho0 = 1000.0
    medium = MediumProps()
    k = omega / medium.c0

    # Choose analytic drive: plane wave and gaussian
    theta = np.pi / 6
    kx = k * np.cos(theta)
    ky = k * np.sin(theta)
    p_bot_plane = bottom_drives.drive_plane_wave(grid, kx, ky, amp=1.0)
    x0 = 0.5 * (grid.x[0] + grid.x[-1])
    y0 = 0.5 * (grid.y[0] + grid.y[-1])
    sigma = 0.001
    p_bot_gauss = bottom_drives.drive_gaussian(grid, x0, y0, sigma, amp=1.0)

    # Output directory for results
    import csv
    results_dir = os.path.join(os.path.dirname(__file__), '../results/helmholtz3d_demo')
    os.makedirs(results_dir, exist_ok=True)

    # Solve for plane wave
    solver = Helmholtz3DSolver(grid, omega, medium)
    field = solver.solve(p_bot_plane)
    metrics = field.diagnostics(p_bot=p_bot_plane)

    # Save diagnostics to CSV
    csv_path_plane = os.path.join(results_dir, 'diagnostics_plane_wave.csv')
    with open(csv_path_plane, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        for key, v in metrics.items():
            writer.writerow([key, v])
    print("Diagnostics for plane wave drive:")
    for key, v in metrics.items():
        print(f"  {key}: {v:.2e}")

    # Debug: print a representative Robin matrix row and field values
    from tweezers.control.fd_helmholtz_3d import Helmholtz3DOperator
    op = Helmholtz3DOperator(grid, k)
    A, b = op.assemble_system(p_bot_plane)
    # Top face, interior (ix=1, iy=1, iz=Nz-1)
    ix, iy, iz = 1, 1, grid.Nz-1
    idx = op._flatten_index(ix, iy, iz)
    row = A.getrow(idx).toarray().ravel()
    nonzero_idx = np.nonzero(row)[0]
    print(f"\n[DEBUG] Top face node (ix={ix}, iy={iy}, iz={iz}) matrix row nonzeros:")
    for j in nonzero_idx:
        print(f"  col={j}, value={row[j]:.3g}")
    print(f"  b[idx]={b[idx]:.3g}")
    print(f"  p_b={field.p[ix,iy,iz]:.3g}, p_in={field.p[ix,iy,iz-1]:.3g}")
    lhs = row[idx]*field.p[ix,iy,iz] + row[op._flatten_index(ix,iy,iz-1)]*field.p[ix,iy,iz-1]
    print(f"  Row equation: lhs={lhs:.3g}, rhs={b[idx]:.3g}, residual={lhs-b[idx]:.3g}")

    # --- Enhanced diagnostics for plane wave ---
    metrics_strict = field.diagnostics(p_bot=p_bot_plane, operator=op, debug_Ab=True, face_mode='strict')
    metrics_all = field.diagnostics(p_bot=p_bot_plane, operator=op, debug_Ab=True, face_mode='all')
    print("\n[PASS/FAIL] Robin and boundary residuals (plane wave):")
    for key in sorted(metrics_strict.keys()):
        if 'robin_row' in key or 'lsys_resid' in key:
            print(f"  {key}: {metrics_strict[key]:.2e} (strict)")
    for key in sorted(metrics_all.keys()):
        if 'robin_row' in key:
            print(f"  {key}: {metrics_all[key]:.2e} (all)")

    # Solve for gaussian
    field_g = solver.solve(p_bot_gauss)
    metrics_g = field_g.diagnostics(p_bot=p_bot_gauss)

    csv_path_gauss = os.path.join(results_dir, 'diagnostics_gaussian.csv')
    with open(csv_path_gauss, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        for key, v in metrics_g.items():
            writer.writerow([key, v])
    print("\nDiagnostics for gaussian drive:")
    for key, v in metrics_g.items():
        print(f"  {key}: {v:.2e}")

    # --- Enhanced diagnostics for gaussian ---
    op_g = Helmholtz3DOperator(grid, k)
    metrics_g_strict = field_g.diagnostics(p_bot=p_bot_gauss, operator=op_g, debug_Ab=True, face_mode='strict')
    metrics_g_all = field_g.diagnostics(p_bot=p_bot_gauss, operator=op_g, debug_Ab=True, face_mode='all')
    print("\n[PASS/FAIL] Robin and boundary residuals (gaussian):")
    for key in sorted(metrics_g_strict.keys()):
        if 'robin_row' in key or 'lsys_resid' in key:
            print(f"  {key}: {metrics_g_strict[key]:.2e} (strict)")
    for key in sorted(metrics_g_all.keys()):
        if 'robin_row' in key:
            print(f"  {key}: {metrics_g_all[key]:.2e} (all)")

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

    # --- Vortex drive ---
    ell = 1
    p_bot_vortex = bottom_drives.drive_vortex(grid, x0, y0, ell, sigma, amp=1.0)
    # Plot |p_bot| and phase for vortex
    fig, axes = plt.subplots(1,2,figsize=(8,4))
    im0 = axes[0].imshow(np.abs(p_bot_vortex), origin='lower', extent=[grid.x[0], grid.x[-1], grid.y[0], grid.y[-1]])
    axes[0].set_title('|p_bot| (vortex)')
    plt.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(np.angle(p_bot_vortex), origin='lower', extent=[grid.x[0], grid.x[-1], grid.y[0], grid.y[-1]], cmap='twilight')
    axes[1].set_title('arg(p_bot) (vortex)')
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
    op_vortex = Helmholtz3DOperator(grid, k)
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

    # --- Axicon drive ---
    alpha = 3000.0  # rad/m
    p_bot_axicon = bottom_drives.drive_axicon(grid, x0, y0, alpha, sigma, amp=1.0)
    # Plot |p_bot| and phase for axicon
    fig, axes = plt.subplots(1,2,figsize=(8,4))
    im0 = axes[0].imshow(np.abs(p_bot_axicon), origin='lower', extent=[grid.x[0], grid.x[-1], grid.y[0], grid.y[-1]])
    axes[0].set_title('|p_bot| (axicon)')
    plt.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(np.angle(p_bot_axicon), origin='lower', extent=[grid.x[0], grid.x[-1], grid.y[0], grid.y[-1]], cmap='twilight')
    axes[1].set_title('arg(p_bot) (axicon)')
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
    op_axicon = Helmholtz3DOperator(grid, k)
    metrics_axicon = field_axicon.diagnostics(p_bot=p_bot_axicon, operator=op_axicon, debug_Ab=True, face_mode='strict')
    csv_path_axicon = os.path.join(results_dir, 'diagnostics_axicon_full.csv')
    with open(csv_path_axicon, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        for key, v in metrics_axicon.items():
            writer.writerow([key, v])
    # Axicon phase check: fit phase vs r
    X, Y = np.meshgrid(grid.x, grid.y, indexing='ij')
    r_grid = np.sqrt((X-x0)**2 + (Y-y0)**2)
    mask = (r_grid > 0.001) & (r_grid < 0.0025)
    phase = np.angle(p_bot_axicon[mask])
    rvals = r_grid[mask]
    from numpy.polynomial.polynomial import polyfit
    slope, intercept = polyfit(rvals, np.unwrap(phase), 1)
    print(f"[AXICON PHASE CHECK] Fitted phase slope: {slope:.1f} (expected {alpha:.1f})")

    # --- Gor'kov and force for vortex ---
    a = 2e-6  # particle radius [m]
    f1 = 1.0  # contrast factors (example)
    f2 = 0.5
    U_vortex = field_vortex.compute_gorkov_potential(a, f1, f2)
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
    U_axicon = field_axicon.compute_gorkov_potential(a, f1, f2)
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
