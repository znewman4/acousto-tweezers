
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import csv
from datetime import datetime
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from tweezers.grid.grid3d import Grid3D
from tweezers.actuation.lens_fields import lens_focus
from tweezers.actuation.bath_propagation import angular_spectrum_propagate
from tweezers.actuation.plate_transmission import apply_plate_transmission

def rms(x):
    return np.sqrt(np.mean(np.abs(x)**2))

def high_k_fraction(p, dx, dy, k_cut):
    P = np.fft.fft2(p)
    ny, nx = p.shape
    fx = np.fft.fftfreq(nx, d=dx)
    fy = np.fft.fftfreq(ny, d=dy)
    FX, FY = np.meshgrid(fx, fy)
    K = 2 * np.pi * np.sqrt(FX**2 + FY**2)
    total = np.sum(np.abs(P)**2)
    high = np.sum(np.abs(P)[K > k_cut]**2)
    return high / total if total > 0 else 0.0

def save_field_plots(p, x, y, out_prefix, title):
    fig, ax = plt.subplots(1,2,figsize=(8,4))
    im0 = ax[0].imshow(np.abs(p), origin='lower', extent=[x[0], x[-1], y[0], y[-1]])
    ax[0].set_title(f'|p| {title}')
    plt.colorbar(im0, ax=ax[0])
    im1 = ax[1].imshow(np.angle(p), origin='lower', extent=[x[0], x[-1], y[0], y[-1]], cmap='twilight')
    ax[1].set_title(f'arg(p) {title}')
    plt.colorbar(im1, ax=ax[1])
    fig.tight_layout()
    fig.savefig(f'{out_prefix}.png')
    plt.close(fig)

def main():
    # Output dir
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join('results', 'actuation_validation', ts)
    os.makedirs(out_dir, exist_ok=True)

    # Grid and params
    Lx = 0.01; Ly = 0.01; Nx = 64; Ny = 64
    dx = Lx / (Nx - 1); dy = Ly / (Ny - 1)
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    omega = 2 * np.pi * 1e6
    c0 = 1500.0
    k = omega / c0
    # Plate props
    rho = 1000.0
    rho_plate = 1180.0
    c_plate = 2730.0
    # Test cases
    focus_params = dict(x0=Lx/2, y0=Ly/2, f=0.01, sigma=0.002, k=k)
    d_sweep = [0, 1e-3, 3e-3, 1e-2]
    t_sweep = [0, 0.5e-3, 1e-3, 2e-3]
    # --- Distance sweep ---
    with open(os.path.join(out_dir, 'distance_sweep.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['d_m','rms_p_lens','rms_p_bath','rms_p_bot','energy_ratio','high_k_bath','high_k_bot'])
        for d in d_sweep:
            p_lens = lens_focus(X, Y, **focus_params)
            p_bath = angular_spectrum_propagate(p_lens, dx, dy, d, k)
            t = 1e-3  # fixed plate thickness
            k_plate = k * c0 / c_plate
            p_bot = apply_plate_transmission(p_bath, dx, dy, k, k_plate, rho, rho_plate, c0, c_plate, t)
            save_field_plots(p_lens, x, y, os.path.join(out_dir, f'focus_d{d*1e3:.1f}mm_lens'), f'lens d={d*1e3:.1f}mm')
            save_field_plots(p_bath, x, y, os.path.join(out_dir, f'focus_d{d*1e3:.1f}mm_bath'), f'bath d={d*1e3:.1f}mm')
            save_field_plots(p_bot, x, y, os.path.join(out_dir, f'focus_d{d*1e3:.1f}mm_bot'), f'bot d={d*1e3:.1f}mm')
            rms_lens = rms(p_lens)
            rms_bath = rms(p_bath)
            rms_bot = rms(p_bot)
            energy_ratio = np.sum(np.abs(p_bot)**2)/np.sum(np.abs(p_bath)**2)
            k_cut = 2*k/3
            highk_bath = high_k_fraction(p_bath, dx, dy, k_cut)
            highk_bot = high_k_fraction(p_bot, dx, dy, k_cut)
            writer.writerow([d, rms_lens, rms_bath, rms_bot, energy_ratio, highk_bath, highk_bot])
    # --- Plate thickness sweep ---
    with open(os.path.join(out_dir, 'plate_thickness_sweep.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['t_m','rms_p_lens','rms_p_bath','rms_p_bot','energy_ratio','high_k_bath','high_k_bot'])
        for t in t_sweep:
            d = 3e-3  # fixed bath distance
            p_lens = lens_focus(X, Y, **focus_params)
            p_bath = angular_spectrum_propagate(p_lens, dx, dy, d, k)
            k_plate = k * c0 / c_plate
            p_bot = apply_plate_transmission(p_bath, dx, dy, k, k_plate, rho, rho_plate, c0, c_plate, t)
            save_field_plots(p_lens, x, y, os.path.join(out_dir, f'focus_t{t*1e3:.1f}mm_lens'), f'lens t={t*1e3:.1f}mm')
            save_field_plots(p_bath, x, y, os.path.join(out_dir, f'focus_t{t*1e3:.1f}mm_bath'), f'bath t={t*1e3:.1f}mm')
            save_field_plots(p_bot, x, y, os.path.join(out_dir, f'focus_t{t*1e3:.1f}mm_bot'), f'bot t={t*1e3:.1f}mm')
            rms_lens = rms(p_lens)
            rms_bath = rms(p_bath)
            rms_bot = rms(p_bot)
            energy_ratio = np.sum(np.abs(p_bot)**2)/np.sum(np.abs(p_bath)**2)
            k_cut = 2*k/3
            highk_bath = high_k_fraction(p_bath, dx, dy, k_cut)
            highk_bot = high_k_fraction(p_bot, dx, dy, k_cut)
            writer.writerow([t, rms_lens, rms_bath, rms_bot, energy_ratio, highk_bath, highk_bot])
    # --- Plane wave normal incidence test ---
    p_lens = np.ones((Nx, Ny), dtype=np.complex128)
    d = 3e-3
    t = 1e-3
    p_bath = angular_spectrum_propagate(p_lens, dx, dy, d, k)
    k_plate = k * c0 / c_plate
    p_bot = apply_plate_transmission(p_bath, dx, dy, k, k_plate, rho, rho_plate, c0, c_plate, t)
    ratio = p_bot / p_bath
    mag = np.abs(ratio)
    phase = np.angle(ratio)
    print("[PLANE WAVE TEST] p_bot/p_bath magnitude: mean {:.3f}, std {:.3e}".format(np.mean(mag), np.std(mag)))
    print("[PLANE WAVE TEST] p_bot/p_bath phase: mean {:.3f}, std {:.3e}".format(np.mean(phase), np.std(phase)))
    with open(os.path.join(out_dir, 'plane_wave_test.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['mean_mag','std_mag','mean_phase','std_phase'])
        writer.writerow([np.mean(mag), np.std(mag), np.mean(phase), np.std(phase)])
    save_field_plots(p_lens, x, y, os.path.join(out_dir, 'plane_lens'), 'plane lens')
    save_field_plots(p_bath, x, y, os.path.join(out_dir, 'plane_bath'), 'plane bath')
    save_field_plots(p_bot, x, y, os.path.join(out_dir, 'plane_bot'), 'plane bot')
    # Pass/fail indicator
    if np.std(mag) < 0.01 and np.std(phase) < 0.01:
        print("[PASS] Plane wave transmission is spatially uniform (as expected)")
    else:
        print("[FAIL] Plane wave transmission is NOT spatially uniform")

if __name__ == "__main__":
    main()
