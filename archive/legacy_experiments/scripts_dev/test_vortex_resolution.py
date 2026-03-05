#!/usr/bin/env python3
"""
Quick diagnostic: solve a single BG vortex case at EPL=3 vs EPL=6
and check phase winding at the observation plane.
"""
import sys, os, time
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

NTHREADS = str(min(os.cpu_count() or 4, 14))
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, NTHREADS)

from acoustweezers.experiments.farfield_petri_cuboid.config import FarFieldConfig
from acoustweezers.experiments.farfield_petri_cuboid.solve_pressure import solve_helmholtz
from acoustweezers.experiments.farfield_petri_cuboid.presets import CORRECTED_PRESET, PETSC_MUMPS
from scipy.interpolate import NearestNDInterpolator, RegularGridInterpolator

PETSC_OPTS = {**PETSC_MUMPS, "mat_mumps_icntl_14": "100", "mat_mumps_icntl_23": "0"}

# Physics
LAM = 1484.0 / 2e6  # 0.742 mm
K_WATER = 2 * np.pi * 2e6 / 1484.0
H_UNDER = CORRECTED_PRESET["H_under"]
H_TOP = CORRECTED_PRESET.get("H_top", 2.0085e-3)
Z_MID = H_UNDER + H_TOP / 2
Z_STAR = Z_MID + 0.25 * LAM

# BG beam config: ℓ=2, R=1mm, w=0.4mm
overrides = {
    **CORRECTED_PRESET,
    "standing_velocity_amplitude": 0.0,
    "disk_velocity_amplitude": 1e-6,
    "lens_drive": "bessel_gauss",
    "lens_l": 2,
    "lens_k_r": 0.5 * K_WATER,
    "lens_beam_waist": 0.4e-3,
    "disk_radius": 1.0e-3,
    "lens_apodization": "cosine_taper",
    "lens_focus_offset_x": 0.0,
    "lens_focus_offset_y": 0.0,
}


def solve_and_check(epl: int, n_xy: int = 200):
    ov = {**overrides, "elements_per_wavelength": epl}
    cfg = FarFieldConfig(**ov)

    print(f"\n{'='*60}")
    print(f"Solving at EPL={epl}")
    print(f"  mesh: {cfg.mesh_nx} x {cfg.mesh_ny} x {cfg.mesh_nz}")
    print(f"{'='*60}")

    t0 = time.time()
    sol = solve_helmholtz(cfg, verbose=True, petsc_options=PETSC_OPTS)
    dt = time.time() - t0
    print(f"  Solve time: {dt:.1f} s")

    # Extract physical DOFs
    coords = sol.coords.copy()
    p_vals = sol.p_values.copy()
    t_xy = cfg.t_pml_xy
    t_z = cfg.t_pml_z
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    in_pml_x = ((x < t_xy) | (x > cfg.Lx - t_xy)) & (z < cfg.H_under)
    in_pml_y = ((y < t_xy) | (y > cfg.Ly - t_xy)) & (z < cfg.H_under)
    in_pml_z = z < t_z
    is_physical = ~(in_pml_x | in_pml_y | in_pml_z)
    phys_coords = coords[is_physical]
    phys_p = p_vals[is_physical]
    print(f"  Physical DOFs: {phys_coords.shape[0]}")
    print(f"  Max |p|: {np.abs(phys_p).max():.4f}")

    # Interpolate to XY grid at Z_STAR
    interp_re = NearestNDInterpolator(phys_coords, np.real(phys_p))
    interp_im = NearestNDInterpolator(phys_coords, np.imag(phys_p))
    xg = np.linspace(t_xy, cfg.Lx - t_xy, n_xy)
    yg = np.linspace(t_xy, cfg.Ly - t_xy, n_xy)
    X, Y = np.meshgrid(xg, yg)
    pts = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, Z_STAR)])
    p_xy = (interp_re(pts) + 1j * interp_im(pts)).reshape(X.shape)

    # Phase winding test
    cx_m, cy_m = cfg.disk_center_x, cfg.disk_center_y
    interp2d = RegularGridInterpolator((yg, xg), p_xy, method='linear',
                                        bounds_error=False, fill_value=0.0)

    print(f"\n  Phase winding (expected: 2.0 for ℓ=2):")
    print(f"  |p| at center: {np.abs(p_xy[n_xy//2, n_xy//2]):.6f}")
    for r_lam in [0.5, 1.0, 1.5, 2.0]:
        r_m = r_lam * LAM
        thetas = np.linspace(0, 2 * np.pi, 128, endpoint=False)
        pts_x = cx_m + r_m * np.cos(thetas)
        pts_y = cy_m + r_m * np.sin(thetas)
        p_ring = interp2d(np.column_stack([pts_y, pts_x]))
        dph = np.diff(np.angle(p_ring))
        dph = (dph + np.pi) % (2 * np.pi) - np.pi
        winding = np.sum(dph) / (2 * np.pi)
        amp = np.abs(p_ring)
        print(f"    r={r_lam:.1f}λ: winding={winding:+.3f}  "
              f"mean|p|={amp.mean():.6f}  min/max={amp.min():.6f}/{amp.max():.6f}")

    return p_xy, xg, yg


if __name__ == "__main__":
    # EPL=3 is the max for this machine. Check both EPL=3 and examine the BC.
    p3, xg3, yg3 = solve_and_check(epl=3)
    print("\nDone.")
