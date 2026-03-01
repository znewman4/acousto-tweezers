#!/usr/bin/env python3
"""
Worker script: solve ONE FEM case, interpolate to MULTIPLE z-planes + XZ slice.

Extension of _solve_worker.py for convergence spot-check.
Reads args from a JSON file, solves, interpolates to grids, saves .npz.

Usage:  python _solve_worker_multi_z.py /path/to/args.json

Expected JSON fields:
  overrides: dict of FarFieldConfig kwargs
  label: str
  trap_z_list: list[float]   # multiple z-plane heights
  mid_y: float               # for XZ slice
  n_xy: int                  # grid resolution
  result_file: str           # output .npz path
"""
from __future__ import annotations
import sys, os, time, json
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

NTHREADS = str(min(os.cpu_count() or 4, 14))
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, NTHREADS)

from acoustweezers.experiments.farfield_petri_cuboid.config import FarFieldConfig
from acoustweezers.experiments.farfield_petri_cuboid.solve_pressure import solve_helmholtz
from acoustweezers.experiments.farfield_petri_cuboid.presets import PETSC_MUMPS
from scipy.interpolate import NearestNDInterpolator

PETSC_OPTS = {
    **PETSC_MUMPS,
    "mat_mumps_icntl_14": "100",
    "mat_mumps_icntl_23": "0",
}


def main():
    args_file = sys.argv[1]
    with open(args_file) as f:
        args = json.load(f)

    overrides   = args["overrides"]
    label       = args["label"]
    trap_z_list = args["trap_z_list"]
    mid_y       = args["mid_y"]
    n_xy        = args["n_xy"]
    result_file = args["result_file"]

    # ── Solve ─────────────────────────────────────────────────────
    cfg = FarFieldConfig(**overrides)
    t0 = time.time()
    sol = solve_helmholtz(cfg, verbose=True, petsc_options=PETSC_OPTS)
    solve_time = time.time() - t0

    # ── Extract physical-domain DOFs (filter PML) ────────────────
    coords = sol.coords.copy()
    p_vals = sol.p_values.copy()
    ksp_reason = int(sol.ksp_converged_reason)

    t_xy = cfg.t_pml_xy
    t_z  = cfg.t_pml_z
    H_under = cfg.H_under

    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    in_pml_x = ((x < t_xy) | (x > cfg.Lx - t_xy)) & (z < H_under)
    in_pml_y = ((y < t_xy) | (y > cfg.Ly - t_xy)) & (z < H_under)
    in_pml_z = z < t_z
    is_physical = ~(in_pml_x | in_pml_y | in_pml_z)

    phys_coords = coords[is_physical]
    phys_p = p_vals[is_physical]
    phys_max = float(np.abs(phys_p).max())

    # ── Build 3D interpolators (once) ────────────────────────────
    interp_re = NearestNDInterpolator(phys_coords, np.real(phys_p))
    interp_im = NearestNDInterpolator(phys_coords, np.imag(phys_p))

    xg = np.linspace(t_xy, cfg.Lx - t_xy, n_xy)
    yg = np.linspace(t_xy, cfg.Ly - t_xy, n_xy)

    # ── Interpolate to XY grids at each z-plane ──────────────────
    save_dict = dict(
        xg=xg, yg=yg,
        phys_max=np.array(phys_max),
        solve_time=np.array(solve_time),
        trap_z_list=np.array(trap_z_list),
    )

    X, Y = np.meshgrid(xg, yg)
    for i, tz in enumerate(trap_z_list):
        pts_xy = np.column_stack([X.ravel(), Y.ravel(),
                                   np.full(X.size, tz)])
        p_xy = (interp_re(pts_xy) + 1j * interp_im(pts_xy)).reshape(X.shape)
        save_dict[f"p_xy_{i}"] = p_xy

    # ── Interpolate to XZ grid at mid_y ──────────────────────────
    xg_xz = xg.copy()
    zg = np.linspace(t_z, cfg.H_total, n_xy)
    X_xz, Z_xz = np.meshgrid(xg_xz, zg)
    pts_xz = np.column_stack([X_xz.ravel(),
                                np.full(X_xz.size, mid_y),
                                Z_xz.ravel()])
    p_xz = (interp_re(pts_xz) + 1j * interp_im(pts_xz)).reshape(X_xz.shape)

    save_dict["xg_xz"] = xg_xz
    save_dict["zg_xz"] = zg
    save_dict["p_xz"] = p_xz

    # ── Report ────────────────────────────────────────────────────
    z_str = ", ".join(f"{tz*1e3:.3f}" for tz in trap_z_list)
    print(f"  [{label}] phys max|p|={phys_max:.3f} Pa  "
          f"KSP={ksp_reason}  {solve_time:.1f}s  "
          f"grid {n_xy}  z-planes=[{z_str}] mm",
          flush=True)

    # ── Save ──────────────────────────────────────────────────────
    np.savez(result_file, **save_dict)


if __name__ == "__main__":
    main()
