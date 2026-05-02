sa#!/usr/bin/env python3
"""
Vortex Function Audit — Comprehensive Analysis
================================================

Answers 20 questions across 6 sections:
  §1  Field integrity  (lattice disruption by the vortex)
  §2  Local authority   (can the vortex move a particle?)
  §3  Energy localisation
  §4  Parameter sensitivity (α, ℓ, Δf sweeps)
  §5  Lens design diagnostics (phase winding, apodisation)
  §6  Ranking & decision

Uses **superposition** to avoid redundant combined solves:
  p_combined(α) = p_stand + α·(V_stand/V_vortex)·p_vortex

Solves needed:
  • standing_only  (1 per frequency)
  • vortex_only    (1 per {ℓ, apod, freq} config)
  ≈ 10 FEM solves total at 4 elem/λ, ~8–12 min

Output → results/vortex_audit_<timestamp>/
"""

from __future__ import annotations
import sys, os, time, json, gc, warnings, ctypes
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

NTHREADS = str(min(os.cpu_count() or 4, 14))
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, NTHREADS)

def _hard_gc():
    """Aggressively free memory (PETSc/MUMPS leaks)."""
    gc.collect(); gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = PROJECT_ROOT / "results" / f"vortex_audit_{TIMESTAMP}"
FIG_DIR = OUT_DIR / "figures"
CSV_DIR = OUT_DIR / "csv"
FIG_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)
print(f"Output → {OUT_DIR}")

from acoustweezers.experiments.farfield_petri_cuboid.presets import (
    CORRECTED_PRESET, PETSC_MUMPS,
)
from acoustweezers.experiments.farfield_petri_cuboid.config import FarFieldConfig
from acoustweezers.experiments.farfield_petri_cuboid.solve_pressure import solve_helmholtz

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.gridspec import GridSpec
from scipy.interpolate import NearestNDInterpolator

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200,
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
})

# =====================================================================
# Constants
# =====================================================================
ELEM_PER_LAMBDA = 4
H_TOP_OPTIMAL = 2.0085e-3
NGRID_XY = 400          # fine 2D grid
NGRID_3D = 80           # coarser 3D grid per axis (for Gor'kov)

PETSC_OPTS = {
    **PETSC_MUMPS,
    "mat_mumps_icntl_14": "100",
    "mat_mumps_icntl_23": "0",
}

COMMON = {
    **CORRECTED_PRESET,
    "H_top": H_TOP_OPTIMAL,
    "elements_per_wavelength": ELEM_PER_LAMBDA,
}

# ── Particle properties (polystyrene in water) ───────────────────
PARTICLE_RADIUS    = 5e-6       # 5 µm
PARTICLE_DENSITY   = 1050.0     # kg/m³
PARTICLE_C         = 2350.0     # m/s
WATER_DENSITY      = 997.0
WATER_C            = 1484.0
WATER_VISCOSITY    = 1.002e-3   # Pa·s

_kappa_p = 1 / (PARTICLE_DENSITY * PARTICLE_C**2)
_kappa_f = 1 / (WATER_DENSITY * WATER_C**2)
F1_CONTRAST = 1 - _kappa_p / _kappa_f          # ~0.62
F2_CONTRAST = 2 * (PARTICLE_DENSITY - WATER_DENSITY) / \
              (2 * PARTICLE_DENSITY + WATER_DENSITY)  # ~0.034
STOKES_MOBILITY = 1 / (6 * np.pi * WATER_VISCOSITY * PARTICLE_RADIUS)

# Amplitude ratio sweep values
ALPHA_VALUES = [0.05, 0.1, 0.2, 0.4, 0.8]

# Topological charge sweep
ELL_VALUES = [1, 2, 3]

# Frequency detune sweep (Hz)
DELTA_F_VALUES = [-10_000, 0, +10_000]

# Apodization variants
APOD_VALUES = ["cosine_taper", "uniform", "tukey"]


# =====================================================================
# §0  Data containers and helpers
# =====================================================================

@dataclass
class SnapShot:
    """Lightweight snapshot of a FEniCSx solution on physical-domain DOFs."""
    label: str
    coords: np.ndarray       # (N, 3)
    p_values: np.ndarray     # (N,) complex
    cfg: FarFieldConfig
    dofs: int
    max_pressure: float
    phys_max_pressure: float
    ksp_reason: int
    solve_time: float
    # Physical domain bounds
    phys_xmin: float; phys_xmax: float
    phys_ymin: float; phys_ymax: float
    phys_zmin: float; phys_zmax: float
    # Physical-only arrays (PML filtered)
    phys_coords: np.ndarray
    phys_p: np.ndarray


def _build_snapshot(sol, label: str) -> SnapShot:
    """Extract a SnapShot from a PressureSolution, filtering PML DOFs."""
    coords = sol.coords.copy()
    p_vals = sol.p_values.copy()
    cfg = sol.cfg
    t_xy = cfg.t_pml_xy
    t_z  = cfg.t_pml_z
    H_under = cfg.H_under

    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    # Lateral PML only below petri (matches solver z-filter fix)
    in_pml_x = ((x < t_xy) | (x > cfg.Lx - t_xy)) & (z < H_under)
    in_pml_y = ((y < t_xy) | (y > cfg.Ly - t_xy)) & (z < H_under)
    in_pml_z = z < t_z
    is_physical = ~(in_pml_x | in_pml_y | in_pml_z)

    phys_coords = coords[is_physical]
    phys_p = p_vals[is_physical]

    return SnapShot(
        label=label,
        coords=coords, p_values=p_vals, cfg=cfg,
        dofs=sol.dofs,
        max_pressure=sol.max_pressure,
        phys_max_pressure=float(np.abs(phys_p).max()),
        ksp_reason=sol.ksp_converged_reason,
        solve_time=sol.solver_time,
        phys_xmin=t_xy, phys_xmax=cfg.Lx - t_xy,
        phys_ymin=t_xy, phys_ymax=cfg.Ly - t_xy,
        phys_zmin=t_z,  phys_zmax=cfg.H_total,
        phys_coords=phys_coords, phys_p=phys_p,
    )


def solve_and_snap(overrides: dict, label: str) -> SnapShot:
    """Solve Helmholtz and return a SnapShot (frees the FEM solution)."""
    cfg = FarFieldConfig(**overrides)
    t0 = time.time()
    sol = solve_helmholtz(cfg, verbose=True, petsc_options=PETSC_OPTS)
    dt = time.time() - t0
    snap = _build_snapshot(sol, label)
    # Aggressively free FEniCSx / PETSc memory
    try:
        sol.p_function.x.destroy()
    except Exception:
        pass
    del sol
    _hard_gc()
    print(f"  [{label}] phys max|p|={snap.phys_max_pressure:.3f} Pa  "
          f"KSP={snap.ksp_reason}  {dt:.1f}s")
    return snap


WORKER_SCRIPT = str(PROJECT_ROOT / "scripts" / "experiments" / "_solve_worker.py")


def solve_to_grid(overrides: dict, label: str,
                  trap_z: float, mid_y: float, n_xy: int = NGRID_XY
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                             np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Solve Helmholtz in a **subprocess** → load grids from .npz.

    Each FEM solve runs in a separate process so ALL PETSc/MUMPS memory
    is returned to the OS when the child exits.  This prevents the
    glibc heap-fragmentation OOM that occurs when doing >3 sequential
    MUMPS solves in a single process on a 7.5 GB machine.

    Returns (xg, yg, p_xy, xg_xz, zg_xz, p_xz, phys_max, solve_time)
    """
    import subprocess

    args_file   = str(OUT_DIR / f"_args_{label}.json")
    result_file = str(OUT_DIR / f"_grid_{label}.npz")

    with open(args_file, "w") as f:
        json.dump({
            "overrides": overrides,
            "label": label,
            "trap_z": trap_z,
            "mid_y": mid_y,
            "n_xy": n_xy,
            "result_file": result_file,
        }, f)

    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, WORKER_SCRIPT, args_file],
        capture_output=False,          # let stdout/stderr flow to parent
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Solve subprocess for '{label}' failed "
                           f"(exit code {proc.returncode})")

    data = np.load(result_file)
    result = (
        data["xg"], data["yg"], data["p_xy"],
        data["xg_xz"], data["zg_xz"], data["p_xz"],
        float(data["phys_max"]), float(data["solve_time"]),
    )

    # Clean up temp files
    os.unlink(args_file)
    os.unlink(result_file)

    wall = time.time() - t0
    print(f"    subprocess wall-clock: {wall:.1f}s", flush=True)
    return result


# =====================================================================
# §0b  Grid interpolation
# =====================================================================

def interp_xy(snap: SnapShot, z_val: float, n: int = NGRID_XY
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate snapshot onto a fine XY grid at given z."""
    interp_re = NearestNDInterpolator(snap.phys_coords, np.real(snap.phys_p))
    interp_im = NearestNDInterpolator(snap.phys_coords, np.imag(snap.phys_p))
    xg = np.linspace(snap.phys_xmin, snap.phys_xmax, n)
    yg = np.linspace(snap.phys_ymin, snap.phys_ymax, n)
    X, Y = np.meshgrid(xg, yg)
    pts = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, z_val)])
    p_grid = (interp_re(pts) + 1j * interp_im(pts)).reshape(X.shape)
    return xg, yg, p_grid


def interp_xz(snap: SnapShot, y_val: float, n: int = NGRID_XY
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate onto XZ grid at given y."""
    interp_re = NearestNDInterpolator(snap.phys_coords, np.real(snap.phys_p))
    interp_im = NearestNDInterpolator(snap.phys_coords, np.imag(snap.phys_p))
    xg = np.linspace(snap.phys_xmin, snap.phys_xmax, n)
    zg = np.linspace(snap.phys_zmin, snap.phys_zmax, n)
    X, Z = np.meshgrid(xg, zg)
    pts = np.column_stack([X.ravel(), np.full(X.size, y_val), Z.ravel()])
    p_grid = (interp_re(pts) + 1j * interp_im(pts)).reshape(X.shape)
    return xg, zg, p_grid


def interp_3d_petri(snap: SnapShot, n: int = NGRID_3D, margin: float = 0.1e-3
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate onto a 3D grid inside the petri slab.
    Returns (xg, yg, zg, p_3d)  with shape (nz, ny, nx)."""
    cfg = snap.cfg
    # In petri slab, full lateral extent is physical (no lateral PML)
    xmin, xmax = margin, cfg.Lx - margin
    ymin, ymax = margin, cfg.Ly - margin
    zmin = cfg.H_under + margin
    zmax = cfg.H_total - margin
    xg = np.linspace(xmin, xmax, n)
    yg = np.linspace(ymin, ymax, n)
    nz = max(8, int(n * (zmax - zmin) / (xmax - xmin)))
    zg = np.linspace(zmin, zmax, nz)

    # Build 3D grid points
    X, Y, Z = np.meshgrid(xg, yg, zg, indexing='ij')  # (nx, ny, nz)
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    interp_re = NearestNDInterpolator(snap.phys_coords, np.real(snap.phys_p))
    interp_im = NearestNDInterpolator(snap.phys_coords, np.imag(snap.phys_p))
    p_3d = (interp_re(pts) + 1j * interp_im(pts)).reshape(X.shape)
    return xg, yg, zg, p_3d


# =====================================================================
# §0c  Gor'kov potential and radiation force on 2D grid
# =====================================================================

def gorkov_2d(p_grid: np.ndarray, dx: float, dy: float,
              omega: float, rho: float = WATER_DENSITY, c: float = WATER_C,
              a: float = PARTICLE_RADIUS) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Gor'kov potential and radiation force on a 2D grid.

    Uses plane-wave approximation for |v|² ≈ |∇p|²/(ω²ρ²).
    Returns (U, Fx, Fy).
    """
    f1, f2 = F1_CONTRAST, F2_CONTRAST
    K = rho * c**2  # bulk modulus

    # |p|² / 2  (time-averaged)
    p2_avg = np.abs(p_grid)**2 / 2

    # gradient of p (complex)
    dp_dx = np.gradient(p_grid, dx, axis=1)  # axis=1 is x
    dp_dy = np.gradient(p_grid, dy, axis=0)  # axis=0 is y
    grad_p_mag2 = np.abs(dp_dx)**2 + np.abs(dp_dy)**2

    # |v|² / 2 = |∇p|² / (2 ω² ρ²)
    v2_avg = grad_p_mag2 / (2 * omega**2 * rho**2)

    # Gor'kov potential
    prefactor = (4 * np.pi / 3) * a**3
    U = prefactor * (f1 * p2_avg / (2 * K) - f2 * (3 * rho / 4) * v2_avg)

    # Force F = -∇U
    Fx = -np.gradient(U, dx, axis=1)
    Fy = -np.gradient(U, dy, axis=0)

    return U, Fx, Fy


def gorkov_3d(p_3d: np.ndarray, dx: float, dy: float, dz: float,
              omega: float, rho: float = WATER_DENSITY, c: float = WATER_C,
              a: float = PARTICLE_RADIUS) -> Tuple[np.ndarray, ...]:
    """
    3D Gor'kov potential.  p_3d shape (nx, ny, nz).
    Returns (U, Fx, Fy, Fz).
    """
    f1, f2 = F1_CONTRAST, F2_CONTRAST
    K = rho * c**2
    p2_avg = np.abs(p_3d)**2 / 2

    dp_dx = np.gradient(p_3d, dx, axis=0)
    dp_dy = np.gradient(p_3d, dy, axis=1)
    dp_dz = np.gradient(p_3d, dz, axis=2)
    grad_p_mag2 = np.abs(dp_dx)**2 + np.abs(dp_dy)**2 + np.abs(dp_dz)**2
    v2_avg = grad_p_mag2 / (2 * omega**2 * rho**2)

    prefactor = (4 * np.pi / 3) * a**3
    U = prefactor * (f1 * p2_avg / (2 * K) - f2 * (3 * rho / 4) * v2_avg)

    Fx = -np.gradient(U, dx, axis=0)
    Fy = -np.gradient(U, dy, axis=1)
    Fz = -np.gradient(U, dz, axis=2)
    return U, Fx, Fy, Fz


# =====================================================================
# §0d  ROI definition
# =====================================================================

def make_roi_mask(xg: np.ndarray, yg: np.ndarray, cx: float, cy: float,
                  radius: float) -> np.ndarray:
    """Circular ROI mask on a meshgrid.  Shape (ny, nx).  True = inside ROI."""
    X, Y = np.meshgrid(xg, yg)
    return (X - cx)**2 + (Y - cy)**2 <= radius**2


# =====================================================================
# §0e  Phase winding number
# =====================================================================

def winding_number(p_grid: np.ndarray, xg: np.ndarray, yg: np.ndarray,
                   cx: float, cy: float, radius: float,
                   n_angles: int = 360) -> float:
    """Compute phase winding number around a circle of given radius."""
    from scipy.interpolate import RegularGridInterpolator
    phase_grid = np.angle(p_grid)
    # RegularGridInterpolator expects (y, x) order matching grid
    interp = RegularGridInterpolator((yg, xg), phase_grid,
                                     method='nearest', bounds_error=False,
                                     fill_value=0.0)
    theta = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    px = cx + radius * np.cos(theta)
    py = cy + radius * np.sin(theta)
    pts = np.column_stack([py, px])  # (y, x) order
    phi = interp(pts)

    # Winding = sum of wrapped phase differences / 2π
    dphi = np.diff(phi)
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi  # wrap to [-π, π]
    winding = np.sum(dphi) / (2 * np.pi)
    return winding


# =====================================================================
# §0f  Node detection
# =====================================================================

def detect_nodes(p_mag: np.ndarray, threshold_frac: float = 0.05
                 ) -> np.ndarray:
    """Return boolean mask of pressure nodes (|p| < threshold_frac * max)."""
    threshold = threshold_frac * np.max(p_mag)
    return p_mag < threshold


def rms_node_shift(nodes_stand: np.ndarray, nodes_comb: np.ndarray,
                   xg: np.ndarray, yg: np.ndarray) -> float:
    """Compute RMS displacement of node centroids between two fields."""
    from scipy import ndimage
    # Label connected regions in standing
    labels_s, n_s = ndimage.label(nodes_stand)
    labels_c, n_c = ndimage.label(nodes_comb)

    if n_s == 0 or n_c == 0:
        return float('nan')

    dx = xg[1] - xg[0] if len(xg) > 1 else 1
    dy = yg[1] - yg[0] if len(yg) > 1 else 1

    # Centroids of standing nodes
    centroids_s = ndimage.center_of_mass(nodes_stand, labels_s,
                                          range(1, n_s + 1))
    centroids_s = np.array(centroids_s)  # (n_s, 2) in grid indices
    centroids_s_m = centroids_s * np.array([dy, dx]) + np.array([yg[0], xg[0]])

    # Centroids of combined nodes
    centroids_c = ndimage.center_of_mass(nodes_comb, labels_c,
                                          range(1, n_c + 1))
    centroids_c = np.array(centroids_c)
    centroids_c_m = centroids_c * np.array([dy, dx]) + np.array([yg[0], xg[0]])

    # Match each standing centroid to nearest combined centroid
    from scipy.spatial import cKDTree
    tree = cKDTree(centroids_c_m)
    dists, _ = tree.query(centroids_s_m)
    return float(np.sqrt(np.mean(dists**2)))


# =====================================================================
# §1–3  Metric computation (for a single configuration)
# =====================================================================

def compute_metrics(
    p_stand_xy: np.ndarray,   # 2D standing grid (ny, nx)
    p_vortex_xy: np.ndarray,  # 2D vortex grid
    p_comb_xy: np.ndarray,    # 2D combined grid
    xg: np.ndarray, yg: np.ndarray,
    roi_mask: np.ndarray,     # bool (ny, nx)
    cfg: FarFieldConfig,
    label: str = "",
) -> Dict:
    """Compute all §1–§3 metrics for one configuration."""
    omega = cfg.omega
    lam = cfg.wavelength
    dx = xg[1] - xg[0]
    dy = yg[1] - yg[0]

    m = {}
    m["label"] = label

    # ── §1 Field integrity ──────────────────────────────────────────
    out = ~roi_mask
    delta = p_comb_xy - p_stand_xy

    # Q1: L2 perturbation outside ROI
    norm_stand_out = np.sqrt(np.sum(np.abs(p_stand_xy[out])**2))
    norm_delta_out = np.sqrt(np.sum(np.abs(delta[out])**2))
    m["eta_out"] = float(norm_delta_out / norm_stand_out) if norm_stand_out > 0 else float('nan')

    # Q2: Correlation outside ROI (on |p| magnitudes)
    s_out = np.abs(p_stand_xy[out])
    c_out = np.abs(p_comb_xy[out])
    if np.std(s_out) > 0 and np.std(c_out) > 0:
        m["corr_out"] = float(np.corrcoef(s_out, c_out)[0, 1])
    else:
        m["corr_out"] = float('nan')

    # Q3: Node stability
    nodes_s = detect_nodes(np.abs(p_stand_xy))
    nodes_c = detect_nodes(np.abs(p_comb_xy))
    m["n_nodes_stand"]  = int(np.sum(nodes_s))
    m["n_nodes_comb"]   = int(np.sum(nodes_c))
    m["rms_node_shift_um"] = rms_node_shift(nodes_s, nodes_c, xg, yg) * 1e6

    # Q4: Phase preservation
    phase_s = np.angle(p_stand_xy[out])
    phase_c = np.angle(p_comb_xy[out])
    dphase = (phase_c - phase_s + np.pi) % (2 * np.pi) - np.pi
    m["rms_phase_shift_rad"] = float(np.sqrt(np.mean(dphase**2)))
    m["max_phase_shift_rad"] = float(np.max(np.abs(dphase)))

    # ── §2 Local authority ──────────────────────────────────────────
    roi = roi_mask

    # Q5: Peak pressures in ROI
    m["peak_vortex_in_roi_Pa"] = float(np.max(np.abs(p_vortex_xy[roi]))) if roi.any() else 0
    m["peak_comb_in_roi_Pa"]   = float(np.max(np.abs(p_comb_xy[roi])))   if roi.any() else 0
    m["peak_stand_in_roi_Pa"]  = float(np.max(np.abs(p_stand_xy[roi])))  if roi.any() else 0

    # Q6–Q8: Gor'kov potential and force
    U_s, Fx_s, Fy_s = gorkov_2d(p_stand_xy, dx, dy, omega)
    U_c, Fx_c, Fy_c = gorkov_2d(p_comb_xy,  dx, dy, omega)

    F_mag_s = np.sqrt(Fx_s**2 + Fy_s**2)
    F_mag_c = np.sqrt(Fx_c**2 + Fy_c**2)

    # Trap depth = max(U) - min(U) in ROI
    if roi.any():
        m["trap_depth_stand_J"] = float(np.max(U_s[roi]) - np.min(U_s[roi]))
        m["trap_depth_comb_J"]  = float(np.max(U_c[roi]) - np.min(U_c[roi]))
        m["delta_trap_depth_J"] = m["trap_depth_comb_J"] - m["trap_depth_stand_J"]
    else:
        m["trap_depth_stand_J"] = m["trap_depth_comb_J"] = m["delta_trap_depth_J"] = 0

    # Q7: Mean force in ROI
    m["mean_F_stand_N"] = float(np.mean(F_mag_s[roi])) if roi.any() else 0
    m["mean_F_comb_N"]  = float(np.mean(F_mag_c[roi])) if roi.any() else 0

    # Q8: Barrier height — find minimum of U_s inside ROI (trap centre)
    #     then find maximum on boundary of ROI (saddle proxy)
    if roi.any():
        # Trap min inside ROI
        roi_idx = np.where(roi)
        Umin_stand = float(np.min(U_s[roi]))
        Umin_comb  = float(np.min(U_c[roi]))

        # Approximate saddle: max of U on the ROI boundary
        from scipy.ndimage import binary_dilation
        roi_dilated = binary_dilation(roi, iterations=2)
        boundary = roi_dilated & (~roi)
        if boundary.any():
            Usaddle_stand = float(np.max(U_s[boundary]))
            Usaddle_comb  = float(np.max(U_c[boundary]))
        else:
            Usaddle_stand = float(np.max(U_s[roi]))
            Usaddle_comb  = float(np.max(U_c[roi]))

        m["barrier_stand_J"] = Usaddle_stand - Umin_stand
        m["barrier_comb_J"]  = Usaddle_comb  - Umin_comb
        m["barrier_reduction_pct"] = (
            100 * (1 - m["barrier_comb_J"] / m["barrier_stand_J"])
            if m["barrier_stand_J"] > 0 else float('nan')
        )
    else:
        m["barrier_stand_J"] = m["barrier_comb_J"] = 0
        m["barrier_reduction_pct"] = float('nan')

    # ── §3 Energy localisation ──────────────────────────────────────
    # Q9: Fraction of vortex energy in ROI
    vortex_energy = np.abs(p_vortex_xy)**2
    total_v_energy = np.sum(vortex_energy)
    roi_v_energy = np.sum(vortex_energy[roi])
    m["vortex_energy_in_roi_pct"] = float(100 * roi_v_energy / total_v_energy) if total_v_energy > 0 else 0

    # Q10: Sidelobe ratio
    if roi.any() and out.any():
        peak_in = np.max(np.abs(p_vortex_xy[roi]))
        peak_out = np.max(np.abs(p_vortex_xy[out]))
        m["sidelobe_ratio"] = float(peak_out / peak_in) if peak_in > 0 else float('inf')
    else:
        m["sidelobe_ratio"] = float('nan')

    # Q11: Vortex centroid vs trap centre
    cx_grid, cy_grid = cfg.Lx / 2, cfg.Ly / 2
    X, Y = np.meshgrid(xg, yg)
    intensity = np.abs(p_vortex_xy)**2
    if np.sum(intensity) > 0:
        vx_centroid = float(np.sum(X * intensity) / np.sum(intensity))
        vy_centroid = float(np.sum(Y * intensity) / np.sum(intensity))
    else:
        vx_centroid = vy_centroid = float('nan')
    m["vortex_centroid_x_mm"] = vx_centroid * 1e3
    m["vortex_centroid_y_mm"] = vy_centroid * 1e3
    m["centroid_offset_mm"] = float(np.sqrt(
        (vx_centroid - cx_grid)**2 + (vy_centroid - cy_grid)**2
    )) * 1e3 if not (np.isnan(vx_centroid) or np.isnan(vy_centroid)) else float('nan')

    # Store Gor'kov data for later plotting
    m["_U_stand"] = U_s
    m["_U_comb"] = U_c
    m["_Fx_s"] = Fx_s; m["_Fy_s"] = Fy_s
    m["_Fx_c"] = Fx_c; m["_Fy_c"] = Fy_c

    return m


# =====================================================================
# §0g  Plotting helpers
# =====================================================================

def plot_4panel_xz(xg, zg, p_s, p_v, p_c, cfg, label, fname):
    """4-panel XZ: standing / vortex / combined / delta."""
    delta = p_c - p_s
    panels = [
        (np.abs(p_s), f"Standing |p|", "inferno"),
        (np.abs(p_v), f"Vortex |p|",   "inferno"),
        (np.abs(p_c), f"Combined |p|", "inferno"),
        (np.abs(delta), f"Δ|p| = |combined − standing|", "magma"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (data, title, cmap) in zip(axes.ravel(), panels):
        vmax = max(data.max(), 1e-10)
        im = ax.pcolormesh(xg * 1e3, zg * 1e3, data, shading="auto",
                           cmap=cmap, vmin=0, vmax=vmax)
        ax.axhspan(cfg.H_under * 1e3, cfg.H_total * 1e3,
                   alpha=0.08, color="cyan")
        ax.axhline(cfg.H_under * 1e3, color="cyan", ls="--", lw=0.6, alpha=0.5)
        ax.set_xlabel("x [mm]"); ax.set_ylabel("z [mm]")
        ax.set_title(f"{title}\nmax = {data.max():.4f} Pa")
        plt.colorbar(im, ax=ax, label="|p| [Pa]")
    fig.suptitle(label, fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, bbox_inches="tight")
    plt.close(fig)


def plot_4panel_xy(xg, yg, p_s, p_v, p_c, label, fname,
                   roi_center=None, roi_radius=None):
    """4-panel XY: standing / vortex / combined / delta.  Returns None."""
    delta = p_c - p_s
    panels = [
        (np.abs(p_s), "Standing |p|", "inferno"),
        (np.abs(p_v), "Vortex |p|",   "inferno"),
        (np.abs(p_c), "Combined |p|", "inferno"),
        (np.abs(delta), "|Δp|", "magma"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for ax, (data, title, cmap) in zip(axes.ravel(), panels):
        vmax = max(data.max(), 1e-10)
        im = ax.pcolormesh(xg * 1e3, yg * 1e3, data, shading="auto",
                           cmap=cmap, vmin=0, vmax=vmax)
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
        ax.set_title(f"{title}   max={data.max():.4f} Pa")
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, label="|p| [Pa]")
        if roi_center and roi_radius:
            circ = plt.Circle((roi_center[0] * 1e3, roi_center[1] * 1e3),
                              roi_radius * 1e3, fill=False, ec="lime",
                              lw=1.5, ls="--", label="ROI")
            ax.add_patch(circ)
            ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(label, fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, bbox_inches="tight")
    plt.close(fig)


def plot_gorkov_panel(xg, yg, U_s, U_c, Fx_s, Fy_s, Fx_c, Fy_c,
                      label, fname, roi_center=None, roi_radius=None):
    """4-panel Gor'kov: U_stand / U_comb / F_stand / F_comb."""
    F_mag_s = np.sqrt(Fx_s**2 + Fy_s**2)
    F_mag_c = np.sqrt(Fx_c**2 + Fy_c**2)
    panels = [
        (U_s, "U (standing)", "viridis"),
        (U_c, "U (combined)", "viridis"),
        (F_mag_s, "|F| (standing)", "hot"),
        (F_mag_c, "|F| (combined)", "hot"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for ax, (data, title, cmap) in zip(axes.ravel(), panels):
        im = ax.pcolormesh(xg * 1e3, yg * 1e3, data, shading="auto",
                           cmap=cmap)
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
        ax.set_title(f"{title}")
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax)
        if roi_center and roi_radius:
            circ = plt.Circle((roi_center[0] * 1e3, roi_center[1] * 1e3),
                              roi_radius * 1e3, fill=False, ec="lime",
                              lw=1.5, ls="--")
            ax.add_patch(circ)
    # Quiver overlay (subsample)
    skip = max(1, len(xg) // 25)
    X, Y = np.meshgrid(xg, yg)
    for ax, (fx, fy) in zip(axes.ravel()[2:], [(Fx_s, Fy_s), (Fx_c, Fy_c)]):
        ax.quiver(X[::skip, ::skip] * 1e3, Y[::skip, ::skip] * 1e3,
                  fx[::skip, ::skip], fy[::skip, ::skip],
                  color='white', alpha=0.6, scale_units='inches',
                  scale=np.max(np.sqrt(fx**2 + fy**2)) * 4)
    fig.suptitle(label, fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, bbox_inches="tight")
    plt.close(fig)


def plot_phase_comparison(xg, yg, p_stand, p_comb, label, fname):
    """Phase maps: standing / combined / Δphase."""
    ph_s = np.angle(p_stand)
    ph_c = np.angle(p_comb)
    dph = (ph_c - ph_s + np.pi) % (2 * np.pi) - np.pi

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, data, title, cmap, vr in [
        (axes[0], ph_s, "Phase (standing)", "twilight", (-np.pi, np.pi)),
        (axes[1], ph_c, "Phase (combined)", "twilight", (-np.pi, np.pi)),
        (axes[2], dph,  "Δphase", "RdBu_r", (-np.pi, np.pi)),
    ]:
        im = ax.pcolormesh(xg * 1e3, yg * 1e3, data, shading="auto",
                           cmap=cmap, vmin=vr[0], vmax=vr[1])
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
        ax.set_title(title); ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, label="rad")
    fig.suptitle(label, fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# MAIN
# =====================================================================

def main():
    t_global = time.time()
    png_count = 0
    all_metrics: List[Dict] = []
    solve_log = []

    cfg_base = FarFieldConfig(**COMMON)
    lam = cfg_base.wavelength
    trap_z = cfg_base.H_under + cfg_base.H_top / 2
    cx = cfg_base.Lx / 2          # domain centre
    cy = cfg_base.Ly / 2
    roi_r = 1.5 * lam             # 1.5 λ ROI radius
    V_stand_base = COMMON["standing_velocity_amplitude"]   # 10 µm/s
    V_vortex_base = COMMON["disk_velocity_amplitude"]      # 1 µm/s
    mid_y = cy

    print("=" * 72)
    print("VORTEX FUNCTION AUDIT")
    print(f"  λ = {lam*1e3:.3f} mm   f = {cfg_base.frequency_hz/1e6:.2f} MHz")
    print(f"  Trap plane z = {trap_z*1e3:.3f} mm  (mid-petri)")
    print(f"  ROI centre = ({cx*1e3:.2f}, {cy*1e3:.2f}) mm  radius = {roi_r*1e3:.3f} mm")
    print(f"  Particle: a={PARTICLE_RADIUS*1e6:.0f} µm  f1={F1_CONTRAST:.4f}  f2={F2_CONTRAST:.4f}")
    print(f"  α sweep: {ALPHA_VALUES}")
    print(f"  ℓ sweep: {ELL_VALUES}")
    print(f"  Δf sweep: {[d/1e3 for d in DELTA_F_VALUES]} kHz")
    print("=" * 72)

    # ==================================================================
    # PHASE 1 — FEM SOLVES  (solve → grid → free, one at a time)
    # ==================================================================
    print(f"\n{'='*72}")
    print("PHASE 1: FEM Solves  (solve → grid → free)")
    print(f"{'='*72}\n")

    # ── Standing-only ─────────────────────────────────────────────
    xg, yg, g_stand_xy, xg_xz, zg_xz, g_stand_xz, _, st = \
        solve_to_grid({**COMMON, "disk_velocity_amplitude": 0.0},
                      "standing_only", trap_z, mid_y)
    solve_log.append(("standing_only", st))
    roi_mask = make_roi_mask(xg, yg, cx, cy, roi_r)
    print(f"  Grid: {g_stand_xy.shape}  ROI pixels: {roi_mask.sum()}")

    # ── Vortex-only for each ℓ ────────────────────────────────────
    g_vortex_xy = {}
    g_vortex_xz = None              # will be set for ℓ=1
    for ell in ELL_VALUES:
        label = f"vortex_l{ell}"
        _, _, gv_xy, _, _, gv_xz, _, st = \
            solve_to_grid({**COMMON, "standing_velocity_amplitude": 0.0,
                           "vortex_topological_charge": ell, "lens_l": ell},
                          label, trap_z, mid_y)
        g_vortex_xy[ell] = gv_xy
        if ell == 1:
            g_vortex_xz = gv_xz     # for XZ cross-section plots
        solve_log.append((label, st))

    # ── Frequency detune (standing + vortex at shifted freq) ──────
    g_stand_freq_xy = {0: g_stand_xy}
    g_vortex_freq_xy = {0: g_vortex_xy[1]}
    for df in DELTA_F_VALUES:
        if df == 0:
            continue
        freq = COMMON["frequency_hz"] + df
        label_s = f"standing_df{df/1e3:+.0f}kHz"
        _, _, gs_xy, _, _, _, _, st = \
            solve_to_grid({**COMMON, "frequency_hz": freq,
                           "disk_velocity_amplitude": 0.0},
                          label_s, trap_z, mid_y)
        g_stand_freq_xy[df] = gs_xy
        solve_log.append((label_s, st))

        label_v = f"vortex_l1_df{df/1e3:+.0f}kHz"
        _, _, gv_xy, _, _, _, _, st = \
            solve_to_grid({**COMMON, "frequency_hz": freq,
                           "standing_velocity_amplitude": 0.0},
                          label_v, trap_z, mid_y)
        g_vortex_freq_xy[df] = gv_xy
        solve_log.append((label_v, st))

    # ── Apodization variants ──────────────────────────────────────
    g_vortex_apod_xy = {"cosine_taper": g_vortex_xy[1]}
    for apod in APOD_VALUES:
        if apod == "cosine_taper":
            continue
        label = f"vortex_l1_{apod}"
        _, _, gv_xy, _, _, _, _, st = \
            solve_to_grid({**COMMON, "standing_velocity_amplitude": 0.0,
                           "lens_apodization": apod},
                          label, trap_z, mid_y)
        g_vortex_apod_xy[apod] = gv_xy
        solve_log.append((label, st))

    total_solve = sum(t for _, t in solve_log)
    print(f"\n  Total solve time: {total_solve:.1f}s  ({len(solve_log)} solves)")

    # ==================================================================
    # PHASE 3 — SUPERPOSITION & METRICS
    # ==================================================================
    print(f"\n{'='*72}")
    print("PHASE 3: Superposition + Metrics")
    print(f"{'='*72}\n")

    # Helper: build combined grid from superposition
    def combine(g_s, g_v, alpha):
        """p_combined = p_stand + α*(V_stand/V_vortex)*p_vortex"""
        scale = alpha * V_stand_base / V_vortex_base
        return g_s + scale * g_v

    omega_base = cfg_base.omega

    # ─── §4.Q12: Amplitude sweep (ℓ=1, baseline freq) ─────────────
    print("  §4 Q12: Amplitude sweep ...")
    for alpha in ALPHA_VALUES:
        g_comb = combine(g_stand_xy, g_vortex_xy[1], alpha)
        g_comb_xz = combine(g_stand_xz, g_vortex_xz, alpha)
        m = compute_metrics(g_stand_xy, g_vortex_xy[1], g_comb,
                            xg, yg, roi_mask, cfg_base,
                            label=f"alpha={alpha:.2f}_l=1")
        m["alpha"] = alpha
        m["ell"] = 1
        m["delta_f_Hz"] = 0
        m["apodization"] = "cosine_taper"
        all_metrics.append(m)
        print(f"    α={alpha:.2f}  η_out={m['eta_out']:.4f}  corr={m['corr_out']:.4f}  "
              f"barrier_red={m['barrier_reduction_pct']:.1f}%")

        # XY panel
        plot_4panel_xy(xg, yg, g_stand_xy, g_vortex_xy[1], g_comb,
                       f"α={alpha:.2f}, ℓ=1", f"xy_alpha{alpha:.2f}_l1.png",
                       roi_center=(cx, cy), roi_radius=roi_r)
        png_count += 1

        # XZ panel
        plot_4panel_xz(xg_xz, zg_xz, g_stand_xz, g_vortex_xz, g_comb_xz,
                       cfg_base, f"XZ  α={alpha:.2f}, ℓ=1",
                       f"xz_alpha{alpha:.2f}_l1.png")
        png_count += 1

        # Gor'kov panel
        plot_gorkov_panel(xg, yg,
                          m["_U_stand"], m["_U_comb"],
                          m["_Fx_s"], m["_Fy_s"],
                          m["_Fx_c"], m["_Fy_c"],
                          f"Gor'kov  α={alpha:.2f}, ℓ=1",
                          f"gorkov_alpha{alpha:.2f}_l1.png",
                          roi_center=(cx, cy), roi_radius=roi_r)
        png_count += 1

    # ─── §4.Q13: Topological charge sweep ──────────────────────────
    print("\n  §4 Q13: Topological charge sweep ...")
    for ell in ELL_VALUES:
        alpha = 0.1  # baseline ratio
        g_comb = combine(g_stand_xy, g_vortex_xy[ell], alpha)
        m = compute_metrics(g_stand_xy, g_vortex_xy[ell], g_comb,
                            xg, yg, roi_mask, cfg_base,
                            label=f"alpha=0.10_l={ell}")
        m["alpha"] = alpha
        m["ell"] = ell
        m["delta_f_Hz"] = 0
        m["apodization"] = "cosine_taper"

        # Winding number
        winding = winding_number(g_vortex_xy[ell], xg, yg, cx, cy,
                                 radius=0.5 * lam)
        m["winding_number"] = float(winding)
        m["winding_error"] = abs(winding - ell)

        # Vortex core radius: distance from centre where |p| is half of ring peak
        X, Y = np.meshgrid(xg, yg)
        r_grid = np.sqrt((X - cx)**2 + (Y - cy)**2)
        pmag_v = np.abs(g_vortex_xy[ell])
        # Find ring peak (max outside a small inner radius)
        ring_mask = r_grid > 0.2 * lam
        if ring_mask.any() and pmag_v[ring_mask].max() > 0:
            ring_peak = pmag_v[ring_mask].max()
            # Core radius: smallest r where pmag_v crosses half of ring peak
            radii = np.linspace(0, 3 * lam, 200)
            profile = np.array([
                np.mean(pmag_v[(r_grid > r - 0.05*lam) & (r_grid < r + 0.05*lam)])
                if np.any((r_grid > r - 0.05*lam) & (r_grid < r + 0.05*lam))
                else 0
                for r in radii
            ])
            half_max = ring_peak / 2
            above = profile >= half_max
            if above.any():
                first_above = np.argmax(above)
                m["core_radius_mm"] = float(radii[first_above] * 1e3)
            else:
                m["core_radius_mm"] = float('nan')
        else:
            m["core_radius_mm"] = float('nan')

        all_metrics.append(m)
        print(f"    ℓ={ell}  winding={winding:.2f}  η_out={m['eta_out']:.4f}  "
              f"core_r={m.get('core_radius_mm', 'N/A')} mm")

        # Panels
        plot_4panel_xy(xg, yg, g_stand_xy, g_vortex_xy[ell], g_comb,
                       f"α=0.10, ℓ={ell}", f"xy_alpha0.10_l{ell}.png",
                       roi_center=(cx, cy), roi_radius=roi_r)
        png_count += 1
        plot_gorkov_panel(xg, yg,
                          m["_U_stand"], m["_U_comb"],
                          m["_Fx_s"], m["_Fy_s"],
                          m["_Fx_c"], m["_Fy_c"],
                          f"Gor'kov  α=0.10, ℓ={ell}",
                          f"gorkov_alpha0.10_l{ell}.png",
                          roi_center=(cx, cy), roi_radius=roi_r)
        png_count += 1

    # ─── §4.Q14: Frequency detune sweep ────────────────────────────
    print("\n  §4 Q14: Frequency detune sweep ...")
    for df in DELTA_F_VALUES:
        alpha = 0.1
        freq = COMMON["frequency_hz"] + df
        cfg_f = FarFieldConfig(**{**COMMON, "frequency_hz": freq})
        g_comb = combine(g_stand_freq_xy[df], g_vortex_freq_xy[df], alpha)
        m = compute_metrics(g_stand_freq_xy[df], g_vortex_freq_xy[df], g_comb,
                            xg, yg, roi_mask, cfg_f,
                            label=f"alpha=0.10_l=1_df={df/1e3:+.0f}kHz")
        m["alpha"] = alpha
        m["ell"] = 1
        m["delta_f_Hz"] = df
        m["apodization"] = "cosine_taper"
        all_metrics.append(m)
        print(f"    Δf={df/1e3:+.0f}kHz  η_out={m['eta_out']:.4f}  "
              f"corr={m['corr_out']:.4f}  "
              f"stand_peak_roi={m['peak_stand_in_roi_Pa']:.3f}")

        plot_4panel_xy(xg, yg, g_stand_freq_xy[df],
                       g_vortex_freq_xy[df], g_comb,
                       f"α=0.10, ℓ=1, Δf={df/1e3:+.0f}kHz",
                       f"xy_alpha0.10_l1_df{df:+.0f}.png",
                       roi_center=(cx, cy), roi_radius=roi_r)
        png_count += 1

    # ─── §5.Q15–Q17: Apodization sweep ────────────────────────────
    print("\n  §5 Q15-17: Apodization sweep ...")
    for apod in APOD_VALUES:
        alpha = 0.1
        g_comb = combine(g_stand_xy, g_vortex_apod_xy[apod], alpha)
        m = compute_metrics(g_stand_xy, g_vortex_apod_xy[apod], g_comb,
                            xg, yg, roi_mask, cfg_base,
                            label=f"alpha=0.10_l=1_{apod}")
        m["alpha"] = alpha
        m["ell"] = 1
        m["delta_f_Hz"] = 0
        m["apodization"] = apod
        all_metrics.append(m)
        print(f"    apod={apod:14s}  η_out={m['eta_out']:.4f}  "
              f"sidelobe={m['sidelobe_ratio']:.4f}  "
              f"vortex_E_roi={m['vortex_energy_in_roi_pct']:.1f}%")

        plot_4panel_xy(xg, yg, g_stand_xy, g_vortex_apod_xy[apod], g_comb,
                       f"α=0.10, ℓ=1, apod={apod}",
                       f"xy_alpha0.10_l1_{apod}.png",
                       roi_center=(cx, cy), roi_radius=roi_r)
        png_count += 1

    # ==================================================================
    # PHASE 4 — ADDITIONAL DIAGNOSTICS (§5 Q15, baseline phase)
    # ==================================================================
    print(f"\n{'='*72}")
    print("PHASE 4: Phase Winding & Phase Comparison")
    print(f"{'='*72}\n")

    # Phase winding for baseline vortex
    for ell in ELL_VALUES:
        wn = winding_number(g_vortex_xy[ell], xg, yg, cx, cy,
                            radius=0.5 * lam)
        print(f"  ℓ={ell}  winding number = {wn:.3f}  "
              f"(expected {ell}, error {abs(wn-ell):.3f})")

    # Phase comparison: standing vs combined (baseline α=0.1)
    g_comb_baseline = combine(g_stand_xy, g_vortex_xy[1], 0.1)
    plot_phase_comparison(xg, yg, g_stand_xy, g_comb_baseline,
                          "Phase: Standing vs Combined (α=0.10, ℓ=1)",
                          "phase_comparison_baseline.png")
    png_count += 1

    # Phase winding plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, ell in zip(axes, ELL_VALUES):
        phase = np.angle(g_vortex_xy[ell])
        im = ax.pcolormesh(xg * 1e3, yg * 1e3, phase, shading="auto",
                           cmap="twilight", vmin=-np.pi, vmax=np.pi)
        # Draw measurement circle
        theta_c = np.linspace(0, 2*np.pi, 100)
        ax.plot(cx*1e3 + 0.5*lam*1e3*np.cos(theta_c),
                cy*1e3 + 0.5*lam*1e3*np.sin(theta_c),
                'g--', lw=1.5, label=f"measurement circle")
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
        wn = winding_number(g_vortex_xy[ell], xg, yg, cx, cy, 0.5*lam)
        ax.set_title(f"Vortex ℓ={ell}  winding={wn:.2f}")
        ax.set_aspect("equal"); ax.legend(fontsize=8)
        plt.colorbar(im, ax=ax, label="Phase [rad]")
    fig.suptitle("Phase Winding Verification", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "phase_winding_all.png", bbox_inches="tight")
    plt.close(fig)
    png_count += 1

    # Vortex energy localisation ring plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    X, Y = np.meshgrid(xg, yg)
    r_grid = np.sqrt((X - cx)**2 + (Y - cy)**2)
    radii_sweep = np.linspace(0, 3 * lam, 200)
    for ax, ell in zip(axes, ELL_VALUES):
        pmag = np.abs(g_vortex_xy[ell])
        profile = []
        for r in radii_sweep:
            annulus = (r_grid >= r - 0.05*lam) & (r_grid < r + 0.05*lam)
            if annulus.any():
                profile.append(np.mean(pmag[annulus]))
            else:
                profile.append(0)
        profile = np.array(profile)
        ax.plot(radii_sweep * 1e3, profile, 'b-', lw=2)
        ax.axvline(roi_r * 1e3, color='g', ls='--', label=f"ROI = {roi_r*1e3:.2f} mm")
        ax.set_xlabel("r [mm]"); ax.set_ylabel("Mean |p| [Pa]")
        ax.set_title(f"Radial profile ℓ={ell}")
        ax.legend()
    fig.suptitle("Vortex Radial Intensity Profiles", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "vortex_radial_profiles.png", bbox_inches="tight")
    plt.close(fig)
    png_count += 1

    # ==================================================================
    # PHASE 5 — ESCAPE PATH ANALYSIS  (2D Gor'kov at trap plane)
    # ==================================================================
    print(f"\n{'='*72}")
    print("PHASE 5: Escape Path Analysis (2D Gor'kov)")
    print(f"{'='*72}\n")

    # Compute 2D Gor'kov for baseline combined (α=0.10, ℓ=1)
    g_comb_esc = combine(g_stand_xy, g_vortex_xy[1], 0.1)
    dx_g = xg[1] - xg[0]
    dy_g = yg[1] - yg[0]
    U_s_esc, Fx_s_esc, Fy_s_esc = gorkov_2d(g_stand_xy, dx_g, dy_g, omega_base)
    U_c_esc, Fx_c_esc, Fy_c_esc = gorkov_2d(g_comb_esc,  dx_g, dy_g, omega_base)

    # Escape path along x at y=centre
    iy_c = len(yg) // 2
    U_path_s = U_s_esc[iy_c, :]
    U_path_c = U_c_esc[iy_c, :]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(xg * 1e3, U_path_s, 'b-', lw=2, label="Standing-only")
    ax.plot(xg * 1e3, U_path_c, 'r-', lw=2, label="Combined (α=0.10)")
    ax.axvspan((cx - roi_r) * 1e3, (cx + roi_r) * 1e3, alpha=0.1,
               color='green', label="ROI")
    ax.set_xlabel("x [mm]"); ax.set_ylabel("U (Gor'kov) [J]")
    ax.set_title("Escape Path: Gor'kov Along x (trap plane, y=centre)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "escape_path_gorkov.png", bbox_inches="tight")
    plt.close(fig)
    png_count += 1

    # Force magnitude along escape path
    F_mag_s_esc = np.sqrt(Fx_s_esc**2 + Fy_s_esc**2)
    F_mag_c_esc = np.sqrt(Fx_c_esc**2 + Fy_c_esc**2)
    F_path_s = F_mag_s_esc[iy_c, :]
    F_path_c = F_mag_c_esc[iy_c, :]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(xg * 1e3, F_path_s, 'b-', lw=2, label="Standing-only")
    ax.plot(xg * 1e3, F_path_c, 'r-', lw=2, label="Combined (α=0.10)")
    ax.axvspan((cx - roi_r) * 1e3, (cx + roi_r) * 1e3, alpha=0.1,
               color='green', label="ROI")
    ax.set_xlabel("x [mm]"); ax.set_ylabel("|F| [N]")
    ax.set_title("Force Magnitude Along x Escape Path")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "escape_path_force.png", bbox_inches="tight")
    plt.close(fig)
    png_count += 1

    # Clean up escape-path temporaries
    del U_s_esc, U_c_esc, Fx_s_esc, Fy_s_esc, Fx_c_esc, Fy_c_esc
    del F_mag_s_esc, F_mag_c_esc, g_comb_esc

    # ==================================================================
    # PHASE 6 — RANKING (§6 Q18–Q20)
    # ==================================================================
    print(f"\n{'='*72}")
    print("PHASE 6: Ranking & Decision")
    print(f"{'='*72}\n")

    # Clean metrics — remove internal arrays before export
    export_metrics = []
    for m in all_metrics:
        m_clean = {k: v for k, v in m.items() if not k.startswith("_")}
        export_metrics.append(m_clean)

    # Compute optimization score:  J = F_escape - λ_reg * η_out
    # Use mean_F_comb_N as escape proxy, and eta_out as disruption penalty
    # Normalize both to [0,1] for fair combination
    eta_vals = np.array([m["eta_out"] for m in export_metrics])
    escape_vals = np.array([m["mean_F_comb_N"] for m in export_metrics])

    eta_max = max(eta_vals.max(), 1e-30)
    esc_max = max(escape_vals.max(), 1e-30)
    eta_norm = eta_vals / eta_max
    esc_norm = escape_vals / esc_max

    LAMBDA_REG = 0.5
    for i, m_clean in enumerate(export_metrics):
        m_clean["J_score"] = float(esc_norm[i] - LAMBDA_REG * eta_norm[i])
        m_clean["eta_norm"] = float(eta_norm[i])
        m_clean["escape_norm"] = float(esc_norm[i])

    # Sort by J descending
    ranked = sorted(export_metrics, key=lambda m: m["J_score"], reverse=True)
    for rank, m in enumerate(ranked, 1):
        m["rank"] = rank

    # Print summary table
    print(f"\n  {'Rank':>4s}  {'Label':40s}  {'eta_out':>8s}  {'corr':>6s}  "
          f"{'barrier_red%':>12s}  {'J_score':>8s}")
    print("  " + "-" * 100)
    for m in ranked:
        br = m.get("barrier_reduction_pct", float('nan'))
        print(f"  {m['rank']:4d}  {m['label']:40s}  "
              f"{m['eta_out']:8.4f}  {m['corr_out']:6.4f}  "
              f"{br:12.1f}  {m['J_score']:8.4f}")

    # Identify best configs
    best_min_disruption = min(ranked, key=lambda m: m["eta_out"])
    best_max_escape = max(ranked, key=lambda m: m.get("mean_F_comb_N", 0))
    best_balanced = ranked[0]  # highest J

    print(f"\n  BEST — Minimal disruption:  {best_min_disruption['label']}")
    print(f"    η_out = {best_min_disruption['eta_out']:.4f}  "
          f"J = {best_min_disruption['J_score']:.4f}")
    print(f"\n  BEST — Maximal escape:      {best_max_escape['label']}")
    print(f"    mean|F| = {best_max_escape['mean_F_comb_N']:.2e} N  "
          f"J = {best_max_escape['J_score']:.4f}")
    print(f"\n  BEST — Balanced (highest J): {best_balanced['label']}")
    print(f"    J = {best_balanced['J_score']:.4f}  η_out = {best_balanced['eta_out']:.4f}")

    # ==================================================================
    # PHASE 7 — SUMMARY PLOTS
    # ==================================================================
    print(f"\n{'='*72}")
    print("PHASE 7: Summary Plots")
    print(f"{'='*72}\n")

    # Amplitude sweep summary
    amp_metrics = [m for m in export_metrics if m.get("ell") == 1
                   and m.get("delta_f_Hz") == 0
                   and m.get("apodization") == "cosine_taper"]
    if amp_metrics:
        alphas = [m["alpha"] for m in amp_metrics]
        etas = [m["eta_out"] for m in amp_metrics]
        barriers = [m["barrier_reduction_pct"] for m in amp_metrics]
        corrs = [m["corr_out"] for m in amp_metrics]

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        axes[0].plot(alphas, etas, 'bo-', lw=2); axes[0].set_ylabel("η_out")
        axes[0].set_xlabel("α = A_vortex/A_stand")
        axes[0].set_title("Lattice Perturbation vs α")
        axes[0].axhline(0.05, color='r', ls='--', lw=1, label="5% threshold")
        axes[0].legend()

        axes[1].plot(alphas, barriers, 'ro-', lw=2); axes[1].set_ylabel("Barrier reduction [%]")
        axes[1].set_xlabel("α"); axes[1].set_title("Trap Barrier Reduction vs α")

        axes[2].plot(alphas, corrs, 'go-', lw=2); axes[2].set_ylabel("Correlation")
        axes[2].set_xlabel("α"); axes[2].set_title("Outside-ROI Correlation vs α")
        axes[2].axhline(0.95, color='r', ls='--', lw=1, label="0.95 threshold")
        axes[2].legend()
        fig.suptitle("§4 Q12: Amplitude Sweep Summary (ℓ=1, cosine_taper)", fontsize=13)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "sweep_alpha_summary.png", bbox_inches="tight")
        plt.close(fig)
        png_count += 1

    # Topological charge summary
    ell_metrics = [m for m in export_metrics if m.get("alpha") == 0.1
                   and m.get("delta_f_Hz") == 0
                   and m.get("apodization") == "cosine_taper"
                   and "winding_number" in m]
    if ell_metrics:
        ells = [m["ell"] for m in ell_metrics]
        etas_l = [m["eta_out"] for m in ell_metrics]
        cores = [m.get("core_radius_mm", float('nan')) for m in ell_metrics]
        slobs = [m["sidelobe_ratio"] for m in ell_metrics]

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        axes[0].bar(ells, etas_l, color='steelblue')
        axes[0].set_xlabel("ℓ"); axes[0].set_ylabel("η_out")
        axes[0].set_title("Perturbation vs ℓ")

        axes[1].bar(ells, cores, color='salmon')
        axes[1].set_xlabel("ℓ"); axes[1].set_ylabel("Core radius [mm]")
        axes[1].set_title("Vortex Core Radius vs ℓ")

        axes[2].bar(ells, slobs, color='goldenrod')
        axes[2].set_xlabel("ℓ"); axes[2].set_ylabel("Sidelobe ratio")
        axes[2].set_title("Sidelobe Ratio vs ℓ")

        fig.suptitle("§4 Q13: Topological Charge Sweep (α=0.10)", fontsize=13)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "sweep_ell_summary.png", bbox_inches="tight")
        plt.close(fig)
        png_count += 1

    # Frequency detune summary
    freq_metrics = [m for m in export_metrics if m.get("ell") == 1
                    and m.get("alpha") == 0.1
                    and m.get("apodization") == "cosine_taper"
                    and "delta_f_Hz" in m]
    freq_metrics = sorted(freq_metrics, key=lambda m: m["delta_f_Hz"])
    if freq_metrics and len(freq_metrics) > 1:
        dfs = [m["delta_f_Hz"] / 1e3 for m in freq_metrics]
        etas_f = [m["eta_out"] for m in freq_metrics]
        peaks_f = [m["peak_stand_in_roi_Pa"] for m in freq_metrics]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(dfs, etas_f, 'bo-', lw=2)
        axes[0].set_xlabel("Δf [kHz]"); axes[0].set_ylabel("η_out")
        axes[0].set_title("Perturbation vs Frequency Detune")

        axes[1].plot(dfs, peaks_f, 'ro-', lw=2)
        axes[1].set_xlabel("Δf [kHz]"); axes[1].set_ylabel("Peak |p_stand| in ROI [Pa]")
        axes[1].set_title("Standing Peak vs Frequency Detune")

        fig.suptitle("§4 Q14: Frequency Detune Sweep", fontsize=13)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "sweep_freq_summary.png", bbox_inches="tight")
        plt.close(fig)
        png_count += 1

    # Apodization comparison
    apod_metrics = [m for m in export_metrics
                    if m.get("alpha") == 0.1 and m.get("ell") == 1
                    and m.get("delta_f_Hz") == 0
                    and m.get("apodization") in APOD_VALUES]
    if apod_metrics:
        ap_names = [m["apodization"] for m in apod_metrics]
        ap_eta = [m["eta_out"] for m in apod_metrics]
        ap_slob = [m["sidelobe_ratio"] for m in apod_metrics]
        ap_roi_E = [m["vortex_energy_in_roi_pct"] for m in apod_metrics]

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        axes[0].bar(ap_names, ap_eta, color='steelblue')
        axes[0].set_ylabel("η_out"); axes[0].set_title("Perturbation vs Apodization")

        axes[1].bar(ap_names, ap_slob, color='salmon')
        axes[1].set_ylabel("Sidelobe ratio"); axes[1].set_title("Sidelobes vs Apodization")

        axes[2].bar(ap_names, ap_roi_E, color='goldenrod')
        axes[2].set_ylabel("Vortex energy in ROI [%]")
        axes[2].set_title("Energy Localisation vs Apodization")

        fig.suptitle("§5 Q16-17: Apodization Comparison (α=0.10, ℓ=1)", fontsize=13)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "sweep_apod_summary.png", bbox_inches="tight")
        plt.close(fig)
        png_count += 1

    # Ranking bar chart
    top_n = min(len(ranked), 10)
    fig, ax = plt.subplots(figsize=(12, 6))
    labels_r = [m["label"] for m in ranked[:top_n]]
    j_scores = [m["J_score"] for m in ranked[:top_n]]
    colors_r = ['gold' if i == 0 else 'steelblue' for i in range(top_n)]
    ax.barh(range(top_n), j_scores, color=colors_r)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(labels_r, fontsize=9)
    ax.set_xlabel("J = F_escape_norm − 0.5·η_norm")
    ax.set_title("Configuration Ranking by Optimisation Score J")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ranking_bar.png", bbox_inches="tight")
    plt.close(fig)
    png_count += 1

    # ==================================================================
    # PHASE 8 — EXPORT
    # ==================================================================
    print(f"\n{'='*72}")
    print("PHASE 8: Export")
    print(f"{'='*72}\n")

    # CSV
    import csv
    csv_path = CSV_DIR / "audit_metrics.csv"
    if ranked:
        keys = [k for k in ranked[0].keys() if not k.startswith("_")]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
            writer.writeheader()
            for m in ranked:
                row = {k: v for k, v in m.items() if not k.startswith("_")}
                writer.writerow(row)
        print(f"  CSV → {csv_path}")

    # JSON
    json_path = OUT_DIR / "results.json"
    out_data = {
        "timestamp": TIMESTAMP,
        "total_time_s": time.time() - t_global,
        "n_solves": len(solve_log),
        "total_solve_time_s": total_solve,
        "n_configs_tested": len(ranked),
        "n_pngs": png_count,
        "particle": {
            "radius_um": PARTICLE_RADIUS * 1e6,
            "rho_kg_m3": PARTICLE_DENSITY,
            "c_m_s": PARTICLE_C,
            "f1": F1_CONTRAST,
            "f2": F2_CONTRAST,
            "mobility": STOKES_MOBILITY,
        },
        "best": {
            "minimal_disruption": best_min_disruption["label"],
            "maximal_escape": best_max_escape["label"],
            "balanced": best_balanced["label"],
        },
        "solve_log": solve_log,
        "ranked_configs": ranked,
    }
    with open(json_path, "w") as f:
        json.dump(out_data, f, indent=2, default=str)
    print(f"  JSON → {json_path}")

    # ==================================================================
    # SUMMARY
    # ==================================================================
    print(f"\n{'='*72}")
    print("VORTEX FUNCTION AUDIT — COMPLETE")
    print(f"{'='*72}")
    print(f"  Total time: {time.time()-t_global:.1f}s")
    print(f"  Solves: {len(solve_log)}")
    print(f"  Configurations tested: {len(ranked)}")
    print(f"  PNG files: {png_count}")
    print(f"  Output: {OUT_DIR}")
    print(f"\n  BEST:")
    print(f"    Minimal disruption: {best_min_disruption['label']}  "
          f"(η_out = {best_min_disruption['eta_out']:.4f})")
    print(f"    Maximal escape:     {best_max_escape['label']}  "
          f"(mean|F| = {best_max_escape['mean_F_comb_N']:.2e} N)")
    print(f"    Balanced:           {best_balanced['label']}  "
          f"(J = {best_balanced['J_score']:.4f})")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
