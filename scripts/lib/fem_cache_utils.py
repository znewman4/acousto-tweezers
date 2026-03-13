"""
Standardised FEM cache loader for the acousto-tweezers project.

Provides a single entry-point for loading any FEM standing-wave NPZ
cache (epl4, epl5, epl6, ...) and optionally interpolating onto a
regular 2-D grid at a given z-slice.

Usage
-----
    from scripts.lib.fem_cache_utils import load_fem_cache, interpolate_slice

    cache = load_fem_cache("results/fem_standing_wave_cache/"
                           "checkpoint_epl5_depth7mm_20260309_113007/"
                           "standing_wave_epl5.npz")

    grid = interpolate_slice(cache, z=cache["z_star"], n_grid=800)
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

# ── Project root (works from any script location) ──────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ── Physical constants ──────────────────────────────────────────────
C_WATER = 1484.0          # m/s
F_HZ = 2.0e6              # Hz
LAM = C_WATER / F_HZ      # 0.742 mm
OMEGA = 2.0 * np.pi * F_HZ
RHO0 = 997.0              # kg/m³

# ── Trapping-plane geometry ─────────────────────────────────────────
# H_TOP: thickness of the petri / trapping region (top slab).
# z_bottom = H_under = start of the trapping slab (petri base).
# z* is determined dynamically by scanning z through the slab and
# finding the plane with the deepest Gor'kov potential well (minimum
# U_min over the central ROI).  This is physically the plane where
# lateral trapping is strongest and where particles are z-levitated.
# For epl5 this gives z* ≈ 5.850 mm, consistent with the acoustic
# pressure antinode inside the 2 mm slab.
# Fallback (no Gor'kov scan): H_under + H_top/2 + lambda/4 (ROI formula).
H_TOP = 2.0e-3            # m  (petri slab thickness, used as fallback)
DZ_SCAN_STEP = 50.0e-6    # m  z-scan step for finding z* (50 µm)

# Default cache path (epl5 — locally available)
DEFAULT_CACHE = (
    PROJECT_ROOT / "results" / "fem_standing_wave_cache"
    / "checkpoint_epl5_depth7mm_20260309_113007"
    / "standing_wave_epl5.npz"
)


# ════════════════════════════════════════════════════════════════════
# Internal helpers
# ════════════════════════════════════════════════════════════════════

def _find_z_star(
    coords: np.ndarray,
    p: np.ndarray,
    z_bottom: float,
    z_top: float,
    lam: float,
    n_grid: int = 120,
    step: float = DZ_SCAN_STEP,
) -> float:
    """
    Find the z-plane with the deepest Gor'kov potential well inside the slab.

    Scans z from z_bottom to z_top in ``step`` increments, evaluates the
    Gor'kov potential on a coarse ``n_grid × n_grid`` grid restricted to the
    central ROI (±1.5λ from domain centre), and returns the z where U_min
    is most negative.

    Falls back to the ROI-studies heuristic H_under + H_top/2 + λ/4 if the
    scan produces no valid result.
    """
    from scipy.spatial import cKDTree as _KDTree

    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    Lx = float(x.max() - x.min())
    Ly = float(y.max() - y.min())
    cx = float(x.min()) + Lx / 2
    cy = float(y.min()) + Ly / 2
    roi_half = 1.5 * lam
    margin = 30e-6

    # Build a 2D grid covering the ROI
    x_roi = np.linspace(cx - roi_half + margin, cx + roi_half - margin, n_grid)
    y_roi = np.linspace(cy - roi_half + margin, cy + roi_half - margin, n_grid)
    dx_roi = float(x_roi[1] - x_roi[0])
    xg, yg = np.meshgrid(x_roi, y_roi)
    query_pts_2d = np.column_stack([xg.ravel(), yg.ravel()])

    omega = 2.0 * np.pi * F_HZ
    # Polystyrene contrast factors (default particle)
    rho_p, c_p = 1050.0, 2350.0
    kappa_w = 1.0 / (RHO0 * C_WATER ** 2)
    kappa_p = 1.0 / (rho_p * c_p ** 2)
    f1 = 1.0 - kappa_p / kappa_w
    f2 = 2.0 * (rho_p - RHO0) / (2.0 * rho_p + RHO0)
    a_p = 50.0e-6

    z_vals = np.arange(z_bottom, z_top + 1e-9, step)
    # Exclude the very bottom and top (boundary artefacts)
    z_vals = z_vals[(z_vals > z_bottom + step * 0.5) & (z_vals < z_top - step * 0.5)]
    if len(z_vals) == 0:
        h_top = z_top - z_bottom
        return float(z_bottom + h_top / 2.0 + 0.25 * lam)

    U_mins = []
    tree = _KDTree(coords)
    k_idw = min(16, len(coords))

    for zs in z_vals:
        # 3D query points at this z
        pts3d = np.column_stack([
            query_pts_2d[:, 0],
            query_pts_2d[:, 1],
            np.full(len(query_pts_2d), zs),
        ])
        dd, ii = tree.query(pts3d, k=k_idw)
        w = 1.0 / np.maximum(dd ** 2, 1e-30)
        w_sum = w.sum(axis=1, keepdims=True)
        p_interp = (w * p[ii]).sum(axis=1) / w_sum[:, 0]
        p_grid = p_interp.reshape(n_grid, n_grid)
        try:
            U, _, _ = gorkov_grid_2d(p_grid, dx_roi, dx_roi,
                                     omega, RHO0, C_WATER, a_p, f1, f2)
            U_mins.append(float(U.min()))
        except Exception:
            U_mins.append(0.0)

    U_mins = np.array(U_mins)
    i_best = int(np.argmin(U_mins))
    z_star = float(z_vals[i_best])
    print(f"[load_fem_cache] z* scan: deepest Gor'kov well at z={z_star*1e3:.3f} mm "
          f"(U_min={U_mins[i_best]:.3e} J)")
    return z_star


# ════════════════════════════════════════════════════════════════════
# Public API
# ════════════════════════════════════════════════════════════════════

def load_fem_cache(path: str | Path | None = None) -> Dict[str, Any]:
    """
    Load a FEM standing-wave NPZ cache.

    Parameters
    ----------
    path : str or Path, optional
        Path to the ``.npz`` file.  Defaults to the epl5 checkpoint.

    Returns
    -------
    dict with keys
        ``coords``   : ndarray (N, 3) float64 — DOF coordinates [m]
        ``p``        : ndarray (N,) complex128   — complex pressure
        ``p_real``   : ndarray (N,) float64
        ``p_imag``   : ndarray (N,) float64
        ``metadata`` : dict (may be empty for older caches)
        ``n_dofs``   : int
        ``p_max``    : float — max |p| [Pa]
        ``domain``   : dict  ``{x_min, x_max, y_min, y_max, z_min, z_max}`` [m]
        ``z_bottom`` : float — bottom of trapping region (petri base) [m]
        ``z_star``   : float — trapping plane z_bottom + DZ_DEFAULT [m]
        ``cache_path``: Path
    """
    path = Path(path) if path is not None else DEFAULT_CACHE
    if not path.exists():
        raise FileNotFoundError(f"FEM cache not found: {path}")

    d = np.load(str(path), allow_pickle=True)

    coords = d["coords"].astype(np.float64)          # (N, 3)
    p_real = d["p_real"].astype(np.float64)           # (N,)
    p_imag = d["p_imag"].astype(np.float64)           # (N,)
    p = p_real + 1j * p_imag

    # Metadata — may be a pickled dict or absent
    metadata: Dict[str, Any] = {}
    if "metadata" in d:
        raw = d["metadata"]
        if hasattr(raw, "item"):
            metadata = raw.item()
        elif isinstance(raw, dict):
            metadata = raw

    x_min, y_min, z_min = coords.min(axis=0)
    x_max, y_max, z_max = coords.max(axis=0)

    # z_bottom = H_under — the petri base / start of the trapping slab.
    # Read from config.json written by the FEM solver beside the cache.
    # Fall back to z_max - H_TOP if config is absent.
    config_json = path.parent / "config.json"
    h_top_cfg = H_TOP
    if config_json.exists():
        with open(config_json) as _cf:
            _cfg = json.load(_cf)
        z_bottom = float(_cfg["H_under"])
        h_top_cfg = float(_cfg.get("H_top", H_TOP))
    else:
        z_bottom = float(z_max - H_TOP)

    # Compute z* as the plane with the deepest Gor'kov well in the slab.
    # Uses a lightweight coarse scan (50 µm steps, 120×120 grid, ROI only).
    z_star = _find_z_star(coords, p, z_bottom, z_bottom + h_top_cfg, LAM)

    return {
        "coords": coords,
        "p": p,
        "p_real": p_real,
        "p_imag": p_imag,
        "metadata": metadata,
        "n_dofs": len(p),
        "p_max": float(np.abs(p).max()),
        "domain": {
            "x_min": float(x_min), "x_max": float(x_max),
            "y_min": float(y_min), "y_max": float(y_max),
            "z_min": float(z_min), "z_max": float(z_max),
        },
        "z_bottom": z_bottom,
        "z_star": float(z_star),
        "cache_path": path,
    }


def interpolate_slice(
    cache: Dict[str, Any],
    z: float | None = None,
    n_grid: int = 800,
    margin: float = 50e-6,
    k_idw: int = 16,
    power: float = 2.0,
) -> Dict[str, Any]:
    """
    Interpolate the FEM field onto a uniform XY grid at a given z-plane.

    Uses inverse-distance-weighted (IDW) interpolation with k nearest
    DOFs — the same method used throughout the project.

    Parameters
    ----------
    cache : dict
        Output of :func:`load_fem_cache`.
    z : float, optional
        Z-coordinate for the slice [m].  Defaults to ``cache["z_star"]``.
    n_grid : int
        Number of grid points per axis.
    margin : float
        Inset from domain boundary to avoid edge artefacts [m].
    k_idw : int
        Number of nearest neighbours for IDW.
    power : float
        IDW distance exponent.

    Returns
    -------
    dict with keys
        ``p_grid``  : ndarray (n_grid, n_grid) complex128
        ``xg``      : ndarray (n_grid,) float64 — x coordinates [m]
        ``yg``      : ndarray (n_grid,) float64 — y coordinates [m]
        ``z``       : float
        ``dx``      : float — grid spacing [m]
        ``dy``      : float — grid spacing [m]
    """
    z = z if z is not None else cache["z_star"]
    dom = cache["domain"]

    xg = np.linspace(dom["x_min"] + margin, dom["x_max"] - margin, n_grid)
    yg = np.linspace(dom["y_min"] + margin, dom["y_max"] - margin, n_grid)
    XX, YY = np.meshgrid(xg, yg)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.full(XX.size, z)])

    tree = cKDTree(cache["coords"])
    t0 = time.time()
    print(f"[interpolate_slice] {n_grid}×{n_grid} at z={z*1e3:.3f} mm …")

    if k_idw == 1:
        _, idx = tree.query(pts, k=1)
        p_flat = cache["p"][idx]
    else:
        dist, idx = tree.query(pts, k=k_idw)
        w = 1.0 / (dist**power + 1e-30)
        w /= w.sum(axis=1, keepdims=True)
        p_flat = (cache["p"][idx] * w).sum(axis=1)

    print(f"  done in {time.time() - t0:.1f}s")

    dx = float(xg[1] - xg[0])
    dy = float(yg[1] - yg[0])

    return {
        "p_grid": p_flat.reshape(n_grid, n_grid),
        "xg": xg,
        "yg": yg,
        "z": z,
        "dx": dx,
        "dy": dy,
    }


def interpolate_xz_slice(
    cache: Dict[str, Any],
    y: float | None = None,
    n_x: int = 800,
    n_z: int = 400,
    margin: float = 50e-6,
    k_idw: int = 16,
    power: float = 2.0,
) -> Dict[str, Any]:
    """
    Interpolate the FEM field onto a uniform XZ grid at a given y-plane.

    Parameters
    ----------
    cache : dict
        Output of :func:`load_fem_cache`.
    y : float, optional
        Y-coordinate for the slice [m].  Defaults to domain centre.
    n_x, n_z : int
        Grid points per axis.
    margin, k_idw, power : see :func:`interpolate_slice`.

    Returns
    -------
    dict with keys ``p_grid``, ``xg``, ``zg``, ``y``, ``dx``, ``dz``
    """
    dom = cache["domain"]
    y = y if y is not None else (dom["y_min"] + dom["y_max"]) / 2.0

    xg = np.linspace(dom["x_min"] + margin, dom["x_max"] - margin, n_x)
    zg = np.linspace(dom["z_min"] + margin, dom["z_max"] - margin, n_z)
    XX, ZZ = np.meshgrid(xg, zg)
    pts = np.column_stack([XX.ravel(), np.full(XX.size, y), ZZ.ravel()])

    tree = cKDTree(cache["coords"])
    t0 = time.time()
    print(f"[interpolate_xz_slice] {n_x}×{n_z} at y={y*1e3:.3f} mm …")

    if k_idw == 1:
        _, idx = tree.query(pts, k=1)
        p_flat = cache["p"][idx]
    else:
        dist, idx = tree.query(pts, k=k_idw)
        w = 1.0 / (dist**power + 1e-30)
        w /= w.sum(axis=1, keepdims=True)
        p_flat = (cache["p"][idx] * w).sum(axis=1)

    print(f"  done in {time.time() - t0:.1f}s")

    return {
        "p_grid": p_flat.reshape(n_z, n_x),
        "xg": xg,
        "zg": zg,
        "y": y,
        "dx": float(xg[1] - xg[0]),
        "dz": float(zg[1] - zg[0]),
    }


# ════════════════════════════════════════════════════════════════════
# Standalone Gor'kov (avoids heavy FEniCSx import from src/)
# ════════════════════════════════════════════════════════════════════
# Duplicated from src/acoustweezers/physics/particles/gorkov.py
# gorkov_grid_2d — kept identical to the canonical implementation.

def gorkov_grid_2d(
    p_grid: np.ndarray,
    dx: float,
    dy: float,
    omega: float,
    rho: float,
    c: float,
    a: float,
    f1: float,
    f2: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Gor'kov potential and radiation force on a 2-D complex-pressure grid.

    Returns (U, Fx, Fy) — all shape (ny, nx).
    """
    K = rho * c**2
    p2_avg = np.abs(p_grid)**2 / 2.0

    dp_dx = np.gradient(p_grid, dx, axis=1)
    dp_dy = np.gradient(p_grid, dy, axis=0)
    grad_p_mag2 = np.abs(dp_dx)**2 + np.abs(dp_dy)**2
    v2_avg = grad_p_mag2 / (2.0 * omega**2 * rho**2)

    prefactor = (4.0 * np.pi / 3.0) * a**3
    U = prefactor * (f1 * p2_avg / (2.0 * K) - f2 * (3.0 * rho / 4.0) * v2_avg)

    Fx = -np.gradient(U, dx, axis=1)
    Fy = -np.gradient(U, dy, axis=0)
    return U, Fx, Fy


# ── Default particle contrast factors (polystyrene in water) ───────
def default_particle_params() -> Dict[str, float]:
    """Return standard polystyrene-in-water particle parameters."""
    rho_p, c_p = 1050.0, 2350.0
    kappa_w = 1.0 / (RHO0 * C_WATER**2)
    kappa_p = 1.0 / (rho_p * c_p**2)
    return {
        "a": 50.0e-6,
        "rho_p": rho_p,
        "c_p": c_p,
        "f1": 1.0 - kappa_p / kappa_w,
        "f2": 2.0 * (rho_p - RHO0) / (2.0 * rho_p + RHO0),
    }
