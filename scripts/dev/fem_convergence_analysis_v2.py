#!/usr/bin/env python3
"""
fem_convergence_analysis_v2.py
==============================
Full post-processing and figure generation for the FEM standing-wave
mesh convergence study.

Reads
-----
  mesh_convergence_study/
    convergence_analysis.csv          ← scalar metrics (already generated)
    <run_folder>/
      fields_manifest.json            ← metadata (EPL, domain size, PML)
      mesh.h5  /  mesh.xdmf
      p_real.h5 / p_real.xdmf
      p_imag.h5 / p_imag.xdmf
      p_mag.h5  / p_mag.xdmf        (optional — recomputed if absent)

Figures produced (publication quality, 300 dpi)
-----------------------------------------------
  fig1_error_vs_h.png                 log-log ε vs h with reference slopes
  fig2_error_vs_epl.png               3-panel ε, spacing error, trap error vs EPL
  fig3_richardson_gci.png             RE extrapolated value + GCI band
  fig4_trap_overlay_coarse.png        EPL=2 trap positions on Gor'kov field
  fig5_trap_overlay_fine.png          EPL=4.5 trap positions on Gor'kov field
  fig6_domain_sensitivity.png         3-panel domain-size study
  fig7_pml_sensitivity.png            3-panel PML sensitivity
  fig8_solve_time.png                 MUMPS solve time vs DOFs
  convergence_analysis_v2.csv         updated metrics with RE/GCI columns
  gci_table.csv                       Richardson extrapolation summary table

Usage
-----
  1. Copy your mesh_convergence_study folder into the same directory as
     this script (or edit STUDY_DIR below).
  2. In VSCode terminal:
       pip install numpy scipy matplotlib h5py pandas
       python fem_convergence_analysis_v2.py

Configuration
-------------
  Edit the USER CONFIGURATION block below to match your setup.

References
----------
  Celik et al., J. Fluids Engng., 130(7), 078001, 2008.  [GCI method]
  Roache, Verification and Validation in Computational Science
    and Engineering, Hermosa, 1998.
  Ihlenburg & Babuška, Comput. Meth. Appl. Mech. Engng., 128, 1995.
    [Helmholtz pollution effect]
"""

from __future__ import annotations

import csv
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from scipy.interpolate import LinearNDInterpolator
from scipy.optimize import brentq
from scipy.spatial import Delaunay

# h5py is optional — field figures skipped gracefully if absent
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    print("[warn] h5py not installed — trap overlay figures will be skipped.")
    print("       pip install h5py")

# Hungarian algorithm for optimal trap matching
try:
    from scipy.optimize import linear_sum_assignment
    HAS_HUNGARIAN = True
except ImportError:
    HAS_HUNGARIAN = False

# ═══════════════════════════════════════════════════════════════════════════════
# USER CONFIGURATION  — edit these paths / constants as needed
# ═══════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Root folder containing all run sub-folders AND convergence_analysis.csv
_STUDY_DIR_CANDIDATES = [
    PROJECT_ROOT / "results" / "mesh_convergence_study",
    PROJECT_ROOT / "mesh_convergence_study",
]
STUDY_DIR = next((p for p in _STUDY_DIR_CANDIDATES if p.exists()),
                 _STUDY_DIR_CANDIDATES[0])

# Where to write figures and CSVs
OUTPUT_DIR = STUDY_DIR / "analysis_v2"

# Physics (must match your solver config)
FREQUENCY_HZ   = 2.15e6
C_SOUND_MS     = 1484.0
RHO_FLUID      = 997.0      # kg/m³  (water)

# Particle properties for Gor'kov potential (polystyrene default)
RHO_PARTICLE   = 1050.0     # kg/m³
C_PARTICLE     = 2350.0     # m/s   (longitudinal)

# Gor'kov contrast factors
_kappa_f = 1.0 / (RHO_FLUID  * C_SOUND_MS**2)
_kappa_p = 1.0 / (RHO_PARTICLE * C_PARTICLE**2)
F1 = 1.0 - _kappa_p / _kappa_f
F2 = 2.0 * (RHO_PARTICLE - RHO_FLUID) / (2.0 * RHO_PARTICLE + RHO_FLUID)

# Derived
WAVELENGTH_M   = C_SOUND_MS / FREQUENCY_HZ
K_WAVE         = 2.0 * np.pi / WAVELENGTH_M

# Grid resolution for Gor'kov interpolation (points per λ)
GORKOV_PPL     = 120

# Trap detection
GORKOV_DEPTH_FRAC   = 0.05   # min depth as fraction of ROI dynamic range
GORKOV_MIN_SEP_M    = WAVELENGTH_M / 4.0  # min separation between traps

# Reference EPL (used as benchmark in convergence study)
REFERENCE_EPL  = 5.0

# EPLs used for Richardson extrapolation (finest three non-reference levels)
# EPL=5.0 is the reference so has NaN eps_L2_roi — use 3.5/4.0/4.5 instead
RE_EPLS        = [3.5, 4.0, 4.5]

# Matplotlib style
plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          10,
    "axes.titlesize":     11,
    "axes.labelsize":     10,
    "legend.fontsize":    9,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "lines.linewidth":    1.5,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "grid.linewidth":     0.5,
})

COLORS = {
    "primary":   "#185FA5",   # blue  — L2 error
    "secondary": "#D85A30",   # coral — spacing error
    "tertiary":  "#3B6D11",   # green — trap position error
    "re":        "#7F77DD",   # purple — Richardson extrapolation
    "ref":       "#185FA5",   # reference traps
    "coarse":    "#D85A30",   # coarse traps
    "fine":      "#D85A30",   # fine traps
    "cost":      "#8B0000",   # solve time
}

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_csv_metrics(study_dir: Path) -> pd.DataFrame:
    """Load the scalar convergence metrics CSV produced by the original script."""
    candidates = [
        study_dir / "convergence_analysis.csv",
        study_dir / "convergence_summary.csv",
        study_dir.parent / "convergence_analysis.csv",
        study_dir.parent / "convergence_summary.csv",
    ]
    csv_path = next((p for p in candidates if p.exists()), None)
    if csv_path is None:
        raise FileNotFoundError(
            f"No supported convergence CSV found in {study_dir}.\n"
            "Looked for convergence_analysis.csv and convergence_summary.csv."
        )
    df = pd.read_csv(csv_path)
    print(f"[✓] Loaded metrics: {len(df)} rows from {csv_path.name}")
    return df


def discover_run_folders(study_dir: Path) -> dict[str, Path]:
    """
    Walk study_dir and return {run_name: folder_path} for every folder that
    contains at least one H5 file.  Falls back to parsing folder name if no
    fields_manifest.json is present.
    """
    runs = {}
    for folder in sorted(study_dir.iterdir()):
        if not folder.is_dir():
            continue
        h5_files = list(folder.glob("*.h5"))
        if not h5_files:
            continue
        runs[folder.name] = folder
    print(f"[✓] Found {len(runs)} run folder(s) with H5 data")
    return runs


def load_manifest(folder: Path) -> dict:
    """Load fields_manifest.json and supplement missing metadata from folder name."""
    manifest_path = folder / "fields_manifest.json"
    meta = {}
    if manifest_path.exists():
        with open(manifest_path) as f:
            meta = json.load(f)

    # Supplement from folder name  e.g. conv_epl4.5_convergence_phys3.0mm_...
    name = folder.name
    meta.setdefault("name", name)
    m = re.search(r"epl(\d+\.?\d*)", name)
    if m and "epl" not in meta:
        meta["epl"] = float(m.group(1))
    m = re.search(r"phys(\d+\.?\d*)mm", name)
    if m and "physical_size_mm" not in meta:
        meta["physical_size_mm"] = float(m.group(1))
    m = re.search(r"pml(\d+\.?\d*)", name)
    if m and "pml_n_wavelengths_xy" not in meta:
        meta["pml_n_wavelengths_xy"] = float(m.group(1))
    elif "pml_n_wavelengths_xy" not in meta:
        meta["pml_n_wavelengths_xy"] = 1.0  # default
    return meta


def discover_npz_files(study_dir: Path) -> dict[str, Path]:
    """
    Return {stem_name: npz_path} for every *.npz file directly under study_dir.
    Also parse EPL and physical_size from the filename.
    """
    result = {}
    for npz in sorted(study_dir.glob("*.npz")):
        result[npz.stem] = npz
    print(f"[✓] Found {len(result)} NPZ file(s)")
    return result


def _parse_npz_meta(stem: str) -> dict:
    """Extract epl and physical_size_mm from an NPZ stem name."""
    meta = {"name": stem, "pml_n_wavelengths_xy": 1.0}
    m = re.search(r"epl(\d+\.?\d*)", stem)
    if m:
        meta["epl"] = float(m.group(1))
    m = re.search(r"phys(\d+\.?\d*)mm", stem)
    if m:
        meta["physical_size_mm"] = float(m.group(1))
    m = re.search(r"pml(\d+\.?\d*)", stem)
    if m:
        meta["pml_n_wavelengths_xy"] = float(m.group(1))
    return meta


def load_npz(npz_path: Path) -> dict | None:
    """
    Load pre-computed Gor'kov potential and trap positions from a solver NPZ file.

    Expected keys in NPZ:
      gorkov_2d      (ny, nx)  raw Gor'kov potential
      x_grid         (nx,)     x coordinates in metres
      y_grid         (ny,)     y coordinates in metres
      trap_positions_m  (N, 2) trap positions in metres

    Returns dict with normalised gorkov_norm ([-1, 1]) or None on failure.
    """
    if not npz_path.exists():
        return None
    try:
        d = np.load(npz_path, allow_pickle=True)
        gorkov  = np.array(d["gorkov_2d"])
        x_grid  = np.array(d["x_grid"])
        y_grid  = np.array(d["y_grid"])
        traps   = (np.array(d["trap_positions_m"])
                   if "trap_positions_m" in d else np.zeros((0, 2)))
        # Normalise Gor'kov to [-1, 1]
        g_min, g_max = gorkov.min(), gorkov.max()
        gorkov_norm = (2.0 * (gorkov - g_min) / (g_max - g_min) - 1.0
                       if g_max > g_min else gorkov)
        return {"gorkov_norm": gorkov_norm,
                "x_grid": x_grid,
                "y_grid": y_grid,
                "traps": traps}
    except Exception as exc:
        print(f"  [warn] Could not load {npz_path.name}: {exc}")
        return None


# ─── H5 reading ───────────────────────────────────────────────────────────────

def _find_dataset(h5file: "h5py.File", candidates: list[str]):
    """Return the first dataset path that exists in the H5 file."""
    found = []
    h5file.visititems(lambda name, obj: found.append(name)
                      if isinstance(obj, h5py.Dataset) else None)
    for c in candidates:
        matches = [f for f in found if c.lower() in f.lower()]
        if matches:
            return matches[0]
    return found[0] if found else None


def load_mesh_h5(folder: Path):
    """
    Return (coords, cells) from mesh.h5.

    FEniCSx XDMF H5 layout (most common):
      /Mesh/mesh/geometry   (N, 2)  float64
      /Mesh/mesh/topology   (M, 3)  int64

    Older DOLFINx layout:
      /mesh/coordinates     (N, 2)
      /mesh/topology        (M, 3)
    """
    if not HAS_H5PY:
        return None, None
    mesh_path = folder / "mesh.h5"
    if not mesh_path.exists():
        return None, None
    with h5py.File(mesh_path, "r") as f:
        # Try FEniCSx layout first
        if "Mesh" in f:
            grp = f["Mesh"]
            mesh_grp = grp[list(grp.keys())[0]]
            coords = np.array(mesh_grp["geometry"])
            cells  = np.array(mesh_grp["topology"])
        elif "mesh" in f:
            coords = np.array(f["mesh"]["coordinates"])
            cells  = np.array(f["mesh"]["topology"])
        else:
            # Auto-discover
            geom_key = _find_dataset(f, ["geometry", "coordinates", "points"])
            topo_key = _find_dataset(f, ["topology", "cells", "connectivity"])
            coords = np.array(f[geom_key]) if geom_key else None
            cells  = np.array(f[topo_key]) if topo_key else None

    if coords is not None and coords.shape[1] > 2:
        coords = coords[:, :2]   # keep x, y only
    return coords, cells


def load_pressure_h5(folder: Path):
    """
    Return complex pressure array (N_dofs,) from p_real.h5 and p_imag.h5.

    FEniCSx layout:
      /Function/<fname>/0   (N_dofs,)   or
      /Function/f_0/0
    """
    if not HAS_H5PY:
        return None

    def _read_scalar_h5(path: Path) -> np.ndarray | None:
        if not path.exists():
            return None
        with h5py.File(path, "r") as f:
            key = _find_dataset(f, ["Function", "f", "values", "vector"])
            if key is None:
                return None
            return np.array(f[key]).ravel()

    p_real = _read_scalar_h5(folder / "p_real.h5")
    p_imag = _read_scalar_h5(folder / "p_imag.h5")

    if p_real is None or p_imag is None:
        # Try p_mag + p_phase as fallback
        p_mag   = _read_scalar_h5(folder / "p_mag.h5")
        p_phase = _read_scalar_h5(folder / "p_phase.h5")
        if p_mag is not None and p_phase is not None:
            p_real = p_mag * np.cos(p_phase)
            p_imag = p_mag * np.sin(p_phase)
        else:
            return None

    return p_real + 1j * p_imag


# ─── Gor'kov potential ─────────────────────────────────────────────────────────

def compute_gorkov_on_grid(coords: np.ndarray,
                           cells: np.ndarray,
                           p_complex: np.ndarray,
                           ppl: int = GORKOV_PPL) -> tuple:
    """
    Interpolate complex pressure onto a regular Cartesian grid and compute
    the Gor'kov acoustic radiation potential.

    U = (V_p / 4) [ f1 * |p|² / (ρ c²) - 3 f2 * ρ |v|² / 2 ]

    In terms of pressure only (plane-wave approximation for ∇p → v):
      |v|² ≈ |∇p|² / (ρ ω)²

    Returns (xx, yy, U_norm) where U_norm is normalised to [-1, 1].
    """
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()

    nx = max(64, int((x_max - x_min) / WAVELENGTH_M * ppl))
    ny = max(64, int((y_max - y_min) / WAVELENGTH_M * ppl))

    xi = np.linspace(x_min, x_max, nx)
    yi = np.linspace(y_min, y_max, ny)
    xx, yy = np.meshgrid(xi, yi)
    pts_grid = np.column_stack([xx.ravel(), yy.ravel()])

    # Interpolate real and imag parts onto grid
    interp = LinearNDInterpolator(coords, p_complex)
    p_grid = interp(pts_grid).reshape(ny, nx)

    # Fill NaN (outside convex hull) with zero
    p_grid = np.nan_to_num(p_grid, nan=0.0)

    # Pressure magnitude squared
    p_mag2 = np.abs(p_grid) ** 2

    # Pressure gradient via finite differences
    dx = xi[1] - xi[0]
    dy = yi[1] - yi[0]
    dp_dx = np.gradient(p_grid, dx, axis=1)
    dp_dy = np.gradient(p_grid, dy, axis=0)
    grad_mag2 = np.abs(dp_dx) ** 2 + np.abs(dp_dy) ** 2

    omega = 2.0 * np.pi * FREQUENCY_HZ
    kappa_f = 1.0 / (RHO_FLUID * C_SOUND_MS ** 2)

    # Gor'kov potential (unnormalised, arbitrary particle volume)
    U = (F1 * kappa_f * p_mag2 / 4.0
         - 3.0 * F2 * grad_mag2 / (4.0 * RHO_FLUID * omega ** 2))

    # Normalise to [-1, 1] for display
    U_range = U.max() - U.min()
    U_norm = 2.0 * (U - U.min()) / U_range - 1.0 if U_range > 0 else U

    return xx, yy, U_norm


def detect_traps(xx: np.ndarray,
                 yy: np.ndarray,
                 U_norm: np.ndarray,
                 roi_frac: float = 0.5,
                 depth_frac: float = GORKOV_DEPTH_FRAC,
                 min_sep_m: float = GORKOV_MIN_SEP_M) -> np.ndarray:
    """
    Detect acoustic traps as local minima of normalised Gor'kov potential.

    Improvements over original script:
      - Threshold is relative to U_norm dynamic range within the ROI
        (removes sensitivity to absolute pressure amplitude)
      - Minima found by scanning 3×3 neighbourhood on grid

    Returns (N, 2) array of trap (x, y) positions in metres.
    """
    from scipy.ndimage import minimum_filter

    nx, ny = xx.shape[1], xx.shape[0]
    x_min, x_max = xx[0, 0], xx[0, -1]
    y_min, y_max = yy[0, 0], yy[-1, 0]

    # Central ROI
    roi_x0 = x_min + (1 - roi_frac) / 2 * (x_max - x_min)
    roi_x1 = x_min + (1 + roi_frac) / 2 * (x_max - x_min)
    roi_y0 = y_min + (1 - roi_frac) / 2 * (y_max - y_min)
    roi_y1 = y_min + (1 + roi_frac) / 2 * (y_max - y_min)

    ix0 = int((roi_x0 - x_min) / (x_max - x_min) * (nx - 1))
    ix1 = int((roi_x1 - x_min) / (x_max - x_min) * (nx - 1))
    iy0 = int((roi_y0 - y_min) / (y_max - y_min) * (ny - 1))
    iy1 = int((roi_y1 - y_min) / (y_max - y_min) * (ny - 1))

    U_roi = U_norm[iy0:iy1, ix0:ix1]

    # Threshold relative to ROI dynamic range
    U_thresh = U_roi.min() + depth_frac * (U_roi.max() - U_roi.min())

    # Local minima
    footprint = np.ones((3, 3), dtype=bool)
    local_min = minimum_filter(U_roi, footprint=footprint)
    is_local_min = (U_roi == local_min) & (U_roi < U_thresh)

    iy_idx, ix_idx = np.where(is_local_min)

    # Convert to physical coordinates (relative to full grid)
    x_traps = xx[0, ix0 + ix_idx]
    y_traps = yy[iy0 + iy_idx, 0]
    positions = np.column_stack([x_traps, y_traps])

    # Non-maximum suppression: enforce min separation
    if len(positions) == 0:
        return positions
    keep = _nms_traps(positions, U_roi[iy_idx, ix_idx], min_sep_m)
    return positions[keep]


def _nms_traps(positions: np.ndarray,
               values: np.ndarray,
               min_sep: float) -> np.ndarray:
    """Greedy non-maximum suppression by Gor'kov value (keep deepest)."""
    order = np.argsort(values)  # deepest (most negative) first
    kept = []
    for i in order:
        pos_i = positions[i]
        too_close = False
        for j in kept:
            if np.linalg.norm(pos_i - positions[j]) < min_sep:
                too_close = True
                break
        if not too_close:
            kept.append(i)
    return np.array(kept, dtype=int)


def match_traps_hungarian(ref_pos: np.ndarray,
                          cur_pos: np.ndarray,
                          max_dist: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Optimally match cur_pos traps to ref_pos traps using the Hungarian
    algorithm (globally optimal assignment, unlike the greedy approach).

    Returns (ref_idx, cur_idx) of matched pairs within max_dist.
    """
    if len(ref_pos) == 0 or len(cur_pos) == 0:
        return np.array([], int), np.array([], int)

    # Cost matrix: pairwise Euclidean distances
    cost = np.linalg.norm(
        ref_pos[:, None, :] - cur_pos[None, :, :], axis=2
    )   # shape (n_ref, n_cur)

    ref_idx, cur_idx = linear_sum_assignment(cost)

    # Reject pairs exceeding max_dist
    dists = cost[ref_idx, cur_idx]
    valid = dists <= max_dist
    return ref_idx[valid], cur_idx[valid]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — RICHARDSON EXTRAPOLATION + GCI
# ═══════════════════════════════════════════════════════════════════════════════

def richardson_gci(f1: float, f2: float, f3: float,
                   h1: float, h2: float, h3: float,
                   fs: float = 1.25) -> dict:
    """
    Celik et al. (2008) Grid Convergence Index.

    Parameters
    ----------
    f1, f2, f3  : quantity of interest at h1 < h2 < h3  (fine → coarse)
    h1, h2, h3  : mesh sizes (h = λ/EPL)
    fs          : safety factor (1.25 for > 3 grids, 3.0 for 2 grids)

    Returns
    -------
    dict with keys:
      p_app    : apparent convergence order
      f_exact  : Richardson-extrapolated value
      e21_a    : approximate relative error (fine pair)
      e21_ext  : extrapolated relative error
      gci_fine : GCI on the fine grid
      gci_med  : GCI on the medium grid
      gci_ratio: GCI ratio (should be ~1 for asymptotic convergence)
    """
    r21 = h2 / h1   # must be > 1 (h2 coarser than h1)
    r32 = h3 / h2

    eps21 = f2 - f1
    eps32 = f3 - f2

    # Solve for apparent order p iteratively (Celik eq. 3)
    if abs(eps32) < 1e-15 or abs(eps21) < 1e-15:
        p_app = float("nan")
    else:
        s = np.sign(eps32 / eps21)
        def _eq(p):
            try:
                return p - abs(np.log(abs(eps32 / eps21))
                               + np.log((r21**p - s) / (r32**p - s))) / np.log(r21)
            except (ValueError, ZeroDivisionError):
                return float("nan")
        try:
            # Extended search range; small h-ratios (~1.1) push roots well above 10
            p_app = brentq(_eq, 0.01, 200.0, xtol=1e-6, maxiter=500)
        except (ValueError, RuntimeError):
            # Direct log formula as fallback (constant-ratio approximation)
            p_app = np.log(abs(eps32 / eps21)) / np.log(r21)

    # Flag whether p_app is physically plausible for FEM (P1–P3: order 2–4)
    p_reliable = not np.isnan(p_app) and 0.5 <= p_app <= 6.0
    # For GCI arithmetic use theoretical P2 order when p_app is unreliable
    p_gci = p_app if p_reliable else 2.0

    # Extrapolated value — only meaningful when convergence is asymptotic
    if not p_reliable:
        # Pre-asymptotic regime: Richardson extrapolation diverges (would give
        # negative or pathological values).  Report NaN; figures will display
        # the fine-grid value f1 with a GCI error-bar instead.
        f_ext = float("nan")
    elif not np.isnan(p_app) and (r21**p_gci - 1.0) != 0:
        f_ext = (r21**p_gci * f1 - f2) / (r21**p_gci - 1.0)
        # Sanity-check: if extrapolation yields negative or >10× the fine
        # value, the data is not in the asymptotic regime despite p being
        # "plausible" — treat as unreliable.
        if f_ext <= 0 or f_ext > 10 * f1:
            f_ext = float("nan")
    else:
        f_ext = float("nan")

    # Error measures
    e21_a   = abs((f1 - f2) / f1) if f1 != 0 else float("nan")
    e21_ext = abs((f_ext - f1) / f_ext) if (not np.isnan(f_ext) and f_ext != 0) else float("nan")

    # GCI (use p_gci so the bound is conservative when p_app is unreliable)
    gci_fine = (fs * e21_a / (r21**p_gci - 1.0)
                if not np.isnan(p_app) and (r21**p_gci - 1.0) != 0
                else float("nan"))
    gci_med  = (fs * abs((f2 - f3) / f2) / (r32**p_gci - 1.0)
                if (f2 != 0 and not np.isnan(p_app) and (r32**p_gci - 1.0) != 0)
                else float("nan"))

    # Asymptotic check (should be ~1.0)
    gci_ratio = (gci_med / (r21**p_gci * gci_fine)
                 if not np.isnan(p_app) and gci_fine not in (0, float("nan"))
                 else float("nan"))

    return {
        "r21":        r21,
        "r32":        r32,
        "p_app":      p_app,
        "p_reliable": p_reliable,
        "p_gci":      p_gci,
        "f1":         f1,
        "f2":         f2,
        "f3":         f3,
        "f_exact":    f_ext,
        "e21_a":      e21_a,
        "e21_ext":    e21_ext,
        "gci_fine":   gci_fine,
        "gci_med":    gci_med,
        "gci_ratio":  gci_ratio,
    }


def compute_re_gci(df: pd.DataFrame) -> dict:
    """Apply Richardson extrapolation to the finest three non-reference EPL levels."""
    # Filter to convergence sweep only (3mm domain, PML=1.0λ, finite error)
    conv = df[
        (df["physical_size_mm"].round(1) == 3.0)
        & (df["pml_n_wavelengths_xy"].round(1) == 1.0)
        & df["eps_L2_roi"].notna()
        & np.isfinite(df["eps_L2_roi"])
    ].copy()
    conv = conv.sort_values("epl")

    re_rows = conv[conv["epl"].isin(RE_EPLS)].sort_values("epl")
    if len(re_rows) < 3:
        print(f"[warn] Need EPL {RE_EPLS} for Richardson extrapolation; "
              f"found {list(re_rows['epl'].values)}.  Skipping GCI.")
        return {}

    rows = re_rows.sort_values("epl").to_dict("records")
    # fine (highest EPL) = smallest h → f1
    f1, f2, f3 = rows[2]["eps_L2_roi"], rows[1]["eps_L2_roi"], rows[0]["eps_L2_roi"]
    h1 = WAVELENGTH_M / rows[2]["epl"]
    h2 = WAVELENGTH_M / rows[1]["epl"]
    h3 = WAVELENGTH_M / rows[0]["epl"]

    result = richardson_gci(f1, f2, f3, h1, h2, h3)
    result.update({
        "epl_fine":   rows[2]["epl"],
        "epl_med":    rows[1]["epl"],
        "epl_coarse": rows[0]["epl"],
    })
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — FIGURE GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def _save(fig, name: str, out_dir: Path):
    path = out_dir / name
    fig.savefig(path)
    print(f"  [saved] {path.name}")
    plt.close(fig)


def _remove_if_exists(path: Path):
    if path.exists():
        path.unlink()
        print(f"  [removed stale] {path.name}")


# ─── Figure 1: ε vs h (log-log) ───────────────────────────────────────────────

def fig_error_vs_h(df: pd.DataFrame, out_dir: Path):
    conv = df[
        (df["physical_size_mm"].round(1) == 3.0) &
        (df["pml_n_wavelengths_xy"].round(1) == 1.0) &
        (df["epl"] < REFERENCE_EPL) &
        df["eps_L2_roi"].notna() &
        np.isfinite(df["eps_L2_roi"])
    ].sort_values("epl")

    h_vals = WAVELENGTH_M / conv["epl"].values * 1e3   # mm
    eps = conv["eps_L2_roi"].values

    fig, ax = plt.subplots(figsize=(6, 5))

    # Main convergence data
    ax.loglog(
        h_vals, eps, "o-",
        color=COLORS["primary"],
        ms=7, lw=1.8,
        label=r"FEM P1  $\varepsilon_{L2}$ ROI"
    )

    # Reference slope for P1: O(h^2)
    h_ref = np.logspace(
        np.log10(h_vals[-1] * 0.7),
        np.log10(h_vals[0] * 1.3),
        80
    )
    slope = eps[-1] * (h_ref / h_vals[-1]) ** 2
    ax.loglog(
        h_ref, slope, "--",
        color="#888780", lw=1.2,
        label=r"$O(h^2)$ expected (P1)"
    )

    # Annotate EPL values
    offsets = [(5, 8), (-40, 8), (5, -14), (-40, -14), (5, 8)]
    for i, (h, e, epl) in enumerate(zip(h_vals, eps, conv["epl"].values)):
        dx, dy = offsets[i % len(offsets)]
        ax.annotate(
            f"EPL={epl:.1f}",
            xy=(h, e),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8
        )

    ax.set_xlabel(r"$h = \lambda/\mathrm{EPL}$  (mm)")
    ax.set_ylabel(r"Relative $L_2$ error in ROI, $\varepsilon$")
    ax.set_title("Mesh convergence — ROI field error vs element size")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    _save(fig, "fig1_error_vs_h.png", out_dir)


# ─── Figure 2: 3-panel ε, spacing, trap error vs EPL ─────────────────────────

def fig_error_vs_epl(df: pd.DataFrame, out_dir: Path):
    conv = df[(df["physical_size_mm"].round(1) == 3.0) &
              (df["epl"] < REFERENCE_EPL)].sort_values("epl")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    # Panel 1 — L2 error
    ax = axes[0]
    ax.semilogy(conv["epl"], conv["eps_L2_roi"],
                "o-", color=COLORS["primary"], ms=7)
    ax.set_xlabel("EPL  (elements per λ)")
    ax.set_ylabel(r"$\varepsilon_{L2}$ ROI  (relative)")
    ax.set_title("ROI field error\n(primary metric)")

    # Panel 2 — centreline trap spacing error
    ax = axes[1]
    valid = conv["spacing_err_pct"].notna()
    ax.semilogy(conv["epl"][valid], conv["spacing_err_pct"][valid],
                "s-", color=COLORS["secondary"], ms=7)
    ax.axhline(WAVELENGTH_M / 2 * 1e3, color="#aaa", lw=0.8, ls=":")
    ax.set_xlabel("EPL  (elements per λ)")
    ax.set_ylabel("Trap spacing error  (%)")
    ax.set_title("Centreline trap spacing error\n(secondary metric)")

    # Panel 3 — mean matched trap position error
    ax = axes[2]
    valid2 = conv["mean_trap_err_m"].notna()
    ax.semilogy(conv["epl"][valid2],
                conv["mean_trap_err_m"][valid2] * 1e6,
                "^-", color=COLORS["tertiary"], ms=7)
    ax.set_xlabel("EPL  (elements per λ)")
    ax.set_ylabel(r"Mean matched trap error  ($\mu$m)")
    ax.set_title("Matched trap position error\n(secondary metric)")

    fig.suptitle(
        f"Convergence metrics vs EPL  |  3.0 mm domain, PML = 1.0λ  |  "
        f"Reference: EPL = {REFERENCE_EPL}",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    _save(fig, "fig2_error_vs_epl.png", out_dir)


# ─── Figure 3: Richardson extrapolation summary ───────────────────────────────

def fig_richardson_gci(df: pd.DataFrame, gci: dict, out_dir: Path):
    if not gci:
        print("  [skip] fig3 — GCI data not available")
        return

    conv = df[(df["physical_size_mm"].round(1) == 3.0)].sort_values("epl")
    epl_vals = conv["epl"].values
    eps_vals = conv["eps_L2_roi"].values

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Left: convergence plot with RE annotation ─────────────────────────────
    ax = axes[0]
    mask = epl_vals < REFERENCE_EPL
    ax.semilogy(epl_vals[mask], eps_vals[mask], "o-",
                color=COLORS["primary"], ms=7, label="Computed ε")

    f_ext        = gci["f_exact"]    # NaN when pre-asymptotic
    f1_fine      = gci["f1"]
    gci_fine_val = gci["gci_fine"]
    pre_asymp    = not gci.get("p_reliable", True) or np.isnan(f_ext)

    if not pre_asymp:
        gci_lo = max(1e-8, f_ext * (1 - gci_fine_val))
        gci_hi = f_ext * (1 + gci_fine_val)
        ax.axhline(f_ext, color=COLORS["re"], lw=1.5, ls="-.",
                   label=f"$\\varepsilon^* = {f_ext:.4e}$  (Richardson)")
        ax.axhspan(gci_lo, gci_hi, color=COLORS["re"], alpha=0.15,
                   label=f"$\\pm$ GCI = {gci_fine_val*100:.1f}%")
    else:
        # Pre-asymptotic: Richardson extrapolation is unreliable.
        # Show a GCI uncertainty bar centred on the finest data point instead.
        gci_lo_f1 = max(1e-8, f1_fine * (1 - gci_fine_val))
        gci_hi_f1 = f1_fine * (1 + gci_fine_val)
        ax.axhspan(gci_lo_f1, gci_hi_f1, color=COLORS["re"], alpha=0.15,
                   label=f"Conservative GCI bound on finest grid\n"
                         f"(GCI = {gci_fine_val*100:.0f} %)")

    p_note = (f"$p_{{app}}$ = {gci['p_app']:.2f}  (unreliable: pre-asymptotic)\n"
              f"GCI uses conservative $p$ = {gci['p_gci']:.0f}"
              if pre_asymp
              else f"Apparent order  $p_{{app}}$ = {gci['p_app']:.2f}\n"
                   f"(expected P2: $p$ = 3)")
    valid_eps = eps_vals[mask & np.isfinite(eps_vals)]
    ann_y = valid_eps[-1] if len(valid_eps) > 0 else f1_fine
    ax.annotate(
        p_note,
        xy=(epl_vals[mask][-1], ann_y),
        xytext=(0.25, 0.55), textcoords="axes fraction",
        fontsize=9, color=COLORS["re"],
        arrowprops=dict(arrowstyle="->", color=COLORS["re"], lw=0.8),
    )

    ax.set_xlabel("EPL  (elements per λ)")
    ax.set_ylabel(r"$\varepsilon_{L2}$ ROI")
    title_suffix = "\n(pre-asymptotic — GCI conservative bound, $p$=2)" if pre_asymp else ""
    ax.set_title("Richardson extrapolation" + title_suffix)
    ax.legend(fontsize=9)

    # ── Right: GCI summary table ──────────────────────────────────────────────
    ax2 = axes[1]
    ax2.axis("off")

    headers = ["Parameter", "Value"]
    rows_tab = [
        ["EPL (fine / med / coarse)",
         f"{gci['epl_fine']:.1f} / {gci['epl_med']:.1f} / {gci['epl_coarse']:.1f}"],
        [r"$h$ ratio $r_{21}$",         f"{gci['r21']:.4f}"],
        [r"$h$ ratio $r_{32}$",         f"{gci['r32']:.4f}"],
        ["Apparent order $p_{app}$",    f"{gci['p_app']:.3f}"
                                        + ("" if gci.get("p_reliable", True) else "  (unreliable)")],
        ["GCI computed with $p$",       f"{gci['p_gci']:.1f}  (conservative)"],
        [r"$\varepsilon_1$ (fine)",     f"{gci['f1']:.4e}"],
        [r"$\varepsilon^*$ (extrap.)",
         (f"{gci['f_exact']:.4e}"
          if not np.isnan(gci["f_exact"])
          else "N/A (pre-asymptotic)")],
        ["Approx. rel. error $e^a_{21}$",
         f"{gci['e21_a']*100:.2f} %"],
        ["Extrap. rel. error $e^{ext}_{21}$",
         (f"{gci['e21_ext']*100:.2f} %"
          if not np.isnan(gci["e21_ext"])
          else "N/A (pre-asymptotic)")],
        ["GCI$_{fine}$",               f"{gci['gci_fine']*100:.2f} %"],
        ["GCI$_{med}$",                f"{gci['gci_med']*100:.2f} %"],
        ["Asymptotic ratio",           f"{gci['gci_ratio']:.4f}  (target ≈ 1)"],
    ]

    col_widths = [0.65, 0.35]
    y_pos = 0.97
    ax2.text(0.02, y_pos + 0.03, "Grid Convergence Index (Celik et al., 2008)",
             fontsize=10, fontweight="bold", transform=ax2.transAxes)

    for i, (h, w) in enumerate(zip(headers, [sum(col_widths[:1]),
                                              sum(col_widths)])):
        ax2.text(sum(col_widths[:i]) + 0.02, y_pos,
                 h, fontsize=9, fontweight="bold",
                 transform=ax2.transAxes, color="#185FA5")
    y_pos -= 0.05

    for row in rows_tab:
        x = 0.02
        for val, cw in zip(row, col_widths):
            ax2.text(x, y_pos, val, fontsize=9, transform=ax2.transAxes)
            x += cw
        y_pos -= 0.08

    # Asymptotic convergence verdict
    ratio = gci.get("gci_ratio", float("nan"))
    verdict = "Asymptotic convergence confirmed" if (
        not np.isnan(ratio) and 0.8 < ratio < 1.2) else \
        "Not yet in asymptotic regime"
    ax2.text(0.02, y_pos - 0.02, verdict, fontsize=9,
             color="#3B6D11" if verdict.startswith("Asymptotic") else "#D85A30",
             transform=ax2.transAxes, fontweight="bold")

    fig.suptitle("Richardson Extrapolation and Grid Convergence Index",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    _save(fig, "fig3_richardson_gci.png", out_dir)


# ─── Figure 4 & 5: Trap overlay plots ─────────────────────────────────────────

def fig_trap_overlay(npz_data: dict,
                     ref_traps: np.ndarray,
                     epl_label: str,
                     fig_name: str,
                     out_dir: Path):
    """
    Plot trap positions overlaid on the pre-computed Gor'kov potential.
    npz_data must be the dict returned by load_npz().
    """
    if npz_data is None:
        print(f"  [skip] {fig_name} — NPZ data not available")
        return

    U_norm = npz_data["gorkov_norm"]
    x_grid = npz_data["x_grid"]
    y_grid = npz_data["y_grid"]
    traps  = npz_data["traps"]

    xx, yy = np.meshgrid(x_grid, y_grid)

    # Match to reference using Hungarian algorithm
    if HAS_HUNGARIAN and ref_traps is not None and len(ref_traps) > 0 and len(traps) > 0:
        ref_idx, cur_idx = match_traps_hungarian(
            ref_traps, traps, max_dist=WAVELENGTH_M / 4
        )
        n_matched = len(ref_idx)
    else:
        n_matched = 0

    fig, ax = plt.subplots(figsize=(6, 6))

    im = ax.pcolormesh(
        xx * 1e3, yy * 1e3, U_norm,
        cmap="RdBu_r", vmin=-1, vmax=1, shading="auto", rasterized=True,
    )
    plt.colorbar(im, ax=ax, label="Normalised Gor'kov potential", shrink=0.85)

    if ref_traps is not None and len(ref_traps) > 0:
        ax.plot(ref_traps[:, 0] * 1e3, ref_traps[:, 1] * 1e3,
                "o", color=COLORS["ref"], ms=6,
                label=f"Reference (EPL={REFERENCE_EPL:.0f}, n={len(ref_traps)})")

    if len(traps) > 0:
        ax.plot(traps[:, 0] * 1e3, traps[:, 1] * 1e3,
                "x", color=COLORS["coarse"], ms=8, mew=1.5,
                label=f"EPL={epl_label}  (n={len(traps)}, {n_matched} matched)")

    ax.set_xlabel("x  (mm)")
    ax.set_ylabel("y  (mm)")
    ax.set_title(f"Trap overlay: EPL={epl_label} vs Reference\n"
                 "Background: normalised Gor'kov potential")
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(fontsize=9, loc="upper right")
    ax.set_aspect("equal")

    _save(fig, fig_name, out_dir)
    return traps


# ─── Figure 6: Domain-size sensitivity ───────────────────────────────────────

def fig_domain_sensitivity(df: pd.DataFrame, out_dir: Path):
    # Group by domain size — use all runs at a consistent EPL if possible
    # Heuristic: keep the highest EPL available per domain size
    domain_df = df.copy()
    domain_df = domain_df[
        domain_df["physical_size_mm"].notna() & domain_df["epl"].notna() &
        np.isfinite(domain_df["max_p_Pa"])
    ].sort_values(["physical_size_mm", "epl"])

    def _pick_best_domain_row(g: pd.DataFrame) -> pd.Series:
        finite = g[np.isfinite(g["max_p_Pa"])].copy()
        src = finite if not finite.empty else g
        return src.sort_values("epl").iloc[-1]

    rows = []
    for phys_mm in sorted(domain_df["physical_size_mm"].dropna().unique()):
        group = domain_df[domain_df["physical_size_mm"] == phys_mm]
        rows.append(_pick_best_domain_row(group).to_dict())
    best_epl_per_domain = pd.DataFrame(rows)

    if best_epl_per_domain.empty:
        print("  [skip] fig6 — no valid domain-size rows found")
        return

    best_epl_per_domain = best_epl_per_domain.sort_values("physical_size_mm").reset_index(drop=True)

    domains = best_epl_per_domain["physical_size_mm"].values
    epls    = best_epl_per_domain["epl"].values

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    # Panel 1 — max pressure
    ax = axes[0]
    bars = ax.bar(domains.astype(str), best_epl_per_domain["max_p_Pa"],
                  color=COLORS["primary"], alpha=0.8, width=0.5)
    ax.set_xlabel("Physical domain size  (mm)")
    ax.set_ylabel("max|p|  (Pa)")
    ax.set_title("Max pressure vs domain size")
    for bar, epl in zip(bars, epls):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"EPL={epl:.1f}", ha="center", va="bottom", fontsize=8)

    # Panel 2 — centreline trap spacing
    ax = axes[1]
    spacing_mm = best_epl_per_domain["cl_spacing_m"] * 1e3
    ax.bar(domains.astype(str), spacing_mm,
           color=COLORS["secondary"], alpha=0.8, width=0.5)
    ax.axhline(WAVELENGTH_M / 2 * 1e3, color="#333", lw=1, ls="--",
               label=f"λ/2 = {WAVELENGTH_M/2*1e3:.3f} mm")
    ax.set_xlabel("Physical domain size  (mm)")
    ax.set_ylabel("Centreline trap spacing  (mm)")
    ax.set_title("Trap spacing vs domain size")
    ax.legend(fontsize=9)

    # Panel 3 — detected trap count
    ax = axes[2]
    ax.bar(domains.astype(str), best_epl_per_domain["n_traps"],
           color=COLORS["tertiary"], alpha=0.8, width=0.5)
    ax.set_xlabel("Physical domain size  (mm)")
    ax.set_ylabel("Trap count in ROI")
    ax.set_title("Detected traps vs domain size")

    fig.suptitle("Domain-size sensitivity study", fontsize=11, y=1.02)
    fig.tight_layout()
    _save(fig, "fig6_domain_sensitivity.png", out_dir)


# ─── Figure 7: PML sensitivity ────────────────────────────────────────────────

def fig_pml_sensitivity(df: pd.DataFrame, out_dir: Path):
    # PML sweep runs: same EPL and domain size, varying pml_n_wavelengths_xy
    # Use the 3mm domain, EPL=5 reference runs
    pml_df = df[(df["physical_size_mm"].round(1) == 3.0) &
                (df["epl"] == REFERENCE_EPL)].sort_values("pml_n_wavelengths_xy")

    if len(pml_df) < 2:
        print("  [skip] fig7 — not enough PML sensitivity runs in CSV")
        return

    pml_vals = pml_df["pml_n_wavelengths_xy"].values

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    ax = axes[0]
    ax.plot(pml_vals, pml_df["max_p_Pa"], "o-", color=COLORS["primary"], ms=7)
    ax.set_xlabel("PML thickness  (wavelengths)")
    ax.set_ylabel("max|p|  (Pa)")
    ax.set_title("Max pressure vs PML thickness")

    ax = axes[1]
    spacing_mm = pml_df["cl_spacing_m"] * 1e3
    ax.plot(pml_vals, spacing_mm, "s-", color=COLORS["secondary"], ms=7)
    ax.axhline(WAVELENGTH_M / 2 * 1e3, color="#333", lw=1, ls="--",
               label=f"λ/2 = {WAVELENGTH_M/2*1e3:.3f} mm")
    ax.set_xlabel("PML thickness  (wavelengths)")
    ax.set_ylabel("Centreline trap spacing  (mm)")
    ax.set_title("Trap spacing vs PML thickness")
    ax.legend(fontsize=9)

    ax = axes[2]
    ax.plot(pml_vals, pml_df["n_traps"], "^-", color=COLORS["tertiary"], ms=7)
    ax.set_xlabel("PML thickness  (wavelengths)")
    ax.set_ylabel("Trap count in ROI")
    ax.set_title("Detected traps vs PML thickness")

    fig.suptitle("PML sensitivity study  |  3.0 mm domain, EPL = 5.0",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    _save(fig, "fig7_pml_sensitivity.png", out_dir)


# ─── Figure 8: Solve time ─────────────────────────────────────────────────────

def fig_solve_time(df: pd.DataFrame, out_dir: Path):
    # Restrict to the convergence sweep only (3mm domain, PML=1.0λ)
    conv = df[(df["physical_size_mm"].round(1) == 3.0) &
              (df["pml_n_wavelengths_xy"].round(1) == 1.0)].sort_values("dofs")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(conv["dofs"], conv["solve_time_s"], "o-",
            color=COLORS["cost"], ms=8, lw=2)

    for _, row in conv.iterrows():
        ax.annotate(f"EPL={row['epl']:.1f}",
                    xy=(row["dofs"], row["solve_time_s"]),
                    xytext=(5, 3), textcoords="offset points", fontsize=8)

    # Fit power law for annotation
    log_dofs = np.log10(conv["dofs"].values.astype(float))
    log_time = np.log10(conv["solve_time_s"].values.astype(float))
    coeffs   = np.polyfit(log_dofs, log_time, 1)
    ax.set_xlabel("Degrees of freedom  (DOFs)")
    ax.set_ylabel("Solve time  (s)")
    ax.set_title(
        f"MUMPS solve time vs problem size\n"
        f"(fitted scaling: $t \\propto N^{{{coeffs[0]:.2f}}}$)"
    )
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x/1000)}k" if x >= 1000 else str(int(x))
    ))

    _save(fig, "fig8_solve_time.png", out_dir)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — SAVE UPDATED CSV + GCI TABLE
# ═══════════════════════════════════════════════════════════════════════════════

def save_gci_csv(gci: dict, out_dir: Path):
    if not gci:
        _remove_if_exists(out_dir / "gci_table.csv")
        return
    path = out_dir / "gci_table.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["parameter", "value", "notes"])
        w.writerow(["epl_fine",         gci["epl_fine"],   "Finest level used"])
        w.writerow(["epl_med",          gci["epl_med"],    "Medium level"])
        w.writerow(["epl_coarse",       gci["epl_coarse"], "Coarsest of the three"])
        w.writerow(["r21",              gci["r21"],        "h2/h1 refinement ratio"])
        w.writerow(["r32",              gci["r32"],        "h3/h2 refinement ratio"])
        w.writerow(["p_apparent",       gci["p_app"],
                    "Apparent convergence order" +
                    (" [UNRELIABLE: pre-asymptotic]" if not gci.get("p_reliable", True) else "")])
        w.writerow(["p_reliable",       gci.get("p_reliable", True), "False = pre-asymptotic regime"])
        w.writerow(["eps_fine",         gci["f1"],         "L2 error at finest level"])
        f_ext_val = gci["f_exact"]
        w.writerow(["eps_extrapolated",
                    "" if np.isnan(f_ext_val) else f_ext_val,
                    "Richardson-extrapolated value (blank = pre-asymptotic, not reliable)"])
        w.writerow(["e21_approx_pct",   gci["e21_a"]*100, "Approximate relative error (%)"])
        e21_ext_val = gci["e21_ext"]
        w.writerow(["e21_extrap_pct",
                    "" if np.isnan(e21_ext_val) else e21_ext_val*100,
                    "Extrapolated relative error (%) — blank when pre-asymptotic"])
        w.writerow(["GCI_fine_pct",     gci["gci_fine"]*100,"GCI on finest grid (%)"])
        w.writerow(["GCI_med_pct",      gci["gci_med"]*100, "GCI on medium grid (%)"])
        w.writerow(["asymptotic_ratio", gci["gci_ratio"],  "Should be ~1.0"])
    print(f"  [saved] gci_table.csv")


def save_metrics_csv(df: pd.DataFrame, out_dir: Path):
    path = out_dir / "convergence_analysis_v2.csv"
    df.to_csv(path, index=False)
    print(f"  [saved] {path.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  FEM Convergence Analysis v2  —  Full post-processing")
    print("=" * 70)
    print(f"  Study dir : {STUDY_DIR.resolve()}")
    print(f"  Output dir: {OUTPUT_DIR.resolve()}")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load scalar metrics ───────────────────────────────────────────────────
    df = load_csv_metrics(STUDY_DIR)

    # Normalise column names (handle slight naming differences)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    rename_map = {
        "requested_epl":        "epl",
        "eps_l2_roi":          "eps_L2_roi",
        "max_p_pa":            "max_p_Pa",
        "solve_time_s":        "solve_time_s",
        "mean_trap_err_m":     "mean_trap_err_m",
        "mean_trap_error_m":   "mean_trap_err_m",
        "max_trap_error_m":    "max_trap_err_m",
        "cl_spacing_m":        "cl_spacing_m",
        "centreline_min_spacing_mm": "centreline_min_spacing_mm",
        "spacing_err_pct":     "spacing_err_pct",
        "trap_spacing_err_pct": "spacing_err_pct",
        "pml_n_wavelengths_xy":"pml_n_wavelengths_xy",
        "physical_size_mm":    "physical_size_mm",
        "n_traps_roi":         "n_traps",
        "n_matched_traps":     "n_matched_traps",
        "n_unmatched_current": "n_unmatched_current",
        "n_unmatched_reference": "n_unmatched_reference",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    if "cl_spacing_m" not in df.columns and "centreline_min_spacing_mm" in df.columns:
        df["cl_spacing_m"] = pd.to_numeric(df["centreline_min_spacing_mm"], errors="coerce") * 1e-3

    numeric_cols = [
        "epl", "eps_L2_roi", "max_p_Pa", "solve_time_s", "mean_trap_err_m",
        "max_trap_err_m", "cl_spacing_m", "spacing_err_pct",
        "pml_n_wavelengths_xy", "physical_size_mm", "dofs", "n_traps",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill missing PML column if absent
    if "pml_n_wavelengths_xy" not in df.columns:
        df["pml_n_wavelengths_xy"] = 1.0

    print(f"  Columns: {list(df.columns)}")
    print()

    # ── Richardson extrapolation ──────────────────────────────────────────────
    print("[1/8] Richardson extrapolation + GCI")
    gci = compute_re_gci(df)
    if gci:
        rel = "" if gci.get("p_reliable", True) else "  [UNRELIABLE — pre-asymptotic; GCI uses p=2]"
        print(f"  Apparent order p = {gci['p_app']:.3f}{rel}")
        print(f"  Extrapolated ε*  = {gci['f_exact']:.4e}")
        print(f"  GCI fine grid    = {gci['gci_fine']*100:.2f} %  (conservative upper bound)")
        print(f"  Asymptotic ratio = {gci['gci_ratio']:.4f}  (target ≈ 1.0)")
    else:
        print("  [skip] GCI unavailable — need three finite reduced-domain error levels")
        _remove_if_exists(OUTPUT_DIR / "fig3_richardson_gci.png")
    save_gci_csv(gci, OUTPUT_DIR)
    save_metrics_csv(df, OUTPUT_DIR)

    # ── Figures from CSV (no field arrays needed) ─────────────────────────────
    print("\n[2/8] Figure 1 — error vs h (log-log)")
    fig_error_vs_h(df, OUTPUT_DIR)

    print("[3/8] Figure 2 — error vs EPL (3-panel)")
    fig_error_vs_epl(df, OUTPUT_DIR)

    print("[4/8] Figure 3 — Richardson extrapolation summary")
    fig_richardson_gci(df, gci, OUTPUT_DIR)

    print("[5/8] Figure 6 — domain-size sensitivity")
    fig_domain_sensitivity(df, OUTPUT_DIR)

    print("[6/8] Figure 7 — PML sensitivity")
    fig_pml_sensitivity(df, OUTPUT_DIR)

    print("[7/8] Figure 8 — solve time vs DOFs")
    fig_solve_time(df, OUTPUT_DIR)

    # ── Figures 4 & 5: trap overlays from pre-computed NPZ files ─────────────
    print("\n[8/8] Figures 4 & 5 — trap overlays (loading from NPZ files)")
    npz_files = discover_npz_files(STUDY_DIR)

    if not npz_files:
        print("  [skip] No NPZ files found under", STUDY_DIR)
    else:
        # Identify reference (EPL=5, phys=3mm, PML=1.0) and overlay NPZs
        ref_npz_path    = None
        coarse_npz_path = None
        fine_npz_path   = None

        for stem, npz_path in npz_files.items():
            meta = _parse_npz_meta(stem)
            epl  = meta.get("epl")
            size = meta.get("physical_size_mm", 3.0)
            pml  = meta.get("pml_n_wavelengths_xy", 1.0)
            if epl is None:
                continue
            # Only take the first match (files sorted alphabetically = earliest timestamp)
            if (ref_npz_path is None
                    and abs(epl - REFERENCE_EPL) < 0.01
                    and abs(size - 3.0) < 0.1
                    and abs(pml - 1.0) < 0.05):
                ref_npz_path = npz_path
            elif (coarse_npz_path is None
                    and abs(epl - 2.0) < 0.01 and abs(size - 3.0) < 0.1):
                coarse_npz_path = npz_path
            elif (fine_npz_path is None
                    and abs(epl - 4.5) < 0.01 and abs(size - 3.0) < 0.1):
                fine_npz_path = npz_path

        # Load reference traps from NPZ
        ref_traps = None
        if ref_npz_path is not None:
            print(f"  Loading reference NPZ: {ref_npz_path.name}")
            ref_data = load_npz(ref_npz_path)
            if ref_data is not None:
                ref_traps = ref_data["traps"]
                print(f"  Reference traps: {len(ref_traps)}")
        else:
            print("  [warn] Reference NPZ (EPL=5, phys=3mm, PML=1.0) not found")

        # Coarse overlay
        if coarse_npz_path is not None:
            print(f"  Generating coarse overlay: {coarse_npz_path.name}")
            fig_trap_overlay(load_npz(coarse_npz_path), ref_traps,
                             "2.0", "fig4_trap_overlay_coarse.png", OUTPUT_DIR)
        else:
            print("  [skip] fig4 — EPL=2.0 / phys=3mm NPZ not found")

        # Fine overlay
        if fine_npz_path is not None:
            print(f"  Generating fine overlay: {fine_npz_path.name}")
            fig_trap_overlay(load_npz(fine_npz_path), ref_traps,
                             "4.5", "fig5_trap_overlay_fine.png", OUTPUT_DIR)
        else:
            print("  [skip] fig5 — EPL=4.5 / phys=3mm NPZ not found")

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  DONE.  Output written to:", OUTPUT_DIR.resolve())
    print()
    print("  Figures:")
    for p in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"    {p.name}")
    print()
    print("  Tables:")
    for p in sorted(OUTPUT_DIR.glob("*.csv")):
        print(f"    {p.name}")
    print("=" * 70)


if __name__ == "__main__":
    main()
