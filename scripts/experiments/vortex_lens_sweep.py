#!/usr/bin/env python3
"""
Vortex Lens Investigation — Phase 1: Lens/Field Sweep + Ranking
================================================================

Compares LG, Bessel, Bessel–Gaussian, and Plastic (baseline) vortex
families by sweeping parameters, computing metrics on xy-plane at z*,
and ranking by a composite score J.

Each (family, ℓ, family_params, offset, apod) requires ONE vortex FEM
solve.  Standing-wave solves are reused.  Combined fields for each α
are assembled post-hoc on the saved grids.

Output:
    results/vortex_lens_sweep_YYYYMMDD_HHMMSS/
      csv/base_metrics.csv
      csv/alpha_metrics.csv
      figures/family_{FAM}_l{ell}_best_6panel.png
      figures/compare_families_l{ell}_panel.png
      figures/ranking_scatter.png
      results.json
      VORTEX_LENS_SWEEP_NOTE.md  (written separately)

Runtime estimate: ~80 vortex solves × 45 s + 1 standing solve ≈ 65 min
"""
from __future__ import annotations

import csv
import json
import os
import sys
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.interpolate import RegularGridInterpolator

# ── Project path ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from acoustweezers.experiments.farfield_petri_cuboid.presets import (
    CORRECTED_PRESET,
)

# =====================================================================
# Constants
# =====================================================================
WATER_DENSITY = 997.0
WATER_C = 1484.0
F_HZ = 2.0e6
OMEGA = 2 * np.pi * F_HZ
LAM = WATER_C / F_HZ               # 0.742 mm
K_WATER = OMEGA / WATER_C           # ≈ 8467.9 rad/m
R_DISK = 1.0e-3                     # 1 mm aperture
DOMAIN_CENTRE = (3.0e-3, 3.0e-3)   # domain centre in m

# Particle (PS bead)
PARTICLE_RADIUS = 5e-6
RHO_P, C_P = 1050.0, 2350.0
F1_CONTRAST = 1 - WATER_DENSITY * WATER_C**2 / (RHO_P * C_P**2)
F2_CONTRAST = 2 * (RHO_P - WATER_DENSITY) / (2 * RHO_P + WATER_DENSITY)

# From pre-lens affirmation
Z_MID = (CORRECTED_PRESET["H_under"] + CORRECTED_PRESET["H_under"] +
         CORRECTED_PRESET.get("H_top", 2.0085e-3)) / 2
# Simpler: z_mid = H_under + H_top / 2
H_UNDER = CORRECTED_PRESET["H_under"]           # 3e-3
H_TOP = CORRECTED_PRESET.get("H_top", 2.0085e-3)
Z_MID_CORRECT = H_UNDER + H_TOP / 2
Z_STAR = Z_MID_CORRECT + 0.25 * LAM              # 4.190 mm

N_GRID = 200
ELEM_PER_LAM = 4

# ROI
ROI_RADIUS = 1.5 * LAM    # ≈ 1.113 mm
CX, CY = DOMAIN_CENTRE

# Alpha values for combined-field post-hoc sweep
ALPHA_VALUES = [0.05, 0.10, 0.20, 0.40]

# Hard constraint thresholds
ETA_THR = 0.02
CORR_THR = 0.995

# Composite score weights
W1, W2, W3, W4, W5 = 0.20, 0.15, 0.25, 0.25, 0.15

# =====================================================================
# Output directory
# =====================================================================
TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
OUT_DIR = PROJECT_ROOT / "results" / f"vortex_lens_sweep_{TIMESTAMP}"
FIG_DIR = OUT_DIR / "figures"
CSV_DIR = OUT_DIR / "csv"


# =====================================================================
# Config builder helpers
# =====================================================================
BASE_CFG = {
    **CORRECTED_PRESET,
    "elements_per_wavelength": ELEM_PER_LAM,
}


def make_vortex_cfg(
    family: str,
    ell: int,
    offset: Tuple[float, float] = (0.0, 0.0),
    apod: str = "cosine_taper",
    # Family-specific
    beam_waist: Optional[float] = None,
    focal_length: Optional[float] = None,
    k_r: Optional[float] = None,
) -> dict:
    """Build a FarFieldConfig overrides dict for a vortex-only solve."""
    cfg = {**BASE_CFG}
    cfg["standing_velocity_amplitude"] = 0.0       # vortex-only
    cfg["disk_velocity_amplitude"] = 1e-6          # V_disk baseline

    cfg["lens_l"] = ell
    cfg["lens_focus_offset_x"] = offset[0]
    cfg["lens_focus_offset_y"] = offset[1]
    cfg["lens_apodization"] = apod

    if family == "lg":
        cfg["lens_drive"] = "lg"
        cfg["lens_beam_waist"] = beam_waist or R_DISK * 0.6
        cfg["lens_focal_length"] = focal_length if focal_length else 0.0
    elif family == "bessel":
        cfg["lens_drive"] = "bessel"
        cfg["lens_k_r"] = k_r or K_WATER
        cfg["lens_beam_waist"] = None
    elif family == "bg":
        cfg["lens_drive"] = "bessel_gauss"
        cfg["lens_k_r"] = k_r or 0.5 * K_WATER
        cfg["lens_beam_waist"] = beam_waist or R_DISK * 0.6
    elif family == "plastic":
        cfg["lens_drive"] = "plastic"
        cfg["lens_focal_length"] = focal_length or 2.0e-3
    else:
        raise ValueError(f"Unknown family: {family}")

    return cfg


def make_standing_cfg() -> dict:
    """Build overrides for the standing-only solve."""
    cfg = {**BASE_CFG}
    cfg["disk_velocity_amplitude"] = 0.0   # no vortex
    return cfg


# =====================================================================
# Subprocess solver (reuse _solve_worker.py pattern)
# =====================================================================
WORKER_SCRIPT = PROJECT_ROOT / "scripts" / "experiments" / "_solve_worker.py"


def solve_to_grid(overrides: dict, label: str,
                  trap_z: float = Z_STAR,
                  mid_y: float = CY,
                  n_xy: int = N_GRID) -> Tuple:
    """
    Solve via subprocess, return (xg, yg, p_xy, xg_xz, zg_xz, p_xz, phys_max, solve_time).
    Also caches .npz in OUT_DIR.
    """
    result_file = str(OUT_DIR / f"_grid_{label}.npz")

    # If already solved (e.g. standing reuse), load cached
    if os.path.exists(result_file):
        d = np.load(result_file)
        return (d["xg"], d["yg"], d["p_xy"],
                d["xg_xz"], d["zg_xz"], d["p_xz"],
                float(d["phys_max"]), float(d["solve_time"]))

    args = {
        "overrides": overrides,
        "label": label,
        "trap_z": trap_z,
        "mid_y": mid_y,
        "n_xy": n_xy,
        "result_file": result_file,
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(args, f, default=str)
        args_file = f.name

    try:
        proc = subprocess.run(
            [sys.executable, str(WORKER_SCRIPT), args_file],
            capture_output=False, timeout=600,
        )
        if proc.returncode != 0:
            print(f"  *** SOLVE FAILED: {label}  rc={proc.returncode}", flush=True)
            return None
    finally:
        os.unlink(args_file)

    d = np.load(result_file)
    return (d["xg"], d["yg"], d["p_xy"],
            d["xg_xz"], d["zg_xz"], d["p_xz"],
            float(d["phys_max"]), float(d["solve_time"]))


# =====================================================================
# Physics helpers
# =====================================================================

def gorkov_2d(p_grid, dx, dy, omega=OMEGA, rho=WATER_DENSITY, c=WATER_C,
              a=PARTICLE_RADIUS):
    """Gor'kov potential and force on 2D grid. Returns (U, Fx, Fy)."""
    f1, f2 = F1_CONTRAST, F2_CONTRAST
    K = rho * c**2
    p2 = np.abs(p_grid)**2 / 2
    dp_dx = np.gradient(p_grid, dx, axis=1)
    dp_dy = np.gradient(p_grid, dy, axis=0)
    grad_p2 = np.abs(dp_dx)**2 + np.abs(dp_dy)**2
    v2 = grad_p2 / (2 * omega**2 * rho**2)
    pf = (4 * np.pi / 3) * a**3
    U = pf * (f1 * p2 / (2 * K) - f2 * (3 * rho / 4) * v2)
    Fx = -np.gradient(U, dx, axis=1)
    Fy = -np.gradient(U, dy, axis=0)
    return U, Fx, Fy


def make_roi(xg, yg, cx, cy, radius):
    """Circular ROI boolean mask, shape (ny, nx)."""
    X, Y = np.meshgrid(xg, yg)
    return (X - cx)**2 + (Y - cy)**2 <= radius**2


def make_annulus(xg, yg, cx, cy, r_in, r_out):
    """Annular mask."""
    X, Y = np.meshgrid(xg, yg)
    d2 = (X - cx)**2 + (Y - cy)**2
    return (d2 >= r_in**2) & (d2 <= r_out**2)


def combine(p_stand, p_vortex, alpha):
    """Post-hoc superposition: p_comb = p_stand + α·(V_stand/V_vortex)·p_vortex."""
    V_ratio = CORRECTED_PRESET["standing_velocity_amplitude"] / \
              CORRECTED_PRESET["disk_velocity_amplitude"]
    return p_stand + alpha * V_ratio * p_vortex


def winding_number(p_grid, xg, yg, cx, cy, radius, n_angles=360):
    """Phase winding number around a circle."""
    phase = np.angle(p_grid)
    interp = RegularGridInterpolator((yg, xg), phase,
                                     method='nearest',
                                     bounds_error=False, fill_value=0.0)
    theta = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    pts = np.column_stack([cy + radius * np.sin(theta),
                           cx + radius * np.cos(theta)])
    phi = interp(pts)
    dphi = np.diff(phi)
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
    return float(np.sum(dphi) / (2 * np.pi))


# =====================================================================
# Metric computation
# =====================================================================

def compute_metrics(
    p_stand: np.ndarray,
    p_vortex: np.ndarray,
    p_comb: np.ndarray,
    xg: np.ndarray,
    yg: np.ndarray,
    roi_mask: np.ndarray,
    label: str = "",
) -> Dict[str, Any]:
    """Compute all metrics for one configuration at one alpha."""
    dx = xg[1] - xg[0]
    dy = yg[1] - yg[0]
    out = ~roi_mask
    m: Dict[str, Any] = {"label": label}

    # ── Global non-disruption ────────────────────────────────────
    delta = p_comb - p_stand
    norm_s = np.sqrt(np.sum(np.abs(p_stand[out])**2))
    norm_d = np.sqrt(np.sum(np.abs(delta[out])**2))
    m["eta_out"] = float(norm_d / norm_s) if norm_s > 0 else float('nan')

    s_out = np.abs(p_stand[out])
    c_out = np.abs(p_comb[out])
    if np.std(s_out) > 0 and np.std(c_out) > 0:
        m["corr_out"] = float(np.corrcoef(s_out, c_out)[0, 1])
    else:
        m["corr_out"] = float('nan')

    # Neighbour trap crosstalk: detect traps in standing, measure ΔU
    U_s, Fx_s, Fy_s = gorkov_2d(p_stand, dx, dy)
    U_c, Fx_c, Fy_c = gorkov_2d(p_comb, dx, dy)

    # Find trap minima in standing (local minima of U)
    from scipy.ndimage import minimum_filter, label as nd_label
    filt_size = max(3, int(0.3 * LAM / dx))
    U_min_filt = minimum_filter(U_s, size=filt_size)
    traps = (U_s == U_min_filt) & out  # traps outside ROI
    trap_coords = np.argwhere(traps)
    n_traps = len(trap_coords)

    if n_traps > 0:
        # For each trap, compute local depth = max(U in 1λ box) - U_min
        depths_s = []
        depths_c = []
        for iy, ix in trap_coords:
            half_w = max(1, int(0.5 * LAM / dx))
            sl_y = slice(max(0, iy - half_w), min(U_s.shape[0], iy + half_w + 1))
            sl_x = slice(max(0, ix - half_w), min(U_s.shape[1], ix + half_w + 1))
            d_s = float(U_s[sl_y, sl_x].max() - U_s[iy, ix])
            d_c = float(U_c[sl_y, sl_x].max() - U_c[iy, ix])
            if d_s > 0:
                depths_s.append(d_s)
                depths_c.append(d_c)
        if depths_s:
            pct = [100 * abs(c - s) / s for s, c in zip(depths_s, depths_c) if s > 0]
            m["trap_crosstalk_mean_pct"] = float(np.mean(pct)) if pct else 0.0
            m["trap_crosstalk_max_pct"] = float(np.max(pct)) if pct else 0.0
        else:
            m["trap_crosstalk_mean_pct"] = 0.0
            m["trap_crosstalk_max_pct"] = 0.0
    else:
        m["trap_crosstalk_mean_pct"] = 0.0
        m["trap_crosstalk_max_pct"] = 0.0
    m["n_traps_outside"] = n_traps

    # ── Local vortex quality ─────────────────────────────────────
    roi = roi_mask

    # Energy fraction
    v_energy = np.abs(p_vortex)**2
    total_v = np.sum(v_energy)
    m["E_roi"] = float(np.sum(v_energy[roi]) / total_v) if total_v > 0 else 0.0

    # Core quality
    p_v_mag = np.abs(p_vortex)
    if roi.any():
        core_min = float(p_v_mag[roi].min())
        peak_roi = float(p_v_mag[roi].max())
    else:
        core_min = peak_roi = 0.0
    m["core_min_Pa"] = core_min
    m["peak_roi_Pa"] = peak_roi

    # Core-to-sidelobe ratio
    annulus = make_annulus(xg, yg, CX, CY, 1.5 * LAM, 3.0 * LAM)
    if annulus.any():
        peak_annulus = float(p_v_mag[annulus].max())
    else:
        peak_annulus = float(p_v_mag[out].max()) if out.any() else 1.0
    m["core_ratio"] = float(peak_roi / peak_annulus) if peak_annulus > 0 else float('inf')

    # Topology: winding number at multiple radii
    windings = {}
    winding_vals = []
    for r_mult in [1.0, 1.5, 2.0]:
        w = winding_number(p_vortex, xg, yg, CX, CY, r_mult * LAM)
        windings[f"w_{r_mult:.1f}lam"] = round(w, 3)
        winding_vals.append(w)
    m.update(windings)

    # Winding stability: std of absolute winding across radii
    m["winding_std"] = float(np.std([abs(w) for w in winding_vals]))

    # Core position
    X, Y = np.meshgrid(xg, yg)
    if roi.any() and np.sum(v_energy[roi]) > 0:
        centroid_x = float(np.sum(X[roi] * v_energy[roi]) / np.sum(v_energy[roi]))
        centroid_y = float(np.sum(Y[roi] * v_energy[roi]) / np.sum(v_energy[roi]))
    else:
        centroid_x = centroid_y = float('nan')
    m["centroid_x_mm"] = centroid_x * 1e3
    m["centroid_y_mm"] = centroid_y * 1e3

    # Core (min |p|) position in ROI
    if roi.any():
        roi_pmag = p_v_mag.copy()
        roi_pmag[~roi] = np.inf
        core_idx = np.unravel_index(roi_pmag.argmin(), roi_pmag.shape)
        m["core_x_mm"] = float(xg[core_idx[1]]) * 1e3
        m["core_y_mm"] = float(yg[core_idx[0]]) * 1e3
    else:
        m["core_x_mm"] = m["core_y_mm"] = float('nan')

    # Barrier
    if roi.any():
        from scipy.ndimage import binary_dilation
        boundary = binary_dilation(roi, iterations=2) & (~roi)
        U_min_roi = float(U_s[roi].min())
        U_saddle = float(U_s[boundary].max()) if boundary.any() else float(U_s[roi].max())
        m["barrier_stand_J"] = U_saddle - U_min_roi

        U_min_c = float(U_c[roi].min())
        U_saddle_c = float(U_c[boundary].max()) if boundary.any() else float(U_c[roi].max())
        m["barrier_comb_J"] = U_saddle_c - U_min_c
        m["barrier_reduction_pct"] = (
            100 * (1 - m["barrier_comb_J"] / m["barrier_stand_J"])
            if m["barrier_stand_J"] > 0 else float('nan')
        )
    else:
        m["barrier_stand_J"] = m["barrier_comb_J"] = 0.0
        m["barrier_reduction_pct"] = float('nan')

    # ── Composite score J ────────────────────────────────────────
    # Hard constraints
    if m["eta_out"] > ETA_THR or m["corr_out"] < CORR_THR:
        m["pass_hard"] = False
        m["J"] = -1.0
    else:
        m["pass_hard"] = True
        # winding_std penalty capped at 1
        ws_capped = min(m["winding_std"], 1.0)
        # core_ratio capped to prevent inf
        cr = min(m["core_ratio"], 10.0) / 10.0
        m["J"] = (
            W1 * (1 - m["eta_out"] / ETA_THR) +
            W2 * m["corr_out"] +
            W3 * m["E_roi"] +
            W4 * cr +
            W5 * (1 - ws_capped)
        )

    return m


# =====================================================================
# Parameter space enumeration
# =====================================================================

@dataclass
class SweepConfig:
    """One point in the parameter sweep."""
    family: str
    ell: int
    offset: Tuple[float, float]
    apod: str
    # Family-specific
    beam_waist: Optional[float] = None    # LG, BG
    focal_length: Optional[float] = None  # LG, Plastic
    k_r: Optional[float] = None           # Bessel, BG
    # Metadata
    label: str = ""

    @property
    def config_id(self) -> str:
        """Unique short ID for this configuration."""
        parts = [self.family, f"l{self.ell}"]
        if self.beam_waist is not None:
            parts.append(f"w{self.beam_waist*1e3:.1f}")
        if self.focal_length is not None:
            parts.append(f"f{self.focal_length*1e3:.1f}")
        if self.k_r is not None:
            parts.append(f"kr{self.k_r:.0f}")
        ox, oy = self.offset
        if ox != 0 or oy != 0:
            parts.append(f"off{ox*1e3:.1f}")
        parts.append(self.apod[:3])
        return "_".join(parts)


def build_sweep_configs() -> List[SweepConfig]:
    """
    Enumerate all sweep configurations.

    Pruned parameter space to keep total vortex solves manageable (~80).
    We use 1 offset (0,0) and 1 apod per family as default, with a
    small cross-check set at offset=(0.2mm,0).
    """
    configs = []
    R = R_DISK  # 1 mm

    offsets_main = [(0.0, 0.0)]
    offsets_check = [(0.2e-3, 0.0)]
    apods_main = ["cosine_taper"]
    apods_check = ["uniform"]

    # --- FAMILY 1: LG ---
    for ell in [1, 2, 3]:
        for w_frac in [0.4, 0.6, 0.8]:
            w = w_frac * R
            # No focus
            configs.append(SweepConfig(
                family="lg", ell=ell, offset=(0, 0), apod="cosine_taper",
                beam_waist=w, focal_length=None))
            # With focus
            for f_mm in [1.5, 2.0, 2.5]:
                configs.append(SweepConfig(
                    family="lg", ell=ell, offset=(0, 0), apod="cosine_taper",
                    beam_waist=w, focal_length=f_mm * 1e-3))

    # LG offset check (best w = 0.6R, f = 2mm, ℓ=1 only)
    configs.append(SweepConfig(
        family="lg", ell=1, offset=(0.2e-3, 0), apod="cosine_taper",
        beam_waist=0.6 * R, focal_length=2e-3))
    configs.append(SweepConfig(
        family="lg", ell=1, offset=(0, 0), apod="uniform",
        beam_waist=0.6 * R, focal_length=2e-3))

    # --- FAMILY 2: Bessel ---
    for ell in [1, 2, 3]:
        for kr_frac in [0.5, 1.0, 1.5]:
            kr = kr_frac * K_WATER
            configs.append(SweepConfig(
                family="bessel", ell=ell, offset=(0, 0), apod="uniform",
                k_r=kr))

    # Bessel offset + apod checks (ℓ=1 only)
    configs.append(SweepConfig(
        family="bessel", ell=1, offset=(0.2e-3, 0), apod="uniform",
        k_r=1.0 * K_WATER))
    configs.append(SweepConfig(
        family="bessel", ell=1, offset=(0, 0), apod="cosine_taper",
        k_r=1.0 * K_WATER))

    # --- FAMILY 3: BG ---
    for ell in [1, 2, 3]:
        for kr_frac in [0.5, 1.0, 1.5]:
            kr = kr_frac * K_WATER
            for w_frac in [0.4, 0.6, 0.8]:
                w = w_frac * R
                configs.append(SweepConfig(
                    family="bg", ell=ell, offset=(0, 0), apod="cosine_taper",
                    k_r=kr, beam_waist=w))

    # BG offset check
    configs.append(SweepConfig(
        family="bg", ell=1, offset=(0.2e-3, 0), apod="cosine_taper",
        k_r=0.5 * K_WATER, beam_waist=0.6 * R))

    # --- FAMILY 4: Plastic baseline ---
    for ell in [1, 2, 3]:
        for f_mm in [1.5, 2.0, 2.5]:
            configs.append(SweepConfig(
                family="plastic", ell=ell, offset=(0, 0), apod="cosine_taper",
                focal_length=f_mm * 1e-3))
            # Also with offset matching CORRECTED_PRESET
            configs.append(SweepConfig(
                family="plastic", ell=ell, offset=(0.2e-3, 0), apod="cosine_taper",
                focal_length=f_mm * 1e-3))

    print(f"  Total sweep configs: {len(configs)}")
    return configs


# =====================================================================
# Main sweep
# =====================================================================

def main():
    t0_global = time.time()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Vortex Lens Sweep — Phase 1")
    print(f"Output: {OUT_DIR}")
    print(f"z* = {Z_STAR*1e3:.4f} mm    λ = {LAM*1e3:.3f} mm    "
          f"k = {K_WATER:.1f} rad/m")
    print(f"ROI radius = {ROI_RADIUS*1e3:.3f} mm    "
          f"elem/λ = {ELEM_PER_LAM}")

    # ── Build parameter space ─────────────────────────────────────
    configs = build_sweep_configs()

    # ── Solve standing-wave once (reused for all) ─────────────────
    print(f"\n{'='*72}")
    print("Solving standing-wave (reused for all configs) ...")
    print("=" * 72)
    stand_cfg = make_standing_cfg()
    stand_result = solve_to_grid(stand_cfg, "standing", Z_STAR, CY, N_GRID)
    if stand_result is None:
        print("FATAL: standing-wave solve failed!")
        return
    xg_s, yg_s, p_stand, _, _, _, pm_s, st_s = stand_result
    print(f"  Standing: max|p|={pm_s:.3f} Pa  solve_time={st_s:.1f}s")

    # ROI mask
    roi_mask = make_roi(xg_s, yg_s, CX, CY, ROI_RADIUS)

    # ── Solve each vortex configuration ───────────────────────────
    solve_log = [("standing", st_s)]
    results_all = []    # list of dicts, one per config × alpha
    base_results = []   # vortex-only base metrics (no combined)
    vortex_cache = {}   # config_id → p_vortex (avoid re-solving identical configs)

    n_total = len(configs)
    n_solves = 0

    print(f"\n{'='*72}")
    print(f"Sweeping {n_total} vortex configurations ...")
    print("=" * 72)

    for i, sc in enumerate(configs):
        cid = sc.config_id
        print(f"\n  [{i+1}/{n_total}] {cid}")

        # Build overrides
        overrides = make_vortex_cfg(
            family=sc.family, ell=sc.ell, offset=sc.offset, apod=sc.apod,
            beam_waist=sc.beam_waist, focal_length=sc.focal_length, k_r=sc.k_r,
        )

        # Solve vortex (or load cached)
        if cid in vortex_cache:
            p_vortex = vortex_cache[cid]
            print(f"    (cached)")
        else:
            vresult = solve_to_grid(overrides, f"vortex_{cid}", Z_STAR, CY, N_GRID)
            if vresult is None:
                print(f"    SKIP: solve failed")
                continue
            xg_v, yg_v, p_vortex, _, _, _, pm_v, st_v = vresult
            solve_log.append((cid, st_v))
            n_solves += 1
            vortex_cache[cid] = p_vortex
            print(f"    max|p_v|={pm_v:.3f} Pa  {st_v:.1f}s")

        # ── Compute metrics at each alpha ─────────────────────────
        for alpha in ALPHA_VALUES:
            p_comb = combine(p_stand, p_vortex, alpha)
            m = compute_metrics(p_stand, p_vortex, p_comb,
                                xg_s, yg_s, roi_mask,
                                label=f"{cid}_a{alpha:.2f}")

            # Attach config metadata
            m["config_id"] = cid
            m["family"] = sc.family
            m["ell"] = sc.ell
            m["offset_x_mm"] = sc.offset[0] * 1e3
            m["offset_y_mm"] = sc.offset[1] * 1e3
            m["apod"] = sc.apod
            m["alpha"] = alpha
            if sc.beam_waist is not None:
                m["beam_waist_mm"] = sc.beam_waist * 1e3
            else:
                m["beam_waist_mm"] = float('nan')
            if sc.focal_length is not None:
                m["focal_length_mm"] = sc.focal_length * 1e3
            else:
                m["focal_length_mm"] = float('nan')
            if sc.k_r is not None:
                m["k_r"] = sc.k_r
            else:
                m["k_r"] = float('nan')

            results_all.append(m)

        # Also store vortex-only base metrics (for Stage 4 topology)
        m_base = compute_metrics(p_stand, p_vortex,
                                 combine(p_stand, p_vortex, 0.20),
                                 xg_s, yg_s, roi_mask,
                                 label=f"{cid}_base")
        m_base["config_id"] = cid
        m_base["family"] = sc.family
        m_base["ell"] = sc.ell
        m_base["_p_vortex"] = p_vortex   # stash for plotting
        base_results.append(m_base)

    total_solve_time = sum(t for _, t in solve_log)
    print(f"\n  Total: {n_solves} vortex solves + 1 standing = "
          f"{n_solves+1} FEM runs  ({total_solve_time:.0f}s)")

    # =====================================================================
    # Stage 5 — CSV outputs
    # =====================================================================
    print(f"\n{'='*72}")
    print("Writing CSV outputs ...")
    print("=" * 72)

    # Base metrics CSV (one row per config, at α=0.20)
    base_keys = [
        "config_id", "family", "ell", "offset_x_mm", "apod",
        "beam_waist_mm", "focal_length_mm", "k_r",
        "eta_out", "corr_out", "E_roi", "core_ratio",
        "core_min_Pa", "peak_roi_Pa",
        "w_1.0lam", "w_1.5lam", "w_2.0lam", "winding_std",
        "centroid_x_mm", "centroid_y_mm",
        "core_x_mm", "core_y_mm",
        "trap_crosstalk_mean_pct", "trap_crosstalk_max_pct",
        "barrier_reduction_pct", "pass_hard", "J",
    ]

    # Filter to alpha=0.20 rows
    rows_020 = [r for r in results_all if abs(r.get("alpha", 0) - 0.20) < 1e-6]
    with open(CSV_DIR / "base_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=base_keys, extrasaction='ignore')
        w.writeheader()
        for row in rows_020:
            w.writerow(row)

    # Alpha metrics CSV (all alpha values)
    alpha_keys = [
        "config_id", "family", "ell", "alpha",
        "eta_out", "corr_out", "E_roi", "core_ratio",
        "barrier_reduction_pct", "pass_hard", "J",
    ]
    with open(CSV_DIR / "alpha_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=alpha_keys, extrasaction='ignore')
        w.writeheader()
        for row in results_all:
            w.writerow(row)

    # =====================================================================
    # Ranking
    # =====================================================================
    print(f"\n{'='*72}")
    print("Ranking configs ...")
    print("=" * 72)

    # Sort by J descending (alpha=0.20)
    ranked = sorted(rows_020, key=lambda r: r.get("J", -999), reverse=True)

    # Best per family per ℓ
    best_per_fam = {}
    for r in ranked:
        key = (r["family"], r["ell"])
        if key not in best_per_fam:
            best_per_fam[key] = r

    # Top 5 overall
    top5 = ranked[:5]
    print("\n  TOP 5 CONFIGS (α=0.20):")
    for i, r in enumerate(top5):
        print(f"    {i+1}. {r['config_id']}  J={r['J']:.4f}  "
              f"eta={r['eta_out']:.4f}  corr={r['corr_out']:.5f}  "
              f"E_roi={r['E_roi']:.3f}  core_ratio={r['core_ratio']:.2f}")

    # Fail list
    fails = [r for r in rows_020 if not r.get("pass_hard", True)]
    print(f"\n  FAILS (hard constraint): {len(fails)} configs")

    # =====================================================================
    # Stage 4 — Panel PNGs
    # =====================================================================
    print(f"\n{'='*72}")
    print("Generating panel PNGs ...")
    print("=" * 72)

    dx = xg_s[1] - xg_s[0]
    dy = yg_s[1] - yg_s[0]
    xg_mm = xg_s * 1e3
    yg_mm = yg_s * 1e3
    extent_mm = [xg_mm[0], xg_mm[-1], yg_mm[0], yg_mm[-1]]

    # ── A) Per-family best panels (per ℓ) ────────────────────────
    for (fam, ell), best_r in best_per_fam.items():
        cid = best_r["config_id"]
        # Find corresponding base_result to get p_vortex
        br = [b for b in base_results if b["config_id"] == cid]
        if not br:
            continue
        p_vortex = br[0]["_p_vortex"]
        p_comb = combine(p_stand, p_vortex, 0.20)
        delta_p = np.abs(p_comb) - np.abs(p_stand)

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))

        # |p| standing
        ax = axes[0, 0]
        im = ax.imshow(np.abs(p_stand), extent=extent_mm, origin='lower', cmap='hot')
        ax.set_title("|p| standing", fontsize=9)
        plt.colorbar(im, ax=ax, shrink=0.7)
        ax.add_patch(Circle((CX*1e3, CY*1e3), ROI_RADIUS*1e3,
                             fill=False, ec='cyan', lw=1, ls='--'))

        # |p| vortex
        ax = axes[0, 1]
        im = ax.imshow(np.abs(p_vortex), extent=extent_mm, origin='lower', cmap='inferno')
        ax.set_title(f"|p| vortex [{cid}]", fontsize=9)
        plt.colorbar(im, ax=ax, shrink=0.7)
        ax.add_patch(Circle((CX*1e3, CY*1e3), ROI_RADIUS*1e3,
                             fill=False, ec='cyan', lw=1, ls='--'))

        # |p| combined
        ax = axes[0, 2]
        im = ax.imshow(np.abs(p_comb), extent=extent_mm, origin='lower', cmap='hot')
        ax.set_title("|p| combined (α=0.20)", fontsize=9)
        plt.colorbar(im, ax=ax, shrink=0.7)
        ax.add_patch(Circle((CX*1e3, CY*1e3), ROI_RADIUS*1e3,
                             fill=False, ec='cyan', lw=1, ls='--'))

        # Δ|p| with ROI and Ω_out
        ax = axes[1, 0]
        vmax_d = max(abs(delta_p.min()), abs(delta_p.max())) or 1
        im = ax.imshow(delta_p, extent=extent_mm, origin='lower',
                       cmap='RdBu_r', vmin=-vmax_d, vmax=vmax_d)
        ax.set_title("Δ|p| (comb − stand)", fontsize=9)
        plt.colorbar(im, ax=ax, shrink=0.7)
        ax.add_patch(Circle((CX*1e3, CY*1e3), ROI_RADIUS*1e3,
                             fill=False, ec='green', lw=2, ls='-', label='ROI'))

        # Phase of vortex with core marker
        ax = axes[1, 1]
        phase = np.angle(p_vortex)
        im = ax.imshow(phase, extent=extent_mm, origin='lower',
                       cmap='twilight', vmin=-np.pi, vmax=np.pi)
        ax.set_title("arg(p_vortex)", fontsize=9)
        plt.colorbar(im, ax=ax, shrink=0.7)
        if not np.isnan(best_r.get("core_x_mm", float('nan'))):
            ax.plot(best_r["core_x_mm"], best_r["core_y_mm"],
                    'w+', ms=10, mew=2)
        ax.add_patch(Circle((CX*1e3, CY*1e3), ROI_RADIUS*1e3,
                             fill=False, ec='cyan', lw=1, ls='--'))

        # Winding vs radius
        ax = axes[1, 2]
        r_mults = [0.5, 1.0, 1.5, 2.0, 2.5]
        winds = [winding_number(p_vortex, xg_s, yg_s, CX, CY, rm * LAM)
                 for rm in r_mults]
        ax.plot(r_mults, winds, 'o-', color='C0', lw=2)
        ax.axhline(y=ell, color='red', ls='--', alpha=0.6, label=f'ℓ={ell}')
        ax.axhline(y=-ell, color='red', ls=':', alpha=0.4)
        ax.set_xlabel("r / λ")
        ax.set_ylabel("winding number")
        ax.set_title("Topology: winding(r)", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        fig.suptitle(
            f"Family={fam}  ℓ={ell}  best config: {cid}\n"
            f"J={best_r['J']:.4f}  η_out={best_r['eta_out']:.4f}  "
            f"corr={best_r['corr_out']:.5f}  E_roi={best_r['E_roi']:.3f}  "
            f"core_ratio={best_r['core_ratio']:.2f}",
            fontsize=10)
        fig.tight_layout()
        fig.savefig(str(FIG_DIR / f"family_{fam}_l{ell}_best_6panel.png"), dpi=120)
        plt.close(fig)
        print(f"  Saved: family_{fam}_l{ell}_best_6panel.png")

    # ── B) Family comparison panels (per ℓ) ──────────────────────
    families = ["lg", "bessel", "bg", "plastic"]
    for ell in [1, 2, 3]:
        fig, axes = plt.subplots(2, len(families), figsize=(5 * len(families), 9))

        for j, fam in enumerate(families):
            key = (fam, ell)
            if key not in best_per_fam:
                continue
            best_r = best_per_fam[key]
            cid = best_r["config_id"]
            br = [b for b in base_results if b["config_id"] == cid]
            if not br:
                continue
            p_v = br[0]["_p_vortex"]

            # Row 0: |p| vortex
            ax = axes[0, j]
            im = ax.imshow(np.abs(p_v), extent=extent_mm, origin='lower', cmap='inferno')
            ax.set_title(f"{fam} ℓ={ell}", fontsize=9)
            ax.add_patch(Circle((CX*1e3, CY*1e3), ROI_RADIUS*1e3,
                                 fill=False, ec='cyan', lw=1, ls='--'))
            plt.colorbar(im, ax=ax, shrink=0.6)

            # Row 1: phase
            ax = axes[1, j]
            im = ax.imshow(np.angle(p_v), extent=extent_mm, origin='lower',
                           cmap='twilight', vmin=-np.pi, vmax=np.pi)
            ax.set_title(f"phase  η={best_r.get('eta_out',0):.4f}\n"
                         f"E_roi={best_r.get('E_roi',0):.3f}  "
                         f"cr={best_r.get('core_ratio',0):.2f}\n"
                         f"w@2λ={best_r.get('w_2.0lam',0):.1f}",
                         fontsize=8)
            plt.colorbar(im, ax=ax, shrink=0.6)

        fig.suptitle(f"Family comparison — ℓ={ell}  (best config per family, α=0.20)",
                     fontsize=11)
        fig.tight_layout()
        fig.savefig(str(FIG_DIR / f"compare_families_l{ell}_panel.png"), dpi=120)
        plt.close(fig)
        print(f"  Saved: compare_families_l{ell}_panel.png")

    # ── C) Global ranking scatter ─────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    cmap_fam = {"lg": "C0", "bessel": "C1", "bg": "C2", "plastic": "C3"}
    marker_ell = {1: "o", 2: "s", 3: "D"}

    for r in rows_020:
        fam = r.get("family", "")
        ell = r.get("ell", 1)
        color = cmap_fam.get(fam, "gray")
        marker = marker_ell.get(ell, "x")
        ax.scatter(r.get("eta_out", 0), r.get("E_roi", 0),
                   c=color, marker=marker, s=40, alpha=0.7)

    # Label top 5
    for i, r in enumerate(top5):
        ax.annotate(f"#{i+1} {r['config_id']}",
                    (r["eta_out"], r["E_roi"]),
                    fontsize=6, ha='left', va='bottom',
                    arrowprops=dict(arrowstyle='->', lw=0.5) if i < 3 else None)

    # Legend: family + ell
    from matplotlib.lines import Line2D
    handles = []
    for fam, c in cmap_fam.items():
        handles.append(Line2D([0], [0], marker='o', color='w',
                               markerfacecolor=c, label=fam, markersize=8))
    for ell, mk in marker_ell.items():
        handles.append(Line2D([0], [0], marker=mk, color='w',
                               markerfacecolor='gray', label=f'ℓ={ell}', markersize=8))
    ax.legend(handles=handles, fontsize=8, loc='lower right')

    ax.axvline(x=ETA_THR, color='red', ls='--', alpha=0.5, label=f'η_thr={ETA_THR}')
    ax.set_xlabel("η_out (lower is better)")
    ax.set_ylabel("E_roi (higher is better)")
    ax.set_title("Global ranking: η_out vs E_roi (α=0.20)", fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(FIG_DIR / "ranking_scatter.png"), dpi=120)
    plt.close(fig)
    print(f"  Saved: ranking_scatter.png")

    # =====================================================================
    # Stage 5 — JSON output
    # =====================================================================
    print(f"\n{'='*72}")
    print("Writing results.json ...")
    print("=" * 72)

    # Clean results for JSON (remove numpy arrays)
    def clean_for_json(row):
        out = {}
        for k, v in row.items():
            if k.startswith("_"):
                continue
            if isinstance(v, (np.floating, float)):
                out[k] = float(v) if not np.isnan(v) else None
            elif isinstance(v, (np.integer, int)):
                out[k] = int(v)
            elif isinstance(v, bool):
                out[k] = bool(v)
            elif isinstance(v, np.bool_):
                out[k] = bool(v)
            else:
                out[k] = v
        return out

    ranked_json = [clean_for_json(r) for r in ranked]
    best_per_fam_json = {f"{fam}_l{ell}": clean_for_json(r)
                         for (fam, ell), r in best_per_fam.items()}
    top5_json = [clean_for_json(r) for r in top5]
    fails_json = [clean_for_json(r) for r in fails]

    output = {
        "sweep_params": {
            "n_configs": len(configs),
            "n_vortex_solves": n_solves,
            "n_standing_solves": 1,
            "alpha_values": ALPHA_VALUES,
            "z_star_mm": Z_STAR * 1e3,
            "roi_radius_mm": ROI_RADIUS * 1e3,
            "elem_per_lam": ELEM_PER_LAM,
            "n_grid": N_GRID,
            "eta_threshold": ETA_THR,
            "corr_threshold": CORR_THR,
        },
        "ranked_list": ranked_json,
        "best_per_family_per_l": best_per_fam_json,
        "best_overall_top5": top5_json,
        "fail_list": fails_json,
        "solve_log": solve_log,
        "total_time_s": time.time() - t0_global,
    }

    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    total = time.time() - t0_global
    print(f"\n{'='*72}")
    print(f"Vortex Lens Sweep complete.  {total:.0f}s total  "
          f"({n_solves+1} FEM solves).")
    print(f"Results:  {OUT_DIR}")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
