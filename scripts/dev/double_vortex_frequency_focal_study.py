#!/usr/bin/env python3
"""
Double-Vortex Frequency & Focal-Length Feasibility Study
=========================================================

Phase 1 field-family study: investigate whether a focused double-vortex
field can create two distinct stable traps whose separation varies
smoothly with frequency and focal length, potentially merging into a
single shared capture region.

Uses ASM propagation only (no standing-wave superposition).
Produces static contact sheets and summary plots (no GIFs).

Outputs are saved to a timestamped folder under results/dev/.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.ndimage import minimum_filter, label

# ── Project paths ───────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "lib"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from asm_utils import (
    C_WATER,
    F_HZ,
    K0,
    LAM,
    make_lens_phase,
    make_vortex_field,
    propagate_asm,
)
from fem_cache_utils import (
    OMEGA,
    RHO0,
    default_particle_params,
    gorkov_grid_2d,
)

# ════════════════════════════════════════════════════════════════════
# Physical / study parameters
# ════════════════════════════════════════════════════════════════════

# Grid
N_GRID = 400                     # points per axis at source plane
DOMAIN_HALF_MM = 4.0             # ±4 mm domain (8 mm square)
DOMAIN_HALF = DOMAIN_HALF_MM * 1e-3

# Double-vortex family defaults
CHARGE = 1                       # topological charge per vortex
WAIST = 0.8e-3                   # vortex waist [m]
VORTEX_SEP_MM = 1.5              # centre-to-centre separation [mm]
VORTEX_SEP = VORTEX_SEP_MM * 1e-3
APERTURE_RADIUS = 3.5e-3         # source aperture [m]

# Propagation distance (source plane → observation plane)
Z_PROP = 5.0e-3                  # 5 mm forward propagation

# Base pressure scale: the ASM field is normalised (max~1), so we
# multiply by BASE_PRESSURE_PA to bring it into the physical regime
# before computing Gor'kov.  3 kPa is the repo-standard P_SCALE.
BASE_PRESSURE_PA = 3000.0        # Pa — matches repo P_SCALE convention

# Amplitude scaling target: Gor'kov trap depth ≈ TARGET_UDEPTH [J]
# Typical polystyrene 50 µm sphere: ~1e-19 J gives mild trapping.
TARGET_UDEPTH = 1.0e-19          # target trap-depth magnitude

# Particle
PP = default_particle_params()   # a, f1, f2

# Frequency sweep (at fixed focal length)
FREQ_SWEEP_MHZ = np.linspace(1.5, 3.0, 8)
FOCAL_FIXED_MM = 10.0            # fixed focal for freq sweep

# Focal-length sweep (at fixed frequency)
FOCAL_SWEEP_MM = np.linspace(5.0, 25.0, 8)
FREQ_FIXED_MHZ = 2.0             # fixed freq for focal sweep

# Trap detection
MIN_DISTANCE_PTS = 8             # minimum separation between traps [grid pts]
N_MINIMA_MAX = 20                # max candidates to detect

# ════════════════════════════════════════════════════════════════════
# Output directory
# ════════════════════════════════════════════════════════════════════
TS = datetime.now().strftime("%Y%m%d_%H%M%S")
STUDY_NAME = "double_vortex_frequency_focal_study"
OUT_DIR = PROJECT_ROOT / "results" / "dev" / f"{STUDY_NAME}_{TS}"
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"[study] output → {OUT_DIR}")


# ════════════════════════════════════════════════════════════════════
# Helper: build grid
# ════════════════════════════════════════════════════════════════════

def make_grid(n: int = N_GRID, half: float = DOMAIN_HALF):
    """Return x, y, XX, YY, dx, dy for a square grid centred at origin."""
    x = np.linspace(-half, half, n)
    y = np.linspace(-half, half, n)
    XX, YY = np.meshgrid(x, y)
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    return x, y, XX, YY, dx, dy


# ════════════════════════════════════════════════════════════════════
# Helper: generate focused double-vortex source field
# ════════════════════════════════════════════════════════════════════

def make_double_vortex_source(
    XX: np.ndarray,
    YY: np.ndarray,
    freq_hz: float,
    focal_m: float,
    sep_m: float = VORTEX_SEP,
    charge: int = CHARGE,
    waist: float = WAIST,
    aperture_radius: float = APERTURE_RADIUS,
) -> np.ndarray:
    """
    Superpose two off-axis focused vortex fields at the source plane.

    Each vortex:
      - LG-like amplitude with given waist and charge
      - Multiplied by a focusing lens phase (ideal family)
      - Offset ±sep/2 along x from domain centre

    Parameters
    ----------
    XX, YY : 2-D meshgrid arrays
    freq_hz : operating frequency [Hz]
    focal_m : lens focal length [m]
    sep_m : vortex centre-to-centre separation [m]
    charge : topological charge (sign flipped for second vortex → ±ℓ)
    waist : beam waist [m]
    aperture_radius : hard aperture [m]

    Returns
    -------
    field : complex 2-D array (source plane)
    """
    cx = 0.0
    cy = 0.0
    k = 2.0 * np.pi * freq_hz / C_WATER

    # Two vortex centres offset along x
    c1 = (cx - sep_m / 2, cy)
    c2 = (cx + sep_m / 2, cy)

    # Vortex 1: charge +ℓ
    v1 = make_vortex_field(
        XX, YY, charge=+charge, waist=waist, k=k,
        center=c1, aperture_radius=aperture_radius,
    )
    # Vortex 2: charge -ℓ (opposite handedness → distinct trap shapes)
    v2 = make_vortex_field(
        XX, YY, charge=-charge, waist=waist, k=k,
        center=c2, aperture_radius=aperture_radius,
    )

    # Focusing lens phase (shared, centred)
    lens_ph = make_lens_phase(
        XX, YY,
        focal_length=focal_m,
        aperture_radius=aperture_radius,
        family="ideal",
        k=k,
        charge=0,
        center=(cx, cy),
    )

    # Combine: superpose vortices, apply focusing
    field = (v1 + v2) * np.exp(1j * lens_ph)
    return field


# ════════════════════════════════════════════════════════════════════
# Helper: propagate, compute Gor'kov, detect traps, scale amplitude
# ════════════════════════════════════════════════════════════════════

def analyse_case(
    XX: np.ndarray,
    YY: np.ndarray,
    dx: float,
    dy: float,
    x: np.ndarray,
    y: np.ndarray,
    freq_hz: float,
    focal_m: float,
) -> Dict[str, Any]:
    """
    Full pipeline for one (frequency, focal) setting.

    1. Generate focused double-vortex source field
    2. ASM propagate to observation plane
    3. Scale amplitude to target Gor'kov trap depth
    4. Compute Gor'kov potential + force
    5. Detect trap minima
    6. Compute trap metrics (separation, depth, curvature)

    Returns dict with all metrics and field arrays.
    """
    lam = C_WATER / freq_hz
    omega = 2.0 * np.pi * freq_hz

    # 1. Source field
    source = make_double_vortex_source(XX, YY, freq_hz, focal_m)

    # 2. Propagate
    p_raw = propagate_asm(source, dx, dy, wavelength=lam, z=Z_PROP, pad_factor=2)

    # Apply base pressure scale (ASM field is normalised, ~max 1)
    p_raw *= BASE_PRESSURE_PA

    # 3. Preliminary Gor'kov to find natural trap depth
    U_raw, _, _ = gorkov_grid_2d(
        p_raw, dx, dy, omega, RHO0, C_WATER,
        PP["a"], PP["f1"], PP["f2"],
    )
    depth_raw = float(U_raw.min())

    # Scale so that |U_min| ≈ TARGET_UDEPTH
    if abs(depth_raw) > 0:
        alpha = np.sqrt(TARGET_UDEPTH / abs(depth_raw))
    else:
        alpha = 1.0
    # Clamp alpha to reasonable range
    alpha = float(np.clip(alpha, 0.01, 50.0))

    p_scaled = p_raw * alpha

    # 4. Gor'kov on scaled field
    U, Fx, Fy = gorkov_grid_2d(
        p_scaled, dx, dy, omega, RHO0, C_WATER,
        PP["a"], PP["f1"], PP["f2"],
    )

    # 5. Detect trap minima
    traps, trap_vals = find_traps(U, x, y)

    # 6. Compute metrics
    metrics = compute_trap_metrics(traps, trap_vals, U, x, y, dx, dy)

    peak_pressure = float(np.abs(p_scaled).max())
    F_mag = np.sqrt(Fx**2 + Fy**2)

    return {
        "freq_hz": freq_hz,
        "freq_mhz": freq_hz / 1e6,
        "focal_m": focal_m,
        "focal_mm": focal_m * 1e3,
        "alpha": alpha,
        "peak_pressure_Pa": peak_pressure,
        "max_gorkov_force_N": float(F_mag.max()),
        "U_min_J": float(U.min()),
        "depth_raw_J": depth_raw,
        # Trap metrics
        **metrics,
        # Arrays (for plotting)
        "_p_scaled": p_scaled,
        "_U": U,
        "_traps": traps,
        "_trap_vals": trap_vals,
    }


# ════════════════════════════════════════════════════════════════════
# Trap detection
# ════════════════════════════════════════════════════════════════════

def find_traps(
    U: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    n_minima: int = N_MINIMA_MAX,
    min_dist_pts: int = MIN_DISTANCE_PTS,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find local minima in Gor'kov potential using morphological filter.

    Returns
    -------
    positions : (n, 2) array of (x, y) [m]
    values    : (n,) array of U values [J]
    """
    hood = 2 * min_dist_pts + 1
    local_min = minimum_filter(U, size=hood)
    mask = (U == local_min)
    # Exclude border
    border = min_dist_pts
    mask[:border, :] = False
    mask[-border:, :] = False
    mask[:, :border] = False
    mask[:, -border:] = False

    iy, ix = np.where(mask)
    if len(iy) == 0:
        return np.empty((0, 2)), np.empty(0)

    vals = U[iy, ix]
    order = np.argsort(vals)
    iy, ix, vals = iy[order], ix[order], vals[order]

    n = min(n_minima, len(vals))
    positions = np.column_stack([x[ix[:n]], y[iy[:n]]])
    return positions, vals[:n]


# ════════════════════════════════════════════════════════════════════
# Trap metrics
# ════════════════════════════════════════════════════════════════════

def compute_trap_metrics(
    traps: np.ndarray,
    trap_vals: np.ndarray,
    U: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    dx: float,
    dy: float,
) -> Dict[str, Any]:
    """
    Compute summary metrics for detected traps.

    Returns dict with trap_count, primary positions, separation, depths,
    Hessian-based curvatures for the two deepest traps.
    """
    n_traps = len(trap_vals)
    out: Dict[str, Any] = {"trap_count": n_traps}

    if n_traps == 0:
        out.update({
            "trap1_x_mm": None, "trap1_y_mm": None,
            "trap2_x_mm": None, "trap2_y_mm": None,
            "separation_mm": None,
            "trap1_depth_J": None, "trap2_depth_J": None,
            "trap1_curvature": None, "trap2_curvature": None,
        })
        return out

    # Primary trap (deepest)
    out["trap1_x_mm"] = float(traps[0, 0] * 1e3)
    out["trap1_y_mm"] = float(traps[0, 1] * 1e3)
    out["trap1_depth_J"] = float(trap_vals[0])

    # Hessian at trap 1
    out["trap1_curvature"] = _hessian_curvature(traps[0], U, x, y, dx, dy)

    if n_traps >= 2:
        out["trap2_x_mm"] = float(traps[1, 0] * 1e3)
        out["trap2_y_mm"] = float(traps[1, 1] * 1e3)
        out["trap2_depth_J"] = float(trap_vals[1])
        out["trap2_curvature"] = _hessian_curvature(traps[1], U, x, y, dx, dy)
        sep = np.linalg.norm(traps[0] - traps[1])
        out["separation_mm"] = float(sep * 1e3)
    else:
        out["trap2_x_mm"] = None
        out["trap2_y_mm"] = None
        out["trap2_depth_J"] = None
        out["trap2_curvature"] = None
        out["separation_mm"] = None

    return out


def _hessian_curvature(
    pos: np.ndarray,
    U: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    dx: float,
    dy: float,
) -> Optional[float]:
    """
    Estimate minimum Hessian eigenvalue at a trap position.

    Positive eigenvalue → stable (concave up).
    Returns None if position is too close to boundary.
    """
    ix = int(np.argmin(np.abs(x - pos[0])))
    iy = int(np.argmin(np.abs(y - pos[1])))
    if ix < 2 or ix >= len(x) - 2 or iy < 2 or iy >= len(y) - 2:
        return None

    Uxx = (U[iy, ix + 1] - 2 * U[iy, ix] + U[iy, ix - 1]) / dx**2
    Uyy = (U[iy + 1, ix] - 2 * U[iy, ix] + U[iy - 1, ix]) / dy**2
    Uxy = (U[iy + 1, ix + 1] - U[iy + 1, ix - 1]
           - U[iy - 1, ix + 1] + U[iy - 1, ix - 1]) / (4 * dx * dy)

    H = np.array([[Uxx, Uxy], [Uxy, Uyy]])
    eigvals = np.linalg.eigvalsh(H)
    return float(eigvals.min())


# ════════════════════════════════════════════════════════════════════
# Trap continuity tracking
# ════════════════════════════════════════════════════════════════════

def track_trap_continuity(
    results: List[Dict[str, Any]],
    sweep_key: str,
) -> List[Dict[str, Any]]:
    """
    Track trap identities across sweep steps via nearest-neighbour matching.

    For each adjacent pair of sweep steps, match the two deepest traps
    by nearest position and report displacement, identity jumps, and
    bifurcations.

    Parameters
    ----------
    results : list of case dicts (ordered by sweep variable)
    sweep_key : "freq_mhz" or "focal_mm"

    Returns
    -------
    list of dicts, one per adjacent pair.
    """
    continuity = []

    for i in range(len(results) - 1):
        r0 = results[i]
        r1 = results[i + 1]
        entry = {
            f"{sweep_key}_from": r0[sweep_key],
            f"{sweep_key}_to": r1[sweep_key],
        }

        t0 = r0["_traps"]
        t1 = r1["_traps"]

        n0 = min(len(t0), 2)
        n1 = min(len(t1), 2)

        entry["traps_before"] = n0
        entry["traps_after"] = n1

        if n0 == 0 or n1 == 0:
            entry["trap1_disp_mm"] = None
            entry["trap2_disp_mm"] = None
            entry["identity_jump"] = True
            continuity.append(entry)
            continue

        # Match by nearest neighbour (greedy, up to 2 traps)
        used = set()
        disps = []
        for j in range(min(n0, 2)):
            best_dist = np.inf
            best_k = -1
            for k in range(min(n1, 2)):
                if k in used:
                    continue
                d = np.linalg.norm(t0[j] - t1[k])
                if d < best_dist:
                    best_dist = d
                    best_k = k
            if best_k >= 0:
                used.add(best_k)
                disps.append(best_dist)
            else:
                disps.append(None)

        entry["trap1_disp_mm"] = float(disps[0] * 1e3) if disps[0] is not None else None
        entry["trap2_disp_mm"] = float(disps[1] * 1e3) if len(disps) > 1 and disps[1] is not None else None

        # Identity jump heuristic: displacement > 0.5 mm or trap count changed
        max_disp = max((d for d in disps if d is not None), default=0)
        entry["identity_jump"] = bool(max_disp > 0.5e-3 or n0 != n1)

        continuity.append(entry)

    return continuity


# ════════════════════════════════════════════════════════════════════
# Contact sheet plotting
# ════════════════════════════════════════════════════════════════════

def plot_contact_sheet(
    results: List[Dict[str, Any]],
    sweep_key: str,
    sweep_label: str,
    x: np.ndarray,
    y: np.ndarray,
    filename: str,
):
    """
    Plot a contact sheet: one Gor'kov panel per sweep value, shared colour limits.

    Overlays detected trap markers.
    """
    n = len(results)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.5 * ncols, 4.0 * nrows),
        squeeze=False,
    )
    extent = [x[0] * 1e3, x[-1] * 1e3, y[0] * 1e3, y[-1] * 1e3]

    # Shared colour limits across all panels
    U_all = np.concatenate([r["_U"].ravel() for r in results])
    vmin = float(np.percentile(U_all, 0.5))
    vmax = float(np.percentile(U_all, 99.5))

    for idx, r in enumerate(results):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        im = ax.imshow(
            r["_U"], origin="lower", extent=extent,
            cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="equal",
        )
        traps = r["_traps"]
        if len(traps) > 0:
            ax.plot(
                traps[:, 0] * 1e3, traps[:, 1] * 1e3,
                "kx", ms=8, mew=2, label="traps",
            )
            # Highlight two deepest
            for j in range(min(2, len(traps))):
                ax.plot(
                    traps[j, 0] * 1e3, traps[j, 1] * 1e3,
                    "o", ms=10, mec="lime", mfc="none", mew=2,
                )

        val = r[sweep_key]
        ax.set_title(
            f"{sweep_label}={val:.2f}\n"
            f"α={r['alpha']:.2f}, traps={r['trap_count']}",
            fontsize=9,
        )
        ax.set_xlabel("x [mm]", fontsize=8)
        ax.set_ylabel("y [mm]", fontsize=8)
        ax.tick_params(labelsize=7)

    # Remove unused panels
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.colorbar(im, ax=axes.ravel().tolist(), label="Gor'kov U [J]", shrink=0.7)
    fig.suptitle(
        f"Double-Vortex Gor'kov — {sweep_label} sweep\n"
        f"charge=±{CHARGE}, sep={VORTEX_SEP_MM:.1f} mm, "
        f"waist={WAIST*1e3:.1f} mm, z_prop={Z_PROP*1e3:.0f} mm",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 0.92, 0.93])
    path = OUT_DIR / filename
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.name}")


# ════════════════════════════════════════════════════════════════════
# Summary line plots
# ════════════════════════════════════════════════════════════════════

def plot_summary_curves(
    results: List[Dict[str, Any]],
    sweep_key: str,
    sweep_label: str,
    prefix: str,
):
    """Generate separation, trap count, depth, alpha plots for one sweep."""
    vals = [r[sweep_key] for r in results]

    def _plot(ydata, ylabel, fname, marker="o-"):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(vals, ydata, marker, ms=5, lw=1.5)
        ax.set_xlabel(sweep_label)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT_DIR / fname, dpi=150)
        plt.close(fig)
        print(f"  saved {fname}")

    # Separation
    sep = [r["separation_mm"] if r["separation_mm"] is not None else np.nan
           for r in results]
    _plot(sep, "Trap separation [mm]", f"{prefix}_separation.png")

    # Trap count
    _plot([r["trap_count"] for r in results],
          "Trap count", f"{prefix}_trap_count.png")

    # Trap depths
    d1 = [r["trap1_depth_J"] if r["trap1_depth_J"] is not None else np.nan
          for r in results]
    d2 = [r["trap2_depth_J"] if r["trap2_depth_J"] is not None else np.nan
          for r in results]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(vals, d1, "o-", label="trap 1 (deepest)", ms=5)
    ax.plot(vals, d2, "s--", label="trap 2", ms=5)
    ax.set_xlabel(sweep_label)
    ax.set_ylabel("Trap depth U [J]")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{prefix}_trap_depth.png", dpi=150)
    plt.close(fig)
    print(f"  saved {prefix}_trap_depth.png")

    # Alpha
    _plot([r["alpha"] for r in results],
          "Scaling α", f"{prefix}_alpha.png")


# ════════════════════════════════════════════════════════════════════
# Optional: line cut through trap axis
# ════════════════════════════════════════════════════════════════════

def plot_line_cut(
    results: List[Dict[str, Any]],
    x: np.ndarray,
    y: np.ndarray,
    sweep_key: str,
    sweep_label: str,
    prefix: str,
):
    """
    1-D line cut along y=0 (the axis connecting the two vortex centres)
    showing how the Gor'kov potential landscape evolves across the sweep.
    """
    iy_mid = len(y) // 2  # closest row to y=0

    fig, ax = plt.subplots(figsize=(8, 4))
    cmap = plt.cm.viridis
    norm = Normalize(
        vmin=min(r[sweep_key] for r in results),
        vmax=max(r[sweep_key] for r in results),
    )

    for r in results:
        U_line = r["_U"][iy_mid, :]
        colour = cmap(norm(r[sweep_key]))
        ax.plot(x * 1e3, U_line, color=colour, lw=1.2,
                label=f"{r[sweep_key]:.2f}")

    ax.set_xlabel("x [mm]")
    ax.set_ylabel("Gor'kov U [J]")
    ax.set_title(f"Line cut along vortex axis — {sweep_label} sweep")
    ax.legend(fontsize=7, ncol=2, title=sweep_label)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{prefix}_line_cut.png", dpi=150)
    plt.close(fig)
    print(f"  saved {prefix}_line_cut.png")


# ════════════════════════════════════════════════════════════════════
# Main study
# ════════════════════════════════════════════════════════════════════

def run_study():
    t_start = time.time()

    # Build grid (origin-centred, independent of FEM cache)
    x, y, XX, YY, dx, dy = make_grid()

    config = {
        "study": STUDY_NAME,
        "timestamp": TS,
        "grid_n": N_GRID,
        "domain_half_mm": DOMAIN_HALF_MM,
        "charge": CHARGE,
        "waist_mm": WAIST * 1e3,
        "vortex_sep_mm": VORTEX_SEP_MM,
        "aperture_radius_mm": APERTURE_RADIUS * 1e3,
        "z_prop_mm": Z_PROP * 1e3,
        "base_pressure_Pa": BASE_PRESSURE_PA,
        "target_Udepth_J": TARGET_UDEPTH,
        "particle": PP,
        "freq_sweep_mhz": FREQ_SWEEP_MHZ.tolist(),
        "focal_fixed_mm": FOCAL_FIXED_MM,
        "focal_sweep_mm": FOCAL_SWEEP_MM.tolist(),
        "freq_fixed_mhz": FREQ_FIXED_MHZ,
    }
    with open(OUT_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)
    print("[study] config saved")

    # ── Frequency sweep ────────────────────────────────────────────
    print("\n═══ FREQUENCY SWEEP ═══")
    freq_results = []
    for i, f_mhz in enumerate(FREQ_SWEEP_MHZ):
        f_hz = f_mhz * 1e6
        focal_m = FOCAL_FIXED_MM * 1e-3
        print(f"  [{i+1}/{len(FREQ_SWEEP_MHZ)}] f={f_mhz:.2f} MHz, "
              f"focal={FOCAL_FIXED_MM:.0f} mm …")
        r = analyse_case(XX, YY, dx, dy, x, y, f_hz, focal_m)
        freq_results.append(r)
        print(f"    α={r['alpha']:.3f}, traps={r['trap_count']}, "
              f"sep={r['separation_mm']}")

    plot_contact_sheet(
        freq_results, "freq_mhz", "f [MHz]", x, y,
        "frequency_sweep_contact_sheet.png",
    )
    plot_summary_curves(freq_results, "freq_mhz", "f [MHz]", "freq")
    plot_line_cut(freq_results, x, y, "freq_mhz", "f [MHz]", "freq")
    freq_cont = track_trap_continuity(freq_results, "freq_mhz")

    # ── Focal-length sweep ─────────────────────────────────────────
    print("\n═══ FOCAL-LENGTH SWEEP ═══")
    focal_results = []
    for i, fl_mm in enumerate(FOCAL_SWEEP_MM):
        f_hz = FREQ_FIXED_MHZ * 1e6
        focal_m = fl_mm * 1e-3
        print(f"  [{i+1}/{len(FOCAL_SWEEP_MM)}] focal={fl_mm:.1f} mm, "
              f"f={FREQ_FIXED_MHZ:.2f} MHz …")
        r = analyse_case(XX, YY, dx, dy, x, y, f_hz, focal_m)
        focal_results.append(r)
        print(f"    α={r['alpha']:.3f}, traps={r['trap_count']}, "
              f"sep={r['separation_mm']}")

    plot_contact_sheet(
        focal_results, "focal_mm", "focal [mm]", x, y,
        "focal_sweep_contact_sheet.png",
    )
    plot_summary_curves(focal_results, "focal_mm", "focal [mm]", "focal")
    plot_line_cut(focal_results, x, y, "focal_mm", "focal [mm]", "focal")
    focal_cont = track_trap_continuity(focal_results, "focal_mm")

    # ── Save metrics CSV ───────────────────────────────────────────
    _save_metrics_csv(freq_results + focal_results)

    # ── Save summary JSON ──────────────────────────────────────────
    summary = {
        "frequency_sweep": _strip_arrays(freq_results),
        "focal_sweep": _strip_arrays(focal_results),
        "frequency_continuity": freq_cont,
        "focal_continuity": focal_cont,
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)
    print("[study] summary.json saved")

    # ── Save NPZ with fields ──────────────────────────────────────
    _save_npz(freq_results, focal_results, x, y)

    elapsed = time.time() - t_start
    print(f"\n[study] complete in {elapsed:.1f}s")
    print(f"[study] results → {OUT_DIR}")

    # ── Print quick summary ────────────────────────────────────────
    _print_summary(freq_results, focal_results, freq_cont, focal_cont)


# ════════════════════════════════════════════════════════════════════
# Save helpers
# ════════════════════════════════════════════════════════════════════

def _strip_arrays(results: List[Dict]) -> List[Dict]:
    """Remove numpy arrays from results for JSON serialisation."""
    out = []
    for r in results:
        out.append({k: v for k, v in r.items() if not k.startswith("_")})
    return out


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _save_metrics_csv(results: List[Dict]):
    """Save flat CSV with key metrics for all cases."""
    keys = [
        "freq_mhz", "focal_mm", "alpha", "peak_pressure_Pa",
        "max_gorkov_force_N", "U_min_J", "trap_count",
        "trap1_x_mm", "trap1_y_mm", "trap2_x_mm", "trap2_y_mm",
        "separation_mm", "trap1_depth_J", "trap2_depth_J",
        "trap1_curvature", "trap2_curvature",
    ]
    path = OUT_DIR / "metrics.csv"
    with open(path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in results:
            row = []
            for k in keys:
                v = r.get(k)
                if v is None:
                    row.append("")
                else:
                    row.append(f"{v}")
            f.write(",".join(row) + "\n")
    print(f"[study] metrics.csv saved ({len(results)} rows)")


def _save_npz(freq_results, focal_results, x, y):
    """Save field arrays for post-processing."""
    path = OUT_DIR / "fields.npz"
    data = {"x": x, "y": y}

    for i, r in enumerate(freq_results):
        data[f"freq_{i:02d}_p"] = r["_p_scaled"]
        data[f"freq_{i:02d}_U"] = r["_U"]
        data[f"freq_{i:02d}_traps"] = r["_traps"]

    for i, r in enumerate(focal_results):
        data[f"focal_{i:02d}_p"] = r["_p_scaled"]
        data[f"focal_{i:02d}_U"] = r["_U"]
        data[f"focal_{i:02d}_traps"] = r["_traps"]

    np.savez_compressed(str(path), **data)
    print(f"[study] fields.npz saved ({path.stat().st_size / 1e6:.1f} MB)")


# ════════════════════════════════════════════════════════════════════
# Summary printout
# ════════════════════════════════════════════════════════════════════

def _print_summary(freq_results, focal_results, freq_cont, focal_cont):
    print("\n" + "=" * 60)
    print("FEASIBILITY SUMMARY")
    print("=" * 60)

    for label, results, cont in [
        ("FREQUENCY SWEEP", freq_results, freq_cont),
        ("FOCAL-LENGTH SWEEP", focal_results, focal_cont),
    ]:
        print(f"\n── {label} ──")

        counts = [r["trap_count"] for r in results]
        seps = [r["separation_mm"] for r in results
                if r["separation_mm"] is not None]
        alphas = [r["alpha"] for r in results]

        print(f"  Trap counts:  {counts}")
        print(f"  Separations:  {['%.3f' % s for s in seps]} mm")
        print(f"  Alpha range:  {min(alphas):.3f} – {max(alphas):.3f}")

        # Two-trap persistence
        two_trap = sum(1 for c in counts if c >= 2)
        print(f"  Two-trap cases: {two_trap}/{len(counts)}")

        # Continuity
        jumps = sum(1 for c in cont if c.get("identity_jump", True))
        print(f"  Identity jumps: {jumps}/{len(cont)} transitions")

        if seps:
            print(f"  Sep range:  {min(seps):.3f} – {max(seps):.3f} mm")
            # Check monotonicity
            diffs = [seps[i+1] - seps[i] for i in range(len(seps)-1)]
            mono = all(d <= 0 for d in diffs) or all(d >= 0 for d in diffs)
            print(f"  Monotonic separation: {mono}")

        # Merge check
        merge = any(r["trap_count"] == 1 for r in results)
        print(f"  Merged to 1 trap: {merge}")


# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    run_study()
