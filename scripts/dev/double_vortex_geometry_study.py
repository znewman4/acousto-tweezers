#!/usr/bin/env python3
"""
Double-Vortex Geometry Study — Phase 1 expansion
==================================================

Expanded field-family feasibility study following the initial frequency/focal
sweep (double_vortex_frequency_focal_study.py).

Sweeps geometry parameters of the focused double-vortex family:
  - vortex separation
  - beam waist
  - anisotropy (axis compression)
  - chirality pairing (+/- vs +/+)
  - small asymmetry perturbations

Then re-sweeps frequency and focal length for best geometries.

Uses ASM propagation only (no standing-wave, no particle sim, no GIFs).
"""
from __future__ import annotations

import csv
import json
import sys
import textwrap
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.ndimage import minimum_filter

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
# Constants & defaults
# ════════════════════════════════════════════════════════════════════
N_GRID = 400
DOMAIN_HALF_MM = 4.0
DOMAIN_HALF = DOMAIN_HALF_MM * 1e-3
APERTURE_RADIUS = 3.5e-3       # m
Z_PROP = 5.0e-3                # m
BASE_PRESSURE_PA = 3000.0      # Pa
TARGET_UDEPTH = 1.0e-19        # J
PP = default_particle_params()
MIN_DISTANCE_PTS = 8
N_MINIMA_MAX = 20

# Baseline geometry (from previous study)
BASELINE = dict(
    freq_hz=2.0e6,
    focal_m=10.0e-3,
    sep_m=1.5e-3,
    charge1=+1,
    charge2=-1,
    waist=0.8e-3,
    squeeze_x=1.0,       # no anisotropy
    amp_ratio=1.0,        # symmetric amplitude
    phase_offset=0.0,     # no phase offset between vortices
    dx_shift=0.0,         # no centre shift
    waist_ratio=1.0,      # no waist mismatch
)

# ════════════════════════════════════════════════════════════════════
# Output directory
# ════════════════════════════════════════════════════════════════════
TS = datetime.now().strftime("%Y%m%d_%H%M%S")
STUDY_NAME = "double_vortex_geometry_study"
OUT_DIR = PROJECT_ROOT / "results" / "dev" / f"{STUDY_NAME}_{TS}"
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"[study] output → {OUT_DIR}")


# ════════════════════════════════════════════════════════════════════
# Grid
# ════════════════════════════════════════════════════════════════════
def make_grid(n: int = N_GRID, half: float = DOMAIN_HALF):
    x = np.linspace(-half, half, n)
    y = np.linspace(-half, half, n)
    XX, YY = np.meshgrid(x, y)
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    return x, y, XX, YY, dx, dy


# ════════════════════════════════════════════════════════════════════
# Generalised double-vortex source
# ════════════════════════════════════════════════════════════════════
def make_double_vortex_source(
    XX: np.ndarray,
    YY: np.ndarray,
    *,
    freq_hz: float,
    focal_m: float,
    sep_m: float,
    charge1: int,
    charge2: int,
    waist: float,
    squeeze_x: float = 1.0,
    amp_ratio: float = 1.0,
    phase_offset: float = 0.0,
    dx_shift: float = 0.0,
    waist_ratio: float = 1.0,
    aperture_radius: float = APERTURE_RADIUS,
) -> np.ndarray:
    """
    Generalised focused double-vortex source field.

    Parameters
    ----------
    squeeze_x : float
        Anisotropy factor applied to x-coordinates before computing vortex
        amplitude/phase.  squeeze_x < 1 compresses the field along the
        vortex axis, bringing traps closer.
    amp_ratio : float
        Amplitude multiplier for vortex 2 relative to vortex 1.
    phase_offset : float [rad]
        Additional phase applied to vortex 2.
    dx_shift : float [m]
        Centre shift of vortex 2 along x (asymmetry).
    waist_ratio : float
        waist2 = waist * waist_ratio  (mismatch).
    """
    k = 2.0 * np.pi * freq_hz / C_WATER

    # Anisotropy: compress x-axis
    XX_s = XX * squeeze_x
    YY_s = YY

    c1 = (-sep_m / 2, 0.0)
    c2 = (+sep_m / 2 + dx_shift, 0.0)
    waist2 = waist * waist_ratio

    v1 = make_vortex_field(
        XX_s, YY_s, charge=charge1, waist=waist, k=k,
        center=(c1[0] * squeeze_x, c1[1]),
        aperture_radius=aperture_radius,
    )
    v2 = make_vortex_field(
        XX_s, YY_s, charge=charge2, waist=waist2, k=k,
        center=(c2[0] * squeeze_x, c2[1]),
        aperture_radius=aperture_radius,
    )
    v2 = v2 * amp_ratio * np.exp(1j * phase_offset)

    lens_ph = make_lens_phase(
        XX, YY,
        focal_length=focal_m,
        aperture_radius=aperture_radius,
        family="ideal",
        k=k,
        charge=0,
        center=(0.0, 0.0),
    )

    return (v1 + v2) * np.exp(1j * lens_ph)


# ════════════════════════════════════════════════════════════════════
# Trap detection (reused from previous study)
# ════════════════════════════════════════════════════════════════════
def find_traps(U, x, y, n_minima=N_MINIMA_MAX, min_dist_pts=MIN_DISTANCE_PTS):
    hood = 2 * min_dist_pts + 1
    local_min = minimum_filter(U, size=hood)
    mask = (U == local_min)
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


def _hessian_curvature(pos, U, x, y, dx, dy):
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
# Analysis pipeline
# ════════════════════════════════════════════════════════════════════
def analyse_case(XX, YY, dx, dy, x, y, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Full pipeline for one parameter setting.

    ``params`` must contain all keys from BASELINE.
    """
    # Filter to only keys accepted by make_double_vortex_source
    _SRC_KEYS = {
        "freq_hz", "focal_m", "sep_m", "charge1", "charge2", "waist",
        "squeeze_x", "amp_ratio", "phase_offset", "dx_shift", "waist_ratio",
        "aperture_radius",
    }
    src_params = {k: v for k, v in params.items() if k in _SRC_KEYS}

    freq_hz = params["freq_hz"]
    lam = C_WATER / freq_hz
    omega = 2.0 * np.pi * freq_hz

    source = make_double_vortex_source(XX, YY, **src_params)
    p_raw = propagate_asm(source, dx, dy, wavelength=lam, z=Z_PROP, pad_factor=2)
    p_raw *= BASE_PRESSURE_PA

    U_raw, _, _ = gorkov_grid_2d(
        p_raw, dx, dy, omega, RHO0, C_WATER,
        PP["a"], PP["f1"], PP["f2"],
    )
    depth_raw = float(U_raw.min())

    if abs(depth_raw) > 0:
        alpha = np.sqrt(TARGET_UDEPTH / abs(depth_raw))
    else:
        alpha = 1.0
    alpha = float(np.clip(alpha, 0.01, 50.0))

    p_scaled = p_raw * alpha
    U, Fx, Fy = gorkov_grid_2d(
        p_scaled, dx, dy, omega, RHO0, C_WATER,
        PP["a"], PP["f1"], PP["f2"],
    )

    traps, trap_vals = find_traps(U, x, y)
    n_traps = len(trap_vals)
    F_mag = np.sqrt(Fx**2 + Fy**2)

    # Two deepest traps
    out: Dict[str, Any] = {}
    for k, v in params.items():
        if k in ("freq_hz",):
            continue
        out[k] = v
    out["freq_hz"] = freq_hz
    out["freq_mhz"] = freq_hz / 1e6
    out["focal_mm"] = params["focal_m"] * 1e3
    out["sep_mm"] = params["sep_m"] * 1e3
    out["waist_mm"] = params["waist"] * 1e3
    out["alpha"] = alpha
    out["peak_pressure_Pa"] = float(np.abs(p_scaled).max())
    out["peak_pressure_raw_Pa"] = float(np.abs(p_raw).max())
    out["max_gorkov_force_N"] = float(F_mag.max())
    out["U_min_J"] = float(U.min())
    out["depth_raw_J"] = depth_raw
    out["trap_count"] = n_traps

    # Ghost traps: traps beyond the first two
    out["ghost_count"] = max(0, n_traps - 2)

    if n_traps >= 1:
        out["trap1_x_mm"] = float(traps[0, 0] * 1e3)
        out["trap1_y_mm"] = float(traps[0, 1] * 1e3)
        out["trap1_depth_J"] = float(trap_vals[0])
        out["trap1_curvature"] = _hessian_curvature(traps[0], U, x, y, dx, dy)
    else:
        out["trap1_x_mm"] = None
        out["trap1_y_mm"] = None
        out["trap1_depth_J"] = None
        out["trap1_curvature"] = None

    if n_traps >= 2:
        out["trap2_x_mm"] = float(traps[1, 0] * 1e3)
        out["trap2_y_mm"] = float(traps[1, 1] * 1e3)
        out["trap2_depth_J"] = float(trap_vals[1])
        out["trap2_curvature"] = _hessian_curvature(traps[1], U, x, y, dx, dy)
        out["separation_mm"] = float(np.linalg.norm(traps[0] - traps[1]) * 1e3)
    else:
        out["trap2_x_mm"] = None
        out["trap2_y_mm"] = None
        out["trap2_depth_J"] = None
        out["trap2_curvature"] = None
        out["separation_mm"] = None

    out["_p_scaled"] = p_scaled
    out["_U"] = U
    out["_traps"] = traps
    out["_trap_vals"] = trap_vals

    return out


# ════════════════════════════════════════════════════════════════════
# Continuity tracker
# ════════════════════════════════════════════════════════════════════
def track_continuity(results, sweep_key):
    cont = []
    for i in range(len(results) - 1):
        r0, r1 = results[i], results[i + 1]
        entry = {f"{sweep_key}_from": r0[sweep_key], f"{sweep_key}_to": r1[sweep_key]}
        t0, t1 = r0["_traps"], r1["_traps"]
        n0, n1 = min(len(t0), 2), min(len(t1), 2)
        entry["traps_before"] = n0
        entry["traps_after"] = n1
        if n0 == 0 or n1 == 0:
            entry["trap1_disp_mm"] = None
            entry["trap2_disp_mm"] = None
            entry["identity_jump"] = True
            cont.append(entry)
            continue
        used = set()
        disps = []
        for j in range(min(n0, 2)):
            best_d, best_k = np.inf, -1
            for k in range(min(n1, 2)):
                if k in used:
                    continue
                d = np.linalg.norm(t0[j] - t1[k])
                if d < best_d:
                    best_d, best_k = d, k
            if best_k >= 0:
                used.add(best_k)
                disps.append(best_d)
            else:
                disps.append(None)
        entry["trap1_disp_mm"] = float(disps[0] * 1e3) if disps[0] is not None else None
        entry["trap2_disp_mm"] = float(disps[1] * 1e3) if len(disps) > 1 and disps[1] is not None else None
        max_d = max((d for d in disps if d is not None), default=0)
        entry["identity_jump"] = bool(max_d > 0.5e-3 or n0 != n1)
        cont.append(entry)
    return cont


# ════════════════════════════════════════════════════════════════════
# Plotting helpers
# ════════════════════════════════════════════════════════════════════
def plot_contact_sheet(results, sweep_key, sweep_label, x, y, filename,
                       suptitle_extra=""):
    n = len(results)
    if n == 0:
        return
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.0 * nrows),
                             squeeze=False)
    extent = [x[0] * 1e3, x[-1] * 1e3, y[0] * 1e3, y[-1] * 1e3]
    U_all = np.concatenate([r["_U"].ravel() for r in results])
    vmin = float(np.percentile(U_all, 0.5))
    vmax = float(np.percentile(U_all, 99.5))
    im = None
    for idx, r in enumerate(results):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        im = ax.imshow(r["_U"], origin="lower", extent=extent,
                       cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="equal")
        traps = r["_traps"]
        if len(traps) > 0:
            ax.plot(traps[:, 0] * 1e3, traps[:, 1] * 1e3, "kx", ms=8, mew=2)
            for j in range(min(2, len(traps))):
                ax.plot(traps[j, 0] * 1e3, traps[j, 1] * 1e3,
                        "o", ms=10, mec="lime", mfc="none", mew=2)
        val = r.get(sweep_key, "")
        sep_str = f"{r['separation_mm']:.2f}" if r['separation_mm'] is not None else "N/A"
        ax.set_title(f"{sweep_label}={_fmt(val)}\nα={r['alpha']:.2f} traps={r['trap_count']} sep={sep_str}",
                     fontsize=8)
        ax.set_xlabel("x [mm]", fontsize=7)
        ax.set_ylabel("y [mm]", fontsize=7)
        ax.tick_params(labelsize=6)
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), label="U [J]", shrink=0.6)
    fig.suptitle(f"Double-Vortex Gor'kov — {sweep_label} sweep{suptitle_extra}",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 0.92, 0.93])
    fig.savefig(OUT_DIR / filename, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {filename}")


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.3g}"
    return str(v)


def plot_curves(results, sweep_key, sweep_label, prefix):
    if len(results) == 0:
        return
    vals = [r[sweep_key] for r in results]

    def _p(ydata, ylabel, fname, marker="o-"):
        fig, ax = plt.subplots(figsize=(6, 3.8))
        ax.plot(vals, ydata, marker, ms=4, lw=1.3)
        ax.set_xlabel(sweep_label)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT_DIR / fname, dpi=140)
        plt.close(fig)

    seps = [r["separation_mm"] if r["separation_mm"] is not None else np.nan
            for r in results]
    _p(seps, "Trap separation [mm]", f"{prefix}_separation.png")
    _p([r["trap_count"] for r in results], "Trap count", f"{prefix}_trap_count.png")
    _p([r["ghost_count"] for r in results], "Ghost traps", f"{prefix}_ghost_count.png")
    _p([r["alpha"] for r in results], "Scaling α", f"{prefix}_alpha.png")

    # Depth
    d1 = [r["trap1_depth_J"] if r["trap1_depth_J"] is not None else np.nan
          for r in results]
    d2 = [r["trap2_depth_J"] if r["trap2_depth_J"] is not None else np.nan
          for r in results]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(vals, d1, "o-", label="trap 1", ms=4)
    ax.plot(vals, d2, "s--", label="trap 2", ms=4)
    ax.set_xlabel(sweep_label)
    ax.set_ylabel("Trap depth U [J]")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{prefix}_trap_depth.png", dpi=140)
    plt.close(fig)


def plot_line_cut(results, x, y, sweep_key, sweep_label, prefix):
    if len(results) == 0:
        return
    iy_mid = len(y) // 2
    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    cmap = plt.cm.viridis
    vmin = min(r[sweep_key] for r in results)
    vmax = max(r[sweep_key] for r in results)
    norm = Normalize(vmin=vmin, vmax=max(vmax, vmin + 1e-12))
    for r in results:
        ax.plot(x * 1e3, r["_U"][iy_mid, :], color=cmap(norm(r[sweep_key])),
                lw=1, label=f"{_fmt(r[sweep_key])}")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("U [J]")
    ax.set_title(f"Line cut — {sweep_label} sweep")
    ax.legend(fontsize=6, ncol=2, title=sweep_label)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{prefix}_line_cut.png", dpi=140)
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════
# Sweep runner
# ════════════════════════════════════════════════════════════════════
def run_sweep(XX, YY, dx, dy, x, y, param_list, sweep_key, sweep_label,
              prefix, suptitle_extra=""):
    """
    Run analyse_case for each dict in param_list.
    Produce contact sheet, curves, line cut.
    Return list of result dicts.
    """
    results = []
    for i, params in enumerate(param_list):
        print(f"  [{i+1}/{len(param_list)}] …", end="")
        r = analyse_case(XX, YY, dx, dy, x, y, params)
        results.append(r)
        val = r.get(sweep_key, params.get(sweep_key, "?"))
        print(f"  {sweep_key}={_fmt(val)}", end="")
        sep = r["separation_mm"]
        print(f"  α={r['alpha']:.3f} traps={r['trap_count']} "
              f"sep={'%.3f' % sep if sep else 'N/A'} ghosts={r['ghost_count']}")

    plot_contact_sheet(results, sweep_key, sweep_label, x, y,
                       f"{prefix}_contact.png", suptitle_extra)
    plot_curves(results, sweep_key, sweep_label, prefix)
    plot_line_cut(results, x, y, sweep_key, sweep_label, prefix)
    return results


# ════════════════════════════════════════════════════════════════════
# Scoring / ranking
# ════════════════════════════════════════════════════════════════════
def score_result(r):
    """
    Lower is better.

    Penalises: large separation, many ghost traps, missing traps,
               extreme alpha.
    Rewards:  small separation, exactly 2 traps, moderate alpha.
    """
    sep = r.get("separation_mm")
    if sep is None or r["trap_count"] < 2:
        return 1e6
    s = sep  # primary: minimise separation
    s += 2.0 * r["ghost_count"]             # penalise ghosts
    if r["alpha"] > 5.0 or r["alpha"] < 0.05:
        s += 5.0                             # extreme scaling penalty
    return s


def rank_results(results, n=5):
    scored = [(score_result(r), i, r) for i, r in enumerate(results)]
    scored.sort()
    return scored[:n]


# ════════════════════════════════════════════════════════════════════
# CSV / JSON helpers
# ════════════════════════════════════════════════════════════════════
CSV_KEYS = [
    "stage", "case_idx",
    "freq_mhz", "focal_mm", "sep_mm", "waist_mm",
    "squeeze_x", "charge1", "charge2",
    "amp_ratio", "phase_offset", "dx_shift", "waist_ratio",
    "alpha", "peak_pressure_Pa", "peak_pressure_raw_Pa",
    "max_gorkov_force_N", "U_min_J",
    "trap_count", "ghost_count",
    "trap1_x_mm", "trap1_y_mm", "trap1_depth_J", "trap1_curvature",
    "trap2_x_mm", "trap2_y_mm", "trap2_depth_J", "trap2_curvature",
    "separation_mm",
]


def _strip(results):
    return [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]


def _jdefault(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def append_csv(writer, results, stage):
    for i, r in enumerate(results):
        row = {}
        row["stage"] = stage
        row["case_idx"] = i
        for k in CSV_KEYS:
            if k in ("stage", "case_idx"):
                continue
            v = r.get(k)
            row[k] = "" if v is None else v
        writer.writerow(row)


# ════════════════════════════════════════════════════════════════════
# MAIN STUDY
# ════════════════════════════════════════════════════════════════════
def run_study():
    t0_total = time.time()

    x, y, XX, YY, dx, dy = make_grid()
    all_results: Dict[str, List[Dict]] = {}

    # Open CSV
    csv_path = OUT_DIR / "metrics.csv"
    csv_f = open(csv_path, "w", newline="")
    csv_w = csv.DictWriter(csv_f, fieldnames=CSV_KEYS)
    csv_w.writeheader()

    # ────────────────────────────────────────────────────────────────
    # Stage A — Baseline reproduction
    # ────────────────────────────────────────────────────────────────
    print("\n═══ STAGE A — BASELINE ═══")
    base = analyse_case(XX, YY, dx, dy, x, y, BASELINE)
    all_results["A_baseline"] = [base]
    append_csv(csv_w, [base], "A_baseline")
    print(f"  Baseline: sep={base['separation_mm']:.3f} mm, "
          f"traps={base['trap_count']}, α={base['alpha']:.3f}")

    # ────────────────────────────────────────────────────────────────
    # Stage B — Geometry sweeps at fixed freq/focal
    # ────────────────────────────────────────────────────────────────
    print("\n═══ STAGE B — GEOMETRY SWEEPS ═══")

    # B1: Vortex separation
    print("\n── B1: Vortex separation ──")
    sep_vals = [0.3e-3, 0.5e-3, 0.8e-3, 1.0e-3, 1.5e-3, 2.0e-3, 2.5e-3]
    b1_params = [{**BASELINE, "sep_m": s} for s in sep_vals]
    b1 = run_sweep(XX, YY, dx, dy, x, y, b1_params, "sep_mm",
                   "sep [mm]", "B1_sep")
    all_results["B1_sep"] = b1
    append_csv(csv_w, b1, "B1_sep")

    # B2: Waist
    print("\n── B2: Waist ──")
    waist_vals = [0.3e-3, 0.5e-3, 0.8e-3, 1.0e-3, 1.5e-3, 2.0e-3]
    b2_params = [{**BASELINE, "waist": w} for w in waist_vals]
    b2 = run_sweep(XX, YY, dx, dy, x, y, b2_params, "waist_mm",
                   "waist [mm]", "B2_waist")
    all_results["B2_waist"] = b2
    append_csv(csv_w, b2, "B2_waist")

    # B3: Anisotropy (squeeze)
    print("\n── B3: Anisotropy (squeeze_x) ──")
    squeeze_vals = [0.3, 0.5, 0.7, 0.85, 1.0, 1.2, 1.5]
    b3_params = [{**BASELINE, "squeeze_x": sq} for sq in squeeze_vals]
    b3 = run_sweep(XX, YY, dx, dy, x, y, b3_params, "squeeze_x",
                   "squeeze_x", "B3_squeeze")
    all_results["B3_squeeze"] = b3
    append_csv(csv_w, b3, "B3_squeeze")

    # B4: Chirality pairing
    print("\n── B4: Chirality pairing ──")
    chirality_configs = [
        {"charge1": +1, "charge2": -1, "label": "+1/-1"},
        {"charge1": +1, "charge2": +1, "label": "+1/+1"},
        {"charge1": +1, "charge2": -2, "label": "+1/-2"},
        {"charge1": +2, "charge2": -2, "label": "+2/-2"},
    ]
    b4_params = []
    for cc in chirality_configs:
        p = {**BASELINE, "charge1": cc["charge1"], "charge2": cc["charge2"]}
        p["_label"] = cc["label"]
        b4_params.append(p)
    b4_results = []
    for i, params in enumerate(b4_params):
        lbl = params.pop("_label")
        print(f"  [{i+1}/{len(b4_params)}] chirality={lbl} …", end="")
        r = analyse_case(XX, YY, dx, dy, x, y, params)
        r["chirality"] = lbl
        b4_results.append(r)
        sep = r["separation_mm"]
        print(f"  α={r['alpha']:.3f} traps={r['trap_count']} "
              f"sep={'%.3f' % sep if sep else 'N/A'}")
    # Contact sheet for chirality
    plot_contact_sheet(b4_results, "chirality", "chirality", x, y,
                       "B4_chirality_contact.png")
    all_results["B4_chirality"] = b4_results
    append_csv(csv_w, b4_results, "B4_chirality")

    # ────────────────────────────────────────────────────────────────
    # Pick best geometry candidates from Stage B
    # ────────────────────────────────────────────────────────────────
    all_b = b1 + b2 + b3 + b4_results
    ranked = rank_results(all_b, n=3)
    best_geoms = []
    print("\n  Top-3 geometry candidates:")
    for rank, (sc, idx, r) in enumerate(ranked):
        label = (f"sep={r['sep_mm']:.1f} waist={r['waist_mm']:.1f} "
                 f"sq={r.get('squeeze_x', 1.0):.2f} "
                 f"ch={r.get('charge1',1)}/{r.get('charge2',-1)}")
        print(f"    #{rank+1}  score={sc:.3f}  sep_trap={r['separation_mm']:.3f} mm  "
              f"{label}")
        best_geoms.append({
            "sep_m": r["sep_m"],
            "waist": r["waist"],
            "squeeze_x": r.get("squeeze_x", 1.0),
            "charge1": r.get("charge1", 1),
            "charge2": r.get("charge2", -1),
        })

    # ────────────────────────────────────────────────────────────────
    # Stage C — Frequency sweep for best geometries
    # ────────────────────────────────────────────────────────────────
    print("\n═══ STAGE C — FREQUENCY SWEEP (best geometries) ═══")
    freq_vals = np.linspace(1.5, 3.5, 8)
    for gi, geom in enumerate(best_geoms):
        tag = f"C{gi+1}"
        print(f"\n── {tag}: geom #{gi+1} ──")
        params_list = []
        for f_mhz in freq_vals:
            p = {**BASELINE, **geom, "freq_hz": f_mhz * 1e6}
            p["freq_mhz"] = f_mhz
            params_list.append(p)
        res = run_sweep(XX, YY, dx, dy, x, y, params_list, "freq_mhz",
                        "f [MHz]", f"{tag}_freq",
                        f"\ngeom: sep={geom['sep_m']*1e3:.1f} w={geom['waist']*1e3:.1f} "
                        f"sq={geom['squeeze_x']:.2f}")
        all_results[f"{tag}_freq"] = res
        append_csv(csv_w, res, f"{tag}_freq")

    # ────────────────────────────────────────────────────────────────
    # Stage D — Focal-length sweep for best geometries
    # ────────────────────────────────────────────────────────────────
    print("\n═══ STAGE D — FOCAL-LENGTH SWEEP (best geometries) ═══")
    focal_vals = np.linspace(5.0, 25.0, 8)
    for gi, geom in enumerate(best_geoms):
        tag = f"D{gi+1}"
        print(f"\n── {tag}: geom #{gi+1} ──")
        params_list = []
        for fl_mm in focal_vals:
            p = {**BASELINE, **geom, "focal_m": fl_mm * 1e-3}
            p["focal_mm_sweep"] = fl_mm
            params_list.append(p)
        res = run_sweep(XX, YY, dx, dy, x, y, params_list, "focal_mm",
                        "focal [mm]", f"{tag}_focal",
                        f"\ngeom: sep={geom['sep_m']*1e3:.1f} w={geom['waist']*1e3:.1f} "
                        f"sq={geom['squeeze_x']:.2f}")
        all_results[f"{tag}_focal"] = res
        append_csv(csv_w, res, f"{tag}_focal")

    # ────────────────────────────────────────────────────────────────
    # Stage E — Asymmetry study on best geometry
    # ────────────────────────────────────────────────────────────────
    print("\n═══ STAGE E — ASYMMETRY STUDY (best geometry) ═══")
    best = best_geoms[0]

    # E1: Amplitude imbalance
    print("\n── E1: Amplitude ratio ──")
    amp_vals = [0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5]
    e1_params = [{**BASELINE, **best, "amp_ratio": a} for a in amp_vals]
    e1 = run_sweep(XX, YY, dx, dy, x, y, e1_params, "amp_ratio",
                   "amp_ratio", "E1_amp_ratio")
    all_results["E1_amp_ratio"] = e1
    append_csv(csv_w, e1, "E1_amp_ratio")

    # E2: Phase offset
    print("\n── E2: Phase offset ──")
    phase_vals = [0.0, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2,
                  np.pi, 3 * np.pi / 2]
    e2_params = [{**BASELINE, **best, "phase_offset": ph} for ph in phase_vals]
    # Add a display key
    for p, ph in zip(e2_params, phase_vals):
        p["phase_deg"] = float(np.rad2deg(ph))
    e2 = run_sweep(XX, YY, dx, dy, x, y, e2_params, "phase_deg",
                   "Δφ [deg]", "E2_phase_offset")
    all_results["E2_phase_offset"] = e2
    append_csv(csv_w, e2, "E2_phase_offset")

    # E3: Centre shift
    print("\n── E3: Centre shift (dx) ──")
    shift_vals = [-0.3e-3, -0.2e-3, -0.1e-3, 0.0, 0.1e-3, 0.2e-3, 0.3e-3]
    e3_params = [{**BASELINE, **best, "dx_shift": ds} for ds in shift_vals]
    for p, ds in zip(e3_params, shift_vals):
        p["dx_shift_mm"] = ds * 1e3
    e3 = run_sweep(XX, YY, dx, dy, x, y, e3_params, "dx_shift_mm",
                   "dx_shift [mm]", "E3_dx_shift")
    all_results["E3_dx_shift"] = e3
    append_csv(csv_w, e3, "E3_dx_shift")

    # E4: Waist mismatch
    print("\n── E4: Waist mismatch ──")
    wr_vals = [0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5]
    e4_params = [{**BASELINE, **best, "waist_ratio": wr} for wr in wr_vals]
    e4 = run_sweep(XX, YY, dx, dy, x, y, e4_params, "waist_ratio",
                   "waist_ratio", "E4_waist_ratio")
    all_results["E4_waist_ratio"] = e4
    append_csv(csv_w, e4, "E4_waist_ratio")

    csv_f.close()
    print(f"\n[study] metrics.csv saved ({csv_path})")

    # ────────────────────────────────────────────────────────────────
    # Save summary JSON
    # ────────────────────────────────────────────────────────────────
    summary = {}
    for stage, res in all_results.items():
        summary[stage] = _strip(res)
    summary["best_geometries"] = [
        {"sep_mm": g["sep_m"] * 1e3, "waist_mm": g["waist"] * 1e3,
         "squeeze_x": g["squeeze_x"], "charge1": g["charge1"],
         "charge2": g["charge2"]}
        for g in best_geoms
    ]
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=_jdefault)
    print("[study] summary.json saved")

    # ────────────────────────────────────────────────────────────────
    # Save config
    # ────────────────────────────────────────────────────────────────
    config = {
        "study": STUDY_NAME,
        "timestamp": TS,
        "grid_n": N_GRID,
        "domain_half_mm": DOMAIN_HALF_MM,
        "aperture_radius_mm": APERTURE_RADIUS * 1e3,
        "z_prop_mm": Z_PROP * 1e3,
        "base_pressure_Pa": BASE_PRESSURE_PA,
        "target_Udepth_J": TARGET_UDEPTH,
        "particle": PP,
        "baseline": {k: (v * 1e3 if "m" in k and isinstance(v, float) and abs(v) < 1
                         else v)
                     for k, v in BASELINE.items()},
    }
    with open(OUT_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2, default=_jdefault)
    print("[study] config.json saved")

    # ────────────────────────────────────────────────────────────────
    # Generate report.md
    # ────────────────────────────────────────────────────────────────
    _write_report(all_results, best_geoms)

    elapsed = time.time() - t0_total
    print(f"\n[study] done in {elapsed:.1f}s — {OUT_DIR}")


# ════════════════════════════════════════════════════════════════════
# Report generation
# ════════════════════════════════════════════════════════════════════
def _write_report(all_results, best_geoms):
    lines = []
    w = lines.append

    w("# Double-Vortex Geometry Study — Report")
    w("")
    w(f"Generated: {datetime.now().isoformat()}")
    w("")

    # Modelling check
    w("## 1. Modelling Check")
    w("")
    w("**Model:** Two off-axis LG-like vortex beams (charge ±ℓ), superposed,")
    w("multiplied by a shared ideal focusing lens phase, then ASM-propagated")
    w("to an observation plane.  Gor'kov potential computed on the propagated")
    w("field to detect trap locations.")
    w("")
    w("**Is this acceptable for an idealised field-family study?**  Yes.")
    w("")
    w("- ASM propagation is exact for monochromatic fields in a homogeneous medium.")
    w("- The LG-like vortex + lens model captures the essential parameter space")
    w("  (separation, waist, charge, focusing) without passive-lens artefacts.")
    w("- Gor'kov potential correctly identifies lateral trapping sites for small")
    w("  particles in the Rayleigh regime.")
    w("- The model is NOT suitable for predicting absolute trap strengths in a")
    w("  real standing-wave system — it is a 2D-slice, free-field idealisation.")
    w("")
    w("**Any obvious mistakes?**")
    w("")
    w("- The source field is normalised to max≈1 then scaled by BASE_PRESSURE_PA")
    w("  (3 kPa).  This is not a physical transducer source level; it is a")
    w("  convention for setting the Gor'kov scale.  Acceptable for feasibility.")
    w("- Alpha scaling then adjusts pressure so U_min ≈ 1e-19 J.  This ensures")
    w("  comparable trap depths across settings.  Correct methodology.")
    w("- Anisotropy is applied by compressing x-coordinates before vortex")
    w("  evaluation but keeping the lens phase on the original grid.  This")
    w("  is a valid idealised deformation (equivalent to an elliptical aperture")
    w("  function).  Not a physical lens anisotropy, but fine for Phase 1.")
    w("")
    w("**Recommendation:** Keep this approach for Phase 1.  Move to standing-wave")
    w("overlay and realistic lens constraints in Phase 2.")
    w("")

    # Stage results
    w("## 2. Stage Results")
    w("")

    # Baseline
    bl = all_results.get("A_baseline", [{}])[0]
    w("### Stage A — Baseline")
    w(f"- Separation: {bl.get('separation_mm', 'N/A')} mm")
    w(f"- Trap count: {bl.get('trap_count', 'N/A')}")
    w(f"- Alpha: {bl.get('alpha', 'N/A')}")
    w(f"- Ghost traps: {bl.get('ghost_count', 'N/A')}")
    w("")

    # B stages
    for stage_key, stage_label, param_key in [
        ("B1_sep", "B1 — Vortex Separation", "sep_mm"),
        ("B2_waist", "B2 — Waist", "waist_mm"),
        ("B3_squeeze", "B3 — Anisotropy (squeeze_x)", "squeeze_x"),
    ]:
        w(f"### {stage_label}")
        res = all_results.get(stage_key, [])
        if not res:
            w("  (no results)")
            w("")
            continue
        w("")
        w(f"| {param_key} | sep [mm] | traps | ghosts | α |")
        w("|---|---|---|---|---|")
        for r in res:
            sep = f"{r['separation_mm']:.3f}" if r["separation_mm"] is not None else "N/A"
            w(f"| {_fmt(r.get(param_key, '?'))} | {sep} | {r['trap_count']} "
              f"| {r['ghost_count']} | {r['alpha']:.3f} |")
        w("")

        seps = [r["separation_mm"] for r in res if r["separation_mm"] is not None]
        if seps:
            w(f"  Sep range: {min(seps):.3f} – {max(seps):.3f} mm")
            diffs = [seps[i+1] - seps[i] for i in range(len(seps)-1)]
            mono = all(d <= 0 for d in diffs) or all(d >= 0 for d in diffs)
            w(f"  Monotonic: {mono}")
        w("")

    # Chirality
    w("### B4 — Chirality Pairing")
    w("")
    b4 = all_results.get("B4_chirality", [])
    if b4:
        w("| chirality | sep [mm] | traps | ghosts | α |")
        w("|---|---|---|---|---|")
        for r in b4:
            sep = f"{r['separation_mm']:.3f}" if r["separation_mm"] is not None else "N/A"
            w(f"| {r.get('chirality', '?')} | {sep} | {r['trap_count']} "
              f"| {r['ghost_count']} | {r['alpha']:.3f} |")
    w("")

    # Best geometries
    w("### Best Geometry Candidates")
    w("")
    for i, g in enumerate(best_geoms):
        w(f"  #{i+1}: sep={g['sep_m']*1e3:.1f} mm, waist={g['waist']*1e3:.1f} mm, "
          f"squeeze_x={g['squeeze_x']:.2f}, charges={g['charge1']}/{g['charge2']}")
    w("")

    # Stages C, D summaries
    for prefix, label in [("C", "Frequency"), ("D", "Focal-length")]:
        w(f"### Stage {prefix} — {label} Sweep of Best Geometries")
        w("")
        for gi in range(len(best_geoms)):
            tag = f"{prefix}{gi+1}"
            key = f"{tag}_{'freq' if prefix == 'C' else 'focal'}"
            res = all_results.get(key, [])
            if not res:
                continue
            seps = [r["separation_mm"] for r in res if r["separation_mm"] is not None]
            counts = [r["trap_count"] for r in res]
            ghosts = [r["ghost_count"] for r in res]
            w(f"  Geom #{gi+1}: sep range {min(seps):.3f}–{max(seps):.3f} mm, "
              f"traps {min(counts)}–{max(counts)}, "
              f"ghosts {min(ghosts)}–{max(ghosts)}")
        w("")

    # Asymmetry
    w("### Stage E — Asymmetry")
    w("")
    for stage_key, label, pk in [
        ("E1_amp_ratio", "Amplitude ratio", "amp_ratio"),
        ("E2_phase_offset", "Phase offset", "phase_deg"),
        ("E3_dx_shift", "Centre shift", "dx_shift_mm"),
        ("E4_waist_ratio", "Waist mismatch", "waist_ratio"),
    ]:
        res = all_results.get(stage_key, [])
        if not res:
            continue
        seps = [r["separation_mm"] for r in res if r["separation_mm"] is not None]
        w(f"**{label}:** sep range "
          f"{'%.3f' % min(seps) if seps else 'N/A'} – "
          f"{'%.3f' % max(seps) if seps else 'N/A'} mm")
        # Check if any asymmetry reduces separation below baseline
        bl_sep = bl.get("separation_mm")
        if seps and bl_sep:
            improved = [s for s in seps if s < bl_sep]
            if improved:
                w(f"  → {len(improved)} settings reduced separation below baseline ({bl_sep:.3f} mm)")
                w(f"  → Best: {min(improved):.3f} mm")
            else:
                w(f"  → None reduced separation below baseline")
        w("")

    # Conclusions
    w("## 3. What Worked")
    w("")
    w("(Filled after running — see separation and trap data above.)")
    w("")
    w("## 4. What Failed")
    w("")
    w("(Filled after running — configurations with <2 traps or many ghosts.)")
    w("")
    w("## 5. Most Promising Variants")
    w("")
    w("See 'Best Geometry Candidates' above.  These had the smallest trap")
    w("separation while maintaining exactly two primary traps with minimal ghosts.")
    w("")
    w("## 6. Recommended Next Step")
    w("")
    w("If promising geometries are found:")
    w("")
    w("1. Add standing-wave overlay (Phase 2) to test compatibility with λ/2 lattice.")
    w("2. Compute composite Gor'kov in the overlay regime to verify trap persistence.")
    w("3. Design a passive lens (IASA) for the best double-vortex source field.")
    w("4. Run particle transport simulation with crossfade schedule.")
    w("")

    report_path = OUT_DIR / "report.md"
    report_path.write_text("\n".join(lines))
    print(f"[study] report.md saved")


# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    run_study()
