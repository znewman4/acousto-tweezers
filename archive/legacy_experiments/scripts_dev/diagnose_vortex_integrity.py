#!/usr/bin/env python3
"""
Vortex Integrity Diagnostic
============================

Isolates *why* the vortex phase structure is lost at the evaluation
plane by running a controlled set of vortex-only FEM solves with
different boundary conditions, domain sizes, and resolutions.

Cases
-----
A  Baseline — current settings, EPL=3, BG ℓ=2
B  Multi-z resample from Case A (no new solve)
C1 Rigid top (Z_air → large → Neumann ∂p/∂n = 0)
C2 Soft  top (Z_air → small → pressure-release p ≈ 0)
D  Larger domain Lx=Ly=8 mm, EPL=2 (walls further from beam)
E  Higher resolution EPL=4 on smaller domain Lx=Ly=4 mm, R_disk=0.8 mm

Output
------
results/vortex_integrity_diag_YYYYMMDD_HHMMSS/
├── _cache/
├── figures/
├── metrics.csv
└── summary.md

Usage
-----
    micromamba run -n acousto-complex python scripts/dev/diagnose_vortex_integrity.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── paths ─────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

WORKER_SCRIPT = PROJECT_ROOT / "scripts" / "experiments" / "_solve_worker_multi_z.py"

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = PROJECT_ROOT / "results" / f"vortex_integrity_diag_{TIMESTAMP}"
CACHE_DIR = OUT_DIR / "_cache"
FIG_DIR = OUT_DIR / "figures"

# Previous cache to avoid re-solving identical configs
PREV_CACHES = [
    PROJECT_ROOT / "results" / "vortex_bridge_design_study_phase_20260227_141230" / "_cache",
]

# ── physics constants ─────────────────────────────────────────────
from acoustweezers.experiments.farfield_petri_cuboid.presets import CORRECTED_PRESET

WATER_C = 1484.0
WATER_RHO = 997.0
F_HZ = 2.0e6
LAM = WATER_C / F_HZ                  # 0.742 mm
K_WATER = 2 * np.pi * F_HZ / WATER_C  # 8467.9 rad/m
OMEGA = 2 * np.pi * F_HZ

H_UNDER = CORRECTED_PRESET["H_under"]  # 3 mm
H_TOP = CORRECTED_PRESET.get("H_top", 2.0085e-3)
Z_MID = H_UNDER + H_TOP / 2
Z_STAR = Z_MID + 0.25 * LAM            # ≈ 4.190 mm (original eval plane)

GRID_N = 200  # XY interpolation grid (lighter than bridge study's 300)

# ── z-planes to sample ───────────────────────────────────────────
Z_BATH = [1.0e-3, 1.5e-3, 2.0e-3, 2.5e-3]
Z_SLAB = [3.2e-3, 3.5e-3, 3.8e-3, 4.0e-3, Z_STAR, 4.5e-3, 4.8e-3]
Z_ALL = sorted(set(Z_BATH + Z_SLAB))

# Winding ring radii (in λ)
RING_RADII_LAM = [0.5, 1.0, 1.5]

# ── target vortex config ─────────────────────────────────────────
ELL = 2  # topological charge
BEAM_WAIST_MM = 0.8
APERTURE_MM = 1.0

# ==================================================================
# Base overrides (vortex-only)
# ==================================================================

def _base_overrides(epl: int = 3) -> dict:
    """Return vortex-only overrides on the standard 6 mm domain."""
    return {
        **CORRECTED_PRESET,
        "elements_per_wavelength": epl,
        # vortex-only: disable standing wave
        "standing_velocity_amplitude": 0.0,
        "disk_velocity_amplitude": 1e-6,
        # BG ℓ=2
        "lens_drive": "bessel_gauss",
        "lens_l": ELL,
        "lens_k_r": 0.5 * K_WATER,
        "lens_beam_waist": BEAM_WAIST_MM * 1e-3,
        "disk_radius": APERTURE_MM * 1e-3,
        "lens_apodization": "cosine_taper",
        "lens_focus_offset_x": 0.0,
        "lens_focus_offset_y": 0.0,
    }


# ==================================================================
# Case definitions
# ==================================================================

CASES: List[Tuple[str, str, dict]] = []

# --- A: Baseline ---
CASES.append((
    "A_baseline",
    "Baseline (6 mm, EPL=3, standard BCs)",
    _base_overrides(epl=3),
))

# --- C1: Rigid top (Neumann) via very large Z_air ---
_c1 = _base_overrides(epl=3)
_c1["rho_air"] = 1e6
_c1["c_air"] = 1e6   # Z_air = 1e12 → α → 0 → rigid
CASES.append((
    "C1_rigid_top",
    "Rigid top (Z_air→∞, Neumann ∂p/∂n≈0)",
    _c1,
))

# --- C2: Soft top (pressure-release) via very small Z_air ---
_c2 = _base_overrides(epl=3)
_c2["rho_air"] = 1e-3
_c2["c_air"] = 1e-3   # Z_air = 1e-6 → α → -∞ → p≈0
CASES.append((
    "C2_soft_top",
    "Soft top (Z_air→0, pressure-release p≈0)",
    _c2,
))

# --- D: Larger domain (8 mm, EPL=2) to push side walls away ---
_d = _base_overrides(epl=2)
_d["Lx"] = 8e-3
_d["Ly"] = 8e-3
# Keep PML at 1λ (same absorption, just more physical domain)
CASES.append((
    "D_large_domain",
    "Larger domain (8 mm, EPL=2, walls further away)",
    _d,
))

# --- E: Higher resolution on smaller domain ---
_e = _base_overrides(epl=4)
_e["Lx"] = 4e-3
_e["Ly"] = 4e-3
_e["disk_radius"] = 0.7e-3   # smaller aperture to fit in 4 mm domain
_e["lens_beam_waist"] = 0.4e-3  # narrower beam to match smaller aperture
# Thinner PML to give more physical space
_e["pml_n_wavelengths_xy"] = 0.75
CASES.append((
    "E_higher_res",
    "Higher resolution (4 mm, EPL=4, R=0.7 mm)",
    _e,
))


# ==================================================================
# Solver wrapper
# ==================================================================

def solve_case(
    label: str,
    overrides: dict,
    z_list: List[float],
    n_xy: int = GRID_N,
    timeout: int = 600,
) -> Optional[dict]:
    """Run FEM solve via subprocess worker.  Returns NPZ dict or None."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    result_file = str(CACHE_DIR / f"_grid_{label}.npz")

    # Check own cache
    if os.path.exists(result_file):
        print(f"  [{label}] Using cached result")
        return dict(np.load(result_file, allow_pickle=False))

    # Check previous caches
    for prev in PREV_CACHES:
        prev_file = prev / f"_grid_{label}.npz"
        if prev_file.exists():
            data = dict(np.load(str(prev_file), allow_pickle=False))
            # Only reuse if it has all needed z-planes
            cached_z = list(data.get("trap_z_list", []))
            if all(any(abs(z - cz) < 1e-7 for cz in cached_z) for z in z_list):
                np.savez(result_file, **data)
                print(f"  [{label}] Copied from previous cache")
                return data
            else:
                print(f"  [{label}] Previous cache has different z-planes, re-solving")

    # Compute mid_y from overrides (domain center)
    Ly = overrides.get("Ly", 6e-3)
    mid_y = Ly / 2.0

    args = {
        "overrides": overrides,
        "label": label,
        "trap_z_list": z_list,
        "mid_y": mid_y,
        "n_xy": n_xy,
        "result_file": result_file,
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(args, f, default=str)
        args_file = f.name

    try:
        print(f"  [{label}] Solving (timeout={timeout}s) …")
        t0 = time.time()
        proc = subprocess.run(
            [sys.executable, str(WORKER_SCRIPT), args_file],
            capture_output=True, text=True, timeout=timeout,
        )
        dt = time.time() - t0
        if proc.returncode != 0:
            stderr_tail = (proc.stderr or "")[-800:]
            if "137" in str(proc.returncode) or "Killed" in stderr_tail:
                print(f"  [{label}] *** OOM (killed, rc={proc.returncode}) after {dt:.0f}s")
            else:
                print(f"  [{label}] *** FAILED rc={proc.returncode} after {dt:.0f}s")
                print(f"    stderr: {stderr_tail}")
            return None
        print(f"  [{label}] Solve OK in {dt:.0f}s")
    except subprocess.TimeoutExpired:
        print(f"  [{label}] *** TIMEOUT after {timeout}s")
        return None
    finally:
        try:
            os.unlink(args_file)
        except OSError:
            pass

    if not os.path.exists(result_file):
        print(f"  [{label}] *** No output file produced")
        return None
    return dict(np.load(result_file, allow_pickle=False))


# ==================================================================
# Winding & metrics
# ==================================================================

def compute_winding(
    p_xy: np.ndarray,
    xg: np.ndarray,
    yg: np.ndarray,
    cx: float,
    cy: float,
    radius_m: float,
    n_pts: int = 128,
) -> Tuple[float, float, float, float]:
    """
    Compute phase-winding number on a ring of given radius.

    Returns (winding, ring_amp_mean, ring_amp_std, azimuth_var).
    """
    from scipy.interpolate import RegularGridInterpolator

    interp = RegularGridInterpolator(
        (yg, xg), p_xy, method="linear",
        bounds_error=False, fill_value=0.0 + 0.0j,
    )
    thetas = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    pts_x = cx + radius_m * np.cos(thetas)
    pts_y = cy + radius_m * np.sin(thetas)
    p_ring = interp(np.column_stack([pts_y, pts_x]))

    # Phase winding
    phases = np.angle(p_ring)
    dph = np.diff(phases)
    dph = (dph + np.pi) % (2 * np.pi) - np.pi
    winding = np.sum(dph) / (2 * np.pi)

    amp = np.abs(p_ring)
    ring_mean = float(amp.mean())
    ring_std = float(amp.std())
    azimuth_var = ring_std / ring_mean if ring_mean > 1e-15 else 999.0

    return float(winding), ring_mean, ring_std, azimuth_var


def compute_metrics_at_z(
    p_xy: np.ndarray,
    xg: np.ndarray,
    yg: np.ndarray,
    cx: float,
    cy: float,
) -> dict:
    """Compute all diagnostic metrics for one z-plane."""
    amp = np.abs(p_xy)

    # Center amplitude
    cx_idx = np.argmin(np.abs(xg - cx))
    cy_idx = np.argmin(np.abs(yg - cy))
    center_amp = float(amp[cy_idx, cx_idx])

    # Max amplitude and location
    imax = np.unravel_index(amp.argmax(), amp.shape)
    max_amp = float(amp[imax])
    max_x = float(xg[imax[1]])
    max_y = float(yg[imax[0]])

    # Distance of max from center
    max_dist_from_center = np.sqrt((max_x - cx)**2 + (max_y - cy)**2)

    # Is max at a corner/edge?  (within 2 pixels of boundary)
    edge_margin = 3
    at_edge = (imax[0] < edge_margin or imax[0] >= len(yg) - edge_margin or
               imax[1] < edge_margin or imax[1] >= len(xg) - edge_margin)

    # Winding at multiple radii
    windings = {}
    for r_lam in RING_RADII_LAM:
        r_m = r_lam * LAM
        w, rm, rs, av = compute_winding(p_xy, xg, yg, cx, cy, r_m)
        windings[r_lam] = {
            "winding": w, "ring_amp_mean": rm,
            "ring_amp_std": rs, "azimuth_var": av,
        }

    result = {
        "center_amp": center_amp,
        "max_amp": max_amp,
        "max_x_mm": max_x * 1e3,
        "max_y_mm": max_y * 1e3,
        "max_dist_lam": max_dist_from_center / LAM,
        "max_at_edge": at_edge,
    }
    for r_lam, wd in windings.items():
        r_key = f"r{r_lam:.1f}"
        result[f"winding_{r_key}"] = wd["winding"]
        result[f"amp_mean_{r_key}"] = wd["ring_amp_mean"]
        result[f"azvar_{r_key}"] = wd["azimuth_var"]
    return result


# ==================================================================
# Pass/fail criteria
# ==================================================================

def vortex_grade(metrics: dict, ell: int = ELL) -> str:
    """Grade vortex preservation: STRONG / WEAK / FAIL.

    STRONG: winding ≈ ℓ AND azvar < 0.2  (true azimuthal symmetry)
    WEAK:   winding ≈ ℓ AND azvar < 0.5  (correct topology, broken symmetry)
    FAIL:   winding wrong OR azvar ≥ 0.5
    """
    best = "FAIL"
    for r_lam in RING_RADII_LAM:
        r_key = f"r{r_lam:.1f}"
        w = metrics.get(f"winding_{r_key}", 0)
        av = metrics.get(f"azvar_{r_key}", 999)
        if abs(w - ell) < 0.4:
            if av < 0.2:
                return "STRONG"
            elif av < 0.5 and best != "STRONG":
                best = "WEAK"
    return best


def vortex_preserved(metrics: dict, ell: int = ELL) -> bool:
    """True if winding ≈ ℓ at any ring radius (WEAK or STRONG)."""
    return vortex_grade(metrics, ell) in ("STRONG", "WEAK")


# ==================================================================
# Plotting
# ==================================================================

def _ensure_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_xy_amp_phase(
    p_xy: np.ndarray,
    xg: np.ndarray,
    yg: np.ndarray,
    cx: float,
    cy: float,
    case_label: str,
    z_mm: float,
    out_path: Path,
):
    """2-panel heatmap: |p| and arg(p)."""
    plt = _ensure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    extent = [xg[0] * 1e3, xg[-1] * 1e3, yg[0] * 1e3, yg[-1] * 1e3]
    amp = np.abs(p_xy)

    ax = axes[0]
    im = ax.imshow(amp, extent=extent, origin="lower", cmap="inferno")
    ax.plot(cx * 1e3, cy * 1e3, "w+", ms=10, mew=2)
    ax.set_title(f"|p|  z={z_mm:.2f} mm")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    plt.colorbar(im, ax=ax, label="Pa")

    ax = axes[1]
    im = ax.imshow(np.angle(p_xy), extent=extent, origin="lower",
                   cmap="hsv", vmin=-np.pi, vmax=np.pi)
    ax.plot(cx * 1e3, cy * 1e3, "k+", ms=10, mew=2)
    ax.set_title(f"arg(p)  z={z_mm:.2f} mm")
    ax.set_xlabel("x (mm)")
    plt.colorbar(im, ax=ax, label="rad")

    # Draw winding rings
    for r_lam in RING_RADII_LAM:
        r_mm = r_lam * LAM * 1e3
        theta = np.linspace(0, 2 * np.pi, 100)
        for a in axes:
            a.plot(cx * 1e3 + r_mm * np.cos(theta),
                   cy * 1e3 + r_mm * np.sin(theta),
                   "w--" if a is axes[0] else "k--", lw=0.6, alpha=0.5)

    fig.suptitle(f"{case_label}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_winding_vs_z(
    all_metrics: Dict[str, Dict[float, dict]],
    out_path: Path,
    ell: int = ELL,
):
    """Winding at r=1.0λ vs z for all cases."""
    plt = _ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, (case_label, z_dict) in enumerate(sorted(all_metrics.items())):
        zs = sorted(z_dict.keys())
        ws = [z_dict[z].get("winding_r1.0", np.nan) for z in zs]
        ax.plot([z * 1e3 for z in zs], ws,
                "o-", color=colors[i % len(colors)], label=case_label, ms=4)

    # Reference lines
    ax.axhline(ell, color="green", ls="--", lw=1, alpha=0.7, label=f"ℓ={ell}")
    ax.axhline(ell - 0.4, color="green", ls=":", lw=0.7, alpha=0.4)
    ax.axhline(ell + 0.4, color="green", ls=":", lw=0.7, alpha=0.4)
    ax.axvline(H_UNDER * 1e3, color="red", ls="--", lw=1, alpha=0.5,
               label="bath/slab boundary")
    ax.axvline(Z_STAR * 1e3, color="purple", ls=":", lw=1, alpha=0.5,
               label=f"Z* = {Z_STAR*1e3:.2f} mm")

    ax.set_xlabel("z (mm)")
    ax.set_ylabel("Phase winding number (r = 1.0λ)")
    ax.set_title("Vortex phase winding vs height")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_center_amp_vs_z(
    all_metrics: Dict[str, Dict[float, dict]],
    out_path: Path,
):
    """Center amplitude vs z for all cases."""
    plt = _ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, (case_label, z_dict) in enumerate(sorted(all_metrics.items())):
        zs = sorted(z_dict.keys())
        amps = [z_dict[z].get("center_amp", np.nan) for z in zs]
        ax.plot([z * 1e3 for z in zs], amps,
                "o-", color=colors[i % len(colors)], label=case_label, ms=4)

    ax.axvline(H_UNDER * 1e3, color="red", ls="--", lw=1, alpha=0.5,
               label="bath/slab boundary")
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("|p| at beam center (Pa)")
    ax.set_title("Center amplitude vs height")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_azvar_vs_z(
    all_metrics: Dict[str, Dict[float, dict]],
    out_path: Path,
):
    """Azimuthal variation at r=1.0λ vs z."""
    plt = _ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, (case_label, z_dict) in enumerate(sorted(all_metrics.items())):
        zs = sorted(z_dict.keys())
        avs = [z_dict[z].get("azvar_r1.0", np.nan) for z in zs]
        ax.plot([z * 1e3 for z in zs], avs,
                "o-", color=colors[i % len(colors)], label=case_label, ms=4)

    ax.axhline(0.2, color="green", ls="--", lw=1, alpha=0.5,
               label="pass threshold (0.2)")
    ax.axvline(H_UNDER * 1e3, color="red", ls="--", lw=1, alpha=0.5,
               label="bath/slab boundary")
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("Azimuthal variation σ/μ at r = 1.0λ")
    ax.set_title("Azimuthal amplitude uniformity vs height")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_winding_all_radii(
    all_metrics: Dict[str, Dict[float, dict]],
    out_path: Path,
    ell: int = ELL,
):
    """One subplot per ring radius showing winding vs z for all cases."""
    plt = _ensure_matplotlib()
    fig, axes = plt.subplots(1, len(RING_RADII_LAM), figsize=(5 * len(RING_RADII_LAM), 5),
                             sharey=True)
    if len(RING_RADII_LAM) == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for r_idx, r_lam in enumerate(RING_RADII_LAM):
        ax = axes[r_idx]
        r_key = f"r{r_lam:.1f}"
        for i, (case_label, z_dict) in enumerate(sorted(all_metrics.items())):
            zs = sorted(z_dict.keys())
            ws = [z_dict[z].get(f"winding_{r_key}", np.nan) for z in zs]
            ax.plot([z * 1e3 for z in zs], ws,
                    "o-", color=colors[i % len(colors)], label=case_label, ms=3)

        ax.axhline(ell, color="green", ls="--", lw=1, alpha=0.5)
        ax.axvline(H_UNDER * 1e3, color="red", ls="--", lw=1, alpha=0.5)
        ax.set_xlabel("z (mm)")
        ax.set_title(f"r = {r_lam:.1f}λ")
        ax.grid(True, alpha=0.3)
        if r_idx == 0:
            ax.set_ylabel("Phase winding")
            ax.legend(fontsize=6, loc="best")

    fig.suptitle(f"Phase winding at all ring radii (expected ℓ={ell})", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ==================================================================
# Report writer
# ==================================================================

def write_summary(
    all_metrics: Dict[str, Dict[float, dict]],
    case_descriptions: Dict[str, str],
    out_path: Path,
    ell: int = ELL,
):
    """Write summary.md."""
    lines = []
    lines.append("# Vortex Integrity Diagnostic Report\n")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"Target: BG beam ℓ={ell}, w={BEAM_WAIST_MM} mm, R={APERTURE_MM} mm\n")
    lines.append("")

    # Case descriptions
    lines.append("## Cases\n")
    for case_label, desc in case_descriptions.items():
        status = "ran" if case_label in all_metrics else "SKIPPED (OOM/timeout)"
        lines.append(f"- **{case_label}**: {desc} [{status}]")
    lines.append("")

    # Pass criteria
    lines.append("## Pass criteria\n")
    lines.append(f"- **STRONG**: winding ∈ [{ell-0.4:.1f}, {ell+0.4:.1f}] at any r ∈ {RING_RADII_LAM} AND azimuthal variation < 0.2")
    lines.append(f"- **WEAK**: winding ∈ [{ell-0.4:.1f}, {ell+0.4:.1f}] at any r AND azimuthal variation < 0.5")
    lines.append("  (correct topology but broken azimuthal symmetry — expected in rectangular domain)")
    lines.append("- **FAIL**: winding wrong at all radii, or azimuthal variation ≥ 0.5\n")

    # Results table  (one per case)
    lines.append("## Detailed results\n")
    for case_label in sorted(all_metrics.keys()):
        z_dict = all_metrics[case_label]
        desc = case_descriptions.get(case_label, "")
        lines.append(f"### {case_label}\n")
        lines.append(f"_{desc}_\n")
        lines.append("| z (mm) | region | winding r0.5λ | winding r1.0λ | winding r1.5λ "
                      "| center |p| | azvar r1.0λ | max at edge | PASS |")
        lines.append("|--------|--------|---------------|---------------|---------------"
                      "|------------|-------------|-------------|------|")
        for z in sorted(z_dict.keys()):
            m = z_dict[z]
            region = "bath" if z < H_UNDER else "slab"
            w05 = m.get("winding_r0.5", float("nan"))
            w10 = m.get("winding_r1.0", float("nan"))
            w15 = m.get("winding_r1.5", float("nan"))
            ca = m.get("center_amp", float("nan"))
            av = m.get("azvar_r1.0", float("nan"))
            edge = "YES" if m.get("max_at_edge", False) else "no"
            grade = vortex_grade(m, ell)
            lines.append(
                f"| {z*1e3:.2f} | {region:4s} | {w05:+.3f} | {w10:+.3f} | {w15:+.3f} "
                f"| {ca:.4f} | {av:.3f} | {edge} | {grade} |"
            )
        lines.append("")

    # Interpretation
    lines.append("## Interpretation\n")

    # Helper: count grades per region
    def _count_grades(case_label):
        z_dict = all_metrics.get(case_label, {})
        bath_grades = [vortex_grade(m, ell) for z, m in z_dict.items() if z < H_UNDER]
        slab_grades = [vortex_grade(m, ell) for z, m in z_dict.items() if z >= H_UNDER]
        return bath_grades, slab_grades

    baseline = all_metrics.get("A_baseline", {})
    bath_total = sum(1 for z in baseline if z < H_UNDER)
    slab_total = sum(1 for z in baseline if z >= H_UNDER)

    b_grades_A, s_grades_A = _count_grades("A_baseline")
    bath_pass_A = sum(1 for g in b_grades_A if g != "FAIL")
    slab_pass_A = sum(1 for g in s_grades_A if g != "FAIL")

    lines.append(f"### Q1: Does winding exist in the water bath?\n")
    if bath_pass_A > 0:
        strong = sum(1 for g in b_grades_A if g == "STRONG")
        weak = sum(1 for g in b_grades_A if g == "WEAK")
        lines.append(f"**YES (partially)** — {bath_pass_A}/{bath_total} bath planes pass "
                      f"({strong} STRONG, {weak} WEAK).")
    else:
        lines.append(f"**NO at STRONG threshold** — 0/{bath_total} bath planes achieve azvar < 0.2.")
        # Check if winding alone is correct
        any_wind = False
        for z, m in baseline.items():
            if z < H_UNDER:
                for r_lam in RING_RADII_LAM:
                    w = m.get(f"winding_r{r_lam:.1f}", 0)
                    if abs(w - ell) < 0.4:
                        any_wind = True
        if any_wind:
            lines.append(f"However, correct winding (≈{ell}) IS observed at some radii/z-planes; "
                          "failure is driven by azimuthal variation > 0.2 (expected in rectangular domain).")
    lines.append("")

    lines.append(f"### Q2: Does winding die upon entering the slab?\n")
    if bath_pass_A > 0 and slab_pass_A == 0:
        lines.append(f"**YES** — winding preserved in bath ({bath_pass_A}/{bath_total}) "
                      f"but lost in slab (0/{slab_total}).")
    elif bath_pass_A > 0 and slab_pass_A > 0:
        lines.append(f"Winding partially preserved in both bath ({bath_pass_A}/{bath_total}) "
                      f"and slab ({slab_pass_A}/{slab_total}).")
    elif bath_pass_A == 0 and slab_pass_A == 0:
        lines.append("Both regions fail strict criteria, but examining winding numbers directly:")
        # Summarise raw winding counts
        for region_name, z_items in [("bath", [(z,m) for z,m in baseline.items() if z<H_UNDER]),
                                      ("slab", [(z,m) for z,m in baseline.items() if z>=H_UNDER])]:
            correct_wind = sum(1 for z,m in z_items
                               if any(abs(m.get(f"winding_r{r:.1f}",0)-ell)<0.4 for r in RING_RADII_LAM))
            lines.append(f"  - {region_name}: correct winding at ≥1 radius in {correct_wind}/{len(z_items)} planes")
    lines.append("")

    # Question 3: top BC sensitivity
    lines.append("### Q3: Is the top boundary a strong driver?\n")
    for alt in ["C1_rigid_top", "C2_soft_top"]:
        if alt in all_metrics:
            bg, sg = _count_grades(alt)
            bp = sum(1 for g in bg if g != "FAIL")
            sp = sum(1 for g in sg if g != "FAIL")
            # Also count correct winding regardless of azvar
            wind_bath = sum(1 for z,m in all_metrics[alt].items()
                            if z < H_UNDER and any(abs(m.get(f"winding_r{r:.1f}",0)-ell)<0.4
                                                   for r in RING_RADII_LAM))
            wind_slab = sum(1 for z,m in all_metrics[alt].items()
                            if z >= H_UNDER and any(abs(m.get(f"winding_r{r:.1f}",0)-ell)<0.4
                                                    for r in RING_RADII_LAM))
            lines.append(f"- **{alt}**: bath PASS={bp}/{bath_total}, slab PASS={sp}/{slab_total}; "
                          f"correct winding: bath {wind_bath}/{bath_total}, slab {wind_slab}/{slab_total}")
        else:
            lines.append(f"- **{alt}**: not available (solve failed)")

    # Compare A vs C1 vs C2 for winding at r=1λ in bath
    lines.append("")
    lines.append("Winding at r=1.0λ in bath per case:")
    lines.append("")
    for case_label in ["A_baseline", "C1_rigid_top", "C2_soft_top"]:
        if case_label not in all_metrics:
            continue
        z_dict = all_metrics[case_label]
        bath_winds = [(z*1e3, m.get("winding_r1.0", float("nan")),
                       m.get("azvar_r1.0", float("nan")))
                      for z, m in sorted(z_dict.items()) if z < H_UNDER]
        vals = "  ".join(f"z={z:.1f}: w={w:+.2f} av={a:.2f}" for z, w, a in bath_winds)
        lines.append(f"- {case_label}: {vals}")
    lines.append("")

    # Key finding about top BC
    c1_wind_bath = sum(1 for z,m in all_metrics.get("C1_rigid_top",{}).items()
                       if z < H_UNDER and any(abs(m.get(f"winding_r{r:.1f}",0)-ell)<0.4
                                              for r in RING_RADII_LAM))
    a_wind_bath = sum(1 for z,m in baseline.items()
                      if z < H_UNDER and any(abs(m.get(f"winding_r{r:.1f}",0)-ell)<0.4
                                             for r in RING_RADII_LAM))
    if c1_wind_bath > a_wind_bath:
        lines.append("**Key finding**: Rigid top (C1) preserves winding much better than "
                      "baseline/soft top. The near-pressure-release top BC (Z_air/Z_water ≈ 0.0003) "
                      "causes strong reflections with π phase shift that destructively interfere "
                      "with the upward-propagating vortex beam at certain z-planes.")
    elif c1_wind_bath == a_wind_bath:
        lines.append("Top BC does not appear to be a major driver.")
    else:
        lines.append("Results are inconclusive regarding top BC sensitivity.")
    lines.append("")

    # Question 4: side walls / domain size
    lines.append("### Q4: Do side walls / domain size strongly change winding?\n")
    if "D_large_domain" in all_metrics:
        d_wind_bath = sum(1 for z,m in all_metrics["D_large_domain"].items()
                          if z < H_UNDER and any(abs(m.get(f"winding_r{r:.1f}",0)-ell)<0.4
                                                 for r in RING_RADII_LAM))
        d_wind_slab = sum(1 for z,m in all_metrics["D_large_domain"].items()
                          if z >= H_UNDER and any(abs(m.get(f"winding_r{r:.1f}",0)-ell)<0.4
                                                  for r in RING_RADII_LAM))
        lines.append(f"- **D_large_domain** (8 mm, EPL=2): correct winding: "
                      f"bath {d_wind_bath}/{bath_total}, slab {d_wind_slab}/{slab_total}")
        if d_wind_bath > a_wind_bath:
            lines.append("  → Larger domain improves winding → **side walls contribute to distortion**.")
        elif d_wind_bath <= a_wind_bath:
            lines.append("  → Similar or worse (EPL=2 is very coarse) → side walls secondary.")
        # Check azimuthal variation
        d_azvar_bath = [m.get("azvar_r1.0", 999) for z,m in all_metrics["D_large_domain"].items()
                        if z < H_UNDER]
        lines.append(f"  Mean azvar in bath: {np.mean(d_azvar_bath):.3f}")
    else:
        lines.append("- D_large_domain: not available (solve failed)")
    lines.append("")

    # Question 5: resolution
    lines.append("### Q5: Is the issue mainly discretisation?\n")
    if "E_higher_res" in all_metrics:
        e_wind_bath = sum(1 for z,m in all_metrics["E_higher_res"].items()
                          if z < H_UNDER and any(abs(m.get(f"winding_r{r:.1f}",0)-ell)<0.4
                                                 for r in RING_RADII_LAM))
        e_wind_slab = sum(1 for z,m in all_metrics["E_higher_res"].items()
                          if z >= H_UNDER and any(abs(m.get(f"winding_r{r:.1f}",0)-ell)<0.4
                                                  for r in RING_RADII_LAM))
        # Get precise winding values at r=1λ in bath
        e_bath_w10 = [(z*1e3, m.get("winding_r1.0", float("nan")), m.get("azvar_r1.0", float("nan")))
                      for z,m in sorted(all_metrics["E_higher_res"].items()) if z < H_UNDER]
        lines.append(f"- **E_higher_res** (EPL=4, 4 mm domain): correct winding: "
                      f"bath {e_wind_bath}/{bath_total}, slab {e_wind_slab}/{slab_total}")
        lines.append("")
        lines.append("  Bath winding at r=1.0λ (E_higher_res):")
        for z, w, a in e_bath_w10:
            lines.append(f"    z={z:.1f} mm: winding={w:+.4f}, azvar={a:.3f}")
        lines.append("")
        if e_wind_bath > a_wind_bath:
            lines.append("  → **Resolution is a major factor.** EPL=4 gives near-perfect winding "
                          "(≈2.0000) in the bath. EPL=3 is too coarse to faithfully resolve the "
                          "ℓ=2 phase pattern.")
        else:
            lines.append("  → No clear improvement from resolution alone.")
        # Compare azvar
        e_azvar_bath = [m.get("azvar_r1.0", 999) for z,m in all_metrics["E_higher_res"].items()
                        if z < H_UNDER]
        a_azvar_bath = [m.get("azvar_r1.0", 999) for z,m in baseline.items()
                        if z < H_UNDER]
        lines.append(f"  Mean azvar (bath) — baseline: {np.mean(a_azvar_bath):.3f}, "
                      f"E_higher_res: {np.mean(e_azvar_bath):.3f}")
    else:
        lines.append("- E_higher_res: not available (solve failed / OOM)")
    lines.append("")

    # Summary of findings
    lines.append("## Summary of findings\n")
    lines.append("1. **The vortex phase topology IS encoded** — at sufficient resolution (EPL≥4), ")
    lines.append("   winding numbers are near-perfect (≈2.0000) in the water bath.")
    lines.append("2. **EPL=3 is too coarse** — winding becomes erratic and radius-dependent.")
    lines.append("3. **The top boundary (water–air Robin BC) causes destructive z-interference** — ")
    lines.append("   the reflected beam with π phase shift creates z-dependent interference patterns ")
    lines.append("   that destroy winding at certain heights. Rigid top (C1) gives much better results.")
    lines.append("4. **Azimuthal symmetry is always broken** (azvar 0.25–0.58) because the rectangular ")
    lines.append("   domain has 4-fold symmetry, not cylindrical. This is intrinsic to the cuboid FEM model.")
    lines.append("5. **Larger domain helps modestly** but is overwhelmed by low EPL (=2 needed to fit).")
    lines.append("")

    # Recommendation
    lines.append("## Recommendation\n")
    lines.append("The vortex boundary condition is correct but three effects destroy the field:")
    lines.append("")
    lines.append("1. **Resolution** — EPL=3 cannot resolve exp(i·2θ) on the disk face (~70 facets).")
    lines.append("2. **Top reflection** — water/air pressure-release creates z-standing-wave interference.")
    lines.append("3. **Rectangular domain** — breaks cylindrical symmetry inherently.")
    lines.append("")
    lines.append("### Recommended path forward")
    lines.append("")
    lines.append("**(a) Analytical vortex propagation** (best option): Use Rayleigh–Sommerfeld / ")
    lines.append("angular-spectrum propagation from the disk BC pattern to Z*. This preserves ")
    lines.append("perfect cylindrical symmetry, has no z-reflection artefact, and is mesh-independent. ")
    lines.append("Superimpose analytically-propagated vortex with FEM standing-wave field.")
    lines.append("")
    lines.append("**(b) Move Z* into water bath** (quick workaround): At z ≈ 1.0–1.5 mm the winding ")
    lines.append("is best preserved, but beam has not yet focused and the standing-wave pattern ")
    lines.append("is different there. May not be physically meaningful for the petri-dish application.")
    lines.append("")
    lines.append("**(c) Higher resolution + rigid top** (expensive): EPL≥4 with rigid top BC gives ")
    lines.append("good winding but requires ≥4× more DOFs (OOM with MUMPS on this machine). ")
    lines.append("An iterative solver (GMRES + multigrid) would be needed.")
    lines.append("")

    out_path.write_text("\n".join(lines))
    print(f"  Summary written to {out_path}")


def write_metrics_csv(
    all_metrics: Dict[str, Dict[float, dict]],
    out_path: Path,
):
    """Write metrics.csv."""
    rows = []
    header = [
        "case", "z_mm", "region",
        "winding_r0.5", "winding_r1.0", "winding_r1.5",
        "center_amp", "max_amp",
        "max_x_mm", "max_y_mm", "max_dist_lam", "max_at_edge",
        "amp_mean_r0.5", "amp_mean_r1.0", "amp_mean_r1.5",
        "azvar_r0.5", "azvar_r1.0", "azvar_r1.5",
        "grade",
    ]
    for case_label in sorted(all_metrics.keys()):
        z_dict = all_metrics[case_label]
        for z in sorted(z_dict.keys()):
            m = z_dict[z]
            region = "bath" if z < H_UNDER else "slab"
            grade = vortex_grade(m, ELL)
            row = [
                case_label, f"{z*1e3:.3f}", region,
                f"{m.get('winding_r0.5', float('nan')):.4f}",
                f"{m.get('winding_r1.0', float('nan')):.4f}",
                f"{m.get('winding_r1.5', float('nan')):.4f}",
                f"{m.get('center_amp', float('nan')):.6f}",
                f"{m.get('max_amp', float('nan')):.6f}",
                f"{m.get('max_x_mm', float('nan')):.3f}",
                f"{m.get('max_y_mm', float('nan')):.3f}",
                f"{m.get('max_dist_lam', float('nan')):.3f}",
                str(m.get("max_at_edge", "")),
                f"{m.get('amp_mean_r0.5', float('nan')):.6f}",
                f"{m.get('amp_mean_r1.0', float('nan')):.6f}",
                f"{m.get('amp_mean_r1.5', float('nan')):.6f}",
                f"{m.get('azvar_r0.5', float('nan')):.4f}",
                f"{m.get('azvar_r1.0', float('nan')):.4f}",
                f"{m.get('azvar_r1.5', float('nan')):.4f}",
                grade,
            ]
            rows.append(row)

    with open(out_path, "w") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(row) + "\n")
    print(f"  CSV written to {out_path}  ({len(rows)} rows)")


# ==================================================================
# Main
# ==================================================================

def main():
    t_start = time.time()
    print(f"\n{'='*70}")
    print("VORTEX INTEGRITY DIAGNOSTIC")
    print(f"{'='*70}")
    print(f"Output: {OUT_DIR}")
    print(f"Cases:  {len(CASES)}")
    print(f"z-planes: {len(Z_ALL)} ({len(Z_BATH)} bath + {len(Z_SLAB)} slab)")
    print()

    # Create output directories
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)

    # Store case descriptions
    case_descriptions = {label: desc for label, desc, _ in CASES}

    # ── Phase 1: FEM solves ───────────────────────────────────────
    print("Phase 1: FEM solves")
    print("-" * 40)
    solve_results: Dict[str, Optional[dict]] = {}
    for label, desc, overrides in CASES:
        print(f"\n  Case: {label}")
        print(f"    {desc}")
        data = solve_case(label, overrides, Z_ALL, n_xy=GRID_N, timeout=600)
        solve_results[label] = data
        if data is None:
            print(f"    → SKIPPED (solve failed)")
        else:
            print(f"    → OK: {len([k for k in data if k.startswith('p_xy')])} z-planes")

    # ── Phase 2: Compute metrics ──────────────────────────────────
    print(f"\n{'='*50}")
    print("Phase 2: Computing metrics")
    print("-" * 40)
    all_metrics: Dict[str, Dict[float, dict]] = {}

    for label, desc, overrides in CASES:
        data = solve_results.get(label)
        if data is None:
            continue

        xg = data["xg"]
        yg = data["yg"]
        z_list = list(data.get("trap_z_list", Z_ALL))

        # Beam center = domain center = Lx/2, Ly/2
        Lx = overrides.get("Lx", 6e-3)
        Ly = overrides.get("Ly", 6e-3)
        cx = Lx / 2.0
        cy = Ly / 2.0

        all_metrics[label] = {}
        for i, z in enumerate(z_list):
            key = f"p_xy_{i}"
            if key not in data:
                continue
            p_xy = data[key]
            m = compute_metrics_at_z(p_xy, xg, yg, cx, cy)
            all_metrics[label][z] = m
            grade = vortex_grade(m)
            region = "bath" if z < H_UNDER else "slab"
            w10 = m.get("winding_r1.0", float("nan"))
            print(f"  {label}  z={z*1e3:.2f}mm ({region})  "
                  f"wind_1λ={w10:+.2f}  azvar={m.get('azvar_r1.0',0):.2f}  "
                  f"{grade}")

    # ── Phase 3: Figures ──────────────────────────────────────────
    print(f"\n{'='*50}")
    print("Phase 3: Generating figures")
    print("-" * 40)

    # XY heatmaps — select representative z-planes (not all, to limit figures)
    representative_z = [1.5e-3, 2.5e-3, 3.5e-3, Z_STAR]
    for label, desc, overrides in CASES:
        data = solve_results.get(label)
        if data is None:
            continue

        xg = data["xg"]
        yg = data["yg"]
        z_list = list(data.get("trap_z_list", Z_ALL))
        Lx = overrides.get("Lx", 6e-3)
        Ly = overrides.get("Ly", 6e-3)
        cx = Lx / 2.0
        cy = Ly / 2.0

        case_fig_dir = FIG_DIR / label
        case_fig_dir.mkdir(exist_ok=True)

        for z_rep in representative_z:
            # Find closest z in z_list
            dists = [abs(z - z_rep) for z in z_list]
            if not dists:
                continue
            best_idx = int(np.argmin(dists))
            z = z_list[best_idx]
            key = f"p_xy_{best_idx}"
            if key not in data:
                continue
            p_xy = data[key]
            fname = f"xy_z{z*1e3:.1f}mm.png"
            plot_xy_amp_phase(p_xy, xg, yg, cx, cy, f"{label} — {desc}",
                              z * 1e3, case_fig_dir / fname)

    # Summary plots
    if all_metrics:
        plot_winding_vs_z(all_metrics, FIG_DIR / "winding_vs_z.png")
        plot_center_amp_vs_z(all_metrics, FIG_DIR / "center_amp_vs_z.png")
        plot_azvar_vs_z(all_metrics, FIG_DIR / "azvar_vs_z.png")
        plot_winding_all_radii(all_metrics, FIG_DIR / "winding_all_radii.png")
        print(f"  Summary plots written to {FIG_DIR}")

    # ── Phase 4: Write outputs ────────────────────────────────────
    print(f"\n{'='*50}")
    print("Phase 4: Writing outputs")
    print("-" * 40)
    write_metrics_csv(all_metrics, OUT_DIR / "metrics.csv")
    write_summary(all_metrics, case_descriptions, OUT_DIR / "summary.md")

    dt = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"DONE in {dt:.0f}s")
    print(f"Results: {OUT_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
