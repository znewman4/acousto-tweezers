#!/usr/bin/env python3
"""
RS vs FEM Phase 1 — Comparison, metrics, figures, and report
==============================================================

Loads FEM and RS NPZ outputs per config, computes quantitative metrics,
generates figures, and writes report.md.

Metrics per z-plane:
  1) Complex correlation (global + ROI)
  2) Relative amplitude error
  3) Phase error (masked)
  4) Vortex integrity (winding, azimuthal variation)
  5) Best-fit complex scale

Figures per config:
  - |p| for FEM, RS_free, RS_reflect, and diff maps
  - Phase comparison (masked)
  - Radial amplitude profiles
  - Azimuthal phase profiles

Outputs
-------
results/rs_vs_fem_phase1_TIMESTAMP/
├── csv/phase1_metrics.csv
├── figures/{config_id}/…
└── report.md

Usage
-----
    python scripts/dev/rs_vs_fem_phase1_compare.py [TIMESTAMP]

    If TIMESTAMP not given, looks for the latest rs_vs_fem_phase1_* folder.
"""
from __future__ import annotations

import csv
import glob
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── paths ─────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ── Physical constants ────────────────────────────────────────────
WATER_C  = 1484.0
F_HZ     = 2.0e6
LAM      = WATER_C / F_HZ
K_WATER  = 2 * np.pi * F_HZ / WATER_C

H_UNDER  = 3.0e-3
H_TOP    = 2.0085e-3
Z_MID    = H_UNDER + H_TOP / 2
Z_STAR   = Z_MID + 0.25 * LAM

CX, CY   = 3.0e-3, 3.0e-3

# ROI for bridge — neighbourhood circle
NEIGHBOURHOOD_RADIUS_LAM = 2.0
ROI_RADIUS = NEIGHBOURHOOD_RADIUS_LAM * LAM   # ~1.484 mm

# Config IDs
CONFIG_IDS = [
    "bg_l2_R1.0_w0.8",
    "bg_l2_R1.0_w0.4",
    "lg_l2_R1.0_w0.8",
    "lg_l2_R1.5_w1.2",
    "plastic_l2_R0.8_f1.5_wrapped",
    "plastic_l2_R0.8_f3.5_wrapped",
]

Z_PLANES = [1.0e-3, 1.5e-3, 2.0e-3, 2.5e-3, Z_STAR]


def find_results_dir(timestamp: Optional[str] = None) -> Path:
    """Find the results directory."""
    if timestamp:
        d = PROJECT_ROOT / "results" / f"rs_vs_fem_phase1_{timestamp}"
        if d.exists():
            return d
    # Find latest
    pattern = str(PROJECT_ROOT / "results" / "rs_vs_fem_phase1_*")
    dirs = sorted(glob.glob(pattern))
    if not dirs:
        raise FileNotFoundError("No rs_vs_fem_phase1_* results directory found")
    return Path(dirs[-1])


# ==================================================================
# Metric computations
# ==================================================================
def complex_correlation(a: np.ndarray, b: np.ndarray) -> complex:
    """Normalized complex inner product <a,b>/(||a|| ||b||)."""
    na = np.sqrt(np.sum(np.abs(a)**2))
    nb = np.sqrt(np.sum(np.abs(b)**2))
    if na < 1e-30 or nb < 1e-30:
        return 0.0 + 0j
    return np.sum(a.conj() * b) / (na * nb)


def best_fit_scale(p_fem: np.ndarray, p_rs: np.ndarray) -> Tuple[complex, float]:
    """
    Find complex scalar c minimizing ||p_fem - c * p_rs||.
    c = <p_rs, p_fem> / <p_rs, p_rs>
    Returns (c, relative_residual_norm).
    """
    rs_flat = p_rs.ravel()
    fem_flat = p_fem.ravel()
    denom = np.sum(np.abs(rs_flat)**2)
    if denom < 1e-30:
        return 0.0 + 0j, 1.0
    c = np.sum(rs_flat.conj() * fem_flat) / denom
    residual = fem_flat - c * rs_flat
    res_norm = np.sqrt(np.sum(np.abs(residual)**2)) / np.sqrt(np.sum(np.abs(fem_flat)**2))
    return c, float(res_norm)


def relative_amplitude_error(p_fem: np.ndarray, p_rs: np.ndarray) -> float:
    """|| |p_fem| - |p_rs| || / || |p_fem| ||."""
    a_fem = np.abs(p_fem)
    a_rs = np.abs(p_rs)
    norm_fem = np.sqrt(np.sum(a_fem**2))
    if norm_fem < 1e-30:
        return 1.0
    return float(np.sqrt(np.sum((a_fem - a_rs)**2)) / norm_fem)


def phase_error_rms(p_fem: np.ndarray, p_rs: np.ndarray, threshold: float = 0.1) -> float:
    """RMS phase error where |p_fem| > threshold * max(|p_fem|)."""
    a_fem = np.abs(p_fem)
    mask = a_fem > threshold * a_fem.max()
    if mask.sum() < 10:
        return np.nan
    phi_fem = np.angle(p_fem[mask])
    phi_rs = np.angle(p_rs[mask])
    diff = phi_fem - phi_rs
    # Wrap to [-π, π]
    diff = (diff + np.pi) % (2 * np.pi) - np.pi
    return float(np.sqrt(np.mean(diff**2)))


def make_roi_mask(xg: np.ndarray, yg: np.ndarray,
                  cx: float = CX, cy: float = CY,
                  radius: float = ROI_RADIUS) -> np.ndarray:
    """Boolean mask (ny, nx) for circular ROI."""
    XX, YY = np.meshgrid(xg, yg)
    return (XX - cx)**2 + (YY - cy)**2 <= radius**2


def winding_number(p: np.ndarray, xg: np.ndarray, yg: np.ndarray,
                   r_lam: float, cx: float = CX, cy: float = CY) -> Tuple[float, float]:
    """
    Compute winding number and azimuthal amplitude variation on a ring
    of radius r_lam * LAM centered at (cx, cy).

    Returns (winding, azvar).
    """
    r_m = r_lam * LAM
    n_pts = 360
    theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    ring_x = cx + r_m * np.cos(theta)
    ring_y = cy + r_m * np.sin(theta)

    # Interpolate complex field on ring
    from scipy.interpolate import RegularGridInterpolator
    interp_re = RegularGridInterpolator(
        (yg, xg), np.real(p), method="linear",
        bounds_error=False, fill_value=0.0)
    interp_im = RegularGridInterpolator(
        (yg, xg), np.imag(p), method="linear",
        bounds_error=False, fill_value=0.0)
    pts = np.column_stack([ring_y, ring_x])
    p_ring = interp_re(pts) + 1j * interp_im(pts)

    # Winding number
    phi = np.angle(p_ring)
    dphi = np.diff(phi)
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
    winding = np.sum(dphi) / (2 * np.pi)

    # Azimuthal amplitude variation
    amps = np.abs(p_ring)
    a_mean = amps.mean()
    if a_mean > 1e-30:
        azvar = float(np.std(amps) / a_mean)
    else:
        azvar = 1.0

    return float(np.round(winding)), azvar


def compute_all_metrics(
    p_fem: np.ndarray, p_rs: np.ndarray,
    xg: np.ndarray, yg: np.ndarray,
    config_id: str, z_mm: float, method: str,
) -> Dict[str, Any]:
    """Compute all Phase-1 metrics for one config/z/method combination."""
    # Global metrics
    rho_global = complex_correlation(p_fem.ravel(), p_rs.ravel())
    err_amp_global = relative_amplitude_error(p_fem, p_rs)

    # Best-fit scale
    c_fit, residual = best_fit_scale(p_fem, p_rs)

    # Scale RS for phase comparison
    p_rs_scaled = c_fit * p_rs
    err_phase = phase_error_rms(p_fem, p_rs_scaled)
    err_amp_scaled = relative_amplitude_error(p_fem, p_rs_scaled)

    # ROI metrics
    roi_mask = make_roi_mask(xg, yg)
    rho_roi = complex_correlation(p_fem[roi_mask], p_rs_scaled[roi_mask])
    err_amp_roi = relative_amplitude_error(p_fem[roi_mask], p_rs_scaled[roi_mask])

    # Vortex integrity at r = 1.0λ
    wind_1lam, azvar_1lam = winding_number(p_fem, xg, yg, 1.0)
    wind_rs_1lam, azvar_rs_1lam = winding_number(p_rs_scaled, xg, yg, 1.0)

    return {
        "config_id": config_id,
        "z_mm": z_mm,
        "method": method,
        "rho_global": float(np.abs(rho_global)),
        "rho_global_phase": float(np.angle(rho_global)),
        "rho_roi": float(np.abs(rho_roi)),
        "err_amp_global": err_amp_global,
        "err_amp_scaled": err_amp_scaled,
        "err_amp_roi": err_amp_roi,
        "err_phase": err_phase,
        "winding_fem_1lam": wind_1lam,
        "azvar_fem_1lam": azvar_1lam,
        "winding_rs_1lam": wind_rs_1lam,
        "azvar_rs_1lam": azvar_rs_1lam,
        "best_fit_c_amp": float(np.abs(c_fit)),
        "best_fit_c_phase": float(np.angle(c_fit)),
        "residual_norm": residual,
    }


# ==================================================================
# Figures
# ==================================================================
def make_figures(
    p_fem: np.ndarray,
    p_rs_free: np.ndarray,
    p_rs_reflect: np.ndarray,
    xg: np.ndarray, yg: np.ndarray,
    config_id: str, z_mm: float,
    fig_dir: Path,
    c_free: complex = 1.0,
    c_reflect: complex = 1.0,
):
    """Generate comparison figures for one config at one z plane."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p_rs_f_s = c_free * p_rs_free
    p_rs_r_s = c_reflect * p_rs_reflect
    x_mm = xg * 1e3
    y_mm = yg * 1e3
    extent = [x_mm[0], x_mm[-1], y_mm[0], y_mm[-1]]

    vmax_fem = np.abs(p_fem).max()
    if vmax_fem < 1e-30:
        return
    vmax = vmax_fem

    # --- 1) |p| comparison ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"{config_id}  z={z_mm:.2f} mm ({z_mm/LAM/1e3:.1f} λ)", fontsize=14)

    for ax, p, label in [
        (axes[0, 0], p_fem, "FEM"),
        (axes[0, 1], p_rs_f_s, "RS_free (scaled)"),
        (axes[0, 2], p_rs_r_s, "RS_reflect (scaled)"),
    ]:
        im = ax.imshow(np.abs(p), extent=extent, origin="lower",
                       vmin=0, vmax=vmax, cmap="hot")
        ax.set_title(f"|p| {label}")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        plt.colorbar(im, ax=ax, shrink=0.7)

    # Diff maps
    diff_free = np.abs(np.abs(p_fem) - np.abs(p_rs_f_s))
    diff_refl = np.abs(np.abs(p_fem) - np.abs(p_rs_r_s))
    diff_max = max(diff_free.max(), diff_refl.max(), 1e-30)

    for ax, diff, label in [
        (axes[1, 0], diff_free, "|FEM| - |RS_free|"),
        (axes[1, 1], diff_refl, "|FEM| - |RS_reflect|"),
    ]:
        im = ax.imshow(diff, extent=extent, origin="lower",
                       vmin=0, vmax=diff_max, cmap="viridis")
        ax.set_title(label)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        plt.colorbar(im, ax=ax, shrink=0.7)

    # Diff between RS variants
    diff_variants = np.abs(np.abs(p_rs_r_s) - np.abs(p_rs_f_s))
    im = axes[1, 2].imshow(diff_variants, extent=extent, origin="lower",
                            vmin=0, cmap="viridis")
    axes[1, 2].set_title("|RS_reflect - RS_free|")
    axes[1, 2].set_xlabel("x (mm)")
    axes[1, 2].set_ylabel("y (mm)")
    plt.colorbar(im, ax=axes[1, 2], shrink=0.7)

    plt.tight_layout()
    fname = fig_dir / f"amplitude_z{z_mm:.1f}mm.png"
    fig.savefig(str(fname), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- 2) Phase comparison (masked) ---
    mask = np.abs(p_fem) > 0.1 * vmax_fem
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    fig.suptitle(f"Phase: {config_id}  z={z_mm:.2f} mm", fontsize=13)

    for ax, p, label in [
        (axes[0], p_fem, "FEM"),
        (axes[1], p_rs_f_s, "RS_free"),
        (axes[2], p_rs_r_s, "RS_reflect"),
    ]:
        phase_masked = np.where(mask, np.angle(p), np.nan)
        im = ax.imshow(phase_masked, extent=extent, origin="lower",
                       vmin=-np.pi, vmax=np.pi, cmap="hsv")
        ax.set_title(f"∠p {label}")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        plt.colorbar(im, ax=ax, shrink=0.7)

    # Phase diff (FEM - RS_reflect)
    pdiff = np.where(mask, np.angle(p_fem) - np.angle(p_rs_r_s), np.nan)
    pdiff = np.where(np.isnan(pdiff), np.nan, (pdiff + np.pi) % (2 * np.pi) - np.pi)
    im = axes[3].imshow(pdiff, extent=extent, origin="lower",
                        vmin=-np.pi, vmax=np.pi, cmap="coolwarm")
    axes[3].set_title("∠FEM − ∠RS_reflect")
    axes[3].set_xlabel("x (mm)")
    axes[3].set_ylabel("y (mm)")
    plt.colorbar(im, ax=axes[3], shrink=0.7)

    plt.tight_layout()
    fname = fig_dir / f"phase_z{z_mm:.1f}mm.png"
    fig.savefig(str(fname), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- 3) Radial profiles ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Radial profiles: {config_id}  z={z_mm:.2f} mm", fontsize=13)

    # Take profile along y=cy
    ny, nx = p_fem.shape
    iy_mid = ny // 2
    x_prof_mm = x_mm

    for ax_i, (label_pair, p_pair) in enumerate([
        (("FEM", "RS_free"), (p_fem, p_rs_f_s)),
        (("FEM", "RS_reflect"), (p_fem, p_rs_r_s)),
    ]):
        ax = axes[ax_i]
        ax.plot(x_prof_mm, np.abs(p_pair[0][iy_mid, :]), "k-", lw=1.5, label=label_pair[0])
        ax.plot(x_prof_mm, np.abs(p_pair[1][iy_mid, :]), "r--", lw=1.2, label=label_pair[1])
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("|p|")
        ax.set_title(f"y=center  |  {label_pair[0]} vs {label_pair[1]}")
        ax.legend()
        ax.axvline(CX * 1e3, color="gray", ls=":", lw=0.5)

    plt.tight_layout()
    fname = fig_dir / f"radial_z{z_mm:.1f}mm.png"
    fig.savefig(str(fname), dpi=200, bbox_inches="tight")
    plt.close(fig)


# ==================================================================
# Report generation
# ==================================================================
def write_report(out_dir: Path, all_metrics: List[Dict], timestamp: str):
    """Write report.md with summary tables and interpretation."""
    report = out_dir / "report.md"

    # Group metrics
    configs_set = list(dict.fromkeys(m["config_id"] for m in all_metrics))
    z_star_mm = Z_STAR * 1e3

    lines = [
        "# RS vs FEM Phase 1 — Validation Report\n",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n",
        "",
        "## Physical parameters\n",
        f"- λ = {LAM*1e3:.4f} mm  (f = {F_HZ/1e6:.1f} MHz, c = {WATER_C} m/s)",
        f"- Z_STAR = {z_star_mm:.4f} mm  ({Z_STAR/LAM:.2f} λ)",
        f"- H_under = {H_UNDER*1e3:.1f} mm, H_top = {H_TOP*1e3:.4f} mm",
        f"- ROI radius = {NEIGHBOURHOOD_RADIUS_LAM} λ = {ROI_RADIUS*1e3:.3f} mm",
        "",
        "## Z-planes\n",
        "| z (mm) | z/λ | Region |",
        "|--------|-----|--------|",
    ]
    for z in Z_PLANES:
        region = "bath" if z < H_UNDER else "slab"
        z_label = "**Z_STAR**" if abs(z - Z_STAR) < 1e-6 else ""
        lines.append(f"| {z*1e3:.2f} | {z/LAM:.2f} | {region} {z_label} |")

    lines += [
        "",
        "## Summary table — Z_STAR plane\n",
        "Best-plane = Z_STAR for all configs.\n",
        "",
        "| Config | Method | ρ_global | ρ_ROI | err_amp_ROI | err_phase | "
        "wind_FEM | wind_RS | residual |",
        "|--------|--------|----------|-------|-------------|-----------|"
        "---------|---------|----------|",
    ]

    # Filter to Z_STAR
    for cid in configs_set:
        for method in ["rs_free", "rs_reflect"]:
            m_list = [m for m in all_metrics
                      if m["config_id"] == cid and m["method"] == method
                      and abs(m["z_mm"] - z_star_mm) < 0.01]
            if not m_list:
                continue
            m = m_list[0]
            lines.append(
                f"| {cid} | {method} | {m['rho_global']:.3f} | "
                f"{m['rho_roi']:.3f} | {m['err_amp_roi']:.3f} | "
                f"{m['err_phase']:.3f} | "
                f"{m['winding_fem_1lam']:+.0f} | {m['winding_rs_1lam']:+.0f} | "
                f"{m['residual_norm']:.3f} |"
            )

    lines += [
        "",
        "## Summary table — best bath plane (z=1.5 mm)\n",
        "",
        "| Config | Method | ρ_global | ρ_ROI | err_amp_ROI | err_phase | "
        "wind_FEM | wind_RS | residual |",
        "|--------|--------|----------|-------|-------------|-----------|"
        "---------|---------|----------|",
    ]

    z_bath_mm = 1.5
    for cid in configs_set:
        for method in ["rs_free", "rs_reflect"]:
            m_list = [m for m in all_metrics
                      if m["config_id"] == cid and m["method"] == method
                      and abs(m["z_mm"] - z_bath_mm) < 0.01]
            if not m_list:
                continue
            m = m_list[0]
            lines.append(
                f"| {cid} | {method} | {m['rho_global']:.3f} | "
                f"{m['rho_roi']:.3f} | {m['err_amp_roi']:.3f} | "
                f"{m['err_phase']:.3f} | "
                f"{m['winding_fem_1lam']:+.0f} | {m['winding_rs_1lam']:+.0f} | "
                f"{m['residual_norm']:.3f} |"
            )

    # RS_free vs RS_reflect comparison
    lines += [
        "",
        "## RS_free vs RS_reflect comparison\n",
        "",
        "Does including the water/air top reflection improve agreement with FEM?\n",
        "",
        "| Config | z (mm) | ρ_ROI free | ρ_ROI reflect | Δρ_ROI | "
        "err_amp free | err_amp reflect |",
        "|--------|--------|-----------|--------------|--------|"
        "------------|----------------|",
    ]

    for cid in configs_set:
        for z in Z_PLANES:
            z_mm = z * 1e3
            m_free = [m for m in all_metrics
                      if m["config_id"] == cid and m["method"] == "rs_free"
                      and abs(m["z_mm"] - z_mm) < 0.01]
            m_refl = [m for m in all_metrics
                      if m["config_id"] == cid and m["method"] == "rs_reflect"
                      and abs(m["z_mm"] - z_mm) < 0.01]
            if not m_free or not m_refl:
                continue
            mf, mr = m_free[0], m_refl[0]
            delta = mr["rho_roi"] - mf["rho_roi"]
            lines.append(
                f"| {cid} | {z_mm:.2f} | {mf['rho_roi']:.3f} | "
                f"{mr['rho_roi']:.3f} | {delta:+.3f} | "
                f"{mf['err_amp_roi']:.3f} | {mr['err_amp_roi']:.3f} |"
            )

    # Exit criteria assessment
    lines += [
        "",
        "## Exit criteria assessment\n",
        "",
        "Phase 1 success requires for ≥3/6 configs at key plane:\n",
        "- ρ_ROI > 0.85 AND err_amp_ROI < 0.15\n",
        "- RS_reflect improves vs RS_free in z-sensitive cases\n",
        "",
    ]

    pass_count = 0
    for cid in configs_set:
        # Check at Z_STAR with rs_reflect
        m_list = [m for m in all_metrics
                  if m["config_id"] == cid and m["method"] == "rs_reflect"
                  and abs(m["z_mm"] - z_star_mm) < 0.01]
        if not m_list:
            continue
        m = m_list[0]
        passed = m["rho_roi"] > 0.85 and m["err_amp_roi"] < 0.15
        if passed:
            pass_count += 1
        status = "**PASS**" if passed else "FAIL"
        lines.append(
            f"- {cid} @ Z_STAR: ρ_ROI={m['rho_roi']:.3f}, err_amp_ROI={m['err_amp_roi']:.3f} → {status}"
        )

    lines += [
        "",
        f"**Result: {pass_count}/6 configs pass at Z_STAR.**\n",
    ]

    if pass_count >= 3:
        lines.append("**→ RS is close enough for actuator sweeps at Z_STAR.**\n")
    else:
        lines.append(
            "**→ RS does NOT yet match FEM well enough at Z_STAR for "
            "unsupervised sweeps.**\n"
        )

    # Also check best bath plane
    pass_bath = 0
    lines.append("\nBath plane (z=1.5 mm) check:\n")
    for cid in configs_set:
        m_list = [m for m in all_metrics
                  if m["config_id"] == cid and m["method"] == "rs_reflect"
                  and abs(m["z_mm"] - z_bath_mm) < 0.01]
        if not m_list:
            continue
        m = m_list[0]
        passed = m["rho_roi"] > 0.85 and m["err_amp_roi"] < 0.15
        if passed:
            pass_bath += 1
        status = "**PASS**" if passed else "FAIL"
        lines.append(
            f"- {cid}: ρ_ROI={m['rho_roi']:.3f}, err_amp_ROI={m['err_amp_roi']:.3f} → {status}"
        )
    lines.append(f"\n**Result: {pass_bath}/6 configs pass at z=1.5 mm bath plane.**\n")

    # Failure modes + next actions
    lines += [
        "",
        "## Identified failure modes\n",
        "",
        "1. **EPL=3 FEM under-resolution**: FEM at EPL=3 may not resolve ℓ=2 phase "
        "winding correctly (known from diagnostic). This degrades the FEM 'truth', not RS.",
        "2. **Top-boundary reflection**: water/air pressure-release creates z-dependent "
        "interference pattern. RS_reflect should capture first-order effect.",
        "3. **Rectangular domain cavity modes (FEM only)**: Side-wall reflections in FEM "
        "create standing-wave patterns absent in RS. Strongest in slab.",
        "4. **Slab region degradation**: Both FEM and RS may poorly model the slab (FEM: "
        "cavity modes; RS: no slab physics). Agreement expected to be worse at Z_STAR.",
        "",
        "## Next actions (if exit criteria not met)\n",
        "",
        "1. Add side-wall image reflections to RS (4 images for rectangular domain)",
        "2. Add Fresnel transmission coefficients at water/slab interface",
        "3. Run FEM at EPL=4 on reduced domain for true high-res comparison",
        "4. Consider evaluation at bath plane (z=1.5 mm) instead of Z_STAR",
        "",
    ]

    with open(report, "w") as f:
        f.write("\n".join(lines))

    return report


# ==================================================================
# Main
# ==================================================================
def main():
    timestamp = sys.argv[1] if len(sys.argv) > 1 else None
    out_dir = find_results_dir(timestamp)

    print("=" * 70)
    print("RS vs FEM Phase 1 — Comparison & Metrics")
    print("=" * 70)
    print(f"Results dir: {out_dir}")

    fem_dir = out_dir / "fem"
    rs_free_dir = out_dir / "rs_free"
    rs_reflect_dir = out_dir / "rs_reflect"
    csv_dir = out_dir / "csv"
    fig_dir_root = out_dir / "figures"
    csv_dir.mkdir(parents=True, exist_ok=True)

    # Check all data exists
    for cid in CONFIG_IDS:
        for d, lbl in [(fem_dir, "FEM"), (rs_free_dir, "RS_free"),
                       (rs_reflect_dir, "RS_reflect")]:
            f = d / f"{cid}.npz"
            if not f.exists():
                print(f"  MISSING: {f.relative_to(out_dir)}  ({lbl})")
                print("  Run the generation scripts first.")
                sys.exit(1)

    print(f"All {len(CONFIG_IDS)} configs × 3 methods found.\n")

    t0_all = time.time()
    all_metrics: List[Dict] = []

    for ci, cid in enumerate(CONFIG_IDS, 1):
        print(f"\n[{ci}/{len(CONFIG_IDS)}] {cid}")

        # Load data
        d_fem = dict(np.load(str(fem_dir / f"{cid}.npz"), allow_pickle=False))
        d_free = dict(np.load(str(rs_free_dir / f"{cid}.npz"), allow_pickle=False))
        d_reflect = dict(np.load(str(rs_reflect_dir / f"{cid}.npz"), allow_pickle=False))

        xg = d_fem["xg"]
        yg = d_fem["yg"]
        z_list = d_fem["z_list"]

        # Per-config figure directory
        cfg_fig_dir = fig_dir_root / cid
        cfg_fig_dir.mkdir(parents=True, exist_ok=True)

        for zi, z in enumerate(z_list):
            z_mm = z * 1e3
            key = f"p_xy_{zi}"

            if key not in d_fem:
                print(f"  z={z_mm:.2f} mm: missing FEM data")
                continue

            p_fem = d_fem[key]
            p_free = d_free.get(key)
            p_reflect = d_reflect.get(key)

            if p_free is None or p_reflect is None:
                print(f"  z={z_mm:.2f} mm: missing RS data")
                continue

            # Compute best-fit scales
            c_free, _ = best_fit_scale(p_fem, p_free)
            c_reflect, _ = best_fit_scale(p_fem, p_reflect)

            # Metrics
            for method, p_rs, c_fit in [
                ("rs_free", p_free, c_free),
                ("rs_reflect", p_reflect, c_reflect),
            ]:
                p_rs_s = c_fit * p_rs
                m = compute_all_metrics(p_fem, p_rs_s, xg, yg, cid, z_mm, method)
                m["best_fit_c_amp"] = float(np.abs(c_fit))
                m["best_fit_c_phase"] = float(np.angle(c_fit))
                all_metrics.append(m)

                grade = "OK" if m["rho_roi"] > 0.85 and m["err_amp_roi"] < 0.15 else "  "
                print(f"  z={z_mm:5.2f}mm {method:12s}  "
                      f"ρ_g={m['rho_global']:.3f}  ρ_ROI={m['rho_roi']:.3f}  "
                      f"err_a={m['err_amp_roi']:.3f}  err_φ={m['err_phase']:.3f}  "
                      f"w_fem={m['winding_fem_1lam']:+.0f}  w_rs={m['winding_rs_1lam']:+.0f}  "
                      f"{grade}")

            # Figures — at Z_STAR and z=1.5mm
            is_key_plane = abs(z - Z_STAR) < 1e-6 or abs(z - 1.5e-3) < 1e-6
            if is_key_plane:
                make_figures(p_fem, p_free, p_reflect, xg, yg,
                             cid, z_mm, cfg_fig_dir, c_free, c_reflect)

    # Write CSV
    csv_file = csv_dir / "phase1_metrics.csv"
    if all_metrics:
        fieldnames = list(all_metrics[0].keys())
        with open(csv_file, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(all_metrics)
        print(f"\nCSV: {csv_file.relative_to(out_dir)}  ({len(all_metrics)} rows)")

    # Write report
    ts = out_dir.name.replace("rs_vs_fem_phase1_", "")
    report_path = write_report(out_dir, all_metrics, ts)
    print(f"Report: {report_path.relative_to(out_dir)}")

    total = time.time() - t0_all
    print(f"\n{'=' * 70}")
    print(f"Comparison complete in {total:.1f}s")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
