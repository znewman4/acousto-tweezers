#!/usr/bin/env python3
"""
RS vs FEM Phase 1A — Compare FEM truth vs RS truth
====================================================

Loads the FEM and RS .npz files produced by the Phase 1A truth test
and computes per-z-plane metrics, figures, CSV, and a go/no-go report.

Outputs
-------
results/rs_vs_fem_phase1A_truth_<TIMESTAMP>/
  csv/phase1A_metrics.csv
  figures/*.png
  report.md

Usage
-----
  python scripts/dev/rs_vs_fem_phase1A_truth_compare.py <TIMESTAMP>
  python scripts/dev/rs_vs_fem_phase1A_truth_compare.py --timestamp $TS
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from acoustweezers.experiments.farfield_petri_cuboid.presets import (
    CORRECTED_PRESET,
)

# ── Physical constants ────────────────────────────────────────────
WATER_C = 1484.0
F_HZ    = 2.0e6
LAM     = WATER_C / F_HZ
K_WATER = 2 * np.pi * F_HZ / WATER_C

_H_UNDER = CORRECTED_PRESET["H_under"]
_H_TOP   = CORRECTED_PRESET.get("H_top", 2.0085e-3)
Z_STAR   = _H_UNDER + _H_TOP / 2 + 0.25 * LAM

Z_PLANES = [1.0e-3, 1.5e-3, 2.0e-3, 2.5e-3, Z_STAR]

CONFIG_ID = "lg_l2_R1.0_w0.8"

# ── Defaults ──────────────────────────────────────────────────────
CORE_ROI_LAM = 2.0          # default ROI = 2 λ
PASS_RHO_ROI = 0.85         # go/no-go correlation threshold
PASS_ERR_AMP = 0.15         # go/no-go amplitude error threshold
WINDING_TARGET = 2           # expected topological charge
WINDING_TOL = 0.4            # winding must be within ±tol


# ==================================================================
# Metric helpers (matching Phase 1 definitions exactly)
# ==================================================================
def complex_correlation(a: np.ndarray, b: np.ndarray) -> complex:
    """Normalized complex inner product <a̅,b>/(||a||·||b||)."""
    na = np.sqrt(np.sum(np.abs(a) ** 2))
    nb = np.sqrt(np.sum(np.abs(b) ** 2))
    if na < 1e-30 or nb < 1e-30:
        return 0.0 + 0j
    return np.sum(a.conj() * b) / (na * nb)


def best_fit_scale(
    p_fem: np.ndarray, p_rs: np.ndarray
) -> Tuple[complex, float]:
    """c = <p_rs, p_fem>/<p_rs, p_rs>, returns (c, residual_relative)."""
    rs_f = p_rs.ravel()
    fem_f = p_fem.ravel()
    denom = np.sum(np.abs(rs_f) ** 2)
    if denom < 1e-30:
        return 0.0 + 0j, 1.0
    c = np.sum(rs_f.conj() * fem_f) / denom
    res = fem_f - c * rs_f
    rel = np.sqrt(np.sum(np.abs(res) ** 2)) / np.sqrt(
        np.sum(np.abs(fem_f) ** 2)
    )
    return c, float(rel)


def relative_amplitude_error(p_a: np.ndarray, p_b: np.ndarray) -> float:
    """|| |a| - |b| ||  /  || |a| ||."""
    aa = np.abs(p_a)
    ab = np.abs(p_b)
    n = np.sqrt(np.sum(aa ** 2))
    if n < 1e-30:
        return 1.0
    return float(np.sqrt(np.sum((aa - ab) ** 2)) / n)


def phase_error_rms(
    p_fem: np.ndarray, p_rs: np.ndarray, threshold: float = 0.1
) -> float:
    """RMS wrapped phase difference where |p_fem| > threshold·max(|p_fem|)."""
    af = np.abs(p_fem)
    mask = af > threshold * af.max()
    if mask.sum() < 10:
        return np.nan
    d = np.angle(p_fem[mask]) - np.angle(p_rs[mask])
    d = (d + np.pi) % (2 * np.pi) - np.pi
    return float(np.sqrt(np.mean(d ** 2)))


def winding_number(
    p: np.ndarray,
    xg: np.ndarray,
    yg: np.ndarray,
    r_lam: float,
    cx: float,
    cy: float,
) -> Tuple[float, float]:
    """
    Winding number and azimuthal amplitude variation on a ring
    of radius r_lam × λ centered at (cx, cy).

    Returns (winding_rounded, azvar).
    """
    from scipy.interpolate import RegularGridInterpolator

    r_m = r_lam * LAM
    n_pts = 360
    theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    ring_x = cx + r_m * np.cos(theta)
    ring_y = cy + r_m * np.sin(theta)

    interp_re = RegularGridInterpolator(
        (yg, xg), np.real(p), method="linear",
        bounds_error=False, fill_value=0.0,
    )
    interp_im = RegularGridInterpolator(
        (yg, xg), np.imag(p), method="linear",
        bounds_error=False, fill_value=0.0,
    )
    pts = np.column_stack([ring_y, ring_x])
    p_ring = interp_re(pts) + 1j * interp_im(pts)

    phi = np.angle(p_ring)
    dphi = np.diff(phi)
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
    winding = np.sum(dphi) / (2 * np.pi)

    amps = np.abs(p_ring)
    a_mean = amps.mean()
    azvar = float(np.std(amps) / a_mean) if a_mean > 1e-30 else 1.0

    return float(np.round(winding)), azvar


def winding_vs_radius(
    p: np.ndarray, xg: np.ndarray, yg: np.ndarray,
    cx: float, cy: float,
    r_lam_list: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute winding and azvar at a list of radii. Returns (wind, azvar)."""
    wind = np.zeros(len(r_lam_list))
    azv = np.zeros(len(r_lam_list))
    for i, rl in enumerate(r_lam_list):
        wind[i], azv[i] = winding_number(p, xg, yg, rl, cx, cy)
    return wind, azv


def make_roi_mask(
    xg: np.ndarray, yg: np.ndarray,
    cx: float, cy: float, radius_m: float,
) -> np.ndarray:
    """Boolean mask (ny, nx) for circular ROI."""
    XX, YY = np.meshgrid(xg, yg)
    return (XX - cx) ** 2 + (YY - cy) ** 2 <= radius_m ** 2


# ==================================================================
# Per-plane metric computation
# ==================================================================
def compute_plane_metrics(
    p_fem: np.ndarray,
    p_rs: np.ndarray,
    xg: np.ndarray,
    yg: np.ndarray,
    z_m: float,
    cx: float,
    cy: float,
    roi_r_m: float,
) -> Dict[str, Any]:
    """Full metric suite for one z-plane."""
    # Global raw
    rho_global = complex_correlation(p_fem.ravel(), p_rs.ravel())
    err_amp_global = relative_amplitude_error(p_fem, p_rs)

    # Best-fit scale + scaled metrics
    c_fit, res_norm = best_fit_scale(p_fem, p_rs)
    p_rs_s = c_fit * p_rs
    err_phase = phase_error_rms(p_fem, p_rs_s)

    # ROI metrics (on scaled RS)
    roi = make_roi_mask(xg, yg, cx, cy, roi_r_m)
    rho_roi = complex_correlation(p_fem[roi], p_rs_s[roi])
    err_amp_roi = relative_amplitude_error(p_fem[roi], p_rs_s[roi])

    # Winding at 0.5λ, 1.0λ, 1.5λ
    w_fem = {}
    w_rs = {}
    for rl in (0.5, 1.0, 1.5):
        tag = str(rl).replace(".", "p")
        wf, af = winding_number(p_fem, xg, yg, rl, cx, cy)
        wr, ar = winding_number(p_rs_s, xg, yg, rl, cx, cy)
        w_fem[tag] = (wf, af)
        w_rs[tag] = (wr, ar)

    # Vortex integrity gate: FEM at 1λ
    fem_intact = abs(w_fem["1p0"][0] - WINDING_TARGET) <= WINDING_TOL

    return {
        "z_mm": z_m * 1e3,
        "rho_global": float(np.abs(rho_global)),
        "rho_roi": float(np.abs(rho_roi)),
        "err_amp_global": err_amp_global,
        "err_amp_roi": err_amp_roi,
        "err_phase_deg": float(np.degrees(err_phase)) if not np.isnan(err_phase) else np.nan,
        "best_fit_amp": float(np.abs(c_fit)),
        "best_fit_phase_deg": float(np.degrees(np.angle(c_fit))),
        "res_norm": res_norm,
        # FEM winding / azvar
        "wf_0p5": w_fem["0p5"][0],
        "wf_1p0": w_fem["1p0"][0],
        "wf_1p5": w_fem["1p5"][0],
        "azv_fem_1p0": w_fem["1p0"][1],
        # RS winding / azvar
        "wr_0p5": w_rs["0p5"][0],
        "wr_1p0": w_rs["1p0"][0],
        "wr_1p5": w_rs["1p5"][0],
        "azv_rs_1p0": w_rs["1p0"][1],
        # Gate
        "fem_intact": fem_intact,
    }


# ==================================================================
# Figure generation
# ==================================================================
def _configure_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 150,
        "font.size": 8,
        "axes.titlesize": 9,
    })
    return plt


def make_amplitude_figure(
    p_fem, p_rs_s, xg, yg, z_mm, cx, cy, roi_r_m, fig_path,
):
    """3-panel amplitude comparison: FEM | RS (scaled) | difference."""
    plt = _configure_mpl()
    extent = [xg[0] * 1e3, xg[-1] * 1e3, yg[0] * 1e3, yg[-1] * 1e3]
    af = np.abs(p_fem)
    ar = np.abs(p_rs_s)
    vmax = max(af.max(), ar.max(), 1e-30)
    diff = af - ar

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    for ax, d, title in zip(
        axes, [af, ar, diff],
        ["|FEM|", "|RS scaled|", "|FEM| − |RS|"],
    ):
        if title.startswith("|FEM| −"):
            im = ax.imshow(d, extent=extent, origin="lower",
                           cmap="RdBu_r", vmin=-vmax * 0.3, vmax=vmax * 0.3)
        else:
            im = ax.imshow(d, extent=extent, origin="lower",
                           cmap="inferno", vmin=0, vmax=vmax)
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(title)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        # ROI circle
        circ = plt.Circle((cx * 1e3, cy * 1e3), roi_r_m * 1e3,
                           fc="none", ec="cyan", ls="--", lw=0.7)
        ax.add_patch(circ)

    fig.suptitle(f"{CONFIG_ID}  z = {z_mm:.2f} mm", fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(fig_path))
    plt.close(fig)


def make_phase_figure(
    p_fem, p_rs_s, xg, yg, z_mm, cx, cy, roi_r_m, fig_path,
):
    """2-panel phase comparison (masked where amplitude > 10% peak)."""
    plt = _configure_mpl()
    extent = [xg[0] * 1e3, xg[-1] * 1e3, yg[0] * 1e3, yg[-1] * 1e3]

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))
    for ax, p, label in zip(axes, [p_fem, p_rs_s], ["FEM", "RS scaled"]):
        a = np.abs(p)
        phi = np.angle(p)
        phi[a < 0.1 * a.max()] = np.nan
        im = ax.imshow(phi, extent=extent, origin="lower",
                       cmap="twilight", vmin=-np.pi, vmax=np.pi)
        fig.colorbar(im, ax=ax, shrink=0.8, label="phase [rad]")
        ax.set_title(f"arg({label})")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        circ = plt.Circle((cx * 1e3, cy * 1e3), roi_r_m * 1e3,
                           fc="none", ec="cyan", ls="--", lw=0.7)
        ax.add_patch(circ)

    fig.suptitle(f"{CONFIG_ID}  z = {z_mm:.2f} mm", fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(fig_path))
    plt.close(fig)


def make_radial_figure(
    p_fem, p_rs_s, xg, yg, z_mm, cx, cy, fig_path,
):
    """Azimuthally-averaged radial amplitude profile, FEM vs RS."""
    plt = _configure_mpl()
    XX, YY = np.meshgrid(xg, yg)
    R = np.sqrt((XX - cx) ** 2 + (YY - cy) ** 2)

    r_max = min(xg[-1] - cx, cx - xg[0], yg[-1] - cy, cy - yg[0])
    r_edges = np.linspace(0, r_max, 80)
    r_mid = 0.5 * (r_edges[:-1] + r_edges[1:])

    prof_fem = np.zeros(len(r_mid))
    prof_rs = np.zeros(len(r_mid))
    for i in range(len(r_mid)):
        mask = (R >= r_edges[i]) & (R < r_edges[i + 1])
        if mask.sum() > 0:
            prof_fem[i] = np.abs(p_fem[mask]).mean()
            prof_rs[i] = np.abs(p_rs_s[mask]).mean()

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(r_mid / LAM, prof_fem, "b-", lw=1.2, label="FEM")
    ax.plot(r_mid / LAM, prof_rs, "r--", lw=1.2, label="RS scaled")
    ax.set_xlabel("r / λ")
    ax.set_ylabel("⟨|p|⟩ (azimuthal avg)")
    ax.set_title(f"{CONFIG_ID}  z = {z_mm:.2f} mm")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(fig_path))
    plt.close(fig)


def make_winding_figure(
    all_planes_fem, all_planes_rs, xg, yg, cx, cy, z_mm_list, fig_path,
):
    """Winding number vs radius for all z-planes, FEM and RS."""
    plt = _configure_mpl()
    r_lam = np.linspace(0.3, 3.0, 30)

    fig, axes = plt.subplots(1, len(z_mm_list), figsize=(3.3 * len(z_mm_list), 3))
    if len(z_mm_list) == 1:
        axes = [axes]

    for ax, (p_fem, p_rs_s, zmm) in zip(axes, zip(all_planes_fem, all_planes_rs, z_mm_list)):
        wf, _ = winding_vs_radius(p_fem, xg, yg, cx, cy, r_lam)
        wr, _ = winding_vs_radius(p_rs_s, xg, yg, cx, cy, r_lam)
        ax.plot(r_lam, wf, "b-", lw=1.2, label="FEM")
        ax.plot(r_lam, wr, "r--", lw=1.2, label="RS")
        ax.axhline(WINDING_TARGET, color="green", ls=":", lw=0.8, label=f"ℓ={WINDING_TARGET}")
        ax.set_title(f"z = {zmm:.2f} mm")
        ax.set_xlabel("r / λ")
        ax.set_ylabel("winding")
        ax.set_ylim(WINDING_TARGET - 2, WINDING_TARGET + 2)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{CONFIG_ID} — Winding number vs radius", fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(fig_path))
    plt.close(fig)


# ==================================================================
# CSV / Report writing
# ==================================================================
CSV_COLUMNS = [
    "z_mm", "rho_global", "rho_roi", "err_amp_global", "err_amp_roi",
    "err_phase_deg", "best_fit_amp", "best_fit_phase_deg", "res_norm",
    "wf_0p5", "wf_1p0", "wf_1p5", "azv_fem_1p0",
    "wr_0p5", "wr_1p0", "wr_1p5", "azv_rs_1p0",
    "fem_intact",
]


def write_csv(csv_path: Path, rows: List[Dict]):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  CSV: {csv_path.relative_to(PROJECT_ROOT)}")


def write_report(out_dir: Path, rows: List[Dict], timestamp: str, roi_lam: float):
    """Write Phase 1A report.md with go/no-go assessment."""
    rpt = out_dir / "report.md"

    # Determine pass / fail
    any_pass = False
    n_intact = 0
    best_rho = 0.0
    best_z = None
    for r in rows:
        if r["fem_intact"]:
            n_intact += 1
            if r["rho_roi"] > best_rho:
                best_rho = r["rho_roi"]
                best_z = r["z_mm"]
            if r["rho_roi"] >= PASS_RHO_ROI and r["err_amp_roi"] <= PASS_ERR_AMP:
                any_pass = True

    verdict = "**PASS** ✅" if any_pass else "**FAIL** ❌"

    lines = [
        f"# RS vs FEM Phase 1A — Truth Test Report",
        f"",
        f"**Timestamp**: `{timestamp}`",
        f"",
        f"**Config**: `{CONFIG_ID}` — LG ℓ=2, R=1.0 mm, w=0.8 mm, cosine_taper",
        f"",
        f"## Setup",
        f"",
        f"| Parameter | Value |",
        f"|-----------|-------|",
        f"| Domain | homogeneous water (matched-impedance top, no standing wave) |",
        f"| ROI radius | {roi_lam:.1f} λ = {roi_lam * LAM * 1e3:.3f} mm |",
        f"| Pass criteria | ρ_ROI > {PASS_RHO_ROI}, err_amp_ROI < {PASS_ERR_AMP} |",
        f"| FEM integrity gate | winding at 1λ = {WINDING_TARGET} ± {WINDING_TOL} |",
        f"",
        f"## Results",
        f"",
        f"| z [mm] | ρ_global | ρ_ROI | err_amp_ROI | err_φ [°] | winding FEM | winding RS | intact? |",
        f"|--------|----------|-------|-------------|-----------|-------------|------------|---------|",
    ]
    for r in rows:
        intact_str = "✅" if r["fem_intact"] else "❌"
        phase_str = f'{r["err_phase_deg"]:.1f}' if not np.isnan(r.get("err_phase_deg", np.nan)) else "—"
        lines.append(
            f'| {r["z_mm"]:.2f} '
            f'| {r["rho_global"]:.4f} '
            f'| {r["rho_roi"]:.4f} '
            f'| {r["err_amp_roi"]:.4f} '
            f'| {phase_str} '
            f'| {r["wf_1p0"]:.0f} '
            f'| {r["wr_1p0"]:.0f} '
            f'| {intact_str} |'
        )

    lines += [
        f"",
        f"## Go / No-Go",
        f"",
        f"**Verdict: {verdict}**",
        f"",
        f"- FEM-intact planes: **{n_intact}/{len(rows)}**",
        f"- Best ρ_ROI (intact planes): **{best_rho:.4f}** at z = {best_z} mm"
        if best_z else "- No intact planes found.",
        f"",
    ]

    if any_pass:
        lines += [
            f"### Interpretation",
            f"",
            f"RS angular-spectrum propagation in unbounded water agrees with the",
            f"high-resolution FEM truth test at the pass threshold.  The RS model",
            f"is **validated as an actuator propagator** for LG ℓ=2 beams.",
            f"",
            f"Remaining FEM–RS discrepancy in the Phase 1 results is attributable to",
            f"slab/cavity/reflection effects, not RS model error.",
        ]
    else:
        lines += [
            f"### Interpretation",
            f"",
            f"The truth test did not pass.  Possible causes:",
            f"",
            f"1. FEM resolution still insufficient (try higher EPL or smaller domain)",
            f"2. Angular-spectrum wraparound artifacts (try larger drive grid)",
            f"3. Apodization or source-plane mismatch between FEM and RS",
            f"",
            f"Review the figures for diagnostic clues.",
        ]

    lines += [
        f"",
        f"---",
        f"*Generated by `rs_vs_fem_phase1A_truth_compare.py`*",
    ]

    with open(rpt, "w") as f:
        f.write("\n".join(lines))
    print(f"  Report: {rpt.relative_to(PROJECT_ROOT)}")


# ==================================================================
# Main
# ==================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Phase 1A — compare FEM truth vs RS truth"
    )
    p.add_argument("timestamp", nargs="?", default=None, help="Timestamp")
    p.add_argument("--timestamp", dest="ts_flag", default=None)
    p.add_argument("--core_roi_lam", type=float, default=CORE_ROI_LAM)
    return p.parse_args()


def find_result_dir(ts: str | None) -> Path:
    """Locate the result directory by timestamp or most recent."""
    results = PROJECT_ROOT / "results"
    if ts:
        d = results / f"rs_vs_fem_phase1A_truth_{ts}"
        if d.is_dir():
            return d
        raise FileNotFoundError(f"No Phase 1A results at {d}")
    # Glob for most recent
    dirs = sorted(results.glob("rs_vs_fem_phase1A_truth_*"))
    if not dirs:
        raise FileNotFoundError("No rs_vs_fem_phase1A_truth_* directory found")
    return dirs[-1]


def main():
    args = parse_args()
    ts = args.timestamp or args.ts_flag
    roi_lam = args.core_roi_lam

    out_dir = find_result_dir(ts)
    ts_actual = out_dir.name.replace("rs_vs_fem_phase1A_truth_", "")

    fem_npz = out_dir / "fem_truth" / f"{CONFIG_ID}.npz"
    rs_npz = out_dir / "rs_truth" / f"{CONFIG_ID}.npz"

    if not fem_npz.exists():
        print(f"ERROR: FEM NPZ not found: {fem_npz}")
        sys.exit(1)
    if not rs_npz.exists():
        print(f"ERROR: RS NPZ not found: {rs_npz}")
        sys.exit(1)

    print("=" * 70)
    print("RS vs FEM Phase 1A — Compare FEM truth vs RS truth")
    print("=" * 70)
    print(f"Results dir: {out_dir}")
    print(f"FEM NPZ    : {fem_npz.name}")
    print(f"RS NPZ     : {rs_npz.name}")
    print(f"ROI radius : {roi_lam:.1f} λ = {roi_lam * LAM * 1e3:.3f} mm")
    print()

    # ── Load data ─────────────────────────────────────────────────
    fem = dict(np.load(str(fem_npz), allow_pickle=False))
    rs = dict(np.load(str(rs_npz), allow_pickle=False))

    xg_fem = fem["xg"]
    yg_fem = fem["yg"]
    xg_rs = rs["xg"]
    yg_rs = rs["yg"]

    # Validate grid compatibility
    if not (np.allclose(xg_fem, xg_rs, atol=1e-8) and
            np.allclose(yg_fem, yg_rs, atol=1e-8)):
        print("WARNING: FEM and RS grids do not match exactly!")
        print(f"  FEM grid: x=[{xg_fem[0]*1e3:.4f}, {xg_fem[-1]*1e3:.4f}] mm, n={len(xg_fem)}")
        print(f"  RS  grid: x=[{xg_rs[0]*1e3:.4f}, {xg_rs[-1]*1e3:.4f}] mm, n={len(xg_rs)}")
        print("  Will interpolate RS onto FEM grid.")

        # Interpolate RS onto FEM grid
        from scipy.interpolate import RegularGridInterpolator
        for zi in range(5):
            key = f"p_xy_{zi}"
            if key not in rs:
                continue
            interp_re = RegularGridInterpolator(
                (yg_rs, xg_rs), np.real(rs[key]),
                method="linear", bounds_error=False, fill_value=0.0,
            )
            interp_im = RegularGridInterpolator(
                (yg_rs, xg_rs), np.imag(rs[key]),
                method="linear", bounds_error=False, fill_value=0.0,
            )
            XX, YY = np.meshgrid(xg_fem, yg_fem)
            pts = np.column_stack([YY.ravel(), XX.ravel()])
            rs[key] = (interp_re(pts) + 1j * interp_im(pts)).reshape(XX.shape)
        xg = xg_fem
        yg = yg_fem
    else:
        xg = xg_fem
        yg = yg_fem

    cx = 0.5 * (xg[0] + xg[-1])
    cy = 0.5 * (yg[0] + yg[-1])
    roi_r_m = roi_lam * LAM

    z_list = fem.get("z_list", np.array(Z_PLANES))

    # ── Per-plane metrics ─────────────────────────────────────────
    fig_dir = out_dir / "figures"
    csv_dir = out_dir / "csv"
    fig_dir.mkdir(exist_ok=True)
    csv_dir.mkdir(exist_ok=True)

    all_metrics: List[Dict] = []
    all_fem_planes: List[np.ndarray] = []
    all_rs_planes: List[np.ndarray] = []

    for zi in range(len(z_list)):
        z_m = float(z_list[zi])
        z_mm = z_m * 1e3
        key = f"p_xy_{zi}"

        if key not in fem or key not in rs:
            print(f"  z={z_mm:.2f} mm — MISSING data, skipped")
            continue

        p_fem = fem[key]
        p_rs = rs[key]

        print(f"  z = {z_mm:.2f} mm:")

        # Metrics
        m = compute_plane_metrics(p_fem, p_rs, xg, yg, z_m, cx, cy, roi_r_m)
        all_metrics.append(m)

        print(f"    ρ_global={m['rho_global']:.4f}  ρ_ROI={m['rho_roi']:.4f}"
              f"  err_amp_ROI={m['err_amp_roi']:.4f}"
              f"  winding FEM={m['wf_1p0']:.0f}  RS={m['wr_1p0']:.0f}"
              f"  intact={'✓' if m['fem_intact'] else '✗'}")

        # Scale RS for figures
        c_fit, _ = best_fit_scale(p_fem, p_rs)
        p_rs_s = c_fit * p_rs

        all_fem_planes.append(p_fem)
        all_rs_planes.append(p_rs_s)

        # Figures
        make_amplitude_figure(
            p_fem, p_rs_s, xg, yg, z_mm, cx, cy, roi_r_m,
            fig_dir / f"{CONFIG_ID}_z{zi}_amplitude.png",
        )
        make_phase_figure(
            p_fem, p_rs_s, xg, yg, z_mm, cx, cy, roi_r_m,
            fig_dir / f"{CONFIG_ID}_z{zi}_phase.png",
        )
        make_radial_figure(
            p_fem, p_rs_s, xg, yg, z_mm, cx, cy,
            fig_dir / f"{CONFIG_ID}_z{zi}_radial.png",
        )

    # Winding overview figure
    if all_fem_planes:
        z_mm_list = [m["z_mm"] for m in all_metrics]
        make_winding_figure(
            all_fem_planes, all_rs_planes,
            xg, yg, cx, cy, z_mm_list,
            fig_dir / f"{CONFIG_ID}_winding_overview.png",
        )

    # ── CSV and report ────────────────────────────────────────────
    write_csv(csv_dir / "phase1A_metrics.csv", all_metrics)
    write_report(out_dir, all_metrics, ts_actual, roi_lam)

    # ── Summary ───────────────────────────────────────────────────
    n_pass = sum(
        1 for m in all_metrics
        if m["fem_intact"]
        and m["rho_roi"] >= PASS_RHO_ROI
        and m["err_amp_roi"] <= PASS_ERR_AMP
    )
    n_intact = sum(1 for m in all_metrics if m["fem_intact"])
    print()
    print(f"  FEM-intact planes : {n_intact}/{len(all_metrics)}")
    print(f"  Passing planes    : {n_pass}/{len(all_metrics)}")
    verdict = "PASS ✅" if n_pass > 0 else "FAIL ❌"
    print(f"  Verdict           : {verdict}")
    print(f"\n{'=' * 70}")
    print("Phase 1A comparison complete.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
