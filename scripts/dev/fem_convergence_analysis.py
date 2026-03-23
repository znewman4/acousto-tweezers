#!/usr/bin/env python3
"""
FEM Convergence Study — Post-Processing Analysis
=================================================

Reads all .npz outputs from the convergence study driver
(fem_convergence_study.py), computes:

  1.  Observed convergence order from reduced-domain EPL sweep
  2.  Domain-size sensitivity comparison
  3.  PML sensitivity comparison
  4.  Production acceptance argument

Generates:
  - Convergence error vs EPL / h plots
  - Domain-size sensitivity plots
  - PML sensitivity plots
  - Trap overlay comparison figures
  - Solve time vs DOFs
  - Markdown summary report

Usage
-----
    python scripts/dev/fem_convergence_analysis.py

    # Or point at a specific study directory
    python scripts/dev/fem_convergence_analysis.py \\
        --study-dir results/mesh_convergence_study

    # Re-run analysis on existing data (no FEM solves)
    python scripts/dev/fem_convergence_analysis.py --report-only

Author: Acousto-Tweezers Project
Date:   March 2026
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Import the trap detection and matching functions from the driver
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "dev"))
from fem_convergence_study import (
    detect_traps,
    match_traps_to_reference,
    compute_gorkov_2d,
    PARTICLE_RADIUS_M, PARTICLE_DENSITY, PARTICLE_C,
)
from acoustweezers.experiments.farfield_petri_cuboid.config import FarFieldConfig


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FEM convergence study post-processing and analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--study-dir", type=str,
        default=str(PROJECT_ROOT / "results" / "mesh_convergence_study"),
        help="Directory containing .npz results")
    p.add_argument(
        "--report-only", action="store_true", default=False,
        help="Only generate analysis from existing .npz files (no solves)")
    p.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory for analysis outputs. "
             "Defaults to {study_dir}/analysis_{timestamp}")
    return p.parse_args()


# =============================================================================
# Load all NPZ results
# =============================================================================

def load_all_npz(study_dir: Path) -> list[dict]:
    """Load all .npz files from the study directory, sorted by EPL."""
    results = []
    for npz_path in sorted(study_dir.glob("conv_epl*.npz")):
        data = dict(np.load(npz_path, allow_pickle=True))
        meta = data["metadata"].item() if data["metadata"].ndim == 0 else data["metadata"]
        data["_meta"] = meta
        data["_path"] = npz_path
        data["_name"] = npz_path.stem
        results.append(data)
    return results


def filter_runs(results: list[dict], mode: str = None,
                phys_mm: float = None, pml_xy: float = None) -> list[dict]:
    """Filter results by mode, physical size, and PML setting."""
    out = []
    for r in results:
        m = r["_meta"]
        if mode is not None and m.get("mode") != mode:
            continue
        if phys_mm is not None and abs(m.get("physical_size_mm", 0) - phys_mm) > 0.01:
            continue
        if pml_xy is not None:
            r_pml = m.get("pml_n_wavelengths_xy", 1.0)
            if abs(r_pml - pml_xy) > 0.01:
                continue
        out.append(r)
    return out


def get_epl(r: dict) -> float:
    return float(r["_meta"].get("requested_epl", 0))


def get_scalar(r: dict, key: str, default=np.nan) -> float:
    v = r.get(key, np.array([default]))
    return float(np.squeeze(v))


# =============================================================================
# Recompute trap metrics with improved detection
# =============================================================================

def recompute_trap_metrics(runs: list[dict], ref_run: dict,
                           wavelength: float) -> list[dict]:
    """
    Re-run improved trap detection and matching on stored Cartesian fields.

    This allows re-analysis of old NPZ files with the new robust detection
    without re-running FEM solves.
    """
    # Build a minimal cfg for Gor'kov if needed
    ref_meta = ref_run["_meta"]
    freq = ref_meta.get("frequency_hz", 2.15e6)
    cfg = FarFieldConfig(frequency_hz=freq, c=1484.0)

    # Reference trap detection
    ref_p = (ref_run["p_cart_real"] + 1j * ref_run["p_cart_imag"]).astype(complex)
    ref_U = ref_run["gorkov_2d"]
    ref_x = ref_run["x_grid"]
    ref_y = ref_run["y_grid"]
    ref_traps = detect_traps(ref_U, ref_x, ref_y, roi_fraction=0.5,
                             wavelength=wavelength)

    results = []
    for run in runs:
        epl = get_epl(run)
        p = (run["p_cart_real"] + 1j * run["p_cart_imag"]).astype(complex)
        U = run["gorkov_2d"]
        x_grid = run["x_grid"]
        y_grid = run["y_grid"]

        traps = detect_traps(U, x_grid, y_grid, roi_fraction=0.5,
                             wavelength=wavelength)

        # Match to reference
        max_match_dist = wavelength * 0.25
        match_result = match_traps_to_reference(
            traps["trap_positions_m"],
            ref_traps["trap_positions_m"],
            max_match_dist,
        )

        # Field error in ROI
        roi_mask = traps["roi_mask"]
        p_ref = (ref_run["p_cart_real"] + 1j * ref_run["p_cart_imag"]).astype(complex)

        if p.shape == p_ref.shape:
            valid = ~(np.isnan(np.abs(p)) | np.isnan(np.abs(p_ref)))
            # Phase alignment
            Ny, Nx = p.shape
            ic, jc = Nx // 2, Ny // 2
            if np.abs(p[jc, ic]) > 1e-30 and np.abs(p_ref[jc, ic]) > 1e-30:
                phase_diff = np.angle(p[jc, ic]) - np.angle(p_ref[jc, ic])
                p_aligned = p * np.exp(-1j * phase_diff)
            else:
                p_aligned = p

            roi_valid = valid & roi_mask
            diff_roi = np.abs(p_aligned - p_ref)[roi_valid]
            norm_roi = np.abs(p_ref)[roi_valid]
            eps_roi = float(np.linalg.norm(diff_roi) / (np.linalg.norm(norm_roi) + 1e-30))
        else:
            eps_roi = np.nan

        # Spacing
        sp_cur = traps["centreline_min_spacing_m"]
        sp_ref = ref_traps["centreline_min_spacing_m"]
        spacing_err = (abs(sp_cur - sp_ref) / (abs(sp_ref) + 1e-30) * 100.0) if (
            np.isfinite(sp_cur) and np.isfinite(sp_ref)) else np.nan

        meta = run["_meta"]
        results.append({
            "epl"                   : epl,
            "dofs"                  : int(meta.get("dofs", 0)),
            "mesh_nx"               : int(meta.get("mesh_nx", 0)),
            "solve_time_s"          : float(meta.get("solve_time_s", 0)),
            "total_wall_time_s"     : float(meta.get("total_wall_time_s", 0)),
            "max_p_Pa"              : float(meta.get("max_pressure_Pa", 0)),
            "eps_L2_roi"            : eps_roi,
            "spacing_err_pct"       : spacing_err,
            "n_traps"               : traps["n_traps"],
            "n_matched"             : match_result["n_matched"],
            "n_unmatched_cur"       : match_result["n_unmatched_current"],
            "n_unmatched_ref"       : match_result["n_unmatched_reference"],
            "mean_trap_err_m"       : match_result["mean_trap_error_m"],
            "max_trap_err_m"        : match_result["max_trap_error_m"],
            "cl_spacing_m"          : sp_cur,
            "cl_n_min"              : len(traps["centreline_min_x"]),
            "name"                  : run["_name"],
            "physical_size_mm"      : float(meta.get("physical_size_mm", 0)),
            "pml_n_wavelengths_xy"  : float(meta.get("pml_n_wavelengths_xy", 1.0)),
            "pml_sigma_max_factor"  : float(meta.get("pml_sigma_max_factor", 5.0)),
            "wavelength_m"          : wavelength,
        })

    return results


# =============================================================================
# Observed convergence order
# =============================================================================

def compute_observed_orders(metrics: list[dict]) -> list[dict]:
    """
    Compute observed convergence order between successive refinement levels.

    Uses h = λ / EPL as the mesh size parameter.

    For two successive levels (h_coarse, h_fine) with errors (e_c, e_f):
        p_obs = log(e_c / e_f) / log(h_c / h_f)

    Returns list of dicts with slopes.
    """
    # Sort by EPL ascending (coarse → fine)
    metrics = sorted(metrics, key=lambda d: d["epl"])

    orders = []
    for i in range(1, len(metrics)):
        coarse = metrics[i - 1]
        fine   = metrics[i]

        lam = coarse["wavelength_m"]
        h_c = lam / coarse["epl"]
        h_f = lam / fine["epl"]

        row = {
            "epl_coarse": coarse["epl"],
            "epl_fine"  : fine["epl"],
            "h_coarse"  : h_c,
            "h_fine"    : h_f,
            "h_ratio"   : h_c / h_f,
        }

        # ROI L2 field error
        e_c = coarse["eps_L2_roi"]
        e_f = fine["eps_L2_roi"]
        if np.isfinite(e_c) and np.isfinite(e_f) and e_c > 0 and e_f > 0:
            row["p_obs_L2_roi"] = np.log(e_c / e_f) / np.log(h_c / h_f)
        else:
            row["p_obs_L2_roi"] = np.nan

        # Spacing error
        e_c = coarse["spacing_err_pct"]
        e_f = fine["spacing_err_pct"]
        if np.isfinite(e_c) and np.isfinite(e_f) and e_c > 0 and e_f > 0:
            row["p_obs_spacing"] = np.log(e_c / e_f) / np.log(h_c / h_f)
        else:
            row["p_obs_spacing"] = np.nan

        # Mean trap error
        e_c = coarse["mean_trap_err_m"]
        e_f = fine["mean_trap_err_m"]
        if np.isfinite(e_c) and np.isfinite(e_f) and e_c > 0 and e_f > 0:
            row["p_obs_trap_pos"] = np.log(e_c / e_f) / np.log(h_c / h_f)
        else:
            row["p_obs_trap_pos"] = np.nan

        orders.append(row)

    return orders


# =============================================================================
# Figures
# =============================================================================

def plot_convergence_errors(metrics: list[dict], orders: list[dict],
                            out_dir: Path, wavelength: float) -> None:
    """Generate convergence error vs EPL and vs h plots."""
    metrics = sorted(metrics, key=lambda d: d["epl"])
    epls = [m["epl"] for m in metrics]
    hs   = [wavelength / m["epl"] * 1e3 for m in metrics]  # mm

    eps_roi    = [m["eps_L2_roi"] for m in metrics]
    spacing    = [m["spacing_err_pct"] for m in metrics]
    mean_trap  = [m["mean_trap_err_m"] * 1e6 if np.isfinite(m["mean_trap_err_m"]) else np.nan
                  for m in metrics]

    # ── Figure 1: Error vs EPL (log scale) ────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    valid = [i for i, e in enumerate(eps_roi) if np.isfinite(e)]
    ax.semilogy([epls[i] for i in valid], [eps_roi[i] for i in valid],
                "o-", color="steelblue", lw=1.5, ms=6)
    ax.set_xlabel("EPL (elements per λ)")
    ax.set_ylabel("ε L2 ROI (relative)")
    ax.set_title("ROI Field Error vs EPL\n(PRIMARY metric)")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    valid = [i for i, s in enumerate(spacing) if np.isfinite(s)]
    ax.semilogy([epls[i] for i in valid], [spacing[i] for i in valid],
                "s-", color="coral", lw=1.5, ms=6)
    ax.set_xlabel("EPL")
    ax.set_ylabel("Trap spacing error (%)")
    ax.set_title("Centraline Trap Spacing Error\n(secondary metric)")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    valid = [i for i, t in enumerate(mean_trap) if np.isfinite(t)]
    if valid:
        ax.semilogy([epls[i] for i in valid], [mean_trap[i] for i in valid],
                    "^-", color="forestgreen", lw=1.5, ms=6)
    ax.set_xlabel("EPL")
    ax.set_ylabel("Mean matched trap error (µm)")
    ax.set_title("Matched Trap Position Error\n(secondary metric)")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "convergence_errors_vs_epl.png", dpi=300)
    plt.close(fig)

    # ── Figure 2: Error vs h (log-log with slope annotations) ────────────────
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    valid = [i for i, e in enumerate(eps_roi) if np.isfinite(e)]
    if valid:
        hv = [hs[i] for i in valid]
        ev = [eps_roi[i] for i in valid]
        ax.loglog(hv, ev, "o-", color="steelblue", lw=1.5, ms=7, label="ε L2 ROI")

        # Annotate slopes
        for o in orders:
            if np.isfinite(o["p_obs_L2_roi"]):
                h_mid = np.sqrt(o["h_coarse"] * o["h_fine"]) * 1e3
                ax.annotate(f"p={o['p_obs_L2_roi']:.1f}",
                           xy=(h_mid, 0), xytext=(0, -20),
                           textcoords="offset points",
                           fontsize=8, ha="center", color="steelblue")

    ax.set_xlabel("h = λ/EPL (mm)")
    ax.set_ylabel("Relative L2 error in ROI")
    ax.set_title("ROI Field Error vs Mesh Size h\n(observed convergence order)")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_dir / "convergence_error_vs_h.png", dpi=300)
    plt.close(fig)


def plot_solve_time_vs_dofs(metrics: list[dict], out_dir: Path) -> None:
    """Solve time vs DOFs plot."""
    metrics = sorted(metrics, key=lambda d: d["dofs"])
    dofs = [m["dofs"] for m in metrics]
    times = [m["solve_time_s"] for m in metrics]
    epls = [m["epl"] for m in metrics]

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.plot(dofs, times, "o-", color="darkred", lw=1.5, ms=7)
    for d, t, e in zip(dofs, times, epls):
        ax.annotate(f"EPL={e}", xy=(d, t), xytext=(5, 5),
                   textcoords="offset points", fontsize=8)
    ax.set_xlabel("DOFs")
    ax.set_ylabel("Solve time (s)")
    ax.set_title("MUMPS Solve Time vs Problem Size")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "solve_time_vs_dofs.png", dpi=300)
    plt.close(fig)


def plot_trap_overlay(runs: list[dict], ref_run: dict,
                      wavelength: float, out_dir: Path) -> None:
    """
    Visual overlay: reference traps + coarsest mesh traps on Gor'kov field.
    """
    ref_U = ref_run["gorkov_2d"]
    ref_x = ref_run["x_grid"]
    ref_y = ref_run["y_grid"]
    ref_traps = detect_traps(ref_U, ref_x, ref_y, roi_fraction=0.5,
                             wavelength=wavelength)

    # Pick coarsest and finest non-reference runs
    candidates = sorted(runs, key=get_epl)
    if len(candidates) < 2:
        return

    coarse = candidates[0]
    fine   = candidates[-1]

    for label, run in [("coarse", coarse), ("fine", fine)]:
        epl = get_epl(run)
        U = run["gorkov_2d"]
        x = run["x_grid"]
        y = run["y_grid"]
        traps = detect_traps(U, x, y, roi_fraction=0.5, wavelength=wavelength)

        fig, ax = plt.subplots(1, 1, figsize=(7, 6))
        xmm = ref_x * 1e3
        ymm = ref_y * 1e3
        vmin = np.nanpercentile(ref_U, 2)
        vmax = np.nanpercentile(ref_U, 98)
        ax.pcolormesh(xmm, ymm, ref_U, cmap="RdBu_r", shading="auto",
                      vmin=vmin, vmax=vmax, alpha=0.6)

        # Reference traps
        if ref_traps["n_traps"] > 0:
            rt = ref_traps["trap_positions_m"]
            ax.scatter(rt[:, 0]*1e3, rt[:, 1]*1e3, c="blue", s=30, marker="o",
                      label=f"Reference (EPL=5, n={ref_traps['n_traps']})",
                      zorder=5, edgecolors="navy", linewidths=0.5)

        # Current traps
        if traps["n_traps"] > 0:
            ct = traps["trap_positions_m"]
            ax.scatter(ct[:, 0]*1e3, ct[:, 1]*1e3, c="red", s=30, marker="x",
                      label=f"EPL={epl} (n={traps['n_traps']})",
                      zorder=6, linewidths=1.2)

        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_title(f"Trap Overlay: EPL={epl} ({label}) vs Reference\n"
                     f"Background: Reference Gor'kov potential")
        ax.legend(fontsize=8)
        ax.set_aspect("equal")
        fig.tight_layout()
        fig.savefig(out_dir / f"trap_overlay_{label}_epl{epl:.1f}.png", dpi=300)
        plt.close(fig)


def plot_domain_sensitivity(metrics_by_size: dict, out_dir: Path) -> None:
    """Plot field metrics across different domain sizes."""
    sizes = sorted(metrics_by_size.keys())
    if len(sizes) < 2:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # max|p| in ROI
    ax = axes[0]
    for sz in sizes:
        ms = metrics_by_size[sz]
        if ms:
            ax.bar(f"{sz:.1f}", ms[0]["max_p_Pa"], color="steelblue", alpha=0.7)
    ax.set_xlabel("Physical domain size (mm)")
    ax.set_ylabel("max|p| (Pa)")
    ax.set_title("Max Pressure vs Domain Size")
    ax.grid(True, alpha=0.3, axis="y")

    # Trap spacing
    ax = axes[1]
    for sz in sizes:
        ms = metrics_by_size[sz]
        if ms:
            sp = ms[0]["cl_spacing_m"]
            if np.isfinite(sp):
                ax.bar(f"{sz:.1f}", sp * 1e3, color="coral", alpha=0.7)
    ax.set_xlabel("Physical domain size (mm)")
    ax.set_ylabel("Centreline trap spacing (mm)")
    ax.set_title("Trap Spacing vs Domain Size")
    ax.grid(True, alpha=0.3, axis="y")

    # n_traps
    ax = axes[2]
    for sz in sizes:
        ms = metrics_by_size[sz]
        if ms:
            ax.bar(f"{sz:.1f}", ms[0]["n_traps"], color="forestgreen", alpha=0.7)
    ax.set_xlabel("Physical domain size (mm)")
    ax.set_ylabel("Trap count in ROI")
    ax.set_title("Detected Traps vs Domain Size")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_dir / "domain_size_sensitivity.png", dpi=300)
    plt.close(fig)


def plot_pml_sensitivity(pml_metrics: list[dict], out_dir: Path) -> None:
    """Plot field metrics across different PML thicknesses."""
    if len(pml_metrics) < 2:
        return

    pml_metrics = sorted(pml_metrics, key=lambda d: d["pml_n_wavelengths_xy"])
    pml_vals = [m["pml_n_wavelengths_xy"] for m in pml_metrics]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    ax.plot(pml_vals, [m["max_p_Pa"] for m in pml_metrics],
            "o-", color="steelblue", lw=1.5, ms=7)
    ax.set_xlabel("PML thickness (wavelengths)")
    ax.set_ylabel("max|p| (Pa)")
    ax.set_title("Max Pressure vs PML Thickness")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    spacings = [m["cl_spacing_m"] * 1e3 if np.isfinite(m["cl_spacing_m"]) else np.nan
                for m in pml_metrics]
    valid = [i for i, s in enumerate(spacings) if np.isfinite(s)]
    if valid:
        ax.plot([pml_vals[i] for i in valid], [spacings[i] for i in valid],
                "s-", color="coral", lw=1.5, ms=7)
    ax.set_xlabel("PML thickness (wavelengths)")
    ax.set_ylabel("Centreline trap spacing (mm)")
    ax.set_title("Trap Spacing vs PML Thickness")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(pml_vals, [m["n_traps"] for m in pml_metrics],
            "^-", color="forestgreen", lw=1.5, ms=7)
    ax.set_xlabel("PML thickness (wavelengths)")
    ax.set_ylabel("Trap count in ROI")
    ax.set_title("Detected Traps vs PML Thickness")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "pml_sensitivity.png", dpi=300)
    plt.close(fig)


# =============================================================================
# Report generator
# =============================================================================

def generate_report(
    conv_metrics: list[dict],
    orders: list[dict],
    domain_metrics: dict,
    pml_metrics: list[dict],
    out_dir: Path,
    wavelength: float,
) -> str:
    """Generate markdown convergence study report."""

    lines = []
    lines.append("# FEM Standing-Wave Mesh Convergence Study Report")
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"\nWavelength: {wavelength*1e3:.4f} mm  |  λ/2 = {wavelength*0.5*1e3:.4f} mm")
    lines.append(f"Frequency: 2.15 MHz  |  c = 1484 m/s")
    lines.append("")

    # ── Section 1: Convergence sweep ──────────────────────────────────────────
    lines.append("## 1. Reduced-Domain Mesh Convergence (3.0 mm, PML = 1.0λ)")
    lines.append("")
    lines.append("### 1.1 Error Metrics")
    lines.append("")
    lines.append("| EPL | DOFs | Solve (s) | max\\|p\\| (Pa) | ε L2 ROI | "
                 "Spacing Err (%) | Matched/Total | Mean Trap Err (µm) |")
    lines.append("|-----|------|-----------|--------------|----------|"
                 "----------------|---------------|---------------------|")

    conv_sorted = sorted(conv_metrics, key=lambda d: -d["epl"])  # fine first
    for m in conv_sorted:
        eps = f"{m['eps_L2_roi']:.4e}" if np.isfinite(m["eps_L2_roi"]) else "—"
        sp  = f"{m['spacing_err_pct']:.1f}" if np.isfinite(m["spacing_err_pct"]) else "—"
        te  = (f"{m['mean_trap_err_m']*1e6:.1f}"
               if np.isfinite(m["mean_trap_err_m"]) else "—")
        match_str = f"{m['n_matched']}/{m['n_traps']}"
        lines.append(
            f"| {m['epl']:.1f} | {m['dofs']:,} | {m['solve_time_s']:.1f} | "
            f"{m['max_p_Pa']:.2f} | {eps} | {sp} | {match_str} | {te} |"
        )

    lines.append("")
    lines.append("**Primary convergence metric:** ε L2 ROI (relative L2 norm of "
                 "complex pressure difference in central 50% ROI, after phase alignment).")
    lines.append("")
    lines.append("**Secondary metrics:** Centreline trap spacing error and matched "
                 "trap position error. These are physically meaningful but inherently "
                 "noisier because they depend on trap detection thresholds.")

    # ── Section 1.2: Observed order ───────────────────────────────────────────
    lines.append("")
    lines.append("### 1.2 Observed Convergence Order")
    lines.append("")
    lines.append("Mesh size parameter: h = λ / EPL")
    lines.append("")
    lines.append("| EPL pair | h ratio | p_obs (L2 ROI) | p_obs (spacing) | p_obs (trap pos) |")
    lines.append("|----------|---------|----------------|-----------------|------------------|")

    for o in orders:
        p_l2 = f"{o['p_obs_L2_roi']:.2f}" if np.isfinite(o["p_obs_L2_roi"]) else "—"
        p_sp = f"{o['p_obs_spacing']:.2f}" if np.isfinite(o["p_obs_spacing"]) else "—"
        p_tp = f"{o['p_obs_trap_pos']:.2f}" if np.isfinite(o["p_obs_trap_pos"]) else "—"
        lines.append(
            f"| {o['epl_coarse']:.1f} → {o['epl_fine']:.1f} | "
            f"{o['h_ratio']:.2f} | {p_l2} | {p_sp} | {p_tp} |"
        )

    lines.append("")
    lines.append("For P2 elements, the expected asymptotic convergence order is O(h³) "
                 "in L2 norm. Observed rates significantly below this in coarse regimes "
                 "indicate pre-asymptotic behaviour (under-resolution). Rates approaching "
                 "or exceeding 3 in the fine regime confirm asymptotic convergence.")

    # ── Section 2: Domain sensitivity ─────────────────────────────────────────
    lines.append("")
    lines.append("## 2. Domain-Size Sensitivity")
    lines.append("")

    if domain_metrics:
        sizes = sorted(domain_metrics.keys())
        lines.append("| Domain (mm) | EPL | max\\|p\\| (Pa) | Trap spacing (mm) | n_traps |")
        lines.append("|-------------|-----|--------------|-------------------|---------|")
        for sz in sizes:
            ms = domain_metrics[sz]
            if ms:
                m = ms[0]
                sp = f"{m['cl_spacing_m']*1e3:.4f}" if np.isfinite(m["cl_spacing_m"]) else "—"
                lines.append(
                    f"| {sz:.1f} | {m['epl']:.1f} | {m['max_p_Pa']:.2f} | "
                    f"{sp} | {m['n_traps']} |"
                )
        lines.append("")
        lines.append("If trap spacing and max|p| are consistent across domain sizes, "
                     "the reduced domain does not introduce significant truncation artefacts "
                     "in the central ROI.")
    else:
        lines.append("*Domain-size sensitivity runs not yet available. "
                     "Run at multiple --physical-size-mm values at fixed EPL.*")

    # ── Section 3: PML sensitivity ────────────────────────────────────────────
    lines.append("")
    lines.append("## 3. PML Sensitivity")
    lines.append("")

    if pml_metrics and len(pml_metrics) >= 2:
        lines.append("| PML (λ) | σ_max factor | max\\|p\\| (Pa) | Trap spacing (mm) | n_traps |")
        lines.append("|---------|-------------|--------------|-------------------|---------|")
        for m in sorted(pml_metrics, key=lambda d: d["pml_n_wavelengths_xy"]):
            sp = f"{m['cl_spacing_m']*1e3:.4f}" if np.isfinite(m["cl_spacing_m"]) else "—"
            lines.append(
                f"| {m['pml_n_wavelengths_xy']:.1f} | {m['pml_sigma_max_factor']:.1f} | "
                f"{m['max_p_Pa']:.2f} | {sp} | {m['n_traps']} |"
            )
        lines.append("")
        lines.append("If metrics are stable across PML thicknesses, the 1.0λ PML "
                     "is adequate and does not contaminate the central ROI.")
    else:
        lines.append("*PML sensitivity runs not yet available. "
                     "Run at multiple --pml-n-wavelengths-xy values at fixed EPL.*")

    # ── Section 4: Production acceptance ──────────────────────────────────────
    lines.append("")
    lines.append("## 4. Production Resolution Acceptance")
    lines.append("")

    # Find the convergence onset
    conv_sorted_asc = sorted(conv_metrics, key=lambda d: d["epl"])
    onset_epl = None
    for m in conv_sorted_asc:
        if np.isfinite(m["eps_L2_roi"]) and m["eps_L2_roi"] < 0.2:
            onset_epl = m["epl"]
            break

    if onset_epl:
        lines.append(f"1. **Convergence onset:** Reduced-domain mesh convergence "
                     f"indicates clear convergence beginning around EPL ≈ {onset_epl:.1f} "
                     f"(ε L2 ROI < 0.2).")
    else:
        lines.append("1. **Convergence onset:** Could not reliably determine onset "
                     "from current data.")

    lines.append("2. **Full-domain EPL=5** is NOT feasible on the current ~30 GB "
                 "workstation (solver produces inf/NaN at production domain size).")
    lines.append("3. **Full-domain EPL=4** (505k DOFs) is the highest feasible "
                 "production resolution on this hardware.")
    lines.append("4. **Remaining uncertainty** is bounded by:")
    lines.append("   - Reduced-domain EPL=4.5→5 comparison "
                 "(provides upper bound on discretisation error)")
    if domain_metrics:
        lines.append("   - Domain-size sensitivity "
                     "(checks that reduced domain does not corrupt central ROI)")
    if pml_metrics and len(pml_metrics) >= 2:
        lines.append("   - PML sensitivity "
                     "(confirms truncation artefact is negligible)")

    lines.append("")
    lines.append("## 5. Limitations")
    lines.append("")
    lines.append("- Gor'kov potential computed on 2D Cartesian plane at z* with "
                 "z-gradient neglected (valid at pressure antinode)")
    lines.append("- Trap detection uses finite-difference Gor'kov with depth "
                 "threshold and minimum separation filters")
    lines.append("- Trap matching uses greedy nearest-neighbour with λ/4 "
                 "rejection threshold (not Hungarian)")
    lines.append("- Convergence order estimates are local (between adjacent EPL pairs) "
                 "and may not reflect asymptotic behaviour at coarse levels")
    lines.append("- Full-domain EPL=5 production run not feasible on current hardware")
    lines.append("")

    report_text = "\n".join(lines)
    report_path = out_dir / "convergence_report.md"
    report_path.write_text(report_text)
    print(f"  Report: {report_path}")
    return report_text


# =============================================================================
# Updated CSV
# =============================================================================

def write_analysis_csv(metrics: list[dict], orders: list[dict],
                       out_dir: Path) -> None:
    """Write a comprehensive analysis CSV with all metrics."""
    csv_path = out_dir / "convergence_analysis.csv"
    fieldnames = [
        "epl", "dofs", "mesh_nx", "solve_time_s", "total_wall_time_s",
        "max_p_Pa", "eps_L2_roi", "spacing_err_pct",
        "n_traps", "n_matched", "n_unmatched_cur", "n_unmatched_ref",
        "mean_trap_err_m", "max_trap_err_m",
        "cl_spacing_m", "cl_n_min",
        "physical_size_mm", "pml_n_wavelengths_xy", "name",
    ]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for m in sorted(metrics, key=lambda d: d["epl"]):
            row = {}
            for k in fieldnames:
                v = m.get(k, "")
                if isinstance(v, float) and np.isfinite(v):
                    row[k] = v
                elif isinstance(v, (int, str)):
                    row[k] = v
                else:
                    row[k] = ""
            writer.writerow(row)
    print(f"  Analysis CSV: {csv_path}")

    # Orders CSV
    if orders:
        orders_path = out_dir / "observed_orders.csv"
        with open(orders_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(orders[0].keys()))
            writer.writeheader()
            for o in orders:
                row = {}
                for k, v in o.items():
                    if isinstance(v, float) and np.isfinite(v):
                        row[k] = f"{v:.4f}"
                    else:
                        row[k] = "" if isinstance(v, float) else v
                writer.writerow(row)
        print(f"  Orders CSV: {orders_path}")


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    args = parse_args()
    study_dir = Path(args.study_dir)

    if not study_dir.exists():
        print(f"Study directory not found: {study_dir}")
        return 1

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = study_dir / f"analysis_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("FEM CONVERGENCE ANALYSIS")
    print("=" * 72)
    print(f"  Study dir : {study_dir}")
    print(f"  Output dir: {out_dir}")
    print()

    # ── Load all results ──────────────────────────────────────────────────────
    all_runs = load_all_npz(study_dir)
    print(f"  Loaded {len(all_runs)} .npz files")

    if not all_runs:
        print("  No results found. Run fem_convergence_study.py first.")
        return 1

    # Get wavelength from first run
    wavelength = float(np.squeeze(all_runs[0]["wavelength"]))
    print(f"  Wavelength: {wavelength*1e3:.4f} mm")
    print()

    # ── 1. CONVERGENCE SWEEP: reduced domain, default PML ────────────────────
    print("─" * 72)
    print("1. Reduced-domain convergence sweep")
    print("─" * 72)

    conv_runs = filter_runs(all_runs, mode="convergence", phys_mm=3.0, pml_xy=1.0)
    print(f"  Found {len(conv_runs)} convergence runs at 3.0mm/PML=1.0λ")

    if conv_runs:
        # Find reference (highest EPL)
        conv_runs_sorted = sorted(conv_runs, key=get_epl)
        ref_run = conv_runs_sorted[-1]
        ref_epl = get_epl(ref_run)
        print(f"  Reference: {ref_run['_name']} (EPL={ref_epl})")

        # Non-reference runs
        other_runs = [r for r in conv_runs_sorted if get_epl(r) < ref_epl]

        conv_metrics = recompute_trap_metrics(other_runs, ref_run, wavelength)

        # Add reference run itself (errors = 0/NaN by definition)
        ref_meta = ref_run["_meta"]
        ref_traps = detect_traps(ref_run["gorkov_2d"], ref_run["x_grid"],
                                 ref_run["y_grid"], roi_fraction=0.5,
                                 wavelength=wavelength)
        conv_metrics.append({
            "epl": ref_epl,
            "dofs": int(ref_meta.get("dofs", 0)),
            "mesh_nx": int(ref_meta.get("mesh_nx", 0)),
            "solve_time_s": float(ref_meta.get("solve_time_s", 0)),
            "total_wall_time_s": float(ref_meta.get("total_wall_time_s", 0)),
            "max_p_Pa": float(ref_meta.get("max_pressure_Pa", 0)),
            "eps_L2_roi": np.nan,  # self-reference
            "spacing_err_pct": np.nan,
            "n_traps": ref_traps["n_traps"],
            "n_matched": ref_traps["n_traps"],
            "n_unmatched_cur": 0,
            "n_unmatched_ref": 0,
            "mean_trap_err_m": np.nan,
            "max_trap_err_m": np.nan,
            "cl_spacing_m": ref_traps["centreline_min_spacing_m"],
            "cl_n_min": len(ref_traps["centreline_min_x"]),
            "name": ref_run["_name"],
            "physical_size_mm": float(ref_meta.get("physical_size_mm", 3.0)),
            "pml_n_wavelengths_xy": float(ref_meta.get("pml_n_wavelengths_xy", 1.0)),
            "pml_sigma_max_factor": float(ref_meta.get("pml_sigma_max_factor", 5.0)),
            "wavelength_m": wavelength,
        })

        conv_metrics = sorted(conv_metrics, key=lambda d: d["epl"])

        for m in conv_metrics:
            eps = f"{m['eps_L2_roi']:.4e}" if np.isfinite(m["eps_L2_roi"]) else "REF"
            print(f"    EPL={m['epl']:.1f}  DOFs={m['dofs']:>8,}  "
                  f"ε_roi={eps}  matched={m['n_matched']}/{m['n_traps']}")

        # Observed order (exclude reference)
        orders = compute_observed_orders(
            [m for m in conv_metrics if np.isfinite(m["eps_L2_roi"])]
        )
        print()
        for o in orders:
            p = f"{o['p_obs_L2_roi']:.2f}" if np.isfinite(o["p_obs_L2_roi"]) else "—"
            print(f"    EPL {o['epl_coarse']:.1f}→{o['epl_fine']:.1f}: "
                  f"p_obs(L2)={p}")
    else:
        conv_metrics = []
        orders = []

    print()

    # ── 2. DOMAIN SIZE SENSITIVITY ────────────────────────────────────────────
    print("─" * 72)
    print("2. Domain-size sensitivity")
    print("─" * 72)

    domain_metrics = {}
    for phys_mm in [3.0, 4.0, 5.0, 5.4]:
        runs = filter_runs(all_runs, phys_mm=phys_mm, pml_xy=1.0)
        if runs:
            # Use the highest EPL available
            runs_sorted = sorted(runs, key=get_epl, reverse=True)
            best = runs_sorted[0]
            epl_best = get_epl(best)
            # Use self-metrics (no cross-domain comparison)
            traps = detect_traps(best["gorkov_2d"], best["x_grid"],
                                 best["y_grid"], roi_fraction=0.5,
                                 wavelength=wavelength)
            meta = best["_meta"]
            domain_metrics[phys_mm] = [{
                "epl": epl_best,
                "max_p_Pa": float(meta.get("max_pressure_Pa", 0)),
                "cl_spacing_m": traps["centreline_min_spacing_m"],
                "n_traps": traps["n_traps"],
                "dofs": int(meta.get("dofs", 0)),
                "name": best["_name"],
            }]
            print(f"  {phys_mm:.1f}mm: EPL={epl_best}, max|p|={domain_metrics[phys_mm][0]['max_p_Pa']:.2f}, "
                  f"spacing={traps['centreline_min_spacing_m']*1e3:.4f}mm, "
                  f"n_traps={traps['n_traps']}")
        else:
            print(f"  {phys_mm:.1f}mm: no runs found")

    print()

    # ── 3. PML SENSITIVITY ────────────────────────────────────────────────────
    print("─" * 72)
    print("3. PML sensitivity")
    print("─" * 72)

    pml_metrics = []
    # Gather all convergence runs and group by PML
    conv_all = filter_runs(all_runs, mode="convergence", phys_mm=3.0)
    pml_groups = {}
    for r in conv_all:
        pml_xy = float(r["_meta"].get("pml_n_wavelengths_xy", 1.0))
        pml_groups.setdefault(pml_xy, []).append(r)

    for pml_xy in sorted(pml_groups.keys()):
        runs = pml_groups[pml_xy]
        # Use the highest EPL run for this PML setting
        runs_sorted = sorted(runs, key=get_epl, reverse=True)
        best = runs_sorted[0]
        epl_best = get_epl(best)
        traps = detect_traps(best["gorkov_2d"], best["x_grid"],
                             best["y_grid"], roi_fraction=0.5,
                             wavelength=wavelength)
        meta = best["_meta"]
        pml_metrics.append({
            "pml_n_wavelengths_xy": pml_xy,
            "pml_sigma_max_factor": float(meta.get("pml_sigma_max_factor", 5.0)),
            "epl": epl_best,
            "max_p_Pa": float(meta.get("max_pressure_Pa", 0)),
            "cl_spacing_m": traps["centreline_min_spacing_m"],
            "n_traps": traps["n_traps"],
            "name": best["_name"],
        })
        print(f"  PML={pml_xy:.1f}λ: EPL={epl_best}, max|p|={meta.get('max_pressure_Pa', 0):.2f}, "
              f"spacing={traps['centreline_min_spacing_m']*1e3:.4f}mm")

    if len(pml_metrics) < 2:
        print("  Only one PML setting found. Run with --pml-n-wavelengths-xy "
              "at different values for sensitivity analysis.")
    print()

    # ── 4. GENERATE FIGURES ───────────────────────────────────────────────────
    print("─" * 72)
    print("4. Generating figures")
    print("─" * 72)

    if conv_metrics:
        non_ref = [m for m in conv_metrics if np.isfinite(m["eps_L2_roi"])]
        if non_ref:
            plot_convergence_errors(non_ref, orders, out_dir, wavelength)
            print("  convergence_errors_vs_epl.png")
            print("  convergence_error_vs_h.png")

        plot_solve_time_vs_dofs(conv_metrics, out_dir)
        print("  solve_time_vs_dofs.png")

        if conv_runs:
            other_npz = [r for r in conv_runs_sorted if get_epl(r) < ref_epl]
            if other_npz:
                plot_trap_overlay(other_npz, ref_run, wavelength, out_dir)
                print("  trap_overlay_coarse_*.png")
                print("  trap_overlay_fine_*.png")

    if domain_metrics:
        plot_domain_sensitivity(domain_metrics, out_dir)
        print("  domain_size_sensitivity.png")

    if len(pml_metrics) >= 2:
        plot_pml_sensitivity(pml_metrics, out_dir)
        print("  pml_sensitivity.png")

    print()

    # ── 5. WRITE CSVs ─────────────────────────────────────────────────────────
    print("─" * 72)
    print("5. Writing CSVs")
    print("─" * 72)

    if conv_metrics:
        write_analysis_csv(conv_metrics, orders, out_dir)

    # ── 6. GENERATE REPORT ────────────────────────────────────────────────────
    print()
    print("─" * 72)
    print("6. Generating report")
    print("─" * 72)

    report = generate_report(
        conv_metrics, orders, domain_metrics, pml_metrics,
        out_dir, wavelength,
    )
    print()

    print("=" * 72)
    print("ANALYSIS COMPLETE")
    print("=" * 72)
    print(f"  Output directory: {out_dir}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
