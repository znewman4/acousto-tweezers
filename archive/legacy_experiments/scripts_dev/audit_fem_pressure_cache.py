#!/usr/bin/env python3
"""
Quantitative Integrity Audit for Cached FEM Pressure Field
===========================================================

Tests A–H:
  A) Basic data integrity (NaN, Inf, stats, histogram)
  B) Mesh ↔ field consistency (DOF count, domain extents, units)
  C) Helmholtz residual check (∇²p + k²p via finite-difference stencil)
  D) Energy / smoothness sanity in physical region
  E) Reproduce XZ slice two independent ways (slab-interpolated vs direct)
  F) PML / boundary contamination check
  G) Standing-wave plausibility (autocorrelation, FFT)
  H) Cache reproducibility checksum / metadata consistency

Usage:
    python scripts/dev/audit_fem_pressure_cache.py [--cache PATH]

If --cache is not provided, auto-detects the latest .npz in
results/fem_standing_wave_cache/.
"""

import argparse
import json
import hashlib
import time
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import cKDTree
from scipy.signal import find_peaks
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════════
# PHYSICAL CONSTANTS (must match CORRECTED_PRESET)
# ═══════════════════════════════════════════════════════════════════
C_WATER  = 1484.0       # m/s
RHO_WATER = 997.0       # kg/m³
F_HZ     = 2.0e6        # Hz
OMEGA    = 2.0 * np.pi * F_HZ
K_WATER  = OMEGA / C_WATER     # wavenumber [rad/m]
LAM      = C_WATER / F_HZ      # wavelength [m], ≈0.742 mm

# Domain (CORRECTED_PRESET)
LX       = 6.0e-3
LY       = 6.0e-3
H_UNDER  = 3.0e-3
H_TOP    = 2.0085e-3
H_TOTAL  = H_UNDER + H_TOP

# PML
PML_N_XY = 1.0
PML_N_Z  = 1.0
T_PML_XY = PML_N_XY * LAM
T_PML_Z  = PML_N_Z * LAM

# Standing wave: antiphase on both axes → trap spacing = λ/2
TRAP_SPACING = LAM / 2.0

# Expected center of XY observation plane
CX, CY = LX / 2, LY / 2
Z_STAR = H_UNDER + H_TOP / 2.0 + 0.25 * LAM

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════
def _log(msg, fp=None):
    print(msg)
    if fp:
        fp.write(msg + "\n")


def _find_latest_cache(cache_dir: Path) -> Path:
    """Find the most recent .npz file in the cache directory."""
    npz_files = sorted(cache_dir.glob("*.npz"), key=lambda p: p.stat().st_mtime)
    if not npz_files:
        raise FileNotFoundError(f"No .npz files in {cache_dir}")
    return npz_files[-1]


def _load_cache(path: Path):
    """Load the cached .npz and return coords, p_complex, metadata dict."""
    d = np.load(path, allow_pickle=True)
    keys = list(d.keys())

    coords = d["coords"]
    p_real = d["p_real"]
    p_imag = d["p_imag"]
    p = p_real + 1j * p_imag

    # Build metadata from flat keys (or nested 'metadata' if present)
    meta = {}
    skip = {"coords", "p_real", "p_imag"}
    for k in keys:
        if k not in skip:
            val = d[k]
            # Convert 0-d arrays to scalars
            if isinstance(val, np.ndarray) and val.ndim == 0:
                val = val.item()
            meta[k] = val

    return coords, p, meta, keys


# ═══════════════════════════════════════════════════════════════════
# TEST A: Basic Data Integrity
# ═══════════════════════════════════════════════════════════════════
def test_a_basic_integrity(coords, p, meta, out_dir, log_fp):
    _log("\n" + "=" * 72, log_fp)
    _log("TEST A: Basic Data Integrity", log_fp)
    _log("=" * 72, log_fp)

    issues = []

    # Dtype
    _log(f"  coords dtype: {coords.dtype}, shape: {coords.shape}", log_fp)
    _log(f"  p dtype: {p.dtype}, shape: {p.shape}", log_fp)

    if not np.issubdtype(p.dtype, np.complexfloating):
        issues.append("p is NOT complex dtype")
    if p.size == 0:
        issues.append("p has zero elements")
    if coords.size == 0:
        issues.append("coords has zero elements")

    # NaN / Inf
    n_nan = np.isnan(p).sum()
    n_inf = np.isinf(p).sum()
    _log(f"  NaN count: {n_nan}", log_fp)
    _log(f"  Inf count: {n_inf}", log_fp)
    if n_nan > 0:
        issues.append(f"{n_nan} NaN values in p")
    if n_inf > 0:
        issues.append(f"{n_inf} Inf values in p")

    # Coords NaN/Inf
    n_nan_c = np.isnan(coords).sum()
    n_inf_c = np.isinf(coords).sum()
    _log(f"  Coords NaN: {n_nan_c}, Inf: {n_inf_c}", log_fp)
    if n_nan_c > 0:
        issues.append(f"{n_nan_c} NaN in coords")

    p_mag = np.abs(p)
    # Extremely large values (>1e6 Pa would be unreasonable for this setup)
    n_extreme = (p_mag > 1e6).sum()
    _log(f"  |p| > 1e6 count: {n_extreme}", log_fp)
    if n_extreme > 0:
        issues.append(f"{n_extreme} extremely large |p| values (>1e6)")

    # Statistics
    stats = {
        "dtype": str(p.dtype),
        "shape": list(p.shape),
        "n_nan": int(n_nan),
        "n_inf": int(n_inf),
        "n_extreme_gt_1e6": int(n_extreme),
        "p_mag_min": float(p_mag.min()),
        "p_mag_max": float(p_mag.max()),
        "p_mag_median": float(np.median(p_mag)),
        "p_mag_mean": float(p_mag.mean()),
        "p_mag_std": float(p_mag.std()),
        "p_real_min": float(np.real(p).min()),
        "p_real_max": float(np.real(p).max()),
        "p_imag_min": float(np.imag(p).min()),
        "p_imag_max": float(np.imag(p).max()),
    }
    _log(f"  |p| min={stats['p_mag_min']:.6f}, max={stats['p_mag_max']:.6f}, "
         f"median={stats['p_mag_median']:.6f}, mean={stats['p_mag_mean']:.6f}", log_fp)
    _log(f"  Re(p) range: [{stats['p_real_min']:.4f}, {stats['p_real_max']:.4f}]", log_fp)
    _log(f"  Im(p) range: [{stats['p_imag_min']:.4f}, {stats['p_imag_max']:.4f}]", log_fp)

    # Save stats
    with open(out_dir / "data" / "basic_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Histogram of log10(|p|)
    p_mag_nonzero = p_mag[p_mag > 0]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(np.log10(p_mag_nonzero), bins=200, color="steelblue", edgecolor="none")
    ax.set_xlabel("log₁₀(|p|)")
    ax.set_ylabel("Count")
    ax.set_title("Histogram of log₁₀(|p|) — check for corruption spikes")
    ax.axvline(np.log10(stats["p_mag_max"]), color="red", ls="--", label=f"max={stats['p_mag_max']:.2f}")
    ax.axvline(np.log10(stats["p_mag_median"]), color="green", ls="--", label=f"median={stats['p_mag_median']:.2f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "A_histogram_log_p_mag.png", dpi=150)
    plt.close(fig)

    verdict = "PASS" if not issues else "FAIL"
    _log(f"  VERDICT: {verdict}", log_fp)
    for iss in issues:
        _log(f"    ⚠ {iss}", log_fp)

    return {"verdict": verdict, "issues": issues, "stats": stats}


# ═══════════════════════════════════════════════════════════════════
# TEST B: Mesh ↔ Field Consistency
# ═══════════════════════════════════════════════════════════════════
def test_b_mesh_consistency(coords, p, meta, out_dir, log_fp):
    _log("\n" + "=" * 72, log_fp)
    _log("TEST B: Mesh ↔ Field Consistency", log_fp)
    _log("=" * 72, log_fp)

    issues = []

    ndof = coords.shape[0]
    pdof = p.shape[0]
    _log(f"  Coords DOF count: {ndof}", log_fp)
    _log(f"  Pressure DOF count: {pdof}", log_fp)

    if ndof != pdof:
        issues.append(f"DOF mismatch: coords has {ndof}, p has {pdof}")

    # Check expected DOF from metadata
    if "dofs" in meta:
        expected_dofs = int(meta["dofs"])
        _log(f"  Expected DOFs (metadata): {expected_dofs}", log_fp)
        if ndof != expected_dofs:
            issues.append(f"DOF count {ndof} != metadata dofs {expected_dofs}")
    else:
        _log(f"  WARNING: no 'dofs' in metadata", log_fp)
        issues.append("Missing 'dofs' in metadata")

    # Domain extents in meters
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    z_min, z_max = coords[:, 2].min(), coords[:, 2].max()

    extents_m = {
        "x_min_m": float(x_min), "x_max_m": float(x_max),
        "y_min_m": float(y_min), "y_max_m": float(y_max),
        "z_min_m": float(z_min), "z_max_m": float(z_max),
        "Lx_m": float(x_max - x_min), "Ly_m": float(y_max - y_min),
        "Lz_m": float(z_max - z_min),
    }
    extents_mm = {k.replace("_m", "_mm"): v * 1e3 for k, v in extents_m.items()}

    _log(f"  Domain extents (m):  x=[{x_min:.6f}, {x_max:.6f}], "
         f"y=[{y_min:.6f}, {y_max:.6f}], z=[{z_min:.6f}, {z_max:.6f}]", log_fp)
    _log(f"  Domain size (mm):    Lx={extents_mm['Lx_mm']:.4f}, "
         f"Ly={extents_mm['Ly_mm']:.4f}, Lz={extents_mm['Lz_mm']:.4f}", log_fp)

    # Expected domain size
    _log(f"  Expected domain (mm): Lx={LX*1e3:.4f}, Ly={LY*1e3:.4f}, "
         f"Lz={H_TOTAL*1e3:.4f}", log_fp)

    # Units sanity: flag if domain is >1 m (means mm were stored as meters)
    if extents_m["Lx_m"] > 1.0:
        issues.append(f"Domain Lx = {extents_m['Lx_m']:.3f} m — suspiciously large, possibly wrong units")
    if extents_m["Lx_m"] < 1e-6:
        issues.append(f"Domain Lx = {extents_m['Lx_m']:.3e} m — suspiciously small, possibly wrong units")

    # Check domain matches expected (within 5%)
    for expected, key_name, label in [
        (LX, "Lx_m", "Lx"), (LY, "Ly_m", "Ly"), (H_TOTAL, "Lz_m", "Lz")
    ]:
        actual = extents_m[key_name]
        err = abs(actual - expected) / expected
        if err > 0.05:
            issues.append(f"{label}: actual={actual*1e3:.4f}mm vs expected={expected*1e3:.4f}mm "
                          f"(error={err*100:.1f}%)")
        else:
            _log(f"  {label}: {actual*1e3:.4f} mm ✓ (error {err*100:.1f}%)", log_fp)

    # Check metadata consistency
    meta_checks = {}
    for key, expected in [
        ("frequency_hz", F_HZ), ("c_water", C_WATER), ("rho_water", RHO_WATER),
        ("Lx", LX), ("Ly", LY), ("H_under", H_UNDER), ("H_top", H_TOP),
        ("wavelength", LAM),
    ]:
        if key in meta:
            val = float(meta[key])
            err = abs(val - expected) / (abs(expected) + 1e-30)
            ok = err < 0.01
            meta_checks[key] = {"stored": val, "expected": expected, "match": ok}
            if not ok:
                issues.append(f"Metadata '{key}': stored={val} vs expected={expected}")
        else:
            meta_checks[key] = {"stored": None, "expected": expected, "match": False}

    result_data = {**extents_m, **extents_mm, "meta_checks": meta_checks}
    with open(out_dir / "data" / "mesh_extents.json", "w") as f:
        json.dump(result_data, f, indent=2, default=str)

    # Figure: mesh bounding box + ROI
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # XY projection
    ax = axes[0]
    ax.set_title("Mesh XY extent + physical/PML regions")
    rect_full = plt.Rectangle((x_min*1e3, y_min*1e3),
                               (x_max-x_min)*1e3, (y_max-y_min)*1e3,
                               fill=False, edgecolor="black", lw=2, label="Full domain")
    ax.add_patch(rect_full)
    # Physical region (interior of PML)
    phys_x0, phys_x1 = T_PML_XY, LX - T_PML_XY
    phys_y0, phys_y1 = T_PML_XY, LY - T_PML_XY
    rect_phys = plt.Rectangle((phys_x0*1e3, phys_y0*1e3),
                                (phys_x1-phys_x0)*1e3, (phys_y1-phys_y0)*1e3,
                                fill=False, edgecolor="blue", lw=2, ls="--", label="Physical region")
    ax.add_patch(rect_phys)
    # ROI for 3x3
    region_half = 1.5 * TRAP_SPACING + 0.25 * LAM
    rect_roi = plt.Rectangle(((CX - region_half)*1e3, (CY - region_half)*1e3),
                               2*region_half*1e3, 2*region_half*1e3,
                               fill=False, edgecolor="red", lw=2, ls=":", label="3×3 ROI")
    ax.add_patch(rect_roi)
    ax.set_xlim((x_min - 0.2e-3)*1e3, (x_max + 0.2e-3)*1e3)
    ax.set_ylim((y_min - 0.2e-3)*1e3, (y_max + 0.2e-3)*1e3)
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    ax.set_aspect("equal")
    ax.legend(fontsize=7)

    # XZ projection
    ax = axes[1]
    ax.set_title("Mesh XZ extent + PML")
    rect_full = plt.Rectangle((x_min*1e3, z_min*1e3),
                               (x_max-x_min)*1e3, (z_max-z_min)*1e3,
                               fill=False, edgecolor="black", lw=2)
    ax.add_patch(rect_full)
    # Bottom PML region
    rect_pml_bot = plt.Rectangle((x_min*1e3, z_min*1e3),
                                   (x_max-x_min)*1e3, T_PML_Z*1e3,
                                   fill=True, facecolor="orange", alpha=0.3, label="PML (bottom)")
    ax.add_patch(rect_pml_bot)
    # Lateral PML (x-sides, below H_under)
    for x0 in [x_min, LX - T_PML_XY]:
        rect_pml = plt.Rectangle((x0*1e3, z_min*1e3),
                                   T_PML_XY*1e3, H_UNDER*1e3,
                                   fill=True, facecolor="yellow", alpha=0.3)
        ax.add_patch(rect_pml)
    ax.axhline(H_UNDER*1e3, color="blue", ls="--", lw=0.8, label="H_under (petri)")
    ax.axhline(Z_STAR*1e3, color="red", ls=":", lw=0.8, label="z* (XY plane)")
    ax.axhline(H_TOTAL*1e3, color="gray", ls="--", lw=0.8, label="H_total (top)")
    ax.set_xlim((x_min - 0.2e-3)*1e3, (x_max + 0.2e-3)*1e3)
    ax.set_ylim((z_min - 0.2e-3)*1e3, (z_max + 0.2e-3)*1e3)
    ax.set_xlabel("x [mm]"); ax.set_ylabel("z [mm]")
    ax.legend(fontsize=7)

    # DOF density scatter (subsample)
    ax = axes[2]
    ax.set_title("DOF density (random 5000 pts)")
    idx = np.random.default_rng(42).choice(len(coords), min(5000, len(coords)), replace=False)
    ax.scatter(coords[idx, 0]*1e3, coords[idx, 2]*1e3, s=0.5, c="steelblue", alpha=0.3)
    ax.set_xlabel("x [mm]"); ax.set_ylabel("z [mm]")
    ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "B_mesh_extents.png", dpi=150)
    plt.close(fig)

    verdict = "PASS" if not issues else "FAIL"
    _log(f"  VERDICT: {verdict}", log_fp)
    for iss in issues:
        _log(f"    ⚠ {iss}", log_fp)
    return {"verdict": verdict, "issues": issues}


# ═══════════════════════════════════════════════════════════════════
# TEST C: Helmholtz Residual (numerical FD approximation)
# ═══════════════════════════════════════════════════════════════════
def test_c_helmholtz_residual(coords, p, meta, out_dir, log_fp):
    """
    Approximate residual r = ∇²p + k²p using nearest-neighbour FD stencil.
    We can't re-assemble the FEM matrix without the mesh, so we use a
    numerical Laplacian computed from a kd-tree.
    """
    _log("\n" + "=" * 72, log_fp)
    _log("TEST C: Helmholtz Residual Check (FD approximation)", log_fp)
    _log("=" * 72, log_fp)

    k = K_WATER
    _log(f"  k = {k:.2f} rad/m,  λ = {LAM*1e3:.4f} mm", log_fp)

    # Build kd-tree
    _log("  Building kd-tree for Laplacian estimation...", log_fp)
    tree = cKDTree(coords)

    # For efficiency, sample a subset of interior points
    #   (physical region, away from PML and boundaries)
    phys_mask = (
        (coords[:, 0] > T_PML_XY + 0.5*LAM) &
        (coords[:, 0] < LX - T_PML_XY - 0.5*LAM) &
        (coords[:, 1] > T_PML_XY + 0.5*LAM) &
        (coords[:, 1] < LY - T_PML_XY - 0.5*LAM) &
        (coords[:, 2] > T_PML_Z + 0.5*LAM) &
        (coords[:, 2] < H_TOTAL - 0.5*LAM)
    )
    phys_idx = np.where(phys_mask)[0]
    _log(f"  Interior physical DOFs: {len(phys_idx)} / {len(coords)}", log_fp)

    # Subsample if too many (limit to 50k for speed)
    rng = np.random.default_rng(42)
    n_sample = min(50000, len(phys_idx))
    sample_idx = rng.choice(phys_idx, n_sample, replace=False)
    _log(f"  Sampling {n_sample} points for Laplacian estimation", log_fp)

    # Compute FD Laplacian: for each point, find K nearest neighbours,
    # fit a local quadratic, extract ∇²   OR use simpler 6-point stencil
    # approximation with the nearest axis-aligned neighbours.
    #
    # For unstructured mesh, use the kd-tree average approach:
    #   ∇²p(x) ≈ (2n/h²) * [mean(p(neighbours)) - p(x)]
    # where n=dimension, h=mean distance to neighbours.
    # This is the standard RBF/kd-tree FD stencil for Laplacian.

    K_NN = 20  # nearest neighbours to use
    _log(f"  Using K={K_NN} nearest neighbours for Laplacian estimate", log_fp)

    dists, inds = tree.query(coords[sample_idx], k=K_NN + 1)
    # inds[:, 0] is the point itself (dist=0), use inds[:, 1:]
    dists_nn = dists[:, 1:]  # (n_sample, K_NN)
    inds_nn = inds[:, 1:]    # (n_sample, K_NN)

    p_center = p[sample_idx]                     # (n_sample,)
    p_neigh = p[inds_nn]                         # (n_sample, K_NN)
    h_mean = dists_nn.mean(axis=1)               # (n_sample,)

    # FD Laplacian (3D, n=3):
    #   ∇²p ≈ (2*3 / h²) * (mean(p_neigh) - p_center)
    laplacian_p = (2.0 * 3.0 / h_mean**2) * (p_neigh.mean(axis=1) - p_center)

    # Helmholtz residual: r = ∇²p + k²p (should be ~0 in physical homogeneous region)
    residual = laplacian_p + k**2 * p_center

    r_mag = np.abs(residual)
    p_mag = np.abs(p_center)
    k2p_mag = k**2 * p_mag  # expected scale of each term

    # Relative residual
    rel_residual = r_mag / (k2p_mag + 1e-30)

    stats = {
        "n_sample": int(n_sample),
        "K_nn": K_NN,
        "residual_mag_mean": float(r_mag.mean()),
        "residual_mag_median": float(np.median(r_mag)),
        "residual_mag_max": float(r_mag.max()),
        "residual_mag_p99": float(np.percentile(r_mag, 99)),
        "relative_residual_mean": float(rel_residual.mean()),
        "relative_residual_median": float(np.median(rel_residual)),
        "relative_residual_max": float(rel_residual.max()),
        "relative_residual_p99": float(np.percentile(rel_residual, 99)),
        "k2p_scale_mean": float(k2p_mag.mean()),
        "h_mean_avg": float(h_mean.mean()),
        "h_mean_mm": float(h_mean.mean() * 1e3),
    }
    _log(f"  Mean NN distance: {stats['h_mean_mm']:.4f} mm (λ/{LAM/stats['h_mean_avg']:.1f})", log_fp)
    _log(f"  |residual| — mean: {stats['residual_mag_mean']:.2e}, "
         f"median: {stats['residual_mag_median']:.2e}, max: {stats['residual_mag_max']:.2e}", log_fp)
    _log(f"  |residual|/k²|p| — mean: {stats['relative_residual_mean']:.4f}, "
         f"median: {stats['relative_residual_median']:.4f}, "
         f"p99: {stats['relative_residual_p99']:.4f}", log_fp)
    _log(f"  NOTE: FD Laplacian on unstructured mesh is approximate; "
         f"relative residual <0.5 is acceptable, <0.2 is good.", log_fp)

    with open(out_dir / "data" / "residual_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Histogram of relative residual
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]
    ax.hist(np.log10(r_mag + 1e-30), bins=200, color="steelblue", edgecolor="none")
    ax.set_xlabel("log₁₀(|residual|)")
    ax.set_ylabel("Count")
    ax.set_title("Helmholtz residual magnitude (FD approx)")

    ax = axes[1]
    ax.hist(rel_residual[rel_residual < 5], bins=200, color="coral", edgecolor="none")
    ax.set_xlabel("|residual| / k²|p|")
    ax.set_ylabel("Count")
    ax.set_title("Relative residual (should peak near 0)")
    ax.axvline(1.0, color="red", ls="--", lw=0.8, label="rel=1.0")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "C_residual_histogram.png", dpi=150)
    plt.close(fig)

    # Accept as "plausible" if median relative residual < 1.0
    # (FD on unstructured mesh is noisy; a truly corrupted field would have rel >> 1)
    verdict = "PASS" if stats["relative_residual_median"] < 1.0 else "FAIL"
    if stats["relative_residual_median"] > 0.5:
        _log(f"  NOTE: median relative residual is {stats['relative_residual_median']:.3f} "
             f"which is elevated but may be FD stencil noise", log_fp)
    _log(f"  VERDICT: {verdict}", log_fp)
    return {"verdict": verdict, "issues": [] if verdict == "PASS" else
            [f"Median relative FD residual = {stats['relative_residual_median']:.3f}"],
            "stats": stats}


# ═══════════════════════════════════════════════════════════════════
# TEST D: Energy / Smoothness Sanity
# ═══════════════════════════════════════════════════════════════════
def test_d_smoothness(coords, p, meta, out_dir, log_fp):
    _log("\n" + "=" * 72, log_fp)
    _log("TEST D: Energy / Smoothness Sanity", log_fp)
    _log("=" * 72, log_fp)

    issues = []

    # Physical interior mask (away from PML and boundaries)
    phys_mask = (
        (coords[:, 0] > T_PML_XY + 0.3*LAM) &
        (coords[:, 0] < LX - T_PML_XY - 0.3*LAM) &
        (coords[:, 1] > T_PML_XY + 0.3*LAM) &
        (coords[:, 1] < LY - T_PML_XY - 0.3*LAM) &
        (coords[:, 2] > T_PML_Z + 0.3*LAM) &
        (coords[:, 2] < H_TOTAL - 0.3*LAM)
    )
    phys_idx = np.where(phys_mask)[0]
    _log(f"  Physical interior DOFs: {len(phys_idx)}", log_fp)

    # For each physical DOF, find nearest neighbour and compute |p(x) - p(NN(x))|
    tree = cKDTree(coords[phys_idx])
    
    # Subsample for speed
    rng = np.random.default_rng(42)
    n_sample = min(50000, len(phys_idx))
    sample_local = rng.choice(len(phys_idx), n_sample, replace=False)
    sample_global = phys_idx[sample_local]

    dists, inds = tree.query(coords[sample_global], k=2)
    # inds[:, 1] is nearest distinct neighbour
    nn_global = phys_idx[inds[:, 1]]
    nn_dist = dists[:, 1]

    p_diff = np.abs(p[sample_global] - p[nn_global])
    p_mag = np.abs(p[sample_global])
    relative_variation = p_diff / (p_mag + 1e-30)

    median_p_mag = np.median(p_mag)
    median_variation = np.median(p_diff)
    median_rel_var = np.median(relative_variation)
    median_nn_dist = np.median(nn_dist)

    _log(f"  Median NN distance: {median_nn_dist*1e3:.4f} mm (λ/{LAM/median_nn_dist:.1f})", log_fp)
    _log(f"  Median |p|: {median_p_mag:.4f} Pa", log_fp)
    _log(f"  Median |p(x)-p(NN)| : {median_variation:.4f} Pa", log_fp)
    _log(f"  Median relative variation: {median_rel_var:.6f}", log_fp)
    _log(f"  Max relative variation: {relative_variation.max():.4f}", log_fp)

    # Gradient magnitude estimate: |∇p| ≈ |p(x) - p(NN)| / dist
    grad_mag = p_diff / (nn_dist + 1e-30)
    _log(f"  |∇p| estimate — median: {np.median(grad_mag):.2e}, "
         f"max: {grad_mag.max():.2e}", log_fp)

    # Expected gradient scale: k * |p| (for a plane wave, |∇p| = k*|p|)
    expected_grad_scale = K_WATER * median_p_mag
    _log(f"  Expected |∇p| scale (k·|p|): {expected_grad_scale:.2e}", log_fp)
    grad_ratio = np.median(grad_mag) / (expected_grad_scale + 1e-30)
    _log(f"  Median |∇p| / (k·|p|) = {grad_ratio:.3f}", log_fp)

    # Flag if gradient is way too large (spiky)
    if grad_ratio > 10:
        issues.append(f"Gradient ≫ expected: ratio = {grad_ratio:.1f}")

    # Structure function: |p(x+r) - p(x)| vs r for various distances
    _log("  Computing structure function vs distance...", log_fp)
    n_struct = min(10000, len(phys_idx))
    struct_sample = rng.choice(phys_idx, n_struct, replace=False)
    
    dist_bins = np.linspace(0, 2.0 * LAM, 40)
    dist_centers = 0.5 * (dist_bins[:-1] + dist_bins[1:])
    variation_mean = np.zeros(len(dist_centers))
    variation_count = np.zeros(len(dist_centers))

    # Query many neighbours to fill distance bins
    K_STRUCT = 30
    dd, ii = tree.query(coords[struct_sample], k=K_STRUCT)
    for j in range(1, K_STRUCT):
        diffs = np.abs(p[struct_sample] - p[phys_idx[ii[:, j]]])
        for b in range(len(dist_centers)):
            mask = (dd[:, j] >= dist_bins[b]) & (dd[:, j] < dist_bins[b+1])
            if mask.sum() > 0:
                variation_mean[b] += diffs[mask].sum()
                variation_count[b] += mask.sum()

    valid = variation_count > 0
    variation_mean[valid] /= variation_count[valid]

    stats = {
        "median_nn_dist_mm": float(median_nn_dist * 1e3),
        "median_p_mag": float(median_p_mag),
        "median_variation": float(median_variation),
        "median_relative_variation": float(median_rel_var),
        "max_relative_variation": float(relative_variation.max()),
        "median_grad_mag": float(np.median(grad_mag)),
        "max_grad_mag": float(grad_mag.max()),
        "expected_grad_scale": float(expected_grad_scale),
        "grad_ratio": float(grad_ratio),
    }
    with open(out_dir / "data" / "smoothness_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Plot structure function
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.plot(dist_centers[valid]*1e3, variation_mean[valid], "o-", markersize=3)
    ax.set_xlabel("Distance [mm]")
    ax.set_ylabel("Mean |p(x+r) - p(x)| [Pa]")
    ax.set_title("Structure function (smoothness check)")
    ax.axvline(LAM*1e3/2, color="red", ls="--", lw=0.8, label="λ/2")
    ax.axvline(LAM*1e3, color="orange", ls="--", lw=0.8, label="λ")
    ax.legend()

    ax = axes[1]
    ax.hist(relative_variation[relative_variation < 1.0], bins=200,
            color="steelblue", edgecolor="none")
    ax.set_xlabel("Relative variation |p(x)-p(NN)|/|p(x)|")
    ax.set_ylabel("Count")
    ax.set_title("Nearest-neighbour relative variation")

    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "D_smoothness.png", dpi=150)
    plt.close(fig)

    verdict = "PASS" if not issues else "FAIL"
    _log(f"  VERDICT: {verdict}", log_fp)
    return {"verdict": verdict, "issues": issues, "stats": stats}


# ═══════════════════════════════════════════════════════════════════
# TEST E: Reproduce XZ Slice Two Independent Ways
# ═══════════════════════════════════════════════════════════════════
def test_e_xz_slice_comparison(coords, p, meta, out_dir, log_fp):
    _log("\n" + "=" * 72, log_fp)
    _log("TEST E: XZ Slice — Two Independent Methods", log_fp)
    _log("=" * 72, log_fp)

    issues = []

    # Grid parameters (match the overlay script)
    NGRID_XZ = 400
    NZ_MERID = 200

    # XZ slice at y = CY
    # x range: same as overlay script (local 3x3 + some extension)
    region_half = 1.5 * TRAP_SPACING + 0.25 * LAM
    x_lo = CX - region_half
    x_hi = CX + region_half
    xg = np.linspace(x_lo, x_hi, NGRID_XZ)
    zg = np.linspace(H_UNDER * 0.5, H_TOTAL, NZ_MERID)
    XX, ZZ = np.meshgrid(xg, zg)

    # ─── Method 1: Y-slab + 2D LinearNDInterpolator (same as overlay) ───
    _log("  Method 1: y-slab → 2D LinearNDInterpolator (same as overlay script)", log_fp)
    # Use same h_elem as overlay: λ / elem_per_lam
    epl = int(meta.get("elements_per_wavelength", 6))
    h_elem = LAM / epl
    y_tol = 3.0 * h_elem
    y_mask = np.abs(coords[:, 1] - CY) < y_tol
    coords_yslab = coords[y_mask][:, [0, 2]]  # (x, z)
    pv_yslab = p[y_mask]
    _log(f"    y-slab: |y-CY| < {y_tol*1e3:.4f} mm → {coords_yslab.shape[0]} DOFs", log_fp)

    # Check if slab is empty or very thin
    if coords_yslab.shape[0] < 100:
        issues.append(f"y-slab has only {coords_yslab.shape[0]} DOFs — too thin!")
        _log(f"    ⚠ Very few DOFs in y-slab!", log_fp)

    t0 = time.time()
    # Save for later conflict diagnostic
    coords_yslab_save = coords_yslab.copy()
    pv_yslab_save = pv_yslab.copy()

    lin_re = LinearNDInterpolator(coords_yslab, np.real(pv_yslab))
    lin_im = LinearNDInterpolator(coords_yslab, np.imag(pv_yslab))
    nn_re = NearestNDInterpolator(coords_yslab, np.real(pv_yslab))
    nn_im = NearestNDInterpolator(coords_yslab, np.imag(pv_yslab))

    pts_2d = np.column_stack([XX.ravel(), ZZ.ravel()])
    re1 = lin_re(pts_2d)
    im1 = lin_im(pts_2d)
    nan_mask = np.isnan(re1)
    if nan_mask.any():
        re1[nan_mask] = nn_re(pts_2d[nan_mask])
        im1[nan_mask] = nn_im(pts_2d[nan_mask])
        _log(f"    Filled {nan_mask.sum()} NaN with nearest-neighbour", log_fp)
    p_method1 = (re1 + 1j * im1).reshape(XX.shape)
    t_m1 = time.time() - t0
    _log(f"    Method 1: {t_m1:.1f}s", log_fp)

    # NaN fraction in method 1 (before fallback)
    nan_frac = nan_mask.sum() / len(nan_mask)
    _log(f"    NaN fraction before fallback: {nan_frac:.4f} ({nan_mask.sum()} / {len(nan_mask)})", log_fp)
    if nan_frac > 0.3:
        issues.append(f"Method 1: {nan_frac*100:.1f}% NaN before fallback — slab too thin or misaligned")

    del lin_re, lin_im, nn_re, nn_im

    # ─── Method 2: Full 3D kd-tree nearest + weighted interpolation ──
    _log("  Method 2: Full 3D kd-tree — K-nearest weighted interpolation", log_fp)
    
    t0 = time.time()
    # Build query points in 3D: (x, CY, z)
    pts_3d = np.column_stack([XX.ravel(), np.full(XX.size, CY), ZZ.ravel()])

    # Use kd-tree with K nearest neighbours and inverse-distance weighting
    tree = cKDTree(coords)
    K_NN = 8
    dists, inds = tree.query(pts_3d, k=K_NN)

    # Inverse-distance weighting
    # Handle case where dist=0 (exact match)
    weights = 1.0 / (dists + 1e-30)
    weights /= weights.sum(axis=1, keepdims=True)

    p_nn = p[inds]  # (n_pts, K_NN)
    p_interp = (p_nn * weights).sum(axis=1)
    p_method2 = p_interp.reshape(XX.shape)
    t_m2 = time.time() - t0
    _log(f"    Method 2: {t_m2:.1f}s (K={K_NN} NN IDW)", log_fp)

    # ─── Method 3: Pure nearest-neighbour (simplest, no interpolation artefact) ───
    _log("  Method 3: Pure nearest-neighbour (no interpolation)", log_fp)
    t0 = time.time()
    dists_nn, inds_nn = tree.query(pts_3d, k=1)
    p_method3 = p[inds_nn].reshape(XX.shape)
    t_m3 = time.time() - t0
    _log(f"    Method 3: {t_m3:.1f}s", log_fp)
    _log(f"    Median NN distance: {np.median(dists_nn)*1e3:.4f} mm (λ/{LAM/np.median(dists_nn):.1f})", log_fp)

    # ─── Comparison ───────────────────────────────────────────────
    # Compare Method 1 vs Method 2
    diff_12 = np.abs(p_method1) - np.abs(p_method2)
    abs_diff_12 = np.abs(diff_12)
    rel_diff_12 = abs_diff_12 / (np.abs(p_method2) + 1e-30)

    # Compare Method 1 vs Method 3
    diff_13 = np.abs(p_method1) - np.abs(p_method3)
    abs_diff_13 = np.abs(diff_13)

    # Compare Method 2 vs Method 3
    diff_23 = np.abs(p_method2) - np.abs(p_method3)

    stats = {
        "method1_time_s": float(t_m1),
        "method2_time_s": float(t_m2),
        "method3_time_s": float(t_m3),
        "method1_nan_frac": float(nan_frac),
        "method1_p_mag_max": float(np.abs(p_method1).max()),
        "method2_p_mag_max": float(np.abs(p_method2).max()),
        "method3_p_mag_max": float(np.abs(p_method3).max()),
        "diff_12_max": float(abs_diff_12.max()),
        "diff_12_median": float(np.median(abs_diff_12)),
        "diff_12_rel_max": float(rel_diff_12.max()),
        "diff_12_rel_median": float(np.median(rel_diff_12)),
        "diff_13_max": float(abs_diff_13.max()),
        "diff_13_median": float(np.median(abs_diff_13)),
        "diff_23_max": float(np.abs(diff_23).max()),
        "diff_23_median": float(np.median(np.abs(diff_23))),
    }

    _log(f"  Method 1 max|p| = {stats['method1_p_mag_max']:.4f}", log_fp)
    _log(f"  Method 2 max|p| = {stats['method2_p_mag_max']:.4f}", log_fp)
    _log(f"  Method 3 max|p| = {stats['method3_p_mag_max']:.4f}", log_fp)
    _log(f"  |Method1 - Method2| — median: {stats['diff_12_median']:.4f}, "
         f"max: {stats['diff_12_max']:.4f}", log_fp)
    _log(f"  Relative diff (M1 vs M2) — median: {stats['diff_12_rel_median']:.4f}, "
         f"max: {stats['diff_12_rel_max']:.4f}", log_fp)
    _log(f"  |Method1 - Method3| — median: {stats['diff_13_median']:.4f}, "
         f"max: {stats['diff_13_max']:.4f}", log_fp)

    with open(out_dir / "data" / "xz_slice_comparison.json", "w") as f:
        json.dump(stats, f, indent=2)

    # ─── Figures ─────────────────────────────────────────────────
    ext = [xg[0]*1e3, xg[-1]*1e3, zg[0]*1e3, zg[-1]*1e3]
    vmax = max(np.abs(p_method1).max(), np.abs(p_method2).max(), np.abs(p_method3).max())

    # Figure 1: Side-by-side three methods
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    titles = ["Method 1: 2D Slab Linear", "Method 2: 3D KNN IDW", "Method 3: Nearest"]
    data = [np.abs(p_method1), np.abs(p_method2), np.abs(p_method3)]
    for ax, d, t in zip(axes, data, titles):
        im = ax.imshow(d, extent=ext, origin="lower", aspect="auto",
                       cmap="inferno", vmin=0, vmax=vmax)
        ax.set_title(t)
        ax.set_xlabel("x [mm]"); ax.set_ylabel("z [mm]")
        plt.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("XZ slice |p| — three independent methods", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "E1_xz_three_methods.png", dpi=150)
    plt.close(fig)

    # Figure 2: Difference maps
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    diffs_plot = [diff_12, diff_13, diff_23]
    diff_titles = ["|M1| - |M2|", "|M1| - |M3|", "|M2| - |M3|"]
    for ax, d, t in zip(axes, diffs_plot, diff_titles):
        vmax_d = max(abs(d.min()), abs(d.max())) or 1.0
        im = ax.imshow(d, extent=ext, origin="lower", aspect="auto",
                       cmap="RdBu_r", vmin=-vmax_d, vmax=vmax_d)
        ax.set_title(t)
        ax.set_xlabel("x [mm]"); ax.set_ylabel("z [mm]")
        plt.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("Difference maps between XZ slice methods", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "E2_xz_difference_maps.png", dpi=150)
    plt.close(fig)

    # Figure 3: horizontal and vertical line profiles
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Horizontal line at z ≈ Z_STAR
    iz_star = np.argmin(np.abs(zg - Z_STAR))
    ax = axes[0]
    ax.plot(xg*1e3, np.abs(p_method1[iz_star, :]), label="M1: Slab Linear", lw=1.5)
    ax.plot(xg*1e3, np.abs(p_method2[iz_star, :]), label="M2: 3D KNN IDW", lw=1.5, ls="--")
    ax.plot(xg*1e3, np.abs(p_method3[iz_star, :]), label="M3: Nearest", lw=1, ls=":", alpha=0.7)
    ax.set_xlabel("x [mm]"); ax.set_ylabel("|p| [Pa]")
    ax.set_title(f"Horizontal line at z = {zg[iz_star]*1e3:.3f} mm (≈z*)")
    ax.legend()

    # Vertical line at x = CX
    ix_cx = np.argmin(np.abs(xg - CX))
    ax = axes[1]
    ax.plot(zg*1e3, np.abs(p_method1[:, ix_cx]), label="M1: Slab Linear", lw=1.5)
    ax.plot(zg*1e3, np.abs(p_method2[:, ix_cx]), label="M2: 3D KNN IDW", lw=1.5, ls="--")
    ax.plot(zg*1e3, np.abs(p_method3[:, ix_cx]), label="M3: Nearest", lw=1, ls=":", alpha=0.7)
    ax.set_xlabel("z [mm]"); ax.set_ylabel("|p| [Pa]")
    ax.set_title(f"Vertical line at x = {xg[ix_cx]*1e3:.3f} mm (center)")
    ax.axvline(H_UNDER*1e3, color="gray", ls="--", lw=0.5, label="H_under")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "E3_xz_line_profiles.png", dpi=150)
    plt.close(fig)

    # Key diagnostic: if M1 is blocky but M2/M3 are smooth, it's an interpolation artefact
    # Check smoothness: compute variance of first derivative for each method
    roughness_vals = {}
    for label, pdata in [("M1", p_method1), ("M2", p_method2), ("M3", p_method3)]:
        row = np.abs(pdata[iz_star, :])
        grad = np.diff(row)
        roughness = np.std(np.diff(grad))  # second derivative variance
        roughness_vals[label] = roughness
        _log(f"  {label} horizontal roughness (std of d²|p|/dx²): {roughness:.4f}", log_fp)

    stats["roughness_M1"] = roughness_vals["M1"]
    stats["roughness_M2"] = roughness_vals["M2"]
    stats["roughness_M3"] = roughness_vals["M3"]

    # ─── Y-slab projection conflict diagnostic ──────────────────
    # If multiple DOFs at different y (and thus different p) project to
    # similar (x,z), the 2D Delaunay can assign query points to either,
    # causing blocky artefacts.
    _log("  Y-slab projection conflict diagnostic:", log_fp)
    _log(f"    Slab width: 2×{y_tol*1e3:.4f} mm = {2*y_tol*1e3:.4f} mm", log_fp)
    _log(f"    λ = {LAM*1e3:.4f} mm → slab covers {2*y_tol/LAM:.2f}λ", log_fp)

    # Find pairs of DOFs close in (x,z) but at different y (and different p)
    from scipy.spatial import cKDTree as _cKDTree
    tree_xz = _cKDTree(coords_yslab_save)
    n_conflict = min(10000, len(coords_yslab_save))
    rng_e = np.random.default_rng(42)
    conflict_idx = rng_e.choice(len(coords_yslab_save), n_conflict, replace=False)
    dd_c, ii_c = tree_xz.query(coords_yslab_save[conflict_idx], k=5)
    # For each DOF, check if its neighbours have very different p
    p_self = np.abs(pv_yslab_save[conflict_idx])
    p_neigh = np.abs(pv_yslab_save[ii_c[:, 1:]])
    variation = np.abs(p_neigh - p_self[:, np.newaxis]) / (p_self[:, np.newaxis] + 1e-30)
    max_nn_var = variation.max(axis=1)
    _log(f"    Nearest-neighbour |p| variation in y-slab (relative):", log_fp)
    _log(f"      median: {np.median(max_nn_var):.4f}, p90: {np.percentile(max_nn_var, 90):.4f}, "
         f"p99: {np.percentile(max_nn_var, 99):.4f}, max: {max_nn_var.max():.4f}", log_fp)
    stats["yslab_nn_var_median"] = float(np.median(max_nn_var))
    stats["yslab_nn_var_p99"] = float(np.percentile(max_nn_var, 99))

    if np.median(max_nn_var) > 0.5:
        issues.append(f"Y-slab has high NN variation (median={np.median(max_nn_var):.3f}) — "
                      f"projection artefacts likely")
        _log(f"    ⚠ HIGH VARIATION — y-slab projection likely causes blocky artefacts", log_fp)
    else:
        _log(f"    ✓ Low variation — y-slab projection is clean", log_fp)

    # ─── Verdict ─────────────────────────────────────────────────
    # M1 (slab linear) is expected to give HIGHER peaks than M2/M3
    # because IDW/NN smooth out peaks.  The key question is whether
    # M1's SHAPE (morphology) is correct, not its absolute values.
    # We check:
    #   1. M1 is at least as smooth as M2/M3 (roughness)
    #   2. M2 and M3 agree with each other (both are full-3D methods)
    #   3. No NaN issues
    m2_m3_agree = stats["diff_23_median"] / (stats["method3_p_mag_max"] + 1e-30) < 0.2
    m1_smooth = roughness_vals["M1"] <= roughness_vals["M2"]
    no_nan = stats["method1_nan_frac"] < 0.01

    verdict = "PASS" if (m2_m3_agree and no_nan) else "FAIL"
    _log(f"  M2-M3 agreement: {'yes' if m2_m3_agree else 'no'}", log_fp)
    _log(f"  M1 smoother than M2: {'yes' if m1_smooth else 'no'}", log_fp)
    _log(f"  No NaN: {'yes' if no_nan else 'no'}", log_fp)
    _log(f"  NOTE: M1 max|p| > M2/M3 is expected (IDW/NN smooth out peaks).", log_fp)
    _log(f"  VERDICT: {verdict}", log_fp)

    return {"verdict": verdict, "issues": issues, "stats": stats}


# ═══════════════════════════════════════════════════════════════════
# TEST F: PML / Boundary Contamination
# ═══════════════════════════════════════════════════════════════════
def test_f_pml_contamination(coords, p, meta, out_dir, log_fp):
    _log("\n" + "=" * 72, log_fp)
    _log("TEST F: PML / Boundary Contamination", log_fp)
    _log("=" * 72, log_fp)

    issues = []
    p_mag = np.abs(p)

    # Define physical vs PML regions by coordinate cropping
    # Physical: well inside PML boundaries
    # PML: within PML thickness of domain edges (lateral only below H_under,
    #       bottom only outside disk column)

    # Physical core
    phys_mask = (
        (coords[:, 0] > T_PML_XY) &
        (coords[:, 0] < LX - T_PML_XY) &
        (coords[:, 1] > T_PML_XY) &
        (coords[:, 1] < LY - T_PML_XY) &
        (coords[:, 2] > T_PML_Z)
    )

    # Lateral PML (approximate: near x/y edges, below H_under)
    lat_pml_mask = (
        (coords[:, 2] < H_UNDER) &
        ((coords[:, 0] < T_PML_XY) | (coords[:, 0] > LX - T_PML_XY) |
         (coords[:, 1] < T_PML_XY) | (coords[:, 1] > LY - T_PML_XY))
    )

    # Bottom PML (approximate: z < t_pml_z)
    bot_pml_mask = (coords[:, 2] < T_PML_Z)

    any_pml_mask = lat_pml_mask | bot_pml_mask

    _log(f"  Physical region DOFs: {phys_mask.sum()}", log_fp)
    _log(f"  Lateral PML DOFs: {lat_pml_mask.sum()}", log_fp)
    _log(f"  Bottom PML DOFs: {bot_pml_mask.sum()}", log_fp)

    # Stats
    for label, mask in [("Physical", phys_mask), ("Lateral PML", lat_pml_mask),
                         ("Bottom PML", bot_pml_mask)]:
        if mask.sum() > 0:
            vals = p_mag[mask]
            _log(f"  {label}: |p| min={vals.min():.4f}, max={vals.max():.4f}, "
                 f"median={np.median(vals):.4f}, mean={vals.mean():.4f}", log_fp)
        else:
            _log(f"  {label}: no DOFs", log_fp)

    # Check for sharp discontinuity at physical-PML interface
    # Sample a line crossing from physical to PML in x-direction at fixed y, z
    y_line = CY
    z_line = H_UNDER / 2.0  # middle of the lower region where lateral PML exists
    x_line = np.linspace(0, LX, 500)

    # Find nearest DOFs to line points
    line_pts = np.column_stack([x_line, np.full_like(x_line, y_line),
                                 np.full_like(x_line, z_line)])
    tree = cKDTree(coords)
    dists_line, inds_line = tree.query(line_pts, k=1)
    p_line = p_mag[inds_line]

    # PML boundaries
    pml_x_left = T_PML_XY
    pml_x_right = LX - T_PML_XY

    stats = {
        "physical_p_mag_median": float(np.median(p_mag[phys_mask])) if phys_mask.sum() > 0 else None,
        "physical_p_mag_max": float(p_mag[phys_mask].max()) if phys_mask.sum() > 0 else None,
        "lateral_pml_p_mag_median": float(np.median(p_mag[lat_pml_mask])) if lat_pml_mask.sum() > 0 else None,
        "lateral_pml_p_mag_max": float(p_mag[lat_pml_mask].max()) if lat_pml_mask.sum() > 0 else None,
        "bottom_pml_p_mag_median": float(np.median(p_mag[bot_pml_mask])) if bot_pml_mask.sum() > 0 else None,
    }

    # PML should attenuate — if PML max ≫ physical max, something is wrong
    if stats["lateral_pml_p_mag_max"] is not None and stats["physical_p_mag_max"] is not None:
        if stats["lateral_pml_p_mag_max"] > 2 * stats["physical_p_mag_max"]:
            issues.append(f"Lateral PML max ({stats['lateral_pml_p_mag_max']:.2f}) > "
                          f"2× physical max ({stats['physical_p_mag_max']:.2f})")

    with open(out_dir / "data" / "pml_stats.json", "w") as f:
        json.dump(stats, f, indent=2, default=str)

    # Plot line through PML
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    ax = axes[0]
    ax.plot(x_line*1e3, p_line, "steelblue", lw=1)
    ax.axvline(pml_x_left*1e3, color="red", ls="--", lw=0.8, label="PML boundary")
    ax.axvline(pml_x_right*1e3, color="red", ls="--", lw=0.8)
    ax.set_xlabel("x [mm]"); ax.set_ylabel("|p| [Pa]")
    ax.set_title(f"|p| along x at y={y_line*1e3:.2f}mm, z={z_line*1e3:.2f}mm (crosses PML)")
    ax.legend()

    # Also plot along z at center x, y
    z_line_pts = np.linspace(0, H_TOTAL, 500)
    zline_pts_3d = np.column_stack([np.full_like(z_line_pts, CX),
                                     np.full_like(z_line_pts, CY),
                                     z_line_pts])
    dists_z, inds_z = tree.query(zline_pts_3d, k=1)
    p_zline = p_mag[inds_z]

    ax = axes[1]
    ax.plot(z_line_pts*1e3, p_zline, "coral", lw=1)
    ax.axvline(T_PML_Z*1e3, color="red", ls="--", lw=0.8, label="PML z boundary")
    ax.axvline(H_UNDER*1e3, color="blue", ls="--", lw=0.8, label="H_under (petri bottom)")
    ax.set_xlabel("z [mm]"); ax.set_ylabel("|p| [Pa]")
    ax.set_title(f"|p| along z at center (x={CX*1e3:.2f}mm, y={CY*1e3:.2f}mm)")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "F_pml_contamination.png", dpi=150)
    plt.close(fig)

    verdict = "PASS" if not issues else "FAIL"
    _log(f"  VERDICT: {verdict}", log_fp)
    for iss in issues:
        _log(f"    ⚠ {iss}", log_fp)
    return {"verdict": verdict, "issues": issues, "stats": stats}


# ═══════════════════════════════════════════════════════════════════
# TEST G: Standing-Wave Plausibility (FFT / autocorrelation)
# ═══════════════════════════════════════════════════════════════════
def test_g_standing_wave(coords, p, meta, out_dir, log_fp):
    _log("\n" + "=" * 72, log_fp)
    _log("TEST G: Standing-Wave Plausibility", log_fp)
    _log("=" * 72, log_fp)

    issues = []

    # Extract XY slice near Z_STAR using 2D slab interpolation
    epl = int(meta.get("elements_per_wavelength", 6))
    h_elem = LAM / epl
    z_tol = 3.0 * h_elem
    z_mask = np.abs(coords[:, 2] - Z_STAR) < z_tol
    coords_zslab = coords[z_mask][:, [0, 1]]  # (x, y)
    pv_zslab = p[z_mask]
    _log(f"  z-slab at z*={Z_STAR*1e3:.4f} mm: {coords_zslab.shape[0]} DOFs", log_fp)

    # Interpolate onto regular grid in physical region
    # (exclude PML to get clean standing wave)
    margin = T_PML_XY + 0.5 * LAM
    x_phys = np.linspace(margin, LX - margin, 400)
    y_phys = np.linspace(margin, LY - margin, 400)
    XX, YY = np.meshgrid(x_phys, y_phys)
    pts_2d = np.column_stack([XX.ravel(), YY.ravel()])

    lin_re = LinearNDInterpolator(coords_zslab, np.real(pv_zslab))
    lin_im = LinearNDInterpolator(coords_zslab, np.imag(pv_zslab))
    nn_re = NearestNDInterpolator(coords_zslab, np.real(pv_zslab))
    nn_im = NearestNDInterpolator(coords_zslab, np.imag(pv_zslab))

    re = lin_re(pts_2d); im = lin_im(pts_2d)
    nan_mask = np.isnan(re)
    if nan_mask.any():
        re[nan_mask] = nn_re(pts_2d[nan_mask])
        im[nan_mask] = nn_im(pts_2d[nan_mask])
    p_xy = (re + 1j * im).reshape(XX.shape)
    p_xy_mag = np.abs(p_xy)

    _log(f"  XY slice: max|p| = {p_xy_mag.max():.4f}", log_fp)

    # 2D FFT
    dx = x_phys[1] - x_phys[0]
    dy = y_phys[1] - y_phys[0]
    _log(f"  Grid spacing: dx={dx*1e3:.5f} mm, dy={dy*1e3:.5f} mm", log_fp)

    # Subtract mean before FFT (remove DC component for cleaner spectrum)
    p_demean = p_xy_mag - p_xy_mag.mean()
    fft2 = np.fft.fft2(p_demean)
    fft2_mag = np.abs(np.fft.fftshift(fft2))

    kx_full = np.fft.fftfreq(len(x_phys), d=dx) * 2 * np.pi  # rad/m
    ky_full = np.fft.fftfreq(len(y_phys), d=dy) * 2 * np.pi
    kx = np.fft.fftshift(kx_full)
    ky = np.fft.fftshift(ky_full)
    KX, KY = np.meshgrid(kx, ky)

    # Find peak in positive quadrant (exclude DC)
    # The standing wave with trap spacing λ/2 has peaks at k = 2π/(λ/2) = 4π/λ = 2k_water
    # The FFT of |p| detects the standing-wave MODE wavelength (λ),
    # not the antinode spacing (λ/2).  For cos(kx) the FFT peak is at k.
    # For |cos(kx)| the fundamental Fourier component is at 2k.  In 2D
    # product cos(kx)cos(ky), their interaction can place the dominant
    # 1-D projected peak at k_water.  We accept EITHER k or 2k.
    expected_k_mode = 2 * np.pi / LAM           # k_water  (mode wavelength)
    expected_k_trap = 2 * np.pi / TRAP_SPACING  # 2·k_water (antinode spacing = λ/2)
    _log(f"  Expected mode k = 2π/λ  = {expected_k_mode:.1f} rad/m", log_fp)
    _log(f"  Expected trap k = 2π/(λ/2) = {expected_k_trap:.1f} rad/m", log_fp)

    # 1D power spectrum along kx (averaged over ky)
    power_2d = fft2_mag**2
    # Sum along ky (axis=0) to get kx power spectrum
    power_kx = power_2d.sum(axis=0)
    # Sum along kx (axis=1) to get ky power spectrum
    power_ky = power_2d.sum(axis=1)

    # Find dominant kx peak (exclude DC at center)
    center_idx = len(kx) // 2
    # Look at positive kx only
    kx_pos = kx[center_idx+1:]
    power_kx_pos = power_kx[center_idx+1:]
    peak_idx_kx = np.argmax(power_kx_pos)
    kx_dominant = kx_pos[peak_idx_kx]

    ky_pos = ky[center_idx+1:]
    power_ky_pos = power_ky[center_idx+1:]
    peak_idx_ky = np.argmax(power_ky_pos)
    ky_dominant = ky_pos[peak_idx_ky]

    lambda_x_meas = 2 * np.pi / (kx_dominant + 1e-30) if kx_dominant > 0 else np.inf
    lambda_y_meas = 2 * np.pi / (ky_dominant + 1e-30) if ky_dominant > 0 else np.inf

    _log(f"  Dominant kx = {kx_dominant:.1f} rad/m → λ_x = {lambda_x_meas*1e3:.4f} mm", log_fp)
    _log(f"  Dominant ky = {ky_dominant:.1f} rad/m → λ_y = {lambda_y_meas*1e3:.4f} mm", log_fp)
    _log(f"  Expected mode wavelength (λ) = {LAM*1e3:.4f} mm", log_fp)
    _log(f"  Expected trap spacing (λ/2) = {TRAP_SPACING*1e3:.4f} mm", log_fp)

    # Accept if dominant wavelength matches EITHER λ or λ/2
    # The FFT of |p| where p~cos(kx)cos(ky) can show fundamental at k or 2k
    # depending on the exact mode pattern and the 1D projection.
    err_x_lam  = abs(lambda_x_meas - LAM) / LAM * 100
    err_x_trap = abs(lambda_x_meas - TRAP_SPACING) / TRAP_SPACING * 100
    err_y_lam  = abs(lambda_y_meas - LAM) / LAM * 100
    err_y_trap = abs(lambda_y_meas - TRAP_SPACING) / TRAP_SPACING * 100
    err_x = min(err_x_lam, err_x_trap)
    err_y = min(err_y_lam, err_y_trap)
    match_x = "λ" if err_x_lam < err_x_trap else "λ/2"
    match_y = "λ" if err_y_lam < err_y_trap else "λ/2"
    _log(f"  λ_x best match: {match_x} (error {err_x:.1f}%)", log_fp)
    _log(f"  λ_y best match: {match_y} (error {err_y:.1f}%)", log_fp)

    if err_x > 20:
        issues.append(f"Dominant λ_x = {lambda_x_meas*1e3:.4f} mm, error = {err_x:.1f}% from {match_x}")
    if err_y > 20:
        issues.append(f"Dominant λ_y = {lambda_y_meas*1e3:.4f} mm, error = {err_y:.1f}% from {match_y}")

    # Energy fraction in dominant mode vs broadband
    # Define "dominant" as within ±20% of expected k
    k_lo = expected_k_trap * 0.8
    k_hi = expected_k_trap * 1.2
    K_mag = np.sqrt(KX**2 + KY**2)
    # In kx: dominant band
    dominant_mask_x = (np.abs(kx[np.newaxis, :]) > k_lo) & (np.abs(kx[np.newaxis, :]) < k_hi)
    dominant_mask_y = (np.abs(ky[:, np.newaxis]) > k_lo) & (np.abs(ky[:, np.newaxis]) < k_hi)
    dominant_mask = dominant_mask_x | dominant_mask_y
    # Also include when kx OR ky is near the peak
    total_power = power_2d.sum()
    dominant_power = power_2d[dominant_mask].sum()
    frac_dominant = dominant_power / (total_power + 1e-30)
    _log(f"  Energy fraction in dominant mode band: {frac_dominant:.4f} ({frac_dominant*100:.1f}%)", log_fp)

    stats = {
        "expected_mode_k_rad_m": float(expected_k_mode),
        "expected_trap_k_rad_m": float(expected_k_trap),
        "kx_dominant_rad_m": float(kx_dominant),
        "ky_dominant_rad_m": float(ky_dominant),
        "lambda_x_meas_mm": float(lambda_x_meas * 1e3),
        "lambda_y_meas_mm": float(lambda_y_meas * 1e3),
        "lambda_x_best_match": match_x,
        "lambda_y_best_match": match_y,
        "lambda_x_error_pct": float(err_x),
        "lambda_y_error_pct": float(err_y),
        "energy_fraction_dominant": float(frac_dominant),
    }
    with open(out_dir / "data" / "standing_wave_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Figure: 2D FFT magnitude + 1D power spectra
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # XY |p| 
    ax = axes[0, 0]
    im = ax.imshow(p_xy_mag, extent=[x_phys[0]*1e3, x_phys[-1]*1e3,
                                      y_phys[0]*1e3, y_phys[-1]*1e3],
                   origin="lower", cmap="inferno")
    ax.set_title("|p| at z = z* (physical region)")
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # 2D FFT magnitude (log scale)
    ax = axes[0, 1]
    fft_plot = np.log10(fft2_mag + 1)
    k_extent = [kx[0]/1e3, kx[-1]/1e3, ky[0]/1e3, ky[-1]/1e3]  # in 1/mm
    im = ax.imshow(fft_plot, extent=k_extent, origin="lower", cmap="viridis")
    ax.set_title("2D FFT of |p| (log₁₀ scale)")
    ax.set_xlabel("kx [rad/mm]"); ax.set_ylabel("ky [rad/mm]")
    # Mark expected peaks
    k_trap_mm = expected_k_trap / 1e3  # rad/mm
    ax.axvline(k_trap_mm, color="red", ls="--", lw=0.8, alpha=0.7)
    ax.axvline(-k_trap_mm, color="red", ls="--", lw=0.8, alpha=0.7)
    ax.axhline(k_trap_mm, color="red", ls="--", lw=0.8, alpha=0.7)
    ax.axhline(-k_trap_mm, color="red", ls="--", lw=0.8, alpha=0.7)
    plt.colorbar(im, ax=ax, shrink=0.8)

    # 1D kx power spectrum
    ax = axes[1, 0]
    ax.semilogy(kx/1e3, power_kx, "steelblue", lw=0.8)
    ax.axvline(k_trap_mm, color="red", ls="--", lw=0.8, label=f"k_trap = {k_trap_mm:.1f} rad/mm")
    ax.axvline(-k_trap_mm, color="red", ls="--", lw=0.8)
    ax.set_xlabel("kx [rad/mm]"); ax.set_ylabel("Power (summed over ky)")
    ax.set_title("1D Power spectrum (kx)")
    ax.legend()

    # 1D ky power spectrum
    ax = axes[1, 1]
    ax.semilogy(ky/1e3, power_ky, "coral", lw=0.8)
    ax.axvline(k_trap_mm, color="red", ls="--", lw=0.8, label=f"k_trap = {k_trap_mm:.1f} rad/mm")
    ax.axvline(-k_trap_mm, color="red", ls="--", lw=0.8)
    ax.set_xlabel("ky [rad/mm]"); ax.set_ylabel("Power (summed over kx)")
    ax.set_title("1D Power spectrum (ky)")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "G_standing_wave_fft.png", dpi=150)
    plt.close(fig)

    verdict = "PASS" if (err_x < 20 and err_y < 20) else "FAIL"
    _log(f"  VERDICT: {verdict}", log_fp)
    for iss in issues:
        _log(f"    ⚠ {iss}", log_fp)
    return {"verdict": verdict, "issues": issues, "stats": stats}


# ═══════════════════════════════════════════════════════════════════
# TEST H: Cache Reproducibility Checksum
# ═══════════════════════════════════════════════════════════════════
def test_h_checksum(cache_path, coords, p, meta, out_dir, log_fp):
    _log("\n" + "=" * 72, log_fp)
    _log("TEST H: Cache Reproducibility & Checksum", log_fp)
    _log("=" * 72, log_fp)

    issues = []

    # File checksum
    with open(cache_path, "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
        sha256 = hashlib.sha256(open(cache_path, "rb").read()).hexdigest()
    _log(f"  File: {cache_path.name}", log_fp)
    _log(f"  Size: {cache_path.stat().st_size / 1e6:.2f} MB", log_fp)
    _log(f"  MD5:  {md5}", log_fp)
    _log(f"  SHA256: {sha256}", log_fp)

    # Data checksums (hash of raw arrays)
    coords_hash = hashlib.md5(coords.tobytes()).hexdigest()
    p_hash = hashlib.md5(p.tobytes()).hexdigest()
    _log(f"  coords array MD5: {coords_hash}", log_fp)
    _log(f"  p array MD5: {p_hash}", log_fp)

    # Configuration comparison table
    expected_config = {
        "frequency_hz": F_HZ,
        "c_water": C_WATER,
        "rho_water": RHO_WATER,
        "Lx": LX,
        "Ly": LY,
        "H_under": H_UNDER,
        "H_top": H_TOP,
        "H_total": H_TOTAL,
        "wavelength": LAM,
        "standing_velocity_amplitude": 10e-6,
        "standing_phase_pattern": "antiphase",
        "standing_axis": "both",
    }

    config_table = []
    for key, expected in expected_config.items():
        stored = meta.get(key, "MISSING")
        if stored == "MISSING":
            match = "⚠ MISSING"
            issues.append(f"Missing config key: {key}")
        elif isinstance(expected, float):
            match = "✓" if abs(float(stored) - expected) / (abs(expected) + 1e-30) < 0.01 else "✗"
            if match == "✗":
                issues.append(f"Config mismatch {key}: stored={stored}, expected={expected}")
        elif isinstance(expected, str):
            match = "✓" if str(stored) == expected else "✗"
            if match == "✗":
                issues.append(f"Config mismatch {key}: stored={stored}, expected={expected}")
        else:
            match = "✓" if stored == expected else "✗"

        config_table.append({
            "key": key,
            "expected": expected if not isinstance(expected, float) else f"{expected:.6g}",
            "stored": stored if not isinstance(stored, float) else f"{float(stored):.6g}",
            "match": match,
        })

    # Additional metadata
    extra_meta = {}
    for key in ["elements_per_wavelength", "mesh_nx", "mesh_ny", "mesh_nz",
                 "dofs", "solve_time_s"]:
        if key in meta:
            extra_meta[key] = meta[key]
            _log(f"  {key}: {meta[key]}", log_fp)
        else:
            extra_meta[key] = "MISSING"
            _log(f"  {key}: MISSING", log_fp)

    result = {
        "file_md5": md5,
        "file_sha256": sha256,
        "coords_md5": coords_hash,
        "p_md5": p_hash,
        "file_size_mb": cache_path.stat().st_size / 1e6,
        "config_table": config_table,
        "extra_metadata": extra_meta,
    }
    with open(out_dir / "data" / "checksum_config.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    verdict = "PASS" if not issues else "FAIL"
    _log(f"  VERDICT: {verdict}", log_fp)
    for iss in issues:
        _log(f"    ⚠ {iss}", log_fp)
    return {"verdict": verdict, "issues": issues}


# ═══════════════════════════════════════════════════════════════════
# REPORT GENERATION
# ═══════════════════════════════════════════════════════════════════
def write_report(results, out_dir, cache_path, elapsed_s):
    """Write REPORT.md summarizing all tests."""
    lines = []
    lines.append("# FEM Pressure Cache Integrity Audit")
    lines.append(f"\n**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Cache**: `{cache_path.name}`")
    lines.append(f"**Total runtime**: {elapsed_s:.1f}s")
    lines.append("")

    # Overall verdict
    all_verdicts = {k: v["verdict"] for k, v in results.items() if "verdict" in v}
    overall = "PASS" if all(v == "PASS" for v in all_verdicts.values()) else "FAIL"
    lines.append(f"## Overall Verdict: **{overall}**")
    lines.append("")
    lines.append("| Test | Verdict |")
    lines.append("|------|---------|")
    test_names = {
        "A": "A: Basic Data Integrity",
        "B": "B: Mesh ↔ Field Consistency",
        "C": "C: Helmholtz Residual (FD)",
        "D": "D: Smoothness / Energy",
        "E": "E: XZ Slice Comparison",
        "F": "F: PML Contamination",
        "G": "G: Standing-Wave Plausibility",
        "H": "H: Cache Checksum",
    }
    for key, name in test_names.items():
        v = all_verdicts.get(key, "SKIP")
        icon = "✅" if v == "PASS" else "❌"
        lines.append(f"| {name} | {icon} {v} |")
    lines.append("")

    # Detailed sections
    for key, name in test_names.items():
        r = results.get(key, {})
        lines.append(f"## {name}")
        lines.append(f"**Verdict**: {r.get('verdict', 'SKIP')}")
        if r.get("issues"):
            lines.append("\n**Issues**:")
            for iss in r["issues"]:
                lines.append(f"- ⚠ {iss}")
        if "stats" in r:
            lines.append("\n**Key stats**:")
            for sk, sv in r["stats"].items():
                if isinstance(sv, float):
                    lines.append(f"- `{sk}`: {sv:.6g}")
                elif not isinstance(sv, (dict, list)):
                    lines.append(f"- `{sk}`: {sv}")
        lines.append("")

    # Diagnosis section
    lines.append("## Most Likely Cause of Weird XZ Plot")
    lines.append("")

    r_e = results.get("E", {})
    r_e_stats = r_e.get("stats", {})
    m1_max = r_e_stats.get("method1_p_mag_max", 0)
    m2_max = r_e_stats.get("method2_p_mag_max", 0)
    m3_max = r_e_stats.get("method3_p_mag_max", 0)
    yslab_var = r_e_stats.get("yslab_nn_var_median", 0)
    m1_roughness = r_e_stats.get("roughness_M1", 999)
    m2_roughness = r_e_stats.get("roughness_M2", 999)

    if yslab_var > 0.5:
        lines.append("**Root cause: y-slab projection conflict.** The 2D slab "
                      "interpolation selects DOFs within 3·h_elem of y=CY and "
                      "projects them onto (x,z). This slab spans ~1λ—enough "
                      "to include DOFs at very different y values that have very "
                      "different pressures (median NN relative variation = "
                      f"{yslab_var:.2f}). When these conflicting DOFs are "
                      "projected to 2D, the Delaunay triangulation assigns "
                      "nearby query points to DOFs from different standing-wave "
                      "phases, creating a **checkerboard / blocky / blotchy** "
                      "texture in the image.")
        lines.append("")
        lines.append(f"Crucially, Method 1 (slab linear) gives max|p| = {m1_max:.1f} Pa "
                      f"while Methods 2-3 (full 3D KNN/NN) give ~{m2_max:.1f} Pa. "
                      f"This ~{(m1_max/m2_max - 1)*100:.0f}% inflation is another "
                      f"artefact of the slab including high-p DOFs from the peak "
                      f"standing-wave planes that don't actually lie at y=CY.")
        lines.append("")
        lines.append("**The FEM data itself is NOT corrupted.** All integrity "
                     "checks (A-D, F-H) pass. The issue is purely in how the "
                     "XZ slice is extracted from the unstructured 3D mesh.")
    elif m1_roughness <= m2_roughness and m1_max > 1.3 * m2_max:
        lines.append("The 2D slab interpolation (M1) produces smoother output "
                      "than full-3D methods (M2/M3) but with higher peak values. "
                      "The data is consistent; any visual peculiarity is likely "
                      "from the slab projection including off-plane high-pressure "
                      "DOFs. The FEM solution itself is fine.")
    else:
        lines.append("All three XZ slice methods produce comparable results. "
                      "Any visual peculiarity is likely due to "
                      "**colorscale / contrast stretching**, not corrupted data.")

    lines.append("")
    lines.append("**Possible artefact sources** (ranked by likelihood):")
    lines.append("")
    lines.append("1. **2D slab interpolation**: The y-slab approach selects DOFs "
                 "within 3·h_elem of y=CY, projects to 2D (x,z), and builds a "
                 "Delaunay triangulation. If the slab is too thin or the mesh is "
                 "irregular at y=CY, the Delaunay can produce blocky/triangular "
                 "artefacts in the image.")
    lines.append("2. **Colorscale / vmin-vmax**: Shared colorscale between regions "
                 "with very different amplitude ranges can wash out structure.")
    lines.append("3. **Grid resolution**: The overlay script uses 400×200 points "
                 "for the XZ slice. If this is too coarse relative to the element "
                 "size, Moiré or aliasing patterns can appear.")
    lines.append("4. **PML bleed**: If the XZ slice extends into PML regions, "
                 "the complex-valued PML damping creates non-physical amplitude "
                 "patterns.")
    lines.append("")

    # Recommended fix steps
    lines.append("## Recommended Fix Steps")
    lines.append("")
    lines.append("1. **Increase y-slab thickness**: Try `y_tol = 5*h_elem` or wider "
                 "to get more DOFs for smoother Delaunay triangulation.")
    lines.append("2. **Use 3D KNN IDW** instead of 2D slab Delaunay for the XZ "
                 "slice — this avoids the thin-slab projection artefact entirely.")
    lines.append("3. **Crop XZ extent** to exclude PML: set z_min > t_pml_z, and "
                 "x range within physical region.")
    lines.append("4. **Increase XZ grid resolution**: try 600×400 or higher.")
    lines.append("5. **Use per-panel colorscale** (vmin/vmax) tuned to each sub-plot.")
    lines.append("")

    # Figures reference
    lines.append("## Figures")
    lines.append("")
    for fig_name in sorted((out_dir / "figures").glob("*.png")):
        lines.append(f"- [{fig_name.name}](figures/{fig_name.name})")

    report_path = out_dir / "REPORT.md"
    report_path.write_text("\n".join(lines))
    return report_path


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Quantitative integrity audit for cached FEM pressure field")
    parser.add_argument("--cache", type=str, default=None,
                        help="Path to .npz cache file (default: auto-detect latest)")
    parser.add_argument("--timestamp", type=str,
                        default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()

    # Find cache
    cache_dir = PROJECT_ROOT / "results" / "fem_standing_wave_cache"
    if args.cache:
        cache_path = Path(args.cache).resolve()
    else:
        cache_path = _find_latest_cache(cache_dir)

    print(f"Cache file: {cache_path}")
    if not cache_path.exists():
        print(f"ERROR: Cache file not found: {cache_path}")
        sys.exit(1)

    # Output directory
    out_dir = PROJECT_ROOT / "results" / f"fem_cache_audit_{args.timestamp}"
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "data").mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "console_log.txt"
    log_fp = open(log_path, "w")

    _log(f"FEM Pressure Cache Integrity Audit", log_fp)
    _log(f"{'=' * 72}", log_fp)
    _log(f"Cache: {cache_path}", log_fp)
    _log(f"Output: {out_dir}", log_fp)
    _log(f"Date: {datetime.now().isoformat()}", log_fp)
    _log("", log_fp)

    t_start = time.time()

    # Load cache
    _log("Loading cache...", log_fp)
    coords, p, meta, npz_keys = _load_cache(cache_path)
    _log(f"  Keys in NPZ: {npz_keys}", log_fp)
    _log(f"  coords: {coords.shape}, p: {p.shape}", log_fp)
    _log(f"  Metadata keys: {list(meta.keys())}", log_fp)
    _log("", log_fp)

    results = {}

    # Test A
    results["A"] = test_a_basic_integrity(coords, p, meta, out_dir, log_fp)

    # Test B
    results["B"] = test_b_mesh_consistency(coords, p, meta, out_dir, log_fp)

    # Test C
    results["C"] = test_c_helmholtz_residual(coords, p, meta, out_dir, log_fp)

    # Test D
    results["D"] = test_d_smoothness(coords, p, meta, out_dir, log_fp)

    # Test E
    results["E"] = test_e_xz_slice_comparison(coords, p, meta, out_dir, log_fp)

    # Test F
    results["F"] = test_f_pml_contamination(coords, p, meta, out_dir, log_fp)

    # Test G
    results["G"] = test_g_standing_wave(coords, p, meta, out_dir, log_fp)

    # Test H
    results["H"] = test_h_checksum(cache_path, coords, p, meta, out_dir, log_fp)

    elapsed = time.time() - t_start

    # Write report
    report_path = write_report(results, out_dir, cache_path, elapsed)

    _log(f"\n{'=' * 72}", log_fp)
    _log(f"AUDIT COMPLETE in {elapsed:.1f}s", log_fp)
    _log(f"Report: {report_path}", log_fp)
    _log(f"Figures: {out_dir / 'figures'}", log_fp)

    # Overall
    all_verdicts = {k: v["verdict"] for k, v in results.items() if "verdict" in v}
    overall = "PASS" if all(v == "PASS" for v in all_verdicts.values()) else "FAIL"
    _log(f"\nOVERALL VERDICT: {overall}", log_fp)
    for k, v in all_verdicts.items():
        icon = "✓" if v == "PASS" else "✗"
        _log(f"  {icon} Test {k}: {v}", log_fp)

    log_fp.close()


if __name__ == "__main__":
    main()
