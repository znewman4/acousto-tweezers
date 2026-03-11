#!/usr/bin/env python3
"""
Full-Domain Gorkov Trap Diagnostic
====================================
Same experiment as the C-shape study's trap detection, but on the
ENTIRE 6×6 mm epl5 FEM domain (single z-slice at z*).

Outputs:
  results/full_domain_gorkov_diagnostic_<TS>/
    full_domain_gorkov_traps.png   — 3-panel: |p|, U_gor, |p| with traps
    full_domain_trap_data.json     — trap coordinates, Hessian eigenvalues
    full_domain_fields.npz         — p_grid, U_gor, xg, yg (for reuse)
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import minimum_filter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ── Project root ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ═══════════════════════════════════════════════════════════════════
# Physical constants  (same as C-shape study)
# ═══════════════════════════════════════════════════════════════════
C_WATER = 1484.0
F_HZ    = 2.0e6
K_WATER = 2.0 * np.pi * F_HZ / C_WATER
LAM     = C_WATER / F_HZ            # 0.742 mm
TRAP_SP = LAM / 2.0                  # ~0.371 mm
OMEGA   = 2.0 * np.pi * F_HZ
RHO0    = 997.0

LX = LY   = 6.0e-3
H_UNDER   = 5.0e-3
H_TOP     = 2.0e-3
CX = CY   = LX / 2.0
Z_STAR    = H_UNDER + H_TOP / 2.0 + 0.25 * LAM   # ~6.1855 mm

K_IDW    = 16

STANDING_CACHE_DIR = PROJECT_ROOT / "results" / "fem_standing_wave_cache"
EPL5_NPZ = (STANDING_CACHE_DIR
            / "checkpoint_epl5_depth7mm_20260309_113007"
            / "standing_wave_epl5.npz")

# Full domain grid — 800×800 gives dx ≈ 7.5 µm  (~50 pts per λ/2)
N_GRID = 800

# Tiny margin to stay inside the FEM mesh
MARGIN = 50e-6   # 50 µm

# ── Output ──────────────────────────────────────────────────────────
TS = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = PROJECT_ROOT / "results" / f"full_domain_gorkov_diagnostic_{TS}"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
# Load & Interpolate
# ═══════════════════════════════════════════════════════════════════

def load_epl5():
    d = np.load(EPL5_NPZ)
    coords = d["coords"]
    p = d["p_real"].astype(np.float64) + 1j * d["p_imag"].astype(np.float64)
    print(f"[load] {EPL5_NPZ.name}: {len(p)} DOFs, |p|_max = {np.abs(p).max():.2f} Pa")
    return coords, p


def sample_idw(tree, p, pts, k=16, power=2.0, eps=1e-12):
    if k == 1:
        d, i = tree.query(pts, k=1)
        return p[i]
    d, i = tree.query(pts, k=k)
    w = 1.0 / (d**power + eps)
    w /= w.sum(axis=1, keepdims=True)
    return (p[i] * w).sum(axis=1)


def interpolate_full_domain(coords, p_fem, z_val):
    """Interpolate FEM data onto the full 6×6 mm domain at z=z_val."""
    tree = cKDTree(coords)
    xg = np.linspace(MARGIN, LX - MARGIN, N_GRID)
    yg = np.linspace(MARGIN, LY - MARGIN, N_GRID)
    XX, YY = np.meshgrid(xg, yg)
    pts_3d = np.column_stack([XX.ravel(), YY.ravel(),
                              np.full(XX.size, z_val)])
    print(f"[interp] {N_GRID}×{N_GRID} = {XX.size:,} query points at z={z_val*1e3:.3f} mm ...")
    t0 = time.time()
    p_flat = sample_idw(tree, p_fem, pts_3d, k=K_IDW)
    print(f"  done in {time.time()-t0:.1f}s")
    return p_flat.reshape(XX.shape), xg, yg


# ═══════════════════════════════════════════════════════════════════
# Gorkov + Trap Detection  (identical physics to C-shape script)
# ═══════════════════════════════════════════════════════════════════

def detect_gorkov_traps(p_grid, xg, yg):
    """
    Detect traps as Hessian-stable local minima of the full Gorkov potential.
    Returns (traps_m, trap_info_list).
    """
    p_abs = np.abs(p_grid)
    dx = xg[1] - xg[0]

    # Gorkov contrast factors — polystyrene in water
    rho_p, c_p = 1050.0, 2350.0
    kappa_w = 1.0 / (RHO0 * C_WATER**2)
    kappa_p = 1.0 / (rho_p * c_p**2)
    f1 = 1.0 - kappa_p / kappa_w
    f2 = 2.0 * (rho_p - RHO0) / (2.0 * rho_p + RHO0)
    print(f"  f1 = {f1:.4f},  f2 = {f2:.4f}")

    # Full Gorkov potential
    p2 = p_abs**2
    dp_dx = np.gradient(p_grid, dx, axis=1)
    dp_dy = np.gradient(p_grid, dx, axis=0)
    grad_p2 = np.abs(dp_dx)**2 + np.abs(dp_dy)**2

    coeff_p = f1 / (2.0 * RHO0 * C_WATER**2)
    coeff_k = 3.0 * f2 / (4.0 * OMEGA**2 * RHO0)
    U_gor = coeff_p * p2 - coeff_k * grad_p2

    print(f"  U range: [{U_gor.min():.3e}, {U_gor.max():.3e}]")
    print(f"  pressure term range: [{(coeff_p*p2).min():.3e}, {(coeff_p*p2).max():.3e}]")
    print(f"  kinetic  term range: [{(coeff_k*grad_p2).min():.3e}, {(coeff_k*grad_p2).max():.3e}]")

    # Local minima via morphological minimum filter
    min_sep_pts = max(3, int(0.4 * TRAP_SP / dx))
    neighborhood = 2 * min_sep_pts + 1
    print(f"  min_sep_pts={min_sep_pts}, neighbourhood={neighborhood}")
    local_min = minimum_filter(U_gor, size=neighborhood)
    mask = (U_gor == local_min)

    # Exclude grid border
    border = max(5, min_sep_pts)
    mask[:border, :] = False
    mask[-border:, :] = False
    mask[:, :border] = False
    mask[:, -border:] = False

    # Keep only genuine minima below 50% of U range
    U_range = U_gor.max() - U_gor.min()
    mask &= (U_gor < U_gor.min() + 0.50 * U_range)

    iy, ix = np.where(mask)
    print(f"  {len(iy)} candidate minima before Hessian check")

    # Hessian stability check
    dUdx  = np.gradient(U_gor, dx, axis=1)
    dUdy  = np.gradient(U_gor, dx, axis=0)
    d2Udx2  = np.gradient(dUdx, dx, axis=1)
    d2Udy2  = np.gradient(dUdy, dx, axis=0)
    d2Udxdy = np.gradient(dUdx, dx, axis=0)

    kept = []
    trap_info = []
    for ci in range(len(iy)):
        Hxx = d2Udx2[iy[ci], ix[ci]]
        Hyy = d2Udy2[iy[ci], ix[ci]]
        Hxy = d2Udxdy[iy[ci], ix[ci]]
        tr = Hxx + Hyy
        det = Hxx * Hyy - Hxy**2
        disc = max(0.0, tr**2 - 4 * det)
        lam_min = (tr - np.sqrt(disc)) / 2.0
        lam_max = (tr + np.sqrt(disc)) / 2.0
        if lam_min > 0:
            kept.append(ci)
            trap_info.append({
                "x_mm": float(xg[ix[ci]] * 1e3),
                "y_mm": float(yg[iy[ci]] * 1e3),
                "p_abs_Pa": float(p_abs[iy[ci], ix[ci]]),
                "U_gor": float(U_gor[iy[ci], ix[ci]]),
                "Hxx": float(Hxx), "Hyy": float(Hyy), "Hxy": float(Hxy),
                "lam_min": float(lam_min), "lam_max": float(lam_max),
            })

    traps_m = np.column_stack([xg[ix[kept]], yg[iy[kept]]]) if kept else np.zeros((0, 2))
    print(f"  {len(traps_m)} Hessian-stable Gorkov minima")

    for ti, info in enumerate(trap_info):
        print(f"    trap {ti:3d}: ({info['x_mm']:.3f}, {info['y_mm']:.3f}) mm  "
              f"|p|={info['p_abs_Pa']:.2f} Pa  U={info['U_gor']:.3e}  "
              f"λ=[{info['lam_min']:.2e}, {info['lam_max']:.2e}]")

    return traps_m, trap_info, U_gor, f1, f2


# ═══════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════

def _add_colorbar(ax, im, label=""):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.06)
    plt.colorbar(im, cax=cax, label=label)


def plot_full_domain_diagnostic(p_grid, U_gor, xg, yg, traps_m, trap_info, save_path):
    """3-panel figure: |p|, U_gor, |p| with trap markers."""
    xg_mm = xg * 1e3
    yg_mm = yg * 1e3
    extent = [xg_mm[0], xg_mm[-1], yg_mm[0], yg_mm[-1]]
    p_abs = np.abs(p_grid)

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    # Panel 1: |p|
    ax = axes[0]
    im = ax.imshow(p_abs, origin="lower", extent=extent, cmap="inferno", aspect="equal")
    _add_colorbar(ax, im, "|p| (Pa)")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("|p| — Standing Wave Amplitude")

    # Panel 2: Gorkov potential
    ax = axes[1]
    im = ax.imshow(U_gor, origin="lower", extent=extent, cmap="RdBu_r", aspect="equal")
    _add_colorbar(ax, im, "U_gor (J)")
    if len(traps_m) > 0:
        ax.plot(traps_m[:, 0] * 1e3, traps_m[:, 1] * 1e3,
                "k+", ms=8, mew=1.2)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(f"Gorkov Potential U(x,y) — {len(traps_m)} stable minima")

    # Panel 3: |p| with trap markers & labels
    ax = axes[2]
    im = ax.imshow(p_abs, origin="lower", extent=extent, cmap="inferno", aspect="equal")
    _add_colorbar(ax, im, "|p| (Pa)")
    if len(traps_m) > 0:
        ax.plot(traps_m[:, 0] * 1e3, traps_m[:, 1] * 1e3,
                "c^", ms=7, mew=1.0, mfc="none")
        for ti, info in enumerate(trap_info):
            ax.annotate(str(ti), (info["x_mm"], info["y_mm"]),
                        fontsize=5, color="white", ha="center", va="bottom",
                        xytext=(0, 3), textcoords="offset points")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("|p| with Gorkov Trap Sites")

    fig.suptitle(
        f"Full-Domain Gorkov Trap Detection — epl5, z* = {Z_STAR*1e3:.4f} mm, "
        f"{N_GRID}×{N_GRID} grid, {len(traps_m)} Hessian-stable minima",
        fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {save_path.name}")


def plot_gorkov_zoomed_centre(p_grid, U_gor, xg, yg, traps_m, trap_info, save_path):
    """Zoomed view around domain centre (same ROI as original study) for comparison."""
    roi_half = 1.1 * LAM
    x_lo, x_hi = CX - roi_half, CX + roi_half
    y_lo, y_hi = CY - roi_half, CY + roi_half

    ix_lo = np.searchsorted(xg, x_lo)
    ix_hi = np.searchsorted(xg, x_hi)
    iy_lo = np.searchsorted(yg, y_lo)
    iy_hi = np.searchsorted(yg, y_hi)

    p_sub = np.abs(p_grid[iy_lo:iy_hi, ix_lo:ix_hi])
    U_sub = U_gor[iy_lo:iy_hi, ix_lo:ix_hi]
    xg_sub = xg[ix_lo:ix_hi] * 1e3
    yg_sub = yg[iy_lo:iy_hi] * 1e3
    extent = [xg_sub[0], xg_sub[-1], yg_sub[0], yg_sub[-1]]

    # Filter traps in this ROI
    in_roi = ((traps_m[:, 0] >= x_lo) & (traps_m[:, 0] <= x_hi) &
              (traps_m[:, 1] >= y_lo) & (traps_m[:, 1] <= y_hi)) if len(traps_m) > 0 else np.array([], dtype=bool)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    im = ax.imshow(p_sub, origin="lower", extent=extent, cmap="inferno", aspect="equal")
    _add_colorbar(ax, im, "|p| (Pa)")
    if np.any(in_roi):
        roi_traps = traps_m[in_roi]
        ax.plot(roi_traps[:, 0]*1e3, roi_traps[:, 1]*1e3, "c^", ms=10, mew=1.5, mfc="none")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("|p| — Centre ROI (1.1λ)")

    ax = axes[1]
    im = ax.imshow(U_sub, origin="lower", extent=extent, cmap="RdBu_r", aspect="equal")
    _add_colorbar(ax, im, "U_gor (J)")
    if np.any(in_roi):
        roi_traps = traps_m[in_roi]
        ax.plot(roi_traps[:, 0]*1e3, roi_traps[:, 1]*1e3, "k+", ms=10, mew=1.5)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("Gorkov Potential — Centre ROI")

    n_in = int(np.sum(in_roi)) if len(traps_m) > 0 else 0
    fig.suptitle(f"Centre ROI Comparison — {n_in} traps in 1.1λ window (cf. original 8–9)", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {save_path.name}")


def plot_trap_spacing_histogram(traps_m, save_path):
    """Histogram of nearest-neighbour trap distances."""
    if len(traps_m) < 2:
        print("[skip] Too few traps for spacing histogram")
        return
    from scipy.spatial import cKDTree
    tree = cKDTree(traps_m)
    dists, _ = tree.query(traps_m, k=2)
    nn_dists = dists[:, 1] * 1e3   # mm

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(nn_dists, bins=30, edgecolor="k", alpha=0.7)
    ax.axvline(TRAP_SP*1e3, color="r", ls="--", lw=1.5, label=f"λ/2 = {TRAP_SP*1e3:.3f} mm")
    ax.axvline(LAM*1e3, color="orange", ls="--", lw=1.5, label=f"λ = {LAM*1e3:.3f} mm")
    ax.set_xlabel("Nearest-neighbour distance (mm)")
    ax.set_ylabel("Count")
    ax.set_title(f"Trap Spacing Distribution — {len(traps_m)} traps")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {save_path.name}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 70)
    print("Full-Domain Gorkov Trap Diagnostic  (epl5)")
    print("=" * 70)

    # 1. Load
    coords, p_fem = load_epl5()

    # 2. Interpolate full domain
    z_val = Z_STAR
    p_grid, xg, yg = interpolate_full_domain(coords, p_fem, z_val)
    dx = xg[1] - xg[0]
    print(f"  |p| range on grid: [{np.abs(p_grid).min():.2f}, {np.abs(p_grid).max():.2f}] Pa")
    print(f"  dx = {dx*1e6:.1f} µm  ({TRAP_SP/dx:.0f} pts per λ/2)")

    # 3. Detect traps
    print("[traps] Detecting Gorkov minima ...")
    traps_m, trap_info, U_gor, f1, f2 = detect_gorkov_traps(p_grid, xg, yg)

    # 4. Plots
    print("[plot] Full-domain diagnostic ...")
    plot_full_domain_diagnostic(p_grid, U_gor, xg, yg, traps_m, trap_info,
                                OUT_DIR / "full_domain_gorkov_traps.png")
    plot_gorkov_zoomed_centre(p_grid, U_gor, xg, yg, traps_m, trap_info,
                              OUT_DIR / "full_domain_centre_zoom.png")
    plot_trap_spacing_histogram(traps_m,
                                OUT_DIR / "trap_spacing_histogram.png")

    # 5. Save data
    trap_data = {
        "z_star_mm": float(z_val * 1e3),
        "n_grid": N_GRID,
        "dx_um": float(dx * 1e6),
        "domain_mm": [float(LX*1e3), float(LY*1e3)],
        "f1": float(f1),
        "f2": float(f2),
        "n_traps": len(traps_m),
        "traps": trap_info,
        "runtime_s": time.time() - t0,
    }
    json_path = OUT_DIR / "full_domain_trap_data.json"
    with open(json_path, "w") as f:
        json.dump(trap_data, f, indent=2)
    print(f"[save] {json_path.name}")

    np.savez_compressed(OUT_DIR / "full_domain_fields.npz",
                        p_grid=p_grid, U_gor=U_gor, xg=xg, yg=yg,
                        traps_m=traps_m)
    print(f"[save] full_domain_fields.npz")

    print(f"\n{'='*70}")
    print(f"Done in {time.time()-t0:.1f}s — {len(traps_m)} Gorkov traps detected")
    print(f"Outputs: {OUT_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
