#!/usr/bin/env python3
"""
Bridge IASA failure analysis — publication figures.

Generates four figures documenting the IASA failure modes and the corrected run:

  1. bridge_fail_montage.png       — IASA failure: target + iters 1/25/100
                                     rows = |p| / Gor'kov,  matched colour scales
  2. bridge_fail_target_cmp.png    — Real target (bridge_roi_particles) vs IASA target
  3. bridge_fail_convergence.png   — Corr + leakage vs iteration (original & corrected)
  4. bridge_fail_corrected.png     — 4-panel corrected IASA (random init, sup=0.5, 4×)

Run:
    python scripts/dev/bridge_failure_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dev.inverse_c_shape_lens_replica import (
    ReplicaConfig,
    make_grid,
    propagate_asm,
)

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
_IASA_DIR = (
    PROJECT_ROOT / "results" / "dev"
    / "inverse_bridge_pressure_lens_replica_scaled2x_20260324_095106"
)
BRIDGE_PRESSURE_NPZ = _IASA_DIR / "bridge_pressure_fields_scaled2x.npz"
BRIDGE_IASA_NPZ     = _IASA_DIR / "bridge_inverse_replica_fields.npz"
OUT_DIR             = PROJECT_ROOT / "results" / "figures"

# ─────────────────────────────────────────────────────────────────────────────
# Physics / grid configuration  (must match original IASA run)
# ─────────────────────────────────────────────────────────────────────────────
FREQUENCY_HZ       = 2_150_000.0
C_WATER            = 1480.0
C_LENS             = 2636.0
N_GRID             = 400
TRANSDUCER_DIAM_MM = 20.0
FOCAL_MM           = 13.21309776965029
H_BASE_MM          = 1.0
SOURCE_PRESSURE_PA = 0.05e6

PARTICLE_RADIUS_MM = 0.05
PARTICLE_DENSITY   = 1050.0
PARTICLE_C_SOUND   = 2350.0
RHO_WATER          = 998.0

# IASA parameters for the ORIGINAL (buggy) run
N_ITER             = 100
OUTSIDE_SUP_ORIG   = 0.0       # ← bug: forces perfect dark-field
# Corrected parameters
OUTSIDE_SUP_FIX    = 0.5
EXTRA_SCALE_FIX    = 4.0       # 4× pre-scale → corridor >> λ

# Snapshot iterations for the montage / convergence
SNAP_ITERS = [1, 2, 5, 10, 25, 50, 100]

ROI_PAD_MM = 0.8
DPI        = 190

CMAP_PRESS  = "hot"
CMAP_GORKOV = "RdBu_r"
CMAP_AMP    = "RdBu_r"
CMAP_THICK  = "viridis"
CMAP_PHASE  = "twilight"

COL_A     = "#e74c3c"
COL_B     = "#3498db"
COL_NEIGH = "#95a5a6"


# ─────────────────────────────────────────────────────────────────────────────
# Core helpers (same physics as the rest of the codebase)
# ─────────────────────────────────────────────────────────────────────────────
def _build_cfg(n_iter: int = N_ITER) -> ReplicaConfig:
    return ReplicaConfig(
        frequency_hz=FREQUENCY_HZ, c_water=C_WATER, c_lens=C_LENS,
        transducer_diameter_mm=TRANSDUCER_DIAM_MM,
        focal_distance_mm=FOCAL_MM, n_grid=N_GRID,
        h_base_mm=H_BASE_MM, n_iter=n_iter,
        source_pressure_pa=SOURCE_PRESSURE_PA,
    )


def _propagate_scaled(lens, cfg, dx, aperture_mask):
    p = propagate_asm(lens, cfg.k_water, cfg.focal_distance_m, dx)
    s = SOURCE_PRESSURE_PA / (
        np.sqrt(np.mean(np.abs(lens[aperture_mask]) ** 2)) + 1e-12
    )
    return p * s


def _gorkov(p, dx, omega):
    rho_w, c_w = RHO_WATER, C_WATER
    rho_p, c_p = PARTICLE_DENSITY, PARTICLE_C_SOUND
    a_m = PARTICLE_RADIUS_MM * 1e-3
    kappa_w = 1.0 / (rho_w * c_w ** 2)
    kappa_p = 1.0 / (rho_p * c_p ** 2)
    f1 = 1.0 - kappa_p / kappa_w
    f2 = 2.0 * (rho_p - rho_w) / (2.0 * rho_p + rho_w)
    Vp = (4.0 / 3.0) * np.pi * a_m ** 3
    dp_dy, dp_dx = np.gradient(p, dx, dx)
    vx = dp_dx / (-1j * omega * rho_w)
    vy = dp_dy / (-1j * omega * rho_w)
    return Vp * (
        f1 * np.abs(p) ** 2 / (4.0 * rho_w * c_w ** 2)
        - 3.0 * f2 * rho_w * (np.abs(vx) ** 2 + np.abs(vy) ** 2) / 8.0
    )


def _lens_thickness_mm(lens, cfg, aperture_mask):
    phi = np.mod(np.angle(lens), 2.0 * np.pi)
    t = cfg.h_base_m + cfg.h_max_m * (phi / (2.0 * np.pi))
    t[~aperture_mask] = np.nan
    return t * 1e3


def _build_target_at_scale(
    p_bridge, x_full, y_full,
    iasa_x, iasa_y,
    base_bridge_scale, extra_scale,
):
    xc_m = 0.5 * (x_full[0] + x_full[-1])
    yc_m = 0.5 * (y_full[0] + y_full[-1])
    xc = x_full - xc_m
    yc = y_full - yc_m
    interp = RegularGridInterpolator(
        (yc, xc), np.abs(p_bridge),
        method="linear", bounds_error=False, fill_value=0.0,
    )
    total_scale = base_bridge_scale * extra_scale
    ys, xs = np.meshgrid(iasa_y, iasa_x, indexing="ij")
    pts = np.column_stack([ys.ravel() / total_scale, xs.ravel() / total_scale])
    return interp(pts).reshape(N_GRID, N_GRID).astype(float)


def _normalise_target(raw_amp, clip_pct=99.5, gamma=0.9):
    a = np.maximum(raw_amp, 0.0)
    # Compute percentiles over non-zero pixels only so that the
    # clip threshold reflects the corridor's actual dynamic range,
    # not the vast zero background.
    nz = a[a > 1e-12]
    if nz.size == 0:
        return np.zeros_like(a)
    a_floor = float(np.percentile(nz, 1.0))
    a = np.maximum(a - a_floor, 0.0)
    nz2 = a[a > 1e-12]
    a_clip = float(np.percentile(nz2, clip_pct)) if nz2.size > 0 else 0.0
    if a_clip <= 0.0:
        return np.zeros_like(a)
    a = np.clip(a / a_clip, 0.0, 1.0)
    return np.power(a, max(gamma, 1e-6))


# ─────────────────────────────────────────────────────────────────────────────
# IASA loop with per-iteration metrics
# ─────────────────────────────────────────────────────────────────────────────
def _run_iasa_metrics(
    cfg: ReplicaConfig,
    aperture_mask: np.ndarray,
    target_amp: np.ndarray,
    roi_mask: np.ndarray,
    dx: float,
    n_iter: int,
    outside_sup: float,
    random_init: bool,
    snapshot_iters: list[int] | None = None,
) -> tuple[
    dict[int, np.ndarray],     # snapshots {iter: lens_field}
    np.ndarray,                # corr  (n_iter,)
    np.ndarray,                # leakage (n_iter,)
]:
    """IASA loop that records Pearson correlation + leakage at every iteration."""
    z   = cfg.focal_distance_m
    sup = float(np.clip(outside_sup, 0.0, 1.0))
    outside_roi = ~roi_mask
    snap_set = set(snapshot_iters or [])

    if random_init:
        rng = np.random.default_rng(seed=42)
        lens = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, (N_GRID, N_GRID)))
    else:
        lens = np.ones((N_GRID, N_GRID), dtype=complex)
    lens[~aperture_mask] = 0.0

    target_roi_flat = target_amp[roi_mask].ravel()
    t_mean = target_roi_flat.mean()
    t_std  = target_roi_flat.std() + 1e-30

    corrs    = np.zeros(n_iter)
    leakages = np.zeros(n_iter)
    snaps: dict[int, np.ndarray] = {}

    for it in range(1, n_iter + 1):
        img = propagate_asm(lens, cfg.k_water, z, dx)

        # --- metrics on the current reconstruction ---------------------------
        recon_amp = np.abs(img)
        roi_flat  = recon_amp[roi_mask].ravel()
        # normalise recon ROI to [0,1] for fair Pearson comparison
        r99 = float(np.percentile(roi_flat, 99.5)) + 1e-30
        roi_norm  = np.clip(roi_flat / r99, 0.0, 1.0)
        r_mean = roi_norm.mean()
        r_std  = roi_norm.std() + 1e-30
        corrs[it - 1] = float(
            np.mean((target_roi_flat - t_mean) * (roi_norm - r_mean))
            / (t_std * r_std)
        )
        E_total = float(np.sum(recon_amp[aperture_mask] ** 2)) + 1e-30
        E_roi   = float(np.sum(recon_amp[roi_mask] ** 2))
        leakages[it - 1] = 1.0 - E_roi / E_total

        # --- Gerchberg–Saxton update ----------------------------------------
        upd = img.copy()
        upd[roi_mask] = target_amp[roi_mask] * np.exp(1j * np.angle(img[roi_mask]))
        if sup < 1.0:
            upd[outside_roi] = (
                sup * np.abs(img[outside_roi])
                * np.exp(1j * np.angle(img[outside_roi]))
            )
        lens = np.exp(1j * np.angle(propagate_asm(upd, cfg.k_water, -z, dx)))
        lens[~aperture_mask] = 0.0

        if it in snap_set:
            snaps[it] = lens.copy()
            print(f"    snapshot iter {it:3d}")

    return snaps, corrs, leakages


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────
def _style(ax, fs=8):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=fs)


def _add_particles(ax, traps_mm, idx_a, idx_b, ms=40):
    neigh = sorted(set(range(len(traps_mm))) - {idx_a, idx_b})
    if neigh:
        ni = np.array(neigh)
        ax.scatter(traps_mm[ni, 0], traps_mm[ni, 1],
                   c=COL_NEIGH, s=ms, zorder=5, edgecolors="k", linewidths=0.4)
    ax.scatter([traps_mm[idx_a, 0]], [traps_mm[idx_a, 1]],
               c=COL_A, s=ms * 1.3, marker="^", zorder=6, edgecolors="k", linewidths=0.4)
    ax.scatter([traps_mm[idx_b, 0]], [traps_mm[idx_b, 1]],
               c=COL_B, s=ms * 1.3, marker="v", zorder=6, edgecolors="k", linewidths=0.4)


def _roi_contour(ax, iasa_x, iasa_y, roi_mask):
    ax.contour(iasa_x * 1e3, iasa_y * 1e3, roi_mask.astype(float),
               levels=[0.5], colors="cyan", linewidths=0.8,
               linestyles="--", alpha=0.75)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — IASA failure montage
# ─────────────────────────────────────────────────────────────────────────────
def fig_failure_montage(
    target_amp_display, p_fields, iasa_x, iasa_y,
    roi_mask, traps_mm, idx_a, idx_b, dx, omega,
    xlim, ylim, out_path,
):
    """2×4 grid: top = |p|, bottom = Gor'kov.  Cols = target, iter 1, 25, 100."""
    cols = [
        ("Target", target_amp_display, None),
        ("Iter 1", np.abs(p_fields[1]), p_fields[1]),
        ("Iter 25", np.abs(p_fields[25]), p_fields[25]),
        ("Iter 100", np.abs(p_fields[100]), p_fields[100]),
    ]
    ext = [iasa_x[0]*1e3, iasa_x[-1]*1e3, iasa_y[0]*1e3, iasa_y[-1]*1e3]

    # Shared scales from iter 100
    vmax_p = float(np.percentile(np.abs(p_fields[100])[roi_mask], 99.0))
    vmax_p = max(vmax_p, 1.0)

    U_ref = _gorkov(p_fields[100], dx, omega) * 1e18
    uv = float(np.percentile(np.abs(U_ref[roi_mask]), 99.0))
    uv = max(uv, 1e-18)

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))

    for j, (label, amp2d, pfield) in enumerate(cols):
        # ---- top row: pressure ----
        ax = axes[0, j]
        ax.imshow(amp2d, origin="lower", extent=ext,
                  cmap=CMAP_PRESS, vmin=0, vmax=vmax_p, aspect="equal")
        _roi_contour(ax, iasa_x, iasa_y, roi_mask)
        _add_particles(ax, traps_mm, idx_a, idx_b)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_ylabel("|p| [Pa]" if j == 0 else "", fontsize=9)
        ax.set_xlabel("x [mm]", fontsize=8)
        _style(ax)

        # ---- bottom row: Gor'kov ----
        ax = axes[1, j]
        if pfield is not None:
            U_aJ = _gorkov(pfield, dx, omega) * 1e18
        else:
            U_aJ = _gorkov(amp2d.astype(complex), dx, omega) * 1e18
        ax.imshow(U_aJ, origin="lower", extent=ext,
                  cmap=CMAP_GORKOV, vmin=-uv, vmax=uv, aspect="equal")
        _roi_contour(ax, iasa_x, iasa_y, roi_mask)
        _add_particles(ax, traps_mm, idx_a, idx_b)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_ylabel("U [aJ]" if j == 0 else "", fontsize=9)
        ax.set_xlabel("x [mm]", fontsize=8)
        _style(ax)

    # Shared colorbars
    fig.subplots_adjust(right=0.88, hspace=0.28, wspace=0.25, top=0.92, bottom=0.06)
    cb1_ax = fig.add_axes([0.90, 0.53, 0.015, 0.36])
    fig.colorbar(axes[0, 3].images[0], cax=cb1_ax, label="|p| [Pa]")
    cb2_ax = fig.add_axes([0.90, 0.06, 0.015, 0.36])
    fig.colorbar(axes[1, 3].images[0], cax=cb2_ax, label="U [aJ]")

    fig.suptitle(
        "IASA failure montage — zero-phase init, outside_sup=0.0\n"
        "(concentric Fresnel rings dominate; corridor shape never emerges)",
        fontsize=11, fontweight="bold",
    )
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Target comparison
# ─────────────────────────────────────────────────────────────────────────────
def fig_target_comparison(
    p_bridge, x_full, y_full, x_roi, y_roi,
    target_raw_amp, iasa_x, iasa_y,
    roi_mask, traps_m, traps_mm, idx_a, idx_b,
    xlim_iasa, ylim_iasa, out_path,
):
    """Left: real target (bridge_roi_particles view).  Right: IASA target on grid."""
    lam_mm = C_WATER / FREQUENCY_HZ * 1e3

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))

    # Left: real bridge field cropped to ROI (original coordinates)
    ax = axes[0]
    x_lo, x_hi = float(x_roi[0]), float(x_roi[-1])
    y_lo, y_hi = float(y_roi[0]), float(y_roi[-1])
    ix = np.where((x_full >= x_lo) & (x_full <= x_hi))[0]
    iy = np.where((y_full >= y_lo) & (y_full <= y_hi))[0]
    field_roi = np.abs(p_bridge)[np.ix_(iy, ix)]
    xc, yc = x_full[ix], y_full[iy]
    ext_roi = [xc[0]*1e3, xc[-1]*1e3, yc[0]*1e3, yc[-1]*1e3]

    vmax_roi = float(np.percentile(field_roi[field_roi > 1e-3], 99.0))
    im0 = ax.imshow(field_roi, origin="lower", extent=ext_roi,
                    cmap=CMAP_AMP, vmin=0, vmax=vmax_roi, aspect="equal",
                    interpolation="bicubic")
    plt.colorbar(im0, ax=ax, label="|p| [Pa]", fraction=0.046, pad=0.02)
    neigh = sorted(set(range(len(traps_m))) - {idx_a, idx_b})
    tr_mm = traps_m * 1e3
    if neigh:
        ni = np.array(neigh)
        ax.scatter(tr_mm[ni, 0], tr_mm[ni, 1], c=COL_NEIGH, s=50,
                   zorder=5, edgecolors="k", linewidths=0.4)
    ax.scatter([tr_mm[idx_a, 0]], [tr_mm[idx_a, 1]],
               c=COL_A, s=70, marker="^", zorder=6, edgecolors="k", linewidths=0.4)
    ax.scatter([tr_mm[idx_b, 0]], [tr_mm[idx_b, 1]],
               c=COL_B, s=70, marker="v", zorder=6, edgecolors="k", linewidths=0.4)
    ax.set_title("Real target — bridge pressure field\n(ROI crop, original coords)", fontsize=10)
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    # λ scalebar
    sb_x = ext_roi[0] + 0.3
    sb_y = ext_roi[2] + 0.2
    ax.plot([sb_x, sb_x + lam_mm], [sb_y, sb_y], "w-", lw=3)
    ax.text(sb_x + lam_mm / 2, sb_y + 0.08, f"λ = {lam_mm:.2f} mm",
            ha="center", va="bottom", color="w", fontsize=8, fontweight="bold")
    _style(ax)

    # Right: IASA target amplitude on IASA grid, zoomed to ROI
    ax = axes[1]
    ext_iasa = [iasa_x[0]*1e3, iasa_x[-1]*1e3, iasa_y[0]*1e3, iasa_y[-1]*1e3]
    vmax_raw = float(np.percentile(target_raw_amp[target_raw_amp > 0], 99.0))
    im1 = ax.imshow(target_raw_amp, origin="lower", extent=ext_iasa,
                    cmap=CMAP_AMP, vmin=0, vmax=vmax_raw, aspect="equal")
    plt.colorbar(im1, ax=ax, label="|p| [Pa]", fraction=0.046, pad=0.02)
    _roi_contour(ax, iasa_x, iasa_y, roi_mask)
    _add_particles(ax, traps_mm, idx_a, idx_b)
    ax.set_xlim(*xlim_iasa); ax.set_ylim(*ylim_iasa)
    ax.set_title("IASA target — interpolated onto\n20 mm aperture grid (ROI zoom)", fontsize=10)
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    # λ scalebar
    sb_x2 = xlim_iasa[0] + 0.3
    sb_y2 = ylim_iasa[0] + 0.2
    ax.plot([sb_x2, sb_x2 + lam_mm], [sb_y2, sb_y2], "w-", lw=3)
    ax.text(sb_x2 + lam_mm / 2, sb_y2 + 0.08, f"λ = {lam_mm:.2f} mm",
            ha="center", va="bottom", color="w", fontsize=8, fontweight="bold")
    _style(ax)

    fig.suptitle(
        "Target comparison — real bridge field vs IASA grid target",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Convergence curves
# ─────────────────────────────────────────────────────────────────────────────
def fig_convergence(corr_orig, leak_orig, corr_fix, leak_fix, out_path):
    iters = np.arange(1, len(corr_orig) + 1)
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax2 = ax1.twinx()

    ax1.plot(iters, corr_orig, "C0-",  lw=1.4, label="Corr — original (zero-init, sup=0)")
    ax1.plot(iters, corr_fix,  "C0--", lw=1.4, label="Corr — corrected (rand-init, sup=0.5)")
    ax2.plot(iters, leak_orig, "C3-",  lw=1.2, label="Leakage — original")
    ax2.plot(iters, leak_fix,  "C3--", lw=1.2, label="Leakage — corrected")

    ax1.set_xlabel("IASA iteration")
    ax1.set_ylabel("Pearson correlation (target vs recon)", color="C0")
    ax2.set_ylabel("Energy leakage fraction", color="C3")
    ax1.set_ylim(-0.1, 1.05)
    ax2.set_ylim(-0.02, 1.05)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="center right")
    ax1.set_title(
        "IASA convergence — original (buggy) vs corrected",
        fontsize=11, fontweight="bold",
    )
    _style(ax1); _style(ax2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Corrected 4-panel
# ─────────────────────────────────────────────────────────────────────────────
def fig_corrected_4panel(
    lens, cfg, aperture_mask, dx,
    iasa_x, iasa_y, roi_mask,
    traps_mm, idx_a, idx_b,
    xlim, ylim, omega, out_path,
):
    ext_mm = [iasa_x[0]*1e3, iasa_x[-1]*1e3, iasa_y[0]*1e3, iasa_y[-1]*1e3]
    r_ap = TRANSDUCER_DIAM_MM / 2.0
    theta_c = np.linspace(0, 2 * np.pi, 300)

    thickness_mm = _lens_thickness_mm(lens, cfg, aperture_mask)
    phase = np.mod(np.angle(lens), 2.0 * np.pi)
    phase[~aperture_mask] = np.nan

    p_fwd = _propagate_scaled(lens, cfg, dx, aperture_mask)
    p_abs = np.abs(p_fwd)
    vp = float(np.percentile(p_abs[roi_mask], 99.0))
    vp = max(vp, 1.0)

    U = _gorkov(p_fwd, dx, omega) * 1e18
    uv = float(np.percentile(np.abs(U[roi_mask]), 99.0))
    uv = max(uv, 1e-18)

    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))

    # Panel 0: thickness
    ax = axes[0]
    im0 = ax.imshow(thickness_mm, origin="lower", extent=ext_mm,
                    cmap=CMAP_THICK, aspect="equal",
                    vmin=float(np.nanmin(thickness_mm)),
                    vmax=float(np.nanmax(thickness_mm)))
    plt.colorbar(im0, ax=ax, label="t [mm]", fraction=0.046, pad=0.02)
    ax.plot(r_ap * np.cos(theta_c), r_ap * np.sin(theta_c), "r--", lw=0.7, alpha=0.6)
    ax.set_title("Lens Thickness"); ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    _style(ax)

    # Panel 1: phase
    ax = axes[1]
    im1 = ax.imshow(phase, origin="lower", extent=ext_mm,
                    cmap=CMAP_PHASE, vmin=0, vmax=2.0 * np.pi, aspect="equal")
    plt.colorbar(im1, ax=ax, label="phase [rad]", fraction=0.046, pad=0.02)
    ax.plot(r_ap * np.cos(theta_c), r_ap * np.sin(theta_c), "w--", lw=0.7, alpha=0.6)
    ax.set_title("Hologram Phase Map"); ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    _style(ax)

    # Panel 2: pressure (zoomed)
    ax = axes[2]
    im2 = ax.imshow(p_abs, origin="lower", extent=ext_mm,
                    cmap=CMAP_PRESS, vmin=0, vmax=vp, aspect="equal")
    plt.colorbar(im2, ax=ax, label="|p| [Pa]", fraction=0.046, pad=0.02)
    _roi_contour(ax, iasa_x, iasa_y, roi_mask)
    _add_particles(ax, traps_mm, idx_a, idx_b)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_title("Pressure at Focus"); ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    _style(ax)

    # Panel 3: Gor'kov (zoomed)
    ax = axes[3]
    im3 = ax.imshow(U, origin="lower", extent=ext_mm,
                    cmap=CMAP_GORKOV, vmin=-uv, vmax=uv, aspect="equal")
    plt.colorbar(im3, ax=ax, label="U [aJ]", fraction=0.046, pad=0.02)
    _roi_contour(ax, iasa_x, iasa_y, roi_mask)
    _add_particles(ax, traps_mm, idx_a, idx_b)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_title("Gor'kov Potential"); ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    _style(ax)

    fig.suptitle(
        "Corrected IASA — random-phase init, outside_sup=0.5, 4× bridge scale",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("BRIDGE FAILURE FIGURES")
    print("=" * 70)

    omega = 2.0 * np.pi * FREQUENCY_HZ

    # ── Load data ─────────────────────────────────────────────────────────
    print("\n[1] Loading data...")
    bp = np.load(BRIDGE_PRESSURE_NPZ)
    x_full  = bp["x_full"].astype(float)
    y_full  = bp["y_full"].astype(float)
    p_bridge = bp["p_bridge_effective_full"].astype(complex)
    traps_m = bp["traps_m"].astype(float)
    x_roi   = bp["x_roi"].astype(float)
    y_roi   = bp["y_roi"].astype(float)
    idx_a   = int(bp["idx_a"])
    idx_b   = int(bp["idx_b"])

    ir = np.load(BRIDGE_IASA_NPZ)
    target_amp     = ir["target_amp"].astype(float)
    target_raw_amp = ir["target_raw_amp"].astype(float)
    roi_mask       = ir["roi_mask"].astype(bool)
    aperture_mask  = ir["aperture_mask"].astype(bool)

    cfg = _build_cfg()
    iasa_x, iasa_y, _, _, _, _, _, dx = make_grid(cfg)

    # Trap positions in IASA coordinates
    xc_m = 0.5 * (x_full[0] + x_full[-1])
    yc_m = 0.5 * (y_full[0] + y_full[-1])
    field_half = max(float(np.abs(x_full - xc_m).max()),
                     float(np.abs(y_full - yc_m).max()))
    bridge_scale = (TRANSDUCER_DIAM_MM * 0.5e-3) / max(field_half, 1e-12)
    traps_iasa_m  = (traps_m[:, :2] - np.array([xc_m, yc_m])) * bridge_scale
    traps_iasa_mm = traps_iasa_m * 1e3

    # ROI bounding box for zoom
    rr, rc = np.where(roi_mask)
    xlim = (iasa_x[rc.min()] * 1e3 - ROI_PAD_MM, iasa_x[rc.max()] * 1e3 + ROI_PAD_MM)
    ylim = (iasa_y[rr.min()] * 1e3 - ROI_PAD_MM, iasa_y[rr.max()] * 1e3 + ROI_PAD_MM)

    print(f"  Bridge scale: {bridge_scale:.4f}x")
    print(f"  λ = {C_WATER / FREQUENCY_HZ * 1e3:.3f} mm")
    print(f"  ROI zoom: x={xlim}, y={ylim}")

    # ── Figure 1: IASA failure montage ────────────────────────────────────
    print("\n[2] Running original (buggy) IASA for failure montage...")
    snaps_orig, corr_orig, leak_orig = _run_iasa_metrics(
        cfg, aperture_mask, target_amp, roi_mask, dx,
        n_iter=N_ITER, outside_sup=OUTSIDE_SUP_ORIG,
        random_init=False, snapshot_iters=[1, 25, 100],
    )
    # Forward-propagate snapshots
    p_fields_orig = {
        it: _propagate_scaled(snaps_orig[it], cfg, dx, aperture_mask)
        for it in snaps_orig
    }
    # Scale target for display (matched to iter-100 peak)
    p100_roi = np.abs(p_fields_orig[100])[roi_mask]
    vmax_p = float(np.percentile(p100_roi, 99.0))
    target_display = target_amp * max(vmax_p, 1.0)

    fig_failure_montage(
        target_display, p_fields_orig,
        iasa_x, iasa_y, roi_mask,
        traps_iasa_mm, idx_a, idx_b, dx, omega,
        xlim, ylim,
        OUT_DIR / "bridge_fail_montage.png",
    )

    # ── Figure 2: Target comparison ───────────────────────────────────────
    print("\n[3] Target comparison figure...")
    fig_target_comparison(
        p_bridge, x_full, y_full, x_roi, y_roi,
        target_raw_amp, iasa_x, iasa_y,
        roi_mask, traps_m, traps_iasa_mm, idx_a, idx_b,
        xlim, ylim,
        OUT_DIR / "bridge_fail_target_cmp.png",
    )

    # ── Run corrected IASA ────────────────────────────────────────────────
    # Build 4×-scaled target on IASA grid
    print("\n[4] Building 4× target & running corrected IASA...")
    target_raw_4x = _build_target_at_scale(
        p_bridge, x_full, y_full,
        iasa_x, iasa_y,
        bridge_scale, EXTRA_SCALE_FIX,
    )
    target_4x = _normalise_target(target_raw_4x) * aperture_mask.astype(float)

    # Build ROI mask for the 4× target
    from scipy.ndimage import binary_dilation
    roi_mask_4x = binary_dilation(
        target_4x > 1e-6,
        structure=np.ones((11, 11), dtype=bool),
    ) & aperture_mask

    print(f"  4× target: max={target_raw_4x.max():.2f} Pa, "
          f"non-zero={np.mean(target_4x > 0.01):.4f}, "
          f"ROI frac={roi_mask_4x.sum() / aperture_mask.sum():.4f}")

    snaps_fix, corr_fix, leak_fix = _run_iasa_metrics(
        cfg, aperture_mask, target_4x, roi_mask_4x, dx,
        n_iter=N_ITER, outside_sup=OUTSIDE_SUP_FIX,
        random_init=True, snapshot_iters=[100],
    )

    # ── Figure 3: Convergence curves ──────────────────────────────────────
    print("\n[5] Convergence figure...")
    fig_convergence(corr_orig, leak_orig, corr_fix, leak_fix,
                    OUT_DIR / "bridge_fail_convergence.png")

    # ── Figure 4: Corrected 4-panel ───────────────────────────────────────
    print("\n[6] Corrected 4-panel figure...")
    # Compute ROI zoom for the 4× mask
    rr4, rc4 = np.where(roi_mask_4x)
    xlim4 = (iasa_x[rc4.min()] * 1e3 - ROI_PAD_MM,
             iasa_x[rc4.max()] * 1e3 + ROI_PAD_MM)
    ylim4 = (iasa_y[rr4.min()] * 1e3 - ROI_PAD_MM,
             iasa_y[rr4.max()] * 1e3 + ROI_PAD_MM)

    fig_corrected_4panel(
        snaps_fix[100], cfg, aperture_mask, dx,
        iasa_x, iasa_y, roi_mask_4x,
        traps_iasa_mm, idx_a, idx_b,
        xlim4, ylim4, omega,
        OUT_DIR / "bridge_fail_corrected.png",
    )

    print("\n" + "=" * 70)
    print("Done!  Figures saved to:", OUT_DIR)
    print("  bridge_fail_montage.png")
    print("  bridge_fail_target_cmp.png")
    print("  bridge_fail_convergence.png")
    print("  bridge_fail_corrected.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
