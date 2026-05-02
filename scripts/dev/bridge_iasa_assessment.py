#!/usr/bin/env python3
"""
Bridge IASA assessment — comprehensive evaluation of how IASA degrades
the bridge pressure field and its ability to transport particle A → B.

Outputs (all in results/bridge_iasa_assessment/):
  1. bridge_iasa_pressure_progression.png   — 2×4 |p| panels, RdBu_r
  2. bridge_iasa_gorkov_progression.png     — 2×4 U panels, RdBu_r
  3. bridge_iasa_degradation_metrics.png    — correlation, force, trajectory
  4. bridge_on_sw_overlay.png               — ideal & IASA bridge + SW
  5. parameters.txt                         — full parameter listing

Run:
    python scripts/dev/bridge_iasa_assessment.py
"""
from __future__ import annotations

import sys, textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dev.inverse_c_shape_lens_replica import (
    ReplicaConfig, make_grid, propagate_asm,
)

# ── Paths ────────────────────────────────────────────────────────────────────
_IASA_DIR = (
    PROJECT_ROOT / "results" / "dev"
    / "inverse_bridge_pressure_lens_replica_scaled2x_20260324_095106"
)
BRIDGE_PRESSURE_NPZ = _IASA_DIR / "bridge_pressure_fields_scaled2x.npz"
BRIDGE_IASA_NPZ     = _IASA_DIR / "bridge_inverse_replica_fields.npz"
SW_ROI_NPZ = (
    PROJECT_ROOT / "results"
    / "c_shape_lens_15mm_overlay_study_20260310_170620"
    / "npz" / "roi_fields.npz"
)
OUT_DIR = PROJECT_ROOT / "results" / "bridge_iasa_assessment"

# ── Physics ──────────────────────────────────────────────────────────────────
FREQUENCY_HZ       = 2_150_000.0
C_WATER            = 1480.0
C_LENS             = 2636.0
N_GRID             = 400
TRANSDUCER_DIAM_MM = 20.0
FOCAL_MM           = 13.21309776965029
H_BASE_MM          = 1.0
SOURCE_PRESSURE_PA = 0.05e6

PARTICLE_RADIUS_M  = 50e-6
PARTICLE_DENSITY   = 1050.0
PARTICLE_C_SOUND   = 2350.0
RHO_WATER          = 998.0
ETA_WATER          = 1e-3

P_SCALE            = 3.0e3          # FEM → real pressure scaling

OUTSIDE_SUP        = 0.0            # IASA outside-ROI suppression
SNAP_ITERS         = [1, 2, 5, 10, 25, 50, 100]
N_TRAJ_STEPS       = 2000
DT_TRAJ            = 1e-4           # s

DPI  = 190
CMAP = "RdBu_r"

COL_A     = "#e74c3c"
COL_B     = "#3498db"
COL_NEIGH = "#95a5a6"

# ── Derived ──────────────────────────────────────────────────────────────────
OMEGA   = 2.0 * np.pi * FREQUENCY_HZ
LAMBDA  = C_WATER / FREQUENCY_HZ
K_WATER = OMEGA / C_WATER
A_M     = PARTICLE_RADIUS_M
GAMMA_STOKES = 6.0 * np.pi * ETA_WATER * A_M


# ═════════════════════════════════════════════════════════════════════════════
# Core helpers
# ═════════════════════════════════════════════════════════════════════════════
def _build_cfg() -> ReplicaConfig:
    return ReplicaConfig(
        frequency_hz=FREQUENCY_HZ, c_water=C_WATER, c_lens=C_LENS,
        transducer_diameter_mm=TRANSDUCER_DIAM_MM,
        focal_distance_mm=FOCAL_MM, n_grid=N_GRID,
        h_base_mm=H_BASE_MM, n_iter=max(SNAP_ITERS),
        source_pressure_pa=SOURCE_PRESSURE_PA,
    )


def _normalise_target(raw_amp, clip_pct=99.5, gamma=0.9):
    a = np.maximum(raw_amp, 0.0)
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


def _propagate_scaled(lens, cfg, dx, aperture_mask):
    p = propagate_asm(lens, cfg.k_water, cfg.focal_distance_m, dx)
    s = SOURCE_PRESSURE_PA / (
        np.sqrt(np.mean(np.abs(lens[aperture_mask]) ** 2)) + 1e-12
    )
    return p * s


def _gorkov_UFxy(p, dx):
    """Gor'kov potential [J] and forces [N] from complex pressure."""
    kappa_w = 1.0 / (RHO_WATER * C_WATER ** 2)
    kappa_p = 1.0 / (PARTICLE_DENSITY * PARTICLE_C_SOUND ** 2)
    f1 = 1.0 - kappa_p / kappa_w
    f2 = 2.0 * (PARTICLE_DENSITY - RHO_WATER) / (2.0 * PARTICLE_DENSITY + RHO_WATER)
    Vp = (4.0 / 3.0) * np.pi * A_M ** 3
    dp_dy, dp_dx = np.gradient(p, dx, dx)
    vx = dp_dx / (-1j * OMEGA * RHO_WATER)
    vy = dp_dy / (-1j * OMEGA * RHO_WATER)
    U = Vp * (
        f1 * np.abs(p) ** 2 / (4.0 * RHO_WATER * C_WATER ** 2)
        - 3.0 * f2 * RHO_WATER * (np.abs(vx) ** 2 + np.abs(vy) ** 2) / 8.0
    )
    dU_dy, dU_dx = np.gradient(U, dx, dx)
    return U, -dU_dx, -dU_dy


def _simulate_trajectory(Fx, Fy, x_grid, y_grid, start_xy, n_steps=N_TRAJ_STEPS, dt=DT_TRAJ):
    """Overdamped trajectory. start_xy = (x, y) in metres. Returns (n_steps+1, 2)."""
    interp_fx = RegularGridInterpolator(
        (y_grid, x_grid), Fx, method="linear", bounds_error=False, fill_value=0.0)
    interp_fy = RegularGridInterpolator(
        (y_grid, x_grid), Fy, method="linear", bounds_error=False, fill_value=0.0)
    traj = np.empty((n_steps + 1, 2))
    traj[0] = start_xy
    for s in range(n_steps):
        x, y = traj[s]
        pt = np.array([[y, x]])
        fx = float(interp_fx(pt)[0])
        fy = float(interp_fy(pt)[0])
        traj[s + 1, 0] = x + fx / GAMMA_STOKES * dt
        traj[s + 1, 1] = y + fy / GAMMA_STOKES * dt
    return traj


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
               c=COL_A, s=ms * 1.3, marker="^", zorder=6, edgecolors="k", linewidths=0.4,
               label="Trap A")
    ax.scatter([traps_mm[idx_b, 0]], [traps_mm[idx_b, 1]],
               c=COL_B, s=ms * 1.3, marker="v", zorder=6, edgecolors="k", linewidths=0.4,
               label="Trap B")


# ═════════════════════════════════════════════════════════════════════════════
# IASA with snapshots + per-iteration metrics
# ═════════════════════════════════════════════════════════════════════════════
def _run_iasa(cfg, aperture_mask, target_amp, roi_mask, dx,
              snapshot_iters=SNAP_ITERS, outside_sup=OUTSIDE_SUP):
    z = cfg.focal_distance_m
    sup = float(np.clip(outside_sup, 0.0, 1.0))
    outside_roi = ~roi_mask
    snap_set = set(snapshot_iters)
    n_iter = max(snapshot_iters)

    rng = np.random.default_rng(seed=42)
    lens = np.exp(1j * rng.uniform(0, 2 * np.pi, (N_GRID, N_GRID)))
    lens[~aperture_mask] = 0.0

    t_roi = target_amp[roi_mask].ravel()
    t_mean, t_std = t_roi.mean(), t_roi.std() + 1e-30

    snaps, corrs = {}, np.zeros(n_iter)
    for it in range(1, n_iter + 1):
        img = propagate_asm(lens, cfg.k_water, z, dx)
        # correlation metric
        r_roi = np.abs(img)[roi_mask].ravel()
        r99 = float(np.percentile(r_roi, 99.5)) + 1e-30
        rn = np.clip(r_roi / r99, 0, 1)
        corrs[it - 1] = float(
            np.mean((t_roi - t_mean) * (rn - rn.mean())) / (t_std * (rn.std() + 1e-30))
        )
        # GS update
        upd = img.copy()
        upd[roi_mask] = target_amp[roi_mask] * np.exp(1j * np.angle(img[roi_mask]))
        if sup < 1.0:
            upd[outside_roi] = sup * np.abs(img[outside_roi]) * np.exp(1j * np.angle(img[outside_roi]))
        lens = np.exp(1j * np.angle(propagate_asm(upd, cfg.k_water, -z, dx)))
        lens[~aperture_mask] = 0.0
        if it in snap_set:
            snaps[it] = lens.copy()
            print(f"    iter {it:3d}")
    return snaps, corrs


# ═════════════════════════════════════════════════════════════════════════════
# Figures
# ═════════════════════════════════════════════════════════════════════════════

# ---------- Figure 1 & 2: progression panels ---------------------------------
def _fig_progression(panels, iasa_x_mm, iasa_y_mm, traps_mm, idx_a, idx_b,
                     xlim, ylim, quantity, unit, out_path):
    """2×4 panel figure — panels = list of (title, 2d_array)."""
    ext = [iasa_x_mm[0], iasa_x_mm[-1], iasa_y_mm[0], iasa_y_mm[-1]]
    nrows = 2; ncols = 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 10))
    axes_flat = axes.ravel()

    # shared colour limits from last panel (the converged IASA)
    last = panels[-1][1]
    crop_mask = np.ones_like(last, dtype=bool)
    # rough ROI crop for percentile
    ix0 = max(0, int((xlim[0] - iasa_x_mm[0]) / (iasa_x_mm[1] - iasa_x_mm[0])))
    ix1 = min(last.shape[1] - 1, int((xlim[1] - iasa_x_mm[0]) / (iasa_x_mm[1] - iasa_x_mm[0])))
    iy0 = max(0, int((ylim[0] - iasa_y_mm[0]) / (iasa_y_mm[1] - iasa_y_mm[0])))
    iy1 = min(last.shape[0] - 1, int((ylim[1] - iasa_y_mm[0]) / (iasa_y_mm[1] - iasa_y_mm[0])))
    roi_crop = last[iy0:iy1+1, ix0:ix1+1]

    if quantity == "gorkov":
        vmax = float(np.percentile(np.abs(roi_crop), 99.0))
        vmin = -vmax
    else:
        vmax = float(np.percentile(roi_crop[roi_crop > 0], 99.0)) if np.any(roi_crop > 0) else 1.0
        vmin = 0.0

    for i, (title, arr) in enumerate(panels):
        ax = axes_flat[i]
        ax.imshow(arr, origin="lower", extent=ext, cmap=CMAP,
                  vmin=vmin, vmax=vmax, aspect="equal")
        _add_particles(ax, traps_mm, idx_a, idx_b)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("x [mm]", fontsize=8)
        ax.set_ylabel("y [mm]" if i % ncols == 0 else "", fontsize=8)
        _style(ax)

    # hide unused axes
    for j in range(len(panels), nrows * ncols):
        axes_flat[j].set_visible(False)

    fig.subplots_adjust(right=0.88, hspace=0.30, wspace=0.25, top=0.91, bottom=0.06)
    cb_ax = fig.add_axes([0.90, 0.10, 0.015, 0.78])
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    fig.colorbar(sm, cax=cb_ax, label=f"{unit}")

    fig.suptitle(
        f"Bridge IASA progression — {quantity} at focus (ROI zoom, RdBu_r)",
        fontsize=12, fontweight="bold",
    )
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ---------- Figure 3: degradation metrics ------------------------------------
def _fig_degradation(
    corrs, iters,
    traj_ideal, traj_iasa,
    trap_a_m, trap_b_m,
    force_ax_ideal, force_ax_iasa, force_s_mm,
    out_path,
):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) Pearson correlation vs iteration
    ax = axes[0]
    ax.plot(np.arange(1, len(corrs) + 1), corrs, "k-", lw=1.4)
    ax.set_xlabel("IASA iteration"); ax.set_ylabel("Pearson correlation")
    ax.set_title("(a) Target–recon correlation", fontweight="bold")
    ax.set_ylim(-0.1, 1.05); _style(ax)

    # (b) Force projection along A→B
    ax = axes[1]
    ax.plot(force_s_mm, force_ax_ideal * 1e15, "k-", lw=1.2, label="Ideal bridge")
    ax.plot(force_s_mm, force_ax_iasa * 1e15, "r--", lw=1.2, label="IASA bridge")
    ax.axhline(0, color="grey", lw=0.5, ls=":")
    ax.set_xlabel("Distance along A→B [mm]")
    ax.set_ylabel("Force component A→B [fN]")
    ax.set_title("(b) Force along corridor", fontweight="bold")
    ax.legend(fontsize=8); _style(ax)

    # (c) Particle trajectory
    ax = axes[2]
    d_AB = np.linalg.norm(trap_b_m - trap_a_m) * 1e3
    t_ms = np.arange(traj_ideal.shape[0]) * DT_TRAJ * 1e3
    dist_ideal = np.linalg.norm(traj_ideal - trap_a_m, axis=1) * 1e3
    dist_iasa  = np.linalg.norm(traj_iasa  - trap_a_m, axis=1) * 1e3
    ax.plot(t_ms, dist_ideal, "k-", lw=1.2, label="Ideal bridge")
    ax.plot(t_ms, dist_iasa, "r--", lw=1.2, label="IASA bridge")
    ax.axhline(d_AB, color=COL_B, lw=0.8, ls=":", label=f"A→B = {d_AB:.3f} mm")
    ax.set_xlabel("Time [ms]"); ax.set_ylabel("Displacement from A [mm]")
    ax.set_title("(c) Particle A displacement", fontweight="bold")
    ax.legend(fontsize=8); _style(ax)

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ---------- Figure 4: SW + bridge overlay ------------------------------------
def _fig_overlay(
    p_sw, p_bridge_ideal, p_bridge_iasa,
    xg_mm, yg_mm, traps_mm, idx_a, idx_b,
    alpha, out_path,
):
    """2-panel: SW + α·bridge_ideal  vs  SW + α·bridge_IASA. """
    p_total_ideal = p_sw + alpha * p_bridge_ideal
    p_total_iasa  = p_sw + alpha * p_bridge_iasa
    ext = [xg_mm[0], xg_mm[-1], yg_mm[0], yg_mm[-1]]

    # matched colour scale
    vmax = max(
        float(np.percentile(np.abs(p_total_ideal), 99.5)),
        float(np.percentile(np.abs(p_total_iasa), 99.5)),
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    titles = [
        f"Standing wave + ideal bridge (α = {alpha:.2f})",
        f"Standing wave + IASA bridge (α = {alpha:.2f})",
    ]
    fields = [np.abs(p_total_ideal), np.abs(p_total_iasa)]
    for ax, title, field in zip(axes, titles, fields):
        im = ax.imshow(field, origin="lower", extent=ext, cmap=CMAP,
                       vmin=0, vmax=vmax, aspect="equal")
        _add_particles(ax, traps_mm, idx_a, idx_b)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
        _style(ax)
        plt.colorbar(im, ax=ax, label="|p| [Pa]", fraction=0.046, pad=0.02)

    fig.suptitle(
        "Bridge field overlaid on standing wave — ideal vs IASA reproduction",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print("BRIDGE IASA ASSESSMENT")
    print("=" * 72)

    cfg = _build_cfg()

    # ── 1. Load data ─────────────────────────────────────────────────────
    print("\n[1] Loading data …")
    bp = np.load(BRIDGE_PRESSURE_NPZ)
    x_full       = bp["x_full"].astype(float)
    y_full       = bp["y_full"].astype(float)
    p_bridge_eff = bp["p_bridge_effective_full"].astype(complex)
    traps_m      = bp["traps_m"].astype(float)
    idx_a        = int(bp["idx_a"])
    idx_b        = int(bp["idx_b"])

    ir = np.load(BRIDGE_IASA_NPZ)
    target_raw_amp = ir["target_raw_amp"].astype(float)
    roi_mask       = ir["roi_mask"].astype(bool)
    aperture_mask  = ir["aperture_mask"].astype(bool)

    target_amp = _normalise_target(target_raw_amp) * aperture_mask.astype(float)

    iasa_x, iasa_y, _, _, _, _, _, dx = make_grid(cfg)
    iasa_x_mm = iasa_x * 1e3
    iasa_y_mm = iasa_y * 1e3

    # Trap positions in IASA coordinate frame
    xc_m = 0.5 * (x_full[0] + x_full[-1])
    yc_m = 0.5 * (y_full[0] + y_full[-1])
    field_half = max(float(np.abs(x_full - xc_m).max()),
                     float(np.abs(y_full - yc_m).max()))
    bridge_scale = (TRANSDUCER_DIAM_MM * 0.5e-3) / max(field_half, 1e-12)
    traps_iasa_m  = (traps_m[:, :2] - np.array([xc_m, yc_m])) * bridge_scale
    traps_iasa_mm = traps_iasa_m * 1e3

    # ROI zoom bounds
    rr, rc = np.where(roi_mask)
    PAD = 0.8
    xlim = (iasa_x_mm[rc.min()] - PAD, iasa_x_mm[rc.max()] + PAD)
    ylim = (iasa_y_mm[rr.min()] - PAD, iasa_y_mm[rr.max()] + PAD)

    trap_a_iasa = traps_iasa_m[idx_a]
    trap_b_iasa = traps_iasa_m[idx_b]

    print(f"  Bridge scale: {bridge_scale:.4f}×")
    print(f"  λ = {LAMBDA*1e3:.3f} mm")
    print(f"  Trap A (IASA): ({trap_a_iasa[0]*1e3:.3f}, {trap_a_iasa[1]*1e3:.3f}) mm")
    print(f"  Trap B (IASA): ({trap_b_iasa[0]*1e3:.3f}, {trap_b_iasa[1]*1e3:.3f}) mm")

    # ── 2. Run IASA ──────────────────────────────────────────────────────
    print("\n[2] Running IASA (100 iterations) …")
    snaps, corrs = _run_iasa(cfg, aperture_mask, target_amp, roi_mask, dx)

    # Forward-propagate all snapshots
    p_snaps = {
        it: _propagate_scaled(snaps[it], cfg, dx, aperture_mask)
        for it in snaps
    }

    # "Ideal" bridge on IASA grid = propagate the target amplitude with
    # perfect phase.  We re-use the raw amplitude scaled to match the
    # iter-100 reconstruction peak, so the comparison is apples-to-apples.
    p100_abs = np.abs(p_snaps[100])
    vmax_p = float(np.percentile(p100_abs[roi_mask], 99.0))

    # ── 3. Pressure progression figure ───────────────────────────────────
    print("\n[3] Pressure progression …")
    target_display = target_amp * max(vmax_p, 1.0)
    panels_p = [("Target (normalised)", target_display)]
    for it in SNAP_ITERS:
        panels_p.append((f"Iter {it}", np.abs(p_snaps[it])))
    _fig_progression(panels_p, iasa_x_mm, iasa_y_mm, traps_iasa_mm,
                     idx_a, idx_b, xlim, ylim, "pressure", "|p| [Pa]",
                     OUT_DIR / "bridge_iasa_pressure_progression.png")

    # ── 4. Gor'kov progression figure ────────────────────────────────────
    print("\n[4] Gor'kov progression …")
    U_target, _, _ = _gorkov_UFxy(target_display.astype(complex), dx)
    panels_g = [("Target", U_target * 1e18)]
    for it in SNAP_ITERS:
        Ui, _, _ = _gorkov_UFxy(p_snaps[it], dx)
        panels_g.append((f"Iter {it}", Ui * 1e18))
    _fig_progression(panels_g, iasa_x_mm, iasa_y_mm, traps_iasa_mm,
                     idx_a, idx_b, xlim, ylim, "gorkov", "U [aJ]",
                     OUT_DIR / "bridge_iasa_gorkov_progression.png")

    # ── 5. Degradation metrics ───────────────────────────────────────────
    print("\n[5] Degradation metrics …")

    # 5a) Force profile along A→B for ideal target & IASA reconstruction
    dir_AB = trap_b_iasa - trap_a_iasa
    d_AB = np.linalg.norm(dir_AB)
    uhat = dir_AB / d_AB
    n_pts = 120
    s_vals = np.linspace(-0.2 * d_AB, 1.2 * d_AB, n_pts)
    line_pts = trap_a_iasa[None, :] + s_vals[:, None] * uhat[None, :]  # (n_pts, 2)

    # ideal: use target_amp * vmax_p as complex field (real-valued)
    p_ideal_complex = target_display.astype(complex)
    _, Fx_id, Fy_id = _gorkov_UFxy(p_ideal_complex, dx)
    # IASA iter 100
    _, Fx_ia, Fy_ia = _gorkov_UFxy(p_snaps[100], dx)

    interp_Fx_id = RegularGridInterpolator((iasa_y, iasa_x), Fx_id, bounds_error=False, fill_value=0.0)
    interp_Fy_id = RegularGridInterpolator((iasa_y, iasa_x), Fy_id, bounds_error=False, fill_value=0.0)
    interp_Fx_ia = RegularGridInterpolator((iasa_y, iasa_x), Fx_ia, bounds_error=False, fill_value=0.0)
    interp_Fy_ia = RegularGridInterpolator((iasa_y, iasa_x), Fy_ia, bounds_error=False, fill_value=0.0)

    pts_yx = np.column_stack([line_pts[:, 1], line_pts[:, 0]])
    F_ax_ideal = interp_Fx_id(pts_yx) * uhat[0] + interp_Fy_id(pts_yx) * uhat[1]
    F_ax_iasa  = interp_Fx_ia(pts_yx) * uhat[0] + interp_Fy_ia(pts_yx) * uhat[1]
    force_s_mm = s_vals * 1e3

    # 5b) Trajectory simulation: ideal vs IASA
    print("  Simulating trajectories …")
    traj_ideal = _simulate_trajectory(Fx_id, Fy_id, iasa_x, iasa_y, trap_a_iasa)
    traj_iasa  = _simulate_trajectory(Fx_ia, Fy_ia, iasa_x, iasa_y, trap_a_iasa)

    _fig_degradation(
        corrs, SNAP_ITERS,
        traj_ideal, traj_iasa,
        trap_a_iasa, trap_b_iasa,
        F_ax_ideal, F_ax_iasa, force_s_mm,
        OUT_DIR / "bridge_iasa_degradation_metrics.png",
    )

    # print key metrics
    disp_ideal = np.linalg.norm(traj_ideal[-1] - trap_a_iasa) * 1e3
    disp_iasa  = np.linalg.norm(traj_iasa[-1] - trap_a_iasa) * 1e3
    d_AB_mm = d_AB * 1e3
    print(f"  Correlation at iter 100:   {corrs[-1]:.4f}")
    print(f"  |A→B| distance:            {d_AB_mm:.4f} mm")
    print(f"  Ideal bridge displacement: {disp_ideal:.4f} mm  ({disp_ideal/d_AB_mm*100:.1f}%)")
    print(f"  IASA  bridge displacement: {disp_iasa:.4f} mm  ({disp_iasa/d_AB_mm*100:.1f}%)")
    print(f"  Displacement ratio (IASA/ideal): {disp_iasa/(disp_ideal+1e-20):.4f}")

    # ── 6. Standing-wave overlay ─────────────────────────────────────────
    print("\n[6] Standing-wave overlay …")
    sw = np.load(SW_ROI_NPZ)
    p_sw       = sw["p_sw"].astype(complex) * P_SCALE  # → real pressure units
    sw_xg      = sw["xg"].astype(float)
    sw_yg      = sw["yg"].astype(float)
    sw_traps   = sw["traps_m"].astype(float)
    sw_idx_a   = int(sw["idx_A"])
    sw_idx_b   = int(sw["idx_B"])
    sw_traps_mm = sw_traps * 1e3

    sw_dx = float(sw_xg[1] - sw_xg[0])

    # Ideal bridge on SW grid: interpolate from 2×-scaled bridge back
    # to the real SW coordinates.
    x_br_c = x_full - xc_m      # bridge coords centred
    y_br_c = y_full - yc_m
    br_centre_real = np.array([xc_m, yc_m])  # in original bridge (= SW) metres

    interp_br = RegularGridInterpolator(
        (y_full, x_full), p_bridge_eff,
        method="linear", bounds_error=False, fill_value=0.0,
    )
    sw_yg2d, sw_xg2d = np.meshgrid(sw_yg, sw_xg, indexing="ij")
    pts_sw = np.column_stack([sw_yg2d.ravel(), sw_xg2d.ravel()])
    p_bridge_on_sw = interp_br(pts_sw).reshape(len(sw_yg), len(sw_xg))

    # IASA bridge on SW grid: map IASA focal field → real-world coords
    p_iasa_focal = p_snaps[100]  # complex, on IASA grid
    interp_ia = RegularGridInterpolator(
        (iasa_y, iasa_x), p_iasa_focal,
        method="linear", bounds_error=False, fill_value=0.0,
    )
    sw_in_iasa_x = (sw_xg2d - xc_m) * bridge_scale
    sw_in_iasa_y = (sw_yg2d - yc_m) * bridge_scale
    pts_ia = np.column_stack([sw_in_iasa_y.ravel(), sw_in_iasa_x.ravel()])
    p_iasa_on_sw_raw = interp_ia(pts_ia).reshape(len(sw_yg), len(sw_xg))

    # Scale IASA field to match ideal bridge peak
    ideal_peak = float(np.percentile(np.abs(p_bridge_on_sw), 99.0))
    iasa_peak  = float(np.percentile(np.abs(p_iasa_on_sw_raw), 99.0))
    alpha_match = ideal_peak / max(iasa_peak, 1e-12)
    p_iasa_on_sw = alpha_match * p_iasa_on_sw_raw
    print(f"  Ideal bridge peak on SW: {ideal_peak:.2f} Pa")
    print(f"  IASA  bridge peak on SW: {iasa_peak:.2f} Pa")
    print(f"  α_match = {alpha_match:.6f}  (scales IASA to ideal peak)")

    # Build combined fields at the SW grid
    p_total_ideal = p_sw + p_bridge_on_sw
    p_total_iasa  = p_sw + p_iasa_on_sw

    # Gor'kov for all three
    U_sw, _, _ = _gorkov_UFxy(p_sw, sw_dx)
    U_ideal, _, _ = _gorkov_UFxy(p_total_ideal, sw_dx)
    U_iasa, _, _ = _gorkov_UFxy(p_total_iasa, sw_dx)

    # 4-panel figure: SW Gor'kov | SW+ideal Gor'kov | SW+IASA Gor'kov | difference
    ext_sw = [sw_xg[0]*1e3, sw_xg[-1]*1e3, sw_yg[0]*1e3, sw_yg[-1]*1e3]
    U_scale = 1e18  # → aJ

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # shared Gor'kov scale
    uv = float(np.percentile(np.abs(U_sw * U_scale), 98.0))

    # Panel (0,0): SW alone — Gor'kov
    ax = axes[0, 0]
    im = ax.imshow(U_sw * U_scale, origin="lower", extent=ext_sw,
                   cmap=CMAP, vmin=-uv, vmax=uv, aspect="equal")
    _add_particles(ax, sw_traps_mm, sw_idx_a, sw_idx_b)
    ax.set_title("Standing wave only — Gor'kov", fontsize=10, fontweight="bold")
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]"); _style(ax)
    plt.colorbar(im, ax=ax, label="U [aJ]", fraction=0.046, pad=0.02)

    # Panel (0,1): SW + ideal bridge
    ax = axes[0, 1]
    im = ax.imshow(U_ideal * U_scale, origin="lower", extent=ext_sw,
                   cmap=CMAP, vmin=-uv, vmax=uv, aspect="equal")
    _add_particles(ax, sw_traps_mm, sw_idx_a, sw_idx_b)
    ax.set_title("SW + ideal bridge — Gor'kov", fontsize=10, fontweight="bold")
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]"); _style(ax)
    plt.colorbar(im, ax=ax, label="U [aJ]", fraction=0.046, pad=0.02)

    # Panel (1,0): SW + IASA bridge
    ax = axes[1, 0]
    im = ax.imshow(U_iasa * U_scale, origin="lower", extent=ext_sw,
                   cmap=CMAP, vmin=-uv, vmax=uv, aspect="equal")
    _add_particles(ax, sw_traps_mm, sw_idx_a, sw_idx_b)
    ax.set_title("SW + IASA bridge — Gor'kov", fontsize=10, fontweight="bold")
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]"); _style(ax)
    plt.colorbar(im, ax=ax, label="U [aJ]", fraction=0.046, pad=0.02)

    # Panel (1,1): Gor'kov difference (ideal − SW)
    dU_ideal = (U_ideal - U_sw) * U_scale
    dU_iasa  = (U_iasa  - U_sw) * U_scale
    dv = max(float(np.percentile(np.abs(dU_ideal), 99.0)), 1e-30)
    ax = axes[1, 1]
    im = ax.imshow(dU_ideal, origin="lower", extent=ext_sw,
                   cmap=CMAP, vmin=-dv, vmax=dv, aspect="equal", alpha=0.85)
    # overlay IASA difference as contours
    ax.contour(sw_xg * 1e3, sw_yg * 1e3, dU_iasa,
               levels=np.linspace(-dv, dv, 11), colors="k", linewidths=0.5, alpha=0.6)
    _add_particles(ax, sw_traps_mm, sw_idx_a, sw_idx_b)
    ax.set_title("ΔU perturbation (ideal=colour, IASA=contours)", fontsize=10, fontweight="bold")
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]"); _style(ax)
    plt.colorbar(im, ax=ax, label="ΔU [aJ]", fraction=0.046, pad=0.02)

    fig.suptitle(
        f"Bridge on standing wave — Gor'kov potential\n"
        f"(bridge ≈ {ideal_peak:.0f} Pa perturbation on {np.abs(p_sw).max():.0f} Pa SW)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT_DIR / "bridge_on_sw_overlay.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: bridge_on_sw_overlay.png")

    # ── 7. Parameter file ────────────────────────────────────────────────
    print("\n[7] Writing parameters …")
    params_text = textwrap.dedent(f"""\
    ══════════════════════════════════════════════════════
    BRIDGE IASA ASSESSMENT — PARAMETERS
    ══════════════════════════════════════════════════════

    Acoustic medium
      Frequency           : {FREQUENCY_HZ/1e6:.3f} MHz
      Speed of sound      : {C_WATER:.1f} m/s  (water)
      Wavelength (λ)      : {LAMBDA*1e3:.4f} mm
      Angular frequency   : {OMEGA:.2f} rad/s
      Wavenumber k         : {K_WATER:.2f} 1/m
      Water density        : {RHO_WATER} kg/m³

    Lens / Transducer
      Transducer diameter : {TRANSDUCER_DIAM_MM:.1f} mm
      Aperture radius     : {TRANSDUCER_DIAM_MM/2:.1f} mm
      Focal distance      : {FOCAL_MM:.4f} mm
      Lens sound speed    : {C_LENS:.1f} m/s
      Lens base thickness : {H_BASE_MM:.2f} mm
      Source pressure      : {SOURCE_PRESSURE_PA/1e3:.1f} kPa

    IASA grid
      Grid size           : {N_GRID} × {N_GRID}
      Pixel pitch (dx)    : {dx*1e6:.2f} µm
      Grid extent         : [{iasa_x_mm[0]:.2f}, {iasa_x_mm[-1]:.2f}] mm
      Bridge scale factor : {bridge_scale:.4f}×

    IASA algorithm
      Iterations          : {max(SNAP_ITERS)}
      Phase initialisation: random (seed=42)
      Outside-ROI sup.    : {OUTSIDE_SUP}
      Normalisation       : non-zero pctile (floor p1, clip p99.5, γ=0.9)

    Particle
      Material            : polystyrene
      Radius (a)          : {A_M*1e6:.1f} µm
      Density             : {PARTICLE_DENSITY} kg/m³
      Sound speed         : {PARTICLE_C_SOUND} m/s
      Stokes drag (γ)     : {GAMMA_STOKES:.4e} N·s/m
      Viscosity (η)       : {ETA_WATER} Pa·s

    Bridge field
      Pre-scaling         : 2× (domain 11.8 mm, mapped onto 20 mm aperture)
      Corridor width      : ~300 µm × 2× = ~600 µm → {0.6*bridge_scale:.2f} mm on IASA grid ≈ {0.6*bridge_scale/(LAMBDA*1e3):.1f}λ
      Alpha (baked)       : ≈64.27  (from tuning sweep)
      ψ (phase offset)    : 0.0 rad

    Standing wave overlay
      P_SCALE             : {P_SCALE:.0f} Pa
      |p_sw| max          : {np.abs(p_sw).max():.1f} Pa (after scaling)

    Trap geometry (IASA frame)
      Trap A (idx {idx_a})    : ({trap_a_iasa[0]*1e3:.3f}, {trap_a_iasa[1]*1e3:.3f}) mm
      Trap B (idx {idx_b})    : ({trap_b_iasa[0]*1e3:.3f}, {trap_b_iasa[1]*1e3:.3f}) mm
      A→B distance        : {d_AB_mm:.4f} mm  ({d_AB_mm/LAMBDA/1e3:.2f}λ)

    Trap geometry (real / SW frame)
      Trap A (idx {sw_idx_a})    : ({sw_traps_mm[sw_idx_a,0]:.4f}, {sw_traps_mm[sw_idx_a,1]:.4f}) mm
      Trap B (idx {sw_idx_b})    : ({sw_traps_mm[sw_idx_b,0]:.4f}, {sw_traps_mm[sw_idx_b,1]:.4f}) mm
      A→B distance        : {np.linalg.norm(sw_traps[sw_idx_a]-sw_traps[sw_idx_b])*1e3:.4f} mm

    Results
      Correlation @ iter 100  : {corrs[-1]:.4f}
      Ideal bridge A displ.   : {disp_ideal:.4f} mm  ({disp_ideal/d_AB_mm*100:.1f}% of A→B)
      IASA bridge A displ.    : {disp_iasa:.4f} mm  ({disp_iasa/d_AB_mm*100:.1f}% of A→B)
      Displacement retention  : {disp_iasa/(disp_ideal+1e-20)*100:.1f}%
      Simulation time         : {N_TRAJ_STEPS * DT_TRAJ * 1e3:.0f} ms ({N_TRAJ_STEPS} steps, dt={DT_TRAJ*1e6:.0f} µs)
    """)
    (OUT_DIR / "parameters.txt").write_text(params_text)
    print(f"  Saved: parameters.txt")

    print("\n" + "=" * 72)
    print(f"Done! All outputs in: {OUT_DIR}")
    print("=" * 72)


if __name__ == "__main__":
    main()
