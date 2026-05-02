#!/usr/bin/env python3
"""
Bridge lens design — publication figure set.

Generates (all saved to OUT_DIR):
  1. bridge_target_rdbur.png          – raw bridge target amplitude, RdBu_r colormap
  2. bridge_roi_particles.png         – bridge amplitude on ROI with 9 particles, bridge-peak scaled
  3. iasa_iter_NNN.png                – IASA snapshots (thickness → phase → pressure → Gorkov),
                                        one PNG per snapshot iteration
  4. bridge_iasa_final_pressure.png   – final reconstructed pressure field from IASA lens
  5. trajectories_ideal.png           – particle trajectories under ideal bridge template
  6. trajectories_iasa.png            – particle trajectories under IASA-reconstructed field

Run:
    python scripts/dev/bridge_lens_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import binary_dilation

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dev.inverse_c_shape_lens_replica import (
    ReplicaConfig,
    make_grid,
    propagate_asm,
    run_iasa_roi,
)

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
_IASA_DIR = (
    PROJECT_ROOT / "results" / "dev"
    / "inverse_bridge_pressure_lens_replica_scaled2x_20260324_095106"
)

BRIDGE_PRESSURE_NPZ  = _IASA_DIR / "bridge_pressure_fields_scaled2x.npz"
BRIDGE_IASA_NPZ      = _IASA_DIR / "bridge_inverse_replica_fields.npz"
BRIDGE_MANIFEST_JSON = _IASA_DIR / "bridge_inverse_replica_manifest.json"
OUT_DIR              = PROJECT_ROOT / "results" / "figures"

# ─────────────────────────────────────────────────────────────────────────────
# Configuration (must match the original IASA run)
# ─────────────────────────────────────────────────────────────────────────────
FREQUENCY_HZ          = 2_150_000.0
C_WATER               = 1480.0
C_LENS                = 2636.0
N_GRID                = 400
TRANSDUCER_DIAM_MM    = 20.0
FOCAL_MM              = 13.21309776965029
OUTSIDE_SUPPRESSION   = 0.0
H_BASE_MM             = 1.0
N_ITER_FULL           = 100
PARTICLE_RADIUS_MM    = 0.05
PARTICLE_DENSITY      = 1050.0
PARTICLE_C_SOUND      = 2350.0
RHO_WATER             = 998.0
ETA_WATER             = 1.0e-3  # Pa·s

# Snapshot iterations (must be <= N_ITER_FULL, in increasing order)
SNAPSHOT_ITERS = [1, 2, 5, 10, 25, 50, 100]

# Source pressure for physical scaling of reconstructed field
SOURCE_PRESSURE_PA = 0.05e6

# Plot constants
CMAP_AMP    = "RdBu_r"
CMAP_THICK  = "viridis"
CMAP_PHASE  = "twilight"
CMAP_PRESS  = "hot"
CMAP_GORKOV = "RdBu_r"

DPI = 190

# Colours for traps
COL_A    = "#e74c3c"   # red
COL_B    = "#3498db"   # blue
COL_NEIGH = "#95a5a6"  # grey


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _style(ax, fontsize: int = 8) -> None:
    """Remove top/right spines; set tick font size."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=fontsize)


def _gorkov_from_complex_p(
    p: np.ndarray,
    dx: float,
    omega: float,
    rho_w: float,
    c_w: float,
    a_m: float,
    rho_p: float,
    c_p: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Gor'kov potential and force field from complex pressure array.
    Returns U [J], Fx [N], Fy [N] on the same grid.
    """
    kappa_w = 1.0 / (rho_w * c_w ** 2)
    kappa_p = 1.0 / (rho_p * c_p ** 2)
    f1 = 1.0 - kappa_p / kappa_w
    f2 = 2.0 * (rho_p - rho_w) / (2.0 * rho_p + rho_w)
    Vp = (4.0 / 3.0) * np.pi * a_m ** 3

    # velocity from pressure gradient
    dp_dy, dp_dx = np.gradient(p, dx, dx)
    vx = -(1.0 / (1j * omega * rho_w)) * dp_dx
    vy = -(1.0 / (1j * omega * rho_w)) * dp_dy
    v2 = np.abs(vx) ** 2 + np.abs(vy) ** 2
    p2 = np.abs(p) ** 2

    U = Vp * (
        (f1 * p2) / (4.0 * rho_w * c_w ** 2)
        - (3.0 * f2 * rho_w * v2) / 8.0
    )
    dU_dy, dU_dx = np.gradient(U, dx, dx)
    Fx = -dU_dx
    Fy = -dU_dy
    return U, Fx, Fy


def _build_cfg() -> ReplicaConfig:
    return ReplicaConfig(
        frequency_hz=FREQUENCY_HZ,
        c_water=C_WATER,
        c_lens=C_LENS,
        transducer_diameter_mm=TRANSDUCER_DIAM_MM,
        focal_distance_mm=FOCAL_MM,
        n_grid=N_GRID,
        h_base_mm=H_BASE_MM,
        n_iter=N_ITER_FULL,
        source_pressure_pa=SOURCE_PRESSURE_PA,
    )


def _lens_to_thickness_mm(lens_field: np.ndarray, cfg: ReplicaConfig, aperture_mask: np.ndarray) -> np.ndarray:
    phi = np.mod(np.angle(lens_field), 2.0 * np.pi)  # [0, 2π]
    t   = cfg.h_base_m + cfg.h_max_m * (phi / (2.0 * np.pi))  # [m]
    t[~aperture_mask] = np.nan
    return t * 1e3  # mm


def _propagate_and_scale(
    lens_field: np.ndarray,
    cfg: ReplicaConfig,
    dx: float,
    aperture_mask: np.ndarray,
) -> np.ndarray:
    """Forward-propagate lens → focal plane, scale to source pressure."""
    p = propagate_asm(lens_field, cfg.k_water, cfg.focal_distance_m, dx)
    scale = SOURCE_PRESSURE_PA / (
        np.sqrt(np.mean(np.abs(lens_field[aperture_mask]) ** 2)) + 1e-12
    )
    return p * scale


def _iasa_with_snapshots(
    cfg: ReplicaConfig,
    aperture_mask: np.ndarray,
    target_amp: np.ndarray,
    roi_mask: np.ndarray,
    dx: float,
    snapshot_iters: list[int],
    outside_suppression: float = 0.0,
) -> dict[int, np.ndarray]:
    """
    Run ROI-constrained IASA and return intermediate lens fields at requested
    iteration indices.  snapshot_iters must be sorted ascending, max <= n_iter.
    Returns {iter_count: lens_field}.
    """
    n   = cfg.n_grid
    z   = cfg.focal_distance_m
    sup = float(np.clip(outside_suppression, 0.0, 1.0))
    outside_roi = ~roi_mask

    lens_field = np.ones((n, n), dtype=complex)
    lens_field[~aperture_mask] = 0.0

    snap_set = set(snapshot_iters)
    max_iter = max(snapshot_iters)
    snapshots: dict[int, np.ndarray] = {}

    for it in range(1, max_iter + 1):
        img_field = propagate_asm(lens_field, cfg.k_water, z, dx)
        img_updated = img_field.copy()
        img_updated[roi_mask] = (
            target_amp[roi_mask] * np.exp(1j * np.angle(img_field[roi_mask]))
        )
        if sup < 1.0:
            img_updated[outside_roi] = (
                sup * np.abs(img_field[outside_roi])
                * np.exp(1j * np.angle(img_field[outside_roi]))
            )
        lens_field = np.exp(
            1j * np.angle(propagate_asm(img_updated, cfg.k_water, -z, dx))
        )
        lens_field[~aperture_mask] = 0.0

        if it in snap_set:
            snapshots[it] = lens_field.copy()
            print(f"  [IASA] snapshot at iter {it}")

    return snapshots


def _simulate_trajectories(
    Fx: np.ndarray,
    Fy: np.ndarray,
    x_grid: np.ndarray,       # 1-D, metres
    y_grid: np.ndarray,       # 1-D, metres
    start_positions: np.ndarray,  # (N,2) in metres, [x, y]
    n_steps: int = 600,
    dt: float = 1e-4,         # s
) -> np.ndarray:
    """
    Overdamped particle dynamics under Gor'kov force.
    γ = 6π η a.  dx/dt = Fx/γ, dy/dt = Fy/γ.

    Returns trajectories array shape (N, n_steps+1, 2) in metres.
    """
    gamma = 6.0 * np.pi * ETA_WATER * (PARTICLE_RADIUS_MM * 1e-3)

    interp_fx = RegularGridInterpolator(
        (y_grid, x_grid), Fx, method="linear", bounds_error=False, fill_value=0.0
    )
    interp_fy = RegularGridInterpolator(
        (y_grid, x_grid), Fy, method="linear", bounds_error=False, fill_value=0.0
    )

    N = start_positions.shape[0]
    traj = np.empty((N, n_steps + 1, 2), dtype=float)
    traj[:, 0, :] = start_positions.copy()

    for s in range(n_steps):
        pos = traj[:, s, :]    # (N,2)  - [x,y]
        pts = np.column_stack([pos[:, 1], pos[:, 0]])   # scipy wants (y,x)
        fx_at = interp_fx(pts)
        fy_at = interp_fy(pts)
        vel_x = fx_at / gamma
        vel_y = fy_at / gamma
        traj[:, s + 1, 0] = pos[:, 0] + vel_x * dt
        traj[:, s + 1, 1] = pos[:, 1] + vel_y * dt

    return traj


def _crop_field(
    field: np.ndarray,
    x_full: np.ndarray,
    y_full: np.ndarray,
    x_lo: float,
    x_hi: float,
    y_lo: float,
    y_hi: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ix = np.where((x_full >= x_lo) & (x_full <= x_hi))[0]
    iy = np.where((y_full >= y_lo) & (y_full <= y_hi))[0]
    return field[np.ix_(iy, ix)], x_full[ix], y_full[iy]


# ─────────────────────────────────────────────────────────────────────────────
# Figure generators
# ─────────────────────────────────────────────────────────────────────────────
def fig_bridge_target_rdbur(
    target_raw_amp: np.ndarray,
    iasa_extent_mm: list[float],
    out_path: Path,
) -> None:
    """PNG 1 – raw bridge target amplitude on IASA grid, RdBu_r."""
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    vmax = float(np.percentile(target_raw_amp[target_raw_amp > 0], 99.5)) \
        if np.any(target_raw_amp > 0) else float(target_raw_amp.max())
    im = ax.imshow(
        target_raw_amp,
        origin="lower",
        extent=iasa_extent_mm,
        cmap=CMAP_AMP,
        vmin=0,
        vmax=vmax,
        aspect="equal",
    )
    plt.colorbar(im, ax=ax, label="|p| [Pa]")
    ax.set_title("Raw bridge target amplitude (RdBu_r)")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    _style(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def fig_bridge_roi_particles(
    p_field: np.ndarray,
    x_full: np.ndarray,
    y_full: np.ndarray,
    x_roi: np.ndarray,
    y_roi: np.ndarray,
    traps_m: np.ndarray,
    idx_a: int,
    idx_b: int,
    out_path: Path,
) -> None:
    """
    PNG 2 – bridge amplitude in the ROI with 9 particles.
    Colorscale set so the bridge tunnel peak is bright red;
    contributions outside the source hotspot appear in blue-white range.
    """
    neigh_idx = np.array(
        sorted(set(range(len(traps_m))) - {idx_a, idx_b}), dtype=int
    )

    # Crop to ROI
    x_lo, x_hi = float(x_roi[0]), float(x_roi[-1])
    y_lo, y_hi = float(y_roi[0]), float(y_roi[-1])
    field_roi, xc, yc = _crop_field(
        np.abs(p_field), x_full, y_full, x_lo, x_hi, y_lo, y_hi
    )
    extent_mm = [xc[0]*1e3, xc[-1]*1e3, yc[0]*1e3, yc[-1]*1e3]
    traps_mm  = traps_m * 1e3

    # Set vmax to bridge hotspot peak so it's bright red;
    # surrounding SW-level amplitude (~20-50 Pa) will appear blue/white.
    vmax = float(np.percentile(field_roi[field_roi > 1e-3], 99.0)) \
        if np.any(field_roi > 1e-3) else float(field_roi.max() + 1e-9)

    fig, ax = plt.subplots(figsize=(6.0, 5.8))
    im = ax.imshow(
        field_roi,
        origin="lower",
        extent=extent_mm,
        cmap=CMAP_AMP,
        vmin=0,
        vmax=vmax,
        aspect="equal",
        interpolation="bicubic",
    )
    plt.colorbar(im, ax=ax, label="|p| [Pa]", fraction=0.046, pad=0.02)

    # Neighbour traps
    if neigh_idx.size:
        ax.scatter(
            traps_mm[neigh_idx, 0], traps_mm[neigh_idx, 1],
            c=COL_NEIGH, s=60, zorder=5, label="neighbours", edgecolors="k", linewidths=0.5,
        )
    # Trap A and B
    ax.scatter([traps_mm[idx_a, 0]], [traps_mm[idx_a, 1]],
               c=COL_A, s=90, marker="^", zorder=6, label=f"A (trap {idx_a})", edgecolors="k", linewidths=0.5)
    ax.scatter([traps_mm[idx_b, 0]], [traps_mm[idx_b, 1]],
               c=COL_B, s=90, marker="v", zorder=6, label=f"B (trap {idx_b})", edgecolors="k", linewidths=0.5)

    ax.set_xlim(extent_mm[0], extent_mm[1])
    ax.set_ylim(extent_mm[2], extent_mm[3])
    ax.set_title("Bridge amplitude on ROI\n(vmax = bridge hotspot peak)")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.legend(fontsize=7, loc="upper right")
    _style(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def fig_iasa_snapshot(
    lens_field: np.ndarray,
    cfg: ReplicaConfig,
    aperture_mask: np.ndarray,
    dx: float,
    iasa_x: np.ndarray,
    iasa_y: np.ndarray,
    iter_n: int,
    out_path: Path,
) -> None:
    """
    One 4-panel snapshot PNG:
      Panel 0: Lens thickness map (viridis, mm)
      Panel 1: Wrapped phase map (twilight, 0-2π rad)
      Panel 2: Forward pressure at focus (hot, kPa)
      Panel 3: Gor'kov potential (RdBu_r, aJ)
    """
    omega = 2.0 * np.pi * FREQUENCY_HZ
    a_m   = PARTICLE_RADIUS_MM * 1e-3
    extent_mm = [iasa_x[0]*1e3, iasa_x[-1]*1e3, iasa_y[0]*1e3, iasa_y[-1]*1e3]

    # Thickness
    thickness_mm = _lens_to_thickness_mm(lens_field, cfg, aperture_mask)
    # Phase
    phase = np.mod(np.angle(lens_field), 2.0 * np.pi)
    phase[~aperture_mask] = np.nan
    # Forward pressure (scaled)
    p_fwd = _propagate_and_scale(lens_field, cfg, dx, aperture_mask)
    p_kpa = np.abs(p_fwd) * 1e-3

    # Gor'kov (in aJ = 1e-18 J)
    U, _, _ = _gorkov_from_complex_p(
        p_fwd, dx, omega, RHO_WATER, C_WATER, a_m,
        PARTICLE_DENSITY, PARTICLE_C_SOUND,
    )
    U_aJ = U * 1e18
    uv = float(np.percentile(np.abs(U_aJ[aperture_mask]), 99.0)) if np.any(aperture_mask) else 1.0
    uv = max(uv, 1e-12)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"Bridge IASA Lens — iteration {iter_n}", fontsize=11, fontweight="bold")

    # Panel 0: thickness
    ax = axes[0]
    t_valid = thickness_mm[aperture_mask]
    im0 = ax.imshow(
        thickness_mm, origin="lower", extent=extent_mm,
        cmap=CMAP_THICK, aspect="equal",
        vmin=float(np.nanmin(thickness_mm)), vmax=float(np.nanmax(thickness_mm)),
    )
    plt.colorbar(im0, ax=ax, label="t [mm]", fraction=0.046, pad=0.02)
    ax.set_title("Lens Thickness")
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    # Aperture circle
    theta_c = np.linspace(0, 2*np.pi, 200)
    r_ap = TRANSDUCER_DIAM_MM / 2
    ax.plot(r_ap*np.cos(theta_c), r_ap*np.sin(theta_c), 'r--', lw=0.7, alpha=0.6)
    _style(ax)

    # Panel 1: phase
    ax = axes[1]
    im1 = ax.imshow(
        phase, origin="lower", extent=extent_mm,
        cmap=CMAP_PHASE, vmin=0, vmax=2.0*np.pi, aspect="equal",
    )
    plt.colorbar(im1, ax=ax, label="phase [rad]", fraction=0.046, pad=0.02)
    ax.set_title("Hologram Phase Map")
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    ax.plot(r_ap*np.cos(theta_c), r_ap*np.sin(theta_c), 'w--', lw=0.7, alpha=0.6)
    _style(ax)

    # Panel 2: pressure
    ax = axes[2]
    vp = float(np.percentile(p_kpa[aperture_mask], 99.5)) if np.any(aperture_mask) else p_kpa.max()
    vp = max(vp, 1e-9)
    im2 = ax.imshow(
        p_kpa, origin="lower", extent=extent_mm,
        cmap=CMAP_PRESS, vmin=0, vmax=vp, aspect="equal",
    )
    plt.colorbar(im2, ax=ax, label="|p| [kPa]", fraction=0.046, pad=0.02)
    ax.set_title("Pressure at Focus")
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    # Zoom to bridge region  (ROI ≈ ±3mm from centre)
    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
    _style(ax)

    # Panel 3: Gor'kov
    ax = axes[3]
    im3 = ax.imshow(
        U_aJ, origin="lower", extent=extent_mm,
        cmap=CMAP_GORKOV, vmin=-uv, vmax=uv, aspect="equal",
    )
    plt.colorbar(im3, ax=ax, label="U [aJ]", fraction=0.046, pad=0.02)
    ax.set_title("Gor'kov Potential")
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
    _style(ax)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def fig_final_pressure(
    p_final: np.ndarray,
    iasa_x: np.ndarray,
    iasa_y: np.ndarray,
    aperture_mask: np.ndarray,
    traps_iasa_mm: np.ndarray,
    idx_a: int,
    idx_b: int,
    out_path: Path,
) -> None:
    """PNG 4 – final IASA lens reconstructed pressure field."""
    extent_mm = [iasa_x[0]*1e3, iasa_x[-1]*1e3, iasa_y[0]*1e3, iasa_y[-1]*1e3]
    amp = np.abs(p_final)
    vmax = float(np.percentile(amp[aperture_mask], 99.5)) if np.any(aperture_mask) else amp.max()
    vmax = max(vmax, 1e-9)

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(
        amp * 1e-3, origin="lower", extent=extent_mm,
        cmap=CMAP_PRESS, vmin=0, vmax=vmax*1e-3, aspect="equal",
    )
    plt.colorbar(im, ax=ax, label="|p| [kPa]")
    ax.set_title("Final IASA lens — pressure at focus")
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)

    # Overlaid trap positions
    neigh_idx = np.array(sorted(set(range(len(traps_iasa_mm))) - {idx_a, idx_b}), dtype=int)
    if neigh_idx.size:
        ax.scatter(traps_iasa_mm[neigh_idx, 0], traps_iasa_mm[neigh_idx, 1],
                   c=COL_NEIGH, s=40, zorder=5, edgecolors="w", linewidths=0.4)
    ax.scatter([traps_iasa_mm[idx_a, 0]], [traps_iasa_mm[idx_a, 1]],
               c=COL_A, s=70, marker="^", zorder=6, label="A", edgecolors="w", linewidths=0.4)
    ax.scatter([traps_iasa_mm[idx_b, 0]], [traps_iasa_mm[idx_b, 1]],
               c=COL_B, s=70, marker="v", zorder=6, label="B", edgecolors="w", linewidths=0.4)
    ax.legend(fontsize=8, loc="upper right")
    _style(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def fig_trajectories(
    traj: np.ndarray,
    Gor: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    traps_m: np.ndarray,
    idx_a: int,
    idx_b: int,
    title: str,
    out_path: Path,
) -> None:
    """
    PNG 5/6 – overdamped particle trajectories overlaid on Gor'kov potential.
    traj: (N, T+1, 2) in metres [x,y]
    """
    N = traj.shape[0]
    neigh_idx = np.array(sorted(set(range(N)) - {idx_a, idx_b}), dtype=int)
    extent_mm = [x_grid[0]*1e3, x_grid[-1]*1e3, y_grid[0]*1e3, y_grid[-1]*1e3]

    G_aJ  = Gor * 1e18
    gv    = float(np.percentile(np.abs(G_aJ), 99.0))
    gv    = max(gv, 1e-12)

    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    im = ax.imshow(
        G_aJ, origin="lower", extent=extent_mm,
        cmap=CMAP_GORKOV, vmin=-gv, vmax=gv, alpha=0.75, aspect="equal",
    )
    plt.colorbar(im, ax=ax, label="U [aJ]", fraction=0.046, pad=0.02)

    # Trajectories
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, N))
    for i in range(N):
        tx = traj[i, :, 0] * 1e3  # mm
        ty = traj[i, :, 1] * 1e3
        c  = COL_A if i == idx_a else (COL_B if i == idx_b else COL_NEIGH)
        ax.plot(tx, ty, lw=1.2, color=c, alpha=0.85, zorder=4)
        # End marker
        ax.plot(tx[-1], ty[-1], ".", ms=5, color=c, zorder=5)

    # Start markers (trap positions)
    if neigh_idx.size:
        ax.scatter(traps_m[neigh_idx, 0]*1e3, traps_m[neigh_idx, 1]*1e3,
                   c=COL_NEIGH, s=55, marker="o", zorder=6, edgecolors="k", lw=0.4)
    ax.scatter([traps_m[idx_a, 0]*1e3], [traps_m[idx_a, 1]*1e3],
               c=COL_A, s=80, marker="^", zorder=7, edgecolors="k", lw=0.5, label="A")
    ax.scatter([traps_m[idx_b, 0]*1e3], [traps_m[idx_b, 1]*1e3],
               c=COL_B, s=80, marker="v", zorder=7, edgecolors="k", lw=0.5, label="B")

    pad = max(0.5, float(np.ptp(traps_m[:, 0])) * 1e3 * 0.4)
    x_lo = min(traps_m[:, 0]) * 1e3 - pad
    x_hi = max(traps_m[:, 0]) * 1e3 + pad
    y_lo = min(traps_m[:, 1]) * 1e3 - pad
    y_hi = max(traps_m[:, 1]) * 1e3 + pad
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_title(title)
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    ax.legend(fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("BRIDGE LENS FIGURES")
    print("=" * 70)

    # ── 0. Load data ──────────────────────────────────────────────────────
    print("\n[1/7] Loading data...")
    bp = np.load(BRIDGE_PRESSURE_NPZ)
    x_full    = bp["x_full"].astype(float)   # original bridge coords (m)
    y_full    = bp["y_full"].astype(float)
    p_bridge  = bp["p_bridge_effective_full"].astype(complex)  # (400,400) bridge field
    traps_m   = bp["traps_m"].astype(float)                   # (9,2) in m
    x_roi     = bp["x_roi"].astype(float)
    y_roi     = bp["y_roi"].astype(float)
    idx_a     = int(bp["idx_a"])
    idx_b     = int(bp["idx_b"])

    ir = np.load(BRIDGE_IASA_NPZ)
    target_raw_amp = ir["target_raw_amp"].astype(float)   # on IASA grid (400,400)
    target_amp     = ir["target_amp"].astype(float)       # normalised
    roi_mask       = ir["roi_mask"].astype(bool)           # (400,400)
    lens_final     = ir["lens_field"].astype(complex)      # final IASA lens (400,400)
    aperture_mask  = ir["aperture_mask"].astype(bool)

    with open(BRIDGE_MANIFEST_JSON) as fj:
        manifest = json.load(fj)

    print(f"  Bridge domain: x=[{x_full[0]*1e3:.2f},{x_full[-1]*1e3:.2f}] mm  "
          f"y=[{y_full[0]*1e3:.2f},{y_full[-1]*1e3:.2f}] mm")
    print(f"  Trap A (idx {idx_a}): ({traps_m[idx_a,0]*1e3:.3f}, {traps_m[idx_a,1]*1e3:.3f}) mm")
    print(f"  Trap B (idx {idx_b}): ({traps_m[idx_b,0]*1e3:.3f}, {traps_m[idx_b,1]*1e3:.3f}) mm")
    print(f"  IASA target_raw_amp: shape={target_raw_amp.shape}, max={target_raw_amp.max():.2f} Pa")

    # ── Build IASA config + grid ──────────────────────────────────────────
    cfg = _build_cfg()
    iasa_x, iasa_y, _, _, _, _, ap_mask2, dx_iasa = make_grid(cfg)
    # Use aperture_mask from the saved data (should match)
    extent_iasa_mm = [iasa_x[0]*1e3, iasa_x[-1]*1e3, iasa_y[0]*1e3, iasa_y[-1]*1e3]

    # Trap positions in IASA coordinate system
    x_center_m = 0.5 * (x_full[0] + x_full[-1])
    y_center_m = 0.5 * (y_full[0] + y_full[-1])
    xc = x_full - x_center_m
    yc = y_full - y_center_m
    field_half_c = max(float(np.abs(xc).max()), float(np.abs(yc).max()))
    aperture_radius_m = TRANSDUCER_DIAM_MM * 0.5e-3
    bridge_scale = aperture_radius_m / max(field_half_c, 1e-12)

    traps_iasa_m = (traps_m[:, :2] - np.array([x_center_m, y_center_m])) * bridge_scale
    traps_iasa_mm = traps_iasa_m * 1e3

    print(f"  Bridge scale: {bridge_scale:.4f}x")
    print(f"  Trap A (IASA mm): {traps_iasa_mm[idx_a]}")
    print(f"  Trap B (IASA mm): {traps_iasa_mm[idx_b]}")

    # ── PNG 1: raw bridge target RdBu_r ──────────────────────────────────
    print("\n[2/7] PNG 1 — bridge target, RdBu_r...")
    fig_bridge_target_rdbur(
        target_raw_amp, extent_iasa_mm,
        OUT_DIR / "bridge_target_rdbur.png",
    )

    # ── PNG 2: bridge ROI with particles ─────────────────────────────────
    print("\n[3/7] PNG 2 — bridge ROI with 9 particles...")
    fig_bridge_roi_particles(
        p_bridge, x_full, y_full, x_roi, y_roi,
        traps_m, idx_a, idx_b,
        OUT_DIR / "bridge_roi_particles.png",
    )

    # ── IASA snapshots ────────────────────────────────────────────────────
    print(f"\n[4/7] Running IASA ({N_ITER_FULL} iters) with snapshots at {SNAPSHOT_ITERS}...")
    snapshots = _iasa_with_snapshots(
        cfg,
        aperture_mask,
        target_amp,
        roi_mask,
        dx_iasa,
        snapshot_iters=SNAPSHOT_ITERS,
        outside_suppression=OUTSIDE_SUPPRESSION,
    )

    # ── PNG 3: Iteration snapshot PNGs ───────────────────────────────────
    print("\n[5/7] Saving IASA iteration figures...")
    for it, lens_snap in sorted(snapshots.items()):
        fig_iasa_snapshot(
            lens_snap, cfg, aperture_mask, dx_iasa,
            iasa_x, iasa_y, it,
            OUT_DIR / f"iasa_iter_{it:03d}.png",
        )

    # ── PNG 4: final pressure ────────────────────────────────────────────
    print("\n[6/7] PNG 4 — final IASA pressure field...")
    omega  = 2.0 * np.pi * FREQUENCY_HZ
    a_m    = PARTICLE_RADIUS_MM * 1e-3
    p_iasa_final = _propagate_and_scale(lens_final, cfg, dx_iasa, aperture_mask)
    fig_final_pressure(
        p_iasa_final, iasa_x, iasa_y, aperture_mask,
        traps_iasa_mm, idx_a, idx_b,
        OUT_DIR / "bridge_iasa_final_pressure.png",
    )

    # ── PNGs 5 & 6: Particle trajectories on IASA grid ──────────────────
    # Compare ideal bridge target vs IASA reconstruction, both on IASA grid.
    # Starting positions are the trap coordinates in IASA-frame metres.
    print("\n[7/7] PNG 5 & 6 — particle trajectories (IASA grid)...")

    start_m_iasa = traps_iasa_m.copy()  # (9,2) in metres on IASA grid

    # Ideal: use bridge target amplitude as real-valued pressure field
    # (Gor'kov depends only on |p|² and |∇p|², so real field is valid for comparison)
    p_ideal_real = target_raw_amp.astype(float)  # (400,400) on IASA grid
    U_ideal, Fx_ideal, Fy_ideal = _gorkov_from_complex_p(
        p_ideal_real.astype(complex),
        dx_iasa, omega, RHO_WATER, C_WATER, a_m,
        PARTICLE_DENSITY, PARTICLE_C_SOUND,
    )
    traj_ideal = _simulate_trajectories(
        Fx_ideal, Fy_ideal, iasa_x, iasa_y,
        start_m_iasa.copy(),
        n_steps=800, dt=5e-5,
    )
    # Build trap_m equivalent in IASA coordinates for the figure helper
    traps_for_fig = traps_iasa_m.copy()
    fig_trajectories(
        traj_ideal, U_ideal, iasa_x, iasa_y,
        traps_for_fig, idx_a, idx_b,
        "Particle trajectories — ideal bridge target (IASA grid)",
        OUT_DIR / "trajectories_ideal.png",
    )

    # IASA resultant: complex focal-plane field from lens propagation
    U_iasa, Fx_iasa, Fy_iasa = _gorkov_from_complex_p(
        p_iasa_final,
        dx_iasa, omega, RHO_WATER, C_WATER, a_m,
        PARTICLE_DENSITY, PARTICLE_C_SOUND,
    )
    traj_iasa = _simulate_trajectories(
        Fx_iasa, Fy_iasa, iasa_x, iasa_y,
        start_m_iasa.copy(),
        n_steps=800, dt=5e-5,
    )
    fig_trajectories(
        traj_iasa, U_iasa, iasa_x, iasa_y,
        traps_for_fig, idx_a, idx_b,
        "Particle trajectories — IASA reconstructed field",
        OUT_DIR / "trajectories_iasa.png",
    )

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("DONE — all figures saved to:", OUT_DIR)
    all_pngs = sorted(OUT_DIR.glob("bridge_target_rdbur.png")) + \
               sorted(OUT_DIR.glob("bridge_roi_particles.png")) + \
               sorted(OUT_DIR.glob("iasa_iter_*.png")) + \
               sorted(OUT_DIR.glob("bridge_iasa_final_pressure.png")) + \
               sorted(OUT_DIR.glob("trajectories_*.png"))
    for p_ in all_pngs:
        print(f"  {p_.name}")
    print("=" * 70)


if __name__ == "__main__":
    main()
