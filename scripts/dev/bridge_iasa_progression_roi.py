#!/usr/bin/env python3
"""
Bridge IASA progression — ROI-zoom figures.

Shows the bridge target amplitude (correctly scaled on the IASA grid,
zoomed to the ROI with trap-particle markers) and how the focal-plane
pressure field evolves through the IASA iterative cycle.

Two output figures (8 panels each, 2 rows × 4 cols, shared colorscale):
  bridge_iasa_prog_pressure_roi.png  –  |p| at focus
  bridge_iasa_prog_gorkov_roi.png    –  Gor'kov potential U

Run:
    python scripts/dev/bridge_iasa_progression_roi.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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
# Configuration  (must match the original IASA run)
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

OUTSIDE_SUPPRESSION = 0.0

# Iterations at which to capture snapshots (ascending; last entry = total iters)
SNAPSHOT_ITERS = [1, 2, 5, 10, 25, 50, 100]

# Padding around the roi_mask bounding box for the zoom window
ROI_PAD_MM = 0.8

DPI         = 190
CMAP_PRESS  = "hot"
CMAP_GORKOV = "RdBu_r"
COL_A       = "#e74c3c"
COL_B       = "#3498db"
COL_NEIGH   = "#95a5a6"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _build_cfg() -> ReplicaConfig:
    return ReplicaConfig(
        frequency_hz=FREQUENCY_HZ,
        c_water=C_WATER,
        c_lens=C_LENS,
        transducer_diameter_mm=TRANSDUCER_DIAM_MM,
        focal_distance_mm=FOCAL_MM,
        n_grid=N_GRID,
        h_base_mm=H_BASE_MM,
        n_iter=max(SNAPSHOT_ITERS),
        source_pressure_pa=SOURCE_PRESSURE_PA,
    )


def _propagate_scaled(
    lens_field: np.ndarray,
    cfg: ReplicaConfig,
    dx: float,
    aperture_mask: np.ndarray,
) -> np.ndarray:
    """Forward-propagate phase-only lens, scale to source pressure."""
    p = propagate_asm(lens_field, cfg.k_water, cfg.focal_distance_m, dx)
    scale = SOURCE_PRESSURE_PA / (
        np.sqrt(np.mean(np.abs(lens_field[aperture_mask]) ** 2)) + 1e-12
    )
    return p * scale


def _gorkov(p: np.ndarray, dx: float, omega: float) -> np.ndarray:
    """Gor'kov potential [J] for a polystyrene bead in water."""
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

    U = Vp * (
        f1 * np.abs(p) ** 2 / (4.0 * rho_w * c_w ** 2)
        - 3.0 * f2 * rho_w * (np.abs(vx) ** 2 + np.abs(vy) ** 2) / 8.0
    )
    return U


def _normalise_target(raw_amp, clip_pct=99.5, gamma=0.9):
    """Normalise raw amplitude to [0, 1] with percentiles over non-zero pixels."""
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


def _iasa_snapshots(
    cfg: ReplicaConfig,
    aperture_mask: np.ndarray,
    target_amp: np.ndarray,
    roi_mask: np.ndarray,
    dx: float,
    snapshot_iters: list[int],
    outside_suppression: float = 0.0,
) -> dict[int, np.ndarray]:
    """
    Run ROI-constrained IASA and capture lens fields at each requested iteration.
    Returns {iter_number: lens_field (complex, phase-only in aperture)}.

    Lens is initialised with random phase (seed=42) inside the aperture to
    avoid the zero-phase trap that locks the first backprop into a Fresnel lens.
    """
    z   = cfg.focal_distance_m
    sup = float(np.clip(outside_suppression, 0.0, 1.0))
    outside_roi = ~roi_mask

    rng = np.random.default_rng(seed=42)
    lens = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, (N_GRID, N_GRID)))
    lens[~aperture_mask] = 0.0

    snap_set = set(snapshot_iters)
    snaps: dict[int, np.ndarray] = {}

    for it in range(1, max(snapshot_iters) + 1):
        img = propagate_asm(lens, cfg.k_water, z, dx)
        upd = img.copy()
        upd[roi_mask] = target_amp[roi_mask] * np.exp(1j * np.angle(img[roi_mask]))
        if sup < 1.0:
            upd[outside_roi] = (
                sup
                * np.abs(img[outside_roi])
                * np.exp(1j * np.angle(img[outside_roi]))
            )
        lens = np.exp(1j * np.angle(propagate_asm(upd, cfg.k_water, -z, dx)))
        lens[~aperture_mask] = 0.0
        if it in snap_set:
            snaps[it] = lens.copy()
            print(f"  [IASA] saved iter {it:3d}")
    return snaps


def _add_particles(
    ax: plt.Axes,
    traps_mm: np.ndarray,
    idx_a: int,
    idx_b: int,
    ms: int = 45,
) -> None:
    """Scatter trap markers (A=red triangle up, B=blue triangle down, rest=grey)."""
    neigh = np.array(sorted(set(range(len(traps_mm))) - {idx_a, idx_b}))
    if neigh.size:
        ax.scatter(
            traps_mm[neigh, 0], traps_mm[neigh, 1],
            c=COL_NEIGH, s=ms, zorder=5, edgecolors="k", linewidths=0.5,
        )
    ax.scatter(
        [traps_mm[idx_a, 0]], [traps_mm[idx_a, 1]],
        c=COL_A, s=ms * 1.4, marker="^", zorder=6, edgecolors="k", linewidths=0.5,
    )
    ax.scatter(
        [traps_mm[idx_b, 0]], [traps_mm[idx_b, 1]],
        c=COL_B, s=ms * 1.4, marker="v", zorder=6, edgecolors="k", linewidths=0.5,
    )


def _style(ax: plt.Axes, fontsize: int = 8) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=fontsize)


def fig_progression(
    panels: list[tuple[str, np.ndarray]],
    iasa_x: np.ndarray,
    iasa_y: np.ndarray,
    roi_xlim: tuple[float, float],
    roi_ylim: tuple[float, float],
    traps_iasa_mm: np.ndarray,
    idx_a: int,
    idx_b: int,
    roi_mask: np.ndarray,
    out_path: Path,
    cmap: str,
    cbar_label: str,
    vmin: float,
    vmax: float,
    suptitle: str,
) -> None:
    """
    Render all panels in a 2×4 grid.  All panels share the same:
      • coordinate zoom (roi_xlim, roi_ylim)
      • colorscale (vmin, vmax)
      • ROI boundary contour (cyan dashed)
      • particle trap markers
    panels: list of (title_string, 2-D numpy array to imshow).
    """
    N     = len(panels)
    ncols = 4
    nrows = int(np.ceil(N / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 5.0, nrows * 4.6),
    )
    axes = np.array(axes).reshape(nrows, ncols)

    ext_mm = [
        iasa_x[0] * 1e3, iasa_x[-1] * 1e3,
        iasa_y[0] * 1e3, iasa_y[-1] * 1e3,
    ]

    mappable = None
    for i, (label, data) in enumerate(panels):
        r, c = divmod(i, ncols)
        ax = axes[r, c]

        mappable = ax.imshow(
            data, origin="lower", extent=ext_mm,
            cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal",
        )

        # ROI boundary as cyan dashed contour
        ax.contour(
            iasa_x * 1e3, iasa_y * 1e3, roi_mask.astype(float),
            levels=[0.5], colors="cyan", linewidths=0.8,
            linestyles="--", alpha=0.75,
        )

        _add_particles(ax, traps_iasa_mm, idx_a, idx_b)

        ax.set_xlim(*roi_xlim)
        ax.set_ylim(*roi_ylim)
        ax.set_title(label, fontsize=9, fontweight="bold")
        ax.set_xlabel("x [mm]", fontsize=8)
        ax.set_ylabel("y [mm]", fontsize=8)
        _style(ax, fontsize=7)

    # Hide any unused panels
    for i in range(N, nrows * ncols):
        r, c = divmod(i, ncols)
        axes[r, c].set_visible(False)

    fig.suptitle(suptitle, fontsize=11, fontweight="bold")
    fig.subplots_adjust(
        right=0.87, top=0.92, bottom=0.08,
        hspace=0.38, wspace=0.32,
    )
    cbar_ax = fig.add_axes([0.895, 0.10, 0.018, 0.78])
    cb = fig.colorbar(mappable, cax=cbar_ax, label=cbar_label)
    cb.ax.tick_params(labelsize=8)

    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("BRIDGE IASA PROGRESSION — ROI-zoom figures")
    print("=" * 70)

    # ── Load data ─────────────────────────────────────────────────────────
    print("\n[1] Loading data...")
    bp = np.load(BRIDGE_PRESSURE_NPZ)
    x_full  = bp["x_full"].astype(float)
    y_full  = bp["y_full"].astype(float)
    traps_m = bp["traps_m"].astype(float)
    idx_a   = int(bp["idx_a"])
    idx_b   = int(bp["idx_b"])

    ir = np.load(BRIDGE_IASA_NPZ)
    target_raw_amp = ir["target_raw_amp"].astype(float)  # raw Pa, IASA grid
    roi_mask      = ir["roi_mask"].astype(bool)
    aperture_mask = ir["aperture_mask"].astype(bool)

    # Re-normalise from raw, computing percentiles over non-zero pixels
    # only (the pre-baked target_amp used global percentiles which clip
    # the entire corridor to 1.0 because it occupies <2% of the grid).
    target_amp = _normalise_target(target_raw_amp) * aperture_mask.astype(float)

    cfg = _build_cfg()
    iasa_x, iasa_y, _, _, _, _, _, dx_iasa = make_grid(cfg)

    # ── Trap positions in IASA coordinate frame ───────────────────────────
    xc_m = 0.5 * (x_full[0] + x_full[-1])
    yc_m = 0.5 * (y_full[0] + y_full[-1])
    field_half = max(
        float(np.abs(x_full - xc_m).max()),
        float(np.abs(y_full - yc_m).max()),
    )
    bridge_scale  = (TRANSDUCER_DIAM_MM * 0.5e-3) / max(field_half, 1e-12)
    traps_iasa_m  = (traps_m[:, :2] - np.array([xc_m, yc_m])) * bridge_scale
    traps_iasa_mm = traps_iasa_m * 1e3

    print(f"  Bridge scale factor: {bridge_scale:.4f}x")
    print(f"  Trap A (IASA): ({traps_iasa_mm[idx_a, 0]:.3f}, {traps_iasa_mm[idx_a, 1]:.3f}) mm")
    print(f"  Trap B (IASA): ({traps_iasa_mm[idx_b, 0]:.3f}, {traps_iasa_mm[idx_b, 1]:.3f}) mm")

    # ── ROI bounding box for zoom window ─────────────────────────────────
    roi_rows, roi_cols = np.where(roi_mask)
    roi_xlim = (
        iasa_x[roi_cols.min()] * 1e3 - ROI_PAD_MM,
        iasa_x[roi_cols.max()] * 1e3 + ROI_PAD_MM,
    )
    roi_ylim = (
        iasa_y[roi_rows.min()] * 1e3 - ROI_PAD_MM,
        iasa_y[roi_rows.max()] * 1e3 + ROI_PAD_MM,
    )
    print(f"  ROI zoom: x={roi_xlim} mm,  y={roi_ylim} mm")

    # ── Run IASA with snapshots ───────────────────────────────────────────
    print(f"\n[2] Running IASA ({max(SNAPSHOT_ITERS)} iterations)...")
    snapshots = _iasa_snapshots(
        cfg, aperture_mask, target_amp, roi_mask,
        dx_iasa, SNAPSHOT_ITERS, OUTSIDE_SUPPRESSION,
    )

    # ── Forward-propagate each snapshot lens ────────────────────────────
    print("\n[3] Forward-propagating snapshot lenses...")
    p_fields: dict[int, np.ndarray] = {}
    for it in sorted(snapshots):
        p_fields[it] = _propagate_scaled(
            snapshots[it], cfg, dx_iasa, aperture_mask
        )
        peak = np.abs(p_fields[it]).max() * 1e-3
        print(f"  iter {it:3d}: |p|_peak = {peak:.2f} kPa")

    # ── Shared colorscale: anchor to the final iteration's ROI peak ───────
    # (so early iterations that lack the bridge pattern appear dim/flat,
    #  and convergence to the target is immediately visible)
    p_ref_roi = np.abs(p_fields[max(SNAPSHOT_ITERS)])[roi_mask]
    vmax_press = float(np.percentile(p_ref_roi, 99.0)) if p_ref_roi.size else 1.0
    vmax_press = max(vmax_press, 1.0)

    # Scale target_amp → same physical units as the reconstructed fields.
    # target_amp ∈ [0,1] represents the normalised amplitude enforced by IASA,
    # so target_amp * vmax_press maps perfectly onto the shared colorscale.
    target_press_display = target_amp * vmax_press

    omega = 2.0 * np.pi * FREQUENCY_HZ

    # ── Pressure panels ──────────────────────────────────────────────────
    print("\n[4] Saving pressure progression figure...")
    press_panels: list[tuple[str, np.ndarray]] = [
        ("Target (normalised)", target_press_display)
    ]
    for it in sorted(p_fields):
        press_panels.append((f"IASA  iter {it}", np.abs(p_fields[it])))

    fig_progression(
        press_panels, iasa_x, iasa_y, roi_xlim, roi_ylim,
        traps_iasa_mm, idx_a, idx_b, roi_mask,
        out_path=OUT_DIR / "bridge_iasa_prog_pressure_roi.png",
        cmap=CMAP_PRESS,
        cbar_label="|p| [Pa]  (shared scale)",
        vmin=0.0,
        vmax=vmax_press,
        suptitle=(
            "Bridge IASA progression — |p| at focus, ROI zoom\n"
            "(target panel normalised to iter-100 peak; "
            "cyan = ROI boundary; ▲=trap A, ▽=trap B)"
        ),
    )

    # ── Gor'kov panels ───────────────────────────────────────────────────
    print("\n[5] Saving Gor'kov progression figure...")

    # Compute Gor'kov for each field (convert to aJ for display)
    U_target = _gorkov(target_press_display.astype(complex), dx_iasa, omega) * 1e18
    U_fields: dict[int, np.ndarray] = {
        it: _gorkov(p_fields[it], dx_iasa, omega) * 1e18
        for it in sorted(p_fields)
    }

    # Shared Gor'kov colorscale anchored to iter-100 within ROI
    U_ref_roi = U_fields[max(SNAPSHOT_ITERS)][roi_mask]
    uv = float(np.percentile(np.abs(U_ref_roi), 99.0)) if U_ref_roi.size else 1e-18
    uv = max(uv, 1e-18)

    gorkov_panels: list[tuple[str, np.ndarray]] = [
        ("Target (normalised)", U_target)
    ]
    for it in sorted(U_fields):
        gorkov_panels.append((f"IASA  iter {it}", U_fields[it]))

    fig_progression(
        gorkov_panels, iasa_x, iasa_y, roi_xlim, roi_ylim,
        traps_iasa_mm, idx_a, idx_b, roi_mask,
        out_path=OUT_DIR / "bridge_iasa_prog_gorkov_roi.png",
        cmap=CMAP_GORKOV,
        cbar_label="U [aJ]  (shared scale)",
        vmin=-uv,
        vmax=uv,
        suptitle=(
            "Bridge IASA progression — Gor'kov potential U, ROI zoom\n"
            "(target panel normalised to iter-100 peak; "
            "cyan = ROI boundary; ▲=trap A, ▽=trap B)"
        ),
    )

    print("\n" + "=" * 70)
    print("Done!  Files saved to:", OUT_DIR)
    print("  bridge_iasa_prog_pressure_roi.png")
    print("  bridge_iasa_prog_gorkov_roi.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
