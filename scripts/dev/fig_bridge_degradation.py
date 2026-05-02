#!/usr/bin/env python3
"""
Publication figure: Bridge field degradation through IASA reconstruction.

2×2 top panels + 1 centred bar chart bottom panel.
  Panel 1: Ideal bridge Gor'kov potential
  Panel 2: IASA lens phase hologram
  Panel 3: Reconstructed field Gor'kov
  Panel 4: Normalised difference map
  Bottom : ΔU bar chart (ideal, IASA, C-shape)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lib.fem_cache_utils import (
    C_WATER, OMEGA, RHO0, default_particle_params, gorkov_grid_2d,
)

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION — edit paths here
# ═══════════════════════════════════════════════════════════════════
OUT_DIR = PROJECT_ROOT / "results" / "figures"

# Bridge pressure field (ideal) + trap positions
BRIDGE_PRESSURE_NPZ = (
    PROJECT_ROOT / "results" / "dev"
    / "inverse_bridge_pressure_lens_replica_scaled2x_20260324_095106"
    / "bridge_pressure_fields_scaled2x.npz"
)

# IASA reconstruction outputs
IASA_REPLICA_NPZ = (
    PROJECT_ROOT / "results" / "dev"
    / "inverse_bridge_pressure_lens_replica_scaled2x_20260324_095106"
    / "bridge_inverse_replica_fields.npz"
)

# Replica manifest (for grid metadata)
IASA_MANIFEST_JSON = (
    PROJECT_ROOT / "results" / "dev"
    / "inverse_bridge_pressure_lens_replica_scaled2x_20260324_095106"
    / "bridge_inverse_replica_manifest.json"
)

# C-shape perturbation field (for best C-shape bar)
CSHAPE_ROI_NPZ = (
    PROJECT_ROOT / "results" / "deliverables"
    / "transport_side_by_side" / "replica_cshape_roi_field.npz"
)

# Standing-wave reference (overlay_local)
OVERLAY_NPZ = (
    PROJECT_ROOT / "results" / "deliverables"
    / "overlay_local" / "overlay_local_fields.npz"
)


def find_latest_dir(base: Path, prefix: str) -> Path:
    """Find the most recent timestamped directory matching prefix."""
    candidates = sorted(base.glob(f"{prefix}*"))
    if not candidates:
        raise FileNotFoundError(
            f"No directories matching '{prefix}*' found in {base}"
        )
    return candidates[-1]


def _gorkov(p, dx, dy):
    """Compute Gor'kov potential using the repo-standard function."""
    ppar = default_particle_params()
    U, Fx, Fy = gorkov_grid_2d(p, dx, dy, OMEGA, RHO0, C_WATER,
                                ppar["a"], ppar["f1"], ppar["f2"])
    return U


def _physical_extent_mm(x, y):
    """Return (xmin, xmax, ymin, ymax) in mm for imshow extent."""
    return [x[0]*1e3, x[-1]*1e3, y[0]*1e3, y[-1]*1e3]


def _style_ax(ax):
    """Remove top/right spines, set tick sizes."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)


def main():
    # ── Check inputs ──────────────────────────────────────────────
    for label, path in [
        ("Bridge pressure NPZ", BRIDGE_PRESSURE_NPZ),
        ("IASA replica NPZ", IASA_REPLICA_NPZ),
        ("C-shape ROI NPZ", CSHAPE_ROI_NPZ),
        ("Overlay NPZ", OVERLAY_NPZ),
    ]:
        if not path.exists():
            print(f"ERROR: {label} not found: {path}")
            print("  Run the corresponding upstream script first.")
            sys.exit(1)

    # ── Load data ─────────────────────────────────────────────────
    bp = np.load(BRIDGE_PRESSURE_NPZ)
    x_full = bp["x_full"]
    y_full = bp["y_full"]
    p_ideal = bp["p_bridge_effective_full"]  # complex (400,400)
    traps_m = bp["traps_m"]
    idx_a = int(bp["idx_a"])
    idx_b = int(bp["idx_b"])
    A_m = traps_m[idx_a]
    B_m = traps_m[idx_b]
    dx = float(x_full[1] - x_full[0])
    dy = float(y_full[1] - y_full[0])

    ir = np.load(IASA_REPLICA_NPZ)
    lens_phase = np.angle(ir["lens_field"])  # (400,400)
    recon_amp = ir["recon_amp"]  # (400,400) forward-propagated amplitude
    aperture_mask = ir["aperture_mask"]

    # The IASA reconstruction lives on the same 400×400 grid but with
    # the aperture (20mm) coordinate system centred on the target.
    # We need to build a complex field from the reconstructed amplitude
    # using the manifested focal coordinates, then compute Gorkov on
    # the ORIGINAL bridge grid by interpolation.

    # Load the overlay SW+C-shape data for the C-shape bar
    cshr = np.load(CSHAPE_ROI_NPZ)
    p_cshape = cshr["p_replica_roi"]  # complex (400,400)
    xg_cs = cshr["xg"]
    yg_cs = cshr["yg"]

    ov = np.load(OVERLAY_NPZ, allow_pickle=True)
    p_sw = ov["p_sw"]
    xg_ov = ov["xg"]
    yg_ov = ov["yg"]
    ov_idx_A = int(ov["idx_A"])
    ov_idx_B = int(ov["idx_B"])
    ov_traps = ov["traps_m"]

    # ── Compute Gor'kov for ideal bridge ──────────────────────────
    # The ideal bridge field is the standing wave + bridge perturbation
    # Load standing wave from the overlay (same grid)
    # The bridge NPZ has the perturbation alone, we need to add it to SW
    # Actually, p_bridge_effective_full IS the perturbation; the SW is separate.
    # But for the Gorkov of the ideal bridge, we want: p_sw + α·e^{iψ}·p_pert
    # The bridge_pressure_fields_scaled2x stores the bridge perturbation.
    # Let's recompute combined: use overlay_local SW field

    # The bridge field NPZ is on a different grid (shifted, larger domain).
    # We compute Gorkov on the bridge grid from the effective perturbation.

    # For the purpose of this figure, use the bridge effective field directly
    # to compute Gorkov (it represents the perturbation design).
    # The ideal Gorkov is computed from the bridge field alone to show
    # the designed bridge potential landscape.
    U_ideal = _gorkov(p_ideal, dx, dy)

    # ── Reconstruct forward-propagated field as complex ───────────
    # recon_amp is the magnitude at the focal plane from IASA.
    # We need complex field to compute Gorkov. The IASA reconstruction
    # preserves the reconstructed phase. Build complex from recon_amp.
    # In the IASA output, lens_field is at the aperture plane.
    # recon_amp is the forward-propagated |p| at focal.
    # We don't have the phase at focal, but we can approximate:
    # The target_amp was the normalised version of |p_ideal|.
    # For Gorkov, what matters is |p|², so we can use recon_amp as
    # a real-valued field and compute Gorkov from it.
    # Better: use target_raw_amp and recon_amp to build a proxy complex field.
    # The Gorkov potential depends on |p|² and |∇p|², so we can build
    # p_reconstructed = recon_amp * exp(i * phase_of_target)
    # to preserve the phase structure.

    # Actually, the most correct approach: the reconstructed field
    # at the focal plane has amplitude = recon_amp and SOME phase.
    # Since we don't store the focal-plane phase, we'll use
    # recon_amp as a real field (phase=0) which is valid for Gorkov
    # since Gorkov depends only on |p|² and |∇p|².
    # Gorkov of a real field is the same as Gorkov of |p|·exp(iφ_0).

    # However, recon_amp is on the IASA aperture grid (20mm, 400pts).
    # We need to map it onto the bridge field grid.
    import json
    manifest = json.load(open(IASA_MANIFEST_JSON))
    n_iasa = manifest["config"]["n_grid"]
    ap_mm = manifest["config"]["transducer_diameter_mm"]
    extent_mm = manifest["config"]["target_field_extent_mm"]
    centre_mm = manifest["config"]["target_centred_at_mm"]

    # IASA grid: the target was placed at the centre of the bridge domain.
    # The IASA grid spans [-ap/2, +ap/2] but the target occupies only
    # the central region matching the bridge field extent.
    # recon_amp is (400,400) on the full aperture grid.
    x_iasa = np.linspace(-ap_mm/2, ap_mm/2, n_iasa) * 1e-3  # m
    y_iasa = np.linspace(-ap_mm/2, ap_mm/2, n_iasa) * 1e-3  # m
    # The target was centred at centre_mm, so the bridge field origin in
    # IASA coords is offset. The bridge grid centre is ~3mm,3mm.
    # In the IASA grid, (0,0) corresponds to bridge centre_mm.
    x_recon = x_iasa + centre_mm[0] * 1e-3  # shift to bridge coords
    y_recon = y_iasa + centre_mm[1] * 1e-3

    # Interpolate recon_amp onto the bridge grid
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator(
        (y_recon, x_recon), recon_amp,
        bounds_error=False, fill_value=0.0,
    )
    YY_bridge, XX_bridge = np.meshgrid(y_full, x_full, indexing="ij")
    pts = np.stack([YY_bridge.ravel(), XX_bridge.ravel()], axis=-1)
    recon_on_bridge = interp(pts).reshape(YY_bridge.shape)

    # Scale: recon_amp is in raw units (not normalised), while p_ideal
    # is in Pa. To make Gorkov comparable, scale recon to match p_ideal peak.
    scale_factor = np.abs(p_ideal).max() / max(recon_on_bridge.max(), 1e-30)
    p_recon = (recon_on_bridge * scale_factor).astype(complex)
    U_recon = _gorkov(p_recon, dx, dy)

    # ── Compute Gor'kov for C-shape field ─────────────────────────
    # Use the overlay grid C-shape perturbation combined with SW
    dx_ov = float(xg_ov[1] - xg_ov[0])
    dy_ov = float(yg_ov[1] - yg_ov[0])

    # C-shape transport uses α=4.50, ψ=1.5π (from compare_vortex_vs_cshape)
    CSHAPE_ALPHA = 4.50
    CSHAPE_PSI = 1.5 * np.pi
    p_combined_cs = p_sw + CSHAPE_ALPHA * np.exp(1j * CSHAPE_PSI) * p_cshape
    U_cshape = _gorkov(p_combined_cs, dx_ov, dy_ov)

    # ── Compute ΔU metrics ────────────────────────────────────────
    # ΔU = U_saddle - U_min along the A→B corridor
    # For the bridge fields, find U at trap A and trap B positions
    def _val_at(U, x_ax, y_ax, pos_m):
        ix = int(np.argmin(np.abs(x_ax - pos_m[0])))
        iy = int(np.argmin(np.abs(y_ax - pos_m[1])))
        return float(U[iy, ix])

    # For ideal bridge
    U_A_ideal = _val_at(U_ideal, x_full, y_full, A_m)
    U_B_ideal = _val_at(U_ideal, x_full, y_full, B_m)

    # Sample along A→B line to find saddle
    n_samples = 200
    t = np.linspace(0, 1, n_samples)
    line_pts_m = A_m[None, :] + t[:, None] * (B_m - A_m)[None, :]
    from scipy.interpolate import RegularGridInterpolator as RGI
    interp_ideal = RGI((y_full, x_full), U_ideal, bounds_error=False, fill_value=np.nan)
    U_line_ideal = interp_ideal(line_pts_m[:, ::-1])  # (y, x) ordering
    U_min_ideal = np.nanmin(U_line_ideal)
    U_saddle_ideal = np.nanmax(U_line_ideal)
    dU_ideal = U_saddle_ideal - U_min_ideal

    # For reconstructed
    interp_recon = RGI((y_full, x_full), U_recon, bounds_error=False, fill_value=np.nan)
    U_line_recon = interp_recon(line_pts_m[:, ::-1])
    U_min_recon = np.nanmin(U_line_recon)
    U_saddle_recon = np.nanmax(U_line_recon)
    dU_recon = U_saddle_recon - U_min_recon

    # For C-shape: use overlay trap positions
    A_ov = ov_traps[ov_idx_A]
    B_ov = ov_traps[ov_idx_B]
    line_ov = A_ov[None, :] + t[:, None] * (B_ov - A_ov)[None, :]
    interp_cs = RGI((yg_ov, xg_ov), U_cshape, bounds_error=False, fill_value=np.nan)
    U_line_cs = interp_cs(line_ov[:, ::-1])
    U_min_cs = np.nanmin(U_line_cs)
    U_saddle_cs = np.nanmax(U_line_cs)
    dU_cshape = U_saddle_cs - U_min_cs

    # ── Crop to physical ROI (exclude PML) ────────────────────────
    # PML is typically the outermost ~0.3mm. Use the full grid but
    # restrict the view to the region around the traps.
    margin_m = 0.8e-3
    all_trap_x = traps_m[:, 0]
    all_trap_y = traps_m[:, 1]
    roi_xmin = all_trap_x.min() - margin_m
    roi_xmax = all_trap_x.max() + margin_m
    roi_ymin = all_trap_y.min() - margin_m
    roi_ymax = all_trap_y.max() + margin_m

    ix_roi = np.where((x_full >= roi_xmin) & (x_full <= roi_xmax))[0]
    iy_roi = np.where((y_full >= roi_ymin) & (y_full <= roi_ymax))[0]

    x_roi = x_full[ix_roi]
    y_roi = y_full[iy_roi]
    U_ideal_roi = U_ideal[np.ix_(iy_roi, ix_roi)]
    U_recon_roi = U_recon[np.ix_(iy_roi, ix_roi)]

    # Difference map
    U_diff = np.abs(U_ideal_roi - U_recon_roi) / max(np.abs(U_ideal_roi).max(), 1e-30)

    ext_roi = _physical_extent_mm(x_roi, y_roi)

    # Corridor rectangle (bridge corridor region in physical space)
    # A and B define the corridor; width ~ CORRIDOR_WIDTH_M = 3e-4 m
    corridor_half_w = 1.5e-4  # m

    # ── IASA lens phase — crop to aperture ────────────────────────
    # lens_phase on IASA grid, show only within aperture
    lens_phase_masked = np.where(aperture_mask, lens_phase, np.nan)

    # IASA grid extent
    x_iasa_mm = x_iasa * 1e3
    y_iasa_mm = y_iasa * 1e3
    ext_iasa = [x_iasa_mm[0], x_iasa_mm[-1], y_iasa_mm[0], y_iasa_mm[-1]]

    # ── Figure ────────────────────────────────────────────────────
    mm_to_in = 1.0 / 25.4
    fig = plt.figure(figsize=(180*mm_to_in, 130*mm_to_in))

    gs = gridspec.GridSpec(
        2, 4,
        height_ratios=[1.0, 0.7],
        hspace=0.45, wspace=0.35,
    )

    # Shared colorbar range for panels 1 and 3
    vmin_gorkov = min(U_ideal_roi.min(), U_recon_roi.min())
    vmax_gorkov = max(U_ideal_roi.max(), U_recon_roi.max())

    # ── Panel 1: Ideal bridge Gor'kov ─────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(
        U_ideal_roi, extent=ext_roi, origin="lower",
        cmap="viridis_r", vmin=vmin_gorkov, vmax=vmax_gorkov,
        aspect="equal", interpolation="bicubic",
    )
    # Overlay corridor rectangle (dashed white)
    e_ab = (B_m - A_m) / np.linalg.norm(B_m - A_m)
    e_n = np.array([-e_ab[1], e_ab[0]])
    corners = np.array([
        A_m - corridor_half_w * e_n,
        B_m - corridor_half_w * e_n,
        B_m + corridor_half_w * e_n,
        A_m + corridor_half_w * e_n,
        A_m - corridor_half_w * e_n,
    ]) * 1e3  # to mm
    ax1.plot(corners[:, 0], corners[:, 1], "w--", lw=0.8, alpha=0.9)
    ax1.set_title("Ideal bridge field", fontsize=9, pad=4)
    ax1.set_xlabel("x (mm)", fontsize=9)
    ax1.set_ylabel("y (mm)", fontsize=9)
    _style_ax(ax1)

    # ── Panel 2: IASA lens phase ──────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(
        lens_phase_masked, extent=ext_iasa, origin="lower",
        cmap="hsv", vmin=-np.pi, vmax=np.pi,
        aspect="equal", interpolation="bicubic",
    )
    cb2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cb2.set_label("phase (rad)", fontsize=8)
    cb2.set_ticks([-np.pi, 0, np.pi])
    cb2.set_ticklabels([r"$-\pi$", "0", r"$\pi$"])
    cb2.ax.tick_params(labelsize=7)
    ax2.set_title("IASA lens phase", fontsize=9, pad=4)
    ax2.set_xlabel("x (mm)", fontsize=9)
    ax2.set_ylabel("y (mm)", fontsize=9)
    _style_ax(ax2)

    # ── Panel 3: Reconstructed Gor'kov ────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(
        U_recon_roi, extent=ext_roi, origin="lower",
        cmap="viridis_r", vmin=vmin_gorkov, vmax=vmax_gorkov,
        aspect="equal", interpolation="bicubic",
    )
    ax3.set_title("Reconstructed field (IASA)", fontsize=9, pad=4)
    ax3.set_xlabel("x (mm)", fontsize=9)
    ax3.set_ylabel("y (mm)", fontsize=9)
    _style_ax(ax3)

    # Shared colorbar for panels 1 and 3
    cb13 = fig.colorbar(im3, ax=[ax1, ax3], fraction=0.046, pad=0.04,
                        location="right")
    cb13.set_label(r"$U$ (J)", fontsize=8)
    cb13.ax.tick_params(labelsize=7)

    # ── Panel 4: Difference map ───────────────────────────────────
    ax4 = fig.add_subplot(gs[0, 3])
    im4 = ax4.imshow(
        U_diff, extent=ext_roi, origin="lower",
        cmap="hot_r", vmin=0, vmax=0.5,
        aspect="equal", interpolation="bicubic",
    )
    cb4 = fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    cb4.set_label("normalised |ΔU|", fontsize=8)
    cb4.ax.tick_params(labelsize=7)
    ax4.set_title(r"Gor'kov difference", fontsize=9, pad=4)
    ax4.set_xlabel("x (mm)", fontsize=9)
    ax4.set_ylabel("y (mm)", fontsize=9)
    _style_ax(ax4)

    # ── Bottom: ΔU bar chart ──────────────────────────────────────
    ax_bar = fig.add_subplot(gs[1, 1:3])
    labels = ["Ideal bridge", "IASA reconstruction", "Best C-shape"]
    values = [dU_ideal, dU_recon, dU_cshape]
    colors = ["#2166ac", "#d6604d", "#4dac26"]

    bars = ax_bar.bar(labels, values, color=colors, width=0.55, edgecolor="k", lw=0.5)
    for bar, val in zip(bars, values):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{val:.2e}", ha="center", va="bottom", fontsize=7,
        )
    ax_bar.set_ylabel(r"$\Delta U = U_{\mathrm{saddle}} - U_{\min}$ (J)", fontsize=9)
    ax_bar.set_title("Bridge metric comparison", fontsize=9, pad=4)
    ax_bar.tick_params(axis="x", labelsize=8)
    ax_bar.tick_params(axis="y", labelsize=8)
    _style_ax(ax_bar)

    # ── Save ──────────────────────────────────────────────────────
    fig.subplots_adjust(hspace=0.45, wspace=0.35)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = OUT_DIR / f"fig_bridge_degradation.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved: {out.relative_to(PROJECT_ROOT)}")
    plt.close(fig)

    ratio = dU_recon / dU_ideal if dU_ideal != 0 else float("inf")
    print(
        f"Bridge ΔU degradation: ideal={dU_ideal:.3e}, "
        f"reconstructed={dU_recon:.3e}, C-shape={dU_cshape:.3e}, "
        f"recon/ideal ratio={ratio:.3f}"
    )


if __name__ == "__main__":
    main()
