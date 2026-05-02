#!/usr/bin/env python3
"""
Publication figure: C-shape lens failure analysis.

Top panel:  IASA convergence — dual y-axis (correlation + leakage)
Bottom panel: Neighbour displacement spatial map under C-shape field
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.interpolate import RegularGridInterpolator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lib.particle_dynamics_utils import gorkov_normalised, DT_DEFAULT, SCALE as PDU_SCALE

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════
OUT_DIR = PROJECT_ROOT / "results" / "figures"

# C-shape IASA study (with convergence history)
CSHAPE_STUDY_DIR = (
    PROJECT_ROOT / "results" / "inverse_c_shape_lens_study_20260316_171745"
)
# Use the 0.6mm case (closest to achievability threshold)
CSHAPE_CASE = "case_D_0p6mm"

# Overlay fields (SW + C-shape perturbation on the overlay grid)
OVERLAY_NPZ = (
    PROJECT_ROOT / "results" / "deliverables"
    / "overlay_local" / "overlay_local_fields.npz"
)

# C-shape replica ROI field (from transport side-by-side)
CSHAPE_ROI_NPZ = (
    PROJECT_ROOT / "results" / "deliverables"
    / "transport_side_by_side" / "replica_cshape_roi_field.npz"
)


def find_latest_dir(base: Path, prefix: str) -> Path:
    candidates = sorted(base.glob(f"{prefix}*"))
    if not candidates:
        raise FileNotFoundError(f"No directories matching '{prefix}*' in {base}")
    return candidates[-1]


def _style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)


def main():
    # ── Check inputs ──────────────────────────────────────────────
    final_npz_path = CSHAPE_STUDY_DIR / CSHAPE_CASE / "npz" / "final_design_data.npz"
    for label, path in [
        ("C-shape final NPZ", final_npz_path),
        ("Overlay NPZ", OVERLAY_NPZ),
        ("C-shape ROI NPZ", CSHAPE_ROI_NPZ),
    ]:
        if not path.exists():
            print(f"ERROR: {label} not found: {path}")
            print("  Run the corresponding upstream script first.")
            sys.exit(1)

    # ── Load C-shape convergence data ─────────────────────────────
    fd = np.load(final_npz_path, allow_pickle=True)
    history_iter = fd["history_iteration"]   # (180,)
    history_corr = fd["history_amp_corr"]    # (180,) correlation
    p_target = fd["p_target"]                # (1024,1024) complex
    p_forward = fd["p_forward"]              # (1024,1024) complex
    x_cs = fd["x"]                           # (1024,)
    y_cs = fd["y"]                           # (1024,)

    # Compute leakage fraction at each snapshot
    # We have full history_amp_corr and history_amp_rmse.
    # For leakage: compute from target vs forward at final state,
    # and show the convergence of amp_corr (which tracks alignment).
    # If per-iteration leakage is not stored, compute from the
    # correlation + RMSE relationship.

    # Build the C-shape ROI mask from the target
    amp_target = np.abs(p_target)
    roi_mask = amp_target > 0.01 * amp_target.max()
    total_energy = np.sum(np.abs(p_forward) ** 2)
    roi_energy = np.sum(np.abs(p_forward[roi_mask]) ** 2)
    leakage_final = 1.0 - roi_energy / max(total_energy, 1e-30)

    # For per-iteration leakage, we can load snapshots if available
    snapshot_dir = CSHAPE_STUDY_DIR / CSHAPE_CASE / "npz"
    snapshot_files = sorted(snapshot_dir.glob("snapshot_iter_*.npz"))

    snap_iters = []
    snap_leakage = []
    snap_corr = []
    for sf in snapshot_files:
        sd = np.load(sf, allow_pickle=True)
        it = int(sd["iteration"])
        p_lens = sd["p_lens_field"]  # complex
        # Forward propagate? No — p_lens_field is already the focal field
        # Actually p_lens_field is the *lens* field, p_target_field is the target.
        # We need to compute leakage from the focal-plane field.
        # The snapshot stores p_target_field which is the constrained field at target.
        p_snap = sd["p_target_field"]  # constrained field at target plane
        snap_en_total = np.sum(np.abs(p_snap) ** 2)
        snap_en_roi = np.sum(np.abs(p_snap[roi_mask]) ** 2)
        snap_leak = 1.0 - snap_en_roi / max(snap_en_total, 1e-30)

        # Compute correlation with target amplitude
        a_snap = np.abs(p_snap)
        a_targ = amp_target
        mask = a_targ > 0
        if mask.any():
            corr = float(np.corrcoef(a_snap[mask].ravel(),
                                      a_targ[mask].ravel())[0, 1])
        else:
            corr = 0.0

        snap_iters.append(it)
        snap_leakage.append(snap_leak)
        snap_corr.append(corr)

    snap_iters = np.array(snap_iters)
    snap_leakage = np.array(snap_leakage)
    snap_corr = np.array(snap_corr)
    sort_idx = np.argsort(snap_iters)
    snap_iters = snap_iters[sort_idx]
    snap_leakage = snap_leakage[sort_idx]
    snap_corr = snap_corr[sort_idx]

    # Merge full history_corr (180 points) with snapshot leakage (5 points)
    # Use full history for correlation, interpolate leakage
    use_full_corr = True

    # ── Load overlay + C-shape fields for displacement map ────────
    ov = np.load(OVERLAY_NPZ, allow_pickle=True)
    p_sw = ov["p_sw"]
    p_cshape = ov["p_cshape"]
    xg = ov["xg"]
    yg = ov["yg"]
    traps_m = ov["traps_m"]
    idx_A = int(ov["idx_A"])
    idx_B = int(ov["idx_B"])
    neigh_idx = ov["neigh_idx"]

    dx = float(xg[1] - xg[0])
    dy = float(yg[1] - yg[0])

    # C-shape transport parameters (from compare_vortex_vs_cshape.py)
    CSHAPE_ALPHA = 4.50
    CSHAPE_PSI = 1.5 * np.pi

    # Combine fields
    p_combined = p_sw + CSHAPE_ALPHA * np.exp(1j * CSHAPE_PSI) * p_cshape
    U_comb, Fx_comb, Fy_comb = gorkov_normalised(p_combined, dx, dy)

    # Compute Gorkov of SW only for background
    U_sw, _, _ = gorkov_normalised(p_sw, dx, dy)

    # ── Compute per-trap displacement ─────────────────────────────
    # For each SW trap, integrate overdamped dynamics under C-shape
    # field for t=200 steps and record max displacement.
    interp_fx = RegularGridInterpolator(
        (yg, xg), Fx_comb, bounds_error=False, fill_value=0.0,
    )
    interp_fy = RegularGridInterpolator(
        (yg, xg), Fy_comb, bounds_error=False, fill_value=0.0,
    )

    T_STEPS = 200
    dt = DT_DEFAULT
    SCALE = PDU_SCALE  # overdamped velocity scale from particle_dynamics_utils

    displacements = np.zeros(len(traps_m))
    for ti, trap_pos in enumerate(traps_m):
        pos = trap_pos.copy()
        max_disp = 0.0
        for step in range(T_STEPS):
            d = np.linalg.norm(pos - trap_pos)
            if d > max_disp:
                max_disp = d
            pt = np.array([pos[1], pos[0]])  # (y, x)
            fx = float(interp_fx(pt.reshape(1, -1))[0])
            fy = float(interp_fy(pt.reshape(1, -1))[0])
            pos = pos + dt * SCALE * np.array([fx, fy])
            pos[0] = np.clip(pos[0], xg[0], xg[-1])
            pos[1] = np.clip(pos[1], yg[0], yg[-1])
        # Final displacement
        final_d = np.linalg.norm(pos - trap_pos)
        displacements[ti] = max(max_disp, final_d)

    disp_um = displacements * 1e6

    # ── Figure ────────────────────────────────────────────────────
    mm_to_in = 1.0 / 25.4
    fig = plt.figure(figsize=(160*mm_to_in, 120*mm_to_in))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1.0, 1.2], hspace=0.35)

    # ── Top panel: IASA convergence ───────────────────────────────
    ax_top = fig.add_subplot(gs[0])
    color_corr = "#2166ac"
    color_leak = "#d6604d"

    if use_full_corr:
        iters_corr = history_iter
        vals_corr = history_corr
    else:
        iters_corr = snap_iters
        vals_corr = snap_corr

    ln1 = ax_top.plot(iters_corr, vals_corr, "-", color=color_corr, lw=1.2,
                      label="Target correlation")
    ax_top.set_xlabel("Iteration", fontsize=9)
    ax_top.set_ylabel("Correlation coefficient", fontsize=9, color=color_corr)
    ax_top.tick_params(axis="y", labelcolor=color_corr, labelsize=8)
    ax_top.tick_params(axis="x", labelsize=8)
    ax_top.axhline(1.0, ls="--", color="gray", lw=0.6, alpha=0.5)
    ax_top.set_ylim(0, 1.1)

    # Annotate final correlation
    ax_top.annotate(
        f"{vals_corr[-1]:.3f}",
        xy=(iters_corr[-1], vals_corr[-1]),
        xytext=(iters_corr[-1] - 20, vals_corr[-1] - 0.1),
        fontsize=7, color=color_corr,
        arrowprops=dict(arrowstyle="->", color=color_corr, lw=0.8),
    )

    ax_top2 = ax_top.twinx()
    ln2 = ax_top2.plot(snap_iters, snap_leakage, "o--", color=color_leak,
                       lw=1.0, ms=4, label="Leakage fraction")
    ax_top2.set_ylabel("Leakage fraction", fontsize=9, color=color_leak)
    ax_top2.tick_params(axis="y", labelcolor=color_leak, labelsize=8)
    ax_top2.set_ylim(0, 1.0)

    # Annotate final leakage
    if len(snap_leakage) > 0:
        ax_top2.annotate(
            f"{snap_leakage[-1]:.3f}",
            xy=(snap_iters[-1], snap_leakage[-1]),
            xytext=(snap_iters[-1] - 25, snap_leakage[-1] + 0.08),
            fontsize=7, color=color_leak,
            arrowprops=dict(arrowstyle="->", color=color_leak, lw=0.8),
        )

    # Combined legend
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax_top.legend(lns, labs, fontsize=8, loc="center right")

    ax_top.set_title("IASA iteration convergence — C-shape target",
                     fontsize=9, pad=4)
    ax_top.spines["top"].set_visible(False)

    # ── Bottom panel: neighbour displacement map ──────────────────
    ax_bot = fig.add_subplot(gs[1])

    # Faint Gorkov background
    ext_mm = [xg[0]*1e3, xg[-1]*1e3, yg[0]*1e3, yg[-1]*1e3]
    ax_bot.imshow(
        U_sw, extent=ext_mm, origin="lower",
        cmap="viridis_r", alpha=0.3, aspect="equal",
        interpolation="bicubic",
    )

    # Scatter plot: circle at each trap, colour = displacement
    trap_x_mm = traps_m[:, 0] * 1e3
    trap_y_mm = traps_m[:, 1] * 1e3
    # Size proportional to displacement (min size 10, max 120)
    sizes = 10 + 110 * disp_um / max(disp_um.max(), 1e-6)

    sc = ax_bot.scatter(
        trap_x_mm, trap_y_mm,
        c=disp_um, s=sizes, cmap="YlOrRd", edgecolors="k", linewidths=0.3,
        zorder=5,
    )
    cb = fig.colorbar(sc, ax=ax_bot, fraction=0.046, pad=0.04)
    cb.set_label("displacement (µm)", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    # Mark A and B
    ax_bot.annotate("A", xy=(trap_x_mm[idx_A], trap_y_mm[idx_A]),
                    xytext=(5, 5), textcoords="offset points",
                    fontsize=9, fontweight="bold", color="#e74c3c")
    ax_bot.annotate("B", xy=(trap_x_mm[idx_B], trap_y_mm[idx_B]),
                    xytext=(5, 5), textcoords="offset points",
                    fontsize=9, fontweight="bold", color="#3498db")

    ax_bot.set_xlabel("x (mm)", fontsize=9)
    ax_bot.set_ylabel("y (mm)", fontsize=9)
    ax_bot.set_title("Neighbour displacement under C-shape field", fontsize=9, pad=4)
    _style_ax(ax_bot)

    # ── Save ──────────────────────────────────────────────────────
    fig.subplots_adjust(hspace=0.35)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = OUT_DIR / f"fig_cshape_failure.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved: {out.relative_to(PROJECT_ROOT)}")
    plt.close(fig)

    print(
        f"C-shape IASA: final correlation={vals_corr[-1]:.3f}, "
        f"final leakage={snap_leakage[-1] if len(snap_leakage) > 0 else 'N/A':.3f}, "
        f"max neighbour displacement={disp_um.max():.1f} µm, "
        f"mean neighbour displacement={disp_um.mean():.1f} µm"
    )


if __name__ == "__main__":
    main()
