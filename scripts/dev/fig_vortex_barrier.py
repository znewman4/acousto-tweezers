#!/usr/bin/env python3
"""
Publication figure: Vortex phase offset barrier suppression.

3 panels side by side:
  Panel 1: 1D Gor'kov profile along A→B at ψ=0  (barrier present)
  Panel 2: 1D Gor'kov profile at optimal ψ       (barrier suppressed)
  Panel 3: Minimum A→B approach distance vs ψ    (overdamped dynamics)
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

from scripts.lib.particle_dynamics_utils import gorkov_normalised, DT_DEFAULT, SCALE as PDU_SCALE

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════
OUT_DIR = PROJECT_ROOT / "results" / "figures"

OVERLAY_NPZ = (
    PROJECT_ROOT / "results" / "deliverables"
    / "overlay_local" / "overlay_local_fields.npz"
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
    ax.grid(True, alpha=0.25)


def _combine(p_sw, p_vortex, alpha, psi):
    return p_sw + alpha * np.exp(1j * psi) * p_vortex


def _gorkov_field(p, dx, dy):
    U, Fx, Fy = gorkov_normalised(p, dx, dy)
    return U, Fx, Fy


def _sample_line(field, xg, yg, A, B, n_pts=300):
    """Sample field along A→B line. Returns (distances_um, values)."""
    t = np.linspace(0, 1, n_pts)
    pts_m = A[None, :] + t[:, None] * (B - A)[None, :]
    interp = RegularGridInterpolator(
        (yg, xg), field, bounds_error=False, fill_value=np.nan,
    )
    vals = interp(pts_m[:, ::-1])  # (y, x)
    dist_um = np.linalg.norm(B - A) * t * 1e6
    return dist_um, vals, pts_m


def _barrier_height(dist, vals):
    """Return (saddle_val, min_val, δU, saddle_idx)."""
    valid = ~np.isnan(vals)
    v = vals[valid]
    d = dist[valid]
    i_min = np.argmin(v)
    # Saddle is the max between the two minima at ends
    i_saddle = np.argmax(v)
    return v[i_saddle], v[i_min], v[i_saddle] - v[i_min], i_saddle


def main():
    if not OVERLAY_NPZ.exists():
        print(f"ERROR: Overlay NPZ not found: {OVERLAY_NPZ}")
        print("  Run scripts/deliverables/overlay_local_study.py first.")
        sys.exit(1)

    # ── Load ──────────────────────────────────────────────────────
    d = np.load(OVERLAY_NPZ, allow_pickle=True)
    p_sw = d["p_sw"]       # (400, 400) complex
    p_vortex = d["p_vortex"]  # (400, 400) complex
    xg = d["xg"]           # (400,) m
    yg = d["yg"]           # (400,) m
    traps_m = d["traps_m"]  # (N, 2) m
    idx_A = int(d["idx_A"])
    idx_B = int(d["idx_B"])

    A = traps_m[idx_A]
    B = traps_m[idx_B]
    dx = float(xg[1] - xg[0])
    dy = float(yg[1] - yg[0])

    # Best alpha from overlay study
    best_alpha = 0.8  # from overlay_local_summary.json

    n_line = 300
    N_PSI_SWEEP = 32
    N_PSI_DYN = 64

    # ── Panel 1: ψ=0 profile ─────────────────────────────────────
    p_comb_0 = _combine(p_sw, p_vortex, best_alpha, 0.0)
    U_0, Fx_0, Fy_0 = _gorkov_field(p_comb_0, dx, dy)
    dist_um, U_line_0, _ = _sample_line(U_0, xg, yg, A, B, n_line)
    saddle_0, min_0, dU_0, i_saddle_0 = _barrier_height(dist_um, U_line_0)

    # ── Panel 2: sweep ψ to find optimal ─────────────────────────
    psi_vals = np.linspace(0, 2 * np.pi, N_PSI_SWEEP, endpoint=False)
    dU_sweep = np.zeros(N_PSI_SWEEP)
    for i, psi in enumerate(psi_vals):
        p_c = _combine(p_sw, p_vortex, best_alpha, psi)
        U_c, _, _ = _gorkov_field(p_c, dx, dy)
        _, vals, _ = _sample_line(U_c, xg, yg, A, B, n_line)
        valid = ~np.isnan(vals)
        if valid.any():
            dU_sweep[i] = np.nanmax(vals) - np.nanmin(vals)
        else:
            dU_sweep[i] = 0.0

    i_opt = np.argmin(dU_sweep)
    psi_opt = psi_vals[i_opt]
    p_comb_opt = _combine(p_sw, p_vortex, best_alpha, psi_opt)
    U_opt, _, _ = _gorkov_field(p_comb_opt, dx, dy)
    _, U_line_opt, _ = _sample_line(U_opt, xg, yg, A, B, n_line)
    _, _, dU_opt, i_saddle_opt = _barrier_height(dist_um, U_line_opt)
    reduction_pct = (1.0 - dU_opt / max(dU_0, 1e-30)) * 100.0

    # ── Panel 3: overdamped dynamics sweep ────────────────────────
    # For each ψ, simulate particle B under combined Gorkov for t steps
    psi_dyn = np.linspace(0, 2 * np.pi, N_PSI_DYN, endpoint=False)
    min_approach = np.zeros(N_PSI_DYN)
    T_SIM = 500
    dt = DT_DEFAULT
    SCALE = PDU_SCALE  # overdamped velocity scale from particle_dynamics_utils

    # Use gorkov_normalised-based force for overdamped stepping
    for i, psi in enumerate(psi_dyn):
        p_c = _combine(p_sw, p_vortex, best_alpha, psi)
        U_c, Fx_c, Fy_c = _gorkov_field(p_c, dx, dy)

        # Build interpolators for force field
        interp_fx = RegularGridInterpolator(
            (yg, xg), Fx_c, bounds_error=False, fill_value=0.0,
        )
        interp_fy = RegularGridInterpolator(
            (yg, xg), Fy_c, bounds_error=False, fill_value=0.0,
        )

        # Start particle at B's location
        pos = B.copy()
        dists = np.zeros(T_SIM)
        for step in range(T_SIM):
            dists[step] = np.linalg.norm(pos - A)
            pt = np.array([pos[1], pos[0]])  # (y, x)
            fx = float(interp_fx(pt.reshape(1, -1))[0])
            fy = float(interp_fy(pt.reshape(1, -1))[0])
            pos = pos + dt * SCALE * np.array([fx, fy])
            # Clamp to grid bounds
            pos[0] = np.clip(pos[0], xg[0], xg[-1])
            pos[1] = np.clip(pos[1], yg[0], yg[-1])

        min_approach[i] = np.min(dists) * 1e6  # µm

    # ── Figure ────────────────────────────────────────────────────
    mm_to_in = 1.0 / 25.4
    fig, axes = plt.subplots(1, 3, figsize=(180*mm_to_in, 60*mm_to_in))

    # Panel 1 — ψ=0
    ax1 = axes[0]
    ax1.plot(dist_um, U_line_0, "k-", lw=1.0)
    # Shade barrier region
    ax1.fill_between(dist_um, np.nanmin(U_line_0), U_line_0,
                     alpha=0.15, color="red")
    # Annotate ΔU
    x_saddle = dist_um[i_saddle_0]
    ax1.annotate(
        "", xy=(x_saddle, saddle_0), xytext=(x_saddle, min_0),
        arrowprops=dict(arrowstyle="<->", color="red", lw=1.2),
    )
    ax1.text(x_saddle + 15, (saddle_0 + min_0) / 2,
             f"ΔU = {dU_0:.2e}", fontsize=7, color="red", va="center")
    ax1.axvline(0, ls=":", color="gray", lw=0.6)
    ax1.axvline(dist_um[-1], ls=":", color="gray", lw=0.6)
    ax1.text(0, ax1.get_ylim()[1] if ax1.get_ylim()[1] != 0 else saddle_0,
             "A", fontsize=9, ha="center", va="bottom", fontweight="bold")
    ax1.text(dist_um[-1], ax1.get_ylim()[1] if ax1.get_ylim()[1] != 0 else saddle_0,
             "B", fontsize=9, ha="center", va="bottom", fontweight="bold")
    ax1.set_xlabel("distance along A→B (µm)", fontsize=9)
    ax1.set_ylabel("U (a.u.)", fontsize=9)
    ax1.set_title(r"$\psi = 0$ — barrier present", fontsize=9, pad=4)
    _style_ax(ax1)

    # Panel 2 — ψ_opt
    ax2 = axes[1]
    ax2.plot(dist_um, U_line_opt, "k-", lw=1.0)
    ax2.fill_between(dist_um, np.nanmin(U_line_opt), U_line_opt,
                     alpha=0.10, color="green")
    ax2.annotate(
        f"ψ_opt = {np.degrees(psi_opt):.0f}°, ΔU reduced by {reduction_pct:.0f}%",
        xy=(0.5, 0.95), xycoords="axes fraction",
        fontsize=7, ha="center", va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
    )
    ax2.axvline(0, ls=":", color="gray", lw=0.6)
    ax2.axvline(dist_um[-1], ls=":", color="gray", lw=0.6)
    ax2.text(0, np.nanmax(U_line_opt), "A", fontsize=9, ha="center",
             va="bottom", fontweight="bold")
    ax2.text(dist_um[-1], np.nanmax(U_line_opt), "B", fontsize=9,
             ha="center", va="bottom", fontweight="bold")
    ax2.set_xlabel("distance along A→B (µm)", fontsize=9)
    ax2.set_ylabel("U (a.u.)", fontsize=9)
    ax2.set_title(r"$\psi = \psi_{\mathrm{opt}}$ — barrier suppressed",
                  fontsize=9, pad=4)
    _style_ax(ax2)

    # Panel 3 — approach distance vs ψ
    ax3 = axes[2]
    ax3.plot(np.degrees(psi_dyn), min_approach, "k-", lw=1.0)
    ax3.axvline(np.degrees(psi_opt), ls="--", color="red", lw=0.8,
                label=r"$\psi_{\mathrm{opt}}$")
    # λ/2 horizontal
    lam_half_um = 0.742e-3 / 2 * 1e6  # ≈371  µm
    ax3.axhline(lam_half_um, ls="--", color="blue", lw=0.8,
                label=r"$\lambda/2$ = 371 µm")
    ax3.set_xlabel(r"$\psi$ (°)", fontsize=9)
    ax3.set_ylabel("min approach distance (µm)", fontsize=9)
    ax3.set_title("Particle B approach distance vs phase offset",
                  fontsize=9, pad=4)
    ax3.legend(fontsize=8, loc="upper right")
    ax3.set_xlim(0, 360)
    _style_ax(ax3)

    # ── Save ──────────────────────────────────────────────────────
    plt.tight_layout(pad=0.5)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = OUT_DIR / f"fig_vortex_barrier.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved: {out.relative_to(PROJECT_ROOT)}")
    plt.close(fig)

    print(
        f"Barrier: ψ=0 ΔU={dU_0:.3e}, ψ_opt={np.degrees(psi_opt):.1f}° "
        f"ΔU={dU_opt:.3e}, reduction={reduction_pct:.1f}%"
    )


if __name__ == "__main__":
    main()
