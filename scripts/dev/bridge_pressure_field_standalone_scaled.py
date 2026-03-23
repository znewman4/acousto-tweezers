#!/usr/bin/env python3
"""
Standalone bridge-pressure visualisation with robust scaling.

Goal:
- run only the rectangular bridge pressure field (no standing-wave background),
- show both template and effective alpha-scaled bridge fields,
- use robust percentile scaling so the corridor structure is visible
  (avoids the "one big red blob" saturation look).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import sys
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dev.rectangular_corridor_bridge import build_corridor_bridge_field


# ---------------------------------------------------------------------------
# Inputs / outputs
# ---------------------------------------------------------------------------
VORTEX_NPZ = (
    PROJECT_ROOT / "results" / "deliverables" / "vortex_stage_transport"
    / "transport" / "transport_case_for_gif.npz"
)
OVERLAY_NPZ = (
    PROJECT_ROOT / "results" / "c_shape_lens_15mm_overlay_study_20260310_170620"
    / "npz" / "roi_fields.npz"
)
TUNING_JSON = (
    PROJECT_ROOT / "results" / "dev" / "transport_replica_cshape_vs_rectangular_bridge"
    / "rectangular_bridge_alpha_tuning.json"
)

OUT_DIR = PROJECT_ROOT / "results" / "dev" / "bridge_pressure_field_standalone_scaled"


# ---------------------------------------------------------------------------
# Bridge defaults (match side-by-side transport script defaults)
# ---------------------------------------------------------------------------
BRIDGE_PSI = 0.0

CORRIDOR_WIDTH_M = 3.0e-4
CORRIDOR_PAD_A_M = 8.0e-5
CORRIDOR_PAD_B_M = 1.0e-4
CORRIDOR_EDGE_S_M = 3.5e-5
CORRIDOR_EDGE_N_M = 3.0e-5

SOURCE_HOTSPOT_PA = 120.0
DESTINATION_POCKET_PA = 0.0
DESTINATION_REST_TRAP_PA = 0.0
SOURCE_SIGMA_M = 1.0e-4
DESTINATION_SIGMA_M = 1.0e-4
DESTINATION_REST_SIGMA_M = 8.0e-5
SOURCE_ABOVE_OFFSET_M = 9.0e-5

CORRIDOR_START_PA = 120.0
CORRIDOR_END_PA = 20.0
CORRIDOR_DECAY_POWER = 1.25
CORRIDOR_TRANSVERSE_SIGMA_RATIO = 0.32
B_QUIET_RADIUS_M = 6.0e-5

# Requested idealised-field adjustments
SYMMETRISE_ABOUT_Y_AXIS = True
EXTRA_B_POCKET_PA = 20.0
EXTRA_B_POCKET_SIGMA_M = 1.25e-4

# Particle-geometry overlay settings
PARTICLE_RADIUS_M = 50.0e-6


def _crop_indices(
    x_full: np.ndarray,
    y_full: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    ix = np.where((x_full >= x_min) & (x_full <= x_max))[0]
    iy = np.where((y_full >= y_min) & (y_full <= y_max))[0]
    if ix.size < 2 or iy.size < 2:
        raise RuntimeError("ROI crop indices are empty; check ROI/full-grid overlap")
    return ix, iy


def _robust_sym_limit(arrays: list[np.ndarray], pct: float = 99.5) -> float:
    vals = np.concatenate([np.abs(a[np.isfinite(a)]).ravel() for a in arrays])
    if vals.size == 0:
        return 1.0
    lim = float(np.percentile(vals, pct))
    return max(lim, 1.0e-12)


def _robust_pos_limits(arrays: list[np.ndarray], lo: float = 1.0, hi: float = 99.5) -> tuple[float, float]:
    vals = np.concatenate([a[np.isfinite(a)].ravel() for a in arrays])
    if vals.size == 0:
        return 0.0, 1.0
    vmin = float(np.percentile(vals, lo))
    vmax = float(np.percentile(vals, hi))
    if vmax <= vmin:
        vmax = vmin + 1.0e-12
    return vmin, vmax


def _nearest_index(axis: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(axis - value)))


def _gaussian_2d(
    xx: np.ndarray,
    yy: np.ndarray,
    x0: float,
    y0: float,
    sigma_m: float,
    amplitude: float,
) -> np.ndarray:
    rr2 = (xx - x0) ** 2 + (yy - y0) ** 2
    s2 = max(float(sigma_m), 1.0e-12) ** 2
    return float(amplitude) * np.exp(-0.5 * rr2 / s2)


def _add_symmetric_b_pocket(
    p_field: np.ndarray,
    x_full: np.ndarray,
    y_full: np.ndarray,
    b_xy: np.ndarray,
    amplitude_pa: float,
    sigma_m: float,
) -> np.ndarray:
    if amplitude_pa == 0.0:
        return p_field
    yy, xx = np.meshgrid(y_full, x_full, indexing="ij")
    bx = float(b_xy[0])
    by = float(b_xy[1])
    pocket = (
        _gaussian_2d(xx, yy, bx, by, sigma_m, amplitude_pa)
        + _gaussian_2d(xx, yy, -bx, by, sigma_m, amplitude_pa)
    )
    return p_field + pocket.astype(complex)


def _symmetrise_about_y_axis(p_field: np.ndarray) -> np.ndarray:
    return 0.5 * (p_field + np.fliplr(p_field))


def _save_pressure_only_and_geometry_overlay(
    p_roi: np.ndarray,
    x_roi_mm: np.ndarray,
    y_roi_mm: np.ndarray,
    traps_mm: np.ndarray,
    idx_a: int,
    idx_b: int,
    out_plain_png: Path,
    out_geom_png: Path,
) -> None:
    extent_roi = [x_roi_mm[0], x_roi_mm[-1], y_roi_mm[0], y_roi_mm[-1]]
    p_amp = np.abs(p_roi)
    vmin, vmax = _robust_pos_limits([p_amp], lo=1.0, hi=99.5)

    fig, ax = plt.subplots(figsize=(8.0, 7.0))
    im = ax.imshow(
        p_amp,
        origin="lower",
        extent=extent_roi,
        cmap="inferno",
        vmin=vmin,
        vmax=vmax,
        interpolation="bicubic",
        aspect="equal",
    )
    ax.set_title("Bridge-only pressure amplitude |p| (ROI)")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    plt.colorbar(im, ax=ax, label="|p| [Pa]")
    fig.tight_layout()
    fig.savefig(out_plain_png, dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 7.0))
    for i, t in enumerate(traps_mm):
        face = "#9fb3c8" if i not in (idx_a, idx_b) else ("#f4a6a6" if i == idx_a else "#a9d2ff")
        edge = "#5f6f7f" if i not in (idx_a, idx_b) else ("#b93f3f" if i == idx_a else "#2f6fa8")
        circ = Circle(
            (float(t[0]), float(t[1])),
            radius=float(PARTICLE_RADIUS_M * 1e3),
            facecolor=face,
            edgecolor=edge,
            linewidth=1.0,
            alpha=0.85,
            zorder=1,
        )
        ax.add_patch(circ)

    im = ax.imshow(
        p_amp,
        origin="lower",
        extent=extent_roi,
        cmap="inferno",
        vmin=vmin,
        vmax=vmax,
        interpolation="bicubic",
        aspect="equal",
        alpha=0.74,
        zorder=2,
    )
    ax.set_title("Bridge-only pressure |p| over particle geometry (ROI, no SW)")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_xlim(extent_roi[0], extent_roi[1])
    ax.set_ylim(extent_roi[2], extent_roi[3])
    plt.colorbar(im, ax=ax, label="|p| [Pa]")
    fig.tight_layout()
    fig.savefig(out_geom_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _load_selected_alpha() -> float:
    if TUNING_JSON.exists():
        try:
            with open(TUNING_JSON, "r", encoding="utf-8") as f:
                d = json.load(f)
            return float(d.get("selected_bridge_alpha", 64.26841))
        except Exception:
            pass
    return 64.26841


def _draw_pressure_row(
    axes_row,
    p_field: np.ndarray,
    extent: list[float],
    traps_mm: np.ndarray,
    idx_a: int,
    idx_b: int,
    title_prefix: str,
    lim_sym: float,
    amp_vmin: float,
    amp_vmax: float,
    log_vmin: float,
    log_vmax: float,
) -> None:
    p_re = np.real(p_field)
    p_im = np.imag(p_field)
    p_amp = np.abs(p_field)
    p_log = np.log10(np.maximum(p_amp, 1.0e-12))

    panels = [
        (p_re, "RdBu_r", -lim_sym, lim_sym, "Re(p) [Pa]"),
        (p_im, "RdBu_r", -lim_sym, lim_sym, "Im(p) [Pa]"),
        (p_amp, "inferno", amp_vmin, amp_vmax, "|p| [Pa]"),
        (p_log, "viridis", log_vmin, log_vmax, "log10|p| [Pa]"),
    ]

    for ax, (data, cmap, vmin, vmax, cblabel) in zip(axes_row, panels):
        im = ax.imshow(
            data,
            origin="lower",
            extent=extent,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="bicubic",
            aspect="equal",
        )
        ax.scatter(traps_mm[:, 0], traps_mm[:, 1], marker="x", c="#2ecc71", s=24, linewidths=0.7, zorder=4)
        ax.scatter([traps_mm[idx_b, 0]], [traps_mm[idx_b, 1]], c="#3498db", s=40, zorder=6)
        ax.scatter([traps_mm[idx_a, 0]], [traps_mm[idx_a, 1]], c="#e74c3c", s=40, zorder=6)
        ax.set_title(f"{title_prefix}: {cblabel}", fontsize=9)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_aspect("equal", adjustable="box")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)


def _save_pressure_figure(
    p_full: np.ndarray,
    p_roi: np.ndarray,
    x_full_mm: np.ndarray,
    y_full_mm: np.ndarray,
    x_roi_mm: np.ndarray,
    y_roi_mm: np.ndarray,
    traps_mm: np.ndarray,
    idx_a: int,
    idx_b: int,
    suptitle: str,
    out_png: Path,
) -> dict[str, float]:
    lim_sym = _robust_sym_limit([np.real(p_full), np.imag(p_full), np.real(p_roi), np.imag(p_roi)], pct=99.5)
    amp_vmin, amp_vmax = _robust_pos_limits([np.abs(p_full), np.abs(p_roi)], lo=1.0, hi=99.5)

    log_full = np.log10(np.maximum(np.abs(p_full), 1.0e-12))
    log_roi = np.log10(np.maximum(np.abs(p_roi), 1.0e-12))
    log_vmin, log_vmax = _robust_pos_limits([log_full, log_roi], lo=1.0, hi=99.5)

    extent_full = [x_full_mm[0], x_full_mm[-1], y_full_mm[0], y_full_mm[-1]]
    extent_roi = [x_roi_mm[0], x_roi_mm[-1], y_roi_mm[0], y_roi_mm[-1]]

    fig, axes = plt.subplots(2, 4, figsize=(18.5, 9.6))

    _draw_pressure_row(
        axes[0],
        p_full,
        extent_full,
        traps_mm,
        idx_a,
        idx_b,
        "Full domain",
        lim_sym,
        amp_vmin,
        amp_vmax,
        log_vmin,
        log_vmax,
    )
    _draw_pressure_row(
        axes[1],
        p_roi,
        extent_roi,
        traps_mm,
        idx_a,
        idx_b,
        "ROI",
        lim_sym,
        amp_vmin,
        amp_vmax,
        log_vmin,
        log_vmax,
    )

    fig.suptitle(suptitle, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)

    return {
        "lim_sym_pa": float(lim_sym),
        "amp_vmin_pa": float(amp_vmin),
        "amp_vmax_pa": float(amp_vmax),
        "log10_amp_vmin": float(log_vmin),
        "log10_amp_vmax": float(log_vmax),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ov = np.load(OVERLAY_NPZ)
    vd = np.load(VORTEX_NPZ)

    x_full = vd["xg"].astype(float)
    y_full = vd["yg"].astype(float)
    p_sw_full = vd["p_sw"].astype(complex)

    x_roi = ov["xg"].astype(float)
    y_roi = ov["yg"].astype(float)
    traps_m = ov["traps_m"].astype(float)
    idx_a = int(ov["idx_A"])
    idx_b = int(ov["idx_B"])

    neigh_idx = np.array(sorted(set(range(len(traps_m))) - {idx_a, idx_b}), dtype=int)

    a_xy = traps_m[idx_a]
    b_xy = traps_m[idx_b]

    p_bridge_template_full, p_bridge_b_anchor_full, frame, _, _, _ = build_corridor_bridge_field(
        x=x_full,
        y=y_full,
        point_a=a_xy,
        point_b=b_xy,
        width_m=CORRIDOR_WIDTH_M,
        pad_a_m=CORRIDOR_PAD_A_M,
        pad_b_m=CORRIDOR_PAD_B_M,
        edge_s_m=CORRIDOR_EDGE_S_M,
        edge_n_m=CORRIDOR_EDGE_N_M,
        source_hotspot_pa=SOURCE_HOTSPOT_PA,
        destination_pocket_pa=DESTINATION_POCKET_PA,
        source_sigma_m=SOURCE_SIGMA_M,
        destination_sigma_m=DESTINATION_SIGMA_M,
        destination_rest_trap_pa=DESTINATION_REST_TRAP_PA,
        destination_rest_sigma_m=DESTINATION_REST_SIGMA_M,
        source_above_offset_m=SOURCE_ABOVE_OFFSET_M,
        corridor_start_pa=CORRIDOR_START_PA,
        corridor_end_pa=CORRIDOR_END_PA,
        corridor_decay_power=CORRIDOR_DECAY_POWER,
        corridor_transverse_sigma_ratio=CORRIDOR_TRANSVERSE_SIGMA_RATIO,
        b_quiet_radius_m=B_QUIET_RADIUS_M,
        neighbour_positions_m=traps_m[neigh_idx],
    )

    # Match anchor phase convention used in transport script.
    ix_b = _nearest_index(x_full, float(b_xy[0]))
    iy_b = _nearest_index(y_full, float(b_xy[1]))
    phase_b_sw = float(np.angle(p_sw_full[iy_b, ix_b]))
    p_bridge_b_anchor_full = p_bridge_b_anchor_full * np.exp(1j * phase_b_sw)

    selected_alpha = _load_selected_alpha()
    p_bridge_scaled_only_full = selected_alpha * np.exp(1j * BRIDGE_PSI) * p_bridge_template_full

    # Add a slightly larger destination pocket around B, mirrored about y-axis.
    p_bridge_scaled_only_full = _add_symmetric_b_pocket(
        p_bridge_scaled_only_full,
        x_full=x_full,
        y_full=y_full,
        b_xy=b_xy,
        amplitude_pa=float(EXTRA_B_POCKET_PA),
        sigma_m=float(EXTRA_B_POCKET_SIGMA_M),
    )

    if SYMMETRISE_ABOUT_Y_AXIS:
        p_bridge_scaled_only_full = _symmetrise_about_y_axis(p_bridge_scaled_only_full)
        p_bridge_b_anchor_full = _symmetrise_about_y_axis(p_bridge_b_anchor_full)

    p_bridge_effective_full = p_bridge_scaled_only_full + p_bridge_b_anchor_full

    ix_roi, iy_roi = _crop_indices(
        x_full,
        y_full,
        float(x_roi[0]),
        float(x_roi[-1]),
        float(y_roi[0]),
        float(y_roi[-1]),
    )

    x_roi_from_full = x_full[ix_roi]
    y_roi_from_full = y_full[iy_roi]

    p_template_roi = p_bridge_template_full[np.ix_(iy_roi, ix_roi)]
    p_effective_roi = p_bridge_effective_full[np.ix_(iy_roi, ix_roi)]

    x_full_mm = x_full * 1e3
    y_full_mm = y_full * 1e3
    x_roi_mm = x_roi_from_full * 1e3
    y_roi_mm = y_roi_from_full * 1e3
    traps_mm = traps_m * 1e3

    fig_template = OUT_DIR / "bridge_pressure_template_only_scaled.png"
    scales_template = _save_pressure_figure(
        p_full=p_bridge_template_full,
        p_roi=p_template_roi,
        x_full_mm=x_full_mm,
        y_full_mm=y_full_mm,
        x_roi_mm=x_roi_mm,
        y_roi_mm=y_roi_mm,
        traps_mm=traps_mm,
        idx_a=idx_a,
        idx_b=idx_b,
        suptitle=(
            "Rectangular Bridge Pressure Template (no SW) — robust scaling\n"
            "(Shows structure without saturation)"
        ),
        out_png=fig_template,
    )

    fig_effective = OUT_DIR / "bridge_pressure_effective_alpha_scaled.png"
    scales_effective = _save_pressure_figure(
        p_full=p_bridge_effective_full,
        p_roi=p_effective_roi,
        x_full_mm=x_full_mm,
        y_full_mm=y_full_mm,
        x_roi_mm=x_roi_mm,
        y_roi_mm=y_roi_mm,
        traps_mm=traps_mm,
        idx_a=idx_a,
        idx_b=idx_b,
        suptitle=(
            f"Effective Bridge-only Pressure (alpha={selected_alpha:.3f}, psi={BRIDGE_PSI:.3f}) — robust scaling\n"
            "(alpha·template + destination anchor, still no SW baseline)"
        ),
        out_png=fig_effective,
    )

    pressure_only_png = OUT_DIR / "bridge_pressure_effective_roi_pressure_only.png"
    pressure_geom_png = OUT_DIR / "bridge_pressure_effective_roi_with_particle_geometry.png"
    _save_pressure_only_and_geometry_overlay(
        p_roi=p_effective_roi,
        x_roi_mm=x_roi_mm,
        y_roi_mm=y_roi_mm,
        traps_mm=traps_mm,
        idx_a=idx_a,
        idx_b=idx_b,
        out_plain_png=pressure_only_png,
        out_geom_png=pressure_geom_png,
    )

    npz_path = OUT_DIR / "bridge_pressure_fields.npz"
    np.savez_compressed(
        npz_path,
        x_full=x_full,
        y_full=y_full,
        x_roi=x_roi_from_full,
        y_roi=y_roi_from_full,
        traps_m=traps_m,
        idx_a=idx_a,
        idx_b=idx_b,
        p_bridge_template_full=p_bridge_template_full,
        p_bridge_anchor_full=p_bridge_b_anchor_full,
        p_bridge_scaled_only_full=p_bridge_scaled_only_full,
        p_bridge_effective_full=p_bridge_effective_full,
    )

    manifest = {
        "script": "scripts/dev/bridge_pressure_field_standalone_scaled.py",
        "purpose": "Standalone rectangular bridge pressure visualisation with robust scaling",
        "inputs": {
            "vortex_npz": str(VORTEX_NPZ.relative_to(PROJECT_ROOT)),
            "overlay_npz": str(OVERLAY_NPZ.relative_to(PROJECT_ROOT)),
            "tuning_json": str(TUNING_JSON.relative_to(PROJECT_ROOT)) if TUNING_JSON.exists() else None,
        },
        "pair": {
            "idx_a": int(idx_a),
            "idx_b": int(idx_b),
            "A_mm": [float(a_xy[0] * 1e3), float(a_xy[1] * 1e3)],
            "B_mm": [float(b_xy[0] * 1e3), float(b_xy[1] * 1e3)],
            "ab_um": float(frame.d_ab * 1e6),
        },
        "bridge_controls": {
            "selected_alpha": float(selected_alpha),
            "bridge_psi": float(BRIDGE_PSI),
            "symmetrise_about_y_axis": bool(SYMMETRISE_ABOUT_Y_AXIS),
            "extra_b_pocket_pa": float(EXTRA_B_POCKET_PA),
            "extra_b_pocket_sigma_um": float(EXTRA_B_POCKET_SIGMA_M * 1e6),
            "corridor_width_um": float(CORRIDOR_WIDTH_M * 1e6),
            "corridor_start_pa": float(CORRIDOR_START_PA),
            "corridor_end_pa": float(CORRIDOR_END_PA),
            "source_hotspot_pa": float(SOURCE_HOTSPOT_PA),
            "destination_pocket_pa": float(DESTINATION_POCKET_PA),
            "destination_rest_trap_pa": float(DESTINATION_REST_TRAP_PA),
        },
        "field_peaks_pa": {
            "template_abs_max": float(np.max(np.abs(p_bridge_template_full))),
            "scaled_only_abs_max": float(np.max(np.abs(p_bridge_scaled_only_full))),
            "effective_abs_max": float(np.max(np.abs(p_bridge_effective_full))),
        },
        "scaling_used": {
            "template": scales_template,
            "effective": scales_effective,
            "notes": "Robust percentiles (not raw min/max) to reveal corridor structure and prevent blob saturation",
        },
        "outputs": {
            "template_png": str(fig_template.relative_to(PROJECT_ROOT)),
            "effective_png": str(fig_effective.relative_to(PROJECT_ROOT)),
            "effective_roi_pressure_only_png": str(pressure_only_png.relative_to(PROJECT_ROOT)),
            "effective_roi_with_particle_geometry_png": str(pressure_geom_png.relative_to(PROJECT_ROOT)),
            "fields_npz": str(npz_path.relative_to(PROJECT_ROOT)),
        },
    }

    manifest_path = OUT_DIR / "bridge_pressure_field_standalone_scaled_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved: {fig_template}")
    print(f"Saved: {fig_effective}")
    print(f"Saved: {pressure_only_png}")
    print(f"Saved: {pressure_geom_png}")
    print(f"Saved: {npz_path}")
    print(f"Saved: {manifest_path}")


if __name__ == "__main__":
    main()
