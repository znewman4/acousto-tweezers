#!/usr/bin/env python3
"""
Side-by-side geometry GIF: original C-shape pressure vs symmetric bridge geometry.

Purpose:
- Reuse the same GIF rendering pattern used in transport scripts (matplotlib frames + PIL),
- Compare static geometry fields with no SW background,
- Overlay particle geometry so relative placement is easy to inspect.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OVERLAY_NPZ = (
    PROJECT_ROOT
    / "results"
    / "c_shape_lens_15mm_overlay_study_20260310_170620"
    / "npz"
    / "roi_fields.npz"
)
BRIDGE_NPZ = (
    PROJECT_ROOT
    / "results"
    / "dev"
    / "bridge_pressure_field_standalone_scaled"
    / "bridge_pressure_fields.npz"
)

DEFAULT_OUT_BASE = PROJECT_ROOT / "results" / "dev"

PARTICLE_RADIUS_M = 50.0e-6
POCKET_SIGMA_M = 125.0e-6


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


def _robust_pos_limits(arrays: list[np.ndarray], lo: float = 1.0, hi: float = 99.5) -> tuple[float, float]:
    vals = np.concatenate([a[np.isfinite(a)].ravel() for a in arrays])
    if vals.size == 0:
        return 0.0, 1.0
    vmin = float(np.percentile(vals, lo))
    vmax = float(np.percentile(vals, hi))
    if vmax <= vmin:
        vmax = vmin + 1.0e-12
    return vmin, vmax


def _draw_particle_geometry(ax, traps_mm: np.ndarray, idx_a: int, idx_b: int) -> None:
    for i, t in enumerate(traps_mm):
        if i == idx_a:
            face = "#f4a6a6"
            edge = "#b93f3f"
        elif i == idx_b:
            face = "#a9d2ff"
            edge = "#2f6fa8"
        else:
            face = "#9fb3c8"
            edge = "#5f6f7f"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render side-by-side C-shape vs symmetric-bridge geometry GIF")
    parser.add_argument("--overlay-npz", type=Path, default=OVERLAY_NPZ)
    parser.add_argument("--bridge-npz", type=Path, default=BRIDGE_NPZ)
    parser.add_argument("--frames", type=int, default=72)
    parser.add_argument("--gif-duration-ms", type=int, default=60)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-name", type=str, default="cshape_vs_symmetric_bridge_geometry.gif")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or (DEFAULT_OUT_BASE / f"cshape_vs_symmetric_bridge_geometry_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    ov = np.load(args.overlay_npz, allow_pickle=False)
    bd = np.load(args.bridge_npz, allow_pickle=False)

    x_roi = ov["xg"].astype(float)
    y_roi = ov["yg"].astype(float)
    p_cshape_roi = ov["p_lens_roi"].astype(complex)

    traps_m = ov["traps_m"].astype(float)
    idx_a = int(ov["idx_A"])
    idx_b = int(ov["idx_B"])

    x_full = bd["x_full"].astype(float)
    y_full = bd["y_full"].astype(float)
    p_bridge_full = bd["p_bridge_effective_full"].astype(complex)

    ix_roi, iy_roi = _crop_indices(
        x_full,
        y_full,
        float(x_roi[0]),
        float(x_roi[-1]),
        float(y_roi[0]),
        float(y_roi[-1]),
    )
    p_bridge_roi = p_bridge_full[np.ix_(iy_roi, ix_roi)]

    x_roi_mm = x_roi * 1e3
    y_roi_mm = y_roi * 1e3
    extent_roi = [x_roi_mm[0], x_roi_mm[-1], y_roi_mm[0], y_roi_mm[-1]]
    traps_mm = traps_m * 1e3

    c_amp = np.abs(p_cshape_roi)
    b_amp = np.abs(p_bridge_roi)
    vmin, vmax = _robust_pos_limits([c_amp, b_amp], lo=1.0, hi=99.5)

    bx_m, by_m = float(traps_m[idx_b, 0]), float(traps_m[idx_b, 1])
    b_mirror_mm = np.array([-bx_m, by_m]) * 1e3

    n_frames = max(12, int(args.frames))
    frame_duration_ms = max(10, int(args.gif_duration_ms))

    frames: list[Image.Image] = []
    t0 = time.perf_counter()

    for fi in range(n_frames):
        pulse = 0.5 * (1.0 + np.sin(2.0 * np.pi * fi / float(n_frames)))
        pocket_r_mm = float((0.85 + 0.25 * pulse) * POCKET_SIGMA_M * 1e3)

        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11.8, 5.6))

        _draw_particle_geometry(ax_l, traps_mm, idx_a, idx_b)
        _draw_particle_geometry(ax_r, traps_mm, idx_a, idx_b)

        im_l = ax_l.imshow(
            c_amp,
            origin="lower",
            extent=extent_roi,
            cmap="inferno",
            vmin=vmin,
            vmax=vmax,
            interpolation="bicubic",
            alpha=0.76,
            zorder=2,
            aspect="equal",
        )
        ax_l.set_title("Original C-shape |p| (no SW)")

        im_r = ax_r.imshow(
            b_amp,
            origin="lower",
            extent=extent_roi,
            cmap="inferno",
            vmin=vmin,
            vmax=vmax,
            interpolation="bicubic",
            alpha=0.76,
            zorder=2,
            aspect="equal",
        )
        ax_r.set_title("New symmetric geometry |p| (no SW)")

        b_xy_mm = traps_mm[idx_b]
        circ_l = Circle(
            (float(b_xy_mm[0]), float(b_xy_mm[1])),
            radius=pocket_r_mm,
            facecolor="none",
            edgecolor="#00c2ff",
            linestyle="--",
            linewidth=1.7,
            alpha=0.85,
            zorder=4,
        )
        ax_l.add_patch(circ_l)

        circ_r1 = Circle(
            (float(b_xy_mm[0]), float(b_xy_mm[1])),
            radius=pocket_r_mm,
            facecolor="none",
            edgecolor="#00c2ff",
            linewidth=2.0,
            alpha=0.95,
            zorder=4,
        )
        circ_r2 = Circle(
            (float(b_mirror_mm[0]), float(b_mirror_mm[1])),
            radius=pocket_r_mm,
            facecolor="none",
            edgecolor="#00c2ff",
            linewidth=2.0,
            alpha=0.95,
            zorder=4,
        )
        ax_r.add_patch(circ_r1)
        ax_r.add_patch(circ_r2)

        for ax in (ax_l, ax_r):
            ax.set_xlabel("x [mm]")
            ax.set_ylabel("y [mm]")
            ax.set_xlim(extent_roi[0], extent_roi[1])
            ax.set_ylim(extent_roi[2], extent_roi[3])

        fig.suptitle(
            "Original C-shape vs new symmetric bridge geometry (particle geometry background, no SW)",
            fontsize=11,
            fontweight="bold",
        )

        cbar = fig.colorbar(im_r, ax=[ax_l, ax_r], fraction=0.025, pad=0.02)
        cbar.set_label("|p| [Pa]")

        fig.subplots_adjust(left=0.06, right=0.92, bottom=0.08, top=0.90, wspace=0.18)

        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        frames.append(Image.fromarray(buf.reshape(h, w, 4)).convert("RGB"))
        plt.close(fig)

    out_gif = out_dir / args.output_name
    frames[0].save(
        str(out_gif),
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration_ms,
        loop=0,
    )

    manifest = {
        "script": "scripts/dev/cshape_vs_symmetric_bridge_geometry_gif.py",
        "inputs": {
            "overlay_npz": str(args.overlay_npz),
            "bridge_npz": str(args.bridge_npz),
        },
        "controls": {
            "frames": int(n_frames),
            "gif_duration_ms": int(frame_duration_ms),
            "particle_radius_um": float(PARTICLE_RADIUS_M * 1e6),
            "pocket_sigma_um": float(POCKET_SIGMA_M * 1e6),
        },
        "outputs": {
            "gif": str(out_gif),
        },
        "timings": {
            "render_s": float(time.perf_counter() - t0),
        },
    }

    manifest_path = out_dir / "cshape_vs_symmetric_bridge_geometry_gif_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved GIF: {out_gif}")
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
