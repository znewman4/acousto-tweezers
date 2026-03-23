#!/usr/bin/env python3
"""
Post-process the side-by-side transport GIF to:
1) slow playback,
2) increase resolution, and
3) approximately synchronize A-particle motion timing between left/right columns.

The input GIF has C-shape on the left column and rectangular bridge on the right
column (for both ROI and full-domain rows). This script independently time-warps
the two columns, then recomposes each output frame.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "results"
    / "dev"
    / "transport_replica_cshape_vs_rectangular_bridge"
    / "transport_replica_cshape_vs_rectangular_bridge.gif"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "dev"
    / "transport_replica_cshape_vs_rectangular_bridge"
)


def _load_gif_frames(path: Path) -> tuple[list[np.ndarray], list[int]]:
    im = Image.open(path)
    frames: list[np.ndarray] = []
    durations_ms: list[int] = []

    while True:
        frames.append(np.asarray(im.convert("RGB"), dtype=np.uint8))
        durations_ms.append(int(im.info.get("duration", 70)))
        try:
            im.seek(im.tell() + 1)
        except EOFError:
            break

    if not frames:
        raise RuntimeError(f"No frames found in GIF: {path}")
    return frames, durations_ms


def _detect_red_centroid(frame_rgb: np.ndarray, prev_xy: np.ndarray | None) -> np.ndarray:
    """Detect red particle centroid in a column frame; fallback to previous point."""
    h, w, _ = frame_rgb.shape

    # Ignore top strip (suptitle area) to reduce text-induced jitter.
    y0 = int(0.08 * h)
    y1 = int(0.98 * h)
    crop = frame_rgb[y0:y1]

    r = crop[..., 0].astype(np.int16)
    g = crop[..., 1].astype(np.int16)
    b = crop[..., 2].astype(np.int16)

    red_mask = (
        (r > 150)
        & (g < 130)
        & (b < 130)
        & ((r - g) > 35)
        & ((r - b) > 35)
    )

    if int(np.count_nonzero(red_mask)) < 25:
        if prev_xy is not None:
            return prev_xy.copy()
        return np.array([0.5 * w, 0.5 * h], dtype=float)

    ys, xs = np.nonzero(red_mask)
    cx = float(np.mean(xs))
    cy = float(np.mean(ys) + y0)
    return np.array([cx, cy], dtype=float)


def _track_red_motion(column_frames: list[np.ndarray]) -> np.ndarray:
    pts = []
    prev = None
    for frm in column_frames:
        c = _detect_red_centroid(frm, prev)
        pts.append(c)
        prev = c
    return np.asarray(pts, dtype=float)


def _motion_window(track_xy: np.ndarray, min_disp_px: float = 3.0) -> tuple[int, int, np.ndarray]:
    """
    Return (onset_idx, near_completion_idx, displacement_curve).
    """
    disp = np.linalg.norm(track_xy - track_xy[0], axis=1)
    total = float(np.max(disp))

    n = len(disp)
    if n < 2:
        return 0, 0, disp

    if total < (1.5 * min_disp_px):
        return 0, n - 1, disp

    onset_thr = max(min_disp_px, 0.12 * total)
    finish_thr = max(onset_thr + 1e-9, 0.88 * total)

    i_on = int(np.where(disp >= onset_thr)[0][0])
    finish_hits = np.where(disp >= finish_thr)[0]
    i_off = int(finish_hits[0]) if finish_hits.size else n - 1
    i_off = max(i_off, min(n - 1, i_on + 1))
    return i_on, i_off, disp


def _map_piecewise(
    t_out: float,
    src_on: float,
    src_off: float,
    tgt_on: float,
    tgt_off: float,
) -> float:
    """Map output normalized time to source normalized time (piecewise-linear)."""
    if t_out <= tgt_on:
        if tgt_on <= 1e-12:
            return 0.0
        return (t_out / tgt_on) * src_on

    if t_out <= tgt_off:
        span = max(tgt_off - tgt_on, 1e-12)
        return src_on + ((t_out - tgt_on) / span) * (src_off - src_on)

    if tgt_off >= 1.0 - 1e-12:
        return 1.0
    return src_off + ((t_out - tgt_off) / (1.0 - tgt_off)) * (1.0 - src_off)


def _compose_synced_frames(
    frames: list[np.ndarray],
    slowdown_factor: float,
    upscale_factor: int,
) -> tuple[list[Image.Image], dict]:
    n_src = len(frames)
    if n_src < 2:
        raise RuntimeError("Need at least 2 source frames for time warp.")

    h, w, _ = frames[0].shape
    mid = w // 2

    left_frames = [f[:, :mid, :] for f in frames]
    right_frames = [f[:, mid:, :] for f in frames]

    track_left = _track_red_motion(left_frames)
    track_right = _track_red_motion(right_frames)

    l_on, l_off, d_left = _motion_window(track_left)
    r_on, r_off, d_right = _motion_window(track_right)

    denom = float(n_src - 1)
    src_left_on = l_on / denom
    src_left_off = l_off / denom
    src_right_on = r_on / denom
    src_right_off = r_off / denom

    tgt_on = 0.5 * (src_left_on + src_right_on)
    tgt_off = 0.5 * (src_left_off + src_right_off)
    if tgt_off <= tgt_on + 1e-6:
        tgt_off = min(1.0, tgt_on + 0.05)

    n_out = max(2, int(round((n_src - 1) * max(1.0, slowdown_factor))) + 1)
    out_images: list[Image.Image] = []

    left_index_map: list[int] = []
    right_index_map: list[int] = []

    for i in range(n_out):
        t = i / float(n_out - 1)

        u_left = np.clip(_map_piecewise(t, src_left_on, src_left_off, tgt_on, tgt_off), 0.0, 1.0)
        u_right = np.clip(_map_piecewise(t, src_right_on, src_right_off, tgt_on, tgt_off), 0.0, 1.0)

        idx_left = int(round(u_left * (n_src - 1)))
        idx_right = int(round(u_right * (n_src - 1)))

        left_index_map.append(idx_left)
        right_index_map.append(idx_right)

        src_l = frames[idx_left]
        src_r = frames[idx_right]

        composed = np.empty_like(src_l)
        composed[:, :mid, :] = src_l[:, :mid, :]
        composed[:, mid:, :] = src_r[:, mid:, :]

        img = Image.fromarray(composed, mode="RGB")
        if upscale_factor > 1:
            img = img.resize(
                (w * upscale_factor, h * upscale_factor),
                resample=Image.Resampling.LANCZOS,
            )
        out_images.append(img)

    meta = {
        "source_frames": int(n_src),
        "output_frames": int(n_out),
        "slowdown_factor": float(slowdown_factor),
        "upscale_factor": int(upscale_factor),
        "source_motion_window": {
            "left": {"onset_frame": int(l_on), "near_completion_frame": int(l_off)},
            "right": {"onset_frame": int(r_on), "near_completion_frame": int(r_off)},
        },
        "source_motion_total_disp_px": {
            "left": float(np.max(d_left)),
            "right": float(np.max(d_right)),
        },
        "target_sync_window_norm": {
            "onset": float(tgt_on),
            "near_completion": float(tgt_off),
        },
        "column_index_maps": {
            "left": left_index_map,
            "right": right_index_map,
        },
    }
    return out_images, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Slow, upscale, and time-sync the side-by-side bridge GIF.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--output-name",
        type=str,
        default="transport_replica_cshape_vs_rectangular_bridge_synced_slow_hd.gif",
    )
    parser.add_argument("--slowdown", type=float, default=2.0, help="Playback slowdown via extra output frames.")
    parser.add_argument("--upscale", type=int, default=2, help="Integer output resolution multiplier.")
    parser.add_argument(
        "--frame-duration-ms",
        type=int,
        default=70,
        help="Per-frame duration for output GIF (default keeps source frame pacing while frame count grows).",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    src_frames, src_durations = _load_gif_frames(args.input)
    out_frames, meta = _compose_synced_frames(
        frames=src_frames,
        slowdown_factor=float(args.slowdown),
        upscale_factor=int(max(1, args.upscale)),
    )

    out_gif = args.output_dir / args.output_name
    out_frames[0].save(
        out_gif,
        save_all=True,
        append_images=out_frames[1:],
        duration=max(10, int(args.frame_duration_ms)),
        loop=0,
        optimize=False,
        disposal=2,
    )

    meta.update(
        {
            "input_gif": str(args.input),
            "output_gif": str(out_gif),
            "source_avg_duration_ms": float(np.mean(src_durations)),
            "output_duration_ms_total": float(len(out_frames) * max(10, int(args.frame_duration_ms))),
        }
    )

    out_meta = args.output_dir / f"{out_gif.stem}_manifest.json"
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved enhanced GIF: {out_gif}")
    print(f"Saved manifest: {out_meta}")


if __name__ == "__main__":
    main()
