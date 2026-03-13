#!/usr/bin/env python3
"""
Part 7 - GIF outputs for rebuilt staged translated-vortex transport study.

Outputs:
  results/deliverables/vortex_stage_transport/moving_vortex_transport.gif
  results/deliverables/vortex_stage_transport/moving_vortex_release_success_or_failure.gif
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lib.vortex_limit_utils import ensure_dir
from scripts.lib.vortex_limit_utils import compute_gorkov_fields, descend_potential
from scripts.lib.vortex_stage_transport_utils import (
    PHASE_SW_RELEASE,
    mm,
    phase_label,
    shift_complex_field,
)

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover
    imageio = None

ROOT_OUT = ensure_dir(PROJECT_ROOT / "results" / "deliverables" / "vortex_stage_transport")

ROI_HALF = 1.20e-3
FRAME_DT = 0.045
PUSH_INTERP = 6
RELEASE_INTERP = 2
SMOOTH_WINDOW = 7
RENDER_PRESETS = {
    "standard": {
        "frame_dt": 0.045,
        "push_interp": 6,
        "release_interp": 2,
        "smooth_window": 7,
    },
    "dense": {
        "frame_dt": 0.032,
        "push_interp": 12,
        "release_interp": 5,
        "smooth_window": 9,
    },
}
SHOW_ALL_TRAPS = True
SHOW_NEIGHBOUR_CURRENT = True
SHOW_NEIGHBOUR_TRAILS = True
SHOW_ALL_PARTICLE_CURRENT = True
SHOW_ALL_PARTICLE_DISPLACEMENTS = True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render translated-vortex transport GIFs.")
    parser.add_argument(
        "--preset",
        choices=sorted(RENDER_PRESETS),
        default="standard",
        help="Named render preset controlling frame density and smoothing.",
    )
    parser.add_argument("--frame-dt", type=float, default=None, help="GIF frame duration in seconds.")
    parser.add_argument("--push-interp", type=int, default=None, help="Interpolated subframes per transport step.")
    parser.add_argument("--release-interp", type=int, default=None, help="Interpolated subframes per release step.")
    parser.add_argument("--smooth-window", type=int, default=None, help="Odd smoothing window for displayed paths.")
    return parser.parse_args()


def _apply_render_config(args: argparse.Namespace) -> dict[str, float | int | str]:
    cfg = dict(RENDER_PRESETS[str(args.preset)])
    if args.frame_dt is not None:
        cfg["frame_dt"] = float(args.frame_dt)
    if args.push_interp is not None:
        cfg["push_interp"] = max(1, int(args.push_interp))
    if args.release_interp is not None:
        cfg["release_interp"] = max(1, int(args.release_interp))
    if args.smooth_window is not None:
        cfg["smooth_window"] = max(1, int(args.smooth_window))

    global FRAME_DT, PUSH_INTERP, RELEASE_INTERP, SMOOTH_WINDOW
    FRAME_DT = float(cfg["frame_dt"])
    PUSH_INTERP = int(cfg["push_interp"])
    RELEASE_INTERP = int(cfg["release_interp"])
    SMOOTH_WINDOW = int(cfg["smooth_window"])
    return {
        "preset": str(args.preset),
        "frame_dt": FRAME_DT,
        "push_interp": PUSH_INTERP,
        "release_interp": RELEASE_INTERP,
        "smooth_window": SMOOTH_WINDOW,
    }


def _write_render_summary(config: dict[str, float | int | str], transport_frames: int, release_frames: int) -> None:
    summary = {
        "renderer": "vortex_stage_transport_gifs.py",
        "transport_gif": "results/deliverables/vortex_stage_transport/moving_vortex_transport.gif",
        "release_gif": "results/deliverables/vortex_stage_transport/moving_vortex_release_success_or_failure.gif",
        "config": config,
        "frame_counts": {
            "transport": int(transport_frames),
            "release": int(release_frames),
        },
    }
    (ROOT_OUT / "gif_render_summary.json").write_text(json.dumps(summary, indent=2))


def _frame_array(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return np.array(rgba[..., :3], copy=True)


def _save_gif(frames: list[np.ndarray], path: Path, duration: float) -> None:
    if Image is not None:
        pil = [Image.fromarray(f) for f in frames]
        pil[0].save(
            path,
            save_all=True,
            append_images=pil[1:],
            duration=int(duration * 1000),
            loop=0,
            optimize=False,
            disposal=2,
        )
        return
    if imageio is not None:
        imageio.mimsave(path, frames, duration=duration)
        return
    raise RuntimeError("Neither PIL nor imageio is available for GIF writing")


def _combined_pressure(
    p_sw: np.ndarray,
    p_v: np.ndarray,
    xg: np.ndarray,
    yg: np.ndarray,
    alpha: float,
    psi: float,
    centre_xy: np.ndarray,
) -> np.ndarray:
    if float(alpha) <= 1e-14:
        return p_sw

    source_c = np.array([0.5 * (xg[0] + xg[-1]), 0.5 * (yg[0] + yg[-1])], dtype=float)
    p_v_shift = shift_complex_field(p_v, xg, yg, centre_xy, source_centre=source_c)
    return p_sw + float(alpha) * np.exp(1j * float(psi)) * p_v_shift


def _field_components(
    p_sw: np.ndarray,
    p_v: np.ndarray,
    xg: np.ndarray,
    yg: np.ndarray,
    alpha: float,
    psi: float,
    centre_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    p_total = _combined_pressure(p_sw, p_v, xg, yg, alpha, psi, centre_xy)

    p_delta = np.abs(p_total - p_sw)
    return np.abs(p_total), p_delta


def _equilibrium_cloud(
    p_sw: np.ndarray,
    p_v: np.ndarray,
    xg: np.ndarray,
    yg: np.ndarray,
    xy_prev: np.ndarray,
    alpha: float,
    psi: float,
    centre_xy: np.ndarray,
) -> np.ndarray:
    p_total = _combined_pressure(p_sw, p_v, xg, yg, alpha, psi, centre_xy)
    dx = float(xg[1] - xg[0])
    dy = float(yg[1] - yg[0])
    U, _, _ = compute_gorkov_fields(p_total, dx, dy)

    prev = np.asarray(xy_prev, dtype=float)
    out = np.zeros_like(prev)
    for j in range(len(prev)):
        out[j] = np.asarray(descend_potential(U, xg, yg, prev[j])["xy"], dtype=float)
    return out


def _compute_all_schedule_states(
    p_sw: np.ndarray,
    p_v: np.ndarray,
    xg: np.ndarray,
    yg: np.ndarray,
    alpha_schedule: np.ndarray,
    psi: float,
    centres_xy: np.ndarray,
    traps_m: np.ndarray,
) -> np.ndarray:
    n_steps = int(len(alpha_schedule))
    states = np.zeros((n_steps, len(traps_m), 2), dtype=float)

    prev = np.asarray(traps_m, dtype=float).copy()
    for k in range(n_steps):
        prev = _equilibrium_cloud(
            p_sw,
            p_v,
            xg,
            yg,
            xy_prev=prev,
            alpha=float(alpha_schedule[k]),
            psi=float(psi),
            centre_xy=np.asarray(centres_xy[k], dtype=float),
        )
        states[k] = prev

    return states


def _phase_title(code: int) -> str:
    mapping = {
        1: "1) standing-wave-only start",
        2: "2) vortex ramp-on",
        3: "3) translated moving vortex centre",
        4: "4) vortex ramp-off",
        5: "5) standing-wave-only release",
    }
    return mapping.get(int(code), phase_label(int(code)))


def _smooth_1d(arr: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(arr, dtype=float)
    if len(x) < 3:
        return x.copy()

    w = int(max(1, window))
    if w % 2 == 0:
        w += 1
    if w <= 1:
        return x.copy()

    pad = w // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(w, dtype=float) / float(w)
    return np.convolve(xp, kernel, mode="valid")


def _smooth_path(path_xy: np.ndarray, window: int = SMOOTH_WINDOW) -> np.ndarray:
    p = np.asarray(path_xy, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 3:
        return p.copy()

    out = np.empty_like(p)
    out[:, 0] = _smooth_1d(p[:, 0], window)
    out[:, 1] = _smooth_1d(p[:, 1], window)

    out[0] = p[0]
    out[-1] = p[-1]
    return out


def _smooth_cloud_path(path_xy: np.ndarray, window: int = SMOOTH_WINDOW) -> np.ndarray:
    p = np.asarray(path_xy, dtype=float)
    if p.ndim != 3 or p.shape[2] != 2:
        return p.copy()

    out = np.empty_like(p)
    for j in range(p.shape[1]):
        out[:, j, :] = _smooth_path(p[:, j, :], window=window)
    return out


def _compute_display_limits(
    p_sw: np.ndarray,
    p_v: np.ndarray,
    xg: np.ndarray,
    yg: np.ndarray,
    alpha_schedule: np.ndarray,
    psi: float,
    centres: np.ndarray,
    sample_count: int = 18,
) -> tuple[float, float, float]:
    n = int(len(alpha_schedule))
    if n == 0:
        return 0.0, 1.0, 1.0

    if n <= sample_count:
        idx = np.arange(n, dtype=int)
    else:
        idx = np.unique(np.linspace(0, n - 1, sample_count).astype(int))

    abs_vmin = float("inf")
    abs_vmax = 0.0
    delta_vmax = 0.0

    for k in idx:
        bg_total, bg_delta = _field_components(
            p_sw,
            p_v,
            xg,
            yg,
            alpha=float(alpha_schedule[k]),
            psi=float(psi),
            centre_xy=np.asarray(centres[k], dtype=float),
        )
        abs_vmin = min(abs_vmin, float(np.percentile(bg_total, 1.0)))
        abs_vmax = max(abs_vmax, float(np.percentile(bg_total, 99.5)))
        delta_vmax = max(delta_vmax, float(np.percentile(bg_delta, 99.5)))

    if not np.isfinite(abs_vmin):
        abs_vmin = float(np.percentile(np.abs(p_sw), 1.0))
    if abs_vmax <= abs_vmin:
        abs_vmax = abs_vmin + 1.0
    if delta_vmax <= 0.0:
        delta_vmax = 1.0

    return abs_vmin, abs_vmax, delta_vmax


def _draw_two_panel_frame(
    bg_total_abs: np.ndarray,
    bg_delta_abs: np.ndarray,
    xg: np.ndarray,
    yg: np.ndarray,
    traps_m: np.ndarray,
    idx_a: int,
    idx_b: int,
    neigh_idx: np.ndarray,
    A_path: np.ndarray,
    B_path: np.ndarray,
    N_path: np.ndarray,
    centres: np.ndarray,
    alpha_schedule: np.ndarray,
    phase_codes: np.ndarray,
    frame_idx: int,
    subtitle: str,
    ring_radius_mm: float = 0.0,
    centre_now: np.ndarray | None = None,
    alpha_now: float | None = None,
    abs_vmin: float | None = None,
    abs_vmax: float | None = None,
    delta_vmax: float | None = None,
    all_home_xy: np.ndarray | None = None,
    all_current_xy: np.ndarray | None = None,
) -> np.ndarray:
    ext = [xg[0] * 1e3, xg[-1] * 1e3, yg[0] * 1e3, yg[-1] * 1e3]

    A0 = traps_m[idx_a]
    B0 = traps_m[idx_b]
    midpoint = 0.5 * (A0 + B0)
    cxy_now = np.asarray(centres[frame_idx] if centre_now is None else centre_now, dtype=float)
    alpha_disp = float(alpha_schedule[frame_idx] if alpha_now is None else alpha_now)

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.9))

    for i, ax in enumerate(axes):
        if i == 0:
            ax.imshow(
                bg_total_abs,
                origin="lower",
                extent=ext,
                cmap="viridis",
                aspect="equal",
                vmin=abs_vmin,
                vmax=abs_vmax,
            )
        else:
            ax.imshow(
                bg_delta_abs,
                origin="lower",
                extent=ext,
                cmap="magma",
                aspect="equal",
                vmin=0.0,
                vmax=delta_vmax,
            )
        if SHOW_ALL_TRAPS:
            ax.scatter(traps_m[:, 0] * 1e3, traps_m[:, 1] * 1e3, s=8, c="w", alpha=0.12)

        if all_home_xy is not None and all_current_xy is not None:
            home = np.asarray(all_home_xy, dtype=float)
            cur = np.asarray(all_current_xy, dtype=float)
            if SHOW_ALL_PARTICLE_DISPLACEMENTS:
                for j in range(len(home)):
                    ax.plot(
                        [home[j, 0] * 1e3, cur[j, 0] * 1e3],
                        [home[j, 1] * 1e3, cur[j, 1] * 1e3],
                        color="white",
                        linewidth=0.32,
                        alpha=0.22,
                        zorder=2,
                    )
            if SHOW_ALL_PARTICLE_CURRENT:
                ax.scatter(
                    cur[:, 0] * 1e3,
                    cur[:, 1] * 1e3,
                    s=6,
                    c="white",
                    alpha=0.52,
                    linewidths=0.0,
                    zorder=3,
                    label="all particle current" if i == 0 else None,
                )

        nn = traps_m[np.asarray(neigh_idx, dtype=int)]
        if len(nn):
            ax.scatter(
                nn[:, 0] * 1e3,
                nn[:, 1] * 1e3,
                s=28,
                marker="s",
                facecolors="none",
                edgecolors="cyan",
                linewidths=0.7,
                label="neighbour homes" if i == 0 else None,
            )

            ncur = np.asarray(N_path[-1], dtype=float)
            for j in range(len(nn)):
                ax.plot(
                    [nn[j, 0] * 1e3, ncur[j, 0] * 1e3],
                    [nn[j, 1] * 1e3, ncur[j, 1] * 1e3],
                    color="white",
                    linewidth=0.7,
                    alpha=0.65,
                    zorder=3,
                )

            if SHOW_NEIGHBOUR_CURRENT:
                ax.scatter(
                    ncur[:, 0] * 1e3,
                    ncur[:, 1] * 1e3,
                    s=14,
                    c="white",
                    edgecolors="0.3",
                    linewidths=0.35,
                    label="neighbour current" if i == 0 else None,
                )

            if SHOW_NEIGHBOUR_TRAILS and N_path.shape[0] > 1:
                for j in range(ncur.shape[0]):
                    ax.plot(
                        N_path[:, j, 0] * 1e3,
                        N_path[:, j, 1] * 1e3,
                        color="white",
                        linewidth=0.55,
                        alpha=0.30,
                    )

        ax.plot(centres[:, 0] * 1e3, centres[:, 1] * 1e3, "m--", linewidth=1.5, label="vortex path" if i == 0 else None)
        ax.scatter(
            cxy_now[0] * 1e3,
            cxy_now[1] * 1e3,
            s=80,
            c="magenta",
            marker="D",
            edgecolors="k",
            linewidths=0.7,
            label="vortex centre" if i == 0 else None,
        )

        # Draw explicit vortex ring circle so movement is visible
        if ring_radius_mm > 0:
            theta_ring = np.linspace(0.0, 2.0 * np.pi, 160)
            cx_mm = cxy_now[0] * 1e3
            cy_mm = cxy_now[1] * 1e3
            ax.plot(
                cx_mm + ring_radius_mm * np.cos(theta_ring),
                cy_mm + ring_radius_mm * np.sin(theta_ring),
                "m-",
                linewidth=1.4,
                alpha=0.85,
                label="vortex ring" if i == 0 else None,
            )

        ax.scatter(A0[0] * 1e3, A0[1] * 1e3, s=124, c="tomato", edgecolors="k", linewidths=0.8, label="A home" if i == 0 else None)
        ax.scatter(B0[0] * 1e3, B0[1] * 1e3, s=124, c="deepskyblue", edgecolors="k", linewidths=0.8, label="B home" if i == 0 else None)
        ax.plot([A0[0] * 1e3, B0[0] * 1e3], [A0[1] * 1e3, B0[1] * 1e3], "w--", linewidth=1.1)

        ax.plot(A_path[:, 0] * 1e3, A_path[:, 1] * 1e3, color="tomato", linewidth=1.4, alpha=0.9)
        ax.plot(B_path[:, 0] * 1e3, B_path[:, 1] * 1e3, color="deepskyblue", linewidth=1.4, alpha=0.9)
        ax.scatter(
            A_path[-1, 0] * 1e3,
            A_path[-1, 1] * 1e3,
            s=120,
            c="gold",
            edgecolors="k",
            linewidths=0.7,
            label="A current" if i == 0 else None,
        )
        ax.scatter(
            B_path[-1, 0] * 1e3,
            B_path[-1, 1] * 1e3,
            s=120,
            c="orange",
            edgecolors="k",
            linewidths=0.7,
            label="B current" if i == 0 else None,
        )

        ax.set_xlim(mm(midpoint[0] - ROI_HALF), mm(midpoint[0] + ROI_HALF))
        ax.set_ylim(mm(midpoint[1] - ROI_HALF), mm(midpoint[1] + ROI_HALF))
        if i == 1:
            ax.set_title("ROI |p_total - p_sw|")
        else:
            ax.set_title("ROI |p_total|")

        if len(nn):
            disp_um = np.linalg.norm(np.asarray(N_path[-1], dtype=float) - nn, axis=1) * 1e6
            ax.text(
                0.02,
                0.03,
                f"N disp mean={float(np.mean(disp_um)):.1f} um | max={float(np.max(disp_um)):.1f} um",
                transform=ax.transAxes,
                fontsize=7,
                color="white",
                bbox={"boxstyle": "round,pad=0.20", "fc": "black", "ec": "none", "alpha": 0.45},
            )

        if all_home_xy is not None and all_current_xy is not None:
            all_disp_um = np.linalg.norm(np.asarray(all_current_xy, dtype=float) - np.asarray(all_home_xy, dtype=float), axis=1) * 1e6
            ax.text(
                0.02,
                0.11,
                f"All disp mean={float(np.mean(all_disp_um)):.1f} um | max={float(np.max(all_disp_um)):.1f} um",
                transform=ax.transAxes,
                fontsize=7,
                color="white",
                bbox={"boxstyle": "round,pad=0.20", "fc": "black", "ec": "none", "alpha": 0.45},
            )

        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")

    code = int(phase_codes[frame_idx])
    fig.suptitle(
        f"{_phase_title(code)} | alpha={alpha_disp:.2f} | frame {frame_idx + 1}/{len(phase_codes)}\n{subtitle}",
        fontsize=12,
    )
    axes[0].legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    arr = _frame_array(fig)
    plt.close(fig)
    return arr


def _interp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return (1.0 - t) * np.asarray(a, dtype=float) + t * np.asarray(b, dtype=float)


def _build_transport_gif() -> int:
    d = np.load(ROOT_OUT / "transport_case_for_gif.npz", allow_pickle=False)
    xg = d["xg"]
    yg = d["yg"]
    p_sw = d["p_sw"]
    p_v = d["p_vortex_centered"]
    traps = d["traps_m"]
    idx_a = int(d["idx_A"])
    idx_b = int(d["idx_B"])
    neigh_idx = d["neighbor_idx"].astype(int)

    psi = float(d["psi"])
    phase_codes = d["phase_codes"].astype(int)
    alpha_schedule = d["alpha_schedule"]
    centres = d["centres_xy"]
    ring_radius_mm = float(d["ring_radius_m"]) * 1e3 if "ring_radius_m" in d else 0.0

    centres_vis = _smooth_path(centres, window=SMOOTH_WINDOW)

    all_states = _compute_all_schedule_states(
        p_sw,
        p_v,
        xg,
        yg,
        alpha_schedule=alpha_schedule,
        psi=psi,
        centres_xy=centres,
        traps_m=traps,
    )
    all_states_vis = _smooth_cloud_path(all_states, window=SMOOTH_WINDOW)

    all_home = np.asarray(traps, dtype=float)

    abs_vmin, abs_vmax, delta_vmax = _compute_display_limits(
        p_sw,
        p_v,
        xg,
        yg,
        alpha_schedule,
        psi,
        centres_vis,
    )

    frames: list[np.ndarray] = []

    A_hist = [all_home[idx_a].copy()]
    B_hist = [all_home[idx_b].copy()]
    N_hist = [all_home[np.asarray(neigh_idx, dtype=int)].copy()]
    all_cur = all_home.copy()

    for k in range(len(phase_codes) - 1):
        code = int(phase_codes[k])
        interp_n = PUSH_INTERP if code != PHASE_SW_RELEASE else 1

        for j in range(interp_n):
            t = float(j) / float(max(interp_n, 1))
            cxy = _interp(centres_vis[k], centres_vis[k + 1], t)
            alpha = float((1.0 - t) * alpha_schedule[k] + t * alpha_schedule[k + 1])

            all_cur = _interp(all_states_vis[k], all_states_vis[k + 1], t)
            A_hist.append(np.asarray(all_cur[idx_a], dtype=float).copy())
            B_hist.append(np.asarray(all_cur[idx_b], dtype=float).copy())
            N_hist.append(np.asarray(all_cur[np.asarray(neigh_idx, dtype=int)], dtype=float).copy())

            bg_total, bg_delta = _field_components(p_sw, p_v, xg, yg, alpha=alpha, psi=psi, centre_xy=cxy)
            frames.append(
                _draw_two_panel_frame(
                    bg_total,
                    bg_delta,
                    xg,
                    yg,
                    traps,
                    idx_a,
                    idx_b,
                    neigh_idx,
                    A_path=np.asarray(A_hist, dtype=float),
                    B_path=np.asarray(B_hist, dtype=float),
                    N_path=np.asarray(N_hist, dtype=float),
                    centres=centres_vis,
                    alpha_schedule=alpha_schedule,
                    phase_codes=phase_codes,
                    frame_idx=k,
                    subtitle="Moving-vortex transport mechanism",
                    ring_radius_mm=ring_radius_mm,
                    centre_now=cxy,
                    alpha_now=alpha,
                    abs_vmin=abs_vmin,
                    abs_vmax=abs_vmax,
                    delta_vmax=delta_vmax,
                    all_home_xy=all_home,
                    all_current_xy=all_cur,
                )
            )

    all_cur = np.asarray(all_states_vis[-1], dtype=float)
    A_hist.append(np.asarray(all_cur[idx_a], dtype=float).copy())
    B_hist.append(np.asarray(all_cur[idx_b], dtype=float).copy())
    N_hist.append(np.asarray(all_cur[np.asarray(neigh_idx, dtype=int)], dtype=float).copy())

    bg_last_total, bg_last_delta = _field_components(
        p_sw,
        p_v,
        xg,
        yg,
        alpha=float(alpha_schedule[-1]),
        psi=psi,
        centre_xy=centres_vis[-1],
    )
    frames.append(
        _draw_two_panel_frame(
            bg_last_total,
            bg_last_delta,
            xg,
            yg,
            traps,
            idx_a,
            idx_b,
            neigh_idx,
            A_path=np.asarray(A_hist, dtype=float),
            B_path=np.asarray(B_hist, dtype=float),
            N_path=np.asarray(N_hist, dtype=float),
            centres=centres_vis,
            alpha_schedule=alpha_schedule,
            phase_codes=phase_codes,
            frame_idx=len(phase_codes) - 1,
            subtitle="Moving-vortex transport mechanism",
            ring_radius_mm=ring_radius_mm,
            centre_now=centres_vis[-1],
            alpha_now=float(alpha_schedule[-1]),
            abs_vmin=abs_vmin,
            abs_vmax=abs_vmax,
            delta_vmax=delta_vmax,
            all_home_xy=all_home,
            all_current_xy=all_cur,
        )
    )

    out = ROOT_OUT / "moving_vortex_transport.gif"
    _save_gif(frames, out, duration=FRAME_DT)
    print(f"Saved {out} with {len(frames)} frames")
    return len(frames)


def _build_release_gif() -> int:
    d = np.load(ROOT_OUT / "release_case_for_gif.npz", allow_pickle=False)
    xg = d["xg"]
    yg = d["yg"]
    p_sw = d["p_sw"]
    p_v = d["p_vortex_centered"]
    traps = d["traps_m"]
    idx_a = int(d["idx_A"])
    idx_b = int(d["idx_B"])
    neigh_idx = d["neighbor_idx"].astype(int)

    psi = float(d["psi"])
    release_success = bool(d["release_success"])
    phase_codes = d["phase_codes"].astype(int)
    alpha_schedule = d["alpha_schedule"]
    centres = d["centres_xy"]
    ring_radius_mm = float(d["ring_radius_m"]) * 1e3 if "ring_radius_m" in d else 0.0

    centres_vis = _smooth_path(centres, window=SMOOTH_WINDOW)

    all_states = _compute_all_schedule_states(
        p_sw,
        p_v,
        xg,
        yg,
        alpha_schedule=alpha_schedule,
        psi=psi,
        centres_xy=centres,
        traps_m=traps,
    )
    all_states_vis = _smooth_cloud_path(all_states, window=SMOOTH_WINDOW)

    all_home = np.asarray(traps, dtype=float)

    abs_vmin, abs_vmax, delta_vmax = _compute_display_limits(
        p_sw,
        p_v,
        xg,
        yg,
        alpha_schedule,
        psi,
        centres_vis,
    )

    rel_where = np.where(phase_codes == PHASE_SW_RELEASE)[0]
    if len(rel_where) == 0:
        raise RuntimeError("Release phase not found in release case")

    lead_start = max(0, int(rel_where[0]) - 4)
    focus_idx = list(range(lead_start, len(phase_codes)))

    frames: list[np.ndarray] = []
    subtitle = "Release outcome: success" if release_success else "Release outcome: failure"

    k0 = int(focus_idx[0])
    all_cur = np.asarray(all_states_vis[k0], dtype=float)
    A_hist = [np.asarray(all_cur[idx_a], dtype=float).copy()]
    B_hist = [np.asarray(all_cur[idx_b], dtype=float).copy()]
    N_hist = [np.asarray(all_cur[np.asarray(neigh_idx, dtype=int)], dtype=float).copy()]

    bg0_total, bg0_delta = _field_components(
        p_sw,
        p_v,
        xg,
        yg,
        alpha=float(alpha_schedule[k0]),
        psi=psi,
        centre_xy=centres_vis[k0],
    )
    frames.append(
        _draw_two_panel_frame(
            bg0_total,
            bg0_delta,
            xg,
            yg,
            traps,
            idx_a,
            idx_b,
            neigh_idx,
            A_path=np.asarray(A_hist, dtype=float),
            B_path=np.asarray(B_hist, dtype=float),
            N_path=np.asarray(N_hist, dtype=float),
            centres=centres_vis,
            alpha_schedule=alpha_schedule,
            phase_codes=phase_codes,
            frame_idx=k0,
            subtitle=subtitle,
            ring_radius_mm=ring_radius_mm,
            centre_now=centres_vis[k0],
            alpha_now=float(alpha_schedule[k0]),
            abs_vmin=abs_vmin,
            abs_vmax=abs_vmax,
            delta_vmax=delta_vmax,
            all_home_xy=all_home,
            all_current_xy=all_cur,
        )
    )

    for i_k in range(len(focus_idx) - 1):
        k = int(focus_idx[i_k])
        k_next = int(focus_idx[i_k + 1])
        for j in range(1, RELEASE_INTERP + 1):
            t = float(j) / float(max(RELEASE_INTERP, 1))
            all_cur = _interp(all_states_vis[k], all_states_vis[k_next], t)
            A_hist.append(np.asarray(all_cur[idx_a], dtype=float).copy())
            B_hist.append(np.asarray(all_cur[idx_b], dtype=float).copy())
            N_hist.append(np.asarray(all_cur[np.asarray(neigh_idx, dtype=int)], dtype=float).copy())

            cxy = _interp(centres_vis[k], centres_vis[k_next], t)
            alpha_now = float((1.0 - t) * alpha_schedule[k] + t * alpha_schedule[k_next])
            bg_total, bg_delta = _field_components(
                p_sw,
                p_v,
                xg,
                yg,
                alpha=alpha_now,
                psi=psi,
                centre_xy=cxy,
            )
            frames.append(
                _draw_two_panel_frame(
                    bg_total,
                    bg_delta,
                    xg,
                    yg,
                    traps,
                    idx_a,
                    idx_b,
                    neigh_idx,
                    A_path=np.asarray(A_hist, dtype=float),
                    B_path=np.asarray(B_hist, dtype=float),
                    N_path=np.asarray(N_hist, dtype=float),
                    centres=centres_vis,
                    alpha_schedule=alpha_schedule,
                    phase_codes=phase_codes,
                    frame_idx=k_next if j == RELEASE_INTERP else k,
                    subtitle=subtitle,
                    ring_radius_mm=ring_radius_mm,
                    centre_now=cxy,
                    alpha_now=alpha_now,
                    abs_vmin=abs_vmin,
                    abs_vmax=abs_vmax,
                    delta_vmax=delta_vmax,
                    all_home_xy=all_home,
                    all_current_xy=all_cur,
                )
            )

    out = ROOT_OUT / "moving_vortex_release_success_or_failure.gif"
    _save_gif(frames, out, duration=FRAME_DT)
    print(f"Saved {out} with {len(frames)} frames")
    return len(frames)


def main() -> None:
    args = _parse_args()
    render_cfg = _apply_render_config(args)

    print("=" * 72)
    print("Vortex stage transport - Part 7 GIFs (rebuilt staged mechanism)")
    print("=" * 72)

    tc = ROOT_OUT / "transport_case_for_gif.npz"
    rc = ROOT_OUT / "release_case_for_gif.npz"
    if not tc.exists() or not rc.exists():
        raise FileNotFoundError("Run vortex_stage_transport_transport.py first")

    transport_frames = _build_transport_gif()
    release_frames = _build_release_gif()
    _write_render_summary(render_cfg, transport_frames, release_frames)

    print("Part 7 complete.")


if __name__ == "__main__":
    main()
