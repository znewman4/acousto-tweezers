#!/usr/bin/env python3
"""
Part 7 - GIF visualizations for symmetric-vortex push and release.

Outputs:
  results/deliverables/vortex_limit/gifs/
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

from scripts.lib.vortex_limit_utils import compute_gorkov_fields, descend_potential, ensure_dir

try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover
    imageio = None

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

ROOT_OUT = ensure_dir(PROJECT_ROOT / "results" / "deliverables" / "vortex_limit")
LIMIT_DIR = ROOT_OUT / "limit_plots"
REF_DIR = ROOT_OUT / "reference"
OUT = ensure_dir(ROOT_OUT / "gifs")

ROI_HALF = 1.20e-3
PUSH_WINDOW = 5
PUSH_INTERP = 9
RELEASE_FRAMES = 24
START_HOLD = 8
TARGET_HOLD = 10
END_HOLD = 12
FRAME_DURATION = 0.11


def _mm(v: float) -> float:
    return float(v * 1e3)


def _frame_array(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return np.array(rgba[..., :3], copy=True)


def _save_gif(frames: list[np.ndarray], path: Path, duration: float) -> None:
    if Image is not None:
        pil_frames = [Image.fromarray(f) for f in frames]
        pil_frames[0].save(
            path,
            save_all=True,
            append_images=pil_frames[1:],
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


def _interp_xy(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return (1.0 - t) * np.asarray(a, dtype=float) + t * np.asarray(b, dtype=float)


def _resample_path(path_xy: np.ndarray, n_out: int) -> np.ndarray:
    pts = np.asarray(path_xy, dtype=float)
    if len(pts) == 0:
        raise ValueError("path_xy must contain at least one point")
    if len(pts) == 1:
        return np.repeat(pts, n_out, axis=0)

    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if float(s[-1]) <= 1e-18:
        return np.repeat(pts[:1], n_out, axis=0)

    q = np.linspace(0.0, float(s[-1]), n_out)
    out = np.zeros((n_out, 2), dtype=float)
    for j, qq in enumerate(q):
        k = int(np.searchsorted(s, qq, side="right") - 1)
        k = min(max(k, 0), len(pts) - 2)
        denom = max(float(s[k + 1] - s[k]), 1e-18)
        t = float((qq - s[k]) / denom)
        out[j] = _interp_xy(pts[k], pts[k + 1], t)
    return out


def _path_ij_to_xy(path_ij: np.ndarray, xg: np.ndarray, yg: np.ndarray) -> np.ndarray:
    return np.array([[xg[int(ix)], yg[int(iy)]] for iy, ix in np.asarray(path_ij, dtype=int)], dtype=float)


def _focus_indices(alpha_all: np.ndarray, i_target: int, window: int = PUSH_WINDOW) -> list[int]:
    i0 = max(0, i_target - window + 1)
    return list(range(i0, i_target + 1))


def _compute_neighbor_equilibria(
    alpha_seq: np.ndarray,
    xg: np.ndarray,
    yg: np.ndarray,
    p_sw: np.ndarray,
    p_vortex: np.ndarray,
    psi_opt: float,
    traps_m: np.ndarray,
    neighbor_idx: np.ndarray,
) -> np.ndarray:
    dx = float(xg[1] - xg[0])
    dy = float(yg[1] - yg[0])
    neighbor_home = traps_m[np.asarray(neighbor_idx, dtype=int)]
    out: list[np.ndarray] = []

    for alpha in np.asarray(alpha_seq, dtype=float):
        p_comb = p_sw + float(alpha) * np.exp(1j * psi_opt) * p_vortex
        U_c, _, _ = compute_gorkov_fields(p_comb, dx, dy)
        cur = np.array(
            [descend_potential(U_c, xg, yg, start_xy)["xy"] for start_xy in neighbor_home],
            dtype=float,
        )
        out.append(cur)

    return np.array(out, dtype=float)


def _draw_frame(
    p_sw: np.ndarray,
    xg: np.ndarray,
    yg: np.ndarray,
    traps_m: np.ndarray,
    A0: np.ndarray,
    B0: np.ndarray,
    Acur: np.ndarray,
    Bcur: np.ndarray,
    neighbor_home: np.ndarray,
    neighbor_cur: np.ndarray,
    A_trail: np.ndarray,
    B_trail: np.ndarray,
    stage: str,
    alpha_text: str,
    p_vortex: np.ndarray | None = None,
    alpha_v: float = 0.0,
    psi_v: float = 0.0,
) -> np.ndarray:
    ext = [xg[0] * 1e3, xg[-1] * 1e3, yg[0] * 1e3, yg[-1] * 1e3]
    midpoint = 0.5 * (A0 + B0)

    if p_vortex is not None and alpha_v > 1e-12:
        p_bg = p_sw + alpha_v * np.exp(1j * psi_v) * p_vortex
        cbar_label = "|p_sw + vortex| [Pa]"
    else:
        p_bg = p_sw
        cbar_label = "|p_sw| [Pa]"

    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    im = ax.imshow(np.abs(p_bg), origin="lower", extent=ext, cmap="viridis", aspect="equal")

    ax.scatter(traps_m[:, 0] * 1e3, traps_m[:, 1] * 1e3, s=10, c="w", alpha=0.20)

    if len(neighbor_home):
        ax.scatter(
            neighbor_home[:, 0] * 1e3,
            neighbor_home[:, 1] * 1e3,
            s=50,
            facecolors="none",
            edgecolors="0.80",
            linewidths=1.0,
            label="adjacent neighbours (home)",
        )
        for home_xy, cur_xy in zip(neighbor_home, neighbor_cur):
            ax.plot(
                [home_xy[0] * 1e3, cur_xy[0] * 1e3],
                [home_xy[1] * 1e3, cur_xy[1] * 1e3],
                color="0.85",
                linewidth=0.9,
                alpha=0.90,
                zorder=3,
            )
        ax.scatter(
            neighbor_cur[:, 0] * 1e3,
            neighbor_cur[:, 1] * 1e3,
            s=22,
            c="white",
            edgecolors="0.35",
            linewidths=0.4,
            alpha=0.95,
            zorder=4,
            label="adjacent neighbours (current)",
        )

    ax.scatter(
        A0[0] * 1e3,
        A0[1] * 1e3,
        s=170,
        facecolors="none",
        edgecolors="tomato",
        linewidths=2.0,
        label="trap A (A home)",
        zorder=6,
    )
    ax.scatter(
        B0[0] * 1e3,
        B0[1] * 1e3,
        s=170,
        facecolors="none",
        edgecolors="deepskyblue",
        linewidths=2.0,
        label="trap B (B home / A target)",
        zorder=6,
    )

    ax.plot([A0[0] * 1e3, B0[0] * 1e3], [A0[1] * 1e3, B0[1] * 1e3], "w--", linewidth=1.2, alpha=0.85)

    if len(A_trail) >= 2:
        ax.plot(A_trail[:, 0] * 1e3, A_trail[:, 1] * 1e3, color="tomato", linewidth=2.2, alpha=0.95, zorder=7)
    if len(B_trail) >= 2:
        ax.plot(B_trail[:, 0] * 1e3, B_trail[:, 1] * 1e3, color="deepskyblue", linewidth=2.0, alpha=0.95, zorder=7)

    ax.scatter(
        Acur[0] * 1e3,
        Acur[1] * 1e3,
        s=140,
        c="tomato",
        marker="o",
        edgecolors="k",
        linewidths=0.8,
        zorder=8,
        label="A (being transported)",
    )
    ax.scatter(
        Bcur[0] * 1e3,
        Bcur[1] * 1e3,
        s=150,
        c="deepskyblue",
        marker="s",
        edgecolors="k",
        linewidths=0.8,
        zorder=8,
        label="B (must stay near trap B)",
    )

    text_dx = 0.030
    text_dy = 0.025
    ax.text(
        Acur[0] * 1e3 + text_dx,
        Acur[1] * 1e3 + text_dy,
        "A",
        fontsize=10,
        weight="bold",
        color="black",
        bbox={"boxstyle": "round,pad=0.22", "fc": "tomato", "ec": "k", "alpha": 0.95},
        zorder=9,
    )
    ax.text(
        Bcur[0] * 1e3 + text_dx,
        Bcur[1] * 1e3 - text_dy,
        "B",
        fontsize=10,
        weight="bold",
        color="black",
        bbox={"boxstyle": "round,pad=0.22", "fc": "deepskyblue", "ec": "k", "alpha": 0.95},
        zorder=9,
    )

    ax.text(
        0.02,
        0.03,
        f"adjacent neighbours shown: {len(neighbor_home)}",
        transform=ax.transAxes,
        fontsize=9,
        color="white",
        bbox={"boxstyle": "round,pad=0.25", "fc": "black", "ec": "none", "alpha": 0.45},
    )

    ax.set_xlim(_mm(midpoint[0] - ROI_HALF), _mm(midpoint[0] + ROI_HALF))
    ax.set_ylim(_mm(midpoint[1] - ROI_HALF), _mm(midpoint[1] + ROI_HALF))
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(f"{stage}\n{alpha_text}")
    ax.legend(loc="upper right", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.82, label=cbar_label)
    fig.tight_layout()

    arr = _frame_array(fig)
    plt.close(fig)
    return arr


def _build_case_gif(
    tag: str,
    alpha_target: float,
    is_transport_success: bool,
    alpha_all: np.ndarray,
    Aeq_all: np.ndarray,
    Beq_all: np.ndarray,
    U_sw: np.ndarray,
    xg: np.ndarray,
    yg: np.ndarray,
    p_sw: np.ndarray,
    p_vortex: np.ndarray,
    psi_opt: float,
    traps_m: np.ndarray,
    idx_a: int,
    idx_b: int,
    neighbor_idx: np.ndarray,
) -> None:
    A0 = np.asarray(traps_m[idx_a], dtype=float)
    B0 = np.asarray(traps_m[idx_b], dtype=float)
    neighbor_idx = np.asarray(neighbor_idx, dtype=int)
    neighbor_home = np.asarray(traps_m[neighbor_idx], dtype=float)

    i_target = int(np.argmin(np.abs(alpha_all - alpha_target)))
    focus_idx = _focus_indices(alpha_all, i_target)
    alpha_focus = np.asarray(alpha_all[focus_idx], dtype=float)
    A_focus = np.asarray(Aeq_all[focus_idx], dtype=float)
    B_focus = np.asarray(Beq_all[focus_idx], dtype=float)
    neigh_focus = _compute_neighbor_equilibria(alpha_focus, xg, yg, p_sw, p_vortex, psi_opt, traps_m, neighbor_idx)

    frames: list[np.ndarray] = []

    baseline = _draw_frame(
        p_sw,
        xg,
        yg,
        traps_m,
        A0,
        B0,
        A0,
        B0,
        neighbor_home,
        neighbor_home,
        A_trail=np.array([A0], dtype=float),
        B_trail=np.array([B0], dtype=float),
        stage="Standing wave only",
        alpha_text="alpha=0.00",
        p_vortex=None,
    )
    frames.extend([baseline.copy() for _ in range(START_HOLD)])

    prev_alpha = 0.0
    prev_A = A0.copy()
    prev_B = B0.copy()
    prev_neigh = neighbor_home.copy()
    A_trail = [A0.copy()]
    B_trail = [B0.copy()]

    for alpha_next, A_next, B_next, neigh_next in zip(alpha_focus, A_focus, B_focus, neigh_focus):
        for j in range(1, PUSH_INTERP + 1):
            t = float(j) / float(PUSH_INTERP)
            alpha_frame = (1.0 - t) * prev_alpha + t * float(alpha_next)
            A_frame = _interp_xy(prev_A, A_next, t)
            B_frame = _interp_xy(prev_B, B_next, t)
            neigh_frame = (1.0 - t) * prev_neigh + t * neigh_next
            trail_A = np.vstack(A_trail + [A_frame])
            trail_B = np.vstack(B_trail + [B_frame])
            frames.append(
                _draw_frame(
                    p_sw,
                    xg,
                    yg,
                    traps_m,
                    A0,
                    B0,
                    A_frame,
                    B_frame,
                    neighbor_home,
                    neigh_frame,
                    A_trail=trail_A,
                    B_trail=trail_B,
                    stage="Vortex ON - pushing A toward trap B",
                    alpha_text=f"alpha={alpha_frame:.2f} | psi={psi_opt:.2f} rad",
                    p_vortex=p_vortex,
                    alpha_v=alpha_frame,
                    psi_v=psi_opt,
                )
            )
        prev_alpha = float(alpha_next)
        prev_A = np.asarray(A_next, dtype=float)
        prev_B = np.asarray(B_next, dtype=float)
        prev_neigh = np.asarray(neigh_next, dtype=float)
        A_trail.append(prev_A.copy())
        B_trail.append(prev_B.copy())

    final_push = frames[-1].copy()
    frames.extend([final_push.copy() for _ in range(TARGET_HOLD)])

    relA = descend_potential(U_sw, xg, yg, prev_A)
    relB = descend_potential(U_sw, xg, yg, prev_B)
    pathA_xy = _resample_path(_path_ij_to_xy(relA["path_ij"], xg, yg), RELEASE_FRAMES)
    pathB_xy = _resample_path(_path_ij_to_xy(relB["path_ij"], xg, yg), RELEASE_FRAMES)

    neigh_release_paths: list[np.ndarray] = []
    for start_xy in prev_neigh:
        relN = descend_potential(U_sw, xg, yg, start_xy)
        neigh_release_paths.append(_resample_path(_path_ij_to_xy(relN["path_ij"], xg, yg), RELEASE_FRAMES))
    neigh_release = np.stack(neigh_release_paths, axis=1) if neigh_release_paths else np.zeros((RELEASE_FRAMES, 0, 2), dtype=float)

    release_title = (
        "Vortex OFF - A remains captured at trap B"
        if is_transport_success
        else "Vortex OFF - A falls back toward trap A"
    )
    for k in range(RELEASE_FRAMES):
        A_frame = pathA_xy[k]
        B_frame = pathB_xy[k]
        neigh_frame = neigh_release[k]
        frames.append(
            _draw_frame(
                p_sw,
                xg,
                yg,
                traps_m,
                A0,
                B0,
                A_frame,
                B_frame,
                neighbor_home,
                neigh_frame,
                A_trail=pathA_xy[: k + 1],
                B_trail=pathB_xy[: k + 1],
                stage=release_title,
                alpha_text=f"release from alpha={alpha_target:.2f}",
                p_vortex=None,
            )
        )

    final_release = frames[-1].copy()
    frames.extend([final_release.copy() for _ in range(END_HOLD)])

    out_path = OUT / f"vortex_push_release_{tag}.gif"
    _save_gif(frames, out_path, duration=FRAME_DURATION)
    print(f"Saved {out_path} with {len(frames)} frames")


def main() -> None:
    print("=" * 68)
    print("Vortex limit study - Part 7 GIFs")
    print("=" * 68)

    gif_npz = LIMIT_DIR / "gif_inputs.npz"
    ref_npz = REF_DIR / "reference_fields.npz"

    if not gif_npz.exists() or not ref_npz.exists():
        raise FileNotFoundError("Run vortex_limit_analysis.py first")

    d = np.load(gif_npz)
    ref = np.load(ref_npz)

    alpha_all = d["alpha"]
    Aeq_all = d["A_eq_xy"]
    Beq_all = d["B_eq_xy"]

    xg = d["xg"]
    yg = d["yg"]
    p_sw = d["p_sw"]
    p_vortex = d["p_vortex_scaled"]
    psi_opt = float(d["psi_opt"])

    traps_m = d["traps_m"]
    idx_a = int(d["idx_A"])
    idx_b = int(d["idx_B"])
    neighbor_idx = d["neighbor_idx"].astype(int)

    U_sw = ref["U_sw"]

    safe_alpha = float(d["safe_alpha"])
    fail_alpha = float(d["fail_alpha"])

    if np.isfinite(safe_alpha):
        _build_case_gif(
            tag="safe",
            alpha_target=safe_alpha,
            is_transport_success=True,
            alpha_all=alpha_all,
            Aeq_all=Aeq_all,
            Beq_all=Beq_all,
            U_sw=U_sw,
            xg=xg,
            yg=yg,
            p_sw=p_sw,
            p_vortex=p_vortex,
            psi_opt=psi_opt,
            traps_m=traps_m,
            idx_a=idx_a,
            idx_b=idx_b,
            neighbor_idx=neighbor_idx,
        )
    else:
        print("No safe alpha identified; skipping safe GIF.")

    if np.isfinite(fail_alpha):
        _build_case_gif(
            tag="failure",
            alpha_target=fail_alpha,
            is_transport_success=False,
            alpha_all=alpha_all,
            Aeq_all=Aeq_all,
            Beq_all=Beq_all,
            U_sw=U_sw,
            xg=xg,
            yg=yg,
            p_sw=p_sw,
            p_vortex=p_vortex,
            psi_opt=psi_opt,
            traps_m=traps_m,
            idx_a=idx_a,
            idx_b=idx_b,
            neighbor_idx=neighbor_idx,
        )
    else:
        print("No failure alpha identified; skipping failure GIF.")

    print("Part 7 complete.")


if __name__ == "__main__":
    main()
