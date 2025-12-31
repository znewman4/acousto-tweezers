# scripts/animate_patch_move_landscape_3d.py
from __future__ import annotations

# Force a non-interactive backend (prevents intermittent blank 3D frames)
import matplotlib
matplotlib.use("Agg")

from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import imageio.v2 as imageio

from acousto.solvers import solve_helmholtz_2d_neumann_velocity
from acousto.force import ParticleProps, gorkov_potential_and_force_2d
from acousto.analysis import find_traps_from_force

# NEW: render utilities live in tweezers/
from tweezers.viz.render_3d import (
    pick_best_stable_trap,
    render_gorkov_landscape_frame_3d,
    png_is_blankish,
)


def vn_patch_y(
    y: np.ndarray,
    *,
    y_center: float,
    patch_len: float,
    v0: float,
    phase: float = 0.0,
) -> np.ndarray:
    """
    Normal velocity patch on a vertical wall:
      vn(y)=v0*exp(i*phase) on [yc-L/2, yc+L/2], else 0.
    """
    y0 = y_center - 0.5 * patch_len
    y1 = y_center + 0.5 * patch_len
    mask = (y >= y0) & (y <= y1)
    return (v0 * mask.astype(float)) * np.exp(1j * phase)


def main() -> None:
    REPO = Path(__file__).resolve().parents[1]
    RESULTS = REPO / "results"
    gorkov_dir = RESULTS / "gorkov"
    frames_dir = RESULTS / "frames_patchmove_3d"
    gorkov_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Square domain (physical)
    # -----------------------------
    L = 2e-3
    Lx = Ly = L
    N = 180
    Nx = Ny = N

    # Medium + frequency
    f = 2e6
    c0 = 1500.0
    rho0 = 1000.0
    loss_eta = 1e-3

    # Patch settings
    v0 = 1e-4
    patch_len = 0.20e-3

    # Particle props
    particle = ParticleProps(a=5e-6, rho_p=1050.0, c_p=2350.0)

    # Animation settings
    nframes = 200
    fps = 30
    surface_stride = 3  # 2 nicer, 3 faster
    elev, azim = 30.0, -60.0

    # Patch path (stay inside boundary)
    margin = 0.5 * patch_len + 1e-6
    y_centers = np.linspace(margin, Ly - margin, nframes)

    track_xy_mm: list[tuple[float, float]] = []
    frame_paths: list[Path] = []

    # Diagnostics control
    print_first = 12         # always print first N frames
    print_every = 20         # then print every M frames

    # Use this same grid for actuation diagnostics (matches solver evaluation)
    y_grid = np.linspace(0.0, Ly, Ny)

    for k, yc in enumerate(y_centers):
        # Define BC for this frame
        vn_left = lambda y, yc=yc: vn_patch_y(y, y_center=yc, patch_len=patch_len, v0=v0, phase=0.0)

        # --- Diagnostics: boundary actuation health ---
        vn_vals = vn_left(y_grid)
        active = int(np.count_nonzero(np.abs(vn_vals) > 0))
        vn_max = float(np.max(np.abs(vn_vals))) if vn_vals.size else 0.0

        do_print = (k < print_first) or (k % print_every == 0)

        if do_print:
            print(f"[{k:04d}] yc={yc*1e3:.4f}mm | active_nodes={active:3d} | max|vn|={vn_max:.3e}")

        # Solve field
        field = solve_helmholtz_2d_neumann_velocity(
            Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny,
            f=f, c0=c0, rho0=rho0,
            vn_left=vn_left,
            vn_right=0.0,
            vn_bottom=0.0,
            vn_top=0.0,
            loss_eta=loss_eta,
        )

        # --- Diagnostics: field health ---
        p_abs = np.abs(field.p)
        p_max = float(np.max(p_abs))
        p_mean = float(np.mean(p_abs))
        p_finite = bool(np.isfinite(field.p).all())

        # Gor'kov
        U, Fx, Fy = gorkov_potential_and_force_2d(field, particle)
        U_min = float(np.min(U))
        U_max = float(np.max(U))
        U_finite = bool(np.isfinite(U).all())

        if do_print:
            print(
                f"[{k:04d}] |p| max={p_max:.3e} mean={p_mean:.3e} finite={p_finite} | "
                f"U[min,max]=[{U_min:.3e},{U_max:.3e}] finite={U_finite}"
            )

        # Traps
        traps = find_traps_from_force(
            field.x, field.y,
            U, Fx, Fy,
            max_traps=12,
            force_rel_thresh=0.02,
            border=3,
        )

        # Track best stable trap
        best = pick_best_stable_trap(traps)
        if best is not None:
            track_xy_mm.append((best.x * 1e3, best.y * 1e3))

        # Render frame (NEW: via tweezers.viz.render_3d)
        x_mm = field.x * 1e3
        y_mm = field.y * 1e3
        out_png = frames_dir / f"frame_{k:04d}.png"

        render_gorkov_landscape_frame_3d(
            out_png=out_png,
            x_mm=x_mm,
            y_mm=y_mm,
            U=U,
            traps=traps,
            y_center_mm=yc * 1e3,
            patch_len_mm=patch_len * 1e3,
            track_xy_mm=track_xy_mm,
            surface_stride=surface_stride,
            elev=elev,
            azim=azim,
        )
        frame_paths.append(out_png)

        # --- Diagnostics: detect blankish PNGs ---
        if do_print:
            blankish, std, mean, fsize = png_is_blankish(out_png)
            if blankish:
                print(f"[{k:04d}] WARNING: saved PNG looks blank-ish (std={std:.3f}, mean={mean:.1f}, size={fsize} bytes) -> {out_png.name}")
            else:
                print(f"[{k:04d}] PNG ok (std={std:.3f}, mean={mean:.1f}, size={fsize} bytes)")

        if k == 0 or (k + 1) % 25 == 0:
            print(f"Rendered {k+1}/{nframes}: {out_png.name}")

    # Build GIF (loop forever)
    gif_path = gorkov_dir / "gorkov_landscape_patchmove_3d_square.gif"
    duration = 1.0 / fps
    images = [imageio.imread(p) for p in frame_paths]
    imageio.mimsave(
        gif_path,
        images,
        duration=duration,
        loop=0,
        subrectangles=False,
    )
    print(f"Saved GIF: {gif_path}")


if __name__ == "__main__":
    main()
