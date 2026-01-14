#!/usr/bin/env python3
"""
Enhanced path_follow with multi-view visualization.
Shows: 3D Gorkov + particle, 2D top-down trajectory, particle position over time.

This is the legacy random-shooting version. For the new structured controller,
see path_follow_controlled.py which uses the ParticleController class.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from typing import Optional

from acousto.force import ParticleProps
from acousto.analysis import find_traps_from_force

from tweezers.control import (
    DishDomain, MediumProps, EvaluatorConfig,
    Control2Pucks, BottomFootprint25DEvaluator,
)

from tweezers.viz.render_3d import (
    Cylinder2D, render_gorkov_landscape_frame_3d,
    png_is_blankish, classify_trap,
    normalize_gorkov_field,
)


def render_with_particle_multiview(
    *,
    out_png: Path,
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    U: np.ndarray,
    traps,
    particle_xy_mm: tuple[float, float],
    track_xy_mm: list[tuple[float, float]] | None = None,
    cylinders=None,
    surface_stride: int = 3,
) -> bool:
    """
    Multi-view rendering:
      - (Top-left) 3D Gorkov landscape with particle on surface
      - (Top-right) 2D trajectory from above (x-y plane)
      - (Bottom) Particle position vs time
    
    Returns:
    --------
    is_flat : bool - True if landscape was flat/nearly-flat (caller can reuse previous frame)
    """
    px_mm, py_mm = particle_xy_mm
    track_len = len(track_xy_mm) if track_xy_mm else 0
    
    U_min = float(np.min(U))
    U_max = float(np.max(U))
    den_orig = U_max - U_min  # For trap positioning (unscaled U space)
    print(f"  [render] U range: [{U_min:.3e}, {U_max:.3e}]  particle at ({px_mm:.3f}, {py_mm:.3f}) mm, track_len={track_len}")
    
    # Prepare data
    from mpl_toolkits.mplot3d import Axes3D
    X, Y = np.meshgrid(x_mm, y_mm)
    
    # EXAGGERATE landscape for visualization (doesn't affect physics, just display)
    # This helps see the surface topology even though actual U values are tiny
    U_display = U * 1e15  # Scale for visualization
    
    # Use centralized normalization
    Uvis, is_flat = normalize_gorkov_field(U_display, verbose=False)
    
    # Compute dynamic range for visualization (display space)
    U_display_min = float(np.nanmin(U_display))
    U_display_max = float(np.nanmax(U_display))
    den = U_display_max - U_display_min
    
    z0 = -0.25
    Xs = X[::surface_stride, ::surface_stride]
    Ys = Y[::surface_stride, ::surface_stride]
    Us = Uvis[::surface_stride, ::surface_stride]
    
    # Create 3-panel figure
    fig = plt.figure(figsize=(14, 10))
    
    # ============ Panel 1: 3D Landscape (top-left) ============
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    ax1.plot_surface(Xs, Ys, Us, linewidth=0, antialiased=True, alpha=0.95, cmap="viridis")
    ax1.contour(X, Y, Uvis, levels=12, offset=z0, alpha=0.4)
    
    # Draw cylinders (transducers)
    if cylinders:
        from tweezers.viz.render_3d import _draw_cylinder_surface
        for cyl in cylinders:
            _draw_cylinder_surface(ax1, cyl=cyl)
    
    # Draw traps
    for t in traps:
        ttype = classify_trap(np.asarray(t.eigvals))
        mx, my = (float(t.x) * 1e3), (float(t.y) * 1e3)
        mz = 0.0 if den_orig == 0.0 else (float(t.U) - U_min) / den_orig
        
        if ttype == "min":
            ax1.scatter(mx, my, mz, s=50, marker="o", color="green", alpha=0.7)
        elif ttype == "saddle":
            ax1.scatter(mx, my, mz, s=50, marker="x", color="blue", alpha=0.7)
        else:
            ax1.scatter(mx, my, mz, s=55, marker="^", color="red", alpha=0.7)
    
    # Draw trajectory with gradient
    if track_xy_mm is not None and len(track_xy_mm) >= 2:
        tx = np.array([p[0] for p in track_xy_mm])
        ty = np.array([p[1] for p in track_xy_mm])
        tz = np.array([z0] * len(tx))
        
        n_pts = len(tx)
        for i in range(n_pts - 1):
            alpha_color = i / max(n_pts - 1, 1)
            color = (1-alpha_color) * np.array([0.5, 0.5, 0.5]) + alpha_color * np.array([0, 1, 1])
            ax1.plot(tx[i:i+2], ty[i:i+2], tz[i:i+2], linewidth=2.0, color=color, alpha=0.9)
    
    # Draw particle on surface - CLEARLY VISIBLE ON TOP
    ix = np.argmin(np.abs(x_mm - px_mm))
    iy = np.argmin(np.abs(y_mm - py_mm))
    pz_on_surface = Uvis[iy, ix] if (0 <= iy < Uvis.shape[0] and 0 <= ix < Uvis.shape[1]) else 0.5
    
    # Offset particle ABOVE surface so it's unmissable - sitting on top like a ball on a hill
    pz_particle = min(1.15, pz_on_surface + 0.08)  # Sits prominently on surface
    
    ax1.scatter(px_mm, py_mm, pz_particle, s=1500, marker="o", color="red", alpha=1.0,
                edgecolors="white", linewidth=6, zorder=1000)
    
    ax1.set_xlabel("x (mm)")
    ax1.set_ylabel("y (mm)")
    ax1.set_zlabel("U (norm)")
    ax1.set_zlim(z0, 1.05)
    ax1.set_box_aspect((np.ptp(x_mm), np.ptp(y_mm), 0.8))
    ax1.view_init(elev=30, azim=-60)
    ax1.set_title("3D: Gorkov Landscape + Particle", fontsize=10, fontweight="bold")
    
    # ============ Panel 2: 2D Top-Down Trajectory (top-right) ============
    ax2 = fig.add_subplot(2, 2, 2)
    
    # Contour of U in x-y plane
    contour = ax2.contourf(X, Y, Uvis, levels=20, cmap="viridis", alpha=0.7)
    ax2.contour(X, Y, Uvis, levels=12, colors="k", linewidths=0.3, alpha=0.3)
    
    # Draw trajectory with gradient
    if track_xy_mm is not None and len(track_xy_mm) >= 2:
        tx = np.array([p[0] for p in track_xy_mm])
        ty = np.array([p[1] for p in track_xy_mm])
        
        n_pts = len(tx)
        for i in range(n_pts - 1):
            alpha_color = i / max(n_pts - 1, 1)
            color = (1-alpha_color) * np.array([0.5, 0.5, 0.5]) + alpha_color * np.array([0, 1, 1])
            ax2.plot(tx[i:i+2], ty[i:i+2], linewidth=3.0, color=color, alpha=0.9)
        
        # Start point (green) and end point (red)
        ax2.scatter(tx[0], ty[0], s=200, marker="o", color="green", edgecolors="white", linewidth=2, label="start", zorder=10)
        ax2.scatter(tx[-1], ty[-1], s=200, marker="*", color="red", edgecolors="white", linewidth=2, label="current", zorder=10)
    
    # Draw transducers
    if cylinders:
        for cyl in cylinders:
            circle = plt.Circle((cyl.x_mm, cyl.y_mm), cyl.r_mm, fill=False, edgecolor="yellow", linewidth=2, linestyle="--")
            ax2.add_patch(circle)
    
    ax2.set_xlabel("x (mm)")
    ax2.set_ylabel("y (mm)")
    ax2.set_title("2D: Top-Down View (x-y)", fontsize=10, fontweight="bold")
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right", fontsize=8)
    
    # ============ Panel 3: Trajectory Time Series (bottom) ============
    ax3 = fig.add_subplot(2, 2, (3, 4))
    
    if track_xy_mm is not None and len(track_xy_mm) >= 2:
        tx = np.array([p[0] for p in track_xy_mm]) * 1e3  # Convert to mm
        ty = np.array([p[1] for p in track_xy_mm]) * 1e3
        t_steps = np.arange(len(tx))
        
        ax3.plot(t_steps, tx, linewidth=2.5, marker="o", markersize=3, label="x position", color="blue", alpha=0.8)
        ax3.plot(t_steps, ty, linewidth=2.5, marker="s", markersize=3, label="y position", color="red", alpha=0.8)
        
        # Mark current position
        ax3.scatter(t_steps[-1], tx[-1], s=100, marker="o", color="blue", edgecolors="white", linewidth=2, zorder=10)
        ax3.scatter(t_steps[-1], ty[-1], s=100, marker="s", color="red", edgecolors="white", linewidth=2, zorder=10)
        
        ax3.set_xlabel("Time Step", fontsize=10)
        ax3.set_ylabel("Position (mm)", fontsize=10)
        ax3.set_title("Particle Position Over Time", fontsize=10, fontweight="bold")
        ax3.grid(True, alpha=0.4)
        ax3.legend(loc="upper left", fontsize=9)
    
    fig.suptitle(f"Path Following Control: Particle at ({px_mm:.3f}, {py_mm:.3f}) mm  Track: {track_len} points", 
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.canvas.draw()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"    [render] saved multi-view frame")
    
    return is_flat


def make_polyline_path(points: list[tuple[float, float]], T: int) -> np.ndarray:
    """Piecewise-linear path sampled at T points."""
    pts = np.array(points, dtype=float)
    seg = pts[1:] - pts[:-1]
    seglen = np.sqrt(np.sum(seg**2, axis=1))
    s = np.concatenate([[0.0], np.cumsum(seglen)])
    total = float(s[-1]) if s[-1] > 0 else 1.0

    tvals = np.linspace(0.0, total, T)
    out = np.zeros((T, 2), dtype=float)

    j = 0
    for i, tv in enumerate(tvals):
        while j < len(seglen) - 1 and tv > s[j + 1]:
            j += 1
        if seglen[j] < 1e-12:
            out[i] = pts[j]
        else:
            a = (tv - s[j]) / seglen[j]
            out[i] = pts[j] + a * seg[j]
    return out


def propose_controls(
    u_prev: Control2Pucks,
    rng: np.random.Generator,
    *,
    n: int,
    dx: float,
    dy: float,
    dv: float,
    dphi: float,
) -> list[Control2Pucks]:
    """Gaussian perturbations around u_prev."""
    out: list[Control2Pucks] = []
    for _ in range(n):
        out.append(
            Control2Pucks(
                xA=u_prev.xA + rng.normal(scale=dx),
                yA=u_prev.yA + rng.normal(scale=dy),
                xB=u_prev.xB + rng.normal(scale=dx),
                yB=u_prev.yB + rng.normal(scale=dy),
                vA=u_prev.vA + rng.normal(scale=dv),
                vB=u_prev.vB + rng.normal(scale=dv),
                phiA=u_prev.phiA + rng.normal(scale=dphi),
                phiB=u_prev.phiB + rng.normal(scale=dphi),
            )
        )
    return out


def main() -> None:
    REPO = Path(__file__).resolve().parents[1]
    RESULTS = REPO / "results"
    frames_dir = RESULTS / "frames_path_follow_multiview"
    out_dir = RESULTS / "path_follow"
    frames_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Domain + medium
    domain = DishDomain(Lx=2e-3, Ly=2e-3, Nx=160, Ny=160)
    medium = MediumProps(f=2e6, c0=1500.0, rho0=1000.0, loss_eta=1e-3, kz=0.0, coupling_alpha=1.0)
    particle = ParticleProps(a=5e-6, rho_p=1050.0, c_p=2350.0)

    cfg = EvaluatorConfig(
        sigma_x=0.10e-3,
        bottom_band=0.25e-3,
        dt=5e-3,               # INCREASED: was 2e-3 (faster particle motion)
        viscosity=1e-3,
        border_penalty=1e6,
        smooth_u=0.0,          # ZERO: allow controls to change freely
        alpha_g=1e6,
    )

    ev = BottomFootprint25DEvaluator(domain, medium, particle, cfg)

    # Desired path
    T = 100
    path = make_polyline_path(
        points=[
            (0.3e-3, 0.7e-3),
            (1.7e-3, 0.7e-3),
            (1.7e-3, 1.5e-3),
            (0.3e-3, 1.5e-3),
            (0.3e-3, 0.7e-3),
        ],
        T=T,
    )

    # Initial state
    xp, yp = float(path[0, 0]), float(path[0, 1])

    u = Control2Pucks(
        xA=0.5e-3, yA=0.15e-3,
        xB=1.5e-3, yB=0.15e-3,
        vA=5e-4, vB=5e-4,
        phiA=0.0, phiB=np.pi,
    )

    rng = np.random.default_rng(0)

    # Optimizer - REDUCED perturbations for smoother transducer motion
    K = 80   # Candidates per step
    dx = 0.05e-3   # Reduced: was 0.25e-3 (too fast)
    dy = 0.03e-3   # Reduced: was 0.12e-3 (too fast)
    dv = 0.5e-4    # Reduced: was 1.5e-4
    dphi = 0.5     # Reduced: was 2.0 (too fast phase changes)

    # Rendering
    render_every = 3  # More frequent rendering
    cyl_r_mm = (2.0 * cfg.sigma_x) * 1e3

    traj_xy_mm: list[tuple[float, float]] = [(xp * 1e3, yp * 1e3)]
    frame_paths: list[Path] = []

    # ✅ FRAME REUSE: when landscape is flat, copy previous frame instead of rendering new flat plane
    prev_frame_path: Optional[Path] = None
    flat_frame_count = 0

    print_first = 10
    print_every = 10

    for t in range(T - 1):
        tx, ty = float(path[t + 1, 0]), float(path[t + 1, 1])

        candidates = propose_controls(u, rng, n=K, dx=dx, dy=dy, dv=dv, dphi=dphi)
        candidates.append(u)

        best_loss = None
        best_u = None
        best_next = None

        for uc in candidates:
            xp1, yp1, loss, _info = ev.step(xp=xp, yp=yp, target_x=tx, target_y=ty, u=uc, u_prev=u)
            if best_loss is None or loss < best_loss:
                best_loss = loss
                best_u = uc
                best_next = (xp1, yp1)

        assert best_u is not None and best_next is not None and best_loss is not None
        u = ev.clip_control(best_u)
        xp, yp = best_next

        traj_xy_mm.append((xp * 1e3, yp * 1e3))

        do_print = (t < print_first) or ((t + 1) % print_every == 0)
        if do_print:
            print(f"[{t+1:03d}/{T-1:03d}] loss={best_loss:.3e}  p=({xp*1e3:.3f}mm,{yp*1e3:.3f}mm)")

        # Render occasionally
        if (t % render_every) == 0:
            xp1, yp1, loss, info, field, U, Fx, Fy = ev.step(
                xp=xp, yp=yp, target_x=tx, target_y=ty, u=u, u_prev=u,
                return_fields=True,
            )

            traps = find_traps_from_force(
                field.x, field.y, U, Fx, Fy,
                max_traps=12, force_rel_thresh=0.02, border=3,
            )

            cylinders = [
                Cylinder2D(x_mm=u.xA * 1e3, y_mm=u.yA * 1e3, r_mm=cyl_r_mm, alpha=0.22, edge_alpha=0.60),
                Cylinder2D(x_mm=u.xB * 1e3, y_mm=u.yB * 1e3, r_mm=cyl_r_mm, alpha=0.22, edge_alpha=0.60),
            ]

            out_png = frames_dir / f"frame_{t:04d}.png"

            # ✅ Check if landscape is flat - if so, reuse previous frame
            # WITH DETAILED DIAGNOSTICS
            frame_label = f"t={t}"
            U_display = U * 1e15
            print(f"\n[DIAGNOSTICS frame {frame_label}]")
            print(f"  U original range: [{float(np.nanmin(U)):.3e}, {float(np.nanmax(U)):.3e}]")
            print(f"  U_display range: [{float(np.nanmin(U_display)):.3e}, {float(np.nanmax(U_display)):.3e}]")
            
            _, is_flat = normalize_gorkov_field(U_display, verbose=True, frame_id=frame_label)
            print(f"  is_flat={is_flat}, prev_frame_path={'exists' if prev_frame_path else 'None'}")
            
            if is_flat and prev_frame_path is not None:
                # Landscape is flat - copy previous frame instead of rendering
                import shutil
                shutil.copy(prev_frame_path, out_png)
                flat_frame_count += 1
                print(f"  ACTION: Reusing previous frame (flat_count={flat_frame_count})")
            else:
                # Normal rendering
                print(f"  ACTION: Rendering new frame...")
                is_flat = render_with_particle_multiview(
                    out_png=out_png,
                    x_mm=field.x * 1e3,
                    y_mm=field.y * 1e3,
                    U=U,
                    traps=traps,
                    particle_xy_mm=(xp * 1e3, yp * 1e3),
                    track_xy_mm=traj_xy_mm,
                    cylinders=cylinders,
                )
                print(f"  Saved to {out_png}")
            
            prev_frame_path = out_png
            frame_paths.append(out_png)

    # Build GIF
    gif_path = out_dir / "path_follow_multiview.gif"
    images = [imageio.imread(p) for p in frame_paths]
    imageio.mimsave(gif_path, images, duration=0.5, loop=0, subrectangles=False)
    print(f"\nSaved multi-view GIF: {gif_path}")

    np.save(out_dir / "traj_xy_mm.npy", np.array(traj_xy_mm, dtype=float))
    print(f"Saved trajectory arrays: {out_dir}")


if __name__ == "__main__":
    main()
