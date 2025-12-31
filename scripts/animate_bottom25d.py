# scripts/animate_bottom_drive_25d_3d.py
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

from pathlib import Path
import numpy as np
import imageio.v2 as imageio

from acousto.solvers import solve_helmholtz_2d_forced_25d
from acousto.force import ParticleProps, gorkov_potential_and_force_2d
from acousto.analysis import find_traps_from_force

from tweezers.viz.render_3d import (
    Cylinder2D,
    pick_best_stable_trap,
    render_gorkov_landscape_frame_3d,
    png_is_blankish,
    classify_trap,
)


def gaussian_puck(X: np.ndarray, Y: np.ndarray, x0: float, y0: float, sigma: float) -> np.ndarray:
    return np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2.0 * sigma * sigma))


def main() -> None:
    REPO = Path(__file__).resolve().parents[1]
    RESULTS = REPO / "results"

    frames_dir = RESULTS / "frames_bottomdrive_25d_3d"
    out_dir = RESULTS / "gorkov"
    frames_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Domain
    # -----------------------------
    Lx = 2e-3
    Ly = 2e-3
    Nx = 180
    Ny = 180

    # -----------------------------
    # Medium
    # -----------------------------
    f = 2e6
    c0 = 1500.0
    rho0 = 1000.0
    loss_eta = 1e-3

    # 2.5D knobs
    kz = 0.0               # start at 0 (pure 2D). You can sweep later.
    coupling_alpha = 1.0   # calibration knob (dimensionless)

    # -----------------------------
    # Actuator (bottom drive) model
    # -----------------------------
    v0 = 2e-6          # m/s peak in the puck region (tune)
    sigma = 0.10e-3    # meters: gaussian width ~ "actuator footprint"
    cyl_r_mm = (2.0 * sigma) * 1e3  # choose radius ~ 2*sigma for visual
    y_mid = 0.5 * Ly

    # Motion: start near ends, move towards each other
    nframes = 180
    fps = 30
    margin = 0.12e-3
    xA_path = np.linspace(margin, 0.5 * Lx - margin, nframes)
    xB_path = np.linspace(Lx - margin, 0.5 * Lx + margin, nframes)

    # Particle props (for Gor'kov)
    particle = ParticleProps(a=5e-6, rho_p=1050.0, c_p=2350.0)

    # Diagnostics controls
    print_first = 10
    print_every = 15

    track_xy_mm: list[tuple[float, float]] = []
    frame_paths: list[Path] = []

    # Pre-build meshgrids for vb evaluation inside diagnostics
    x = np.linspace(0.0, Lx, Nx)
    y = np.linspace(0.0, Ly, Ny)
    X, Y = np.meshgrid(x, y)

    for k in range(nframes):
        xA = float(xA_path[k])
        xB = float(xB_path[k])
        t = k / (nframes - 1)

        # Define vb(x,y) for this frame (two pucks)
        vb_xy = (
            v0 * gaussian_puck(X, Y, xA, y_mid, sigma)
            + v0 * gaussian_puck(X, Y, xB, y_mid, sigma)
        ).astype(np.complex128)

        # Basic actuation diagnostics
        vb_abs = np.abs(vb_xy)
        vb_max = float(vb_abs.max())
        vb_mean = float(vb_abs.mean())
        vb_rms = float(np.sqrt(np.mean(vb_abs**2)))
        active_frac = float(np.mean(vb_abs > (0.05 * vb_max)))  # fraction above 5% peak

        do_print = (k < print_first) or (k % print_every == 0)

        if do_print:
            print(
                f"\n[{k:04d}/{nframes-1:04d}] t={t:.3f} "
                f"| puckA=({xA*1e3:.3f}mm,{y_mid*1e3:.3f}mm) "
                f"puckB=({xB*1e3:.3f}mm,{y_mid*1e3:.3f}mm)"
            )
            print(
                f"  vb: max={vb_max:.3e} m/s  mean={vb_mean:.3e}  rms={vb_rms:.3e}  active_frac(>5%)={active_frac:.3f}"
            )

        # Solve forced 2.5D Helmholtz
        field = solve_helmholtz_2d_forced_25d(
            Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny,
            f=f, c0=c0, rho0=rho0,
            left_type="neumann", right_type="neumann",
            bottom_type="neumann", top_type="neumann",
            left=0.0, right=0.0, bottom=0.0, top=0.0,
            kz=kz,
            vb=vb_xy,                 # array works directly
            coupling_alpha=coupling_alpha,
            loss_eta=loss_eta,
        )

        # Field diagnostics
        p = field.p
        p_abs = np.abs(p)
        p_max = float(p_abs.max())
        p_mean = float(p_abs.mean())
        p_finite = bool(np.isfinite(p).all())

        if do_print:
            print(f"  field: |p| max={p_max:.3e}  mean={p_mean:.3e}  finite={p_finite}")

        # Gor'kov
        U, Fx, Fy = gorkov_potential_and_force_2d(field, particle)

        U_min = float(np.min(U))
        U_max = float(np.max(U))
        U_finite = bool(np.isfinite(U).all())


        Fmag = np.sqrt(Fx**2 + Fy**2)
        F_max = float(np.max(Fmag))
        F_mean = float(np.mean(Fmag))

        if do_print:
            print(f"  gorkov: U[min,max]=[{U_min:.3e},{U_max:.3e}] finite={U_finite}")
            print(f"         |F| max={F_max:.3e}  mean={F_mean:.3e}")

        # Traps
        traps = find_traps_from_force(
            field.x, field.y,
            U, Fx, Fy,
            max_traps=12,
            force_rel_thresh=0.02,
            border=3,
        )
        n_min = sum(1 for tr in traps if classify_trap(tr.eigvals) == "min")
        n_sad = sum(1 for tr in traps if classify_trap(tr.eigvals) == "saddle")
        n_max = sum(1 for tr in traps if classify_trap(tr.eigvals) == "max")

        if do_print:
            print(f"  traps: total={len(traps)}  mins={n_min}  saddles={n_sad}  maxs={n_max}")
            if traps:
                # show first 3 by |F|
                traps_sorted = sorted(traps, key=lambda tr: (tr.Fx**2 + tr.Fy**2))
                for ii, tr in enumerate(traps_sorted[:3]):
                    tt = classify_trap(tr.eigvals)
                    f_here = (tr.Fx**2 + tr.Fy**2) ** 0.5
                    print(
                        f"    trap[{ii}] {tt:6s}: (x,y)=({tr.x*1e3:.3f}mm,{tr.y*1e3:.3f}mm) "
                        f"U={tr.U:.3e} |F|={f_here:.3e} eig={np.array(tr.eigvals)}"
                    )

        # Track best stable trap
        best = pick_best_stable_trap(traps)
        if best is not None:
            track_xy_mm.append((best.x * 1e3, best.y * 1e3))

        # Cylinder overlays
        cylinders = [
            Cylinder2D(x_mm=xA * 1e3, y_mm=y_mid * 1e3, r_mm=cyl_r_mm, alpha=0.22, edge_alpha=0.60),
            Cylinder2D(x_mm=xB * 1e3, y_mm=y_mid * 1e3, r_mm=cyl_r_mm, alpha=0.22, edge_alpha=0.60),
        ]

        # Render frame
        out_png = frames_dir / f"frame_{k:04d}.png"
        render_gorkov_landscape_frame_3d(
            out_png=out_png,
            x_mm=field.x * 1e3,
            y_mm=field.y * 1e3,
            U=U,
            traps=traps,
            track_xy_mm=track_xy_mm,
            cylinders=cylinders,
            surface_stride=3,
            elev=30.0,
            azim=-60.0,
        )
        frame_paths.append(out_png)

        # Saved-image diagnostics
        if do_print:
            blankish, std, mean, fsize = png_is_blankish(out_png)
            if blankish:
                print(f"  png: WARNING blank-ish (std={std:.3f}, mean={mean:.1f}, size={fsize}) -> {out_png.name}")
            else:
                print(f"  png: ok (std={std:.3f}, mean={mean:.1f}, size={fsize})")

        if k == 0 or (k + 1) % 25 == 0:
            print(f"Rendered {k+1}/{nframes}: {out_png.name}")

    # Build GIF
    gif_path = out_dir / "gorkov_landscape_bottomdrive_25d_cylinders.gif"
    duration = 1.0 / fps
    images = [imageio.imread(p) for p in frame_paths]
    imageio.mimsave(gif_path, images, duration=duration, loop=0, subrectangles=False)
    print(f"\nSaved GIF: {gif_path}")


if __name__ == "__main__":
    main()
