# demo_helmholtz_2d.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from acousto.solvers import solve_helmholtz_2d_neumann_velocity
from acousto.analysis import find_traps_from_force


def main() -> None:

    results_dir = Path("results") / "fields"
    results_dir.mkdir(parents=True, exist_ok=True)


    # Default geometry + medium (water-ish)
    Lx = 2e-3
    Ly = 0.5e-3
    Nx = 200
    Ny = 80
    f = 2e6
    c0 = 1500.0
    rho0 = 1000.0

    def vn_patch_y(y: np.ndarray, *, y0: float, y1: float, v0: float, phase: float):
        mask = (y >= y0) & (y <= y1)
        return (v0 * mask.astype(float)) * np.exp(1j * phase)

    v0 = 1e-4
    y0, y1 = 0.15e-3, 0.35e-3

    vn_left  = lambda y: vn_patch_y(y, y0=y0, y1=y1, v0=v0, phase=0.0)
    vn_right = lambda y: vn_patch_y(y, y0=y0, y1=y1, v0=v0, phase=0.5*np.pi)

    field = solve_helmholtz_2d_neumann_velocity(
        Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny,
        f=f, c0=c0, rho0=rho0,
        vn_left=vn_left,
        vn_right=vn_right,
        vn_bottom=0.0,
        vn_top=0.0,
        loss_eta=1e-3,
    )

    abs_p = np.abs(field.p)

    plt.figure(figsize=(8, 3))
    extent = [field.x[0] * 1e3, field.x[-1] * 1e3, field.y[0] * 1e3, field.y[-1] * 1e3]
    plt.imshow(abs_p, origin="lower", aspect="auto", extent=extent)
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.title("|p(x,y)| (arb.)")
    plt.colorbar(label="|p|")
    out = results_dir / "helmholtz_2d_abs_p.png"
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()

    print(f"Saved: {out}")

    from acousto.force import ParticleProps, gorkov_potential_and_force_2d

    # Example particle props (you should use the same constructor pattern as your 1D code)
    # If your ParticleProps requires different args, mirror the 1D demo usage.

    particle = ParticleProps(
        a=5e-6,          # radius [m]
        rho_p=1050.0,    # e.g. polystyrene-ish / cell-ish density [kg/m^3]
        c_p=2350.0,      # example [m/s] (pick what you want later)
    )


    U, Fx, Fy = gorkov_potential_and_force_2d(field, particle)

    Fmag = np.sqrt(Fx**2 + Fy**2)
    print(f"U range: [{U.min():.3e}, {U.max():.3e}]")
    print(f"|F| max: {Fmag.max():.3e}, mean: {Fmag.mean():.3e}")


    traps = find_traps_from_force(
        field.x, field.y,
        U, Fx, Fy,
        max_traps=8,
        force_rel_thresh=0.02,
        border=3,
    )

    print("\nDetected traps (coarse grid candidates):")
    for t in traps:
        # stability heuristic: both eigenvalues > 0 => local minimum of U
        stable = (t.eigvals[0] > 0) and (t.eigvals[1] > 0)
        print(
            f"  (x={t.x*1e3:.3f} mm, y={t.y*1e3:.3f} mm) "
            f"U={t.U:.3e}, |F|={np.hypot(t.Fx,t.Fy):.3e}, "
            f"eig(K)={t.eigvals}, stable={stable}"
        )

    stable_traps = [t for t in traps if (t.eigvals[0] > 0) and (t.eigvals[1] > 0)]
    print(f"\nStable traps: {len(stable_traps)} / {len(traps)}")


    # Potential heatmap
    plt.figure(figsize=(8, 3))
    plt.imshow(U, origin="lower", aspect="auto", extent=extent)
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.title("Gor'kov potential U(x,y) (arb.)")
    plt.colorbar(label="U")
    outU = results_dir / "gorkov_2d_potential.png"

    if stable_traps:
        xs_tr = [t.x * 1e3 for t in stable_traps]
        ys_tr = [t.y * 1e3 for t in stable_traps]

        plt.scatter(xs_tr, ys_tr, s=40, marker="x")
        
    plt.tight_layout()
    plt.savefig(outU, dpi=200)
    plt.close()
    print(f"Saved: {outU}")

    # Force quiver (downsample for readability)
    step_x = max(1, Nx // 40)
    step_y = max(1, Ny // 20)

    Xg, Yg = np.meshgrid(field.x * 1e3, field.y * 1e3)  # mm grids
    plt.figure(figsize=(8, 3))
    plt.imshow(U, origin="lower", aspect="auto", extent=extent)
    plt.quiver(
        Xg[::step_y, ::step_x],
        Yg[::step_y, ::step_x],
        Fx[::step_y, ::step_x],
        Fy[::step_y, ::step_x],
        scale=float(Fmag.max()) * 30.0,
        width=0.0025,
    )
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.title("Radiation force F = -∇U (quiver on U)")
    outF = results_dir / "gorkov_2d_force_quiver.png"
    plt.tight_layout()
    plt.savefig(outF, dpi=200)
    plt.close()
    print(f"Saved: {outF}")


    from acousto.dynamics import simulate_overdamped_2d

    mu = 1e-3  # Pa·s water
    dx = field.x[1] - field.x[0]
    dy = field.y[1] - field.y[0]

    gamma = 6.0 * np.pi * mu * particle.a
    vmax = float(Fmag.max() / (gamma + 1e-30))  # m/s, avoid divide-by-zero

    dt = 0.2 * min(dx, dy) / (vmax + 1e-30)     # seconds
    dt = min(dt, 1e-2)   # cap at 0.01 s for demo stability
    dt = max(dt, 1e-6)   # avoid absurdly tiny dt too
    print(f"Auto dt = {dt:.3e} s (dx={dx:.3e}, dy={dy:.3e}, vmax={vmax:.3e})")

    xs, ys = simulate_overdamped_2d(
        field.x, field.y, Fx, Fy,
        x0=0.2e-3, y0=0.25e-3,
        mu=mu, a=particle.a,
        dt=dt, steps=400,
    )


    plt.figure(figsize=(8, 3))
    plt.imshow(U, origin="lower", aspect="auto", extent=extent)
    plt.plot(xs * 1e3, ys * 1e3, linewidth=2)
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.title("Overdamped trajectory on U")
    plt.colorbar(label="U")
    outT = results_dir / "trajectory_overdamped_2d.png"
    plt.tight_layout()
    plt.savefig(outT, dpi=200)
    plt.close()
    print(f"Saved: {outT}")


if __name__ == "__main__":
    main()
