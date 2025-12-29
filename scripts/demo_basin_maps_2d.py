from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from acousto.solvers import solve_helmholtz_2d_neumann_velocity
from acousto.force import ParticleProps, gorkov_potential_and_force_2d
from acousto.analysis import find_traps_from_force
from acousto.dynamics import simulate_overdamped_2d


def vn_patch_y(y: np.ndarray, *, y0: float, y1: float, v0: float, phase: float):
    """Patch actuation on a vertical wall: vn(y)=v0*exp(i*phase) on [y0,y1], else 0."""
    mask = (y >= y0) & (y <= y1)
    return (v0 * mask.astype(float)) * np.exp(1j * phase)


def classify_endpoint(xs, ys, traps, tol=2e-5):
    """
    Classify trajectory endpoint by nearest stable trap.
    Returns trap index or -1 if none close.
    """
    xe, ye = xs[-1], ys[-1]
    for i, t in enumerate(traps):
        if np.hypot(xe - t.x, ye - t.y) < tol:
            return i
    return -1


def main() -> None:
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Geometry + medium
    Lx = 2e-3
    Ly = 0.5e-3
    Nx = 160
    Ny = 70
    f = 2e6
    c0 = 1500.0
    rho0 = 1000.0
    loss_eta = 1e-3

    # Particle + fluid
    particle = ParticleProps(a=5e-6, rho_p=1050.0, c_p=2350.0)
    mu = 1e-3  # Pa·s

    # Patch actuation parameters
    v0 = 1e-4
    y0, y1 = 0.15e-3, 0.35e-3

    # Phases to visualise
    phase_cases = {
        "phi0": 0.0,
        "phi_pi2": 0.5 * np.pi,
        "phi_pi": np.pi,
    }

    for tag, dphi in phase_cases.items():
        print(f"\nRunning basin + streamlines for {tag}")

        vn_left = lambda y: vn_patch_y(y, y0=y0, y1=y1, v0=v0, phase=0.0)
        vn_right = lambda y: vn_patch_y(y, y0=y0, y1=y1, v0=v0, phase=dphi)

        field = solve_helmholtz_2d_neumann_velocity(
            Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny,
            f=f, c0=c0, rho0=rho0,
            vn_left=vn_left,
            vn_right=vn_right,
            vn_bottom=0.0,
            vn_top=0.0,
            loss_eta=loss_eta,
        )

        U, Fx, Fy = gorkov_potential_and_force_2d(field, particle)

        traps = find_traps_from_force(
            field.x, field.y,
            U, Fx, Fy,
            max_traps=12,
            force_rel_thresh=0.02,
            border=3,
        )
        stable_traps = [t for t in traps if (t.eigvals[0] > 0) and (t.eigvals[1] > 0)]

        extent = [
            field.x[0] * 1e3, field.x[-1] * 1e3,
            field.y[0] * 1e3, field.y[-1] * 1e3,
        ]

        # -------------------------
        # 1) Force streamlines plot
        # -------------------------
        Xg, Yg = np.meshgrid(field.x * 1e3, field.y * 1e3)  # mm

        plt.figure(figsize=(8, 3))
        plt.imshow(U, origin="lower", aspect="auto", extent=extent)

        # streamplot is happier if we downsample everything consistently
        sx = max(1, Nx // 80)
        sy = max(1, Ny // 40)
        plt.streamplot(
            Xg[::sy, ::sx],
            Yg[::sy, ::sx],
            Fx[::sy, ::sx],
            Fy[::sy, ::sx],
            color="k",
            density=1.1,
            linewidth=0.7,
            arrowsize=0.8,
        )

        if stable_traps:
            xs = [t.x * 1e3 for t in stable_traps]
            ys = [t.y * 1e3 for t in stable_traps]
            plt.scatter(xs, ys, s=40, marker="x")

        plt.xlabel("x (mm)")
        plt.ylabel("y (mm)")
        plt.title(f"Force streamlines on U ({tag}) — Neumann patch drive")
        plt.colorbar(label="U")
        out = results_dir / f"streamlines_{tag}.png"
        plt.tight_layout()
        plt.savefig(out, dpi=200)
        plt.close()
        print(f"Saved: {out}")

        # -------------------------
        # 2) Basin of attraction map
        # -------------------------
        nx_seed = 25
        ny_seed = 12

        xs0 = np.linspace(field.x[0], field.x[-1], nx_seed)
        ys0 = np.linspace(field.y[0], field.y[-1], ny_seed)

        basin = -np.ones((ny_seed, nx_seed), dtype=int)

        gamma = 6.0 * np.pi * mu * particle.a
        dx = field.x[1] - field.x[0]
        dy = field.y[1] - field.y[0]
        Fmag = np.sqrt(Fx**2 + Fy**2)
        vmax = float(np.max(Fmag) / (gamma + 1e-30))

        # dt heuristic + clamp (avoid absurd dt if forces are tiny)
        dt = 0.25 * min(dx, dy) / (vmax + 1e-30)
        dt = min(dt, 1e-2)
        dt = max(dt, 1e-6)

        for j, y_init in enumerate(ys0):
            for i, x_init in enumerate(xs0):
                xs, ys = simulate_overdamped_2d(
                    field.x, field.y, Fx, Fy,
                    x0=x_init, y0=y_init,
                    mu=mu, a=particle.a,
                    dt=dt, steps=500,
                )
                basin[j, i] = classify_endpoint(xs, ys, stable_traps)

        plt.figure(figsize=(8, 3))
        plt.imshow(
            basin,
            origin="lower",
            aspect="auto",
            extent=extent,
            interpolation="nearest",
        )
        plt.xlabel("x (mm)")
        plt.ylabel("y (mm)")
        plt.title(f"Basin of attraction ({tag}) — Neumann patch drive")
        plt.colorbar(label="trap index (-1 = escape)")
        out = results_dir / f"basin_{tag}.png"
        plt.tight_layout()
        plt.savefig(out, dpi=200)
        plt.close()
        print(f"Saved: {out}")

        # Optional: print trap summary
        print(f"  Stable traps: {len(stable_traps)}")
        for t in stable_traps[:6]:
            print(f"    x={t.x*1e3:.3f} mm, y={t.y*1e3:.3f} mm, minEig={np.min(t.eigvals):.3e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
