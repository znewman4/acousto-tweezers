#animate_trap_steering_2d.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from acousto.solvers import solve_helmholtz_2d_dirichlet
from acousto.force import ParticleProps, gorkov_potential_and_force_2d
from acousto.dynamics import simulate_overdamped_2d


def phase_schedule(frame: int, n_frames: int, mode: str = "ramp") -> float:
    """Return Δφ for a given frame."""
    t = frame / (n_frames - 1)
    if mode == "ramp":
        return 2.0 * np.pi * t
    if mode == "sine":
        # oscillate 0 -> 2pi -> 0
        return np.pi * (1.0 - np.cos(2.0 * np.pi * t))
    raise ValueError(f"Unknown schedule mode: {mode}")


def main() -> None:
    REPO = Path(__file__).resolve().parents[1]
    results_dir = REPO / "results" / "dynamics"
    results_dir.mkdir(parents=True, exist_ok=True)


    # -----------------------
    # Domain + medium settings
    # -----------------------
    Lx = 2e-3
    Ly = 0.5e-3
    Nx = 160
    Ny = 70
    f = 2e6
    c0 = 1500.0
    rho0 = 1000.0
    loss_eta = 1e-3

    # -----------------------
    # Particle + dynamics
    # -----------------------
    particle = ParticleProps(a=5e-6, rho_p=1050.0, c_p=2350.0)
    mu = 1e-3  # Pa·s

    # Seed position (meters)
    x0 = 0.2e-3
    y0 = 0.25e-3

    # Animation controls
    n_frames = 120
    schedule = "ramp"         # "ramp" or "sine"
    substeps_per_frame = 8    # particle integration steps per frame
    steps_per_substep = 15    # integrate this many steps each substep call (cheap + stable)
    fps = 25

    # Background choice
    background = "U"          # "U" or "abs_p"

    # Drive pattern (Dirichlet BCs)
    p_left = 1.0 + 0.0j
    p_bottom = 0.0 + 0.0j
    p_top = 0.0 + 0.0j

    # Precompute grid extent once (mm)
    # We'll solve once at frame 0 just to get axes objects set up.
    dphi0 = phase_schedule(0, n_frames, schedule)
    field0 = solve_helmholtz_2d_dirichlet(
        Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny,
        f=f, c0=c0, rho0=rho0,
        p_left=p_left,
        p_right=np.exp(1j * dphi0),
        p_bottom=p_bottom,
        p_top=p_top,
        loss_eta=loss_eta,
    )
    extent = [field0.x[0] * 1e3, field0.x[-1] * 1e3, field0.y[0] * 1e3, field0.y[-1] * 1e3]

    # State (particle position + trail)
    x = float(x0)
    y = float(y0)
    trail_x: list[float] = [x]
    trail_y: list[float] = [y]

    # Figure setup
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")

    # Create initial background
    U0, Fx0, Fy0 = gorkov_potential_and_force_2d(field0, particle)
    if background == "U":
        img0 = U0
        title0 = "Gor'kov potential U"
    elif background == "abs_p":
        img0 = np.abs(field0.p)
        title0 = "|p|"
    else:
        raise ValueError("background must be 'U' or 'abs_p'")

    im = ax.imshow(img0, origin="lower", aspect="auto", extent=extent)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(background)

    (trail_line,) = ax.plot([x * 1e3], [y * 1e3], linewidth=2)
    (dot,) = ax.plot([x * 1e3], [y * 1e3], marker="o", markersize=6, linestyle="None")

    title = ax.set_title(f"{title0}  |  Δφ = {dphi0:.2f} rad")

    # Helper to choose dt from current force field (stability-ish)
    def pick_dt(field, Fx, Fy) -> float:
        dx = float(field.x[1] - field.x[0])
        dy = float(field.y[1] - field.y[0])
        gamma = 6.0 * np.pi * mu * particle.a
        Fmag = np.sqrt(Fx**2 + Fy**2)
        vmax = float(np.max(Fmag) / (gamma + 1e-30))
        # Conservative CFL-like step for advection on the grid
        return 0.25 * min(dx, dy) / (vmax + 1e-30)

    # Update function
    def update(frame: int):
        nonlocal x, y

        dphi = phase_schedule(frame, n_frames, schedule)

        # 1) Recompute steady field for current phase
        field = solve_helmholtz_2d_dirichlet(
            Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny,
            f=f, c0=c0, rho0=rho0,
            p_left=p_left,
            p_right=np.exp(1j * dphi),
            p_bottom=p_bottom,
            p_top=p_top,
            loss_eta=loss_eta,
        )

        # 2) Compute potential + force
        U, Fx, Fy = gorkov_potential_and_force_2d(field, particle)

        # 3) Advance particle in the quasi-static force field
        dt = pick_dt(field, Fx, Fy)

        for _ in range(substeps_per_frame):
            xs, ys = simulate_overdamped_2d(
                field.x, field.y, Fx, Fy,
                x0=x, y0=y,
                mu=mu, a=particle.a,
                dt=dt, steps=steps_per_substep,
            )
            x = float(xs[-1])
            y = float(ys[-1])
            trail_x.append(x)
            trail_y.append(y)

        # 4) Update visuals
        if background == "U":
            im.set_data(U)
        else:
            im.set_data(np.abs(field.p))

        dot.set_data([x * 1e3], [y * 1e3])
        trail_line.set_data(np.array(trail_x) * 1e3, np.array(trail_y) * 1e3)
        title.set_text(f"{title0}  |  Δφ = {dphi:.2f} rad")

        return im, dot, trail_line, title

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, blit=False)

    out = results_dir / "anim_trap_steering.gif"
    anim.save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
