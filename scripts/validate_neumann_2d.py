from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from acousto.solvers import solve_helmholtz_2d_neumann_velocity
from acousto.force import ParticleProps, gorkov_potential_and_force_2d
from acousto.analysis import find_traps_from_force


def vn_patch_y(y: np.ndarray, *, y0: float, y1: float, v0: float, phase: float):
    """Patch actuation on a vertical wall: vn(y)=v0*exp(i*phase) on [y0,y1], else 0."""
    mask = (y >= y0) & (y <= y1)
    return (v0 * mask.astype(float)) * np.exp(1j * phase)


def solve_case(*, tag: str, Nx: int, Ny: int, vn_left, vn_right, vn_bottom=0.0, vn_top=0.0):
    # Geometry + medium
    Lx = 2e-3
    Ly = 0.5e-3
    f = 2e6
    c0 = 1500.0
    rho0 = 1000.0
    loss_eta = 1e-3

    # Particle (same as demos)
    particle = ParticleProps(a=5e-6, rho_p=1050.0, c_p=2350.0)

    field = solve_helmholtz_2d_neumann_velocity(
        Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny,
        f=f, c0=c0, rho0=rho0,
        vn_left=vn_left,
        vn_right=vn_right,
        vn_bottom=vn_bottom,
        vn_top=vn_top,
        loss_eta=loss_eta,
    )

    U, Fx, Fy = gorkov_potential_and_force_2d(field, particle)

    traps = find_traps_from_force(
        field.x, field.y, U, Fx, Fy,
        max_traps=12,
        force_rel_thresh=0.02,
        border=3,
    )
    stable = [t for t in traps if (t.eigvals[0] > 0) and (t.eigvals[1] > 0)]
    best = None
    if stable:
        best = max(stable, key=lambda t: float(np.min(t.eigvals)))

    # Metrics
    pmax = float(np.max(np.abs(field.p)))
    Umax = float(np.max(np.abs(U)))
    Fmag = np.sqrt(Fx**2 + Fy**2)
    Fmax = float(np.max(Fmag))

    return field, U, Fx, Fy, stable, best, (pmax, Umax, Fmax)


def save_plots(results: Path, tag: str, field, U, stable):
    extent = [field.x[0]*1e3, field.x[-1]*1e3, field.y[0]*1e3, field.y[-1]*1e3]

    # |p|
    plt.figure(figsize=(8, 3))
    plt.imshow(np.abs(field.p), origin="lower", aspect="auto", extent=extent)
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.title(f"|p| ({tag})")
    plt.colorbar(label="|p|")
    plt.tight_layout()
    plt.savefig(results / "validate" / f"validate_abs_p_{tag}.png", dpi=200)
    plt.close()

    # U + traps
    plt.figure(figsize=(8, 3))
    plt.imshow(U, origin="lower", aspect="auto", extent=extent)
    if stable:
        xs = [t.x*1e3 for t in stable]
        ys = [t.y*1e3 for t in stable]
        plt.scatter(xs, ys, s=40, marker="x")
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.title(f"U + stable traps ({tag})")
    plt.colorbar(label="U")
    plt.tight_layout()
    plt.savefig(results / "validate" / f"validate_U_{tag}.png", dpi=200)
    plt.close()


def main():
    results = Path("results")
    results.mkdir(parents=True, exist_ok=True)

    print("\n=== Neumann/velocity validation ===")

    # -------------------
    # A) Zero-drive test
    # -------------------
    field, U, Fx, Fy, stable, best, (pmax, Umax, Fmax) = solve_case(
        tag="zero_drive",
        Nx=120, Ny=60,
        vn_left=0.0, vn_right=0.0,
        vn_bottom=0.0, vn_top=0.0,
    )
    print("\n[A] Zero-drive")
    print(f"  max|p| = {pmax:.3e}")
    print(f"  max|U| = {Umax:.3e}")
    print(f"  max|F| = {Fmax:.3e}")
    print(f"  stable traps = {len(stable)}")
    save_plots(results, "zero_drive", field, U, stable)

    # -------------------
    # B) Symmetry test
    # -------------------
    v0 = 1e-4
    y0, y1 = 0.15e-3, 0.35e-3
    vnL = lambda y: vn_patch_y(y, y0=y0, y1=y1, v0=v0, phase=0.0)
    vnR = lambda y: vn_patch_y(y, y0=y0, y1=y1, v0=v0, phase=0.0)

    field, U, Fx, Fy, stable, best, (pmax, Umax, Fmax) = solve_case(
        tag="symmetry",
        Nx=120, Ny=60,
        vn_left=vnL, vn_right=vnR,
    )
    print("\n[B] Symmetry (left == right, same phase)")
    print(f"  max|p| = {pmax:.3e}")
    print(f"  max|U| = {Umax:.3e}")
    print(f"  max|F| = {Fmax:.3e}")
    print(f"  stable traps = {len(stable)}")
    if best:
        print(f"  best trap: x={best.x*1e3:.3f} mm, y={best.y*1e3:.3f} mm, minEig={np.min(best.eigvals):.3e}")
    save_plots(results, "symmetry", field, U, stable)

    # -------------------
    # C) Grid refinement
    # -------------------
    dphi = 0.5*np.pi
    vnL = lambda y: vn_patch_y(y, y0=y0, y1=y1, v0=v0, phase=0.0)
    vnR = lambda y: vn_patch_y(y, y0=y0, y1=y1, v0=v0, phase=dphi)

    grids = [(80, 40), (120, 60), (160, 80)]
    print("\n[C] Grid refinement (phase offset pi/2)")
    for (Nx, Ny) in grids:
        field, U, Fx, Fy, stable, best, (pmax, Umax, Fmax) = solve_case(
            tag=f"ref_{Nx}x{Ny}",
            Nx=Nx, Ny=Ny,
            vn_left=vnL, vn_right=vnR,
        )
        if best:
            print(f"  {Nx:3d}x{Ny:3d}: best x={best.x*1e3:.3f} mm, y={best.y*1e3:.3f} mm, minEig={np.min(best.eigvals):.3e}")
        else:
            print(f"  {Nx:3d}x{Ny:3d}: no stable trap")
        save_plots(results, f"ref_{Nx}x{Ny}", field, U, stable)

    print("\nSaved plots: results/validate_*.png")


if __name__ == "__main__":
    main()
