# demo_phase_sweep_2d.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from acousto.solvers import solve_helmholtz_2d_neumann_velocity
from acousto.force import ParticleProps, gorkov_potential_and_force_2d
from acousto.analysis import find_traps_from_force


def pick_best_stable_trap(stable_traps):
    """
    Pick a single 'best' trap for tracking vs phase.

    Heuristic: choose the trap with the largest minimum stiffness eigenvalue
    (i.e. strongest weakest-direction confinement).
    """
    # Each trap has eigvals sorted ascending from np.linalg.eigh
    best = max(stable_traps, key=lambda t: float(np.min(t.eigvals)))
    return best


def save_snapshot(results_dir: Path, tag: str, field, U, stable_traps, extent):
    """Save a U snapshot with stable trap markers."""
    plt.figure(figsize=(8, 3))
    plt.imshow(U, origin="lower", aspect="auto", extent=extent)
    if stable_traps:
        xs = [t.x * 1e3 for t in stable_traps]
        ys = [t.y * 1e3 for t in stable_traps]
        plt.scatter(xs, ys, s=40, marker="x")
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.title(f"Gor'kov potential U(x,y) @ {tag}")
    plt.colorbar(label="U")
    out = results_dir / f"phase_sweep_snapshot_U_{tag}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Saved: {out}")

    # Optional: also save |p|
    abs_p = np.abs(field.p)
    plt.figure(figsize=(8, 3))
    plt.imshow(abs_p, origin="lower", aspect="auto", extent=extent)
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.title(f"|p(x,y)| @ {tag}")
    plt.colorbar(label="|p|")
    outp = results_dir / f"phase_sweep_snapshot_abs_p_{tag}.png"
    plt.tight_layout()
    plt.savefig(outp, dpi=200)
    plt.close()
    print(f"Saved: {outp}")


def main() -> None:
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # --- Sweep settings (keep smaller grid for speed) ---
    Lx = 2e-3
    Ly = 0.5e-3
    Nx = 140
    Ny = 60
    f = 2e6
    c0 = 1500.0
    rho0 = 1000.0
    loss_eta = 1e-3

    # Trap detection settings
    max_traps = 12
    force_rel_thresh = 0.02
    border = 3

    # Particle (same as your demo)
    particle = ParticleProps(a=5e-6, rho_p=1050.0, c_p=2350.0)

    # Phase sweep
    nphi = 61
    phis = np.linspace(0.0, 2.0 * np.pi, nphi)

    # Storage (NaN means no stable trap found)
    x_best = np.full(nphi, np.nan)
    y_best = np.full(nphi, np.nan)
    lam1 = np.full(nphi, np.nan)
    lam2 = np.full(nphi, np.nan)
    U_best = np.full(nphi, np.nan)
    has_stable = np.zeros(nphi, dtype=bool)

    # For snapshots
    snapshot_phis = {
        "phi0": 0.0,
        "phi_pi2": 0.5 * np.pi,
        "phi_pi": np.pi,
    }
    snapshot_done = {k: False for k in snapshot_phis}

    # We'll reuse extent once x,y exist
    extent = None

    print("Running phase sweep...")
    for k, dphi in enumerate(phis):
     # Boundary actuation: two driven walls with phase difference

        v0 = 1e-4  # m/s (start here; we can scale later)
        vn_left = v0
        vn_right = v0 * np.exp(1j * dphi)

        field = solve_helmholtz_2d_neumann_velocity(
            Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny,
            f=f, c0=c0, rho0=rho0,
            vn_left=vn_left,
            vn_right=vn_right,
            vn_bottom=0.0,
            vn_top=0.0,
            loss_eta=loss_eta,
        )

        if extent is None:
            extent = [field.x[0]*1e3, field.x[-1]*1e3, field.y[0]*1e3, field.y[-1]*1e3]

        U, Fx, Fy = gorkov_potential_and_force_2d(field, particle)

        traps = find_traps_from_force(
            field.x, field.y,
            U, Fx, Fy,
            max_traps=max_traps,
            force_rel_thresh=force_rel_thresh,
            border=border,
        )
        stable_traps = [t for t in traps if (t.eigvals[0] > 0) and (t.eigvals[1] > 0)]

        if stable_traps:
            has_stable[k] = True
            best = pick_best_stable_trap(stable_traps)

            x_best[k] = best.x
            y_best[k] = best.y
            lam1[k], lam2[k] = best.eigvals[0], best.eigvals[1]
            U_best[k] = best.U

        # Save snapshots at representative phases (nearest match)
        for tag, target in snapshot_phis.items():
            if not snapshot_done[tag] and abs(dphi - target) <= (phis[1] - phis[0]) / 2:
                save_snapshot(results_dir, tag, field, U, stable_traps, extent)
                snapshot_done[tag] = True

        if (k + 1) % 10 == 0 or (k + 1) == nphi:
            print(f"  {k+1:3d}/{nphi} done")

    # --- Plot 1: Stable trap existence vs phase ---
    plt.figure(figsize=(8, 2.5))
    plt.plot(phis, has_stable.astype(int), marker="o", linewidth=1)
    plt.yticks([0, 1], ["no stable trap", "stable trap"])
    plt.xlabel("phase difference Δφ (rad)")
    plt.title("Stable trap existence vs phase")
    out1 = results_dir / "phase_sweep_exists.png"
    plt.tight_layout()
    plt.savefig(out1, dpi=200)
    plt.close()
    print(f"Saved: {out1}")

    # --- Plot 2: Best trap position vs phase ---
    plt.figure(figsize=(8, 3))
    plt.plot(phis, x_best * 1e3, marker="o", linewidth=1, label="x* (mm)")
    plt.plot(phis, y_best * 1e3, marker="o", linewidth=1, label="y* (mm)")
    plt.xlabel("phase difference Δφ (rad)")
    plt.ylabel("position (mm)")
    plt.title("Best stable trap position vs phase")
    plt.legend()
    out2 = results_dir / "phase_sweep_trap_position.png"
    plt.tight_layout()
    plt.savefig(out2, dpi=200)
    plt.close()
    print(f"Saved: {out2}")

    # --- Plot 3: Stiffness eigenvalues vs phase ---
    plt.figure(figsize=(8, 3))
    plt.plot(phis, lam1, marker="o", linewidth=1, label="λ1")
    plt.plot(phis, lam2, marker="o", linewidth=1, label="λ2")
    plt.xlabel("phase difference Δφ (rad)")
    plt.ylabel("stiffness eigenvalues (arb. units)")
    plt.title("Best stable trap stiffness vs phase")
    plt.legend()
    out3 = results_dir / "phase_sweep_stiffness.png"
    plt.tight_layout()
    plt.savefig(out3, dpi=200)
    plt.close()
    print(f"Saved: {out3}")

    # --- Plot 4: Potential at best trap vs phase (optional but informative) ---
    plt.figure(figsize=(8, 3))
    plt.plot(phis, U_best, marker="o", linewidth=1)
    plt.xlabel("phase difference Δφ (rad)")
    plt.ylabel("U at best stable trap (arb.)")
    plt.title("Best stable trap potential vs phase")
    out4 = results_dir / "phase_sweep_Ubest.png"
    plt.tight_layout()
    plt.savefig(out4, dpi=200)
    plt.close()
    print(f"Saved: {out4}")

    # Summary
    n_ok = int(np.sum(has_stable))
    print("\nSummary:")
    print(f"  Stable trap found at {n_ok}/{nphi} phase values")
    if n_ok > 0:
        # Choose phase with strongest weakest-direction stiffness
        idx = np.nanargmax(np.nanmin(np.vstack([lam1, lam2]), axis=0))
        print(f"  Strongest (max min-eig) phase: Δφ = {phis[idx]:.3f} rad")
        print(f"    trap at x={x_best[idx]*1e3:.3f} mm, y={y_best[idx]*1e3:.3f} mm")
        print(f"    eig(K) = [{lam1[idx]:.3e}, {lam2[idx]:.3e}]")
    else:
        print("  No stable traps found. Next escalation: more physical BCs (Neumann hard walls) or different actuation pattern.")


if __name__ == "__main__":
    main()
