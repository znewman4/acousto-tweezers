# demo_helmholtz_1d.py
import numpy as np
import matplotlib.pyplot as plt
import os
from acousto.force import ParticleProps, gorkov_potential_and_force_1d
from acousto.solvers.fd_helmholtz_1d import (
    solve_helmholtz_1d_dirichlet,
    analytic_standing_wave_dirichlet,
)

def main():

    import os
    os.makedirs("results", exist_ok=True)

    # --- Phase-1 demo parameters (edit freely) ---
    L = 2e-3         # 2 mm cavity
    N = 401          # grid points
    f = 2e6          # 2 MHz
    c0 = 1480.0      # m/s (water)
    rho0 = 1000.0    # kg/m^3 (water)
    loss_eta = 1e-3

    p_left = 1.0 + 0j
    p_right = 0.0 + 0j

    field = solve_helmholtz_1d_dirichlet(
        L=L, N=N, f=f, c0=c0, rho0=rho0, p_left=p_left, p_right=p_right, loss_eta=loss_eta
    )

    p_ana = analytic_standing_wave_dirichlet(
        field.x, L=L, k=field.k, p_left=p_left, p_right=p_right
    )

    # --- Particle properties (example: polystyrene in water-ish) ---
    particle = ParticleProps(
        a=5e-6,        # 5 microns radius
        rho_p=1050.0,  # kg/m^3
        c_p=2350.0,    # m/s (rough order)
    )

    U, F, v = gorkov_potential_and_force_1d(field, particle)


    os.makedirs("results", exist_ok=True)

    # 1) Pressure amplitude
    plt.figure()
    plt.plot(field.x, np.abs(field.p), label="FD |p|")
    plt.plot(field.x, np.abs(p_ana), linestyle="--", label="Analytic |p|")
    plt.xlabel("x [m]")
    plt.ylabel("|p| (arb.)")
    plt.title("1D Helmholtz (Dirichlet) — Phase 1 bootstrap")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/helmholtz_1d_pressure.png", dpi=150)
    plt.close()

    # 2) Gor'kov potential (shape matters; absolute scale depends on assumptions)
    plt.figure()
    plt.plot(field.x, U)
    plt.xlabel("x [m]")
    plt.ylabel("U(x) [J] (model)")
    plt.title("Gor'kov radiation potential U(x)")
    plt.tight_layout()
    plt.savefig("results/gorkov_1d_potential.png", dpi=150)
    plt.close()

    # 3) Radiation force
    plt.figure()
    plt.plot(field.x, F)
    plt.xlabel("x [m]")
    plt.ylabel("F(x) [N] (model)")
    plt.title("Radiation force F(x) = -dU/dx")
    plt.tight_layout()
    plt.savefig("results/gorkov_1d_force.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
