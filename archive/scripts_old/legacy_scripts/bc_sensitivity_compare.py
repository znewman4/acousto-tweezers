#!/usr/bin/env python3
"""
BC Sensitivity Analysis Script

Runs Greedy and Adjoint MPC controllers under multiple boundary condition (BC) variants.
Produces logs, plots, and GIFs for each variant, and a summary comparison.

Usage:
    python scripts/bc_sensitivity_compare.py --fast
    python scripts/bc_sensitivity_compare.py --variants V0,V1,V2 --T 40 --K 5 --n_iters 8 --Nx 48 --Ny 48
"""
import os
import sys
import argparse
import datetime
import numpy as np
import json
from pathlib import Path

from tweezers.control.bc_variants import BC_VARIANTS
# ...existing code: imports for Evaluator4Pucks, controllers, plotting, etc. will be added as needed...

def main():
    parser = argparse.ArgumentParser(description="BC Sensitivity Analysis: Greedy vs MPC under different boundary conditions.")
    parser.add_argument('--variants', type=str, default='V0_baseline,V1_dirichlet_top,V2_lossy',
                        help='Comma-separated list of BC variants to run (default: all)')
    parser.add_argument('--T', type=int, default=40, help='Number of executed steps (default: 40)')
    parser.add_argument('--K', type=int, default=5, help='MPC horizon (default: 5)')
    parser.add_argument('--n_iters', type=int, default=8, help='MPC inner optimisation iterations (default: 8)')
    parser.add_argument('--Nx', type=int, default=48, help='Grid size x (default: 48)')
    parser.add_argument('--Ny', type=int, default=48, help='Grid size y (default: 48)')
    parser.add_argument('--fast', action='store_true', help='Use fast preset (Nx=48, T=20, K=3, n_iters=5, variants=V0,V1,V2)')
    # ...existing code: add more CLI args as needed...
    args = parser.parse_args()

    # Fast preset
    if args.fast:
        args.Nx = args.Ny = 48
        args.T = 20
        args.K = 3
        args.n_iters = 5
        args.variants = 'V0_baseline,V1_dirichlet_top,V2_lossy'

    variants = [v.strip() for v in args.variants.split(',') if v.strip() in BC_VARIANTS]
    if not variants:
        print("No valid BC variants specified.")
        sys.exit(1)

    # Output directory
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out_root = Path(f'results/bc_sensitivity_compare/run_{now}')
    out_root.mkdir(parents=True, exist_ok=True)


    # Import required classes and functions from mpc_vs_greedy_4puck.py
    from tweezers.control import DishDomain, MediumProps, EvaluatorConfig, Control4Pucks, default_4puck_config
    from tweezers.control.evaluator_4pucks import Evaluator4Pucks
    from macro_actions_4puck import get_standard_actions_4puck, MacroActionType4Puck, MacroAction4Puck, apply_macro_action_4puck
    from acousto.force import ParticleProps, gorkov_potential_and_force_2d, bilinear_sample_vec
    import numpy as np
    import time
    import csv

    # --- Helper functions (copied/minimally adapted from mpc_vs_greedy_4puck.py) ---
    class Config:
        # ...existing code...
        def __init__(self, **kwargs):
            # Set defaults
            self.Lx = 2.0e-3
            self.Ly = 2.0e-3
            self.Nx = kwargs.get('Nx', 48)
            self.Ny = kwargs.get('Ny', 48)
            self.f = 2.0e6
            self.c0 = 1500.0
            self.rho0 = 1000.0
            self.loss_eta = kwargs.get('loss_eta', 1e-3)
            self.kz = 0.0
            self.coupling_alpha = 1.0
            self.sigma_x = 0.10e-3
            self.sigma_y = 0.15e-3
            self.bottom_band = 0.25e-3
            self.particle_a = 5.0e-6
            self.particle_rho_p = 1050.0
            self.particle_c_p = 2350.0
            self.dt = 5e-3
            self.viscosity = 1e-3
            self.alpha_g = 2e3
            self.max_step = 0.08e-3
            self.macro_magnitude = 0.05e-3
            self.macro_phase_step = 0.15
            self.macro_amplitude_step = 0.01
            self.w_align = 1.0
            self.w_push = 1e6
            self.w_switch = 0.05
            self.min_force_threshold = 1e-12
            self.K = 3
            self.T = kwargs.get('T', 40)
            self.n_top_actions = 5
            self.mpc_discount = 0.95
            self.cx = 1.0e-3
            self.cy = 1.1e-3
            self.R = 0.4e-3
            self.ccw = True
            self.n_waypoints = 400
            self.waypoint_tol = 0.12e-3
            self.k_radial = 2.0
            self.theta0 = 0.0
            # Allow override
            for k, v in kwargs.items():
                setattr(self, k, v)

    def create_evaluator(cfg: Config, bc_variant: dict) -> Evaluator4Pucks:
        domain = DishDomain(Lx=cfg.Lx, Ly=cfg.Ly, Nx=cfg.Nx, Ny=cfg.Ny)
        medium = MediumProps(
            f=cfg.f, c0=cfg.c0, rho0=cfg.rho0,
            loss_eta=bc_variant['loss_eta'], kz=cfg.kz, coupling_alpha=cfg.coupling_alpha
        )
        particle = ParticleProps(a=cfg.particle_a, rho_p=cfg.particle_rho_p, c_p=cfg.particle_c_p)
        ev_cfg = EvaluatorConfig(
            sigma_x=cfg.sigma_x,
            sigma_y=cfg.sigma_y,
            bottom_band=cfg.bottom_band,
            dt=cfg.dt,
            viscosity=cfg.viscosity,
            alpha_g=cfg.alpha_g,
            max_step=cfg.max_step,
            use_2d_forcing=True,
        )
        ev = Evaluator4Pucks(
            domain, medium, particle, ev_cfg,
            left_type=bc_variant['left_type'],
            right_type=bc_variant['right_type'],
            bottom_type=bc_variant['bottom_type'],
            top_type=bc_variant['top_type'],
        )
        return ev

    def generate_circle_waypoints(cfg: Config):
        waypoints = []
        for i in range(cfg.n_waypoints):
            theta = cfg.theta0 + 2 * np.pi * i / cfg.n_waypoints if cfg.ccw else cfg.theta0 - 2 * np.pi * i / cfg.n_waypoints
            x = cfg.cx + cfg.R * np.cos(theta)
            y = cfg.cy + cfg.R * np.sin(theta)
            waypoints.append((x, y))
        return waypoints

    # --- Main experiment loop ---
    print(f"Planned BC variants: {variants}")
    print(f"Output root: {out_root}")


    # --- Run Greedy controller for each BC variant ---
    for vname in variants:
        vcfg = BC_VARIANTS[vname]
        vdir = out_root / vname
        vdir.mkdir(exist_ok=True)
        print(f"[INFO] Running variant {vname}: {vcfg['description']}")

        cfg = Config(Nx=args.Nx, Ny=args.Ny, T=args.T, loss_eta=vcfg['loss_eta'])
        ev = create_evaluator(cfg, vcfg)
        waypoints = generate_circle_waypoints(cfg)
        x0 = cfg.cx + cfg.R * np.cos(cfg.theta0)
        y0 = cfg.cy + cfg.R * np.sin(cfg.theta0)
        ctrl0 = Control4Pucks(
            xA=x0 - 0.4e-3, yA=0.03e-3, vA=0.08, phiA=0.0, gateA=True,
            xB=x0 + 0.4e-3, yB=0.03e-3, vB=0.08, phiB=np.pi, gateB=True,
            xC=x0, yC=0.20e-3, vC=0.08, phiC=np.pi/2, gateC=True,
            xD=x0, yD=1.8e-3, vD=0.05, phiD=-np.pi/2, gateD=True,
        )
        ctrl0 = ev.clip_control(ctrl0)
        from mpc_vs_greedy_4puck import run_greedy, StepLog, save_steps_csv
        print("   Running Greedy controller...")
        greedy_logs, greedy_U, greedy_ctrls = run_greedy(ev, x0, y0, ctrl0, waypoints, cfg, verbose=True)
        save_steps_csv(vdir / "greedy_steps.csv", greedy_logs)
        summary = {
            'n_steps': len(greedy_logs),
            'final_waypoint': greedy_logs[-1].target_idx if greedy_logs else None,
            'mean_tracking_error_um': float(np.mean([log.tracking_error for log in greedy_logs]) * 1e6) if greedy_logs else None,
            'max_tracking_error_um': float(np.max([log.tracking_error for log in greedy_logs]) * 1e6) if greedy_logs else None,
            'mean_force_N': float(np.mean([log.Fp_mag for log in greedy_logs])) if greedy_logs else None,
            'max_force_N': float(np.max([log.Fp_mag for log in greedy_logs])) if greedy_logs else None,
        }
        with open(vdir / "greedy_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"   Done: {vname}\n")

    # --- Post-processing: aggregate and compare results ---
    import pandas as pd
    import matplotlib.pyplot as plt

    compare_summary = {}
    cte_curves = {}
    arc_curves = {}
    switch_curves = {}
    for vname in variants:
        vdir = out_root / vname
        steps_path = vdir / "greedy_steps.csv"
        if not steps_path.exists():
            print(f"[WARN] Missing steps for {vname}")
            continue
        df = pd.read_csv(steps_path)
        # Compute metrics
        mean_cte = df['cross_track_error'].abs().mean() * 1e6
        max_cte = df['cross_track_error'].abs().max() * 1e6
        # Arc-length progress: sum of stepwise displacements
        dx = df['particle_x'].diff().fillna(0)
        dy = df['particle_y'].diff().fillna(0)
        arc_progress = (dx**2 + dy**2).pow(0.5).sum() * 1e3  # mm
        # Action switches
        n_switches = df['action_switched'].sum()
        compare_summary[vname] = {
            'mean_cross_track_error_um': float(mean_cte),
            'max_cross_track_error_um': float(max_cte),
            'arc_progress_mm': float(arc_progress),
            'n_action_switches': int(n_switches),
            'n_steps': int(len(df)),
        }
        cte_curves[vname] = df['cross_track_error'].abs() * 1e6
        arc_curves[vname] = (dx**2 + dy**2).pow(0.5).cumsum() * 1e3
        switch_curves[vname] = df['action_switched'].astype(int).cumsum()

    # Save compare_summary.json
    with open(out_root / "compare_summary.json", "w") as f:
        json.dump(compare_summary, f, indent=2)
    print(f"[POST] Saved: {out_root / 'compare_summary.json'}")

    # Generate comparison plots
    plt.figure(figsize=(10, 6))
    for vname, cte in cte_curves.items():
        plt.plot(cte.values, label=f"{vname}")
    plt.xlabel("Timestep")
    plt.ylabel("Cross-track error (µm)")
    plt.title("Cross-track error vs timestep (Greedy)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_root / "compare_cte.png", dpi=120)
    plt.close()

    plt.figure(figsize=(10, 6))
    for vname, arc in arc_curves.items():
        plt.plot(arc.values, label=f"{vname}")
    plt.xlabel("Timestep")
    plt.ylabel("Arc-length progress (mm)")
    plt.title("Arc-length progress vs timestep (Greedy)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_root / "compare_arc.png", dpi=120)
    plt.close()

    print(f"[POST] Saved: {out_root / 'compare_cte.png'} and compare_arc.png")

if __name__ == '__main__':
    main()
