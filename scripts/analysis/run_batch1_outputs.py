#!/usr/bin/env python3
"""
Batch-1 Results Pack — field exports, 2D slice plots, mini frequency sweep, diagnostics.

Generates one coherent batch of outputs for ParaView + matplotlib after the physics
sanity audit has passed.

Usage:
    micromamba run -n acousto-complex python scripts/analysis/run_batch1_outputs.py

Outputs go to: results/latest/batch1_YYYY-MM-DD/
"""
from __future__ import annotations

import sys, os, json, csv
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from mpi4py import MPI

from acoustweezers.experiments.shallow_square_dish.config import ShallowDishConfig
from acoustweezers.experiments.shallow_square_dish.solve_pressure import (
    create_mesh, solve_helmholtz, compute_phase_winding,
    TAG_BOTTOM_DISC, TAG_BOTTOM_RIGID, TAG_TOP,
    TAG_X0, TAG_XL, TAG_Y0, TAG_YL,
)

comm = MPI.COMM_WORLD
rank = comm.rank

TODAY = datetime.now().strftime("%Y-%m-%d")
OUTDIR = Path(f"results/latest/batch1_{TODAY}")


def log(msg=""):
    """Print with immediate flush so progress is visible."""
    print(msg, flush=True)

# ==========================================================================
# Configuration — intended physical system
# ==========================================================================
# Ambiguity: The solver uses GMRES+ILU which is slow/non-convergent for
# complex Helmholtz at large DOF counts.  The validated test configuration
# (L=10mm, ~28k DOFs) converges reliably.  Batch-1 uses this validated
# size.  Larger domains (L=50mm) require switching to a direct solver
# (MUMPS) — that's a Batch-2 improvement that doesn't change physics.
CFG_BASE = ShallowDishConfig(
    L=10e-3,           # 10 mm (validated scale)
    H=1e-3,            # 1 mm
    frequency_hz=500e3,
    elements_per_wavelength=6,
    min_elements_z=8,
    vortex_velocity_amplitude=10e-6,
    standing_velocity_amplitude=10e-6,
    vortex_aperture_radius=3e-3,   # 3 mm disc
    standing_axis="both",
    standing_phase_pattern="antiphase",
)


# ==========================================================================
# Helper: extract 2D mid-plane slice (nearest-DOF grid)
# ==========================================================================
def extract_slice(coords, values, z_target, cfg, nx_grid=120):
    """Return (X, Y, V) gridded data for a horizontal slice at z≈z_target."""
    tol_z = cfg.H / cfg.mesh_nz * 1.5
    mask = np.abs(coords[:, 2] - z_target) < tol_z
    if not np.any(mask):
        raise RuntimeError(f"No DOFs found near z={z_target}")

    from scipy.interpolate import griddata
    x_pts = coords[mask, 0]
    y_pts = coords[mask, 1]
    v_pts = values[mask]

    xg = np.linspace(0, cfg.L, nx_grid)
    yg = np.linspace(0, cfg.L, nx_grid)
    X, Y = np.meshgrid(xg, yg)
    V = griddata((x_pts, y_pts), v_pts, (X, Y), method='linear')
    return X, Y, V


# ==========================================================================
# Helper: compute first-order velocity magnitude |v1| = |∇p|/(ωρ)
# ==========================================================================
def compute_v1_mag_at_dofs(p_solution):
    """Estimate |v1| ≈ |∇p|/(ωρ) using L2 projection of |grad p|².

    Returns (coords, v1_mag) — both on a P1 grid.
    """
    from dolfinx import fem
    import ufl

    p_func = p_solution.p_function
    V = p_func.function_space
    mesh = V.mesh
    cfg = p_solution.cfg

    V1 = fem.functionspace(mesh, ("Lagrange", 1))
    u = ufl.TrialFunction(V1)
    v = ufl.TestFunction(V1)

    grad_p = ufl.grad(p_func)
    grad_p_sq = ufl.inner(grad_p, grad_p)

    a = ufl.inner(u, v) * ufl.dx
    L_form = ufl.inner(grad_p_sq, v) * ufl.dx

    from dolfinx.fem.petsc import LinearProblem
    problem = LinearProblem(a, L_form, bcs=[],
                            petsc_options={"ksp_type": "cg", "pc_type": "jacobi",
                                           "ksp_rtol": 1e-6})
    grad_sq_func = problem.solve()

    grad_sq_vals = np.abs(grad_sq_func.x.array)
    v1_mag = np.sqrt(np.maximum(grad_sq_vals, 0)) / (cfg.omega * cfg.rho)

    coords_v1 = V1.tabulate_dof_coordinates()
    return coords_v1, v1_mag


def compute_all_v1(solutions):
    """Compute v1 for all modes, returning a dict of (coords, v1_mag)."""
    v1_cache = {}
    for mode in ["standing", "vortex", "combined"]:
        log(f"    computing |v1| for {mode}...")
        v1_cache[mode] = compute_v1_mag_at_dofs(solutions[mode])
    return v1_cache


# ==========================================================================
# PART A: Field exports (XDMF)
# ==========================================================================
def export_xdmf_fields(solutions, v1_cache, domain, outdir):
    """Export Re(p), Im(p), |p|, arg(p), |v1| for each mode to VTU.

    We use VTKFile (VTU) because DOLFINx's XDMFFile requires P1 functions and
    real scalars, while our P2 complex environment makes that cumbersome.
    VTU handles P2 natively and is well-supported in ParaView.
    """
    from dolfinx import fem
    from dolfinx.io import VTKFile, XDMFFile

    field_dir = outdir / "fields"
    field_dir.mkdir(parents=True, exist_ok=True)

    V = solutions["standing"].p_function.function_space

    # Also write mesh XDMF for convenience
    mesh_path = field_dir / "mesh.xdmf"
    with XDMFFile(domain.comm, str(mesh_path), "w") as xf:
        xf.write_mesh(domain)
    log(f"  XDMF mesh: {mesh_path.name}")

    for mode, sol in solutions.items():
        p_vals = sol.p_values

        f_re = fem.Function(V, name="p_real")
        f_re.x.array[:] = np.real(p_vals)

        f_im = fem.Function(V, name="p_imag")
        f_im.x.array[:] = np.imag(p_vals)

        f_mag = fem.Function(V, name="p_mag")
        f_mag.x.array[:] = np.abs(p_vals)

        f_phase = fem.Function(V, name="p_phase")
        f_phase.x.array[:] = np.angle(p_vals)

        vtu_path = field_dir / f"{mode}_pressure.vtu"
        with VTKFile(domain.comm, str(vtu_path), "w") as vtk:
            vtk.write_function([f_re, f_im, f_mag, f_phase])
        log(f"  VTU: {vtu_path.name}  (max|p|={sol.max_pressure:.2f} Pa)")

    # |v1| for each mode (from cache)
    for mode in ["standing", "vortex", "combined"]:
        coords_v1, v1_mag = v1_cache[mode]
        V1 = fem.functionspace(domain, ("Lagrange", 1))
        f_v1 = fem.Function(V1, name="v1_mag")
        f_v1.x.array[:] = v1_mag

        vtu_v1 = field_dir / f"{mode}_v1_mag.vtu"
        with VTKFile(domain.comm, str(vtu_v1), "w") as vtk:
            vtk.write_function([f_v1])
        log(f"  VTU: {vtu_v1.name}  (max|v1|={np.max(v1_mag)*1e6:.2f} μm/s)")

    return field_dir


# ==========================================================================
# PART B: 2D slice plots (matplotlib)
# ==========================================================================
def make_slice_plots(solutions, v1_cache, outdir, cfg):
    """Generate mid-plane and near-bottom PNGs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = outdir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    z_mid = cfg.H / 2
    z_bot = cfg.H / 10
    L_mm = cfg.L * 1e3

    for mode in ["standing", "vortex", "combined"]:
        sol = solutions[mode]
        coords = sol.coords
        p_vals = sol.p_values

        # --- Mid-plane |p| and phase ---
        X, Y, Pmag = extract_slice(coords, np.abs(p_vals), z_mid, cfg)
        X, Y, Pphase = extract_slice(coords, np.angle(p_vals), z_mid, cfg)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"{mode.upper()} — mid-plane z = H/2", fontsize=14)

        im0 = axes[0].pcolormesh(X * 1e3, Y * 1e3, Pmag, shading='auto', cmap='viridis')
        axes[0].set_title("|p| [Pa]")
        axes[0].set_xlabel("x [mm]"); axes[0].set_ylabel("y [mm]")
        axes[0].set_aspect("equal")
        fig.colorbar(im0, ax=axes[0], shrink=0.8)

        im1 = axes[1].pcolormesh(X * 1e3, Y * 1e3, Pphase, shading='auto',
                                  cmap='twilight', vmin=-np.pi, vmax=np.pi)
        axes[1].set_title("arg(p) [rad]")
        axes[1].set_xlabel("x [mm]"); axes[1].set_ylabel("y [mm]")
        axes[1].set_aspect("equal")
        fig.colorbar(im1, ax=axes[1], shrink=0.8)

        fig.tight_layout()
        fig.savefig(plots_dir / f"{mode}_midplane.png", dpi=150)
        plt.close(fig)
        log(f"  Plot: {mode}_midplane.png")

        # --- Near-bottom |p| and |v1| ---
        X, Y, Pmag_bot = extract_slice(coords, np.abs(p_vals), z_bot, cfg)

        # |v1| on near-bottom — use cached v1
        coords_v1, v1_mag = v1_cache[mode]
        X_v1, Y_v1, V1grid = extract_slice(coords_v1, v1_mag, z_bot, cfg)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"{mode.upper()} — near-bottom z = H/10", fontsize=14)

        im0 = axes[0].pcolormesh(X * 1e3, Y * 1e3, Pmag_bot, shading='auto', cmap='viridis')
        axes[0].set_title("|p| [Pa]")
        axes[0].set_xlabel("x [mm]"); axes[0].set_ylabel("y [mm]")
        axes[0].set_aspect("equal")
        fig.colorbar(im0, ax=axes[0], shrink=0.8)

        im1 = axes[1].pcolormesh(X_v1 * 1e3, Y_v1 * 1e3, V1grid * 1e6, shading='auto', cmap='inferno')
        axes[1].set_title("|v₁| [μm/s]")
        axes[1].set_xlabel("x [mm]"); axes[1].set_ylabel("y [mm]")
        axes[1].set_aspect("equal")
        fig.colorbar(im1, ax=axes[1], shrink=0.8)

        fig.tight_layout()
        fig.savefig(plots_dir / f"{mode}_bottom.png", dpi=150)
        plt.close(fig)
        log(f"  Plot: {mode}_bottom.png")

    # --- Combined-minus-standing heatmap ---
    coords_c = solutions["combined"].coords
    coords_s = solutions["standing"].coords
    p_c = solutions["combined"].p_values
    p_s = solutions["standing"].p_values

    dp = np.abs(p_c) - np.abs(p_s)
    X, Y, DP = extract_slice(coords_c, dp, z_mid, cfg)

    fig, ax = plt.subplots(figsize=(6, 5))
    vmax = np.nanmax(np.abs(DP))
    im = ax.pcolormesh(X * 1e3, Y * 1e3, DP, shading='auto',
                        cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_title("|p_comb| − |p_stand|  [Pa],  mid-plane")
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(plots_dir / "delta_p_midplane.png", dpi=150)
    plt.close(fig)
    log(f"  Plot: delta_p_midplane.png")

    return plots_dir


# ==========================================================================
# PART C: Mini frequency sweep
# ==========================================================================
def frequency_sweep(domain, facet_tags, cfg_base, outdir):
    """5-point frequency sweep for standing and vortex modes.

    Re-uses the nominal-frequency mesh for all 5 points.  Only the wavenumber k
    (and omega) change between solves.  This avoids expensive mesh regeneration
    and is valid for the ±5 % range where mesh adequacy doesn't change.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sweep_dir = outdir / "freq_sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    f0 = cfg_base.frequency_hz
    fracs = np.array([-0.05, -0.025, 0.0, 0.025, 0.05])
    freqs = f0 * (1 + fracs)

    rows = []
    results_by_mode = {"standing": {"max_p": [], "mean_p": []},
                       "vortex":   {"max_p": [], "mean_p": []}}

    z_mid = cfg_base.H / 2

    for fi, freq in enumerate(freqs):
        # Create config with new frequency but REUSE existing mesh
        cfg_i = ShallowDishConfig(
            L=cfg_base.L, H=cfg_base.H,
            frequency_hz=freq,
            elements_per_wavelength=cfg_base.elements_per_wavelength,
            min_elements_z=cfg_base.min_elements_z,
            vortex_velocity_amplitude=cfg_base.vortex_velocity_amplitude,
            standing_velocity_amplitude=cfg_base.standing_velocity_amplitude,
            vortex_aperture_radius=cfg_base.vortex_aperture_radius,
            standing_axis=cfg_base.standing_axis,
            standing_phase_pattern=cfg_base.standing_phase_pattern,
        )

        for mode in ["standing", "vortex"]:
            sol = solve_helmholtz(domain, facet_tags, cfg_i, mode=mode, verbose=False)
            p_mag = np.abs(sol.p_values)
            coords = sol.coords
            tol_z = cfg_i.H / cfg_i.mesh_nz * 1.5
            mid = np.abs(coords[:, 2] - z_mid) < tol_z
            mp = float(np.max(p_mag))
            mean_mid = float(np.mean(p_mag[mid])) if np.any(mid) else 0
            results_by_mode[mode]["max_p"].append(mp)
            results_by_mode[mode]["mean_p"].append(mean_mid)
            rows.append({
                "freq_hz": freq, "mode": mode,
                "max_p_Pa": round(mp, 4),
                "mean_p_midplane_Pa": round(mean_mid, 4),
            })
        log(f"  Sweep {fi+1}/5: f={freq/1e3:.1f} kHz done")

    # Save CSV
    csv_path = sweep_dir / "freq_sweep.csv"
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=["freq_hz", "mode", "max_p_Pa", "mean_p_midplane_Pa"])
        w.writeheader()
        w.writerows(rows)
    log(f"  CSV: {csv_path.name}")

    # Plots
    for mode in ["standing", "vortex"]:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        fig.suptitle(f"Frequency sweep — {mode}", fontsize=14)

        axes[0].plot(freqs / 1e3, results_by_mode[mode]["max_p"], 'o-', lw=2)
        axes[0].axvline(f0 / 1e3, ls='--', color='gray', alpha=0.5, label='f₀')
        axes[0].set_xlabel("Frequency [kHz]")
        axes[0].set_ylabel("max |p| [Pa]")
        axes[0].set_title("Peak pressure")
        axes[0].legend()

        axes[1].plot(freqs / 1e3, results_by_mode[mode]["mean_p"], 's-', lw=2, color='C1')
        axes[1].axvline(f0 / 1e3, ls='--', color='gray', alpha=0.5, label='f₀')
        axes[1].set_xlabel("Frequency [kHz]")
        axes[1].set_ylabel("Mean |p| mid-plane [Pa]")
        axes[1].set_title("Spatial-mean pressure (mid-plane)")
        axes[1].legend()

        fig.tight_layout()
        fig.savefig(sweep_dir / f"freq_sweep_{mode}.png", dpi=150)
        plt.close(fig)
        log(f"  Plot: freq_sweep_{mode}.png")

    return sweep_dir, rows


# ==========================================================================
# PART D: Quantitative diagnostics JSON + table
# ==========================================================================
def compute_diagnostics_batch1(solutions, v1_cache, cfg, winding, outdir):
    """Compute and save quantitative diagnostics."""
    diag_dir = outdir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    coords = solutions["combined"].coords
    z_mid = cfg.H / 2
    tol_z = cfg.H / cfg.mesh_nz * 1.5
    mid = np.abs(coords[:, 2] - z_mid) < tol_z

    # max |p| per mode
    max_p = {m: float(solutions[m].max_pressure) for m in ["standing", "vortex", "combined"]}

    # |v1| on mid-plane (from cache)
    v1_data = {}
    for m in ["standing", "vortex", "combined"]:
        coords_v1, v1_mag = v1_cache[m]
        tol_z_v1 = cfg.H / cfg.mesh_nz * 1.5
        mid_v1 = np.abs(coords_v1[:, 2] - z_mid) < tol_z_v1
        v1_data[m] = float(np.max(v1_mag[mid_v1])) if np.any(mid_v1) else 0

    # Interaction metric
    p_c = np.abs(solutions["combined"].p_values)
    p_s = np.abs(solutions["standing"].p_values)
    delta = np.abs(p_c - p_s)
    interaction = float(np.max(delta[mid]) / max_p["standing"]) if max_p["standing"] > 0 else 0

    diag = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "L_mm": cfg.L * 1e3,
            "H_mm": cfg.H * 1e3,
            "freq_kHz": cfg.frequency_hz / 1e3,
            "standing_axis": cfg.standing_axis,
            "standing_phase_pattern": cfg.standing_phase_pattern,
            "V_stand_um_s": cfg.standing_velocity_amplitude * 1e6,
            "V_vortex_um_s": cfg.vortex_velocity_amplitude * 1e6,
            "disc_radius_mm": cfg.bottom_disc_radius_effective * 1e3,
            "vortex_charge": cfg.vortex_topological_charge,
        },
        "max_p_Pa": max_p,
        "max_v1_midplane_um_s": {m: round(v * 1e6, 4) for m, v in v1_data.items()},
        "phase_winding": round(winding, 4),
        "interaction_metric": round(interaction, 5),
    }

    json_path = diag_dir / "batch1_diagnostics.json"
    with open(json_path, 'w') as f:
        json.dump(diag, f, indent=2)
    log(f"  JSON: {json_path.name}")

    # Human-readable table
    txt_path = diag_dir / "batch1_summary.txt"
    lines = [
        "Batch-1 Diagnostics Summary",
        "=" * 50,
        f"Date:       {TODAY}",
        f"Domain:     {cfg.L*1e3:.0f} mm × {cfg.L*1e3:.0f} mm × {cfg.H*1e3:.1f} mm",
        f"Frequency:  {cfg.frequency_hz/1e3:.0f} kHz",
        f"Axis:       {cfg.standing_axis}  ({cfg.standing_phase_pattern})",
        "",
        "Pressure [Pa]            Standing   Vortex   Combined",
        f"  max|p| (global)       {max_p['standing']:9.2f}  {max_p['vortex']:8.2f}  {max_p['combined']:9.2f}",
        "",
        "Velocity [μm/s]          Standing   Vortex   Combined",
        f"  max|v1| (mid-plane)   {v1_data['standing']*1e6:9.2f}  {v1_data['vortex']*1e6:8.2f}  {v1_data['combined']*1e6:9.2f}",
        "",
        f"Phase winding (vortex):  {winding:.3f}  (expected {cfg.vortex_topological_charge})",
        f"Interaction metric:      {interaction:.4f}",
        f"  = max(||p_comb|-|p_stand||) / max|p_stand|  on mid-plane",
    ]
    with open(txt_path, 'w') as f:
        f.write("\n".join(lines) + "\n")
    log(f"  TXT: {txt_path.name}")

    return diag


# ==========================================================================
# MAIN
# ==========================================================================
def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    log(f"\n{'#'*70}")
    log(f"  BATCH-1 RESULTS PACK — {TODAY}")
    log(f"  Output: {OUTDIR.resolve()}")
    log(f"{'#'*70}\n")

    cfg = CFG_BASE

    # ----- Create mesh once (reused for nominal frequency + sweep) -----
    log("[1/6] Mesh generation...")
    domain, facet_tags, tag_map = create_mesh(cfg, verbose=True)

    # ----- Solve all three modes -----
    log("[2/6] Solving Helmholtz (3 modes)...")
    solutions = {}
    for mode in ["standing", "vortex", "combined"]:
        solutions[mode] = solve_helmholtz(domain, facet_tags, cfg, mode=mode, verbose=True)

    # Phase winding check
    winding = compute_phase_winding(
        solutions["vortex"],
        center_xy=(cfg.L / 2, cfg.L / 2),
        radius=2e-3,
        z=cfg.H / 2,
        n_samples=300,
    )
    log(f"  Phase winding = {winding:.3f}  (expected {cfg.vortex_topological_charge})")

    # ----- Compute |v1| for all modes ONCE -----
    log("\n[3/6] Computing |v1| fields (3 modes)...")
    v1_cache = compute_all_v1(solutions)

    # ----- A: XDMF/VTU exports -----
    log("\n[4/6] Exporting field files...")
    export_xdmf_fields(solutions, v1_cache, domain, OUTDIR)

    # ----- B: 2D slice plots -----
    log("\n[5/6] Generating 2D slice plots...")
    make_slice_plots(solutions, v1_cache, OUTDIR, cfg)

    # ----- C: Frequency sweep (REUSES nominal mesh) -----
    log("\n[6/6] Mini frequency sweep (5 points × 2 modes = 10 solves)...")
    _, sweep_rows = frequency_sweep(domain, facet_tags, cfg, OUTDIR)

    # ----- D: Diagnostics -----
    log("\n[D] Quantitative diagnostics...")
    diag = compute_diagnostics_batch1(solutions, v1_cache, cfg, winding, OUTDIR)

    # ----- Summary -----
    log(f"\n{'#'*70}")
    log("  BATCH-1 COMPLETE")
    log(f"{'#'*70}")
    log(f"  Output directory: {OUTDIR.resolve()}")
    log(f"  Modes solved: standing, vortex, combined")
    log(f"  max|p| standing:  {diag['max_p_Pa']['standing']:.2f} Pa")
    log(f"  max|p| vortex:    {diag['max_p_Pa']['vortex']:.2f} Pa")
    log(f"  max|p| combined:  {diag['max_p_Pa']['combined']:.2f} Pa")
    log(f"  Phase winding:    {winding:.3f}")
    log(f"  Interaction:      {diag['interaction_metric']:.4f}")
    log()


if __name__ == "__main__":
    main()
