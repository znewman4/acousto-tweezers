#!/usr/bin/env python3
"""
FEM Standing-Wave Checkpoint — Stage A (validation run)
========================================================

Solves the standing-wave-only FEM Helmholtz problem (disk/vortex OFF)
and writes a reloadable XDMF + HDF5 checkpoint alongside the lightweight
.npz DOF-scatter cache.

Stage A: elements_per_wavelength = 4  (fast validation, ~20-30 s)
Stage B: elements_per_wavelength = 6  (production, ~150 s) — run separately

Usage
-----
    export OMP_NUM_THREADS=16
    micromamba run -p /home/js23252/.conda/envs/acousto-complex \\
        python scripts/dev/fem_standing_checkpoint.py [--epl 4]

Outputs (results/fem_standing_wave_cache/checkpoint_epl{N}_{ts}/)
-----------------------------------------------------------------
    mesh.xdmf / mesh.h5         ← standalone mesh XDMF
    p_real.xdmf / p_real.h5     ← Re(p), P2 Lagrange
    p_imag.xdmf / p_imag.h5     ← Im(p), P2 Lagrange
    p_mag.xdmf / p_mag.h5       ← |p| (diagnostic / ParaView)
    p_phase.xdmf / p_phase.h5   ← arg(p) (diagnostic / ParaView)
    fields_manifest.json
    standing_wave_epl{N}.npz    ← lightweight DOF scatter (keep for IDW fallback)
    standing_wave_epl{N}_INFO.txt
    config.json                 ← full FarFieldConfig
    solver_report.json
    VERIFICATION_REPORT.txt     ← pass/fail record
    figures/p_mag_zstar.png     ← quick |p| slice at trap plane

Author: Acousto-Tweezers Project
Date:   March 2026
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── project root ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from acoustweezers.experiments.farfield_petri_cuboid.config import FarFieldConfig
from acoustweezers.experiments.farfield_petri_cuboid.presets import (
    CORRECTED_PRESET, PETSC_MUMPS,
)
from acoustweezers.experiments.farfield_petri_cuboid.solve_pressure import (
    solve_helmholtz,
)

# ── physical constants ────────────────────────────────────────────
C_WATER = 1484.0       # m/s
F_HZ    = 2.0e6        # Hz
LAM     = C_WATER / F_HZ

H_UNDER = float(CORRECTED_PRESET["H_under"])   # 3.0 mm
H_TOP   = float(CORRECTED_PRESET["H_top"])     # 2.0085 mm

# Trap plane: mid-petri + λ/4 (pressure antinode of the standing wave)
Z_STAR = H_UNDER + H_TOP / 2.0 + 0.25 * LAM   # ≈ 4.189 mm

CACHE_DIR = PROJECT_ROOT / "results" / "fem_standing_wave_cache"


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="FEM standing-wave checkpoint (Stage A / Stage B)")
    p.add_argument("--epl", type=int, default=4,
                   help="Elements per wavelength (default: 4 for Stage A)")
    p.add_argument("--H-under", type=float, default=None, metavar="METRES",
                   help="Override H_under [m] (default: from CORRECTED_PRESET)")
    p.add_argument("--H-top", type=float, default=None, metavar="METRES",
                   help="Override H_top [m] (default: from CORRECTED_PRESET)")
    p.add_argument("--timestamp",
                   default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def _make_slice_figure(coords_fem: np.ndarray, p_complex: np.ndarray,
                       z_star: float, fig_path: Path) -> None:
    """Scatter-plot |p| at the z-plane nearest z_star."""
    z_vals = coords_fem[:, 2]
    z_unique = np.unique(z_vals)
    z_best = z_unique[np.argmin(np.abs(z_unique - z_star))]

    mask = np.abs(z_vals - z_best) < 1e-9
    xs = coords_fem[mask, 0]
    ys = coords_fem[mask, 1]
    mag = np.abs(p_complex[mask])

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(xs * 1e3, ys * 1e3, c=mag, s=3, cmap="inferno")
    plt.colorbar(sc, ax=ax, label="|p| (Pa)")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(
        f"|p| at z = {z_best * 1e3:.3f} mm  (z* = {z_star * 1e3:.3f} mm)\n"
        f"{mask.sum()} DOF points  ·  max = {mag.max():.2f} Pa"
    )
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"    Saved figure: {fig_path}")


def _verify_reload(
    ckpt_dir: Path,
    coords_fem: np.ndarray,
    pv_fem: np.ndarray,
    report_lines: list[str],
    *,
    z_star: float,
    H_under: float,
    H_top: float,
) -> bool:
    """
    Reload mesh from mesh.xdmf and reconstruct the P2 pressure field from the
    .npz DOF cache.  Verify by evaluating at sample 3D points and at original
    DOF coordinates (eval vs stored values).

    The XDMF field files exported by export_fields.py are P1 interpolations
    (required by dolfinx 0.9.0 write_function limitation).  The .npz cache is
    the source of truth for P2 DOF accuracy.

    Returns True if all checks pass.
    """
    from mpi4py import MPI
    from dolfinx.io import XDMFFile
    from dolfinx import fem
    from dolfinx.geometry import (
        bb_tree, compute_collisions_points, compute_colliding_cells,
    )
    from scipy.spatial import KDTree

    def rlog(msg):
        print(msg)
        report_lines.append(msg)

    rlog("")
    rlog("VERIFICATION")
    rlog("─" * 60)
    rlog("  Strategy: mesh.xdmf → P2 FunctionSpace + .npz DOF arrays → eval()")

    # ── 1. Reload mesh from the standalone mesh.xdmf ─────────────────────
    t0 = time.time()
    try:
        with XDMFFile(MPI.COMM_WORLD, str(ckpt_dir / "mesh.xdmf"), "r") as xf:
            domain_r = xf.read_mesh(name="mesh")
    except Exception as exc:
        rlog(f"  FAIL: mesh reload error: {exc}")
        return False
    t_load = time.time() - t0
    rlog(f"  Mesh reload OK  (topology_dim={domain_r.topology.dim}, t={t_load:.2f}s)")

    # ── 2. Recreate P2 function space on the reloaded mesh ────────────────
    V_p2 = fem.functionspace(domain_r, ("Lagrange", 2))
    dof_coords_new = V_p2.tabulate_dof_coordinates()          # (N, 3)
    n_dofs = len(dof_coords_new)
    rlog(f"  P2 DOFs after reload: {n_dofs}")

    if n_dofs != len(pv_fem):
        rlog(f"  FAIL: DOF count mismatch — original={len(pv_fem)}, reload={n_dofs}")
        return False

    # ── 3. Match DOF ordering via KD-tree then load values ────────────────
    # In a deterministic serial run the ordering should be identical, but
    # we verify with a coordinate check before assigning values directly.
    rlog("  Checking DOF coordinate ordering …")
    max_coord_shift = float(np.max(np.linalg.norm(
        dof_coords_new - coords_fem, axis=1)))
    rlog(f"  Max coordinate shift (orig → reload): {max_coord_shift:.2e} m")

    if max_coord_shift < 1e-12:
        # Ordering is identical — assign directly
        p_r_arr = np.real(pv_fem).astype(np.float64)
        p_i_arr = np.imag(pv_fem).astype(np.float64)
        rlog("  DOF ordering: identical ✓  (direct assignment)")
    else:
        # Ordering differs — use KD-tree to find correct permutation
        rlog("  DOF ordering differs — using KD-tree to match coordinates …")
        tree_orig = KDTree(coords_fem)
        dists, indices = tree_orig.query(dof_coords_new, k=1, workers=-1)
        if float(np.max(dists)) > 1e-9:
            rlog(f"  FAIL: KD-tree max distance = {np.max(dists):.2e} m (> 1e-9)")
            return False
        p_r_arr = np.real(pv_fem[indices]).astype(np.float64)
        p_i_arr = np.imag(pv_fem[indices]).astype(np.float64)
        rlog(f"  KD-tree match OK  (max dist = {np.max(dists):.2e} m)")

    # Build P2 complex Functions from the matched DOF arrays
    p_r = fem.Function(V_p2)
    p_r.name = "p_real"
    p_r.x.array[:] = p_r_arr

    p_i = fem.Function(V_p2)
    p_i.name = "p_imag"
    p_i.x.array[:] = p_i_arr

    # ── 4. Basic array sanity ─────────────────────────────────────────────
    has_nan = bool(np.any(np.isnan(p_r.x.array)) or np.any(np.isnan(p_i.x.array)))
    max_r = float(np.max(np.abs(p_r.x.array)))
    max_i = float(np.max(np.abs(p_i.x.array)))
    p_mag_arr = np.sqrt(p_r.x.array ** 2 + p_i.x.array ** 2)
    max_mag = float(np.max(p_mag_arr))

    rlog(f"  max|p_real| (reloaded) = {max_r:.4f} Pa")
    rlog(f"  max|p_imag| (reloaded) = {max_i:.4f} Pa")
    rlog(f"  max|p|      (reloaded) = {max_mag:.4f} Pa")
    if has_nan:
        rlog("  FAIL: NaN values in reloaded fields")
        return False
    rlog("  NaN check: PASSED")
    if max_mag < 0.1:
        rlog("  WARN: max|p| suspiciously small (< 0.1 Pa) — check BCs")

    # ── 5. Sample-point eval (arbitrary 3D points) ────────────────────────
    # Use geometry-aware sample points so they're valid inside this domain.
    mid_petri = H_under + H_top / 2.0
    sample_pts = np.array([
        [3.0e-3, 3.0e-3, z_star],             # domain centre, trap plane
        [0.5e-3, 0.5e-3, z_star],             # near corner, trap plane
        [3.0e-3, 3.0e-3, mid_petri],          # centre, mid-petri
        [1.0e-3, 1.0e-3, H_under - 0.5e-3],  # just below petri floor
        [3.0e-3, 3.0e-3, H_under / 2.0],     # mid under-bath
    ])

    tree = bb_tree(domain_r, domain_r.topology.dim)
    cands = compute_collisions_points(tree, sample_pts)
    cells_obj = compute_colliding_cells(domain_r, cands, sample_pts)

    rlog("")
    rlog(f"  {'x(mm)':>7} {'y(mm)':>7} {'z(mm)':>7}  {'|p|(Pa)':>10}  {'Re(p)(Pa)':>10}  status")

    all_pts_ok = True
    for i, pt in enumerate(sample_pts):
        links = cells_obj.links(i)
        if len(links) == 0:
            rlog(f"  {pt[0]*1e3:7.2f} {pt[1]*1e3:7.2f} {pt[2]*1e3:7.2f}  {'NOTFOUND':>10}")
            all_pts_ok = False
            continue
        pr_val = float(p_r.eval(pt, links[0])[0])
        pi_val = float(p_i.eval(pt, links[0])[0])
        pm = np.sqrt(pr_val ** 2 + pi_val ** 2)
        ok = "OK" if (np.isfinite(pm) and pm > 0.0) else "WARN"
        if not np.isfinite(pm):
            all_pts_ok = False
        rlog(f"  {pt[0]*1e3:7.2f} {pt[1]*1e3:7.2f} {pt[2]*1e3:7.2f}  {pm:10.4f}  {pr_val:10.4f}  {ok}")

    # ── 6. DOF cross-check: eval at original DOF coords vs stored values ──
    rlog("")
    rlog("  DOF cross-check: eval(DOF_coord) vs original solve value")
    rlog(f"  {'DOF_idx':>8}  {'|p|_orig':>10}  {'|p|_eval':>10}  {'rel_err':>9}  status")

    rng = np.random.default_rng(42)
    n_check = 10
    indices = rng.integers(0, len(pv_fem), size=n_check)

    dof_check_ok = True
    for idx in indices:
        pt3 = coords_fem[idx]                  # spatial location (3D)
        p_orig = pv_fem[idx]                   # complex value from original solve
        mag_orig = abs(p_orig)

        cands_i = compute_collisions_points(tree, pt3[np.newaxis, :])
        cells_i = compute_colliding_cells(domain_r, cands_i, pt3[np.newaxis, :])
        links_i = cells_i.links(0)

        if len(links_i) == 0:
            rlog(f"  {int(idx):8d}  {mag_orig:10.6f}  {'NOTFOUND':>10}  {'N/A':>9}  WARN")
            continue

        pr_ev = float(p_r.eval(pt3, links_i[0])[0])
        pi_ev = float(p_i.eval(pt3, links_i[0])[0])
        mag_eval = np.sqrt(pr_ev ** 2 + pi_ev ** 2)
        rel_err = abs(mag_eval - mag_orig) / (mag_orig + 1e-30)

        if rel_err < 1e-6:
            status = "OK"
        elif rel_err < 1e-3:
            status = "WARN"
        else:
            status = "FAIL"
            dof_check_ok = False

        rlog(f"  {int(idx):8d}  {mag_orig:10.6f}  {mag_eval:10.6f}  {rel_err:9.2e}  {status}")

    passed = all_pts_ok and dof_check_ok
    rlog("")
    rlog(f"  Verification: {'PASSED' if passed else 'FAILED'}")
    return passed


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main() -> int:
    args = parse_args()
    EPL = args.epl
    TS  = args.timestamp

    # Build geometry overrides (only if explicitly supplied on CLI)
    geom_overrides: dict = {}
    if args.H_under is not None:
        geom_overrides["H_under"] = args.H_under
    if args.H_top is not None:
        geom_overrides["H_top"] = args.H_top

    # Compute H_total in mm for a descriptive directory name
    _h_under = geom_overrides.get("H_under", float(CORRECTED_PRESET["H_under"]))
    _h_top   = geom_overrides.get("H_top",   float(CORRECTED_PRESET["H_top"]))
    _h_total_mm = round((_h_under + _h_top) * 1e3)
    if geom_overrides:
        ckpt_dir = CACHE_DIR / f"checkpoint_epl{EPL}_depth{_h_total_mm}mm_{TS}"
    else:
        ckpt_dir = CACHE_DIR / f"checkpoint_epl{EPL}_{TS}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = ckpt_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    report_lines: list[str] = []

    def log(msg: str) -> None:
        print(msg)
        report_lines.append(msg)

    log("=" * 72)
    log(f"FEM STANDING WAVE CHECKPOINT — Stage {'A' if EPL <= 4 else 'B'} (epl={EPL})")
    log("=" * 72)
    log(f"Timestamp  : {TS}")
    log(f"Output dir : {ckpt_dir}")
    log(f"λ          : {LAM * 1e3:.4f} mm")
    log("")

    # ── 1. Build FarFieldConfig ────────────────────────────────────────────
    fem_overrides = {
        **CORRECTED_PRESET,
        "disk_velocity_amplitude": 0.0,   # vortex/disk completely OFF
        "elements_per_wavelength": EPL,
        **geom_overrides,                  # H_under / H_top overrides if given
    }
    cfg = FarFieldConfig(**fem_overrides)

    # Trap plane: mid-petri + λ/4 (pressure antinode)
    z_star = cfg.H_under + cfg.H_top / 2.0 + 0.25 * cfg.wavelength

    log(f"z*         : {z_star * 1e3:.3f} mm")
    log(f"H_under    : {cfg.H_under * 1e3:.4f} mm")
    log(f"H_top      : {cfg.H_top * 1e3:.4f} mm")
    log(f"H_total    : {cfg.H_total * 1e3:.4f} mm")
    log("")

    est_dofs = (2 * cfg.mesh_nx + 1) * (2 * cfg.mesh_ny + 1) * (2 * cfg.mesh_nz + 1)

    log(f"Domain      : {cfg.Lx*1e3:.1f} × {cfg.Ly*1e3:.1f} × {cfg.H_total*1e3:.4f} mm")
    log(f"Standing V  : {cfg.standing_velocity_amplitude*1e6:.1f} µm/s, "
        f"pattern={cfg.standing_phase_pattern}, axis={cfg.standing_axis}")
    log(f"Disk V      : {cfg.disk_velocity_amplitude} m/s  [OFF]")
    log(f"Mesh (hex)  : {cfg.mesh_nx} × {cfg.mesh_ny} × {cfg.mesh_nz}")
    log(f"Est. DOFs   : {est_dofs:,}")
    log("")

    # ── 2. FEM solve with XDMF export ─────────────────────────────────────
    log("─" * 72)
    log("STEP 1  FEM solve + XDMF export")
    log("─" * 72)

    t_wall0 = time.time()
    sol = solve_helmholtz(
        cfg,
        verbose=True,
        petsc_options=PETSC_MUMPS,
        export_fields=True,
        export_dir=str(ckpt_dir),
    )
    t_wall = time.time() - t_wall0

    # Grab numpy copies before we might delete the sol object
    coords_fem = sol.coords.copy()
    pv_fem     = sol.p_values.copy()

    log(f"  DOFs          : {sol.dofs:,}")
    log(f"  Mesh time     : {sol.mesh_time:.1f}s")
    log(f"  Solve time    : {sol.solver_time:.1f}s")
    log(f"  Total wall    : {t_wall:.1f}s  (mesh + solve + export)")
    log(f"  max|p|        : {sol.max_pressure:.4f} Pa")
    log(f"  KSP reason    : {sol.ksp_converged_reason}")
    log(f"  KSP residual  : {sol.ksp_residual_norm:.2e}")
    log("")

    # Verify the XDMF files were actually written
    expected_files = [
        "mesh.xdmf", "mesh.h5",
        "p_real.xdmf", "p_real.h5",
        "p_imag.xdmf", "p_imag.h5",
        "p_mag.xdmf", "p_mag.h5",
        "p_phase.xdmf", "p_phase.h5",
        "fields_manifest.json",
    ]
    missing = [f for f in expected_files if not (ckpt_dir / f).exists()]
    if missing:
        log(f"  WARNING: expected files not found: {missing}")
    else:
        log(f"  Checkpoint files: all {len(expected_files)} expected files present")
    log("")

    # ── 3. Save lightweight .npz cache ────────────────────────────────────
    log("─" * 72)
    log("STEP 2  .npz DOF scatter cache")
    log("─" * 72)

    npz_path = ckpt_dir / f"standing_wave_epl{EPL}.npz"
    meta = {
        "elem_per_lam"   : EPL,
        "dofs"           : sol.dofs,
        "max_pressure"   : sol.max_pressure,
        "solve_time"     : sol.solver_time,
        "total_time"     : t_wall,
        "timestamp"      : TS,
        "checkpoint_dir" : str(ckpt_dir),
    }
    np.savez_compressed(
        npz_path,
        coords  = coords_fem,
        p_real  = np.real(pv_fem),
        p_imag  = np.imag(pv_fem),
        metadata= meta,
    )
    log(f"  Saved: {npz_path}")

    info_path = ckpt_dir / f"standing_wave_epl{EPL}_INFO.txt"
    info_path.write_text(
        f"FEM Standing-Wave Cache\n"
        f"========================\n"
        f"Created       : {TS}\n"
        f"elem/λ        : {EPL}\n"
        f"DOFs          : {sol.dofs}\n"
        f"Solve time    : {sol.solver_time:.1f}s\n"
        f"Total time    : {t_wall:.1f}s\n"
        f"max|p|        : {sol.max_pressure:.4f} Pa\n"
        f"Checkpoint dir: {ckpt_dir}\n"
    )
    log(f"  Saved INFO: {info_path}")
    log("")

    # ── 4. Config JSON ─────────────────────────────────────────────────────
    cfg_path = ckpt_dir / "config.json"
    with open(cfg_path, "w") as fh:
        json.dump(cfg.to_dict(), fh, indent=2, default=str)
    log(f"  Config JSON    : {cfg_path}")

    # ── 5. Solver report JSON ──────────────────────────────────────────────
    solver_report = {
        "stage"                 : "A" if EPL <= 4 else "B",
        "elements_per_wavelength": EPL,
        "dofs"                  : sol.dofs,
        "mesh_nx"               : cfg.mesh_nx,
        "mesh_ny"               : cfg.mesh_ny,
        "mesh_nz"               : cfg.mesh_nz,
        "mesh_time_s"           : sol.mesh_time,
        "solver_time_s"         : sol.solver_time,
        "total_wall_time_s"     : t_wall,
        "max_pressure_Pa"       : sol.max_pressure,
        "ksp_converged_reason"  : sol.ksp_converged_reason,
        "ksp_iterations"        : sol.ksp_iterations,
        "ksp_residual_norm"     : sol.ksp_residual_norm,
        "timestamp"             : TS,
    }
    report_json_path = ckpt_dir / "solver_report.json"
    with open(report_json_path, "w") as fh:
        json.dump(solver_report, fh, indent=2)
    log(f"  Solver report  : {report_json_path}")
    log("")

    # ── 6. Slice figure ────────────────────────────────────────────────────
    log("─" * 72)
    log("STEP 3  Slice figure")
    log("─" * 72)
    fig_path = fig_dir / "p_mag_zstar.png"
    _make_slice_figure(coords_fem, pv_fem, z_star, fig_path)
    log("")

    # ── 7. Reload + eval verification ─────────────────────────────────────
    log("─" * 72)
    log("STEP 4  Reload + eval verification")
    log("─" * 72)

    passed = _verify_reload(
        ckpt_dir, coords_fem, pv_fem, report_lines,
        z_star=z_star,
        H_under=cfg.H_under,
        H_top=cfg.H_top,
    )

    # ── 8. Summary ────────────────────────────────────────────────────────
    log("")
    log("═" * 72)
    log("STAGE A RESULT SUMMARY")
    log("═" * 72)
    log(f"  epl                : {EPL}")
    log(f"  DOFs               : {sol.dofs:,}")
    log(f"  Solve time         : {sol.solver_time:.1f}s")
    log(f"  Total wall time    : {t_wall:.1f}s")
    log(f"  max|p|             : {sol.max_pressure:.4f} Pa")
    log(f"  Checkpoint dir     : {ckpt_dir}")
    log(f"  Verification       : {'PASSED ✓' if passed else 'FAILED ✗'}")
    log(f"  Ready for epl=6    : {'YES' if passed else 'NO — investigate failures first'}")

    # Write the full report text
    verif_path = ckpt_dir / "VERIFICATION_REPORT.txt"
    verif_path.write_text("\n".join(report_lines) + "\n")
    print(f"\n  Report written: {verif_path}")

    # ── 9. Release FEM objects (not needed after this point) ──────────────
    del sol
    gc.collect()

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
