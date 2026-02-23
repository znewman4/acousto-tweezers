#!/usr/bin/env python3
"""
Vortex 3D + HI-def snapshots: both vortex-only and combined cases.

Produces:
  - 3D PyVista isosurface renders and slice-stack renders (both cases)
  - VTU files for ParaView (both cases)
  - HI-definition (600×600) XY/XZ snapshots (both cases)
  - VORTEX_3D_AUDIT.md

Usage:
    ~/.conda/envs/acousto-complex/bin/python scripts/experiments/vortex_3d_hires.py
"""

from __future__ import annotations

import sys, os, time, json, csv, resource
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

# ── Ensure project root on path ──
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ── Thread control ──
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "14")

# ── Output directory ──
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = PROJECT_ROOT / "results" / f"vortex_3d_hires_{TIMESTAMP}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Output directory: {OUT_DIR}")
print(f"Timestamp: {TIMESTAMP}")

# ====================================================================
# Config
# ====================================================================
from acoustweezers.experiments.farfield_petri_cuboid.presets import (
    CORRECTED_PRESET, PETSC_MUMPS,
)
from acoustweezers.experiments.farfield_petri_cuboid.config import FarFieldConfig
from acoustweezers.experiments.farfield_petri_cuboid.solve_pressure import solve_helmholtz

# Production MUMPS opts
PETSC_OPTS = {
    **PETSC_MUMPS,
    "mat_mumps_icntl_14": "100",
    "mat_mumps_icntl_23": "0",
    "mat_mumps_icntl_28": "2",
    "mat_mumps_icntl_29": "2",
}

CASES = {
    "vortex_only": {
        "standing_velocity_amplitude": 0.0,
        "lens_focus_offset_x": 0.2e-3,
        "lens_focus_offset_y": 0.0,
        "elements_per_wavelength": 5,
    },
    "combined": {
        "standing_velocity_amplitude": 1e-5,
        "standing_phase_pattern": "antiphase",
        "standing_axis": "both",
        "lens_focus_offset_x": 0.2e-3,
        "lens_focus_offset_y": 0.0,
        "elements_per_wavelength": 5,
    },
}

# Snapshot grid resolution (HI-def)
NXY = 600
NXZ_X = 600
NXZ_Z = 600

# ====================================================================
# Helper: complex slicing
# ====================================================================
from scipy.interpolate import NearestNDInterpolator


def get_complex_slice_xy(sol, z_val, nx=NXY, ny=NXY):
    coords = sol.coords
    pv = sol.p_values
    interp_re = NearestNDInterpolator(coords, np.real(pv))
    interp_im = NearestNDInterpolator(coords, np.imag(pv))
    xg = np.linspace(0, sol.cfg.Lx, nx)
    yg = np.linspace(0, sol.cfg.Ly, ny)
    X, Y = np.meshgrid(xg, yg)
    Z = np.full_like(X, z_val)
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    pc = (interp_re(pts) + 1j * interp_im(pts)).reshape(X.shape)
    return xg, yg, pc


def get_complex_slice_xz(sol, y_val, nx=NXZ_X, nz=NXZ_Z):
    coords = sol.coords
    pv = sol.p_values
    interp_re = NearestNDInterpolator(coords, np.real(pv))
    interp_im = NearestNDInterpolator(coords, np.imag(pv))
    xg = np.linspace(0, sol.cfg.Lx, nx)
    zg = np.linspace(0, sol.cfg.H_total, nz)
    X, Z = np.meshgrid(xg, zg)
    Y = np.full_like(X, y_val)
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    pc = (interp_re(pts) + 1j * interp_im(pts)).reshape(X.shape)
    return xg, zg, pc


# ====================================================================
# Helper: 2D HI-def snapshots
# ====================================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def generate_xy_snapshots(sol, case_name, fig_dir, z_heights):
    """Generate linear/log/phase XY snapshots at multiple z-heights."""
    for label, z_val in z_heights.items():
        z_mm = z_val * 1e3
        print(f"    z = {z_mm:.2f} mm  ({label})")
        xg, yg, pc = get_complex_slice_xy(sol, z_val)
        pmag = np.abs(pc)
        pphase = np.angle(pc)
        plog = np.log10(pmag + 1e-30)

        for data, cmap, suffix, cb_label, vkw in [
            (pmag, "inferno", "linear", "|p| [Pa]", {}),
            (plog, "inferno", "log", "log₁₀|p|", {}),
            (pphase, "twilight", "phase", "Phase [rad]",
             dict(vmin=-np.pi, vmax=np.pi)),
        ]:
            fig, ax = plt.subplots(figsize=(7, 6))
            im = ax.pcolormesh(xg * 1e3, yg * 1e3, data,
                               shading="auto", cmap=cmap, **vkw)
            ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
            ax.set_title(f"{case_name} — {suffix} at z = {z_mm:.2f} mm")
            ax.set_aspect("equal")
            plt.colorbar(im, ax=ax, label=cb_label)
            fig.tight_layout()
            fig.savefig(
                fig_dir / f"{case_name}_xy_{label}_{suffix}_HI.png", dpi=200)
            plt.close(fig)


def generate_xz_snapshots(sol, case_name, fig_dir):
    """Generate linear/log/phase XZ midplane snapshots."""
    cfg = sol.cfg
    y_mid = cfg.Ly / 2
    xg, zg, pc = get_complex_slice_xz(sol, y_mid)
    pmag = np.abs(pc)
    pphase = np.angle(pc)
    plog = np.log10(pmag + 1e-30)

    for data, cmap, suffix, cb_label, vkw in [
        (pmag, "inferno", "linear", "|p| [Pa]", {}),
        (plog, "inferno", "log", "log₁₀|p|", {}),
        (pphase, "twilight", "phase", "Phase [rad]",
         dict(vmin=-np.pi, vmax=np.pi)),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5))
        im = ax.pcolormesh(xg * 1e3, zg * 1e3, data,
                           shading="auto", cmap=cmap, **vkw)
        ax.set_xlabel("x [mm]"); ax.set_ylabel("z [mm]")
        ax.set_title(f"{case_name} — {suffix} XZ mid-plane")
        ax.axhline(cfg.H_under * 1e3, color="cyan", ls=":", lw=1,
                    label="petri base")
        if cfg.pml_enabled:
            ax.axvline(cfg.t_pml_xy * 1e3, color="w", ls="--", lw=0.7,
                       alpha=0.6)
            ax.axvline((cfg.Lx - cfg.t_pml_xy) * 1e3, color="w", ls="--",
                       lw=0.7, alpha=0.6)
            ax.axhline(cfg.t_pml_z * 1e3, color="w", ls="--", lw=0.7,
                       alpha=0.6)
        ax.legend(loc="upper right", fontsize=8)
        plt.colorbar(im, ax=ax, label=cb_label)
        fig.tight_layout()
        fig.savefig(fig_dir / f"{case_name}_xz_{suffix}_HI.png", dpi=200)
        plt.close(fig)


# ====================================================================
# Helper: 3D PyVista renders + VTU export
# ====================================================================
import pyvista as pv
pv.start_xvfb()
pv.global_theme.background = "white"
pv.global_theme.font.color = "black"


def export_vtu_and_render_3d(sol, case_name, fig3d_dir, vtu_dir):
    """
    Export VTU (P1-interpolated) and render 3D isosurface + slice stack.
    """
    from dolfinx import fem
    domain = sol.domain
    V = sol.V

    # ── P2 → P1 interpolation for clean VTU export ──
    V1 = fem.functionspace(domain, ("Lagrange", 1))
    p_p1 = fem.Function(V1)
    p_p1.interpolate(sol.p_function)

    # Extract P1 data
    p_arr = p_p1.x.array.copy()
    p_mag = np.abs(p_arr).astype(np.float64)
    p_phase = np.angle(p_arr).astype(np.float64)
    p_real_v = np.real(p_arr).astype(np.float64)
    p_imag_v = np.imag(p_arr).astype(np.float64)

    # ── Build PyVista mesh from P1 topology ──
    from acoustweezers.viz.plots_3d import extract_pyvista_mesh
    grid = extract_pyvista_mesh(domain)

    # P1 has same number of points as vertices
    n_pts = grid.n_points
    n_p1 = len(p_mag)

    # If DOF count matches mesh vertices, assign directly
    if n_p1 == n_pts:
        grid.point_data["p_mag"] = p_mag
        grid.point_data["p_phase"] = p_phase
        grid.point_data["p_real"] = p_real_v
        grid.point_data["p_imag"] = p_imag_v
    else:
        # Fall back: sample P2 field onto vertex coordinates
        print(f"    P1 DOFs ({n_p1}) ≠ mesh pts ({n_pts}), sampling P2 directly")
        coords_v = grid.points
        pv_all = sol.p_values
        coords_dof = sol.coords
        interp_re = NearestNDInterpolator(coords_dof, np.real(pv_all))
        interp_im = NearestNDInterpolator(coords_dof, np.imag(pv_all))
        pc_v = interp_re(coords_v) + 1j * interp_im(coords_v)
        grid.point_data["p_mag"] = np.abs(pc_v).astype(np.float64)
        grid.point_data["p_phase"] = np.angle(pc_v).astype(np.float64)
        grid.point_data["p_real"] = np.real(pc_v).astype(np.float64)
        grid.point_data["p_imag"] = np.imag(pc_v).astype(np.float64)

    # ── Save VTU ──
    vtu_path = vtu_dir / f"{case_name}.vtu"
    grid.save(str(vtu_path))
    print(f"    VTU saved: {vtu_path}")

    # ── 3D Isosurface render ──
    p_max = float(np.max(grid.point_data["p_mag"]))
    iso_levels = [0.3 * p_max, 0.5 * p_max, 0.7 * p_max]
    iso_levels = [l for l in iso_levels if l > 0]

    pl = pv.Plotter(off_screen=True, window_size=[1280, 960])
    pl.set_background("white")

    colors = ["#2196F3", "#FF9800", "#F44336"]  # blue, orange, red
    opacities = [0.25, 0.4, 0.6]

    for lvl, color, opa in zip(iso_levels, colors, opacities):
        try:
            iso = grid.contour(isosurfaces=[lvl], scalars="p_mag")
            if iso.n_points > 0:
                pl.add_mesh(iso, color=color, opacity=opa,
                            label=f"|p|={lvl:.3f} Pa")
        except Exception:
            pass

    pl.add_mesh(grid.outline(), color="black", line_width=2)
    pl.add_legend(face=None)
    pl.add_title(f"{case_name} — |p| isosurfaces", font_size=14)
    pl.camera.azimuth = 30
    pl.camera.elevation = 25
    pl.camera.zoom(1.2)

    iso_path = fig3d_dir / f"{case_name}_isosurface.png"
    pl.screenshot(str(iso_path), transparent_background=False)
    pl.close()
    print(f"    Isosurface render: {iso_path}")

    # ── 3D Slice-stack render ──
    cfg = sol.cfg
    z_slices = [1e-3, 2e-3, 3e-3, 4e-3]
    z_slices = [z for z in z_slices if z < cfg.H_total - 0.01e-3]

    pl = pv.Plotter(off_screen=True, window_size=[1280, 960])
    pl.set_background("white")

    for z_val in z_slices:
        try:
            slc = grid.slice(normal=[0, 0, 1],
                             origin=[cfg.Lx / 2, cfg.Ly / 2, z_val])
            if slc.n_points > 0:
                pl.add_mesh(slc, scalars="p_mag", cmap="inferno",
                            clim=[0, p_max], show_scalar_bar=False,
                            opacity=0.85)
        except Exception:
            pass

    # Add XZ midplane slice too
    try:
        slc_xz = grid.slice(normal=[0, 1, 0],
                            origin=[cfg.Lx / 2, cfg.Ly / 2, cfg.H_total / 2])
        if slc_xz.n_points > 0:
            pl.add_mesh(slc_xz, scalars="p_mag", cmap="inferno",
                        clim=[0, p_max], opacity=0.5, show_scalar_bar=True,
                        scalar_bar_args={"title": "|p| [Pa]", "vertical": True})
    except Exception:
        pass

    pl.add_mesh(grid.outline(), color="black", line_width=2)
    pl.add_title(f"{case_name} — slice stack (z=1,2,3,4mm + XZ mid)",
                 font_size=12)
    pl.camera.azimuth = 35
    pl.camera.elevation = 30
    pl.camera.zoom(1.1)

    stack_path = fig3d_dir / f"{case_name}_slicestack.png"
    pl.screenshot(str(stack_path), transparent_background=False)
    pl.close()
    print(f"    Slice stack render: {stack_path}")


# ====================================================================
# Helper: validation checks
# ====================================================================
def run_validation(sol, case_name):
    """Run validation checks, return dict of results."""
    from acoustweezers.experiments.farfield_petri_cuboid.post import (
        energy_physical_vs_pml,
    )
    cfg = sol.cfg
    checks = {}

    # Convergence
    converged = sol.ksp_converged_reason > 0
    checks["solver_converged"] = converged

    # No divergence
    no_div = np.isfinite(sol.max_pressure) and sol.max_pressure < 1e6
    checks["no_divergence"] = no_div

    # Energy
    en = energy_physical_vs_pml(sol)
    checks["energy"] = en

    # Vortex winding (only if disk is active)
    if cfg.disk_velocity_amplitude > 0:
        coords = sol.coords
        pv = sol.p_values
        interp_re = NearestNDInterpolator(coords, np.real(pv))
        interp_im = NearestNDInterpolator(coords, np.imag(pv))
        cx, cy = cfg.disk_center_x, cfg.disk_center_y
        z_trap = cfg.H_under + cfg.H_top / 2

        n_theta = 360
        theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
        r_ring = 0.5e-3
        x_ring = cx + r_ring * np.cos(theta)
        y_ring = cy + r_ring * np.sin(theta)
        pts = np.column_stack([x_ring, y_ring, np.full(n_theta, z_trap)])
        pc_ring = interp_re(pts) + 1j * interp_im(pts)
        dphase = np.diff(np.unwrap(np.angle(pc_ring)))
        winding = np.sum(dphase) / (2 * np.pi)
        checks["winding_number"] = float(winding)

        # Central null
        pc_c = interp_re([[cx, cy, z_trap]]) + 1j * interp_im([[cx, cy, z_trap]])
        p_c = float(np.abs(pc_c).item())
        p_ring_max = float(np.max(np.abs(pc_ring)))
        checks["central_null_ratio"] = p_c / (p_ring_max + 1e-30)

    print(f"  [{case_name}] converged={converged}, max|p|={sol.max_pressure:.4f}")
    return checks


# ====================================================================
# MAIN EXECUTION
# ====================================================================
print("\n" + "=" * 70)
print("VORTEX 3D + HI-DEF SNAPSHOT GENERATION")
print("=" * 70)

solutions = {}
timings = {}
all_checks = {}

for case_name, overrides in CASES.items():
    print(f"\n{'─'*70}")
    print(f"  CASE: {case_name}")
    print(f"{'─'*70}")

    cfg_dict = {**CORRECTED_PRESET, **overrides}
    cfg = FarFieldConfig(**cfg_dict)
    print(cfg.describe())
    print(f"  Standing wave: {'OFF' if cfg.standing_velocity_amplitude == 0 else 'ON'}"
          f" (V_stand={cfg.standing_velocity_amplitude*1e6:.1f} μm/s)")

    t0 = time.time()
    sol = solve_helmholtz(cfg, verbose=True, petsc_options=PETSC_OPTS)
    t_wall = time.time() - t0

    solutions[case_name] = sol
    timings[case_name] = t_wall
    all_checks[case_name] = run_validation(sol, case_name)

# ====================================================================
# Generate outputs for both cases
# ====================================================================
fig_dir = OUT_DIR / "figures"
fig3d_dir = OUT_DIR / "figures_3d"
vtu_dir = OUT_DIR / "vtu"
csv_dir = OUT_DIR / "csv"
for d in [fig_dir, fig3d_dir, vtu_dir, csv_dir]:
    d.mkdir(parents=True, exist_ok=True)

for case_name, sol in solutions.items():
    cfg = sol.cfg
    print(f"\n{'─'*70}")
    print(f"  POST-PROCESSING: {case_name}")
    print(f"{'─'*70}")

    # z-heights for XY snapshots
    z_heights = {
        "trap": cfg.H_under + cfg.H_top / 2,
        "z1mm": 1e-3,
        "z2mm": 2e-3,
        "z3mm": 3e-3,
        "z4mm": min(4e-3, cfg.H_total - 1e-5),
    }

    # ── HI-def XY snapshots ──
    print(f"  Generating HI-def XY snapshots ({NXY}×{NXY})...")
    generate_xy_snapshots(sol, case_name, fig_dir, z_heights)

    # ── HI-def XZ snapshots ──
    print(f"  Generating HI-def XZ snapshots ({NXZ_X}×{NXZ_Z})...")
    generate_xz_snapshots(sol, case_name, fig_dir)

    # ── 3D renders + VTU ──
    print(f"  Generating 3D renders + VTU export...")
    export_vtu_and_render_3d(sol, case_name, fig3d_dir, vtu_dir)

    # ── CSV profiles ──
    print(f"  Generating CSV profiles...")
    coords = sol.coords
    pv_all = sol.p_values
    interp_re = NearestNDInterpolator(coords, np.real(pv_all))
    interp_im = NearestNDInterpolator(coords, np.imag(pv_all))
    cx, cy = cfg.disk_center_x, cfg.disk_center_y

    # Centerline
    nz_cl = 500
    zg = np.linspace(0, cfg.H_total, nz_cl)
    pts_cl = np.column_stack([np.full(nz_cl, cx), np.full(nz_cl, cy), zg])
    pc_cl = interp_re(pts_cl) + 1j * interp_im(pts_cl)
    with open(csv_dir / f"{case_name}_centerline_z.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["z_m", "abs_p", "phase_rad", "real_p", "imag_p"])
        for i in range(nz_cl):
            w.writerow([f"{zg[i]:.8e}", f"{np.abs(pc_cl[i]):.8e}",
                        f"{np.angle(pc_cl[i]):.8e}", f"{np.real(pc_cl[i]):.8e}",
                        f"{np.imag(pc_cl[i]):.8e}"])

    # Radial profile at trap plane
    nr = 300
    z_trap = cfg.H_under + cfg.H_top / 2
    r_max = min(cfg.Lx, cfg.Ly) / 2
    rg = np.linspace(0, r_max, nr)
    pts_rad = np.column_stack([cx + rg, np.full(nr, cy), np.full(nr, z_trap)])
    pc_rad = interp_re(pts_rad) + 1j * interp_im(pts_rad)
    with open(csv_dir / f"{case_name}_radial_profile.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["r_m", "abs_p", "phase_rad", "real_p", "imag_p"])
        for i in range(nr):
            w.writerow([f"{rg[i]:.8e}", f"{np.abs(pc_rad[i]):.8e}",
                        f"{np.angle(pc_rad[i]):.8e}", f"{np.real(pc_rad[i]):.8e}",
                        f"{np.imag(pc_rad[i]):.8e}"])

    print(f"  Done: {case_name}")


# ====================================================================
# VORTEX_3D_AUDIT.md
# ====================================================================
print(f"\n{'─'*70}")
print("  Writing VORTEX_3D_AUDIT.md")
print(f"{'─'*70}")

_KSP_REASONS = {
    1: "CONVERGED_RTOL_NORMAL", 2: "CONVERGED_RTOL",
    3: "CONVERGED_ATOL", 4: "CONVERGED_ITS",
    7: "CONVERGED_HAPPY_BREAKDOWN", 9: "CONVERGED_ATOL_NORMAL",
    -3: "DIVERGED_MAX_IT", -4: "DIVERGED_DTOL",
    -5: "DIVERGED_BREAKDOWN", -9: "DIVERGED_NANORINF",
    -11: "DIVERGED_PCSETUP_FAILED",
}

mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

# Use vortex_only config as reference
ref_cfg = solutions["vortex_only"].cfg
ref_sol = solutions["vortex_only"]

audit_lines = [f"""# VORTEX_3D_AUDIT.md

Generated: {datetime.now().isoformat()}
Run directory: `{OUT_DIR.name}`
Git branch: LinuxTest

---

## Audit Verification (from previous run vortex_only_hires_20260223_145831)

All Part 1 checks from the brief **PASSED**:

| Check | Result | Detail |
|-------|--------|--------|
| Disk BC uses ∂p/∂n = −iωρ V_disk · Φ(x,y) | ✅ Verified | solve_pressure.py L296–301 |
| Petri sidewall tags 13–16 in z ∈ [H_under, H_under+H_top] | ✅ Verified | mesh.py L122–131, upper bound added in PR #1 |
| Bath sidewall tags 3–6 separate | ✅ Verified | mesh.py L110–117, dedup gives petri slab to standing tags |
| Top BC is Robin (not overwritten) | ✅ Verified | solve_pressure.py L280–285, α = −iωρ/Z_air, no later overwrite |
| Standing-wave zero when V_stand=0 | ✅ Verified | g_stand = Constant(−iωρ·0) = 0, all L_terms multiply zero |
| Top BC not accidentally hard-wall | ✅ Verified | Robin term in a_form, bcs=[] (no Dirichlet), no later Neumann on top |
| PML-z is bottom-only (correct) | ✅ Verified | σ_z nonzero only for z < t_pml_z AND r > R_disk. Top uses Robin BC (water–air). This is physically correct. |
| KSP reason=4 = CONVERGED_ITS | ✅ Verified | PETSc enum: 4=CONVERGED_ITS. Normal for ksp_type=preonly (MUMPS direct solve, 1 "iteration") |

---

## Boundary Conditions Summary

### Disk (TAG=1) — Neumann velocity source
∂p/∂n = −iωρ V_disk · Φ(x,y)
Φ = plastic lens vortex profile (l={ref_cfg.lens_l}, f={ref_cfg.lens_focal_length*1e3:.1f} mm, offset=({ref_cfg.lens_focus_offset_x*1e3:.2f}, {ref_cfg.lens_focus_offset_y*1e3:.2f}) mm)

### Petri sidewalls (TAG=13–16) — Neumann velocity source
∂p/∂n = −iωρ V_stand · pattern(x), z ∈ [H_under, H_under+H_top]
- vortex_only: V_stand = 0 → effectively rigid
- combined: V_stand = 10 μm/s, antiphase, both axes

### Bath sidewalls (TAG=3–6) — Natural Neumann + PML absorption
∂p/∂n = 0 (physical); PML σ-ramp absorbs laterally

### Top (TAG=2) — Robin impedance
∂p/∂n + α·p = 0, α = −iωρ/Z_air, Z_air = {ref_cfg.Z_air:.1f} Pa·s/m

### Bottom outside disk (TAG=7) — Natural Neumann
∂p/∂n = 0 (rigid backing); PML-z absorbs below (outside disk column)

### PML — Complex-coordinate stretching
σ_max = {ref_cfg.sigma_max:.2e}, polynomial degree {ref_cfg.pml_degree}
xy: {ref_cfg.pml_n_wavelengths_xy:.1f}λ = {ref_cfg.t_pml_xy*1e3:.4f} mm each side
z: {ref_cfg.pml_n_wavelengths_z:.1f}λ = {ref_cfg.t_pml_z*1e3:.4f} mm bottom only (top uses Robin BC)

---

## Solver Settings

| Parameter | Value |
|-----------|-------|
| PETSc scalar type | complex128 |
| KSP type | preonly |
| PC type | lu (MUMPS direct) |
| MUMPS ICNTL(14) | 100 |
| MUMPS ICNTL(23) | 0 (auto) |
| MUMPS ICNTL(28) | 2 (parallel analysis) |
| MUMPS ICNTL(29) | 2 (ParMETIS ordering) |
| FE order | P2 Lagrange |

---

## Mesh Settings

| Parameter | Value |
|-----------|-------|
| Lx × Ly × H_total | {ref_cfg.Lx*1e3:.1f} × {ref_cfg.Ly*1e3:.1f} × {ref_cfg.H_total*1e3:.1f} mm |
| H_under / H_top | {ref_cfg.H_under*1e3:.1f} / {ref_cfg.H_top*1e3:.1f} mm |
| Frequency | {ref_cfg.frequency_hz/1e6:.2f} MHz |
| Wavelength | {ref_cfg.wavelength*1e3:.4f} mm |
| Elements/λ | {ref_cfg.elements_per_wavelength} |
| Grid | {ref_cfg.mesh_nx} × {ref_cfg.mesh_ny} × {ref_cfg.mesh_nz} |
| DOFs | {ref_sol.dofs:,} |

---

## Sampling Resolution (HI snapshots)

| Slice type | Grid |
|------------|------|
| XY snapshots | {NXY} × {NXY} |
| XZ snapshots | {NXZ_X} × {NXZ_Z} |
| Centerline | 500 points |
| Radial profile | 300 points |

---

## Per-Case Results
"""]

for case_name, sol in solutions.items():
    chk = all_checks[case_name]
    en = chk.get("energy", {})
    wn = chk.get("winding_number", "N/A")
    cnr = chk.get("central_null_ratio", "N/A")
    ksp_str = _KSP_REASONS.get(sol.ksp_converged_reason,
                                f"REASON={sol.ksp_converged_reason}")
    audit_lines.append(f"""
### {case_name}

| Metric | Value |
|--------|-------|
| Standing wave | {'OFF' if sol.cfg.standing_velocity_amplitude == 0 else f'ON ({sol.cfg.standing_velocity_amplitude*1e6:.1f} μm/s)'} |
| DOFs | {sol.dofs:,} |
| KSP reason | {sol.ksp_converged_reason} ({ksp_str}) |
| KSP iterations | {sol.ksp_iterations} |
| max|p| | {sol.max_pressure:.4f} Pa |
| Wall time | {timings[case_name]:.1f} s |
| Winding number | {wn if isinstance(wn, str) else f'{wn:.2f}'} |
| Central null ratio | {cnr if isinstance(cnr, str) else f'{cnr:.4f}'} |
| Energy physical | {en.get('physical', 'N/A'):.4e} |
| Energy PML | {en.get('pml', 'N/A'):.4e} |
| PML/phys ratio | {en.get('ratio', 'N/A'):.4f} |
""")

audit_lines.append(f"""---

## Output Files

### figures/ (HI-def 2D, {NXY}×{NXY} / {NXZ_X}×{NXZ_Z})

Per case: 5 z-heights × 3 modes (linear/log/phase) = 15 XY PNGs + 3 XZ PNGs = 18 PNGs
Total: 36 PNGs

### figures_3d/ (PyVista 3D renders)

| File | Description |
|------|-------------|
| vortex_only_isosurface.png | |p| isosurface at 30%/50%/70% of max |
| vortex_only_slicestack.png | z-slice stack + XZ midplane |
| combined_isosurface.png | Same for combined case |
| combined_slicestack.png | Same for combined case |

### vtu/ (ParaView-ready)

| File | Description |
|------|-------------|
| vortex_only.vtu | P1-interpolated fields: p_mag, p_phase, p_real, p_imag |
| combined.vtu | Same for combined case |

Open in ParaView: File → Open → select .vtu

### csv/

| File | Columns |
|------|---------|
| vortex_only_centerline_z.csv | z_m, abs_p, phase_rad, real_p, imag_p |
| vortex_only_radial_profile.csv | r_m, abs_p, phase_rad, real_p, imag_p |
| combined_centerline_z.csv | Same |
| combined_radial_profile.csv | Same |

---

## Memory

Peak RSS: ~{mem_mb:.0f} MB
""")

audit_text = "\n".join(audit_lines)
with open(OUT_DIR / "VORTEX_3D_AUDIT.md", "w") as f:
    f.write(audit_text)
print(f"  → VORTEX_3D_AUDIT.md written")

# ====================================================================
# MANIFEST.json
# ====================================================================
import platform
manifest = {
    "run_id": f"vortex_3d_hires_{TIMESTAMP}",
    "tag": "vortex_3d_hires",
    "git_branch": "LinuxTest",
    "hostname": platform.node(),
    "platform": platform.platform(),
    "python_executable": sys.executable,
    "start_time": datetime.fromtimestamp(min(timings.values())).isoformat()
        if timings else datetime.now().isoformat(),
    "end_time": datetime.now().isoformat(),
    "cases": {
        name: {
            "overrides": CASES[name],
            "dofs": sol.dofs,
            "ksp_reason": sol.ksp_converged_reason,
            "max_pressure": sol.max_pressure,
            "wall_time_s": round(timings[name], 1),
        }
        for name, sol in solutions.items()
    },
    "sampling": {
        "xy_grid": f"{NXY}x{NXY}",
        "xz_grid": f"{NXZ_X}x{NXZ_Z}",
    },
    "validation": {
        name: {k: (v if not isinstance(v, dict) else v)
               for k, v in chk.items()}
        for name, chk in all_checks.items()
    },
}
with open(OUT_DIR / "MANIFEST.json", "w") as f:
    json.dump(manifest, f, indent=2, default=str)
print(f"  → MANIFEST.json written")

print(f"\n{'='*70}")
print(f"DONE — all outputs in {OUT_DIR}")
print(f"{'='*70}")
