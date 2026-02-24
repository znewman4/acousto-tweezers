#!/usr/bin/env python3
"""
Physics Baseline + High-Resolution Visualisation Suite
=======================================================
Runs three canonical cases (standing_only, vortex_only, combined) at
moderate resolution, performs sanity / physics baseline checks, and
generates high-resolution PNGs of XY, XZ, and YZ slices plus
centerline profiles and comparison panels.

Outputs → results/baseline_hires_<timestamp>/
    figures/           high-res PNGs (300 DPI, 800×800 grid)
    csv/               numeric profiles
    baseline_report.txt   pass/fail summary

Usage:
    python scripts/experiments/physics_baseline_hires.py
"""

from __future__ import annotations

import sys, os, time, json
import numpy as np
from pathlib import Path
from datetime import datetime

# ── Project root on path ──
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ── Thread control ──
NTHREADS = str(min(os.cpu_count() or 4, 14))
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, NTHREADS)

# ── Output directories ──
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = PROJECT_ROOT / "results" / f"baseline_hires_{TIMESTAMP}"
FIG_DIR = OUT_DIR / "figures"
CSV_DIR = OUT_DIR / "csv"
FIG_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)

print(f"Output → {OUT_DIR}")

# ====================================================================
# Config
# ====================================================================
from acoustweezers.experiments.farfield_petri_cuboid.presets import (
    CORRECTED_PRESET, PETSC_MUMPS,
)
from acoustweezers.experiments.farfield_petri_cuboid.config import FarFieldConfig
from acoustweezers.experiments.farfield_petri_cuboid.solve_pressure import solve_helmholtz
from acoustweezers.experiments.farfield_petri_cuboid.post import (
    slice_xy, slice_xz, centerline_z, energy_physical_vs_pml,
)

PETSC_OPTS = {
    **PETSC_MUMPS,
    "mat_mumps_icntl_14": "100",
    "mat_mumps_icntl_23": "0",
    "mat_mumps_icntl_28": "2",
    "mat_mumps_icntl_29": "2",
}

ELEM_PER_LAMBDA = 4   # 4 elem/λ fits in memory for 3 sequential cases
DPI = 300              # high-DPI output
NGRID = 800            # interpolation grid per axis (high-res)

# Three canonical cases
CASES = {
    "standing_only": {
        "standing_velocity_amplitude": 10e-6,
        "standing_phase_pattern": "antiphase",
        "standing_axis": "both",
        "disk_velocity_amplitude": 0.0,   # vortex OFF
        "elements_per_wavelength": ELEM_PER_LAMBDA,
    },
    "vortex_only": {
        "standing_velocity_amplitude": 0.0,   # standing OFF
        "lens_focus_offset_x": 0.2e-3,
        "lens_focus_offset_y": 0.0,
        "elements_per_wavelength": ELEM_PER_LAMBDA,
    },
    "combined": {
        "standing_velocity_amplitude": 10e-6,
        "standing_phase_pattern": "antiphase",
        "standing_axis": "both",
        "lens_focus_offset_x": 0.2e-3,
        "lens_focus_offset_y": 0.0,
        "elements_per_wavelength": ELEM_PER_LAMBDA,
    },
}

# ====================================================================
# Matplotlib setup
# ====================================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.interpolate import NearestNDInterpolator

plt.rcParams.update({
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})

# ====================================================================
# Slicing helpers (return complex field for richer plots)
# ====================================================================

def complex_slice_xy(sol, z_val, n=NGRID):
    coords, pv = sol.coords, sol.p_values
    interp_re = NearestNDInterpolator(coords, np.real(pv))
    interp_im = NearestNDInterpolator(coords, np.imag(pv))
    xg = np.linspace(0, sol.cfg.Lx, n)
    yg = np.linspace(0, sol.cfg.Ly, n)
    X, Y = np.meshgrid(xg, yg)
    pts = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, z_val)])
    pc = (interp_re(pts) + 1j * interp_im(pts)).reshape(X.shape)
    return xg, yg, pc


def complex_slice_xz(sol, y_val, n=NGRID):
    coords, pv = sol.coords, sol.p_values
    interp_re = NearestNDInterpolator(coords, np.real(pv))
    interp_im = NearestNDInterpolator(coords, np.imag(pv))
    xg = np.linspace(0, sol.cfg.Lx, n)
    zg = np.linspace(0, sol.cfg.H_total, n)
    X, Z = np.meshgrid(xg, zg)
    pts = np.column_stack([X.ravel(), np.full(X.size, y_val), Z.ravel()])
    pc = (interp_re(pts) + 1j * interp_im(pts)).reshape(X.shape)
    return xg, zg, pc


def complex_slice_yz(sol, x_val, n=NGRID):
    coords, pv = sol.coords, sol.p_values
    interp_re = NearestNDInterpolator(coords, np.real(pv))
    interp_im = NearestNDInterpolator(coords, np.imag(pv))
    yg = np.linspace(0, sol.cfg.Ly, n)
    zg = np.linspace(0, sol.cfg.H_total, n)
    Y, Z = np.meshgrid(yg, zg)
    pts = np.column_stack([np.full(Y.size, x_val), Y.ravel(), Z.ravel()])
    pc = (interp_re(pts) + 1j * interp_im(pts)).reshape(Y.shape)
    return yg, zg, pc


# ====================================================================
# Plotting helpers
# ====================================================================

def plot_mag_phase(xg, yg, pc, xlabel, ylabel, title_prefix, fname_base,
                   x_scale=1e3, y_scale=1e3, aspect="equal"):
    """Generate magnitude (linear + log) and phase PNGs."""
    pmag = np.abs(pc)
    pphase = np.angle(pc)

    # --- Linear magnitude ---
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.pcolormesh(xg * x_scale, yg * y_scale, pmag,
                       shading="auto", cmap="inferno")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(f"{title_prefix} — |p|")
    if aspect: ax.set_aspect(aspect)
    plt.colorbar(im, ax=ax, label="|p| [Pa]", shrink=0.85)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{fname_base}_mag.png")
    plt.close(fig)

    # --- Log magnitude (floor at 1 Pa to avoid -inf) ---
    plog = np.log10(np.clip(pmag, 1.0, None))
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.pcolormesh(xg * x_scale, yg * y_scale, plog,
                       shading="auto", cmap="inferno")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(f"{title_prefix} — log₁₀|p|")
    if aspect: ax.set_aspect(aspect)
    plt.colorbar(im, ax=ax, label="log₁₀|p|", shrink=0.85)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{fname_base}_logmag.png")
    plt.close(fig)

    # --- Phase ---
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.pcolormesh(xg * x_scale, yg * y_scale, pphase,
                       shading="auto", cmap="twilight",
                       vmin=-np.pi, vmax=np.pi)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(f"{title_prefix} — arg(p)")
    if aspect: ax.set_aspect(aspect)
    plt.colorbar(im, ax=ax, label="Phase [rad]", shrink=0.85)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{fname_base}_phase.png")
    plt.close(fig)


# ====================================================================
# SOLVE ALL CASES — extract arrays and release FEniCSx memory
# ====================================================================
import gc

# Store lightweight data only (coords + pressure values + config)
class LightSolution:
    """Minimal container holding only numpy arrays and config."""
    def __init__(self, sol):
        self.coords = sol.coords.copy()
        self.p_values = sol.p_values.copy()  # already returns a copy
        self.cfg = sol.cfg
        self.dofs = sol.dofs
        self.ksp_converged_reason = sol.ksp_converged_reason
        self.ksp_iterations = sol.ksp_iterations
        self.ksp_residual_norm = sol.ksp_residual_norm
        self.max_pressure = sol.max_pressure

solutions = {}
metrics = {}
report_lines = []

def log(msg):
    print(msg)
    report_lines.append(msg)

log("=" * 72)
log("PHYSICS BASELINE + HIGH-RES VISUALISATION")
log(f"Timestamp: {TIMESTAMP}")
log(f"Resolution: {ELEM_PER_LAMBDA} elem/λ, grid {NGRID}×{NGRID}, {DPI} DPI")
log("=" * 72)

for case_name, overrides in CASES.items():
    log(f"\n{'─'*72}")
    log(f"  Case: {case_name}")
    log(f"{'─'*72}")

    cfg_dict = {**CORRECTED_PRESET, **overrides}
    cfg = FarFieldConfig(**cfg_dict)

    t0 = time.time()
    sol = solve_helmholtz(cfg, verbose=True, petsc_options=PETSC_OPTS,
                          export_fields=False)
    dt = time.time() - t0

    # Compute energy split before we discard the full solution object
    energy = energy_physical_vs_pml(sol)

    # Extract lightweight arrays and free FEniCSx/PETSc memory
    lsol = LightSolution(sol)
    del sol
    gc.collect()

    solutions[case_name] = lsol

    # Metrics
    pv = lsol.p_values
    maxp = lsol.max_pressure
    meanp = float(np.mean(np.abs(pv)))

    m = {
        "wall_time_s": round(dt, 1),
        "dofs": lsol.dofs,
        "ksp_reason": lsol.ksp_converged_reason,
        "ksp_iters": lsol.ksp_iterations,
        "ksp_resid": float(lsol.ksp_residual_norm),
        "max_p_Pa": round(maxp, 2),
        "mean_p_Pa": round(meanp, 2),
        "energy_phys": energy["physical"],
        "energy_pml": energy["pml"],
        "energy_ratio_pml_phys": round(energy["ratio"], 4),
        "has_nan": bool(np.any(np.isnan(pv))),
    }
    metrics[case_name] = m

    log(f"  DOFs         : {m['dofs']}")
    log(f"  Wall time    : {m['wall_time_s']:.1f} s")
    log(f"  KSP reason   : {m['ksp_reason']}  (>0 = converged)")
    log(f"  KSP iters    : {m['ksp_iters']}")
    log(f"  max|p|       : {m['max_p_Pa']:.2f} Pa")
    log(f"  mean|p|      : {m['mean_p_Pa']:.2f} Pa")
    log(f"  E_pml/E_phys : {m['energy_ratio_pml_phys']:.4f}")
    log(f"  NaN?         : {m['has_nan']}")

# ====================================================================
# PHYSICS BASELINE CHECKS
# ====================================================================
log("\n" + "=" * 72)
log("PHYSICS BASELINE CHECKS")
log("=" * 72)

checks = {}

# 1) All solvers converged
for cn, m in metrics.items():
    key = f"{cn}_converged"
    ok = m["ksp_reason"] > 0
    checks[key] = ok
    log(f"  [{'PASS' if ok else 'FAIL'}] {cn}: solver converged (reason={m['ksp_reason']})")

# 2) No NaN in any solution
for cn, m in metrics.items():
    key = f"{cn}_no_nan"
    ok = not m["has_nan"]
    checks[key] = ok
    log(f"  [{'PASS' if ok else 'FAIL'}] {cn}: no NaN")

# 3) max|p| > 0 for all cases
for cn, m in metrics.items():
    key = f"{cn}_nonzero"
    ok = m["max_p_Pa"] > 0
    checks[key] = ok
    log(f"  [{'PASS' if ok else 'FAIL'}] {cn}: max|p| > 0  ({m['max_p_Pa']:.2f} Pa)")

# 4) Standing-only should have NO vortex phase singularity
#    (phase at trap plane should be smooth, not winding)
cfg_s = solutions["standing_only"].cfg
trap_z = cfg_s.H_under + cfg_s.H_top / 2
_, _, pc_stand_trap = complex_slice_xy(solutions["standing_only"], trap_z, n=200)
phase_s = np.angle(pc_stand_trap)
# Approximate winding: sum of angular differences around a small circle
cx_i, cy_i = 100, 100  # center pixel
radius_px = 20
theta = np.linspace(0, 2*np.pi, 64, endpoint=False)
ix = np.clip((cx_i + radius_px * np.cos(theta)).astype(int), 0, 199)
iy = np.clip((cy_i + radius_px * np.sin(theta)).astype(int), 0, 199)
dphase = np.diff(np.unwrap(phase_s[iy, ix]))
winding_stand = abs(np.sum(dphase)) / (2*np.pi)
ok = winding_stand < 0.5  # should be ~0 for standing-only
checks["standing_no_vortex"] = ok
log(f"  [{'PASS' if ok else 'FAIL'}] standing_only: winding = {winding_stand:.2f} (expect ~0)")

# 5) Vortex-only should have winding ~1
_, _, pc_vort_trap = complex_slice_xy(solutions["vortex_only"], trap_z, n=200)
phase_v = np.angle(pc_vort_trap)
ix2 = np.clip((cx_i + radius_px * np.cos(theta)).astype(int), 0, 199)
iy2 = np.clip((cy_i + radius_px * np.sin(theta)).astype(int), 0, 199)
dphase_v = np.diff(np.unwrap(phase_v[iy2, ix2]))
winding_vort = abs(np.sum(dphase_v)) / (2*np.pi)
ok = 0.7 < winding_vort < 1.5
checks["vortex_winding"] = ok
log(f"  [{'PASS' if ok else 'FAIL'}] vortex_only: winding = {winding_vort:.2f} (expect ~1)")

# 6) Combined max|p| should differ from standing-only (interaction)
mp_stand = metrics["standing_only"]["max_p_Pa"]
mp_comb = metrics["combined"]["max_p_Pa"]
rel_diff = abs(mp_comb - mp_stand) / (mp_stand + 1e-30)
ok = rel_diff > 0.01  # at least 1% difference
checks["combined_interaction"] = ok
log(f"  [{'PASS' if ok else 'FAIL'}] combined ≠ standing  (Δ = {rel_diff*100:.1f}%)")

# 7) PML energy ratio should be small (< 0.5 for physical confinement)
for cn in CASES:
    key = f"{cn}_pml_energy"
    r = metrics[cn]["energy_ratio_pml_phys"]
    ok = r < 0.5
    checks[key] = ok
    log(f"  [{'PASS' if ok else 'FAIL'}] {cn}: PML energy ratio = {r:.4f} (< 0.5)")

n_pass = sum(checks.values())
n_total = len(checks)
log(f"\n  Baseline: {n_pass}/{n_total} checks passed")

# ====================================================================
# HIGH-RES PNG GENERATION
# ====================================================================
log("\n" + "=" * 72)
log("GENERATING HIGH-RES PNGs")
log("=" * 72)

png_count = 0

for case_name, sol in solutions.items():
    cfg = sol.cfg
    trap_z = cfg.H_under + cfg.H_top / 2
    y_mid = cfg.Ly / 2
    x_mid = cfg.Lx / 2

    # --- XY at trap plane ---
    log(f"\n  {case_name}: XY slice at z = {trap_z*1e3:.1f} mm (trap plane)")
    xg, yg, pc_xy = complex_slice_xy(sol, trap_z)
    plot_mag_phase(xg, yg, pc_xy, "x [mm]", "y [mm]",
                   f"{case_name} — XY z={trap_z*1e3:.1f}mm",
                   f"{case_name}_xy_trap")
    png_count += 3

    # --- XY at bath midplane ---
    z_bath = cfg.H_under / 2
    log(f"  {case_name}: XY slice at z = {z_bath*1e3:.1f} mm (bath mid)")
    xg, yg, pc_xy2 = complex_slice_xy(sol, z_bath)
    plot_mag_phase(xg, yg, pc_xy2, "x [mm]", "y [mm]",
                   f"{case_name} — XY z={z_bath*1e3:.1f}mm",
                   f"{case_name}_xy_bath")
    png_count += 3

    # --- XZ mid-plane ---
    log(f"  {case_name}: XZ slice at y = {y_mid*1e3:.1f} mm")
    xg, zg, pc_xz = complex_slice_xz(sol, y_mid)
    plot_mag_phase(xg, zg, pc_xz, "x [mm]", "z [mm]",
                   f"{case_name} — XZ y={y_mid*1e3:.1f}mm",
                   f"{case_name}_xz_mid", aspect=None)
    png_count += 3

    # --- YZ mid-plane ---
    log(f"  {case_name}: YZ slice at x = {x_mid*1e3:.1f} mm")
    yg, zg, pc_yz = complex_slice_yz(sol, x_mid)
    plot_mag_phase(yg, zg, pc_yz, "y [mm]", "z [mm]",
                   f"{case_name} — YZ x={x_mid*1e3:.1f}mm",
                   f"{case_name}_yz_mid", aspect=None)
    png_count += 3

    # --- Centerline profile ---
    log(f"  {case_name}: centerline z-profile")
    zg_cl, pmag_cl = centerline_z(sol, nz=800)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(zg_cl * 1e3, pmag_cl, linewidth=1.5)
    ax.axvspan(cfg.H_under * 1e3, (cfg.H_under + cfg.H_top) * 1e3,
               alpha=0.15, color="red", label="Petri slab")
    ax.set_xlabel("z [mm]"); ax.set_ylabel("|p| [Pa]")
    ax.set_title(f"{case_name} — centerline |p|(z)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{case_name}_centerline_z.png")
    plt.close(fig)
    png_count += 1

    # Save CSV
    np.savetxt(CSV_DIR / f"{case_name}_centerline_z.csv",
               np.column_stack([zg_cl, pmag_cl]),
               header="z_m, |p|_Pa", delimiter=",")

# ====================================================================
# COMPARISON PANELS
# ====================================================================
log("\n  Generating comparison panels …")

# --- 3-panel XY comparison at trap plane ---
fig, axes = plt.subplots(1, 3, figsize=(22, 7))
for ax, cn in zip(axes, ["standing_only", "vortex_only", "combined"]):
    sol_c = solutions[cn]
    trap_z_c = sol_c.cfg.H_under + sol_c.cfg.H_top / 2
    xg, yg, pc = complex_slice_xy(sol_c, trap_z_c, n=NGRID)
    pmag = np.abs(pc)
    im = ax.pcolormesh(xg * 1e3, yg * 1e3, pmag, shading="auto", cmap="inferno")
    ax.set_title(cn.replace("_", " ").title())
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax, label="|p| [Pa]", shrink=0.85)
fig.suptitle("XY Trap-Plane Comparison — |p|", fontsize=15, y=1.01)
fig.tight_layout()
fig.savefig(FIG_DIR / "comparison_xy_trap_mag.png", bbox_inches="tight")
plt.close(fig)
png_count += 1

# --- 3-panel phase comparison ---
fig, axes = plt.subplots(1, 3, figsize=(22, 7))
for ax, cn in zip(axes, ["standing_only", "vortex_only", "combined"]):
    sol_c = solutions[cn]
    trap_z_c = sol_c.cfg.H_under + sol_c.cfg.H_top / 2
    xg, yg, pc = complex_slice_xy(sol_c, trap_z_c, n=NGRID)
    pphase = np.angle(pc)
    im = ax.pcolormesh(xg * 1e3, yg * 1e3, pphase, shading="auto",
                       cmap="twilight", vmin=-np.pi, vmax=np.pi)
    ax.set_title(cn.replace("_", " ").title())
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax, label="Phase [rad]", shrink=0.85)
fig.suptitle("XY Trap-Plane Comparison — Phase", fontsize=15, y=1.01)
fig.tight_layout()
fig.savefig(FIG_DIR / "comparison_xy_trap_phase.png", bbox_inches="tight")
plt.close(fig)
png_count += 1

# --- 3-panel XZ comparison ---
fig, axes = plt.subplots(1, 3, figsize=(22, 7))
for ax, cn in zip(axes, ["standing_only", "vortex_only", "combined"]):
    sol_c = solutions[cn]
    y_mid_c = sol_c.cfg.Ly / 2
    xg, zg, pc = complex_slice_xz(sol_c, y_mid_c, n=NGRID)
    pmag = np.abs(pc)
    im = ax.pcolormesh(xg * 1e3, zg * 1e3, pmag, shading="auto", cmap="inferno")
    ax.set_title(cn.replace("_", " ").title())
    ax.set_xlabel("x [mm]"); ax.set_ylabel("z [mm]")
    plt.colorbar(im, ax=ax, label="|p| [Pa]", shrink=0.85)
fig.suptitle("XZ Mid-Plane Comparison — |p|", fontsize=15, y=1.01)
fig.tight_layout()
fig.savefig(FIG_DIR / "comparison_xz_mid_mag.png", bbox_inches="tight")
plt.close(fig)
png_count += 1

# --- Centerline overlay ---
fig, ax = plt.subplots(figsize=(10, 6))
for cn in ["standing_only", "vortex_only", "combined"]:
    zg_cl, pmag_cl = centerline_z(solutions[cn], nz=800)
    ax.plot(zg_cl * 1e3, pmag_cl, linewidth=1.5, label=cn.replace("_", " ").title())
cfg0 = solutions["standing_only"].cfg
ax.axvspan(cfg0.H_under * 1e3, (cfg0.H_under + cfg0.H_top) * 1e3,
           alpha=0.12, color="red", label="Petri slab")
ax.set_xlabel("z [mm]"); ax.set_ylabel("|p| [Pa]")
ax.set_title("Centerline |p|(z) — All Cases")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "comparison_centerline_z.png")
plt.close(fig)
png_count += 1

# ====================================================================
# Save report
# ====================================================================
log(f"\n  Total PNGs generated: {png_count}")
log(f"\n  Output directory: {OUT_DIR}")

# Write metrics JSON
with open(OUT_DIR / "metrics.json", "w") as fh:
    json.dump(metrics, fh, indent=2)

# Write baseline report
with open(OUT_DIR / "baseline_report.txt", "w") as fh:
    fh.write("\n".join(report_lines))

# Write checks JSON
with open(OUT_DIR / "baseline_checks.json", "w") as fh:
    json.dump({k: bool(v) for k, v in checks.items()}, fh, indent=2)

log("\nDone.")

# Exit code
sys.exit(0 if all(checks.values()) else 1)
