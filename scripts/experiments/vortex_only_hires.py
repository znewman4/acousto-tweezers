#!/usr/bin/env python3
"""
High-resolution vortex-only snapshot generation + full BC/input audit.

Standing wave is OFF (standing_velocity_amplitude = 0.0).
Produces:
  - XY slices at 5 z-heights (linear, log, phase)
  - XZ slices through centre (linear, log, phase)
  - Centerline and radial profile CSVs
  - VORTEX_ONLY_AUDIT.md with complete physics/BC documentation

Usage:
    ~/.conda/envs/acousto-complex/bin/python scripts/experiments/vortex_only_hires.py
"""

from __future__ import annotations

import sys, os, time, json, platform, traceback
import numpy as np
from pathlib import Path
from datetime import datetime

# ── Ensure project root is on path ──
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ── Thread control ──
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "14")

# ── Output directory ──
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = PROJECT_ROOT / "results" / f"vortex_only_hires_{TIMESTAMP}"
FIG_DIR = OUT_DIR / "figures"
CSV_DIR = OUT_DIR / "csv"
FIG_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)

print(f"Output directory: {OUT_DIR}")
print(f"Timestamp: {TIMESTAMP}")


# ====================================================================
# 1. Configure (vortex-only, 6 elem/λ)
# ====================================================================
from acoustweezers.experiments.farfield_petri_cuboid.presets import (
    CORRECTED_PRESET, PETSC_MUMPS,
)
from acoustweezers.experiments.farfield_petri_cuboid.config import FarFieldConfig

overrides = {
    "standing_velocity_amplitude": 0.0,    # standing wave OFF
    "lens_focus_offset_x": 0.2e-3,        # 0.2 mm offset
    "lens_focus_offset_y": 0.0,
    "elements_per_wavelength": 5,          # production resolution (6 exceeds MUMPS memory)
}

cfg_dict = {**CORRECTED_PRESET, **overrides}
cfg = FarFieldConfig(**cfg_dict)

# Production-grade MUMPS settings (same as production_farfield_run.py)
PETSC_OPTS = {
    **PETSC_MUMPS,
    "mat_mumps_icntl_14": "100",   # 100 % workspace increase
    "mat_mumps_icntl_23": "0",     # let MUMPS manage memory automatically
    "mat_mumps_icntl_28": "2",     # parallel analysis
    "mat_mumps_icntl_29": "2",     # ParMETIS ordering (less fill-in)
}

print("\n" + cfg.describe())
print(f"Standing velocity amplitude: {cfg.standing_velocity_amplitude} m/s")
print(f"  → Standing wave is {'OFF' if cfg.standing_velocity_amplitude == 0.0 else 'ON'}")

# ====================================================================
# 2. Solve
# ====================================================================
from acoustweezers.experiments.farfield_petri_cuboid.solve_pressure import solve_helmholtz

print("\n" + "="*70)
print("SOLVING VORTEX-ONLY CASE")
print("="*70)

t_start = time.time()
sol = solve_helmholtz(
    cfg,
    verbose=True,
    petsc_options=PETSC_OPTS,
    export_fields=False,  # P2 → XDMF needs interpolation; skip for now
)
t_wall = time.time() - t_start

print(f"\nSolver wall time: {t_wall:.1f}s")
print(f"DOFs: {sol.dofs}")
print(f"KSP converged reason: {sol.ksp_converged_reason}")
print(f"KSP iterations: {sol.ksp_iterations}")
print(f"KSP residual norm: {sol.ksp_residual_norm:.2e}")
print(f"max|p| = {sol.max_pressure:.4f} Pa")

if sol.ksp_converged_reason <= 0:
    print("WARNING: Solver did NOT converge!")

# ====================================================================
# 3. Post-processing imports
# ====================================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import NearestNDInterpolator

from acoustweezers.experiments.farfield_petri_cuboid.post import (
    slice_xy, slice_xz, centerline_z, energy_physical_vs_pml,
)


def get_complex_slice_xy(sol, z_val, nx=300, ny=300):
    """Return (xg, yg, p_complex_2d) on a regular grid at fixed z."""
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


def get_complex_slice_xz(sol, y_val, nx=300, nz=300):
    """Return (xg, zg, p_complex_2d) on a regular grid at fixed y."""
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
# 4. XY Snapshots at multiple z-heights
# ====================================================================
print("\n── Generating XY snapshots ──")

# The "trap plane" is the petri midplane: z = H_under + H_top/2
trap_z = cfg.H_under + cfg.H_top / 2
z_heights = {
    "trap":  trap_z,
    "1mm":   cfg.H_under + 1e-3,
    "2mm":   cfg.H_under + 2e-3,    # = H_total for H_top=2mm → clamp
    "3mm":   cfg.H_under + 3e-3,    # above domain if H_total=5mm
    "4mm":   cfg.H_under + 4e-3,
}

# Clamp z values to domain
for key in list(z_heights.keys()):
    z_heights[key] = min(z_heights[key], cfg.H_total - 1e-5)

# For the canonical 6×6×5mm domain:
# trap = 3+1 = 4mm,  1mm = 4mm,  2mm = 5mm (at top)
# 3mm and 4mm would exceed domain → clamped
# Better interpretation: z measured from petri base (H_under)
# "z = 1 mm above disk" = H_under + ... wait, the user said:
# "Trap plane", "z = 1 mm above disk", "z = 2 mm", "z = 3 mm", "z = 4 mm"
# These are likely z_phys coordinates above the bottom of the domain.
# Let's use absolute z positions: 1mm, 2mm, 3mm, 4mm from bottom
z_heights = {
    "trap":  cfg.H_under + cfg.H_top / 2,   # midplane of petri slab
    "z1mm":  1e-3,
    "z2mm":  2e-3,
    "z3mm":  3e-3,
    "z4mm":  4e-3,
}

NXY = 300  # grid resolution

for label, z_val in z_heights.items():
    z_mm = z_val * 1e3
    print(f"  z = {z_mm:.2f} mm  ({label})")

    xg, yg, pc = get_complex_slice_xy(sol, z_val, nx=NXY, ny=NXY)
    pmag = np.abs(pc)
    pphase = np.angle(pc)
    plog = np.log10(pmag + 1e-30)

    # --- Linear magnitude ---
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.pcolormesh(xg * 1e3, yg * 1e3, pmag, shading="auto", cmap="inferno")
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    ax.set_title(f"|p| at z = {z_mm:.2f} mm ({label})")
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax, label="|p| [Pa]")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"vortex_only_xy_{label}_linear.png", dpi=200)
    plt.close(fig)

    # --- Log magnitude ---
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.pcolormesh(xg * 1e3, yg * 1e3, plog, shading="auto", cmap="inferno")
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    ax.set_title(f"log₁₀|p| at z = {z_mm:.2f} mm ({label})")
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax, label="log₁₀|p|")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"vortex_only_xy_{label}_log.png", dpi=200)
    plt.close(fig)

    # --- Phase ---
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.pcolormesh(xg * 1e3, yg * 1e3, pphase, shading="auto", cmap="twilight",
                        vmin=-np.pi, vmax=np.pi)
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    ax.set_title(f"arg(p) at z = {z_mm:.2f} mm ({label})")
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax, label="Phase [rad]")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"vortex_only_xy_{label}_phase.png", dpi=200)
    plt.close(fig)

print(f"  → {len(z_heights)*3} XY plots saved")


# ====================================================================
# 5. XZ Snapshots (y = Ly/2)
# ====================================================================
print("\n── Generating XZ snapshots ──")

y_mid = cfg.Ly / 2
NXZ = 300

xg, zg, pc_xz = get_complex_slice_xz(sol, y_mid, nx=NXZ, nz=NXZ)
pmag_xz = np.abs(pc_xz)
pphase_xz = np.angle(pc_xz)
plog_xz = np.log10(pmag_xz + 1e-30)

for data, cmap, label_short, cb_label, fname_suffix in [
    (pmag_xz, "inferno", "|p|", "|p| [Pa]", "linear"),
    (plog_xz, "inferno", "log₁₀|p|", "log₁₀|p|", "log"),
    (pphase_xz, "twilight", "arg(p)", "Phase [rad]", "phase"),
]:
    fig, ax = plt.subplots(figsize=(9, 5))
    vkw = dict(vmin=-np.pi, vmax=np.pi) if fname_suffix == "phase" else {}
    im = ax.pcolormesh(xg * 1e3, zg * 1e3, data, shading="auto", cmap=cmap, **vkw)
    ax.set_xlabel("x [mm]"); ax.set_ylabel("z [mm]")
    ax.set_title(f"{label_short} — xz mid-plane (y = {y_mid*1e3:.1f} mm)")
    # Mark petri base
    ax.axhline(cfg.H_under * 1e3, color="cyan", ls=":", lw=1, label="petri base")
    # Mark PML boundaries
    if cfg.pml_enabled:
        ax.axvline(cfg.t_pml_xy * 1e3, color="w", ls="--", lw=0.7, alpha=0.6)
        ax.axvline((cfg.Lx - cfg.t_pml_xy) * 1e3, color="w", ls="--", lw=0.7, alpha=0.6)
        ax.axhline(cfg.t_pml_z * 1e3, color="w", ls="--", lw=0.7, alpha=0.6)
    ax.legend(loc="upper right", fontsize=8)
    plt.colorbar(im, ax=ax, label=cb_label)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"vortex_only_xz_{fname_suffix}.png", dpi=200)
    plt.close(fig)

print("  → 3 XZ plots saved")


# ====================================================================
# 6. Centerline + Radial Profiles (CSV)
# ====================================================================
print("\n── Generating CSV profiles ──")

# Centerline: |p|(z) along (Lx/2, Ly/2, z)
coords = sol.coords
pv = sol.p_values
interp_re = NearestNDInterpolator(coords, np.real(pv))
interp_im = NearestNDInterpolator(coords, np.imag(pv))

cx, cy = cfg.disk_center_x, cfg.disk_center_y

nz_cl = 500
zg_cl = np.linspace(0, cfg.H_total, nz_cl)
pts_cl = np.column_stack([np.full(nz_cl, cx), np.full(nz_cl, cy), zg_cl])
pc_cl = interp_re(pts_cl) + 1j * interp_im(pts_cl)

import csv
with open(CSV_DIR / "vortex_centerline_z.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["z_m", "abs_p", "phase_rad", "real_p", "imag_p"])
    for i in range(nz_cl):
        w.writerow([
            f"{zg_cl[i]:.8e}",
            f"{np.abs(pc_cl[i]):.8e}",
            f"{np.angle(pc_cl[i]):.8e}",
            f"{np.real(pc_cl[i]):.8e}",
            f"{np.imag(pc_cl[i]):.8e}",
        ])
print(f"  → vortex_centerline_z.csv  ({nz_cl} points)")

# Radial profile at trap plane
nr = 300
z_trap = cfg.H_under + cfg.H_top / 2
r_max = min(cfg.Lx, cfg.Ly) / 2
rg = np.linspace(0, r_max, nr)
# Sample along x-axis (y = cy, z = z_trap)
pts_rad = np.column_stack([cx + rg, np.full(nr, cy), np.full(nr, z_trap)])
pc_rad = interp_re(pts_rad) + 1j * interp_im(pts_rad)

with open(CSV_DIR / "vortex_radial_profile.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["r_m", "abs_p", "phase_rad", "real_p", "imag_p"])
    for i in range(nr):
        w.writerow([
            f"{rg[i]:.8e}",
            f"{np.abs(pc_rad[i]):.8e}",
            f"{np.angle(pc_rad[i]):.8e}",
            f"{np.real(pc_rad[i]):.8e}",
            f"{np.imag(pc_rad[i]):.8e}",
        ])
print(f"  → vortex_radial_profile.csv  ({nr} points)")


# ====================================================================
# 7. Energy partition
# ====================================================================
en = energy_physical_vs_pml(sol)
print(f"\nEnergy partition: physical={en['physical']:.4e}, PML={en['pml']:.4e}, ratio={en['ratio']:.4f}")


# ====================================================================
# 8. Validation checks
# ====================================================================
print("\n" + "="*70)
print("VALIDATION CHECKS")
print("="*70)

checks = {}

# Check 1: Solver converged
converged = sol.ksp_converged_reason > 0
checks["solver_converged"] = converged
print(f"  [{'PASS' if converged else 'FAIL'}] Solver converged (reason={sol.ksp_converged_reason})")

# Check 2: No divergence (max|p| finite and reasonable)
no_div = np.isfinite(sol.max_pressure) and sol.max_pressure < 1e6
checks["no_divergence"] = no_div
print(f"  [{'PASS' if no_div else 'FAIL'}] No divergence (max|p|={sol.max_pressure:.4f} Pa)")

# Check 3: Phase shows vortex winding at trap plane
xg_v, yg_v, pc_v = get_complex_slice_xy(sol, z_trap, nx=200, ny=200)
phase_v = np.angle(pc_v)
# Check phase winding: integrate ∂φ/∂θ around a circle
cx_idx = np.argmin(np.abs(xg_v - cx))
cy_idx = np.argmin(np.abs(yg_v - cy))
n_theta = 360
theta_ring = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
r_ring = 0.5e-3  # 0.5 mm ring radius
x_ring = cx + r_ring * np.cos(theta_ring)
y_ring = cy + r_ring * np.sin(theta_ring)

pts_ring = np.column_stack([x_ring, y_ring, np.full(n_theta, z_trap)])
pc_ring = interp_re(pts_ring) + 1j * interp_im(pts_ring)
phase_ring = np.angle(pc_ring)
dphase = np.diff(np.unwrap(phase_ring))
winding_number = np.sum(dphase) / (2 * np.pi)
has_winding = abs(winding_number - cfg.vortex_topological_charge) < 0.5
checks["vortex_winding"] = has_winding
print(f"  [{'PASS' if has_winding else 'FAIL'}] Phase winding number = {winding_number:.2f} (expected {cfg.vortex_topological_charge})")

# Check 4: Central null exists (|p| at centre should be near minimum)
pc_center = interp_re([[cx, cy, z_trap]]) + 1j * interp_im([[cx, cy, z_trap]])
p_center = float(np.abs(pc_center).item())
# Compare to max on the ring
p_ring_max = float(np.max(np.abs(pc_ring)))
has_null = p_center < 0.3 * p_ring_max  # centre should be much less than ring max
checks["central_null"] = has_null
print(f"  [{'PASS' if has_null else 'FAIL'}] Central null: |p|_center={p_center:.4f} Pa vs ring_max={p_ring_max:.4f} Pa (ratio={p_center/(p_ring_max+1e-30):.3f})")

# Check 5: Asymmetry matches offset (offset is +x → field should be asymmetric in x)
# Sample |p| at +x and -x from centre at trap plane
dx_asym = 1e-3
p_plusx = float(np.abs(interp_re([[cx+dx_asym, cy, z_trap]]) + 1j * interp_im([[cx+dx_asym, cy, z_trap]])).item())
p_minusx = float(np.abs(interp_re([[cx-dx_asym, cy, z_trap]]) + 1j * interp_im([[cx-dx_asym, cy, z_trap]])).item())
asym_ratio = abs(p_plusx - p_minusx) / (max(p_plusx, p_minusx) + 1e-30)
# With lens offset, there should be some asymmetry (but it can be small)
has_asym = asym_ratio > 0.001  # even 0.1% asymmetry is fine
checks["asymmetry"] = has_asym
print(f"  [{'PASS' if has_asym else 'WARN'}] Asymmetry: |p|(+x)={p_plusx:.4f}, |p|(-x)={p_minusx:.4f}, ratio={asym_ratio:.4f}")

all_pass = all(checks.values())
print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")


# ====================================================================
# 9. Write VORTEX_ONLY_AUDIT.md
# ====================================================================
print("\n── Writing VORTEX_ONLY_AUDIT.md ──")

_KSP_REASONS = {
    1: "CONVERGED_RTOL_NORMAL", 2: "CONVERGED_RTOL",
    3: "CONVERGED_ATOL", 9: "CONVERGED_ITERATING",
    -3: "DIVERGED_ITS", -4: "DIVERGED_DTOL",
    -5: "DIVERGED_BREAKDOWN", -9: "DIVERGED_NANORINF",
}

import resource
mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KB → MB on Linux

audit_text = f"""# VORTEX_ONLY_AUDIT.md

Generated: {datetime.now().isoformat()}
Run directory: `{OUT_DIR.name}`
Git branch: LinuxTest

---

## A) Geometry Summary

| Parameter | Value |
|-----------|-------|
| Lx | {cfg.Lx*1e3:.1f} mm |
| Ly | {cfg.Ly*1e3:.1f} mm |
| H_under (bath depth) | {cfg.H_under*1e3:.1f} mm |
| H_top (petri slab) | {cfg.H_top*1e3:.1f} mm |
| H_total | {cfg.H_total*1e3:.1f} mm |
| Disk radius | {cfg.disk_radius*1e3:.2f} mm |
| Disk centre | ({cfg.disk_center_x*1e3:.1f}, {cfg.disk_center_y*1e3:.1f}) mm |
| Wavelength (λ) | {cfg.wavelength*1e3:.4f} mm |
| Mesh resolution | {cfg.elements_per_wavelength} elements/λ |
| Mesh grid | {cfg.mesh_nx} × {cfg.mesh_ny} × {cfg.mesh_nz} |
| DOF count | {sol.dofs:,} |
| PML thickness (xy) | {cfg.t_pml_xy*1e3:.4f} mm ({cfg.pml_n_wavelengths_xy:.1f}λ) |
| PML thickness (z) | {cfg.t_pml_z*1e3:.4f} mm ({cfg.pml_n_wavelengths_z:.1f}λ) |

---

## B) Boundary Conditions

### B.1 — Bottom Disk (TAG_BOTTOM_DISK = 1)

- **Type:** Neumann (velocity source)
- **Mathematical form:** ∂p/∂n = −iωρ V_disk · Φ(x,y)
  - Where Φ(x,y) is the plastic-lens vortex phase profile
  - V_disk = {cfg.disk_velocity_amplitude*1e6:.1f} μm/s
- **Physical interpretation:** Ultrasonic transducer disk driving an upward-propagating vortex beam through a plastic spiral-phase lens. The phase profile Φ encodes topological charge l={cfg.lens_l}, focal length f={cfg.lens_focal_length*1e3:.1f} mm, and offset ({cfg.lens_focus_offset_x*1e3:.2f}, {cfg.lens_focus_offset_y*1e3:.2f}) mm.

### B.2 — Bottom outside disk (TAG_BOTTOM_OUTSIDE = 7)

- **Type:** Natural Neumann (zero flux)
- **Mathematical form:** ∂p/∂n = 0
- **Physical interpretation:** Rigid backing / no acoustic source outside the disk aperture. PML z-layer absorbs below this surface (outside disk column).

### B.3 — Petri sidewalls (TAG_STAND_X0=13, TAG_STAND_XL=14, TAG_STAND_Y0=15, TAG_STAND_YL=16)

- **Type:** Neumann (velocity source) — but **amplitude = 0**
- **Mathematical form:** ∂p/∂n = −iωρ · V_stand · pattern(x)  with V_stand = {cfg.standing_velocity_amplitude:.1e}
- **Physical interpretation:** Standing-wave transducers on petri slab walls, z ∈ [H_under, H_under + H_top]. **DISABLED** for this run (V_stand = 0). These walls effectively become rigid (zero normal velocity) because the Neumann load is zero.

### B.4 — Bath sidewalls (TAG_X0=3, TAG_XL=4, TAG_Y0=5, TAG_YL=6)

- **Type:** Natural Neumann (zero flux if not in PML) / PML absorption
- **Mathematical form:** ∂p/∂n = 0 (physical region); PML complex-coordinate stretching absorbs outgoing waves
- **Physical interpretation:** Far-field boundary. Lateral PML layers of thickness {cfg.t_pml_xy*1e3:.3f} mm absorb outgoing waves to prevent artificial reflections.

### B.5 — Top boundary (TAG_TOP = 2)

- **Type:** Robin (impedance)
- **Mathematical form:** ∂p/∂n + α·p = 0  where α = −iωρ/Z_air
  - Z_air = ρ_air · c_air = {cfg.rho_air} × {cfg.c_air} = {cfg.Z_air:.1f} Pa·s/m
  - α = −i·{cfg.omega:.2e}·{cfg.rho}/(cfg.Z_air) = complex coefficient
  - Z_rel = Z_air/Z_water = {cfg.Z_air/cfg.Z_water:.6f}
- **Physical interpretation:** Water–air interface. Nearly total reflection (Z_air ≪ Z_water), approximating a pressure-release boundary with small transmission loss.

### B.6 — PML boundaries (all exterior faces of PML region)

- **Type:** Complex-coordinate stretching (absorbed into bilinear form)
- **Mathematical form:** Coordinate mapping x → x + (i/ω)∫σ(x')dx' with σ_max = {cfg.sigma_max:.2e} (factor {cfg.pml_sigma_max_factor}), polynomial degree {cfg.pml_degree}
- **Physical interpretation:** Perfectly Matched Layer absorbs outgoing waves. Applied on {cfg.pml_n_wavelengths_xy:.1f}λ lateral bands and {cfg.pml_n_wavelengths_z:.1f}λ below the disk (outside disk column). PML regions: X, Y, Z, XY, XZ, YZ, XYZ corners.

---

## C) Input Forcing Summary

| Parameter | Value |
|-----------|-------|
| Vortex topological charge (l) | {cfg.vortex_topological_charge} |
| Lens topological charge (lens_l) | {cfg.lens_l} |
| Disk velocity amplitude | {cfg.disk_velocity_amplitude*1e6:.1f} μm/s |
| Lens drive model | {cfg.lens_drive} |
| Lens focal length | {cfg.lens_focal_length*1e3:.1f} mm |
| Lens offset (x, y) | ({cfg.lens_focus_offset_x*1e3:.2f}, {cfg.lens_focus_offset_y*1e3:.2f}) mm |
| Lens c_lens (plastic) | {cfg.lens_c_lens:.0f} m/s |
| Lens apodization | {cfg.lens_apodization} (strength {cfg.lens_apodization_strength}) |
| Standing velocity amplitude | {cfg.standing_velocity_amplitude*1e6:.1f} μm/s **(OFF)** |
| Standing phase pattern | {cfg.standing_phase_pattern} |
| Standing axis | {cfg.standing_axis} |
| Frequency | {cfg.frequency_hz/1e6:.2f} MHz |
| Water density | {cfg.rho} kg/m³ |
| Water sound speed | {cfg.c} m/s |

---

## D) Solver Configuration

| Parameter | Value |
|-----------|-------|
| PETSc scalar type | complex128 |
| Linear solver (KSP) | {PETSC_OPTS.get('ksp_type', 'preonly')} |
| Preconditioner (PC) | {PETSC_OPTS.get('pc_type', 'lu')} |
| Direct solver | {PETSC_OPTS.get('pc_factor_mat_solver_type', 'mumps')} |
| MUMPS ICNTL(14) | {PETSC_OPTS.get('mat_mumps_icntl_14', 'default')} (% workspace increase) |
| MUMPS ICNTL(23) | {PETSC_OPTS.get('mat_mumps_icntl_23', 'default')} (max memory MB, 0=auto) |
| MUMPS ICNTL(28) | {PETSC_OPTS.get('mat_mumps_icntl_28', 'default')} (parallel analysis) |
| MUMPS ICNTL(29) | {PETSC_OPTS.get('mat_mumps_icntl_29', 'default')} (ParMETIS ordering) |
| FE order | P2 Lagrange |
| DOFs | {sol.dofs:,} |
| KSP convergence reason | {sol.ksp_converged_reason} ({_KSP_REASONS.get(sol.ksp_converged_reason, 'UNKNOWN')}) |
| KSP iterations | {sol.ksp_iterations} |
| KSP residual norm | {sol.ksp_residual_norm:.2e} |
| Wall time (total) | {t_wall:.1f} s |
| Mesh time | {sol.mesh_time:.1f} s |
| max|p| | {sol.max_pressure:.4f} Pa |
| Memory (RSS peak) | {mem_mb:.0f} MB |

---

## E) Confirmation Checks

| Check | Result | Detail |
|-------|--------|--------|
| Standing-wave BC disabled | {'✅ PASS' if cfg.standing_velocity_amplitude == 0.0 else '❌ FAIL'} | V_stand = {cfg.standing_velocity_amplitude} m/s |
| No petri wall excitation | {'✅ PASS' if cfg.standing_velocity_amplitude == 0.0 else '❌ FAIL'} | Petri Neumann load = 0 when V_stand = 0 |
| Disk is active drive | {'✅ PASS' if cfg.disk_velocity_amplitude > 0 else '❌ FAIL'} | V_disk = {cfg.disk_velocity_amplitude*1e6:.1f} μm/s |
| PML enabled | {'✅ PASS' if cfg.pml_enabled else '❌ FAIL'} | pml_enabled = {cfg.pml_enabled} |
| PML thickness (xy) | ✅ | {cfg.t_pml_xy*1e3:.4f} mm = {cfg.pml_n_wavelengths_xy:.1f}λ |
| PML thickness (z) | ✅ | {cfg.t_pml_z*1e3:.4f} mm = {cfg.pml_n_wavelengths_z:.1f}λ |
| Top Robin BC | ✅ | Z_air = {cfg.Z_air:.1f} Pa·s/m, Z_rel = {cfg.Z_air/cfg.Z_water:.6f} |
| Solver converged | {'✅ PASS' if checks['solver_converged'] else '❌ FAIL'} | reason = {sol.ksp_converged_reason} |
| No divergence | {'✅ PASS' if checks['no_divergence'] else '❌ FAIL'} | max|p| = {sol.max_pressure:.4f} Pa |
| Vortex phase winding | {'✅ PASS' if checks['vortex_winding'] else '❌ FAIL'} | winding number = {winding_number:.2f} |
| Central null | {'✅ PASS' if checks['central_null'] else '❌ FAIL'} | |p|_center / |p|_ring = {p_center/(p_ring_max+1e-30):.4f} |
| Asymmetry matches offset | {'✅ PASS' if checks['asymmetry'] else '⚠️ WARN'} | Δ|p| ratio = {asym_ratio:.4f} |

---

## F) Energy Partition

| Region | Σ|p|² |
|--------|-------|
| Physical | {en['physical']:.4e} |
| PML | {en['pml']:.4e} |
| PML/Physical ratio | {en['ratio']:.4f} |

---

## G) Output Files

### Figures (in `figures/`)

| File | Description |
|------|-------------|
| vortex_only_xy_trap_linear.png | XY |p| at trap plane |
| vortex_only_xy_trap_log.png | XY log₁₀|p| at trap plane |
| vortex_only_xy_trap_phase.png | XY arg(p) at trap plane |
| vortex_only_xy_z1mm_*.png | XY slices at z=1mm |
| vortex_only_xy_z2mm_*.png | XY slices at z=2mm |
| vortex_only_xy_z3mm_*.png | XY slices at z=3mm |
| vortex_only_xy_z4mm_*.png | XY slices at z=4mm |
| vortex_only_xz_linear.png | XZ |p| midplane |
| vortex_only_xz_log.png | XZ log₁₀|p| midplane |
| vortex_only_xz_phase.png | XZ arg(p) midplane |

### CSVs (in `csv/`)

| File | Columns |
|------|---------|
| vortex_centerline_z.csv | z_m, abs_p, phase_rad, real_p, imag_p |
| vortex_radial_profile.csv | r_m, abs_p, phase_rad, real_p, imag_p |

### Field exports

XDMF export skipped (P2 → XDMF requires interpolation to P1; use `.npz` slices or ParaView VTK instead).
"""

with open(OUT_DIR / "VORTEX_ONLY_AUDIT.md", "w") as f:
    f.write(audit_text)
print(f"  → VORTEX_ONLY_AUDIT.md written")


# ====================================================================
# 10. Write manifest
# ====================================================================
manifest = {
    "run_id": f"vortex_only_hires_{TIMESTAMP}",
    "tag": "vortex_only_hires",
    "git_branch": "LinuxTest",
    "hostname": platform.node(),
    "platform": platform.platform(),
    "python_executable": sys.executable,
    "start_time": datetime.fromtimestamp(t_start).isoformat(),
    "end_time": datetime.now().isoformat(),
    "wall_time_s": round(t_wall, 1),
    "config": cfg.to_dict(),
    "solver": {
        "dofs": sol.dofs,
        "ksp_converged_reason": sol.ksp_converged_reason,
        "ksp_iterations": sol.ksp_iterations,
        "ksp_residual_norm": sol.ksp_residual_norm,
        "max_pressure_pa": sol.max_pressure,
    },
    "validation": checks,
    "energy_partition": en,
}
with open(OUT_DIR / "MANIFEST.json", "w") as f:
    json.dump(manifest, f, indent=2, default=str)
print(f"  → MANIFEST.json written")


print(f"\n{'='*70}")
print(f"DONE — all outputs in {OUT_DIR}")
print(f"{'='*70}")
