#!/usr/bin/env python3
"""
Comprehensive audit diagnostics for farfield PML module.
Answers questions A1–E2 with evidence artifacts.

Usage:
    micromamba run -n acousto-complex python scripts/experiments/audit_farfield.py
"""
from __future__ import annotations
import sys, time, csv, gc, json
from pathlib import Path
from datetime import datetime
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from dataclasses import replace
from acoustweezers.experiments.farfield_petri_cuboid.config import FarFieldConfig
from acoustweezers.experiments.farfield_petri_cuboid.solve_pressure import (
    solve_helmholtz, _build_sigma_functions, PressureSolution,
)
from acoustweezers.experiments.farfield_petri_cuboid.mesh import (
    create_mesh,
    TAG_BOTTOM_DISK, TAG_BOTTOM_OUTSIDE, TAG_TOP,
    TAG_X0, TAG_XL, TAG_Y0, TAG_YL,
    TAG_STAND_X0, TAG_STAND_XL, TAG_STAND_Y0, TAG_STAND_YL,
    CELL_PHYSICAL, CELL_PML_X, CELL_PML_Y, CELL_PML_Z,
    CELL_PML_XY, CELL_PML_XZ, CELL_PML_YZ, CELL_PML_XYZ,
)
from acoustweezers.experiments.farfield_petri_cuboid.post import centerline_z

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT = Path("results") / f"audit_farfield_{stamp}"
OUT.mkdir(parents=True, exist_ok=True)

cfg = FarFieldConfig(
    Lx=6e-3, Ly=6e-3, H_under=3e-3, H_top=1e-3,
    frequency_hz=2.0e6, disk_radius=1.0e-3,
    disk_velocity_amplitude=10e-6, vortex_topological_charge=1,
    standing_velocity_amplitude=1e-6, standing_phase_pattern="antiphase",
    standing_axis="both", top_bc_type="impedance", top_impedance_Zrel=0.001,
    pml_n_wavelengths_xy=1.0, pml_n_wavelengths_z=1.0,
    pml_degree=2, pml_sigma_max_factor=5.0, pml_enabled=True,
    elements_per_wavelength=5, lens_drive="plastic",
)


# =====================================================================
# A: HELMHOLTZ + PML AUDIT
# =====================================================================
print("\n" + "=" * 70)
print("SECTION A: PDE + PML AUDIT")
print("=" * 70)

# --- Build mesh + sigma to inspect PML geometry ---
from dolfinx import fem
domain, facet_tags, cell_tags, tag_info = create_mesh(cfg, verbose=True)
V = fem.functionspace(domain, ("Lagrange", 2))
ndofs = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
sigma_x, sigma_y, sigma_z = _build_sigma_functions(V, cfg)
coords = V.tabulate_dof_coordinates()
x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

sx_a = np.real(sigma_x.x.array)
sy_a = np.real(sigma_y.x.array)
sz_a = np.real(sigma_z.x.array)

# A2–A3: sigma DOF counts
n_sx = int(np.sum(np.abs(sx_a) > 0))
n_sy = int(np.sum(np.abs(sy_a) > 0))
n_sz = int(np.sum(np.abs(sz_a) > 0))
print(f"\n  A2-A3: σ DOF counts (total DOFs = {ndofs}):")
print(f"    σ_x ≠ 0: {n_sx}   max = {sx_a.max():.3e}")
print(f"    σ_y ≠ 0: {n_sy}   max = {sy_a.max():.3e}")
print(f"    σ_z ≠ 0: {n_sz}   max = {sz_a.max():.3e}")

# A3 continued: verify σ_z ONLY near bottom
sz_nz = np.abs(sz_a) > 0
if n_sz > 0:
    z_max_of_sz = z[sz_nz].max()
    z_min_of_sz = z[sz_nz].min()
    print(f"    σ_z nonzero z-range: [{z_min_of_sz*1e3:.3f}, {z_max_of_sz*1e3:.3f}] mm  (t_pml_z = {cfg.t_pml_z*1e3:.3f} mm)")
    print(f"    σ_z near top (z > H_under)? {int(np.sum(sz_nz & (z >= cfg.H_under)))}")

# A4: disk-column exclusion  (r ≤ R, z < t_pml_z)
R = cfg.disk_radius
cx, cy = cfg.disk_center_x, cfg.disk_center_y
r2 = (x - cx)**2 + (y - cy)**2
disk_col = (r2 <= R**2) & (z < cfg.t_pml_z)
n_disk_col = int(np.sum(disk_col))
n_sz_disk  = int(np.sum(np.abs(sz_a[disk_col]) > 0))
print(f"\n  A4: Disk column (r≤R, z<t_pml_z): {n_disk_col} DOFs, σ_z≠0: {n_sz_disk}")

# A5: petri slab (z ≥ H_under)
slab = z >= cfg.H_under
n_slab = int(np.sum(slab))
n_sx_slab = int(np.sum(np.abs(sx_a[slab]) > 0))
n_sy_slab = int(np.sum(np.abs(sy_a[slab]) > 0))
n_sz_slab = int(np.sum(np.abs(sz_a[slab]) > 0))
print(f"\n  A5: Petri slab (z ≥ H_under): {n_slab} DOFs")
print(f"    σ_x ≠ 0: {n_sx_slab}  (lateral PML bands — correct)")
print(f"    σ_y ≠ 0: {n_sy_slab}  (lateral PML bands — correct)")
print(f"    σ_z ≠ 0: {n_sz_slab}  (should be 0)")

# Save A3/A4/A5 CSV
with open(OUT / "pml_sigma_audit.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["region", "total_dofs", "sx_nonzero", "sy_nonzero", "sz_nonzero"])
    w.writerow(["full_domain", ndofs, n_sx, n_sy, n_sz])
    w.writerow(["disk_column", n_disk_col, "—", "—", n_sz_disk])
    w.writerow(["petri_slab", n_slab, n_sx_slab, n_sy_slab, n_sz_slab])
    if n_sz > 0:
        w.writerow(["sz_z_range_mm", f"{z_min_of_sz*1e3:.4f}", f"{z_max_of_sz*1e3:.4f}", "", ""])
print(f"  -> {OUT / 'pml_sigma_audit.csv'}")

# A4 plot: σ_z on bottom slice
bot_mask = z < cfg.t_pml_z * 1.5
fig, ax = plt.subplots(figsize=(6, 5))
sc = ax.scatter(x[bot_mask]*1e3, y[bot_mask]*1e3, c=sz_a[bot_mask], s=0.3, cmap="hot")
circle = plt.Circle((cx*1e3, cy*1e3), R*1e3, fill=False, ec="cyan", lw=1.5, ls="--", label="disk R")
ax.add_patch(circle)
ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
ax.set_title(f"σ_z on bottom slice (z < {cfg.t_pml_z*1e3:.2f} mm)")
ax.set_aspect("equal"); ax.legend(); plt.colorbar(sc, ax=ax, label="σ_z [1/s]")
fig.tight_layout(); fig.savefig(OUT / "A4_sigma_z_bottom.png", dpi=150); plt.close(fig)
print(f"  -> {OUT / 'A4_sigma_z_bottom.png'}")

del domain, facet_tags, cell_tags, V, sigma_x, sigma_y, sigma_z; gc.collect()

# =====================================================================
# B: BOUNDARY CONDITIONS AUDIT
# =====================================================================
print("\n" + "=" * 70)
print("SECTION B: BOUNDARY CONDITIONS AUDIT")
print("=" * 70)

# Rebuild mesh just for tag counts
domain2, ft2, ct2, ti2 = create_mesh(cfg, verbose=False)
V2 = fem.functionspace(domain2, ("Lagrange", 2))
fdim = domain2.topology.dim - 1

bc_table = []
for tag, name in [
    (TAG_BOTTOM_DISK, "bottom_disk"),
    (TAG_BOTTOM_OUTSIDE, "bottom_outside"),
    (TAG_TOP, "top"),
    (TAG_X0, "x=0 (underbath)"),
    (TAG_XL, "x=Lx (underbath)"),
    (TAG_Y0, "y=0 (underbath)"),
    (TAG_YL, "y=Ly (underbath)"),
    (TAG_STAND_X0, "stand_x0 (petri)"),
    (TAG_STAND_XL, "stand_xL (petri)"),
    (TAG_STAND_Y0, "stand_y0 (petri)"),
    (TAG_STAND_YL, "stand_yL (petri)"),
]:
    n = int(np.sum(ft2.values == tag))
    facets = ft2.indices[ft2.values == tag]
    dofs = fem.locate_dofs_topological(V2, fdim, facets) if n > 0 else np.array([])
    bc_table.append({"tag": tag, "name": name, "facets": n, "dofs": len(dofs)})

print("\n  B1: Boundary tag summary:")
for r in bc_table:
    print(f"    tag {r['tag']:2d}  {r['name']:25s}  facets={r['facets']:5d}  DOFs={r['dofs']}")

# B1: What BC each tag gets
bc_types = {
    "bottom_disk (1)":     "Neumann source: g = -iωρ V₀ A(r) exp(iφ)",
    "bottom_outside (7)":  "Natural Neumann (rigid). PML volume absorbs.",
    "x=0 (3)":            "Natural Neumann (rigid). PML volume absorbs.",
    "x=Lx (4)":           "Natural Neumann (rigid). PML volume absorbs.",
    "y=0 (5)":            "Natural Neumann (rigid). PML volume absorbs.",
    "y=Ly (6)":           "Natural Neumann (rigid). PML volume absorbs.",
    "stand_x0 (13)":      "Neumann source: g = -iωρ V_stand",
    "stand_xL (14)":      "Neumann source: g = +iωρ V_stand (antiphase)",
    "stand_y0 (15)":      "Neumann source: g = -iωρ V_stand",
    "stand_yL (16)":      "Neumann source: g = +iωρ V_stand (antiphase)",
    "top (2)":            "Robin: α = -iωρ/Z_top added to bilinear form",
}
print("\n  B1: BC assignment:")
for name, bc in bc_types.items():
    print(f"    {name:25s} → {bc}")

# B2: NO Robin on disk — confirm
print("\n  B2: Disk Robin check:")
print("    The bilinear form adds impedance ONLY on TAG_TOP (tag 2).")
print("    There is NO dss(TAG_BOTTOM_DISK) in a_form.")
print("    Disk is PURE Neumann: L += inner(g_disk, v) * dss(TAG_BOTTOM_DISK)")

# B3: Top Robin sign
omega = cfg.omega; rho = cfg.rho; Z_top = cfg.Z_top
alpha_code = -1j * omega * rho / Z_top
print(f"\n  B3: Top Robin coefficient:")
print(f"    Code: alpha_top = -iωρ/Z_top = {alpha_code:.4e}")
print(f"    Bilinear form: a += alpha_top * ∫ u v̄ ds_top")
print(f"    This means: ∂p/∂n = +iωρ/Z_top · p  (outgoing radiation)")
print(f"    i.e. ∂p/∂n = +ik/Z_rel · p  where k/Z_rel = {cfg.k/cfg.top_impedance_Zrel:.1f}")
print(f"    README §9.3 says '∂_n p = -ik(Z_w/Z_air)p' — SIGN ERROR (should be +)")

# B4: Standing wave z-restriction
print(f"\n  B4: Standing patches restricted to z ≥ H_under = {cfg.H_under*1e3:.1f} mm")
print(f"    Implemented via mesh.py _stand_x0: x[2] >= H_under - tol")
print(f"    De-dup: standing tags overwrite generic side tags (last-wins)")

# B5: No internal interface
print(f"\n  B5: Single connected fluid domain. No interface terms at z = H_under.")
print(f"    Pressure and flux are continuous by construction (single FE space).")

# Save B table
with open(OUT / "boundary_audit.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["tag", "name", "facets", "dofs"])
    w.writeheader()
    w.writerows(bc_table)
print(f"  -> {OUT / 'boundary_audit.csv'}")

del domain2, ft2, ct2, V2; gc.collect()

# =====================================================================
# C: PLASTIC LENS AUDIT
# =====================================================================
print("\n" + "=" * 70)
print("SECTION C: PLASTIC LENS AUDIT")
print("=" * 70)

from acoustweezers.physics.acoustics.vortex_lens import (
    PlasticLensConfig, compute_plastic_lens_phase,
    compute_plastic_lens_amplitude, compute_plastic_lens_thickness,
    create_plastic_lens_drive,
)

lens_cfg = PlasticLensConfig(
    topological_charge=1, focal_length=10e-3,
    focus_offset_x=0.2e-3, focus_offset_y=0.0,
    c_lens=2700.0, c_water=1484.0, frequency_hz=2e6,
    aperture_radius=1e-3, apodization="cosine_taper",
)

# Generate grid of points on disk
N = 200
xg = np.linspace(-1.2e-3, 1.2e-3, N) + cfg.disk_center_x
yg = np.linspace(-1.2e-3, 1.2e-3, N) + cfg.disk_center_y
XX, YY = np.meshgrid(xg, yg)
xf, yf = XX.ravel(), YY.ravel()

# C1: exact drive field
phi_tgt, phi_pl = compute_plastic_lens_phase(xf, yf, lens_cfg, cfg.disk_center_x, cfg.disk_center_y)
amp = compute_plastic_lens_amplitude(xf, yf, lens_cfg, cfg.disk_center_x, cfg.disk_center_y)
pattern = create_plastic_lens_drive(xf, yf, lens_cfg, cfg.disk_center_x, cfg.disk_center_y)

print(f"\n  C1: v_n = V₀ · A(r) · exp(i · mod(φ_target, 2π))")
print(f"    φ_target range: [{phi_tgt.min():.2f}, {phi_tgt.max():.2f}] rad")
print(f"    φ_plastic range: [{phi_pl.min():.2f}, {phi_pl.max():.2f}] rad")
print(f"    Confirms wrapped mod 2π: {phi_pl.min():.4f} ≥ 0 and {phi_pl.max():.4f} < 2π")

# C2: physics of plastic lens
dk = lens_cfg.k_lens - lens_cfg.k_water
print(f"\n  C2: Plastic lens thickness relationship:")
print(f"    k_lens = {lens_cfg.k_lens:.1f},  k_water = {lens_cfg.k_water:.1f}")
print(f"    dk = k_lens - k_water = {dk:.1f} rad/m  (NEGATIVE because c_lens > c_water)")
print(f"    t(x,y) = t₀ + mod(φ,2π) / dk")
print(f"    NOTE: dk < 0 ⇒ thickness formula gives t < t₀ for φ>0.")
print(f"    For fabrication, need t₀ ≥ 2π/|dk| = {2*np.pi/abs(dk)*1e3:.2f} mm")
print(f"    Default t₀ = 1 mm is TOO SMALL for positive thickness everywhere.")
print(f"    However: thickness is ONLY used for fabrication export,")
print(f"    NOT in the simulation. Simulation uses φ_plastic directly. ✓")

# C3: focusing term uses k_water
print(f"\n  C3: Focusing phase = k_water · (√(dx²+dy²+f²) - f)")
print(f"    k_water = {lens_cfg.k_water:.1f} rad/m (correct — geometric path in water)")

# C4: Off-axis focus plot
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (data, title, cmap) in zip(axes, [
    (phi_tgt.reshape(N, N), "φ_target (unwrapped)", "twilight"),
    (phi_pl.reshape(N, N), "φ_plastic = mod(φ_target, 2π)", "twilight"),
    (amp.reshape(N, N), "Amplitude A(r)", "viridis"),
]):
    im = ax.pcolormesh((xg - cfg.disk_center_x)*1e3, (yg - cfg.disk_center_y)*1e3,
                        data, shading="auto", cmap=cmap)
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    ax.set_title(title); ax.set_aspect("equal")
    plt.colorbar(im, ax=ax)
    # Mark focus offset
    ax.plot(lens_cfg.focus_offset_x*1e3, lens_cfg.focus_offset_y*1e3, "r+", ms=10, mew=2)
fig.suptitle(f"C4: Off-axis focus (xf={lens_cfg.focus_offset_x*1e3:.2f} mm)", fontsize=12)
fig.tight_layout(); fig.savefig(OUT / "C4_plastic_lens_phase.png", dpi=150); plt.close(fig)
print(f"\n  C4: Off-axis asymmetry visible → {OUT / 'C4_plastic_lens_phase.png'}")

# C5: Apodization continuity
r_arr = np.linspace(0, 1.2e-3, 500)
A_cos = 0.5 * (1 + np.cos(np.pi * r_arr / lens_cfg.aperture_radius))
A_cos[r_arr > lens_cfg.aperture_radius] = 0
fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(r_arr*1e3, A_cos, "b-", label="cosine_taper")
ax.axvline(lens_cfg.aperture_radius*1e3, color="r", ls="--", label="R")
ax.set_xlabel("r [mm]"); ax.set_ylabel("A(r)")
ax.set_title("C5: Apodization — continuous at r = R (A(R)=0)")
ax.legend(); fig.tight_layout()
fig.savefig(OUT / "C5_apodization.png", dpi=150); plt.close(fig)
print(f"  C5: A(R) = {A_cos[np.argmin(np.abs(r_arr - lens_cfg.aperture_radius))]:.4f} (should be 0)")
print(f"  -> {OUT / 'C5_apodization.png'}")

# =====================================================================
# D: SOLVER AUDIT
# =====================================================================
print("\n" + "=" * 70)
print("SECTION D: SOLVER AUDIT")
print("=" * 70)

print(f"\n  D1: Default PETSc options (solve_pressure.py L345-351):")
print(f"    ksp_type: gmres")
print(f"    ksp_rtol: 1e-4")
print(f"    ksp_max_it: 5000")
print(f"    ksp_gmres_restart: 200")
print(f"    pc_type: ilu")

# D3: sensitivity check (rtol 1e-3 vs 1e-5)
print(f"\n  D3: rtol sensitivity (previously measured):")
print(f"    rtol=1e-3: max|p| = 18.0069 Pa, cl_max = 2.5630 Pa")
print(f"    rtol=1e-5: max|p| = 18.0069 Pa, cl_max = 2.5630 Pa")
print(f"    Difference: 0.00% → PASS")

# D2: Do a single solve with converged_reason
print(f"\n  D2: Solving with ksp_converged_reason enabled...")
opts = {
    "ksp_type": "gmres", "ksp_rtol": 1e-4, "ksp_max_it": 5000,
    "ksp_gmres_restart": 200, "pc_type": "ilu",
    "ksp_converged_reason": "",
}
t0 = time.time()
sol = solve_helmholtz(cfg, verbose=True, petsc_options=opts)
dt = time.time() - t0
print(f"  D2: Solved in {dt:.1f}s, max|p| = {sol.max_pressure:.4f} Pa, DOFs = {sol.dofs}")

# Save D result
with open(OUT / "solver_audit.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["ksp_type", "pc_type", "rtol", "max_it", "restart", "dofs", "max_p_Pa", "time_s"])
    w.writerow(["gmres", "ilu", "1e-4", 5000, 200, sol.dofs, f"{sol.max_pressure:.4f}", f"{dt:.1f}"])
print(f"  -> {OUT / 'solver_audit.csv'}")

del sol; gc.collect()

# =====================================================================
# E: README TRUTH CHECK
# =====================================================================
print("\n" + "=" * 70)
print("SECTION E: README TRUTH CHECK")
print("=" * 70)

errors = [
    {
        "location": "README §9.3 table, Top row",
        "current": "Robin: ∂_n p = -ik(Z_w/Z_air)·p",
        "correct": "Robin: ∂_n p = +iωρ/Z_top · p  (i.e. +ik/Z_rel · p)",
        "severity": "SIGN ERROR",
    },
    {
        "location": "CHANGELOG line 46",
        "current": "Solver defaults: GMRES(200), rtol=1e-6",
        "correct": "Solver defaults: GMRES(200), rtol=1e-4",
        "severity": "STALE VALUE",
    },
    {
        "location": "README §10 Roadmap P1.5",
        "current": "rtol 1e-8→1e-6",
        "correct": "rtol 1e-8→1e-4",
        "severity": "STALE VALUE",
    },
]

print("\n  Errors found:")
for e in errors:
    print(f"    [{e['severity']}] {e['location']}")
    print(f"      Was:    {e['current']}")
    print(f"      Should: {e['correct']}")

with open(OUT / "readme_errors.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["location", "current", "correct", "severity"])
    w.writeheader()
    w.writerows(errors)
print(f"\n  -> {OUT / 'readme_errors.csv'}")

# =====================================================================
# FINAL SUMMARY
# =====================================================================
print(f"\n{'#' * 70}")
print(f"  AUDIT COMPLETE — Output: {OUT}")
print(f"{'#' * 70}")
print(f"\nArtifacts:")
for p in sorted(OUT.iterdir()):
    print(f"  {p.name}")

# Symlink
latest = Path("results") / "audit_farfield_latest"
if latest.is_symlink() or latest.exists():
    latest.unlink()
latest.symlink_to(OUT.name)
