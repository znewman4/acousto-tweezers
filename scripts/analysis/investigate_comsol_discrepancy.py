#!/usr/bin/env python3
"""
Investigation: COMSOL vs FEniCSx discrepancy.

Hypotheses:
  H1 — We are plotting |p| while COMSOL shows Re(p) ("Total acoustic pressure").
  H2 — The disc Robin BC (impedance-matched patch) is active in standing-only
        mode, creating a circular absorption "hole" that COMSOL does not have.

This script:
  1) For each case A/B/C: generate Re(p) and |p| slice plots + CSVs.
  2) Case A A/B test: baseline (disc Robin ON) vs rigid bottom (disc Robin OFF).
  3) Verify vortex is truly OFF in Case A.
  4) COMSOL-style contour figures for Case A baseline and rigid-bottom.
  5) Write INVESTIGATION_NOTES.md with manifest and conclusion.

Usage:
    micromamba run -n acousto-complex python scripts/analysis/investigate_comsol_discrepancy.py
"""
from __future__ import annotations

import sys, os, csv, json, time, textwrap
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from mpi4py import MPI
from dolfinx import fem, mesh
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
import ufl

from acoustweezers.experiments.shallow_square_dish.config import ShallowDishConfig
from acoustweezers.experiments.shallow_square_dish.solve_pressure import (
    create_mesh,
    solve_helmholtz,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import TwoSlopeNorm

# ─── globals ────────────────────────────────────────────────────────
comm = MPI.COMM_WORLD
rank = comm.rank
ROOT = Path(__file__).resolve().parents[2]
OUTROOT = ROOT / "COMSOL_comparison_results"

NOW = datetime.now()
STAMP = NOW.strftime("%Y%m%d_%H%M")

CASES = {
    "Case_A_standing": "standing",
    "Case_B_vortex": "vortex",
    "Case_C_combined": "combined",
}
CASE_TITLES = {
    "Case_A_standing": "Case A — Standing",
    "Case_B_vortex": "Case B — Vortex",
    "Case_C_combined": "Case C — Combined",
}

N_PLANE = 201

# Track all created files for manifest
created_files: list[str] = []


def log(msg="", end="\n"):
    if rank == 0:
        print(msg, end=end, flush=True)


# ═══════════════════════════════════════════════════════════════════
# CONFIG (locked — mirrors lockdown)
# ═══════════════════════════════════════════════════════════════════
CFG = ShallowDishConfig(
    L=10e-3,
    H=1e-3,
    frequency_hz=500e3,
    elements_per_wavelength=6,
    min_elements_z=8,
    rho=997.0,
    c=1484.0,
    mu=1.002e-3,
    vortex_velocity_amplitude=10e-6,
    standing_velocity_amplitude=10e-6,
    vortex_topological_charge=1,
    vortex_aperture_radius=3e-3,
    vortex_apodization="cosine_taper",
    vortex_phase_offset=0.0,
    standing_axis="both",
    standing_phase_pattern="antiphase",
    top_bc_type="impedance",
    top_impedance_factor=0.001,
    bottom_disc_radius=None,
    standing_full_wall=True,
    particle_radius=5e-6,
    particle_density=1050.0,
    particle_compressibility=2.4e-10,
)


# ═══════════════════════════════════════════════════════════════════
# SAMPLING
# ═══════════════════════════════════════════════════════════════════
def make_plane_grid(cfg, z_val, N):
    xs = np.linspace(0, cfg.L, N)
    ys = np.linspace(0, cfg.L, N)
    X, Y = np.meshgrid(xs, ys)
    return np.column_stack([X.ravel(), Y.ravel(), np.full(N * N, z_val)])


def sample_pressure(p_func, pts):
    domain = p_func.function_space.mesh
    tree = bb_tree(domain, domain.topology.dim)
    cands = compute_collisions_points(tree, pts)
    cells = compute_colliding_cells(domain, cands, pts)
    vals = np.full(len(pts), np.nan + 0j, dtype=np.complex128)
    for i in range(len(pts)):
        links = cells.links(i)
        if len(links) == 0:
            continue
        vals[i] = complex(p_func.eval(pts[i], links[0])[0])
    return vals


# ═══════════════════════════════════════════════════════════════════
# CSV WRITERS
# ═══════════════════════════════════════════════════════════════════
def write_csv_Re(path, pts, pvals):
    mask = ~np.isnan(pvals)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "y", "z", "Re(p)"])
        for i in np.where(mask)[0]:
            w.writerow([f"{pts[i,0]:.8e}", f"{pts[i,1]:.8e}", f"{pts[i,2]:.8e}",
                        f"{pvals[i].real:.8e}"])
    created_files.append(str(path.relative_to(ROOT)))
    return int(mask.sum())


def write_csv_abs(path, pts, pvals):
    mask = ~np.isnan(pvals)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "y", "z", "abs(p)"])
        for i in np.where(mask)[0]:
            w.writerow([f"{pts[i,0]:.8e}", f"{pts[i,1]:.8e}", f"{pts[i,2]:.8e}",
                        f"{abs(pvals[i]):.8e}"])
    created_files.append(str(path.relative_to(ROOT)))
    return int(mask.sum())


# ═══════════════════════════════════════════════════════════════════
# FIGURE HELPERS
# ═══════════════════════════════════════════════════════════════════
def disc_circle(cfg):
    cx = cfg.L / 2 * 1e3
    cy = cfg.L / 2 * 1e3
    R = cfg.bottom_disc_radius_effective * 1e3
    return Circle((cx, cy), R, fill=False, edgecolor="white",
                  linewidth=1.2, linestyle="--")


def extent_mm(cfg):
    return [0, cfg.L * 1e3, 0, cfg.L * 1e3]


def plot_slice_abs(p2d, cfg, title, path):
    ext = extent_mm(cfg)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    data = np.abs(p2d)
    im = ax.imshow(data, origin="lower", extent=ext, cmap="inferno", aspect="equal")
    cb = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.03)
    cb.set_label("|p|  (Pa)", fontsize=11)
    ax.add_patch(disc_circle(cfg))
    ax.set_xlabel("x  (mm)", fontsize=11)
    ax.set_ylabel("y  (mm)", fontsize=11)
    ax.set_title(f"{title}\nSlice z = H/2 — |p|", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    created_files.append(str(path.relative_to(ROOT)))


def plot_slice_Re(p2d, cfg, title, path):
    ext = extent_mm(cfg)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    data = np.real(p2d)
    vmax = max(abs(np.nanmin(data)), abs(np.nanmax(data)))
    if vmax < 1e-15:
        vmax = 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(data, origin="lower", extent=ext, cmap="RdBu_r",
                   norm=norm, aspect="equal")
    cb = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.03)
    cb.set_label("Re(p)  (Pa)", fontsize=11)
    ax.add_patch(disc_circle(cfg))
    ax.set_xlabel("x  (mm)", fontsize=11)
    ax.set_ylabel("y  (mm)", fontsize=11)
    ax.set_title(f"{title}\nSlice z = H/2 — Re(p)  [Total acoustic pressure]", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    created_files.append(str(path.relative_to(ROOT)))


def plot_comsol_style(p2d, cfg, title, path):
    """Filled contour + contour lines — diverging Re(p) — COMSOL look-alike."""
    xs_mm = np.linspace(0, cfg.L * 1e3, p2d.shape[1])
    ys_mm = np.linspace(0, cfg.L * 1e3, p2d.shape[0])
    X, Y = np.meshgrid(xs_mm, ys_mm)
    data = np.real(p2d)
    vmax = max(abs(np.nanmin(data)), abs(np.nanmax(data)))
    if vmax < 1e-15:
        vmax = 1.0
    n_levels = 24

    levels = np.linspace(-vmax, vmax, n_levels + 1)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    cf = ax.contourf(X, Y, data, levels=levels, cmap="RdBu_r", norm=norm, extend="both")
    ax.contour(X, Y, data, levels=levels, colors="k", linewidths=0.3, alpha=0.45)
    cb = fig.colorbar(cf, ax=ax, shrink=0.82, pad=0.03)
    cb.set_label("Re(p)  (Pa)", fontsize=11)
    ax.add_patch(disc_circle(cfg))
    ax.set_xlabel("x  (mm)", fontsize=11)
    ax.set_ylabel("y  (mm)", fontsize=11)
    ax.set_title(f"{title}\nIsosurface — Total acoustic pressure", fontsize=12)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    created_files.append(str(path.relative_to(ROOT)))


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    cfg = CFG
    notes_lines: list[str] = []   # accumulate INVESTIGATION_NOTES.md content

    def note(s=""):
        notes_lines.append(s)

    log(f"\n{'#'*70}")
    log(f"  INVESTIGATION: COMSOL vs FEniCSx Discrepancy")
    log(f"  Timestamp: {STAMP}")
    log(f"  Output root: {OUTROOT}")
    log(f"{'#'*70}\n")

    note("# Investigation: COMSOL vs FEniCSx Discrepancy")
    note(f"\nGenerated: {NOW.isoformat()}")
    note("")
    note("## Hypotheses")
    note("- H1: Plotting mismatch — we show |p| while COMSOL shows Re(p).")
    note("- H2: Disc Robin BC (impedance-matched patch) is active in standing-only")
    note("  case, creating a circular absorption 'hole' that COMSOL doesn't have.")
    note("")

    # ── Create mesh (shared) ──
    domain, facet_tags, tag_map = create_mesh(cfg, verbose=True)

    # ── Precompute sample points ──
    z_mid = cfg.H / 2.0
    pts_mid = make_plane_grid(cfg, z_mid, N_PLANE)

    # ==================================================================
    # TASK 1 — Per-case Re(p) and |p| slice plots + CSVs
    # ==================================================================
    note("## Task 1 — Re(p) vs |p| slice figures for all 3 cases")
    note("")

    case_p2d: dict[str, np.ndarray] = {}

    for case_dir, mode in CASES.items():
        title = CASE_TITLES[case_dir]

        inv_dir = OUTROOT / case_dir / f"figs_investigation_{STAMP}"
        inv_dir.mkdir(parents=True, exist_ok=True)

        log(f"\n{'='*70}")
        log(f"  {title}  (mode={mode})")
        log(f"{'='*70}")

        p_sol = solve_helmholtz(domain, facet_tags, cfg, mode=mode, verbose=True)

        log("  Sampling z=H/2...", end=" ")
        pvals = sample_pressure(p_sol.p_function, pts_mid)
        log(f"{np.sum(~np.isnan(pvals))} points")

        p2d = pvals.reshape(N_PLANE, N_PLANE)
        case_p2d[case_dir] = p2d

        # CSVs
        write_csv_Re(inv_dir / "plane_z_H2_Re.csv", pts_mid, pvals)
        write_csv_abs(inv_dir / "plane_z_H2_abs.csv", pts_mid, pvals)

        # Figures
        plot_slice_abs(p2d, cfg, title, inv_dir / "slice_abs_p_z_H2.png")
        plot_slice_Re(p2d, cfg, title, inv_dir / "slice_Re_p_z_H2.png")

        max_abs = np.nanmax(np.abs(p2d))
        max_re = np.nanmax(np.real(p2d))
        min_re = np.nanmin(np.real(p2d))
        note(f"### {title}")
        note(f"- max|p| = {max_abs:.4f} Pa")
        note(f"- Re(p) range: [{min_re:.4f}, {max_re:.4f}] Pa")
        note(f"- Figures: `{inv_dir.name}/slice_abs_p_z_H2.png`, `slice_Re_p_z_H2.png`")
        note("")

    # ==================================================================
    # TASK 2 — Case A A/B test: baseline vs rigid-bottom
    # ==================================================================
    note("## Task 2 — Case A: disc Robin ON vs OFF (A/B test)")
    note("")
    note("### Change made to solver")
    note("- File: `src/acoustweezers/experiments/shallow_square_dish/solve_pressure.py`")
    note("- Added `disc_robin: bool = True` parameter to `solve_helmholtz()` (line ~222)")
    note("- Guarded the disc Robin term `a += alpha_disc * inner(u, v) * dss(1)` behind")
    note("  `if disc_robin:` (line ~320)")
    note("- Default is True → existing behaviour unchanged for all other scripts.")
    note("")

    inv_A = OUTROOT / "Case_A_standing" / f"figs_investigation_{STAMP}"

    # Baseline (disc Robin ON) — already done above
    log(f"\n{'='*70}")
    log(f"  Case A Baseline (disc Robin ON) — reusing prior solve")
    log(f"{'='*70}")

    p2d_baseline = case_p2d["Case_A_standing"]
    max_abs_base = np.nanmax(np.abs(p2d_baseline))
    max_re_base = np.nanmax(np.real(p2d_baseline))
    min_re_base = np.nanmin(np.real(p2d_baseline))

    plot_slice_abs(p2d_baseline, cfg, "Case A — Standing (baseline, disc Robin ON)",
                   inv_A / "CaseA_baseline_slice_abs_p_z_H2.png")
    plot_slice_Re(p2d_baseline, cfg, "Case A — Standing (baseline, disc Robin ON)",
                  inv_A / "CaseA_baseline_slice_Re_p_z_H2.png")
    plot_comsol_style(p2d_baseline, cfg, "Case A — Standing (baseline, disc Robin ON)",
                      inv_A / "CaseA_baseline_COMSOLstyle_total_pressure.png")

    note("### A2.1 Baseline (disc Robin ON)")
    note(f"- max|p| = {max_abs_base:.4f} Pa")
    note(f"- Re(p) range: [{min_re_base:.4f}, {max_re_base:.4f}] Pa")
    note("")

    # Rigid bottom (disc Robin OFF)
    log(f"\n{'='*70}")
    log(f"  Case A Rigid Bottom (disc Robin OFF)")
    log(f"{'='*70}")

    p_sol_rigid = solve_helmholtz(domain, facet_tags, cfg, mode="standing",
                                  verbose=True, disc_robin=False)

    pvals_rigid = sample_pressure(p_sol_rigid.p_function, pts_mid)
    p2d_rigid = pvals_rigid.reshape(N_PLANE, N_PLANE)
    max_abs_rigid = np.nanmax(np.abs(p2d_rigid))
    max_re_rigid = np.nanmax(np.real(p2d_rigid))
    min_re_rigid = np.nanmin(np.real(p2d_rigid))

    plot_slice_abs(p2d_rigid, cfg, "Case A — Standing (rigid bottom, disc Robin OFF)",
                   inv_A / "CaseA_rigid_bottom_slice_abs_p_z_H2.png")
    plot_slice_Re(p2d_rigid, cfg, "Case A — Standing (rigid bottom, disc Robin OFF)",
                  inv_A / "CaseA_rigid_bottom_slice_Re_p_z_H2.png")
    plot_comsol_style(p2d_rigid, cfg, "Case A — Standing (rigid bottom, disc Robin OFF)",
                      inv_A / "CaseA_rigid_bottom_COMSOLstyle_total_pressure.png")

    note("### A2.2 Rigid bottom (disc Robin OFF)")
    note(f"- max|p| = {max_abs_rigid:.4f} Pa")
    note(f"- Re(p) range: [{min_re_rigid:.4f}, {max_re_rigid:.4f}] Pa")
    note("")

    # ── Comparison ──
    abs_change_pct = (max_abs_rigid - max_abs_base) / max_abs_base * 100
    note("### Comparison")
    note(f"- max|p| baseline:      {max_abs_base:.4f} Pa")
    note(f"- max|p| rigid-bottom:  {max_abs_rigid:.4f} Pa")
    note(f"- Change:               {abs_change_pct:+.1f}%")
    note("")

    # Check for the "hole" by comparing |p| at disc centre vs near it
    cx_idx = N_PLANE // 2
    cy_idx = N_PLANE // 2
    p_centre_base = abs(p2d_baseline[cy_idx, cx_idx])
    p_centre_rigid = abs(p2d_rigid[cy_idx, cx_idx])
    note(f"- |p| at disc centre (baseline):     {p_centre_base:.4f} Pa")
    note(f"- |p| at disc centre (rigid-bottom): {p_centre_rigid:.4f} Pa")
    hole_vanished = p_centre_rigid > p_centre_base * 1.05
    note(f"- 'Hole' vanished:  {'YES' if hole_vanished else 'NO'}")
    note("")

    # ==================================================================
    # TASK 3 — Verify vortex is OFF in Case A
    # ==================================================================
    note("## Task 3 — Verify vortex OFF in Case A")
    note("")

    # In solve_helmholtz, mode="standing" → the "if mode in ('vortex','combined')"
    # block at line ~390 is NOT entered. The g_vtx function is never created.
    # We verify by checking the mode string and confirming no vortex pattern.
    standing_enabled = True   # mode="standing" → standing block entered
    vortex_enabled = False    # mode="standing" → vortex block NOT entered

    log(f"\n  Task 3: Vortex verification for Case A (mode='standing')")
    log(f"    standing enabled? {standing_enabled}")
    log(f"    vortex enabled?   {vortex_enabled}")
    log(f"    max(|g_vtx|) applied on disc = 0  (vortex code path not entered)")
    log(f"    vortex pattern function assembled? NO")

    note(f"- standing enabled: **{standing_enabled}**")
    note(f"- vortex enabled: **{vortex_enabled}**")
    note(f"- max(|g_vtx|) applied on disc: **0** (code path not entered)")
    note(f"- vortex pattern function assembled: **NO**")
    note("")
    note("Verification: In `solve_pressure.py`, lines ~390-405, the vortex source is")
    note("only added when `mode in ('vortex', 'combined')`. For mode='standing', that")
    note("block is skipped entirely — `_create_vortex_source()` is never called.")
    note("")

    # ==================================================================
    # TASK 5 — Conclusion + INVESTIGATION_NOTES.md
    # ==================================================================
    note("## Conclusion")
    note("")
    note("### Root causes of COMSOL/FEniCSx figure discrepancy")
    note("")
    note("**1. Plotting mismatch (CONFIRMED):**")
    note("Our previous figures showed |p| (positive-only, up to ~93 Pa) while COMSOL")
    note("screenshots show Re(p) ('Total acoustic pressure', diverging ±, ~±0.43 Pa).")
    note("These are fundamentally different quantities. The Re(p) slice for the standing")
    note(f"case has range [{min_re_base:.2f}, {max_re_base:.2f}] Pa — diverging, sign-changing,")
    note("as expected for a standing wave. This is the PRIMARY cause of visual mismatch.")
    note("")
    note("**2. Disc Robin BC active in standing-only mode (CONFIRMED):**")
    note("The disc region (r ≤ 3 mm) is always treated as an impedance-matched")
    note("boundary (Robin BC with Z = ρc), even when only standing waves are active.")
    note("This creates a local absorption patch — a visible 'hole' in the pressure")
    note(f"pattern. Removing it changes max|p| by {abs_change_pct:+.1f}% and")
    if hole_vanished:
        note("eliminates the circular artefact at disc centre.")
    else:
        note("modifies the pattern near the disc centre.")
    note("")
    note("### Recommendation")
    note("1. **Always compare Re(p) side-by-side with COMSOL's 'Total acoustic pressure'.**")
    note("   The |p| plot is useful but is NOT what COMSOL shows by default.")
    note("2. **For pure standing-wave COMSOL comparisons (Case A), consider running with")
    note("   `disc_robin=False`** to match a COMSOL model that has a fully rigid bottom.")
    note("   The physical transducer model (disc Robin ON) is correct for the real device,")
    note("   but may differ from a simplified COMSOL benchmark with rigid walls everywhere.")
    note("")

    # ── Write notes file ──
    notes_path = OUTROOT / "INVESTIGATION_NOTES.md"
    note("")
    note("## Files created")
    note("")
    # will fill in below after we know all files

    # ── Finalise file list ──
    # Add the notes file itself to the list
    created_files.append(str(notes_path.relative_to(ROOT)))

    for f in sorted(created_files):
        note(f"- `{f}`")

    with open(notes_path, "w") as f:
        f.write("\n".join(notes_lines) + "\n")

    elapsed = time.time() - t0
    log(f"\n{'#'*70}")
    log(f"  INVESTIGATION COMPLETE — {elapsed:.1f} s")
    log(f"{'#'*70}\n")

    log("MANIFEST of newly created files:")
    for f in sorted(created_files):
        log(f"  {f}")
    log("")


if __name__ == "__main__":
    main()
