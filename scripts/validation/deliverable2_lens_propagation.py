#!/usr/bin/env python3
"""
Deliverable 2 — Plastic Lens Propagation Proof

A) Boundary drive visuals: |v_n|, arg(v_n), Re(v_n), Im(v_n)
B) Z-Stack proof: 6 z-planes with |p|, arg(p), I_z
C) Radial profile at trapping plane: |p|(r) with null at r=0 and ring max
D) Winding number computation

Output:  results/Deliverable2_LensPropagation/
"""
from __future__ import annotations

import json
import sys
import time
import numpy as np
from datetime import datetime
from pathlib import Path
from scipy.interpolate import NearestNDInterpolator

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from dolfinx import fem
from acoustweezers.experiments.farfield_petri_cuboid.config import (
    FarFieldConfig, demo_config,
)
from acoustweezers.experiments.farfield_petri_cuboid.solve_pressure import (
    solve_helmholtz, PressureSolution, TAG_BOTTOM_DISK,
)

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "Deliverable2_LensPropagation"
FIGS = OUT / "figs"


def _save(fig, name, dpi=150):
    fig.savefig(FIGS / name, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _interp_complex(sol, pts):
    coords = sol.coords
    pv = sol.p_values
    interp_re = NearestNDInterpolator(coords, np.real(pv))
    interp_im = NearestNDInterpolator(coords, np.imag(pv))
    return interp_re(pts) + 1j * interp_im(pts)


def _interp_complex_xy(sol, z_val, nx=200, ny=200):
    cfg = sol.cfg
    xg = np.linspace(0, cfg.Lx, nx)
    yg = np.linspace(0, cfg.Ly, ny)
    X, Y = np.meshgrid(xg, yg)
    Z = np.full_like(X, z_val)
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    pc = _interp_complex(sol, pts).reshape(X.shape)
    return xg, yg, pc


# ═════════════════════════════════════════════════════════════════════
#  A — Boundary Drive Visuals
# ═════════════════════════════════════════════════════════════════════

def part_A(sol: PressureSolution, results):
    """Extract and plot the disk boundary pattern: |v_n|, arg(v_n), Re(v_n), Im(v_n)."""
    print("\n  ── Part A: Boundary Drive Visuals ──")
    cfg = sol.cfg
    V = sol.V
    domain = sol.domain
    facet_tags = sol.facet_tags

    fdim = domain.topology.dim - 1
    disk_facets = facet_tags.indices[facet_tags.values == TAG_BOTTOM_DISK]
    disk_dofs = fem.locate_dofs_topological(V, fdim, disk_facets)
    coords = V.tabulate_dof_coordinates()

    # The disk source pattern is embedded in p_function on disk DOFs.
    # Better: re-create the disk source to get raw v_n pattern.
    from acoustweezers.experiments.farfield_petri_cuboid.solve_pressure import (
        _create_disk_source,
    )
    g_disk = _create_disk_source(V, domain, facet_tags, cfg, verbose=False)
    pattern = g_disk.x.array[disk_dofs]  # complex pattern (normalised)

    # Scale to physical velocity
    V_disk = cfg.disk_velocity_amplitude
    v_n = pattern * V_disk  # m/s

    x_d = (coords[disk_dofs, 0] - cfg.disk_center_x) * 1e3  # mm from center
    y_d = (coords[disk_dofs, 1] - cfg.disk_center_y) * 1e3

    titles = ["|v_n|", "arg(v_n)", "Re(v_n)", "Im(v_n)"]
    data = [np.abs(v_n), np.angle(v_n), np.real(v_n), np.imag(v_n)]
    cmaps = ["inferno", "twilight", "RdBu_r", "RdBu_r"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for ax, d, t, cm in zip(axes.ravel(), data, titles, cmaps):
        sc = ax.scatter(x_d, y_d, c=d, cmap=cm, s=6, marker="o")
        ax.set_title(t, fontsize=13)
        ax.set_xlabel("Δx [mm]")
        ax.set_ylabel("Δy [mm]")
        ax.set_aspect("equal")
        plt.colorbar(sc, ax=ax)

    fig.suptitle("Disk Boundary Drive Pattern", fontsize=14)
    fig.tight_layout()
    _save(fig, "A_disk_boundary_drive.png")

    results["A_boundary_drive"] = {
        "n_disk_dofs": len(disk_dofs),
        "max_abs_v_n_um_s": float(np.max(np.abs(v_n)) * 1e6),
        "phase_range_rad": float(np.ptp(np.angle(v_n[np.abs(v_n) > 1e-10 * V_disk]))),
    }
    print(f"    Disk DOFs: {len(disk_dofs)}")
    print(f"    max|v_n|: {np.max(np.abs(v_n))*1e6:.2f} μm/s")
    print(f"    Saved A_disk_boundary_drive.png")


# ═════════════════════════════════════════════════════════════════════
#  B — Z-Stack Proof
# ═════════════════════════════════════════════════════════════════════

def part_B(sol: PressureSolution, results):
    """6 z-planes: |p|, arg(p), I_z with consistent colour limits."""
    print("\n  ── Part B: Z-Stack Proof ──")
    cfg = sol.cfg

    z_lo = cfg.t_pml_z + 0.1e-3
    z_hi = cfg.H_total - 0.05e-3
    z_levels = np.linspace(z_lo, z_hi, 6)

    nx, ny = 180, 180

    # Precompute
    all_pc = []
    all_Iz = []
    for zv in z_levels:
        xg, yg, pc = _interp_complex_xy(sol, zv, nx, ny)
        all_pc.append(pc)

        # I_z via finite difference
        h_fd = cfg.wavelength / cfg.elements_per_wavelength * 1.5
        _, _, pc_up = _interp_complex_xy(sol, zv + h_fd, nx, ny)
        dp_dz = (pc_up - pc) / h_fd
        vz = 1.0 / (1j * cfg.omega * cfg.rho) * dp_dz
        Iz_slice = 0.5 * np.real(pc * np.conj(vz))
        all_Iz.append(Iz_slice)

    vmax_abs = max(np.abs(pc).max() for pc in all_pc) * 1.02
    Iz_absmax = max(np.percentile(np.abs(Iz_s), 99) for Iz_s in all_Iz)
    if Iz_absmax < 1e-20:
        Iz_absmax = 1.0

    for iz, zv in enumerate(z_levels):
        pc = all_pc[iz]
        Iz_s = all_Iz[iz]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        im0 = axes[0].pcolormesh(xg * 1e3, yg * 1e3, np.abs(pc),
                                  shading="auto", cmap="inferno",
                                  vmin=0, vmax=vmax_abs)
        axes[0].set_title(f"|p|  z={zv*1e3:.2f} mm")
        plt.colorbar(im0, ax=axes[0], label="Pa")

        im1 = axes[1].pcolormesh(xg * 1e3, yg * 1e3, np.angle(pc),
                                  shading="auto", cmap="twilight",
                                  vmin=-np.pi, vmax=np.pi)
        axes[1].set_title(f"arg(p)  z={zv*1e3:.2f} mm")
        plt.colorbar(im1, ax=axes[1], label="rad")

        im2 = axes[2].pcolormesh(xg * 1e3, yg * 1e3, Iz_s,
                                  shading="auto", cmap="RdBu_r",
                                  vmin=-Iz_absmax, vmax=Iz_absmax)
        axes[2].set_title(f"$I_z$  z={zv*1e3:.2f} mm")
        plt.colorbar(im2, ax=axes[2], label="W/m²")

        for ax in axes:
            ax.set_xlabel("x [mm]")
            ax.set_ylabel("y [mm]")
            ax.set_aspect("equal")

        fig.suptitle(f"Z-slice {iz+1}/6  z = {zv*1e3:.2f} mm", fontsize=13)
        fig.tight_layout()
        _save(fig, f"B_zslice_{iz:02d}_{zv*1e3:.2f}mm.png")

    print(f"    Saved 6 z-slice triplets to figs/")


# ═════════════════════════════════════════════════════════════════════
#  C — Radial Profile at Trapping Plane
# ═════════════════════════════════════════════════════════════════════

def part_C(sol: PressureSolution, results):
    """Plot |p|(r) at trapping-plane height (mid petri slab)."""
    print("\n  ── Part C: Radial Profile at Trapping Plane ──")
    cfg = sol.cfg

    # Trapping plane = mid-height of petri slab
    z_trap = cfg.H_under + cfg.H_top / 2
    cx, cy = cfg.disk_center_x, cfg.disk_center_y

    # Sample along radial line (average over 8 azimuths)
    n_r = 200
    r_max = min(cfg.Lx, cfg.Ly) / 2 - cfg.t_pml_xy
    r_vals = np.linspace(0, r_max, n_r)
    n_az = 8
    azimuths = np.linspace(0, 2 * np.pi, n_az, endpoint=False)

    p_radial = np.zeros(n_r)
    for theta in azimuths:
        x_r = cx + r_vals * np.cos(theta)
        y_r = cy + r_vals * np.sin(theta)
        z_r = np.full(n_r, z_trap)
        pts = np.column_stack([x_r, y_r, z_r])
        p_r = np.abs(_interp_complex(sol, pts))
        p_radial += p_r
    p_radial /= n_az

    # Find ring max and on-axis value
    p_axis = float(p_radial[0])
    i_max = int(np.argmax(p_radial))
    r_max_val = float(r_vals[i_max])
    p_ring_max = float(p_radial[i_max])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(r_vals * 1e3, p_radial, "b-", lw=2)
    ax.axvline(r_max_val * 1e3, color="r", ls="--", alpha=0.5,
               label=f"Ring max at r={r_max_val*1e3:.2f} mm")
    ax.set_xlabel("r [mm]")
    ax.set_ylabel("|p| [Pa]")
    ax.set_title(f"Radial Profile at Trapping Plane (z={z_trap*1e3:.2f} mm)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, "C_radial_profile.png")

    results["C_radial_profile"] = {
        "z_trap_mm": z_trap * 1e3,
        "p_at_r0_Pa": p_axis,
        "p_ring_max_Pa": p_ring_max,
        "r_ring_max_mm": r_max_val * 1e3,
        "null_confirmed": p_axis < 0.3 * p_ring_max,
    }
    print(f"    z_trap = {z_trap*1e3:.2f} mm")
    print(f"    |p|(r=0) = {p_axis:.4f} Pa")
    print(f"    ring max = {p_ring_max:.4f} Pa at r = {r_max_val*1e3:.2f} mm")
    print(f"    Null confirmed: {p_axis < 0.3 * p_ring_max}")


# ═════════════════════════════════════════════════════════════════════
#  D — Winding Number Computation
# ═════════════════════════════════════════════════════════════════════

def part_D(sol: PressureSolution, results):
    """Discrete winding number on circular ring at trapping plane."""
    print("\n  ── Part D: Winding Number ──")
    cfg = sol.cfg

    z_trap = cfg.H_under + cfg.H_top / 2
    cx, cy = cfg.disk_center_x, cfg.disk_center_y
    ring_r = 0.3e-3
    n_pts = 360
    theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)

    x_ring = cx + ring_r * np.cos(theta)
    y_ring = cy + ring_r * np.sin(theta)
    z_ring = np.full(n_pts, z_trap)
    pts = np.column_stack([x_ring, y_ring, z_ring])

    p_ring = _interp_complex(sol, pts)
    phase = np.angle(p_ring)

    diffs = np.diff(phase)
    diffs_wrapped = (diffs + np.pi) % (2 * np.pi) - np.pi
    total_wind = np.sum(diffs_wrapped)
    close_diff = phase[0] - phase[-1]
    close_wrapped = (close_diff + np.pi) % (2 * np.pi) - np.pi
    total_wind += close_wrapped

    ell = total_wind / (2 * np.pi)

    results["D_winding_number"] = {
        "ell": float(ell),
        "total_winding_rad": float(total_wind),
        "z_mm": z_trap * 1e3,
        "ring_r_mm": ring_r * 1e3,
    }
    print(f"    ℓ = {ell:.4f}  (winding = {total_wind:.4f} rad)")


# ═════════════════════════════════════════════════════════════════════
#  Intensity metrics (for report)
# ═════════════════════════════════════════════════════════════════════

def _compute_Iz_metrics(sol: PressureSolution):
    """Quick mean(I_z) above source and core_ratio (same as vortex diag)."""
    cfg = sol.cfg
    cx, cy = cfg.disk_center_x, cfg.disk_center_y
    R = cfg.disk_radius
    omega, rho = cfg.omega, cfg.rho

    # Sample I_z on a grid above the source
    nz = 100
    z_min = cfg.t_pml_z + 0.2e-3
    z_max = cfg.H_under
    zg = np.linspace(z_min, z_max, nz)

    h_fd = cfg.wavelength / cfg.elements_per_wavelength * 1.5

    Iz_vals = []
    for zv in zg:
        pts_c = np.array([[cx, cy, zv]])
        pts_u = np.array([[cx, cy, zv + h_fd]])
        p_c = _interp_complex(sol, pts_c)
        p_u = _interp_complex(sol, pts_u)
        dp_dz = (p_u - p_c) / h_fd
        vz = 1.0 / (1j * omega * rho) * dp_dz
        Iz_vals.append(0.5 * float(np.real(p_c * np.conj(vz)).item()))

    mean_Iz = float(np.mean(Iz_vals))

    # Core ratio
    p_axis_max = float(np.max(np.abs(_interp_complex(
        sol, np.column_stack([np.full(nz, cx), np.full(nz, cy), zg])))))
    ring_r = 0.4e-3
    n_ring = 8
    theta_ring = np.linspace(0, 2 * np.pi, n_ring, endpoint=False)
    p_ring_max = 0.0
    for zv in zg:
        x_r = cx + ring_r * np.cos(theta_ring)
        y_r = cy + ring_r * np.sin(theta_ring)
        pts_r = np.column_stack([x_r, y_r, np.full(n_ring, zv)])
        p_r = np.max(np.abs(_interp_complex(sol, pts_r)))
        p_ring_max = max(p_ring_max, p_r)

    core_ratio = p_axis_max / p_ring_max if p_ring_max > 0 else float("inf")

    return mean_Iz, core_ratio


# ═════════════════════════════════════════════════════════════════════
#  REPORT
# ═════════════════════════════════════════════════════════════════════

def _build_report(results, ts):
    lines = [
        "# Deliverable 2 — Plastic Lens Propagation Proof\n",
        f"Generated: {ts}\n\n",
    ]

    lines.append("## Solver\n\n")
    s = results["solver"]
    lines.append(f"- DOFs: {s['DOFs']}\n")
    lines.append(f"- max|p|: {s['max_p_Pa']:.2f} Pa\n")
    lines.append(f"- Solver: MUMPS direct\n")
    lines.append(f"- Solve time: {s['solve_time_s']:.1f}s\n\n")

    lines.append("## Summary Metrics\n\n")
    lines.append("| Metric | Value |\n|--------|-------|\n")
    if "D_winding_number" in results:
        lines.append(f"| ℓ (topological charge) | {results['D_winding_number']['ell']:.4f} |\n")
    if "intensity_metrics" in results:
        im = results["intensity_metrics"]
        lines.append(f"| core_ratio | {im['core_ratio']:.4f} |\n")
        lines.append(f"| mean(I_z) above source | {im['mean_Iz_above']:.6e} W/m² |\n")
    lines.append(f"| DOFs | {s['DOFs']} |\n")
    lines.append(f"| Solver | MUMPS direct |\n\n")

    lines.append("## A: Boundary Drive\n\n")
    if "A_boundary_drive" in results:
        a = results["A_boundary_drive"]
        lines.append(f"- Disk DOFs: {a['n_disk_dofs']}\n")
        lines.append(f"- max|v_n|: {a['max_abs_v_n_um_s']:.2f} μm/s\n")
        lines.append(f"- Phase range: {a['phase_range_rad']:.4f} rad\n\n")
    lines.append("![Disk BC](figs/A_disk_boundary_drive.png)\n\n")

    lines.append("## B: Z-Stack Proof\n\n")
    lines.append("6 z-planes with |p|, arg(p), I_z:\n\n")
    for i in range(6):
        lines.append(f"![Z-slice {i+1}](figs/B_zslice_{i:02d}_*.png)\n")
    lines.append("\n")

    lines.append("## C: Radial Profile\n\n")
    if "C_radial_profile" in results:
        c = results["C_radial_profile"]
        lines.append(f"- |p|(r=0) = {c['p_at_r0_Pa']:.4f} Pa\n")
        lines.append(f"- Ring max = {c['p_ring_max_Pa']:.4f} Pa at r = {c['r_ring_max_mm']:.2f} mm\n")
        lines.append(f"- Null confirmed: {c['null_confirmed']}\n\n")
    lines.append("![Radial](figs/C_radial_profile.png)\n\n")

    lines.append("## D: Winding Number\n\n")
    if "D_winding_number" in results:
        d = results["D_winding_number"]
        lines.append(f"- **ℓ = {d['ell']:.4f}**\n")
        lines.append(f"- Winding: {d['total_winding_rad']:.4f} rad\n\n")

    return "".join(lines)


# ═════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════

def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUT.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  DELIVERABLE 2 — PLASTIC LENS PROPAGATION PROOF")
    print(f"  Output: {OUT}")
    print(f"  Time:   {ts}")
    print(f"{'='*70}")

    results = {"timestamp": ts}

    # Solve
    cfg = demo_config(
        standing_velocity_amplitude=0.0,
        top_impedance_Zrel=1.0,
        elements_per_wavelength=5,
    )
    print(f"\n  Solving lens-only …")
    t0 = time.time()
    sol = solve_helmholtz(cfg, verbose=True)
    dt = time.time() - t0
    results["solver"] = {
        "DOFs": sol.dofs,
        "max_p_Pa": sol.max_pressure,
        "solve_time_s": round(dt, 1),
        "ksp_reason": sol.ksp_converged_reason,
    }

    part_A(sol, results)
    part_B(sol, results)
    part_C(sol, results)
    part_D(sol, results)

    # Additional intensity metrics
    print("\n  Computing I_z metrics …")
    mean_Iz, core_ratio = _compute_Iz_metrics(sol)
    results["intensity_metrics"] = {
        "mean_Iz_above": mean_Iz,
        "core_ratio": core_ratio,
    }
    print(f"    mean(I_z) above: {mean_Iz:.6e} W/m²")
    print(f"    core_ratio: {core_ratio:.4f}")

    # Write outputs
    def _conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o

    with open(OUT / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=_conv)
    with open(OUT / "REPORT.md", "w") as f:
        f.write(_build_report(results, ts))

    print(f"\n{'='*70}")
    print(f"  DELIVERABLE 2 COMPLETE")
    print(f"  ℓ = {results['D_winding_number']['ell']:.4f}")
    print(f"  core_ratio = {core_ratio:.4f}")
    print(f"  mean(I_z) = {mean_Iz:.6e}")
    print(f"  Report: {OUT / 'REPORT.md'}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
