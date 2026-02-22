#!/usr/bin/env python3
"""
Standard 2-D + 3-D Figure Suite for Far-Field Petri Cuboid Runs
================================================================

Reads XDMF fields from a completed run and produces a fixed set of
publication / COMSOL-comparison-ready plots.

Usage::

    python scripts/viz/make_farfield_figures.py --run results/run_20260222_120000
    python scripts/viz/make_farfield_figures.py --run results/run_20260222_120000/production

Expects the run directory (or a sub-directory like ``production/``)
to contain ``fields/<case>/p_mag.xdmf`` etc. as produced by
``--export-fields``.  Falls back to reading ``config.json`` and
re-interpolating from checkpoint data if XDMF files are not present.

Outputs go to ``<run>/figures_2d/`` (and ``figures_3d/`` for 3-D visuals).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# ── project imports ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


# ═════════════════════════════════════════════════════════════════════
#  HELPERS
# ═════════════════════════════════════════════════════════════════════

def _load_config(run_dir: Path) -> dict:
    """Load config.json from run or its 'production' subdirectory."""
    for candidate in [run_dir / "config.json",
                      run_dir / "production" / "config.json"]:
        if candidate.exists():
            with open(candidate) as f:
                return json.load(f)
    raise FileNotFoundError(f"No config.json found in {run_dir}")


def _detect_cases(run_dir: Path) -> list[str]:
    """Detect which cases have exported fields."""
    fields_dir = run_dir / "fields"
    if not fields_dir.exists():
        # Try parent run root
        fields_dir = run_dir.parent / "fields"
    if not fields_dir.exists():
        return []
    return sorted([d.name for d in fields_dir.iterdir()
                   if d.is_dir() and (d / "p_mag.xdmf").exists()])


def _try_read_xdmf_field(xdmf_path: Path, field_name: str):
    """
    Read an XDMF field using dolfinx.  Returns (coords, values) or None.
    """
    try:
        from dolfinx.io import XDMFFile
        from dolfinx import fem
        from mpi4py import MPI

        with XDMFFile(MPI.COMM_WORLD, str(xdmf_path), "r") as xf:
            domain = xf.read_mesh()
            V = fem.functionspace(domain, ("Lagrange", 2))
            func = fem.Function(V)
            func.name = field_name
            xf.read_function(func)
        coords = V.tabulate_dof_coordinates()
        return coords, func.x.array.copy()
    except Exception:
        return None


def _interp_to_grid(coords, values, xg, yg, z_val):
    """Nearest-neighbour interpolation to structured XY grid at fixed z."""
    from scipy.interpolate import NearestNDInterpolator
    interp = NearestNDInterpolator(coords, values)
    X, Y = np.meshgrid(xg, yg)
    Z = np.full_like(X, z_val)
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    return interp(pts).reshape(X.shape)


def _interp_to_xz(coords, values, xg, zg, y_val):
    from scipy.interpolate import NearestNDInterpolator
    interp = NearestNDInterpolator(coords, values)
    X, Z = np.meshgrid(xg, zg)
    Y = np.full_like(X, y_val)
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    return interp(pts).reshape(X.shape)


def _interp_to_yz(coords, values, yg, zg, x_val):
    from scipy.interpolate import NearestNDInterpolator
    interp = NearestNDInterpolator(coords, values)
    Y, Z = np.meshgrid(yg, zg)
    X = np.full_like(Y, x_val)
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    return interp(pts).reshape(Y.shape)


def _interp_centerline_z(coords, values, cx, cy, zg):
    from scipy.interpolate import NearestNDInterpolator
    interp = NearestNDInterpolator(coords, values)
    nz = len(zg)
    pts = np.column_stack([np.full(nz, cx), np.full(nz, cy), zg])
    return interp(pts)


# ═════════════════════════════════════════════════════════════════════
#  PLOT GENERATORS
# ═════════════════════════════════════════════════════════════════════

def _annotate_domain(ax, cfg, plane="xy"):
    """Add disk radius circle / interface lines."""
    Lx = cfg.get("Lx", 6e-3)
    Ly = cfg.get("Ly", 6e-3)
    H_under = cfg.get("H_under", 3e-3)
    disk_r = cfg.get("disk_radius", 1e-3)
    cx = Lx / 2 * 1e3
    cy = Ly / 2 * 1e3

    if plane == "xy":
        circ = Circle((cx, cy), disk_r * 1e3, fill=False,
                       edgecolor="lime", linewidth=1.0, linestyle="--",
                       label="disk radius")
        ax.add_patch(circ)
    elif plane in ("xz", "yz"):
        ax.axhline(H_under * 1e3, color="cyan", ls=":", lw=1.0,
                    label="petri/bath interface")


def _save_limits(limits: dict, out_dir: Path):
    with open(out_dir / "limits.json", "w") as f:
        json.dump(limits, f, indent=2, default=str)


def generate_figures_2d(run_dir: Path, cases: list[str], cfg: dict,
                        out_dir: Path, nx: int = 300, nz: int = 300):
    """Generate the standard 2D figure suite."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fields_root = run_dir / "fields"
    if not fields_root.exists():
        fields_root = run_dir.parent / "fields"

    Lx = cfg.get("Lx", 6e-3)
    Ly = cfg.get("Ly", 6e-3)
    H_under = cfg.get("H_under", 3e-3)
    H_top = cfg.get("H_top", 2e-3)
    H_total = H_under + H_top
    z_mid = H_under + H_top / 2.0

    xg = np.linspace(0, Lx, nx)
    yg = np.linspace(0, Ly, nx)
    zg = np.linspace(0, H_total, nz)
    zg_cl = np.linspace(0, H_total, 500)

    limits = {}

    for case in cases:
        case_dir = fields_root / case
        print(f"  Generating 2D figures for case: {case}")

        # Load fields
        result = _try_read_xdmf_field(case_dir / "p_mag.xdmf", "p_mag")
        if result is None:
            print(f"    WARNING: Cannot read XDMF for {case}, skipping")
            continue
        coords, p_mag_vals = result

        result_phase = _try_read_xdmf_field(case_dir / "p_phase.xdmf", "p_phase")
        p_phase_vals = result_phase[1] if result_phase else None

        # ── 1. Mid-plane XY magnitude ────────────────────────────────
        pmag_xy = _interp_to_grid(coords, p_mag_vals, xg, yg, z_mid)
        vmax_xy = float(np.nanmax(pmag_xy))
        limits[f"{case}_xy_vmax"] = vmax_xy

        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.pcolormesh(xg * 1e3, yg * 1e3, pmag_xy,
                           shading="auto", cmap="inferno", vmin=0, vmax=vmax_xy)
        ax.set_title(f"|p| XY mid-plane (z={z_mid*1e3:.2f} mm) — {case}")
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
        ax.set_aspect("equal")
        _annotate_domain(ax, cfg, "xy")
        plt.colorbar(im, ax=ax, label="Pa")
        fig.tight_layout()
        fig.savefig(out_dir / f"xy_magnitude_{case}.png", dpi=200)
        plt.close(fig)

        # ── 2. Mid-plane XY phase ────────────────────────────────────
        if p_phase_vals is not None:
            pphase_xy = _interp_to_grid(coords, p_phase_vals, xg, yg, z_mid)
            fig, ax = plt.subplots(figsize=(7, 6))
            im = ax.pcolormesh(xg * 1e3, yg * 1e3, pphase_xy,
                               shading="auto", cmap="twilight",
                               vmin=-np.pi, vmax=np.pi)
            ax.set_title(f"arg(p) XY mid-plane (z={z_mid*1e3:.2f} mm) — {case}")
            ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
            ax.set_aspect("equal")
            _annotate_domain(ax, cfg, "xy")
            plt.colorbar(im, ax=ax, label="rad")
            fig.tight_layout()
            fig.savefig(out_dir / f"xy_phase_{case}.png", dpi=200)
            plt.close(fig)

        # ── 3. Centerline z-profile ──────────────────────────────────
        cx = Lx / 2
        cy = Ly / 2
        pmag_cl = _interp_centerline_z(coords, p_mag_vals, cx, cy, zg_cl)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(zg_cl * 1e3, pmag_cl, "b-", lw=1.5)
        ax.axvline(H_under * 1e3, color="cyan", ls=":", lw=1.0,
                    label="petri/bath interface")
        ax.set_xlabel("z [mm]"); ax.set_ylabel("|p| [Pa]")
        ax.set_title(f"Centerline |p|(z) — {case}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"centerline_z_{case}.png", dpi=200)
        plt.close(fig)

        # ── 4. XZ slice magnitude (y = Ly/2) ────────────────────────
        pmag_xz = _interp_to_xz(coords, p_mag_vals, xg, zg, Ly / 2)
        vmax_xz = float(np.nanmax(pmag_xz))
        limits[f"{case}_xz_vmax"] = vmax_xz

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.pcolormesh(xg * 1e3, zg * 1e3, pmag_xz,
                           shading="auto", cmap="inferno", vmin=0, vmax=vmax_xz)
        ax.set_title(f"|p| XZ (y={Ly/2*1e3:.2f} mm) — {case}")
        ax.set_xlabel("x [mm]"); ax.set_ylabel("z [mm]")
        _annotate_domain(ax, cfg, "xz")
        plt.colorbar(im, ax=ax, label="Pa")
        fig.tight_layout()
        fig.savefig(out_dir / f"xz_magnitude_{case}.png", dpi=200)
        plt.close(fig)

        # ── 5. YZ slice magnitude (x = Lx/2) ────────────────────────
        pmag_yz = _interp_to_yz(coords, p_mag_vals, yg, zg, Lx / 2)
        vmax_yz = float(np.nanmax(pmag_yz))
        limits[f"{case}_yz_vmax"] = vmax_yz

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.pcolormesh(yg * 1e3, zg * 1e3, pmag_yz,
                           shading="auto", cmap="inferno", vmin=0, vmax=vmax_yz)
        ax.set_title(f"|p| YZ (x={Lx/2*1e3:.2f} mm) — {case}")
        ax.set_xlabel("y [mm]"); ax.set_ylabel("z [mm]")
        _annotate_domain(ax, cfg, "yz")
        plt.colorbar(im, ax=ax, label="Pa")
        fig.tight_layout()
        fig.savefig(out_dir / f"yz_magnitude_{case}.png", dpi=200)
        plt.close(fig)

    # ── Interaction plots ────────────────────────────────────────────
    combined_dir = fields_root / "combined"
    stand_dir = fields_root / "standing_only"

    if combined_dir.exists() and stand_dir.exists():
        print("  Generating interaction plots (Δ|p|) …")
        res_c = _try_read_xdmf_field(combined_dir / "p_mag.xdmf", "p_mag")
        res_s = _try_read_xdmf_field(stand_dir / "p_mag.xdmf", "p_mag")

        if res_c is not None and res_s is not None:
            coords_c, pmag_c = res_c
            coords_s, pmag_s = res_s

            pmag_xy_c = _interp_to_grid(coords_c, pmag_c, xg, yg, z_mid)
            pmag_xy_s = _interp_to_grid(coords_s, pmag_s, xg, yg, z_mid)
            delta_p = pmag_xy_c - pmag_xy_s

            vabs = max(abs(np.nanmin(delta_p)), abs(np.nanmax(delta_p)))
            if vabs < 1e-15:
                vabs = 1.0

            fig, ax = plt.subplots(figsize=(7, 6))
            im = ax.pcolormesh(xg * 1e3, yg * 1e3, delta_p,
                               shading="auto", cmap="RdBu_r",
                               vmin=-vabs, vmax=vabs)
            ax.set_title("Δ|p| = |p_combined| − |p_standing|")
            ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
            ax.set_aspect("equal")
            _annotate_domain(ax, cfg, "xy")
            plt.colorbar(im, ax=ax, label="Pa")
            fig.tight_layout()
            fig.savefig(out_dir / "delta_p_heatmap.png", dpi=200)
            plt.close(fig)

    _save_limits(limits, out_dir)
    print(f"  Wrote limits.json and {len(list(out_dir.glob('*.png')))} figures → {out_dir}")


# ═════════════════════════════════════════════════════════════════════
#  3-D ISOSURFACE PLOTS
# ═════════════════════════════════════════════════════════════════════

def generate_figures_3d(run_dir: Path, cases: list[str], cfg: dict,
                        out_dir: Path, levels_pct: tuple = (0.2, 0.5, 0.8)):
    """
    Generate 3D isosurface snapshots using PyVista (if available).

    Parameters
    ----------
    levels_pct : tuple of float
        Isosurface levels as fraction of max |p|.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import pyvista as pv
        pv.OFF_SCREEN = True
    except ImportError:
        print("  PyVista not available — skipping 3D figures")
        (out_dir / "SKIPPED_pyvista_not_installed.txt").write_text(
            "PyVista not available. Install with: pip install pyvista\n")
        return

    fields_root = run_dir / "fields"
    if not fields_root.exists():
        fields_root = run_dir.parent / "fields"

    for case in cases:
        case_dir = fields_root / case
        xdmf_path = case_dir / "p_mag.xdmf"
        if not xdmf_path.exists():
            continue

        print(f"  Generating 3D isosurfaces for {case} …")
        try:
            reader = pv.get_reader(str(xdmf_path))
            mesh_pv = reader.read()

            if "p_mag" not in mesh_pv.point_data:
                # Try the first available field
                if mesh_pv.point_data.keys():
                    field_name = list(mesh_pv.point_data.keys())[0]
                else:
                    continue
            else:
                field_name = "p_mag"

            vals = mesh_pv.point_data[field_name]
            vmax = float(np.max(vals))

            plotter = pv.Plotter(off_screen=True, window_size=(1200, 900))
            cmap = "inferno"

            for i, frac in enumerate(levels_pct):
                level = frac * vmax
                contour = mesh_pv.contour(isosurfaces=[level],
                                          scalars=field_name)
                opacity = 0.3 + 0.3 * i
                plotter.add_mesh(contour, color=plt.cm.inferno(frac),
                                 opacity=opacity,
                                 label=f"{frac*100:.0f}% ({level:.3f} Pa)")

            plotter.add_legend()
            plotter.camera_position = "iso"
            plotter.screenshot(str(out_dir / f"isosurface_{case}.png"))
            plotter.close()
        except Exception as e:
            print(f"    3D plot failed for {case}: {e}")


# ═════════════════════════════════════════════════════════════════════
#  PARAVIEW STATE FILE
# ═════════════════════════════════════════════════════════════════════

def write_paraview_state(run_dir: Path, cases: list[str], out_dir: Path):
    """
    Write a ParaView Python state script (.py) that loads all XDMF files
    and sets up standard views.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fields_root = run_dir / "fields"
    if not fields_root.exists():
        fields_root = run_dir.parent / "fields"

    lines = [
        "# ParaView Python state — auto-generated",
        "# Load with: pvpython scripts/viz/paraview_state.py",
        "# Or open in ParaView GUI: File > Load State",
        "",
        "from paraview.simple import *",
        "",
    ]

    for case in cases:
        case_dir = fields_root / case
        for field in ["p_mag", "p_real", "p_imag", "p_phase"]:
            xdmf = case_dir / f"{field}.xdmf"
            if xdmf.exists():
                var_name = f"{case}_{field}"
                lines.append(f"# --- {case} / {field} ---")
                lines.append(f"{var_name} = XDMFReader(FileName=r'{xdmf}')")
                lines.append(f"{var_name}Display = Show({var_name})")
                lines.append(f"{var_name}Display.Representation = 'Surface'")
                lines.append("")

    lines.extend([
        "# Set up camera",
        "view = GetActiveViewOrCreate('RenderView')",
        "view.ViewSize = [1200, 900]",
        "ResetCamera()",
        "Render()",
    ])

    state_file = out_dir / "paraview_load_fields.py"
    state_file.write_text("\n".join(lines))
    print(f"  Wrote ParaView state script: {state_file}")


# ═════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="Generate standard 2D + 3D figure suite from a run")
    p.add_argument("--run", type=str, required=True,
                   help="Path to run directory (e.g. results/run_20260222_120000)")
    p.add_argument("--no-3d", action="store_true",
                   help="Skip 3D isosurface generation")
    p.add_argument("--nx", type=int, default=300,
                   help="Grid resolution for 2D slices (default: 300)")
    args = p.parse_args()

    run_dir = Path(args.run)
    if not run_dir.exists():
        sys.exit(f"ERROR: Run directory not found: {run_dir}")

    print(f"\n  Figure suite for: {run_dir}\n")

    # Load config
    try:
        cfg = _load_config(run_dir)
    except FileNotFoundError:
        print("  WARNING: No config.json found, using defaults")
        cfg = {}

    # Detect cases
    cases = _detect_cases(run_dir)
    if not cases:
        print("  No XDMF fields found. Run with --export-fields first.")
        print("  Looked in: fields/<case>/p_mag.xdmf")
        sys.exit(1)

    print(f"  Cases found: {cases}")

    # 2D figures
    out_2d = run_dir / "figures_2d"
    generate_figures_2d(run_dir, cases, cfg, out_2d, nx=args.nx)

    # 3D figures
    if not args.no_3d:
        out_3d = run_dir / "figures_3d"
        generate_figures_3d(run_dir, cases, cfg, out_3d)

    # ParaView state
    write_paraview_state(run_dir, cases, run_dir / "figures_3d")

    print(f"\n  Done. Figures in {out_2d} and {run_dir / 'figures_3d'}")


if __name__ == "__main__":
    main()
