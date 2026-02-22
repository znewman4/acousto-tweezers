#!/usr/bin/env python3
"""
COMSOL ↔ FEniCSx Quantitative Comparison Pipeline
===================================================

Reads COMSOL CSV grid exports and FEniCSx XDMF fields, interpolates
onto a common grid, and produces difference heatmaps + error metrics.

Usage::

    python scripts/comsol/compare_to_comsol.py \\
        --fenics-run results/run_20260222_120000 \\
        --comsol-dir /path/to/comsol_exports

See ``comsol_compare/README.md`` for the COMSOL export specification.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


# ═════════════════════════════════════════════════════════════════════
#  CSV READER
# ═════════════════════════════════════════════════════════════════════

def read_comsol_csv(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read a COMSOL CSV export with columns (x, y, value) or (x, z, value).

    Returns (coord1, coord2, values) as 1-D arrays.
    """
    data = np.loadtxt(str(path), delimiter=",", skiprows=1)
    if data.shape[1] < 3:
        raise ValueError(f"Expected >= 3 columns in {path}, got {data.shape[1]}")
    return data[:, 0], data[:, 1], data[:, 2]


def _grid_from_scatter(c1, c2, vals, n1=300, n2=300):
    """Re-grid scattered (c1, c2, vals) onto a regular grid."""
    from scipy.interpolate import griddata
    g1 = np.linspace(np.nanmin(c1), np.nanmax(c1), n1)
    g2 = np.linspace(np.nanmin(c2), np.nanmax(c2), n2)
    G1, G2 = np.meshgrid(g1, g2)
    grid_vals = griddata((c1, c2), vals, (G1, G2), method="linear")
    return g1, g2, grid_vals


# ═════════════════════════════════════════════════════════════════════
#  FENICS FIELD READER
# ═════════════════════════════════════════════════════════════════════

def _try_read_xdmf(xdmf_path: Path, field_name: str):
    """Read XDMF and return (coords, values) or None."""
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
        return V.tabulate_dof_coordinates(), func.x.array.copy()
    except Exception:
        return None


def _interp_fenics_to_grid(coords, values, g1, g2, slice_type, fixed_val, cfg):
    """Interpolate FEniCSx DOF values onto a structured grid."""
    from scipy.interpolate import NearestNDInterpolator
    interp = NearestNDInterpolator(coords, values)

    G1, G2 = np.meshgrid(g1, g2)

    if slice_type == "xy":
        pts = np.column_stack([G1.ravel(), G2.ravel(),
                               np.full(G1.size, fixed_val)])
    elif slice_type == "xz":
        pts = np.column_stack([G1.ravel(),
                               np.full(G1.size, fixed_val),
                               G2.ravel()])
    elif slice_type == "yz":
        pts = np.column_stack([np.full(G1.size, fixed_val),
                               G1.ravel(), G2.ravel()])
    else:
        raise ValueError(f"Unknown slice_type: {slice_type}")

    return interp(pts).reshape(G1.shape)


# ═════════════════════════════════════════════════════════════════════
#  METRICS
# ═════════════════════════════════════════════════════════════════════

def compute_error_metrics(comsol_grid, fenics_grid):
    """Compute L2, Linf relative errors and spatial correlation."""
    mask = np.isfinite(comsol_grid) & np.isfinite(fenics_grid)
    c = comsol_grid[mask]
    f = fenics_grid[mask]

    if len(c) == 0:
        return {"L2_rel": float("nan"), "Linf_rel": float("nan"),
                "pearson_r": float("nan"), "n_valid": 0}

    diff = f - c
    L2_abs = np.sqrt(np.mean(diff**2))
    L2_ref = np.sqrt(np.mean(c**2))
    L2_rel = L2_abs / (L2_ref + 1e-30)

    Linf = float(np.max(np.abs(diff)))
    Linf_ref = float(np.max(np.abs(c)))
    Linf_rel = Linf / (Linf_ref + 1e-30)

    # Pearson correlation
    if np.std(c) > 1e-30 and np.std(f) > 1e-30:
        pearson_r = float(np.corrcoef(c, f)[0, 1])
    else:
        pearson_r = float("nan")

    return {
        "L2_rel": float(L2_rel),
        "Linf_rel": float(Linf_rel),
        "pearson_r": pearson_r,
        "n_valid": int(len(c)),
        "max_comsol": float(np.max(c)),
        "max_fenics": float(np.max(f)),
        "max_diff": float(np.max(np.abs(diff))),
    }


# ═════════════════════════════════════════════════════════════════════
#  COMPARISON DRIVER
# ═════════════════════════════════════════════════════════════════════

SLICE_MAP = {
    "xy_midplane_pmag.csv": {"slice": "xy", "field": "p_mag"},
    "xy_midplane_phase.csv": {"slice": "xy", "field": "p_phase"},
    "xz_center_pmag.csv": {"slice": "xz", "field": "p_mag"},
    "yz_center_pmag.csv": {"slice": "yz", "field": "p_mag"},
}


def compare_case(
    fenics_fields_dir: Path,
    comsol_case_dir: Path,
    output_dir: Path,
    cfg: dict,
    case_name: str,
) -> dict:
    """Compare one case. Returns metrics dict."""
    output_dir.mkdir(parents=True, exist_ok=True)

    Lx = cfg.get("Lx", 6e-3)
    Ly = cfg.get("Ly", 6e-3)
    H_under = cfg.get("H_under", 3e-3)
    H_top = cfg.get("H_top", 2e-3)
    z_mid = H_under + H_top / 2.0

    fixed_vals = {
        "xy": z_mid,
        "xz": Ly / 2,
        "yz": Lx / 2,
    }

    case_metrics = {}

    for csv_name, info in SLICE_MAP.items():
        comsol_csv = comsol_case_dir / csv_name
        if not comsol_csv.exists():
            continue

        slice_type = info["slice"]
        field_name = info["field"]
        xdmf_path = fenics_fields_dir / f"{field_name}.xdmf"

        if not xdmf_path.exists():
            print(f"    WARNING: {xdmf_path} not found — skipping {csv_name}")
            continue

        label = csv_name.replace(".csv", "")
        print(f"    Comparing {label} …")

        # Read COMSOL
        c1, c2, c_vals = read_comsol_csv(comsol_csv)
        g1, g2, comsol_grid = _grid_from_scatter(c1, c2, c_vals)

        # Read FEniCSx
        result = _try_read_xdmf(xdmf_path, field_name)
        if result is None:
            print(f"    WARNING: Cannot read {xdmf_path}")
            continue
        coords, f_vals = result
        fenics_grid = _interp_fenics_to_grid(
            coords, f_vals, g1, g2, slice_type, fixed_vals[slice_type], cfg)

        # Compute metrics
        metrics = compute_error_metrics(comsol_grid, fenics_grid)
        case_metrics[label] = metrics

        # Difference heatmap
        diff = fenics_grid - comsol_grid
        vabs = max(abs(np.nanmin(diff)), abs(np.nanmax(diff)), 1e-15)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        im0 = axes[0].pcolormesh(g1 * 1e3, g2 * 1e3, comsol_grid,
                                  shading="auto", cmap="inferno")
        axes[0].set_title(f"COMSOL — {label}")
        plt.colorbar(im0, ax=axes[0])

        im1 = axes[1].pcolormesh(g1 * 1e3, g2 * 1e3, fenics_grid,
                                  shading="auto", cmap="inferno")
        axes[1].set_title(f"FEniCSx — {label}")
        plt.colorbar(im1, ax=axes[1])

        im2 = axes[2].pcolormesh(g1 * 1e3, g2 * 1e3, diff,
                                  shading="auto", cmap="RdBu_r",
                                  vmin=-vabs, vmax=vabs)
        axes[2].set_title(f"Difference (FEniCSx − COMSOL)")
        plt.colorbar(im2, ax=axes[2])

        for ax in axes:
            ax.set_xlabel("coord1 [mm]")
            ax.set_ylabel("coord2 [mm]")
            ax.set_aspect("equal")

        fig.suptitle(f"{case_name} / {label}\n"
                     f"L2_rel={metrics['L2_rel']:.3f}  "
                     f"Linf_rel={metrics['Linf_rel']:.3f}  "
                     f"r={metrics['pearson_r']:.4f}")
        fig.tight_layout()
        fig.savefig(output_dir / f"diff_{label}.png", dpi=200)
        plt.close(fig)

    return case_metrics


# ═════════════════════════════════════════════════════════════════════
#  REPORT WRITER
# ═════════════════════════════════════════════════════════════════════

def write_report(all_metrics: dict, output_dir: Path, cfg: dict):
    """Write REPORT.md with pass/fail metrics and summary tables."""
    lines = [
        "# COMSOL ↔ FEniCSx Comparison Report\n",
        f"**Generated:** {datetime.now().isoformat()}\n",
        "## Pass/Fail Criteria\n",
        "| Metric | Threshold | Description |",
        "|--------|-----------|-------------|",
        "| Focus z-location error | < 0.5 mm | argmax(\\|p\\|) on centerline |",
        "| Max \\|p\\| error in ROI | < 20% relative | physical region only |",
        "| Spatial correlation (XY mag) | > 0.85 | Pearson r |",
        "| L2 relative error | < 30% | per slice |",
        "",
    ]

    for case_name, case_metrics in all_metrics.items():
        lines.append(f"## Case: {case_name}\n")
        lines.append("| Slice | L2 rel | L∞ rel | Pearson r | max COMSOL | max FEniCSx | Pass? |")
        lines.append("|-------|--------|--------|-----------|------------|-------------|-------|")
        for slice_name, m in case_metrics.items():
            l2 = m.get("L2_rel", float("nan"))
            linf = m.get("Linf_rel", float("nan"))
            r = m.get("pearson_r", float("nan"))
            mc = m.get("max_comsol", float("nan"))
            mf = m.get("max_fenics", float("nan"))
            # Advisory pass/fail
            passed = l2 < 0.3 and r > 0.85 if np.isfinite(l2) and np.isfinite(r) else False
            icon = "✅" if passed else "❌"
            lines.append(
                f"| {slice_name} | {l2:.4f} | {linf:.4f} | {r:.4f} | "
                f"{mc:.2f} | {mf:.2f} | {icon} |"
            )
        lines.append("")

    report_path = output_dir / "REPORT.md"
    report_path.write_text("\n".join(lines))
    print(f"  Wrote {report_path}")

    # Also save raw metrics as JSON
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)


# ═════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="Compare FEniCSx results to COMSOL exports")
    p.add_argument("--fenics-run", type=str, required=True,
                   help="Path to FEniCSx run directory")
    p.add_argument("--comsol-dir", type=str, required=True,
                   help="Path to COMSOL export directory")
    args = p.parse_args()

    fenics_run = Path(args.fenics_run)
    comsol_dir = Path(args.comsol_dir)

    if not fenics_run.exists():
        sys.exit(f"ERROR: FEniCSx run not found: {fenics_run}")
    if not comsol_dir.exists():
        sys.exit(f"ERROR: COMSOL directory not found: {comsol_dir}")

    # Load config
    cfg = {}
    for candidate in [fenics_run / "config.json",
                      fenics_run / "production" / "config.json"]:
        if candidate.exists():
            with open(candidate) as f:
                cfg = json.load(f)
            break

    # Find fields directory
    fields_root = fenics_run / "fields"
    if not fields_root.exists():
        # Try production subdirectory
        fields_root = fenics_run / "production" / "fields"
    if not fields_root.exists():
        sys.exit(f"ERROR: No fields/ directory found. Run with --export-fields first.")

    # Find COMSOL cases
    comsol_cases = sorted([d.name for d in comsol_dir.iterdir()
                           if d.is_dir()])
    if not comsol_cases:
        # Try flat structure (CSVs directly in comsol_dir)
        comsol_cases = [""]

    output_dir = fenics_run / "comsol_compare"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  COMSOL comparison pipeline")
    print(f"  FEniCSx run: {fenics_run}")
    print(f"  COMSOL dir:  {comsol_dir}")
    print(f"  Cases:       {comsol_cases}")
    print()

    all_metrics = {}

    for case_name in comsol_cases:
        comsol_case_dir = comsol_dir / case_name if case_name else comsol_dir
        fenics_case_fields = fields_root / case_name if case_name else fields_root

        if not fenics_case_fields.exists():
            print(f"  WARNING: No FEniCSx fields for case '{case_name}', skipping")
            continue

        print(f"  Comparing case: {case_name or '<root>'}")
        case_out = output_dir / case_name if case_name else output_dir
        metrics = compare_case(fenics_case_fields, comsol_case_dir,
                               case_out, cfg, case_name or "root")
        all_metrics[case_name or "root"] = metrics

    write_report(all_metrics, output_dir, cfg)
    print(f"\n  Comparison complete → {output_dir}")


if __name__ == "__main__":
    main()
