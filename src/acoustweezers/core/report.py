"""
Auto-generated audit report and CSV standardisation helpers.

Produces:
  - ``audit/AUDIT_REPORT.md`` — human-readable run summary
  - ``csv/summary_cases.csv`` — one row per case
  - ``csv/roi_metrics.csv`` — ROI metrics (standardised columns)
  - ``csv/focus_metrics.csv`` — z_focus, peak |p|, ring radius estimate
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


def write_summary_cases_csv(
    out_dir: Path,
    cases: List[Dict],
) -> Path:
    """
    Write ``csv/summary_cases.csv`` — one row per solved case.

    Each dict in *cases* should have keys:
        case, dofs, ksp_type, pc_type, ksp_converged_reason,
        max_abs_p_Pa, wall_time_s, status
    """
    csv_dir = out_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    out = csv_dir / "summary_cases.csv"

    fieldnames = [
        "case", "dofs", "ksp_type", "pc_type",
        "ksp_converged_reason", "max_abs_p_Pa", "wall_time_s", "status",
    ]
    # Add any extra keys from the first row
    if cases:
        for k in cases[0]:
            if k not in fieldnames:
                fieldnames.append(k)

    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(cases)
    return out


def write_roi_metrics_csv(
    out_dir: Path,
    rows: List[Dict],
) -> Path:
    """
    Write ``csv/roi_metrics.csv`` with standardised columns.

    Columns: case, mean_abs_p, max_abs_p, energy_physical, energy_pml, energy_ratio
    """
    csv_dir = out_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    out = csv_dir / "roi_metrics.csv"

    fieldnames = [
        "case", "mean_abs_p", "max_abs_p",
        "energy_physical", "energy_pml", "energy_ratio",
    ]

    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return out


def write_focus_metrics_csv(
    out_dir: Path,
    rows: List[Dict],
) -> Path:
    """
    Write ``csv/focus_metrics.csv``.

    Columns: case, z_focus_mm, peak_p_Pa, ring_radius_mm
    """
    csv_dir = out_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    out = csv_dir / "focus_metrics.csv"

    fieldnames = ["case", "z_focus_mm", "peak_p_Pa", "ring_radius_mm"]

    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return out


def write_audit_report(
    run_dir: Path,
    *,
    cases_status: Dict[str, str],
    dofs_by_case: Dict[str, int],
    walltimes: Dict[str, float],
    pml_settings: Dict,
    warnings: List[str],
    total_walltime_s: float = 0.0,
) -> Path:
    """
    Write ``audit/AUDIT_REPORT.md``.

    Parameters
    ----------
    run_dir : Path
        Run root directory.
    cases_status : dict
        case_name -> "OK" | "FAILED: reason"
    dofs_by_case : dict
        case_name -> DOF count
    walltimes : dict
        case_name -> seconds
    pml_settings : dict
        PML parameters used.
    warnings : list of str
        Any warnings to include.
    total_walltime_s : float
        Total pipeline wall time.

    Returns
    -------
    Path
    """
    audit_dir = run_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    out = audit_dir / "AUDIT_REPORT.md"

    lines = [
        "# Audit Report\n",
        f"**Generated:** {datetime.now().isoformat()}\n",
        f"**Run directory:** `{run_dir}`\n",
        f"**Total wall time:** {total_walltime_s:.1f} s "
        f"({total_walltime_s/60:.1f} min)\n",
        "",
        "## Case Status\n",
        "| Case | Status | DOFs | Wall time [s] |",
        "|------|--------|------|---------------|",
    ]

    n_ok = 0
    n_fail = 0
    for case, status in cases_status.items():
        dofs = dofs_by_case.get(case, 0)
        wt = walltimes.get(case, 0.0)
        icon = "✅" if "OK" in status.upper() or "YES" in status.upper() else "❌"
        lines.append(f"| {case} | {icon} {status} | {dofs:,} | {wt:.1f} |")
        if "FAIL" in status.upper():
            n_fail += 1
        else:
            n_ok += 1

    lines.append("")
    lines.append(f"**Summary:** {n_ok} succeeded, {n_fail} failed\n")

    # PML settings
    lines.append("## PML Settings\n")
    for k, v in pml_settings.items():
        lines.append(f"- **{k}:** {v}")
    lines.append("")

    # Warnings
    if warnings:
        lines.append("## Warnings\n")
        for w in warnings:
            lines.append(f"- ⚠️ {w}")
        lines.append("")
    else:
        lines.append("## Warnings\n")
        lines.append("None.\n")

    out.write_text("\n".join(lines))
    return out
