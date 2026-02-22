"""
Case configuration loader.

Loads a canonical case JSON and merges it into FarFieldConfig overrides.
The JSON ``flat_overrides`` key is the authoritative set of FarFieldConfig
field names that the case file controls.

Usage in scripts::

    from acoustweezers.core.case_loader import load_case_overrides, write_case_summary
    overrides = load_case_overrides("configs/cases/canonical_farfield.json")
    cfg = FarFieldConfig(**{**CORRECTED_PRESET, **overrides})
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


def load_case_overrides(case_path: str | Path) -> Dict:
    """
    Read a case JSON and return the ``flat_overrides`` dict.

    The ``flat_overrides`` key must map directly to
    :class:`~acoustweezers.experiments.farfield_petri_cuboid.config.FarFieldConfig`
    constructor keyword arguments.

    Parameters
    ----------
    case_path : str or Path
        Path to the canonical case JSON file.

    Returns
    -------
    dict
        Keyword overrides suitable for ``FarFieldConfig(**overrides)``.
    """
    case_path = Path(case_path)
    if not case_path.exists():
        raise FileNotFoundError(f"Case file not found: {case_path}")

    with open(case_path) as f:
        data = json.load(f)

    if "flat_overrides" not in data:
        raise KeyError(
            f"Case file {case_path} has no 'flat_overrides' key.  "
            "The JSON must contain a 'flat_overrides' dict whose keys "
            "match FarFieldConfig fields."
        )

    return dict(data["flat_overrides"])


def write_case_summary(
    run_dir: Path,
    case_path: Optional[str | Path],
    effective_params: Dict,
    *,
    extra_info: Optional[Dict] = None,
) -> Path:
    """
    Write ``CASE_SUMMARY.md`` describing the case in human-readable form.

    Parameters
    ----------
    run_dir : Path
        Root of the results run directory.
    case_path : str, Path, or None
        Path to the case JSON that was loaded (``None`` if presets used).
    effective_params : dict
        The final merged parameter dict that was actually used.
    extra_info : dict, optional
        Additional key/value pairs to include.

    Returns
    -------
    Path
        Path to the written file.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "CASE_SUMMARY.md"

    lines = [
        "# Case Summary\n",
        f"**Generated:** {datetime.now().isoformat()}\n",
    ]

    if case_path is not None:
        lines.append(f"**Case file:** `{case_path}`\n")
    else:
        lines.append("**Case file:** *(none — using built-in presets)*\n")

    # Human-readable parameter table
    lines.append("## Parameters\n")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")

    # Group for readability
    _groups = {
        "Geometry": ["Lx", "Ly", "H_under", "H_top"],
        "Frequency": ["frequency_hz"],
        "Disk / Lens": [
            "disk_radius", "disk_velocity_amplitude",
            "vortex_topological_charge",
            "lens_drive", "lens_l", "lens_focal_length",
            "lens_focus_offset_x", "lens_focus_offset_y",
            "lens_c_lens", "lens_apodization", "lens_apodization_strength",
        ],
        "Standing wave": [
            "standing_velocity_amplitude", "standing_phase_pattern",
            "standing_axis",
        ],
        "PML": [
            "pml_n_wavelengths_xy", "pml_n_wavelengths_z",
            "pml_degree", "pml_sigma_max_factor", "pml_enabled",
        ],
    }
    listed = set()
    for group, keys in _groups.items():
        lines.append(f"| **{group}** | |")
        for k in keys:
            if k in effective_params:
                v = effective_params[k]
                # Pretty-print lengths in mm
                if isinstance(v, float) and abs(v) < 1 and abs(v) > 1e-9 and k not in ("pml_sigma_max_factor", "lens_apodization_strength"):
                    lines.append(f"| {k} | {v} ({v*1e3:.3f} mm) |")
                else:
                    lines.append(f"| {k} | {v} |")
                listed.add(k)
    # Remaining params
    remaining = {k: v for k, v in effective_params.items() if k not in listed}
    if remaining:
        lines.append(f"| **Other** | |")
        for k, v in sorted(remaining.items()):
            lines.append(f"| {k} | {v} |")

    lines.append("")

    if extra_info:
        lines.append("## Additional Info\n")
        for k, v in extra_info.items():
            lines.append(f"- **{k}:** {v}")
        lines.append("")

    # Exact JSON dump for reproducibility
    lines.append("## Exact JSON Parameters\n")
    lines.append("```json")
    lines.append(json.dumps(effective_params, indent=2, default=str))
    lines.append("```\n")

    out.write_text("\n".join(lines))
    return out
