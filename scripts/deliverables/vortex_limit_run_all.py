#!/usr/bin/env python3
"""
Run all parts of the symmetric-vortex limit study.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCRIPTS = [
    PROJECT_ROOT / "scripts" / "deliverables" / "vortex_limit_reference.py",
    PROJECT_ROOT / "scripts" / "deliverables" / "vortex_limit_calibration.py",
    PROJECT_ROOT / "scripts" / "deliverables" / "vortex_limit_analysis.py",
    PROJECT_ROOT / "scripts" / "deliverables" / "vortex_limit_gifs.py",
    PROJECT_ROOT / "scripts" / "deliverables" / "vortex_limit_report.py",
]


def main() -> None:
    for script in SCRIPTS:
        print("=" * 70)
        print(f"Running: {script.relative_to(PROJECT_ROOT)}")
        print("=" * 70)
        res = subprocess.run([sys.executable, str(script)], cwd=str(PROJECT_ROOT))
        if res.returncode != 0:
            raise SystemExit(res.returncode)

    print("=" * 70)
    print("Vortex limit study completed.")
    print("Summary: results/deliverables/vortex_limit/INDEX.md")
    print("=" * 70)


if __name__ == "__main__":
    main()
