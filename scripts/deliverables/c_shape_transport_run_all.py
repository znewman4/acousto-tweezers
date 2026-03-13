#!/usr/bin/env python3
"""Run the reproducible C-shape transport refinement study."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "deliverables" / "c_shape_transport"
STUDY = PROJECT_ROOT / "scripts" / "dev" / "c_shape_transport_refinement_study.py"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the reproducible C-shape transport workflow.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Stable output directory for reproducible reruns.",
    )
    parser.add_argument(
        "--n-frames",
        type=int,
        default=280,
        help="Stored/rendered frames for the best and baseline GIFs.",
    )
    parser.add_argument(
        "--gif-duration-ms",
        type=int,
        default=55,
        help="GIF frame duration in milliseconds.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cmd = [
        sys.executable,
        str(STUDY),
        "--output-dir",
        str(Path(args.output_dir)),
        "--n-frames",
        str(int(args.n_frames)),
        "--gif-duration-ms",
        str(int(args.gif_duration_ms)),
    ]

    print("=" * 72)
    print(f"Running: {STUDY.relative_to(PROJECT_ROOT)}")
    print(f"Output:  {Path(args.output_dir).relative_to(PROJECT_ROOT)}")
    print("=" * 72)
    res = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if res.returncode != 0:
        raise SystemExit(res.returncode)

    print("=" * 72)
    print("C-shape transport study completed.")
    print(f"Summary: {Path(args.output_dir).relative_to(PROJECT_ROOT) / 'INDEX.md'}")
    print("=" * 72)


if __name__ == "__main__":
    main()