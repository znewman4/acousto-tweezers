#!/usr/bin/env python3
"""Run all parts of the translated-vortex stage transport study."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCRIPTS = [
    PROJECT_ROOT / "scripts" / "deliverables" / "vortex_stage_transport_reference.py",
    PROJECT_ROOT / "scripts" / "deliverables" / "vortex_stage_transport_calibration.py",
    PROJECT_ROOT / "scripts" / "deliverables" / "vortex_stage_transport_transport.py",
    PROJECT_ROOT / "scripts" / "deliverables" / "vortex_stage_transport_gifs.py",
    PROJECT_ROOT / "scripts" / "deliverables" / "vortex_stage_transport_report.py",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full translated-vortex transport pipeline.")
    parser.add_argument(
        "--gif-preset",
        choices=["standard", "dense"],
        default="standard",
        help="Named GIF render preset forwarded to vortex_stage_transport_gifs.py.",
    )
    parser.add_argument("--gif-frame-dt", type=float, default=None, help="Override GIF frame duration in seconds.")
    parser.add_argument("--gif-push-interp", type=int, default=None, help="Override transport subframes per step.")
    parser.add_argument("--gif-release-interp", type=int, default=None, help="Override release subframes per step.")
    parser.add_argument("--gif-smooth-window", type=int, default=None, help="Override display smoothing window.")
    return parser.parse_args()


def _gif_args(args: argparse.Namespace) -> list[str]:
    out = ["--preset", str(args.gif_preset)]
    if args.gif_frame_dt is not None:
        out.extend(["--frame-dt", str(args.gif_frame_dt)])
    if args.gif_push_interp is not None:
        out.extend(["--push-interp", str(args.gif_push_interp)])
    if args.gif_release_interp is not None:
        out.extend(["--release-interp", str(args.gif_release_interp)])
    if args.gif_smooth_window is not None:
        out.extend(["--smooth-window", str(args.gif_smooth_window)])
    return out


def main() -> None:
    args = _parse_args()
    gif_args = _gif_args(args)

    for script in SCRIPTS:
        print("=" * 72)
        print(f"Running: {script.relative_to(PROJECT_ROOT)}")
        print("=" * 72)
        cmd = [sys.executable, str(script)]
        if script.name == "vortex_stage_transport_gifs.py":
            cmd.extend(gif_args)
        res = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        if res.returncode != 0:
            raise SystemExit(res.returncode)

    print("=" * 72)
    print("Translated-vortex stage transport study completed.")
    print("Summary: results/deliverables/vortex_stage_transport/report/INDEX.md")
    print("=" * 72)


if __name__ == "__main__":
    main()
