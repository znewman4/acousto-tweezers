#!/usr/bin/env python3
"""Run the canonical full-domain translated replica C-shape transport method."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRESET_JSON = PROJECT_ROOT / "configs" / "cases" / "replica_cshape_fullfield_transport_standard.json"
STUDY_SCRIPT = PROJECT_ROOT / "scripts" / "deliverables" / "transport_side_by_side_replica_cshape.py"


def _to_env_str(value) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the canonical full-domain translated replica C-shape transport method.",
    )
    parser.add_argument(
        "--preset-json",
        type=Path,
        default=PRESET_JSON,
        help="Path to the canonical preset JSON.",
    )
    parser.add_argument(
        "--out-gif-name",
        type=str,
        default="",
        help="Optional override for OUT_GIF_NAME.",
    )
    parser.add_argument(
        "--display-subframes",
        type=int,
        default=None,
        help="Optional override for DISPLAY_SUBFRAMES.",
    )
    parser.add_argument(
        "--max-output-frames",
        type=int,
        default=None,
        help="Optional frame cap for debugging.",
    )
    parser.add_argument(
        "--gif-duration-ms",
        type=int,
        default=None,
        help="Optional override for GIF_DURATION_MS.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved environment without executing study script.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    with open(args.preset_json, "r", encoding="utf-8") as f:
        preset = json.load(f)

    env_cfg = dict(preset.get("env", {}))
    if not env_cfg:
        raise SystemExit(f"No env block found in preset: {args.preset_json}")

    if args.out_gif_name:
        env_cfg["OUT_GIF_NAME"] = args.out_gif_name
    if args.display_subframes is not None:
        env_cfg["DISPLAY_SUBFRAMES"] = int(args.display_subframes)
    if args.max_output_frames is not None:
        env_cfg["MAX_OUTPUT_FRAMES"] = int(args.max_output_frames)
    if args.gif_duration_ms is not None:
        env_cfg["GIF_DURATION_MS"] = int(args.gif_duration_ms)

    run_env = os.environ.copy()
    run_env.update({k: _to_env_str(v) for k, v in env_cfg.items()})

    print("=" * 80)
    print("Canonical replica transport run")
    print(f"Preset: {args.preset_json.relative_to(PROJECT_ROOT)}")
    print(f"Method: {preset.get('_method_id', 'unknown')}")
    print(f"Script: {STUDY_SCRIPT.relative_to(PROJECT_ROOT)}")
    print("=" * 80)

    if args.dry_run:
        print("Dry-run mode. Resolved environment:")
        for key in sorted(env_cfg):
            print(f"  {key}={env_cfg[key]}")
        return

    cmd = [sys.executable, str(STUDY_SCRIPT)]
    res = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=run_env)
    if res.returncode != 0:
        raise SystemExit(res.returncode)

    print("=" * 80)
    print("Canonical replica transport run completed.")
    for p in preset.get("expected_outputs", []):
        print(f"  - {p}")
    print("=" * 80)


if __name__ == "__main__":
    main()
