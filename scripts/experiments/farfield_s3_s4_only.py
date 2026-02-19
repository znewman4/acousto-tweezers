#!/usr/bin/env python3
"""Run only S3 and S4 diagnostics (S1/S2 already confirmed PASS)."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

# Reuse from the main diagnostics script
from farfield_part1_diagnostics import make_cfg, diagnostic_s3, diagnostic_s4

def main():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("results") / f"farfield_s3s4_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#' * 70}")
    print(f"  S3+S4 DIAGNOSTICS (S1,S2 already PASS)")
    print(f"  Output: {out_dir}")
    print(f"{'#' * 70}")

    cfg = make_cfg()
    diagnostic_s3(cfg, out_dir)
    diagnostic_s4(cfg, out_dir)

    print(f"\n{'#' * 70}")
    print(f"  S3+S4 DIAGNOSTICS COMPLETE -- output: {out_dir}")
    print(f"{'#' * 70}\n")

if __name__ == "__main__":
    main()
