#!/usr/bin/env python3
"""
Cleanup RS Investigation Artifacts — Safe Archival
====================================================

Moves superseded / affirmation-only / duplicate diagnostic result
directories into ``results/_archive/`` and writes an archive log.

Patterns targeted (conservative):
  - physics_affirmation_*       — pre-lens sanity checks, superseded
  - pre_lens_affirmation_*      — pre-lens affirmation runs, superseded
  - vortex_audit_*              — early vortex audits, superseded by RS investigation
  - vortex_lens_sweep_*         — early lens sweeps before RS validation
  - vortex_convergence_spotcheck_* — one-off convergence check
  - vortex_balance_*            — early amplitude balance runs
  - vortex_integrity_diag_*     — pre-RS integrity diagnostics
  - fixed_gallery_*             — early gallery outputs
  - resonance_sweep_*           — single resonance-sweep run
  - rs_hybrid_validation_*_144830 — first two incomplete hybrid runs (keep latest)
  - rs_hybrid_validation_*_144934 —   (keep *_145033 as the definitive run)
  - rs_vs_fem_phase1_*          — Phase 1 (no apodization fix), superseded by Phase 1A
  - Stale log files: vortex_audit_run.log, vortex_lens_sweep_log.txt

Directories NOT moved:
  - rs_investigation_*          — current investigation baseline
  - rs_vs_fem_phase1A_*         — current validated Phase 1A truth
  - rs_hybrid_validation_*_145033 — latest definitive hybrid validation
  - vortex_bridge_design_study_* — bridge-design work (may be reused)
  - vortex_static_authority_*   — authority analysis (may be reused)
  - vortex_minimum_mobility_*   — mobility analysis (may be reused)
  - _mobility_fem_cache/        — cache (useful)

This script is **idempotent**: running twice moves nothing the second time.

Usage
-----
  python scripts/dev/cleanup_rs_investigation_artifacts.py [--dry-run]
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
ARCHIVE_DIR = RESULTS_DIR / "_archive"
ARCHIVE_LOG = ARCHIVE_DIR / "ARCHIVE_LOG.md"

# ── Patterns to archive ──────────────────────────────────────────
# (prefix, reason)
ARCHIVE_PATTERNS: list[tuple[str, str]] = [
    ("physics_affirmation_", "Pre-lens affirmation; superseded by RS investigation"),
    ("pre_lens_affirmation_", "Pre-lens affirmation; superseded"),
    ("vortex_audit_", "Early vortex audit; superseded by RS investigation audit"),
    ("vortex_lens_sweep_", "Early lens sweep; superseded by RS validation"),
    ("vortex_convergence_spotcheck_", "One-off convergence check; superseded"),
    ("vortex_balance_", "Early amplitude balance; superseded by hybrid validation"),
    ("vortex_integrity_diag_", "Pre-RS integrity diagnostic; superseded"),
    ("fixed_gallery_", "Early gallery output; superseded"),
    ("resonance_sweep_", "Single resonance sweep run; superseded"),
    ("rs_vs_fem_phase1_2", "Phase 1 (no apodization fix); superseded by Phase 1A"),
]

# Specific directories / files to archive
ARCHIVE_SPECIFIC: list[tuple[str, str]] = [
    ("rs_hybrid_validation_20260302_144830", "Incomplete hybrid run #1; superseded by *_145033"),
    ("rs_hybrid_validation_20260302_144934", "Incomplete hybrid run #2; superseded by *_145033"),
    ("vortex_audit_run.log", "Stale log file"),
    ("vortex_lens_sweep_log.txt", "Stale log file"),
]


def gather_targets() -> list[tuple[Path, str]]:
    """Return (path, reason) for every item to archive."""
    targets = []
    if not RESULTS_DIR.exists():
        return targets

    for entry in sorted(RESULTS_DIR.iterdir()):
        if entry.name.startswith("_"):
            continue  # skip _archive, _mobility_fem_cache

        # Check prefix patterns
        for prefix, reason in ARCHIVE_PATTERNS:
            if entry.name.startswith(prefix):
                targets.append((entry, reason))
                break
        else:
            # Check specific names
            for name, reason in ARCHIVE_SPECIFIC:
                if entry.name == name:
                    targets.append((entry, reason))
                    break

    return targets


def write_log_entry(moved: list[tuple[str, str]], dry_run: bool) -> str:
    """Build a markdown log block."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"\n## Archive run — {ts}{'  (DRY RUN)' if dry_run else ''}\n",
        f"Moved {len(moved)} item(s):\n",
    ]
    for name, reason in moved:
        lines.append(f"- `{name}` — {reason}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Archive old result directories")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be moved without moving")
    args = parser.parse_args()

    targets = gather_targets()
    if not targets:
        print("Nothing to archive — all clean.")
        return

    print(f"{'DRY RUN — ' if args.dry_run else ''}Archive targets ({len(targets)}):")
    for path, reason in targets:
        tag = "DIR " if path.is_dir() else "FILE"
        print(f"  [{tag}] {path.name}  — {reason}")

    if args.dry_run:
        print("\nDry run complete. Pass without --dry-run to execute.")
        return

    # Create archive dir
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    moved = []
    for path, reason in targets:
        dest = ARCHIVE_DIR / path.name
        if dest.exists():
            print(f"  SKIP (already archived): {path.name}")
            continue
        shutil.move(str(path), str(dest))
        moved.append((path.name, reason))
        print(f"  MOVED: {path.name}")

    # Write / append log
    log_entry = write_log_entry(moved, dry_run=False)
    if ARCHIVE_LOG.exists():
        with open(ARCHIVE_LOG, "a") as f:
            f.write(log_entry)
    else:
        header = "# Archive Log\n\nAutomated archival of superseded result directories.\n"
        with open(ARCHIVE_LOG, "w") as f:
            f.write(header + log_entry)

    print(f"\nDone. Moved {len(moved)} items to {ARCHIVE_DIR.relative_to(PROJECT_ROOT)}/")
    print(f"Log: {ARCHIVE_LOG.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
