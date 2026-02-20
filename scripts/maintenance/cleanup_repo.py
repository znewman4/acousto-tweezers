#!/usr/bin/env python3
"""
Repository Cleanup Script
=========================

Moves old results to results/_archive/YYYYMMDD/, removes empty files/dirs
and failed partial runs, and writes an ARCHIVE_LOG.md.

Usage:
    python scripts/maintenance/cleanup_repo.py [--dry-run]
"""
from __future__ import annotations

import argparse
import os
import shutil
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"

# Folders / symlinks that should NEVER be archived
KEEP_NAMES = {
    "_archive",
}

# Symlink suffixes that are canonical pointers — keep them
KEEP_SYMLINK_SUFFIXES = {"_latest"}


def _is_keep(name: str) -> bool:
    """Return True if the item should be kept in results/."""
    if name in KEEP_NAMES:
        return True
    if name.startswith("."):
        return True
    # Keep *_latest symlinks
    for suffix in KEEP_SYMLINK_SUFFIXES:
        if name.endswith(suffix):
            return True
    return False


def _is_failed_run(path: Path) -> bool:
    """Heuristic: a directory with no files or only empty files is a failed run."""
    if not path.is_dir():
        return False
    files = list(path.rglob("*"))
    real_files = [f for f in files if f.is_file() and f.stat().st_size > 0]
    return len(real_files) == 0


def _remove_empty(root: Path, dry_run: bool) -> list[str]:
    """Remove empty files and empty directories under root."""
    removed = []
    # Empty files
    for f in root.rglob("*"):
        if f.is_file() and f.stat().st_size == 0:
            removed.append(f"empty file: {f.relative_to(REPO_ROOT)}")
            if not dry_run:
                f.unlink()
    # Empty directories (bottom-up)
    for d in sorted(root.rglob("*"), key=lambda p: -len(p.parts)):
        if d.is_dir() and not any(d.iterdir()):
            removed.append(f"empty dir:  {d.relative_to(REPO_ROOT)}")
            if not dry_run:
                d.rmdir()
    return removed


def main():
    parser = argparse.ArgumentParser(description="Clean up results/ directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be done without doing it")
    args = parser.parse_args()
    dry_run = args.dry_run

    if not RESULTS_DIR.exists():
        print("results/ directory not found — nothing to clean.")
        return

    stamp = datetime.now().strftime("%Y%m%d")
    archive_dir = RESULTS_DIR / "_archive" / stamp
    if not dry_run:
        archive_dir.mkdir(parents=True, exist_ok=True)

    moved = []
    failed_removed = []
    empty_removed = []

    # ── 1. Identify items to archive ──────────────────────────────────
    for item in sorted(RESULTS_DIR.iterdir()):
        name = item.name
        if _is_keep(name):
            continue

        # Failed partial runs → delete
        if _is_failed_run(item):
            failed_removed.append(str(item.relative_to(REPO_ROOT)))
            if not dry_run:
                shutil.rmtree(item)
            continue

        # Symlinks that are NOT *_latest → remove
        if item.is_symlink():
            # dangling or non-latest symlinks
            if not any(name.endswith(s) for s in KEEP_SYMLINK_SUFFIXES):
                failed_removed.append(f"symlink: {item.relative_to(REPO_ROOT)}")
                if not dry_run:
                    item.unlink()
            continue

        # Everything else → archive
        if item.is_dir():
            dest = archive_dir / name
            moved.append(f"{item.relative_to(REPO_ROOT)} → _archive/{stamp}/{name}")
            if not dry_run:
                shutil.move(str(item), str(dest))

    # ── 2. Remove empty files/dirs ────────────────────────────────────
    empty_removed = _remove_empty(RESULTS_DIR, dry_run)

    # ── 3. Fix dangling *_latest symlinks ─────────────────────────────
    for item in RESULTS_DIR.iterdir():
        if item.is_symlink() and not item.resolve().exists():
            empty_removed.append(f"dangling symlink: {item.name}")
            if not dry_run:
                item.unlink()

    # ── 4. Write archive log ──────────────────────────────────────────
    log_path = RESULTS_DIR / "_archive" / "ARCHIVE_LOG.md"
    log_lines = []
    if log_path.exists():
        log_lines = log_path.read_text().splitlines()

    log_lines.append(f"\n## Cleanup {datetime.now().isoformat()}")
    if dry_run:
        log_lines.append("**(DRY RUN — nothing was actually changed)**\n")
    log_lines.append(f"\n### Archived to `_archive/{stamp}/`\n")
    for m in moved:
        log_lines.append(f"- {m}")
    if not moved:
        log_lines.append("- (none)")
    log_lines.append(f"\n### Failed/empty runs removed\n")
    for f in failed_removed:
        log_lines.append(f"- {f}")
    if not failed_removed:
        log_lines.append("- (none)")
    log_lines.append(f"\n### Empty files/dirs removed\n")
    for e in empty_removed:
        log_lines.append(f"- {e}")
    if not empty_removed:
        log_lines.append("- (none)")
    log_lines.append("")

    if not dry_run:
        archive_dir.mkdir(parents=True, exist_ok=True)
        log_path.write_text("\n".join(log_lines))

    # ── Summary ───────────────────────────────────────────────────────
    prefix = "[DRY RUN] " if dry_run else ""
    print(f"\n{prefix}Cleanup complete:")
    print(f"  Archived:        {len(moved)} directories → _archive/{stamp}/")
    print(f"  Failed removed:  {len(failed_removed)}")
    print(f"  Empty removed:   {len(empty_removed)}")
    if not dry_run:
        print(f"  Log: {log_path.relative_to(REPO_ROOT)}")
    print()


if __name__ == "__main__":
    main()
