#!/usr/bin/env python3
"""
Cleanup old results while preserving recent and important runs.

Extended Policy (2026-02-08):
- KEEP: Results from last 5 days
- KEEP: Directories matching canonical run patterns (configurable)
- KEEP: path_tracking_comparison/*, square_dish_phase1/*, phase2_*
- ARCHIVE: Older exploratory runs to ARCHIVE_OLD/_auto_archive_YYYYMMDD/
- CURATE: Keep only designated "gold runs" at top level

Canonical Run Types (for shallow dish workflow):
- device-aligned baseline
- standing-only
- vortex-only
- combined
- streaming-only
- particle-trajectory demo

Usage:
  python scripts/tools/cleanup_results.py --dry-run                    # Preview
  python scripts/tools/cleanup_results.py                              # Execute
  python scripts/tools/cleanup_results.py --auto-archive               # Archive with timestamp
  python scripts/tools/cleanup_results.py --curate device_shallow_*    # Keep only matching
"""

import argparse
from pathlib import Path
from datetime import datetime, timedelta
import shutil
import os
import re
import json

def parse_date_from_dirname(dirname: str) -> datetime | None:
    """Extract date from run_YYYYMMDD_HHMMSS format"""
    if dirname.startswith("run_"):
        try:
            date_str = dirname.split("_")[1]  # YYYYMMDD
            return datetime.strptime(date_str, "%Y%m%d")
        except (IndexError, ValueError):
            return None
    return None

def get_dir_size_mb(path: Path) -> float:
    """Get directory size in MB"""
    total = 0
    for entry in path.rglob("*"):
        if entry.is_file():
            total += entry.stat().st_size
    return total / (1024 * 1024)

def main():
    parser = argparse.ArgumentParser(description="Cleanup old results")
    parser.add_argument("--dry-run", action="store_true", help="Preview without making changes")
    args = parser.parse_args()
    
    repo_root = Path(__file__).parent.parent.parent
    results_dir = repo_root / "results"
    archive_dir = results_dir / "ARCHIVE_OLD"
    keep_dir = results_dir / "KEEP"
    
    # Cutoff: keep results from Feb 2, 2026 onwards
    cutoff_date = datetime(2026, 2, 2)
    
    # Patterns to always keep
    keep_patterns = [
        "path_tracking_comparison",
        "square_dish_phase1",
        "phase2_",
        "logs",
        "ARCHIVE_OLD",
        "KEEP",
    ]
    
    # Create archive directory
    if not args.dry_run and not archive_dir.exists():
        archive_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created: {archive_dir}")
    
    print("\n" + "="*70)
    print(f"RESULTS CLEANUP - {'DRY RUN' if args.dry_run else 'EXECUTING'}")
    print("="*70)
    print(f"Cutoff date: {cutoff_date.strftime('%Y-%m-%d')}")
    print(f"Keep patterns: {', '.join(keep_patterns)}\n")
    
    # Scan results directory
    kept = []
    archived = []
    
    for item in results_dir.iterdir():
        if not item.is_dir():
            continue
        
        # Check if in keep patterns
        if any(pattern in item.name for pattern in keep_patterns):
            kept.append(item.name)
            continue
        
        # Check date-based runs
        if item.name.startswith("run_"):
            run_date = parse_date_from_dirname(item.name)
            if run_date and run_date >= cutoff_date:
                kept.append(item.name)
                continue
        
        # Check for dated subdirectories
        has_recent = False
        for subdir in item.iterdir():
            if subdir.is_dir():
                run_date = parse_date_from_dirname(subdir.name)
                if run_date and run_date >= cutoff_date:
                    has_recent = True
                    break
        
        if has_recent:
            kept.append(item.name)
            continue
        
        # Archive old results
        size_mb = get_dir_size_mb(item)
        archived.append((item.name, size_mb))
        
        if not args.dry_run:
            dest = archive_dir / item.name
            print(f"ARCHIVING: {item.name} ({size_mb:.1f} MB) -> {dest}")
            shutil.move(str(item), str(dest))
        else:
            print(f"WOULD ARCHIVE: {item.name} ({size_mb:.1f} MB)")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"KEPT ({len(kept)} directories):")
    for name in sorted(kept):
        print(f"  ✓ {name}")
    
    print(f"\nARCHIVED ({len(archived)} directories):")
    total_archived_mb = 0
    for name, size in sorted(archived):
        total_archived_mb += size
        print(f"  → {name} ({size:.1f} MB)")
    
    print(f"\nTotal archived: {total_archived_mb:.1f} MB")
    
    if args.dry_run:
        print("\n⚠️  DRY RUN - No changes made")
        print("Run without --dry-run to execute")
    else:
        print("\n✅ Cleanup complete")


def auto_archive_exploratory(results_dir: Path, dry_run: bool = True, 
                             keep_patterns: list = None) -> dict:
    """
    Move old exploratory runs to ARCHIVE_OLD/_auto_archive_YYYYMMDD/
    
    Returns dict with archived/kept counts.
    """
    timestamp = datetime.now().strftime("%Y%m%d")
    archive_dest = results_dir / "ARCHIVE_OLD" / f"_auto_archive_{timestamp}"
    
    if keep_patterns is None:
        # Default canonical patterns to keep at top level
        keep_patterns = [
            r"device_shallow_\d{8}",  # Device demo runs
            r"canonical_.*",           # Explicitly canonical
            r"gold_.*",                # Gold runs
            r"path_tracking_comparison",
            r"square_dish_phase1",
            r"phase2_.*",
            r"logs",
            r"ARCHIVE_OLD",
            r"validation",
        ]
    
    # Compile patterns
    keep_re = [re.compile(p) for p in keep_patterns]
    
    archived = []
    kept = []
    
    for item in results_dir.iterdir():
        if not item.is_dir():
            continue
        
        # Check if matches any keep pattern
        should_keep = any(r.match(item.name) for r in keep_re)
        
        if should_keep:
            kept.append(item.name)
        else:
            archived.append(item.name)
            if not dry_run:
                archive_dest.mkdir(parents=True, exist_ok=True)
                dest = archive_dest / item.name
                shutil.move(str(item), str(dest))
    
    return {
        "archived": archived,
        "kept": kept,
        "archive_dest": str(archive_dest) if not dry_run else None,
    }


def generate_index(results_dir: Path, output_path: Path = None) -> str:
    """
    Generate INDEX.md documenting current results structure.
    """
    if output_path is None:
        output_path = results_dir / "INDEX.md"
    
    lines = [
        "# Results Index",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Canonical Runs (Gold)",
        "",
        "These are the curated, production-quality runs for the shallow dish workflow:",
        "",
        "| Run Directory | Description | Key Outputs |",
        "|--------------|-------------|-------------|",
    ]
    
    # Look for canonical patterns
    canonical = []
    for item in sorted(results_dir.iterdir()):
        if not item.is_dir():
            continue
        if item.name.startswith("device_shallow_"):
            # Check for config.json
            config_file = item / "meta" / "config.json"
            desc = "Device demo run"
            outputs = []
            
            if config_file.exists():
                try:
                    with open(config_file) as f:
                        cfg = json.load(f)
                    desc = cfg.get("description", desc)
                except:
                    pass
            
            # Check what outputs exist
            for vtu in item.glob("*.vtu"):
                outputs.append(vtu.stem)
            
            output_str = ", ".join(outputs[:4])
            if len(outputs) > 4:
                output_str += f" (+{len(outputs)-4} more)"
            
            lines.append(f"| `{item.name}` | {desc} | {output_str or 'N/A'} |")
            canonical.append(item.name)
    
    if not canonical:
        lines.append("| *(none yet)* | Run `run_device_demo.py` to generate | |")
    
    lines.extend([
        "",
        "## Other Recent Runs",
        "",
    ])
    
    # List other runs
    other = []
    for item in sorted(results_dir.iterdir()):
        if not item.is_dir():
            continue
        if item.name in canonical:
            continue
        if item.name in ["ARCHIVE_OLD", "logs", "validation"]:
            continue
        
        size_mb = sum(f.stat().st_size for f in item.rglob("*") if f.is_file()) / 1e6
        other.append((item.name, size_mb))
    
    for name, size in other[:20]:
        lines.append(f"- `{name}` ({size:.1f} MB)")
    
    if len(other) > 20:
        lines.append(f"- ... and {len(other) - 20} more")
    
    lines.extend([
        "",
        "## Archive",
        "",
        "Older exploratory runs are moved to `ARCHIVE_OLD/`.",
        "",
        "---",
        "*Use `scripts/tools/cleanup_results.py` to manage results.*",
    ])
    
    content = "\n".join(lines)
    
    with open(output_path, "w") as f:
        f.write(content)
    
    return content


if __name__ == "__main__":
    main()
