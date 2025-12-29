# scripts/organise_results.py
from __future__ import annotations

from pathlib import Path
import shutil


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def move_matches(results: Path, pattern: str, dest: Path) -> int:
    ensure_dir(dest)
    n = 0
    for f in results.glob(pattern):
        if f.is_file():
            target = dest / f.name
            # Avoid overwriting: if exists, rename with suffix
            if target.exists():
                stem = target.stem
                suffix = target.suffix
                k = 1
                while True:
                    alt = dest / f"{stem}__{k}{suffix}"
                    if not alt.exists():
                        target = alt
                        break
                    k += 1
            shutil.move(str(f), str(target))
            n += 1
    return n


def main() -> None:
    results = Path("results")
    if not results.exists():
        print("No results/ folder found.")
        return

    mapping = [
        ("validate_*.png", results / "validate"),
        ("phase_sweep_*.png", results / "phase_sweep"),
        ("streamlines_*.png", results / "basin"),
        ("basin_*.png", results / "basin"),
        ("phase_sweep_snapshot_*.png", results / "snapshots"),
        ("helmholtz_2d_*.png", results / "fields"),
        ("gorkov_2d_*.png", results / "gorkov"),
        ("trajectory_*.png", results / "dynamics"),
    ]

    total = 0
    for pattern, dest in mapping:
        moved = move_matches(results, pattern, dest)
        if moved:
            print(f"Moved {moved:3d} files matching {pattern} -> {dest}")
        total += moved

    print(f"\nDone. Total moved: {total}")
    print("Tip: if you have older mixed plots you want to keep, put them in results/_archive/ manually.")


if __name__ == "__main__":
    main()
