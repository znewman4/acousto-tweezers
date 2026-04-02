"""
Particle ensemble recorder for inertial particle diagnostics.

Writes particles.csv with columns:
    t_snap, particle_id, x_m, y_m, vx_mps, vy_mps

Records are taken at snapshot times (GIF frame cadence) rather than every
dynamics sub-step, keeping file size proportional to n_gif_frames × N.

Schema is completely separate from TimeSeriesRecorder — timeseries.csv
and proximity_vs_time.png are unaffected.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, List

import numpy as np

from ..particles.particle_state import ParticleEnsemble


class ParticleRecorder:
    """
    Accumulates per-snapshot particle state for the inertial ensemble.

    Usage
    -----
    recorder = ParticleRecorder()

    # At each GIF snapshot:
    recorder.record_ensemble(state)

    # After the run:
    recorder.to_csv(out_dir / "particles.csv")
    """

    COLUMNS: List[str] = [
        "t_snap", "particle_id",
        "x_m", "y_m",
        "vx_mps", "vy_mps",
    ]

    def __init__(self) -> None:
        self._rows: List[List[Any]] = []
        self._t_snap: int = 0

    def record_ensemble(self, state: ParticleEnsemble) -> None:
        """
        Record all N particles for the current snapshot.

        Increments the internal snapshot counter after each call.
        """
        t = self._t_snap
        for i in range(state.N):
            self._rows.append([
                t, i,
                float(state.pos[i, 0]), float(state.pos[i, 1]),
                float(state.vel[i, 0]), float(state.vel[i, 1]),
            ])
        self._t_snap += 1

    @property
    def rows(self) -> List[List[Any]]:
        return self._rows

    @property
    def n_snapshots(self) -> int:
        return self._t_snap

    def to_csv(self, path: Path) -> None:
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(self.COLUMNS)
            w.writerows(self._rows)
