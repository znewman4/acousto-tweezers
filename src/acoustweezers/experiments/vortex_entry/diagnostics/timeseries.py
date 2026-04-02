"""
Time-series recorder for vortex entry diagnostics.

Accumulates per-sub-step particle state and computes derived
quantities (separation, barrier distance, B status).
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


class TimeSeriesRecorder:
    """Lightweight accumulator for per-sub-step particle diagnostics."""

    COLUMNS = [
        "t_step", "v_step", "dyn_i", "phase_label",
        "psi", "alpha",
        "A_x_m", "A_y_m", "B_x_m", "B_y_m",
        "vc_x_m", "vc_y_m",
        "d_AB_m", "d_A_vc_m", "d_B_vc_m",
        "r_barrier_m", "B_status",
    ]

    def __init__(self) -> None:
        self._rows: List[List[Any]] = []
        self._t: int = 0

    def record(
        self,
        v_step: int,
        dyn_i: int,
        phase_label: str,
        psi: float,
        alpha: float,
        pos_A: np.ndarray,
        pos_B: np.ndarray,
        vortex_center: np.ndarray,
        r_barrier: float,
    ) -> None:
        d_AB = float(np.linalg.norm(pos_A - pos_B))
        d_A_vc = float(np.linalg.norm(pos_A - vortex_center))
        d_B_vc = float(np.linalg.norm(pos_B - vortex_center))

        if d_B_vc > r_barrier * 1.05:
            b_stat = "OUTSIDE"
        elif d_B_vc < r_barrier * 0.95:
            b_stat = "INSIDE"
        else:
            b_stat = "ON"

        self._rows.append([
            self._t, v_step, dyn_i, phase_label,
            psi, alpha,
            pos_A[0], pos_A[1], pos_B[0], pos_B[1],
            vortex_center[0], vortex_center[1],
            d_AB, d_A_vc, d_B_vc,
            r_barrier, b_stat,
        ])
        self._t += 1

    @property
    def rows(self) -> List[List[Any]]:
        return self._rows

    def get_column(self, name: str) -> np.ndarray:
        """Extract a named column as a numpy array."""
        idx = self.COLUMNS.index(name)
        return np.array([r[idx] for r in self._rows])

    def to_csv(self, path: Path) -> None:
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(self.COLUMNS)
            w.writerows(self._rows)
