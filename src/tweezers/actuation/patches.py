#src/tweezers/actuation/patches.py
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import numpy as np
from typing import Callable, Iterable

from acousto.solvers.fd_helmholtz_2d import BoundarySpec

class Wall(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    BOTTOM = "bottom"
    TOP = "top"

@dataclass(frozen=True)
class TransducerPatch:
    wall: Wall
    center: float     # along-wall coordinate (y for L/R, x for B/T) [m]
    length: float     # patch length along wall [m]
    amp: float        # normal velocity amplitude [m/s]
    phase: float      # radians
    window: str = "tophat"  # "tophat" or "hann" (optional)

def _window_mask(s: np.ndarray, s0: float, L: float, window: str) -> np.ndarray:
    half = 0.5 * L
    a = s0 - half
    b = s0 + half
    if window == "tophat":
        return ((s >= a) & (s <= b)).astype(float)
    if window == "hann":
        # smooth edges: 0..1..0 over [a,b]
        w = np.zeros_like(s, dtype=float)
        m = (s >= a) & (s <= b)
        if np.any(m):
            xi = (s[m] - a) / (L + 1e-30)  # 0..1
            w[m] = 0.5 - 0.5 * np.cos(2.0 * np.pi * xi)
        return w
    raise ValueError(f"Unknown window: {window}")

def _vn_patch(s: np.ndarray, patch: TransducerPatch) -> np.ndarray:
    mask = _window_mask(s, patch.center, patch.length, patch.window)
    return (patch.amp * mask) * np.exp(1j * patch.phase)

def vn_from_patches(
    patches: Iterable[TransducerPatch],
    *,
    Lx: float,
    Ly: float,
) -> dict[str, BoundarySpec]:
    """
    Return boundary specs vn_left/vn_right/vn_bottom/vn_top as callables.
    They accept coordinate arrays (y for left/right, x for bottom/top).
    """
    patches = tuple(patches)

    left_p  = [p for p in patches if p.wall == Wall.LEFT]
    right_p = [p for p in patches if p.wall == Wall.RIGHT]
    bot_p   = [p for p in patches if p.wall == Wall.BOTTOM]
    top_p   = [p for p in patches if p.wall == Wall.TOP]

    def sum_patches_on(coord: np.ndarray, ps: list[TransducerPatch], limit: float) -> np.ndarray:
        # (Optional) clip centers to keep patches inside the wall; safer for interactive work.
        out = np.zeros_like(coord, dtype=np.complex128)
        for p in ps:
            # if patch center is out of bounds, you’ll just get mostly zero; but clipping is nicer
            pc = float(np.clip(p.center, 0.0, limit))
            pp = TransducerPatch(p.wall, pc, p.length, p.amp, p.phase, p.window)
            out += _vn_patch(coord, pp)
        return out

    vn_left   = (lambda y: sum_patches_on(np.asarray(y), left_p,  Ly)) if left_p  else 0.0 + 0.0j
    vn_right  = (lambda y: sum_patches_on(np.asarray(y), right_p, Ly)) if right_p else 0.0 + 0.0j
    vn_bottom = (lambda x: sum_patches_on(np.asarray(x), bot_p,   Lx)) if bot_p   else 0.0 + 0.0j
    vn_top    = (lambda x: sum_patches_on(np.asarray(x), top_p,   Lx)) if top_p   else 0.0 + 0.0j

    return dict(vn_left=vn_left, vn_right=vn_right, vn_bottom=vn_bottom, vn_top=vn_top)
