from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Sequence, Callable
import numpy as np

from .patches import TransducerPatch

@dataclass(frozen=True)
class ActuationState:
    patches: tuple[TransducerPatch, ...]

class Schedule(Protocol):
    def state(self, t: float) -> ActuationState: ...

def smoothstep(u: float) -> float:
    u = float(np.clip(u, 0.0, 1.0))
    return u*u*(3.0 - 2.0*u)

@dataclass(frozen=True)
class Hold:
    s: ActuationState
    def state(self, t: float) -> ActuationState:
        return self.s

@dataclass(frozen=True)
class Ramp:
    a: ActuationState
    b: ActuationState
    T: float
    ease: Callable[[float], float] = smoothstep

    def state(self, t: float) -> ActuationState:
        u = self.ease(0.0 if self.T <= 0 else t / self.T)
        # interpolate patch-by-patch (assumes same ordering / same walls/lengths)
        pa = self.a.patches
        pb = self.b.patches
        if len(pa) != len(pb):
            raise ValueError("Ramp requires same number of patches in a and b.")
        out = []
        for A, B in zip(pa, pb):
            out.append(
                TransducerPatch(
                    wall=A.wall,
                    center=(1-u)*A.center + u*B.center,
                    length=(1-u)*A.length + u*B.length,
                    amp=(1-u)*A.amp + u*B.amp,
                    phase=(1-u)*A.phase + u*B.phase,
                    window=A.window,
                )
            )
        return ActuationState(tuple(out))

@dataclass(frozen=True)
class SequenceSchedule:
    parts: Sequence[tuple[float, Schedule]]  # list of (duration, schedule)

    def state(self, t: float) -> ActuationState:
        tt = float(t)
        for dur, sch in self.parts:
            if tt <= dur:
                return sch.state(tt)
            tt -= dur
        # if beyond end, hold last
        return self.parts[-1][1].state(self.parts[-1][0])
