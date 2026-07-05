"""
season_schedule.py — irregular long-season drift schedule for the Paper 3D
MongoDB experiment (Addendum 16 / 21).

WHY: the existing arm uses a REGULAR 4-phase x 6-window schedule, which is the
near-WORST case for similarity-gating (short stable runs, drift catchable by a
fixed period). To let gating win HONESTLY we need LONG, IRREGULAR,
HIGH-MAGNITUDE seasons:
  - each season = one modality phase (edge|geo|text|review) needing its own index
  - season lengths variable (irregular) in [min_len, max_len]
  - consecutive seasons are DIFFERENT phases (high magnitude: prior index stale)

Under this structure gating beats BOTH always_on (invocation frugality over long
stable runs) and periodic (timing on irregular boundaries). A 'regular' mode is
provided for the sensitivity contrast (regular -> gating≈periodic; irregular ->
gating>periodic).

Deterministic given `seed` (paired-RCB compatible). Pure Python, no mongod I/O.
"""
from __future__ import annotations
import random
from dataclasses import dataclass

DEFAULT_PHASES = ["edge", "geo", "text", "review"]


@dataclass
class Season:
    phase: str
    start: int          # 0-based window index (inclusive)
    length: int

    @property
    def end(self) -> int:        # exclusive
        return self.start + self.length


def make_schedule(n_windows: int,
                  seed: int,
                  phases: list[str] | None = None,
                  mode: str = "irregular",
                  min_len: int = 12,
                  max_len: int = 36,
                  period: int = 6):
    """Build a drift schedule.

    Returns (phase_per_window, seasons, drift_windows):
      phase_per_window : list[str]      length == n_windows
      seasons          : list[Season]
      drift_windows    : set[int]       0-based windows where a new season starts
                                        (i.e. phase != previous window's phase)

    mode='irregular' (default): variable season lengths in [min_len, max_len];
        next phase chosen at random with NO immediate repeat (guarantees a real
        modality switch at every boundary -> high magnitude).
    mode='regular': fixed-length seasons of `period`, phases cycled in order
        (reproduces the legacy 4xperiod schedule for the sensitivity axis).
    """
    phases = list(phases or DEFAULT_PHASES)
    rng = random.Random(seed)
    phase_per_window: list[str] = []
    seasons: list[Season] = []

    if mode == "regular":
        k = 0
        while len(phase_per_window) < n_windows:
            ph = phases[k % len(phases)]
            ln = min(period, n_windows - len(phase_per_window))
            seasons.append(Season(ph, len(phase_per_window), ln))
            phase_per_window += [ph] * ln
            k += 1
    elif mode == "irregular":
        prev = None
        while len(phase_per_window) < n_windows:
            choices = [p for p in phases if p != prev] or phases
            ph = rng.choice(choices)
            ln = rng.randint(min_len, max_len)
            ln = min(ln, n_windows - len(phase_per_window))
            seasons.append(Season(ph, len(phase_per_window), ln))
            phase_per_window += [ph] * ln
            prev = ph
    else:
        raise ValueError(f"unknown mode: {mode!r}")

    drift_windows = {s.start for s in seasons if s.start > 0}
    return phase_per_window, seasons, drift_windows


if __name__ == "__main__":
    for mode in ("irregular", "regular"):
        pw, ss, dw = make_schedule(120, seed=7100, mode=mode)
        print(f"\n=== mode={mode}  n_windows={len(pw)}  seasons={len(ss)}  "
              f"drifts={len(dw)} ===")
        for s in ss:
            print(f"  win[{s.start:3d}..{s.end-1:3d}] len={s.length:2d}  {s.phase}")
