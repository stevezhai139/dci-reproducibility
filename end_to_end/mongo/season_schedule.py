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



def make_mixed_schedule(n_windows: int = 24,
                        win_per_ph: int = 4,
                        phases: list[str] | None = None,
                        base_count: int = 20,
                        hi_count: int = 48):
    """Paper 3C — S5 Part 2 (2026-07-07): live MIXED-drift schedule.

    The legacy 4x6 schedule moves ONLY the modality (edge->geo->text->review),
    so every boundary is a template-structure move and a block is a
    single-cause cell. This builder reproduces the OFFLINE mixed cells of
    Sec 6.2 (cost_benefit.build_trajectory) live: successive phase boundaries
    ALTERNATE the drift type — template (modality switch, count held) /
    volume (count toggled base<->hi, modality held). hi_count=48 was
    chosen by offline pre-validation: modality swaps are violent on S_T/S_A,
    so the S_V toggle needs dip 1-20/48=0.583 to balance the deviation
    spectrum into the Sec 6.2 mixed band (20<->40 left the pooled DCI ~1.57).

    Defaults (n_windows=24, win_per_ph=4, phases=edge/geo/review) give six
    phases with onsets at 1-based windows 5, 9, 13, 17, 21:

        win  1- 4 : edge   @ 20   (steady phase = the calibration distribution)
        win  5- 8 : geo    @ 20   <- TEMPLATE (modality)
        win  9-12 : geo    @ 48   <- VOLUME   (pure S_V move, mix held)
        win 13-16 : review @ 48   <- TEMPLATE (modality)
        win 17-20 : review @ 20   <- VOLUME
        win 21-24 : edge   @ 20   <- TEMPLATE (modality; phases cycle)

    Design notes (offline pre-validation, part2_validate_offline.py):
      * 3 template + 2 volume onsets balance the deviation spectrum so the
        pooled onset DCI lands in the Sec 6.2 mixed band (~1.7-2.0) — a
        single volume onset against violent modality swaps stays ~1.3.
      * "text" is deliberately absent: $text REQUIRES a text index, so a
        missed text onset produces fast structural FAILURES, contaminating
        the query-latency-fidelity metric with error latencies. edge, geo
        and review degrade gracefully without their advisor index
        ($geoWithin needs no 2dsphere), keeping the latency leg clean.
      * Deterministic (no RNG) -> identical across blocks/strategies/tau
        configs; RCB pairing preserved.

    Returns (phase_per_window, count_per_window, boundary_types):
      phase_per_window : list[str]  length n_windows
      count_per_window : list[int]  length n_windows (ops per window)
      boundary_types   : dict {0-based onset window index: "template"|"volume"}
    """
    phases = list(phases or ["edge", "geo", "review"])
    n_phases = max(1, n_windows // win_per_ph)
    phase_per_window: list[str] = []
    count_per_window: list[int] = []
    boundary_types: dict[int, str] = {}
    cur_phase_i, cur_count = 0, base_count
    for ph in range(n_phases):
        if ph > 0:
            btype = "template" if ph % 2 == 1 else "volume"
            if btype == "template":
                cur_phase_i += 1          # modality moves (cycled), count held
            else:
                cur_count = hi_count if cur_count == base_count else base_count
            boundary_types[len(phase_per_window)] = btype
        name = phases[cur_phase_i % len(phases)]
        ln = win_per_ph if ph < n_phases - 1 else n_windows - len(phase_per_window)
        phase_per_window += [name] * ln
        count_per_window += [cur_count] * ln
    return phase_per_window, count_per_window, boundary_types


if __name__ == "__main__":
    pw, cw, bt = make_mixed_schedule(24)
    print("=== mixed (S5 Part 2) ===")
    for i in range(24):
        tag = f"  <- {bt[i]}" if i in bt else ""
        print(f"  win[{i:2d}] {pw[i]:7s} n={cw[i]}{tag}")
    for mode in ("irregular", "regular"):
        pw, ss, dw = make_schedule(120, seed=7100, mode=mode)
        print(f"\n=== mode={mode}  n_windows={len(pw)}  seasons={len(ss)}  "
              f"drifts={len(dw)} ===")
        for s in ss:
            print(f"  win[{s.start:3d}..{s.end-1:3d}] len={s.length:2d}  {s.phase}")
