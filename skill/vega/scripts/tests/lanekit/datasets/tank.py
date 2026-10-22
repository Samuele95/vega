"""
tank — one process tank, one row a minute (lanes L14, L15, L16, L17).

    frame = make(seed, rows)        # `rows` minutes in time order
    SCHEMA, RECIPE, VOCAB           # the lanekit.datasets protocol

One row is one minute of the tank's controller log: the level, the measured inflow and outflow,
the outlet valve opening and the bottom pressure. The generator simulates the tank: a supply
line fills it (about 120 l/min, a booster adds 80 l/min when the level falls under 40 %, the
supply is throttled above 90 %), the outlet drains it through a valve the operator moves
between 30 % and 95 % (flow grows with the opening and with the square root of the head), and
the supply fails now and then for 20-90 minutes — the episodes that drain the tank. The
measured flows and the pressure carry sensor noise; the level is read to 0.1 %.

Three label columns describe the minute, and each lane selects one (the others leave as
leakage, lanekit.recipes):

- `alarm` (L14, L15; binary): 1 when the level falls under 12 % within the next 5 minutes;
- `minutes_to_empty` (L16; regression): the controller's estimate — the minute-by-minute time
  to empty (the level over that minute's net drain rate of the true flows, clipped to 0-120,
  120 while the tank fills), averaged over the trailing 15 minutes (A37/A38: a mean of a
  non-linear quantity of each minute, which an lstm pooling its steps reads and a linear model
  over the window cannot);
- `grade` (L17; ordinal 0-3): the operating band, read from the 15-minute trailing mean of the
  level and of the valve opening (cut at the quartiles of the whole log).

Each label reads the recent past, not one minute: the noisy flows average out over a window,
the drain rate is a trend, the band a trailing mean — what the window lanes are for. Every part
of the time split (70 / 15 / 15) holds a supply failure long enough to raise the alarm (a held part
two, one early and one late, outages()), so the positive class and the low bands are in the
train, valid and test parts alike, and in the first and the last quarter of each held part.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from . import ColumnSpec
from ..recipes import standard_recipe, time_split_cuts

__all__ = ["make", "SCHEMA", "RECIPE", "VOCAB", "VOCAB_WITHHELD", "START", "ALARM_LEVEL", "ALARM_AHEAD",
           "HORIZON_CAP", "outages"]

START = "2026-03-02 06:00:00"
CAPACITY_L = 10000.0                   # 100 l of net flow move the level by 1 % a minute
ALARM_LEVEL = 12.0                     # % : the alarm threshold
ALARM_AHEAD = 5                        # minutes the alarm looks ahead
HORIZON_CAP = 120.0                    # minutes: the clip of minutes_to_empty
BAND_WINDOW = 15                       # minutes of the trailing means grade reads
SUPPLY_LPM, BOOST_LPM = 120.0, 80.0
OUTAGE = (20, 90)                      # a supply failure lasts 20-90 minutes

SCHEMA = (
    ColumnSpec("ts", "datetime", "time", None, "the minute the controller logged"),
    ColumnSpec("level_pct", "float", "numeric", "%", "the tank's fill level"),
    ColumnSpec("inflow_lpm", "float", "numeric", "l/min", "the measured supply inflow"),
    ColumnSpec("outflow_lpm", "float", "numeric", "l/min", "the measured outlet flow"),
    ColumnSpec("valve_pct", "float", "numeric", "%", "the outlet valve opening"),
    ColumnSpec("pressure_bar", "float", "numeric", "bar", "the pressure at the tank's bottom"),
    ColumnSpec("alarm", "int", "label", None, "1 when the level falls under 12 % within 5 minutes, else 0"),
    ColumnSpec("minutes_to_empty", "float", "label", "min", "the time to empty at each minute's net drain "
               "rate (0-120), averaged over the trailing 15 minutes"),
    ColumnSpec("grade", "int", "label", None, "the operating band, 0 (lowest) to 3 (comfortable)"),
)

# the words this dataset brings to a page (D6 §4: column names, class names, entity, row noun),
# read by the literal lint and the page leak check (lint_templates.py, D6 §10.6)
VOCAB = frozenset({"tank", "inflow", "outflow", "valve", "pressure", "level_pct", "inflow_lpm",
                   "outflow_lpm", "valve_pct", "pressure_bar", "minutes_to_empty", "supply", "outlet",
                   "booster"})
# D6 §4's words VOCAB withholds for now. The shared templates use each in a sense of their own (the
# `alarm` answer and field kind, a page fixture's ordinal grade), and lint-allow.json has no
# sense-scoped entry for them yet, so in VOCAB they would fail the literal gate there (and the
# serving, monitor and page suites that run it on their own files). They join VOCAB once
# lint-allow.json holds test_lanekit_windows.SENSE_ALLOW (a need of T04, its owner), unless the
# narrower VOCAB is ratified instead; that test checks the entries cover every line that uses them.
VOCAB_WITHHELD = frozenset({"alarm", "grade"})

ENTITY_COMMENT = "One minute of a process tank's controller log."
COMPONENT = "process"
OVERVIEW = (
    "# tank\n\n"
    "The controller log of one process tank, one row a minute.\n\n"
    "The domain has one class, Minute: what the controller measured in one minute — the fill "
    "level, the supply inflow, the outlet flow, the outlet valve opening and the bottom pressure. "
    "Three attributes describe where the tank is heading: `alarm`, whether the level falls under "
    "the alarm threshold within five minutes; `minutes_to_empty` at the recent drain rates; and "
    "`grade`, the operating band. The chain partitions on one of them at a time.\n")


def RECIPE(raw_path: Path, shaped_dir: Path, lane: Any) -> dict:  # noqa: N802 (the protocol's name)
    """The standard recipe over SCHEMA (lanekit.recipes): the lane's label is selected, the
    other two leave as leakage, the split is in time."""
    return standard_recipe(SCHEMA, raw_path, shaped_dir, lane)


# --------------------------------------------------------------------------- the log
def outages(seed: int, minutes: int) -> list[tuple[int, int, bool]]:
    """(first minute, length, forced) of every supply failure over `minutes`, in time order,
    deterministic in (seed, minutes). Each part of the time split holds a forced failure early
    in the part (3-8 % in); a held part (valid, test) holds a second one past two thirds of it
    (66-72 % in), so the alarm rises in the first and in the last quarter of the part; a forced
    failure lasts 50-60 minutes, and the valve is wide open through it (_valve), so it drains
    the tank to the alarm. The train part holds about one more every five hours, at random."""
    import numpy as np
    rng = np.random.default_rng([int(seed), 0x7A2])
    cuts, out = time_split_cuts(minutes), []
    for i, (a, b) in enumerate(zip(cuts, cuts[1:])):
        span = b - a
        if span < 30:
            continue
        starts = [a + int(span * rng.uniform(0.03, 0.08))]
        if i and span >= 120:
            starts.append(a + int(span * rng.uniform(0.66, 0.72)))
        forced = [(first, int(min(rng.integers(50, 61), b - first)), True) for first in starts]
        out += forced
        if i == 0:
            for _ in range(int(rng.poisson(span / 300))):
                first = int(rng.integers(0, max(1, span - OUTAGE[0])))
                length = int(min(rng.integers(OUTAGE[0], OUTAGE[1] + 1), span - first))
                if all(first + length + 30 < f or first > f + n + 60 for f, n, _ in forced):
                    out.append((first, length, False))
    return sorted(out)


def _valve(rng, minutes: int, forced):
    """The operator's valve opening: steps to a new set point every 20-60 minutes, eased in;
    wide open (80-92 %) through a forced supply failure — the demand that drains the tank — and
    held nearly shut for 45 minutes after it, so the tank refills."""
    import numpy as np
    target_at = np.full(minutes, np.nan)
    for first, length in forced:
        target_at[first:first + length] = rng.uniform(80, 92)
        target_at[first + length:first + length + 45] = 32.0
    out, value, target, next_move = np.empty(minutes), 60.0, 60.0, 0
    for t in range(minutes):
        if not np.isnan(target_at[t]):
            target, next_move = float(target_at[t]), t + 1
        elif t >= next_move:
            target = float(rng.uniform(30, 95))
            next_move = t + int(rng.integers(20, 61))
        value += 0.3 * (target - value)
        out[t] = value
    return out


def _simulate(rng, minutes: int, failed, valve):
    """(level %, true inflow, true outflow) minute by minute: the supply with its booster and
    throttle, the outlet through the valve, the level they leave."""
    import numpy as np
    level, lv = 65.0, np.empty(minutes)
    inflow, outflow = np.empty(minutes), np.empty(minutes)
    supply = SUPPLY_LPM
    for t in range(minutes):
        supply += 0.2 * (SUPPLY_LPM - supply) + rng.normal(0, 3)
        q_in = 0.0 if failed[t] else supply + (BOOST_LPM if level < 40 else 0.0) - (90.0 if level > 90 else 0.0)
        q_in = max(q_in, 0.0)
        q_out = 2.6 * valve[t] * np.sqrt(max(level, 0.0) / 100.0 + 0.05)
        level = float(np.clip(level + (q_in - q_out) / CAPACITY_L * 100.0, 0.0, 100.0))
        lv[t], inflow[t], outflow[t] = level, q_in, q_out
    return lv, inflow, outflow


def _labels(level, inflow, outflow, valve, rows: int):
    """alarm, minutes_to_empty and grade of the first `rows` minutes (module docstring); the
    log runs ALARM_AHEAD minutes longer so the last rows' alarm has its future."""
    import numpy as np
    import pandas as pd
    ahead = np.array([level[t + 1:t + 1 + ALARM_AHEAD].min() for t in range(rows)])
    alarm = (ahead < ALARM_LEVEL).astype(int)
    drain = (outflow - inflow)[:rows] / CAPACITY_L * 100.0               # % a minute
    with np.errstate(divide="ignore", invalid="ignore"):
        tte = np.where(drain > 1e-6, level[:rows] / drain, HORIZON_CAP)
    tte = np.clip(tte, 0.0, HORIZON_CAP)
    # the controller's estimate: the minute-by-minute time to empty, averaged over the trailing
    # BAND_WINDOW minutes — a mean of a non-linear quantity of each minute (a ratio, clipped)
    tte = np.round(pd.Series(tte).rolling(BAND_WINDOW, min_periods=1).mean().to_numpy(), 2)
    lv = pd.Series(level[:rows]).rolling(BAND_WINDOW, min_periods=1).mean().to_numpy()
    vv = pd.Series(valve[:rows]).rolling(BAND_WINDOW, min_periods=1).mean().to_numpy()
    score = lv - 0.25 * (vv - 60.0)
    grade = np.searchsorted(np.quantile(score, [0.25, 0.5, 0.75]), score, side="right").astype(int)
    return alarm, tte, grade


def make(seed: int, rows: int):
    """`rows` minutes from START, deterministic in (seed, rows)."""
    import numpy as np
    import pandas as pd

    rows = int(rows)
    if rows < 30:
        raise ValueError(f"tank needs at least 30 rows (minutes), got {rows}")
    minutes = rows + ALARM_AHEAD
    rng = np.random.default_rng([int(seed), 0x7A1])
    failed = np.zeros(minutes, dtype=bool)
    spans = outages(seed, rows)
    for first, length, _ in spans:
        failed[first:first + length] = True
    valve = _valve(rng, minutes, [(first, length) for first, length, forced in spans if forced])
    level, inflow, outflow = _simulate(rng, minutes, failed, valve)
    alarm, tte, grade = _labels(level, inflow, outflow, valve, rows)
    n = rows
    frame = pd.DataFrame({
        "ts": pd.date_range(START, periods=n, freq="min").strftime("%Y-%m-%d %H:%M:%S"),
        "level_pct": np.round(level[:n] + rng.normal(0, 0.3, n), 1).clip(0, 100),
        "inflow_lpm": np.round(np.clip(inflow[:n] + rng.normal(0, 6, n), 0, None), 1),
        "outflow_lpm": np.round(np.clip(outflow[:n] + rng.normal(0, 6, n), 0, None), 1),
        "valve_pct": np.round(valve[:n] + rng.normal(0, 0.5, n), 1),
        "pressure_bar": np.round(0.15 + 0.009 * level[:n] + rng.normal(0, 0.01, n), 3),
        "alarm": alarm, "minutes_to_empty": tte, "grade": grade})
    return frame[[c.name for c in SCHEMA]]
