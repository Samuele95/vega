"""
vibration — one-second condition-monitoring frames of one rotating machine (lanes L12, L13).

    frame = make(seed, rows)        # `rows` frames, one a second, in time order
    SCHEMA, RECIPE, VOCAB           # the lanekit.datasets protocol

One row is one frame: five statistics a vibration sensor computes over one second of signal.
The machine runs through contiguous segments of 40-120 frames, each with a condition — no fault,
or a set of the three faults `imbalance`, `misalignment` and `bearing` (most segments carry one,
a few carry two). The generator plants what each fault does to the statistics:

- imbalance raises the RMS velocity and keeps the spectral peak at the running speed (25 Hz);
- misalignment moves the peak to twice the running speed (50 Hz) and warms the machine a little;
- a bearing defect sends impulses on about a third of its frames — kurtosis and crest factor jump
  and the peak moves into the bearing band (130-160 Hz) — and heats the machine slowly;

each fault ramps in over the first frames of its segment, the temperature follows the heat with
a lag of about half a minute, and every statistic carries noise. A single frame is often
ambiguous (an impulse-free frame of a bearing segment looks healthy); a window of frames is
not — which is what the window lanes are for.

Two label columns describe the condition, and each lane selects one (the other leaves as
leakage, lanekit.recipes):

- `state` (L12, multiclass): the dominant condition — bearing over misalignment over imbalance,
  `normal` without a fault;
- `faults` (L13, multilabel): the faults present, `|`-delimited in class order, the empty cell
  for none.

The segments are laid out per part of the time split the recipe applies (70 / 15 / 15), and each
part opens with every single-fault condition and the healthy one, so every class and every
label is present, and absent, in the train, valid and test parts alike.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from . import ColumnSpec
from ..recipes import standard_recipe, time_split_cuts

__all__ = ["make", "SCHEMA", "RECIPE", "VOCAB", "VOCAB_WITHHELD", "FAULTS", "STATES", "START", "layout"]

FAULTS = ("imbalance", "misalignment", "bearing")
STATES = ("normal",) + FAULTS
START = "2026-06-01 08:00:00"
SEGMENT = (40, 120)                   # a condition lasts 40-120 frames
RUNNING_HZ = 25.0
# the conditions a segment may carry after the opening ones, and how often
_CONDITIONS = ((), ("imbalance",), ("misalignment",), ("bearing",), ("imbalance", "misalignment"),
               ("imbalance", "bearing"), ("misalignment", "bearing"))
_WEIGHTS = (0.3, 0.16, 0.16, 0.16, 0.08, 0.07, 0.07)

SCHEMA = (
    ColumnSpec("ts", "datetime", "time", None, "the second the frame was recorded"),
    ColumnSpec("rms", "float", "numeric", "mm/s", "the RMS vibration velocity over the frame"),
    ColumnSpec("kurtosis", "float", "numeric", None, "the kurtosis of the frame's signal (3 for a smooth one)"),
    ColumnSpec("crest", "float", "numeric", None, "the crest factor: peak over RMS"),
    ColumnSpec("peak_hz", "float", "numeric", "Hz", "the frequency of the frame's largest spectral peak"),
    ColumnSpec("temp_c", "float", "numeric", "degC", "the housing temperature"),
    ColumnSpec("state", "str", "label", None, "the machine's dominant condition: normal, imbalance, "
               "misalignment or bearing"),
    ColumnSpec("faults", "str", "label", None, "the faults present, delimited by |, empty for none"),
)

# the words this dataset brings to a page (D6 §4: column names, class names, entity, row noun),
# read by the literal lint and the page leak check (lint_templates.py, D6 §10.6)
VOCAB = frozenset({"vibration", "kurtosis", "crest", "peak_hz", "temp_c", "faults", "imbalance",
                   "misalignment", "bearing", "housing", "rotating"})
# D6 §4's words VOCAB withholds for now. The shared templates use each in a sense of their own (a
# build machine, a machine code, a page renderer's root mean square), and lint-allow.json has no
# sense-scoped entry for them yet, so in VOCAB they would fail the literal gate there (and the
# serving and page suites that run it on their own files). They join VOCAB once lint-allow.json
# holds test_lanekit_windows.SENSE_ALLOW (a need of T04, its owner), unless the narrower VOCAB is
# ratified instead; that test checks the entries cover every line that uses them.
VOCAB_WITHHELD = frozenset({"machine", "rms"})

ENTITY_COMMENT = "One second of a rotating machine's vibration, summarised as five statistics."
COMPONENT = "monitoring"
OVERVIEW = (
    "# vibration\n\n"
    "Condition monitoring of one rotating machine, one frame a second.\n\n"
    "The domain has one class, Frame: one second of the machine's vibration signal, summarised by "
    "the sensor as its RMS velocity, kurtosis, crest factor and dominant frequency, with the "
    "housing temperature. Two attributes describe the machine's condition in that second: "
    "`state`, the dominant condition, and `faults`, every fault present; the chain partitions on "
    "one of them at a time.\n")


def RECIPE(raw_path: Path, shaped_dir: Path, lane: Any) -> dict:  # noqa: N802 (the protocol's name)
    """The standard recipe over SCHEMA (lanekit.recipes): the lane's label is selected, the
    other label column leaves as leakage, the split is in time."""
    return standard_recipe(SCHEMA, raw_path, shaped_dir, lane)


# --------------------------------------------------------------------------- the segments
def _portions(rows: int) -> list[tuple[int, int]]:
    """The spans of the time split's parts, in order (train, valid, test)."""
    cuts = time_split_cuts(rows)
    return [(a, b) for a, b in zip(cuts, cuts[1:]) if b > a]


def _lengths(rng, n: int, k: int) -> list[int]:
    """k segment lengths that sum to n, each within SEGMENT when n / k is: equal shares, then
    random transfers between two segments that keep both in range."""
    lo, hi = min(SEGMENT[0], n // k), max(SEGMENT[1], -(-n // k))
    lengths = [n // k + (1 if i < n % k else 0) for i in range(k)]
    for _ in range(8 * k):
        i, j = (int(x) for x in rng.choice(k, size=2, replace=False))
        d = int(rng.integers(1, 30))
        if lengths[i] - d >= lo and lengths[j] + d <= hi:
            lengths[i] -= d
            lengths[j] += d
    return lengths


def _portion(rng, n: int) -> list[tuple[tuple[str, ...], int]]:
    """One part's segments: the healthy condition and each single fault first, in a random
    order, then conditions drawn by _WEIGHTS, no two neighbours alike."""
    k = max(4, int(round(2 * n / sum(SEGMENT))))
    while k > 4 and n / k < SEGMENT[0]:
        k -= 1
    opening = [()] + [(f,) for f in FAULTS]
    order = [opening[int(i)] for i in rng.permutation(4)]
    while len(order) < k:
        pick = _CONDITIONS[int(rng.choice(len(_CONDITIONS), p=_WEIGHTS))]
        if pick != order[-1]:
            order.append(pick)
    return list(zip(order, _lengths(rng, n, k)))


def layout(seed: int, rows: int) -> list[tuple[tuple[str, ...], int, int]]:
    """(condition, first frame, frame count) of every segment, deterministic in (seed, rows)."""
    import numpy as np
    rng = np.random.default_rng([int(seed), 0x5EB])
    out, at = [], 0
    for a, b in _portions(int(rows)):
        for condition, n in _portion(rng, b - a):
            out.append((condition, at, n))
            at += n
    return out


# --------------------------------------------------------------------------- the frames
def _state(condition: tuple[str, ...]) -> str:
    for fault in ("bearing", "misalignment", "imbalance"):
        if fault in condition:
            return fault
    return "normal"


def _faults(condition: tuple[str, ...]) -> str:
    return "|".join(f for f in FAULTS if f in condition)


def _severity(segments, rows: int, rng):
    """{fault: per-frame severity in [0, 1]}: 0 outside the fault's segments, ramping in over
    the first 6-12 frames of each, then holding with a little wander."""
    import numpy as np
    out = {f: np.zeros(rows) for f in FAULTS}
    for condition, first, n in segments:
        ramp = int(rng.integers(6, 13))
        shape = np.minimum(1.0, (np.arange(n) + 1) / ramp)
        for fault in condition:
            wander = np.clip(0.85 + 0.15 * np.cumsum(rng.normal(0, 0.05, n)), 0.6, 1.1)
            out[fault][first:first + n] = shape * wander
    return out


def _statistics(sev, rows: int, rng):
    """The five statistics of every frame from the faults' severities (module docstring)."""
    import numpy as np
    imb, mis, bear = sev["imbalance"], sev["misalignment"], sev["bearing"]
    ar = np.zeros(rows)
    for t in range(1, rows):
        ar[t] = 0.7 * ar[t - 1] + rng.normal(0, 0.08)
    impulse = (rng.random(rows) < 0.33) & (bear > 0.3)
    burst = impulse * rng.uniform(4.0, 8.0, rows)
    rms = 1.2 + 1.9 * imb + 0.9 * mis + 0.4 * bear + ar + rng.normal(0, 0.05, rows)
    kurt = 3.0 + rng.normal(0, 0.25, rows) + bear * burst + 0.3 * mis
    crest = 3.1 + 0.3 * imb + 0.25 * mis + bear * (0.35 + 0.28 * burst) + rng.normal(0, 0.12, rows)
    strength = np.stack([1.0 + 2.2 * imb + rng.normal(0, 0.3, rows),              # 1x running speed
                         0.4 + 2.4 * mis + rng.normal(0, 0.3, rows),              # 2x
                         bear * burst * 0.6 + rng.normal(0, 0.1, rows)])          # bearing band
    band = np.argmax(strength, axis=0)
    peak = np.select([band == 0, band == 1], [RUNNING_HZ, 2 * RUNNING_HZ], rng.uniform(130, 160, rows))
    peak = peak + rng.normal(0, 0.6, rows)
    heat, temp = 4.0 * mis + 8.0 * bear, np.empty(rows)
    level = 0.0
    for t in range(rows):
        level += (heat[t] - level) / 30.0
        temp[t] = 41.0 + level + 0.8 * np.sin(2 * np.pi * t / 900) + rng.normal(0, 0.25)
    return {"rms": np.round(np.clip(rms, 0.2, None), 4), "kurtosis": np.round(kurt, 4),
            "crest": np.round(np.clip(crest, 1.5, None), 4), "peak_hz": np.round(peak, 2),
            "temp_c": np.round(temp, 3)}


def make(seed: int, rows: int):
    """`rows` frames from START, one a second, deterministic in (seed, rows)."""
    import numpy as np
    import pandas as pd

    rows = int(rows)
    if rows < 4:
        raise ValueError(f"vibration needs at least 4 rows (one frame per condition), got {rows}")
    segments = layout(seed, rows)
    rng = np.random.default_rng([int(seed), 0x71B])
    sev = _severity(segments, rows, rng)
    condition = [c for c, _, n in segments for _ in range(n)]
    frame = pd.DataFrame({"ts": pd.date_range(START, periods=rows, freq="s").strftime("%Y-%m-%d %H:%M:%S"),
                          **_statistics(sev, rows, rng),
                          "state": [_state(c) for c in condition], "faults": [_faults(c) for c in condition]})
    return frame[[c.name for c in SCHEMA]]
