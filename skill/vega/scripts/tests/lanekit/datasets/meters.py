"""
meters — the daily load profile of households, read by their smart meters in twelve two-hour
bins (lane L07: a PCA-floor reconstruction, no label).

    frame = make(seed, rows)        # meter_id, day, kwh_00_02 .. kwh_22_24, tariff

A row is one household-day. The signal is a low-rank one, by construction:

- every household follows one of three daily profiles — commuters (a morning and an evening
  peak), people at home (a broad daytime plateau), night loads (storage heating, a car
  charged after midnight);
- a household scales its profile by its own level (a lognormal consumption), and each day
  by the weather (a heating day uses more), with a weekend shifting commuters toward the
  at-home shape;
- each two-hour reading carries metering noise and never goes below zero.

So the twelve readings live near a three-dimensional subspace, which is what the lane's
bottleneck of three reconstructs; the weekend shift and the noise are what it cannot.

`tariff` is the household's tariff (standard, night_rate, daytime_saver). It follows the
profile — most night loads are on the night rate — and it is the label a supervised reading of
these rows would take; the reconstruction lane takes none, so its geometry states the
candidate and abstains, and the recipe drops the tariff before the split by the lane's goal
stance (lanekit.goal).
Deterministic in (seed, rows).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from . import ColumnSpec
from ..goal import goal_recipe

__all__ = ["SCHEMA", "RECIPE", "VOCAB", "VOCAB_WITHHELD", "BINS", "PROFILES", "DAYS_PER_METER", "make"]

BINS = tuple(f"kwh_{h:02d}_{h + 2:02d}" for h in range(0, 24, 2))
DAYS_PER_METER = 14
START = "2026-01-05"
# the share of the day's energy each profile draws in each two-hour bin
PROFILES = {
    "commuter": (0.03, 0.02, 0.02, 0.14, 0.06, 0.04, 0.04, 0.05, 0.08, 0.20, 0.19, 0.13),
    "home":     (0.03, 0.02, 0.02, 0.07, 0.10, 0.12, 0.13, 0.12, 0.10, 0.12, 0.10, 0.07),
    "night":    (0.16, 0.18, 0.15, 0.06, 0.04, 0.04, 0.04, 0.04, 0.05, 0.08, 0.07, 0.09)}
PROFILE_SHARES = (0.45, 0.35, 0.20)
TARIFFS = ("standard", "night_rate", "daytime_saver")
# P(tariff | profile), in TARIFFS order
TARIFF_GIVEN = {"commuter": (0.80, 0.10, 0.10), "home": (0.55, 0.05, 0.40), "night": (0.15, 0.80, 0.05)}

SCHEMA = (
    ColumnSpec("meter_id", "str", "key", comment="The smart meter's serial number."),
    ColumnSpec("day", "datetime", "time", comment="The day the readings were taken."),
    *(ColumnSpec(b, "float", "numeric", "kWh",
                 f"The energy drawn between {b[4:6]}:00 and {b[7:9]}:00.") for b in BINS),
    ColumnSpec("tariff", "str", "label", comment="The tariff the household is billed on."),
)

# the words this dataset brings to a page (D6 §4: column names, class names, entity, row noun),
# read by the literal lint and the page leak check (lint_templates.py, D6 §10.6)
VOCAB = frozenset({"meter", "meters", "meter_id", "MeterDay", "kwh", *BINS, "tariff", "night_rate",
                   "daytime_saver", "household", "households", "commuter"})
# D6 §4's words VOCAB leaves out: `standard` (a tariff) is the templates' own word (standard
# input, standard errors, 7 lines), so in VOCAB it would fail the literal gate there; it joins
# VOCAB with a sense-scoped lint-allow.json entry per line (T04's file)
VOCAB_WITHHELD = frozenset({"standard"})
COMPONENT = "metering"
ENTITY_COMMENT = "One household-day: the energy a smart meter read in twelve two-hour bins."
OVERVIEW = """# meters

A smart meter reads the energy a household draws, in twelve two-hour bins a day. A row is one
household-day, and no property is singled out as a label: the chain reads the rows as they
are and asks how few directions describe them.

The domain has one class, MeterDay. The meter's serial number and the day name a row without
describing it, so the chain drops them before anything is fitted. The household's tariff is
recorded too; it is a label a supervised question could take, and this one does not.

The twelve readings share most of their shape: a household is a commuter, at home, or a night
load, scaled by how much it draws and by how cold the day is.
"""


def make(seed: int, rows: int):
    """`rows` household-days (DAYS_PER_METER per meter, the last meter cut short),
    deterministic in (seed, rows)."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    n = int(rows)
    meters = -(-n // DAYS_PER_METER)
    kinds = tuple(PROFILES)
    profile = rng.choice(len(kinds), size=meters, p=PROFILE_SHARES)
    level = rng.lognormal(np.log(10.0), 0.35, meters)                  # kWh a day
    tariff = np.array([rng.choice(TARIFFS, p=TARIFF_GIVEN[kinds[k]]) for k in profile])
    days = pd.date_range(START, periods=DAYS_PER_METER, freq="D")
    heating = rng.normal(1.0, 0.12, DAYS_PER_METER)                    # the weather, shared by all
    m = np.repeat(np.arange(meters), DAYS_PER_METER)[:n]
    d = np.tile(np.arange(DAYS_PER_METER), meters)[:n]
    shapes = np.array([PROFILES[k] for k in kinds])[profile[m]]
    weekend = (days.dayofweek.to_numpy()[d] >= 5) & (profile[m] == 0)
    shapes = np.where(weekend[:, None], 0.5 * shapes + 0.5 * np.array(PROFILES["home"]), shapes)
    energy = level[m] * heating[d] * rng.lognormal(0.0, 0.10, n)
    kwh = np.clip(shapes * energy[:, None] + rng.normal(0.0, 0.05, (n, len(BINS))), 0.0, None)
    frame = pd.DataFrame({"meter_id": [f"M{int(k) + 1:05d}" for k in m],
                          "day": days[d].strftime("%Y-%m-%d")})
    for j, b in enumerate(BINS):
        frame[b] = np.round(kwh[:, j], 3)
    frame["tariff"] = tariff[m]
    return frame[[c.name for c in SCHEMA]]


def RECIPE(raw_path: Path, shaped_dir: Path, lane: Any) -> dict:  # noqa: N802 (the protocol's name)
    """The standard recipe (lanekit.recipes) through lanekit.goal: the meter and the
    day dropped, the tariff dropped by the lane's goal stance (a reconstruction takes no label),
    a random split, the twelve readings scaled on train."""
    return goal_recipe(SCHEMA, raw_path, shaped_dir, lane)
