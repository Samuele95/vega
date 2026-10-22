"""
turbine — ten-minute SCADA readings of a wind farm's turbines, with the faults the maintenance
log confirmed (lanes L06 autoencoder, L08 VAE, L09 deep SVDD: the anomaly lanes).

    frame = make(seed, rows)        # turbine, ts, rotor_rpm, power_kw, wind_ms, pitch_deg,
                                    # gen_temp_c, gear_temp_c, vibration_mm_s, yaw_err_deg,
                                    # nacelle_temp_c, ambient_c, fault

The healthy machine is physics, stated as formulas: the wind follows a persistent process per
turbine; the power follows the power curve (nothing below cut-in at 3 m/s, the cube of the
wind up to the rated 12 m/s, the rated 2 MW above it, nothing past cut-out at 25 m/s); the rotor
speed rises with the wind to its rated speed; the blades pitch out only above the rated wind;
the nacelle warms with the ambient air and the power, and the generator and the gearbox warm
with the nacelle and the power; the drive train vibrates with the rotor speed.

A fault breaks one of those relations for a few consecutive readings of one turbine: a gearbox
running hot for the power it delivers (a drift of 10 to 20 degrees), a vibration burst (two and
a half to four times the rotor's own), or both. About 3 % of the readings are faulty — the rare
side an anomaly lane is about. `fault` says which readings (ok or fault), as the maintenance
log confirmed them.

One table serves the three lanes: L06 and L09 select `fault` as their rare label; L08 is the
unlabelled variant — its lane takes no label, so its geometry states `fault` as a candidate and
abstains, the recipe drops it before the split by the lane's goal stance (lanekit.goal),
and nothing the network or the evaluation reads carries it. `turbine` and `ts`
say which machine and when: identifiers on these row lanes, dropped before anything is fitted.
Deterministic in (seed, rows).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from . import ColumnSpec
from ..goal import goal_recipe

__all__ = ["SCHEMA", "RECIPE", "VOCAB", "VOCAB_WITHHELD", "STATES", "TURBINES", "FAULT_SHARE", "make"]

STATES = ("ok", "fault")
TURBINES = 8
FAULT_SHARE = 0.03
START = "2026-03-01 00:00:00"
RATED_KW, CUT_IN, RATED_WIND, CUT_OUT = 2000.0, 3.0, 12.0, 25.0

SCHEMA = (
    ColumnSpec("turbine", "str", "series", comment="Which turbine of the farm took the reading."),
    ColumnSpec("ts", "datetime", "time", comment="The end of the ten-minute interval the reading averages."),
    ColumnSpec("rotor_rpm", "float", "numeric", "rpm", "The rotor's speed."),
    ColumnSpec("power_kw", "float", "numeric", "kW", "The active power delivered to the grid."),
    ColumnSpec("wind_ms", "float", "numeric", "m/s", "The wind speed at the hub."),
    ColumnSpec("pitch_deg", "float", "numeric", "deg", "The blades' pitch angle."),
    ColumnSpec("gen_temp_c", "float", "numeric", "degC", "The generator's winding temperature."),
    ColumnSpec("gear_temp_c", "float", "numeric", "degC", "The gearbox oil temperature."),
    ColumnSpec("vibration_mm_s", "float", "numeric", "mm/s", "The drive train's vibration velocity."),
    ColumnSpec("yaw_err_deg", "float", "numeric", "deg", "The angle between the nacelle and the wind."),
    ColumnSpec("nacelle_temp_c", "float", "numeric", "degC", "The air temperature inside the nacelle."),
    ColumnSpec("ambient_c", "float", "numeric", "degC", "The outside air temperature."),
    ColumnSpec("fault", "str", "label", comment="Whether maintenance confirmed a fault in the interval."),
)

# the words this dataset brings to a page (D6 §4: column names, class names, entity, row noun),
# read by the literal lint and the page leak check (lint_templates.py, D6 §10.6). `vibration` is
# the vibration lanes' word too: each lane owns what its page says
VOCAB = frozenset({"turbine", "turbines", "rotor_rpm", "power_kw", "wind_ms", "pitch_deg", "gen_temp_c",
                   "gear_temp_c", "vibration_mm_s", "yaw_err_deg", "nacelle_temp_c", "ambient_c", "nacelle",
                   "gearbox", "rotor", "vibration", "scada"})
# D6 §4's words VOCAB leaves out: the templates use them in a sense of their own — `fault` "the
# field at fault" (4 lines), `anomalous` the anomaly task's own class word (79 lines, the word
# every anomaly lane shares) — so in VOCAB each would fail the literal gate there. `fault` joins
# VOCAB with a sense-scoped lint-allow.json entry per line (T04's file); `anomalous` belongs to
# the `generic` words of foreign-literals.json rather than to one lane. (`ok` is too short for
# the lint to read.)
VOCAB_WITHHELD = frozenset({"fault", "anomalous"})
COMPONENT = "farm"
ENTITY_COMMENT = "One ten-minute reading of one wind turbine, as its controller averaged it."
OVERVIEW = """# turbine

A wind turbine's controller averages its sensors every ten minutes: the wind, the power, the
rotor, the blades' pitch, the temperatures of the nacelle, the generator and the gearbox, the
vibration of the drive train. Maintenance later confirms which intervals were faulty.

The domain has one class, Reading. The turbine and the time say which reading it is without
describing it, so the chain drops them before anything is fitted.

A healthy turbine obeys its physics: the power follows the wind along the power curve, and the
temperatures follow the power. A fault is a reading that breaks one of those relations — a
gearbox too hot for what it delivers, a drive train shaking — and it is rare.
"""


def make(seed: int, rows: int):
    """`rows` readings, deterministic in (seed, rows): TURBINES machines read in turn every ten
    minutes, the healthy physics, then the fault episodes."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    n = int(rows)
    step = np.arange(n) // TURBINES                     # the ten-minute interval
    machine = np.arange(n) % TURBINES
    ambient = 9.0 + 5.0 * np.sin(2 * np.pi * (step / 144.0 - 0.3)) + rng.normal(0.0, 0.6, n)
    wind = _wind(rng, step, machine, n)
    yaw = rng.normal(0.0, 3.5, n)
    power = _power(wind) * np.cos(np.radians(yaw)) ** 2 * rng.normal(1.0, 0.03, n)
    rotor = np.where(wind < CUT_IN, rng.uniform(0.5, 2.0, n),
                     6.0 + 10.0 * np.clip((wind - CUT_IN) / (RATED_WIND - CUT_IN), 0.0, 1.0)) + rng.normal(0.0, 0.3, n)
    rotor = np.where(wind >= CUT_OUT, rng.uniform(0.2, 1.0, n), rotor)
    pitch = np.where(wind > RATED_WIND, 2.0 + 1.6 * (wind - RATED_WIND), 0.0) + np.abs(rng.normal(0.0, 0.4, n))
    nacelle = ambient + 8.0 + 0.004 * power + rng.normal(0.0, 0.8, n)
    gen = nacelle + 15.0 + 0.012 * power + rng.normal(0.0, 1.2, n)
    gear = nacelle + 10.0 + 0.009 * power + rng.normal(0.0, 1.0, n)
    vibration = np.abs(0.8 + 0.12 * rotor + rng.normal(0.0, 0.15, n))
    faulty, gear, vibration = _faults(rng, machine, n, gear, vibration)
    stamps = pd.Timestamp(START) + pd.to_timedelta(10 * (step + 1), unit="min")
    frame = pd.DataFrame({"turbine": [f"WT{int(k) + 1:02d}" for k in machine],
                          "ts": stamps.strftime("%Y-%m-%d %H:%M:%S"),
                          "rotor_rpm": np.round(rotor, 2), "power_kw": np.round(np.clip(power, 0.0, None), 1),
                          "wind_ms": np.round(wind, 2), "pitch_deg": np.round(pitch, 2),
                          "gen_temp_c": np.round(gen, 2), "gear_temp_c": np.round(gear, 2),
                          "vibration_mm_s": np.round(vibration, 3), "yaw_err_deg": np.round(yaw, 2),
                          "nacelle_temp_c": np.round(nacelle, 2), "ambient_c": np.round(ambient, 2),
                          "fault": np.where(faulty, STATES[1], STATES[0])})
    return frame[[c.name for c in SCHEMA]]


def _wind(rng: Any, step: Any, machine: Any, n: int):
    """A persistent wind per turbine: an AR(1) on the log speed around a farm-wide level."""
    import numpy as np

    steps = int(step.max()) + 1 if n else 0
    shocks = rng.normal(0.0, 0.18, (steps, TURBINES))
    log_wind = np.empty((steps, TURBINES))
    level = np.log(7.5)
    if steps:
        log_wind[0] = level + rng.normal(0.0, 0.5, TURBINES)
    for t in range(1, steps):
        log_wind[t] = level + 0.93 * (log_wind[t - 1] - level) + shocks[t]
    return np.clip(np.exp(log_wind[step, machine]), 0.3, 28.0)


def _power(wind: Any):
    """The power curve: nothing below cut-in, the cube of the wind to the rated speed, the rated
    power above it, nothing past cut-out."""
    import numpy as np

    ramp = np.clip((wind - CUT_IN) / (RATED_WIND - CUT_IN), 0.0, 1.0) ** 3
    return np.where((wind < CUT_IN) | (wind >= CUT_OUT), 0.0, RATED_KW * ramp)


def _faults(rng: Any, machine: Any, n: int, gear: Any, vibration: Any):
    """(faulty, gear, vibration): episodes of 3 to 8 consecutive readings of one turbine until
    FAULT_SHARE of the rows are faulty, each a gearbox drift, a vibration burst, or both."""
    import numpy as np

    faulty = np.zeros(n, dtype=bool)
    gear, vibration = gear.copy(), vibration.copy()
    target = int(round(FAULT_SHARE * n))
    while faulty.sum() < target:
        k = int(rng.integers(0, TURBINES))
        rows = np.flatnonzero(machine == k)
        start = int(rng.integers(0, max(1, len(rows) - 8)))
        episode = rows[start:start + int(rng.integers(3, 9))][: target - int(faulty.sum())]
        kind = int(rng.integers(0, 3))                   # 0 gearbox, 1 vibration, 2 both
        if kind in (0, 2):
            gear[episode] += rng.uniform(10.0, 20.0)
        if kind in (1, 2):
            vibration[episode] *= rng.uniform(2.5, 4.0)
        faulty[episode] = True
    return faulty, gear, vibration


def RECIPE(raw_path: Path, shaped_dir: Path, lane: Any) -> dict:  # noqa: N802 (the protocol's name)
    """The standard recipe (lanekit.recipes) through lanekit.goal: the turbine and the
    time dropped; on a labelled lane `fault` selected and a stratified split, on the unlabelled
    one `fault` dropped by the lane's goal stance and a random split; the ten sensors scaled on
    train."""
    return goal_recipe(SCHEMA, raw_path, shaped_dir, lane)
