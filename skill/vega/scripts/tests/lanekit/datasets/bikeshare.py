"""
bikeshare — hourly rentals at three docking stations of one city (lanes L10, L11, L18).

    frame = make(seed, rows)        # `rows` rows: ceil(rows / 3) hours x 3 stations, cut to rows
    SCHEMA, RECIPE, VOCAB           # the lanekit.datasets protocol

One row is one hour at one station. The three stations share the city's weather (a persistent
`weather` state, the temperature, the humidity) and differ in who rides from them: `campus` is a
commuter station (morning and evening peaks on working days), `harbour` a leisure one (a broad
afternoon peak, strongest on weekends and holidays), `oldtown` a mix of the two. The label is
`demand`, the rentals of that hour:

    demand = base[station] * profile[station](hour, working day) * weather effect
             * temperature effect * exp(e_t) + noise,     e_t = 0.6 e_(t-1) + N(0, 0.12)

so the signal is a daily and weekly seasonality (season 24), a weather and temperature effect
the exogenous columns carry, and an autocorrelated residual only the target's own past reveals —
which is why L10 reads its lags and its autoregressive channel, and why L11 (exogenous only,
the user's stance) still has the seasonality and the weather to learn.

The rows come in time order, the three stations of an hour together; `station` is the series
key (`series_by`), `ts` the order (`order_by`). The recipe is lanekit.recipes.standard_recipe:
`weather` is encoded ordinal where L10 embeds it and one-hot where L11 and L18 ask for it, the
lag of `demand` is L10's (`recipe_options.lag`), the split is in time.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

from . import ColumnSpec
from ..recipes import standard_recipe

__all__ = ["make", "SCHEMA", "RECIPE", "VOCAB", "VOCAB_WITHHELD", "STATIONS", "WEATHER", "START", "HOLIDAYS"]

STATIONS = ("harbour", "campus", "oldtown")
WEATHER = ("sunny", "overcast", "rainy")
START = "2026-04-01 00:00:00"
HOLIDAYS = ("2026-04-06", "2026-04-25", "2026-05-01")       # public holidays in the span

# station -> (base rentals an hour, share of the commuter profile, microclimate offset in degC,
# extra wind in m/s): the harbour is breezier and a degree cooler
_STATION = {"harbour": (38.0, 0.1, -1.0, 1.5), "campus": (52.0, 0.9, 0.0, 0.0),
            "oldtown": (30.0, 0.5, 0.5, 0.3)}
_WEATHER_EFFECT = {"sunny": 1.0, "overcast": 0.85, "rainy": 0.45}
# hourly persistence of the weather state and where it goes when it changes
_WEATHER_STAY = 0.94
_WEATHER_NEXT = {"sunny": (0.0, 0.75, 0.25), "overcast": (0.5, 0.0, 0.5), "rainy": (0.35, 0.65, 0.0)}

SCHEMA = (
    ColumnSpec("ts", "datetime", "time", None, "the start of the hour the rentals were counted in"),
    ColumnSpec("station", "str", "series", None, "the docking station the hour was counted at"),
    ColumnSpec("temp_c", "float", "numeric", "degC", "the air temperature at the station"),
    ColumnSpec("humidity_pct", "float", "numeric", "%", "the relative humidity"),
    ColumnSpec("wind_ms", "float", "numeric", "m/s", "the mean wind speed"),
    ColumnSpec("hour", "int", "numeric", "h", "the hour of the day, 0 to 23 (a daily cycle)"),
    ColumnSpec("weekday", "int", "numeric", None, "the day of the week, 0 Monday to 6 Sunday"),
    ColumnSpec("holiday", "int", "numeric", None, "1 on a public holiday, else 0"),
    ColumnSpec("weather", "str", "nominal", None, "the weather of the hour: sunny, overcast or rainy"),
    ColumnSpec("demand", "float", "label", "rentals", "the bikes rented at the station in the hour"),
)

# the words this dataset brings to a page (D6 §4: column names, class names, entity, row noun),
# read by the literal lint and the page leak check (lint_templates.py, D6 §10.6)
VOCAB = frozenset({"bikeshare", "bike", "bikes", "rental", "rentals", "docking", "sunny", "overcast",
                   "rainy", "temp_c", "humidity_pct", "wind_ms", "weekday", "holiday", "harbour",
                   "campus", "oldtown", "commuter"})
# D6 §4's words VOCAB withholds for now. The shared templates use each in a sense of their own
# ("opens on demand", a page fixture's series keyed by station, a roles test's weather basis), and
# lint-allow.json has no sense-scoped entry for them yet, so in VOCAB they would fail the literal
# gate there. They join VOCAB once lint-allow.json holds test_lanekit_windows.SENSE_ALLOW (a need
# of T04, its owner), unless the narrower VOCAB is ratified instead; that test checks the entries
# cover every line that uses them.
VOCAB_WITHHELD = frozenset({"demand", "station", "weather"})

ENTITY_COMMENT = "One hour of rentals at one docking station, with the weather of that hour."
COMPONENT = "rentals"
OVERVIEW = (
    "# bikeshare\n\n"
    "Hourly bike rentals at three docking stations of one city.\n\n"
    "The domain has one class, Hour: one hour at one station, counted from the start of the hour. "
    "Every property is known when the hour ends: the weather and the temperature the city had, "
    "the calendar of the day, and `demand`, the bikes rented at the station in that hour — the "
    "attribute the chain forecasts. The stations are three series of the same city: they share "
    "its weather and differ in who rides from them.\n")


def RECIPE(raw_path: Path, shaped_dir: Path, lane: Any) -> dict:  # noqa: N802 (the protocol's name)
    """The standard recipe over SCHEMA (lanekit.recipes): the lane decides the encoding of
    `weather`, the lag of `demand` and the time split."""
    return standard_recipe(SCHEMA, raw_path, shaped_dir, lane)


# --------------------------------------------------------------------------- the rows
def _bump(hour, centre: float, width: float):
    import numpy as np
    return np.exp(-0.5 * ((hour - centre) / width) ** 2)


def _profile(hour, working, commuter: float):
    """The share of the station's base rented in an hour: a commuter's two peaks on working
    days, a leisure rider's broad afternoon (strongest on days off), mixed by `commuter`."""
    import numpy as np
    night = 0.06
    commute = np.where(working,
                       1.0 * _bump(hour, 8, 1.3) + 0.9 * _bump(hour, 17.5, 1.6) + 0.2 * _bump(hour, 13, 3),
                       0.45 * _bump(hour, 13, 3.5))
    leisure = np.where(working, 0.45 * _bump(hour, 13.5, 3.0) + 0.35 * _bump(hour, 18.5, 1.5),
                       1.0 * _bump(hour, 14, 3.5))
    return night + commuter * commute + (1.0 - commuter) * leisure


def _weather(rng, hours: int):
    """The city's weather state, hour by hour: persistent, changing to a neighbouring state."""
    import numpy as np
    state = np.empty(hours, dtype=object)
    current = "sunny"
    for t in range(hours):
        if t and rng.random() > _WEATHER_STAY:
            current = WEATHER[int(rng.choice(3, p=_WEATHER_NEXT[current]))]
        state[t] = current
    return state


def _city(rng, stamps, weather):
    """The city's temperature (a daily cycle, a slow warming, a day-to-day anomaly, cooler under
    rain) and humidity (wetter under clouds and rain, drier when warm)."""
    import numpy as np
    hours = len(stamps)
    hour = stamps.hour.to_numpy()
    day = (stamps.normalize() - stamps[0].normalize()).days.to_numpy()
    anomaly = np.zeros(int(day.max()) + 1)
    for d in range(1, len(anomaly)):
        anomaly[d] = 0.6 * anomaly[d - 1] + rng.normal(0, 1.6)
    wet = np.array([{"sunny": 0.0, "overcast": 1.0, "rainy": 2.5}[w] for w in weather])
    temp = (11.0 + 0.15 * day + 5.5 * np.sin(2 * np.pi * (hour - 9) / 24) + anomaly[day]
            - 1.2 * wet + rng.normal(0, 0.4, hours))
    humidity = np.clip(62 + 9 * wet - 1.1 * (temp - 13) + rng.normal(0, 3, hours), 18, 100)
    return temp, humidity


class _City(NamedTuple):
    """What the three stations share, hour by hour: the time, the weather state, the
    temperature and humidity, and whether the day is a working day."""
    stamps: Any
    weather: Any
    temp: Any
    humidity: Any
    working: Any


def _station_rows(rng, name: str, city: _City):
    """One station's hours: its own wind and microclimate, and its rentals (module docstring)."""
    import numpy as np
    import pandas as pd
    base, commuter, offset, breeze = _STATION[name]
    stamps, weather, humidity = city.stamps, city.weather, city.humidity
    hours = len(stamps)
    wind = np.empty(hours)
    level = 3.0 + breeze
    for t in range(hours):
        level = max(0.2, 0.85 * level + 0.15 * (3.0 + breeze) + rng.normal(0, 0.6))
        wind[t] = level
    t_here = city.temp + offset
    e = np.zeros(hours)
    for t in range(1, hours):
        e[t] = 0.6 * e[t - 1] + rng.normal(0, 0.12)
    effect = np.array([_WEATHER_EFFECT[w] for w in weather])
    warmth = np.clip(1 + 0.035 * (t_here - 14), 0.4, 1.5)
    gust = np.clip(1 - 0.03 * np.maximum(wind - 6, 0), 0.6, 1.0)
    hour = stamps.hour.to_numpy()
    demand = base * _profile(hour, city.working, commuter) * effect * warmth * gust * np.exp(e)
    demand = np.clip(demand + rng.normal(0, 0.8, hours), 0, None)
    return pd.DataFrame({
        "ts": stamps.strftime("%Y-%m-%d %H:%M:%S"), "station": name,
        "temp_c": np.round(t_here, 2),
        "humidity_pct": np.round(humidity + rng.normal(0, 1, hours), 1).clip(15, 100),
        "wind_ms": np.round(wind, 2), "hour": hour.astype(int),
        "weekday": stamps.dayofweek.to_numpy().astype(int),
        "holiday": np.isin(stamps.strftime("%Y-%m-%d"), HOLIDAYS).astype(int), "weather": weather,
        "demand": np.round(demand, 1)})


def make(seed: int, rows: int):
    """`rows` rows, deterministic in (seed, rows): ceil(rows / 3) hours from START at each of the
    three stations, in time order (the stations of an hour together), cut to `rows`."""
    import numpy as np
    import pandas as pd

    if int(rows) < len(STATIONS):
        raise ValueError(f"bikeshare needs at least {len(STATIONS)} rows (one hour per station), got {rows}")
    rng = np.random.default_rng(int(seed))
    hours = -(-int(rows) // len(STATIONS))
    stamps = pd.date_range(START, periods=hours, freq="h")
    weather = _weather(rng, hours)
    temp, humidity = _city(rng, stamps, weather)
    holiday = np.isin(stamps.strftime("%Y-%m-%d"), HOLIDAYS)
    working = (stamps.dayofweek.to_numpy() < 5) & ~holiday
    city = _City(stamps, weather, temp, humidity, working)
    frames = [_station_rows(rng, name, city).assign(_t=np.arange(hours), _s=i) for i, name in enumerate(STATIONS)]
    frame = pd.concat(frames, ignore_index=True).sort_values(["_t", "_s"], kind="stable")
    frame = frame.drop(columns=["_t", "_s"]).head(int(rows)).reset_index(drop=True)
    return frame[[c.name for c in SCHEMA]]
