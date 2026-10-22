"""
The `roads` generator (lane L21, D6 §4): the speed sensors of a regional road network, one row
a sensor.

    frame = make(seed, rows)        # `rows` sensors on a 40 x 40 km region, deterministic in (seed, rows)

No column names a neighbour: the relation between sensors is their distance. The lane's
`graph_edges` step builds it as the 6 nearest neighbours in `[x_km, y_km]` (knn, weighed by the
distance, fitted on the train part — so no held-out sensor is ever another's neighbour), and its
network samples 6 then 4 neighbours a layer (GraphSAGE, `batching: sampled`).

The planted signal: `mean_speed` is the speed limit cut by a **smooth congestion field** — a
few town centres whose congestion fades with the distance — plus the sensor's own noise. A
sensor measures the field only through its `occupancy`, and that reading is noisy; the
occupancies of its nearest neighbours, averaged, read the field far better than its own does.
That is the reason message passing over the knn graph has something to learn that a
structure-blind model does not. Lanes and speed limits follow the road class (a motorway has
more lanes and a higher limit; a town street fewer and lower); `flow_vph` follows the lanes and
the occupancy as a fundamental diagram does.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from . import ColumnSpec
from ..recipes import standard_recipe

__all__ = ["make", "SCHEMA", "RECIPE", "VOCAB", "VOCAB_WITHHELD", "EXTENT_KM", "TOWNS"]

EXTENT_KM = 40.0
TOWNS = 4
ROAD_CLASSES = ((1, 50), (2, 70), (2, 90), (3, 110), (4, 130))    # (lanes, speed limit km/h)

SCHEMA = (
    ColumnSpec("sensor_id", "str", "key", comment="The sensor's identifier."),
    ColumnSpec("x_km", "float", "numeric", "km", "The sensor's easting from the region's corner."),
    ColumnSpec("y_km", "float", "numeric", "km", "The sensor's northing from the region's corner."),
    ColumnSpec("lanes", "int", "numeric", None, "The lanes of the carriageway the sensor watches."),
    ColumnSpec("speed_limit", "int", "numeric", "km/h", "The posted speed limit at the sensor."),
    ColumnSpec("flow_vph", "float", "numeric", "vehicles/h", "The vehicles the sensor counts an hour at peak."),
    ColumnSpec("occupancy", "float", "numeric", "%", "The share of time the sensor's loop is occupied at peak."),
    ColumnSpec("mean_speed", "float", "label", "km/h", "The mean speed the sensor measures at peak."),
)

# The words this dataset brings to a page (D6 §4: column names, class names, entity, row noun),
# read by the literal lint and the page leak check (lint_templates.py, D6 §10.6). `traffic` is
# L21's own (its source is a traffic network, and the page leak check must not read the word as
# the soft CIC token there). The entity and row noun, `sensor`, is one of foreign-literals.json's
# generic words, which are never a lane's.
VOCAB = frozenset({"sensor_id", "x_km", "y_km", "speed_limit", "flow_vph", "occupancy",
                   "mean_speed", "traffic", "road", "roads", "carriageway"})
# D6 §4's words VOCAB withholds for now. `lanes` (a column): the shared templates use it in a sense
# of their own (the vega-show `lanes` motif, its id, mark and field drawing, and the page's word for
# the lanes a string serves), and lint-allow.json has no sense-scoped entry for it yet, so in VOCAB
# it would fail the literal gate there.
# They join VOCAB once lint-allow.json holds test_lanekit_graph_generators.SENSE_ALLOW (a need of
# T04, its owner), unless the narrower VOCAB is ratified instead; that test checks the entries
# cover every line that uses them, and that no word is withheld that no template uses.
VOCAB_WITHHELD = frozenset({"lanes"})

ENTITY_COMMENT = "One speed sensor of a regional road network, with what it measured at peak."
COMPONENT = "roads"


def RECIPE(raw_path: Path, shaped_dir: Path, lane: Any) -> dict:  # noqa: N802 (the protocol's name)
    """The standard recipe over this SCHEMA: `sensor_id` dropped, every number scaled, a random
    split (a regression label has no class to stratify on), and `graph_edges` knn with k = 6 on
    the scaled `[x_km, y_km]`, weighed by the distance (lanekit.recipes)."""
    return standard_recipe(SCHEMA, raw_path, shaped_dir, lane)


def _positions(rng, rows: int):
    """(x, y, congestion): most sensors near one of the towns, the rest anywhere; the congestion
    field in [0, 1] at each, the sum of the towns' Gaussian bumps."""
    import numpy as np

    centre = rng.uniform(0.2 * EXTENT_KM, 0.8 * EXTENT_KM, (TOWNS, 2))
    strength = rng.uniform(0.6, 1.0, TOWNS)
    reach = rng.uniform(3.0, 6.0, TOWNS)
    near = rng.random(rows) < 0.6
    town = rng.integers(0, TOWNS, rows)
    xy = np.where(near[:, None], centre[town] + rng.normal(0.0, 3.5, (rows, 2)),
                  rng.uniform(0.0, EXTENT_KM, (rows, 2)))
    xy = np.clip(xy, 0.0, EXTENT_KM)
    d2 = ((xy[:, None, :] - centre[None, :, :]) ** 2).sum(axis=2)
    field = (strength[None, :] * np.exp(-d2 / (2.0 * reach[None, :] ** 2))).sum(axis=1)
    return xy[:, 0], xy[:, 1], np.clip(field, 0.0, 1.0)


def make(seed: int, rows: int):
    """`rows` sensors (module docstring), deterministic in (seed, rows)."""
    import numpy as np
    import pandas as pd

    if rows < 8:
        raise ValueError(f"roads needs at least 8 sensors (a sensor and its 6 neighbours), got {rows}")
    rng = np.random.default_rng(seed)
    x, y, congestion = _positions(rng, rows)
    # town streets where congestion is high, faster road classes away from the towns
    road = np.clip(np.round((1.0 - congestion) * 4.0 + rng.normal(0.0, 0.8, rows)), 0, 4).astype(np.int64)
    lanes = np.array([ROAD_CLASSES[r][0] for r in road], dtype=np.int64)
    limit = np.array([ROAD_CLASSES[r][1] for r in road], dtype=np.int64)
    occupancy = np.clip(6.0 + 42.0 * congestion + rng.normal(0.0, 9.0, rows), 1.0, 95.0)
    density = occupancy / 100.0
    flow = lanes * 2000.0 * density * (1.0 - density) * 2.2 + rng.normal(0.0, 60.0, rows)
    speed = limit * (1.0 - 0.6 * congestion) - 0.08 * (occupancy - 6.0 - 42.0 * congestion) \
        + rng.normal(0.0, 3.0, rows)
    return pd.DataFrame({
        "sensor_id": np.array([f"S{i + 1:05d}" for i in range(rows)], dtype=object),
        "x_km": np.round(x, 4), "y_km": np.round(y, 4), "lanes": lanes, "speed_limit": limit,
        "flow_vph": np.round(np.maximum(flow, 0.0), 1), "occupancy": np.round(occupancy, 3),
        "mean_speed": np.round(np.maximum(speed, 5.0), 3)})[[c.name for c in SCHEMA]]
