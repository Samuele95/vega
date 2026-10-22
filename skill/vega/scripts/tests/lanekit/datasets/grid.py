"""
The `grid` generator (lane L19, D6 §4): the buses of a radial distribution grid, one row a bus.

    frame = make(seed, rows)        # `rows` buses in six supply zones, deterministic in (seed, rows)

Each zone is a feeder tree: its first bus is the substation (no `feeder_to`), and every later
bus is fed by an earlier bus of the same zone, mostly a recent one, so a zone grows as feeders
with branches. `feeder_to` names the upstream bus's `bus_key`; the lane's `graph_edges` step
reads the pair `[bus_key, feeder_to]` as a symmetric key relation (D6 §4: key_pair, the
source an analysis finding's candidate — lanekit.recipes F3/T3).

The planted signal is the one D6 names: `overloaded` compares the net load (demand less the
distributed generation) with the line capacity **summed over the bus's feeder neighbourhood**
— the bus, the bus feeding it and the buses it feeds — so a bus's own columns carry part of the
answer and its neighbours carry the rest: a structure-blind model reads the part, message
passing can read the whole. The voltage sags with the hops from the substation and with the
bus's own loading, as a feeder's does. About 30 % of the buses are overloaded, a stratified
split keeps the share in every part, and the positive class is listed second (`ok`,
`overloaded`), as netspec-contract asks of a binary task.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from . import ColumnSpec
from ..recipes import standard_recipe

__all__ = ["make", "SCHEMA", "RECIPE", "VOCAB", "VOCAB_WITHHELD", "ZONES", "CLASSES", "OVERLOADED_SHARE"]

ZONES = ("Z1", "Z2", "Z3", "Z4", "Z5", "Z6")
CLASSES = ("ok", "overloaded")
OVERLOADED_SHARE = 0.30
LINE_RATINGS_MW = (3.0, 5.0, 8.0, 12.0)

SCHEMA = (
    ColumnSpec("bus_key", "str", "key", comment="The bus's identifier."),
    ColumnSpec("feeder_to", "str", "key",
               comment="The key of the upstream bus that feeds this one; empty at a substation."),
    ColumnSpec("load_mw", "float", "numeric", "MW", "The bus's peak demand."),
    ColumnSpec("gen_mw", "float", "numeric", "MW", "The distributed generation connected at the bus."),
    ColumnSpec("voltage_pu", "float", "numeric", "p.u.", "The bus voltage at peak, per unit of nominal."),
    ColumnSpec("line_cap_mw", "float", "numeric", "MW", "The rated capacity of the line feeding the bus."),
    ColumnSpec("zone", "str", "nominal", comment="The supply zone the bus belongs to."),
    ColumnSpec("overloaded", "str", "label",
               comment="Whether the bus's feeder neighbourhood is overloaded at peak (ok or overloaded)."),
)

# The words this dataset brings to a page (D6 §4: column names, class names, entity, row noun),
# read by the literal lint and the page leak check (lint_templates.py, D6 §10.6). The class `ok`
# is too short to be a token.
VOCAB = frozenset({"bus", "buses", "bus_key", "feeder_to", "load_mw", "gen_mw", "voltage_pu",
                   "line_cap_mw", "overloaded", "feeder", "feeders", "substation", "substations"})
# D6 §4's words VOCAB withholds for now. `zone` (a column): the shared templates use it in a sense
# of their own (the page's prediction and registry fixtures' made-up nominal input, the head
# renderer's hover zone), and lint-allow.json has no sense-scoped entry for it yet, so in VOCAB it
# would fail the literal gate there.
# They join VOCAB once lint-allow.json holds test_lanekit_graph_generators.SENSE_ALLOW (a need of
# T04, its owner), unless the narrower VOCAB is ratified instead; that test checks the entries
# cover every line that uses them, and that no word is withheld that no template uses.
VOCAB_WITHHELD = frozenset({"zone"})

ENTITY_COMMENT = "One bus of a radial distribution grid, as the operator recorded it at peak."
COMPONENT = "distribution"


def RECIPE(raw_path: Path, shaped_dir: Path, lane: Any) -> dict:  # noqa: N802 (the protocol's name)
    """The standard recipe over this SCHEMA: the keys dropped (the edges read them raw), `zone`
    encoded, the numbers scaled, a stratified split on `overloaded`, and `graph_edges` key_pair
    on `[bus_key, feeder_to]`, symmetric, citing the relation finding (lanekit.recipes)."""
    return standard_recipe(SCHEMA, raw_path, shaped_dir, lane)


def _feeder_trees(rng, rows: int):
    """(zone index, parent row or -1, hops from the substation) for every bus, zone by zone:
    each zone's first bus is its substation; a later bus hangs off an earlier bus of its zone,
    a recent one more often than an old one."""
    import numpy as np

    zone = np.sort(np.arange(rows) % len(ZONES))
    parent = np.full(rows, -1, dtype=np.int64)
    depth = np.zeros(rows, dtype=np.int64)
    for z in range(len(ZONES)):
        members = np.flatnonzero(zone == z)
        for j in range(1, len(members)):
            back = 1 + int(rng.exponential(6.0))
            p = members[max(0, j - back)]
            parent[members[j]] = p
            depth[members[j]] = depth[p] + 1
    return zone, parent, depth


def _neighbourhood_sum(values, parent):
    """Σ over the bus, its parent and its children of `values` (the feeder neighbourhood)."""
    import numpy as np

    total = values.astype(np.float64).copy()
    has = parent >= 0
    np.add.at(total, np.flatnonzero(has), values[parent[has]])      # the bus feeding it
    np.add.at(total, parent[has], values[has])                       # the buses it feeds
    return total


def make(seed: int, rows: int):
    """`rows` buses (module docstring), deterministic in (seed, rows); the rows are shuffled so
    the raw file is not ordered by zone."""
    import numpy as np
    import pandas as pd

    if rows < 2 * len(ZONES):
        raise ValueError(f"grid needs at least {2 * len(ZONES)} buses (two a zone), got {rows}")
    rng = np.random.default_rng(seed)
    zone, parent, depth = _feeder_trees(rng, rows)
    substation = parent < 0
    load = np.where(substation, rng.uniform(0.2, 0.6, rows), rng.lognormal(np.log(2.0), 0.45, rows))
    gen = np.where(rng.random(rows) < 0.25, rng.uniform(0.3, 2.5, rows), 0.0)
    zone_rating = rng.uniform(0.85, 1.15, len(ZONES))[zone]
    cap = rng.choice(LINE_RATINGS_MW, rows, p=(0.25, 0.35, 0.25, 0.15)) * zone_rating
    cap = np.where(substation, 20.0 * zone_rating, cap)
    net = load - gen
    stress = _neighbourhood_sum(net, parent) / _neighbourhood_sum(cap, parent)
    noisy = stress + rng.normal(0.0, 0.02, rows)
    over = noisy > np.quantile(noisy, 1.0 - OVERLOADED_SHARE)
    voltage = 1.03 - 0.0015 * depth - 0.04 * (net / cap) + rng.normal(0.0, 0.006, rows)
    keys = np.array([f"B{i + 1:05d}" for i in range(rows)], dtype=object)
    feeder = np.where(substation, "", keys[np.maximum(parent, 0)])
    frame = pd.DataFrame({
        "bus_key": keys, "feeder_to": feeder, "load_mw": np.round(load, 4), "gen_mw": np.round(gen, 4),
        "voltage_pu": np.round(voltage, 5), "line_cap_mw": np.round(cap, 3),
        "zone": np.array(ZONES, dtype=object)[zone],
        "overloaded": np.where(over, CLASSES[1], CLASSES[0])})
    order = rng.permutation(rows)
    return frame.iloc[order].reset_index(drop=True)[[c.name for c in SCHEMA]]
