"""
roundtrip_plan — the pack-time round trip per sample shape (bundle-contract §5; build plan S10,
R-A3, R-A5, R-A6): the bundle's own predict path over the raw rows behind every stored test
prediction, against what /vega-eval stored.

    run(ctx, predictor) -> {"roundtrip": …}                    PackExit(3) when it fails
    evaluated(ctx, predictor) -> Evaluated                     what the round trip and the
                                                               examples read, planned once
    plan_for(sample, stored, raw, fitted, schema) -> RoundTripPlan
    test_rows(ctx) -> (raw test rows, stored predictions)      (export_pack's rows)

This is the definition of "the same model": raw rows -> the Recipe -> the network here equals
shaped rows -> the network there. It exposes nothing — the outputs are compared and discarded,
no metric is computed, the seal is not touched. The stored predictions are read through their
writer's own reader (`vega-eval/scripts/predictions_layout.py`, `vega/predictions-layout@1`), and
every stored prediction is paired with the query row answering it by `row_id` (the raw
position of the row, the window's anchor or the node), or by `graph_id` at graph level — never
by position, never re-mapped.

**Per shape** (`plan_for`, one planner per shape):

  row     the test rows; a lag recipe gets the empty history window the shaper's twin check
          runs with (the evaluation's parts were shaped with their lags restarting).
  window  the test rows as the query; per series, the last `history_rows` rows of the
          previous parts (train then valid, in `order_by` order, as the evaluation borrowed its
          context) as the history, under `context: previous-part`; none under `context: none`.
          The stored count is `len(test) − |unscored|` (`unscored.json`). With a lag, the first
          `(length − 1) + lag_depth` test rows of each series are R-A6's boundary: excluded from
          the comparison and recorded in `boundary`.
  node    the whole test part as queries against the reference graph, in chunks — legal
          because an answer never depends on the other rows of the request (R-A3).
  graph   the test rows; each stored graph is compared with the answer of its `graph_id`.

A series and its order are read the way the evaluation read them: off the SHAPED key columns
(`RawRows.keys`: the Recipe's own output for the key columns, e.g. a parsed timestamp), falling
back to the raw column where the Recipe returns none. The comparison is `roundtrip_compare`'s.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from bundle_slice import transformer_file
from pack_context import PackExit, out_ok
from roundtrip_compare import (compared_sources, expected_frame, explain_failure, round_trip,
                               tolerance_for)
from roundtrip_inputs import (LAYOUT, RawRows, Stored, id_text, raw_rows, recipe_facts,
                              series_keys, series_ranks, shape_of, split_of)

__all__ = ["BOUNDARY_REASON", "CHUNK_ROWS", "LAYOUT", "Evaluated", "RawRows", "RoundTripPlan",
           "Stored", "evaluated", "id_text", "plan_for", "raw_rows", "refuse_before", "run",
           "series_keys", "series_ranks", "shape_of", "split_of", "test_rows"]

BOUNDARY_REASON = ("lag restarts at the part start in the evaluation; served answers read the "
                   "history")
# node-level queries per predict call: an answer never depends on the other rows sent, so the
# chunk only bounds the query side of each assembled graph (the reference is whole each time)
CHUNK_ROWS = 4096
ROW_SCORED = frozenset({"reconstruction", "anomaly-reconstruction", "anomaly-svdd"})
EDGE_FUNCTIONS = {"node": "edges_for", "graph": "edges_within"}
NO_ROW_SCORE = ("the evaluation stored no per-row score for a {reading} head; re-evaluate with "
                "vega-eval v2.3")


@dataclass(frozen=True, eq=False)
class RoundTripPlan:
    """What is sent and how its answers pair with the stored predictions.

    `query` / `history` are the raw rows sent (ignored columns dropped); `positions[j]` the raw
    position of query row j; `stored_row_ids[i]` the raw position of the query row answering
    stored prediction i (None at graph level, where `graph_ids[i]` is the stored graph's id as
    text and `group_column` the raw column naming each row's graph); `excluded[i]` R-A6's
    boundary, recorded in `boundary`; `chunk` the rows per predict call; `series` / `rank` a
    window's series code and in-series rank per query row."""

    shape: str
    query: Any
    history: Any
    positions: Any
    stored_row_ids: Any
    graph_ids: Any
    excluded: Any
    boundary: Mapping | None
    chunk: int
    group_column: str | None = None
    series: Any = None
    rank: Any = None


@dataclass(frozen=True, eq=False)
class Evaluated:
    """One pack run's view of the evaluation: the output and sample records, the stored
    predictions, the raw rows, the plan, and the expected answer per stored prediction."""

    output: Mapping
    sample: Mapping
    stored: Stored
    raw: RawRows
    plan: RoundTripPlan
    expected: Any


# --------------------------------------------------------------------------- the phase

def run(ctx, predictor) -> dict:
    """The round trip over every stored test prediction, or `PackExit(3)` naming the first
    disagreement."""
    ev = evaluated(ctx, predictor)
    tolerance = tolerance_for(ev.stored.frame, ev.output)
    trip = round_trip(predictor, ev, tolerance)
    if not trip.passed:
        explain_failure(trip, ctx.campaign.recipe, ctx.markers)
        decided = (trip.first_bad or {}).get("kind") == "decided"
        raise PackExit(3, "the bundle does not reproduce the evaluation's predictions"
                          + (" — a decided field disagrees with the evaluation's decision"
                             if decided else "") + "; it does not ship", keep_markers=True)
    boundary = ev.plan.boundary or {}
    out_ok(f"round trip ({ev.plan.shape}): {trip.rows} stored prediction(s) answered from raw "
           f"rows without their label"
           + (f", {boundary['rows']} at R-A6's lag boundary not compared" if boundary else "")
           + f"; {trip.columns} compared field(s) each, worst |Δ| {trip.max_abs_diff:.3g} "
           f"against {trip.tolerance.value:.3g} ({trip.tolerance.rule})"
           + (f"; {trip.decided} decision(s) equal to the evaluation's" if trip.decided
              else ""), ctx.markers)
    return {"roundtrip": trip.as_json()}


def evaluated(ctx, predictor) -> Evaluated:
    """The stub manifest's records, the stored predictions (refused before anything runs when
    they cannot answer this reading), the raw rows and the plan."""
    stub = json.loads(ctx.paths.manifest.read_text(encoding="utf-8"))
    output, sample = stub.get("output") or {}, stub.get("sample") or {"shape": "row"}
    stored = Stored.read(ctx.campaign.eval_dir, ctx.campaign.metrics)
    recipe_file = transformer_file(ctx.paths.pkg)
    refuse_before(output, sample, stored, recipe_file)
    raw = raw_rows(ctx.campaign.raw, recipe_file, sample, split_of(ctx.campaign.fitted),
                   predictor)
    plan = plan_for(sample, stored, raw, ctx.campaign.fitted, ctx.schema)
    expected = expected_frame(stored.frame, output, predictor.reader)
    return Evaluated(output, sample, stored, raw, plan, expected)


def test_rows(ctx):
    """(the raw rows of the test part in its order, the stored predictions frame)."""
    split = split_of(ctx.campaign.fitted)
    stored = Stored.read(ctx.campaign.eval_dir, ctx.campaign.metrics)
    return ctx.campaign.raw.iloc[list(split["test"])].reset_index(drop=True), stored.frame


# --------------------------------------------------------------------------- refusals

def refuse_before(output, sample, stored, recipe_file) -> None:
    """The refusals before the round trip (exit 3): a row-score reading whose stored layout
    lacks `row_score` (N-S1), a stored layout without a column the record compares, a graph
    bundle whose emitted Recipe lacks `edges_for` (node) or `edges_within` (graph level)."""
    reading = output.get("reading")
    if reading in ROW_SCORED and "row_score" not in stored.frame.columns:
        raise PackExit(3, NO_ROW_SCORE.format(reading=reading))
    missing = [c for c in compared_sources(output) if c not in stored.frame.columns]
    if missing:
        raise PackExit(3, f"the stored predictions carry no {', '.join(missing)}, which "
                          f"MANIFEST.output compares; vega/predictions-layout@1 stores the raw "
                          "head, every head_out column — re-evaluate with vega-eval v2.3")
    need = EDGE_FUNCTIONS.get(shape_of(sample))
    if need and need not in recipe_facts(recipe_file)["functions"]:
        raise PackExit(3, f"the bundle's {Path(recipe_file).name} has no {need}(); a graph "
                          f"bundle attaches its queries through it (R-A3) — re-emit the "
                          "transformer with dataset-shaper v2.3's --transform-only")


# --------------------------------------------------------------------------- the plans

def plan_for(sample, stored, raw, fitted, schema) -> RoundTripPlan:
    """The plan of `sample`'s shape (MANIFEST.sample), or `PackExit(3)` when the stored
    predictions and the test part do not account for each other."""
    return _PLANNERS[shape_of(sample)](sample, stored, raw, split_of(fitted), schema)


def _row_plan(sample, stored, raw, split, schema) -> RoundTripPlan:
    test = split["test"]
    ids = _stored_positions(stored, test, "row", None)
    history = raw.rows(test[:0], schema) if raw.lags else None
    return _plan("row", raw.rows(test, schema), history, test, ids)


def _node_plan(sample, stored, raw, split, schema) -> RoundTripPlan:
    test = split["test"]
    ids = _stored_positions(stored, test, "node", None)
    plan = _plan("node", raw.rows(test, schema), None, test, ids)
    return _replace(plan, chunk=CHUNK_ROWS)


def _graph_plan(sample, stored, raw, split, schema) -> RoundTripPlan:
    import numpy as np
    test, column = split["test"], sample.get("graph_id")
    if "graph_id" not in stored.frame.columns and "row_id" not in stored.frame.columns:
        raise PackExit(3, "the stored predictions of a graph-level evaluation carry no graph_id "
                          f"column; {LAYOUT} stores one row per graph with its id")
    ids = stored.frame["graph_id" if "graph_id" in stored.frame.columns else "row_id"]
    graphs = [id_text(g) for g in ids.to_numpy()]
    test_graphs = {id_text(g) for g in raw.key(column, test)} if column else set()
    if len(set(graphs)) != len(graphs) or set(graphs) != test_graphs:
        raise PackExit(3, f"the evaluation stored {len(graphs)} graph(s) and the test part's "
                          f"{column or 'graph_id'} column names {len(test_graphs)}; these are "
                          "not the same graphs")
    plan = _plan("graph", raw.rows(test, schema), None, test, None)
    return _replace(plan, graph_ids=np.asarray(graphs, dtype=object), group_column=column,
                    excluded=np.zeros(len(graphs), dtype=bool))


def _window_plan(sample, stored, raw, split, schema) -> RoundTripPlan:
    import numpy as np
    test = split["test"]
    ids = _stored_positions(stored, test, "window", _unscored(stored))
    series, rank = series_ranks(sample, raw, test)
    borrowed = (sample.get("context") or "previous-part") == "previous-part"
    history = _history(sample, raw, split, set(series_keys(sample, raw, test))) \
        if borrowed else None
    depth = int(sample.get("lag_depth") or 0)
    plan = _plan("window", raw.rows(test, schema), None if history is None else
                 raw.rows(history, schema), test, ids)
    boundary = None
    if borrowed and depth:
        edge = int(sample.get("length") or 1) - 1 + depth
        inside = set(np.asarray(test)[rank < edge].tolist())
        excluded = np.asarray([int(r) in inside for r in ids], dtype=bool)
        boundary = {"rows": int(excluded.sum()), "reason": BOUNDARY_REASON}
        plan = _replace(plan, excluded=excluded)
    return _replace(plan, boundary=boundary, series=series, rank=rank)


def _plan(shape, query, history, positions, ids) -> RoundTripPlan:
    import numpy as np
    n = 0 if ids is None else len(ids)
    return RoundTripPlan(shape=shape, query=query, history=history,
                         positions=np.asarray(positions, dtype=np.int64), stored_row_ids=ids,
                         graph_ids=None, excluded=np.zeros(n, dtype=bool), boundary=None,
                         chunk=max(1, len(query)))


def _replace(plan, **changes) -> RoundTripPlan:
    from dataclasses import replace
    return replace(plan, **changes)


_PLANNERS = {"row": _row_plan, "window": _window_plan, "node": _node_plan,
             "graph": _graph_plan}


# --------------------------------------------------------------------------- the windows

def _history(sample, raw, split, wanted):
    """R-A6's history: per series of the test part, the last `history_rows` rows of the
    previous parts (train then valid, as the evaluation concatenated them), in their order."""
    import numpy as np
    before = np.concatenate([split.get("train", np.zeros(0, np.int64)),
                             split.get("valid", np.zeros(0, np.int64))]).astype(np.int64)
    need = int(sample.get("history_rows") or int(sample.get("length") or 1) - 1)
    codes, rank = series_ranks(sample, raw, before)
    keys = series_keys(sample, raw, before)
    size = np.bincount(codes, minlength=int(codes.max()) + 1 if len(codes) else 0)
    tail = rank >= (size[codes] - need) if len(codes) else np.zeros(0, dtype=bool)
    keep = tail & np.asarray([k in wanted for k in keys], dtype=bool)
    return before[keep]


# --------------------------------------------------------------------------- the accounting

def _stored_positions(stored, test, what, unscored):
    """The stored row ids as raw positions, each a test row, every test row either stored or
    (windows) unscored — else `PackExit(3)` saying which count is off."""
    import numpy as np
    ids = np.asarray(stored.row_ids).astype(np.int64)
    test_set, stored_set = set(np.asarray(test).tolist()), set(ids.tolist())
    left = set() if unscored is None else set(unscored)
    if what == "window" and len(ids) != len(test) - len(left):
        raise PackExit(3, f"the evaluation stored {len(ids)} windows; the test part's "
                          f"{len(test)} rows less the {len(left)} unscored.json lists give "
                          f"{len(test) - len(left)}")
    if len(ids) != len(stored_set) or not stored_set <= test_set or stored_set & left \
            or stored_set | left != test_set:
        besides = f", with the {len(left)} unscored.json lists," if left else ""
        raise PackExit(3, f"the evaluation stored {len(ids)} {what} prediction(s) whose row_id"
                          f"{besides} do not account for the {len(test)} rows of the test part "
                          "(__split_raw__['test']); these are not the same part")
    return ids


def _unscored(stored):
    if stored.unscored is None:
        raise PackExit(3, "the evaluation wrote no unscored.json; the windows it left unscored "
                          "cannot be accounted for — re-evaluate with vega-eval v2.3")
    return [int(r["row_id"]) for r in stored.unscored.get("rows") or ()]
