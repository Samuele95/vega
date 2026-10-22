#!/usr/bin/env python3
"""
test_roundtrip_plan.py — the pack-time round trip and the shipped examples per sample shape
(build plan S10; bundle-contract §5; R-A3, R-A5, R-A6).

    python3 -m pytest test_roundtrip_plan.py [-q] [-k "not packed"]
    python3 test_roundtrip_plan.py                     (without pytest)

Two halves.

**The logic, on frames built here** (fast): the plan of each shape — rows paired by `row_id`,
never by position; a window's history is each series' last `history_rows` rows of the previous
parts in `order_by` order, and its lag boundary the first `(length − 1) + lag_depth` test rows of
each series (none without a lag, none under `context: none`); node-level queries in chunks; graph
answers by `graph_id` — and the comparison over the record's `compared` / `decided` pairs, the
examples chosen per shape and the request each shape ships. The negatives' logic: N-S1 (a
row-score reading whose stored layout lacks `row_score`: exit 3, `per-row score`), N-S2 (a window
count off by one: exit 3, `windows`), N-S8 (a stored decision that disagrees with the threshold:
exit 3, `decided`, through `run` itself), a graph Recipe without `edges_for` (exit 3).

**The real thing** (`packed`): B15's three spike fixtures — a lagged multi-series
autoregressive forecast window, a node-level key-pair graph, a graph-level set — trained, then
evaluated by the REAL `vega-eval/scripts/evaluate.py` and packed by the REAL `pack.py`
(`roundtrip_fixtures.py`): every round trip passes (the window's R-A6 boundary is exactly three
series × 13 rows), every bundle's `predict.py --check` reproduces its shipped examples,
`request.json` has each shape's form; N-S8 on the real node bundle's campaign (a stored
`prediction` flipped: pack exits 3 naming the decided field), N-S2 on the real window's stored
predictions, and N-S9 (a `graph/reference-nodes.parquet` altered after packing: the bundle's
`predict.verify()` refuses it by digest, `predict.py --check` exits 1 naming it).
`VEGA_ROUNDTRIP_WORK=<dir>` keeps the packed fixtures there (default: a temporary directory).
"""
from __future__ import annotations

import atexit
import json
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
PACK = HERE.parent
for _p in (str(PACK), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import examples_pack as ep  # noqa: E402
import roundtrip_compare as rc  # noqa: E402
import roundtrip_fixtures as rf  # noqa: E402
import roundtrip_plan as rp  # noqa: E402
from pack_context import PackExit  # noqa: E402
from roundtrip_inputs import LAYOUT, RawRows, Stored  # noqa: E402

BINARY = {"schema": "vega/output@2", "reading": "binary", "head_out": 1,
          "fields": ["label", "p_no", "p_yes", "score", "prediction", "warnings"],
          "field_kinds": {"score": {"kind": "raw"}, "prediction": {"kind": "decision"}},
          "compared": [["score", "score"]], "decided": [["prediction", "prediction"]]}
FORECAST = {"schema": "vega/output@2", "reading": "forecast", "head_out": 2,
            "fields": ["value_h1", "value_h2", "answered", "warnings"],
            "field_kinds": {"answered": {"kind": "answered"}},
            "compared": [["value_h1", "score_0"], ["value_h2", "score_1"]], "decided": []}


# ============================================================ frames built here

def _refusal(fn, *args):
    with pytest.raises(PackExit) as caught:
        fn(*args)
    return caught.value


def _stored(frame, unscored=()):
    return Stored(pd.DataFrame(frame), {"vega.predictions_layout": LAYOUT},
                  {"count": len(unscored), "rows": [{"row_id": r, "reason": "x"}
                                                    for r in unscored]})


def _series_raw(seed=3):
    """Two series `a`, `b`, t = 0…19 each, stored in a shuffled order: 10 train, 4 valid and 6
    test rows per series by t; the split lists each part's positions in raw-file order."""
    rng = np.random.default_rng(seed)
    rows = [{"s": s, "t": t, "x": float(t) + (0.5 if s == "b" else 0.0), "y": float(t % 3)}
            for s in ("a", "b") for t in range(20)]
    frame = pd.DataFrame(rows).iloc[rng.permutation(40)].reset_index(drop=True)
    part = np.where(frame["t"] < 10, "train", np.where(frame["t"] < 14, "valid", "test"))
    split = {p: [int(i) for i in np.flatnonzero(part == p)] for p in ("train", "valid", "test")}
    return frame, {"__split_raw__": split}


WINDOW = {"shape": "window", "length": 3, "lag_depth": 1, "history_rows": 3, "series_by": "s",
          "order_by": "t", "context": "previous-part"}


def _window_plan(sample=WINDOW, drop=0):
    frame, fitted = _series_raw()
    test = fitted["__split_raw__"]["test"]
    stored = _stored({"row_id": test[drop:], "score_0": np.zeros(len(test) - drop),
                      "score_1": np.zeros(len(test) - drop)})
    return rp.plan_for(sample, stored, RawRows(frame), fitted, {"ignored": ["y"]}), frame


def test_a_window_history_is_each_series_tail_of_the_previous_parts():
    plan, frame = _window_plan()
    history = plan.history
    assert sorted(zip(history["s"], history["t"])) == \
        [("a", 11), ("a", 12), ("a", 13), ("b", 11), ("b", 12), ("b", 13)], history
    assert "y" not in plan.query.columns and "y" not in history.columns
    assert list(plan.positions) == rp.split_of(_series_raw()[1])["test"].tolist()


def test_the_lag_boundary_is_the_first_rows_of_each_series():
    plan, frame = _window_plan()
    t = frame["t"].to_numpy()[np.asarray(plan.stored_row_ids)]
    assert sorted(t[plan.excluded].tolist()) == [14, 14, 15, 15, 16, 16]
    assert plan.boundary == {"rows": 6, "reason": rp.BOUNDARY_REASON}
    plain, _ = _window_plan({**WINDOW, "lag_depth": 0, "history_rows": 2})
    assert plain.boundary is None and not plain.excluded.any()
    assert len(plain.history) == 4                      # length − 1 rows per series


def test_context_none_sends_no_history_and_draws_no_boundary():
    frame, fitted = _series_raw()
    test = fitted["__split_raw__"]["test"]
    t = frame["t"].to_numpy()
    scored = [p for p in test if t[p] >= 16]            # the first L − 1 rows go unscored
    stored = _stored({"row_id": scored, "score_0": np.zeros(len(scored)),
                      "score_1": np.zeros(len(scored))}, [p for p in test if t[p] < 16])
    plan = rp.plan_for({**WINDOW, "context": "none"}, stored, RawRows(frame), fitted, {})
    assert plan.history is None and plan.boundary is None and not plan.excluded.any()


def test_the_series_order_is_read_off_the_shaped_keys():
    """A key the Recipe reshapes (here: a text the raw file orders wrongly) orders by its shaped
    values, the evaluation's."""
    frame, fitted = _series_raw()
    frame["t"] = frame["t"].map(lambda v: f"day {v}")   # "day 9" > "day 13" as text
    split = rp.split_of(fitted)
    before = np.concatenate([split["train"], split["valid"]])
    keys = pd.DataFrame({"t": [int(v.split()[1]) for v in frame["t"]]})
    codes, rank = rp.series_ranks(WINDOW, RawRows(frame, keys), before)
    shaped = keys["t"].to_numpy()[before]
    assert all(rank[i] == shaped[i] for i in range(len(rank)))
    _codes, wrong = rp.series_ranks(WINDOW, RawRows(frame), before)
    assert not np.array_equal(wrong, rank)


def test_n_s2_a_window_count_off_by_one_is_refused():
    refusal = _refusal(_window_plan, WINDOW, 1)
    assert refusal.code == 3 and "windows" in refusal.lines[0], refusal.lines
    frame, fitted = _series_raw()
    test = fitted["__split_raw__"]["test"]
    stored = Stored(pd.DataFrame({"row_id": test, "score_0": 0.0, "score_1": 0.0}), {}, None)
    refusal = _refusal(rp.plan_for, WINDOW, stored, RawRows(frame), fitted, {})
    assert refusal.code == 3 and "unscored.json" in refusal.lines[0]


def test_rows_pair_by_row_id_and_a_lag_recipe_gets_the_empty_history():
    frame, fitted = _series_raw()
    test = fitted["__split_raw__"]["test"]
    shuffled = list(reversed(test))
    stored = _stored({"row_id": shuffled, "score": np.arange(len(test), dtype=float)})
    plain = rp.plan_for({"shape": "row"}, stored, RawRows(frame), fitted, {"ignored": ["y"]})
    assert plain.history is None and list(plain.stored_row_ids) == shuffled
    lagged = rp.plan_for({"shape": "row"}, stored, RawRows(frame, reads=("y",), lags=True),
                         fitted, {"ignored": ["y"]})
    assert lagged.history is not None and len(lagged.history) == 0
    assert "y" in lagged.query.columns and "y" in lagged.history.columns
    extra = _stored({"row_id": test + [fitted["__split_raw__"]["train"][0]], "score": 0.0})
    assert _refusal(rp.plan_for, {"shape": "row"}, extra, RawRows(frame), fitted, {}).code == 3


class _Stub:
    """A Predictor as the round trip reads one: `predict(rows, history)` records its calls and
    answers `score = x`, `prediction = score ≥ 0.5`; `reader.from_stored` reads the stored
    `prediction` as the decision (the binary reader's pair)."""

    def __init__(self, answered=None):
        self.calls, self.answered = [], answered
        self.reader = SimpleNamespace(from_stored=lambda stored: pd.DataFrame(
            {"prediction": stored["prediction"].to_numpy()}, index=stored.index))

    def predict(self, rows, history=None):
        self.calls.append((len(rows), None if history is None else len(history)))
        score = rows["x"].to_numpy(dtype=float)
        out = pd.DataFrame({"score": score, "prediction": (score >= 0.5).astype(np.int64)})
        if self.answered is not None:
            out["answered"] = self.answered(rows)
        return out

    def prepare(self, rows, history=None):
        return SimpleNamespace(shaped=rows)


def test_node_queries_go_in_chunks_and_every_answer_comes_back_in_order():
    frame, fitted = _series_raw()
    test = fitted["__split_raw__"]["test"]
    stored = _stored({"row_id": test, "score": frame["x"].to_numpy()[test]})
    plan = rp.plan_for({"shape": "graph", "level": "node"}, stored, RawRows(frame), fitted, {})
    assert plan.chunk == rp.CHUNK_ROWS and plan.history is None
    stub = _Stub()
    got = rc.answers(stub, replace(plan, chunk=5))
    assert [c[0] for c in stub.calls] == [5, 5, 2]
    assert np.allclose(got["score"], frame["x"].to_numpy()[test])


def test_graph_answers_pair_by_graph_id():
    frame = pd.DataFrame({"g": ["g1", "g1", "g2", "g2", "g2", "g3"], "x": [0.0, 0, 1, 1, 1, 2]})
    fitted = {"__split_raw__": {"train": [5], "test": [0, 1, 2, 3, 4]}}
    stored = _stored({"row_id": ["g2", "g1"], "graph_id": ["g2", "g1"], "score": [1.0, 0.0]})
    sample = {"shape": "graph", "level": "graph", "graph_id": "g"}
    plan = rp.plan_for(sample, stored, RawRows(frame), fitted, {})
    output = {"field_kinds": {"graph_id": {"kind": "group"}}}
    got = pd.DataFrame({"graph_id": ["g1", "g1", "g2", "g2", "g2"]})
    assert list(rc.pairing(plan, got, output)) == [2, 0]
    other = _stored({"row_id": ["g2", "g9"], "graph_id": ["g2", "g9"], "score": [1.0, 0.0]})
    assert _refusal(rp.plan_for, sample, other, RawRows(frame), fitted, {}).code == 3


def test_n_s1_and_the_graph_recipe_refusals(tmp_path):
    stored = _stored({"row_id": [0], "score_0": [0.1]})
    svdd = {"reading": "anomaly-svdd", "compared": [["score", "row_score"]]}
    refusal = _refusal(rp.refuse_before, svdd, {"shape": "row"}, stored, tmp_path / "p.py")
    assert refusal.code == 3 and "per-row score" in refusal.lines[0]
    missing = _refusal(rp.refuse_before, FORECAST, {"shape": "row"}, stored, tmp_path / "p.py")
    assert missing.code == 3 and "score_1" in missing.lines[0]
    recipe = tmp_path / "preprocess.py"
    recipe.write_text("REQUIRED_COLUMNS = ['x']\n\ndef transform(df, history=None):\n"
                      "    return df\n", encoding="utf-8")
    stored = _stored({"row_id": [0], "score": [0.1], "prediction": [0]})
    node = {"shape": "graph", "level": "node"}
    refusal = _refusal(rp.refuse_before, BINARY, node, stored, recipe)
    assert refusal.code == 3 and "edges_for" in refusal.lines[0]
    recipe.write_text(recipe.read_text() + "\ndef edges_for(df, reference, history=None):\n"
                      "    return df\n", encoding="utf-8")
    rp.refuse_before(BINARY, node, stored, recipe)
    level = _refusal(rp.refuse_before, BINARY, {"shape": "graph", "level": "graph"}, stored,
                     recipe)
    assert "edges_within" in level.lines[0]


def _evaluated(plan, stored, output):
    expected = rc.expected_frame(stored.frame, output, _Stub().reader)
    return rp.Evaluated(output, {}, stored, None, plan, expected)


def test_compared_fields_are_numeric_decided_fields_exact_unanswered_rows_fail():
    frame, fitted = _series_raw()
    test = fitted["__split_raw__"]["test"]
    x = frame["x"].to_numpy()[test]
    stored = _stored({"row_id": test, "score": x, "prediction": (x >= 0.5).astype(int)})
    plan = rp.plan_for({"shape": "row"}, stored, RawRows(frame), fitted, {})
    tol = rc.tolerance_for(stored.frame, BINARY)
    assert tol.value == pytest.approx(1e-6 * max(1.0, float(np.abs(x).max())))
    trip = rc.round_trip(_Stub(), _evaluated(plan, stored, BINARY), tol)
    assert trip.passed and trip.rows == len(test) and trip.decided == len(test)
    flipped = _stored({"row_id": test, "score": x, "prediction": 1 - (x >= 0.5).astype(int)})
    trip = rc.round_trip(_Stub(), _evaluated(plan, flipped, BINARY), tol)
    assert not trip.passed and trip.first_bad["kind"] == "decided"
    assert trip.first_bad["row_id"] == test[0]
    moved = _stored({"row_id": test, "score": x + 1e-3, "prediction": (x >= 0.5).astype(int)})
    trip = rc.round_trip(_Stub(), _evaluated(plan, moved, BINARY), tol)
    assert trip.first_bad["kind"] == "compared" and trip.first_bad["column"] == "score"
    silent = _Stub(answered=lambda rows: rows["x"].to_numpy() < 17)
    trip = rc.round_trip(silent, _evaluated(plan, stored, BINARY), tol)
    assert trip.first_bad["kind"] == "answered"


def test_the_boundary_rows_are_not_compared_and_are_recorded():
    frame, fitted = _series_raw()
    frame["x"] = frame["t"].astype(float)
    plan, _ = _window_plan()
    ids = np.asarray(plan.stored_row_ids)
    t = frame["t"].to_numpy()[ids].astype(float)
    t_bad = np.where(plan.excluded, t + 5.0, t)          # the boundary rows disagree
    output = {**FORECAST, "compared": [["score", "score_0"]], "decided": []}
    stored = _stored({"row_id": ids, "score_0": t_bad, "score_1": t})
    plan = replace(plan, query=frame.iloc[plan.positions].reset_index(drop=True))
    trip = rc.round_trip(_Stub(), _evaluated(plan, stored, output),
                         rc.tolerance_for(stored.frame, output))
    assert trip.passed, trip.first_bad
    assert trip.as_json()["boundary"] == {"rows": 6, "reason": rp.BOUNDARY_REASON}
    everything = replace(plan, excluded=np.zeros(len(ids), dtype=bool))
    trip = rc.round_trip(_Stub(), _evaluated(everything, stored, output),
                         rc.tolerance_for(stored.frame, output))
    assert not trip.passed


def _write_layout(frame, path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.Table.from_pandas(frame, preserve_index=False)
    meta = dict(table.schema.metadata or {})
    meta[b"vega.predictions_layout"] = LAYOUT.encode()
    pq.write_table(table.replace_schema_metadata(meta), path)


def test_n_s8_run_refuses_a_stored_decision_the_threshold_does_not_make(tmp_path):
    """`run` over a stored `prediction` that disagrees: exit 3, the decided field named."""
    frame, fitted = _series_raw()
    test = fitted["__split_raw__"]["test"]
    x = frame["x"].to_numpy()[test]
    (tmp_path / "eval").mkdir()
    _write_layout(pd.DataFrame({"row_id": test, "score": x,
                                "prediction": 1 - (x >= 0.5).astype(int)}),
                  tmp_path / "eval" / "predictions.test.parquet")
    (tmp_path / "pkg" / "features").mkdir(parents=True)
    (tmp_path / "pkg" / "features" / "preprocess.py").write_text("REQUIRED_COLUMNS = ['x']\n")
    manifest = tmp_path / "MANIFEST.json"
    manifest.write_text(json.dumps({"output": BINARY, "sample": {"shape": "row"}}))
    ctx = SimpleNamespace(
        paths=SimpleNamespace(manifest=manifest, pkg=tmp_path / "pkg"),
        campaign=SimpleNamespace(eval_dir=tmp_path / "eval", metrics={}, raw=frame,
                                 fitted=fitted, recipe={"steps": []}),
        schema={"ignored": ["y"]}, markers=[])
    refusal = _refusal(rp.run, ctx, _Stub())
    assert refusal.code == 3 and "decided" in refusal.lines[0], refusal.lines
    assert any("decided field prediction" in m for m in ctx.markers), ctx.markers


def test_window_examples_ship_context_first_and_compare_inside_the_test_part():
    plan, frame = _window_plan()
    ids = np.asarray(plan.stored_row_ids)
    stored = _stored({"row_id": ids, "score_0": frame["x"].to_numpy()[ids],
                      "score_1": frame["x"].to_numpy()[ids]})
    ev = rp.Evaluated(FORECAST, WINDOW, stored, RawRows(frame), plan,
                      rc.expected_frame(stored.frame, FORECAST, None))
    chosen = ep._TEST["window"](ev, 5)
    assert chosen.context_rows == 2
    rows, at = chosen.rows, chosen.expected[ep.ROW].tolist()
    first = plan.query["s"].iloc[0]                     # the series the test part opens with
    assert list(rows["s"][:6]) == [first] * 6 and list(rows["t"][:6]) == list(range(14, 20))
    assert at[:4] == [2, 3, 4, 5] and len(at) == 5 and len(rows) == 9
    assert rows["s"].iloc[at[4]] != first and rows["t"].iloc[at[4]] == 16
    assert np.allclose(chosen.expected["value_h1"], rows["x"].to_numpy()[at])
    query, history = ep._request_rows(rows, chosen.expected, WINDOW)
    assert list(query["t"]) == [16] and list(history["t"]) == [14, 15]


def test_graph_examples_ship_whole_graphs_answered_at_their_first_row():
    frame = pd.DataFrame({"g": ["g1", "g1", "g2", "g2", "g2", "g3"], "x": [0.0, 0, 1, 1, 1, 2]})
    fitted = {"__split_raw__": {"train": [], "test": [0, 1, 2, 3, 4, 5]}}
    stored = _stored({"row_id": ["g1", "g2", "g3"], "graph_id": ["g1", "g2", "g3"],
                      "score": [0.0, 1.0, 2.0]})
    sample = {"shape": "graph", "level": "graph", "graph_id": "g"}
    plan = rp.plan_for(sample, stored, RawRows(frame), fitted, {})
    output = {"compared": [["score", "score"]], "decided": []}
    ev = rp.Evaluated(output, sample, stored, RawRows(frame), plan,
                      rc.expected_frame(stored.frame, output, None))
    chosen = ep._TEST["graph"](ev, 3)
    assert len(chosen.rows) == 5 and chosen.expected[ep.ROW].tolist() == [0, 2]
    assert chosen.expected["graph_id"].tolist() == ["g1", "g2"]
    query, history = ep._request_rows(chosen.rows, chosen.expected, sample)
    assert list(query["g"]) == ["g1", "g1"] and history is None


def test_synthetic_examples_are_the_predictors_own_answers_per_shape():
    frame, fitted = _series_raw()
    ctx = SimpleNamespace(campaign=SimpleNamespace(fitted=fitted), schema={"ignored": ["y"]})
    test = fitted["__split_raw__"]["test"]
    stored = _stored({"row_id": test, "score": 0.0, "prediction": 0})
    plan = rp.plan_for({"shape": "row"}, stored, RawRows(frame), fitted, ctx.schema)
    ev = rp.Evaluated(BINARY, {"shape": "row"}, stored, RawRows(frame), plan, None)
    rows = ep._SYNTHETIC["row"](ev, ctx, _Stub(), 5)
    assert len(rows.rows) == 5 and rows.expected[ep.ROW].tolist() == list(range(5))
    assert np.allclose(rows.expected["score"], rows.rows["x"])
    assert "y" not in rows.rows.columns and rows.context_rows is None
    plan, _ = _window_plan()
    ev = rp.Evaluated(FORECAST, WINDOW, None, RawRows(frame), plan, None)
    stub = _Stub(answered=lambda r: np.arange(len(r)) >= 2)
    stub.predict = _with_values(stub.predict)
    run = ep._SYNTHETIC["window"](ev, ctx, stub, 4)
    assert run.context_rows == 2 and run.expected[ep.ROW].tolist() == [2, 3, 4, 5]
    assert run.rows["s"].nunique() == 1 and list(run.rows["t"]) == list(range(4, 10))
    graphs = pd.DataFrame({"g": ["g1", "g1", "g2", "g2", "g2", "g3"], "x": [0.0, 0, 1, 1, 1, 2]})
    split = {"__split_raw__": {"train": [0, 1, 2, 3, 4, 5], "test": [5]}}
    sample = {"shape": "graph", "level": "graph", "graph_id": "g"}
    stored = _stored({"row_id": ["g3"], "graph_id": ["g3"], "score": [2.0], "prediction": [1]})
    plan = rp.plan_for(sample, stored, RawRows(graphs), split, {})
    ev = rp.Evaluated(BINARY, sample, stored, RawRows(graphs), plan, None)
    ctx = SimpleNamespace(campaign=SimpleNamespace(fitted=split), schema={})
    whole = ep._SYNTHETIC["graph"](ev, ctx, _Stub(), 3)
    assert len(whole.rows) == 5 and whole.expected[ep.ROW].tolist() == [0, 2]
    assert whole.expected["graph_id"].tolist() == ["g1", "g2"]


def _with_values(predict):
    """A forecast Predictor stub: the two horizon fields beside the stub's own."""
    def answer(rows, history=None):
        out = predict(rows, history)
        return out.assign(value_h1=out["score"], value_h2=out["score"] + 1.0)
    return answer


# ============================================================ the real thing

_WORK: list = []
_PACKED: dict = {}


def _work() -> Path:
    if not _WORK:
        named = os.environ.get("VEGA_ROUNDTRIP_WORK")
        if named:
            Path(named).mkdir(parents=True, exist_ok=True)
            _WORK.append(Path(named))
        else:
            made = Path(tempfile.mkdtemp(prefix="vega-roundtrip-"))
            atexit.register(shutil.rmtree, made, True)
            _WORK.append(made)
    return _WORK[0]


def _flip_one_prediction(result) -> rf.Packed:
    """N-S8 on the real node bundle's campaign: a copy whose first stored `prediction` is
    flipped, packed again."""
    import pyarrow.parquet as pq
    camp = Path(result.campaign)
    seeded = camp.parent / "campaign-ns8"
    shutil.rmtree(seeded, ignore_errors=True)
    shutil.copytree(camp, seeded)
    path = seeded / "eval" / rf.RUN / "predictions.test.parquet"
    table = pq.read_table(path)
    frame = table.to_pandas()
    frame.loc[0, "prediction"] = 1 - int(frame.loc[0, "prediction"])
    _write_layout(frame, path)
    out = rf.pack(seeded, result, camp.parent / "bundle-ns8")
    return replace(result, campaign=str(seeded), pack_exit=out.returncode,
                   pack_output=out.stdout + out.stderr, bundle=None)


def _job(name):
    """One fixture through the harness in a child process (the spike's emitter runs in the
    process that makes a fixture), then N-S8's seeded re-pack of the node campaign."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    proc = subprocess.run([sys.executable, rf.__file__, str(_work()), "--fixture", name,
                           "--json"], capture_output=True, text=True, timeout=1800, env=env)
    try:
        result = rf.Packed.from_dict(json.loads(proc.stdout)[0])
    except (json.JSONDecodeError, IndexError) as exc:
        raise AssertionError(f"{name}: the harness failed (exit {proc.returncode})\n"
                             f"{proc.stderr[-3000:]}") from exc
    extra = {"node-keypair": _flip_one_prediction, "window": _synthetic}.get(name)
    return name, result, extra(result) if extra else None


def _synthetic(result) -> rf.Packed:
    """The window campaign packed again with `--examples synthetic`."""
    camp = Path(result.campaign)
    out = rf.pack(camp, result, camp.parent / "bundle-synthetic", ("--examples", "synthetic"))
    bundles = sorted((camp.parent / "bundle-synthetic").glob("*/MANIFEST.json"))
    return replace(result, pack_exit=out.returncode, pack_output=out.stdout + out.stderr,
                   bundle=str(bundles[0].parent) if bundles else None)


def packed():
    """The three fixtures made, trained, evaluated and packed (in parallel), and N-S8's seeded
    re-pack of the node campaign."""
    if not _PACKED:
        with ThreadPoolExecutor(max_workers=3) as pool:
            for name, result, extra in pool.map(_job, rf.CLOSING):
                _PACKED[name] = result
                if extra is not None:
                    _PACKED[{"node-keypair": "ns8", "window": "synthetic"}[name]] = extra
    return _PACKED


def _check(bundle) -> subprocess.CompletedProcess:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run([sys.executable, "predict.py", "--check"], cwd=bundle,
                          capture_output=True, text=True, timeout=600, env=env)


def test_packed_the_round_trip_passes_on_the_spike_fixtures():
    for name in rf.CLOSING:
        result = packed()[name]
        assert result.eval_exit == 0, result.eval_output[-1500:]
        assert result.pack_exit == 0 and result.bundle, result.pack_output[-2500:]
        trip = result.manifest["roundtrip"]
        stored = pd.read_parquet(Path(result.campaign) / "eval" / rf.RUN /
                                 "predictions.test.parquet")
        assert trip["status"] == "pass" and trip["rows"] == len(stored), (name, trip)
        assert trip["shaper_transform_check"] == "pass"
        assert trip["max_abs_diff"] <= trip["tolerance"]["value"]
        sample = result.manifest["sample"]
        if name == "window":
            per_series = sample["length"] - 1 + sample["lag_depth"]
            assert trip["boundary"] == {"rows": 3 * per_series, "reason": rp.BOUNDARY_REASON}
        else:
            assert "boundary" not in trip, trip
        output = result.manifest["output"]
        assert trip["columns_compared"] == len(output["compared"])
        assert trip["decisions_compared"] == (len(stored) if output["decided"] else 0)


def test_packed_bundles_reproduce_their_examples_and_ship_the_shapes_request():
    for name in rf.CLOSING:
        result = packed()[name]
        root = Path(result.bundle)
        check = _check(root)
        assert check.returncode == 0 and "2 checks passed" in check.stdout, check.stdout
        expected = pd.read_parquet(root / "examples" / "expected.parquet")
        examples = result.manifest["examples"]
        assert examples["kind"] == "test" and expected.columns[0] == "__row__"
        assert examples["compared_rows"] == len(expected)
        request = json.loads((root / "examples" / "request.json").read_text())
        sample = result.manifest["sample"]
        if name == "window":
            assert examples["context_rows"] == sample["length"] - 1
            assert len(request["rows"]) == 1 and len(request["history"]) >= sample["length"] - 1
        elif name == "graph-level":
            ids = {row[sample["graph_id"]] for row in request["rows"]}
            assert len(ids) == 1 and len(request["rows"]) > 1 and "history" not in request
        else:
            assert len(request["rows"]) == 1 and "history" not in request
            assert {"team_id"} <= set(request["rows"][0]), request   # edges_for reads it


def test_packed_synthetic_window_examples_reproduce_the_bundle_itself():
    result = packed()["synthetic"]
    assert result.pack_exit == 0 and result.bundle, result.pack_output[-2000:]
    examples = result.manifest["examples"]
    assert examples["kind"] == "synthetic" and examples["source"] == "predictor"
    assert examples["context_rows"] == result.manifest["sample"]["length"] - 1
    check = _check(Path(result.bundle))
    assert check.returncode == 0 and "2 checks passed" in check.stdout, check.stdout


def test_packed_the_layer_and_the_smoke_read_each_bundle_by_its_reading():
    """apply_bundle_layer's `validate_reading` and smoke_bundle's `reading_patterns` over the
    real manifests: nothing to refuse, the patterns the schema describes; then each thing the
    layer refuses, seeded one at a time on the window bundle's records."""
    import apply_bundle_layer as abl
    import smoke_bundle as sb
    for name in rf.CLOSING:
        manifest = packed()[name].manifest
        schema = json.loads((Path(packed()[name].bundle) / "schema.json").read_text())
        assert abl.validate_reading(manifest, schema, manifest["roundtrip"]) == [], name
        patterns, fields = sb.reading_patterns(manifest)
        assert patterns == sorted(schema["output"]) and fields == manifest["output"]["fields"]
    window = packed()["window"]
    manifest, trip = window.manifest, window.manifest["roundtrip"]
    schema = json.loads((Path(window.bundle) / "schema.json").read_text())
    unbounded = {k: v for k, v in trip.items() if k != "boundary"}
    assert any("no boundary" in e for e in abl.validate_reading(manifest, schema, unbounded))
    assert any("compared 1 column" in e for e in abl.validate_reading(
        manifest, schema, {**trip, "columns_compared": 1}))
    thin = {**schema, "output": {k: v for k, v in schema["output"].items()
                                 if k != "value_h<h>"}}
    assert any("output patterns" in e for e in abl.validate_reading(manifest, thin, trip))
    old = {**manifest, "output": {**manifest["output"], "schema": "vega/output@1"}}
    assert any("re-pack" in e for e in abl.validate_reading(old, schema, trip))
    bare = {**manifest, "examples": {k: v for k, v in manifest["examples"].items()
                                     if k != "context_rows"}}
    assert any("context_rows" in e for e in abl.validate_reading(bare, schema, trip))
    assert sb.reading_patterns(old) == (None, manifest["output"]["fields"])


def test_packed_n_s8_a_flipped_stored_decision_stops_the_pack():
    result = packed()["ns8"]
    assert result.pack_exit == 3, result.pack_output[-2000:]
    assert "decided" in result.pack_output and "does not ship" in result.pack_output


def test_packed_n_s2_the_real_window_count_off_by_one_is_refused():
    result = packed()["window"]
    camp = Path(result.campaign)
    stored = Stored.read(camp / "eval" / rf.RUN, {})
    fewer = replace(stored, frame=stored.frame.iloc[1:].reset_index(drop=True))
    manifest = result.manifest
    fitted = json.loads((Path(result.shaped) / "fitted.json").read_text())
    raw = RawRows(pd.read_csv(Path(result.shaped).parent / "raw.csv"))
    rp.plan_for(manifest["sample"], stored, raw, fitted, {})
    refusal = _refusal(rp.plan_for, manifest["sample"], fewer, raw, fitted, {})
    assert refusal.code == 3 and "windows" in refusal.lines[0], refusal.lines


def test_packed_n_s9_an_altered_reference_graph_is_refused_by_digest(tmp_path):
    result = packed()["node-keypair"]
    loose = tmp_path / Path(result.bundle).name
    shutil.copytree(result.bundle, loose)
    nodes = loose / "graph" / "reference-nodes.parquet"
    frame = pd.read_parquet(nodes)
    column = [c for c in frame.columns if c != "ref"][0]
    frame.loc[0, column] = frame.loc[0, column] + 1.0
    frame.to_parquet(nodes, index=False)
    check = _check(loose)
    assert check.returncode == 1, check.stdout
    assert "graph/reference-nodes.parquet is not the file MANIFEST.json pinned" in check.stdout


def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            if "tmp_path" in fn.__code__.co_varnames[:fn.__code__.co_argcount]:
                fn(Path(tempfile.mkdtemp(prefix="rt-")))
            else:
                fn()
            print(f"  ok    {name}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {name} — {exc}")
    print(f"\n{len(tests) - failed} of {len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
