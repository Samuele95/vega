#!/usr/bin/env python3
"""
test_predictor.py — the one Predictor (`templates/serving/package/predict.py`) and the bundle CLI
(`templates/serving/bundle/predict.py`), build plan S05 (D2 §4.6, §9; R-A2, R-A5, R-D15, R-D16).

    python3 -m pytest test_predictor.py [-q]          (or: python3 test_predictor.py)

Everything runs on *rendered* packages (`predictor_fixtures.py`: the Model side and the serving
package templates placed by `render_templates.py`) over the shaper's *real* outputs, and on
bundle directories laid out as pack lays them out, which the rendered CLI loads.

What is proved:

  readings  every row reading (all nine) answers the test part's raw rows with exactly
            `MANIFEST.output.fields`, in order, one row per row with the caller's index, and
            equals what the evaluation stored for those rows (`vega/predictions-layout@1`
            computed from the *shaped* test part, read by `reader.from_stored`): the compared
            fields within the round-trip tolerance, the decided ones exactly — the raw rows
            through the Recipe and the batch are the evaluation's inputs. `predictions()`
            carries the reading's own decision through the `LabelSet` (never `Prediction.of`'s
            0.5), `predict_one` is the frame's row, a row's answer does not depend on the other
            rows sent, the warnings are the guard's, then the shape's, then the reading's.
  shapes    windows: a row is answered iff its series has `length − 1` rows before it (history
            included), an unanswered row keeps its fields empty with `answered` false and the
            warning, an answer does not depend on how rows are split between history and
            request, `predict_one` takes history (R-D16); graphs: a node-level answer is the
            sealed test view's (the network gathers `target_index` once, the Predictor never:
            a model that returns every node is refused, R-A2), graph answers do not depend on
            the batch, rows of one graph share one answer under their `graph_id`, history is
            refused.
  refusals  the records value is required; a spec with no features, an svdd model without its
            centre, a model of another spec, a model that is not a staged network (its
            anatomy is None), a graph without its reference graph, a model in training mode
            (at construction and at a request: refused, never switched).
  the port  every Predictor here is handed its network behind `port_fixtures.PortOnly`: the
            Model port's methods and nothing else, so every test is also the check that the
            Predictor reaches the network through the port alone (`head`, `anatomy()`).
  module    C14 on `package/predict.py` (only `Predictor` public; a public function beside it is
            a finding), C11–C13 on both templates, the support-only imports (C5), no foreign
            literal (lint_templates.py).
  CLI       `load` builds the Predictor from a bundle (the reference graph through
            `load_context`), `check_examples` passes on a `__row__` set and on a positional
            (v2.2-shaped) set and names each disagreement (a compared value, a decided one, an
            unanswered row, a missing field), `verify` refuses an altered file (a tampered
            reference graph is N-S9's refusal), `load_monitor` builds the Monitor,
            `--check` / `--input` / `--one [--history]` from a clean shell.

Exit 0 when every test passes.
"""
from __future__ import annotations

import ast
import importlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import predictor_fixtures as pf  # noqa: E402
import conformance  # noqa: E402
from vega_pack_core import Tolerance  # noqa: E402

LINT = pf.SKILLS / "vega" / "scripts" / "tests" / "lint_templates.py"
PREDICT_TEMPLATE = pf.PACKAGE_TEMPLATES / "predict.py"
CLI_TEMPLATE = pf.SERVING_TEMPLATES / "bundle" / "predict.py"
ROW_READINGS = tuple(pf.ROW_READINGS)
SHAPE_FIELDS = {"group", "answered", "warnings"}


# --------------------------------------------------------------------------- helpers

@pytest.fixture(scope="module", autouse=True)
def _own_fitted_variable():
    """The bundles' CLIs point VEGA_FITTED at their parameters (the variable a Recipe module
    reads at import): it is given back as it was, so a test module run after this one in the
    same process imports its own Recipe modules against their own parameters."""
    saved = os.environ.get("VEGA_FITTED")
    yield
    if saved is None:
        os.environ.pop("VEGA_FITTED", None)
    else:
        os.environ["VEGA_FITTED"] = saved


def _pkg():
    return pf.package()


def _tolerance(case) -> float:
    return Tolerance.for_scores(pf._scores(case)).value


def _numeric(values) -> bool:
    return pd.api.types.is_numeric_dtype(pd.Series(values).infer_objects())


def _agree(got: pd.DataFrame, want: pd.DataFrame, tol: float, where: str) -> None:
    """Every column of `want` in `got`: numbers within `tol` (NaN only against NaN), names
    exactly (None only against None)."""
    assert len(got) == len(want), (where, len(got), len(want))
    for name in want.columns:
        a, b = got[name].to_numpy(), want[name].to_numpy()
        if _numeric(a) and _numeric(b):
            a, b = a.astype(np.float64), b.astype(np.float64)
            both = np.isnan(a) & np.isnan(b)
            diff = np.where(both, 0.0, np.abs(a - b))
            assert not np.isnan(diff).any() and float(diff.max(initial=0.0)) <= tol, \
                (where, name, float(np.nanmax(diff)) if diff.size else None, tol)
        else:
            assert [None if _missing(x) else str(x) for x in a] == \
                   [None if _missing(x) else str(x) for x in b], (where, name)


def _missing(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


def _raises(kind, needle, fn):
    with pytest.raises(kind) as info:
        fn()
    assert needle in str(info.value), (needle, str(info.value))


def _moved(rows):
    """The same rows under an index a caller might have (not 0…n−1)."""
    return rows.set_axis(pd.RangeIndex(1000, 1000 + 3 * len(rows), 3), axis=0)


def _sorted_rows(case, station=None):
    rows = case.rows if station is None else case.rows[case.rows["station"] == station]
    return rows.sort_values(["station", "ts"], kind="stable").reset_index(drop=True)


def _history(case, per_series):
    """The last `per_series` rows of train ∪ valid of every station, as a caller keeps them."""
    s = case.shaped
    past = pd.concat([s.rows("train"), s.rows("valid")]).sort_values("ts", kind="stable")
    past = past[[c for c in case.rows.columns]]
    return pd.concat([past[past["station"] == st].tail(per_series)
                      for st in sorted(past["station"].unique())], ignore_index=True)


# =========================================================================== the readings (rows)

@pytest.mark.parametrize("reading", ROW_READINGS)
def test_every_row_reading_answers_what_the_evaluation_stored(reading):
    pkg, case = _pkg(), pf.row_case(reading)
    predictor = case.predictor(pkg)
    rows = _moved(case.rows)
    got = predictor.predict(rows)
    out = case.manifest["output"]
    assert out["reading"] == reading
    assert list(got.columns) == list(out["fields"]), (list(got.columns), out["fields"])
    assert got.index.equals(rows.index)
    want = predictor.reader.from_stored(case.stored)
    _agree(got.reset_index(drop=True), want, _tolerance(case), reading)
    for field, source in out["compared"]:           # each raw field is the stored column
        stored = case.stored[source].to_numpy(np.float64)
        assert np.abs(got[field].to_numpy(np.float64) - stored).max() <= _tolerance(case), field
    for field, source in out["decided"]:            # each decision is the evaluation's
        if source in case.stored.columns:
            assert got[field].astype(int).tolist() == case.stored[source].astype(int).tolist()
    assert set(got["warnings"].map(type)) == {str}


def test_the_decisions_are_the_readings_own_not_the_batchs():
    """Oracles written here from bundle-contract §2.1, in numpy, on the stored raw head."""
    pkg = _pkg()
    binary = pf.row_case("binary")
    got = binary.predictor(pkg).predict(binary.rows)
    p = _sigmoid(binary.stored["score"])
    assert np.allclose(got["p_1"], p, atol=1e-6) and np.allclose(got["p_0"], 1 - p, atol=1e-6)
    assert got["prediction"].tolist() == (p >= 0.41).astype(int).tolist()
    assert got["label"].tolist() == np.where(p >= 0.41, "1", "0").tolist()
    multi = pf.row_case("multiclass")
    got = multi.predictor(pkg).predict(multi.rows)
    logits = multi.stored[[f"score_{k}" for k in range(3)]].to_numpy()
    assert got["label"].tolist() == np.asarray(["a", "b", "c"])[logits.argmax(axis=1)].tolist()
    for reading in ("anomaly-reconstruction", "anomaly-svdd"):
        case = pf.row_case(reading)
        got = case.predictor(pkg).predict(case.rows)
        cut = case.threshold["value"]
        assert np.allclose(got["score"], case.stored["row_score"], atol=1e-5)
        assert got["prediction"].tolist() == (got["score"] >= cut).astype(int).tolist()
        assert 0 < got["prediction"].sum() < len(got)   # the score scale, both decisions
    recon = pf.row_case("reconstruction")
    got = recon.predictor(pkg).predict(recon.rows)
    assert (got["reconstruction_error"] >= 0).all()


@pytest.mark.parametrize("reading", ROW_READINGS)
def test_an_answer_does_not_depend_on_the_other_rows_sent(reading):
    pkg, case = _pkg(), pf.row_case(reading)
    predictor = case.predictor(pkg)
    together = predictor.predict(case.rows)
    for k in (0, 7, len(case.rows) - 1):
        alone = predictor.predict(case.rows.iloc[[k]])
        _agree(alone.reset_index(drop=True), together.iloc[[k]].reset_index(drop=True),
               1e-6, f"{reading} row {k}")


@pytest.mark.parametrize("reading", ROW_READINGS)
def test_predictions_carry_the_readings_decision_and_the_versions(reading):
    pkg, case = _pkg(), pf.row_case(reading)
    predictor = case.predictor(pkg)
    rows = _moved(case.rows.head(20))
    frame, preds = predictor.predict(rows), predictor.predictions(rows)
    records = predictor.records
    assert len(preds) == len(rows) and [p.row_id for p in preds] == list(rows.index)
    kinds = case.manifest["output"]["field_kinds"]
    classes = [f for f in frame.columns if kinds[f]["kind"] == "class"]
    for pred, (_, rec) in zip(preds, frame.iterrows()):
        assert isinstance(pred, pkg.domain.Prediction)
        assert pred.model_version == records.model_version
        assert pred.contract_version == records.spec.feature_contract.manifest_digest
        assert pred.warnings == tuple(w for w in rec["warnings"].split("; ") if w)
        if classes and not _missing(rec[classes[0]]):
            assert str(pred.label.name) == str(rec[classes[0]])
            assert pred.label.index == [str(c) for c in case.spec["task"]["classes"]].index(
                str(rec[classes[0]]))
        else:
            assert pred.label is None
        assert len(pred.score) >= 1 and all(isinstance(v, float) for v in pred.score)


@pytest.mark.parametrize("reading", ("binary", "ordinal", "anomaly-svdd"))
def test_predict_one_is_the_row_of_the_frame(reading):
    pkg, case = _pkg(), pf.row_case(reading)
    predictor = case.predictor(pkg)
    frame = predictor.predict(case.rows)
    for k in (0, 5):
        one = predictor.predict_one(case.rows.iloc[k].to_dict())
        assert list(one) == list(frame.columns)
        for key, value in one.items():
            want = frame.iloc[k][key]
            if isinstance(value, float):
                assert abs(value - float(want)) <= 1e-6, key
            else:
                assert str(value) == str(want), key
    _raises(TypeError, "one raw row as a mapping", lambda: predictor.predict_one([1, 2]))


def test_prepare_is_the_evaluations_batch_and_the_one_path():
    """The raw test rows through the Recipe and the row assembler are the shaped test part the
    evaluation scored: the same numerics and nominal ids, in the contract's order."""
    pkg, case = _pkg(), pf.row_case("regression")
    prepared = case.predictor(pkg).prepare(case.rows)
    want = pf._row_batch(case.shaped.part("test"))
    assert torch.allclose(prepared.batch["x_num"], want["x_num"], atol=1e-6)
    assert torch.equal(prepared.batch["x_cat"], want["x_cat"])
    assert prepared.n_samples == len(case.rows)
    assert prepared.index.rows == tuple(range(len(case.rows)))
    assert prepared.sample_of_row.tolist() == list(range(len(case.rows)))
    assert isinstance(prepared, pkg.samples.Prepared)


class _Double(torch.nn.Module):
    """A test network around the case's, served through the Model port like any other: its
    `head` and its `anatomy()` (the case's, with this module's own forward) read this
    module's `forward`."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def head(self, batch):
        with torch.no_grad():
            return self(batch)

    def anatomy(self):
        """A view of this module (its forward, its mode) with the case's network structure."""
        return type(self.inner.anatomy())(self)

    def __getattr__(self, name):
        if name in ("ids", "kinds", "types", "shapes", "stages", "readout", "sample",
                    "spec_digest", "device"):
            return getattr(self.inner, name)
        return super().__getattr__(name)


class _Unordered(_Double):
    """An ordinal head whose cumulative logits rise on every other sample: probabilities that
    need clipping (the builder's rank-consistent head never gives them)."""

    def forward(self, batch):
        out = self.inner(batch)
        rise = (torch.arange(out.shape[0]) % 2 == 0)[:, None]
        return torch.where(rise, out.flip(-1), out)


def test_warnings_are_the_guards_then_the_shapes_then_the_readings():
    pkg, case = _pkg(), pf.row_case("ordinal")
    model = _Unordered(case.model).eval()
    predictor = case.predictor(pkg, model=model)
    rows = case.rows.head(40).reset_index(drop=True).copy()
    rows.loc[0, "x0"] = 1e6                                  # outside the range seen
    rows.loc[1, "zone"] = "never-seen"                       # an unseen category
    rows.loc[2, ["x0", "zone"]] = [1e6, "never-seen"]
    got = predictor.predict(rows)
    assert got.loc[0, "warnings"].startswith("x0 outside the range seen in training")
    assert got.loc[1, "warnings"].startswith("unseen category in zone")
    first, second = got.loc[2, "warnings"].split("; ")[:2]
    assert first.startswith("x0 outside") and second == "unseen category in zone"
    prepared = predictor.prepare(rows)
    with torch.no_grad():
        clipped = case.model.readout.clipped(model(prepared.batch)).numpy()
    assert 0 < clipped.sum() < len(clipped)
    clip = importlib.import_module(f"{pkg.name}.serving.heads").CLIPPED
    probabilities = got[[f"p_{c}" for c in ("low", "mid", "high", "top")]].to_numpy()
    assert np.allclose(probabilities.sum(axis=1), 1.0) and (probabilities >= 0).all()
    for k, flag in enumerate(clipped):
        assert (clip in got.loc[k, "warnings"]) == bool(flag), k
        if flag and k < 3:
            assert got.loc[k, "warnings"].endswith(clip)


def test_an_empty_request_and_what_is_not_a_frame():
    pkg, case = _pkg(), pf.row_case("binary")
    port = pf.port_only(case.model)
    predictor = case.predictor(pkg, model=port)
    empty = predictor.predict(case.rows.iloc[0:0])
    assert list(empty.columns) == case.manifest["output"]["fields"] and len(empty) == 0
    assert predictor.predictions(case.rows.iloc[0:0]) == ()
    _raises(TypeError, "DataFrame of raw rows", lambda: predictor.predict(case.rows.to_dict()))
    assert predictor.info() == json.loads(json.dumps(case.manifest))
    assert predictor.model is port and predictor.records.output.reading.value == "binary"


# =========================================================================== windows

def _network_answers(case, predictor, rows, history=None):
    """What the network gives on the prepared batch, read by the reader: sample → fields."""
    prepared = predictor.prepare(rows, history)
    with torch.no_grad():
        out = case.model(prepared.batch)
    reader = predictor.reader
    return prepared, reader.frame(reader.head(out, prepared.batch, case.model))


@pytest.mark.parametrize("kind", ("forecast", "sequence-regression", "sequence-binary"))
def test_a_window_row_is_answered_iff_its_series_has_length_minus_one_rows_before_it(kind):
    pkg, case = _pkg(), pf.window_case(kind)
    predictor = case.predictor(pkg)
    rows = case.rows.sample(frac=1.0, random_state=3)          # any order a caller sends
    got = predictor.predict(rows)
    out = case.manifest["output"]
    assert list(got.columns) == out["fields"] and got["answered"].dtype == bool
    assert got.index.equals(rows.index)
    rank = rows.groupby("station")["ts"].rank(method="first").to_numpy() - 1
    assert got["answered"].tolist() == (rank >= 3).tolist()
    values = [f for f in out["fields"] if f not in ("answered", "warnings")]
    missed = ~got["answered"].to_numpy()
    assert got.loc[missed, values].isna().all().all()
    # a row whose window reads a lag with no history (this fixture imputes none) answers a
    # non-finite output: it keeps the window's answer, says so, and carries no decision
    held = got["warnings"].str.contains("no decision is read").to_numpy() & ~missed
    decided = [f for f in values if out["field_kinds"][f]["kind"] == "decision"]
    for field in decided:                     # a decision stays an integer beside the gaps
        assert str(got[field].dtype) == "Int64", got[field].dtype
        assert set(got.loc[~missed & ~held, field].tolist()) <= {0, 1}
        assert got.loc[held, field].isna().all()
    assert got.loc[held, "label"].isna().all() if "label" in got else True
    assert got.loc[missed, "warnings"].str.contains(
        "window incomplete: needs 3 preceding rows").all()
    prepared, theirs = _network_answers(case, predictor, rows)
    kept = ~missed & ~held
    at = prepared.sample_of_row[kept]
    _agree(got.loc[kept, values].reset_index(drop=True),
           theirs.iloc[at][values].reset_index(drop=True), 1e-6, kind)
    preds = predictor.predictions(rows)
    assert all(p.score == () and p.label is None for p, m in zip(preds, missed) if m)
    assert all(len(p.score) == len(out["compared"]) for p, m in zip(preds, missed) if not m)
    if out["reading"] == "binary":                  # the window's decision, named
        labels = [p.label for p, k in zip(preds, kept) if k]
        assert [str(c.name) for c in labels] == got.loc[kept, "label"].tolist()
        assert all(p.label is None for p, h in zip(preds, held) if h), "no class read from NaN"


def test_history_completes_the_windows_and_the_split_does_not_matter():
    pkg, case = _pkg(), pf.window_case("forecast")
    predictor = case.predictor(pkg)
    rows, history = _sorted_rows(case), _history(case, 5)
    whole = predictor.predict(rows, history=history)
    assert whole["answered"].all()
    values = ["value_h1", "value_h2"]
    # the first k rows of each series moved into the history: the rest answer the same
    k = 4
    head = rows.groupby("station").head(k)
    tail = rows.drop(index=head.index)
    moved = predictor.predict(tail, history=pd.concat([history, head], ignore_index=True))
    _agree(moved[values].reset_index(drop=True),
           whole.loc[tail.index, values].reset_index(drop=True), 1e-5, "split")
    # history as a list of raw rows (the JSON a caller sends) is the same history
    listed = predictor.predict(rows.head(6), history=history.to_dict("records"))
    _agree(listed[values], whole.head(6)[values], 1e-6, "listed history")


def test_predict_one_takes_its_series_history():
    """R-D16: a window bundle's single row is answered with `history_rows` rows of history."""
    pkg, case = _pkg(), pf.window_case("forecast")
    predictor = case.predictor(pkg)
    rows = _sorted_rows(case, "B")
    history = _history(case, 5)
    history = history[history["station"] == "B"]
    want = predictor.predict(rows.head(1), history=history)
    one = predictor.predict_one(rows.iloc[0].to_dict(), history=history)
    assert one["answered"] is True and abs(one["value_h1"] - want["value_h1"].iloc[0]) < 1e-6
    bare = predictor.predict_one(rows.iloc[0].to_dict())
    assert bare["answered"] is False and bare["value_h1"] is None
    assert "window incomplete" in bare["warnings"]


def test_a_forecast_request_without_its_target_is_refused_naming_it():
    pkg, case = _pkg(), pf.window_case("forecast")
    _raises(ValueError, "'target'",
            lambda: case.predictor(pkg).predict(case.rows.drop(columns=["target"])))


# =========================================================================== graphs

@pytest.mark.parametrize("name", ("graph-keypair", "graph-knn"))
def test_a_node_level_answer_is_the_sealed_views_gathered_once(name):
    """R-A2: the network gathers `target_index` after its head and the Predictor never gathers
    again, so a served node answers what the evaluation's sealed test view stored for it."""
    pkg, case = _pkg(), pf.graph_case(name)
    predictor = case.predictor(pkg)
    got = predictor.predict(case.rows)
    assert list(got.columns) == case.manifest["output"]["fields"]
    want = predictor.reader.from_stored(case.stored)
    _agree(got.reset_index(drop=True), want, max(_tolerance(case), 1e-6), name)


class _Ungathered(_Double):
    """A node-level model that forgets its own gather: every node's output comes back."""

    def forward(self, batch):
        return self.inner.forward({k: v for k, v in batch.items() if k != "target_index"})


def test_a_model_that_returns_every_node_is_refused_not_gathered_again():
    pkg, case = _pkg(), pf.graph_case("graph-knn")
    predictor = case.predictor(pkg, model=_Ungathered(case.model).eval())
    _raises(RuntimeError, "the Predictor never gathers again",
            lambda: predictor.predict(case.rows.head(5)))


def test_node_answers_do_not_depend_on_the_batch():
    pkg, case = _pkg(), pf.graph_case("graph-keypair")
    predictor = case.predictor(pkg)
    together = predictor.predict(case.rows)
    for q in range(0, len(case.rows), 17):
        alone = predictor.predict(case.rows.iloc[[q]])
        _agree(alone.reset_index(drop=True), together.iloc[[q]].reset_index(drop=True), 1e-6,
               f"row {q}")
    perm = np.random.default_rng(2).permutation(len(case.rows))
    moved = predictor.predict(case.rows.iloc[perm])
    _agree(moved.reset_index(drop=True), together.iloc[perm].reset_index(drop=True), 1e-6,
           "permuted")
    _raises(ValueError, "a graph sample takes no history",
            lambda: predictor.predict(case.rows.head(3), history=case.rows.tail(2)))


def test_rows_of_one_graph_share_one_answer_under_its_graph_id():
    pkg, case = _pkg(), pf.graph_case("graph-level")
    predictor = case.predictor(pkg)
    got = predictor.predict(case.rows)
    fields = case.manifest["output"]["fields"]
    assert list(got.columns) == fields and fields[0] == "graph_id"
    ids = case.rows["molecule_id"].astype(str)
    assert got["graph_id"].tolist() == ids.tolist()
    answer = [f for f in fields if f not in ("graph_id", "warnings")]
    for gid, rows in got.groupby("graph_id"):
        assert len(rows[answer].drop_duplicates()) == 1, gid
    prepared, theirs = _network_answers(case, predictor, case.rows)
    _agree(got[answer].reset_index(drop=True),
           theirs.iloc[prepared.sample_of_row][answer].reset_index(drop=True), 1e-6, "level")
    one = ids.iloc[0]
    alone = predictor.predict(case.rows[ids == one])
    _agree(alone[answer].reset_index(drop=True), got[ids == one][answer].reset_index(drop=True),
           1e-5, "one graph alone")


# =========================================================================== refusals

def test_construction_refusals_name_what_is_missing():
    pkg, case = _pkg(), pf.row_case("binary")
    P = pkg.predict.Predictor
    transform = case.shaped.module.transform
    _raises(TypeError, "pass the ServingRecords value",
            lambda: P(case.model, transform, case.manifest))
    # a spec with no features
    bare = json.loads(json.dumps(case.spec))
    bare["data_contract"]["features"] = []
    records = pkg.records.ServingRecords.of(bare, case.manifest, case.schema)
    _raises(ValueError, "lists no features", lambda: P(case.model, transform, records))
    # a model built from another spec
    other = pkg.build(pf.row_case("regression").spec)
    _raises(ValueError, "they are not one bundle's",
            lambda: P(other, transform, case.records(pkg)))
    # a graph sample without its reference graph; a row sample handed one
    graph = pf.graph_case("graph-knn")
    _raises(ValueError, "answers through its reference graph",
            lambda: P(graph.model, graph.shaped.module.transform, graph.records(pkg), None))
    _raises(ValueError, "has no reference graph",
            lambda: P(case.model, transform, case.records(pkg), graph.context))


def test_an_svdd_answer_needs_the_models_centre():
    pkg, case = _pkg(), pf.row_case("anomaly-svdd")
    transform, records = case.shaped.module.transform, case.records(pkg)
    unset = pkg.build(case.spec)                       # the centre buffer, never set
    _raises(ValueError, "a centre that was never set",
            lambda: pkg.predict.Predictor(unset, transform, records))

    class _NoCentre(_Double):                          # the port says: no centre buffer
        def forward(self, batch):                      # (the view finds no `svdd_center`)
            return self.inner(batch)
    _raises(ValueError, "no svdd_center buffer",
            lambda: pkg.predict.Predictor(pf.port_only(_NoCentre(case.model).eval()), transform,
                                          records))
    # a model that is not a staged network (a baseline's anatomy is None) is not served
    class _Unstaged(_Double):
        def anatomy(self):
            return None
    _raises(ValueError, "not a staged network",
            lambda: pkg.predict.Predictor(pf.port_only(_Unstaged(case.model)), transform,
                                          records))


def test_a_model_in_training_mode_is_refused_never_switched():
    """The Predictor reads the mode through the port (`anatomy().training`) and never sets it:
    a network in training mode is refused when the Predictor is built and at every request."""
    pkg, case = _pkg(), pf.row_case("regression")
    model = pkg.build(case.spec)
    model.train()
    _raises(RuntimeError, "training mode", lambda: case.predictor(pkg, model=model))
    assert model.training                             # refused, not switched
    model.eval()
    predictor = case.predictor(pkg, model=model)
    predictor.predict(case.rows.head(3))
    model.train()
    try:
        _raises(RuntimeError, "training mode", lambda: predictor.predict(case.rows.head(3)))
        assert model.training
    finally:
        model.eval()


# =========================================================================== the module

def _rendered(rel):
    return (_pkg().root / rel).read_text(encoding="utf-8")


def _map_with_implemented_predictor(root):
    bmap = json.loads((root / "blueprint-map.json").read_text(encoding="utf-8"))
    for entry in bmap["entries"]:
        if entry.get("role") == "Predictor":
            entry["status"] = "implemented"
    return bmap


def _c14(root, pipeline):
    markers, findings = conformance.check_report(root, _map_with_implemented_predictor(root),
                                                 pipeline)
    return [f for f in findings if f.code == "C14" and f.file == "serving/predict.py"]


def test_the_predictor_module_holds_c14_and_the_size_discipline():
    pkg = _pkg()
    pipeline = pf._pipeline(pkg.name)
    assert _c14(pkg.root, pipeline) == []
    tree = ast.parse(_rendered("serving/predict.py"))
    public = {n.name for n in tree.body
              if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and not n.name.startswith("_")}
    assert public == {"Predictor"}
    exported = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                    and getattr(n.targets[0], "id", None) == "__all__")
    assert [e.value for e in exported.elts] == ["Predictor"]
    assert conformance.size_findings(tree, "serving/predict.py") == []
    cli = ast.parse(re.sub(r"\{(package|env|mod_[a-z]+)\}", "x",
                           CLI_TEMPLATE.read_text(encoding="utf-8")))
    assert conformance.size_findings(cli, "predict.py") == []
    # the check is live: a public function beside the class (v2.2's build) is a C14 finding
    copy = pf.WORK / "c14-copy"
    shutil.rmtree(copy, ignore_errors=True)
    shutil.copytree(pkg.root, copy / pkg.name)
    target = copy / pkg.name / "serving" / "predict.py"
    target.write_text(target.read_text(encoding="utf-8")
                      + "\n\ndef build(model):\n    return Predictor(model, None, None)\n",
                      encoding="utf-8")
    found = _c14(copy / pkg.name, pipeline)
    assert [f.symbol for f in found] == ["build"], found


def test_the_predictor_imports_support_files_only():
    """C5 (D2 §4): the adapter imports the domain and its support files, never another
    adapter; the Model's readout reaches it through the readers. `inspection.inspector` is a
    support file too (I06: `Predictor.inspector()` imports it when first called, so a bundle
    packed with --no-inspect, which has no inspection modules, still predicts). The Model
    port is imported for its annotation only (under TYPE_CHECKING: ports import no code)."""
    tree = ast.parse(_rendered("serving/predict.py"))
    inner = {(n.level, n.module) for n in ast.walk(tree)
             if isinstance(n, ast.ImportFrom) and n.level}
    assert inner == {(2, "domain"), (2, "ports"), (1, "guards"), (1, "records"), (1, "samples"),
                     (1, "values"), (1, "inspection.inspector")}, inner
    guarded = [n for n in tree.body if isinstance(n, ast.If)
               and getattr(n.test, "id", None) == "TYPE_CHECKING"]
    assert [(i.level, i.module) for g in guarded for i in g.body] == [(2, "ports")]
    lazy = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "inspector"]
    assert lazy and any(isinstance(n, ast.ImportFrom) and n.module == "inspection.inspector"
                        for n in ast.walk(lazy[0])), "the inspection modules are imported lazily"


def test_no_foreign_literal_in_the_two_templates():
    assert LINT.is_file(), f"{LINT} (T04) is on this tree"
    proc = subprocess.run([sys.executable, str(LINT), str(PREDICT_TEMPLATE), str(CLI_TEMPLATE)],
                          capture_output=True, text=True, cwd=str(LINT.parent), timeout=300)
    assert proc.returncode == 0, proc.stdout + proc.stderr


# =========================================================================== the bundle CLI

@pytest.fixture(scope="module")
def binary_bundle():
    return pf.Bundle.build(pf.row_case("binary"), "s05_b_binary")


def test_load_builds_the_predictor_and_check_examples_passes(binary_bundle):
    cli = pf.load_cli(binary_bundle)
    predictor = cli.load(binary_bundle.root)
    assert type(predictor).__name__ == "Predictor" and predictor is not None
    assert type(predictor).__module__ == f"{binary_bundle.package}.serving.predict"
    assert predictor.records.model_version == json.loads(
        (binary_bundle.root / "MANIFEST.json").read_text())["model_version"]
    result = cli.check_examples(binary_bundle.root, predictor)
    assert result["passed"], result["problems"]
    assert result["rows"] == 12 and result["decisions_match"] and result["labels_match"]
    assert result["positional"] is False and result["columns"] == 1
    assert result["evidence"] == "authenticated against MANIFEST.files"
    assert result["max_abs_diff"] <= result["tolerance"]
    again = cli.check_examples(binary_bundle.root)      # load()ed by itself
    assert again["passed"] and again["max_abs_diff"] == result["max_abs_diff"]
    got = cli.predict(binary_bundle.case.rows.head(4).to_dict("records"), binary_bundle.root)
    assert list(got.columns) == binary_bundle.case.manifest["output"]["fields"]


@pytest.mark.parametrize("reading", ("binary", "multiclass", "quantile"))
def test_check_examples_passes_on_a_positional_v22_example_set(reading):
    bundle = pf.Bundle.build(pf.row_case(reading), f"s05_pos_{reading}", examples="positional")
    expected = pd.read_parquet(bundle.root / "examples" / "expected.parquet")
    assert "__row__" not in expected.columns
    result = pf.load_cli(bundle).check_examples(bundle.root)
    assert result["passed"], result["problems"]
    assert result["positional"] is True and result["rows"] == 12
    assert result["decisions"] == (1 if reading == "multiclass" else 0)


def _rewrite_expected(bundle, change):
    path = bundle.root / "examples" / "expected.parquet"
    frame = pd.read_parquet(path)
    change(frame)
    frame.to_parquet(path, index=False)
    bundle.write_manifest()


def test_check_examples_names_each_disagreement():
    bundle = pf.Bundle.build(pf.row_case("binary"), "s05_b_wrong")
    cli = pf.load_cli(bundle)
    original = pd.read_parquet(bundle.root / "examples" / "expected.parquet")

    def reset():
        original.to_parquet(bundle.root / "examples" / "expected.parquet", index=False)
        bundle.write_manifest()
    bump = np.zeros(12)
    bump[3] = 0.5
    _rewrite_expected(bundle, lambda f: f.__setitem__("score", f["score"] + bump))
    bad = cli.check_examples(bundle.root)
    assert not bad["passed"] and "score at example row 3" in bad["problems"][0]
    reset()
    _rewrite_expected(bundle, lambda f: f.__setitem__("prediction", 1 - f["prediction"]))
    bad = cli.check_examples(bundle.root)
    assert not bad["passed"] and not bad["decisions_match"]
    assert "prediction at example row 0" in bad["problems"][0]
    reset()
    _rewrite_expected(bundle, lambda f: f.drop(columns=["prediction"], inplace=True))
    bad = cli.check_examples(bundle.root)
    assert not bad["passed"] and "lacks prediction" in bad["problems"][0]
    reset()
    _rewrite_expected(bundle, lambda f: f.__setitem__("__row__", f["__row__"] + 100))
    bad = cli.check_examples(bundle.root)
    assert not bad["passed"] and "__row__ points outside the 12 rows" in bad["problems"][0]
    reset()
    assert cli.check_examples(bundle.root)["passed"]
    # an example file altered after pack is refused before anything is compared
    rows = bundle.root / "examples" / "rows.csv"
    rows.write_text(rows.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    _raises(ValueError, "examples/rows.csv is not the file MANIFEST.json pinned",
            lambda: cli.check_examples(bundle.root))


@pytest.fixture(scope="module")
def window_bundle():
    return pf.Bundle.build(pf.window_case("forecast"), "s05_b_window", examples="window")


def test_a_window_bundle_checks_its_answered_rows_and_names_an_unanswered_one(window_bundle):
    cli = pf.load_cli(window_bundle)
    result = cli.check_examples(window_bundle.root)
    assert result["passed"], result["problems"]
    assert result["rows"] == 8 and result["unanswered"] == [] and result["columns"] == 2
    _rewrite_expected(window_bundle,
                      lambda f: f.__setitem__("__row__", [0] + f["__row__"].tolist()[1:]))
    bad = cli.check_examples(window_bundle.root)
    assert not bad["passed"] and bad["unanswered"] == [0]
    assert "1 compared example row(s) went unanswered (first: row 0)" in bad["problems"]
    window_bundle._examples("window")
    window_bundle.write_manifest()


@pytest.fixture(scope="module")
def graph_bundle():
    return pf.Bundle.build(pf.graph_case("graph-keypair"), "s05_b_graph")


def test_a_graph_bundle_loads_its_reference_graph(graph_bundle):
    cli = pf.load_cli(graph_bundle)
    root = graph_bundle.root
    preprocess = cli._recipe(root)
    context = cli.load_context(root, preprocess)
    assert type(context).__name__ == "GraphContext" and context.n_reference > 0
    result = cli.check_examples(root)
    assert result["passed"], result["problems"]
    nodes = root / "graph" / "reference-nodes.parquet"
    saved = nodes.read_bytes()
    try:
        frame = pd.read_parquet(nodes)
        frame.iloc[0, frame.columns.get_loc("load_mw")] += 1.0
        frame.to_parquet(nodes, index=False)
        _raises(ValueError, "graph/reference-nodes.parquet is not the file MANIFEST.json pinned",
                lambda: cli.load(root))                  # N-S9: refused, never answered from
        _raises(ValueError, "reference-nodes.parquet",
                lambda: cli.load_context(root, preprocess))
    finally:
        nodes.write_bytes(saved)
    assert cli.check_examples(root)["passed"]


def test_verify_refuses_an_altered_data_file_by_name(binary_bundle):
    cli = pf.load_cli(binary_bundle)
    spec = binary_bundle.root / "netspec.json"
    saved = spec.read_bytes()
    try:
        spec.write_bytes(saved + b" ")
        _raises(ValueError, "netspec.json is not the file MANIFEST.json pinned",
                lambda: cli.verify(binary_bundle.root))
        _raises(ValueError, "netspec.json", lambda: cli.load(binary_bundle.root))
    finally:
        spec.write_bytes(saved)
    assert cli.verify(binary_bundle.root) == binary_bundle.root.resolve()
    manifest = json.loads((binary_bundle.root / "MANIFEST.json").read_text())
    try:
        binary_bundle.write_manifest(changes={"roles": None})
        _raises(ValueError, "records no file roles", lambda: cli.verify(binary_bundle.root))
    finally:
        binary_bundle.write_manifest(manifest["model_version"])


def test_load_monitor_builds_the_monitor_over_the_bundles_reference(binary_bundle):
    cli = pf.load_cli(binary_bundle)
    monitor = cli.load_monitor(binary_bundle.root)
    assert type(monitor).__name__ == "Monitor"
    window = binary_bundle.case.rows
    readings = monitor.report(window, cli.load(binary_bundle.root).predict(window))
    assert len(readings) == 1 and readings[0].measure == "psi"
    assert isinstance(monitor.check(window), tuple)


def _clean_run(bundle, *args):
    """The CLI from a shell with the skills unreachable: an empty environment."""
    env = {"PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin", "HOME": str(pf.WORK),
           "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": ""}
    return subprocess.run([sys.executable, "predict.py", *map(str, args)], cwd=str(bundle.root),
                          env=env, capture_output=True, text=True, timeout=600)


def test_the_cli_from_a_clean_shell(binary_bundle):
    done = _clean_run(binary_bundle, "--check")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "ok  every file hashes to what MANIFEST.json records" in done.stdout
    assert done.stdout.strip().splitlines()[-1] == "2 checks passed"
    out = pf.WORK / "answers.csv"
    done = _clean_run(binary_bundle, "--input", "examples/rows.csv", "--output", out)
    assert done.returncode == 0 and done.stdout.startswith("OK: 12 answer(s)"), done.stdout
    assert list(pd.read_csv(out).columns) == binary_bundle.case.manifest["output"]["fields"]
    row = binary_bundle.case.rows.iloc[0].to_dict()
    done = _clean_run(binary_bundle, "--one", json.dumps(row))
    assert done.returncode == 0, done.stderr
    assert set(json.loads(done.stdout)) == set(binary_bundle.case.manifest["output"]["fields"])


def test_a_window_bundle_from_a_clean_shell(window_bundle):
    """`--one` with its series' history from a file (R-D16), `--check`, an altered file."""
    case = window_bundle.case
    rows = _sorted_rows(case, "C")
    history = _history(case, 5)
    history = history[history["station"] == "C"]
    hist_file = pf.WORK / "history.csv"
    history.to_csv(hist_file, index=False)
    done = _clean_run(window_bundle, "--one", json.dumps(rows.iloc[0].to_dict()),
                      "--history", hist_file)
    assert done.returncode == 0, done.stderr
    answer = json.loads(done.stdout)
    assert answer["answered"] is True and answer["value_h1"] is not None
    done = _clean_run(window_bundle, "--check")
    assert done.returncode == 0 and "2 checks passed" in done.stdout, done.stdout + done.stderr
    # an altered file fails the check, named
    target = window_bundle.root / "fitted.json"
    saved = target.read_bytes()
    try:
        target.write_bytes(saved + b"\n")
        done = _clean_run(window_bundle, "--check")
        assert done.returncode == 1 and "ERROR: fitted.json is not the file" in done.stdout
    finally:
        target.write_bytes(saved)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
