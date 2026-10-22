#!/usr/bin/env python3
"""
test_monitor_kinds.py — the monitor per reading (build plan S06; D2 §7; bundle-contract §7.5).

    python3 -m pytest test_monitor_kinds.py [-q]          (or: python3 test_monitor_kinds.py)

The template `templates/serving/package/monitor.py` is rendered into the Monitor's module of a
package beside the live `domain.py` (`render_templates.py`, B04) and imported; the bundle's
readers (`records`, `heads`, `values`, S03) are rendered beside it, so the pack side measures its
references through a real reader. Every output record is vega-pack's own
(`output_record.output_record`, S01), and the pack side is vega-pack's `monitor_pack`,
`monitor_plan` and `monitor_reference`, imported from the tree.

What it proves:

1. **Each new kind on synthetic frames**: `output_psi`, `prediction_mix`, `label_rate`,
   `alert_rate` and `residual` are quiet on a window drawn like their reference and move on a
   shifted one; a missing field, a missing target or too few rows is `unmeasured`, never a
   guess; a forecast step's target is the truth h rows later in the same series. The v2.2 kinds
   (`psi`, `rate`, `unseen_rate`, `class_recall`) keep their readings.
2. **`answered == False` rows are dropped**, before any reading, and every reading says so
   (only an explicit false: `answered` defaults to true).
3. **`validate_plan` refusals**: an unknown kind, a kind not admitted for the reading, a missing
   parameter, an unknown column (or a predicate operator or value the monitor cannot read),
   field (or a field of the wrong kind, an `output_psi` on any field but the record's
   `score_field`, or a forecast quantile other than the point one), class or label, a truth the
   bundle does not take, an anomaly alarm without a threshold, a parameter the monitor cannot
   read (a count, a limit, `thresholds`, `direction`, a NaN), an id that is not a string, is
   duplicated or reserved — each a problem (exit 2), never a traceback. A check naming no field
   reads its kind's default (the score field, the decided class, the alarm).
4. **References are measured on valid only** (the answer checks and the unseen rate; the input
   checks on train, and the test part is never asked for): through the bundle's reader over the
   stored valid predictions, or — where those cannot answer a check (a per-row score or a truth
   not stored, a graph-level store of one row per graph) — the Predictor over the valid part's
   raw rows, its unanswered rows left out; the residual target rule of the pack is the
   Monitor's.
5. **The automatic output check is added**: `output_psi` on the score field, else the class
   mix, else the label rates — valid for every reading; `render` writes it into
   `vega/monitor-reference@2` (strict JSON: open bin edges are `null`), renders the adapter,
   and the smoke loads it through `load_monitor` — the train window on its inputs, the valid
   window with its truth and the Predictor's frame. Without a plan the stub stays. The v2.2
   bridge of build plan §2.12(3) is closed (S07 deleted v2.2's template).
6. The template's discipline: C11–C14 (only `Monitor` public, the rest `_prefixed` helpers and
   UPPER_CASE constants), C7, its imports, its kinds equal to vega-pack's catalog, the literal
   gate on the template and the three pack modules.

Exit 0 when every test passes.
"""
from __future__ import annotations

import ast
import copy
import importlib
import json
import math
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
SKILLS = SCRIPTS.parent.parent
TEMPLATES = SCRIPTS / "backends" / "pytorch" / "templates"
PACKAGE_TEMPLATES = TEMPLATES / "serving" / "package"
PACK_SCRIPTS = SKILLS / "vega-pack" / "scripts"
LINT = SKILLS / "vega" / "scripts" / "tests" / "lint_templates.py"
READERS = ("records.py", "heads.py", "values.py", "guards.py", "samples.py")
PKG = "s06_monitor"
for _p in (str(SCRIPTS), str(HERE), str(PACK_SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import conformance  # noqa: E402
import monitor_pack  # noqa: E402  (vega-pack, S06)
import monitor_plan  # noqa: E402  (vega-pack, S06)
import monitor_reference as mref  # noqa: E402  (vega-pack, S06)
import output_record as orec  # noqa: E402  (vega-pack, S01: the producer of vega/output@2)
import render_templates as rt  # noqa: E402
from pack_context import PackExit  # noqa: E402
from vega_pack_core import CATALOG, monitor_kinds_for  # noqa: E402


# ================================================================ the rendered package

def _purge(prefix):
    for name in [m for m in sys.modules if m == prefix or m.startswith(prefix + ".")]:
        del sys.modules[name]


def _add_graphs(pipeline, rec):
    """S04's `graphs.py` (what `samples.py` imports), rendered with the emitter's own mapping
    into the Predictor's module and mapped there, as pack renders `package/*.py`: the v2.3 emit
    table would place any `graphs.py` in the ingestion module (test_samples' way)."""
    emit, spec, root = rt.load_emit(), rt.reference_spec("row"), Path(rec["root"])
    mapping = emit.common_mapping(spec, pipeline, emit.plan(spec, pipeline))
    text = (PACKAGE_TEMPLATES / "graphs.py").read_text(encoding="utf-8")
    (root / "serving" / "graphs.py").write_text(emit.render_text(text, mapping), encoding="utf-8")
    doc = rec["blueprint_map"]
    row = next(e for e in doc["entries"] if e["file"] == "serving/samples.py")
    doc["entries"].append(dict(row, file="serving/graphs.py"))
    (root / "blueprint-map.json").write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


@pytest.fixture(scope="module")
def pkg(tmp_path_factory):
    out = tmp_path_factory.mktemp("s06-render")
    pipeline = rt.reference_pipeline()
    pipeline["system"]["package"] = PKG
    rt.render(TEMPLATES, out, pipeline, "row", only=["domain.py", "readout.py", "assemble.py"])
    rt.render(PACKAGE_TEMPLATES, out, pipeline, "row", only=list(READERS), into="Predictor")
    rec = rt.render(PACKAGE_TEMPLATES, out, pipeline, "row", only=["monitor.py"], into="Monitor")
    assert "serving/monitor.py" in rec["files"], rec["files"]
    _add_graphs(pipeline, rec)
    sys.path.insert(0, str(out))
    try:
        yield SimpleNamespace(
            root=out / PKG, pipeline=pipeline,
            domain=importlib.import_module(f"{PKG}.domain"),
            records=importlib.import_module(f"{PKG}.serving.records"),
            samples=importlib.import_module(f"{PKG}.serving.samples"),
            values=importlib.import_module(f"{PKG}.serving.values"),
            monitor=importlib.import_module(f"{PKG}.serving.monitor"))
    finally:
        sys.path.remove(str(out))
        _purge(PKG)


def monitor_of(pkg, checks, recipe=None, labels=None):
    reference = {"schema": mref.REFERENCE_SCHEMA, "checks": checks}
    return pkg.monitor.Monitor(reference, recipe=recipe,
                               labels=labels if labels is not None else pkg.domain.LabelSet(()))


def rows_of(n):
    return pd.DataFrame({"x0": np.zeros(n)})


def only(readings):
    assert len(readings) == 1, readings
    return readings[0]


def window_of(reading) -> dict:
    """The reading's provenance as the five facts it carries (a `ReadingProvenance`): `rows`
    its window's rows, and `note` only when the measurement noted something."""
    w = reading.window
    out = {"rows": w.window_rows, "check": w.check, "concern": w.concern, "title": w.title}
    return dict(out, note=w.note) if w.note else out


# ================================================================ specs and output records

def _num(name):
    return {"name": name, "dtype": "float32", "role": "numeric"}


NUMERIC = [_num(f"x{i}") for i in range(4)]
WINDOW = {"shape": "window", "length": 4, "stride": 1, "series_by": "s", "order_by": "ts",
          "context": "previous-part"}
FORECAST = {**WINDOW, "horizon": 2, "autoregressive": True}
PROB = {"applies": True, "value": 0.5, "scale": "probability", "source": "valid-optimum"}
SCORE = {"applies": True, "value": 0.8, "scale": "score", "source": "valid-optimum"}
UNDECIDED = {"applies": False, "value": None, "scale": "score"}
SCHEMA = {"input": [{"name": f"x{i}"} for i in range(4)] + [{"name": "zone"}],
          "ignored": ["y"]}


def spec_of(task, head, loss="bce_with_logits", sample=None):
    return {"schema": "vega/netspec@1", "seed": 7, "task": task,
            "input": {"manifest_digest": "sha256:" + "ab" * 32},
            "data_contract": {"input_dim": len(NUMERIC), "features": NUMERIC,
                              "sample": sample or {"shape": "row"}},
            "family": {"name": "user-defined"},
            "blocks": [{"id": "B1", "type": "dense", "params": {"units": 8}},
                       {"id": "B2", "type": "head", "params": {"out": head}}],
            "loss": loss if isinstance(loss, dict) else {"name": loss, "params": {}},
            "metrics": {"primary": "pr_auc"}, "optimizer": {"name": "adamw"},
            "training": {"max_epochs": 1, "batch_size": 8}}


def _quantile(qs):
    return {"name": "quantile", "params": {"quantiles": qs}}


REGRESSION = {"kind": "regression", "label": "y"}
LABELLED = {"kind": "anomaly", "label": "y", "classes": [0, 1], "positive": 1}
CASES = {
    "binary": (spec_of({"kind": "binary", "label": "y", "classes": [0, 1], "positive": 1}, 1),
               PROB),
    "multiclass": (spec_of({"kind": "multiclass", "label": "y", "classes": ["a", "b", "c"]}, 3,
                           "cross_entropy"), None),
    "multilabel": (spec_of({"kind": "multilabel", "label": "y",
                            "classes": ["red", "green", "blue"],
                            "label_format": {"kind": "delimited", "sep": "|"}}, 3), PROB),
    "ordinal": (spec_of({"kind": "ordinal", "label": "y", "classes": ["lo", "mid", "hi"]}, 2,
                        "cumulative_link"), None),
    "regression": (spec_of(REGRESSION, 1, "mse"), None),
    "quantile": (spec_of(REGRESSION, 3, _quantile([0.1, 0.5, 0.9])), None),
    "quantile-no-median": (spec_of(REGRESSION, 2, _quantile([0.25, 0.75])), None),
    "forecast": (spec_of({"kind": "forecast", "label": "y"}, 2, "mse", FORECAST), None),
    "forecast-quantile": (spec_of({"kind": "forecast", "label": "y"}, 6,
                                  _quantile([0.1, 0.5, 0.9]), FORECAST), None),
    "reconstruction": (spec_of({"kind": "unsupervised"}, 4, "reconstruction"), None),
    "anomaly-reconstruction": (spec_of(LABELLED, 4, "reconstruction"), SCORE),
    "anomaly-unlabelled": (spec_of({"kind": "anomaly"}, 4, "vae"), SCORE),
    "anomaly-svdd": (spec_of({"kind": "anomaly"}, 3, "svdd"), SCORE),
    "anomaly-undecided": (spec_of({"kind": "anomaly"}, 3, "svdd"), UNDECIDED),
}
READINGS = {"binary": "binary", "multiclass": "multiclass", "multilabel": "multilabel",
            "ordinal": "ordinal", "regression": "regression", "quantile": "quantile",
            "quantile-no-median": "quantile", "forecast": "forecast",
            "forecast-quantile": "forecast",
            "reconstruction": "reconstruction",
            "anomaly-reconstruction": "anomaly-reconstruction",
            "anomaly-unlabelled": "anomaly-reconstruction", "anomaly-svdd": "anomaly-svdd",
            "anomaly-undecided": "anomaly-svdd"}


def output_of(name):
    spec, threshold = CASES[name]
    return orec.output_record(spec, threshold, (spec.get("task") or {}).get("label"))


def manifest_of(name):
    spec, threshold = CASES[name]
    return {"output": output_of(name), "sample": orec.sample_record(spec, {}, None)}


def reader_of(pkg, name):
    spec, _ = CASES[name]
    records = pkg.records.ServingRecords.of(spec, manifest_of(name), {"input": []},
                                            "sha256:model")
    return pkg.values.reader_for(records)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# ================================================================ 1. the new kinds on frames

def test_output_psi_is_quiet_in_distribution_and_moves_on_a_shift(pkg):
    rng = np.random.default_rng(1)
    reference = mref._psi_reference({"bins_n": 10}, pd.Series(rng.beta(2, 5, 4000)))
    mon = monitor_of(pkg, [{"id": "o", "kind": "output_psi", "field": "p_1", **reference}])
    same = pd.DataFrame({"p_1": rng.beta(2, 5, 2000)})
    moved = pd.DataFrame({"p_1": rng.beta(5, 2, 2000)})
    r = only(mon.report(rows_of(2000), same))
    assert (r.column, r.measure, r.verdict) == ("p_1", "psi", "ok"), r
    assert mon.check(rows_of(2000), same) == ()
    assert only(mon.report(rows_of(2000), moved)).verdict == "alert"
    assert only(mon.check(rows_of(2000), moved)).column == "p_1"
    # too few answered rows, no such field, no predictions: unmeasured, never a guess
    assert only(mon.report(rows_of(10), same.iloc[:10])).verdict == "unmeasured"
    assert only(mon.report(rows_of(2000), same.rename(columns={"p_1": "p_0"}))).verdict \
        == "unmeasured"
    r = only(mon.report(rows_of(2000)))
    assert r.verdict == "unmeasured" and r.value is None and "'p_1'" in window_of(r)["note"]


def test_prediction_mix_reads_the_decided_classes(pkg):
    rng = np.random.default_rng(2)
    classes, mix = ["a", "b", "c"], [0.5, 0.3, 0.2]
    check = {"id": "m", "kind": "prediction_mix", "field": "label", "categorical": True,
             "bins": classes, "reference": mix + [0.0]}
    mon = monitor_of(pkg, [check])
    same = pd.DataFrame({"label": rng.choice(classes, 3000, p=mix)})
    one = pd.DataFrame({"label": ["a"] * 3000})
    other = pd.DataFrame({"label": rng.choice(classes + ["zz"], 3000, p=[0.35, 0.2, 0.15, 0.3])})
    assert only(mon.report(rows_of(3000), same)).verdict == "ok"
    assert only(mon.report(rows_of(3000), one)).verdict == "alert"
    # an answer outside the classes lands in the rest bin, whose reference share is 0
    r = only(mon.report(rows_of(3000), other))
    assert r.verdict == "alert" and (r.column, r.measure) == ("label", "psi")


def test_label_rate_reads_each_labels_flag(pkg):
    rng = np.random.default_rng(3)
    check = {"id": "l", "kind": "label_rate", "labels": ["red", "green"],
             "fields": {"red": "pred_red", "green": "pred_green"},
             "reference": {"red": 0.3, "green": 0.6}, "reference_rows": 4000}
    mon = monitor_of(pkg, [check])

    def frame(red, green):
        return pd.DataFrame({"pred_red": (rng.random(2000) < red).astype(int),
                             "pred_green": (rng.random(2000) < green).astype(int)})

    readings = {r.column: r for r in mon.report(rows_of(2000), frame(0.3, 0.6))}
    assert set(readings) == {"red", "green"}
    assert all(r.verdict == "ok" and r.measure == "rate" for r in readings.values()), readings
    readings = {r.column: r for r in mon.report(rows_of(2000), frame(0.5, 0.6))}
    assert readings["red"].verdict == "alert" and readings["green"].verdict == "ok"
    assert abs(readings["red"].value - 0.5) < 0.05


def test_alert_rate_watches_a_rise_by_default(pkg):
    rng = np.random.default_rng(4)
    base = {"id": "a", "kind": "alert_rate", "field": "prediction", "reference": 0.05,
            "reference_rows": 4000}

    def frame(rate):
        return pd.DataFrame({"prediction": (rng.random(2000) < rate).astype(int)})

    for check in (base, dict(base, direction="up")):
        mon = monitor_of(pkg, [check])
        assert only(mon.report(rows_of(2000), frame(0.05))).verdict == "ok"
        assert only(mon.report(rows_of(2000), frame(0.2))).verdict == "alert"
        assert only(mon.report(rows_of(2000), frame(0.0))).verdict == "ok", \
            "a fall in alarms is not news: only a rise is a reading"
    both = monitor_of(pkg, [dict(base, direction="both")])
    assert only(both.report(rows_of(2000), frame(0.0))).verdict == "alert"


def _errors(rng, n, scale):
    truth = rng.normal(10, 3, n)
    return truth, truth + rng.normal(0, scale, n)


def test_residual_is_the_mae_against_the_valid_parts(pkg):
    rng = np.random.default_rng(5)
    truth, value = _errors(rng, 5000, 1.0)
    error = np.abs(truth - value)
    check = {"id": "r", "kind": "residual", "field": "value", "target_column": "y",
             "target_offset": 0, "reference": {"mae": float(error.mean()),
                                               "p95": float(np.quantile(error, 0.95))}}
    mon = monitor_of(pkg, [check])
    for scale, verdict in ((1.0, "ok"), (1.35, "watch"), (2.0, "alert")):
        truth, value = _errors(rng, 3000, scale)
        r = only(mon.report(pd.DataFrame({"y": truth}), pd.DataFrame({"value": value})))
        assert (r.column, r.measure, r.verdict) == ("value", "mae", verdict), (scale, r)
        assert "ratio" in window_of(r)["note"] and "p95" in window_of(r)["note"]
    # the window must carry the target column: without it, unmeasured
    r = only(mon.report(pd.DataFrame({"x0": truth}), pd.DataFrame({"value": value})))
    assert r.verdict == "unmeasured" and "'y'" in window_of(r)["note"]


def _two_series(n=400, error=0.0):
    """Two series interleaved row by row; `value_h1` is the next row's truth of the same series
    (plus `error`), what a forecast of step 1 answers."""
    rng = np.random.default_rng(6)
    frame = pd.DataFrame({"s": np.tile(["A", "B"], n // 2), "ts": np.repeat(np.arange(n // 2), 2),
                          "y": rng.normal(0, 5, n)})
    nxt = frame.groupby("s")["y"].shift(-1)
    return frame, pd.DataFrame({"value_h1": nxt + error, "value_h2": nxt})


def test_a_forecast_steps_target_is_h_rows_later_in_its_series(pkg):
    frame, answers = _two_series()
    check = {"id": "f", "kind": "residual", "field": "value_h1", "target_column": "y",
             "target_offset": 1, "series_by": "s", "order_by": "ts",
             "reference": {"mae": 0.5, "p95": 1.0}}
    r = only(monitor_of(pkg, [check]).report(frame, answers))
    assert r.value == 0 and r.verdict == "ok", r
    # shuffled rows answer the same: the series and their order are read, not the positions
    order = np.random.default_rng(7).permutation(len(frame))
    r = only(monitor_of(pkg, [check]).report(frame.iloc[order], answers.iloc[order]))
    assert r.value == 0, r
    # without the series, the next row is the other series' — the error the rule prevents
    blind = {k: v for k, v in check.items() if k not in ("series_by", "order_by")}
    assert only(monitor_of(pkg, [blind]).report(frame, answers)).value > 1


def test_the_v22_kinds_keep_their_readings(pkg):
    rng = np.random.default_rng(8)
    train = pd.Series(rng.normal(0, 1, 4000))
    psi = {"id": "p", "kind": "psi", "column": "x0", **mref._psi_reference({}, train)}
    rate = {"id": "q", "kind": "rate", "name": "big", "reference": 0.16, "reference_rows": 4000,
            "predicate": {"column": "x0", "op": ">", "value": 1.0}}
    unseen = {"id": "u", "kind": "unseen_rate", "column": "zone", "reference": 0.0,
              "reference_rows": 4000, "direction": "up"}
    recall = {"id": "c", "kind": "class_recall", "label_column": "y", "classes": ["0", "1"],
              "reference": {"0": 0.9, "1": 0.8}}
    calls = []

    def recipe(df, history=None, with_unseen=False):
        calls.append((len(df), None if history is None else len(history)))
        return df, tuple(("zone",) if z == "new" else () for z in df["zone"])

    mon = monitor_of(pkg, [psi, rate, unseen, recall], recipe=recipe,
                     labels=pkg.domain.LabelSet((0, 1)))
    window = pd.DataFrame({"x0": rng.normal(0, 1, 2000), "zone": ["old"] * 1990 + ["new"] * 10,
                           "y": ["0"] * 1000 + ["1"] * 990 + ["7"] * 10})
    answers = pd.DataFrame({"label": ["0"] * 1000 + ["1"] * 1000})
    got = {(r.column, r.measure): r for r in mon.report(window, answers)}
    assert got[("x0", "psi")].verdict == "ok" and got[("big", "rate")].verdict == "ok"
    assert got[("zone", "rate")].verdict == "alert" and got[("zone", "rate")].value == 0.005
    assert calls == [(2000, 0)], "the Recipe is asked with an empty history (a lag's start)"
    assert got[("0", "recall")].value == 1.0 and got[("1", "recall")].value == 1.0
    assert "10 row(s) with labels the spec does not list" in window_of(got[("1", "recall")])["note"]
    # the monitor's truth is the spec's classes as text, so int classes read as "0" and "1"
    assert got[("0", "recall")].verdict == "ok"


def test_a_plan_check_without_a_title_is_titled_by_its_kind():
    """release3 final3r1: a model-authored plan named its checks by id and `why` alone, the
    readings shipped with title None, and the bundle's own monitor test failed on them."""
    output = {"reading": "multiclass", "score_field": None,
              "fields": [{"name": "label", "kind": "class"}], "classes": ["a", "b"]}
    for check in ({"id": "p", "kind": "psi", "column": "elevation_m"},
                  {"id": "u", "kind": "unseen_rate", "column": "tile"},
                  {"id": "w", "kind": "rate", "predicate": {"column": "b08", "op": "<", "value": 0.08}},
                  {"id": "c", "kind": "class_recall", "classes": ["a"]}):
        title = mref.resolve_check(check, output)["title"]
        assert title and "None" not in title and "{" not in title, (check, title)
    assert "elevation_m" in mref.resolve_check({"id": "p", "kind": "psi", "column": "elevation_m"},
                                               output)["title"]
    kept = mref.resolve_check({"id": "p", "kind": "psi", "column": "x", "title": "mine"}, output)
    assert kept["title"] == "mine"


def test_a_window_recalling_what_valid_did_is_not_a_drop(pkg):
    """release3 final3r1: valid recalled 30 of 31, the pack stored round(30/31, 6) = 0.967742,
    and an identical window read 0.9677419... < 0.967742 — a watch on no change at all."""
    ref = round(30 / 31, 6)
    recall = {"id": "c", "kind": "class_recall", "label_column": "y", "classes": ["a"],
              "reference": {"a": ref}}
    mon = monitor_of(pkg, [recall], labels=pkg.domain.LabelSet(("a", "b")))
    same = pd.DataFrame({"y": ["a"] * 31})
    got = only(mon.report(same, pd.DataFrame({"label": ["a"] * 30 + ["b"]})))
    assert got.value == ref and got.verdict == "ok", got
    worse = only(mon.report(same, pd.DataFrame({"label": ["a"] * 29 + ["b"] * 2})))
    assert worse.verdict == "watch", worse


def test_unanswered_rows_are_dropped_before_any_reading_with_the_note(pkg):
    rng = np.random.default_rng(9)
    reference = mref._psi_reference({"bins_n": 10}, pd.Series(rng.normal(0, 1, 4000)))
    checks = [{"id": "o", "kind": "output_psi", "field": "value", **reference},
              {"id": "r", "kind": "residual", "field": "value", "target_column": "y",
               "reference": {"mae": 1.0, "p95": 2.0}},
              {"id": "p", "kind": "psi", "column": "x0", **reference}]
    n, bad = 3000, 300
    # the answered rows are drawn like the reference (value and x0 ~ N(0, 1), |y − value| of
    # MAE ≈ 0.8 against the reference's 1.0); the unanswered ones would move every reading
    answered = np.arange(n) >= bad
    value = np.where(answered, rng.normal(0, 1, n), 1e6)
    truth = value + rng.normal(0, 1, n)
    window = pd.DataFrame({"x0": np.where(answered, rng.normal(0, 1, n), 1e6), "y": truth})
    answers = pd.DataFrame({"value": value, "answered": answered})
    mon = monitor_of(pkg, checks)
    readings = mon.report(window, answers)
    assert [r.verdict for r in readings] == ["ok", "ok", "ok"], readings
    for r in readings:
        w = window_of(r)
        assert w["rows"] == n - bad, w
        assert f"{bad} row(s) answered false" in w["note"], w
    # the same window with those rows removed reads the same values
    kept = mon.report(window.iloc[bad:], answers.iloc[bad:].drop(columns=["answered"]))
    assert [r.value for r in kept] == [r.value for r in readings]
    assert all("note" not in window_of(r) or "answered false" not in window_of(r)["note"]
               for r in kept)


def test_only_an_explicit_false_leaves_a_row_out(pkg):
    """`answered` defaults to true: a numpy false and a Python false leave the row out; true,
    a missing value or a column the Predictor did not send keep it."""
    check = {"id": "a", "kind": "alert_rate", "field": "prediction", "reference": 0.5,
             "min_rows": 1}
    mon = monitor_of(pkg, [check])
    answers = pd.DataFrame({"prediction": [1, 1, 0, 0, 1],
                            "answered": pd.Series([np.False_, False, True, None, np.True_],
                                                  dtype=object)})
    r = only(mon.report(rows_of(5), answers))
    assert window_of(r)["rows"] == 3 and "2 row(s) answered false" in window_of(r)["note"]
    assert r.value == pytest.approx(1 / 3), "the kept rows answer 0, 0, 1"
    r = only(mon.report(rows_of(5), answers.drop(columns=["answered"])))
    assert window_of(r)["rows"] == 5
    assert "answered false" not in window_of(r).get("note", "")


def test_a_recipe_that_cannot_read_the_window_leaves_the_unseen_rate_unmeasured(pkg):
    def lagging(df, history=None, with_unseen=False):
        assert history is not None and len(history) == 0
        raise KeyError("the history rows do not carry y")

    check = {"id": "u", "kind": "unseen_rate", "column": "zone", "reference": 0.0,
             "reference_rows": 100}
    r = only(monitor_of(pkg, [check], recipe=lagging).report(pd.DataFrame({"zone": ["a"] * 200})))
    assert (r.column, r.verdict, r.value) == ("zone", "unmeasured", None)
    assert "the Recipe could not read the window" in window_of(r)["note"]


def test_the_monitor_refuses_what_it_cannot_measure(pkg):
    good = {"id": "o", "kind": "output_psi", "field": "value", "bins": [None, 0, None],
            "reference": [0.5, 0.5]}
    with pytest.raises(ValueError, match="no checks"):
        monitor_of(pkg, [])
    with pytest.raises(ValueError, match="unknown check kind"):
        monitor_of(pkg, [dict(good, kind="nope")])
    with pytest.raises(ValueError, match="without recipe"):
        pkg.monitor.Monitor({"checks": [{"id": "u", "kind": "unseen_rate", "column": "z"}]},
                            recipe=None, labels=pkg.domain.LabelSet(()))
    with pytest.raises(ValueError, match="without labels"):
        pkg.monitor.Monitor({"checks": [{"id": "c", "kind": "class_recall", "classes": []}]},
                            recipe=None, labels=None)
    with pytest.raises(ValueError, match="monitor-reference@3"):
        pkg.monitor.Monitor({"schema": "vega/monitor-reference@3", "checks": [good]},
                            recipe=None, labels=None)
    mon = monitor_of(pkg, [good])
    with pytest.raises(TypeError, match="DataFrame"):
        mon.check([{"x0": 1}])
    with pytest.raises(TypeError, match="Predictor's frame"):
        mon.check(rows_of(3), [{"value": 1}] * 3)
    with pytest.raises(ValueError, match="one row per row"):
        mon.check(rows_of(3), pd.DataFrame({"value": [1.0, 2.0]}))
    # a v2.2 reference (the four input kinds) still loads
    assert pkg.monitor.Monitor({"schema": "vega/monitor-reference@1", "checks": [good]},
                               recipe=None, labels=None)


# ================================================================ 3. validate_plan

def _problems(name, *checks, schema=SCHEMA):
    return monitor_pack.validate_plan({"checks": list(checks)}, output_of(name), schema)


def _one_problem(name, check, fragment):
    problems = _problems(name, check)
    assert len(problems) == 1 and fragment in problems[0], problems
    return problems[0]


def test_a_plan_the_bundle_can_honour_has_no_problem():
    assert _problems("binary",
                     {"id": "a", "kind": "psi", "column": "x0", "concern": "[analysis:F3]"},
                     {"id": "b", "kind": "rate",
                      "predicate": {"any": [{"column": "x1", "op": ">", "value": 1},
                                            {"column": "zone", "op": "==", "value": 3}]}},
                     {"id": "c", "kind": "unseen_rate", "column": "zone"},
                     {"id": "d", "kind": "class_recall", "classes": ["0", "1"],
                      "label_column": "y"},
                     {"id": "e", "kind": "output_psi", "field": "p_1"},
                     {"id": "f", "kind": "prediction_mix"}) == []
    assert _problems("forecast", {"id": "r", "kind": "residual", "field": "value_h2"}) == []
    assert _problems("multilabel", {"kind": "label_rate", "labels": ["red", "blue"]}) == []
    assert _problems("anomaly-svdd", {"kind": "alert_rate"},
                     {"kind": "output_psi", "field": "score"}) == []


def test_an_unknown_kind_is_refused():
    _one_problem("binary", {"id": "k", "kind": "drift_magic", "column": "x0"},
                 "check k: unknown kind 'drift_magic'")
    # a malformed plan is a list of problems, never a traceback
    assert _problems("binary", {"id": "k", "kind": ["psi"]}, "not a check",
                     {"id": "c", "kind": "psi", "column": ["x0"]},
                     {"id": "f", "kind": "output_psi", "field": {"p": 1}},
                     {"id": "l", "kind": "class_recall", "classes": "01"}) == [
        "check k: unknown kind ['psi'] (the kinds are " + ", ".join(CATALOG["monitor_kinds"])
        + ")", "checks[1] is not an object",
        "check c: column ['x0'] is not an input column of schema.json",
        "check f: field {'p': 1} is not one of MANIFEST.output.fields (label, p_0, p_1, score, "
        "prediction, warnings)", "check l: classes is a list of names, not '01'"]


@pytest.mark.parametrize("name,kind", [("regression", "class_recall"),
                                       ("multiclass", "output_psi"),
                                       ("multilabel", "prediction_mix"),
                                       ("binary", "label_rate"), ("regression", "alert_rate"),
                                       ("binary", "residual"),
                                       ("reconstruction", "class_recall")])
def test_a_kind_the_reading_does_not_admit_is_refused(name, kind):
    problem = _one_problem(name, {"id": "n", "kind": kind, "field": "value", "classes": ["a"],
                                  "labels": ["a"]}, "not admitted")
    assert f"the {READINGS[name]} reading" in problem, problem


def test_a_missing_parameter_is_refused():
    _one_problem("binary", {"id": "m", "kind": "psi"}, "psi needs column")
    _one_problem("binary", {"id": "m", "kind": "rate"}, "rate needs predicate")
    _one_problem("multiclass", {"id": "m", "kind": "class_recall"}, "class_recall needs classes")
    _one_problem("multilabel", {"id": "m", "kind": "label_rate", "labels": []},
                 "label_rate needs labels")
    # a reading whose answer has no field the kind reads by default: named, with its cause
    problems = _problems("anomaly-unlabelled", {"id": "m", "kind": "class_recall",
                                                "classes": ["0"]})
    assert problems == ["check m: class_recall needs field",
                        "check m: class_recall reads a class truth, and this bundle takes no "
                        "truth"], problems


def test_a_field_the_plan_leaves_out_is_the_kinds_default():
    """`output_psi` and `residual` read the record's score field, `prediction_mix` and
    `class_recall` the decided class, `alert_rate` the alarm; classes are read as text."""
    resolved = [(name, mref.resolve_check(check, output_of(name))) for name, check in (
        ("binary", {"kind": "output_psi"}), ("forecast", {"kind": "residual"}),
        ("quantile", {"kind": "residual"}), ("multiclass", {"kind": "prediction_mix"}),
        ("binary", {"kind": "class_recall", "classes": [0, 1]}),
        ("anomaly-svdd", {"kind": "alert_rate"}))]
    assert [(n, c["field"]) for n, c in resolved] == [
        ("binary", "p_1"), ("forecast", "value_h1"), ("quantile", "value"),
        ("multiclass", "label"), ("binary", "label"), ("anomaly-svdd", "prediction")]
    assert resolved[4][1]["classes"] == ["0", "1"]
    for name, check in resolved:
        assert _problems(name, dict(check, id="d")) == [], (name, check)
    # a named field wins over the default
    assert mref.resolve_check({"kind": "residual", "field": "value_h2"},
                              output_of("forecast"))["field"] == "value_h2"


def test_an_unknown_column_field_class_or_label_is_refused():
    _one_problem("binary", {"id": "c", "kind": "psi", "column": "nope"},
                 "column 'nope' is not an input column of schema.json")
    _one_problem("binary", {"id": "c", "kind": "rate",
                            "predicate": {"all": [{"column": "x0", "op": ">", "value": 0},
                                                  {"column": "gone", "op": ">", "value": 0}]}},
                 "column 'gone'")
    _one_problem("binary", {"id": "f", "kind": "output_psi", "field": "p_2"},
                 "field 'p_2' is not one of MANIFEST.output.fields")
    _one_problem("binary", {"id": "f", "kind": "output_psi", "field": "label"},
                 "check f: output_psi reads the record's score_field 'p_1', not 'label'")
    _one_problem("quantile-no-median", {"id": "f", "kind": "residual", "field": "value"},
                 "field 'value' is not one of")
    _one_problem("quantile", {"id": "f", "kind": "residual", "field": "q_10"},
                 "field 'q_10' is a quantile field; residual reads a value or horizon field")
    _one_problem("multiclass", {"id": "k", "kind": "class_recall", "classes": ["a", "z"]},
                 "class or label 'z' is not one of the output's classes (a, b, c)")
    _one_problem("multilabel", {"id": "k", "kind": "label_rate", "labels": ["red", "pink"]},
                 "class or label 'pink'")


def test_a_predicate_the_monitor_cannot_read_is_refused():
    problems = _problems("binary", {"id": "q", "kind": "rate", "predicate": {"any": [
        {"column": "x0", "op": "~=", "value": 1}, {"column": "x1", "op": ">", "value": "big"},
        {"column": "x2", "op": ">", "value": True}, "x3 > 1"]}})
    assert problems == ["check q: predicate op '~=' is not one of ==, !=, <, <=, >, >=",
                        "check q: predicate value 'big' is not a number",
                        "check q: predicate value True is not a number",
                        "check q: a predicate term is not an object {column, op, value}"]
    _one_problem("binary", {"id": "q", "kind": "rate", "predicate": {"all": []}},
                 "a predicate term is not an object")


def test_a_forecast_residual_reads_the_point_quantile():
    """Under a quantile loss a forecast's point answer is the median's field of each step;
    the residual of another quantile is not an error of the forecast, and is refused."""
    assert _problems("forecast-quantile", {"id": "r", "kind": "residual"}) == []
    assert _problems("forecast-quantile",
                     {"id": "r", "kind": "residual", "field": "value_h2_q50"}) == []
    _one_problem("forecast-quantile", {"id": "r", "kind": "residual", "field": "value_h1_q10"},
                 "field 'value_h1_q10' is the 0.1 quantile of step 1; residual reads the point "
                 "forecast, the 0.5 quantile's field")
    assert mref.horizon_of(output_of("forecast-quantile")["field_kinds"]["value_h2_q50"]) == 2


def test_a_truth_or_threshold_the_bundle_does_not_record_is_refused():
    _one_problem("binary", {"id": "t", "kind": "class_recall", "classes": ["1"],
                            "label_column": "other"}, "the truth column is 'y', not 'other'")
    problems = _problems("anomaly-unlabelled", {"id": "t", "kind": "class_recall",
                                                "classes": ["0"], "field": "score"})
    assert any("reads a class truth, and this bundle takes no truth" in p for p in problems)
    _one_problem("forecast", {"id": "t", "kind": "residual", "target_column": "other"},
                 "the truth column is 'y', not 'other'")
    _one_problem("anomaly-undecided", {"id": "t", "kind": "alert_rate"},
                 "records no threshold")


def test_duplicate_and_reserved_ids_are_refused():
    problems = _problems("regression", {"id": "a", "kind": "psi", "column": "x0"},
                         {"id": "a", "kind": "psi", "column": "x1"},
                         {"id": "auto-output", "kind": "psi", "column": "x2"})
    assert problems == ["check id 'a' is used twice",
                        "check id 'auto-output' is the output check pack adds itself"]
    assert monitor_pack.validate_plan({"checks": []}, output_of("binary"), SCHEMA) == \
        ["the plan carries no checks"]
    # an id that is not a string is named, before the duplicates are counted (an unhashable
    # id raised a TypeError there)
    assert _problems("binary", {"id": ["a"], "kind": "psi", "column": "x0"},
                     {"id": ["a"], "kind": "psi", "column": "x0"}, {"id": 7, "kind": "psi",
                                                                    "column": "x1"}) == [
        "checks[0]: id ['a'] is not a string", "checks[1]: id ['a'] is not a string",
        "checks[2]: id 7 is not a string"]


def test_output_psi_reads_the_score_field_alone():
    """bundle-contract §7.5 and the catalog (`monitor_kinds.output_psi.reads: score_field`):
    `output_psi` watches the record's score field, the one the automatic check reads; another
    answer field is refused, however numeric."""
    assert monitor_plan.SCORE_READERS == ("output_psi",)
    admitted = [n for n in sorted(CASES)
                if "output_psi" in monitor_kinds_for(output_of(n)["reading"])]
    assert len(admitted) == 11, admitted
    for name in admitted:
        output = output_of(name)
        score = output["score_field"]
        assert _problems(name, {"id": "o", "kind": "output_psi"}) == [], name
        assert _problems(name, {"id": "o", "kind": "output_psi", "field": score}) == [], name
        for field in (f for f in output["field_kinds"] if f != score):
            _one_problem(name, {"id": "o", "kind": "output_psi", "field": field},
                         f"check o: output_psi reads the record's score_field {score!r}, not "
                         f"{field!r}")
    _one_problem("binary", {"id": "o", "kind": "output_psi", "field": "p_0"},
                 "output_psi reads the record's score_field 'p_1', not 'p_0'")
    _one_problem("anomaly-svdd", {"id": "o", "kind": "output_psi", "field": "emb_0"},
                 "output_psi reads the record's score_field 'score', not 'emb_0'")


def test_malformed_values_are_problems_not_tracebacks():
    """Every parameter the monitor reads as a number, a limit or a direction is checked as the
    monitor reads it: a plan that would raise at measurement (`int('ten')`) or in the smoke
    (`float('high')`, a list of thresholds) is refused here, exit 2."""
    rate = {"column": "x1", "op": ">", "value": 0}
    problems = _problems(
        "binary",
        {"id": "b", "kind": "output_psi", "bins_n": "ten"},
        {"id": "t", "kind": "psi", "column": "x0", "thresholds": [1]},
        {"id": "u", "kind": "psi", "column": "x0", "thresholds": {"watch": "low", "warn": 0.2}},
        {"id": "z", "kind": "rate", "predicate": rate, "watch_z": "high", "alert_z": None,
         "min_rows": 0},
        {"id": "k", "kind": "psi", "column": "zone", "categorical": True, "top_k": 2.5,
         "epsilon": 0},
        {"id": "d", "kind": "unseen_rate", "column": "zone", "direction": "sideways"},
        {"id": "ok", "kind": "psi", "column": "x0", "bins_n": 5, "thresholds": {"alert": 0.3},
         "epsilon": 1e-5, "min_rows": 50.0},
        {"id": "r", "kind": "rate", "predicate": rate, "watch_z": 1.5, "alert_z": 2,
         "direction": "both"})
    assert problems == [
        "check b: bins_n 'ten' is not a whole number of at least 1",
        "check t: thresholds [1] is not an object {watch, alert}",
        "check u: thresholds names 'warn'; its limits are watch and alert",
        "check u: thresholds.watch 'low' is not a number",
        "check z: min_rows 0 is not a whole number of at least 1",
        "check z: watch_z 'high' is not a number", "check z: alert_z None is not a number",
        "check k: top_k 2.5 is not a whole number of at least 1",
        "check k: epsilon 0 is not a number above 0",
        "check d: direction 'sideways' is not one of up, down, both"], problems
    _one_problem("regression", {"id": "r", "kind": "residual", "watch_ratio": [1.2]},
                 "check r: watch_ratio [1.2] is not a number")
    # a NaN or an Infinity anywhere: the strict-JSON reference could not carry it
    for bad in (math.nan, math.inf):
        assert _problems("binary", {"id": "n", "kind": "psi", "column": "x0", "note": bad}) == [
            "the plan is not strict JSON (a NaN, an Infinity): the reference it becomes could "
            "not carry it"]


# ================================================================ 5. the automatic output check

EXPECTED_OUTPUT_CHECK = {
    "binary": ("output_psi", "p_1"), "multiclass": ("prediction_mix", "label"),
    "multilabel": ("label_rate", None), "ordinal": ("prediction_mix", "label"),
    "regression": ("output_psi", "value"), "quantile": ("output_psi", "value"),
    "quantile-no-median": ("output_psi", "q_25"), "forecast": ("output_psi", "value_h1"),
    "forecast-quantile": ("output_psi", "value_h1_q50"),
    "reconstruction": ("output_psi", "reconstruction_error"),
    "anomaly-reconstruction": ("output_psi", "score"),
    "anomaly-unlabelled": ("output_psi", "score"), "anomaly-svdd": ("output_psi", "score"),
    "anomaly-undecided": ("output_psi", "score")}


@pytest.mark.parametrize("name", sorted(CASES))
def test_every_reading_gets_an_output_check_that_its_plan_would_pass(name):
    output = output_of(name)
    check = monitor_pack.output_check(output)
    kind, field = EXPECTED_OUTPUT_CHECK[name]
    # the score field where the record names one (`<median>`: the quantile nearest 0.5 when
    # 0.5 is not one of them), else the decided classes' mix, else the labels' rates
    assert output["score_field"] == (field if kind == "output_psi" else None), output
    assert (check["kind"], check.get("field")) == (kind, field), check
    assert check["id"] == "auto-output" and check["concern"] == "[vega-pack:output-drift]"
    assert check["title"].endswith("distribution against the valid part")
    if kind == "label_rate":
        assert check["labels"] == ["red", "green", "blue"]
    # admissible for the reading: the plan's own gate passes it (under an id a plan may use;
    # `auto-output` is reserved for this check)
    assert monitor_pack.validate_plan({"checks": [dict(check, id="probe")]}, output, SCHEMA) \
        == []


def test_the_output_check_is_added_once():
    output, markers = output_of("binary"), []
    plan = [{"id": "a", "kind": "psi", "column": "x0"}]
    added = monitor_pack._with_output_check(list(plan), output, markers)
    assert [c["id"] for c in added] == ["a", "auto-output"]
    mine = plan + [{"id": "mine", "kind": "output_psi", "field": "p_1"}]
    assert monitor_pack._with_output_check(list(mine), output, markers) == mine
    assert any("already watches the output" in m for m in markers)


# ================================================================ 4. references on valid only

class Answering:
    """A Predictor stand-in: its reader is the bundle's own (or none), and `predict` answers
    each raw row through `answer`; it records the rows it was asked about."""

    def __init__(self, answer, reader=None):
        self._answer, self.seen = answer, []
        if reader is not None:
            self.reader = reader

    def predict(self, rows):
        self.seen.append(rows.copy())
        return self._answer(rows)


def _refusing(_rows):
    raise AssertionError("the stored valid predictions answer this reading; predict is not "
                         "called")


def _raw(rng, n, rid0):
    return pd.DataFrame({"rid": np.arange(rid0, rid0 + n), "x0": rng.normal(0, 1, n),
                         "x1": rng.normal(0, 1, n), "x2": rng.normal(0, 1, n),
                         "x3": rng.normal(0, 1, n), "zone": rng.choice(["p", "q", "r"], n),
                         "y": rng.integers(0, 2, n)})


def _unseen_recipe(df, history=None, with_unseen=False):
    return df, tuple(("zone",) if z == "r" else () for z in df["zone"])


def _inputs(pkg, name, *, train, valid, stored, predictor, sample=None, ignored=("y",)):
    spec, _ = CASES[name]
    return mref.MonitorInputs(train=train, valid=valid, stored_valid=stored,
                              predictor=predictor, recipe=lambda: _unseen_recipe,
                              classes=tuple(str(c) for c in spec["task"].get("classes") or ()),
                              label="y", decode={}, ignored=ignored,
                              sample=sample or {"shape": "row"})


BINARY_PLAN = [{"id": "p", "kind": "psi", "column": "x0"},
               {"id": "q", "kind": "rate", "predicate": {"column": "x1", "op": ">", "value": 0}},
               {"id": "u", "kind": "unseen_rate", "column": "zone"},
               {"id": "c", "kind": "class_recall", "classes": ["0", "1"]},
               {"id": "o", "kind": "output_psi", "field": "p_1"},
               {"id": "m", "kind": "prediction_mix", "field": "label"}]


def _binary_reference(pkg, train, valid, stored):
    predictor = Answering(_refusing, reader_of(pkg, "binary"))
    inputs = _inputs(pkg, "binary", train=train, valid=valid, stored=stored, predictor=predictor)
    doc = mref.MonitorRender(output_of("binary"), SCHEMA, inputs).render(
        {"checks": copy.deepcopy(BINARY_PLAN)})
    return {c["id"]: c for c in doc["checks"]}, doc


def test_answer_references_are_read_on_the_stored_valid_predictions(pkg):
    rng = np.random.default_rng(10)
    train, valid = _raw(rng, 3000, 0), _raw(rng, 1000, 3000)
    stored = pd.DataFrame({"row_id": valid["rid"], "score": rng.normal(0, 2, 1000),
                           "y": valid["y"].to_numpy()})
    checks, doc = _binary_reference(pkg, train, valid, stored)
    assert doc["schema"] == "vega/monitor-reference@2" and doc["reading"] == "binary"
    parts = {k: (c["reference_part"], c["reference_rows"]) for k, c in checks.items()}
    assert parts == {"p": ("train", 3000), "q": ("train", 3000), "u": ("valid", 1000),
                     "c": ("valid", 1000), "o": ("valid", 1000), "m": ("valid", 1000)}
    p1 = sigmoid(stored["score"].to_numpy())
    decided = np.where(p1 >= 0.5, "1", "0")
    truth = stored["y"].astype(str).to_numpy()
    for k in ("0", "1"):
        want = ((truth == k) & (decided == k)).sum() / (truth == k).sum()
        assert checks["c"]["reference"][k] == pytest.approx(want, abs=1e-6)
        assert checks["c"]["reference_support"][k] == int((truth == k).sum())
    assert checks["c"]["label_column"] == "y"
    assert checks["m"]["bins"] == ["0", "1"] and checks["m"]["categorical"]
    assert checks["m"]["reference"][:2] == pytest.approx([(decided == "0").mean(),
                                                          (decided == "1").mean()])
    bins = checks["o"]["bins"]
    assert bins[0] is None and bins[-1] is None, "the outer edges are open, written null"
    edges = np.asarray([-np.inf, *bins[1:-1], np.inf], dtype=float)
    counts = np.histogram(p1, bins=edges)[0] / len(p1)
    assert counts == pytest.approx(checks["o"]["reference"], abs=1e-6)
    assert checks["u"]["reference"] == pytest.approx((valid["zone"] == "r").mean())
    assert checks["q"]["reference"] == pytest.approx((train["x1"] > 0).mean())


def test_changing_train_moves_only_the_input_references(pkg):
    rng = np.random.default_rng(11)
    train, valid = _raw(rng, 3000, 0), _raw(rng, 1000, 3000)
    stored = pd.DataFrame({"score": rng.normal(0, 2, 1000), "y": valid["y"].to_numpy()})
    before, _ = _binary_reference(pkg, train, valid, stored)
    shifted = train.assign(x0=train["x0"] + 3, x1=train["x1"] + 3,
                           zone=["r"] * len(train), y=1)
    after, _ = _binary_reference(pkg, shifted, valid, stored)
    for k in ("u", "c", "o", "m"):
        assert after[k] == before[k], k
    assert after["q"]["reference"] != before["q"]["reference"]
    assert after["p"]["bins"] != before["p"]["bins"]
    moved, _ = _binary_reference(pkg, train, valid, stored.assign(score=stored["score"] + 2))
    # quantile bins hold equal shares by construction: a moved answer moves their edges
    assert moved["o"]["bins"] != before["o"]["bins"]
    assert moved["m"]["reference"] != before["m"]["reference"]


def test_without_stored_predictions_the_predictor_answers_the_valid_rows_only(pkg):
    rng = np.random.default_rng(12)
    train, valid = _raw(rng, 3000, 0), _raw(rng, 800, 3000)
    valid["y"] = valid["x0"] * 2 + rng.normal(0, 0.5, len(valid))

    def answer(rows):
        return pd.DataFrame({"value": rows["x0"].to_numpy() * 2, "warnings": ""})

    predictor = Answering(answer)                     # no reader: a Predictor without one
    inputs = _inputs(pkg, "regression", train=train, valid=valid, stored=None,
                     predictor=predictor)
    plan = {"checks": [{"id": "r", "kind": "residual", "field": "value"},
                       {"id": "o", "kind": "output_psi", "field": "value"}]}
    checks = {c["id"]: c for c in
              mref.MonitorRender(output_of("regression"), SCHEMA, inputs).render(plan)["checks"]}
    assert len(predictor.seen) == 1, "the valid part's answers are computed once"
    asked = predictor.seen[0]
    assert asked["rid"].tolist() == valid["rid"].tolist(), "only the valid part's raw rows"
    assert "y" not in asked.columns, "the schema's ignored columns are left out of predict"
    error = np.abs(valid["y"] - valid["x0"] * 2)
    assert checks["r"]["reference"]["mae"] == pytest.approx(error.mean())
    assert checks["r"]["reference"]["p95"] == pytest.approx(np.quantile(error, 0.95))
    assert (checks["r"]["target_column"], checks["r"]["target_offset"]) == ("y", 0)
    assert checks["o"]["reference_part"] == "valid"


def _forecast_valid(error):
    frame, _ = _two_series(600)
    frame["x0"], frame["rid"] = 0.0, np.arange(len(frame))

    def answer(rows):
        nxt = rows.groupby("s")["y"].shift(-1)
        second = rows.groupby("s")["y"].shift(-2)
        answered = rows.groupby("s").cumcount().to_numpy() >= 3
        return pd.DataFrame({"value_h1": np.where(answered, nxt + error, 1e6),
                             "value_h2": np.where(answered, second, 1e6),
                             "answered": answered, "warnings": ""})

    return frame, answer


def test_the_pack_and_the_monitor_read_a_forecast_step_alike(pkg):
    """The residual reference of step h is |truth h rows later in the series − value_h| over the
    answered valid rows; the Monitor's reading of the same window is the same number."""
    valid, answer = _forecast_valid(error=0.5)
    sample = {"shape": "window", "series_by": "s", "order_by": "ts"}
    # an autoregressive forecast reads its target as an input (R-A7): the schema ignores it not
    inputs = _inputs(pkg, "forecast", train=valid, valid=valid, stored=None,
                     predictor=Answering(answer), sample=sample, ignored=())
    plan = {"checks": [{"id": "h1", "kind": "residual", "field": "value_h1"},
                       {"id": "h2", "kind": "residual", "field": "value_h2"}]}
    checks = mref.MonitorRender(output_of("forecast"), SCHEMA, inputs).render(plan)["checks"]
    h1, h2 = checks
    assert (h1["target_offset"], h2["target_offset"]) == (1, 2)
    assert (h1["series_by"], h1["order_by"]) == ("s", "ts")
    # the 1e6 answers of the unanswered rows are left out: step 1 is off by 0.5 exactly
    assert h1["reference"]["mae"] == pytest.approx(0.5) and h2["reference"]["mae"] == 0
    readings = monitor_of(pkg, checks).report(valid, answer(valid))
    assert readings[0].value == pytest.approx(0.5) and readings[0].verdict == "ok"
    assert "ratio 1.000" in window_of(readings[0])["note"]
    worse, answer2 = _forecast_valid(error=1.0)
    assert monitor_of(pkg, checks[:1]).report(worse, answer2(worse))[0].verdict == "alert"


def test_label_and_alert_rate_references_are_the_valid_parts_shares(pkg):
    rng = np.random.default_rng(15)
    train, valid = _raw(rng, 600, 0), _raw(rng, 400, 600)
    logits = rng.normal(0, 2, (400, 3))
    stored = pd.DataFrame({f"score_{k}": logits[:, k] for k in range(3)})
    inputs = _inputs(pkg, "multilabel", train=train, valid=valid, stored=stored,
                     predictor=Answering(_refusing, reader_of(pkg, "multilabel")))
    check = mref.MonitorRender(output_of("multilabel"), SCHEMA, inputs).render(
        {"checks": [{"id": "l", "kind": "label_rate", "labels": ["red", "blue"]}]})["checks"][0]
    assert check["fields"] == {"red": "pred_red", "blue": "pred_blue"}
    assert check["reference"]["red"] == pytest.approx((sigmoid(logits[:, 0]) >= 0.5).mean())
    assert check["reference"]["blue"] == pytest.approx((sigmoid(logits[:, 2]) >= 0.5).mean())
    # an anomaly reading needs the per-row score, which this store lacks: the Predictor
    # answers the valid rows, and the alarm share is read on its frame
    predictor = Answering(lambda rows: pd.DataFrame(
        {"score": rows["x0"].to_numpy(), "prediction": (rows["x0"] >= 0.8).astype(int),
         "warnings": ""}), reader_of(pkg, "anomaly-svdd"))
    inputs = _inputs(pkg, "anomaly-svdd", train=train, valid=valid, predictor=predictor,
                     stored=pd.DataFrame({f"score_{k}": logits[:, k] for k in range(3)}))
    check = mref.MonitorRender(output_of("anomaly-svdd"), SCHEMA, inputs).render(
        {"checks": [{"id": "a", "kind": "alert_rate"}]})["checks"][0]
    assert (check["field"], check["direction"], check["reference_part"]) == \
        ("prediction", "up", "valid")
    assert check["reference"] == pytest.approx((valid["x0"] >= 0.8).mean())
    assert len(predictor.seen) == 1 and predictor.seen[0]["rid"].tolist() == \
        valid["rid"].tolist()


def test_a_truth_the_store_lacks_is_read_from_the_valid_rows(pkg):
    """A one-column store (`y`: step 1's truth) answers the residual of step 1; step 2's
    truth is not stored, so that check is answered by the Predictor over the valid rows."""
    valid, answer = _forecast_valid(error=0.0)
    answers = answer(valid)
    kept = answers["answered"].to_numpy()
    stored = pd.DataFrame({"score_0": answers["value_h1"][kept].to_numpy() + 0.25,
                           "score_1": answers["value_h2"][kept].to_numpy(),
                           "y": valid.groupby("s")["y"].shift(-1)[kept].to_numpy()})
    predictor = Answering(answer, reader_of(pkg, "forecast"))
    inputs = _inputs(pkg, "forecast", train=valid, valid=valid, stored=stored,
                     predictor=predictor, ignored=(),
                     sample={"shape": "window", "series_by": "s", "order_by": "ts"})
    h1, h2 = mref.MonitorRender(output_of("forecast"), SCHEMA, inputs).render(
        {"checks": [{"id": "h1", "kind": "residual", "field": "value_h1"},
                    {"id": "h2", "kind": "residual", "field": "value_h2"}]})["checks"]
    assert h1["reference"]["mae"] == pytest.approx(0.25), "step 1 from the store"
    assert h2["reference"]["mae"] == pytest.approx(0.0), "step 2 from the Predictor"
    assert len(predictor.seen) == 1


def test_a_field_the_answer_lacks_is_a_refusal_not_a_guess(pkg):
    rng = np.random.default_rng(13)
    train, valid = _raw(rng, 500, 0), _raw(rng, 200, 500)

    def v22(rows):                                   # a v2.2-shaped frame: no p_<class>
        return pd.DataFrame({"score": np.zeros(len(rows)), "prediction": 0})

    inputs = _inputs(pkg, "binary", train=train, valid=valid, stored=None,
                     predictor=Answering(v22))
    with pytest.raises(PackExit) as refusal:
        mref.MonitorRender(output_of("binary"), SCHEMA, inputs).render(
            {"checks": [{"id": "o", "kind": "output_psi", "field": "p_1"}]})
    assert refusal.value.code == 3 and "no field 'p_1'" in refusal.value.lines[0]
    empty = _inputs(pkg, "binary", train=train, valid=valid.iloc[:0], stored=None,
                    predictor=Answering(v22))
    with pytest.raises(PackExit) as refusal:
        mref.MonitorRender(output_of("binary"), SCHEMA, empty).render(
            {"checks": [{"id": "u", "kind": "unseen_rate", "column": "zone"}]})
    assert refusal.value.code == 2 and "no valid rows" in refusal.value.lines[0]


def _binary_frame(pkg, logits):
    """The binary reader's frame over these logits: what the Predictor answers."""
    head = pkg.samples.HeadOutput(np.asarray(logits, dtype=float).reshape(-1, 1), None)
    return reader_of(pkg, "binary").frame(head).assign(warnings="")


def test_a_graph_level_bundles_answers_are_read_per_row(pkg):
    """A graph-level store holds one row per graph; the served frame one per raw row, the rows
    of a graph sharing its answer, and the Monitor reads that frame by row. The answer
    references are the Predictor's per-row frame, so a graph weighs as many rows in the
    reference as in a window; a node-level store (one row per node) is read as stored."""
    rng = np.random.default_rng(16)
    train, valid = _raw(rng, 600, 0), _raw(rng, 800, 600)
    # 40 graphs, the large ones answering high: per graph and per row disagree
    sizes = np.where(np.arange(40) < 10, 50, 10)
    graph = np.repeat(np.arange(40), sizes)
    logit = np.linspace(-3, 3, 40)
    valid = valid.assign(g=graph[:len(valid)])
    stored = pd.DataFrame({"row_id": np.arange(40), "score": logit, "y": 0})
    predictor = Answering(lambda rows: _binary_frame(pkg, logit[rows["g"].to_numpy()]),
                          reader_of(pkg, "binary"))
    per_graph = {"shape": "graph", "level": "graph", "graph_id": "g"}
    inputs = _inputs(pkg, "binary", train=train, valid=valid, stored=stored,
                     predictor=predictor, sample=per_graph)
    check = mref.MonitorRender(output_of("binary"), SCHEMA, inputs).render(
        {"checks": [{"id": "o", "kind": "output_psi"}]})["checks"][0]
    assert len(predictor.seen) == 1 and check["reference_rows"] == len(valid) == 800
    per_row = mref._psi_reference({"bins_n": 10}, pd.Series(sigmoid(logit[graph])))
    assert (check["bins"], check["reference"]) == (per_row["bins"], per_row["reference"])
    by_graph = mref._psi_reference({"bins_n": 10}, pd.Series(sigmoid(logit)))
    assert by_graph["bins"] != per_row["bins"], "the weighting is what the test tells apart"
    # node level: one stored row per node, read as stored, and no predict call
    nodes = pd.DataFrame({"row_id": valid["rid"], "score": logit[graph], "y": 0})
    inputs = _inputs(pkg, "binary", train=train, valid=valid, stored=nodes,
                     predictor=Answering(_refusing, reader_of(pkg, "binary")),
                     sample=dict(per_graph, level="node"))
    check = mref.MonitorRender(output_of("binary"), SCHEMA, inputs).render(
        {"checks": [{"id": "o", "kind": "output_psi"}]})["checks"][0]
    assert (check["bins"], check["reference"]) == (per_row["bins"], per_row["reference"])


# ================================================================ render, end to end

PREPROCESS = '''
def transform(df_raw, history=None, with_unseen=False):
    out = df_raw.copy()
    unseen = tuple(("zone",) if z == "r" else () for z in df_raw["zone"])
    return (out, unseen) if with_unseen else out
'''


def _ctx(pkg, tmp, plan, spec):
    rng = np.random.default_rng(14)
    raw = pd.concat([_raw(rng, 2000, 0), _raw(rng, 700, 2000), _raw(rng, 300, 2700)],
                    ignore_index=True)
    split = {"train": list(range(2000)), "valid": list(range(2000, 2700)),
             "test": list(range(2700, 3000))}
    root, run = tmp / "bundle", tmp / "run"
    (root / "pkg" / "features").mkdir(parents=True)
    (root / "pkg" / "serving").mkdir(parents=True)
    (root / "pkg" / "features" / "preprocess.py").write_text(PREPROCESS, encoding="utf-8")
    run.mkdir()
    valid = raw.iloc[split["valid"]]
    # the stored valid logits are what the stand-in Predictor answers (x0): the smoke's valid
    # window is then drawn like the output reference
    pd.DataFrame({"row_id": split["valid"], "score": valid["x0"].to_numpy(),
                  "y": valid["y"].to_numpy()}).to_parquet(run / "predictions.valid.parquet")
    (tmp / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    (root / "MANIFEST.json").write_text(json.dumps(manifest_of("binary")), encoding="utf-8")
    ports = {"Monitor": {"file": "serving/monitor.py", "module": "serving", "name": "DriftWatch",
                         "responsibility": "watch what the analysis found fragile"},
             "FeatureTransformer": {"file": "features/preprocess.py"}}

    def part_rows(part):
        # the test part is never asked for: the pack's references read train and valid only
        assert part in ("train", "valid"), part
        return raw.iloc[split[part]]

    campaign = SimpleNamespace(part_rows=part_rows, run_dir=run, label="y",
                               backend_name="pytorch")
    return SimpleNamespace(
        port=lambda role: dict(ports.get(role, {})), options=SimpleNamespace(
            monitor_plan=str(tmp / "plan.json")),
        paths=SimpleNamespace(manifest=root / "MANIFEST.json", root=root, pkg=root / "pkg"),
        schema=dict(SCHEMA, label={"name": "y", "decode": {}}), campaign=campaign,
        pipeline=lambda: {"monitoring": [{"name": "x0 drift", "source": "[analysis:F3]"}]},
        markers=[], templates=TEMPLATES / "serving", package=PKG, spec=spec), raw


def _not_json(token):
    raise AssertionError(f"monitor-reference.json holds {token}: not strict JSON")


def test_render_writes_the_reference_the_adapter_and_runs_the_smoke(pkg, tmp_path, monkeypatch):
    spec = CASES["binary"][0]
    plan = {"checks": [{"id": "x0", "kind": "psi", "column": "x0", "concern": "[analysis:F3]",
                        "title": "x0 distribution against train"},
                       {"id": "z", "kind": "unseen_rate", "column": "zone", "direction": "up",
                        "concern": "[analysis:F3]", "title": "unseen zones"}],
            "note": "one concern, two checks"}
    ctx, raw = _ctx(pkg, tmp_path, plan, spec)
    predictor = Answering(lambda rows: reader_of(pkg, "binary").frame(
        pkg.samples.HeadOutput(rows[["x0"]].to_numpy(), None)).assign(warnings=""),
        reader_of(pkg, "binary"))
    loaded = []

    def load_monitor(root):
        doc = json.loads((Path(root) / "monitor-reference.json").read_text(encoding="utf-8"))
        loaded.append(doc)
        return pkg.monitor.Monitor(doc, recipe=_unseen_recipe,
                                   labels=pkg.domain.LabelSet.from_spec(spec))

    monkeypatch.setitem(sys.modules, "predict", types.SimpleNamespace(load_monitor=load_monitor))
    record = monitor_pack.render(ctx, predictor)["monitor"]
    text = (ctx.paths.root / "monitor-reference.json").read_text(encoding="utf-8")
    doc = json.loads(text, parse_constant=_not_json)
    assert doc["schema"] == "vega/monitor-reference@2" and doc["concerns"] == ["[analysis:F3]"]
    assert [c["id"] for c in doc["checks"]] == ["x0", "z", "auto-output"]
    auto = doc["checks"][-1]
    assert (auto["kind"], auto["field"], auto["reference_part"]) == ("output_psi", "p_1", "valid")
    assert doc["checks"][1]["reference"] == pytest.approx(
        (raw.iloc[2000:2700]["zone"] == "r").mean())
    assert record["schema"] == "vega/monitor-reference@2" and record["checks"] == 3
    assert record["kinds"] == ["output_psi", "psi", "unseen_rate"]
    assert record["smoke"]["run"] and record["smoke"]["moved"] == {"train": [], "valid": []}
    assert any(m.startswith("OK: monitor smoke") for m in ctx.markers), ctx.markers
    assert loaded == [doc], "the smoke loads the reference pack wrote, through load_monitor"
    # the references read the store; the only rows predicted are the smoke's valid window
    assert len(predictor.seen) == 1 and predictor.seen[0]["rid"].between(2000, 2699).all()
    assert "y" not in predictor.seen[0].columns, "the schema's ignored columns stay out"
    adapter = (ctx.paths.pkg / "serving" / "monitor.py").read_text(encoding="utf-8")
    assert "{{" not in adapter and "the DriftWatch adapter (role: Monitor)" in adapter
    assert f"{PKG}.serving.monitor" in adapter and "_PENDING = (" not in adapter
    assert any("auto" not in m and "rendered" in m for m in ctx.markers), ctx.markers


def test_without_a_plan_or_a_port_the_monitor_stays_the_stub(pkg, tmp_path):
    ctx, _ = _ctx(pkg, tmp_path, {"checks": []}, CASES["binary"][0])
    ctx.options.monitor_plan = None
    assert monitor_pack.render(ctx, Answering(_refusing)) == {}
    assert ctx.markers == ["WARN: no --monitor-plan: the Monitor stays the forge's pending stub "
                           "and the map says so"]
    ctx, _ = _ctx(pkg, tmp_path / "b", {"checks": []}, CASES["binary"][0])
    ctx.port = lambda role: {}
    assert monitor_pack.render(ctx, Answering(_refusing)) == {}
    assert ctx.markers == ["WARN: the map names no Monitor adapter; nothing to render"]
    assert not (ctx.paths.root / "monitor-reference.json").exists()


def test_render_refuses_a_plan_before_measuring(pkg, tmp_path):
    plan = {"checks": [{"id": "r", "kind": "residual", "field": "value",
                        "concern": "[analysis:F3]"}]}
    ctx, _ = _ctx(pkg, tmp_path, plan, CASES["binary"][0])
    with pytest.raises(PackExit) as refusal:
        monitor_pack.render(ctx, Answering(_refusing))
    assert refusal.value.code == 2
    assert refusal.value.lines == ("monitor plan: check r: kind residual is not admitted for "
                                   "the binary reading (admitted: psi, rate, unseen_rate, "
                                   "class_recall, output_psi, prediction_mix)",)
    assert not (ctx.paths.root / "monitor-reference.json").exists()
    unanswered = {"checks": [{"id": "p", "kind": "psi", "column": "x0", "concern": "other"}]}
    ctx, _ = _ctx(pkg, tmp_path / "b", unanswered, CASES["binary"][0])
    with pytest.raises(PackExit, match="answers no check to the blueprint concern"):
        monitor_pack.render(ctx, Answering(_refusing))
    # a parameter the monitor cannot read is refused here, exit 2, not at measurement (exit 3)
    malformed = {"checks": [{"id": "o", "kind": "output_psi", "bins_n": "ten",
                             "concern": "[analysis:F3]"}]}
    ctx, _ = _ctx(pkg, tmp_path / "c", malformed, CASES["binary"][0])
    with pytest.raises(PackExit) as refusal:
        monitor_pack.render(ctx, Answering(_refusing, reader_of(pkg, "binary")))
    assert (refusal.value.code, refusal.value.lines) == (
        2, ("monitor plan: check o: bins_n 'ten' is not a whole number of at least 1",))
    assert not (ctx.paths.root / "monitor-reference.json").exists()


# ================================================================ the v2.2 bridge, closed (S07)

def test_the_v22_bridge_is_closed():
    """S07 deleted v2.2's `templates/serving/monitor.py`, and the bridge of build plan §2.12(3)
    went with it: every bundle is watched by the serving package's Monitor, over a
    `vega/monitor-reference@2` with pack's output check."""
    assert not (TEMPLATES / "serving" / "monitor.py").exists()
    assert not [name for name in vars(monitor_pack) if name.lower().startswith(("_v22", "v22"))]


# ================================================================ 6. the template's discipline

def _rendered(pkg):
    return (pkg.root / "serving" / "monitor.py").read_text(encoding="utf-8")


def test_the_template_meets_c11_to_c13(pkg):
    text = _rendered(pkg)
    assert "{{" not in text
    findings = conformance.size_findings(ast.parse(text), "serving/monitor.py")
    assert not findings, "\n".join(f.message() for f in findings)
    proc = subprocess.run([sys.executable, str(SCRIPTS / "conformance.py"), "--size-severity",
                           "error", "--size-only", str(pkg.root / "serving" / "monitor.py"),
                           str(PACK_SCRIPTS / "monitor_pack.py"),
                           str(PACK_SCRIPTS / "monitor_plan.py"),
                           str(PACK_SCRIPTS / "monitor_reference.py")],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_c14_only_the_monitor_class_is_public(pkg, tmp_path):
    """The gate's C14 on the rendered adapter, marked implemented as pack's map marks it; and
    the same gate finds a public function slipped into a copy (the check reads the file)."""
    def c14(root):
        bmap = json.loads((root / "blueprint-map.json").read_text(encoding="utf-8"))
        for entry in bmap["entries"]:
            if entry.get("role") == "Monitor":
                entry["status"] = "implemented"
        _markers, findings = conformance.check_report(root, bmap, pkg.pipeline)
        return [f.message() for f in findings if "monitor" in f.message()]

    assert c14(pkg.root) == []
    tree = ast.parse(_rendered(pkg))
    public = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
              and not n.name.startswith("_")]
    assert public == ["Monitor"]
    exported = next(n for n in tree.body if isinstance(n, ast.Assign)
                    and getattr(n.targets[0], "id", "") == "__all__")
    assert ast.literal_eval(exported.value) == ["Monitor"]
    import shutil
    copy_root = tmp_path / PKG
    shutil.copytree(pkg.root, copy_root, ignore=shutil.ignore_patterns("__pycache__"))
    target = copy_root / "serving" / "monitor.py"
    target.write_text(target.read_text(encoding="utf-8") + "\n\ndef build(reference):\n"
                      "    return Monitor(reference, recipe=None, labels=None)\n",
                      encoding="utf-8")
    assert any("C14" in m and "build" in m for m in c14(copy_root))


ALLOWED_ROOTS = {"__future__", "dataclasses", "math", "typing", "numpy", "pandas"}


def test_state_and_imports(pkg):
    for node in ast.parse(_rendered(pkg)).body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = [t.id for t in targets if isinstance(t, ast.Name)]
            if isinstance(node.value, conformance.MUTABLE_NODES) and names != ["__all__"]:
                assert all(n.lstrip("_").isupper() for n in names), f"C7: {names}"
        if isinstance(node, ast.Import):
            assert {a.name.split(".")[0] for a in node.names} <= ALLOWED_ROOTS
        if isinstance(node, ast.ImportFrom):
            if node.level:
                assert (node.module, node.level) == ("domain", 2), node.module
            else:
                assert node.module.split(".")[0] in ALLOWED_ROOTS, node.module


def test_the_kinds_are_the_catalogs(pkg):
    """vega-pack's catalog `monitor_kinds` is the list of record: the Monitor measures exactly
    those kinds, and needs a collaborator exactly where the catalog says so."""
    catalog = CATALOG["monitor_kinds"]
    assert set(pkg.monitor.KINDS) == set(catalog)
    needs = {k: row["needs"] for k, row in catalog.items() if row["needs"] in ("recipe", "labels")}
    assert pkg.monitor.NEEDS == needs
    assert set(monitor_plan.REQUIRED) <= set(catalog) and set(monitor_plan.FIELD_READS) <= \
        set(catalog)
    kinds = CATALOG["field_kinds"]
    for reads in monitor_plan.FIELD_READS.values():
        assert set(reads) <= set(kinds), reads
    assert monitor_plan.SCORE_READERS == tuple(
        k for k, row in catalog.items() if row.get("reads") == "score_field")
    for kind in catalog:
        assert hasattr(mref.MonitorRender, "_" + kind), f"no reference measure for {kind}"


def test_the_literal_gate_finds_nothing():
    """R-F10: `lint_templates.py` (T04) is the one deny-list gate."""
    assert LINT.is_file(), f"{LINT} (T04) is on this tree"
    proc = subprocess.run([sys.executable, str(LINT), str(PACKAGE_TEMPLATES / "monitor.py"),
                           str(PACK_SCRIPTS / "monitor_pack.py"),
                           str(PACK_SCRIPTS / "monitor_plan.py"),
                           str(PACK_SCRIPTS / "monitor_reference.py")],
                          capture_output=True, text=True, cwd=str(LINT.parent))
    assert proc.returncode == 0, proc.stdout + proc.stderr


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
