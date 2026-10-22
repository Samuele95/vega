"""
tests/test_api_answers.py — what the service answers, read through the bundle's output record.

    python3 -m pytest tests/test_api_answers.py

Every answer is the Predictor's frame encoded by field kind (MANIFEST.output, vega/output@2;
bundle-contract §7.3), so these tests read the reading, its fields and their kinds from the
record (tests/_oracle.py) and hold the answers to it, for whatever reading this bundle has: the
stored examples are answered again — every compared field within the round-trip tolerance,
every decided field exactly; an answer's keys are the record's (the gated ones — logits, a
reconstruction, an embedding — only on `include`), one entry per field of each kind; the
probabilities sum to one only for a reading whose classes exclude each other; the decision
follows the threshold /v1/schema publishes, on its scale; the CSV answer's columns are the
record's fields and its values the JSON answer's; the truth is held to its type (a class, a
number, label names) on every way in; the monitor reads it; the presets expect the reading's
kind of answer; and the domain's `Prediction` values carry the reading's own decision. An empty
numeric input is never a confident answer: the recipe fills it, or the row goes unanswered naming
the input (a row whose context holds the hole keeps its answer, says so, and carries no decision).
"""
import io
import math

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
from _oracle import (EXPECTED_KINDS, GATED, KEYS, MAPPED, ORACLE, ROOT, TAIL,  # noqa: E402
                     applies, client, numbers_agree, plain_numeric, same_value)  # noqa: F401

MANIFEST = ORACLE.manifest
TOL = ORACLE.tolerance
INCLUDE = ORACLE.includes()
HAS_MONITOR = bool(MANIFEST.get("monitor"))
DEMO = (MANIFEST.get("examples") or {}).get("demo") or {}
DECIDING = ("binary", "multiclass", "multilabel", "ordinal", "anomaly-reconstruction",
            "anomaly-svdd")
NUMERIC_KINDS = ("probability", "raw", "value", "quantile", "horizon", "score", "row_score",
                 "cumulative", "recon", "embedding")
WARNED_EDGES = ("edge_range", "edge_unseen")        # the edge presets the guard must flag
# the kinds whose CSV cell is a 0|1 value, and the spellings the frame writes one in: an integer
# column (int64) or a boolean one
ZERO_ONE_KINDS = ("alarm", "decision", "flag")
ZERO_ONE = {"0": 0, "1": 1, "False": 0, "True": 1}


def _include(values) -> list:
    return [("include", v) for v in values]


def _answers(client, body, include=()) -> list:
    res = client.post("/v1/predict", params=_include(include), json=body)
    assert res.status_code == 200, res.text
    return res.json()["predictions"]


def _examples(client) -> dict:
    return client.get("/v1/examples").json()


def _table(name="rows.csv"):
    text = {c["name"]: "string" for c in ORACLE.inputs if c.get("dtype") == "string"}
    return pd.read_csv(ROOT / "examples" / name, dtype=text)


def _upload(client, table, include=(), headers=None, **flags):
    params = _include(include) + [(k, str(v).lower()) for k, v in flags.items()]
    return client.post("/v1/predict/csv", params=params, headers=headers or {}, files={
        "file": ("t.csv", table.to_csv(index=False).encode("utf-8"), "text/csv")})


def _structure(prediction, include) -> list:
    """What is wrong with the shape of one answered prediction: each mapped kind holds one entry
    per field `of`, a forecast one per horizon, an embedding one per coordinate, the flagged
    names are labels, every number is finite and a per-row score is not negative."""
    problems = []
    for kind in dict.fromkeys(ORACLE.kind_of(f) for f in ORACLE.shown(include)):
        key, ofs = KEYS[kind], [ORACLE.kinds[f].get("of") for f in ORACLE.fields_of(kind)]
        held = prediction.get(key)
        if kind in MAPPED and set(held) != {str(o) for o in ofs}:
            problems.append(f"{key} holds {sorted(held)}, the record {sorted(map(str, ofs))}")
        if kind == "horizon" and set(held) != {f"h{str(o).partition('/')[0]}" for o in ofs}:
            problems.append(f"{key} holds {sorted(held)}")
        if kind == "embedding" and len(held) != len(ofs):
            problems.append(f"{key} holds {len(held)} coordinates, the record {len(ofs)}")
        if kind == "flag" and not set(held) <= set(ORACLE.classes):
            problems.append(f"{key} names {held}, not labels of the record")
    for field in ORACLE.shown(include):
        value = ORACLE.value(prediction, field)
        if ORACLE.kind_of(field) in NUMERIC_KINDS and not (
                isinstance(value, (int, float)) and math.isfinite(value)):
            problems.append(f"{field} is {value!r}, not a finite number")
        elif ORACLE.kind_of(field) == "row_score" and value < 0:
            problems.append(f"{field} is {value}, a negative per-row score")
    return problems


EMPTY = plain_numeric(ORACLE)


def _one(**changes) -> dict:
    body = ORACLE.one_request()
    body["rows"] = [{**r, **(changes if i == 0 else {})} for i, r in enumerate(body["rows"])]
    return body


@applies(EMPTY is not None)
def test_an_empty_numeric_input_is_never_a_confident_answer(client):
    for empty in (None, "NaN"):
        res = client.post("/v1/predict", json=_one(**{EMPTY: empty}))
        if res.status_code == 422:              # refused by name (a coordinate a graph needs)
            assert EMPTY in [e.get("column") for e in res.json().get("errors") or ()], res.text
            continue
        assert res.status_code == 200, res.text
        a = res.json()["predictions"][0]
        if a["answered"]:
            assert not _structure(a, ()), (empty, _structure(a, ()), a)
            continue
        assert any(EMPTY in w or "context" in w for w in a["warnings"]), (empty, a)
        assert all(a[k] is None for k in ORACLE.keys() if k not in (*TAIL, "graph_id")), (empty, a)
    table = _table()
    table.loc[table.index[0], EMPTY] = None
    res = _upload(client, table)
    if res.status_code == 422:
        assert EMPTY in [e.get("column") for e in res.json().get("errors") or ()], res.text
        return
    assert res.status_code == 200, res.text
    for a in res.json()["predictions"]:
        if not a["answered"]:
            assert a["warnings"], a
        elif _structure(a, ()):                  # numbers lost to a hole in the row's context:
            assert any("no decision is read" in w for w in a["warnings"]), a   # said, and
            assert all(a.get(k) in (None, [], {}) for k in ("label", "labels", "anomalous")), a


def test_answers_reproduce_the_stored_examples(client):
    examples = _examples(client)
    body = ORACLE.request(examples)
    predictions = _answers(client, body, INCLUDE)
    assert len(predictions) == len(body["rows"]) == MANIFEST["examples"]["rows"]
    expected, problems = ORACLE.expected(), []
    for i, at in enumerate(ORACLE.compared_positions()):
        problems += [f"example row {at}: {p}"
                     for p in ORACLE.check(predictions[at], expected.iloc[i].to_dict(), TOL)]
    assert not problems, f"{len(problems)} disagreement(s): " + "; ".join(problems[:5])


def test_answer_keys_follow_the_output_record(client):
    contract = client.app.state.contract
    body = ORACLE.one_request()
    for include in ((), INCLUDE):
        assert tuple(contract.answers.keys(include)) == ORACLE.keys(include), include
        for prediction in _answers(client, body, include):
            assert set(prediction) == set(ORACLE.keys(include)), (include, sorted(prediction))
            assert prediction["answered"] is True, prediction
            assert not _structure(prediction, include), _structure(prediction, include)


def test_probabilities_sum_to_one_only_where_they_should(client):
    predictions = [p for p in _answers(client, ORACLE.request(_examples(client))) if
                   p["answered"]]
    assert predictions, "no shipped example row was answered"
    for prediction in predictions:
        if not ORACLE.fields_of("probability"):
            assert "probabilities" not in prediction, prediction
            continue
        values = list(prediction["probabilities"].values())
        assert all(0.0 <= v <= 1.0 for v in values), prediction
        if ORACLE.sums_to_one():
            assert abs(sum(values) - 1.0) < 1e-5, prediction


def _decision_problem(p, output) -> str | None:
    """Why one answered prediction's decision does not follow the published threshold (None
    when it does), per reading: a class by threshold, argmax or cumulative count; label names
    by their own thresholds; the anomaly flag by the score's threshold."""
    threshold, reading = output.get("threshold"), ORACLE.reading
    if reading == "binary":
        above = p["probabilities"][ORACLE.positive] >= (0.5 if threshold is None else threshold)
        return None if (p["label"] == ORACLE.positive) == above else "binary threshold"
    if reading == "multiclass":
        best = max(p["probabilities"], key=p["probabilities"].get)
        return None if p["label"] == best else f"argmax is {best}"
    if reading == "multilabel":
        cuts = output.get("threshold_per_label") or {}
        want = {name for name, v in p["probabilities"].items()
                if v >= cuts.get(name, 0.5 if threshold is None else threshold)}
        return None if set(p["labels"]) == want else f"labels {sorted(want)}"
    if reading == "ordinal":
        count = sum(1 for z in p["logits"].values() if z >= 0)
        return None if p["label"] == ORACLE.classes[count] else "cumulative count"
    if threshold is None:           # no threshold: the score is answered undecided, never cut
        undecided = p["anomalous"] is None and p.get("label") is None
        return None if undecided else f"decided {p['anomalous']!r} without a threshold"
    flagged = p["score"] >= threshold
    if p["anomalous"] != flagged:
        return "score threshold"
    named = ORACLE.fields_of("class") and p["label"] != ORACLE.classes[int(flagged)]
    return f"label {p['label']!r} is not the class the flag names" if named else None


@applies(ORACLE.reading in DECIDING)
def test_the_decision_follows_the_published_threshold(client):
    output = client.get("/v1/schema").json()["output"]
    assert output["threshold_scale"] == MANIFEST["output"].get("threshold_scale")
    predictions = [p for p in _answers(client, ORACLE.request(_examples(client)), INCLUDE)
                   if p["answered"]]
    problems = [(i, _decision_problem(p, output)) for i, p in enumerate(predictions)]
    problems = [(i, why) for i, why in problems if why]
    assert not problems, problems[:5]


def test_logits_only_on_request(client):
    # and every other gated key (a reconstruction, an embedding) only on its own `include`; an
    # `include` the record has no field for adds nothing
    body = ORACLE.one_request()
    plain = _answers(client, body)[0]
    every = _answers(client, body, tuple(GATED.values()))[0]
    for kind, include in GATED.items():
        present = bool(ORACLE.fields_of(kind))
        assert KEYS[kind] not in plain, (kind, plain)
        assert (KEYS[kind] in every) is present, (kind, every)
        alone = _answers(client, body, (include,))[0]
        assert (KEYS[kind] in alone) is present and set(alone) - set(plain) <= {KEYS[kind]}
    assert set(every) - set(plain) == {KEYS[k] for k in GATED if ORACLE.fields_of(k)}


def _csv_cell(field, text):
    """One CSV answer cell read by its field's declared kind (MANIFEST.output.field_kinds;
    bundle-contract §7.3), as `ORACLE.value` reads the JSON answer: a number kind as a number,
    a 0|1 kind (the anomaly flag, a binary decision, a label's flag) as 0 or 1 — the frame
    writes it as an integer, or as a boolean — anything else (a class, a graph id) as its text;
    an empty cell is null. A cell its kind cannot hold is a failure, never a guess."""
    if text == "":
        return None
    kind = ORACLE.kind_of(field)
    if kind in NUMERIC_KINDS:
        return float(text)
    if kind in ZERO_ONE_KINDS:
        assert text in ZERO_ONE, f"{field} ({kind}) holds {text!r}, not a 0|1 cell"
        return ZERO_ONE[text]
    return text


def test_csv_answer_columns_are_the_output_fields(client):
    contract, table = client.app.state.contract, _table()
    truth = ORACLE.truth_column in table.columns and not ORACLE.truth_is_input()
    for include in ((), INCLUDE):
        res = _upload(client, table, include, {"accept": "text/csv"})
        assert res.status_code == 200, res.text
        answer = pd.read_csv(io.StringIO(res.text), dtype=object, keep_default_na=False)
        assert list(answer.columns) == ORACLE.csv_columns(include, truth)
        assert contract.answers.columns(include, truth) == ORACLE.csv_columns(include, truth)
    predictions = _upload(client, table, INCLUDE).json()["predictions"]
    for field in (f for f in ORACLE.shown(INCLUDE) if ORACLE.kind_of(f) not in TAIL):
        kind = ORACLE.kind_of(field)
        for i, p in enumerate(predictions):
            cell = _csv_cell(field, answer[field][i])
            agree = numbers_agree(cell, ORACLE.value(p, field), TOL) \
                if kind in NUMERIC_KINDS else same_value(cell, ORACLE.value(p, field))
            assert agree, (field, i, answer[field][i], ORACLE.value(p, field))


def test_truth_is_held_to_its_type(client):
    body, column = ORACLE.one_request(), ORACLE.truth_column
    n = len(body["rows"])
    if ORACLE.truth_type is None:                # a reading without truth takes none
        res = client.post("/v1/monitor", json=dict(body, truth=[1.0] * n))
        assert res.status_code == 422, res.text
        return
    good = client.post("/v1/monitor", json=dict(body, truth=[ORACLE.good_truth()] * n))
    assert good.status_code in (200, 404), good.text      # 404: no monitor reference
    for bad in (ORACLE.wrong_truth(), True):
        res = client.post("/v1/monitor", json=dict(body, truth=[bad] * n))
        assert res.status_code == 422, (bad, res.text)
        assert res.json()["errors"][0]["column"] == column, (bad, res.json())
    res = client.post("/v1/monitor", json=dict(body, truth=[ORACLE.good_truth()] * (n + 1)))
    assert res.status_code == 422 and res.json()["errors"][0]["column"] == column
    if not ORACLE.truth_is_input():               # the truth sent inside a row, same rule
        rows = [dict(r) for r in body["rows"]]
        rows[0][column] = ORACLE.good_truth()
        assert client.post("/v1/predict", json=dict(body, rows=rows)).status_code == 200
        for bad in (ORACLE.wrong_truth(), {"nested": [1, 2]}, True):
            rows[0][column] = bad
            res = client.post("/v1/predict", json=dict(body, rows=rows))
            assert res.status_code == 422, (bad, res.text)
            assert res.json()["errors"][0]["column"] == column, (bad, res.json())


def _truth_cell(value):
    """A truth as a CSV cell: label names joined on the record's separator."""
    return ORACLE.truth["sep"].join(value) if isinstance(value, list) else value


@applies(not ORACLE.window and ORACLE.truth_type is not None and not ORACLE.truth_is_input())
def test_a_tables_truth_is_decoded_or_refused(client):
    # a row or graph bundle carries its labelled sample (a window's items are series, no table);
    # its absence is a skip the skip policy refuses, never a silent pass
    if not (ROOT / "examples" / "demo.csv").is_file():
        pytest.skip("no labelled demonstration sample in this bundle")
    table, column = _table("demo.csv"), ORACLE.truth_column
    res = _upload(client, table, ("rows",), monitor=True)
    assert res.status_code == 200, res.text
    body = res.json()
    assert len(body["truth"]) == len(body["rows"]) == len(table)
    known = set(ORACLE.classes)
    for value in body["truth"]:                  # decoded to the truth's own type
        if ORACLE.truth_type == "labels":
            assert set(value) <= known, value
        elif ORACLE.truth_type == "class":
            assert value in known, value
        else:
            assert value is None or math.isfinite(value), value
    if HAS_MONITOR:
        assert body["monitor"]["readings"] is not None
    bad = table.copy()
    bad[column] = bad[column].astype(object)
    bad.loc[0, column] = _truth_cell(ORACLE.wrong_truth())
    res = _upload(client, bad)
    assert res.status_code == 422 and res.json()["errors"][0]["column"] == column, res.text


def test_monitor_accepts_the_readings_truth(client):
    demo = _examples(client).get("demo") or {}
    item = (demo.get("items") or [None])[0]
    body = ORACLE.truth_payload(item) if item else ORACLE.one_request()
    if not HAS_MONITOR:
        res = client.post("/v1/monitor", json=body)
        assert res.status_code == 404 and res.json()["title"], res.text  # a permanent state
        pytest.skip("no monitor reference in this bundle (pack ran without --monitor-plan)")
    res = client.post("/v1/monitor", params={"all": "true"}, json=body)
    assert res.status_code == 200, res.text
    readings = res.json()["readings"]
    assert readings and all(r["title"] and r["verdict"] for r in readings), readings
    assert all(r["value"] is None or math.isfinite(r["value"]) for r in readings)
    moved = client.post("/v1/monitor", json=body).json()
    assert moved["moved"] == len(moved["readings"])
    assert all(r["verdict"] in ("watch", "alert") for r in moved["readings"])


def _preset_body(preset) -> dict:
    body = {"rows": [preset["row"]] if preset.get("row") is not None else preset["rows"]}
    if preset.get("history"):
        body["history"] = preset["history"]
    return body


def _decided(answer, expected):
    """The answer's own decision, in the form a preset's `expected` has."""
    kind = expected["kind"]
    if kind == "labels":
        return sorted(answer["labels"]) == sorted(expected["value"])
    held = answer["anomalous"] if kind == "alarm" else answer["label"]
    return same_value(held, expected["value"])


def test_presets_expect_the_readings_kind(client):
    # collected on every served bundle: a bundle without presets reports it as a skip, which the
    # verification program's skip policy refuses (a served bundle ships its presets)
    presets = _examples(client).get("presets")
    if not presets:
        pytest.skip("no presets in this bundle")
    counted = (DEMO.get("presets") or {}).get("count")
    assert len(presets) == counted, f"{len(presets)} presets served, the manifest counts {counted}"
    for preset in presets:
        assert preset["group"] in ("typical", "edge"), preset["id"]
        answer = _answers(client, _preset_body(preset))[0]
        expected = preset["expected"]
        if expected is not None:
            assert expected["kind"] in EXPECTED_KINDS[ORACLE.reading], preset["id"]
        if preset["kind"] == "typical_correct":
            assert _decided(answer, expected), (preset["id"], answer)
        if preset["kind"] == "typical_none_correct":
            assert not _decided(answer, expected), (preset["id"], answer)
        if preset["kind"] in WARNED_EDGES:
            assert answer["warnings"], (preset["id"], answer)


def test_domain_predictions_carry_the_readings_decision():
    from predict import load, read_rows
    predictor = load(ROOT)
    rows = read_rows(ROOT / "examples" / "rows.csv", ROOT)
    frame, domain = predictor.predict(rows), predictor.predictions(rows)
    assert len(domain) == len(frame) == len(rows)
    named, decision = ORACLE.fields_of("class"), ORACLE.fields_of("decision")
    compared = [f for f, _ in ORACLE.compared]
    for prediction, record in zip(domain, frame.to_dict("records")):
        assert prediction.model_version == MANIFEST["model_version"]
        if not record.get("answered", True):
            assert prediction.label is None and not prediction.score and prediction.warnings
            continue
        if named:
            assert str(prediction.label.name) == str(record[named[0]]), (prediction, record)
        else:
            assert prediction.label is None, prediction
        if decision:
            assert prediction.label.index == int(record[decision[0]]), (prediction, record)
        assert prediction.score and all(
            any(np.isclose(s, record[f], rtol=0, atol=TOL) for f in compared)
            for s in prediction.score), (prediction.score, record)
