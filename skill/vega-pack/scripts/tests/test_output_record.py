#!/usr/bin/env python3
"""
test_output_record.py — the reading vocabulary made executable: `MANIFEST.output`
(`vega/output@2`), `MANIFEST.sample` and `schema.json` for every resolve rule and every shape.

    python3 test_output_record.py            # or: python3 -m pytest test_output_record.py

Pure: nothing is packed. For each of R1–R12 a minimal spec gives the reading, its link (the one
link vocabulary, R-C1: the same map vega-forge's `inspection.links` holds), the fields, what the
round trip compares and decides, and every field's kind; `fields ⊇ compared ∪ decided` holds on
every record; window and graph sample records carry R-A5's and R-A3's keys; the schema lists an
autoregressive target and a graph-level group as inputs (R-A7); truth is `{column, type, sep}`
(R-A8); and a kind no rule resolves is refused with exit 2.
"""
from __future__ import annotations

import contextlib
import copy
import io
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

TESTS = Path(__file__).resolve().parent
SCRIPTS = TESTS.parent
SKILLS = SCRIPTS.parent.parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SKILLS / "vega" / "scripts" / "tests"))

import field_patterns  # noqa: E402
import input_schema  # noqa: E402
import output_record as orec  # noqa: E402
from pack_context import PackExit  # noqa: E402
from vega_pack_core import CATALOG  # noqa: E402

FORGE_CATALOG = json.loads((SKILLS / "vega-forge" / "scripts" / "catalog.json")
                           .read_text(encoding="utf-8"))
FEATURES = [{"name": "a", "role": "numeric"}, {"name": "b", "role": "numeric"},
            {"name": "zone", "role": "nominal", "cardinality": 3}]
WINDOW = {"shape": "window", "length": 6, "stride": 2, "series_by": "station",
          "order_by": "ts", "context": "previous-part"}


def make_spec(kind, head=None, loss=None, sample=None, **task):
    spec = {"task": {"kind": kind, **task},
            "data_contract": {"features": FEATURES, "sample": sample or {"shape": "row"}}}
    if head is not None:
        spec["blocks"] = [{"type": "dense", "params": {"units": 8}},
                          {"type": "head", "params": {"out": head}}]
    if loss:
        spec["loss"] = {"name": loss} if isinstance(loss, str) else loss
    return spec


def refusal(fn, *args):
    try:
        fn(*args)
    except PackExit as exc:
        return exc
    return None


# --------------------------------------------------------------------------- R1–R12

QUANTILE = {"name": "quantile", "params": {"quantiles": [0.1, 0.5, 0.9]}}
CASES = [
    # (rule, spec, threshold.json, truth column, what the record must say)
    ("R1", make_spec("binary", 1, "bce_with_logits", classes=[0, 1], positive=1),
     {"value": 0.41, "source": "valid-optimum", "applies": True}, "late",
     {"reading": "binary", "base": None, "link": "sigmoid", "decision": "threshold",
      "classes": ["0", "1"], "positive": "1", "threshold": 0.41,
      "threshold_source": "valid-optimum", "threshold_scale": "probability",
      "fields": ["label", "p_0", "p_1", "score", "prediction", "warnings"],
      "field_kinds": {"label": {"kind": "class", "of": None},
                      "p_0": {"kind": "probability", "of": "0"},
                      "p_1": {"kind": "probability", "of": "1"},
                      "score": {"kind": "raw", "of": "1"},
                      "prediction": {"kind": "decision", "of": None},
                      "warnings": {"kind": "warnings", "of": None}},
      "compared": [["score", "score"]], "decided": [["prediction", "prediction"]],
      "score_field": "p_1", "truth": {"column": "late", "type": "class", "sep": None}}),
    ("R1", make_spec("binary", 1, "focal", classes=["yes", "no"], positive="yes"), None, "y",
     {"classes": ["no", "yes"], "positive": "yes", "threshold": None, "threshold_source": None,
      "fields": ["label", "p_no", "p_yes", "score", "prediction", "warnings"],
      "field_kinds": {"score": {"kind": "raw", "of": "yes"}}, "score_field": "p_yes"}),
    ("R2", make_spec("multiclass", 3, "cross_entropy", classes=["a", "b", "c"]),
     {"value": 0.5, "source": "vega:default", "applies": True}, "kind",
     {"reading": "multiclass", "link": "softmax", "decision": "argmax", "threshold": None,
      "threshold_source": None, "threshold_scale": None, "positive": None,
      "fields": ["label", "p_a", "p_b", "p_c", "logit_a", "logit_b", "logit_c", "warnings"],
      "field_kinds": {"logit_b": {"kind": "raw", "of": "b"},
                      "p_c": {"kind": "probability", "of": "c"}},
      "compared": [["logit_a", "score_0"], ["logit_b", "score_1"], ["logit_c", "score_2"]],
      "decided": [["label", "argmax"]], "score_field": None,
      "truth": {"column": "kind", "type": "class", "sep": None}}),
    ("R3", make_spec("multilabel", 2, "bce_with_logits", classes=["x", "y"],
                     label_format={"kind": "delimited", "sep": ";"}),
     {"value": 0.5, "per_label": {"x": 0.3, "y": 0.6}, "source": "valid-optimum",
      "scale": "probability"}, "tags",
     {"reading": "multilabel", "link": "sigmoid_each", "decision": "per-label-threshold",
      "threshold": 0.5, "threshold_per_label": {"x": 0.3, "y": 0.6},
      "threshold_scale": "probability",
      "fields": ["p_x", "p_y", "logit_x", "logit_y", "pred_x", "pred_y", "warnings"],
      "field_kinds": {"pred_y": {"kind": "flag", "of": "y"},
                      "logit_x": {"kind": "raw", "of": "x"}},
      "compared": [["logit_x", "score_0"], ["logit_y", "score_1"]],
      "decided": [["pred_x", "threshold"], ["pred_y", "threshold"]],
      "truth": {"column": "tags", "type": "labels", "sep": ";"}}),
    ("R4", make_spec("ordinal", 2, "cumulative_link", classes=["low", "mid", "high"]),
     None, "grade",
     {"reading": "ordinal", "link": "cumulative", "decision": "cumulative",
      "fields": ["label", "p_low", "p_mid", "p_high", "cum_low", "cum_mid", "logit_cum_low",
                 "logit_cum_mid", "warnings"],
      "field_kinds": {"cum_mid": {"kind": "cumulative", "of": "mid"},
                      "logit_cum_low": {"kind": "raw", "of": "low"}},
      "compared": [["logit_cum_low", "score_0"], ["logit_cum_mid", "score_1"]],
      "decided": [["label", "cumulative"]]}),
    ("R5", make_spec("regression", 3, QUANTILE), None, "price",
     {"reading": "quantile", "link": "identity", "decision": "none",
      "quantiles": [0.1, 0.5, 0.9],
      "fields": ["value", "q_10", "q_50", "q_90", "warnings"],
      "field_kinds": {"value": {"kind": "value", "of": None},
                      "q_10": {"kind": "quantile", "of": 0.1}},
      "compared": [["q_10", "score_0"], ["q_50", "score_1"], ["q_90", "score_2"]],
      "decided": [], "score_field": "value",
      "truth": {"column": "price", "type": "number", "sep": None}}),
    ("R5", make_spec("regression", 2, {"name": "quantile",
                                       "params": {"quantiles": [0.025, 0.975]}}), None, "p",
     {"reading": "quantile", "fields": ["q_2p5", "q_97p5", "warnings"],
      "compared": [["q_2p5", "score_0"], ["q_97p5", "score_1"]], "score_field": "q_2p5"}),
    ("R6", make_spec("regression", 1, "mse"), None, "price",
     {"reading": "regression", "link": "identity", "fields": ["value", "warnings"],
      "compared": [["value", "score"]], "decided": [], "score_field": "value",
      "quantiles": None, "truth": {"column": "price", "type": "number", "sep": None}}),
    ("R6", make_spec("regression", 1, QUANTILE), None, "price",
     {"reading": "regression", "quantiles": None, "fields": ["value", "warnings"]}),
    ("R7", make_spec("unsupervised", 3, "reconstruction"), None, "ignored",
     {"reading": "reconstruction", "link": "reconstruction", "decision": "none",
      "classes": None, "truth": None,
      "fields": ["reconstruction_error", "recon_a", "recon_b", "recon_zone", "warnings"],
      "field_kinds": {"recon_zone": {"kind": "recon", "of": "zone"},
                      "reconstruction_error": {"kind": "row_score", "of": None}},
      "compared": [["recon_a", "score_0"], ["recon_b", "score_1"], ["recon_zone", "score_2"],
                   ["reconstruction_error", "row_score"]],
      "decided": [], "score_field": "reconstruction_error"}),
    ("R7", make_spec("unsupervised", 5, "vae"), None, None,
     {"reading": "reconstruction",
      "fields": ["reconstruction_error", "out_0", "out_1", "out_2", "out_3", "out_4",
                 "warnings"],
      "field_kinds": {"out_3": {"kind": "recon", "of": 3}},
      "compared": [["out_0", "score_0"], ["out_1", "score_1"], ["out_2", "score_2"],
                   ["out_3", "score_3"], ["out_4", "score_4"],
                   ["reconstruction_error", "row_score"]]}),
    ("R8", make_spec("anomaly", 4, "svdd"),
     {"value": 2.5, "source": "valid-optimum", "applies": True, "scale": "score"}, None,
     {"reading": "anomaly-svdd", "link": "distance", "decision": "threshold",
      "threshold": 2.5, "threshold_scale": "score", "classes": None, "positive": None,
      "truth": None,
      "fields": ["score", "prediction", "emb_0", "emb_1", "emb_2", "emb_3", "warnings"],
      "field_kinds": {"emb_2": {"kind": "embedding", "of": 2},
                      "prediction": {"kind": "alarm", "of": None},
                      "score": {"kind": "score", "of": None}},
      "compared": [["emb_0", "score_0"], ["emb_1", "score_1"], ["emb_2", "score_2"],
                   ["emb_3", "score_3"], ["score", "row_score"]],
      "decided": [["prediction", "prediction"]], "score_field": "score"}),
    ("R8", make_spec("anomaly", 2, "svdd", classes=["normal", "fraud"], positive="fraud"),
     {"value": 1.0, "applies": True}, "is_fraud",
     {"classes": ["normal", "fraud"], "positive": "fraud",
      "fields": ["score", "prediction", "label", "emb_0", "emb_1", "warnings"],
      "truth": {"column": "is_fraud", "type": "class", "sep": None}}),
    ("R9", make_spec("anomaly", 3, "reconstruction"), {"value": 0.2, "applies": True}, None,
     {"reading": "anomaly-reconstruction", "link": "reconstruction", "threshold_scale": "score",
      "fields": ["score", "prediction", "recon_a", "recon_b", "recon_zone", "warnings"],
      "compared": [["recon_a", "score_0"], ["recon_b", "score_1"], ["recon_zone", "score_2"],
                   ["score", "row_score"]],
      "decided": [["prediction", "prediction"]], "truth": None}),
    ("R10", make_spec("sequence-classification", 3, "cross_entropy", WINDOW,
                      base_kind="multiclass", classes=["a", "b", "c"]), None, "event",
     {"reading": "multiclass", "base": "multiclass",
      "fields": ["label", "p_a", "p_b", "p_c", "logit_a", "logit_b", "logit_c", "answered",
                 "warnings"],
      "field_kinds": {"answered": {"kind": "answered", "of": None}},
      "compared": [["logit_a", "score_0"], ["logit_b", "score_1"], ["logit_c", "score_2"]]}),
    ("R10", make_spec("sequence-classification", 1, "bce_with_logits", WINDOW, classes=[0, 1]),
     {"value": 0.7}, "event",
     {"reading": "binary", "base": "binary", "threshold": 0.7,
      "fields": ["label", "p_0", "p_1", "score", "prediction", "answered", "warnings"]}),
    ("R10", make_spec("sequence-classification", 3, "bce_with_logits", WINDOW,
                      classes=["u", "v", "w"]), None, "tags",
     {"reading": "multilabel", "base": "multilabel",
      "truth": {"column": "tags", "type": "labels", "sep": "|"}}),
    ("R11", make_spec("sequence-regression", 2, "cumulative_link", WINDOW,
                      base_kind="ordinal", classes=["lo", "hi", "top"]), None, "level",
     {"reading": "ordinal", "base": "ordinal",
      "compared": [["logit_cum_lo", "score_0"], ["logit_cum_hi", "score_1"]]}),
    ("R11", make_spec("sequence-regression", 3, QUANTILE, WINDOW, base_kind="regression"),
     None, "load",
     {"reading": "quantile", "base": "regression",
      "fields": ["value", "q_10", "q_50", "q_90", "answered", "warnings"]}),
    ("R11", make_spec("sequence-regression", 2, "cumulative_link", WINDOW,
                      classes=["lo", "hi", "top"]), None, "level",
     {"reading": "ordinal", "base": "ordinal"}),
    ("R12", make_spec("forecast", 3, "mse", dict(WINDOW, horizon=3)), None, "load",
     {"reading": "forecast", "base": "regression", "link": "horizon", "decision": "none",
      "horizons": [1, 2, 3], "quantiles": None,
      "fields": ["value_h1", "value_h2", "value_h3", "answered", "warnings"],
      "field_kinds": {"value_h2": {"kind": "horizon", "of": 2}},
      "compared": [["value_h1", "score_0"], ["value_h2", "score_1"], ["value_h3", "score_2"]],
      "score_field": "value_h1", "truth": {"column": "load", "type": "number", "sep": None}}),
    ("R12", make_spec("forecast", 4, {"name": "quantile", "params": {"quantiles": [0.1, 0.9]}},
                      dict(WINDOW, horizon=2)), None, "load",
     {"reading": "forecast", "horizons": [1, 2], "quantiles": [0.1, 0.9],
      "fields": ["value_h1_q10", "value_h1_q90", "value_h2_q10", "value_h2_q90", "answered",
                 "warnings"],
      "field_kinds": {"value_h2_q10": {"kind": "horizon", "of": "2/0.1"}},
      "compared": [["value_h1_q10", "score_0"], ["value_h1_q90", "score_1"],
                   ["value_h2_q10", "score_2"], ["value_h2_q90", "score_3"]],
      "score_field": "value_h1_q10"}),
    ("R12", make_spec("forecast", 1, "mse", dict(WINDOW, horizon=1)), None, "load",
     {"fields": ["value_h1", "answered", "warnings"], "compared": [["value_h1", "score"]]}),
]


def test_every_rule_gives_its_reading_fields_and_pairs():
    for rule, spec, threshold, truth, want in CASES:
        record = orec.output_record(spec, threshold, truth)
        assert orec.resolve_reading(spec) == record["reading"], rule
        for key, value in want.items():
            if key == "field_kinds":
                for field, kind in value.items():
                    assert record["field_kinds"][field] == kind, (rule, field,
                                                                  record["field_kinds"][field])
            else:
                assert record[key] == value, (rule, key, record[key], value)


def test_the_cases_cover_every_rule_and_every_reading():
    rules = {r["rule"] for r in CATALOG["output_readings"]["resolve"]}
    assert {case[0] for case in CASES} == rules
    reached = {orec.resolve_reading(case[1]) for case in CASES}
    assert reached == set(CATALOG["output_readings"]["readings"]), reached


def test_the_link_is_the_forge_link_vocabulary():
    """R-C1: `MANIFEST.output.link` is one of vega-forge's `inspection.links`, and that link
    lists the reading."""
    links = FORGE_CATALOG["inspection"]["links"]
    for rule, spec, threshold, truth, _ in CASES:
        record = orec.output_record(spec, threshold, truth)
        assert record["link"] in links, (rule, record["link"])
        assert record["reading"] in links[record["link"]]["readings"], (rule, record["link"])


def test_every_record_is_closed_and_complete():
    """`fields ⊇ compared ∪ decided`, a kind for every field and only known kinds, the stored
    side a column the evaluation writes, the keys the catalog's record_keys in order, and the
    lane program's reading of the record finds nothing wrong."""
    from lanekit import contracts
    stored = re.compile(r"score(_\d+)?|row_score|prediction|argmax|cumulative|threshold")
    for rule, spec, threshold, truth, _ in CASES:
        record = orec.output_record(spec, threshold, truth)
        fields = record["fields"]
        assert {p[0] for p in record["compared"] + record["decided"]} <= set(fields), rule
        assert list(record["field_kinds"]) == fields, rule
        assert {v["kind"] for v in record["field_kinds"].values()} <= set(CATALOG["field_kinds"])
        assert all(stored.fullmatch(p[1]) for p in record["compared"] + record["decided"]), rule
        assert list(record) == CATALOG["output_readings"]["record_keys"], rule
        assert fields[-1] == "warnings" and len(set(fields)) == len(fields), rule
        assert orec.field_kinds(record) == record["field_kinds"], rule
        assert contracts.output_record_problems({"output": record}) == [], rule
        assert record["score_field"] in fields + [None], rule
        json.dumps(record, allow_nan=False)


def test_a_graph_level_record_names_its_group_first():
    spec = make_spec("multiclass", 2, "cross_entropy",
                     {"shape": "graph", "level": "graph", "graph_id": "mol"}, classes=["p", "q"])
    record = orec.output_record(spec, None, "kind")
    assert record["fields"] == ["graph_id", "label", "p_p", "p_q", "logit_p", "logit_q",
                                "warnings"]
    assert record["field_kinds"]["graph_id"] == {"kind": "group", "of": None}
    node = make_spec("binary", 1, "bce_with_logits", {"shape": "graph", "level": "node"},
                     classes=[0, 1])
    assert "graph_id" not in orec.output_record(node, None, "y")["fields"]


def test_every_conditional_pattern_has_its_gate():
    for reading, row in CATALOG["output_readings"]["readings"].items():
        for pattern in row["conditional"]:
            assert pattern in field_patterns.CONDITIONS, (reading, pattern)
            assert pattern in row["fields"], (reading, pattern)
        assert set(row["describe"]) == set(row["fields"]), reading


# --------------------------------------------------------------------------- refusals

def test_a_kind_no_rule_resolves_is_refused_with_exit_2():
    import pack
    for spec, why in ((make_spec("clustering", 3), "task.kind clustering has no serving reading"),
                      (make_spec("sequence-classification", 1, base_kind="regression"),
                       "task.kind sequence-classification has no serving reading")):
        for fn, args in ((orec.resolve_reading, (spec,)),
                         (orec.output_record, (spec, None, "y"))):
            refused = refusal(fn, *args)
            assert isinstance(refused, PackExit) and refused.code == 2, (fn, refused)
            assert refused.lines[0].startswith(why), refused.lines
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = pack._refused(refusal(orec.resolve_reading, spec), None, [])
        assert code == 2 and out.getvalue().startswith(f"ERROR: {why}"), out.getvalue()


def test_the_threshold_applies_only_where_the_reading_decides_by_one():
    binary = make_spec("binary", 1, "bce_with_logits", classes=[0, 1])
    off = orec.output_record(binary, {"value": 0.4, "applies": False, "source": "x"}, "y")
    assert (off["threshold"], off["threshold_source"], off["threshold_scale"]) == \
        (None, None, "probability")
    wrong = refusal(orec.output_record, binary, {"value": 0.4, "scale": "score"}, "y")
    assert isinstance(wrong, PackExit) and wrong.code == 2 and "scale" in wrong.lines[0]
    ordinal = make_spec("ordinal", 1, "cumulative_link", classes=["a", "b"])
    assert orec.output_record(ordinal, {"value": 0.5, "applies": True}, "y")["threshold"] is None


def test_two_classes_that_read_as_one_field_are_refused():
    spec = make_spec("multiclass", 3, "cross_entropy", classes=[1, "1", 2])
    refused = refusal(orec.output_record, spec, None, "y")
    assert isinstance(refused, PackExit) and refused.code == 2 and "p_1" in refused.lines[0]


def test_a_compared_field_outside_fields_is_refused_with_exit_3():
    row = orec.TABLE["readings"]["regression"]
    saved = copy.deepcopy(row)
    row["compared"] = [["value", "score"], ["residual", "score"]]
    try:
        refused = refusal(orec.output_record, make_spec("regression", 1, "mse"), None, "y")
    finally:
        row.clear()
        row.update(saved)
    assert isinstance(refused, PackExit) and refused.code == 3 and "residual" in refused.lines[0]


# --------------------------------------------------------------------------- MANIFEST.sample

LAG_RECIPE = {"steps": [{"op": "split", "params": {"method": "time"}},
                        {"op": "lag", "params": {"lags": [1, 3]}},
                        {"op": "lag", "params": {"lags": [2]}}]}


def test_lag_depth_is_the_largest_lag():
    assert orec.lag_depth({}) == 0 and orec.lag_depth(None) == 0
    assert orec.lag_depth(LAG_RECIPE) == 3
    assert orec.lag_depth({"steps": [{"op": "lag", "params": {"lags": 4}}]}) == 4
    assert orec.lag_depth({"steps": [{"op": "lag", "params": {"group_by": "s"}}]}) == 1   # shape.py's default


def test_a_forecast_window_record_carries_horizon_autoregressive_and_history():
    spec = make_spec("forecast", 3, "mse", dict(WINDOW, horizon=3))
    sample = orec.sample_record(spec, LAG_RECIPE)
    table = CATALOG["sample_shapes"]["window"]
    assert list(sample) == table["record"]
    assert sample == {"shape": "window", "length": 6, "stride": 2, "horizon": 3,
                      "series_by": "station", "order_by": "ts", "context": "previous-part",
                      "autoregressive": True, "lag_depth": 3, "history_rows": 8,
                      "rule": table["rule"]}
    off = orec.sample_record(make_spec("forecast", 3, "mse",
                                       dict(WINDOW, horizon=3, autoregressive=False)), {})
    assert off["autoregressive"] is False and off["history_rows"] == 5


def test_a_sequence_window_record_has_no_forecast_keys():
    spec = make_spec("sequence-classification", 3, "cross_entropy",
                     {"shape": "window", "length": 4}, base_kind="multiclass")
    sample = orec.sample_record(spec, {})
    assert "horizon" not in sample and "autoregressive" not in sample
    assert (sample["stride"], sample["context"], sample["series_by"]) == (1, "previous-part", None)
    assert sample["history_rows"] == 3 and sample["lag_depth"] == 0
    short = make_spec("sequence-regression", 1, sample={"shape": "window", "length": 1})
    refused = refusal(orec.sample_record, short, {})
    assert isinstance(refused, PackExit) and refused.code == 2 and "length" in refused.lines[0]


def test_a_graph_record_reads_the_reference_counts():
    block = {"method": "knn", "n_nodes": {"train": 10, "valid": 4, "test": 3}, "n_edges": 45,
             "within_part_edges": {"train": 20, "valid": 6, "test": 4},
             "cross_part_edges": {"train-valid": 8, "train-test": 5, "valid-test": 2},
             "directed": False, "self_loops": False}
    node = {"shape": "graph", "level": "node", "graph_id": None, "protocol": "inductive",
            "directed": False, "self_loops": True}
    sample = orec.sample_record(make_spec("binary", 1, sample=node), {}, block)
    table = CATALOG["sample_shapes"]["graph"]
    assert list(sample) == table["record"]
    assert sample["reference"] == {"parts": ["train", "valid"], "nodes": 14, "edges": 34,
                                   "files": {"nodes": "graph/reference-nodes.parquet",
                                             "edges": "graph/reference-edges.parquet",
                                             "meta": "graph/graph.json"}}
    assert (sample["method"], sample["level"], sample["protocol"], sample["self_loops"]) == \
        ("knn", "node", "inductive", True)
    meta = {"method": "key_pair", "n_nodes": 14, "n_edges": 34}
    graph = dict(node, level="graph", graph_id="mol")
    again = orec.sample_record(make_spec("multiclass", 2, sample=graph), {}, meta)
    assert again["reference"]["nodes"] == 14 and again["reference"]["edges"] == 34
    assert (again["graph_id"], again["method"]) == ("mol", "key_pair")
    assert orec.sample_record(make_spec("binary", 1), {}) == {"shape": "row"}


# --------------------------------------------------------------------------- schema.json

def test_the_schema_output_describes_every_field_pattern_once():
    for rule, spec, threshold, truth, _ in CASES:
        record = orec.output_record(spec, threshold, truth)
        sample = orec.sample_record(spec, LAG_RECIPE)
        half = orec.schema_output(record, sample)
        described = [re.compile("".join(".+" if i % 2 else re.escape(p) for i, p in
                                         enumerate(field_patterns.HOLE.split(k))))
                     for k in half["output"]]
        for field in record["fields"]:
            assert sum(bool(r.fullmatch(field)) for r in described) >= 1, (rule, field)
        assert half["truth"] == record["truth"], rule
        assert ("sample" in half) == (sample["shape"] == "window"), rule
        assert "<positive>" not in json.dumps(half) and "<median>" not in json.dumps(half)


def test_the_binary_schema_output_is_the_contract_example():
    record = orec.output_record(CASES[0][1], CASES[0][2], CASES[0][3])
    half = orec.schema_output(record)
    assert list(half["output"]) == ["label", "p_<class>", "score", "prediction", "warnings"]
    assert "logit" in half["output"]["score"] and "p_1 >= threshold" in half["output"]["prediction"]
    assert half["truth"] == {"column": "late", "type": "class", "sep": None}


def _campaign(raw, recipe, spec):
    """A campaign whose fitted file records no split: the schema reads the whole raw file."""
    return SimpleNamespace(raw=raw, recipe=recipe, spec=spec, split={}, threshold=None,
                           shaped_manifest={"label": recipe.get("target"), "steps": []},
                           label=recipe.get("target"), shaped_dir=Path("."))


def test_an_autoregressive_forecast_lists_its_target_as_an_input():
    import pandas as pd
    raw = pd.DataFrame({"ts": range(8), "station": ["s1"] * 4 + ["s2"] * 4,
                        "load": [1.5, 2.0, 2.5, 3.0, 1.0, 1.25, 1.5, 1.75],
                        "temp": [10, 11, 12, 13, 9, 8, 7, 6], "note": ["x"] * 8})
    recipe = dict(LAG_RECIPE, target="load",
                  steps=[{"op": "drop_identity", "columns": ["note"]}] + LAG_RECIPE["steps"])
    spec = make_spec("forecast", 3, "mse", dict(WINDOW, horizon=3, autoregressive=True))
    with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
        schema = orec.write_schema(_campaign(raw, recipe, spec), td, [])
        on_disk = json.loads((Path(td) / "schema.json").read_text(encoding="utf-8"))
    assert on_disk == schema
    names = [c["name"] for c in schema["input"]]
    assert names == ["ts", "station", "temp", "load"], names
    target = schema["input"][-1]
    assert (target["role"], target["required"], target["dtype"]) == ("autoregressive", True,
                                                                     "float")
    assert schema["ignored"] == ["note"]
    assert schema["sample"] == {"history_rows": 8}
    assert schema["truth"] == {"column": "load", "type": "number", "sep": None}
    assert list(schema["output"]) == ["value_h<h>", "answered", "warnings"]
    assert list(schema) == ["input", "output", "truth", "sample", "ignored"]
    sequence = make_spec("sequence-regression", 1, "mse", WINDOW, base_kind="regression")
    with contextlib.redirect_stdout(io.StringIO()):
        plain = input_schema.derive(_campaign(raw, recipe, sequence),
                                    orec.sample_record(sequence, recipe), [])
    assert "load" in plain["ignored"] and "load" not in [c["name"] for c in plain["input"]]


def test_a_graph_level_bundle_takes_its_group_column_as_a_string_input():
    import pandas as pd
    raw = pd.DataFrame({"mol": [1, 1, 2, 2], "x": [0.1, 0.2, 0.3, 0.4], "y": ["a", "b", "a", "b"]})
    recipe = {"target": "y", "steps": []}
    spec = make_spec("multiclass", 2, "cross_entropy",
                     {"shape": "graph", "level": "graph", "graph_id": "mol"}, classes=["a", "b"])
    with contextlib.redirect_stdout(io.StringIO()):
        half = input_schema.derive(_campaign(raw, recipe, spec),
                                   orec.sample_record(spec, recipe), [])
    group = next(c for c in half["input"] if c["name"] == "mol")
    assert (group["dtype"], group["required"]) == ("string", True) and "range_seen" not in group
    assert group["allowed"] == "any"                 # a domain the bundle page can draw
    assert "mol" not in half["ignored"] and half["ignored"] == ["y"]


def test_a_graph_bundle_takes_the_columns_its_edges_read_as_inputs():
    import pandas as pd
    raw = pd.DataFrame({"bus": ["B1", "B2", "B3", "B4"], "feeds": ["B2", "B3", "B4", "B1"],
                        "cap": [1.5, 2.5, 3.5, 4.5], "load": [0.1, 0.2, 0.3, 0.4],
                        "hot": ["no", "yes", "no", "yes"]})
    recipe = {"target": "hot", "steps": [{"op": "drop_identity", "columns": ["bus", "feeds"]}]}
    graph = {"params": {"method": "key_pair", "columns": ["bus", "feeds"], "weight": "cap"},
             "edges_for": {"raw": {"raw:bus": "bus", "raw:feeds": "feeds", "raw:cap": "cap"}}}
    spec = make_spec("binary", 1, "bce_with_logits",
                     {"shape": "graph", "level": "node", "protocol": "inductive"},
                     classes=["no", "yes"])
    campaign = _campaign(raw, recipe, spec)
    campaign.shaped_manifest["graph"] = graph
    with contextlib.redirect_stdout(io.StringIO()):
        half = input_schema.derive(campaign, orec.sample_record(spec, recipe), [])
    by_name = {c["name"]: c for c in half["input"]}
    assert [c["name"] for c in half["input"]] == ["cap", "load", "bus", "feeds"], by_name
    for name in ("bus", "feeds"):          # read by edges_for: required, no train values shown
        assert (by_name[name]["role"], by_name[name]["required"], by_name[name]["dtype"]) == \
            ("graph", True, "string") and by_name[name]["allowed"] == "any", by_name[name]
        assert "distinct_seen" not in by_name[name] and "range_seen" not in by_name[name]
    assert "role" not in by_name["cap"] and by_name["cap"]["range_seen"] == [1.5, 4.5]
    assert half["ignored"] == ["hot"]
    assert input_schema.graph_columns(graph) == ["bus", "feeds", "cap"]
    row = make_spec("binary", 1, "bce_with_logits", None, classes=["no", "yes"])
    with contextlib.redirect_stdout(io.StringIO()):
        plain = input_schema.derive(_campaign(raw, recipe, row), {"shape": "row"}, [])
    assert plain["ignored"] == ["bus", "feeds", "hot"]


# --------------------------------------------------------------------------- the phase

def test_a_missing_raw_label_is_no_spelling_and_the_schema_stays_json():
    """L13 (W7): a multilabel row with no label is missing in the raw file and in the shaped
    part alike; NaN != NaN once made it a 'repaired' spelling, a NaN key schema.json cannot
    hold. Rows without a raw label are rows without truth: they are not paired. A raw spelling
    the recipe blanks is reported and not recorded; a real repair still is."""
    import pandas as pd
    raw = pd.DataFrame({"faults": ["imbalance", None, "bearing|imbalance", None, "be\u00e4ring",
                                   "odd", "bearing"]})
    shaped = pd.DataFrame({"faults": ["imbalance", None, "bearing|imbalance", None, "bearing",
                                      None, "bearing"]})
    markers = []
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "train.parquet"
        shaped.to_parquet(path, index=False)
        with contextlib.redirect_stdout(io.StringIO()):
            decoding = input_schema.label_decoding(raw, path, "faults", markers)
    assert decoding["decode"] == {"be\u00e4ring": "bearing"}, decoding
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in decoding["decode"].items())
    json.dumps(decoding, allow_nan=False)
    assert any(m.startswith("WARN:") and "1 train row(s)" in m for m in markers), markers
    assert any(m.startswith("OK:") and "2 row(s) without a raw label" in m for m in markers)


def test_records_is_the_first_phase_and_calls_no_v22_alias():
    spec = CASES[0][1]
    campaign = SimpleNamespace(threshold=CASES[0][2], label="late", spec=spec, recipe={},
                               shaped_manifest={})
    ctx = SimpleNamespace(spec=spec, campaign=campaign, markers=[])
    with contextlib.redirect_stdout(io.StringIO()):
        got = orec.records(ctx)
    assert got == {"output": orec.output_record(spec, CASES[0][2], "late"),
                   "sample": {"shape": "row"}}
    assert ctx.markers and ctx.markers[0].startswith("OK: output: reading binary")
    call = re.compile(r"\boutput_fields\s*\(|import[^\n]*\boutput_fields\b")
    for script in sorted(SCRIPTS.glob("*.py")):
        if script.name != "vega_pack_core.py":
            assert not call.search(script.read_text(encoding="utf-8")), script.name


def test_the_modules_hold_the_size_discipline():
    files = [str(SCRIPTS / f"{name}.py") for name in ("output_record", "field_patterns",
                                                      "input_schema")]
    gate = SKILLS / "vega-forge" / "scripts" / "conformance.py"
    proc = subprocess.run([sys.executable, str(gate), "--size-only", *files, "--json"],
                          capture_output=True, text=True, timeout=300)
    findings = json.loads(proc.stdout)["size"]["findings"]
    assert not findings, [f["message"] for f in findings]


def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ok    {name}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {name} — {exc}")
    print(f"\n{len(tests) - failed} of {len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
