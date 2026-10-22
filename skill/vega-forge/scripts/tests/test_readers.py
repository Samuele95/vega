#!/usr/bin/env python3
"""
test_readers.py — the serving readers and records (build plan S03; D2 §4.1, §4.2, §4.5; R-B7,
R-C1–R-C4; bundle-contract §2.1, §5).

    python3 -m pytest test_readers.py [-q]          (or: python3 test_readers.py)

The four templates `templates/serving/package/{records, heads, values, guards}.py` are rendered
into the Predictor's module of a package, beside the live `domain.py` and the Model module's
`readout.py` (`render_templates.py`, B04), and imported. `HeadOutput` is `samples.py`'s (D2 §4.3,
S04): with S04's `samples.py` and `graphs.py` on the tree they are rendered too (over the Model
module's `assemble.py`); a tree without them gets a stand-in holding D2's `HeadOutput` only. Every output record is the one vega-pack
writes — `output_record.output_record(spec, threshold.json, truth column)` and
`sample_record(spec, recipe)` (S01) over a small netspec per reading — so the fixtures follow the
producer of `vega/output@2`, not a copy of it.

What it proves:

1. **frame() equals from_stored()** on the same raw outputs for every reading (all ten, plus the
   quantile head without 0.5, the horizon-major forecast quantiles, a reconstruction wider than
   its features, an unlabelled anomaly, a sequence task over windows, a graph-level record):
   the stored side is `vega/predictions-layout@1` (`score` / `score_<k>` float32, `row_score`,
   the evaluation's `prediction`), and the frame's columns are the record's `fields` without the
   shape's own. Every value also equals an oracle written here from the contract's table
   (bundle-contract §2.1) in numpy, not from the readers.
2. **The binary decision is independent of batch composition**: a logit read alone is read as
   in any batch (0.3 is σ(0.3), never the probability 0.3), for every reading row by row. The
   logit is the log-odds of `task.classes[1]` (the class the trainer's `LabelSet` encodes as 1):
   a binary record whose positive is listed first is refused, never answered inverted.
3. **Ordinal probabilities sum to 1 with the clipping warning**: non-monotone cumulative logits
   are clipped at 0 and renormalised, and exactly those rows carry the warning.
4. **The anomaly decision is on the score scale**: `score ≥ threshold` with the threshold the
   evaluation chose, never a squashed score; the svdd score is ‖f(x) − c‖² with the model's
   centre, the reconstruction error numeric MSE plus the nominal cross-entropy (`head()`).
5. **No reader re-implements a link**: no `sigmoid|softmax|expit|exp` in the code of the four
   modules (strings and comments aside), and every reader reaches the spec's `TaskReadout` for
   what it derives (a spy on the readout).
6. The records: a v1 record, an unknown reading, a field without a kind, a pair outside the
   fields, a bad width, a compared column outside the head are refused; sample records per
   shape; `ServingRecords.of` refuses records of another model and is immutable. The readers
   refuse a record the spec's readout does not read the same way (width, link, classes); an
   anomaly record without a threshold answers the score undecided; `from_stored` refuses a
   layout without its raw columns or its per-row score, and takes the evaluation's stored
   decision as it is (so the round trip sees a disagreement).
7. `class_index` / `score_vector` build the domain's `Prediction` with the reading's decision.
8. `SchemaGuard` gives v2.2's warning texts, row by row (and, while the v2.2 template exists,
   exactly what its `_schema_warnings` gave on the same rows).
9. The size discipline (C11–C13 on the rendered files, D2's line budgets), module-level state
   (C7), the imports (the support arrows of D2 §4), one `HeadOutput` (samples.py's, read-only
   heads read without a torch warning and never written) and the one literal gate.

Exit 0 when every test passes.
"""
from __future__ import annotations

import ast
import dataclasses
import importlib
import io
import json
import re
import subprocess
import sys
import tokenize
import warnings
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
SKILLS = SCRIPTS.parent.parent
TEMPLATES = SCRIPTS / "backends" / "pytorch" / "templates"
PACKAGE_TEMPLATES = TEMPLATES / "serving" / "package"
V22_PREDICT = TEMPLATES / "serving" / "predict.py"
LINT = SKILLS / "vega" / "scripts" / "tests" / "lint_templates.py"
FILES = ("records.py", "heads.py", "values.py", "guards.py")
S04_FILES = ("samples.py", "graphs.py")      # D2 §4.3–4.4: HeadOutput's home and what it imports
# D2 §4.1, §4.2, §4.5: the line budget of each support file
BUDGETS = {"records.py": 220, "guards.py": 150, "heads.py": 300, "values.py": 300}
PKG = "s03_readers"
for _p in (str(SCRIPTS), str(HERE), str(SKILLS / "vega-pack" / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import conformance  # noqa: E402
import output_record as orec  # noqa: E402  (vega-pack, S01: the producer of vega/output@2)
import render_templates as rt  # noqa: E402

SHAPE_KINDS = {"group", "answered", "warnings"}


# ================================================================ the rendered package

SAMPLES_STAND_IN = '''"""A stand-in for S04's samples.py on a tree without it: D2 §4.3's HeadOutput."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, eq=False)
class HeadOutput:
    raw: Any
    row_score: Any = None

    def __post_init__(self):
        raw = np.asarray(self.raw, dtype=np.float64)       # D2's sketch: may be a read-only view
        raw = raw.reshape(-1, 1) if raw.ndim == 1 else raw
        if raw.ndim != 2:
            raise ValueError(f"a head output is (n_samples, head_out); got shape {raw.shape}")
        object.__setattr__(self, "raw", raw)
        if self.row_score is not None:
            score = np.asarray(self.row_score, dtype=np.float64).reshape(-1)
            if len(score) != len(raw):
                raise ValueError(f"{len(score)} row score(s) for {len(raw)} sample(s)")
            object.__setattr__(self, "row_score", score)
'''


def _purge(prefix):
    for name in [m for m in sys.modules if m == prefix or m.startswith(prefix + ".")]:
        del sys.modules[name]


def _render_samples(out, pipeline, root):
    """S04's samples.py in the Predictor's module, and its `graphs.py` rendered with the
    emitter's own mapping (the emit table puts any `graphs.py` in the ingestion module; this is
    test_samples' way); on a tree without them, the stand-in."""
    if not all((PACKAGE_TEMPLATES / name).is_file() for name in S04_FILES):
        (root / "serving" / "samples.py").write_text(SAMPLES_STAND_IN, encoding="utf-8")
        return
    rt.render(PACKAGE_TEMPLATES, out, pipeline, "row", only=["samples.py"], into="Predictor")
    emit, spec = rt.load_emit(), rt.reference_spec("row")
    mapping = emit.common_mapping(spec, pipeline, emit.plan(spec, pipeline))
    text = (PACKAGE_TEMPLATES / "graphs.py").read_text(encoding="utf-8")
    (root / "serving" / "graphs.py").write_text(emit.render_text(text, mapping), encoding="utf-8")


@pytest.fixture(scope="module")
def pkg(tmp_path_factory):
    out = tmp_path_factory.mktemp("s03-render")
    pipeline = rt.reference_pipeline()
    pipeline["system"]["package"] = PKG
    rt.render(TEMPLATES, out, pipeline, "row", only=["domain.py", "readout.py", "assemble.py"])
    rec = rt.render(PACKAGE_TEMPLATES, out, pipeline, "row", only=list(FILES), into="Predictor")
    assert {f"serving/{name}" for name in FILES} | {"model/readout.py", "domain.py"} \
        <= set(rec["files"]), rec["files"]
    _render_samples(out, pipeline, Path(rec["root"]))
    sys.path.insert(0, str(out))
    try:
        yield SimpleNamespace(
            root=out / PKG, domain=importlib.import_module(f"{PKG}.domain"),
            readout=importlib.import_module(f"{PKG}.model.readout"),
            samples=importlib.import_module(f"{PKG}.serving.samples"),
            records=importlib.import_module(f"{PKG}.serving.records"),
            heads=importlib.import_module(f"{PKG}.serving.heads"),
            values=importlib.import_module(f"{PKG}.serving.values"),
            guards=importlib.import_module(f"{PKG}.serving.guards"))
    finally:
        sys.path.remove(str(out))
        _purge(PKG)


# ================================================================ specs, records, heads

def _num(name):
    return {"name": name, "dtype": "float32", "role": "numeric"}


def _nominal(name, card):
    return {"name": name, "dtype": "int64", "role": "nominal", "cardinality": card,
            "unseen_category_id": card - 1}


NUMERIC = [_num(f"x{i}") for i in range(4)]
MIXED = NUMERIC + [_nominal("zone", 3)]
WINDOW = {"shape": "window", "length": 6, "stride": 1, "series_by": "series", "order_by": "ts",
          "context": "previous-part"}
FORECAST = {**WINDOW, "horizon": 3, "autoregressive": True}
GRAPH_LEVEL = {"shape": "graph", "edges": "shaped/edges.parquet", "level": "graph",
               "graph_id": "g", "protocol": "inductive", "batching": "graphs"}
PROB = {"applies": True, "value": 0.41, "scale": "probability", "source": "valid-optimum"}
SCORE = {"applies": True, "value": 0.8, "scale": "score", "source": "valid-optimum"}
PER_LABEL = {"applies": True, "value": 0.5, "scale": "probability", "source": "valid-optimum",
             "per_label": {"red": 0.3, "green": 0.6}}


def spec_of(task, head, loss="bce_with_logits", sample=None, features=None):
    """A netspec the readers and vega-pack's producer read (its family is never read here)."""
    features = NUMERIC if features is None else features
    return {"schema": "vega/netspec@1", "seed": 7, "task": task,
            "input": {"manifest_digest": "sha256:" + "ab" * 32},
            "data_contract": {"input_dim": len(features), "features": features,
                              "sample": sample or {"shape": "row"}},
            "family": {"name": "user-defined"},
            "blocks": [{"id": "B1", "type": "dense", "params": {"units": 8}},
                       {"id": "B2", "type": "head", "params": {"out": head}}],
            "loss": loss if isinstance(loss, dict) else {"name": loss, "params": {}},
            "metrics": {"primary": "pr_auc"}, "optimizer": {"name": "adamw"},
            "training": {"max_epochs": 1, "batch_size": 8}}


def _quantile(qs):
    return {"name": "quantile", "params": {"quantiles": qs}}


BINARY = {"kind": "binary", "label": "y", "classes": [0, 1], "positive": 1}
REGRESSION = {"kind": "regression", "label": "y"}
LABELLED_ANOMALY = {"kind": "anomaly", "label": "y", "classes": [0, 1], "positive": 1}
SEQUENCE_BINARY = {"kind": "sequence-classification", "base_kind": "binary", "label": "y",
                   "classes": [0, 1], "positive": 1}
CASES = {
    # name: (reading, spec, threshold.json)
    "binary": ("binary", spec_of(BINARY, 1), PROB),
    "binary-names": ("binary", spec_of({"kind": "binary", "label": "y", "classes": ["no", "yes"],
                                        "positive": "yes"}, 1), PROB),
    "multiclass": ("multiclass", spec_of({"kind": "multiclass", "label": "y",
                                          "classes": ["a", "b", "c"]}, 3, "cross_entropy"), None),
    "multilabel": ("multilabel", spec_of({"kind": "multilabel", "label": "y",
                                          "classes": ["red", "green", "blue"],
                                          "label_format": {"kind": "delimited", "sep": "|"}},
                                         3), PER_LABEL),
    "ordinal": ("ordinal", spec_of({"kind": "ordinal", "label": "y",
                                    "classes": ["low", "mid", "high", "top"]}, 3,
                                   "cumulative_link"), None),
    "regression": ("regression", spec_of(REGRESSION, 1, "mse"), None),
    "quantile": ("quantile", spec_of(REGRESSION, 3, _quantile([0.1, 0.5, 0.9])), None),
    "quantile-no-median": ("quantile", spec_of(REGRESSION, 2, _quantile([0.25, 0.75])), None),
    "forecast": ("forecast", spec_of({"kind": "forecast", "label": "y"}, 3, "mse", FORECAST), None),
    "forecast-quantile": ("forecast", spec_of({"kind": "forecast", "label": "y"}, 6,
                                              _quantile([0.1, 0.9]), FORECAST), None),
    "reconstruction": ("reconstruction", spec_of({"kind": "unsupervised"}, 4, "reconstruction"),
                       None),
    "reconstruction-nominal": ("reconstruction", spec_of({"kind": "unsupervised"}, 7,
                                                         "reconstruction", features=MIXED), None),
    "anomaly-reconstruction": ("anomaly-reconstruction",
                               spec_of(LABELLED_ANOMALY, 4, "reconstruction"), SCORE),
    "anomaly-unlabelled": ("anomaly-reconstruction", spec_of({"kind": "anomaly"}, 4, "vae"), SCORE),
    "anomaly-svdd": ("anomaly-svdd", spec_of({"kind": "anomaly"}, 5, "svdd"),
                     {"applies": True, "value": 1.5, "scale": "score"}),
    "sequence-binary": ("binary", spec_of(SEQUENCE_BINARY, 1, sample=WINDOW), PROB),
    "graph-level": ("multiclass", spec_of({"kind": "multiclass", "label": "y",
                                           "classes": ["p", "q", "r"]}, 3, "cross_entropy",
                                          GRAPH_LEVEL), None),
}


def manifest_of(spec, threshold, recipe=None):
    """What pack writes: MANIFEST.output and MANIFEST.sample, by vega-pack's own producer."""
    truth = (spec.get("task") or {}).get("label")
    return {"output": orec.output_record(spec, threshold, truth),
            "sample": orec.sample_record(spec, recipe or {}, None)}


def records_of(pkg, name, threshold="case", spec=None):
    _, case_spec, case_threshold = CASES[name]
    spec = spec or case_spec
    manifest = manifest_of(case_spec, case_threshold if threshold == "case" else threshold)
    return pkg.records.ServingRecords.of(spec, manifest, {"input": []}, "sha256:model")


def reader_of(pkg, name, threshold="case", spec=None):
    return pkg.values.reader_for(records_of(pkg, name, threshold, spec))


def raw_of(reader, n=24, seed=0):
    """A float32 head (what the network gives and the evaluation stores) and, for the readings
    that have one, a positive per-row score."""
    rng = np.random.default_rng(seed)
    raw = (rng.normal(size=(n, reader.output.head_out)) * 2.0).astype(np.float32)
    row = (rng.random(n) * 2.0).astype(np.float32) if reader.needs_row_score else None
    return raw, row


def stored_of(reader, raw, row):
    """`vega/predictions-layout@1` for these outputs: `score` / `score_<k>`, `row_score`, and
    the evaluation's `prediction` for a threshold reading (its own decision, computed here)."""
    stored = pd.DataFrame({"row_id": np.arange(len(raw)) * 3 + 7})
    if raw.shape[1] == 1:
        stored["score"] = raw[:, 0]
    for k in range(raw.shape[1] if raw.shape[1] > 1 else 0):
        stored[f"score_{k}"] = raw[:, k]
    if row is not None:
        stored["row_score"] = row
    rec = reader.output
    if rec.reading.value == "binary":
        cut = rec.threshold if rec.threshold is not None else 0.5
        stored["prediction"] = (sigmoid(raw[:, 0].astype(np.float64)) >= cut).astype(int)
    if rec.reading.value.startswith("anomaly-") and rec.threshold is not None:
        stored["prediction"] = (row.astype(np.float64) >= rec.threshold).astype(int)
    return stored


def head_of(pkg, raw, row=None):
    return pkg.samples.HeadOutput(raw.astype(np.float64), None if row is None else row)


class _Centred:
    """The part of a model the svdd reading reads: its centre buffer."""

    def __init__(self, centre):
        self._centre = torch.as_tensor(centre, dtype=torch.float32)

    def objective_center(self):
        return self._centre


# ================================================================ the oracle (numpy, from §2.1)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def qtoken(q):
    return format(q * 100, "g").replace(".", "p")


def ordinal_probs(z):
    s = sigmoid(z)
    p = np.concatenate([1.0 - s[:, :1], s[:, :-1] - s[:, 1:], s[:, -1:]], axis=1)
    p = np.clip(p, 0.0, None)
    return p / p.sum(axis=1, keepdims=True)


def expected(rec, raw, row):
    """The answer bundle-contract §2.1 describes, field by field, for a float64 head."""
    reading = rec.reading.value
    names = np.asarray(rec.classes, dtype=object)
    if reading in ("binary", "multiclass", "multilabel", "ordinal"):
        return _expected_classes(rec, raw, names)
    if reading in ("regression", "quantile", "forecast"):
        return _expected_values(rec, raw)
    return _expected_row_scored(rec, raw, row, names)


def _expected_classes(rec, raw, names):
    reading, classes = rec.reading.value, list(rec.classes)
    if reading == "binary":
        p = sigmoid(raw[:, 0])
        d = (p >= (rec.threshold if rec.threshold is not None else 0.5)).astype(np.int64)
        return {"label": names[d], f"p_{classes[0]}": 1.0 - p, f"p_{classes[1]}": p,
                "score": raw[:, 0], "prediction": d}
    if reading == "multilabel":
        p, per = sigmoid(raw), dict(rec.threshold_per_label or {})
        out = {f"pred_{c}": (p[:, k] >= per.get(c, rec.threshold)).astype(np.int64)
               for k, c in enumerate(classes)}
        out.update({f"logit_{c}": raw[:, k] for k, c in enumerate(classes)})
    elif reading == "multiclass":
        p = softmax(raw)
        out = {"label": names[raw.argmax(axis=1)]}
        out.update({f"logit_{c}": raw[:, k] for k, c in enumerate(classes)})
    else:
        p = ordinal_probs(raw)
        out = {"label": names[(sigmoid(raw) >= 0.5).sum(axis=1)]}
        out.update({f"cum_{c}": p[:, k + 1:].sum(axis=1) for k, c in enumerate(classes[:-1])})
        out.update({f"logit_cum_{c}": raw[:, k] for k, c in enumerate(classes[:-1])})
    out.update({f"p_{c}": p[:, k] for k, c in enumerate(classes)})
    return out


def _expected_values(rec, raw):
    reading = rec.reading.value
    if reading == "regression":
        return {"value": raw[:, 0]}
    if reading == "quantile":
        out = {f"q_{qtoken(q)}": raw[:, k] for k, q in enumerate(rec.quantiles)}
        if 0.5 in rec.quantiles:
            out["value"] = raw[:, rec.quantiles.index(0.5)]
        return out
    qs, out = rec.quantiles or (None,), {}
    for h in rec.horizons:
        for j, q in enumerate(qs):
            name = f"value_h{h}" + (f"_q{qtoken(q)}" if q is not None else "")
            out[name] = raw[:, (h - 1) * len(qs) + j]
    return out


def _expected_row_scored(rec, raw, row, names):
    reading = rec.reading.value
    columns = [f for f in rec.fields if f.startswith(("recon_", "out_", "emb_"))]
    out = {name: raw[:, k] for k, name in enumerate(columns)}
    if reading == "reconstruction":
        out["reconstruction_error"] = row
        return out
    d = (row >= rec.threshold).astype(np.int64)
    out.update({"score": row, "prediction": d})
    if rec.classes:
        out["label"] = names[d]
    return out


def assert_matches(frame, want):
    assert set(want) == set(frame.columns), (sorted(want), list(frame.columns))
    for name, values in want.items():
        got = frame[name].to_numpy()
        if got.dtype == object or np.asarray(values).dtype.kind in "iub":
            assert list(got) == list(values), name
        else:
            np.testing.assert_allclose(got, values, rtol=1e-12, atol=1e-12, err_msg=name)


# ================================================================ 1. frame == from_stored

def test_the_cases_cover_every_reading(pkg):
    assert {reading for reading, _, _ in CASES.values()} == {r.value for r in pkg.records.Reading}


@pytest.mark.parametrize("name", CASES)
def test_frame_equals_from_stored_and_the_contract(pkg, name):
    reader = reader_of(pkg, name)
    assert reader.reading.value == CASES[name][0]
    raw, row = raw_of(reader)
    stored = stored_of(reader, raw, row)
    frame = reader.frame(head_of(pkg, raw, row))
    again = reader.from_stored(stored)
    pd.testing.assert_frame_equal(frame, again)
    record = manifest_of(CASES[name][1], CASES[name][2])["output"]
    answer = [f for f in record["fields"] if record["field_kinds"][f]["kind"] not in SHAPE_KINDS]
    assert list(frame.columns) == list(reader.fields) == answer
    assert_matches(frame, expected(reader.output, raw.astype(np.float64), row))


def test_from_stored_keeps_the_stored_index(pkg):
    reader = reader_of(pkg, "multiclass")
    raw, row = raw_of(reader, n=5)
    stored = stored_of(reader, raw, row).set_index("row_id")
    assert list(reader.from_stored(stored).index) == list(stored.index)


def test_the_shape_fields_are_the_predictors_not_the_readers(pkg):
    window = records_of(pkg, "sequence-binary").output
    assert window.fields[-2:] == ("answered", "warnings")
    graph = records_of(pkg, "graph-level").output
    assert graph.fields[0] == "graph_id"
    for rec in (window, graph):
        assert set(rec.fields) - set(rec.answer_fields) <= {"graph_id", "answered", "warnings"}
        assert "warnings" in rec.fields and "warnings" not in rec.answer_fields


# ================================================================ 2. batch independence

def test_a_binary_logit_is_read_alike_alone_and_in_any_batch(pkg):
    reader = reader_of(pkg, "binary")
    alone = reader.frame(head_of(pkg, np.array([[0.3]])))
    assert alone["p_1"].iloc[0] == pytest.approx(sigmoid(0.3))          # never the probability 0.3
    assert alone["prediction"].iloc[0] == 1 and alone["label"].iloc[0] == "1"
    # a batch whose logits all happen to lie in [0, 1] is still read as logits (v2.2's heuristic)
    inside = np.array([[0.2], [0.3], [0.9]])
    np.testing.assert_allclose(reader.frame(head_of(pkg, inside))["p_1"], sigmoid(inside[:, 0]))
    np.testing.assert_allclose(reader.frame(head_of(pkg, inside))["score"], inside[:, 0])


@pytest.mark.parametrize("name", CASES)
def test_no_answer_depends_on_the_other_rows(pkg, name):
    reader = reader_of(pkg, name)
    raw, row = raw_of(reader, n=9, seed=4)
    whole = reader.frame(head_of(pkg, raw, row))
    for i in range(len(raw)):
        one = reader.frame(head_of(pkg, raw[i:i + 1], None if row is None else row[i:i + 1]))
        pd.testing.assert_frame_equal(one, whole.iloc[i:i + 1].reset_index(drop=True))
    parts = [reader.frame(head_of(pkg, raw[a:b], None if row is None else row[a:b]))
             for a, b in ((0, 2), (2, 7), (7, 9))]
    pd.testing.assert_frame_equal(pd.concat(parts, ignore_index=True), whole)


def _batch_for(name, n):
    """The inputs a reconstruction row score reads, for `n` samples of a case's features."""
    features = CASES[name][1]["data_contract"]["features"]
    batch = {"x_num": torch.randn(n, sum(1 for f in features if f["role"] == "numeric"))}
    if any(f["role"] == "nominal" for f in features):
        batch["x_cat"] = torch.zeros(n, 1, dtype=torch.long)
    return batch


@pytest.mark.parametrize("name", CASES)
def test_a_head_of_no_sample_answers_an_empty_frame(pkg, name):
    """A window request whose rows all lack their history leaves the readers no sample."""
    reader = reader_of(pkg, name)
    width = reader.output.head_out
    head = reader.head(torch.zeros(0, width), _batch_for(name, 0), _Centred([0.0] * width))
    frame = reader.frame(head)
    assert len(frame) == 0 and list(frame.columns) == list(reader.fields)
    assert reader.warnings(head) == ()


def test_a_one_output_head_given_flat_is_one_column(pkg):
    reader = reader_of(pkg, "binary")
    head = reader.head(torch.tensor([0.3, -1.0, 2.0]))
    assert head.raw.shape == (3, 1)
    np.testing.assert_allclose(reader.frame(head)["score"], [0.3, -1.0, 2.0], rtol=1e-6)


def test_the_binary_threshold_is_the_records_on_the_probability_scale(pkg):
    reader = reader_of(pkg, "binary")
    p = np.array([0.405, 0.415, 0.52])            # either side of 0.41, and above 0.5
    z = np.log(p / (1 - p))[:, None]
    assert list(reader.frame(head_of(pkg, z))["prediction"]) == [0, 1, 1]
    default = reader_of(pkg, "binary", threshold={"applies": False})
    assert default.output.threshold is None
    assert list(default.frame(head_of(pkg, z))["prediction"]) == [0, 0, 1]   # 0.5, vega:default


def test_the_binary_logit_is_the_log_odds_of_the_second_class(pkg):
    """The trainer encodes the label through `LabelSet(task.classes)` (frames._class_ids), so
    the head's logit is the log-odds of `task.classes[1]`, the record's `positive`."""
    spec = CASES["binary-names"][1]
    labels = pkg.domain.LabelSet.from_spec(spec)
    assert labels.encode(["no", "yes"]) == (0, 1)                  # what the loss saw as 1
    reader = reader_of(pkg, "binary-names")
    assert reader.output.classes == ("no", "yes") and reader.output.positive == "yes"
    frame = reader.frame(head_of(pkg, np.array([[3.0], [-3.0]])))
    assert list(frame["label"]) == ["yes", "no"] and list(frame["prediction"]) == [1, 0]
    np.testing.assert_allclose(frame["p_yes"], sigmoid(np.array([3.0, -3.0])), rtol=1e-12)
    rows = frame.to_dict("records")
    assert [labels.at(reader.class_index(r)).name for r in rows] == ["yes", "no"]


def test_a_binary_task_whose_positive_is_listed_first_is_refused(pkg):
    """classes [yes, no] with positive yes: the trainer's 1 is `no`, pack's record reads
    `[no, yes]`, so answering would invert every probability and label (the round trip cannot
    see it: score and prediction agree on both sides). The reader refuses by name."""
    spec = spec_of({"kind": "binary", "label": "y", "classes": ["yes", "no"],
                    "positive": "yes"}, 1)
    assert pkg.domain.LabelSet.from_spec(spec).encode(["yes", "no"]) == (0, 1)
    manifest = manifest_of(spec, PROB)
    assert manifest["output"]["classes"] == ["no", "yes"]
    records = pkg.records.ServingRecords.of(spec, manifest, {})
    with pytest.raises(ValueError, match=r"log-odds of task\.classes\[1\] \('no'\); "
                                         r"task\.positive names 'yes': re-forge with the "
                                         r"positive class listed second"):
        pkg.values.reader_for(records)
    unnamed = spec_of({"kind": "binary", "label": "y", "classes": ["yes", "no"]}, 1)
    assert pkg.values.reader_for(pkg.records.ServingRecords.of(
        unnamed, manifest_of(unnamed, PROB), {})).output.positive == "no"   # the last class


# ================================================================ 3. ordinal

def test_ordinal_probabilities_are_a_distribution_with_the_clipping_warning(pkg):
    reader = reader_of(pkg, "ordinal")
    z = np.array([[2.0, -0.5, -2.0],         # rank-consistent: σ decreasing
                  [-2.0, 1.0, 0.5],          # not: P(mid) = σ(-2) − σ(1) < 0
                  [0.5, 0.5, 0.5]])          # ties: P(mid) = P(high) = 0, no clipping needed
    head = head_of(pkg, z)
    frame = reader.frame(head)
    p = frame[[f"p_{c}" for c in reader.output.classes]].to_numpy()
    assert (p >= 0).all()
    np.testing.assert_allclose(p.sum(axis=1), 1.0, rtol=0, atol=1e-12)
    assert reader.warnings(head) == ((), (pkg.heads.CLIPPED,), ())
    assert pkg.heads.CLIPPED == "cumulative logits not monotone; class probabilities clipped"
    np.testing.assert_allclose(frame.loc[0, ["cum_low", "cum_mid", "cum_high"]].to_numpy(float),
                               sigmoid(z[0]), rtol=1e-12)
    assert list(frame["label"]) == ["mid", "high", "top"]           # Σ_j [σ(z_j) ≥ 0.5]
    assert reader.warnings(head_of(pkg, z[:1])) == ((),)


def test_other_readings_add_no_warning_of_their_own(pkg):
    for name in ("binary", "multiclass", "regression", "anomaly-svdd"):
        reader = reader_of(pkg, name)
        raw, row = raw_of(reader, n=3)
        assert reader.warnings(head_of(pkg, raw, row)) == ((), (), ())


# ================================================================ 4. the anomaly readings

def test_the_anomaly_decision_is_on_the_score_scale(pkg):
    reader = reader_of(pkg, "anomaly-reconstruction")
    assert reader.output.threshold == 0.8 and reader.output.threshold_scale == "score"
    row = np.array([0.5, 0.79, 0.8, 0.85, 5.0])
    frame = reader.frame(head_of(pkg, np.zeros((5, 4)), row))
    assert list(frame["prediction"]) == [0, 0, 1, 1, 1]
    assert list(frame["label"]) == ["0", "0", "1", "1", "1"]
    squashed = (sigmoid(row) >= 0.8).astype(int)             # what a squashed reading would say
    assert list(frame["prediction"]) != list(squashed)
    np.testing.assert_array_equal(frame["score"], row)


def test_the_svdd_score_is_the_distance_to_the_models_centre(pkg):
    reader = reader_of(pkg, "anomaly-svdd")
    out = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5, 0.5], [0.0] * 5])
    centre = [0.5] * 5
    head = reader.head(out, None, _Centred(centre))
    want = ((out.numpy().astype(np.float64) - 0.5) ** 2).sum(axis=1)
    np.testing.assert_allclose(head.row_score, want, rtol=1e-6)
    frame = reader.frame(head)
    assert list(frame["prediction"]) == [0, 0, 0]                    # 1.25 < 1.5
    assert list(frame.columns) == ["score", "prediction", "emb_0", "emb_1", "emb_2", "emb_3",
                                   "emb_4"]
    np.testing.assert_allclose(frame[[f"emb_{j}" for j in range(5)]].to_numpy(), out.numpy())
    with pytest.raises(ValueError, match="centre"):
        reader.head(out, None, None)


def test_the_reconstruction_error_is_mse_plus_the_nominal_cross_entropy(pkg):
    reader = reader_of(pkg, "reconstruction-nominal")
    assert reader.fields == ("reconstruction_error",) + tuple(f"out_{k}" for k in range(7))
    g = torch.Generator().manual_seed(3)
    out = torch.randn(6, 7, generator=g)
    batch = {"x_num": torch.randn(6, 4, generator=g), "x_cat": torch.tensor([[0], [1], [2],
                                                                             [2], [1], [0]])}
    head = reader.head(out, batch, None)
    o, x = out.numpy().astype(np.float64), batch["x_num"].numpy().astype(np.float64)
    logp = o[:, 4:] - np.log(np.exp(o[:, 4:]).sum(axis=1, keepdims=True))
    ce = -logp[np.arange(6), batch["x_cat"][:, 0].numpy()]
    np.testing.assert_allclose(head.row_score, ((o[:, :4] - x) ** 2).mean(axis=1) + ce, rtol=1e-5)
    np.testing.assert_allclose(reader.frame(head)["reconstruction_error"], head.row_score)


# ================================================================ 5. the one source of links

LINK_WORDS = re.compile(r"(?i)sigmoid|softmax|expit|logistic|^exp$|^tanh$")


def _code_names(text):
    """Every NAME token of the module (strings and comments are not code)."""
    return [tok.string for tok in tokenize.generate_tokens(io.StringIO(text).readline)
            if tok.type == tokenize.NAME]


def test_no_reader_reimplements_a_link(pkg):
    seeded = 'p = torch.sigmoid(z)  # "softmax" in a comment\nq = np.exp(-z)\nr = "sigmoid"\n'
    assert sorted(n for n in _code_names(seeded) if LINK_WORDS.search(n)) == ["exp", "sigmoid"]
    for name in FILES:
        text = (pkg.root / "serving" / name).read_text(encoding="utf-8")
        hits = sorted({n for n in _code_names(text) if LINK_WORDS.search(n)})
        assert not hits, f"serving/{name} computes a link itself: {hits} (R-B7)"
    for name in ("heads.py", "values.py"):
        tree = ast.parse((pkg.root / "serving" / name).read_text(encoding="utf-8"))
        # the spec's readout, through the model module's published API (its `__init__.py`)
        sources = {(node.module, node.level, alias.name) for node in ast.walk(tree)
                   if isinstance(node, ast.ImportFrom) for alias in node.names}
        assert ("model", 2, "TaskReadout") in sources, f"serving/{name} must read the spec's readout"


class _Spy:
    """The readout, recording which of its readings a reader asked for."""

    METHODS = ("score", "proba", "reduce", "row_score", "clipped")

    def __init__(self, inner):
        self._inner, self.calls = inner, set()

    def __getattr__(self, name):
        attr = getattr(self._inner, name)
        if name not in self.METHODS:
            return attr

        def called(*args, **kwargs):
            self.calls.add(name)
            return attr(*args, **kwargs)
        return called


READOUT_CALLS = {"binary": {"proba"}, "multiclass": {"proba", "reduce"},
                 "multilabel": {"proba"}, "ordinal": {"proba", "reduce"}, "regression": {"score"},
                 "quantile": {"score"}, "forecast": {"score"}, "reconstruction": {"row_score"},
                 "anomaly-reconstruction": {"row_score"}, "anomaly-svdd": {"row_score"}}


@pytest.mark.parametrize("name", [n for n in CASES if n != "forecast-quantile"])
def test_every_reader_derives_through_the_readout(pkg, name):
    reader = reader_of(pkg, name)
    spy = _Spy(reader._readout)
    reader._readout = spy
    out = torch.randn(4, reader.output.head_out)
    reader.frame(reader.head(out, _batch_for(name, 4), _Centred([0.0] * reader.output.head_out)))
    assert READOUT_CALLS[reader.reading.value] <= spy.calls, spy.calls


def test_the_readers_link_is_the_manifests_link(pkg):
    for name in CASES:
        reader = reader_of(pkg, name)
        assert reader.output.link == reader._readout.link, name


# ================================================================ 6. records and refusals

def _manifest(name):
    return manifest_of(CASES[name][1], CASES[name][2])


def test_a_v1_output_record_is_refused(pkg):
    v1 = {"output": {"kind": "multiclass", "threshold": 0.5, "fields": ["label", "p_a"]}}
    with pytest.raises(ValueError, match="vega/output@1; re-pack with vega-pack v2.3"):
        pkg.records.OutputRecord.from_manifest(v1)
    with pytest.raises(ValueError, match="vega/output@1"):
        pkg.records.OutputRecord.from_manifest({})


@pytest.mark.parametrize("change, message", [
    (lambda o: o.update(reading="multi-class"), "is not one of"),
    (lambda o: o["field_kinds"].pop("p_a"), "have no kind"),
    (lambda o: o["compared"].append(["logit_z", "score_9"]), "not \\[field, source\\]"),
    (lambda o: o.update(head_out=0), "positive integer"),
    (lambda o: o.update(head_out=True), "positive integer"),
    (lambda o: o["compared"][2].__setitem__(1, "score_7"),
     "compares 'logit_c' with 'score_7', and head_out is 3: the stored head is score_0…score_2"),
    (lambda o: o["compared"][0].__setitem__(1, "score"),
     "compares 'logit_a' with 'score', and head_out is 3"),
])
def test_a_malformed_output_record_is_refused_by_name(pkg, change, message):
    manifest = json.loads(json.dumps(_manifest("multiclass")))
    assert manifest["output"]["compared"][2] == ["logit_c", "score_2"]
    change(manifest["output"])
    with pytest.raises(ValueError, match=message):
        pkg.values.reader_for(pkg.records.ServingRecords.of(CASES["multiclass"][1], manifest, {}))


def test_the_output_record_is_the_manifests(pkg):
    manifest = _manifest("quantile")
    rec = pkg.records.OutputRecord.from_manifest(manifest)
    out = manifest["output"]
    assert rec.reading is pkg.records.Reading.QUANTILE and rec.kind == "regression"
    assert rec.fields == tuple(out["fields"]) and rec.quantiles == (0.1, 0.5, 0.9)
    assert rec.compared == tuple(tuple(p) for p in out["compared"])
    assert rec.of("q_10") == 0.1 and rec.fields_of("quantile") == ("q_10", "q_50", "q_90")
    assert rec.score_field == "value" and rec.truth["type"] == "number"
    with pytest.raises(dataclasses.FrozenInstanceError):
        rec.head_out = 9
    with pytest.raises(TypeError):
        rec.field_kinds["value"] = {"kind": "score"}


def test_the_sample_record_per_shape(pkg):
    SampleRecord, Shape = pkg.records.SampleRecord, pkg.domain.SampleShape
    assert SampleRecord.from_manifest({}) == SampleRecord() and SampleRecord().shape is Shape.ROW
    assert SampleRecord.from_manifest({"sample": {"shape": "row"}}).shape is Shape.ROW
    recipe = {"steps": [{"op": "lag", "params": {"lags": [1, 3]}}]}
    spec = CASES["forecast"][1]
    window = SampleRecord.from_manifest({"sample": orec.sample_record(spec, recipe, None)})
    assert (window.shape, window.length, window.horizon, window.lag_depth) == \
        (Shape.WINDOW, 6, 3, 3)
    assert window.history_rows == 5 + 3 and window.autoregressive and window.series_by == "series"
    unstated = SampleRecord.from_manifest({"sample": {"shape": "window", "length": 4,
                                                      "lag_depth": 2}})
    assert unstated.history_rows == 5 and unstated.horizon is None
    graph = SampleRecord.from_manifest({"sample": orec.sample_record(
        CASES["graph-level"][1], {}, {"n_nodes": {"train": 30, "valid": 10}, "n_edges": 55})})
    assert (graph.shape, graph.level, graph.graph_id) == (Shape.GRAPH, "graph", "g")
    assert graph.reference["nodes"] == 40 and isinstance(graph.reference, MappingProxyType)
    for bad, message in (({"shape": "tree"}, "is not one of"),
                         ({"shape": "window", "length": 0}, "positive integer length"),
                         ({"shape": "window"}, "positive integer length")):
        with pytest.raises(ValueError, match=message):
            SampleRecord.from_manifest({"sample": bad})


def test_serving_records_hold_one_model(pkg):
    spec = CASES["binary"][1]
    manifest = _manifest("binary")
    records = pkg.records.ServingRecords.of(spec, manifest, {"input": [{"name": "x0"}]}, "m1")
    assert isinstance(records.spec, pkg.domain.NetSpec)
    same = pkg.records.ServingRecords.of(records.spec, manifest, {}, None)
    assert same.output == records.output and same.sample == records.sample
    assert pkg.records.thawed(records.manifest) == json.loads(json.dumps(manifest))
    assert records.versions.contract == spec["input"]["manifest_digest"]
    assert records.versions.model == "m1"
    with pytest.raises(dataclasses.FrozenInstanceError):
        records.model_version = "m2"
    with pytest.raises(TypeError):
        records.manifest["output"] = {}
    with pytest.raises(TypeError):
        records.schema["input"][0]["name"] = "x9"
    with pytest.raises(ValueError, match="another model"):
        pkg.records.ServingRecords.of(CASES["multiclass"][1], manifest, {})
    with pytest.raises(ValueError, match="window samples and the spec a binary task over row"):
        pkg.records.ServingRecords.of(spec, _manifest("sequence-binary") | {
            "output": manifest["output"]}, {})


def test_a_reader_refuses_a_record_its_readout_reads_otherwise(pkg):
    # a regression record (one output) over a quantile head of three
    with pytest.raises(ValueError, match="MANIFEST.output.head_out"):
        reader_of(pkg, "regression", spec=CASES["quantile"][1])
    # an svdd record (distance) over a reconstruction head
    with pytest.raises(ValueError, match="another model"):
        pkg.values.reader_for(pkg.records.ServingRecords.of(
            spec_of({"kind": "anomaly"}, 5, "reconstruction"), _manifest("anomaly-svdd"), {}))
    # a binary record whose classes the spec does not list
    other = spec_of({"kind": "binary", "label": "y", "classes": ["yes", "no"]}, 1)
    with pytest.raises(ValueError, match="does not list"):
        reader_of(pkg, "binary", spec=other)


@pytest.mark.parametrize("name", ["anomaly-reconstruction", "anomaly-unlabelled", "anomaly-svdd"])
def test_an_anomaly_record_without_a_threshold_answers_the_score_undecided(pkg, name):
    """`threshold.json.applies == false` gives `threshold: null` (bundle-contract §2): the
    score is answered, the decision and its label are null, and no cut is guessed."""
    reader = reader_of(pkg, name, threshold={"applies": False})
    assert reader.output.threshold is None
    raw, row = raw_of(reader, n=4)
    frame = reader.frame(head_of(pkg, raw, row))
    np.testing.assert_array_equal(frame["score"], row.astype(np.float64))
    assert list(frame["prediction"]) == [None] * 4
    assert "label" not in frame.columns or list(frame["label"]) == [None] * 4
    pd.testing.assert_frame_equal(frame, reader.from_stored(stored_of(reader, raw, row)))
    assert [reader.class_index(rec) for rec in frame.to_dict("records")] == [None] * 4


def test_from_stored_refuses_a_layout_without_its_columns(pkg):
    reader = reader_of(pkg, "multiclass")
    raw, row = raw_of(reader, n=4)
    stored = stored_of(reader, raw, row).drop(columns=["score_2"])
    with pytest.raises(ValueError, match="score_2"):
        reader.from_stored(stored)
    anomaly = reader_of(pkg, "anomaly-svdd")
    raw, row = raw_of(anomaly, n=4)
    with pytest.raises(ValueError, match="stored no per-row score for a anomaly-svdd head; "
                                         "re-evaluate with vega-eval v2.3"):
        anomaly.from_stored(stored_of(anomaly, raw, row).drop(columns=["row_score"]))
    with pytest.raises(ValueError, match="carries none"):
        anomaly.frame(head_of(pkg, raw))
    with pytest.raises(ValueError, match="head_out is 5"):
        anomaly.frame(head_of(pkg, raw[:, :3], row))


def test_from_stored_takes_the_evaluations_decision_as_stored(pkg):
    for name in ("binary", "anomaly-reconstruction"):
        reader = reader_of(pkg, name)
        raw, row = raw_of(reader, n=8, seed=5)
        stored = stored_of(reader, raw, row)
        stored.loc[3, "prediction"] = 1 - stored.loc[3, "prediction"]
        frame, again = reader.frame(head_of(pkg, raw, row)), reader.from_stored(stored)
        assert again.loc[3, "prediction"] == stored.loc[3, "prediction"]
        assert again.loc[3, "label"] != frame.loc[3, "label"]
        assert (again["prediction"] != frame["prediction"]).sum() == 1   # the round trip sees it


# ================================================================ 7. Prediction values

SCORE_VECTOR = {"binary": ["score"], "multiclass": ["logit_a", "logit_b", "logit_c"],
                "ordinal": ["logit_cum_low", "logit_cum_mid", "logit_cum_high"],
                "regression": ["value"], "quantile": ["q_10", "q_50", "q_90"],
                "forecast": ["value_h1", "value_h2", "value_h3"],
                "reconstruction": ["reconstruction_error"],
                "anomaly-svdd": ["score"], "multilabel": ["logit_red", "logit_green",
                                                          "logit_blue"]}


@pytest.mark.parametrize("name", CASES)
def test_the_domain_prediction_carries_the_readings_decision(pkg, name):
    reader = reader_of(pkg, name)
    raw, row = raw_of(reader, n=6, seed=2)
    frame = reader.frame(head_of(pkg, raw, row))
    labels = pkg.domain.LabelSet.from_spec(CASES[name][1])
    for i, rec in enumerate(frame.to_dict("records")):
        k = reader.class_index(rec)
        prediction = pkg.domain.Prediction(
            row_id=i, score=reader.score_vector(rec), label=None if k is None else labels.at(k),
            versions=pkg.domain.Versions("c", "m"))
        if "label" in frame.columns:
            assert str(prediction.label.name) == rec["label"]
        else:
            assert prediction.label is None
        if name in SCORE_VECTOR:
            assert list(prediction.score) == [float(rec[f]) for f in SCORE_VECTOR[name]]


# ================================================================ 8. the schema guard

GUARD_SCHEMA = {"input": [
    {"name": "amount", "dtype": "float", "required": True, "range_seen": [0.5, 90.0]},
    {"name": "code", "dtype": "string", "required": True, "pattern": "^[0-9]+$"},
    {"name": "region", "dtype": "string", "required": False},
    {"name": "ratio", "dtype": "float", "required": True, "range_seen": [0, 1],
     "non_finite_seen": False},
    {"name": "load", "dtype": "float", "required": False, "range_seen": [0, 10],
     "non_finite_seen": True},
    {"name": "gone", "dtype": "float", "required": True},
    {"name": "coded", "dtype": "string", "required": False}]}


def _guard_rows():
    return pd.DataFrame({"amount": [1.0, 100.0, -3.0, np.nan],
                         "code": pd.Series(["80", 80.0, "x", None], dtype=object),
                         "ratio": pd.Series([0.5, np.inf, 2.0, "abc"], dtype=object),
                         "load": [np.inf, 5.0, 11.0, -np.inf],
                         "coded": [443.0, np.nan, 22.0, 80.0]})


GUARD_WANT = [
    ["required column 'gone' is missing",
     "coded arrived as float; the schema declares it text (read the file with the schema's "
     "dtypes)"],
    ["amount outside the range seen in training [0.5, 90.0]",
     "code arrived as float; the schema declares it text (read the file with the schema's "
     "dtypes)",
     "ratio is non-finite, which training never showed in this column",
     "required column 'gone' is missing"],
    ["amount outside the range seen in training [0.5, 90.0]",
     "ratio outside the range seen in training [0, 1]",
     "load outside the range seen in training [0, 10]",
     "required column 'gone' is missing",
     "coded arrived as float; the schema declares it text (read the file with the schema's "
     "dtypes)"],
    ["required column 'gone' is missing",
     "coded arrived as float; the schema declares it text (read the file with the schema's "
     "dtypes)"],
]


def _v22_schema_warnings():
    """v2.2's `Predictor._schema_warnings`, compiled from its template, or None once S07 has
    deleted it."""
    if not V22_PREDICT.is_file():
        return None
    tree = ast.parse(V22_PREDICT.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Predictor")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
              and n.name == "_schema_warnings")
    namespace = {"np": np, "pd": pd}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(V22_PREDICT), "exec"), namespace)
    return namespace["_schema_warnings"]


def test_the_schema_guard_keeps_the_warning_texts(pkg):
    guard = pkg.guards.SchemaGuard(GUARD_SCHEMA)
    rows = _guard_rows()
    got = guard.warnings(rows)
    assert got == GUARD_WANT
    frozen = pkg.records.ServingRecords.of(CASES["binary"][1], _manifest("binary"), GUARD_SCHEMA)
    assert pkg.guards.SchemaGuard(frozen.schema).warnings(rows) == GUARD_WANT
    assert guard.warnings(rows.set_index(pd.Index([9, 3, 5, 1]))) == GUARD_WANT
    assert pkg.guards.SchemaGuard({}).warnings(rows) == [[], [], [], []]
    legacy = _v22_schema_warnings()
    if legacy is not None:
        assert got == legacy(SimpleNamespace(_schema=GUARD_SCHEMA), rows)


# ================================================================ 9. size, state, imports, literals

def test_the_templates_meet_c11_to_c13_and_their_budgets(pkg):
    for name in FILES:
        lines = len((PACKAGE_TEMPLATES / name).read_text(encoding="utf-8").splitlines())
        assert lines <= BUDGETS[name], f"package/{name} is {lines} lines > D2's {BUDGETS[name]}"
        text = (pkg.root / "serving" / name).read_text(encoding="utf-8")
        assert "{{" not in text, f"serving/{name} keeps a placeholder"
        findings = conformance.size_findings(ast.parse(text), f"serving/{name}")
        assert not findings, "\n".join(f.message() for f in findings)
    proc = subprocess.run([sys.executable, str(SCRIPTS / "conformance.py"),
                           "--size-severity", "error", "--size-only",
                           *[str(pkg.root / "serving" / n) for n in FILES]],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr


ALLOWED_ROOTS = {"__future__", "dataclasses", "enum", "math", "re", "types", "typing", "numpy",
                 "pandas", "torch"}
ALLOWED_RELATIVE = {("domain", 2), ("model", 2), ("records", 1), ("heads", 1),
                    ("samples", 1)}


def test_state_and_imports_follow_the_support_arrows(pkg):
    for name in FILES:
        tree = ast.parse((pkg.root / "serving" / name).read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                names = [t.id for t in targets if isinstance(t, ast.Name)]
                if isinstance(node.value, conformance.MUTABLE_NODES) and names != ["__all__"]:
                    assert all(n.lstrip("_").isupper() for n in names), f"C7: {names} in {name}"
            if isinstance(node, ast.Import):
                assert {a.name.split(".")[0] for a in node.names} <= ALLOWED_ROOTS, name
            if isinstance(node, ast.ImportFrom):
                if node.level:
                    assert (node.module, node.level) in ALLOWED_RELATIVE, (name, node.module)
                else:
                    assert node.module.split(".")[0] in ALLOWED_ROOTS, (name, node.module)
    heads = ast.parse((pkg.root / "serving" / "heads.py").read_text(encoding="utf-8"))
    assert not any(isinstance(n, ast.ImportFrom) and n.module == "values"
                   for n in ast.walk(heads)), "heads.py never imports values.py (no cycle)"


def test_one_head_output_is_samples_and_a_read_only_head_is_never_written(pkg):
    """D2 §4.3: `HeadOutput` is samples.py's; the readers define none. Its `raw` may be a
    read-only view (a stored frame's values under pandas copy-on-write): the readers copy it
    into torch (never `from_numpy` / `as_tensor`, which warn and share it) and never write it."""
    assert pkg.heads.HeadOutput is pkg.samples.HeadOutput
    for name in FILES:
        text = (pkg.root / "serving" / name).read_text(encoding="utf-8")
        tree = ast.parse(text)
        assert not any(isinstance(n, ast.ClassDef) and n.name == "HeadOutput"
                       for n in ast.walk(tree)), f"serving/{name} defines a second HeadOutput"
        shared = {n for n in _code_names(text) if n in ("from_numpy", "as_tensor")}
        assert not shared, f"serving/{name} shares a head array with torch: {shared}"
    for name in CASES:
        reader = reader_of(pkg, name)
        raw, row = raw_of(reader, n=5, seed=6)
        raw = raw.astype(np.float64)
        raw.setflags(write=False)
        head = pkg.samples.HeadOutput(raw, row)
        before = head.raw.copy()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            reader.frame(head)
            reader.warnings(head)
        np.testing.assert_array_equal(head.raw, before)


def test_the_literal_gate_finds_nothing_in_the_four_templates():
    """R-F10: `lint_templates.py` (T04) is the one deny-list gate."""
    assert LINT.is_file(), f"{LINT} (T04) is on this tree"
    proc = subprocess.run([sys.executable, str(LINT), *[str(PACKAGE_TEMPLATES / n)
                                                        for n in FILES]],
                          capture_output=True, text=True, cwd=str(LINT.parent))
    assert proc.returncode == 0, proc.stdout + proc.stderr


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
