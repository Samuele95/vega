#!/usr/bin/env python3
"""
make_predictions.py — the answers the page's task modules read, one case per reading, made by
the serving machinery itself (tests/registry.test.js reads predictions.json).

    python3 make_predictions.py [--check]        # write predictions.json, or compare with it

For each case a minimal spec goes through vega-pack's `output_record.output_record` (the
bundle's `MANIFEST.output`, `vega/output@2`: the reading, its fields and their kinds, the
threshold on its scale), a small frame of the Predictor's fields is made with values that hold
together (a label is the arg-max of its probabilities, a flag is the score at or above the
threshold, the ordinal classes are what the cumulative steps passed lead to, a window row can be
unanswered), and the serving template's `serve/answers.py` `AnswerCodec.rows` writes each row's
`RowPrediction` (D2 §5.2) — once without and once with every `include` (logits,
reconstruction, embedding). So the keys the page is tested on are the keys the service writes
(BUILD-PLAN R-C4), never a hand-typed copy of them.

A case also carries what the page reads beside the answers: `/v1/schema` (`classes`, `output`,
`sample`, `truth`, as `ApiContract.describe` serves them), `page.json#lane` (the lane the page
renders), the truth a reader could send (one per row), a `class_roles@1` block when the lane
has roles, a results `breakdown@1` with its story values, the attribution axis columns, the
reference quantiles of the reconstruction lanes, and for row 0 the members of
`vega/explanation@2` a task module reads (`prediction`, `inputs`, `outputs`).

Exit 0: written (or, with --check, the file is this script's output). 1: --check found it
stale. 2: the machinery refused a case.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SKILLS = HERE.parents[5]
sys.path.insert(0, str(SKILLS / "vega-pack" / "scripts"))

import output_record as orec  # noqa: E402

ANSWERS = SKILLS / "vega-forge" / "scripts" / "backends" / "pytorch" / "templates" / "serving" / "serve" / "answers.py"
OUT = HERE / "predictions.json"
N = 6
INCLUDES = ("logits", "reconstruction", "embedding")
FEATURES = [{"name": "a", "role": "numeric"}, {"name": "b", "role": "numeric"},
            {"name": "zone", "role": "nominal", "cardinality": 3}]
WINDOW = {"shape": "window", "length": 6, "stride": 1, "series_by": "station", "order_by": "ts",
          "context": "previous-part"}
QUANTILES = [0.1, 0.5, 0.9]


def _codec():
    spec = importlib.util.spec_from_file_location("vega_serving_answers", ANSWERS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.AnswerCodec


def spec_of(kind, head, loss, sample=None, **task):
    return {"task": {"kind": kind, **task},
            "data_contract": {"features": FEATURES, "sample": sample or {"shape": "row"}},
            "blocks": [{"type": "dense", "params": {"units": 8}}, {"type": "head", "params": {"out": head}}],
            "loss": {"name": loss} if isinstance(loss, str) else loss}


def lane_of(task, base, view_key, link, shape="row", level=None, score_source=None, goal="predict"):
    return {"task": task, "base": base, "view_key": view_key, "shape": shape, "level": level, "link": link,
            "family": "mlp-plain", "goal": goal, "score_source": score_source,
            "source": "netspec:task, netspec:data_contract.sample, netspec:family, netspec:loss"}


# (id, spec, threshold.json, truth column, page lane)
CASES = [
    ("multiclass", spec_of("multiclass", 3, "cross_entropy", classes=["calm", "gust", "storm"]), None, "kind",
     lane_of("multiclass", None, "classes", "softmax")),
    ("binary", spec_of("binary", 1, "bce_with_logits", classes=["ok", "delayed"], positive="delayed"),
     {"value": 0.4, "source": "valid-optimum", "applies": True}, "delayed",
     lane_of("binary", None, "binary", "sigmoid")),
    ("multilabel", spec_of("multilabel", 3, "bce_with_logits", classes=["red", "green", "blue"],
                           label_format={"kind": "delimited", "sep": ";"}),
     {"value": 0.5, "per_label": {"red": 0.3, "green": 0.6, "blue": 0.5}, "source": "valid-optimum", "scale": "probability"},
     "tags", lane_of("multilabel", None, "labels", "sigmoid_each")),
    ("ordinal", spec_of("ordinal", 2, "cumulative_link", classes=["low", "mid", "high"]), None, "grade",
     lane_of("ordinal", None, "ordinal", "cumulative")),
    ("regression", spec_of("regression", 1, "mse"), None, "price", lane_of("regression", None, "value", "identity")),
    ("quantile", spec_of("regression", 3, {"name": "quantile", "params": {"quantiles": QUANTILES}}), None, "price",
     lane_of("regression", None, "value", "identity")),
    ("forecast", spec_of("forecast", 3, "mse", dict(WINDOW, horizon=3)), None, "load",
     lane_of("forecast", None, "horizon", "horizon", shape="window")),
    ("forecast-quantile", spec_of("forecast", 6, {"name": "quantile", "params": {"quantiles": QUANTILES}},
                                  dict(WINDOW, horizon=2)), None, "load",
     lane_of("forecast", None, "horizon", "horizon", shape="window")),
    ("reconstruction", spec_of("unsupervised", 3, "reconstruction"), None, None,
     lane_of("unsupervised", None, "reconstruction", "reconstruction", goal="reconstruct")),
    ("anomaly-reconstruction", spec_of("anomaly", 3, "reconstruction"), {"value": 0.2, "applies": True}, None,
     lane_of("anomaly", None, "anomaly", "reconstruction", score_source="reconstruction", goal="anomaly")),
    ("anomaly-svdd", spec_of("anomaly", 2, "svdd", classes=["normal", "fraud"], positive="fraud"),
     {"value": 1.0, "applies": True, "scale": "score"}, "is_fraud",
     lane_of("anomaly", None, "anomaly", "distance", score_source="distance", goal="anomaly")),
    ("sequence-binary", spec_of("sequence-classification", 1, "bce_with_logits", WINDOW, base_kind="binary",
                                classes=["ok", "delayed"], positive="delayed"), {"value": 0.6}, "event",
     lane_of("sequence-classification", "binary", "binary", "sigmoid", shape="window")),
    ("sequence-ordinal", spec_of("sequence-regression", 2, "cumulative_link", WINDOW, base_kind="ordinal",
                                 classes=["low", "mid", "high"]), None, "level",
     lane_of("sequence-regression", "ordinal", "ordinal", "cumulative", shape="window")),
    ("graph-multiclass", spec_of("multiclass", 3, "cross_entropy", {"shape": "graph", "level": "graph", "graph_id": "mol"},
                                 classes=["calm", "gust", "storm"]), None, "kind",
     lane_of("multiclass", None, "classes", "softmax", shape="graph", level="graph")),
]


# --------------------------------------------------------------------------- the frame

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


class Frame:
    """The Predictor's frame for one record: every field the record names, values that agree."""

    def __init__(self, record: dict, rng: np.random.Generator):
        self.record, self.rng, self.cols = record, rng, {}
        self.kinds = record["field_kinds"]

    def of_kind(self, kind):
        return [f for f in self.record["fields"] if self.kinds[f]["kind"] == kind]

    def build(self) -> pd.DataFrame:
        reading = self.record["reading"]
        getattr(self, "_" + reading.replace("-", "_"))()
        for f in self.of_kind("group"):
            self.cols[f] = [f"g-{k}" for k in range(N)]
        for f in self.of_kind("answered"):
            self.cols[f] = [k != N - 1 for k in range(N)]          # the last window lacked history
        self.cols["warnings"] = [["unseen category 'x' in column 'zone'"] if k == 1 else [] for k in range(N)]
        return pd.DataFrame({f: self.cols[f] for f in self.record["fields"]})

    def _classes_from(self, probs, logits=None):
        classes = self.record["classes"]
        for f in self.of_kind("probability"):
            self.cols[f] = probs[:, classes.index(str(self.kinds[f]["of"]))].tolist()
        if logits is not None:
            for f in self.of_kind("raw"):
                self.cols[f] = logits[:, classes.index(str(self.kinds[f]["of"]))].tolist()

    def _multiclass(self):
        z = self.rng.normal(0, 2, (N, len(self.record["classes"])))
        self._classes_from(softmax(z), z)
        self.cols["label"] = [self.record["classes"][i] for i in z.argmax(axis=1)]

    def _binary(self):
        classes, positive, t = self.record["classes"], self.record["positive"], self.record["threshold"] or 0.5
        z = self.rng.normal(0, 2, N)
        p = sigmoid(z)
        probs = np.stack([1 - p if c != positive else p for c in classes], axis=1)
        for f in self.of_kind("probability"):
            self.cols[f] = probs[:, classes.index(str(self.kinds[f]["of"]))].tolist()
        for f in self.of_kind("raw"):
            self.cols[f] = z.tolist()
        negative = next(c for c in classes if c != positive)
        self.cols["prediction"] = [bool(x >= t) for x in p]
        self.cols["label"] = [positive if x >= t else negative for x in p]

    def _multilabel(self):
        labels, per = self.record["classes"], self.record.get("threshold_per_label") or {}
        z = self.rng.normal(0, 2, (N, len(labels)))
        p = sigmoid(z)
        for f in self.of_kind("probability"):
            self.cols[f] = p[:, labels.index(str(self.kinds[f]["of"]))].tolist()
        for f in self.of_kind("raw"):
            self.cols[f] = z[:, labels.index(str(self.kinds[f]["of"]))].tolist()
        for f in self.of_kind("flag"):
            j = labels.index(str(self.kinds[f]["of"]))
            t = per.get(labels[j], self.record["threshold"])
            self.cols[f] = [bool(x >= t) for x in p[:, j]]

    def _ordinal(self):
        classes = self.record["classes"]
        z = np.sort(self.rng.normal(0, 2, (N, len(classes) - 1)), axis=1)[:, ::-1]     # rank-consistent
        s = sigmoid(z)
        probs = np.concatenate([1 - s[:, :1], s[:, :-1] - s[:, 1:], s[:, -1:]], axis=1)
        self._classes_from(probs)
        steps = classes[:-1]
        for f in self.of_kind("cumulative"):
            self.cols[f] = s[:, steps.index(str(self.kinds[f]["of"]))].tolist()
        for f in self.of_kind("raw"):
            self.cols[f] = z[:, steps.index(str(self.kinds[f]["of"]))].tolist()
        self.cols["label"] = [classes[int((row >= 0.5).sum())] for row in s]

    def _regression(self):
        self.cols["value"] = (200 + 50 * self.rng.normal(size=N)).round(3).tolist()

    def _quantile(self):
        mid = 200 + 50 * self.rng.normal(size=N)
        for f in self.of_kind("quantile"):
            q = float(self.kinds[f]["of"])
            self.cols[f] = (mid + 60 * (q - 0.5)).round(3).tolist()
        if "value" in self.record["fields"]:
            self.cols["value"] = mid.round(3).tolist()

    def _forecast(self):
        base = 10 + self.rng.normal(size=N)
        for f in self.of_kind("horizon"):
            step, _, q = str(self.kinds[f]["of"]).partition("/")
            self.cols[f] = (base + 0.5 * int(step) + (4 * (float(q) - 0.5) if q else 0)).round(3).tolist()

    def _recon(self):
        x = self.rng.normal(size=(N, len(self.of_kind("recon"))))
        xhat = x + 0.3 * self.rng.normal(size=x.shape)
        for j, f in enumerate(self.of_kind("recon")):
            self.cols[f] = xhat[:, j].round(4).tolist()
        self.inputs = x
        return ((xhat - x) ** 2).mean(axis=1)

    def _reconstruction(self):
        self.cols["reconstruction_error"] = self._recon().round(5).tolist()

    def _alarm(self, score):
        t = self.record["threshold"]
        for f in self.of_kind("score"):
            self.cols[f] = score.round(5).tolist()
        for f in self.of_kind("alarm"):
            self.cols[f] = [bool(s >= t) for s in score]
        if "label" in self.record["fields"]:
            classes, positive = self.record["classes"], self.record["positive"]
            negative = next(c for c in classes if c != positive)
            self.cols["label"] = [positive if s >= t else negative for s in score]

    def _anomaly_reconstruction(self):
        score = self._recon()
        score[0] = self.record["threshold"] * 2         # one row flagged, whatever the draw
        self._alarm(score)

    def _anomaly_svdd(self):
        emb = self.rng.normal(0, 0.8, (N, len(self.of_kind("embedding"))))
        for f in self.of_kind("embedding"):
            self.cols[f] = emb[:, int(self.kinds[f]["of"])].round(4).tolist()
        score = (emb ** 2).sum(axis=1)
        score[0] = self.record["threshold"] * 3
        self._alarm(score)


# --------------------------------------------------------------------------- what the page reads beside

def truth_of(case_id, record, frame: pd.DataFrame):
    """The truth a reader could send, one per row: the answered value for most rows, another for
    two of them (so every summary has hits and misses)."""
    reading, classes = record["reading"], record["classes"] or []
    if reading in ("multiclass", "binary", "ordinal"):
        out = [str(v) for v in frame["label"]]
        out[2] = next(c for c in classes if c != out[2])
        out[4] = classes[0]
        return out
    if reading == "multilabel":
        out = [[str(record["field_kinds"][f]["of"]) for f in record["fields"]
                if record["field_kinds"][f]["kind"] == "flag" and frame[f][k]] for k in range(N)]
        out[3] = classes[:1]
        return out
    if reading in ("regression", "quantile"):
        col = "value" if "value" in frame else record["fields"][0]
        return [round(float(v) + (7 if k % 2 else -4), 3) for k, v in enumerate(frame[col])]
    if reading == "forecast":
        first = [f for f in record["fields"] if record["field_kinds"][f]["kind"] == "horizon"][0]
        return [round(float(v) + 0.25, 3) for v in frame[first]]
    if reading == "anomaly-svdd" and "label" in frame:
        out = [str(v) for v in frame["label"]]
        out[5] = record["positive"]
        return out
    return None


def schema_of(record, sample, truth_column):
    return {"classes": list(record["classes"] or []), "output": record, "sample": sample,
            "truth": record.get("truth"), "label": {"column": truth_column, "decode": {}},
            "input": [{"name": f["name"], "dtype": "string" if f["role"] == "nominal" else "float64"} for f in FEATURES]}


def roles_of(case_id, record, lane):
    vk = lane["view_key"]
    if vk in ("value", "reconstruction", "horizon"):
        return None
    if vk == "anomaly":
        normal = record["classes"][0] if record["classes"] else "normal"
        anomalous = record["positive"] if record["classes"] else "anomalous"
        return {"schema": "vega/class-roles@1", "roles": {normal: "neutral", anomalous: "alert"},
                "source": {normal: "task:anomaly", anomalous: "task:anomaly"},
                "proposed": {"neutral": normal, "rule": "task:anomaly", "alert": [anomalous], "evidence": ["task:anomaly"]}}
    classes = record["classes"]
    first = classes[0]
    roles = {c: "plain" for c in classes}
    roles[first] = "neutral"
    source = {first: "structural:majority"}
    if vk in ("classes", "binary"):
        roles[classes[-1]] = "alert"
        source[classes[-1]] = "user:turn-2"
    return {"schema": "vega/class-roles@1", "roles": roles, "source": source,
            "proposed": {"neutral": first, "rule": "structural:majority", "alert": [], "evidence": []}}


BREAKDOWN_KIND = {"classes": "per_class", "binary": "per_class", "labels": "per_label", "ordinal": "per_class",
                  "value": "per_bin", "reconstruction": "per_feature", "anomaly": "at_k", "horizon": "per_horizon"}


def breakdown_of(record, lane):
    vk = lane["view_key"]
    names = {"per_class": record["classes"], "per_label": record["classes"], "per_bin": ["q1", "q2", "q3", "q4", "q5"],
             "per_feature": [f["name"] for f in FEATURES], "at_k": ["k10", "k50"],
             "per_horizon": [f"h{h}" for h in (record["horizons"] or [])]}[BREAKDOWN_KIND[vk]]
    values = [round(0.9 - 0.2 * k, 3) for k in range(len(names))]
    section = {"kind": BREAKDOWN_KIND[vk], "metric": "f1", "names": "breakdown_names", "values": "breakdown_values",
               "weak": "breakdown_weak", "direction": "max", "caption": "The score per part on the sealed test part."}
    return section, {"breakdown_names": names, "breakdown_values": values, "breakdown_weak": [v < 0.6 for v in values]}


def explanation_of(record, frame, predictions, fr: Frame):
    """Row 0 as vega/explanation@2 carries it, the members a task module reads."""
    out = {"schema": "vega/explanation@2", "prediction": predictions[0]}
    if hasattr(fr, "inputs"):
        names = [record["field_kinds"][f]["of"] for f in record["fields"] if record["field_kinds"][f]["kind"] == "recon"]
        xhat = [float(frame[f][0]) for f in record["fields"] if record["field_kinds"][f]["kind"] == "recon"]
        out["inputs"] = {"order": [str(n) for n in names], "shaped": [round(float(v), 4) for v in fr.inputs[0]], "raw": None}
        out["outputs"] = {"kind": "reconstruction", "names": [str(n) for n in names], "link": "reconstruction",
                          "raw": xhat, "read": xhat, "answer": {}}
    return out


def build() -> dict:
    Codec = _codec()
    cases = []
    for k, (case_id, spec, threshold, truth_column, lane) in enumerate(CASES):
        record = orec.output_record(spec, threshold, truth_column)
        fr = Frame(record, np.random.default_rng(1000 + k))
        frame = fr.build()
        codec = Codec(record)
        plain, included = codec.rows(frame), codec.rows(frame, include=INCLUDES)
        sample = spec["data_contract"]["sample"]
        section, values = breakdown_of(record, lane)
        reference = ({"kind": "reconstruction_error", "quantiles": {"0.5": 0.05, "0.9": 0.2, "0.99": 0.6}, "n": 40,
                      "source": "the bundle on the demo sample (training part)"} if lane["view_key"] == "reconstruction" else None)
        cases.append({
            "id": case_id, "reading": record["reading"], "lane": lane,
            "schema": schema_of(record, sample, truth_column),
            "keys": list(codec.keys()), "keys_included": list(codec.keys(INCLUDES)),
            "predictions": plain, "predictions_included": included, "truth": truth_of(case_id, record, frame),
            "class_roles": roles_of(case_id, record, lane),
            "breakdown": section, "values": values, "reference": reference,
            "explanation": explanation_of(record, frame, included, fr),
        })
    return {"schema": "vega/page-predictions-fixture@1",
            "about": "RowPrediction answers per reading, written by the serving AnswerCodec from vega-pack's output record "
                     "(tests/fixtures/predictions/make_predictions.py); read by tests/registry.test.js.",
            "rows": N, "includes": list(INCLUDES), "cases": cases}


def _dump(doc) -> str:
    return json.dumps(doc, indent=1, ensure_ascii=False, allow_nan=False) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    ap.add_argument("--check", action="store_true", help="compare with predictions.json instead of writing it")
    a = ap.parse_args(argv)
    try:
        text = _dump(build())
    except SystemExit as exc:          # PackExit: the machinery refused a case
        print(f"ERROR: a case was refused: {exc}", file=sys.stderr)
        return 2
    if a.check:
        same = OUT.is_file() and OUT.read_text(encoding="utf-8") == text
        print(("OK: " if same else "FAIL: ") + f"{OUT.name} {'is' if same else 'is not'} this script's output")
        return 0 if same else 1
    OUT.write_text(text, encoding="utf-8")
    print(f"OK: wrote {OUT} ({len(CASES)} cases)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
