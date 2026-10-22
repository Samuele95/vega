#!/usr/bin/env python3
"""
test_inspect_targets.py — the explained targets and the explain request
(`templates/serving/package/inspection/{targets, target_kinds, query}.py`), proved on *rendered*
packages (build plan I04; inspect-contract §4, §5.4; R-C5, R-F6).

    python3 test_inspect_targets.py            (or: python3 -m pytest test_inspect_targets.py -q)

One package is rendered per case — every effective task kind (binary, multiclass, multilabel,
regression with and without the quantile loss, ordinal, unsupervised, anomaly under the
reconstruction, vae and svdd losses, forecast with and without quantiles), two sequence tasks
read through their base, and both graph levels — with the model templates (`domain.py`,
`network.py`, `readout.py`, `assemble.py`, the spliced builder and blocks) and the serving
package (S03's readers, S04's samples, I02's view, these modules) in the Predictor's module.
`MANIFEST.output` is vega-pack's own producer's (`output_record.py`, S01), and the served frame
the Predictor answers is S03's reader over the same head.

  table      the target kinds, selectors, units, methods and details are catalog.json's; each
             task's kinds and default are the view's (`view.TARGETS` / `BY_LOSS`), which are the
             catalog's `inspection.targets`; the readout names are `network.task.outputs`'.
  D3 §7.2    binary: the sign flips with the class; multiclass and multilabel logits; the
             regression value; quantile names and columns; ordinal `expected_rank == Σσ(z)` and
             the threshold logits; the reconstruction error equals `TaskReadout`'s row score and
             is the mean of its feature terms; `svdd_distance == ‖f(x) − c‖²` with the buffer,
             and an unset centre refuses to open; forecast step h picks column h−1 (the median
             column under a quantile loss); sequence tasks read through their base; both graph
             levels; every scalar differentiable in the head; `why`.
  readout    `explanation.outputs` validates the explanation-2 schema, `read` claims no
             distribution the head does not produce, the answers follow the thresholds.
  served     `TARGET_COLUMNS` resolved through `MANIFEST.output.field_kinds` by `(kind, of)`:
             for every target of every case the served field's value equals the target's value
             (binary times the sign); the lookup survives renamed fields; no field, no value.
  requests   row | rows+history+at | rows+at; a window shorter than L, history on a graph, a
             row sample with two rows, edges, unknown fields → 422 naming the field; selectors
             checked for form and for the task; the window rule agrees with the assembler row
             by row; the explained sample resolves through `SampleIndex`.
  hygiene    the modules pass C11–C13 at `error`, hold no module-level mutable state, import
             only the standard library, numpy, torch, pandas (query) and their package, and carry
             no dataset literal (`lint_templates.py`).

Exit 0 when every test passes.
"""
from __future__ import annotations

import ast
import copy
import importlib
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from jsonschema import Draft202012Validator

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
BACKEND = SCRIPTS / "backends" / "pytorch"
TEMPLATES = BACKEND / "templates"
PACKAGE_TEMPLATES = TEMPLATES / "serving" / "package"
INSPECTION = PACKAGE_TEMPLATES / "inspection"
REPO = SCRIPTS.parents[1]
CATALOG = json.loads((SCRIPTS / "catalog.json").read_text(encoding="utf-8"))
PACK_CATALOG = json.loads((REPO / "vega-pack" / "scripts" / "catalog.json")
                          .read_text(encoding="utf-8"))
EXPLANATION = json.loads((SCRIPTS.parent / "references" / "schemas" /
                          "explanation-2.schema.json").read_text(encoding="utf-8"))
MINE = ("targets.py", "target_kinds.py", "query.py")
for _p in (str(SCRIPTS), str(HERE), str(BACKEND), str(REPO / "vega-pack" / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import conformance  # noqa: E402
import output_record as orec  # noqa: E402  (vega-pack, S01: the producer of vega/output@2)
import render_templates as rt  # noqa: E402

STUB_LOSSES = '''"""A stub of the losses source: the targets read networks, never a loss."""
# --8<-- losses
def build_loss(spec):
    raise NotImplementedError("test_inspect_targets.py renders no loss")
# -->8-- losses
'''

# ================================================================ specs

def _num(name):
    return {"name": name, "dtype": "float32", "role": "numeric"}


def _nominal(name, card):
    return {"name": name, "dtype": "int64", "role": "nominal", "cardinality": card,
            "unseen_category_id": card - 1}


def _b(bid, block_type, **params):
    return {"id": bid, "type": block_type, "params": params}


def _i(block_type, **params):
    return {"type": block_type, "params": params}


NUM = [_num(f"x{i}") for i in range(4)]
MIXED = NUM + [_nominal("zone", 5)]
EMB = [_b("E", "embedding", columns=["zone"], dims={"zone": 3}), _b("C", "concat")]
WINDOW = {"shape": "window", "length": 5, "stride": 1, "context": "previous-part"}
NODE = {"shape": "graph", "edges": "shaped/edges.parquet", "level": "node",
        "protocol": "inductive", "batching": "full"}
LEVEL = {"shape": "graph", "edges": "shaped/edges.parquet", "level": "graph", "graph_id": "g",
         "protocol": "inductive", "batching": "graphs"}
MSE = {"name": "mse", "params": {}}
CE = {"name": "cross_entropy", "params": {}}
BCE = {"name": "bce_with_logits", "params": {}}
CUMULATIVE = {"name": "cumulative_link", "params": {}}
RECON = {"name": "reconstruction", "params": {}}
PROB = {"applies": True, "value": 0.41, "scale": "probability", "source": "valid-optimum"}
PER_LABEL = {"applies": True, "value": 0.5, "scale": "probability", "source": "valid-optimum",
             "per_label": {"red": 0.3, "green": 0.6}}
SCORE = {"applies": True, "value": 0.8, "scale": "score", "source": "valid-optimum"}


def _quantile(qs):
    return {"name": "quantile", "params": {"quantiles": qs}}


def _spec(family, blocks, task, *, sample=None, features=None, loss=None):
    features = NUM if features is None else features
    return {"schema": "vega/netspec@1", "seed": 7, "task": task,
            "input": {"manifest_digest": "sha256:" + "ab" * 32},
            "data_contract": {"input_dim": len(features), "features": features,
                              "sample": sample or {"shape": "row"}},
            "family": {"name": family, "source": "user:test_inspect_targets"}, "blocks": blocks,
            "loss": loss or BCE, "metrics": {"primary": "pr_auc"}, "optimizer": {"name": "adamw"},
            "training": {"max_epochs": 1, "batch_size": 8},
            "init": {"scheme": "kaiming_uniform", "bias": "zeros"}}


def _seq(base, **extra):
    kind = "sequence-regression" if base in ("regression", "ordinal") else \
        "sequence-classification"
    return {"kind": kind, "base_kind": base, "label": "y", **extra}


def _forecast(horizon):
    return {**WINDOW, "horizon": horizon, "autoregressive": True}


ORDINAL = ["lo", "mid", "hi", "top"]
CASES = {
    # name: (spec, threshold.json)
    "binary": (_spec("mlp-plain", EMB + [
        _b("D", "dense", units=8, activation="relu"), _b("H", "head", out=1)],
        {"kind": "binary", "label": "y", "classes": ["no", "yes"], "positive": "yes"},
        features=MIXED), PROB),
    "multiclass": (_spec("mlp-plain", EMB + [
        _b("D", "dense", units=8, activation="relu"), _b("H", "head", out=3)],
        {"kind": "multiclass", "label": "y", "classes": ["a", "b", "c"]}, features=MIXED,
        loss=CE), None),
    "multilabel": (_spec("linear", EMB + [_b("H", "head", out=3)],
                         {"kind": "multilabel", "label": "y", "classes": ["red", "green", "blue"],
                          "label_format": {"kind": "delimited", "sep": "|"}}, features=MIXED),
                   PER_LABEL),
    "regression": (_spec("mlp-plain", [
        _b("C", "concat"), _b("D", "dense", units=8, activation="relu"), _b("H", "head", out=1)],
        {"kind": "regression", "label": "y"}, loss=MSE), None),
    "quantile": (_spec("mlp-plain", [
        _b("C", "concat"), _b("D", "dense", units=8, activation="relu"), _b("H", "head", out=3)],
        {"kind": "regression", "label": "y"}, loss=_quantile([0.1, 0.5, 0.9])), None),
    "quantile-off": (_spec("linear", [_b("C", "concat"), _b("H", "head", out=2)],
                           {"kind": "regression", "label": "y"},
                           loss=_quantile([0.25, 0.75])), None),
    "ordinal": (_spec("linear", [_b("C", "concat"), _b("H", "head", out=3)],
                      {"kind": "ordinal", "label": "y", "classes": ORDINAL}, loss=CUMULATIVE),
                None),
    "unsupervised": (_spec("autoencoder", EMB + [
        _b("Z", "dense", units=3), _b("D", "decoder", widths=[6], activation="gelu"),
        _b("H", "head", out=9, activation="none")], {"kind": "unsupervised", "label": None},
        features=MIXED, loss=RECON), None),
    "anomaly": (_spec("vae", [
        _b("C", "concat"), _b("S", "sequential", inner=[_i("dense", units=8, activation="relu")]),
        _b("M", "dense", units=6), _b("Z", "reparameterize", z=3),
        _b("D", "decoder", widths=[8]), _b("H", "head", out=9)],
        {"kind": "anomaly", "label": None}, features=MIXED, loss={"name": "vae", "params": {}}),
        SCORE),
    "anomaly-labelled": (_spec("autoencoder", [
        _b("C", "concat"), _b("Z", "dense", units=3), _b("D", "decoder", widths=[6]),
        _b("H", "head", out=4)],
        {"kind": "anomaly", "label": "y", "classes": [0, 1], "positive": 1}, loss=RECON), SCORE),
    "svdd": (_spec("deep-svdd", [
        _b("C", "concat"), _b("S", "sequential", inner=[_i("dense", units=8, activation="relu")]),
        _b("H", "head", out=4, bias=False)], {"kind": "anomaly", "label": None},
        loss={"name": "svdd", "params": {}}), {"applies": True, "value": 1.5, "scale": "score"}),
    "forecast": (_spec("gru", [
        _b("R", "gru", hidden=8, **{"return": "last"}), _b("H", "head", out=3)],
        {"kind": "forecast", "label": "y"}, sample=_forecast(3), loss=MSE), None),
    "forecast-quantile": (_spec("lstm-attention", [
        _b("R", "lstm", hidden=8, **{"return": "all"}), _b("A", "attention_pool", hidden=4),
        _b("H", "head", out=6)], {"kind": "forecast", "label": "y"}, sample=_forecast(2),
        loss=_quantile([0.1, 0.5, 0.9])), None),
    "sequence-binary": (_spec("rnn", [
        _b("R", "rnn", hidden=8, **{"return": "last"}), _b("H", "head", out=1)],
        _seq("binary", classes=[0, 1], positive=1), sample=WINDOW), PROB),
    "sequence-ordinal": (_spec("bi-lstm", [
        _b("R", "lstm", hidden=6, bidirectional=True, **{"return": "last"}),
        _b("H", "head", out=3)], _seq("ordinal", classes=ORDINAL), sample=WINDOW,
        loss=CUMULATIVE), None),
    "node": (_spec("gcn", EMB + [
        _b("G1", "gcn_conv", out=8), _b("A1", "activation", name="relu"),
        _b("H", "head", out=4)],
        {"kind": "multiclass", "label": "y", "classes": ["p", "q", "r", "s"]}, sample=NODE,
        features=MIXED, loss=CE), None),
    "graph-level": (_spec("gin", [
        _b("G1", "gin_conv", out=8), _b("A1", "activation", name="relu"),
        _b("P", "graph_pool", readout="sum"), _b("H", "head", out=1)],
        {"kind": "regression", "label": "y"}, sample=LEVEL, loss=MSE), None),
}

# ================================================================ batches

EDGES = [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2), (4, 5), (5, 4), (5, 6), (6, 5),
         (7, 8), (8, 7), (8, 9), (9, 8), (3, 0)]


def batch_for(spec, seed=3):
    """An example batch of the spec's sample shape (the rows the Predictor would assemble)."""
    sample = spec["data_contract"]["sample"]
    feats = spec["data_contract"]["features"]
    numeric = sum(1 for f in feats if f["role"] != "nominal") + \
        (1 if sample.get("autoregressive") else 0)
    cards = [f["cardinality"] for f in feats if f["role"] == "nominal"]
    g = torch.Generator().manual_seed(seed)
    lead = (10,) if sample["shape"] == "graph" else \
        (6, sample["length"]) if sample["shape"] == "window" else (6,)
    batch = {"x_num": torch.randn(*lead, numeric, generator=g) * 1.5}
    if cards:
        batch["x_cat"] = torch.stack([torch.randint(0, c, lead, generator=g) for c in cards], -1)
    if sample["shape"] == "graph":
        batch["edge_index"] = torch.tensor(EDGES, dtype=torch.long).t().contiguous()
        if sample["level"] == "node":
            batch["edge_weight"] = torch.rand(len(EDGES), generator=g) + 0.5
            batch["target_index"] = torch.tensor([0, 2, 5, 9])
        else:
            batch["graph_index"] = torch.tensor([0, 0, 0, 0, 1, 1, 1, 2, 2, 2])
            batch["num_graphs"] = 3
    return batch


# ================================================================ the rendered packages

class Case:
    """One rendered package: its modules, its network (eval mode), a batch and its head, the
    MANIFEST.output pack writes for it, the targets and the Predictor's served frame."""

    def __init__(self, name, root, package, spec, threshold):
        self.name, self.root, self.package, self.spec = name, Path(root), package, spec
        mod = lambda m: importlib.import_module(f"{package}.{m}")  # noqa: E731
        self.domain, self.network = mod("domain"), mod("model.network")
        self.targets_mod, self.kinds_mod = mod("serving.inspection.targets"), \
            mod("serving.inspection.target_kinds")
        self.query_mod, self.view = mod("serving.inspection.query"), \
            mod("serving.inspection.view")
        self.records, self.values = mod("serving.records"), mod("serving.values")
        self.assemble = mod("model.assemble")
        self.model = self.network.Network.build(self.domain.NetSpec.from_dict(spec)).eval()
        if self.model.readout.objective.value == "svdd":
            with torch.no_grad():
                self.model.svdd_center.copy_(torch.linspace(-0.5, 0.5, 4))
                self.model.svdd_center_set.fill_(True)
        self.batch = batch_for(spec)
        with torch.no_grad():
            self.head = self.model(self.batch)
        truth = (spec.get("task") or {}).get("label")
        self.output = orec.output_record(spec, threshold, truth)
        self.manifest = {"output": self.output, "sample": orec.sample_record(spec, {}, None)}
        self.targets = self.targets_mod.TaskTargets.of(spec, self.output, self.model)

    @property
    def n(self) -> int:
        return int(self.head.shape[0])

    def frame(self, output=None):
        """The frame the Predictor serves for this head (S03's reader of the reading)."""
        manifest = dict(self.manifest, output=output or self.output)
        records = self.records.ServingRecords.of(self.spec, manifest, {"input": []},
                                                  "sha256:model")
        reader = self.values.reader_for(records)
        return reader.frame(reader.head(self.head, self.batch, self.model))

    def fresh_head(self):
        """The head of a forward that records the graph: for gradient checks."""
        return self.model(self.batch)


def _render(name, work, spec, threshold):
    package = "i04_" + name.replace("-", "_")
    src = work / "sources"
    if not src.is_dir():
        src.mkdir()
        shutil.copyfile(BACKEND / "blocks.py", src / "blocks.py")
        shutil.copyfile(BACKEND / "builder.py", src / "builder.py")
        (src / "losses.py").write_text(STUB_LOSSES, encoding="utf-8")
    pipeline = rt.reference_pipeline()
    pipeline["system"]["package"] = package
    out = work / name
    rt.render(TEMPLATES, out, pipeline, spec,
              only=["domain.py", "network.py", "readout.py", "assemble.py"], splice=True,
              sources=str(src))
    rec = rt.render(PACKAGE_TEMPLATES, out, pipeline, spec, into="Predictor", recursive=True)
    assert {f"serving/inspection/{f}" for f in MINE} <= set(rec["files"]), rec["files"]
    emit = rt.load_emit()           # graphs.py with the emitter's mapping, in the serving module
    mapping = emit.common_mapping(spec, pipeline, emit.plan(spec, pipeline))
    text = (PACKAGE_TEMPLATES / "graphs.py").read_text(encoding="utf-8")
    (Path(rec["root"]) / "serving" / "graphs.py").write_text(emit.render_text(text, mapping),
                                                             encoding="utf-8")
    sys.path.insert(0, str(out))
    return Case(name, out, package, spec, threshold)


_WORK = tempfile.TemporaryDirectory(prefix="inspect-targets-")
_CASES = {}


def case(name) -> Case:
    if name not in _CASES:
        spec, threshold = CASES[name]
        _CASES[name] = _render(name, Path(_WORK.name), copy.deepcopy(spec), threshold)
    return _CASES[name]


def _valid(schema, instance):
    wrapped = {"$defs": EXPLANATION["$defs"], **schema}
    errors = sorted(Draft202012Validator(wrapped).iter_errors(instance),
                    key=lambda e: list(e.path))
    assert not errors, "; ".join(f"{list(e.path)}: {e.message[:200]}" for e in errors[:5])


def _close(a, b, tol=1e-6):
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(a)), abs(float(b)))


def _effective(spec):
    task = spec["task"]
    return task.get("base_kind") or task["kind"]


def _row_of(c):
    kind, loss = _effective(c.spec), c.spec["loss"]["name"]
    return c.view.BY_LOSS.get((kind, loss)) or c.view.TARGETS[kind]


def _query_error(c):
    return c.query_mod.QueryError


def _all_targets(c):
    """Every target of the case: the default, and each index every selectable kind admits."""
    tt, f = c.targets, c.targets._facts
    out = [tt.choose(c.head, c.batch, None, 0)]
    for kind in tt.kinds():
        out.append(tt.choose(c.head, c.batch, {"kind": kind}, 0))
        selector = c.kinds_mod.KINDS[kind][0]
        values = {"class": f.classes, "label": f.classes, "feature": f.features,
                  "threshold": range(len(f.classes) - 1), "quantile": range(len(f.quantiles)),
                  "step": range(1, f.horizon + 1)}.get(selector, ())
        fits = [k for k in tt.kinds() if c.kinds_mod.KINDS[k][0] == selector]
        if selector and fits and fits[0] == kind:
            out += [tt.choose(c.head, c.batch, {selector: v}, 0) for v in values]
    return out


# ================================================================ the table

def test_kinds_selectors_and_units_are_the_catalogs():
    c = case("binary")
    inspection = CATALOG["inspection"]
    kinds = c.kinds_mod.KINDS
    assert list(kinds) == list(inspection["target_kinds"]) == list(c.targets_mod.TARGET_KINDS)
    assert {k: (v["selector"], v["unit"]) for k, v in inspection["target_kinds"].items()} == \
        dict(kinds)
    assert list(c.query_mod.SELECTORS) == inspection["selectors"]
    assert list(c.query_mod.DETAILS) == inspection["detail"]
    assert list(c.query_mod.METHODS) == inspection["methods"]
    columns = c.targets_mod.TARGET_COLUMNS
    assert set(columns) <= set(kinds)
    assert {k for v in columns.values() for k in v} <= set(PACK_CATALOG["field_kinds"])
    # the aliases name only target kinds, never shadow one
    for alias, named in c.targets_mod.ALIASES.items():
        assert alias not in kinds and set(named) <= set(kinds), alias


@pytest.mark.parametrize("name", list(CASES))
def test_each_task_explains_its_rows_targets(name):
    c = case(name)
    link, _, default, selectable = _row_of(c)
    assert c.targets.kinds() == selectable and c.targets.default_kind() == default
    assert c.targets.link == link == c.output["link"]
    row = CATALOG["inspection"]["targets"][_effective(c.spec)]
    row = (row.get("by_loss") or {}).get(c.spec["loss"]["name"], row)
    assert (row["default"], tuple(row["selectable"])) == (default, selectable)
    section = c.view.task_section(c.spec, c.model, c.output)
    assert c.targets.outputs == section["outputs"]


# ================================================================ D3 §7.2, kind by kind

@pytest.mark.parametrize("name", ["binary", "sequence-binary"])
def test_binary_sign_flips_with_the_class(name):
    c = case(name)
    tt, z = c.targets, c.head[:, 0]
    classes = c.output["classes"]
    positive, negative = c.output["positive"], classes[0]
    pos = tt.choose(c.head, c.batch, {"class": positive}, 0)
    neg = tt.choose(c.head, c.batch, {"class": negative}, 0)
    assert (pos.kind, pos.sign, pos.name, pos.index, pos.why) == \
        ("binary_logit", 1, positive, 1, "requested")
    assert (neg.sign, neg.name, neg.index) == (-1, negative, 0)
    assert torch.equal(tt.scalar(pos, c.head, c.batch), z)
    assert torch.equal(tt.scalar(neg, c.head, c.batch), -z)
    cut = c.output["threshold"]
    for k in range(c.n):
        t = tt.choose(c.head, c.batch, None, k)
        wanted = positive if float(torch.sigmoid(z[k])) >= cut else negative
        assert (t.name, t.why, t.sign) == (wanted, "predicted", 1 if wanted == positive else -1)
        assert tt.value(t, c.head, c.batch, k) == pytest.approx(t.sign * float(z[k]), abs=1e-6)
        answer = tt.readout(c.head, c.batch, k)["answer"]
        assert answer["class"] == wanted and answer["threshold"] == pytest.approx(cut)
    # at a threshold inside the batch's probabilities both decisions occur, each at its side
    mid = float(torch.sigmoid(z).median())
    other = c.targets_mod.TaskTargets.of(c.spec, dict(c.output, threshold=mid), c.model)
    signs = [other.choose(c.head, c.batch, None, k).sign for k in range(c.n)]
    assert signs == [1 if float(torch.sigmoid(z[k])) >= mid else -1 for k in range(c.n)]
    assert set(signs) == {1, -1}


@pytest.mark.parametrize("name", ["multiclass", "node"])
def test_multiclass_logit_is_the_class_column(name):
    c = case(name)
    tt, classes = c.targets, c.output["classes"]
    for k in range(c.n):
        t = tt.choose(c.head, c.batch, None, k)
        assert (t.kind, t.index, t.why) == ("logit", int(c.head[k].argmax()), "predicted")
        assert t.name == classes[t.index]
    for j, cls in enumerate(classes):
        t = tt.choose(c.head, c.batch, {"class": cls}, 1)
        assert (t.index, t.name, t.sign) == (j, cls, 1)
        assert torch.equal(tt.scalar(t, c.head, c.batch), c.head[:, j])


def test_multilabel_label_logit_and_per_label_thresholds():
    c = case("multilabel")
    tt, labels = c.targets, c.output["classes"]
    t = tt.choose(c.head, c.batch, None, 2)
    assert (t.kind, t.index, t.why) == ("label_logit", int(c.head[2].argmax()), "predicted")
    for j, label in enumerate(labels):
        t = tt.choose(c.head, c.batch, {"label": label}, 0)
        assert torch.equal(tt.scalar(t, c.head, c.batch), c.head[:, j])
    cuts = {"red": 0.3, "green": 0.6, "blue": 0.5}
    for k in range(c.n):
        out = tt.readout(c.head, c.batch, k)
        p = torch.sigmoid(c.head[k])
        assert out["answer"]["labels"] == [lab for j, lab in enumerate(labels)
                                           if float(p[j]) >= cuts[lab]]
        assert out["read"] == pytest.approx(p.tolist(), abs=1e-7)


def test_regression_value_is_the_head():
    c = case("regression")
    t = c.targets.choose(c.head, c.batch, None, 0)
    assert (t.kind, t.index, t.name, t.why, t.unit) == ("value", 0, "value", "default", "target")
    assert torch.equal(c.targets.scalar(t, c.head, c.batch), c.head[:, 0])
    assert c.targets.readout(c.head, c.batch, 3)["answer"] == \
        {"value": pytest.approx(float(c.head[3, 0]), abs=1e-6)}


@pytest.mark.parametrize("name,levels", [("quantile", [0.1, 0.5, 0.9]),
                                         ("quantile-off", [0.25, 0.75])])
def test_quantile_names_and_columns(name, levels):
    c = case(name)
    tt = c.targets
    names = [f"q{q:g}" for q in levels]
    assert tt.outputs == {"kind": "quantiles", "names": names}
    median = min(range(len(levels)), key=lambda i: abs(levels[i] - 0.5))
    t = tt.choose(c.head, c.batch, None, 0)
    assert (t.kind, t.index, t.name, t.why) == ("quantile", median, names[median], "default")
    for q, level in enumerate(levels):
        by_index = tt.choose(c.head, c.batch, {"quantile": q}, 0)
        by_level = tt.choose(c.head, c.batch, {"quantile": level}, 0)
        assert (by_index.index, by_index.name) == (by_level.index, by_level.name) == (q, names[q])
        assert torch.equal(tt.scalar(by_index, c.head, c.batch), c.head[:, q])
    value = tt.choose(c.head, c.batch, {"kind": "value"}, 0)
    assert (value.index, value.name, value.why) == (median, "value", "requested")
    assert torch.equal(tt.scalar(value, c.head, c.batch), c.head[:, median])
    with pytest.raises(_query_error(c)) as err:
        tt.choose(c.head, c.batch, {"quantile": len(levels)}, 0)
    assert err.value.field == "target.quantile"
    with pytest.raises(_query_error(c)):
        tt.validate({"quantile": 0.3})


@pytest.mark.parametrize("name", ["ordinal", "sequence-ordinal"])
def test_ordinal_expected_rank_is_the_sum_of_sigmoids(name):
    c = case(name)
    tt = c.targets
    t = tt.choose(c.head, c.batch, None, 0)
    assert (t.kind, t.index, t.name, t.why, t.unit) == \
        ("expected_rank", None, None, "default", "rank")
    s = tt.scalar(t, c.head, c.batch)
    assert torch.allclose(s, torch.sigmoid(c.head).sum(dim=-1), atol=0, rtol=0)
    for j in range(len(ORDINAL) - 1):
        th = tt.choose(c.head, c.batch, {"threshold": j}, 0)
        assert (th.kind, th.index, th.name) == ("threshold_logit", j, f">{ORDINAL[j]}")
        assert torch.equal(tt.scalar(th, c.head, c.batch), c.head[:, j])
    for k in range(c.n):
        p = torch.sigmoid(c.head[k])
        passed = int((p >= 0.5).sum())
        picked = tt.choose(c.head, c.batch, {"kind": "threshold_logit"}, k)
        assert (picked.index, picked.why) == (min(passed, len(ORDINAL) - 2), "predicted")
        out = tt.readout(c.head, c.batch, k)
        assert out["answer"] == {"rank": passed, "class": ORDINAL[passed],
                                 "expected_rank": pytest.approx(float(p.sum()), abs=1e-6)}
        assert out["read"] == pytest.approx(p.tolist(), abs=1e-7)


@pytest.mark.parametrize("name", ["unsupervised", "anomaly", "anomaly-labelled"])
def test_reconstruction_error_is_the_readouts_row_score(name):
    c = case(name)
    tt = c.targets
    t = tt.choose(c.head, c.batch, None, 0)
    assert (t.kind, t.why, t.unit) == ("reconstruction_error", "default", "error")
    row = c.model.readout.row_score(c.head, c.batch, c.model)
    assert torch.equal(tt.scalar(t, c.head, c.batch), row)
    f = tt._facts
    terms = c.kinds_mod.feature_errors(f, c.head, c.batch)
    n = f.n_numeric
    mean = terms[:, :n].mean(dim=-1) + (terms[:, n:].mean(dim=-1) if f.cards else 0)
    assert torch.allclose(mean, row, atol=1e-6)
    for i, feature in enumerate(f.features):
        ft = tt.choose(c.head, c.batch, {"feature": feature}, 0)
        assert (ft.kind, ft.index, ft.name) == ("feature_error", i, feature)
        assert torch.equal(tt.scalar(ft, c.head, c.batch), terms[:, i])
    x = c.batch["x_num"][:, :n]
    assert torch.allclose(terms[:, :n], (c.head[:, :n] - x) ** 2)
    k = 2
    picked = tt.choose(c.head, c.batch, {"kind": "feature_error"}, k)
    assert (picked.index, picked.why) == (int(terms[k].argmax()), "predicted")
    answer = tt.readout(c.head, c.batch, k)["answer"]
    assert answer["error"] == pytest.approx(float(row[k]), abs=1e-6)
    assert answer["per_feature"] == pytest.approx(terms[k].tolist(), abs=1e-6)
    if c.spec["task"]["kind"] == "anomaly":
        assert answer["anomalous"] == (float(row[k]) >= 0.8) and answer["threshold"] == 0.8


def test_svdd_distance_reads_the_centre_buffer():
    c = case("svdd")
    tt = c.targets
    t = tt.choose(c.head, c.batch, None, 0)
    assert (t.kind, t.why, t.unit) == ("svdd_distance", "default", "distance")
    centre = c.model.svdd_center
    assert float(centre.abs().sum()) > 0
    wanted = ((c.head - centre) ** 2).sum(dim=-1)
    assert torch.allclose(tt.scalar(t, c.head, c.batch), wanted, atol=0, rtol=1e-6)
    out = tt.readout(c.head, c.batch, 1)
    assert out["read"] is None and out["kind"] == "embedding"
    assert out["answer"]["distance"] == pytest.approx(float(wanted[1]), rel=1e-6)
    assert out["answer"]["anomalous"] == (float(wanted[1]) >= 1.5)


def test_an_unset_svdd_centre_refuses_to_open():
    c = case("svdd")
    fresh = c.network.Network.build(c.domain.NetSpec.from_dict(c.spec)).eval()
    with pytest.raises(c.targets_mod.TargetError) as err:
        c.targets_mod.TaskTargets.of(c.spec, c.output, fresh)
    assert str(err.value) == c.targets_mod.SVDD_WHY == \
        "the one-class centre was not persisted with the weights"
    assert isinstance(err.value, c.view.GraphError)      # the inspector does not open


@pytest.mark.parametrize("name,horizon,quantiles", [("forecast", 3, None),
                                                    ("forecast-quantile", 2, [0.1, 0.5, 0.9])])
def test_forecast_step_h_picks_column_h_minus_1(name, horizon, quantiles):
    c = case(name)
    tt = c.targets
    t = tt.choose(c.head, c.batch, None, 0)
    assert (t.kind, t.why, t.unit) == ("forecast_step", "default", "target")
    steps = c.head if not quantiles else c.head.reshape(c.n, horizon, len(quantiles))[..., 1]
    for h in range(1, horizon + 1):
        st = tt.choose(c.head, c.batch, {"step": h}, 0)
        column = h - 1 if not quantiles else (h - 1) * len(quantiles) + 1
        name_h = f"t+{h}" if not quantiles else f"t+{h}/q0.5"
        assert (st.index, st.name) == (column, name_h)
        assert st.name == tt.outputs["names"][st.index]
        assert torch.equal(tt.scalar(st, c.head, c.batch), steps[:, h - 1])
        assert torch.equal(tt.scalar(st, c.head, c.batch), c.head[:, column])
    assert (t.index, t.name) == (0 if not quantiles else 1, tt.outputs["names"][t.index])
    answer = tt.readout(c.head, c.batch, 0)["answer"]
    assert answer["steps"] == pytest.approx(steps[0].tolist(), abs=1e-6)
    for bad in (0, horizon + 1):
        with pytest.raises(_query_error(c)) as err:
            tt.choose(c.head, c.batch, {"step": bad}, 0)
        assert err.value.field == "target.step"


def test_graph_level_value_per_graph():
    c = case("graph-level")
    assert c.n == 3
    t = c.targets.choose(c.head, c.batch, None, 2)
    assert torch.equal(c.targets.scalar(t, c.head, c.batch), c.head[:, 0])


@pytest.mark.parametrize("name", list(CASES))
def test_every_scalar_is_differentiable_in_the_head(name):
    c = case(name)
    head = c.fresh_head()
    for t in _all_targets(c):
        s = c.targets.scalar(t, head, c.batch)
        assert s.shape == (c.n,), (t, s.shape)
        grad, = torch.autograd.grad(s[1], head, retain_graph=True)
        assert float(grad.abs().sum()) > 0, t
        assert float(grad[0].abs().sum()) == 0, "sample 1's scalar reads sample 1 only"
    assert all(p.grad is None for p in c.model.parameters())


def test_why_follows_the_request():
    c = case("multiclass")
    tt = c.targets
    assert tt.choose(c.head, c.batch, None, 0).why == "predicted"
    assert tt.choose(c.head, c.batch, {"kind": "logit"}, 0).why == "predicted"
    assert tt.choose(c.head, c.batch, {"class": "b"}, 0).why == "requested"
    r = case("ordinal")
    assert r.targets.choose(r.head, r.batch, None, 0).why == "default"
    assert r.targets.choose(r.head, r.batch, {"kind": "expected_rank"}, 0).why == "requested"
    assert r.targets.choose(r.head, r.batch, {"kind": "expected_ordinal"}, 0).why == "requested"


# ================================================================ refusals and aliases

@pytest.mark.parametrize("name,request_,field,words", [
    ("regression", {"class": "a"}, "target.class",
     "target.class does not apply to a regression task"),
    ("regression", {"quantile": 1}, "target.quantile",
     "target.quantile does not apply to a regression task trained with the mse loss"),
    ("multiclass", {"class": "z"}, "target.class", "'z' is not one of the task's outputs"),
    ("multiclass", {"step": 1}, "target.step", "does not apply to a multiclass task"),
    ("multilabel", {"label": "purple"}, "target.label", "not one of the task's outputs"),
    ("ordinal", {"threshold": 3}, "target.threshold", "outside the 3 thresholds"),
    ("unsupervised", {"feature": "nope"}, "target.feature", "not one of the task's outputs"),
    ("svdd", {"feature": "x0"}, "target.feature",
     "does not apply to an anomaly task trained with the svdd loss"),
    ("svdd", {"kind": "reconstruction_error"}, "target.kind", "does not apply"),
    ("binary", {"kind": "no_such_kind"}, "target.kind", "is not a target kind"),
    ("binary", {"class": "no", "kind": "binary_logit"}, "target", "at most one selector"),
    ("binary", {"colour": "x"}, "target.colour", "is not a target selector"),
])
def test_a_selector_the_task_does_not_admit_is_a_422(name, request_, field, words):
    c = case(name)
    for call in (lambda: c.targets.validate(request_),
                 lambda: c.targets.choose(c.head, c.batch, request_, 0)):
        with pytest.raises(_query_error(c)) as err:
            call()
        assert err.value.field == field and err.value.status == 422
        assert words in str(err.value), str(err.value)


@pytest.mark.parametrize("name,alias,kind", [
    ("binary", "class", "binary_logit"), ("binary", "true_class", "binary_logit"),
    ("binary", "positive_logit", "binary_logit"), ("multiclass", "class", "logit"),
    ("multilabel", "label", "label_logit"), ("ordinal", "expected_ordinal", "expected_rank"),
    ("unsupervised", "error", "reconstruction_error"),
    ("anomaly", "row_score", "reconstruction_error"), ("svdd", "row_score", "svdd_distance"),
    ("svdd", "distance", "svdd_distance"), ("forecast", "horizon_step", "forecast_step"),
    ("forecast", "horizon", "forecast_step"), ("quantile", "value", "value"),
])
def test_the_request_vocabulary_maps_onto_the_target_kinds(name, alias, kind):
    c = case(name)
    assert c.targets.choose(c.head, c.batch, {"kind": alias}, 0).kind == kind


# ================================================================ the readout

@pytest.mark.parametrize("name", list(CASES))
def test_the_readout_validates_and_claims_no_distribution(name):
    c = case(name)
    outputs = EXPLANATION["properties"]["outputs"]
    for k in range(c.n):
        out = c.targets.readout(c.head, c.batch, k)
        _valid(outputs, out)
        assert (out["kind"], out["names"], out["link"]) == \
            (c.targets.outputs["kind"], c.targets.outputs["names"], c.output["link"])
        assert out["raw"] == pytest.approx(c.head[k].tolist(), abs=1e-6)
        if out["kind"] == "classes":
            assert sum(out["read"]) == pytest.approx(1.0, abs=1e-6)
            assert out["answer"]["class"] == out["names"][int(np.argmax(out["read"]))]
        if out["kind"] in ("labels", "thresholds", "score"):
            assert out["read"] == pytest.approx(torch.sigmoid(c.head[k]).tolist(), abs=1e-7)
        if out["kind"] == "reconstruction":
            assert len(out["read"]) == len(out["names"]) == len(out["answer"]["per_feature"])
    target = c.targets.choose(c.head, c.batch, None, 0)
    _valid(EXPLANATION["properties"]["target"],
           target.to_dict(c.targets.value(target, c.head, c.batch, 0)))


# ================================================================ served fields (R-C5)

_NO_FIELD = {"expected_rank", "feature_error"}


@pytest.mark.parametrize("name", list(CASES))
def test_served_fields_are_found_through_field_kinds(name):
    c = case(name)
    frame = c.frame()
    kinds = c.output["field_kinds"]
    seen = set()
    for t in _all_targets(c):
        found = c.targets.served_field(t)
        no_value = t.kind == "value" and "value" not in c.output["fields"]
        if t.kind in _NO_FIELD or no_value:
            assert found is None, (t, found)
            assert c.targets.served(t, frame.iloc[0]) is None
            continue
        assert found is not None, t
        field, factor = found
        assert kinds[field]["kind"] in c.targets_mod.TARGET_COLUMNS[t.kind]
        seen.add(field)
        for k in range(c.n):
            served = c.targets.served(t, frame.iloc[k])
            value = c.targets.value(t, c.head, c.batch, k)
            assert _close(served, value), (t, k, field, served, value)
            assert factor == (t.sign if t.kind == "binary_logit" else 1)
    # every raw, value, quantile, horizon, row_score and score field answers some target
    answering = {f for f, v in kinds.items()
                 if v["kind"] in {"raw", "value", "quantile", "horizon", "row_score", "score"}
                 and not (v["kind"] == "horizon" and "/" in str(v["of"])
                          and not str(v["of"]).endswith("/0.5"))}
    assert answering == seen, sorted(answering ^ seen)


@pytest.mark.parametrize("name", ["binary", "multiclass", "quantile", "forecast-quantile",
                                  "ordinal", "anomaly", "svdd", "unsupervised"])
def test_the_lookup_reads_kinds_and_of_never_names(name):
    """Renamed fields (and reordered) still resolve: the lookup never reads a field name."""
    c = case(name)
    renamed = {f: f"f{i:02d}" for i, f in enumerate(reversed(c.output["fields"]))}
    output = copy.deepcopy(c.output)
    output["field_kinds"] = {renamed[f]: v for f, v in reversed(list(output["field_kinds"]
                                                                     .items()))}
    other = c.targets_mod.TaskTargets.of(c.spec, output, c.model)
    frame = c.frame().rename(columns=renamed)
    for t in _all_targets(c):
        before = c.targets.served_field(t)
        after = other.served_field(t)
        assert (before is None and after is None) or after == (renamed[before[0]], before[1])
        if after is not None:
            assert other.served(t, frame.iloc[1]) == c.targets.served(t, c.frame().iloc[1])


def test_no_field_no_served_value():
    c = case("multiclass")
    bare = c.targets_mod.TaskTargets.of(c.spec, None, c.model)
    t = bare.choose(c.head, c.batch, None, 0)
    assert bare.served_field(t) is None and bare.served(t, {"logit_a": 1.0}) is None
    t = c.targets.choose(c.head, c.batch, {"class": "a"}, 0)
    assert c.targets.served(t, {"logit_a": float("nan")}) is None
    assert c.targets.served(t, {"logit_a": None}) is None
    assert c.targets.served(t, {"logit_a": 2.5}) == 2.5


def test_a_manifest_of_another_network_refuses():
    c = case("regression")
    output = dict(c.output, link="softmax")
    with pytest.raises(c.targets_mod.TargetError, match="another model"):
        c.targets_mod.TaskTargets.of(c.spec, output, c.model)


# ================================================================ the request (query.py)

ROW = {"x0": 0.1, "x1": -1.0, "x2": 2.0, "x3": 0.0, "zone": 3}


def _rules(name, **kw):
    c = case(name)
    return c, c.query_mod.QueryRules.of(c.spec, **kw)


def _refused(c, call, field, words=""):
    with pytest.raises(c.query_mod.QueryError) as err:
        call()
    assert err.value.field == field and err.value.status == 422, (err.value.field, err.value)
    assert words in str(err.value), str(err.value)


def test_row_requests():
    c, rules = _rules("binary", max_rows=10)
    q = rules.parse({"row": ROW})
    assert (len(q.rows), q.at, q.history, q.target, q.method, q.steps, q.detail) == \
        (1, 0, None, None, "grad_x_input", 32, "full")
    assert list(q.rows.columns) == list(ROW)
    assert rules.parse({"rows": [ROW], "at": 0}).at == 0
    q = rules.parse({"row": ROW, "history": [ROW, ROW], "target": {"class": "yes"},
                     "method": "integrated_gradients", "steps": 64, "detail": "exact",
                     "rows": None})
    assert (len(q.history), dict(q.target), q.method, q.steps, q.detail) == \
        (2, {"class": "yes"}, "integrated_gradients", 64, "exact")
    assert q.sample(c.domain.SampleIndex.of_rows(1)) == 0
    _refused(c, lambda: rules.parse({"row": ROW, "rows": [ROW]}), "row", "exactly one")
    _refused(c, lambda: rules.parse({}), "row", "exactly one")
    _refused(c, lambda: rules.parse({"rows": [ROW, ROW]}), "rows", "explains one row")
    _refused(c, lambda: rules.parse({"row": ROW, "at": 3}), "at", "its one row")
    _refused(c, lambda: rules.parse({"row": [ROW]}), "row", "object of columns")
    _refused(c, lambda: rules.parse({"rows": 5}), "rows", "list of raw rows")   # a 422, no TypeError
    _refused(c, lambda: rules.parse({"row": ROW, "history": [ROW] * 11}), "history", "holds 11")


def _series_rows(n, series, start=0):
    return [{"s": series, "t": start + i, "x0": float(i), "x1": 0.5, "x2": -0.5, "x3": 1.0}
            for i in range(n)]


def _window_spec(series_by="s", order_by="t", length=5):
    spec = copy.deepcopy(CASES["sequence-binary"][0])
    spec["data_contract"]["sample"] = dict(spec["data_contract"]["sample"], length=length,
                                           series_by=series_by, order_by=order_by)
    return spec


def test_window_requests():
    c = case("sequence-binary")
    rules = c.query_mod.QueryRules.of(_window_spec())
    rows = _series_rows(3, "a", 10)
    history = _series_rows(4, "a", 5) + _series_rows(6, "b", 0)
    q = rules.parse({"rows": rows, "history": history})
    assert (q.at, len(q.rows), len(q.history)) == (2, 3, 10)        # default: the last row
    assert rules.parse({"rows": rows, "history": history, "at": 0}).at == 0
    assert rules.parse({"rows": rows, "history": history, "at": -2}).at == 1
    # 4 rows of history + the anchor: one short of L = 5 at the first request row? 4 ≥ 4
    short = _series_rows(3, "a", 5)
    _refused(c, lambda: rules.parse({"rows": rows, "history": short, "at": 0}), "at",
             "a window needs 5 rows ending at `at`")
    # another series' history does not count
    _refused(c, lambda: rules.parse({"rows": rows, "history": _series_rows(9, "b"), "at": 2}),
             "at", "has 2 earlier row(s) of its series")
    # rows only, one series long enough
    assert rules.parse({"rows": _series_rows(5, "a")}).at == 4
    _refused(c, lambda: rules.parse({"rows": _series_rows(4, "a")}), "at", "send 1 more")
    _refused(c, lambda: rules.parse({"row": ROW}), "row", "takes `rows`")
    _refused(c, lambda: rules.parse({"rows": rows, "at": 3}), "at", "outside the 3 row(s)")
    _refused(c, lambda: rules.parse({"rows": [{"t": 1, "x0": 1.0}]}), "rows", "'s' column")
    _refused(c, lambda: rules.parse({"rows": rows, "history": [{"s": "a", "x0": 1.0}]}),
             "history", "'t' column")
    # order_by is honoured: the anchor's earlier rows are the smaller `t`, not the earlier rows
    shuffled = _series_rows(5, "a")[::-1]                    # t = 4, 3, 2, 1, 0
    assert rules.parse({"rows": shuffled, "at": 0}).at == 0  # t = 4 has 4 earlier rows
    _refused(c, lambda: rules.parse({"rows": shuffled, "at": 4}), "at", "has 0 earlier")


def test_the_window_rule_agrees_with_the_assembler():
    """Row by row: a request the query admits at `at` is one the assembler answers at `at`."""
    c = case("sequence-binary")
    rng = np.random.default_rng(5)
    outcomes = {True: 0, False: 0}
    for length, series_by, order_by in ((5, "s", "t"), (3, "s", None), (4, None, "t"),
                                        (3, None, None)):
        spec = _window_spec(series_by, order_by, length)
        rules = c.query_mod.QueryRules.of(spec)
        assembler = c.assemble.SampleAssembler(c.domain.NetSpec.from_dict(spec))
        for trial in range(6):
            rows = [{"s": str(rng.choice(["a", "b", "c"])), "t": int(rng.integers(0, 50)),
                     "x0": float(v), "x1": 0.0, "x2": 1.0, "x3": -1.0}
                    for v in rng.normal(size=int(rng.integers(2, 12)))]
            history = rows[:int(rng.integers(0, len(rows)))]
            request = rows[len(history):] or rows[-1:]
            got = assembler.assemble(pd.DataFrame(request),
                                     pd.DataFrame(history) if history else None)
            for at in range(len(request)):
                answered = int(got.sample_of_row[at]) >= 0
                body = {"rows": request, "history": history or None, "at": at}
                try:
                    q = rules.parse(body)
                except c.query_mod.QueryError as err:
                    assert not answered and err.field == "at", (length, trial, at, str(err))
                    outcomes[False] += 1
                    continue
                assert answered, (length, series_by, order_by, trial, at)
                k = q.sample(got.index)
                assert got.index.rows[k] == at
                outcomes[True] += 1
    assert min(outcomes.values()) >= 10, outcomes      # both sides of the rule were exercised


def test_graph_requests():
    c, rules = _rules("node")
    rows = [dict(ROW, x0=float(i)) for i in range(4)]
    q = rules.parse({"rows": rows})
    assert (q.at, q.graph, q.history) == (0, None, None)
    assert rules.parse({"rows": rows, "at": 3}).at == 3
    assert rules.parse({"rows": rows, "history": []}).history is None     # empty is none
    _refused(c, lambda: rules.parse({"rows": rows, "history": rows}), "history",
             "a graph sample takes no history: the reference graph is its context")
    _refused(c, lambda: rules.parse({"rows": rows, "at": 4}), "at", "outside the 4 row(s)")
    _refused(c, lambda: rules.parse({"row": ROW}), "row", "takes `rows`")
    _refused(c, lambda: rules.parse({"rows": rows, "edges": [{"src": 0, "dst": 1}]}), "edges",
             "is not a field of an explain request")
    index = c.domain.SampleIndex(c.domain.SampleShape.GRAPH, (0, 1, 2, 3))
    assert rules.parse({"rows": rows, "at": 2}).sample(index) == 2
    # graph level: `at` is a row of the explained graph; no history either
    g, level = _rules("graph-level")
    grows = [{"g": gid, "x0": 1.0, "x1": 0.0, "x2": 0.0, "x3": 0.0}
             for gid in ("u", "u", "v", "w", "v")]
    q = level.parse({"rows": grows, "at": 4})
    assert (q.at, q.graph) == (4, "v")
    _refused(g, lambda: level.parse({"rows": grows, "history": grows}), "history",
             "a query is a whole graph")
    _refused(g, lambda: level.parse({"rows": [{"x0": 1.0}]}), "rows", "'g' column")
    _refused(g, lambda: level.parse({"rows": [dict(grows[0], g=None)]}), "rows", "names its graph")
    shaped = pd.DataFrame(grows)
    got = g.assemble.SampleAssembler(g.domain.NetSpec.from_dict(g.spec)).assemble(shaped)
    assert got.index.graphs == ("u", "v", "w")
    assert q.sample(got.index) == 1
    assert level.parse({"rows": grows}).sample(got.index) == 0
    missing = g.domain.SampleIndex(g.domain.SampleShape.GRAPH, (-1,), graphs=("u",))
    _refused(g, lambda: q.sample(missing), "at", "does not hold")


@pytest.mark.parametrize("body,field,words", [
    ({"row": ROW, "edges": []}, "edges", "is not a field"),
    ({"row": ROW, "method": "shap"}, "method", "grad_x_input, integrated_gradients"),
    ({"row": ROW, "steps": 3}, "steps", "from 4 to 128"),
    ({"row": ROW, "steps": 129}, "steps", "from 4 to 128"),
    ({"row": ROW, "steps": True}, "steps", "from 4 to 128"),
    ({"row": ROW, "detail": "everything"}, "detail", "full, summary, exact"),
    ({"row": ROW, "target": "yes"}, "target", "an object"),
    ({"row": ROW, "target": {"class": "yes", "label": "x"}}, "target", "at most one"),
    ({"row": ROW, "target": {"colour": "x"}}, "target.colour", "not a target selector"),
    ({"row": ROW, "target": {"class": 3}}, "target.class", "a name"),
    ({"row": ROW, "target": {"threshold": -1}}, "target.threshold", "≥ 0"),
    ({"row": ROW, "target": {"threshold": 1.5}}, "target.threshold", "≥ 0"),
    ({"row": ROW, "target": {"step": 0}}, "target.step", "≥ 1"),
    ({"row": ROW, "target": {"quantile": 1.5}}, "target.quantile", "level"),
    ({"row": ROW, "target": {"quantile": True}}, "target.quantile", "level"),
    ({"row": ROW, "at": "last"}, "at", "its one row"),
    ([ROW], "body", "an object"),
])
def test_request_fields_are_checked(body, field, words):
    c, rules = _rules("binary")
    _refused(c, lambda: rules.parse(body), field, words)


def test_a_null_field_is_an_absent_one():
    c, rules = _rules("binary")
    q = rules.parse({"row": ROW, "rows": None, "history": None, "at": None, "target": None,
                     "method": None, "steps": None, "detail": None})
    assert (q.at, q.method, q.steps, q.detail, q.target) == (0, "grad_x_input", 32, "full", None)
    assert rules.parse({"row": ROW, "target": {"class": None, "kind": "logit"}}).target == \
        {"kind": "logit"}


def test_the_integrated_gradient_batch_is_bounded():
    limits = case("node").query_mod.LIMITS
    c, rules = _rules("node", limits=dict(limits, ig_batch_max=100))
    rows = [dict(ROW, x0=float(i)) for i in range(4)]
    assert rules.parse({"rows": rows, "method": "integrated_gradients", "steps": 25}).steps == 25
    _refused(c, lambda: rules.parse({"rows": rows, "method": "integrated_gradients",
                                     "steps": 26}), "steps", "above ig_batch_max (100)")
    q = rules.parse({"rows": rows, "method": "integrated_gradients", "steps": 20})
    q.interpolants(5, rules.limits)
    _refused(c, lambda: q.interpolants(6, rules.limits), "steps", "120 nodes")
    plain = rules.parse({"rows": rows, "steps": 40})
    plain.interpolants(10 ** 6, rules.limits)            # grad × input interpolates nothing
    assert limits["ig_batch_max"] == 65536


def test_the_explained_sample_resolves_through_the_sample_index():
    c = case("sequence-binary")
    q = c.query_mod.ExplainQuery(pd.DataFrame([ROW] * 4), None, 2)
    index = c.domain.SampleIndex(c.domain.SampleShape.WINDOW, (1, 2, 3),
                                 spans=((-3, 1), (-2, 2), (-1, 3)))
    assert q.sample(index) == 1
    unanswered = c.query_mod.ExplainQuery(pd.DataFrame([ROW] * 4), None, 0)
    _refused(c, lambda: unanswered.sample(index), "at", "not answered by the prepared batch")


def test_the_limits_are_the_catalogs():
    c = case("binary")
    assert dict(c.query_mod.LIMITS) == CATALOG["inspection"]["limits"]


# ================================================================ hygiene

STDLIB = {"__future__", "dataclasses", "math", "types", "typing"}


@pytest.mark.parametrize("file", MINE)
def test_size_discipline_state_and_imports(file):
    path = INSPECTION / file
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assert conformance.size_findings(tree, file) == [], "C11–C13 must hold at error"
    allowed = STDLIB | {"numpy", "torch"} | ({"pandas"} if file == "query.py" else set())
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = [t.id for t in targets if isinstance(t, ast.Name)]
            if isinstance(node.value, conformance.MUTABLE_NODES) and names != ["__all__"]:
                assert all(n.lstrip("_").isupper() for n in names), f"C7: {names} in {file}"
        if isinstance(node, ast.Import):
            roots = {a.name.split(".")[0] for a in node.names}
            assert roots <= allowed, f"{file} imports {roots}"
        if isinstance(node, ast.ImportFrom) and not node.level:
            assert node.module.split(".")[0] in allowed, f"{file} imports {node.module}"


def test_the_templates_carry_no_dataset_literal():
    lint = REPO / "vega" / "scripts" / "tests" / "lint_templates.py"
    run = subprocess.run([sys.executable, str(lint), *[str(INSPECTION / f) for f in MINE]],
                         capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, run.stdout + run.stderr


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
