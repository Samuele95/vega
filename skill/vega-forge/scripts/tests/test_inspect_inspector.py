#!/usr/bin/env python3
"""
test_inspect_inspector.py — the inspector facade (`templates/serving/package/inspection/
{inspector, explanation, attribution_table, __init__}.py`), `Predictor.inspector()` and the bundle
CLI's `load_inspector` (`templates/serving/{package,bundle}/predict.py`), proved on *rendered*
packages over the shaper's real outputs (build plan I06; inspect-contract §5.3–§6.3; D3 §9,
§10.3; R-D15).

    python3 test_inspect_inspector.py        (or: python3 -m pytest test_inspect_inspector.py -q)

The cases are `_inspector_fixtures.py`'s: an `mlp-plain` network over rows for each of the nine
row readings, an `lstm` over windows (an autoregressive forecast; a sequence task with a binary
base), a `gcn` over a node-level graph and a `gin` over a graph-level set, each behind its own
Predictor; three bundles (rows, windows, the node-level graph) for the CLI.

  explain     every target kind each task allows, and every selector value (class, label,
              threshold, quantile, feature, step), gives an explanation that validates
              `explanation-2.schema.json`, whose head fidelity is ≤ 1e-6 on the CPU, whose
              target agrees with the served field that answers it within the round-trip
              tolerance; both methods and the three details validate; the explained sample is
              the served batch's (inputs, raw head, warnings, span, graph section); the dense
              chain recomputes W·post + b = pre from the published weights; the binary sign flips
              with the class; refusals (a target the task does not admit, an unanswered row, a
              non-finite answer, the interpolant bound) are the query's 422s; a hand-built query
              is checked as `QueryRules.parse` checks one (a selector's form, `at`, the method,
              the detail); an explanation that does not reproduce the served answer is
              `FidelityError`, never a body.
  open        the self-test statuses: ready with coverage, fidelity over every finite example
              sample (≤ 16) and the recompute; idempotent; the pinned digest (pinned, or "the
              network the service reads is not the one packed"); a svdd network without its
              centre; unfaithful answers; inspection switched off; weights withheld; not opened;
              an example with no finite sample. A failed opening leaves no hook.
  network     `network()` validates `network-view-2.schema.json`, is the canonical bytes, carries
              the manifest's model version and every spec block; tensors decode to the live
              parameters.
  attribute   `vega/attribution-table@1` (the contract's §5.5 as a schema) per shape; its
              entries and target values equal the explanations' (rows, windows' per-step and
              per-feature sums, nodes and their hops, graphs); selectors per row, their
              refusals (a selector of the wrong form is never read as an index); chunks of
              `CHUNK` samples give the one-batch numbers; non-finite rows refused.
  warnings    a warning holding the frame's `; ` (the schema guard's) stays one warning in the
              explanation and the table: the Predictor's `predictions` list, not the joined
              text split (which only a harness without `predictions` falls back to).
  hygiene     no parameter `.grad`, no hook added per request, state and mode unchanged;
              concurrent explanations and predictions byte-identical to sequential ones;
              `inspector()` built once; a package without the inspection modules says so, and a
              partial render is not mistaken for one; one default tolerance;
              C3/C4/C5/C7/C14 on the rendered package, C11–C13 at `error` on the new modules,
              their imports and state; no dataset literal.
  bundle      `load_inspector` opens on each bundle's own `examples/rows.csv`; the CLI keeps no
              v2.2 path and reads a missing decision as missing; the bundle's own
              `tests/test_roundtrip.py`, unchanged, is green beside the new CLI.

Exit 0 when every test passes.
"""
from __future__ import annotations

import ast
import importlib
import json
import math
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest
import torch
from jsonschema import Draft202012Validator

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import _inspector_fixtures as ix  # noqa: E402

import conformance  # noqa: E402  (on sys.path through the fixtures)

pf = ix.pf
INSPECTION = ix.INSPECTION
SERVING = ix.pf.SERVING_TEMPLATES
REPO = ix.SCRIPTS.parents[1]
MINE = ("inspector.py", "explanation.py", "attribution_table.py", "__init__.py")
EXPLANATION = Draft202012Validator(ix.EXPLANATION)
NETWORK = Draft202012Validator(ix.NETWORK)
TABLE = Draft202012Validator(ix.TABLE)
ROWS = tuple(pf.ROW_READINGS)
WINDOWS = ("forecast", "sequence-binary")
GRAPHS = ("graph-keypair", "graph-level")
CASES = tuple(f"row:{r}" for r in ROWS) + tuple(f"window:{w}" for w in WINDOWS) + \
    tuple(f"graph:{g}" for g in GRAPHS)
NAMED = ("row:binary", "window:forecast", "graph:graph-keypair")    # mlp-plain, lstm, gcn


# ================================================================ the kit

def _records(frame) -> list:
    """Raw rows as a JSON request carries them."""
    return json.loads(frame.to_json(orient="records"))


class Kit:
    """One case: its Predictor, the inspector opened on its example, the request rules and a
    request's rows (JSON records, as the explain route receives them)."""

    def __init__(self, name):
        shape, which = name.split(":")
        self.name, self.shape = name, shape
        self.case = getattr(ix, f"{shape}_case")(which)
        self.predictor, self.inspector = ix.opened(self.case)
        self.pkg = ix.package()
        self.rules = self.pkg.query.QueryRules.of(self.case.spec)
        if shape == "window":
            frame = ix.window_rows(self.case)
        elif shape == "graph":
            frame = self.case.rows.head(6 if which == "graph-keypair" else 12)
        else:
            frame = self.case.rows.head(4)
        self.frame = frame.reset_index(drop=True)
        self.records = _records(self.frame)

    @property
    def opened(self):
        return self.inspector._open

    def body(self, at=None, **fields) -> dict:
        """An explain request body for row `at` (a row sample sends that row alone)."""
        if self.shape == "row":
            return {"row": self.records[at or 0], **fields}
        return {"rows": self.records, **({} if at is None else {"at": at}), **fields}

    def explain(self, at=None, **fields) -> dict:
        return self.inspector.explain(self.rules.parse(self.body(at, **fields)))

    def plain_head(self, frame=None):
        """(the prepared request, the network's own head on it) — the test's own forward."""
        prepared = self.predictor.prepare(self.frame if frame is None else frame)
        with torch.no_grad():
            return prepared, self.case.model(prepared.batch)


_KITS = {}


def kit(name) -> Kit:
    if name not in _KITS:
        _KITS[name] = Kit(name)
    return _KITS[name]


def _valid(validator, doc):
    doc = json.loads(json.dumps(doc, allow_nan=False))
    errors = sorted(validator.iter_errors(doc), key=lambda e: list(e.path))
    assert not errors, f"{list(errors[0].path)}: {errors[0].message[:400]}"
    return doc


def _selectors(k: Kit) -> list:
    """Every target request a task admits: each kind by name, and each selector's values."""
    net, out = k.inspector.network(), k.case.manifest["output"]
    task, names = net["task"], net["task"]["outputs"]["names"]
    per = {"logit": [{"class": c} for c in names],
           "binary_logit": [{"class": c} for c in out.get("classes") or ()],
           "label_logit": [{"label": c} for c in names],
           "threshold_logit": [{"threshold": j} for j in range(len(names))],
           "quantile": [{"quantile": j} for j in range(len(task["quantiles"] or ()))],
           "forecast_step": [{"step": h} for h in range(1, int(net["sample"]["horizon"] or 0) + 1)],
           "feature_error": [{"feature": f} for f in names]}
    out = []
    for kind in task["targets"]:
        out += [{"kind": kind}] + per.get(kind, [])
    return out


# ================================================================ explain

@pytest.mark.parametrize("name", CASES)
def test_every_allowed_target_explains_and_validates(name):
    k = kit(name)
    tolerance = float(k.case.manifest["roundtrip"]["tolerance"]["value"])
    seen = set()
    for request in _selectors(k):
        body = _valid(EXPLANATION, k.explain(target=request))
        fid = body["fidelity"]
        assert fid["ok"] and fid["head_max_abs_diff"] <= 1e-6, (request, fid)
        assert fid["served_max_abs_diff"] is None or fid["served_max_abs_diff"] <= tolerance
        assert fid["tolerance"] == pytest.approx(tolerance)
        # a selector is requested; a kind named alone picks its class or label from the head
        assert body["target"]["why"] in (("requested", "predicted") if "kind" in request
                                         else ("requested",)), request
        if "kind" in request:
            assert body["target"]["kind"] == request["kind"]
        seen.add(body["target"]["kind"])
    assert seen == set(k.inspector.network()["task"]["targets"])
    default = _valid(EXPLANATION, k.explain())
    assert default["target"]["kind"] == k.inspector.network()["task"]["default_target"]
    assert default["target"]["why"] in ("predicted", "default")


# the kinds a served field answers (R-C5): the explanation's target is compared with it
SERVED = {"row:binary", "row:multiclass", "row:multilabel", "row:regression", "row:quantile",
          "row:reconstruction", "row:anomaly-reconstruction", "row:anomaly-svdd",
          "window:forecast", "window:sequence-binary", "graph:graph-keypair",
          "graph:graph-level"}


@pytest.mark.parametrize("name", sorted(SERVED))
def test_the_default_target_is_the_served_answer(name):
    k = kit(name)
    body = k.explain()
    assert body["fidelity"]["served_max_abs_diff"] is not None, body["target"]
    assert body["fidelity"]["served_max_abs_diff"] <= body["fidelity"]["tolerance"]


@pytest.mark.parametrize("name", NAMED)
@pytest.mark.parametrize("method", ("grad_x_input", "integrated_gradients"))
@pytest.mark.parametrize("detail", ("full", "summary", "exact"))
def test_methods_and_details_validate(name, method, detail):
    k = kit(name)
    body = _valid(EXPLANATION, k.explain(method=method, detail=detail, steps=16))
    assert body["detail"] == detail and body["attribution"]["method"] == method
    assert (body["attribution"]["completeness"] is None) == (method == "grad_x_input")
    levels = {n["level"] for n in body["nodes"].values()}
    assert levels <= {detail, "summary", "omitted"} and not body["degraded"]
    assert set(body["nodes"]) == {n["id"] for n in k.inspector.network()["nodes"]}


@pytest.mark.parametrize("name", NAMED)
def test_integrated_gradients_complete_on_the_three_families(name):
    k = kit(name)
    done = k.explain(method="integrated_gradients", steps=64)["attribution"]["completeness"]
    assert done["gap"] <= 1e-2 * max(1.0, abs(done["target_minus_baseline"])), done


@pytest.mark.parametrize("name", CASES)
def test_the_explained_sample_is_the_served_batchs(name):
    k = kit(name)
    query = k.rules.parse(k.body())
    prepared, head = k.plain_head(query.rows)
    sample = query.sample(prepared.index)
    body = k.inspector.explain(query)
    assert body["sample"]["index"] == sample and body["sample"]["row"] == query.at
    assert np.allclose(body["outputs"]["raw"], head[sample].reshape(-1).numpy(), atol=1e-6)
    expected = k.predictor.predictions(query.rows)[query.at]
    assert body["prediction"]["warnings"] == list(expected.warnings) == body["warnings"]
    served = k.predictor.predict(query.rows).iloc[query.at]
    for field, value in body["prediction"].items():
        if field == "warnings":
            continue
        if pd.isna(served[field]):
            assert value is None, field
        else:
            _same_value(value, served[field], field)
    assert body["inputs"]["order"] == k.inspector.network()["inputs"]["order"]
    if k.shape == "row":
        assert body["inputs"]["raw"] == query.rows.iloc[0].where(query.rows.iloc[0].notna(),
                                                                 None).to_dict()


def _same_value(mine, theirs, field):
    if isinstance(theirs, (float, np.floating)):
        assert mine == pytest.approx(float(theirs), rel=1e-9, abs=1e-12), field
    else:
        assert str(mine) == str(theirs), field


def test_the_inputs_are_the_prepared_rows_values():
    k = kit("row:binary")
    query = k.rules.parse(k.body())
    prepared = k.predictor.prepare(query.rows)
    body = k.inspector.explain(query)
    x, ids = prepared.batch["x_num"][0], prepared.batch["x_cat"][0]
    assert body["inputs"]["shaped"] == pytest.approx([*x.tolist(), *ids.double().tolist()])
    w = kit("window:forecast")
    query = w.rules.parse(w.body())
    prepared = w.predictor.prepare(query.rows)
    body = w.inspector.explain(query)
    k_ = query.sample(prepared.index)
    x = prepared.batch["x_num"][k_]
    shaped = np.asarray(body["inputs"]["shaped"])
    assert shaped.shape == (4, 5)                      # L × (numeric, zone, past target)
    assert np.allclose(shaped[:, :3], x[:, :3].numpy()) and np.allclose(shaped[:, 4],
                                                                        x[:, 3].numpy())
    assert body["sample"]["span"] == list(prepared.index.spans[k_])
    assert body["sample"]["span"][1] == query.at


def test_the_graph_sections():
    k = kit("graph:graph-keypair")
    body = k.explain(at=2)
    prepared = k.predictor.prepare(k.rules.parse(k.body(at=2)).rows)
    graph = dict(body["sample"]["graph"])
    target = int(prepared.batch["target_index"][body["sample"]["index"]])
    names = graph.pop("names")
    assert graph == {"level": "node", "nodes": int(prepared.batch["x_num"].shape[0]),
                     "edges": int(prepared.batch["edge_index"].shape[1]),
                     "target_node": target}
    # every reference node the answer mentions by its own key (bus_key), never by its index
    context, att = k.case.context, body["attribution"]
    frame = context.nodes.set_index("pos")["raw:" + context.node_key]
    mentioned = {n["node"] for n in att["neighbours"]} | {v for e in att["edges"] or () for v in (e["src"], e["dst"])}
    refs = {i for i in mentioned if i < context.n_reference}
    assert names["column"] == context.node_key == "bus_key" and refs
    assert names["of"] == {str(i): str(frame[i]) for i in sorted(refs | {int(i) for i in names["of"]})}
    assert str(target) not in names["of"] and all(int(i) < context.n_reference for i in names["of"])
    assert body["attribution"]["shape"] == "graph" and body["attribution"]["level"] == "node"
    assert body["inputs"]["shaped"] == pytest.approx(prepared.batch["x_num"][target].tolist())
    level = kit("graph:graph-level").explain()
    assert level["sample"]["graph"]["level"] == "graph" and "names" not in level["sample"]["graph"]
    assert level["sample"]["graph"]["target_node"] is None
    assert len(level["inputs"]["shaped"]) == level["sample"]["graph"]["nodes"]


def test_only_a_graph_that_keys_its_nodes_names_them():
    context = ix.package().graphs.GraphContext
    meta = lambda params: context(meta={"params": params}, level=None, reference=None,  # noqa: E731
                                  nodes=None, entry=None)
    assert meta({"method": "knn", "k": 5}).node_key is None
    assert meta({"method": "spatial", "k": 5}).node_key is None
    assert meta({"method": "key_pair", "columns": ["street"]}).node_key is None   # a shared value
    keyed = meta({"method": "key_pair", "columns": ["bus_key", "feeder_to"]})
    assert keyed.node_key == "bus_key" and keyed.reference_names([0, 1]) == {}     # no reference


def test_the_dense_chain_recomputes_from_the_published_weights():
    k = kit("row:multiclass")
    net = k.inspector.network()
    assert net["chain"] is not None
    body = k.explain()
    chain = body["chain"]
    decode = k.pkg.codec.decode
    xs = [chain["inputs"]] + chain["post"][:-1]
    for layer, x, pre in zip(net["chain"]["layers"], xs, chain["pre"]):
        w = decode(net["tensors"][layer["weight"]])
        b = decode(net["tensors"][layer["bias"]])
        assert np.allclose(w @ np.asarray(x) + b, pre, rtol=1e-4, atol=1e-5), layer["node"]
    assert kit("window:forecast").explain()["chain"] is None


def test_the_binary_sign_flips_with_the_class():
    k = kit("row:binary")
    negative, positive = k.case.manifest["output"]["classes"]
    a = k.explain(target={"class": positive})["target"]
    b = k.explain(target={"class": negative})["target"]
    assert (a["sign"], b["sign"]) == (1, -1) and a["value"] == pytest.approx(-b["value"])


def test_explain_refusals_are_the_querys(monkeypatch):
    k = kit("row:regression")
    q = k.pkg.query
    with pytest.raises(q.QueryError) as got:
        k.explain(target={"class": "a"})
    assert got.value.field == "target.class" and "regression" in str(got.value)
    w = kit("window:forecast")
    rows = ix.window_rows(w.case, context=3)
    short = w.rules.parse({"rows": _records(rows)})
    unanswered = q.ExplainQuery(short.rows, None, 1)
    with pytest.raises(q.QueryError, match="not answered by the prepared batch"):
        w.inspector.explain(unanswered)
    nan = q.ExplainQuery(short.rows, None, 3)
    with pytest.raises(q.QueryError, match="not finite") as got:
        w.inspector.explain(nan)
    assert got.value.field == "at" and "lag features" in str(got.value)
    g = kit("graph:graph-keypair")        # one query row, but the reference graph's nodes too
    query = g.rules.parse({"rows": g.records[:1], "method": "integrated_gradients", "steps": 8})
    codec = g.opened.graph.codec
    monkeypatch.setattr(codec, "_limits", MappingProxyType(dict(codec.limits, ig_batch_max=100)))
    with pytest.raises(q.QueryError, match="ig_batch_max") as got:
        g.inspector.explain(query)
    assert got.value.field == "steps"


class _Shifted:
    """The Predictor's answers with every served number moved by `shift`: an explanation of
    them cannot reproduce them."""

    def __init__(self, predictor, shift=1.0):
        self.p, self.shift = predictor, shift

    def prepare(self, df_raw, history=None):
        return self.p.prepare(df_raw, history)

    def predict(self, df_raw, history=None):
        frame = self.p.predict(df_raw, history)
        numbers = [c for c in frame.columns if frame[c].dtype.kind == "f"]
        return frame.assign(**{c: frame[c] + self.shift for c in numbers})

    def info(self):
        return self.p.info()


def _inspector(k, answers, model=None):
    """A fresh inspector of the case's network, built again, over `answers`."""
    model = ix.fresh_model(k.case) if model is None else model
    manifest = k.case.manifest
    return k.pkg.inspector.NetworkInspector(ix.pf.port_only(model), k.case.spec, answers,
                                            manifest["output"],
                                            manifest["roundtrip"]["tolerance"]["value"]), model


def test_an_explanation_that_does_not_reproduce_the_served_answer_is_refused():
    k = kit("row:regression")
    answers = _Shifted(k.predictor, 0.0)
    inspector, model = _inspector(k, answers)
    assert inspector.open(ix.example_query(k.case)).ready
    answers.shift = 1.0                    # the served answers change after the opening
    with pytest.raises(k.pkg.inspection.FidelityError, match="does not reproduce"):
        inspector.explain(k.rules.parse(k.body()))
    inspector.close()
    assert _hooks_on(model) == 0


# ================================================================ open

@pytest.mark.parametrize("name", CASES)
def test_open_reports_ready_with_its_self_test(name):
    k = kit(name)
    status = k.inspector.status()
    assert status["enabled"] and status["ready"] and status["why"] is None, status
    assert status["weights"] == "full" and status["pinned"] is False
    net = k.inspector.network()
    assert status["network_digest"] == net["network_digest"]
    assert status["coverage"] == {"nodes": len(net["nodes"]),
                                  "types": sorted({n["type"] for n in net["nodes"]}),
                                  "opaque": []}
    prepared = k.predictor.prepare(ix.example_query(k.case).rows)
    finite = k.opened.explainer.finite(prepared)
    assert status["fidelity"]["samples"] == min(16, len(finite)) > 0
    assert status["fidelity"]["max_abs_diff"] <= status["fidelity"]["tolerance"]
    assert status["recompute"]["ok"] and status["recompute"]["worst"] >= 0
    hooks = k.opened.hooks.count
    again = k.inspector.open(ix.example_query(k.case))
    assert again.to_dict() == status and k.opened.hooks.count == hooks
    assert _hooks_on(k.case.model) == hooks


def _hooks_on(model) -> int:
    return sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules())


def test_the_window_example_skips_samples_without_a_finite_answer():
    k = kit("window:forecast")
    rows = ix.window_rows(k.case, context=3)
    inspector, model = _inspector(k, k.predictor)
    status = inspector.open(ix.example_query(k.case, rows))
    prepared = k.predictor.prepare(rows)
    answered = [s for s in np.asarray(prepared.sample_of_row) if s >= 0]
    assert status.ready and status.fidelity["samples"] == len(answered) - 2
    inspector.close()
    assert _hooks_on(model) == 0
    none = inspector.open(ix.example_query(k.case, rows.head(3)))
    assert not none.ready and "no sample the network answers with finite numbers" in none.why
    assert _hooks_on(model) == 0


def test_the_pinned_digest_is_checked():
    k = kit("row:multiclass")
    digests = {mode: json.loads(k.inspector.network_bytes(mode))["network_digest"]
               for mode in ("full", "summary")}
    pinned = ix.opened(k.case, ix.with_serve(k.case, {"network_digest": digests}),
                       ix.fresh_model(k.case))[1]
    assert pinned.status()["ready"] and pinned.status()["pinned"]
    pinned.close()
    wrong = dict(digests, full="sha256:" + "0" * 64)
    model = ix.fresh_model(k.case)
    refused = ix.opened(k.case, ix.with_serve(k.case, {"network_digest": wrong}), model)[1]
    status = refused.status()
    assert not status["ready"] and status["why"].startswith(
        "the network the service reads is not the one packed")
    assert status["coverage"] is None and _hooks_on(model) == 0
    with pytest.raises(k.pkg.inspector.InspectionUnavailable, match="not the one packed"):
        refused.explain(k.rules.parse(k.body()))
    with pytest.raises(k.pkg.inspector.InspectionUnavailable):
        refused.network()


def test_a_svdd_network_without_its_centre_does_not_open():
    k = kit("row:anomaly-svdd")
    model = ix.fresh_model(k.case)
    p = ix.predictor(k.case, model=model)
    with torch.no_grad():
        model.svdd_center_set.fill_(False)
    status = p.inspector().open(ix.example_query(k.case))
    assert not status.ready and status.why == "the one-class centre was not persisted with the " \
                                              "weights"
    assert _hooks_on(model) == 0


def test_unfaithful_answers_fail_the_self_test():
    k = kit("row:binary")
    inspector, model = _inspector(k, _Shifted(k.predictor, 0.5))
    status = inspector.open(ix.example_query(k.case))
    assert not status.ready and "does not reproduce the served answer" in status.why
    assert _hooks_on(model) == 0


def test_inspection_switched_off_by_the_manifest():
    k = kit("row:regression")
    p = ix.predictor(k.case, ix.with_serve(k.case, {"enabled": False}),
                     ix.fresh_model(k.case))
    status = p.inspector().open(ix.example_query(k.case))
    assert status.to_dict()["enabled"] is False and not status.ready
    assert status.why == "this bundle ships without inspection"
    with pytest.raises(k.pkg.inspector.InspectionUnavailable, match="without inspection"):
        p.inspector().explain(k.rules.parse(k.body()))


def test_not_opened():
    k = kit("row:regression")
    inspector, _ = _inspector(k, k.predictor)
    assert inspector.status() == {"enabled": True, "ready": False,
                                  "why": "the inspector has not been opened on example rows",
                                  "weights": "full", "network_digest": None, "pinned": False,
                                  "coverage": None, "fidelity": None, "recompute": None}
    with pytest.raises(k.pkg.inspector.InspectionUnavailable, match="not been opened"):
        inspector.attribute(k.frame)


def test_weights_the_bundle_withholds_are_not_published():
    k = kit("row:multiclass")
    p, inspector = ix.opened(k.case, ix.with_serve(k.case, {"weights": "summary"}),
                             ix.fresh_model(k.case))
    assert inspector.status()["ready"] and inspector.weights == "summary"
    with pytest.raises(k.pkg.inspector.InspectionUnavailable, match="raw weights"):
        inspector.network("full")
    summary = inspector.network("summary")
    assert {t["encoding"] for t in summary["tensors"].values()} == {"withheld"}
    with pytest.raises(k.pkg.codec.TensorUnavailable, match="raw weights"):
        inspector.tensor(next(iter(summary["tensors"])))
    with pytest.raises(ValueError, match="weights mode"):
        inspector.network("raw")
    inspector.close()


# ================================================================ network

@pytest.mark.parametrize("name", CASES)
def test_the_network_validates_and_is_canonical(name):
    k = kit(name)
    net = _valid(NETWORK, k.inspector.network())
    assert k.inspector.network_bytes() == k.pkg.view.canonical(net)
    assert net["network_digest"] == k.pkg.view.network_digest(net)
    assert net["model_version"] == ix.MODEL_VERSION
    ids = {n["id"] for n in net["nodes"]}
    assert {b["id"] for b in k.case.spec["blocks"]} <= ids
    assert net["params"]["total"] == int(k.case.model.param_count()[0])
    assert k.inspector.network() == net and k.inspector.network() is not net


def test_tensors_decode_to_the_live_parameters():
    k = kit("row:binary")
    net = k.inspector.network()
    params = dict(k.case.model.named_parameters())
    weight = net["chain"]["layers"][0]["weight"]
    got = k.pkg.codec.decode(k.inspector.tensor(weight))
    module = k.opened.graph.module(net["chain"]["layers"][0]["node"])
    live = getattr(module, "linear", module).weight.detach().numpy()
    assert np.array_equal(got, live) and any(np.array_equal(live, p.detach().numpy())
                                             for p in params.values())
    with pytest.raises(k.pkg.codec.TensorUnavailable, match="no tensor"):
        k.inspector.tensor("nowhere.weight")


# ================================================================ attribute

def _table_rows(k):
    """(rows, history) a table is asked for: a window sends its context as history."""
    if k.shape == "window":
        n = k.case.manifest["sample"]["history_rows"]
        return k.frame.iloc[n:].reset_index(drop=True), k.frame.iloc[:n]
    return k.frame, None


def _explaining(k, rows, history, row, method="grad_x_input"):
    """The explain query of the sample answering request row `row`: a row alone; a window's
    anchor among the rows, their history before them; a query node; a row of the graph."""
    q = k.pkg.query
    if k.shape == "row":
        return q.ExplainQuery(rows.iloc[[row]].reset_index(drop=True), None, 0, method=method,
                              steps=8)
    sample = k.rules.sample
    graph = rows[sample.graph.graph_id].iloc[row] if sample.is_graph_level else None
    return q.ExplainQuery(rows, history, row, method=method, steps=8, graph=graph)


def _entry_of(k, inputs, mine) -> list:
    """The table entry `explanation.attribution` (`mine`) implies, flat: a row's attributions;
    a window's per-step then per-feature sums over `inputs` (a nominal summed over its steps);
    a node's own attributions then its per-hop totals; a graph's per-feature sums over its
    nodes (each node lists every feature here) then its one hop."""
    if k.shape == "row":
        return list(mine["features"]["attribution"])
    if k.shape == "window":
        numeric, nominal = mine["features"]["names"], mine["nominal"] or {"names": []}
        per = [mine["per_feature"][numeric.index(n)] if n in numeric else
               sum(nominal["attribution"][nominal["names"].index(n)]) for n in inputs]
        return list(mine["per_step"]) + per
    hops = [mine["hops"][h] for h in sorted(mine["hops"])]
    if mine["level"] == "node":
        return list(mine["self"]) + hops
    assert abs(mine["outside"]) <= 1e-9 and all(len(n["top"]) == len(inputs)
                                                for n in mine["neighbours"])
    per = [0.0] * len(inputs)
    for node in mine["neighbours"]:
        for f, value in node["top"]:
            per[f] += value
    return per + hops


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("method", ("grad_x_input", "integrated_gradients"))
def test_the_attribution_table_validates_and_is_the_explanations(name, method):
    k = kit(name)
    rows, history = _table_rows(k)
    if name.endswith("level"):
        rows = k.case.rows.head(40).reset_index(drop=True)          # several graphs
    table = _valid(TABLE, k.inspector.attribute(rows, method=method, steps=8, history=history))
    prepared = k.predictor.prepare(rows, history)
    pairs = k.pkg.explanation.answered_rows(prepared)
    assert [s["row"] for s in table["samples"]] == [row for _, row in pairs]
    assert table["shape"] == k.shape and table["inputs"] == \
        k.inspector.network()["inputs"]["order"]
    assert len(table["targets"]) == len(pairs) and table["method"] == method
    served = k.predictor.predictions(rows, history)
    for sample in table["samples"]:
        assert sample["prediction"]["warnings"] == list(served[sample["row"]].warnings)
    assert len(pairs) > (0 if k.shape == "row" else 1), "more than one sample to compare"
    for (sample_k, row), entry in list(zip(pairs, table["samples"]))[:3]:
        body = k.inspector.explain(_explaining(k, rows, history, row, method))
        assert body["sample"]["index"] == (0 if k.shape == "row" else sample_k)
        assert entry["target_value"] == pytest.approx(body["target"]["value"], abs=1e-6)
        got = entry["attribution"]
        if isinstance(got, dict):
            if "hops" in got:
                assert sorted(got["hops"]) == sorted(body["attribution"]["hops"]), row
            got = [*got.get("per_step", ()), *(got.get("per_feature") or got["self"]),
                   *(got["hops"][h] for h in sorted(got.get("hops") or {}))]
        want = _entry_of(k, table["inputs"], body["attribution"])
        assert got == pytest.approx(want, abs=1e-5), (row, got, want)


def test_the_table_takes_one_selector_per_row():
    k = kit("row:multiclass")
    classes = k.case.manifest["output"]["classes"]
    chosen = [{"class": classes[i % len(classes)]} for i in range(len(k.frame))]
    table = k.inspector.attribute(k.frame, targets=chosen)
    assert [t["name"] for t in table["targets"]] == [c["class"] for c in chosen]
    assert {t["why"] for t in table["targets"]} == {"requested"}
    mixed = kit("row:multiclass").inspector.attribute(k.frame, targets=[None, *chosen[1:]])
    assert mixed["targets"][0]["why"] == "predicted"
    with pytest.raises(ValueError, match="one per row"):
        k.inspector.attribute(k.frame, targets=chosen[:-1])
    with pytest.raises(ValueError, match="a mapping"):
        k.inspector.attribute(k.frame, targets=["a"] * len(k.frame))
    with pytest.raises(k.pkg.query.QueryError, match="target.label"):
        k.inspector.attribute(k.frame, targets=[{"label": "a"}] * len(k.frame))
    o = kit("row:ordinal")
    with pytest.raises(ValueError, match="one unit"):
        o.inspector.attribute(o.frame, targets=[{"threshold": 0}] + [None] * (len(o.frame) - 1))


@pytest.mark.parametrize("name", ("row:multiclass", "window:sequence-binary",
                                  "graph:graph-keypair", "graph:graph-level"))
def test_chunks_give_the_one_batch_numbers(name, monkeypatch):
    k = kit(name)
    rows, history = _table_rows(k)
    if name.endswith("level"):
        rows = k.case.rows.head(40).reset_index(drop=True)
    whole = k.inspector.attribute(rows, method="integrated_gradients", steps=8, history=history)
    monkeypatch.setattr(k.pkg.table, "CHUNK", 2)
    chunked = k.inspector.attribute(rows, method="integrated_gradients", steps=8,
                                    history=history)
    assert len(whole["samples"]) > 2
    assert json.dumps(whole["targets"]) == json.dumps(chunked["targets"])
    for a, b in zip(whole["samples"], chunked["samples"]):
        assert a["row"] == b["row"] and a["target_value"] == pytest.approx(b["target_value"])
        assert json.dumps(_flat(a["attribution"])) == json.dumps(_flat(b["attribution"])) or \
            _flat(a["attribution"]) == pytest.approx(_flat(b["attribution"]), abs=1e-6)


def _flat(value) -> list:
    if isinstance(value, dict):
        return [x for key in sorted(value) for x in _flat(value[key])]
    if isinstance(value, list):
        return [x for v in value for x in _flat(v)]
    return [] if value is None else [float(value)]


def test_the_table_refuses_rows_without_a_finite_answer():
    k = kit("window:forecast")
    with pytest.raises(ValueError, match="not finite"):
        k.inspector.attribute(ix.window_rows(k.case, context=3))


def _floated(k):
    """(rows, history) of the table's request with the text column `zone` sent as a float: the
    schema guard's warning then holds the frame's separator ("zone arrived as float; the
    schema declares it text (...)")."""
    rows, history = _table_rows(k)
    return rows.assign(zone=1.0), None if history is None else history.assign(zone=1.0)


@pytest.mark.parametrize("name", ("row:binary", "window:forecast", "graph:graph-keypair"))
def test_a_warning_holding_the_separator_stays_one_warning(name):
    k = kit(name)
    rows, history = _floated(k)
    served = k.predictor.predictions(rows, history)
    sep = k.pkg.explanation.WARNING_SEP
    assert any(sep in w for p in served for w in p.warnings), "the case this test is about"
    for at in sorted({0, len(rows) - 1}):
        query = _explaining(k, rows, history, at)
        body = k.inspector.explain(query)
        want = list(k.predictor.predictions(query.rows, query.history)[query.at].warnings)
        assert body["warnings"] == body["prediction"]["warnings"] == want, (at, want)
    table = k.inspector.attribute(rows, history=history)
    assert table["samples"]
    for sample in table["samples"]:
        assert sample["prediction"]["warnings"] == list(served[sample["row"]].warnings)


def test_the_joined_text_is_split_only_where_it_is_unambiguous():
    k = kit("row:binary")
    served_of, calls = k.pkg.explanation.Served.of, []

    class Counting:
        """The Predictor, counting the `predictions` runs."""

        def predict(self, df_raw, history=None):
            return k.predictor.predict(df_raw, history)

        def predictions(self, df_raw, history=None):
            calls.append(len(df_raw))
            return k.predictor.predictions(df_raw, history)

    def lists(served, rows):
        return [served.prediction(i)["warnings"] for i in range(len(rows))]

    plain = k.frame.assign(zone=["no-such-zone"] + list(k.frame["zone"].iloc[1:]))
    served = served_of(Counting(), plain)          # one warning or none per row: no ambiguity
    assert served.notes is None and not calls
    assert lists(served, plain) == [list(p.warnings) for p in k.predictor.predictions(plain)]
    assert lists(served, plain)[0], "row 0 carries its unseen-category warning"
    rows, _ = _floated(k)
    served = served_of(Counting(), rows)
    assert served.notes is not None and calls == [len(rows)]
    assert lists(served, rows) == [list(p.warnings) for p in k.predictor.predictions(rows)]
    # a harness without `predictions` has only the joined text: it is split, as documented
    split = served_of(_Shifted(k.predictor, 0.0), rows).prediction(0)["warnings"]
    assert len(split) == len(k.predictor.predictions(rows)[0].warnings) + 1


def test_a_selector_of_the_wrong_form_is_refused_not_read_as_an_index():
    o = kit("row:ordinal")
    q, n = o.pkg.query, len(o.frame)
    for bad in ({"threshold": 0.5}, {"threshold": True}, {"threshold": "0"},
                {"threshold": -1}, {"threshold": np.float64(1.0)}):
        with pytest.raises(q.QueryError) as got:
            o.inspector.attribute(o.frame, targets=[None] * (n - 1) + [bad])
        assert got.value.field == "target.threshold" and f"targets[{n - 1}]" in str(got.value)
        with pytest.raises(q.QueryError) as got:
            o.inspector.explain(q.ExplainQuery(o.frame.head(1), None, 0, target=bad))
        assert got.value.field == "target.threshold", bad
    for good in (1, np.int64(1)):
        table = o.inspector.attribute(o.frame, targets=[{"threshold": good}] * n)
        assert {t["index"] for t in table["targets"]} == {1}
        body = o.inspector.explain(q.ExplainQuery(o.frame.head(1), None, 0,
                                                  target={"threshold": good}))
        assert body["target"]["index"] == 1
    m = kit("row:multiclass")
    for bad, field in (({"class": 1}, "target.class"), ({"klass": "a"}, "target.klass"),
                       ({"class": "a", "label": "b"}, "target")):
        with pytest.raises(q.QueryError) as got:
            m.inspector.attribute(m.frame, targets=[bad] * len(m.frame))
        assert got.value.field == field and "targets[0]" in str(got.value), bad
    for fields, field in (({"method": "saliency"}, "method"), ({"detail": "all"}, "detail"),
                          ({"at": True}, "at"), ({"at": 0.0}, "at")):
        fields = {"at": 0, **fields}
        with pytest.raises(q.QueryError) as got:
            m.inspector.explain(q.ExplainQuery(m.frame.head(1), None, **fields))
        assert got.value.field == field, fields
    assert m.inspector.explain(q.ExplainQuery(m.frame.head(1), None, np.int64(0)))["sample"]
    with pytest.raises(ValueError, match="attribution method"):
        m.inspector.attribute(m.frame, method="saliency")


# ================================================================ hygiene

@pytest.mark.parametrize("name", NAMED)
def test_nothing_is_written_to_the_network(name):
    k = kit(name)
    model = k.case.model
    state = {key: v.clone() for key, v in model.state_dict().items()}
    hooks = _hooks_on(model)
    rows, history = _table_rows(k)
    for method in ("grad_x_input", "integrated_gradients"):
        k.explain(method=method, detail="exact", steps=8)
        k.inspector.attribute(rows, method=method, steps=8, history=history)
    assert all(p.grad is None for p in model.parameters())
    assert _hooks_on(model) == hooks == k.opened.hooks.count and not model.training
    assert all(torch.equal(state[key], v) for key, v in model.state_dict().items())


@pytest.mark.parametrize("name", ("row:multiclass", "window:forecast"))
def test_concurrent_explanations_and_predictions_are_the_sequential_ones(name):
    k = kit(name)
    query = k.rules.parse(k.body())
    want = json.dumps(k.inspector.explain(query), sort_keys=True)
    answer = k.predictor.predict(k.frame).to_json()
    seen, answers, errors = [], [], []

    def explain():
        try:
            for _ in range(5):
                seen.append(json.dumps(k.inspector.explain(query), sort_keys=True))
        except Exception as exc:          # pragma: no cover - reported below
            errors.append(exc)

    def predict():
        for _ in range(15):
            answers.append(k.predictor.predict(k.frame).to_json())
    threads = [threading.Thread(target=explain), threading.Thread(target=explain),
               threading.Thread(target=predict)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and len(seen) == 10 and all(s == want for s in seen)
    assert len(answers) == 15 and all(a == answer for a in answers)


def test_the_predictor_builds_its_inspector_once():
    k = kit("row:regression")
    p = ix.predictor(k.case, model=ix.fresh_model(k.case))
    got = []
    threads = [threading.Thread(target=lambda: got.append(p.inspector())) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len({id(g) for g in got}) == 1 and got[0] is p.inspector()
    assert type(got[0]).__name__ == "NetworkInspector"


def test_a_package_without_the_inspection_modules_says_so():
    case = pf.row_case("regression")
    predictor = case.predictor(pf.package())           # S05's package: no inspection/
    with pytest.raises(LookupError, match="--no-inspect"):
        predictor.inspector()
    assert len(predictor.predict(case.rows.head(2))) == 2


def test_a_partial_render_is_not_read_as_a_bundle_without_inspection(monkeypatch):
    k = kit("row:regression")
    inspection = f"{k.pkg.name}.serving.inspection"
    monkeypatch.delitem(sys.modules, f"{inspection}.inspector")
    monkeypatch.setitem(sys.modules, f"{inspection}.attribution_table", None)    # not rendered
    with pytest.raises(ModuleNotFoundError) as got:
        ix.predictor(k.case, model=ix.fresh_model(k.case)).inspector()
    assert got.value.name == f"{inspection}.attribution_table"
    assert not isinstance(got.value, LookupError)       # no inspection/ at all: the test above


def test_the_inspector_falls_back_to_the_one_default_tolerance():
    k = kit("row:regression")
    manifest = {key: v for key, v in k.case.manifest.items() if key != "roundtrip"}
    p, inspector = ix.opened(k.case, manifest, ix.fresh_model(k.case))
    assert inspector.status()["fidelity"]["tolerance"] == \
        k.pkg.explanation.DEFAULT_TOLERANCE
    assert not hasattr(k.pkg.predict, "DEFAULT_TOLERANCE")
    inspector.close()


def test_the_rendered_package_holds_the_gate():
    pkg = ix.package()
    bmap = json.loads((pkg.root / "blueprint-map.json").read_text(encoding="utf-8"))
    for entry in bmap["entries"]:
        if entry.get("role") == "Predictor":
            entry["status"] = "implemented"
    mapped = {e["file"] for e in bmap["entries"]}
    assert {f"serving/{n}" for n in ix.inspection_names()} <= mapped
    markers, findings = conformance.check_report(pkg.root, bmap, pf._pipeline(pkg.name))
    wrong = [m for m in markers if m.rule in ("C3", "C4", "C5", "C7", "C14") and
             m.level.value == "ERROR"]
    assert not wrong, wrong
    assert not [f for f in findings if f.file.startswith("serving/")], findings


STDLIB = {"__future__", "copy", "dataclasses", "json", "math", "numbers", "threading", "types",
          "typing"}


@pytest.mark.parametrize("file", MINE)
def test_size_discipline_state_and_imports(file):
    text = (INSPECTION / file).read_text(encoding="utf-8")
    tree = ast.parse(text)
    assert conformance.size_findings(tree, file) == [], "C11–C13 must hold at error"
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = [t.id for t in targets if isinstance(t, ast.Name)]
            if isinstance(node.value, conformance.MUTABLE_NODES) and names != ["__all__"]:
                assert all(n.lstrip("_").isupper() for n in names), f"C7: {names} in {file}"
        if isinstance(node, ast.Import):
            assert {a.name.split(".")[0] for a in node.names} <= \
                STDLIB | {"numpy", "pandas", "torch"}, file
        if isinstance(node, ast.ImportFrom) and not node.level:
            assert node.module.split(".")[0] in STDLIB | {"numpy", "pandas", "torch"}, \
                node.module
        if isinstance(node, ast.ImportFrom) and node.level:
            assert node.level == 1, (file, node.module)          # siblings only, no adapter
    calls = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and
             isinstance(n.func, ast.Attribute)}
    assert not calls & {"backward", "train", "eval", "zero_grad", "register_forward_hook",
                        "register_forward_pre_hook", "load_state_dict", "copy_"}, file
    assert "{{" not in text


def test_the_templates_carry_no_dataset_literal():
    lint = REPO / "vega" / "scripts" / "tests" / "lint_templates.py"
    files = [INSPECTION / f for f in MINE] + [SERVING / "package" / "predict.py",
                                              SERVING / "bundle" / "predict.py"]
    run = subprocess.run([sys.executable, str(lint), *map(str, files)], capture_output=True,
                         text=True, timeout=300)
    assert run.returncode == 0, run.stdout + run.stderr


# ================================================================ bundle

@pytest.fixture(scope="module")
def bundles():
    return {"row": ix.bundle(ix.row_case("binary"), "i06_b_row", examples="positional"),
            "window": ix.bundle(ix.window_case("forecast"), "i06_b_window", examples="window"),
            "graph": ix.bundle(ix.graph_case("graph-keypair"), "i06_b_graph")}


@pytest.mark.parametrize("which", ("row", "window", "graph"))
def test_load_inspector_opens_on_the_bundles_examples(bundles, which):
    b = bundles[which]
    cli = pf.load_cli(b)
    inspector = cli.load_inspector(b.root)
    status = inspector.status()
    assert status["ready"] and status["fidelity"]["samples"] > 0, status
    assert inspector.network()["model_version"] == \
        json.loads((b.root / "MANIFEST.json").read_text(encoding="utf-8"))["model_version"]
    query = cli._example_query(b.root)
    assert query.detail == "summary" and len(query.rows) == len(
        cli.read_rows(b.root / "examples" / "rows.csv", b.root))
    assert cli.check_examples(b.root)["passed"]
    rows, history = query.rows.head(3), None
    if which == "window":                  # the last two rows, every earlier one their history
        rows, history = query.rows.iloc[-2:].reset_index(drop=True), query.rows.iloc[:-2]
    table = _valid(TABLE, inspector.attribute(rows, history=history))
    assert len(table["samples"]) == len(rows)


def test_the_cli_keeps_no_v22_path_and_reads_na_as_missing(bundles):
    text = (SERVING / "bundle" / "predict.py").read_text(encoding="utf-8")
    assert "V22" not in text and "store.load" not in text and "_adapter" not in text
    cli = pf.load_cli(bundles["row"])
    assert "load_inspector" in cli.__all__
    assert cli._same(pd.NA, None) and cli._same(pd.NA, float("nan"))
    assert not cli._same(pd.NA, 1) and cli._same(1, 1.0) and not cli._same("a", None)
    tree = ast.parse(re.sub(r"\{(package|env|mod_[a-z]+)\}", "x", text))
    assert conformance.size_findings(tree, "predict.py") == []


def test_the_bundles_roundtrip_test_is_green_beside_the_new_cli(bundles):
    b = bundles["row"]
    tests = b.root / "tests"
    tests.mkdir(exist_ok=True)
    shutil.copyfile(SERVING / "tests" / "test_roundtrip.py", tests / "test_roundtrip.py")
    env = {k: v for k, v in __import__("os").environ.items() if k != "VEGA_FITTED"}
    run = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                          "tests/test_roundtrip.py"], cwd=str(b.root), capture_output=True,
                         text=True, timeout=600, env=env)
    assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-2000:]
    assert "3 passed" in run.stdout          # the serving seam is the third claim
    shutil.rmtree(tests)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
