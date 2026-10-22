#!/usr/bin/env python3
"""
test_inspect_graph.py — the stage graph and its readers (`templates/serving/package/inspection/
{graph, wiring, view, registry, readers, readers_group, readers_seq, readers_graph}.py`),
proved on *rendered* packages (build plan I02; inspect-contract §3, §4.1, §5.3; R-B1, R-B2,
R-B5, R-B6, R-F7).

    python3 test_inspect_graph.py            (or: python3 -m pytest test_inspect_graph.py -q)
    CIDS_ROOT=<dir> python3 test_inspect_graph.py   (adds the CIC-IDS2017 R7 chain check)

One package is rendered per case — each of the catalog's 21 families, plus five user-defined
cases that reach what no family composition does (batchnorm, parallel, reshape, custom, a
learned positional table, an unpooled attention, colliding block ids, a rank-consistent head
on an otherwise dense chain, a recurrent block behind a channel-mixing one) — with the model templates (`domain.py`, `network.py`,
`readout.py`, the spliced `builder.py` and block files) and the serving package's inspection
templates layered in the Predictor's module. On each:

  closing    `/v1/network` validates `network-view-2.schema.json` (weights `full` and
             `summary`); every spec block id is a node; no `opaque` node but `custom`;
             `params.total == param_count()[0]`; `chain` is non-null exactly for the specs the
             dense-chain rule admits; every top-level `out_shape == Network.shapes[i]`.
  table      the readers' view table (view, call, formula, payload keys) is catalog.json's
             `inspection.views` for the 33 block types and the 5 synthetic ones; every node's
             payload carries exactly its type's keys; the target table is `inspection.targets`.
  structure  nested ids and edges (residual repeats, projections and adds with skip edges,
             parallel branches and merges, decoder layers, encoder layers), the stage-kind
             edges (embed, edges, gdrop, gpool, readout), `per` on graphs, axes.
  payloads   tensor ids resolve and decode to the live parameters; biases, the ordinal
             thresholds, batchnorm running statistics and the svdd centre (buffers); the chain
             recomputes the network's own output with its published weights.
  sections   `task` (positive, threshold, quantiles, decision rule, targets, from the spec or
             MANIFEST.output) and `inputs` (names equal `data_contract.features[].name`, the
             autoregressive channel, steps and tokens).
  refusals   a block type without a reader (`no inspection reader for block type`), a network
             that is not the spec's, a spec param the module contradicts, a training network.
  hygiene    building leaves no hook, no `.grad` and no changed state on the model; two builds
             give the same bytes and the digest verifies; code bindings name package files;
             the modules pass C11–C13 at `error`, hold no module-level mutable state, import
             only numpy, torch, the standard library and their package, and carry no dataset
             literal (`lint_templates.py`).

Exit 0 when every test passes.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from jsonschema import Draft202012Validator
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
BACKEND = SCRIPTS / "backends" / "pytorch"
TEMPLATES = BACKEND / "templates"
INSPECTION = TEMPLATES / "serving" / "package" / "inspection"
REPO = SCRIPTS.parents[1]
CATALOG = json.loads((SCRIPTS / "catalog.json").read_text(encoding="utf-8"))
SCHEMA = json.loads((SCRIPTS.parent / "references" / "schemas" / "network-view-2.schema.json")
                    .read_text(encoding="utf-8"))
MINE = ("graph.py", "wiring.py", "view.py", "registry.py", "readers.py", "readers_group.py",
        "readers_seq.py", "readers_graph.py")
for _p in (str(SCRIPTS), str(HERE), str(BACKEND)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import conformance  # noqa: E402
import render_templates as rt  # noqa: E402

STUB_LOSSES = '''"""A stub of the losses source: the stage graph builds networks, never a loss."""
# --8<-- losses
def build_loss(spec):
    raise NotImplementedError("test_inspect_graph.py renders no loss")
# -->8-- losses
'''

CUSTOM_GAIN = '''"""A custom block: y = x * g, one gain per feature."""
import torch
from torch import nn


class _Gain(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.gain = nn.Parameter(torch.linspace(0.5, 1.5, int(width)))

    def forward(self, x):
        return x * self.gain


def build(in_shape, params):
    return _Gain(in_shape[-1])
'''

# ================================================================ specs

def _num(name):
    return {"name": name, "dtype": "float32", "role": "numeric"}


def _nominal(name, card):
    return {"name": name, "dtype": "int64", "role": "nominal", "cardinality": card,
            "unseen_category_id": card - 1}


NUM = [_num(f"x{i}") for i in range(4)]
MIXED = NUM + [_nominal("zone", 5)]
WINDOW = {"shape": "window", "length": 6, "stride": 1, "context": "previous-part"}
NODE = {"shape": "graph", "edges": "shaped/edges.parquet", "level": "node",
        "protocol": "inductive", "batching": "full"}
LEVEL = {"shape": "graph", "edges": "shaped/edges.parquet", "level": "graph", "graph_id": "g",
         "protocol": "inductive", "batching": "graphs"}
BINARY = {"kind": "binary", "label": "y", "classes": [0, 1], "positive": 1}
ABC = ["a", "b", "c"]


def _b(bid, block_type, **params):
    return {"id": bid, "type": block_type, "params": params}


def _i(block_type, **params):
    return {"type": block_type, "params": params}


def _spec(family, blocks, task, *, sample=None, features=None, loss=None):
    features = MIXED if features is None else features
    return {"schema": "vega/netspec@1", "seed": 7, "task": task,
            "input": {"manifest_digest": "sha256:" + "ab" * 32},
            "data_contract": {"input_dim": len(features), "features": features,
                              "sample": sample or {"shape": "row"}},
            "family": {"name": family, "source": "user:test_inspect_graph"}, "blocks": blocks,
            "loss": loss or {"name": "bce_with_logits", "params": {}},
            "metrics": {"primary": "pr_auc"}, "optimizer": {"name": "adamw"},
            "training": {"max_epochs": 1, "batch_size": 8},
            "init": {"scheme": "kaiming_uniform", "bias": "zeros"}}


EMB = [_b("E", "embedding", columns=["zone"], dims={"zone": 3}), _b("C", "concat")]
MSE = {"name": "mse", "params": {}}
CE = {"name": "cross_entropy", "params": {}}
RECON = {"name": "reconstruction", "params": {}}


def _seq_task(base, **extra):
    kind = "sequence-regression" if base in ("regression", "ordinal") else \
        "sequence-classification"
    return {"kind": kind, "base_kind": base, "label": "y", **extra}


def _forecast(horizon, **sample):
    return {**WINDOW, "horizon": horizon, "autoregressive": True, **sample}


CASES = {
    "mlp-plain": _spec("mlp-plain", EMB + [
        _b("B3", "dense", units=16, activation="none"), _b("B4", "activation", name="gelu"),
        _b("B5", "dropout", p=0.1), _b("B6", "dense", units=8, activation="relu"),
        _b("B7", "dropout", p=0.2), _b("B8", "head", out=1)], BINARY),
    "mlp-residual": _spec("mlp-residual", EMB + [
        _b("B3", "dense", units=16, activation="relu"),
        _b("B4", "residual", repeat=2, projection="auto",
           inner=[_i("dense", units=8, activation="relu"), _i("layernorm"),
                  _i("activation", name="relu"), _i("dropout", p=0.1)]),
        _b("B5", "head", out=3)], {"kind": "multiclass", "label": "y", "classes": ABC}, loss=CE),
    "ft-transformer": _spec("ft-transformer", [
        _b("T", "feature_tokenizer", d=8),
        _b("X", "transformer_encoder", d=8, heads=2, layers=2, ff=16, dropout=0.1),
        _b("P", "pool", kind="cls"), _b("H", "head", out=1)],
        {"kind": "regression", "label": "y"}, loss=MSE),
    "cnn1d": _spec("cnn1d", [
        _b("E", "embedding", columns=["zone"], dims={"zone": 2}), _b("C", "concat"),
        _b("K", "conv1d", channels=8, kernel=3), _b("A", "activation", name="relu"),
        _b("Q", "pool1d", kind="max", size=2), _b("F", "flatten"),
        _b("D", "dense", units=8, activation="relu"), _b("H", "head", out=3)],
        _seq_task("multiclass", classes=ABC), sample=WINDOW, loss=CE),
    "gru": _spec("gru", [
        _b("R", "gru", hidden=8, num_layers=2, dropout=0.1, **{"return": "all"}),
        _b("A", "attention", heads=2, pool=True), _b("H", "head", out=3)],
        {"kind": "forecast", "label": "y"}, sample=_forecast(3), features=NUM, loss=MSE),
    "lstm": _spec("lstm", [
        _b("E", "embedding", columns=["zone"], dims={"zone": 2}), _b("C", "concat"),
        _b("R", "lstm", hidden=8, **{"return": "all"}), _b("P", "pool", kind="mean"),
        _b("H", "head", out=1)], _seq_task("regression"), sample=WINDOW, loss=MSE),
    "transformer-seq": _spec("transformer-seq", [
        _b("D", "dense", units=8, activation="none"), _b("P", "positional", max_len=64),
        _b("X", "transformer_encoder", d=8, heads=2, layers=1, ff=16),
        _b("M", "pool", kind="mean"), _b("H", "head", out=1)],
        _seq_task("binary", classes=[0, 1]), sample=WINDOW, features=NUM),
    "autoencoder": _spec("autoencoder", [
        _b("E", "embedding", columns=["zone"], dims={"zone": 2}), _b("C", "concat"),
        _b("S", "sequential", inner=[_i("dense", units=8, activation="relu"),
                                     _i("dense", units=6, activation="relu")]),
        _b("Z", "dense", units=3), _b("D", "decoder", widths=[6, 8], activation="gelu"),
        _b("H", "head", out=9, activation="none")],
        {"kind": "unsupervised", "label": None}, loss=RECON),
    "vae": _spec("vae", [
        _b("C", "concat"), _b("S", "sequential", inner=[_i("dense", units=8, activation="relu")]),
        _b("M", "dense", units=6), _b("Z", "reparameterize", z=3),
        _b("D", "decoder", widths=[8]), _b("H", "head", out=9)],
        {"kind": "anomaly", "label": None}, loss={"name": "vae", "params": {}}),
    "deep-svdd": _spec("deep-svdd", [
        _b("C", "concat"), _b("S", "sequential", inner=[
            _i("dense", units=8, activation="relu"), _i("dense", units=6, activation="relu")]),
        _b("H", "head", out=4, bias=False)],
        {"kind": "anomaly", "label": None}, features=NUM, loss={"name": "svdd", "params": {}}),
    "linear": _spec("linear", EMB + [_b("H", "head", out=3)],
                    {"kind": "multilabel", "label": "y", "classes": ABC,
                     "label_format": {"kind": "delimited", "sep": "|"}}),
    "pca-floor": _spec("pca-floor", [
        _b("C", "concat"), _b("Z", "dense", units=2, activation="none"),
        _b("H", "head", out=4, activation="none")],
        {"kind": "unsupervised", "label": None}, features=NUM, loss=RECON),
    "rnn": _spec("rnn", [
        _b("R", "rnn", hidden=8, num_layers=2, nonlinearity="relu", **{"return": "last"}),
        _b("D", "dropout", p=0.1), _b("H", "head", out=3)],
        _seq_task("multilabel", classes=ABC, label_format={"kind": "delimited", "sep": "|"}),
        sample=WINDOW, features=NUM),
    "bi-lstm": _spec("bi-lstm", [
        _b("R", "lstm", hidden=8, bidirectional=True, **{"return": "last"}),
        _b("D", "dropout", p=0.1), _b("H", "head", out=3)],
        _seq_task("ordinal", classes=["lo", "mid", "hi", "top"]), sample=WINDOW, features=NUM,
        loss={"name": "cumulative_link", "params": {}}),
    "bi-gru": _spec("bi-gru", EMB[:1] + [
        _b("C", "concat"),
        _b("R", "gru", hidden=6, bidirectional=True, num_layers=2, **{"return": "last"}),
        _b("D", "dropout", p=0.1), _b("H", "head", out=1)],
        _seq_task("binary", classes=[0, 1]), sample=WINDOW),
    "lstm-attention": _spec("lstm-attention", [
        _b("R", "lstm", hidden=8, **{"return": "all"}), _b("A", "attention_pool", hidden=4),
        _b("D", "dropout", p=0.1), _b("H", "head", out=6)],
        {"kind": "forecast", "label": "y"}, sample=_forecast(2), features=NUM,
        loss={"name": "quantile", "params": {"quantiles": [0.1, 0.5, 0.9]}}),
    "gru-attention": _spec("gru-attention", EMB + [
        _b("R", "gru", hidden=8, **{"return": "all"}), _b("A", "attention_pool", hidden=8),
        _b("D", "dropout", p=0.1), _b("H", "head", out=3)],
        _seq_task("multiclass", classes=ABC), sample=WINDOW, loss=CE),
    "gcn": _spec("gcn", [
        _b("G0", "edge_dropout", p=0.1)] + EMB + [
        _b("G1", "gcn_conv", out=8), _b("A1", "activation", name="relu"),
        _b("D1", "dropout", p=0.1), _b("G2", "gcn_conv", out=8, normalize="rw"),
        _b("H", "head", out=4)],
        {"kind": "multiclass", "label": "y", "classes": ["a", "b", "c", "d"]}, sample=NODE,
        loss=CE),
    "graphsage": _spec("graphsage", [
        _b("G1", "sage_conv", out=8), _b("A1", "activation", name="relu"),
        _b("D1", "dropout", p=0.1),
        _b("G2", "sage_conv", out=8, aggregator="max", root_weight=False),
        _b("H", "head", out=1)], {"kind": "regression", "label": "y"}, sample=NODE,
        features=NUM, loss=MSE),
    "gat": _spec("gat", [
        _b("G1", "gat_conv", out=4, heads=2, concat=True), _b("A1", "activation", name="relu"),
        _b("D1", "dropout", p=0.1), _b("G2", "gat_conv", out=4, heads=1, concat=False),
        _b("P", "graph_pool", readout="mean"), _b("H", "head", out=1)], BINARY, sample=LEVEL,
        features=NUM),
    "gin": _spec("gin", [
        _b("G1", "gin_conv", out=8, train_eps=True), _b("A1", "activation", name="relu"),
        _b("D1", "dropout", p=0.1), _b("G2", "gin_conv", out=8),
        _b("P", "graph_pool", readout="sum"), _b("H", "head", out=1)],
        {"kind": "regression", "label": "y"}, sample=LEVEL, features=NUM, loss=MSE),
    # beyond the families: what no composition reaches
    "user-mixed": _spec("user-defined", [
        _b("C", "concat"), _b("D", "dense", units=16, activation="relu"), _b("N", "batchnorm"),
        _b("P", "parallel", merge="concat", inner=[
            _i("sequential", inner=[_i("dense", units=8, activation="relu"),
                                    _i("activation", name="tanh")]),
            _i("dense", units=8)]),
        _b("R", "reshape", shape=[4, 4]), _b("M", "pool", kind="max"),
        _b("U", "custom", code_ref="gain.py", out_shape=[4]), _b("H", "head", out=1)],
        {"kind": "regression", "label": "y"}, loss=MSE),
    "user-ids": _spec("user-defined", [
        _i("concat"), _i("dense", units=8, activation="relu"),
        _i("dense", units=8, activation="relu"),
        _i("parallel", merge="sum", inner=[_i("dense", units=8), _i("dense", units=8)]),
        _i("head", out=1)], {"kind": "regression", "label": "y"}, features=NUM, loss=MSE),
    "user-ordinal": _spec("linear", [_b("C", "concat"), _b("H", "head", out=3)],
                          {"kind": "ordinal", "label": "y", "classes": ["lo", "mid", "hi", "top"]},
                          features=NUM, loss={"name": "cumulative_link", "params": {}}),
    "user-attention": _spec("user-defined", [
        _b("D", "dense", units=8), _b("P", "positional", kind="learned", max_len=16),
        _b("A", "attention", heads=2, pool=False), _b("M", "pool", kind="mean"),
        _b("H", "head", out=1)], _seq_task("regression"), sample=WINDOW, features=NUM,
        loss=MSE),
    "user-recurrent": _spec("user-defined", [
        _b("D", "dense", units=4), _b("R", "gru", hidden=4, **{"return": "last"}),
        _b("H", "head", out=1)], _seq_task("regression"), sample=WINDOW, features=NUM, loss=MSE),
}
FAMILIES = [n for n in CASES if not n.startswith("user-")]


# ================================================================ batches

def _gen(seed):
    return torch.Generator().manual_seed(seed)


EDGES = [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2), (4, 5), (5, 4), (5, 6), (6, 5),
         (7, 8), (8, 7), (8, 9), (9, 8), (3, 0)]


def batch_for(spec, seed=1):
    """An example batch of the spec's sample shape (the rows the Predictor would assemble)."""
    sample = spec["data_contract"]["sample"]
    feats = spec["data_contract"]["features"]
    numeric = sum(1 for f in feats if f["role"] != "nominal") + \
        (1 if sample.get("autoregressive") else 0)
    cards = [f["cardinality"] for f in feats if f["role"] == "nominal"]
    g = _gen(seed)
    lead = (10,) if sample["shape"] == "graph" else \
        (8, sample["length"]) if sample["shape"] == "window" else (8,)
    batch = {"x_num": torch.randn(*lead, numeric, generator=g)}
    if cards:
        batch["x_cat"] = torch.stack([torch.randint(0, c, lead, generator=g) for c in cards], -1)
    if sample["shape"] == "graph":
        batch["edge_index"] = torch.tensor(EDGES, dtype=torch.long).t().contiguous()
        if sample["level"] == "node":
            batch["edge_weight"] = torch.rand(len(EDGES), generator=g) + 0.5
            batch["target_index"] = torch.tensor([0, 2, 5])
        else:
            batch["graph_index"] = torch.tensor([0, 0, 0, 0, 1, 1, 1, 2, 2, 2])
            batch["num_graphs"] = 3
    return batch


# ================================================================ the rendered packages

class Case:
    """One rendered package: its modules, its network (eval mode), its graph and body."""

    def __init__(self, name, root, package, spec):
        self.name, self.root, self.package, self.spec = name, Path(root), package, spec
        mod = lambda m: importlib.import_module(f"{package}.{m}")  # noqa: E731
        self.domain, self.network = mod("domain"), mod("model.network")
        self.graph_mod, self.view = mod("serving.inspection.graph"), mod("serving.inspection.view")
        self.registry = mod("serving.inspection.registry")
        self.codec = mod("serving.inspection.codec")
        self.readers = mod("serving.inspection.readers")
        self.model = self.network.Network.build(self.netspec()).eval()
        self.batch = batch_for(self.spec)
        self.graph = self.build()
        self.body = self.view.network_view(self.graph, "full", "sha256:" + "1" * 64)

    def netspec(self, spec=None):
        return self.domain.NetSpec.from_dict(spec or self.spec)

    def build(self, spec=None, output=None, model=None):
        return self.graph_mod.StageGraph.build(model or self.model, self.netspec(spec),
                                               (self.batch, None), output)

    def nodes(self):
        return {n["id"]: n for n in self.body["nodes"]}


def _render(name, work, spec):
    package = "i02_" + name.replace("-", "_")
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
              only=["domain.py", "network.py", "readout.py"], splice=True, sources=str(src))
    rec = rt.render(TEMPLATES / "serving" / "package", out, pipeline, spec,
                    into="Predictor", recursive=True)
    assert {f"serving/inspection/{f}" for f in MINE} <= set(rec["files"]), rec["files"]
    custom = out / package / "model" / "custom"
    custom.mkdir(exist_ok=True)
    (custom / "gain.py").write_text(CUSTOM_GAIN, encoding="utf-8")
    sys.path.insert(0, str(out))
    return Case(name, out, package, spec)


_WORK = tempfile.TemporaryDirectory(prefix="inspect-graph-")
_CASES = {}


def case(name, spec=None) -> Case:
    if name not in _CASES:
        _CASES[name] = _render(name, Path(_WORK.name), spec or CASES[name])
    return _CASES[name]


def _valid(body):
    errors = sorted(Draft202012Validator(SCHEMA).iter_errors(body), key=lambda e: list(e.path))
    assert not errors, "; ".join(f"{list(e.path)}: {e.message[:240]}" for e in errors[:5])


def _top(c):
    """(expected node id, spec block) per top-level block (`<id>@<i>` on a collision)."""
    ids = [str(b.get("id") or b.get("type")) for b in c.spec["blocks"]]
    return [(f"{bid}@{i}" if ids.count(bid) > 1 else bid, b)
            for i, (bid, b) in enumerate(zip(ids, c.spec["blocks"]))]


def chain_admitted(spec) -> bool:
    """The dense-chain rule, read from the spec alone (inspect-contract §5.3)."""
    blocks = spec["blocks"]
    types = [b["type"] for b in blocks]
    task = spec["task"]
    if spec["data_contract"]["sample"]["shape"] != "row" or types[-1] != "head" or \
            not set(types) <= {"embedding", "concat", "dense", "activation", "dropout", "head"}:
        return False
    if task.get("base_kind", task["kind"]) == "ordinal":
        return False            # a rank-consistent head is not a W·x + b layer
    acts = None
    for b in blocks:
        if b["type"] in ("dense", "head"):
            if acts is not None and acts > 1:
                return False
            acts = int(b["params"].get("activation", "none") not in (None, "none"))
        elif b["type"] == "activation":
            if acts is None:
                return False
            acts += 1
    return acts is not None and acts <= 1


# ================================================================ closing tests

@pytest.mark.parametrize("name", list(CASES))
def test_network_validates_the_schema(name):
    c = case(name)
    _valid(c.body)
    summary = c.view.network_view(c.graph, "summary")
    _valid(summary)
    assert all(t["encoding"] == "withheld" and t["data"] is None
               for t in summary["tensors"].values())
    assert summary["model_version"] == c.body["spec_digest"] == c.model.spec_digest


def test_every_family_and_every_type_is_covered():
    assert sorted(FAMILIES) == sorted(CATALOG["families"])
    seen = {n["type"] for name in CASES for n in case(name).body["nodes"]}
    wanted = set(CATALOG["blocks"]) | set(CATALOG["inspection"]["synthetic"])
    assert seen == wanted, sorted(wanted ^ seen)


@pytest.mark.parametrize("name", list(CASES))
def test_every_spec_block_id_is_a_node(name):
    c = case(name)
    nodes = c.nodes()
    for i, (node_id, block) in enumerate(_top(c)):
        node = nodes[node_id]
        assert (node["type"], node["path"], node["parent"]) == (block["type"], f"stages.{i}",
                                                                 None)
        assert node["block"] == str(block.get("id") or block.get("type"))
        assert node["call"] == c.model.kinds[i].value


@pytest.mark.parametrize("name", list(CASES))
def test_no_opaque_node_but_custom(name):
    c = case(name)
    for n in c.body["nodes"]:
        assert (n["view"] == "opaque") == (n["type"] == "custom"), n["id"]


@pytest.mark.parametrize("name", list(CASES))
def test_params_total_is_the_networks(name):
    c = case(name)
    total, trainable = c.model.param_count()
    assert (c.body["params"]["total"], c.body["params"]["trainable"]) == (total, trainable)
    nodes = c.nodes()
    top = [nodes[i] for i, _ in _top(c)]
    assert sum(n["params"]["count"] for n in top) == total
    for n in c.body["nodes"]:          # a group holds no parameter of its own
        if n["children"]:
            assert n["params"]["count"] == sum(nodes[k]["params"]["count"]
                                               for k in n["children"]), n["id"]


@pytest.mark.parametrize("name", list(CASES))
def test_chain_exactly_for_eligible_specs(name):
    c = case(name)
    assert (c.body["chain"] is not None) == chain_admitted(c.spec), c.body["chain"]


@pytest.mark.parametrize("name", list(CASES))
def test_top_level_out_shape_is_the_networks(name):
    c = case(name)
    nodes = c.nodes()
    for (node_id, _), shape in zip(_top(c), c.model.shapes):
        assert nodes[node_id]["out_shape"] == [int(d) for d in shape], node_id


# ================================================================ the view table

def test_view_table_is_the_catalogs():
    reg = case("mlp-plain").registry
    views = CATALOG["inspection"]["views"]
    assert set(reg.READERS) == set(views) == set(CATALOG["blocks"]) | set(
        CATALOG["inspection"]["synthetic"])
    for t, row in views.items():
        r = reg.READERS[t]
        assert (r.TYPE, r.VIEW, r.CALL, r.FORMULA, list(r.PAYLOAD)) == \
            (t, row["view"], row["call"], row["formula"], row["payload"]), t
    table = reg.view_table()
    text = json.dumps(table, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    assert reg.readers_digest() == "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def test_target_table_is_the_catalogs():
    v = case("mlp-plain").view
    targets = CATALOG["inspection"]["targets"]
    for kind, row in targets.items():
        if "as_base" in row:
            assert all(b in v.TARGETS for b in row["as_base"])
            continue
        assert v.TARGETS[kind] == (row["link"], row["decision_rule"], row["default"],
                                   tuple(row["selectable"])), kind
        for loss, alt in (row.get("by_loss") or {}).items():
            assert v.BY_LOSS[(kind, loss)] == (alt["link"], alt["decision_rule"],
                                               alt["default"], tuple(alt["selectable"]))
    assert len(v.BY_LOSS) == sum(len(r.get("by_loss") or {}) for r in targets.values())


@pytest.mark.parametrize("name", list(CASES))
def test_payload_keys_are_the_view_tables(name):
    views = CATALOG["inspection"]["views"]
    for n in case(name).body["nodes"]:
        assert set(n["params"]) - {"count", "trainable"} == set(views[n["type"]]["payload"]), \
            n["id"]
        assert (n["view"], n["formula"]) == (views[n["type"]]["view"], views[n["type"]]["formula"])
        if n["parent"] is not None:
            assert n["call"] == "inner", n["id"]


# ================================================================ structure

def _edges(c):
    return {(e["from"], e["to"], e["kind"]) for e in c.body["edges"]}


def test_residual_repeats_projections_and_adds():
    c = case("mlp-residual")
    nodes, edges = c.nodes(), _edges(c)
    assert nodes["B4"]["children"] == ["B4.r0.0", "B4.r0.1", "B4.r0.2", "B4.r0.3", "B4.r0.proj",
                                       "B4.r0.add", "B4.r1.0", "B4.r1.1", "B4.r1.2", "B4.r1.3",
                                       "B4.r1.add"]
    proj = nodes["B4.r0.proj"]
    assert (proj["type"], proj["role"], proj["path"], proj["in_shape"], proj["out_shape"]) == \
        ("dense", "projection", "stages.3.projections.0", [16], [8])
    assert nodes["B4"]["params"]["repeats"] == [
        {"inner": ["B4.r0.0", "B4.r0.1", "B4.r0.2", "B4.r0.3"], "projection": "B4.r0.proj",
         "add": "B4.r0.add"},
        {"inner": ["B4.r1.0", "B4.r1.1", "B4.r1.2", "B4.r1.3"], "projection": None,
         "add": "B4.r1.add"}]
    assert {("B3", "B4", "data"), ("B4", "B4.r0.0", "data"), ("B4.r0.3", "B4.r0.add", "data"),
            ("B4", "B4.r0.proj", "skip"), ("B4.r0.proj", "B4.r0.add", "skip"),
            ("B4.r0.add", "B4.r1.0", "data"), ("B4.r0.add", "B4.r1.add", "skip"),
            ("B4.r1.add", "B5", "data"), ("B5", "output", "readout")} <= edges
    add = nodes["B4.r0.add"]
    assert (add["view"], add["call"], add["in_shape"], add["out_shape"]) == \
        ("merge", "inner", None, [8])
    assert nodes["B4.r0.1"]["type"] == "layernorm" and nodes["B4.r0.1"]["path"] == \
        "stages.3.repeats.0.1"


def test_parallel_branches_nest_and_merge():
    c = case("user-mixed")
    nodes, edges = c.nodes(), _edges(c)
    assert nodes["P"]["children"] == ["P.b0", "P.b1", "P.merge"]
    assert nodes["P"]["params"] == {**nodes["P"]["params"], "branches": ["P.b0", "P.b1"],
                                    "merge": "concat"}
    assert nodes["P.b0"]["type"] == "sequential" and \
        nodes["P.b0"]["children"] == ["P.b0.0", "P.b0.1"]
    assert {("P", "P.b0", "branch"), ("P", "P.b1", "branch"), ("P.b0", "P.b0.0", "data"),
            ("P.b0.1", "P.merge", "merge"), ("P.b1", "P.merge", "merge"),
            ("P.merge", "R", "data")} <= edges
    assert nodes["P.merge"]["out_shape"] == [16] and nodes["P.merge"]["params"]["merge"] == \
        "concat"
    ids = case("user-ids").nodes()
    assert {"concat", "dense@1", "dense@2", "parallel", "head"} <= set(ids)
    assert ids["parallel.merge"]["params"]["merge"] == "sum"


def test_decoder_layers_and_encoder_layers():
    c = case("autoencoder")
    nodes = c.nodes()
    layers = [nodes[f"D.l{k}"] for k in range(3)]
    assert [(n["type"], n["role"], n["path"]) for n in layers] == \
        [("dense", "decoder_layer", f"stages.4.net.{j}") for j in (0, 2, 4)]
    assert [n["params"]["activation"] for n in layers] == ["gelu", "gelu", "none"]
    assert [n["out_shape"] for n in layers] == [[6], [8], [9]]
    assert all(n["code"] is None for n in layers)          # torch's nn.Linear
    x = case("ft-transformer").nodes()
    assert x["X"]["children"] == ["X.layer0", "X.layer1"]
    layer = x["X.layer0"]
    assert (layer["type"], layer["view"], layer["call"], layer["axes"], layer["out_shape"]) == \
        ("encoder_layer", "attention", "inner", ["token", "channel"], [6, 8])
    assert layer["params"]["self_attn"]["heads"] == 2 and layer["params"]["norm_first"] is True
    assert x["X"]["params"] == {**x["X"]["params"], "layers": 2, "d": 8, "heads": 2, "ff": 16}


def test_graph_edges_calls_and_per():
    c = case("gcn")
    nodes, edges = c.nodes(), _edges(c)
    assert {("input.edges", "G0", "edges"), ("G0", "G1", "edges"), ("G0", "G2", "edges"),
            ("input.cat", "E", "embed"), ("input.num", "C", "data"), ("E", "C", "data"),
            ("C", "G1", "data"), ("G2", "H", "data")} <= edges
    assert ("input.num", "G0", "data") not in edges
    assert [nodes[n]["call"] for n in ("G0", "G1", "input.edges")] == ["gdrop", "graph", "io"]
    assert {n["per"] for n in c.body["nodes"] if n["id"] != "input.edges"} == {"node"}
    assert nodes["input.edges"]["per"] == "graph"
    assert c.body["inputs"]["edges"] == {"weighted": True}
    g = case("gat")
    per = {n["id"]: n["per"] for n in g.body["nodes"]}
    assert (per["G2"], per["P"], per["H"], per["output"]) == ("node", "graph", "graph", "graph")
    assert g.nodes()["P"]["call"] == "gpool" and g.body["inputs"]["edges"] == {"weighted": False}


@pytest.mark.parametrize("name", list(CASES))
def test_axes_name_every_kept_dimension(name):
    for n in case(name).body["nodes"]:
        assert len(n["axes"]) == len(n["out_shape"]), n["id"]
    nodes = case("gru").nodes()
    assert nodes["input.num"]["axes"] == ["step", "feature"] and \
        nodes["input.num"]["out_shape"] == [6, 5]
    assert nodes["R"]["axes"] == ["step", "channel"] and nodes["H"]["axes"] == ["output"]


# ================================================================ payloads

def _walk_refs(value, prefix, found):
    if isinstance(value, str) and value.startswith(prefix):
        found.append(value)
    elif isinstance(value, dict):
        for v in value.values():
            _walk_refs(v, prefix, found)
    elif isinstance(value, list):
        for v in value:
            _walk_refs(v, prefix, found)
    return found


@pytest.mark.parametrize("name", list(CASES))
def test_tensor_ids_resolve_and_decode_to_the_live_parameters(name):
    c = case(name)
    tensors, nodes = c.body["tensors"], c.nodes()
    for n in c.body["nodes"]:          # a payload cites its own tensors, or its child nodes
        for ref in _walk_refs(n["params"], n["id"] + ".", []):
            assert ref in tensors or ref in nodes, f"{n['id']} cites {ref}"
            assert ref not in tensors or tensors[ref]["node"] == n["id"]
    for tid, t in tensors.items():
        assert t["node"] in nodes and tid.startswith(t["node"] + ".")
        assert t["role"] in ("weight", "bias", "table", "buffer")
        if t["encoding"] in ("json", "base64-le"):
            live = c.codec.decode(c.graph.codec.fetch(tid))
            assert np.array_equal(c.codec.decode(t), live), tid


def test_weights_are_the_modules_own():
    c = case("mlp-plain")
    t = c.body["tensors"]
    dense = c.model.stages[2].linear
    assert np.array_equal(c.codec.decode(t["B3.weight"]), dense.weight.detach().numpy())
    assert np.array_equal(c.codec.decode(t["B3.bias"]), dense.bias.detach().numpy())
    table = c.nodes()["E"]["params"]["tables"][0]
    assert (table["column"], table["cardinality"], table["dim"], table["unseen_row"],
            table["table"]) == ("zone", 5, 3, 4, "E.table.zone")
    assert t["E.table.zone"]["role"] == "table" and len(table["pca2"]) == 5
    r = case("bi-lstm").nodes()["R"]["params"]
    assert (r["cell"], r["layers"], r["directions"], r["gates"], r["return"]) == \
        ("lstm", 1, 2, ["i", "f", "g", "o"], "last")
    assert [(p["layer"], p["direction"], p["w_ih"]) for p in r["per_layer"]] == \
        [(0, "forward", "R.l0.w_ih"), (0, "reverse", "R.l0.w_ih.reverse")]
    assert r["per_layer"][0]["input_labels"] == ["x0", "x1", "x2", "x3"]
    rnn = case("rnn").nodes()["R"]["params"]
    assert rnn["gates"] == ["h"] and [p["input_labels"] is None for p in rnn["per_layer"]] == \
        [False, True]


def test_channel_labels_follow_the_blocks_that_keep_channels():
    joined = case("lstm").nodes()["R"]["params"]["per_layer"][0]["input_labels"]
    assert joined == ["x0", "x1", "x2", "x3", "zone·e0", "zone·e1"]
    mixed = case("user-recurrent").nodes()["R"]["params"]["per_layer"][0]
    assert mixed["input_labels"] is None      # a dense mixes the channels: no feature names
    tokens = case("ft-transformer").nodes()["T"]["params"]["token_norms"]
    assert tokens["labels"] == ["x0", "x1", "x2", "x3"]
    assert case("ft-transformer").nodes()["T"]["params"]["tables"][0]["column"] == "zone"


def test_ordinal_head_is_the_shared_score_and_its_thresholds():
    c = case("user-ordinal")
    head = c.nodes()["H"]
    module = c.model.stages[1]
    assert head["params"]["bias"] is None and c.body["tensors"]["H.weight"]["shape"] == [1, 4]
    with torch.no_grad():
        theta = module.thresholds().numpy()
    assert np.array_equal(np.float32(head["params"]["thresholds"]), theta)
    assert head["config"] == {"out": 3, "activation": "none", "cumulative": True}
    assert {"H.first", "H.gaps"} <= set(c.body["tensors"])
    assert c.body["chain"] is None


def test_buffers_running_statistics_positional_and_the_svdd_centre():
    t = case("user-mixed").body["tensors"]
    assert (t["N.running_mean"]["role"], t["N.running_var"]["role"]) == ("buffer", "buffer")
    assert t["N.gamma"]["role"] == "weight" and t["N.beta"]["role"] == "bias"
    pe = case("transformer-seq").body["tensors"]["P.pe"]
    assert (pe["role"], pe["shape"]) == ("buffer", [6, 8])
    learned = case("user-attention").body["tensors"]["P.pe"]
    assert (learned["role"], learned["shape"]) == ("weight", [6, 8])
    c = case("deep-svdd")
    assert "H.centre" not in c.body["tensors"]         # never set: nothing is published
    model = copy.deepcopy(c.model)
    model.svdd_center.copy_(torch.tensor([0.5, -1.0, 2.0, 0.25]))
    model.svdd_center_set.fill_(True)
    body = c.view.network_view(c.build(model=model))
    centre = body["tensors"]["H.centre"]
    assert (centre["role"], centre["node"], centre["data"]) == ("buffer", "H",
                                                                [0.5, -1.0, 2.0, 0.25])


ACTS = {None: lambda z: z, "relu": F.relu, "gelu": F.gelu, "tanh": torch.tanh,
        "sigmoid": torch.sigmoid, "silu": F.silu, "leaky_relu": F.leaky_relu}


@pytest.mark.parametrize("name", [n for n in CASES if chain_admitted(CASES[n])])
def test_chain_recomputes_the_network_with_its_published_weights(name):
    c = case(name)
    chain, t = c.body["chain"], c.body["tensors"]
    types = [b["type"] for b in c.spec["blocks"]]
    with torch.no_grad():
        emb = c.model.stages[types.index("embedding")](c.batch["x_cat"]) \
            if "embedding" in types else None
        h = c.model.stages[types.index("concat")](c.batch["x_num"], emb)
        assert chain["inputs"] == c.body["inputs"]["numeric"] + [
            f"zone·e{j}" for j in range(h.shape[1] - 4)]
        for layer in chain["layers"]:
            w = torch.from_numpy(c.codec.decode(t[layer["weight"]]))
            b = torch.from_numpy(c.codec.decode(t[layer["bias"]])) if layer["bias"] else 0.0
            assert (layer["in"], layer["out"]) == (w.shape[1], w.shape[0]) == (h.shape[1],
                                                                             w.shape[0])
            h = ACTS[layer["activation"]](h @ w.T + b)
        assert torch.allclose(h, c.model(c.batch), atol=1e-6)


def test_the_chain_folds_activations_and_dropouts_into_their_layers():
    layers = case("mlp-plain").body["chain"]["layers"]
    assert [(k["node"], k["in"], k["out"], k["activation"], k["dropout"]) for k in layers] == \
        [("B3", 7, 16, "gelu", 0.1), ("B6", 16, 8, "relu", 0.2), ("B8", 8, 1, None, 0.0)]
    assert case("pca-floor").body["chain"]["inputs"] == ["x0", "x1", "x2", "x3"]


R7_BUNDLE = "bundle/cicids2017-label-R7"


def test_r7_keeps_the_dense_drawing():
    """With `CIDS_ROOT`: the R7 bundle's spec and weights give D3 §5.4's chain — B1 (65 → 256,
    gelu from B2, dropout from B3), B4 (256 → 256), B7 (256 → 15, no activation) — whose
    published weights are the bundle's and recompute its network."""
    root = os.environ.get("CIDS_ROOT")
    if not root or not (Path(root) / R7_BUNDLE / "netspec.json").is_file():
        pytest.skip(f"CIDS_ROOT names no {R7_BUNDLE}/netspec.json (set it to run the R7 check)")
    bundle = Path(root) / R7_BUNDLE
    c = case("r7", json.loads((bundle / "netspec.json").read_text(encoding="utf-8")))
    saved = torch.load(bundle / "weights.pt", map_location="cpu", weights_only=True)
    weights = saved.get("state_dict", saved)
    c.model.load_state_dict(weights)
    body = c.view.network_view(c.build())
    _valid(body)
    assert [(k["node"], k["in"], k["out"], k["activation"], k["dropout"])
            for k in body["chain"]["layers"]] == \
        [("B1", 65, 256, "gelu", 0.1), ("B4", 256, 256, "gelu", 0.1), ("B7", 256, 15, None, 0.0)]
    assert body["tensors"]["B1.weight"]["encoding"] == "base64-le"
    assert np.array_equal(c.codec.decode(body["tensors"]["B1.weight"]),
                          weights["stages.0.linear.weight"].numpy())
    assert (body["task"]["link"], len(body["task"]["outputs"]["names"])) == ("softmax", 15)
    h = c.batch["x_num"]
    with torch.no_grad():
        for k in body["chain"]["layers"]:
            w = torch.from_numpy(c.codec.decode(body["tensors"][k["weight"]]))
            h = ACTS[k["activation"]](h @ w.T + torch.from_numpy(
                c.codec.decode(body["tensors"][k["bias"]])))
        assert torch.allclose(h, c.model(c.batch), atol=1e-5)


# ================================================================ sections

def test_task_sections():
    b = case("mlp-plain").body["task"]
    assert (b["link"], b["outputs"], b["positive"], b["threshold"], b["threshold_scale"],
            b["decision_rule"], b["targets"], b["default_target"]) == \
        ("sigmoid", {"kind": "score", "names": ["1"]}, "1", 0.5, "probability", "threshold",
         ["binary_logit"], "binary_logit")
    c = case("mlp-plain")
    record = {"classes": ["miss", "hit"], "positive": "hit", "threshold": 0.3,
              "threshold_scale": "probability"}
    t = c.build(output=record).task
    assert (t["outputs"]["names"], t["positive"], t["threshold"]) == (["hit"], "hit", 0.3)
    q = case("lstm-attention").body["task"]
    assert (q["kind"], q["link"], q["quantiles"], q["outputs"]["kind"]) == \
        ("forecast", "horizon", [0.1, 0.5, 0.9], "horizon")
    assert q["outputs"]["names"][:4] == ["t+1/q0.1", "t+1/q0.5", "t+1/q0.9", "t+2/q0.1"]
    o = case("bi-lstm").body["task"]
    assert (o["kind"], o["base"], o["link"], o["decision_rule"], o["outputs"]) == \
        ("sequence-regression", "ordinal", "cumulative", "cumulative_count",
         {"kind": "thresholds", "names": [">lo", ">mid", ">hi"]})
    s = case("deep-svdd").body["task"]
    assert (s["link"], s["outputs"]["kind"], s["targets"], s["threshold_scale"]) == \
        ("distance", "embedding", ["svdd_distance"], "score")
    r = case("autoencoder").body["task"]
    assert (r["link"], r["outputs"]["names"], r["positive"], r["threshold"]) == \
        ("reconstruction", ["x0", "x1", "x2", "x3", "zone"], None, None)
    m = case("rnn").body["task"]
    assert (m["base"], m["link"], m["decision_rule"], m["outputs"]["kind"], m["threshold"]) == \
        ("multilabel", "sigmoid_each", "per_label_threshold", "labels", 0.5)


def test_inputs_equal_the_contract_features():
    for name in CASES:
        c = case(name)
        feats = c.spec["data_contract"]["features"]
        inputs = c.body["inputs"]
        assert inputs["numeric"] == [f["name"] for f in feats if f["role"] != "nominal"]
        assert [n["name"] for n in inputs["nominal"]] == \
            [f["name"] for f in feats if f["role"] == "nominal"]
    ar = case("gru").body["inputs"]
    assert (ar["past_target"], ar["order"][-1], ar["axis_labels"]["feature"][-1]) == \
        ("y", "y", {"name": "y", "past": True})
    assert ar["axis_labels"]["step"] == ["t−5", "t−4", "t−3", "t−2", "t−1", "t"]
    assert ar["axis_labels"]["token"] == ar["axis_labels"]["step"]
    ft = case("ft-transformer").body["inputs"]
    assert ft["axis_labels"]["token"] == ["[cls]", "x0", "x1", "x2", "x3", "zone"]
    assert ft["nominal"] == [{"name": "zone", "cardinality": 5, "dim": 8}] and ft["width"] == 5
    plain = case("mlp-plain").body["inputs"]
    assert (plain["width"], plain["nominal"][0]["dim"], plain["axis_labels"]["step"]) == \
        (7, 3, None)


# ================================================================ refusals

def test_a_block_type_without_a_reader_is_refused(monkeypatch):
    c = case("mlp-plain")
    readers = {k: v for k, v in c.registry.READERS.items() if k != "dense"}
    monkeypatch.setattr(c.registry, "READERS", readers)
    with pytest.raises(c.readers.GraphError, match="no inspection reader for block type 'dense'"):
        c.build()


def test_a_network_that_is_not_the_specs_is_refused():
    c = case("mlp-plain")
    longer = copy.deepcopy(c.spec)
    longer["blocks"].insert(2, {"id": "X", "type": "dropout", "params": {"p": 0.5}})
    with pytest.raises(c.readers.GraphError,
                       match="the network is not the spec's: 9 blocks, 8 stages"):
        c.build(spec=longer)
    wider = copy.deepcopy(c.spec)
    wider["blocks"][2]["params"]["units"] = 32
    with pytest.raises(c.readers.GraphError, match=r"B3 \(dense\): the spec says units = 32"):
        c.build(spec=wider)
    inner = copy.deepcopy(CASES["mlp-residual"])
    inner["blocks"][3]["params"]["inner"][3]["params"]["p"] = 0.4
    r = case("mlp-residual")
    with pytest.raises(r.readers.GraphError, match="the spec says p = 0.4"):
        r.build(spec=inner)


def test_a_training_network_is_refused():
    c = case("mlp-plain")
    model = copy.deepcopy(c.model).train()
    with pytest.raises(c.readers.GraphError, match="eval mode"):
        c.build(model=model)


# ================================================================ hygiene

def _hooks(model):
    return sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules())


@pytest.mark.parametrize("name", list(CASES))
def test_building_leaves_the_model_untouched(name):
    c = case(name)
    before = {k: v.clone() for k, v in c.model.state_dict().items()}
    hooks = _hooks(c.model)
    c.build()
    assert _hooks(c.model) == hooks == 0
    assert all(p.grad is None for p in c.model.parameters())
    assert not c.model.training
    after = c.model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)


@pytest.mark.parametrize("name", ["mlp-plain", "gru", "gcn", "ft-transformer", "user-mixed"])
def test_two_builds_give_the_same_bytes_and_the_digest_verifies(name):
    c = case(name)
    again = c.view.network_view(c.build(), "full", "sha256:" + "1" * 64)
    assert c.view.canonical(again) == c.view.canonical(c.body)
    rest = {k: v for k, v in c.body.items() if k != "network_digest"}
    text = json.dumps(rest, sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert c.body["network_digest"] == "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def test_code_names_the_package_files():
    for name in ("mlp-plain", "user-mixed", "gcn", "gru"):
        c = case(name)
        for n in c.body["nodes"]:
            code = n["code"]
            if code is None:
                continue
            text = (c.root / code["file"]).read_text(encoding="utf-8")
            assert code["file"].startswith(c.package + "/") and code["anchor"] in text
    nodes = case("user-mixed").nodes()
    assert nodes["U"]["code"] == {"file": "i02_user_mixed/model/custom/gain.py",
                                  "class": "_Gain", "anchor": "class _Gain("}
    assert nodes["U"]["params"]["verified"] is False and \
        nodes["U"]["params"]["declared_out_shape"] == [4]
    assert nodes["input.num"]["code"] is None and nodes["output"]["code"] is None


STDLIB = {"__future__", "copy", "dataclasses", "hashlib", "json", "math", "pathlib", "re",
          "sys", "threading", "types", "typing"}


@pytest.mark.parametrize("file", MINE)
def test_size_discipline_state_and_imports(file):
    path = INSPECTION / file
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assert conformance.size_findings(tree, file) == [], "C11–C13 must hold at error"
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = [t.id for t in targets if isinstance(t, ast.Name)]
            if isinstance(node.value, conformance.MUTABLE_NODES) and names != ["__all__"]:
                assert all(n.lstrip("_").isupper() for n in names), f"C7: {names} in {file}"
        if isinstance(node, ast.Import):
            roots = {a.name.split(".")[0] for a in node.names}
            assert roots <= STDLIB | {"numpy", "torch"}, f"{file} imports {roots}"
        if isinstance(node, ast.ImportFrom) and not node.level:
            assert node.module.split(".")[0] in STDLIB | {"numpy", "torch"}, \
                f"{file} imports {node.module}"
        if isinstance(node, ast.ImportFrom) and node.level == 3:
            assert node.module == "domain", f"{file} reaches {node.module} beyond the domain"
    assert "{{" not in path.read_text(encoding="utf-8")


def test_the_templates_carry_no_dataset_literal():
    lint = REPO / "vega" / "scripts" / "tests" / "lint_templates.py"
    run = subprocess.run([sys.executable, str(lint), *[str(INSPECTION / f) for f in MINE]],
                         capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, run.stdout + run.stderr


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
