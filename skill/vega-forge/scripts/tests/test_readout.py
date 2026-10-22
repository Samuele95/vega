#!/usr/bin/env python3
"""
test_readout.py — the Model adapter of generation v2.3: `Network`, `NetworkBuilder` and
`TaskReadout` (WP B06).

    python3 -m pytest test_readout.py [-q]          (or: python3 test_readout.py)
    CIDS_ROOT=<dir> python3 -m pytest test_readout.py   (adds the CIC-IDS2017 R7 check)

What it proves (BUILD-PLAN §2.4 B06's closing tests; D1 §3.4, §7; R-A2, R-B1, R-B4–R-B7, R-B9,
R-C1, R-C3, R-C6):

1. **TaskReadout per effective kind**, on hand-built head outputs: `score` / `proba` /
   `reduce` / `row_score` shapes, dtypes and values for binary, multiclass, multilabel,
   regression (plain and quantile), ordinal, forecast (plain and quantile, horizon-major),
   the sequence kinds through their base, unsupervised, anomaly under reconstruction, vae and
   svdd; ordinal class probabilities ≥ 0 summing to 1 — the differences of the cumulative ones
   for a rank-consistent head, clipped and renormalised (the rows named by `clipped`) for any
   other; the link of each reading is R-C1's; `label_only` names catalog metrics only; an
   objective no task row admits and a head of the wrong width are refused by name.
2. **The builder**: `_DISPATCH` is the catalog's 33 types; every type is made by its class's
   `from_params` with the params the builder resolves (cardinalities, kind, out = recon_dim,
   the composites' modules), is called by the stage kind `blocks.STAGE_KIND_FOR` names, and
   records the per-sample output dims a forward pass produces; `num_layers` falls back to
   `layers`; `init.recurrent: orthogonal` makes every `weight_hh_*` orthogonal; the
   autoregressive channel widens the start (R-A7); the runtime twins of W2/G2 refuse a
   misplaced block by id, a sequence left for the head and node rows at level graph; a custom
   block's code is read from `<model>/custom/<stem>.py` only.
3. **The Network**: `stages/ids/kinds/shapes` read-only and aligned with `spec.blocks`; the
   node-level gather after the head; graph-level rows; an eval forward, `score` and
   `predict_proba` write no attribute anywhere (R-B4) and leave a training network training;
   `kl()` / `kl(batch)`; the SVDD centre is a pair of buffers and the score is identical after a
   `state_dict` round trip; two builds of one spec are identical.
4. **The goldens**: on the forge fixture's orders spec, `param_count()` and `block_shapes()`
   equal B01's `golden/orders-smoke.json` (`params_total`, `per_block`), and the initial
   weights and forward passes are bit-identical to the v2.2 generation's `build_model`; with
   `CIDS_ROOT`, the R7 spec's `params_total` and `per_block` equal its bundle's
   `records/architecture.json`.
5. **The size discipline**: the rendered `network.py`, `readout.py` and `builder.py` meet
   C11–C13, and `network.py`'s top level is C14's (the class, helpers, constants, the
   `build_model = Network.build` alias) with `Network(...)` called only inside `build` (C9).

The templates are rendered with `render_templates.py` (the live `templates/domain.py`,
`network.py`, `readout.py`; the live `blocks.py` and `builder.py` spliced by the emitter's own
code), into a package whose spec uses every block type, so every block file is present.
"""
from __future__ import annotations

import ast
import importlib
import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
BACKEND = SCRIPTS / "backends" / "pytorch"
TEMPLATES = BACKEND / "templates"
for _p in (str(BACKEND), str(HERE), str(SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import blocks as B  # noqa: E402
import render_templates as rt  # noqa: E402
import vega_regularization  # noqa: E402

CATALOG = json.loads((SCRIPTS / "catalog.json").read_text(encoding="utf-8"))
GOLDEN = HERE / "golden"
PKG = "b06_readout"
PKG_V22 = "b06_golden_v22"
ALL_TYPES = tuple(CATALOG["blocks"])
R7_BUNDLE = "bundle/cicids2017-label-R7"

STUB_LOSSES = '''"""A stub of the losses source: test_readout.py builds networks, never a loss."""
# --8<-- losses
def build_loss(spec):
    raise NotImplementedError("test_readout.py renders no loss")
# -->8-- losses
'''

CUSTOM_GAIN = '''"""A custom block: y = x * g, one gain per feature."""
import torch
from torch import nn


class _Gain(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(int(width)))

    def forward(self, x):
        return x * self.gain


def build(in_shape, params):
    return _Gain(in_shape[-1])
'''


# ================================================================ the rendered package

def _purge(prefix):
    for name in [m for m in sys.modules if m == prefix or m.startswith(prefix + ".")]:
        del sys.modules[name]


class Package:
    """The rendered v2.3 package, imported: its domain, readout, builder and network modules."""

    def __init__(self, root):
        self.root = Path(root)
        self.domain = importlib.import_module(f"{PKG}.domain")
        self.readout = importlib.import_module(f"{PKG}.model.readout")
        self.builder = importlib.import_module(f"{PKG}.model.builder")
        self.network = importlib.import_module(f"{PKG}.model.network")

    def spec(self, d):
        return self.domain.NetSpec.from_dict(d)

    def build(self, d, run_dir=None):
        return self.network.Network.build(self.spec(d), run_dir)

    def readout_of(self, d):
        return self.readout.TaskReadout.for_spec(self.spec(d))

    def model_file(self, name):
        return self.root / PKG / "model" / name


@pytest.fixture(scope="module")
def pkg(tmp_path_factory):
    work = tmp_path_factory.mktemp("b06-render")
    src = work / "sources"
    src.mkdir()
    shutil.copyfile(BACKEND / "blocks.py", src / "blocks.py")
    shutil.copyfile(BACKEND / "builder.py", src / "builder.py")
    (src / "losses.py").write_text(STUB_LOSSES, encoding="utf-8")
    pipeline = rt.reference_pipeline()
    pipeline["system"]["package"] = PKG
    wide = spec_of([{"id": f"W{i}", "type": t, "params": {}} for i, t in enumerate(ALL_TYPES)])
    out = work / "out"
    rec = rt.render(TEMPLATES, out, pipeline, wide,
                    only=["domain.py", "network.py", "readout.py"], splice=True,
                    sources=str(src))
    assert {"model/network.py", "model/readout.py", "model/builder.py", "model/blocks_core.py",
            "model/blocks_seq.py", "model/blocks_graph.py", "model/blocks_gen.py"} \
        <= set(rec["files"])
    custom = out / PKG / "model" / "custom"
    custom.mkdir()
    (custom / "gain.py").write_text(CUSTOM_GAIN, encoding="utf-8")
    sys.path.insert(0, str(out))
    try:
        yield Package(out)
    finally:
        sys.path.remove(str(out))
        _purge(PKG)


# ================================================================ specs and batches

def _num(name):
    return {"name": name, "dtype": "float32", "role": "numeric"}


def _nominal(name, card):
    return {"name": name, "dtype": "int64", "role": "nominal", "cardinality": card,
            "unseen_category_id": card - 1}


ROW_FEATURES = [_num(f"x{i}") for i in range(4)] + [_nominal("zone", 5)]
NUMERIC_ONLY = [_num(f"x{i}") for i in range(4)]
GRAPH_FEATURES = [_num(f"x{i}") for i in range(3)]
BINARY = {"kind": "binary", "label": "y", "classes": [0, 1], "positive": 1}
SEQ_BINARY = {"kind": "sequence-classification", "base_kind": "binary", "label": "y",
              "classes": [0, 1]}
MULTICLASS3 = {"kind": "multiclass", "label": "y", "classes": ["a", "b", "c"]}
WINDOW = {"shape": "window", "length": 6, "stride": 1, "context": "previous-part"}
GRAPH_NODE = {"shape": "graph", "edges": "shaped/edges.parquet", "level": "node",
              "protocol": "inductive", "batching": "full"}
GRAPH_LEVEL = {"shape": "graph", "edges": "shaped/edges.parquet", "level": "graph",
               "graph_id": "g", "protocol": "inductive", "batching": "graphs"}


def spec_of(blocks, *, task=None, sample=None, features=None, loss=None, init=None, seed=7):
    features = ROW_FEATURES if features is None else features
    return {
        "schema": "vega/netspec@1", "seed": seed, "task": task or dict(BINARY),
        "input": {"manifest_digest": "sha256:" + "ab" * 32},
        "data_contract": {"input_dim": len(features), "features": features,
                          "sample": sample or {"shape": "row"}},
        "family": {"name": "user-defined"}, "blocks": blocks,
        "loss": loss or {"name": "bce_with_logits", "params": {}},
        "metrics": {"primary": "pr_auc"}, "optimizer": {"name": "adamw"},
        "training": {"max_epochs": 1, "batch_size": 8},
        "init": init or {"scheme": "kaiming_uniform", "bias": "zeros"}}


def _gen(seed):
    return torch.Generator().manual_seed(seed)


def row_batch(n=8, numeric=4, cards=(5,), seed=1):
    g = _gen(seed)
    batch = {"x_num": torch.randn(n, numeric, generator=g)}
    if cards:
        batch["x_cat"] = torch.stack([torch.randint(0, c, (n,), generator=g) for c in cards], -1)
    return batch


def window_batch(n=4, length=6, numeric=4, cards=(5,), seed=2):
    g = _gen(seed)
    batch = {"x_num": torch.randn(n, length, numeric, generator=g)}
    if cards:
        batch["x_cat"] = torch.stack(
            [torch.randint(0, c, (n, length), generator=g) for c in cards], -1)
    return batch


# a 10-node graph of three components (0-3, 4-6, 7-9): a graph-level batch of three graphs,
# or one node-level graph whose targets are 0, 2 and 5
EDGES = [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2), (4, 5), (5, 4), (5, 6), (6, 5),
         (7, 8), (8, 7), (8, 9), (9, 8), (3, 0)]


def graph_batch(level="node", seed=3):
    g = _gen(seed)
    batch = {"x_num": torch.randn(10, 3, generator=g),
             "edge_index": torch.tensor(EDGES, dtype=torch.long).t().contiguous()}
    if level == "node":
        batch["target_index"] = torch.tensor([0, 2, 5])
    else:
        batch["graph_index"] = torch.tensor([0, 0, 0, 0, 1, 1, 1, 2, 2, 2])
        batch["num_graphs"] = 3
    return batch


# ================================================================ 1. TaskReadout

def _out(*shape, seed=11):
    return torch.randn(*shape, generator=_gen(seed))


def _readout(pkg, task, loss=None, sample=None, features=None, head=1):
    blocks = [{"id": "H", "type": "head", "params": {"out": head}}]
    return pkg.readout_of(spec_of(blocks, task=task, loss=loss, sample=sample,
                                  features=features))


def _check_reduce(r, out, batch=None, model=None):
    red = r.reduce(out, batch, model)
    assert tuple(red.shape) == (out.shape[0],) + tuple(r.reduced_shape())
    assert r.label_only() <= set(CATALOG["metric_direction"]), r.label_only()
    return red


LINKS = {"binary": "sigmoid", "multiclass": "softmax", "multilabel": "sigmoid_each",
         "ordinal": "cumulative", "regression": "identity", "forecast": "horizon",
         "reconstruction": "reconstruction", "anomaly-reconstruction": "reconstruction",
         "anomaly-svdd": "distance"}


def test_the_links_are_the_catalogs(pkg):
    """R-C1: each reading's link is in `inspection.links` and names that reading (the
    unsupervised task's reading is `reconstruction`)."""
    cases = {
        "binary": _readout(pkg, BINARY),
        "multiclass": _readout(pkg, MULTICLASS3, {"name": "cross_entropy"}),
        "multilabel": _readout(pkg, {"kind": "multilabel", "label": "t",
                                     "classes": ["a", "b", "c"]}),
        "ordinal": _readout(pkg, {"kind": "ordinal", "label": "y", "classes": [0, 1, 2, 3]},
                            {"name": "cumulative_link"}),
        "regression": _readout(pkg, {"kind": "regression", "label": "y"}, {"name": "mse"}),
        "forecast": _readout(pkg, {"kind": "forecast", "label": "y"}, {"name": "mse"},
                             dict(WINDOW, horizon=3), NUMERIC_ONLY),
        "reconstruction": _readout(pkg, {"kind": "unsupervised"}, {"name": "reconstruction"}),
        "anomaly-reconstruction": _readout(pkg, {"kind": "anomaly"},
                                           {"name": "reconstruction"}),
        "anomaly-svdd": _readout(pkg, {"kind": "anomaly"}, {"name": "svdd"}),
    }
    links = CATALOG["inspection"]["links"]
    for reading, r in cases.items():
        assert r.link == LINKS[reading], reading
        assert r.link in links and reading in links[r.link]["readings"], reading


def test_binary_scores_are_logits_always(pkg):
    r = _readout(pkg, BINARY)
    out = torch.tensor([[-2.0], [0.0], [0.3], [1e-9], [0.9]])     # some logits inside [0, 1]
    assert torch.equal(r.score(out), out[:, 0])
    assert torch.allclose(r.proba(out), torch.sigmoid(out[:, 0]))
    red = _check_reduce(r, out)
    assert red.dtype == torch.int32 and red.tolist() == [0, 0, 1, 1, 1]
    assert r.row_score(out) is None and r.kind.value == "binary"
    assert {"f1", "accuracy", "mcc"} <= r.label_only() and "roc_auc" not in r.label_only()


def test_multiclass_and_the_sequence_kind_over_it(pkg):
    out = _out(6, 3)
    seq = {"kind": "sequence-classification", "base_kind": "multiclass", "label": "y",
           "classes": ["a", "b", "c"]}
    for task, sample, feats in ((MULTICLASS3, None, None), (seq, WINDOW, NUMERIC_ONLY)):
        r = _readout(pkg, task, {"name": "cross_entropy"}, sample, feats, head=3)
        assert r.kind.value == "multiclass" and r.link == "softmax"
        assert torch.equal(r.score(out), out)
        p = r.proba(out)
        assert torch.allclose(p.sum(-1), torch.ones(6)) and (p >= 0).all()
        red = _check_reduce(r, out)
        assert red.dtype == torch.int32 and torch.equal(red, out.argmax(-1).to(torch.int32))


def test_multilabel_reads_each_label(pkg):
    r = _readout(pkg, {"kind": "multilabel", "label": "t", "classes": ["a", "b", "c", "d"]},
                 head=4)
    out = _out(5, 4)
    assert torch.equal(r.score(out), out)
    assert torch.allclose(r.proba(out), torch.sigmoid(out))
    red = _check_reduce(r, out)
    assert r.reduced_shape() == (4,) and red.dtype == torch.int8
    assert torch.equal(red, (out > 0).to(torch.int8))


def test_regression_plain_and_quantile(pkg):
    reg = {"kind": "regression", "label": "y"}
    plain = _readout(pkg, reg, {"name": "mse"})
    out = _out(7, 1)
    assert torch.equal(plain.score(out), out[:, 0]) and torch.equal(plain.proba(out), out[:, 0])
    assert _check_reduce(plain, out).dtype == torch.float32
    q = _readout(pkg, reg, {"name": "quantile", "params": {"quantiles": [0.1, 0.5, 0.9]}},
                 head=3)
    out = _out(7, 3)
    assert torch.equal(q.score(out), out[:, 1])                 # the q = 0.5 column
    assert {"rmse", "mae", "r2"} <= q.label_only()


def test_ordinal_probabilities_are_a_distribution(pkg):
    task = {"kind": "ordinal", "label": "y", "classes": [0, 1, 2, 3, 4]}
    r = _readout(pkg, task, {"name": "cumulative_link"}, head=4)
    ordered = torch.sort(_out(9, 4), dim=-1, descending=True).values     # z_0 >= z_1 >= ...
    p = r.proba(ordered)
    s = torch.sigmoid(ordered)
    expected = torch.cat([1 - s[:, :1], s[:, :-1] - s[:, 1:], s[:, -1:]], dim=-1)
    assert p.shape == (9, 5) and torch.allclose(p, expected)
    assert (p >= 0).all() and torch.allclose(p.sum(-1), torch.ones(9))
    assert not r.clipped(ordered).any()
    # a head that is not rank-consistent: clipped at 0 and renormalised (bundle-contract §4),
    # and the rows that needed it named for the serving warning
    unordered = _out(9, 4, seed=5)
    s = torch.sigmoid(unordered)
    raw = torch.cat([1 - s[:, :1], s[:, :-1] - s[:, 1:], s[:, -1:]], dim=-1)
    p = r.proba(unordered)
    assert (p >= 0).all() and torch.allclose(p.sum(-1), torch.ones(9), atol=1e-6)
    clipped = raw.clamp(min=0)
    assert torch.allclose(p, clipped / clipped.sum(-1, keepdim=True), atol=1e-6)
    assert torch.equal(r.clipped(unordered), (raw < 0).any(-1)) and r.clipped(unordered).any()
    assert _readout(pkg, BINARY).clipped(_out(3, 1)) is None
    red = _check_reduce(r, ordered)
    assert red.dtype == torch.int32
    assert torch.equal(red, (torch.sigmoid(ordered) >= 0.5).sum(-1).to(torch.int32))  # R-C6
    assert torch.equal(r.score(ordered), ordered)
    seq = {"kind": "sequence-regression", "base_kind": "ordinal", "label": "y",
           "classes": [0, 1, 2, 3, 4]}
    assert _readout(pkg, seq, {"name": "cumulative_link"}, WINDOW, NUMERIC_ONLY,
                    head=4).link == "cumulative"


def test_forecast_plain_and_quantile_horizon_major(pkg):
    task = {"kind": "forecast", "label": "y"}
    sample = dict(WINDOW, horizon=3)
    plain = _readout(pkg, task, {"name": "mse"}, sample, NUMERIC_ONLY, head=3)
    out = _out(4, 3)
    assert torch.equal(plain.score(out), out) and plain.reduced_shape() == (3,)
    assert _check_reduce(plain, out).dtype == torch.float32
    q = _readout(pkg, task, {"name": "quantile", "params": {"quantiles": [0.25, 0.5, 0.75]}},
                 sample, NUMERIC_ONLY, head=9)
    out = _out(4, 9)
    got = q.score(out)
    assert got.shape == (4, 3)
    for h in range(3):
        assert torch.equal(got[:, h], out[:, h * 3 + 1])         # step h, quantile 0.5
    assert _check_reduce(q, out).shape == (4, 3)


def _recon_by_hand(out, batch, n_num=4, cards=(5,)):
    y = out.detach().numpy().astype(np.float64)
    x = batch["x_num"].numpy().astype(np.float64)
    mse = ((y[:, :n_num] - x[:, :n_num]) ** 2).mean(axis=1)
    ces, at = [], n_num
    for i, card in enumerate(cards):
        logits = y[:, at:at + card]
        logp = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
        ces.append(-logp[np.arange(len(y)), batch["x_cat"][:, i].numpy()])
        at += card
    return mse, mse + np.mean(ces, axis=0)


def test_reconstruction_readings(pkg):
    """recon_dim = 4 numeric + 5 zone logits; the anomaly score is numeric MSE + the mean
    nominal cross-entropy (R-C2), the unsupervised score the numeric reconstruction."""
    batch = row_batch(6)
    out = _out(6, 9)
    mse, score = _recon_by_hand(out, batch)
    uns = _readout(pkg, {"kind": "unsupervised"}, {"name": "reconstruction"}, head=9)
    assert torch.equal(uns.score(out, batch), out[:, :4])
    assert np.allclose(uns.reduce(out, batch).numpy(), mse, atol=1e-6)
    assert np.allclose(uns.row_score(out, batch).numpy(), score, atol=1e-5)
    assert uns.label_only() == {"reconstruction_error"}
    for loss in ("reconstruction", "vae"):
        an = _readout(pkg, {"kind": "anomaly"}, {"name": loss}, head=9)
        assert an.objective.value == loss and an.link == "reconstruction"
        got = an.score(out, batch)
        assert got.shape == (6,) and np.allclose(got.numpy(), score, atol=1e-5)
        assert torch.equal(an.proba(out, batch), got) and torch.equal(an.row_score(out, batch),
                                                                       got)
        assert _check_reduce(an, out, batch).shape == (6,)
    with pytest.raises(ValueError, match="x_cat"):
        an.score(out, {"x_num": batch["x_num"]})
    numeric = _readout(pkg, {"kind": "anomaly"}, {"name": "reconstruction"}, None,
                       NUMERIC_ONLY, head=4)
    only = numeric.score(out[:, :4], {"x_num": batch["x_num"]})
    assert np.allclose(only.numpy(), mse, atol=1e-6)


class _Centred:
    def __init__(self, centre):
        self.centre = centre

    def objective_center(self):
        return self.centre


def test_svdd_reads_the_models_centre(pkg):
    r = _readout(pkg, {"kind": "anomaly"}, {"name": "svdd"}, head=3)
    out, centre = _out(5, 3), torch.tensor([0.5, -0.1, 0.2])
    got = r.score(out, None, _Centred(centre))
    assert torch.allclose(got, ((out - centre) ** 2).sum(-1))
    assert torch.equal(r.reduce(out, None, _Centred(centre)), got)
    with pytest.raises(ValueError, match="objective_center"):
        r.score(out)


def test_refusals_name_the_reading(pkg):
    with pytest.raises(ValueError, match="no reading for a unsupervised task trained with the "
                                         "svdd objective"):
        _readout(pkg, {"kind": "unsupervised"}, {"name": "svdd"})
    r = _readout(pkg, MULTICLASS3, {"name": "cross_entropy"}, head=3)
    with pytest.raises(ValueError, match="outputs 2 value.*multiclass reading.*reads 3"):
        r.check_head(2)
    r.check_head(3)
    with pytest.raises(ValueError, match="reads 3"):
        pkg.build(spec_of([{"id": "H", "type": "head", "params": {"out": 2}}],
                          task=MULTICLASS3, loss={"name": "cross_entropy"}))


# ================================================================ 2. the builder

def test_the_dispatch_is_the_catalogs_33_types(pkg):
    """The builder's own table and its composites part's hold the methods themselves (the T5
    round 2 improve pass: a table of method names resolved by getattr turned a typo into a
    runtime AttributeError), and between them cover the catalog's 33 types once."""
    own = pkg.builder.NetworkBuilder._DISPATCH
    part = pkg.builder._CompositeBlocks._COMPOSITE
    assert not set(own) & set(part)
    assert set(own) | set(part) == set(ALL_TYPES) and len(ALL_TYPES) == 33
    assert set(B.CLASS_FOR) | {"custom"} == set(ALL_TYPES)
    # one record per type (T5 round 3): its class — the catalog's, the class blocks.py names
    # for it — and its method, the method itself
    for t, (cls, method) in own.items():
        assert callable(method) and method is vars(pkg.builder.NetworkBuilder)[method.__name__], t
        assert cls.__name__ == B.CLASS_FOR[t], t
    for t, (cls, method) in part.items():
        assert callable(method) and method is vars(pkg.builder._CompositeBlocks)[method.__name__], t
        assert cls is None if t == "custom" else cls.__name__ == B.CLASS_FOR[t], t


def _t(t, params=None):
    return {"id": "T", "type": t, "params": params or {}}


def _head(out=1):
    return {"id": "H", "type": "head", "params": {"out": out}}


_UNSUP = {"kind": "unsupervised"}
_RECON = {"name": "reconstruction"}

# type -> (blocks, spec options, batch); the block under test has id "T"
TYPE_CASES = {
    "dense": ([_t("dense", {"units": 8}), _head()], {}, "row"),
    "embedding": ([_t("embedding", {"columns": ["zone"], "dim": 3}),
                   {"id": "C", "type": "concat"}, _head()], {}, "row"),
    "concat": ([{"id": "E", "type": "embedding", "params": {"columns": ["zone"],
                                                            "dims": {"zone": 2}}},
                _t("concat"), _head()], {}, "row"),
    "batchnorm": ([_t("batchnorm"), _head()], {}, "row"),
    "layernorm": ([_t("layernorm"), _head()], {}, "row"),
    "dropout": ([_t("dropout", {"p": 0.3}), _head()], {}, "row"),
    "activation": ([_t("activation", {"name": "gelu"}), _head()], {}, "row"),
    "flatten": ([_t("flatten"), _head()], {"sample": WINDOW, "task": SEQ_BINARY}, "window"),
    "reshape": ([{"id": "D", "type": "dense", "params": {"units": 6}},
                 _t("reshape", {"shape": [2, 3]}), {"id": "F", "type": "flatten"}, _head()],
                {}, "row"),
    "pool": ([{"id": "K", "type": "feature_tokenizer", "params": {"d": 8}},
              _t("pool", {"kind": "cls"}), _head()], {}, "row"),
    "sequential": ([_t("sequential", {"inner": [{"type": "dense", "params": {"units": 8}},
                                                {"type": "activation",
                                                 "params": {"name": "relu"}}]}), _head()],
                   {}, "row"),
    "residual": ([_t("residual", {"repeat": 2, "projection": "auto",
                                  "inner": [{"type": "dense", "params": {"units": 8}}]}),
                  _head()], {}, "row"),
    "parallel": ([_t("parallel", {"merge": "concat",
                                  "inner": [{"type": "dense", "params": {"units": 3}},
                                            {"type": "dense", "params": {"units": 5}}]}),
                  _head()], {}, "row"),
    "conv1d": ([_t("conv1d", {"channels": 4, "kernel": 3, "stride": 2}),
                {"id": "F", "type": "flatten"}, _head()],
               {"sample": WINDOW, "task": SEQ_BINARY}, "window"),
    "pool1d": ([_t("pool1d", {"kind": "max", "size": 2}), {"id": "F", "type": "flatten"},
                _head()], {"sample": WINDOW, "task": SEQ_BINARY}, "window"),
    "rnn": ([_t("rnn", {"hidden": 5, "nonlinearity": "relu"}), _head()],
            {"sample": WINDOW, "task": SEQ_BINARY}, "window"),
    "gru": ([_t("gru", {"hidden": 5, "return": "mean"}), _head()],
            {"sample": WINDOW, "task": SEQ_BINARY}, "window"),
    "lstm": ([_t("lstm", {"hidden": 5, "num_layers": 2, "bidirectional": True}), _head()],
             {"sample": WINDOW, "task": SEQ_BINARY}, "window"),
    "attention": ([_t("attention", {"heads": 2, "dim": 4, "pool": True}), _head()],
                  {"sample": WINDOW, "task": SEQ_BINARY}, "window"),
    "attention_pool": ([{"id": "R", "type": "gru", "params": {"hidden": 6, "return": "all"}},
                        _t("attention_pool", {"hidden": 4}), _head()],
                       {"sample": WINDOW, "task": SEQ_BINARY}, "window"),
    "transformer_encoder": ([{"id": "K", "type": "feature_tokenizer", "params": {"d": 8}},
                             _t("transformer_encoder", {"d": 8, "heads": 2, "layers": 1}),
                             {"id": "P", "type": "pool", "params": {"kind": "mean"}}, _head()],
                            {}, "row"),
    "positional": ([_t("positional", {"kind": "sinusoidal"}), {"id": "F", "type": "flatten"},
                    _head()], {"sample": WINDOW, "task": SEQ_BINARY}, "window"),
    "feature_tokenizer": ([_t("feature_tokenizer", {"d": 8}),
                           {"id": "P", "type": "pool", "params": {"kind": "cls"}}, _head()],
                          {}, "row"),
    "head": ([_t("head", {"out": 1})], {}, "row"),
    "decoder": ([{"id": "Z", "type": "dense", "params": {"units": 3}},
                 _t("decoder", {"widths": [6]}), _head(9)],
                {"task": _UNSUP, "loss": _RECON}, "row"),
    "reparameterize": ([{"id": "Z", "type": "dense", "params": {"units": 6}},
                        _t("reparameterize", {"z": 3}),
                        {"id": "D", "type": "decoder", "params": {"widths": [5]}}, _head(9)],
                       {"task": _UNSUP, "loss": {"name": "vae"}}, "row"),
    "gcn_conv": ([_t("gcn_conv", {"out": 4, "normalize": "rw"}), _head(3)],
                 {"sample": GRAPH_NODE, "task": MULTICLASS3, "features": GRAPH_FEATURES,
                  "loss": {"name": "cross_entropy"}}, "node"),
    "sage_conv": ([_t("sage_conv", {"out": 4, "aggregator": "max"}), _head(3)],
                  {"sample": GRAPH_NODE, "task": MULTICLASS3, "features": GRAPH_FEATURES,
                   "loss": {"name": "cross_entropy"}}, "node"),
    "gat_conv": ([_t("gat_conv", {"out": 4, "heads": 2, "concat": True}), _head(3)],
                 {"sample": GRAPH_NODE, "task": MULTICLASS3, "features": GRAPH_FEATURES,
                  "loss": {"name": "cross_entropy"}}, "node"),
    "gin_conv": ([_t("gin_conv", {"out": 4}), _head(3)],
                 {"sample": GRAPH_NODE, "task": MULTICLASS3, "features": GRAPH_FEATURES,
                  "loss": {"name": "cross_entropy"}}, "node"),
    "graph_pool": ([{"id": "G1", "type": "gin_conv", "params": {"out": 4}},
                    _t("graph_pool", {"readout": "sum"}), _head()],
                   {"sample": GRAPH_LEVEL, "features": GRAPH_FEATURES}, "graph"),
    "edge_dropout": ([_t("edge_dropout", {"p": 0.2}),
                      {"id": "G1", "type": "gcn_conv", "params": {"out": 4}}, _head(3)],
                     {"sample": GRAPH_NODE, "task": MULTICLASS3, "features": GRAPH_FEATURES,
                      "loss": {"name": "cross_entropy"}}, "node"),
    "custom": ([_t("custom", {"code_ref": "custom/gain.py", "out_shape": [4]}), _head()],
               {"features": NUMERIC_ONLY}, "row"),
}

BATCHES = {"row": row_batch, "window": window_batch,
           "node": lambda: graph_batch("node"), "graph": lambda: graph_batch("graph")}


def test_every_type_has_a_case():
    assert set(TYPE_CASES) == set(ALL_TYPES)


@pytest.mark.parametrize("block_type", ALL_TYPES)
def test_every_type_builds_through_from_params(pkg, block_type):
    """The class `blocks.CLASS_FOR` names, the stage kind `STAGE_KIND_FOR` names, and the
    recorded per-sample dims equal to what the forward pass produces."""
    blocks, opts, batch_kind = TYPE_CASES[block_type]
    net = pkg.build(spec_of(blocks, **opts))
    i = net.ids.index("T")
    module = net.stages[i]
    if block_type == "custom":
        assert type(module).__name__ == "_Gain"
    else:
        assert type(module).__name__ == B.CLASS_FOR[block_type], block_type
    assert net.kinds[i].value == B.STAGE_KIND_FOR[block_type]
    batch = BATCHES[batch_kind]()
    rows = net.block_shapes(batch)
    assert [r["id"] for r in rows] == list(net.ids)
    for row, dims in zip(rows, net.shapes):
        assert row["out_shape"] == list(dims), (block_type, row, dims)
    out = net(batch)
    assert out.shape[0] == {"row": 8, "window": 4, "node": 3, "graph": 3}[batch_kind]


def test_the_builder_resolves_params_from_the_contract(pkg):
    net = pkg.build(spec_of([
        {"id": "E", "type": "embedding", "params": {"columns": ["zone"], "dim": 3}},
        {"id": "C", "type": "concat"}, {"id": "Z", "type": "dense", "params": {"units": 3}},
        {"id": "D", "type": "decoder", "params": {"widths": [6]}}, _head(9)],
        task=_UNSUP, loss=_RECON))
    emb, dec = net.stages[0], net.stages[3]
    assert emb.embeddings[0].num_embeddings == 5 and net.shapes[:2] == ((3,), (7,))
    assert dec.net[-1].out_features == 9                  # recon_dim = 4 numeric + 5 logits
    tok = pkg.build(spec_of([{"id": "K", "type": "feature_tokenizer", "params": {"d": 8}},
                             {"id": "P", "type": "pool", "params": {"kind": "cls"}}, _head()]))
    assert tok.stages[0].cardinalities == (5,) and tok.shapes[0] == (6, 8)
    ordinal = pkg.build(spec_of([_head(3)], task={"kind": "ordinal", "label": "y",
                                                   "classes": [0, 1, 2, 3]},
                                loss={"name": "cumulative_link"}))
    assert type(ordinal.stages[0]).__name__ == "CumulativeHead"


def test_recurrent_num_layers_falls_back_to_layers(pkg):
    def layers(params):
        spec = spec_of([_t("gru", params), _head()], sample=WINDOW, task=SEQ_BINARY)
        return pkg.build(spec).stages[0].rnn.num_layers
    assert layers({"hidden": 4}) == 1
    assert layers({"hidden": 4, "layers": 2}) == 2
    assert layers({"hidden": 4, "layers": 2, "num_layers": 3}) == 3


def test_init_recurrent_orthogonal(pkg):
    def weights(init):
        spec = spec_of([_t("gru", {"hidden": 6}), _head()], sample=WINDOW, task=SEQ_BINARY,
                       init=init)
        rnn = pkg.build(spec).stages[0].rnn
        return rnn.weight_hh_l0.detach()
    ortho = weights({"scheme": "kaiming_uniform", "recurrent": "orthogonal"})
    assert ortho.shape == (18, 6)
    assert torch.allclose(ortho.t() @ ortho, torch.eye(6), atol=1e-5)
    default = weights({"scheme": "kaiming_uniform"})
    assert not torch.allclose(default.t() @ default, torch.eye(6), atol=1e-2)
    with pytest.raises(ValueError, match="recurrent 'glorot'"):
        weights({"scheme": "kaiming_uniform", "recurrent": "glorot"})


@pytest.mark.parametrize("blocks, opts, message", [
    ([_t("gcn_conv", {"out": 4}), _head()], {}, r"block T \(gcn_conv\) needs a graph sample"),
    ([_t("gru", {"hidden": 4}), _head()], {}, r"block T \(gru\) reads a sequence"),
    ([{"id": "G1", "type": "gcn_conv", "params": {"out": 4}}, _t("graph_pool"), _head(3)],
     {"sample": GRAPH_NODE, "task": MULTICLASS3, "features": GRAPH_FEATURES,
      "loss": {"name": "cross_entropy"}}, r"block T \(graph_pool\) is allowed only at "
                                          r"sample.level graph"),
    ([{"id": "G1", "type": "gin_conv", "params": {"out": 4}}, {"id": "P", "type": "graph_pool"},
      _t("gin_conv", {"out": 4}), _head()],
     {"sample": GRAPH_LEVEL, "features": GRAPH_FEATURES}, r"block T \(gin_conv\) comes after "
                                                         r"graph_pool"),
    ([{"id": "E", "type": "embedding", "params": {"columns": ["zone"]}},
      _t("feature_tokenizer", {"d": 4}), {"id": "P", "type": "pool"}, _head()], {},
     r"block T \(feature_tokenizer\) reads the nominals itself"),
    ([_t("embedding", {"columns": ["zip"]}), {"id": "C", "type": "concat"}, _head()], {},
     r"nominal column 'zip' has no cardinality"),
    ([_t("convolution"), _head()], {}, r"block T: this package carries no 'convolution' block"),
    ([_t("residual", {"inner": [{"type": "dense", "params": {"units": 8}}]}), _head()], {},
     r"residual block T changes width 4 -> 8 and projection is not 'auto'"),
    ([_t("sequential", {"inner": [{"type": "embedding", "params": {"columns": ["zone"]}}]}),
      _head()], {}, r"composite block T holds a embedding block"),
    ([_t("parallel", {"merge": "sum", "inner": [{"type": "dense", "params": {"units": 3}},
                                                {"type": "dense", "params": {"units": 5}}]}),
      _head()], {}, r"parallel block T sums branches of widths \[3, 5\]"),
    ([_t("gru", {"hidden": 4, "return": "all"}), _head()],
     {"sample": WINDOW, "task": SEQ_BINARY}, r"the last block H outputs a sequence \(6, 1\)"),
    ([{"id": "G1", "type": "gin_conv", "params": {"out": 4}}, _head()],
     {"sample": GRAPH_LEVEL, "features": GRAPH_FEATURES},
     r"at sample.level graph .* a graph_pool block must pool the node rows"),
])
def test_the_builder_refuses_a_misplaced_block_by_id(pkg, blocks, opts, message):
    with pytest.raises(ValueError, match=message):
        pkg.build(spec_of(blocks, **opts))


def test_the_autoregressive_channel_widens_the_start(pkg):
    """R-A7: a forecast's x_num carries the target's past as one more channel (F' = F + 1)."""
    task = {"kind": "forecast", "label": "y"}

    def width(sample):
        spec = spec_of([_t("gru", {"hidden": 4}), _head(3)], task=task, sample=sample,
                       features=NUMERIC_ONLY, loss={"name": "mse"})
        net = pkg.build(spec)
        assert net(window_batch(numeric=net.stages[0].rnn.input_size, cards=())).shape == (4, 3)
        return net.stages[0].rnn.input_size
    assert width(dict(WINDOW, horizon=3)) == 5                  # the default for forecast
    assert width(dict(WINDOW, horizon=3, autoregressive=False)) == 4


def test_custom_code_is_read_from_the_package_only(pkg, tmp_path):
    spec = spec_of([_t("custom", {"code_ref": "custom/absent.py", "out_shape": [4]}), _head()],
                   features=NUMERIC_ONLY)
    (tmp_path / "custom").mkdir()
    (tmp_path / "custom" / "absent.py").write_text(CUSTOM_GAIN, encoding="utf-8")
    expected = str(pkg.model_file("custom") / "absent.py")
    with pytest.raises(ValueError, match="re-emit the package") as info:
        pkg.build(spec, run_dir=str(tmp_path))            # a run directory's copy is not read
    assert expected in str(info.value)


# ================================================================ 3. the Network

MLP_RESIDUAL = [
    {"id": "B1", "type": "embedding", "params": {"columns": ["zone"], "dims": {"zone": 3}}},
    {"id": "B2", "type": "concat", "params": {}},
    {"id": "B3", "type": "dense", "params": {"units": 8}},
    {"id": "B4", "type": "residual", "params": {"repeat": 2, "inner": [
        {"type": "dense", "params": {"units": 8}}, {"type": "batchnorm", "params": {}},
        {"type": "activation", "params": {"name": "gelu"}},
        {"type": "dropout", "params": {"p": 0.2}}]}},
    {"id": "B5", "type": "head", "params": {"out": 1}}]


def test_the_structure_is_read_only_and_aligned(pkg):
    spec = spec_of(MLP_RESIDUAL)
    net = pkg.build(spec)
    assert net.ids == ("B1", "B2", "B3", "B4", "B5")
    assert net.types == ("embedding", "concat", "dense", "residual", "head")
    assert [k.value for k in net.kinds] == ["emb", "concat", "tensor", "tensor", "tensor"]
    assert net.shapes == ((3,), (7,), (8,), (8,), (1,))
    assert isinstance(net.stages, nn.ModuleList) and len(net.stages) == 5
    assert net.get_submodule("stages.3") is net.stages[3]
    for name in ("ids", "kinds", "shapes", "types"):
        with pytest.raises(AttributeError):
            setattr(net, name, ())
    with pytest.raises(AttributeError, match="read-only"):
        net.stages = nn.ModuleList()
    assert net.spec_digest == pkg.spec(spec).digest
    assert net.task.effective.value == "binary" and net.readout.link == "sigmoid"
    assert net.sample.shape.value == "row" and net.objective_center() is None
    assert not any(hasattr(net, a) for a in ("describe", "trace"))          # R-B5


def test_node_level_outputs_are_gathered_after_the_head(pkg):
    spec = spec_of([{"id": "G1", "type": "gcn_conv", "params": {"out": 4}},
                    {"id": "A", "type": "activation", "params": {"name": "relu"}}, _head(3)],
                   sample=GRAPH_NODE, task=MULTICLASS3, features=GRAPH_FEATURES,
                   loss={"name": "cross_entropy"})
    net = pkg.build(spec).eval()
    batch = graph_batch("node")
    every = dict(batch)
    every.pop("target_index")
    with torch.no_grad():
        all_nodes = net(every)
        targets = net(batch)
    assert all_nodes.shape == (10, 3) and targets.shape == (3, 3)
    assert torch.equal(targets, all_nodes[batch["target_index"]])
    assert torch.equal(net.score(batch), targets)
    assert torch.allclose(net.predict_proba(batch), torch.softmax(targets, -1))


def test_graph_level_outputs_one_row_per_graph(pkg):
    spec = spec_of([{"id": "D", "type": "edge_dropout", "params": {"p": 0.5}},
                    {"id": "G1", "type": "gin_conv", "params": {"out": 4}},
                    {"id": "P", "type": "graph_pool", "params": {"readout": "mean"}}, _head()],
                   sample=GRAPH_LEVEL, features=GRAPH_FEATURES)
    net = pkg.build(spec)
    batch = graph_batch("graph")
    assert net(batch).shape == (3, 1) and net.score(batch).shape == (3,)
    net.eval()
    with torch.no_grad():
        a, b = net(batch), net(batch)
    assert torch.equal(a, b)                             # edge_dropout is off at eval


class _SetattrSpy:
    """Every nn.Module attribute write (the Network's own `__setattr__` reaches it too)."""

    def __enter__(self):
        self.writes, self._orig = [], nn.Module.__setattr__
        spy = self

        def record(module, name, value):
            spy.writes.append((type(module).__name__, name))
            spy._orig(module, name, value)
        nn.Module.__setattr__ = record
        return self

    def __exit__(self, *exc):
        nn.Module.__setattr__ = self._orig
        return False


# R-B4's concurrency cells: mlp-residual, lstm, lstm-attention, gat
R_B4_CELLS = {
    "mlp-residual": (MLP_RESIDUAL, {}, "row"),
    "lstm": ([{"id": "E", "type": "embedding", "params": {"columns": ["zone"], "dim": 2}},
              {"id": "C", "type": "concat"},
              {"id": "R", "type": "lstm", "params": {"hidden": 5, "bidirectional": True}},
              _head()], {"sample": WINDOW, "task": SEQ_BINARY}, "window"),
    "lstm-attention": ([{"id": "R", "type": "lstm", "params": {"hidden": 5, "return": "all"}},
                        {"id": "A", "type": "attention_pool", "params": {"hidden": 4}},
                        _head()], {"sample": WINDOW, "task": SEQ_BINARY}, "window"),
    "gat": ([{"id": "D", "type": "edge_dropout", "params": {"p": 0.2}},
             {"id": "G1", "type": "gat_conv", "params": {"out": 4, "heads": 2,
                                                         "attn_dropout": 0.5}},
             _head(3)], {"sample": GRAPH_NODE, "task": MULTICLASS3, "features": GRAPH_FEATURES,
                         "loss": {"name": "cross_entropy"}}, "node"),
}


@pytest.mark.parametrize("cell", sorted(R_B4_CELLS))
def test_an_eval_pass_writes_nothing(pkg, cell):
    blocks, opts, batch_kind = R_B4_CELLS[cell]
    net = pkg.build(spec_of(blocks, **opts)).eval()
    before = {k: v.clone() for k, v in net.state_dict().items()}
    batch = BATCHES[batch_kind]()
    with _SetattrSpy() as spy:
        with torch.no_grad():
            net(batch)
        net.score(batch)
        net.predict_proba(batch)
        net.block_shapes(batch)
    assert spy.writes == [], spy.writes
    assert not net.training
    after = net.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_scoring_a_training_network_restores_training(pkg):
    net = pkg.build(spec_of(MLP_RESIDUAL))
    assert net.training
    batch = row_batch(16)
    torch.manual_seed(0)
    first = net.predict_proba(batch)
    torch.manual_seed(1)
    second = net.predict_proba(batch)
    assert torch.equal(first, second)                    # eval mode: dropout off
    assert net.training and all(m.training for m in net.modules())
    assert torch.allclose(first, torch.sigmoid(net.score(batch)))
    assert first.shape == (16,) and (first >= 0).all() and (first <= 1).all()


def test_predict_proba_per_task_fixes_e10(pkg):
    """E10: an ordinal's class probabilities, a sequence-classification over binary's σ, and an
    anomaly's score — never the raw head."""
    ordinal = pkg.build(spec_of([_head(3)], task={"kind": "ordinal", "label": "y",
                                                  "classes": [0, 1, 2, 3]},
                                loss={"name": "cumulative_link"}))
    p = ordinal.predict_proba(row_batch())
    assert p.shape == (8, 4) and (p >= 0).all() and torch.allclose(p.sum(-1), torch.ones(8))
    seq = pkg.build(spec_of([_t("gru", {"hidden": 4}), _head()], sample=WINDOW,
                            task=SEQ_BINARY))
    batch = window_batch()
    assert torch.allclose(seq.predict_proba(batch), torch.sigmoid(seq.score(batch)))
    anomaly = pkg.build(spec_of([{"id": "Z", "type": "dense", "params": {"units": 3}},
                                 _head(9)], task={"kind": "anomaly"}, loss=_RECON))
    batch = row_batch()
    assert anomaly.predict_proba(batch).shape == (8,)
    assert torch.equal(anomaly.predict_proba(batch), anomaly.score(batch))


def test_kl_of_the_training_pass_and_of_a_batch(pkg):
    spec = spec_of([{"id": "Z", "type": "dense", "params": {"units": 6}},
                    {"id": "R", "type": "reparameterize", "params": {"z": 3}},
                    {"id": "D", "type": "decoder", "params": {"widths": [5]}}, _head(9)],
                   task=_UNSUP, loss={"name": "vae"})
    net = pkg.build(spec)
    batch = row_batch()
    assert net.kl() is None
    net.eval()
    with torch.no_grad():
        net(batch)
    assert net.kl() is None                               # an eval forward writes nothing
    with torch.no_grad():
        pre = net.stages[0](batch["x_num"])
        expected = net.stages[1].divergence(pre)
        assert torch.equal(net.kl(batch), expected)
    net.train()
    torch.manual_seed(0)
    net(batch)
    assert net.kl() is not None and torch.allclose(net.kl(), expected)
    assert pkg.build(spec_of(MLP_RESIDUAL)).kl(batch) is None


def test_svdd_centre_is_a_buffer_and_survives_a_state_dict_round_trip(pkg):
    spec = spec_of([{"id": "E", "type": "embedding", "params": {"columns": ["zone"], "dim": 2}},
                    {"id": "C", "type": "concat"},
                    {"id": "S", "type": "sequential", "params": {"inner": [
                        {"type": "dense", "params": {"units": 8}},
                        {"type": "activation", "params": {"name": "relu"}}]}},
                    {"id": "H", "type": "head", "params": {"out": 4, "bias": False}}],
                   task={"kind": "anomaly"}, loss={"name": "svdd"})
    net = pkg.build(spec)
    centre = net.objective_center()
    assert centre is net.svdd_center and centre.shape == (4,)
    assert not bool(net.svdd_center_set)
    state = net.state_dict()
    assert {"svdd_center", "svdd_center_set"} <= set(state)
    assert net.param_count()[0] == sum(p.numel() for p in net.parameters())
    # what the loss does on the first training batch (R-B6): copy_ the centre, set the flag
    net.objective_center().copy_(torch.tensor([0.3, -0.2, 0.15, 0.1]))
    net.svdd_center_set.fill_(True)
    batch = row_batch(12)
    before = net.score(batch)
    expected = ((net.eval()(batch) - net.svdd_center) ** 2).sum(-1)
    assert before.shape == (12,) and torch.allclose(before, expected.detach())
    buf = io.BytesIO()
    torch.save(net.state_dict(), buf)
    buf.seek(0)
    fresh = pkg.build(spec_of(spec["blocks"], task={"kind": "anomaly"}, loss={"name": "svdd"},
                              seed=99))
    assert not bool(fresh.svdd_center_set)
    fresh.load_state_dict(torch.load(buf, weights_only=True))
    assert bool(fresh.svdd_center_set)
    assert torch.equal(fresh.score(batch), before)


def test_two_builds_of_one_spec_are_one_network(pkg):
    spec = spec_of(MLP_RESIDUAL)
    a, b = pkg.build(spec).state_dict(), pkg.build(spec).state_dict()
    assert list(a) == list(b) and all(torch.equal(a[k], b[k]) for k in a)
    c = pkg.build(spec_of(MLP_RESIDUAL, seed=8)).state_dict()
    assert not all(torch.equal(a[k], c[k]) for k in a)


# ================================================================ 4. the goldens

@pytest.fixture(scope="module")
def orders(tmp_path_factory):
    work = tmp_path_factory.mktemp("b06-orders")
    r = subprocess.run([sys.executable, str(HERE / "make_fixture.py"), str(work)],
                       capture_output=True, text=True)
    assert r.returncode == 0, f"make_fixture.py failed:\n{r.stdout}\n{r.stderr}"
    spec = json.loads((work / "netspec.json").read_text(encoding="utf-8"))
    pipeline = json.loads(Path(spec["input"]["pipeline"]).read_text(encoding="utf-8"))
    return spec, pipeline, work


def _orders_batch(spec, n=64, seed=5):
    feats = spec["data_contract"]["features"]
    numeric = [f for f in feats if f.get("role") != "nominal"]
    cards = [int(f["cardinality"]) for f in feats if f.get("role") == "nominal"]
    return row_batch(n, len(numeric), tuple(cards), seed)


def test_orders_params_and_blocks_equal_the_golden(pkg, orders):
    spec, _, _ = orders
    golden = json.loads((GOLDEN / "orders-smoke.json").read_text(encoding="utf-8"))
    net = pkg.build(spec)
    assert net.param_count() == (golden["params_total"], golden["params_trainable"])
    assert net.block_shapes(_orders_batch(spec, golden["batch_size"])) == golden["per_block"]


def test_orders_initial_weights_are_bitwise_the_v22_builds(pkg, orders, tmp_path):
    """The builder keeps v2.2's parameter creation and initialisation order (D1 §17.3): the
    same seeded weights, the same forward passes, eval and train. v2.2 reads the
    regularization live paths, not the declaration (rule R2 came after it), so it builds from
    the spec resolved (`vega_regularization.resolve`) — the network v2.3 builds from the
    declaration."""
    spec, pipeline, _ = orders
    emit = rt.baseline_emitter(tmp_path)          # c849f8b's emitter: the v2.2 layout
    pipeline = json.loads(json.dumps(pipeline))
    pipeline["system"]["package"] = PKG_V22
    out = tmp_path / "v22"
    rec = emit.emit(spec, pipeline, out, run_dir=out)
    assert "generation" not in rec["blueprint_map"]
    sys.path.insert(0, str(out))
    try:
        v22 = importlib.import_module(f"{PKG_V22}.model.network")
        dom22 = importlib.import_module(f"{PKG_V22}.domain")
        old = v22.build_model(dom22.NetSpec.from_dict(vega_regularization.resolve(spec)))
    finally:
        sys.path.remove(str(out))
        _purge(PKG_V22)
    new = pkg.build(spec)
    sa, sb = old.state_dict(), new.state_dict()
    assert list(sa) == list(sb)
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    batch = _orders_batch(spec)
    for mode in ("eval", "train"):
        getattr(old, mode)()
        getattr(new, mode)()
        torch.manual_seed(3)
        a = old(batch)
        torch.manual_seed(3)
        b = new(batch)
        assert torch.equal(a, b), mode


def _cids_root():
    root = os.environ.get("CIDS_ROOT")
    if not root or not (Path(root) / R7_BUNDLE / "netspec.json").is_file():
        pytest.skip(f"CIDS_ROOT names no {R7_BUNDLE}/netspec.json (set it to run the R7 check)")
    return Path(root)


def test_r7_params_equal_its_architecture_record(pkg):
    bundle = _cids_root() / R7_BUNDLE
    spec = json.loads((bundle / "netspec.json").read_text(encoding="utf-8"))
    record = json.loads((bundle / "records" / "architecture.json").read_text(
        encoding="utf-8"))["smoke"]
    net = pkg.build(spec)
    assert net.param_count() == (record["params_total"], record["params_trainable"])
    feats = spec["data_contract"]["features"]
    batch = row_batch(8, sum(1 for f in feats if f.get("role") != "nominal"), ())
    assert net.block_shapes(batch) == record["per_block"]
    golden = json.loads((GOLDEN / "cic-r7-smoke.json").read_text(encoding="utf-8"))
    assert golden["params_total"] == record["params_total"]


# ================================================================ 5. the size discipline

def _tree(pkg, name):
    return ast.parse(pkg.model_file(name).read_text(encoding="utf-8"))


def test_the_rendered_files_meet_c11_to_c13(pkg):
    import conformance
    for name in ("network.py", "readout.py", "builder.py"):
        findings = conformance.size_findings(_tree(pkg, name), f"model/{name}")
        assert findings == [], [f.message() for f in findings]


def test_network_py_is_one_class_and_its_only_constructor(pkg):
    """C14's top level and C9's single constructor, on the rendered adapter."""
    tree = _tree(pkg, "network.py")
    public = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Expr)):
            continue
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            public += [node.name] if not node.name.startswith("_") else []
            continue
        names = [t.id for t in node.targets] if isinstance(node, ast.Assign) else []
        public += [n for n in names if not (n.startswith("_") or n.isupper())]
    assert public == ["Network", "build_model"], public
    alias = next(n for n in tree.body if isinstance(n, ast.Assign)
                 and n.targets[0].id == "build_model")
    assert ast.unparse(alias.value) == "Network.build"
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Network")
    build = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "build")
    inside = {id(n) for n in ast.walk(build)}
    trees = {"network.py": tree, "readout.py": _tree(pkg, "readout.py"),
             "builder.py": _tree(pkg, "builder.py")}
    calls = [(name, node) for name, t in trees.items() for node in ast.walk(t)
             if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "Network"]
    assert calls and all(name == "network.py" and id(node) in inside for name, node in calls), \
        [(name, node.lineno) for name, node in calls]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
