"""
_inspect_fixtures.py — rendered packages, example batches and one explanation pass, shared by
`test_inspect_probes.py` and `test_inspect_concurrency.py` (build plan I03).

A case is one spec rendered as a package — the model templates (`domain.py`, `network.py`,
`readout.py`, the spliced `builder.py` and block files) and the serving package's inspection
templates layered in the Predictor's module, as /vega-pack renders them — with its network in
eval mode, an example batch of the spec's sample shape, its stage graph, and the permanent
hooks the inspector attaches once (`capture.HookSet`).

`explain(case, ...)` is one explanation's node section, as the inspector assembles it: a
`Capture` pass over the batch, the explained scalar differentiated once, every node placed
under the budget (`parts.NodeParts`), the recompute report and the dense chain. The explained
scalar is the head's first output for the sample: the per-task target table is `targets.py`'s
(I04), and the node section reads any scalar the pass computed the same way.

The specs cover the catalog's 21 families and user-defined compositions that reach what no
family does (batchnorm, parallel, reshape, custom, a learned positional table, an unpooled
attention, colliding block ids, a rank-consistent head, convs without self loops); together
they hold every one of the 33 block types and the 5 synthetic ones.
"""
from __future__ import annotations

import importlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
BACKEND = SCRIPTS / "backends" / "pytorch"
TEMPLATES = BACKEND / "templates"
INSPECTION = TEMPLATES / "serving" / "package" / "inspection"
REPO = SCRIPTS.parents[1]
CATALOG = json.loads((SCRIPTS / "catalog.json").read_text(encoding="utf-8"))
EXPLANATION = json.loads((SCRIPTS.parent / "references" / "schemas" /
                          "explanation-2.schema.json").read_text(encoding="utf-8"))
for _p in (str(SCRIPTS), str(HERE), str(BACKEND)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import render_templates as rt  # noqa: E402

STUB_LOSSES = '''"""A stub of the losses source: the inspection tests build networks, never a loss."""
# --8<-- losses
def build_loss(spec):
    raise NotImplementedError("the inspection tests render no loss")
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
            "family": {"name": family, "source": "user:test_inspect_probes"}, "blocks": blocks,
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
    "user-sage": _spec("user-defined", [
        _b("G1", "sage_conv", out=6, aggregator="mean", normalize=True),
        _b("G2", "sage_conv", out=4, aggregator="sum"), _b("H", "head", out=1)],
        {"kind": "regression", "label": "y"}, sample=NODE, features=NUM, loss=MSE),
    "user-noloop": _spec("user-defined", [
        _b("G1", "gcn_conv", out=6, self_loops=False),
        _b("G2", "gat_conv", out=3, heads=2, concat=True, self_loops=False),
        _b("H", "head", out=1)], {"kind": "regression", "label": "y"}, sample=NODE,
        features=NUM, loss=MSE),
}
FAMILIES = [n for n in CASES if not n.startswith("user-")]


# ================================================================ batches

EDGES = [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2), (4, 5), (5, 4), (5, 6), (6, 5),
         (7, 8), (8, 7), (8, 9), (9, 8), (3, 0)]


def batch_for(spec, seed=1):
    """An example batch of the spec's sample shape (the rows the Predictor would assemble)."""
    sample = spec["data_contract"]["sample"]
    feats = spec["data_contract"]["features"]
    numeric = sum(1 for f in feats if f["role"] != "nominal") + \
        (1 if sample.get("autoregressive") else 0)
    cards = [f["cardinality"] for f in feats if f["role"] == "nominal"]
    g = torch.Generator().manual_seed(seed)
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
    """One rendered package: its modules, its network (eval mode), its graph and its hooks."""

    def __init__(self, name, root, package, spec):
        self.name, self.root, self.package, self.spec = name, Path(root), package, spec
        mod = lambda m: importlib.import_module(f"{package}.{m}")  # noqa: E731
        self.domain, self.network = mod("domain"), mod("model.network")
        self.graph_mod = mod("serving.inspection.graph")
        self.capture = mod("serving.inspection.capture")
        self.parts = mod("serving.inspection.parts")
        self.reduce = mod("serving.inspection.reduce")
        self.codec = mod("serving.inspection.codec")
        self.registry = mod("serving.inspection.registry")
        self.view = mod("serving.inspection.view")
        self.model = self.network.Network.build(self.domain.NetSpec.from_dict(spec)).eval()
        self.batch = batch_for(spec)
        self.graph = self.graph_mod.StageGraph.build(
            self.model, self.domain.NetSpec.from_dict(spec), (self.batch, None))
        self.hooks = self.capture.HookSet.attach(self.model, self.graph)
        self.body = self.view.network_view(self.graph, "full", "sha256:" + "1" * 64)

    def samples(self, batch=None) -> int:
        with torch.no_grad():
            return int(self.model(batch or self.batch).shape[0])


def _render(name, work, spec):
    package = "i03_" + name.replace("-", "_")
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
    rt.render(TEMPLATES / "serving" / "package", out, pipeline, spec, into="Predictor",
              recursive=True)
    custom = out / package / "model" / "custom"
    custom.mkdir(exist_ok=True)
    (custom / "gain.py").write_text(CUSTOM_GAIN, encoding="utf-8")
    sys.path.insert(0, str(out))
    return Case(name, out, package, spec)


_WORK = tempfile.TemporaryDirectory(prefix="inspect-probes-")
_CASES = {}


def case(name) -> Case:
    """The rendered case `name` (rendered once per process)."""
    if name not in _CASES:
        _CASES[name] = _render(name, Path(_WORK.name), CASES[name])
    return _CASES[name]


# ================================================================ one explanation

def scalar(head, index):
    """The explained number: the head's first output for sample `index`."""
    return head[index].reshape(-1)[0]


def section(c, cap, head, index, detail="full", limits=None) -> dict:
    """The node section of a pass that ran: the scalar differentiated, the nodes placed."""
    cap.differentiate(scalar(head, index))
    parts = c.parts.NodeParts(cap)
    budget = c.reduce.Budget(detail, limits or c.graph.codec.limits)
    nodes = parts.place(budget)
    return {"detail": detail, "degraded": budget.degraded, "nodes": nodes,
            "chain": parts.chain(c.graph.chain()), "recompute": parts.report(),
            "budget": budget.report(), "head": c.codec.nums(head[index].detach())}


def explain(c, index=0, detail="full", *, batch=None, limits=None):
    """(the node section of one explanation, its capture): see the module docstring."""
    cap = c.capture.Capture(c.graph)
    head = cap.run(c.model, batch if batch is not None else c.batch, index)
    return section(c, cap, head, index, detail, limits), cap


def canonical(section) -> bytes:
    """The bytes an explanation is compared by (the route's serialisation)."""
    return json.dumps(section, sort_keys=True, separators=(",", ":"), allow_nan=False,
                      ensure_ascii=True).encode("utf-8")


def hooks_on(model) -> int:
    """Every forward and forward-pre hook registered on the network's modules."""
    return sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules())
