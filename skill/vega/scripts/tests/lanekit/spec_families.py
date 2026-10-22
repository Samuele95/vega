"""
The families' default compositions and each block's smallest spec, for the unit tier
(lanekit.spec re-exports them).

    FAMILY_BLOCKS["gru"](target, contract)        # a family's default blocks over a contract
    REQUIRED_TYPES["mlp-plain"]                   # the types its catalog composition requires
    spec = family_spec("gru", "forecast", Contract.synthetic("window"))
    spec = minimal_block_spec("gat_conv")

`FAMILY_BLOCKS` is the executable reading of the catalog's compositions: for each of the 21
families, its default composition at the smallest sizes (width 16/32, depth 1-2) over a
`Contract`. `family_spec` and `minimal_block_spec` build complete specs from it; they carry no
chain inputs, so they are for the builder, not for `compile_spec.py`.

`REQUIRED_TYPES` is read from the catalog's `composition` text by `required_types`, not written
by hand: a block type the family's `block_types` lists is required when the text names it
outside an optional group (`x?`, `[a -> b]?`), an alternative (`a|b`, `a(..) or b(..)`), a
clause conditional on the graph level (`(+ graph_pool(mean) before the head at level graph)`),
a repeat that runs zero times at depth 1 (`[..] x (depth - 1)`) and a parameter's key
(`concat: true`). When the text gives one branch per shape or level (`linear`, `gcn`) the first
branch is read, the family's default context. One interpretation is `FAMILY_BLOCKS`'s own:
`[embedding?] -> concat` makes `concat` required by the letter, yet a default writes it only
with the embedding it joins — over a contract without nominals a concat of one input is the
identity — so the unit tier checks the required set over a contract that has a nominal.
"""
from __future__ import annotations

import copy
import re
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

from .spec_data import (CATALOG, OPTIMIZER_PARAMS, SCHEMA, TASK_ROWS, Contract, SpecError, Target,
                        head_out)

__all__ = ["FAMILY_BLOCKS", "REQUIRED_TYPES", "required_types", "family_spec", "minimal_block_spec",
           "synthetic_target"]

Blocks = Callable[[Target, Contract], list[dict]]


# --------------------------------------------------------------------------- block builders
def _embed_prefix(contract: Contract) -> list[dict]:
    noms = contract.nominals
    if not noms:
        return []
    dims = {c: max(2, min(50, 2 * round(1.6 * card ** 0.56 / 2))) for c, card in noms.items()}
    return [{"type": "embedding", "params": {"columns": list(noms), "dims": dims}},
            {"type": "concat", "params": {}}]


def _head(target: Target, contract: Contract, z: int | None = None, bias: bool = True) -> dict:
    params: dict[str, Any] = {"out": head_out(target, contract, z), "activation": "none"}
    if not bias:
        params["bias"] = False
    return {"type": "head", "params": params}


def _d(units: int, activation: str = "none") -> dict:
    return {"type": "dense", "params": {"units": units, "activation": activation}}


def _a(name: str) -> dict:
    return {"type": "activation", "params": {"name": name}}


def _p(p: float) -> dict:
    return {"type": "dropout", "params": {"p": p}}


def _t(block_type: str, /, **params: Any) -> dict:
    return {"type": block_type, "params": params}


def _graph_tail(contract: Contract, readout: str = "mean") -> list[dict]:
    level = (contract.graph or {}).get("level")
    return [_t("graph_pool", readout=readout)] if level == "graph" else []


def _conv_family(conv: Callable[[bool], dict], act: str = "relu", p: float = 0.5, readout: str = "mean") -> Blocks:
    def build(t: Target, c: Contract) -> list[dict]:
        return _embed_prefix(c) + [conv(False), _a(act), _p(p), conv(True)] + _graph_tail(c, readout) + [_head(t, c)]
    return build


def _rec(kind: str, ret: str, bidirectional: bool = False) -> dict:
    params: dict[str, Any] = {"hidden": 32, "num_layers": 1, "return": ret}
    if bidirectional:
        params["bidirectional"] = True
    return {"type": kind, "params": params}


def _linear(t: Target, c: Contract) -> list[dict]:
    tail = {"window": [_t("flatten")]}.get(c.shape, [])
    return _embed_prefix(c) + tail + _graph_tail(c) + [_head(t, c)]


def _encoder(d: int, heads: int, ff: int, **extra: Any) -> dict:
    return _t("transformer_encoder", d=d, heads=heads, layers=1, ff=ff, **extra)


# --------------------------------------------------------------------------- the families
FAMILY_BLOCKS: Mapping[str, Blocks] = MappingProxyType({
    "mlp-plain": lambda t, c: _embed_prefix(c) + [_d(32), _a("gelu"), _p(0.1), _d(32), _a("gelu"), _p(0.1), _head(t, c)],
    "mlp-residual": lambda t, c: _embed_prefix(c) + [_d(32), _t(
        "residual", repeat=2, projection="auto", inner=[_d(32), _t("batchnorm"), _a("gelu"), _p(0.2)]), _head(t, c)],
    "ft-transformer": lambda t, c: [_t("feature_tokenizer", d=32), _encoder(32, 4, 64, dropout=0.1),
                                    _t("pool", kind="cls"), _head(t, c)],
    "linear": _linear,
    "pca-floor": lambda t, c: _embed_prefix(c) + [_d(3), _head(t, c)],
    "autoencoder": lambda t, c: _embed_prefix(c) + [_t("sequential", inner=[_d(16), _a("relu"), _p(0.1)]), _d(4),
                                                    _t("decoder", widths=[16], activation="relu"), _head(t, c)],
    "vae": lambda t, c: _embed_prefix(c) + [_t("sequential", inner=[_d(16, "gelu")]), _d(8), _t("reparameterize", z=4),
                                            _t("decoder", widths=[16], activation="gelu"), _head(t, c)],
    "deep-svdd": lambda t, c: _embed_prefix(c) + [_t("sequential", inner=[_d(16), _a("tanh")]),
                                                  _head(t, c, z=8, bias=False)],
    "cnn1d": lambda t, c: _embed_prefix(c) + [_t("conv1d", channels=16, kernel=3, padding=1), _a("gelu"),
                                              _t("pool1d", kind="max", size=2), _t("flatten"), _d(32), _head(t, c)],
    "gru": lambda t, c: _embed_prefix(c) + [_rec("gru", "all"), _t("pool", kind="mean"), _head(t, c)],
    "lstm": lambda t, c: _embed_prefix(c) + [_rec("lstm", "all"), _t("pool", kind="mean"), _head(t, c)],
    "transformer-seq": lambda t, c: _embed_prefix(c) + [_d(32), _t("positional", kind="sinusoidal"),
                                                        _encoder(32, 4, 64, dropout=0.1), _t("pool", kind="mean"),
                                                        _head(t, c)],
    "rnn": lambda t, c: _embed_prefix(c) + [_t("rnn", hidden=32, num_layers=1, nonlinearity="tanh", **{"return": "last"}),
                                            _p(0.1), _head(t, c)],
    "bi-lstm": lambda t, c: _embed_prefix(c) + [_rec("lstm", "last", True), _p(0.1), _head(t, c)],
    "bi-gru": lambda t, c: _embed_prefix(c) + [_rec("gru", "last", True), _p(0.1), _head(t, c)],
    "lstm-attention": lambda t, c: _embed_prefix(c) + [_rec("lstm", "all"), _t("attention_pool", hidden=32),
                                                       _p(0.1), _head(t, c)],
    "gru-attention": lambda t, c: _embed_prefix(c) + [_rec("gru", "all"), _t("attention_pool", hidden=32),
                                                      _p(0.1), _head(t, c)],
    "gcn": _conv_family(lambda last: _t("gcn_conv", out=32)),
    "graphsage": _conv_family(lambda last: _t("sage_conv", out=32, aggregator="mean")),
    "gat": _conv_family(lambda last: _t("gat_conv", out=8, heads=1 if last else 4, concat=not last), act="leaky_relu"),
    "gin": _conv_family(lambda last: _t("gin_conv", out=32), p=0.1, readout="sum"),
})

# what the composition text says outside the required path (the module docstring, in order)
_NOT_REQUIRED = (re.compile(r"\w+\?"),                                          # x?
                 re.compile(r"\[[^\]]*\]\?"),                                    # [a -> b]?
                 re.compile(r"\[[^\]]*\]\s*x\s*\(depth\s*-\s*1\)"),              # [..] x (depth - 1)
                 re.compile(r"\(\+[^;]*?level \w+\)"),                           # (+ .. at level graph)
                 re.compile(r"\w+(?:\([^()]*\))?\s+or\s+\w+(?:\([^()]*\))?"),    # a(..) or b(..)
                 re.compile(r"\w+(?:\|\w+)+"),                                   # a|b
                 re.compile(r"\b\w+\s*:"))                                       # a parameter's key
_BRANCH_LABEL = re.compile(r"^\s*[\w ,]+:\s+")                                   # "row: ", "graph, level node: "


def required_types(family: str) -> frozenset[str]:
    """The block types a family's catalog `composition` requires (the module docstring)."""
    row = CATALOG["families"][family]
    text = _BRANCH_LABEL.sub("", str(row["composition"]).split(";")[0])
    for pattern in _NOT_REQUIRED:
        text = pattern.sub(" ", text)
    types = set(row.get("block_types") or ())
    return frozenset(word for word in re.findall(r"[a-z][a-z0-9_]*", text) if word in types)


REQUIRED_TYPES: Mapping[str, frozenset[str]] = MappingProxyType(
    {family: required_types(family) for family in CATALOG["families"]})


# --------------------------------------------------------------------------- the unit tier's specs
def _numbered(blocks: Sequence[Mapping[str, Any]]) -> list[dict]:
    return [{"id": f"B{i}", **copy.deepcopy(dict(b))} for i, b in enumerate(blocks, 1)]


def synthetic_target(task: str, family: str, horizon: int = 3) -> Target:
    """The target of a unit-tier cell: `task` is a kind, or `sequence-*/<base>` as the admission
    table writes it; the loss is the family's, else the task row's default."""
    kind, _, base = task.partition("/")
    base = base or None
    effective = base if kind.startswith("sequence-") and base else kind
    loss = CATALOG["families"].get(family, {}).get("loss") or TASK_ROWS[effective].default_loss
    k = {"binary": 2, "multiclass": 3, "multilabel": 3, "ordinal": 4}.get(effective)
    return Target(kind, base, k, horizon if kind == "forecast" else None, 0, loss)


def _unit_task(target: Target) -> dict:
    effective = target.effective
    task: dict[str, Any] = {"kind": target.kind, "label": None if effective == "unsupervised" else "y",
                            "source": "user:stance/unit"}
    if target.base:
        task["base_kind"] = target.base
    if target.n_classes:
        task["classes"] = list(range(target.n_classes))
    if effective == "multilabel":
        task["classes"] = [f"c{i}" for i in range(target.n_classes or 3)]
        task["label_format"] = {"kind": "delimited", "sep": "|"}
    if effective in ("binary", "anomaly") and target.n_classes:
        task["positive"] = 1
    return task


def _unit_sample(target: Target, contract: Contract) -> dict:
    sample: dict[str, Any] = {"shape": contract.shape}
    if contract.shape == "window":
        sample.update({"length": 8, "stride": 1, "series_by": None, "order_by": "t", "context": "previous-part",
                       "autoregressive": target.kind == "forecast", "season": None, "source": "user:stance/unit"})
        if target.kind == "forecast":
            sample["horizon"] = target.horizon
    elif contract.shape == "graph":
        level = (contract.graph or {}).get("level", "node")
        sample.update({"edges": contract.edges, "level": level, "graph_id": (contract.graph or {}).get("graph_id"),
                       "directed": False, "self_loops": False, "protocol": "inductive",
                       "batching": "graphs" if level == "graph" else "full", "neighbours": None,
                       "exposure": None, "source": "user:stance/unit"})
    return sample


def _synthetic_spec(family: str, target: Target, contract: Contract, blocks: Sequence[Mapping[str, Any]]) -> dict:
    """A complete spec over a synthetic contract, for the builder (no chain inputs)."""
    sample = _unit_sample(target, contract)
    loss = target.loss or TASK_ROWS[target.effective].default_loss
    primary = TASK_ROWS[target.effective].primary if target.effective != "anomaly" else "anomaly_score_mean"
    return {
        "schema": SCHEMA,
        "input": {"model": None, "model_digest": None, "layers_read": [], "shaped_dir": None,
                  "manifest": None, "manifest_digest": None, "pipeline": None, "pipeline_digest": None},
        "seed": 7, "task": _unit_task(target),
        "data_contract": {"source": "user:stance/unit", "parts": {**dict(contract.parts), "test": "sealed"},
                          "features": [dict(f) for f in contract.features], "input_dim": contract.input_dim,
                          "keys": list(contract.keys), "sample": sample},
        "family": {"name": family, "source": "user:stance/unit", "rationale": "the unit tier's cell",
                   "alternatives": []},
        "blocks": _numbered(blocks),
        "loss": {"name": loss, "params": {"quantiles": [0.1, 0.5, 0.9]} if loss == "quantile" else {}},
        "metrics": {"primary": primary, "secondary": [], "selection": {
            "monitor": f"valid.{primary}", "mode": CATALOG["metric_direction"].get(primary, "max")}},
        "optimizer": {"name": "adamw", "params": dict(OPTIMIZER_PARAMS["adamw"])},
        "schedule": {"name": "none", "params": {}},
        "regularization": [], "early_stopping": {"monitor": f"valid.{primary}", "patience": 2},
        "init": {"scheme": "orthogonal" if contract.shape == "window" else "kaiming_uniform", "bias": "zeros"},
        "training": {"batch_size": "full" if sample.get("batching") == "full" else 32, "max_epochs": 2,
                     "device": "cpu", "precision": "fp32", "num_workers": 0, "budget": {"wall_clock_min": 1}},
        "backend": {"name": "pytorch", "min_version": "2.2", "hints": {}},
        "decision_families": copy.deepcopy(dict(CATALOG["decision_families"])),
        "outputs": {"run_dir": None, "package": "unit_pipeline", "code": [], "notebook": None},
    }


def family_spec(family: str, task: str, contract: Contract) -> dict:
    """A family's default composition for one task over a contract, as a complete spec (the
    unit tier's build matrix). `task` is a kind, or `sequence-*/<base>`."""
    if family not in FAMILY_BLOCKS:
        raise SpecError(f"no default composition for family {family!r}")
    target = synthetic_target(task, family)
    return _synthetic_spec(family, target, contract, FAMILY_BLOCKS[family](target, contract))


_ROW, _SEQ = ("row", None, "binary"), ("window", None, "sequence-classification/binary")
_NODE, _GRAPH = ("graph", "node", "binary"), ("graph", "graph", "binary")
_POOL_MEAN, _FLAT = _t("pool", kind="mean"), _t("flatten")
# block type -> ((sample shape, graph level, task), blocks): the smallest spec each block builds in
_MINIMAL: Mapping[str, tuple[tuple[str, str | None, str], Blocks]] = MappingProxyType({
    "dense": (_ROW, lambda t, c: [_d(8), _head(t, c)]),
    "embedding": (_ROW, lambda t, c: _embed_prefix(c) + [_head(t, c)]),
    "concat": (_ROW, lambda t, c: _embed_prefix(c) + [_head(t, c)]),
    "batchnorm": (_ROW, lambda t, c: [_d(8), _t("batchnorm"), _head(t, c)]),
    "layernorm": (_ROW, lambda t, c: [_d(8), _t("layernorm"), _head(t, c)]),
    "dropout": (_ROW, lambda t, c: [_d(8), _p(0.1), _head(t, c)]),
    "activation": (_ROW, lambda t, c: [_d(8), _a("gelu"), _head(t, c)]),
    "residual": (_ROW, lambda t, c: [_d(8), _t("residual", repeat=1, projection="auto", inner=[_d(8), _a("relu")]),
                                     _head(t, c)]),
    "sequential": (_ROW, lambda t, c: [_t("sequential", inner=[_d(8), _a("relu")]), _head(t, c)]),
    "parallel": (_ROW, lambda t, c: [_t("parallel", merge="concat", inner=[_d(4), _d(4)]), _head(t, c)]),
    "reshape": (_ROW, lambda t, c: [_d(8), _t("reshape", shape=[2, 4]), _POOL_MEAN, _head(t, c)]),
    "pool": (_SEQ, lambda t, c: [_rec("gru", "all"), _POOL_MEAN, _head(t, c)]),
    "flatten": (_SEQ, lambda t, c: [_FLAT, _head(t, c)]),
    "conv1d": (_SEQ, lambda t, c: [_t("conv1d", channels=4, kernel=3, padding=1), _FLAT, _head(t, c)]),
    "pool1d": (_SEQ, lambda t, c: [_t("pool1d", kind="max", size=2), _FLAT, _head(t, c)]),
    "rnn": (_SEQ, lambda t, c: [_t("rnn", hidden=8, **{"return": "last"}), _head(t, c)]),
    "gru": (_SEQ, lambda t, c: [_t("gru", hidden=8, **{"return": "last"}), _head(t, c)]),
    "lstm": (_SEQ, lambda t, c: [_t("lstm", hidden=8, **{"return": "last"}), _head(t, c)]),
    "attention": (_SEQ, lambda t, c: [_d(8), _t("attention", heads=2, dim=8, pool=True), _head(t, c)]),
    "attention_pool": (_SEQ, lambda t, c: [_t("gru", hidden=8, **{"return": "all"}), _t("attention_pool", hidden=8),
                                           _head(t, c)]),
    "transformer_encoder": (_SEQ, lambda t, c: [_d(8), _encoder(8, 2, 16), _POOL_MEAN, _head(t, c)]),
    "positional": (_SEQ, lambda t, c: [_d(8), _t("positional", kind="sinusoidal"), _POOL_MEAN, _head(t, c)]),
    "feature_tokenizer": (_ROW, lambda t, c: [_t("feature_tokenizer", d=8), _t("pool", kind="cls"), _head(t, c)]),
    "head": (_ROW, lambda t, c: [_head(t, c)]),
    "decoder": (("row", None, "unsupervised"), lambda t, c: [_d(4), _t("decoder", widths=[8]), _head(t, c)]),
    "reparameterize": (("row", None, "unsupervised"), lambda t, c: [_d(8), _t("reparameterize", z=4),
                                                                    _t("decoder", widths=[8]), _head(t, c)]),
    "custom": (_ROW, lambda t, c: [_t("custom", code_ref="custom/custom_gain.py", out_shape=[c.input_dim]), _head(t, c)]),
    "gcn_conv": (_NODE, lambda t, c: [_t("gcn_conv", out=8), _head(t, c)]),
    "sage_conv": (_NODE, lambda t, c: [_t("sage_conv", out=8), _head(t, c)]),
    "gat_conv": (_NODE, lambda t, c: [_t("gat_conv", out=8, heads=2), _head(t, c)]),
    "gin_conv": (_NODE, lambda t, c: [_t("gin_conv", out=8), _head(t, c)]),
    "graph_pool": (_GRAPH, lambda t, c: [_t("gcn_conv", out=8), _t("graph_pool", readout="mean"), _head(t, c)]),
    "edge_dropout": (_NODE, lambda t, c: [_t("edge_dropout", p=0.1), _t("gcn_conv", out=8), _head(t, c)]),
})


def minimal_block_spec(block_type: str) -> dict:
    """The smallest complete spec `block_type` builds in: the block, what its sample shape needs
    around it, and a head (unit tier)."""
    if block_type not in _MINIMAL:
        raise SpecError(f"no minimal spec for block type {block_type!r}")
    (shape, level, task), build = _MINIMAL[block_type]
    nominals = {"c1": 5} if block_type in ("embedding", "concat") else None
    contract = Contract.synthetic(shape, nominals=nominals, level=level)
    family = {"row": "mlp-plain", "window": "gru", "graph": "gcn"}[shape]
    target = synthetic_target(task, family if task != "unsupervised" else "autoencoder")
    return _synthetic_spec(family, target, contract, build(target, contract))
