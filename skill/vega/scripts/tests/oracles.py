"""
oracles.py — the lane checks' door to the one float64 oracle module (BUILD-PLAN R-H4, D6 §8.3).

`vega-forge/scripts/tests/block_oracles.py` (B03) is the release's only oracle: numpy float64,
every catalog block type, written from the formulas and never from the block code. This module
re-exports it unchanged — `ORACLES`, `oracle`, `BLOCK_TYPES`, `SHAPE_ONLY`, `activation`,
`affine`, `coefficients`, `gat_alpha`, `attention_pool_weights`, `encoder_layer`,
`sinusoidal_table`, `adjacency`, `check_custom`, `weights_of` — and adds the two things a lane
check needs that a module-level oracle cannot know:

    shape_rule(block_type, params, in_shape, lane=None) -> tuple
        D6 §8.3's signature: per-sample shapes (no batch axis), the context read from the
        lane (a `lanekit.registry.Lane`: the embedding width and nominal count its blocks
        declare), or a mapping of the oracle's context keys (`emb_width`, `n_cat`,
        `num_graphs`, and `recon_dim`, the width a decoder without `out` reconstructs);
    view_block(view, node_id) -> (type, weights, params)
        one node of a served `vega/network-view@2` payload read as the oracle reads a torch
        block: the tensors it publishes under the oracle's role names, its configuration, and
        for a composite its children as `(type, weights, params)` triples. `view` is anything
        with `node(id)`, `children(id)` and `tensors_of(id) -> {name: array}` — the lane
        checks' `NetworkView` (inspect_recompute.py) fills the arrays from `GET /v1/network`.

`encoder_layer_any` covers the post-norm encoder layer too (the oracle module's
`encoder_layer` is the pre-norm one the backend builds by default).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

HERE = Path(__file__).resolve().parent
FORGE_TESTS = HERE.parents[2] / "vega-forge" / "scripts" / "tests"
if str(FORGE_TESTS) not in sys.path:
    sys.path.insert(0, str(FORGE_TESTS))

import block_oracles as _O  # noqa: E402  (the one oracle module, R-H4)
from block_oracles import (  # noqa: E402,F401  (re-exported)
    BLOCK_TYPES, ORACLES, SHAPE_ONLY, activation, adjacency, affine, attention_pool_weights,
    check_custom, coefficients, encoder_layer, gat_alpha, oracle, sinusoidal_table, weights_of)

__all__ = ["BLOCK_TYPES", "COMPOSITES", "ORACLES", "SHAPE_ONLY", "activation", "adjacency",
           "affine", "attention_pool_weights", "check_custom", "coefficients",
           "encoder_layer", "encoder_layer_any", "gat_alpha", "oracle", "shape_rule",
           "sinusoidal_table", "view_block", "weights_of"]

COMPOSITES = frozenset({"sequential", "residual", "parallel", "decoder", "transformer_encoder"})
# a node's configuration keys that are not the oracle's parameter names
_PARAM_ALIASES = {"layers": "num_layers"}
# the published name of a tensor -> the oracle's role name, per block type
_RENAMES = {"sage_conv": {"bias": "neigh_bias"}}
_ENCODER_NAMES = {"self_attn.in_proj": "in_proj", "self_attn.in_proj_bias": "in_proj_bias",
                  "self_attn.out_proj": "out_proj", "self_attn.out_proj_bias": "out_proj_bias",
                  "ff.linear1": "linear1", "ff.linear1_bias": "linear1_bias",
                  "ff.linear2": "linear2", "ff.linear2_bias": "linear2_bias",
                  "norm1.gamma": "norm1_gamma", "norm1.beta": "norm1_beta",
                  "norm2.gamma": "norm2_gamma", "norm2.beta": "norm2_beta"}


# --------------------------------------------------------------------------- shapes
def _context(lane: Any) -> dict:
    """The oracle's shape context from a lane: `emb_width` (the embedding dims its blocks
    declare), `n_cat` (the nominal columns they embed), `num_graphs` (one per sample)."""
    if lane is None:
        return {"num_graphs": 1}
    if isinstance(lane, Mapping) and "blocks" not in lane:
        return {"num_graphs": 1, **dict(lane)}
    blocks = lane.get("blocks") if isinstance(lane, Mapping) else lane.blocks
    width, columns = 0, []
    for block in _walk(blocks or ()):
        params = block.get("params") or {}
        if block.get("type") == "embedding":
            cols = [str(c) for c in params.get("columns") or ()]
            columns += cols
            width += sum(_O._embedding_dims({"columns": cols, **params}))
    return {"emb_width": width, "n_cat": len(columns), "num_graphs": 1}


def _walk(blocks):
    for block in blocks:
        yield block
        yield from _walk((block.get("params") or {}).get("inner") or ())


def _oracle_params(block_type: str, params: Mapping) -> dict:
    """A netspec block's params as the oracle's rule reads them: a composite's `inner` spec
    blocks become its `children` (type, params) pairs, recursively."""
    params = dict(params or {})
    inner = params.get("inner")
    if block_type in ("sequential", "residual", "parallel") and inner is not None and \
            "children" not in params:
        params["children"] = [(b["type"], _oracle_params(b["type"], b.get("params") or {}))
                              for b in inner]
    return params


def shape_rule(block_type: str, params: Mapping, in_shape, lane: Any = None) -> tuple:
    """The per-sample output shape of a block (D6 §8.3) — the oracle's rule with a unit
    leading axis added and removed. `in_shape` is the block's primary input per sample: x_num
    for concat and feature_tokenizer, x_cat for embedding, a node's features for the graph
    blocks (a graph pool reads one sample's nodes and writes one row). `params` are the
    netspec's (a composite's `inner` blocks) or the oracle's (`children`)."""
    per_sample, context = tuple(int(s) for s in in_shape), _context(lane)
    params = _oracle_params(block_type, params)
    if block_type == "decoder" and "out" not in params:
        if "recon_dim" not in context:            # the builder gives a decoder out = recon_dim
            raise KeyError("a decoder without `out` reconstructs `recon_dim` features: pass "
                           "it in the context")
        params["out"] = int(context["recon_dim"])
    out = _O.shape_rule(block_type, params, (1,) + per_sample, context)
    return tuple(int(s) for s in out[1:])


# --------------------------------------------------------------------------- the served view
def _scalars(node: Mapping) -> dict:
    """A node's configuration, then its payload's plain values (tensor ids and derived
    vectors left out), under the oracle's parameter names."""
    params = dict(node.get("config") or {})
    for key, value in (node.get("params") or {}).items():
        if key in ("count", "trainable") or isinstance(value, (list, dict)) or key in params:
            continue
        if isinstance(value, str) and "." in value and value.startswith(node["id"] + "."):
            continue                                         # a tensor id
        params[key] = value
    for published, name in _PARAM_ALIASES.items():
        if published in params and name not in params:
            params[name] = params[published]
    if node["type"] in ("rnn", "gru", "lstm"):
        params["kind"] = node["type"]                   # the oracle's cell switch
    return params


def _weights(view, node: Mapping) -> dict:
    kind, own = node["type"], dict(view.tensors_of(node["id"]))
    if kind == "embedding":
        tables = (node.get("params") or {}).get("tables") or ()
        return {f"table.{t['column']}": view.tensor(t["table"]) for t in tables}
    if kind == "feature_tokenizer":
        tables = (node.get("params") or {}).get("tables") or ()
        keep = {k: v for k, v in own.items() if not k.startswith("table.")}
        keep.update({f"table.{j}": view.tensor(t["table"]) for j, t in enumerate(tables)})
        return keep
    if kind == "gin_conv" and "eps" not in own:
        own["eps"] = np.asarray([float((node.get("params") or {}).get("eps", 0.0))])
    renames = _RENAMES.get(kind, {})
    return {renames.get(k, k): v for k, v in own.items()}


def view_block(view, node_id: str) -> tuple:
    """(type, weights, params) of one served node, as the oracle reads a block; a composite's
    params carry its children's triples in the oracle's layout."""
    node = view.node(node_id)
    kind = node["type"]
    params = _scalars(node)
    if kind == "encoder_layer":
        return kind, {f"layer0.{_ENCODER_NAMES.get(k, k)}": v
                      for k, v in view.tensors_of(node_id).items()}, params
    if kind not in COMPOSITES:
        return kind, _weights(view, node), params
    return _composite(view, node, params)


def _composite(view, node: Mapping, params: dict) -> tuple:
    kind, payload = node["type"], node.get("params") or {}
    if kind == "residual":
        repeats = payload.get("repeats") or ()
        params["repeats"] = [[view_block(view, c) for c in r["inner"]] for r in repeats]
        params["projections"] = [None if not r.get("projection") else
                                 view_block(view, r["projection"])[1] for r in repeats]
        return kind, {}, params
    if kind == "parallel":
        params["children"] = [view_block(view, c) for c in payload.get("branches") or ()]
        return kind, {}, params
    children = [c for c in view.children(node["id"])]
    if kind == "decoder":
        weights = {}
        for k, child in enumerate(children):
            _, w, _ = view_block(view, child)
            weights[f"layer{k}.weight"], weights[f"layer{k}.bias"] = w["weight"], w.get("bias")
        return kind, weights, params
    if kind == "transformer_encoder":
        weights = {}
        for k, child in enumerate(children):
            for name, value in view_block(view, child)[1].items():
                weights[f"layer{k}." + name[len("layer0."):]] = value
            params.setdefault("activation", view.node(child)["config"].get("activation"))
        return kind, weights, params
    params["children"] = [view_block(view, c) for c in children]
    return kind, {}, params


def encoder_layer_any(weights: Mapping, x, params: Mapping, k: int = 0):
    """One encoder layer, pre-norm (`norm_first`, the oracle module's) or post-norm."""
    if params.get("norm_first", True):
        return encoder_layer(weights, x, params, k)
    pre, eps = f"layer{k}.", float(params.get("eps", 1e-5))
    x = _O._f64(x)
    x = _O._ln(x + _O._mha(x, weights, int(params["heads"]), prefix=pre),
               weights[pre + "norm1_gamma"], weights[pre + "norm1_beta"], eps)
    h = activation(params.get("activation", "relu"),
                   x @ _O._f64(weights[pre + "linear1"]).T + _O._f64(weights[pre + "linear1_bias"]))
    h = h @ _O._f64(weights[pre + "linear2"]).T + _O._f64(weights[pre + "linear2_bias"])
    return _O._ln(x + h, weights[pre + "norm2_gamma"], weights[pre + "norm2_beta"], eps)
