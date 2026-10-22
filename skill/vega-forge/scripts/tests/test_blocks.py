#!/usr/bin/env python3
"""
test_blocks.py — the PyTorch block library (`backends/pytorch/blocks.py`) against its float64
oracles (`block_oracles.py`), and the block contract of BUILD-PLAN R-B1–R-B4, R-B8 and D1 §5.

    python3 test_blocks.py            (or: python3 -m pytest test_blocks.py -q)

What it proves, case by case over every catalog block type (the 32 with a value oracle; the
user-code `custom` block by its declared shape):
- the block, built through its `from_params`, equals its numpy float64 oracle on random
  inputs (float64 at 1e-9; float32 at the lane tolerance rtol 1e-4, atol 1e-5·max(1, |v|)),
  and its output shape equals `shape_rule`;
- the graph convolutions: GCN (sym and rw, weighted, with and without self loops, with a loop
  already in the data) equals the dense `D^-½ Â D^-½ X W`; SAGE mean / max / sum equal the
  dense loops; GAT's α sums to 1 per destination and head and equals a masked softmax; GIN
  equals `MLP((1 + ε) x + A x)`; `coefficients()` reproduces each conv's output at every
  target within 1e-5; `graph_pool` equals a pandas groupby over graphs of unequal size (and an
  empty one); `edge_dropout` is the identity in eval and keeps ≈ 1 − p in training;
- the recurrent blocks: `return: last` equals `h_n` of the top layer (uni- and bidirectional,
  rnn / gru / lstm, stacked), `num_layers` and its old name `layers` are honoured (D1 E7);
- `attention_pool` weights are non-negative and sum to 1, and the output is `Σ α_t h_t`;
- `CumulativeHead` is monotone and its class probabilities are non-negative;
- the tokenizer with nominals gives (B, n_num + n_cat + 1, d) and, without them, is the v2.2
  tokenizer exactly;
- R-B4: a setattr spy sees no attribute written during an eval forward (with and without
  autograd), the parameters and buffers stay bitwise equal, no `.grad` appears, and no block
  keeps a `last_*` attribute;
- two seeded CPU builds give identical weights, one-step losses and gradients, and every
  parameter gets a finite gradient;
- the emitter's maps: every catalog type has its segments, file, class and stage kind; the
  maps parse without importing torch (as `emit.py` reads them); each emitted support file,
  rebuilt from its segments plus the prelude, defines and runs its classes alone.
"""
from __future__ import annotations

import ast
import json
import re
import sys
import types
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pytest
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
BACKEND = SCRIPTS / "backends" / "pytorch"
for _p in (str(BACKEND), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import blocks as B  # noqa: E402
import block_oracles as O  # noqa: E402

BLOCKS_PY = BACKEND / "blocks.py"
BUILDER_PY = BACKEND / "builder.py"
CATALOG = json.loads((SCRIPTS / "catalog.json").read_text(encoding="utf-8"))
SUPPORT_FILES = {"blocks_core", "blocks_seq", "blocks_graph", "blocks_gen"}


# ------------------------------------------------------------------ inputs

def _gen(seed):
    return torch.Generator().manual_seed(seed)


def _rows(b=5, f=6, seed=1):
    return torch.randn(b, f, generator=_gen(seed))


def _window(b=4, length=7, f=6, seed=2):
    return torch.randn(b, length, f, generator=_gen(seed))


def _ids(shape, cards, seed=3):
    g = _gen(seed)
    cols = [torch.randint(0, c, shape, generator=g) for c in cards]
    return torch.stack(cols, dim=-1)


# the 7-node fixture graph: a duplicate edge (2 -> 3 twice), a node with out-edges only (5)
# and an isolated node (6)
EDGES = [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2), (3, 4), (4, 3), (0, 2), (5, 4), (2, 3)]
N_NODES, NODE_F = 7, 5


def _graph(loop=False, weighted=False, seed=4):
    edges = EDGES + ([(2, 2)] if loop else [])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    g = _gen(seed)
    weight = (0.25 + torch.rand(edge_index.shape[1], generator=g)) if weighted else None
    x = torch.randn(N_NODES, NODE_F, generator=g)
    return x, edge_index, weight


def _pool_inputs(seed=5):
    x = torch.randn(12, 4, generator=_gen(seed))
    graph_index = torch.tensor([0] * 7 + [2] * 5)       # graph 1 is empty
    return x, graph_index, 3


# ------------------------------------------------------------------ the cases

@dataclass
class Case:
    """One block under test: how to build it (through `from_params`), what its forward
    takes, and the params and context `shape_rule` reads."""

    id: str
    type: str
    build: Callable[[], nn.Module]
    inputs: Callable[[], tuple]
    params: dict = field(default_factory=dict)
    context: dict = field(default_factory=dict)
    stochastic: bool = False          # training forward differs from eval by design
    width: int | None = None          # what `from_params` was called with (None: composite)
    build_params: dict | None = None
    f64_tol: float = 1e-9             # a block holding float32-computed constants: 1e-6

    @property
    def cls(self):
        return type(self.build()).__name__


def _from(cls_name, width, params):
    return lambda: getattr(B, cls_name).from_params(width, params)


def _simple(type_, width, params, inputs, cid=None, context=None, cls_name=None,
            stochastic=False):
    return Case(cid or type_, type_, _from(cls_name or B.CLASS_FOR[type_], width, params),
                inputs, dict(params), dict(context or {}), stochastic, width, dict(params))


def _randomise_norm(m):
    norm = m.bn if hasattr(m, "bn") else m.ln
    g = _gen(11)
    with torch.no_grad():
        norm.weight.copy_(0.5 + torch.rand(norm.weight.shape, generator=g))
        norm.bias.copy_(torch.randn(norm.bias.shape, generator=g))
        if hasattr(m, "bn"):
            norm.running_mean.copy_(torch.randn(norm.running_mean.shape, generator=g))
            norm.running_var.copy_(0.5 + torch.rand(norm.running_var.shape, generator=g))
    return m


def _norm_case(type_, cid, x):
    width = x().shape[-1]

    def build():
        return _randomise_norm(getattr(B, B.CLASS_FOR[type_]).from_params(width, {}))
    return Case(cid, type_, build, lambda: (x(),), {}, {}, type_ == "batchnorm", width, {})


def _row_cases():
    x, xw = (lambda: _rows()), (lambda: _window())
    emb = {"columns": ["zone", "kind"], "dims": {"zone": 3, "kind": 4},
           "cardinalities": {"zone": 5, "kind": 12}}
    cases = [
        _simple("dense", 6, {"units": 5, "activation": "gelu"}, lambda: (x(),)),
        _simple("dense", 6, {"units": 4, "activation": "silu", "bias": False},
                lambda: (x(),), "dense-nobias"),
        _simple("dense", 6, {"units": 3, "activation": "leaky_relu"}, lambda: (xw(),),
                "dense-window"),
        _simple("head", 6, {"out": 3}, lambda: (x(),)),
        _simple("head", 6, {"out": 2, "bias": False, "activation": "sigmoid"},
                lambda: (x(),), "head-nobias"),
        _simple("head", 6, {"out": 4}, lambda: (x(),), "head-ordinal",
                cls_name="CumulativeHead"),
        _simple("head", 6, {"out": 1}, lambda: (x(),), "head-ordinal-k2",
                cls_name="CumulativeHead"),
        _simple("embedding", 0, emb, lambda: (_ids((5,), (5, 12)),)),
        _simple("embedding", 0, emb, lambda: (_ids((4, 7), (5, 12)),), "embedding-window"),
        _simple("concat", 6, {}, lambda: (x(), torch.randn(5, 7, generator=_gen(8))),
                context={"emb_width": 7}),
        _simple("concat", 6, {}, lambda: (xw(), torch.randn(4, 7, 3, generator=_gen(8))),
                "concat-window", context={"emb_width": 3}),
        _simple("concat", 6, {}, lambda: (x(), None), "concat-numeric-only"),
        _norm_case("batchnorm", "batchnorm", x),
        _norm_case("batchnorm", "batchnorm-window", xw),
        _norm_case("layernorm", "layernorm", x),
        _norm_case("layernorm", "layernorm-window", xw),
        _simple("dropout", 6, {"p": 0.3}, lambda: (x(),), stochastic=True),
        _simple("flatten", 6, {}, lambda: (xw(),)),
        _simple("reshape", 12, {"shape": [3, 4]}, lambda: (_rows(5, 12),)),
        _simple("pool", 6, {"kind": "mean"}, lambda: (xw(),)),
        _simple("pool", 6, {"kind": "max"}, lambda: (xw(),), "pool-max"),
        _simple("pool", 6, {"kind": "cls"}, lambda: (xw(),), "pool-cls"),
        _simple("pool", 6, {"kind": "mean"}, lambda: (x(),), "pool-row"),
        _simple("reparameterize", 6, {"z": 3}, lambda: (x(),), stochastic=True),
        _simple("decoder", 4, {"widths": [6, 8], "out": 7, "activation": "gelu"},
                lambda: (_rows(5, 4),)),
        _simple("decoder", 4, {"out": 5}, lambda: (_rows(5, 4),), "decoder-linear"),
    ]
    for name in ("none", "relu", "gelu", "silu", "tanh", "sigmoid", "leaky_relu"):
        cases.append(_simple("activation", 6, {"name": name}, lambda: (x(),),
                             f"activation-{name}"))
    return cases


def _composite_cases():
    def sequential():
        return B.Sequential.from_params(6, {"modules": [
            B.Dense(6, 5, "relu"), B.Dropout(0.2), _randomise_norm(B.LayerNorm(5))]})

    def residual():
        first = [B.Dense(6, 8), _randomise_norm(B.BatchNorm(8)), B.Activation("gelu"),
                 B.Dropout(0.2)]
        second = [B.Dense(8, 8, "tanh")]
        return B.Residual.from_params(6, {"modules": [first, second],
                                          "projections": [nn.Linear(6, 8), nn.Identity()]})

    def parallel(merge, widths):
        return lambda: B.Parallel.from_params(6, {"merge": merge, "modules": [
            B.Dense(6, w, act) for w, act in widths]})
    x = (lambda: (_rows(),))
    seq_children = [("dense", {"units": 5}), ("dropout", {}), ("layernorm", {})]
    res_children = [("dense", {"units": 8}), ("batchnorm", {}), ("activation", {}),
                    ("dropout", {})]
    return [
        Case("sequential", "sequential", sequential, x, {"children": seq_children},
             stochastic=True),
        Case("residual", "residual", residual, x,
             {"repeats": [res_children, [("dense", {"units": 8})]]}, stochastic=True),
        Case("parallel", "parallel", parallel("concat", [(3, "relu"), (4, "tanh")]), x,
             {"children": [("dense", {"units": 3}), ("dense", {"units": 4})]}),
        Case("parallel-sum", "parallel", parallel("sum", [(4, "none"), (4, "gelu")]), x,
             {"children": [("dense", {"units": 4}), ("dense", {"units": 4})], "merge": "sum"}),
    ]


def _f32_table(case):
    """The sinusoidal table is computed in float32 by the block and exactly by the oracle."""
    case.f64_tol = 1e-6
    return case


def _sequence_cases():
    xw = (lambda: (_window(),))
    x8 = (lambda: (_window(f=8),))
    cases = [
        _simple("conv1d", 6, {"channels": 5, "kernel": 3}, xw),
        _simple("conv1d", 6, {"channels": 4, "kernel": 4}, xw, "conv1d-even-same"),
        _simple("conv1d", 6, {"channels": 4, "kernel": 3, "stride": 2}, xw, "conv1d-stride"),
        _simple("conv1d", 6, {"channels": 4, "kernel": 2, "padding": 0}, xw, "conv1d-valid"),
        _simple("conv1d", 6, {"channels": 3, "kernel": 3, "padding": 1}, xw, "conv1d-pad1"),
        _simple("pool1d", 6, {"kind": "max", "size": 2}, xw),
        _simple("pool1d", 6, {"kind": "avg", "size": 3}, xw, "pool1d-avg"),
        _simple("attention", 8, {"heads": 2, "dim": 8}, x8),
        _simple("attention", 8, {"heads": 2, "dim": 8, "pool": True}, x8, "attention-pool"),
        _simple("attention_pool", 6, {"hidden": 5}, xw),
        _simple("transformer_encoder", 8, {"d": 8, "heads": 2, "layers": 2, "ff": 16,
                                           "dropout": 0.1}, x8, stochastic=True),
        _f32_table(_simple("positional", 8, {"kind": "sinusoidal"}, x8)),
        _f32_table(_simple("positional", 7, {"kind": "sinusoidal"}, (lambda: (_window(f=7),)),
                           "positional-odd")),
        _simple("positional", 8, {"kind": "learned"}, x8, "positional-learned"),
        _simple("feature_tokenizer", 6, {"d": 8}, (lambda: (_rows(), None)),
                "feature_tokenizer-numeric"),
        _simple("feature_tokenizer", 6, {"d": 8, "cardinalities": [5, 12]},
                (lambda: (_rows(), _ids((5,), (5, 12)))), context={"n_cat": 2}),
    ]
    variants = [
        ("rnn", {"hidden": 4}), ("rnn", {"hidden": 4, "nonlinearity": "relu", "return": "mean"}),
        ("rnn", {"hidden": 3, "bidirectional": True, "num_layers": 2}),
        ("rnn", {"hidden": 3, "return": "all"}),
        ("gru", {"hidden": 4}), ("gru", {"hidden": 4, "bidirectional": True}),
        ("gru", {"hidden": 4, "return": "max"}),
        ("gru", {"hidden": 3, "bidirectional": True, "layers": 2, "return": "all",
                 "dropout": 0.2}),
        ("lstm", {"hidden": 4}), ("lstm", {"hidden": 3, "bidirectional": True, "num_layers": 2}),
        ("lstm", {"hidden": 4, "return": "mean"}), ("lstm", {"hidden": 4, "return": "all"}),
    ]
    for k, (kind, params) in enumerate(variants):
        params = dict(params, kind=kind)
        stochastic = params.get("dropout", 0) > 0 and int(
            params.get("num_layers", params.get("layers", 1))) > 1
        cases.append(_simple(kind, 6, params, xw, f"{kind}-{k}", stochastic=stochastic))
    return cases


def _graph_cases():
    def g(loop=False, weighted=False):
        return lambda: _graph(loop, weighted)
    specs = [
        ("gcn_conv", "gcn-sym", {"out": 4}, g()),
        ("gcn_conv", "gcn-rw", {"out": 4, "normalize": "rw"}, g()),
        ("gcn_conv", "gcn-sym-weighted", {"out": 4}, g(weighted=True)),
        ("gcn_conv", "gcn-rw-weighted-loop", {"out": 3, "normalize": "rw"}, g(True, True)),
        ("gcn_conv", "gcn-noloops", {"out": 4, "self_loops": False}, g(weighted=True)),
        ("gcn_conv", "gcn-data-loop", {"out": 4, "bias": False}, g(loop=True)),
        ("sage_conv", "sage-mean", {"out": 4}, g()),
        ("sage_conv", "sage-mean-weighted", {"out": 4}, g(weighted=True)),
        ("sage_conv", "sage-max", {"out": 4, "aggregator": "max"}, g(weighted=True)),
        ("sage_conv", "sage-sum", {"out": 4, "aggregator": "sum"}, g(weighted=True)),
        ("sage_conv", "sage-noroot-norm", {"out": 3, "root_weight": False, "normalize": True},
         g(loop=True)),
        ("gat_conv", "gat-concat", {"out": 3, "heads": 2}, g()),
        ("gat_conv", "gat-mean", {"out": 3, "heads": 2, "concat": False}, g(weighted=True)),
        ("gat_conv", "gat-noloops", {"out": 2, "heads": 3, "self_loops": False,
                                     "negative_slope": 0.1}, g(weighted=True)),
        ("gat_conv", "gat-dropout", {"out": 3, "heads": 1, "attn_dropout": 0.5}, g(loop=True)),
        ("gin_conv", "gin", {"out": 4, "eps": 0.1}, g()),
        ("gin_conv", "gin-train-eps", {"out": 4, "eps": -0.2, "train_eps": True},
         g(weighted=True)),
        ("gin_conv", "gin-1layer", {"out": 3, "mlp_layers": 1}, g(loop=True)),
        ("gin_conv", "gin-3layers", {"out": 3, "mlp_layers": 3}, g()),
    ]
    cases = [_simple(t, NODE_F, p, inputs, cid, stochastic="dropout" in cid)
             for t, cid, p, inputs in specs]
    for readout in ("sum", "mean", "max"):
        cases.append(_simple("graph_pool", 4, {"readout": readout}, _pool_inputs,
                             f"graph_pool-{readout}", context={"num_graphs": 3}))
    cases.append(_simple("edge_dropout", 0, {"p": 0.3},
                         lambda: _graph(weighted=True)[1:], stochastic=True))
    cases.append(_simple("edge_dropout", 0, {"p": 0.3}, lambda: (_graph()[1], None),
                         "edge_dropout-unweighted", stochastic=True))
    return cases


CASES = _row_cases() + _composite_cases() + _sequence_cases() + _graph_cases()
GRAPH_CONVS = ("gcn_conv", "sage_conv", "gat_conv", "gin_conv")


def case_ids():
    return [c.id for c in CASES]


# ------------------------------------------------------------------ helpers

def _cast(args, dtype):
    return tuple(a.to(dtype) if torch.is_tensor(a) and a.is_floating_point() else a
                 for a in args)


def _numpy(args):
    out = tuple(a.detach().numpy() if torch.is_tensor(a) else a for a in args)
    return out if len(out) > 1 else out[0]


def _outputs(out):
    return list(out) if isinstance(out, tuple) else [out]


def _within(got, want, rtol, atol):
    got, want = np.asarray(got, dtype=np.float64), np.asarray(want, dtype=np.float64)
    assert got.shape == want.shape, f"shape {got.shape} != oracle {want.shape}"
    bound = rtol * np.abs(want) + atol * np.maximum(1.0, np.abs(want))
    worst = float(np.max(np.abs(got - want) - bound)) if got.size else -1.0
    assert worst <= 0.0, f"max |block - oracle| exceeds the bound by {worst:.3e}"


def _check_against_oracle(case, dtype, rtol, atol):
    torch.manual_seed(0)
    module = case.build().to(dtype).eval()
    args = _cast(case.inputs(), dtype)
    with torch.no_grad():
        got = module(*args)
    weights, params = O.weights_of(module, case.type)
    want = O.oracle(case.type)(weights, _numpy(args), params)
    for g, w in zip(_outputs(got), _outputs(want)):
        if g is None or w is None:
            assert g is None and w is None
            continue
        _within(g.detach().numpy(), w, rtol, atol)
    return module, args, got


def _primary_shape(case, args):
    return tuple(args[0].shape) if args[0] is not None else tuple(args[1].shape)


# ------------------------------------------------------------------ oracles

@pytest.mark.parametrize("case", CASES, ids=case_ids())
def test_block_equals_its_float64_oracle(case):
    _, args, got = _check_against_oracle(case, torch.float64, case.f64_tol, case.f64_tol)
    want_shape = O.shape_rule(case.type, case.params, _primary_shape(case, args), case.context)
    assert tuple(_outputs(got)[0].shape) == want_shape


@pytest.mark.parametrize("case", CASES, ids=case_ids())
def test_block_float32_within_lane_tolerance(case):
    _check_against_oracle(case, torch.float32, 1e-4, 1e-5)


def test_every_catalog_type_has_an_oracle_or_is_shape_only():
    catalog = set(CATALOG["blocks"])
    assert set(O.BLOCK_TYPES) == catalog, sorted(set(O.BLOCK_TYPES) ^ catalog)
    assert {c.type for c in CASES} == catalog - O.SHAPE_ONLY
    with pytest.raises(KeyError, match="no value oracle"):
        O.oracle("custom")


def test_custom_is_checked_by_its_declared_shape():
    params = {"code_ref": "custom/gain.py", "out_shape": [3]}
    assert O.shape_rule("custom", params, (5, 6)) == (5, 3)
    assert O.check_custom(np.zeros((5, 3)), params)
    assert not O.check_custom(np.full((5, 3), np.inf), params)
    assert B.SEGMENTS_FOR["custom"] == [] and "custom" not in B.CLASS_FOR


# ------------------------------------------------------------------ graph blocks

def _graph_case(cid):
    return next(c for c in CASES if c.id == cid)


def _sorted_edges(src, dst, coef):
    src, dst = np.asarray(src), np.asarray(dst)
    coef = np.asarray(coef, dtype=np.float64).reshape(len(src), -1)
    order = np.lexsort((coef[:, 0], src, dst))
    return src[order], dst[order], coef[order]


def _conv_from_coefficients(module, kind, x, edge_index, edge_weight):
    """The conv output rebuilt at every node from `coefficients` alone."""
    n = x.shape[0]
    if kind == "gat_conv":
        src, dst, alpha = module.coefficients(x, edge_index, edge_weight)
        hx = module.lin(x).view(n, module.heads, -1)
        out = torch.zeros(n, module.heads, hx.shape[-1]).index_add(
            0, dst, hx[src] * alpha.unsqueeze(-1))
        out = out.reshape(n, -1) if module.concat else out.mean(dim=1)
        return out if module.bias is None else out + module.bias
    src, dst, coef = module.coefficients(edge_index, n, edge_weight)
    if kind == "gcn_conv":
        out = torch.zeros(n, module.lin.out_features).index_add(
            0, dst, module.lin(x)[src] * coef.unsqueeze(-1))
        return out if module.bias is None else out + module.bias
    msg = x[src] * coef.unsqueeze(-1)
    if kind == "gin_conv":
        return module.mlp(torch.zeros_like(x).index_add(0, dst, msg))
    if module.aggregator == "max":
        agg = torch.stack([msg[dst == i].max(dim=0).values if (dst == i).any()
                           else torch.zeros(x.shape[1]) for i in range(n)])
    else:
        agg = torch.zeros_like(x).index_add(0, dst, msg)
    out = module.lin_neigh(agg) + (0 if module.lin_root is None else module.lin_root(x))
    return nn.functional.normalize(out, dim=-1) if module.normalize else out


@pytest.mark.parametrize("case", [c for c in CASES if c.type in GRAPH_CONVS],
                         ids=[c.id for c in CASES if c.type in GRAPH_CONVS])
def test_coefficients_reproduce_the_conv_at_every_target(case):
    torch.manual_seed(0)
    module = case.build().eval()
    x, edge_index, edge_weight = case.inputs()
    with torch.no_grad():
        direct = module(x, edge_index, edge_weight)
        rebuilt = _conv_from_coefficients(module, case.type, x, edge_index, edge_weight)
        got = (module.coefficients(x, edge_index, edge_weight) if case.type == "gat_conv"
               else module.coefficients(edge_index, x.shape[0], edge_weight))
    assert torch.allclose(direct, rebuilt, atol=1e-5, rtol=0), case.id
    weights, params = O.weights_of(module, case.type)
    want = O.coefficients(case.type, weights, (x.numpy(), edge_index.numpy(),
                                               None if edge_weight is None
                                               else edge_weight.numpy()), params)
    got_sorted = _sorted_edges(*(t.numpy() for t in got))
    want_sorted = _sorted_edges(*want)
    assert np.array_equal(got_sorted[0], want_sorted[0])
    assert np.array_equal(got_sorted[1], want_sorted[1])
    assert np.allclose(got_sorted[2], want_sorted[2], atol=1e-6)


def test_gcn_equals_the_dense_normalised_adjacency():
    """D1 §17.1: D^-½(A+I)D^-½ X W + b (sym) and D^-1(A+I) X W + b (rw), weighted, written
    out here as matrices, independently of the oracle module."""
    x, edge_index, weight = _graph(weighted=True)
    a = torch.zeros(N_NODES, N_NODES, dtype=torch.float64)
    a.index_put_((edge_index[1], edge_index[0]), weight.double(), accumulate=True)
    a_hat = a + torch.eye(N_NODES, dtype=torch.float64)
    deg = a_hat.sum(dim=1)
    for normalize, norm in (("sym", torch.diag(deg ** -0.5) @ a_hat @ torch.diag(deg ** -0.5)),
                            ("rw", torch.diag(1.0 / deg) @ a_hat)):
        m = B.GCNConv(NODE_F, 3, normalize=normalize).double().eval()
        with torch.no_grad():
            m.bias.copy_(torch.randn(3, dtype=torch.float64))
            want = norm @ (x.double() @ m.lin.weight.T) + m.bias
            got = m(x.double(), edge_index, weight.double())
        assert torch.allclose(got, want, atol=1e-12), normalize


def test_gat_alpha_is_a_softmax_per_destination():
    torch.manual_seed(0)
    m = B.GATConv.from_params(NODE_F, {"out": 3, "heads": 2}).eval()
    x, edge_index, _ = _graph()
    with torch.no_grad():
        src, dst, alpha = m.coefficients(x, edge_index)
    sums = torch.zeros(N_NODES, 2).index_add(0, dst, alpha)
    assert torch.allclose(sums, torch.ones(N_NODES, 2), atol=1e-6)   # every node has a loop
    assert bool((alpha >= 0).all())
    _, _, weighted = m.coefficients(x, edge_index, torch.full((edge_index.shape[1],), 0.5))
    # the weight multiplies α after the softmax, without renormalising (R-B3)
    assert torch.allclose(weighted[: edge_index.shape[1]], 0.5 * alpha[: edge_index.shape[1]])


def test_graph_pool_equals_a_pandas_groupby():
    pd = pytest.importorskip("pandas")
    x, graph_index, num_graphs = _pool_inputs()
    frame = pd.DataFrame(x.numpy().astype(np.float64))
    frame["g"] = graph_index.numpy()
    for readout in ("sum", "mean", "max"):
        got = B.GraphPool(readout)(x.double(), graph_index, num_graphs)
        want = frame.groupby("g").agg(readout).reindex(range(num_graphs), fill_value=0.0)
        assert np.allclose(got.numpy(), want.to_numpy(), atol=1e-12), readout
        assert got.shape == (num_graphs, x.shape[1])


def test_edge_dropout_is_the_identity_in_eval_and_keeps_one_minus_p_in_training():
    m = B.EdgeDropout(0.3)
    _, edge_index, weight = _graph(weighted=True)
    m.eval()
    kept_index, kept_weight = m(edge_index, weight)
    assert kept_index is edge_index and kept_weight is weight
    m.train()
    big = torch.randint(0, 50, (2, 20000), generator=_gen(6))
    big_weight = torch.arange(20000, dtype=torch.float32)
    torch.manual_seed(7)
    kept_index, kept_weight = m(big, big_weight)
    share = kept_index.shape[1] / big.shape[1]
    assert abs(share - 0.7) < 0.015, share
    # the weights are filtered with the edges: each kept weight still names its edge
    assert torch.equal(big[:, kept_weight.long()], kept_index)
    torch.manual_seed(7)
    again, _ = m(big, big_weight)
    assert torch.equal(again, kept_index)                  # seeded: deterministic


def test_graph_blocks_follow_their_input_size_under_a_trace():
    """A trace made on one graph runs on another (the TorchScript export, D1 Appendix A)."""
    torch.manual_seed(0)
    g = _gen(9)
    big_x = torch.randn(40, NODE_F, generator=g)
    big_edges = torch.randint(0, 40, (2, 160), generator=g)
    x, edge_index, weight = _graph(loop=True, weighted=True)
    for kind, params in (("gcn_conv", {"out": 4}), ("sage_conv", {"out": 4}),
                         ("gat_conv", {"out": 3, "heads": 2}), ("gin_conv", {"out": 4})):
        m = getattr(B, B.CLASS_FOR[kind]).from_params(NODE_F, params).eval()
        with torch.no_grad(), warnings.catch_warnings():
            warnings.simplefilter("ignore")           # the tracer's deprecation notices
            traced = torch.jit.trace(m, (big_x, big_edges, torch.ones(160)), check_trace=False)
            assert torch.allclose(traced(x, edge_index, weight), m(x, edge_index, weight),
                                  atol=1e-5), kind


def _pyg_twin(kind, m):
    """The torch_geometric layer with `m`'s weights copied in (D6 §14.1 item 6)."""
    pyg = pytest.importorskip("torch_geometric.nn")
    f = m.lin.in_features if hasattr(m, "lin") else (
        m.lin_neigh.in_features if hasattr(m, "lin_neigh") else m.mlp[0].in_features)
    if kind == "gcn_conv":
        twin = pyg.GCNConv(f, m.lin.out_features)
        pairs = [(twin.lin.weight, m.lin.weight), (twin.bias, m.bias)]
    elif kind == "sage_conv":
        twin = pyg.SAGEConv(f, m.lin_neigh.out_features, aggr=m.aggregator)
        pairs = [(twin.lin_l.weight, m.lin_neigh.weight), (twin.lin_l.bias, m.lin_neigh.bias),
                 (twin.lin_r.weight, m.lin_root.weight)]
    elif kind == "gat_conv":
        out = m.lin.out_features // m.heads
        twin = pyg.GATConv(f, out, heads=m.heads, concat=m.concat,
                           negative_slope=m.negative_slope)
        lin = getattr(twin, "lin", None) or twin.lin_src
        pairs = [(lin.weight, m.lin.weight), (twin.att_src, m.att_src.view(1, m.heads, out)),
                 (twin.att_dst, m.att_dst.view(1, m.heads, out)), (twin.bias, m.bias)]
    else:
        twin = pyg.GINConv(m.mlp, eps=float(m.eps))
        pairs = []
    with torch.no_grad():
        for dst, src in pairs:
            dst.copy_(src.reshape(dst.shape))
    return twin.eval()


@pytest.mark.parametrize("kind,params", [("gcn_conv", {"out": 4}), ("sage_conv", {"out": 4}),
                                         ("sage_conv", {"out": 4, "aggregator": "max"}),
                                         ("gat_conv", {"out": 3, "heads": 2}),
                                         ("gat_conv", {"out": 3, "heads": 2, "concat": False}),
                                         ("gin_conv", {"out": 4, "eps": 0.1})])
def test_graph_convs_match_torch_geometric_when_it_imports(kind, params):
    """Unweighted, loop-free input (PyG's SAGE and GIN take no edge weight, and its GAT
    replaces a loop already in the data); skipped, as D6 allows, without torch_geometric."""
    torch.manual_seed(0)
    m = getattr(B, B.CLASS_FOR[kind]).from_params(NODE_F, params).eval()
    twin = _pyg_twin(kind, m)
    x, edge_index, _ = _graph()
    with torch.no_grad():
        assert torch.allclose(m(x, edge_index), twin(x, edge_index), atol=1e-5), kind


# ------------------------------------------------------------------ recurrent blocks

@pytest.mark.parametrize("kind", ["rnn", "gru", "lstm"])
@pytest.mark.parametrize("bidirectional", [False, True])
@pytest.mark.parametrize("layers", [1, 2])
def test_recurrent_last_equals_h_n(kind, bidirectional, layers):
    torch.manual_seed(0)
    m = B.RecurrentBlock.from_params(6, {"kind": kind, "hidden": 4, "num_layers": layers,
                                         "bidirectional": bidirectional}).eval()
    x = _window()
    with torch.no_grad():
        last = m(x)
        _, state = m.rnn(x)
    h_n = state[0] if kind == "lstm" else state
    want = torch.cat([h_n[-2], h_n[-1]], dim=-1) if bidirectional else h_n[-1]
    assert torch.allclose(last, want, atol=1e-6)
    assert last.shape == (4, 4 * (2 if bidirectional else 1))


def test_recurrent_readouts_and_layer_count():
    x = _window()
    for ret, shape in (("all", (4, 7, 8)), ("mean", (4, 8)), ("max", (4, 8)), ("last", (4, 8))):
        m = B.RecurrentBlock.from_params(6, {"kind": "gru", "hidden": 4, "bidirectional": True,
                                             "return": ret}).eval()
        assert tuple(m(x).shape) == shape, ret
        assert m.ret == ret
    # D1 E7: the catalog's num_layers, and its old name layers, both reach the module
    assert B.RecurrentBlock.from_params(6, {"kind": "lstm", "hidden": 3,
                                            "num_layers": 3}).rnn.num_layers == 3
    assert B.RecurrentBlock.from_params(6, {"kind": "gru", "hidden": 3,
                                            "layers": 2}).rnn.num_layers == 2
    assert isinstance(B.RecurrentBlock.from_params(6, {"kind": "rnn", "hidden": 3}).rnn, nn.RNN)
    assert B.RecurrentBlock.from_params(6, {"kind": "rnn", "hidden": 3,
                                            "nonlinearity": "relu"}).rnn.nonlinearity == "relu"
    with pytest.raises(ValueError, match="return"):
        B.RecurrentBlock.from_params(6, {"kind": "gru", "hidden": 3, "return": "first"})
    with pytest.raises(ValueError, match="kind"):
        B.RecurrentBlock.from_params(6, {"hidden": 3})


# ------------------------------------------------------------------ attention, heads, tokens

def test_attention_pool_weights_sum_to_one():
    torch.manual_seed(0)
    m = B.AttentionPool.from_params(6, {"hidden": 5}).eval()
    x = _window()
    with torch.no_grad():
        alpha = m.weights(x)
        out = m(x)
    assert alpha.shape == (4, 7)
    assert bool((alpha >= 0).all())
    assert torch.allclose(alpha.sum(dim=-1), torch.ones(4), atol=1e-6)
    assert torch.allclose(out, (alpha.unsqueeze(-1) * x).sum(dim=1), atol=1e-6)
    w, _ = O.weights_of(m, "attention_pool")
    assert np.allclose(alpha.numpy(), O.attention_pool_weights(w, x.numpy()), atol=1e-6)


def test_cumulative_head_is_monotone_and_its_probabilities_are_valid():
    torch.manual_seed(3)
    m = B.CumulativeHead.from_params(6, {"out": 5})
    with torch.no_grad():
        m.gaps.copy_(torch.randn(4) * 3.0)                   # any raw gap, even negative
        m.first.copy_(torch.randn(1))
        z = m(torch.randn(200, 6) * 5.0)
    assert z.shape == (200, 5)
    assert bool((z[:, :-1] >= z[:, 1:]).all())                # z_0 >= z_1 >= ...
    p_gt = torch.sigmoid(z)
    probs = torch.cat([1 - p_gt[:, :1], p_gt[:, :-1] - p_gt[:, 1:], p_gt[:, -1:]], dim=1)
    assert bool((probs >= -1e-7).all())
    assert torch.allclose(probs.sum(dim=1), torch.ones(200), atol=1e-6)
    assert m.linear is m.score and m.score.bias is None and isinstance(m.act, nn.Identity)
    assert sorted(n for n, _ in m.named_parameters()) == ["first", "gaps", "score.weight"]
    with pytest.raises(ValueError):
        B.CumulativeHead(6, 0)


def test_head_bias_flag():
    assert B.Head.from_params(6, {"out": 3, "bias": False}).linear.bias is None
    assert B.Head.from_params(6, {"out": 3}).linear.bias is not None


def test_feature_tokenizer_with_nominals_and_the_v22_form():
    torch.manual_seed(0)
    m = B.FeatureTokenizer.from_params(6, {"d": 8, "cardinalities": [5, 12]})
    out = m(_rows(), _ids((5,), (5, 12)))
    assert out.shape == (5, 6 + 2 + 1, 8)
    with pytest.raises(ValueError, match="x_cat"):
        m(_rows())
    # v2.2: FeatureTokenizer(n, d) over the numeric columns, exactly the c849f8b formula
    torch.manual_seed(0)
    old = B.FeatureTokenizer(6, 8)
    x = _rows()
    tok = x.unsqueeze(-1) * old.weight.unsqueeze(0) + old.bias.unsqueeze(0)
    want = torch.cat([old.cls.expand(x.shape[0], -1, -1), tok], dim=1)
    assert torch.equal(old(x), want)
    assert len(old.embeddings) == 0


def test_concat_names_both_shapes_on_a_mismatch():
    with pytest.raises(ValueError, match=r"\(4, 7, 6\).*\(4, 3\)"):
        B.Concat()(_window(), torch.randn(4, 3))


def test_reparameterize_keeps_kl_in_training_only():
    m = B.Reparameterize(3)
    x = _rows()
    m.eval()
    assert torch.equal(m(x), x[:, :3]) and m.kl is None
    m.train()
    torch.manual_seed(0)
    m(x)
    assert m.kl is not None and torch.allclose(m.kl, m.divergence(x))
    kl = m.kl
    m.eval()
    m(x)
    assert m.kl is kl                                          # eval wrote nothing


# ------------------------------------------------------------------ R-B4: no state in a forward

class _SetattrSpy:
    """Records every nn.Module attribute write while active."""

    def __init__(self):
        self.writes = []

    def __enter__(self):
        self._orig = nn.Module.__setattr__
        spy = self

        def record(module, name, value):
            spy.writes.append((type(module).__name__, name))
            spy._orig(module, name, value)
        nn.Module.__setattr__ = record
        return self

    def __exit__(self, *exc):
        nn.Module.__setattr__ = self._orig
        return False


def _state(module):
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


@pytest.mark.parametrize("grad", [False, True], ids=["no_grad", "enable_grad"])
@pytest.mark.parametrize("case", CASES, ids=case_ids())
def test_an_eval_forward_writes_nothing(case, grad):
    torch.manual_seed(0)
    module = case.build().eval()
    before = _state(module)
    args = case.inputs()
    with _SetattrSpy() as spy, torch.set_grad_enabled(grad):
        module(*args)
        if case.type in GRAPH_CONVS:
            if case.type == "gat_conv":
                module.coefficients(*args)
            else:
                module.coefficients(args[1], args[0].shape[0], args[2])
        if case.type == "attention_pool":
            module.weights(*args)
    assert spy.writes == [], spy.writes
    after = _state(module)
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert all(p.grad is None for p in module.parameters())


def test_the_spy_sees_a_training_write():
    """The positive control: a training forward of reparameterize writes `kl`."""
    m = B.Reparameterize(3).train()
    with _SetattrSpy() as spy:
        m(_rows())
    assert ("Reparameterize", "kl") in spy.writes


@pytest.mark.parametrize("case", CASES, ids=case_ids())
def test_no_block_keeps_a_last_attribute(case):
    module = case.build()
    for sub in module.modules():
        names = set(vars(sub)) | set(dir(type(sub)))
        assert not [n for n in names if n.startswith("last_")], type(sub).__name__


# ------------------------------------------------------------------ determinism and gradients

def _one_step(case):
    torch.manual_seed(0)
    module = case.build().train()
    args = case.inputs()
    torch.manual_seed(1)
    out = _outputs(module(*args))[0]
    if not out.is_floating_point():                            # edge_dropout: an index
        return module, None
    target = torch.randn(out.shape, generator=_gen(13), dtype=out.dtype)
    loss = (out * target).sum()
    if loss.requires_grad:
        loss.backward()
    return module, loss.detach()


@pytest.mark.parametrize("case", CASES, ids=case_ids())
def test_two_seeded_cpu_runs_are_identical(case):
    was = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        first, loss_1 = _one_step(case)
        second, loss_2 = _one_step(case)
    finally:
        torch.use_deterministic_algorithms(was)
    s1, s2 = first.state_dict(), second.state_dict()
    assert s1.keys() == s2.keys() and all(torch.equal(s1[k], s2[k]) for k in s1)
    assert (loss_1 is None) == (loss_2 is None)
    if loss_1 is not None:
        assert torch.equal(loss_1, loss_2)
    for (name, p1), (_, p2) in zip(first.named_parameters(), second.named_parameters()):
        assert (p1.grad is None) == (p2.grad is None), name
        if p1.grad is not None:
            assert torch.equal(p1.grad, p2.grad), name


@pytest.mark.parametrize("case", [c for c in CASES if any(True for _ in c.build().parameters())],
                         ids=[c.id for c in CASES if any(True for _ in c.build().parameters())])
def test_every_parameter_gets_a_finite_gradient(case):
    module, _ = _one_step(case)
    grads = [(n, p.grad) for n, p in module.named_parameters() if p.numel()]
    missing = [n for n, g in grads if g is None]
    assert not missing, missing
    assert all(bool(torch.isfinite(g).all()) for _, g in grads)
    assert any(bool((g != 0).any()) for _, g in grads)


@pytest.mark.parametrize("case", CASES, ids=case_ids())
def test_eval_is_deterministic_and_train_differs_only_by_design(case):
    torch.manual_seed(0)
    module = case.build().eval()
    args = case.inputs()
    with torch.no_grad():
        a, b = _outputs(module(*args))[0], _outputs(module(*args))[0]
        assert torch.equal(a, b)
        module.train()
        torch.manual_seed(5)
        t = _outputs(module(*args))[0]
    if not case.stochastic:
        assert t.shape == a.shape and torch.allclose(t, a, atol=1e-6), case.id


# ------------------------------------------------------------------ MAdds

def test_madds_of_the_recurrent_and_graph_blocks():
    m = B.RecurrentBlock.from_params(6, {"kind": "gru", "hidden": 4, "num_layers": 2,
                                         "bidirectional": True})
    # layer 0: 3 gates · 2 dirs · L · 4 · (6 + 4); layer 1 reads 8 features
    assert m.madds(10, 0, 7) == 10 * (3 * 2 * 7 * 4 * (6 + 4) + 3 * 2 * 7 * 4 * (8 + 4))
    rnn = B.RecurrentBlock.from_params(6, {"kind": "rnn", "hidden": 4})
    assert rnn.madds(1, 0, 5) == 5 * 4 * (6 + 4)             # E19: nn.RNN is counted
    pool = B.AttentionPool(6, 5)
    assert pool.madds(2, 0, 7) == 2 * 7 * (6 * 5 + 5 + 6)
    gcn = B.GCNConv(5, 4)
    assert gcn.madds(7, 11, 1) == 7 * 5 * 4 + (11 + 7) * 4
    for m in (B.SAGEConv.from_params(5, {"out": 4}), B.GATConv.from_params(5, {"out": 3,
                                                                               "heads": 2}),
              B.GINConv(5, 4)):
        assert isinstance(m.madds(7, 11, 1), int) and m.madds(7, 11, 1) > 0
    assert B.GraphPool("sum").madds(7) == 0 and B.EdgeDropout(0.2).madds(7, 11) == 0


# ------------------------------------------------------------------ the emitter's maps

def _literal(name):
    """A map read exactly as emit.py reads SEGMENTS_FOR: by regex and literal_eval, no import."""
    text = BLOCKS_PY.read_text(encoding="utf-8")
    m = re.search(rf"^{name}\s*=\s*(\{{[\s\S]*?\n\}})", text, re.M)
    assert m, f"{name} is not a literal dict at the top level of blocks.py"
    return ast.literal_eval(m.group(1))


def _segments():
    text = BLOCKS_PY.read_text(encoding="utf-8")
    names = re.findall(r"^#\s*--8<--\s*(\w+)\s*$", text, re.M)
    body = {}
    for name in names:
        m = re.search(rf"#\s*--8<--\s*{name}\s*\n([\s\S]*?)\n\s*#\s*-->8--\s*{name}\b", text)
        assert m, f"segment {name} has no closing marker"
        body[name] = m.group(1)
    return names, body


def test_every_catalog_type_is_mapped():
    catalog = set(CATALOG["blocks"])
    segments_for, segment_file = _literal("SEGMENTS_FOR"), _literal("SEGMENT_FILE")
    assert segments_for == B.SEGMENTS_FOR and segment_file == B.SEGMENT_FILE
    assert set(segments_for) == catalog
    assert set(B.STAGE_KIND_FOR) == catalog
    assert set(B.CLASS_FOR) == catalog - {"custom"}
    assert set(B.STAGE_KIND_FOR.values()) == {"emb", "concat", "tokens", "tensor", "graph",
                                               "gdrop", "gpool"}
    names, _ = _segments()
    assert set(segment_file) == set(names), set(segment_file) ^ set(names)
    assert set(segment_file.values()) == SUPPORT_FILES
    used = {s for segs in segments_for.values() for s in segs}
    assert used <= set(names)
    for type_, cls_names in [(t, [c]) for t, c in B.CLASS_FOR.items()] + \
            [(t, list(v.values())) for t, v in B.CLASS_VARIANTS.items()]:
        for cls_name in cls_names:
            assert hasattr(B, cls_name), cls_name
            where = [s for s in segments_for[type_] if f"class {cls_name}(" in _segments()[1][s]]
            assert where, f"{cls_name} is not defined in a segment of {type_}"


def _file_text(target, names, body):
    """An emitted support file rebuilt from its segments, prelude first (what every emitted
    file needs, whatever header the emitter writes)."""
    parts = [body["prelude"]] + [body[n] for n in names
                                 if n != "prelude" and B.SEGMENT_FILE[n] == target]
    return "from __future__ import annotations\n\n" + "\n\n\n".join(parts) + "\n"


@pytest.mark.parametrize("target", sorted(SUPPORT_FILES))
def test_each_support_file_stands_alone(target):
    names, body = _segments()
    module_name = f"emitted_{target}"
    emitted = types.ModuleType(module_name)
    sys.modules[module_name] = emitted            # dataclasses resolve their module by name
    try:
        exec(compile(_file_text(target, names, body), f"<{target}.py>", "exec"),
             emitted.__dict__)
    finally:
        sys.modules.pop(module_name, None)
    namespace = emitted.__dict__
    ran = 0
    for case in CASES:
        segs = B.SEGMENTS_FOR[case.type]
        own = [s for s in segs if B.SEGMENT_FILE[s] == target]
        if not own or case.type in ("sequential", "residual", "parallel"):
            continue
        cls = namespace[case.cls]
        original = case.build()
        module = cls.from_params(case.width, case.build_params)
        module.load_state_dict(original.state_dict())
        module.eval()
        original.eval()
        with torch.no_grad():
            for g, w in zip(_outputs(module(*case.inputs())), _outputs(original(*case.inputs()))):
                assert (g is None and w is None) or torch.equal(g, w), case.id
        ran += 1
    assert ran > 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
