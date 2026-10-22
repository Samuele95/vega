#!/usr/bin/env python3
"""
block_oracles.py — a float64 numpy oracle for every catalog block type (33), written from
the formulas, never from the block code.

    ORACLES[type](weights, x, params) -> np.ndarray      (32 types; `custom` is shape-only)
    shape_rule(type, params, in_shape, context=None) -> tuple
    coefficients(type, weights, x, params) -> (src, dst, coef)   (the four graph convs)
    weights_of(module, type) -> (weights, params)       (a torch block, read as the oracle reads)

This is the only oracle module of the release (BUILD-PLAN R-H4): `test_blocks.py` checks every
block against it, and `vega/scripts/tests/oracles.py` re-exports it for the lane checks,
which fill `weights` from `GET /v1/network` instead of from a module.

Arguments. `x` is what the block's forward receives, positionally (its stage kind, R-B1): one
array for a `tensor` block, and a tuple for the others — `concat` (x_num, emb), `embedding`
x_cat, `feature_tokenizer` (x_num, x_cat), a graph conv (h, edge_index, edge_weight),
`graph_pool` (h, graph_index, num_graphs), `edge_dropout` (edge_index, edge_weight).
`weights` maps a role name (the tensor roles of inspect-contract §5, `vega/network-view@2`) to
a float64 array; `params` holds the block's resolved configuration. Composite blocks carry
their children as `(type, weights, params)` triples in `params`. The eval-mode behaviour is
the oracle's: dropout, edge dropout and the reparameterisation noise are inactive.

Role names per type (what `weights_of` produces and the lane checks must supply):

| type | weights | params |
|---|---|---|
| dense, head | weight (out, in), bias (out) or absent | activation |
| head (ordinal) | weight (1, in) (the `score` projection), first (1,), gaps (K−2,) | cumulative: true |
| embedding | table.<column> (card, dim) | columns |
| batchnorm | gamma, beta, running_mean, running_var | eps |
| layernorm | gamma, beta | eps |
| conv1d | kernel (C', C, k), bias (C') | stride, padding (int or "same") |
| pool1d | — | kind, size |
| rnn, gru, lstm | l<k>.w_ih, l<k>.w_hh, l<k>.b_ih, l<k>.b_hh (+ ".reverse") | kind, hidden, num_layers, bidirectional, return, nonlinearity |
| attention | in_proj (3d, d), in_proj_bias, out_proj (d, d), out_proj_bias | heads, pool |
| attention_pool | score (k, H), score_bias (k), query (1, k) | — |
| transformer_encoder | layer<k>.{in_proj, in_proj_bias, out_proj, out_proj_bias, linear1, linear1_bias, linear2, linear2_bias, norm1_gamma, norm1_beta, norm2_gamma, norm2_beta} | heads, layers, eps, activation |
| feature_tokenizer | weight (n, d), bias (n, d), cls (d,), table.<j> (card, d) | — |
| positional | pe (L, d) (learned only) | kind |
| decoder | layer<k>.weight, layer<k>.bias | activation |
| gcn_conv | weight (out, in), bias | normalize, self_loops |
| sage_conv | neigh (out, in), neigh_bias, root (out, in) or absent | aggregator, normalize |
| gat_conv | weight (heads·out, in), att_src (heads, out), att_dst, bias | heads, out, concat, negative_slope, self_loops |
| gin_conv | mlp.<k>.weight, mlp.<k>.bias, eps (1,) | — |
| activation, pool, reshape, graph_pool, reparameterize | — | name / kind / shape / readout / z |

The graph oracles are dense-matrix formulations (A[i, j] = the summed weight of the edges
j -> i), with the self-loop rule of the blocks: a unit loop on every node that has none, an
existing loop keeping its weight. GAT and SAGE-max, which a matrix cannot express with
duplicate edges, loop over the edges per destination.
"""
from __future__ import annotations

import math
from types import MappingProxyType
from typing import Any, Callable, Mapping

import numpy as np

try:  # the exact gelu needs erf; scipy ships with scikit-learn, a backend requirement
    from scipy.special import erf as _erf
except ImportError:  # pragma: no cover - the fallback is exact too, only slower
    _erf = np.vectorize(math.erf)


# ------------------------------------------------------------------ elementwise pieces

def activation(name, z):
    """The catalog activation `name` applied to `z` (gelu is the exact, erf one: torch's
    default; leaky_relu has slope 0.01)."""
    z = np.asarray(z, dtype=np.float64)
    fns = {
        "none": lambda v: v,
        "relu": lambda v: np.maximum(v, 0.0),
        "gelu": lambda v: 0.5 * v * (1.0 + _erf(v / math.sqrt(2.0))),
        "silu": lambda v: v * _sigmoid(v),
        "tanh": np.tanh,
        "sigmoid": _sigmoid,
        "leaky_relu": lambda v: np.where(v >= 0, v, 0.01 * v),
    }
    if (name or "none") not in fns:
        raise KeyError(f"no oracle for activation {name!r}")
    return fns[name or "none"](z)


def _sigmoid(v):
    return 1.0 / (1.0 + np.exp(-v))


def _softmax(v, axis=-1):
    e = np.exp(v - v.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def _softplus(v):
    return np.log1p(np.exp(-np.abs(v))) + np.maximum(v, 0.0)


def _f64(a):
    return None if a is None else np.asarray(a, dtype=np.float64)


def affine(weights, x):
    """`x · Wᵀ + b`, the pre-activation of a dense or head block."""
    z = _f64(x) @ _f64(weights["weight"]).T
    return z + _f64(weights["bias"]) if weights.get("bias") is not None else z


# ------------------------------------------------------------------ row blocks

def _dense(w, x, p):
    return activation(p.get("activation", "none"), affine(w, x))


def _head(w, x, p):
    if "gaps" not in w:
        return _dense(w, x, p)
    steps = np.concatenate([[0.0], _softplus(_f64(w["gaps"]))]).cumsum()
    thresholds = _f64(w["first"])[0] - steps
    return _f64(x) @ _f64(w["weight"]).T + thresholds


def _embedding(w, x_cat, p):
    ids = np.asarray(x_cat)
    parts = [_f64(w[f"table.{c}"])[ids[..., i]] for i, c in enumerate(p["columns"])]
    return np.concatenate(parts, axis=-1)


def _concat(w, x, p):
    x_num, emb = x
    if emb is None:
        return _f64(x_num)
    if x_num is None:
        return _f64(emb)
    return np.concatenate([_f64(x_num), _f64(emb)], axis=-1)


def _batchnorm(w, x, p):
    scale = _f64(w["gamma"]) / np.sqrt(_f64(w["running_var"]) + float(p.get("eps", 1e-5)))
    return (_f64(x) - _f64(w["running_mean"])) * scale + _f64(w["beta"])


def _layernorm(w, x, p):
    x = _f64(x)
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    y = (x - mean) / np.sqrt(var + float(p.get("eps", 1e-5)))
    return y * _f64(w["gamma"]) + _f64(w["beta"])


def _identity(w, x, p):
    return _f64(x)


def _activation(w, x, p):
    return activation(p.get("name", "relu"), x)


def _flatten(w, x, p):
    x = _f64(x)
    return x.reshape(x.shape[0], -1)


def _reshape(w, x, p):
    x = _f64(x)
    return x.reshape((x.shape[0],) + tuple(int(s) for s in p["shape"]))


def _pool(w, x, p):
    x = _f64(x)
    if x.ndim == 2:
        return x
    kind = p.get("kind", "mean")
    if kind == "mean":
        return x.mean(axis=1)
    if kind == "max":
        return x.max(axis=1)
    return x[:, 0]


def _run(child, x):
    kind, weights, params = child
    return ORACLES[kind](weights, x, params)


def _sequential(w, x, p):
    for child in p["children"]:
        x = _run(child, x)
    return _f64(x)


def _residual(w, x, p):
    x = _f64(x)
    for chain, projection in zip(p["repeats"], p["projections"]):
        h = x
        for child in chain:
            h = _run(child, h)
        x = (x if projection is None else affine(projection, x)) + h
    return x


def _parallel(w, x, p):
    outs = [_run(child, x) for child in p["children"]]
    if p.get("merge", "concat") == "sum":
        return np.sum(outs, axis=0)
    return np.concatenate(outs, axis=-1)


def _decoder(w, x, p):
    x = _f64(x)
    n = len([k for k in w if k.endswith(".weight")])
    for k in range(n):
        x = affine({"weight": w[f"layer{k}.weight"], "bias": w.get(f"layer{k}.bias")}, x)
        if k < n - 1:
            x = activation(p.get("activation", "gelu"), x)
    return x


def _reparameterize(w, x, p):
    return _f64(x)[..., : int(p["z"])]


# ------------------------------------------------------------------ sequence blocks

def _conv1d(w, x, p):
    """A direct correlation loop over (B, L, C) with torch's padding rule."""
    x, kernel = _f64(x), _f64(w["kernel"])
    out_c, _, k = kernel.shape
    stride, padding = int(p.get("stride", 1)), p.get("padding", 0)
    if padding == "same":
        left = (k - 1) // 2
        x = np.pad(x, ((0, 0), (left, k - 1 - left), (0, 0)))
    elif int(padding):
        x = np.pad(x, ((0, 0), (int(padding), int(padding)), (0, 0)))
    steps = (x.shape[1] - k) // stride + 1
    y = np.zeros((x.shape[0], steps, out_c))
    for t in range(steps):
        window = x[:, t * stride: t * stride + k, :]          # (B, k, C)
        y[:, t, :] = np.einsum("bkc,ock->bo", window, kernel)
    return y + (_f64(w["bias"]) if w.get("bias") is not None else 0.0)


def _pool1d(w, x, p):
    x = _f64(x)
    size = int(p.get("size", 2))
    steps = x.shape[1] // size
    blocks = x[:, : steps * size, :].reshape(x.shape[0], steps, size, x.shape[2])
    return blocks.max(axis=2) if p.get("kind", "max") == "max" else blocks.mean(axis=2)


def _cell(kind, p, gates, h, c):
    """One recurrent step in torch's gate layout; `gates` = (W_ih x + b_ih, W_hh h + b_hh)."""
    gi, gh = gates
    if kind == "rnn":
        pre = gi + gh
        return (np.tanh(pre) if p.get("nonlinearity", "tanh") == "tanh"
                else np.maximum(pre, 0.0)), None
    if kind == "gru":
        ir, iz, i_n = np.split(gi, 3, axis=-1)
        hr, hz, h_n = np.split(gh, 3, axis=-1)
        r, z = _sigmoid(ir + hr), _sigmoid(iz + hz)
        n = np.tanh(i_n + r * h_n)
        return (1.0 - z) * n + z * h, None
    i, f, g, o = np.split(gi + gh, 4, axis=-1)
    c = _sigmoid(f) * c + _sigmoid(i) * np.tanh(g)
    return _sigmoid(o) * np.tanh(c), c


def _direction(w, x, p, layer, reverse):
    """(B, L, H) outputs of one layer and direction, from a zero state."""
    key = (lambda name: f"l{layer}.{name}" + (".reverse" if reverse else ""))
    w_ih, w_hh = _f64(w[key("w_ih")]), _f64(w[key("w_hh")])
    b_ih, b_hh = _f64(w[key("b_ih")]), _f64(w[key("b_hh")])
    hidden = w_hh.shape[1]
    h = np.zeros((x.shape[0], hidden))
    c = np.zeros((x.shape[0], hidden))
    out = np.zeros((x.shape[0], x.shape[1], hidden))
    order = range(x.shape[1] - 1, -1, -1) if reverse else range(x.shape[1])
    for t in order:
        gates = (x[:, t] @ w_ih.T + b_ih, h @ w_hh.T + b_hh)
        h, c_new = _cell(p["kind"], p, gates, h, c)
        c = c if c_new is None else c_new
        out[:, t] = h
    return out


def _recurrent(w, x, p):
    x = _f64(x)
    dirs = 2 if p.get("bidirectional") else 1
    for layer in range(int(p.get("num_layers", 1))):
        outs = [_direction(w, x, p, layer, reverse=bool(d)) for d in range(dirs)]
        x = np.concatenate(outs, axis=-1)
    ret, hidden = p.get("return", "last"), int(p["hidden"])
    if ret == "all":
        return x
    if ret == "mean":
        return x.mean(axis=1)
    if ret == "max":
        return x.max(axis=1)
    if dirs == 1:
        return x[:, -1]
    return np.concatenate([x[:, -1, :hidden], x[:, 0, hidden:]], axis=-1)


def _mha(x, w, heads, prefix=""):
    """Self multi-head attention of (B, L, d): softmax(QKᵀ/√d_h)V per head, out-projected."""
    x = _f64(x)
    d = x.shape[-1]
    qkv = x @ _f64(w[prefix + "in_proj"]).T + _f64(w[prefix + "in_proj_bias"])
    q, k, v = np.split(qkv, 3, axis=-1)
    dh = d // heads
    split = (lambda t: t.reshape(t.shape[0], t.shape[1], heads, dh).transpose(0, 2, 1, 3))
    q, k, v = split(q), split(k), split(v)
    a = _softmax(q @ k.transpose(0, 1, 3, 2) / math.sqrt(dh), axis=-1)
    y = (a @ v).transpose(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], d)
    return y @ _f64(w[prefix + "out_proj"]).T + _f64(w[prefix + "out_proj_bias"])


def _attention(w, x, p):
    y = _mha(x, w, int(p["heads"]))
    return y.mean(axis=1) if p.get("pool") else y


def _attention_pool(w, x, p):
    x = _f64(x)
    s = np.tanh(x @ _f64(w["score"]).T + _f64(w["score_bias"])) @ _f64(w["query"]).T
    alpha = _softmax(s[..., 0], axis=-1)
    return (alpha[..., None] * x).sum(axis=1)


def attention_pool_weights(w, x):
    """α (B, L) of an attention_pool block."""
    x = _f64(x)
    s = np.tanh(x @ _f64(w["score"]).T + _f64(w["score_bias"])) @ _f64(w["query"]).T
    return _softmax(s[..., 0], axis=-1)


def _ln(x, gamma, beta, eps):
    mean, var = x.mean(axis=-1, keepdims=True), x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * _f64(gamma) + _f64(beta)


def encoder_layer(w, x, p, k):
    """One pre-norm encoder layer: x + MHA(LN₁x), then + W₂·act(W₁·LN₂x + b₁) + b₂."""
    pre, eps = f"layer{k}.", float(p.get("eps", 1e-5))
    x = _f64(x)
    x = x + _mha(_ln(x, w[pre + "norm1_gamma"], w[pre + "norm1_beta"], eps), w,
                 int(p["heads"]), prefix=pre)
    h = _ln(x, w[pre + "norm2_gamma"], w[pre + "norm2_beta"], eps)
    h = activation(p.get("activation", "relu"),
                   h @ _f64(w[pre + "linear1"]).T + _f64(w[pre + "linear1_bias"]))
    return x + h @ _f64(w[pre + "linear2"]).T + _f64(w[pre + "linear2_bias"])


def _transformer_encoder(w, x, p):
    for k in range(int(p["layers"])):
        x = encoder_layer(w, x, p, k)
    return _f64(x)


def _feature_tokenizer(w, x, p):
    x_num, x_cat = x if isinstance(x, tuple) else (x, None)
    cls = _f64(w["cls"]).reshape(1, 1, -1)
    rows = (x_num if x_num is not None else x_cat).shape[0]
    tokens = [np.repeat(cls, rows, axis=0)]
    if x_num is not None and _f64(w["weight"]).shape[0] > 0:
        tokens.append(_f64(x_num)[..., None] * _f64(w["weight"])[None] + _f64(w["bias"])[None])
    tables = sorted((k for k in w if k.startswith("table.")), key=lambda k: int(k.split(".")[1]))
    if tables:
        ids = np.asarray(x_cat)
        tokens.append(np.stack([_f64(w[t])[ids[:, j]] for j, t in enumerate(tables)], axis=1))
    return np.concatenate(tokens, axis=1)


def sinusoidal_table(length, d):
    """The sinusoidal position table (L, d): sin on even columns, cos on odd ones."""
    pos = np.arange(length, dtype=np.float64)[:, None]
    div = np.exp(np.arange(0, d, 2, dtype=np.float64) * (-math.log(10000.0) / d))
    pe = np.zeros((length, d))
    pe[:, 0::2] = np.sin(pos * div)
    pe[:, 1::2] = np.cos(pos * div[: pe[:, 1::2].shape[1]])
    return pe


def _positional(w, x, p):
    x = _f64(x)
    length, d = x.shape[1], x.shape[2]
    if p.get("kind", "sinusoidal") == "sinusoidal":
        return x + sinusoidal_table(length, d)[None]
    return x + _f64(w["pe"])[:length][None]


# ------------------------------------------------------------------ graph blocks

def _edges(edge_index, edge_weight):
    src, dst = np.asarray(edge_index[0]), np.asarray(edge_index[1])
    weight = np.ones(len(src)) if edge_weight is None else _f64(edge_weight)
    return src, dst, weight


def _loops(src, dst, weight, n):
    """The blocks' self-loop rule: a unit loop for every node without one, appended."""
    has = np.zeros(n, dtype=bool)
    has[src[src == dst]] = True
    missing = np.flatnonzero(~has)
    return (np.concatenate([src, missing]), np.concatenate([dst, missing]),
            np.concatenate([weight, np.ones(len(missing))]))


def adjacency(src, dst, weight, n):
    """Dense A (n, n) with A[i, j] = the summed weight of the edges j -> i."""
    a = np.zeros((n, n))
    np.add.at(a, (dst, src), weight)
    return a


def _gcn_matrix(src, dst, weight, n, p):
    if p.get("self_loops", True):
        src, dst, weight = _loops(src, dst, weight, n)
    a = adjacency(src, dst, weight, n)
    deg = a.sum(axis=1)
    safe = np.where(deg > 0, deg, 1.0)
    if p.get("normalize", "sym") == "sym":
        inv = np.where(deg > 0, safe ** -0.5, 0.0)
        return inv[:, None] * a * inv[None, :]
    return np.where(deg > 0, 1.0 / safe, 0.0)[:, None] * a


def _gcn(w, x, p):
    h, edge_index, edge_weight = x
    h = _f64(h)
    src, dst, weight = _edges(edge_index, edge_weight)
    out = _gcn_matrix(src, dst, weight, h.shape[0], p) @ (h @ _f64(w["weight"]).T)
    return out + _f64(w["bias"]) if w.get("bias") is not None else out


def _sage(w, x, p):
    h, edge_index, edge_weight = x
    h, n = _f64(h), _f64(h).shape[0]
    src, dst, weight = _edges(edge_index, edge_weight)
    aggregator = p.get("aggregator", "mean")
    if aggregator == "max":
        agg = np.zeros_like(h)
        for i in range(n):
            into = dst == i
            if into.any():
                agg[i] = (h[src[into]] * weight[into, None]).max(axis=0)
    else:
        a = adjacency(src, dst, weight, n)
        if aggregator == "mean":
            count = np.bincount(dst, minlength=n).astype(np.float64)
            a = a / np.maximum(count, 1.0)[:, None]
        agg = a @ h
    out = agg @ _f64(w["neigh"]).T
    if w.get("neigh_bias") is not None:
        out = out + _f64(w["neigh_bias"])
    if w.get("root") is not None:
        out = out + h @ _f64(w["root"]).T
    if p.get("normalize"):
        out = out / np.maximum(np.linalg.norm(out, axis=-1, keepdims=True), 1e-12)
    return out


def gat_alpha(w, h, edge_index, edge_weight, p):
    """(src, dst, α·w (E, heads)) of a GAT block, softmax per destination over its edges."""
    h = _f64(h)
    n, heads, out = h.shape[0], int(p["heads"]), int(p["out"])
    src, dst, weight = _edges(edge_index, edge_weight)
    if p.get("self_loops", True):
        src, dst, weight = _loops(src, dst, weight, n)
    hx = (h @ _f64(w["weight"]).T).reshape(n, heads, out)
    a_src = (hx * _f64(w["att_src"])).sum(axis=-1)
    a_dst = (hx * _f64(w["att_dst"])).sum(axis=-1)
    e = a_src[src] + a_dst[dst]
    e = np.where(e >= 0, e, float(p.get("negative_slope", 0.2)) * e)
    alpha = np.zeros_like(e)
    for i in range(n):
        into = dst == i
        if into.any():
            alpha[into] = _softmax(e[into], axis=0)
    return src, dst, alpha * weight[:, None]


def _gat(w, x, p):
    h, edge_index, edge_weight = x
    h = _f64(h)
    n, heads, out = h.shape[0], int(p["heads"]), int(p["out"])
    src, dst, coef = gat_alpha(w, h, edge_index, edge_weight, p)
    hx = (h @ _f64(w["weight"]).T).reshape(n, heads, out)
    y = np.zeros((n, heads, out))
    np.add.at(y, dst, coef[..., None] * hx[src])
    y = y.reshape(n, heads * out) if p.get("concat", True) else y.mean(axis=1)
    return y + _f64(w["bias"]) if w.get("bias") is not None else y


def _gin(w, x, p):
    h, edge_index, edge_weight = x
    h = _f64(h)
    src, dst, weight = _edges(edge_index, edge_weight)
    z = (1.0 + float(_f64(w["eps"]).reshape(-1)[0])) * h + adjacency(src, dst, weight, h.shape[0]) @ h
    layers = sorted({int(k.split(".")[1]) for k in w if k.startswith("mlp.")})
    for i, k in enumerate(layers):
        z = z @ _f64(w[f"mlp.{k}.weight"]).T + _f64(w[f"mlp.{k}.bias"])
        if i < len(layers) - 1:
            z = np.maximum(z, 0.0)
    return z


def _graph_pool(w, x, p):
    h, graph_index, num_graphs = x
    h, graph_index = _f64(h), np.asarray(graph_index)
    out = np.zeros((int(num_graphs), h.shape[1]))
    for g in range(int(num_graphs)):
        rows = h[graph_index == g]
        if len(rows):
            out[g] = {"sum": rows.sum, "mean": rows.mean, "max": rows.max}[
                p.get("readout", "mean")](axis=0)
    return out


def _edge_dropout(w, x, p):
    edge_index, edge_weight = x
    return np.asarray(edge_index), (None if edge_weight is None else _f64(edge_weight))


def coefficients(block_type, weights, x, params):
    """(src, dst, coef) as the block's `coefficients` must return them, so that the conv
    output at node i is recoverable from the edges into i (sorted by (dst, src) for
    comparison: the order of a block's list is its own)."""
    h, edge_index, edge_weight = x
    n = _f64(h).shape[0]
    src, dst, weight = _edges(edge_index, edge_weight)
    if block_type == "gat_conv":
        return gat_alpha(weights, h, edge_index, edge_weight, params)
    if block_type == "gcn_conv":
        if params.get("self_loops", True):
            src, dst, weight = _loops(src, dst, weight, n)
        deg = adjacency(src, dst, weight, n).sum(axis=1)
        inv = np.where(deg > 0, 1.0 / np.where(deg > 0, deg, 1.0), 0.0)
        if params.get("normalize", "sym") == "sym":
            return src, dst, np.sqrt(inv[src]) * weight * np.sqrt(inv[dst])
        return src, dst, weight * inv[dst]
    if block_type == "sage_conv":
        if params.get("aggregator", "mean") == "mean":
            count = np.bincount(dst, minlength=n).astype(np.float64)
            return src, dst, weight / np.maximum(count, 1.0)[dst]
        return src, dst, weight
    if block_type == "gin_conv":
        nodes = np.arange(n)
        eps = float(_f64(weights["eps"]).reshape(-1)[0])
        return (np.concatenate([src, nodes]), np.concatenate([dst, nodes]),
                np.concatenate([weight, np.full(n, 1.0 + eps)]))
    raise KeyError(f"{block_type!r} is not a graph convolution")


# ------------------------------------------------------------------ the tables

ORACLES: Mapping[str, Callable[[Mapping, Any, Mapping], Any]] = MappingProxyType({
    "dense": _dense, "head": _head, "embedding": _embedding, "concat": _concat,
    "batchnorm": _batchnorm, "layernorm": _layernorm, "dropout": _identity,
    "activation": _activation, "flatten": _flatten, "reshape": _reshape, "pool": _pool,
    "sequential": _sequential, "residual": _residual, "parallel": _parallel,
    "conv1d": _conv1d, "pool1d": _pool1d,
    "rnn": _recurrent, "gru": _recurrent, "lstm": _recurrent,
    "attention": _attention, "attention_pool": _attention_pool,
    "transformer_encoder": _transformer_encoder, "feature_tokenizer": _feature_tokenizer,
    "positional": _positional, "decoder": _decoder, "reparameterize": _reparameterize,
    "gcn_conv": _gcn, "sage_conv": _sage, "gat_conv": _gat, "gin_conv": _gin,
    "graph_pool": _graph_pool, "edge_dropout": _edge_dropout,
})

# a `custom` block is user code: only its declared shape and finiteness can be checked
SHAPE_ONLY = frozenset({"custom"})

BLOCK_TYPES = tuple(sorted(set(ORACLES) | SHAPE_ONLY))


def oracle(block_type):
    """The value oracle of a block type; a shape-only type has none, and says so."""
    if block_type in SHAPE_ONLY:
        raise KeyError(f"{block_type!r} has no value oracle: check shape_rule and finiteness")
    return ORACLES[block_type]


def check_custom(out, params):
    """A custom block's output: its declared `out_shape` per sample, every value finite."""
    out = np.asarray(out)
    return tuple(out.shape[1:]) == tuple(int(s) for s in params["out_shape"]) \
        and bool(np.isfinite(out).all())


# ------------------------------------------------------------------ shapes

def _width_rule(block_type, params, in_shape, context):
    rules = {
        "dense": lambda: in_shape[:-1] + (int(params["units"]),),
        "head": lambda: in_shape[:-1] + (int(params["out"]),),
        "decoder": lambda: in_shape[:-1] + (int(params["out"]),),
        "reparameterize": lambda: in_shape[:-1] + (int(params["z"]),),
        "embedding": lambda: in_shape[:-1] + (sum(_embedding_dims(params)),),
        "concat": lambda: in_shape[:-1] + (in_shape[-1] + int(context.get("emb_width", 0)),),
        "gcn_conv": lambda: (in_shape[0], int(params["out"])),
        "sage_conv": lambda: (in_shape[0], int(params["out"])),
        "gin_conv": lambda: (in_shape[0], int(params["out"])),
        "gat_conv": lambda: (in_shape[0], int(params["out"]) * (
            int(params.get("heads", 1)) if params.get("concat", True) else 1)),
        "graph_pool": lambda: (int(context["num_graphs"]), in_shape[-1]),
        "custom": lambda: (in_shape[0],) + tuple(int(s) for s in params["out_shape"]),
    }
    return rules[block_type]() if block_type in rules else None


def _embedding_dims(params):
    dims = params.get("dims") or {c: int(params.get("dim", 8)) for c in params["columns"]}
    return [int(dims[c]) for c in params["columns"]]


def _sequence_rule(block_type, params, in_shape, context):
    b, rest = in_shape[0], in_shape[1:]
    if block_type in ("rnn", "gru", "lstm"):
        width = int(params["hidden"]) * (2 if params.get("bidirectional") else 1)
        return (b, rest[0], width) if params.get("return", "last") == "all" else (b, width)
    if block_type == "conv1d":
        k, s, pad = int(params["kernel"]), int(params.get("stride", 1)), params.get("padding", "same")
        pad = (k // 2 if s != 1 else None) if pad == "same" else int(pad)
        steps = rest[0] if pad is None else (rest[0] + 2 * pad - k) // s + 1
        return (b, steps, int(params["channels"]))
    rules = {
        "pool1d": lambda: (b, rest[0] // int(params.get("size", 2)), rest[1]),
        "attention": lambda: (b, rest[-1]) if params.get("pool") else in_shape,
        "attention_pool": lambda: (b, rest[-1]),
        "pool": lambda: in_shape if len(in_shape) == 2 else (b, rest[-1]),
        "flatten": lambda: (b, int(np.prod(rest))),
        "reshape": lambda: (b,) + tuple(int(s) for s in params["shape"]),
        "feature_tokenizer": lambda: (b, 1 + rest[0] + int(context.get("n_cat", 0)),
                                      int(params["d"])),
    }
    return rules[block_type]() if block_type in rules else None


def shape_rule(block_type, params, in_shape, context=None):
    """The output shape of a block, leading axis included, from its params and the shape of
    its primary input (x_num for concat and feature_tokenizer, x_cat for embedding, node
    rows for the graph blocks, the edge index (2, E) for edge_dropout). `context` supplies
    what the primary input does not carry: `emb_width` (concat), `n_cat` (feature_tokenizer),
    `num_graphs` (graph_pool). Composite blocks fold their `children` (type, params) pairs."""
    in_shape, context = tuple(int(s) for s in in_shape), dict(context or {})
    if block_type in ("sequential", "parallel", "residual"):
        return _composite_rule(block_type, params, in_shape, context)
    for rule in (_width_rule, _sequence_rule):
        out = rule(block_type, params, in_shape, context)
        if out is not None:
            return tuple(out)
    return in_shape            # every other block preserves its input's shape


def _kids(params):
    """(type, params) of each child; `children` may hold (type, params) pairs or the
    oracle's (type, weights, params) triples."""
    return [(child[0], child[-1]) for child in params["children"]]


def _composite_rule(block_type, params, in_shape, context):
    """sequential and residual fold their children (residual `repeat` times: a projection
    lets the width change); parallel concatenates or sums its branches."""
    if block_type == "parallel":
        outs = [shape_rule(kind, child, in_shape, context) for kind, child in _kids(params)]
        if params.get("merge", "concat") == "sum":
            return outs[0]
        return outs[0][:-1] + (sum(o[-1] for o in outs),)
    if "repeats" in params:            # the residual as weights_of reads it
        chains = [[(c[0], c[-1]) for c in chain] for chain in params["repeats"]]
    else:
        repeats = int(params.get("repeat", 1)) if block_type == "residual" else 1
        chains = [_kids(params)] * repeats
    shape = in_shape
    for chain in chains:
        for kind, child in chain:
            shape = shape_rule(kind, child, shape, context)
    return shape


# ------------------------------------------------------------------ reading a torch block

def _np(t):
    return t.detach().cpu().double().numpy()


_TYPE_OF_CLASS = {
    "Dense": "dense", "Embedding": "embedding", "Concat": "concat", "BatchNorm": "batchnorm",
    "LayerNorm": "layernorm", "Dropout": "dropout", "Activation": "activation",
    "Flatten": "flatten", "Reshape": "reshape", "Pool": "pool", "Sequential": "sequential",
    "Residual": "residual", "Parallel": "parallel", "Conv1d": "conv1d", "Pool1d": "pool1d",
    "Attention": "attention", "AttentionPool": "attention_pool",
    "TransformerEncoder": "transformer_encoder", "FeatureTokenizer": "feature_tokenizer",
    "Positional": "positional", "Decoder": "decoder", "Reparameterize": "reparameterize",
    "GCNConv": "gcn_conv", "SAGEConv": "sage_conv", "GATConv": "gat_conv",
    "GINConv": "gin_conv", "GraphPool": "graph_pool", "EdgeDropout": "edge_dropout",
    "Head": "head", "CumulativeHead": "head",
}


def type_of(module):
    """The catalog type of a block instance (a recurrent block says which of the three)."""
    cfg = getattr(module, "cfg", None)
    if cfg is not None and getattr(cfg, "kind", None) in ("rnn", "gru", "lstm"):
        return cfg.kind
    name = type(module).__name__
    if name not in _TYPE_OF_CLASS:
        raise KeyError(f"no catalog type for block class {name}")
    return _TYPE_OF_CLASS[name]


def _act_name(act):
    names = {"Identity": "none", "ReLU": "relu", "GELU": "gelu", "SiLU": "silu",
             "Tanh": "tanh", "Sigmoid": "sigmoid", "LeakyReLU": "leaky_relu"}
    return names[type(act).__name__]


def _linear(lin):
    return {"weight": _np(lin.weight), "bias": None if lin.bias is None else _np(lin.bias)}


def _read_units(m):
    if hasattr(m, "gaps"):
        return ({"weight": _np(m.score.weight), "first": _np(m.first), "gaps": _np(m.gaps)},
                {"cumulative": True})
    return _linear(m.linear), {"activation": _act_name(m.act)}


def _read_embedding(m):
    return ({f"table.{c}": _np(e.weight) for c, e in zip(m.columns, m.embeddings)},
            {"columns": list(m.columns)})


def _read_norm(m):
    norm = m.bn if hasattr(m, "bn") else m.ln
    w = {"gamma": _np(norm.weight), "beta": _np(norm.bias)}
    if hasattr(m, "bn"):
        w.update(running_mean=_np(norm.running_mean), running_var=_np(norm.running_var))
    return w, {"eps": float(norm.eps)}


def _read_recurrent(m):
    cfg, w = m.cfg, {}
    for layer in range(cfg.num_layers):
        for reverse in range(cfg.directions):
            suffix = f"_l{layer}" + ("_reverse" if reverse else "")
            tail = ".reverse" if reverse else ""
            for role in ("ih", "hh"):
                w[f"l{layer}.w_{role}{tail}"] = _np(getattr(m.rnn, f"weight_{role}{suffix}"))
                w[f"l{layer}.b_{role}{tail}"] = _np(getattr(m.rnn, f"bias_{role}{suffix}"))
    return w, {"kind": cfg.kind, "hidden": cfg.hidden, "num_layers": cfg.num_layers,
               "bidirectional": cfg.bidirectional, "return": cfg.ret,
               "nonlinearity": cfg.nonlinearity}


def _mha_weights(attn, prefix=""):
    return {prefix + "in_proj": _np(attn.in_proj_weight),
            prefix + "in_proj_bias": _np(attn.in_proj_bias),
            prefix + "out_proj": _np(attn.out_proj.weight),
            prefix + "out_proj_bias": _np(attn.out_proj.bias)}


def _read_encoder(m):
    w, layers = {}, list(m.enc.layers)
    for k, layer in enumerate(layers):
        pre = f"layer{k}."
        w.update(_mha_weights(layer.self_attn, pre))
        for name in ("linear1", "linear2"):
            lin = getattr(layer, name)
            w[pre + name], w[pre + name + "_bias"] = _np(lin.weight), _np(lin.bias)
        for name in ("norm1", "norm2"):
            norm = getattr(layer, name)
            w[pre + name + "_gamma"], w[pre + name + "_beta"] = _np(norm.weight), _np(norm.bias)
    first = layers[0]
    act = first.activation
    act_name = act if isinstance(act, str) else getattr(act, "__name__", "relu")
    return w, {"heads": first.self_attn.num_heads, "layers": len(layers),
               "eps": float(first.norm1.eps), "activation": act_name}


def _read_tokenizer(m):
    w = {"weight": _np(m.weight), "bias": _np(m.bias), "cls": _np(m.cls).reshape(-1)}
    for j, emb in enumerate(m.embeddings):
        w[f"table.{j}"] = _np(emb.weight)
    return w, {"d": int(m.cls.shape[-1])}


def _read_gat(m):
    w = {"weight": _np(m.lin.weight), "att_src": _np(m.att_src), "att_dst": _np(m.att_dst),
         "bias": None if m.bias is None else _np(m.bias)}
    return w, {"heads": m.cfg.heads, "out": m.cfg.out, "concat": m.cfg.concat,
               "negative_slope": m.cfg.negative_slope, "self_loops": m.cfg.self_loops}


def _read_sage(m):
    w = {"neigh": _np(m.lin_neigh.weight),
         "neigh_bias": None if m.lin_neigh.bias is None else _np(m.lin_neigh.bias),
         "root": None if m.lin_root is None else _np(m.lin_root.weight)}
    return w, {"aggregator": m.aggregator, "normalize": m.normalize}


def _read_gin(m):
    w = {"eps": _np(m.eps)}
    for k, layer in enumerate(m.mlp):
        if hasattr(layer, "weight"):
            w[f"mlp.{k}.weight"], w[f"mlp.{k}.bias"] = _np(layer.weight), _np(layer.bias)
    return w, {}


def _read_decoder(m):
    w, k = {}, 0
    for layer in m.net:
        if hasattr(layer, "weight"):
            w[f"layer{k}.weight"], w[f"layer{k}.bias"] = _np(layer.weight), _np(layer.bias)
            k += 1
    acts = [layer for layer in m.net if not hasattr(layer, "weight")]
    return w, {"activation": _act_name(acts[0]) if acts else "none"}


def _read_children(modules):
    return [(type_of(c),) + weights_of(c) for c in modules]


def _read_residual(m):
    projections = [None if type(p).__name__ == "Identity" else _linear(p) for p in m.projections]
    return {}, {"repeats": [_read_children(chain) for chain in m.repeats],
                "projections": projections}


_READERS = {
    "dense": _read_units, "head": _read_units, "embedding": _read_embedding,
    "batchnorm": _read_norm, "layernorm": _read_norm,
    "rnn": _read_recurrent, "gru": _read_recurrent, "lstm": _read_recurrent,
    "transformer_encoder": _read_encoder, "feature_tokenizer": _read_tokenizer,
    "gat_conv": _read_gat, "sage_conv": _read_sage, "gin_conv": _read_gin,
    "decoder": _read_decoder, "residual": _read_residual,
    "concat": lambda m: ({}, {}), "dropout": lambda m: ({}, {}),
    "flatten": lambda m: ({}, {}), "edge_dropout": lambda m: ({}, {}),
    "activation": lambda m: ({}, {"name": _act_name(m.act)}),
    "reshape": lambda m: ({}, {"shape": list(m.shape)}),
    "pool": lambda m: ({}, {"kind": m.kind}),
    "pool1d": lambda m: ({}, {"kind": "max" if "Max" in type(m.pool).__name__ else "avg",
                              "size": int(np.ravel(m.pool.kernel_size)[0])}),
    "conv1d": lambda m: ({"kernel": _np(m.conv.weight),
                          "bias": None if m.conv.bias is None else _np(m.conv.bias)},
                         {"stride": int(m.conv.stride[0]),
                          "padding": m.conv.padding if isinstance(m.conv.padding, str)
                          else int(m.conv.padding[0])}),
    "attention": lambda m: (_mha_weights(m.attn), {"heads": m.attn.num_heads, "pool": m.pool}),
    "attention_pool": lambda m: ({"score": _np(m.score.weight), "score_bias": _np(m.score.bias),
                                  "query": _np(m.query.weight)}, {}),
    "positional": lambda m: ({"pe": _np(m.pe)[0]}, {"kind": m.kind}),
    "reparameterize": lambda m: ({}, {"z": m.z}),
    "gcn_conv": lambda m: ({"weight": _np(m.lin.weight),
                            "bias": None if m.bias is None else _np(m.bias)},
                           {"normalize": m.normalize, "self_loops": m.self_loops}),
    "graph_pool": lambda m: ({}, {"readout": m.readout}),
    "sequential": lambda m: ({}, {"children": _read_children(m.inner)}),
    "parallel": lambda m: ({}, {"children": _read_children(m.branches), "merge": m.merge}),
}


def weights_of(module, block_type=None):
    """(weights, params) of a torch block, in the oracle's role names, as float64 arrays."""
    block_type = block_type or type_of(module)
    if block_type not in _READERS:
        raise KeyError(f"no reader for block type {block_type!r}")
    weights, params = _READERS[block_type](module)
    return weights, params


__all__ = ["ORACLES", "SHAPE_ONLY", "BLOCK_TYPES", "oracle", "shape_rule", "coefficients",
           "weights_of", "type_of", "activation", "affine", "adjacency", "gat_alpha",
           "attention_pool_weights", "encoder_layer", "sinusoidal_table", "check_custom"]
