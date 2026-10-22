"""
inspection.probes_seq — the probes of the sequence blocks (inspect-contract §3.1; build plan
R-B3, R-B8, R-F5): `conv1d`, `pool1d`, `rnn`, `gru`, `lstm`, `attention`, `attention_pool`,
the synthetic `encoder_layer`, `feature_tokenizer` and `positional`.

- **conv1d**: `maps` — the output's heat map (steps × channels) and each channel's mean; the
  check is the convolution of the input by the published kernel.
- **pool1d**: `winners` — per input step, how many (output step, channel) cells it won (max);
  the check is the pooling formula.
- **rnn, gru, lstm**: `unroll` — the cell equations of the view table re-run step by step from
  the published weights, for every layer and direction, from a zero state (layer k > 0 reads
  both directions of layer k − 1): per layer and direction the heat map of the gates' means
  per step (torch's gate order), the hidden state's norm per step and the cell state's (lstm),
  and `max_abs_diff` against the module's own sequence output (the `:seq` call). The check is
  that difference, and the block's readout (`return`) of the unrolled sequence against the
  node's output. `exact` carries the hidden states, (layers·directions) × steps × units.
- **attention**: `weights` — heads × steps × steps, the module's own attention map
  (`attn(x, x, x, need_weights=True)`, a re-call under `cap.suspended()`); the check
  recomputes softmax(QKᵀ/√d)·V and the output projection from the published `in_proj` and
  `out_proj` (the mean over steps when the block pools) and compares both the map and the
  output. `exact` carries `alpha`.
- **attention_pool**: `weights` — α over the steps, the block's own `weights(x)` (R-B8); the
  check recomputes α = softmax_t(q·tanh(W h_t + b)) from the `score` and `query` tensors and
  y = Σ_t α_t h_t. `exact` carries `alpha`.
- **encoder_layer**: `weights` — the layer's self-attention map over its normalised input
  (`norm_first`), `cls_row` — the [cls] query's row per head when the tokens come from a
  feature tokenizer — and `stream_norms`, each token's norm in and out; the check is
  x + MHA(LN₁ x), then + FF(LN₂ ·) (post-norm when the layer is not `norm_first`).
- **feature_tokenizer**: `tokens` — each token's norm; the check is [cls ; x_j·W_j + b_j ;
  E_c[id_c]].
- **positional**: the check is x + P[:L].
"""
from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from .codec import num, nums
from .probes import Probe, f64, gap, norm_last, sample_io

__all__ = ["ATTENTION", "ATTENTION_POOL", "CONV1D", "ENCODER_LAYER", "FEATURE_TOKENIZER",
           "POOL1D", "POSITIONAL", "RECURRENT", "attend", "unroll"]

CELLS = ((nn.LSTM, "lstm"), (nn.GRU, "gru"), (nn.RNN, "rnn"))


class _Conv1d(Probe):
    KEYS = ("maps",)
    CHECKS = True

    def probe(self, cap, node):
        _, y = sample_io(cap, node)
        return {"maps": {"heat": cap.reducer.heat(y), "channel_mean": nums(y.mean(dim=0))}}

    def recompute(self, cap, node):
        x, y = sample_io(cap, node)
        conv = cap.module(node.id).conv
        z = F.conv1d(x.T.unsqueeze(0), f64(conv.weight), f64(conv.bias), conv.stride,
                     conv.padding, conv.dilation, conv.groups)
        return gap(z[0].T, y)


class _Pool1d(Probe):
    KEYS = ("winners",)
    CHECKS = True

    def _read(self, cap, node, indices: bool = False):
        x, y = sample_io(cap, node)
        pool = cap.module(node.id).pool
        seq = x.T.unsqueeze(0)
        if isinstance(pool, nn.MaxPool1d):
            read = F.max_pool1d(seq, pool.kernel_size, pool.stride, pool.padding, pool.dilation,
                                pool.ceil_mode, return_indices=indices)
        else:
            read = F.avg_pool1d(seq, pool.kernel_size, pool.stride, pool.padding,
                                pool.ceil_mode, pool.count_include_pad)
        return x, y, read

    def probe(self, cap, node):
        if not isinstance(cap.module(node.id).pool, nn.MaxPool1d):
            return {"winners": None}
        x, _, (_, where) = self._read(cap, node, indices=True)
        return {"winners": torch.bincount(where.flatten(), minlength=x.shape[0]).tolist()}

    def recompute(self, cap, node):
        _, y, read = self._read(cap, node)
        return gap(read[0].T, y)


class Run(NamedTuple):
    """One layer and direction of an unrolled recurrent block (steps in time order)."""

    layer: int
    direction: str
    hidden: object          # (L, H)
    cell: object            # (L, H) for an lstm, else None
    gates: object           # (L, G, H), the gate activations


def _kind(rnn) -> str:
    return next(kind for cls, kind in CELLS if isinstance(rnn, cls))


def _cell(kind: str, pre: tuple, state: tuple, relu: bool) -> tuple:
    """One step: (h, c, gates) from gi = W_ih x + b_ih, gh = W_hh h + b_hh."""
    gi, gh = pre
    h, c = state
    width = h.shape[0]
    if kind == "rnn":
        a = torch.relu(gi + gh) if relu else torch.tanh(gi + gh)
        return a, c, a.unsqueeze(0)
    if kind == "gru":
        r = torch.sigmoid(gi[:width] + gh[:width])
        z = torch.sigmoid(gi[width:2 * width] + gh[width:2 * width])
        n = torch.tanh(gi[2 * width:] + r * gh[2 * width:])
        return (1 - z) * n + z * h, c, torch.stack([r, z, n])
    i, f, g, o = (gi + gh).split(width)
    i, f, g, o = torch.sigmoid(i), torch.sigmoid(f), torch.tanh(g), torch.sigmoid(o)
    c = f * c + i * g
    return o * torch.tanh(c), c, torch.stack([i, f, g, o])


def _direction(rnn, x, where: tuple) -> Run:
    layer, reverse = where
    sfx = f"_l{layer}" + ("_reverse" if reverse else "")
    w_ih, w_hh, b_ih, b_hh = (f64(getattr(rnn, f"{n}{sfx}", None))
                              for n in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"))
    kind, steps = _kind(rnn), x.shape[0]
    gi = F.linear(x, w_ih, b_ih)
    h = c = x.new_zeros(w_hh.shape[1])
    out = [None] * steps
    for t in (range(steps - 1, -1, -1) if reverse else range(steps)):
        h, c, g = _cell(kind, (gi[t], F.linear(h, w_hh, b_hh)), (h, c),
                        getattr(rnn, "nonlinearity", "tanh") == "relu")
        out[t] = (h, c, g)
    hidden, cell, gates = (torch.stack([s[i] for s in out]) if steps else x.new_zeros(0)
                           for i in range(3))
    return Run(layer, "reverse" if reverse else "forward", hidden,
               cell if kind == "lstm" else None, gates)


def unroll(rnn, x) -> tuple:
    """(runs, top): every layer and direction of `rnn` over one sample `x` (L, F), float64,
    and the top layer's sequence output (L, H·directions)."""
    runs, inp = [], f64(x)
    for layer in range(int(rnn.num_layers)):
        outs = []
        for d in range(2 if rnn.bidirectional else 1):
            run = _direction(rnn, inp, (layer, bool(d)))
            runs.append(run)
            outs.append(run.hidden)
        inp = torch.cat(outs, dim=-1)
    return tuple(runs), inp


def _readout(module, top):
    """The block's `return` of a sequence output (a bidirectional `last` reads each direction's
    final state)."""
    ret, width = str(module.ret), int(module.rnn.hidden_size)
    if ret == "all":
        return top
    if ret == "mean":
        return top.mean(dim=0)
    if ret == "max":
        return top.max(dim=0).values
    if not module.rnn.bidirectional:
        return top[-1]
    return torch.cat([top[-1, :width], top[0, width:]])


class _Recurrent(Probe):
    KEYS = ("unroll",)
    EXACT = ("hidden",)
    CHECKS = True

    def _run(self, cap, node) -> tuple:
        def compute():
            x = cap.take(cap.value_in(node.id), node, "in")
            runs, top = unroll(cap.module(node.id).rnn, x)
            return runs, top, f64(cap.take(cap.seq(node.id), node))
        return cap.memo((node.id, "unroll"), compute)

    def probe(self, cap, node):
        runs, top, seq = self._run(cap, node)
        per_layer = [{"layer": r.layer, "direction": r.direction,
                      "gates": cap.reducer.heat(r.gates.mean(dim=-1)),
                      "hidden_norm": nums(norm_last(r.hidden)),
                      "cell_norm": None if r.cell is None else nums(norm_last(r.cell))}
                     for r in runs]
        return {"unroll": {"per_layer": per_layer, "max_abs_diff": num(gap(top, seq))}}

    def recompute(self, cap, node):
        _, top, seq = self._run(cap, node)
        out = f64(cap.take(cap.value(node.id), node))
        return max(gap(top, seq), gap(_readout(cap.module(node.id), top), out))

    def exact(self, cap, node):
        runs, _, _ = self._run(cap, node)
        return {"hidden": (torch.stack([r.hidden for r in runs]),
                           ("layer_direction", "step", "unit"))}


def attend(attn, x) -> tuple:
    """(alpha (heads, L, L), y (L, d)): a packed multi-head self-attention over one sample
    `x` (L, d), recomputed in float64 from its published `in_proj` and `out_proj`."""
    dim, heads = int(attn.embed_dim), int(attn.num_heads)
    width, steps = dim // heads, x.shape[0]
    qkv = F.linear(x, f64(attn.in_proj_weight), f64(attn.in_proj_bias))
    q, k, v = (qkv[:, i * dim:(i + 1) * dim].reshape(steps, heads, width).transpose(0, 1)
               for i in range(3))
    alpha = torch.softmax(q @ k.transpose(1, 2) / math.sqrt(width), dim=-1)
    y = (alpha @ v).transpose(0, 1).reshape(steps, dim)
    return alpha, F.linear(y, f64(attn.out_proj.weight), f64(attn.out_proj.bias))


def _module_map(cap, attn, h) -> torch.Tensor:
    """The attention module's own map over `h` (1, L, d): a re-call, so under suspended()."""
    with cap.suspended(), torch.no_grad():
        _, w = attn(h, h, h, need_weights=True, average_attn_weights=False)
    return f64(w[0])


class _Attention(Probe):
    KEYS = ("weights",)
    EXACT = ("alpha",)
    CHECKS = True

    def _map(self, cap, node):
        x = cap.take(cap.value_in(node.id), node, "in").detach()
        return cap.memo((node.id, "alpha"), lambda: _module_map(
            cap, cap.module(node.id).attn, x.unsqueeze(0)))

    def probe(self, cap, node):
        return {"weights": cap.reducer.attention(self._map(cap, node))}

    def recompute(self, cap, node):
        x, out = sample_io(cap, node)
        module = cap.module(node.id)
        alpha, y = attend(module.attn, x)
        return max(gap(alpha, self._map(cap, node)),
                   gap(y.mean(dim=0) if module.pool else y, out))

    def exact(self, cap, node):
        return {"alpha": (self._map(cap, node), ("head", "query", "key"))}


class _AttentionPool(Probe):
    KEYS = ("weights",)
    EXACT = ("alpha",)
    CHECKS = True

    def _alpha(self, cap, node):
        def compute():
            x = cap.take(cap.value_in(node.id), node, "in").detach()
            with cap.suspended(), torch.no_grad():
                return f64(cap.module(node.id).weights(x.unsqueeze(0))[0])
        return cap.memo((node.id, "alpha"), compute)

    def probe(self, cap, node):
        return {"weights": cap.reducer.attention(self._alpha(cap, node))}

    def recompute(self, cap, node):
        x, out = sample_io(cap, node)
        module, alpha = cap.module(node.id), self._alpha(cap, node)
        score = F.linear(torch.tanh(F.linear(x, f64(module.score.weight),
                                             f64(module.score.bias))),
                         f64(module.query.weight), f64(module.query.bias))
        formula = torch.softmax(score.squeeze(-1), dim=-1)
        return max(gap(formula, alpha), gap((alpha.unsqueeze(-1) * x).sum(dim=0), out))

    def exact(self, cap, node):
        return {"alpha": (self._alpha(cap, node), ("step",))}


def _norm(norm, t):
    return F.layer_norm(t, norm.normalized_shape, f64(norm.weight), f64(norm.bias),
                        float(norm.eps))


def _encoder(layer, x):
    """One transformer encoder layer over a sample (L, d), float64 (eval: no dropout)."""
    def feed(t):
        return F.linear(layer.activation(F.linear(t, f64(layer.linear1.weight),
                                                  f64(layer.linear1.bias))),
                        f64(layer.linear2.weight), f64(layer.linear2.bias))
    if layer.norm_first:
        x = x + attend(layer.self_attn, _norm(layer.norm1, x))[1]
        return x + feed(_norm(layer.norm2, x))
    x = _norm(layer.norm1, x + attend(layer.self_attn, x)[1])
    return _norm(layer.norm2, x + feed(x))


class _EncoderLayer(Probe):
    KEYS = ("weights", "cls_row", "stream_norms")
    EXACT = ("alpha",)
    CHECKS = True

    def _map(self, cap, node):
        def compute():
            layer = cap.module(node.id)
            x = cap.take(cap.value_in(node.id), node, "in").detach().unsqueeze(0)
            with cap.suspended(), torch.no_grad():
                h = layer.norm1(x) if layer.norm_first else x
            return _module_map(cap, layer.self_attn, h)
        return cap.memo((node.id, "alpha"), compute)

    def probe(self, cap, node):
        x, y = sample_io(cap, node)
        alpha = self._map(cap, node)
        tokens = any(n.type == "feature_tokenizer" for n in cap.graph.top_level())
        return {"weights": cap.reducer.attention(alpha),
                "cls_row": nums(alpha[:, 0, :]) if tokens else None,
                "stream_norms": {"in": nums(norm_last(x)), "out": nums(norm_last(y))}}

    def recompute(self, cap, node):
        x, y = sample_io(cap, node)
        with torch.no_grad():
            return gap(_encoder(cap.module(node.id), x), y)

    def exact(self, cap, node):
        return {"alpha": (self._map(cap, node), ("head", "query", "key"))}


class _FeatureTokenizer(Probe):
    KEYS = ("tokens",)
    CHECKS = True

    def probe(self, cap, node):
        _, y = sample_io(cap, node)
        return {"tokens": nums(norm_last(y))}

    def recompute(self, cap, node):
        x, y = sample_io(cap, node)
        module = cap.module(node.id)
        tokens = [f64(module.cls).reshape(1, -1)]
        if x is not None and module.weight.shape[0] > 0:
            tokens.append(x.unsqueeze(-1) * f64(module.weight) + f64(module.bias))
        if len(module.embeddings):
            ids = cap.take(cap.extras(node.id)[0], node, "in")
            tokens.append(torch.stack([f64(e.weight)[int(ids[j])]
                                       for j, e in enumerate(module.embeddings)]))
        return gap(torch.cat(tokens, dim=0), y)


class _Positional(Probe):
    CHECKS = True

    def recompute(self, cap, node):
        x, y = sample_io(cap, node)
        return gap(x + f64(cap.module(node.id).pe[0, :x.shape[0]]), y)


CONV1D = _Conv1d()
POOL1D = _Pool1d()
RECURRENT = _Recurrent()
ATTENTION = _Attention()
ATTENTION_POOL = _AttentionPool()
ENCODER_LAYER = _EncoderLayer()
FEATURE_TOKENIZER = _FeatureTokenizer()
POSITIONAL = _Positional()
