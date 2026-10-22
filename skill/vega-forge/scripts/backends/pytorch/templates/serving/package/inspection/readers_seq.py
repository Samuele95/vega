"""
inspection.readers_seq — the readers of the sequence blocks: `conv1d`, `pool1d`, `rnn`, `gru`,
`lstm`, `attention`, `attention_pool`, `transformer_encoder` (with its synthetic
`encoder_layer` children), `feature_tokenizer` and `positional` (inspect-contract §3.1, §5.3;
build plan R-B3, R-B8).

- **Recurrent** blocks are read through `.rnn` (the torch `RNN` / `GRU` / `LSTM`) and `.ret`:
  per layer and direction the tensors `l<k>.w_ih`, `l<k>.w_hh`, `l<k>.b_ih`, `l<k>.b_hh` (a
  `.reverse` suffix for the backward direction), each gate's norms, the column norms of its
  input block (labelled by feature name on layer 0 when the block reads the inputs), and the
  spectral radius of its recurrent block (`eigvals` up to 1024 units, 50 power steps beyond).
  The gates are torch's order: `h` (rnn), `r z n` (gru), `i f g o` (lstm).
- **Attention**: the packed `in_proj` (q, k, v stacked) and `out_proj`, and per head the
  Frobenius norms of its q, k, v and o slices. A transformer encoder is a group whose children
  are its layers `<id>.layer<k>`, each an `encoder_layer` node (self-attention, the
  feed-forward pair, both norms).
- **Tokens**: the tokenizer's numeric token weights, its per-nominal tables (the embedding
  table entries), the [cls] vector; the positional table's first L rows (L the node's
  sequence length), a weight when learned, a buffer when sinusoidal.

What an explanation reads of each (the unroll, the attention maps, the token norms, their
checks) is `probes_seq.py`'s; a transformer encoder is read through its layers.
"""
from __future__ import annotations

import math
from types import MappingProxyType

import numpy as np
from torch import nn

from . import probes_seq as PS
from .codec import num, nums, summary
from .readers import GraphError, _Reader, act_name, arr, fro, norms, table_entry

__all__ = ["SEQ"]

GATES = MappingProxyType({"rnn": ("h",), "gru": ("r", "z", "n"), "lstm": ("i", "f", "g", "o")})
EIG_MAX = 1024              # spectral radii by eigvals up to this size, power iteration beyond
POWER_STEPS = 50


def _matrix(site, values):
    """A nested list, or the shape and a summary past `json_list_max` numbers."""
    a = np.asarray(values, dtype=np.float64)
    if a.size <= site.codec.limits["json_list_max"]:
        return nums(a)
    return {"shape": [int(d) for d in a.shape], "summary": summary(a)}


def spectral_radius(a: np.ndarray) -> float:
    """max |λ| of a square matrix: its eigenvalues, or power iteration past `EIG_MAX`."""
    if a.shape[0] <= EIG_MAX:
        return num(np.max(np.abs(np.linalg.eigvals(a))))
    v, r = np.ones(a.shape[0]) / math.sqrt(a.shape[0]), 0.0
    for _ in range(POWER_STEPS):
        w = np.add.reduce(a * v[None, :], axis=1)
        r = math.sqrt(float(np.add.reduce(w * w)))
        if r == 0.0:
            break
        v = w / r
    return num(r)


class _Conv1d(_Reader):
    TYPE, VIEW = "conv1d", "conv"
    PROBE = PS.CONV1D
    FORMULA = "y[t,o] = b[o] + Σ_c Σ_k W[o,c,k]·x[t·s+k−p, c]"
    PAYLOAD = ("kernel", "bias", "channel_norms", "tap_norms", "pair_norms", "stride",
               "padding", "dilation")

    def config(self, block, module):
        conv = module.conv
        pad = conv.padding if isinstance(conv.padding, str) else int(conv.padding[0])
        return {"channels": int(conv.out_channels), "kernel": int(conv.kernel_size[0]),
                "stride": int(conv.stride[0]), "padding": pad,
                "dilation": int(conv.dilation[0])}

    def params(self, module, site):
        conv, held = module.conv, self.config(None, module)
        w = arr(conv.weight)
        return {"kernel": site.tensor(conv.weight, "kernel", "weight", ("out", "in", "tap")),
                "bias": site.tensor(conv.bias, "bias", "bias", ("out",)),
                "channel_norms": site.vector(np.sqrt(np.add.reduce(w * w, axis=(1, 2)))),
                "tap_norms": site.vector(np.sqrt(np.add.reduce(w * w, axis=(0, 1)))),
                "pair_norms": _matrix(site, np.sqrt(np.add.reduce(w * w, axis=2))),
                "stride": held["stride"], "padding": held["padding"],
                "dilation": held["dilation"]}


class _Pool1d(_Reader):
    TYPE, VIEW, FORMULA, PAYLOAD = "pool1d", "pool", "y[t,c] = max|mean_{m} x[t·m+m', c]", \
        ("kind", "size")
    PROBE = PS.POOL1D

    def config(self, block, module):
        size = module.pool.kernel_size
        return {"kind": "max" if isinstance(module.pool, nn.MaxPool1d) else "avg",
                "size": int(size[0] if isinstance(size, (tuple, list)) else size)}


class _Recurrent(_Reader):
    """`rnn`, `gru` or `lstm` — one instance per type."""

    VIEW = "recurrent"
    PROBE = PS.RECURRENT
    PAYLOAD = ("cell", "hidden", "layers", "directions", "gates", "return", "per_layer")
    ALIASES = MappingProxyType({"num_layers": "layers"})

    def __init__(self, kind: str, formula: str):
        self.TYPE, self.FORMULA = kind, formula

    def config(self, block, module):
        rnn = module.rnn
        held = {"hidden": int(rnn.hidden_size), "num_layers": int(rnn.num_layers),
                "bidirectional": bool(rnn.bidirectional), "return": str(module.ret)}
        if rnn.num_layers > 1:       # torch drops the between-layer dropout of one layer
            held["dropout"] = float(rnn.dropout)
        if self.TYPE == "rnn":
            held["nonlinearity"] = str(rnn.nonlinearity)
        return held

    def params(self, module, site):
        rnn = module.rnn
        dirs = 2 if rnn.bidirectional else 1
        return {"cell": self.TYPE, "hidden": int(rnn.hidden_size), "layers": int(rnn.num_layers),
                "directions": dirs, "gates": list(GATES[self.TYPE]), "return": str(module.ret),
                "per_layer": [self._layer(rnn, site, k, d) for k in range(rnn.num_layers)
                              for d in range(dirs)]}

    def _layer(self, rnn, site, k, d):
        torch_sfx, sfx = f"_l{k}" + ("_reverse" if d else ""), ".reverse" if d else ""
        w_ih, w_hh = (getattr(rnn, f"weight_{m}{torch_sfx}") for m in ("ih", "hh"))
        b_ih, b_hh = (getattr(rnn, f"bias_{m}{torch_sfx}", None) for m in ("ih", "hh"))
        labels = list(site.labels) if k == 0 and site.labels and \
            len(site.labels) == int(w_ih.shape[1]) else None
        return {"layer": k, "direction": "reverse" if d else "forward",
                "w_ih": site.tensor(w_ih, f"l{k}.w_ih{sfx}", "weight", ("gate_unit", "in")),
                "w_hh": site.tensor(w_hh, f"l{k}.w_hh{sfx}", "weight", ("gate_unit", "unit")),
                "b_ih": site.tensor(b_ih, f"l{k}.b_ih{sfx}", "bias", ("gate_unit",)),
                "b_hh": site.tensor(b_hh, f"l{k}.b_hh{sfx}", "bias", ("gate_unit",)),
                **self._gates(site, (w_ih, w_hh, b_ih, b_hh), int(rnn.hidden_size)),
                "input_labels": labels}

    def _gates(self, site, tensors, hidden) -> dict:
        """Per gate: the norms of its blocks, its input columns, its recurrent radius."""
        w_ih, w_hh, b_ih, b_hh = (None if t is None else arr(t) for t in tensors)
        out = {"gate_norms": {}, "input_columns": {}, "rho_hh": {}}
        for g, gate in enumerate(GATES[self.TYPE]):
            rows = slice(g * hidden, (g + 1) * hidden)
            bias = [b[rows] for b in (b_ih, b_hh) if b is not None]
            out["gate_norms"][gate] = {"ih": fro(w_ih[rows]), "hh": fro(w_hh[rows]),
                                       "b": fro(np.concatenate(bias)) if bias else 0.0}
            out["input_columns"][gate] = site.vector(norms(w_ih[rows], 0))
            out["rho_hh"][gate] = spectral_radius(w_hh[rows])
        return out


def attention_payload(attn, site, prefix: str = "") -> dict:
    """A multi-head attention's `heads`, `dim`, `head_dim`, `in_proj`, `out_proj`, `per_head`
    (the tensors registered under `<node>.<prefix>…`)."""
    if attn.in_proj_weight is None:
        raise GraphError(f"node {site.node}: the attention's q, k, v have separate widths; the "
                         "inspector reads a packed in_proj only")
    dim, heads = int(attn.embed_dim), int(attn.num_heads)
    d, w, o = dim // heads, arr(attn.in_proj_weight), arr(attn.out_proj.weight)
    per_head = [{"q": fro(w[h * d:(h + 1) * d]), "k": fro(w[dim + h * d:dim + (h + 1) * d]),
                 "v": fro(w[2 * dim + h * d:2 * dim + (h + 1) * d]),
                 "o": fro(o[:, h * d:(h + 1) * d])} for h in range(heads)]
    site.tensor(attn.in_proj_bias, f"{prefix}in_proj_bias", "bias", ("qkv",))
    site.tensor(attn.out_proj.bias, f"{prefix}out_proj_bias", "bias", ("out",))
    return {"heads": heads, "dim": dim, "head_dim": d,
            "in_proj": site.tensor(attn.in_proj_weight, f"{prefix}in_proj", "weight",
                                   ("qkv", "in")),
            "out_proj": site.tensor(attn.out_proj.weight, f"{prefix}out_proj", "weight",
                                    ("out", "in")),
            "per_head": per_head}


class _Attention(_Reader):
    TYPE, VIEW = "attention", "attention"
    PROBE = PS.ATTENTION
    FORMULA = "A_h = softmax(Q_h K_hᵀ/√d_h) ; y = W_o·[A_1 V_1 ; … ; A_H V_H]"
    PAYLOAD = ("heads", "dim", "head_dim", "in_proj", "out_proj", "per_head", "pool")

    def config(self, block, module):
        return {"dim": int(module.attn.embed_dim), "heads": int(module.attn.num_heads),
                "pool": bool(module.pool)}

    def params(self, module, site):
        return {**attention_payload(module.attn, site), "pool": bool(module.pool)}


class _AttentionPool(_Reader):
    TYPE, VIEW, PAYLOAD = "attention_pool", "attention", ("score", "query")
    PROBE = PS.ATTENTION_POOL
    FORMULA = "α = softmax_t(q·tanh(W h_t + b)) ; y = Σ_t α_t h_t"

    def config(self, block, module):
        return {"hidden": int(module.score.out_features)}

    def params(self, module, site):
        site_score = site.tensor(module.score.weight, "score", "weight", ("unit", "in"))
        site.tensor(module.score.bias, "score_bias", "bias", ("unit",))
        return {"score": site_score,
                "query": site.tensor(module.query.weight, "query", "weight", ("one", "unit"))}


def _layer_config(layer) -> dict:
    return {"d": int(layer.self_attn.embed_dim), "heads": int(layer.self_attn.num_heads),
            "ff": int(layer.linear1.out_features), "dropout": float(layer.dropout.p),
            "norm_first": bool(layer.norm_first), "activation": act_name(layer.activation)}


class _TransformerEncoder(_Reader):
    """A group of `encoder_layer` children `<id>.layer<k>` (module path `enc.layers.<k>`)."""

    TYPE, VIEW = "transformer_encoder", "group"
    FORMULA = "x ← x + MHA(LN₁ x) ; x ← x + FF(LN₂ x)"
    PAYLOAD = ("layers", "d", "heads", "ff", "norm_first")

    def config(self, block, module):
        held = _layer_config(module.enc.layers[0])
        held.pop("activation")
        return {**held, "layers": len(module.enc.layers)}

    def expand(self, ctx, block, module, place):
        ctx.node(self.TYPE, place, module, self.checked(block, module))
        last = place.node
        for k, layer in enumerate(module.enc.layers):
            child = place.child(f"layer{k}", f"enc.layers.{k}")
            ctx.node("encoder_layer", child, layer, _layer_config(layer))
            ctx.edge(last, child.node, "data")
            last = child.node
        return last


def _pair(site, lin, name) -> str:
    site.tensor(lin.bias, f"{name}_bias", "bias", ("out",))
    return site.tensor(lin.weight, name, "weight", ("out", "in"))


class _EncoderLayer(_Reader):
    TYPE, VIEW, CALL = "encoder_layer", "attention", "inner"
    PROBE = PS.ENCODER_LAYER
    FORMULA = "x ← x + MHA(LN₁ x) ; x ← x + FF(LN₂ x)"
    PAYLOAD = ("self_attn", "ff", "norm1", "norm2", "norm_first", "activation")

    def params(self, module, site):
        norms_ = {n: {"gamma": site.tensor(getattr(module, n).weight, f"{n}.gamma", "weight",
                                           ("unit",)),
                      "beta": site.tensor(getattr(module, n).bias, f"{n}.beta", "bias",
                                          ("unit",))} for n in ("norm1", "norm2")}
        ff = {"linear1": _pair(site, module.linear1, "ff.linear1"),
              "linear2": _pair(site, module.linear2, "ff.linear2"),
              "norms": {"linear1": fro(module.linear1.weight),
                        "linear2": fro(module.linear2.weight)}}
        return {"self_attn": attention_payload(module.self_attn, site, "self_attn."), "ff": ff,
                **norms_, "norm_first": bool(module.norm_first),
                "activation": act_name(module.activation)}


class _FeatureTokenizer(_Reader):
    """Tokens `[cls ; T_1 … T_n]`: numeric token j is `x_j·W_j + b_j`, a nominal token its own
    table's row (`tables`, one per nominal column in `x_cat` order)."""

    TYPE, VIEW, CALL = "feature_tokenizer", "tokens", "tokens"
    PROBE = PS.FEATURE_TOKENIZER
    FORMULA = "T_j = x_j·W_j + b_j ; tokens = [cls ; T_1 … T_n]"
    PAYLOAD = ("weight", "bias", "cls", "tables", "token_norms", "cls_norm")

    def config(self, block, module):
        return {"d": int(module.weight.shape[1]), "numeric": int(module.weight.shape[0]),
                "cardinalities": [int(e.num_embeddings) for e in module.embeddings]}

    def params(self, module, site):
        n = int(module.weight.shape[0])
        labels = list(site.labels or ())
        names = [labels[j] if j < len(labels) else f"x{j}" for j in range(n)]
        nominal = [labels[n + j] if n + j < len(labels) else str(j)
                   for j in range(len(module.embeddings))]
        return {"weight": site.tensor(module.weight, "weight", "weight", ("feature", "channel")),
                "bias": site.tensor(module.bias, "bias", "bias", ("feature", "channel")),
                "cls": site.tensor(module.cls, "cls", "weight", ("one", "token", "channel")),
                "tables": [table_entry(site, c, e.weight, k)
                           for k, (c, e) in enumerate(zip(nominal, module.embeddings))],
                "token_norms": {"labels": names, "weight": site.vector(norms(module.weight, 1)),
                                "bias": site.vector(norms(module.bias, 1))},
                "cls_norm": fro(module.cls)}


class _Positional(_Reader):
    TYPE, VIEW, FORMULA, PAYLOAD = "positional", "tokens", "x_t ← x_t + P_t", \
        ("kind", "pe", "learned")
    PROBE = PS.POSITIONAL

    def config(self, block, module):
        return {"kind": str(module.kind), "max_len": int(module.pe.shape[1])}

    def params(self, module, site):
        learned = isinstance(module.pe, nn.Parameter)
        steps = int(site.in_shape[0]) if site.in_shape and len(site.in_shape) > 1 else \
            int(module.pe.shape[1])
        return {"kind": str(module.kind), "learned": learned,
                "pe": site.tensor(module.pe[0, :steps], "pe", "weight" if learned else "buffer",
                                  ("step", "channel"))}


SEQ = (_Conv1d(), _Pool1d(),
       _Recurrent("rnn", "h_t = tanh(W_ih·x_t + b_ih + W_hh·h_{t−1} + b_hh)"),
       _Recurrent("gru", "r,z = σ(…) ; n = tanh(W_in·x+b_in + r⊙(W_hn·h+b_hn)) ; "
                         "h_t = (1−z)⊙n + z⊙h_{t−1}"),
       _Recurrent("lstm", "i,f,o = σ(…) ; g = tanh(…) ; c_t = f⊙c_{t−1} + i⊙g ; "
                          "h_t = o⊙tanh(c_t)"),
       _Attention(), _AttentionPool(), _TransformerEncoder(), _EncoderLayer(),
       _FeatureTokenizer(), _Positional())
