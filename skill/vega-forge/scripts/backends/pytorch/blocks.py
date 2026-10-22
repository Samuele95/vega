"""
PyTorch block implementations — one nn.Module construction per catalog block type (33 types).

This file is BOTH a runnable module (the unit tests and the smoke pass import it) and the
source the emitter copies from: `emit.py` splices the segments the spec uses into the emitted
package, so the emitted files stand alone (a bundle must run without the skills on disk). The
segment markers below are load-bearing — `# --8<-- <name>` opens a segment and
`# -->8-- <name>` closes it.

Where a segment goes. `SEGMENT_FILE` names the emitted support file of each segment (the v2.3
layout, D1 §4.1): `blocks_core.py` (prelude, dense, embedding, concat, norm, simple, composite,
head), `blocks_seq.py` (conv, recurrent, attention, attention_pool), `blocks_gen.py`
(generative) and `blocks_graph.py` (graph_ops, graph_conv, graph_pool, graph_edges).
`SEGMENTS_FOR` names the segments one block type needs, dependencies included (a graph conv
needs `graph_ops`), and `CLASS_FOR` names the class that implements it.

What a segment may use. Every segment may use `math`, `dataclass`, `torch`, `nn`, `F` and the
helpers of the `prelude` segment (`_act`, `_prod`), so every emitted file needs the prelude's
imports and those two helpers. `graph_conv` and `graph_pool` also use the `graph_ops`
helpers, which `SEGMENTS_FOR` lists beside them. Nothing else crosses a segment boundary.

The block contract (backend-contract §2, D1 §5, BUILD-PLAN R-B1–R-B4, R-B8):
- every block class has `@classmethod from_params(cls, width, params)`, the one factory the
  builder calls (`width` is the running last dimension; `params` the spec block's params plus
  what the builder resolves from the contract: `cardinalities` for `embedding` and
  `feature_tokenizer`, `kind` for the recurrent types, `out` for `decoder`, and the built
  inner `modules` for the composites), and an `__init__` of at most 5 parameters; blocks with
  more options take a frozen config dataclass defined in the same segment;
- the call convention of each block's `forward` is its stage kind (`STAGE_KIND_FOR`, the
  values of the domain's `StageKind`): `emb` m(x_cat), `concat` m(x_num, emb), `tokens`
  m(x_num, x_cat), `tensor` m(h), `graph` m(h, edge_index, edge_weight), `gdrop`
  m(edge_index, edge_weight) -> (edge_index, edge_weight), `gpool` m(h, graph_index,
  num_graphs) — the batch keys are passed positionally (R-B2);
- the attributes `coverage.json.inspection.surface[type]` lists are the inspection surface
  (R-B3): the inspector binds to them and to nothing else;
- no block writes an attribute during an eval forward pass (R-B4): nothing like a
  `last_weights` is kept; what an inspector needs is recomputed through a pure method
  (`AttentionPool.weights`, the graph convs' `coefficients`, `Reparameterize.divergence`);
- a block whose arithmetic is not all inside the torch modules the smoke's counter knows
  (`nn.Linear`, `nn.Conv1d`, `nn.MultiheadAttention`) counts its own multiply-adds with
  `madds(n_rows, n_edges=0, length=1)` — the recurrent, attention-pool and graph blocks, the
  feature tokenizer, the positional encoding and the cumulative head — for `n_rows` rows (or
  windows, or nodes), `n_edges` edges and `length` steps (the smoke's MAdds, D1 E19); the
  counter then reads no module inside it.

The v2.2 class names (`Recurrent`) were removed at the cutover (BUILD-PLAN §2.12, WP B14): the
builder constructs every block through `from_params`.

Contract: references/family-catalog.md §3 (params and shape rules),
          references/backend-contract.md §2 (what the PyTorch backend does with them),
          references/inspect-contract.md §3 (the inspection surface).
"""
from __future__ import annotations

# --8<-- prelude
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

_ACT = {"none": nn.Identity, "relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU,
        "tanh": nn.Tanh, "sigmoid": nn.Sigmoid, "leaky_relu": nn.LeakyReLU}


def _act(name):
    """The activation module for a catalog activation name."""
    key = (name or "none")
    if key not in _ACT:
        raise ValueError(f"unknown activation {name!r}; the catalog lists {sorted(_ACT)}")
    return _ACT[key]()


def _prod(xs):
    n = 1
    for x in xs:
        n *= int(x)
    return n
# -->8-- prelude


# --8<-- dense
class Dense(nn.Module):
    """`dense` — (..., in) -> (..., units)."""

    def __init__(self, in_features, units, activation="none", bias=True):
        super().__init__()
        self.linear = nn.Linear(int(in_features), int(units), bias=bool(bias))
        self.act = _act(activation)

    @classmethod
    def from_params(cls, width, params):
        return cls(width, params["units"], params.get("activation", "none"),
                   params.get("bias", True))

    def forward(self, x):
        return self.act(self.linear(x))
# -->8-- dense


# --8<-- embedding
class Embedding(nn.Module):
    """`embedding` — nominal ids (..., C) -> (..., sum(dim)).

    One nn.Embedding per nominal column with `cardinality` rows; the last row is the
    shaper's reserved unseen-category id, so an unseen value at inference is a valid index
    rather than a crash. The ids may carry any leading shape: rows `(B, C)` embed to
    `(B, sum(dim))`, windows `(B, L, C)` to `(B, L, sum(dim))`.
    """

    def __init__(self, columns, cardinalities, dims):
        super().__init__()
        self.columns = list(columns)
        self.embeddings = nn.ModuleList(
            [nn.Embedding(int(cardinalities[c]), int(dims[c])) for c in self.columns])
        self.out_features = sum(int(dims[c]) for c in self.columns)

    @classmethod
    def from_params(cls, width, params):
        """`params.cardinalities` ({column: cardinality}) comes from the contract's features
        (the builder resolves it); the emitter never guesses a cardinality."""
        columns = list(params.get("columns") or [])
        cards = params.get("cardinalities") or {}
        missing = [c for c in columns if c not in cards]
        if missing:
            raise ValueError(f"embedding columns {missing} have no cardinality; the builder "
                             "passes them from data_contract.features[].cardinality")
        dims = params.get("dims") or {c: int(params.get("dim", 8)) for c in columns}
        return cls(columns, cards, dims)

    def forward(self, x_cat):
        if x_cat is None or not self.columns:
            return None
        parts = [emb(x_cat[..., i]) for i, emb in enumerate(self.embeddings)]
        return torch.cat(parts, dim=-1)
# -->8-- embedding


# --8<-- concat
class Concat(nn.Module):
    """`concat` — structural: joins the numeric features and every embedding output."""

    def __init__(self):
        super().__init__()

    @classmethod
    def from_params(cls, width, params):
        return cls()

    def forward(self, x_num, emb):
        if emb is None:
            return x_num
        if x_num is None:
            return emb
        # a trace records the shapes of the batch it runs on, which already passed here
        if not torch.jit.is_tracing() and x_num.shape[:-1] != emb.shape[:-1]:
            raise ValueError(
                f"concat: the numeric features {tuple(x_num.shape)} and the embeddings "
                f"{tuple(emb.shape)} differ before their last dimension; a window's nominals "
                "must be (B, L, C) like its numeric part, not cut to the last step")
        return torch.cat([x_num, emb], dim=-1)
# -->8-- concat


# --8<-- norm
class BatchNorm(nn.Module):
    """`batchnorm` — shape-preserving. Normalises the last dimension over the batch."""

    def __init__(self, features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.bn = nn.BatchNorm1d(int(features), eps=float(eps), momentum=float(momentum))

    @classmethod
    def from_params(cls, width, params):
        return cls(width, params.get("eps", 1e-5), params.get("momentum", 0.1))

    def forward(self, x):
        if x.dim() == 2:
            return self.bn(x)
        return self.bn(x.transpose(1, -1)).transpose(1, -1)


class LayerNorm(nn.Module):
    """`layernorm` — shape-preserving."""

    def __init__(self, features, eps=1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(int(features), eps=float(eps))

    @classmethod
    def from_params(cls, width, params):
        return cls(width, params.get("eps", 1e-5))

    def forward(self, x):
        return self.ln(x)
# -->8-- norm


# --8<-- simple
class Dropout(nn.Module):
    """`dropout` — shape-preserving, inactive at eval."""

    def __init__(self, p=0.1):
        super().__init__()
        self.drop = nn.Dropout(float(p))

    @classmethod
    def from_params(cls, width, params):
        return cls(params.get("p", 0.1))

    def forward(self, x):
        return self.drop(x)


class Activation(nn.Module):
    """`activation` — shape-preserving."""

    def __init__(self, name="relu"):
        super().__init__()
        self.act = _act(name)

    @classmethod
    def from_params(cls, width, params):
        return cls(params.get("name", "relu"))

    def forward(self, x):
        return self.act(x)


class Flatten(nn.Module):
    """`flatten` — (B, ...) -> (B, prod(...))."""

    @classmethod
    def from_params(cls, width, params):
        return cls()

    def forward(self, x):
        return x.reshape(x.shape[0], -1)


class Reshape(nn.Module):
    """`reshape` — (B, n) -> (B, *shape)."""

    def __init__(self, shape):
        super().__init__()
        self.shape = [int(s) for s in shape]

    @classmethod
    def from_params(cls, width, params):
        return cls(params["shape"])

    def forward(self, x):
        return x.reshape(x.shape[0], *self.shape)


class Pool(nn.Module):
    """`pool` — (B, L, d) -> (B, d)."""

    def __init__(self, kind="mean"):
        super().__init__()
        if kind not in ("mean", "max", "cls"):
            raise ValueError(f"pool.kind must be mean|max|cls, got {kind!r}")
        self.kind = kind

    @classmethod
    def from_params(cls, width, params):
        return cls(params.get("kind", "mean"))

    def forward(self, x):
        if x.dim() == 2:
            return x
        if self.kind == "mean":
            return x.mean(dim=1)
        if self.kind == "max":
            return x.max(dim=1).values
        return x[:, 0]
# -->8-- simple


# --8<-- composite
class Sequential(nn.Module):
    """`sequential` — the chain of its inner blocks' rules."""

    def __init__(self, inner):
        super().__init__()
        self.inner = nn.ModuleList(inner)

    @classmethod
    def from_params(cls, width, params):
        """`params.modules`: the inner blocks, already built by the builder in spec order."""
        return cls(params["modules"])

    def forward(self, x):
        for m in self.inner:
            x = m(x)
        return x


class Residual(nn.Module):
    """`residual` — x + f(x), repeated; `projection: auto` inserts a dense when widths differ."""

    def __init__(self, blocks_per_repeat, projections):
        super().__init__()
        self.repeats = nn.ModuleList([nn.ModuleList(bs) for bs in blocks_per_repeat])
        self.projections = nn.ModuleList(projections)

    @classmethod
    def from_params(cls, width, params):
        """`params.modules`: one list of built inner blocks per repeat; `params.projections`:
        one `nn.Linear` (the width changed) or `nn.Identity` per repeat."""
        return cls(params["modules"], params["projections"])

    def forward(self, x):
        for bs, proj in zip(self.repeats, self.projections):
            h = x
            for m in bs:
                h = m(h)
            x = (proj(x) if not isinstance(proj, nn.Identity) else x) + h
        return x


class Parallel(nn.Module):
    """`parallel` — every branch sees the input; `concat` sums the widths, `sum` needs equal."""

    def __init__(self, branches, merge="concat"):
        super().__init__()
        if merge not in ("concat", "sum"):
            raise ValueError(f"parallel.merge must be concat|sum, got {merge!r}")
        self.branches = nn.ModuleList(branches)
        self.merge = merge

    @classmethod
    def from_params(cls, width, params):
        """`params.modules`: the built branches, in spec order."""
        return cls(params["modules"], params.get("merge", "concat"))

    def forward(self, x):
        outs = [b(x) for b in self.branches]
        if self.merge == "sum":
            acc = outs[0]
            for o in outs[1:]:
                acc = acc + o
            return acc
        return torch.cat(outs, dim=-1)
# -->8-- composite


# --8<-- conv
class Conv1d(nn.Module):
    """`conv1d` — (B, L, C) -> (B, L', C'); the channel axis is transposed in and out so the
    rest of the graph keeps the (B, L, F) convention every other block uses."""

    def __init__(self, in_channels, channels, kernel, stride=1, padding="same"):
        super().__init__()
        pad = padding if isinstance(padding, str) else int(padding)
        if pad == "same" and int(stride) != 1:
            pad = int(kernel) // 2
        self.conv = nn.Conv1d(int(in_channels), int(channels), int(kernel),
                              stride=int(stride), padding=pad)

    @classmethod
    def from_params(cls, width, params):
        return cls(width, params["channels"], params["kernel"], params.get("stride", 1),
                   params.get("padding", "same"))

    def forward(self, x):
        return self.conv(x.transpose(1, 2)).transpose(1, 2)


class Pool1d(nn.Module):
    """`pool1d` — (B, L, C) -> (B, L', C)."""

    def __init__(self, kind="max", size=2):
        super().__init__()
        cls = {"max": nn.MaxPool1d, "avg": nn.AvgPool1d}.get(kind)
        if cls is None:
            raise ValueError(f"pool1d.kind must be max|avg, got {kind!r}")
        self.pool = cls(int(size))

    @classmethod
    def from_params(cls, width, params):
        return cls(params.get("kind", "max"), params.get("size", 2))

    def forward(self, x):
        return self.pool(x.transpose(1, 2)).transpose(1, 2)
# -->8-- conv


# --8<-- recurrent
_RNN_GATES = {"rnn": 1, "gru": 3, "lstm": 4}


@dataclass(frozen=True)
class RecurrentConfig:
    """The resolved params of one `rnn` / `gru` / `lstm` block (family-catalog §3)."""

    kind: str                     # rnn | gru | lstm (the block type)
    hidden: int
    num_layers: int = 1           # params.num_layers, else params.layers (the old name)
    bidirectional: bool = False
    dropout: float = 0.0          # between layers; ignored when num_layers == 1
    ret: str = "last"             # params.return: last | all | mean | max
    nonlinearity: str = "tanh"    # rnn only: tanh | relu

    def __post_init__(self):
        if self.kind not in _RNN_GATES:
            raise ValueError(f"recurrent kind must be rnn|gru|lstm, got {self.kind!r}")
        if self.ret not in ("last", "all", "mean", "max"):
            raise ValueError(f"{self.kind}.return must be last|all|mean|max, got {self.ret!r}")
        if self.nonlinearity not in ("tanh", "relu"):
            raise ValueError(f"rnn.nonlinearity must be tanh|relu, got {self.nonlinearity!r}")
        if self.hidden < 1 or self.num_layers < 1:
            raise ValueError(f"{self.kind} needs hidden >= 1 and num_layers >= 1")

    @classmethod
    def from_params(cls, kind, params):
        layers = params.get("num_layers", params.get("layers", 1))
        return cls(kind=str(kind), hidden=int(params["hidden"]), num_layers=int(layers or 1),
                   bidirectional=bool(params.get("bidirectional", False)),
                   dropout=float(params.get("dropout", 0.0) or 0.0),
                   ret=str(params.get("return", "last")),
                   nonlinearity=str(params.get("nonlinearity", "tanh")))

    @property
    def directions(self):
        return 2 if self.bidirectional else 1


def _rnn_module(in_features, cfg):
    """The torch recurrent module a config describes (batch first, like every block)."""
    common = dict(batch_first=True, bidirectional=cfg.bidirectional,
                  num_layers=cfg.num_layers,
                  dropout=cfg.dropout if cfg.num_layers > 1 else 0.0)
    if cfg.kind == "rnn":
        return nn.RNN(int(in_features), cfg.hidden, nonlinearity=cfg.nonlinearity, **common)
    cls = nn.GRU if cfg.kind == "gru" else nn.LSTM
    return cls(int(in_features), cfg.hidden, **common)


class RecurrentBlock(nn.Module):
    """`rnn` / `gru` / `lstm` — (B, L, F) -> (B, H·dirs) for `return` last|mean|max, and
    (B, L, H·dirs) for `all`.

    `last` of a bidirectional block is the final state of each direction: the forward one at
    the last step and the backward one at the first, `cat(out[:, -1, :H], out[:, 0, H:])`,
    which equals `h_n` of the top layer (D1 E6: `out[:, -1]` alone read the backward direction
    after one step).
    """

    def __init__(self, in_features, cfg):
        super().__init__()
        self.cfg = cfg
        self.rnn = _rnn_module(in_features, cfg)

    @classmethod
    def from_params(cls, width, params):
        """`params.kind` is the block type (`rnn`, `gru` or `lstm`), added by the builder."""
        if params.get("kind") not in _RNN_GATES:
            raise ValueError("a recurrent block needs params.kind = rnn|gru|lstm (the block "
                             f"type, which the builder passes), got {params.get('kind')!r}")
        return cls(width, RecurrentConfig.from_params(params["kind"], params))

    @property
    def ret(self):
        return self.cfg.ret

    def forward(self, x):
        out, _ = self.rnn(x)
        return self._READ[self.cfg.ret](self, out)

    def _last(self, out):
        if not self.cfg.bidirectional:
            return out[:, -1]
        h = self.cfg.hidden
        return torch.cat([out[:, -1, :h], out[:, 0, h:]], dim=-1)

    def _all(self, out):
        return out

    def _mean(self, out):
        return out.mean(dim=1)

    def _max(self, out):
        return out.max(dim=1).values

    _READ = {"last": _last, "all": _all, "mean": _mean, "max": _max}

    def madds(self, n_rows, n_edges=0, length=1):
        """Multiply-adds of one forward pass over `n_rows` windows of `length` steps:
        per layer and direction, gates · L · H · (in + H)."""
        g, h, dirs = _RNN_GATES[self.cfg.kind], self.cfg.hidden, self.cfg.directions
        per, width = 0, int(self.rnn.input_size)
        for _ in range(self.cfg.num_layers):
            per += g * dirs * int(length) * h * (width + h)
            width = h * dirs
        return int(n_rows) * per

# -->8-- recurrent


# --8<-- attention
class Attention(nn.Module):
    """`attention` — (B, L, d) -> (B, d) when `pool: true`, else (B, L, d)."""

    def __init__(self, dim, heads=4, pool=False):
        super().__init__()
        self.attn = nn.MultiheadAttention(int(dim), int(heads), batch_first=True)
        self.pool = bool(pool)

    @classmethod
    def from_params(cls, width, params):
        return cls(params.get("dim", width), params.get("heads", 4), params.get("pool", False))

    def forward(self, x):
        h, _ = self.attn(x, x, x, need_weights=False)
        return h.mean(dim=1) if self.pool else h


class TransformerEncoder(nn.Module):
    """`transformer_encoder` — (B, L, d) -> (B, L, d), pre-norm (`norm_first`)."""

    def __init__(self, d, heads=4, layers=2, ff=None, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=int(d), nhead=int(heads), dim_feedforward=int(ff or 4 * int(d)),
            dropout=float(dropout), batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=int(layers))

    @classmethod
    def from_params(cls, width, params):
        return cls(params.get("d", width), params.get("heads", 4), params.get("layers", 2),
                   params.get("ff"), params.get("dropout", 0.1))

    def forward(self, x):
        return self.enc(x)


class Positional(nn.Module):
    """`positional` — shape-preserving."""

    def __init__(self, d, kind="sinusoidal", max_len=4096):
        super().__init__()
        self.kind = kind
        if kind == "learned":
            self.pe = nn.Parameter(torch.zeros(1, int(max_len), int(d)))
            nn.init.normal_(self.pe, std=0.02)
        elif kind == "sinusoidal":
            pe = torch.zeros(int(max_len), int(d))
            pos = torch.arange(0, int(max_len), dtype=torch.float32).unsqueeze(1)
            div = torch.exp(torch.arange(0, int(d), 2).float() * (-math.log(10000.0) / int(d)))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
            self.register_buffer("pe", pe.unsqueeze(0))
        else:
            raise ValueError(f"positional.kind must be sinusoidal|learned, got {kind!r}")

    @classmethod
    def from_params(cls, width, params):
        return cls(width, params.get("kind", "sinusoidal"), params.get("max_len", 4096))

    def forward(self, x):
        return x + self.pe[:, : x.shape[1]]

    def madds(self, n_rows, n_edges=0, length=1):
        return 0                     # an addition, sinusoidal or learned


class FeatureTokenizer(nn.Module):
    """`feature_tokenizer` — a row -> (B, 1 + n_num + n_cat, d): the [cls] token, one token
    per numeric column (`x_j·W_j + b_j`) and one per nominal column (its own embedding table
    of `cardinality` rows, the last one the unseen-category id). Stage kind `tokens`: it
    reads `x_num` and `x_cat` itself, so an ft-transformer needs no `embedding` block.

    Without `cardinalities` it tokenizes the numeric columns only (`forward(x_num)`).
    """

    def __init__(self, n_num, d, cardinalities=()):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(int(n_num), int(d)))
        self.bias = nn.Parameter(torch.zeros(int(n_num), int(d)))
        self.cls = nn.Parameter(torch.empty(1, 1, int(d)))
        nn.init.normal_(self.weight, std=0.02)
        nn.init.normal_(self.cls, std=0.02)
        self.cardinalities = tuple(int(c) for c in cardinalities)
        self.embeddings = nn.ModuleList([nn.Embedding(c, int(d)) for c in self.cardinalities])

    @classmethod
    def from_params(cls, width, params):
        """`params.cardinalities`: one cardinality per nominal column, in `x_cat` order (the
        builder resolves them from the contract); `width` is the numeric column count."""
        return cls(width, params["d"], tuple(params.get("cardinalities") or ()))

    def forward(self, x_num, x_cat=None):
        rows = (x_num if x_num is not None else x_cat).shape[0]
        tokens = [self.cls.expand(rows, -1, -1)]
        if x_num is not None and self.weight.shape[0] > 0:
            tokens.append(x_num.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0))
        if len(self.embeddings) > 0:
            if x_cat is None:
                raise ValueError(f"feature_tokenizer embeds {len(self.embeddings)} nominal "
                                 "columns and was called without x_cat")
            tokens.append(torch.stack(
                [emb(x_cat[:, j]) for j, emb in enumerate(self.embeddings)], dim=1))
        return torch.cat(tokens, dim=1)

    def madds(self, n_rows, n_edges=0, length=1):
        """One product per numeric column and token width (`x_j·W_j`); the nominal tokens
        are lookups."""
        n_num, d = self.weight.shape
        return int(n_rows) * int(n_num) * int(d)
# -->8-- attention


# --8<-- attention_pool
class AttentionPool(nn.Module):
    """`attention_pool` — (B, L, H) -> (B, H): one learned query weighs the steps,
    `α = softmax_t(q · tanh(W h_t + b))`, `y = Σ_t α_t h_t` (R-B8).

    `weights(x)` is the pure (B, L) map of α; nothing is kept between calls.
    """

    def __init__(self, in_features, hidden):
        super().__init__()
        self.score = nn.Linear(int(in_features), int(hidden))
        self.query = nn.Linear(int(hidden), 1, bias=False)

    @classmethod
    def from_params(cls, width, params):
        return cls(width, params.get("hidden", width))

    def weights(self, x):
        """α, the (B, L) softmax over the steps of `x` (B, L, H)."""
        return torch.softmax(self.query(torch.tanh(self.score(x))).squeeze(-1), dim=-1)

    def forward(self, x):
        return (self.weights(x).unsqueeze(-1) * x).sum(dim=1)

    def madds(self, n_rows, n_edges=0, length=1):
        """Per window: the score projection and query (L · (H·hidden + hidden)) and the
        weighted sum (L · H)."""
        h, k = self.score.in_features, self.score.out_features
        return int(n_rows) * int(length) * (h * k + k + h)
# -->8-- attention_pool


# --8<-- generative
class Decoder(nn.Module):
    """`decoder` — (B, z) -> (B, out); the encoder's dense widths reversed by default."""

    def __init__(self, in_features, widths, out_features, activation="gelu"):
        super().__init__()
        mods, prev = [], int(in_features)
        for w in widths or []:
            mods += [nn.Linear(prev, int(w)), _act(activation)]
            prev = int(w)
        mods.append(nn.Linear(prev, int(out_features)))
        self.net = nn.Sequential(*mods)

    @classmethod
    def from_params(cls, width, params):
        """`params.out`: the reconstruction width (recon_dim), resolved by the builder."""
        return cls(width, params.get("widths") or [], params["out"],
                   params.get("activation", "gelu"))

    def forward(self, x):
        return self.net(x)


class Reparameterize(nn.Module):
    """`reparameterize` — (B, 2z) -> (B, z): `mu` at inference, `mu + σ·ε` in training.

    The KL term of a training forward pass is kept in `kl` for the vae loss; an eval forward
    writes nothing (R-B4: a served model is never mutated by a request). `divergence(x)` is
    the same KL computed purely from an input, for whoever needs it outside training.
    """

    def __init__(self, z):
        super().__init__()
        self.z = int(z)
        self.kl = None

    @classmethod
    def from_params(cls, width, params):
        return cls(params["z"])

    def split(self, x):
        """(mu, logvar) of an input, logvar clamped to [-10, 10]."""
        return x[..., : self.z], x[..., self.z:].clamp(-10.0, 10.0)

    def divergence(self, x):
        """KL(q(z|x) ‖ N(0, I)), averaged over the rows; pure."""
        mu, logvar = self.split(x)
        return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1))

    def forward(self, x):
        mu, logvar = self.split(x)
        if not self.training:
            return mu
        self.kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1))
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
# -->8-- generative


# --8<-- graph_ops
def _weights_or_ones(edge_index, edge_weight, dtype):
    """The edge weights, or a weight of 1 per edge."""
    if edge_weight is not None:
        return edge_weight
    return torch.ones(edge_index.shape[1], dtype=dtype, device=edge_index.device)


def _with_self_loops(edge_index, weight, n):
    """The edges plus a loop (i, i) of weight 1 for every node that has none; an existing
    loop keeps its weight (PyG's `add_remaining_self_loops`). The loops come last."""
    loops = edge_index[0] == edge_index[1]
    has_loop = torch.zeros(n, dtype=torch.bool, device=edge_index.device)
    has_loop[edge_index[0][loops]] = True
    missing = torch.arange(n, device=edge_index.device)[~has_loop]
    edge_index = torch.cat([edge_index, missing.unsqueeze(0).expand(2, -1)], dim=1)
    weight = torch.cat([weight, weight.new_ones(missing.shape[0])])
    return edge_index, weight


def _degree(index, n, weight):
    """(n,) weighted in-degree: the sum of `weight` over the edges into each node."""
    return weight.new_zeros(n).index_add_(0, index, weight)


def _aggregate(msg, dst, n, reduce):
    """Messages (E, ...) reduced per destination into (n, ...): `sum` (index_add_), `mean`
    (sum over the message count, at least 1) or `max` (isolated nodes get 0)."""
    out = msg.new_zeros((n,) + tuple(msg.shape[1:]))
    if reduce == "max":
        index = dst.view((-1,) + (1,) * (msg.dim() - 1)).expand_as(msg)
        return out.scatter_reduce(0, index, msg, reduce="amax", include_self=False)
    out = out.index_add(0, dst, msg)
    if reduce == "mean":
        count = msg.new_zeros(n).index_add_(0, dst, msg.new_ones(dst.shape[0]))
        out = out / count.clamp(min=1.0).view((-1,) + (1,) * (msg.dim() - 1))
    return out


def _segment_softmax(score, dst, n):
    """(E, H) scores -> softmax over the edges sharing a destination (max-subtracted)."""
    index = dst.unsqueeze(-1).expand_as(score)
    top = score.new_zeros((n,) + tuple(score.shape[1:])).scatter_reduce(
        0, index, score.detach(), reduce="amax", include_self=False)
    e = torch.exp(score - top.index_select(0, dst))
    total = e.new_zeros(top.shape).index_add(0, dst, e)
    return e / total.index_select(0, dst)
# -->8-- graph_ops


# --8<-- graph_conv
class GCNConv(nn.Module):
    """`gcn_conv` — (N, F) + edges -> (N, out) (Kipf & Welling): `sym` `D^-½ Â D^-½ X W + b`,
    `rw` `D^-1 Â X W + b`; Â[i, j] is the weight of the edge j -> i, plus a unit loop on every
    node without one when `self_loops`, and D its weighted in-degree. `coefficients` are the
    entries of the normalised Â per edge: `out_i = Σ coef · (W h_src) + b`."""

    def __init__(self, in_features, out, normalize="sym", self_loops=True, bias=True):
        super().__init__()
        if normalize not in ("sym", "rw"):
            raise ValueError(f"gcn_conv.normalize must be sym|rw, got {normalize!r}")
        self.lin = nn.Linear(int(in_features), int(out), bias=False)
        if bias:
            self.bias = nn.Parameter(torch.zeros(int(out)))
        else:
            self.register_parameter("bias", None)
        self.normalize = normalize
        self.self_loops = bool(self_loops)

    @classmethod
    def from_params(cls, width, params):
        return cls(width, params["out"], params.get("normalize", "sym"),
                   params.get("self_loops", True), params.get("bias", True))

    def coefficients(self, edge_index, n, edge_weight=None):
        """(src, dst, coef) of every message, self loops included; pure."""
        w = _weights_or_ones(edge_index, edge_weight, self.lin.weight.dtype)
        if self.self_loops:
            edge_index, w = _with_self_loops(edge_index, w, n)
        src, dst = edge_index[0], edge_index[1]
        deg = _degree(dst, n, w)
        present = (deg > 0).to(deg.dtype)
        if self.normalize == "sym":
            inv = deg.clamp(min=1e-12).pow(-0.5) * present
            return src, dst, inv.index_select(0, src) * w * inv.index_select(0, dst)
        inv = deg.clamp(min=1e-12).reciprocal() * present
        return src, dst, w * inv.index_select(0, dst)

    def forward(self, x, edge_index, edge_weight=None):
        n = x.shape[0]
        src, dst, coef = self.coefficients(edge_index, n, edge_weight)
        h = self.lin(x)
        out = _aggregate(h.index_select(0, src) * coef.unsqueeze(-1), dst, n, "sum")
        return out if self.bias is None else out + self.bias

    def madds(self, n_rows, n_edges=0, length=1):
        """The projection and the weighted sum over edges and loops."""
        f, o = self.lin.in_features, self.lin.out_features
        loops = int(n_rows) if self.self_loops else 0
        return int(n_rows) * f * o + (int(n_edges) + loops) * o


@dataclass(frozen=True)
class SageConfig:
    """The resolved params of one `sage_conv` block."""

    out: int
    aggregator: str = "mean"      # mean | max | sum
    root_weight: bool = True
    normalize: bool = False       # L2-normalise each output row
    bias: bool = True

    def __post_init__(self):
        if self.aggregator not in ("mean", "max", "sum"):
            raise ValueError(f"sage_conv.aggregator must be mean|max|sum, got {self.aggregator!r}")

    @classmethod
    def from_params(cls, params):
        return cls(out=int(params["out"]), aggregator=str(params.get("aggregator", "mean")),
                   root_weight=bool(params.get("root_weight", True)),
                   normalize=bool(params.get("normalize", False)),
                   bias=bool(params.get("bias", True)))


class SAGEConv(nn.Module):
    """`sage_conv` — (N, F) + edges -> (N, out) (Hamilton et al.):
    `W_n · agg_{j→i} (w_ji x_j) + W_r x_i`, then an optional L2 normalisation.

    The root term is kept apart from the neighbours (`lin_root`, None without
    `root_weight`). `coefficients` covers the neighbours only: the aggregate is
    `Σ coef · x_src` for `sum` and `mean` (coef = w / in-edge count), and the element-wise
    `max` of `coef · x_src` for `max`.
    """

    def __init__(self, in_features, cfg):
        super().__init__()
        self.cfg = cfg
        self.aggregator, self.normalize = cfg.aggregator, cfg.normalize
        self.lin_neigh = nn.Linear(int(in_features), cfg.out, bias=cfg.bias)
        self.lin_root = (nn.Linear(int(in_features), cfg.out, bias=False)
                         if cfg.root_weight else None)

    @classmethod
    def from_params(cls, width, params):
        return cls(width, SageConfig.from_params(params))

    def coefficients(self, edge_index, n, edge_weight=None):
        """(src, dst, coef) of every neighbour message; pure."""
        w = _weights_or_ones(edge_index, edge_weight, self.lin_neigh.weight.dtype)
        src, dst = edge_index[0], edge_index[1]
        if self.cfg.aggregator != "mean":
            return src, dst, w
        count = w.new_zeros(n).index_add_(0, dst, w.new_ones(dst.shape[0]))
        return src, dst, w / count.clamp(min=1.0).index_select(0, dst)

    def forward(self, x, edge_index, edge_weight=None):
        n = x.shape[0]
        src, dst, coef = self.coefficients(edge_index, n, edge_weight)
        msg = x.index_select(0, src) * coef.unsqueeze(-1)
        agg = _aggregate(msg, dst, n, "max" if self.cfg.aggregator == "max" else "sum")
        out = self.lin_neigh(agg)
        if self.lin_root is not None:
            out = out + self.lin_root(x)
        return F.normalize(out, p=2.0, dim=-1) if self.cfg.normalize else out

    def madds(self, n_rows, n_edges=0, length=1):
        """The weighted neighbour messages (E·F) and the one or two projections."""
        f, o = self.lin_neigh.in_features, self.lin_neigh.out_features
        projections = 2 if self.lin_root is not None else 1
        return int(n_edges) * f + int(n_rows) * f * o * projections


@dataclass(frozen=True)
class GatConfig:
    """The resolved params of one `gat_conv` block."""

    out: int
    heads: int = 1
    concat: bool = True           # (N, heads·out) when true, else the heads' mean (N, out)
    attn_dropout: float = 0.0     # drops attention coefficients in training, not features
    negative_slope: float = 0.2
    self_loops: bool = True
    bias: bool = True

    @classmethod
    def from_params(cls, params):
        return cls(out=int(params["out"]), heads=int(params.get("heads", 1)),
                   concat=bool(params.get("concat", True)),
                   attn_dropout=float(params.get("attn_dropout", 0.0) or 0.0),
                   negative_slope=float(params.get("negative_slope", 0.2)),
                   self_loops=bool(params.get("self_loops", True)),
                   bias=bool(params.get("bias", True)))


class GATConv(nn.Module):
    """`gat_conv` — (N, F) + edges -> (N, heads·out) or (N, out) (Veličković et al.).

    Per head: `e_ji = LeakyReLU(a_srcᵀ W x_j + a_dstᵀ W x_i)`, `α_ji` the softmax of `e` over
    the edges into i, `h_i = Σ_j α_ji w_ji W x_j`. The edge weight multiplies α after the
    softmax, without renormalising (R-B3). `coefficients(h, ...)` returns that `α·w` per edge
    and head, self loops included, as the eval forward uses it.
    """

    def __init__(self, in_features, cfg):
        super().__init__()
        self.cfg = cfg
        self.heads, self.concat, self.negative_slope = cfg.heads, cfg.concat, cfg.negative_slope
        self.lin = nn.Linear(int(in_features), cfg.heads * cfg.out, bias=False)
        self.att_src = nn.Parameter(torch.empty(cfg.heads, cfg.out))
        self.att_dst = nn.Parameter(torch.empty(cfg.heads, cfg.out))
        if cfg.bias:
            self.bias = nn.Parameter(torch.zeros(cfg.heads * cfg.out if cfg.concat else cfg.out))
        else:
            self.register_parameter("bias", None)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    @classmethod
    def from_params(cls, width, params):
        return cls(width, GatConfig.from_params(params))

    def _alpha(self, hx, edge_index, edge_weight):
        """(src, dst, α (E, heads), w (E,)) over the projected features `hx` (N, heads, out)."""
        n = hx.shape[0]
        w = _weights_or_ones(edge_index, edge_weight, hx.dtype)
        if self.cfg.self_loops:
            edge_index, w = _with_self_loops(edge_index, w, n)
        src, dst = edge_index[0], edge_index[1]
        a_src = (hx * self.att_src).sum(dim=-1)
        a_dst = (hx * self.att_dst).sum(dim=-1)
        e = F.leaky_relu(a_src.index_select(0, src) + a_dst.index_select(0, dst),
                         self.cfg.negative_slope)
        return src, dst, _segment_softmax(e, dst, n), w

    def coefficients(self, h, edge_index, edge_weight=None):
        """(src, dst, α·w (E, heads)) for the block input `h` (N, F); pure, no dropout."""
        src, dst, alpha, w = self._alpha(self._project(h), edge_index, edge_weight)
        return src, dst, alpha * w.unsqueeze(-1)

    def _project(self, x):
        return self.lin(x).view(x.shape[0], self.cfg.heads, self.cfg.out)

    def forward(self, x, edge_index, edge_weight=None):
        n = x.shape[0]
        hx = self._project(x)
        src, dst, alpha, w = self._alpha(hx, edge_index, edge_weight)
        alpha = F.dropout(alpha, p=self.cfg.attn_dropout, training=self.training)
        coef = (alpha * w.unsqueeze(-1)).unsqueeze(-1)
        out = _aggregate(hx.index_select(0, src) * coef, dst, n, "sum")
        out = out.reshape(n, -1) if self.cfg.concat else out.mean(dim=1)
        return out if self.bias is None else out + self.bias

    def madds(self, n_rows, n_edges=0, length=1):
        """The projection, the two attention logits per node and the weighted sum."""
        f, width = self.lin.in_features, self.cfg.heads * self.cfg.out
        loops = int(n_rows) if self.cfg.self_loops else 0
        return int(n_rows) * (f * width + 2 * width) + (int(n_edges) + loops) * width


class GINConv(nn.Module):
    """`gin_conv` — (N, F) + edges -> (N, out) (Xu et al.):
    `MLP((1 + ε) x_i + Σ_{j→i} w_ji x_j)`, the MLP `mlp_layers` Linear layers with ReLU between.

    ε is a parameter iff `train_eps`, else a buffer. `coefficients` lists the neighbour
    messages and then the self terms `(i, i, 1 + ε)`, so the MLP input is `Σ coef · x_src`.
    """

    def __init__(self, in_features, out, eps=0.0, train_eps=False, mlp_layers=2):
        super().__init__()
        if int(mlp_layers) < 1:
            raise ValueError(f"gin_conv.mlp_layers must be >= 1, got {mlp_layers!r}")
        mods = [nn.Linear(int(in_features), int(out))]
        for _ in range(int(mlp_layers) - 1):
            mods += [nn.ReLU(), nn.Linear(int(out), int(out))]
        self.mlp = nn.Sequential(*mods)
        if train_eps:
            self.eps = nn.Parameter(torch.tensor([float(eps)]))
        else:
            self.register_buffer("eps", torch.tensor([float(eps)]))

    @classmethod
    def from_params(cls, width, params):
        return cls(width, params["out"], params.get("eps", 0.0), params.get("train_eps", False),
                   params.get("mlp_layers", 2))

    def coefficients(self, edge_index, n, edge_weight=None):
        """(src, dst, coef): the neighbour messages, then one self term per node; pure."""
        w = _weights_or_ones(edge_index, edge_weight, self.eps.dtype)
        nodes = torch.arange(n, device=edge_index.device)
        src = torch.cat([edge_index[0], nodes])
        dst = torch.cat([edge_index[1], nodes])
        return src, dst, torch.cat([w, (1 + self.eps).expand(n)])

    def forward(self, x, edge_index, edge_weight=None):
        n = x.shape[0]
        w = _weights_or_ones(edge_index, edge_weight, x.dtype)
        msg = x.index_select(0, edge_index[0]) * w.unsqueeze(-1)
        return self.mlp((1 + self.eps) * x + _aggregate(msg, edge_index[1], n, "sum"))

    def madds(self, n_rows, n_edges=0, length=1):
        """The weighted sum and the (1 + ε) scale (E·F + N·F) and the MLP."""
        f = self.mlp[0].in_features
        mlp = sum(m.in_features * m.out_features for m in self.mlp if isinstance(m, nn.Linear))
        return int(n_edges) * f + int(n_rows) * (f + mlp)
# -->8-- graph_conv


# --8<-- graph_pool
class GraphPool(nn.Module):
    """`graph_pool` — node rows (N, F) + graph_index (N,) -> graph rows (G, F), by `readout`
    mean | max | sum (an empty graph reads 0). Level graph only (W2)."""

    def __init__(self, readout="mean"):
        super().__init__()
        if readout not in ("mean", "max", "sum"):
            raise ValueError(f"graph_pool.readout must be mean|max|sum, got {readout!r}")
        self.readout = readout

    @classmethod
    def from_params(cls, width, params):
        return cls(params.get("readout", "mean"))

    def forward(self, x, graph_index, num_graphs):
        return _aggregate(x, graph_index, num_graphs, self.readout)

    def madds(self, n_rows, n_edges=0, length=1):
        return 0                     # a readout adds (and at most divides)
# -->8-- graph_pool


# --8<-- graph_edges
class EdgeDropout(nn.Module):
    """`edge_dropout` — (edge_index, edge_weight) -> the same with each edge kept with
    probability 1 − p, in training only; the identity at eval. Node features are untouched,
    and the kept edges replace the batch graph for every later block of the pass."""

    def __init__(self, p=0.1):
        super().__init__()
        if not 0.0 <= float(p) < 1.0:
            raise ValueError(f"edge_dropout.p must be in [0, 1), got {p!r}")
        self.p = float(p)

    @classmethod
    def from_params(cls, width, params):
        return cls(params.get("p", 0.1))

    def forward(self, edge_index, edge_weight=None):
        if not self.training or self.p == 0.0:
            return edge_index, edge_weight
        keep = torch.rand(edge_index.shape[1], device=edge_index.device) >= self.p
        return edge_index[:, keep], (None if edge_weight is None else edge_weight[keep])

    def madds(self, n_rows, n_edges=0, length=1):
        return 0
# -->8-- graph_edges


# --8<-- head
class Head(nn.Module):
    """`head` — (..., in) -> (..., out). The activation is the catalog's, and for every
    supervised task row that is `none`: the loss applies the link function, never the head.
    `bias: false` is the deep-svdd default (a bias lets the hypersphere collapse).

    A regression-family head (`target_affine`, which the builder resolves from the task, A37)
    ends on the fixed train-fitted target affine `y = act(z) · target_scale + target_shift`, one
    value per output channel: buffers, never trained, identity until `fit_target` writes the
    train target's mean and standard deviation once (`target_fitted`); a checkpoint carries all
    three, so a resumed run keeps them. The network learns in standardised units and answers in
    the target's own."""

    def __init__(self, in_features, out, activation="none", bias=True, target_affine=False):
        super().__init__()
        self.linear = nn.Linear(int(in_features), int(out), bias=bool(bias))
        self.act = _act(activation)
        self.affine = bool(target_affine)
        if self.affine:
            self.register_buffer("target_shift", torch.zeros(int(out)))
            self.register_buffer("target_scale", torch.ones(int(out)))
            self.register_buffer("target_fitted", torch.zeros((), dtype=torch.bool))

    @classmethod
    def from_params(cls, width, params):
        return cls(width, params["out"], params.get("activation", "none"),
                   params.get("bias", True), params.get("target_affine", False))

    @torch.no_grad()
    def fit_target(self, shift, scale) -> None:
        """Write the train target's affine (one value per output channel) and mark it fitted."""
        if not self.affine:
            raise ValueError("this head carries no target affine: only a regression-family "
                             "head ends on one (A37)")
        self.target_shift.copy_(torch.as_tensor(shift, dtype=self.target_shift.dtype))
        self.target_scale.copy_(torch.as_tensor(scale, dtype=self.target_scale.dtype))
        self.target_fitted.fill_(True)

    def forward(self, x):
        y = self.act(self.linear(x))
        return y * self.target_scale + self.target_shift if self.affine else y


class CumulativeHead(nn.Module):
    """`head` under an ordinal task — (..., in) -> (..., K−1) cumulative logits, rank
    consistent: one shared score `s(x) = w·x` and ordered thresholds,
    `z_j = s(x) + θ_j` with `θ_0 ≥ θ_1 ≥ …`, so `σ(z_j) = P(y > j)` decreases in j and every
    class probability `σ(z_{j−1}) − σ(z_j)` is non-negative.

    `score` is the shared `nn.Linear(in, 1, bias=False)` (the thresholds are the
    intercepts), `first` and `gaps` the threshold parameters, `thresholds()` their θ =
    `first − cumsum([0, softplus(gaps)])`. `linear` (the same module) and `act` (the identity)
    are the surface every `head` exposes, so a head reader reads either head.
    """

    def __init__(self, in_features, n_thresholds):
        super().__init__()
        if int(n_thresholds) < 1:
            raise ValueError(f"an ordinal head needs K−1 >= 1 thresholds, got {n_thresholds!r}")
        self.score = nn.Linear(int(in_features), 1, bias=False)
        self.act = nn.Identity()
        self.first = nn.Parameter(torch.zeros(1))
        # softplus(0.5413) = 1.0: the thresholds start one logit apart
        self.gaps = nn.Parameter(torch.full((int(n_thresholds) - 1,), 0.5413))

    @classmethod
    def from_params(cls, width, params):
        """`params.out` is K−1, the head width the task row forces for ordinal."""
        return cls(width, params["out"])

    @property
    def linear(self):
        return self.score

    def thresholds(self):
        steps = torch.cat([self.first.new_zeros(1), F.softplus(self.gaps)]).cumsum(0)
        return self.first - steps

    def forward(self, x):
        return self.act(self.score(x)) + self.thresholds()

    def madds(self, n_rows, n_edges=0, length=1):
        """The shared score `w·x` per row (the thresholds are additions)."""
        return int(n_rows) * int(length) * int(self.score.in_features)
# -->8-- head


# ------------------------------------------------------------------ segments the emitter uses

SEGMENTS_FOR = {
    "dense": ["dense"], "embedding": ["embedding"], "concat": ["concat"],
    "batchnorm": ["norm"], "layernorm": ["norm"],
    "dropout": ["simple"], "activation": ["simple"], "flatten": ["simple"],
    "reshape": ["simple"], "pool": ["simple"],
    "sequential": ["composite"], "residual": ["composite", "dense"], "parallel": ["composite"],
    "conv1d": ["conv"], "pool1d": ["conv"],
    "rnn": ["recurrent"], "gru": ["recurrent"], "lstm": ["recurrent"],
    "attention": ["attention"], "transformer_encoder": ["attention"],
    "positional": ["attention"], "feature_tokenizer": ["attention"],
    "attention_pool": ["attention_pool"],
    "decoder": ["generative"], "reparameterize": ["generative"],
    "gcn_conv": ["graph_ops", "graph_conv"], "sage_conv": ["graph_ops", "graph_conv"],
    "gat_conv": ["graph_ops", "graph_conv"], "gin_conv": ["graph_ops", "graph_conv"],
    "graph_pool": ["graph_ops", "graph_pool"], "edge_dropout": ["graph_edges"],
    "head": ["head"], "custom": [],
}

SEGMENT_FILE = {
    "prelude": "blocks_core", "dense": "blocks_core", "embedding": "blocks_core",
    "concat": "blocks_core", "norm": "blocks_core", "simple": "blocks_core",
    "composite": "blocks_core", "head": "blocks_core",
    "conv": "blocks_seq", "recurrent": "blocks_seq", "attention": "blocks_seq",
    "attention_pool": "blocks_seq",
    "generative": "blocks_gen",
    "graph_ops": "blocks_graph", "graph_conv": "blocks_graph", "graph_pool": "blocks_graph",
    "graph_edges": "blocks_graph",
}

CLASS_FOR = {
    "dense": "Dense", "embedding": "Embedding", "concat": "Concat",
    "batchnorm": "BatchNorm", "layernorm": "LayerNorm", "dropout": "Dropout",
    "activation": "Activation", "flatten": "Flatten", "reshape": "Reshape", "pool": "Pool",
    "sequential": "Sequential", "residual": "Residual", "parallel": "Parallel",
    "conv1d": "Conv1d", "pool1d": "Pool1d",
    "rnn": "RecurrentBlock", "gru": "RecurrentBlock", "lstm": "RecurrentBlock",
    "attention": "Attention", "attention_pool": "AttentionPool",
    "transformer_encoder": "TransformerEncoder", "positional": "Positional",
    "feature_tokenizer": "FeatureTokenizer",
    "decoder": "Decoder", "reparameterize": "Reparameterize",
    "gcn_conv": "GCNConv", "sage_conv": "SAGEConv", "gat_conv": "GATConv",
    "gin_conv": "GINConv", "graph_pool": "GraphPool", "edge_dropout": "EdgeDropout",
    "head": "Head",
}

# a type implemented by more than one class, by the variant names of
# coverage.json inspection.surface_variants: the builder picks CumulativeHead for `head`
# when the effective task is ordinal (D1 §7.3)
CLASS_VARIANTS = {
    "head": {"cumulative": "CumulativeHead"},
}

# the call convention of each block's forward (the values of the domain's StageKind, R-B1)
STAGE_KIND_FOR = {
    "embedding": "emb", "concat": "concat", "feature_tokenizer": "tokens",
    "gcn_conv": "graph", "sage_conv": "graph", "gat_conv": "graph", "gin_conv": "graph",
    "edge_dropout": "gdrop", "graph_pool": "gpool",
    "dense": "tensor", "batchnorm": "tensor", "layernorm": "tensor", "dropout": "tensor",
    "activation": "tensor", "flatten": "tensor", "reshape": "tensor", "pool": "tensor",
    "sequential": "tensor", "residual": "tensor", "parallel": "tensor",
    "conv1d": "tensor", "pool1d": "tensor", "rnn": "tensor", "gru": "tensor", "lstm": "tensor",
    "attention": "tensor", "attention_pool": "tensor", "transformer_encoder": "tensor",
    "positional": "tensor", "decoder": "tensor", "reparameterize": "tensor",
    "head": "tensor", "custom": "tensor",
}
