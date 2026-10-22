"""
The spec -> stages builder: `NetworkBuilder(spec, run_dir).stages()` reads `spec.blocks` in order
and returns one `BuiltStage` per block — the module, the call convention (`StageKind`) and the
per-sample output dims — and `initialise(model)` applies the spec's `init` (EJ-2: a builder
reads the blocks in order; no constructor with a dozen positional parameters).

This file is a splice source (D1 §4.1, generation v2.3). The emitter writes its `prelude` and
`builder` segments, whole, into `<package>/<model>/builder.py` (a `kind: support` file used by
the Model adapter) and resolves the package-relative imports from the names the segments use:
the block classes from the sibling `blocks_core.py` / `blocks_seq.py` / `blocks_graph.py` /
`blocks_gen.py`, and `NetSpec`, `Regularization`, `SampleSpec`, `StageKind`, `Task`, `TaskKind`
from `domain.py`.
The builder's tables and methods are cut by block segment (`# >>> segment: <name> ...` sections,
named as blocks.py's segments, and `custom`): the emitter keeps the sections of the segments it
spliced into the block files (`emit.select_segments`), so an emitted builder names only the
block classes its package holds and refuses any other block type by name. Markers
`# --8<-- name` / `# -->8-- name` delimit the splice segments; the imports above the first
marker are for a reader and a linter only.

What the builder guarantees (BUILD-PLAN R-B1, R-B9; D1 §7.3):

- it **never constructs the `Network`** (C9): `Network.build` in `<model>/network.py` seeds,
  asks this builder for the stages, constructs, and asks it to initialise;
- every block is made by its class's `from_params(width, params)` — the one factory of the
  block contract — with what the builder resolves from the contract added to the params:
  `cardinalities` (embedding: {column: cardinality}; feature_tokenizer: one per nominal column,
  in `x_cat` order), `kind` (rnn / gru / lstm), `out` (decoder: `recon_dim` by default),
  `target_affine` (a regression-family head, A37) and the built inner `modules` (and
  `projections`) of the composites;
- parameters are created in the order v2.2's `_build_one` created them — blocks in spec order,
  a composite's inner blocks before the composite, a residual projection after its repeat —
  so the seeded initial weights of every v2.2 spec are bit-identical (D1 §17.3);
- the start shape is `(n_numeric + 1 if autoregressive, seq = L on a window, graph on a graph
  sample)`, and each block's runtime refusal names the block id: a graph block off a graph
  sample, after `graph_pool` or on a sequence; `graph_pool` off level graph; a sequence block
  on a flat input; an embedding before a feature_tokenizer; a composite holding a block that
  is not a tensor stage; a last block that leaves a sequence, or node rows at level graph
  (the runtime twins of the compiler's W2, G2);
- every dropout rate a `regularization[kind=dropout]` declaration covers is the declaration's
  (`Regularization.blocks`; netspec-contract §3, R2): the builder reads the blocks with the
  declared rates written into them, so a ladder delta on the declaration moves the weights;
- a `custom` block's `code_ref` module is read from `<model>/custom/<stem>.py`, beside the
  spliced builder, where the emitter copied it (R-B9) — never from a run directory, where a
  stale copy may lie.

Contract: references/backend-contract.md §1 (`build_model = Network.build`),
          references/family-catalog.md §3 (params and shape rules).
"""
from __future__ import annotations

from blocks import (Activation, Attention, AttentionPool, BatchNorm, Concat,  # noqa: F401
                    Conv1d, CumulativeHead, Decoder, Dense, Dropout, EdgeDropout, Embedding,
                    FeatureTokenizer, Flatten, GATConv, GCNConv, GINConv, GraphPool, Head,
                    LayerNorm, Parallel, Pool, Pool1d, Positional, RecurrentBlock,
                    Reparameterize, Reshape, Residual, SAGEConv, Sequential,
                    TransformerEncoder, _prod)

# --8<-- prelude
import importlib.util
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

from torch import nn
# -->8-- prelude


# --8<-- builder
@dataclass(frozen=True)
class _Shape:
    """The running per-sample shape (+ graph rows?, pooled?, an embedding not yet joined)."""

    width: int
    seq: Any = None
    graph: bool = False
    pooled: bool = False
    emb: Any = None

    def dims(self) -> tuple:
        return (int(self.width),) if self.seq is None else (int(self.seq), int(self.width))


@dataclass(frozen=True, eq=False)
class BuiltStage:
    """One spec block, built; `out_dims` per sample (node, graph): (width,) | (seq, width)."""

    id: str
    type: str
    module: Any
    kind: Any
    out_dims: tuple


@dataclass(frozen=True, eq=False)
class _Made:
    module: Any
    kind: Any
    shape: _Shape
    dims: Any = None                 # the stage's own output dims when not `shape.dims()`


# >>> segment: recurrent conv attention attention_pool
# the types that read (B, L, F): torch reads a flat (B, F) as one unbatched sequence, silently
_NEEDS_SEQUENCE = frozenset({"rnn", "gru", "lstm", "conv1d", "pool1d", "attention",
                             "attention_pool", "transformer_encoder", "positional"})
# <<< segment
# >>> segment: graph_conv graph_pool graph_edges
_GRAPH_BLOCKS = frozenset({"gcn_conv", "sage_conv", "gat_conv", "gin_conv", "graph_pool",
                           "edge_dropout"})
# <<< segment

_INIT = {
    "kaiming_uniform": lambda w: nn.init.kaiming_uniform_(w, nonlinearity="relu"),
    "kaiming_normal": lambda w: nn.init.kaiming_normal_(w, nonlinearity="relu"),
    "xavier_uniform": nn.init.xavier_uniform_, "xavier_normal": nn.init.xavier_normal_,
    "orthogonal": nn.init.orthogonal_}
_RECURRENT_INIT = ("default", "orthogonal")


def _block_id(block) -> str:
    return str(block.get("id") or block.get("type"))


# the runtime twins of W2/G2: (the block cannot read the running shape, why) — named by id
_MISPLACED = (
    # >>> segment: recurrent conv attention attention_pool
    (lambda t, sh, sm: t in _NEEDS_SEQUENCE and sh.seq is None,
     "reads a sequence (B, L, F); it needs a window, a tokenizer or `return: all` before it"),
    # <<< segment
    # >>> segment: graph_conv graph_pool graph_edges
    (lambda t, sh, sm: t in _GRAPH_BLOCKS and not sh.graph,
     "needs a graph sample (data_contract.sample.shape graph); these samples are not graphs (W2)"),
    (lambda t, sh, sm: t in _GRAPH_BLOCKS and sh.pooled,
     "comes after graph_pool: the rows are graphs now, with no edges between them (W2)"),
    (lambda t, sh, sm: t in _GRAPH_BLOCKS and t != "edge_dropout" and sh.seq is not None,
     "receives a sequence per node; a graph block reads one feature vector per node"),
    # <<< segment
    # >>> segment: graph_pool
    (lambda t, sh, sm: t == "graph_pool" and not sm.is_graph_level,
     "is allowed only at sample.level graph; at level node each node is its own answer (W2)"),
    # <<< segment
    # >>> segment: attention
    (lambda t, sh, sm: t == "feature_tokenizer" and sh.emb is not None,
     "reads the nominals itself; an embedding block before it would embed them twice (G2)"),
    # <<< segment
)


def _refuse_misplaced(block, shape, sample):
    t = block.get("type")
    for wrong, why in _MISPLACED:
        if wrong(t, shape, sample):
            raise ValueError(f"block {_block_id(block)} ({t}) {why}")


def _refuse_output(block, shape, sample):
    """One vector per answer: per row, window, node or — after graph_pool — graph (G2, W2)."""
    if shape.seq is not None:
        raise ValueError(f"the last block {_block_id(block)} outputs a sequence {shape.dims()} "
                         "per sample; a reducer (pool, flatten, `return: last`) comes first (G2)")
    if sample.is_graph_level and not shape.pooled:
        raise ValueError("at sample.level graph each answer is a graph: a graph_pool block must "
                         "pool the node rows before the head (W2)")


# >>> segment: conv
def _conv_length(seq, block_params):
    """The output length of `conv1d` (torch's formula; `same` keeps L at stride 1)."""
    k, s = int(block_params["kernel"]), int(block_params.get("stride", 1))
    pad = block_params.get("padding", "same")
    if pad == "same" and s == 1:
        return seq
    pad = {"same": k // 2, "valid": 0}.get(pad, pad)
    return (int(seq) + 2 * int(pad) - k) // s + 1
# <<< segment


# >>> segment: custom
def _load_custom(code_ref, block_id):
    """`build(in_shape, params) -> module` of a custom block, from the package's own copy."""
    path = Path(__file__).resolve().parent / "custom" / f"{Path(str(code_ref)).stem}.py"
    if not path.is_file():
        raise ValueError(f"custom block {block_id}: its code_ref {code_ref!r} ships at {path} "
                         "(copied by the emitter, R-B9), which is missing; re-emit the package")
    spec = importlib.util.spec_from_file_location(f"custom_{path.stem}", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "build", None)):
        raise ValueError(f"{path} exposes no build(in_shape, params) -> module")
    return module.build
# <<< segment


def _initialise(model, init):
    """`init.scheme` on Linear weights (biases zeroed by default), N(0, 0.1) on embeddings,
    orthogonal `weight_hh_*` iff `recurrent: orthogonal`; registration order."""
    scheme, recurrent = init.get("scheme", "kaiming_uniform"), init.get("recurrent", "default")
    if scheme not in _INIT or recurrent not in _RECURRENT_INIT:
        raise ValueError(f"init scheme {scheme!r} / recurrent {recurrent!r}: the catalog lists "
                         f"{sorted(_INIT)} and recurrent {list(_RECURRENT_INIT)}")
    for m in model.modules():
        if isinstance(m, nn.Linear):
            _INIT[scheme](m.weight)
            if m.bias is not None and init.get("bias", "zeros") == "zeros":
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.1)
        elif isinstance(m, nn.RNNBase) and recurrent == "orthogonal":
            for name, weight in m.named_parameters():
                if name.startswith("weight_hh"):
                    nn.init.orthogonal_(weight)
    return model


class _CompositeBlocks:
    """The composites and `custom`: a part the builder holds, handed its `one`/`make` (EJ-18)."""

    def __init__(self, one, make):
        self._one, self._make = one, make

    # >>> segment: composite
    def _inner(self, block, inner, shape):
        """The inner blocks of a composite, built in order from `shape`: (modules, shape)."""
        mods, cur = [], shape
        for ib in inner or []:
            made = self._one(ib, cur)
            if made.kind is not StageKind.TENSOR:
                raise ValueError(f"composite block {_block_id(block)} holds a {ib.get('type')} "
                                 "block; a composite chains tensor blocks only (embedding, "
                                 "concat, tokenizer and graph blocks are top-level stages)")
            mods.append(made.module)
            cur = made.shape
        return mods, cur

    def _sequential(self, block, p, shape):
        p["modules"], out = self._inner(block, p.get("inner"), shape)
        return _Made(self._make(block, p, shape.width), StageKind.TENSOR, out)

    def _parallel(self, block, p, shape):
        branches = [self._inner(block, [ib], shape) for ib in p.get("inner") or []]
        if not branches:
            raise ValueError(f"parallel block {_block_id(block)} has no branch in params.inner")
        widths = [out.width for _, out in branches]
        if p.get("merge", "concat") == "sum" and len(set(widths)) > 1:
            raise ValueError(f"parallel block {_block_id(block)} sums branches of widths "
                             f"{widths}; `merge: sum` needs equal widths")
        p["modules"] = [mods[0] for mods, _ in branches]
        width = sum(widths) if p.get("merge", "concat") == "concat" else widths[0]
        return _Made(self._make(block, p, shape.width), StageKind.TENSOR,
                     replace(branches[0][1], width=width))

    def _residual(self, block, p, shape):
        repeats, projections, cur = [], [], shape
        for _ in range(int(p.get("repeat", 1))):
            mods, inner = self._inner(block, p.get("inner"), cur)
            if inner.width != cur.width and p.get("projection") != "auto":
                raise ValueError(f"residual block {_block_id(block)} changes width {cur.width} "
                                 f"-> {inner.width} and projection is not 'auto'; fix the spec")
            projections.append(nn.Linear(cur.width, inner.width) if inner.width != cur.width
                               else nn.Identity())
            repeats.append(mods)
            cur = inner
        p["modules"], p["projections"] = repeats, projections
        return _Made(self._make(block, p, shape.width), StageKind.TENSOR, cur)
    # <<< segment

    # >>> segment: custom
    def _custom(self, block, p, shape):
        m = _load_custom(p["code_ref"], _block_id(block))(tuple(shape.dims()), p)
        out = [int(x) for x in p["out_shape"]]
        return _Made(m, StageKind.TENSOR,
                     replace(shape, width=out[-1], seq=out[0] if len(out) == 2 else None))
    # <<< segment

    _COMPOSITE = MappingProxyType({                # type -> (its class, its method)
        # >>> segment: composite
        "sequential": (Sequential, _sequential), "residual": (Residual, _residual),
        "parallel": (Parallel, _parallel),
        # <<< segment
        # >>> segment: custom
        "custom": (None, _custom),
        # <<< segment
    })


class NetworkBuilder:
    """Walks `spec.blocks` once, tracking the running `_Shape`; `_DISPATCH` (C11) maps a type
    the package carries to its class and its method, `_CompositeBlocks._COMPOSITE` a composite's."""

    def __init__(self, spec, run_dir=None):
        self._spec = NetSpec.require(spec)        # run_dir: the port's, unread (R-B9)
        self._run_dir = run_dir
        self._blocks = tuple(Regularization.of(self._spec).blocks(self._spec.blocks))
        self._init = self._spec.init
        self._task = Task.from_spec(self._spec)
        self._sample = SampleSpec.from_spec(self._spec)
        contract = self._spec.feature_contract
        self._numeric = len(contract.numeric_columns)
        self._cards = {f.name: f.cardinality for f in contract.nominal_features}
        self._composites = _CompositeBlocks(self._one, self._make)

    def start(self) -> _Shape:
        """The shape of `x_num` per sample (D1 §7.3)."""
        window = self._sample.window
        width = self._numeric + (1 if window is not None and window.autoregressive else 0)
        return _Shape(width, seq=None if window is None else int(window.length),
                      graph=self._sample.graph is not None)

    def recon_dim(self) -> int:          # n_numeric + Σ cardinalities: the reconstruction width
        return self._numeric + sum(int(self._cardinality(c)) for c in self._cards)

    def stages(self) -> tuple:
        """One BuiltStage per spec block, in spec order."""
        shape, out = self.start(), []
        for block in self._blocks:
            made = self._one(block, shape)
            shape = made.shape
            out.append(BuiltStage(_block_id(block), block.get("type"), made.module,
                                  made.kind, tuple(made.dims or shape.dims())))
        if self._blocks:
            _refuse_output(self._blocks[-1], shape, self._sample)
        return tuple(out)

    def initialise(self, model):
        return _initialise(model, self._init)       # the spec's `init`; returns the model

    def _one(self, block, shape) -> _Made:
        own, part = self._DISPATCH.get(t := block.get("type")), _CompositeBlocks._COMPOSITE.get(t)
        if own is None and part is None:
            carried = ", ".join(sorted({*self._DISPATCH, *_CompositeBlocks._COMPOSITE}))
            raise ValueError(f"block {_block_id(block)}: this package carries no {t!r} block — "
                             f"it builds {carried}; another block's spec is built by a package "
                             "re-emitted for it (/vega-train re-emits on an architecture delta)")
        _refuse_misplaced(block, shape, self._sample)
        params = dict(block.get("params") or {})
        return (own[1](self, block, params, shape) if own is not None
                else part[1](self._composites, block, params, shape))

    def _make(self, block, params, width):
        record = self._DISPATCH.get(t := block["type"]) or _CompositeBlocks._COMPOSITE[t]
        return record[0].from_params(int(width), params)

    def _cardinality(self, column):
        card = self._cards.get(column)
        if card is None:
            raise ValueError(f"nominal column {column!r} has no cardinality in data_contract."
                             "features; the builder will not guess it (the compiler checks it)")
        return int(card)

    # >>> segment: norm simple attention
    def _same(self, block, p, shape):
        return _Made(self._make(block, p, shape.width), StageKind.TENSOR, shape)
    # <<< segment

    # >>> segment: dense
    def _dense(self, block, p, shape):
        m = self._make(block, p, shape.width)
        return _Made(m, StageKind.TENSOR, replace(shape, width=int(p["units"])))
    # <<< segment

    # >>> segment: head
    def _head(self, block, p, shape):
        cls = CumulativeHead if self._task.effective is TaskKind.ORDINAL else Head
        # >>> task: regression forecast
        if TargetAffine.applies(self._task):
            p["target_affine"] = True       # A37: a regression-family head ends on the affine
        # <<< task
        m = cls.from_params(int(shape.width), p)
        return _Made(m, StageKind.TENSOR, replace(shape, width=int(p["out"])))
    # <<< segment

    # >>> segment: embedding
    def _embedding(self, block, p, shape):
        cols = list(p.get("columns") or [])
        p["cardinalities"] = {c: self._cardinality(c) for c in cols}
        m = self._make(block, p, shape.width)
        emb = replace(shape, width=int(m.out_features))
        return _Made(m, StageKind.EMB, replace(shape, emb=int(m.out_features)), emb.dims())
    # <<< segment

    # >>> segment: concat
    def _concat(self, block, p, shape):
        width = shape.width + (shape.emb or 0)
        m = self._make(block, p, shape.width)
        return _Made(m, StageKind.CONCAT, replace(shape, width=width, emb=None))
    # <<< segment

    # >>> segment: attention
    def _tokenizer(self, block, p, shape):
        p["cardinalities"] = tuple(self._cardinality(c) for c in self._cards)
        m = self._make(block, p, shape.width)
        seq = shape.width + len(p["cardinalities"]) + 1
        return _Made(m, StageKind.TOKENS, replace(shape, width=int(p["d"]), seq=seq))
    # <<< segment

    # >>> segment: simple
    def _flatten(self, block, p, shape):
        out = replace(shape, width=shape.width * (shape.seq or 1), seq=None)
        return _Made(self._make(block, p, shape.width), StageKind.TENSOR, out)
    # <<< segment

    # >>> segment: simple
    def _reshape(self, block, p, shape):
        dims = [int(s) for s in p["shape"]]
        m = self._make(block, p, shape.width)
        out = (replace(shape, width=dims[1], seq=dims[0]) if len(dims) == 2
               else replace(shape, width=int(_prod(dims)), seq=None))
        return _Made(m, StageKind.TENSOR, out)
    # <<< segment

    # >>> segment: simple attention_pool
    def _readout(self, block, p, shape):
        return _Made(self._make(block, p, shape.width), StageKind.TENSOR,
                     replace(shape, seq=None))
    # <<< segment

    # >>> segment: attention
    def _attention(self, block, p, shape):
        m = self._make(block, p, shape.width)
        return _Made(m, StageKind.TENSOR, replace(shape, seq=None) if m.pool else shape)
    # <<< segment

    # >>> segment: conv
    def _conv1d(self, block, p, shape):
        m = self._make(block, p, shape.width)
        return _Made(m, StageKind.TENSOR, replace(shape, width=int(p["channels"]),
                                                  seq=_conv_length(shape.seq, p)))
    # <<< segment

    # >>> segment: conv
    def _pool1d(self, block, p, shape):
        m = self._make(block, p, shape.width)
        seq = max(1, int(shape.seq) // int(p.get("size", 2)))
        return _Made(m, StageKind.TENSOR, replace(shape, seq=seq))
    # <<< segment

    # >>> segment: recurrent
    def _recurrent(self, block, p, shape):
        p["kind"] = block["type"]
        m = self._make(block, p, shape.width)
        width = m.cfg.hidden * m.cfg.directions
        return _Made(m, StageKind.TENSOR,
                     replace(shape, width=width, seq=shape.seq if m.ret == "all" else None))
    # <<< segment

    # >>> segment: generative
    def _decoder(self, block, p, shape):
        p["out"] = int(p.get("out") or self.recon_dim())
        m = self._make(block, p, shape.width)
        return _Made(m, StageKind.TENSOR, replace(shape, width=p["out"]))
    # <<< segment

    # >>> segment: generative
    def _reparameterize(self, block, p, shape):
        m = self._make(block, p, shape.width)
        return _Made(m, StageKind.TENSOR, replace(shape, width=int(p["z"])))
    # <<< segment

    # >>> segment: graph_conv
    def _graph_conv(self, block, p, shape):
        m = self._make(block, p, shape.width)
        width = int(p["out"])
        if block["type"] == "gat_conv" and m.cfg.concat:
            width *= m.cfg.heads
        return _Made(m, StageKind.GRAPH, replace(shape, width=width))
    # <<< segment

    # >>> segment: graph_pool
    def _graph_pool(self, block, p, shape):
        return _Made(self._make(block, p, shape.width), StageKind.GRAPH_POOL,
                     replace(shape, pooled=True))
    # <<< segment

    # >>> segment: graph_edges
    def _edge_dropout(self, block, p, shape):
        return _Made(self._make(block, p, shape.width), StageKind.GRAPH_DROP, shape)
    # <<< segment

    _DISPATCH = MappingProxyType({                 # type -> (its class, its method)
        # >>> segment: dense
        "dense": (Dense, _dense),
        # <<< segment
        # >>> segment: embedding
        "embedding": (Embedding, _embedding),
        # <<< segment
        # >>> segment: concat
        "concat": (Concat, _concat),
        # <<< segment
        # >>> segment: norm
        "batchnorm": (BatchNorm, _same), "layernorm": (LayerNorm, _same),
        # <<< segment
        # >>> segment: simple
        "dropout": (Dropout, _same), "activation": (Activation, _same),
        "flatten": (Flatten, _flatten), "reshape": (Reshape, _reshape), "pool": (Pool, _readout),
        # <<< segment
        # >>> segment: attention
        "positional": (Positional, _same), "transformer_encoder": (TransformerEncoder, _same),
        "attention": (Attention, _attention), "feature_tokenizer": (FeatureTokenizer, _tokenizer),
        # <<< segment
        # >>> segment: attention_pool
        "attention_pool": (AttentionPool, _readout),
        # <<< segment
        # >>> segment: conv
        "conv1d": (Conv1d, _conv1d), "pool1d": (Pool1d, _pool1d),
        # <<< segment
        # >>> segment: recurrent
        "rnn": (RecurrentBlock, _recurrent), "gru": (RecurrentBlock, _recurrent),
        "lstm": (RecurrentBlock, _recurrent),
        # <<< segment
        # >>> segment: head
        "head": (Head, _head),
        # <<< segment
        # >>> segment: generative
        "decoder": (Decoder, _decoder), "reparameterize": (Reparameterize, _reparameterize),
        # <<< segment
        # >>> segment: graph_conv
        "gcn_conv": (GCNConv, _graph_conv), "sage_conv": (SAGEConv, _graph_conv),
        "gat_conv": (GATConv, _graph_conv), "gin_conv": (GINConv, _graph_conv),
        # <<< segment
        # >>> segment: graph_pool
        "graph_pool": (GraphPool, _graph_pool),
        # <<< segment
        # >>> segment: graph_edges
        "edge_dropout": (EdgeDropout, _edge_dropout),
        # <<< segment
    })
# -->8-- builder
