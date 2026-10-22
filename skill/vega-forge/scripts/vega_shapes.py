"""
vega_shapes — the torch-free shape walk behind the compiler's rule G2 (D1 §14.1, R-A14).

The emitter's smoke pass is the authority on shapes: it builds the network and runs a real
batch. What it reports on a failure is a framework traceback from inside a block, though, and a
spec author needs a sentence that names the block and the fix. This walk is the part of the
shape algebra that can be settled from the spec alone — not widths, only **axes**:

  * a window sample enters as `(B, L, F)`: one step axis beyond the features, which must be
    reduced before the head (`pool`, `flatten`, `attention` with `pool: true`,
    `attention_pool`, or a recurrent block whose `return` is not `all`);
  * a row sample enters as `(B, F)`: a sequence block (the catalog's `window_only` types) has
    no step axis to read there, and a block that reads a step or token axis (`transformer_encoder`,
    `attention`, `positional`) needs one a `feature_tokenizer` or a `reshape` put there first;
  * a graph sample enters as `(N, F)` over nodes: at level graph `graph_pool` turns node rows
    into graph rows, and no convolution may follow it (the edges address nodes that are gone);
  * an `embedding` placed before a `feature_tokenizer` is refused: the tokenizer reads the
    nominal columns itself (R-A14), so the embedding would feed it ids twice.

The table `_RULES` is the catalog's `blocks[*].shape_rule` strings made behaviour, one row per
block type, and `ShapeWalk` is a value: it holds the sample it walks and returns findings; it
prints nothing and raises nothing (the compiler turns findings into `ERROR:` markers).

Which rule refuses what: `graph_pool` present at level node (a reducer over the nodes, when
every node is a sample) and absent at level graph is W2's; the order of the blocks around it
is G2's, here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

__all__ = ["ShapeWalk", "ShapeFinding", "AXIS_REDUCERS"]

#: what reduces a step or token axis before a head, in the words a finding uses
AXIS_REDUCERS = ("pool", "flatten", "attention with pool: true", "attention_pool",
                 "a recurrent block whose return is not all")

_RECURRENT = ("rnn", "gru", "lstm")
_GRAPH_AFTER_POOL = ("gcn_conv", "sage_conv", "gat_conv", "gin_conv", "edge_dropout",
                     "graph_pool")

# type -> (reads a step or token axis, effect on the axes). The effects:
#   keep     the axes pass through;          reduce    the step/token axis is summed away;
#   maybe    reduce when one is there, else a no-op (pool and flatten on a (B, F) input);
#   add      a token axis appears;            reshape / custom   the axes the params declare;
#   return   recurrent: reduce unless return == all;  attention: reduce iff pool is true;
#   gpool    node rows become graph rows.
_RULES: Mapping[str, tuple] = {
    "dense": (False, "keep"), "embedding": (False, "keep"), "concat": (False, "keep"),
    "batchnorm": (False, "keep"), "layernorm": (False, "keep"), "dropout": (False, "keep"),
    "activation": (False, "keep"), "head": (False, "keep"), "decoder": (False, "keep"),
    "reparameterize": (False, "keep"), "positional": (True, "keep"),
    "conv1d": (True, "keep"), "pool1d": (True, "keep"), "transformer_encoder": (True, "keep"),
    "rnn": (True, "return"), "gru": (True, "return"), "lstm": (True, "return"),
    "attention": (True, "attention"), "attention_pool": (True, "reduce"),
    "pool": (False, "maybe"), "flatten": (False, "maybe"),
    "feature_tokenizer": (False, "add"), "reshape": (False, "reshape"),
    "custom": (False, "custom"),
    "gcn_conv": (False, "keep"), "sage_conv": (False, "keep"), "gat_conv": (False, "keep"),
    "gin_conv": (False, "keep"), "edge_dropout": (False, "keep"), "graph_pool": (False, "gpool"),
}


@dataclass(frozen=True)
class ShapeFinding:
    """One place the walk found an axis the next block cannot take."""

    where: str
    kind: str
    message: str


@dataclass(frozen=True)
class _Axes:
    extra: int = 0          # step or token axes beyond (batch-or-node, features)
    pooled: bool = False    # graph level: the node rows are graph rows now

    def with_extra(self, n: int) -> "_Axes":
        return _Axes(max(0, int(n)), self.pooled)


@dataclass(frozen=True)
class _Walked:
    axes: _Axes
    findings: tuple = ()
    embedded: bool = False   # an embedding has been seen on the way


@dataclass(frozen=True)
class ShapeWalk:
    """The walk over one spec's blocks, for one sample shape.

    `window_only` is the set of block types the catalog marks `window_only` (read from
    `catalog.json`, never hard-coded here), so a new sequence block is a catalog row and not an
    edit of this module.
    """

    shape: str
    level: str | None = None
    window_only: frozenset = field(default_factory=frozenset)

    @staticmethod
    def for_spec(spec: Mapping[str, Any], catalog_blocks: Mapping[str, Any]) -> "ShapeWalk":
        sample = ((spec.get("data_contract") or {}).get("sample")) or {}
        only = frozenset(k for k, v in (catalog_blocks or {}).items()
                         if isinstance(v, Mapping) and v.get("window_only"))
        return ShapeWalk(shape=str(sample.get("shape") or "row"), level=sample.get("level"),
                         window_only=only)

    def walk(self, blocks: Sequence[Any]) -> tuple:
        """Every finding along the blocks, in order."""
        start = _Walked(_Axes(1 if self.shape == "window" else 0))
        return self._chain(blocks, start, "").findings

    # -- the chain and one block

    def _chain(self, blocks, state: _Walked, prefix: str) -> _Walked:
        """The blocks in order; `where` names each as `walk_blocks` does."""
        for i, block in enumerate(blocks or ()):
            if isinstance(block, Mapping):
                state = self._block(block, state, f"{prefix}blocks[{block.get('id', i)}]")
        return state

    def _block(self, block: Mapping, state: _Walked, where: str) -> _Walked:
        kind = block.get("type")
        params = block.get("params") or {}
        if kind in ("residual", "sequential", "parallel"):
            return self._composite(kind, params, state, where)
        found = list(self._refusals(kind, params, state, where))
        axes = self._effect(kind, params, state.axes)
        return _Walked(axes, state.findings + tuple(found),
                       state.embedded or kind == "embedding")

    def _refusals(self, kind, params, state: _Walked, where):
        axes = state.axes
        if kind in self.window_only and self.shape != "window":
            yield ShapeFinding(where, "sequence_block",
                               f"{kind!r} is a sequence block and the sample is "
                               f"{self.shape!r}: there is no step axis to read — a sequence "
                               "block needs data_contract.sample.shape window")
        elif _RULES.get(kind, (False,))[0] and axes.extra == 0:
            yield ShapeFinding(where, "needs_axis",
                               f"{kind!r} reads a step or token axis, and none is left here "
                               "(the input is (rows, features)): on a window the axis was "
                               "already reduced by an earlier block; on rows a "
                               "feature_tokenizer or a reshape must create it first")
        if kind == "feature_tokenizer" and state.embedded:
            yield ShapeFinding(where, "embedding_before_tokenizer",
                               "an embedding block precedes the feature_tokenizer, which reads "
                               "the nominal columns itself (one token per nominal from its own "
                               "table) — drop the embedding (and its concat)")
        if kind == "head" and axes.extra > 0:
            what = "step" if self.shape == "window" else "token"
            yield ShapeFinding(where, "unreduced",
                               f"the {what} axis reaches the head unreduced — put one of "
                               f"{', '.join(AXIS_REDUCERS)} before it")
        if kind in _GRAPH_AFTER_POOL and axes.pooled:
            yield ShapeFinding(where, "after_graph_pool",
                               f"{kind!r} follows graph_pool: the node rows are graph rows by "
                               "then and the edges address nodes that no longer exist — move "
                               "every convolution (and edge_dropout) before graph_pool")

    @staticmethod
    def _effect(kind, params, axes: _Axes) -> _Axes:
        effect = _RULES.get(kind, (False, "keep"))[1]
        if effect in ("reduce", "maybe"):
            return axes.with_extra(axes.extra - 1)
        if effect == "return":
            return axes if params.get("return", "last") == "all" else axes.with_extra(
                axes.extra - 1)
        if effect == "attention":
            return axes.with_extra(axes.extra - 1) if params.get("pool") else axes
        if effect == "add":
            return axes.with_extra(1)
        if effect in ("reshape", "custom"):
            dims = params.get("shape" if effect == "reshape" else "out_shape")
            return axes.with_extra(len(dims) - 1) if isinstance(dims, (list, tuple)) else axes
        if effect == "gpool":
            return _Axes(axes.extra, True)
        return axes

    # -- composites

    def _composite(self, kind, params, state: _Walked, where) -> _Walked:
        inner = params.get("inner") or []
        if kind == "sequential":
            return self._chain(inner, state, f"{where}.inner.")
        if kind == "residual":
            return self._residual(inner, state, where)
        return self._parallel(inner, state, where)

    def _residual(self, inner, state: _Walked, where) -> _Walked:
        """x + f(x): walked once — a chain that keeps its axes keeps them at every repeat."""
        out = self._chain(inner, state, f"{where}.inner.")
        if out.axes != state.axes:
            finding = ShapeFinding(where, "residual_axes",
                                   "the inner chain changes the axes of its input, so "
                                   "x + f(x) cannot add them — keep every axis-changing "
                                   "block outside the residual")
            return _Walked(state.axes, out.findings + (finding,), out.embedded)
        return out

    def _parallel(self, inner, state: _Walked, where) -> _Walked:
        """Every branch sees the input; the merge joins their last axis."""
        branches = [self._block(b, _Walked(state.axes, (), state.embedded),
                                f"{where}.inner.blocks[{b.get('id', i)}]")
                    for i, b in enumerate(inner) if isinstance(b, Mapping)]
        if not branches:
            return state
        found = state.findings + tuple(f for b in branches for f in b.findings)
        if len({b.axes for b in branches}) > 1:
            found += (ShapeFinding(where, "parallel_axes",
                                   "the branches leave different axes, and their merge joins "
                                   "the last axis only — give every branch the same axes"),)
        return _Walked(branches[0].axes, found, any(b.embedded for b in branches))
