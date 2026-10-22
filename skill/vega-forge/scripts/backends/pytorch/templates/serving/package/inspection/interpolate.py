"""
inspection.interpolate — the straight path integrated gradients take: scaled copies of a batch
run through the served network as one batch (inspect-contract §5.4 "Attribution"; D3 §8.3;
build plan I05), for `attribution.Attributor`.

The path runs from the baseline — 0 in shaped space, which is the shaped centre, and the zero
vector at each embedding output — to the batch's own inputs. `Path(model, graphed).run(batch,
alphas)` feeds the network R = len(alphas) copies of the batch as one batch: copy m holds
α_m·x_num and α_m·e at every embedding output (an `embedding` stage's output, a
`feature_tokenizer`'s nominal tokens), each a leaf of the autograd graph; ids, edges and edge
weights are copied as they are — graph edges are held at their weight (1 on an unweighted
graph) and never interpolated. A graph batch is copied as disjoint graphs: node ids,
`target_index` and `graph_index` are offset by the copy, so no message crosses copies and every
copy is the whole graph. `run` with α = (1,) is the batch itself, its embedding outputs leaves:
gradient × input over many samples at once reads it.

An embedding output can only be scaled between two stages, so the copies run by the network's
own stages (its Model port's `anatomy()`: the stage modules and their kinds) and stage calls — the domain's `StageKind` table (build plan R-B1, R-B2): `emb` m(x_cat),
`concat` m(x_num, emb), `tokens` m(x_num, x_cat), `tensor` m(h), `graph` m(h, edge_index,
edge_weight), `gdrop` the edge stream, `gpool` m(h, graph_index, num_graphs) — with the
node-level targets gathered after the head. The network is never modified, hooked or
re-parameterised, so a concurrent prediction on it is untouched. `Copies.check(plain, copy)`
holds the walk to the network's own forward: the copy whose α is 1 must reproduce
`model(batch)` within `PATH_TOLERANCE`·max(1, |head|), else `AttributionError` — an
attribution is the served network's, or none is given.

`Copies.gradients(s)` is ∂s/∂(each leaf) through one `torch.autograd.grad` call — no
parameter's `.grad` is written — averaged over the copies (or read at one copy), per row of one
copy: the numeric channels (`x_num`), each embedding stage (by stage index) and, when asked for,
the edge weights. `Copies.base(stage)` is one copy's unscaled embedding output.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import torch

from .capture import active
from ...domain import StageKind

__all__ = ["EDGES", "NUMERIC", "PATH_TOLERANCE", "AttributionError", "Copies", "Path",
           "midpoints"]

PATH_TOLERANCE = 1e-4        # the walk at α = 1 against the network's forward, × max(1, |head|)
NUMERIC, EDGES = "x_num", "edge_weight"
# stage kind -> the `_Walk` method that calls a stage of that kind (R-B1)
_CALLS = MappingProxyType({
    StageKind.EMB: "embed", StageKind.CONCAT: "concat", StageKind.TOKENS: "tokens",
    StageKind.TENSOR: "tensor", StageKind.GRAPH: "graph", StageKind.GRAPH_DROP: "edges",
    StageKind.GRAPH_POOL: "pool"})


class AttributionError(RuntimeError):
    """An attribution cannot be given: a network in training mode, a capture active on the
    thread, a path that does not reproduce the served network's forward, a pass beyond the
    bounds. The explain route answers it as a defect (500)."""


def midpoints(steps: int, like=None) -> torch.Tensor:
    """α_m = (m + ½)/steps, m = 0…steps − 1: the midpoint rule over [0, 1], in the dtype and on
    the device of `like` (float32 on the CPU without one)."""
    steps = int(steps)
    if steps < 1:
        raise ValueError(f"integrated gradients take at least one step, not {steps}")
    dtype = like.dtype if torch.is_tensor(like) and like.is_floating_point() else torch.float32
    device = like.device if torch.is_tensor(like) else None
    alphas = (torch.arange(steps, dtype=torch.float64) + 0.5) / steps
    return alphas.to(dtype=dtype, device=device)


def _copies(batch: Mapping, alphas, graphed: bool, weights: bool) -> tuple:
    """(the fed copies, the rows of one copy): see the module docstring."""
    x, ids = batch.get(NUMERIC), batch.get("x_cat")
    lead = x if x is not None else ids
    rows, r = int(lead.shape[0]), int(alphas.shape[0])
    fed = {}
    if x is not None:
        a = alphas.to(dtype=x.dtype, device=x.device).view(r, *([1] * x.dim()))
        scaled = (a * x.detach().unsqueeze(0)).reshape(r * rows, *x.shape[1:])
        fed[NUMERIC] = scaled.requires_grad_(True)
    if ids is not None:
        fed["x_cat"] = ids.repeat(r, *([1] * (ids.dim() - 1)))
    if graphed:
        fed.update(_graph_copies(batch, r, rows, weights))
    return fed, rows


def _graph_copies(batch: Mapping, r: int, n: int, weights: bool) -> dict:
    """The graph keys of r disjoint copies of an n-node graph."""
    edge_index = batch["edge_index"]
    e, device = int(edge_index.shape[1]), edge_index.device
    copy = torch.arange(r, device=device)
    out = {"edge_index": edge_index.repeat(1, r) + (copy * n).repeat_interleave(e)}
    w, x = batch.get(EDGES), batch.get(NUMERIC)
    dtype = x.dtype if torch.is_tensor(x) and x.is_floating_point() else torch.float32
    base = w.detach() if torch.is_tensor(w) else torch.ones(e, dtype=dtype, device=device)
    out[EDGES] = base.repeat(r).requires_grad_(bool(weights))
    targets = batch.get("target_index")
    if targets is not None:
        out["target_index"] = targets.repeat(r) + (copy * n).repeat_interleave(
            int(targets.shape[0]))
    groups = batch.get("graph_index")
    if groups is not None:
        g = int(batch["num_graphs"])
        out["graph_index"] = groups.repeat(r) + (copy * g).repeat_interleave(n)
        out["num_graphs"] = r * g
    return out


class _Walk:
    """One run of the network's stages over the fed copies, each embedding output replaced by
    the leaf `scale ⊙ e` (see the module docstring)."""

    def __init__(self, fed: Mapping, scale):
        self.fed, self.scale, self.leaves = fed, scale, {}
        self.h, self.emb = fed.get(NUMERIC), None
        self.edge_index, self.edge_weight = fed.get("edge_index"), fed.get(EDGES)

    def _leaf(self, stage: int, value):
        base = value.detach()
        s = self.scale.to(dtype=base.dtype, device=base.device)
        leaf = (s.view(-1, *([1] * (base.dim() - 1))) * base).requires_grad_(True)
        self.leaves[stage] = (leaf, base)
        return leaf

    def embed(self, stage: int, module) -> None:
        e = module(self.fed.get("x_cat"))
        self.emb = None if e is None else self._leaf(stage, e)

    def concat(self, stage: int, module) -> None:
        self.h = module(self.fed.get(NUMERIC), self.emb)

    def tokens(self, stage: int, module) -> None:
        t = module(self.fed.get(NUMERIC), self.fed.get("x_cat"))
        n = len(getattr(module, "embeddings", None) or ())
        self.h = t if n == 0 else torch.cat([t[:, :-n], self._leaf(stage, t[:, -n:])], dim=1)

    def tensor(self, stage: int, module) -> None:
        self.h = module(self.h)

    def graph(self, stage: int, module) -> None:
        self.h = module(self.h, self.edge_index, self.edge_weight)

    def edges(self, stage: int, module) -> None:
        self.edge_index, self.edge_weight = module(self.edge_index, self.edge_weight)

    def pool(self, stage: int, module) -> None:
        self.h = module(self.h, self.fed.get("graph_index"), self.fed.get("num_graphs"))

    def head(self, anatomy):
        """The head's rows, gathered at the node-level targets as the network gathers them."""
        targets = self.fed.get("target_index") if anatomy.sample.is_node_level else None
        return self.h if targets is None else self.h.index_select(0, targets)


@dataclass(frozen=True, eq=False)
class Copies:
    """R scaled copies of a batch after one run: the fed batch (the scaled `x_num` a leaf),
    the head, the embedding leaves with their unscaled values by stage, R and the rows of one
    copy."""

    fed: Mapping
    head: object
    leaves: Mapping
    copies: int
    rows: int

    @property
    def scoring(self) -> dict:
        """What a target reads beside the head: the scaled `x_num` (the leaf) and the ids."""
        return {k: self.fed[k] for k in (NUMERIC, "x_cat") if self.fed.get(k) is not None}

    def base(self, stage: int):
        """One copy's unscaled output of an embedding stage (every copy holds the same ids)."""
        base = self.leaves[stage][1]
        return base.reshape(self.copies, -1, *base.shape[1:])[0]

    def check(self, plain, copy: int) -> None:
        """Copy `copy` (its α is 1) reproduces `plain`, the network's own head on the batch."""
        n = int(plain.shape[0])
        got = self.head[copy * n:(copy + 1) * n].detach()
        worst = float((got.double() - plain.detach().double()).abs().max()) if n else 0.0
        bound = PATH_TOLERANCE * max(1.0, float(plain.detach().abs().max()) if n else 1.0)
        if not worst <= bound:
            raise AttributionError(f"the attribution path at α = 1 differs from the served "
                                   f"network's forward by {worst:.3g} (bound {bound:.3g}): the "
                                   "stage walk is not the network's")

    def gradients(self, s, copy: int | None = None, retain: bool = False) -> dict:
        """{`x_num` | stage | `edge_weight`: ∂s/∂leaf per row of one copy} — averaged over the
        copies, or read at copy `copy`; None where `s` does not read the leaf."""
        named = [(NUMERIC, self.fed.get(NUMERIC)), (EDGES, self.fed.get(EDGES)),
                 *((stage, leaf) for stage, (leaf, _) in self.leaves.items())]
        named = [(k, t) for k, t in named if torch.is_tensor(t) and t.requires_grad]
        if not named:
            return {}
        grads = torch.autograd.grad(s, [t for _, t in named], allow_unused=True,
                                    retain_graph=retain)
        return {k: None if g is None else self._per_copy(g, copy)
                for (k, _), g in zip(named, grads)}

    def _per_copy(self, g, copy):
        split = g.reshape(self.copies, -1, *g.shape[1:])
        return split.mean(dim=0) if copy is None else split[copy]


class Path:
    """The straight path from the baseline through one served network (see the module
    docstring); `graphed` says the batches are graphs."""

    def __init__(self, model, graphed: bool):
        self.model, self.graphed = model, bool(graphed)        # the network, as its Model port

    def run(self, batch: Mapping, alphas, weights: bool = False) -> Copies:
        """R = len(alphas) scaled copies of `batch` run as one; `weights` makes the copied edge
        weights a leaf too (their relevance)."""
        anatomy = self.model.anatomy()
        if anatomy.training:
            raise AttributionError("the inspector reads a network in eval mode, and this one "
                                   "is training; it never switches a served model")
        if active() is not None:
            raise AttributionError("integrated gradients run outside an explanation's "
                                   "capture: the copies must not reach its hooks")
        calls = [_CALLS.get(StageKind(kind)) for kind in anatomy.kinds]
        if None in calls:
            raise AttributionError(f"stage {calls.index(None)} is of a kind the attribution "
                                   "path has no call for")
        fed, rows = _copies(batch, alphas, self.graphed, weights)
        with torch.enable_grad():
            walk = _Walk(fed, alphas.repeat_interleave(rows))
            for stage, (call, module) in enumerate(zip(calls, anatomy.stages)):
                getattr(walk, call)(stage, module)
            head = walk.head(anatomy)
        return Copies(MappingProxyType(fed), head, MappingProxyType(walk.leaves),
                      int(alphas.shape[0]), rows)
