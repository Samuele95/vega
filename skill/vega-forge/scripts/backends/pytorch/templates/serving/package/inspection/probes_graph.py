"""
inspection.probes_graph — the probes of the graph blocks (inspect-contract §3.1; build plan
R-A3, R-B2, R-B3): `gcn_conv`, `sage_conv`, `gat_conv`, `gin_conv`, `graph_pool`,
`edge_dropout`.

A conv is read at the explained sample's **targets** T — at node level the explained node, at
graph level every node of the explained graph — through its pure `coefficients(...)` (R-B3) and
the published weights. Every edge j → i with i ∈ T carries one **message** m, in the space the
conv sums in:

| type | m | the output at i |
|---|---|---|
| gcn | coef · W h_j | Σ m + b |
| sage | W_n (coef · h_j); under `max` only the channels j won; the root term W_r h_i apart | Σ m + root + b, L2-normalised when `normalize` |
| gat | α·w · W^h h_j per head (coef = α·w per head), heads concatenated or averaged | Σ m + b |
| gin | coef · h_j, the self term (1 + ε) h_i among them | mlp(Σ m) |

`messages` lists, per message, `src`, `dst`, `hop` (0 for a self term or the root, else 1),
`root`, `coef`, the message's `norm` and its `relevance` ⟨J_iᵀ ∂s/∂out_i, m⟩ — the first-order
share of the target the message carries (J_i the map from the sum to the output: the identity,
the normalisation, the MLP) — nearer hops first, then by |relevance|, at most `graph_nodes_max`.
The check recomputes the output at every target from all its messages. `exact` carries the
GAT's `alpha` (messages × heads) with its `edge_index`.

- **graph_pool**: `node_share`, each node's share of the pooled row — 1/N (mean), the fraction
  of channels it won (max), ‖h_i‖ / Σ‖h‖ (sum); the check is the readout over the graph's nodes.
- **edge_dropout**: the check is edges in = edges out (inactive at inference): the count of
  differing edge ids, else the largest weight difference.
"""
from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .codec import num, nums
from .probes import Probe, f64, gap, norm_last, sample_io

__all__ = ["EDGE_DROPOUT", "GAT", "GCN", "GIN", "GRAPH_POOL", "SAGE"]


class _Terms(NamedTuple):
    """One conv's messages into the targets (see the module docstring)."""

    src: Any                # (M,) sources
    dst: Any                # (M,) destinations, each a target
    coef: Any               # (M,) or (M, heads)
    msg: Any                # (M, W) the messages, float64
    root: Any = None        # (T, W) the root term per target (sage), or None
    bias: Any = None        # (W,) or None
    post: Any = None        # the map from the sum to the output, None for the identity


def _kept(coefficients, targets) -> tuple:
    src, dst, coef = coefficients
    keep = torch.isin(dst, targets)
    return src[keep], dst[keep], f64(coef[keep])


def _gcn(module, inputs, edges, targets) -> _Terms:
    h, _ = inputs
    src, dst, coef = _kept(module.coefficients(edges[0], h.shape[0], edges[1]), targets)
    msg = coef.unsqueeze(-1) * F.linear(h[src], f64(module.lin.weight))
    return _Terms(src, dst, coef, msg, bias=f64(module.bias))


def _winners(x, dst):
    """1 where message e holds the max of its destination's channel (the first such), else 0."""
    mask = torch.zeros_like(x)
    channels = torch.arange(x.shape[1])
    for i in dst.unique().tolist():
        rows = torch.nonzero(dst == i).flatten()
        mask[rows[x[rows].argmax(dim=0)], channels] = 1.0
    return mask


def _normalised(t):
    return F.normalize(t, p=2.0, dim=-1)


def _sage(module, inputs, edges, targets) -> _Terms:
    h, _ = inputs
    src, dst, coef = _kept(module.coefficients(edges[0], h.shape[0], edges[1]), targets)
    x = coef.unsqueeze(-1) * h[src]
    if str(module.aggregator) == "max":
        x = x * _winners(x, dst)
    root = None if module.lin_root is None else F.linear(h[targets], f64(module.lin_root.weight))
    return _Terms(src, dst, coef, F.linear(x, f64(module.lin_neigh.weight)), root,
                  f64(module.lin_neigh.bias), _normalised if module.normalize else None)


def _gat(module, inputs, edges, targets) -> _Terms:
    h, raw = inputs
    src, dst, alpha = _kept(module.coefficients(raw, edges[0], edges[1]), targets)
    hx = F.linear(h[src], f64(module.lin.weight)).unflatten(-1, (int(module.heads), -1))
    m = alpha.unsqueeze(-1) * hx
    msg = m.flatten(1) if module.concat else m.mean(dim=1)
    return _Terms(src, dst, alpha, msg, bias=f64(module.bias))


def _mlp(mlp):
    def run(t):
        for layer in mlp:
            t = F.linear(t, f64(layer.weight), f64(layer.bias)) if isinstance(layer, nn.Linear) \
                else layer(t)
        return t
    return run


def _gin(module, inputs, edges, targets) -> _Terms:
    h, _ = inputs
    src, dst, coef = _kept(module.coefficients(edges[0], h.shape[0], edges[1]), targets)
    return _Terms(src, dst, coef, coef.unsqueeze(-1) * h[src], post=_mlp(module.mlp))


def _pull(post, pre, g):
    """Jᵀ g for the map `post` at `pre` — a vector-Jacobian product over float64 copies, which
    touches no parameter."""
    with torch.enable_grad():
        at = pre.detach().requires_grad_(True)
        (back,) = torch.autograd.grad(post(at), at, grad_outputs=g)
    return back.detach()


class _Read(NamedTuple):
    terms: _Terms
    targets: Any            # (T,)
    pre: Any                # (T, W) the sum at each target (root and bias included)
    got: Any                # (T, out) the recomputed output
    out: Any                # (T, out) the network's
    relevance: Any          # (M,) or None
    root_relevance: Any     # (T,) or None


class _Conv(Probe):
    """A graph conv, one instance per type, with its message rule."""

    KEYS = ("messages",)
    CHECKS = True

    def __init__(self, terms, exact: tuple = ()):
        self._terms, self.EXACT = terms, tuple(exact)

    def _read(self, cap, node) -> _Read:
        return cap.memo((node.id, "messages"), lambda: self._compute(cap, node))

    def _compute(self, cap, node) -> _Read:
        raw, focus = cap.value_in(node.id), cap.focus
        ids, weights = (tuple(cap.extras(node.id)) + (None, None))[:2]
        edges = (ids, None if weights is None else weights.detach())
        targets = torch.as_tensor(focus.nodes if focus.level == "graph" else (focus.target,),
                                  dtype=torch.long, device=raw.device)
        with cap.suspended(), torch.no_grad():
            t = self._terms(cap.module(node.id), (f64(raw), raw.detach()), edges, targets)
        pos = torch.full((raw.shape[0],), -1, dtype=torch.long, device=raw.device)
        pos[targets] = torch.arange(targets.shape[0], device=raw.device)
        pos = pos[t.dst]
        pre = t.msg.new_zeros((targets.shape[0], t.msg.shape[1])).index_add(0, pos, t.msg)
        pre = pre + (t.root if t.root is not None else 0) + (t.bias if t.bias is not None else 0)
        with torch.no_grad():
            got = t.post(pre) if t.post is not None else pre
        g = cap.grad(node.id)
        g = None if g is None else (f64(g)[targets] if t.post is None else
                                    _pull(t.post, pre, f64(g)[targets]))
        return _Read(t, targets, pre, got, f64(cap.value(node.id))[targets],
                     None if g is None else (g[pos] * t.msg).sum(dim=-1),
                     None if g is None or t.root is None else (g * t.root).sum(dim=-1))

    def probe(self, cap, node):
        read, limit = self._read(cap, node), cap.graph.codec.limits["graph_nodes_max"]
        rows = _rows(read)
        rank = np.zeros(rows["src"].shape[0]) if rows["relevance"] is None else \
            -np.abs(rows["relevance"])
        order = np.lexsort((rank, rows["hop"]))[:limit]
        return {"messages": [_record(rows, int(e)) for e in order]}

    def recompute(self, cap, node):
        read = self._read(cap, node)
        return gap(read.got, read.out)

    def exact(self, cap, node):
        if "alpha" not in self.EXACT:
            return {}
        t = self._read(cap, node).terms
        return {"alpha": (t.coef, ("edge", "head")),
                "edge_index": (torch.stack([t.src, t.dst]), ("end", "edge"))}


class _GINConv(_Conv):
    """A GIN conv also shows the sum its MLP reads at each target (`pre_mlp`)."""

    KEYS = ("messages", "pre_mlp")

    def probe(self, cap, node):
        pre = self._read(cap, node).pre
        return {**super().probe(cap, node),
                "pre_mlp": cap.reducer.reduce(pre[0] if cap.focus.level != "graph" else pre)}


def _host(t) -> np.ndarray:
    return t.detach().cpu().numpy()


def _rows(read: _Read) -> dict:
    """The messages, then the root terms, as columns (numpy). A target no edge reaches (no
    self loop, no in-edge) has no message: the columns are then empty, the root apart."""
    t = read.terms
    src, dst = _host(t.src), _host(t.dst)
    coef = _host(t.coef).reshape(src.shape[0], t.coef.shape[1] if t.coef.dim() > 1 else 1)
    norm = _host(norm_last(t.msg))
    rel = None if read.relevance is None else _host(read.relevance)
    root = np.zeros(src.shape[0], dtype=bool)
    if t.root is not None:
        tg = _host(read.targets)
        src, dst = np.concatenate([src, tg]), np.concatenate([dst, tg])
        coef = np.concatenate([coef, np.ones((tg.shape[0], coef.shape[1]))])
        norm = np.concatenate([norm, _host(norm_last(t.root))])
        rel = None if rel is None else np.concatenate([rel, _host(read.root_relevance)])
        root = np.concatenate([root, np.ones(tg.shape[0], dtype=bool)])
    return {"src": src, "dst": dst, "hop": (src != dst).astype(np.int64), "root": root,
            "coef": coef, "per_head": t.coef.dim() > 1, "norm": norm, "relevance": rel}


def _record(rows: dict, e: int) -> dict:
    coef = rows["coef"][e]
    return {"src": int(rows["src"][e]), "dst": int(rows["dst"][e]), "hop": int(rows["hop"][e]),
            "root": bool(rows["root"][e]),
            "coef": nums(coef) if rows["per_head"] else num(coef[0]),
            "norm": num(rows["norm"][e]),
            "relevance": None if rows["relevance"] is None else num(rows["relevance"][e])}


class _GraphPool(Probe):
    KEYS = ("node_share",)
    CHECKS = True

    def probe(self, cap, node):
        x, _ = sample_io(cap, node)
        readout, n = str(cap.module(node.id).readout), int(x.shape[0])
        if n == 0 or readout == "mean":
            share = x.new_full((n,), 1.0 / max(n, 1))
        elif readout == "max":
            share = torch.bincount(x.argmax(dim=0), minlength=n).to(x.dtype) / x.shape[1]
        else:
            size = norm_last(x)
            total = float(size.sum())
            share = size / total if total > 0 else x.new_zeros(n)
        limit = cap.graph.codec.limits["graph_nodes_max"]
        return {"node_share": {"nodes": list(cap.focus.nodes[:limit]),
                               "share": nums(share[:limit])}}

    def recompute(self, cap, node):
        x, y = sample_io(cap, node)
        readout = str(cap.module(node.id).readout)
        if x.shape[0] == 0:
            return gap(torch.zeros_like(y), y)
        read = x.mean(dim=0) if readout == "mean" else x.max(dim=0).values \
            if readout == "max" else x.sum(dim=0)
        return gap(read, y)


class _EdgeDropout(Probe):
    CHECKS = True

    def recompute(self, cap, node):
        rec = cap.seen(node.id)
        ids_in, ids_out = rec.input, rec.output
        w_in = rec.extras[0] if rec.extras else None
        w_out = rec.state[0] if rec.state else None
        if tuple(ids_in.shape) != tuple(ids_out.shape):
            return float(abs(int(ids_in.shape[-1]) - int(ids_out.shape[-1])))
        differing = int((ids_in != ids_out).sum())
        if differing or (w_in is None) != (w_out is None):
            return float(differing or int(ids_in.shape[-1]))
        return 0.0 if w_in is None else gap(w_in, w_out)


GCN = _Conv(_gcn)
SAGE = _Conv(_sage)
GAT = _Conv(_gat, ("alpha", "edge_index"))
GIN = _GINConv(_gin)
GRAPH_POOL = _GraphPool()
EDGE_DROPOUT = _EdgeDropout()
