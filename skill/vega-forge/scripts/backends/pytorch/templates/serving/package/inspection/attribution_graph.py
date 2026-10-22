"""
inspection.attribution_graph — a graph sample's attribution: the nodes the explained output
reads, by hop, and the relevance of the edges between them (inspect-contract §5.4, §5.5;
D3 §8.3; build plan I05), for `attribution.Attributor`.

`Around(layout)` reads the node attributions of a graph batch (`attribution_parts.Shares`, one
row per node, ordered over `inputs.order`):

- `reach(edge_index, targets, hops)` — every node whose features reach `targets` in at most
  `hops` message steps (a message goes from an edge's source to its destination), with its
  distance, uncapped (`focus.neighbourhood`'s walk); the explained output of a node-level
  network reads the walk from its target over the graph convs, a graph-level one the explained
  graph's nodes (hop 0);
- `kept(reach, totals)` — the neighbourhood reported: nearer hops first and, within a hop, the
  larger |total| first (the walk's order on a tie), at most `graph_nodes_max` nodes;
- `payload(shares, focus, edges)` — the explanation's graph payload: `self` (node level),
  `neighbours` (each with its `TOP_FEATURES` largest |a|), `hops` (the per-hop totals over every
  node read), `outside` (`total` − the kept nodes' totals), `edges` and `total`;
- `edges` — the relevance ∂s/∂w_e of every edge between two kept nodes (at node level, only
  those whose message reaches the output within the convs: a destination nearer than the last
  conv's hop), `hop` the destination's hop + 1 (the message step, counted back from the output,
  at which the edge first carries; 1 for every edge at graph level), ordered by hop then edge
  order, at the explained input for both methods; null when `s` does not read the edge
  weights;
- `node_entry(shares, target, edge_index)` — a node-level sample's table entry `{self, hops}`.
"""
from __future__ import annotations

import torch

from .attribution_parts import TOP_FEATURES, host64
from .codec import num, nums
from .focus import neighbourhood

__all__ = ["Around", "edges_back"]


def edges_back(grad, keep):
    """The edge gradients of one extracted graph put back on the batch's edges (`keep` the
    batch edges it holds); None stays None."""
    if grad is None:
        return None
    out = grad.new_zeros(int(keep.shape[0]))
    out[keep] = grad
    return out


class Around:
    """The neighbourhood side of a graph attribution (see the module docstring)."""

    def __init__(self, layout):
        self.layout = layout

    @property
    def cap(self) -> int:
        return int(self.layout.limits["graph_nodes_max"])

    @staticmethod
    def reach(edge_index, targets, hops: int) -> tuple:
        """(ids, hops) of every node within `hops` message steps of `targets`, uncapped."""
        targets = tuple(int(t) for t in targets)
        bound = (int(edge_index.shape[1]) if edge_index is not None else 0) + len(targets) + 1
        return neighbourhood(edge_index, targets, int(hops), bound)

    def kept(self, reach, totals=None) -> tuple:
        """(ids, hops) reported: nearer hops, then larger |total|, at most `graph_nodes_max`."""
        ids, rings = reach

        def key(p):
            weight = -abs(float(totals[ids[p]])) if totals is not None else 0.0
            return rings[p], weight, p
        order = sorted(range(len(ids)), key=key)[:self.cap]
        return tuple(ids[p] for p in order), tuple(rings[p] for p in order)

    def _read(self, focus, edge_index) -> tuple:
        """(ids, hops) of every node the explained output reads."""
        if focus.level == "node":
            return self.reach(edge_index, (focus.target,), self.layout.convs)
        return tuple(int(j) for j in focus.nodes), (0,) * len(focus.nodes)

    def payload(self, sh, focus, edges) -> dict:
        """The explanation's graph payload; `edges` = (edge_index, ∂s/∂w or None)."""
        per = self.layout.ordered(sh.attr, sh.nominal)             # (N, F)
        totals = per.sum(dim=-1)
        edge_index, wgrad = edges
        node = focus.level == "node"
        read = self._read(focus, edge_index)
        ids, rings = self.kept(read, totals)
        total = totals.sum()
        kept = totals.index_select(0, torch.as_tensor(ids, dtype=torch.long)).sum()
        return {"shape": "graph", "level": focus.level,
                "target_node": int(focus.target) if node else None,
                "features": {"names": list(self.layout.order)},
                "self": nums(per[int(focus.target)]) if node else None,
                "neighbours": [self._neighbour(per[j], j, h) for j, h in zip(ids, rings)],
                "hops": self._hops(read, totals), "outside": num(total - kept),
                "edges": self._edges(edge_index, wgrad, dict(zip(ids, rings)), node),
                "total": num(total)}

    @staticmethod
    def _neighbour(per_node, j: int, hop: int) -> dict:
        order = sorted(range(int(per_node.shape[0])),
                       key=lambda f: (-abs(float(per_node[f])), f))[:TOP_FEATURES]
        return {"node": int(j), "hop": int(hop), "total": num(per_node.sum()),
                "top": [[int(f), num(per_node[f])] for f in order]}

    @staticmethod
    def _hops(read, totals) -> dict:
        sums = {}
        for j, h in zip(*read):
            sums[int(h)] = sums.get(int(h), 0.0) + float(totals[int(j)])
        return {str(h): num(v) for h, v in sorted(sums.items())}

    def _edges(self, edge_index, wgrad, rings, node: bool):
        if wgrad is None or edge_index is None:
            return None
        src, dst = edge_index[0].tolist(), edge_index[1].tolist()
        g, convs = host64(wgrad), self.layout.convs
        rows = sorted((rings[i] + 1, e) for e, (j, i) in enumerate(zip(src, dst))
                      if j in rings and i in rings and (not node or rings[i] < convs))
        return [{"src": int(src[e]), "dst": int(dst[e]), "hop": int(h),
                 "relevance": num(g[e])} for h, e in rows]

    def node_entry(self, sh, target: int, edge_index) -> dict:
        """A node-level sample's table entry: its target node's own attributions and the
        per-hop totals of the nodes its output reads."""
        per = self.layout.ordered(sh.attr, sh.nominal)
        read = self.reach(edge_index, (int(target),), self.layout.convs)
        return {"self": nums(per[int(target)]), "hops": self._hops(read, per.sum(dim=-1))}
