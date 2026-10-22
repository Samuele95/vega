"""
inspection.focus — the explained sample of a batch: which slice of every captured tensor
belongs to it (inspect-contract §1 R7, §5.4; build plan R-A2, R-A3).

`Focus.of(graph, batch, index)` names sample `index` of the batch the network ran:

- a **row** or **window** sample is row `index` of every per-sample tensor;
- at graph **node level** it is the target node `t = target_index[index]` — the network
  gathers the targets after the head (R-A2), so the head it returns is per sample while every
  block before it is per node; a batch without `target_index` answers every node, and `t` is
  `index` — and its **neighbourhood**: the nodes whose features reach `t` through the
  network's graph layers, found by walking the edges backwards (a message goes from an edge's
  source to its destination) once per graph conv, nearer hops first and, within a hop, in the
  order the edges list them, at most `graph_nodes_max` nodes, `t` first at hop 0;
- at graph **graph level** it is graph `index`: every node whose `graph_index` is `index` (in
  node order, hop 0), and row `index` of the pooled tensors.

`take(t, per)` slices a tensor over the whole batch by what one of its rows is (`per`:
`sample`, `node` or `graph`, the stage graph's `StageNode.per`): a row (window) sample or a
pooled graph gives row `index`; a node-level tensor gives the target node's row; a graph-level
node tensor gives the graph's node rows. `neighbourhood` is the walk alone, for a reader of the
same graph (the attribution's neighbours).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .codec import LIMITS

__all__ = ["GRAPH", "NODE", "SAMPLE", "Focus", "neighbourhood"]

SAMPLE, NODE, GRAPH = "sample", "node", "graph"


def _host(t) -> np.ndarray:
    return np.asarray(t.detach().cpu() if torch.is_tensor(t) else t, dtype=np.int64)


def neighbourhood(edge_index, targets, hops: int, limit: int = LIMITS["graph_nodes_max"]):
    """(ids, hops) of the nodes that reach `targets` in at most `hops` message steps (see the
    module docstring); `edge_index` is (2, E) source → destination."""
    ids = list(dict.fromkeys(int(t) for t in targets))[:int(limit)]
    hop = [0] * len(ids)
    if edge_index is None or not ids:
        return tuple(ids), tuple(hop)
    src, dst = _host(edge_index[0]), _host(edge_index[1])
    seen, frontier = set(ids), np.asarray(ids, dtype=np.int64)
    for h in range(1, int(hops) + 1):
        if len(ids) >= limit or frontier.size == 0:
            break
        reached = src[np.isin(dst, frontier)].tolist()
        fresh = [j for j in dict.fromkeys(reached) if j not in seen][:int(limit) - len(ids)]
        ids += fresh
        hop += [h] * len(fresh)
        seen.update(fresh)
        frontier = np.asarray(fresh, dtype=np.int64)
    return tuple(ids), tuple(hop)


def _targets(batch):
    """The node-level batch's answered nodes (`target_index`), or None: every node."""
    return batch.get("target_index")


def _count(graph, batch) -> int:
    """The samples a batch holds, as the network answers them."""
    inputs = graph.inputs
    if inputs.shape == "graph" and inputs.graph_level:
        return int(batch["num_graphs"])
    if inputs.shape == "graph" and _targets(batch) is not None:
        return int(_targets(batch).shape[0])
    rows = batch.get("x_num")
    if rows is None:
        rows = batch["x_cat"]
    return int(rows.shape[0])


@dataclass(frozen=True)
class Focus:
    """The explained sample (see the module docstring): its index in the batch and, on a graph,
    its level, its target node and the nodes it reads with their hops."""

    shape: str
    index: int
    level: str | None = None
    target: int | None = None
    nodes: tuple = ()
    hops: tuple = ()

    @staticmethod
    def of(graph, batch, index: int = 0) -> "Focus":
        """Sample `index` of `batch`, which `graph`'s network runs."""
        k, n = int(index), _count(graph, batch)
        if not 0 <= k < n:
            raise IndexError(f"sample {k} is not one of the batch's {n}")
        shape = graph.inputs.shape
        if shape != "graph":
            return Focus(shape, k)
        if graph.inputs.graph_level:
            members = torch.nonzero(batch["graph_index"] == k).flatten().tolist()
            nodes = tuple(int(i) for i in members)
            return Focus(shape, k, GRAPH, None, nodes, (0,) * len(nodes))
        target = k if _targets(batch) is None else int(_targets(batch)[k])
        convs = sum(1 for node in graph.top_level() if node.call == "graph")
        ids, hops = neighbourhood(batch["edge_index"], (target,), convs,
                                  graph.codec.limits["graph_nodes_max"])
        return Focus(shape, k, NODE, target, ids, hops)

    def take(self, t, per: str):
        """The explained sample's slice of `t` (a tensor over the whole batch), or None."""
        if t is None:
            return None
        if self.shape != "graph" or per in (SAMPLE, GRAPH):
            return t[self.index]
        if self.level == NODE:
            return t[self.target]
        return t.index_select(0, torch.as_tensor(self.nodes, dtype=torch.long,
                                                 device=t.device))
