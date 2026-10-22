"""
{{PACKAGE}}.{{MODULE_INGESTION}}.graphs — the graph half of the {{PORT_DATASOURCE}} adapter
(support; emitted when `data_contract.sample.shape` is `graph`).

Emitted by /vega-forge from `backends/pytorch/templates/graphs.py`. Fix defects in the template
and re-emit; never in this copy (code-contract §4).

Which nodes and edges a pass reads is decided once, in `domain.py` (`GraphSpec.view`, the
inference-graph rule R-A3 serving applies too): query nodes receive messages from reference
nodes only, never from another query. This file resolves and tensorises:

  * `EdgeTable` maps each end of the shaper's edges part (raw-file positions) to its node
    through `fitted.json.__split_raw__` — never `__split__`, which a `dedupe` shifts — as global
    ids over `[train | valid | test]`; an end no part lists is refused by position and part;
  * `GraphParts` builds a view's `GraphPart`: its nodes' features (the test part's only in the
    sealed view, or its feature columns alone through the transductive door), its targets'
    labels, their raw positions (graph ids at graph level) as row ids, the exposure record;
  * `NeighbourSampler` draws `sampled` training batches with replacement, seeded `seed + epoch`,
    relabelled seeds first; `GraphCollate` cuts graph-level batches by whole graphs.
"""
from __future__ import annotations

from typing import Any, Iterator, Mapping, NamedTuple, Sequence

import numpy as np
import torch

from ..domain import (GraphEdges, GraphExposure, GraphLevel, GraphPart, GraphProtocol,
                      GraphSpec, NetSpec, Part, Task)
from .frames import PartFrames, UnsealedFrames, feature_tensors, label_tensor

_PARTS = (Part.TRAIN, Part.VALID, Part.TEST)


def _part_index(ids, sizes: tuple):
    return np.searchsorted(np.cumsum(sizes), ids, side="right")


def _weights(frame):
    """The edges' weights, float32, or None when the part is unweighted (null throughout)."""
    if "weight" not in frame.columns:
        return None
    weight = frame["weight"].to_numpy(dtype=np.float64, na_value=np.nan)
    missing = np.isnan(weight)
    if missing.all():
        return None
    if missing.any():
        raise ValueError(f"edge {int(np.argmax(missing))} has no weight while others do; an edges "
                         "part is weighted throughout or not at all")
    return weight.astype(np.float32)


class _RawIndex:
    """Raw-file positions -> global node ids over `[train | valid | test]`."""

    def __init__(self, raw_by_part: Mapping):
        raw = [np.asarray(raw_by_part[p], dtype=np.int64) for p in _PARTS]
        self.sizes = tuple(len(r) for r in raw)
        positions = np.concatenate(raw)
        self._order = np.argsort(positions, kind="stable")
        self._sorted = positions[self._order]
        twice = np.flatnonzero(self._sorted[1:] == self._sorted[:-1])
        if len(twice):
            raise ValueError(f"raw position {int(self._sorted[twice[0]])} is listed by two parts "
                             "in fitted.json __split_raw__; a row lies in one part")

    def ids(self, frame, end: str):
        """The global node id of each edge's `end` ("src" or "dst")."""
        column, named = f"{end}_pos", f"{end}_part"
        if column not in frame.columns:
            raise ValueError(f"the edges part has no {column} column (shape-contract §2)")
        pos = frame[column].to_numpy(dtype=np.int64)
        parts = frame[named].astype(str).to_numpy() if named in frame.columns else None
        at = np.minimum(np.searchsorted(self._sorted, pos), max(len(self._sorted) - 1, 0))
        found = (self._sorted[at] == pos) if len(self._sorted) else np.zeros(len(pos), bool)
        if not found.all():
            k = int(np.argmin(found))
            where = f" (part {parts[k]})" if parts is not None else ""
            raise ValueError(f"edge {k}: {column} {int(pos[k])}{where} is no row of any part — "
                             "fitted.json __split_raw__ does not list it; the edges part and the "
                             "split disagree")
        ids = self._order[at]
        if parts is not None:
            got = np.asarray([p.value for p in _PARTS])[_part_index(ids, self.sizes)]
            bad = np.flatnonzero(got != parts)
            if len(bad):
                k = int(bad[0])
                raise ValueError(f"edge {k}: {column} {int(pos[k])} is a row of the {got[k]} "
                                 f"part, but the edges part names {parts[k]}")
        return ids


class EdgeTable(NamedTuple):
    """The edges part as `GraphEdges` over global node ids, and the parts' sizes."""

    edges: GraphEdges
    sizes: tuple

    @staticmethod
    def load(spec: NetSpec, frames: PartFrames | UnsealedFrames) -> "EdgeTable":
        raw = {p: frames.raw_positions(spec, p) for p in _PARTS}
        return EdgeTable.resolve(frames.edges(spec), raw)

    @staticmethod
    def resolve(frame, raw_by_part: Mapping) -> "EdgeTable":
        index = _RawIndex(raw_by_part)
        return EdgeTable(GraphEdges(index.ids(frame, "src"), index.ids(frame, "dst"),
                                    _weights(frame)), index.sizes)


class _Slice(NamedTuple):
    """One view being tensorised: the spec, the view, the parts opened for it, the sizes."""

    spec: Any
    view: Any
    opened: Mapping
    sizes: tuple
    exposure: GraphExposure


def _offset(part: Part, sizes: tuple) -> int:
    return int(sum(sizes[:_PARTS.index(part)]))


def _features(s: _Slice) -> tuple:
    """`(x_num, x_cat)` of the view's nodes, in `node_ids` order."""
    parts = s.view.feature_parts
    xs = [feature_tensors(s.opened[p].frame, s.spec.feature_contract, p.value) for p in parts]
    x_num = torch.cat([x[0] for x in xs])
    x_cat = None if xs[0][1] is None else torch.cat([x[1] for x in xs])
    local = np.asarray(s.view.node_ids, dtype=np.int64) - _offset(parts[0], s.sizes)
    if len(local) and (local.min() < 0 or local.max() >= len(x_num)):
        raise ValueError("the view names nodes outside the parts whose features it reads")
    idx = torch.from_numpy(local)
    return x_num[idx], (None if x_cat is None else x_cat[idx])


def _edges(s: _Slice) -> tuple:
    weight = s.view.edge_weight
    return (torch.from_numpy(np.ascontiguousarray(s.view.edge_index, dtype=np.int64)),
            None if weight is None else torch.from_numpy(np.asarray(weight, dtype=np.float32)))


class GraphParts:
    """A part of a graph sample as the `GraphPart` of its view, and the exposure record."""

    def __init__(self, frames: PartFrames | UnsealedFrames):
        self._frames = frames

    def part(self, spec: NetSpec, part) -> GraphPart:
        spec, part = NetSpec.require(spec), Part(part)
        gspec = GraphSpec.from_spec(spec)        # refuses a transductive spec with no exposure
        if gspec.level is GraphLevel.GRAPH:
            self._check_split(spec, part, gspec.graph_id)
        table = EdgeTable.load(spec, self._frames)
        view = gspec.view(table.sizes, table.edges, part)
        s = _Slice(spec, view, self._open(spec, view), table.sizes,
                   gspec.exposure(table.sizes, table.edges))
        return self._graphs(s, gspec) if gspec.level is GraphLevel.GRAPH else self._nodes(s)

    def exposure(self, spec: NetSpec) -> GraphExposure:
        """What the training graph saw of the test part: nothing for an inductive or a
        graph-level spec (no file is read), counted from the training view otherwise."""
        spec = NetSpec.require(spec)
        gspec = GraphSpec.from_spec(spec)
        if gspec.protocol is GraphProtocol.INDUCTIVE or gspec.level is GraphLevel.GRAPH:
            return GraphExposure(gspec.protocol)
        table = EdgeTable.load(spec, self._frames)
        return gspec.exposure(table.sizes, table.edges)

    def _open(self, spec: NetSpec, view) -> dict:
        """The parts the view reads. The test part through the sealed view's unsealed reader —
        or, in a training/valid view (transductive only), its feature columns alone."""
        opened = {}
        for p in dict.fromkeys(view.feature_parts + view.label_parts):
            door = p is Part.TEST and view.part is not Part.TEST
            opened[p] = (self._frames.open_features_only(spec, p) if door
                         else self._frames.open(spec, p))
        return opened

    def _nodes(self, s: _Slice) -> GraphPart:
        x_num, x_cat = _features(s)
        edge_index, edge_weight = _edges(s)
        rows = s.opened[s.view.part]
        local = s.view.node_ids[s.view.target_index] - _offset(s.view.part, s.sizes)
        y = label_tensor(rows.frame, Task.from_spec(s.spec))
        return GraphPart(x_num=x_num, x_cat=x_cat, edge_index=edge_index, edge_weight=edge_weight,
                         target_index=torch.from_numpy(np.asarray(s.view.target_index)),
                         y=None if y is None else y[torch.from_numpy(local)],
                         row_ids=np.asarray(rows.raw_pos)[local],
                         columns=s.spec.feature_contract.columns, exposure=s.exposure)

    def _graphs(self, s: _Slice, gspec: GraphSpec) -> GraphPart:
        frame = s.opened[s.view.part].frame
        y = label_tensor(frame, Task.from_spec(s.spec))
        groups = GraphSpec.graphs(frame[gspec.graph_id].to_numpy(), s.view.edge_index,
                                  None if y is None else y.numpy())
        x_num, x_cat = _features(s)
        edge_index, edge_weight = _edges(s)
        return GraphPart(x_num=x_num, x_cat=x_cat, edge_index=edge_index, edge_weight=edge_weight,
                         target_index=None,
                         y=None if y is None else y[torch.from_numpy(groups.first_rows)],
                         row_ids=np.asarray(groups.values),
                         graph_index=torch.from_numpy(groups.graph_index),
                         num_graphs=groups.num_graphs,
                         columns=s.spec.feature_contract.columns, exposure=s.exposure)

    def _check_split(self, spec: NetSpec, part: Part, graph_id: str) -> None:
        """A graph lies in one part: the graph ids of the parts this view may read (the test
        part only in the sealed view) are compared before any edge is resolved."""
        ids = {}
        for p in (_PARTS if part is Part.TEST else _PARTS[:2]):
            frame = self._frames.open(spec, p, columns=[graph_id]).frame
            ids[p] = frame[graph_id].to_numpy()
        GraphSpec.check_split(ids)


def _ranges(starts, sizes):
    """`cat(arange(a, a + n) for a, n in zip(starts, sizes))`, vectorised."""
    offsets = torch.cumsum(sizes, 0) - sizes
    return (torch.repeat_interleave(starts - offsets, sizes)
            + torch.arange(int(sizes.sum()), dtype=torch.long))


def _host(values):
    return values.detach().to("cpu", torch.long)


def _gathered(part: GraphPart, nodes, edges: Mapping) -> dict:
    """The batch dict of the nodes `nodes` of `part` (host indices) with the given relabelled
    edges (`index`, and `rows` of the part's edges for their weights)."""
    dev = part.x_num.device
    batch = {"x_num": part.x_num[nodes.to(dev)], "edge_index": edges["index"].to(dev)}
    if part.x_cat is not None:
        batch["x_cat"] = part.x_cat[nodes.to(dev)]
    if part.edge_weight is not None:
        batch["edge_weight"] = part.edge_weight[edges["rows"].to(dev)]
    return batch


class NeighbourSampler:
    """Seeded k-hop neighbour sampling over a node-level part, with replacement (D1 §9.4).

    A batch's seeds are `batch_size` targets of a seeded permutation. At hop k every node of
    the frontier F_k draws `fanouts[k]` of its in-edges with replacement (none when it has none);
    F_{k+1} = F_k ∪ the drawn sources. The union subgraph is relabelled seeds first, so the
    batch's `target_index` is `0 … S−1` and its `y` the seeds' labels. Scoring never samples.
    """

    def __init__(self, part: GraphPart, fanouts: Sequence[int], batch_size: int):
        if part.level is not GraphLevel.NODE:
            raise ValueError("neighbour sampling reads a node-level graph part")
        self._fanouts = tuple(int(k) for k in fanouts)
        if not self._fanouts or min(self._fanouts) < 1 or int(batch_size) < 1:
            raise ValueError("the sampler needs one positive fan-out per hop and a positive "
                             "batch size")
        self._part, self._bs = part, int(batch_size)
        src, dst = _host(part.edge_index)
        self._n = int(part.x_num.shape[0])
        order = torch.argsort(dst, stable=True)
        self._nbr, self._rows = src[order], order
        self._indptr = torch.cat([torch.zeros(1, dtype=torch.long),
                                  torch.cumsum(torch.bincount(dst, minlength=self._n), 0)])
        self._targets = _host(part.target_index)

    @staticmethod
    def for_spec(spec: NetSpec, part: GraphPart) -> "NeighbourSampler":
        """`sample.neighbours` fan-outs, `training.batch_size` seeds per batch."""
        gspec = GraphSpec.from_spec(NetSpec.require(spec))
        if not gspec.neighbours:
            raise ValueError("sample.neighbours is not set: the spec does not sample (batching "
                             f"{gspec.batching.value!r})")
        plan = NetSpec.require(spec).training
        return NeighbourSampler(part, gspec.neighbours,
                                plan.batch_size.for_part(len(part)) if plan.batch_size_stated
                                else len(part))

    def __len__(self):
        return -(-len(self._targets) // self._bs)

    def epoch(self, seed: int) -> Iterator[dict]:
        """One epoch's batches, from a CPU generator seeded `seed` (`spec.seed + epoch`)."""
        generator = torch.Generator().manual_seed(int(seed))
        order = torch.randperm(len(self._targets), generator=generator)
        for start in range(0, len(order), self._bs):
            yield self.sample(order[start:start + self._bs], generator)

    def sample(self, picks, generator) -> dict:
        """The batch whose seeds are the targets at positions `picks` of `target_index`."""
        picks = torch.as_tensor(picks, dtype=torch.long)
        seeds = self._targets[picks]
        frontier, drawn = seeds, []
        for fanout in self._fanouts:
            hop = self._hop(frontier, fanout, generator)
            drawn.append(hop)
            frontier = torch.unique(torch.cat([frontier, hop[0]]))
        src, dst, rows = (torch.cat([h[i] for h in drawn]) for i in range(3))
        others = torch.unique(torch.cat([src, dst]))
        nodes = torch.cat([seeds, others[~torch.isin(others, seeds)]])
        where = torch.full((self._n,), -1, dtype=torch.long)
        where[nodes] = torch.arange(len(nodes))
        batch = _gathered(self._part, nodes,
                          {"index": torch.stack([where[src], where[dst]]), "rows": rows})
        batch["target_index"] = torch.arange(len(seeds), device=self._part.x_num.device)
        if self._part.y is not None:
            batch["y"] = self._part.y[picks.to(self._part.y.device)]
        return batch

    def _hop(self, frontier, fanout: int, generator) -> tuple:
        """(sources, destinations, edge rows) of `fanout` draws per frontier node."""
        start = self._indptr[frontier]
        degree = self._indptr[frontier + 1] - start
        has = degree > 0
        frontier, start, degree = frontier[has], start[has], degree[has]
        draw = torch.rand((len(frontier), fanout), generator=generator, dtype=torch.float64)
        offs = torch.minimum((draw * degree[:, None]).floor().long(), (degree - 1)[:, None])
        at = (start[:, None] + offs).reshape(-1)
        return self._nbr[at], frontier.repeat_interleave(fanout), self._rows[at]


class GraphCollate:
    """Graph-level mini-batches (D1 §9.4): whole graphs; nodes concatenated in batch order,
    edge indices offset by the node count so far, `graph_index` the graph's slot in the batch,
    `num_graphs` the batch's graph count. Training shuffles by a seeded permutation; scoring
    uses the same collate in stored order."""

    def __init__(self, part: GraphPart, batch_size: int):
        if part.level is not GraphLevel.GRAPH:
            raise ValueError("graph collation reads a graph-level part")
        if int(batch_size) < 1:
            raise ValueError("graphs per batch must be positive")
        self._part, self._bs, self._g = part, int(batch_size), int(part.num_graphs)
        graph = _host(part.graph_index)
        self._nodes = torch.argsort(graph, stable=True)
        counts = torch.bincount(graph, minlength=self._g)
        self._node_ptr = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(counts, 0)])
        self._local = torch.empty_like(graph)
        self._local[self._nodes] = (torch.arange(len(graph))
                                    - self._node_ptr[graph[self._nodes]])
        self._index = _host(part.edge_index)
        of_edge = graph[self._index[0]]
        self._edges = torch.argsort(of_edge, stable=True)
        self._edge_ptr = torch.cat([torch.zeros(1, dtype=torch.long),
                                    torch.cumsum(torch.bincount(of_edge, minlength=self._g), 0)])

    def __len__(self):
        return -(-self._g // self._bs)

    def batches(self, seed: int | None = None) -> Iterator[dict]:
        """Every graph once: a permutation seeded `seed`, or the stored order when None."""
        order = (torch.arange(self._g) if seed is None else
                 torch.randperm(self._g, generator=torch.Generator().manual_seed(int(seed))))
        for start in range(0, self._g, self._bs):
            yield self.collate(order[start:start + self._bs])

    def collate(self, graphs) -> dict:
        """The batch of the graphs `graphs` (part-level graph indices), in that order."""
        graphs = torch.as_tensor(graphs, dtype=torch.long)
        n_sizes = self._node_ptr[graphs + 1] - self._node_ptr[graphs]
        e_sizes = self._edge_ptr[graphs + 1] - self._edge_ptr[graphs]
        nodes = self._nodes[_ranges(self._node_ptr[graphs], n_sizes)]
        rows = self._edges[_ranges(self._edge_ptr[graphs], e_sizes)]
        shift = (torch.cumsum(n_sizes, 0) - n_sizes).repeat_interleave(e_sizes)
        batch = _gathered(self._part, nodes,
                          {"index": self._local[self._index[:, rows]] + shift, "rows": rows})
        dev = self._part.x_num.device
        batch["graph_index"] = torch.repeat_interleave(torch.arange(len(graphs)),
                                                       n_sizes).to(dev)
        batch["num_graphs"] = int(len(graphs))
        if self._part.y is not None:
            batch["y"] = self._part.y[graphs.to(self._part.y.device)]
        return batch
