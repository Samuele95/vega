"""
{{PACKAGE}}.{{MODULE_TRAINING}}.graph_engines — the epoch sources of a graph sample (a
`kind: support` file of the Trainer's module; emitted when `data_contract.sample.shape` is
`graph`).

Emitted by /vega-forge from `backends/pytorch/templates/graph_engines.py`. A design defect here
is fixed in that template and the package re-emitted; never in this copy (code-contract §4).

Which nodes and edges a view holds is the DataSource's (`GraphSpec.view`, the inference-graph
rule R-A3): the training view's reference and targets are the train nodes; the valid view adds
the valid nodes as queries attached by reference → query edges only. This file only orders the
passes over them (backend design §10.2):

| level, batching | source | train batches | valid | `shuffle.kind` |
|---|---|---|---|---|
| node, `full` | `FullGraph` | one batch: the train view | the valid view, one batch | `full-graph` |
| node, `sampled` | `SampledGraph` | `NeighbourSampler` over the train view, seeded `seed + epoch`, with replacement | the full valid view | `neighbour-sampled` |
| graph | `GraphMinibatches` | `GraphCollate`: whole graphs, a permutation seeded `seed + epoch` | the valid graphs, stored order | `graph-minibatches` |

Scoring never samples: a valid pass is one full forward over the view (R-A11). A view that does
not fit the device's memory is an error naming `sample.batching` (`guard`), not a crash. The
`cuda_graph` hint is ignored — edge counts vary per batch under sampling, and a full-batch
epoch is one step, so there is nothing to amortise — and the record says so.
"""
from __future__ import annotations

import contextlib
from types import MappingProxyType
from typing import Iterator

from ..domain import GraphBatching, GraphSpec, NetSpec, Part
from ..ports import TrainingData
from ..domain import on_device
from .engines import batch_size

__all__ = ["FullGraph", "GraphMinibatches", "SampledGraph", "graph_source"]

#: what the record says of a `cuda_graph` hint on a graph sample
_NO_CAPTURE = "not applicable to graph samples"
_OOM_MARKS = ("out of memory", "can't allocate memory")


def _view_facts(part) -> dict:
    """{targets, nodes, edges} of a node-level view."""
    return {"targets": int(len(part.target_index)), "nodes": int(part.x_num.shape[0]),
            "edges": int(part.edge_index.shape[1])}


class _GraphSource:
    """What the graph sources share: both views held on the fit's device, the epoch as eager
    steps, the memory guard, the record's capture note."""

    zero_to_none = True
    capturable = False
    KIND = "graph"

    def __init__(self, spec: NetSpec, data: TrainingData, device, hints: dict):
        self._spec, self._device = spec, device
        self.train = data.load_resident(spec, Part.TRAIN, device)
        self.valid = data.load_resident(spec, Part.VALID, device)
        self._note = {"capture": _NO_CAPTURE} if hints.get("cuda_graph") else {}

    def run_epoch(self, epoch: int, driver) -> None:
        for batch in self.train_batches(epoch):
            driver.eager(on_device(batch, self._device))

    @contextlib.contextmanager
    def guard(self):
        """A pass that does not fit the device's memory, named by what changes it."""
        try:
            yield
        except RuntimeError as exc:           # torch.cuda.OutOfMemoryError is one
            if not any(mark in str(exc) for mark in _OOM_MARKS):
                raise
            raise MemoryError(
                f"the {self.KIND} pass over the graph view does not fit the device's memory "
                f"({self.train.x_num.shape[0]} train nodes); set data_contract.sample.batching "
                "to 'sampled' (neighbour sampling) or 'graphs', or train on a larger device "
                "(R-A11: scoring is full-batch over the view)") from exc

    def _described(self, epoch: int, facts: dict) -> dict:
        return {"kind": self.KIND, "seed": int(self._spec.seed + epoch), "sampling": facts,
                **self._note}


class FullGraph(_GraphSource):
    """Node level, `full`: every epoch is one step over the whole train view."""

    KIND = "full-graph"
    steps_per_epoch = 1

    def train_batches(self, epoch: int) -> Iterator[dict]:
        yield self.train.as_batch()

    def valid_batches(self) -> Iterator[dict]:
        yield self.valid.as_batch()

    def describe(self, epoch: int) -> dict:
        return self._described(epoch, _view_facts(self.train))


class SampledGraph(_GraphSource):
    """Node level, `sampled`: seed targets in `training.batch_size` batches, each with its
    sampled k-hop neighbourhood (`sample.neighbours` fan-outs, with replacement)."""

    KIND = "neighbour-sampled"

    def __init__(self, spec: NetSpec, data: TrainingData, device, hints: dict):
        super().__init__(spec, data, device, hints)
        from ..{{MODULE_INGESTION}}.graphs import NeighbourSampler
        self._sampler = NeighbourSampler.for_spec(spec, self.train)
        self._neighbours = [int(k) for k in GraphSpec.from_spec(spec).neighbours]
        self._drawn = {"nodes": 0, "edges": 0}
        self.steps_per_epoch = len(self._sampler)

    def train_batches(self, epoch: int) -> Iterator[dict]:
        self._drawn = {"nodes": 0, "edges": 0}
        for batch in self._sampler.epoch(int(self._spec.seed + epoch)):
            self._drawn["nodes"] += int(batch["x_num"].shape[0])
            self._drawn["edges"] += int(batch["edge_index"].shape[1])
            yield batch

    def valid_batches(self) -> Iterator[dict]:
        yield self.valid.as_batch()

    def describe(self, epoch: int) -> dict:
        """The epoch's sampled subgraphs, summed: the nodes and edges the convolutions saw."""
        facts = {"targets": int(len(self.train.target_index)), **self._drawn}
        return {**self._described(epoch, facts), "neighbours": self._neighbours}


class GraphMinibatches(_GraphSource):
    """Graph level: whole graphs, `training.batch_size` graphs per batch ("full": all)."""

    KIND = "graph-minibatches"

    def __init__(self, spec: NetSpec, data: TrainingData, device, hints: dict):
        super().__init__(spec, data, device, hints)
        from ..{{MODULE_INGESTION}}.graphs import GraphCollate
        size = batch_size(spec, len(self.train))
        self._train, self._valid = GraphCollate(self.train, size), GraphCollate(self.valid, size)
        self.steps_per_epoch = len(self._train)

    def train_batches(self, epoch: int) -> Iterator[dict]:
        return self._train.batches(int(self._spec.seed + epoch))

    def valid_batches(self) -> Iterator[dict]:
        return self._valid.batches(None)

    def describe(self, epoch: int) -> dict:
        """The train part's graphs (the samples the loss reads: its targets), nodes, edges."""
        facts = {"targets": int(self.train.num_graphs), "nodes": int(self.train.x_num.shape[0]),
                 "edges": int(self.train.edge_index.shape[1])}
        return self._described(epoch, facts)


# sample.batching -> the source ("graphs" is graph level's, and only its: GraphSpec checks it)
_BY_BATCHING = MappingProxyType({GraphBatching.FULL: FullGraph,
                                 GraphBatching.SAMPLED: SampledGraph,
                                 GraphBatching.GRAPHS: GraphMinibatches})


def graph_source(spec: NetSpec, data, device, hints: dict):
    """The source for a graph sample, by `sample.batching`."""
    gspec = GraphSpec.from_spec(NetSpec.require(spec))
    return _BY_BATCHING[gspec.batching](spec, data, device, hints)
