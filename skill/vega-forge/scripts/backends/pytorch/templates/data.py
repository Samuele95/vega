"""
{{PACKAGE}}.{{MODULE_INGESTION}}.data — the {{PORT_DATASOURCE}} adapter (role: DataSource).

{{RESPONSIBILITY_DATASOURCE}}
Source: {{SOURCE_DATASOURCE}}

Emitted by /vega-forge from `backends/pytorch/templates/data.py`. Fix defects in the template
and re-emit; never in this copy (code-contract §4).

`ShapedParts` hands out the shaped parts as the sample the spec names (backend design §9),
each with the raw-file position of every sample (the graph id at graph level) as its row id.
The reading and checking of files is `frames.py`'s; the plans live in `domain.py`, shared with
serving. The package carries its spec's sample shape and no other (`domain.CARRIED_SHAPES`,
selected when /vega-forge emitted it): a spec of another shape is refused by name where it
becomes a `NetSpec`, before this adapter sees it. The shapes carried here:
# >>> shape: row
  row     a `TensorPart` of rows; a row part is also streamed in chunks, never held whole.
# <<< shape
# >>> shape: window
  window  a `WindowedPart` of windows, cut by `windows.py`.
# <<< shape
# >>> shape: graph
  graph   a `GraphPart` of a graph view, built by `graphs.py`.
# <<< shape

The seal (netspec-contract §1, backend-contract §1) is a property of this file: `load_part`,
`iter_batches` and `load_resident` raise `SealedPartError` for the test part *before anything
is opened*, and the one door, `load_sealed_part`, demands a token that only /vega-eval's
`test_seal.json` carries — a file that does not exist until the evaluation runs, so neither the
forge nor the trainer can hold a valid one.
# >>> shape: window
A window of the test part reads train and valid rows as its context only through that door.
# <<< shape
# >>> shape: graph
A graph's train and valid views drop every edge touching a test node before anything becomes
a tensor.
# <<< shape
"""
from __future__ import annotations

from dataclasses import replace
from typing import Iterator

import numpy as np
# >>> shape: row
import pandas as pd
# <<< shape
import torch

from ..domain import (BatchPlan, BatchSize, DesignMatrix, DesignView, NetSpec, Part,
                      SamplePart, SampleShape, SampleSpec, SealedPartError,
                      UnsupportedShapeError,
                      # >>> shape: row
                      Task, TensorPart,
                      # <<< shape
                      # >>> shape: window graph
                      # >>> task: regression forecast
                      Task,
                      # <<< task
                      # <<< shape
                      # >>> shape: window
                      WindowedPart,
                      # <<< shape
                      # >>> shape: graph
                      GraphExposure, GraphLevel, GraphPart, host_array,
                      # >>> task: regression forecast
                      GraphSpec,
                      # <<< task
                      # <<< shape
                      )
from .frames import PartFrames, UnsealedFrames, unseal
# >>> shape: row
from .frames import row_tensors
# <<< shape
# >>> task: regression forecast
from .frames import label_tensor
# <<< task
# >>> shape: window
from .windows import WindowParts
# <<< shape
# >>> shape: graph
from .graphs import GraphCollate, GraphParts
# <<< shape

__all__ = ["ShapedParts"]

# >>> shape: row window
# the samples a design matrix is built from at a time (a window part is cut slice by slice)
_DESIGN_ROWS = 4096
# <<< shape


# >>> shape: row
def _shuffled(frame, to_batch, batch_size: int, rng) -> Iterator[dict]:
    order = rng.permutation(len(frame))
    for start in range(0, len(frame), batch_size):
        yield to_batch(frame.iloc[order[start:start + batch_size]])


def _streamed(chunks, to_batch, batch_size: int, plan: BatchPlan, rng) -> Iterator[dict]:
    """The row stream: chunks in file order, or a buffered shuffle over `shuffle_buffer` rows
    (a stream cannot permute what it has not read, so the buffer is part of what a run is)."""
    if not plan.shuffle:
        for frame in chunks:
            if len(frame):
                yield to_batch(frame)
        return
    buffer_rows = plan.buffer_rows(batch_size)
    buffer, held = [], 0
    for frame in chunks:
        if not len(frame):
            continue
        buffer.append(frame)
        held += len(frame)
        if held >= buffer_rows:
            yield from _shuffled(pd.concat(buffer, ignore_index=True), to_batch, batch_size, rng)
            buffer, held = [], 0
    if buffer:
        yield from _shuffled(pd.concat(buffer, ignore_index=True), to_batch, batch_size, rng)


def _streamed_count(sizes, batch_size: int, plan: BatchPlan) -> int:
    """How many batches `_streamed` yields over chunks of these row counts: its own arithmetic
    (a chunk a batch, or each buffer flushed in batches), with no row made a tensor."""
    if not plan.shuffle:
        return sum(1 for n in sizes if n)
    buffer_rows = plan.buffer_rows(batch_size)
    count, held = 0, 0
    for n in sizes:
        held += n
        if n and held >= buffer_rows:
            count, held = count + -(-held // batch_size), 0
    return count + -(-held // batch_size)


def _row_batch_size(spec: NetSpec, batch_size) -> int:
    """`batch_size`, else `training.batch_size` as a stream reads it (never `full`)."""
    return int(batch_size) if batch_size else spec.training.batch_size.streamed()
# <<< shape


def _batches(dataset, order, batch_size: int) -> Iterator[dict]:
    device = getattr(dataset.x_num, "device", None)
    for start in range(0, len(order), batch_size):
        yield dataset[torch.as_tensor(order[start:start + batch_size].copy(), dtype=torch.long,
                                      device=device)]


#: why every door but the sealed one refuses the test part, before a file is opened
_SEALED = ("{door} refuses the test part. The test is opened exactly once, by /vega-eval, "
           "through load_sealed_part(spec, token) with the token from the campaign's "
           "test_seal.json — which no other stage can have written.")


def _open_part(part, door: str) -> Part:
    """`part`, one the DataSource opens without a seal (train, valid); the test part is refused
    before a file is opened (`_SEALED`)."""
    part = Part(part)
    if part is Part.TEST:
        raise SealedPartError(_SEALED.format(door=door))
    return part


# >>> shape: graph
def _graph_design(dataset: GraphPart, view: DesignView) -> DesignMatrix:
    """FLAT: the targets' own features (structure-blind); NEIGHBOUR_MEAN: `[x | mean of the
    in-neighbours' x]` over the view's edges; GRAPH_POOLED: `[mean | max | sum]` per graph."""
    graph_level = dataset.level is GraphLevel.GRAPH
    if graph_level != (view is DesignView.GRAPH_POOLED):
        raise ValueError(f"the {view.value} design view does not read a "
                         f"{dataset.level.value}-level graph part: graph level reads "
                         "graph-pooled, node level flat or neighbour-mean")
    x = DesignMatrix.rows_of({"x_num": dataset.x_num, "x_cat": dataset.x_cat})
    if graph_level:
        x = DesignMatrix.pooled(x, dataset.graph_index, int(dataset.num_graphs))
    else:
        if view is DesignView.NEIGHBOUR_MEAN:
            x = np.concatenate([x, DesignMatrix.neighbour_mean(x, dataset.edge_index)], axis=1)
        x = x[host_array(dataset.target_index)]
    return DesignMatrix(x=x.astype(np.float32), y=host_array(dataset.y),
                        row_ids=np.asarray(dataset.row_ids))
# <<< shape


# >>> task: regression forecast
def _targets(y) -> np.ndarray | None:
    return None if y is None else np.asarray(y, dtype=np.float64).reshape(len(y), -1)
# <<< task


class ShapedParts:
    """The DataSource port's adapter: the shaped parts, as samples, behind the seal."""

    def __init__(self, reader: PartFrames | None = None):
        """Stateless by default. The sealed reader of the shaped files is injectable (tests); it
        has no door to the test part, so injecting one grants nothing: the one door is
        `frames.unseal`, which `load_sealed_part` calls itself."""
        self._frames = reader if reader is not None else PartFrames()

    def load_part(self, spec: NetSpec, part: Part) -> SamplePart:
        """The whole part as the spec's sample. Any request for `test` raises before a file is
        opened."""
        if Part(part) is Part.TEST:
            raise SealedPartError(_SEALED.format(door="load_part"))
        return self._build(NetSpec.require(spec), Part(part), self._frames)

    def iter_batches(self, spec: NetSpec, part: Part, batch_size: int | None = None,
                     columns=None, plan: BatchPlan | None = None) -> Iterator[dict]:
        """The part's batches; `plan` is the shuffle, its seed and its buffer.
# >>> shape: row
        A row part is streamed, never held whole (the memory budget is measured, and a reader
        that materialises the part to hand it out in slices spends it for nothing).
# <<< shape
# >>> shape: window graph
        This sample cannot be cut at a chunk boundary: its batches come from the held part, in
        the same seeded order `make_loader` gives.
# <<< shape
        """
        part = _open_part(part, "iter_batches")
        spec, plan = NetSpec.require(spec), plan or BatchPlan()
        shape = SampleSpec.from_spec(spec).shape
        # >>> shape: row
        if shape is SampleShape.ROW:
            return self._stream(spec, part, batch_size, columns, plan)
        # <<< shape
        # >>> shape: window graph
        if shape in (SampleShape.WINDOW, SampleShape.GRAPH):
            return self._held_batches(spec, part, batch_size, columns, plan)
        # <<< shape
        raise UnsupportedShapeError(shape)

    def batch_count(self, spec: NetSpec, part: Part, batch_size: int | None = None,
                    plan: BatchPlan | None = None) -> int:
        """How many batches `iter_batches(spec, part, batch_size, plan=plan)` yields, counted
        without building one: a streamed row part's from the row counts of a one-column read
        of its chunks, a held part's from its batches. The test part is refused."""
        part = _open_part(part, "batch_count")
        spec, plan = NetSpec.require(spec), plan or BatchPlan()
        shape = SampleSpec.from_spec(spec).shape
        # >>> shape: row
        if shape is SampleShape.ROW:
            size = _row_batch_size(spec, batch_size)
            one = (spec.feature_contract.columns + (Task.from_spec(spec).label,))[:1]
            chunks = self._frames.chunks(spec, part, [c for c in one if c], size)
            return _streamed_count((len(frame) for frame in chunks), size, plan)
        # <<< shape
        # >>> shape: window graph
        if shape in (SampleShape.WINDOW, SampleShape.GRAPH):
            return sum(1 for _ in self._held_batches(spec, part, batch_size, None, plan))
        # <<< shape
        raise UnsupportedShapeError(shape)

    def load_resident(self, spec: NetSpec, part: Part, device) -> SamplePart:
        """The same part as `load_part`, moved to `device` once; every minibatch is then a
        gather on the device. The test part is refused before a file is opened."""
        return self.load_part(spec, _open_part(part, "load_resident")).to(torch.device(device))

    def make_loader(self, dataset: SamplePart, spec: NetSpec, shuffle: bool = False,
                    seed: int | None = None, batch_size: int | None = None) -> Iterator[dict]:
        """Batches of a held part in a seeded order (no worker pool: it would change the order).
# >>> shape: graph
        A node-level graph part is one batch of all its targets unless `batch_size` is given
        (scoring is one full forward over the view); a graph-level part is collated by whole
        graphs.
# <<< shape
        """
        spec = NetSpec.require(spec)
        seed = int(seed if seed is not None else spec.seed)
        # >>> shape: graph
        if isinstance(dataset, GraphPart) and dataset.level is GraphLevel.GRAPH:
            return GraphCollate(dataset, self._batch_size(spec, dataset, batch_size)).batches(
                seed if shuffle else None)
        if isinstance(dataset, GraphPart) and batch_size is None:
            batch_size = "full"          # a node-level part: one batch of all its targets
        # <<< shape
        size = self._batch_size(spec, dataset, batch_size)
        order = np.arange(len(dataset))
        if shuffle:
            np.random.default_rng(seed).shuffle(order)
        return _batches(dataset, order, size)

    def load_sealed_part(self, spec: NetSpec, token: str,
                         seal_path: str = "test_seal.json") -> SamplePart:
        """The one door to the test part. /vega-eval's evaluator calls it; nothing else may.
        The path and digest come from the shaped manifest's `outputs.files.test` — the spec
        never names them — and the token is checked against the seal file the evaluation
        wrote before anything is opened."""
        spec = NetSpec.require(spec)
        frames = unseal(spec, token, seal_path)
        return replace(self._build(spec, Part.TEST, frames), seal=frames.seal)

    def design_matrix(self, spec: NetSpec, dataset: SamplePart,
                      view: DesignView = DesignView.FLAT) -> DesignMatrix:
        """A held part as the flat matrix a baseline reads (D1 §11.3):
# >>> shape: row
        rows as they are;
# <<< shape
# >>> shape: window
        windows flattened;
# <<< shape
# >>> shape: graph
        graph targets by their own features or with their neighbours' mean, graphs pooled;
# <<< shape
        the design view is the spec's."""
        NetSpec.require(spec)
        view = DesignView(view)
        # >>> shape: graph
        if isinstance(dataset, GraphPart):
            return _graph_design(dataset, view)
        # <<< shape
        # >>> shape: row window
        if view is not DesignView.FLAT:
            raise ValueError(f"the {view.value} design view reads a graph part, and this part "
                             f"is a {type(dataset).__name__}")
        return DesignMatrix.of_part(dataset, _DESIGN_ROWS)
        # <<< shape

    # >>> shape: graph
    def exposure(self, spec: NetSpec) -> GraphExposure:
        """The graph record /vega-eval writes: what training saw of the test part (zero and
        false for inductive by construction). A graph package's method only (SOLID-I)."""
        return GraphParts(self._frames).exposure(NetSpec.require(spec))
    # <<< shape

    # >>> shape: window
    def target_scale(self, spec: NetSpec, season: int = 1) -> float | None:
        """mase's train-side seasonal scale: a forecast window's (None for a window that is no
        forecast: `MetricOptions.mase_scale` None, mase unavailable, the record says why). A
        window package's method only (SOLID-I)."""
        return WindowParts(self._frames).target_scale(NetSpec.require(spec), season)
    # <<< shape

    # >>> task: regression forecast
    def train_targets(self, spec: NetSpec) -> np.ndarray | None:
        """The TRAIN part's targets, one row per training sample as the loss reads it, `(N, k)`
        float64, or None without a label:
# >>> shape: row
        a row's label column;
# <<< shape
# >>> shape: window
        a window's H future rows (its own label for a sequence task);
# <<< shape
# >>> shape: graph
        a node's label column, a graph's label at graph level;
# <<< shape
        only the train part is opened (A37: what a regression-family head's target affine is
        fitted on)."""
        spec = NetSpec.require(spec)
        sample, task = SampleSpec.from_spec(spec), Task.from_spec(spec)
        if not task.label:
            return None
        # >>> shape: window
        if sample.window is not None:
            part = WindowParts(self._frames).part(spec, Part.TRAIN)
            y = None if part.y is None else np.asarray(part.y)[np.asarray(part.targets)]
            return _targets(y)
        # <<< shape
        keys = sample.keys if sample.is_graph_level else ()
        frame = self._frames.open(spec, Part.TRAIN, columns=[task.label, *keys]).frame
        y = label_tensor(frame, task)
        y = None if y is None else y.numpy()
        # >>> shape: graph
        if y is not None and keys:
            y = y[GraphSpec.graphs(frame[keys[0]].to_numpy()).first_rows]
        # <<< shape
        return _targets(y)
    # <<< task

    def _build(self, spec: NetSpec, part: Part, frames: PartFrames | UnsealedFrames) -> SamplePart:
        # the sample spec is read (and a graph spec's declared exposure checked) before any file
        shape = SampleSpec.from_spec(spec).shape
        # >>> shape: row
        if shape is SampleShape.ROW:
            return self._rows(spec, part, frames)
        # <<< shape
        # >>> shape: window
        if shape is SampleShape.WINDOW:
            return self._windows(spec, part, frames)
        # <<< shape
        # >>> shape: graph
        if shape is SampleShape.GRAPH:
            return self._graphs(spec, part, frames)
        # <<< shape
        raise UnsupportedShapeError(shape)

    # >>> shape: row
    @staticmethod
    def _rows(spec: NetSpec, part: Part, frames: PartFrames | UnsealedFrames) -> TensorPart:
        opened = frames.open(spec, part)
        contract = spec.feature_contract
        x_num, x_cat, y = row_tensors(opened.frame, contract, Task.from_spec(spec), part.value)
        return TensorPart(x_num, x_cat, y, row_ids=opened.raw_pos,
                          columns=contract.columns, digest=opened.digest)
    # <<< shape

    # >>> shape: window
    @staticmethod
    def _windows(spec: NetSpec, part: Part, frames: PartFrames | UnsealedFrames) -> WindowedPart:
        return WindowParts(frames).part(spec, part)
    # <<< shape

    # >>> shape: graph
    @staticmethod
    def _graphs(spec: NetSpec, part: Part, frames: PartFrames | UnsealedFrames) -> GraphPart:
        return GraphParts(frames).part(spec, part)
    # <<< shape

    # >>> shape: window graph
    def _held_batches(self, spec: NetSpec, part: Part, batch_size, columns, plan: BatchPlan):
        """A held sample's batches, in `make_loader`'s seeded order."""
        if columns:
            raise ValueError("a held sample reads every feature column; `columns` projects a "
                             "streamed row part only")
        return self.make_loader(self.load_part(spec, part), spec, plan.shuffle, plan.seed,
                                batch_size)
    # <<< shape

    # >>> shape: row
    def _stream(self, spec: NetSpec, part: Part, batch_size, columns, plan: BatchPlan):
        contract, task = spec.feature_contract, Task.from_spec(spec)
        size = _row_batch_size(spec, batch_size)
        cols = (list(columns) if columns else
                [c for c in contract.columns + (task.label,) if c])
        chunks = self._frames.chunks(spec, part, cols, size)        # checked when opened

        def to_batch(frame):
            x_num, x_cat, y = row_tensors(frame, contract, task, part.value)
            out = {"x_num": x_num, "x_cat": x_cat, "y": y}
            return {k: v for k, v in out.items() if v is not None}

        return _streamed(chunks, to_batch, int(size), plan, plan.rng(spec.seed))
    # <<< shape

    @staticmethod
    def _batch_size(spec: NetSpec, dataset, batch_size) -> int:
        """`batch_size`, else `training.batch_size`; "full" is every sample of the part in one
        batch."""
        size = spec.training.batch_size if batch_size is None else BatchSize.of(batch_size)
        return size.for_part(len(dataset))
