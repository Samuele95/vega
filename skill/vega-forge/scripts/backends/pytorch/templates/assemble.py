"""
{{PACKAGE}}.{{MODULE_MODEL}}.assemble — a served request's shaped rows as the batch the network
reads (support; used by the Predictor). Emitted by /vega-forge from
`backends/pytorch/templates/assemble.py`: fix the template and re-emit, never this copy.

`SampleAssembler(spec, reference=None).assemble(shaped, history=None, query_edges=None)` is the
serving side's one step from shaped rows to a batch (the Predictor's `prepare()`, `/v1/explain`,
pack's round trip). It cuts by the plans training and evaluation cut by (in `domain.py`) into
the same parts, so a served sample cannot drift from the evaluated one (`{{MODULE_INGESTION}}`
holds the training half). The package carries its spec's sample shape (`domain.CARRIED_SHAPES`), and so
does the strategy table:

# >>> shape: row
  row           `shaped` -> `x_num (n, F)`, `x_cat (n, C)`; sample i answers row i.
# <<< shape
# >>> shape: window
  window        `history ++ shaped` per `series_by` series, in `order_by` order. Every row is a
                window end, answered when L-1 rows precede it in its series (history included:
                the dense scoring plan with the history as each series' context, whatever
                `context` said of training's parts), else unscored `insufficient_context`. A
                forecast answers the request's last rows too (it forecasts the future), so the
                plan has no target offset. The autoregressive channel is the target column of
                `history` and `shaped`, last (F' = F + 1); `x_cat (B, L, C)`.
# <<< shape
# >>> shape: graph
  graph, node   the reference nodes (train ∪ valid), then the query rows; the reference edges,
                then each query link once, reference -> query (`query_edges`: `edges_for`'s
                `query_pos, ref_pos, weight`, or a `QueryEdges`); `target_index` = the query
                nodes. No query sends a message: no answer depends on the other rows (R-A3).
  graph, graph  each distinct `graph_id` of `shaped` is one graph (first-seen order), with all
                its `edges_within` edges (`src_pos, dst_pos, weight`) and none between graphs.
# <<< shape

Request coordinates: `index.rows[k]` is the position in `shaped` of the row sample k answers
(-1: a whole graph), `index.spans[k]` the first and last position a window read (history from
-len(history) to -1), `index.graphs[k]` a graph's id; samples come in request order, and
`sample_of_row[i]` is row i's sample (-1: unscored, `unscored` says why). History feeds the
transform's lags (R-A6) and a window's context only; an empty one is none. The batch is on the CPU.

Effective Java: EJ-23 (one strategy class per sample shape, chosen from a table); EJ-17 (the
result is a frozen value); EJ-49 (a refusal names the column, the row or the edge).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch

from ..domain import (NetSpec, SampleIndex, SampleShape, SampleSpec,
                      # >>> shape: row
                      TensorPart,
                      # <<< shape
                      # >>> shape: window
                      SeriesRows, WindowedPart,
                      # <<< shape
                      # >>> shape: graph
                      GraphPart, GraphProtocol, GraphReference, GraphSpec, QueryEdges,
                      # <<< shape
                      )


__all__ = ["AssembledBatch", "SampleAssembler"]


@dataclass(frozen=True, eq=False)
class AssembledBatch:
    """One request, assembled: the batch, which request row each sample answers (`index`, its
    inverse `sample_of_row`), the rows none answers and why (`unscored`). Compared by identity."""

    batch: Mapping
    index: SampleIndex
    unscored: tuple
    sample_of_row: Any

    def __len__(self):
        return len(self.index)


# --------------------------------------------------------------------------- inputs

def _require(frame, columns, what: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"the {what} lack the column(s) {missing} the spec reads; send them "
                         "as the shaped frame carries them")


def _refuse(value, name: str, why: str) -> None:
    if value is not None:
        raise ValueError(f"{name} was passed, but {why}")


def _tensor(values):
    """A tensor of its own: an array a frame lends read-only (copy-on-write) is copied."""
    if values is None:
        return None
    arr = np.ascontiguousarray(values)
    return torch.from_numpy(arr if arr.flags.writeable else arr.copy())


# >>> shape: window graph
def _keys(frame, column, what: str):
    """A key column (`series_by`, `order_by`, `graph_id`) as the plans read it: a missing
    value is None or NaN, whatever the frame's dtype called it."""
    if not column:
        return None
    _require(frame, (column,), what)
    values = frame[column]
    arr = values.to_numpy()
    missing = np.asarray(values.isna().to_numpy(), dtype=bool)
    if missing.any() and arr.dtype.kind == "O":
        arr = arr.copy()
        arr[missing] = None
    return arr
# <<< shape


# >>> shape: graph
def _weights(frame, what: str):
    """An edge frame's `weight` column: None when absent or missing throughout (an
    unweighted graph), refused when missing on some edges only."""
    if "weight" not in frame.columns:
        return None
    weight = np.asarray(frame["weight"].to_numpy(dtype="float64", na_value=np.nan))
    missing = np.isnan(weight)
    if missing.all():
        return None
    if missing.any():
        raise ValueError(f"the {what} have no weight at edge {int(np.argmax(missing))} while "
                         "others do; a graph is weighted throughout or not at all")
    return weight.astype(np.float32)
# <<< shape


def _numbers(frame, columns, dtype, what: str) -> np.ndarray:
    """Feature columns as one `dtype` block (a missing value NaN), refused by name when a
    column holds something else: a raw value the transform did not shape."""
    try:
        return np.array(frame[list(columns)].to_numpy(dtype=dtype, na_value=np.nan),
                        dtype=dtype, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"the {what}' column(s) {list(columns)} do not all read as numbers; "
                         f"the assembler reads shaped rows ({exc})") from exc


# >>> shape: graph
def _positions(frame, column: str):
    return np.asarray(frame[column].to_numpy(dtype="int64"), dtype=np.int64)


def _links(query_edges) -> QueryEdges:
    """`edges_for`'s frame (`query_pos, ref_pos, weight`) as the domain's `QueryEdges`."""
    if query_edges is None:
        return QueryEdges(query=(), ref=())
    if isinstance(query_edges, QueryEdges):
        return query_edges
    _require(query_edges, ("query_pos", "ref_pos"), "query edges (edges_for)")
    return QueryEdges(query=_positions(query_edges, "query_pos"),
                      ref=_positions(query_edges, "ref_pos"),
                      weight=_weights(query_edges, "query edges"))


def _within(query_edges, n_rows: int) -> tuple:
    """`edges_within`'s frame (`src_pos, dst_pos, weight`) as `(edge_index (2, E), weight)`,
    each end a position among the `n_rows` query rows."""
    if query_edges is None:
        return np.zeros((2, 0), dtype=np.int64), None
    _require(query_edges, ("src_pos", "dst_pos"), "graph edges (edges_within)")
    edges = np.stack([_positions(query_edges, "src_pos"), _positions(query_edges, "dst_pos")])
    outside = np.flatnonzero((edges < 0).any(axis=0) | (edges >= n_rows).any(axis=0))
    if len(outside):
        k = int(outside[0])
        raise ValueError(f"edge {k} ({int(edges[0, k])} → {int(edges[1, k])}) names a row "
                         f"outside the {n_rows} row(s) of the request")
    return edges, _weights(query_edges, "graph edges")


def _inductive(sample: SampleSpec) -> None:
    if sample.graph.protocol is GraphProtocol.TRANSDUCTIVE:
        raise ValueError("transductive graph protocol: the served answers would depend on the "
                         "fixed deployment graph; the assembler serves inductive graphs only")
# <<< shape


# >>> shape: window
def _answered(n_rows: int, rows) -> np.ndarray:
    """Request row -> the sample answering it, -1 for none."""
    out = np.full(int(n_rows), -1, dtype=np.int64)
    out[np.asarray(rows, dtype=np.int64)] = np.arange(len(rows), dtype=np.int64)
    return out
# <<< shape


class _Features:
    """The FeatureContract's columns of a shaped frame, read out of it (refused by name when
    one is missing or does not read as a number) and cut by the contract's one rule
    (`FeatureContract.encode`) — the rule the DataSource cuts training rows by."""

    def __init__(self, spec: NetSpec):
        self.contract = spec.feature_contract
        self._columns = self.contract.columns

    def x_num(self, frame, what: str) -> np.ndarray:
        _require(frame, self._columns, what)
        return self.contract.numeric_block(
            _numbers(frame, self.contract.numeric_columns, np.float32, what))

    def x_cat(self, frame, what: str):
        if not self.contract.nominal_columns:
            return None
        return self.contract.nominal_ids(
            _numbers(frame, self.contract.nominal_columns, np.float64, what))


# --------------------------------------------------------------------------- the strategies

# >>> shape: row
class _RowSamples:
    """`shape: row` — one sample per row, in request order."""

    def __init__(self, spec: NetSpec, sample: SampleSpec, reference):
        _refuse(reference, "a reference graph", "a row sample has no graph")
        self._features = _Features(spec)

    def assemble(self, shaped, history, query_edges) -> AssembledBatch:
        _refuse(history, "history", "a row sample reads only its rows (history feeds the "
                "transform's lags, before the batch is assembled)")
        _refuse(query_edges, "query_edges", "a row sample has no graph")
        n = len(shaped)
        part = TensorPart(x_num=_tensor(self._features.x_num(shaped, "rows")),
                          x_cat=_tensor(self._features.x_cat(shaped, "rows")), y=None,
                          row_ids=np.arange(n))
        return AssembledBatch(part.gather(torch.arange(n)), SampleIndex.of_rows(n), (),
                              np.arange(n, dtype=np.int64))
# <<< shape


# >>> shape: window
class _WindowSamples:
    """`shape: window` — every request row is a window end, over its series' history."""

    def __init__(self, spec: NetSpec, sample: SampleSpec, reference):
        _refuse(reference, "a reference graph", "a window sample has no graph")
        window = sample.window
        self._features = _Features(spec)
        self._length = int(window.length)
        self._series_by, self._order_by = window.series_by, window.order_by
        self._channel = spec.label if window.autoregressive else None
        if window.autoregressive and not spec.label:
            raise ValueError("sample.autoregressive reads the target's past, and the spec "
                             "names no task.label")
        # serving has no targets: the dense scoring plan without a forecast's target offset,
        # and the history as the context of every series
        self._plan = window.laid_out()

    def assemble(self, shaped, history, query_edges) -> AssembledBatch:
        _refuse(query_edges, "query_edges", "a window sample has no graph")
        n, n_h = len(shaped), (len(history) if history is not None else 0)
        rows_at = np.arange(n, dtype=np.int64)
        history_at = np.arange(-n_h, 0, dtype=np.int64) if history is not None else None
        past = None if history is None else self._series(history, history_at, "history rows")
        layout = self._plan.layout(self._series(shaped, rows_at, "rows"), past)
        plan = self._plan.plan(layout.blocks, dense=True)
        order = np.argsort(plan.row_ids, kind="stable")
        anchors, rows = plan.anchors[order], np.asarray(plan.row_ids, dtype=np.int64)[order]
        at = layout.take(history_at, rows_at)
        spans = np.stack([at[anchors - (self._length - 1)], at[anchors]], axis=1)
        held = WindowedPart(x_num=_tensor(self._flat_numeric(layout, shaped, history)),
                            x_cat=_tensor(self._flat_nominal(layout, shaped, history)), y=None,
                            anchors=torch.as_tensor(anchors),
                            targets=torch.as_tensor(anchors[:, None]), row_ids=rows,
                            length=self._length)
        unscored = tuple(sorted(plan.unscored, key=lambda u: u.raw_pos))
        index = SampleIndex(SampleShape.WINDOW, rows, spans=spans)
        return AssembledBatch(held.gather(torch.arange(len(anchors))), index, unscored,
                              _answered(n, rows))

    def _series(self, frame, positions, what: str) -> SeriesRows:
        return SeriesRows(raw_pos=positions, series=_keys(frame, self._series_by, what),
                          order=_keys(frame, self._order_by, what))

    def _numeric(self, frame, what: str) -> np.ndarray:
        x = self._features.x_num(frame, what)
        if self._channel is None:
            return x
        return np.concatenate([x, _past_target(frame, self._channel, what)[:, None]], axis=1)

    def _flat_numeric(self, layout, shaped, history) -> np.ndarray:
        past = None if history is None else self._numeric(history, "history rows")
        return layout.take(past, self._numeric(shaped, "rows"))

    def _flat_nominal(self, layout, shaped, history):
        rows = self._features.x_cat(shaped, "rows")
        if rows is None:
            return None
        past = None if history is None else self._features.x_cat(history, "history rows")
        return layout.take(past, rows)


def _past_target(frame, label: str, what: str) -> np.ndarray:
    """The autoregressive channel: the target's value at each row, every one present."""
    if label not in frame.columns:
        raise ValueError(f"the {what} lack the target column {label!r}: an autoregressive "
                         "forecast reads the target's past as an input channel, so the column "
                         "is required in the rows and in the history")
    try:
        values = np.asarray(frame[label].to_numpy(dtype="float64", na_value=np.nan))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"the target column {label!r} of the {what} is not numeric") from exc
    missing = np.flatnonzero(np.isnan(values))
    if len(missing):
        raise ValueError(f"the target column {label!r} is missing at {what} position "
                         f"{int(missing[0])}; the autoregressive channel reads the target at "
                         "every input row of a window")
    return values.astype(np.float32)
# <<< shape


# >>> shape: graph
class _NodeSamples:
    """`shape: graph`, `level: node` — the query rows against the reference graph."""

    def __init__(self, spec: NetSpec, sample: SampleSpec, reference):
        _inductive(sample)
        if not isinstance(reference, GraphReference):
            raise ValueError("a node-level graph sample answers through the reference graph "
                             "(train ∪ valid, vega/graph-reference@1); pass "
                             "GraphReference.from_frames(nodes, edges, spec.feature_contract)")
        self._features, self._reference = _Features(spec), reference
        self._ref_num = np.asarray(reference.x_num, dtype=np.float32)
        self._ref_cat = None if reference.x_cat is None else np.asarray(reference.x_cat, np.int64)
        have = (self._ref_num.shape[1], 0 if self._ref_cat is None else self._ref_cat.shape[1])
        want = (len(self._features.contract.numeric_columns),
                len(self._features.contract.nominal_columns))
        if have != want:
            raise ValueError(f"the reference graph has {have[0]} numeric and {have[1]} nominal "
                             f"column(s); the spec's FeatureContract names {want[0]} and {want[1]}")

    def assemble(self, shaped, history, query_edges) -> AssembledBatch:
        _refuse(history, "history", "a graph sample takes no history: the reference graph "
                "is its context")
        q = len(shaped)
        g = self._reference.attach(_links(query_edges), q)
        x_num = np.concatenate([self._ref_num, self._features.x_num(shaped, "rows")])
        cat = self._features.x_cat(shaped, "rows")
        x_cat = None if cat is None else np.concatenate([self._ref_cat, cat])
        part = GraphPart(x_num=_tensor(x_num), x_cat=_tensor(x_cat), y=None, row_ids=np.arange(q),
                         edge_index=_tensor(g.edge_index), edge_weight=_tensor(g.edge_weight),
                         target_index=_tensor(g.target_index))
        rows = np.arange(q, dtype=np.int64)
        return AssembledBatch(part.as_batch(), SampleIndex(SampleShape.GRAPH, rows), (), rows)


class _GraphSamples:
    """`shape: graph`, `level: graph` — every distinct `graph_id` of the request is one graph."""

    def __init__(self, spec: NetSpec, sample: SampleSpec, reference):
        _inductive(sample)
        _refuse(reference, "a reference graph", "at level graph a query is a whole graph, "
                "with no reference to join")
        self._features = _Features(spec)
        self._graph_id = sample.graph.graph_id

    def assemble(self, shaped, history, query_edges) -> AssembledBatch:
        _refuse(history, "history", "a graph sample takes no history: a query is a whole graph")
        edge_index, weight = _within(query_edges, len(shaped))
        groups = GraphSpec.graphs(_keys(shaped, self._graph_id, "rows"), edge_index)
        part = GraphPart(x_num=_tensor(self._features.x_num(shaped, "rows")),
                         x_cat=_tensor(self._features.x_cat(shaped, "rows")),
                         edge_index=_tensor(edge_index), edge_weight=_tensor(weight),
                         target_index=None, y=None,
                         row_ids=np.asarray(groups.values, dtype=object),
                         graph_index=_tensor(groups.graph_index), num_graphs=groups.num_graphs)
        index = SampleIndex(SampleShape.GRAPH, (-1,) * groups.num_graphs, graphs=groups.values)
        return AssembledBatch(part.as_batch(), index, (),
                              np.asarray(groups.graph_index, dtype=np.int64))
# <<< shape


# sample shape -> its strategy: the shapes this package carries (domain.CARRIED_SHAPES)
_STRATEGIES = {
    # >>> shape: row
    SampleShape.ROW: _RowSamples,
    # <<< shape
    # >>> shape: window
    SampleShape.WINDOW: _WindowSamples,
    # <<< shape
    # >>> shape: graph
    SampleShape.GRAPH: lambda spec, sample, reference: (
        _NodeSamples if sample.is_node_level else _GraphSamples)(spec, sample, reference),
    # <<< shape
}


# --------------------------------------------------------------------------- the assembler

class SampleAssembler:
    """Shaped rows of one request -> `AssembledBatch`, by the spec's sample shape.

    Built once per served model: the spec's sample and FeatureContract are read, the reference
    graph (node level only) held, never mutated; `assemble` keeps no state, so requests share it.
    """

    def __init__(self, spec: NetSpec, reference=None):
        """`reference`: a node-level graph's reference graph (`GraphReference`), else None."""
        spec = NetSpec.require(spec)
        self._sample = SampleSpec.from_spec(spec)
        self._strategy = _STRATEGIES[self._sample.shape](spec, self._sample, reference)

    @property
    def sample(self) -> SampleSpec:
        return self._sample

    def assemble(self, shaped, history=None, query_edges=None) -> AssembledBatch:
        """`shaped`: the request's rows, shaped (the contract's columns, the key columns, an
        autoregressive forecast's target). `history`: shaped rows before them (windows).
        `query_edges`: `edges_for`'s links (node level), `edges_within`'s edges (graph level)."""
        for name, frame in (("shaped", shaped), ("history", history)):
            if (frame is not None or name == "shaped") and not hasattr(frame, "columns"):
                raise TypeError(f"{name} is a {type(frame).__name__}; the assembler reads "
                                "shaped rows as a pandas DataFrame")
        empty = history is not None and not len(history)
        return self._strategy.assemble(shaped, None if empty else history, query_edges)
