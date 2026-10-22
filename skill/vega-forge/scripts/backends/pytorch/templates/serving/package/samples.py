"""
{{PACKAGE}}.{{MODULE_SERVING}}.samples — a request's raw rows as the batch the network reads,
by the sample shape (support; used by the Predictor, whose `prepare()` is the one raw -> batch
path, and through it by `/v1/explain` and pack's round trip).

Rendered by /vega-pack from `backends/pytorch/templates/serving/package/samples.py`: fix the
template and re-pack, never this copy.

`assembler_for(records, transform, context=None)` returns the adapter of the spec's sample
shape; each is a thin adapter over the Model module's `SampleAssembler` (the plans training and
evaluation cut by, so a served batch cannot drift from the evaluated one): it runs the Recipe's
`transform`, hands the shaped rows over, and says per request row why it was or was not
answered. `assemble(df_raw, history=None) -> Prepared`.

# >>> shape: row
  row     `transform(rows, history=history)` — history feeds a lag recipe's lags only — then
          one sample per row.
# <<< shape
# >>> shape: window
  window  `history ++ rows` is transformed as ONE frame with an empty lag history (R-A6): lags
          run from the history into the rows. The shaped history is each series' context;
          every request row is a window end, answered iff `length − 1` rows of its series
          precede it (history included), else `window incomplete` (R-A5). An autoregressive
          forecast's target column is required in the rows and the history (copied from them:
          the transform drops the label). A row whose window starts within its series' first
          `lag_depth` rows gets the lag warning (`history_rows` rows of history: determined).
# <<< shape
# >>> shape: graph
  graph   no history (the reference graph, or the whole graph, is the context). Node level:
          each row is a query node attached by the Recipe's `edges_for` to the reference graph
          (`GraphContext`), messages reference -> query only (an answer never depends on the
          other rows sent; a row with no link is answered from its own features, and says so).
          Graph level: each `graph_id` is one graph, its `edges_within` edges all kept.
# <<< shape
# >>> shape: window graph

A key column (`series_by`, `order_by`, `graph_id`) the transform did not return is taken from
the raw rows.
# <<< shape

`Prepared.shaped` is the shaped frame the batch was cut from, indexed in the request's
coordinates (request rows 0…n−1); `warnings` holds one tuple per request row (the shape's note
first, then unseen categories); the schema guard's warnings are the Predictor's to add.
# >>> shape: window
A window's history rows are −len(history)…−1 in `shaped`: the coordinates of `index.spans`.
# <<< shape

Effective Java: EJ-23 (one adapter per shape, chosen from a table); EJ-17 (`Prepared` and
`HeadOutput` are frozen values); EJ-49 (a refusal names the column, the shape or the level).
"""
from __future__ import annotations

# >>> shape: window
import inspect
# <<< shape
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..domain import (NetSpec, SampleIndex, SampleShape, SampleSpec,
                      # >>> shape: window
                      SeriesRows,
                      # <<< shape
                      )
from ..{{MODULE_MODEL}} import SampleAssembler
# >>> shape: graph
from .graphs import GraphContext
# <<< shape

__all__ = [
    # >>> shape: row
    "RowAssembler",
    # <<< shape
    # >>> shape: window
    "KeyRefused", "WindowAssembler",
    # <<< shape
    # >>> shape: graph
    "GraphAssembler",
    # <<< shape
    "HeadOutput", "Prepared", "assembler_for"]

# the texts are the HTTP contract's data, not chrome: a caller reads them per row
UNSEEN = "unseen category in {columns}"
# >>> shape: window
INCOMPLETE = ("window incomplete: needs {need} preceding rows, {have} available — send them "
              "as history")
LAG_START = ("lag features at the start of the window lack history ({lag_depth} rows) — send "
             "{history_rows} rows of history for a fully determined answer")
# <<< shape
# >>> shape: graph
NO_NEIGHBOUR = "no neighbour in the reference graph — answered from the row's own features"
# <<< shape
# >>> shape: window
KEY_UNREAD = "the recipe reads no {what} in {value!r}, and a row without one has no window"
NO_TARGET = ("the {what} lack the target column {target!r}: an autoregressive forecast reads "
             "the target's past as an input channel, so the column is required in the rows "
             "and in the history")


class KeyRefused(ValueError):
    """A window key the recipe left missing (rows in the request's coordinates): 422, named."""

    def __init__(self, column: str, rows: list, error: str):
        super().__init__(f"{column}: {error} (rows {rows[:5]})")
        self.column, self.rows, self.error = column, rows, error

    def problem_errors(self) -> list:
        return [{"column": self.column, "error": self.error, "rows": self.rows[:5], "count": len(self.rows)}]
# <<< shape


@dataclass(frozen=True, eq=False)
class Prepared:
    """One request, prepared: the `batch` the network reads, the `shaped` frame it was cut from,
    `sample_of_row[i]` the sample answering request row i (−1: unanswered), `index` (what each
    sample answers), a warnings tuple per request row, `n_samples`. Holds tensors: by identity."""

    batch: Mapping
    shaped: pd.DataFrame
    sample_of_row: np.ndarray
    index: SampleIndex
    warnings: tuple
    n_samples: int

    @property
    def answered(self) -> np.ndarray:
        """Per request row: whether a sample answers it."""
        return np.asarray(self.sample_of_row) >= 0


@dataclass(frozen=True, eq=False)
class HeadOutput:
    """The network's outputs for the samples of one `Prepared`, as the readers take them: `raw`
    `(n_samples, head_out)` float64 (a one-output head as `(n, 1)`), and the per-row score
    `(n_samples,)` of the reconstruction and anomaly readings, else None."""

    raw: Any
    row_score: Any = None

    def __post_init__(self):
        raw = np.asarray(self.raw, dtype=np.float64)
        raw = raw.reshape(-1, 1) if raw.ndim == 1 else raw
        if raw.ndim != 2:
            raise ValueError(f"a head output is (n_samples, head_out); got shape {raw.shape}")
        object.__setattr__(self, "raw", raw)
        if self.row_score is not None:
            score = np.asarray(self.row_score, dtype=np.float64).reshape(-1)
            if len(score) != len(raw):
                raise ValueError(f"{len(score)} row score(s) for {len(raw)} sample(s)")
            object.__setattr__(self, "row_score", score)


# --------------------------------------------------------------------------- helpers

def _frame(value, what: str) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"{what} is a {type(value).__name__}; a request's rows are a pandas "
                        "DataFrame of raw rows")
    return value.reset_index(drop=True)


# >>> shape: window
def _history(value):
    """The raw history rows, or None: absent, or empty (an empty history is none)."""
    frame = None if value is None else _frame(value, "history")
    return frame if frame is not None and len(frame) else None
# <<< shape


def _shaped(transform, frame, history) -> tuple:
    """`transform(frame, history=…, with_unseen=True)`: (shaped rows 0…n−1, unseen columns/row)."""
    shaped, unseen = transform(frame, history=history, with_unseen=True)
    if len(shaped) != len(frame):
        raise ValueError(f"the transform returned {len(shaped)} row(s) for {len(frame)}; it "
                         "answers one row per row it is given")
    return shaped.reset_index(drop=True), tuple(tuple(u) for u in unseen)


# >>> shape: window graph
def _kept(shaped, frame, columns) -> pd.DataFrame:
    """`shaped` plus each of `columns` it lacks, from the raw `frame` (a key, an AR target)."""
    copy = {c: frame[c].to_numpy() for c in columns
            if c and c not in shaped.columns and c in frame.columns}
    return shaped.assign(**copy) if copy else shaped
# <<< shape


def _warnings(n: int, unseen, notes: Mapping) -> tuple:
    """One tuple per request row: the shape's notes, then the unseen categories."""
    out = []
    for i in range(int(n)):
        seen = (UNSEEN.format(columns=", ".join(unseen[i])),) if unseen[i] else ()
        out.append(tuple(notes.get(i, ())) + seen)
    return tuple(out)


def _prepared(got, shaped, warnings) -> Prepared:
    return Prepared(batch=got.batch, shaped=shaped,
                    sample_of_row=np.asarray(got.sample_of_row, dtype=np.int64),
                    index=got.index, warnings=warnings, n_samples=len(got))


# >>> shape: window
def _required(transform) -> tuple:
    """The raw columns the Recipe reads (`REQUIRED_COLUMNS` of the transform's module)."""
    return tuple(getattr(inspect.getmodule(transform), "REQUIRED_COLUMNS", None) or ())


def _key(frame, column):
    return None if not column else frame[column].to_numpy()
# <<< shape


# --------------------------------------------------------------------------- the adapters

# >>> shape: row
class RowAssembler:
    """`shape: row` — one sample per request row."""

    def __init__(self, spec: NetSpec, transform):
        self._transform = transform
        self._assembler = SampleAssembler(spec)

    @property
    def sample(self) -> SampleSpec:
        return self._assembler.sample

    def assemble(self, df_raw, history=None) -> Prepared:
        """`history`: the raw rows before these, read by a lag recipe's lags only."""
        rows = _frame(df_raw, "rows")
        past = None if history is None else _frame(history, "history")
        shaped, unseen = _shaped(self._transform, rows, past)
        got = self._assembler.assemble(shaped)
        return _prepared(got, shaped, _warnings(len(rows), unseen, {}))
# <<< shape


# >>> shape: window
class WindowAssembler:
    """`shape: window` — every request row is a window end over its series' history."""

    def __init__(self, spec: NetSpec, transform, lag_depth: int = 0):
        self._transform = transform
        self._assembler = SampleAssembler(spec)
        window = self._assembler.sample.window
        # the order the assembler lays a series out in, for the per-row notes
        self._plain = window.laid_out()
        self._length, self._lag_depth = int(window.length), int(lag_depth or 0)
        self._target = spec.label if window.autoregressive else None
        self._keys = tuple(k for k in (window.series_by, window.order_by) if k)
        self._required = _required(transform)

    @property
    def sample(self) -> SampleSpec:
        return self._assembler.sample

    @property
    def history_rows(self) -> int:
        """The rows of history per series a fully determined answer needs (R-A5)."""
        return self._length - 1 + self._lag_depth

    def assemble(self, df_raw, history=None) -> Prepared:
        rows, past = _frame(df_raw, "rows"), _history(history)
        self._check(rows, past)
        n_h = 0 if past is None else len(past)
        frame = rows if past is None else pd.concat([past, rows], ignore_index=True, sort=False)
        shaped, unseen = _shaped(self._transform, frame, frame.iloc[0:0])
        shaped = _kept(shaped, frame, (*self._keys, self._target))
        self._refuse_unread_keys(shaped, frame, n_h)
        got = self._assembler.assemble(shaped.iloc[n_h:].reset_index(drop=True),
                                       shaped.iloc[:n_h] if n_h else None)
        notes = self._notes(got, shaped, n_h)
        shaped = shaped.set_axis(pd.RangeIndex(-n_h, len(rows)), axis=0)
        return _prepared(got, shaped, _warnings(len(rows), unseen[n_h:], notes))

    def _refuse_unread_keys(self, shaped, frame, n_h: int) -> None:
        """A key column the recipe left missing in some row: KeyRefused, before the plan."""
        for column, what in ((self._plain.series_by, "series"), (self._plain.order_by, "order")):
            lost = np.flatnonzero(pd.isna(shaped[column]).to_numpy(bool)) if column in shaped.columns else ()
            if len(lost):
                raw = frame[column].iloc[int(lost[0])] if column in frame.columns else None
                raise KeyRefused(column, [int(i) - n_h for i in lost], KEY_UNREAD.format(what=what, value=None if pd.isna(raw) else str(raw)))

    def _check(self, rows, past) -> None:
        """The columns the recipe reads, in the rows and in the history alike: concatenated,
        a column one of them lacks would read as missing values instead of being refused."""
        for frame, what in ((rows, "rows"), (past, "history rows")):
            if frame is None:
                continue
            if self._target and self._target not in frame.columns:
                raise ValueError(NO_TARGET.format(what=what, target=self._target))
            thin = [c for c in self._required if c not in frame.columns]
            if thin:
                raise KeyError(f"the {what} do not carry {', '.join(thin)}; the recipe reads "
                               "them in the rows and in the history alike")

    def _notes(self, got, shaped, n_h: int) -> dict:
        """Request row -> the window notes: why a row is unanswered, or that its window's
        first lags read no history."""
        rank, notes = self._ranks(shaped, n_h), {}
        for unscored in got.unscored:
            row = int(unscored.raw_pos)
            notes.setdefault(row, []).append(INCOMPLETE.format(need=self._length - 1,
                                                               have=int(rank[n_h + row])))
        if self._lag_depth:
            text = LAG_START.format(lag_depth=self._lag_depth, history_rows=self.history_rows)
            for first, last in got.index.spans or ():
                if rank[n_h + int(first)] < self._lag_depth:
                    notes.setdefault(int(last), []).append(text)
        return notes

    def _ranks(self, shaped, n_h: int) -> np.ndarray:
        """Each frame row's place in its series as the assembler lays it out: a history row
        among the history rows of its series, a request row after all of them."""
        rank, before = np.zeros(len(shaped), dtype=np.int64), {}
        for lo, hi in ((0, n_h), (n_h, len(shaped))):
            part = shaped.iloc[lo:hi]
            rows = SeriesRows(raw_pos=np.arange(lo, hi, dtype=np.int64),
                              series=_key(part, self._plain.series_by),
                              order=_key(part, self._plain.order_by))
            counts = {}
            for block in (self._plain.layout(rows).blocks if hi > lo else ()):
                rank[block.raw_pos] = before.get(block.key, 0) + np.arange(block.n_rows)
                counts[block.key] = block.n_rows
            before = counts
        return rank
# <<< shape


# >>> shape: graph
class GraphAssembler:
    """`shape: graph` — the request's rows against the reference graph (node level), or as
    whole graphs (graph level); a graph sample takes no history."""

    def __init__(self, spec: NetSpec, transform, context: GraphContext):
        if not isinstance(context, GraphContext):
            raise ValueError("a graph sample answers through its reference graph: pass the "
                             "GraphContext the bundle's predict.load_context() returns "
                             "(graph/*, vega/graph-reference@1)")
        graph = SampleSpec.from_spec(spec).graph
        if graph is None:
            raise ValueError("the spec's sample is not a graph; a GraphAssembler answers graph "
                             "samples only")
        level = graph.level
        if context.level is not level:
            raise ValueError(f"the reference graph is level {context.level.value}, and the "
                             f"spec's sample is level {level.value}: they are not one bundle's")
        self._transform, self._context = transform, context
        self._assembler = SampleAssembler(spec, context.reference)
        self._node = self._assembler.sample.is_node_level

    @property
    def sample(self) -> SampleSpec:
        return self._assembler.sample

    def assemble(self, df_raw, history=None) -> Prepared:
        if history is not None and len(history):
            raise ValueError("a graph sample takes no history: " + (
                "the reference graph is its context" if self._node else
                "a query is a whole graph"))
        rows = _frame(df_raw, "rows")
        shaped, unseen = _shaped(self._transform, rows, None)
        shaped = _kept(shaped, rows, self.sample.keys)
        edges = self._context.query_edges(rows)
        got = self._assembler.assemble(shaped, None, edges)
        notes = {}
        if self._node:
            linked = set(np.asarray(edges["query_pos"], dtype=np.int64).tolist())
            notes = {i: [NO_NEIGHBOUR] for i in range(len(rows)) if i not in linked}
        return _prepared(got, shaped, _warnings(len(rows), unseen, notes))
# <<< shape


# --------------------------------------------------------------------------- the choice

def _field(record, name: str):
    """A field of a record given as an object (`SampleRecord`) or a mapping (MANIFEST.sample);
    None for no record."""
    if isinstance(record, Mapping):
        return record.get(name)
    return getattr(record, name, None)


def _spec_of(records) -> NetSpec:
    spec = _field(records, "spec")
    if isinstance(spec, NetSpec):
        return spec
    if isinstance(spec, Mapping):
        return NetSpec.from_dict(spec)
    raise TypeError(f"records.spec is a {type(spec).__name__}; the serving records carry the "
                    "bundle's NetSpec")


# >>> shape: row window
def _no_context(context, shape: SampleShape) -> None:
    if context is not None:
        raise ValueError(f"a {shape.value} sample has no reference graph; the GraphContext is "
                         "a graph bundle's")
# <<< shape


# >>> shape: row
def _row(spec, transform, records, context) -> RowAssembler:
    _no_context(context, SampleShape.ROW)
    return RowAssembler(spec, transform)
# <<< shape


# >>> shape: window
def _window(spec, transform, records, context) -> WindowAssembler:
    _no_context(context, SampleShape.WINDOW)
    return WindowAssembler(spec, transform, int(_field(_field(records, "sample"),
                                                       "lag_depth") or 0))
# <<< shape


# >>> shape: graph
def _graph(spec, transform, records, context) -> GraphAssembler:
    return GraphAssembler(spec, transform, context)
# <<< shape


# sample shape -> its adapter: the shapes this bundle carries (domain.CARRIED_SHAPES)
_ADAPTERS = MappingProxyType({
    # >>> shape: row
    SampleShape.ROW: _row,
    # <<< shape
    # >>> shape: window
    SampleShape.WINDOW: _window,
    # <<< shape
    # >>> shape: graph
    SampleShape.GRAPH: _graph,
    # <<< shape
})


def assembler_for(records, transform, context=None):
    """The adapter of the bundle's sample shape. `records`: the serving records (`spec`, and
    `sample` — MANIFEST.sample — for the lag depth); `transform`: the Recipe's
    `transform(df_raw, history=None, with_unseen=False)`; `context`: the `GraphContext` of a
    graph bundle, else None. An uncarried shape is refused: `domain.UnsupportedShapeError`."""
    if not callable(transform):
        raise TypeError(f"transform is a {type(transform).__name__}; pass the Recipe module's "
                        "transform function")
    spec = _spec_of(records)
    shape = SampleSpec.from_spec(spec).shape
    stated = _field(_field(records, "sample"), "shape")
    if stated is not None and stated != shape.value:
        raise ValueError(f"MANIFEST.sample.shape is {stated!r} and the spec's sample is "
                         f"{shape.value!r}: they are not one bundle's")
    return _ADAPTERS[shape](spec, transform, records, context)
