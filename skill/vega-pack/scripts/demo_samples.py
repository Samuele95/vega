"""
demo_samples — what a demonstration sample is drawn from: the train part's raw rows, one
candidate per sample of the bundle's shape, and the JSON a caller sends for each
(bundle-contract §7.4, build plan R-D8, R-A3, R-A5).

    DemoInputs(train, output, sample, schema, per_stratum, recipe)      what the demo reads
    samples_for(inputs) -> RowSamples | NodeSamples | GraphSamples | WindowSamples
    RequestColumns.of(schema, recipe, sample, columns)
    RowCodec(schema).rows(frame, edit=None) -> [{column: JSON cell}, …]

A candidate is a key (an int) of its shape:

  row    a train row: the request is the row alone.
  node   a train row as a query node (graph, level node): the request is the row alone, attached
         to the reference graph by the Recipe's `edges_for`; the raw key columns it is attached
         by (`GRAPH["query_columns"]`) travel with it.
  graph  a whole train graph (graph, level graph): the request is its rows, grouped by the raw
         `graph_id` column.
  window a train row that ends a window: a row with `history_rows` predecessors of its own
         series in the train part (and, for a forecast, `horizon` successors, so its truth is
         known). The request is the row, with those predecessors as its history — exactly what
         a caller sends for a fully determined answer (R-A5, R-A6).

`answers(predictor, keys)` is the bundle's own Predictor over the candidates, one answer row per
key: rows and nodes in chunks (an answer never depends on the other rows sent), graphs in chunks
of whole graphs, windows as runs of each series whose every requested row has its full history
inside the run. Nothing here reads the valid or the test part: `train` is all it is given.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

__all__ = ["DemoInputs", "GraphSamples", "NodeSamples", "RequestColumns", "RowCodec",
           "RowSamples", "WindowSamples", "samples_for"]

CHUNK_ROWS = 4096           # rows per Predictor call (rows, nodes, a window run)
CHUNK_GRAPHS = 256          # graphs per Predictor call
TOKENS = {math.inf: "Infinity", -math.inf: "-Infinity"}


@dataclass(frozen=True, eq=False)
class DemoInputs:
    """What the demonstration is drawn from: the train part's raw rows (index 0…n−1, in split
    order), the output and sample records the Predictor was built from, the input schema, the
    items per stratum, and the bundle's Recipe module (its `transform`, and `GRAPH`,
    `HISTORY_COLUMNS` when it has them)."""

    train: pd.DataFrame
    output: Mapping
    sample: Mapping
    schema: Mapping
    per_stratum: int
    recipe: Any
    seed: int = 7

    @property
    def shape(self) -> str:
        """row | window | node | graph."""
        shape = self.sample.get("shape") or "row"
        if shape == "graph":
            return "graph" if self.sample.get("level") == "graph" else "node"
        return shape


@dataclass(frozen=True)
class RequestColumns:
    """The raw columns a request row carries: the schema's inputs, then the raw key columns the
    Recipe attaches a graph query by when the schema does not list them; a history row adds the
    columns the Recipe's lags read that a request lacks (a lagged label). `keys` are the columns
    that place a row (series, order, graph): an edge preset never edits them."""

    request: tuple
    history: tuple
    keys: tuple

    @staticmethod
    def of(schema, recipe, sample, columns) -> "RequestColumns":
        present = set(columns)
        inputs = [c["name"] for c in schema.get("input") or () if c["name"] in present]
        graph = getattr(recipe, "GRAPH", None) or {}
        query = [c for c in graph.get("query_columns") or () if c in present and c not in inputs]
        request = tuple(inputs + query)
        lagged = [c for c in getattr(recipe, "HISTORY_COLUMNS", None) or ()
                  if c in present and c not in request]
        keys = tuple(dict.fromkeys(c for c in (sample.get("series_by"), sample.get("order_by"),
                                               sample.get("graph_id"), *query) if c))
        return RequestColumns(request=request, history=request + tuple(lagged), keys=keys)


class RowCodec:
    """Raw cells as the request contract types them (serve/rows.py): a text column as text (a
    code read as a number keeps its digits), a numeric column as a number or a non-finite token
    (`Infinity`, `-Infinity`), a missing value as null; a column the schema does not list (a
    graph key, a lagged label) as its own plain value."""

    def __init__(self, schema):
        inputs = {c["name"]: c for c in schema.get("input") or ()}
        self._text = {n for n, c in inputs.items() if c.get("dtype") == "string"}
        self._ints = {n for n, c in inputs.items() if c.get("dtype") == "int"}
        self._numeric = set(inputs) - self._text

    def rows(self, frame, edit=None) -> list:
        """One JSON row per frame row; `edit` = {column: value} applied to every row."""
        out = []
        for record in frame.to_dict("records"):
            record.update(edit or {})
            out.append({str(k): self.cell(k, v) for k, v in record.items()})
        return out

    def cell(self, column, value):
        if _missing(value):
            return None
        value = value.item() if isinstance(value, np.generic) else value
        if column in self._text:
            return _text(value)
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, float) and math.isinf(value):
            return TOKENS[value]
        if column in self._numeric:
            return _number(value, column in self._ints)
        return value


def _missing(value) -> bool:
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    return isinstance(value, (float, np.floating)) and math.isnan(float(value))


def _text(value) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _number(value, integral):
    if isinstance(value, str):
        return value                       # a token the source spelled (the recipe reads it)
    if integral and float(value).is_integer():
        return int(value)
    return float(value) if isinstance(value, float) else value


# --------------------------------------------------------------------------- the shapes

class RowSamples:
    """`shape: row` — each train row is a candidate, sent alone."""

    shape = "row"

    def __init__(self, inputs: DemoInputs, columns: RequestColumns):
        self._train, self._columns = inputs.train, columns
        self._codec = RowCodec(inputs.schema)
        self._label = (inputs.output.get("truth") or {}).get("column")

    @property
    def keys(self) -> np.ndarray:
        return np.arange(len(self._train), dtype=np.int64)

    @property
    def columns(self) -> RequestColumns:
        return self._columns

    def truth_positions(self, keys) -> np.ndarray:
        """(len(keys), k): the train positions whose truth cells make each candidate's truth
        (k is 1, or a forecast's horizon)."""
        return np.asarray(keys, dtype=np.int64).reshape(-1, 1)

    def anchor(self, key):
        """The train position an edge preset edits (None: the shape has no edge presets)."""
        return int(key)

    def answers(self, predictor, keys) -> pd.DataFrame:
        keys = np.asarray(keys, dtype=np.int64)
        parts = [_answer(predictor, self._train.iloc[keys[lo:lo + CHUNK_ROWS]]
                         [list(self._columns.request)])
                 for lo in range(0, len(keys), CHUNK_ROWS)]
        return _indexed(parts, keys)

    def item(self, key) -> dict:
        return {"rows": self._rows([key]), "history": None}

    def body(self, key, edit=None) -> dict:
        return {"row": self._rows([key], edit)[0], "rows": None, "history": None}

    def table(self, keys):
        """The demo.csv rows: the requests, with the truth column beside them."""
        cols = list(self._columns.request) + [c for c in (self._label,)
                                              if c and c in self._train.columns
                                              and c not in self._columns.request]
        return self._train.iloc[np.asarray(keys, dtype=np.int64)][cols]

    def _rows(self, positions, edit=None) -> list:
        frame = self._train.iloc[list(map(int, positions))][list(self._columns.request)]
        return self._codec.rows(frame, edit)


class NodeSamples(RowSamples):
    """`shape: graph`, level node — each train row is a query node, sent alone (R-A3)."""

    shape = "node"

    def body(self, key, edit=None) -> dict:
        return {"row": None, "rows": self._rows([key], edit), "history": None}


class GraphSamples(RowSamples):
    """`shape: graph`, level graph — each train graph is a candidate, sent whole."""

    shape = "graph"

    def __init__(self, inputs: DemoInputs, columns: RequestColumns):
        super().__init__(inputs, columns)
        group = inputs.sample.get("graph_id")
        ids = self._train[group].astype(str).to_numpy()
        first = pd.unique(ids)
        where = pd.Series(np.arange(len(ids)), dtype=np.int64).groupby(ids, sort=False)
        by_id = {gid: pos.to_numpy(dtype=np.int64) for gid, pos in where}
        self._graphs = tuple(by_id[gid] for gid in first)

    @property
    def keys(self) -> np.ndarray:
        return np.arange(len(self._graphs), dtype=np.int64)

    def truth_positions(self, keys) -> np.ndarray:
        firsts = np.asarray([g[0] for g in self._graphs], dtype=np.int64)
        return firsts[np.asarray(keys, dtype=np.int64)].reshape(-1, 1)

    def anchor(self, key):
        return None

    def answers(self, predictor, keys) -> pd.DataFrame:
        keys = np.asarray(keys, dtype=np.int64)
        parts = []
        for lo in range(0, len(keys), CHUNK_GRAPHS):
            chosen = [self._graphs[int(k)] for k in keys[lo:lo + CHUNK_GRAPHS]]
            frame = self._train.iloc[np.concatenate(chosen)][list(self._columns.request)]
            got = _answer(predictor, frame)
            firsts = np.cumsum([0] + [len(g) for g in chosen[:-1]])
            parts.append(got.iloc[firsts])
        return _indexed(parts, keys)

    def item(self, key) -> dict:
        return {"rows": self._rows(self._graphs[int(key)]), "history": None}

    def body(self, key, edit=None) -> dict:
        return {"row": None, "rows": self._rows(self._graphs[int(key)], edit), "history": None}

    def table(self, keys):
        rows = np.concatenate([self._graphs[int(k)] for k in keys]) if len(keys) else []
        return super().table(rows)


class WindowSamples(RowSamples):
    """`shape: window` — each train row with `history_rows` predecessors of its series in the
    train part (and `horizon` successors for a forecast) ends a candidate window; its request is
    the row, with the predecessors as history."""

    shape = "window"

    def __init__(self, inputs: DemoInputs, columns: RequestColumns):
        super().__init__(inputs, columns)
        sample = inputs.sample
        self._need = int(sample.get("history_rows") or max(int(sample.get("length") or 1) - 1, 0))
        self._ahead = int(sample.get("horizon") or 0) \
            if inputs.output.get("reading") == "forecast" else 0
        self._series = _series_order(self._train, sample.get("series_by"),
                                     sample.get("order_by"))
        self._where = {int(p): (s, r) for s, rows in enumerate(self._series)
                       for r, p in enumerate(rows)}

    @property
    def keys(self) -> np.ndarray:
        keep = [p for rows in self._series
                for p in rows[self._need:len(rows) - self._ahead]]
        return np.asarray(sorted(int(p) for p in keep), dtype=np.int64)

    def truth_positions(self, keys) -> np.ndarray:
        """A sequence task's truth is the anchor's own label; a forecast's the `horizon` rows
        after the anchor in its series (R-A5)."""
        keys = np.asarray(keys, dtype=np.int64)
        if not self._ahead:
            return keys.reshape(-1, 1)
        out = np.empty((len(keys), self._ahead), dtype=np.int64)
        for i, key in enumerate(keys):
            s, r = self._where[int(key)]
            out[i] = self._series[s][r + 1:r + 1 + self._ahead]
        return out

    def history_positions(self, key) -> np.ndarray:
        s, r = self._where[int(key)]
        return self._series[s][r - self._need:r]

    def answers(self, predictor, keys) -> pd.DataFrame:
        keys = np.asarray(keys, dtype=np.int64)
        parts, order = [], []
        for series, ranks in self._runs(keys):
            rows = self._series[series]
            lo, hi = ranks[0] - self._need, ranks[-1] + 1
            frame = self._train.iloc[rows[lo:hi]][list(self._columns.history)]
            got = _answer(predictor, frame)
            parts.append(got.iloc[[r - lo for r in ranks]])
            order.extend(int(rows[r]) for r in ranks)
        got = _indexed(parts, np.asarray(order, dtype=np.int64))
        return got.loc[keys]

    def item(self, key) -> dict:
        return {"rows": self._rows([key]), "history": self._history(key)}

    def body(self, key, edit=None) -> dict:
        return {"row": None, "rows": self._rows([key], edit), "history": self._history(key)}

    def table(self, keys):
        return None                     # a window is a sequence: demo.json carries it whole

    def _history(self, key) -> list:
        frame = self._train.iloc[self.history_positions(key)][list(self._columns.history)]
        return self._codec.rows(frame)

    def _runs(self, keys):
        """(series, ranks): the requested anchors of each series, in runs of at most
        CHUNK_ROWS rows whose every anchor has its full history inside the run."""
        by_series = {}
        for key in keys:
            s, r = self._where[int(key)]
            by_series.setdefault(s, []).append(r)
        for s, ranks in sorted(by_series.items()):
            run = []
            for r in sorted(set(ranks)):
                if run and (r - self._need > run[-1] + 1 or r - run[0] + self._need >= CHUNK_ROWS):
                    yield s, run
                    run = []
                run.append(r)
            if run:
                yield s, run


def _series_order(train, series_by, order_by) -> tuple:
    """Each series' train positions, in the order the assembler lays the series out: by
    `order_by` when set (stable, so ties keep the part's order), else the part's order."""
    frame = pd.DataFrame({"pos": np.arange(len(train), dtype=np.int64)})
    frame["series"] = train[series_by].astype(str).to_numpy() if series_by else ""
    if order_by:
        frame["order"] = train[order_by].to_numpy()
        frame = frame.sort_values(["order"], kind="stable")
    return tuple(g["pos"].to_numpy(dtype=np.int64)
                 for _, g in frame.groupby("series", sort=True))


SHAPES = {"row": RowSamples, "node": NodeSamples, "graph": GraphSamples, "window": WindowSamples}


def samples_for(inputs: DemoInputs):
    """The candidates of the bundle's sample shape."""
    columns = RequestColumns.of(inputs.schema, inputs.recipe, inputs.sample,
                                inputs.train.columns)
    return SHAPES[inputs.shape](inputs, columns)


def _answer(predictor, frame) -> pd.DataFrame:
    """The Predictor's answer to raw rows, positionally (index 0…n−1)."""
    return predictor.predict(frame.reset_index(drop=True)).reset_index(drop=True)


def _indexed(parts, keys) -> pd.DataFrame:
    """The answer rows, one per key, indexed by the keys."""
    if not parts:
        return pd.DataFrame(index=pd.Index(keys, dtype=np.int64))
    got = pd.concat(parts, ignore_index=True)
    got.index = pd.Index(keys, dtype=np.int64)
    return got
