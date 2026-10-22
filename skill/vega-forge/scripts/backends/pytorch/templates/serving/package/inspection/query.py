"""
inspection.query — one `POST /v1/explain` request, parsed and checked before the model runs
(inspect-contract §5.4; build plan R-A3, R-A5, R-F6).

`QueryRules.of(spec, max_rows=None)` reads, once, what a request of this bundle may hold: the
sample shape, the window's length and key columns, the graph's level and `graph_id` column, the
contract's `max_rows` bound on the history and the §7 limits. `rules.parse(body)` turns a
decoded request body into a frozen `ExplainQuery`, or raises `QueryError` — the 422 problem the
explain route answers, naming the request field:

  row     exactly one of `row`, or `rows` of length 1; `history` (a lag recipe's past rows)
          may travel with it; `at` is 0 or absent.
  window  `rows` in time order (several series when `series_by` is set), `history` the rows
          that precede them, `at` the index in `rows` of the explained window's anchor
          (default the last row). The anchor must have `length − 1` rows of its own series
          before it in `history + rows`, counted as the assembler lays a series out (history
          rows first, each part ordered by `order_by`): else "a window needs L rows ending at
          `at`".
  graph   `rows` only: the query nodes (node level; `at` the explained node, default the
          first) or the rows of the graphs (graph level; `at` a row of the explained graph,
          default the first). No edges travel — serving derives them (`edges_for`,
          `edges_within`) — and no history: a graph's context is the reference graph, or the
          graph itself.

`target` holds at most one selector (`class`, `label`, `threshold`, `quantile`, `feature`,
`step`, `kind`), checked here for its form and by `targets.TaskTargets` for the task; `method`
is `grad_x_input` or `integrated_gradients`, `steps` 4…`ig_steps_max` (default
`ig_steps_default`), `detail` `full`, `summary` or `exact`. A null field is an absent one; any
other field is refused. `ExplainQuery.sample(index)` resolves the explained sample `k` in the
batch `Predictor.prepare()` returned, through its `SampleIndex` (R-A2: an explained node is found
among the targets, never by its position in the batch); `interpolants(nodes)` bounds an
integrated-gradient batch (`steps × nodes ≤ ig_batch_max`) once the assembled node count is
known.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .codec import LIMITS
from ...domain import (SampleIndex, SampleShape, SampleSpec,
                       # >>> shape: window
                       SeriesRows, WindowContext,
                       # <<< shape
                       )

__all__ = ["DETAILS", "FIELDS", "METHODS", "SELECTORS", "ExplainQuery", "QueryError",
           "QueryRules", "target_form"]

FIELDS = ("row", "rows", "history", "at", "target", "method", "steps", "detail")
SELECTORS = ("class", "label", "threshold", "quantile", "feature", "step", "kind")
METHODS = ("grad_x_input", "integrated_gradients")
DETAILS = ("full", "summary", "exact")
MIN_STEPS = 4
# selector -> (the form it takes, its lower bound when an integer)
_NAMED = frozenset({"class", "label", "feature", "kind"})
_COUNTED = MappingProxyType({"threshold": 0, "step": 1})


class QueryError(ValueError):
    """A request the explain route refuses with 422: `field` names the request field."""

    status = 422

    def __init__(self, field: str, message: str):
        super().__init__(message)
        self.field = field


def _integer(value) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _plain(value):
    """A numpy scalar as its Python value."""
    return value.item() if isinstance(value, np.generic) else value


def _missing(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _frame(records, field: str, bounds: tuple) -> pd.DataFrame:
    """Raw rows as a frame: a DataFrame, or a list of row objects, `bounds[0]…bounds[1]` long."""
    if isinstance(records, pd.DataFrame):
        frame = records.reset_index(drop=True)
    elif isinstance(records, (list, tuple)):
        for i, rec in enumerate(records):
            if not isinstance(rec, Mapping):
                raise QueryError(f"{field}[{i}]", f"{field}[{i}] is a "
                                 f"{type(rec).__name__}; a raw row is an object of columns")
        frame = pd.DataFrame.from_records([dict(r) for r in records])
    else:
        raise QueryError(field, f"`{field}` is a {type(records).__name__}; it is a list of raw "
                                "rows")
    lo, hi = bounds
    if not lo <= len(frame) <= hi:
        raise QueryError(field, f"`{field}` holds {len(frame)} row(s); an explain request sends "
                                f"{lo} to {hi}")
    return frame


def _body(body) -> dict:
    """The request's fields, a null one dropped; an unknown field is refused."""
    if not isinstance(body, Mapping):
        raise QueryError("body", f"an explain request is an object, not a {type(body).__name__}")
    unknown = sorted(str(k) for k in body if k not in FIELDS)
    if unknown:
        raise QueryError(unknown[0], f"`{unknown[0]}` is not a field of an explain request "
                                     f"(the fields: {', '.join(FIELDS)})")
    return {k: v for k, v in body.items() if v is not None}


def _selector(key: str, value) -> None:
    """One selector's form: a name, a non-negative index, a step from 1, a quantile's index or
    its level in (0, 1)."""
    where = f"target.{key}"
    if key in _NAMED:
        if not isinstance(value, str) or not value:
            raise QueryError(where, f"{where} is a name (a non-empty string)")
    elif key in _COUNTED:
        if not _integer(value) or value < _COUNTED[key]:
            raise QueryError(where, f"{where} is an integer ≥ {_COUNTED[key]}")
    elif not ((_integer(value) and value >= 0) or
              (isinstance(value, float) and 0.0 < value < 1.0)):
        raise QueryError(where, f"{where} is a quantile's index (an integer ≥ 0) or its level "
                                "(a number in (0, 1))")


def target_form(value):
    """The target: absent, or an object with at most one selector of a known form. Public: the
    inspector and the attribution table check a hand-built query's or a row's selector with it."""
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise QueryError("target", f"`target` is an object with at most one selector, not a "
                                   f"{type(value).__name__}")
    given = {str(k): v for k, v in value.items() if v is not None}
    unknown = sorted(k for k in given if k not in SELECTORS)
    if unknown:
        raise QueryError(f"target.{unknown[0]}", f"target.{unknown[0]} is not a target "
                                                 f"selector (one of {', '.join(SELECTORS)})")
    if len(given) > 1:
        raise QueryError("target", "a target takes at most one selector; the request names "
                                   + ", ".join(sorted(given)))
    for key, v in given.items():
        _selector(key, v)
    return MappingProxyType(given) if given else None


def _choice(body, field: str, allowed: tuple, default: str) -> str:
    value = body.get(field, default)
    if value not in allowed:
        raise QueryError(field, f"`{field}` is one of {', '.join(allowed)}, not {value!r}")
    return value


def _steps(body, limits) -> int:
    value = body.get("steps", int(limits["ig_steps_default"]))
    top = int(limits["ig_steps_max"])
    if not _integer(value) or not MIN_STEPS <= value <= top:
        raise QueryError("steps", f"`steps` is an integer from {MIN_STEPS} to {top} (the "
                                  f"integrated-gradient interpolants), not {value!r}")
    return int(value)


def _index(value, n: int, default: int) -> int:
    """`at` as a position in `rows`: absent is `default`, a negative index counts from the end."""
    if value is None:
        return default
    if not _integer(value):
        raise QueryError("at", f"`at` is an integer index into `rows`, not {value!r}")
    at = int(value) + (n if value < 0 else 0)
    if not 0 <= at < n:
        raise QueryError("at", f"`at` is {value}, outside the {n} row(s) sent")
    return at


@dataclass(frozen=True, eq=False)
class ExplainQuery:
    """One parsed explain request: the raw `rows` and `history` frames, `at` (the explained
    row, a position in `rows`), the target selector, the method, its steps and the detail;
    at graph level `graph` is the `graph_id` of the explained row."""

    rows: pd.DataFrame
    history: pd.DataFrame | None
    at: int
    target: Mapping | None = None
    method: str = "grad_x_input"
    steps: int = int(LIMITS["ig_steps_default"])
    detail: str = "full"
    graph: Any = None

    def sample(self, index: SampleIndex) -> int:
        """The sample `k` of the prepared batch that answers the explained row: the graph of
        row `at` at graph level, else the sample whose answer belongs to row `at`."""
        if index.graphs is not None:
            hits = [k for k, g in enumerate(index.graphs)
                    if g == self.graph or str(g) == str(self.graph)]
            if not hits:
                raise QueryError("at", f"row {self.at} belongs to graph {self.graph!r}, which "
                                       "the prepared batch does not hold")
            return hits[0]
        try:
            return index.sample_of(self.at)
        except ValueError as exc:
            raise QueryError("at", f"row {self.at} is not answered by the prepared batch: "
                                   f"{exc}") from exc

    def interpolants(self, nodes: int, limits: Mapping = LIMITS) -> None:
        """Refuse an integrated-gradient batch of more than `ig_batch_max` interpolated nodes
        (`steps × nodes`, the nodes of one assembled sample)."""
        top = int(limits["ig_batch_max"])
        if self.method == "integrated_gradients" and self.steps * int(nodes) > top:
            raise QueryError("steps", f"integrated gradients at {self.steps} steps over "
                                      f"{int(nodes)} node(s) interpolate {self.steps * int(nodes)}"
                                      f" nodes, above ig_batch_max ({top}); lower `steps`")


@dataclass(frozen=True, eq=False)
class QueryRules:
    """What an explain request of one bundle may hold: its sample, the history bound (the
    serving contract's `max_rows`) and the §7 limits."""

    sample: SampleSpec
    max_rows: int | None = None
    limits: Mapping = LIMITS

    @staticmethod
    def of(spec, max_rows: int | None = None, limits: Mapping | None = None) -> "QueryRules":
        data = spec.to_dict() if callable(getattr(spec, "to_dict", None)) else dict(spec)
        return QueryRules(SampleSpec.from_spec(data),
                          None if max_rows is None else int(max_rows),
                          MappingProxyType(dict(limits or LIMITS)))

    @property
    def shape(self) -> str:
        return self.sample.shape.value

    def parse(self, body) -> ExplainQuery:
        """The request body (decoded JSON) as an `ExplainQuery`, or the `QueryError` that names
        the field at fault."""
        fields = _body(body)
        rows = self._rows(fields)
        history = self._history(fields.get("history"))
        at = self._at(fields.get("at"), len(rows))
        # >>> shape: window
        if self.sample.window is not None:
            self._window(rows, history, at)
        # <<< shape
        query = ExplainQuery(rows, history, at, target_form(fields.get("target")),
                             _choice(fields, "method", METHODS, METHODS[0]),
                             _steps(fields, self.limits),
                             _choice(fields, "detail", DETAILS, DETAILS[0]),
                             self._graph_of(rows, at))
        if self.sample.graph is not None:
            query.interpolants(len(rows), self.limits)
        return query

    # -- the rows -------------------------------------------------------------------------
    def _rows(self, fields) -> pd.DataFrame:
        one, many = fields.get("row"), fields.get("rows")
        bounds = (1, int(self.limits["explain_rows_max"]))
        if self.sample.shape is not SampleShape.ROW:
            if one is not None or many is None:
                raise QueryError("row" if one is not None else "rows",
                                 f"a {self.shape} sample takes `rows` (1 to {bounds[1]} raw "
                                 "rows), never `row`")
            return _frame(many, "rows", bounds)
        if (one is None) == (many is None):
            raise QueryError("row", "a row sample takes exactly one of `row`, or `rows` of "
                                    "length 1")
        if one is not None and not isinstance(one, Mapping):
            raise QueryError("row", f"`row` is an object of columns, not a {type(one).__name__}")
        if isinstance(many, (list, tuple, pd.DataFrame)) and len(many) != 1:
            raise QueryError("rows", f"a row sample explains one row: `rows` holds {len(many)};"
                                     " send `row`, or `rows` of length 1")
        return _frame([one] if one is not None else many, "rows", (1, 1))

    def _history(self, value) -> pd.DataFrame | None:
        if value is None or (hasattr(value, "__len__") and not len(value)):
            return None
        if self.sample.graph is not None:
            raise QueryError("history", "a graph sample takes no history: " + (
                "the reference graph is its context" if self.sample.is_node_level else
                "a query is a whole graph"))
        top = self.max_rows if self.max_rows is not None else int(1e12)
        return _frame(value, "history", (1, top))

    def _at(self, value, n: int) -> int:
        if self.sample.shape is SampleShape.ROW:
            if value is not None and _integer(value) and int(value) in (0, -1):
                return 0
            if value is not None:
                raise QueryError("at", f"a row sample explains its one row; `at` is 0 or "
                                       f"absent, not {value!r}")
            return 0
        return _index(value, n, n - 1 if self.sample.window is not None else 0)

    def _graph_of(self, rows, at: int):
        """The `graph_id` of the explained row, at graph level (else None)."""
        if not self.sample.is_graph_level:
            return None
        column = self.sample.graph.graph_id
        if column not in rows.columns:
            raise QueryError("rows", f"the rows lack the {column!r} column that groups them "
                                     "into graphs")
        value = _plain(rows[column].iloc[at])
        if _missing(value):
            raise QueryError("rows", f"row {at} has no {column!r}: every row names its graph")
        return value

    # >>> shape: window
    # -- the window -----------------------------------------------------------------------
    def _window(self, rows, history, at: int) -> None:
        length = int(self.sample.window.length)
        have = self._preceding(rows, history, at)
        if have < length - 1:
            raise QueryError("at", f"a window needs {length} rows ending at `at`: row {at} has "
                                   f"{have} earlier row(s) of its series in history + rows; send"
                                   f" {length - 1 - have} more as history")

    def _preceding(self, rows, history, at: int) -> int:
        """The rows of the anchor's series laid out before it: the series' history rows, then
        its request rows before it in `order_by` order (stable), as the assembler lays them."""
        plain = replace(self.sample.window, horizon=0, stride=1, autoregressive=False,
                        context=WindowContext.PREVIOUS_PART)
        key, rank = None, 0
        for block in self._layout(plain, rows, "rows"):
            where = np.flatnonzero(np.asarray(block.raw_pos) == at)
            if len(where):
                key, rank = block.key, int(where[0])
        if history is None:
            return rank
        earlier = [int(b.n_rows) for b in self._layout(plain, history, "history")
                   if b.key == key]
        return rank + sum(earlier)

    @staticmethod
    def _layout(plain, frame, field: str) -> tuple:
        """The series blocks of one frame, laid out by the window's key columns."""
        keys = []
        for column in (plain.series_by, plain.order_by):
            if column and column not in frame.columns:
                raise QueryError(field, f"the {field} lack the {column!r} column the window "
                                        "groups and orders its rows by")
            keys.append(frame[column].to_numpy() if column else None)
        try:
            rows = SeriesRows(raw_pos=np.arange(len(frame), dtype=np.int64), series=keys[0],
                              order=keys[1])
            return plain.layout(rows).blocks
        except ValueError as exc:            # the plan's own refusal: it names the column
            raise QueryError(field, f"the {field} cannot be laid out as series: {exc}") from exc
        except TypeError as exc:             # values that do not compare: never the library's words
            named = " and ".join(c for c in (plain.series_by, plain.order_by) if c)
            raise QueryError(field, f"the {field} cannot be laid out as series: a {named} value "
                                    "is empty or of another type than the others") from exc
    # <<< shape
