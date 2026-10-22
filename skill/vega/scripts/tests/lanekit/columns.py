"""
Where every raw column of a lane goes (lanekit.recipes): what the recipe keeps, drops, encodes and
scales, read from a generator's SCHEMA and the lane's declaration.

    plan = ColumnPlan.of(SCHEMA, lane)          # a ValueError names a column the lane and SCHEMA disagree on
    effective_task(lane)                        # `target.base` for a sequence-* lane
    task_prose(lane)                            # the geometry's words for the lane's task
    embedded_columns(blocks)                    # every column an `embedding` block names

A column's role (lanekit.datasets.ROLES) decides its place: the lane's label is selected and
every other label is leakage; keys, and on a lane that is not a window the timestamps and series
ids, identify rather than predict; a nominal is encoded ordinal when the network embeds it (an
`embedding` block names it, or a `feature_tokenizer` reads every nominal), one-hot otherwise —
the lane's `recipe_options.encode` first. Standard library only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

__all__ = ["ColumnPlan", "CLASS_TASKS", "effective_task", "task_prose", "embedded_columns"]

# the effective tasks whose label is a class (a stratified split keeps each class in each part)
CLASS_TASKS = frozenset({"binary", "multiclass", "ordinal", "anomaly"})
TASK_PROSE = {"binary": "binary classification", "multiclass": "multiclass classification",
              "multilabel": "multilabel classification", "regression": "regression",
              "ordinal": "ordinal regression", "anomaly": "binary classification",
              "forecast": "regression", "unsupervised": "none"}
ENCODE_NAMES = {"onehot": "one-hot", "one-hot": "one-hot", "one_hot": "one-hot",
                "ordinal": "ordinal"}


def _raw(lane: Any) -> Mapping[str, Any]:
    return lane.raw if hasattr(lane, "raw") else lane


def effective_task(lane: Any) -> str:
    """The task every reader of the base uses: `target.base` for a sequence-* lane."""
    raw = _raw(lane)
    if str(raw["task"]).startswith("sequence-"):
        return (raw.get("target") or {}).get("base") or raw["task"]
    return raw["task"]


def task_prose(lane: Any) -> str:
    """The geometry's task words for the lane's partition (the catalog's task_map, reversed)."""
    return TASK_PROSE.get(effective_task(lane), "regression")


def embedded_columns(blocks: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    """Every column an `embedding` block names, composites' inner blocks included."""
    out: list[str] = []
    for block in blocks or ():
        params = block.get("params") or {}
        if block.get("type") == "embedding":
            out.extend(c for c in params.get("columns") or () if c not in out)
        out.extend(c for c in embedded_columns(params.get("inner") or ()) if c not in out)
    return tuple(out)


def _has_block(blocks: Sequence[Mapping[str, Any]], kind: str) -> bool:
    for block in blocks or ():
        if block.get("type") == kind or _has_block((block.get("params") or {}).get("inner") or (), kind):
            return True
    return False


# --------------------------------------------------------------------------- the column plan
@dataclass(frozen=True)
class ColumnPlan:
    """Where every raw column of a lane goes: what the recipe keeps, drops, encodes, scales."""
    label: str | None
    leak: tuple[str, ...]
    keys: tuple[str, ...]
    times: tuple[str, ...]
    nominal: tuple[str, ...]
    ordinal: tuple[str, ...]
    onehot: tuple[str, ...]
    numeric: tuple[str, ...]
    series_by: str | None
    order_by: str | None
    graph_id: str | None
    lag_column: str | None
    lags: tuple[int, ...]
    user_encoded: tuple[str, ...] = field(default=())

    @classmethod
    def of(cls, schema: Sequence[Any], lane: Any) -> "ColumnPlan":
        """The plan of a lane over a generator's SCHEMA; a ValueError names the first column the
        lane declares and the schema does not hold in the role the lane needs."""
        raw = _raw(lane)
        by_role: dict[str, list[str]] = {}
        for col in schema:
            by_role.setdefault(col.role, []).append(col.name)
        label = (raw.get("target") or {}).get("column")
        series_by, order_by, graph_id = _lane_columns(raw, {c.name for c in schema}, by_role)
        times = (order_by,) if raw["shape"] == "window" else ()
        dropped_ids = (by_role.get("key", []) + [c for c in by_role.get("time", []) if c not in times]
                       + [c for c in by_role.get("series", []) if c != series_by]
                       + [c for c in by_role.get("group", []) if c != graph_id])
        nominal = tuple(by_role.get("nominal", []))
        ordinal, user_encoded = _ordinal(nominal, raw)
        lag = (raw.get("recipe_options") or {}).get("lag") or {}
        leak = tuple(c for c in by_role.get("label", []) if c != label) + tuple(by_role.get("drop", []))
        return cls(label=label, leak=leak, keys=tuple(dropped_ids), times=times, nominal=nominal,
                   ordinal=ordinal, onehot=tuple(c for c in nominal if c not in ordinal),
                   numeric=tuple(by_role.get("numeric", [])), series_by=series_by, order_by=order_by,
                   graph_id=graph_id, lag_column=lag.get("column"),
                   lags=tuple(range(1, int(lag.get("depth") or 0) + 1)), user_encoded=user_encoded)

    @property
    def lag_outputs(self) -> tuple[str, ...]:
        return tuple(f"{self.lag_column}_lag{k}" for k in self.lags) if self.lag_column else ()

    @property
    def retyped(self) -> tuple[str, ...]:
        return self.nominal + ((self.series_by,) if self.series_by else ())


def _lane_columns(raw: Mapping[str, Any], names: set[str],
                  by_role: Mapping[str, list[str]]) -> tuple[str | None, str | None, str | None]:
    """(series_by, order_by, graph_id) the lane declares, each checked against the schema: the
    target is a label column, the keys are columns, a window's order_by is a time column."""
    label = (raw.get("target") or {}).get("column")
    if label is not None and label not in by_role.get("label", []):
        raise ValueError(f"lane {raw.get('id')}: target column {label!r} is not a label column "
                         f"of the schema (labels: {by_role.get('label', [])})")
    window, graph = raw.get("window") or {}, raw.get("graph") or {}
    is_window = raw["shape"] == "window"
    series_by = window.get("series_by") if is_window else None
    order_by = window.get("order_by") if is_window else None
    graph_id = graph.get("graph_id") if raw["shape"] == "graph" else None
    for name, what in ((series_by, "series_by"), (order_by, "order_by"), (graph_id, "graph_id")):
        if name is not None and name not in names:
            raise ValueError(f"lane {raw.get('id')}: {what} {name!r} is not a column of the schema")
    if is_window and order_by not in by_role.get("time", []):
        raise ValueError(f"lane {raw.get('id')}: a window lane's order_by must name a time "
                         f"column of the schema (the time split reads it), got {order_by!r}")
    return series_by, order_by, graph_id


def _ordinal(nominal: Sequence[str], raw: Mapping[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(the nominals encoded ordinal, the nominals the lane's recipe_options.encode names): the
    lane's option first, else ordinal when the network embeds the column (an embedding block
    names it, or a feature_tokenizer reads every nominal)."""
    options = (raw.get("recipe_options") or {}).get("encode") or {}
    encode_opts = {c: ENCODE_NAMES[str(v).lower()] for c, v in options.items()}
    blocks = raw.get("blocks") or []
    embedded, tokenized = set(embedded_columns(blocks)), _has_block(blocks, "feature_tokenizer")
    ordinal = tuple(c for c in nominal if encode_opts.get(c) == "ordinal"
                    or (c not in encode_opts and (c in embedded or tokenized)))
    return ordinal, tuple(c for c in nominal if c in encode_opts)
