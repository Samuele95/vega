"""
A stand-in generator: one derived from a lane's own declaration, for the lanes whose generator
has not landed yet (T05–T07 write them in W5). compose_check.py builds such a lane's fixture on
it (BUILD-PLAN T02's W1 check).

    gen = StandIn(lane)                   # keeps the lanekit.datasets protocol
    frame = gen.make(seed, rows)          # SCHEMA, RECIPE (lanekit.recipes.standard_recipe), VOCAB

The columns are the ones the lane names — its target and classes, the columns its blocks embed
or its recipe encodes, its window's series and time columns, its graph's keys and graph id —
plus five numeric columns the label reads. The stand-in proves the composer and the fixture on
the lane's shape, task and blocks; the lane's real data arrives with its generator, and
compose_check.py switches to it by file (lanekit.datasets.GENERATORS).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .datasets import ColumnSpec
from .recipes import embedded_columns, standard_recipe
from .registry import Lane

__all__ = ["StandIn"]


def _label_dtype(raw: Mapping[str, Any]) -> str:
    classes = (raw.get("target") or {}).get("classes")
    if not classes:
        return "float"
    return "int" if all(str(c).lstrip("-").isdigit() for c in classes) else "str"


def _structure_columns(raw: Mapping[str, Any]) -> list:
    """The columns that are no feature: a row key, a window's time and series, a graph's keys
    and its graph id."""
    window, graph = raw.get("window") or {}, raw.get("graph") or {}
    cols = [] if raw["shape"] == "graph" else [ColumnSpec("row_key", "str", "key", comment="a row's identifier")]
    if raw["shape"] == "window":
        cols.append(ColumnSpec(window["order_by"], "datetime", "time", comment="when the row was recorded"))
        if window.get("series_by"):
            cols.append(ColumnSpec(window["series_by"], "str", "series", comment="which series the row belongs to"))
    if raw["shape"] == "graph":
        if graph["method"] == "key_pair":
            cols += [ColumnSpec(c, "str", "key", comment="a key the graph relates rows by") for c in graph["columns"]]
        else:
            cols.append(ColumnSpec("node_key", "str", "key", comment="a node's identifier"))
        if graph.get("graph_id"):
            cols.append(ColumnSpec(graph["graph_id"], "str", "group", comment="the graph a node belongs to"))
    return cols


def _feature_columns(raw: Mapping[str, Any]) -> list:
    """Five numeric columns (a knn graph's own among them), each nominal the lane embeds or
    encodes, two more nominals for a feature tokenizer; then the label and, on a row lane, a
    column known only after the fact."""
    graph, target = raw.get("graph") or {}, raw.get("target") or {}
    knn = list(graph.get("columns") or []) if graph.get("method") == "knn" else []
    cols = [ColumnSpec(c, "float", "numeric", "unit", "a measured quantity")
            for c in knn + [f"x{i}" for i in range(1, 6 - len(knn))]]
    nominal = list(embedded_columns(raw.get("blocks") or [])) + list((raw.get("recipe_options") or {}).get("encode") or {})
    nominal += ["n1", "n2"] if raw["family"] == "ft-transformer" else []
    cols += [ColumnSpec(c, "str", "nominal", comment="a category") for c in dict.fromkeys(nominal)]
    if target.get("column"):
        cols.append(ColumnSpec(target["column"], _label_dtype(raw), "label", comment="the label"))
        if raw["shape"] == "row":
            cols.append(ColumnSpec("known_after", "float", "drop", comment="known only after the fact"))
    return cols


class StandIn:
    """A generator derived from a lane's declaration (the protocol of lanekit.datasets): the
    columns the lane's target, blocks, window and graph name, plus five numeric columns."""

    def __init__(self, lane: Lane) -> None:
        self.lane = lane
        self.SCHEMA = tuple(_structure_columns(lane.raw) + _feature_columns(lane.raw))
        self.VOCAB = frozenset()
        self.ENTITY_COMMENT = f"One {lane.raw['entity']} of the stand-in data for lane {lane.id}."

    def RECIPE(self, raw_path: Path, shaped_dir: Path, lane: Any) -> dict:  # noqa: N802 (the protocol's name)
        return standard_recipe(self.SCHEMA, raw_path, shaped_dir, lane)

    def make(self, seed: int, rows: int):
        """`rows` rows (graphs, at graph level), deterministic in (seed, rows): noisy numeric
        columns, a label that reads x1, a window's series in time order, a graph's keys."""
        import numpy as np
        import pandas as pd

        raw = self.lane.raw
        rng = np.random.default_rng(seed)
        groups, n = self._groups(rng, rows)
        frame = pd.DataFrame({c.name: np.round(rng.normal(0, 1, n) * 3 + 10, 4)
                              for c in self.SCHEMA if c.role == "numeric"})
        signal = frame["x1"].to_numpy() - 10 if "x1" in frame else rng.normal(0, 1, n)
        if raw["shape"] == "window":
            frame, signal, n = self._window(frame, signal, n)
        if raw["shape"] == "graph":
            self._graph_columns(frame, groups, rng, n)
        for col in self.SCHEMA:
            if col.role == "key" and col.name not in frame:
                frame[col.name] = [f"K{i:06d}" for i in range(n)]
            elif col.role == "nominal":
                frame[col.name] = rng.choice(["a", "b", "c", "d"], n)
        label = (raw.get("target") or {}).get("column")
        if label:
            frame[label] = self._labels(signal, groups, rng)
            if "known_after" in {c.name for c in self.SCHEMA}:
                frame["known_after"] = np.round(signal * 2 + rng.normal(0, 0.1, n), 4)
        return frame[[c.name for c in self.SCHEMA]]

    def _groups(self, rng, rows: int):
        """(the graph each node belongs to, the node count): 8-16 nodes a graph at graph level,
        else (None, rows)."""
        import numpy as np

        if self.lane.raw["shape"] != "graph" or not (self.lane.raw.get("graph") or {}).get("graph_id"):
            return None, rows
        sizes = rng.integers(8, 17, size=rows)
        return np.repeat(np.arange(rows), sizes), int(sizes.sum())

    def _window(self, frame, signal, n: int):
        """The rows cut to whole series (three when the lane names a series key), each series
        one hourly run of timestamps."""
        import numpy as np
        import pandas as pd

        window = self.lane.raw["window"]
        series = window.get("series_by")
        k = 3 if series else 1
        per = n // k
        n = per * k
        frame = frame.iloc[:n].reset_index(drop=True)
        stamps = pd.date_range("2026-01-01", periods=per, freq="h")
        frame[window["order_by"]] = np.tile(stamps.astype(str), k)
        if series:
            frame[series] = np.repeat([f"s{i}" for i in range(k)], per)
        return frame, signal[:n], n

    def _graph_columns(self, frame, groups, rng, n: int) -> None:
        """A key_pair graph's two key columns (each node linked to a recent one of its own
        graph) and its graph id."""
        graph = self.lane.raw["graph"]
        keys = [f"K{i:06d}" for i in range(n)]
        if graph["method"] != "key_pair":
            return
        src, dst = graph["columns"]
        frame[src] = keys
        parent, start = [], 0
        for i in range(n):
            if groups is not None and i and groups[i] != groups[i - 1]:
                start = i                        # a new graph: its first node has no parent
            low = max(start, i - 8)
            linked = i > low and (groups is not None or i % 5)
            parent.append(keys[int(rng.integers(low, i))] if linked else "")
        frame[dst] = parent
        if graph.get("graph_id"):
            frame[graph["graph_id"]] = [f"G{int(g):04d}" for g in groups]

    def _labels(self, signal, groups, rng):
        import numpy as np

        target = self.lane.raw["target"]
        classes = target.get("classes")
        base = signal + rng.normal(0, 0.5, len(signal))
        if groups is not None:
            means = {g: base[groups == g].mean() for g in np.unique(groups)}
            base = np.array([means[g] for g in groups])
        if not classes:
            return np.round(base * 4 + 20, 4)
        if target.get("label_format") == "delimited":
            sep = target.get("sep") or "|"
            picks = [[c for j, c in enumerate(classes) if (i + j) % 3 == 0 or j == i % len(classes)]
                     for i in range(len(base))]
            return [sep.join(p) for p in picks]
        if len(classes) == 2 and target.get("positive") is not None and str(classes[1]) == str(target.get("positive")):
            cut = np.quantile(base, 0.85 if self.lane.raw["task"] == "anomaly" else 0.6)
            chosen = np.where(base > cut, 1, 0)
        else:
            edges = np.quantile(base, np.linspace(0, 1, len(classes) + 1)[1:-1])
            chosen = np.searchsorted(edges, base)
        values = [int(c) if _label_dtype(self.lane.raw) == "int" else str(c) for c in classes]
        return [values[i] for i in chosen]
