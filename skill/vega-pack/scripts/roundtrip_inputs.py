"""
roundtrip_inputs — what the pack-time round trip reads, as values (support of `roundtrip_plan`
and `examples_pack`; build plan S10, bundle-contract §5).

    Stored.read(eval_dir, metrics) -> Stored       the stored predictions, through their
                                                   writer's reader (vega/predictions-layout@1)
    raw_rows(frame, recipe_file, sample, split, predictor) -> RawRows
    series_ranks(sample, raw, positions) -> (series code, in-series rank)
    split_of(fitted), shape_of(sample)

`Stored` is read by `vega-eval/scripts/predictions_layout.py`'s own `read` (frame, `vega.*`
metadata) and refused unless it is `vega/predictions-layout@1` with its `row_id`; `unscored.json`
rides beside it. `RawRows` is the raw file with what the bundle's emitted Recipe says about it,
read off its literal module constants without importing it (the columns it reads, whether it
lags) and, for a window, the key columns as the Recipe shapes them (the Predictor's own
`prepare`): a series and its order are read the way the evaluation read them, off the SHAPED
key values (a parsed timestamp orders like a timestamp), falling back to the raw column where
the Recipe returns none.
"""
from __future__ import annotations

import ast
import functools
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from pack_context import SKILLS, PackExit

__all__ = ["LAYOUT", "RawRows", "Stored", "id_text", "raw_rows", "recipe_facts",
           "series_keys", "series_ranks", "shape_of", "split_of"]

LAYOUT = "vega/predictions-layout@1"
LAYOUT_READER = SKILLS / "vega-eval" / "scripts" / "predictions_layout.py"

# --------------------------------------------------------------------------- the values

@dataclass(frozen=True, eq=False)
class Stored:
    """What /vega-eval stored for the test part: the predictions frame, its `vega.*` metadata
    and `unscored.json` (None when the evaluation wrote none)."""

    frame: Any
    meta: Mapping
    unscored: Mapping | None = None
    source: str = "predictions.test.parquet"

    @staticmethod
    def read(eval_dir, metrics) -> "Stored":
        """The stored predictions through their writer's reader, or `PackExit(3)` when they
        are in no layout this pack reads."""
        eval_dir = Path(eval_dir)
        frame, meta = _layout_reader().read(eval_dir / "predictions.test.parquet")
        layout = meta.get("vega.predictions_layout") or (metrics or {}).get("predictions_layout")
        if layout != LAYOUT or "row_id" not in frame.columns:
            raise PackExit(3, f"the evaluation stored its predictions as {layout or 'no named '
                              'layout'}; the round trip reads {LAYOUT} (row_id, the raw head, "
                              "row_score, prediction) — re-evaluate with vega-eval v2.3")
        path = eval_dir / "unscored.json"
        unscored = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
        return Stored(frame, dict(meta), unscored)

    @property
    def row_ids(self):
        return self.frame["row_id"].to_numpy()

    def __len__(self):
        return len(self.frame)


@dataclass(frozen=True, eq=False)
class RawRows:
    """The raw file the recipe consumed, with what the bundle's Recipe says about it: `keys`,
    the shaped `series_by` / `order_by` values indexed by raw position (None: read the raw
    columns); `reads`, every raw column the Recipe reads for a row (its REQUIRED_COLUMNS, a
    lag's HISTORY_COLUMNS, a graph's query columns and `graph_id`) — kept in a request even
    where the schema lists them as ignored; `lags`, whether the Recipe lags (and so takes a
    history)."""

    frame: Any
    keys: Any = None
    reads: tuple = ()
    lags: bool = False

    def rows(self, positions, schema: Mapping):
        """The raw rows at `positions` as a request carries them (reset index): the columns
        the schema ignores and the Recipe does not read are left out."""
        drop = set((schema or {}).get("ignored") or ()) - set(self.reads)
        frame = self.frame.iloc[list(positions)]
        return frame.drop(columns=[c for c in frame.columns if c in drop]).reset_index(drop=True)

    def key(self, column, positions):
        """A key column's values at `positions`: the shaped ones when known, else the raw."""
        import numpy as np
        if self.keys is not None and column in self.keys.columns:
            return self.keys[column].reindex(list(positions)).to_numpy()
        return np.asarray(self.frame[column].to_numpy())[list(positions)]


# --------------------------------------------------------------------------- the raw rows

def raw_rows(frame, recipe_file, sample, split, predictor=None) -> RawRows:
    """`RawRows` over the raw file: what the emitted Recipe reads (its module constants, read
    without importing it) and, for a window, the shaped key columns of the train, valid and
    test rows (the Predictor's own `prepare`, which runs the Recipe)."""
    facts = recipe_facts(recipe_file)
    graph = facts.get("GRAPH") or {}
    lags = bool(facts.get("NEEDS_HISTORY"))
    reads = [*(facts.get("REQUIRED_COLUMNS") or ()), *(graph.get("query_columns") or ()),
             *((facts.get("HISTORY_COLUMNS") or ()) if lags else ()), sample.get("graph_id")]
    keys = None
    if shape_of(sample) == "window" and predictor is not None:
        keys = _shaped_keys(predictor, frame, sample, split)
    return RawRows(frame, keys, tuple(c for c in reads if c), lags)


def _shaped_keys(predictor, frame, sample, split):
    """The window's key columns as the Recipe shapes them, indexed by raw position."""
    import numpy as np
    columns = [c for c in (sample.get("series_by"), sample.get("order_by")) if c]
    if not columns:
        return None
    positions = np.concatenate([split[p] for p in ("train", "valid", "test") if p in split])
    try:
        shaped = predictor.prepare(frame.iloc[list(positions)].reset_index(drop=True)).shaped
    except (KeyError, TypeError, ValueError) as exc:
        raise PackExit(3, f"the window's series and order cannot be read through the bundle's "
                          f"own Predictor ({type(exc).__name__}: {exc}); the round trip orders "
                          "each series the way the evaluation did, off the shaped key "
                          "columns") from exc
    have = [c for c in columns if c in shaped.columns]
    keys = shaped[have].reset_index(drop=True)
    keys.index = positions
    return keys


def recipe_facts(recipe_file) -> dict:
    """The emitted Recipe's literal module constants and its top-level function names."""
    tree = ast.parse(Path(recipe_file).read_text(encoding="utf-8"))
    facts = {"functions": frozenset(n.name for n in tree.body if isinstance(n, ast.FunctionDef))}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id.isupper():
            try:
                facts[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                continue
    return facts


def series_ranks(sample, raw, positions) -> tuple:
    """(series code, in-series rank) per row at `positions`: rows grouped by `series_by`
    (one series without it) and ordered stably by `order_by` (the given order without it) —
    the evaluation's rule 1, on the shaped key values."""
    import numpy as np
    import pandas as pd
    codes = pd.factorize(pd.Series(series_keys(sample, raw, positions), dtype=object),
                         use_na_sentinel=False)[0]
    frame = pd.DataFrame({"s": codes, "at": np.arange(len(codes))})
    order_by = sample.get("order_by")
    if order_by:
        frame["o"] = raw.key(order_by, positions)
        frame = frame.sort_values("o", kind="stable")
    rank = np.empty(len(codes), dtype=np.int64)
    rank[frame["at"].to_numpy()] = frame.groupby("s", sort=False).cumcount().to_numpy()
    return np.asarray(codes, dtype=np.int64), rank


def series_keys(sample, raw, positions) -> list:
    """Each row's series key as text (one key without `series_by`)."""
    column = sample.get("series_by")
    if not column:
        return ["" for _ in range(len(positions))]
    return [str(v) for v in raw.key(column, positions)]


def id_text(value) -> str:
    """A graph id as text, one spelling for one id whichever side read it: an integral number
    is its digits (7, 7.0 and "7" are one graph), anything else its own text."""
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def split_of(fitted) -> dict:
    """`__split_raw__` as int64 arrays, or `PackExit(3)` without the test part's positions."""
    import numpy as np
    split = (fitted or {}).get("__split_raw__") or {}
    if "test" not in split:
        raise PackExit(3, "the shaper recorded no raw index for the test part, so the raw rows "
                          "behind the evaluation's predictions cannot be recovered; re-run the "
                          "shaper with a version that records __split_raw__")
    return {part: np.asarray(v, dtype=np.int64) for part, v in split.items()}


def shape_of(sample) -> str:
    """row | window | node | graph (a graph sample's level)."""
    shape = (sample or {}).get("shape") or "row"
    if shape == "graph":
        return "graph" if (sample.get("level") or "node") == "graph" else "node"
    if shape not in ("row", "window"):
        raise PackExit(3, f"MANIFEST.sample.shape {shape!r} has no round trip")
    return shape


@functools.lru_cache(maxsize=1)
def _layout_reader():
    """vega-eval's `predictions_layout` module: the writer's own reader of the layout."""
    if not LAYOUT_READER.is_file():
        raise PackExit(2, f"vega-eval's {LAYOUT_READER.name} is not at {LAYOUT_READER}; the "
                          "stored predictions are read by their writer's reader")
    import sys
    name = "vega_eval_predictions_layout"
    spec = importlib.util.spec_from_file_location(name, LAYOUT_READER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module              # its dataclasses resolve their module by name
    spec.loader.exec_module(module)
    return module
