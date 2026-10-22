"""
{{PACKAGE}}.{{MODULE_SERVING}}.guards — what `schema.json` says about raw rows before they are
answered (support; used by the Predictor). Emitted by /vega-pack from
`backends/pytorch/templates/serving/package/guards.py`: fix the template, never this copy.

`SchemaGuard(schema).warnings(df_raw)` gives one warning list per row: a required column that is
missing, a text column whose values arrived as floats, a value outside the range training saw, a
non-finite value in a column that never had one. None of them is an error — a model asked about
a row unlike its training data still answers — but the answer is extrapolation, and the caller
is told so. The texts are the API contract's (data, not chrome): they do not change.

Two rules the checks keep. A code column arriving as float is flagged per row, because 80.0 is
not the category 80 to the recipe (a CSV read without the schema's dtypes, or one float in a
list of rows). The range check reads finite values only, and a non-finite value is news only
when the schema says the training part never carried one there (`non_finite_seen`): an
infinity can be a source's token for an undefined value, not a large number.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

__all__ = ["SchemaGuard"]


class SchemaGuard:
    """The schema's per-row warnings over the input columns it lists (`schema.json` `input`)."""

    def __init__(self, schema: Mapping):
        self._columns = tuple(dict(column) for column in (schema or {}).get("input") or ())

    def warnings(self, df_raw: pd.DataFrame) -> list:
        """`list[list[str]]`: one list per row of `df_raw`, in its order."""
        out = [[] for _ in range(len(df_raw))]
        for column in self._columns:
            if column.get("name") not in df_raw.columns:
                self._missing(column, df_raw, out)
            elif column.get("dtype") == "string":
                self._text_arrived_as_float(column, df_raw, out)
            else:
                self._outside_range(column, df_raw, out)
        return out

    @staticmethod
    def _missing(column, df_raw, out) -> None:
        if column.get("required"):
            for notes in out:
                notes.append(f"required column {column.get('name')!r} is missing")

    @staticmethod
    def _text_arrived_as_float(column, df_raw, out) -> None:
        name = column["name"]
        series = df_raw[name]
        if pd.api.types.is_float_dtype(series):
            flags = series.notna().to_numpy()
        elif series.dtype == object:
            flags = series.map(lambda v: isinstance(v, float) and not pd.isna(v)) \
                .to_numpy(dtype=bool)
        else:
            return
        for i in np.flatnonzero(flags):
            out[i].append(f"{name} arrived as float; the schema declares it text (read the file "
                          "with the schema's dtypes)")

    @staticmethod
    def _outside_range(column, df_raw, out) -> None:
        seen = column.get("range_seen")
        if not seen or len(seen) != 2:
            return
        name = column["name"]
        values = pd.to_numeric(df_raw[name], errors="coerce")
        finite = np.isfinite(values.to_numpy(dtype="float64", na_value=np.nan))
        outside = ((values < seen[0]) | (values > seen[1])).to_numpy(dtype=bool, na_value=False) \
            & finite
        news = ~finite & values.notna().to_numpy() & (not column.get("non_finite_seen"))
        for i in np.flatnonzero(outside | news):
            out[i].append(f"{name} outside the range seen in training [{seen[0]}, {seen[1]}]"
                          if outside[i] else
                          f"{name} is non-finite, which training never showed in this column")
