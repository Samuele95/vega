"""
user_slices — the slices the user names (`--slices COL=VALUE …`), declared before the seal.

A slice the user names is first-exposure only (rule T2 refuses one on a further look), so it is
declared once and must not be lost to a spelling: its literal is **coerced to the column's
dtype** — the dtype the shaper's manifest records for the shaped output
(`output_schema[col].dtype`), else the train part's own (eval-contract §2). `region=4` then
matches a column holding `4` or `4.0`, `promo=true` a boolean one, and `zip=007` a string column
as written. Everything here is decided without a test row: a value missing, a column that is not
one of the features the slices read, and a literal the dtype cannot hold are refused before the
seal (`declare`); whether the slice *has* rows is a fact about the test part, said after the
look (`evaluate.build_slices`).

Pure functions and a frozen value: no argv, no printing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

__all__ = ["UserSlice", "coerce", "declare", "dtype_family", "dtypes_of"]


def dtype_family(dtype) -> str:
    """`integer`, `float`, `bool`, `datetime` or `string` — what a dtype's spelling says, in
    pandas' (`Int64`, `float64`, `boolean`, `datetime64[ns]`, `object`) or Arrow's (`int64`,
    `double`, `bool`, `timestamp[ns]`, `large_string`) words."""
    text = str(dtype or "").strip().lower()
    if text.startswith(("int", "uint")):
        return "integer"
    if text.startswith(("float", "double", "halffloat", "decimal")):
        return "float"
    if text.startswith("bool"):
        return "bool"
    if text.startswith(("datetime", "timestamp", "date")):
        return "datetime"
    return "string"


_TRUE, _FALSE = ("true", "1", "yes"), ("false", "0", "no")


def coerce(literal: str, dtype) -> Any:
    """The literal as a value of `dtype`'s family; ValueError, naming what was expected, when the
    dtype cannot hold it (`4.5` for an integer column, `abc` for a float one)."""
    family = dtype_family(dtype)
    text = str(literal).strip()
    if family == "integer":
        number = float(text)
        if not math.isfinite(number) or not number.is_integer():
            raise ValueError(f"{literal!r} is not an integer, and the column's dtype is {dtype}")
        return int(number)
    if family == "float":
        number = float(text)
        if not math.isfinite(number):
            raise ValueError(f"{literal!r} names no finite value, and a slice of NaN or inf "
                             "matches no row")
        return number
    if family == "bool":
        if text.lower() in _TRUE:
            return True
        if text.lower() in _FALSE:
            return False
        raise ValueError(f"{literal!r} is not a boolean (true/false, 1/0, yes/no)")
    if family == "datetime":
        import pandas as pd
        return pd.Timestamp(text)
    return str(literal)


@dataclass(frozen=True)
class UserSlice:
    """One declared slice: the literal as typed, the column, and the value it was coerced to."""

    raw: str
    column: str
    literal: str
    value: Any
    dtype: str

    @property
    def name(self) -> str:
        return f"{self.column}={self.literal}"

    def mask(self, frame):
        """The rows of `frame` whose `column` equals the coerced value — compared as values, so
        an integer code read back as a float (the design matrix's 4.0) still matches 4."""
        import numpy as np
        column = frame[self.column]
        if dtype_family(self.dtype) == "string":
            hit = column.astype(str) == self.value
        else:
            hit = column == self.value
        return np.asarray(hit.fillna(False) if hasattr(hit, "fillna") else hit, dtype=bool)


def dtypes_of(manifest: Mapping | None, train_path=None) -> dict:
    """{column: dtype}: the shaper's manifest `output_schema` first (the shaped output's own
    record), then the train part's Arrow schema for a column the manifest does not type. Only
    the train part is read — its schema, not a row — and never the test part."""
    out = {}
    for column, entry in (((manifest or {}).get("output_schema")) or {}).items():
        if isinstance(entry, Mapping) and entry.get("dtype"):
            out[str(column)] = str(entry["dtype"])
    if train_path and Path(train_path).is_file():
        try:
            import pyarrow.parquet as pq
            for field in pq.read_schema(train_path):
                out.setdefault(str(field.name), str(field.type))
        except Exception:  # noqa: BLE001 — an unreadable train part leaves the manifest's types
            pass
    return out


def declare(raw_slices: Iterable[str] | None, dtypes: Mapping, features: Iterable[str]):
    """(declared slices, problems): every `--slices` item checked before the seal. A problem is a
    sentence naming the item and the fix; a declared slice carries its coerced value."""
    features = [str(f) for f in features]
    declared, problems = [], []
    for raw in raw_slices or ():
        column, sep, literal = str(raw).partition("=")
        column = column.strip()
        if not sep or not literal.strip():
            problems.append(f"--slices {raw!r} gives no value: a slice is COL=VALUE, and a bare "
                            "column names no rows")
            continue
        if column not in features:
            problems.append(f"--slices {raw!r}: {column!r} is not one of the features the "
                            f"slices read ({', '.join(features) or 'none'})")
            continue
        dtype = dtypes.get(column)
        if dtype is None:
            problems.append(f"--slices {raw!r}: the dtype of {column!r} is recorded neither in "
                            "the shaper's output_schema nor in the train part, so the literal "
                            "cannot be coerced to it")
            continue
        try:
            value = coerce(literal, dtype)
        except ValueError as exc:
            problems.append(f"--slices {raw!r}: {exc} — the literal is coerced to the column's "
                            f"dtype ({dtype}), and this one cannot be")
            continue
        declared.append(UserSlice(raw=str(raw), column=column, literal=literal.strip(),
                                  value=value, dtype=dtype))
    return tuple(declared), tuple(problems)
