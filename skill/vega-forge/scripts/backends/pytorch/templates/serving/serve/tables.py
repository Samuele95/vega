"""
serve.tables — rows to the raw frame the recipe reads, and frames back to JSON-safe rows.

Every way into the service goes through the same row model. A JSON body is validated by it
before the route runs; a CSV upload is read with the schema's types and then validated by it
too (`TableCodec.accept`), so a malformed cell in a file is refused with the column named
exactly as in a JSON body — never answered with a 500 or a silent coercion. The truth column of
an upload is held to its type on the same terms (`TruthRule`): a class, a number, or label names
split on `sep`.

The one conversion this module owns is from JSON to the raw row the recipe expects, and it is a
conversion of *encoding*, not of meaning: JSON has no literal for the tokens `Infinity` and
`NaN`, so they travel as strings and become floats here; a code column travels as an integer or
its digits and becomes the text the recipe was fitted on. No value is scaled, clipped or imputed
— that is the recipe's work, done once, by the recipe. The truth and the ignored columns never
reach the recipe (an autoregressive target, listed as an input, does: it is an input).
"""
from __future__ import annotations

import math
from typing import Mapping, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from .problems import ContractError, ProblemErrors
from .rows import TOKENS, InputColumns, RowModelSet, TruthRule

__all__ = ["TableCodec"]

REFUSED = "the table does not honour the input schema"
FIRST_ROWS = 5


class TableCodec:
    """JSON rows and CSV tables to the raw frame, and frames to JSON-safe records."""

    def __init__(self, columns: InputColumns, models: RowModelSet, problems: ProblemErrors,
                 max_rows: int):
        self.columns = columns
        self.truth: TruthRule = models.truth
        self._rows = models.rows
        self._problems = problems
        self.max_rows = int(max_rows)
        # what the recipe never reads: the ignored columns and the truth, unless the schema lists
        # one as an input — an autoregressive target is an input channel the rows carry, not the
        # truth of their answers (which forecast later steps), so a table never returns it as one
        inputs = {c["name"] for c in columns.inputs}
        dropped = list(columns.ignored) + ([self.truth.column] if self.truth.column else [])
        self._dropped = tuple(dict.fromkeys(c for c in dropped if c not in inputs))
        self._truth_column = None if self.truth.column in inputs else self.truth.column

    # -- JSON rows -> the raw frame --------------------------------------------------
    def frame(self, rows) -> pd.DataFrame:
        """Validated rows (models, or dicts keyed by column name) as a raw DataFrame: numeric
        tokens decoded, text columns as text, the truth and ignored columns dropped."""
        records = [r.model_dump(by_alias=True) if isinstance(r, BaseModel) else dict(r)
                   for r in rows]
        df = pd.DataFrame.from_records(records)
        df = df.drop(columns=[c for c in self._dropped if c in df.columns])
        for col in self.columns.numeric:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].map(lambda v: TOKENS.get(v, v)
                                                    if isinstance(v, str) else v),
                                        errors="coerce").astype("float64")
        for col in self.columns.text:
            if col in df.columns:
                df[col] = df[col].map(_code_text).astype("string")
        return df

    # -- a table (a CSV upload) -> (the raw frame, the truth) ------------------------
    def accept(self, df: pd.DataFrame) -> tuple:
        """A table read with the schema's types, held to the same contract as a JSON body: the
        same row cap, the same names, every cell through the same row model, the truth to its
        type. Returns (the raw frame, the truth per row or None when the table has none)."""
        self._refuse_layout(df)
        truth = self._truth(df)
        inputs = self._numbers_back(df.drop(columns=[c for c in self._dropped
                                                     if c in df.columns]))
        try:
            self._rows.validate_python(self.column_records(inputs))
        except ValidationError as exc:
            raise ContractError(REFUSED, self._problems.group(exc.errors(),
                                                              rows_field=None)) from None
        # the rows passed the one row model; the frame the recipe reads is derived from the
        # table column-wise — the frame `frame(records)` would build, without a second pass
        return self._table_frame(inputs), truth

    def _refuse_layout(self, df: pd.DataFrame) -> None:
        errors = []
        if not len(df):
            errors.append({"column": None, "error": "no rows"})
        if len(df) > self.max_rows:
            errors.append({"column": None, "error": f"more than {self.max_rows} rows; at most "
                           f"{self.max_rows} per request"})
        known = self.columns.known | ({self.truth.column} if self.truth.column else set())
        errors += [{"column": c, "error": "required column missing"}
                   for c in self.columns.required if c not in df.columns]
        errors += [{"column": c, "error": "unknown column"} for c in df.columns if c not in known]
        if errors:
            raise ContractError(REFUSED, errors)

    def _truth(self, df: pd.DataFrame) -> Optional[list]:
        """The truth column of the table held to its type, or None when the table has none."""
        rule = self.truth
        if rule.type is None or not self._truth_column or rule.column not in df.columns:
            return None
        values, bad = [], []
        for i, cell in enumerate(df[rule.column].tolist()):
            try:
                values.append(_truth_cell(rule, cell))
            except ValueError:
                bad.append(i)
                values.append(None)
        if bad:
            raise ContractError(REFUSED, [{"column": rule.column, "error": rule.refusal,
                                           "rows": bad[:FIRST_ROWS], "count": len(bad)}])
        return values

    def _numbers_back(self, inputs: pd.DataFrame) -> pd.DataFrame:
        """One unparseable cell turns a whole CSV column into text; the cells that are numbers
        are numbers again, so only the ones that are not are named."""
        out = inputs.copy()
        for col in self.columns.numeric:
            if col in out.columns and not pd.api.types.is_numeric_dtype(out[col]):
                text = out[col]
                num = pd.to_numeric(text, errors="coerce")
                bad = num.isna() & text.notna() & ~text.astype(str).isin(list(TOKENS))
                out[col] = num.astype(object).where(~bad, text)
        return out

    def _table_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """The raw frame `frame()` would build from these validated rows, column-wise."""
        out = df.copy()
        for col in self.columns.numeric:
            if col in out.columns:
                series = out[col]
                if not pd.api.types.is_numeric_dtype(series) or \
                        pd.api.types.is_bool_dtype(series):
                    series = series.map(lambda v: TOKENS.get(v, v) if isinstance(v, str) else v)
                out[col] = pd.to_numeric(series, errors="coerce").astype("float64")
        for col in self.columns.text:
            if col in out.columns:
                series = out[col]
                out[col] = (series.astype("string") if pd.api.types.is_string_dtype(series)
                            and series.dtype != object else
                            series.map(_code_text).astype("string"))
        return out

    # -- frames -> JSON-safe records -------------------------------------------------
    @staticmethod
    def column_records(df: pd.DataFrame) -> list:
        """Rows as JSON-safe records (infinities as the tokens, missing as null), converted one
        column array at a time."""
        columns = [(name, _json_column(df[name])) for name in df.columns]
        return [{name: values[i] for name, values in columns} for i in range(len(df))]

    @staticmethod
    def json_rows(df: pd.DataFrame) -> list:
        """Rows as JSON-safe records: what a caller would send for these rows."""
        return TableCodec.column_records(df)


def _missing(cell) -> bool:
    return cell is None or cell is pd.NA or (isinstance(cell, (float, np.floating))
                                             and math.isnan(cell))


def _code_text(v) -> Optional[str]:
    """A code cell as the text the recipe was fitted on: an integral number is its digits."""
    if _missing(v):
        return None
    if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool) \
            and float(v).is_integer():
        return str(int(v))
    return str(v)


def _truth_cell(rule: TruthRule, cell):
    """One truth cell of a table, held to the rule's type (ValueError when it is not)."""
    if rule.type == "labels":
        return rule.label_names("" if _missing(cell) else str(cell))
    if rule.type != "number":
        if _missing(cell):
            raise ValueError(rule.refusal)
        return rule.one_class(cell if isinstance(cell, str) else _code_text(cell))
    if _missing(cell) or (isinstance(cell, str) and not cell.strip()):
        return None
    if isinstance(cell, (bool, np.bool_)):
        raise ValueError(rule.refusal)
    value = float(cell)                 # ValueError for text that is not a number
    if not math.isfinite(value):
        raise ValueError(rule.refusal)
    return int(cell) if isinstance(cell, (int, np.integer)) else value


def _json_column(series: pd.Series) -> np.ndarray:
    """One column as JSON-safe Python values."""
    if pd.api.types.is_bool_dtype(series):
        return series.to_numpy(dtype=object)
    if pd.api.types.is_float_dtype(series):
        raw = series.to_numpy(dtype="float64", na_value=np.nan)
        values = raw.astype(object)
        values[np.isnan(raw)] = None
        values[np.isposinf(raw)] = "Infinity"
        values[np.isneginf(raw)] = "-Infinity"
        return values
    if pd.api.types.is_integer_dtype(series) and not series.isna().any():
        return series.to_numpy().astype(object)            # numpy ints -> Python ints
    # text, or a numeric column the coercion left mixed: per cell, but only here
    return np.array([_json_cell(v) for v in series.to_numpy(dtype=object)], dtype=object)


def _json_cell(v):
    """One cell as JSON: infinities as the tokens, missing as null."""
    if isinstance(v, (float, np.floating)):
        if math.isnan(v):
            return None
        if math.isinf(v):
            return "Infinity" if v > 0 else "-Infinity"
        return float(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if v is pd.NA or v is None:
        return None
    if isinstance(v, Mapping):
        return dict(v)
    return v
