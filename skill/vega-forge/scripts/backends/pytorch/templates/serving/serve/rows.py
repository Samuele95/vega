"""
serve.rows — the request models, generated from the bundle's own records.

The row model is a function of `schema.json` (the raw input columns and their types, which are
required, which the predictor ignores, how the truth's raw spellings decode to the classes), of
`MANIFEST.output.truth` (a class, a number, or label names in one text split on `sep`) and of
`MANIFEST.sample` (whether a request may carry `history`; which column names a graph). Nothing
is written per column or per task: the API can never disagree with the recipe about a {{NOUN}}.

    InputColumns.of(schema)                        the input half of schema.json, read once
    TruthRule.of(output, schema)                   what the truth is, and how it is refused
    RowModels(schema, output, sample, max_rows).build() -> RowModelSet
        Row             one {{NOUN}} in the raw schema (extra columns forbidden)
        PredictRequest  {rows, history?}           history for the row and window shapes only
        MonitorRequest  {rows, history?, truth?}   one truth per row; `labels` is its alias
        OneRowRequest   {row, history?}

A code column (training saw only runs of digits) travels as a non-negative integer or its
digits; a numeric column as a number or a token JSON has no literal for (`Infinity`,
`-Infinity`, `NaN`); a boolean is never a number. A null is a missing value, the recipe's to
handle, except in a column a sample is cut by (a graph id, a window's series and order keys).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Annotated, Any, Literal, Mapping, Optional, Union

from pydantic import (AfterValidator, AliasChoices, AllowInfNan, BaseModel, ConfigDict, Field,
                      Strict, StrictFloat, StrictInt, StringConstraints, TypeAdapter,
                      create_model, model_validator)
from pydantic_core import PydanticCustomError

from .problems import frozen

__all__ = ["InputColumns", "RowModelSet", "RowModels", "TOKENS", "TruthRule"]

TOKENS = MappingProxyType({"Infinity": math.inf, "+Infinity": math.inf,
                           "-Infinity": -math.inf, "NaN": math.nan})
DIGITS = "^[0-9]+$"
TRUTH_TYPES = ("class", "number", "labels")
HISTORY_SHAPES = ("row", "window")          # the shapes whose requests may carry history
FINITE = Union[StrictInt, Annotated[float, Strict(), AllowInfNan(False)]]


def _pyname(column: str, taken: set) -> str:
    """A Python identifier for a column (the model field; the column stays its alias)."""
    base = "".join(ch if ch.isalnum() else "_" for ch in column).strip("_").lower() or "col"
    base = f"c_{base}" if base[0].isdigit() else base
    name, k = base, 1
    while name in taken:
        k += 1
        name = f"{base}_{k}"
    taken.add(name)
    return name


@dataclass(frozen=True)
class InputColumns:
    """The input half of `schema.json`: the columns a caller sends and how each is typed. Every
    entry is held frozen, so nothing handed out can change what the service validates."""
    inputs: tuple
    ignored: tuple
    required: tuple
    text: tuple
    numeric: tuple
    digit_codes: frozenset
    known: frozenset

    @staticmethod
    def of(schema: Mapping) -> "InputColumns":
        inputs = frozen(list(schema.get("input") or ()))
        text = tuple(c["name"] for c in inputs if c.get("dtype") == "string")
        return InputColumns(
            inputs=inputs, ignored=tuple(schema.get("ignored") or ()),
            required=tuple(c["name"] for c in inputs if c.get("required")),
            text=text, numeric=tuple(c["name"] for c in inputs if c["name"] not in text),
            digit_codes=frozenset(c["name"] for c in inputs if c["name"] in text
                                  and c.get("pattern") == DIGITS),
            known=frozenset({c["name"] for c in inputs} | set(schema.get("ignored") or ())))


@dataclass(frozen=True)
class TruthRule:
    """What the truth holds (`MANIFEST.output.truth`): a class — decoded through the pairs the
    recipe produced on the train part, one of `classes` —, a finite number, or label names out
    of `classes` (one text split on `sep` in a row or a table). `type` is None without truth."""
    column: Optional[str]
    type: Optional[str]
    sep: Optional[str]
    classes: tuple
    decode: Mapping

    @staticmethod
    def of(output: Mapping, schema: Mapping) -> "TruthRule":
        truth = output.get("truth") or schema.get("truth") or {}
        kind = truth.get("type")
        if kind is not None and kind not in TRUTH_TYPES:
            raise ValueError(f"MANIFEST.output.truth.type {kind!r} is not one of {TRUTH_TYPES}")
        if kind == "labels" and not truth.get("sep"):
            raise ValueError("MANIFEST.output.truth is labels without a sep: names cannot split")
        return TruthRule(column=truth.get("column"), type=kind, sep=truth.get("sep"),
                         classes=tuple(str(c) for c in output.get("classes") or ()),
                         decode=MappingProxyType(dict((schema.get("label") or {})
                                                      .get("decode") or {})))

    @property
    def refusal(self) -> str:
        """What a refused truth value is told, the same on every path."""
        if self.type == "number":
            return "expects a number, or null"
        if self.type == "labels":
            return (f"expects label names out of the {len(self.classes)} labels: a list, or one "
                    f"text split on {self.sep!r}")
        return f"not one of the {len(self.classes)} classes"

    def one_class(self, value) -> str:
        """A class truth: a name or a raw spelling the recipe decodes (an integer is its digits)."""
        text = str(value)
        decoded = self.decode.get(text, text)
        if self.classes and decoded not in self.classes:
            raise ValueError(self.refusal)
        return decoded

    def label_names(self, value) -> list:
        """A labels truth: a list of names, or one text of names split on `sep` ("" is none)."""
        names = value if isinstance(value, list) else \
            [n.strip() for n in str(value).split(self.sep) if n.strip()]
        if any(str(n) not in self.classes for n in names):
            raise ValueError(self.refusal)
        return [str(n) for n in names]

    def item_type(self) -> Any:
        """The type of one entry of a request's `truth` list."""
        if self.type == "number":
            return Optional[FINITE]
        if self.type == "labels":
            return Annotated[list[str], AfterValidator(self.label_names)]
        return Annotated[Union[StrictInt, str], AfterValidator(self.one_class)]

    def cell_type(self) -> Any:
        """The type of the truth column inside a row (null: not known for that row)."""
        if self.type == "labels":
            return Optional[Annotated[Union[list[str], str], AfterValidator(self.label_names)]]
        return Optional[self.item_type()]


@dataclass(frozen=True)
class RowModelSet:
    """The request models of one bundle, and the adapter a table's rows are validated by."""
    Row: type
    PredictRequest: type
    MonitorRequest: type
    OneRowRequest: type
    rows: TypeAdapter
    truth: TruthRule
    columns: InputColumns


class RowModels:
    """Builds the `RowModelSet` of one bundle from its schema, output record and sample record."""

    def __init__(self, schema: Mapping, output: Mapping, sample: Mapping, max_rows: int):
        sample = sample or {}
        self.columns = InputColumns.of(schema)
        self.truth = TruthRule.of(output, schema)
        self.history = sample.get("shape", "row") in HISTORY_SHAPES
        keys = (("graph_id",) if sample.get("level") == "graph" else ()) + \
            (("series_by", "order_by") if sample.get("shape") == "window" else ())  # never null
        self.never_null = frozenset(c for c in map(sample.get, keys) if c)
        self.max_rows = int(max_rows)

    def build(self) -> RowModelSet:
        row = self._row_model()
        rows = (list[row], Field(..., min_length=1, max_length=self.max_rows,
                                 description=f"1 to {self.max_rows} {{NOUNS}} in the raw schema"))
        more = {} if not self.history else {"history": (Optional[list[row]], Field(
            None, max_length=self.max_rows, description="the {{NOUNS}} preceding the rows, "
            "oldest first: a window's context, a lag recipe's history"))}
        truth = {} if self.truth.type is None else {"truth": (
            Optional[list[self.truth.item_type()]],
            Field(None, validation_alias=AliasChoices("truth", "labels"),
                  description="the truth of each row, one per row, for the checks that read "
                              "it; `labels` is its alias"))}
        return RowModelSet(
            Row=row, PredictRequest=_request("PredictRequest", rows=rows, **more),
            MonitorRequest=_request("MonitorRequest", rows=rows, **more, **truth),
            OneRowRequest=_request("OneRowRequest", row=(row, Field(
                ..., description="one {{NOUN}} in the raw schema")), **more),
            rows=TypeAdapter(list[row]), truth=self.truth, columns=self.columns)

    def _row_model(self) -> type:
        taken, fields = set(), {}
        for c in self.columns.inputs:
            fields[_pyname(c["name"], taken)] = _input_field(c, self.columns,
                                                             c["name"] not in self.never_null)
        inputs = {c["name"] for c in self.columns.inputs}
        for col in dict.fromkeys(c for c in (*self.columns.ignored, self.truth.column)
                                 if c and c not in inputs):
            fields[_pyname(col, taken)] = self._ignored_field(col)
        return create_model("{{ROW_MODEL}}", __config__=ConfigDict(extra="forbid",
                                                                  populate_by_name=False),
                            **fields)

    def _ignored_field(self, column: str) -> tuple:
        """The truth, when sent, is held to the CSV path's and the monitor's rule; any other
        column the model does not read is accepted and ignored."""
        if column == self.truth.column and self.truth.type is not None:
            return self.truth.cell_type(), Field(None, alias=column, description="the truth")
        return (Optional[Union[StrictInt, StrictFloat, str]],
                Field(None, alias=column, description="accepted and ignored by the model"))


def _request(name: str, **fields) -> type:
    """A request model: its fields and nothing else, and at most one truth per row."""
    return create_model(name, __config__=ConfigDict(extra="forbid"),
                        __validators__={"one_truth_per_row":
                                        model_validator(mode="after")(_one_truth_per_row)},
                        **fields)


def _input_field(column: Mapping, columns: InputColumns, nullable: bool) -> tuple:
    """(type, Field) of one input column."""
    name = column["name"]
    if name in columns.digit_codes:
        # training saw only runs of digits: '80.0' or '-7' is refused, not a category unfitted
        kind = Union[Annotated[StrictInt, Field(ge=0)],
                     Annotated[str, StringConstraints(pattern=DIGITS)]]
        desc = "a code: a non-negative integer, or its decimal digits as text"
    elif name in columns.text:
        kind, desc = str, "a code, as text"
    else:
        kind = Union[StrictInt, StrictFloat, Literal[tuple(TOKENS)]]
        seen = column.get("range_seen")
        desc = ("a number, or one of the tokens Infinity / -Infinity / NaN"
                + (f"; seen in training: [{seen[0]}, {seen[1]}]" if seen else ""))
    desc += "; null if missing" if nullable else "; never null"
    desc += f" (role: {column['role']})" if column.get("role") else ""
    default = ... if column.get("required") else None
    return (Optional[kind] if nullable else kind), Field(default, alias=name, description=desc)


def _one_truth_per_row(request: BaseModel) -> BaseModel:
    truth = getattr(request, "truth", None)
    if truth is not None and len(truth) != len(request.rows):
        raise PydanticCustomError(
            "truth_count", "{truths} truths for {rows} rows; give one per row, or none",
            {"truths": len(truth), "rows": len(request.rows)})
    return request
