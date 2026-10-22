"""
serve.problems — how a refusal is said: RFC 9457 bodies and the request errors, grouped.

A request the contract refuses is refused whole, with every offending column named exactly as
the caller wrote it — never by the branch of a type union the validator tried last, and never by
the Python name a column was given inside the row model. `ProblemErrors` turns the validator's
error list into that answer: one entry per (column, reason), with the first few rows that broke
it and how many did; a `history` row at fault is named with `field: "history"`.

`ProblemDetail` (an RFC 9457 body) and `ContractError` (a refusal raised below a route, rendered
as a 422 by the service) cross the boundary from here. `frozen` and `thawed` are the deep
read-only copy and its plain-JSON inverse every serve module uses for the records it holds.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Optional

__all__ = ["ContractError", "ProblemDetail", "ProblemErrors", "frozen", "thawed"]

ONE_ROW_FIELDS = ("row",)                   # the request fields that carry one row, not a list
TRUTH_FIELDS = ("truth", "labels")          # the truth, under its name and its permanent alias
FIRST_ROWS = 5
CODE = "expects a code: a non-negative integer, or its decimal digits as text"
NUMBER = "expects a number, null, or one of the tokens Infinity, -Infinity, NaN"


def frozen(value):
    """A deep, read-only copy: mappings become MappingProxyType, lists and tuples tuples."""
    if isinstance(value, Mapping):
        return MappingProxyType({k: frozen(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(frozen(v) for v in value)
    return value


def thawed(value):
    """The plain-JSON copy of a frozen value, for a caller who may change it."""
    if isinstance(value, Mapping):
        return {k: thawed(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [thawed(v) for v in value]
    return value


class ContractError(ValueError):
    """A request that does not honour the input schema: refused whole, every offending column
    named, rather than half-answered. `errors` is held frozen; `thawed(errors)` is its JSON."""

    def __init__(self, detail: str, errors):
        super().__init__(detail)
        self.detail = detail
        self.errors = frozen(list(errors))


@dataclass(frozen=True)
class ProblemDetail:
    """An RFC 9457 problem body. `errors` names the offending columns, when there are any."""
    status: int
    title: str
    detail: str
    errors: tuple = ()
    type: str = "about:blank"

    def __post_init__(self):
        object.__setattr__(self, "errors", frozen(list(self.errors)))

    def body(self) -> dict:
        out = {"type": self.type, "title": self.title, "status": self.status,
               "detail": self.detail}
        if self.errors:
            out["errors"] = thawed(self.errors)
        return out


@dataclass(frozen=True)
class _Where:
    """Where one validator error points: the request field, the row and the column."""
    field: Optional[str]
    row: Optional[int]
    column: Optional[str]


class ProblemErrors:
    """The validator's errors grouped by column, each with what the column expects: the truth
    (`truth_column`, `truth_refusal`, per `MANIFEST.output.truth`; a truth that is also an input
    is refused as that input) and the inputs `columns` names (`numeric`, `digit_codes`)."""

    def __init__(self, truth_column: Optional[str], truth_refusal: str, columns):
        self.truth_column = truth_column
        self.truth_refusal = truth_refusal
        self.digit_codes = frozenset(columns.digit_codes)
        self.numeric_columns = frozenset(columns.numeric)
        inputs = self.numeric_columns | frozenset(columns.text)
        self.row_truth = None if truth_column in inputs else truth_column

    def group(self, errors, rows_field: Optional[str] = "rows") -> list:
        """`[{column, error, rows, count[, field]}]` per (field, column, reason); `field` names
        rows at fault outside `rows_field` (a `history`). `None`: a bare list of rows (a table)."""
        grouped: dict = {}
        for err in errors:
            where = _locate(err, rows_field)
            column, message = self._message(err, where)
            field = where.field if where.field not in (rows_field, *ONE_ROW_FIELDS) else None
            entry = grouped.setdefault((field, column, message), {
                "column": column, "error": message, "rows": [], "count": 0})
            if field is not None and where.column is not None:
                entry["field"] = field
            if where.row is not None and where.row not in entry["rows"]:
                entry["count"] += 1
                if len(entry["rows"]) < FIRST_ROWS:
                    entry["rows"].append(where.row)
        return [_without_empty_rows(entry) for entry in grouped.values()]

    def _message(self, err, where: _Where) -> tuple:
        """(the column the entry names, what that column expects) for one error."""
        if err.get("type") == "truth_count":
            return self.truth_column, err.get("msg")
        if (where.field in TRUTH_FIELDS and where.row is not None) or \
                (where.column is not None and where.column == self.row_truth):
            return self.truth_column, self.truth_refusal
        return self._column_message(err, where)

    def _column_message(self, err, where: _Where) -> tuple:
        kind, column = err.get("type"), where.column
        if kind == "missing" and column is not None:
            return column, "required column missing"
        if kind == "extra_forbidden" and column is not None:
            return column, "unknown column"
        if kind == "extra_forbidden" and where.field is not None and where.row is None:
            return None, f"{where.field}: not a field this bundle's request takes"
        if column in self.digit_codes:
            return column, CODE
        if column in self.numeric_columns:
            return column, NUMBER
        if column is None and where.field:
            return None, f"{where.field}: {err.get('msg')}"
        return column, err.get("msg")


def _locate(err, rows_field) -> _Where:
    """The field, row and column an error's `loc` names ("body" and union branches aside)."""
    loc = [p for p in err.get("loc", ()) if p != "body"]
    field = loc[0] if loc and isinstance(loc[0], str) and rows_field else None
    rest = loc[1:] if field else loc
    if field in ONE_ROW_FIELDS:
        return _Where(field, None, rest[0] if rest and isinstance(rest[0], str) else None)
    row = rest[0] if rest and isinstance(rest[0], int) else None
    named = row is not None and len(rest) > 1 and isinstance(rest[1], str)
    return _Where(field, row, rest[1] if named and field not in TRUTH_FIELDS else None)


def _without_empty_rows(entry: dict) -> dict:
    """A request-level refusal (no row at fault) carries no `rows` and no `count`."""
    if not entry["rows"]:
        entry.pop("rows")
        entry.pop("count")
    return entry
