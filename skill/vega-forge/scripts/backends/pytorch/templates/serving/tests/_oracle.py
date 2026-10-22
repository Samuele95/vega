"""
tests/_oracle.py — what the bundle's tests know about this bundle: its own records, read once.

Not a test module (pytest collects `test_*.py` only): the API tests import it. Every expectation
they hold an answer to is derived here from the bundle's records — MANIFEST.json (`output`, the
vega/output@2 record: the reading, its fields and their kinds, what the round trip compares and
decides, the thresholds, the truth; `sample`, the sample shape; `roundtrip.tolerance`;
`serve.identity`) and schema.json (the input columns) — never from a column, a class or a task
named in a test, so one suite holds every reading and every sample shape.

Where each field of the answer frame lands in a `RowPrediction` is the field-kind table of the
bundle contract (§7.3), restated here on purpose rather than imported from `serve/answers.py`:
the tests hold the service's codec to the contract, so they do not share its code.

    ORACLE = Oracle.of(ROOT)               the records of this bundle
    .request(examples)                     /v1/examples' shipped rows as a /v1/predict body
    .one_request()                         examples/request.json: one complete request
    .expected(), .compared_positions()     examples/expected.parquet and its `__row__`
    .value(prediction, field)              where one field of the frame is in a JSON answer
    .check(prediction, expected, tol)      [] when one answer agrees with one expected row
    .keys(include), .csv_columns(...)      the JSON keys of an answer, the CSV answer's columns
    .includes()                            the `include` values that show every gated field
    .sums_to_one()                         whether the reading's probabilities sum to one
    .truth_payload(item), .wrong_truth()   a demonstration item as a monitor body with its
                                           truth; a value the truth's type refuses
    applies(flag)                          a test collected only where it applies
    client                                 the module-scoped fixture: this bundle's service
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Mapping

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

__all__ = ["EXPECTED_KINDS", "GATED", "KEYS", "MAPPED", "ORACLE", "Oracle", "ROOT", "TAIL",
           "applies", "client", "numbers_agree", "same_value"]

# the field-kind table (bundle-contract §7.3): the JSON key each kind of field lands under
KEYS = {"class": "label", "flag": "labels", "alarm": "anomalous", "probability": "probabilities",
        "raw": "logits", "value": "value", "quantile": "quantiles", "horizon": "values",
        "score": "score", "row_score": "reconstruction_error", "cumulative": "cumulative",
        "recon": "reconstruction", "embedding": "embedding", "decision": None,
        "group": "graph_id", "answered": "answered", "warnings": "warnings"}
# the kinds shown only on request, and the `include` value that shows each
GATED = {"raw": "logits", "recon": "reconstruction", "embedding": "embedding"}
MAPPED = ("probability", "raw", "quantile", "cumulative", "recon")      # {of: number}
TAIL = ("answered", "warnings")
SUMS_TO_ONE = ("binary", "multiclass", "ordinal")
# what a preset may say the answer is, per reading (vega/presets@2 `expected.kind`)
EXPECTED_KINDS = {"binary": ("class",), "multiclass": ("class",), "ordinal": ("class",),
                  "multilabel": ("labels",), "regression": ("value",),
                  "quantile": ("value",), "forecast": ("value",), "reconstruction": (),
                  "anomaly-reconstruction": ("alarm", "class"),
                  "anomaly-svdd": ("alarm", "class")}


def _read(name: str) -> dict:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def _missing(value) -> bool:
    return value is None or value is pd.NA or (isinstance(value, float) and math.isnan(value))


def numbers_agree(got, want, tol: float) -> bool:
    """Two numbers within `tol`; missing agrees only with missing."""
    if _missing(got) or _missing(want):
        return _missing(got) and _missing(want)
    return abs(float(got) - float(want)) <= tol


def same_value(got, want) -> bool:
    """Exactly the same answer: numbers by value (1 == 1.0, True == 1), names by text."""
    if _missing(got) or _missing(want):
        return _missing(got) and _missing(want)
    try:
        return float(got) == float(want)
    except (TypeError, ValueError):
        return str(got) == str(want)


def applies(flag: bool):
    """A test decorator: the test is collected only where it applies (`flag`), and is absent
    elsewhere — never a skip, which the verification program reads as a check that could not
    run (its skip policy admits a closed list of reasons)."""
    return (lambda test: test) if flag else (lambda test: None)


def _typed_truth(truth: Mapping, value, row: Mapping):
    """One demonstration truth typed as MANIFEST.output.truth says: label names, a number (a
    forecast item's truth is the steps after its row; the row's own is its column), a class;
    null where the item's truth is not known (vega/demo@2), never the text of a null."""
    if value is None:
        return None
    if truth["type"] == "labels":
        return list(value) if isinstance(value, list) else \
            [n for n in str(value).split(truth["sep"]) if n]
    if truth["type"] == "number":
        return row.get(truth["column"]) if isinstance(value, list) else value
    return str(value)


def _read_parquet(path):
    """A parquet file through Arrow's own reader: no Python in its threads (A40)."""
    from pyarrow import fs
    return pd.read_parquet(path, filesystem=fs.LocalFileSystem())


class Oracle:
    """This bundle's records, read once: the reading, the shape, the fields, the truth."""

    def __init__(self, manifest: Mapping, schema: Mapping):
        self.manifest, self.schema = manifest, schema
        output, sample = manifest.get("output") or {}, manifest.get("sample") or {}
        self.output, self.sample = output, sample
        self.reading = str(output.get("reading"))
        self.fields = tuple(output.get("fields") or ())
        self.kinds = {f: dict(output["field_kinds"][f]) for f in self.fields}
        self.classes = tuple(str(c) for c in output.get("classes") or ())
        self.positive = None if output.get("positive") is None else str(output["positive"])
        self.compared = tuple(tuple(p) for p in output.get("compared") or ())
        self.decided = tuple(tuple(p) for p in output.get("decided") or ())
        self.truth = dict(output.get("truth") or {}) or None
        self.shape, self.level = sample.get("shape", "row"), sample.get("level")
        self.tolerance = float(manifest["roundtrip"]["tolerance"]["value"])
        self.env = manifest["serve"]["identity"]["env"]
        self.inputs = tuple(schema.get("input") or ())
        embedded = sorted((f for f in self.fields if self.kinds[f]["kind"] == "embedding"),
                          key=lambda f: int(self.kinds[f]["of"]))
        self._embedding_rank = {f: i for i, f in enumerate(embedded)}

    @staticmethod
    def of(root: Path = ROOT) -> "Oracle":
        manifest = json.loads((Path(root) / "MANIFEST.json").read_text(encoding="utf-8"))
        schema = json.loads((Path(root) / manifest.get("input_schema", "schema.json"))
                            .read_text(encoding="utf-8"))
        return Oracle(manifest, schema)

    # -- the shape -----------------------------------------------------------------------
    @property
    def window(self) -> bool:
        return self.shape == "window"

    @property
    def graph_level(self) -> bool:
        return self.shape == "graph" and self.level == "graph"

    @property
    def node_level(self) -> bool:
        return self.shape == "graph" and self.level != "graph"

    def kind_of(self, field: str) -> str:
        return self.kinds[field]["kind"]

    def fields_of(self, kind: str) -> tuple:
        return tuple(f for f in self.fields if self.kind_of(f) == kind)

    # -- the inputs ----------------------------------------------------------------------
    def input_names(self) -> tuple:
        return tuple(c["name"] for c in self.inputs)

    def required(self) -> str:
        """The first required input: the column the refusal tests leave out."""
        return next(c["name"] for c in self.inputs if c.get("required"))

    def numeric(self):
        """The first numeric input (None without one): where the type tests write text."""
        return next((c["name"] for c in self.inputs if c.get("dtype") != "string"), None)

    def ranged(self):
        """The first required numeric input with a range training saw, or None."""
        return next((c for c in self.inputs if c.get("required")
                     and c.get("dtype") != "string" and c.get("range_seen")), None)

    def non_finite(self) -> list:
        return [c["name"] for c in self.inputs if c.get("non_finite_seen")][:2]

    def codes(self) -> list:
        return [c["name"] for c in self.inputs if c.get("pattern")]

    @property
    def truth_column(self):
        return (self.truth or {}).get("column")

    @property
    def truth_type(self):
        return (self.truth or {}).get("type")

    def truth_is_input(self) -> bool:
        return self.truth_column in self.input_names()

    # -- requests ------------------------------------------------------------------------
    def request(self, examples: Mapping) -> dict:
        """The shipped test rows (and their history, when the examples carry one) as a body."""
        test = examples["test"]
        body = {"rows": list(test["rows"])}
        if test.get("history"):
            body["history"] = list(test["history"])
        return body

    def one_request(self) -> dict:
        """examples/request.json: one complete request of this shape (a row; a window's anchor
        and its history; one graph's rows) — the one whose first row is always answered."""
        return _read("examples/request.json")

    def expected(self) -> pd.DataFrame:
        return _read_parquet(ROOT / "examples" / "expected.parquet")

    def compared_positions(self) -> list:
        """The rows of examples/rows.csv that carry an expected answer (`__row__`)."""
        expected = self.expected()
        if "__row__" in expected.columns:
            return [int(r) for r in expected["__row__"]]
        return list(range(len(expected)))

    # -- answers -------------------------------------------------------------------------
    def includes(self) -> tuple:
        """The `include` values that show every gated field this record has."""
        return tuple(sorted({GATED[self.kind_of(f)] for f in self.fields
                             if self.kind_of(f) in GATED}))

    def shown(self, include=()) -> tuple:
        return tuple(f for f in self.fields
                     if self.kind_of(f) not in GATED or GATED[self.kind_of(f)] in include)

    def keys(self, include=()) -> tuple:
        """The JSON keys of one answer, in order: `answered`, the shown fields' keys, then
        `warnings` (a `decision` field has no key: the class carries it)."""
        body = [KEYS[self.kind_of(f)] for f in self.shown(include)
                if KEYS[self.kind_of(f)] and self.kind_of(f) not in TAIL]
        return ("answered", *dict.fromkeys(body), "warnings")

    def csv_columns(self, include=(), truth: bool = False) -> list:
        """The CSV answer's columns: the shown fields, the truth as `true_<column>` when the
        upload carried it, then `answered` and `warnings` where the record has them."""
        body = [f for f in self.shown(include) if self.kind_of(f) not in TAIL]
        if truth:
            body.append(f"true_{self.truth_column}")
        return body + [f for f in self.fields if self.kind_of(f) in TAIL]

    def sums_to_one(self) -> bool:
        return self.reading in SUMS_TO_ONE

    def value(self, prediction: Mapping, field: str):
        """One field of the answer frame, read where the field-kind table puts it in a JSON
        answer (None where the answer holds nothing for it)."""
        kind, of = self.kind_of(field), self.kinds[field].get("of")
        if kind == "decision":                  # binary: the class carries the 0|1 decision
            label = prediction.get("label")
            return None if label is None else int(str(label) == self.positive)
        if kind == "flag":
            names = prediction.get("labels")
            return None if names is None else int(str(of) in names)
        held = prediction.get(KEYS[kind])
        if held is None:
            return None
        if kind in MAPPED:
            return held.get(str(of))
        if kind == "horizon":
            step, _, q = str(of).partition("/")
            entry = held.get(f"h{step}")
            return entry.get(q) if q and isinstance(entry, Mapping) else entry
        if kind == "embedding":
            return held[self._embedding_rank[field]]
        return int(held) if kind == "alarm" else held

    def check(self, prediction: Mapping, expected: Mapping, tol: float) -> list:
        """What disagrees between one answer and one expected row: every compared field within
        `tol`, every decided field exactly; [] when the answer agrees."""
        if prediction.get("answered") is False:
            return ["the row was not answered: " + "; ".join(prediction.get("warnings") or ())]
        problems = []
        for field, _ in self.compared:
            got, want = self.value(prediction, field), expected.get(field)
            if not numbers_agree(got, want, tol):
                problems.append(f"{field}: answered {got!r}, expected {want!r} (tolerance "
                                f"{tol:.3g})")
        for field, _ in self.decided:
            got, want = self.value(prediction, field), expected.get(field)
            if not same_value(got, want):
                problems.append(f"{field}: decided {got!r}, expected {want!r}")
        return problems

    # -- the truth -----------------------------------------------------------------------
    def truth_payload(self, item: Mapping) -> dict:
        """A demonstration item (vega/demo@2) as a /v1/monitor body: its rows, its history,
        and one truth per row, typed as MANIFEST.output.truth says — none when the item knows
        no truth for any of its rows (the monitor takes a body without truth)."""
        body = {"rows": list(item["rows"])}
        if item.get("history"):
            body["history"] = list(item["history"])
        if self.truth_type is not None:
            truths = [_typed_truth(self.truth, item.get("truth"), row) for row in body["rows"]]
            if any(t is not None for t in truths):
                body["truth"] = truths
        return body

    def good_truth(self):
        """One value the truth's type accepts."""
        return {"class": (self.classes or ("",))[0], "number": 1.5,
                "labels": list(self.classes[:1])}.get(self.truth_type)

    def wrong_truth(self):
        """One value the truth's type refuses: a name no class has, text for a number, a label
        name the record does not list."""
        stranger = "none-of:" + "|".join(self.classes)
        return {"class": stranger, "number": "not a number",
                "labels": [stranger]}.get(self.truth_type)


ORACLE = Oracle.of(ROOT)



def sample_keys(oracle) -> tuple:
    """The columns a sample is cut by: a window's series and order, a graph's id."""
    s = oracle.sample or {}
    return tuple(c for c in (s.get("series_by"), s.get("order_by"), s.get("graph_id")) if c)


def plain_numeric(oracle):
    """The first numeric input that is no sample key (None without one): where an empty cell is
    sent."""
    return next((c["name"] for c in oracle.inputs if c.get("dtype") != "string"
                 and c["name"] not in sample_keys(oracle)), None)


@pytest.fixture(scope="module")
def client():
    """This bundle's service over the in-process test client, started once per module."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from serve.app import create_app
    with TestClient(create_app(ROOT)) as c:
        yield c
