"""
serve.contract — the HTTP boundary's contract, generated from the bundle's own records.

The facade the route modules use. It is built once at start-up from `schema.json` and
`MANIFEST.json` (`output`, `vega/output@2`; `sample`; the chain's digests), frozen after, and
versioned by the digest of the `schema.json` it was built from: two services reporting the same
version accept exactly the same requests. What it is made of lives beside it:

    models    RowModelSet   the request models (rows.py): Row, PredictRequest, MonitorRequest,
                            OneRowRequest, typed truth and history per the records
    tables    TableCodec    JSON rows and CSV tables to the raw frame, and back (tables.py)
    answers   AnswerCodec   the frame to `RowPrediction` JSON and to the CSV answer, generic
                            over `MANIFEST.output.field_kinds` (answers.py)
    problems  ProblemErrors the validator's errors, grouped by column (problems.py)

Two contracts, two names. `contract_version` on the wire is this API's input contract — the
digest of `schema.json`, the raw columns a caller sends. `feature_contract_version` is the
network's feature contract — the digest of the shaped manifest, the columns the recipe hands the
network, which is what the domain's `Prediction.contract_version` carries. They change for
different reasons, so an answer states both rather than one word meaning either. Every answer
also names its reading (`output_kind`), so a caller knows which keys to read.

A route returns the codec's dicts under `-> contract.PredictResponse` (FastAPI validates the
answer against the record, so a drifted key is a 500 in tests, never a silent change). The
`RowPrediction` inside it serializes only the keys the codec wrote, so under FastAPI's default
serialization a gated key without its `include` is absent, not null, and an unanswered row's
nulls stay (D2 §5.2): `response_model_exclude_unset` is not needed, and harmless if given.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Optional

from pydantic import create_model

from .answers import AnswerCodec
from .problems import ProblemErrors, frozen, thawed
from .rows import TOKENS, RowModels
from .tables import TableCodec

__all__ = ["ApiContract", "schema_digest"]

# the manifest keys the contract reads, held frozen with it
MANIFEST_KEYS = ("output", "sample", "chain")
ENVELOPE = {"api_version": (str, ...), "model_version": (str, ...),
            "contract_version": (Optional[str], None),
            "feature_contract_version": (Optional[str], None), "n_rows": (int, ...),
            "output_kind": (str, ...)}


def schema_digest(path) -> str:
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ApiContract:
    """The request and response models and codecs of one bundle, immutable after start-up."""

    def __init__(self, schema: dict, manifest: dict, max_rows: int = 10000,
                 version: Optional[str] = None):
        # held deep-frozen: no caller, and no response built from them, can change the
        # contract a running service validates against
        self.schema = frozen(schema)
        self.manifest = frozen({k: manifest.get(k) for k in MANIFEST_KEYS})
        self.version = version
        self.feature_version = ((manifest.get("chain") or {}).get("shape") or {}) \
            .get("manifest_digest")
        self.max_rows = int(max_rows)
        output, sample = dict(manifest.get("output") or {}), dict(manifest.get("sample") or {})
        self.answers = AnswerCodec(output)
        self.models = RowModels(schema, output, sample, self.max_rows).build()
        columns, truth = self.models.columns, self.models.truth
        self.problems = ProblemErrors(truth.column, truth.refusal, columns)
        self.tables = TableCodec(columns, self.models, self.problems, self.max_rows)
        self.PredictResponse, self.Reading, self.MonitorResponse = _response_models(
            self.answers.response_model())
        self._frozen = True

    def __setattr__(self, name, value):
        if getattr(self, "_frozen", False):
            raise AttributeError(f"ApiContract is immutable; {name!r} cannot be set")
        super().__setattr__(name, value)

    @staticmethod
    def from_bundle(root, max_rows: int = 10000) -> "ApiContract":
        """The contract of the bundle at `root`, refusing a `schema.json` the manifest does not
        pin, or whose digest is not the one pinned — the API would otherwise describe another
        bundle (a bundle's files are closed-world: pack pins every one, bundle-contract §7)."""
        root = Path(root)
        manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
        name = manifest.get("input_schema", "schema.json")
        version = schema_digest(root / name)
        pinned = (manifest.get("files") or {}).get(name)
        if not pinned:
            raise ValueError(f"MANIFEST.json does not pin {name} (no files[{name!r}] digest); "
                             "the input contract cannot be shown to be this bundle's")
        if pinned != version:
            raise ValueError(f"{name} digests to {version}, but MANIFEST.json records {pinned}; "
                             "the input contract is not the one this bundle was packed with")
        schema = json.loads((root / name).read_text(encoding="utf-8"))
        return ApiContract(schema, manifest, max_rows=max_rows, version=version)

    @property
    def reading(self) -> str:
        return self.answers.reading

    def envelope(self, model_version, n_rows: int, api_version: str) -> dict:
        """The version stamp every answer carries: the API, the model (its weights' digest), this
        input contract (its schema's digest), the network's feature contract (the shaped
        manifest's digest, the one each domain `Prediction` carries) and the reading."""
        return {"api_version": api_version, "model_version": model_version,
                "contract_version": self.version,
                "feature_contract_version": self.feature_version, "n_rows": int(n_rows),
                "output_kind": self.reading}

    def describe(self) -> dict:
        """`/v1/schema`: the input columns and how each is typed, the truth and its decoding,
        the whole output record (fields, kinds, thresholds) and the sample record. A copy: what
        a caller does to it never reaches the contract."""
        columns, truth = self.models.columns, self.models.truth
        output = thawed(self.manifest.get("output")) or {}
        return {"version": self.version, "feature_contract_version": self.feature_version,
                "input": thawed(columns.inputs), "ignored": list(columns.ignored),
                "required": list(columns.required), "classes": list(self.answers.classes),
                "max_rows": self.max_rows,
                "label": {"column": truth.column, "decode": dict(truth.decode)},
                "truth": thawed(output.get("truth")), "output": output,
                "sample": thawed(self.manifest.get("sample")) or {"shape": "row"},
                "tokens": sorted(TOKENS)}

    @staticmethod
    def readings(readings) -> list:
        """Monitor readings as JSON, each with the human title of the check that produced it
        (the reading's `ReadingProvenance`, read as fields)."""
        out = []
        for r in readings:
            origin, value = r.window, r.value
            out.append({"title": origin.title if origin else None,
                        "column": str(r.column), "measure": r.measure,
                        "value": None if value is None or not math.isfinite(float(value))
                        else float(value),
                        "verdict": r.verdict, "check": origin.check if origin else None,
                        "concern": origin.concern if origin else None,
                        "note": origin.note if origin else None,
                        "window_rows": origin.window_rows if origin else None})
        return out


def _response_models(row_prediction: type) -> tuple:
    """(PredictResponse, DriftReadingOut, MonitorResponse): the envelope around the answers and
    around the monitor's readings."""
    predict = create_model("PredictResponse", **ENVELOPE,
                           predictions=(list[row_prediction], ...))
    reading = create_model(
        "DriftReadingOut", title=(Optional[str], None), column=(str, ...), measure=(str, ...),
        value=(Optional[float], None), verdict=(str, ...), check=(Optional[str], None),
        concern=(Optional[str], None), note=(Optional[str], None),
        window_rows=(Optional[int], None))
    monitor = create_model("MonitorResponse", **ENVELOPE, moved=(int, ...),
                           readings=(list[reading], ...))
    return predict, reading, monitor
