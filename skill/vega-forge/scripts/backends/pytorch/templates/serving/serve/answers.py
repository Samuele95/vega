"""
serve.answers — the predictor's frame as the HTTP answer, generic over `field_kinds`.

`MANIFEST.output` (`vega/output@2`) gives every field of the Predictor's frame a kind, and the
kind alone decides where it lands in a `RowPrediction` and whether it is shown (bundle-contract
§7.3; vega-pack's catalog `field_kinds`): no branch per task. `probability`, `raw`, `quantile`,
`cumulative`, `recon` land as `{of: number}`; `horizon` as `{"h<h>": v}`, or `{"h<h>": {"<q>":
v}}` when `of` is `"h/q"`; `flag` as the names flagged 1; `embedding` in coordinate order;
`decision` in the CSV answer only; `raw`, `recon`, `embedding` only on their `include`, absent
without it under any serialization (FastAPI's default too). An unanswered row keeps its keys,
null, with `answered: false`; NaN and the infinities are null. The CSV answer is the record's
`fields` in order (gated on include), the upload's truth as `true_<column>` before `answered`.
"""
from __future__ import annotations

import math
from types import MappingProxyType
from typing import Any, Literal, Mapping, NamedTuple, Optional, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, create_model, model_serializer

__all__ = ["AnswerCodec", "FIELD_KEYS", "INCLUDES"]

FIELD_KEYS = MappingProxyType({"class": "label", "flag": "labels", "alarm": "anomalous",
    "probability": "probabilities", "raw": "logits", "value": "value", "quantile": "quantiles",
    "horizon": "values", "score": "score", "row_score": "reconstruction_error", "cumulative":
    "cumulative", "recon": "reconstruction", "embedding": "embedding", "decision": None,
    "group": "graph_id", "answered": "answered", "warnings": "warnings"})
# the kinds shown only on request, and the `include` value that shows each
INCLUDES = MappingProxyType({"raw": "logits", "recon": "reconstruction", "embedding": "embedding"})
MAPPED = frozenset({"probability", "raw", "quantile", "cumulative", "recon"})
TAIL = ("answered", "warnings")
WARNING_SEP = "; "
V1 = "MANIFEST.output is vega/output@1 (no reading, no field_kinds): re-pack with vega-pack v2.3"
# one field of the frame: its kind, its `of`, its JSON key and the include that shows it
_Slot = NamedTuple("_Slot", [("field", str), ("kind", str), ("of", Any), ("key", Optional[str]),
                             ("gate", Optional[str])])


class _Answer(BaseModel):
    """The base of `RowPrediction`: its JSON holds the keys the codec set, and always the tail."""
    model_config = ConfigDict(extra="forbid")

    @model_serializer(mode="wrap")
    def _written(self, handler):
        return {k: v for k, v in handler(self).items() if k in self.model_fields_set or k in TAIL}


class AnswerCodec:
    """The frame -> `RowPrediction` JSON and the CSV answer, for one output record."""

    def __init__(self, output: Mapping):
        if not output.get("field_kinds") or not output.get("reading"):
            raise ValueError(V1)
        self.reading = str(output["reading"])
        self.classes = tuple(str(c) for c in output.get("classes") or ())
        self.fields = tuple(output.get("fields") or ())
        self.truth = MappingProxyType(dict(output.get("truth") or {}))
        self._slots = _slots(self.fields, output["field_kinds"])
        self._model = self._row_model()

    def keys(self, include=()) -> tuple:
        """The JSON keys of one answer of this bundle, in the order they are written."""
        body = [s.key for s in self._shown(include) if s.key and s.kind not in TAIL]
        return ("answered", *dict.fromkeys(body), "warnings")

    def columns(self, include=(), truth: bool = False) -> list:
        """The CSV answer's columns: the fields (a gated kind only on its include), then the
        truth as `true_<column>` when the upload carried it, then `answered` and `warnings`."""
        body = [s.field for s in self._shown(include) if s.kind not in TAIL]
        if truth and self.truth.get("column"):
            body.append(f"true_{self.truth['column']}")
        return body + [s.field for s in self._slots if s.kind in TAIL]

    def response_model(self) -> type:
        """The pydantic `RowPrediction` of this bundle; a key the record does not name fails."""
        return self._model

    def rows(self, frame: pd.DataFrame, include=()) -> list:
        """One `RowPrediction` dict per frame row; gated keys only on their include."""
        out = [{"answered": done} for done in _answered(frame, self._field_of("answered"))]
        for slot in (s for s in self._shown(include) if s.kind not in TAIL):
            for i, value in enumerate(frame[slot.field].tolist()):
                _put(out[i], slot, value, out[i]["answered"])
        for i, notes in enumerate(_warnings_of(frame, self._field_of("warnings"))):
            out[i]["warnings"] = notes
        return out

    def table(self, frame: pd.DataFrame, truth=None, include=()) -> pd.DataFrame:
        """The CSV answer: the frame's columns of `columns(include, truth)`, the truth as
        `true_<column>` (label names joined on the truth's `sep`), warnings as one text."""
        truth_column, warnings = f"true_{self.truth.get('column')}", self._field_of("warnings")
        out = pd.DataFrame(index=range(len(frame)))
        for name in self.columns(include, truth is not None):
            if name == truth_column:
                out[name] = [_truth_text(v, self.truth.get("sep")) for v in truth]
            elif name == warnings:
                out[name] = [WARNING_SEP.join(w) for w in _warnings_of(frame, name)]
            else:
                out[name] = frame[name].array  # a nullable 0|1 column stays 0|1
        return out

    def _shown(self, include) -> list:
        return [s for s in self._slots if s.gate is None or s.gate in (include or ())]

    def _field_of(self, kind: str) -> Optional[str]:
        return next((s.field for s in self._slots if s.kind == kind), None)

    def _row_model(self) -> type:
        fields: dict = {"answered": (bool, Field(True)),
                        "warnings": (list[str], Field(default_factory=list))}
        for slot in self._slots:
            if slot.key and slot.key not in fields:
                fields[slot.key] = (_key_type(slot.kind, self.classes), Field(None))
        return create_model("RowPrediction", __base__=_Answer, **fields)


def _slots(fields, field_kinds: Mapping) -> tuple:
    """The frame's fields as slots, embeddings in coordinate order. A field without a kind, or a
    kind this codec does not read, is a pack defect: refused here, never guessed."""
    slots = []
    for name in fields:
        kind = (field_kinds.get(name) or {}).get("kind")
        if kind not in FIELD_KEYS:
            raise ValueError(f"MANIFEST.output.field_kinds[{name!r}] is {kind!r}, not a kind the "
                             f"answer codec reads ({sorted(FIELD_KEYS)})")
        entry = field_kinds[name]
        slots.append(_Slot(name, kind, entry.get("of"), FIELD_KEYS[kind], INCLUDES.get(kind)))
    order = iter(sorted((s for s in slots if s.kind == "embedding"), key=lambda s: int(s.of)))
    return tuple(next(order) if s.kind == "embedding" else s for s in slots)


def _key_type(kind: str, classes: tuple) -> Any:
    """The pydantic type of the key a kind lands under: every key may be null, and so may every
    number inside one (`_number`: JSON has no literal for NaN or an infinity)."""
    name, num = (Literal[classes] if classes else str), Optional[float]
    types = {"class": name, "flag": list[name], "alarm": bool, "group": str,
             "horizon": dict[str, Union[num, dict[str, num]]], "embedding": list[num]}
    return Optional[types.get(kind, dict[str, num] if kind in MAPPED else float)]


def _put(item: dict, slot: _Slot, value, answered: bool) -> None:
    """Place one field's value of one row under its key."""
    if slot.key is None:
        return
    if not answered and slot.kind != "group":
        item[slot.key] = None
    elif slot.kind in MAPPED:
        item.setdefault(slot.key, {})[str(slot.of)] = _number(value)
    elif slot.kind == "horizon":
        step, _, q = str(slot.of).partition("/")       # "h", or "h/q" with quantiles
        values = item.setdefault(slot.key, {})
        (values.setdefault(f"h{step}", {}) if q else values)[q or f"h{step}"] = _number(value)
    elif slot.kind == "flag":
        names = item.setdefault(slot.key, [])
        if not _missing(value) and bool(value):
            names.append(str(slot.of))
    elif slot.kind == "embedding":
        item.setdefault(slot.key, []).append(_number(value))
    elif _missing(value):
        item[slot.key] = None
    else:
        item[slot.key] = (bool(value) if slot.kind == "alarm" else
                          str(value) if slot.kind in ("class", "group") else _number(value))


def _missing(value) -> bool:
    return value is None or value is pd.NA or (isinstance(value, (float, np.floating))
                                               and math.isnan(value))


def _number(value) -> Optional[float]:
    """A JSON number, or null where JSON has none (missing, NaN, an infinity)."""
    return None if _missing(value) or not math.isfinite(float(value)) else float(value)


def _answered(frame: pd.DataFrame, field: Optional[str]) -> list:
    """Whether each row was answered: unless its field, or the Predictor's `attrs`, says not."""
    if field is None or field not in frame.columns:
        held = frame.attrs.get("answered")
        return [bool(a) for a in held] if held is not None and len(held) == len(frame) \
            else [True] * len(frame)
    return [not _missing(v) and bool(v) for v in frame[field].tolist()]


def _warnings_of(frame: pd.DataFrame, field: Optional[str]) -> list:
    """Each row's warnings as a list of text (the frame holds a list, or one text)."""
    if field is None or field not in frame.columns:
        return [[] for _ in range(len(frame))]
    return [[str(w) for w in value if str(w)] if isinstance(value, (list, tuple, np.ndarray))
            else [w for w in ("" if _missing(value) else str(value)).split(WARNING_SEP) if w]
            for value in frame[field].tolist()]


def _truth_text(value, sep: Optional[str]):
    """One truth of the upload as a CSV cell: label names joined on `sep`."""
    return (sep or WARNING_SEP).join(map(str, value)) if isinstance(value, (list, tuple)) \
        else value
