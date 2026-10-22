"""
{{PACKAGE}}.{{MODULE_SERVING}}.records — `MANIFEST.output` (`vega/output@2`) and `MANIFEST.sample`
read once, as frozen values (support of the Predictor; emitted by /vega-pack from
`backends/pytorch/templates/serving/package/records.py`: fix the template, never this copy).

The reading, not the task, drives serving (bundle-contract §2.1): pack wrote the fields, their
kinds and the compared and decided pairs once, and nothing here re-derives the task.
`ServingRecords.of` holds both records with the spec, the schema and the model version; a v2.2
manifest (no `schema: vega/output@2`) is refused by name. EJ-17 values, EJ-49 named refusals.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

from ..domain import NetSpec, SampleShape, Versions

__all__ = ["OUTPUT_SCHEMA", "ROW_SCORED", "SHAPE_KINDS", "TASK_KIND", "OutputRecord", "Reading",
           "SampleRecord", "ServingRecords", "thawed"]

OUTPUT_SCHEMA = "vega/output@2"
V1 = "MANIFEST.output is vega/output@1; re-pack with vega-pack v2.3"
SHAPE_KINDS = frozenset({"group", "answered", "warnings"})    # the shape's own fields (§2.1)


class Reading(str, Enum):
    """The ten readings of bundle-contract §2.1; a `sequence-*` task reads as its base."""

    BINARY = "binary"
    MULTICLASS = "multiclass"
    MULTILABEL = "multilabel"
    ORDINAL = "ordinal"
    REGRESSION = "regression"
    QUANTILE = "quantile"
    FORECAST = "forecast"
    RECONSTRUCTION = "reconstruction"
    ANOMALY_RECONSTRUCTION = "anomaly-reconstruction"
    ANOMALY_SVDD = "anomaly-svdd"


# the readings whose answer carries a per-row score (`row_score` in the stored layout), and the
# effective task kind whose readout reads a reading's head, where it is not the reading itself
ROW_SCORED = frozenset({"reconstruction", "anomaly-reconstruction", "anomaly-svdd"})
TASK_KIND = MappingProxyType({"quantile": "regression", "reconstruction": "unsupervised",
                              "anomaly-reconstruction": "anomaly", "anomaly-svdd": "anomaly"})


@dataclass(frozen=True)
class OutputRecord:
    """`MANIFEST.output` (`vega/output@2`) as a value: every key of bundle-contract §2.1."""

    reading: Reading
    kind: str
    base: str | None
    link: str
    decision: str
    head_out: int
    classes: tuple
    positive: str | None
    threshold: float | None
    threshold_source: str | None
    threshold_scale: str | None
    threshold_per_label: Mapping | None
    quantiles: tuple
    horizons: tuple
    fields: tuple
    field_kinds: Mapping
    compared: tuple
    decided: tuple
    score_field: str | None
    truth: Mapping | None

    @staticmethod
    def from_manifest(manifest: Mapping) -> "OutputRecord":
        """Refused when it is not `vega/output@2` or its fields, kinds and pairs disagree."""
        out = (manifest or {}).get("output")
        if not isinstance(out, Mapping) or out.get("schema") != OUTPUT_SCHEMA:
            raise ValueError(V1)
        record = OutputRecord(
            reading=_member(Reading, out.get("reading"), "MANIFEST.output.reading"),
            kind=str(out.get("kind")), base=out.get("base"), link=str(out.get("link")),
            decision=str(out.get("decision")), head_out=out.get("head_out"),
            classes=tuple(str(c) for c in out.get("classes") or ()),
            positive=None if out.get("positive") is None else str(out["positive"]),
            threshold=None if out.get("threshold") is None else float(out["threshold"]),
            threshold_source=out.get("threshold_source"),
            threshold_scale=out.get("threshold_scale"),
            threshold_per_label=_frozen(out.get("threshold_per_label")) or None,
            quantiles=tuple(float(q) for q in out.get("quantiles") or ()),
            horizons=tuple(int(h) for h in out.get("horizons") or ()),
            fields=tuple(str(f) for f in out.get("fields") or ()),
            field_kinds=_frozen(out.get("field_kinds") or {}),
            compared=tuple(tuple(str(v) for v in pair) for pair in out.get("compared") or ()),
            decided=tuple(tuple(str(v) for v in pair) for pair in out.get("decided") or ()),
            score_field=out.get("score_field"), truth=_frozen(out.get("truth")))
        record._check()
        return record

    def of(self, field: str) -> Any:
        return self.field_kinds[field].get("of")

    def fields_of(self, kind: str) -> tuple:
        """The fields of one kind, in the record's order."""
        return tuple(f for f in self.fields if self.field_kinds[f]["kind"] == kind)

    @property
    def answer_fields(self) -> tuple:
        """`fields` without the shape's own (`graph_id`, `answered`, `warnings`)."""
        return tuple(f for f in self.fields if self.field_kinds[f]["kind"] not in SHAPE_KINDS)

    def _check(self) -> None:
        width = self.head_out
        if not isinstance(width, int) or isinstance(width, bool) or width < 1:
            raise ValueError(f"MANIFEST.output.head_out is {width!r}; a positive integer is "
                             "expected")
        unkinded = [f for f in self.fields if not isinstance(self.field_kinds.get(f), Mapping)]
        outside = [list(pair) for pair in self.compared + self.decided
                   if len(pair) != 2 or pair[0] not in self.fields]
        if unkinded or outside:
            raise ValueError(f"MANIFEST.output: field(s) {unkinded} have no kind, pair(s) "
                             f"{outside} are not [field, source] over its fields; the answer is "
                             "read by kind only — re-pack with vega-pack v2.3")


@dataclass(frozen=True)
class SampleRecord:
    """`MANIFEST.sample` (bundle-contract §2.2) as a value; a manifest without it reads rows."""

    shape: SampleShape = SampleShape.ROW
    length: int = 1
    stride: int = 1
    horizon: int | None = None
    lag_depth: int = 0
    history_rows: int = 0
    series_by: str | None = None
    order_by: str | None = None
    context: str | None = None
    autoregressive: bool = False
    level: str | None = None
    protocol: str | None = None
    graph_id: str | None = None
    reference: Mapping | None = None

    @staticmethod
    def from_manifest(manifest: Mapping) -> "SampleRecord":
        """R-A5, R-A3; a window's `history_rows` is `(length − 1) + lag_depth` when unstated."""
        sample = dict((manifest or {}).get("sample") or {})
        shape = _member(SampleShape, sample.get("shape") or "row", "MANIFEST.sample.shape")
        values = {key: sample[key] for key in _SAMPLE_KEYS if sample.get(key) is not None}
        if shape is SampleShape.WINDOW:
            length = values.get("length")
            if not isinstance(length, int) or isinstance(length, bool) or length < 1:
                raise ValueError(f"MANIFEST.sample.length is {length!r}; a window sample needs "
                                 "a positive integer length")
            values.setdefault("history_rows", length - 1 + int(values.get("lag_depth", 0)))
        if shape is SampleShape.GRAPH:
            values.setdefault("level", "node")
        return SampleRecord(shape=shape, reference=_frozen(sample.get("reference")), **values)


_SAMPLE_KEYS = ("length", "stride", "horizon", "lag_depth", "history_rows", "series_by",
                "order_by", "context", "autoregressive", "level", "protocol", "graph_id")


@dataclass(frozen=True, eq=False)
class ServingRecords:
    """What one Predictor serves, the manifest's two records read (compared by identity)."""

    spec: NetSpec
    manifest: Mapping
    schema: Mapping
    model_version: str | None
    output: OutputRecord
    sample: SampleRecord

    @staticmethod
    def of(spec, manifest: Mapping, schema: Mapping, model_version=None) -> "ServingRecords":
        """`spec` a NetSpec or its dict; records written for another model are refused."""
        spec = spec if isinstance(spec, NetSpec) else NetSpec.from_dict(spec)
        output = OutputRecord.from_manifest(manifest)
        sample = SampleRecord.from_manifest(manifest)
        data = spec.to_dict()
        kind = (data.get("task") or {}).get("kind")
        stated = ((data.get("data_contract") or {}).get("sample") or {}).get("shape") or "row"
        if output.kind != kind or sample.shape.value != stated:
            raise ValueError(f"the manifest describes a {output.kind} task over "
                             f"{sample.shape.value} samples and the spec a {kind} task over "
                             f"{stated} samples: it was written for another model — re-pack")
        return ServingRecords(spec, _frozen(manifest), _frozen(schema or {}), model_version,
                              output, sample)

    @property
    def versions(self) -> Versions:
        """The feature contract and the model an answer comes from."""
        return Versions(contract=self.spec.feature_contract.manifest_digest,
                        model=self.model_version)


def _member(enum, value, where: str):
    try:
        return enum(value)
    except ValueError:
        raise ValueError(f"{where} {value!r} is not one of "
                         f"{', '.join(m.value for m in enum)}") from None


def _frozen(value):
    """A read-only copy: mappings as `MappingProxyType`, lists as tuples, all the way down."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _frozen(v) for key, v in value.items()})
    return tuple(_frozen(v) for v in value) if isinstance(value, (list, tuple)) else value


def thawed(value):
    """A frozen record back as plain JSON values (dicts and lists)."""
    if isinstance(value, Mapping):
        return {key: thawed(v) for key, v in value.items()}
    return [thawed(v) for v in value] if isinstance(value, (list, tuple)) else value
