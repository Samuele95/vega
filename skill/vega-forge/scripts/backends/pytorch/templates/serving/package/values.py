"""
{{PACKAGE}}.{{MODULE_SERVING}}.values — the readers of the value and score readings (regression,
quantile, forecast, reconstruction, the two anomaly readings) and `reader_for`, the dispatch
over all ten (support; used by the Predictor, and by /vega-pack through the bundle's own
readers).

Emitted by /vega-pack from `backends/pytorch/templates/serving/package/values.py`: a defect is
fixed in that template and the bundle re-packed, never in this copy.

The raw fields of these readings are the head itself (the `compared` pairs of `heads.py`): the
quantile columns `q_<q>`, the forecast steps (horizon-major under a quantile loss), the
reconstruction `recon_<feature>` / `out_<k>` and the embedding `emb_<j>`. What a reader derives
comes from the spec's `TaskReadout`: the point value (`score`: the 0.5 quantile of a quantile
head, the steps of a forecast) and the per-row score of the reconstruction and anomaly readings
(`reconstruction_error`, an anomaly's `score`: `row_score`, computed by `heads.BaseReader.head`
from the batch's inputs, or from the model's svdd centre, the buffer the checkpoint carries).

An anomaly decision is `score ≥ threshold` on the **score's own scale**: the threshold the
evaluation chose on valid is a score quantile, never squashed into a probability. A record with
no threshold (`threshold.json.applies == false` gives `threshold: null`, bundle-contract §2)
answers the score undecided: `prediction` and `label` are null, never a guessed cut.

`reader_for(records)` builds the reader of `records.output.reading` over
`TaskReadout.for_spec(records.spec)`; it refuses a record whose reading, link, head width or
classes the spec's readout does not share, a compared column outside the head, and a binary
record whose positive is not `task.classes[1]` (the class the head's logit reads).
"""
from __future__ import annotations

from types import MappingProxyType

import numpy as np

from ..{{MODULE_MODEL}} import TaskReadout
from .heads import (BaseReader, BinaryReader, MulticlassReader, MultilabelReader,
                    OrdinalReader, Reader)
from .records import Reading, ServingRecords

__all__ = ["READERS", "AnomalyReader", "ForecastReader", "QuantileReader",
           "ReconstructionReader", "RegressionReader", "reader_for"]


class RegressionReader(BaseReader):
    """`value` is the readout's score of the one-output head."""

    def _derived(self, head, z, decision):
        return self._one("value", self._array(self._readout.score(z)))


class QuantileReader(RegressionReader):
    """`q_<q>` are the head's columns; `value` (present when 0.5 is one of the quantiles) is
    the readout's point estimate, the 0.5 quantile's column."""


class ForecastReader(BaseReader):
    """`value_h<h>` is the readout's step h; under a quantile loss every `value_h<h>_q<q>` is a
    head column (horizon-major), which the compared pairs already name."""

    def _derived(self, head, z, decision):
        if self.output.quantiles:
            return {}
        return self._by_of("horizon", self._array(self._readout.score(z)), self.output.horizons)


class ReconstructionReader(BaseReader):
    """`reconstruction_error` is the per-row score (numeric squared error plus the mean
    cross-entropy of the nominal columns); `recon_<feature>` / `out_<k>` are the head."""


class AnomalyReader(BaseReader):
    """`score` is the per-row score (the reconstruction error, or ‖f(x) − c‖² for svdd);
    `prediction` is `score ≥ threshold` on the score's scale; `label` (a labelled task) the
    class the decision names, `[negative, positive]`. Without a threshold both are null."""

    def _derived(self, head, z, decision):
        if decision is None and self.output.threshold is None:
            undecided = np.full(head.raw.shape[0], None, dtype=object)
            return {**self._one("alarm", undecided), **self._one("class", undecided)}
        if decision is None:
            decision = (head.row_score >= float(self.output.threshold)).astype(np.int64)
        out = self._one("alarm", np.asarray(decision, dtype=np.int64))
        if self.output.classes:
            out.update(self._one("class", self._names(decision)))
        return out


READERS = MappingProxyType({
    Reading.BINARY: BinaryReader, Reading.MULTICLASS: MulticlassReader,
    Reading.MULTILABEL: MultilabelReader, Reading.ORDINAL: OrdinalReader,
    Reading.REGRESSION: RegressionReader, Reading.QUANTILE: QuantileReader,
    Reading.FORECAST: ForecastReader, Reading.RECONSTRUCTION: ReconstructionReader,
    Reading.ANOMALY_RECONSTRUCTION: AnomalyReader, Reading.ANOMALY_SVDD: AnomalyReader})


def reader_for(records: ServingRecords) -> Reader:
    """The reader of the bundle's reading, over the spec's one readout (R-B7)."""
    readout = TaskReadout.for_spec(records.spec)
    return READERS[records.output.reading](records, readout)
