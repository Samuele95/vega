"""
{{PACKAGE}}.{{MODULE_SERVING}}.heads — the answer frame of a head: the reader every reading
shares, and the class readings' readers. Support of the Predictor, read by /vega-pack too
(examples, the round trip, the monitor); emitted from `backends/pytorch/templates/serving/`.

A reader turns the head's raw outputs into the answer: one row per sample, one column per field
of `MANIFEST.output.fields` (the shape's `graph_id`, `answered`, `warnings` aside: the Predictor
adds them). What an output *means* is never decided here: `TaskReadout.for_spec(spec)` (the
Model module's `readout.py`, R-B7) is the one source of the score, the probabilities, the argmax
and cumulative decisions and the per-row score, for trainer, evaluator and bundle alike; a
reader keeps field and class names and the threshold decision on the record's scale.

  head(out, batch, anatomy)  the output tensor -> `HeadOutput` (raw head, row score: anatomy's)
  frame(head) / from_stored(stored)   the answer, from a head or from the evaluation's
                           `vega/predictions-layout@1` rows (`score` / `score_<k>`, `row_score`,
                           and a stored `prediction` as the decision of the pair that names it)
  warnings(head)           the reading's own per-sample warnings (ordinal: clipped probabilities)
  class_index(rec), score_vector(rec), probability_vector(rec)   an answered row's class,
                           score (the raw head, or the per-row score) and probabilities

The raw fields are the record's `compared` pairs, each naming the stored column it equals, so
the round trip compares like with like by construction. `HeadOutput` is `samples.py`'s (D2 §4.3),
the one class the Predictor, the readers and inspection share; a reader never writes into it.
`values.py` holds the other readers and `reader_for(records)`; it imports this module, never
the reverse. EJ-23, EJ-49.
"""
from __future__ import annotations

import math
import re
from typing import Mapping, Protocol

import numpy as np
import pandas as pd
import torch

from ..{{MODULE_MODEL}} import TaskReadout
from .records import ROW_SCORED, TASK_KIND, Reading, ServingRecords
from .samples import HeadOutput

__all__ = ["CLIPPED", "BaseReader", "BinaryReader", "MulticlassReader", "MultilabelReader",
           "OrdinalReader", "Reader"]

CLIPPED = "cumulative logits not monotone; class probabilities clipped"
DEFAULT_PROBABILITY_THRESHOLD = 0.5          # vega:default, when the evaluation recorded none
COMPUTED_SOURCES = frozenset({"argmax", "cumulative", "threshold"})   # else a stored column
STORED_SCORE = re.compile(r"score(?:_(\d+))?")
ROW_SCORE = "row_score"


class Reader(Protocol):
    """What the Predictor, the inspection targets and pack read an answer through."""

    reading: Reading
    fields: tuple
    needs_row_score: bool

    def head(self, out, batch=None, anatomy=None) -> HeadOutput: ...
    def frame(self, head: HeadOutput) -> pd.DataFrame: ...
    def from_stored(self, stored: pd.DataFrame) -> pd.DataFrame: ...
    def warnings(self, head: HeadOutput) -> tuple: ...
    def class_index(self, rec: Mapping) -> int | None: ...
    def score_vector(self, rec: Mapping) -> tuple: ...
    def probability_vector(self, rec: Mapping) -> tuple: ...


class BaseReader:
    """What every reading shares; a subclass names the fields it derives (`_derived`)."""

    def __init__(self, records: ServingRecords, readout: TaskReadout):
        output = records.output
        _agree(output, readout)
        self.output, self.reading, self.fields = output, output.reading, output.answer_fields
        self.needs_row_score = output.reading.value in ROW_SCORED
        self._readout = readout
        self._head_columns = {field: _head_column(stored, field, output.head_out)  # None: row
                              for field, stored in output.compared}
        self._index = {str(c): i for i, c in enumerate(readout.task.labels.classes)}
        self._stored_decision = next((pair for pair in output.decided
                                      if pair[1] not in COMPUTED_SOURCES), None)
        by_column = sorted((k, f) for f, k in self._head_columns.items() if k is not None)
        rows = tuple(f for f, k in self._head_columns.items() if k is None)
        self._score_fields = rows if self.needs_row_score else tuple(f for _, f in by_column)

    def head(self, out, batch=None, anatomy=None) -> HeadOutput:
        """The output `(B, head_out)` as a `HeadOutput`, the per-row score the readout's."""
        tensor = out.detach()
        raw = self._array(tensor.unsqueeze(-1) if tensor.dim() == 1 else tensor)
        row = self._array(self._readout.row_score(tensor, batch, anatomy)) \
            if self.needs_row_score else None
        return self._checked(HeadOutput(raw, row))

    def frame(self, head: HeadOutput) -> pd.DataFrame:
        return self._frame(self._checked(head), None)

    def from_stored(self, stored: pd.DataFrame) -> pd.DataFrame:
        """The frame the evaluation's stored rows answer, indexed like `stored`."""
        row = None
        if self.needs_row_score:
            if ROW_SCORE not in stored.columns:
                raise ValueError(f"the evaluation stored no per-row score for a "
                                 f"{self.reading.value} head; re-evaluate with vega-eval v2.3")
            row = stored[ROW_SCORE].to_numpy(dtype=np.float64)
        head = self._checked(HeadOutput(_stored_raw(stored, self.output.head_out), row))
        source = self._stored_decision[1] if self._stored_decision else None
        decision = stored[source].to_numpy(dtype=np.int64) if source in stored.columns else None
        return self._frame(head, decision).set_axis(stored.index)

    def warnings(self, head: HeadOutput) -> tuple:
        return tuple(() for _ in range(head.raw.shape[0]))

    def class_index(self, rec: Mapping) -> int | None:
        """The spec's index of the record's class (for `LabelSet.at`), None without one."""
        named = self.output.fields_of("class")
        value = rec.get(named[0]) if named else None
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        if str(value) not in self._index:
            raise ValueError(f"the answer's class {value!r} is not one of the spec's task.classes")
        return self._index[str(value)]

    def score_vector(self, rec: Mapping) -> tuple:
        return tuple(float(rec[field]) for field in self._score_fields)

    def probability_vector(self, rec: Mapping) -> tuple:
        """The record's probabilities in class order; () without any, or any not finite."""
        by = {str(self.output.of(f)): f for f in self.output.fields_of("probability")}
        out = tuple(float(rec[by[str(c)]]) for c in self.output.classes if str(c) in by)
        return out if all(map(math.isfinite, out)) else ()

    # -- inside ----------------------------------------------------------------------------
    def _frame(self, head: HeadOutput, decision) -> pd.DataFrame:
        columns = {field: head.row_score if k is None else head.raw[:, k]
                   for field, k in self._head_columns.items()}
        columns.update(self._derived(head, _tensor(head.raw), decision))
        unfilled = [field for field in self.fields if field not in columns]
        if unfilled:
            raise ValueError(f"the {self.reading.value} reader fills no value for "
                             f"{', '.join(unfilled)}: the record and the reader disagree")
        return pd.DataFrame({field: columns[field] for field in self.fields},
                            index=pd.RangeIndex(head.raw.shape[0]))

    def _derived(self, head: HeadOutput, z, decision) -> dict:
        return {}

    def _checked(self, head: HeadOutput) -> HeadOutput:
        if head.raw.shape[1] != self.output.head_out:
            raise ValueError(f"the head gives {head.raw.shape[1]} output(s) per sample and "
                             f"MANIFEST.output.head_out is {self.output.head_out}")
        if self.needs_row_score and head.row_score is None:
            raise ValueError(f"a {self.reading.value} answer reads the per-row score, and the "
                             "head output carries none")
        return head

    def _one(self, kind: str, values) -> dict:
        return {field: values for field in self.output.fields_of(kind)}

    def _by_of(self, kind: str, matrix, order) -> dict:
        """Each field of `kind` takes the column of `matrix` its `of` names in `order`."""
        at = {str(v): i for i, v in enumerate(order)}
        out = {}
        for field in self.output.fields_of(kind):
            of = str(self.output.of(field))
            if of not in at:
                raise ValueError(f"MANIFEST.output.field_kinds[{field!r}] reads the {kind} of "
                                 f"{of!r}, which the record does not name")
            out[field] = matrix[:, at[of]]
        return out

    def _names(self, index) -> np.ndarray:
        index, classes = np.asarray(index, dtype=np.int64), self.output.classes
        if len(index) and (index.min() < 0 or index.max() >= len(classes)):
            raise ValueError(f"a decision outside the {len(classes)} class(es) of the record")
        return np.asarray(classes, dtype=object)[index]

    @staticmethod
    def _array(tensor) -> np.ndarray:
        """A readout result as a numpy array on the host (float64 stays float64)."""
        return tensor.detach().cpu().numpy()

    def _cut(self, label=None) -> float:
        """The probability-scale threshold (a label's own, for a multilabel record)."""
        per_label = self.output.threshold_per_label or {}
        if label is not None and per_label.get(label) is not None:
            return float(per_label[label])
        threshold = self.output.threshold
        return DEFAULT_PROBABILITY_THRESHOLD if threshold is None else float(threshold)


class BinaryReader(BaseReader):
    """`p_<class>` from the readout's probability (the logit's sigmoid, whatever the batch);
    `prediction` is `p_<positive> ≥ threshold`; `label` the class it names; `score` the logit.
    The trainer encodes `task.classes` in spec order, so σ(z) is P(task.classes[1]): a record
    whose positive is another class would answer every probability and label inverted."""

    def __init__(self, records: ServingRecords, readout: TaskReadout):
        super().__init__(records, readout)
        second = str(readout.task.labels.at(1).name)
        if self.output.positive != second:
            raise ValueError(f"the head's logit is the log-odds of task.classes[1] ({second!r}); "
                             f"task.positive names {self.output.positive!r}: re-forge with the "
                             "positive class listed second")

    def _derived(self, head, z, decision):
        p = self._array(self._readout.proba(z))
        if decision is None:
            decision = (p >= self._cut()).astype(np.int64)
        out = self._by_of("probability", np.stack([1.0 - p, p], axis=1), self.output.classes)
        out.update(self._one("class", self._names(decision)))
        out.update(self._one("decision", np.asarray(decision, dtype=np.int64)))
        return out


class MulticlassReader(BaseReader):
    """`p_<class>` the readout's distribution, `label` its argmax, `logit_<class>` the head."""

    def _derived(self, head, z, decision):
        p = self._array(self._readout.proba(z))
        out = self._by_of("probability", p, self.output.classes)
        out.update(self._one("class", self._names(self._array(self._readout.reduce(z)))))
        return out


class MultilabelReader(BaseReader):
    """`p_<label>` per label from the readout; `pred_<label>` is `p ≥ t_<label>` (the label's
    own threshold, else the record's, else 0.5)."""

    def _derived(self, head, z, decision):
        labels = self.output.classes
        p = self._array(self._readout.proba(z))
        flags = (p >= np.asarray([self._cut(label) for label in labels])).astype(np.int64)
        out = self._by_of("probability", p, labels)
        out.update(self._by_of("flag", flags, labels))
        return out


class OrdinalReader(BaseReader):
    """`p_<class>` the readout's class distribution (clipped at 0 and renormalised), `label`
    its cumulative count `Σ_j [σ(z_j) ≥ 0.5]`, `cum_<c_j>` = P(y > c_j) of that distribution
    (the cumulative probability itself for a rank-consistent head), `logit_cum_<c_j>` the
    head. A row whose probabilities needed clipping carries `CLIPPED`."""

    def _derived(self, head, z, decision):
        classes = self.output.classes
        p = self._array(self._readout.proba(z))
        above = p[:, ::-1].cumsum(axis=1)[:, ::-1][:, 1:]
        out = self._by_of("probability", p, classes)
        out.update(self._by_of("cumulative", above, classes[:-1]))
        out.update(self._one("class", self._names(self._array(self._readout.reduce(z)))))
        return out

    def warnings(self, head: HeadOutput) -> tuple:
        clipped = self._array(self._readout.clipped(_tensor(self._checked(head).raw)))
        return tuple((CLIPPED,) if flag else () for flag in clipped)


def _agree(output, readout: TaskReadout) -> None:
    """Refuse an output record the spec's readout does not read the same way."""
    kind = TASK_KIND.get(output.reading.value, output.reading.value)
    if kind != readout.kind.value or output.link != readout.link:
        raise ValueError(f"MANIFEST.output is a {output.reading.value} reading ({output.link} "
                         f"link) and the spec's readout reads a {readout.kind.value} head "
                         f"({readout.link} link): the manifest was written for another model")
    readout.check_head(output.head_out, "MANIFEST.output.head_out")
    unknown = [c for c in output.classes if c not in {str(k) for k in readout.task.labels.classes}]
    if unknown:
        raise ValueError(f"MANIFEST.output names class(es) {', '.join(unknown)} the spec's "
                         "task.classes does not list")


def _head_column(stored: str, field: str, head_out: int):
    """The head column a `compared` source names: `score` 0, `score_<k>` k, `row_score` None."""
    if stored == ROW_SCORE:
        return None
    match = STORED_SCORE.fullmatch(stored)
    if match is None:
        raise ValueError(f"MANIFEST.output compares {field!r} with {stored!r}, which is not a "
                         "column of vega/predictions-layout@1 (score, score_<k>, row_score)")
    k = int(match.group(1) or 0)
    if k >= head_out or (match.group(1) is None and head_out != 1):
        raise ValueError(f"MANIFEST.output compares {field!r} with {stored!r}, and head_out is "
                         f"{head_out}: the stored head is score_0…score_{head_out - 1} (`score` "
                         "for a one-output head)")
    return k


def _tensor(raw: np.ndarray) -> torch.Tensor:
    """The head as a tensor the readout reads: a copy (the `HeadOutput` is never written)."""
    return torch.tensor(raw, dtype=torch.float64)


def _stored_raw(stored: pd.DataFrame, head_out: int) -> np.ndarray:
    """The raw head from the stored layout: `score` for one output, else `score_0…`."""
    names = ["score"] if head_out == 1 and "score" in stored.columns else \
        [f"score_{k}" for k in range(head_out)]
    missing = [name for name in names if name not in stored.columns]
    if missing:
        raise ValueError(f"the stored predictions carry no {', '.join(missing)} column(s): "
                         "vega/predictions-layout@1 stores the raw head, every head_out column")
    return stored[names].to_numpy(dtype=np.float64)
