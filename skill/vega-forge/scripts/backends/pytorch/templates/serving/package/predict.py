"""
{{PACKAGE}}.{{MODULE_SERVING}}.predict — the {{PORT_PREDICTOR}} adapter (role: Predictor).

{{RESPONSIBILITY_PREDICTOR}}
Source: {{SOURCE_PREDICTOR}}

Status: **implemented** — rendered by /vega-pack from
`backends/pytorch/templates/serving/package/predict.py`, beside its support files (`records`,
`guards`, `samples`, `graphs`, `heads`, `values`): fix the template and re-pack, never this copy.

The Predictor is the bundle's front door: raw rows -> the Recipe -> the Model -> the answer, one
row out per row in and in the same order, for every reading (bundle-contract §2.1) and every
sample shape (§2.2). It is the one Predictor (BUILD-PLAN R-D15): the bundle's `predict.py`
builds it directly, `Predictor(model, transform, records, context=None)`, from the pieces the
other ports provide — the Model, the Recipe's `transform`, the serving records
(`records.ServingRecords`: MANIFEST.output, MANIFEST.sample, schema.json, the model version)
and, for a graph bundle, the reference graph (`graphs.GraphContext`).

  predict(df_raw, history=None)      the answer frame: exactly `MANIFEST.output.fields`, in
                                     their order (`graph_id` first at graph level, `answered`
                                     for a window, `warnings` last, as one text per row),
                                     indexed like `df_raw`
  predictions(df_raw, history=None)  the same answer as the domain's `Prediction` values: the
                                     class the reading decided, the score it compares, the
                                     feature-contract and model versions, the row's warnings
  predict_one(row, history=None)     one row's answer as plain values (NaN as None)
  info()                             the manifest, which is what this bundle is
  prepare(df_raw, history=None)      the one raw -> batch path (`samples.assembler_for`): what
                                     `predict` runs and what the inspection machinery explains
  inspector()                        the read-only view of this network (`inspection/`), built
                                     once over this Predictor and opened by its caller
  model, reader, records, context    read-only: the network, reading, records, reference graph

The network is held as the {{PORT_MODEL}} port and read through it alone — its `head` and its
`anatomy()` (identity, mode, device, one-class centre) —, so whatever implements it is served.

How one request is answered. The schema guard reads the raw rows (`guards.SchemaGuard`). The
shape's adapter prepares them (`Prepared`: the batch, `sample_of_row`, the `SampleIndex`, the
shape's notes per row) — a window row is answered iff its series has `length − 1` preceding
rows in the history and the request, a graph query receives messages from the reference graph
only, so no answer depends on the other rows sent. The network runs once over the batch without
gradients, in eval mode (the store loads it so; one in training mode is refused, never
switched); at node level it gathers `target_index` itself (R-A2), so `_forward` never gathers
and refuses a model whose output rows are not the samples. The reader of the reading
(`values.reader_for`, over the spec's `TaskReadout`, R-B7) turns the head into the answer's
fields; each request row takes its
sample's answer through `sample_of_row`, and an unanswered row keeps its fields empty (NaN, NA
for a decision, None for a name) with `answered` false. So does a row whose sample's head output
is not finite because its own numeric input is empty (the recipe leaves it NaN, and `NaN ≥ t` is
false: read as an answer it would be a confident negative class), with a warning naming the
input; when the hole is in the row's context (a lag whose history was not sent), the row keeps
the shape's answer but no decision is read from the output (its decided fields empty), with a
warning saying so. A row's warnings are the guard's, then the shape's, then the reading's own
(an ordinal row whose probabilities needed clipping).

It does not fit (every parameter it applies was fitted upstream on the train part and travels
in `fitted.json`, so new rows cannot quietly re-learn a scaling) and does not seed (inference is
deterministic). Nothing in a request writes to the model, the records or the reference graph:
concurrent requests share one Predictor. The one thing it builds after construction is its
inspector, once, under a lock, imported only then (a --no-inspect bundle still predicts).

Serving mode declared by the blueprint: {{SERVING_MODE}}

Effective Java: EJ-17 (the records and the context are frozen values the Predictor only reads),
EJ-18 (it is composed of its reader, its assembler and its guard, each chosen from a table),
EJ-49 (a refusal names the model, the record or the row it cannot answer), EJ-64 (the network
is held as its Model port, and the port takes the records value, never the manifest's dicts).
"""
from __future__ import annotations

import math
import threading
from dataclasses import replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping

import numpy as np
import pandas as pd
import torch

from ..domain import LabelSet, Prediction
from .guards import SchemaGuard
from .records import Reading, ServingRecords, thawed
from .samples import HeadOutput, Prepared, assembler_for
from .values import reader_for

if TYPE_CHECKING:                        # the port is a type here; ports import no code
    from ..ports import {{PORT_MODEL}} as _Model

__all__ = ["Predictor"]

# a row's warnings travel in the frame as one text, joined with this (the CSV answer's form)
WARNING_SEP = "; "
NO_FEATURES = ("the spec's feature contract lists no features; the column order the network "
               "was built against cannot be recovered, and guessing it from a frame would score "
               "plausible nonsense")
NO_ANATOMY = ("the model's anatomy() is None: it is not a staged network (a baseline), and the "
              "Predictor serves the bundle's Network")
NO_CENTRE = ("an anomaly-svdd answer is the distance to the model's centre, and this model "
             "carries {why}: this package predates vega v2.3's per-row score, or the checkpoint "
             "was not trained under the svdd objective; re-forge the campaign")
OTHER_SPEC = ("the model was built from spec {model}, and the serving records hold spec "
              "{records}: they are not one bundle's")
GATHERED = ("the network returned {got} output row(s) for {want} sample(s): a node-level network "
            "gathers target_index itself (R-A2) and the Predictor never gathers again — this "
            "model is not the bundle's Network")
TRAINING = ("the model is in training mode: dropout and batch statistics would make each answer "
            "depend on chance and on the other rows sent; the Predictor serves a network in eval "
            "mode (the ModelStore loads one so) and never switches the shared model")
NO_INSPECTION = ("this bundle ships without inspection (packed with --no-inspect): its "
                 "Predictor's module holds no inspection/ modules")
# the texts are the HTTP contract's data, not chrome: a caller reads them per row
NOT_FINITE = ("not answered: the network's output is not finite for this row ({why}), and a "
              "decision read from it would be a guess")
NOT_FINITE_INPUTS = "{columns} empty or non-finite"
NO_DECISION = ("the network's output is not finite for this row (an input it reads in the row's "
               "context is empty or non-finite): no decision is read from it")
DECIDED = frozenset({"class", "decision", "flag", "alarm"})     # the field kinds a decision fills


class Predictor:
    """Raw rows -> the answer, through the Recipe and the Model, for every reading and shape.

    Built once per bundle and shared by every request; it reads its collaborators and writes
    none. `model` is the network as its Model port; `records` the `ServingRecords` value
    `load` reads from the manifest and the schema; `context` a graph bundle's `GraphContext`.
    """

    def __init__(self, model: _Model, transform, records: ServingRecords, context=None):
        if not isinstance(records, ServingRecords):
            raise TypeError(f"records is a {type(records).__name__}; pass the ServingRecords "
                            "value (records.ServingRecords.of(spec, manifest, schema, "
                            "model_version))")
        _check_features(records.spec)
        _check_model(model, records)
        self._model, self._transform, self._records, self._context = model, transform, records, context
        self._reader = reader_for(records)
        self._assembler = assembler_for(records, transform, context)
        self._guard = SchemaGuard(records.schema)
        self._labels = LabelSet.from_spec(records.spec.to_dict())
        output = records.output
        self._kinds = MappingProxyType({f: output.field_kinds[f]["kind"] for f in output.fields})
        self._inspector, self._building = None, threading.Lock()

    # -- the port (ROLE_SIGNATURES["Predictor"]) -------------------------------------------
    def predict(self, df_raw: pd.DataFrame, history=None) -> pd.DataFrame:
        """The answer, one row per row of `df_raw`, in its order and with its index: a caller
        joins the result back to its own rows by position, so nothing here drops, filters or
        reorders. `history`: the raw rows before these (a window's context, a lag's past)."""
        return self._answer(df_raw, history)[0]

    def predictions(self, df_raw: pd.DataFrame, history=None) -> tuple:
        """The same answer as domain values: one `Prediction` per row, its class the reading's
        own decision read through the `LabelSet` (never `Prediction.of`'s default reading), its
        score what the round trip compares, the versions that produced it. An unanswered
        window row has no score and no class, and its warnings say why."""
        frame, notes = self._answer(df_raw, history)
        versions, done = self._records.versions, frame.attrs.get("answered")
        return tuple(self._prediction(row_id, {**rec, "answered": done[i]} if done else rec,
                                      notes[i], versions)
                     for i, (row_id, rec) in enumerate(zip(frame.index,
                                                           frame.to_dict("records"))))

    def predict_one(self, row: Mapping, history=None) -> dict:
        """One raw row's answer as plain values; a missing value (an unanswered row's) is
        None. `history`: a frame, or a list of raw rows, before this one."""
        if not isinstance(row, Mapping):
            raise TypeError(f"predict_one takes one raw row as a mapping of column -> value; "
                            f"got {type(row).__name__}")
        frame = self.predict(pd.DataFrame([dict(row)]), history)
        return {str(key): _plain(value) for key, value in frame.to_dict("records")[0].items()}

    # -- beside the port -------------------------------------------------------------------
    def info(self) -> dict:
        """The manifest, which is what this bundle is."""
        return thawed(self._records.manifest)

    def prepare(self, df_raw: pd.DataFrame, history=None) -> Prepared:
        """The one raw -> batch path: `df_raw` (and `history`) through the Recipe into the
        batch of the bundle's sample shape, with the sample answering each row."""
        return self._assembler.assemble(df_raw, _history(history))

    def inspector(self):
        """The read-only view of this Predictor's network (`inspection.NetworkInspector`
        over the model, the spec, this Predictor — its `prepare`, `predict` and `predictions` —,
        MANIFEST.output and the round-trip tolerance), built once: every call returns the same
        object. Its caller opens it on example rows (`predict.load_inspector`, the inspection
        route). Only a missing `inspection` package is a bundle without inspection
        (LookupError); a missing module inside it is a broken render, raised as it is."""
        with self._building:
            if self._inspector is None:
                try:
                    from .inspection.inspector import NetworkInspector
                except ModuleNotFoundError as exc:
                    if exc.name != f"{__package__}.inspection":
                        raise                  # a partial render, or a module's own import
                    raise LookupError(NO_INSPECTION) from exc
                manifest = self._records.manifest
                self._inspector = NetworkInspector(self._model, self._records.spec, self,
                                                   thawed(manifest.get("output") or {}),
                                                   _tolerance(manifest))
            return self._inspector

    @property
    def model(self) -> _Model:
        """The network the answers come from, as its Model port (never train or mutate it)."""
        return self._model

    @property
    def reader(self):
        """The reader of this bundle's reading (`values.reader_for`)."""
        return self._reader

    @property
    def context(self):
        """The reference graph a graph bundle answers through (`graphs.GraphContext`), else None."""
        return self._context

    @property
    def records(self) -> ServingRecords:
        """The serving records this Predictor answers by."""
        return self._records

    # -- inside ----------------------------------------------------------------------------
    def _answer(self, df_raw, history) -> tuple:
        """(the answer frame, one warnings tuple per row)."""
        if not isinstance(df_raw, pd.DataFrame):
            raise TypeError("predict takes a DataFrame of raw rows in the input schema; got "
                            f"{type(df_raw).__name__}")
        if not len(df_raw):
            return pd.DataFrame(columns=list(self._kinds), index=df_raw.index), ()
        guard = self._guard.warnings(df_raw)
        prepared = self.prepare(df_raw, history)
        return self._frame(df_raw.index, prepared, self._forward(prepared), guard, df_raw)

    def _forward(self, prepared: Prepared) -> HeadOutput:
        """The port's `head` over the prepared batch, read into the reading's `HeadOutput`
        (the raw head, and a per-row score where the reading has one); no sample, no network."""
        width = self._records.output.head_out
        if not prepared.n_samples:
            return HeadOutput(np.zeros((0, width)),
                              np.zeros(0) if self._reader.needs_row_score else None)
        anatomy = self._model.anatomy()
        if anatomy.training:
            raise RuntimeError(TRAINING)
        batch = _on(prepared.batch, anatomy.device)
        with torch.no_grad():
            out = self._model.head(batch)
            if int(out.shape[0]) != prepared.n_samples:
                raise RuntimeError(GATHERED.format(got=int(out.shape[0]),
                                                   want=prepared.n_samples))
            return self._reader.head(out, batch, anatomy)

    def _frame(self, index, prepared: Prepared, head: HeadOutput, guard, df_raw=None) -> tuple:
        """Each request row takes its sample's answer; the shape's fields beside them. A row
        whose sample's head output is not finite is not answered when its own numeric input is
        empty, and keeps the shape's answer without a decision when the hole is in its context
        (module docstring)."""
        at = np.asarray(prepared.sample_of_row, dtype=np.int64)
        lost = _not_finite(head)
        reached = (at >= 0) & lost[np.clip(at, 0, None)] if len(lost) else np.zeros(len(at), bool)
        empty = {int(i): self._empty_inputs(df_raw, int(i)) for i in np.flatnonzero(reached)}
        dropped = np.array([bool(empty.get(i)) for i in range(len(at))], dtype=bool)
        own = self._reader.warnings(head)
        notes = [tuple(guard[i]) + tuple(prepared.warnings[i])
                 + (tuple(own[s]) if s >= 0 else ()) for i, s in enumerate(at)]
        for i, columns in empty.items():
            notes[i] += (NOT_FINITE.format(why=NOT_FINITE_INPUTS.format(columns=columns)) if columns
                         else NO_DECISION,)
        notes = tuple(notes)
        taken = np.where(dropped, -1, at)
        answered = taken >= 0
        answers = _taken(self._reader.frame(head), taken, answered)
        held = reached & ~dropped                        # no decision read from a non-finite output
        for field, kind in self._kinds.items():
            if kind in DECIDED and field in answers.columns and held.any():
                col = answers[field]
                answers[field] = col.astype("boolean" if col.dtype.kind == "b" else "Int64").mask(held) \
                    if col.dtype.kind in "iub" or str(col.dtype) in ("Int64", "boolean") else col.astype(object).where(~held, None)
        shape = {"group": lambda: _groups(prepared.index, at), "answered": lambda: answered,
                 "warnings": lambda: [WARNING_SEP.join(n) for n in notes]}
        columns = {field: shape[kind]() if kind in shape else answers[field].array
                   for field, kind in self._kinds.items()}
        frame = pd.DataFrame(columns, index=index)
        # a shape whose record carries no `answered` field (a row) still has unanswered rows: the
        # frame says which, for the service's answer and the domain values
        frame.attrs["answered"] = [bool(a) for a in answered]
        return frame, notes

    def _empty_inputs(self, df_raw, i: int) -> str:
        """Row i's empty or non-finite numeric inputs, by name ("" when none is)."""
        inputs = (self._records.schema or {}).get("input") or ()
        names = [c["name"] for c in inputs if c.get("dtype") != "string"
                 and df_raw is not None and c["name"] in df_raw.columns]
        row = df_raw.iloc[i] if df_raw is not None and i < len(df_raw) else None
        return ", ".join(c for c in dict.fromkeys(names)
                         if row is not None and not np.isfinite(_float(row[c])))

    def _prediction(self, row_id, rec: Mapping, notes: tuple, versions) -> Prediction:
        score, label, probabilities = (), None, ()
        if bool(rec.get("answered", True)):
            k, reader = self._reader.class_index(rec), self._reader
            score, probabilities = reader.score_vector(rec), reader.probability_vector(rec)
            label = None if k is None else self._labels.at(k)
        return Prediction(row_id=row_id, score=score, label=label, probabilities=probabilities,
                          versions=replace(versions, warnings=tuple(notes)))


# --------------------------------------------------------------------------- helpers

def _check_features(spec) -> None:
    contract = spec.feature_contract
    if not any(c for c in (*contract.numeric_columns, *contract.nominal_columns)):
        raise ValueError(NO_FEATURES)


def _check_model(model: _Model, records: ServingRecords) -> None:
    """The model, read through its anatomy, is a staged network in eval mode, the records' (its
    spec digest) and, for the svdd reading, carries the centre its answer is a distance to."""
    anatomy = model.anatomy()
    if anatomy is None:
        raise ValueError(NO_ANATOMY)
    if anatomy.training:
        raise RuntimeError(TRAINING)
    if anatomy.spec_digest != records.spec.digest:
        raise ValueError(OTHER_SPEC.format(model=anatomy.spec_digest,
                                           records=records.spec.digest))
    if records.output.reading is not Reading.ANOMALY_SVDD:
        return
    if anatomy.centre is None:
        raise ValueError(NO_CENTRE.format(why="no svdd_center buffer"))
    if not anatomy.centre_set:
        raise ValueError(NO_CENTRE.format(why="a centre that was never set"))


def _tolerance(manifest) -> float | None:
    """MANIFEST.roundtrip.tolerance.value; None without one (the inspector's default)."""
    value = ((manifest.get("roundtrip") or {}).get("tolerance") or {}).get("value")
    return None if value is None else float(value)


def _history(value):
    """The history as a frame: a DataFrame, a list of raw rows, or None."""
    if value is None or isinstance(value, pd.DataFrame):
        return value
    if isinstance(value, (list, tuple)) and all(isinstance(r, Mapping) for r in value):
        return pd.DataFrame([dict(r) for r in value])
    raise TypeError(f"history is a {type(value).__name__}; pass a DataFrame of raw rows, or a "
                    "list of them as mappings")


def _on(batch: Mapping, device) -> dict:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}


def _not_finite(head: HeadOutput) -> np.ndarray:
    """Per sample: whether its head output (or per-row score) holds a non-finite number."""
    lost = ~np.isfinite(head.raw).all(axis=1) if head.raw.size else np.zeros(head.raw.shape[0], bool)
    if head.row_score is not None and len(head.row_score):
        lost = lost | ~np.isfinite(np.asarray(head.row_score, dtype=np.float64))
    return lost


def _float(value) -> float:
    """A raw cell as a float; anything unreadable as nan."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _taken(frame: pd.DataFrame, at: np.ndarray, answered: np.ndarray) -> pd.DataFrame:
    """Row i is sample `at[i]`'s answer; an unanswered row (−1) keeps every field empty: NaN
    for a number, NA for a decision (which stays an integer), None for a name."""
    taken = frame.reindex(at).reset_index(drop=True)
    if answered.all():
        return taken
    for name in frame.columns:
        if frame[name].dtype.kind in "iu":
            taken[name] = frame[name].astype("Int64").reindex(at).reset_index(drop=True)
        elif frame[name].dtype == object:
            taken[name] = taken[name].where(answered, None)
    return taken


def _groups(index, at: np.ndarray) -> np.ndarray:
    """The graph each row's answer belongs to, as text (the `group` field, graph level)."""
    graphs = index.graphs or ()
    return np.asarray([str(graphs[s]) if 0 <= s < len(graphs) else None for s in at],
                      dtype=object)


def _plain(value):
    """A frame cell as a plain Python value; NaN and NA as None."""
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or value is pd.NA or (isinstance(value, float) and math.isnan(value)):
        return None
    return value
