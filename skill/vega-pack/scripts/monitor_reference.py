"""
monitor_reference — the numbers a monitor reference holds, measured on the train and valid parts
and never on the test part (bundle-contract §7.5).

    MonitorInputs.of(ctx, predictor, sample) -> MonitorInputs
    MonitorRender(output, schema, inputs).render(plan) -> vega/monitor-reference@2   PackExit
    MonitorRender(output, schema, inputs).measure(check) -> the check with its reference

`psi` and `rate` are read on the train part's raw rows; the unseen rate on the valid part,
through the bundle's own Recipe (the one definition of "unseen" it carries, which the Monitor
asks again at serving time). Every answer reference is read on the valid part through the
bundle's own reader, `reader.from_stored(predictions.valid.parquet)`, or — where the stored rows
cannot answer a check (no reader, no file, no stored per-row score or truth) — the Predictor
over the valid part's raw rows, its unanswered rows left out. A graph-level store holds one row
per graph while the Monitor reads one per raw row, so a graph-level bundle's answers are the
Predictor's, weighted per row as served. The truth is the stored `y` / `y_<k>` (a class index
read through `task.classes`) or the raw target decoded as the schema decodes it; a forecast step
h's target is the truth h rows later in the same series, the Monitor's rule. A numeric check's
outer bin edges are open, written `null` (strict JSON).
"""
from __future__ import annotations

import importlib.util
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from pack_context import PackExit
from vega_pack_core import read_parquet

__all__ = ["NAMED", "OPS", "REFERENCE_SCHEMA", "MonitorInputs", "MonitorRender", "answered",
           "category_text", "field_of", "horizon_of", "plan_predicate", "resolve_check"]

REFERENCE_SCHEMA = "vega/monitor-reference@2"
STORED_TRUTH = re.compile(r"y(?:_\d+)?")
OPS = {"==": np.equal, "!=": np.not_equal, "<": np.less, "<=": np.less_equal,
       ">": np.greater, ">=": np.greater_equal}
# the field an answer check reads when the plan names none, and the key naming its classes
DEFAULT_FIELD = {"output_psi": "score_field", "residual": "score_field",
                 "prediction_mix": "class", "class_recall": "class", "alert_rate": "alarm"}
NAMED = {"class_recall": "classes", "prediction_mix": "classes", "label_rate": "labels"}


@dataclass(frozen=True, eq=False)
class MonitorInputs:
    """What the references are measured on: the train and valid parts' raw rows, the valid
    part's stored predictions, and the bundle's own Predictor and Recipe."""

    train: Any
    valid: Any
    stored_valid: Any            # predictions.valid.parquet (vega/predictions-layout@1) or None
    predictor: Any               # the loaded bundle's Predictor (its reader, its predict)
    recipe: Any                  # () -> the bundle's Recipe transform, loaded on first use
    classes: tuple = ()          # task.classes as text, in the head's order
    label: Any = None            # the raw label column (left out of what the Recipe reads)
    decode: Any = None           # schema.json label.decode: a raw spelling -> its class
    ignored: tuple = ()          # schema.json ignored columns, left out of a predict call
    sample: Any = None           # MANIFEST.sample

    @staticmethod
    def of(ctx, predictor, sample) -> "MonitorInputs":
        campaign = ctx.campaign
        stored = campaign.run_dir / "predictions.valid.parquet"
        transformer = ctx.port("FeatureTransformer").get("file") or "features/preprocess.py"
        return MonitorInputs(
            train=campaign.part_rows("train"), valid=campaign.part_rows("valid"),
            stored_valid=read_parquet(stored) if stored.is_file() else None,
            predictor=predictor, recipe=lambda: bundle_transform(ctx.paths.pkg / transformer),
            classes=tuple(str(c) for c in (ctx.spec.get("task") or {}).get("classes") or ()),
            label=campaign.label, decode=dict((ctx.schema.get("label") or {}).get("decode") or {}),
            ignored=tuple(ctx.schema.get("ignored") or ()), sample=dict(sample or {}))


class MonitorRender:
    """The plan's checks with their references measured: `render(plan)` is the reference
    document; a reference that cannot be measured is a `PackExit` naming the check."""

    def __init__(self, output, schema, inputs: MonitorInputs):
        self._output, self._schema, self._in = dict(output), dict(schema), inputs
        self._stored_read, self._stored_answers, self._predicted_answers = False, None, None

    def render(self, plan) -> dict:
        return {"schema": REFERENCE_SCHEMA, "reading": self._output["reading"],
                "checks": [self.measure(c) for c in plan.get("checks") or []],
                "note": plan.get("note") or ""}

    def measure(self, check) -> dict:
        """One check as the reference holds it: resolved, with its reference measured."""
        check = resolve_check(check, self._output)
        return {**check, **getattr(self, "_" + check["kind"])(dict(check))}

    # -- on the train part -------------------------------------------------------------------
    def _psi(self, c):
        return {**_psi_reference(c, self._in.train[c["column"]]), **_part("train", self._in.train)}

    def _rate(self, c):
        mask = plan_predicate(c["predicate"], self._in.train)
        return {"reference": round(float(np.mean(mask)), 10), **_part("train", self._in.train)}

    # -- on the valid part -------------------------------------------------------------------
    def _unseen_rate(self, c):
        valid = self._valid_rows(c)
        query = valid.drop(columns=[x for x in (self._in.label,) if x and x in valid.columns])
        # an empty history: a recipe that lags reads its lags as starting with the part
        _shaped, unseen = self._in.recipe()(query, history=valid.iloc[:0], with_unseen=True)
        return {"reference": round(float(np.mean([c["column"] in u for u in unseen])), 10),
                **_part("valid", valid)}

    def _class_recall(self, c):
        answers = self._answers(c, truth_at=0)
        truth = answers.classes(self._in.classes, c)
        predicted = answers.field(c["field"], c).astype(str).to_numpy()
        reference, support = {}, {}
        for name in c["classes"]:
            support[name] = int((truth == name).sum())
            hits = int(((truth == name) & (predicted == name)).sum())
            reference[name] = round(hits / support[name], 6) if support[name] else None
        return {"label_column": (self._output.get("truth") or {}).get("column"),
                "reference": reference,
                "reference_support": support, **_part("valid", answers.frame)}

    def _output_psi(self, c):
        answers = self._answers(c)
        values = pd.to_numeric(answers.field(c["field"], c), errors="coerce")
        if not np.isfinite(values.to_numpy(dtype=float)).any():
            raise PackExit(3, f"check {c.get('id')}: the field {c['field']!r} answered no "
                              "finite value on the valid part")
        return {**_psi_reference(c, values), **_part("valid", answers.frame)}

    def _prediction_mix(self, c):
        answers = self._answers(c)
        classes = [str(k) for k in (c.get("classes") or self._output.get("classes") or ())]
        values = answers.field(c["field"], c).astype(str)
        shares = [float((values == k).mean()) for k in classes]
        return {"categorical": True, "bins": classes,
                "reference": [round(s, 8) for s in shares] + [round(max(0.0, 1 - sum(shares)), 8)],
                **_part("valid", answers.frame)}

    def _label_rate(self, c):
        answers = self._answers(c)
        fields = {label: field_of(self._output, "flag", label) for label in c["labels"]}
        return {"fields": fields, **_part("valid", answers.frame),
                "reference": {label: round(float(_flags(answers.field(f, c)).mean()), 10)
                              for label, f in fields.items()}}

    def _alert_rate(self, c):
        answers = self._answers(c)
        return {"direction": c.get("direction", "up"), **_part("valid", answers.frame),
                "reference": round(float(_flags(answers.field(c["field"], c)).mean()), 10)}

    def _residual(self, c):
        offset = horizon_of(self._output["field_kinds"][c["field"]])
        answers = self._answers(c, truth_at=offset)
        values = pd.to_numeric(answers.field(c["field"], c), errors="coerce").to_numpy(float)
        error = np.abs(answers.target(offset, c) - values)
        error = error[np.isfinite(error)]
        if not len(error):
            raise PackExit(3, f"check {c.get('id')}: no valid row has both {c['field']!r} and "
                              "its target; the residual reference cannot be measured")
        sample = self._in.sample or {}
        return {"target_column": (self._output.get("truth") or {}).get("column"),
                "target_offset": offset,
                "series_by": sample.get("series_by"), "order_by": sample.get("order_by"),
                "reference": {"mae": round(float(error.mean()), 10),
                              "p95": round(float(np.quantile(error, 0.95)), 10)},
                "reference_part": "valid", "reference_rows": int(len(error))}

    # -- the valid part's answers ------------------------------------------------------------
    def _valid_rows(self, c):
        if not len(self._in.valid):
            raise PackExit(2, f"check {c.get('id')}: fitted.json records no valid rows; the "
                              f"{c['kind']} reference is measured on the valid part")
        return self._in.valid

    def _answers(self, c, truth_at=None) -> "_ValidAnswers":
        """The valid part's answers: the stored ones read by the bundle's reader when they can
        answer this check (`truth_at`: the forecast step whose truth it reads, 0 for the row's
        own), else the Predictor's over the valid part's raw rows — each computed once."""
        self._valid_rows(c)
        if not self._stored_read:
            self._stored_read, self._stored_answers = True, self._stored()
        stored = self._stored_answers
        if stored is not None and (truth_at is None or stored.truth_column(truth_at)):
            return stored
        if self._predicted_answers is None:
            self._predicted_answers = self._predicted()
        answers = self._predicted_answers
        if not len(answers.frame):
            raise PackExit(3, f"check {c.get('id')}: the bundle answered no row of the valid "
                              f"part ({answers.source}); the {c['kind']} reference is measured "
                              "on its answers")
        return answers

    def _stored(self):
        reader, stored = getattr(self._in.predictor, "reader", None), self._in.stored_valid
        sample = self._in.sample or {}
        per_graph = (sample.get("shape"), sample.get("level")) == ("graph", "graph")
        if per_graph or reader is None or stored is None or not len(stored) or \
                (reader.needs_row_score and "row_score" not in stored.columns):
            return None
        try:
            frame = reader.from_stored(stored).reset_index(drop=True)
        except (KeyError, ValueError) as exc:
            raise PackExit(3, f"the bundle's reader cannot read predictions.valid.parquet: "
                              f"{exc}") from exc
        truth = stored[[k for k in stored.columns if STORED_TRUTH.fullmatch(str(k))]]
        return _ValidAnswers(frame, truth.reset_index(drop=True), "stored")

    def _predicted(self):
        valid = self._in.valid.reset_index(drop=True)
        rows = valid.drop(columns=[c for c in self._in.ignored if c in valid.columns])
        try:
            frame = self._in.predictor.predict(rows).reset_index(drop=True)
        except Exception as exc:                                      # noqa: BLE001
            raise PackExit(3, f"the bundle's Predictor does not answer the valid part: "
                              f"{type(exc).__name__}: {exc}") from exc
        kept = answered(frame)
        truth = _raw_truth(valid, self._output, self._in)
        return _ValidAnswers(frame.loc[kept].reset_index(drop=True),
                             truth.loc[kept].reset_index(drop=True), "predicted")


@dataclass(frozen=True, eq=False)
class _ValidAnswers:
    """The valid part's answers (the answer fields, one row per answered sample) and the truth
    aligned to them, named as the stored layout names it (`y`, `y_<h-1>` for step h)."""

    frame: Any
    truth: Any
    source: str

    def field(self, name, check):
        if name not in self.frame.columns:
            raise PackExit(3, f"check {check.get('id')}: the bundle's answer carries no field "
                              f"{name!r} ({self.source} valid predictions); the reader answers "
                              "every field of MANIFEST.output.fields")
        return self.frame[name]

    def classes(self, names, check):
        """The truth as class names: a stored index read through `task.classes`, a raw
        target as the schema decoded it."""
        values = self.truth[self._column(0, check)]
        if self.source == "stored" and names:
            index = pd.to_numeric(values, errors="coerce")
            return np.asarray([names[int(k)] if k == k and 0 <= int(k) < len(names) else None
                               for k in index], dtype=object)
        return values.astype(str).to_numpy()

    def truth_column(self, offset):
        """The truth column a field of step `offset` answers, or None: `y` (or `y_0`) for the
        row's own target (0), `y_<h-1>` for step h — or a store's one-column `y` for step 1."""
        names = ("y", "y_0") if not offset else \
            (f"y_{offset - 1}",) + (("y",) if offset == 1 and self.source == "stored" else ())
        return next((n for n in names if n in self.truth.columns), None)

    def target(self, offset, check):
        return pd.to_numeric(self.truth[self._column(offset, check)],
                             errors="coerce").to_numpy(dtype=float)

    def _column(self, offset, check):
        column = self.truth_column(offset)
        if column is None:
            raise PackExit(2, f"check {check.get('id')}: the valid part carries no truth "
                              + (f"for step {offset}" if offset else "for its rows"))
        return column


def _raw_truth(valid, output, inputs) -> pd.DataFrame:
    """The raw target of every valid row, named as the stored layout names it: `y` for the row
    itself, `y_<h-1>` for forecast step h (the truth h rows later in the same series)."""
    truth = output.get("truth") or {}
    if truth.get("column") not in valid.columns:
        return pd.DataFrame(index=valid.index)
    raw = valid[truth["column"]]
    if truth.get("type") == "class":
        decode = inputs.decode or {}
        return pd.DataFrame({"y": category_text(raw).map(lambda v: decode.get(v, v))})
    out = pd.DataFrame({"y": pd.to_numeric(raw, errors="coerce")})
    sample = inputs.sample or {}
    for h in output.get("horizons") or ():
        out[f"y_{h - 1}"] = _shifted(out["y"], valid, int(h), sample)
    return out


def _shifted(truth, frame, offset, sample):
    series = sample.get("series_by") if sample.get("series_by") in frame.columns else None
    keys = [c for c in (series, sample.get("order_by")) if c and c in frame.columns]
    ordered = frame.sort_values(keys, kind="mergesort") if keys else frame
    later = truth.loc[ordered.index]
    later = (later.groupby(ordered[series].to_numpy()).shift(-offset) if series
             else later.shift(-offset))
    return later.reindex(frame.index)


def answered(frame) -> np.ndarray:
    """Which rows of a Predictor frame were answered: all but those whose `answered` is an
    explicit false (a window without its full context)."""
    if "answered" not in frame.columns:
        return np.ones(len(frame), dtype=bool)
    return np.asarray([not (isinstance(v, (bool, np.bool_)) and not v)
                       for v in frame["answered"].tolist()], dtype=bool)


def horizon_of(kind) -> int:
    """The forecast step a field answers: `of` of a `horizon` field (h, or "h/q"), else 0."""
    if (kind or {}).get("kind") != "horizon":
        return 0
    return int(str(kind.get("of")).split("/")[0])


def field_of(output, kind, of=None):
    """The first field of `kind` (and `of`) in the output record, or None."""
    return next((f for f, k in (output.get("field_kinds") or {}).items()
                 if k.get("kind") == kind and (of is None or str(k.get("of")) == str(of))), None)


def resolve_check(check, output) -> dict:
    """A check as it is measured: the default field of its kind where the plan names none
    (`output_psi` and `residual` the record's `score_field`, `prediction_mix` and
    `class_recall` the decided class, `alert_rate` the alarm), its classes or labels as text."""
    check, kind = dict(check), check.get("kind")
    default = DEFAULT_FIELD.get(kind)
    if default and not check.get("field"):
        check["field"] = output.get("score_field") if default == "score_field" else \
            field_of(output, default)
    if isinstance(check.get(NAMED.get(kind)), list):
        check[NAMED[kind]] = [str(name) for name in check[NAMED[kind]]]
    # every served reading carries a title (the bundle's own monitor test): a plan's, else this
    subject = check.get("column") or check.get("field") or check.get("name") or check.get("id")
    check["title"] = check.get("title") or f"{str(kind).replace('_', ' ')} of {subject}"
    return check


def _part(name, frame) -> dict:
    return {"reference_part": name, "reference_rows": int(len(frame))}


def _flags(values):
    return pd.to_numeric(values, errors="coerce").fillna(0).to_numpy(dtype=float) >= 0.5


def _psi_reference(c, col) -> dict:
    """Categorical: the `top_k` categories' shares and the rest. Numeric: `bins_n` quantile
    bins over the finite values, the outer edges open — written `null`, so the reference stays
    strict JSON (the Monitor reads a `null` outer edge as ∓∞)."""
    if c.get("categorical"):
        cats = category_text(col)
        top = cats.value_counts().head(int(c.get("top_k", 20)))
        ref = (top / max(len(cats), 1)).tolist()
        return {"bins": top.index.tolist(),
                "reference": [round(x, 8) for x in ref + [max(0.0, 1.0 - sum(ref))]]}
    values = pd.to_numeric(col, errors="coerce")
    finite = values[np.isfinite(values.to_numpy(dtype=float))]
    if not len(finite):
        raise PackExit(2, f"check {c.get('id')}: {c.get('column') or c.get('field')!r} holds no "
                          "finite value on its reference part; no bins can be drawn")
    edges = _edges(np.unique(np.quantile(finite, np.linspace(0, 1, int(c.get("bins_n", 10)) + 1))))
    counts = np.histogram(values.dropna(), bins=edges)[0]
    bins = [None, *(float(e) for e in edges[1:-1]), None]
    return {"bins": bins, "reference": (counts / max(counts.sum(), 1)).round(8).tolist()}


def _edges(q):
    """The quantiles as bin edges, the outer ones open: the inner quantiles as they are (v2.2's
    rule), the midpoint of two distinct values, or the one value of a constant column."""
    inner = q[1:-1] if len(q) > 2 else [(q[0] + q[1]) / 2] if len(q) == 2 else [q[0]]
    return np.concatenate([[-np.inf], np.asarray(inner, dtype=float), [np.inf]])


def category_text(series):
    """As the monitor's PSI bins read a category (whole floats as integers)."""
    if pd.api.types.is_float_dtype(series) and bool((series.dropna() % 1 == 0).all()):
        return series.astype("Int64").astype(str)
    return series.astype(str)


def plan_predicate(spec, frame):
    """A `rate` check's predicate over a frame (the plan's `{"column", "op", "value"}`, or
    `{"any": […]}` / `{"all": […]}` of them), as the Monitor reads it."""
    if "any" in spec or "all" in spec:
        parts = [plan_predicate(x, frame) for x in (spec.get("any") or spec.get("all"))]
        return (np.logical_or if "any" in spec else np.logical_and).reduce(parts)
    values = pd.to_numeric(frame[spec["column"]], errors="coerce").to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        return np.nan_to_num(OPS[spec["op"]](values, float(spec["value"])), nan=0).astype(bool)


def bundle_transform(path):
    """The bundle's emitted Recipe adapter, imported by file so it reads the bundle's
    fitted.json."""
    spec = importlib.util.spec_from_file_location("_bundle_preprocess", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.transform
