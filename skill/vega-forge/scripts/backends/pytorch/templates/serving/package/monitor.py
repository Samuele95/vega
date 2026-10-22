"""
{{PACKAGE}}.{{MONITOR_MODULE}}.monitor — the {{PORT_MONITOR}} adapter (role: Monitor).

{{RESPONSIBILITY_MONITOR}}
Source: {{SOURCE_MONITOR}}

Status: **implemented** — completed by /vega-pack from `pipeline.json.monitoring[]` and the
bundle's reading (`MANIFEST.output`). Emitted from the backend's template
`templates/serving/package/monitor.py`: a defect is fixed there and the bundle re-packed.

It watches what the analysis found fragile, plus the model's own answer: every check of
`monitor-reference.json` (`vega/monitor-reference@2`) names the concern it answers (a blueprint
tag such as `[analysis:F3]`, or `[vega-pack:output-drift]` for the check pack adds). References
were measured on the train part (inputs) and the valid part (the unseen rate, every answer
reference), never on the test part. On the inputs: `psi`, `rate` (the share meeting a
predicate, limits at `watch_z` / `alert_z` standard errors, one-sided with `direction`),
`unseen_rate` (the share in which the Recipe port met a category it was never fitted on). On
the answers — `predictions` is the Predictor's frame, one row per window row, a field read by
name (`check["field"]`): `output_psi`, `prediction_mix` (the decided classes' shares as a PSI),
`label_rate` (per label, the share flagged), `alert_rate` (`up` by default). Against a truth the
window carries: `class_recall`, and `residual` (the MAE of a point field against its target as
a ratio to the valid part's; forecast step h's target is the truth h rows later in its series).

Rows the Predictor answered `false` (a window without its full context) are left out before any
reading, and every reading says how many were. `check()` is the port: the readings that moved
(`watch` or `alert`), an empty tuple — never `None` — when none did; `report()` returns every
reading. A monitor reports and never gates.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..domain import DriftReading, ReadingProvenance

__all__ = ["Monitor"]

SCHEMAS = ("vega/monitor-reference@1", "vega/monitor-reference@2")
MOVED = ("watch", "alert")
# the collaborator a kind cannot be measured without: a construction error, never a note
NEEDS = {"unseen_rate": "recipe", "class_recall": "labels"}
# the fewest rows a kind reads a window on; a check's own `min_rows` overrides it
MIN_ROWS = {"psi": 500, "rate": 100, "unseen_rate": 100, "output_psi": 500,
            "prediction_mix": 500, "label_rate": 100, "alert_rate": 100, "residual": 100}
PSI_LIMITS = {"watch": 0.1, "alert": 0.25}
Z_LIMITS = {"watch": 2.0, "alert": 3.0}
RATIO_LIMITS = {"watch": 1.25, "alert": 1.5}
OPS = {"==": np.equal, "!=": np.not_equal, "<": np.less, "<=": np.less_equal,
       ">": np.greater, ">=": np.greater_equal}


class Monitor:
    """The DriftWatch adapter over one reference, built once and applied per window."""

    def __init__(self, reference: dict, *, recipe, labels):
        schema = (reference or {}).get("schema")
        if schema is not None and schema not in SCHEMAS:
            raise ValueError(f"the monitor reads {' or '.join(SCHEMAS)}; this reference is "
                             f"{schema!r}")
        self._checks = tuple(dict(c) for c in (reference or {}).get("checks") or ())
        if not self._checks:
            raise ValueError("the monitor reference carries no checks; a monitor that watches "
                             "nothing would report that nothing moved")
        unknown = sorted({repr(c.get("kind")) for c in self._checks if c.get("kind") not in KINDS})
        if unknown:
            raise ValueError(f"the monitor reference names unknown check kind(s) "
                             f"{', '.join(unknown)}; this monitor measures {', '.join(KINDS)}")
        # a missing collaborator is a construction error, not an "unmeasured" note per window
        # that blames the window's columns
        given = {"recipe": recipe, "labels": labels}
        missing = sorted({f"{need} (check {c.get('id')}, {c['kind']})" for c in self._checks
                          for need in [NEEDS.get(c["kind"])] if need and given[need] is None})
        if missing:
            raise ValueError("the monitor cannot measure its checks without "
                             + ", ".join(missing))
        self._recipe = recipe            # the Recipe port's transform, for unseen categories
        self._labels = labels            # the LabelSet the head was trained against

    # -- the port ----------------------------------------------------------------------------
    def check(self, batch, predictions=None) -> tuple:
        """The readings that moved in this window; an empty tuple when none did."""
        return tuple(r for r in self.report(batch, predictions) if r.verdict in MOVED)

    def report(self, batch, predictions=None) -> tuple:
        """Every reading for this window, whatever its verdict."""
        window = self._window(batch, predictions)
        return tuple(_reading(c, measured, window) for c in self._checks
                     for measured in KINDS[c["kind"]](c, window))

    # -- inside ------------------------------------------------------------------------------
    def _window(self, batch, predictions) -> "_Window":
        if not isinstance(batch, pd.DataFrame):
            raise TypeError("the monitor reads a DataFrame of raw rows in the input schema; "
                            f"got {type(batch).__name__}")
        if predictions is not None and not isinstance(predictions, pd.DataFrame):
            raise TypeError("the monitor reads the Predictor's frame (`predict(window)`) as its "
                            f"predictions; got {type(predictions).__name__}")
        if predictions is not None and len(predictions) != len(batch):
            raise ValueError(f"{len(predictions)} prediction row(s) for {len(batch)} window "
                             "row(s): the Predictor answers one row per row given")
        answers = None if predictions is None else predictions.reset_index(drop=True)
        # only an explicit false leaves a row out: `answered` defaults to true
        kept = np.ones(len(batch), dtype=bool) if answers is None or \
            "answered" not in answers.columns else np.asarray(
                [not (isinstance(v, (bool, np.bool_)) and not v)
                 for v in answers["answered"].tolist()], dtype=bool)
        return _Window(batch.reset_index(drop=True), answers, kept, self._recipe, self._labels)


@dataclass(frozen=True, eq=False)
class _Window:
    """One window as the checks read it: its rows, the Predictor's frame aligned to them by
    position, which rows were answered, and the two collaborators."""

    batch: pd.DataFrame
    answers: Any
    kept: np.ndarray
    recipe: Any
    labels: Any

    @property
    def rows(self) -> int:
        return int(self.kept.sum())

    def note(self):
        dropped = len(self.kept) - self.rows
        return (f"{dropped} row(s) answered false (a window without its full context) left "
                "out" if dropped else None)

    def inputs(self) -> pd.DataFrame:
        return self.batch.loc[self.kept]

    def answer(self, field, kept=True):
        """The Predictor's `field`, over the answered rows (every row with `kept=False`);
        None when no predictions came with the window or they carry no such field."""
        if self.answers is None or field not in self.answers.columns:
            return None
        return self.answers[field].loc[self.kept] if kept else self.answers[field]


@dataclass(frozen=True)
class _Measured:
    column: Any
    measure: str
    value: Any
    verdict: str
    note: Any = None


def _reading(check, measured, window):
    # the check's human title travels with the reading, so a consumer never reads the
    # reference file's layout to label what the port measured
    note = "; ".join(n for n in (measured.note, window.note()) if n)
    origin = ReadingProvenance(check=str(check.get("id")), concern=check.get("concern"),
                               title=check.get("title"), window_rows=window.rows,
                               note=note or None)
    return DriftReading(column=measured.column, measure=measured.measure, value=measured.value,
                        verdict=measured.verdict, window=origin)


def _unmeasured(column, measure, why):
    return _Measured(column, measure, None, "unmeasured", why)


def _min_rows(check) -> int:
    return int(check.get("min_rows", MIN_ROWS.get(check["kind"], 1)))


def _verdict(value, limits, check, suffix=""):
    """`alert` / `watch` / `ok` against the check's limits (`watch<suffix>`, `alert<suffix>`),
    the kind's defaults where the check sets none."""
    watch = float(check.get("watch" + suffix, limits["watch"]))
    alert = float(check.get("alert" + suffix, limits["alert"]))
    return "alert" if value >= alert else "watch" if value >= watch else "ok"


# -- the kinds: each `_kind(check, window) -> [_Measured, …]` ---------------------------------

def _psi(check, window):
    column, rows, need = check["column"], window.inputs(), _min_rows(check)
    if column not in rows.columns or len(rows) < need:
        return [_unmeasured(column, "psi", f"needs column {column!r} and at least {need} rows")]
    values = _as_category(rows[column]) if check.get("categorical") else rows[column]
    return [_psi_of(check, column, values)]


def _rate(check, window):
    name = check.get("name") or check.get("column") or check.get("id")
    return [_rate_of(check, name, _predicate(check["predicate"], window.inputs()))]


def _unseen(check, window):
    column, rows = check["column"], window.inputs()
    name = check.get("name") or column
    if column not in rows.columns:
        return [_rate_of(check, name, None)]
    try:
        # an empty history: a recipe that lags reads its lags as starting with the window
        _shaped, unseen = window.recipe(rows, history=rows.iloc[:0], with_unseen=True)
    except (KeyError, ValueError) as exc:
        return [_unmeasured(name, "rate", f"the Recipe could not read the window: {exc}")]
    return [_rate_of(check, name, np.asarray([column in cols for cols in unseen], dtype=bool))]


def _recall(check, window):
    classes = [str(k) for k in check.get("classes") or ()]
    column = check.get("label_column")
    predicted = window.answer(check.get("field") or "label")
    if column not in window.batch.columns or predicted is None:
        return [_unmeasured(k, "recall", "needs the true label in the window and the "
                                         "predictions") for k in classes]
    truth = window.inputs()[column].astype(str).to_numpy()
    predicted = predicted.astype(str).to_numpy()
    # the LabelSet owns "is this a class": recall of class k needs the rows whose truth is k,
    # not a window free of foreign labels — those rows are left out, and said so
    admitted = np.isin(truth, [str(c) for c in window.labels.classes])
    foreign = sorted(set(truth[~admitted].tolist()))
    dropped = (f"{int((~admitted).sum())} row(s) with labels the spec does not list left out: "
               f"{foreign[:5]}" if foreign else None)
    return [_recall_of(check, k, (truth[admitted], predicted[admitted]), dropped)
            for k in classes]


def _recall_of(check, name, pair, dropped):
    truth, predicted = pair
    support = int((truth == name).sum())
    if not support:
        return _unmeasured(name, "recall", "; ".join(
            n for n in ("no row of this class in the window", dropped) if n))
    recall = float(((truth == name) & (predicted == name)).sum() / support)
    ref = (check.get("reference") or {}).get(name)
    # the reference is stored at six decimals: read the window's recall at the same precision,
    # so a window that recalls exactly what valid did (30/31 against 0.967742) is not a drop
    verdict = "watch" if ref is not None and round(recall, 6) < round(float(ref), 6) else "ok"
    note = "; ".join(n for n in (f"support {support}; valid-part recall {ref}", dropped) if n)
    return _Measured(name, "recall", round(recall, 6), verdict, note)


def _output_psi(check, window):
    field, need = check["field"], _min_rows(check)
    values = window.answer(field)
    if values is None or len(values) < need:
        return [_unmeasured(field, "psi", f"needs the predictions' {field!r} and at least "
                                          f"{need} answered rows")]
    return [_psi_of(check, field, pd.to_numeric(values, errors="coerce"))]


def _prediction_mix(check, window):
    field, need = check.get("field") or "label", _min_rows(check)
    values = window.answer(field)
    if values is None or len(values) < need:
        return [_unmeasured(field, "psi", f"needs the predictions' {field!r} and at least "
                                          f"{need} answered rows")]
    return [_psi_of(dict(check, categorical=True), field, values.astype(str))]


def _label_rate(check, window):
    reference = check.get("reference") or {}
    out = []
    for label, field in (check.get("fields") or {}).items():
        flags = window.answer(field)
        out.append(_rate_of(dict(check, reference=reference.get(label)), label,
                            None if flags is None else _flags(flags)))
    return out


def _alert_rate(check, window):
    field = check.get("field") or "prediction"
    flags = window.answer(field)
    return [_rate_of({"direction": "up", **check}, field,
                     None if flags is None else _flags(flags))]


def _residual(check, window):
    field, need = check["field"], _min_rows(check)
    values, target = window.answer(field, kept=False), _target(check, window.batch)
    if values is None or target is None:
        return [_unmeasured(field, "mae", f"needs the target column "
                                          f"{check.get('target_column')!r} in the window and "
                                          f"the predictions' {field!r}")]
    error = np.abs(target - pd.to_numeric(values, errors="coerce").to_numpy(dtype=float))
    error = error[window.kept]
    error = error[np.isfinite(error)]
    if len(error) < need:
        return [_unmeasured(field, "mae", f"needs at least {need} answered rows with a target")]
    ref = check.get("reference") or {}
    mae, ref_mae, p95 = float(error.mean()), float(ref["mae"]), float(ref["p95"])
    ratio = mae / ref_mae if ref_mae > 0 else (math.inf if mae > 0 else 1.0)
    above = float(np.mean(error > p95))
    return [_Measured(field, "mae", round(mae, 6), _verdict(ratio, RATIO_LIMITS, check, "_ratio"),
                      f"reference MAE {ref_mae:.6g} (valid part), ratio {ratio:.3f}; "
                      f"{above:.1%} of rows above the valid part's p95 {p95:.6g}")]


# -- the measures the kinds share ------------------------------------------------------------

def _psi_of(check, name, values):
    """The population stability of `values` against the check's bins and reference shares:
    categorical bins (one per name, plus the rest), or numeric edges."""
    bins, ref = list(check["bins"]), np.asarray(check["reference"], dtype=float)
    if check.get("categorical"):
        known = values.isin(bins)
        counts = [int((values == b).sum()) for b in bins] + [int((~known).sum())]
    else:
        numeric = pd.to_numeric(values, errors="coerce").dropna()
        counts = np.histogram(numeric, bins=_edges(bins))[0].tolist()
    current = np.asarray(counts, dtype=float) / max(sum(counts), 1)
    eps = float(check.get("epsilon", 1e-4))
    psi = float(np.sum((current - ref) * np.log((current + eps) / (ref + eps))))
    limits = {**PSI_LIMITS, **(check.get("thresholds") or {})}
    return _Measured(name, "psi", round(psi, 6), _verdict(psi, limits, {}))


def _edges(bins):
    """A numeric check's bin edges; an outer edge written `null` is open (−∞ first, +∞ last),
    and a v1 reference's ±Infinity reads as itself."""
    edges = np.asarray([math.nan if b is None else float(b) for b in bins], dtype=float)
    edges[0] = -math.inf if math.isnan(edges[0]) else edges[0]
    edges[-1] = math.inf if math.isnan(edges[-1]) else edges[-1]
    return edges


def _rate_of(check, name, mask):
    """A share against its reference: z standard errors, both ways, or one with `direction`."""
    need, ref = _min_rows(check), check.get("reference")
    if mask is None or ref is None or len(mask) < need:
        return _unmeasured(name, "rate", f"needs its columns, a reference and at least {need} "
                                         "rows")
    p, ref = float(np.mean(mask)), float(ref)
    # a zero reference still has an uncertainty: the reference part's size bounds it
    floor = 1.0 / max(int(check.get("reference_rows", 1)), 1)
    se = math.sqrt(max(ref, floor) * (1 - min(ref, 1 - 1e-12)) / len(mask))
    signed = (p - ref) / se if se > 0 else 0.0
    # `direction: up` for a share whose fall is good news (unseen categories, alarms): only a
    # rise is a reading; `down` the reverse; the default watches both ways
    z = {"up": signed, "down": -signed}.get(check.get("direction"), abs(signed))
    return _Measured(name, "rate", round(p, 6), _verdict(z, Z_LIMITS, check, "_z"),
                     f"reference {ref:.6g}, z {z:.2f}")


def _flags(values):
    """A 0/1 (or boolean) answer field as a mask; an undecided answer is not raised."""
    return pd.to_numeric(values, errors="coerce").fillna(0).to_numpy(dtype=float) >= 0.5


def _target(check, batch):
    """The truth each row's field answers, by position: the target column itself, or for a
    forecast step h the target h rows later in the same series (NaN where the window ends)."""
    column = check.get("target_column")
    if not column or column not in batch.columns:
        return None
    truth = pd.to_numeric(batch[column], errors="coerce")
    offset = int(check.get("target_offset") or 0)
    if not offset:
        return truth.to_numpy(dtype=float)
    series = check.get("series_by") if check.get("series_by") in batch.columns else None
    keys = [c for c in (series, check.get("order_by")) if c and c in batch.columns]
    ordered = batch.sort_values(keys, kind="mergesort") if keys else batch
    later = truth.loc[ordered.index]
    later = (later.groupby(ordered[series].to_numpy()).shift(-offset) if series
             else later.shift(-offset))
    return later.reindex(batch.index).to_numpy(dtype=float)


def _as_category(series):
    """A category column as text for the PSI bins (whole floats as integers, as the Recipe's
    retype writes them); membership in the encoder's vocabulary is asked of the Recipe."""
    if pd.api.types.is_float_dtype(series) and bool((series.dropna() % 1 == 0).all()):
        return series.astype("Int64").astype(str)
    return series.astype(str)


def _predicate(spec, batch):
    """`{"column", "op", "value"}`, or `{"any": [...]}` / `{"all": [...]}` of them; `None`
    when a column the predicate reads is absent from the window."""
    if "any" in spec or "all" in spec:
        parts = [_predicate(s, batch) for s in (spec.get("any") or spec.get("all"))]
        if any(p is None for p in parts):
            return None
        return (np.logical_or if "any" in spec else np.logical_and).reduce(parts)
    if spec["column"] not in batch.columns:
        return None
    values = pd.to_numeric(batch[spec["column"]], errors="coerce").to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        return np.nan_to_num(OPS[spec["op"]](values, float(spec["value"])), nan=0).astype(bool)


# every kind the monitor measures, by the name `monitor-reference.json` gives it (vega-pack's
# catalog `monitor_kinds` is the list of record; test_monitor_kinds holds the two equal)
KINDS = {"psi": _psi, "rate": _rate, "unseen_rate": _unseen, "class_recall": _recall,
         "output_psi": _output_psi, "prediction_mix": _prediction_mix,
         "label_rate": _label_rate, "alert_rate": _alert_rate, "residual": _residual}
