#!/usr/bin/env python3
"""
serving_check.py — the serving API invariants per task (D6 §7, as BUILD-PLAN R-C3/R-C4 rewrite
them) against a bundle's live service.

    python3 serving_check.py --base-url URL --bundle DIR [--lane ID] [--lanes-file FILE]
                             [--json OUT]

It posts the bundle's shipped examples — `/v1/examples` `test.rows` as JSON to `/v1/predict`
(with every `?include=` the bundle's field kinds gate), and `examples/rows.csv` as an upload to
`/v1/predict/csv`, answered as JSON and as `text/csv` — and holds every answer to:

  SA1   one answer per request row, its keys exactly MANIFEST.output's (through
        `contracts.OUTPUT_FIELDS`), the CSV answer's columns the record's `fields`
  SA2   the answers equal examples/expected.parquet at its `__row__`: compared fields within
        MANIFEST.roundtrip.tolerance, decided fields exactly
  SA3   the CSV answer equals the JSON answer, row for row
  SA4   binary: p(positive) ∈ [0, 1], the decision is p ≥ MANIFEST.output.threshold, and the
        raw logit's σ is p (R-C3)
  SA5   multiclass: the probabilities sum to 1 ± 1e-6, the label is their argmax (softmax of
        the logits)
  SA6   multilabel: one probability per label in [0, 1], no sum constraint, labels =
        {l : p_l ≥ t_l} (threshold_per_label, else threshold), σ of each logit
  SA7   regression (and quantile): the value (and every quantile) finite
  SA8   ordinal: the class probabilities from the cumulative ones — non-negative, summing to 1,
        the cumulative non-increasing — and the label the count of cumulative ≥ 0.5
  SA9   forecast: `values` holds the horizon's steps, all finite
  SA10  reconstruction: reconstruction_error ≥ 0; on include, one reconstruction entry per
        reconstructed field
  SA11  anomaly: score ≥ 0, anomalous = (score ≥ threshold); undecided (null) without one
  SA12  window: a row without its history is answered `answered: false` with a warning
        naming `history` (never refused); graph (node level): a row whose keys reach no
        reference node is answered with a warning, not refused
  SA13  every answer carries the four versions — JSON body keys, CSV answer headers — and
        they are the service's
  SA14  a malformed request is never a 500 nor a confident answer (seeded negatives): a null
        numeric input (the first that is no sample key) is answered with every number finite,
        or not answered with a warning naming it, or refused 422 naming it — never `answered:
        true` with a null number (a decision read from NaN); a window row whose `series_by`
        is null is refused 422 naming it, and one whose `order_by` holds a value the recipe
        may not read is answered or refused 422 naming it, never 500

Exit 0 when every invariant holds (or does not apply to the bundle's task), 1 when one fails,
2 when the check cannot run (no service, no examples). `--json` writes
`vega/serving-check@1`. Field names come from `lanekit.contracts` and MANIFEST.output's
field kinds, never from a per-task list here.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from check_kit import (BundleRecords, CheckError, Client, Report, close, finite,  # noqa: E402
                       write_result)
from lanekit import contracts as C  # noqa: E402

PROB_TOL = 1e-6
MAPPED = frozenset({"probability", "raw", "quantile", "cumulative", "recon"})


# --------------------------------------------------------------------------- reading answers
def json_value(answer: Mapping, kind: str, of: Any):
    """A frame field's value in one RowPrediction, by its kind (bundle-contract §7.3)."""
    key = C.FIELD_KIND_KEYS.get(kind)
    if key is None:
        return None                                           # decision: CSV only
    value = answer.get(key)
    if value is None:
        return None
    if kind in MAPPED:
        return value.get(str(of))
    if kind == "horizon":
        step, _, q = str(of).partition("/")
        cell = value.get(f"h{step}")
        return cell.get(q) if q and isinstance(cell, Mapping) else cell
    if kind == "flag":
        return 1 if str(of) in value else 0
    if kind == "embedding":
        return value[int(of)] if int(of) < len(value) else None
    return value


def _number(text: str):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def same_cell(a, b, rtol: float = 1e-12) -> bool:
    """Two answer cells agree: numbers by value, text by text, missing only as missing."""
    missing = [v is None or v == "" or (isinstance(v, float) and math.isnan(v)) for v in (a, b)]
    if any(missing):
        return all(missing)
    if isinstance(a, bool) or isinstance(b, bool):
        return str(a).lower() in ("1", "true") if str(b).lower() in ("1", "true") else \
            str(a).lower() in ("0", "false")
    x, y = _number(a), _number(b)
    if x is not None and y is not None:
        return close(x, y, rtol, 1e-15)
    return str(a) == str(b)


def read_csv(body: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(body), dtype=str, keep_default_na=False)


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


# --------------------------------------------------------------------------- the run
class ServingCheck:
    """One run of SA1–SA14 against a live service (see the module docstring)."""

    def __init__(self, bundle: BundleRecords, client: Client, report: Report):
        self.bundle, self.client, self.report = bundle, client, report
        self.output = bundle.output
        self.answers: list = []                     # (what, JSON body) of every JSON answer
        self.csv_answers: list = []                 # (what, headers) of every CSV answer

    # -- requests --------------------------------------------------------------------------
    def _predict(self, doc: Mapping, include=(), what: str = "") -> dict:
        query = "".join(f"&include={i}" for i in include).lstrip("&")
        path = C.PREDICT_ROUTE + (f"?{query}" if query else "")
        body = self.client.json_of(self.client.post(path, doc), f"POST {path} ({what})")
        self.answers.append((what or path, body))
        return body

    def _upload(self, accept: str | None, include=()) -> Any:
        csv = (self.bundle.root / C.EXAMPLE_ROWS).read_bytes()
        query = "".join(f"&include={i}" for i in include).lstrip("&")
        answer = self.client.upload(C.PREDICT_CSV_ROUTE + (f"?{query}" if query else ""), csv,
                                    accept)
        if answer.status != 200:
            raise CheckError(f"POST {C.PREDICT_CSV_ROUTE} answered {answer.status}: "
                             f"{answer.body[:300]!r}")
        if accept == C.CSV_ACCEPT:
            self.csv_answers.append(("csv upload (text/csv)", answer.headers))
            return read_csv(answer.body)
        self.answers.append(("csv upload (json)", answer.json))
        return answer.json

    def includes(self) -> tuple:
        kinds = {self.bundle.field_kind(f)[0] for f in self.output.get("fields") or ()}
        return tuple(C.FIELD_KIND_INCLUDE[k] for k in C.FIELD_KIND_INCLUDE if k in kinds)

    def run(self) -> None:
        examples = self.client.json_of(self.client.get(C.EXAMPLES_ROUTE), C.EXAMPLES_ROUTE)
        test = examples.get("test") or {}
        self.rows, history = list(test.get("rows") or ()), test.get("history")
        if not self.rows:
            raise CheckError("/v1/examples carries no test rows")
        doc = {"rows": self.rows, **({"history": history} if history else {})}
        plain = self._predict(doc, what="examples")
        full = self._predict(doc, self.includes(), "examples, every include")
        table, upload = self._upload(C.CSV_ACCEPT), self._upload(None)
        self.sa1(plain, full, table, upload)
        self.sa2(full, self._upload(C.CSV_ACCEPT, self.includes()))
        self.sa3(plain, upload, table)
        self.task_checks(full)
        self.sa12(plain)
        self.sa14()
        self.sa13()

    # -- SA1 -------------------------------------------------------------------------------
    def sa1(self, plain, full, table, upload) -> None:
        r, n = self.report, len(self.rows)
        for what, body, include in (("json", plain, ()), ("json+include", full, self.includes()),
                                    ("csv upload as json", upload, ())):
            preds = body.get("predictions") or []
            want = {"answered", *C.OUTPUT_FIELDS(self.bundle.manifest, include)}
            keys = [set(p) for p in preds]
            r.record("SA1", len(preds) == n, f"{len(preds)} answer(s) for {n} row(s)", what)
            r.record("SA1", all(k == want for k in keys),
                     f"answer keys {sorted(keys[0]) if keys else []} = MANIFEST.output's "
                     f"{sorted(want)}", what)
        r.record("SA1", list(table.columns) == self.csv_columns(table),
                 f"CSV columns {list(table.columns)} = {self.csv_columns(table)}", "csv")
        r.record("SA1", len(table) == n, f"{len(table)} CSV row(s) for {n}", "csv")

    def csv_columns(self, table) -> list:
        """MANIFEST.output.fields as the CSV answer lays them out: the shown fields, the
        upload's truth as true_<column> when it carried one, then answered and warnings."""
        fields = list(self.output.get("fields") or ())
        tail = [f for f in fields if self.bundle.field_kind(f)[0] in C.ANSWER_TAIL]
        body = [f for f in fields if f not in tail and
                self.bundle.field_kind(f)[0] not in C.FIELD_KIND_INCLUDE]
        truth = (self.output.get("truth") or {}).get("column")
        echo = [C.TRUTH_PREFIX + truth] if truth and C.TRUTH_PREFIX + truth in table.columns \
            else []
        return body + echo + tail

    # -- SA2 -------------------------------------------------------------------------------
    def sa2(self, full, table) -> None:
        expected = self.bundle.expected()
        tol = self.bundle.tolerance
        at = expected[C.EXAMPLE_ROW_KEY].to_numpy(dtype="int64") if C.EXAMPLE_ROW_KEY in \
            expected.columns else np.arange(len(expected))
        preds = full.get("predictions") or []
        compared = [f for f, _ in self.output.get("compared") or ()]
        decided = [f for f, _ in self.output.get("decided") or ()]
        for field in compared:
            got_csv = [_number(table[field].iloc[i]) if field in table.columns else None
                       for i in at]
            kind, of = self.bundle.field_kind(field)
            got_json = [json_value(preds[i], kind, of) for i in at]
            want = expected[field].to_numpy(dtype="float64", na_value=np.nan)
            for name, got in (("csv", got_csv), ("json", got_json)):
                if name == "json" and kind == "decision":
                    continue
                gaps = [abs(float(g) - w) if g is not None and np.isfinite(w) else
                        (0.0 if g is None and not np.isfinite(w) else math.inf)
                        for g, w in zip(got, want)]
                worst = max(gaps) if gaps else 0.0
                self.report.record("SA2", worst <= tol, f"{field}: worst |Δ| {worst:.3g} over "
                                   f"{len(gaps)} row(s) (tolerance {tol:.3g})", name)
        for field in decided:
            col = table[field].tolist() if field in table.columns else [None] * len(table)
            bad = [int(i) for i, w in zip(at, expected[field].tolist()) if not same_cell(col[i], w)]
            self.report.record("SA2", not bad, f"{field}: decided exactly on {len(at)} row(s) "
                               f"(differ at {bad[:5]})", "csv")
        answered = [bool(preds[i].get("answered", True)) for i in at]
        self.report.record("SA2", all(answered), f"{sum(answered)} of {len(at)} compared "
                           "row(s) answered", "json")

    # -- SA3 -------------------------------------------------------------------------------
    def sa3(self, plain, upload, table) -> None:
        truth = list(upload.get("truth") or ())
        for what, body in (("csv vs /v1/predict json", plain), ("csv vs csv-upload json", upload)):
            preds, bad = body.get("predictions") or [], []
            for i, pred in enumerate(preds[:len(table)]):
                for column in table.columns:
                    want = self._json_cell(pred, column, truth, i)
                    if want is not _SKIP and not same_cell(table[column].iloc[i], want):
                        bad.append((i, column, table[column].iloc[i], want))
            self.report.record("SA3", not bad and len(preds) == len(table),
                               f"{len(table)} CSV row(s) × {len(table.columns)} column(s) equal "
                               f"the JSON answer (first difference {bad[:1]})", what)

    def _json_cell(self, pred, column, truth, i):
        if column.startswith(C.TRUTH_PREFIX):
            return truth[i] if i < len(truth) else _SKIP
        kind, of = self.bundle.field_kind(column)
        if kind == "warnings":
            return C.WARNING_SEP.join(pred.get("warnings") or ())
        if kind == "decision":
            return _SKIP                     # CSV only: SA2 and the task checks hold it
        if kind == "answered":
            return pred.get("answered", True)
        return json_value(pred, kind, of)

    # -- SA4–SA11 ----------------------------------------------------------------------------
    def task_checks(self, full) -> None:
        reading = self.bundle.reading
        mine = C.READING_CHECK.get(reading)
        for check_id in sorted(set(C.READING_CHECK.values())):
            if check_id != mine:
                self.report.not_applicable(check_id, f"the bundle's reading is {reading}")
        if mine is None:
            raise CheckError(f"MANIFEST.output.reading {reading!r} has no per-task invariant")
        rows = [p for p in full.get("predictions") or () if p.get("answered", True)]
        if not rows:
            self.report.record(mine, False, "no answered row to hold to the invariant")
        method = getattr(self, "_" + mine.lower())
        for i, pred in enumerate(rows):
            for ok, detail in method(pred):
                self.report.record(mine, ok, detail, f"row {i}")

    def _probabilities(self, pred) -> dict:
        return {str(k): v for k, v in (pred.get("probabilities") or {}).items()}

    def _sa4(self, pred) -> list:
        positive = str(self.output.get("positive"))
        threshold = float(self.output.get("threshold"))
        p = self._probabilities(pred).get(positive)
        out = [(finite(p) and 0.0 <= p <= 1.0, f"p({positive}) = {p} ∈ [0, 1]")]
        if finite(p):
            out.append(((pred.get("label") == positive) == (p >= threshold),
                        f"label {pred.get('label')} = (p ≥ {threshold:.6g})"))
            logit = (pred.get("logits") or {}).get(positive)
            if logit is not None:
                out.append((close(_sigmoid(float(logit)), p, PROB_TOL, PROB_TOL),
                            f"σ(logit {logit:.6g}) = p (the score is the raw logit)"))
        return out

    def _sa5(self, pred) -> list:
        probs = self._probabilities(pred)
        values = [float(v) for v in probs.values()]
        best = max(probs, key=lambda k: probs[k]) if probs else None
        out = [(all(0.0 <= v <= 1.0 for v in values) and abs(sum(values) - 1.0) <= PROB_TOL,
                f"Σ p = {sum(values):.9g} over {len(values)} class(es)"),
               (pred.get("label") == best, f"label {pred.get('label')} = argmax {best}")]
        logits = pred.get("logits")
        if logits:
            z = np.asarray([float(logits[k]) for k in probs])
            soft = np.exp(z - z.max()) / np.exp(z - z.max()).sum()
            out.append((bool(np.allclose(soft, values, atol=PROB_TOL)), "softmax(logits) = p"))
        return out

    def _sa6(self, pred) -> list:
        probs = self._probabilities(pred)
        per = self.output.get("threshold_per_label") or {}
        base = self.output.get("threshold")
        flagged = {l for l, p in probs.items() if p >= float(per.get(l, base))}
        out = [(all(finite(p) and 0.0 <= p <= 1.0 for p in probs.values()),
                f"{len(probs)} label probabilit(ies) in [0, 1], no sum constraint"),
               (set(pred.get("labels") or ()) == flagged,
                f"labels {sorted(pred.get('labels') or ())} = {{l : p_l ≥ t_l}} {sorted(flagged)}")]
        for label, z in (pred.get("logits") or {}).items():
            out.append((close(_sigmoid(float(z)), probs.get(str(label)), PROB_TOL, PROB_TOL),
                        f"σ(logit {label}) = p({label})"))
        return out

    def _sa7(self, pred) -> list:
        out = [(finite(pred.get("value")), f"value {pred.get('value')} finite")]
        quantiles = pred.get("quantiles")
        if quantiles is not None:
            out.append((all(finite(v) for v in quantiles.values()),
                        f"{len(quantiles)} quantile(s) finite"))
        return out

    def _sa8(self, pred) -> list:
        classes = [str(c) for c in self.output.get("classes") or ()]
        probs = self._probabilities(pred)
        cum_fields = self.bundle.fields_of("cumulative")
        cum = [json_value(pred, "cumulative", self.bundle.field_kind(f)[1]) for f in cum_fields]
        p = [probs.get(c) for c in classes]
        if any(v is None for v in p + cum):
            return [(False, f"ordinal answer without every probability and cumulative: {pred}")]
        c = np.asarray(cum, dtype=np.float64)
        want = np.concatenate([[1.0 - c[0]], c[:-1] - c[1:], [c[-1]]])
        label = classes[int((c >= 0.5).sum())]
        return [(min(p) >= 0.0 and abs(sum(p) - 1.0) <= PROB_TOL,
                 f"{len(p)} class probabilities ≥ 0, Σ = {sum(p):.9g}"),
                (bool((np.diff(c) <= PROB_TOL).all()), "the cumulative probabilities do not rise"),
                (bool(np.allclose(p, want, atol=PROB_TOL)), "p_j = c_{j−1} − c_j"),
                (pred.get("label") == label, f"label {pred.get('label')} = the count of c ≥ 0.5 "
                                             f"({label})")]

    def _sa9(self, pred) -> list:
        horizons = [int(h) for h in self.output.get("horizons") or ()]
        values = pred.get("values") or {}
        want = {f"h{h}" for h in horizons}
        cells = [v for cell in values.values() for v in
                 (cell.values() if isinstance(cell, Mapping) else [cell])]
        return [(set(values) == want and all(finite(v) for v in cells),
                 f"values {sorted(values)} = the horizon {sorted(want)}, all finite")]

    def _sa10(self, pred) -> list:
        err = pred.get("reconstruction_error")
        out = [(finite(err) and err >= 0.0, f"reconstruction_error {err} ≥ 0")]
        recon = pred.get("reconstruction")
        if recon is not None:
            n = len(self.bundle.fields_of("recon"))
            out.append((len(recon) == n, f"{len(recon)} reconstruction entr(ies) = {n} fields"))
        return out

    def _sa11(self, pred) -> list:
        score, flag = pred.get("score"), pred.get("anomalous")
        threshold = self.output.get("threshold")
        out = [(finite(score) and score >= 0.0, f"score {score} ≥ 0")]
        if threshold is None:
            out.append((flag is None, f"no threshold: anomalous is undecided ({flag})"))
        elif finite(score):
            out.append((flag == (score >= float(threshold)),
                        f"anomalous {flag} = (score ≥ {float(threshold):.6g})"))
        return out

    # -- SA12 ------------------------------------------------------------------------------
    def sa12(self, plain) -> None:
        shape, r = self.bundle.shape, self.report
        if shape == "row":
            r.not_applicable("SA12", "a row sample reads no history and no graph")
            return
        preds = plain.get("predictions") or []
        if shape == "window":
            k = next((i for i, p in enumerate(preds) if p.get("answered", True)), None)
            if k is None:
                r.record("SA12", False, "no example row is answered with its history")
                return
            alone = self._predict({"rows": [self.rows[k]]}, what="a window row alone")
            pred = (alone.get("predictions") or [{}])[0]
            notes = " ".join(pred.get("warnings") or ())
            values = [v for key, v in pred.items() if key not in C.ANSWER_TAIL]
            r.record("SA12", pred.get("answered") is False and C.HISTORY_NEEDLE in notes and
                     all(v is None for v in values),
                     f"answered {pred.get('answered')}, warnings {pred.get('warnings')}, the "
                     "answer's values null", "window row without history")
            return
        self._sa12_graph()

    def _sa12_graph(self) -> None:
        r = self.report
        if self.bundle.level != "node":
            r.not_applicable("SA12", "at level graph a query is a whole graph: it reaches no "
                                     "reference node")
            return
        columns = self._key_columns()
        if not columns:
            r.not_applicable("SA12", f"the {self.bundle.sample.get('method')} graph links rows "
                                     "by their features, not by keys a row could miss")
            return
        row = dict(self.rows[0])
        for k, col in enumerate(columns):
            row[col] = f"vega-check-unreached-{k}"
        answer = self.client.post(C.PREDICT_ROUTE, {"rows": [row]})
        body = answer.json or {}
        if answer.status == 200:
            self.answers.append(("an unreached graph row", body))
        pred = (body.get("predictions") or [{}])[0]
        r.record("SA12", answer.status == 200 and pred.get("answered", True) is True and
                 bool(pred.get("warnings")),
                 f"status {answer.status}, answered {pred.get('answered')}, warnings "
                 f"{pred.get('warnings')}", "graph row reaching no reference node")

    def _key_columns(self) -> list:
        meta = self.bundle.root / C.GRAPH_REFERENCE_FILES["meta"]
        doc = json.loads(meta.read_text(encoding="utf-8")) if meta.is_file() else {}
        if doc.get("method") != "key_pair":
            return []
        return [str(c) for c in (doc.get("params") or {}).get("columns") or ()]

    # -- SA14 ------------------------------------------------------------------------------
    def _one_body(self, **changes) -> dict:
        """examples/request.json with its first row changed (a request whose first row is
        always answered as shipped)."""
        doc = json.loads((self.bundle.root / C.EXAMPLE_REQUEST).read_text(encoding="utf-8"))
        doc["rows"] = [dict(r) for r in doc["rows"]]
        doc["rows"][0].update(changes)
        return doc

    def _plain_numeric(self):
        keys = {self.bundle.sample.get(k) for k in ("series_by", "order_by", "graph_id")}
        return next((c["name"] for c in self.bundle.schema.get("input") or ()
                     if c.get("dtype") != "string" and c["name"] not in keys), None)

    def sa14(self) -> None:
        column = self._plain_numeric()
        if column is None:
            self.report.not_applicable("SA14", "no numeric input that is no sample key")
        else:
            answer = self.client.post(C.PREDICT_ROUTE, self._one_body(**{column: None}))
            self.report.record("SA14", *sa14_empty_cell(answer.status, answer.json, column,
                                                         self.output),
                               f"a null {column}")
        if self.bundle.shape != "window":
            return
        series, order = self.bundle.sample.get("series_by"), self.bundle.sample.get("order_by")
        for key, value, refused in ((series, None, True), (order, None, True),
                                    (order, "vega-check-unreadable", False)):
            if not key:
                continue
            answer = self.client.post(C.PREDICT_ROUTE, self._one_body(**{key: value}))
            self.report.record("SA14", *sa14_key(answer.status, answer.json, key, refused),
                               f"window key {key}={value!r}")

    # -- SA13 ------------------------------------------------------------------------------
    def sa13(self) -> None:
        health = self.client.json_of(self.client.get(C.HEALTH_ROUTE), C.HEALTH_ROUTE)
        info = self.client.json_of(self.client.get(C.INFO_ROUTE), C.INFO_ROUTE)
        want = {k: health.get(k) if k != "api_version" else info.get(k) for k in C.VERSION_KEYS}
        for what, body in self.answers:
            got = {k: body.get(k) for k in C.VERSION_KEYS}
            self.report.record("SA13", got == want and all(got.values()),
                               f"versions {got} = the service's", what)
        for what, headers in self.csv_answers:
            got = {k: headers.get(h) for k, h in C.VERSION_HEADERS.items()}
            self.report.record("SA13", got == {k: str(v) for k, v in want.items()},
                               f"headers {got} = the service's versions", what)


def _names(body) -> list:
    return [e.get("column") for e in ((body or {}).get("errors") or ())]


def sa14_empty_cell(status: int, body, column: str, output: Mapping) -> tuple:
    """(holds, what): a null numeric input answered with every number finite, not answered
    with a warning naming it, or refused 422 naming it (SA14)."""
    if status == 422:
        return column in _names(body), f"refused 422 naming {_names(body)}"
    if status != 200:
        return False, f"answered {status}"
    pred = ((body or {}).get("predictions") or [{}])[0]
    if pred.get("answered") is False:
        notes = " ".join(pred.get("warnings") or ())
        return column in notes or "context" in notes, f"not answered, warnings {pred.get('warnings')}"
    shown = []
    for field, k in (output.get("field_kinds") or {}).items():
        kind = (k or {}).get("kind")
        if kind in ("probability", "value", "quantile", "score", "row_score", "cumulative", "raw") \
                and C.FIELD_KIND_KEYS.get(kind) in pred:      # a gated kind only when shown
            shown.append(json_value(pred, kind, k.get("of")))
    bad = [v for v in shown if not finite(v)]
    return bool(shown) and not bad, f"answered with {len(shown)} number(s), {len(bad)} null or non-finite"


def sa14_key(status: int, body, key: str, refused: bool) -> tuple:
    """(holds, what): a window key the recipe cannot read refused 422 naming it (`refused`:
    it must be), else answered 200 — never 500 (SA14)."""
    if status == 422:
        return key in _names(body), f"refused 422 naming {_names(body)}"
    return (status == 200 and not refused), f"answered {status}"


_SKIP = object()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--bundle", required=True, type=Path)
    ap.add_argument("--lane", default="")
    ap.add_argument("--lanes-file", default=None, help="accepted for the stage CLI; unused")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    report = Report(C.SERVING_CHECK, args.lane, args.base_url, str(args.bundle),
                    C.SERVING_CHECK_IDS)
    try:
        bundle = BundleRecords.load(args.bundle)
        check = ServingCheck(bundle, Client(args.base_url.rstrip("/")), report)
        check.run()
    except CheckError as exc:
        print(f"ERROR: serving_check cannot run: {exc}")
        write_result(args.json, {**report.as_dict(), "verdict": "error", "error": str(exc)})
        return 2
    report.print_lines()
    write_result(args.json, report.as_dict(reading=bundle.reading, rows=len(check.rows),
                                           shape=bundle.shape))
    print(("OK: " if report.verdict == "pass" else "ERROR: ") +
          f"serving_check {report.verdict} on {args.lane or args.bundle.name}")
    return 0 if report.verdict == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
