"""
roundtrip_compare — the bundle's answers against what the evaluation stored, pair by pair
(bundle-contract §5; build plan S10). Support of `roundtrip_plan` (the pack-time round trip) and
`examples_pack` (the shipped examples, checked before they ship).

    round_trip(predictor, evaluated, tolerance) -> RoundTrip
    expected_frame(stored, output, reader) -> the evaluation's answer per stored prediction,
                                              by bundle field
    compare(got, at, expected, output, tolerance) -> Comparison
    tolerance_for(stored, output) -> Tolerance
    explain_failure(trip, recipe, markers)

The pairs come from the output record (`MANIFEST.output`, vega/output@2), never from column
names: every `compared` pair is numeric, `|bundle − stored| ≤ tolerance` (the contract's formula
over the stored compared columns — absolute near zero, relative above one); every `decided`
field is exact against the evaluation's own decision, read through the bundle's reader
(`reader.from_stored`: the stored `prediction`, or the argmax / cumulative count / per-label
threshold of the stored head); a stored prediction whose query row went unanswered is itself a
failure. A missing value on one side is an infinite difference; on both, none.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from pack_context import out_warn
from roundtrip_inputs import id_text
from vega_pack_core import RoundTrip, Tolerance

__all__ = ["Comparison", "answers", "compare", "compared_sources", "expected_frame",
           "explain_failure", "pairing", "round_trip", "tolerance_for"]

FITTED_OPS = ("impute", "clip", "winsorize", "transform", "bin", "encode", "scale", "project")


@dataclass(frozen=True)
class Comparison:
    """One comparison: how many rows, the worst compared difference, how many decisions were
    compared, and the first disagreement (`row` is the expected frame's index label; `kind`
    compared | decided | answered), None when everything agrees."""

    rows: int
    columns: int
    decisions: int
    worst: float
    first_bad: Mapping | None

    @property
    def passed(self) -> bool:
        return self.first_bad is None


# --------------------------------------------------------------------------- the pairs

def compared_sources(output) -> list:
    """The stored columns the record's `compared` pairs name, in order."""
    return [str(pair[1]) for pair in (output or {}).get("compared") or ()]


def _fields(output, which) -> list:
    return [str(pair[0]) for pair in (output or {}).get(which) or ()]


def expected_frame(stored, output, reader):
    """The evaluation's answer per stored prediction, by bundle field: each compared field the
    stored column its pair names, each decided field the bundle reader's reading of the stored
    rows (`from_stored`: the evaluation's decision, not this build's). Indexed like `stored`."""
    import pandas as pd
    columns = {field: stored[source].to_numpy()
               for field, source in zip(_fields(output, "compared"), compared_sources(output))}
    decided = _fields(output, "decided")
    if decided:
        answer = reader.from_stored(stored)
        columns.update({field: answer[field].to_numpy() for field in decided})
    return pd.DataFrame(columns, index=stored.index)


def tolerance_for(stored, output) -> Tolerance:
    """The contract's tolerance over every stored compared value."""
    import numpy as np
    sources = compared_sources(output)
    values = (stored[sources].to_numpy(dtype=np.float64, na_value=np.nan).ravel()
              if sources else np.zeros(0))
    finite = values[np.isfinite(values)]
    return Tolerance.for_scores(finite.tolist())


# --------------------------------------------------------------------------- the comparison

def compare(got, at, expected, output, tolerance) -> Comparison:
    """`got` (the bundle's answer frame) at rows `at` (−1: no answer) against `expected` row
    for row, over the record's compared and decided fields."""
    import numpy as np
    at = np.asarray(at, dtype=np.int64)
    compared, decided = _fields(output, "compared"), _fields(output, "decided")
    first = _unanswered(got, at, expected)
    worst, where = _worst(got, at, expected, compared)
    if first is None and worst > tolerance.value:
        first = where
    if first is None:
        first = _decisions(got, at, expected, decided)
    return Comparison(rows=len(at), columns=len(compared),
                      decisions=len(at) if decided else 0, worst=worst, first_bad=first)


def _unanswered(got, at, expected):
    import numpy as np
    done = (np.asarray(at) >= 0) & (np.asarray(at) < len(got))
    if "answered" in got.columns and len(got):
        flags = got["answered"].to_numpy(dtype=bool)
        done &= np.where(done, flags[np.clip(at, 0, len(got) - 1)], False)
    if done.all():
        return None
    k = int(np.flatnonzero(~done)[0])
    return {"row": _plain(expected.index[k]), "column": "answered", "kind": "answered",
            "bundle": False, "evaluation": True, "abs_diff": None}


def _worst(got, at, expected, fields) -> tuple:
    """(worst |Δ| over the compared fields, the disagreement it is)."""
    import numpy as np
    worst, where = 0.0, None
    safe = np.clip(at, 0, max(len(got) - 1, 0))
    for field in fields:
        mine = got[field].to_numpy(dtype=np.float64, na_value=np.nan)[safe] if len(got) \
            else np.full(len(at), np.nan)
        theirs = expected[field].to_numpy(dtype=np.float64, na_value=np.nan)
        with np.errstate(invalid="ignore"):
            diff = np.abs(mine - theirs)
        diff[(mine == theirs) | (np.isnan(mine) & np.isnan(theirs))] = 0.0
        diff[np.isnan(diff)] = np.inf
        if diff.size and float(diff.max()) > worst:
            k = int(np.argmax(diff))
            worst = float(diff[k])
            where = {"row": _plain(expected.index[k]), "column": field, "kind": "compared",
                     "bundle": float(mine[k]), "evaluation": float(theirs[k]),
                     "abs_diff": worst}
    return worst, where


def _decisions(got, at, expected, fields):
    import numpy as np
    safe = np.clip(at, 0, max(len(got) - 1, 0))
    for field in fields:
        mine = got[field].to_numpy(dtype=object)[safe]
        theirs = expected[field].to_numpy(dtype=object)
        for k, (a, b) in enumerate(zip(mine, theirs)):
            if not _same(a, b):
                return {"row": _plain(expected.index[k]), "column": field, "kind": "decided",
                        "bundle": _plain(a), "evaluation": _plain(b), "abs_diff": None}
    return None


def _same(a, b) -> bool:
    """Numbers by value (1 == 1.0), names by text, missing (None, NaN, NA) only as missing."""
    missing = [_missing(v) for v in (a, b)]
    if any(missing):
        return all(missing)
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return str(a) == str(b)


def _missing(value) -> bool:
    """None, NaN of any float width, NA, NaT; a list or a text is never missing."""
    import pandas as pd
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _plain(value):
    if hasattr(value, "item"):
        value = value.item()
    if _missing(value):
        return None
    return value


# --------------------------------------------------------------------------- the round trip

def answers(predictor, plan):
    """The bundle's answer to every query row, in query order (node level: in chunks)."""
    import pandas as pd
    query, chunk = plan.query, max(1, int(plan.chunk))
    if chunk >= len(query):
        return predictor.predict(query, plan.history).reset_index(drop=True)
    parts = [predictor.predict(query.iloc[i:i + chunk].reset_index(drop=True), plan.history)
             for i in range(0, len(query), chunk)]
    return pd.concat(parts, ignore_index=True)


def pairing(plan, got, output):
    """Stored prediction i -> the query row answering it (−1: none): by raw position, or at
    graph level by the answer's graph id (its `group` field, else the raw id column)."""
    import numpy as np
    if plan.graph_ids is None:
        where = {int(p): j for j, p in enumerate(plan.positions)}
        return np.asarray([where.get(int(r), -1) for r in plan.stored_row_ids], dtype=np.int64)
    kinds = (output or {}).get("field_kinds") or {}
    group = next((f for f, k in kinds.items() if (k or {}).get("kind") == "group"), None)
    ids = got[group] if group in got.columns else plan.query[plan.group_column]
    first = {}
    for j, g in enumerate(ids.to_numpy()):
        first.setdefault(id_text(g), j)
    return np.asarray([first.get(id_text(g), -1) for g in plan.graph_ids], dtype=np.int64)


def round_trip(predictor, evaluated, tolerance) -> RoundTrip:
    """Every stored prediction answered again from its raw rows and compared (R-A6's boundary
    excluded and recorded)."""
    plan, stored = evaluated.plan, evaluated.stored
    got = answers(predictor, plan)
    at = pairing(plan, got, evaluated.output)
    kept = ~plan.excluded
    result = compare(got, at[kept], evaluated.expected[kept], evaluated.output, tolerance)
    first = dict(result.first_bad) if result.first_bad else None
    if first is not None:
        first["row_id"] = _row_id(stored.frame, first["row"])
    return RoundTrip(rows=len(stored), source=stored.source, tolerance=tolerance,
                     max_abs_diff=result.worst, status="pass" if first is None else "fail",
                     shaper_transform_check="pass", first_bad=first, columns=result.columns,
                     decisions=result.decisions, boundary=plan.boundary)


def _row_id(frame, label):
    if "row_id" not in frame.columns or label not in frame.index:
        return None
    return _plain(frame.at[label, "row_id"])


def explain_failure(trip, recipe, markers) -> None:
    """Name what disagreed, and the steps most likely responsible.

    A round-trip failure is a defect in the transformer, in a copied file, or in the record the
    bundle decides by. The fitted steps are where a numeric divergence almost always is, because
    they are the only ones carrying parameters that can be misread, so they are listed in the
    order they ran; a decided field that disagrees while every compared field agrees is the
    decision rule or its threshold.
    """
    bad = trip.first_bad or {}
    kind = bad.get("kind") or "compared"
    what = {"decided": f"the decided field {bad.get('column')}",
            "answered": "an answer (the row went unanswered)"}.get(kind,
                                                                   f"{bad.get('column')}")
    out_warn(f"round trip FAILED on {what} at stored row {bad.get('row')} (row_id "
             f"{bad.get('row_id')}): the bundle answers {bad.get('bundle')}, the evaluation "
             f"stored {bad.get('evaluation')}"
             + (f", |Δ| {bad.get('abs_diff')} against a tolerance of {trip.tolerance.value}"
                if bad.get("abs_diff") is not None else ""), markers)
    if kind == "decided":
        out_warn("every compared field agreed and the decision did not: the bundle decides by "
                 "MANIFEST.output's rule and threshold, the evaluation stored its own — the "
                 "threshold.json the bundle was packed from is not the one the stored decision "
                 "was made with", markers)
        return
    fitted = [f"{s['id']} {s['op']}" for s in ((recipe or {}).get("steps") or [])
              if s.get("op") in FITTED_OPS]
    out_warn("the fitted steps are where a divergence of this kind comes from, in the order "
             f"they ran: {', '.join(fitted) or '(none)'}. Fix the shaper's emitter or the "
             "copied file — never the tolerance, which is the contract's formula and not a "
             "knob", markers)
