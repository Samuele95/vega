"""
demo_presets — the presets of a demonstration: one typical case per stratum and the edge cases
the schema and the Recipe imply, as `vega/presets@2` entries that carry a `kind` and `facts` and
no sentence (bundle-contract §7.4; build plan R-D8). The page renders `preset.<kind>.name` and
`preset.<kind>.text` from its own strings table.

    AnswerReader(output).check(frame)                    the answer read through its field kinds
    typical(stratum, context) -> preset | None
    edges(context, neutral, base_key) -> Edges(presets, notes, remarks)

**Typical cases** (from the stratum's items, answered by the bundle's own Predictor exactly as
the item is sent):

  class stratum    the median-confidence item among those the model decides correctly
                   (`typical_correct`, facts `{right, of}`), else the median item
                   (`typical_none_correct`, `right` 0) — confidence is the probability of the
                   stratum's class, or, for a labelled anomaly, the score towards its side;
  label / none     the same, "correct" meaning the decided label set is the true one;
  bin              the median-|residual| item (`typical_bin`, facts `{bin}`);
  all, flagged, typical  the median-score item (`typical_score`, facts `{of}`);
  a window's       the same item, as `typical_sequence` (facts `{of}` and the stratum's own).

`expected` is what the stratum says the answer is: its class, the item's label set, the item's
value with the bin (a forecast's: one number per step ahead), the alarm a score stratum stands
for, or null where the reading decides nothing. **Edge cases** start from the neutral stratum's typical item (a row, a query node, a
window's last row; none at graph level): the first ranged input × 10 (`edge_range`), the
columns training saw non-finite values in set to Infinity (`edge_infinity`), and per text input
a code the Recipe was never fitted on (`edge_unseen`) — found by asking the Recipe
(`transform(…, with_unseen=True)`) about candidates `max+1 … max+4096` for digit codes and
`<column>-unseen-<k>` (k < 64) for other text, never guessed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from pack_context import PackExit

__all__ = ["AnswerReader", "Edges", "PresetContext", "UnreadAnswer", "edges", "typical",
           "unseen_value"]

RANGE_FACTOR = 10
DIGIT_CANDIDATES = 4096
TEXT_CANDIDATES = 64
DIGITS = "^[0-9]+$"
ID_TEXT = re.compile(r"[^A-Za-z0-9._-]+")
CLASS_KINDS = ("class", "label", "none")


class UnreadAnswer(PackExit):
    """The Predictor's answer does not carry the output record's fields (exit 3)."""


class AnswerReader:
    """The Predictor's answer frame read through MANIFEST.output's field kinds (bundle-contract
    §2.1, §7.3): the decided class, a class's probability, the decided label set, the reading's
    score field, whether a window row was answered. No field name is assumed."""

    def __init__(self, output):
        self._fields = list(output.get("fields") or ())
        kinds = {f: dict(k) for f, k in (output.get("field_kinds") or {}).items()}
        self._class = next((f for f in self._fields if kinds[f]["kind"] == "class"), None)
        self._prob = {str(k["of"]): f for f, k in kinds.items() if k["kind"] == "probability"}
        self._flag = {str(k["of"]): f for f, k in kinds.items() if k["kind"] == "flag"}
        self._answered = next((f for f, k in kinds.items() if k["kind"] == "answered"), None)
        self._score = output.get("score_field")

    def check(self, frame) -> None:
        """UnreadAnswer unless the answer carries every field of MANIFEST.output.fields."""
        absent = [f for f in self._fields if f not in frame.columns]
        if absent:
            raise UnreadAnswer(3, f"the bundle's Predictor answers no {', '.join(absent)} of "
                                  "MANIFEST.output.fields; the demonstration presets read the "
                                  "answer through the output record, as the service does")

    def decided(self, frame) -> np.ndarray:
        if self._class is None:
            return np.full(len(frame), None, dtype=object)
        return np.asarray([None if _gap(v) else str(v) for v in frame[self._class]],
                          dtype=object)

    def probability(self, frame, name):
        field = self._prob.get(str(name))
        return None if field is None else _floats(frame[field])

    def probabilities(self, frame) -> np.ndarray:
        """(rows, labels) of the label probabilities, in the record's label order."""
        cols = [_floats(frame[f]) for f in self._prob.values()]
        return np.column_stack(cols) if cols else np.zeros((len(frame), 0))

    def label_sets(self, frame) -> list:
        flags = {name: frame[f].to_numpy() for name, f in self._flag.items()}
        return [frozenset(n for n, v in flags.items() if not _gap(v[i]) and int(v[i]) == 1)
                for i in range(len(frame))]

    def score(self, frame) -> np.ndarray:
        if not self._score or self._score not in frame.columns:
            return np.full(len(frame), np.nan)
        return _floats(frame[self._score])

    def answered(self, frame) -> np.ndarray:
        if self._answered is None:
            return np.ones(len(frame), dtype=bool)
        return np.asarray([bool(v) if not _gap(v) else False for v in frame[self._answered]])


def _gap(value) -> bool:
    return value is None or value is pd.NA or (isinstance(value, float) and value != value)


def _floats(series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)


@dataclass(frozen=True, eq=False)
class PresetContext:
    """What a preset is made from: the inputs, the candidates, the truths, the answers to the
    drawn items (indexed by key) and the reader of those answers."""

    inputs: Any
    samples: Any
    truths: Any
    answers: pd.DataFrame
    reader: AnswerReader

    @property
    def window(self) -> bool:
        return self.samples.shape == "window"


# --------------------------------------------------------------------------- typical cases

def typical(stratum, ctx: PresetContext):
    """The stratum's typical preset (None for an empty stratum): the key it was cut from and the
    preset."""
    if not stratum.members:
        return None
    frame = ctx.answers.loc[list(stratum.members)]
    if stratum.kind in CLASS_KINDS:
        pick, facts, expected = _classes(stratum, frame, ctx)
        kind = "typical_correct" if facts["right"] else "typical_none_correct"
    elif stratum.kind == "bin":
        pick, facts, expected = _bin(stratum, frame, ctx)
        kind = "typical_bin"
    else:
        pick, facts, expected = _score(stratum, frame, ctx)
        kind = "typical_score"
    if ctx.window:
        kind, facts = "typical_sequence", {"of": len(stratum.members), **facts}
    preset = {"id": f"typical-{stratum.id}", "kind": kind, "group": "typical",
              "stratum": stratum.id, "expected": expected, "facts": facts,
              **ctx.samples.body(pick)}
    return pick, preset


def _median(pool, values) -> int:
    """The pool member at the median of `values` (the upper middle on an even count)."""
    order = np.argsort(values, kind="stable")
    return int(pool[order[len(order) // 2]])


def _classes(stratum, frame, ctx):
    keys = np.asarray(stratum.members, dtype=np.int64)
    answered = ctx.reader.answered(frame)
    correct, confidence = _correct(stratum, frame, ctx)
    correct = correct & answered
    ok = np.isfinite(confidence) & answered
    pool = np.flatnonzero(correct & ok) if (correct & ok).any() else np.flatnonzero(ok)
    pick = _median(keys[pool], confidence[pool]) if len(pool) else int(keys[0])
    if stratum.kind == "class":
        expected = {"kind": "class", "value": stratum.name}
    else:
        expected = {"kind": "labels", "value": list(ctx.truths.values[pick])}
    return pick, {"right": int(correct.sum()), "of": len(keys)}, expected


def _correct(stratum, frame, ctx):
    """(decided correctly, confidence) per item of a class, label or none stratum."""
    reader = ctx.reader
    if stratum.kind == "class":
        correct = reader.decided(frame) == stratum.name
        prob = reader.probability(frame, stratum.name)
        return correct, prob if prob is not None else _side(stratum, frame, ctx)
    sets = reader.label_sets(frame)
    truth = [frozenset(ctx.truths.values[int(k)]) for k in stratum.members]
    correct = np.asarray([s == t for s, t in zip(sets, truth)], dtype=bool)
    if stratum.kind == "label":
        return correct, reader.probability(frame, stratum.name)
    probs = reader.probabilities(frame)
    return correct, 1.0 - (probs.max(axis=1) if probs.shape[1] else np.zeros(len(frame)))


def _side(stratum, frame, ctx) -> np.ndarray:
    """A labelled anomaly's confidence in a class: the score towards that class's side (up
    for the positive class, down for the other)."""
    output = ctx.inputs.output
    classes = [str(c) for c in output.get("classes") or ()]
    positive = str(output["positive"]) if output.get("positive") is not None else classes[-1]
    score = ctx.reader.score(frame)
    return score if stratum.name == positive else -score


def _bin(stratum, frame, ctx):
    keys = np.asarray(stratum.members, dtype=np.int64)
    truth = np.asarray([ctx.truths.first(k) for k in keys], dtype=np.float64)
    residual = np.abs(ctx.reader.score(frame) - truth)
    ok = np.isfinite(residual) & ctx.reader.answered(frame)
    pick = _median(keys[ok], residual[ok]) if ok.any() else int(keys[0])
    lo, hi = stratum.range
    expected = {"kind": "value", "value": _value_truth(ctx.truths.values[int(pick)]), "range": [lo, hi]}
    return pick, {"bin": [lo, hi]}, expected


def _value_truth(value):
    """A value preset's truth as the page compares it: a number, or a forecast's whole horizon
    (one number per step ahead, its first step the one the bin was cut on) — never the first
    step alone, which the page would read as the error over every step."""
    if isinstance(value, (list, tuple, np.ndarray)):
        return [float(v) for v in value]
    return float(value)


def _score(stratum, frame, ctx):
    keys = np.asarray(stratum.members, dtype=np.int64)
    score = ctx.reader.score(frame)
    ok = np.isfinite(score) & ctx.reader.answered(frame)
    pick = _median(keys[ok], score[ok]) if ok.any() else int(keys[0])
    output = ctx.inputs.output
    alarm = output.get("reading") != "reconstruction" and output.get("threshold") is not None
    expected = {"kind": "alarm", "value": stratum.kind == "flagged"} if alarm else None
    return pick, {"of": len(keys)}, expected


# --------------------------------------------------------------------------- edge cases

@dataclass(frozen=True)
class Edges:
    """The edge presets, what went wrong finding them (`notes`: WARN lines) and what is merely
    so (`remarks`: a text column the Recipe does not encode has no unseen code)."""

    presets: tuple
    notes: tuple = ()
    remarks: tuple = ()


def edges(ctx: PresetContext, neutral, base_key) -> Edges:
    """The edge presets on the neutral stratum's typical item."""
    anchor = ctx.samples.anchor(base_key)
    if anchor is None:
        return Edges(())
    inputs, notes, remarks = ctx.inputs, [], []
    keys = set(ctx.samples.columns.keys)
    columns = [c for c in inputs.schema.get("input") or ()
               if c["name"] not in keys and c["name"] in inputs.train.columns]
    found = _range(columns) + _infinity(columns)
    for column in (c for c in columns if c.get("dtype") == "string"):
        try:
            value = unseen_value(inputs, anchor, column)
        except (KeyError, TypeError, ValueError) as exc:
            notes.append(f"no edge_unseen preset for {column['name']}: the Recipe refused the "
                         f"candidate codes ({type(exc).__name__}: {exc})")
            continue
        if value is None:
            remarks.append(f"no edge_unseen preset for {column['name']}: the Recipe reports "
                           "none of its candidates unseen (it encodes no category there)")
            continue
        found.append(("edge_unseen", {"column": column["name"], "value": value},
                      {column["name"]: value}))
    ids = set()
    presets = tuple({"id": _edge_id(kind, facts, ids), "kind": kind, "group": "edge",
                     "stratum": neutral, "expected": None, "facts": facts,
                     **ctx.samples.body(base_key, edit)} for kind, facts, edit in found)
    return Edges(presets, tuple(notes), tuple(remarks))


def _range(columns) -> list:
    """The first input with a positive training maximum, ten times over it."""
    ranged = [c for c in columns if (c.get("range_seen") or [None, None])[1] is not None
              and float(c["range_seen"][1]) > 0]
    if not ranged:
        return []
    column = ranged[0]
    value = float(column["range_seen"][1]) * RANGE_FACTOR
    return [("edge_range", {"column": column["name"], "factor": RANGE_FACTOR},
             {column["name"]: value})]


def _infinity(columns) -> list:
    names = [c["name"] for c in columns if c.get("non_finite_seen")]
    if not names:
        return []
    return [("edge_infinity", {"columns": names}, {n: float("inf") for n in names})]


def _edge_id(kind, facts, taken) -> str:
    base = kind.replace("_", "-")
    if kind == "edge_unseen":
        base = f"{base}-{ID_TEXT.sub('-', str(facts['column'])).strip('-')[:48] or 'column'}"
    name, k = base, 1
    while name in taken:
        k += 1
        name = f"{base}-{k}"
    taken.add(name)
    return name


def unseen_value(inputs, anchor, column):
    """A code of `column` the Recipe was never fitted on, confirmed by asking the Recipe with
    the anchor row's other values: the digit codes above the train part's largest, then (for a
    column that is not a digit code) `<column>-unseen-<k>`. None when no candidate is unseen."""
    name = column["name"]
    seen = inputs.train[name].dropna()
    digits = [int(float(v)) for v in seen.astype(str) if re.fullmatch(r"[0-9]+(\.0)?", v)]
    candidates = []
    if digits:
        top = max(digits)
        candidates += [str(v) for v in range(top + 1, top + 1 + DIGIT_CANDIDATES)]
    if column.get("pattern") != DIGITS:
        candidates += [f"{name}-unseen-{k}" for k in range(TEXT_CANDIDATES)]
    if not candidates:
        return None
    base = inputs.train.iloc[[int(anchor)]]
    frame = pd.concat([base] * len(candidates), ignore_index=True)
    frame[name] = pd.Series(candidates, dtype="string")
    unseen = _asked(inputs.recipe, frame)
    return next((c for c, u in zip(candidates, unseen) if name in u), None)


def _asked(recipe, frame) -> list:
    """The Recipe's unseen columns per row, with an empty lag history (the lags do not matter:
    the question is which category an encoder was fitted on)."""
    lagged = [c for c in getattr(recipe, "HISTORY_COLUMNS", None) or () if c not in frame]
    history = frame.iloc[0:0].assign(**{c: pd.Series(dtype="float64") for c in lagged})
    _shaped, unseen = recipe.transform(frame, history=history, with_unseen=True)
    return [tuple(u) for u in unseen]
