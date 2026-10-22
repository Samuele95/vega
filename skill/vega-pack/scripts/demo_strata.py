"""
demo_strata — the strata of a demonstration sample, per reading, drawn from the train part only
(bundle-contract §7.4; D2 §8.4; build plan R-D8).

    Truths.of(inputs, samples) -> Truths                  each candidate's truth, typed
    draw(inputs, samples, truths, scorer) -> Draw          the strata and their members

| reading | strata |
|---|---|
| binary, multiclass, ordinal (a sequence task's class base alike) | one per class present in train |
| multilabel | one per label (the candidates where it is positive), plus `none` (no label positive) |
| regression, quantile, forecast (a sequence task's value base alike) | five train quantile `bin`s of the truth (forecast: the first horizon's target), their edges recorded |
| reconstruction | `all` (5·n random candidates) and `flagged` (the top n of a 5 000-candidate pool by the reconstruction error) |
| anomaly, labelled | the classes |
| anomaly, unlabelled | `typical` (the lowest-score n of the pool) and `flagged` (score ≥ threshold, possibly empty — and then said so) |

Members are drawn with one generator seeded 7, in strata order, n per stratum (`--demo-sample`),
so the same bundle inputs draw the same sample. The neutral stratum — the base of the edge
presets — is chosen by counts, never by name: the most frequent class, `none` for multilabel,
the middle bin, `all`, `typical`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = ["Draw", "POOL", "Stratum", "Truths", "draw"]

POOL = 5000                     # candidates scored to find the score strata
BINS = 5
SCORE_READINGS = ("reconstruction", "anomaly-reconstruction", "anomaly-svdd")


@dataclass(frozen=True)
class Stratum:
    """One stratum: its id and kind, the data's own name for it (a class or a label, else
    None), a bin's edges, the candidates drawn into it (sorted keys) and how many candidates of
    the train part it holds."""

    id: str
    kind: str
    name: str | None
    range: tuple | None
    members: tuple
    size: int

    def as_json(self) -> dict:
        return {"id": self.id, "kind": self.kind, "name": self.name, "n": len(self.members),
                "range": list(self.range) if self.range is not None else None}


@dataclass(frozen=True)
class Draw:
    """The strata in order, the neutral one's id, and what the draw has to say (WARN lines)."""

    strata: tuple
    neutral: str | None
    notes: tuple = field(default=())


class Truths:
    """Each candidate's truth as MANIFEST.output.truth types it: a class name, a list of label
    names, a number (a forecast: one per horizon step), or None for a reading without truth.
    `keys` are the candidates that have one, `array` their truths in the same order; a
    candidate whose truth is missing or outside the record's classes has none and is left out
    of the class, label and bin strata (`dropped`)."""

    def __init__(self, keys, array, kind, dropped: int):
        self.keys = np.asarray(keys, dtype=np.int64)
        self.array, self.kind, self.dropped = array, kind, dropped
        self.values = dict(zip(self.keys.tolist(), array))

    @staticmethod
    def of(inputs, samples) -> "Truths":
        truth = inputs.output.get("truth") or {}
        kind, column = truth.get("type"), truth.get("column")
        keys = samples.keys
        if kind is None or not column or column not in inputs.train.columns:
            return Truths(keys, np.full(len(keys), None, dtype=object), None, 0)
        cells = inputs.train[column].to_numpy(dtype=object)[samples.truth_positions(keys)]
        got = _Reader(inputs.output, inputs.schema).read(kind, cells)
        keep = np.asarray([g is not None for g in got], dtype=bool)
        return Truths(keys[keep], got[keep], kind, int((~keep).sum()))

    def first(self, key) -> float:
        """A value truth's number (a forecast's first horizon step)."""
        value = self.values[int(key)]
        return float(value[0] if isinstance(value, list) else value)

    def firsts(self) -> np.ndarray:
        """Every value truth's number, in `keys` order."""
        return np.asarray([v[0] if isinstance(v, list) else v for v in self.array],
                          dtype=np.float64)


class _Reader:
    """Raw truth cells read the way the service reads a request's truth (serve/rows.py)."""

    def __init__(self, output, schema):
        self._classes = [str(c) for c in output.get("classes") or ()]
        self._decode = dict((schema.get("label") or {}).get("decode") or {})
        self._sep = (output.get("truth") or {}).get("sep")
        self._horizons = output.get("reading") == "forecast"

    def read(self, kind, cells) -> np.ndarray:
        """One truth (or None) per row of `cells` (rows × the truth's positions)."""
        if kind == "number":
            return self._numbers(cells)
        one = self.one_class if kind == "class" else self.labels
        codes, uniques = pd.factorize(pd.Series(cells[:, 0], dtype=object), use_na_sentinel=True)
        table = np.empty(len(uniques) + 1, dtype=object)
        for i, cell in enumerate(uniques):              # element by element: a truth may be
            table[i] = one(cell)                        # a list, which numpy would unpack
        table[-1] = one(None)                           # the missing cells (code −1)
        return table[codes]

    def _numbers(self, cells) -> np.ndarray:
        values = pd.to_numeric(pd.Series(cells.ravel(), dtype=object), errors="coerce") \
            .to_numpy(dtype=np.float64).reshape(cells.shape)
        ok = np.isfinite(values).all(axis=1)
        out = np.full(len(values), None, dtype=object)
        for i in np.flatnonzero(ok):
            out[i] = [float(v) for v in values[i]] if self._horizons else float(values[i, 0])
        return out

    def one_class(self, cell):
        if _missing(cell):
            return None
        text = _spelling(cell)
        name = self._decode.get(text, self._decode.get(str(cell), text))
        return name if name in self._classes else None

    def labels(self, cell):
        """A multilabel truth: the names in one text split on `sep`; an empty cell is none."""
        if _missing(cell):
            return []
        names = [n.strip() for n in str(cell).split(self._sep or "|") if n.strip()]
        if any(n not in self._classes for n in names):
            return None
        return [c for c in self._classes if c in set(names)]


def _missing(cell) -> bool:
    return cell is None or cell is pd.NA or (isinstance(cell, (float, np.floating))
                                              and math.isnan(float(cell)))


def _spelling(cell) -> str:
    """A class cell as text: an integral number without its `.0` (a label column read as float
    because it had a gap)."""
    if isinstance(cell, (float, np.floating)) and float(cell).is_integer():
        return str(int(cell))
    return str(cell.item() if isinstance(cell, np.generic) else cell)


# --------------------------------------------------------------------------- the draw

def draw(inputs, samples, truths: Truths, scorer) -> Draw:
    """The strata of the bundle's reading, members drawn; `scorer(keys) -> np.ndarray` is the
    reading's per-candidate score (the Predictor's), read only for the score strata."""
    rng = np.random.default_rng(inputs.seed)
    n = int(inputs.per_stratum)
    reading = inputs.output.get("reading")
    if reading in SCORE_READINGS and truths.kind is None:
        return _score_strata(inputs, samples, scorer, rng)
    notes = ((f"{truths.dropped} train candidate(s) carry no readable truth and are left out "
              "of the strata",) if truths.dropped else ())
    if truths.kind == "class":
        got = _class_strata(inputs.output, truths, n, rng)
    elif truths.kind == "labels":
        got = _label_strata(inputs.output, truths, n, rng)
    elif truths.kind == "number":
        got = _bin_strata(truths, n, rng)
    else:
        return Draw((), None, notes + ("the reading has no truth to stratify by",))
    return Draw(got.strata, got.neutral, notes + got.notes)


def _take(rng, members, n) -> tuple:
    """n members at random (all when fewer), as sorted keys."""
    members = np.asarray(members, dtype=np.int64)
    if len(members) > n:
        members = rng.choice(members, n, replace=False)
    return tuple(sorted(int(m) for m in members))


def _class_strata(output, truths, n, rng) -> Draw:
    """One stratum per class present in train, in the record's class order; the neutral one
    the most frequent (the first of them on a tie)."""
    classes = [str(c) for c in output.get("classes") or ()]
    by_class = {c: truths.keys[truths.array == c] for c in classes}
    strata = tuple(Stratum(f"class-{i}", "class", c, None, _take(rng, by_class[c], n),
                           len(by_class[c]))
                   for i, c in enumerate(classes) if len(by_class[c]))
    neutral = max(strata, key=lambda s: s.size).id if strata else None
    return Draw(strata, neutral)


def _label_strata(output, truths, n, rng) -> Draw:
    """One stratum per label with a positive candidate, then `none` (no label positive) —
    listed even when empty. The neutral one is `none`, else the most frequent label."""
    labels = [str(c) for c in output.get("classes") or ()]
    strata = []
    for i, name in enumerate(labels):
        members = truths.keys[np.asarray([name in v for v in truths.array], dtype=bool)]
        if len(members):
            strata.append(Stratum(f"label-{i}", "label", name, None, _take(rng, members, n),
                                  len(members)))
    labelled = list(strata)
    empty = truths.keys[np.asarray([not v for v in truths.array], dtype=bool)]
    strata.append(Stratum("none", "none", None, None, _take(rng, empty, n), len(empty)))
    notes = () if len(empty) else ("stratum none is empty: every train candidate has a label",)
    neutral = "none" if len(empty) else (max(labelled, key=lambda s: s.size).id if labelled
                                    else None)
    return Draw(tuple(strata), neutral, notes)


def _bin_strata(truths, n, rng) -> Draw:
    """Five quantile bins of the train truth (a forecast's first step), empty ones dropped; the
    last bin holds its upper edge. The neutral one is the middle bin."""
    keys, values = truths.keys, truths.firsts()
    if not len(keys):
        return Draw((), None)
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, BINS + 1)))
    if len(edges) == 1:
        edges = np.asarray([edges[0], edges[0]])
    where = np.searchsorted(edges[1:-1], values, side="right")
    strata = []
    for b in range(len(edges) - 1):
        members = keys[where == b]
        if len(members):
            strata.append(Stratum(f"bin-{len(strata)}", "bin", None,
                                  (float(edges[b]), float(edges[b + 1])),
                                  _take(rng, members, n), len(members)))
    return Draw(tuple(strata), strata[len(strata) // 2].id if strata else None)


def _score_strata(inputs, samples, scorer, rng) -> Draw:
    """reconstruction: `all` and `flagged`; an unlabelled anomaly: `typical` and `flagged`,
    read on a pool of at most POOL candidates scored by the bundle's own Predictor."""
    n, keys = int(inputs.per_stratum), samples.keys
    pool = np.asarray(sorted(rng.choice(keys, min(POOL, len(keys)), replace=False)),
                      dtype=np.int64) if len(keys) else keys
    scores = np.asarray(scorer(pool), dtype=np.float64) if len(pool) else np.zeros(0)
    finite = np.isfinite(scores)
    order = pool[finite][np.argsort(scores[finite], kind="stable")]
    if inputs.output.get("reading") == "reconstruction":
        first = Stratum("all", "all", None, None, _take(rng, keys, 5 * n), len(keys))
        flagged = tuple(sorted(int(k) for k in order[::-1][:n]))
        second = Stratum("flagged", "flagged", None, None, flagged, len(order))
        return Draw((first, second), "all")
    first = Stratum("typical", "typical", None, None,
                    tuple(sorted(int(k) for k in order[:n])), len(order))
    threshold = inputs.output.get("threshold")
    over = (pool[finite & (scores >= float(threshold))] if threshold is not None
            else np.zeros(0, dtype=np.int64))
    second = Stratum("flagged", "flagged", None, None, _take(rng, over, n), len(over))
    notes = () if len(over) else (
        "stratum flagged is empty: " + ("no candidate of the pool scores at or above the "
                                        f"threshold {threshold}" if threshold is not None
                                        else "the bundle records no threshold"),)
    return Draw((first, second), "typical", notes)
