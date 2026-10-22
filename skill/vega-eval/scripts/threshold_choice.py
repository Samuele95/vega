"""
threshold_choice — the decision threshold of a deciding reading, chosen on the **valid** part
and never on test (eval-contract §2, BUILD-PLAN R-C2/R-C3).

A binary or multilabel score is a logit: its threshold is a probability and σ is applied to the
score unconditionally — a logit that happens to lie inside [0, 1] is still a logit (E18). A
multilabel task gets the per-label chooser (each label the threshold that maximises its own f1,
keyed by class name; the one uniform threshold that maximises `f1_micro` is `value`, and a label
valid cannot place takes it). An anomaly score's threshold lives on the score's own scale: the
valid quantile at the expected anomaly rate, else its f1 optimum on a labelled valid part, else
none at all (nothing says where normal ends). The scale of each deciding reading is the
catalog's `threshold_scale`, which `tests/test_catalog_sync.py` holds equal to the vega-pack
reading table's, so a threshold is served on the scale it was chosen on.

Every candidate is scored through the run's `Scoring` (`at_threshold`), so the chooser reads a
metric exactly as the headline will. Pure: arrays in, a frozen `ThresholdChoice` out.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

__all__ = ["THRESHOLD_SCALE", "ThresholdChoice", "check_threshold_arg", "choose_threshold"]

_CATALOG = json.loads((Path(__file__).resolve().parent / "catalog.json")
                      .read_text(encoding="utf-8"))
# the scale each deciding reading's threshold lives on: a probability for a logit, the score's
# own for an anomaly score
THRESHOLD_SCALE = dict(_CATALOG["threshold_scale"])


@dataclass(frozen=True)
class ThresholdChoice:
    """The decision threshold of a deciding reading, and where it came from — chosen on
    **valid**, never on test. `scale` is the space it lives in (`probability`: a binary or
    multilabel score is a logit and σ is applied, always; `score`: an anomaly score's own
    scale); `per_label` holds a multilabel task's thresholds by class name, which the metrics
    read in class order. A reading that decides by argmax, by cumulative count or not at all
    has none: `applies` is false and `value` is None."""

    applies: bool
    value: float | None
    scale: str | None
    source: str | None
    why: str
    per_label: Mapping | None = None

    def metric_threshold(self, classes=()):
        """`MetricOptions.threshold`: the per-label thresholds in class order, else the value;
        None when no threshold applies (the options keep their default, which nothing reads)."""
        if not self.applies:
            return None
        if self.per_label:
            return tuple(float(self.per_label[str(c)]) for c in classes)
        return self.value

    def record(self, kind, effective, metrics_at) -> dict:
        """threshold.json (R-C2): `{applies, value, scale, per_label}` and the provenance."""
        return {"schema": "vega/threshold@1", "task": kind, "task_effective": effective,
                "applies": bool(self.applies), "value": self.value, "scale": self.scale,
                "per_label": dict(self.per_label) if self.per_label else None,
                "source": self.source, "why": self.why, "metrics_at_threshold": metrics_at}


def _no_threshold(why) -> ThresholdChoice:
    return ThresholdChoice(False, None, None, None, why)


def _probability_grid(sigmoid, s):
    """The candidate thresholds, drawn where the metric thresholds: the probabilities σ(score)
    — unconditionally, since a logit that happens to lie inside [0, 1] is still a logit (E18)."""
    import numpy as np
    return np.unique(np.quantile(sigmoid(np.asarray(s, dtype=np.float64)).reshape(-1),
                                 np.linspace(0.01, 0.99, 99)))


def _best_on(grid, value_at):
    """(the candidate with the highest `value_at`, that value); ties keep the lowest."""
    best, best_value = None, -1.0
    for candidate in grid:
        got = value_at(float(candidate))
        if got is not None and math.isfinite(float(got)) and float(got) > best_value:
            best, best_value = float(candidate), float(got)
    return best, best_value


#: the `why` of a threshold the caller asked to be the catalog's 0.5 (`--threshold 0.5`), told
#: apart from a valid-optimum request that fell back to 0.5 — `requested_mode` reads it back
CATALOG_HALF = "the catalog's 0.5"


def requested_mode(record) -> str | None:
    """The `--threshold` mode a further look must keep after a look whose `threshold.json` is
    `record`: `"0.5"` after a look at the catalog's 0.5, `"valid-optimum"` after a valid-optimum
    look (its fallbacks to 0.5 included) and after a user value — which a further look can never
    repeat, so it takes the stage's default rather than a mode chosen knowing the answer — or
    None when no threshold was read (eval-contract §2)."""
    if not isinstance(record, dict) or not record.get("applies"):
        return None
    source = str(record.get("source") or "")
    if source == "vega:default" and record.get("why") == CATALOG_HALF:
        return "0.5"
    return "valid-optimum"


def choose_threshold(scoring, y_valid, s_valid, requested, *, effective, classes=(),
                     rate=None, rate_source=None, applies=True) -> ThresholdChoice:
    """The decision threshold of `effective`'s reading, on its scale, chosen on the valid part:
    `f1` for a binary head (on σ of the logit), `f1` per label for a multilabel head (the
    per-label chooser; `value` is the one threshold that maximises `f1_micro`, the fallback of a
    label that valid cannot place), and the valid quantile at the expected anomaly rate for an
    anomaly score (its f1 optimum on the score's scale when no rate is known but valid is
    labelled). `requested` is `valid-optimum` (None), `0.5` or a first-exposure value."""
    import numpy as np
    scale = THRESHOLD_SCALE.get(effective)
    if not applies or scale is None:
        return _no_threshold(f"a {effective} reading decides by argmax, by cumulative count or "
                             "not at all; no threshold is chosen")
    if requested not in (None, "valid-optimum"):
        if requested == "0.5":
            return ThresholdChoice(True, 0.5, scale, "vega:default", CATALOG_HALF)
        return ThresholdChoice(True, float(requested), scale, "user:turn",
                               "a value the user gave on this first exposure")
    y = None if y_valid is None else np.asarray(y_valid)
    s = None if s_valid is None else np.asarray(s_valid, dtype=np.float64)
    labelled = y is not None and s is not None and len(y) > 0
    if effective == "anomaly":
        return _anomaly_threshold(scoring, y if labelled else None, s, rate, rate_source)
    if not labelled:
        return ThresholdChoice(True, 0.5, scale, "vega:default",
                               "no labelled valid part was available to optimise on")
    if effective == "multilabel":
        return _per_label_threshold(scoring, y, s, classes)
    best, best_f1 = _best_on(_probability_grid(scoring.metrics.sigmoid, s),
                             lambda t: scoring.at_threshold(t).point("f1", y, s))
    if best is None:
        return ThresholdChoice(True, 0.5, scale, "vega:default",
                               "f1 could not be computed on the valid part at any candidate")
    return ThresholdChoice(True, best, scale, "valid-optimum",
                           "the threshold on the valid part that maximises f1 "
                           f"({best_f1:.4g}), on the probability scale (σ of the logit)")


def _per_label_threshold(scoring, y, s, classes) -> ThresholdChoice:
    """One threshold per label, each maximising that label's f1 on valid; the one uniform
    threshold maximising f1_micro is `value`, and a label whose valid column holds one value
    (nothing to separate) takes it."""
    import numpy as np
    sigmoid = scoring.metrics.sigmoid
    value, micro = _best_on(_probability_grid(sigmoid, s),
                            lambda t: scoring.at_threshold(t).point("f1_micro", y, s))
    if value is None:
        value, micro = 0.5, float("nan")
    one = replace(scoring, task="binary")
    per_label, fell_back = {}, []
    for j, name in enumerate(classes):
        column, score = np.asarray(y[:, j]), s[:, j]
        best = None
        if len(np.unique(column)) > 1:
            best, _ = _best_on(_probability_grid(sigmoid, score),
                               lambda t, c=column, z=score: one.at_threshold(t).point("f1", c, z))
        if best is None:
            fell_back.append(str(name))
        per_label[str(name)] = float(value if best is None else best)
    why = ("per label, the threshold on the valid part that maximises that label's f1 (σ of "
           f"its logit); `value` is the one threshold that maximises f1_micro ({micro:.4g})")
    if fell_back:
        why += (f"; {', '.join(fell_back)} hold(s) one value on valid and take(s) it")
    return ThresholdChoice(True, float(value), "probability", "valid-optimum", why, per_label)


def _anomaly_threshold(scoring, y, s, rate, rate_source) -> ThresholdChoice:
    import numpy as np
    if s is None or not len(s):
        return _no_threshold("no valid part was scored, so nothing places the anomaly threshold")
    if rate:
        value = float(np.quantile(s, 1.0 - float(rate)))
        return ThresholdChoice(True, value, "score", "valid-optimum",
                               f"the valid quantile at the expected anomaly rate {rate:.4g} "
                               f"({rate_source}), on the anomaly score's own scale")
    if y is not None and len(np.unique(y.reshape(-1))) > 1:
        grid = np.unique(np.quantile(s, np.linspace(0.01, 0.99, 99)))
        best, best_f1 = _best_on(grid, lambda t: scoring.at_threshold(t).point("f1", y, s))
        if best is not None:
            return ThresholdChoice(True, best, "score", "valid-optimum",
                                   "the threshold on the valid part that maximises f1 "
                                   f"({best_f1:.4g}), on the anomaly score's own scale")
    return _no_threshold("no expected anomaly rate (no labelled train positives, no "
                         "task.anomaly_rate) and no labelled valid part: nothing says where "
                         "normal ends, so the reading answers the score and decides nothing")


def check_threshold_arg(value, effective: str):
    """A `--threshold` this stage can apply, or why not — decided before the seal, since a
    value the metrics cannot read would otherwise fail after the look."""
    if value in (None, "valid-optimum", "0.5"):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return (f"--threshold {value!r} is not valid-optimum, 0.5 or a number")
    if not math.isfinite(number):
        return f"--threshold {value!r} is not a finite number"
    scale = THRESHOLD_SCALE.get(effective)
    if scale == "probability" and not 0.0 <= number <= 1.0:
        return (f"--threshold {value!r} is outside [0, 1]: a {effective} threshold is a "
                "probability (the score is a logit, read through σ)")
    return None
