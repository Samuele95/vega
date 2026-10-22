"""
metric_gates — the metrics whose meaning depends on the data, decided on the train part.

eval-contract §3 reports `mape` for a regression only "when |y| > 0.1 for 99 % of train rows": a
percentage error divides by the truth, and on a target near zero it measures the denominator
rather than the model. The rule is the catalog's (`metric_gates`), read here once; the decision
reads the **train** part's targets only — never the test part, whose values may not choose what
is reported about them — and is recorded in `metrics.json.gates` with the share it found, so a
withheld metric is said with its reason rather than silently missing.

Pure functions and a frozen value; no argv, no printing.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

__all__ = ["GATES", "GateReading", "decide", "decided"]

GATES = json.loads((Path(__file__).resolve().parent / "catalog.json")
                   .read_text(encoding="utf-8")).get("metric_gates") or {}


@dataclass(frozen=True)
class GateReading:
    """One gate's decision: whether the metric is reported, on what share, and why."""

    metric: str
    applies: bool
    share: float | None
    rule: str
    why: str

    def as_json(self) -> dict:
        return {"applies": self.applies, "share": self.share, "rule": self.rule,
                "part": "train", "why": self.why}

    @property
    def note(self) -> str:
        """The note a withheld metric carries in every model's record."""
        return f"withheld by its gate ({self.rule}): {self.why}"


def decide(metric: str, rule: Mapping, y_train) -> GateReading:
    """Whether `metric` is reported, read off the train targets."""
    import numpy as np
    text = str(rule.get("rule") or metric)
    if y_train is None:
        return GateReading(metric, False, None, text,
                           "the train part carries no target to decide the gate on")
    values = np.asarray(y_train, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if not values.size:
        return GateReading(metric, False, None, text,
                           "the train part holds no finite target to decide the gate on")
    share = float(np.mean(np.abs(values) > float(rule["abs_y_above"])))
    need = float(rule["share"])
    if share >= need:
        return GateReading(metric, True, round(share, 4), text,
                           f"|y| > {rule['abs_y_above']} on {share:.2%} of the train rows")
    return GateReading(metric, False, round(share, 4), text,
                       f"|y| > {rule['abs_y_above']} on only {share:.2%} of the train rows "
                       f"(the gate asks {need:.0%}); "
                       + str(rule.get("why") or "the metric would not mean what it says"))


def decided(kind: str, names: Iterable[str], y_train, gates: Mapping = GATES) -> dict:
    """{metric: GateReading} for every gated metric among `names` on a task the gate covers —
    applied or not, so the record says the gate was decided either way."""
    out = {}
    for name in names:
        rule = gates.get(name)
        if rule is None or kind not in (rule.get("tasks") or ()):
            continue
        out[name] = decide(name, rule, y_train)
    return out
