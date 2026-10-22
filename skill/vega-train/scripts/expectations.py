"""
expectations — a rung's `expect` as a check the ledger can judge, and the judging.

A rung's `expect` is the prediction its hypothesis makes, in words; `expect_check` is the same
prediction in a closed grammar a script can evaluate against the ledger, and `expect_held` on
the ledger row is that evaluation (run-contract §3-§4). The words are the skill's to edit, and
so is the check; what no one does is decide afterwards, by reading the numbers, whether the
prediction held — the trainer judges it when the row is written, and again for every row still
pending whenever another row lands (the reference run may run later).

The grammar (one object, or `{"any": [...]}` for a disjunction):

    {"relation": "learns"}
        held when the run is not diverged or flat (no D1, D2) and improved on its first epoch
    {"relation": "better" | "worse" | "not_better" | "not_worse",
     "metric": "valid.primary" | "valid.loss" | "train.primary" | "train.loss" | "gap.loss",
     "than": "<run id>", "margin": <fraction of |reference|, default 0>}
        compared in the metric's direction: the primary in `metrics.selection.mode`, a loss and
        the train/valid loss gap downwards (a wider gap is worse)

A check that cannot be judged yet (the reference run has not run, the value was not recorded)
is `null` with the reason in `expect_evidence`, never a guess.

Pure: nothing here reads a file, prints or exits (code-contract §4's functional core).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

__all__ = ["ExpectError", "ExpectCheck", "RELATIONS", "METRICS", "judge_row", "rejudge",
           "check_for"]

RELATIONS = ("learns", "better", "worse", "not_better", "not_worse")
METRICS = ("valid.primary", "valid.loss", "train.primary", "train.loss", "gap.loss")
#: the diagnostics that make a run one that did not learn (the selection's ineligible codes)
NOT_LEARNING = ("D1", "D2")


class ExpectError(ValueError):
    """An `expect_check` outside the grammar, with what is wrong named."""


@dataclass(frozen=True)
class ExpectCheck:
    """One checkable prediction, or a disjunction of them (`any_of`)."""

    relation: str | None = None
    metric: str = "valid.primary"
    than: str | None = None
    margin: float = 0.0
    any_of: tuple = ()

    @staticmethod
    def parse(raw: Any, run_id: str | None = None) -> "ExpectCheck":
        """The check `raw` states, validated; `run_id` is the rung's own id, which a check may
        not compare against."""
        if not isinstance(raw, Mapping):
            raise ExpectError(f"an expect_check is an object, got {raw!r}")
        if "any" in raw:
            extra = sorted(set(raw) - {"any", "text"})
            items = raw.get("any")
            if extra or not isinstance(items, list) or not items:
                raise ExpectError("an `any` expect_check is {\"any\": [<check>, …]} with at "
                                  f"least one check, got {raw!r}")
            return ExpectCheck(any_of=tuple(ExpectCheck.parse(i, run_id) for i in items))
        relation = raw.get("relation")
        if relation not in RELATIONS:
            raise ExpectError(f"expect_check relation {relation!r} is not one of "
                              f"{', '.join(RELATIONS)}")
        extra = sorted(set(raw) - {"relation", "metric", "than", "margin", "text"})
        if extra:
            raise ExpectError(f"expect_check carries {', '.join(extra)}, which the grammar "
                              "does not have")
        if relation == "learns":
            if any(k in raw for k in ("metric", "than", "margin")):
                raise ExpectError("a `learns` expect_check compares with nothing: no metric, "
                                  "than or margin")
            return ExpectCheck(relation="learns")
        metric = raw.get("metric", "valid.primary")
        if metric not in METRICS:
            raise ExpectError(f"expect_check metric {metric!r} is not one of "
                              f"{', '.join(METRICS)}")
        than = raw.get("than")
        if not isinstance(than, str) or not than:
            raise ExpectError(f"a {relation!r} expect_check names the run it compares with "
                              "(`than`)")
        if run_id is not None and than == run_id:
            raise ExpectError(f"the expect_check of {run_id} compares {run_id} with itself")
        margin = raw.get("margin", 0.0)
        if isinstance(margin, bool) or not isinstance(margin, (int, float)) or margin < 0:
            raise ExpectError(f"expect_check margin {margin!r} is not a fraction >= 0")
        return ExpectCheck(relation=relation, metric=metric, than=than, margin=float(margin))

    def as_json(self, primary: str | None = None) -> dict:
        if self.any_of:
            out = {"any": [c.as_json() for c in self.any_of]}
        elif self.relation == "learns":
            out = {"relation": "learns"}
        else:
            out = {"relation": self.relation, "metric": self.metric, "than": self.than}
            if self.margin:
                out["margin"] = self.margin
        if primary is not None:
            out["text"] = self.describe(primary)
        return out

    def references(self) -> set:
        if self.any_of:
            return set().union(*(c.references() for c in self.any_of))
        return {self.than} if self.than else set()

    def describe(self, primary: str | None) -> str:
        if self.any_of:
            return " or ".join(c.describe(primary) for c in self.any_of)
        if self.relation == "learns":
            return "the run learns: no D1/D2 and an improvement on its first epoch"
        words = {"better": "better than", "worse": "worse than", "not_better": "no better than",
                 "not_worse": "at least as good as"}[self.relation]
        margin = f" by more than {self.margin:g} of its value" if self.margin and \
            self.relation in ("better", "worse") else (
                f" within {self.margin:g} of its value" if self.margin else "")
        return f"{_metric_name(self.metric, primary)} {words} {self.than}'s{margin}"


def _metric_name(metric, primary) -> str:
    part, _, key = metric.partition(".")
    return f"{part}.{primary if key == 'primary' and primary else key}"


def _value(row: Mapping, metric: str, primary: str | None):
    part, _, key = metric.partition(".")
    if part == "gap":
        return (row.get("gap") or {}).get(key)
    key = primary if key == "primary" else key
    if part == "valid" and key == primary and row.get("primary") is not None:
        return row.get("primary")
    return (row.get(part) or {}).get(key)


def _upward(metric: str, mode: str) -> bool:
    """Whether a larger value is better for this metric."""
    return metric.endswith(".primary") and mode != "min"


def _verdict(check: ExpectCheck, row, rows, primary, mode) -> tuple:
    if check.any_of:
        verdicts = [_verdict(c, row, rows, primary, mode) for c in check.any_of]
        if any(v is True for v, _ in verdicts):
            return True, "; ".join(e for v, e in verdicts if v is True)
        if all(v is False for v, _ in verdicts):
            return False, "; ".join(e for _, e in verdicts)
        return None, "; ".join(e for v, e in verdicts if v is None)
    if check.relation == "learns":
        fired = [c for c in (row.get("diagnostics") or []) if c in NOT_LEARNING]
        best = row.get("best_epoch")
        if fired:
            return False, f"{row.get('id')} fired {', '.join(fired)}"
        if best is None:
            return None, f"{row.get('id')} recorded no best epoch"
        if int(best) <= 1:
            return False, f"{row.get('id')} never bettered its first epoch (best epoch {best})"
        return True, f"{row.get('id')} learned (best epoch {best}, no D1/D2)"
    name = _metric_name(check.metric, primary)
    ref = next((r for r in rows if r.get("id") == check.than), None)
    if ref is None:
        return None, f"{check.than} has not run yet; judged when its row lands"
    mine, theirs = _value(row, check.metric, primary), _value(ref, check.metric, primary)
    if mine is None or theirs is None:
        who = row.get("id") if mine is None else check.than
        return None, f"{who} recorded no {name}"
    mine, theirs = float(mine), float(theirs)
    band = check.margin * abs(theirs)
    up = _upward(check.metric, mode)
    better = mine > theirs + band if up else mine < theirs - band
    worse = mine < theirs - band if up else mine > theirs + band
    held = {"better": better, "worse": worse, "not_better": not better,
            "not_worse": not worse}[check.relation]
    return held, (f"{name} {row.get('id')} {mine:.6g} against {check.than} {theirs:.6g}"
                  f"{f' (band {band:.3g})' if band else ''}: "
                  f"{'held' if held else 'did not hold'}")


def judge_row(check: ExpectCheck | None, row: Mapping, rows, spec: Mapping) -> tuple:
    """(expect_held, expect_evidence) for one ledger row (or a study's record shaped like one:
    `id`, `primary`) against the ledger's rows."""
    if check is None:
        return None, "no expect_check"
    metrics = (spec or {}).get("metrics") or {}
    mode = (metrics.get("selection") or {}).get("mode", "max")
    return _verdict(check, row, list(rows or []), metrics.get("primary"), mode)


def check_for(rung_check, source: str, catalog: Mapping | None, run_id: str | None = None):
    """(ExpectCheck, None), or (None, why not): the rung's own `expect_check`, else the default
    of the ladder rule its source names (`netspec` the baseline's, `netspec:<path>` the
    spec-alternative's, `vega:ladder/<rule>` that rule's, `train:diagnostic/…` the diagnostic
    rule's) from catalog.json; a `user:` or `analysis:` rung states its own."""
    raw = rung_check
    if raw is None:
        rule = _rule_of(source)
        raw = ((((catalog or {}).get("ladder_rules") or {}).get(rule)) or {}).get(
            "expect_check") if rule else None
        if raw is None:
            return None, (f"the rung states no expect_check, and its source {source!r} names "
                          "no ladder rule whose catalog row carries one")
    try:
        return ExpectCheck.parse(raw, run_id), None
    except ExpectError as exc:
        return None, str(exc)


def _rule_of(source: str) -> str | None:
    source = str(source or "")
    if source == "netspec":
        return "baseline"
    if source.startswith("netspec:"):
        return "spec-alternative"
    if source.startswith("vega:ladder/"):
        return source.split("/", 1)[1]
    if source.startswith("train:diagnostic/"):
        return "diagnostic"
    return None


def rejudge(ledger: Mapping, spec: Mapping) -> int:
    """Judge every row (and every study record) whose `expect_check` is recorded and whose
    verdict is still pending, in place. Returns how many verdicts were settled."""
    rows = list(ledger.get("runs") or [])
    settled = 0
    for target in [*rows, *(ledger.get("hpo") or [])]:
        raw = target.get("expect_check")
        if raw is None or target.get("expect_held") is not None:
            continue
        try:
            check = ExpectCheck.parse(raw)
        except ExpectError as exc:
            target["expect_evidence"] = f"the recorded expect_check is malformed: {exc}"
            continue
        subject = target if "id" in target else {"id": target.get("study"),
                                                  "primary": target.get("best_value")}
        held, evidence = judge_row(check, subject, rows, spec)
        target["expect_evidence"] = evidence
        if held is not None:
            target["expect_held"] = held
            settled += 1
    return settled
