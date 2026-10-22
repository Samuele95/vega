"""
story_sections — what `resolve_story.StoryResolver` writes from an authored plan that passed
V1–V15: the values table of `vega/story@2` (`StoryValues`) and its four sections
(`SectionWriter`: `context`, `network`, `experiments`, `results`, and `many` when a monitor note
was written), contract §4.

**Values hold only what some field references** (§4.1): every bare placeholder of every text
the story carries (`StoryValues.text`), and every structured key — a context fact, a run's
primary, a model's value, a results fact, the breakdown's names, values and weak flags, the
verdict's difference, a code note's report number and page. Each value comes with its format
and its ref, the fact's own; the few values the resolver makes (the weak flags; the numbers and
pages a proposed report role binds) carry the ref they were computed from. The reserved values
come first: `primary`, the network's primary on the sealed test as `{value, ci, n}`, and
`ci_level`, the evaluation's bootstrap level — absent on an evaluation that records none, with
`WARN:` (the page then prints the interval without a level).

**The sections.** The context steps in the lane's order (`context_steps.by_shape`); the network
choices in the authored order, each with its `focus@1` (`plan_rules.focus_of`: a tested input the
text names, else the block, else the decision family); the runs in the draft's order (the
metric's direction), each with its authored label, the selected and reference flags from the
facts; the results with the network first among the models, the results facts the draft proposed
with their notes, the breakdown and the verdict as the evaluation recorded it (an evaluation
that records no `earned` is refused: the page never softens or invents a verdict; a missing
`beaten_by_baseline` — no baseline could be compared — is false; a verdict that is not
assessable — `earned` null with its reason, A30 — is carried as `earned` and
`beaten_by_baseline` null with the reason's values key, never as a win or a loss), and the
limits in the authored order.

**`weak`** (R-E3, `breakdown@1`): a unit is weak when its value is worse than the trivial
baseline's value for that unit, in the direction the breakdown reads; and, where the fact set
carries per-unit intervals and the evaluation's bar (the breakdown group's optional `intervals`
and `bar` keys), when the interval's far end is worse than the bar. With neither a trivial side
nor a bar, `weak` is null and a `WARN:` says so. The flags are the value
`<values key>_weak`, a list of booleans, its ref the trivial side's (or the bar's).
"""
from __future__ import annotations

import re
from typing import Mapping, Sequence

from digit_rule import placeholders
from plan_rules import focus_of
from vega_show_core import CATALOG, FactMissing, FactSet, ShowError, fact_key, thaw

__all__ = ["StoryValues", "SectionWriter", "weak_flags", "estimate"]

_KEY = re.compile(CATALOG["fact_key_pattern"])
_METRIC = re.compile(r"^[a-z][a-z0-9_]*$")
_NETWORK = "network"


def estimate(value) -> dict:
    """A reserved `{value, ci, n}` from a fact value (an estimate, or a bare number)."""
    doc = value if isinstance(value, Mapping) else {"value": value}
    n = doc.get("n")
    return {"value": doc.get("value"), "ci": doc.get("ci"),
            "n": int(n) if isinstance(n, (int, float)) and float(n).is_integer() else None}


def weak_flags(values: Sequence, trivial: Sequence | None, direction: str,
               intervals: Sequence | None = None, bar=None) -> list[bool]:
    """Per unit: worse than the trivial side's unit, or (intervals and a bar given) the
    interval's far end worse than the bar, in `direction` (`max`: lower is worse)."""
    def worse(a, b) -> bool:
        return a < b if direction == "max" else a > b
    out = []
    for i, value in enumerate(values):
        flag = trivial is not None and worse(value, trivial[i])
        if intervals is not None and bar is not None:
            low, high = intervals[i]
            flag = flag or worse(high if direction == "max" else low, bar)
        out.append(bool(flag))
    return out


class StoryValues:
    """The story's `values`, `formats` and `refs`: the facts some field references, and the
    values the resolver makes."""

    def __init__(self, facts: FactSet) -> None:
        self._facts, self._used, self._made = facts, {}, {}
        self._markers: list[str] = []
        self.placeholders_n = 0

    @property
    def markers(self) -> tuple[str, ...]:
        return tuple(self._markers)

    def use(self, key) -> str:
        """A fact key some field references (it must be one: V3 saw to that)."""
        if key not in self._made:
            self._used[key] = self._facts.get(key)
        return key

    def text(self, text: str) -> str:
        """An authored text placed in the story: its bare placeholders are used facts."""
        for p in placeholders(text):
            self.placeholders_n += 1
            if not p.is_lexicon_path:
                self.use(p.name)
        return text

    def make(self, key: str, value, fmt: str, ref: str) -> str:
        """A value the resolver computed or bound; never one a fact already holds."""
        if not _KEY.fullmatch(key) or key in self._facts.keys():
            raise ShowError(f"resolve: the made value {key!r} is no free key of the fact set")
        self._made[key] = (value, fmt, ref)
        return key

    def value(self, key: str):
        return self._made[key][0] if key in self._made else self._facts.get(key).json_value

    def tables(self, primary_key: str) -> tuple[dict, dict, dict]:
        """(values, formats, refs), the reserved values first, then every key in name order."""
        primary = self._facts.get(primary_key)
        values = {"primary": estimate(primary.json_value)}
        formats, refs = {"primary": "ci"}, {"primary": str(primary.ref)}
        try:
            level = self._facts.get("eval_ci_level")
            values["ci_level"], formats["ci_level"], refs["ci_level"] = \
                level.json_value, level.fmt, str(level.ref)
        except FactMissing:
            absent = CATALOG["reserved_values"]["ci_level"]["absent"]
            self._markers.append(absent if absent.startswith("WARN:") else f"WARN: {absent}")
        for key in sorted({*self._used, *self._made}):
            if key in self._made:
                values[key], formats[key], refs[key] = self._made[key]
            else:
                fact = self._used[key]
                values[key], formats[key], refs[key] = fact.json_value, fact.fmt, str(fact.ref)
        return values, formats, refs


class SectionWriter:
    """The story's sections, from the authored slots and the draft's proposals."""

    def __init__(self, facts: FactSet, values: StoryValues, catalog: Mapping = CATALOG) -> None:
        self._facts, self._values, self._cat = facts, values, catalog
        self._groups = thaw(facts.groups)
        self._markers: list[str] = []

    @property
    def markers(self) -> tuple[str, ...]:
        return tuple(self._markers)

    def sections(self, authored: Mapping, proposals: Mapping) -> list[dict]:
        out = [self.context(authored["context"]), self.network(authored["network"]),
               self.experiments(authored["experiments"], proposals),
               self.results(authored["results"], proposals)]
        note = authored.get("monitor_note")
        return out + ([{"id": "many", "monitor_note": self._values.text(note)}]
                      if isinstance(note, str) else [])

    def context(self, ctx: Mapping) -> dict:
        v, steps = self._values, ctx["steps"]
        order = self._cat["context_steps"]["by_shape"][self._facts.task.sample_shape]
        return {"id": "context", "steps": [{"id": s, "text": v.text(steps[s])} for s in order],
                "facts": [{"label": v.text(f["label"]), "value": v.use(f["fact"])}
                          for f in ctx["facts"]]}

    def network(self, net: Mapping) -> dict:
        v = self._values
        by_id = {str(d.get("id")): d for d in self._groups.get("decisions") or ()}
        choices = []
        for choice in net["choices"]:
            did, title, text = choice["decision"], v.text(choice["title"]), v.text(choice["text"])
            choices.append({"title": title, "text": text,
                            "focus": focus_of(by_id.get(did, {"id": did}), f"{title} {text}",
                                              self._cat)})
        return {"id": "network", "lead": v.text(net["lead"]), "choices": choices}

    def experiments(self, exp: Mapping, proposals: Mapping) -> dict:
        v, labels = self._values, exp["labels"]
        runs = [{"id": r["id"], "value": v.use(r["primary"]), "label": v.text(labels[r["id"]]),
                 "selected": bool(r.get("selected")), "reference": bool(r.get("reference"))}
                for r in proposals.get("runs") or ()]
        return {"id": "experiments", "lead": v.text(exp["lead"]), "caption": v.text(exp["caption"]),
                "metric": self._metric(), "runs": runs}

    def _metric(self) -> dict:
        """The metric the runs were chosen by: the selection's, when it names one with a
        direction; else the task's primary."""
        sel, task = self._groups.get("selection") or {}, self._facts.task
        name, direction = sel.get("primary_name"), sel.get("direction")
        if isinstance(name, str) and _METRIC.fullmatch(name) and direction in ("max", "min"):
            return {"name": name, "direction": direction}
        return {"name": task.primary, "direction": task.direction}

    def results(self, res: Mapping, proposals: Mapping) -> dict:
        v, notes, keys = self._values, res.get("notes") or {}, self._facts.keys()
        models = sorted((m for m in self._groups.get("models") or () if m.get("primary") in keys),
                        key=lambda m: m.get("name") != _NETWORK)
        facts = []
        for key in proposals.get("results_facts") or ():
            if key in keys:
                # the sealed part's size reads as the test's own fact, n_test
                fact = {"metric": "n_test" if key == "test_rows" else key.removeprefix("eval_network_"),
                        "value": v.use(key)}
                facts.append({**fact, "note": v.text(notes[key])} if key in notes else fact)
        return {"id": "results", "headline": v.text(res["headline"]),
                "models": [{"name": m["name"], "value": v.use(m["primary"]),
                            "selected": m["name"] == _NETWORK} for m in models],
                "facts": facts, "breakdown": self._breakdown(res.get("breakdown_caption")),
                "verdict": self._verdict(),
                "limits": [self._limit(lim) for lim in res["limits"]]}

    def _limit(self, lim: Mapping) -> dict:
        """An authored limit, with the record's own words beside it when the facts hold them
        (`recorded`: a warning's line, the ship stance, the small-support classes) — quoted on
        the page as the record wrote them, so a limit never stands in for the words it cites."""
        v = self._values
        out = {"id": lim["id"], "title": v.text(lim["title"]), "text": v.text(lim["text"])}
        recorded = next((str(f["recorded"]) for f in self._groups.get("limits") or ()
                         if f.get("id") == lim["id"] and f.get("recorded")), None)
        return {**out, "recorded": recorded} if recorded else out

    def _breakdown(self, caption) -> dict | None:
        group = self._groups.get("breakdown")
        if not group:
            if caption is not None:
                self._markers.append("WARN: a breakdown caption on a lane whose evaluation has no "
                                     "per-unit breakdown; it is not shown")
            return None
        direction = group.get("direction")
        if direction not in ("max", "min"):
            self._markers.append(f"WARN: the {group.get('kind')} breakdown reads a metric with no "
                                 "max/min direction; it is not shown")
            return None
        v = self._values
        return {"kind": group["kind"], "metric": group["metric"], "names": v.use(group["names"]),
                "values": v.use(group["values"]), "weak": self._weak(group, direction),
                "direction": direction, "caption": None if caption is None else v.text(caption)}

    def _weak(self, group: Mapping, direction: str) -> str | None:
        """The weak flags' values key (module docstring), or None when nothing to read against."""
        fact = self._fact_value
        values, trivial = fact(group["values"]), fact(group.get("trivial"))
        intervals, bar = fact(group.get("intervals")), fact(group.get("bar"))
        bar = bar.get("value") if isinstance(bar, Mapping) else bar
        if trivial is None and (intervals is None or bar is None):
            self._markers.append(f"WARN: the {group['kind']} breakdown has no trivial side and no "
                                 "bar: no unit is marked weak")
            return None
        against = group.get("trivial") if trivial is not None else group.get("bar")
        flags = weak_flags(values, trivial, direction, intervals, bar)
        return self._values.make(fact_key((group["values"], "weak")), flags, "list",
                                 str(self._facts.get(against).ref))

    def _fact_value(self, key):
        if not isinstance(key, str):
            return None
        try:
            return self._facts.get(key).json_value
        except FactMissing:
            return None

    def _verdict(self) -> dict:
        verdict, names = self._groups.get("verdict") or {}, self._cat["result_models"]["names"]
        earned, beaten = verdict.get("earned"), verdict.get("beaten_by_baseline")
        if earned is None and verdict.get("reason") is not None:
            return self._not_assessable(verdict)
        if not isinstance(earned, bool) or not isinstance(beaten, (bool, type(None))):
            raise ShowError("resolve facts.groups.verdict: the evaluation records no verdict "
                            "(earned); the page carries the evaluation's verdict and never "
                            "invents one — re-run /vega-eval")
        beaten = bool(beaten)  # absent when no baseline was compared: none was found better
        strongest = verdict.get("strongest_baseline")
        if strongest not in names or strongest == _NETWORK:
            if strongest is not None:
                self._markers.append(f"WARN: the strongest baseline {strongest!r} is no result "
                                     "model of the catalog; the verdict names none")
            strongest = None
        diff = verdict.get("diff") if verdict.get("diff") in self._facts.keys() else None
        if earned is False:
            # the contract's WARN: an unearned verdict is shown as recorded and said in the markers,
            # which step 8 carries into the layer
            self._markers.append(f"WARN: verdict not earned{' — beaten by ' + strongest if beaten and strongest else ''}"
                                 f" (strongest baseline {strongest or 'none named'}, difference "
                                 f"{diff or 'not recorded'}): the page shows it as recorded")
        return {"earned": earned, "beaten_by_baseline": beaten, "strongest_baseline": strongest,
                "diff": None if diff is None else self._values.use(diff)}

    def _not_assessable(self, verdict: Mapping) -> dict:
        """A30: the evaluation recorded no paired verdict (`earned` null) and why — neither a
        win nor a loss, so no baseline, no difference and `beaten_by_baseline` null, never
        false; the reason is a values key the page says in words. A record that names a
        baseline, a difference or a beaten flag beside a null `earned` contradicts itself."""
        reason = verdict["reason"]
        named = [k for k in ("beaten_by_baseline", "strongest_baseline", "diff")
                 if verdict.get(k) is not None]
        if named or reason not in self._facts.keys():
            what = f"names {', '.join(named)}" if named else "has no reason fact"
            raise ShowError("resolve facts.groups.verdict: a verdict that is not assessable "
                            f"(earned null) {what}; the page carries the evaluation's verdict "
                            "and never invents one — re-run /vega-eval")
        return {"earned": None, "beaten_by_baseline": None, "strongest_baseline": None,
                "diff": None, "reason": self._values.use(reason)}
