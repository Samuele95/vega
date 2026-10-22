#!/usr/bin/env python3
"""
autoauthor.py — a mechanical author of `vega/story-plan@2`: the model's half of the plan written
by rule, so the coverage matrix and the acceptance lanes can run extract → draft → author →
resolve → build without a model (D5 §12.1 steps 5–6; risk 13 of the build plan).

    python3 autoauthor.py --plan DRAFT --facts plan/facts.json --out plan/story-plan.json
                          [--report R] [--json]

`AutoAuthor(facts).author(plan)` returns the plan with `status: authored` and every authored
slot filled. It **completes**, it does not overwrite: a slot a carry (`draft_plan.py --carry`)
filled is kept and completed — the mandatory choices it lacks are added, a label for every
proposed run, the mandatory limits, a text for every context step and code step, the
context's facts brought to 2–6 — and the carry's `_unplaced` group is dissolved: each unplaced
input joins the carried group that shares the most members with the draft's group of that input
(the first such group on a tie), which is the recorded completion lane B of the acceptance
relies on. The empty slots are written from the proposals:

- the unattended answers of the forks (`RoleProposer.unattended`, the motif's base or stance),
  the lexicon and the report roles as proposed;
- plain texts that pass V2–V4 and V12 by construction: no digit (a number is a `{fact key}`
  placeholder: a size fact in the data step, a decision's first setting in its choice), words
  of the lane's lexicon as `{row.many}` paths, a tested input named where the draft names one
  (the resolver then focuses the input), each within its length bound.

The plan is then validated (`PlanValidator`, V2–V14; V1 is the resolver's) and every slot a
rule still refuses is written again from the proposals alone, `WARN:` naming the slot and the
rule; a slot that fails twice is left for the refusal to be read. `--report` names the report's
`.aux` when its roles are proposed (else the one the facts pinned is looked for). Exit 0 with
the plan written, 2 when the authored plan still fails a rule (nothing written) or an input
cannot be read.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path
from typing import Mapping

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS.parent))

from carry import UNPLACED  # noqa: E402
from digit_rule import DigitRule, check_placeholders  # noqa: E402
from memory_stances import MemoryStances  # noqa: E402
from plan_rules import PlanPins, PlanValidator, covered_classes  # noqa: E402
from propose_motif import MotifProposer  # noqa: E402
from propose_roles import RoleProposer  # noqa: E402
from resolve_story import report_labels_for  # noqa: E402
from show_ports import LabelSource  # noqa: E402
from vega_show_core import (CATALOG, FactSet, ShowError, read_json, thaw,  # noqa: E402
                            write_json)

__all__ = ["AutoAuthor", "main"]

LENGTHS = CATALOG["authored_lengths"]
_STEP_TEXT = {"problem": "What the network is asked about each {row.one}, and why it matters.",
              "data": "Where the {row.many} come from, and what each records.",
              "windows": "How each answer reads a window of {step.many} of a single series.",
              "graph": "How the {row.many} are linked, and what each {node.one} reads.",
              "cleaning": "How the {source.one} was cleaned before the network read it.",
              "split": "How the {row.many} were split into the parts to learn, choose and test."}
_FACT_LABELS = {"source_rows": "rows in the source", "train_rows": "rows to learn from",
                "valid_rows": "rows to choose with", "test_rows": "rows sealed for the test",
                "shaped_inputs": "network inputs", "source_columns": "source columns",
                "classes_n": "classes"}
_FAMILY_TITLES = {"architecture": "How the network is built",
                  "objective": "What it is trained to do", "optimization": "How it learns",
                  "regularization": "What holds it back"}
_LIMIT_TITLES = {"split": "How the parts were drawn", "exposures": "The test was opened once",
                 "one_seed": "A single seed", "robustness_not_run": "Robustness not checked",
                 "verdict_not_earned": "The verdict was not earned",
                 "verdict_beaten": "A baseline did better",
                 "verdict_not_assessable": "The verdict is not assessable",
                 "ship_stance": "Shipped by stance",
                 "roundtrip_boundary": "Where the round trip restarts",
                 "examples_synthetic": "Synthetic examples", "calibration": "Calibration",
                 "slices_empty": "No slices", "drift_suspected": "Drift suspected",
                 "serving_forced": "Served for the demonstration"}
_WORD = re.compile(r"^[a-z]+$")


class AutoAuthor:
    """The authored half of a story plan, completed by rule from its proposals."""

    def __init__(self, facts: FactSet, labels: LabelSource | None = None,
                 catalog: Mapping = CATALOG) -> None:
        self._facts, self._labels, self._cat = facts, labels, catalog
        self._markers: list[str] = []
        self._slots = _Slots(facts, catalog, self._markers)

    @property
    def markers(self) -> tuple[str, ...]:
        return tuple(self._markers)

    def author(self, plan: Mapping) -> dict:
        plan = copy.deepcopy(dict(plan))
        props, old = plan["proposals"], dict(plan.get("authored") or {})
        plan["authored"] = {slot: self._slot(slot, old.get(slot), props) for slot in old}
        plan["status"] = "authored"
        for _ in range(3):
            problems = [p for p in self.problems(plan) if p.where.startswith("authored.")]
            if not problems:
                break
            for p in problems:
                self._repair(plan["authored"], p, props)
        return plan

    def problems(self, plan: Mapping) -> list:
        pins = PlanPins((plan.get("facts") or {}).get("digest"), plan.get("template") or {})
        return [p for p in PlanValidator(self._facts, pins, self._labels, self._cat).validate(plan)
                if p.level == "ERROR" and p.rule != "V1"]

    def _repair(self, authored: dict, problem, props: Mapping) -> None:
        """Write again what a rule refused: the one text field when its path is a chain of
        keys the fresh slot has too (a carried headline too long), else the whole slot."""
        path = problem.where[len("authored."):].split(".")
        slot = path[0].split("[")[0]          # `input_groups[1].label` is the input_groups slot
        fresh = self._slot(slot, None, props)
        path[0] = slot
        if slot not in authored or "[" in problem.where:
            path = path[:1]
        held, fresh_held = authored, {slot: fresh}
        for key in path[:-1]:
            held, fresh_held = held.get(key), (fresh_held or {}).get(key)
            if not isinstance(held, dict) or not isinstance(fresh_held, Mapping):
                held, path = authored, path[:1]
                fresh_held = {slot: fresh}
                break
        leaf = path[-1]
        if leaf in fresh_held:
            held[leaf] = copy.deepcopy(fresh_held[leaf])
        else:
            held.pop(leaf, None)
        self._markers.append(f"WARN: autoauthor wrote authored.{'.'.join(path)} again "
                             f"({problem.rule}: {problem.message})")

    def _slot(self, slot: str, value, props: Mapping):
        return getattr(self._slots, slot)(value, props)


class _Slots:
    """The writer of each authored slot: `<slot>(value, proposals)` keeps what `value` holds
    that still reads, and completes the rest from the proposals."""

    def __init__(self, facts: FactSet, catalog: Mapping, markers: list) -> None:
        self._facts, self._cat, self._markers = facts, catalog, markers
        self._groups = thaw(facts.groups)
        self._rule = DigitRule.for_facts(facts)

    def locale(self, value, props):
        return value if value in self._cat["locales"] else self._cat["locales"][0]

    def lexicon(self, value, props):
        proposed = {k: {f: e[f] for f in ("one", "many", "article")}
                    for k, e in (props.get("lexicon") or {}).items()}
        kept = {k: e for k, e in (value or {}).items() if k in self._cat["lexicon"]["keys"]
                and isinstance(e, Mapping)}
        return {**proposed, **kept}

    def motif(self, value, props):
        motif = props["motif"]
        return value if value in motif["admissible"] else \
            MotifProposer(motif["modality"], MemoryStances(), self._cat).answer(motif)

    def class_roles(self, value, props):
        if not covered_classes(self._facts.task, self._cat) or not props.get("class_roles"):
            return None
        carried = value if isinstance(value, Mapping) and isinstance(value.get("roles"), Mapping) \
            and isinstance(value.get("source"), Mapping) else None
        roles, _ = RoleProposer(self._facts.task, {}, MemoryStances(), self._cat).unattended(
            props["class_roles"], carried)
        return roles

    def report_roles(self, value, props):
        report = props.get("report_roles") or {}
        if report.get("status") != "proposed":
            return None
        if isinstance(value, Mapping):
            return dict(value)
        return {role: b["label"] for role, b in (report.get("bindings") or {}).items()}

    def headline(self, value, props):
        return value if isinstance(value, str) else "What the network says about each {row.one}"

    def lead(self, value, props):
        return value if isinstance(value, str) else \
            "A network reads the {feature.many} of each {row.one} and answers from them alone."

    def context(self, value, props):
        value = value if isinstance(value, Mapping) else {}
        steps = value.get("steps") if isinstance(value.get("steps"), Mapping) else {}
        keys = self._facts.keys()
        facts = [f for f in value.get("facts") or () if isinstance(f, Mapping)
                 and f.get("fact") in keys][:6]
        spare = [(k, label) for k, label in _FACT_LABELS.items() if k in keys] + \
            [(f.key, "a count of the records") for f in self._facts.facts
             if str(f.kind) in ("count", "size") and f.key not in _FACT_LABELS]
        for key, label in spare:
            if len(facts) >= 4:
                break
            if key not in {f["fact"] for f in facts}:
                facts.append({"label": label, "fact": key})
        order = self._cat["context_steps"]["by_shape"][self._facts.task.sample_shape]
        return {"steps": {s: steps[s] if isinstance(steps.get(s), str) else _STEP_TEXT[s]
                          for s in order}, "facts": facts}

    def network(self, value, props):
        value = value if isinstance(value, Mapping) else {}
        offered = {c["decision"]: c for c in props.get("choices") or ()}
        choices = [dict(c) for c in value.get("choices") or () if isinstance(c, Mapping)
                   and c.get("decision") in offered]
        named = {c["decision"] for c in choices}
        choices += [self._choice(c) for d, c in offered.items()
                    if c["mandatory"] and d not in named]
        spare = [c for d, c in offered.items() if d not in {x["decision"] for x in choices}]
        while len(choices) < min(self._cat["choices"]["min"], len(offered)) and spare:
            choices.append(self._choice(spare.pop(0)))
        extra = len(choices) - self._cat["choices"]["max"]
        if extra > 0:  # drop the latest optional ones first
            optional = [c for c in reversed(choices) if not offered[c["decision"]]["mandatory"]]
            for c in optional[:extra]:
                choices.remove(c)
        lead = value.get("lead") if isinstance(value.get("lead"), str) else \
            "The decisions that shaped the network, each with the reason the chain recorded."
        return {"lead": lead, "choices": _distinct(choices, "title", "choice_title")}

    def _choice(self, proposal: Mapping) -> dict:
        decision = next((d for d in self._groups.get("decisions") or ()
                         if d.get("id") == proposal["decision"]), {})
        text = "The chain recorded why it made this decision"
        params = [(p, k) for p, k in (proposal.get("params") or {}).items() if _WORD.fullmatch(
            str(p).replace("_", ""))]
        text += f"; its {params[0][0].replace('_', ' ')} is {{{params[0][1]}}}." if params else "."
        tested = [n for n in decision.get("features_tested") or () if self._fits(f" {n}", 60)]
        text += f" A run tested it without {tested[0]}." if tested else ""
        return {"decision": proposal["decision"],
                "title": _FAMILY_TITLES.get(proposal.get("family"), "A decision"), "text": text}

    def experiments(self, value, props):
        value = value if isinstance(value, Mapping) else {}
        labels = value.get("labels") if isinstance(value.get("labels"), Mapping) else {}
        out = {}
        for run in props.get("runs") or ():
            old = labels.get(run["id"])
            out[run["id"]] = old if isinstance(old, str) else self._run_label(run)
        return {"lead": value["lead"] if isinstance(value.get("lead"), str) else
                "Each run changed a single thing, and was scored on the validation part.",
                "caption": value["caption"] if isinstance(value.get("caption"), str) else
                "The validation score of each run.", "labels": out}

    def _run_label(self, run: Mapping) -> str:
        proposed = run.get("label") or ""
        if self._fits(proposed, LENGTHS["run_label"]):
            return proposed
        if run.get("reference"):
            return "the spec as engineered"
        return f"run {run['id']}" if self._fits(f"run {run['id']}", LENGTHS["run_label"]) \
            else "a run of the ladder"

    def _fits(self, text: str, bound: int) -> bool:
        """A mechanical text passes the digit rule and the placeholders, within `bound`."""
        return bool(text.strip()) and len(text) <= bound and not \
            self._rule.refusals(text, "t") and not check_placeholders(text, "t", self._facts.keys())

    def results(self, value, props):
        value = value if isinstance(value, Mapping) else {}
        shown = list(props.get("results_facts") or ())
        notes = {k: n for k, n in (value.get("notes") or {}).items() if k in shown}
        notes.update({k: "Measured once on the sealed test." for k in shown if k not in notes})
        caption = value.get("breakdown_caption")
        if not isinstance(caption, str):
            caption = "Each unit on the sealed test." if self._groups.get("breakdown") else None
        return {"headline": value["headline"] if isinstance(value.get("headline"), str) else
                "What the sealed test said about the network.", "notes": notes,
                "breakdown_caption": caption, "limits": self._limits(value.get("limits"), props)}

    def _limits(self, value, props) -> list[dict]:
        known = {lim["id"]: lim for lim in props.get("limits") or ()}
        out, said = [], set()
        for lim in value if isinstance(value, list) else ():
            if isinstance(lim, Mapping) and lim.get("id") in known and lim["id"] not in said:
                out.append(dict(lim))
                said.add(lim["id"])
        fresh = not isinstance(value, list)
        for lid, lim in known.items():
            if lid not in said and (fresh or lim.get("mandatory")):
                facts = [k for k in lim.get("facts") or () if k in self._facts.keys()]
                text = "Stated as the chain recorded it" + (f"; it rests on {{{facts[0]}}}."
                                                           if facts else ".")
                title = _LIMIT_TITLES.get(lid, "A warning of the chain")
                out.append({"id": lid, "title": title, "text": text})
        return out

    def input_groups(self, value, props):
        proposed = props.get("input_groups") or []
        groups = [dict(g) for g in value or () if isinstance(g, Mapping)] \
            if isinstance(value, list) else []
        kept = [g for g in groups if g.get("id") != UNPLACED]
        if not kept:
            return _distinct([{"id": g["suggested_id"], "label": _label(g["suggested_id"]),
                               "text": "Inputs the network reads together.",
                               "members": list(g["members"])} for g in proposed])
        peers = {m: set(g["members"]) for g in proposed for m in g["members"]}
        unplaced = [m for g in groups if g.get("id") == UNPLACED for m in g.get("members") or ()]
        for name in unplaced:
            target = max(kept, key=lambda g: (len(peers.get(name, set()) & set(g["members"])),
                                              -kept.index(g)))
            target["members"] = [*target["members"], name]
        if unplaced:
            self._markers.append(f"OK: autoauthor placed {len(unplaced)} unplaced input(s)")
        return _distinct(kept)

    def code_notes(self, value, props):
        value = value if isinstance(value, Mapping) else {}
        return {s["step"]: value[s["step"]] if isinstance(value.get(s["step"]), str) else
                "Where this step lives in the code." for s in props.get("code_steps") or ()}

    def monitor_note(self, value, props):
        return value if isinstance(value, str) else None



_ORDINALS = ("second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
             "eleventh", "twelfth")


def _distinct(items: list, key: str = "label", limit: str = "group_label") -> list:
    """The items with one `key` each (V15: two input families sharing a label, or two choices
    sharing a title, are two a reader cannot tell apart): a repeat takes an ordinal word, never a
    digit (the resolver's rule), within the slot's length."""
    seen: dict = {}
    for item in items:
        base = str(item.get(key) or "inputs")
        n = seen.get(base, 0)
        seen[base] = n + 1
        if n:
            word = _ORDINALS[n - 1] if n <= len(_ORDINALS) else "further"
            item[key] = f"{base[:max(1, LENGTHS[limit] - len(word) - 2)]}, {word}"
    return items


def _label(suggested_id: str) -> str:
    words = re.findall(r"[a-z]+", suggested_id)
    return " ".join(words)[:LENGTHS["group_label"]] or "inputs"


# =========================================================================== the command line

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--plan", required=True, help="the draft (or carried) story plan")
    ap.add_argument("--facts", required=True, help="the fact set the plan was drafted from")
    ap.add_argument("--out", required=True, help="where the authored plan goes")
    ap.add_argument("--report", help="the report's .aux, PDF or directory (proposed roles)")
    ap.add_argument("--json", action="store_true", help="print a machine result")
    args = ap.parse_args(argv)
    try:
        lines, code = _run(args)
    except (ShowError, OSError, KeyError, TypeError) as exc:
        lines, code = [f"ERROR: autoauthor: {exc}"], 2
    if args.json:
        print(json.dumps({"status": "ok" if code == 0 else "refused", "out": args.out,
                          "markers": lines}, ensure_ascii=False))
    else:
        print("\n".join(lines))
    return code


def _run(args) -> tuple[list[str], int]:
    facts_path = Path(args.facts)
    doc, error = read_json(facts_path, "the fact set")
    plan, plan_error = read_json(args.plan, "the plan")
    if error or plan_error:
        raise ShowError(error or plan_error)
    facts = FactSet.from_json(doc)
    author = AutoAuthor(facts, report_labels_for(args.report, facts, facts_path))
    authored = author.author(plan)
    problems = author.problems(authored)
    if problems:
        return [*author.markers, *map(str, problems),
                f"ERROR: autoauthor: the authored plan still fails {len(problems)} rule(s)"], 2
    write_json(args.out, authored)
    return [*author.markers, f"OK: authored {args.out}"], 0


if __name__ == "__main__":
    sys.exit(main())
