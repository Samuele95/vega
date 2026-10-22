#!/usr/bin/env python3
"""
draft_plan.py — the story plan's proposals (`vega/story-plan@2`, contract §3), drafted from the
fact set so the model authors words, never numbers or choices the records settle.

    python3 draft_plan.py --facts plan/facts.json [--page-manifest assets/page/TEMPLATE.json]
                          [--memory .claude/domain-forge-memory.md] [--dataset NAME]
                          [--carry OLD-PLAN] [--bundle B] [--unattended]
                          --out .work/story-plan.draft.json [--markers PATH] [--json]

`PlanDrafter` writes `proposals` and every `authored` slot as null, `status: draft`, and pins
the fact set (`facts.digest`, the sha256 of the file read) and the page template
(`template.version` and `manifest_digest`, the digest of `TEMPLATE.json`, or of the manifest a
fresh `page_template.py --write` gives while none is installed) — both checked by `V1`. One
owner per proposer (R-E9): the lexicon (`propose_lexicon`), the motif (`propose_motif`), the
class roles (`propose_roles`), the input groups (`propose_families`), the report roles and code
steps (`propose_labels`), the recorded stances (`memory_stances`); this module adds the
mechanical rest (contract §3.7):

- **choices** — one per decision of the fact set except the `choice_skip_types` blocks, with
  its settings (fact keys), rationale ref and the runs that tested it; mandatory for the first
  decision of each vega-forge decision family (`family` else the first block; `loss` else
  `metrics`; `optimizer`, …; the first regularizer else `early_stopping`) and for every decision
  a delta of the selected run touched;
- **runs** — every `baseline` and `ablation` run with status `done`, the best trial of each HPO
  study and the selected run, ordered best first in the metric's direction, each with its
  mechanical label (the fact set's `delta_label` label; empty for a run that changed nothing);
- **results facts** — `test_rows` (the sealed test part's size) last, and before it
  `eval_network_<m>` for each metric of the task's `results_facts` that the
  evaluation computed as a point estimate, the primary excepted (it is `values.primary`);
- **limits**, **context steps** (by sample shape) and **prose sources**, as the facts give them.

The code steps are the page's walk ids (`page_config.build_walk`'s grammar): the head, one id per
distinct block type in forward order — composites' inner blocks included, read from the bundle's
pinned `netspec.json` (the fact set's header names the bundle; `--bundle` overrides) when that
bundle is the one the facts describe, else from the fact set's top-level blocks with a `WARN:` —
then the tail. For a report without a role map the `.aux` the facts pinned is read (its digest
must still be the recorded one) and its labels proposed per role.

`--carry` fills authored slots from an earlier plan (`carry.PlanCarrier`: story-plan@1 or @2;
each slot it cannot carry is a `WARN:`). `--unattended` settles forks S2 and S3 with their
unattended answers in `authored.class_roles` and `authored.motif` (a stance's, else the
proposal: no alert without a `user:`, `show:` or `task:anomaly` basis, the modality's base; a
carried slot is kept, its unsupported alerts made plain) and prints the catalog's markers;
every other slot stays for the model. The forks left open are printed (and are the `--json`
result's `forks`): S3 while no motif is authored, S2 while no authored role rests on a `user:`
turn or a `show:` stance, S1 when there is no report; none after `--unattended`.

The draft is validated against `story-plan-2.schema.json` before it is written. Markers are
printed (`--json`: one result) and appended to `--markers` (by default the `.work/markers.json`
of the showcase directory `--out` lies in). Exit 0 written; 2 refused: the facts unreadable or
not `vega/show-facts@1`, the template missing, the report's `.aux` changed since the facts were
extracted (gate PF6: re-run `extract_facts.py`), the plan to carry unreadable or not a story
plan, or a draft its schema rejects.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import draft_io  # noqa: E402  (the command line's IO edge)
from carry import PlanCarrier  # noqa: E402
from fact_sink import Refusal  # noqa: E402
from memory_stances import MemoryStances, datasets_of  # noqa: E402
from propose_families import FamilyProposer  # noqa: E402
from propose_labels import code_steps, report_proposal  # noqa: E402
from propose_lexicon import LexiconProposer  # noqa: E402
from propose_motif import MotifProposer  # noqa: E402
from propose_roles import RoleProposer  # noqa: E402
from vega_show_core import (CATALOG, FactMissing, FactSet, ShowError, digest_file,  # noqa: E402
                            thaw, write_json)

__all__ = ["PlanDrafter", "DraftInputs", "AUTHORED_SLOTS", "main"]

SCHEMA = HERE.parent / CATALOG["schemas"]["vega/story-plan@2"]["file"]
_SCHEMA_DOC = json.loads(SCHEMA.read_text(encoding="utf-8"))
AUTHORED_SLOTS = tuple(_SCHEMA_DOC["properties"]["authored"]["required"])
FACTS_FILE = _SCHEMA_DOC["properties"]["facts"]["properties"]["file"]["const"]
_RUN_KINDS = ("baseline", "ablation")
_POINT = ("metric", "ci_metric")


@dataclass(frozen=True)
class DraftInputs:
    """What the draft pins and reads beside the fact set: the fact set's digest, the template's
    `{version, manifest_digest}`, the report's `.aux` labels (a report without a role map) and
    the network's distinct block types in forward order (None: the fact set's)."""

    facts_digest: str
    template: Mapping
    labels: Mapping | None = None
    block_types: tuple | None = None


class PlanDrafter:
    """The draft story plan of one fact set: proposals, and every authored slot null."""

    def __init__(self, facts: FactSet, inputs: DraftInputs, stances: MemoryStances,
                 catalog: Mapping = CATALOG) -> None:
        self._facts, self._inputs, self._stances, self._cat = facts, inputs, stances, catalog
        self._groups = thaw(facts.groups)
        self._markers: list[str] = []

    @property
    def markers(self) -> tuple[str, ...]:
        return tuple(self._markers)

    def draft(self) -> dict:
        self._markers = list(self._stances.markers)
        return {"schema": "vega/story-plan@2", "status": "draft",
                "facts": {"file": FACTS_FILE, "digest": self._inputs.facts_digest},
                "template": {"version": self._inputs.template["version"],
                             "manifest_digest": self._inputs.template["manifest_digest"]},
                "carried_from": None, "proposals": self._proposals(),
                "authored": {slot: None for slot in AUTHORED_SLOTS}}

    def unattended(self, plan: dict) -> dict:
        """Forks S2 and S3 settled by their unattended answers, in `authored` (a slot a carry
        filled is kept; its unsupported alerts become plain)."""
        props, authored = plan["proposals"], plan["authored"]
        if authored.get("motif") is None:
            motif = MotifProposer(props["modality"], self._stances, self._cat)
            authored["motif"] = motif.answer(props["motif"])
            if props["motif"]["fork"]:
                self._markers.append(self._cat["forks"]["S3"]["marker"].replace(
                    "<base>", props["motif"]["base"]))
        roles = RoleProposer(self._facts.task, {}, self._stances, self._cat)
        authored["class_roles"], markers = roles.unattended(props["class_roles"],
                                                            authored.get("class_roles"))
        self._markers += markers
        return plan

    # ---------------------------------------------------------------- proposals
    def _proposals(self) -> dict:
        task, groups = self._facts.task, self._groups
        lexicon = LexiconProposer(task, groups.get("identity"), self._stances, self._cat)
        motif = MotifProposer(task.modality, self._stances, self._cat)
        roles = RoleProposer(task, self._balance(), self._stances, self._cat)
        report = report_proposal(groups.get("report"), self._inputs.labels, self._cat)
        out = {"modality": task.modality, "view_key": task.view_key, "lexicon": lexicon.propose(),
               "motif": motif.propose(), "class_roles": roles.propose(),
               "input_groups": FamilyProposer(groups.get("inputs") or {},
                                              groups.get("recipe") or (), self._cat).propose(),
               "choices": self._choices(), "runs": self._runs(),
               "results_facts": self._results_facts(), "limits": groups.get("limits") or [],
               "code_steps": code_steps(self._walk(), report, groups.get("briefs") or {},
                                        self._cat),
               "report_roles": report,
               "context_steps": list(self._cat["context_steps"]["by_shape"][task.sample_shape]),
               "prose_sources": groups.get("prose_sources") or []}
        for proposer in (lexicon, motif, roles):
            self._markers += proposer.markers
        return out

    def _balance(self) -> dict:
        """{class: (train-share fact key, its value)} from the `class_balance` group."""
        balance = self._groups.get("class_balance") or {}
        return {str(c): (k, self._value(k)) for c, k in balance.items()}

    def _choices(self) -> list[dict]:
        skip = set(self._cat["choice_skip_types"])
        decisions = [d for d in self._groups.get("decisions") or () if d.get("family") and not
                     (str(d.get("path", "")).startswith("blocks[") and d.get("type") in skip)]
        runs = {r["id"]: r for r in self._groups.get("runs") or ()}
        reference = next((r["primary"] for r in runs.values() if r.get("reference")), None)
        selected = (self._groups.get("selection") or {}).get("run")
        firsts = self._first_of_families(decisions)
        out = []
        for d in decisions:
            tested = [{"run": r, "primary": runs[r].get("primary"), "reference": reference}
                      for r in d.get("tested_by") or () if r in runs]
            out.append({"decision": d["id"], "family": d["family"], "type": d.get("type"),
                        "params": {p: k for p, k in (d.get("params") or {}).items()
                                   if self._kind(k)},
                        "rationale_ref": d.get("rationale_ref"), "tested_by": tested,
                        "mandatory": d["id"] in firsts or selected in (d.get("tested_by") or ())})
        if sum(c["mandatory"] for c in out) > self._cat["choices"]["max"]:
            self._markers.append(f"WARN: {sum(c['mandatory'] for c in out)} mandatory choices, "
                                 f"more than the {self._cat['choices']['max']} a story holds")
        return out

    def _first_of_families(self, decisions: list) -> set:
        """The first decision of each decision family, by the family's member order."""
        firsts = set()
        for members in self._cat["decision_families"].values():
            for member in members:
                found = next((d["id"] for d in decisions if _is_member(d, member)), None)
                if found is not None:
                    firsts.add(found)
                    break
        return firsts

    def _runs(self) -> list[dict]:
        best = {h.get("best") for h in self._groups.get("hpo") or () if h.get("best")}
        kept = [r for r in self._groups.get("runs") or ()
                if (r.get("kind") in _RUN_KINDS and r.get("status") == "done")
                or r.get("id") in best or r.get("selected")]
        unscored = [r["id"] for r in kept if not r.get("primary")]
        if unscored:
            self._markers.append(f"WARN: runs {unscored} have no primary fact; they are not "
                                 "proposed (a run the page cannot draw)")
        kept = [r for r in kept if r.get("primary")]
        selection = self._groups.get("selection") or {}
        direction = selection.get("direction", self._facts.task.direction)
        sign = -1 if direction == "max" else 1
        values = {r["id"]: _number(self._value(r["primary"])) for r in kept}
        kept.sort(key=lambda r: (values[r["id"]] is None, sign * (values[r["id"]] or 0)))
        return [{"id": r["id"], "label": r.get("label") or "", "selected": bool(r.get("selected")),
                 "reference": bool(r.get("reference")), "primary": r["primary"]} for r in kept]

    def _results_facts(self) -> list[str]:
        """The network's secondary metrics on the sealed test part, then the part's size (the
        rows every figure of the view was read on, when the facts hold it)."""
        task = self._facts.task
        keys = [f"eval_network_{m}" for m in self._cat["results_facts"].get(task.kind, ())
                if m != task.primary]
        size = ["test_rows"] if "test_rows" in self._facts.keys() else []
        return [k for k in dict.fromkeys(keys) if self._kind(k) in _POINT] + size

    def _walk(self) -> list[str]:
        types = self._inputs.block_types
        if types is None:
            blocks = (self._groups.get("network") or {}).get("blocks") or ()
            types = tuple(dict.fromkeys(str(b["type"]) for b in blocks if b.get("type")))
        walk = self._cat["walk"]
        return [*walk["head"], *types, *walk["tail"]]

    def _value(self, key):
        try:
            return self._facts.get(key).json_value
        except FactMissing:
            return None

    def _kind(self, key) -> str | None:
        try:
            return str(self._facts.get(key).kind)
        except FactMissing:
            return None


def _number(value) -> float | None:
    """A run's primary as a number to order by; None (ordered last) when it is not one."""
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _is_member(decision: Mapping, member: str) -> bool:
    did = str(decision["id"])
    if member == "blocks":
        return str(decision.get("path", "")).startswith("blocks[")
    if member == "regularization":
        return did.startswith("regularization[")
    return did == member


# =========================================================================== the command line

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--facts", required=True, help="the fact set (plan/facts.json)")
    ap.add_argument("--page-manifest", "--template", dest="template",
                    help="the page template's TEMPLATE.json or its directory (default: installed)")
    ap.add_argument("--memory", help=".claude/domain-forge-memory.md, for the show: stances")
    ap.add_argument("--dataset", action="append", default=[],
                    help="a name the stance lines give this dataset (beside the model stem, slug)")
    ap.add_argument("--carry", help="an earlier story plan (vega/story-plan@1 or @2) to carry")
    ap.add_argument("--bundle", help="the bundle directory (default: the one the facts name)")
    ap.add_argument("--unattended", action="store_true", help="settle forks S2 and S3 unattended")
    ap.add_argument("--out", required=True, help="where the draft plan goes")
    ap.add_argument("--markers", help="the JSON list the markers are appended to")
    ap.add_argument("--json", action="store_true", help="print a machine result")
    args = ap.parse_args(argv)
    try:
        lines, forks, code = *_run(args), 0
    except (ShowError, OSError) as exc:  # a refusal names its gate; any other is the tool's
        lines, forks, code = [f"ERROR: {exc}"], [], 2
    lines = draft_io.append_markers(draft_io.markers_path(args.markers, args.out), lines)
    if args.json:
        print(json.dumps({"status": "ok" if code == 0 else "refused", "out": args.out,
                          "forks": forks, "markers": lines}, ensure_ascii=False))
    else:
        print("\n".join(lines))
    return code


def _run(args) -> tuple[list[str], list[str]]:
    """Draft, carry, settle unattended, validate, write: (the lines to print, the open forks)."""
    facts_path = Path(args.facts)
    facts = draft_io.load_facts(facts_path)
    stances = MemoryStances.read(args.memory, [*datasets_of(thaw(facts.header)), *args.dataset])
    types, note = draft_io.block_types(facts, facts_path, args.bundle)
    inputs = DraftInputs(digest_file(facts_path), draft_io.template_digests(args.template),
                         draft_io.report_labels(facts, facts_path), types)
    drafter = PlanDrafter(facts, inputs, stances)
    plan, carried = drafter.draft(), []
    if args.carry:
        plan["authored"], carried = _carry(facts, plan, args.carry)
        plan["carried_from"] = str(args.carry)
    if args.unattended:
        plan = drafter.unattended(plan)
    draft_io.refuse_invalid(plan, SCHEMA)
    write_json(args.out, plan)
    forks = [] if args.unattended else draft_io.open_forks(plan)
    props = plan["proposals"]
    return [*drafter.markers, *([note] if note else []), *carried,
            f"OK: draft {args.out}: {len(props['choices'])} choices, {len(props['runs'])} runs, "
            f"{len(props['input_groups'])} input groups; forks open: {', '.join(forks) or 'none'}"
            ], forks


def _carry(facts: FactSet, plan: dict, path) -> tuple[dict, list[str]]:
    """The draft's authored slots carried from the plan at `path`, and the carry's WARN lines;
    a plan that does not read as its schema says is refused, never half carried."""
    previous = draft_io.load_plan(path)
    try:
        authored, problems = PlanCarrier(facts, previous).carry(plan)
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise Refusal("draft", path, f"the plan to carry is not a story plan this carry reads "
                                     f"({type(exc).__name__}: {exc})") from exc
    return authored, [str(p) for p in problems]


if __name__ == "__main__":
    sys.exit(main())
