#!/usr/bin/env python3
"""
resolve_story.py — the authored story plan, validated (V1–V15) and resolved into the page's
data document, `vega/story@2` (contract §3.8, §4; R-E3).

    python3 resolve_story.py --plan plan/story-plan.json --facts plan/facts.json
                             [--page-manifest assets/page/TEMPLATE.json] [--report R]
                             --out story.json [--check] [--markers PATH] [--json]

The plan is validated first (`plan_rules.PlanValidator`): every refusal is printed with its rule
id, `ERROR: V<n> <where>: <why>`, and nothing is written — the plan is fixed and resolved again;
`story.json` is never edited by hand. Number words are `WARN:` and recorded. `StoryResolver`
then writes exactly the blocks D4 §15 names: `task`, `lexicon@1` (with the source of each entry
the draft proposed and the model kept), `class_roles@1` (the authored roles and bases, and what
the draft proposed), the polarity (derived, never authored), `values` / `formats` / `refs` (only
what some field references, the reserved `primary` and `ci_level` first), the headline and lead,
the sections (`story_sections`), `input_groups@2` (members in the network's input order),
`code_notes` and `report` (`story_report`: bound through report roles — an emitted map's, or the
confirmed proposed roles bound against the `.aux`), and the bundle it describes. The motif is
not in the story: it lives in `page.json` alone.

The story is validated against `story-2.schema.json` before it is written (a story its schema
refuses is a defect of the resolver, never written) and written once, `ensure_ascii=False`,
`allow_nan=False`. `--check` resolves again and compares with the existing `--out` byte for
byte: exit 0 when equal, 3 when it differs (the plan, the facts or the report moved since), 2
when there is nothing to compare.

`--page-manifest` (or `--template`) is the installed page template's `TEMPLATE.json` or its
directory (default: the installed template); V1 compares the plan's pin with its
`page_template.digests()`. `--report` names the report's `.aux` (or its PDF, or the directory
holding it) when the roles are proposed; by default the `.aux` the fact set pinned is looked
for where `draft_plan.py` looks; its digest must still be the one the facts recorded (gate
PF6: re-run `extract_facts.py`). Markers are printed (`--json`: one result) and appended to
`--markers`, by default the `.work/markers.json` of the showcase directory `--out` lies in
(never on `--check`). Exit 0 resolved (or identical), 2 refused, 3 differs (`--check`).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Mapping

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import draft_io  # noqa: E402  (the draft's IO edge: template pins, the pinned .aux)
from fact_sink import Refusal  # noqa: E402
from plan_rules import PlanPins, PlanValidator, covered_classes  # noqa: E402
from plan_texts import STORY_SCHEMA  # noqa: E402
from report_labels import AuxLabelSource, ReportUnreadable  # noqa: E402
from show_ports import LabelSource  # noqa: E402
from story_report import ReportBinding, report_mode  # noqa: E402
from story_sections import SectionWriter, StoryValues  # noqa: E402
from vega_show_core import (CATALOG, FactSet, ShowError, digest_bytes, digest_file,  # noqa: E402
                            fact_key, read_json, thaw, write_json)

__all__ = ["StoryResolver", "story_bytes", "report_labels_for", "main"]

_RULE = "resolve"
_EVIDENCE = re.compile(STORY_SCHEMA["$defs"]["class_roles"]["properties"]["proposed"]
                       ["properties"]["evidence"]["items"]["pattern"])
_ENTRY = ("one", "many", "article")


def story_bytes(story: Mapping) -> bytes:
    """The bytes `write_json` gives the story: the unit `--check` compares."""
    return (json.dumps(story, indent=1, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


class StoryResolver:
    """`vega/story@2` from an authored plan (validated), its fact set and the report's labels."""

    def __init__(self, facts: FactSet, labels: LabelSource | None = None,
                 catalog: Mapping = CATALOG) -> None:
        self._facts, self._labels, self._cat = facts, labels, catalog
        self._groups = thaw(facts.groups)
        self._markers: list[str] = []
        self.placeholders_n = 0

    @property
    def markers(self) -> tuple[str, ...]:
        return tuple(self._markers)

    def resolve(self, plan: Mapping) -> dict:
        a, p, values = plan["authored"], plan["proposals"], StoryValues(self._facts)
        sections = SectionWriter(self._facts, values, self._cat)
        report = ReportBinding(self._facts, self._labels, values, self._cat)
        roles = self._class_roles(a.get("class_roles"), p.get("class_roles"))
        story = {"schema": "vega/story@2", "locale": a["locale"], "task": self._task(),
                 "lexicon": self._lexicon(a["lexicon"], p.get("lexicon") or {}),
                 "class_roles": roles, "polarity": self._polarity(roles),
                 "values": None, "formats": None, "refs": None,
                 "headline": values.text(a["headline"]), "lead": values.text(a["lead"]),
                 "sections": sections.sections(a, p),
                 "input_groups": [self._group(g, values) for g in a["input_groups"]],
                 "code_notes": report.code_notes(a.get("code_notes") or {},
                                                 a.get("report_roles"), p),
                 "report": report.block(), "bundle": self._bundle()}
        names = {str(k): str(v) for k, v in (a.get("class_names") or {}).items()}
        if names:                        # the words each class is shown by (V15 held the keys)
            story["class_names"] = names
        primary = fact_key(("eval", "network", self._facts.task.primary))
        story["values"], story["formats"], story["refs"] = values.tables(primary)
        self._markers = [*sections.markers, *report.markers, *values.markers]
        self.placeholders_n = values.placeholders_n
        return story

    def _task(self) -> dict:
        """The task as the page's lane reads it: `base` is a `sequence-*` task's base only
        (the fact set also names a forecast's), as `page_derive.lane_record` gives it."""
        t = self._facts.task
        base = t.base if t.kind.startswith("sequence-") else None
        return {"kind": t.kind, "base": base, "goal": t.goal, "sample_shape": t.sample_shape,
                "graph_level": t.graph_level, "family": t.family, "view_key": t.view_key,
                "classes": None if t.classes is None else list(t.classes),
                "outcomes": None if t.outcomes is None else list(t.outcomes),
                "primary": t.primary, "direction": t.direction}

    def _lexicon(self, authored: Mapping, proposed: Mapping) -> dict:
        """`lexicon@1`: the authored entries in the catalog's key order, and the source of every
        entry the model kept as the draft proposed it."""
        out = {"schema": self._cat["lexicon"]["schema"]}
        out.update({k: {f: authored[k][f] for f in _ENTRY} for k in self._cat["lexicon"]["keys"]
                    if k in authored})
        kept = {k: proposed[k]["source"] for k in out if k in proposed and isinstance(
            proposed[k], Mapping) and all(proposed[k].get(f) == out[k][f] for f in _ENTRY)}
        return {**out, "source_of": kept} if kept else out

    def _class_roles(self, authored: Mapping | None, proposal: Mapping | None) -> dict | None:
        covered = covered_classes(self._facts.task, self._cat)
        if authored is None or not covered:
            return None
        proposal = proposal or {}
        neutral = proposal.get("neutral") or {}
        return {"schema": "vega/class-roles@1",
                "roles": {c: authored["roles"][c] for c in covered},
                "source": {c: authored["source"][c] for c in covered},
                "proposed": {"neutral": neutral.get("class"), "rule": neutral.get("rule") or "none",
                             "alert": [c for c, r in (proposal.get("roles") or {}).items()
                                       if r == "alert"],
                             "evidence": [e for e in proposal.get("evidence") or ()
                                          if isinstance(e, str) and _EVIDENCE.fullmatch(e)]}}

    def _polarity(self, roles: Mapping | None) -> str:
        """The first polarity of the task when any class is alert, else the second."""
        by_task = self._cat["polarity"]["by_task"][self._facts.task.kind]
        alert = bool(roles) and "alert" in roles["roles"].values()
        return by_task[0] if alert else by_task[1]

    def _group(self, group: Mapping, values: StoryValues) -> dict:
        order = {n: i for i, n in enumerate((self._groups.get("inputs") or {}).get("shaped") or ())}
        return {"id": group["id"], "label": values.text(group["label"]),
                "text": values.text(group["text"]),
                "members": sorted(group["members"], key=lambda n: order.get(n, len(order)))}

    def _bundle(self) -> dict:
        bundle = thaw(self._facts.header).get("bundle") or {}
        return {k: bundle.get(k) for k in ("manifest_digest", "slug", "run")}


# =========================================================================== the command line

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--plan", required=True, help="the authored story plan (vega/story-plan@2)")
    ap.add_argument("--facts", required=True, help="the fact set the plan was drafted from")
    ap.add_argument("--page-manifest", "--template", dest="template",
                    help="the page template's TEMPLATE.json or its directory (default: installed)")
    ap.add_argument("--report", help="the report's .aux, PDF or directory (proposed roles)")
    ap.add_argument("--out", required=True, help="where story.json goes (or is, with --check)")
    ap.add_argument("--check", action="store_true", help="resolve again and compare byte for byte")
    ap.add_argument("--markers", help="the JSON list the markers are appended to")
    ap.add_argument("--json", action="store_true", help="print a machine result")
    args = ap.parse_args(argv)
    try:
        lines, code = _run(args)
    except (ShowError, OSError) as exc:  # a refusal names its gate; any other is the tool's
        lines, code = [f"ERROR: {exc}"], 2
    if not args.check:
        lines = _append_markers(_markers_path(args.markers, args.out), lines)
    if args.json:
        status = {0: "ok", 2: "refused", 3: "differs"}[code]
        print(json.dumps({"status": status, "out": args.out, "markers": lines}, ensure_ascii=False))
    else:
        print("\n".join(lines))
    return code


def _run(args) -> tuple[list[str], int]:
    facts_path = Path(args.facts)
    facts, plan = _load_facts(facts_path), _load_plan(Path(args.plan))
    labels = report_labels_for(args.report, facts, facts_path)
    pins = PlanPins(digest_file(facts_path), _template(args.template))
    problems = PlanValidator(facts, pins, labels).validate(plan)
    errors = [str(p) for p in problems if p.level == "ERROR"]
    warnings = [str(p) for p in problems if p.level != "ERROR"]
    if errors:
        return [*errors, *warnings, f"ERROR: {_RULE} {args.plan}: {len(errors)} refusal(s) — fix "
                                    "the plan and resolve again (story.json is never edited)"], 2
    resolver = StoryResolver(facts, labels)
    try:
        story = resolver.resolve(plan)
    except (AttributeError, KeyError, TypeError, ValueError) as exc:  # a hand-edited proposal
        raise Refusal(_RULE, args.plan, f"the plan does not read as vega/story-plan@2 says "
                                        f"({type(exc).__name__}: {exc}) — draft it again") from exc
    _refuse_invalid(story)
    data, out = story_bytes(story), Path(args.out)
    lines = [*warnings, *resolver.markers]
    if args.check:
        return _check(out, data, lines)
    if write_json(out, story) != digest_bytes(data):
        raise ShowError(f"{_RULE} {out}: the story written is not the story resolved")
    sections = ", ".join(s["id"] for s in story["sections"])
    return [*lines, f"OK: story {out}: {len(story['values'])} values, {resolver.placeholders_n} "
                    f"placeholders, sections {sections}"], 0


def _check(out: Path, data: bytes, lines: list[str]) -> tuple[list[str], int]:
    if not out.is_file():
        raise Refusal(_RULE, out, "no story to check — resolve the plan first")
    if out.read_bytes() == data:
        return [*lines, f"OK: {out} is byte for byte the story this plan resolves to"], 0
    return [*lines, f"ERROR: {_RULE} {out}: differs from the story this plan resolves to — "
                    "resolve again (the plan, the facts or the report moved; story.json is never "
                    "edited)"], 3


def _load_facts(path: Path) -> FactSet:
    doc, error = read_json(path, "the fact set")
    if error or not isinstance(doc, dict):
        raise Refusal(_RULE, path, error or "the fact set is not a JSON object")
    try:
        return FactSet.from_json(doc)
    except (ValueError, KeyError, TypeError) as exc:
        raise Refusal(_RULE, path, f"not a vega/show-facts@1 fact set ({exc}) — run "
                                   "extract_facts.py") from exc


def _load_plan(path: Path) -> dict:
    doc, error = read_json(path, "the story plan")
    if error or not isinstance(doc, dict) or doc.get("schema") != "vega/story-plan@2":
        raise Refusal(_RULE, path, error or "not a vega/story-plan@2 (draft it with draft_plan.py)")
    return doc


def _template(arg) -> dict:
    try:
        return draft_io.template_digests(arg)
    except Refusal as exc:  # V1 cannot be checked against a template that cannot be pinned
        raise Refusal("V1", exc.where, exc.message) from exc


class _PinnedLabels(LabelSource):
    """The labels of the `.aux` the fact set pinned (`draft_io.report_labels` checked its
    digest), as a `LabelSource`."""

    def __init__(self, labels: Mapping, digest: str) -> None:
        self._labels, self._digest = labels, digest

    def labels(self) -> Mapping:
        return self._labels

    def digest(self) -> str:
        return self._digest


def report_labels_for(report_arg, facts: FactSet, facts_path: Path) -> LabelSource | None:
    """The report's labels when its roles are proposed (V14 and the binding need them); None
    for an emitted map (the facts carry its numbers) or no report."""
    report = thaw(facts.groups).get("report") or {}
    if report_mode(report) != "proposed":
        return None
    recorded = (report.get("aux") or {}).get("digest")
    if report_arg is None:
        return _PinnedLabels(draft_io.report_labels(facts, facts_path), recorded)
    aux = _aux_path(Path(report_arg), (report.get("aux") or {}).get("source"))
    try:
        source = AuxLabelSource(aux)
    except ReportUnreadable as exc:
        raise Refusal("PF6", aux, str(exc)) from exc
    if source.digest() != recorded:
        raise Refusal("PF6", aux, "the report's .aux changed since the facts were extracted — "
                                  "re-run extract_facts.py")
    return source


def _aux_path(path: Path, recorded) -> Path:
    if path.suffix in (".aux", ".pdf"):
        return path.with_suffix(".aux")
    named = path / Path(str(recorded)).name if recorded else None
    if named is not None and named.is_file():
        return named
    found = sorted(path.glob("*.aux")) if path.is_dir() else []
    return found[0] if found else path / "main.aux"


def _refuse_invalid(story: Mapping) -> None:
    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:
        raise ShowError("jsonschema (draft 2020-12) is needed to validate the story before it is "
                        "written; install jsonschema>=4") from exc
    errors = [f"{'/'.join(map(str, e.path)) or '(root)'}: {e.message}"
              for e in Draft202012Validator(STORY_SCHEMA).iter_errors(story)]
    if errors:
        raise Refusal(_RULE, "story", "the resolved story does not validate vega/story@2 (a defect "
                                      "of the resolver): " + "; ".join(errors[:5]))


def _markers_path(markers, out) -> Path | None:
    """`--markers`, else the `.work/markers.json` of the showcase directory `out` lies in (at its
    root, or under its `plan/` or `.work/`)."""
    if markers:
        return Path(markers)
    out = Path(out).resolve()
    dirs = (CATALOG["showcase"]["dir"], *CATALOG["showcase"]["siblings"].values())
    for parent in (out.parent, out.parent.parent):
        if parent.name in dirs:
            return parent / ".work" / "markers.json"
    return None


def _append_markers(path: Path | None, lines: list[str]) -> list[str]:
    """Append `lines` to the markers list at `path` (a JSON list; anything else is left as it
    is, with a `WARN:`); the lines to print."""
    if path is None:
        return lines
    doc, error = read_json(path, "the markers file")
    if not isinstance(doc, list) and (path.exists() or path.is_symlink()):
        return [*lines, f"WARN: the markers file {path} is not a JSON list "
                        f"({error or type(doc).__name__}); these markers are only printed"]
    try:
        write_json(path, (doc if isinstance(doc, list) else []) +
                   [f"resolve_story.py: {m}" for m in lines])
    except (OSError, ValueError) as exc:
        return [*lines, f"WARN: the markers could not be appended to {path} ({exc})"]
    return lines


if __name__ == "__main__":
    sys.exit(main())
