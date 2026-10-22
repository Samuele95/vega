"""
propose_labels — where the page's code notes point: `LabelRoleProposer` (report roles proposed
from a hand-authored report's `.aux`, contract §6.5), `report_proposal()` (the story-plan's
`proposals.report_roles`) and `code_steps()` (the walk steps, each bound to a role, §6.2).

**Report roles.** A role names a kind of object the report carries (the network's forward
equation, the serving slice); the page links to it through its label, never a typed number.

- `emitted`: `/document-project` wrote the label-role map, and the fact set's `report.roles`
  holds its bindings (source `document-project`);
- `proposed`: a compiled report without a map (a hand-authored one). `LabelRoleProposer` ranks
  the `.aux` labels per catalog role: the label's kind must be the role's; the score counts the
  role's `keywords` found among the label name's tokens and among its title's words (one each);
  a chapter role prefers a label whose title carries a keyword, then the shallower section; a
  tie keeps the label the `.aux` defines first; a role no label scores is absent, never guessed.
  The model confirms or edits the bindings in `authored.report_roles` (`V14`);
- `absent`: no report — fork `S1`, whose unattended answer is the brief links.

**Code steps.** Every walk id of the network (`walk.head`, one per distinct block type in forward
order, `walk.tail`) is bound to the first role of `catalog.walk_roles[<id>]` that resolves:
`bound` to a role the report binds (emitted or proposed); in brief-link mode `brief` to a role
whose stage has a brief the fact set found (`groups.briefs`); else `no-role` with no role.
"""
from __future__ import annotations

import re
from typing import Iterable, Mapping

from show_ports import LabelRef
from vega_show_core import CATALOG

__all__ = ["LabelRoleProposer", "report_proposal", "code_steps"]

_WORD = re.compile(r"[a-z]+")
_LATEX_CMD = re.compile(r"\\[A-Za-z]+")
_EMITTED, _PROPOSED, _ABSENT = "emitted", "proposed", "absent"


class LabelRoleProposer:
    """The best `.aux` label per report role, for a report that emitted no role map."""

    def __init__(self, labels: Mapping[str, LabelRef], roles_catalog: Mapping[str, dict]) -> None:
        self._labels = list(labels.values())
        self._roles = dict(roles_catalog)

    def propose(self) -> dict:
        """`{role: {label, source: "proposed", evidence}}` for every role a label scores."""
        out = {}
        for role, spec in self._roles.items():
            best = self._best(role, spec)
            if best is not None:
                out[role] = best
        return out

    def _best(self, role: str, spec: Mapping) -> dict | None:
        keywords = [k.lower() for k in spec.get("keywords") or ()]
        scored = []
        for index, label in enumerate(self._labels):
            if str(label.kind) != spec.get("kind"):
                continue
            in_name = [k for k in keywords if k in _tokens(label.label)]
            in_title = [k for k in keywords if k in _title_words(label.title)]
            if in_name or in_title:
                chapter = bool(in_title) if role.startswith("chapter.") else True
                depth = str(label.number).count(".")
                scored.append(((chapter, len(in_name) + len(in_title), -depth, -index),
                               label, in_name, in_title))
        if not scored:
            return None
        _, label, in_name, in_title = max(scored, key=lambda s: s[0])
        found = [f"name has {', '.join(in_name)}"] * bool(in_name) + \
                [f"title has {', '.join(in_title)}"] * bool(in_title)
        return {"label": label.label, "source": "proposed",
                "evidence": "; ".join([f"{label.kind} {label.anchor or label.number}", *found])}


def report_proposal(report: Mapping | None, labels: Mapping[str, LabelRef] | None,
                    catalog: Mapping = CATALOG) -> dict:
    """`proposals.report_roles`: `{status, bindings, fork}` from the fact set's `report` group
    and, for a report without a role map, its `.aux` labels."""
    report = report or {}
    if report.get("status") != "linked":
        return {"status": _ABSENT, "bindings": {}, "fork": "S1"}
    roles = report.get("roles") or {}
    if report.get("roles_source"):
        bindings = {role: {"label": b.get("label"), "source": "document-project", "evidence": None}
                    for role, b in roles.items() if role in catalog["report_roles"]}
        return {"status": _EMITTED, "bindings": bindings, "fork": None}
    proposed = LabelRoleProposer(labels or {}, catalog["report_roles"]).propose()
    return {"status": _PROPOSED, "bindings": proposed, "fork": None}


def code_steps(walk_ids: Iterable[str], report_roles: Mapping, briefs: Mapping,
               catalog: Mapping = CATALOG) -> list[dict]:
    """`[{step, role, status}]`, one per walk id, each bound to the first role that resolves."""
    bound = set(report_roles.get("bindings") or ())
    brief_mode = report_roles.get("status") == _ABSENT
    stages = {role: spec.get("stage") for role, spec in catalog["report_roles"].items()}
    out = []
    for step in walk_ids:
        roles = catalog["walk_roles"].get(step) or []
        if brief_mode:
            role = next((r for r in roles if stages.get(r) in briefs), None)
            status = "brief" if role else "no-role"
        else:
            role = next((r for r in roles if r in bound), None)
            status = "bound" if role else "no-role"
        out.append({"step": step, "role": role, "status": status})
    return out


def _tokens(name: str) -> set[str]:
    return set(_WORD.findall(str(name).lower()))


def _title_words(title: str) -> set[str]:
    return set(_WORD.findall(_LATEX_CMD.sub(" ", str(title)).lower()))
