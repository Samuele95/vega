"""
story_report — `ReportBinding`: where the story's code notes point, and the story's `report`
block (contract §4.1 `code_notes`, §6; R-E7). A note links into the report through a **role**,
never through a typed number: the role names the object (the network's forward equation, the
serving slice), a label is that object, and the report's `.aux` gives its number and page.

The binding a walk step takes is the first role of `catalog.walk_roles[<step>]` that resolves
(`propose_labels.code_steps`, the draft's own rule), against the bindings of the report's mode:

- **emitted** (`/document-project` wrote the label-role map): the fact set's `report.roles`, each
  with its `report_<role>_number` and `_page` facts;
- **proposed** (a compiled report without a role map): the roles the model confirmed in
  `authored.report_roles`, **bound against the `.aux`** — each confirmed label read from the
  report's `LabelSource` (V14 checked it exists with the role's kind), its number and page made
  values `report_<role>_number` / `report_<role>_page` whose refs are
  `report:<role>.number|page`, as the extractor keys an emitted binding;
- **brief links** (no report, fork S1's unattended answer): a step whose role's stage has a
  stage brief the fact set found links to the brief's copy, `showcase/briefs/<stage>-brief.md`.

A bound note is `{note, link: {href: "showcase/report/report.pdf#page=<page>", kind: "report"},
ref: {kind, number, page, title, chapter?}}` (`number` and `page` values keys, so the page composes
the label from `report.kind.<kind>` and no English label is written here; `chapter` — `{number,
title}` of the role's stage chapter, when that role is bound and is not the note's own — lets the
page say where the object sits, "§n Title · Figure k"); a note whose text names an object of
another kind than the one bound ("the dashed units of the figure" bound to a table) is a `WARN:`; a brief note is
`{note, link: {href, kind: "file"}, ref: null}`; a step with no role, or a report whose PDF the
facts did not find, has no link (`WARN:` for the latter). The `report` block is the report's
status and digests as the fact set recorded them, or the brief copies' paths.
"""
from __future__ import annotations

from typing import Mapping

from propose_labels import code_steps
from show_ports import LabelSource
from story_sections import StoryValues
from vega_show_core import CATALOG, FactMissing, FactSet, fact_key, thaw

__all__ = ["ReportBinding", "report_mode"]

_PDF_HREF = "showcase/report/report.pdf"
_KINDS = ("figure", "table", "equation", "section")
_PDF_FILE = "report/report.pdf"


def report_mode(report: Mapping | None) -> str:
    """`emitted`, `proposed` or `absent`: the report's roles as the fact set found them."""
    report = report or {}
    if report.get("status") != "linked":
        return "absent"
    return "emitted" if report.get("roles_source") else "proposed"


class ReportBinding:
    """The code notes' links and the story's report block, for one fact set and its report."""

    def __init__(self, facts: FactSet, labels: LabelSource | None, values: StoryValues,
                 catalog: Mapping = CATALOG) -> None:
        self._facts, self._labels, self._values, self._cat = facts, labels, values, catalog
        groups = thaw(facts.groups)
        self._report, self._briefs = groups.get("report") or {}, groups.get("briefs") or {}
        self._mode = report_mode(self._report)
        self._markers: list[str] = []

    @property
    def markers(self) -> tuple[str, ...]:
        return tuple(self._markers)

    def block(self) -> dict:
        """The story's `report`: status, the PDF's copy, the digests, the roles' origin, or the
        brief copies in brief-link mode."""
        if self._mode == "absent":
            briefs = {s: f"briefs/{s}-brief.md" for s in self._briefs}
            return {"status": "brief-links", "file": None, "pdf_digest": None, "aux_digest": None,
                    "roles_digest": None, "roles_source": None, "briefs": briefs or None}
        pdf, aux = self._report.get("pdf") or {}, self._report.get("aux") or {}
        roles = self._report.get("roles_source") or {}
        return {"status": "linked", "file": _PDF_FILE if pdf else None,
                "pdf_digest": pdf.get("digest"), "aux_digest": aux.get("digest"),
                "roles_digest": roles.get("digest"),
                "roles_source": roles.get("origin", "document-project") if roles else "proposed",
                "briefs": None}

    def code_notes(self, notes: Mapping, authored_roles: Mapping | None,
                   proposals: Mapping) -> dict:
        """`{walk id: {note, link, ref}}` for every authored note, in the draft's step order."""
        bindings = self._bindings(authored_roles)
        steps = [str(s.get("step")) for s in proposals.get("code_steps") or ()]
        status = {"status": self._mode, "bindings": bindings}
        out = {}
        for step in code_steps(steps, status, self._briefs, self._cat):
            if step["step"] in notes:
                link, ref = self._target(step, bindings)
                out[step["step"]] = {"note": self._values.text(notes[step["step"]]), "link": link,
                                     "ref": ref}
                named = [k for k in _KINDS if f"the {k}" in str(notes[step["step"]]).lower()]
                if ref and named and ref["kind"] not in named:
                    self._markers.append(f"WARN: the code note on {step['step']!r} names the {named[0]}, "
                                         f"and its role binds a {ref['kind']}: the note was written for "
                                         "another object — rewrite it for the one it links")
        if self._mode != "absent" and not self._report.get("pdf") and out:
            self._markers.append("WARN: the report's PDF was not found when the facts were "
                                 "extracted; the code notes name their report objects without a "
                                 "link")
        return out

    def _bindings(self, authored_roles: Mapping | None) -> dict:
        """{role: {kind, number, page, title}}, `number`/`page` the value each would take."""
        if self._mode == "emitted":
            return {role: b for role, b in (self._report.get("roles") or {}).items()
                    if role in self._cat["report_roles"]}
        if self._mode == "absent" or not authored_roles or self._labels is None:
            return {}
        aux, out = self._labels.labels(), {}
        for role, label in authored_roles.items():
            found = aux.get(label)
            if role in self._cat["report_roles"] and found is not None:
                out[role] = {"label": label, "kind": str(found.kind), "number": found.number,
                             "page": found.page, "title": found.title, "bound": True}
        return out

    def _target(self, step: Mapping, bindings: Mapping) -> tuple[dict | None, dict | None]:
        role = step.get("role")
        if step.get("status") == "brief":
            stage = self._cat["report_roles"][role]["stage"]
            return {"href": f"showcase/briefs/{stage}-brief.md", "kind": "file"}, None
        if step.get("status") != "bound" or role not in bindings:
            return None, None
        number, page, binding = self._number_page(role, bindings[role])
        ref = {"kind": binding["kind"], "number": number, "page": page,
               "title": str(binding.get("title") or "")} if number and page else None
        chapter = f"chapter.{self._cat['report_roles'][role]['stage']}"
        if ref and chapter != role and chapter in bindings:
            c_number, _, c_binding = self._number_page(chapter, bindings[chapter], ("number",))
            if c_number:
                ref["chapter"] = {"number": c_number, "title": str(c_binding.get("title") or "")}
        if not self._report.get("pdf"):
            return None, ref
        at = self._page_value(page)
        href = f"{_PDF_HREF}#page={at}" if isinstance(at, int) and not isinstance(at, bool) \
            else _PDF_HREF
        return {"href": href, "kind": "report"}, ref

    def _number_page(self, role: str, binding: Mapping, wanted=("number", "page")
                     ) -> tuple[str | None, str | None, Mapping]:
        """The values keys of a binding's number and page (used, or made for a proposed one);
        a field not `wanted` (a chapter's page) is neither used nor made: None."""
        if not binding.get("bound"):
            keys = [binding.get(f) if f in wanted and binding.get(f) in self._facts.keys() else None
                    for f in ("number", "page")]
            return (*(self._values.use(k) if k else None for k in keys), binding)
        made = []
        for field in ("number", "page"):
            value = binding.get(field) if field in wanted else None
            key = fact_key(("report", role, field))
            made.append(self._values.make(key, value, "verbatim", f"report:{role}.{field}")
                        if value not in (None, "") else None)
        return (*made, binding)

    def _page_value(self, key: str | None):
        if key is None:
            return None
        try:
            return self._values.value(key)
        except (FactMissing, KeyError):
            return None
