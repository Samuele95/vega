"""
showcase_manifest — `SHOWCASE.json` (`vega/showcase@1`, contract §5.3) of a staged showcase, the
build's options and fork S1, and the file states gates BD7 and BD8 compare.

    BuildOptions(locale, (unattended, brief_links), template).settle_s1(report_status)
    ShowcaseDocument(inputs, stage, target).assemble(pins, requires, facts) -> dict
    tree_state(root, skip) -> {relative path: (size, mtime_ns)};  changed(before, after) -> [paths]

**Pinned** is every file of the staged showcase outside `.work/`, `.smoke/` and `docs/`, and never
`SHOWCASE.json` itself. **Roles** follow the file's place: `page/**` and `page.json` are `page`,
`story.json` is `story`, `report/**` and `briefs/**` are `report`, `data/**` is `data`, the rest
(`plan/**`) `record`; **served** are the files of the served roles (`page`, `story`, `report`,
`data`). **Supersedes** is the showcase in place (its `SHOWCASE.json` digest and creation time),
or null. **Markers** are the build's `OK:` and `WARN:` lines. The document is validated against
`showcase-1.schema.json` before it is written (gate BD6).

**Fork S1** (the report): when the fact set found no report, the build goes on only with the
fork's answer — `--unattended` (its unattended answer, brief links) or `--brief-links` (the
user's) — and otherwise stops with `FORK:` before anything is written (D6 VS16, VS17).
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import page_template
from fact_sink import Refusal
from showcase_check import showcase_schema_problems
from showcase_inputs import BuildInputs, relative
from vega_show_core import CATALOG, ShowError, digest_file, now, read_json, write_json

__all__ = ["ATTRIBUTION", "BuildOptions", "Fork", "ShowcaseDocument", "tree_state", "changed",
           "template_identity", "wrap_prose"]

SPEC = CATALOG["showcase"]
ATTRIBUTION = "data/attribution.json"
REQUIRES = {"network_view": "vega/network-view@2", "explanation": "vega/explanation@2",
            "examples": "vega/examples@2"}
_ROLE_OF = (("page/", "page"), ("page.json", "page"), ("story.json", "story"),
            ("report/", "report"), ("briefs/", "report"), ("data/", "data"))


class Fork(ShowError):
    """A fork the build stops at (`FORK:`, exit 2, nothing written)."""


class BuildOptions:
    """How to build: the locale, fork S1's answer (`unattended`: its unattended answer;
    `brief_links`: the user's), the page template."""

    def __init__(self, locale: str = "en", s1: tuple = (False, False),
                 template: Path | None = None) -> None:
        if locale not in CATALOG["build_locales"]:
            raise Refusal("BD7", "--locale", f"{locale!r} is not a build locale "
                          f"({', '.join(CATALOG['build_locales'])})")
        self.locale, (self.unattended, self.brief_links) = locale, s1
        self.template = Path(template) if template else page_template.TEMPLATE_DIR

    def settle_s1(self, report_status: str | None) -> str | None:
        """Fork S1 when the facts found no report: the answer's marker, or `Fork`."""
        if report_status != "brief-links":
            return None
        if self.unattended:
            return "OK: fork S1 (report) answered unattended: brief links"
        if self.brief_links:
            return "OK: fork S1 (report) answered: brief links (--brief-links)"
        raise Fork("S1 report — no compiled report was found: continue with brief links "
                   "(--brief-links; --unattended applies them), or stop and run "
                   "/document-project vega <lane> first; nothing was written")


def _role(rel: str) -> str:
    return next((role for head, role in _ROLE_OF if rel == head or
                 (head.endswith("/") and rel.startswith(head))), "record")


class ShowcaseDocument:
    """`SHOWCASE.json` of the showcase staged in `stage`, to be swapped into `target`."""

    def __init__(self, inputs: BuildInputs, stage: Path, target: Path) -> None:
        self._in, self._stage, self._target = inputs, Path(stage), Path(target)

    def assemble(self, pins: Mapping, requires: list, facts: Mapping) -> dict:
        """The document, validated and written. `facts`: `placeholders` (the story's count),
        `report` (`SHOWCASE.json.report`) and `markers` (the build's lines so far)."""
        files = self.pinned()
        roles = {r: [p for p in files if _role(p) == r] for r in SPEC["roles"]}
        served = sorted(p for r in SPEC["served_roles"] for p in roles[r])
        markers = [*facts["markers"], f"OK: showcase built — {len(files)} file(s) pinned, "
                                      f"{len(served)} served"]
        doc = {"schema": "vega/showcase@1", "created_at": now(), "produced_by": "/vega-show",
               "bundle": self._bundle(), "model": self._model(), "entry": SPEC["entry"],
               "template": dict(pins), "page": {"file": "page.json"},
               "story": self._story(facts["placeholders"]),
               "attribution": {"file": ATTRIBUTION} if ATTRIBUTION in files else None,
               "plan": {"file": "plan/story-plan.json", "facts": "plan/facts.json"},
               "report": facts["report"], "files": files, "roles": roles, "served": served,
               "requires": {"endpoints": list(requires), **REQUIRES},
               "supersedes": self._previous(),
               "markers": [m for m in markers if m.split(":")[0] in ("OK", "WARN")]}
        errors = showcase_schema_problems(doc)
        if errors:
            raise Refusal("BD6", SPEC["manifest"], "; ".join(errors[:5]))
        write_json(self._stage / SPEC["manifest"], doc)
        return doc

    def pinned(self) -> dict:
        """{relative path: digest} of every file the showcase pins."""
        out = {}
        for path in sorted(self._stage.rglob("*")):
            rel = path.relative_to(self._stage).as_posix()
            if path.is_file() and rel != SPEC["manifest"] and \
                    not rel.startswith(tuple(SPEC["unpinned_dirs"])):
                out[rel] = digest_file(path)
        return out

    def _bundle(self) -> dict:
        manifest = self._in.manifest
        return {"manifest": "MANIFEST.json", "manifest_digest": self._in.store.manifest_digest(),
                "slug": str(manifest.get("slug")), "run": str(manifest.get("run")),
                "model_version": str(manifest.get("model_version"))}

    def _model(self) -> dict:
        return {"path": relative(self._in.model, self._in.root),
                "turtle_digest": self._in.turtle_digest}

    def _story(self, placeholders: int) -> dict:
        story = self._in.json("story")
        return {"file": "story.json", "schema": story["schema"], "values_n": len(story["values"]),
                "placeholders_n": int(placeholders),
                "sections": [s["id"] for s in story["sections"]]}

    def _previous(self) -> dict | None:
        path = self._target / SPEC["manifest"]
        doc, _ = read_json(path, SPEC["manifest"])
        if not isinstance(doc, dict) or not isinstance(doc.get("created_at"), str):
            return None
        return {"digest": digest_file(path), "created_at": doc["created_at"]}


def tree_state(root: Path, skip: Path | None = None) -> dict:
    """{relative path: (size, mtime_ns)} of every file under `root`, outside `skip`."""
    out, root = {}, Path(root)
    if not root.is_dir():
        return out
    for path in sorted(root.rglob("*")):
        if path.is_file() and (skip is None or Path(skip) not in path.parents):
            info = path.lstat()
            out[path.relative_to(root).as_posix()] = (info.st_size, info.st_mtime_ns)
    return out


def changed(before: Mapping, after: Mapping) -> list[str]:
    """The paths whose state differs (added, removed or rewritten)."""
    return sorted(r for r in set(before) | set(after) if before.get(r) != after.get(r))


# A pseudo-locale build's story: every authored string wrapped ⟪…⟫ (D6 §10.5), so the page smoke
# tells the show's prose from chrome the strings table should hold; ids, keys, enums, patterns,
# value tables and references are not prose and stay as they are.
PROSE = ("\u27ea", "\u27eb")
NOT_PROSE = frozenset({"schema", "locale", "task", "lexicon", "class_roles", "polarity", "values", "formats",
                       "refs", "id", "value", "names", "weak", "direction", "metric", "focus", "members",
                       "report", "bundle", "selected", "reference", "link", "ref", "kind", "format",
                       "verdict", "name", "strongest_baseline"})


def wrap_prose(node, key=None):
    """The story (or one of its nodes) with its prose wrapped."""
    if key in NOT_PROSE:
        return node
    if isinstance(node, str):
        return f"{PROSE[0]}{node}{PROSE[1]}" if node else node
    if isinstance(node, list):
        return [wrap_prose(v) for v in node]
    if isinstance(node, dict):
        return {k: wrap_prose(v, k) for k, v in node.items()}
    return node


def template_identity(template: Path) -> Mapping | None:
    """`{version, manifest_digest}` of the template a build copies (its `TEMPLATE.json`), which
    page.json names: the installed one, or the one --template gives. None without the file."""
    path = Path(template) / CATALOG["template"]["manifest"]
    if not path.is_file():
        return None
    doc, _ = read_json(path, "the page template's TEMPLATE.json")
    return {"version": (doc or {}).get("version"), "manifest_digest": digest_file(path)}
