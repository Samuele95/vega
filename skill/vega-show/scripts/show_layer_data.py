"""
show_layer_data — the `showcase` layer's data (`vega/showcase-layer@1`, contract §8, D5 §9.1),
spliced from a built showcase, and the gates the layer is refused on (LY1–LY6; LY7 and the
render's cap are the CLI's, `apply_show_layer.py`).

    ShowLayer.load(authoring_path) -> ShowLayer      the authoring file and what it names
    layer.problems()               -> [Problem]      LY1–LY5, in order (empty: the layer may go)
    layer.document()               -> dict           the layer data

The authoring file (`<bundle>/showcase/.work/showcase-layer.json`) carries only the judgement:
`{"from_showcase": "<bundle>/showcase", "readings": {"abstract": "…"}, "markers": […]}`.
Everything else is read from the showcase, never from the authoring file: `SHOWCASE.json`
verbatim, the story summary and the bindings (`story.json`, `page.json`), the report binding,
the build check (`.work/checks.json`), the served walk (`.smoke/smoke.json`, written by
`smoke_show.py --serve`), the stage's markers (`.work/markers.json`) and the usage
(`MANIFEST.serve`).

| gate | holds that |
|---|---|
| LY1 | the build check recorded for this showcase passed: `.work/checks.json` status `pass`, written after the showcase was built, and every pinned file still has its digest, every served path is pinned |
| LY2 | the served walk of this showcase passed (`smoke.json` status `pass`, its showcase digest this `SHOWCASE.json`'s); or it did not run (no `smoke.json`, or status `not-run`) and the authored markers carry `WARN: served page not checked`. A walk that ran and failed is refused whatever the markers say |
| LY3 | `readings.abstract` is present, at least `abstract_min` characters, digit-free under the fact set's names, and holds no placeholder |
| LY4 | in brief-link mode the authored markers carry `WARN: no report` |
| LY5 | `SHOWCASE.json` reads and pins the sha256 of the bundle's current `MANIFEST.json` |
| LY6 | the data is within `max_data_kb` (the render's cap is checked by the CLI) |
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from digit_rule import PLACEHOLDER_RE, DigitRule
from show_layer_story import bindings, story_summary
from showcase_check import digest_problems, served_problems
from vega_show_core import CATALOG, Problem, ShowPaths, digest_bytes, read_json

__all__ = ["ShowLayer", "SMOKE_SCHEMA", "NOT_CHECKED", "NO_REPORT", "usage_of"]

LAYER = CATALOG["layer"]
SCHEMA = LAYER["schema"]
SMOKE_SCHEMA = "vega/show-smoke@1"
SMOKE_FILE = "smoke.json"
NOT_CHECKED = "WARN: served page not checked"
NO_REPORT = "WARN: no report"
ABSTRACT_MIN = CATALOG["authored_lengths"]["abstract_min"]
MAX_DATA = LAYER["max_data_kb"] * 1024


def _authored(markers, prefix: str) -> bool:
    return any(isinstance(m, str) and m.strip().startswith(prefix) for m in markers)


def _showcase_dir(raw, data_path: Path) -> Path:
    """`from_showcase` as written: absolute, else from the working directory, else beside the
    authoring file; without one, the showcase whose `.work/` holds the authoring file."""
    if not raw:
        return data_path.resolve().parent.parent
    p = Path(raw)
    if p.is_absolute() or p.is_dir():
        return p.resolve()
    return (data_path.resolve().parent / p).resolve()


def usage_of(manifest: Mapping, bundle: Path) -> dict:
    """How to open the page: the serve command and the container line the bundle's service
    names (`MANIFEST.serve.run`, `.identity`), and its address."""
    serve = manifest.get("serve") or {}
    ident = serve.get("identity") or {}
    port = ident.get("port") or 8000
    parts = [p.strip() for p in str(serve.get("run") or "").split("  |  ") if p.strip()]
    command = parts[0] if parts else (f"{ident['cmd']}-serve --port {port}" if ident.get("cmd") else None)
    return {"serve": command or f"python -m serve.app --port {port}",
            "module": f"python -m serve.app --port {port}",
            "url": f"http://127.0.0.1:{port}/",
            "docker": parts[1] if len(parts) > 1 else None,
            "bundle": bundle.name,
            "note": f"From the bundle directory {bundle.name}/; the showcase is served by the "
                    "service's own slot, which verifies it at start."}


@dataclass
class ShowLayer:
    """One authoring file and the showcase, bundle and records it names."""

    data_path: Path
    authoring: dict
    paths: ShowPaths
    showcase: dict | None = None
    showcase_bytes: bytes = b""
    errors: list[Problem] = field(default_factory=list)

    @classmethod
    def load(cls, data_path) -> "ShowLayer":
        data_path = Path(data_path)
        doc, why = read_json(data_path, "the layer's authoring file")
        if not isinstance(doc, dict):
            raise ValueError(why or f"{data_path} is not a JSON object carrying from_showcase and readings")
        showcase = _showcase_dir(doc.get("from_showcase"), data_path)
        layer = cls(data_path, doc, ShowPaths.of(showcase.parent))
        if layer.paths.showcase.resolve() != showcase:
            layer.errors.append(Problem("LY5", "from_showcase", f"{showcase} is not a bundle's showcase/ directory"))
        layer._read_showcase()
        return layer

    def _read_showcase(self) -> None:
        try:
            self.showcase_bytes = self.paths.manifest.read_bytes()
            self.showcase = json.loads(self.showcase_bytes)
        except (OSError, ValueError) as exc:
            self.errors.append(Problem("LY5", "SHOWCASE.json", f"cannot be read: {exc}"))
            return
        if not isinstance(self.showcase, dict) or self.showcase.get("schema") != "vega/showcase@1":
            self.errors.append(Problem("LY5", "SHOWCASE.json", "is not vega/showcase@1"))
            self.showcase = None

    @property
    def markers(self) -> list:
        m = self.authoring.get("markers")
        return m if isinstance(m, list) else []

    def _json(self, path: Path, what: str):
        doc, _ = read_json(path, what)
        return doc

    # ------------------------------------------------------------------------- the gates
    def problems(self) -> list[Problem]:
        if self.errors or self.showcase is None:
            return list(self.errors)
        out = self._ly1_check() + self._ly2_served() + self._ly3_abstract() + self._ly4_report()
        return out + self._ly5_bundle()

    def _ly1_check(self) -> list[Problem]:
        doc = self._json(self.paths.checks, "checks.json")
        if not isinstance(doc, dict):
            return [Problem("LY1", ".work/checks.json", "no build check recorded for this showcase — "
                            "run build_showcase.py --check <bundle>/showcase")]
        if doc.get("status") != "pass":
            return [Problem("LY1", "checks.status", f"the build check is {doc.get('status')!r}, not pass: "
                            + "; ".join(f"{p.get('rule')} {p.get('message')}" for p in doc.get("problems") or []))]
        if str(doc.get("checked_at") or "") < str(self.showcase.get("created_at") or ""):
            return [Problem("LY1", "checks.checked_at", "the recorded check predates this build — "
                            "run build_showcase.py --check <bundle>/showcase again")]
        found = digest_problems(self.paths.showcase, self.showcase.get("files") or {}, "LY1")
        return found + served_problems(self.showcase, "LY1")

    def smoke(self) -> dict | None:
        doc = self._json(self.paths.smoke / SMOKE_FILE, "smoke.json")
        return doc if isinstance(doc, dict) else None

    def _ly2_served(self) -> list[Problem]:
        smoke = self.smoke()
        if smoke is None or smoke.get("status") == "not-run":
            if _authored(self.markers, NOT_CHECKED):
                return []
            why = "no served walk was recorded (.smoke/smoke.json)" if smoke is None else \
                f"the served walk did not run ({smoke.get('why_not_run')})"
            return [Problem("LY2", "checks.served", f"{why} and the markers do not say "
                            f"'{NOT_CHECKED}' — run smoke_show.py --serve <bundle> --strict")]
        if smoke.get("schema") != SMOKE_SCHEMA:
            return [Problem("LY2", ".smoke/smoke.json", f"is not {SMOKE_SCHEMA}")]
        if smoke.get("showcase_digest") != digest_bytes(self.showcase_bytes):
            return [Problem("LY2", ".smoke/smoke.json", "the served walk recorded is of another build of "
                            "this showcase — run smoke_show.py --serve again")]
        if smoke.get("status") != "pass":
            failed = [c.get("id") for c in smoke.get("checks") or [] if not c.get("ok")]
            return [Problem("LY2", "checks.served", f"the served walk failed ({', '.join(map(str, failed)) or smoke.get('status')}); "
                            "a walk that ran and failed is fixed and re-run, never carried as not checked")]
        return []

    def _ly3_abstract(self) -> list[Problem]:
        text = (self.authoring.get("readings") or {}).get("abstract")
        if not isinstance(text, str) or len(text.strip()) < ABSTRACT_MIN:
            return [Problem("LY3", "readings.abstract", f"missing or shorter than {ABSTRACT_MIN} characters")]
        facts = self._json(self.paths.facts, "facts.json") or {}
        out = [Problem("LY3", "readings.abstract", p.message)
               for p in DigitRule(facts.get("names") or ()).refusals(text, "readings.abstract")]
        if PLACEHOLDER_RE.search(text):
            out.append(Problem("LY3", "readings.abstract", "holds a placeholder: the abstract is prose, "
                               "never filled with values"))
        return out

    def _ly4_report(self) -> list[Problem]:
        if (self.showcase.get("report") or {}).get("status") != "brief-links" or _authored(self.markers, NO_REPORT):
            return []
        return [Problem("LY4", "markers", f"the showcase links the stage briefs and the markers do not "
                        f"carry '{NO_REPORT}'")]

    def _ly5_bundle(self) -> list[Problem]:
        try:
            raw = self.paths.bundle_manifest.read_bytes()
        except OSError as exc:
            return [Problem("LY5", "MANIFEST.json", f"cannot be read: {exc}")]
        if (self.showcase.get("bundle") or {}).get("manifest_digest") != digest_bytes(raw):
            return [Problem("LY5", "bundle.manifest_digest", "SHOWCASE.json does not pin the bundle's "
                            "current MANIFEST.json — re-run /vega-show")]
        return []

    # ------------------------------------------------------------------------- the data
    def document(self) -> dict:
        story = self._json(self.paths.story, "story.json") or {}
        page = self._json(self.paths.page_config, "page.json") or {}
        manifest = self._json(self.paths.bundle_manifest, "MANIFEST.json") or {}
        return {"schema": SCHEMA, "showcase": self.showcase,
                "story": story_summary(story, page), "bindings": bindings(story),
                "report": self._report(), "checks": self._checks(),
                "usage": usage_of(manifest, self.paths.bundle),
                "markers": self._markers(), "readings": dict(self.authoring.get("readings") or {})}

    def _report(self) -> dict:
        rep = self.showcase.get("report") or {}
        roles = rep.get("roles") or {}
        digest = lambda part: (rep.get(part) or {}).get("digest")  # noqa: E731
        return {"status": rep.get("status"), "origin": roles.get("origin"),
                "bound": roles.get("bound") or [], "missing": roles.get("missing") or [],
                "pdf_digest": digest("pdf"), "aux_digest": digest("aux"), "roles_digest": roles.get("digest"),
                "briefs": {k: v.get("file") for k, v in (rep.get("briefs") or {}).items()} or None}

    def _checks(self) -> dict:
        build = self._json(self.paths.checks, "checks.json") or {}
        smoke = self.smoke()
        if smoke is None:
            served = {"status": "not-run", "why_not_run": "smoke_show.py --serve did not run for this build"}
        else:
            keep = ("status", "views", "page_errors", "placeholders_unresolved", "links", "screenshots",
                    "why_not_run", "checked_at", "verified")
            served = {k: smoke.get(k) for k in keep if k in smoke}
        return {"build": {k: build.get(k) for k in ("status", "checked_at", "problems", "warnings")},
                "served": served}

    def _markers(self) -> list[str]:
        out = [m for m in self.markers if isinstance(m, str)]
        recorded = self._json(self.paths.markers, "markers.json")
        for line in recorded if isinstance(recorded, list) else []:
            if isinstance(line, str) and line not in out and line.split(": ", 1)[-1] not in out:
                out.append(line)
        return out

    def size_problems(self, data_text: str) -> list[Problem]:
        size = len(data_text.encode("utf-8"))
        if size > MAX_DATA:
            return [Problem("LY6", "data", f"{size // 1024} KB; the contract caps the layer data at "
                            f"{MAX_DATA // 1024} KB")]
        return []

    def missing_keys(self, doc: Mapping) -> list[str]:
        out = []
        for key in LAYER["required"]:
            node = doc
            for part in key.split("."):
                node = node.get(part) if isinstance(node, Mapping) else None
            if node is None:
                out.append(key)
        return out
