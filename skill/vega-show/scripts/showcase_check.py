#!/usr/bin/env python3
"""
showcase_check — is a built showcase still the one to serve? `ShowcaseChecker` (contract §5.5,
gates CK1–CK10; R-D11), and the invariants `build_showcase.py` holds before it swaps (BD3–BD5).

    python3 build_showcase.py --check <bundle>/showcase [--report R] [--json]
    python3 showcase_check.py <bundle>/showcase [--report R] [--json]

The checks run in the gates' order and each finding names its gate:

| gate | status | holds that |
|---|---|---|
| CK1 | broken | `SHOWCASE.json` reads and validates `vega/showcase@1` (nothing else is checked otherwise) |
| CK2 | broken | every `files` digest is the file's (a missing file too) |
| CK3 | broken | every pinned path is relative, normalised and admitted where it lies; nothing is a symlink; no refused extension anywhere; no page extension outside `page/` (and the pseudo-locale build's `.work/qps/page/`); no `.png` outside `.smoke/`; no servable file `files` does not pin outside `.work/`, `.smoke/` and `docs/` |
| CK4 | broken | `served ⊆ files`; the entry (an `.html` under `page/`), the story and a linked report served |
| CK5 | broken | `bundle.manifest_digest` is the sha256 of the bundle's `MANIFEST.json` and `bundle.model_version` its model version — unless CK6 applies |
| CK6 | stale | the digest is a `MANIFEST.reemits[].previous_manifest_digest`: the bundle was re-emitted after this showcase |
| CK7 | stale | the report's PDF, `.aux` and role-map sources still have their recorded digests |
| CK8 | stale | in brief-link mode, every stage brief's source still has its recorded digest |
| CK9 | stale | no report exists now where the showcase links the stage briefs |
| CK10 | WARN | the installed page template is the one the showcase was built with |

A source recorded relative (`docs/main.pdf`) is read from the project that holds the bundle
(`<project>/bundle/<slug>-<run>/`), then from the working directory; with `--report` (a
`docs/` directory or the PDF), the report's files are read there first. A role map the model
confirmed from the `.aux` (`origin: proposed`) has the pinned plan as its source, relative to
the showcase. A report "exists" when a `label-roles.json`, or a PDF with a same-stem `.aux`, is
in `--report`, in the `docs/` of the project or beside the model page, or in `./docs`.

The status is `broken` when any broken finding holds, else `stale`, else `pass`; the exit code is
the catalog's `check_status` (0, 2, 3). The result is written to the showcase's
`.work/checks.json` (`{status, problems: [{rule, where, message}], warnings, checked_at}`) when
the showcase directory exists; `--json` prints it.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Mapping

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from vega_show_core import (CATALOG, CheckStatus, Problem, digest_bytes, digest_file,  # noqa: E402
                            now, read_json, write_json)

__all__ = ["ShowcaseChecker", "files_problems", "tree_problems", "served_problems",
           "digest_problems", "showcase_schema_problems", "bundle_of", "main"]

SPEC, EXT = CATALOG["showcase"], CATALOG["showcase_extensions"]
MANIFEST = SPEC["manifest"]
UNPINNED = tuple(SPEC["unpinned_dirs"])
PATH_RE = re.compile(SPEC["path_pattern"])
MEDIA = frozenset(CATALOG["slot"]["media"])
SCHEMA_FILE = HERE.parent / CATALOG["schemas"]["vega/showcase@1"]["file"]
STALE_REEMIT = CATALOG["slot"]["stale_reason"]


def _suffix(rel: str) -> str:
    return PurePosixPath(rel).suffix.lower()


def extension_problem(rel: str) -> str | None:
    """Why the file at `rel` may not lie there (§5.2), or None."""
    ext = _suffix(rel)
    if ext in EXT["refused"]:
        return f"a {ext} file: the showcase holds no code"
    under = [d for d, exts in EXT["under"].items() if ext in exts]
    if under and not any(rel.startswith(d) for d in under):
        return f"a {ext} file outside {' and '.join(under)}"
    return None


def files_problems(files, rules: tuple = ("CK3", "CK3")) -> list[Problem]:
    """The path invariants of `files` (§5.3): relative, normalised, no `..`, not `SHOWCASE.json`,
    an extension admitted where it lies. `rules`: (the path gate, the extension gate)."""
    if not isinstance(files, Mapping) or not files:
        return [Problem(rules[0], "files", "the showcase pins no file")]
    out = []
    for rel in files:
        parts = str(rel).split("/")
        if not PATH_RE.fullmatch(str(rel)) or any(p in ("", ".", "..") for p in parts):
            out.append(Problem(rules[0], f"files.{rel}", "not a relative, normalised path inside "
                                                        "the showcase"))
        elif rel == MANIFEST:
            out.append(Problem(rules[0], f"files.{rel}", f"{MANIFEST} is never in its own files"))
        elif _suffix(rel) not in EXT["anywhere"] and not any(
                _suffix(rel) in exts and rel.startswith(d) for d, exts in EXT["under"].items()):
            out.append(Problem(rules[1], f"files.{rel}", f"a {_suffix(rel) or 'extensionless'} "
                                                        "file is not one the showcase pins"))
    return out


def tree_problems(root: Path, files, rules: tuple = ("CK3", "CK3")) -> list[Problem]:
    """What lies under `root` (§5.2, R-D5): no symlink, no refused or misplaced extension, and
    no servable file that `files` does not pin outside the unpinned directories."""
    out, pinned = [], set(files or ())
    for top, dirs, names in os.walk(root, followlinks=False):
        for name in sorted(dirs + names):
            path = Path(top) / name
            rel = path.relative_to(root).as_posix()
            if path.is_symlink():
                out.append(Problem(rules[0], rel, "a symlink: a showcase holds none"))
                continue
            if name in dirs:
                continue
            why = extension_problem(rel)
            if why:
                out.append(Problem(rules[1], rel, why))
            elif (rel != MANIFEST and rel not in pinned and not rel.startswith(UNPINNED)
                  and _suffix(rel) in MEDIA):
                out.append(Problem(rules[1], rel, f"a servable file {MANIFEST} does not pin, "
                                                  f"outside {' '.join(UNPINNED)}"))
    return out


def served_problems(doc: Mapping, rule: str = "CK4") -> list[Problem]:
    """`served ⊆ files`; the entry an `.html` page under `page/`; the story and a linked report
    served."""
    files, served = doc.get("files") or {}, doc.get("served")
    if not isinstance(served, list):
        return [Problem(rule, "served", "not a list of paths")]
    out = [Problem(rule, f"served.{s}", "served but not pinned in files")
           for s in served if s not in files]
    entry = doc.get("entry")
    if not (isinstance(entry, str) and entry.startswith("page/") and entry.endswith(".html")):
        out.append(Problem(rule, "entry", f"{entry!r} is not an .html page under page/"))
    pdf = (doc.get("report") or {}).get("pdf")
    wanted = {"entry": entry, "story.file": (doc.get("story") or {}).get("file")}
    if isinstance(pdf, Mapping):
        wanted["report.pdf.file"] = pdf.get("file")
    out += [Problem(rule, name, f"{rel!r} is not a file the showcase serves")
            for name, rel in wanted.items() if rel not in served]
    return out


def digest_problems(root: Path, files: Mapping, rule: str = "CK2") -> list[Problem]:
    """Every pinned file's digest is the one `files` records."""
    out = []
    for rel, digest in sorted((files or {}).items()):
        path = Path(root) / rel
        if not path.is_file() or path.is_symlink():
            out.append(Problem(rule, rel, "pinned but missing"))
        elif digest_file(path) != digest:
            out.append(Problem(rule, rel, "its bytes are not the ones the showcase pinned "
                                          f"({digest_file(path)} on disk, {digest} pinned)"))
    return out


def showcase_schema_problems(doc) -> list[str]:
    """`SHOWCASE.json` against `showcase-1.schema.json` (one line per error)."""
    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:
        raise RuntimeError("jsonschema (draft 2020-12) is needed to check vega/showcase@1; "
                           "install jsonschema>=4") from exc
    schema = json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))
    return [f"{'/'.join(map(str, e.path)) or '(root)'}: {e.message}"
            for e in Draft202012Validator(schema).iter_errors(doc)]


def bundle_of(showcase: Path) -> Path | None:
    """The bundle a showcase directory belongs to: the nearest directory above it that holds a
    `MANIFEST.json` (a pseudo-locale build lies deeper, under `.work/qps/`)."""
    for parent in Path(showcase).resolve().parents:
        if (parent / "MANIFEST.json").is_file():
            return parent
    return None


def report_exists(dirs) -> bool:
    """A role map, or a PDF with a same-stem `.aux`, in one of `dirs`."""
    for d in dirs:
        if (d / "label-roles.json").is_file():
            return True
        if d.is_dir() and any(p.with_suffix(".aux").is_file() for p in d.glob("*.pdf")):
            return True
    return False


class ShowcaseChecker:
    """One showcase against its bundle, its report or briefs, and the installed template."""

    def __init__(self, showcase: Path, report: Path | None = None, template: Path | None = None,
                 catalog: Mapping = CATALOG) -> None:
        self._dir, self._report = Path(showcase), Path(report) if report else None
        self._template, self._cat = template, catalog
        self._bundle = bundle_of(self._dir)
        self._root = self._bundle.parent.parent if self._bundle is not None else None
        self.warnings: list[Problem] = []

    def check(self) -> tuple[CheckStatus, list[Problem]]:
        """(status, problems): the problems of every gate that holds no longer, in order."""
        doc, problems = self._manifest()
        if doc is None:
            return CheckStatus.BROKEN, problems
        files = doc["files"]
        problems += digest_problems(self._dir, files)
        problems += files_problems(files) + tree_problems(self._dir, files)
        problems += served_problems(doc)
        problems += self._bundle_problems(doc)
        mode = doc["report"]["status"]
        problems += self._report_problems(doc) if mode == "linked" else self._brief_problems(doc)
        self.warnings = self._template_warnings(doc)
        return self._status(problems), problems

    # -- CK1 --------------------------------------------------------------------------------
    def _manifest(self) -> tuple[dict | None, list[Problem]]:
        path = self._dir / MANIFEST
        if not self._dir.is_dir():
            return None, [Problem("CK1", str(self._dir), "no showcase directory")]
        doc, error = read_json(path, MANIFEST)
        if not isinstance(doc, dict) or doc.get("schema") != "vega/showcase@1":
            why = error or f"not vega/showcase@1 (schema {(doc or {}).get('schema')!r})"
            return None, [Problem("CK1", MANIFEST, why)]
        errors = showcase_schema_problems(doc)
        if errors:
            return None, [Problem("CK1", MANIFEST, "does not validate vega/showcase@1: " +
                                  "; ".join(errors[:5]))]
        return doc, []

    # -- CK5, CK6 ---------------------------------------------------------------------------
    def _bundle_problems(self, doc: Mapping) -> list[Problem]:
        if self._bundle is None:
            return [Problem("CK5", "bundle", "no bundle (MANIFEST.json) holds this showcase")]
        raw = (self._bundle / "MANIFEST.json").read_bytes()
        pinned, manifest = doc["bundle"]["manifest_digest"], json.loads(raw)
        if pinned != digest_bytes(raw):
            reemits = manifest.get("reemits") if isinstance(manifest.get("reemits"), list) else []
            if any(isinstance(r, Mapping) and r.get("previous_manifest_digest") == pinned
                   for r in reemits):
                return [Problem("CK6", "bundle.manifest_digest", STALE_REEMIT)]
            return [Problem("CK5", "bundle.manifest_digest", "not the sha256 of the bundle's "
                            "MANIFEST.json: the bundle was altered in place (re-run /vega-pack)")]
        if doc["bundle"]["model_version"] != manifest.get("model_version"):
            return [Problem("CK5", "bundle.model_version", "not the bundle's model_version")]
        return []

    # -- CK7, CK8, CK9 ----------------------------------------------------------------------
    def _report_problems(self, doc: Mapping) -> list[Problem]:
        report, out = doc["report"], []
        roles = report["roles"]
        for part, entry in (("pdf", report["pdf"]), ("aux", report["aux"]), ("roles", roles)):
            where = "showcase" if part == "roles" and roles.get("origin") == "proposed" else "report"
            out += self._source_problem("CK7", f"report.{part}", entry, where)
        return out

    def _brief_problems(self, doc: Mapping) -> list[Problem]:
        out = []
        for stage, entry in sorted((doc["report"]["briefs"] or {}).items()):
            out += self._source_problem("CK8", f"report.briefs.{stage}", entry, "project")
        if report_exists(self._report_dirs(doc)):
            out.append(Problem("CK9", "report.status", "a compiled report exists now where the "
                               "showcase links the stage briefs — re-run /vega-show"))
        return out

    def _source_problem(self, rule: str, where: str, entry: Mapping, base: str) -> list[Problem]:
        path = self._source(entry["source"], base)
        if path is None:
            return [Problem(rule, where, f"its source {entry['source']} is gone")]
        if digest_file(path) != entry["digest"]:
            return [Problem(rule, where, f"its source {path} changed since the showcase was "
                                         "built — re-run /vega-show")]
        return []

    def _source(self, recorded: str, base: str) -> Path | None:
        """A recorded source on disk: `showcase` sources lie in the showcase; a `report` source
        in `--report` first; any other where it was recorded (absolute), else in the project,
        else in the working directory."""
        rec = Path(recorded)
        if base == "showcase":
            candidates = [self._dir / rec]
        else:
            candidates = []
            if base == "report" and self._report is not None:
                folder = self._report if self._report.is_dir() else self._report.parent
                candidates.append(folder / rec.name)
            candidates += [rec] if rec.is_absolute() else [
                r / rec for r in (self._root, Path.cwd()) if r is not None]
        return next((c for c in candidates if c.is_file()), None)

    def _report_dirs(self, doc: Mapping) -> list[Path]:
        dirs = []
        if self._report is not None:
            dirs.append(self._report if self._report.is_dir() else self._report.parent)
        model = Path(doc["model"]["path"])
        if self._root is not None:
            dirs.append(self._root / "docs")
            dirs.append((model if model.is_absolute() else self._root / model).parent / "docs")
        return dirs + [Path.cwd() / "docs"]

    # -- CK10 --------------------------------------------------------------------------------
    def _template_warnings(self, doc: Mapping) -> list[Problem]:
        import page_template
        try:
            installed = page_template.digests(self._template or page_template.TEMPLATE_DIR)
        except (page_template.TemplateError, OSError) as exc:
            return [Problem("CK10", "template", f"the installed template cannot be pinned ({exc})",
                            "WARN")]
        if installed["manifest_digest"] != doc["template"]["manifest_digest"]:
            return [Problem("CK10", "template", "template updated since build", "WARN")]
        return []

    @staticmethod
    def _status(problems: list[Problem]) -> CheckStatus:
        stale = {g["id"] for g in CATALOG["gates"] if g["group"] == "check" and g["exit"] == 3}
        if any(p.rule not in stale for p in problems):
            return CheckStatus.BROKEN
        return CheckStatus.STALE if problems else CheckStatus.PASS


def checks_doc(status: CheckStatus, problems, warnings) -> dict:
    """`.work/checks.json`."""
    keep = ("rule", "where", "message")
    return {"status": str(status), "problems": [{k: p.to_json()[k] for k in keep} for p in problems],
            "warnings": [{k: p.to_json()[k] for k in keep} for p in warnings],
            "checked_at": now()}


def run_check(showcase: Path, report: Path | None, template: Path | None = None) -> tuple[int, dict]:
    """Check `showcase`, write its `.work/checks.json`; (exit code, the checks document)."""
    checker = ShowcaseChecker(showcase, report, template)
    try:
        status, problems = checker.check()
    except (OSError, ValueError, RuntimeError) as exc:
        status, problems = CheckStatus.BROKEN, [Problem("CK1", str(showcase), str(exc))]
    doc = checks_doc(status, problems, checker.warnings)
    if Path(showcase).is_dir():
        doc = _recorded_check(showcase, doc) or doc
        write_json(Path(showcase) / ".work" / "checks.json", doc)
    return int(CATALOG["check_status"][str(status)]), doc


def _recorded_check(showcase, doc: Mapping) -> dict | None:
    """The check already recorded for this build, when it reads what `doc` reads (status,
    problems, warnings) and postdates the build's `created_at`: a re-check is a reading, and the
    stage brief pins `.work/checks.json` by digest — the orchestrator's own `--check` after the
    brief must not move it. Anything else (a new outcome, a rebuilt showcase) is written anew."""
    old, _ = read_json(Path(showcase) / ".work" / "checks.json", "checks.json")
    if not isinstance(old, dict) or any(old.get(k) != doc.get(k) for k in ("status", "problems", "warnings")):
        return None
    built, _ = read_json(Path(showcase) / MANIFEST, MANIFEST)
    created = str((built or {}).get("created_at") or "")
    return old if created and str(old.get("checked_at") or "") >= created else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="check a built showcase (gates CK1-CK10)")
    ap.add_argument("showcase", help="<bundle>/showcase")
    ap.add_argument("--report", help="the report's docs/ directory or its PDF")
    ap.add_argument("--template", help="the page template directory (default: installed)")
    ap.add_argument("--json", action="store_true", help="print the checks document")
    args = ap.parse_args(argv)
    code, doc = run_check(Path(args.showcase), args.report and Path(args.report),
                          args.template and Path(args.template))
    if args.json:
        print(json.dumps(doc, ensure_ascii=False))
        return code
    stale = {g["id"] for g in CATALOG["gates"] if g["group"] == "check" and g["exit"] == 3}
    for p in doc["problems"]:
        print(f"{'WARN' if p['rule'] in stale else 'ERROR'}: {p['rule']} {p['where']}: {p['message']}")
    for p in doc["warnings"]:
        print(f"WARN: {p['rule']} {p['where']}: {p['message']}")
    summary = {0: "OK", 3: "WARN"}.get(code, "ERROR")
    print(f"{summary}: check {args.showcase}: {doc['status']}"
          + (" — re-run /vega-show" if code == 3 else ""))
    return code


if __name__ == "__main__":
    sys.exit(main())
