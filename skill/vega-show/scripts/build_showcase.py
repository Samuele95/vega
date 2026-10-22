#!/usr/bin/env python3
"""
build_showcase.py — the showcase directory of one bundle: the page template copied byte for
byte, `page.json`, the story, the demonstration sample's attribution, the report or the stage
briefs, the plan and the facts, pinned by `SHOWCASE.json` (`vega/showcase@1`), built beside
the bundle's showcase and swapped into place (contract §5; build plan R-D9, R-D11, R-E1, R-E2,
R-E11).

    python3 build_showcase.py --bundle B [--model M.pack.html] [--plan P] [--story S] [--facts F]
                              [--report R] [--label-roles J] [--out B/showcase]
                              [--locale en|qps-ploc] [--unattended | --brief-links]
                              [--template DIR] [--json]
    python3 build_showcase.py --check B/showcase [--report R] [--json]

**Build.** Nothing is written before every input is read and checked: fork S1 (below); the
template (PG7: no `{{…}}` in it; its `TEMPLATE.json` identity); the bundle, the model page, the
fact set, the plan and the story (`showcase_inputs`: PF1, PF2, PF4, PF8; the plan validated
again, V1–V15, and resolved to the story handed in, rule `resolve`); PF9 (`MANIFEST.files` pins
no `showcase/` path) and PF5 (`TEMPLATE.json`'s `requires_endpoints` ⊆
`MANIFEST.serve.endpoints`); the report, or the stage briefs, still the ones the facts saw (PF6;
a brief-link showcase links one brief at least). The showcase is then written into
`showcase.building/` (its `.work/`, where the earlier steps left their markers and the draft,
is kept):

- `page/` — the template's `TEMPLATE.json` files byte for byte, nothing substituted, then
  `verify_template_copy` empty (BD1);
- `plan/facts.json`, `plan/story-plan.json`, `story.json` — the bytes read;
- `data/attribution.json` — `attribution_build.py`, run in a child process over the demo strata
  (`vega/attribution@2`, validated); without it (no demonstration sample, no inspection, an
  inspector not ready) the showcase pins none, with `WARN:`;
- `page.json` — `page_config.build_page_config` over the bundle's records, the examples, the
  demo sample's row scores (reconstruction and anomaly lanes) and the story, refused unless
  `check_page_config` (against the network view's lane) is empty (BD2);
- `report/report.pdf` (linked) or `briefs/<stage>-brief.md` (brief links, `WARN: no report …`).
  Brief links are fork S1's answer: `--unattended` applies its unattended answer, `--brief-links`
  is the user's; with neither the build stops with `FORK:` (exit 2) before writing anything;
- `SHOWCASE.json` — every file outside `.work/`, `.smoke/` and `docs/` pinned, the roles, the
  served files, the requirements, the previous showcase it supersedes, the markers; validated
  (BD6).

The staged directory is then held to BD1, BD3 (extensions), BD4 (paths, no symlink), BD5
(`served ⊆ files`, the entry a served page) and BD8 (no file of the bundle outside `showcase*/`
changed, `MANIFEST.json`'s bytes included), and swapped: `showcase/` → `showcase.prev/` (the
previous one kept, one level), `showcase.building/` → `showcase/`. The markers are appended to
`showcase/.work/markers.json`.

`--locale qps-ploc` builds the same showcase into `showcase/.work/qps/` only (BD7: nothing else
is written, the published showcase untouched, its markers only printed), its story's prose
wrapped ⟪…⟫ (D6 §10.5; `showcase_manifest.wrap_prose`): the lanes' page stage serves it in front
of the bundle's service at `/?locale=qps-ploc` (`page_smoke.py`, P-QPS), and it is never published.

**Check.** `--check <bundle>/showcase` is `showcase_check.ShowcaseChecker` (CK1–CK10): exit 0
pass, 2 broken, 3 stale.

Every refusal is `ERROR: <gate> <where>: <why>`, exit 2 (also appended to
`showcase.building/.work/markers.json` when the earlier steps keep their markers there); nothing
is swapped then: the staged directory is left for inspection, the published showcase untouched.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from typing import Mapping

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import page_checks  # noqa: E402
import page_template  # noqa: E402
import showcase_check as sc  # noqa: E402
from fact_sink import Refusal  # noqa: E402
from page_config import Catalogs, build_page_config, check_page_config, page_records  # noqa: E402
from showcase_inputs import (BuildInputs, BuildRequest, bundle_run, examples_body,  # noqa: E402
                             pinned_json, report_sources, snapshot)
from showcase_manifest import (ATTRIBUTION, BuildOptions, Fork, ShowcaseDocument,  # noqa: E402
                               changed, template_identity, tree_state, wrap_prose)
from vega_show_core import CATALOG, ShowError, ShowPaths, digest_file, read_json, thaw, write_json  # noqa: E402

__all__ = ["ShowcaseBuilder", "BuildOptions", "Fork", "main"]

SPEC = CATALOG["showcase"]
PSEUDO = CATALOG["build_locales"][1]
QPS_DIR = CATALOG["probe"]["qps_dir"].rstrip("/")
ATTRIBUTION_SCHEMA = HERE.parent / CATALOG["schemas"]["vega/attribution@2"]["file"]


def _schema_errors(doc, schema_path: Path) -> list[str]:
    from jsonschema import Draft202012Validator
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    return [f"{'/'.join(map(str, e.path)) or '(root)'}: {e.message}"
            for e in Draft202012Validator(schema).iter_errors(doc)]


def _empty(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


class ShowcaseBuilder:
    """One showcase of one bundle (see the module docstring)."""

    def __init__(self, inputs: BuildInputs, options: BuildOptions) -> None:
        self._in, self._opt = inputs, options
        self.paths = ShowPaths.of(inputs.bundle)
        self.pseudo = options.locale == PSEUDO
        self.target = self.paths.showcase / QPS_DIR if self.pseudo else self.paths.showcase
        self.stage = self.target / ".building" if self.pseudo else self.paths.building
        self.markers: list[str] = []

    # -- the build ---------------------------------------------------------------------------
    def build(self, report: Path | None) -> dict:
        """Write the staged showcase and return its `SHOWCASE.json` (nothing swapped yet)."""
        self._opt.settle_s1((thaw(self._in.facts.groups).get("report") or {}).get("status"))
        before = (snapshot(self._in.bundle), tree_state(self.paths.showcase, self.target))
        pins, requires = self._template_pins()
        placeholders, notes = self._in.validate(report, self._opt.template)
        self.markers += notes
        self._preflight(requires)
        sources = report_sources(self._in)
        self._fresh_stage()
        self._copy_template()
        self._write_documents()
        self._page(self._bundle_run())
        self._copy_report(sources)
        doc = ShowcaseDocument(self._in, self.stage, self.target).assemble(
            pins, requires, {"placeholders": placeholders, "report": sources.record,
                             "markers": self.markers})
        self.markers = list(doc["markers"])
        self._verify(doc, before)
        return doc

    def _template_pins(self) -> tuple[dict, list[str]]:
        template = self._opt.template
        try:
            leftover = page_checks.brace_tokens(template)
            if leftover:
                raise Refusal("PG7", template, "; ".join(leftover[:3]))
            pins = page_template.digests(template)
            pinned = page_template.read_manifest(template)
            requires = list((pinned or {}).get("requires_endpoints") or
                            page_template.REQUIRED_ENDPOINTS)
        except (page_template.TemplateError, OSError) as exc:
            raise Refusal("BD1", template, str(exc)) from exc
        return pins, requires

    def _preflight(self, requires: list[str]) -> None:
        manifest = self._in.manifest
        owned = sorted(r for r in manifest.get("files") or {} if str(r).startswith(SPEC["dir"] + "/"))
        if owned:
            raise Refusal("PF9", f"MANIFEST.json#files.{owned[0]}", "the bundle pins a showcase "
                          "file: a showcase file would have two owners — re-pack")
        served = (manifest.get("serve") or {}).get("endpoints") or []
        missing = [e for e in requires if e not in served]
        if missing:
            raise Refusal("PF5", "MANIFEST.json#serve.endpoints", f"the page needs {missing}, "
                          "which the bundle's service does not serve — re-pack with the v2.3 "
                          "serving templates (/vega-pack … --serve)")
        self.markers.append(f"OK: PF5 the service serves the {len(requires)} route(s) the page "
                            "needs; PF9 the bundle pins no showcase file")

    def _fresh_stage(self) -> None:
        """The staged directory, emptied but for its `.work/` (the earlier steps' markers)."""
        self.stage.mkdir(parents=True, exist_ok=True)
        for child in self.stage.iterdir():
            if not (child.name == ".work" and child.is_dir() and not child.is_symlink()):
                _empty(child)

    def _copy_template(self) -> None:
        page = self.stage / "page"
        for rel in page_template.template_files(self._opt.template):
            (page / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self._opt.template / rel, page / rel)
        problems = page_template.verify_template_copy(page, self._opt.template)
        if problems:
            raise Refusal("BD1", page, "; ".join(problems[:5]))

    def _write_documents(self) -> None:
        for rel, name in (("plan/facts.json", "facts"), ("plan/story-plan.json", "plan"),
                          ("story.json", "story")):
            (self.stage / rel).parent.mkdir(parents=True, exist_ok=True)
            body = self._in.documents[name]
            if self.pseudo and name == "story":     # D6 §10.5: the builder wraps the prose ⟪…⟫
                body = json.dumps(wrap_prose(json.loads(body)), ensure_ascii=False).encode("utf-8")
            (self.stage / rel).write_bytes(body)

    def _bundle_run(self) -> dict:
        """The child run: `data/attribution.json` written when it validates; its result."""
        run, last = bundle_run(self._in.bundle, self.stage / ".work")
        if run is None:
            self.markers.append(f"WARN: the demonstration sample could not be run through the "
                                f"bundle ({last}); no attribution, no reference scores")
            return {}
        if run.get("reference_reason"):
            self.markers.append(f"WARN: the demo sample's row scores could not be read "
                                f"({run['reference_reason']})")
        doc = run.get("attribution")
        errors = _schema_errors(doc, ATTRIBUTION_SCHEMA) if doc is not None else []
        if doc is None or errors:
            why = run.get("reason") or f"it does not validate vega/attribution@2: {errors[:3]}"
            self.markers.append(f"WARN: no attribution of the demonstration sample — {why}; the "
                                "page hides its reading on real rows")
            return run
        write_json(self.stage / ATTRIBUTION, doc)
        self.markers.append(f"OK: attribution of {sum(c['n'] for c in doc['cells'].values())} "
                            f"demo sample(s) over {len(doc['axis']['columns'])} "
                            f"{doc['axis']['kind']} column(s), target {doc['target']}")
        return run

    def _page(self, run: Mapping) -> None:
        """`page.json`, refused unless `check_page_config` holds (BD2)."""
        staged = ShowPaths(self._in.bundle, self.stage, self.stage, self.paths.previous)
        netspec = pinned_json(self._in.bundle, self._in.manifest.get("files") or {},
                              "netspec.json")
        records = page_records(self._in.store, examples_body(self._in), run.get("reference_scores"))
        catalogs = Catalogs.load()                  # with the identity of the template copied
        catalogs = replace(catalogs, template=template_identity(self._opt.template) or catalogs.template)
        try:
            page = build_page_config(staged, netspec, records, self._in.json("story"), catalogs)
        except ShowError as exc:
            raise Refusal("BD2", "page.json", str(exc)) from exc
        problems = check_page_config(page, run.get("network"), catalogs)
        if problems:
            raise Refusal("BD2", "page.json", "; ".join(problems[:5]))
        write_json(self.stage / "page.json", page)

    def _copy_report(self, sources) -> None:
        for rel, source in sources.copies.items():
            (self.stage / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, self.stage / rel)
        if sources.record["status"] == "brief-links":
            self.markers += [CATALOG["forks"]["S1"]["marker"],
                             self._opt.settle_s1(sources.record["status"])]
        else:
            roles = sources.record["roles"]
            self.markers.append(f"OK: report linked — {len(roles['bound'])} role(s) bound, "
                                f"{len(roles['missing'])} missing ({roles['origin']})")

    # -- the staged showcase, checked, and swapped -------------------------------------------
    def _verify(self, doc: Mapping, before: tuple) -> None:
        """BD3, BD4, BD5 on the staged directory; BD8 and BD7 on what the build left alone."""
        files = doc["files"]
        found = [*sc.files_problems(files, ("BD4", "BD3")),
                 *sc.tree_problems(self.stage, files, ("BD4", "BD3")),
                 *sc.served_problems(doc, "BD5"), *sc.digest_problems(self.stage, files, "BD4")]
        if found:
            raise Refusal(found[0].rule, found[0].where, "; ".join(
                f"{p.rule} {p.where}: {p.message}" for p in found[:5]))
        moved = changed(before[0], snapshot(self._in.bundle))
        if moved or digest_file(self._in.bundle / "MANIFEST.json") != \
                self._in.store.manifest_digest():
            raise Refusal("BD8", self._in.bundle, f"the build changed the bundle outside "
                          f"showcase*/: {moved[:5] or ['MANIFEST.json']}")
        moved = changed(before[1], tree_state(self.paths.showcase, self.target))
        if self.pseudo and moved:
            raise Refusal("BD7", self.paths.showcase, f"the {PSEUDO} build wrote outside "
                          f"{QPS_DIR}/: {moved[:5]}")

    def swap(self) -> None:
        """`showcase/` → `showcase.prev/`, `showcase.building/` → `showcase/` (a pseudo-locale
        build replaces what `.work/qps/` held and keeps no previous one)."""
        if self.pseudo:
            for child in self.target.iterdir():
                if child != self.stage:
                    _empty(child)
            for child in self.stage.iterdir():
                child.rename(self.target / child.name)
            self.stage.rmdir()
            return
        if self.paths.previous.exists():
            shutil.rmtree(self.paths.previous)
        if self.target.exists():
            self.target.rename(self.paths.previous)
        try:
            self.stage.rename(self.target)
        except OSError:
            if self.paths.previous.exists() and not self.target.exists():
                self.paths.previous.rename(self.target)
            raise


# --------------------------------------------------------------------------- the command line

def _parse(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    ap.add_argument("--check", metavar="SHOWCASE", help="check a built showcase (CK1-CK10)")
    ap.add_argument("--bundle", help="the bundle directory")
    ap.add_argument("--model", help="the model page <stem>.pack.html (default: the facts')")
    ap.add_argument("--plan", help="the authored story plan (default: showcase.building/plan/)")
    ap.add_argument("--story", help="the resolved story (default: showcase.building/story.json)")
    ap.add_argument("--facts", help="the fact set (default: showcase.building/plan/facts.json)")
    ap.add_argument("--report", help="the report's docs/ directory, .aux or PDF")
    ap.add_argument("--label-roles", help="the label-role map (its directory is the report's)")
    ap.add_argument("--out", help="the showcase directory: <bundle>/showcase")
    ap.add_argument("--locale", default="en", help="en, or qps-ploc (into showcase/.work/qps/)")
    ap.add_argument("--unattended", action="store_true", help="fork S1 answered unattended")
    ap.add_argument("--brief-links", action="store_true",
                    help="fork S1 answered: no report, the code notes link the stage briefs")
    ap.add_argument("--template", help="the page template directory (default: installed)")
    ap.add_argument("--json", action="store_true", help="print a machine result")
    args = ap.parse_args(argv)
    if not args.check and not args.bundle:
        ap.error("--bundle is required (or --check SHOWCASE)")
    return args


def _report_of(args) -> Path | None:
    if args.report:
        return Path(args.report)
    return Path(args.label_roles).parent if args.label_roles else None


def _build(args) -> tuple[list[str], dict]:
    report = _report_of(args)
    request = BuildRequest(Path(args.bundle), args.model and Path(args.model),
                           args.plan and Path(args.plan), args.story and Path(args.story),
                           args.facts and Path(args.facts), report)
    options = BuildOptions(args.locale, (args.unattended, args.brief_links),
                           args.template and Path(args.template))
    inputs = BuildInputs.load(request)
    builder = ShowcaseBuilder(inputs, options)
    if args.out and Path(args.out).resolve() != builder.paths.showcase.resolve():
        raise Refusal("BD8", args.out, f"the showcase is {builder.paths.showcase}: the slot "
                      "serves <bundle>/showcase, and nothing is written elsewhere")
    doc = builder.build(report)
    kept = builder.target.exists() and not builder.pseudo
    builder.swap()
    lines = [*builder.markers, f"OK: {builder.target} swapped into place"
             + (" (pseudo-locale: never published)" if builder.pseudo else
                f", the previous one kept as {builder.paths.previous.name}/" if kept else "")]
    if not builder.pseudo:
        _append_markers(builder.paths.markers, lines)
    return lines, doc


def _refusal_markers(args, lines: list[str]) -> None:
    """A refused build's lines, appended to the staged showcase's markers when the stage's
    earlier steps keep them there (`showcase.building/.work/`); a pseudo-locale build writes
    nothing outside `.work/qps/`."""
    work = ShowPaths.of(Path(args.bundle)).building / ".work"
    if args.locale != PSEUDO and work.is_dir() and not work.is_symlink():
        _append_markers(work / "markers.json", lines)


def _append_markers(path: Path, lines: list[str]) -> None:
    doc, _ = read_json(path, "the markers file")
    if doc is not None and not isinstance(doc, list):
        lines.append(f"WARN: the markers file {path} is not a JSON list; left as it is")
        return
    write_json(path, (doc or []) + [f"build_showcase.py: {m}" for m in lines])


def main(argv=None) -> int:
    args = _parse(argv)
    if args.check:
        return sc.main([args.check, *(["--report", args.report] if args.report else []),
                        *(["--template", args.template] if args.template else []),
                        *(["--json"] if args.json else [])])
    try:
        lines, doc, code = *_build(args), 0
    except Fork as exc:
        lines, doc, code = [f"FORK: {exc}"], None, 2
    except (RuntimeError, OSError, ValueError) as exc:     # ShowError is a RuntimeError
        lines, doc, code = [f"ERROR: {exc}"], None, 2
        _refusal_markers(args, lines)
    if args.json:
        print(json.dumps({"status": "built" if code == 0 else "refused", "markers": lines,
                          "files": len(doc["files"]) if doc else 0}, ensure_ascii=False))
    else:
        print("\n".join(lines))
    return code


if __name__ == "__main__":
    sys.exit(main())
